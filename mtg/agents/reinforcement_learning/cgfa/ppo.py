"""CGFA-PPO algorithm: MaskablePPO + per-factor GAE + residual gating.

Algorithmic differences from :class:`MaskablePPO`:

1. **Rollout buffer** is :class:`CGFAMaskableRolloutBuffer`, which
   carries an extra ``(buffer_size, n_envs, K)`` tensor of per-factor
   rewards, per-factor value-head predictions, and SCM-predicted
   per-factor changes (``eps``).

2. **Rollout collection** writes ``info["factor_rewards"]`` and
   ``info["factor_eps"]`` (produced by :class:`CGFAEnvWrapper`) into
   the buffer, along with the per-factor values returned by the
   policy.  The per-state residual gate ``g(s)`` is logged for
   diagnostics.

3. **Train loop** computes per-factor GAE advantages on top of the
   scalar advantages and blends them via the (state-conditional)
   residual gate ``g(s)``:

       A_used(s) = (1 - g(s)) * A_scalar(s) + g(s) * sum_k w_k * A_k(s)

   When ``learnable_gate`` is False this collapses back to a constant
   blend at ``cgfa_alpha``.  Per-factor value heads
   are trained against per-factor returns via additional MSE losses.
   The intervention-calibration loss (Pearson maximisation between
   ``A_k`` and SCM-predicted ``eps_k``) pushes per-factor advantages
   to align with the structural prior.

4. **Logging** records per-factor:

   * advantage / return statistics,
   * current mixture weights ``w_k``,
   * residual gate ``g(s)`` mean/std/min/max,
   * Pearson correlation between ``A_k`` and ``eps_k``,
   * sign-agreement rate between ``A_k`` and ``eps_k``,
   * per-factor *credit share* ``|w_k * A_k| / sum_j |w_j * A_j|``,

   so the contribution of each causal factor to learning can be
   inspected step-by-step.
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar, TypeVar

import numpy as np
import torch as th
import torch.nn.functional as functional
from gymnasium import spaces
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from mtg.agents.reinforcement_learning.cgfa.buffer import (
    CGFAMaskableRolloutBuffer,
    CGFAMaskableRolloutBufferSamples,
)
from mtg.agents.reinforcement_learning.cgfa.factor_spec import FactorSpec
from mtg.agents.reinforcement_learning.cgfa.policy import (
    CGFAMaskablePolicy,
    make_cgfa_policy_class,
)

SelfCGFAMaskablePPO = TypeVar("SelfCGFAMaskablePPO", bound="CGFAMaskablePPO")


class CGFAMaskablePPO(MaskablePPO):
    """MaskablePPO variant with Causal Graph-Factored Advantages."""

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = dict(MaskablePPO.policy_aliases)
    policy: CGFAMaskablePolicy  # type: ignore[assignment]
    rollout_buffer: CGFAMaskableRolloutBuffer  # type: ignore[assignment]

    def __init__(
        self,
        policy: str | type[CGFAMaskablePolicy],
        env: GymEnv | str,
        factor_spec: FactorSpec | None = None,
        cgfa_alpha: float = 0.5,
        factor_value_coef: float = 0.5,
        intervention_calibration_coef: float = 0.0,
        gate_entropy_coef: float = 0.0,
        learnable_gate: bool = True,
        state_conditional_gate: bool = True,
        gate_hidden_dim: int = 32,
        learning_rate: float | Schedule = 3e-4,
        n_steps: int = 2048,
        batch_size: int | None = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float | Schedule = 0.2,
        clip_range_vf: None | float | Schedule = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rollout_buffer_class: type[RolloutBuffer] | None = None,
        rollout_buffer_kwargs: dict[str, Any] | None = None,
        target_kl: float | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        self.factor_spec = factor_spec or FactorSpec()
        self.cgfa_alpha = float(cgfa_alpha)
        self.factor_value_coef = float(factor_value_coef)
        self.intervention_calibration_coef = float(intervention_calibration_coef)
        self.gate_entropy_coef = float(gate_entropy_coef)
        self.learnable_gate = bool(learnable_gate)
        self.state_conditional_gate = bool(state_conditional_gate)
        self.gate_hidden_dim = int(gate_hidden_dim)
        self._diag_gate_means: list[float] = []
        # Tracks (key, reason) pairs for which we already issued a
        # warning, so a single missing-info bug does not flood the log
        # but still surfaces once.
        self._factor_collect_warned: set[tuple[str, str]] = set()

        if not 0.0 <= self.cgfa_alpha <= 1.0:
            raise ValueError(f"cgfa_alpha must be in [0, 1], got {self.cgfa_alpha}")

        # The policy carries the gate; forward our gate config so the
        # head is built consistently regardless of how the policy was
        # specified.
        policy_kwargs = dict(policy_kwargs or {})
        policy_kwargs.setdefault("residual_gate_init", self.cgfa_alpha)
        policy_kwargs.setdefault("state_conditional_gate", self.state_conditional_gate)
        policy_kwargs.setdefault("gate_hidden_dim", self.gate_hidden_dim)

        if isinstance(policy, str):
            # Bind a CGFA policy subclass with the factor spec baked in
            # so str-based policy specification keeps working.
            policy = make_cgfa_policy_class(self.factor_spec)
        elif isinstance(policy, type) and issubclass(policy, CGFAMaskablePolicy):
            # Make sure the spec we hold is forwarded into the policy.
            policy_kwargs.setdefault("factor_spec", self.factor_spec)

        # Force the buffer class even if the caller does not specify it.
        if rollout_buffer_class is None:
            rollout_buffer_class = CGFAMaskableRolloutBuffer
        rollout_buffer_kwargs = dict(rollout_buffer_kwargs or {})
        rollout_buffer_kwargs.setdefault("n_factors", self.factor_spec.n_factors)

        super().__init__(
            policy=policy,  # type: ignore[arg-type]
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollouts(  # type: ignore[override]
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:
        """Collect a rollout, attaching per-factor signals into the buffer."""
        assert isinstance(
            rollout_buffer, CGFAMaskableRolloutBuffer
        ), "CGFAMaskablePPO requires a CGFAMaskableRolloutBuffer"
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)
        n_steps = 0
        action_masks = None
        rollout_buffer.reset()

        if use_masking and not is_masking_supported(env):
            raise ValueError(
                "Environment does not support action masking. Consider using ActionMasker wrapper"
            )

        callback.on_rollout_start()
        self._diag_gate_means = []

        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                if use_masking:
                    action_masks = get_action_masks(env)
                actions, values, log_probs = self.policy(obs_tensor, action_masks=action_masks)
                v_factors = self.policy.consume_cached_factor_values()
                gate_step = self.policy.consume_cached_gate()
            if gate_step is not None:
                self._diag_gate_means.append(float(gate_step.mean().item()))

            actions_np = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions_np)

            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions_np = actions_np.reshape(-1, 1)

            factor_rewards = self._collect_factor_array(infos, "factor_rewards")
            factor_eps = self._collect_factor_array(infos, "factor_eps")

            # Bootstrap on truncation, like the parent class does. The
            # SAME correction is applied symmetrically to the scalar
            # reward AND every per-factor reward channel; otherwise
            # scalar GAE and per-factor GAE see different MDPs at every
            # truncation boundary (~30% of episodes at max_turns=10).
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                        _, terminal_v_factors = self.policy.predict_factor_values(terminal_obs)
                    rewards[idx] += self.gamma * terminal_value
                    factor_rewards[idx] += self.gamma * terminal_v_factors[0].cpu().numpy()

            rollout_buffer.add(
                self._last_obs,
                actions_np,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                action_masks=action_masks,
                factor_rewards=factor_rewards,
                factor_values=v_factors if v_factors is not None else None,
                factor_eps=factor_eps,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            obs_t = obs_as_tensor(new_obs, self.device)
            values = self.policy.predict_values(obs_t)
            _, last_v_factors = self.policy.predict_factor_values(obs_t)
            last_v_factors_np = last_v_factors.cpu().numpy()

        rollout_buffer.compute_returns_and_advantage(
            last_values=values,
            dones=dones,
            last_factor_values=last_v_factors_np,
        )

        callback.on_rollout_end()
        return True

    def _collect_factor_array(
        self,
        infos: list[dict[str, Any]],
        key: str,
    ) -> np.ndarray:
        """Stack the per-env per-factor arrays from ``info`` into ``(N, K)``.

        Silently filling in zeros when ``key`` is missing or has the
        wrong shape would corrupt every downstream CGFA signal (per-
        factor GAE, calibration target, gate logging) without any
        visible failure. We instead emit a single guarded warning the
        first time it happens per (key, reason) tuple and zero-fill the
        slot, so wiring bugs are loud but a single bad step from a
        third-party wrapper does not crash the whole rollout.
        """
        n_envs = len(infos)
        out = np.zeros((n_envs, self.factor_spec.n_factors), dtype=np.float32)
        warned: set[tuple[str, str]] = self._factor_collect_warned
        for i, info in enumerate(infos):
            v = info.get(key)
            if v is None:
                signature = (key, "missing")
                if signature not in warned:
                    warned.add(signature)
                    warnings.warn(
                        f"CGFA: env {i} did not publish info[{key!r}]; "
                        "zero-filling. Check that CGFAEnvWrapper is the "
                        "innermost wrapper around the base MTG env.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                continue
            arr = np.asarray(v, dtype=np.float32).reshape(-1)
            if arr.shape[0] != self.factor_spec.n_factors:
                signature = (key, f"shape={arr.shape[0]}")
                if signature not in warned:
                    warned.add(signature)
                    warnings.warn(
                        f"CGFA: info[{key!r}] has shape {arr.shape[0]} but "
                        f"FactorSpec.n_factors={self.factor_spec.n_factors}; "
                        "zero-filling. The wrapper and the agent likely use "
                        "mismatched FactorSpec instances.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                continue
            out[i] = arr
        return out

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> None:  # type: ignore[override]
        """Run PPO updates with per-factor losses and CGFA-blended advantages."""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(  # type: ignore[operator]
                self._current_progress_remaining
            )
        else:
            clip_range_vf = None

        entropy_losses: list[float] = []
        pg_losses: list[float] = []
        value_losses: list[float] = []
        factor_value_losses: list[float] = []
        calibration_losses: list[float] = []
        clip_fractions: list[float] = []
        approx_kl_divs: list[float] = []
        gate_means: list[float] = []
        gate_stds: list[float] = []
        gate_mins: list[float] = []
        gate_maxs: list[float] = []
        last_loss: th.Tensor | None = None

        continue_training = True

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                assert isinstance(rollout_data, CGFAMaskableRolloutBufferSamples)
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy, v_factors_pred, gate = (
                    self.policy.evaluate_actions_factor(
                        rollout_data.observations,
                        actions,
                        action_masks=rollout_data.action_masks,
                    )
                )
                values = values.flatten()

                # CGFA advantage blend.
                # When learnable_gate is True we use the (potentially
                # state-conditional) residual gate g(s) to mix the
                # scalar advantage with the per-factor advantage.  The
                # gradient flows back into the gate so the policy can
                # learn how much to trust the SCM-aligned signal in
                # each state.  When False we collapse to a constant
                # blend at ``cgfa_alpha``.
                a_scalar = rollout_data.advantages
                a_factor = rollout_data.factor_advantages  # (B, K)
                weights = self.policy.cgfa_head.mixture_weights  # (K,)
                a_factor_blend = (a_factor * weights).sum(dim=-1)
                if self.learnable_gate:
                    advantages = (1.0 - gate) * a_scalar + gate * a_factor_blend
                else:
                    advantages = (
                        1.0 - self.cgfa_alpha
                    ) * a_scalar + self.cgfa_alpha * a_factor_blend

                # Track gate statistics across mini-batches.
                with th.no_grad():
                    gate_means.append(float(gate.mean().item()))
                    gate_stds.append(float(gate.std().item()))
                    gate_mins.append(float(gate.min().item()))
                    gate_maxs.append(float(gate.max().item()))

                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # PPO clipped surrogate
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1.0) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # Scalar value loss
                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = functional.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Per-factor value loss: MSE per factor, summed
                factor_value_loss = functional.mse_loss(v_factors_pred, rollout_data.factor_returns)
                factor_value_losses.append(factor_value_loss.item())

                # Intervention-calibration loss: Pearson correlation
                # maximisation between A_k and SCM-predicted eps_k per
                # factor. The loss must consume live per-factor
                # advantages computed from the current per-factor value
                # head, not ``rollout_data.factor_advantages`` (which
                # is loaded from the rollout buffer as a constant
                # tensor with no autograd graph). Routing through
                # ``A_k_live = factor_returns - V_k_pred`` gives the
                # loss a real gradient path into V_k. The SCM target
                # ``factor_eps`` is a target signal and is detached
                # defensively.
                if self.intervention_calibration_coef > 0.0:
                    a_factor_live = rollout_data.factor_returns - v_factors_pred
                    cal_loss = self._calibration_loss(
                        a_factor_live,
                        rollout_data.factor_eps.detach(),
                    )
                    calibration_losses.append(cal_loss.item())
                else:
                    cal_loss = th.tensor(0.0, device=values.device)
                    calibration_losses.append(0.0)

                entropy_loss = -th.mean(-log_prob) if entropy is None else -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # Gate entropy regulariser: keeps the residual gate g(s)
                # from collapsing to {0, 1}.  Maximising binary entropy of
                # the per-state gate (subtracting a *negative* term from
                # ``loss``) encourages a non-degenerate mixture; coef 0.0
                # disables for ablation parity.
                if self.learnable_gate and self.gate_entropy_coef > 0.0 and gate is not None:
                    g = gate.clamp(1e-6, 1.0 - 1e-6)
                    gate_entropy = -(g * th.log(g) + (1.0 - g) * th.log(1.0 - g)).mean()
                    gate_entropy_term = -self.gate_entropy_coef * gate_entropy
                else:
                    gate_entropy_term = th.tensor(0.0, device=values.device)

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.factor_value_coef * factor_value_loss
                    + self.intervention_calibration_coef * cal_loss
                    + gate_entropy_term
                )

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1.0) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at epoch {epoch} due to "
                            f"reaching max KL: {approx_kl_div:.2f}"
                        )
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                last_loss = loss

            if not continue_training:
                break

        self._n_updates += self.n_epochs

        # ---------------- Logging ----------------
        from stable_baselines3.common.utils import explained_variance

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )
        self.logger.record("train/entropy_loss", float(np.mean(entropy_losses)))
        self.logger.record("train/policy_gradient_loss", float(np.mean(pg_losses)))
        self.logger.record("train/value_loss", float(np.mean(value_losses)))
        self.logger.record("train/factor_value_loss", float(np.mean(factor_value_losses)))
        self.logger.record("train/calibration_loss", float(np.mean(calibration_losses)))
        self.logger.record("train/approx_kl", float(np.mean(approx_kl_divs)))
        self.logger.record("train/clip_fraction", float(np.mean(clip_fractions)))
        if last_loss is not None:
            self.logger.record("train/loss", float(last_loss.item()))
        self.logger.record("train/explained_variance", float(explained_var))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", float(clip_range))
        if clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", float(clip_range_vf))
        self.logger.record("cgfa/alpha", float(self.cgfa_alpha))
        self.logger.record("cgfa/learnable_gate", float(self.learnable_gate))
        self.logger.record("cgfa/state_conditional_gate", float(self.state_conditional_gate))
        self.logger.record("cgfa/gate_entropy_coef", float(self.gate_entropy_coef))

        # Residual gate diagnostics (training mini-batches).
        if gate_means:
            self.logger.record("cgfa/gate/mean", float(np.mean(gate_means)))
            self.logger.record("cgfa/gate/std", float(np.mean(gate_stds)))
            self.logger.record("cgfa/gate/min", float(np.min(gate_mins)))
            self.logger.record("cgfa/gate/max", float(np.max(gate_maxs)))

        # Rollout-time gate (during data collection, off-policy view).
        if self._diag_gate_means:
            self.logger.record("cgfa/gate/rollout_mean", float(np.mean(self._diag_gate_means)))

        weights_np = self.policy.cgfa_head.mixture_weights.detach().cpu().numpy()
        for name, w in zip(self.factor_spec.names, weights_np, strict=False):
            self.logger.record(f"cgfa/blend/{name}", float(w))

        # ----- Per-factor calibration + credit-share diagnostics -----
        adv_buf = self.rollout_buffer.factor_advantages.reshape(-1, self.factor_spec.n_factors)
        eps_buf = self.rollout_buffer.factor_eps.reshape(-1, self.factor_spec.n_factors)
        # Absolute weighted advantage per factor over the whole rollout,
        # used as a credit-assignment signal that exposes which causal
        # variables are actually shaping the policy gradient.
        contributions: list[float] = []
        for k, name in enumerate(self.factor_spec.names):
            a_k = adv_buf[:, k]
            e_k = eps_buf[:, k]
            ret_k = self.rollout_buffer.factor_returns[..., k]

            self.logger.record(f"cgfa/factor_adv/{name}/mean", float(a_k.mean()))
            self.logger.record(f"cgfa/factor_adv/{name}/std", float(a_k.std()))
            self.logger.record(f"cgfa/factor_ret/{name}/mean", float(ret_k.mean()))

            if a_k.std() > 1e-8 and e_k.std() > 1e-8:
                corr_k = float(np.corrcoef(a_k, e_k)[0, 1])
            else:
                corr_k = 0.0
            self.logger.record(f"cgfa/factor_corr/{name}", corr_k)

            sign_a = np.sign(a_k)
            sign_e = np.sign(e_k)
            both_nonzero = (sign_a != 0) & (sign_e != 0)
            if both_nonzero.any():
                sign_agree = float((sign_a[both_nonzero] == sign_e[both_nonzero]).mean())
            else:
                sign_agree = 0.0
            self.logger.record(f"cgfa/factor_sign_agree/{name}", sign_agree)

            contribution_k = float(np.abs(weights_np[k] * a_k).mean())
            contributions.append(contribution_k)
            self.logger.record(f"cgfa/factor_contribution/{name}", contribution_k)

        total = float(sum(contributions))
        for name, c_k in zip(self.factor_spec.names, contributions, strict=False):
            share = (c_k / total) if total > 1e-8 else 0.0
            self.logger.record(f"cgfa/factor_share/{name}", share)

    # ------------------------------------------------------------------
    # Auxiliary: intervention calibration loss
    # ------------------------------------------------------------------

    def _calibration_loss(
        self,
        factor_advantages: th.Tensor,
        factor_eps: th.Tensor,
        min_std: float = 1e-3,
    ) -> th.Tensor:
        """Pearson-correlation-based loss between A_k and SCM eps_k.

        For each of the K factors we compute the negative of the
        Pearson correlation across the minibatch and average.  This
        encourages per-factor advantages to point in the same
        direction as the SCM-predicted change for the action that was
        taken, anchoring CGFA to the structural prior without
        dictating the magnitude.

        Numerics: the per-factor std in the denominator is clamped to
        ``min_std`` (default 1e-3 in the *normalised factor space* used
        by the wrapper), which is small enough to leave a real gradient
        when both signals have variance and large enough to keep the
        loss bounded if a factor goes flat for an entire minibatch
        (otherwise Pearson explodes from 1/eps amplification).  Per-
        factor terms whose denominator hits the floor are zero-weighted
        in the mean so a single dead factor cannot dominate the signal.
        """
        eps = 1e-8
        a = factor_advantages
        b = factor_eps
        a_centered = a - a.mean(dim=0, keepdim=True)
        b_centered = b - b.mean(dim=0, keepdim=True)
        # Use population std (correction=0) so the resulting Pearson
        # correlation is exactly bounded in [-1, 1] at finite batch
        # sizes (Bessel-corrected std would multiply by (N-1)/N).
        a_std = a_centered.std(dim=0, keepdim=True, correction=0)
        b_std = b_centered.std(dim=0, keepdim=True, correction=0)
        a_norm = a_centered / (a_std.clamp(min=min_std) + eps)
        b_norm = b_centered / (b_std.clamp(min=min_std) + eps)
        corr = (a_norm * b_norm).mean(dim=0)  # (K,)
        # Mask factors with degenerate std (no real signal this batch)
        # so they do not contribute (or dominate) the gradient.
        valid = ((a_std.squeeze(0) > min_std) & (b_std.squeeze(0) > min_std)).float()
        if valid.sum() < 1.0:
            return th.zeros((), device=corr.device)
        return -(corr * valid).sum() / valid.sum()

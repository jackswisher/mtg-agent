"""CGFA-PPO agent: PPO with Causal Graph-Factored Advantages.

This is the agent-level wrapper around :class:`CGFAMaskablePPO` that
slots into the canonical :class:`mtg.training.train.Trainer`.  It
mirrors :class:`mtg.agents.reinforcement_learning.ppo_agent.PPOAgent`
in structure (so we get the same LR / entropy / clip-range schedules
and SB3 lifecycle) but swaps in:

* :class:`CGFAMaskablePPO` as the SB3 algorithm.
* A :class:`CGFAMaskablePolicy` initialised with the agent's
  :class:`FactorSpec` and residual-gate configuration.

The trainer is responsible for wrapping each rollout env with
:class:`CGFAEnvWrapper` *before* the ``ActionMasker`` so that the
``info`` dict contains the per-factor signals
(``factor_values`` / ``factor_rewards`` / ``factor_eps``) that the
algorithm reads from the rollout buffer.

This is the entry point used by the Trainer for end-to-end CGFA-PPO
runs and by the research pipeline (ablation suite, transfer
experiment, intervention-calibration plots, case study).
"""

from __future__ import annotations

import dataclasses
import typing as tp
from pathlib import Path

import numpy as np
import torch

from mtg.agents.reinforcement_learning.cgfa import (
    CGFAEnvWrapper,
    CGFAMaskablePPO,
    FactorSpec,
    factor_blend_from_scm_weights,
    make_cgfa_policy_class,
)
from mtg.agents.reinforcement_learning.ppo_agent import HAS_SB3, PPOAgent, _mask_fn
from mtg.causal.scm import SCMWeights, StructuralCausalModel


class CGFAAgent(PPOAgent):
    """PPO agent with Causal Graph-Factored Advantages.

    Inherits the LR / clip / entropy scheduling machinery from
    :class:`PPOAgent` and overrides ``initialize_model`` to construct
    a :class:`CGFAMaskablePPO` instead of plain ``MaskablePPO``.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        # CGFA-specific knobs ----------------------------------------------
        factor_spec: FactorSpec | None = None,
        cgfa_alpha: float = 0.5,
        learnable_gate: bool = True,
        state_conditional_gate: bool = True,
        gate_hidden_dim: int = 32,
        factor_value_coef: float = 0.5,
        intervention_calibration_coef: float = 0.1,
        gate_entropy_coef: float = 0.0,
        init_blend_from_scm: bool = True,
        calibration_mode: str = "factual",
        scm: StructuralCausalModel | None = None,
        # Inherited from PPOAgent ------------------------------------------
        learning_rate: float = 3e-4,
        lr_end: float = 1e-5,
        batch_size: int = 256,
        n_steps: int = 2048,
        gamma: float = 0.995,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_end: float | None = None,
        clip_range_vf: float | None = 0.2,
        ent_coef: float = 0.05,
        ent_coef_end: float = 0.005,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        target_kl: float | None = 0.03,
        vf_coef: float = 0.5,
        use_set_encoder: bool = False,
        features_dim: int = 512,
        seed: int | None = None,
    ) -> None:
        """Initialize the CGFA agent.

        Args:
            observation_dim: Dimension of observation space.
            action_dim: Number of discrete actions.
            factor_spec: K-factor spec.  Defaults to the canonical 6
                SCM ``win_prob`` parents.
            cgfa_alpha: Initial value of the residual gate ``g(s)`` used
                to blend the scalar advantage with the per-factor
                advantage.  Also used as the constant blend when
                ``learnable_gate`` is False.
            learnable_gate: When True the residual gate is learnable
                and its gradient flows back into the policy update.
                When False the blend collapses to a constant
                ``cgfa_alpha``.
            state_conditional_gate: When True the gate is produced by a
                small MLP from the critic latent (per-state).  When
                False the gate is a single learnable scalar.
            gate_hidden_dim: Hidden width of the gate MLP.
            factor_value_coef: Loss weight for the per-factor value
                heads (analogous to PPO's ``vf_coef``).
            intervention_calibration_coef: Loss weight for the
                Pearson-correlation calibration auxiliary that aligns
                per-factor advantages with SCM-predicted per-factor
                changes.  Set to ``0.0`` to disable.
            gate_entropy_coef: Weight on the binary-entropy regulariser
                applied to the per-state residual gate ``g(s)``.
                Encourages a non-degenerate mixture (gate not collapsed
                to ``{0, 1}``). Defaults to ``0.0`` (disabled).
            init_blend_from_scm: When True the per-factor mixture
                weights ``w_k`` are initialised from the SCM's
                ``win_prob`` weights so CGFA starts pre-loaded with
                the structural prior.
            calibration_mode: ``"factual"`` uses the historical SCM
                consistency delta between consecutive states.
                ``"interventional"`` uses an experimental action-metadata
                intervention target when one can be mapped cleanly.
            scm: Optional :class:`StructuralCausalModel` instance.
                Used by :class:`CGFAEnvWrapper` to compute
                SCM-predicted per-factor changes for the calibration
                loss.  If None, a fresh default SCM is constructed
                (only used to initialise blend weights).
            learning_rate: Initial (peak) learning rate.
            lr_end: Final learning rate after linear annealing.
            batch_size: Minibatch size.
            n_steps: Steps per rollout buffer.
            gamma: Discount factor.
            gae_lambda: GAE lambda.
            clip_range: PPO clip range.
            clip_range_end: Final clip range after linear annealing.
            clip_range_vf: Value function clipping range.
            ent_coef: Initial entropy coefficient.
            ent_coef_end: Final entropy coefficient after annealing.
            max_grad_norm: Max gradient norm for clipping.
            n_epochs: Number of PPO epochs per rollout.
            target_kl: Early-stop threshold on the approximate KL.
            vf_coef: Scalar value function loss coefficient.
            use_set_encoder: Use the permutation-invariant set encoder.
            features_dim: Output dim of the set encoder.
            seed: Random seed.
        """
        super().__init__(
            observation_dim=observation_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            lr_end=lr_end,
            batch_size=batch_size,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_end=clip_range_end,
            clip_range_vf=clip_range_vf,
            ent_coef=ent_coef,
            ent_coef_end=ent_coef_end,
            max_grad_norm=max_grad_norm,
            n_epochs=n_epochs,
            target_kl=target_kl,
            vf_coef=vf_coef,
            use_set_encoder=use_set_encoder,
            features_dim=features_dim,
            seed=seed,
        )
        self.name = "CGFAAgent"
        self.scm = scm or StructuralCausalModel()
        base_spec = factor_spec or FactorSpec()
        if init_blend_from_scm:
            blend = factor_blend_from_scm_weights(
                self.scm.weights if hasattr(self.scm, "weights") else SCMWeights(),
                names=base_spec.names,
            )
            self.factor_spec = dataclasses.replace(base_spec, blend_init=blend)
        else:
            self.factor_spec = base_spec
        self.cgfa_alpha = float(cgfa_alpha)
        self.learnable_gate = bool(learnable_gate)
        self.state_conditional_gate = bool(state_conditional_gate)
        self.gate_hidden_dim = int(gate_hidden_dim)
        self.factor_value_coef = float(factor_value_coef)
        self.intervention_calibration_coef = float(intervention_calibration_coef)
        self.gate_entropy_coef = float(gate_entropy_coef)
        if calibration_mode not in {"factual", "interventional"}:
            raise ValueError(
                "CGFAAgent calibration_mode must be 'factual' or "
                f"'interventional', got {calibration_mode!r}."
            )
        self.calibration_mode = calibration_mode

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def initialize_model(self, env: tp.Any) -> None:
        """Initialize a :class:`CGFAMaskablePPO` model with ``env``.

        The trainer should wrap each rollout env with
        :class:`CGFAEnvWrapper` *before* :class:`ActionMasker` so that
        the ``info`` dict contains the per-factor signals required by
        the algorithm.  When the env is not pre-wrapped, this method
        wraps it on the fly (single-env path).
        """
        if not HAS_SB3:
            return

        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.vec_env import VecEnv

        if isinstance(env, VecEnv):
            wrapped_env = env
        else:
            # Single-env path.  Wrap CGFA before ActionMasker so the
            # per-factor info dict is built on the inner step() and
            # forwarded transparently through ActionMasker.
            cgfa_env = CGFAEnvWrapper(
                env,
                factor_spec=self.factor_spec,
                scm=self.scm,
                calibration_mode=self.calibration_mode,
            )
            wrapped_env = ActionMasker(cgfa_env, _mask_fn)

        policy_kwargs: dict[str, tp.Any] = {
            "net_arch": {"pi": [512, 256], "vf": [512, 256]},
            "activation_fn": torch.nn.ReLU,
            "factor_spec": self.factor_spec,
            "residual_gate_init": self.cgfa_alpha,
            "state_conditional_gate": self.state_conditional_gate,
            "gate_hidden_dim": self.gate_hidden_dim,
        }
        if self.use_set_encoder:
            from mtg.agents.reinforcement_learning.features import MTGFeaturesExtractor

            policy_kwargs["features_extractor_class"] = MTGFeaturesExtractor
            policy_kwargs["features_extractor_kwargs"] = {
                "features_dim": self.features_dim,
            }
            policy_kwargs["net_arch"] = {"pi": [256, 128], "vf": [256, 128]}

        ppo_kwargs: dict[str, tp.Any] = {
            "learning_rate": self._make_lr_schedule(),
            "batch_size": self.batch_size,
            "n_steps": self.n_steps,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self._make_clip_range_schedule(),
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "n_epochs": self.n_epochs,
            "target_kl": self.target_kl,
            "normalize_advantage": True,
            "seed": self.seed,
            "verbose": 0,
            "policy_kwargs": policy_kwargs,
            # CGFA-specific knobs
            "factor_spec": self.factor_spec,
            "cgfa_alpha": self.cgfa_alpha,
            "factor_value_coef": self.factor_value_coef,
            "intervention_calibration_coef": self.intervention_calibration_coef,
            "gate_entropy_coef": self.gate_entropy_coef,
            "learnable_gate": self.learnable_gate,
            "state_conditional_gate": self.state_conditional_gate,
            "gate_hidden_dim": self.gate_hidden_dim,
        }
        clip_vf = self._make_clip_range_vf_schedule()
        if clip_vf is not None:
            ppo_kwargs["clip_range_vf"] = clip_vf

        policy_cls = make_cgfa_policy_class(self.factor_spec)
        self.model = CGFAMaskablePPO(policy_cls, wrapped_env, **ppo_kwargs)
        self.model._ppo_agent_ref = self

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------

    def load(self, path: str | Path) -> None:
        """Load a checkpoint produced by ``save``."""
        if HAS_SB3:
            self.model = CGFAMaskablePPO.load(str(path))
            self.model._ppo_agent_ref = self

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        info: dict[str, tp.Any] | None = None,
    ) -> int:
        """Select action using the learned policy."""
        if self.model is None or not HAS_SB3:
            legal = np.where(action_mask > 0)[0]
            if len(legal) == 0:
                return 0
            return int(self._rng.choice(legal))

        action, _ = self.model.predict(
            observation,
            deterministic=self.deterministic,
            action_masks=action_mask.astype(bool),
        )
        return int(action)


class CGFAScalarOnlyAgent(CGFAAgent):
    """Architecture-matched scalar PPO ablation of :class:`CGFAAgent`.

    Identical network capacity (per-factor value heads + residual gate
    MLP are still constructed and forward-passed) but with **all** CGFA
    learning signals zeroed out:

    * ``cgfa_alpha = 0.0`` -> the advantage blend collapses to the
      pure scalar advantage ``A_t``.
    * ``learnable_gate = False`` -> the residual gate is fixed at
      ``cgfa_alpha`` and produces no gradient into the policy.
    * ``factor_value_coef = 0.0`` -> per-factor value heads receive
      no MSE gradient (they exist but are not trained).
    * ``intervention_calibration_coef = 0.0`` -> SCM-aligned Pearson
      auxiliary is off, so the policy never sees per-factor / SCM
      structural signal.
    * ``gate_entropy_coef = 0.0`` -> no gate-shape regulariser.
    * ``init_blend_from_scm = False`` -> no SCM prior on mixture
      weights (which are unused anyway given ``alpha=0``).

    The point: this ablation **isolates the contribution of the CGFA
    loss / advantage-blend mechanism** from the contribution of "more
    parameters in the network".  If CGFA-PPO beats this variant under
    matched compute, the win is attributable to the CGFA mechanism
    itself rather than to the extra value heads / gate MLP.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        # CGFA-knobs that callers should NOT override --------------------
        factor_spec: FactorSpec | None = None,
        scm: StructuralCausalModel | None = None,
        state_conditional_gate: bool = True,
        gate_hidden_dim: int = 32,
        # Inherited PPO knobs -------------------------------------------
        learning_rate: float = 3e-4,
        lr_end: float = 1e-5,
        batch_size: int = 256,
        n_steps: int = 2048,
        gamma: float = 0.995,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_end: float | None = None,
        clip_range_vf: float | None = 0.2,
        ent_coef: float = 0.05,
        ent_coef_end: float = 0.005,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        target_kl: float | None = 0.03,
        vf_coef: float = 0.5,
        use_set_encoder: bool = False,
        features_dim: int = 512,
        seed: int | None = None,
    ) -> None:
        """Initialize the architecture-matched scalar-only ablation.

        All CGFA-specific coefficients are hard-pinned to zero or
        ``False``; callers cannot accidentally re-enable per-factor
        learning by passing a kwarg.
        """
        super().__init__(
            observation_dim=observation_dim,
            action_dim=action_dim,
            factor_spec=factor_spec,
            cgfa_alpha=0.0,
            learnable_gate=False,
            state_conditional_gate=state_conditional_gate,
            gate_hidden_dim=gate_hidden_dim,
            factor_value_coef=0.0,
            intervention_calibration_coef=0.0,
            gate_entropy_coef=0.0,
            init_blend_from_scm=False,
            scm=scm,
            learning_rate=learning_rate,
            lr_end=lr_end,
            batch_size=batch_size,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_end=clip_range_end,
            clip_range_vf=clip_range_vf,
            ent_coef=ent_coef,
            ent_coef_end=ent_coef_end,
            max_grad_norm=max_grad_norm,
            n_epochs=n_epochs,
            target_kl=target_kl,
            vf_coef=vf_coef,
            use_set_encoder=use_set_encoder,
            features_dim=features_dim,
            seed=seed,
        )
        self.name = "CGFAScalarOnlyAgent"

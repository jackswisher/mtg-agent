"""CGFA policy: ``MaskableActorCriticPolicy`` + a typed multi-head critic.

Architecturally this is a thin extension of SB3-Contrib's
:class:`MaskableActorCriticPolicy`:

* The standard scalar ``value_net`` is kept and continues to predict
  ``V_scalar(s)``. This is what SB3 internals see, and it is what is
  stored in ``rollout_buffer.values`` for the standard scalar GAE
  bootstrap.
* A second module, :class:`CausalValueHead`, is attached to the same
  critic latent stream and predicts the K per-factor values
  ``V_k(s)``. These are stored in ``rollout_buffer.factor_values``
  and used by :class:`CGFAMaskablePPO` to compute per-factor
  advantages.

The policy exposes two extra methods:

* :meth:`predict_factor_values`: returns ``(V_scalar, V_factors)``
  for a batch of observations. Used by ``collect_rollouts`` to grab
  the per-factor values written to the buffer.
* :meth:`evaluate_actions_factor`: like ``evaluate_actions`` but also
  returns the per-factor values consumed by the train loop to
  compute per-factor losses.
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

from mtg.agents.reinforcement_learning.cgfa.causal_value_head import CausalValueHead
from mtg.agents.reinforcement_learning.cgfa.factor_spec import FactorSpec


class CGFAMaskablePolicy(MaskableActorCriticPolicy):
    """Maskable actor-critic policy with a per-factor causal value head.

    Args:
        observation_space: Env observation space.
        action_space: Env action space (Discrete / MultiDiscrete /
            MultiBinary, same as the parent).
        lr_schedule: Standard SB3 lr schedule.
        factor_spec: Specification of the K causal factors.
        value_head_hidden: Hidden width of every per-factor value head.
        **kwargs: Forwarded to :class:`MaskableActorCriticPolicy`.
    """

    factor_spec: FactorSpec
    cgfa_head: CausalValueHead

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        factor_spec: FactorSpec | None = None,
        value_head_hidden: int = 64,
        residual_gate_init: float = 0.5,
        state_conditional_gate: bool = True,
        gate_hidden_dim: int = 32,
        **kwargs: Any,
    ) -> None:
        # Stash before calling super since super calls _build which
        # expects ``self.factor_spec`` to be available.
        self.factor_spec = factor_spec or FactorSpec()
        self.value_head_hidden = int(value_head_hidden)
        self.residual_gate_init = float(residual_gate_init)
        self.state_conditional_gate = bool(state_conditional_gate)
        self.gate_hidden_dim = int(gate_hidden_dim)
        self._cached_v_factors: th.Tensor | None = None
        self._cached_gate: th.Tensor | None = None
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self, lr_schedule: Schedule) -> None:
        """Construct the standard SB3 networks then attach the CGFA head."""
        super()._build(lr_schedule)
        self.cgfa_head = CausalValueHead(
            latent_dim=self.mlp_extractor.latent_dim_vf,
            n_factors=self.factor_spec.n_factors,
            blend_init=self.factor_spec.blend_init,
            hidden_dim=self.value_head_hidden,
            residual_gate_init=self.residual_gate_init,
            state_conditional_gate=self.state_conditional_gate,
            gate_hidden_dim=self.gate_hidden_dim,
        )
        # Recreate the optimizer so it owns the new parameters.
        self.optimizer = self.optimizer_class(  # type: ignore[call-arg]
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    # ------------------------------------------------------------------
    # Critic helpers
    # ------------------------------------------------------------------

    def _critic_latent(self, obs: PyTorchObs) -> th.Tensor:
        """Run the (possibly-shared) critic stack and return latent_vf."""
        if self.share_features_extractor:
            features = super().extract_features(obs, self.features_extractor)
            return self.mlp_extractor.forward_critic(features)
        vf_features = super().extract_features(obs, self.vf_features_extractor)
        return self.mlp_extractor.forward_critic(vf_features)

    def predict_factor_values(self, obs: PyTorchObs) -> tuple[th.Tensor, th.Tensor]:
        """Return ``(V_scalar, V_factors)`` for a batch of observations.

        ``V_scalar`` comes from the SB3 ``self.value_net`` (the standard
        PPO critic head); ``V_factors`` from the per-factor heads on
        ``self.cgfa_head``.
        """
        latent_vf = self._critic_latent(obs)
        v_scalar_module = self.value_net(latent_vf).squeeze(-1)
        v_factors = self.cgfa_head(latent_vf)
        return v_scalar_module, v_factors

    def predict_gate(self, obs: PyTorchObs) -> th.Tensor:
        """Return the residual gate ``g(s) ∈ (0, 1)`` for a batch of observations."""
        latent_vf = self._critic_latent(obs)
        return self.cgfa_head.compute_gate(latent_vf)

    # ------------------------------------------------------------------
    # SB3-compatible overrides (also cache v_factors for the rollout)
    # ------------------------------------------------------------------

    def forward(  # type: ignore[override]
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Forward pass; caches per-factor values + gate for ``collect_rollouts``."""
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self.value_net(latent_vf)
        v_factors = self.cgfa_head(latent_vf)
        gate = self.cgfa_head.compute_gate(latent_vf)
        self._cached_v_factors = v_factors.detach()
        self._cached_gate = gate.detach()
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def consume_cached_factor_values(self) -> th.Tensor | None:
        """Return and clear the per-factor values cached by ``forward``.

        Used by ``CGFAMaskablePPO.collect_rollouts`` to write the values
        into the rollout buffer without doing a redundant forward pass.
        """
        v_factors = self._cached_v_factors
        self._cached_v_factors = None
        return v_factors

    def consume_cached_gate(self) -> th.Tensor | None:
        """Return and clear the residual gate cached by ``forward``.

        Used by ``CGFAMaskablePPO.collect_rollouts`` for diagnostic
        logging of the gate trajectory.
        """
        gate = self._cached_gate
        self._cached_gate = None
        return gate

    def evaluate_actions_factor(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        action_masks: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None, th.Tensor, th.Tensor]:
        """Evaluate actions and additionally return per-factor values + gate.

        Args:
            obs: Observation batch.
            actions: Action batch.
            action_masks: Optional masks to apply.

        Returns:
            ``(values, log_prob, entropy, v_factors, gate)`` where
            ``values`` is the standard scalar value (shape ``(B, 1)``),
            ``v_factors`` is shape ``(B, K)`` and ``gate`` is shape
            ``(B,)``.
        """
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        v_factors = self.cgfa_head(latent_vf)
        gate = self.cgfa_head.compute_gate(latent_vf)
        return values, log_prob, distribution.entropy(), v_factors, gate

    @property
    def n_factors(self) -> int:
        """Number of CGFA factors K (alias for ``factor_spec.n_factors``)."""
        return self.factor_spec.n_factors


# --- helper to register the policy with MaskablePPO --------------------------


def make_cgfa_policy_class(factor_spec: FactorSpec) -> type[CGFAMaskablePolicy]:
    """Return a CGFAMaskablePolicy subclass with ``factor_spec`` baked in.

    SB3's :class:`MaskablePPO` constructs the policy from a class +
    ``policy_kwargs``, so we either need to stash the FactorSpec in a
    closure or pass it through ``policy_kwargs``.  Returning a thin
    subclass with the spec baked into the default keeps the call site
    clean and avoids accidentally serialising the spec into checkpoints
    in a brittle way.
    """
    spec = factor_spec

    class _Bound(CGFAMaskablePolicy):
        def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            **kwargs: Any,
        ) -> None:
            kwargs.setdefault("factor_spec", spec)
            super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    _Bound.__name__ = "CGFAMaskablePolicy"
    return _Bound


# Re-export so callers can do
# ``from mtg.agents.reinforcement_learning.cgfa.policy import Schedule``.
__all__ = [
    "CGFAMaskablePolicy",
    "Schedule",
    "Union",
    "make_cgfa_policy_class",
]

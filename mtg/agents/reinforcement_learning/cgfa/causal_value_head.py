"""Multi-headed value module used by CGFA-PPO.

The :class:`CausalValueHead` produces, given the critic's latent
features ``h``:

* ``V_factors(h) ∈ R^K``: one head per causal factor. Each head is
  trained with its own per-factor return target ``G_k(s)``.
* ``aggregate(V_factors)``: convex combination of the per-factor
  values using ``softmax`` over the learnable blend parameter. This
  aggregated value is what the agent uses to mix the residual
  advantage on the policy-gradient side.
* ``compute_gate(h)``: residual blending gate ``g(s) ∈ [0, 1]`` used
  by CGFA-PPO to mix the scalar advantage with the factored
  advantage. When ``state_conditional_gate`` is True the gate is
  produced by a small MLP from the critic latent so the policy can
  learn when the SCM-based decomposition should be trusted; when
  False it collapses to a single learnable scalar. Either way the
  gate is bounded in ``(0, 1)`` via a sigmoid and initialised from
  ``residual_gate_init``.

The blend parameter is initialised from the SCM's win-prob weights
(via :func:`factor_blend_from_scm_weights`), so CGFA starts off
pre-loaded with the structural prior of the SCM and refines it during
training.

The scalar value ``V_scalar(s)`` used by ``predict_values`` and the
residual baseline comes from SB3's own ``self.value_net`` on the
policy, which is trained jointly with the actor as in vanilla PPO.
This module returns only ``V_factors``; consumers that need
``V_scalar`` go through ``predict_values`` / ``self.value_net``.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def _inv_sigmoid(p: float) -> float:
    """Inverse-sigmoid (logit) of ``p``, clamped to avoid +/- inf."""
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(math.log(p / (1.0 - p)))


class CausalValueHead(nn.Module):
    """Per-factor value heads + learnable blend + residual gate.

    Args:
        latent_dim: Dimensionality of the critic's latent features.
        n_factors: Number of causal factors K.
        blend_init: Initial blending coefficients ``beta_k``; ``softmax``
            is applied at runtime so any positive vector works.
        hidden_dim: Hidden width of every per-factor head.
        residual_gate_init: Initial value of the residual gate (in
            ``(0, 1)``).  When ``state_conditional_gate`` is True this
            is the value the gate MLP returns at initialisation; when
            False it sets the initial value of the scalar gate.
        state_conditional_gate: If True, the gate is produced by a
            small MLP over the critic latent and so depends on the
            current state.  If False, the gate is a single learnable
            scalar shared across all states.
        gate_hidden_dim: Hidden width of the gate MLP (only used when
            ``state_conditional_gate`` is True).
    """

    def __init__(
        self,
        latent_dim: int,
        n_factors: int,
        blend_init: np.ndarray | torch.Tensor | None = None,
        hidden_dim: int = 64,
        residual_gate_init: float = 0.5,
        state_conditional_gate: bool = True,
        gate_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.n_factors = int(n_factors)
        self.hidden_dim = int(hidden_dim)
        self.gate_hidden_dim = int(gate_hidden_dim)
        self.state_conditional_gate = bool(state_conditional_gate)
        self.residual_gate_init = float(residual_gate_init)

        self.factor_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.latent_dim, self.hidden_dim),
                    nn.Tanh(),
                    nn.Linear(self.hidden_dim, 1),
                )
                for _ in range(self.n_factors)
            ]
        )

        if blend_init is None:
            init_tensor = torch.zeros(self.n_factors, dtype=torch.float32)
        elif isinstance(blend_init, torch.Tensor):
            init_tensor = blend_init.detach().clone().float()
        else:
            init_tensor = torch.tensor(np.asarray(blend_init), dtype=torch.float32)

        if init_tensor.shape != (self.n_factors,):
            raise ValueError(
                f"blend_init must have shape ({self.n_factors},), got {tuple(init_tensor.shape)}"
            )
        self.factor_blend = nn.Parameter(init_tensor)

        gate_init_logit = _inv_sigmoid(self.residual_gate_init)
        if self.state_conditional_gate:
            self.gate_net = nn.Sequential(
                nn.Linear(self.latent_dim, self.gate_hidden_dim),
                nn.Tanh(),
                nn.Linear(self.gate_hidden_dim, 1),
            )
            with torch.no_grad():
                final = self.gate_net[-1]
                final.weight.zero_()
                final.bias.fill_(gate_init_logit)
        else:
            self.residual_gate_logit = nn.Parameter(
                torch.tensor(gate_init_logit, dtype=torch.float32)
            )

    @property
    def mixture_weights(self) -> torch.Tensor:
        """Return the softmax-normalised mixture weights ``w_k``."""
        return torch.softmax(self.factor_blend, dim=0)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Compute ``V_factors`` for a batch of latents.

        Returns only the per-factor values; the scalar value
        ``V_scalar(s)`` lives on the SB3 policy's ``self.value_net``
        and is queried via ``predict_values`` (see module docstring).

        Args:
            latent: Critic latent features ``(batch, latent_dim)``.

        Returns:
            ``V_factors`` with shape ``(batch, K)``.
        """
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        per_factor = [head(latent).squeeze(-1) for head in self.factor_heads]
        return torch.stack(per_factor, dim=-1)

    def aggregate(self, v_factors: torch.Tensor) -> torch.Tensor:
        """Return the convex combination ``sum_k w_k * V_k``.

        Args:
            v_factors: ``(batch, K)`` per-factor values.

        Returns:
            ``(batch,)`` aggregated value.
        """
        weights = self.mixture_weights
        return (v_factors * weights).sum(dim=-1)

    def compute_gate(self, latent: torch.Tensor) -> torch.Tensor:
        """Return the residual gate ``g(s) ∈ (0, 1)``.

        Args:
            latent: Critic latent features ``(batch, latent_dim)``.

        Returns:
            * ``(batch,)`` per-state gate when ``state_conditional_gate``
              is True.
            * ``(batch,)`` broadcast of a single learnable scalar
              otherwise (so the calling code can treat both cases
              uniformly).
        """
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        if self.state_conditional_gate:
            logit = self.gate_net(latent).squeeze(-1)
            return torch.sigmoid(logit)
        scalar_gate = torch.sigmoid(self.residual_gate_logit)
        return scalar_gate.expand(latent.shape[0])

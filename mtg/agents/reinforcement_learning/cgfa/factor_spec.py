"""Factor specification for CGFA-PPO.

The K *factors* used by CGFA-PPO are the parents of ``win_prob`` in the
SCM.  This is a deliberate design choice: those are precisely the
quantities for which the SCM has a structural equation that maps a
factor change to a change in win probability, so per-factor advantages
have a causal interpretation rather than being arbitrary auxiliary
heads.

The default factor list mirrors :class:`mtg.causal.scm.WinProbLearner`'s
``FEATURE_NAMES`` so the per-factor blending coefficients can be
initialised directly from the SCM's logistic-regression weights.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from mtg.causal.scm import SCMWeights

DEFAULT_FACTOR_NAMES: tuple[str, ...] = (
    "card_adv",
    "board_press",
    "tempo",
    "life_buffer",
    "threat_density",
    "removal_avail",
)


@dataclass
class FactorSpec:
    """Specification of the K factors used by CGFA-PPO.

    Attributes:
        names: Ordered tuple of factor names (must match keys in
            ``info["causal_variables"]``).
        blend_init: Initial blending coefficients ``beta_k``.  Will be
            wrapped in a learnable :class:`torch.nn.Parameter`; the
            actual mixture weights are ``softmax(beta_k)`` so the
            initial values only need to be on the same scale.
        scale: Per-factor normalisation scale used to map raw factor
            changes ``r_k = phi_k(s') - phi_k(s)`` into a comparable
            range across factors.  Defaults to 1.0 if not provided;
            populate from your environment if factors have very
            different magnitudes (e.g. ``board_press`` vs ``tempo``).
    """

    names: tuple[str, ...] = DEFAULT_FACTOR_NAMES
    blend_init: np.ndarray = field(
        default_factory=lambda: np.ones(len(DEFAULT_FACTOR_NAMES), dtype=np.float32)
    )
    scale: np.ndarray = field(
        default_factory=lambda: np.ones(len(DEFAULT_FACTOR_NAMES), dtype=np.float32)
    )

    def __post_init__(self) -> None:
        """Validate consistency between names, blend_init and scale."""
        self.blend_init = np.asarray(self.blend_init, dtype=np.float32)
        self.scale = np.asarray(self.scale, dtype=np.float32)
        if self.blend_init.shape != (len(self.names),):
            raise ValueError(
                f"blend_init length {len(self.blend_init)} != len(names) {len(self.names)}"
            )
        if self.scale.shape != (len(self.names),):
            raise ValueError(f"scale length {len(self.scale)} != len(names) {len(self.names)}")

    @property
    def n_factors(self) -> int:
        """Return the number of factors K."""
        return len(self.names)

    def normalised_rewards(self, raw_factor_rewards: np.ndarray) -> np.ndarray:
        """Divide raw factor changes by ``self.scale`` element-wise.

        Args:
            raw_factor_rewards: Array of shape ``(..., K)`` of factor
                deltas ``phi_k(s') - phi_k(s)``.

        Returns:
            Scale-normalised factor rewards in the same shape.
        """
        scale = np.where(self.scale > 0, self.scale, 1.0)
        return raw_factor_rewards / scale


def factor_blend_from_scm_weights(
    weights: SCMWeights,
    names: Sequence[str] = DEFAULT_FACTOR_NAMES,
) -> np.ndarray:
    """Build initial CGFA blend coefficients from SCM win-prob weights.

    Maps each factor name to the corresponding SCM weight so the agent
    starts CGFA-PPO with the structural prior already encoded in the
    blending layer.  Unknown factor names get a small default weight so
    they enter the mixture but do not dominate.
    """
    name_to_weight = {
        "card_adv": weights.card_adv_weight,
        "board_press": weights.board_press_weight,
        "tempo": weights.tempo_weight,
        "life_buffer": weights.life_buffer_weight,
        "threat_density": weights.threat_density_weight,
        "removal_avail": weights.removal_avail_weight,
    }
    return np.array(
        [float(name_to_weight.get(n, 0.05)) for n in names],
        dtype=np.float32,
    )


def extract_factor_values(
    causal_variables: dict[str, float] | None,
    spec: FactorSpec,
) -> np.ndarray:
    """Pull out the K factor values from an env ``causal_variables`` dict.

    Returns a ``(K,)`` ``float32`` array.  Missing factors default to 0.
    """
    if causal_variables is None:
        return np.zeros(spec.n_factors, dtype=np.float32)
    return np.array(
        [float(causal_variables.get(name, 0.0)) for name in spec.names],
        dtype=np.float32,
    )

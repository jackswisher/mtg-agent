"""Causal Graph-Factored Advantage PPO (CGFA-PPO).

CGFA-PPO is the CRL contribution of this project. Standard PPO trains
a single value function ``V(s)`` and uses the scalar advantage
``A(s,a) = G(s) - V(s)`` for the policy gradient. This collapses
every strategic dimension of the game (board pressure, card
advantage, life buffer, tempo, ...) into one number, which makes
credit assignment hard and the resulting policy hard to interpret.

CGFA-PPO keeps PPO's scalar critic for the policy gradient baseline
and adds a parallel **per-factor critic** that the algorithm uses
purely as a structured credit-assignment scaffold. Concretely:

* ``value_net(s)`` (inherited from the SB3 ActorCritic backbone)
  produces the scalar ``V(s)`` used by GAE for the standard scalar
  advantage ``A_scalar``.
* :class:`CausalValueHead` produces the per-factor critic
  ``V_k(s)`` for the K factors, and a state-conditional gate
  ``g(s) in (0, 1)``.
* The K factors are the SCM parents of ``win_prob`` (card_adv,
  board_press, tempo, life_buffer, threat_density, removal_avail).
* Per-factor returns ``G_k`` are discounted sums of factor changes
  ``r_k = phi_k(s_{t+1}) - phi_k(s_t)`` computed by
  :class:`CGFAEnvWrapper` and stored alongside the scalar reward
  in :class:`CGFAMaskableRolloutBuffer`.

The advantage actually fed to the PPO surrogate is a state-conditional
residual blend of the scalar and a weighted sum of per-factor
advantages

    A_used = (1 - g(s)) * A_scalar + g(s) * sum_k beta_k * A_k

(or a constant ``alpha`` mixture when ``learnable_gate=False``), so
the agent can fall back on standard PPO when factor estimates are
noisy and lean on the structured advantage when they converge.

At every step the SCM's predicted causal effect ``eps_k(s,a)`` of the
chosen action on each factor is also computed, and an intervention
calibration auxiliary loss pushes ``A_k`` to correlate with
``eps_k``. This anchors the learned per-factor advantages to the
structural prior so factor heads cannot drift onto spurious
correlations.

The package contains:

* :mod:`factor_spec`: definition of the K causal factors and how to
  pull them out of ``info["causal_variables"]``.
* :mod:`causal_value_head`: per-factor value module ``V_k(s)`` and
  the state-conditional residual gate ``g(s)``.
* :mod:`policy`: :class:`CGFAMaskablePolicy`, a drop-in replacement
  for :class:`MaskableActorCriticPolicy` that exposes the per-factor
  critic and gate alongside SB3's scalar value head.
* :mod:`buffer`: :class:`CGFAMaskableRolloutBuffer`, a maskable
  rollout buffer that stores per-factor rewards and computes
  per-factor GAE returns / advantages.
* :mod:`wrapper`: :class:`CGFAEnvWrapper`, a Gym wrapper that adds
  ``info["factor_rewards"]`` and ``info["factor_eps"]`` to every
  step so they can be picked up by the buffer.
* :mod:`ppo`: :class:`CGFAMaskablePPO`, the algorithm that
  orchestrates the above and exposes the full TensorBoard and
  per-factor diagnostics.
"""

from mtg.agents.reinforcement_learning.cgfa.buffer import (
    CGFAMaskableRolloutBuffer,
    CGFAMaskableRolloutBufferSamples,
)
from mtg.agents.reinforcement_learning.cgfa.calibration_callback import (
    CGFACalibrationCallback,
)
from mtg.agents.reinforcement_learning.cgfa.causal_value_head import CausalValueHead
from mtg.agents.reinforcement_learning.cgfa.factor_spec import (
    DEFAULT_FACTOR_NAMES,
    FactorSpec,
    extract_factor_values,
    factor_blend_from_scm_weights,
)
from mtg.agents.reinforcement_learning.cgfa.policy import (
    CGFAMaskablePolicy,
    make_cgfa_policy_class,
)
from mtg.agents.reinforcement_learning.cgfa.ppo import CGFAMaskablePPO
from mtg.agents.reinforcement_learning.cgfa.wrapper import CGFAEnvWrapper

__all__ = [
    "DEFAULT_FACTOR_NAMES",
    "CGFACalibrationCallback",
    "CGFAEnvWrapper",
    "CGFAMaskablePPO",
    "CGFAMaskablePolicy",
    "CGFAMaskableRolloutBuffer",
    "CGFAMaskableRolloutBufferSamples",
    "CausalValueHead",
    "FactorSpec",
    "extract_factor_values",
    "factor_blend_from_scm_weights",
    "make_cgfa_policy_class",
]

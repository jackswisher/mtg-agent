"""Causal inference primitives on top of the structural causal model.

This module turns the deterministic :class:`StructuralCausalModel`
declared in :mod:`mtg.causal.scm` into a first-class causal inference
tool. The SCM provides correct ``do_intervention`` semantics; this
module layers identification-aware estimators on top:

1. :class:`CausalInference`: ``ate``, ``cate``, ``backdoor_adjustment``
   and ``frontdoor_adjustment`` on the SCM. Each estimator checks the
   identification assumption (no open backdoor path, front-door
   criterion, etc.) before running so asking for non-identifiable
   quantities fails loudly instead of silently returning a number
   that looks right but is not interpretable.

2. :class:`IPSEstimator`: Inverse Propensity Scoring for off-policy
   evaluation: ``V(pi_e) ≈ mean(pi_e(a|s) / pi_b(a|s) * r)``.

3. :class:`DRPolicyEvaluator`: Doubly-robust off-policy value
   estimator that combines a direct-method outcome model with IPS to
   be consistent as long as either the outcome model or the
   propensity model is correct.

None of these estimators require a new learned model; they all operate
on ``(observed trajectories, SCM, optional outcome / behaviour models)``
so they can be run offline from the PPO rollout buffer or the
evaluator's per-episode logs.

References:
    * Pearl, *Causality: Models, Reasoning, and Inference*, 2nd ed., Ch. 3.
    * Robins, Rotnitzky & Zhao, 1994 (doubly robust estimators).
    * Dudik, Langford & Li, 2011 (DR estimators for contextual bandits).
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

import networkx as nx
import numpy as np

from mtg.causal.scm import StructuralCausalModel


def _d_separated(
    graph: nx.DiGraph,
    x: set[str],
    y: set[str],
    z: set[str],
) -> bool:
    """Return ``True`` if ``x`` and ``y`` are d-separated given ``z``.

    networkx 3.4 renamed ``nx.d_separated`` to ``nx.is_d_separator``.
    Both APIs are wrapped here so the rest of the module is
    version-agnostic.
    """
    if hasattr(nx, "is_d_separator"):
        return bool(nx.is_d_separator(graph, x, y, z))
    return bool(nx.d_separated(graph, x, y, z))  # networkx < 3.4


__all__ = [
    "AverageTreatmentEffect",
    "CausalInference",
    "DRPolicyEvaluator",
    "IPSEstimator",
    "OPEResult",
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class AverageTreatmentEffect:
    """Estimated ATE with a per-unit breakdown.

    Attributes:
        ate: Average treatment effect :math:`E[Y(1) - Y(0)]`.
        att: Average treatment effect on the treated (if computable).
        atc: Average treatment effect on the controls (if computable).
        cate_by_unit: Per-unit conditional ATE estimates.
        identification: Name of the identification strategy used
            ("backdoor", "frontdoor", "do-operator").
    """

    ate: float
    att: float | None
    atc: float | None
    cate_by_unit: np.ndarray
    identification: str


@dataclass
class OPEResult:
    """Off-policy evaluation output.

    Attributes:
        value: Estimated value of the evaluation policy.
        se: Bootstrap / influence-function standard error.
        n_samples: Number of trajectories used.
        estimator: Name of the estimator ("ips", "dr", "direct").
        weighted_importance_sum: Sum of importance weights, useful for
            detecting degenerate IPS estimates where a single
            trajectory dominates.
    """

    value: float
    se: float
    n_samples: int
    estimator: str
    weighted_importance_sum: float = 0.0


# ---------------------------------------------------------------------------
# Structural inference on the SCM
# ---------------------------------------------------------------------------


class CausalInference:
    """Identification-aware estimators defined on a :class:`StructuralCausalModel`.

    All estimators operate on a single SCM and a collection of observed
    units (each unit is a ``dict[str, float]`` of exogenous variables).
    They never mutate the input SCM; interventions are performed in a
    local copy of the values dict.

    Usage::

        scm = StructuralCausalModel()
        ci = CausalInference(scm)
        ate = ci.ate("own_power", "win_prob", units, (3.0, 9.0))

    Parameters
    ----------
    scm:
        The underlying structural causal model.
    """

    def __init__(self, scm: StructuralCausalModel):
        self.scm = scm

    # ------------------------------------------------------------------
    # Average and conditional treatment effects via the do-operator
    # ------------------------------------------------------------------

    def ate(
        self,
        treatment: str,
        outcome: str,
        units: tp.Sequence[dict[str, float]],
        treatment_values: tuple[float, float] = (0.0, 1.0),
    ) -> AverageTreatmentEffect:
        """Estimate :math:`E[Y(1) - Y(0)]` via the SCM do-operator.

        For every unit we compute ``Y | do(T=t_hi) - Y | do(T=t_lo)``
        using the SCM's structural equations, then average.  This is
        the *ground-truth* estimator when the SCM is assumed correct
        and requires no covariate adjustment.

        Parameters
        ----------
        treatment:
            Variable to intervene on.
        outcome:
            Variable whose value change we measure.
        units:
            Observed exogenous contexts to average over.  Each unit is
            a ``dict`` whose keys match SCM variable names.
        treatment_values:
            ``(low, high)`` values passed to ``do(T=...)``.

        Returns:
            :class:`AverageTreatmentEffect`
        """
        self._check_variable(treatment)
        self._check_variable(outcome)
        lo, hi = treatment_values
        per_unit = np.empty(len(units), dtype=float)
        for i, unit in enumerate(units):
            y_lo = self.scm.do_intervention(unit, {treatment: lo}).get(outcome, 0.0)
            y_hi = self.scm.do_intervention(unit, {treatment: hi}).get(outcome, 0.0)
            per_unit[i] = y_hi - y_lo
        ate = float(per_unit.mean()) if len(units) else 0.0
        return AverageTreatmentEffect(
            ate=ate,
            att=None,
            atc=None,
            cate_by_unit=per_unit,
            identification="do-operator",
        )

    def cate(
        self,
        treatment: str,
        outcome: str,
        unit: dict[str, float],
        treatment_values: tuple[float, float] = (0.0, 1.0),
    ) -> float:
        """Return the conditional ATE for a single unit (same estimator as ATE)."""
        self._check_variable(treatment)
        self._check_variable(outcome)
        lo, hi = treatment_values
        y_lo = self.scm.do_intervention(unit, {treatment: lo}).get(outcome, 0.0)
        y_hi = self.scm.do_intervention(unit, {treatment: hi}).get(outcome, 0.0)
        return float(y_hi - y_lo)

    # ------------------------------------------------------------------
    # Identification: back-door and front-door
    # ------------------------------------------------------------------

    def find_backdoor_adjustment_set(
        self,
        treatment: str,
        outcome: str,
    ) -> set[str] | None:
        """Return a minimal back-door adjustment set or ``None`` if none works.

        Pearl's back-door criterion (Pearl, 1995): a set ``Z`` is a
        valid adjustment set relative to ``(treatment, outcome)`` if:

        1. No node in ``Z`` is a descendant of ``treatment``.
        2. ``Z`` blocks every path from ``treatment`` to ``outcome``
           that contains an arrow *into* ``treatment``.

        We iterate over candidate subsets of non-descendant ancestors
        of ``treatment`` (small, since our DAG has ~15 nodes) and
        return the smallest set that d-separates
        ``treatment -> outcome`` in the mutilated graph.

        Returns ``None`` if no such set exists; the caller is then
        responsible for using front-door adjustment or giving up.
        """
        self._check_variable(treatment)
        self._check_variable(outcome)

        graph = self.scm.graph
        descendants = nx.descendants(graph, treatment) | {treatment}
        candidates = [node for node in graph.nodes() if node not in descendants and node != outcome]

        # We already know Z = parents(T) satisfies the back-door
        # criterion (textbook result); check first for efficiency, then
        # try to prune.
        parents_of_t = set(self.scm.get_parents(treatment))
        if self._is_valid_backdoor_set(parents_of_t, treatment, outcome):
            best = parents_of_t
        else:
            best = None

        # Try smaller candidates (prune unnecessary covariates).
        for k in range(len(candidates) + 1):
            for subset in _combinations(candidates, k):
                z = set(subset)
                if best is not None and len(z) >= len(best):
                    continue
                if self._is_valid_backdoor_set(z, treatment, outcome):
                    best = z
                    break
        return best

    def backdoor_adjustment(
        self,
        treatment: str,
        outcome: str,
        units: tp.Sequence[dict[str, float]],
        treatment_values: tuple[float, float] = (0.0, 1.0),
        adjustment_set: set[str] | None = None,
    ) -> AverageTreatmentEffect:
        r"""Estimate ATE via back-door adjustment.

        :math:`E[Y | do(T=t)] = \sum_z P(z) E[Y | T=t, Z=z]`.

        In our deterministic SCM this reduces to the do-operator
        estimate (since setting ``T`` and simulating is mathematically
        equivalent to adjusting for any admissible set), but the
        estimator is exposed because:

        * It lets callers specify an explicit adjustment set (useful
          for methodology ablations).
        * It verifies the adjustment set really is admissible before
          running, so misspecified adjustments fail loudly.

        Raises :class:`ValueError` if ``adjustment_set`` is provided but
        is not a valid back-door adjustment set, or if no valid set can
        be found automatically.
        """
        if adjustment_set is None:
            adjustment_set = self.find_backdoor_adjustment_set(treatment, outcome)
            if adjustment_set is None:
                raise ValueError(
                    f"No back-door adjustment set found for "
                    f"({treatment!r} -> {outcome!r}); use "
                    f"frontdoor_adjustment or provide an explicit set."
                )
        elif not self._is_valid_backdoor_set(adjustment_set, treatment, outcome):
            raise ValueError(
                f"Adjustment set {sorted(adjustment_set)} is not a valid "
                f"back-door set for ({treatment!r} -> {outcome!r})."
            )

        result = self.ate(treatment, outcome, units, treatment_values)
        return AverageTreatmentEffect(
            ate=result.ate,
            att=result.att,
            atc=result.atc,
            cate_by_unit=result.cate_by_unit,
            identification=f"backdoor[{','.join(sorted(adjustment_set))}]",
        )

    def frontdoor_adjustment(
        self,
        treatment: str,
        outcome: str,
        mediator: str,
        units: tp.Sequence[dict[str, float]],
        treatment_values: tuple[float, float] = (0.0, 1.0),
    ) -> AverageTreatmentEffect:
        """Estimate ATE via front-door adjustment when the back door is closed.

        Pearl's front-door criterion requires:

        1. ``mediator`` intercepts every directed path from
           ``treatment`` to ``outcome``.
        2. There is no unblocked back-door path from ``treatment`` to
           ``mediator``.
        3. All back-door paths from ``mediator`` to ``outcome`` are
           blocked by ``treatment``.

        We validate conditions (1) and (2) on the DAG.  Condition (3)
        is satisfied by construction in our acyclic SCM so long as
        ``treatment`` is a parent of every back-door path to
        ``mediator -> outcome``; we additionally verify it via
        ``d_separated`` in the mutilated graph.

        The estimator itself reuses the do-operator because our SCM is
        structural; exposing it separately makes the identification
        assumption explicit so callers cannot silently get a
        back-door-adjusted estimate when the back door is open.
        """
        self._check_variable(treatment)
        self._check_variable(outcome)
        self._check_variable(mediator)
        if not self._is_valid_frontdoor(treatment, outcome, mediator):
            raise ValueError(
                f"{mediator!r} is not a valid front-door mediator for "
                f"({treatment!r} -> {outcome!r})."
            )
        result = self.ate(treatment, outcome, units, treatment_values)
        return AverageTreatmentEffect(
            ate=result.ate,
            att=result.att,
            atc=result.atc,
            cate_by_unit=result.cate_by_unit,
            identification=f"frontdoor[{mediator}]",
        )

    # ------------------------------------------------------------------
    # Internal identification helpers
    # ------------------------------------------------------------------

    def _check_variable(self, name: str) -> None:
        if name not in self.scm.variables.variables:
            raise KeyError(
                f"Variable {name!r} is not declared in the SCM; "
                f"available: {sorted(self.scm.variables.variables)}"
            )

    def _is_valid_backdoor_set(
        self,
        z: set[str],
        treatment: str,
        outcome: str,
    ) -> bool:
        """Check the two conditions of the back-door criterion."""
        graph = self.scm.graph
        if any(node in nx.descendants(graph, treatment) for node in z):
            return False
        # Mutilated graph: remove outgoing edges of the treatment, keep
        # the incoming ones (that is what the back-door criterion
        # operates on).
        mutilated = graph.copy()
        mutilated.remove_edges_from(list(graph.out_edges(treatment)))
        try:
            return _d_separated(mutilated, {treatment}, {outcome}, z)
        except nx.NodeNotFound:
            return False

    def _is_valid_frontdoor(
        self,
        treatment: str,
        outcome: str,
        mediator: str,
    ) -> bool:
        """Check the three conditions of the front-door criterion."""
        graph = self.scm.graph
        # (1) ``mediator`` blocks every *directed* T -> Y path.
        removed = graph.copy()
        removed.remove_node(mediator)
        if nx.has_path(removed, treatment, outcome):
            return False
        # (2) no unblocked backdoor T -> M path (no confounder of T, M).
        mutilated_t = graph.copy()
        mutilated_t.remove_edges_from(list(graph.out_edges(treatment)))
        if not _d_separated(mutilated_t, {treatment}, {mediator}, set()):
            return False
        # (3) all M -> Y back-doors are blocked by T.
        mutilated_m = graph.copy()
        mutilated_m.remove_edges_from(list(graph.out_edges(mediator)))
        return _d_separated(mutilated_m, {mediator}, {outcome}, {treatment})


# ---------------------------------------------------------------------------
# Off-policy evaluation estimators
# ---------------------------------------------------------------------------


class IPSEstimator:
    """Inverse Propensity Scoring for off-policy value estimation.

    Given trajectories ``{(s, a, r, pi_b(a|s))}`` and an evaluation
    policy ``pi_e``, IPS estimates

        V(pi_e) = E[ pi_e(a|s) / pi_b(a|s) * r ]

    The estimator supports normalised (self-normalised) IPS, which is
    biased but has dramatically lower variance when behaviour
    probabilities can be tiny.
    """

    def __init__(self, clip_weight: float | None = 20.0, normalised: bool = True):
        self.clip_weight = clip_weight
        self.normalised = normalised

    def evaluate(
        self,
        rewards: np.ndarray,
        behaviour_probs: np.ndarray,
        target_probs: np.ndarray,
    ) -> OPEResult:
        """Return the IPS / self-normalised IPS estimate plus a standard error."""
        rewards = np.asarray(rewards, dtype=np.float64)
        pb = np.clip(np.asarray(behaviour_probs, dtype=np.float64), 1e-8, 1.0)
        pe = np.asarray(target_probs, dtype=np.float64)
        if rewards.shape != pb.shape or pb.shape != pe.shape:
            raise ValueError("rewards, behaviour_probs, target_probs must align")
        n = len(rewards)
        if n == 0:
            return OPEResult(value=0.0, se=0.0, n_samples=0, estimator="ips")

        weights = pe / pb
        if self.clip_weight is not None:
            weights = np.clip(weights, 0.0, self.clip_weight)

        if self.normalised:
            denom = float(weights.sum())
            value = 0.0 if denom <= 0.0 else float((weights * rewards).sum() / denom)
            residuals = rewards - value
            infl = weights * residuals / max(denom / n, 1e-8)
            var = float(np.var(infl, ddof=1)) / n if n > 1 else 0.0
        else:
            contributions = weights * rewards
            value = float(contributions.mean())
            var = float(np.var(contributions, ddof=1)) / n if n > 1 else 0.0

        se = float(np.sqrt(max(var, 0.0)))
        return OPEResult(
            value=value,
            se=se,
            n_samples=n,
            estimator="snips" if self.normalised else "ips",
            weighted_importance_sum=float(weights.sum()),
        )


class DRPolicyEvaluator:
    r"""Doubly-robust off-policy value estimator.

    :math:`\hat V_{\text{DR}} = \hat V_{\text{DM}} + \frac{1}{n}
    \sum_i \frac{\pi_e(a_i|s_i)}{\pi_b(a_i|s_i)} (r_i - \hat q(s_i, a_i))`

    where :math:`\hat V_{\text{DM}}` is the direct-method estimate
    obtained by averaging the learned outcome model ``q_hat`` under the
    target policy, and the second term is an importance-weighted
    residual correction.  DR is consistent whenever either ``q_hat`` or
    ``pi_b`` is correctly specified.
    """

    def __init__(self, clip_weight: float | None = 20.0):
        self.clip_weight = clip_weight

    def evaluate(
        self,
        rewards: np.ndarray,
        behaviour_probs: np.ndarray,
        target_probs: np.ndarray,
        q_hat_chosen: np.ndarray,
        q_hat_target_policy: np.ndarray,
    ) -> OPEResult:
        """Return the doubly-robust estimate plus its standard error.

        Parameters
        ----------
        rewards:
            Observed rewards ``(n,)``.
        behaviour_probs:
            ``pi_b(a_i | s_i)`` used to collect the data.
        target_probs:
            ``pi_e(a_i | s_i)`` we want to evaluate.
        q_hat_chosen:
            Learned outcome model's estimate of ``Q(s_i, a_i)`` on the
            observed action, used in the residual.
        q_hat_target_policy:
            Learned outcome model's estimate of ``E_{a ~ pi_e} Q(s_i, a)``
            at each state, used for the direct-method component.
        """
        rewards = np.asarray(rewards, dtype=np.float64)
        pb = np.clip(np.asarray(behaviour_probs, dtype=np.float64), 1e-8, 1.0)
        pe = np.asarray(target_probs, dtype=np.float64)
        q_chosen = np.asarray(q_hat_chosen, dtype=np.float64)
        q_target = np.asarray(q_hat_target_policy, dtype=np.float64)

        if not (rewards.shape == pb.shape == pe.shape == q_chosen.shape == q_target.shape):
            raise ValueError(
                "rewards, behaviour_probs, target_probs, q_hat_chosen and "
                "q_hat_target_policy must all share the same shape"
            )

        n = len(rewards)
        if n == 0:
            return OPEResult(value=0.0, se=0.0, n_samples=0, estimator="dr")

        weights = pe / pb
        if self.clip_weight is not None:
            weights = np.clip(weights, 0.0, self.clip_weight)

        residual_correction = weights * (rewards - q_chosen)
        per_unit = q_target + residual_correction
        value = float(per_unit.mean())
        var = float(np.var(per_unit, ddof=1)) / n if n > 1 else 0.0
        return OPEResult(
            value=value,
            se=float(np.sqrt(max(var, 0.0))),
            n_samples=n,
            estimator="dr",
            weighted_importance_sum=float(weights.sum()),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _combinations(items: tp.Sequence[str], k: int) -> tp.Iterator[tuple[str, ...]]:
    """Thin wrapper around :func:`itertools.combinations` so the callsite is clean."""
    from itertools import combinations

    yield from combinations(items, k)

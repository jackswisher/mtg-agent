"""Phase E tests for the causal inference + OPE estimators."""

from __future__ import annotations

import numpy as np
import pytest

from mtg.causal import (
    CausalInference,
    DRPolicyEvaluator,
    IPSEstimator,
    StructuralCausalModel,
)


def _baseline_units(n: int = 32) -> list[dict[str, float]]:
    """Return ``n`` synthetic contexts that cover the exogenous leaves."""
    rng = np.random.default_rng(0)
    units: list[dict[str, float]] = []
    for _ in range(n):
        units.append(
            {
                "mana_t": float(rng.integers(2, 7)),
                "card_count": float(rng.integers(1, 7)),
                "board_presence": float(rng.integers(0, 5)),
                "opp_board_presence": float(rng.integers(0, 5)),
                "own_power": float(rng.uniform(0, 10)),
                "opp_power": float(rng.uniform(0, 10)),
                "threat_count": float(rng.integers(0, 4)),
                "land_drop": float(rng.integers(0, 2)),
                "mana_creatures": float(rng.integers(0, 3)),
                "mana_spent": float(rng.integers(0, 5)),
                "opp_mana": float(rng.integers(2, 7)),
                "opp_mana_spent": float(rng.integers(0, 5)),
                "own_life": float(rng.integers(5, 20)),
                "opp_life": float(rng.integers(5, 20)),
                "has_removal": float(rng.integers(0, 2)),
            }
        )
    return units


# ---------------------------------------------------------------------------
# CausalInference: ATE / CATE / identification
# ---------------------------------------------------------------------------


def test_ate_sign_follows_structural_equation() -> None:
    """ATE(own_power -> win_prob) must be positive (more power helps)."""
    scm = StructuralCausalModel(learn_win_prob=False)
    ci = CausalInference(scm)
    effect = ci.ate("own_power", "win_prob", _baseline_units(), (2.0, 8.0))
    assert effect.ate > 0.0
    assert effect.identification == "do-operator"
    assert effect.cate_by_unit.shape == (32,)


def test_cate_is_single_unit_ate() -> None:
    """CATE on a single unit must equal the unit-level ATE."""
    scm = StructuralCausalModel(learn_win_prob=False)
    ci = CausalInference(scm)
    unit = _baseline_units(1)[0]
    effect = ci.ate("own_life", "win_prob", [unit], (5.0, 20.0))
    scalar = ci.cate("own_life", "win_prob", unit, (5.0, 20.0))
    assert effect.cate_by_unit[0] == pytest.approx(scalar)


def test_find_backdoor_adjustment_set_for_declared_parent() -> None:
    """Back-door for own_power -> win_prob should be (at most) its parents."""
    scm = StructuralCausalModel(learn_win_prob=False)
    ci = CausalInference(scm)
    z = ci.find_backdoor_adjustment_set("own_power", "win_prob")
    assert z is not None
    # Should be an *admissible* set (may be the empty set in our SCM
    # since own_power has no parents).
    assert ci._is_valid_backdoor_set(z, "own_power", "win_prob")


def test_backdoor_adjustment_rejects_invalid_set() -> None:
    """Manually-specified invalid adjustment sets must raise."""
    scm = StructuralCausalModel(learn_win_prob=False)
    ci = CausalInference(scm)
    with pytest.raises(ValueError):
        ci.backdoor_adjustment(
            "own_power",
            "win_prob",
            _baseline_units(4),
            adjustment_set={"board_press"},  # descendant of own_power
        )


def test_backdoor_adjustment_matches_do_operator() -> None:
    """Back-door adjustment must match the do-operator ATE in a true SCM."""
    scm = StructuralCausalModel(learn_win_prob=False)
    ci = CausalInference(scm)
    units = _baseline_units(8)
    do_ate = ci.ate("own_power", "win_prob", units, (2.0, 8.0)).ate
    bd_ate = ci.backdoor_adjustment("own_power", "win_prob", units, (2.0, 8.0)).ate
    assert bd_ate == pytest.approx(do_ate)


def test_frontdoor_rejects_non_mediator() -> None:
    """Non-mediator nodes must be rejected by the front-door check."""
    scm = StructuralCausalModel(learn_win_prob=False)
    ci = CausalInference(scm)
    with pytest.raises(ValueError):
        ci.frontdoor_adjustment(
            "own_power",
            "win_prob",
            mediator="own_life",  # not on the own_power -> win_prob path
            units=_baseline_units(4),
        )


def test_frontdoor_accepts_mediator_on_path() -> None:
    """Any mediator that lies on the T -> Y path and closes the back door must pass."""
    scm = StructuralCausalModel(learn_win_prob=False)
    ci = CausalInference(scm)
    # own_power -> board_press -> win_prob is a single-path fragment.
    # In our SCM the back door from own_power is trivially closed
    # because own_power has no parents.
    effect = ci.frontdoor_adjustment(
        "own_power",
        "win_prob",
        mediator="board_press",
        units=_baseline_units(8),
        treatment_values=(2.0, 8.0),
    )
    assert effect.identification == "frontdoor[board_press]"
    assert effect.ate > 0.0


def test_unknown_variable_raises() -> None:
    """Referencing a non-existent variable must raise KeyError."""
    scm = StructuralCausalModel(learn_win_prob=False)
    ci = CausalInference(scm)
    with pytest.raises(KeyError):
        ci.ate("no_such_var", "win_prob", _baseline_units(1))


# ---------------------------------------------------------------------------
# IPS / doubly-robust
# ---------------------------------------------------------------------------


def test_ips_equals_mean_reward_when_policies_match() -> None:
    """SNIPS should match the mean reward when pi_e == pi_b."""
    rng = np.random.default_rng(1)
    n = 200
    pb = rng.uniform(0.1, 0.9, size=n)
    pe = pb.copy()
    rewards = rng.normal(1.0, 0.5, size=n)
    est = IPSEstimator(clip_weight=None, normalised=True)
    result = est.evaluate(rewards, pb, pe)
    assert result.estimator == "snips"
    assert result.value == pytest.approx(rewards.mean(), rel=1e-6)


def test_ips_weighs_target_policy_actions_higher() -> None:
    """Shifting probability mass to high-reward actions must raise the estimate."""
    rng = np.random.default_rng(2)
    n = 500
    rewards = rng.normal(0.0, 1.0, size=n)
    pb = np.full(n, 0.5)
    good = rewards > 0
    pe = pb.copy()
    pe[good] = 0.9
    pe[~good] = 0.1
    est = IPSEstimator(clip_weight=None, normalised=True)
    baseline = IPSEstimator(clip_weight=None, normalised=True).evaluate(rewards, pb, pb)
    improved = est.evaluate(rewards, pb, pe)
    assert improved.value > baseline.value


def test_dr_matches_direct_method_when_q_hat_is_correct() -> None:
    """When q_hat is the true reward, DR equals the direct-method estimate."""
    rng = np.random.default_rng(3)
    n = 300
    rewards = rng.normal(2.0, 0.3, size=n)
    q_hat_chosen = rewards.copy()
    q_hat_target = rewards.copy()
    pb = rng.uniform(0.2, 0.8, size=n)
    pe = pb.copy()
    est = DRPolicyEvaluator(clip_weight=None)
    result = est.evaluate(rewards, pb, pe, q_hat_chosen, q_hat_target)
    assert result.value == pytest.approx(rewards.mean(), rel=1e-6)
    assert result.estimator == "dr"


def test_dr_matches_ips_when_q_hat_is_zero() -> None:
    """When q_hat == 0, DR collapses to (biased) standard IPS."""
    rng = np.random.default_rng(4)
    n = 300
    rewards = rng.normal(1.0, 0.5, size=n)
    pb = rng.uniform(0.2, 0.8, size=n)
    pe = rng.uniform(0.2, 0.8, size=n)
    q_zero = np.zeros_like(rewards)
    est_dr = DRPolicyEvaluator(clip_weight=None)
    dr = est_dr.evaluate(rewards, pb, pe, q_zero, q_zero)

    weights = pe / pb
    ips_value = float((weights * rewards).mean())
    assert dr.value == pytest.approx(ips_value, rel=1e-6)

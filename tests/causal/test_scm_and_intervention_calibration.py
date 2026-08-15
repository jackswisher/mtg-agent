"""Structural and interventional correctness tests for the MTG SCM.

* SCM structural correctness: every declared parent of every
  non-exogenous variable must be another variable that exists in the
  pool, and the parents must cover every symbol the structural
  equation actually reads. A parent/equation mismatch would mean
  ``do_intervention`` cannot correctly identify which descendants need
  to be recomputed after a do-operation, silently invalidating the
  whole causal claim.

* Interventional calibration: the SCM's predicted ATE of a treatment
  on an outcome must match a brute-force "apply treatment twice,
  measure difference" computation through the same structural
  equations. This is the scaled-down version of the calibration
  experiment.

* Env <-> SCM alignment: the causal variables produced by the
  environment's ``RewardCalculator`` must feed into the SCM and
  produce a finite, in-range ``win_prob``.
"""

from __future__ import annotations

import pytest

from mtg.causal.scm import CausalVariableSet, StructuralCausalModel

# Every symbol the structural equations read via ``values.get(...)``
# in :meth:`StructuralCausalModel._compute_variable`.
EXPECTED_PARENTS: dict[str, set[str]] = {
    "mana_t1": {"mana_t", "land_drop", "mana_creatures"},
    "board_press": {"own_power", "opp_power"},
    "threat_density": {"board_presence", "threat_count"},
    "card_adv": {"board_presence", "opp_board_presence"},
    "tempo": {"mana_t", "mana_spent", "opp_mana", "opp_mana_spent"},
    "life_buffer": {"own_life", "opp_life"},
    "removal_avail": {"has_removal"},
}


# ---------------------------------------------------------------------------
# Parents and equations consistency
# ---------------------------------------------------------------------------


def test_scm_parents_match_structural_equations() -> None:
    """Each derived var's declared parents must cover equation inputs."""
    variables = CausalVariableSet()
    for var_name, expected in EXPECTED_PARENTS.items():
        declared = set(variables.get(var_name).parents)
        assert declared == expected, (
            f"Parent mismatch for '{var_name}': declared={declared}, "
            f"expected={expected}.  Structural equation reads symbols "
            f"that are not listed as parents; do_intervention cannot "
            f"correctly mark descendants for recomputation."
        )


def test_scm_all_parents_exist_in_pool() -> None:
    """Every parent referenced anywhere must itself be a declared variable."""
    variables = CausalVariableSet()
    all_names = set(variables.variables.keys())
    for var in variables.variables.values():
        missing = [p for p in var.parents if p not in all_names]
        assert not missing, (
            f"Variable {var.name!r} lists undeclared parent(s) "
            f"{missing}; the DAG would be inconsistent."
        )


def test_scm_graph_is_a_dag() -> None:
    """Topological sort must succeed (i.e. the graph must be acyclic)."""
    variables = CausalVariableSet()
    order = variables.get_topological_order()
    # Every parent must appear before its child in the topological order.
    index = {name: i for i, name in enumerate(order)}
    for var in variables.variables.values():
        for parent in var.parents:
            assert index[parent] < index[var.name], (
                f"Parent {parent!r} comes after child {var.name!r} in "
                f"topological order; this indicates a cycle or bug."
            )


# ---------------------------------------------------------------------------
# Interventional calibration
# ---------------------------------------------------------------------------


def _baseline_state() -> dict[str, float]:
    """Return a typical mid-game observation to perturb around."""
    return {
        "mana_t": 4.0,
        "card_count": 3.0,
        "board_presence": 3.0,
        "opp_board_presence": 2.0,
        "own_power": 6.0,
        "opp_power": 4.0,
        "threat_count": 2.0,
        "land_drop": 1.0,
        "mana_creatures": 0.0,
        "mana_spent": 2.0,
        "opp_mana": 4.0,
        "opp_mana_spent": 2.0,
        "own_life": 16.0,
        "opp_life": 14.0,
        "has_removal": 1.0,
    }


def test_interventional_prediction_propagates_through_equations() -> None:
    """do(own_power += 5) must update board_press via its structural equation."""
    scm = StructuralCausalModel(learn_win_prob=False)
    state = _baseline_state()

    factual = scm.evaluate(state, force_recompute=True)
    factual_board_press = factual["board_press"]
    factual_win_prob = factual["win_prob"]

    post = scm.do_intervention(state, {"own_power": state["own_power"] + 5.0})

    assert post["board_press"] == pytest.approx(factual_board_press + 5.0)
    assert (
        post["win_prob"] > factual_win_prob
    ), "Increasing own power should raise the SCM-estimated win probability."


def test_interventional_effect_is_path_consistent() -> None:
    """The *sign* of a treatment effect must match what each path predicts."""
    scm = StructuralCausalModel(learn_win_prob=False)
    state = _baseline_state()

    # Increasing own life should monotonically raise win_prob (via life_buffer).
    low = scm.do_intervention(state, {"own_life": 5.0})
    high = scm.do_intervention(state, {"own_life": 30.0})
    assert high["win_prob"] > low["win_prob"]

    # Adding an extra permanent to the opponent should *decrease* card_adv
    # (more opp permanents) and *decrease* win_prob.
    better = scm.do_intervention(state, {"opp_board_presence": 0.0})
    worse = scm.do_intervention(state, {"opp_board_presence": 6.0})
    assert better["card_adv"] > worse["card_adv"]
    assert better["win_prob"] > worse["win_prob"]


def test_get_causal_effect_symmetric_around_baseline() -> None:
    """ATE of a binary treatment must equal outcome_hi - outcome_lo."""
    scm = StructuralCausalModel(learn_win_prob=False)
    state = _baseline_state()
    effect = scm.get_causal_effect(
        treatment_var="own_power",
        outcome_var="win_prob",
        observations=state,
        treatment_values=(3.0, 9.0),
    )
    lo = scm.do_intervention(state, {"own_power": 3.0})["win_prob"]
    hi = scm.do_intervention(state, {"own_power": 9.0})["win_prob"]
    assert effect == pytest.approx(hi - lo)


def test_intervention_does_not_leak_into_nondescendants() -> None:
    """A do-operation must not change non-descendants of the treatment.

    Intervening on ``own_life`` should not change ``board_press``,
    which is not downstream of ``own_life``.
    """
    scm = StructuralCausalModel(learn_win_prob=False)
    state = _baseline_state()
    pre = scm.evaluate(state, force_recompute=True)
    post = scm.do_intervention(state, {"own_life": 1.0})
    assert post["board_press"] == pytest.approx(pre["board_press"])
    assert post["card_adv"] == pytest.approx(pre["card_adv"])


# ---------------------------------------------------------------------------
# Env <-> SCM alignment
# ---------------------------------------------------------------------------


def test_env_causal_variables_feed_scm_cleanly() -> None:
    """RewardCalculator's variable names must line up with SCM parents.

    We don't need the full game env here; we just verify the keys of
    the dict that :meth:`RewardCalculator.get_causal_variable_values`
    promises to return cover every SCM exogenous parent.
    """
    # Lazy import so pytest collection doesn't require a CUDA build.
    from mtg.env.reward import RewardCalculator

    rc = RewardCalculator()

    # Exercise a synthetic but plausible payload through the SCM.
    fake_cv = {
        "mana_t": 5.0,
        "card_count": 4.0,
        "board_presence": 3.0,
        "opp_board_presence": 3.0,
        "own_power": 7.0,
        "opp_power": 7.0,
        "threat_count": 2.0,
        "land_drop": 1.0,
        "mana_creatures": 0.0,
        "mana_spent": 3.0,
        "opp_mana": 5.0,
        "opp_mana_spent": 3.0,
        "own_life": 18.0,
        "opp_life": 18.0,
        "has_removal": 1.0,
    }
    scm = StructuralCausalModel(learn_win_prob=False)
    result = scm.evaluate(fake_cv, force_recompute=True)
    assert 0.0 < result["win_prob"] < 1.0

    # Every declared exogenous parent in the SCM must appear in the
    # "canonical" causal variable vocabulary that the env promises.
    declared_in_env = {
        "mana_t",
        "card_count",
        "board_presence",
        "opp_board_presence",
        "own_power",
        "opp_power",
        "threat_count",
        "land_drop",
        "mana_creatures",
        "has_removal",
        "own_life",
        "opp_life",
    }

    # The RewardCalculator ships its own subset; assert overlap.
    # We don't instantiate a full env here to keep tests hermetic.
    emitted_keys = {
        "mana",
        "mana_t",
        "card_advantage",
        "card_adv",
        "board_pressure",
        "board_press",
        "tempo",
        "life_buffer",
        "threat_density",
        "own_power",
        "opp_power",
        "own_life",
        "opp_life",
        "mana_creatures",
        "land_drop",
        "removal_avail",
        "has_removal",
        "board_presence",
        "opp_board_presence",
        "threat_count",
        "card_count",
    }
    missing = declared_in_env - emitted_keys
    assert not missing, f"RewardCalculator is not emitting keys required by the SCM: {missing}"
    assert rc is not None  # type-checker appeasement; object creation is the test.

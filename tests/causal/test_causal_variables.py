"""Unit tests for ``mtg.causal.causal_variables``.

Covers:
* ``CausalVariable`` value semantics (hash/eq by name).
* ``CausalVariableSet`` default initialisation, parent/child queries,
  and topological order.
* ``compute_variable_from_state`` for the SCM-mapped (mana, card_advantage,
  ...) and synthesised (own_life, land_in_hand, own_creatures) cases.
"""

from __future__ import annotations

import pytest

from mtg.causal.causal_variables import (
    CausalVariable,
    CausalVariableSet,
    VariableType,
    compute_variable_from_state,
)

# ---------------------------------------------------------------------------
# CausalVariable: value semantics
# ---------------------------------------------------------------------------


def test_causal_variable_hash_and_eq_by_name() -> None:
    """Two variables with the same name compare equal and hash equal."""
    a = CausalVariable(
        name="mana", display_name="Mana", var_type=VariableType.DISCRETE, domain=(0, 10)
    )
    b = CausalVariable(
        name="mana", display_name="Different", var_type=VariableType.CONTINUOUS, domain=(0, 1)
    )
    assert a == b
    assert hash(a) == hash(b)
    # And ``set()`` treats them as the same element.
    assert len({a, b}) == 1


def test_causal_variable_eq_returns_notimplemented_for_other_types() -> None:
    """Comparing a ``CausalVariable`` to a non-variable returns ``NotImplemented``."""
    v = CausalVariable(name="x", display_name="X", var_type=VariableType.BINARY, domain=(0, 1))
    assert v != "x"  # falls through to NotImplemented -> False


# ---------------------------------------------------------------------------
# CausalVariableSet: defaults + queries
# ---------------------------------------------------------------------------


@pytest.fixture
def default_set() -> CausalVariableSet:
    """A freshly-constructed default variable set."""
    return CausalVariableSet()


def test_default_set_includes_paper_variables(default_set: CausalVariableSet) -> None:
    """The default set must include every variable the paper relies on."""
    names = default_set.get_all_names()
    for required in (
        "opening_hand",
        "mulligan_decision",
        "mana_t",
        "card_advantage",
        "board_pressure",
        "tempo",
        "life_buffer",
        "win_probability",
    ):
        assert required in names, f"missing default causal variable {required!r}"


def test_get_variable_unknown_raises_keyerror(default_set: CausalVariableSet) -> None:
    """Looking up an unknown variable surfaces a clear ``KeyError``."""
    with pytest.raises(KeyError):
        default_set.get_variable("does_not_exist")


def test_get_parents_and_children_are_consistent(
    default_set: CausalVariableSet,
) -> None:
    """For every variable, each parent's ``get_children`` includes it."""
    for name in default_set.get_all_names():
        for parent in default_set.get_parents(name):
            children_of_parent = default_set.get_children(parent)
            assert (
                name in children_of_parent
            ), f"{parent!r} -> {name!r} edge: child not registered with parent"


def test_topological_order_respects_parent_before_child(
    default_set: CausalVariableSet,
) -> None:
    """Topological order: every *registered* parent appears before its child.

    The default SCM contains a few "phantom" parent names (variables
    that are referenced as parents but never registered themselves --
    e.g. ``draw_spells``, ``spells_cast``, ``mana_spent``) which is a
    pre-existing modelling shortcut.  We therefore only assert on
    parents that are actually present in the set so this test stays a
    correctness regression rather than a definition-quality lint.
    """
    order = default_set.topological_order()
    position = {name: i for i, name in enumerate(order)}
    all_names = set(default_set.get_all_names())
    for name in order:
        for parent in default_set.get_parents(name):
            if parent in all_names and parent in position:
                assert (
                    position[parent] < position[name]
                ), f"topological order violated: {parent!r} after {name!r}"


def test_topological_order_does_not_emit_unknown_names(
    default_set: CausalVariableSet,
) -> None:
    """Every name in the order must be a real variable in the set."""
    order = default_set.topological_order()
    all_names = set(default_set.get_all_names())
    assert set(order).issubset(all_names)


def test_add_variable_extends_set() -> None:
    """``add_variable`` makes a new variable retrievable by name."""
    s = CausalVariableSet()  # default-initialised
    initial = len(s)
    custom = CausalVariable(
        name="custom_test_only",
        display_name="Custom",
        var_type=VariableType.CONTINUOUS,
        domain=(0.0, 1.0),
    )
    s.add_variable(custom)
    assert s.get_variable("custom_test_only") is custom
    assert len(s) == initial + 1


def test_iter_yields_variable_objects(default_set: CausalVariableSet) -> None:
    """Iterating the set yields ``CausalVariable`` instances, not names."""
    seen = list(default_set)
    assert seen, "iterating must yield at least one variable"
    assert all(isinstance(v, CausalVariable) for v in seen)


# ---------------------------------------------------------------------------
# compute_variable_from_state: integrates with the live MTGEnv
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_game_state():
    """Build a real game state via MTGEnv so we exercise the full path."""
    from mtg.env import MTGEnv

    env = MTGEnv(
        deck_archetype="mono_red_aggro",
        opponent_archetype="mono_red_aggro",
        max_turns=10,
        seed=0,
    )
    env.reset()
    return env.state


def test_compute_variable_from_state_returns_float_for_known_keys(
    fresh_game_state,
) -> None:
    """The mapped SCM keys never raise and always return a float."""
    for key in ("mana_t", "card_advantage", "board_pressure", "tempo", "life_buffer"):
        value = compute_variable_from_state(key, fresh_game_state, player_id=0)
        assert isinstance(value, float)


def test_compute_variable_from_state_synthesised_keys(fresh_game_state) -> None:
    """Synthesised variables: own_life, land_in_hand, own_creatures."""
    own_life = compute_variable_from_state("own_life", fresh_game_state, player_id=0)
    land_in_hand = compute_variable_from_state("land_in_hand", fresh_game_state, player_id=0)
    own_creatures = compute_variable_from_state("own_creatures", fresh_game_state, player_id=0)
    assert own_life > 0
    assert land_in_hand >= 0
    assert own_creatures >= 0


def test_compute_variable_from_state_unknown_returns_zero(fresh_game_state) -> None:
    """Unknown variable name must return 0.0 as a graceful fallback (not raise)."""
    assert compute_variable_from_state("totally_made_up", fresh_game_state, player_id=0) == 0.0

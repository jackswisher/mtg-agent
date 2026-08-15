"""Tests for the seeded stochastic opponent mulligan policy.

The opponent mulligan policy
:meth:`mtg.env.mtg_env.MTGEnv._decide_opponent_mulligan_keep`:

* Returns ``True`` deterministically for hands with <=1 card or after 3
  prior mulligans (no further mulligans are available under London rules).
* Otherwise samples ``rng.random() < keep_prob`` against a hand-quality
  derived ``keep_prob``.
* Uses ``self._np_random`` so play stays bit-exactly reproducible per
  ``reset(seed=...)``.

These tests exercise the helper directly (rather than running full
episodes) because the env's mulligan path interleaves with many other
systems and narrow assertions stay fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from mtg.env import MTGEnv
from mtg.env.card_definitions import CardType, ManaCost
from mtg.env.rules import GamePhase


@pytest.fixture
def env() -> MTGEnv:
    """Build a seeded env fixed to a deterministic deck for fast tests."""
    e = MTGEnv(deck_archetype="mono_red_aggro", seed=12345, max_turns=5)
    e.reset(seed=12345)
    return e


# ---------------------------------------------------------------------------
# Hand fixtures
# ---------------------------------------------------------------------------


class _StubCard:
    """Minimal stand-in for a real Card.

    We only need ``card_type`` and ``mana_cost.cmc`` for the mulligan
    heuristic, so we keep this lightweight to avoid pulling the full
    ``Card`` constructor (which requires a registry lookup).
    """

    def __init__(self, card_type: CardType, cmc: float) -> None:
        self.card_type = card_type
        self.mana_cost = ManaCost(generic=int(cmc))


def _land() -> _StubCard:
    return _StubCard(CardType.LAND, 0.0)


def _spell(cmc: float) -> _StubCard:
    return _StubCard(CardType.CREATURE, cmc)


def _set_opponent_hand(env: MTGEnv, hand: list[_StubCard]) -> None:
    """Replace the opponent's current hand with ``hand`` in-place."""
    assert env.state is not None
    env.state.players[1].hand = list(hand)


# ---------------------------------------------------------------------------
# Sanity: helper exists and basic forced-keep cases
# ---------------------------------------------------------------------------


def test_helper_returns_true_for_ultra_low_hand_size(env: MTGEnv) -> None:
    """A 1-card opener has no further mulligans worth taking."""
    _set_opponent_hand(env, [_land()])
    assert env._decide_opponent_mulligan_keep() is True


def test_helper_returns_true_after_three_mulligans(env: MTGEnv) -> None:
    """Once we've used 3 mulligans we must keep (London-mulligan cliff)."""
    assert env.state is not None
    env.state.mulligan_count[1] = 3
    _set_opponent_hand(env, [_land()] * 4)
    assert env._decide_opponent_mulligan_keep() is True


def test_helper_returns_true_for_five_card_hand(env: MTGEnv) -> None:
    """Marginal value of another mulligan goes negative below 6 cards."""
    _set_opponent_hand(
        env,
        [_land(), _land(), _spell(1), _spell(3), _spell(5)],
    )
    assert env._decide_opponent_mulligan_keep() is True


# ---------------------------------------------------------------------------
# Hand-quality scoring: mulligans bad hands more often than good hands
# ---------------------------------------------------------------------------


def _empirical_keep_rate(env: MTGEnv, hand: list[_StubCard], n: int = 4000) -> float:
    """Compute the empirical keep-rate of a particular opener.

    We pin the env's RNG to a fresh deterministic generator before the
    sample so the test is reproducible regardless of test ordering.
    """
    env._np_random = np.random.default_rng(98765)
    keeps = 0
    for _ in range(n):
        _set_opponent_hand(env, hand)
        if env._decide_opponent_mulligan_keep():
            keeps += 1
    return keeps / n


def test_zero_land_hand_is_almost_always_mulliganed(env: MTGEnv) -> None:
    """0-land hands should mulligan ~95% of the time (target ~5% keep)."""
    no_lands = [_spell(1), _spell(2), _spell(3), _spell(4), _spell(5), _spell(6), _spell(7)]
    rate = _empirical_keep_rate(env, no_lands)
    assert rate < 0.10, f"0-land hand kept {rate:.2%} of the time (expected <10%)"


def test_seven_land_hand_is_almost_always_mulliganed(env: MTGEnv) -> None:
    """All-land hands have no spells to cast and should be mulliganed."""
    all_lands = [_land()] * 7
    rate = _empirical_keep_rate(env, all_lands)
    assert rate < 0.10, f"7-land hand kept {rate:.2%} of the time (expected <10%)"


def test_three_land_hand_with_curve_is_almost_always_kept(env: MTGEnv) -> None:
    """3 lands + a 1/2-drop curve is the textbook MTG snap-keep."""
    good_keep = [
        _land(),
        _land(),
        _land(),
        _spell(1),
        _spell(2),
        _spell(3),
        _spell(4),
    ]
    rate = _empirical_keep_rate(env, good_keep)
    assert rate > 0.80, f"3-land curve hand kept only {rate:.2%} of the time (expected >80%)"


def test_one_land_hand_is_borderline(env: MTGEnv) -> None:
    """1-land + 2 early plays should be keepable but not snap-kept."""
    snap_keep_aggro = [
        _land(),
        _spell(1),
        _spell(2),
        _spell(2),
        _spell(4),
        _spell(5),
        _spell(6),
    ]
    rate = _empirical_keep_rate(env, snap_keep_aggro)
    assert 0.50 <= rate <= 0.95, (
        f"1-land aggro hand kept {rate:.2%} of the time; "
        "expected to be in the borderline band [50%, 95%]"
    )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_and_hand_gives_identical_decisions() -> None:
    """Two envs seeded the same way must produce identical decision sequences."""
    hand = [_land(), _land(), _spell(1), _spell(2), _spell(3), _spell(4), _spell(5)]
    decisions_a: list[bool] = []
    decisions_b: list[bool] = []
    for env_decisions in (decisions_a, decisions_b):
        env = MTGEnv(deck_archetype="mono_red_aggro", seed=777, max_turns=5)
        env.reset(seed=777)
        for _ in range(50):
            _set_opponent_hand(env, hand)
            env_decisions.append(env._decide_opponent_mulligan_keep())
    assert decisions_a == decisions_b


def test_different_seeds_produce_different_decision_sequences() -> None:
    """Two envs with different seeds should diverge somewhere in 50 trials."""
    hand = [_land(), _spell(1), _spell(2), _spell(2), _spell(4), _spell(5), _spell(6)]
    env_a = MTGEnv(deck_archetype="mono_red_aggro", seed=11, max_turns=5)
    env_a.reset(seed=11)
    env_b = MTGEnv(deck_archetype="mono_red_aggro", seed=22, max_turns=5)
    env_b.reset(seed=22)

    seq_a = [
        env_a._decide_opponent_mulligan_keep()
        for _ in range(50)
        if (_set_opponent_hand(env_a, hand) or True)
    ]
    seq_b = [
        env_b._decide_opponent_mulligan_keep()
        for _ in range(50)
        if (_set_opponent_hand(env_b, hand) or True)
    ]
    assert seq_a != seq_b, "Different seeds produced identical 50-decision sequences"


# ---------------------------------------------------------------------------
# Integration: env still completes a real episode with the new policy
# ---------------------------------------------------------------------------


def test_env_completes_episode_with_stochastic_mulligan() -> None:
    """End-to-end smoke: the env still progresses past the mulligan phase."""
    env = MTGEnv(deck_archetype="mono_red_aggro", seed=2026, max_turns=10)
    obs, info = env.reset(seed=2026)
    assert obs is not None
    assert env.state is not None
    # After reset() returns control to the player, either we're in a
    # priority phase or the opener was bottomed out; either way the
    # mulligan phase must have completed for the opponent (no infinite
    # loop, no exceptions).
    assert env.state.phase != GamePhase.MULLIGAN or env.state.priority_player == 0


def test_env_invokes_helper_when_opponent_has_mulligan_priority() -> None:
    """End-to-end: ``_decide_opponent_mulligan_keep`` is actually called.

    The env breaks out of the mulligan loop as soon as priority returns
    to player 0, so we have to *drive* player 0's mulligan decision via
    ``env.step()`` to push the engine into the opponent's mulligan path.
    A single keep-hand action by the player is enough; the env then
    runs the opponent through its mulligan, which must invoke our
    helper at least once.
    """
    from mtg.env.action_mask import ActionKind

    env = MTGEnv(deck_archetype="mono_red_aggro", seed=4242, max_turns=5)
    calls = {"n": 0}
    original = env._decide_opponent_mulligan_keep

    def _counting_helper() -> bool:
        calls["n"] += 1
        return original()

    env._decide_opponent_mulligan_keep = _counting_helper  # type: ignore[method-assign]

    for s in range(8):
        _, info = env.reset(seed=4242 + s)
        # If the env is in the mulligan phase, drive the player's
        # decision once so the opponent gets to act.
        if env.state is None or env.state.phase != GamePhase.MULLIGAN:
            continue
        meta = info.get("action_metadata") or env.action_builder.get_action_metadata(env.state)
        action_mask = info["action_mask"]
        keep_action: int | None = None
        for action_idx, m in meta.items():
            if not action_mask[action_idx]:
                continue
            if str(m.get("kind", "")) == ActionKind.KEEP_HAND.value:
                keep_action = action_idx
                break
        if keep_action is None:
            continue
        env.step(keep_action)

    assert calls["n"] >= 1, (
        "Opponent mulligan helper was never called across 8 resets+keeps; "
        "either the env is bypassing the mulligan phase or the helper "
        "wiring regressed."
    )

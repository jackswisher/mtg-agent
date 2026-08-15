"""Tests for the seeded stochastic tie-break in the heuristic base agent.

The base heuristic ``select_action`` breaks score-ties by sampling
uniformly among the tied best actions using the agent's seeded
``self.rng``. This avoids systematically favouring whichever tied
option happens to be earliest in the action schema and gives the
heuristic baseline genuine mirror-match variance.

These tests pin three properties:

* same seed -> identical action stream when scores are constructed to
  produce ties (reproducibility);
* over many random scorings, every tied action gets selected with
  approximately uniform frequency (no pinning to the lowest index);
* when only one action is best (no tie), the deterministic argmax path
  is used.
"""

from __future__ import annotations

import typing as tp
from collections import Counter

import numpy as np
import pytest

from mtg.agents.heuristics.heuristic_base_agent import HeuristicBaseAgent


class _ScoreStubAgent(HeuristicBaseAgent):
    """Heuristic that returns a caller-supplied score per action.

    Bypasses the entire MTG scoring pipeline by short-circuiting
    ``_build_context``, ``_first_action``, ``_best_land_action``, and
    ``_best_creature_action`` so the only path through ``select_action``
    is: build score dict -> tie-break.
    """

    def __init__(self, *, scores: dict[int, float], seed: int) -> None:
        super().__init__(name="score_stub", seed=seed)
        self._scores = dict(scores)

    def _first_action(self, legal, info, kind):
        return None

    def _build_context(self, info):
        return {}

    def _best_land_action(self, legal, info, context):
        return None

    def _best_creature_action(self, legal, info, context):
        return None

    def _score_action(self, action: int, info, context) -> float:
        return float(self._scores.get(int(action), 0.0))


def _make_info(*, hand_size: int = 7) -> dict[str, tp.Any]:
    """Minimal ``info`` dict that bypasses the mulligan and main-phase paths."""
    return {
        "hand_size": hand_size,
        "pending_action_type": "anything",
        "phase_enum": "END_OF_TURN",
        "stack_size": 0,
        "active_player": 0,
        "current_player_index": 1,
    }


def _legal_mask(n_actions: int, legal: list[int]) -> np.ndarray:
    mask = np.zeros(n_actions, dtype=np.int8)
    for a in legal:
        mask[a] = 1
    return mask


# ---------------------------------------------------------------------------
# Tie-break uniformity
# ---------------------------------------------------------------------------


def test_tie_break_distributes_uniformly_across_tied_best_actions() -> None:
    """All tied best actions must be selected with roughly uniform probability."""
    n_actions = 8
    tied = [2, 3, 5, 7]  # 4 tied options
    scores = {a: (10.0 if a in tied else 0.0) for a in range(n_actions)}
    agent = _ScoreStubAgent(scores=scores, seed=0)
    mask = _legal_mask(n_actions, list(range(n_actions)))
    obs = np.zeros(4, dtype=np.float32)

    n_trials = 4000
    counts: Counter[int] = Counter()
    for _ in range(n_trials):
        a = agent.select_action(obs, mask, _make_info())
        counts[int(a)] += 1

    # All non-tied actions must never be picked.
    for a in range(n_actions):
        if a not in tied:
            assert counts[a] == 0, f"non-tied action {a} was picked {counts[a]} times"

    expected = n_trials / len(tied)
    for a in tied:
        # Allow generous tolerance on a 4-way tie (35% slack).
        assert counts[a] == pytest.approx(expected, rel=0.35), (
            f"tied action {a} drawn {counts[a]} times, expected ~{expected}; "
            f"distribution: {dict(counts)}"
        )


def test_tie_break_is_not_pinned_to_lowest_action_index() -> None:
    """The lowest tied index must not dominate the sampling distribution."""
    n_actions = 6
    tied = [0, 1, 2, 3]
    scores = {a: (5.0 if a in tied else -1.0) for a in range(n_actions)}
    agent = _ScoreStubAgent(scores=scores, seed=123)
    mask = _legal_mask(n_actions, list(range(n_actions)))
    obs = np.zeros(4, dtype=np.float32)

    counts: Counter[int] = Counter()
    for _ in range(2000):
        a = agent.select_action(obs, mask, _make_info())
        counts[int(a)] += 1

    # The lowest index (0) should be roughly 1/4 of the trials, not 100%.
    share = counts[0] / 2000
    assert 0.15 < share < 0.35, (
        f"lowest tied index dominates the tie-break "
        f"(share={share:.3f}); expected ~0.25 for a 4-way tie"
    )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_tie_break_is_reproducible_for_same_seed() -> None:
    """Two agents sharing a seed produce the same action sequence on ties."""
    n_actions = 5
    tied = [1, 2, 4]
    scores = {a: (1.0 if a in tied else 0.0) for a in range(n_actions)}
    mask = _legal_mask(n_actions, list(range(n_actions)))
    obs = np.zeros(4, dtype=np.float32)

    agent_a = _ScoreStubAgent(scores=scores, seed=42)
    agent_b = _ScoreStubAgent(scores=scores, seed=42)
    seq_a = [int(agent_a.select_action(obs, mask, _make_info())) for _ in range(50)]
    seq_b = [int(agent_b.select_action(obs, mask, _make_info())) for _ in range(50)]
    assert seq_a == seq_b, "same-seed heuristic agents must produce identical tie-break streams"


def test_tie_break_diverges_for_different_seeds() -> None:
    """Different seeds eventually diverge on ties (sanity)."""
    n_actions = 5
    tied = [0, 2, 4]
    scores = {a: (1.0 if a in tied else 0.0) for a in range(n_actions)}
    mask = _legal_mask(n_actions, list(range(n_actions)))
    obs = np.zeros(4, dtype=np.float32)

    agent_a = _ScoreStubAgent(scores=scores, seed=1)
    agent_b = _ScoreStubAgent(scores=scores, seed=999)
    seq_a = [int(agent_a.select_action(obs, mask, _make_info())) for _ in range(50)]
    seq_b = [int(agent_b.select_action(obs, mask, _make_info())) for _ in range(50)]
    assert seq_a != seq_b, "different seeds must produce different tie-break streams"


# ---------------------------------------------------------------------------
# Single-best-action fast path
# ---------------------------------------------------------------------------


def test_no_tie_returns_the_single_best_action_deterministically() -> None:
    """When only one action has the max score, that action is always returned."""
    n_actions = 4
    scores = {0: 0.0, 1: 0.0, 2: 99.0, 3: 1.0}  # action 2 is uniquely best
    agent = _ScoreStubAgent(scores=scores, seed=7)
    mask = _legal_mask(n_actions, list(range(n_actions)))
    obs = np.zeros(4, dtype=np.float32)
    for _ in range(20):
        assert agent.select_action(obs, mask, _make_info()) == 2

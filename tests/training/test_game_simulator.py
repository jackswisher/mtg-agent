"""Unit and integration tests for ``mtg.simulation.game_simulator``.

The game simulator is the runner used by ``mtg-research case-study``,
the eval pipeline, and the ``run_gameplay`` script. Tests here
exercise:

* ``GameSimulator.setup`` and ``run_game`` end-to-end with two
  heuristics so the asserts are purely about *the simulator wiring*,
  not about agent learning quality.
* ``run_game`` / ``run_evaluation`` convenience functions.
* ``save_report`` round-trip (recorder -> HTML/JSON on disk).
* The action classifier helper because it powers the case-study
  step labels.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mtg.agents import get_agent
from mtg.simulation.game_simulator import (
    EvaluationResult,
    GameResult,
    GameSimulator,
    SimulationConfig,
    run_evaluation,
    run_game,
)

# ---------------------------------------------------------------------------
# Action classifier (pure function, fast)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Cast Lightning Strike", "CAST"),
        ("Play land Mountain", "PLAY_LAND"),
        ("Attack with Goblin", "ATTACK"),
        ("Block with Soldier", "BLOCK"),
        ("Draw card", "DRAW"),
        ("Mulligan", "MULLIGAN"),
        ("Pass priority", "PASS"),
        ("End turn", "PASS"),  # falls through to PASS
    ],
)
def test_classify_action_dispatches_on_keyword(name: str, expected: str) -> None:
    """The action-name keyword routes to the correct UI category."""
    sim = GameSimulator(
        config=SimulationConfig(visualize=False, record=False),
        player_agent=get_agent("random", seed=0),
    )
    assert sim._classify_action(name) == expected


# ---------------------------------------------------------------------------
# End-to-end: simulator drives a real game with two real heuristics
# ---------------------------------------------------------------------------


def test_simulator_run_game_terminates_with_valid_result() -> None:
    """One full game produces a GameResult with internally-consistent fields."""
    cfg = SimulationConfig(
        player_deck="mono_red_aggro",
        opponent_deck="mono_red_aggro",
        max_turns=8,
        seed=0,
        visualize=False,  # never sleep in tests
        record=True,
    )
    sim = GameSimulator(
        config=cfg,
        player_agent=get_agent("greedy_aggro", seed=0),
        opponent_agent=get_agent("random", seed=1),
    )
    result = sim.run_game()

    assert isinstance(result, GameResult)
    assert result.winner in {"Player", "Opponent", "Draw"}
    assert result.turns_played >= 1
    assert result.actions_taken >= 1
    assert result.recorder is not None  # we asked record=True
    # Life totals are integers within 0..40 (life can go negative for
    # commander-like edge cases, but in this 20-life env it should not).
    assert -100 <= result.player_life <= 40
    assert -100 <= result.opponent_life <= 40


def test_simulator_lazy_setup() -> None:
    """``run_game`` without an explicit ``setup()`` still constructs the env."""
    sim = GameSimulator(
        config=SimulationConfig(max_turns=3, visualize=False, record=False, seed=7),
        player_agent=get_agent("random", seed=7),
    )
    assert sim.env is None
    sim.run_game()
    assert sim.env is not None


def test_simulator_disabled_recorder_returns_none() -> None:
    """``record=False`` means no recorder is attached to the result."""
    cfg = SimulationConfig(max_turns=3, visualize=False, record=False, seed=11)
    sim = GameSimulator(
        config=cfg,
        player_agent=get_agent("random", seed=11),
    )
    result = sim.run_game()
    assert result.recorder is None
    assert result.game_id == ""


def test_simulator_save_report_writes_html_and_json(tmp_path: Path) -> None:
    """``save_report`` produces both replay.html and replay.json on disk."""
    cfg = SimulationConfig(max_turns=4, visualize=False, record=True, seed=3)
    sim = GameSimulator(
        config=cfg,
        player_agent=get_agent("greedy_aggro", seed=3),
    )
    result = sim.run_game()
    out_dir = sim.save_report(result, tmp_path / "reports")
    assert out_dir.is_dir()
    assert (out_dir / "replay.html").is_file()
    assert (out_dir / "replay.json").is_file()
    # And the HTML file is non-trivial (not an empty stub).
    assert (out_dir / "replay.html").stat().st_size > 200


def test_simulator_save_report_without_recorder_raises() -> None:
    """Saving when ``record=False`` must raise a clear ValueError."""
    cfg = SimulationConfig(max_turns=2, visualize=False, record=False, seed=5)
    sim = GameSimulator(
        config=cfg,
        player_agent=get_agent("random", seed=5),
    )
    result = sim.run_game()
    with pytest.raises(ValueError, match="not recorded"):
        sim.save_report(result, "/tmp/should_not_exist")


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def test_run_game_convenience_wrapper_returns_game_result() -> None:
    """``run_game(...)`` builds the simulator + returns a single GameResult."""
    result = run_game(
        player_agent=get_agent("greedy_aggro", seed=0),
        player_deck="mono_red_aggro",
        opponent_deck="mono_red_aggro",
        max_turns=6,
        seed=0,
        visualize=False,
        record=False,
    )
    assert isinstance(result, GameResult)
    assert result.winner in {"Player", "Opponent", "Draw"}


def test_run_evaluation_aggregates_across_seeds() -> None:
    """``run_evaluation`` plays one game per seed and aggregates correctly."""
    agent = get_agent("greedy_aggro", seed=0)
    result = run_evaluation(
        agent=agent,
        player_deck="mono_red_aggro",
        opponent_deck="mono_red_aggro",
        num_games=3,
        seeds=[0, 1, 2],
        show_progress=False,
    )
    assert isinstance(result, EvaluationResult)
    assert result.num_games == 3
    assert result.wins + result.losses + result.draws == result.num_games
    assert 0.0 <= result.win_rate <= 1.0
    assert result.avg_turns >= 1.0

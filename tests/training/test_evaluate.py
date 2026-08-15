"""Tests for ``mtg.training.evaluate`` (was 32% covered).

The evaluator is the single source of truth for win-rate / reward /
length statistics across training, the research pipeline, and the
ablation suite.  Bugs in the bootstrap CI or per-seed decomposition
would invalidate every paper plot.

We cover:

* ``bootstrap_ci`` and ``bootstrap_half_width`` -- mean recovery,
  edge cases (empty / single value), determinism with a seed.
* ``EvaluationConfig.from_yaml`` and ``env_config`` mapping.
* ``EvaluationResult.summary`` / ``to_dict`` JSON round-trip.
* ``Evaluator.evaluate`` end-to-end with a real heuristic agent.
* ``compare_agents`` per-seed decomposition (means *over seeds*).
* ``_is_win`` truth table for the variety of ``info`` shapes the env
  emits at termination.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from mtg.agents import get_agent
from mtg.training.evaluate import (
    EvaluationConfig,
    EvaluationResult,
    Evaluator,
    _is_win,
    bootstrap_ci,
    bootstrap_half_width,
    compare_agents,
    evaluate,
)

# ---------------------------------------------------------------------------
# Bootstrap utilities
# ---------------------------------------------------------------------------


def test_bootstrap_ci_recovers_mean_for_constant_input() -> None:
    """All-equal input => mean == lower == upper, regardless of seed."""
    mean, lo, hi = bootstrap_ci([0.5] * 10, n_bootstrap=50, seed=0)
    assert mean == lo == hi == 0.5


def test_bootstrap_ci_handles_empty_and_single_value() -> None:
    """Edge cases: empty input -> all zeros, singleton -> degenerate CI."""
    assert bootstrap_ci([], n_bootstrap=10) == (0.0, 0.0, 0.0)
    assert bootstrap_ci([0.42], n_bootstrap=10) == (0.42, 0.42, 0.42)


def test_bootstrap_ci_is_deterministic_with_seed() -> None:
    """Same input + same seed -> identical CI on every call."""
    values = [0.0, 0.1, 0.5, 0.7, 0.9, 1.0]
    a = bootstrap_ci(values, n_bootstrap=200, seed=123)
    b = bootstrap_ci(values, n_bootstrap=200, seed=123)
    assert a == b


def test_bootstrap_half_width_is_nonnegative_and_symmetric() -> None:
    """Half-width is non-negative and the CI brackets the sample mean."""
    values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hw = bootstrap_half_width(values, n_bootstrap=200, seed=7)
    assert hw >= 0.0
    # And it actually contains the sample mean.
    mean, lo, hi = bootstrap_ci(values, n_bootstrap=200, seed=7)
    assert lo - 1e-9 <= mean <= hi + 1e-9


# ---------------------------------------------------------------------------
# EvaluationConfig + EvaluationResult
# ---------------------------------------------------------------------------


def test_evaluation_config_from_yaml_round_trip(tmp_path: Path) -> None:
    """``EvaluationConfig.from_yaml`` parses every field correctly."""
    cfg_path = tmp_path / "eval.yaml"
    payload = {
        "deck_archetype": "azorius_control",
        "opponent_archetype": "mono_red_aggro",
        "n_episodes": 7,
        "seed": 13,
    }
    cfg_path.write_text(yaml.safe_dump(payload))
    cfg = EvaluationConfig.from_yaml(cfg_path)
    assert cfg.deck_archetype == "azorius_control"
    assert cfg.opponent_archetype == "mono_red_aggro"
    assert cfg.n_episodes == 7
    assert cfg.seed == 13


def test_evaluation_config_env_config_carries_decks() -> None:
    """``env_config()`` propagates deck names + seed to the inner ``EnvConfig``."""
    cfg = EvaluationConfig(
        deck_archetype="mono_red_aggro",
        opponent_archetype="azorius_control",
        seed=99,
    )
    env_cfg = cfg.env_config()
    assert env_cfg.player_deck == "mono_red_aggro"
    assert env_cfg.opponent_deck == "azorius_control"
    assert env_cfg.seed == 99


def test_evaluation_config_env_config_defaults_opponent_to_self() -> None:
    """When ``opponent_archetype`` is ``None`` we mirror the player."""
    cfg = EvaluationConfig(deck_archetype="mono_red_aggro", opponent_archetype=None)
    assert cfg.env_config().opponent_deck == "mono_red_aggro"


def test_evaluation_result_summary_and_dict_round_trip() -> None:
    """``summary()`` is human-readable; ``to_dict()`` is JSON-serialisable."""
    result = EvaluationResult(
        n_episodes=5,
        win_rate=0.4,
        mean_reward=1.5,
        std_reward=0.3,
        mean_length=12.0,
        std_length=2.1,
        wins=[True, False, True, False, False],
        rewards=[1.0, 2.0, 1.5, 1.5, 1.5],
        lengths=[10, 14, 12, 11, 13],
        win_rate_ci95=0.05,
    )
    summary = result.summary()
    assert "Win Rate: 40.00%" in summary
    assert "Mean Reward" in summary

    d = result.to_dict()
    # The dict must be JSON-serialisable; every downstream aggregator
    # depends on this contract.
    text = json.dumps(d)
    parsed = json.loads(text)
    assert parsed["win_rate"] == pytest.approx(0.4)
    assert parsed["n_episodes"] == 5


# ---------------------------------------------------------------------------
# _is_win helper: the various info shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("info", "expected"),
    [
        ({"game_result": "win"}, True),
        ({"game_result": "loss"}, False),
        ({"game_result": "draw"}, False),
        ({"winner": "player"}, True),
        ({"winner": "opponent"}, False),
        ({"winner": 0}, True),
        ({"winner": 1}, False),
        ({"terminal_info": {"game_result": "win"}}, True),
        ({}, False),
    ],
)
def test_is_win_truth_table(info: dict, expected: bool) -> None:
    """``_is_win`` handles every shape of ``info`` we observe at termination."""
    assert _is_win(info) is expected


# ---------------------------------------------------------------------------
# Evaluator end-to-end
# ---------------------------------------------------------------------------


def test_evaluator_runs_and_produces_consistent_result() -> None:
    """Tiny end-to-end run on the real env with a heuristic agent."""
    cfg = EvaluationConfig(
        deck_archetype="mono_red_aggro",
        opponent_archetype="mono_red_aggro",
        max_turns=6,
        max_steps_per_episode=120,
        n_episodes=4,
        seed=0,
        deterministic=True,
    )
    evaluator = Evaluator(cfg)
    agent = get_agent("greedy_aggro", seed=0)
    result = evaluator.evaluate(agent, progress_bar=False)

    assert result.n_episodes == 4
    assert len(result.rewards) == 4
    assert len(result.wins) == 4
    assert 0.0 <= result.win_rate <= 1.0
    # Bootstrap CI half-widths must be non-negative.
    assert result.win_rate_ci95 >= 0.0
    assert result.reward_ci95 >= 0.0
    assert result.length_ci95 >= 0.0


def test_module_evaluate_helper_uses_evaluator_under_the_hood() -> None:
    """Single-call ``evaluate(...)`` returns the same shape as ``Evaluator``."""
    agent = get_agent("greedy_aggro", seed=0)
    result = evaluate(
        agent,
        n_episodes=2,
        deck_archetype="mono_red_aggro",
        opponent_archetype="mono_red_aggro",
        max_turns=4,
        max_steps_per_episode=80,
        seed=42,
        deterministic=True,
    )
    assert isinstance(result, EvaluationResult)
    assert result.n_episodes == 2


def test_compare_agents_aggregates_per_seed_means() -> None:
    """``compare_agents`` returns per-seed lists with length ``n_seeds``."""
    cfg = EvaluationConfig(
        deck_archetype="mono_red_aggro",
        opponent_archetype="mono_red_aggro",
        max_turns=4,
        max_steps_per_episode=80,
        n_episodes=2,
        seed=0,
    )
    agents = {
        "random": get_agent("random", seed=0),
        "greedy": get_agent("greedy_aggro", seed=0),
    }
    out = compare_agents(agents, cfg, n_seeds=2)
    assert set(out) == {"random", "greedy"}
    for name, res in out.items():
        assert len(res.per_seed_win_rates) == 2, name
        assert len(res.per_seed_rewards) == 2, name
        assert len(res.per_seed_lengths) == 2, name
        # The reported win_rate is the mean *over seeds*, so it must
        # equal np.mean(per_seed_win_rates).
        assert res.win_rate == pytest.approx(np.mean(res.per_seed_win_rates))

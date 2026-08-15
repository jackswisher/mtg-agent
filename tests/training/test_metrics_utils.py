"""Unit tests for ``mtg.utils.metrics``.

This module is small but is used by the league system, the evaluator,
and the research aggregator. We cover all public surfaces:
``compute_metrics``, ``compute_win_rate``, ``compute_sample_efficiency``,
``MetricsTracker``, ``RollingAverage``, and ``compute_elo_rating``.
"""

from __future__ import annotations

import numpy as np
import pytest

from mtg.utils.metrics import (
    MetricsSummary,
    MetricsTracker,
    RollingAverage,
    compute_elo_rating,
    compute_metrics,
    compute_sample_efficiency,
    compute_win_rate,
)

# ---------------------------------------------------------------------------
# compute_metrics + MetricsSummary
# ---------------------------------------------------------------------------


def test_compute_metrics_basic_values() -> None:
    """``compute_metrics`` returns the expected mean/std/min/max/count."""
    summary = compute_metrics([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary.mean == pytest.approx(3.0)
    assert summary.min == 1.0
    assert summary.max == 5.0
    assert summary.count == 5
    # Population std (np.std default), not the unbiased ddof=1 estimator.
    assert summary.std == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0, 5.0]))


def test_compute_metrics_empty_returns_zeros() -> None:
    """An empty input must not raise; all fields default to 0."""
    summary = compute_metrics([])
    assert summary == MetricsSummary(mean=0.0, std=0.0, min=0.0, max=0.0, count=0)


def test_metrics_summary_to_dict_round_trips() -> None:
    """``MetricsSummary.to_dict`` exposes every field as a JSON-friendly value."""
    summary = compute_metrics([2.0, 4.0])
    d = summary.to_dict()
    assert set(d) == {"mean", "std", "min", "max", "count"}
    assert d["mean"] == pytest.approx(3.0)
    assert d["count"] == 2


# ---------------------------------------------------------------------------
# compute_win_rate
# ---------------------------------------------------------------------------


def test_compute_win_rate_proportion() -> None:
    """Win rate is the simple proportion of ``True`` entries."""
    assert compute_win_rate([True, True, True, False]) == pytest.approx(0.75)


def test_compute_win_rate_empty_returns_zero() -> None:
    """Empty wins list must return 0 (not raise ``ZeroDivisionError``)."""
    assert compute_win_rate([]) == 0.0


# ---------------------------------------------------------------------------
# compute_sample_efficiency
# ---------------------------------------------------------------------------


def test_compute_sample_efficiency_returns_episode_index() -> None:
    """The first window whose mean clears the threshold returns the index."""
    rewards = [0.0] * 50 + [1.0] * 60  # rolling mean crosses 0.5 around idx ~80
    out = compute_sample_efficiency(rewards, threshold=0.5, window=10)
    assert out is not None
    assert isinstance(out, int)
    # Inclusive index so it must be after the warm-up.
    assert out > 50


def test_compute_sample_efficiency_returns_none_if_threshold_unreached() -> None:
    """Threshold never reached -> return ``None`` to signal no convergence."""
    rewards = [0.0] * 200
    assert compute_sample_efficiency(rewards, threshold=1.0, window=20) is None


def test_compute_sample_efficiency_returns_none_if_too_few_episodes() -> None:
    """Fewer episodes than the smoothing window -> not enough data."""
    assert compute_sample_efficiency([1.0] * 5, threshold=0.5, window=10) is None


# ---------------------------------------------------------------------------
# MetricsTracker
# ---------------------------------------------------------------------------


def test_metrics_tracker_add_and_summary() -> None:
    """Adding values then querying ``get_summary`` returns aggregate stats."""
    tracker = MetricsTracker()
    for v in (1.0, 2.0, 3.0):
        tracker.add("reward", v)
    summary = tracker.get_summary("reward")
    assert summary.mean == pytest.approx(2.0)
    assert summary.count == 3


def test_metrics_tracker_add_many_distributes_keys() -> None:
    """``add_many`` routes each ``{name: value}`` pair to its own buffer."""
    tracker = MetricsTracker()
    tracker.add_many({"a": 1.0, "b": 10.0})
    tracker.add_many({"a": 3.0, "b": 30.0})
    assert tracker.get_summary("a").mean == pytest.approx(2.0)
    assert tracker.get_summary("b").mean == pytest.approx(20.0)


def test_metrics_tracker_get_recent_truncates_to_window() -> None:
    """``get_recent(n=k)`` only summarises the most recent ``k`` values."""
    tracker = MetricsTracker()
    for v in range(10):
        tracker.add("x", float(v))
    recent = tracker.get_recent("x", n=3)
    assert recent.count == 3
    assert recent.mean == pytest.approx((7 + 8 + 9) / 3)


def test_metrics_tracker_step_and_clear() -> None:
    """``step()`` increments the step counter; ``clear()`` resets everything."""
    tracker = MetricsTracker()
    tracker.add("loss", 0.5)
    tracker.step()
    tracker.step()
    assert tracker.current_step == 2

    tracker.clear()
    assert tracker.current_step == 0
    assert tracker.get_summary("loss").count == 0


def test_metrics_tracker_get_values_returns_copy() -> None:
    """Mutating the returned list must not affect the tracker."""
    tracker = MetricsTracker()
    tracker.add("a", 1.0)
    out = tracker.get_values("a")
    out.append(99.0)
    assert tracker.get_summary("a").count == 1


def test_metrics_tracker_unknown_metric_returns_empty_summary() -> None:
    """Querying a never-added metric must not raise."""
    tracker = MetricsTracker()
    s = tracker.get_summary("nope")
    assert s.count == 0
    assert s.mean == 0.0


# ---------------------------------------------------------------------------
# RollingAverage
# ---------------------------------------------------------------------------


def test_rolling_average_window_truncation() -> None:
    """Adding more than ``window`` values keeps only the most recent."""
    roll = RollingAverage(window=3)
    for v in (1.0, 2.0, 3.0, 4.0, 5.0):
        roll.add(v)
    # Only the last 3 values (3, 4, 5) are kept.
    assert roll.mean() == pytest.approx(4.0)


def test_rolling_average_empty_state() -> None:
    """A fresh ``RollingAverage`` returns 0.0 for both mean and std."""
    roll = RollingAverage(window=10)
    assert roll.mean() == 0.0
    assert roll.std() == 0.0


def test_rolling_average_std_requires_two_samples() -> None:
    """``std()`` returns 0 with one sample and the true std with two+."""
    roll = RollingAverage(window=10)
    roll.add(1.0)
    assert roll.std() == 0.0  # n < 2
    roll.add(3.0)
    assert roll.std() == pytest.approx(np.std([1.0, 3.0]))


# ---------------------------------------------------------------------------
# compute_elo_rating
# ---------------------------------------------------------------------------


def test_elo_winner_gains_loser_loses_symmetric() -> None:
    """A single win between equal-rated players moves them by the same amount."""
    initial = 1500.0
    ratings = compute_elo_rating(
        [(0, 1, 1.0)],  # player 0 wins vs player 1
        k_factor=32.0,
        initial_rating=initial,
    )
    delta_winner = ratings[0] - initial
    delta_loser = initial - ratings[1]
    assert delta_winner > 0
    assert delta_winner == pytest.approx(delta_loser)


def test_elo_draw_between_equal_players_is_no_op() -> None:
    """A draw between two equal-rated players leaves both ratings unchanged."""
    initial = 1500.0
    ratings = compute_elo_rating(
        [(0, 1, 0.5)],
        k_factor=32.0,
        initial_rating=initial,
    )
    assert ratings[0] == pytest.approx(initial)
    assert ratings[1] == pytest.approx(initial)


def test_elo_repeated_wins_monotonically_increase_rating() -> None:
    """A player that wins every game must end with the highest rating."""
    games = [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)]
    ratings = compute_elo_rating(games)
    assert ratings[0] > ratings[1]
    assert ratings[0] > ratings[2]
    assert ratings[0] > ratings[3]

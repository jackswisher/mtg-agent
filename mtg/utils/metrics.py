"""Metrics computation and tracking utilities.

This module provides metrics computation for evaluating
agent performance in the MTG environment.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class MetricsSummary:
    """Summary of computed metrics."""

    mean: float
    std: float
    min: float
    max: float
    count: int

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "count": self.count,
        }


def compute_metrics(values: list[float]) -> MetricsSummary:
    """Compute summary statistics for a list of values.

    Args:
        values: List of numeric values.

    Returns:
        MetricsSummary with statistics.
    """
    if not values:
        return MetricsSummary(mean=0.0, std=0.0, min=0.0, max=0.0, count=0)

    arr = np.array(values)
    return MetricsSummary(
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        count=len(arr),
    )


def compute_win_rate(wins: list[bool]) -> float:
    """Compute win rate from list of win indicators.

    Args:
        wins: List of boolean win indicators.

    Returns:
        Win rate (0-1).
    """
    if not wins:
        return 0.0
    return sum(wins) / len(wins)


def compute_sample_efficiency(
    rewards: list[float],
    threshold: float,
    window: int = 100,
) -> int | None:
    """Compute sample efficiency (timesteps to reach threshold).

    Args:
        rewards: List of episode rewards.
        threshold: Reward threshold to reach.
        window: Smoothing window size.

    Returns:
        Number of episodes to reach threshold, or None if not reached.
    """
    if len(rewards) < window:
        return None

    for i in range(window, len(rewards) + 1):
        window_mean = np.mean(rewards[i - window : i])
        if window_mean >= threshold:
            return i

    return None


class MetricsTracker:
    """Tracker for accumulating metrics over time."""

    def __init__(self):
        """Initialize the metrics tracker."""
        self._metrics: dict[str, list[float]] = defaultdict(list)
        self._step = 0

    def add(self, name: str, value: float) -> None:
        """Add a value to a metric.

        Args:
            name: Metric name.
            value: Value to add.
        """
        self._metrics[name].append(value)

    def add_many(self, metrics: dict[str, float]) -> None:
        """Add multiple metrics at once.

        Args:
            metrics: Dictionary of metric names to values.
        """
        for name, value in metrics.items():
            self.add(name, value)

    def get_summary(self, name: str) -> MetricsSummary:
        """Get summary for a metric.

        Args:
            name: Metric name.

        Returns:
            MetricsSummary for the metric.
        """
        return compute_metrics(self._metrics.get(name, []))

    def get_recent(
        self,
        name: str,
        n: int = 100,
    ) -> MetricsSummary:
        """Get summary for recent values of a metric.

        Args:
            name: Metric name.
            n: Number of recent values to include.

        Returns:
            MetricsSummary for recent values.
        """
        values = self._metrics.get(name, [])
        recent = values[-n:] if len(values) > n else values
        return compute_metrics(recent)

    def get_all_summaries(self) -> dict[str, MetricsSummary]:
        """Get summaries for all metrics.

        Returns:
            Dictionary of metric names to summaries.
        """
        return {name: self.get_summary(name) for name in self._metrics}

    def get_values(self, name: str) -> list[float]:
        """Get all values for a metric.

        Args:
            name: Metric name.

        Returns:
            List of values.
        """
        return self._metrics.get(name, []).copy()

    def clear(self) -> None:
        """Clear all tracked metrics."""
        self._metrics.clear()
        self._step = 0

    def step(self) -> None:
        """Increment the step counter."""
        self._step += 1

    @property
    def current_step(self) -> int:
        """Get current step."""
        return self._step


class RollingAverage:
    """Compute rolling average of values."""

    def __init__(self, window: int = 100):
        """Initialize rolling average.

        Args:
            window: Window size.
        """
        self.window = window
        self._values: list[float] = []

    def add(self, value: float) -> None:
        """Add a value.

        Args:
            value: Value to add.
        """
        self._values.append(value)
        if len(self._values) > self.window:
            self._values.pop(0)

    def mean(self) -> float:
        """Get current mean.

        Returns:
            Rolling mean.
        """
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    def std(self) -> float:
        """Get current standard deviation.

        Returns:
            Rolling std.
        """
        if len(self._values) < 2:
            return 0.0
        return float(np.std(self._values))


def compute_elo_rating(
    results: list[tuple[int, int, float]],
    k_factor: float = 32.0,
    initial_rating: float = 1500.0,
) -> dict[int, float]:
    """Compute Elo ratings from match results.

    Args:
        results: List of (player1_id, player2_id, result) tuples.
                 Result is 1.0 for player1 win, 0.0 for loss, 0.5 for draw.
        k_factor: Elo K-factor.
        initial_rating: Initial rating for new players.

    Returns:
        Dictionary of player IDs to Elo ratings.
    """
    ratings: dict[int, float] = {}

    def get_rating(player: int) -> float:
        return ratings.get(player, initial_rating)

    def expected_score(rating_a: float, rating_b: float) -> float:
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    for p1, p2, result in results:
        r1 = get_rating(p1)
        r2 = get_rating(p2)

        e1 = expected_score(r1, r2)
        e2 = expected_score(r2, r1)

        ratings[p1] = r1 + k_factor * (result - e1)
        ratings[p2] = r2 + k_factor * ((1 - result) - e2)

    return ratings

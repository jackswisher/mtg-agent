"""Tests for ``scripts.runner.run_training.evaluate_agent``.

The CLI training entry-point (``mtg-train``, ``run_training.py``)
threads a ``base_seed`` through ``evaluate_agent`` and calls
``env.reset(seed=base_seed + episode_idx)`` for every episode. This
matches the formal :class:`mtg.training.evaluate.Evaluator` behaviour
and is required for "fixed eval set" claims.

This test pins four contracts:

* same ``base_seed`` -> identical episode-level statistics on a stub
  env that mirrors the real reset/step API;
* different ``base_seed`` -> different reset seeds threaded through;
* ``base_seed=None`` -> no ``seed=`` kwarg in the reset call
  (non-deterministic path);
* the optional ``obs_normaliser`` callable is invoked on every
  observation so VecNormalize stats can be applied in CLI evaluation.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest

from scripts.runner.run_training import evaluate_agent

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _DummySpace:
    def __init__(self, n: int) -> None:
        self.n = n


class _RecordingEnv:
    """Env stub that records every reset seed it receives."""

    def __init__(self, *, action_dim: int = 4, max_steps: int = 3) -> None:
        self.observation_space = _DummySpace(8)
        self.action_space = _DummySpace(action_dim)
        self.max_steps = max_steps
        self.reset_seeds: list[int | None] = []
        self._step_count = 0
        self._rng = np.random.default_rng(0)

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        self.reset_seeds.append(seed)
        self._rng = np.random.default_rng(0 if seed is None else seed)
        self._step_count = 0
        obs = self._rng.standard_normal(self.observation_space.n).astype(np.float32)
        info = {"action_mask": np.ones(self.action_space.n, dtype=bool)}
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1
        terminated = self._step_count >= self.max_steps
        # Wins are deterministic given the seed: alternating by step.
        win = bool(self._rng.integers(0, 2))
        info = {
            "action_mask": np.ones(self.action_space.n, dtype=bool),
            "game_result": "win" if (terminated and win) else "loss",
        }
        obs = self._rng.standard_normal(self.observation_space.n).astype(np.float32)
        reward = float(self._rng.standard_normal())
        return obs, reward, terminated, False, info


class _DummyAgent:
    """Picks the first legal action; deterministic given a fixed env seed."""

    def __init__(self) -> None:
        self.deterministic = False

    def select_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        info: dict,
    ) -> int:
        legal = np.flatnonzero(action_mask)
        return int(legal[0])


# ---------------------------------------------------------------------------
# Per-episode seeding
# ---------------------------------------------------------------------------


def _seeds_passed(env: _RecordingEnv) -> list[int | None]:
    return list(env.reset_seeds)


def test_evaluate_agent_seeds_each_episode_with_base_plus_index() -> None:
    """``base_seed=K`` resets episode i with seed ``K+i`` (no exceptions)."""
    env = _RecordingEnv()
    agent = _DummyAgent()
    n_episodes = 4
    base_seed = 100
    evaluate_agent(env, agent, n_episodes=n_episodes, base_seed=base_seed)
    assert _seeds_passed(env) == [base_seed + i for i in range(n_episodes)]


def test_evaluate_agent_is_deterministic_given_same_base_seed() -> None:
    """Two independent runs with the same base_seed produce the same metrics."""
    env_a = _RecordingEnv()
    env_b = _RecordingEnv()
    agent_a = _DummyAgent()
    agent_b = _DummyAgent()
    res_a = evaluate_agent(env_a, agent_a, n_episodes=5, base_seed=42)
    res_b = evaluate_agent(env_b, agent_b, n_episodes=5, base_seed=42)
    assert res_a["win_rate"] == res_b["win_rate"]
    assert res_a["avg_reward"] == pytest.approx(res_b["avg_reward"])
    assert res_a["avg_episode_length"] == pytest.approx(res_b["avg_episode_length"])


def test_evaluate_agent_changes_with_different_base_seed() -> None:
    """Different ``base_seed`` produces a different reset-seed sequence."""
    env_a = _RecordingEnv()
    env_b = _RecordingEnv()
    agent_a = _DummyAgent()
    agent_b = _DummyAgent()
    evaluate_agent(env_a, agent_a, n_episodes=4, base_seed=10)
    evaluate_agent(env_b, agent_b, n_episodes=4, base_seed=99)
    assert _seeds_passed(env_a) != _seeds_passed(env_b)
    # Specifically, seeds should not overlap when bases are far apart.
    assert set(_seeds_passed(env_a)).isdisjoint(set(_seeds_passed(env_b)))


def test_evaluate_agent_records_base_seed_in_results() -> None:
    """The returned dict surfaces the ``base_seed`` used (None when unseeded)."""
    env_seeded = _RecordingEnv()
    env_unseeded = _RecordingEnv()
    agent = _DummyAgent()
    res_seeded = evaluate_agent(env_seeded, agent, n_episodes=2, base_seed=7)
    res_unseeded = evaluate_agent(env_unseeded, _DummyAgent(), n_episodes=2, base_seed=None)
    assert res_seeded["base_seed"] == 7
    assert res_unseeded["base_seed"] is None


def test_evaluate_agent_with_none_base_seed_does_not_pass_seed_kwarg() -> None:
    """``base_seed=None`` falls back to ``env.reset()`` without a seed."""
    env = _RecordingEnv()
    agent = _DummyAgent()
    evaluate_agent(env, agent, n_episodes=3, base_seed=None)
    assert _seeds_passed(env) == [None, None, None]


# ---------------------------------------------------------------------------
# Optional VecNormalize obs_normaliser plumbing
# ---------------------------------------------------------------------------


def test_evaluate_agent_applies_obs_normaliser_when_provided() -> None:
    """If ``obs_normaliser`` is given, it's called on every observation."""
    env = _RecordingEnv(max_steps=2)
    agent = _DummyAgent()

    seen: list[np.ndarray] = []

    def normaliser(obs: np.ndarray) -> np.ndarray:
        seen.append(np.array(obs, copy=True))
        return obs * 2.0

    evaluate_agent(env, agent, n_episodes=2, base_seed=0, obs_normaliser=normaliser)
    # 2 episodes x (1 reset + 2 step transitions) = 6 observations normalised.
    assert len(seen) == 6
    for obs in seen:
        assert obs.shape == (env.observation_space.n,)


def test_evaluate_agent_handles_missing_obs_normaliser_as_identity() -> None:
    """Without ``obs_normaliser`` the function must still work unchanged."""
    env = _RecordingEnv()
    agent = _DummyAgent()
    res = evaluate_agent(env, agent, n_episodes=3, base_seed=0)
    # win_rate is in [0, 1]; n_episodes is set; CI is non-negative.
    assert 0.0 <= res["win_rate"] <= 1.0
    assert res["n_episodes"] == 3
    assert res["win_rate_ci95"] >= 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_iterable(x: object) -> bool:
    return isinstance(x, Iterable)

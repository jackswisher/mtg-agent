"""Regression tests for the round-robin rotation-cadence fix.

The bug
-------
``_train_round_robin`` advertises ``Rotating opponent every 50 episodes``
but swaps are deferred to PPO rollout boundaries (intentional, to keep
GAE advantage estimates within a rollout opponent-homogeneous).  With
the legacy default ``n_steps = max(512, min(4096, total // 4))`` and
``n_envs = 15``, each rollout spans ``4096 * 15 = 61,440`` wall steps
(~900 MTG episodes), so the agent stays glued to a single opponent for
~900 episodes between swaps.  The console message lies and the
"round-robin" experiment degenerates into a few sequential mini-trainings.

The fix
-------
:func:`scripts.runner.run_training.create_agent` accepts a new
``is_round_robin`` flag.  When set, ``n_steps`` is auto-tuned so a
single rollout naturally contains ≈ :data:`RR_ROTATE_EVERY` episodes,
honoring the cadence promise:

    n_steps = max(RR_MIN_N_STEPS,
                  (RR_ROTATE_EVERY * RR_AVG_EP_LEN) // n_envs)

These tests pin three contracts:

1. With ``is_round_robin=True``, the resulting model's rollout
   (``n_steps * n_envs``) is at most ~2× the rotation target across the
   typical range of ``n_envs`` we use in production (1, 4, 8, 15, 32).
2. With ``is_round_robin=False`` (or default), the legacy formula is
   preserved so non-round-robin training is unaffected.
3. The bug as observed in production (``n_steps=4096`` with
   ``n_envs=15`` producing a 17×-too-large rollout) is fixed: the new
   path produces a rollout ≤ 2× the cadence target.
"""

from __future__ import annotations

import pytest

# Gymnasium is required by SB3 anyway; if it isn't installed we skip the
# entire module rather than fail with an import-time error.
gym = pytest.importorskip("gymnasium")
np = pytest.importorskip("numpy")

from gymnasium import spaces  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from scripts.runner.run_training import (  # noqa: E402
    RR_AVG_EP_LEN,
    RR_MIN_N_STEPS,
    RR_ROTATE_EVERY,
    create_agent,
)

# ---------------------------------------------------------------------------
# Test fixture: a minimal mask-aware env that lets MaskablePPO initialize
# without standing up the full MTG stack.
# ---------------------------------------------------------------------------


class _TinyMaskedEnv(gym.Env):
    """Minimal env exposing a Discrete action mask via ``action_masks``."""

    metadata = {"render_modes": []}

    def __init__(self, obs_dim: int = 4, action_dim: int = 3) -> None:
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(action_dim)
        self._t = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._t = 0
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action: int):
        self._t += 1
        done = self._t >= 16
        return (
            np.zeros(self.observation_space.shape, dtype=np.float32),
            0.0,
            done,
            False,
            {},
        )

    def action_masks(self) -> np.ndarray:
        return np.ones(self.action_space.n, dtype=bool)


def _make_vec(n_envs: int) -> DummyVecEnv:
    return DummyVecEnv([_TinyMaskedEnv for _ in range(n_envs)])


# ---------------------------------------------------------------------------
# Round-robin path: n_steps must size the rollout to ~rotate_every episodes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_envs", [1, 4, 8, 15, 32])
def test_create_agent_round_robin_tunes_rollout_to_cadence(n_envs: int) -> None:
    """A round-robin rollout must contain ≤ ~2× ``RR_ROTATE_EVERY`` episodes.

    Pre-fix: with ``n_envs=15`` the rollout was 61,440 steps ≈ 877
    episodes — 17× the ``rotate_every=50`` target, so swaps happened
    every 877 episodes instead of every 50.  This test would have
    failed with the legacy formula at any ``n_envs >= 1``.
    """
    env = _make_vec(n_envs)
    try:
        agent = create_agent(
            "ppo",
            env,
            seed=0,
            total_timesteps=200_000,  # large enough to trigger legacy n_steps=4096
            is_round_robin=True,
        )
        rollout_steps = agent.model.n_steps * env.num_envs
        rollout_eps = rollout_steps / RR_AVG_EP_LEN
        cadence_target = RR_ROTATE_EVERY

        # The rollout should be small enough that swaps fire on
        # (nearly) every rollout boundary, honoring the promise.
        assert rollout_eps <= 2 * cadence_target, (
            f"n_envs={n_envs}: rollout = {rollout_steps:,} steps "
            f"≈ {rollout_eps:.1f} episodes, but cadence target is "
            f"{cadence_target}. The 'every {cadence_target} episodes' "
            f"round-robin promise will be broken — rotation cadence "
            f"will be ~{rollout_eps:.0f} episodes instead."
        )

        # Also pin the floor so very-high ``n_envs`` configurations
        # don't degenerate into useless 1-step rollouts.
        assert agent.model.n_steps >= RR_MIN_N_STEPS, (
            f"n_steps={agent.model.n_steps} is below the round-robin "
            f"floor {RR_MIN_N_STEPS}; PPO updates will be too noisy."
        )
    finally:
        env.close()


def test_create_agent_round_robin_matches_explicit_formula() -> None:
    """Pin the exact formula used to size round-robin rollouts.

    Locks ``n_steps = max(RR_MIN_N_STEPS, RR_ROTATE_EVERY * RR_AVG_EP_LEN
    // n_envs)`` so any future tweak to the constants or formula has to
    be deliberate (and update this test).
    """
    env = _make_vec(15)  # production default for the paper sweeps
    try:
        agent = create_agent(
            "ppo",
            env,
            seed=0,
            total_timesteps=200_000,
            is_round_robin=True,
        )
        expected = max(RR_MIN_N_STEPS, (RR_ROTATE_EVERY * RR_AVG_EP_LEN) // 15)
        assert agent.model.n_steps == expected
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Default (sequential / single-opponent) path: legacy formula must hold
# ---------------------------------------------------------------------------


def test_create_agent_default_path_uses_legacy_n_steps_formula() -> None:
    """Non-round-robin runs must keep the legacy ``n_steps`` heuristic.

    Single-opponent training and round-robin training have very
    different optimal rollout sizes; this test ensures the round-robin
    fix doesn't accidentally shrink rollouts for the sequential / single-
    opponent case (which would slow learning).
    """
    env = _make_vec(15)
    try:
        agent = create_agent(
            "ppo",
            env,
            seed=0,
            total_timesteps=200_000,
            # is_round_robin defaults to False
        )
        legacy_expected = max(512, min(4096, 200_000 // 4))  # = 4096
        assert agent.model.n_steps == legacy_expected, (
            f"Default n_steps changed: got {agent.model.n_steps}, "
            f"expected {legacy_expected}. The round-robin auto-tune "
            f"must not affect the default code path."
        )
    finally:
        env.close()


def test_create_agent_round_robin_smaller_than_default_for_typical_n_envs() -> None:
    """The whole point of the fix: round-robin n_steps < default n_steps.

    A direct guard against silent regressions where someone might ``or``
    the two paths together by accident.
    """
    env = _make_vec(15)
    try:
        rr_agent = create_agent("ppo", env, seed=0, total_timesteps=200_000, is_round_robin=True)
        # Build a fresh env for the second agent so SB3 doesn't reuse state.
        env2 = _make_vec(15)
        try:
            default_agent = create_agent(
                "ppo", env2, seed=0, total_timesteps=200_000, is_round_robin=False
            )
            assert rr_agent.model.n_steps < default_agent.model.n_steps, (
                f"Round-robin n_steps ({rr_agent.model.n_steps}) must be "
                f"smaller than default n_steps ({default_agent.model.n_steps}) "
                f"so swaps fire on rollout boundaries — otherwise the "
                f"promised 'every {RR_ROTATE_EVERY} episodes' cadence is broken."
            )
        finally:
            env2.close()
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Production-bug regression: the EXACT scenario from the user's terminal
# ---------------------------------------------------------------------------


def test_production_scenario_no_longer_collapses_to_9_swaps() -> None:
    """Regression test for the user-observed bug.

    Production scenario observed in the terminal:
      * 3 opponents, 200k steps each → 600k total budget
      * 15 parallel envs (auto-detected on the user's machine)
      * Pre-fix: n_steps=4096 → rollout=61,440 → ~9 swaps total in 600k

    This test verifies that with ``is_round_robin=True`` the same
    600k-step budget now affords at least 50 swaps (more than 5× the
    pre-fix count), so each opponent gets meaningful interleaving.
    """
    env = _make_vec(15)
    try:
        agent = create_agent(
            "ppo",
            env,
            seed=0,
            total_timesteps=200_000,
            is_round_robin=True,
        )
        rollout_steps = agent.model.n_steps * env.num_envs
        total_budget = 600_000
        # One swap per rollout boundary (ignoring the final partial
        # rollout) — see ``_train_round_robin._on_rollout_end``.
        expected_swaps = max(0, total_budget // rollout_steps - 1)
        assert expected_swaps >= 50, (
            f"Pre-fix this scenario afforded ~9 swaps over 600k steps. "
            f"Post-fix we expect >=50, but rollout_steps={rollout_steps:,} "
            f"only allows ~{expected_swaps} swaps. The cadence fix has "
            f"regressed."
        )
    finally:
        env.close()

"""Tests for the CGFA env wrapper and rollout buffer."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
# ---------------------------------------------------------------------------
# Stub env (mirrors the bits CGFAEnvWrapper actually touches)
# ---------------------------------------------------------------------------
import gymnasium as gym  # noqa: E402
import torch as th  # noqa: E402
from gymnasium.spaces import Box, Discrete  # noqa: E402

from mtg.agents.reinforcement_learning.cgfa import (  # noqa: E402
    CGFAEnvWrapper,
    CGFAMaskableRolloutBuffer,
    FactorSpec,
)


class _StubEnv(gym.Env):
    """Tiny env that yields scripted causal_variables on each step."""

    metadata: dict = {"render_modes": []}

    def __init__(self, scripted: list[dict[str, float]]) -> None:
        super().__init__()
        self.scripted = scripted
        self.t = 0
        self.action_space = Discrete(2)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        info = {"causal_variables": dict(self.scripted[0])}
        return np.zeros(4, dtype=np.float32), info

    def step(self, action):
        self.t += 1
        cv = self.scripted[min(self.t, len(self.scripted) - 1)]
        info = {"causal_variables": dict(cv)}
        terminated = self.t >= len(self.scripted) - 1
        return (
            np.zeros(4, dtype=np.float32),
            float(action),
            terminated,
            False,
            info,
        )


def test_cgfa_env_wrapper_emits_factor_arrays() -> None:
    """Wrapper computes per-factor deltas and writes them to info."""
    spec = FactorSpec()
    env = _StubEnv(
        scripted=[
            {
                "card_adv": 0.0,
                "board_press": 0.0,
                "tempo": 0.0,
                "life_buffer": 0.0,
                "threat_density": 0.0,
                "removal_avail": 0.0,
            },
            {
                "card_adv": 1.0,
                "board_press": 2.0,
                "tempo": 0.1,
                "life_buffer": -1.0,
                "threat_density": 0.5,
                "removal_avail": 1.0,
            },
            {
                "card_adv": 1.5,
                "board_press": 1.0,
                "tempo": 0.0,
                "life_buffer": -2.0,
                "threat_density": 0.4,
                "removal_avail": 0.0,
            },
        ]
    )
    wrapped = CGFAEnvWrapper(env, factor_spec=spec)

    obs, info = wrapped.reset()
    assert info["factor_rewards"].shape == (spec.n_factors,)
    np.testing.assert_array_equal(info["factor_rewards"], np.zeros(spec.n_factors))

    obs, reward, term, trunc, info = wrapped.step(action=0)
    expected = np.array([1.0, 2.0, 0.1, -1.0, 0.5, 1.0], dtype=np.float32)
    np.testing.assert_allclose(info["factor_rewards"], expected, rtol=1e-5)

    obs, reward, term, trunc, info = wrapped.step(action=0)
    expected = np.array([0.5, -1.0, -0.1, -1.0, -0.1, -1.0], dtype=np.float32)
    np.testing.assert_allclose(info["factor_rewards"], expected, rtol=1e-5)
    assert term


def test_cgfa_env_wrapper_normalises_by_scale() -> None:
    """Per-factor scale divides the raw deltas element-wise."""
    scale = np.array([1.0, 2.0, 1.0, 4.0, 1.0, 1.0], dtype=np.float32)
    spec = FactorSpec(scale=scale)
    env = _StubEnv(
        scripted=[
            {
                "card_adv": 0.0,
                "board_press": 0.0,
                "tempo": 0.0,
                "life_buffer": 0.0,
                "threat_density": 0.0,
                "removal_avail": 0.0,
            },
            {
                "card_adv": 2.0,
                "board_press": 4.0,
                "tempo": 0.5,
                "life_buffer": 8.0,
                "threat_density": 0.1,
                "removal_avail": 0.0,
            },
        ]
    )
    wrapped = CGFAEnvWrapper(env, factor_spec=spec)
    wrapped.reset()
    _, _, _, _, info = wrapped.step(0)
    expected = np.array([2.0, 2.0, 0.5, 2.0, 0.1, 0.0], dtype=np.float32)
    np.testing.assert_allclose(info["factor_rewards"], expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# Buffer tests
# ---------------------------------------------------------------------------


def _make_buffer(
    buffer_size: int = 4, n_envs: int = 1, n_factors: int = 3
) -> CGFAMaskableRolloutBuffer:
    from gymnasium.spaces import Box, Discrete

    obs_space = Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
    act_space = Discrete(3)
    buf = CGFAMaskableRolloutBuffer(
        buffer_size=buffer_size,
        observation_space=obs_space,
        action_space=act_space,
        device="cpu",
        gae_lambda=1.0,
        gamma=1.0,
        n_envs=n_envs,
        n_factors=n_factors,
    )
    buf.reset()
    return buf


def _add_step(buf, *, reward, value, factor_rewards, factor_values, action_mask=None):
    obs = np.zeros((buf.n_envs, 2), dtype=np.float32)
    action = np.zeros((buf.n_envs,), dtype=np.int64)
    if action_mask is None:
        action_mask = np.ones((buf.n_envs, buf.mask_dims), dtype=np.float32)
    log_prob = th.zeros(buf.n_envs)
    val_t = th.tensor([[value]] * buf.n_envs, dtype=th.float32)
    buf.add(
        obs,
        action,
        np.array([reward] * buf.n_envs, dtype=np.float32),
        np.zeros((buf.n_envs,), dtype=np.float32),
        val_t,
        log_prob,
        action_masks=action_mask,
        factor_rewards=np.tile(factor_rewards.reshape(1, -1), (buf.n_envs, 1)),
        factor_values=th.tensor(
            np.tile(factor_values.reshape(1, -1), (buf.n_envs, 1)), dtype=th.float32
        ),
    )


def test_buffer_per_factor_gae_with_lambda_one_matches_monte_carlo() -> None:
    """With gamma=1, lambda=1, factor advantages = discounted sum - V_k(s_0)."""
    buf = _make_buffer(buffer_size=3, n_envs=1, n_factors=2)
    _add_step(
        buf,
        reward=0.0,
        value=0.0,
        factor_rewards=np.array([1.0, 0.5]),
        factor_values=np.array([0.0, 0.0]),
    )
    _add_step(
        buf,
        reward=0.0,
        value=0.0,
        factor_rewards=np.array([2.0, -0.5]),
        factor_values=np.array([0.0, 0.0]),
    )
    _add_step(
        buf,
        reward=0.0,
        value=0.0,
        factor_rewards=np.array([3.0, 1.0]),
        factor_values=np.array([0.0, 0.0]),
    )

    last_values = th.tensor([[0.0]], dtype=th.float32)
    buf.compute_returns_and_advantage(
        last_values, dones=np.array([1.0]), last_factor_values=np.zeros((1, 2))
    )

    # Sum of factor rewards over the rollout is the Monte-Carlo target.
    np.testing.assert_allclose(buf.factor_returns[0, 0], np.array([6.0, 1.0]), rtol=1e-5)
    np.testing.assert_allclose(buf.factor_returns[1, 0], np.array([5.0, 0.5]), rtol=1e-5)
    np.testing.assert_allclose(buf.factor_returns[2, 0], np.array([3.0, 1.0]), rtol=1e-5)


def test_buffer_per_factor_gae_propagates_value_baseline() -> None:
    """Per-factor advantages subtract V_k(s_t) from the per-factor return."""
    buf = _make_buffer(buffer_size=2, n_envs=1, n_factors=1)
    _add_step(
        buf, reward=0.0, value=0.0, factor_rewards=np.array([1.0]), factor_values=np.array([0.4])
    )
    _add_step(
        buf, reward=0.0, value=0.0, factor_rewards=np.array([2.0]), factor_values=np.array([1.0])
    )

    buf.compute_returns_and_advantage(
        last_values=th.tensor([[0.0]], dtype=th.float32),
        dones=np.array([1.0]),
        last_factor_values=np.zeros((1, 1)),
    )
    # delta_t1 = 2.0 + 0 - 1.0 = 1.0 -> A_t1 = 1.0
    # delta_t0 = 1.0 + 1.0 - 0.4 = 1.6 -> A_t0 = 1.6 + 1.0 = 2.6
    np.testing.assert_allclose(buf.factor_advantages[1, 0], np.array([1.0]), rtol=1e-5)
    np.testing.assert_allclose(buf.factor_advantages[0, 0], np.array([2.6]), rtol=1e-5)


def test_buffer_get_returns_cgfa_samples_with_factor_tensors() -> None:
    """Iterating get() yields CGFA samples with per-factor tensors of shape (B, K)."""
    buf = _make_buffer(buffer_size=4, n_envs=1, n_factors=3)
    for _ in range(4):
        _add_step(
            buf,
            reward=0.0,
            value=0.0,
            factor_rewards=np.array([0.1, 0.2, -0.1], dtype=np.float32),
            factor_values=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
    buf.compute_returns_and_advantage(
        last_values=th.tensor([[0.0]], dtype=th.float32),
        dones=np.array([1.0]),
        last_factor_values=np.zeros((1, 3)),
    )
    samples = list(buf.get(batch_size=2))
    assert len(samples) == 2
    s = samples[0]
    assert s.factor_advantages.shape == (2, 3)
    assert s.factor_returns.shape == (2, 3)
    assert s.factor_old_values.shape == (2, 3)
    assert s.factor_rewards.shape == (2, 3)
    assert s.factor_eps.shape == (2, 3)


def test_buffer_factor_eps_round_trips_through_add_and_get() -> None:
    """factor_eps written via add() comes back unchanged through get()."""
    buf = _make_buffer(buffer_size=2, n_envs=1, n_factors=2)
    obs = np.zeros((1, 2), dtype=np.float32)
    action = np.zeros((1,), dtype=np.int64)
    val_t = th.zeros(1, 1)
    log_prob = th.zeros(1)
    action_mask = np.ones((1, buf.mask_dims), dtype=np.float32)

    eps0 = np.array([[0.5, -0.2]], dtype=np.float32)
    eps1 = np.array([[0.7, 0.3]], dtype=np.float32)
    buf.add(
        obs,
        action,
        np.zeros(1, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        val_t,
        log_prob,
        action_masks=action_mask,
        factor_rewards=np.zeros((1, 2), dtype=np.float32),
        factor_values=th.zeros(1, 2),
        factor_eps=eps0,
    )
    buf.add(
        obs,
        action,
        np.zeros(1, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        val_t,
        log_prob,
        action_masks=action_mask,
        factor_rewards=np.zeros((1, 2), dtype=np.float32),
        factor_values=th.zeros(1, 2),
        factor_eps=eps1,
    )
    buf.compute_returns_and_advantage(
        th.zeros(1, 1), np.array([1.0]), last_factor_values=np.zeros((1, 2))
    )
    samples = list(buf.get(batch_size=2))
    eps_out = samples[0].factor_eps.cpu().numpy()
    # Order may be permuted by the random shuffle but the *set* of rows matches.
    eps_in = np.concatenate([eps0, eps1], axis=0)
    eps_out_sorted = eps_out[np.lexsort(eps_out.T)]
    eps_in_sorted = eps_in[np.lexsort(eps_in.T)]
    np.testing.assert_allclose(eps_out_sorted, eps_in_sorted, rtol=1e-6)

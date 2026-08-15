"""Rollout buffer that tracks per-factor signals on top of ``MaskableRolloutBuffer``.

This buffer stores three additional ``(buffer_size, n_envs, K)`` arrays:

* ``factor_rewards``: normalised changes in the K causal factors.
* ``factor_values``: per-factor value-head predictions from the
  CGFA policy at each visited state.
* ``factor_eps``: SCM-predicted per-factor changes for the same
  transition (used for the intervention-calibration loss).

When the rollout is full, per-factor GAE returns and advantages are
computed using the same recursion as scalar GAE applied independently
to each of the K factor channels. Per-factor returns are used as
targets for the per-factor value-head losses and per-factor
advantages are blended into the policy-gradient advantage by the
CGFA PPO algorithm.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize


class CGFAMaskableRolloutBufferSamples(NamedTuple):
    """Mini-batch sample produced by :class:`CGFAMaskableRolloutBuffer`."""

    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    action_masks: th.Tensor
    factor_old_values: th.Tensor
    factor_advantages: th.Tensor
    factor_returns: th.Tensor
    factor_rewards: th.Tensor
    factor_eps: th.Tensor


class CGFAMaskableRolloutBuffer(MaskableRolloutBuffer):
    """Maskable rollout buffer with per-factor returns and advantages.

    Args:
        buffer_size: Number of rollout steps per env.
        observation_space: Observation space of the env.
        action_space: Action space of the env.
        device: Torch device.
        gae_lambda: GAE lambda for both scalar and per-factor channels.
        gamma: Discount factor for both scalar and per-factor channels.
        n_envs: Number of parallel envs.
        n_factors: Number of CGFA factors K.
    """

    factor_rewards: np.ndarray
    factor_values: np.ndarray
    factor_advantages: np.ndarray
    factor_returns: np.ndarray
    factor_eps: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
        n_envs: int = 1,
        n_factors: int = 6,
    ) -> None:
        self.n_factors = int(n_factors)
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_envs,
        )

    def reset(self) -> None:
        """Allocate the per-factor arrays alongside the maskable defaults."""
        super().reset()
        shape = (self.buffer_size, self.n_envs, self.n_factors)
        self.factor_rewards = np.zeros(shape, dtype=np.float32)
        self.factor_values = np.zeros(shape, dtype=np.float32)
        self.factor_advantages = np.zeros(shape, dtype=np.float32)
        self.factor_returns = np.zeros(shape, dtype=np.float32)
        self.factor_eps = np.zeros(shape, dtype=np.float32)

    def add(  # type: ignore[override]
        self,
        *args,
        action_masks: np.ndarray | None = None,
        factor_rewards: np.ndarray | None = None,
        factor_values: np.ndarray | th.Tensor | None = None,
        factor_eps: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        """Capture per-factor signals at the current pos before delegating."""
        pos = self.pos
        if factor_rewards is not None:
            self.factor_rewards[pos] = np.asarray(factor_rewards, dtype=np.float32).reshape(
                (self.n_envs, self.n_factors)
            )
        if factor_values is not None:
            if isinstance(factor_values, th.Tensor):
                fv = factor_values.detach().cpu().numpy()
            else:
                fv = np.asarray(factor_values, dtype=np.float32)
            self.factor_values[pos] = fv.reshape((self.n_envs, self.n_factors))
        if factor_eps is not None:
            self.factor_eps[pos] = np.asarray(factor_eps, dtype=np.float32).reshape(
                (self.n_envs, self.n_factors)
            )

        super().add(*args, action_masks=action_masks, **kwargs)

    def compute_factor_returns_and_advantage(
        self,
        last_factor_values: np.ndarray | th.Tensor,
        dones: np.ndarray,
    ) -> None:
        """Compute per-factor GAE returns and advantages.

        Mirrors ``RolloutBuffer.compute_returns_and_advantage`` but on
        the K factor channels independently.
        """
        if isinstance(last_factor_values, th.Tensor):
            last = last_factor_values.detach().cpu().numpy()
        else:
            last = np.asarray(last_factor_values, dtype=np.float32)
        last = last.reshape(self.n_envs, self.n_factors)

        last_gae = np.zeros((self.n_envs, self.n_factors), dtype=np.float32)
        dones_arr = np.asarray(dones).astype(np.float32)
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = (1.0 - dones_arr)[:, None]
                next_values = last
            else:
                next_non_terminal = (1.0 - self.episode_starts[step + 1])[:, None]
                next_values = self.factor_values[step + 1]
            delta = (
                self.factor_rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.factor_values[step]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.factor_advantages[step] = last_gae
        self.factor_returns = self.factor_advantages + self.factor_values

    def compute_returns_and_advantage(  # type: ignore[override]
        self,
        last_values: th.Tensor,
        dones: np.ndarray,
        last_factor_values: np.ndarray | th.Tensor | None = None,
    ) -> None:
        """Run scalar GAE then per-factor GAE.

        ``last_factor_values`` is the per-factor bootstrap from the
        terminal observation, analogous to ``last_values`` for the
        scalar channel.  When ``None`` we treat the last factor values
        as zero, which is an unbiased Monte-Carlo bootstrap when the
        rollout ends on a real episode boundary.
        """
        super().compute_returns_and_advantage(last_values=last_values, dones=dones)
        if last_factor_values is None:
            last_factor_values = np.zeros((self.n_envs, self.n_factors), dtype=np.float32)
        self.compute_factor_returns_and_advantage(last_factor_values, dones)

    def get(  # type: ignore[override]
        self,
        batch_size: int | None = None,
    ) -> Generator[CGFAMaskableRolloutBufferSamples, None, None]:
        """Yield CGFA mini-batches with per-factor tensors attached.

        We re-implement the loop body of
        :meth:`MaskableRolloutBuffer.get` so we can flatten both the
        scalar tensors *and* the per-factor tensors together and then
        slice them with the same permutation indices.
        """
        assert self.full, "Cannot iterate a non-full rollout buffer"
        flatten_keys = [
            "observations",
            "actions",
            "values",
            "log_probs",
            "advantages",
            "returns",
            "action_masks",
            "factor_rewards",
            "factor_values",
            "factor_advantages",
            "factor_returns",
            "factor_eps",
        ]
        if not self.generator_ready:
            for key in flatten_keys:
                self.__dict__[key] = self.swap_and_flatten(self.__dict__[key])
            self.generator_ready = True

        n = self.buffer_size * self.n_envs
        if batch_size is None:
            batch_size = n
        indices = np.random.permutation(n)
        start = 0
        while start < n:
            yield self._get_cgfa_samples(indices[start : start + batch_size])
            start += batch_size

    def _get_cgfa_samples(
        self,
        batch_inds: np.ndarray,
        env: VecNormalize | None = None,
    ) -> CGFAMaskableRolloutBufferSamples:
        # All the relevant arrays have already been ``swap_and_flatten``'d
        # (see ``get``), so they have shape ``(buffer_size * n_envs, ...)``
        # and slicing with a 1-D ``batch_inds`` array gives the right
        # mini-batch shape.
        return CGFAMaskableRolloutBufferSamples(
            observations=self.to_torch(self.observations[batch_inds]),
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
            action_masks=self.to_torch(self.action_masks[batch_inds].reshape(-1, self.mask_dims)),
            factor_old_values=self.to_torch(
                self.factor_values[batch_inds].reshape(-1, self.n_factors)
            ),
            factor_advantages=self.to_torch(
                self.factor_advantages[batch_inds].reshape(-1, self.n_factors)
            ),
            factor_returns=self.to_torch(
                self.factor_returns[batch_inds].reshape(-1, self.n_factors)
            ),
            factor_rewards=self.to_torch(
                self.factor_rewards[batch_inds].reshape(-1, self.n_factors)
            ),
            factor_eps=self.to_torch(self.factor_eps[batch_inds].reshape(-1, self.n_factors)),
        )

"""Evaluation utilities for MTG agents.

This module provides the ``Evaluator`` class used by every evaluation
entry point. It shares the ``EnvConfig`` abstraction with ``Trainer``
so that agent training and agent evaluation are guaranteed to happen
in the same MDP (same agency settings, same step cap, same heuristic
opponent) unless the caller explicitly overrides that.

The module-level ``compare_agents`` function performs a proper per-seed
decomposition with bootstrap confidence intervals so multi-seed
comparisons report the right uncertainty band rather than pooling
episodes across seeds.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from mtg.training.env_factory import EnvConfig, create_env
from mtg.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config and result containers
# ---------------------------------------------------------------------------


@dataclass
class EvaluationConfig:
    """Configuration for evaluating a single agent against a single opponent.

    Environment fields map one-to-one onto ``EnvConfig``.  Evaluation-
    specific fields (number of episodes, determinism, output dir)
    stay on this dataclass.
    """

    # MDP fields (mirror EnvConfig) ---------------------------------
    deck_archetype: str = "mono_red_aggro"
    opponent_archetype: str | None = None
    reward_type: str = "shaped"
    max_turns: int = 20
    max_steps_per_episode: int = 500
    auto_combat: bool = False
    auto_target: bool = False
    auto_mana: bool = True
    use_heuristic_opponent: bool = True

    # Evaluation-specific -------------------------------------------
    n_episodes: int = 100
    seed: int = 42
    deterministic: bool = True

    # Output
    output_dir: str = "results/evaluation"
    save_trajectories: bool = False

    # Distribution-shift safety. Path to the ``vec_normalize.pkl`` saved
    # at training time. When set, the Evaluator wraps the eval env with
    # the SAME frozen observation statistics so train and eval are over
    # the same observation distribution.
    vec_normalize_path: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvaluationConfig:
        """Load an ``EvaluationConfig`` from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def env_config(self) -> EnvConfig:
        """Return the shared ``EnvConfig`` derived from this evaluation config."""
        return EnvConfig(
            player_deck=self.deck_archetype,
            opponent_deck=self.opponent_archetype or self.deck_archetype,
            reward_type=self.reward_type,
            max_turns=self.max_turns,
            max_steps_per_episode=self.max_steps_per_episode,
            auto_combat=self.auto_combat,
            auto_target=self.auto_target,
            auto_mana=self.auto_mana,
            use_heuristic_opponent=self.use_heuristic_opponent,
            seed=self.seed,
        )


@dataclass
class EvaluationResult:
    """Per-run evaluation metrics with bootstrap confidence intervals.

    ``win_rate_ci95`` is the half-width of a 95% percentile bootstrap
    interval (so the CI is ``[win_rate - ci, win_rate + ci]``).  For
    multi-seed runs, ``per_seed_win_rates`` contains one entry per seed.
    """

    n_episodes: int
    win_rate: float
    mean_reward: float
    std_reward: float
    mean_length: float
    std_length: float

    # Detailed / raw
    rewards: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    wins: list[bool] = field(default_factory=list)

    # Bootstrap / per-seed
    win_rate_ci95: float = 0.0
    reward_ci95: float = 0.0
    length_ci95: float = 0.0
    per_seed_win_rates: list[float] = field(default_factory=list)
    per_seed_rewards: list[float] = field(default_factory=list)
    per_seed_lengths: list[float] = field(default_factory=list)

    # Causal variable stats
    causal_var_means: dict[str, float] = field(default_factory=dict)
    causal_var_stds: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a multi-line human-readable summary."""
        per_seed = ""
        if self.per_seed_win_rates:
            per_seed = f"  Per-seed WR: {[f'{wr:.2%}' for wr in self.per_seed_win_rates]}\n"
        return (
            f"Evaluation Results ({self.n_episodes} episodes):\n"
            f"  Win Rate: {self.win_rate:.2%} "
            f"(95% bootstrap CI ±{self.win_rate_ci95:.2%})\n"
            f"  Mean Reward: {self.mean_reward:.3f} "
            f"± {self.std_reward:.3f} "
            f"(95% CI ±{self.reward_ci95:.3f})\n"
            f"  Mean Length: {self.mean_length:.1f} "
            f"± {self.std_length:.1f} "
            f"(95% CI ±{self.length_ci95:.2f})\n"
            f"{per_seed}"
        )

    def to_dict(self) -> dict[str, tp.Any]:
        """Serialise all fields to a plain dict (JSON-safe)."""
        return {
            "n_episodes": self.n_episodes,
            "win_rate": self.win_rate,
            "win_rate_ci95": self.win_rate_ci95,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "reward_ci95": self.reward_ci95,
            "mean_length": self.mean_length,
            "std_length": self.std_length,
            "length_ci95": self.length_ci95,
            "per_seed_win_rates": self.per_seed_win_rates,
            "per_seed_rewards": self.per_seed_rewards,
            "per_seed_lengths": self.per_seed_lengths,
            "causal_var_means": self.causal_var_means,
            "causal_var_stds": self.causal_var_stds,
        }


# ---------------------------------------------------------------------------
# Bootstrap utilities
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: tp.Sequence[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Return ``(mean, lower, upper)`` for a percentile bootstrap CI."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    if arr.size == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, arr.size, size=arr.size)
        boot_means[i] = arr[idx].mean()

    alpha = 1.0 - confidence
    lower = float(np.quantile(boot_means, alpha / 2))
    upper = float(np.quantile(boot_means, 1.0 - alpha / 2))
    return float(arr.mean()), lower, upper


def bootstrap_half_width(
    values: tp.Sequence[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> float:
    """Symmetric half-width of a bootstrap CI around the mean."""
    mean, lo, hi = bootstrap_ci(values, n_bootstrap, confidence, seed)
    return float(max(mean - lo, hi - mean))


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class Evaluator:
    """Evaluate an agent on the shared MDP configured by ``EnvConfig``.

    When the trained agent was wrapped in :class:`VecNormalize` at
    training time, callers MUST pass ``vec_normalize_path`` so the
    same observation normalisation is applied at evaluation time.
    Without it the policy is queried on un-normalised observations and
    the eval results are not comparable to the training distribution.
    """

    def __init__(
        self,
        config: EvaluationConfig,
        vec_normalize_path: str | Path | None = None,
    ) -> None:
        self.config = config
        self.env: tp.Any = None
        self.logger = get_logger(f"{__name__}.Evaluator")
        self._vec_normalize_path = (
            Path(vec_normalize_path) if vec_normalize_path is not None else None
        )
        # Populated lazily in ``setup()`` when stats are available.
        self._obs_rms: tp.Any = None
        self._clip_obs: float = 10.0
        self._obs_eps: float = 1e-8

    def setup(self) -> None:
        """Construct the evaluation environment lazily on first use."""
        self.env = create_env(self.config.env_config())
        if self._vec_normalize_path is not None and self._vec_normalize_path.exists():
            from stable_baselines3.common.vec_env import (
                DummyVecEnv,
                VecNormalize,
            )

            # VecNormalize.load needs a venv to bind to; we discard the
            # bound venv and keep only the running statistics so the
            # non-vec eval loop can apply them per-obs.
            scratch_vec = DummyVecEnv([lambda: self.env])
            loaded = VecNormalize.load(str(self._vec_normalize_path), scratch_vec)
            self._obs_rms = loaded.obs_rms
            self._clip_obs = float(loaded.clip_obs)
            self._obs_eps = float(loaded.epsilon)
            # Re-extract the underlying env so we control stepping.
            self.env = scratch_vec.envs[0]
            self.logger.info(
                "Evaluator: loaded frozen VecNormalize stats from %s (obs_rms.mean[:3]=%s)",
                self._vec_normalize_path,
                np.asarray(self._obs_rms.mean[:3]).tolist()
                if hasattr(self._obs_rms, "mean")
                else "?",
            )

    def _normalise_obs(self, obs: np.ndarray) -> np.ndarray:
        """Apply the loaded ``VecNormalize`` running stats to a single obs.

        Mirrors ``stable_baselines3.common.vec_env.VecNormalize.normalize_obs``
        for the (non-Dict) Box case.
        """
        if self._obs_rms is None:
            return obs
        normalised = (obs - self._obs_rms.mean) / np.sqrt(self._obs_rms.var + self._obs_eps)
        return np.clip(normalised, -self._clip_obs, self._clip_obs).astype(obs.dtype)

    def evaluate(
        self,
        agent: tp.Any,
        n_episodes: int | None = None,
        progress_bar: bool = True,
        seed_offset: int = 0,
    ) -> EvaluationResult:
        """Run one-seed evaluation for ``n_episodes`` games.

        Episode-level seeds are ``self.config.seed + seed_offset + ep``
        so that multi-seed ``compare_agents`` can request disjoint sets
        of games simply by adjusting ``seed_offset``.
        """
        if self.env is None:
            self.setup()
        assert self.env is not None

        n_episodes = n_episodes or self.config.n_episodes
        prev_det = getattr(agent, "deterministic", None)
        if hasattr(agent, "deterministic"):
            agent.deterministic = self.config.deterministic

        rewards: list[float] = []
        lengths: list[int] = []
        wins: list[bool] = []
        causal_vars_history: list[dict[str, float]] = []

        iterator = range(n_episodes)
        if progress_bar:
            iterator = tqdm(iterator, desc="Evaluating")

        try:
            for ep in iterator:
                obs, info = self.env.reset(seed=self.config.seed + seed_offset + ep)
                obs = self._normalise_obs(obs)
                done = False
                ep_reward = 0.0
                ep_length = 0
                episode_causal_vars: list[dict[str, float]] = []

                while not done:
                    action_mask = info.get(
                        "action_mask",
                        np.ones(self.env.action_space.n, dtype=bool),
                    )
                    action = agent.select_action(obs, action_mask, info)
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    obs = self._normalise_obs(obs)
                    done = terminated or truncated
                    ep_reward += float(reward)
                    ep_length += 1
                    if "causal_variables" in info:
                        episode_causal_vars.append(info["causal_variables"])

                rewards.append(ep_reward)
                lengths.append(ep_length)
                wins.append(_is_win(info))
                if episode_causal_vars:
                    causal_vars_history.append(episode_causal_vars[-1])
        finally:
            if prev_det is not None and hasattr(agent, "deterministic"):
                agent.deterministic = prev_det

        wins_arr = np.asarray(wins, dtype=float)
        rewards_arr = np.asarray(rewards, dtype=float)
        lengths_arr = np.asarray(lengths, dtype=float)

        result = EvaluationResult(
            n_episodes=n_episodes,
            win_rate=float(wins_arr.mean()) if wins_arr.size else 0.0,
            mean_reward=float(rewards_arr.mean()) if rewards_arr.size else 0.0,
            std_reward=float(rewards_arr.std()) if rewards_arr.size else 0.0,
            mean_length=float(lengths_arr.mean()) if lengths_arr.size else 0.0,
            std_length=float(lengths_arr.std()) if lengths_arr.size else 0.0,
            rewards=rewards,
            lengths=lengths,
            wins=wins,
            win_rate_ci95=bootstrap_half_width(wins_arr.tolist(), seed=self.config.seed),
            reward_ci95=bootstrap_half_width(rewards_arr.tolist(), seed=self.config.seed),
            length_ci95=bootstrap_half_width(lengths_arr.tolist(), seed=self.config.seed),
        )

        if causal_vars_history:
            var_names = causal_vars_history[0].keys()
            for var in var_names:
                values = [cv[var] for cv in causal_vars_history if var in cv]
                if values:
                    result.causal_var_means[var] = float(np.mean(values))
                    result.causal_var_stds[var] = float(np.std(values))

        self.logger.info(result.summary())
        return result


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _is_win(info: dict[str, tp.Any]) -> bool:
    if "terminal_info" in info and isinstance(info["terminal_info"], dict):
        info = info["terminal_info"]
    result = info.get("game_result")
    if isinstance(result, str):
        return result.lower() == "win"
    winner = info.get("winner")
    if isinstance(winner, str):
        return winner.lower() in {"player", "win", "agent"}
    if isinstance(winner, int | np.integer):
        return winner == 0
    return False


def evaluate(
    agent: tp.Any,
    env: tp.Any | None = None,
    n_episodes: int = 100,
    config_path: str | None = None,
    vec_normalize_path: str | Path | None = None,
    **kwargs: tp.Any,
) -> EvaluationResult:
    """Single-call entry point used by tests and quick scripts."""
    if config_path:
        config = EvaluationConfig.from_yaml(config_path)
    else:
        config = EvaluationConfig(n_episodes=n_episodes, **kwargs)
    if vec_normalize_path is not None:
        config.vec_normalize_path = str(vec_normalize_path)

    evaluator = Evaluator(config, vec_normalize_path=config.vec_normalize_path)
    if env is not None:
        evaluator.env = env
    return evaluator.evaluate(agent)


def compare_agents(
    agents: dict[str, tp.Any],
    config: EvaluationConfig,
    n_seeds: int = 5,
) -> dict[str, EvaluationResult]:
    """Compare multiple agents with proper per-seed aggregation.

    For each agent we run ``n_seeds`` evaluations of
    ``config.n_episodes`` each. The reported win rate, reward, and
    length are the mean over seeds (so each seed is one i.i.d.
    observation), and the confidence interval is the bootstrap CI over
    the per-seed means. This is a proper generalisation-across-seeds
    statistic; pooling all episodes across seeds would underestimate
    variance because within-seed episodes are not independent of the
    seed.

    ``EvaluationResult.per_seed_*`` holds the raw per-seed statistics
    so callers can produce scatter / violin plots.
    """
    results: dict[str, EvaluationResult] = {}

    for name, agent in agents.items():
        logger.info("Evaluating %s over %s seeds...", name, n_seeds)
        per_seed_wr: list[float] = []
        per_seed_rw: list[float] = []
        per_seed_len: list[float] = []
        all_rewards: list[float] = []
        all_wins: list[bool] = []
        all_lengths: list[int] = []

        for seed_idx in range(n_seeds):
            seed = config.seed + seed_idx * 1000
            cfg = EvaluationConfig(
                deck_archetype=config.deck_archetype,
                opponent_archetype=config.opponent_archetype,
                reward_type=config.reward_type,
                max_turns=config.max_turns,
                max_steps_per_episode=config.max_steps_per_episode,
                auto_combat=config.auto_combat,
                auto_target=config.auto_target,
                auto_mana=config.auto_mana,
                use_heuristic_opponent=config.use_heuristic_opponent,
                n_episodes=config.n_episodes,
                seed=seed,
                deterministic=config.deterministic,
                vec_normalize_path=config.vec_normalize_path,
            )
            evaluator = Evaluator(cfg, vec_normalize_path=cfg.vec_normalize_path)
            res = evaluator.evaluate(agent, progress_bar=False)

            per_seed_wr.append(res.win_rate)
            per_seed_rw.append(res.mean_reward)
            per_seed_len.append(res.mean_length)
            all_rewards.extend(res.rewards)
            all_wins.extend(res.wins)
            all_lengths.extend(res.lengths)

        mean_wr = float(np.mean(per_seed_wr))
        mean_rw = float(np.mean(per_seed_rw))
        mean_len = float(np.mean(per_seed_len))

        results[name] = EvaluationResult(
            n_episodes=len(all_rewards),
            win_rate=mean_wr,
            mean_reward=mean_rw,
            std_reward=float(np.std(per_seed_rw)),
            mean_length=mean_len,
            std_length=float(np.std(per_seed_len)),
            rewards=all_rewards,
            lengths=all_lengths,
            wins=all_wins,
            win_rate_ci95=bootstrap_half_width(per_seed_wr, seed=config.seed),
            reward_ci95=bootstrap_half_width(per_seed_rw, seed=config.seed),
            length_ci95=bootstrap_half_width(per_seed_len, seed=config.seed),
            per_seed_win_rates=per_seed_wr,
            per_seed_rewards=per_seed_rw,
            per_seed_lengths=per_seed_len,
        )
        logger.info(
            "%s: Win rate = %.2f%% ± %.2f%% (per-seed)",
            name,
            mean_wr * 100,
            results[name].win_rate_ci95 * 100,
        )

    return results

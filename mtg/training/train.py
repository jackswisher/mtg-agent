"""Training utilities for MTG agents.

This module provides the canonical ``Trainer`` class used by every
training entry point in the project (``mtg-research``, the interactive
CLI, ablation scripts, unit tests). It is built on top of
``mtg.training.env_factory`` and the SB3 callbacks defined in
``mtg.training.callbacks`` so that:

* Training and evaluation share a single ``EnvConfig``.
* Checkpointing, early stopping, entropy annealing, and periodic
  evaluation all run as part of the SB3 callback chain.
* Metrics returned from ``Trainer.train`` are real episode-level
  data, not empty dicts.
"""

from __future__ import annotations

import contextlib
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from mtg.agents import (
    CausalAgent,
    CGFAAgent,
    CGFAScalarOnlyAgent,
    GreedyAggroAgent,
    PPOAgent,
    RandomAgent,
)
from mtg.training.callbacks import (
    AdaptiveKLCallback,
    CallbackList,
    CheckpointCallback,
    EarlyStoppingCallback,
    EntropyScheduleCallback,
    EpisodeLoggerCallback,
    EvalCallback,
    LeagueCallback,
    MetricsCallback,
)
from mtg.training.env_factory import (
    EnvConfig,
    create_env,
    create_vec_env,
    get_default_agent_for_deck,
)
from mtg.training.league import League, LeagueConfig
from mtg.utils.logging import get_logger
from mtg.utils.run_metadata import snapshot_run_metadata
from mtg.utils.seeding import set_global_seed

logger = get_logger(__name__)


AGENT_REGISTRY: dict[str, type] = {
    "random": RandomAgent,
    "greedy_aggro": GreedyAggroAgent,
    "ppo": PPOAgent,
    "causal": CausalAgent,
    "cgfa": CGFAAgent,
    "cgfa_scalar_only": CGFAScalarOnlyAgent,
}

# RL-style agents that go through the SB3 ``learn`` loop and need
# vectorised envs / SB3 callback stacks.  Kept as a single set so
# callers don't have to enumerate it everywhere.
_RL_AGENTS: set[str] = {"ppo", "causal", "cgfa", "cgfa_scalar_only"}

# CGFA-family agents (require the env wrapped in ``CGFAEnvWrapper`` so
# the rollout buffer can read ``factor_rewards`` / ``factor_eps``).
# Includes the ``cgfa_scalar_only`` ablation because its policy still
# carries the per-factor heads even though their loss coefficients are
# pinned to zero.
_CGFA_AGENTS: set[str] = {"cgfa", "cgfa_scalar_only"}


@dataclass
class TrainingConfig:
    """Canonical training configuration.

    Environment / MDP fields are pinned into an ``EnvConfig`` that is
    passed through to the evaluator.  This guarantees train/eval MDP
    match by construction.
    """

    # Agent ----------------------------------------------------------
    agent_type: str = "ppo"
    agent_kwargs: dict[str, tp.Any] = field(default_factory=dict)

    # Environment (populated into EnvConfig on setup) -----------------
    deck_archetype: str = "mono_red_aggro"
    opponent_archetype: str | None = None
    reward_type: str = "shaped"
    max_turns: int = 20
    max_steps_per_episode: int = 500
    auto_combat: bool = False
    auto_target: bool = False
    auto_mana: bool = True
    use_heuristic_opponent: bool = True

    # Training -------------------------------------------------------
    total_timesteps: int = 100_000
    n_envs: int = 1
    seed: int = 42

    # Single source of truth for discount factor.  Propagated to
    # ``EnvConfig.gamma`` (reward shaping) and the PPO agent's ``gamma``
    # so they cannot drift apart.
    gamma: float = 0.995

    # Callbacks ------------------------------------------------------
    enable_checkpointing: bool = True
    enable_entropy_schedule: bool = True
    enable_early_stopping: bool = False
    enable_periodic_eval: bool = True
    enable_episode_logger: bool = True
    enable_adaptive_kl: bool = False

    # When True the training VecEnv is wrapped in ``VecNormalize`` with
    # running-mean/var observation normalisation. The running statistics
    # are saved alongside the policy so evaluation can load them and
    # apply identical normalisation.
    enable_vec_normalize: bool = False
    normalize_obs: bool = True
    normalize_reward: bool = False

    log_dir: str = "results/logs"
    checkpoint_dir: str = "results/checkpoints"
    experiment_name: str = "default"

    # Entropy annealing (applied when enable_entropy_schedule=True and
    # the agent is a PPO-family agent).
    ent_coef_start: float = 0.05
    ent_coef_end: float = 0.005
    entropy_schedule: str = "linear"

    # Adaptive KL (B4)
    adaptive_kl_target: float = 0.02
    adaptive_kl_min_clip: float = 0.05
    adaptive_kl_max_clip: float = 0.3

    # League / PFSP (C2) ---------------------------------------------
    # When enabled, training draws opponents from a pool of heuristic
    # archetypes (and, optionally, historical snapshots of the learner
    # itself) via Prioritised Fictitious Self-Play.  Opponent Elo
    # ratings are maintained in memory and persisted at training end.
    enable_league: bool = False
    league_opponents: list[str] = field(
        default_factory=lambda: [
            "mono_red_aggro",
            "azorius_control",
            "simic_ramp",
            "selesnya_midrange",
            "boros_convoke",
        ]
    )
    league_sampling: str = "pfsp"
    league_pfsp_p: float = 2.0
    league_elo_k: float = 16.0
    league_snapshot_interval: int | None = None
    league_snapshot_warmup: int = 0
    league_max_historical: int = 6

    # Logging intervals
    log_interval: int = 10
    save_interval: int = 50_000
    eval_interval: int = 50_000
    eval_episodes: int = 20

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.01
    early_stopping_check_interval: int = 25_000
    early_stopping_window: int = 100

    # ----------------------------------------------------------------
    # Derived
    # ----------------------------------------------------------------

    def env_config(self) -> EnvConfig:
        """Construct the shared ``EnvConfig`` from this training config."""
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
            gamma=self.gamma,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        """Load a ``TrainingConfig`` from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Serialise this config to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)


class Trainer:
    """Canonical trainer for MTG agents.

    Lifecycle:

        trainer = Trainer(config)
        trainer.setup()      # builds envs and agent
        metrics = trainer.train()
        trainer.close()

    For PPO/Causal agents this uses SB3's ``learn`` with the real
    callback stack; for heuristic / tabular agents it falls back to a
    plain Python training loop that still honours checkpointing and
    episode logging.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.env: tp.Any = None
        self.agent: tp.Any = None
        self.league: League | None = None
        self.logger = get_logger(f"{__name__}.Trainer")

        self.log_dir = Path(config.log_dir) / config.experiment_name
        self.checkpoint_dir = Path(config.checkpoint_dir) / config.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Remembered after setup for metric collection
        self._metrics_callback: MetricsCallback | None = None
        self._episode_logger: EpisodeLoggerCallback | None = None

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Build env + agent using the shared ``EnvConfig``."""
        set_global_seed(self.config.seed)

        env_cfg = self.config.env_config()
        self.logger.info("Training env: %s", env_cfg.describe())

        if self.config.enable_league:
            self.league = self._build_league()
        n_envs = self.config.n_envs
        if self.config.enable_league and n_envs > 1:
            self.logger.warning(
                "League training forces n_envs=1 (League state cannot be "
                "shared across subprocesses). Requested n_envs=%s was ignored.",
                n_envs,
            )
            n_envs = 1

        agent_cls = AGENT_REGISTRY.get(self.config.agent_type)
        if agent_cls is None:
            raise ValueError(f"Unknown agent type: {self.config.agent_type}")

        # CGFA agents need the env wrapped with CGFAEnvWrapper *before*
        # ActionMasker so the per-factor signals reach the rollout
        # buffer.  Build an interim agent (no env yet) so we can pull
        # the FactorSpec / SCM out and pass them to the env factory,
        # then attach the env once it exists.
        cgfa_factor_spec = None
        cgfa_scm = None
        agent_kwargs = dict(self.config.agent_kwargs)
        if self.config.agent_type in _RL_AGENTS:
            agent_kwargs.setdefault("gamma", self.config.gamma)
        if self.config.agent_type in _CGFA_AGENTS:
            from mtg.agents.reinforcement_learning.cgfa import FactorSpec
            from mtg.causal.scm import StructuralCausalModel

            cgfa_factor_spec = agent_kwargs.get("factor_spec") or FactorSpec()
            cgfa_scm = agent_kwargs.get("scm") or StructuralCausalModel()

        if self.config.agent_type in _RL_AGENTS:
            self.env = create_vec_env(
                env_cfg,
                n_envs=n_envs,
                normalize=self.config.enable_vec_normalize,
                norm_obs=self.config.normalize_obs,
                norm_reward=self.config.normalize_reward,
                league=self.league,
                cgfa_factor_spec=cgfa_factor_spec,
                cgfa_scm=cgfa_scm,
            )
        else:
            self.env = create_env(env_cfg)

        if self.config.agent_type in _RL_AGENTS:
            obs_space = self.env.observation_space
            obs_dim = obs_space.shape[0]
            act_dim = self.env.action_space.n
            self.agent = agent_cls(
                observation_dim=obs_dim,
                action_dim=act_dim,
                seed=self.config.seed,
                **agent_kwargs,
            )
            if hasattr(self.agent, "initialize_model"):
                self.agent.initialize_model(self.env)
        else:
            self.agent = agent_cls(seed=self.config.seed, **self.config.agent_kwargs)

        self.logger.info(
            "Set up %s agent on %s",
            self.config.agent_type,
            self.config.deck_archetype,
        )

    def close(self) -> None:
        """Close vec envs to release subprocess workers."""
        if self.env is not None and hasattr(self.env, "close"):
            with contextlib.suppress(Exception):
                self.env.close()

    def _build_league(self) -> League:
        """Construct the league pool from ``config.league_opponents``."""
        import numpy as np

        from mtg.agents import get_agent

        # Seed the league RNG from TrainingConfig.seed so PFSP / uniform
        # opponent sampling is reproducible across runs. Without this the
        # league would default to a fresh ``np.random.default_rng()``,
        # making opponent sequences nondeterministic even when the rest
        # of the run is fully seeded.
        league_seed = self.config.seed if self.config.seed is not None else 0
        league_rng = np.random.default_rng(league_seed)

        league = League(
            LeagueConfig(
                elo_k=self.config.league_elo_k,
                sampling=self.config.league_sampling,
                pfsp_p=self.config.league_pfsp_p,
                snapshot_dir=self.checkpoint_dir / "league_snapshots",
                max_historical=self.config.league_max_historical,
            ),
            rng=league_rng,
        )
        for deck in self.config.league_opponents:
            agent_name = get_default_agent_for_deck(deck)
            opponent = get_agent(agent_name)
            # Disambiguate if the same heuristic plays multiple decks
            pool_name = f"{agent_name}_{deck}"
            if any(entry.name == pool_name for entry in league.pool):
                pool_name = f"{pool_name}_{len(league.pool)}"
            league.add_heuristic(pool_name, deck=deck, agent=opponent)
        self.logger.info(
            "Built league with %s opponents (sampling=%s)",
            len(league.pool),
            self.config.league_sampling,
        )
        return league

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _build_callback_stack(self) -> tp.Any:
        callbacks: list[tp.Any] = []
        self._metrics_callback = MetricsCallback()
        callbacks.append(self._metrics_callback)

        if self.config.enable_episode_logger:
            self._episode_logger = EpisodeLoggerCallback(
                log_dir=str(self.log_dir),
                log_interval=self.config.log_interval,
            )
            callbacks.append(self._episode_logger)

        if self.config.enable_checkpointing:
            callbacks.append(
                CheckpointCallback(
                    save_dir=str(self.checkpoint_dir),
                    save_interval=self.config.save_interval,
                )
            )

        if self.config.enable_entropy_schedule and self.config.agent_type in _RL_AGENTS:
            callbacks.append(
                EntropyScheduleCallback(
                    ent_coef_start=self.config.ent_coef_start,
                    ent_coef_end=self.config.ent_coef_end,
                    total_timesteps=self.config.total_timesteps,
                    schedule=self.config.entropy_schedule,
                )
            )

        if self.config.enable_early_stopping:
            callbacks.append(
                EarlyStoppingCallback(
                    patience=self.config.early_stopping_patience,
                    min_delta=self.config.early_stopping_min_delta,
                    window=self.config.early_stopping_window,
                    check_interval=self.config.early_stopping_check_interval,
                )
            )

        if self.config.enable_periodic_eval:
            callbacks.append(
                EvalCallback(
                    eval_env_config=self.config.env_config(),
                    n_eval_episodes=self.config.eval_episodes,
                    eval_interval=self.config.eval_interval,
                    log_dir=str(self.log_dir / "eval"),
                )
            )

        if self.config.enable_adaptive_kl and self.config.agent_type in _RL_AGENTS:
            callbacks.append(
                AdaptiveKLCallback(
                    target_kl=self.config.adaptive_kl_target,
                    min_clip=self.config.adaptive_kl_min_clip,
                    max_clip=self.config.adaptive_kl_max_clip,
                )
            )

        if self.config.enable_league and self.league is not None:
            callbacks.append(
                LeagueCallback(
                    league=self.league,
                    snapshot_interval=self.config.league_snapshot_interval,
                    snapshot_after=self.config.league_snapshot_warmup,
                    log_dir=str(self.log_dir / "league"),
                )
            )

        if self.config.agent_type == "cgfa":
            from mtg.agents.reinforcement_learning.cgfa import CGFACalibrationCallback

            callbacks.append(
                CGFACalibrationCallback(
                    log_dir=str(self.log_dir / "cgfa"),
                )
            )

        return CallbackList(callbacks)

    def train(self) -> dict[str, tp.Any]:
        """Run the canonical training loop and return per-episode metrics."""
        if self.env is None or self.agent is None:
            self.setup()

        assert self.env is not None
        assert self.agent is not None

        # Snapshot the full reproducibility manifest *before* a single
        # gradient step happens.  Even a crashed run leaves behind
        # ``run_metadata.json`` so we can correlate logs back to the
        # exact (git SHA, lockfile digest, config) tuple.
        try:
            manifest_path = self.log_dir / "run_metadata.json"
            snapshot_run_metadata(
                manifest_path,
                config=self.config,
                extra={
                    "experiment_name": self.config.experiment_name,
                    "agent_type": self.config.agent_type,
                    "deck_archetype": self.config.deck_archetype,
                    "total_timesteps": int(self.config.total_timesteps),
                    "n_envs": int(self.config.n_envs),
                    "seed": int(self.config.seed),
                    "log_dir": str(self.log_dir),
                    "checkpoint_dir": str(self.checkpoint_dir),
                },
            )
            self.logger.info("Wrote reproducibility manifest to %s", manifest_path)
        except Exception as exc:  # noqa: BLE001 - never fail training over manifest
            self.logger.warning("Failed to write run_metadata.json: %s", exc)

        self.logger.info("Starting training for %s timesteps", self.config.total_timesteps)

        if self.config.agent_type in _RL_AGENTS and hasattr(self.agent, "train"):
            callback = self._build_callback_stack()
            self.agent.train(
                total_timesteps=self.config.total_timesteps,
                callback=callback,
                progress_bar=False,
            )
            metrics = self._collect_metrics()
        else:
            metrics = self._custom_training_loop()

        final_path = self.checkpoint_dir / "final_model"
        if hasattr(self.agent, "save"):
            self.agent.save(str(final_path))
            self.logger.info("Saved final model to %s", final_path)

        if self.config.enable_vec_normalize and self.env is not None:
            try:
                from stable_baselines3.common.vec_env import VecNormalize

                if isinstance(self.env, VecNormalize):
                    vn_path = self.checkpoint_dir / "vec_normalize.pkl"
                    self.env.save(str(vn_path))
                    self.logger.info("Saved VecNormalize stats to %s", vn_path)
            except ImportError:
                pass

        self.config.to_yaml(self.log_dir / "config.yaml")
        return metrics

    # ------------------------------------------------------------------
    # Metric collection
    # ------------------------------------------------------------------

    def _collect_metrics(self) -> dict[str, tp.Any]:
        mc = self._metrics_callback
        if mc is None:
            return {"episode_rewards": [], "episode_lengths": [], "win_rate": 0.0}
        win_rate = float(np.mean(mc.episode_wins)) if mc.episode_wins else 0.0
        mean_length = float(np.mean(mc.episode_lengths)) if mc.episode_lengths else 0.0
        mean_reward = float(np.mean(mc.episode_rewards)) if mc.episode_rewards else 0.0
        return {
            "episode_rewards": list(mc.episode_rewards),
            "episode_lengths": list(mc.episode_lengths),
            "episode_wins": list(mc.episode_wins),
            "win_rate": win_rate,
            "mean_reward": mean_reward,
            "mean_length": mean_length,
            "total_episodes": len(mc.episode_rewards),
        }

    # ------------------------------------------------------------------
    # Fallback loop for non-SB3 agents
    # ------------------------------------------------------------------

    def _custom_training_loop(self) -> dict[str, tp.Any]:
        assert self.env is not None
        assert self.agent is not None

        episode_rewards: list[float] = []
        episode_lengths: list[int] = []
        wins = 0
        total_episodes = 0
        timesteps = 0

        while timesteps < self.config.total_timesteps:
            obs, info = self.env.reset()
            done = False
            ep_reward = 0.0
            ep_length = 0

            while not done:
                action_mask = info.get(
                    "action_mask",
                    np.ones(self.env.action_space.n, dtype=bool),
                )
                action = self.agent.select_action(obs, action_mask, info)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                ep_reward += float(reward)
                ep_length += 1
                timesteps += 1

                if hasattr(self.agent, "learn"):
                    self.agent.learn(
                        observation=obs,
                        action=action,
                        reward=reward,
                        next_observation=next_obs,
                        done=done,
                        info=info,
                    )

                obs = next_obs

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            total_episodes += 1
            if _is_win(info):
                wins += 1

            if total_episodes % self.config.log_interval == 0 and self.config.enable_episode_logger:
                recent = episode_rewards[-self.config.log_interval :]
                self.logger.info(
                    "Timesteps: %s, Episodes: %s, Mean reward: %.3f, Win rate: %.3f",
                    timesteps,
                    total_episodes,
                    float(np.mean(recent)),
                    wins / total_episodes,
                )

        win_rate = wins / total_episodes if total_episodes > 0 else 0.0
        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "win_rate": win_rate,
            "total_episodes": total_episodes,
        }


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


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def train(
    agent_class: type | None = None,
    agent_type: str = "ppo",
    total_timesteps: int = 100_000,
    config_path: str | None = None,
    **kwargs: tp.Any,
) -> tp.Any:
    """Single-call entry point used by tests and quick scripts.

    Args:
        agent_class: Unused (kept for backward compatibility).
        agent_type: Registered agent name (``"ppo"``, ``"causal"``, etc.).
        total_timesteps: Budget in env steps.
        config_path: Optional YAML file to load ``TrainingConfig`` from.
        **kwargs: Additional fields forwarded to ``TrainingConfig``.

    Returns:
        The trained agent instance.
    """
    del agent_class  # back-compat placeholder

    if config_path:
        config = TrainingConfig.from_yaml(config_path)
    else:
        config = TrainingConfig(
            agent_type=agent_type,
            total_timesteps=total_timesteps,
            **kwargs,
        )

    trainer = Trainer(config)
    try:
        trainer.setup()
        trainer.train()
    finally:
        trainer.close()
    return trainer.agent

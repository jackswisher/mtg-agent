"""Training module for MTG-Causal-RL.

This module provides training loops, evaluation utilities, and
callbacks for experiment tracking.  ``EnvConfig`` is the canonical MDP
specification shared between ``Trainer`` and ``Evaluator`` so train/eval
cannot silently drift into different environments.
"""

from mtg.training.callbacks import (
    AdaptiveKLCallback,
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
    LeagueEnvWrapper,
    create_env,
    create_vec_env,
    evaluate_policy_on_env,
    make_env_fn,
)
from mtg.training.evaluate import (
    EvaluationConfig,
    EvaluationResult,
    Evaluator,
    bootstrap_ci,
    bootstrap_half_width,
    compare_agents,
    evaluate,
)
from mtg.training.league import (
    League,
    LeagueConfig,
    OpponentEntry,
    PFSPSampler,
    elo_update,
    expected_score,
    snapshot_policy,
)
from mtg.training.train import Trainer, TrainingConfig, train

__all__ = [
    "AdaptiveKLCallback",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "EntropyScheduleCallback",
    "EnvConfig",
    "EpisodeLoggerCallback",
    "EvalCallback",
    "EvaluationConfig",
    "EvaluationResult",
    "Evaluator",
    "League",
    "LeagueCallback",
    "LeagueConfig",
    "LeagueEnvWrapper",
    "MetricsCallback",
    "OpponentEntry",
    "PFSPSampler",
    "Trainer",
    "TrainingConfig",
    "bootstrap_ci",
    "bootstrap_half_width",
    "compare_agents",
    "create_env",
    "create_vec_env",
    "elo_update",
    "evaluate",
    "evaluate_policy_on_env",
    "expected_score",
    "make_env_fn",
    "snapshot_policy",
    "train",
]

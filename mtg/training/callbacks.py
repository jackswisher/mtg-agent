"""Training callbacks.

This module provides Stable-Baselines3-compatible callbacks that plug
directly into ``BaseAlgorithm.learn(..., callback=...)``. They run as
part of SB3's callback chain so checkpointing, early stopping, and
periodic evaluation all execute in lock-step with the rollout / train
loop.

Exposed callbacks:

* ``EpisodeLoggerCallback``: records per-episode reward / length / win
  from SB3 rollouts and optionally writes a JSON log on training end.
* ``CheckpointCallback``: serialises the wrapped agent (not just the
  raw SB3 model) every ``save_interval`` timesteps.
* ``EarlyStoppingCallback``: stops training when recent win rate has
  not improved by ``min_delta`` for ``patience`` eval windows.
* ``EvalCallback``: periodically evaluates on a frozen eval env and
  triggers best-model checkpointing.
* ``EntropyScheduleCallback``: linear / cosine entropy-coefficient
  annealing that works from any ``Trainer`` entry point (not just
  ``scripts/runner/run_training.py``).
* ``MetricsCallback``: lightweight accumulator used by ``Trainer`` to
  return per-episode metrics for downstream plots.

Every callback here inherits from ``stable_baselines3.common.callbacks.BaseCallback``
(with a fallback to ``sb3_contrib.common.callbacks.BaseCallback``) so
they can be combined with SB3's ``CallbackList``.
"""

from __future__ import annotations

import json
import math
import time
import typing as tp
from pathlib import Path

import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback as _SB3BaseCallback
    from stable_baselines3.common.callbacks import CallbackList

    HAS_SB3 = True
except ImportError:  # pragma: no cover - SB3 is a hard dep for training
    try:
        from sb3_contrib.common.callbacks import (
            BaseCallback as _SB3BaseCallback,  # type: ignore[no-redef]
        )
        from sb3_contrib.common.callbacks import CallbackList  # type: ignore[no-redef]

        HAS_SB3 = True
    except ImportError:
        HAS_SB3 = False

        class _SB3BaseCallback:  # type: ignore[no-redef]
            """Fallback stub used when SB3 is unavailable."""

            def __init__(self, verbose: int = 0):
                self.verbose = verbose
                self.model: tp.Any = None
                self.num_timesteps = 0
                self.logger: tp.Any = None

            def init_callback(self, model: tp.Any) -> None:
                self.model = model

            def on_step(self) -> bool:
                return True

            def on_training_start(self, *args: tp.Any, **kwargs: tp.Any) -> None:
                pass

            def on_training_end(self) -> None:
                pass

            def on_rollout_start(self) -> None:
                pass

            def on_rollout_end(self) -> None:
                pass

            @property
            def locals(self) -> dict[str, tp.Any]:  # pragma: no cover
                return {}

        class CallbackList(_SB3BaseCallback):  # type: ignore[no-redef]
            """Fallback CallbackList used when SB3 is unavailable."""

            def __init__(self, callbacks: list[_SB3BaseCallback]) -> None:
                super().__init__()
                self.callbacks = callbacks

            def _on_step(self) -> bool:
                return all(cb.on_step() for cb in self.callbacks)


__all__ = [
    "AdaptiveKLCallback",
    "BaseCallback",
    "CallbackList",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "EntropyScheduleCallback",
    "EpisodeLoggerCallback",
    "EvalCallback",
    "LeagueCallback",
    "MetricsCallback",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_game_result(info: dict[str, tp.Any]) -> str | None:
    """Return ``"win" | "loss" | "draw"`` from an info dict."""
    if "terminal_info" in info and isinstance(info["terminal_info"], dict):
        info = info["terminal_info"]
    result = info.get("game_result")
    if isinstance(result, str):
        return result.lower()
    return None


def _extract_win(info: dict[str, tp.Any]) -> bool:
    result = _extract_game_result(info)
    return result == "win"


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseCallback(_SB3BaseCallback):
    """Project-local base that adapts SB3's ``BaseCallback`` protocol.

    All callbacks in this module must implement ``_on_step`` and can
    optionally override ``_on_rollout_start`` / ``_on_rollout_end``.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)

    def _on_step(self) -> bool:  # pragma: no cover - overridden by children
        return True


# ---------------------------------------------------------------------------
# Episode metrics / logging
# ---------------------------------------------------------------------------


class MetricsCallback(BaseCallback):
    """Accumulate per-episode reward, length, win flag from SB3 rollouts.

    Trainer reads this callback's ``episode_*`` lists after ``learn``
    completes to produce a plain-python metrics dict.  Unlike
    ``DisplayCallback`` in ``scripts/runner/run_training.py`` this one
    has no Rich / terminal dependencies and is safe to import in tests.
    """

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_wins: list[float] = []
        self._current_rewards: dict[int, float] = {}
        self._current_lengths: dict[int, int] = {}

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", np.array([False]))
        rewards = self.locals.get("rewards", np.array([0.0]))
        infos = self.locals.get("infos", [{}])

        for i in range(len(dones)):
            r = float(rewards[i]) if i < len(rewards) else 0.0
            self._current_rewards[i] = self._current_rewards.get(i, 0.0) + r
            self._current_lengths[i] = self._current_lengths.get(i, 0) + 1

            if bool(dones[i]):
                info = infos[i] if i < len(infos) else {}
                self.episode_rewards.append(self._current_rewards.pop(i, 0.0))
                self.episode_lengths.append(self._current_lengths.pop(i, 0))
                self.episode_wins.append(1.0 if _extract_win(info) else 0.0)

        return True


class EpisodeLoggerCallback(BaseCallback):
    """Log per-episode metrics to disk (JSON) and stdout."""

    def __init__(
        self,
        log_dir: str | Path = "results/logs",
        log_interval: int = 10,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = max(1, log_interval)

        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_wins: list[float] = []
        self._current_rewards: dict[int, float] = {}
        self._current_lengths: dict[int, int] = {}
        self._start_time = time.time()

    def _on_training_start(self) -> None:
        self._start_time = time.time()

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", np.array([False]))
        rewards = self.locals.get("rewards", np.array([0.0]))
        infos = self.locals.get("infos", [{}])

        for i in range(len(dones)):
            r = float(rewards[i]) if i < len(rewards) else 0.0
            self._current_rewards[i] = self._current_rewards.get(i, 0.0) + r
            self._current_lengths[i] = self._current_lengths.get(i, 0) + 1

            if bool(dones[i]):
                info = infos[i] if i < len(infos) else {}
                ep_reward = self._current_rewards.pop(i, 0.0)
                ep_length = self._current_lengths.pop(i, 0)
                won = 1.0 if _extract_win(info) else 0.0
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                self.episode_wins.append(won)

                n = len(self.episode_rewards)
                if self.verbose > 0 and n % self.log_interval == 0:
                    recent = self.episode_rewards[-self.log_interval :]
                    recent_wins = self.episode_wins[-self.log_interval :]
                    elapsed = time.time() - self._start_time
                    print(
                        f"[episode] n={n} "
                        f"mean_reward={np.mean(recent):.3f} "
                        f"win_rate={np.mean(recent_wins):.2%} "
                        f"elapsed={elapsed:.1f}s"
                    )
        return True

    def _on_training_end(self) -> None:
        log_path = self.log_dir / "episode_log.json"
        with open(log_path, "w") as f:
            json.dump(
                {
                    "episode_rewards": self.episode_rewards,
                    "episode_lengths": self.episode_lengths,
                    "episode_wins": self.episode_wins,
                    "total_time": time.time() - self._start_time,
                },
                f,
                indent=2,
            )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


class CheckpointCallback(BaseCallback):
    """Periodically save the *wrapped* agent (not just the raw SB3 model).

    Resolves the project-level agent through ``self.model._ppo_agent_ref``
    set by ``PPOAgent.initialize_model`` / ``CausalAgent.initialize_model``;
    falls back to saving the raw SB3 model if that reference is missing.

    Args:
        save_dir: Directory to write checkpoints to.
        save_interval: Save every ``save_interval`` timesteps.
        name_prefix: Filename prefix for numeric checkpoints.
        save_final: If True, always save a final checkpoint when training ends.
    """

    def __init__(
        self,
        save_dir: str | Path,
        save_interval: int = 50_000,
        name_prefix: str = "checkpoint",
        save_final: bool = True,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = max(1, int(save_interval))
        self.name_prefix = name_prefix
        self.save_final = save_final
        self._last_save_step = 0

    def _resolve_agent(self) -> tp.Any:
        """Return the project-level agent if attached, else the raw model."""
        return getattr(self.model, "_ppo_agent_ref", self.model)

    def _save(self, step: int) -> None:
        path = self.save_dir / f"{self.name_prefix}_{step}"
        agent = self._resolve_agent()
        if hasattr(agent, "save"):
            agent.save(str(path))
        elif hasattr(self.model, "save"):
            self.model.save(str(path))
        if self.verbose:
            print(f"[checkpoint] saved to {path} @ step={step}")

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save_step >= self.save_interval:
            self._last_save_step = self.num_timesteps
            self._save(self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        if self.save_final:
            self._save(self.num_timesteps)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class EarlyStoppingCallback(BaseCallback):
    """Stop training when a rolling metric stops improving.

    By default this monitors episode win rate over the last
    ``window`` completed episodes.  Evaluation happens every
    ``check_interval`` timesteps.

    Args:
        patience: Number of consecutive non-improving checks before stop.
        min_delta: Minimum improvement needed to count as "improved".
        window: Number of recent episodes to average over.
        check_interval: Evaluate the metric every N timesteps.
        metric: "win_rate" (default) or "reward".
        mode: "max" (default) or "min" (useful if metric is a loss).
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.01,
        window: int = 100,
        check_interval: int = 25_000,
        metric: str = "win_rate",
        mode: str = "max",
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.patience = max(1, patience)
        self.min_delta = float(min_delta)
        self.window = max(1, window)
        self.check_interval = max(1, check_interval)
        self.metric = metric
        self.mode = mode.lower()
        assert self.mode in {"max", "min"}
        assert self.metric in {"win_rate", "reward"}

        self._episode_rewards: list[float] = []
        self._episode_wins: list[float] = []
        self._current_rewards: dict[int, float] = {}
        self._last_check_step = 0
        self._best_value = -math.inf if self.mode == "max" else math.inf
        self._counter = 0

    def _better(self, value: float) -> bool:
        if self.mode == "max":
            return value > self._best_value + self.min_delta
        return value < self._best_value - self.min_delta

    def _current_metric(self) -> float | None:
        if self.metric == "win_rate":
            recent = self._episode_wins[-self.window :]
        else:
            recent = self._episode_rewards[-self.window :]
        if not recent:
            return None
        return float(np.mean(recent))

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", np.array([False]))
        rewards = self.locals.get("rewards", np.array([0.0]))
        infos = self.locals.get("infos", [{}])
        for i in range(len(dones)):
            r = float(rewards[i]) if i < len(rewards) else 0.0
            self._current_rewards[i] = self._current_rewards.get(i, 0.0) + r
            if bool(dones[i]):
                info = infos[i] if i < len(infos) else {}
                self._episode_rewards.append(self._current_rewards.pop(i, 0.0))
                self._episode_wins.append(1.0 if _extract_win(info) else 0.0)

        if self.num_timesteps - self._last_check_step < self.check_interval:
            return True
        self._last_check_step = self.num_timesteps

        value = self._current_metric()
        if value is None:
            return True

        if self._better(value):
            self._best_value = value
            self._counter = 0
        else:
            self._counter += 1
            if self.verbose:
                print(
                    f"[early_stopping] no improvement: metric={self.metric} "
                    f"value={value:.3f} best={self._best_value:.3f} "
                    f"counter={self._counter}/{self.patience}"
                )
            if self._counter >= self.patience:
                if self.verbose:
                    print(
                        f"[early_stopping] stopping at step={self.num_timesteps} "
                        f"(no improvement for {self.patience} checks)"
                    )
                return False
        return True


# ---------------------------------------------------------------------------
# Periodic evaluation
# ---------------------------------------------------------------------------


class EvalCallback(BaseCallback):
    """Periodic evaluation on a frozen eval environment.

    After every ``eval_interval`` timesteps the current agent is run on
    the provided eval env for ``n_eval_episodes`` episodes and metrics
    are written to ``log_dir/eval_log.json``.  If the mean win rate
    improves, the agent is saved to ``log_dir/best_model.zip``.

    The eval env is supplied *as an EnvConfig* (not a live env) so the
    callback can construct a fresh env each evaluation and avoid
    coupling training and eval state.
    """

    def __init__(
        self,
        eval_env_config: tp.Any,
        n_eval_episodes: int = 20,
        eval_interval: int = 50_000,
        log_dir: str | Path = "results/eval",
        deterministic: bool = True,
        save_best: bool = True,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_env_config = eval_env_config
        self.n_eval_episodes = max(1, n_eval_episodes)
        self.eval_interval = max(1, eval_interval)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.deterministic = deterministic
        self.save_best = save_best

        self._last_eval_step = 0
        self._best_win_rate = -math.inf
        self._history: list[dict[str, tp.Any]] = []

    def _resolve_agent(self) -> tp.Any:
        return getattr(self.model, "_ppo_agent_ref", self.model)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_interval:
            return True
        self._last_eval_step = self.num_timesteps
        self._run_eval()
        return True

    def _run_eval(self) -> None:
        from mtg.training.env_factory import (
            create_env,
            evaluate_policy_on_env,
            make_obs_normaliser_from_vec_normalize,
        )

        agent = self._resolve_agent()
        eval_env = create_env(self.eval_env_config, seed_offset=10_000)

        # If the trainer wrapped its rollout env in VecNormalize, the
        # policy was trained on normalised observations. The same
        # running statistics MUST be applied during periodic eval;
        # otherwise the reported win-rate curve is on a different
        # observation distribution from what the agent learned on.
        try:
            from stable_baselines3.common.vec_env import VecNormalize

            train_env = getattr(self.model, "env", None)
            train_vn = train_env if isinstance(train_env, VecNormalize) else None
        except ImportError:
            train_vn = None
        obs_normaliser = make_obs_normaliser_from_vec_normalize(train_vn)

        result = evaluate_policy_on_env(
            eval_env,
            agent,
            n_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
            obs_normaliser=obs_normaliser,
            seed_offset=10_000,
        )
        record = {
            "step": int(self.num_timesteps),
            "win_rate": result["win_rate"],
            "mean_reward": result["mean_reward"],
            "mean_length": result["mean_length"],
        }
        self._history.append(record)
        with open(self.log_dir / "eval_log.json", "w") as f:
            json.dump(self._history, f, indent=2)

        if self.verbose:
            print(
                f"[eval] step={record['step']} "
                f"win_rate={record['win_rate']:.2%} "
                f"mean_reward={record['mean_reward']:.3f}"
            )

        if self.save_best and result["win_rate"] > self._best_win_rate:
            self._best_win_rate = result["win_rate"]
            best_path = self.log_dir / "best_model"
            if hasattr(agent, "save"):
                agent.save(str(best_path))
            elif hasattr(self.model, "save"):
                self.model.save(str(best_path))


# ---------------------------------------------------------------------------
# Entropy schedule
# ---------------------------------------------------------------------------


class EntropyScheduleCallback(BaseCallback):
    """Linear / cosine entropy-coefficient annealing.

    Implemented as an SB3 callback so every training entry point
    (``Trainer.train``, ablation scripts, research workflows) gets the
    same annealing behaviour.

    Args:
        ent_coef_start: Initial coefficient at progress=1.0.
        ent_coef_end: Final coefficient at progress=0.0.
        total_timesteps: Total scheduler horizon. The ``progress`` used
            for interpolation is ``1 - num_timesteps / total_timesteps``.
        schedule: ``"linear"`` (default) or ``"cosine"``.
    """

    def __init__(
        self,
        ent_coef_start: float,
        ent_coef_end: float,
        total_timesteps: int,
        schedule: str = "linear",
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.ent_coef_start = float(ent_coef_start)
        self.ent_coef_end = float(ent_coef_end)
        self.total_timesteps = max(1, int(total_timesteps))
        self.schedule = schedule
        assert schedule in {"linear", "cosine"}

    def _coef_for_progress(self, progress_remaining: float) -> float:
        p = max(0.0, min(1.0, float(progress_remaining)))
        if self.schedule == "linear":
            return self.ent_coef_end + (self.ent_coef_start - self.ent_coef_end) * p
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * (1.0 - p)))
        return self.ent_coef_end + (self.ent_coef_start - self.ent_coef_end) * cos_factor

    def _on_step(self) -> bool:
        if self.model is None:
            return True
        progress = 1.0 - (self.num_timesteps / self.total_timesteps)
        self.model.ent_coef = self._coef_for_progress(progress)
        return True


# ---------------------------------------------------------------------------
# Adaptive KL controller
# ---------------------------------------------------------------------------


class AdaptiveKLCallback(BaseCallback):
    """Adapt PPO's clip range to keep the measured policy KL inside a band.

    Optionally co-scales the learning rate by the same factor.

    Rationale
    ---------
    A fixed ``clip_range=0.2`` is a blunt instrument: early in training
    the policy barely changes (measured KL ~ 1e-4) and late in training
    the policy can still take oversized steps on high-advantage
    minibatches (measured KL ~ 5e-2, triggering SB3's ``target_kl``
    early-epoch termination). Following Heess et al. (2017) and the
    Dota Five / OpenAI baselines implementations the trust region size
    is adapted multiplicatively in response to the most recent update's
    mean KL:

        if kl > 1.5 * target_kl:   clip *= 0.8     # tighten
        if kl < target_kl / 1.5:   clip *= 1.25    # relax

    Clip range is rebuilt as a constant schedule (``lambda _: clip``) so
    SB3 picks up the new value on its next ``train`` call.

    Args:
        target_kl: Desired per-update mean KL.
        increase_factor / decrease_factor: Multiplicative updates.
        min_clip / max_clip: Safety bounds so the policy never collapses
            to argmax nor walks arbitrarily far.
        adapt_lr: If True, also scale learning rate by the same factor
            (recommended when combined with a decaying lr schedule).
    """

    def __init__(
        self,
        target_kl: float = 0.02,
        increase_factor: float = 1.25,
        decrease_factor: float = 0.8,
        min_clip: float = 0.05,
        max_clip: float = 0.3,
        adapt_lr: bool = False,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.target_kl = float(target_kl)
        self.increase_factor = float(increase_factor)
        self.decrease_factor = float(decrease_factor)
        self.min_clip = float(min_clip)
        self.max_clip = float(max_clip)
        self.adapt_lr = adapt_lr
        self._history: list[dict[str, float]] = []

    def _current_clip(self) -> float | None:
        if self.model is None:
            return None
        cr = getattr(self.model, "clip_range", None)
        if callable(cr):
            try:
                return float(cr(1.0))
            except Exception:
                return None
        if cr is None:
            return None
        try:
            return float(cr)
        except (TypeError, ValueError):
            return None

    def _set_clip(self, value: float) -> None:
        value = float(np.clip(value, self.min_clip, self.max_clip))

        def _const(_: float, _value: float = value) -> float:
            return _value

        self.model.clip_range = _const

    def _on_rollout_end(self) -> None:
        if self.model is None:
            return
        logger = getattr(self.model, "logger", None)
        if logger is None:
            return
        # SB3 stashes the last-update approx_kl in logger.name_to_value
        kl = None
        try:
            kl = logger.name_to_value.get("train/approx_kl")
        except Exception:
            kl = None
        if kl is None:
            return

        clip = self._current_clip()
        if clip is None:
            return

        new_clip = clip
        if kl > 1.5 * self.target_kl:
            new_clip = clip * self.decrease_factor
        elif kl < self.target_kl / 1.5:
            new_clip = clip * self.increase_factor

        if not np.isclose(new_clip, clip):
            self._set_clip(new_clip)
            if self.adapt_lr and hasattr(self.model, "learning_rate"):
                lr = self.model.learning_rate
                if callable(lr):
                    try:
                        base = float(lr(1.0))
                    except Exception:
                        base = None
                else:
                    try:
                        base = float(lr)
                    except (TypeError, ValueError):
                        base = None
                if base is not None:
                    ratio = new_clip / max(clip, 1e-8)
                    new_lr = float(np.clip(base * ratio, 1e-6, 1e-2))

                    def _const_lr(_: float, _value: float = new_lr) -> float:
                        return _value

                    self.model.learning_rate = _const_lr

            if self.verbose:
                print(
                    f"[adaptive_kl] kl={kl:.4f} clip {clip:.4f} -> "
                    f"{new_clip:.4f} (target={self.target_kl:.4f})"
                )

        self._history.append({"kl": float(kl), "clip": float(new_clip)})

    def _on_step(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# League integration
# ---------------------------------------------------------------------------


class LeagueCallback(BaseCallback):
    """Update a ``League`` after each completed episode.

    This callback is purely observational: it consumes the
    ``episode_winner`` / ``game_result`` info field produced by the
    environment, matches it to the currently-active opponent reported
    by the env, and forwards the result to ``League.record_match``.

    It can optionally dump policy snapshots into the league at a
    fixed timestep cadence so the learner starts playing against its
    own past selves after an initial warm-up.
    """

    def __init__(
        self,
        league: tp.Any,
        snapshot_interval: int | None = None,
        snapshot_after: int = 0,
        log_dir: str | Path | None = None,
        flush_every_n_matches: int = 50,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.league = league
        self.snapshot_interval = snapshot_interval
        self.snapshot_after = int(snapshot_after)
        self._last_snapshot_step = 0
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.flush_every_n_matches = max(1, int(flush_every_n_matches))
        self._matches_since_flush = 0

    def _league_payload(self) -> dict[str, tp.Any]:
        return {
            "final_learner_rating": self.league.learner_rating,
            "standings": self.league.standings(),
            "match_history": [
                {
                    "opponent": m.opponent,
                    "win": m.win,
                    "learner_rating_after": m.learner_rating_after,
                }
                for m in self.league.match_history
            ],
        }

    def _flush_league_json(self) -> None:
        """Atomically dump the current league state to ``league.json``.

        Called periodically and at training end so a crash mid-run still
        leaves the most recent league snapshot on disk.
        """
        if self.log_dir is None:
            return
        import json

        target = self.log_dir / "league.json"
        tmp = target.with_suffix(target.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(self._league_payload(), f, indent=2)
        tmp.replace(target)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", np.array([False]))
        infos = self.locals.get("infos", [{}])

        for i in range(len(dones)):
            if not bool(dones[i]):
                continue
            info = infos[i] if i < len(infos) else {}
            won = _extract_win(info)
            opp_name = info.get("active_opponent")
            if opp_name is None and "terminal_info" in info:
                opp_name = info["terminal_info"].get("active_opponent")
            if opp_name is None:
                continue
            try:
                self.league.record_match(opp_name, won)
                self._matches_since_flush += 1
            except KeyError:
                # Opponent no longer in the pool (evicted); skip.
                continue

        if self._matches_since_flush >= self.flush_every_n_matches:
            self._matches_since_flush = 0
            try:
                self._flush_league_json()
            except OSError as e:
                if self.verbose:
                    print(f"[league] periodic flush failed: {e}")

        if (
            self.snapshot_interval is not None
            and self.num_timesteps - self._last_snapshot_step >= self.snapshot_interval
            and self.num_timesteps >= self.snapshot_after
        ):
            self._last_snapshot_step = self.num_timesteps
            agent = getattr(self.model, "_ppo_agent_ref", None)
            if agent is not None:
                try:
                    from mtg.training.league import snapshot_policy

                    snapshot_policy(self.league, agent, deck="learner")
                    if self.verbose:
                        print(f"[league] snapshotted learner at step={self.num_timesteps}")
                except Exception as e:  # noqa: BLE001
                    if self.verbose:
                        print(f"[league] snapshot failed: {e}")

        return True

    def _on_training_end(self) -> None:
        if self.log_dir is None:
            return
        try:
            self._flush_league_json()
        except OSError as e:
            if self.verbose:
                print(f"[league] final flush failed: {e}")

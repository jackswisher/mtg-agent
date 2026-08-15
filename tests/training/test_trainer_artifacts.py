"""End-to-end Trainer artefact-on-disk tests.

These are the smallest possible runs that exercise the full
``Trainer.setup -> train -> close`` lifecycle and assert that the
side-effect artefacts other tools depend on actually land on disk
and that downstream consumers actually use them:

* ``vec_normalize.pkl``: written when ``enable_vec_normalize=True``
  and loaded back by :class:`mtg.training.evaluate.Evaluator` and by
  :class:`mtg.training.callbacks.EvalCallback` so that policies
  trained on normalised observations are evaluated on the same
  observation distribution.
* ``league.json``: written by :class:`LeagueCallback` when
  ``enable_league=True``; consumed by analysis tooling and the league
  resume / reporting paths.

The component-level behaviour of these subsystems is covered by
``test_features_and_kl.py`` (feature stack) and ``test_league.py``
(League / PFSP / Elo). These tests close the remaining gap between
"the components work" and "the Trainer wires them up and persists
their state to the canonical paths".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# The Trainer end-to-end path requires SB3 + torch.  Skip the whole
# module if either is missing rather than emitting one ImportError per
# test.
pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from mtg.training import Trainer, TrainingConfig  # noqa: E402

# ---------------------------------------------------------------------------
# VecNormalize artefact
# ---------------------------------------------------------------------------


def _make_vec_normalize_trainer(tmp_path: Path) -> Trainer:
    """Build a tiny PPO Trainer with VecNormalize enabled."""
    cfg = TrainingConfig(
        agent_type="ppo",
        deck_archetype="mono_red_aggro",
        opponent_archetype="azorius_control",
        reward_type="shaped",
        # Aggressive caps so we get one full PPO update + the
        # end-of-train save block fast (this test only needs the
        # save block to fire, not real learning).
        max_turns=5,
        max_steps_per_episode=60,
        auto_combat=True,
        auto_target=True,
        total_timesteps=256,
        n_envs=1,
        seed=0,
        enable_vec_normalize=True,
        enable_checkpointing=False,
        enable_periodic_eval=False,
        enable_episode_logger=False,
        enable_entropy_schedule=False,
        enable_early_stopping=False,
        enable_adaptive_kl=False,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpts"),
        experiment_name="test_vec_normalize",
        agent_kwargs={
            "n_steps": 128,
            "batch_size": 64,
            "n_epochs": 2,
            "learning_rate": 3e-4,
            "use_set_encoder": False,
        },
    )
    return Trainer(cfg)


def test_trainer_saves_vec_normalize_pkl_after_training(tmp_path: Path) -> None:
    """``enable_vec_normalize=True`` persists running stats to disk.

    The eval pipeline loads ``vec_normalize.pkl`` to re-apply the
    training-time observation normalisation, so this artefact must
    land at the canonical path inside the experiment checkpoint dir.
    """
    trainer = _make_vec_normalize_trainer(tmp_path)
    trainer.setup()
    try:
        trainer.train()
    finally:
        trainer.close()

    vn_path = trainer.checkpoint_dir / "vec_normalize.pkl"
    assert vn_path.exists(), f"VecNormalize stats not saved at {vn_path}"
    assert vn_path.stat().st_size > 0, "VecNormalize pickle is empty"


def test_evaluator_loads_and_applies_vec_normalize_stats(tmp_path: Path) -> None:
    """``Evaluator(vec_normalize_path=...)`` actually re-applies stats.

    Two things must be true:

    1. The Evaluator picks up the saved ``obs_rms`` from
       ``vec_normalize.pkl`` (so its ``_obs_rms`` attribute is non-None
       after ``setup()``).
    2. ``_normalise_obs`` actually transforms a probe observation
       differently from the identity (i.e. there is a non-zero
       running mean OR non-unit running variance), so the agent
       sees normalised inputs at eval time.
    """
    import numpy as np

    from mtg.training.evaluate import EvaluationConfig, Evaluator

    trainer = _make_vec_normalize_trainer(tmp_path)
    trainer.setup()
    try:
        trainer.train()
    finally:
        trainer.close()

    vn_path = trainer.checkpoint_dir / "vec_normalize.pkl"
    assert vn_path.exists(), "precondition: vec_normalize.pkl must be on disk"

    eval_cfg = EvaluationConfig(
        deck_archetype="mono_red_aggro",
        opponent_archetype="azorius_control",
        max_turns=5,
        max_steps_per_episode=60,
        n_episodes=1,
        seed=0,
        vec_normalize_path=str(vn_path),
    )
    evaluator = Evaluator(eval_cfg, vec_normalize_path=str(vn_path))
    evaluator.setup()

    assert (
        evaluator._obs_rms is not None
    ), "Evaluator never loaded VecNormalize stats; vec_normalize_path was silently ignored."
    assert evaluator.env is not None
    obs, _ = evaluator.env.reset(seed=42)
    normed = evaluator._normalise_obs(obs)
    assert normed.shape == obs.shape

    moved_mean = bool(np.any(np.abs(evaluator._obs_rms.mean) > 1e-6))
    moved_var = bool(np.any(np.abs(evaluator._obs_rms.var - 1.0) > 1e-3))
    assert moved_mean or moved_var, (
        "Loaded obs_rms is still at the (mean=0, var=1) initialisation; "
        "running stats were not actually populated by the trainer."
    )

    diff = float(np.max(np.abs(normed.astype(float) - obs.astype(float))))
    assert diff > 0.0, (
        "Normalised obs is bit-identical to raw obs; "
        "Evaluator is not actually applying the loaded stats."
    )


def test_eval_callback_uses_vec_normalize_stats_during_training(tmp_path: Path) -> None:
    """Periodic eval inside training also normalises observations.

    Builds a Trainer with both ``enable_vec_normalize=True`` and
    ``enable_periodic_eval=True``, monkey-patches
    ``evaluate_policy_on_env`` to capture the ``obs_normaliser``
    argument, and asserts:

    * the EvalCallback passed in a non-None normaliser, and
    * applying that normaliser to a fresh observation produces a
      different array than the identity.
    """
    import numpy as np

    cfg = TrainingConfig(
        agent_type="ppo",
        deck_archetype="mono_red_aggro",
        opponent_archetype="azorius_control",
        reward_type="shaped",
        max_turns=5,
        max_steps_per_episode=60,
        auto_combat=True,
        auto_target=True,
        total_timesteps=256,
        n_envs=1,
        seed=0,
        enable_vec_normalize=True,
        enable_checkpointing=False,
        enable_periodic_eval=True,
        eval_interval=128,
        eval_episodes=1,
        enable_episode_logger=False,
        enable_entropy_schedule=False,
        enable_early_stopping=False,
        enable_adaptive_kl=False,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpts"),
        experiment_name="test_eval_callback_normalises",
        agent_kwargs={
            "n_steps": 128,
            "batch_size": 64,
            "n_epochs": 2,
            "learning_rate": 3e-4,
            "use_set_encoder": False,
        },
    )
    trainer = Trainer(cfg)

    captured: dict[str, object] = {"normaliser": None, "called": False}

    from mtg.training import env_factory as _env_factory

    real_evaluate = _env_factory.evaluate_policy_on_env

    def _spy_evaluate_policy_on_env(*args: object, **kwargs: object) -> dict[str, float]:
        captured["called"] = True
        captured["normaliser"] = kwargs.get("obs_normaliser")
        return {
            "n_episodes": 1,
            "win_rate": 0.0,
            "mean_reward": 0.0,
            "mean_length": 0.0,
            "std_reward": 0.0,
            "std_length": 0.0,
            "rewards": [0.0],
            "lengths": [1],
            "wins": [False],
        }

    _env_factory.evaluate_policy_on_env = _spy_evaluate_policy_on_env  # type: ignore[assignment]
    try:
        trainer.setup()
        try:
            trainer.train()
        finally:
            trainer.close()
    finally:
        _env_factory.evaluate_policy_on_env = real_evaluate  # type: ignore[assignment]

    assert captured["called"], "EvalCallback never invoked evaluate_policy_on_env"
    normaliser = captured["normaliser"]
    assert normaliser is not None, (
        "EvalCallback called the eval loop without an obs_normaliser; "
        "the train/eval distribution mismatch is back."
    )
    assert callable(normaliser)
    # The bound obs_rms must have moved off its (mean=0, var=1) prior;
    # if not, the trainer's VecNormalize never accumulated stats.
    inner_obs_rms = getattr(normaliser, "__closure__", None)
    assert inner_obs_rms is not None
    closures = [c.cell_contents for c in normaliser.__closure__]  # type: ignore[union-attr]
    obs_rms_match = next(
        (c for c in closures if hasattr(c, "mean") and hasattr(c, "var")),
        None,
    )
    assert obs_rms_match is not None, "Normaliser does not close over an obs_rms"
    assert obs_rms_match.mean.shape == obs_rms_match.var.shape

    probe = np.zeros_like(obs_rms_match.mean, dtype=np.float32)
    out = normaliser(probe)
    assert out.shape == probe.shape
    assert isinstance(out, np.ndarray)
    assert bool(np.any(np.abs(obs_rms_match.mean) > 1e-6)) or bool(
        np.any(np.abs(obs_rms_match.var - 1.0) > 1e-3)
    ), "Bound obs_rms is still at the (mean=0, var=1) initialisation"


# ---------------------------------------------------------------------------
# League artefact
# ---------------------------------------------------------------------------


def _make_league_trainer(tmp_path: Path) -> Trainer:
    """Build a tiny PPO Trainer with a 3-opponent league enabled."""
    cfg = TrainingConfig(
        agent_type="ppo",
        deck_archetype="mono_red_aggro",
        reward_type="shaped",
        # Aggressive caps so each episode terminates fast and we can
        # accumulate dozens of matches in a sub-second rollout buffer.
        max_turns=5,
        max_steps_per_episode=60,
        auto_combat=True,
        auto_target=True,
        total_timesteps=512,
        n_envs=1,
        seed=0,
        enable_league=True,
        league_opponents=[
            "mono_red_aggro",
            "azorius_control",
            "simic_ramp",
        ],
        league_sampling="pfsp",
        league_elo_k=32.0,
        league_snapshot_interval=None,
        enable_vec_normalize=False,
        enable_checkpointing=False,
        enable_periodic_eval=False,
        enable_episode_logger=False,
        enable_entropy_schedule=False,
        enable_early_stopping=False,
        enable_adaptive_kl=False,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpts"),
        experiment_name="test_league_artifacts",
        agent_kwargs={
            "n_steps": 128,
            "batch_size": 64,
            "n_epochs": 2,
            "learning_rate": 3e-4,
            "use_set_encoder": False,
        },
    )
    return Trainer(cfg)


def test_trainer_writes_league_json_with_standings_and_history(tmp_path: Path) -> None:
    """``enable_league=True`` produces a populated ``league.json`` on disk.

    Two-part end-to-end assertion:

    1. The file lands at the canonical ``<log_dir>/league/league.json``
       path with the expected top-level keys.
    2. The Elo bookkeeping actually ran: at least one opponent's
       rating has moved off the 1500 default, proving the
       ``LeagueCallback`` propagated match outcomes to the League
       (and is not silently dropping them).
    """
    trainer = _make_league_trainer(tmp_path)
    trainer.setup()
    try:
        assert trainer.league is not None
        assert (
            len(trainer.league.pool) == 3
        ), f"expected 3 opponents in the pool, got {len(trainer.league.pool)}"
        trainer.train()

        # In-memory ratings (matches the smoke_phase_c invariant).
        # League is still attached after train(); we read it before
        # close() to keep this independent of the on-disk payload.
        opponent_ratings = {e.name: e.rating for e in trainer.league.pool}
    finally:
        trainer.close()

    log_json = trainer.log_dir / "league" / "league.json"
    assert log_json.exists(), f"expected league log at {log_json}"

    payload = json.loads(log_json.read_text())
    assert "standings" in payload
    assert "match_history" in payload
    assert "final_learner_rating" in payload

    assert payload["match_history"], "no matches recorded; LeagueCallback never fired"
    assert any(
        abs(r - 1500.0) > 1e-6 for r in opponent_ratings.values()
    ), f"all opponent ratings stayed at 1500: {opponent_ratings}"

"""Tests for the CGFA calibration callback + plot + case study scripts.

Covers:
* :class:`CGFACalibrationCallback` writes a well-formed CSV with the
  ``cgfa/*`` logger keys after a short Trainer run.
* :func:`scripts.research.calibration_plot.render` produces a PNG when
  given the CSV from a real training run.
* :func:`scripts.research.case_study.rollout_one_episode` produces one
  row per environment step, each with the full per-factor decomposition.
* :func:`scripts.research.case_study.render_case_study` materialises the
  3-panel PNG given fabricated step records.
* :func:`scripts.runner.run_training._maybe_attach_cgfa_callback` only
  injects the calibration callback when the agent is CGFA, so the
  research pipeline (``mtg-research train``/``pipeline``) emits
  ``cgfa_calibration.csv`` for CGFA runs and is a no-op otherwise.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless backend for CI

from mtg.training.train import Trainer, TrainingConfig  # noqa: E402
from scripts.research.calibration_plot import render as render_calibration  # noqa: E402
from scripts.research.case_study import (  # noqa: E402
    render_case_study,
    rollout_one_episode,
    write_case_study_csv,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trainer(tmp_path: Path) -> Trainer:
    """Build a minimal CGFA Trainer that completes one PPO update fast."""
    cfg = TrainingConfig(
        agent_type="cgfa",
        deck_archetype="mono_red_aggro",
        opponent_archetype="mono_red_aggro",
        reward_type="shaped",
        max_turns=10,
        max_steps_per_episode=200,
        auto_combat=True,
        auto_target=True,
        total_timesteps=320,
        n_envs=1,
        seed=0,
        enable_checkpointing=False,
        enable_periodic_eval=False,
        enable_episode_logger=False,
        enable_entropy_schedule=False,
        enable_early_stopping=False,
        enable_adaptive_kl=False,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpts"),
        experiment_name="test_cgfa_calibration",
        agent_kwargs={
            "n_steps": 128,
            "batch_size": 64,
            "n_epochs": 2,
            "learning_rate": 3e-4,
            "intervention_calibration_coef": 0.1,
            "learnable_gate": True,
            "state_conditional_gate": True,
        },
    )
    return Trainer(cfg)


# ---------------------------------------------------------------------------
# CGFACalibrationCallback: CSV is written and contains cgfa/* columns
# ---------------------------------------------------------------------------


def test_calibration_callback_emits_csv(tmp_path: Path) -> None:
    """After a short CGFA training run a populated CSV is on disk."""
    trainer = _make_trainer(tmp_path)
    trainer.setup()
    try:
        trainer.train()
        csv_path = trainer.log_dir / "cgfa" / "cgfa_calibration.csv"
        assert csv_path.exists(), f"expected calibration CSV at {csv_path}"

        with csv_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) >= 1, "calibration CSV has no rows after training"

        header = rows[0].keys()
        assert "step" in header
        assert "n_updates" in header
        # At least one cgfa/* column must be present.
        cgfa_cols = [k for k in header if k.startswith("cgfa/")]
        assert cgfa_cols, "no cgfa/* columns recorded in calibration CSV"

        # We expect per-factor keys: gate, factor_corr, factor_share.
        names = trainer.agent.factor_spec.names
        assert "cgfa/gate/mean" in header
        assert any(f"cgfa/factor_corr/{n}" in header for n in names)
        assert any(f"cgfa/factor_share/{n}" in header for n in names)
    finally:
        trainer.close()


# ---------------------------------------------------------------------------
# Calibration plot script: render() produces a PNG from the CSV
# ---------------------------------------------------------------------------


def test_calibration_plot_renders_png(tmp_path: Path) -> None:
    """End-to-end: train -> CSV -> calibration_plot.render() -> PNG."""
    trainer = _make_trainer(tmp_path)
    trainer.setup()
    try:
        trainer.train()
        csv_path = trainer.log_dir / "cgfa" / "cgfa_calibration.csv"
        assert csv_path.exists()

        out = tmp_path / "calibration.png"
        render_calibration([csv_path], out, labels=["seed0"])
        assert out.exists()
        # PNG signature must be present.
        assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    finally:
        trainer.close()


def test_calibration_plot_handles_missing_columns_gracefully(tmp_path: Path) -> None:
    """A CSV without ``cgfa/*`` columns still renders (empty panels)."""
    csv_path = tmp_path / "minimal.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "n_updates"])
        writer.writeheader()
        writer.writerow({"step": 0, "n_updates": 0})
        writer.writerow({"step": 100, "n_updates": 1})
    out = tmp_path / "fallback.png"
    render_calibration([csv_path], out)
    assert out.exists()


def test_calibration_callback_writes_csv_for_single_update_run(tmp_path: Path) -> None:
    """Regression: when only ONE PPO ``train()`` fits in the budget the CSV is still written.

    With ``n_envs=N`` and a budget that fits exactly one rollout, SB3's
    ``OnPolicyAlgorithm.learn`` loop runs:

    1. ``collect_rollouts`` (calls callback ``on_rollout_end`` -- empty
       because no ``train()`` has happened yet);
    2. ``train()`` (populates ``cgfa/*`` keys in ``logger.name_to_value``);
    3. exits the loop because ``num_timesteps >= total_timesteps``.

    The original implementation only snapshotted on ``on_rollout_end``,
    so this case left a header-only (empty) CSV.  The fix snapshots on
    ``on_step`` (gated by ``_n_updates``) and on ``on_training_end``,
    so the final ``train()``'s metrics still land in the CSV.
    """
    cfg = TrainingConfig(
        agent_type="cgfa",
        deck_archetype="mono_red_aggro",
        opponent_archetype="mono_red_aggro",
        reward_type="shaped",
        max_turns=10,
        max_steps_per_episode=200,
        auto_combat=True,
        auto_target=True,
        # Budget = exactly one rollout (n_steps * n_envs); kept tiny on
        # purpose to surface the "one PPO update only" edge case.
        total_timesteps=128,
        n_envs=1,
        seed=0,
        enable_checkpointing=False,
        enable_periodic_eval=False,
        enable_episode_logger=False,
        enable_entropy_schedule=False,
        enable_early_stopping=False,
        enable_adaptive_kl=False,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpts"),
        experiment_name="test_cgfa_single_update",
        agent_kwargs={
            "n_steps": 128,
            "batch_size": 64,
            "n_epochs": 1,
            "learning_rate": 3e-4,
            "intervention_calibration_coef": 0.0,
            "learnable_gate": True,
            "state_conditional_gate": False,
        },
    )
    trainer = Trainer(cfg)
    trainer.setup()
    try:
        trainer.train()
        csv_path = trainer.log_dir / "cgfa" / "cgfa_calibration.csv"
        assert csv_path.exists(), "single-update CGFA run produced no CSV"
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 1, (
            "single-update CGFA run wrote header but no data rows -- "
            "the on_training_end snapshot is missing"
        )
        # And the row must contain real cgfa/* metrics, not just step/n_updates.
        cgfa_cols = [k for k in rows[0] if k.startswith("cgfa/")]
        assert cgfa_cols, "single-update CSV row has no cgfa/* metrics"
    finally:
        trainer.close()


# ---------------------------------------------------------------------------
# Case study: rollout, CSV writer, figure renderer
# ---------------------------------------------------------------------------


def test_case_study_rollout_records_one_row_per_step(tmp_path: Path) -> None:
    """``rollout_one_episode`` returns one record per env step."""
    trainer = _make_trainer(tmp_path)
    trainer.setup()
    try:
        trainer.train()
        # Grab the live agent + a single non-vec env wrapped with CGFA.
        agent = trainer.agent
        # Wrap a fresh env so we don't touch the running vec env.
        from mtg.agents.reinforcement_learning.cgfa import CGFAEnvWrapper
        from mtg.training.env_factory import create_env

        env = CGFAEnvWrapper(
            create_env(trainer.config.env_config()),
            factor_spec=agent.factor_spec,
            scm=agent.scm,
        )
        rows, outcome = rollout_one_episode(
            agent=agent,
            env=env,
            deterministic=True,
            max_steps=64,
            episode_seed=123,
        )
        assert rows, "rollout produced no steps"
        # Every row must carry the per-factor columns and basics.
        names = agent.factor_spec.names
        first = rows[0]
        for must_have in (
            "step",
            "turn",
            "action",
            "reward",
            "done",
            "v_scalar",
            "gate",
            "blended_advantage",
            "scalar_advantage",
        ):
            assert must_have in first, f"row missing column {must_have!r}"
        for name in names:
            assert f"V_{name}" in first
            assert f"A_{name}" in first
            assert f"eps_{name}" in first
            assert f"r_{name}" in first

        # Gate must be in (0, 1).
        for r in rows:
            assert 0.0 < r["gate"] < 1.0
        # The last row must mark `done=1` if we hit a terminal.
        assert any(r["done"] == 1 for r in rows) or len(rows) == 64
        # The outcome dict must surface a usable game_result label.
        for must_have in (
            "game_result",
            "total_reward",
            "total_steps",
            "truncated",
            "max_steps_reached",
            "episode_seed",
        ):
            assert must_have in outcome, f"outcome missing key {must_have!r}"
        assert outcome["episode_seed"] == 123
        assert outcome["total_steps"] == len(rows)
        assert isinstance(outcome["truncated"], bool)
        assert isinstance(outcome["max_steps_reached"], bool)
    finally:
        trainer.close()


def test_case_study_csv_writer_round_trips(tmp_path: Path) -> None:
    """``write_case_study_csv`` round-trips records through CSV."""
    rows = [
        {
            "step": 0,
            "turn": 1,
            "action": 7,
            "action_name": "play_land_0",
            "reward": 0.1,
            "done": 0,
            "v_scalar": 0.3,
            "gate": 0.5,
            "blended_advantage": 0.2,
            "scalar_advantage": 0.15,
            "V_card_adv": 0.05,
            "A_card_adv": 0.02,
            "eps_card_adv": 0.01,
            "r_card_adv": 0.0,
        },
    ]
    csv_path = tmp_path / "case.csv"
    write_case_study_csv(rows, csv_path)
    assert csv_path.exists()
    with csv_path.open() as f:
        loaded = list(csv.DictReader(f))
    assert len(loaded) == 1
    # str values come back as strings; just verify the schema survived.
    assert loaded[0].keys() == rows[0].keys()


def test_case_study_render_produces_png(tmp_path: Path) -> None:
    """``render_case_study`` produces a valid PNG given fabricated rows."""
    factor_names = ["card_adv", "tempo", "life_buffer"]
    rng = np.random.default_rng(42)
    rows: list[dict] = []
    for t in range(5):
        row = {
            "step": t,
            "turn": t + 1,
            "action": int(rng.integers(0, 10)),
            "action_name": f"a_{t}",
            "reward": float(rng.normal()),
            "done": int(t == 4),
            "v_scalar": float(rng.normal()),
            "gate": float(rng.uniform(0.1, 0.9)),
            "blended_advantage": float(rng.normal()),
            "scalar_advantage": float(rng.normal()),
        }
        for name in factor_names:
            row[f"V_{name}"] = float(rng.normal())
            row[f"A_{name}"] = float(rng.normal())
            row[f"eps_{name}"] = float(rng.normal())
            row[f"r_{name}"] = float(rng.normal())
        rows.append(row)
    out = tmp_path / "case.png"
    render_case_study(rows, out, factor_names, title_suffix="unit-test")
    assert out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# Production training pipeline: _maybe_attach_cgfa_callback
# ---------------------------------------------------------------------------


def _stub_training_config(agent_type: str):
    """Lightweight stand-in for TrainingConfig that just exposes ``agent_type``."""

    class _C:
        pass

    c = _C()
    c.agent_type = agent_type
    return c


def test_maybe_attach_cgfa_callback_is_noop_for_non_cgfa(tmp_path: Path) -> None:
    """Non-CGFA agents must get the original callback back unchanged."""
    from scripts.runner.run_training import _maybe_attach_cgfa_callback

    sentinel = object()
    cfg = _stub_training_config("ppo")
    out = _maybe_attach_cgfa_callback(cfg, sentinel, tmp_path)
    assert out is sentinel, "non-CGFA path must not wrap the base callback"


def test_maybe_attach_cgfa_callback_wraps_for_cgfa(tmp_path: Path) -> None:
    """CGFA agents must get a CallbackList containing the calibration callback."""
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList

    from mtg.agents.reinforcement_learning.cgfa import CGFACalibrationCallback
    from scripts.runner.run_training import _maybe_attach_cgfa_callback

    class _Sentinel(BaseCallback):
        def __init__(self) -> None:
            super().__init__()

        def _on_step(self) -> bool:
            return True

    base = _Sentinel()
    cfg = _stub_training_config("cgfa")
    wrapped = _maybe_attach_cgfa_callback(cfg, base, tmp_path)
    assert isinstance(wrapped, CallbackList), "CGFA path must return a CallbackList"
    members = list(wrapped.callbacks)
    assert base in members, "original callback must still be in the chain"
    assert any(
        isinstance(m, CGFACalibrationCallback) for m in members
    ), "CGFACalibrationCallback must be appended for CGFA runs"


def test_create_agent_cgfa_returns_cgfa_agent() -> None:
    """``create_agent('cgfa', ...)`` must instantiate a CGFAAgent, not PPO.

    Regression test for the silent fallback bug where the production
    research pipeline (``mtg-research train``) was training a vanilla
    PPO agent under the ``cgfa`` label.
    """
    from mtg.agents.causal.cgfa_agent import CGFAAgent
    from mtg.agents.reinforcement_learning.cgfa import FactorSpec
    from mtg.agents.reinforcement_learning.ppo_agent import PPOAgent
    from mtg.causal.scm import StructuralCausalModel
    from mtg.training.env_factory import create_vec_env as engine_create_vec_env
    from mtg.training.train import EnvConfig
    from scripts.runner.run_training import create_agent

    factor_spec = FactorSpec()
    scm = StructuralCausalModel()
    env = engine_create_vec_env(
        EnvConfig(
            player_deck="mono_red_aggro",
            opponent_deck="mono_red_aggro",
            reward_type="shaped",
            max_turns=10,
            max_steps_per_episode=200,
            auto_combat=True,
            auto_target=True,
            seed=0,
        ),
        n_envs=1,
        cgfa_factor_spec=factor_spec,
        cgfa_scm=scm,
    )
    try:
        agent = create_agent(
            "cgfa",
            env,
            seed=0,
            total_timesteps=512,
            cgfa_factor_spec=factor_spec,
            cgfa_scm=scm,
        )
        assert isinstance(agent, CGFAAgent), (
            f"create_agent('cgfa', ...) returned {type(agent).__name__}, "
            "expected CGFAAgent (regression: silent PPO fallback)"
        )
        # CGFAAgent inherits from PPOAgent; the type check above is
        # already strict because CGFAAgent != PPOAgent (instance check
        # would pass for both but we want the concrete subclass).
        assert not isinstance(agent, PPOAgent) or type(agent) is not PPOAgent
        # The underlying SB3 model must be CGFAMaskablePPO, not MaskablePPO.
        from mtg.agents.reinforcement_learning.cgfa import CGFAMaskablePPO

        assert isinstance(agent.model, CGFAMaskablePPO), (
            f"agent.model is {type(agent.model).__name__}; expected "
            "CGFAMaskablePPO so per-factor logger keys are emitted"
        )
    finally:
        env.close()


def test_create_agent_cgfa_requires_factor_spec() -> None:
    """``create_agent('cgfa', ...)`` must raise without a FactorSpec/SCM.

    A silent fallback would re-introduce the bug where CGFA runs
    train as vanilla PPO without producing per-factor diagnostics.
    """
    from mtg.training.env_factory import create_vec_env as engine_create_vec_env
    from mtg.training.train import EnvConfig
    from scripts.runner.run_training import create_agent

    env = engine_create_vec_env(
        EnvConfig(
            player_deck="mono_red_aggro",
            opponent_deck="mono_red_aggro",
            reward_type="shaped",
            max_turns=10,
            max_steps_per_episode=200,
            auto_combat=True,
            auto_target=True,
            seed=0,
        ),
        n_envs=1,
    )
    try:
        with pytest.raises(ValueError, match="cgfa_factor_spec"):
            create_agent("cgfa", env, seed=0, total_timesteps=512)
    finally:
        env.close()


def test_create_env_wraps_with_cgfa_env_wrapper() -> None:
    """``create_env(..., cgfa_factor_spec=...)`` must yield CGFAEnvWrapper.

    Without this wrapping the per-factor signals never reach
    CGFA-PPO's rollout buffer, so calibration metrics never get
    logged (and the calibration CSV stays empty).
    """
    from mtg.agents.reinforcement_learning.cgfa import CGFAEnvWrapper, FactorSpec
    from mtg.causal.scm import StructuralCausalModel
    from scripts.runner.run_training import create_env

    factor_spec = FactorSpec()
    scm = StructuralCausalModel()
    env = create_env(
        player_deck="mono_red_aggro",
        opponent_deck="mono_red_aggro",
        reward_type="shaped",
        seed=0,
        max_turns=10,
        max_steps_per_episode=200,
        auto_combat=True,
        auto_target=True,
        cgfa_factor_spec=factor_spec,
        cgfa_scm=scm,
    )
    # Walk the wrapper chain looking for a CGFAEnvWrapper.
    cur = env
    found = False
    for _ in range(10):
        if isinstance(cur, CGFAEnvWrapper):
            found = True
            break
        cur = getattr(cur, "env", None)
        if cur is None:
            break
    assert found, "create_env(cgfa_factor_spec=...) did not insert CGFAEnvWrapper"

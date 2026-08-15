"""Regression test: ``evaluate_sweep`` must evaluate resumed (skipped) runs.

When a paper run is resumed, ``train_sweep`` records already-trained models with
``status: "skipped"`` (the model is on disk but no new training was performed).
A previous version of ``eval_sweep.evaluate_sweep`` filtered with
``status == "completed"`` only, silently dropping every resumed model and
producing eval_results.json files with ``"trained": []`` for those variants.

This test pins the contract: any run whose model file exists on disk must be
evaluated, regardless of whether the manifest marks it ``"completed"`` (freshly
trained) or ``"skipped"`` (resumed from a previous invocation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.research import eval_sweep


def _write_manifest(experiment_dir: Path, runs: list[dict[str, Any]]) -> None:
    """Write a minimal sweep_manifest.yaml the evaluator can consume."""
    manifest = {
        "experiment_name": experiment_dir.name,
        "created": "2026-04-24T00:00:00",
        "config": {
            "seeds": [42, 123, 456],
            "opponents": ["mono_red_aggro", "azorius_control"],
        },
        "runs": runs,
    }
    (experiment_dir / "sweep_manifest.yaml").write_text(yaml.safe_dump(manifest))


def _make_run(
    *,
    agent: str,
    deck: str,
    seed: int,
    status: str,
    output_dir: str,
) -> dict[str, Any]:
    return {
        "agent": agent,
        "player_deck": deck,
        "seed": seed,
        "opponents": ["mono_red_aggro", "azorius_control"],
        "output_dir": output_dir,
        "status": status,
        "error": None,
        "training_time_seconds": None,
        "timesteps_per_opponent": 1000,
        "total_timesteps": 2000,
        "agent_kwargs": {},
    }


def _stage_model(experiment_dir: Path, agent: str, deck: str, output_dir: str) -> None:
    """Create the run dir and a placeholder model zip _model_path() will find."""
    rdir = experiment_dir / output_dir
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / f"{agent}_{deck}.zip").write_bytes(b"placeholder")


def _patched_evaluate_run(
    monkeypatch: pytest.MonkeyPatch, opponents: list[str]
) -> list[tuple[str, str, int, Path | None]]:
    """Patch _evaluate_run to record dispatch tuples and skip real episode work."""
    calls: list[tuple[str, str, int, Path | None]] = []

    def _fake_run(
        *,
        agent_type: str,
        player_deck: str,
        seed: int,
        model_path: Path | None,
        opponents: list[str],
        n_episodes: int,
        max_turns: int,
        auto_combat: bool,
        auto_target: bool,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        calls.append((agent_type, player_deck, seed, model_path))
        per_opp = {
            opp: {
                "n": n_episodes,
                "win_rate": 0.5,
                "win_rate_ci_lo": 0.4,
                "win_rate_ci_hi": 0.6,
                "draw_rate": 0.0,
                "avg_reward": 0.0,
                "reward_std": 1.0,
                "reward_ci_lo": -0.1,
                "reward_ci_hi": 0.1,
                "avg_length": 10.0,
                "length_std": 2.0,
            }
            for opp in opponents
        }
        return per_opp, []

    monkeypatch.setattr(eval_sweep, "_evaluate_run", _fake_run)
    return calls


def test_evaluate_sweep_includes_resumed_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All three skipped (resumed) runs must be evaluated when models exist.

    This is the exact failure mode that nuked three ablation variants in the
    paper_lite_20260420_2312 run: every run had ``status: "skipped"`` (the
    models were on disk from a pre-resume training invocation), so the old
    filter dropped them and produced ``trained: []``.
    """
    experiment_dir = tmp_path / "ppo"
    experiment_dir.mkdir()

    runs = []
    for seed in (42, 123, 456):
        odir = f"ppo_mono_red_aggro_vs_multi_seed{seed}"
        runs.append(
            _make_run(
                agent="ppo",
                deck="mono_red_aggro",
                seed=seed,
                status="skipped",
                output_dir=odir,
            )
        )
        _stage_model(experiment_dir, "ppo", "mono_red_aggro", odir)

    _write_manifest(experiment_dir, runs)
    calls = _patched_evaluate_run(monkeypatch, opponents=["mono_red_aggro", "azorius_control"])

    eval_sweep.evaluate_sweep(
        experiment_dir,
        n_episodes=10,
        include_baselines=False,
    )

    trained_calls = [c for c in calls if c[3] is not None]
    assert len(trained_calls) == 3, (
        f"All 3 skipped runs should have been evaluated; got {len(trained_calls)} "
        f"trained dispatches: {trained_calls}"
    )
    assert {c[2] for c in trained_calls} == {42, 123, 456}


def test_evaluate_sweep_mixes_completed_and_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mixed manifests (some completed, some skipped) must dispatch both kinds."""
    experiment_dir = tmp_path / "cgfa_no_gate"
    experiment_dir.mkdir()

    runs = [
        _make_run(
            agent="cgfa",
            deck="mono_red_aggro",
            seed=42,
            status="skipped",
            output_dir="cgfa_seed42",
        ),
        _make_run(
            agent="cgfa",
            deck="mono_red_aggro",
            seed=123,
            status="completed",
            output_dir="cgfa_seed123",
        ),
        _make_run(
            agent="cgfa",
            deck="mono_red_aggro",
            seed=456,
            status="completed",
            output_dir="cgfa_seed456",
        ),
    ]
    for r in runs:
        _stage_model(experiment_dir, "cgfa", "mono_red_aggro", r["output_dir"])

    _write_manifest(experiment_dir, runs)
    calls = _patched_evaluate_run(monkeypatch, opponents=["mono_red_aggro", "azorius_control"])

    eval_sweep.evaluate_sweep(
        experiment_dir,
        n_episodes=10,
        include_baselines=False,
    )

    trained_seeds = sorted(c[2] for c in calls if c[3] is not None)
    assert trained_seeds == [42, 123, 456], (
        f"Expected all three seeds to be evaluated regardless of status, " f"got {trained_seeds}"
    )


def test_evaluate_sweep_skips_runs_without_model_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run whose model file is missing must be silently skipped (not crash)."""
    experiment_dir = tmp_path / "broken"
    experiment_dir.mkdir()

    runs = [
        _make_run(
            agent="ppo",
            deck="mono_red_aggro",
            seed=42,
            status="completed",
            output_dir="present",
        ),
        _make_run(
            agent="ppo",
            deck="mono_red_aggro",
            seed=123,
            status="skipped",
            output_dir="missing_model",
        ),
    ]
    _stage_model(experiment_dir, "ppo", "mono_red_aggro", "present")
    (experiment_dir / "missing_model").mkdir()

    _write_manifest(experiment_dir, runs)
    calls = _patched_evaluate_run(monkeypatch, opponents=["mono_red_aggro", "azorius_control"])

    eval_sweep.evaluate_sweep(
        experiment_dir,
        n_episodes=10,
        include_baselines=False,
    )

    trained_seeds = sorted(c[2] for c in calls if c[3] is not None)
    assert trained_seeds == [42], (
        "Only the run with a model file on disk should have been evaluated; "
        f"got seeds {trained_seeds}"
    )

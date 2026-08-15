"""Tests for the three "regenerate-from-disk" entry points.

Each of the project's three figure-producing workflows (training,
standalone evaluation, research) can now be re-rendered from its
saved JSON / CSV without re-running the underlying training or
evaluation pipeline.  These tests guarantee that contract holds:

* ``regenerate_plots --from-results-json`` (eval run) re-renders
  ``win_rate_comparison.png`` + ``reward_comparison.png`` from
  ``results.json``.
* ``case_study --from-csv`` re-renders ``case_study.png`` from
  ``case_study_steps.csv`` without loading a CGFA model.
* ``transfer_sweep --from-report`` re-renders ``transfer_gap.png``
  from ``transfer_report.json`` without re-training.

If any of these regress, refactoring the figures would again
require burning a full training run, which is exactly what we
want to prevent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# matplotlib / numpy are required for the renderers.
pytest.importorskip("matplotlib")
np = pytest.importorskip("numpy")

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# 1. Eval run replot: regenerate_plots.regenerate_evaluation_run_plots
# ---------------------------------------------------------------------------


def _write_eval_results_json(path: Path) -> None:
    """Write a minimal ``results.json`` mimicking ``run_evaluation.py``."""
    payload = {
        "config": {
            "player_deck": "mono_red_aggro",
            "opponent_decks": ["azorius_control", "dimir_midrange"],
            "episodes": 100,
            "episodes_per_opponent": 100,
            "episodes_per_seed": 25,
            "max_turns": 10,
            "seeds": [1, 2, 3, 4],
            "model_path": None,
            "evaluation_time_seconds": 12.3,
            "timestamp": "2026-01-01T00:00:00",
        },
        "results": {
            "ppo_vs_azorius_control": {
                "agent": "ppo",
                "opponent": "azorius_control",
                "win_rate": 0.65,
                "win_rate_std": 0.05,
                "avg_reward": 0.4,
                "reward_std": 0.1,
                "per_seed": [],
            },
            "ppo_vs_dimir_midrange": {
                "agent": "ppo",
                "opponent": "dimir_midrange",
                "win_rate": 0.45,
                "win_rate_std": 0.06,
                "avg_reward": -0.1,
                "reward_std": 0.15,
                "per_seed": [],
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def test_regenerate_evaluation_run_plots_writes_pngs(tmp_path: Path) -> None:
    """``results.json`` -> the two eval bar charts on disk, no eval re-run."""
    from scripts.runner.regenerate_plots import regenerate_evaluation_run_plots

    run_dir = tmp_path / "eval_run"
    _write_eval_results_json(run_dir / "results.json")

    plots_dir = run_dir / "plots"
    paths = regenerate_evaluation_run_plots(run_dir, plots_dir)

    assert len(paths) == 2, f"expected 2 plots, got {len(paths)}: {paths}"
    names = {Path(p).name for p in paths}
    assert names == {"win_rate_comparison.png", "reward_comparison.png"}
    for p in paths:
        assert Path(p).exists()
        assert Path(p).read_bytes()[:8] == PNG_MAGIC


def test_regenerate_evaluation_run_plots_skips_when_no_results(tmp_path: Path) -> None:
    """Missing ``results.json`` returns empty list, does not raise."""
    from scripts.runner.regenerate_plots import regenerate_evaluation_run_plots

    paths = regenerate_evaluation_run_plots(tmp_path, tmp_path / "plots")
    assert paths == []


def test_regenerate_plots_main_handles_eval_run(tmp_path: Path) -> None:
    """The CLI ``main`` auto-detects ``results.json`` and renders eval plots."""
    from scripts.runner.regenerate_plots import main

    run_dir = tmp_path / "eval_run"
    _write_eval_results_json(run_dir / "results.json")

    rc = main_wrapper(main, ["regenerate_plots", str(run_dir)])
    assert rc == 0
    assert (run_dir / "plots" / "win_rate_comparison.png").exists()
    assert (run_dir / "plots" / "reward_comparison.png").exists()


def main_wrapper(main_fn, argv: list[str]) -> int:
    """Run an ``argparse``-based ``main`` with a synthetic ``sys.argv``."""
    import sys

    saved = sys.argv
    try:
        sys.argv = argv
        return main_fn()
    finally:
        sys.argv = saved


# ---------------------------------------------------------------------------
# 2. Case-study replot: case_study.read_case_study_csv + render_case_study
# ---------------------------------------------------------------------------


def _synthetic_case_study_rows(
    factor_names: tuple[str, ...] = ("life", "mana", "tempo"),
    n_steps: int = 6,
) -> list[dict]:
    """Build deterministic per-step records compatible with render_case_study."""
    rng = np.random.default_rng(0)
    rows: list[dict] = []
    for t in range(n_steps):
        row: dict = {
            "step": t,
            "turn": t + 1,
            "action": int(rng.integers(0, 50)),
            "action_name": f"play_card_{t}",
            "reward": float(rng.normal(0, 0.5)),
            "done": int(t == n_steps - 1),
            "v_scalar": float(rng.normal(0, 1)),
            "gate": float(rng.uniform(0, 1)),
            "blended_advantage": float(rng.normal(0, 0.5)),
            "scalar_advantage": float(rng.normal(0, 0.5)),
        }
        for name in factor_names:
            row[f"V_{name}"] = float(rng.normal(0, 1))
            row[f"A_{name}"] = float(rng.normal(0, 0.5))
            row[f"eps_{name}"] = float(rng.normal(0, 0.3))
            row[f"r_{name}"] = float(rng.normal(0, 0.2))
        rows.append(row)
    return rows


def test_case_study_csv_round_trip_recovers_factor_names(tmp_path: Path) -> None:
    """Write -> read recovers ``factor_names`` from ``V_*`` columns."""
    from scripts.research.case_study import read_case_study_csv, write_case_study_csv

    rows = _synthetic_case_study_rows(factor_names=("alpha", "beta"))
    csv_path = tmp_path / "case_study_steps.csv"
    write_case_study_csv(rows, csv_path)

    loaded, factor_names = read_case_study_csv(csv_path)
    assert factor_names == ["alpha", "beta"]
    assert len(loaded) == len(rows)
    # Numeric columns must be cast back to numbers (not strings).
    for r in loaded:
        assert isinstance(r["turn"], int)
        assert isinstance(r["V_alpha"], float)
        assert isinstance(r["A_beta"], float)


def test_case_study_render_from_csv_writes_png(tmp_path: Path) -> None:
    """End-to-end: write CSV -> read CSV -> render PNG (no model, no env)."""
    from scripts.research.case_study import (
        read_case_study_csv,
        render_case_study,
        write_case_study_csv,
    )

    rows = _synthetic_case_study_rows()
    csv_path = tmp_path / "case_study_steps.csv"
    write_case_study_csv(rows, csv_path)

    loaded, factor_names = read_case_study_csv(csv_path)
    out = tmp_path / "case_study.png"
    render_case_study(loaded, out, factor_names, title_suffix="from-csv smoke")

    assert out.exists()
    assert out.read_bytes()[:8] == PNG_MAGIC


# ---------------------------------------------------------------------------
# 3. Transfer replot: transfer_sweep.replot_transfer_from_report
# ---------------------------------------------------------------------------


def _synthetic_transfer_report() -> dict:
    """Minimal ``transfer_report.json`` payload for two agents."""
    return {
        "per_agent": {
            "ppo": {
                "in_dist_mean": 0.85,
                "heldout_mean": 0.45,
                "gap_mean": 0.40,
                "gap_ci_lo": 0.30,
                "gap_ci_hi": 0.50,
                "p_value": 0.01,
                "n_pairs": 4,
            },
            "cgfa": {
                "in_dist_mean": 0.70,
                "heldout_mean": 0.65,
                "gap_mean": 0.05,
                "gap_ci_lo": -0.05,
                "gap_ci_hi": 0.15,
                "p_value": 0.40,
                "n_pairs": 4,
            },
        },
        "long": [],
        "per_opponent_heldout": {},
    }


def test_replot_transfer_from_report_writes_png(tmp_path: Path) -> None:
    """JSON -> PNG via the public helper, no training, no eval."""
    from scripts.research.transfer_sweep import replot_transfer_from_report

    report_path = tmp_path / "transfer" / "transfer_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(_synthetic_transfer_report()))

    out = replot_transfer_from_report(report_path)

    assert out.exists()
    assert out.name == "transfer_gap.png"
    assert out.parent.name == "figures"
    assert out.read_bytes()[:8] == PNG_MAGIC


def test_replot_transfer_from_report_respects_explicit_output(tmp_path: Path) -> None:
    """``out_path`` override puts the PNG exactly where the caller asks."""
    from scripts.research.transfer_sweep import replot_transfer_from_report

    report_path = tmp_path / "transfer_report.json"
    report_path.write_text(json.dumps(_synthetic_transfer_report()))

    explicit = tmp_path / "elsewhere" / "custom.png"
    out = replot_transfer_from_report(report_path, explicit)

    assert out == explicit
    assert explicit.exists()
    assert explicit.read_bytes()[:8] == PNG_MAGIC


def test_replot_transfer_from_report_raises_on_missing_path(tmp_path: Path) -> None:
    """Clear error when the report file is missing (not a silent failure)."""
    from scripts.research.transfer_sweep import replot_transfer_from_report

    with pytest.raises(FileNotFoundError, match="transfer_report not found"):
        replot_transfer_from_report(tmp_path / "does_not_exist.json")

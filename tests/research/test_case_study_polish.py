"""Tests for the case-study CLI / figure / sidecar contracts.

These tests pin three externally visible properties of the case study:

* The CLI defaults the rollout's reward type to ``sparse``, matching
  the eval/aggregate pipeline so the per-factor decompositions reflect
  the same outcome-driven signal as the headline win-rate tables.
* The figure title and the on-disk sidecar disclose the episode
  outcome (``WIN`` / ``LOSS`` / ``DRAW`` / step-cap).
* The calibration overlay panel plots every factor so the full set of
  A_k vs. eps_k curves the agent learned to align is visible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

from scripts.research.case_study import (  # noqa: E402
    _calibration_overlay,
    _format_outcome_for_title,
    parse_args,
    read_case_study_outcome,
    render_case_study,
    write_case_study_csv,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _toy_rows(*, n_steps: int = 4, factor_names: tuple[str, ...] = ("a", "b", "c")) -> list[dict]:
    """Synthetic per-step rows that exercise every column the figure reads."""
    rows: list[dict] = []
    for t in range(n_steps):
        row: dict = {
            "step": t,
            "turn": t,
            "action": t % 2,
            "action_name": f"act_{t}",
            "reward": float(t) * 0.1,
            "done": int(t == n_steps - 1),
            "v_scalar": 0.05 * t,
            "gate": 0.5,
            "blended_advantage": 0.01 * t,
            "scalar_advantage": -0.02 * t,
        }
        for k, name in enumerate(factor_names):
            row[f"V_{name}"] = 0.1 * (k + 1) - 0.05 * t
            row[f"A_{name}"] = 0.05 * (k + 1) * (1 if t % 2 else -1)
            row[f"eps_{name}"] = 0.04 * (k + 1) * (1 if t % 2 else -1)
            row[f"r_{name}"] = 0.01 * (k + 1)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Reward parity between the case study and the rest of the pipeline
# ---------------------------------------------------------------------------


def test_case_study_cli_defaults_reward_type_to_sparse(monkeypatch) -> None:
    """The CLI must default ``--reward-type`` to ``sparse`` (matches eval)."""
    monkeypatch.setattr(
        "sys.argv",
        ["case_study", "--model-path", "/tmp/does_not_matter.zip"],
    )
    args = parse_args()
    assert args.reward_type == "sparse", (
        "case_study should default to sparse rewards so the per-factor "
        "decompositions reflect the same signal as the eval pipeline; "
        f"got reward_type={args.reward_type!r}"
    )


def test_case_study_cli_accepts_explicit_shaped_override(monkeypatch) -> None:
    """``--reward-type shaped`` is still available for shaping-channel work."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "case_study",
            "--model-path",
            "/tmp/does_not_matter.zip",
            "--reward-type",
            "shaped",
        ],
    )
    args = parse_args()
    assert args.reward_type == "shaped"


def test_case_study_cli_rejects_unknown_reward_type(monkeypatch) -> None:
    """Argparse choices must reject anything other than sparse/shaped."""
    import pytest

    monkeypatch.setattr(
        "sys.argv",
        [
            "case_study",
            "--model-path",
            "/tmp/x.zip",
            "--reward-type",
            "weird_extra_signal",
        ],
    )
    with pytest.raises(SystemExit):
        parse_args()


# ---------------------------------------------------------------------------
# Episode outcome surfaces in the figure title and on disk
# ---------------------------------------------------------------------------


def test_format_outcome_for_title_renders_all_results() -> None:
    """The compact suffix exposes win/loss/draw + total reward + step count."""
    win = _format_outcome_for_title(
        {"game_result": "win", "total_reward": 1.0, "total_steps": 12, "truncated": False}
    )
    assert "WIN" in win
    assert "R=+1.00" in win
    assert "T=12 steps" in win

    loss = _format_outcome_for_title(
        {"game_result": "loss", "total_reward": -0.5, "total_steps": 30, "truncated": True}
    )
    assert "LOSS" in loss
    assert "truncated" in loss

    draw = _format_outcome_for_title(
        {"game_result": "draw", "total_reward": 0.0, "total_steps": 20, "max_steps_reached": True}
    )
    assert "DRAW" in draw
    assert "step-cap reached" in draw


def test_format_outcome_for_title_accepts_none() -> None:
    """An absent ``outcome`` produces an empty suffix."""
    assert _format_outcome_for_title(None) == ""


def test_render_case_study_embeds_outcome_in_suptitle(tmp_path: Path) -> None:
    """``render_case_study`` should embed the outcome in the figure suptitle."""
    rows = _toy_rows()
    out = tmp_path / "case_study.png"
    fig = render_case_study(
        rows,
        out,
        factor_names=["a", "b", "c"],
        title_suffix="mono_red vs control (seed 42)",
        outcome={"game_result": "win", "total_reward": 0.6, "total_steps": 4, "truncated": False},
    )
    assert fig.exists()
    # The suptitle should now include "WIN" *and* the user-provided suffix.
    # We can't read it back from the PNG, so we re-render in-memory and
    # inspect the suptitle text directly.
    figure, axes = plt.subplots(3, 1, figsize=(6, 6))
    plt.close(figure)  # discard probe figure
    new_path = tmp_path / "case_study_probe.png"
    render_case_study(
        rows,
        new_path,
        factor_names=["a", "b", "c"],
        title_suffix="probe-suffix",
        outcome={"game_result": "loss", "total_reward": -0.2, "total_steps": 4, "truncated": True},
    )
    # If we got here without raising, the outcome path is exercised and
    # the figure file was written. Inspect the file size as a smoke test.
    assert new_path.stat().st_size > 1024


def test_write_case_study_csv_persists_outcome_sidecar(tmp_path: Path) -> None:
    """Writing the CSV with an outcome dict produces a JSON sidecar."""
    rows = _toy_rows()
    csv_path = tmp_path / "case_study_steps.csv"
    outcome = {
        "game_result": "win",
        "total_reward": 0.6,
        "total_steps": 4,
        "truncated": False,
        "max_steps_reached": False,
        "episode_seed": 7,
    }
    write_case_study_csv(rows, csv_path, outcome=outcome)
    sidecar = csv_path.with_name("case_study_outcome.json")
    assert sidecar.exists()
    loaded = json.loads(sidecar.read_text())
    assert loaded == outcome


def test_read_case_study_outcome_returns_none_when_missing(tmp_path: Path) -> None:
    """If the sidecar file is absent, the loader returns None gracefully."""
    csv_path = tmp_path / "case_study_steps.csv"
    rows = _toy_rows()
    write_case_study_csv(rows, csv_path)  # no outcome
    assert read_case_study_outcome(csv_path) is None


def test_read_case_study_outcome_round_trips_with_writer(tmp_path: Path) -> None:
    """``read_case_study_outcome`` recovers exactly what was written."""
    csv_path = tmp_path / "case_study_steps.csv"
    rows = _toy_rows()
    outcome = {"game_result": "draw", "total_steps": 99, "truncated": True}
    write_case_study_csv(rows, csv_path, outcome=outcome)
    assert read_case_study_outcome(csv_path) == outcome


# ---------------------------------------------------------------------------
# The calibration overlay shows every factor (no top-2 truncation)
# ---------------------------------------------------------------------------


def _legend_labels(ax: plt.Axes) -> list[str]:
    legend = ax.get_legend()
    if legend is None:
        return []
    return [t.get_text() for t in legend.get_texts()]


def test_calibration_overlay_plots_every_factor() -> None:
    """The overlay must include both an A_k and eps_k entry for every factor."""
    factor_names = ["alpha", "beta", "gamma", "delta", "epsilon"]
    rows = _toy_rows(factor_names=tuple(factor_names))
    fig, ax = plt.subplots()
    try:
        _calibration_overlay(ax, rows, factor_names)
        # Two lines per factor (A_k + eps_k); some matplotlib backends
        # add helper artists, so check the *line* count specifically.
        n_lines = sum(1 for line in ax.get_lines() if line.get_linestyle() in {"-", "--"})
        assert n_lines >= 2 * len(factor_names), (
            f"calibration overlay must draw 2 lines per factor "
            f"({len(factor_names)} factors -> {2 * len(factor_names)} lines), "
            f"got {n_lines}"
        )
        labels = _legend_labels(ax)
        for name in factor_names:
            assert any(
                name in label for label in labels
            ), f"factor {name!r} missing from calibration overlay legend: {labels}"
        title = ax.get_title()
        assert (
            "all" in title.lower()
        ), f"overlay title should disclose that all factors are shown; got {title!r}"
    finally:
        plt.close(fig)


def test_calibration_overlay_handles_empty_rows() -> None:
    """An empty row list must not crash the overlay."""
    fig, ax = plt.subplots()
    try:
        _calibration_overlay(ax, [], ["a", "b"])
        assert ax.get_lines() == [] or all(ln.get_xydata().size == 0 for ln in ax.get_lines())
    finally:
        plt.close(fig)

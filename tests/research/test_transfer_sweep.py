"""Tests for the transfer experiment runner.

These tests exercise :mod:`scripts.research.transfer_sweep` with synthetic
``eval_results.json`` files so the heavy training pipeline is not required.
We cover:

* ``TransferConfig.validate`` enforces non-empty inputs and disjoint
  train/held-out opponent sets.
* ``build_transfer_report`` correctly indexes both eval files, computes
  per-agent generalisation gap, paired-bootstrap CI, and a per-opponent
  held-out summary.
* ``write_transfer_artifacts`` round-trips the report into JSON, the
  long-form CSV, and the per-opponent CSV with the expected columns.
* ``render_transfer_figure`` produces a non-empty PNG.
* The CLI ``--smoke`` config is well-formed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.research.transfer_sweep import (
    TransferConfig,
    _smoke_config,
    build_transfer_report,
    render_transfer_figure,
    write_transfer_artifacts,
)

# ---------------------------------------------------------------------------
# Helpers: build deterministic synthetic eval_results.json payloads
# ---------------------------------------------------------------------------


def _trained_entry(
    agent: str,
    deck: str,
    seed: int,
    per_opponent: dict[str, float],
) -> dict[str, object]:
    """Build a ``trained``-block entry with deterministic per-opponent stats.

    ``per_opponent`` maps opponent -> win-rate; we synthesise plausible
    auxiliary statistics around it.
    """
    per_opp_payload: dict[str, dict[str, float]] = {}
    for opp, wr in per_opponent.items():
        per_opp_payload[opp] = {
            "n": 100,
            "win_rate": wr,
            "win_rate_ci_lo": max(0.0, wr - 0.05),
            "win_rate_ci_hi": min(1.0, wr + 0.05),
            "draw_rate": 0.0,
            "avg_reward": wr - 0.5,
            "reward_std": 0.5,
            "reward_ci_lo": wr - 0.6,
            "reward_ci_hi": wr - 0.4,
            "avg_length": 8.0,
            "length_std": 2.0,
        }
    return {
        "agent": agent,
        "player_deck": deck,
        "seed": seed,
        "per_opponent": per_opp_payload,
    }


def _eval_payload(entries: list[dict[str, object]]) -> dict[str, object]:
    return {"trained": entries, "baselines": []}


def _write_eval_file(path: Path, entries: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_eval_payload(entries)))


# ---------------------------------------------------------------------------
# TransferConfig.validate
# ---------------------------------------------------------------------------


def _make_cfg(
    *, train_opponents: list[str], heldout_opponents: list[str], **overrides
) -> TransferConfig:
    base = {
        "experiment_name": "exp",
        "agents": ["ppo"],
        "player_decks": ["mono_red_aggro"],
        "seeds": [42],
        "train_opponents": train_opponents,
        "heldout_opponents": heldout_opponents,
        "timesteps_per_opponent": 1000,
    }
    base.update(overrides)
    return TransferConfig(**base)  # type: ignore[arg-type]


def test_validate_accepts_disjoint_sets() -> None:
    """Disjoint train and held-out opponent sets validate successfully."""
    cfg = _make_cfg(
        train_opponents=["mono_red_aggro", "azorius_control"],
        heldout_opponents=["dimir_midrange"],
    )
    cfg.validate()  # must not raise


def test_validate_rejects_overlap() -> None:
    """Overlap between train and held-out raises a clear ValueError."""
    cfg = _make_cfg(
        train_opponents=["mono_red_aggro", "azorius_control"],
        heldout_opponents=["azorius_control"],
    )
    with pytest.raises(ValueError, match="overlap"):
        cfg.validate()


def test_validate_rejects_empty_sets() -> None:
    """Empty agents/seeds/decks/opponents raise ValueError."""
    cfg = _make_cfg(
        train_opponents=["mono_red_aggro"],
        heldout_opponents=["azorius_control"],
        agents=[],
    )
    with pytest.raises(ValueError, match="`agents` must be non-empty"):
        cfg.validate()


# ---------------------------------------------------------------------------
# build_transfer_report: numerical correctness on synthetic data
# ---------------------------------------------------------------------------


def test_report_computes_positive_gap_when_in_dist_higher(tmp_path: Path) -> None:
    """An agent that wins more in-distribution than held-out has a positive gap."""
    cfg = _make_cfg(
        train_opponents=["mono_red_aggro", "azorius_control"],
        heldout_opponents=["dimir_midrange"],
        agents=["ppo"],
        player_decks=["mono_red_aggro"],
        seeds=[1, 2, 3],
    )
    in_dist_entries = [
        _trained_entry("ppo", "mono_red_aggro", s, {"mono_red_aggro": 0.9, "azorius_control": 0.8})
        for s in (1, 2, 3)
    ]
    heldout_entries = [
        _trained_entry("ppo", "mono_red_aggro", s, {"dimir_midrange": 0.4}) for s in (1, 2, 3)
    ]
    in_dist_path = tmp_path / "eval" / "eval_results.json"
    heldout_path = tmp_path / "eval_heldout" / "eval_results.json"
    _write_eval_file(in_dist_path, in_dist_entries)
    _write_eval_file(heldout_path, heldout_entries)

    report = build_transfer_report(cfg, in_dist_path, heldout_path)
    assert "ppo" in report["per_agent"]
    p = report["per_agent"]["ppo"]
    assert p["n_pairs"] == 3
    assert p["in_dist_mean"] == pytest.approx(0.85, abs=1e-9)
    assert p["heldout_mean"] == pytest.approx(0.40, abs=1e-9)
    assert p["gap_mean"] == pytest.approx(0.45, abs=1e-9)
    assert p["gap_ci_lo"] <= p["gap_mean"] <= p["gap_ci_hi"]


def test_report_distinguishes_two_agents(tmp_path: Path) -> None:
    """Per-agent stats are computed independently for each agent in the eval file.

    PPO has a big in-distribution lead but collapses on the held-out
    opponent; CGFA has a smaller in-distribution lead but transfers
    almost perfectly.  The per-agent gap should reflect that ordering.
    """
    cfg = _make_cfg(
        train_opponents=["mono_red_aggro", "azorius_control"],
        heldout_opponents=["dimir_midrange"],
        agents=["ppo", "cgfa"],
        player_decks=["mono_red_aggro"],
        seeds=[1, 2, 3, 4],
    )
    in_entries: list[dict[str, object]] = []
    out_entries: list[dict[str, object]] = []
    for s in (1, 2, 3, 4):
        in_entries.append(
            _trained_entry(
                "ppo",
                "mono_red_aggro",
                s,
                {"mono_red_aggro": 0.9, "azorius_control": 0.8},
            )
        )
        in_entries.append(
            _trained_entry(
                "cgfa",
                "mono_red_aggro",
                s,
                {"mono_red_aggro": 0.7, "azorius_control": 0.7},
            )
        )
        out_entries.append(_trained_entry("ppo", "mono_red_aggro", s, {"dimir_midrange": 0.30}))
        out_entries.append(_trained_entry("cgfa", "mono_red_aggro", s, {"dimir_midrange": 0.65}))

    in_path = tmp_path / "eval" / "eval_results.json"
    out_path = tmp_path / "eval_heldout" / "eval_results.json"
    _write_eval_file(in_path, in_entries)
    _write_eval_file(out_path, out_entries)

    report = build_transfer_report(cfg, in_path, out_path)
    ppo = report["per_agent"]["ppo"]
    cgfa = report["per_agent"]["cgfa"]
    assert ppo["gap_mean"] > cgfa["gap_mean"]
    assert ppo["heldout_mean"] < cgfa["heldout_mean"]


def test_report_handles_empty_overlap(tmp_path: Path) -> None:
    """When in-dist and held-out have no shared (deck, seed), gap is NaN."""
    cfg = _make_cfg(
        train_opponents=["mono_red_aggro"],
        heldout_opponents=["azorius_control"],
        seeds=[1],
    )
    in_path = tmp_path / "eval" / "eval_results.json"
    out_path = tmp_path / "eval_heldout" / "eval_results.json"
    _write_eval_file(in_path, [_trained_entry("ppo", "mono_red_aggro", 1, {"mono_red_aggro": 0.9})])
    # Different seed -> no pair available.
    _write_eval_file(
        out_path, [_trained_entry("ppo", "mono_red_aggro", 2, {"azorius_control": 0.4})]
    )

    report = build_transfer_report(cfg, in_path, out_path)
    p = report["per_agent"]["ppo"]
    assert p["n_pairs"] == 0
    import math

    assert math.isnan(p["gap_mean"])


def test_report_per_opponent_heldout_summary(tmp_path: Path) -> None:
    """Per-opponent held-out summary aggregates across (deck, seed) per opponent."""
    cfg = _make_cfg(
        train_opponents=["mono_red_aggro"],
        heldout_opponents=["dimir_midrange", "domain_ramp"],
        seeds=[1, 2],
    )
    in_path = tmp_path / "eval" / "eval_results.json"
    out_path = tmp_path / "eval_heldout" / "eval_results.json"
    _write_eval_file(
        in_path,
        [_trained_entry("ppo", "mono_red_aggro", s, {"mono_red_aggro": 0.6}) for s in (1, 2)],
    )
    _write_eval_file(
        out_path,
        [
            _trained_entry("ppo", "mono_red_aggro", s, {"dimir_midrange": 0.3, "domain_ramp": 0.5})
            for s in (1, 2)
        ],
    )

    report = build_transfer_report(cfg, in_path, out_path)
    by_opp = report["per_opponent_heldout"]
    assert set(by_opp.keys()) == {"dimir_midrange", "domain_ramp"}
    assert by_opp["dimir_midrange"]["ppo"]["mean"] == pytest.approx(0.30, abs=1e-9)
    assert by_opp["domain_ramp"]["ppo"]["mean"] == pytest.approx(0.50, abs=1e-9)
    assert by_opp["dimir_midrange"]["ppo"]["n"] == 2


# ---------------------------------------------------------------------------
# write_transfer_artifacts: file shape + round-trip
# ---------------------------------------------------------------------------


def _toy_report(cfg: TransferConfig, tmp_path: Path) -> dict[str, object]:
    in_path = tmp_path / "eval" / "eval_results.json"
    out_path = tmp_path / "eval_heldout" / "eval_results.json"
    _write_eval_file(
        in_path,
        [_trained_entry("ppo", "mono_red_aggro", s, {"mono_red_aggro": 0.7}) for s in (1, 2)]
        + [_trained_entry("cgfa", "mono_red_aggro", s, {"mono_red_aggro": 0.6}) for s in (1, 2)],
    )
    _write_eval_file(
        out_path,
        [_trained_entry("ppo", "mono_red_aggro", s, {"azorius_control": 0.4}) for s in (1, 2)]
        + [_trained_entry("cgfa", "mono_red_aggro", s, {"azorius_control": 0.55}) for s in (1, 2)],
    )
    return build_transfer_report(cfg, in_path, out_path)


def test_write_transfer_artifacts_emits_expected_files(tmp_path: Path) -> None:
    """The runner writes JSON, two CSVs, and a PNG with the expected layout."""
    cfg = _make_cfg(
        train_opponents=["mono_red_aggro"],
        heldout_opponents=["azorius_control"],
        agents=["ppo", "cgfa"],
        seeds=[1, 2],
        output_root=str(tmp_path),
        experiment_name="exp_artifacts",
    )
    report = _toy_report(cfg, tmp_path)
    artifacts = write_transfer_artifacts(cfg, report)

    assert set(artifacts.keys()) == {"json", "summary_csv", "per_opponent_csv", "figure"}
    for path in artifacts.values():
        assert Path(path).exists(), f"{path} was not written"

    with open(artifacts["json"]) as f:
        loaded = json.load(f)
    assert loaded["per_agent"].keys() == report["per_agent"].keys()

    with open(artifacts["summary_csv"]) as f:
        rows = list(csv.DictReader(f))
    assert {r["agent"] for r in rows} == {"ppo", "cgfa"}
    assert {r["split"] for r in rows} == {"in_dist", "heldout"}

    with open(artifacts["per_opponent_csv"]) as f:
        rows = list(csv.DictReader(f))
    assert any(r["opponent"] == "azorius_control" for r in rows)
    assert all("mean_win_rate" in r for r in rows)


def test_render_transfer_figure_creates_non_empty_png(tmp_path: Path) -> None:
    """The figure renderer writes a non-empty PNG even with two agents."""
    cfg = _make_cfg(
        train_opponents=["mono_red_aggro"],
        heldout_opponents=["azorius_control"],
        agents=["ppo", "cgfa"],
        seeds=[1, 2],
        output_root=str(tmp_path),
        experiment_name="exp_fig",
    )
    report = _toy_report(cfg, tmp_path)
    out_path = tmp_path / "fig.png"
    render_transfer_figure(report, out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 1000  # arbitrary non-empty threshold


def test_render_transfer_figure_handles_no_pairs(tmp_path: Path) -> None:
    """The figure still renders when there are no per-agent pairs (placeholder)."""
    out = tmp_path / "empty.png"
    render_transfer_figure({"per_agent": {}}, out)
    assert out.exists()


# ---------------------------------------------------------------------------
# CLI smoke config sanity
# ---------------------------------------------------------------------------


def test_smoke_config_is_well_formed() -> None:
    """``--smoke`` should produce a config that passes ``validate``."""
    cfg = _smoke_config()
    cfg.validate()
    assert cfg.timesteps_per_opponent <= 50_000
    assert cfg.eval_episodes <= 50
    assert cfg.train_opponents and cfg.heldout_opponents
    assert set(cfg.train_opponents).isdisjoint(cfg.heldout_opponents)

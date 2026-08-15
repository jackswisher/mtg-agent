"""Tests for within-source headline agent-pair significance."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.research.aggregate import _pairwise_significance, aggregate


def _headline_seed_maps() -> dict[str, dict[tuple[str, str, str], dict[int, float]]]:
    """One source with PPO and CGFA on the same deck/opponents/seeds."""
    return {
        "headline": {
            ("ppo", "mono_red_aggro", "mono_red_aggro"): {
                42: 0.40,
                123: 0.41,
                456: 0.42,
                789: 0.43,
                1024: 0.44,
            },
            ("ppo", "mono_red_aggro", "dimir_midrange"): {
                42: 0.50,
                123: 0.51,
                456: 0.52,
                789: 0.53,
                1024: 0.54,
            },
            ("cgfa", "mono_red_aggro", "mono_red_aggro"): {
                42: 0.45,
                123: 0.46,
                456: 0.47,
                789: 0.48,
                1024: 0.49,
            },
            ("cgfa", "mono_red_aggro", "dimir_midrange"): {
                42: 0.60,
                123: 0.61,
                456: 0.62,
                789: 0.63,
                1024: 0.64,
            },
        }
    }


def test_within_source_headline_agent_pair_is_reported() -> None:
    """A single headline eval file can still report CGFA-vs-PPO significance."""
    rows = _pairwise_significance(
        _headline_seed_maps(),
        baseline_agent="ppo",
        headline_compare_agents=["ppo", "cgfa"],
    )

    headline_rows = [r for r in rows if r["holm_family"] == "headline_agent_pair"]
    assert len(headline_rows) == 1
    row = headline_rows[0]
    assert row["source_a"] == "headline(ppo)"
    assert row["source_b"] == "headline(cgfa)"
    assert row["agent"] == "headline"
    assert row["opponent"] == "all_opponents"
    assert row["n_seeds"] == 5
    assert row["diff"] > 0.0
    assert row["p_paired_bootstrap"] is not None


def test_planned_source_pair_compares_ablation_variants_across_agent_names() -> None:
    """Ablation sources can compare PPO against CGFA-full by variant name."""
    rows = _pairwise_significance(
        {
            "ppo": {
                ("ppo", "mono_red_aggro", "dimir_midrange"): {
                    42: 0.40,
                    123: 0.41,
                    456: 0.42,
                    789: 0.43,
                    1024: 0.44,
                },
                ("ppo", "mono_red_aggro", "azorius_control"): {
                    42: 0.50,
                    123: 0.51,
                    456: 0.52,
                    789: 0.53,
                    1024: 0.54,
                },
            },
            "cgfa_full": {
                ("cgfa", "mono_red_aggro", "dimir_midrange"): {
                    42: 0.46,
                    123: 0.47,
                    456: 0.48,
                    789: 0.49,
                    1024: 0.50,
                },
                ("cgfa", "mono_red_aggro", "azorius_control"): {
                    42: 0.55,
                    123: 0.56,
                    456: 0.57,
                    789: 0.58,
                    1024: 0.59,
                },
            },
        },
        baseline_agent=None,
        source_compare_pairs=[("ppo", "cgfa_full")],
    )

    planned_rows = [r for r in rows if r["holm_family"] == "planned_source_pair"]
    assert {r["opponent"] for r in planned_rows} == {
        "azorius_control",
        "dimir_midrange",
        "all_opponents",
    }
    assert all(r["source_a"] == "ppo" for r in planned_rows)
    assert all(r["source_b"] == "cgfa_full" for r in planned_rows)
    assert all(r["diff"] > 0.0 for r in planned_rows)


def test_duplicate_source_labels_are_rejected(tmp_path: Path) -> None:
    """Duplicate source labels must not silently overwrite an eval file."""
    eval_a = tmp_path / "a" / "eval" / "eval_results.json"
    eval_b = tmp_path / "b" / "eval" / "eval_results.json"
    payload = {"trained": [], "baselines": []}
    eval_a.parent.mkdir(parents=True)
    eval_b.parent.mkdir(parents=True)
    eval_a.write_text(json.dumps(payload))
    eval_b.write_text(json.dumps(payload))

    try:
        aggregate(
            eval_paths=[eval_a, eval_b],
            output_dir=tmp_path / "out",
            baseline_agent=None,
            source_labels=["same", "same"],
        )
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:  # pragma: no cover - explicit failure branch for clearer assertion
        raise AssertionError("duplicate labels should raise ValueError")

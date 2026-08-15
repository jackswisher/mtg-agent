"""Tests for Holm-Bonferroni handling in ``aggregate.py``.

The aggregator applies Holm-Bonferroni across the full family of
paired-bootstrap comparisons emitted in a single run and writes both
``p`` (raw) and ``p_holm`` (corrected) into ``significance.tex`` and
the JSON dump. This avoids p-hacking from a per-test 0.05 threshold
applied across many cells.

These tests pin that behaviour:

* every row in the JSON dump and every row in ``significance.tex`` has an
  adjusted ``p_holm``;
* ``p_holm`` is monotone in the raw p-values within a family (Holm
  preserves ordering after multiplication and cumulative max);
* the LaTeX table includes the Holm column with the family size in the
  caption.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.aggregate import (
    _pairwise_significance,
    _write_significance_table,
)

# ---------------------------------------------------------------------------
# Synthetic seed maps
# ---------------------------------------------------------------------------


def _make_two_source_seed_maps(
    *,
    n_seeds: int = 8,
    delta: float,
    rng_seed: int = 0,
) -> dict[str, dict[tuple[str, str, str], dict[int, float]]]:
    """Build two sources with one matching cell separated by ``delta`` mean."""
    rng = np.random.default_rng(rng_seed)
    seeds = list(range(n_seeds))
    # Source A is centred at 0; source B is centred at +delta (paired noise).
    base = rng.normal(0.0, 0.05, size=n_seeds)
    a_vals = {s: float(base[i]) for i, s in enumerate(seeds)}
    b_vals = {s: float(base[i] + delta + rng.normal(0.0, 0.01)) for i, s in enumerate(seeds)}
    key = ("agentX", "deck1", "oppA")
    return {
        "src_A": {key: a_vals},
        "src_B": {key: b_vals},
    }


def _make_multi_cell_seed_maps(
    *,
    n_seeds: int = 8,
    deltas: list[float],
    rng_seed: int = 0,
) -> dict[str, dict[tuple[str, str, str], dict[int, float]]]:
    """Build two sources whose shared cells span a range of effect sizes.

    ``deltas`` must be one entry per cell.  This generates enough rows for
    Holm-Bonferroni to do meaningful work (and to verify monotonicity of
    the adjusted p-values).
    """
    rng = np.random.default_rng(rng_seed)
    seeds = list(range(n_seeds))
    map_a: dict[tuple[str, str, str], dict[int, float]] = {}
    map_b: dict[tuple[str, str, str], dict[int, float]] = {}
    for i, delta in enumerate(deltas):
        key = ("agentX", f"deck{i}", "oppA")
        base = rng.normal(0.0, 0.05, size=n_seeds)
        map_a[key] = {s: float(base[j]) for j, s in enumerate(seeds)}
        map_b[key] = {
            s: float(base[j] + delta + rng.normal(0.0, 0.01)) for j, s in enumerate(seeds)
        }
    return {"src_A": map_a, "src_B": map_b}


# ---------------------------------------------------------------------------
# Holm-Bonferroni propagation
# ---------------------------------------------------------------------------


def test_pairwise_significance_emits_p_holm_for_every_row() -> None:
    """Every row returned by ``_pairwise_significance`` carries an adjusted p."""
    seed_maps = _make_two_source_seed_maps(delta=0.4)
    rows = _pairwise_significance(seed_maps, baseline_agent=None)
    assert rows, "expected at least one significance row for shared cell"
    for r in rows:
        assert "p_holm" in r, "missing Holm-Bonferroni adjusted p in row"
        assert "p_holm_family_size" in r
        assert 0.0 <= r["p_holm"] <= 1.0


def test_p_holm_is_at_least_p_paired_bootstrap_per_row() -> None:
    """For any single row, Holm-Bonferroni is conservative: ``p_holm >= p_raw``."""
    seed_maps = _make_multi_cell_seed_maps(deltas=[0.05, 0.10, 0.20, 0.40, 0.80])
    rows = _pairwise_significance(seed_maps, baseline_agent=None)
    assert len(rows) == 5, f"expected 5 paired comparisons, got {len(rows)}"
    for r in rows:
        assert r["p_holm"] + 1e-12 >= r["p_paired_bootstrap"], (
            "Holm-adjusted p should never be smaller than the raw p-value: "
            f"raw={r['p_paired_bootstrap']:.6g}, holm={r['p_holm']:.6g}"
        )


def test_p_holm_preserves_rank_order_of_raw_p_values() -> None:
    """If raw p1 <= p2 <= p3, then p_holm1 <= p_holm2 <= p_holm3 must hold.

    Holm-Bonferroni multiplies the i-th smallest raw p-value by ``(n - i)``
    and then takes a cumulative maximum, which is monotone.  We verify that
    the rank order is preserved across our synthetic family.
    """
    seed_maps = _make_multi_cell_seed_maps(
        deltas=[0.02, 0.06, 0.12, 0.25, 0.50, 1.00],
        rng_seed=42,
    )
    rows = _pairwise_significance(seed_maps, baseline_agent=None)
    assert len(rows) == 6
    sorted_by_raw = sorted(rows, key=lambda r: r["p_paired_bootstrap"])
    holm_sequence = [r["p_holm"] for r in sorted_by_raw]
    for prev, curr in zip(holm_sequence[:-1], holm_sequence[1:], strict=True):
        assert (
            curr + 1e-12 >= prev
        ), f"Holm-adjusted p-values must be monotone after sorting by raw p: {holm_sequence}"


def test_p_holm_family_size_matches_row_count() -> None:
    """``p_holm_family_size`` must equal the number of rows in the family."""
    seed_maps = _make_multi_cell_seed_maps(deltas=[0.1, 0.3, 0.5])
    rows = _pairwise_significance(seed_maps, baseline_agent=None)
    assert rows
    family_size = rows[0]["p_holm_family_size"]
    assert family_size == len(rows)
    for r in rows:
        assert r["p_holm_family_size"] == family_size


# ---------------------------------------------------------------------------
# LaTeX table rendering
# ---------------------------------------------------------------------------


def test_significance_table_emits_holm_column(tmp_path: Path) -> None:
    r"""``significance.tex`` must include a ``$p_\text{Holm}$`` column header."""
    seed_maps = _make_multi_cell_seed_maps(deltas=[0.1, 0.3])
    rows = _pairwise_significance(seed_maps, baseline_agent=None)
    out = tmp_path / "significance.tex"
    written = _write_significance_table(rows, out)
    assert written is not None
    text = written.read_text()
    assert r"$p_\text{Holm}$" in text, "missing Holm-Bonferroni column header"
    # Family size should appear in the caption.
    assert (
        f"family of {len(rows)} comparisons" in text
    ), f"caption should disclose the multiple-testing family size; got: {text}"


def test_significance_table_returns_none_for_empty_rows(tmp_path: Path) -> None:
    """No rows -> no table file (caller should not crash)."""
    assert _write_significance_table([], tmp_path / "significance.tex") is None


# ---------------------------------------------------------------------------
# JSON dump propagation (end-to-end smoke through the dump structure used
# by aggregate.aggregate -> aggregated_results.json)
# ---------------------------------------------------------------------------


def test_p_holm_round_trips_through_json_dump(tmp_path: Path) -> None:
    """``p_holm`` must survive the ``json.dump(...)`` step."""
    seed_maps = _make_multi_cell_seed_maps(deltas=[0.1, 0.3, 0.5, 0.7])
    rows = _pairwise_significance(seed_maps, baseline_agent=None)
    dump = {"significance": rows}
    out = tmp_path / "aggregated_results.json"
    out.write_text(json.dumps(dump, indent=2))
    loaded = json.loads(out.read_text())
    for r in loaded["significance"]:
        assert "p_holm" in r
        assert "p_holm_family_size" in r

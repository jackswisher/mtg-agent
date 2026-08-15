"""Tests for the minimum-seed-for-bootstrap-p threshold in ``aggregate``.

Paired-bootstrap p-values computed from only 2 to 4 seed pairs are not
trustworthy: the bootstrap distribution at that sample size has only
2 to 4 unique values to resample, so the resulting two-sided p-value
collapses to a handful of discrete buckets. The aggregator therefore
censors the inferential columns (``p_paired_bootstrap``, ``ci_low``,
``ci_high``, ``p_wilcoxon``, ``p_holm``) for cells with fewer than
:data:`~scripts.research.aggregate.MIN_SEEDS_FOR_BOOTSTRAP_P` paired
seeds and renders them as ``$N/A$`` in ``significance.tex``.
Descriptive deltas (``mean_a``, ``mean_b``, ``diff``) are still
reported so the comparison remains visible.

These tests pin the contract:

* ``MIN_SEEDS_FOR_BOOTSTRAP_P`` is at least 5.
* Cells with ``2 <= n < threshold`` get ``insufficient_seeds=True``
  and ``None`` for every inferential field.
* Cells with ``n >= threshold`` keep their full schema.
* The Holm-Bonferroni family size counts only the cells that actually
  carry a valid p-value, so the correction is not artificially
  inflated by censored rows.
* ``significance.tex`` renders ``$N/A$`` for censored cells, keeps the
  numeric ``Diff``, and discloses the censoring in its caption.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.research.aggregate import (
    MIN_SEEDS_FOR_BOOTSTRAP_P,
    _pairwise_significance,
    _write_significance_table,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _two_source_seed_map_with_n(
    *,
    n_seeds: int,
    delta: float = 0.3,
    rng_seed: int = 0,
    key: tuple[str, str, str] = ("agentX", "deck1", "oppA"),
) -> dict[str, dict[tuple[str, str, str], dict[int, float]]]:
    """Two sources sharing exactly one cell with ``n_seeds`` paired seeds."""
    rng = np.random.default_rng(rng_seed)
    seeds = list(range(n_seeds))
    base = rng.normal(0.0, 0.05, size=n_seeds)
    a_vals = {s: float(base[i]) for i, s in enumerate(seeds)}
    b_vals = {s: float(base[i] + delta + rng.normal(0.0, 0.01)) for i, s in enumerate(seeds)}
    return {
        "src_A": {key: a_vals},
        "src_B": {key: b_vals},
    }


def _mixed_n_seed_maps(
    *,
    rng_seed: int = 0,
) -> dict[str, dict[tuple[str, str, str], dict[int, float]]]:
    """Two sources with a mix of cells: some above and some below the threshold.

    Layout:

    * ``deck0`` -- 8 paired seeds (above threshold, real p-value)
    * ``deck1`` -- 3 paired seeds (below threshold, censored)
    * ``deck2`` -- 6 paired seeds (above threshold, real p-value)
    * ``deck3`` -- 2 paired seeds (below threshold, censored)
    """
    rng = np.random.default_rng(rng_seed)
    map_a: dict[tuple[str, str, str], dict[int, float]] = {}
    map_b: dict[tuple[str, str, str], dict[int, float]] = {}
    spec: list[tuple[str, int, float]] = [
        ("deck0", 8, 0.3),
        ("deck1", 3, 0.5),  # censored
        ("deck2", 6, 0.2),
        ("deck3", 2, 0.4),  # censored
    ]
    for deck, n, delta in spec:
        seeds = list(range(n))
        base = rng.normal(0.0, 0.05, size=n)
        key = ("agentX", deck, "oppA")
        map_a[key] = {s: float(base[i]) for i, s in enumerate(seeds)}
        map_b[key] = {
            s: float(base[i] + delta + rng.normal(0.0, 0.01)) for i, s in enumerate(seeds)
        }
    return {"src_A": map_a, "src_B": map_b}


# ---------------------------------------------------------------------------
# Threshold constant
# ---------------------------------------------------------------------------


def test_min_seeds_for_bootstrap_p_is_at_least_five() -> None:
    """The reviewer-grade default must be at least 5.

    NeurIPS reviewers consistently flag bootstrap p-values from <5
    paired seeds as unreliable; the constant must not regress below
    that floor.
    """
    assert MIN_SEEDS_FOR_BOOTSTRAP_P >= 5, (
        f"MIN_SEEDS_FOR_BOOTSTRAP_P regressed to {MIN_SEEDS_FOR_BOOTSTRAP_P}; "
        "must be >= 5 for trustworthy paired-bootstrap reporting."
    )


# ---------------------------------------------------------------------------
# Censoring on insufficient n
# ---------------------------------------------------------------------------


def test_below_threshold_cell_is_reported_with_censored_p_values() -> None:
    """A 3-seed cell still appears in the result list, but with N/A p-values."""
    seed_maps = _two_source_seed_map_with_n(n_seeds=3, delta=0.5)
    rows = _pairwise_significance(seed_maps, baseline_agent=None)
    assert (
        len(rows) == 1
    ), f"expected the censored cell to still be reported (descriptive stats); got {len(rows)} rows"
    r = rows[0]
    assert r["insufficient_seeds"] is True
    assert r["n_seeds"] == 3
    # Descriptive stats survive.
    assert isinstance(r["mean_a"], float)
    assert isinstance(r["mean_b"], float)
    assert isinstance(r["diff"], float)
    # Inferential fields are censored.
    assert r["p_paired_bootstrap"] is None
    assert r["p_wilcoxon"] is None
    assert r["ci_low"] is None
    assert r["ci_high"] is None
    assert r["p_holm"] is None


def test_at_or_above_threshold_cell_keeps_full_inferential_payload() -> None:
    """A 5-seed cell (``== threshold``) must carry real numeric p-values."""
    seed_maps = _two_source_seed_map_with_n(n_seeds=MIN_SEEDS_FOR_BOOTSTRAP_P, delta=0.4)
    rows = _pairwise_significance(seed_maps, baseline_agent=None)
    assert len(rows) == 1
    r = rows[0]
    assert r["insufficient_seeds"] is False
    assert r["n_seeds"] == MIN_SEEDS_FOR_BOOTSTRAP_P
    assert isinstance(r["p_paired_bootstrap"], float)
    assert isinstance(r["ci_low"], float)
    assert isinstance(r["ci_high"], float)
    assert isinstance(r["p_holm"], float)
    assert 0.0 <= r["p_paired_bootstrap"] <= 1.0
    assert 0.0 <= r["p_holm"] <= 1.0


def test_two_seed_cell_is_censored_not_skipped() -> None:
    """Even the bare-minimum 2-seed pair shows up in the table as censored.

    Reviewers want to see that the comparison was attempted; suppressing
    it entirely would let a careless author hide low-power comparisons.
    """
    seed_maps = _two_source_seed_map_with_n(n_seeds=2, delta=0.6)
    rows = _pairwise_significance(seed_maps, baseline_agent=None)
    assert len(rows) == 1
    r = rows[0]
    assert r["insufficient_seeds"] is True
    assert r["p_paired_bootstrap"] is None


def test_one_seed_cell_is_skipped_entirely() -> None:
    """A 1-seed pair has no paired difference; it must be dropped."""
    seed_maps = _two_source_seed_map_with_n(n_seeds=1, delta=0.7)
    rows = _pairwise_significance(seed_maps, baseline_agent=None)
    assert rows == []


# ---------------------------------------------------------------------------
# Holm-Bonferroni family size: only counts valid p-values
# ---------------------------------------------------------------------------


def test_holm_family_size_excludes_censored_cells() -> None:
    """Holm correction must only divide across cells with real p-values.

    With 4 cells (2 valid, 2 censored), the family size must be 2, not
    4 -- otherwise we'd be over-correcting by including bootstrap-p
    values that we explicitly told the reader were unreliable.
    """
    seed_maps = _mixed_n_seed_maps()
    rows = _pairwise_significance(seed_maps, baseline_agent=None)
    assert len(rows) == 4, "all four cells should appear in the result"

    valid = [r for r in rows if r["p_paired_bootstrap"] is not None]
    censored = [r for r in rows if r["p_paired_bootstrap"] is None]
    assert len(valid) == 2 and len(censored) == 2

    family_sizes = {r["p_holm_family_size"] for r in rows}
    assert family_sizes == {2}, (
        f"Holm-Bonferroni family size should be 2 (the count of cells with "
        f"n >= {MIN_SEEDS_FOR_BOOTSTRAP_P}), not 4. Got family_sizes={family_sizes}"
    )

    for r in censored:
        assert r["p_holm"] is None, (
            "Censored cells must not get a numeric Holm p; otherwise readers "
            "will mistake the adjusted value for evidence of significance."
        )


def test_holm_family_size_zero_when_every_cell_is_censored() -> None:
    """If no cell meets the threshold, family size collapses to 0."""
    seed_maps = _two_source_seed_map_with_n(n_seeds=3)
    rows = _pairwise_significance(seed_maps, baseline_agent=None)
    assert len(rows) == 1
    assert rows[0]["insufficient_seeds"]
    assert rows[0]["p_holm_family_size"] == 0
    assert rows[0]["p_holm"] is None


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------


def test_significance_table_renders_n_a_for_censored_cells(tmp_path: Path) -> None:
    """``significance.tex`` must show ``$N/A$`` (not '0' or '1.000') for censored rows.

    The censored row must keep its numeric ``Diff`` so reviewers can
    still see the direction of the effect, but the inferential
    columns (CI / p / p_Holm) must all be ``$N/A$``.  Otherwise the
    table is silently lying about how much evidence we have.
    """
    seed_maps = _mixed_n_seed_maps()
    rows = _pairwise_significance(seed_maps, baseline_agent=None)
    out = tmp_path / "significance.tex"
    written = _write_significance_table(rows, out)
    assert written is not None
    text = written.read_text()

    # At least one censored cell has been rendered with an explicit N/A.
    assert "$N/A$" in text, "censored cells must be rendered as $N/A$; got:\n" + text
    # The deck1 (n=3) and deck3 (n=2) rows must each contain $N/A$.
    deck1_line = next(
        (line for line in text.splitlines() if "deck1" in line),
        None,
    )
    deck3_line = next(
        (line for line in text.splitlines() if "deck3" in line),
        None,
    )
    assert (
        deck1_line is not None and "$N/A$" in deck1_line
    ), f"deck1 (n=3) line should render $N/A$ for inferential columns; got: {deck1_line}"
    assert (
        deck3_line is not None and "$N/A$" in deck3_line
    ), f"deck3 (n=2) line should render $N/A$ for inferential columns; got: {deck3_line}"

    # Caption discloses the censoring count and threshold.
    assert "N/A marks" in text, f"caption should disclose how many cells were censored; got: {text}"
    assert (
        f"$n_s < {MIN_SEEDS_FOR_BOOTSTRAP_P}$" in text
    ), f"caption should disclose the threshold; got: {text}"


def test_significance_table_keeps_real_numbers_for_uncensored_rows(
    tmp_path: Path,
) -> None:
    """Rows above the threshold must still render real numeric CIs / p-values.

    Censoring must be applied per-row, not file-wide. The deck0 (n=8)
    and deck2 (n=6) cells in the mixed-n fixture must continue to
    render numeric CIs and p-values.
    """
    seed_maps = _mixed_n_seed_maps()
    rows = _pairwise_significance(seed_maps, baseline_agent=None)
    out = tmp_path / "significance.tex"
    written = _write_significance_table(rows, out)
    text = written.read_text()

    # deck0 row (n=8) should NOT be N/A in any column.
    deck0_line = next(
        (line for line in text.splitlines() if "deck0" in line),
        None,
    )
    assert deck0_line is not None, f"deck0 row missing from table:\n{text}"
    assert "$N/A$" not in deck0_line, f"deck0 (n=8) must not be censored; got: {deck0_line}"
    # deck2 row (n=6) should NOT be N/A.
    deck2_line = next(
        (line for line in text.splitlines() if "deck2" in line),
        None,
    )
    assert deck2_line is not None
    assert "$N/A$" not in deck2_line, f"deck2 (n=6) must not be censored; got: {deck2_line}"


def test_significance_table_caption_reports_only_valid_family_size(
    tmp_path: Path,
) -> None:
    """The caption must say 'family of N' where N is the count of valid p-values."""
    seed_maps = _mixed_n_seed_maps()
    rows = _pairwise_significance(seed_maps, baseline_agent=None)
    out = tmp_path / "significance.tex"
    written = _write_significance_table(rows, out)
    text = written.read_text()
    assert "family of 2 comparisons" in text, (
        "caption should advertise the actual Holm family size (2 valid cells), "
        f"not 4 (total rows including censored).  Got: {text}"
    )


# ---------------------------------------------------------------------------
# Backwards-compat: full-strength runs still emit the standard schema
# ---------------------------------------------------------------------------


def test_full_strength_run_still_passes_holm_invariants() -> None:
    """When every cell has n>=threshold, schema matches the standard contract.

    Specifically: ``p_holm`` is numeric, ``p_holm >= p_paired_bootstrap``,
    and the family size equals the number of rows.  This guards against a
    regression where the censoring path accidentally censors valid rows.
    """
    seed_maps = _two_source_seed_map_with_n(n_seeds=8, delta=0.4)
    rows = _pairwise_significance(seed_maps, baseline_agent=None)
    assert len(rows) == 1
    r = rows[0]
    assert r["insufficient_seeds"] is False
    assert isinstance(r["p_paired_bootstrap"], float)
    assert isinstance(r["p_holm"], float)
    assert r["p_holm"] + 1e-12 >= r["p_paired_bootstrap"]
    assert r["p_holm_family_size"] == 1

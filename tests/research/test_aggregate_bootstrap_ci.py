"""Tests for bootstrap-CI rendering in ``aggregate.py``.

The aggregator:

* stores a 95% percentile bootstrap CI per cell in ``_aggregate_across_seeds``;
* pools seed-level win rates within each plot bar and reports a 95%
  percentile bootstrap CI on the pooled vector;
* writes the same CI (mean and ``[lo, hi]``) into ``headline.tex``;
* labels axes and captions with "95% bootstrap CI".

These tests pin those invariants on synthetic seed maps.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import numpy as np  # noqa: E402

from scripts.research.aggregate import (  # noqa: E402
    _aggregate_across_seeds,
    _plot_headline,
    _plot_win_rate_by_opponent,
    _write_headline_table,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_seed_map(
    *,
    agents: list[str],
    decks: list[str],
    opponents: list[str],
    seeds: list[int],
    base_winrate: float,
    rng_seed: int = 0,
) -> dict[tuple[str, str, str], dict[int, float]]:
    """Produce a seed map that mimics the on-disk eval_results layout."""
    rng = np.random.default_rng(rng_seed)
    out: dict[tuple[str, str, str], dict[int, float]] = {}
    for a in agents:
        for d in decks:
            for o in opponents:
                vals = rng.normal(loc=base_winrate, scale=0.1, size=len(seeds))
                vals = np.clip(vals, 0.0, 1.0)
                out[(a, d, o)] = {s: float(vals[i]) for i, s in enumerate(seeds)}
    return out


# ---------------------------------------------------------------------------
# Per-cell CI: percentile bootstrap with metadata
# ---------------------------------------------------------------------------


def test_aggregate_across_seeds_emits_bootstrap_ci_with_metadata() -> None:
    """Each cell gets a percentile bootstrap CI with method metadata."""
    seed_map = _make_seed_map(
        agents=["ppo"],
        decks=["aggro"],
        opponents=["control", "midrange"],
        seeds=list(range(8)),
        base_winrate=0.55,
    )
    agg = _aggregate_across_seeds(seed_map)
    for key, summary in agg.items():
        assert "ci_lo" in summary and "ci_hi" in summary
        assert "ci_method" in summary
        assert summary["ci_method"] == "percentile_bootstrap_95"
        assert summary["ci_n_resamples"] == 10_000
        assert (
            0.0 <= summary["ci_lo"] <= summary["mean"] + 1e-9
        ), f"ci_lo > mean for {key}: {summary}"
        assert (
            summary["mean"] - 1e-9 <= summary["ci_hi"] <= 1.0
        ), f"mean > ci_hi or > 1.0 for {key}: {summary}"


def test_aggregate_across_seeds_ci_contains_population_mean_with_high_prob() -> None:
    """Sanity check: across many synthetic cells, the 95% CI captures the truth.

    We don't pin to ``>= 0.95`` (10k resamples, finite n) but the CIs
    must be nontrivially calibrated, with empirical coverage clearly
    above chance.
    """
    rng = np.random.default_rng(42)
    n_cells = 60
    n_seeds = 12
    truth = 0.6
    contains = 0
    for _c in range(n_cells):
        vals = rng.normal(truth, 0.12, size=n_seeds)
        vals = np.clip(vals, 0.0, 1.0)
        seed_map = {
            ("agent", "deck", "opp"): {s: float(vals[i]) for i, s in enumerate(range(n_seeds))}
        }
        agg = _aggregate_across_seeds(seed_map)
        s = agg[("agent", "deck", "opp")]
        if s["ci_lo"] <= truth <= s["ci_hi"]:
            contains += 1
    coverage = contains / n_cells
    assert coverage >= 0.85, f"bootstrap CI coverage too low: {coverage:.2f}; expected >= 0.85"


# ---------------------------------------------------------------------------
# Plot pooling: every contributing seed observation is bootstrapped
# ---------------------------------------------------------------------------


def test_plot_win_rate_by_opponent_pools_seed_level_observations(tmp_path: Path) -> None:
    """The figure runs end-to-end and writes a PNG with the new pooling logic."""
    seed_map = _make_seed_map(
        agents=["ppo", "cgfa"],
        decks=["aggro", "control"],
        opponents=["control_agent", "midrange_agent"],
        seeds=list(range(6)),
        base_winrate=0.5,
    )
    seed_maps_by_source = {"run_A": seed_map}
    agg_by_source = {"run_A": _aggregate_across_seeds(seed_map)}
    out = tmp_path / "win_rate_by_opponent.png"
    written = _plot_win_rate_by_opponent(agg_by_source, seed_maps_by_source, out)
    assert written is not None
    assert written.exists()
    assert written.stat().st_size > 1024, "expected a non-trivial PNG"


def test_plot_win_rate_by_opponent_returns_none_for_empty_input(tmp_path: Path) -> None:
    """Empty inputs degrade cleanly (regression for caller resilience)."""
    assert _plot_win_rate_by_opponent({}, {}, tmp_path / "win_rate_by_opponent.png") is None


def test_plot_headline_pools_seed_level_observations(tmp_path: Path) -> None:
    """Headline figure uses pooled seed-level observations and writes a PNG."""
    seed_map = _make_seed_map(
        agents=["ppo", "cgfa"],
        decks=["aggro"],
        opponents=["control_agent", "midrange_agent"],
        seeds=list(range(8)),
        base_winrate=0.6,
    )
    seed_maps_by_source = {"run_A": seed_map}
    agg_by_source = {"run_A": _aggregate_across_seeds(seed_map)}
    out = tmp_path / "headline_comparison.png"
    written = _plot_headline(agg_by_source, seed_maps_by_source, out)
    assert written is not None
    assert written.exists()
    assert written.stat().st_size > 1024


# ---------------------------------------------------------------------------
# LaTeX table: bootstrap CI in the cell, no more `\pm SEM`
# ---------------------------------------------------------------------------


def test_headline_table_emits_bootstrap_ci_in_cells(tmp_path: Path) -> None:
    r"""``headline.tex`` writes ``mean [lo, hi]`` and never the misleading ``\pm SEM``."""
    seed_map = _make_seed_map(
        agents=["ppo"],
        decks=["aggro"],
        opponents=["control_agent"],
        seeds=list(range(8)),
        base_winrate=0.55,
    )
    seed_maps_by_source = {"run_A": seed_map}
    agg_by_source = {"run_A": _aggregate_across_seeds(seed_map)}
    out = tmp_path / "headline.tex"
    written = _write_headline_table(agg_by_source, seed_maps_by_source, out)
    text = written.read_text()
    assert r"\caption" in text
    assert (
        "95\\% percentile bootstrap CI" in text
    ), "caption should disclose the bootstrap CI method"
    # Cell format: ``$mean\;[lo,\,hi]$``; check both delimiters.
    assert "\\;[" in text, f"bootstrap CI delimiter missing from cells:\n{text}"
    assert ",\\," in text, f"bootstrap CI inter-bound delimiter missing from cells:\n{text}"
    assert "]$" in text, f"bootstrap CI closing bracket missing from cells:\n{text}"
    # No more pm SEM cells.
    assert "\\pm" not in text, "table must not render error bars with `\\pm SEM`; use bootstrap CIs"

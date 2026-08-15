r"""Aggregate evaluation sweeps into figures, tables, and statistics.

Reads one or more ``eval_results.json`` files (produced by
:mod:`scripts.research.eval_sweep`), aggregates metrics across seeds, performs
paired-bootstrap significance tests, and writes:

- ``figures/win_rate_by_opponent.png`` (mean +/- 95% CI bars per opponent x agent)
- ``figures/headline_comparison.png``  (overall win rate per agent x deck)
- ``figures/per_matchup_heatmap.png``  (when multiple agents/decks)
- ``tables/headline.tex``              (LaTeX table of mean +/- std per (agent, deck))
- ``tables/significance.tex``          (pairwise paired-bootstrap CI + p-values)
- ``aggregated_results.json``          (flat machine-readable form)

Usage:
    uv run python -m scripts.research.aggregate \\
        --eval-results results/research/ppo_baseline_v1/eval/eval_results.json \\
                       results/research/causal_v1/eval/eval_results.json \\
        --output-dir results/research/comparison_v1 \\
        --baseline-agent ppo

If a single eval-results file is provided, the headline figure is generated
without cross-experiment comparisons.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mtg.utils.cli_display import console, print_divider, print_logo
from scripts.research.stats import paired_bootstrap_test, wilcoxon_signed_rank

PALETTE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
    "#F0E442",
]

HEURISTIC_AGENTS = {
    "greedy_aggro",
    "control",
    "midrange",
    "ramp",
    "convoke_aggro",
}

DISPLAY_LABELS = {
    "cgfa": "CGFA-PPO",
    "ppo": "PPO",
    "causal": "Causal",
    "cgfa_full": "CGFA full",
    "cgfa_no_cal": "CGFA no calibration",
    "cgfa_no_gate": "CGFA no gate",
    "cgfa_scalar_only": "CGFA scalar-only",
    "random": "Random",
    **dict.fromkeys(HEURISTIC_AGENTS, "Heuristic"),
}

METHOD_COLORS = {
    "CGFA-PPO": "#0072B2",
    "PPO": "#D55E00",
    "Causal": "#009E73",
    "CGFA full": "#0072B2",
    "CGFA no calibration": "#D55E00",
    "CGFA no gate": "#009E73",
    "CGFA scalar-only": "#CC79A7",
    "Random": "#999999",
    "Heuristic": "#E69F00",
}


def _display_label(name: str) -> str:
    """Return a paper-friendly method label."""
    return DISPLAY_LABELS.get(name, name.replace("_", " ").title())


def _method_color(label: str, fallback_idx: int = 0) -> str:
    """Return a stable color for a method label."""
    return METHOD_COLORS.get(label, PALETTE[fallback_idx % len(PALETTE)])


# Minimum paired seeds required to report a paired-bootstrap p-value.
#
# A paired bootstrap with only 2 to 4 seeds has a degenerate resampling
# distribution (the bootstrap can only ever select among 2 to 4 unique
# values), so the resulting two-sided p-value collapses to a handful of
# discrete buckets and the apparent precision (e.g. ``p=0.0625`` from a
# 4-seed run) is essentially noise. At ``n=5`` the bootstrap distribution
# has 3,125 unique resamples and starts behaving like the asymptotic
# percentile bootstrap.
#
# Cells with 2 to 4 paired seeds are still emitted in the significance
# table so the comparison is visible, but their ``p_paired_bootstrap`` /
# ``p_holm`` columns are set to ``None`` and rendered as "$N/A$".
# Descriptive stats (mean_a / mean_b / diff) are kept because they are
# unbiased estimators of the underlying quantity even at small ``n``.
#
# Cells with fewer than 2 paired seeds are skipped entirely; there is no
# meaningful paired difference to compute from a single seed.
MIN_SEEDS_FOR_BOOTSTRAP_P: int = 5


def _load_eval(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _seed_level_winrates(
    eval_data: dict[str, Any],
    *,
    kind: str = "trained",
) -> dict[tuple[str, str, str], dict[int, float]]:
    """Index (agent, deck, opponent) -> {seed: win_rate}.

    ``kind`` selects the ``"trained"`` or ``"baselines"`` block.
    """
    out: dict[tuple[str, str, str], dict[int, float]] = defaultdict(dict)
    for entry in eval_data.get(kind, []):
        agent = entry["agent"]
        deck = entry["player_deck"]
        seed = int(entry["seed"])
        for opp, summary in entry["per_opponent"].items():
            out[(agent, deck, opp)][seed] = float(summary["win_rate"])
    return out


def _aggregate_across_seeds(
    seed_map: dict[tuple[str, str, str], dict[int, float]],
) -> dict[tuple[str, str, str], dict[str, float]]:
    """Compute mean / std / SEM and a 95% bootstrap CI for every cell.

    Per-cell summary fields:

    * ``mean`` / ``std`` / ``sem``: standard moments across seeds.
    * ``ci_lo`` / ``ci_hi``: 95% percentile bootstrap CI on the mean
      (10k resamples). This is the field used everywhere downstream
      (figures, tables, JSON dumps). The percentile bootstrap is used
      instead of a t-distribution interval because the latter assumes
      Gaussian seed-level win rates, which is brittle with the small
      ``n_seeds`` typical of our runs.
    * ``n_seeds``: number of seeds in the cell.
    """
    from scripts.research.stats import bootstrap_mean_ci

    agg: dict[tuple[str, str, str], dict[str, float]] = {}
    for key, by_seed in seed_map.items():
        vals = np.array(list(by_seed.values()), dtype=float)
        n = vals.size
        if n == 0:
            continue
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if n > 1 else 0.0
        sem = std / np.sqrt(n) if n > 0 else 0.0
        ci = bootstrap_mean_ci(vals, n_resamples=10_000)
        agg[key] = {
            "mean": mean,
            "std": std,
            "sem": sem,
            "ci_lo": float(np.clip(ci.lo, 0.0, 1.0)),
            "ci_hi": float(np.clip(ci.hi, 0.0, 1.0)),
            "ci_method": "percentile_bootstrap_95",
            "ci_n_resamples": 10_000,
            "n_seeds": int(n),
        }
    return agg


def _plot_win_rate_by_opponent(
    agg_by_source: dict[str, dict[tuple[str, str, str], dict[str, float]]],
    seed_maps_by_source: dict[str, dict[tuple[str, str, str], dict[int, float]]],
    out_path: Path,
) -> Path | None:
    r"""Grouped bar chart: opponents on x-axis, agents/decks as colored groups.

    For each (source, opponent) bar we pool every per-seed win rate
    contributing to that bar (across decks/agents in this source) and
    compute a 95% percentile bootstrap CI on the pooled vector. This
    avoids the bias of a normal-approximation interval such as
    :math:`1.96\cdot\sqrt{\text{mean}(\text{sem}^2)}`, which
    underestimates uncertainty when the per-cell means are themselves
    heterogeneous and is not a calibrated 95% CI.

    Note: ``agg_by_source`` is still required for the per-cell mean used
    as the bar height; ``seed_maps_by_source`` carries the raw seed-level
    values needed for the bootstrap.
    """
    from scripts.research.stats import bootstrap_mean_ci

    if not agg_by_source:
        return None
    opponents = sorted({key[2] for src in agg_by_source.values() for key in src})
    if not opponents:
        return None

    sources = sorted(agg_by_source.keys())
    show_source_in_legend = len(sources) > 1
    x = np.arange(len(opponents))
    width = 0.8 / max(1, len(sources))

    fig, ax = plt.subplots(figsize=(max(8, len(opponents) * 1.6), 5))
    for i, src in enumerate(sources):
        means: list[float] = []
        errs_lo: list[float] = []
        errs_hi: list[float] = []
        for opp in opponents:
            keys = [k for k in agg_by_source[src] if k[2] == opp]
            if not keys:
                means.append(0.0)
                errs_lo.append(0.0)
                errs_hi.append(0.0)
                continue
            pooled = []
            for k in keys:
                pooled.extend(seed_maps_by_source.get(src, {}).get(k, {}).values())
            if not pooled:
                # Fall back to mean of per-cell means with no error bar
                vals = np.array([agg_by_source[src][k]["mean"] for k in keys])
                means.append(float(vals.mean()))
                errs_lo.append(0.0)
                errs_hi.append(0.0)
                continue
            arr = np.asarray(pooled, dtype=float)
            mean = float(arr.mean())
            ci = bootstrap_mean_ci(arr, n_resamples=10_000)
            means.append(mean)
            # matplotlib expects positive offsets relative to the bar height.
            errs_lo.append(max(0.0, mean - float(np.clip(ci.lo, 0.0, 1.0))))
            errs_hi.append(max(0.0, float(np.clip(ci.hi, 0.0, 1.0)) - mean))

        ax.bar(
            x + i * width - 0.4 + width / 2,
            means,
            width=width,
            label=src if show_source_in_legend else None,
            color=PALETTE[i % len(PALETTE)],
            edgecolor="white",
            yerr=[errs_lo, errs_hi],
            capsize=4,
            ecolor="#333",
            error_kw={"linewidth": 1, "alpha": 0.7},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([o.replace("_", " ").title() for o in opponents], rotation=20, ha="right")
    ax.set_ylabel(
        "Win Rate (mean, 95% percentile bootstrap CI on pooled seed-level rates)",
        fontweight="medium",
    )
    ax.set_title("Win Rate by Opponent", fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="50% baseline")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_headline(
    agg_by_source: dict[str, dict[tuple[str, str, str], dict[str, float]]],
    seed_maps_by_source: dict[str, dict[tuple[str, str, str], dict[int, float]]],
    out_path: Path,
) -> Path | None:
    """Headline overall win rate per (source, agent, deck), one bar per cell.

    Error bars are 95% percentile bootstrap CIs computed on the pooled
    seed-level win rates that contribute to the bar (one observation per
    (opponent, seed)). The percentile bootstrap is preferred over an
    SEM-on-per-cell-means estimator, which underestimates error and is
    not a calibrated CI.
    """
    from scripts.research.stats import bootstrap_mean_ci

    sources = sorted(agg_by_source.keys())
    show_source_in_label = len(sources) > 1

    records: list[dict[str, Any]] = []
    color_by_label: dict[str, str] = {}
    for src in sources:
        per_agent_deck: dict[tuple[str, str], list[float]] = defaultdict(list)
        for (agent, deck, opp), _s in agg_by_source[src].items():
            if show_source_in_label and agent in {"random", *HEURISTIC_AGENTS}:
                continue
            for v in seed_maps_by_source.get(src, {}).get((agent, deck, opp), {}).values():
                per_agent_deck[(agent, deck)].append(float(v))
        for (agent, deck), vals in sorted(per_agent_deck.items()):
            arr = np.array(vals, dtype=float)
            if arr.size == 0:
                continue
            mean = float(arr.mean())
            ci = bootstrap_mean_ci(arr, n_resamples=10_000)
            label = _display_label(src if show_source_in_label else agent)
            color_by_label.setdefault(
                label,
                _method_color(label, fallback_idx=len(color_by_label)),
            )
            records.append(
                {
                    "deck": deck,
                    "label": label,
                    "mean": mean,
                    "err_lo": max(0.0, mean - float(np.clip(ci.lo, 0.0, 1.0))),
                    "err_hi": max(0.0, float(np.clip(ci.hi, 0.0, 1.0)) - mean),
                    "color": color_by_label[label],
                }
            )

    if not records:
        return None

    decks = sorted({str(record["deck"]) for record in records})
    ncols = min(3, len(decks))
    nrows = int(np.ceil(len(decks) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 3.2 * nrows),
        squeeze=False,
        sharex=True,
    )

    for ax, deck in zip(axes.flat, decks, strict=False):
        deck_records = [record for record in records if record["deck"] == deck]
        deck_records = sorted(deck_records, key=lambda r: float(r["mean"]), reverse=True)
        y = np.arange(len(deck_records))
        means_arr = np.asarray([record["mean"] for record in deck_records], dtype=float)
        errs_lo_arr = np.asarray([record["err_lo"] for record in deck_records], dtype=float)
        errs_hi_arr = np.asarray([record["err_hi"] for record in deck_records], dtype=float)
        labels = [str(record["label"]) for record in deck_records]
        colors = [str(record["color"]) for record in deck_records]
        bars = ax.barh(
            y,
            means_arr,
            color=colors,
            edgecolor="black",
            linewidth=0.6,
            xerr=[errs_lo_arr, errs_hi_arr],
            capsize=4,
            ecolor="#333",
        )
        for b, m in zip(bars, means_arr, strict=False):
            ax.annotate(
                f"{m:.0%}",
                xy=(min(float(m) + 0.025, 1.05), b.get_y() + b.get_height() / 2),
                ha="left",
                va="center",
                fontsize=8,
                fontweight="bold",
            )
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(deck.replace("_", " ").title(), fontweight="bold")
        ax.set_xlim(0, 1.1)
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, axis="x", alpha=0.3)

    for ax in axes.flat[len(decks) :]:
        ax.axis("off")

    title = (
        "Ablation Comparison by Player Deck"
        if show_source_in_label
        else "Benchmark Comparison by Player Deck"
    )
    fig.suptitle(title, fontweight="bold", fontsize=13)
    fig.supxlabel(
        "Overall win rate (mean across opponents and seeds; 95% bootstrap CI)",
        fontsize=9,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_heatmap(
    agg: dict[tuple[str, str, str], dict[str, float]],
    out_path: Path,
) -> Path | None:
    """Deck-faceted heatmap of method x opponent win rates."""
    if not agg:
        return None
    decks = sorted({deck for (_, deck, _) in agg})
    opponents = sorted({opp for (_, _, opp) in agg})
    if not decks or not opponents:
        return None

    ncols = min(3, len(decks))
    nrows = int(np.ceil(len(decks) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.4 * ncols, 3.6 * nrows),
        squeeze=False,
        sharex=True,
    )
    im = None
    for ax, deck in zip(axes.flat, decks, strict=False):
        agents = sorted({agent for (agent, row_deck, _) in agg if row_deck == deck})
        labels = [_display_label(agent) for agent in agents]
        matrix = np.full((len(agents), len(opponents)), np.nan, dtype=float)
        for i, agent in enumerate(agents):
            for j, opp in enumerate(opponents):
                entry = agg.get((agent, deck, opp))
                if entry is not None:
                    matrix[i, j] = entry["mean"]

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(opponents)))
        ax.set_xticklabels(
            [opp.replace("_", " ").title() for opp in opponents],
            rotation=25,
            ha="right",
            fontsize=8,
        )
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(deck.replace("_", " ").title(), fontweight="bold")
        for i in range(len(agents)):
            for j in range(len(opponents)):
                v = matrix[i, j]
                if np.isnan(v):
                    continue
                ax.text(
                    j,
                    i,
                    f"{v:.0%}",
                    ha="center",
                    va="center",
                    color="black" if 0.3 < v < 0.7 else "white",
                    fontsize=8,
                    fontweight="bold",
                )

    for ax in axes.flat[len(decks) :]:
        ax.axis("off")
    if im is not None:
        fig.subplots_adjust(
            bottom=0.24,
            left=0.10,
            right=0.88,
            top=0.84,
            wspace=0.30,
            hspace=0.55,
        )
        cax = fig.add_axes((0.90, 0.25, 0.02, 0.56))
        fig.colorbar(im, cax=cax, label="Win Rate")
    fig.suptitle("Per-Matchup Win Rate by Player Deck", fontweight="bold", fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _write_headline_table(
    agg_by_source: dict[str, dict[tuple[str, str, str], dict[str, float]]],
    seed_maps_by_source: dict[str, dict[tuple[str, str, str], dict[int, float]]],
    out_path: Path,
) -> Path:
    """Write a LaTeX booktabs table of overall win rate per (source, agent, deck).

    Each cell is mean and 95% percentile bootstrap CI computed on the
    pooled seed-level win rates that contribute to the row (one
    observation per (opponent, seed)). The bootstrap CI is the same
    statistic shown on the headline figure and is preferred over a
    ``mean +/- SEM`` summary, which underestimates uncertainty.
    """
    from scripts.research.stats import bootstrap_mean_ci

    lines: list[str] = [
        "% auto-generated by scripts/research/aggregate.py",
        "\\begin{table}[t]",
        "\\centering",
        (
            "\\caption{Overall win rate (mean and 95\\% percentile bootstrap CI "
            "on pooled seed-level rates, 10k resamples).}"
        ),
        "\\label{tab:headline}",
        "\\begin{tabular}{llll}",
        "\\toprule",
        "Source & Agent & Player Deck & Win Rate (95\\% CI) \\\\",
        "\\midrule",
    ]
    for src in sorted(agg_by_source.keys()):
        per_agent_deck: dict[tuple[str, str], list[float]] = defaultdict(list)
        for (agent, deck, opp), _s in agg_by_source[src].items():
            for v in seed_maps_by_source.get(src, {}).get((agent, deck, opp), {}).values():
                per_agent_deck[(agent, deck)].append(float(v))
        for (agent, deck), vals in sorted(per_agent_deck.items()):
            arr = np.array(vals, dtype=float)
            if arr.size == 0:
                continue
            mean = float(arr.mean())
            ci = bootstrap_mean_ci(arr, n_resamples=10_000)
            lo = float(np.clip(ci.lo, 0.0, 1.0))
            hi = float(np.clip(ci.hi, 0.0, 1.0))
            lines.append(
                f"{src} & {agent} & {deck.replace('_', ' ')} & "
                f"${mean:.3f}\\;[{lo:.3f},\\,{hi:.3f}]$ \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    return out_path


def _pairwise_significance(
    seed_maps_by_source: dict[str, dict[tuple[str, str, str], dict[int, float]]],
    baseline_agent: str | None,
    *,
    headline_compare_agents: list[str] | None = None,
    source_compare_pairs: list[tuple[str, str]] | None = None,
    source_compare_seed_maps_by_source: (
        dict[str, dict[tuple[str, str, str], dict[int, float]]] | None
    ) = None,
    min_seeds_for_p: int = MIN_SEEDS_FOR_BOOTSTRAP_P,
) -> list[dict[str, Any]]:
    """Run paired-bootstrap tests across seeds for every comparable cell.

    A "cell" is identified by (agent, deck, opponent). For each pair of sources
    that both contain the same cell with overlapping seeds, we compute the
    paired difference. If ``baseline_agent`` is given we also run cross-source
    comparisons for the same (deck, opponent) where one source has the baseline
    agent and another has a non-baseline agent.

    Minimum-seed enforcement
    ~~~~~~~~~~~~~~~~~~~~~~~~
    Cells with ``< min_seeds_for_p`` (default :data:`MIN_SEEDS_FOR_BOOTSTRAP_P`
    = 5) paired seeds are still reported (they appear in the result
    list with their descriptive stats ``mean_a``, ``mean_b``, ``diff``)
    but their inferential fields (``p_paired_bootstrap``, ``ci_low``,
    ``ci_high``, ``p_wilcoxon``, ``p_holm``) are set to ``None`` and the
    cell is flagged with ``insufficient_seeds=True``. The
    Holm-Bonferroni family size counts only the comparisons that
    actually carry a valid p-value, so the multiple-testing correction
    is not artificially inflated by the censored rows.

    Cells with ``< 2`` paired seeds are skipped entirely (there is no
    paired difference to compute from a single seed pair).

    After collecting raw paired-bootstrap p-values, valid subsets are
    Holm-Bonferroni adjusted within their pre-registered comparison family
    so headline agent comparisons are not corrected together with
    cross-protocol roll-ups. See :func:`scripts.research.stats.holm_bonferroni`.
    """
    from scripts.research.stats import holm_bonferroni

    results: list[dict[str, Any]] = []
    sources = sorted(seed_maps_by_source.keys())

    def _build_row(
        *,
        source_a: str,
        source_b: str,
        agent: str,
        deck: str,
        opp: str,
        a_arr: np.ndarray,
        b_arr: np.ndarray,
        holm_family: str,
    ) -> dict[str, Any]:
        """Build one row for the displayed comparison ``source_b - source_a``."""
        n = int(a_arr.size)
        mean_a = float(a_arr.mean())
        mean_b = float(b_arr.mean())
        diff = float(mean_b - mean_a)
        if n >= min_seeds_for_p:
            pb = paired_bootstrap_test(b_arr, a_arr, n_resamples=10_000)
            wx = wilcoxon_signed_rank(a_arr, b_arr) if n >= 6 else None
            return {
                "source_a": source_a,
                "source_b": source_b,
                "agent": agent,
                "player_deck": deck,
                "opponent": opp,
                "holm_family": holm_family,
                "n_seeds": n,
                "mean_a": mean_a,
                "mean_b": mean_b,
                "diff": pb.mean_diff,
                "ci_low": pb.ci_low,
                "ci_high": pb.ci_high,
                "p_paired_bootstrap": pb.p_value,
                "p_wilcoxon": wx.p_value if wx else None,
                "insufficient_seeds": False,
                "min_seeds_for_p": int(min_seeds_for_p),
            }
        # 2 <= n < min_seeds_for_p: report descriptive stats, censor p-values.
        return {
            "source_a": source_a,
            "source_b": source_b,
            "agent": agent,
            "player_deck": deck,
            "opponent": opp,
            "holm_family": holm_family,
            "n_seeds": n,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "diff": diff,
            "ci_low": None,
            "ci_high": None,
            "p_paired_bootstrap": None,
            "p_wilcoxon": None,
            "insufficient_seeds": True,
            "min_seeds_for_p": int(min_seeds_for_p),
        }

    def _values_for_source_cell(
        seed_map: dict[tuple[str, str, str], dict[int, float]],
        *,
        deck: str,
        opp: str,
    ) -> dict[int, list[float]]:
        """Collect all agent values in one source for a deck/opponent cell."""
        out: dict[int, list[float]] = defaultdict(list)
        for (_agent, d, o), by_seed in seed_map.items():
            if d != deck or o != opp:
                continue
            for seed, value in by_seed.items():
                out[int(seed)].append(float(value))
        return out

    def _append_planned_source_pair_rows(
        *,
        source_a: str,
        source_b: str,
        map_a: dict[tuple[str, str, str], dict[int, float]],
        map_b: dict[tuple[str, str, str], dict[int, float]],
    ) -> None:
        """Compare two sources even when their trained agent names differ.

        This is used for ablations, where sources are variant names
        (``ppo``, ``cgfa_scalar_only``, ``cgfa_full``) and the scientific
        comparison is source-vs-source rather than same-agent-vs-same-agent.
        Rows are emitted per opponent and as an all-opponents per-deck mean.
        """
        deck_opp_pairs = sorted(
            {(deck, opp) for (_agent, deck, opp) in map_a}
            & {(deck, opp) for (_agent, deck, opp) in map_b}
        )
        by_deck_scores_a: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        by_deck_scores_b: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        for deck, opp in deck_opp_pairs:
            vals_a = _values_for_source_cell(map_a, deck=deck, opp=opp)
            vals_b = _values_for_source_cell(map_b, deck=deck, opp=opp)
            shared = sorted(set(vals_a) & set(vals_b))
            if len(shared) < 2:
                continue
            a_arr = np.asarray([float(np.mean(vals_a[s])) for s in shared], dtype=float)
            b_arr = np.asarray([float(np.mean(vals_b[s])) for s in shared], dtype=float)
            results.append(
                _build_row(
                    source_a=source_a,
                    source_b=source_b,
                    agent="planned_source_pair",
                    deck=deck,
                    opp=opp,
                    a_arr=a_arr,
                    b_arr=b_arr,
                    holm_family="planned_source_pair",
                )
            )
            for seed, value in zip(shared, a_arr, strict=True):
                by_deck_scores_a[deck][seed].append(float(value))
            for seed, value in zip(shared, b_arr, strict=True):
                by_deck_scores_b[deck][seed].append(float(value))

        for deck in sorted(set(by_deck_scores_a) & set(by_deck_scores_b)):
            shared = sorted(set(by_deck_scores_a[deck]) & set(by_deck_scores_b[deck]))
            if len(shared) < 2:
                continue
            a_arr = np.asarray(
                [float(np.mean(by_deck_scores_a[deck][seed])) for seed in shared],
                dtype=float,
            )
            b_arr = np.asarray(
                [float(np.mean(by_deck_scores_b[deck][seed])) for seed in shared],
                dtype=float,
            )
            results.append(
                _build_row(
                    source_a=source_a,
                    source_b=source_b,
                    agent="planned_source_pair",
                    deck=deck,
                    opp="all_opponents",
                    a_arr=a_arr,
                    b_arr=b_arr,
                    holm_family="planned_source_pair",
                )
            )

    for i, src_a in enumerate(sources):
        for src_b in sources[i + 1 :]:
            map_a = seed_maps_by_source[src_a]
            map_b = seed_maps_by_source[src_b]
            common_keys = set(map_a) & set(map_b)
            for key in sorted(common_keys):
                seeds_a = map_a[key]
                seeds_b = map_b[key]
                shared = sorted(set(seeds_a) & set(seeds_b))
                if len(shared) < 2:
                    continue
                a_arr = np.array([seeds_a[s] for s in shared])
                b_arr = np.array([seeds_b[s] for s in shared])
                results.append(
                    _build_row(
                        source_a=src_a,
                        source_b=src_b,
                        agent=key[0],
                        deck=key[1],
                        opp=key[2],
                        a_arr=a_arr,
                        b_arr=b_arr,
                        holm_family="same_agent_cross_source",
                    )
                )

    if source_compare_pairs:
        planned_maps = source_compare_seed_maps_by_source or seed_maps_by_source
        for src_a, src_b in source_compare_pairs:
            if src_a not in planned_maps or src_b not in planned_maps:
                continue
            _append_planned_source_pair_rows(
                source_a=src_a,
                source_b=src_b,
                map_a=planned_maps[src_a],
                map_b=planned_maps[src_b],
            )

    if headline_compare_agents and len(headline_compare_agents) >= 2:
        baseline = headline_compare_agents[0]
        challengers = list(dict.fromkeys(headline_compare_agents[1:]))
        for src in sources:
            seed_map = seed_maps_by_source[src]
            decks = sorted({deck for (_agent, deck, _opp) in seed_map})
            for challenger in challengers:
                for deck in decks:
                    shared_seeds = sorted(
                        {
                            seed
                            for (agent, d, _opp), by_seed in seed_map.items()
                            if agent in {baseline, challenger} and d == deck
                            for seed in by_seed
                        }
                    )
                    baseline_scores: list[float] = []
                    challenger_scores: list[float] = []
                    for seed in shared_seeds:
                        baseline_opps = {
                            opp
                            for (agent, d, opp), by_seed in seed_map.items()
                            if agent == baseline and d == deck and seed in by_seed
                        }
                        challenger_opps = {
                            opp
                            for (agent, d, opp), by_seed in seed_map.items()
                            if agent == challenger and d == deck and seed in by_seed
                        }
                        shared_opps = sorted(baseline_opps & challenger_opps)
                        if not shared_opps:
                            continue
                        baseline_scores.append(
                            float(
                                np.mean(
                                    [seed_map[(baseline, deck, opp)][seed] for opp in shared_opps]
                                )
                            )
                        )
                        challenger_scores.append(
                            float(
                                np.mean(
                                    [seed_map[(challenger, deck, opp)][seed] for opp in shared_opps]
                                )
                            )
                        )
                    if len(baseline_scores) < 2:
                        continue
                    results.append(
                        _build_row(
                            source_a=f"{src}({baseline})",
                            source_b=f"{src}({challenger})",
                            agent="headline",
                            deck=deck,
                            opp="all_opponents",
                            a_arr=np.asarray(baseline_scores, dtype=float),
                            b_arr=np.asarray(challenger_scores, dtype=float),
                            holm_family="headline_agent_pair",
                        )
                    )

    if baseline_agent:
        # Cross-source comparison: same (deck, opponent), baseline_agent vs others.
        for src_a in sources:
            for src_b in sources:
                if src_a == src_b:
                    continue
                map_a = seed_maps_by_source[src_a]
                map_b = seed_maps_by_source[src_b]
                cells_a = {
                    (d, o): seeds for (a, d, o), seeds in map_a.items() if a == baseline_agent
                }
                cells_b = {
                    (d, o): seeds for (a, d, o), seeds in map_b.items() if a != baseline_agent
                }
                for (deck, opp), seeds_a in cells_a.items():
                    seeds_b = cells_b.get((deck, opp))
                    if seeds_b is None:
                        continue
                    shared = sorted(set(seeds_a) & set(seeds_b))
                    if len(shared) < 2:
                        continue
                    a_arr = np.array([seeds_a[s] for s in shared])
                    b_arr = np.array([seeds_b[s] for s in shared])
                    row = _build_row(
                        source_a=f"{src_a}({baseline_agent})",
                        source_b=src_b,
                        agent="vs_baseline",
                        deck=deck,
                        opp=opp,
                        a_arr=a_arr,
                        b_arr=b_arr,
                        holm_family="cross_agent_baseline",
                    )
                    # Cross-source baseline pairs do not run Wilcoxon
                    # (paired-bootstrap is the only test on this path)
                    # so the entry is cleared so the schema stays consistent.
                    if row.get("p_wilcoxon") is not None and not row.get("insufficient_seeds"):
                        row["p_wilcoxon"] = None
                    results.append(row)

    # ---- Family-wise Holm-Bonferroni correction ------------------------
    # Only adjust comparisons whose paired-bootstrap p-value is valid
    # (i.e. the cell met the min-seed threshold). Censored rows get
    # ``p_holm = None`` so the renderer can show "$N/A$" without
    # propagating spurious "1.000" adjusted values that would still
    # look like real numbers in a table.
    families = sorted({str(r.get("holm_family", "default")) for r in results})
    for family in families:
        family_indices = [
            i for i, r in enumerate(results) if str(r.get("holm_family", "default")) == family
        ]
        valid_indices = [
            i for i in family_indices if results[i].get("p_paired_bootstrap") is not None
        ]
        family_size = len(valid_indices)
        if valid_indices:
            raw_p = [float(results[i]["p_paired_bootstrap"]) for i in valid_indices]
            adjusted = holm_bonferroni(raw_p)
            for idx, p_adj in zip(valid_indices, adjusted, strict=True):
                results[idx]["p_holm"] = float(p_adj)
        for idx in family_indices:
            results[idx].setdefault("p_holm", None)
            results[idx]["p_holm_family_size"] = family_size
    return results


def _write_significance_table(rows: list[dict[str, Any]], out_path: Path) -> Path | None:
    r"""Write a LaTeX table of paired-bootstrap pairwise comparisons.

    Two p-value columns are emitted:

    * ``p`` (raw paired-bootstrap p-value, 10k resamples)
    * ``p_holm`` (Holm-Bonferroni adjusted p-value inside the row's
      pre-registered comparison family)

    The corrected column is shown alongside the raw column so claims
    like "12 of 12 pairwise tests are significant" still address the
    multiple-testing burden. Holm correction is applied separately to
    comparison families such as headline agent-pair tests and cross-source
    baseline roll-ups.

    Cells with fewer than :data:`MIN_SEEDS_FOR_BOOTSTRAP_P` paired
    seeds render their ``CI``, ``p``, and ``p_\text{Holm}`` columns as
    ``$N/A$``. Their descriptive ``Diff`` is still shown so the
    direction of the effect is visible, but the inferential columns are
    suppressed because the bootstrap distribution at ``n<5`` is too
    degenerate to produce a trustworthy p-value.
    """
    if not rows:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    family_sizes_by_name: dict[str, int] = {}
    for r in rows:
        family = str(r.get("holm_family", "default"))
        family_sizes_by_name[family] = max(
            family_sizes_by_name.get(family, 0),
            int(r.get("p_holm_family_size", 0) or 0),
        )
    family_summary = ", ".join(
        f"{name.replace('_', ' ')} family of {size} comparisons"
        for name, size in sorted(family_sizes_by_name.items())
    )
    n_censored = sum(1 for r in rows if r.get("insufficient_seeds"))
    min_seeds = int(rows[0].get("min_seeds_for_p", MIN_SEEDS_FOR_BOOTSTRAP_P))
    caption_extra = ""
    if n_censored:
        caption_extra = (
            f" \\textit{{N/A marks {n_censored} comparison"
            f"{'s' if n_censored != 1 else ''} with $n_s < {min_seeds}$ "
            "paired seeds, where the bootstrap distribution is too "
            "degenerate to produce a trustworthy p-value; descriptive "
            "deltas are still reported.}}"
        )

    lines: list[str] = [
        "% auto-generated by scripts/research/aggregate.py",
        "\\begin{table}[t]",
        "\\centering",
        (
            "\\caption{Pairwise comparisons (paired bootstrap on seed-level "
            "win rates). $p$ is the raw paired-bootstrap p-value (10k resamples); "
            "$p_\\text{Holm}$ is the Holm-Bonferroni adjusted p-value "
            "within each comparison family "
            f"({family_summary}) with $n_s \\geq {min_seeds}$ paired seeds."
            f"{caption_extra}}}"
        ),
        "\\label{tab:significance}",
        "\\begin{tabular}{llllllll}",
        "\\toprule",
        ("Comparison & Deck & Opponent & $n_s$ & Diff & 95\\% CI & $p$ & $p_\\text{Holm}$ \\\\"),
        "\\midrule",
    ]
    for r in rows:
        comp = f"{r['source_b']} - {r['source_a']}"
        diff_str = f"{r['diff']:+.3f}"
        if r.get("insufficient_seeds") or r.get("p_paired_bootstrap") is None:
            ci_str = "$N/A$"
            p_str = "$N/A$"
            p_holm_str = "$N/A$"
        else:
            ci_low = float(r["ci_low"])
            ci_high = float(r["ci_high"])
            p_raw = float(r["p_paired_bootstrap"])
            p_holm_val = r.get("p_holm")
            p_holm = float(p_holm_val) if p_holm_val is not None else p_raw
            ci_str = f"[{ci_low:+.3f}, {ci_high:+.3f}]"
            p_str = f"{p_raw:.3g}"
            p_holm_str = f"{p_holm:.3g}"
        lines.append(
            f"{comp} & {r['player_deck'].replace('_', ' ')} & "
            f"{r['opponent'].replace('_', ' ')} & {r['n_seeds']} & "
            f"{diff_str} & {ci_str} & {p_str} & {p_holm_str} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    out_path.write_text("\n".join(lines))
    return out_path


def aggregate(
    eval_paths: list[Path],
    output_dir: Path,
    baseline_agent: str | None,
    source_labels: list[str] | None,
    headline_compare_agents: list[str] | None = None,
    source_compare_pairs: list[tuple[str, str]] | None = None,
) -> Path:
    """Aggregate one or more eval_results.json files into figures and tables."""
    print_logo()
    print_divider("Aggregating Results")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    if source_labels and len(source_labels) != len(eval_paths):
        raise ValueError("source_labels length must match number of eval_paths")
    sources: list[str] = source_labels or [p.parent.parent.name for p in eval_paths]
    if len(set(sources)) != len(sources):
        duplicates = sorted({src for src in sources if sources.count(src) > 1})
        raise ValueError(f"source_labels must be unique; duplicate label(s): {duplicates}")

    seed_maps_by_source: dict[str, dict[tuple[str, str, str], dict[int, float]]] = {}
    trained_seed_maps_by_source: dict[str, dict[tuple[str, str, str], dict[int, float]]] = {}
    agg_by_source: dict[str, dict[tuple[str, str, str], dict[str, float]]] = {}

    for src, path in zip(sources, eval_paths, strict=False):
        data = _load_eval(path)
        # Trained + baselines combined (baselines tagged via agent name)
        seed_map_trained = _seed_level_winrates(data, kind="trained")
        seed_map_base = _seed_level_winrates(data, kind="baselines")
        seed_map = {**seed_map_trained, **seed_map_base}
        trained_seed_maps_by_source[src] = seed_map_trained
        seed_maps_by_source[src] = seed_map
        agg_by_source[src] = _aggregate_across_seeds(seed_map)
        console.print(
            f"  loaded [bold]{src}[/bold]: "
            f"{len(seed_map)} (agent, deck, opp) cells, "
            f"{len(data.get('trained', []))} trained runs, "
            f"{len(data.get('baselines', []))} baseline runs"
        )

    # ---- Figures ----
    p1 = _plot_win_rate_by_opponent(
        agg_by_source,
        seed_maps_by_source,
        figures_dir / "win_rate_by_opponent.png",
    )
    if p1:
        console.print(f"  figure: {p1}")
    p2 = _plot_headline(
        agg_by_source,
        seed_maps_by_source,
        figures_dir / "headline_comparison.png",
    )
    if p2:
        console.print(f"  figure: {p2}")
    # heatmap from the first source (most common case: a single trained sweep)
    if agg_by_source:
        first_src = next(iter(agg_by_source))
        p3 = _plot_heatmap(agg_by_source[first_src], figures_dir / "per_matchup_heatmap.png")
        if p3:
            console.print(f"  figure: {p3}")

    # ---- Tables ----
    t1 = _write_headline_table(
        agg_by_source,
        seed_maps_by_source,
        tables_dir / "headline.tex",
    )
    console.print(f"  table:  {t1}")

    # ---- Significance ----
    sig_rows = _pairwise_significance(
        seed_maps_by_source,
        baseline_agent=baseline_agent,
        headline_compare_agents=headline_compare_agents,
        source_compare_pairs=source_compare_pairs,
        source_compare_seed_maps_by_source=trained_seed_maps_by_source,
    )
    t2 = _write_significance_table(sig_rows, tables_dir / "significance.tex")
    if t2:
        console.print(f"  table:  {t2}")

    # ---- Machine-readable dump ----
    dump = {
        "sources": sources,
        "eval_paths": [str(p) for p in eval_paths],
        "baseline_agent": baseline_agent,
        "headline_compare_agents": headline_compare_agents,
        "source_compare_pairs": source_compare_pairs,
        "aggregated": {
            src: {f"{a}|{d}|{o}": v for (a, d, o), v in agg_by_source[src].items()}
            for src in agg_by_source
        },
        "significance": sig_rows,
    }
    out_json = output_dir / "aggregated_results.json"
    out_json.write_text(json.dumps(dump, indent=2))
    console.print(f"  data:   {out_json}")

    print_divider("Aggregation Complete")
    return out_json


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        prog="aggregate",
        description="Aggregate eval sweeps into figures and tables for reporting.",
    )
    p.add_argument("--eval-results", nargs="+", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument(
        "--baseline-agent",
        default=None,
        help=(
            "Agent name treated as the baseline for cross-source comparisons "
            "(e.g. 'ppo' to compare causal vs ppo)."
        ),
    )
    p.add_argument(
        "--source-labels",
        nargs="*",
        default=None,
        help="Optional human-readable label per --eval-results entry.",
    )
    p.add_argument(
        "--headline-compare-agents",
        nargs="+",
        default=None,
        help=(
            "Within each source, compare the first agent to each later agent on "
            "per-seed headline means pooled over shared opponents."
        ),
    )
    return p.parse_args()


def main() -> int:
    """Entry point for ``python -m scripts.research.aggregate``."""
    args = parse_args()
    aggregate(
        eval_paths=args.eval_results,
        output_dir=args.output_dir,
        baseline_agent=args.baseline_agent,
        source_labels=args.source_labels,
        headline_compare_agents=args.headline_compare_agents,
        source_compare_pairs=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

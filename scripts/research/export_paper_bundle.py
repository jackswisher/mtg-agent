r"""Export completed research-run artifacts into a paper bundle.

The script copies figures and LaTeX table fragments from a completed
``mtg-research paper`` run into a directory with the same layout and file
names expected by ``.paper/neurips_2026.tex``:

    figures/
    tables/
    raw_numbers/

Two paper-specific summary tables (``tab3_main_results.tex`` and
``tab4_transfer_gap.tex``) are also generated programmatically from the
research-run JSON dumps so that every table consumed by the manuscript is
produced by this script and lives under ``tables/``.

Example:
    uv run python -m scripts.research.export_paper_bundle \\
        --experiment-name paper_full_20260428_1200 \\
        --output-dir results/research/paper_full_20260428_1200/paper_bundle

Then copy the generated directory contents into ``.paper/``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.stats import (
    bootstrap_mean_ci,
    holm_bonferroni,
    paired_bootstrap_test,
)

MIN_SEEDS_FOR_BOOTSTRAP_P = 5

_DECK_LABELS: dict[str, str] = {
    "mono_red_aggro": "Mono-Red Aggro",
    "azorius_control": "Azorius Control",
    "dimir_midrange": "Dimir Midrange",
    "domain_ramp": "Domain Ramp",
    "boros_convoke": "Boros Convoke",
}

_AGENT_LABELS: dict[str, str] = {
    "ppo": "PPO",
    "cgfa": "CGFA-PPO",
    "causal": "CWM-PPO",
    "cgfa_scalar_only": "CGFA scalar-only",
    "cgfa_no_gate": "CGFA no-gate",
    "cgfa_no_cal": "CGFA no-cal",
    "cgfa_full": "CGFA-PPO",
    "cgfa_no_scm_init": "CGFA no-SCM-init",
    "cgfa_interventional_cal": "CGFA interventional-cal",
}


@dataclass(frozen=True)
class CopySpec:
    """One source-to-destination artifact copy."""

    source: Path
    destination: Path
    required: bool = True


def _resolve_root(args: argparse.Namespace) -> Path:
    if args.root is not None:
        return args.root
    return args.results_root / args.experiment_name


def _pretty_deck(name: str) -> str:
    return _DECK_LABELS.get(name, name.replace("_", " ").title())


def _pretty_agent(name: str) -> str:
    return _AGENT_LABELS.get(name, name.replace("_", " "))


def _parse_agent_from_source(source: str) -> str:
    """Extract ``ppo`` from a label like ``headline(ppo)``."""
    match = re.search(r"\(([^)]+)\)$", source)
    return match.group(1) if match else source


def _pick_transfer_figure(root: Path, transfer_fold: str | None) -> Path | None:
    """Return the transfer figure to expose as ``fig5_transfer_gap.png``.

    In fixed-transfer mode there is a single ``transfer/transfer`` directory.
    In leave-one-out mode there are multiple ``transfer/loo_*`` folds; callers
    can select one via ``--transfer-fold``. If omitted, the first fold in sorted
    order is used and all fold figures are also copied with fold-specific names.
    """
    fixed = root / "transfer" / "transfer" / "figures" / "transfer_gap.png"
    if fixed.exists():
        return fixed

    fold_dirs = sorted((root / "transfer").glob("loo_*"))
    if transfer_fold:
        selected = root / "transfer" / transfer_fold / "transfer" / "figures" / "transfer_gap.png"
        return selected if selected.exists() else None
    for fold_dir in fold_dirs:
        candidate = fold_dir / "transfer" / "figures" / "transfer_gap.png"
        if candidate.exists():
            return candidate
    return None


def _transfer_fold_specs(root: Path, output_dir: Path) -> list[CopySpec]:
    specs: list[CopySpec] = []
    for fold_dir in sorted((root / "transfer").glob("loo_*")):
        src = fold_dir / "transfer" / "figures" / "transfer_gap.png"
        if src.exists():
            specs.append(
                CopySpec(
                    src,
                    output_dir / "figures" / f"fig5_transfer_gap_{fold_dir.name}.png",
                    required=False,
                )
            )
    return specs


def _transfer_report_specs(root: Path, output_dir: Path) -> list[CopySpec]:
    specs: list[CopySpec] = []
    fixed = root / "transfer" / "transfer" / "transfer_report.json"
    if fixed.exists():
        specs.append(
            CopySpec(
                fixed,
                output_dir / "raw_numbers" / "transfer_report.json",
                required=False,
            )
        )

    for fold_dir in sorted((root / "transfer").glob("loo_*")):
        src = fold_dir / "transfer" / "transfer_report.json"
        if src.exists():
            specs.append(
                CopySpec(
                    src,
                    output_dir / "raw_numbers" / f"transfer_report_{fold_dir.name}.json",
                    required=False,
                )
            )
    return specs


def build_copy_specs(root: Path, output_dir: Path, transfer_fold: str | None) -> list[CopySpec]:
    """Build the artifact mapping from a paper-run root to bundle layout."""
    specs = [
        CopySpec(
            root / "headline" / "aggregated" / "figures" / "headline_comparison.png",
            output_dir / "figures" / "fig3a_headline_comparison.png",
        ),
        CopySpec(
            root / "headline" / "aggregated" / "figures" / "per_matchup_heatmap.png",
            output_dir / "figures" / "fig3b_per_matchup_heatmap.png",
        ),
        CopySpec(
            root / "ablation" / "aggregated" / "figures" / "headline_comparison.png",
            output_dir / "figures" / "fig4_ablation.png",
        ),
        CopySpec(
            root / "ablation" / "figures" / "cgfa_calibration.png",
            output_dir / "figures" / "fig6_cgfa_calibration.png",
        ),
        CopySpec(
            root / "ablation" / "case_study" / "case_study.png",
            output_dir / "figures" / "fig7_case_study.png",
        ),
        CopySpec(
            root / "cross_source" / "tables" / "headline.tex",
            output_dir / "tables" / "tab1_headline.tex",
        ),
        CopySpec(
            root / "ablation" / "aggregated" / "tables" / "headline.tex",
            output_dir / "tables" / "tab2_ablation.tex",
        ),
        CopySpec(
            root / "headline" / "aggregated" / "aggregated_results.json",
            output_dir / "raw_numbers" / "headline_aggregated.json",
            required=False,
        ),
        CopySpec(
            root / "ablation" / "aggregated" / "aggregated_results.json",
            output_dir / "raw_numbers" / "ablation_aggregated.json",
            required=False,
        ),
        CopySpec(
            root / "cross_source" / "aggregated_results.json",
            output_dir / "raw_numbers" / "cross_source_aggregated.json",
            required=False,
        ),
        CopySpec(
            root / "ablation" / "case_study" / "case_study_outcome.json",
            output_dir / "raw_numbers" / "case_study_outcome.json",
            required=False,
        ),
    ]

    transfer = _pick_transfer_figure(root, transfer_fold)
    if transfer is not None:
        specs.append(
            CopySpec(
                transfer,
                output_dir / "figures" / "fig5_transfer_gap.png",
                required=False,
            )
        )
    else:
        placeholder_transfer = (
            root
            / "transfer"
            / (transfer_fold or "<selected_fold>")
            / "transfer"
            / "figures"
            / "transfer_gap.png"
        )
        specs.append(
            CopySpec(
                placeholder_transfer,
                output_dir / "figures" / "fig5_transfer_gap.png",
                required=False,
            )
        )

    specs.extend(_transfer_fold_specs(root, output_dir))
    specs.extend(_transfer_report_specs(root, output_dir))
    return specs


def _load_pooled_rates(
    eval_paths: list[Path],
) -> dict[tuple[str, str], list[float]]:
    """Return ``{(agent, deck): [per-(opponent, seed) win rates]}``.

    Pools the per-cell-per-seed rates so a percentile-bootstrap on the
    list reproduces the marginal-win-rate CI used by the headline figure
    and the canonical ``headline.tex`` aggregator output. Eval files from
    multiple sources are pooled (e.g. headline + ablation under
    ``cross_source``).
    """
    pooled: dict[tuple[str, str], list[float]] = defaultdict(list)
    for path in eval_paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for entry in data.get("trained", []):
            agent = entry.get("agent")
            deck = entry.get("player_deck")
            if agent is None or deck is None:
                continue
            for opp_stats in entry.get("per_opponent", {}).values():
                wr = opp_stats.get("win_rate")
                if wr is not None:
                    pooled[(agent, deck)].append(float(wr))
    return pooled


def _load_seed_marginal_rates(
    eval_paths: list[Path],
) -> dict[tuple[str, str], dict[int, float]]:
    """Return ``{(agent, deck): {seed: mean_over_opponents}}``.

    These paired seed marginals are the unit for headline PPO-vs-CGFA tests.
    They let the export script report exploratory bootstrap p-values even when
    the upstream aggregator censors inferential p-values for very small runs.
    """
    marginals: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for path in eval_paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for entry in data.get("trained", []):
            agent = entry.get("agent")
            deck = entry.get("player_deck")
            seed = entry.get("seed")
            if agent is None or deck is None or seed is None:
                continue
            rates = [
                float(stats["win_rate"])
                for stats in entry.get("per_opponent", {}).values()
                if stats.get("win_rate") is not None
            ]
            if rates:
                marginals[(agent, deck)][int(seed)] = float(np.mean(rates))
    return marginals


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}"


_AGENT_SORT_PRIORITY: dict[str, int] = {"ppo": 0, "cgfa": 1, "cgfa_full": 1, "causal": 2}


def _agent_sort_key(agent: str) -> tuple[int, str]:
    return _AGENT_SORT_PRIORITY.get(agent, 99), agent


def _format_p(value: float | None, insufficient: bool) -> str:
    if insufficient or value is None:
        return "N/A"
    if value < 1e-4:
        return r"$<0.0001$"
    return f"${value:.4f}$"


def _generate_main_results_table(root: Path, output_path: Path) -> bool:
    """Produce ``tab3_main_results.tex`` from headline aggregation data.

    Uses the headline aggregation rather than the cross-source aggregate, since
    the latter also contains transfer-source learned-agent comparisons that
    belong in Table 4 rather than the headline results table.
    Per-(agent, deck) marginal CIs are bootstrapped from per-seed marginal win
    rates derived from each ``eval_results.json`` referenced by the aggregator.
    """
    headline_path = root / "headline" / "aggregated" / "aggregated_results.json"
    agg_path = headline_path
    if not agg_path.exists():
        print(f"WARN: cannot find aggregated_results.json under {root}; skipping tab3.")
        return False

    agg = json.loads(agg_path.read_text())
    eval_paths = [Path(p) for p in agg.get("eval_paths", [])]
    if not eval_paths:
        print("WARN: no eval_paths in aggregated_results.json; skipping tab3.")
        return False

    pooled = _load_pooled_rates(eval_paths)
    seed_marginals = _load_seed_marginal_rates(eval_paths)

    sig_entries = [
        s for s in agg.get("significance", []) if s.get("holm_family") == "headline_agent_pair"
    ]
    if not sig_entries:
        print("WARN: no headline_agent_pair entries in significance; skipping tab3.")
        return False

    sig_entries = sorted(sig_entries, key=lambda s: (s["player_deck"], s["source_b"]))

    rows: list[dict[str, Any]] = []
    for sig in sig_entries:
        deck = sig["player_deck"]
        agent_a = _parse_agent_from_source(sig["source_a"])
        agent_b = _parse_agent_from_source(sig["source_b"])

        pool_a = pooled.get((agent_a, deck), [])
        pool_b = pooled.get((agent_b, deck), [])
        ci_a = bootstrap_mean_ci(pool_a) if pool_a else None
        ci_b = bootstrap_mean_ci(pool_b) if pool_b else None

        p_boot = sig.get("p_paired_bootstrap")
        p_holm = sig.get("p_holm")
        exploratory_p = False
        if p_boot is None:
            paired_a = seed_marginals.get((agent_a, deck), {})
            paired_b = seed_marginals.get((agent_b, deck), {})
            paired_seeds = sorted(set(paired_a) & set(paired_b))
            if paired_seeds:
                test = paired_bootstrap_test(
                    [paired_b[seed] for seed in paired_seeds],
                    [paired_a[seed] for seed in paired_seeds],
                    rng=np.random.default_rng(0),
                )
                p_boot = test.p_value
                exploratory_p = len(paired_seeds) < MIN_SEEDS_FOR_BOOTSTRAP_P

        rows.append(
            {
                "deck": deck,
                "agent_a": agent_a,
                "agent_b": agent_b,
                "mean_a": float(sig["mean_a"]),
                "mean_b": float(sig["mean_b"]),
                "ci_a": ci_a,
                "ci_b": ci_b,
                "n_s": int(sig["n_seeds"]),
                "diff": float(sig["diff"]),
                "p_boot": p_boot,
                "p_holm": p_holm,
                "insufficient": bool(sig.get("insufficient_seeds", False)),
                "exploratory_p": exploratory_p,
            }
        )

    rows_with_p = [row for row in rows if row["p_boot"] is not None]
    adjusted = holm_bonferroni([float(row["p_boot"]) for row in rows_with_p])
    for row, p_holm in zip(rows_with_p, adjusted, strict=False):
        if row["p_holm"] is None:
            row["p_holm"] = p_holm

    exploratory = any(row["exploratory_p"] for row in rows)
    caption_extra = (
        r" For $n_s < 5$, these $p$-values are reported as exploratory "
        r"descriptive diagnostics rather than confirmatory inference because "
        r"the paired-bootstrap distribution is necessarily coarse."
        if exploratory
        else ""
    )

    lines = [
        "% auto-generated by scripts/research/export_paper_bundle.py",
        r"\begin{table}[!t]",
        r"\centering",
        (
            r"\caption{Headline learned-agent results per player deck. Win rates "
            r"are means over the in-distribution opponent pool and paired seeds; "
            r"95\% CIs are percentile-bootstrap intervals over pooled "
            r"per-opponent seed rates (10k resamples). $n_s$ is "
            r"the number of paired seeds available; $\Delta$ is the per-deck mean "
            r"win rate of the test agent minus the reference agent; $p_\text{boot}$ "
            r"and $p_\text{Holm}$ are the raw and Holm-Bonferroni adjusted "
            r"paired-bootstrap $p$-values within the headline learned-agent "
            r"comparison family." + caption_extra + "}"
        ),
        r"\label{tab:main_results}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        (
            r"Deck & Agent & Win Rate (\%) & 95\% CI & $n_s$ & $\Delta$ (pp) "
            r"& $p_\text{boot}$ & $p_\text{Holm}$ \\"
        ),
        r"\midrule",
    ]

    for idx, row in enumerate(rows):
        deck_label = _pretty_deck(row["deck"])
        agent_a_label = _pretty_agent(row["agent_a"])
        agent_b_label = _pretty_agent(row["agent_b"])

        ci_a_str = (
            f"$[{_format_pct(row['ci_a'].lo)},\\,{_format_pct(row['ci_a'].hi)}]$"
            if row["ci_a"] is not None
            else "---"
        )
        ci_b_str = (
            f"$[{_format_pct(row['ci_b'].lo)},\\,{_format_pct(row['ci_b'].hi)}]$"
            if row["ci_b"] is not None
            else "---"
        )

        delta_str = f"${row['diff'] * 100:+.1f}$"
        p_boot_str = _format_p(row["p_boot"], False)
        p_holm_str = _format_p(row["p_holm"], False)

        lines.append(
            f"{deck_label} & {agent_a_label} & "
            f"${_format_pct(row['mean_a'])}$ & {ci_a_str} & "
            f"${row['n_s']}$ & --- & --- & --- \\\\"
        )
        lines.append(
            f"{deck_label} & \\textbf{{{agent_b_label}}} & "
            f"$\\mathbf{{{_format_pct(row['mean_b'])}}}$ & {ci_b_str} & "
            f"${row['n_s']}$ & {delta_str} & {p_boot_str} & {p_holm_str} \\\\"
        )
        if idx < len(rows) - 1:
            lines.append(r"\midrule")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    return True


def _collect_transfer_per_agent(
    root: Path,
) -> dict[str, dict[str, Any]]:
    """Return per-agent transfer-gap stats aggregated across available folds.

    Each value is ``{n_pairs, in_dist_mean, heldout_mean, gap_mean,
    gap_ci_lo, gap_ci_hi, p_value}``. Pools the ``pairs`` arrays from
    every available transfer report (fixed-mode
    ``transfer/transfer/transfer_report.json`` and / or LOO folds
    ``transfer/loo_*/transfer/transfer_report.json``), recomputes means
    from the pooled pairs, and runs a paired bootstrap on the gap when
    there are enough pairs. With a single fold the recomputed values
    coincide with what the upstream ``transfer_sweep`` already wrote to
    ``transfer_report.json``.
    """
    pairs_by_agent: dict[str, list[tuple[float, float]]] = defaultdict(list)
    candidates: list[Path] = []
    fixed = root / "transfer" / "transfer" / "transfer_report.json"
    if fixed.exists():
        candidates.append(fixed)
    candidates.extend(sorted((root / "transfer").glob("loo_*/transfer/transfer_report.json")))

    for path in candidates:
        report = json.loads(path.read_text())
        for agent, payload in report.get("per_agent", {}).items():
            for entry in payload.get("pairs", []):
                in_dist = entry.get("in_dist")
                heldout = entry.get("heldout")
                if in_dist is None or heldout is None:
                    continue
                pairs_by_agent[agent].append((float(in_dist), float(heldout)))

    out: dict[str, dict[str, Any]] = {}
    for agent, agent_pairs in pairs_by_agent.items():
        if not agent_pairs:
            continue
        in_dist = np.array([p[0] for p in agent_pairs], dtype=float)
        heldout = np.array([p[1] for p in agent_pairs], dtype=float)
        gap = in_dist - heldout
        n = int(in_dist.size)
        if n >= 2:
            test = paired_bootstrap_test(in_dist, heldout)
            ci_lo, ci_hi, p_value = test.ci_low, test.ci_high, test.p_value
        else:
            ci_lo = ci_hi = p_value = None
        out[agent] = {
            "n_pairs": n,
            "in_dist_mean": float(in_dist.mean()),
            "heldout_mean": float(heldout.mean()),
            "gap_mean": float(gap.mean()),
            "gap_ci_lo": ci_lo,
            "gap_ci_hi": ci_hi,
            "p_value": p_value,
        }
    return out


def _generate_transfer_gap_table(root: Path, output_path: Path) -> bool:
    """Produce ``tab4_transfer_gap.tex`` from transfer reports.

    Aggregates over all available folds (fixed-mode and / or leave-one-out)
    and applies Holm-Bonferroni within the per-agent family. CIs and
    $p$-values are reported as long as the upstream paired bootstrap
    succeeds; the manuscript caveats the trustworthiness of $p$-values at
    small $n_s$ separately.
    """
    per_agent = _collect_transfer_per_agent(root)
    if not per_agent:
        print(f"WARN: no transfer_report.json found under {root}; skipping tab4.")
        return False

    rows: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    for agent, payload in sorted(per_agent.items(), key=lambda kv: _agent_sort_key(kv[0])):
        rows.append(
            {
                "agent": agent,
                "n_s": int(payload["n_pairs"]),
                "in_dist": float(payload["in_dist_mean"]),
                "heldout": float(payload["heldout_mean"]),
                "gap": float(payload["gap_mean"]),
                "ci_lo": payload["gap_ci_lo"],
                "ci_hi": payload["gap_ci_hi"],
                "p_value": payload["p_value"],
                "insufficient": payload["p_value"] is None,
            }
        )
        if payload["p_value"] is not None:
            raw_p_values.append(float(payload["p_value"]))

    if not rows:
        return False

    if raw_p_values:
        adjusted = holm_bonferroni(raw_p_values)
        adj_iter = iter(adjusted)
        for row in rows:
            row["p_holm"] = next(adj_iter) if row["p_value"] is not None else None
    else:
        for row in rows:
            row["p_holm"] = None

    caption_extra = ""

    lines = [
        "% auto-generated by scripts/research/export_paper_bundle.py",
        r"\begin{table}[!htbp]",
        r"\centering",
        (
            r"\caption{Cross-archetype generalisation gap per learned agent, "
            r"paired across seeds. Held-out evaluation aggregates over the "
            r"leave-one-out folds of the opponent pool; in-distribution "
            r"evaluation aggregates over the matched training-pool folds. "
            r"$\Delta = \mathrm{WinRate}_{\mathrm{in\text{-}dist}} - "
            r"\mathrm{WinRate}_{\mathrm{held\text{-}out}}$ (positive $=$ "
            r"generalisation drop, negative $=$ held-out is easier). $n_s$ is "
            r"the number of paired seed-by-fold observations; $p_\text{boot}$ "
            r"tests $\Delta = 0$ with a paired bootstrap (10k resamples); "
            r"$p_\text{Holm}$ is the Holm-Bonferroni adjusted $p$-value within "
            r"the per-agent transfer family." + caption_extra + "}"
        ),
        r"\label{tab:transfer_gap}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        (
            r"Agent & $n_s$ & In-dist (\%) & Held-out (\%) "
            r"& $\Delta$ (pp) & 95\% CI on $\Delta$ "
            r"& $p_\text{boot}$ & $p_\text{Holm}$ \\"
        ),
        r"\midrule",
    ]
    for row in rows:
        if row["ci_lo"] is None or row["ci_hi"] is None:
            ci_str = "---"
        else:
            ci_str = f"$[{row['ci_lo'] * 100:+.1f},\\,{row['ci_hi'] * 100:+.1f}]$"
        lines.append(
            f"{_pretty_agent(row['agent'])} & ${row['n_s']}$ & "
            f"${_format_pct(row['in_dist'])}$ & ${_format_pct(row['heldout'])}$ & "
            f"${row['gap'] * 100:+.1f}$ & {ci_str} & "
            f"{_format_p(row['p_value'], row['insufficient'])} & "
            f"{_format_p(row['p_holm'], row['insufficient'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    return True


def export_bundle(
    root: Path,
    output_dir: Path,
    transfer_fold: str | None,
    strict: bool,
) -> tuple[int, int]:
    """Copy artifacts and return ``(copied, missing)`` counts."""
    specs = build_copy_specs(root, output_dir, transfer_fold)
    copied = 0
    missing = 0

    for subdir in ("figures", "tables", "raw_numbers"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    for spec in specs:
        if not spec.source.exists():
            missing += 1
            level = "ERROR" if spec.required or strict else "WARN"
            print(f"{level}: missing {spec.source}")
            if spec.required or strict:
                continue
            continue

        spec.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(spec.source, spec.destination)
        copied += 1
        print(f"copied {spec.source} -> {spec.destination}")

    if _generate_main_results_table(root, output_dir / "tables" / "tab3_main_results.tex"):
        copied += 1
        print(f"generated {output_dir / 'tables' / 'tab3_main_results.tex'}")
    else:
        missing += 1

    if _generate_transfer_gap_table(root, output_dir / "tables" / "tab4_transfer_gap.tex"):
        copied += 1
        print(f"generated {output_dir / 'tables' / 'tab4_transfer_gap.tex'}")
    else:
        missing += 1

    if strict and missing:
        raise FileNotFoundError(f"{missing} expected artifact(s) were missing")

    required_missing = [spec.source for spec in specs if spec.required and not spec.source.exists()]
    if required_missing:
        raise FileNotFoundError(
            "Required artifact(s) missing:\n" + "\n".join(f"- {path}" for path in required_missing)
        )

    return copied, missing


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Export a completed mtg-research paper run into figures/, tables/, " "and raw_numbers/."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--experiment-name",
        help="Experiment name under --results-root, e.g. paper_full_...",
    )
    source.add_argument(
        "--root",
        type=Path,
        help="Direct path to the completed experiment root",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/research"),
        help="Parent directory used with --experiment-name",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination bundle directory; copy its contents into .paper/",
    )
    parser.add_argument(
        "--transfer-fold",
        help=(
            "Optional leave-one-out fold directory name to expose as fig5_transfer_gap.png, "
            "for example loo_azorius_control"
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any optional artifact is missing, not just required figures/tables.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the export command."""
    args = parse_args()
    root = _resolve_root(args)
    if not root.exists():
        raise FileNotFoundError(f"Experiment root does not exist: {root}")

    copied, missing = export_bundle(root, args.output_dir, args.transfer_fold, args.strict)
    print(f"\nExported {copied} artifact(s) to {args.output_dir}")
    if missing:
        print(f"Skipped {missing} missing optional artifact(s).")
    print(f"Copy the contents of {args.output_dir} into .paper/ before uploading the bundle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

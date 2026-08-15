r"""Render publication-quality CGFA calibration plots.

Reads ``cgfa_calibration.csv`` (produced by
:class:`mtg.agents.reinforcement_learning.cgfa.CGFACalibrationCallback`)
and emits a multi-panel PNG showing how CGFA-PPO's per-factor structure
evolves over training:

1. **Per-factor calibration (Pearson correlation)**:
   ``cgfa/factor_corr/<name>`` over training steps.  Tracks how well the
   learned per-factor advantage A_k aligns with the SCM-predicted
   intervention eps_k.  A well-calibrated CGFA agent has all curves
   trending towards +1.

2. **Per-factor credit shares**:
   ``cgfa/factor_share/<name>`` over training steps.  Stacked area showing
   which factors are driving the policy gradient at each phase of
   training.  Useful for the "why did the agent do that?" narrative.

3. **Residual gate activity**:
   ``cgfa/gate/mean`` (with min/max envelope) over training steps.
   Reveals whether the agent is mixing in the scalar advantage (gate
   close to 0) or relying on the structured advantage (gate close to 1).

Usage::

    uv run python -m scripts.research.calibration_plot \
        results/research/<exp>/<run>/logs/cgfa/cgfa_calibration.csv \
        --output figures/calibration.png \
        --player-deck mono_red_aggro

Multiple CSVs can be passed; they are overlaid on the same axes for
seed/variant comparisons.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Distinguishable colourblind-friendly palette (Tableau-10 inspired).
_PALETTE = [
    "#4E79A7",  # blue
    "#F28E2B",  # orange
    "#E15759",  # red
    "#76B7B2",  # teal
    "#59A14F",  # green
    "#EDC948",  # yellow
    "#B07AA1",  # purple
]


def _read_csv(path: Path) -> dict[str, list[float]]:
    """Load a calibration CSV into a column-major dict.

    NaNs and missing values are converted to ``float('nan')`` so plotting
    libraries handle them gracefully.
    """
    columns: dict[str, list[float]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k is None:
                    continue
                columns.setdefault(k, [])
                try:
                    columns[k].append(float(v))
                except (TypeError, ValueError):
                    columns[k].append(float("nan"))
    return columns


def _factor_keys(columns: dict[str, list[float]], group: str) -> list[str]:
    """Return all ``cgfa/<group>/<factor>`` keys, sorted by factor name."""
    prefix = f"cgfa/{group}/"
    return sorted(k for k in columns if k.startswith(prefix))


def _split_factor_name(key: str) -> str:
    """Pull the factor name out of ``cgfa/<group>/<factor>``."""
    return key.split("/")[-1]


def _format_deck_name(deck: str) -> str:
    """Format a deck identifier for figure titles."""
    return deck.replace("_", " ").title()


def _plot_panel_calibration(
    ax: plt.Axes,
    runs: dict[str, dict[str, list[float]]],
) -> None:
    """Plot per-factor Pearson correlation over training (one line per factor).

    When multiple runs (CSVs) are supplied they are rendered as
    transparent overlays so the per-seed spread is visible.
    """
    if not runs:
        return
    # Use the first run to discover the factor names.
    first = next(iter(runs.values()))
    factor_keys = _factor_keys(first, "factor_corr")
    if not factor_keys:
        ax.text(0.5, 0.5, "no factor_corr keys found", ha="center", va="center")
        return

    # Pre-compute mean per factor across runs at each step.
    for i, key in enumerate(factor_keys):
        color = _PALETTE[i % len(_PALETTE)]
        for run_label, columns in runs.items():
            steps = columns.get("step", [])
            values = columns.get(key, [])
            if not steps or not values:
                continue
            n = min(len(steps), len(values))
            ax.plot(
                steps[:n],
                values[:n],
                color=color,
                alpha=0.4 if len(runs) > 1 else 0.95,
                label=_split_factor_name(key) if run_label == next(iter(runs)) else None,
            )
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"Per-factor Pearson($A_k$, $\hat{\epsilon}_k$)")
    ax.set_title("Intervention calibration over training", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(alpha=0.25)


def _plot_panel_credit_share(
    ax: plt.Axes,
    runs: dict[str, dict[str, list[float]]],
) -> None:
    """Stacked-area plot of per-factor credit shares over training.

    Aggregates across runs by averaging at each step.
    """
    if not runs:
        return
    first = next(iter(runs.values()))
    keys = _factor_keys(first, "factor_share")
    if not keys:
        ax.text(0.5, 0.5, "no factor_share keys found", ha="center", va="center")
        return

    # Build a step grid by intersecting the runs (simplest: use the
    # first run's step column; assumes all runs share the same logging
    # cadence, which is true for runs from the same training config).
    steps = first.get("step", [])
    if not steps:
        return
    n = len(steps)
    arrays: list[np.ndarray] = []
    labels: list[str] = []
    for key in keys:
        values_per_run = []
        for columns in runs.values():
            v = columns.get(key, [])
            if len(v) == 0:
                continue
            v = v[:n] + [float("nan")] * max(0, n - len(v))
            values_per_run.append(v)
        if not values_per_run:
            continue
        mean = np.nanmean(np.array(values_per_run, dtype=float), axis=0)
        arrays.append(mean)
        labels.append(_split_factor_name(key))

    if not arrays:
        return
    matrix = np.array(arrays)
    # Replace NaNs with 0 for stacking; stack to 1.
    matrix = np.nan_to_num(matrix, nan=0.0)
    # Normalise per-step in case shares don't sum to exactly 1 (mostly
    # because 'cgfa/factor_share' is normalised after-the-fact).
    totals = matrix.sum(axis=0)
    totals = np.where(totals > 0, totals, 1.0)
    matrix = matrix / totals
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(arrays))]
    ax.stackplot(steps, matrix, labels=labels, colors=colors, alpha=0.85, edgecolor="white")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Per-factor credit share")
    ax.set_ylim(0, 1)
    ax.set_title("Where does the policy gradient come from?", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.2, axis="y")


def _plot_panel_gate(
    ax: plt.Axes,
    runs: dict[str, dict[str, list[float]]],
) -> None:
    """Plot the residual-gate mean (and per-update std envelope when present)."""
    if not runs:
        return
    plotted_any = False
    for label, columns in runs.items():
        steps = columns.get("step", [])
        gate_mean = columns.get("cgfa/gate/mean", [])
        gate_std = columns.get("cgfa/gate/std", [])
        if not steps or not gate_mean:
            continue
        n = min(len(steps), len(gate_mean))
        ax.plot(steps[:n], gate_mean[:n], color="#4E79A7", lw=2, label=label)
        if gate_std and len(gate_std) >= n:
            mean_arr = np.array(gate_mean[:n])
            std_arr = np.array(gate_std[:n])
            ax.fill_between(
                steps[:n],
                np.maximum(0.0, mean_arr - std_arr),
                np.minimum(1.0, mean_arr + std_arr),
                color="#4E79A7",
                alpha=0.2,
            )
        plotted_any = True
    if not plotted_any:
        ax.text(0.5, 0.5, "no cgfa/gate keys found", ha="center", va="center")
        return
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"Residual gate $\bar{g}(s)$")
    ax.set_title("Scalar / factored mixing over training", fontweight="bold")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)


def render(
    csv_paths: list[Path],
    output_path: Path,
    labels: list[str] | None = None,
    player_deck: str | None = None,
) -> Path:
    """Read one or more calibration CSVs and write a multi-panel PNG."""
    if not csv_paths:
        raise ValueError("At least one CSV path must be provided")

    if labels is not None and len(labels) != len(csv_paths):
        raise ValueError("labels must match the number of csv_paths")
    labels = labels or [p.stem for p in csv_paths]

    runs: dict[str, dict[str, list[float]]] = {}
    for label, path in zip(labels, csv_paths, strict=True):
        runs[label] = _read_csv(path)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _plot_panel_calibration(axes[0], runs)
    _plot_panel_credit_share(axes[1], runs)
    _plot_panel_gate(axes[2], runs)

    title = "CGFA-PPO calibration diagnostics"
    if player_deck:
        title = f"{title}: {_format_deck_name(player_deck)} player deck"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        prog="calibration_plot",
        description=(
            "Render the CGFA-PPO intervention-calibration diagnostic plot from "
            "one or more cgfa_calibration.csv files."
        ),
    )
    p.add_argument(
        "csv_paths",
        nargs="+",
        type=Path,
        help="One or more cgfa_calibration.csv files (one per run/seed).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("figures/cgfa_calibration.png"),
        help="Output PNG path.",
    )
    p.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help=(
            "Optional human-readable label per CSV (e.g. 'seed 42', 'cgfa_full'). "
            "Length must match the number of CSVs."
        ),
    )
    p.add_argument(
        "--player-deck",
        default=None,
        help="Optional player deck identifier to include in the figure title.",
    )
    return p.parse_args()


def main() -> int:
    """Entry point for ``python -m scripts.research.calibration_plot``."""
    args = parse_args()
    out = render(
        args.csv_paths,
        args.output,
        labels=args.labels,
        player_deck=args.player_deck,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

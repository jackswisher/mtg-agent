r"""Regenerate training and evaluation plots from a saved run directory.

Auto-detects which JSON the run directory contains and re-renders the
appropriate figures into ``plots/`` without touching the underlying
training or evaluation pipeline:

* ``metrics.json`` (produced by ``mtg-train`` / ``run_training.py``)
  -> ``training_curves.png`` + ``evaluation_results.png``.
* ``results.json``  (produced by ``mtg-eval`` / ``run_evaluation.py``)
  -> ``win_rate_comparison.png`` + ``reward_comparison.png``.

Usage::

    # Training run
    uv run python -m scripts.runner.regenerate_plots \\
        results/trained_agents/ppo_mono_red_aggro_vs_multi_20260416_175612

    # Standalone evaluation run
    uv run python -m scripts.runner.regenerate_plots \\
        results/evaluations/eval_ppo_mono_red_aggro_vs_multi_20260416_181244

Notes:
- The training-run evaluation bar chart is always regenerable (all
  data in ``metrics.json``).
- Training curves require ``training.per_opponent_history`` in
  ``metrics.json``; runs produced before this field was added will
  only regenerate the eval chart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

COLORS = [
    "#2A9D8F",  # Teal
    "#E63946",  # Red
    "#457B9D",  # Blue
    "#E9C46A",  # Yellow
    "#F77F00",  # Orange
    "#9B59B6",  # Purple
    "#1ABC9C",  # Turquoise
]


def _rolling_mean(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    n = len(values)
    w = max(1, min(window, n))
    out: list[float] = []
    for i in range(n):
        lo = max(0, i - w // 2)
        hi = min(n, lo + w)
        lo = max(0, hi - w)
        out.append(float(sum(values[lo:hi]) / max(1, hi - lo)))
    return out


def _plot_with_smoothing(
    ax,
    values: list[float],
    color: str,
    label: str,
    x_offset: int = 0,
) -> None:
    if not values:
        return
    window = max(5, min(51, len(values) // 20))
    smoothed = _rolling_mean(values, window)
    xs = range(x_offset, x_offset + len(values))
    ax.plot(xs, values, color=color, linewidth=0.7, alpha=0.18)
    ax.plot(xs, smoothed, color=color, linewidth=2.0, label=label, alpha=0.95)


def regenerate_training_curves(
    metrics: dict,
    config: dict,
    plots_dir: Path,
) -> Path | None:
    """Regenerate the 4-panel training curves plot from saved history."""
    history_by_opp = metrics.get("training", {}).get("per_opponent_history", {})
    if not history_by_opp:
        print(
            "  [skip] training_curves.png: no 'per_opponent_history' in metrics.json "
            "(run predates history persistence)"
        )
        return None

    agent_type = str(config.get("agent_type", "agent")).upper()
    player_deck = config.get("player_deck", "")
    mode = metrics.get("training", {}).get("training_mode", "")
    suffix = " (Multi-Opponent)" if len(history_by_opp) > 1 else ""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Training: {agent_type} on {player_deck}{suffix}  [{mode}]",
        fontsize=14,
        fontweight="bold",
    )

    for idx, (opp_name, history) in enumerate(history_by_opp.items()):
        if not history:
            continue
        color = COLORS[idx % len(COLORS)]
        label = opp_name.replace("_", " ").title()

        wr_data = [h.get("win_rate", 0) for h in history]
        _plot_with_smoothing(axes[0, 0], wr_data, color, label)

        rw_data = [h.get("avg_reward", 0) for h in history]
        _plot_with_smoothing(axes[0, 1], rw_data, color, label)

        el_data = [h.get("episode_length", 0) for h in history]
        _plot_with_smoothing(axes[1, 0], el_data, color, label)

        loss_data = [h.get("loss", 0) for h in history]
        nonzero_indices = [i for i, v in enumerate(loss_data) if v is not None and v != 0]
        if nonzero_indices:
            start_idx = nonzero_indices[0]
            _plot_with_smoothing(
                axes[1, 1], loss_data[start_idx:], color, label, x_offset=start_idx
            )

    axes[0, 0].set_title("Win Rate (rolling mean; raw faded)")
    axes[0, 0].set_ylabel("Win Rate")
    axes[0, 0].set_ylim(-0.05, 1.05)
    axes[0, 0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc="best", fontsize=8)

    axes[0, 1].set_title("Average Reward (rolling mean; raw faded)")
    axes[0, 1].set_ylabel("Reward")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(loc="best", fontsize=8)

    axes[1, 0].set_title("Episode Length (rolling mean; raw faded)")
    axes[1, 0].set_ylabel("Steps")
    axes[1, 0].set_xlabel("PPO update")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(loc="best", fontsize=8)

    axes[1, 1].set_title("Loss (rolling mean; raw faded)")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].set_xlabel("PPO update")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(loc="best", fontsize=8)

    plt.tight_layout()
    out = plots_dir / "training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def regenerate_evaluation_plot(
    metrics: dict,
    config: dict,
    plots_dir: Path,
) -> Path | None:
    """Regenerate the evaluation bar chart with 95% Wilson CI error bars."""
    eval_results = metrics.get("evaluation", {})
    if not eval_results:
        print("  [skip] evaluation_results.png: no 'evaluation' section in metrics.json")
        return None

    agent_type = str(config.get("agent_type", "agent")).upper()
    opponents = list(eval_results.keys())
    win_rates = [eval_results[o].get("win_rate", 0) for o in opponents]
    cis = [eval_results[o].get("win_rate_ci95", 0) for o in opponents]
    n_eps_list = [eval_results[o].get("n_episodes") for o in opponents]
    n_eps = next((n for n in n_eps_list if n), config.get("eval_episodes", 100))

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2A9D8F" if wr >= 0.5 else "#E63946" for wr in win_rates]
    bars = ax.bar(
        range(len(opponents)),
        win_rates,
        color=colors,
        edgecolor="white",
        yerr=cis,
        capsize=6,
        ecolor="#333333",
        error_kw={"linewidth": 1.5, "alpha": 0.8},
    )
    for bar, wr, ci in zip(bars, win_rates, cis, strict=False):
        ax.annotate(
            f"{wr:.0%}\n±{ci:.0%}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + ci + 0.02),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(range(len(opponents)))
    ax.set_xticklabels([o.replace("_", " ").title() for o in opponents], rotation=25, ha="right")
    ax.set_ylabel("Win Rate (mean over eval episodes)", fontweight="medium")
    ax.set_title(
        f"Evaluation: {agent_type} vs Opponents (n={n_eps} episodes/opponent, 95% Wilson CI)",
        fontweight="bold",
        fontsize=11,
    )
    ax.set_ylim(0, 1.20)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% baseline")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = plots_dir / "evaluation_results.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def regenerate_evaluation_run_plots(
    run_dir: Path,
    plots_dir: Path,
    agent_label: str | None = None,
) -> list[Path]:
    """Regenerate evaluation plots from a ``results.json`` (eval-run path).

    Reads ``run_dir/results.json`` (produced by ``run_evaluation.py``)
    and re-renders the win-rate + reward bar charts via the same
    pure ``generate_evaluation_plots`` function the live eval pipeline
    uses, so the figures are byte-identical to a fresh run.

    Args:
        run_dir: Directory containing ``results.json``.
        plots_dir: Where to write the regenerated PNGs.
        agent_label: Optional override for ``EvaluationConfig.agent_type``
            (controls the bar labels). Defaults to ``"all"`` which
            produces ``"AGENT vs Opp"`` labels, the typical choice when
            replotting a multi-agent sweep.

    Returns:
        List of regenerated PNG paths (may be empty if results.json is empty).
    """
    from mtg.utils.interactive import EvaluationConfig
    from scripts.runner.run_evaluation import generate_evaluation_plots

    results_path = run_dir / "results.json"
    if not results_path.exists():
        print(f"  [skip] no results.json at {results_path}")
        return []

    with open(results_path) as f:
        payload = json.load(f)

    saved_cfg = payload.get("config", {})
    results = payload.get("results", {})
    if not results:
        print(f"  [skip] {results_path} has no 'results' section")
        return []

    opponent_decks = saved_cfg.get("opponent_decks") or []
    cfg = EvaluationConfig(
        agent_type=agent_label or "all",
        player_deck=saved_cfg.get("player_deck", ""),
        opponent_deck=",".join(opponent_decks) if opponent_decks else "",
        episodes=int(saved_cfg.get("episodes", 100)),
        max_turns=int(saved_cfg.get("max_turns", 10)),
        seeds=list(saved_cfg.get("seeds") or []),
        model_path=saved_cfg.get("model_path"),
    )
    plots_dir.parent.mkdir(parents=True, exist_ok=True)
    return generate_evaluation_plots(plots_dir.parent, results, cfg)


def main() -> int:
    """CLI entry point: regenerate plots from a saved run directory."""
    parser = argparse.ArgumentParser(
        description="Regenerate training/evaluation plots from a saved run directory.",
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help=(
            "Path to a run directory containing either metrics.json "
            "(training run) or results.json (evaluation run)."
        ),
    )
    parser.add_argument(
        "--agent-label",
        default=None,
        help=(
            "Optional override for the EvaluationConfig.agent_type used "
            "during eval-run replots (controls bar labels). Defaults to 'all'."
        ),
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    metrics_path = run_dir / "metrics.json"
    results_path = run_dir / "results.json"
    config_path = run_dir / "config.yaml"

    if not metrics_path.exists() and not results_path.exists():
        print(f"ERROR: neither {metrics_path} nor {results_path} found")
        return 1

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"Regenerating plots for {run_dir}...")

    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        config: dict = {}
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}

        tc = regenerate_training_curves(metrics, config, plots_dir)
        if tc:
            print(f"  Saved: {tc}")
        ev = regenerate_evaluation_plot(metrics, config, plots_dir)
        if ev:
            print(f"  Saved: {ev}")

    if results_path.exists():
        eval_paths = regenerate_evaluation_run_plots(
            run_dir, plots_dir, agent_label=args.agent_label
        )
        for p in eval_paths:
            print(f"  Saved: {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

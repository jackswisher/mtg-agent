r"""Transfer-experiment runner.

The transfer experiment asks the central CRL question for the paper:

    Does CGFA-PPO generalise to opponents it has never trained against
    better than vanilla PPO does?

To answer it we:

1.  Train every (agent, player_deck, seed) on a *training opponent set*
    (default: 3 opponents).  This is exactly what
    :mod:`scripts.research.train_sweep` already does, so we delegate to it.

2.  Evaluate every trained model **twice**:

    * **in-distribution**: against the same opponents the agent was
      trained on.
    * **held-out**: against a disjoint *evaluation opponent set*
      (default: the remaining 2 opponents).

    The two evaluation passes are written to ``eval/`` and ``eval_heldout/``
    sub-directories of the experiment directory so they never collide.

3.  Build a "transfer report" that lives next to the eval folders:

    * ``transfer_report.json`` - per (agent, deck, seed) breakdown plus
      the **generalisation gap** ``mean(in-dist win rate) - mean(held-out
      win rate)`` and a paired-bootstrap test against zero per agent.
    * ``transfer_summary.csv`` - flat, pandas-friendly view (one row per
      agent x deck x seed x opponent, with split labelled).
    * ``transfer_per_opponent.csv`` - one row per held-out opponent x agent
      with mean win rate + 95% CI across (deck, seed).
    * ``figures/transfer_gap.png`` - bar chart of in-dist vs held-out per
      agent (mean +- 95% CI across seeds & decks) and the gap.

CLI usage::

    uv run python -m scripts.research.transfer_sweep \\
        --experiment-name transfer_v1 \\
        --agents ppo cgfa \\
        --player-decks mono_red_aggro \\
        --seeds 42 123 456 \\
        --train-opponents mono_red_aggro azorius_control dimir_midrange \\
        --heldout-opponents domain_ramp boros_convoke \\
        --timesteps-per-opponent 1_000_000 \\
        --eval-episodes 500

A smoke variant is exposed via ``--smoke`` (tiny budget, single seed,
single deck, single train/held-out opponent each) and is what the unit
test exercises end-to-end.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mtg.utils.cli_display import console, print_divider, print_logo  # noqa: E402
from scripts.research.eval_sweep import evaluate_sweep  # noqa: E402
from scripts.research.stats import (  # noqa: E402
    bootstrap_mean_ci,
    paired_bootstrap_test,
)
from scripts.research.train_sweep import (  # noqa: E402
    ALL_AGENTS,
    ALL_DECKS,
    SweepConfig,
    run_sweep,
)

# ---------------------------------------------------------------------------
# Public configuration object
# ---------------------------------------------------------------------------


@dataclass
class TransferConfig:
    """Configuration for a single transfer experiment.

    Attributes:
        experiment_name: Directory name under ``output_root``.
        agents: Which agent types to train (each one trains independently
            against the same opponent set).
        player_decks: Player decks the agent pilots.
        seeds: Random seeds (>=3 recommended for paired-bootstrap stats).
        train_opponents: Opponent decks visible during training.
        heldout_opponents: Disjoint set of opponent decks reserved for
            the held-out evaluation pass.  ``ValueError`` is raised if
            it overlaps ``train_opponents``.
        timesteps_per_opponent: Per-opponent training budget.
        eval_episodes: Episodes per (agent, deck, seed, opponent) cell
            in **each** of the two eval passes.
        n_envs: Parallel envs (int or ``"auto"``).
        agency_mode: ``"auto"|"full"|"curriculum"`` (passed through to the
            trainer and the evaluator).
        reward_type: Reward shaping mode forwarded to the trainer.
        max_turns: Hard turn cap.
        training_mode: ``"round-robin"|"sequential"``.
        output_root: Directory under which ``experiment_name/`` is created.
        force: If True, retrain runs even when a saved model already exists.

    """

    experiment_name: str
    agents: list[str]
    player_decks: list[str]
    seeds: list[int]
    train_opponents: list[str]
    heldout_opponents: list[str]
    timesteps_per_opponent: int
    eval_episodes: int = 500
    n_envs: int | str = "auto"
    agency_mode: str = "auto"
    reward_type: str = "shaped"
    max_turns: int = 20
    training_mode: str = "round-robin"
    output_root: str = "results/research"
    force: bool = False
    agent_kwargs: dict[str, dict[str, tp.Any]] = field(default_factory=dict)

    def experiment_dir(self) -> Path:
        """Absolute path to this experiment's directory."""
        return Path(self.output_root) / self.experiment_name

    def validate(self) -> None:
        """Validate the configuration; raises ``ValueError`` on conflict."""
        if not self.agents:
            raise ValueError("`agents` must be non-empty")
        if not self.player_decks:
            raise ValueError("`player_decks` must be non-empty")
        if not self.seeds:
            raise ValueError("`seeds` must be non-empty")
        if not self.train_opponents:
            raise ValueError("`train_opponents` must be non-empty")
        if not self.heldout_opponents:
            raise ValueError("`heldout_opponents` must be non-empty")
        overlap = set(self.train_opponents) & set(self.heldout_opponents)
        if overlap:
            raise ValueError(
                f"Train and held-out opponent sets overlap: {sorted(overlap)} appear in both."
            )


# ---------------------------------------------------------------------------
# Stage 1: training (delegate to train_sweep)
# ---------------------------------------------------------------------------


def _train_phase(cfg: TransferConfig) -> Path:
    """Run the multi-seed training sweep on ``train_opponents``."""
    sweep_cfg = SweepConfig(
        experiment_name=cfg.experiment_name,
        agents=cfg.agents,
        player_decks=cfg.player_decks,
        seeds=cfg.seeds,
        opponents=list(cfg.train_opponents),
        timesteps_per_opponent=cfg.timesteps_per_opponent,
        n_envs=cfg.n_envs,
        agency_mode=cfg.agency_mode,
        reward_type=cfg.reward_type,
        max_turns=cfg.max_turns,
        training_mode=cfg.training_mode,
        output_root=cfg.output_root,
        sample_games=0,
        eval_episodes=min(cfg.eval_episodes, 50),
        agent_kwargs_by_agent={agent: dict(kwargs) for agent, kwargs in cfg.agent_kwargs.items()},
    )
    return run_sweep(sweep_cfg, force=cfg.force)


# ---------------------------------------------------------------------------
# Stage 2: evaluate twice (in-distribution + held-out)
# ---------------------------------------------------------------------------


def _eval_both_splits(cfg: TransferConfig) -> tuple[Path, Path]:
    """Run the in-distribution + held-out evaluations.

    Returns:
        ``(in_dist_results_path, heldout_results_path)``.

    """
    experiment_dir = cfg.experiment_dir()
    print_divider("Transfer eval: in-distribution split")
    in_dist_path = evaluate_sweep(
        experiment_dir=experiment_dir,
        n_episodes=cfg.eval_episodes,
        include_baselines=False,
        baseline_overrides=None,
        extra_player_decks=None,
        max_turns=cfg.max_turns,
        agency_mode=cfg.agency_mode,
        opponents_override=list(cfg.train_opponents),
        eval_subdir="eval",
    )
    print_divider("Transfer eval: held-out split")
    heldout_path = evaluate_sweep(
        experiment_dir=experiment_dir,
        n_episodes=cfg.eval_episodes,
        include_baselines=False,
        baseline_overrides=None,
        extra_player_decks=None,
        max_turns=cfg.max_turns,
        agency_mode=cfg.agency_mode,
        opponents_override=list(cfg.heldout_opponents),
        eval_subdir="eval_heldout",
    )
    return in_dist_path, heldout_path


# ---------------------------------------------------------------------------
# Stage 3: build the transfer report
# ---------------------------------------------------------------------------


def _index_eval_results(
    eval_results: dict[str, tp.Any],
) -> dict[tuple[str, str, int, str], dict[str, float]]:
    """Index a loaded ``eval_results.json`` by ``(agent, deck, seed, opp)``.

    The "trained" block is the only one we care about; baselines are
    skipped since the transfer experiment never runs them.
    """
    out: dict[tuple[str, str, int, str], dict[str, float]] = {}
    for entry in eval_results.get("trained", []):
        agent = entry["agent"]
        deck = entry["player_deck"]
        seed = int(entry["seed"])
        for opp, summary in entry["per_opponent"].items():
            out[(agent, deck, seed, opp)] = {
                "win_rate": float(summary["win_rate"]),
                "win_rate_ci_lo": float(summary["win_rate_ci_lo"]),
                "win_rate_ci_hi": float(summary["win_rate_ci_hi"]),
                "draw_rate": float(summary["draw_rate"]),
                "avg_reward": float(summary["avg_reward"]),
                "n": int(summary["n"]),
            }
    return out


def _per_seed_split_means(
    indexed: dict[tuple[str, str, int, str], dict[str, float]],
    agent: str,
) -> dict[tuple[str, int], float]:
    """Per-(deck, seed) mean win rate across opponents for a given agent.

    This is the natural unit for a paired-bootstrap test between splits:
    the same (deck, seed) is paired across the in-dist and held-out passes.
    """
    bucket: dict[tuple[str, int], list[float]] = {}
    for (a, deck, seed, _opp), s in indexed.items():
        if a != agent:
            continue
        bucket.setdefault((deck, seed), []).append(s["win_rate"])
    return {k: float(np.mean(v)) for k, v in bucket.items() if v}


def _per_opponent_summary(
    indexed: dict[tuple[str, str, int, str], dict[str, float]],
    opponents: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate held-out cells by opponent and agent.

    Output shape: ``{opponent: {agent: {mean, ci_lo, ci_hi, n}}}``.
    """
    out: dict[str, dict[str, dict[str, float]]] = {}
    agents = sorted({k[0] for k in indexed})
    for opp in opponents:
        out[opp] = {}
        for agent in agents:
            vals = [
                s["win_rate"]
                for (a, _deck, _seed, o), s in indexed.items()
                if a == agent and o == opp
            ]
            if not vals:
                continue
            ci = bootstrap_mean_ci(np.asarray(vals, dtype=float))
            out[opp][agent] = {
                "mean": float(np.mean(vals)),
                "ci_lo": ci.lo,
                "ci_hi": ci.hi,
                "n": len(vals),
            }
    return out


def build_transfer_report(
    cfg: TransferConfig,
    in_dist_path: Path,
    heldout_path: Path,
) -> dict[str, tp.Any]:
    """Compute the transfer report from two ``eval_results.json`` files.

    The report is a JSON-serialisable dict with three top-level keys:

    * ``"per_agent"``: generalisation gap and paired-bootstrap CI per
      agent, computed by pairing each (deck, seed) in-dist mean with its
      held-out counterpart.
    * ``"per_opponent_heldout"``: per held-out opponent x agent mean
      win rate plus 95% bootstrap CI across (deck, seed).
    * ``"long"``: one entry per (agent, deck, seed, opponent, split)
      cell suitable for round-tripping into a CSV.
    """
    with open(in_dist_path) as f:
        in_dist_data = json.load(f)
    with open(heldout_path) as f:
        heldout_data = json.load(f)

    in_dist = _index_eval_results(in_dist_data)
    heldout = _index_eval_results(heldout_data)

    agents = sorted({k[0] for k in (set(in_dist) | set(heldout))})

    per_agent: dict[str, dict[str, tp.Any]] = {}
    for agent in agents:
        a_in = _per_seed_split_means(in_dist, agent)
        a_out = _per_seed_split_means(heldout, agent)
        common = sorted(set(a_in) & set(a_out))
        if not common:
            per_agent[agent] = {
                "n_pairs": 0,
                "in_dist_mean": float("nan"),
                "heldout_mean": float("nan"),
                "gap_mean": float("nan"),
                "gap_ci_lo": float("nan"),
                "gap_ci_hi": float("nan"),
                "p_value": float("nan"),
            }
            continue
        in_arr = np.array([a_in[k] for k in common], dtype=float)
        out_arr = np.array([a_out[k] for k in common], dtype=float)
        test = paired_bootstrap_test(in_arr, out_arr)
        per_agent[agent] = {
            "n_pairs": int(in_arr.size),
            "in_dist_mean": float(in_arr.mean()),
            "heldout_mean": float(out_arr.mean()),
            "gap_mean": float(test.mean_diff),
            "gap_ci_lo": float(test.ci_low),
            "gap_ci_hi": float(test.ci_high),
            "p_value": float(test.p_value),
            "pairs": [
                {"deck": d, "seed": s, "in_dist": float(i), "heldout": float(o)}
                for (d, s), i, o in zip(common, in_arr, out_arr, strict=False)
            ],
        }

    per_opponent_heldout = _per_opponent_summary(heldout, list(cfg.heldout_opponents))

    long_rows: list[dict[str, tp.Any]] = []
    for split_label, idx in (("in_dist", in_dist), ("heldout", heldout)):
        for (agent, deck, seed, opp), s in sorted(idx.items()):
            long_rows.append(
                {
                    "agent": agent,
                    "player_deck": deck,
                    "seed": seed,
                    "opponent": opp,
                    "split": split_label,
                    "win_rate": s["win_rate"],
                    "win_rate_ci_lo": s["win_rate_ci_lo"],
                    "win_rate_ci_hi": s["win_rate_ci_hi"],
                    "draw_rate": s["draw_rate"],
                    "avg_reward": s["avg_reward"],
                    "n": s["n"],
                }
            )

    return {
        "experiment_name": cfg.experiment_name,
        "train_opponents": list(cfg.train_opponents),
        "heldout_opponents": list(cfg.heldout_opponents),
        "agents": agents,
        "per_agent": per_agent,
        "per_opponent_heldout": per_opponent_heldout,
        "long": long_rows,
    }


def write_transfer_artifacts(
    cfg: TransferConfig,
    report: dict[str, tp.Any],
) -> dict[str, Path]:
    """Persist the transfer report to disk and render its figure.

    Returns a mapping of artefact name -> path so the runner can
    surface them in the final summary.
    """
    transfer_dir = cfg.experiment_dir() / "transfer"
    transfer_dir.mkdir(parents=True, exist_ok=True)

    json_path = transfer_dir / "transfer_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    summary_csv = transfer_dir / "transfer_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "agent",
                "player_deck",
                "seed",
                "opponent",
                "split",
                "win_rate",
                "win_rate_ci_lo",
                "win_rate_ci_hi",
                "draw_rate",
                "avg_reward",
                "n",
            ],
        )
        writer.writeheader()
        writer.writerows(report["long"])

    per_opp_csv = transfer_dir / "transfer_per_opponent.csv"
    with open(per_opp_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["opponent", "agent", "mean_win_rate", "ci_lo", "ci_hi", "n_pairs"])
        for opp, by_agent in report["per_opponent_heldout"].items():
            for agent, s in by_agent.items():
                writer.writerow([opp, agent, s["mean"], s["ci_lo"], s["ci_hi"], s["n"]])

    fig_path = transfer_dir / "figures" / "transfer_gap.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    render_transfer_figure(report, fig_path)

    return {
        "json": json_path,
        "summary_csv": summary_csv,
        "per_opponent_csv": per_opp_csv,
        "figure": fig_path,
    }


def render_transfer_figure(report: dict[str, tp.Any], out_path: Path) -> Path:
    """Render the in-dist vs held-out bar chart to ``out_path``.

    Each agent gets two bars (in-distribution and held-out), with a
    thin annotation line showing the generalisation gap and its 95% CI.
    The figure is intentionally compact (a single axes) so it can drop
    straight into a paper as a one-column figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_agent = report.get("per_agent", {})
    agents = sorted(per_agent.keys())
    if not agents:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(
            0.5,
            0.5,
            "no transfer pairs",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    in_means = [per_agent[a]["in_dist_mean"] for a in agents]
    out_means = [per_agent[a]["heldout_mean"] for a in agents]
    gaps = [per_agent[a]["gap_mean"] for a in agents]
    gap_lo = [per_agent[a]["gap_ci_lo"] for a in agents]
    gap_hi = [per_agent[a]["gap_ci_hi"] for a in agents]

    x = np.arange(len(agents))
    width = 0.35
    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(max(6, 1.8 * len(agents) + 2), 6),
        gridspec_kw={"height_ratios": [3, 1.5]},
        sharex=True,
    )
    ax_top.bar(
        x - width / 2,
        in_means,
        width=width,
        label="in-distribution",
        color="#2A9D8F",
        edgecolor="white",
    )
    ax_top.bar(
        x + width / 2,
        out_means,
        width=width,
        label="held-out",
        color="#E63946",
        edgecolor="white",
    )
    ax_top.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax_top.set_ylim(0.0, 1.05)
    ax_top.set_ylabel("Mean win rate (over deck x seed)")
    ax_top.set_title("Transfer to held-out opponents")
    ax_top.legend(loc="upper right", fontsize=9)
    for xi, (im, om) in enumerate(zip(in_means, out_means, strict=False)):
        ax_top.annotate(
            f"{im:.0%}",
            xy=(xi - width / 2, im + 0.02),
            ha="center",
            fontsize=8,
        )
        ax_top.annotate(
            f"{om:.0%}",
            xy=(xi + width / 2, om + 0.02),
            ha="center",
            fontsize=8,
        )

    err_lo = [g - lo for g, lo in zip(gaps, gap_lo, strict=False)]
    err_hi = [hi - g for g, hi in zip(gaps, gap_hi, strict=False)]
    ax_bot.bar(
        x,
        gaps,
        width=0.5,
        color=["#457B9D" if g >= 0 else "#E76F51" for g in gaps],
        edgecolor="white",
        yerr=[err_lo, err_hi],
        capsize=4,
        ecolor="#333",
    )
    ax_bot.axhline(0.0, color="black", linewidth=1)
    ax_bot.set_ylabel("Gap = in - held-out")
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(agents)
    for xi, g in enumerate(gaps):
        ax_bot.annotate(
            f"{g:+.1%}",
            xy=(xi, g),
            xytext=(0, 6 if g >= 0 else -12),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def run_transfer(cfg: TransferConfig) -> dict[str, Path]:
    """Run the full transfer pipeline (train -> 2-pass eval -> report).

    Returns:
        Mapping ``{"json", "summary_csv", "per_opponent_csv", "figure"}`` to
        absolute paths.

    """
    cfg.validate()
    print_logo()
    print_divider(f"Transfer Experiment: {cfg.experiment_name}")
    console.print(
        f"train opponents:   [cyan]{', '.join(cfg.train_opponents)}[/]\n"
        f"held-out opponents:[red]{', '.join(cfg.heldout_opponents)}[/]\n"
        f"agents:            [cyan]{', '.join(cfg.agents)}[/]   "
        f"decks: [cyan]{', '.join(cfg.player_decks)}[/]   "
        f"seeds: [cyan]{cfg.seeds}[/]"
    )

    overall_started = time.time()
    _train_phase(cfg)
    in_dist_path, heldout_path = _eval_both_splits(cfg)
    report = build_transfer_report(cfg, in_dist_path, heldout_path)
    artifacts = write_transfer_artifacts(cfg, report)

    print_divider("Transfer Complete")
    elapsed = time.time() - overall_started
    console.print(f"elapsed: [cyan]{elapsed:.1f}s[/]")
    for agent, stats in report["per_agent"].items():
        gap = stats.get("gap_mean", float("nan"))
        ci_lo = stats.get("gap_ci_lo", float("nan"))
        ci_hi = stats.get("gap_ci_hi", float("nan"))
        p = stats.get("p_value", float("nan"))
        console.print(
            f"  [bold]{agent:<8}[/] in={stats['in_dist_mean']:.1%}  "
            f"held-out={stats['heldout_mean']:.1%}  "
            f"gap={gap:+.1%}  "
            f"95% CI=[{ci_lo:+.1%}, {ci_hi:+.1%}]  p={p:.3f}"
        )
    for name, path in artifacts.items():
        console.print(f"  {name:<18} [bold]{path}[/]")
    return artifacts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        prog="transfer_sweep",
        description="Train on K opponents, evaluate on a disjoint held-out set.",
    )
    p.add_argument(
        "--experiment-name",
        default=None,
        help="Sub-directory under --output-root.  Defaults to a timestamp.",
    )
    p.add_argument(
        "--agents",
        nargs="+",
        default=["ppo", "cgfa"],
        choices=ALL_AGENTS,
    )
    p.add_argument(
        "--player-decks",
        nargs="+",
        default=["mono_red_aggro"],
        choices=ALL_DECKS,
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument(
        "--train-opponents",
        nargs="+",
        default=["mono_red_aggro", "azorius_control", "dimir_midrange"],
        choices=ALL_DECKS,
    )
    p.add_argument(
        "--heldout-opponents",
        nargs="+",
        default=["domain_ramp", "boros_convoke"],
        choices=ALL_DECKS,
    )
    p.add_argument("--timesteps-per-opponent", type=int, default=1_000_000)
    p.add_argument("--eval-episodes", type=int, default=500)
    p.add_argument("--n-envs", default="auto")
    p.add_argument("--agency", choices=["auto", "full", "curriculum"], default="auto")
    p.add_argument("--reward-type", choices=["sparse", "shaped", "dense"], default="shaped")
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument("--training-mode", choices=["round-robin", "sequential"], default="round-robin")
    p.add_argument("--output-root", default="results/research")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny end-to-end smoke run (single seed, ~10K steps, 5 eval eps).",
    )
    p.add_argument(
        "--from-report",
        type=Path,
        default=None,
        help=(
            "Skip training and evaluation and re-render the transfer-gap "
            "figure from an existing transfer_report.json. All other CLI "
            "args are ignored when this is set."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PNG path for --from-report mode. "
            "Defaults to <report-dir>/figures/transfer_gap.png."
        ),
    )
    return p.parse_args()


def _smoke_config() -> TransferConfig:
    from datetime import datetime

    return TransferConfig(
        experiment_name=f"transfer_smoke_{datetime.now().strftime('%H%M%S')}",
        agents=["ppo"],
        player_decks=["mono_red_aggro"],
        seeds=[42],
        train_opponents=["azorius_control"],
        heldout_opponents=["dimir_midrange"],
        timesteps_per_opponent=5_000,
        eval_episodes=5,
        n_envs=1,
        agency_mode="auto",
        reward_type="shaped",
        max_turns=20,
        training_mode="round-robin",
        force=True,
    )


def replot_transfer_from_report(
    report_path: Path,
    out_path: Path | None = None,
) -> Path:
    """Re-render the transfer-gap figure from a saved ``transfer_report.json``.

    Pure load and render: no training, no evaluation, no env
    construction. Use this whenever you want to restyle the figure
    without re-running the whole pipeline.

    Args:
        report_path: Path to an existing ``transfer_report.json``.
        out_path: Output PNG path. Defaults to
            ``<report-dir>/figures/transfer_gap.png`` (the canonical
            location next to the source report).

    Returns:
        Absolute path to the rendered PNG.
    """
    if not report_path.exists():
        raise FileNotFoundError(f"transfer_report not found: {report_path}")
    report = json.loads(report_path.read_text())
    if out_path is None:
        out_path = report_path.parent / "figures" / "transfer_gap.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return render_transfer_figure(report, out_path)


def main() -> int:
    """CLI entry point."""
    args = parse_args()

    # --- Replot-from-disk path: skip the entire pipeline. -------------
    if args.from_report is not None:
        out = replot_transfer_from_report(args.from_report, args.output)
        console.print(f"[bold]wrote[/] {out}  (replot from {args.from_report})")
        return 0

    if args.smoke:
        cfg = _smoke_config()
    else:
        from datetime import datetime

        cfg = TransferConfig(
            experiment_name=(
                args.experiment_name or f"transfer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ),
            agents=args.agents,
            player_decks=args.player_decks,
            seeds=args.seeds,
            train_opponents=args.train_opponents,
            heldout_opponents=args.heldout_opponents,
            timesteps_per_opponent=args.timesteps_per_opponent,
            eval_episodes=args.eval_episodes,
            n_envs=int(args.n_envs) if str(args.n_envs).isdigit() else args.n_envs,
            agency_mode=args.agency,
            reward_type=args.reward_type,
            max_turns=args.max_turns,
            training_mode=args.training_mode,
            output_root=args.output_root,
            force=args.force,
        )
    run_transfer(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "TransferConfig",
    "build_transfer_report",
    "render_transfer_figure",
    "replot_transfer_from_report",
    "run_transfer",
    "write_transfer_artifacts",
]

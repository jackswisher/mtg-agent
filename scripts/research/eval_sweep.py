r"""Multi-seed evaluation sweep with optional baseline pairing.

Reads a sweep manifest produced by :mod:`scripts.research.train_sweep`, loads
each trained model, and evaluates it against every opponent over many episodes.

By default, each player deck is paired with two strategically appropriate
baselines:

* ``random``      - uniform random over legal actions (sanity floor)
* ``<heuristic>`` - the deck-matched heuristic from
  :data:`mtg.agents.DECK_TO_HEURISTIC` (e.g. ``control`` for
  ``azorius_control``, ``midrange`` for ``dimir_midrange``)

This avoids the nonsensical comparison of, say, running ``greedy_aggro``
piloting an Azorius Control deck. Use ``--baseline-agents`` to override the
auto-mapping with a fixed list applied to every deck (advanced), or
``--no-baselines`` to skip baselines entirely.

Per-episode outcomes (win/loss/draw + reward + length) are written to disk
for downstream statistical analysis (paired bootstrap, Wilcoxon, etc.).

Outputs (under ``<experiment_dir>/eval/``):

- ``eval_results.json``       Aggregated per (agent, deck, seed, opponent) summary
- ``eval_episodes.csv``       One row per episode (wide-format, suitable for pandas)
- ``eval_summary.csv``        One row per (agent, deck, seed, opponent) with mean/CI
- ``eval_manifest.yaml``      What was evaluated, with timing + config

Usage:
    uv run python -m scripts.research.eval_sweep \\
        results/research/ppo_baseline_v1 \\
        --eval-episodes 500

Smoke test:
    uv run python -m scripts.research.eval_sweep \\
        results/research/smoke_test --eval-episodes 5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import typing as tp
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mtg.agents import heuristic_for_deck
from mtg.utils.cli_display import console, print_divider, print_logo
from mtg.utils.interactive import format_duration
from scripts.research.stats import bootstrap_mean_ci, wilson_ci


def _resolve_baselines_for_deck(
    deck: str,
    *,
    include_baselines: bool,
    baseline_overrides: list[str] | None,
) -> list[str]:
    """Decide which baseline agents should be evaluated on a given player deck.

    Args:
        deck: Player deck archetype name.
        include_baselines: Master switch. If False, returns ``[]``.
        baseline_overrides: If provided, this list is used verbatim for every
            deck (advanced override). If ``None``, the function auto-pairs
            ``random`` plus the canonical heuristic for the deck.

    Returns:
        Ordered, de-duplicated list of baseline agent names.

    """
    if not include_baselines:
        return []
    if baseline_overrides is not None:
        return list(dict.fromkeys(baseline_overrides))
    auto = ["random"]
    matched = heuristic_for_deck(deck)
    if matched is not None:
        auto.append(matched)
    return list(dict.fromkeys(auto))


def _evaluate_episodes(
    env: tp.Any,
    agent: tp.Any,
    n_episodes: int,
    max_steps: int = 500,
    *,
    description: str | None = None,
) -> list[dict[str, float]]:
    """Run ``n_episodes`` games and return one row of metrics per episode.

    If ``description`` is provided, a live Rich progress bar is rendered with
    running W/L/D counts and current win rate. Pass ``None`` to suppress the
    bar (useful for unit tests / batch mode).
    """
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeRemainingColumn,
    )

    rows: list[dict[str, float]] = []
    wins = losses = draws = 0

    if description is None:
        for ep in range(n_episodes):
            obs, info = env.reset()
            done = False
            ep_reward = 0.0
            ep_len = 0
            while not done and ep_len < max_steps:
                action_mask = info.get("action_mask", np.ones(env.action_space.n, dtype=bool))
                action = agent.select_action(obs, action_mask, info)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                ep_len += 1
                done = bool(terminated or truncated)
            result = info.get("game_result", "")
            won = 1 if result == "win" else 0
            drew = 1 if result == "draw" else 0
            rows.append(
                {"episode": ep, "win": won, "draw": drew, "reward": ep_reward, "length": ep_len}
            )
        return rows

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30, complete_style="green"),
        TaskProgressColumn(),
        TextColumn("•"),
        TextColumn("[green]{task.fields[stats]}[/green]"),
        TimeRemainingColumn(),
        console=console,
        transient=True,  # disappears when complete; final summary line follows
        refresh_per_second=8,
    ) as progress:
        task = progress.add_task(
            description,
            total=n_episodes,
            stats="W=0 L=0 D=0  WR=0.0%",
        )
        for ep in range(n_episodes):
            obs, info = env.reset()
            done = False
            ep_reward = 0.0
            ep_len = 0
            while not done and ep_len < max_steps:
                action_mask = info.get("action_mask", np.ones(env.action_space.n, dtype=bool))
                action = agent.select_action(obs, action_mask, info)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                ep_len += 1
                done = bool(terminated or truncated)
            result = info.get("game_result", "")
            won = 1 if result == "win" else 0
            drew = 1 if result == "draw" else 0
            if won:
                wins += 1
            elif drew:
                draws += 1
            else:
                losses += 1
            rows.append(
                {"episode": ep, "win": won, "draw": drew, "reward": ep_reward, "length": ep_len}
            )
            wr = wins / (ep + 1)
            progress.update(
                task,
                advance=1,
                stats=f"W={wins} L={losses} D={draws}  WR={wr:5.1%}",
            )
    return rows


def _load_trained_agent(
    agent_type: str,
    obs_dim: int,
    act_dim: int,
    model_path: Path,
    seed: int,
) -> tp.Any:
    """Instantiate and load a trained PPO/Causal agent."""
    from mtg.agents import get_agent

    agent = get_agent(
        agent_type,
        observation_dim=obs_dim,
        action_dim=act_dim,
        seed=seed,
    )
    agent.load(str(model_path))
    return agent


def _build_baseline_agent(name: str, seed: int) -> tp.Any:
    """Instantiate a baseline (random/heuristic) agent with no training."""
    from mtg.agents import get_agent

    if name == "greedy_aggro":
        return get_agent(name, aggression=0.7, seed=seed)
    return get_agent(name, seed=seed)


def _create_eval_env(
    player_deck: str,
    opponent_deck: str,
    seed: int,
    max_turns: int,
    auto_combat: bool,
    auto_target: bool,
) -> tp.Any:
    from scripts.runner.run_training import create_env

    return create_env(
        player_deck=player_deck,
        opponent_deck=opponent_deck,
        reward_type="sparse",  # eval should use sparse reward (game result only)
        seed=seed,
        max_turns=max_turns,
        max_steps_per_episode=500,
        auto_combat=auto_combat,
        auto_target=auto_target,
    )


def _summarise(rows: list[dict[str, float]]) -> dict[str, float | int]:
    """Reduce per-episode rows to a single summary dict."""
    if not rows:
        return {
            "n": 0,
            "win_rate": 0.0,
            "win_rate_ci_lo": 0.0,
            "win_rate_ci_hi": 0.0,
            "draw_rate": 0.0,
            "avg_reward": 0.0,
            "reward_ci_lo": 0.0,
            "reward_ci_hi": 0.0,
            "avg_length": 0.0,
        }
    wins = int(sum(r["win"] for r in rows))
    draws = int(sum(r["draw"] for r in rows))
    rewards = np.array([r["reward"] for r in rows], dtype=float)
    lengths = np.array([r["length"] for r in rows], dtype=float)
    n = len(rows)
    wci = wilson_ci(wins, n)
    rci = bootstrap_mean_ci(rewards, n_resamples=2_000)
    return {
        "n": n,
        "win_rate": wins / n,
        "win_rate_ci_lo": wci.lo,
        "win_rate_ci_hi": wci.hi,
        "draw_rate": draws / n,
        "avg_reward": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "reward_ci_lo": rci.lo,
        "reward_ci_hi": rci.hi,
        "avg_length": float(lengths.mean()),
        "length_std": float(lengths.std()),
    }


def _evaluate_run(
    *,
    agent_type: str,
    player_deck: str,
    seed: int,
    model_path: Path | None,
    opponents: list[str],
    n_episodes: int,
    max_turns: int,
    auto_combat: bool,
    auto_target: bool,
    show_progress: bool = True,
) -> tuple[dict[str, dict[str, tp.Any]], list[dict[str, tp.Any]]]:
    """Evaluate a single (agent, deck, seed) on every opponent."""
    summary: dict[str, dict[str, tp.Any]] = {}
    long_rows: list[dict[str, tp.Any]] = []
    for opp in opponents:
        env = _create_eval_env(
            player_deck=player_deck,
            opponent_deck=opp,
            seed=seed + 7919,  # stable eval seed offset
            max_turns=max_turns,
            auto_combat=auto_combat,
            auto_target=auto_target,
        )
        if model_path is None:
            agent = _build_baseline_agent(agent_type, seed=seed)
        else:
            agent = _load_trained_agent(
                agent_type=agent_type,
                obs_dim=env.observation_space.shape[0],
                act_dim=env.action_space.n,
                model_path=model_path,
                seed=seed,
            )
        desc = f"vs {opp:<18}" if show_progress else None
        rows = _evaluate_episodes(env, agent, n_episodes=n_episodes, description=desc)
        summary[opp] = _summarise(rows)
        for r in rows:
            long_rows.append(
                {
                    "agent": agent_type,
                    "player_deck": player_deck,
                    "seed": seed,
                    "opponent": opp,
                    **r,
                }
            )
        wr = summary[opp]["win_rate"]
        ci_lo = summary[opp]["win_rate_ci_lo"]
        ci_hi = summary[opp]["win_rate_ci_hi"]
        avg_len = summary[opp]["avg_length"]
        console.print(
            f"      vs {opp:<18} n={n_episodes:>4}  "
            f"WR={wr:6.1%} [{ci_lo:.1%}, {ci_hi:.1%}]  "
            f"avg_len={avg_len:.0f}"
        )
    return summary, long_rows


def _model_path(experiment_dir: Path, run: dict[str, tp.Any]) -> Path | None:
    """Resolve the saved model file for a sweep run."""
    rdir = experiment_dir / run["output_dir"]
    candidate = rdir / f"{run['agent']}_{run['player_deck']}.zip"
    return candidate if candidate.exists() else None


def evaluate_sweep(
    experiment_dir: Path,
    n_episodes: int,
    *,
    include_baselines: bool = True,
    baseline_overrides: list[str] | None = None,
    extra_player_decks: list[str] | None = None,
    max_turns: int = 20,
    agency_mode: str = "auto",
    opponents_override: list[str] | None = None,
    eval_subdir: str = "eval",
) -> Path:
    """Evaluate all completed runs in a sweep and write artifacts.

    Args:
        experiment_dir: Path to the sweep directory containing
            ``sweep_manifest.yaml``.
        n_episodes: Number of evaluation episodes per (agent, deck, seed,
            opponent) cell.
        include_baselines: If True (default), baseline agents are evaluated
            alongside trained models. If False, only trained models are run.
        baseline_overrides: Advanced override. If provided, this exact list of
            agent names is used for every player deck. If ``None`` (default),
            each deck is paired with ``["random", heuristic_for_deck(deck)]``.
        extra_player_decks: Optional extra decks (beyond what was trained) to
            run baselines on. Useful for "what if" analyses.
        max_turns: Hard cap on turns per episode.
        agency_mode: ``"auto"`` (auto-combat + auto-target), ``"full"`` (fine
            agent control of combat/targets), or ``"curriculum"`` (informational
            only at eval time; treated as auto).
        opponents_override: When provided, evaluate against this list of
            opponents instead of the ones recorded in the sweep manifest.
            Used by the transfer experiment runner to score the trained
            agents on held-out matchups.
        eval_subdir: Sub-directory under ``experiment_dir`` to write
            artefacts into.  Defaults to ``"eval"``; transfer runs use
            ``"eval_heldout"`` so in-distribution and held-out evals
            don't collide on disk.

    Returns:
        Path to the written ``eval_results.json``.

    """
    manifest_path = experiment_dir / "sweep_manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Sweep manifest not found: {manifest_path}")
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    cfg_block = manifest.get("config", {})
    if opponents_override is not None:
        opponents = list(opponents_override)
    else:
        opponents = cfg_block.get("opponents", [])
    if not opponents:
        raise ValueError("Manifest has no opponents defined")
    auto_combat = agency_mode in {"auto", "curriculum"}
    auto_target = agency_mode in {"auto", "curriculum"}

    eval_dir = experiment_dir / eval_subdir
    eval_dir.mkdir(parents=True, exist_ok=True)

    aggregated: dict[str, dict[str, tp.Any]] = {"trained": [], "baselines": []}
    all_long_rows: list[dict[str, tp.Any]] = []

    print_logo()
    print_divider("Evaluation Sweep")
    console.print(f"experiment: [bold]{experiment_dir.name}[/bold]")
    console.print(
        f"opponents:  [cyan]{', '.join(opponents)}[/cyan]    "
        f"episodes:   [cyan]{n_episodes}[/cyan]/opponent"
    )

    completed_runs = [
        r
        for r in manifest["runs"]
        if r["status"] in {"completed", "skipped"} and _model_path(experiment_dir, r) is not None
    ]

    # ---- Pre-compute total work for the sweep-level counter ----
    baseline_decks = list(
        dict.fromkeys(
            list({r["player_deck"] for r in manifest["runs"]}) + (extra_player_decks or [])
        )
    )
    baseline_seeds = manifest.get("config", {}).get("seeds", [42])
    baselines_by_deck: dict[str, list[str]] = {
        deck: _resolve_baselines_for_deck(
            deck,
            include_baselines=include_baselines,
            baseline_overrides=baseline_overrides,
        )
        for deck in baseline_decks
    }
    n_trained_evals = sum(1 for r in completed_runs if _model_path(experiment_dir, r) is not None)
    n_baseline_evals = sum(
        len(baselines) * len(baseline_seeds) for baselines in baselines_by_deck.values()
    )
    total_evals = n_trained_evals + n_baseline_evals
    console.print(
        f"plan:       [cyan]{n_trained_evals}[/cyan] trained + "
        f"[cyan]{n_baseline_evals}[/cyan] baseline = "
        f"[bold]{total_evals}[/bold] (agent x deck x seed) evaluations, "
        f"each over [cyan]{len(opponents)}[/cyan] opponents"
    )
    started = time.time()
    eval_idx = 0

    def _format_eta(done: int, total: int, elapsed_so_far: float) -> str:
        if done == 0 or total == 0:
            return "calculating"
        avg = elapsed_so_far / done
        remaining = max(0, total - done) * avg
        return format_duration(remaining)

    # ---- Trained-agent runs ----
    for run in completed_runs:
        mpath = _model_path(experiment_dir, run)
        if mpath is None:
            console.print(
                f"  [yellow]\u26a0 missing model[/yellow] for {run['output_dir']}, skipping"
            )
            continue
        eval_idx += 1
        eta = _format_eta(eval_idx - 1, total_evals, time.time() - started)
        console.print(
            f"\n[bold cyan]\u25b6 [{eval_idx}/{total_evals}][/bold cyan] "
            f"{run['agent']} on {run['player_deck']} seed={run['seed']}  "
            f"[dim](ETA {eta})[/dim]"
        )
        per_opp, long_rows = _evaluate_run(
            agent_type=run["agent"],
            player_deck=run["player_deck"],
            seed=run["seed"],
            model_path=mpath,
            opponents=opponents,
            n_episodes=n_episodes,
            max_turns=max_turns,
            auto_combat=auto_combat,
            auto_target=auto_target,
        )
        aggregated["trained"].append(
            {
                "agent": run["agent"],
                "player_deck": run["player_deck"],
                "seed": run["seed"],
                "per_opponent": per_opp,
            }
        )
        all_long_rows.extend(long_rows)

    # ---- Baseline agents (no training; per-deck auto-pairing by default) ----
    if include_baselines:
        console.print("\n[bold]Baseline agents (auto-paired to each player deck):[/bold]")
        for deck, names in baselines_by_deck.items():
            shown = ", ".join(names) if names else "(none)"
            console.print(f"  - {deck:<18} -> [cyan]{shown}[/cyan]")

    for deck, baselines in baselines_by_deck.items():
        for baseline in baselines:
            for seed in baseline_seeds:
                eval_idx += 1
                eta = _format_eta(eval_idx - 1, total_evals, time.time() - started)
                console.print(
                    f"\n[bold magenta]\u2733 [{eval_idx}/{total_evals}] baseline[/bold magenta] "
                    f"{baseline} on {deck} seed={seed}  [dim](ETA {eta})[/dim]"
                )
                per_opp, long_rows = _evaluate_run(
                    agent_type=baseline,
                    player_deck=deck,
                    seed=seed,
                    model_path=None,
                    opponents=opponents,
                    n_episodes=n_episodes,
                    max_turns=max_turns,
                    auto_combat=auto_combat,
                    auto_target=auto_target,
                )
                aggregated["baselines"].append(
                    {
                        "agent": baseline,
                        "player_deck": deck,
                        "seed": seed,
                        "per_opponent": per_opp,
                    }
                )
                all_long_rows.extend(long_rows)

    elapsed = time.time() - started

    # ---- Write artifacts ----
    results_path = eval_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "experiment": manifest["experiment_name"],
                "evaluated_at": datetime.now().isoformat(timespec="seconds"),
                "n_episodes_per_opponent": n_episodes,
                "max_turns": max_turns,
                "agency_mode": agency_mode,
                "opponents": opponents,
                "include_baselines": include_baselines,
                "baseline_overrides": baseline_overrides,
                "baselines_by_deck": baselines_by_deck,
                "trained": aggregated["trained"],
                "baselines": aggregated["baselines"],
                "elapsed_seconds": elapsed,
            },
            f,
            indent=2,
        )

    long_csv = eval_dir / "eval_episodes.csv"
    with open(long_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "agent",
                "player_deck",
                "seed",
                "opponent",
                "episode",
                "win",
                "draw",
                "reward",
                "length",
            ],
        )
        writer.writeheader()
        writer.writerows(all_long_rows)

    summary_csv = eval_dir / "eval_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "kind",
                "agent",
                "player_deck",
                "seed",
                "opponent",
                "n",
                "win_rate",
                "win_rate_ci_lo",
                "win_rate_ci_hi",
                "draw_rate",
                "avg_reward",
                "reward_std",
                "reward_ci_lo",
                "reward_ci_hi",
                "avg_length",
                "length_std",
            ]
        )
        for kind in ("trained", "baselines"):
            for entry in aggregated[kind]:
                for opp, s in entry["per_opponent"].items():
                    writer.writerow(
                        [
                            kind,
                            entry["agent"],
                            entry["player_deck"],
                            entry["seed"],
                            opp,
                            s["n"],
                            s["win_rate"],
                            s["win_rate_ci_lo"],
                            s["win_rate_ci_hi"],
                            s["draw_rate"],
                            s["avg_reward"],
                            s["reward_std"],
                            s["reward_ci_lo"],
                            s["reward_ci_hi"],
                            s["avg_length"],
                            s["length_std"],
                        ]
                    )

    eval_manifest = eval_dir / "eval_manifest.yaml"
    with open(eval_manifest, "w") as f:
        yaml.safe_dump(
            {
                "experiment": manifest["experiment_name"],
                "evaluated_at": datetime.now().isoformat(timespec="seconds"),
                "n_episodes_per_opponent": n_episodes,
                "include_baselines": include_baselines,
                "baseline_overrides": baseline_overrides,
                "baselines_by_deck": baselines_by_deck,
                "max_turns": max_turns,
                "agency_mode": agency_mode,
                "elapsed_seconds": elapsed,
                "n_trained_runs": len(aggregated["trained"]),
                "n_baseline_runs": len(aggregated["baselines"]),
                "opponents": opponents,
                "eval_subdir": eval_subdir,
                "outputs": {
                    "results_json": str(results_path.relative_to(experiment_dir)),
                    "summary_csv": str(summary_csv.relative_to(experiment_dir)),
                    "episodes_csv": str(long_csv.relative_to(experiment_dir)),
                },
            },
            f,
            sort_keys=False,
        )

    print_divider("Evaluation Complete")
    console.print(
        f"trained_runs={len(aggregated['trained'])}  "
        f"baseline_runs={len(aggregated['baselines'])}  "
        f"total_time={format_duration(elapsed)}"
    )
    console.print(f"results: [bold]{results_path}[/bold]")
    return results_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        prog="eval_sweep",
        description="Evaluate every model in a training sweep, plus baselines.",
    )
    p.add_argument("experiment_dir", type=Path, help="Path to a sweep dir")
    p.add_argument("--eval-episodes", type=int, default=500)
    p.add_argument(
        "--no-baselines",
        action="store_true",
        help="Skip baseline evaluation entirely (only score trained models).",
    )
    p.add_argument(
        "--baseline-agents",
        nargs="*",
        default=None,
        help=(
            "Advanced override: explicit baseline agent names to apply to every "
            "deck. If omitted (default), each deck is paired with `random` plus "
            "its canonical heuristic from mtg.agents.DECK_TO_HEURISTIC."
        ),
    )
    p.add_argument(
        "--extra-player-decks",
        nargs="*",
        default=[],
        help="Additional player decks to evaluate baselines on.",
    )
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument("--agency", choices=["auto", "full", "curriculum"], default="auto")
    return p.parse_args()


def main() -> int:
    """Entry point for ``python -m scripts.research.eval_sweep``."""
    args = parse_args()
    evaluate_sweep(
        experiment_dir=args.experiment_dir,
        n_episodes=args.eval_episodes,
        include_baselines=not args.no_baselines,
        baseline_overrides=args.baseline_agents,
        extra_player_decks=args.extra_player_decks,
        max_turns=args.max_turns,
        agency_mode=args.agency,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

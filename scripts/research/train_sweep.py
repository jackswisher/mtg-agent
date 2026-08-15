r"""Multi-seed, multi-deck, multi-agent training sweep.

Trains every combination of ``(agent x player_deck x seed)`` against the
specified opponent set and writes a ``sweep_manifest.yaml`` for downstream
evaluation and aggregation.

Each individual run is delegated to :func:`scripts.runner.run_training.train_agent`
(so it benefits from the full training stack) but with sweep defaults:

- ``training_mode = "round-robin"``  (multi-opponent in one process)
- ``sample_games = 0``               (skip per-run replays; aggregate later)
- ``eval_episodes = 50``             (quick sanity check; real eval is in eval_sweep)

Sweeps are *resumable*: a run is skipped if a model file with the expected
name already exists in its output directory.

Usage:
    uv run python -m scripts.research.train_sweep \\
        --experiment-name ppo_baseline_v1 \\
        --agents ppo \\
        --player-decks mono_red_aggro azorius_control \\
        --seeds 42 123 456 \\
        --opponents mono_red_aggro azorius_control dimir_midrange domain_ramp boros_convoke \\
        --timesteps-per-opponent 2000000 \\
        --n-envs auto \\
        --agency auto

For a quick smoke test::

    uv run python -m scripts.research.train_sweep --quick
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mtg.utils.cli_display import console, print_divider, print_logo
from mtg.utils.interactive import TrainingConfig, format_duration

# Allowed values (kept in sync with the rest of the codebase).
ALL_AGENTS = ["ppo", "causal", "cgfa", "cgfa_scalar_only"]
ALL_DECKS = [
    "mono_red_aggro",
    "azorius_control",
    "dimir_midrange",
    "domain_ramp",
    "boros_convoke",
]


@dataclass
class SweepRun:
    """A single (agent, deck, seed) entry in a sweep."""

    agent: str
    player_deck: str
    seed: int
    opponents: list[str]
    output_dir: str  # relative to experiment dir
    status: str = "pending"  # pending | completed | skipped | failed
    error: str | None = None
    training_time_seconds: float | None = None
    timesteps_per_opponent: int | None = None
    total_timesteps: int | None = None
    # Per-run agent constructor kwargs.  Used by the ablation runner so
    # the same ``agent`` type can be trained with different hyper-parameters
    # (e.g. CGFA with/without learnable_gate) within one sweep.
    agent_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-serialisable representation."""
        return {
            "agent": self.agent,
            "player_deck": self.player_deck,
            "seed": self.seed,
            "opponents": self.opponents,
            "output_dir": self.output_dir,
            "status": self.status,
            "error": self.error,
            "training_time_seconds": self.training_time_seconds,
            "timesteps_per_opponent": self.timesteps_per_opponent,
            "total_timesteps": self.total_timesteps,
            "agent_kwargs": dict(self.agent_kwargs),
        }


@dataclass
class SweepConfig:
    """Top-level sweep configuration."""

    experiment_name: str
    agents: list[str]
    player_decks: list[str]
    seeds: list[int]
    opponents: list[str]
    timesteps_per_opponent: int
    n_envs: int | str = "auto"
    agency_mode: str = "auto"
    reward_type: str = "shaped"
    max_turns: int = 20
    training_mode: str = "round-robin"
    output_root: str = "results/research"
    sample_games: int = 0
    eval_episodes: int = 50  # quick sanity per run; real eval is in eval_sweep
    # In-line "Quick evaluation" episodes shown right after each training run.
    # Kept tiny in sweeps because eval_sweep is what actually counts for the
    # paper; raise this only if you want richer per-run sanity output.
    quick_eval_episodes: int = 5
    agent_kwargs_by_agent: dict[str, dict[str, Any]] = field(default_factory=dict)
    runs: list[SweepRun] = field(default_factory=list)

    def experiment_dir(self) -> Path:
        """Absolute path to this experiment's directory."""
        return Path(self.output_root) / self.experiment_name


def _resolve_n_envs(n_envs: int | str) -> int:
    if isinstance(n_envs, int):
        return max(1, n_envs)
    if isinstance(n_envs, str) and n_envs.lower() == "auto":
        return max(1, (mp.cpu_count() or 2) - 1)
    try:
        return max(1, int(n_envs))
    except (TypeError, ValueError):
        return 1


def _build_runs(cfg: SweepConfig) -> list[SweepRun]:
    runs: list[SweepRun] = []
    for agent in cfg.agents:
        for deck in cfg.player_decks:
            for seed in cfg.seeds:
                run_dir = f"{agent}__{deck}__seed{seed}"
                runs.append(
                    SweepRun(
                        agent=agent,
                        player_deck=deck,
                        seed=seed,
                        opponents=list(cfg.opponents),
                        output_dir=run_dir,
                        agent_kwargs=dict(cfg.agent_kwargs_by_agent.get(agent, {})),
                    )
                )
    return runs


def _make_training_config(cfg: SweepConfig, run: SweepRun) -> TrainingConfig:
    """Build a TrainingConfig that mirrors the production runner."""
    return TrainingConfig(
        agent_type=run.agent,
        player_deck=run.player_deck,
        opponent_deck=",".join(run.opponents),
        timesteps=cfg.timesteps_per_opponent,
        reward_type=cfg.reward_type,
        seed=run.seed,
        max_turns=cfg.max_turns,
        n_envs=_resolve_n_envs(cfg.n_envs),
        training_mode=cfg.training_mode,
        agency_mode=cfg.agency_mode,
        eval_episodes=cfg.eval_episodes,
        quick_eval_episodes=cfg.quick_eval_episodes,
        sample_games=cfg.sample_games,
        output_dir=str(cfg.experiment_dir()),
        agent_kwargs=dict(run.agent_kwargs),
    )


def _model_path_for(run_dir: Path, agent: str, deck: str) -> Path:
    """Return the expected model path for a completed run."""
    return run_dir / f"{agent}_{deck}.zip"


def _save_manifest(cfg: SweepConfig) -> Path:
    """Write the sweep manifest to disk."""
    manifest_path = cfg.experiment_dir() / "sweep_manifest.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_name": cfg.experiment_name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "agents": cfg.agents,
            "player_decks": cfg.player_decks,
            "seeds": cfg.seeds,
            "opponents": cfg.opponents,
            "timesteps_per_opponent": cfg.timesteps_per_opponent,
            "training_mode": cfg.training_mode,
            "agency_mode": cfg.agency_mode,
            "reward_type": cfg.reward_type,
            "max_turns": cfg.max_turns,
            "n_envs": cfg.n_envs,
            "sample_games": cfg.sample_games,
            "eval_episodes": cfg.eval_episodes,
            "agent_kwargs_by_agent": {
                agent: dict(kwargs) for agent, kwargs in cfg.agent_kwargs_by_agent.items()
            },
        },
        "runs": [r.to_dict() for r in cfg.runs],
    }
    with open(manifest_path, "w") as f:
        yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)
    return manifest_path


def _find_existing_run_dir(cfg: SweepConfig, run: SweepRun) -> Path | None:
    """Find an existing directory for this run, if any.

    Strategy: scan every sub-directory inside the experiment dir and inspect
    its ``config.yaml`` to find a match on (agent_type, player_deck, seed).
    The trainer always writes a config.yaml, so this is reliable across
    timestamped names.
    """
    parent = cfg.experiment_dir()
    if not parent.exists():
        return None
    for sub in sorted(parent.iterdir()):
        if not sub.is_dir():
            continue
        cfg_path = sub / "config.yaml"
        model_path = _model_path_for(sub, run.agent, run.player_deck)
        if not (cfg_path.exists() and model_path.exists()):
            continue
        try:
            with open(cfg_path) as f:
                existing = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            continue
        if (
            existing.get("agent_type") == run.agent
            and existing.get("player_deck") == run.player_deck
            and int(existing.get("seed", -1)) == run.seed
            and dict(existing.get("agent_kwargs") or {}) == dict(run.agent_kwargs)
        ):
            return sub
    return None


def _run_one(cfg: SweepConfig, run: SweepRun, force: bool) -> None:
    """Train a single sweep run, writing artifacts under the experiment dir.

    Updates ``run`` in place with status, timing, and total step counts.
    """
    existing = _find_existing_run_dir(cfg, run)
    if existing is not None and not force:
        run.output_dir = existing.name
        run.status = "skipped"
        n_opps = len(run.opponents)
        run.timesteps_per_opponent = cfg.timesteps_per_opponent
        run.total_timesteps = cfg.timesteps_per_opponent * n_opps
        console.print(f"  [yellow]\u21bb skipped (model exists)[/yellow] {existing}")
        return

    # We override TrainingConfig.get_run_name by passing output_dir as the
    # experiment dir; the trainer will append its own timestamped subdir.
    # We capture the subdir afterwards by reading what was created.
    cfg.experiment_dir().mkdir(parents=True, exist_ok=True)
    pre = {p.name for p in cfg.experiment_dir().iterdir() if p.is_dir()}

    training_config = _make_training_config(cfg, run)
    started = time.time()
    try:
        # Lazy import so the sweep file stays cheap to import.
        from scripts.runner.run_training import train_agent

        _agent, _metrics = train_agent(training_config)
        elapsed = time.time() - started
    except Exception as exc:  # noqa: BLE001 - record failure, keep going
        elapsed = time.time() - started
        run.status = "failed"
        run.error = f"{type(exc).__name__}: {exc}"
        run.training_time_seconds = elapsed
        console.print(f"  [red]\u2717 failed[/red] {run.output_dir}: {run.error}")
        return

    post = {p.name for p in cfg.experiment_dir().iterdir() if p.is_dir()}
    new_dirs = sorted(post - pre)
    actual_dir = new_dirs[-1] if new_dirs else run.output_dir
    run.output_dir = actual_dir
    run.status = "completed"
    run.training_time_seconds = elapsed
    run.timesteps_per_opponent = cfg.timesteps_per_opponent
    run.total_timesteps = cfg.timesteps_per_opponent * len(run.opponents)
    console.print(f"  [green]\u2713 completed[/green] {actual_dir} in {format_duration(elapsed)}")


def run_sweep(cfg: SweepConfig, force: bool = False) -> Path:
    """Execute the sweep, writing the manifest after every run.

    If ``cfg.runs`` is already populated (e.g. by the ablation runner)
    those runs are used as-is.  Otherwise we generate the canonical
    Cartesian product over ``agents x player_decks x seeds``.
    """
    print_logo()
    print_divider("Training Sweep")
    if not cfg.runs:
        cfg.runs = _build_runs(cfg)
    console.print(
        f"[bold]{cfg.experiment_name}[/bold]: "
        f"{len(cfg.agents)} agent(s) x "
        f"{len(cfg.player_decks)} deck(s) x "
        f"{len(cfg.seeds)} seed(s) = "
        f"[cyan]{len(cfg.runs)} runs[/cyan]"
    )
    console.print(f"  output: [dim]{cfg.experiment_dir()}[/dim]")
    manifest_path = _save_manifest(cfg)

    overall_started = time.time()
    for i, run in enumerate(cfg.runs, start=1):
        console.print(
            f"\n[bold cyan]\u25b6 [{i}/{len(cfg.runs)}][/bold cyan] "
            f"{run.agent} on {run.player_deck} seed={run.seed}"
        )
        _run_one(cfg, run, force=force)
        # Persist after every run so partial sweeps are recoverable.
        _save_manifest(cfg)

    total_time = time.time() - overall_started
    print_divider("Sweep Complete")
    n_done = sum(1 for r in cfg.runs if r.status == "completed")
    n_skip = sum(1 for r in cfg.runs if r.status == "skipped")
    n_fail = sum(1 for r in cfg.runs if r.status == "failed")
    console.print(
        f"completed={n_done}  skipped={n_skip}  failed={n_fail}  "
        f"total_time={format_duration(total_time)}"
    )
    console.print(f"manifest: [bold]{manifest_path}[/bold]")
    return manifest_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        prog="train_sweep",
        description="Multi-seed/deck/agent training sweep used for reporting.",
    )
    p.add_argument("--experiment-name", required=False, default=None)
    p.add_argument("--agents", nargs="+", default=["ppo"], choices=ALL_AGENTS)
    p.add_argument("--player-decks", nargs="+", default=["mono_red_aggro"], choices=ALL_DECKS)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--opponents", nargs="+", default=ALL_DECKS, choices=ALL_DECKS)
    p.add_argument("--timesteps-per-opponent", type=int, default=2_000_000)
    p.add_argument("--n-envs", default="auto", help="int or 'auto' (CPU - 1)")
    p.add_argument("--agency", choices=["auto", "full", "curriculum"], default="auto")
    p.add_argument("--reward-type", choices=["sparse", "shaped", "dense"], default="shaped")
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument(
        "--training-mode",
        choices=["round-robin", "sequential"],
        default="round-robin",
    )
    p.add_argument("--output-root", default="results/research")
    p.add_argument("--sample-games", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if a model file already exists.",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Tiny smoke test: 1 agent x 1 deck x 1 seed x 5k steps x 1 opponent.",
    )
    return p.parse_args()


def main() -> int:
    """Entry point for ``python -m scripts.research.train_sweep``."""
    args = parse_args()
    if args.quick:
        cfg = SweepConfig(
            experiment_name=args.experiment_name or "smoke_test",
            agents=["ppo"],
            player_decks=["mono_red_aggro"],
            seeds=[42],
            opponents=["azorius_control"],
            timesteps_per_opponent=5_000,
            n_envs=1,
            agency_mode="auto",
            reward_type="shaped",
            max_turns=20,
            training_mode="round-robin",
            output_root=args.output_root,
            sample_games=0,
            eval_episodes=10,
        )
    else:
        if args.experiment_name is None:
            args.experiment_name = f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cfg = SweepConfig(
            experiment_name=args.experiment_name,
            agents=args.agents,
            player_decks=args.player_decks,
            seeds=args.seeds,
            opponents=args.opponents,
            timesteps_per_opponent=args.timesteps_per_opponent,
            n_envs=args.n_envs,
            agency_mode=args.agency,
            reward_type=args.reward_type,
            max_turns=args.max_turns,
            training_mode=args.training_mode,
            output_root=args.output_root,
            sample_games=args.sample_games,
            eval_episodes=args.eval_episodes,
        )
    run_sweep(cfg, force=args.force)
    return 0


if __name__ == "__main__":
    # Avoid re-importing heavy modules in workers spawned by SB3 VecEnv.
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    raise SystemExit(main())

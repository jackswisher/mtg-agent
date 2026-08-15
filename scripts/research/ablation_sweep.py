r"""6-point CGFA-PPO ablation sweep.

Runs the canonical 6-point ablation over a (player_decks x seeds) grid:

1. ``ppo``              : vanilla MaskablePPO
2. ``causal``           : CausalAgent (causal world model on top of PPO)
3. ``cgfa_scalar_only`` : architecture-matched scalar PPO with CGFA loss
   coefficients pinned to zero (controls for parameter count)
4. ``cgfa_no_gate``     : CGFA with the residual gate frozen at alpha=1
5. ``cgfa_no_cal``      : CGFA with intervention calibration disabled
6. ``cgfa_full``        : full CGFA-PPO (proposed method)

Each variant becomes its own sub-experiment under
``<output_root>/<experiment_name>/<variant_name>/``. The runner
reuses :func:`scripts.research.train_sweep.run_sweep` and
:func:`scripts.research.eval_sweep.evaluate_sweep` so the per-variant
manifests, eval JSONs, and per-episode CSVs are identical in shape to
a standalone sweep; only the agent kwargs differ.

After all variants finish, the runner aggregates them with paired-bootstrap
significance tests against ``--baseline-variant`` (default ``cgfa_full``)
and writes a unified ablation table to
``<output_root>/<experiment_name>/aggregated/``.

Example usage::

    uv run mtg-research ablation \
        --experiment-name cgfa_ablation_v1 \
        --player-decks mono_red_aggro \
        --seeds 42 123 456 \
        --opponents mono_red_aggro azorius_control dimir_midrange \
        --timesteps-per-opponent 1_000_000 \
        --eval-episodes 500
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import typing as tp
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mtg.experiments.ablation import (
    AblationVariant,
    default_cgfa_ablation_variants,
    load_ablation_variants,
    save_ablation_variants,
    stress_cgfa_ablation_variants,
    variants_by_name,
)
from mtg.utils.cli_display import console, print_divider, print_logo
from mtg.utils.interactive import format_duration
from scripts.research.aggregate import aggregate
from scripts.research.eval_sweep import evaluate_sweep
from scripts.research.train_sweep import (
    ALL_DECKS,
    SweepConfig,
    SweepRun,
    run_sweep,
)


def _build_variant_runs(
    variant: AblationVariant,
    *,
    player_decks: list[str],
    seeds: list[int],
    opponents: list[str],
) -> list[SweepRun]:
    """Materialise one ``SweepRun`` per (deck, seed) for this variant.

    The variant's ``agent_kwargs`` are attached to every run so the trainer
    forwards them to the agent constructor.
    """
    runs: list[SweepRun] = []
    for deck in player_decks:
        for seed in seeds:
            runs.append(
                SweepRun(
                    agent=variant.agent_type,
                    player_deck=deck,
                    seed=seed,
                    opponents=list(opponents),
                    output_dir=f"{variant.name}__{deck}__seed{seed}",
                    agent_kwargs=dict(variant.agent_kwargs),
                )
            )
    return runs


def _run_one_variant(
    variant: AblationVariant,
    *,
    base_dir: Path,
    player_decks: list[str],
    seeds: list[int],
    opponents: list[str],
    timesteps_per_opponent: int,
    n_envs: int | str,
    agency_mode: str,
    reward_type: str,
    max_turns: int,
    training_mode: str,
    eval_episodes: int,
    include_baselines: bool,
    baseline_overrides: list[str] | None,
    extra_player_decks: list[str],
    force: bool,
) -> Path:
    """Run train_sweep + evaluate_sweep for a single ablation variant.

    Returns:
        Path to ``eval_results.json`` for this variant.
    """
    print_divider(f"Variant: {variant.name}")
    console.print(f"  description: [dim]{variant.description}[/]")
    if variant.agent_kwargs:
        console.print(f"  agent_kwargs: [cyan]{variant.agent_kwargs}[/]")
    else:
        console.print("  agent_kwargs: [dim](defaults)[/]")

    variant_dir = base_dir / variant.name
    cfg = SweepConfig(
        experiment_name=variant.name,
        agents=[variant.agent_type],  # display only; runs override
        player_decks=list(player_decks),
        seeds=list(seeds),
        opponents=list(opponents),
        timesteps_per_opponent=timesteps_per_opponent,
        n_envs=n_envs,
        agency_mode=agency_mode,
        reward_type=reward_type,
        max_turns=max_turns,
        training_mode=training_mode,
        output_root=str(base_dir),
        sample_games=0,
        eval_episodes=min(eval_episodes, 50),
    )
    cfg.runs = _build_variant_runs(
        variant,
        player_decks=player_decks,
        seeds=seeds,
        opponents=opponents,
    )

    started = time.time()
    run_sweep(cfg, force=force)
    train_elapsed = time.time() - started
    console.print(
        f"  [green]variant {variant.name} training done[/] in {format_duration(train_elapsed)}"
    )

    # ---- Evaluation -----------------------------------------------------
    print_divider(f"Evaluating variant: {variant.name}")
    evaluate_sweep(
        experiment_dir=variant_dir,
        n_episodes=eval_episodes,
        include_baselines=include_baselines,
        baseline_overrides=baseline_overrides,
        extra_player_decks=extra_player_decks,
        max_turns=max_turns,
        agency_mode=agency_mode,
    )
    return variant_dir / "eval" / "eval_results.json"


def run_ablation(
    *,
    experiment_name: str,
    variants: tp.Sequence[AblationVariant],
    player_decks: list[str],
    seeds: list[int],
    opponents: list[str],
    timesteps_per_opponent: int,
    n_envs: int | str = "auto",
    agency_mode: str = "auto",
    reward_type: str = "shaped",
    max_turns: int = 20,
    training_mode: str = "round-robin",
    output_root: str = "results/research",
    eval_episodes: int = 500,
    include_baselines: bool = True,
    baseline_overrides: list[str] | None = None,
    extra_player_decks: list[str] | None = None,
    baseline_variant: str = "cgfa_full",
    force: bool = False,
) -> Path:
    """Run the full ablation suite end-to-end.

    Returns:
        Path to the aggregated ``aggregated_results.json`` file.
    """
    print_logo()
    print_divider(f"Ablation Suite: {experiment_name}")
    base_dir = Path(output_root) / experiment_name
    base_dir.mkdir(parents=True, exist_ok=True)

    # Persist the variant list so the run is reproducible from the
    # output directory alone.
    save_ablation_variants(variants, base_dir / "ablation_variants.yaml")

    console.print(f"  output:   [dim]{base_dir}[/]")
    console.print(f"  variants: [cyan]{', '.join(v.name for v in variants)}[/]")
    console.print(f"  decks:    [cyan]{', '.join(player_decks)}[/]")
    console.print(f"  seeds:    [cyan]{seeds}[/]")
    console.print(f"  budget:   [cyan]{timesteps_per_opponent:,}[/] / opponent")

    eval_paths: list[Path] = []
    variant_started = time.time()
    for variant in variants:
        eval_path = _run_one_variant(
            variant,
            base_dir=base_dir,
            player_decks=player_decks,
            seeds=seeds,
            opponents=opponents,
            timesteps_per_opponent=timesteps_per_opponent,
            n_envs=n_envs,
            agency_mode=agency_mode,
            reward_type=reward_type,
            max_turns=max_turns,
            training_mode=training_mode,
            eval_episodes=eval_episodes,
            include_baselines=include_baselines,
            baseline_overrides=baseline_overrides,
            extra_player_decks=extra_player_decks or [],
            force=force,
        )
        eval_paths.append(eval_path)

    total_elapsed = time.time() - variant_started
    print_divider(f"All variants complete ({format_duration(total_elapsed)})")

    # ---- Final cross-variant aggregation -------------------------------
    agg_dir = base_dir / "aggregated"
    print_divider("Cross-variant aggregation (paired-bootstrap)")
    source_compare_pairs = [
        (variant.name, baseline_variant) for variant in variants if variant.name != baseline_variant
    ]
    aggregate(
        eval_paths=eval_paths,
        output_dir=agg_dir,
        baseline_agent=None,  # baselines are sweep-level, not variant-level
        source_labels=[v.name for v in variants],
        headline_compare_agents=None,
        source_compare_pairs=source_compare_pairs,
    )

    # Highlight the baseline-variant comparisons in the console.
    if baseline_variant in {v.name for v in variants}:
        console.print(
            f"\n[bold green]Ablation done[/]: comparisons in "
            f"[cyan]{agg_dir / 'tables' / 'significance.tex'}[/] "
            f"are pairwise; the most relevant column is "
            f"[bold]{baseline_variant} vs others[/]."
        )
    else:
        console.print(f"\n[bold green]Ablation done[/]: aggregated results in [cyan]{agg_dir}[/]")
    return agg_dir / "aggregated_results.json"


def _resolve_variants(
    variants_arg: list[str],
    variants_yaml: Path | None,
) -> list[AblationVariant]:
    """Resolve the user-specified variants to a list of dataclasses."""
    if variants_yaml is not None:
        pool = load_ablation_variants(variants_yaml)
        if variants_arg and variants_arg != ["all"]:
            return variants_by_name(variants_arg, available=pool)
        return pool
    if variants_arg == ["stress"]:
        return stress_cgfa_ablation_variants()
    pool = default_cgfa_ablation_variants()
    if not variants_arg or variants_arg == ["all"]:
        return pool
    return variants_by_name(variants_arg, available=pool)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        prog="ablation_sweep",
        description=(
            "6-point CGFA-PPO ablation sweep: train + eval + cross-variant "
            "aggregation in one command."
        ),
    )
    p.add_argument("--experiment-name", default=None)
    p.add_argument(
        "--variants",
        nargs="+",
        default=["all"],
        help=(
            "Subset of variants to run. 'all' (default) runs the full "
            "6-point suite; 'stress' adds reviewer-facing CGFA stress "
            "variants. Otherwise specify variant names from the canonical "
            "suite (e.g. 'ppo cgfa_full')."
        ),
    )
    p.add_argument(
        "--variants-yaml",
        type=Path,
        default=None,
        help=(
            "Optional YAML file with custom variant definitions (see "
            "mtg/experiments/ablations.yaml for the schema)."
        ),
    )
    p.add_argument(
        "--player-decks",
        nargs="+",
        default=["mono_red_aggro"],
        choices=ALL_DECKS,
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument(
        "--opponents",
        nargs="+",
        default=ALL_DECKS,
        choices=ALL_DECKS,
    )
    p.add_argument("--timesteps-per-opponent", type=int, default=2_000_000)
    p.add_argument(
        "--training-mode",
        choices=["round-robin", "sequential"],
        default="round-robin",
    )
    p.add_argument("--agency", choices=["auto", "full", "curriculum"], default="auto")
    p.add_argument("--n-envs", default="auto", help="int or 'auto' (CPU - 1)")
    p.add_argument(
        "--reward-type",
        choices=["sparse", "shaped", "dense"],
        default="shaped",
    )
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument("--output-root", default="results/research")
    p.add_argument("--eval-episodes", type=int, default=500)
    p.add_argument(
        "--no-baselines",
        action="store_true",
        help="Skip baseline (random + heuristic) evaluation entirely.",
    )
    p.add_argument(
        "--baseline-agents",
        nargs="*",
        default=None,
        help="Override the auto-paired baseline list (applied to every deck).",
    )
    p.add_argument("--extra-player-decks", nargs="*", default=[])
    p.add_argument(
        "--baseline-variant",
        default="cgfa_full",
        help="Variant used as the reference in the final significance summary.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-train even if a model exists.",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Tiny smoke test: only the cgfa_full variant, 1 deck, 1 seed, "
            "5k steps, 5 eval episodes."
        ),
    )
    return p.parse_args()


def main() -> int:
    """Entry point for ``python -m scripts.research.ablation_sweep``."""
    args = parse_args()
    if args.quick:
        variants = variants_by_name(["cgfa_full"])
        return (
            0
            if run_ablation(
                experiment_name=args.experiment_name or "ablation_smoke",
                variants=variants,
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
                eval_episodes=10,
                include_baselines=True,
                baseline_overrides=["random"],
                extra_player_decks=[],
                baseline_variant="cgfa_full",
                force=True,
            )
            else 1
        )

    variants = _resolve_variants(args.variants, args.variants_yaml)

    experiment_name = args.experiment_name or f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_ablation(
        experiment_name=experiment_name,
        variants=variants,
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
        eval_episodes=args.eval_episodes,
        include_baselines=not args.no_baselines,
        baseline_overrides=args.baseline_agents,
        extra_player_decks=args.extra_player_decks,
        baseline_variant=args.baseline_variant,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    raise SystemExit(main())

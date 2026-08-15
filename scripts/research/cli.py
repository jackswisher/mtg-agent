"""``mtg-research``: composable research pipeline entry point.

Three stages (``train_sweep`` -> ``eval_sweep`` -> ``aggregate``) are unified
behind a single command with subcommands and an interactive wizard::

    mtg-research -i                    # interactive wizard (runs full pipeline)
    mtg-research pipeline ...          # full pipeline with flags
    mtg-research train ...             # Stage 1 only (delegates to train_sweep)
    mtg-research eval ...              # Stage 2 only (delegates to eval_sweep)
    mtg-research aggregate ...         # Stage 3 only (delegates to aggregate)

Each subcommand is functionally identical to the ``python -m
scripts.research.<stage>`` invocation; the wrapper exists so users don't have
to remember the module paths.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.prompt import Confirm, IntPrompt, Prompt  # noqa: E402

from mtg.agents import heuristic_for_deck  # noqa: E402
from mtg.utils.cli_display import console, print_divider, print_logo  # noqa: E402
from mtg.utils.interactive import format_duration  # noqa: E402

# NOTE: heavier imports are deferred into the dispatchers below so ``--help``
# stays snappy and so argparse of subcommands does not pay their start-up cost.


ALL_AGENTS = ["ppo", "causal", "cgfa", "cgfa_scalar_only"]
ALL_DECKS = [
    "mono_red_aggro",
    "azorius_control",
    "dimir_midrange",
    "domain_ramp",
    "boros_convoke",
]


def _preview_auto_baselines(player_decks: list[str]) -> dict[str, list[str]]:
    """Return the auto-paired baselines we'd evaluate against each deck.

    Used purely for display in the wizard summary; the real resolution is
    done inside ``evaluate_sweep`` so a single source of truth applies.
    """
    preview: dict[str, list[str]] = {}
    for deck in player_decks:
        names = ["random"]
        matched = heuristic_for_deck(deck)
        if matched is not None:
            names.append(matched)
        preview[deck] = names
    return preview


# ---------------------------------------------------------------------------
# Subcommand dispatchers (delegate to the existing per-stage main()s)
# ---------------------------------------------------------------------------


def _dispatch_train(argv: list[str]) -> int:
    from scripts.research import train_sweep

    sys.argv = ["mtg-research train", *argv]
    return train_sweep.main()


def _dispatch_eval(argv: list[str]) -> int:
    from scripts.research import eval_sweep

    sys.argv = ["mtg-research eval", *argv]
    return eval_sweep.main()


def _dispatch_aggregate(argv: list[str]) -> int:
    from scripts.research import aggregate

    sys.argv = ["mtg-research aggregate", *argv]
    return aggregate.main()


def _dispatch_ablation(argv: list[str]) -> int:
    from scripts.research import ablation_sweep

    sys.argv = ["mtg-research ablation", *argv]
    return ablation_sweep.main()


def _dispatch_calibration_plot(argv: list[str]) -> int:
    from scripts.research import calibration_plot

    sys.argv = ["mtg-research calibration-plot", *argv]
    return calibration_plot.main()


def _dispatch_case_study(argv: list[str]) -> int:
    from scripts.research import case_study

    sys.argv = ["mtg-research case-study", *argv]
    return case_study.main()


def _dispatch_transfer(argv: list[str]) -> int:
    from scripts.research import transfer_sweep

    sys.argv = ["mtg-research transfer", *argv]
    return transfer_sweep.main()


# ---------------------------------------------------------------------------
# End-to-end pipeline (programmatic, no sys.argv munging)
# ---------------------------------------------------------------------------


def _run_pipeline(
    *,
    experiment_name: str,
    agents: list[str],
    player_decks: list[str],
    seeds: list[int],
    opponents: list[str],
    timesteps_per_opponent: int,
    training_mode: str,
    agency_mode: str,
    n_envs: int | str,
    reward_type: str,
    max_turns: int,
    output_root: str,
    eval_episodes: int,
    include_baselines: bool,
    baseline_overrides: list[str] | None,
    extra_player_decks: list[str],
    baseline_agent: str | None,
    aggregate_dir: Path | None,
    source_labels: list[str] | None,
    agent_kwargs_by_agent: dict[str, dict[str, object]] | None,
    force: bool,
) -> int:
    """Run Stages 1, 2 and 3 for a single experiment end-to-end.

    Baseline semantics: ``include_baselines=True`` (default) auto-pairs each
    player deck with ``random`` plus the canonical heuristic from
    :data:`mtg.agents.DECK_TO_HEURISTIC` (e.g. ``azorius_control`` ->
    ``random`` + ``control``). ``baseline_overrides`` is an advanced escape
    hatch that forces a fixed list of baseline agent names onto every deck --
    only useful for specific ablations. Set ``include_baselines=False`` to
    skip baseline evaluation entirely.

    For A-vs-B paired comparisons (e.g. PPO vs Causal), call this twice with
    the same seeds but different ``experiment_name`` and ``agents`` values,
    then pass both eval_results.json files to ``mtg-research aggregate``.
    """
    from scripts.research.aggregate import aggregate
    from scripts.research.eval_sweep import evaluate_sweep
    from scripts.research.train_sweep import SweepConfig, run_sweep

    # ---- Stage 1: train_sweep -----------------------------------------------
    print_logo()
    print_divider(f"Research Pipeline: {experiment_name}")

    cfg = SweepConfig(
        experiment_name=experiment_name,
        agents=agents,
        player_decks=player_decks,
        seeds=seeds,
        opponents=opponents,
        timesteps_per_opponent=timesteps_per_opponent,
        n_envs=n_envs,
        agency_mode=agency_mode,
        reward_type=reward_type,
        max_turns=max_turns,
        training_mode=training_mode,
        output_root=output_root,
        sample_games=0,
        eval_episodes=min(eval_episodes, 50),  # quick sanity inside the sweep
        agent_kwargs_by_agent=agent_kwargs_by_agent or {},
    )
    run_sweep(cfg, force=force)
    experiment_dir = cfg.experiment_dir()

    # ---- Stage 2: eval_sweep ------------------------------------------------
    print_divider("Stage 2: Evaluation")
    evaluate_sweep(
        experiment_dir=experiment_dir,
        n_episodes=eval_episodes,
        include_baselines=include_baselines,
        baseline_overrides=baseline_overrides,
        extra_player_decks=extra_player_decks,
        max_turns=max_turns,
        agency_mode=agency_mode,
    )

    # ---- Stage 3: aggregate -------------------------------------------------
    print_divider("Stage 3: Aggregation")
    eval_results_path = experiment_dir / "eval" / "eval_results.json"
    agg_out = aggregate_dir or (experiment_dir / "aggregated")
    aggregate(
        eval_paths=[eval_results_path],
        output_dir=agg_out,
        baseline_agent=baseline_agent,
        source_labels=source_labels,
        headline_compare_agents=agents if len(agents) >= 2 else None,
    )

    console.print(
        f"\n[bold green]Pipeline complete[/]: [dim]{experiment_dir}[/]\n"
        f"  figures: [cyan]{agg_out / 'figures'}[/]\n"
        f"  tables:  [cyan]{agg_out / 'tables'}[/]"
    )
    return 0


# ---------------------------------------------------------------------------
# Interactive wizard
# ---------------------------------------------------------------------------


def _prompt_checklist(
    label: str,
    options: list[str],
    defaults: list[str] | None = None,
) -> list[str]:
    """Ask the user to pick a comma-separated subset of options.

    Returns the selected options, preserving the canonical order in ``options``.
    """
    defaults = defaults if defaults is not None else list(options)
    console.print(f"\n[bold]{label}[/bold]")
    for i, opt in enumerate(options, start=1):
        marker = "[green]✓[/]" if opt in defaults else "  "
        console.print(f"  {marker} [cyan]{i}[/] {opt}")
    console.print(
        "  [dim]Enter comma-separated indices (e.g. '1,3,4'), or press Enter for defaults.[/dim]"
    )
    raw = Prompt.ask("Selection", default="all")
    raw = raw.strip().lower()
    if raw in {"", "all"}:
        return defaults
    picks: list[str] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            idx = int(tok) - 1
            if 0 <= idx < len(options):
                picks.append(options[idx])
        except ValueError:
            if tok in options:
                picks.append(tok)
    if not picks:
        console.print("[yellow]No valid selection; falling back to defaults.[/]")
        return defaults
    # Preserve canonical order, deduplicate.
    seen: set[str] = set()
    ordered: list[str] = []
    for opt in options:
        if opt in picks and opt not in seen:
            seen.add(opt)
            ordered.append(opt)
    return ordered


def _prompt_seeds(default: list[int]) -> list[int]:
    console.print(
        "\n[bold]Random seeds[/bold] [dim](one trained model per seed; "
        ">=3 recommended for paired-bootstrap tests)[/dim]"
    )
    raw = Prompt.ask(
        "Seeds (comma or space separated)",
        default=" ".join(str(s) for s in default),
    )
    seeds: list[int] = []
    for tok in raw.replace(",", " ").split():
        try:
            seeds.append(int(tok))
        except ValueError:
            continue
    return seeds or default


def interactive_wizard() -> int:
    """Walk the user through a research-pipeline configuration and run it."""
    print_logo()
    print_divider("Research Pipeline (Interactive)")

    console.print(
        "[dim]This wizard configures a research run "
        "(train_sweep -> eval_sweep -> aggregate).[/dim]\n"
    )

    # --- Mode selection -----------------------------------------------------
    console.print("[bold]Step 1: What are you running?[/bold]")
    console.print(
        "  [cyan]1[/] [green]Single method[/]: train+eval+aggregate one agent (e.g. PPO baseline)"
    )
    console.print(
        "  [cyan]2[/] [green]A/B comparison[/]: train PPO and Causal at the "
        "SAME seeds, then compare with paired-bootstrap significance tests "
        "(recommended for paper claims)"
    )
    console.print(
        "  [cyan]3[/] [yellow]Quick smoke test[/]: a short dry run to verify "
        "the pipeline end-to-end"
    )
    mode = IntPrompt.ask("Select (1-3)", default=2)

    if mode == 3:
        return _run_quick_smoke_test()

    comparison = mode == 2

    # --- Experiment name ----------------------------------------------------
    default_name = (
        "paper_comparison" if comparison else "research_run"
    ) + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_name = Prompt.ask(
        "\n[bold]Experiment name[/bold]",
        default=default_name,
    )

    # --- Agents -------------------------------------------------------------
    if comparison:
        agents_main = ["ppo"]
        agents_compare = ["causal"]
        console.print(
            "\n[bold]Step 2: Agents[/bold]: locked to [green]PPO vs Causal[/] for A/B comparison."
        )
    else:
        console.print("\n[bold]Step 2: Which agent?[/bold]")
        console.print("  [cyan]1[/] ppo     : vanilla PPO (MaskablePPO)")
        console.print("  [cyan]2[/] causal  : PPO + Causal World Model")
        agent_choice = IntPrompt.ask("Select (1-2)", default=1)
        agents_main = ["causal"] if agent_choice == 2 else ["ppo"]
        agents_compare = []

    # --- Player decks -------------------------------------------------------
    console.print("\n[bold]Step 3: Player decks[/bold] (decks the agent plays)")
    player_decks = _prompt_checklist(
        "Select player decks",
        ALL_DECKS,
        defaults=["mono_red_aggro"],
    )

    # --- Opponents ----------------------------------------------------------
    console.print("\n[bold]Step 4: Opponents[/bold] (decks played against)")
    opponents = _prompt_checklist(
        "Select opponents",
        ALL_DECKS,
        defaults=ALL_DECKS,
    )

    # --- Seeds --------------------------------------------------------------
    seeds = _prompt_seeds([42, 123, 456])

    # --- Training budget ----------------------------------------------------
    console.print(
        "\n[bold]Step 5: Training budget[/bold]\n"
        "  [dim]Timesteps per opponent (multiplied by |opponents| for "
        "round-robin total). Recommended: 500K (quick), 1M (standard), "
        "2M+ (publication).[/dim]"
    )
    timesteps_per_opponent = IntPrompt.ask("Timesteps per opponent", default=2_000_000)

    # --- Training mode ------------------------------------------------------
    training_mode = "round-robin"
    if len(opponents) > 1:
        console.print(
            "\n[bold]Training mode:[/bold]\n"
            "  [cyan]1[/] [green]round-robin[/] (recommended): interleave "
            "opponents every swap-window\n"
            "  [cyan]2[/] [yellow]sequential[/]: full budget per opponent in order"
        )
        training_mode = (
            "sequential" if IntPrompt.ask("Select (1-2)", default=1) == 2 else "round-robin"
        )

    # --- Agency -------------------------------------------------------------
    console.print(
        "\n[bold]Step 6: Decision agency[/bold]\n"
        "  [cyan]1[/] [green]auto[/] [dim](recommended for PPO)[/]: "
        "all-or-nothing combat, auto-target spells\n"
        "  [cyan]2[/] [green]curriculum[/] [dim](recommended for Causal)[/]: "
        "70% auto then 30% full\n"
        "  [cyan]3[/] [yellow]full[/]: per-attacker/per-target agency "
        "(>=3M steps/opponent)"
    )
    default_agency = 2 if comparison or "causal" in agents_main else 1
    agency_choice = IntPrompt.ask("Select (1-3)", default=default_agency)
    agency_mode = {1: "auto", 2: "curriculum", 3: "full"}.get(agency_choice, "auto")

    # --- n_envs -------------------------------------------------------------
    cpu_count = os.cpu_count() or 4
    console.print(
        f"\n[bold]Parallel environments[/bold] [dim](0 = auto = {cpu_count - 1} cores)[/]"
    )
    n_envs_input = IntPrompt.ask("n_envs", default=0)
    n_envs: int | str = "auto" if n_envs_input <= 0 else n_envs_input

    # --- Evaluation budget --------------------------------------------------
    console.print(
        "\n[bold]Step 7: Evaluation[/bold]\n"
        "  [dim]500+ eps/opponent is recommended to keep win-rate CIs at "
        "±5% or tighter.[/dim]"
    )
    eval_episodes = IntPrompt.ask("Episodes per opponent", default=500)

    # --- Baselines ----------------------------------------------------------
    # By default, each player deck is auto-paired with `random` + the
    # deck-matched heuristic (e.g. `control` for `azorius_control`). This is
    # almost always what users want; surfacing the per-deck list keeps the
    # behaviour transparent.
    auto_preview = _preview_auto_baselines(player_decks)
    console.print(
        "\n[bold]Baselines[/bold] [dim](evaluated alongside the trained agent "
        "for free; gives every method a sanity floor + a strategically matched "
        "non-RL opponent)[/]"
    )
    for deck, names in auto_preview.items():
        console.print(f"  {deck:<18} -> [cyan]{', '.join(names)}[/]")

    include_baselines = Confirm.ask(
        "Include these auto-paired baselines?",
        default=True,
    )
    baseline_overrides: list[str] | None = None
    if include_baselines and Confirm.ask(
        "[dim]Advanced: override with a custom baseline list (applied to every deck)?[/]",
        default=False,
    ):
        raw = Prompt.ask(
            "Baseline agent names (space-separated)",
            default="random",
        )
        baseline_overrides = raw.split() or None

    # --- Output + confirm ---------------------------------------------------
    output_root = "results/research"
    console.print(
        f"\n[bold]Output root:[/] [cyan]{output_root}[/]  "
        f"[dim](each experiment lands in <output_root>/<experiment_name>)[/]"
    )

    console.print("\n[bold]Summary[/bold]")
    total_timesteps = timesteps_per_opponent * (
        len(opponents) if training_mode == "round-robin" else 1
    )
    console.print(f"  Experiment  : [cyan]{experiment_name}[/]")
    console.print(f"  Agents      : [cyan]{', '.join(agents_main + agents_compare)}[/]")
    console.print(f"  Player decks: [cyan]{', '.join(player_decks)}[/]")
    console.print(f"  Opponents   : [cyan]{', '.join(opponents)}[/]")
    console.print(f"  Seeds       : [cyan]{seeds}[/]")
    console.print(
        f"  Budget      : [cyan]{timesteps_per_opponent:,}[/] / opponent "
        f"([cyan]{total_timesteps:,}[/] total, {training_mode})"
    )
    console.print(f"  Agency      : [cyan]{agency_mode}[/] (applied to BOTH arms)")
    console.print(f"  Eval        : [cyan]{eval_episodes}[/] episodes / opponent")
    if not include_baselines:
        console.print("  Baselines   : [yellow]disabled[/]")
    elif baseline_overrides is not None:
        console.print(
            f"  Baselines   : [cyan]{', '.join(baseline_overrides)}[/] "
            "[dim](custom override, applied to every deck)[/]"
        )
    else:
        per_deck = ", ".join(f"{d}->{'+'.join(names)}" for d, names in auto_preview.items())
        console.print(f"  Baselines   : [cyan]{per_deck}[/] [dim](auto-paired)[/]")

    if not Confirm.ask("\nStart pipeline?", default=True):
        console.print("[yellow]Cancelled.[/]")
        return 1

    # --- Run ----------------------------------------------------------------
    if not comparison:
        return _run_pipeline(
            experiment_name=experiment_name,
            agents=agents_main,
            player_decks=player_decks,
            seeds=seeds,
            opponents=opponents,
            timesteps_per_opponent=timesteps_per_opponent,
            training_mode=training_mode,
            agency_mode=agency_mode,
            n_envs=n_envs,
            reward_type="shaped",
            max_turns=20,
            output_root=output_root,
            eval_episodes=eval_episodes,
            include_baselines=include_baselines,
            baseline_overrides=baseline_overrides,
            extra_player_decks=[],
            baseline_agent=None,
            aggregate_dir=None,
            source_labels=None,
            agent_kwargs_by_agent=None,
            force=False,
        )

    # A/B comparison: run both halves then cross-aggregate with paired tests.
    # We only need to score baselines once (on the PPO sweep) since they are
    # identical models; the Causal sweep skips them to avoid duplicate work.
    #
    # CRITICAL: both arms MUST use the same ``agency_mode`` so they
    # operate in the same MDP. Mixing agency modes across arms makes
    # them act in different action spaces and silently invalidates
    # every paired-bootstrap claim downstream.
    name_a = f"{experiment_name}__ppo"
    name_b = f"{experiment_name}__causal"
    for name, agents_here, agency in [
        (name_a, agents_main, agency_mode),
        (name_b, agents_compare, agency_mode),
    ]:
        rc = _run_pipeline(
            experiment_name=name,
            agents=agents_here,
            player_decks=player_decks,
            seeds=seeds,
            opponents=opponents,
            timesteps_per_opponent=timesteps_per_opponent,
            training_mode=training_mode,
            agency_mode=agency,
            n_envs=n_envs,
            reward_type="shaped",
            max_turns=20,
            output_root=output_root,
            eval_episodes=eval_episodes,
            include_baselines=include_baselines if name == name_a else False,
            baseline_overrides=baseline_overrides if name == name_a else None,
            extra_player_decks=[],
            baseline_agent=None,
            aggregate_dir=None,
            source_labels=None,
            agent_kwargs_by_agent=None,
            force=False,
        )
        if rc != 0:
            return rc

    # Final cross-sweep aggregation with paired-bootstrap tests.
    from scripts.research.aggregate import aggregate

    print_divider("Cross-sweep aggregation (paired comparison)")
    eval_a = Path(output_root) / name_a / "eval" / "eval_results.json"
    eval_b = Path(output_root) / name_b / "eval" / "eval_results.json"
    compare_out = Path(output_root) / f"{experiment_name}__compare"
    aggregate(
        eval_paths=[eval_a, eval_b],
        output_dir=compare_out,
        baseline_agent="ppo",
        source_labels=["ppo", "causal"],
        headline_compare_agents=None,
    )
    console.print(
        f"\n[bold green]A/B comparison complete[/]: paired-bootstrap "
        f"tests in [cyan]{compare_out / 'tables' / 'significance.tex'}[/]"
    )
    return 0


# ---------------------------------------------------------------------------
# Single-command paper reproduction (Figs. 3-7 in one shot)
# ---------------------------------------------------------------------------


PAPER_STAGES: tuple[str, ...] = (
    "headline",
    "ablation",
    "transfer",
    "calibration",
    "case-study",
    "cross-source",
)

TRANSFER_MODES: tuple[str, ...] = ("fixed", "leave-one-out")


def _resolve_paper_stages(
    only: list[str] | None,
    skip: list[str] | None,
) -> list[str]:
    """Resolve the ordered set of stages to run for ``mtg-research paper``.

    ``only`` and ``skip`` are mutually exclusive.  When both are empty
    every stage runs in canonical order.
    """
    if only and skip:
        raise ValueError("--only and --skip are mutually exclusive")
    if only:
        unknown = sorted(set(only) - set(PAPER_STAGES))
        if unknown:
            raise ValueError(
                f"Unknown stage(s) in --only: {unknown}. " f"Choose from {list(PAPER_STAGES)}."
            )
        return [s for s in PAPER_STAGES if s in set(only)]
    if skip:
        unknown = sorted(set(skip) - set(PAPER_STAGES))
        if unknown:
            raise ValueError(
                f"Unknown stage(s) in --skip: {unknown}. " f"Choose from {list(PAPER_STAGES)}."
            )
        return [s for s in PAPER_STAGES if s not in set(skip)]
    return list(PAPER_STAGES)


def _find_cgfa_full_artifacts(
    ablation_dir: Path,
    player_deck: str,
) -> list[tuple[Path, Path, int]]:
    """Locate (model, calibration_csv, seed) triples under the ablation tree.

    Reads the per-variant ``sweep_manifest.yaml`` to map seeds to the
    timestamped run sub-directories, then verifies that the expected
    artefacts actually exist on disk.  Variants train one (deck, seed)
    grid each, so this returns one row per seed for ``cgfa_full``.
    """
    import yaml

    variant_dir = ablation_dir / "cgfa_full"
    manifest_path = variant_dir / "sweep_manifest.yaml"
    if not manifest_path.exists():
        return []
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f) or {}
    artefacts: list[tuple[Path, Path, int]] = []
    for run in manifest.get("runs", []):
        if run.get("status") not in {"completed", "skipped"}:
            continue
        if run.get("player_deck") != player_deck:
            continue
        run_dir = variant_dir / run["output_dir"]
        model = run_dir / f"{run['agent']}_{run['player_deck']}.zip"
        cal_csv = run_dir / "cgfa" / "cgfa_calibration.csv"
        if model.exists() and cal_csv.exists():
            artefacts.append((model, cal_csv, int(run["seed"])))
    return artefacts


def _loo_folds(opponents: list[str]) -> list[tuple[str, list[str]]]:
    """Build leave-one-out (heldout, train_set) folds from an opponent pool.

    For an opponent pool of size N, returns N folds: in each fold, one
    opponent is the held-out test set and the other N-1 are the training
    set. Order is preserved from ``opponents`` so re-runs are stable.

    Raises ``ValueError`` if the pool has fewer than 2 opponents (LOO
    needs at least one held-out and one train opponent).
    """
    if len(opponents) < 2:
        raise ValueError(
            f"--transfer-mode leave-one-out requires at least 2 opponents in "
            f"the pool (--opponents + --heldout-opponents), got {len(opponents)}."
        )
    folds: list[tuple[str, list[str]]] = []
    for heldout in opponents:
        train_set = [o for o in opponents if o != heldout]
        folds.append((heldout, train_set))
    return folds


def _discover_eval_results(paper_root: Path) -> list[tuple[str, Path]]:
    """Walk ``paper_root`` and return every ``(label, eval_results.json)`` found.

    Sources discovered (canonical paper layout):
    * ``headline/<run>/eval/eval_results.json`` -> label ``headline``
    * ``ablation/<variant>/eval/eval_results.json`` -> label ``ablation_<variant>``
    * ``transfer/eval/eval_results.json`` (fixed split) -> label ``transfer_indist``
    * ``transfer/eval_heldout/eval_results.json`` (fixed split) -> label ``transfer_heldout``
    * ``transfer/loo_<deck>/eval/eval_results.json`` (LOO mode)
      -> label ``transfer_loo_<deck>_indist``
    * ``transfer/loo_<deck>/eval_heldout/eval_results.json`` (LOO mode)
      -> label ``transfer_loo_<deck>_heldout``

    Files are returned in a deterministic, lexicographic order so the
    cross-source aggregate produces stable output across re-runs. Missing
    sub-directories are skipped silently --- this lets the user run e.g.
    ``--only headline`` and still get a (single-source) cross-source roll-up.
    """
    discovered: list[tuple[str, Path]] = []
    headline_root = paper_root / "headline"
    if headline_root.is_dir():
        for cand in sorted(headline_root.glob("**/eval/eval_results.json")):
            discovered.append(("headline", cand))
    ablation_root = paper_root / "ablation"
    if ablation_root.is_dir():
        for variant_dir in sorted(p for p in ablation_root.iterdir() if p.is_dir()):
            cand = variant_dir / "eval" / "eval_results.json"
            if cand.exists():
                discovered.append((f"ablation_{variant_dir.name}", cand))
    transfer_root = paper_root / "transfer"
    if transfer_root.is_dir():
        fixed_indist = transfer_root / "eval" / "eval_results.json"
        if fixed_indist.exists():
            discovered.append(("transfer_indist", fixed_indist))
        fixed_heldout = transfer_root / "eval_heldout" / "eval_results.json"
        if fixed_heldout.exists():
            discovered.append(("transfer_heldout", fixed_heldout))
        for fold_dir in sorted(transfer_root.glob("loo_*")):
            if not fold_dir.is_dir():
                continue
            indist = fold_dir / "eval" / "eval_results.json"
            heldout = fold_dir / "eval_heldout" / "eval_results.json"
            if indist.exists():
                discovered.append((f"transfer_{fold_dir.name}_indist", indist))
            if heldout.exists():
                discovered.append((f"transfer_{fold_dir.name}_heldout", heldout))
    return discovered


def _print_paper_plan(
    *,
    experiment_name: str,
    stages: list[str],
    agents: list[str],
    player_decks: list[str],
    ablation_decks: list[str],
    transfer_decks: list[str],
    opponents: list[str],
    heldout_opponents: list[str],
    transfer_mode: str,
    seeds: list[int],
    timesteps_per_opponent: int,
    eval_episodes: int,
    cgfa_calibration_mode: str,
    output_root: str,
    case_study_player_deck: str,
    case_study_opponent_deck: str,
    case_study_seed: int,
) -> None:
    """Pretty-print the planned commands so the user can sanity-check first.

    Every stage writes under a single master directory
    ``<output_root>/<experiment_name>/`` so multiple paper runs sit side
    by side without polluting the top-level results folder.
    """
    print_divider(f"Paper reproduction plan: {experiment_name}")
    paper_root = f"{output_root}/{experiment_name}"
    seeds_str = " ".join(str(s) for s in seeds)
    base = (
        f" --seeds {seeds_str}"
        f" --timesteps-per-opponent {timesteps_per_opponent}"
        f" --eval-episodes {eval_episodes}"
        f" --output-root {paper_root}"
    )
    if "headline" in stages:
        console.print(
            "\n[bold cyan]1. Headline (Fig. 3)[/]\n"
            "   mtg-research pipeline --experiment-name headline"
            f" --agents {' '.join(agents)}"
            f" --player-decks {' '.join(player_decks)}"
            f" --opponents {' '.join(opponents)}{base}\n"
            f"   CGFA calibration mode: {cgfa_calibration_mode}"
        )
    if "ablation" in stages:
        console.print(
            "\n[bold cyan]2. 6-point ablation (Fig. 4)[/]\n"
            "   mtg-research ablation --experiment-name ablation"
            " --variants all"
            f" --player-decks {' '.join(ablation_decks)}"
            f" --opponents {' '.join(opponents)}{base}\n"
            f"   CGFA calibration mode: {cgfa_calibration_mode}"
        )
    if "transfer" in stages:
        if transfer_mode == "leave-one-out":
            loo_pool = list(dict.fromkeys([*opponents, *heldout_opponents]))
            folds = _loo_folds(loo_pool)
            console.print(
                "\n[bold cyan]3. Held-out transfer (Fig. 5) [yellow]LEAVE-ONE-OUT[/yellow][/]"
                f"   pool=[cyan]{', '.join(loo_pool)}[/]   "
                f"folds=[cyan]{len(folds)}[/]\n"
                "   [yellow]Cost warning:[/] each fold trains a fresh sweep on "
                f"{len(loo_pool) - 1} opponents, so transfer training is "
                f"~[bold]{len(folds) * (len(loo_pool) - 1)}/{len(opponents)}x[/] "
                "the fixed-split cost (no model reuse across folds).\n"
                f"   CGFA calibration mode: {cgfa_calibration_mode}"
            )
            for heldout, train_set in folds:
                console.print(
                    f"     fold heldout=[red]{heldout}[/]   "
                    f"train=[cyan]{', '.join(train_set)}[/]"
                )
        else:
            console.print(
                "\n[bold cyan]3. Held-out transfer (Fig. 5) [dim]fixed split[/dim][/]\n"
                "   mtg-research transfer --experiment-name transfer"
                f" --agents {' '.join(agents)}"
                f" --player-decks {' '.join(transfer_decks)}"
                f" --train-opponents {' '.join(opponents)}"
                f" --heldout-opponents {' '.join(heldout_opponents)}{base}\n"
                f"   CGFA calibration mode: {cgfa_calibration_mode}"
            )
    if "calibration" in stages:
        cal_glob = f"{paper_root}/ablation/cgfa_full/*/cgfa/cgfa_calibration.csv"
        cal_out = f"{paper_root}/ablation/figures/cgfa_calibration.png"
        console.print(
            "\n[bold cyan]4. Calibration plot (Fig. 6)[/]\n"
            f"   mtg-research calibration-plot {cal_glob} --output {cal_out}"
            f" --player-deck {case_study_player_deck}"
        )
    if "case-study" in stages:
        ckpt_glob = f"{paper_root}/ablation/cgfa_full/*/cgfa_{case_study_player_deck}.zip"
        cs_out = f"{paper_root}/ablation/case_study"
        console.print(
            "\n[bold cyan]5. Case study (Fig. 7)[/]\n"
            f"   mtg-research case-study --model-path <first {ckpt_glob}>"
            f" --player-deck {case_study_player_deck}"
            f" --opponent-deck {case_study_opponent_deck}"
            f" --episode-seed {case_study_seed}"
            f" --output-dir {cs_out}"
        )
    if "cross-source" in stages:
        console.print(
            "\n[bold cyan]6. Cross-source aggregate (significance roll-up)[/]\n"
            f"   mtg-research aggregate --output-dir {paper_root}/cross_source\n"
            f"     --eval-results <auto-discovered eval_results.json under {paper_root}/>\n"
            f"     --baseline-agent {agents[0] if agents else 'ppo'}"
        )
    console.print()


def _run_paper(
    *,
    experiment_name: str,
    agents: list[str],
    player_decks: list[str],
    ablation_decks: list[str],
    transfer_decks: list[str],
    opponents: list[str],
    heldout_opponents: list[str],
    seeds: list[int],
    timesteps_per_opponent: int,
    eval_episodes: int,
    training_mode: str,
    agency_mode: str,
    n_envs: int | str,
    reward_type: str,
    max_turns: int,
    output_root: str,
    case_study_player_deck: str,
    case_study_opponent_deck: str,
    case_study_seed: int,
    include_baselines: bool,
    transfer_mode: str,
    cgfa_calibration_mode: str,
    only: list[str] | None,
    skip: list[str] | None,
    force: bool,
    dry_run: bool,
) -> int:
    """Reproduce every paper figure (Figs. 3-7) end-to-end in one command.

    All stage outputs are nested under a single master directory
    ``<output_root>/<experiment_name>/`` so each paper run is
    self-contained and easy to archive::

        results/research/<experiment_name>/
            headline/      # Stage 1 (Fig. 3)
            ablation/      # Stages 2, 4, 5 (Figs. 4, 6, 7)
            transfer/      # Stage 3 (Fig. 5)

    Stages run in this order:

    1. ``headline``    -> ``mtg-research pipeline`` for paired
       PPO vs CGFA-PPO comparison (Fig. 3 + headline tables).
    2. ``ablation``    -> ``mtg-research ablation`` over all six
       variants (Fig. 4 + ``tables/significance.tex``).
    3. ``transfer``    -> ``mtg-research transfer`` with a disjoint
       held-out opponent split (Fig. 5 + ``transfer_report.json``).
    4. ``calibration`` -> ``mtg-research calibration-plot`` rendered
       from the ``cgfa_full`` calibration CSVs (Fig. 6).
    5. ``case-study``  -> ``mtg-research case-study`` on the first
       trained ``cgfa_full`` checkpoint (Fig. 7).

    Path-chaining (calibration CSVs and case-study checkpoint) is done
    by inspecting the ablation-suite ``sweep_manifest.yaml`` so the
    correct seed-to-run mapping is preserved (no fragile ``ls | head``).

    Use ``--only`` or ``--skip`` to run a subset, ``--dry-run`` to
    preview the plan without launching anything, and ``--force`` to
    re-train models that already exist on disk.
    """
    from mtg.experiments.ablation import default_cgfa_ablation_variants
    from scripts.research.ablation_sweep import run_ablation
    from scripts.research.calibration_plot import render as render_calibration
    from scripts.research.transfer_sweep import TransferConfig, run_transfer

    if transfer_mode not in TRANSFER_MODES:
        raise ValueError(
            f"--transfer-mode must be one of {list(TRANSFER_MODES)}, got " f"{transfer_mode!r}."
        )

    stages = _resolve_paper_stages(only, skip)

    paper_root = Path(output_root) / experiment_name
    paper_root_str = str(paper_root)

    print_logo()
    print_divider(f"Paper reproduction: {experiment_name}")
    console.print(
        f"  experiment root: [cyan]{paper_root}/[/]\n"
        f"  agents:          [cyan]{', '.join(agents)}[/]\n"
        f"  decks:           headline=[cyan]{', '.join(player_decks)}[/]  "
        f"ablation=[cyan]{', '.join(ablation_decks)}[/]  "
        f"transfer=[cyan]{', '.join(transfer_decks)}[/]\n"
        f"  opponents:       train=[cyan]{', '.join(opponents)}[/]  "
        f"held-out=[red]{', '.join(heldout_opponents)}[/]\n"
        f"  seeds:           [cyan]{seeds}[/]   "
        f"budget=[cyan]{timesteps_per_opponent:,}[/] / opponent\n"
        f"  eval:            [cyan]{eval_episodes}[/] episodes / cell\n"
        f"  CGFA cal:        [cyan]{cgfa_calibration_mode}[/]\n"
        f"  stages:          [cyan]{' -> '.join(stages)}[/]"
    )

    _print_paper_plan(
        experiment_name=experiment_name,
        stages=stages,
        agents=agents,
        player_decks=player_decks,
        ablation_decks=ablation_decks,
        transfer_decks=transfer_decks,
        opponents=opponents,
        heldout_opponents=heldout_opponents,
        transfer_mode=transfer_mode,
        seeds=seeds,
        timesteps_per_opponent=timesteps_per_opponent,
        eval_episodes=eval_episodes,
        cgfa_calibration_mode=cgfa_calibration_mode,
        output_root=output_root,
        case_study_player_deck=case_study_player_deck,
        case_study_opponent_deck=case_study_opponent_deck,
        case_study_seed=case_study_seed,
    )

    if dry_run:
        console.print(
            "[yellow]Dry run: no stage executed.[/yellow]  "
            "Re-run without [cyan]--dry-run[/] to launch the pipeline."
        )
        return 0

    overall_started = time.time()
    paper_root.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1: headline (Fig. 3) ---------------------------------------
    if "headline" in stages:
        rc = _run_pipeline(
            experiment_name="headline",
            agents=agents,
            player_decks=player_decks,
            seeds=seeds,
            opponents=opponents,
            timesteps_per_opponent=timesteps_per_opponent,
            training_mode=training_mode,
            agency_mode=agency_mode,
            n_envs=n_envs,
            reward_type=reward_type,
            max_turns=max_turns,
            output_root=paper_root_str,
            eval_episodes=eval_episodes,
            include_baselines=include_baselines,
            baseline_overrides=None,
            extra_player_decks=[],
            baseline_agent=agents[0] if agents else None,
            aggregate_dir=None,
            source_labels=None,
            agent_kwargs_by_agent={"cgfa": {"calibration_mode": cgfa_calibration_mode}},
            force=force,
        )
        if rc != 0:
            return rc

    # ---- Stage 2: 6-point ablation (Fig. 4) ------------------------------
    ablation_dir = paper_root / "ablation"
    if "ablation" in stages:
        ablation_variants = []
        for variant in default_cgfa_ablation_variants():
            if variant.agent_type != "cgfa":
                ablation_variants.append(variant)
                continue
            kwargs = dict(variant.agent_kwargs)
            kwargs["calibration_mode"] = cgfa_calibration_mode
            ablation_variants.append(dataclasses.replace(variant, agent_kwargs=kwargs))
        run_ablation(
            experiment_name="ablation",
            variants=ablation_variants,
            player_decks=ablation_decks,
            seeds=seeds,
            opponents=opponents,
            timesteps_per_opponent=timesteps_per_opponent,
            n_envs=n_envs,
            agency_mode=agency_mode,
            reward_type=reward_type,
            max_turns=max_turns,
            training_mode=training_mode,
            output_root=paper_root_str,
            eval_episodes=eval_episodes,
            include_baselines=include_baselines,
            baseline_overrides=None,
            extra_player_decks=[],
            baseline_variant="cgfa_full",
            force=force,
        )

    # ---- Stage 3: held-out transfer (Fig. 5) -----------------------------
    if "transfer" in stages:
        if transfer_mode == "leave-one-out":
            loo_pool = list(dict.fromkeys([*opponents, *heldout_opponents]))
            folds = _loo_folds(loo_pool)
            print_divider(
                f"Stage 3: leave-one-out transfer ({len(folds)} folds over "
                f"{len(loo_pool)} opponents)"
            )
            transfer_root = paper_root / "transfer"
            transfer_root.mkdir(parents=True, exist_ok=True)
            for fold_idx, (heldout, train_set) in enumerate(folds, start=1):
                console.print(
                    f"\n[bold]Fold {fold_idx}/{len(folds)}[/]  "
                    f"heldout=[red]{heldout}[/]   "
                    f"train=[cyan]{', '.join(train_set)}[/]"
                )
                transfer_cfg = TransferConfig(
                    experiment_name=f"transfer/loo_{heldout}",
                    agents=agents,
                    player_decks=transfer_decks,
                    seeds=seeds,
                    train_opponents=train_set,
                    heldout_opponents=[heldout],
                    timesteps_per_opponent=timesteps_per_opponent,
                    eval_episodes=eval_episodes,
                    n_envs=n_envs,
                    agency_mode=agency_mode,
                    reward_type=reward_type,
                    max_turns=max_turns,
                    training_mode=training_mode,
                    output_root=paper_root_str,
                    agent_kwargs={"cgfa": {"calibration_mode": cgfa_calibration_mode}},
                    force=force,
                )
                run_transfer(transfer_cfg)
        else:
            transfer_cfg = TransferConfig(
                experiment_name="transfer",
                agents=agents,
                player_decks=transfer_decks,
                seeds=seeds,
                train_opponents=opponents,
                heldout_opponents=heldout_opponents,
                timesteps_per_opponent=timesteps_per_opponent,
                eval_episodes=eval_episodes,
                n_envs=n_envs,
                agency_mode=agency_mode,
                reward_type=reward_type,
                max_turns=max_turns,
                training_mode=training_mode,
                output_root=paper_root_str,
                agent_kwargs={"cgfa": {"calibration_mode": cgfa_calibration_mode}},
                force=force,
            )
            run_transfer(transfer_cfg)

    # Stages 4 and 5 both consume cgfa_full artefacts produced by Stage 2.
    # If the ablation was skipped, locate them on disk; if nothing is
    # there, skip these stages with a clear warning.
    needs_cgfa = ("calibration" in stages) or ("case-study" in stages)
    cgfa_artefacts: list[tuple[Path, Path, int]] = []
    if needs_cgfa:
        cgfa_artefacts = _find_cgfa_full_artifacts(ablation_dir, case_study_player_deck)
        if not cgfa_artefacts:
            console.print(
                f"\n[yellow]No cgfa_full artefacts found under {ablation_dir} "
                f"for player deck '{case_study_player_deck}'. "
                "Skipping calibration plot and case study.[/]"
            )

    # ---- Stage 4: calibration plot (Fig. 6) -------------------------------
    if "calibration" in stages and cgfa_artefacts:
        print_divider("Stage 4: CGFA calibration plot")
        csv_paths = [csv for (_model, csv, _seed) in cgfa_artefacts]
        labels = [f"seed{seed}" for (_model, _csv, seed) in cgfa_artefacts]
        cal_out = ablation_dir / "figures" / "cgfa_calibration.png"
        cal_out.parent.mkdir(parents=True, exist_ok=True)
        render_calibration(
            csv_paths,
            cal_out,
            labels=labels,
            player_deck=case_study_player_deck,
        )
        console.print(f"  wrote [cyan]{cal_out}[/]  ({len(csv_paths)} seed(s))")

    # ---- Stage 5: case study (Fig. 7) -------------------------------------
    if "case-study" in stages and cgfa_artefacts:
        print_divider("Stage 5: CGFA case study")
        # Pick the lowest-seed model so re-runs of --only case-study are
        # deterministic across invocations.
        cgfa_artefacts.sort(key=lambda t: t[2])
        ckpt = cgfa_artefacts[0][0]
        cs_out = ablation_dir / "case_study"
        cs_argv = [
            "--model-path",
            str(ckpt),
            "--player-deck",
            case_study_player_deck,
            "--opponent-deck",
            case_study_opponent_deck,
            "--episode-seed",
            str(case_study_seed),
            "--output-dir",
            str(cs_out),
            "--max-turns",
            str(max_turns),
            "--agency",
            "auto" if agency_mode == "auto" else "full",
        ]
        rc = _dispatch_case_study(cs_argv)
        if rc != 0:
            console.print(
                f"[yellow]Case study returned non-zero exit ({rc}); "
                "subsequent stages are unaffected.[/]"
            )

    # ---- Stage 6: cross-source aggregate (paired-bootstrap roll-up) -------
    cross_source_dir: Path | None = None
    if "cross-source" in stages:
        print_divider("Stage 6: cross-source aggregate")
        discovered = _discover_eval_results(paper_root)
        if not discovered:
            console.print(
                f"[yellow]No eval_results.json files found under {paper_root}; "
                "skipping cross-source aggregate.[/]"
            )
        else:
            from scripts.research.aggregate import aggregate

            labels = [lbl for (lbl, _path) in discovered]
            paths = [path for (_lbl, path) in discovered]
            console.print(f"  discovered [cyan]{len(paths)}[/] eval_results.json files:")
            for lbl, path in discovered:
                console.print(f"    [bold]{lbl:<40}[/] {path}")
            cross_source_dir = paper_root / "cross_source"
            aggregate(
                eval_paths=paths,
                output_dir=cross_source_dir,
                baseline_agent=agents[0] if agents else None,
                source_labels=labels,
                headline_compare_agents=agents if len(agents) >= 2 else None,
                source_compare_pairs=None,
            )
            console.print(
                f"  unified roll-up: [cyan]{cross_source_dir}/[/]\n"
                f"    figures: [cyan]{cross_source_dir / 'figures'}[/]\n"
                f"    tables:  [cyan]{cross_source_dir / 'tables'}[/]"
            )

    elapsed = time.time() - overall_started
    print_divider(f"Paper reproduction complete ({format_duration(elapsed)})")
    console.print(
        f"  master:       [cyan]{paper_root}[/]\n"
        f"  headline:     [cyan]{paper_root / 'headline'}[/]\n"
        f"  ablation:     [cyan]{ablation_dir}[/]\n"
        f"  transfer:     [cyan]{paper_root / 'transfer'}[/]"
        + (f"\n  cross_source: [cyan]{cross_source_dir}[/]" if cross_source_dir is not None else "")
    )
    return 0


def _run_quick_smoke_test() -> int:
    """Tiny end-to-end smoke test (a few minutes)."""
    from scripts.research.aggregate import aggregate
    from scripts.research.eval_sweep import evaluate_sweep
    from scripts.research.train_sweep import SweepConfig, run_sweep

    print_divider("Smoke test")
    cfg = SweepConfig(
        experiment_name="smoke_test",
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
        output_root="results/research",
        sample_games=0,
        eval_episodes=10,
    )
    run_sweep(cfg, force=True)
    experiment_dir = cfg.experiment_dir()
    evaluate_sweep(
        experiment_dir=experiment_dir,
        n_episodes=10,
        include_baselines=True,
        baseline_overrides=["random"],
        extra_player_decks=[],
        max_turns=20,
        agency_mode="auto",
    )
    aggregate(
        eval_paths=[experiment_dir / "eval" / "eval_results.json"],
        output_dir=experiment_dir / "aggregated",
        baseline_agent="ppo",
        source_labels=None,
        headline_compare_agents=None,
    )
    console.print(f"\n[bold green]Smoke test complete[/] -> [cyan]{experiment_dir}[/]")
    return 0


def _run_preflight_signal(
    *,
    experiment_name: str,
    seeds: list[int],
    timesteps_per_opponent: int,
    eval_episodes: int,
    n_envs: int | str,
    output_root: str,
    reward_type: str,
    agency_mode: str,
    training_mode: str,
    max_turns: int,
    cgfa_calibration_mode: str,
    force: bool,
    dry_run: bool,
) -> int:
    """Run a small publication-budget signal test before the full paper sweep."""
    cells = [
        ("azorius_vs_dimir", "azorius_control", "dimir_midrange"),
        ("monored_mirror", "mono_red_aggro", "mono_red_aggro"),
    ]
    cgfa_kwargs: dict[str, object] = {}
    if cgfa_calibration_mode != "factual":
        cgfa_kwargs["calibration_mode"] = cgfa_calibration_mode

    print_logo()
    print_divider(f"CGFA pre-flight signal test: {experiment_name}")
    console.print(
        f"  cells:      [cyan]{', '.join(tag for tag, _p, _o in cells)}[/]\n"
        f"  seeds:      [cyan]{seeds}[/]\n"
        f"  budget:     [cyan]{timesteps_per_opponent:,}[/] steps / cell\n"
        f"  eval:       [cyan]{eval_episodes}[/] episodes / cell\n"
        f"  CGFA cal:   [cyan]{cgfa_calibration_mode}[/]\n"
        f"  output:     [cyan]{Path(output_root) / experiment_name}[/]"
    )
    if dry_run:
        for tag, player, opp in cells:
            console.print(f"  would run: {tag}: agents=ppo,cgfa deck={player} opponent={opp}")
        return 0

    for tag, player, opp in cells:
        print_divider(f"Pre-flight cell: {tag}")
        rc = _run_pipeline(
            experiment_name=f"{experiment_name}/{tag}",
            agents=["ppo", "cgfa"],
            player_decks=[player],
            seeds=seeds,
            opponents=[opp],
            timesteps_per_opponent=timesteps_per_opponent,
            training_mode=training_mode,
            agency_mode=agency_mode,
            n_envs=n_envs,
            reward_type=reward_type,
            max_turns=max_turns,
            output_root=output_root,
            eval_episodes=eval_episodes,
            include_baselines=False,
            baseline_overrides=None,
            extra_player_decks=[],
            baseline_agent="ppo",
            aggregate_dir=None,
            source_labels=None,
            agent_kwargs_by_agent={"cgfa": cgfa_kwargs} if cgfa_kwargs else None,
            force=force,
        )
        if rc != 0:
            return rc

    console.print(
        "\n[bold green]Pre-flight complete.[/] Inspect each "
        f"[cyan]{Path(output_root) / experiment_name / '<cell>' / 'aggregated'}[/] "
        "directory, especially tables/significance.tex and aggregated_results.json."
    )
    return 0


# ---------------------------------------------------------------------------
# Top-level argparse
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mtg-research",
        description=(
            "Composable research pipeline for MTG-Causal-RL. "
            "Train sweeps, high-fidelity evaluation, and paper-ready "
            "aggregation with paired-bootstrap significance tests."
        ),
    )
    p.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Launch the interactive configuration wizard and run the full pipeline.",
    )

    sub = p.add_subparsers(
        dest="command",
        metavar=(
            "{train,eval,aggregate,pipeline,ablation,calibration-plot,"
            "case-study,transfer,paper,preflight-signal}"
        ),
    )

    sub.add_parser(
        "train",
        add_help=False,
        help="Run Stage 1 (multi-seed training sweep). Forwards to train_sweep.",
    )
    sub.add_parser(
        "eval",
        add_help=False,
        help="Run Stage 2 (high-fidelity evaluation). Forwards to eval_sweep.",
    )
    sub.add_parser(
        "aggregate",
        add_help=False,
        help="Run Stage 3 (figures, LaTeX tables, significance tests).",
    )
    sub.add_parser(
        "ablation",
        add_help=False,
        help=(
            "Run the 6-point CGFA-PPO ablation suite end-to-end "
            "(train + eval + cross-variant aggregation)."
        ),
    )
    sub.add_parser(
        "calibration-plot",
        add_help=False,
        help=(
            "Render the CGFA intervention-calibration diagnostic plot "
            "from one or more cgfa_calibration.csv files."
        ),
    )
    sub.add_parser(
        "case-study",
        add_help=False,
        help=(
            "Run a deterministic CGFA episode and produce a per-step "
            "factor-attribution table + figure."
        ),
    )
    sub.add_parser(
        "transfer",
        add_help=False,
        help=(
            "Train on K opponents and evaluate on a disjoint held-out set "
            "to quantify transfer / generalisation."
        ),
    )

    preflight = sub.add_parser(
        "preflight-signal",
        help="Run two publication-budget PPO-vs-CGFA cells before the full paper sweep.",
    )
    preflight.add_argument("--experiment-name", default=None)
    preflight.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1024])
    preflight.add_argument("--timesteps-per-opponent", type=int, default=1_000_000)
    preflight.add_argument("--eval-episodes", type=int, default=300)
    preflight.add_argument("--n-envs", default="auto", help="int or 'auto' (CPU - 1)")
    preflight.add_argument("--output-root", default="results/research")
    preflight.add_argument("--reward-type", choices=["sparse", "shaped", "dense"], default="shaped")
    preflight.add_argument("--agency", choices=["auto", "full", "curriculum"], default="auto")
    preflight.add_argument(
        "--training-mode", choices=["round-robin", "sequential"], default="round-robin"
    )
    preflight.add_argument("--max-turns", type=int, default=20)
    preflight.add_argument(
        "--cgfa-calibration-mode",
        choices=["factual", "interventional"],
        default="interventional",
        help=(
            "CGFA calibration target for this signal test. 'interventional' is "
            "experimental but keeps PPO budget/opponents identical."
        ),
    )
    preflight.add_argument("--force", action="store_true")
    preflight.add_argument("--dry-run", action="store_true")

    pipe = sub.add_parser(
        "pipeline",
        help="Run Stages 1+2+3 end-to-end for a single (agents, decks, seeds) config.",
    )
    pipe.add_argument("--experiment-name", default=None)
    pipe.add_argument("--agents", nargs="+", default=["ppo"], choices=ALL_AGENTS)
    pipe.add_argument("--player-decks", nargs="+", default=["mono_red_aggro"], choices=ALL_DECKS)
    pipe.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    pipe.add_argument("--opponents", nargs="+", default=ALL_DECKS, choices=ALL_DECKS)
    pipe.add_argument("--timesteps-per-opponent", type=int, default=2_000_000)
    pipe.add_argument(
        "--training-mode", choices=["round-robin", "sequential"], default="round-robin"
    )
    pipe.add_argument("--agency", choices=["auto", "full", "curriculum"], default="auto")
    pipe.add_argument("--n-envs", default="auto", help="int or 'auto' (CPU - 1)")
    pipe.add_argument("--reward-type", choices=["sparse", "shaped", "dense"], default="shaped")
    pipe.add_argument("--max-turns", type=int, default=20)
    pipe.add_argument("--output-root", default="results/research")
    pipe.add_argument("--eval-episodes", type=int, default=500)
    pipe.add_argument(
        "--no-baselines",
        action="store_true",
        help="Skip baseline evaluation entirely (Stage 2 only scores trained models).",
    )
    pipe.add_argument(
        "--baseline-agents",
        nargs="*",
        default=None,
        help=(
            "Advanced override: explicit baseline agent names to apply to every "
            "deck. Default (omitted) auto-pairs `random` + the canonical "
            "deck-matched heuristic from mtg.agents.DECK_TO_HEURISTIC."
        ),
    )
    pipe.add_argument("--extra-player-decks", nargs="*", default=[])
    pipe.add_argument("--baseline-agent", default=None)
    pipe.add_argument("--aggregate-dir", type=Path, default=None)
    pipe.add_argument("--source-labels", nargs="*", default=None)
    pipe.add_argument("--force", action="store_true", help="Re-train even if a model exists.")

    # ---- ``paper`` subcommand ---------------------------------------------
    # Single-command reproduction of every paper figure (3-7).
    paper = sub.add_parser(
        "paper",
        help=(
            "Reproduce every paper figure (Figs. 3-7) in one command: "
            "headline + 6-point ablation + held-out transfer + "
            "calibration plot + case study."
        ),
        description=(
            "Run the full paper-reproduction pipeline end-to-end.  Each stage "
            "writes its artefacts under <output-root>/<experiment-name>_<stage>/. "
            "The default budget reproduces the 'lite' recipe in the README; "
            "raise --timesteps-per-opponent to 2000000 and --eval-episodes to "
            "500 to reproduce the paper-grade tables."
        ),
    )
    paper.add_argument(
        "--experiment-name",
        default=None,
        help=("Common prefix for all sub-experiment names " "(default: paper_<timestamp>)."),
    )
    paper.add_argument(
        "--agents",
        nargs="+",
        default=["ppo", "cgfa"],
        choices=ALL_AGENTS,
        help="Agents trained in the headline + transfer stages.",
    )
    paper.add_argument(
        "--player-decks",
        nargs="+",
        default=ALL_DECKS,
        choices=ALL_DECKS,
        help="Player decks for the headline pipeline (Fig. 3). Defaults to all deck archetypes.",
    )
    paper.add_argument(
        "--ablation-decks",
        nargs="+",
        default=None,
        choices=ALL_DECKS,
        help=(
            "Player decks for the ablation suite (Fig. 4).  "
            "Defaults to all --player-decks entries."
        ),
    )
    paper.add_argument(
        "--transfer-decks",
        nargs="+",
        default=None,
        choices=ALL_DECKS,
        help=(
            "Player decks for the transfer experiment (Fig. 5).  "
            "Defaults to all --player-decks entries."
        ),
    )
    paper.add_argument(
        "--opponents",
        nargs="+",
        default=["mono_red_aggro", "dimir_midrange", "domain_ramp"],
        choices=ALL_DECKS,
        help="Training-time opponent set (shared by headline, ablation, transfer).",
    )
    paper.add_argument(
        "--heldout-opponents",
        nargs="+",
        default=["azorius_control", "boros_convoke"],
        choices=ALL_DECKS,
        help=(
            "Disjoint held-out opponent set evaluated only by the transfer stage. "
            "Ignored when --transfer-mode=leave-one-out (the LOO loop derives "
            "its splits from the union of --opponents and --heldout-opponents)."
        ),
    )
    paper.add_argument(
        "--transfer-mode",
        choices=TRANSFER_MODES,
        default="leave-one-out",
        help=(
            "Transfer-stage protocol. 'fixed' trains once on "
            "--opponents and evaluates against --heldout-opponents. "
            "'leave-one-out' (default) loops over the full opponent pool, "
            "holding out one deck per fold and training on the remaining N-1; "
            "this is more rigorous (no cherry-picked split) but trains "
            "~N*(N-1)/|--opponents| times more rollouts."
        ),
    )
    paper.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456, 789, 1024, 2048, 4096],
    )
    paper.add_argument("--timesteps-per-opponent", type=int, default=1_000_000)
    paper.add_argument("--eval-episodes", type=int, default=300)
    paper.add_argument(
        "--training-mode", choices=["round-robin", "sequential"], default="round-robin"
    )
    paper.add_argument("--agency", choices=["auto", "full", "curriculum"], default="auto")
    paper.add_argument("--n-envs", default="auto", help="int or 'auto' (CPU - 1)")
    paper.add_argument("--reward-type", choices=["sparse", "shaped", "dense"], default="shaped")
    paper.add_argument("--max-turns", type=int, default=20)
    paper.add_argument("--output-root", default="results/research")
    paper.add_argument(
        "--cgfa-calibration-mode",
        choices=["factual", "interventional"],
        default="interventional",
        help=(
            "CGFA factor-epsilon target mode for headline, ablation, and transfer "
            "CGFA runs. Keep this matched to preflight-signal."
        ),
    )
    paper.add_argument(
        "--no-baselines",
        action="store_true",
        help="Skip random + heuristic baselines in the headline + ablation eval passes.",
    )
    paper.add_argument(
        "--case-study-player-deck",
        default=None,
        choices=ALL_DECKS,
        help="Player deck for the case study (default: first --ablation-decks entry).",
    )
    paper.add_argument(
        "--case-study-opponent-deck",
        default="azorius_control",
        choices=ALL_DECKS,
        help="Opponent deck for the case study.",
    )
    paper.add_argument(
        "--case-study-seed",
        type=int,
        default=7,
        help="Episode seed for the case study rollout (independent of training seeds).",
    )
    paper.add_argument(
        "--only",
        nargs="+",
        default=None,
        choices=PAPER_STAGES,
        help="Run only these stages (mutually exclusive with --skip).",
    )
    paper.add_argument(
        "--skip",
        nargs="+",
        default=None,
        choices=PAPER_STAGES,
        help="Skip these stages (mutually exclusive with --only).",
    )
    paper.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands and exit without launching anything.",
    )
    paper.add_argument(
        "--force",
        action="store_true",
        help="Re-train models even if a saved checkpoint already exists.",
    )

    return p


def main() -> int:
    """Entry point for the ``mtg-research`` console script."""
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    parser = _build_parser()

    # ``train|eval|aggregate|ablation`` forward the REST of argv verbatim to
    # their existing argparse parsers. We strip only the leading subcommand token.
    argv = sys.argv[1:]
    if argv and argv[0] in {
        "train",
        "eval",
        "aggregate",
        "ablation",
        "calibration-plot",
        "case-study",
        "transfer",
    }:
        sub = argv[0]
        rest = argv[1:]
        dispatchers = {
            "train": _dispatch_train,
            "eval": _dispatch_eval,
            "aggregate": _dispatch_aggregate,
            "ablation": _dispatch_ablation,
            "calibration-plot": _dispatch_calibration_plot,
            "case-study": _dispatch_case_study,
            "transfer": _dispatch_transfer,
        }
        return dispatchers[sub](rest)

    args = parser.parse_args(argv)

    if args.interactive:
        return interactive_wizard()

    if args.command == "preflight-signal":
        experiment_name = (
            args.experiment_name
            or f"preflight_cgfa_signal_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )
        return _run_preflight_signal(
            experiment_name=experiment_name,
            seeds=args.seeds,
            timesteps_per_opponent=args.timesteps_per_opponent,
            eval_episodes=args.eval_episodes,
            n_envs=args.n_envs,
            output_root=args.output_root,
            reward_type=args.reward_type,
            agency_mode=args.agency,
            training_mode=args.training_mode,
            max_turns=args.max_turns,
            cgfa_calibration_mode=args.cgfa_calibration_mode,
            force=args.force,
            dry_run=args.dry_run,
        )

    if args.command == "pipeline":
        experiment_name = (
            args.experiment_name or f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        return _run_pipeline(
            experiment_name=experiment_name,
            agents=args.agents,
            player_decks=args.player_decks,
            seeds=args.seeds,
            opponents=args.opponents,
            timesteps_per_opponent=args.timesteps_per_opponent,
            training_mode=args.training_mode,
            agency_mode=args.agency,
            n_envs=args.n_envs,
            reward_type=args.reward_type,
            max_turns=args.max_turns,
            output_root=args.output_root,
            eval_episodes=args.eval_episodes,
            include_baselines=not args.no_baselines,
            baseline_overrides=args.baseline_agents,
            extra_player_decks=args.extra_player_decks,
            baseline_agent=args.baseline_agent,
            aggregate_dir=args.aggregate_dir,
            source_labels=args.source_labels,
            agent_kwargs_by_agent=None,
            force=args.force,
        )

    if args.command == "paper":
        experiment_name = (
            args.experiment_name or f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        ablation_decks = args.ablation_decks or list(args.player_decks)
        transfer_decks = args.transfer_decks or list(args.player_decks)
        case_study_player_deck = args.case_study_player_deck or ablation_decks[0]
        try:
            return _run_paper(
                experiment_name=experiment_name,
                agents=args.agents,
                player_decks=args.player_decks,
                ablation_decks=ablation_decks,
                transfer_decks=transfer_decks,
                opponents=args.opponents,
                heldout_opponents=args.heldout_opponents,
                seeds=args.seeds,
                timesteps_per_opponent=args.timesteps_per_opponent,
                eval_episodes=args.eval_episodes,
                training_mode=args.training_mode,
                agency_mode=args.agency,
                n_envs=args.n_envs,
                reward_type=args.reward_type,
                max_turns=args.max_turns,
                output_root=args.output_root,
                case_study_player_deck=case_study_player_deck,
                case_study_opponent_deck=args.case_study_opponent_deck,
                case_study_seed=args.case_study_seed,
                include_baselines=not args.no_baselines,
                transfer_mode=args.transfer_mode,
                cgfa_calibration_mode=args.cgfa_calibration_mode,
                only=args.only,
                skip=args.skip,
                force=args.force,
                dry_run=args.dry_run,
            )
        except ValueError as exc:
            console.print(f"[bold red]error:[/bold red] {exc}")
            return 2

    # No subcommand + no -i => show help.
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

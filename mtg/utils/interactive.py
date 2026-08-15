"""Interactive CLI prompts for MTG-Causal-RL workflows.

This module provides Rich-based interactive prompts for selecting agents,
deck archetypes, and configuration options across all CLI workflows.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from rich import box
from rich.console import Console
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

console = Console()


# =============================================================================
# Data Classes for Configuration
# =============================================================================


@dataclass
class TrainingConfig:
    """Configuration for training workflow.

    Attributes:
        agent_type: Type of agent to train ('ppo' or 'causal').
        player_deck: Agent's deck archetype.
        opponent_deck: Opponent deck(s), comma-separated for multi-opponent.
        timesteps: Total training timesteps.
        reward_type: Reward shaping type.
        seed: Random seed.
        eval_episodes: Final post-training evaluation episodes per opponent
            (this is the official, statistically meaningful number).
        quick_eval_episodes: In-line "Quick evaluation" episodes per opponent
            shown right after training, used as a fast sanity check. Runs
            deterministically; small samples (default 20) can show wide swings
            from the training-time win rate.
        output_dir: Output directory for artifacts.

    """

    agent_type: str
    player_deck: str
    opponent_deck: str
    timesteps: int = 1_000_000
    reward_type: str = "shaped"
    seed: int = 42
    max_turns: int = 20
    n_envs: int = 4
    training_mode: str = "round-robin"  # "round-robin" or "sequential"
    agency_mode: str = "auto"  # "auto", "full", or "curriculum"
    eval_episodes: int = 100
    quick_eval_episodes: int = 20
    sample_games: int = 3
    sample_opponents: str = ""  # comma-separated; empty = same as training opponents
    output_dir: str = "results/trained_agents"
    # Per-variant agent constructor kwargs.  Used by the ablation
    # runner to inject CGFA-specific switches (``learnable_gate``,
    # ``intervention_calibration_coef`` etc.) without polluting the CLI.
    agent_kwargs: dict[str, tp.Any] = field(default_factory=dict)

    @property
    def opponent_decks(self) -> list[str]:
        """Get list of opponent decks (supports comma-separated multi-opponent)."""
        return [d.strip() for d in self.opponent_deck.split(",")]

    @property
    def is_multi_opponent(self) -> bool:
        """Whether training against multiple opponents."""
        return len(self.opponent_decks) > 1

    @property
    def auto_combat(self) -> bool:
        """Whether combat is auto-resolved (all-or-nothing)."""
        return self.agency_mode == "auto"

    @property
    def auto_target(self) -> bool:
        """Whether spell targeting is auto-resolved."""
        return self.agency_mode == "auto"

    @property
    def sample_opponent_decks(self) -> list[str]:
        """Get list of opponents to generate sample reports against.

        Falls back to training opponents if not explicitly set.
        """
        if self.sample_opponents.strip():
            return [d.strip() for d in self.sample_opponents.split(",")]
        return self.opponent_decks

    def get_model_name(self) -> str:
        """Generate a clean model name for saving/loading.

        Convention: {agent_type}_{player_deck}
        e.g., ppo_mono_red_aggro, causal_azorius_control

        Returns:
            Clean model name.

        """
        return f"{self.agent_type}_{self.player_deck}"

    def get_run_name(self) -> str:
        """Generate a unique run name for the training run directory.

        Returns:
            Formatted run name with timestamp.

        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.is_multi_opponent:
            return f"{self.agent_type}_{self.player_deck}_vs_multi_{timestamp}"
        return f"{self.agent_type}_{self.player_deck}_vs_{self.opponent_deck}_{timestamp}"

    def to_engine_config(
        self,
        experiment_name: str | None = None,
        output_dir: str | None = None,
        **overrides: tp.Any,
    ) -> tp.Any:
        """Convert this CLI-facing config into a canonical ``TrainingConfig``.

        The engine ``TrainingConfig`` (``mtg.training.TrainingConfig``) is
        the single source of truth consumed by ``Trainer``.  The CLI
        config holds a few additional fields (``sample_games``,
        ``sample_opponents``, ``quick_eval_episodes``) that are only
        meaningful for the interactive workflow; those are filtered out
        here.  For multi-opponent runs (comma-separated
        ``opponent_deck``) the first opponent is used as the
        ``opponent_archetype`` and the full list is passed to the
        league via ``league_opponents``.  Callers can enable/disable
        league training explicitly via ``enable_league=...`` overrides.
        """
        from mtg.training.train import TrainingConfig as EngineTrainingConfig

        base_dir = output_dir or self.output_dir
        run_name = experiment_name or self.get_run_name()

        opponents = self.opponent_decks
        primary_opponent = opponents[0] if opponents else None
        enable_league = bool(overrides.pop("enable_league", len(opponents) > 1))

        engine_kwargs: dict[str, tp.Any] = {
            "agent_type": self.agent_type,
            "agent_kwargs": dict(self.agent_kwargs),
            "deck_archetype": self.player_deck,
            "opponent_archetype": primary_opponent,
            "reward_type": self.reward_type,
            "max_turns": self.max_turns,
            "total_timesteps": self.timesteps,
            "n_envs": self.n_envs,
            "seed": self.seed,
            "auto_combat": self.auto_combat,
            "auto_target": self.auto_target,
            "experiment_name": run_name,
            "log_dir": f"{base_dir}/{run_name}/logs",
            "checkpoint_dir": f"{base_dir}/{run_name}/checkpoints",
            "enable_league": enable_league,
            "league_opponents": list(opponents) if enable_league else [],
            "eval_episodes": self.quick_eval_episodes,
        }
        engine_kwargs.update(overrides)
        return EngineTrainingConfig(**engine_kwargs)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation workflow.

    Attributes:
        agent_type: Type of agent to evaluate.
        model_path: Path to trained model (if applicable).
        player_deck: Player's deck archetype.
        opponent_deck: Opponent deck(s), comma-separated for multi-opponent eval.
        episodes: Number of evaluation episodes.
        max_turns: Maximum turns per game.
        seeds: Random seeds for evaluation.
        save_reports: Whether to save HTML reports.
        show_games: Number of games to visualize per opponent.
        show_games_opponents: Opponents for visualized games
            (comma-sep; empty = all eval opponents).
        output_dir: Output directory for results.

    """

    agent_type: str
    model_path: str | None = None
    player_deck: str = "mono_red_aggro"
    opponent_deck: str = "azorius_control"
    episodes: int = 500
    max_turns: int = 10
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456, 789, 1000])
    save_reports: bool = False
    show_games: int = 0
    show_games_opponents: str = ""  # comma-separated; empty = same as eval opponents
    output_dir: str = "results/evaluations"
    verbose: bool = False
    # Path to ``vec_normalize.pkl`` from the training run. When
    # provided, the evaluator applies the SAME frozen observation
    # normalisation the policy was trained with, eliminating any
    # train/eval distribution shift.
    vec_normalize_path: str | None = None

    @property
    def opponent_decks(self) -> list[str]:
        """Get list of opponent decks (supports comma-separated multi-opponent)."""
        return [d.strip() for d in self.opponent_deck.split(",")]

    @property
    def is_multi_opponent(self) -> bool:
        """Whether evaluating against multiple opponents."""
        return len(self.opponent_decks) > 1

    @property
    def show_games_opponent_decks(self) -> list[str]:
        """Get opponents for visualized/reported games.

        Falls back to eval opponents if not explicitly set.
        """
        if self.show_games_opponents.strip():
            return [d.strip() for d in self.show_games_opponents.split(",")]
        return self.opponent_decks

    def get_run_name(self) -> str:
        """Generate a unique run name.

        Returns:
            Formatted run name with timestamp.

        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.is_multi_opponent:
            return f"eval_{self.agent_type}_{self.player_deck}_vs_multi_{timestamp}"
        return f"eval_{self.agent_type}_{self.player_deck}_vs_{self.opponent_decks[0]}_{timestamp}"

    def to_engine_config(self, **overrides: tp.Any) -> tp.Any:
        """Convert to a canonical ``mtg.training.EvaluationConfig``."""
        from mtg.training.evaluate import EvaluationConfig as EngineEvalConfig

        primary_opp = self.opponent_decks[0] if self.opponent_decks else None
        engine_kwargs: dict[str, tp.Any] = {
            "deck_archetype": self.player_deck,
            "opponent_archetype": primary_opp,
            "n_episodes": self.episodes,
            "max_turns": self.max_turns,
            "output_dir": self.output_dir,
        }
        if self.seeds:
            engine_kwargs["seed"] = int(self.seeds[0])
        engine_kwargs.update(overrides)
        return EngineEvalConfig(**engine_kwargs)


@dataclass
class GameplayConfig:
    """Configuration for gameplay workflow.

    Attributes:
        player_agent: Agent type for player ('greedy_aggro', 'control', 'ppo', 'causal', etc.).
        opponent_agent: Agent type for opponent (auto-assigned or specified).
        player_model_path: Path to trained model for player (if RL agent).
        opponent_model_path: Path to trained model for opponent (if RL agent).
        player_deck: Player's deck archetype.
        opponent_deck: Opponent's deck archetype.
        num_turns: Maximum turns to play.
        speed: Visualization speed preset ('slow', 'medium', 'fast').
        save_report: Whether to save HTML report.
        seed: Random seed for reproducibility (None for random).
        is_demo: Whether this is demo mode (fixed decks + reproducible).

    """

    player_agent: str = "greedy_aggro"
    opponent_agent: str | None = None  # Auto-assigned if None
    player_model_path: str | None = None
    opponent_model_path: str | None = None
    player_deck: str = "mono_red_aggro"
    opponent_deck: str = "azorius_control"
    num_turns: int = 5
    speed: str = "medium"
    save_report: bool = True
    seed: int | None = None  # None = random, fixed value = reproducible
    is_demo: bool = False


# =============================================================================
# Discovery Functions
# =============================================================================


def discover_trained_models(base_dir: str = "results/trained_agents") -> list[dict[str, tp.Any]]:
    """Discover trained models in the specified directory.

    Args:
        base_dir: Base directory to search for models.

    Returns:
        List of model info dictionaries with name, path, and metadata.

    """
    models: list[dict[str, tp.Any]] = []
    base_path = Path(base_dir)

    if not base_path.exists():
        return models

    for run_dir in sorted(base_path.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue

        # Look for model files
        model_files = list(run_dir.glob("*.zip")) + list(run_dir.glob("model.*"))
        config_file = run_dir / "config.yaml"

        if model_files:
            model_info: dict[str, tp.Any] = {
                "name": run_dir.name,
                "path": str(model_files[0]),
                "dir": str(run_dir),
                "has_config": config_file.exists(),
            }

            # Try to load config for accurate metadata
            if config_file.exists():
                try:
                    import yaml

                    with open(config_file) as f:
                        cfg = yaml.safe_load(f)
                    model_info["agent_type"] = cfg.get("agent_type", "unknown")
                    model_info["player_deck"] = cfg.get("player_deck", "unknown")
                    model_info["opponent_deck"] = cfg.get("opponent_deck", "unknown")
                except Exception:
                    pass

            # Fallback: parse run name for agent/deck info
            if "agent_type" not in model_info:
                parts = run_dir.name.split("_")
                if len(parts) >= 4:
                    model_info["agent_type"] = parts[0]
                    try:
                        vs_idx = parts.index("vs")
                        model_info["player_deck"] = "_".join(parts[1:vs_idx])
                        model_info["opponent_deck"] = "_".join(parts[vs_idx + 1 : -1])
                    except ValueError:
                        pass

            models.append(model_info)

    return models


def find_model_for_deck(
    agent_type: str,
    player_deck: str,
    base_dir: str = "results/trained_agents",
) -> str | None:
    """Find the latest trained model for a specific agent type and deck.

    Args:
        agent_type: Agent type ('ppo' or 'causal').
        player_deck: Deck archetype name.
        base_dir: Base directory for trained models.

    Returns:
        Path to model file, or None if not found.

    """
    models = discover_trained_models(base_dir)
    for model in models:
        if model.get("agent_type") == agent_type and model.get("player_deck") == player_deck:
            return model["path"]
    return None


def get_available_agents() -> list[str]:
    """Get list of available agent types.

    Returns:
        List of registered agent names.

    """
    from mtg.agents import list_agents

    return list_agents()


def get_available_archetypes() -> list[str]:
    """Get list of available deck archetypes.

    Returns:
        List of registered archetype names.

    """
    from mtg.env.deck_archetypes import list_archetypes

    return list_archetypes()


def get_archetype_info(name: str) -> dict[str, tp.Any]:
    """Get detailed info about an archetype.

    Args:
        name: Archetype name.

    Returns:
        Dictionary with archetype details.

    """
    from mtg.env.deck_archetypes import get_archetype

    try:
        archetype = get_archetype(name)
        return {
            "name": archetype.name,
            "display_name": archetype.display_name,
            "description": archetype.description,
            "strategy": archetype.strategy.name,
            "colors": archetype.colors,
            "tier": archetype.tier,
            "land_count": archetype.get_land_count(),
        }
    except KeyError:
        return {}


# =============================================================================
# Interactive Prompts
# =============================================================================


def print_section_header(title: str) -> None:
    """Print a styled section header.

    Args:
        title: Section title.

    """
    console.print()
    console.rule(f"[bold cyan]{title}[/bold cyan]")
    console.print()


def prompt_agent_selection(
    prompt_text: str = "Select agent type",
    include_trained: bool = True,
    trained_models: list[dict[str, tp.Any]] | None = None,
) -> tuple[str, str | None]:
    """Prompt user to select an agent type.

    Args:
        prompt_text: Prompt message.
        include_trained: Whether to include trained models as options.
        trained_models: Pre-loaded trained models list.

    Returns:
        Tuple of (agent_type, model_path or None).

    """
    agents = get_available_agents()
    # In workflows that include trained-model discovery (evaluation/gameplay),
    # hide untrained PPO/Causal placeholders from the base list to reduce noise.
    if include_trained:
        agents = [a for a in agents if a not in {"ppo", "causal"}]

    # Build options table
    table = Table(title="Available Agents", box=box.ROUNDED)
    table.add_column("#", style="cyan", width=3)
    table.add_column("Agent", style="green")
    table.add_column("Status", style="yellow", width=12)
    table.add_column("Description", style="dim")

    # Descriptions with training requirement info
    agent_info = {
        "random": ("Ready", "Uniform random action selection"),
        "greedy_aggro": (
            "Ready",
            "Heuristic aggro baseline (fast curve, face-pressure decisions)",
        ),
        "control": (
            "Ready",
            "Heuristic control baseline (hold interaction, value/timing focused)",
        ),
        "midrange": (
            "Ready",
            "Heuristic midrange baseline (balanced board control and pressure)",
        ),
        "ramp": (
            "Ready",
            "Heuristic ramp baseline (mana acceleration into high-impact plays)",
        ),
        "convoke_aggro": (
            "Ready",
            "Heuristic convoke baseline (token-wide pressure and efficient convoke)",
        ),
        "ppo": ("Needs training", "PPO agent - train first for best results"),
        "causal": ("Needs training", "Causal RL agent - train first for best results"),
    }

    for i, agent in enumerate(agents, 1):
        status, desc = agent_info.get(agent, ("Custom", "Custom agent"))
        table.add_row(str(i), agent, status, desc)

    # Add trained models if requested
    model_offset = len(agents)
    models = trained_models or []

    if include_trained and not models:
        models = discover_trained_models()

    if models:
        table.add_row("", "", "", "")  # Separator
        table.add_row("", "[bold]Trained Models[/bold]", "", "")

        for i, model in enumerate(models[:10], model_offset + 1):
            deck_info = f"{model.get('player_deck', '?')} vs {model.get('opponent_deck', '?')}"
            table.add_row(
                str(i),
                f"[yellow]{model['name'][:30]}[/yellow]",
                "Trained",
                deck_info,
            )
    elif include_trained:
        console.print("\n[dim]No trained models found. Train agents using:[/dim]")
        console.print("[dim]  uv run python scripts/runner/run_training.py --interactive[/dim]\n")

    console.print(table)
    console.print()

    max_choice = len(agents) + len(models)
    choice = IntPrompt.ask(
        f"{prompt_text} (1-{max_choice})",
        default=2,  # Default to greedy_aggro
    )

    if 1 <= choice <= len(agents):
        return agents[choice - 1], None
    elif choice <= max_choice:
        model = models[choice - len(agents) - 1]
        return model.get("agent_type", "ppo"), model["path"]
    else:
        console.print("[red]Invalid choice, using greedy_aggro agent[/red]")
        return "greedy_aggro", None


def prompt_deck_selection(
    prompt_text: str = "Select deck archetype",
    default: str | None = None,
) -> str:
    """Prompt user to select a deck archetype.

    Args:
        prompt_text: Prompt message.
        default: Default selection.

    Returns:
        Selected archetype name.

    """
    archetypes = get_available_archetypes()

    # Build options table
    table = Table(title="Available Deck Archetypes", box=box.ROUNDED)
    table.add_column("#", style="cyan", width=3)
    table.add_column("Archetype", style="green")
    table.add_column("Strategy", style="yellow")
    table.add_column("Colors", style="magenta")
    table.add_column("Description", style="dim", max_width=40)

    for i, name in enumerate(archetypes, 1):
        info = get_archetype_info(name)
        colors = " ".join(info.get("colors", []))
        desc = info.get("description", "")[:40] + "..."
        table.add_row(
            str(i),
            info.get("display_name", name),
            info.get("strategy", ""),
            colors,
            desc,
        )

    console.print(table)
    console.print()

    default_idx = archetypes.index(default) + 1 if default in archetypes else 1
    choice = IntPrompt.ask(
        f"{prompt_text} (1-{len(archetypes)})",
        default=default_idx,
    )

    if 1 <= choice <= len(archetypes):
        return archetypes[choice - 1]
    else:
        console.print(f"[red]Invalid choice, using {archetypes[0]}[/red]")
        return archetypes[0]


def prompt_trainable_agent_selection() -> str:
    """Prompt user to select a trainable agent (PPO or Causal only).

    Returns:
        Selected agent type string.

    """
    trainable_agents = [
        ("ppo", "PPO (Proximal Policy Optimization)", "MaskablePPO with action masking"),
        ("causal", "Causal RL", "PPO + SCM counterfactual reasoning"),
    ]

    table = Table(title="Trainable Agents", box=box.ROUNDED)
    table.add_column("#", style="cyan", width=3)
    table.add_column("Agent", style="green")
    table.add_column("Description", style="dim")

    for i, (_, name, desc) in enumerate(trainable_agents, 1):
        table.add_row(str(i), name, desc)

    console.print(table)
    console.print()

    choice = IntPrompt.ask(
        "Select agent to train (1-2)",
        default=1,
    )

    if 1 <= choice <= len(trainable_agents):
        return trainable_agents[choice - 1][0]
    console.print("[red]Invalid choice, defaulting to PPO[/red]")
    return "ppo"


def prompt_opponent_strategy() -> tuple[str, list[str]]:
    """Prompt user for opponent training strategy.

    Returns:
        Tuple of (strategy_name, list_of_opponent_decks).

    """
    archetypes = get_available_archetypes()

    console.print("[bold]Opponent Training Strategy[/bold]")
    console.print(
        "  1. [green]All Opponents[/green] - Train against all deck archetypes "
        "(recommended for robust agent)"
    )
    console.print(
        "  2. [yellow]Single Opponent[/yellow] - Train against one specific deck "
        "(for transfer learning / ablation)"
    )
    console.print()

    choice = IntPrompt.ask("Select strategy (1-2)", default=1)

    if choice == 2:
        opponent_deck = prompt_deck_selection(
            prompt_text="Select opponent deck",
            default="azorius_control",
        )
        return "single", [opponent_deck]
    else:
        console.print(
            f"[dim]Training against all {len(archetypes)} archetypes: {', '.join(archetypes)}[/dim]"
        )
        return "all", archetypes


def prompt_training_config() -> TrainingConfig:
    """Interactively prompt for training configuration.

    Returns:
        Complete TrainingConfig.

    """
    print_section_header("Training Configuration")

    # Agent selection - only trainable agents
    console.print("[bold]Step 1: Select Agent to Train[/bold]")
    agent_type = prompt_trainable_agent_selection()

    # Deck selection
    console.print("\n[bold]Step 2: Select Agent Deck[/bold]")
    console.print("[dim]The agent will be trained to play this deck[/dim]\n")
    player_deck = prompt_deck_selection(
        prompt_text="Select agent deck",
        default="mono_red_aggro",
    )

    console.print("\n[bold]Step 3: Select Opponent Strategy[/bold]")
    opponent_strategy, opponent_decks = prompt_opponent_strategy()

    # For the config, store the first opponent; training loop will iterate
    # Store all opponents in a comma-separated string or use the first
    opponent_deck = ",".join(opponent_decks)

    # Training mode selection (only relevant for multi-opponent)
    training_mode = "round-robin"
    if len(opponent_decks) > 1:
        console.print("\n[bold]Training Mode:[/bold]")
        console.print(
            "  [cyan]1[/] [green]Round-Robin[/green] (recommended) "
            "- Interleave opponents for balanced learning"
        )
        console.print(
            "  [cyan]2[/] [yellow]Sequential[/yellow] "
            "- Train full budget against each opponent in order"
        )
        mode_choice = IntPrompt.ask("Select training mode (1-2)", default=1)
        training_mode = "sequential" if mode_choice == 2 else "round-robin"

    # Agent agency settings
    console.print("\n[bold]Step 4: Agent Decision Agency[/bold]")
    console.print(
        "  [dim]Controls how much strategic freedom the agent has over combat "
        "and spell targeting decisions.[/dim]\n"
    )
    console.print(
        "  [cyan]1[/] [green]Auto[/green] [dim](recommended)[/dim]\n"
        "     Combat is all-or-nothing, spell targets are auto-picked by heuristics.\n"
        "     Simplest action space, learns fastest. Best for budgets under 2M steps.\n"
    )
    console.print(
        "  [cyan]2[/] [green]Curriculum[/green]\n"
        "     Starts with Auto for 70% of the budget to learn basic strategy,\n"
        "     then switches to Full for the remaining 30% to fine-tune combat\n"
        "     and targeting decisions. Designed for the Causal RL agent, which\n"
        "     leverages learned causal effects to handle the expanded action space.\n"
        "     Not recommended for vanilla PPO (use Auto instead).\n"
    )
    console.print(
        "  [cyan]3[/] [yellow]Full[/yellow]\n"
        "     Agent selects individual attackers and picks spell targets from\n"
        "     scratch. Richest decision space but needs 3-5M+ steps per opponent\n"
        "     to learn effectively. Use for long training runs only.\n"
    )
    agency_choices = {1: "auto", 2: "curriculum", 3: "full"}
    agency_choice = IntPrompt.ask("Select agency mode (1-3)", default=1)
    agency_mode = agency_choices.get(agency_choice, "auto")

    # Training parameters
    console.print("\n[bold]Step 5: Training Parameters[/bold]")

    console.print("  [dim]Recommended: 500K (quick test), 1M (standard), 2M+ (paper-quality)[/dim]")
    timesteps = IntPrompt.ask(
        "Total training timesteps",
        default=1_000_000,
    )

    console.print("\n[bold]Max turns per game:[/bold]")
    console.print(
        "  [dim]How many MTG turns before a game ends in a draw. More turns = "
        "longer games, richer strategy, but slower training.[/dim]"
    )
    console.print(
        "  [dim]Recommendation: 5 (fast iteration), 10 (quick test), 20 (standard, default)[/dim]"
    )
    max_turns = IntPrompt.ask("Max turns per game", default=20)

    reward_options = ["sparse", "shaped", "dense"]
    console.print("\n[bold]Reward types:[/bold]")
    console.print(
        "  [cyan]1[/] [bold]sparse[/bold]   : +1 for win, -1 for loss (cleanest RL signal, "
        "slower to learn)"
    )
    console.print(
        "  [cyan]2[/] [bold]shaped[/bold]  : intermediate rewards for life changes, board "
        "advantage, card draw [dim](recommended)[/dim]"
    )
    console.print(
        "  [cyan]3[/] [bold]dense[/bold]    : per-step rewards for every game event "
        "(fastest learning, risk of reward hacking)"
    )
    reward_choice = IntPrompt.ask("Select reward type (1-3)", default=2)
    reward_type = reward_options[min(reward_choice - 1, 2)]

    import os

    cpu_count = os.cpu_count() or 4
    console.print("\n[bold]Parallel environments:[/bold]")
    console.print(
        f"  [dim]More environments = faster training via parallel rollout "
        f"collection. Use 0 or 'auto' for all available cores ({cpu_count}), "
        f"or 1 to disable parallelism.[/dim]"
    )
    n_envs_input = IntPrompt.ask("Number of parallel environments (0 = auto)", default=0)
    n_envs = cpu_count if n_envs_input <= 0 else n_envs_input

    seed = IntPrompt.ask("Random seed", default=42)

    eval_episodes = IntPrompt.ask(
        "Evaluation episodes per opponent after training",
        default=100,
    )
    console.print(
        "[dim]The 'Quick evaluation' run during training is a fast deterministic "
        "sanity check (small samples can be very noisy). The official numbers "
        "come from the full evaluation above.[/]"
    )
    quick_eval_episodes = IntPrompt.ask(
        "Quick-evaluation episodes per opponent (during training)",
        default=20,
    )

    # Sample game reports
    console.print("\n[bold]Step 6: Sample Game Reports[/bold]")
    console.print(
        "[dim]After training, sample games are recorded as interactive HTML replays "
        "so you can inspect how the agent actually plays.[/dim]\n"
    )

    sample_games = IntPrompt.ask(
        "Number of sample games to record per opponent",
        default=3,
    )

    # Choose opponents for sample reports
    sample_opponents_str = ""
    if len(opponent_decks) > 1:
        console.print(
            "\n[dim]You trained against multiple opponents. "
            "Which should the sample games be played against?[/dim]"
        )
        console.print("  [cyan]1[/] All training opponents")
        console.print("  [cyan]2[/] Select specific opponents")
        sample_opp_choice = IntPrompt.ask("Select (1-2)", default=1)

        if sample_opp_choice == 2:
            console.print(f"\nAvailable opponents: {', '.join(opponent_decks)}")
            selected = Prompt.ask(
                "Enter opponent(s) for sample games [dim](comma-separated)[/dim]",
                default=opponent_decks[0],
            )
            sample_opponents_str = selected
    else:
        console.print(f"[dim]Sample games will be against: {opponent_decks[0]}[/dim]")

    return TrainingConfig(
        agent_type=agent_type,
        player_deck=player_deck,
        opponent_deck=opponent_deck,
        timesteps=timesteps,
        reward_type=reward_type,
        seed=seed,
        max_turns=max_turns,
        n_envs=n_envs,
        training_mode=training_mode,
        agency_mode=agency_mode,
        eval_episodes=eval_episodes,
        quick_eval_episodes=quick_eval_episodes,
        sample_games=sample_games,
        sample_opponents=sample_opponents_str,
    )


def prompt_evaluation_config() -> EvaluationConfig:
    """Interactively prompt for evaluation configuration.

    Returns:
        Complete EvaluationConfig.

    """
    print_section_header("Evaluation Configuration")

    # Discover trained models
    models = discover_trained_models()

    # Agent selection
    console.print("[bold]Step 1: Select Agent to Evaluate[/bold]")
    agent_type, model_path = prompt_agent_selection(
        prompt_text="Select agent",
        include_trained=True,
        trained_models=models,
    )

    # Player deck selection is locked to the selected trained model deck when available.
    selected_model = None
    if model_path:
        selected_model = next((m for m in models if m.get("path") == model_path), None)

    if selected_model and selected_model.get("player_deck"):
        player_deck = str(selected_model["player_deck"])
        console.print("\n[bold]Step 2: Player Deck (from selected model)[/bold]")
        console.print(f"[dim]Using trained model deck:[/dim] [cyan]{player_deck}[/cyan]")
    else:
        console.print("\n[bold]Step 2: Select Player Deck[/bold]")
        console.print("[dim]Note: Baseline agents can be evaluated on any deck matchup.[/dim]\n")
        player_deck = prompt_deck_selection(
            prompt_text="Select player deck",
            default="mono_red_aggro",
        )

    # Opponent selection: single, all, or a selected subset
    console.print("\n[bold]Step 3: Select Opponent Benchmark Scope[/bold]")
    archetypes = get_available_archetypes()
    console.print("  [cyan]1[/] Single opponent")
    console.print("  [cyan]2[/] All opponents (comprehensive benchmark)")
    console.print("  [cyan]3[/] Select specific opponents")
    opp_choice = IntPrompt.ask("Select scope (1-3)", default=1)

    if opp_choice == 2:
        opponent_deck = ",".join(archetypes)
        console.print(f"[dim]Evaluating against all {len(archetypes)} archetypes[/dim]")
    elif opp_choice == 3:
        console.print(f"\nAvailable opponents: {', '.join(archetypes)}")
        selected = Prompt.ask(
            "Enter opponent(s) [dim](comma-separated)[/dim]",
            default=archetypes[0],
        )
        selected_list = [d.strip() for d in selected.split(",") if d.strip()]
        valid = [d for d in selected_list if d in archetypes]
        if not valid:
            valid = [archetypes[0]]
        opponent_deck = ",".join(valid)
        console.print(f"[dim]Evaluating against: {', '.join(valid)}[/dim]")
    else:
        opponent_deck = prompt_deck_selection(
            prompt_text="Select opponent deck",
            default="azorius_control",
        )

    # Evaluation parameters
    console.print("\n[bold]Step 4: Evaluation Parameters[/bold]")

    episodes = IntPrompt.ask(
        "Total evaluation episodes per opponent",
        default=500,
    )
    console.print(
        "[dim]Episode budget is per-opponent and distributed across seeds "
        "(not split across opponents).[/dim]"
    )

    console.print("\n[bold]Max turns per game:[/bold]")
    console.print("  [dim]Should match the turn limit used during training for consistency.[/dim]")
    max_turns = IntPrompt.ask("Max turns per game", default=10)

    # Report generation (no detailed CLI game visualization in interactive mode)
    console.print("\n[bold]Step 5: Optional Sample Reports[/bold]")
    console.print(
        "[dim]Generate HTML reports after evaluation (without detailed in-terminal "
        "game-state playback).[/dim]"
    )
    show_games = IntPrompt.ask(
        "Number of sample games to record per opponent (0 for none)",
        default=1,
    )

    show_games_opponents_str = ""
    selected_eval_opponents = [d.strip() for d in opponent_deck.split(",") if d.strip()]
    if show_games > 0 and len(selected_eval_opponents) > 1:
        console.print("Sample report opponents:")
        console.print("  [cyan]1[/] All evaluation opponents")
        console.print("  [cyan]2[/] Select specific opponents")
        reports_choice = IntPrompt.ask("Select option (1-2)", default=1)
        if reports_choice == 2:
            console.print(f"\nAvailable opponents: {', '.join(selected_eval_opponents)}")
            selected_reports = Prompt.ask(
                "Enter opponent(s) for sample reports [dim](comma-separated)[/dim]",
                default=selected_eval_opponents[0],
            )
            selected_reports_list = [d.strip() for d in selected_reports.split(",") if d.strip()]
            valid_reports = [d for d in selected_reports_list if d in selected_eval_opponents]
            if valid_reports:
                show_games_opponents_str = ",".join(valid_reports)
            else:
                show_games_opponents_str = selected_eval_opponents[0]
            console.print(
                f"[dim]Sample reports will be generated for: "
                f"{', '.join(show_games_opponents_str.split(','))}[/dim]"
            )

    save_reports = show_games > 0

    return EvaluationConfig(
        agent_type=agent_type,
        model_path=model_path,
        player_deck=player_deck,
        opponent_deck=opponent_deck,
        episodes=episodes,
        max_turns=max_turns,
        show_games=show_games,
        show_games_opponents=show_games_opponents_str,
        save_reports=save_reports,
        verbose=False,
    )


def prompt_gameplay_config() -> GameplayConfig:
    """Interactively prompt for gameplay configuration.

    Returns:
        Complete GameplayConfig.

    """
    print_section_header("Gameplay Configuration")

    # Discover trained models for potential selection
    models = discover_trained_models()

    # Player agent selection
    console.print("[bold]Step 1: Select Player Agent[/bold]")
    console.print("[dim]Note: Opponent uses built-in greedy_aggro agent[/dim]\n")
    player_agent, player_model = prompt_agent_selection(
        prompt_text="Select player agent",
        include_trained=True,
        trained_models=models,
    )

    # Deck selection
    console.print("\n[bold]Step 2: Select Decks[/bold]")

    player_deck = prompt_deck_selection(
        prompt_text="Select player deck",
        default="mono_red_aggro",
    )

    opponent_deck = prompt_deck_selection(
        prompt_text="Select opponent deck",
        default="azorius_control",
    )

    # Game parameters
    console.print("\n[bold]Step 3: Game Parameters[/bold]")

    num_turns = IntPrompt.ask(
        "Maximum turns to play",
        default=5,
    )

    speed_options = ["slow", "medium", "fast"]
    console.print("Speed: [cyan]1[/] slow (5s)  [cyan]2[/] medium (3s)  [cyan]3[/] fast (1s)")
    speed_choice = IntPrompt.ask("Select speed (1-3)", default=2)
    speed = speed_options[min(speed_choice - 1, 2)]

    save_report = Confirm.ask("Save HTML replay report?", default=True)

    return GameplayConfig(
        player_agent=player_agent,
        player_model_path=player_model,
        player_deck=player_deck,
        opponent_deck=opponent_deck,
        num_turns=num_turns,
        speed=speed,
        save_report=save_report,
    )


def confirm_config(config: tp.Any, config_type: str) -> bool:
    """Display configuration summary and confirm.

    Args:
        config: Configuration object.
        config_type: Type of configuration (Training/Evaluation/Gameplay).

    Returns:
        True if user confirms, False otherwise.

    """
    print_section_header(f"{config_type} Configuration Summary")

    table = Table(box=box.ROUNDED, show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    for field_name, value in config.__dict__.items():
        display_name = field_name.replace("_", " ").title()
        table.add_row(display_name, str(value))

    console.print(table)
    console.print()

    return Confirm.ask("Proceed with this configuration?", default=True)


# =============================================================================
# Utility Functions
# =============================================================================


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string (e.g., "1h 23m 45s").

    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")

    return " ".join(parts)


def create_output_directory(
    base_dir: str,
    run_name: str,
) -> Path:
    """Create output directory for a run.

    Args:
        base_dir: Base output directory.
        run_name: Unique run name.

    Returns:
        Path to created directory.

    """
    output_path = Path(base_dir) / run_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (output_path / "plots").mkdir(exist_ok=True)
    (output_path / "reports").mkdir(exist_ok=True)

    return output_path


__all__ = [
    "TrainingConfig",
    "EvaluationConfig",
    "GameplayConfig",
    "discover_trained_models",
    "find_model_for_deck",
    "get_available_agents",
    "get_available_archetypes",
    "get_archetype_info",
    "prompt_agent_selection",
    "prompt_trainable_agent_selection",
    "prompt_opponent_strategy",
    "prompt_deck_selection",
    "prompt_training_config",
    "prompt_evaluation_config",
    "prompt_gameplay_config",
    "confirm_config",
    "create_output_directory",
    "format_duration",
    "print_section_header",
]

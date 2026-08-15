"""Rich CLI display for training and evaluation.

This module provides beautiful terminal visualizations using the rich library,
including live progress displays, game state rendering, and result tables.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

# Console instance
console = Console()

# MTG-themed color scheme
THEME = {
    "primary": "bold cyan",
    "secondary": "blue",
    "success": "bold green",
    "warning": "bold yellow",
    "danger": "bold red",
    "mana_white": "bold white",
    "mana_blue": "bold blue",
    "mana_black": "bold magenta",
    "mana_red": "bold red",
    "mana_green": "bold green",
    "agent_random": "dim white",
    "agent_heuristic": "blue",
    "agent_ppo": "red",
    "agent_causal": "green",
}

# Mana color symbols
MANA_SYMBOLS = {
    "W": "⚪",  # White
    "U": "🔵",  # Blue
    "B": "⚫",  # Black
    "R": "🔴",  # Red
    "G": "🟢",  # Green
    "C": "◇",  # Colorless
}

# Card type legend for display
CARD_TYPE_LEGEND = "⚔️=Creature ✨=Instant 🌟=Sorcery 🔮=Enchantment ⚙️=Artifact 🌍=Land 🎴=Token"


def format_mana_cost(mana_cost: str) -> str:
    """Convert text mana cost to emoji symbols.

    Examples:
        "R" -> "🔴"
        "1R" -> "1🔴"
        "1UU" -> "1🔵🔵"
        "3WW" -> "3⚪⚪"

    Args:
        mana_cost: Text mana cost like "1R", "2UU", etc.

    Returns:
        Formatted mana cost with emoji symbols.

    """
    if not mana_cost:
        return ""

    result = []
    i = 0
    while i < len(mana_cost):
        char = mana_cost[i]
        if char.isdigit():
            # Collect all consecutive digits
            num = ""
            while i < len(mana_cost) and mana_cost[i].isdigit():
                num += mana_cost[i]
                i += 1
            result.append(num)
        elif char.upper() in MANA_SYMBOLS:
            result.append(MANA_SYMBOLS[char.upper()])
            i += 1
        else:
            result.append(char)
            i += 1

    return "".join(result)


def _format_card_with_type(card_name: str) -> str:
    """Format a card name with standard display: Name - P/T (mana, type).

    Looks up card data from CardRegistry automatically.

    Args:
        card_name: Name of the card.

    Returns:
        Formatted string like:
        - Creature: "Goblin Guide - 2/2 (🔴, ⚔️)"
        - Instant: "Lightning Bolt (🔴, ✨)"
        - Land: "Mountain (🌍)"

    """
    from mtg.env.card_definitions import CardRegistry, CardType

    # Type symbols
    type_symbols = {
        CardType.CREATURE: "⚔️",
        CardType.INSTANT: "✨",
        CardType.SORCERY: "🌟",
        CardType.ENCHANTMENT: "🔮",
        CardType.ARTIFACT: "⚙️",
        CardType.LAND: "🌍",
        CardType.PLANESWALKER: "👤",
    }

    registry = CardRegistry.get_instance()
    card = registry.get(card_name)

    if not card:
        return card_name

    type_symbol = type_symbols.get(card.card_type, "❓")

    # Get mana cost display
    mana_display = ""
    if card.mana_cost:
        mana_display = format_mana_cost(card.mana_cost.to_text())

    # Get power/toughness for creatures
    stats_str = ""
    if card.card_type == CardType.CREATURE and card.power is not None:
        stats_str = f" - {card.power}/{card.toughness}"

    # Build the display string
    if mana_display:
        return f"{card_name}{stats_str} ({mana_display}, {type_symbol})"
    else:
        return f"{card_name}{stats_str} ({type_symbol})"


# =============================================================================
# ASCII Art
# =============================================================================

LOGO = """
[bold cyan]╔═══════════════════════════════════════════════════════════════════╗
║  [bold white]███╗   ███╗████████╗ ██████╗        ██████╗██████╗ ██╗[/bold white]           ║
║  [bold white]████╗ ████║╚══██╔══╝██╔════╝       ██╔════╝██╔══██╗██║[/bold white]           ║
║  [bold white]██╔████╔██║   ██║   ██║  ███╗█████╗██║     ██████╔╝██║[/bold white]           ║
║  [bold white]██║╚██╔╝██║   ██║   ██║   ██║╚════╝██║     ██╔══██╗██║[/bold white]           ║
║  [bold white]██║ ╚═╝ ██║   ██║   ╚██████╔╝      ╚██████╗██║  ██║███████╗[/bold white]      ║
║  [bold white]╚═╝     ╚═╝   ╚═╝    ╚═════╝        ╚═════╝╚═╝  ╚═╝╚══════╝[/bold white]      ║
║                                                                   ║
║            [bold green]Causal Reinforcement Learning Benchmark[/bold green]                ║
║                    [dim]for Magic: The Gathering[/dim]                       ║
╚═══════════════════════════════════════════════════════════════════╝[/bold cyan]
"""

CARD_TEMPLATE = """
┌─────────────────┐
│ {name:<15} │
│ {cost:>15} │
├─────────────────┤
│                 │
│   {type:<11}   │
│                 │
├─────────────────┤
│ {stats:<15} │
└─────────────────┘
"""


def print_logo() -> None:
    """Print the MTG-Causal-RL logo."""
    console.print(LOGO)


def print_divider(title: str = "", style: str = "cyan") -> None:
    """Print a styled divider line.

    Args:
        title: Optional title in the divider.
        style: Rich style for the divider.

    """
    if title:
        console.rule(f"[bold {style}]{title}[/]", style=style)
    else:
        console.rule(style=style)


# =============================================================================
# Training Display
# =============================================================================


@dataclass
class TrainingMetrics:
    """Container for training metrics.

    Attributes:
        timesteps: Current timestep count.
        episodes: Episodes completed.
        win_rate: Current win rate.
        avg_reward: Average episode reward.
        episode_length: Average episode length.
        loss: Current loss value.
        entropy: Policy entropy.
        learning_rate: Current learning rate.
        fps: Frames per second.

    """

    timesteps: int = 0
    episodes: int = 0
    win_rate: float = 0.0
    avg_reward: float = 0.0
    episode_length: float = 0.0
    loss: float = 0.0
    entropy: float = 0.0
    learning_rate: float = 3e-4
    fps: float = 0.0
    history: list[dict[str, float]] = field(default_factory=list)


class TrainingDisplay:
    """Live training display with progress and metrics.

    Attributes:
        agent_name: Name of the agent being trained.
        total_timesteps: Total training timesteps.
        metrics: Current training metrics.

    """

    def __init__(
        self,
        agent_name: str,
        total_timesteps: int,
        deck: str = "mono_red_aggro",
        opponent: str = "azorius_control",
    ) -> None:
        """Initialize the training display.

        Args:
            agent_name: Agent being trained.
            total_timesteps: Total training steps.
            deck: Player deck archetype.
            opponent: Opponent deck archetype.

        """
        self.agent_name = agent_name
        self.total_timesteps = total_timesteps
        self.deck = deck
        self.opponent = opponent
        self.metrics = TrainingMetrics()
        self.start_time = time.time()
        self._live: Live | None = None

    def _create_header(self) -> Panel:
        """Create the header panel."""
        elapsed = time.time() - self.start_time
        elapsed_str = f"{int(elapsed // 60)}:{int(elapsed % 60):02d}"

        header_text = Text()
        header_text.append("🎮 ", style="bold")
        header_text.append("MTG-Causal-RL Training", style="bold cyan")
        header_text.append(f"\n\n⏱️  Elapsed: {elapsed_str}")
        header_text.append(f"   📊 {self.metrics.timesteps:,} / {self.total_timesteps:,} steps")

        return Panel(
            header_text,
            title=f"[bold]{self.agent_name}[/bold]",
            subtitle=f"[dim]{self.deck} vs {self.opponent}[/dim]",
            border_style="cyan",
            padding=(0, 2),
        )

    def _create_progress_bar(self) -> Progress:
        """Create the progress bar."""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=50, complete_style="green", finished_style="green"),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        )
        return progress

    def _create_metrics_table(self) -> Table:
        """Create the metrics table."""
        table = Table(
            box=box.ROUNDED,
            border_style="cyan",
            header_style="bold cyan",
            show_header=True,
            padding=(0, 1),
        )

        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_column("Trend", justify="center")

        # Calculate trends
        def get_trend(metric: str, current: float) -> str:
            if len(self.metrics.history) < 2:
                return "[dim]—[/dim]"
            prev = self.metrics.history[-2].get(metric, current)
            if current > prev:
                return "[green]▲[/green]"
            elif current < prev:
                return "[red]▼[/red]"
            return "[dim]—[/dim]"

        # Win rate with color coding
        wr = self.metrics.win_rate
        wr_style = "green" if wr > 0.5 else "yellow" if wr > 0.3 else "red"
        table.add_row(
            "Win Rate",
            f"[{wr_style}]{wr:.1%}[/{wr_style}]",
            get_trend("win_rate", wr),
        )

        table.add_row(
            "Avg Reward",
            f"{self.metrics.avg_reward:.3f}",
            get_trend("avg_reward", self.metrics.avg_reward),
        )

        table.add_row(
            "Episodes",
            f"{self.metrics.episodes:,}",
            "",
        )

        table.add_row(
            "Ep. Length",
            f"{self.metrics.episode_length:.1f}",
            get_trend("episode_length", self.metrics.episode_length),
        )

        if self.metrics.loss > 0:
            table.add_row(
                "Loss",
                f"{self.metrics.loss:.4f}",
                get_trend("loss", self.metrics.loss),
            )

        table.add_row(
            "FPS",
            f"{self.metrics.fps:.0f}",
            "",
        )

        return table

    def _create_sparkline(self, data: list[float], width: int = 30) -> str:
        """Create a sparkline visualization.

        Args:
            data: Data points to visualize.
            width: Character width of sparkline.

        Returns:
            Sparkline string.

        """
        if not data or len(data) < 2:
            return "[dim]" + "─" * width + "[/dim]"

        chars = "▁▂▃▄▅▆▇█"
        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val if max_val != min_val else 1

        # Resample to width
        step = max(1, len(data) // width)
        sampled = data[::step][:width]

        result = ""
        for val in sampled:
            idx = int((val - min_val) / range_val * (len(chars) - 1))
            result += chars[idx]

        return f"[green]{result}[/green]"

    def _create_display(self) -> Panel:
        """Create the full display."""
        # Progress percentage (capped at 100%)
        pct = min(1.0, self.metrics.timesteps / max(1, self.total_timesteps))
        bar_width = 50
        filled = min(bar_width, int(pct * bar_width))

        # Build progress bar using Text with proper styles (not markup strings)
        progress_text = Text()
        progress_text.append("\n  ")
        progress_text.append("█" * filled, style="green")
        progress_text.append("░" * (bar_width - filled), style="dim")
        progress_text.append(f" {pct:.1%}\n")

        # Metrics table
        metrics_table = self._create_metrics_table()

        # Sparkline for win rate history
        wr_history = [h.get("win_rate", 0.5) for h in self.metrics.history[-50:]]
        sparkline = self._create_sparkline(wr_history)

        content = Group(
            self._create_header(),
            Panel(progress_text, title="Progress", border_style="blue"),
            Columns(
                [
                    metrics_table,
                    Panel(
                        f"Win Rate History:\n{sparkline}",
                        title="Trend",
                        border_style="green",
                    ),
                ]
            ),
        )

        return Panel(
            content,
            title="[bold cyan]🧠 Training Progress[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )

    def start(self) -> None:
        """Start the live display."""
        self._live = Live(
            self._create_display(),
            console=console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.start()

    def update(
        self,
        timesteps: int | None = None,
        episodes: int | None = None,
        win_rate: float | None = None,
        avg_reward: float | None = None,
        episode_length: float | None = None,
        loss: float | None = None,
        fps: float | None = None,
    ) -> None:
        """Update metrics and refresh display.

        Args:
            timesteps: Current timestep count.
            episodes: Episodes completed.
            win_rate: Current win rate.
            avg_reward: Average reward.
            episode_length: Average episode length.
            loss: Current loss.
            fps: Frames per second.

        """
        if timesteps is not None:
            self.metrics.timesteps = timesteps
        if episodes is not None:
            self.metrics.episodes = episodes
        if win_rate is not None:
            self.metrics.win_rate = win_rate
        if avg_reward is not None:
            self.metrics.avg_reward = avg_reward
        if episode_length is not None:
            self.metrics.episode_length = episode_length
        if loss is not None:
            self.metrics.loss = loss
        if fps is not None:
            self.metrics.fps = fps

        # Record history
        self.metrics.history.append(
            {
                "win_rate": self.metrics.win_rate,
                "avg_reward": self.metrics.avg_reward,
                "episode_length": self.metrics.episode_length,
                "loss": self.metrics.loss,
            }
        )

        if self._live:
            self._live.update(self._create_display())

    def stop(self) -> None:
        """Stop the live display."""
        if self._live:
            self._live.stop()


# =============================================================================
# Evaluation Display
# =============================================================================


def print_evaluation_results(
    results: dict[str, dict[str, float]],
    title: str = "Evaluation Results",
) -> None:
    """Print a beautiful evaluation results table.

    Args:
        results: Dict mapping agent names to metrics.
        title: Table title.

    """
    table = Table(
        title=f"[bold cyan]{title}[/bold cyan]",
        box=box.DOUBLE_EDGE,
        border_style="cyan",
        header_style="bold white on blue",
        show_header=True,
        padding=(0, 1),
    )

    table.add_column("🤖 Agent", style="bold", min_width=15)
    table.add_column("🏆 Win Rate", justify="right", min_width=12)
    table.add_column("📊 WR 95% CI", justify="right", min_width=10)
    table.add_column("💰 Avg Reward", justify="right", min_width=12)
    table.add_column("📈 Reward Std", justify="right", min_width=10)

    # Sort by win rate
    sorted_agents = sorted(
        results.items(),
        key=lambda x: x[1].get("win_rate", 0),
        reverse=True,
    )

    for rank, (agent, metrics) in enumerate(sorted_agents, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "  ")

        wr = metrics.get("win_rate", 0)
        wr_style = "green" if wr > 0.5 else "yellow" if wr > 0.3 else "red"

        agent_style = THEME.get(f"agent_{agent.lower()}", "white")

        ci = metrics.get("win_rate_ci95", 0)
        reward_std = metrics.get("reward_std", 0)

        table.add_row(
            f"{medal} [{agent_style}]{agent}[/{agent_style}]",
            f"[{wr_style}]{wr:.1%}[/{wr_style}]",
            f"±{ci:.1%}",
            f"{metrics.get('avg_reward', 0):.3f}",
            f"{reward_std:.3f}" if reward_std else "—",
        )

    console.print()
    console.print(table)
    console.print()


@dataclass
class PlayerState:
    """State of a single player.

    Attributes:
        name: Player identifier.
        life: Life total.
        hand_size: Cards in hand.
        lands: Lands in play.
        lands_tapped: Tapped lands.
        creatures: List of (name, power, toughness, tapped).
        mana_available: Available mana.

    """

    name: str
    life: int = 20
    hand_size: int = 7
    lands: int = 0
    lands_tapped: int = 0
    creatures: list = field(default_factory=list)
    mana_available: int = 0


@dataclass
class GameStateSnapshot:
    """Complete game state for visualization.

    Attributes:
        turn: Current turn number.
        phase: Current phase name.
        active_player: Who is the active player.
        player: Player state.
        opponent: Opponent state.
        last_action: Description of last action taken.
        action_history: List of recent actions.
        is_mulligan: Whether in mulligan phase.
        mulligan_count: Number of mulligans taken.

    """

    turn: int = 0
    phase: str = "Mulligan"
    active_player: str = "Player"
    player: PlayerState = field(default_factory=lambda: PlayerState("Player"))
    opponent: PlayerState = field(default_factory=lambda: PlayerState("Opponent"))
    last_action: str = ""
    action_history: list = field(default_factory=list)
    is_mulligan: bool = True
    mulligan_count: int = 0


# Phase display order (matches GamePhase enum)
PHASES = ["Untap", "Upkeep", "Draw", "Main 1", "Combat", "Blockers", "Damage", "Main 2", "End"]

# Priority-related phases where responses are possible
PRIORITY_PHASE_NAMES = {"Upkeep", "Main 1", "Combat", "Attackers", "Blockers", "Main 2", "End"}


def _life_bar(life: int, max_life: int = 20, width: int = 10) -> str:
    """Create a life bar visualization.

    Args:
        life: Current life.
        max_life: Maximum life for scaling.
        width: Character width of bar.

    Returns:
        Colored life bar string.

    """
    ratio = min(1.0, max(0.0, life / max_life))
    filled = int(ratio * width)
    empty = width - filled

    if ratio > 0.5:
        color = "green"
    elif ratio > 0.25:
        color = "yellow"
    else:
        color = "red"

    return f"[{color}]{'█' * filled}[/{color}][dim]{'░' * empty}[/dim]"


def _phase_indicator(current_phase: str) -> Text:
    """Create a phase progression indicator aligned with labels.

    Args:
        current_phase: Current phase name.

    Returns:
        Phase indicator as Rich Text.

    """
    indicator = Text()
    current_idx = PHASES.index(current_phase) if current_phase in PHASES else -1

    for i, phase in enumerate(PHASES):
        if i > 0:
            indicator.append("   ")  # 3 spaces to align with 3-char labels + spacing

        # Center the dot in a 3-character space
        indicator.append(" ")
        if phase == current_phase:
            indicator.append("●", style="bold cyan")
        elif i < current_idx:
            indicator.append("●", style="green")
        else:
            indicator.append("○", style="dim")
        indicator.append(" ")

    return indicator


def _phase_labels() -> Text:
    """Create phase label line aligned with indicators.

    Returns:
        Phase labels as Rich Text.

    """
    labels = Text()
    for i, phase in enumerate(PHASES):
        if i > 0:
            labels.append("   ")  # 3 spaces between labels
        labels.append(phase[:3], style="dim")
    return labels


def _format_token(token_data: tuple) -> str:
    """Format a token for display without P/T for artifact tokens.

    Args:
        token_data: Tuple of (name, power, toughness) or just (name,).

    Returns:
        Formatted token string. Artifact tokens show "Name (🎴)",
        creature tokens show "Name - P/T (🎴)".

    """
    if not token_data:
        return ""

    name = token_data[0]
    power = token_data[1] if len(token_data) > 1 else None
    toughness = token_data[2] if len(token_data) > 2 else None

    # Check if this is an artifact token (non-creature)
    from mtg.env.card_definitions import CardRegistry

    registry = CardRegistry.get_instance()
    card = registry.get(f"{name} Token") or registry.get(name)

    is_creature_token = False
    if card:
        from mtg.env.card_definitions import CardType

        is_creature_token = card.card_type == CardType.CREATURE
    elif power is not None and power > 0:
        # If we have P/T > 0, assume creature
        is_creature_token = True

    if is_creature_token and power is not None and toughness is not None:
        return f"{name} - {power}/{toughness} (🎴)"
    return f"{name} (🎴)"


def _format_creatures(creatures: list) -> str:
    """Format creature list for display with standard formatting.

    Args:
        creatures: List of (name, power, toughness, tapped) or
                   (name, power, toughness, tapped, mana_cost) tuples.

    Returns:
        Formatted creature string using standard Name - P/T (mana, ⚔️) format.

    """
    if not creatures:
        return "[dim]No creatures[/dim]"

    from mtg.env.card_definitions import CardRegistry

    registry = CardRegistry.get_instance()

    parts = []
    for creature in creatures[:4]:  # Show max 4
        name = creature[0]
        power = creature[1]
        toughness = creature[2]
        tapped = creature[3] if len(creature) > 3 else False

        # Get mana cost from registry
        card = registry.get(name)
        mana_display = ""
        if card and card.mana_cost:
            mana_display = format_mana_cost(card.mana_cost.to_text())

        tap_indicator = " [dim](T)[/dim]" if tapped else ""
        if mana_display:
            parts.append(f"{name} - {power}/{toughness} ({mana_display}, ⚔️){tap_indicator}")
        else:
            parts.append(f"{name} - {power}/{toughness} (⚔️){tap_indicator}")

    if len(creatures) > 4:
        parts.append(f"[dim]+{len(creatures) - 4} more[/dim]")

    return ", ".join(parts)


def _format_graveyard(graveyard: list, max_show: int = 2) -> str:
    """Format graveyard list for display.

    Args:
        graveyard: List of (name, card_type) tuples.
        max_show: Maximum cards to show individually.

    Returns:
        Formatted graveyard string.

    """
    if not graveyard:
        return "[dim]Empty[/dim]"

    # Count cards by type for summary
    creatures = sum(1 for _, ct in graveyard if ct == "creature")
    instants = sum(1 for _, ct in graveyard if ct in ("instant", "sorcery"))
    other = len(graveyard) - creatures - instants

    # Build summary first
    summary = []
    if creatures > 0:
        summary.append(f"💀×{creatures}")
    if instants > 0:
        summary.append(f"✨×{instants}")
    if other > 0:
        summary.append(f"📜×{other}")

    # Show last few cards (abbreviated names)
    recent = graveyard[-max_show:]
    parts = []
    for name, _card_type in reversed(recent):
        # Abbreviate long card names
        short_name = name[:12] + "..." if len(name) > 15 else name
        parts.append(short_name)

    card_str = ", ".join(parts)
    if len(graveyard) > max_show:
        card_str += f" +{len(graveyard) - max_show}"

    summary_str = " ".join(summary) if summary else ""
    return f"[dim]{card_str}[/dim] ({summary_str})" if summary_str else f"[dim]{card_str}[/dim]"


def print_play_draw_selection(
    player_on_play: bool,
) -> None:
    """Print play/draw selection result.

    Args:
        player_on_play: True if player is on the play, False if on draw.

    """
    if player_on_play:
        result = "[bold green]You are ON THE PLAY[/bold green] (go first, no draw T1)"
        coin = "🪙 Coin flip: Heads"
    else:
        result = "[bold blue]You are ON THE DRAW[/bold blue] (go second, draw T1)"
        coin = "🪙 Coin flip: Tails"

    panel = Panel(
        f"""
[bold]GAME START[/bold]

{coin}

{result}
        """.strip(),
        title="[bold]Play/Draw Selection[/bold]",
        border_style="yellow",
        width=60,
    )
    console.print(panel)


def print_mulligan_state(
    hand_size: int,
    mulligan_count: int,
    action: str = "",
    kept: bool = False,
    player_name: str = "Player",
) -> None:
    """Print mulligan phase visualization.

    Args:
        hand_size: Current hand size.
        mulligan_count: Number of mulligans taken.
        action: Action description.
        kept: Whether hand was kept.
        player_name: Name of the player mulliganing.

    """
    # Show actual hand size with dimmed placeholders for missing cards
    cards_in_hand = "🃏 " * hand_size
    cards_lost = "[dim]✕[/dim] " * (7 - hand_size) if hand_size < 7 else ""
    cards_display = cards_in_hand + cards_lost

    player_style = "green" if player_name == "Player" else "red"

    if kept:
        status = f"[bold green]✓ Kept {hand_size} cards[/bold green]"
    elif mulligan_count == 0:
        status = f"[cyan]Opening hand: {hand_size} cards[/cyan]"
    else:
        status = f"[yellow]Mulligan #{mulligan_count}: {hand_size} cards[/yellow]"

    action_line = f"\n[dim]Action:[/dim] {action}" if action else ""

    panel = Panel(
        f"""
[bold]MULLIGAN PHASE[/bold] - [{player_style}]{player_name}[/{player_style}]

{status}

{cards_display}
{action_line}
        """.strip(),
        title="[bold]Pre-Game[/bold]",
        border_style="magenta",
        width=60,
    )
    console.print(panel)


def print_game_state(
    turn: int,
    phase: str,
    player_life: int,
    opponent_life: int,
    hand_size: int,
    lands: int,
    board_power: int,
    *,
    opponent_lands: int = 0,
    opponent_power: int = 0,
    player_creatures: list | None = None,
    opponent_creatures: list | None = None,
    player_graveyard: list | None = None,
    opponent_graveyard: list | None = None,
    player_tokens: list | None = None,
    opponent_tokens: list | None = None,
    last_action: str = "",
    action_history: list | None = None,
    active_player: str = "Player",
    mana_available: int = 0,
    opponent_hand_size: int = 0,
    player_lands_detail: dict | None = None,
    opponent_lands_detail: dict | None = None,
    player_mana_detail: dict | None = None,
) -> None:
    """Print a comprehensive game state visualization.

    Args:
        turn: Current turn number.
        phase: Current game phase.
        player_life: Player's life total.
        opponent_life: Opponent's life total.
        hand_size: Cards in player's hand.
        lands: Player's lands in play.
        board_power: Player's total creature power.
        opponent_lands: Opponent's lands in play.
        opponent_power: Opponent's total creature power.
        player_creatures: List of player's creatures.
        opponent_creatures: List of opponent's creatures.
        player_graveyard: List of player's graveyard cards.
        opponent_graveyard: List of opponent's graveyard cards.
        player_tokens: List of player's tokens.
        opponent_tokens: List of opponent's tokens.
        last_action: Most recent action taken.
        action_history: List of recent actions.
        active_player: Who is the active player.
        mana_available: Available mana (total).
        opponent_hand_size: Cards in opponent's hand.
        player_lands_detail: Player lands by type (e.g., {"Mountain": 2}).
        opponent_lands_detail: Opponent lands by type.
        player_mana_detail: Player mana by color (e.g., {"R": 2}).

    """
    player_creatures = player_creatures or []
    opponent_creatures = opponent_creatures or []
    player_graveyard = player_graveyard or []
    opponent_graveyard = opponent_graveyard or []
    player_tokens = player_tokens or []
    opponent_tokens = opponent_tokens or []
    action_history = action_history or []

    # Phase indicator (as Rich Text objects)
    phase_line = _phase_indicator(phase)
    phase_labels_text = _phase_labels()

    # Create battlefield table with headers
    battlefield = Table(
        box=box.DOUBLE_EDGE,
        show_header=True,
        header_style="bold dim",
        padding=(0, 1),
        border_style="blue",
    )
    battlefield.add_column("Player", style="bold", width=10)
    battlefield.add_column("Life", width=20)
    battlefield.add_column("🃏", justify="center", width=5)  # Hand
    battlefield.add_column("🌍", justify="center", width=5)  # Lands
    battlefield.add_column("💎", justify="center", width=5)  # Mana
    battlefield.add_column("⚔️", justify="center", width=5)  # Power

    # Opponent row
    opp_active = " ◀" if active_player == "Opponent" else ""
    opp_life_bar = _life_bar(opponent_life)
    battlefield.add_row(
        f"[red]OPPONENT[/red]{opp_active}",
        f"{opp_life_bar} [bold]{opponent_life}[/bold]",
        f"[bold]{opponent_hand_size}[/bold]",
        f"[bold]{opponent_lands}[/bold]",
        "[dim]—[/dim]",
        f"[bold]{opponent_power}[/bold]",
    )

    # Opponent hand (hidden)
    battlefield.add_row(
        "",
        f"[dim]Hand:[/dim] [italic dim]{opponent_hand_size} cards (hidden)[/italic dim]",
        "",
        "",
        "",
        "",
    )

    # Opponent creatures
    opp_creatures = _format_creatures(opponent_creatures)
    battlefield.add_row("", f"[dim]Creatures:[/dim] {opp_creatures}", "", "", "", "")

    # Opponent tokens
    if opponent_tokens:
        tokens_str = ", ".join([f"{t[0]} {t[1]}/{t[2]}" for t in opponent_tokens[:3]])
        battlefield.add_row("", f"[dim]Tokens:[/dim] {tokens_str}", "", "", "", "")

    # Opponent graveyard
    opp_gy = _format_graveyard(opponent_graveyard)
    battlefield.add_row("", f"[dim]Graveyard:[/dim] {opp_gy}", "", "", "", "")

    # Opponent lands detail
    if opponent_lands_detail:
        lands_str = ", ".join([f"{k} x{v}" for k, v in opponent_lands_detail.items()])
        battlefield.add_row("", f"[dim]Lands:[/dim] {lands_str}", "", "", "", "")

    # Separator
    battlefield.add_row(
        "[dim]─" * 8 + "[/dim]",
        "[dim]─" * 16 + "[/dim]",
        "[dim]──[/dim]",
        "[dim]──[/dim]",
        "[dim]──[/dim]",
        "[dim]──[/dim]",
    )

    # Player row
    player_active = " ◀" if active_player == "Player" else ""
    player_life_bar = _life_bar(player_life)

    # Format mana display
    mana_display = str(mana_available)
    if player_mana_detail:
        mana_parts = []
        for color, amt in player_mana_detail.items():
            if amt > 0:
                icon = {"R": "🔴", "U": "🔵", "W": "⚪", "B": "⚫", "G": "🟢"}.get(color, color)
                mana_parts.append(f"{icon}{amt}")
        if mana_parts:
            mana_display = " ".join(mana_parts)

    battlefield.add_row(
        f"[green]PLAYER[/green]{player_active}",
        f"{player_life_bar} [bold]{player_life}[/bold]",
        f"[bold]{hand_size}[/bold]",
        f"[bold]{lands}[/bold]",
        f"[bold]{mana_display}[/bold]",
        f"[bold]{board_power}[/bold]",
    )

    # Player creatures
    player_creature_str = _format_creatures(player_creatures)
    battlefield.add_row("", f"[dim]Creatures:[/dim] {player_creature_str}", "", "", "", "")

    # Player tokens
    if player_tokens:
        tokens_str = ", ".join([_format_token(t) for t in player_tokens[:3]])
        battlefield.add_row("", f"[dim]Tokens:[/dim] {tokens_str}", "", "", "", "")

    # Player graveyard
    player_gy = _format_graveyard(player_graveyard)
    battlefield.add_row("", f"[dim]Graveyard:[/dim] {player_gy}", "", "", "", "")

    # Player lands detail
    if player_lands_detail:
        lands_str = ", ".join([f"{k} x{v}" for k, v in player_lands_detail.items()])
        battlefield.add_row("", f"[dim]Lands:[/dim] {lands_str}", "", "", "", "")

    # Build header - "Turn X - Player/Opponent"
    header = Text()
    header.append(f"Turn {turn}", style="bold")
    header.append(" - ", style="dim")
    header.append(active_player, style="bold green" if active_player == "Player" else "bold red")
    header.append(" │ ", style="dim")
    header.append(phase, style="bold cyan")

    # Phase progress - combine indicator and labels
    phase_progress = Group(phase_line, phase_labels_text)

    # Action section
    action_content = ""
    if last_action:
        action_content = f"[bold yellow]→ {last_action}[/bold yellow]"

    if action_history:
        recent = action_history[-3:]
        history_lines = "\n".join([f"  [dim]•[/dim] {a}" for a in recent])
        if action_content:
            action_content += f"\n\n[dim]Recent:[/dim]\n{history_lines}"
        else:
            action_content = f"[dim]Recent Actions:[/dim]\n{history_lines}"

    # Combine into panel
    content = Group(
        header,
        Text(),  # Spacer
        phase_progress,
        Text(),  # Spacer
        battlefield,
    )

    if action_content:
        content = Group(
            header,
            Text(),
            phase_progress,
            Text(),
            battlefield,
            Text(),
            Text.from_markup(action_content),
        )

    panel = Panel(
        content,
        title="[bold]Game State[/bold]",
        border_style="blue",
    )
    console.print(panel)


def print_instant_response(
    player_name: str,
    spell_name: str,
    phase: str,
    response_to: str = "",
    mana_cost: str = "",
    card_text: str = "",
    card_type: str = "",
    resolution_effects: list[str] | None = None,
) -> None:
    """Print an instant-speed response visualization.

    Args:
        player_name: Who is casting the instant.
        spell_name: Name of the instant being cast (can include mana in parentheses for backward compat).
        phase: Current game phase.
        response_to: What this is in response to (if any).
        mana_cost: Mana cost like "1W" to be converted to symbols.
        card_text: Card rules text to display.
        card_type: Card type like "Instant".
        resolution_effects: List of resolution effect strings.

    """
    player_style = "bold green" if player_name == "Player" else "bold red"

    # Format mana cost with symbols and add type symbol if provided
    type_symbol = ""
    if card_type:
        type_symbols = {
            "creature": "⚔️",
            "instant": "✨",
            "sorcery": "🌟",
            "enchantment": "🔮",
            "artifact": "⚙️",
            "land": "🌍",
            "planeswalker": "👤",
            "token": "🎴",
        }
        type_symbol = type_symbols.get(card_type.lower(), "")

    if mana_cost:
        mana_display = format_mana_cost(mana_cost)
        spell_display = (
            f"{spell_name} ({mana_display}, {type_symbol})"
            if type_symbol
            else f"{spell_name} ({mana_display})"
        )
    else:
        spell_display = f"{spell_name} ({type_symbol})" if type_symbol else spell_name

    content = Text()
    content.append("⚡ INSTANT SPEED ⚡", style="bold yellow")
    content.append("\n\n")
    content.append(player_name, style=player_style)
    content.append(" casts ")
    content.append(spell_display, style="bold cyan")

    # Add card type and text if provided
    if card_type and card_text:
        content.append("\n")
        content.append(f"  {card_type}: {card_text}", style="dim italic")
    elif card_text:
        content.append("\n")
        content.append(f"  {card_text}", style="dim italic")

    if response_to:
        content.append("\n\n")
        content.append("In response to: ", style="dim")
        content.append(response_to)

    content.append("\n\n")
    content.append("During: ", style="dim")
    content.append(phase)

    # Add resolution effects if provided
    if resolution_effects:
        content.append("\n\n")
        # Get type symbol for the spell
        from mtg.env.card_definitions import CardRegistry

        type_symbol = ""
        try:
            card = CardRegistry.get_card(spell_name)
            if card:
                card_type_str = (
                    card.card_type.name.lower()
                    if hasattr(card.card_type, "name")
                    else str(card.card_type).lower()
                )
                type_symbols = {
                    "creature": "⚔️",
                    "instant": "✨",
                    "sorcery": "🌟",
                    "enchantment": "🔮",
                    "artifact": "⚙️",
                    "land": "🌍",
                    "planeswalker": "👤",
                    "token": "🎴",
                }
                for key, sym in type_symbols.items():
                    if key in card_type_str:
                        type_symbol = sym
                        break
        except Exception:
            type_symbol = "✨"  # Default for instants

        resolve_display = (
            f"{spell_name} ({format_mana_cost(mana_cost)}, {type_symbol})"
            if mana_cost
            else f"{spell_name} ({type_symbol})"
        )
        content.append(f"{resolve_display} resolves:", style="bold")
        for effect in resolution_effects:
            content.append(f"\n  -> {effect}")

    panel = Panel(
        content,
        title="[bold yellow]Priority Response[/bold yellow]",
        border_style="yellow",
    )
    console.print(panel)


def print_priority_window(
    phase: str,
    active_player: str,
    priority_player: str,
    can_respond: bool = True,
) -> None:
    """Print a priority window indicator.

    Args:
        phase: Current game phase.
        active_player: Whose turn it is.
        priority_player: Who has priority.
        can_respond: Whether responses are possible.

    """
    priority_style = "bold green" if priority_player == "Player" else "bold red"

    content = Text()
    content.append("Phase: ", style="dim")
    content.append(phase, style="cyan")
    content.append("  │  Active: ")
    content.append(active_player)
    content.append("  │  ")

    if can_respond:
        content.append(priority_player, style=priority_style)
        content.append(" has priority")
        content.append("\n")
        content.append("Cast instants or pass", style="dim")
    else:
        content.append("Priority passed", style="dim")

    console.print(
        Panel(
            content,
            title="[bold blue]Priority Window[/bold blue]",
            border_style="blue",
            padding=(0, 1),
        )
    )


def print_stack_item(
    spell_name: str,
    controller: str,
    targets: str = "",
) -> None:
    """Print a spell on the stack.

    Args:
        spell_name: Name of the spell.
        controller: Who controls it.
        targets: Target description.

    """
    controller_style = "green" if controller == "Player" else "red"

    content = f"[bold]{spell_name}[/bold] (controlled by [{controller_style}]{controller}[/{controller_style}])"
    if targets:
        content += f"\n[dim]Targets: {targets}[/dim]"

    console.print(f"  [yellow]📜[/yellow] {content}")


def _format_graveyard_with_counts(graveyard: list[tuple[str, str]]) -> str:
    """Format graveyard with card type counts.

    Args:
        graveyard: List of (card_name, card_type) tuples.

    Returns:
        Formatted string with cards and type counts.

    """
    if not graveyard:
        return ""

    # Count by type
    type_counts: dict[str, int] = {}
    type_icons = {
        "creature": "⚔️",
        "instant": "✨",
        "sorcery": "🌟",
        "enchantment": "🔮",
        "artifact": "⚙️",
        "land": "🌍",
        "planeswalker": "👤",
        "token": "🎴",
    }

    for _, card_type in graveyard:
        card_type_lower = card_type.lower()
        type_counts[card_type_lower] = type_counts.get(card_type_lower, 0) + 1

    # Format counts
    count_parts = []
    for type_name, count in type_counts.items():
        icon = type_icons.get(type_name, "❓")
        count_parts.append(f"{icon}{count}")

    # Format each card name with proper display format
    card_displays = [_format_card_with_type(name) for name, _ in graveyard]
    return f"{', '.join(card_displays)} ({' '.join(count_parts)})"


def print_turn_summary(
    turn: int,
    player_actions: list[tuple[str, str]],
    opponent_actions: list[tuple[str, str]],
    player_life_change: int = 0,
    opponent_life_change: int = 0,
    player_life: int = 20,
    opponent_life: int = 20,
    player_lands: dict[str, int] | None = None,
    opponent_lands: dict[str, int] | None = None,
    player_hand: list[tuple[str, str]] | None = None,
    opponent_hand_count: int = 0,
    opponent_hand: list[tuple[str, str]] | None = None,
    player_creatures: list[str] | None = None,
    opponent_creatures: list[str] | None = None,
    player_enchantments: list[str] | None = None,
    opponent_enchantments: list[str] | None = None,
    player_graveyard: list[tuple[str, str]] | None = None,
    opponent_graveyard: list[tuple[str, str]] | None = None,
    player_exile: list[str] | None = None,
    opponent_exile: list[str] | None = None,
    player_tokens: list[str] | None = None,
    opponent_tokens: list[str] | None = None,
) -> None:
    """Print a summary of a complete turn with full board state.

    Args:
        turn: Turn number.
        player_actions: List of (phase, action) tuples for player.
        opponent_actions: List of (phase, action) tuples for opponent.
        player_life_change: Net life change for player.
        opponent_life_change: Net life change for opponent.
        player_life: Current player life total.
        opponent_life: Current opponent life total.
        player_lands: Player's lands by type (e.g., {"Mountain": 2}).
        opponent_lands: Opponent's lands by type.
        player_hand: Player's hand cards (name, mana).
        opponent_hand_count: Number of cards in opponent's hand (fallback if opponent_hand not provided).
        opponent_hand: Opponent's hand cards (name, mana) - shown with (hidden) indicator.
        player_creatures: Player's creature names.
        opponent_creatures: Opponent's creature names.
        player_enchantments: Player's enchantment names.
        opponent_enchantments: Opponent's enchantment names.
        player_graveyard: Cards in player's graveyard as (name, type) tuples.
        opponent_graveyard: Cards in opponent's graveyard as (name, type) tuples.
        player_exile: Cards in player's exile.
        opponent_exile: Cards in opponent's exile.
        player_tokens: Player's tokens.
        opponent_tokens: Opponent's tokens.

    """
    # Build action tables - half terminal width each
    half_width = max(40, (console.width - 10) // 2)

    player_table = Table(
        title="[bold green]Player's Turn[/bold green]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold green",
        border_style="green",
        width=half_width,
    )
    player_table.add_column("Phase", style="green", width=12)
    player_table.add_column("Action", style="white", overflow="fold")

    for phase, action in player_actions:
        player_table.add_row(phase, action)

    if not player_actions:
        player_table.add_row("—", "[dim]No actions[/dim]")

    opponent_table = Table(
        title="[bold red]Opponent's Turn[/bold red]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold red",
        border_style="red",
        width=half_width,
    )
    opponent_table.add_column("Phase", style="red", width=12)
    opponent_table.add_column("Action", style="white", overflow="fold")

    for phase, action in opponent_actions:
        opponent_table.add_row(phase, action)

    if not opponent_actions:
        opponent_table.add_row("—", "[dim]No actions[/dim]")

    # Life change summary
    p_change = (
        f"[green]+{player_life_change}[/green]"
        if player_life_change > 0
        else f"[red]{player_life_change}[/red]"
        if player_life_change < 0
        else "[dim]0[/dim]"
    )
    o_change = (
        f"[green]+{opponent_life_change}[/green]"
        if opponent_life_change > 0
        else f"[red]{opponent_life_change}[/red]"
        if opponent_life_change < 0
        else "[dim]0[/dim]"
    )

    # Life bar visualization
    p_bar = _life_bar(player_life)
    o_bar = _life_bar(opponent_life)

    # Build player board state with ORDERED categories (always shown, even if empty)
    p_board_lines = []
    # 1. Lands - format each land type with card display
    if player_lands:
        p_lands_parts = [
            f"{_format_card_with_type(name)} x{count}" for name, count in player_lands.items()
        ]
        p_board_lines.append(f"[dim]🌍 Lands:[/dim] [green]{', '.join(p_lands_parts)}[/green]")
    else:
        p_board_lines.append("[dim]🌍 Lands:[/dim] [dim]None[/dim]")
    # 2. Creatures - already formatted when passed in
    if player_creatures:
        p_board_lines.append(
            f"[dim]⚔️ Creatures:[/dim] [green]{', '.join(player_creatures)}[/green]"
        )
    else:
        p_board_lines.append("[dim]⚔️ Creatures:[/dim] [dim]None[/dim]")
    # 3. Enchantments - format each with card display
    if player_enchantments:
        ench_parts = [_format_card_with_type(name) for name in player_enchantments]
        p_board_lines.append(f"[dim]🔮 Enchantments:[/dim] [green]{', '.join(ench_parts)}[/green]")
    else:
        p_board_lines.append("[dim]🔮 Enchantments:[/dim] [dim]None[/dim]")
    # 4. Artifacts (placeholder - not yet tracked separately)
    p_board_lines.append("[dim]⚙️ Artifacts:[/dim] [dim]None[/dim]")
    # 5. Tokens - format with counts
    if player_tokens:
        # Count each unique token and format
        token_counts: dict[str, int] = {}
        for t in player_tokens:
            token_counts[t] = token_counts.get(t, 0) + 1
        token_parts = [
            f"{name} x{count}" if count > 1 else name for name, count in token_counts.items()
        ]
        p_board_lines.append(f"[dim]🎴 Tokens:[/dim] [green]{', '.join(token_parts)}[/green]")
    else:
        p_board_lines.append("[dim]🎴 Tokens:[/dim] [dim]None[/dim]")
    # 6. Hand
    if player_hand:
        hand_parts = [_format_card_with_type(name) for name, _mana in player_hand]
        p_board_lines.append(
            f"[dim]🃏 Hand ({len(player_hand)}):[/dim] [green]{', '.join(hand_parts)}[/green]"
        )
    else:
        p_board_lines.append("[dim]🃏 Hand:[/dim] [dim]Empty[/dim]")
    # 7. Graveyard
    if player_graveyard:
        gy_str = _format_graveyard_with_counts(player_graveyard)
        p_board_lines.append(f"[dim]💀 Graveyard:[/dim] [green]{gy_str}[/green]")
    else:
        p_board_lines.append("[dim]💀 Graveyard:[/dim] [dim]Empty[/dim]")
    # 8. Exile
    if player_exile:
        exile_parts = [_format_card_with_type(name) for name in player_exile]
        p_board_lines.append(f"[dim]✨ Exile:[/dim] [green]{', '.join(exile_parts)}[/green]")
    else:
        p_board_lines.append("[dim]✨ Exile:[/dim] [dim]None[/dim]")

    # Build opponent board state with ORDERED categories (always shown, even if empty)
    o_board_lines = []
    # 1. Lands - format each land type with card display
    if opponent_lands:
        o_lands_parts = [
            f"{_format_card_with_type(name)} x{count}" for name, count in opponent_lands.items()
        ]
        o_board_lines.append(f"[dim]🌍 Lands:[/dim] [red]{', '.join(o_lands_parts)}[/red]")
    else:
        o_board_lines.append("[dim]🌍 Lands:[/dim] [dim]None[/dim]")
    # 2. Creatures - already formatted when passed in
    if opponent_creatures:
        o_board_lines.append(f"[dim]⚔️ Creatures:[/dim] [red]{', '.join(opponent_creatures)}[/red]")
    else:
        o_board_lines.append("[dim]⚔️ Creatures:[/dim] [dim]None[/dim]")
    # 3. Enchantments - format each with card display
    if opponent_enchantments:
        ench_parts = [_format_card_with_type(name) for name in opponent_enchantments]
        o_board_lines.append(f"[dim]🔮 Enchantments:[/dim] [red]{', '.join(ench_parts)}[/red]")
    else:
        o_board_lines.append("[dim]🔮 Enchantments:[/dim] [dim]None[/dim]")
    # 4. Artifacts (placeholder - not yet tracked separately)
    o_board_lines.append("[dim]⚙️ Artifacts:[/dim] [dim]None[/dim]")
    # 5. Tokens - format with counts
    if opponent_tokens:
        # Count each unique token and format
        opp_token_counts: dict[str, int] = {}
        for t in opponent_tokens:
            opp_token_counts[t] = opp_token_counts.get(t, 0) + 1
        token_parts = [
            f"{name} x{count}" if count > 1 else name for name, count in opp_token_counts.items()
        ]
        o_board_lines.append(f"[dim]🎴 Tokens:[/dim] [red]{', '.join(token_parts)}[/red]")
    else:
        o_board_lines.append("[dim]🎴 Tokens:[/dim] [dim]None[/dim]")
    # 6. Hand
    if opponent_hand:
        hand_parts = [_format_card_with_type(name) for name, _mana in opponent_hand]
        o_board_lines.append(
            f"[dim]🃏 Hand ({len(opponent_hand)}):[/dim] [dim italic]{', '.join(hand_parts)} (hidden)[/dim italic]"
        )
    else:
        o_board_lines.append(f"[dim]🃏 Hand:[/dim] [red]{opponent_hand_count} cards (hidden)[/red]")
    # 7. Graveyard
    if opponent_graveyard:
        gy_str = _format_graveyard_with_counts(opponent_graveyard)
        o_board_lines.append(f"[dim]💀 Graveyard:[/dim] [red]{gy_str}[/red]")
    else:
        o_board_lines.append("[dim]💀 Graveyard:[/dim] [dim]Empty[/dim]")
    # 8. Exile
    if opponent_exile:
        exile_parts = [_format_card_with_type(name) for name in opponent_exile]
        o_board_lines.append(f"[dim]✨ Exile:[/dim] [red]{', '.join(exile_parts)}[/red]")
    else:
        o_board_lines.append("[dim]✨ Exile:[/dim] [dim]None[/dim]")

    p_board_str = "\n".join(p_board_lines) if p_board_lines else "[dim]No permanents[/dim]"
    o_board_str = "\n".join(o_board_lines) if o_board_lines else "[dim]No permanents[/dim]"

    summary_text = f"""
Life Totals: [green]Player[/green] {p_bar} [bold]{player_life}[/bold] ({p_change})  │  [red]Opponent[/red] {o_bar} [bold]{opponent_life}[/bold] ({o_change})

[bold green]Player Board:[/bold green]
{p_board_str}

[bold red]Opponent Board:[/bold red]
{o_board_str}
    """.strip()

    # Card type legend
    legend = "[dim]Card Types: ⚔️=Creatures ✨=Instants 🌟=Sorceries 🔮=Enchantments ⚙️=Artifacts 🌍=Lands 👤=Planeswalkers 🎴=Tokens[/dim]"

    console.print()
    console.print(
        Panel(
            Group(
                Columns([player_table, opponent_table], padding=(0, 2)),
                Text(),
                Text.from_markup(summary_text),
                Text(),
                Text.from_markup(legend),
            ),
            title=f"[bold cyan]Turn {turn} Complete[/bold cyan]",
            border_style="cyan",
            width=console.width,
        )
    )
    console.print()

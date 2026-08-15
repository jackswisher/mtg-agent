#!/usr/bin/env python3
"""Gameplay script for MTG-Causal-RL.

This script runs interactive MTG games with visualization and replay recording.
All game logic is handled by MTGEnv to ensure consistency with training.

Usage:
    uv run python scripts/runner/run_gameplay.py
    uv run mtg-gameplay
"""

from __future__ import annotations

import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from rich import box
from rich.columns import Columns
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.table import Table

from mtg.agents import get_agent, list_agents
from mtg.env.card_definitions import CardRegistry, CardType
from mtg.env.deck_archetypes import get_archetype, list_archetypes
from mtg.env.mtg_env import MTGEnv
from mtg.utils.cli_display import (
    CARD_TYPE_LEGEND,
    console,
    format_mana_cost,
    print_divider,
    print_logo,
    print_turn_summary,
)
from mtg.utils.html_report import (
    GameRecorder,
    generate_html_report,
)
from mtg.utils.interactive import GameplayConfig

# Speed presets (seconds)
SPEED_PRESETS = {
    "slow": {"phase": 3.0, "action": 2.0, "turn": 1.0},
    "medium": {"phase": 1.5, "action": 1.0, "turn": 0.5},
    "fast": {"phase": 0.3, "action": 0.1, "turn": 0.1},
    "instant": {"phase": 0.0, "action": 0.0, "turn": 0.0},
}

CARD_TYPE_SYMBOLS = {
    CardType.CREATURE: "⚔️",
    CardType.INSTANT: "✨",
    CardType.SORCERY: "🌟",
    CardType.ENCHANTMENT: "🔮",
    CardType.ARTIFACT: "⚙️",
    CardType.LAND: "🌍",
}


# =============================================================================
# Text Helpers
# =============================================================================


def _strip_rich_markup(text: str) -> str:
    """Strip Rich markup tags from text while preserving content."""
    cleaned = re.sub(r"\[[^\]]*\]", "", text)
    return re.sub(r"\s+", " ", cleaned).strip()


# =============================================================================
# Card Formatting Helpers
# =============================================================================


def format_card_display(
    card_name: str,
    power: int | None = None,
    toughness: int | None = None,
    attached_tokens: list[str] | None = None,
    graveyard_instant_sorcery_count: int | None = None,
) -> str:
    """Format card name with mana cost, type symbol, P/T, and attached tokens.

    Standard format:
    - Creatures: "Name - P/T (mana, ⚔️)" or "Name - P/T (mana, ⚔️) [🎴Monster Role]"
    - Non-creatures: "Name (mana, symbol)"
    - Lands: "Name (🌍)"

    For Haughty Djinn, if graveyard_instant_sorcery_count is provided, use that as power.
    """
    try:
        registry = CardRegistry.get_instance()
        card = registry.get(card_name)
    except (KeyError, AttributeError):
        if power is not None and toughness is not None:
            return f"{card_name} - {power}/{toughness} (🎴)"
        return f"{card_name} (🎴)"

    mana_str = ""
    if hasattr(card, "mana_cost") and card.mana_cost:
        mana_str = format_mana_cost(card.mana_cost.to_text())

    type_symbol = CARD_TYPE_SYMBOLS.get(card.card_type, "🃏")

    # Format attachment string if present
    attachment_str = ""
    if attached_tokens:
        token_strs = [f"🎴{token}" for token in attached_tokens]
        attachment_str = f" [{', '.join(token_strs)}]"

    if card.card_type == CardType.CREATURE:
        p = power if power is not None else card.power
        t = toughness if toughness is not None else card.toughness
        # Handle Haughty Djinn's variable power
        if card_name == "Haughty Djinn" and graveyard_instant_sorcery_count is not None:
            p = graveyard_instant_sorcery_count
        if mana_str:
            return f"{card_name} - {p}/{t} ({mana_str}, {type_symbol}){attachment_str}"
        return f"{card_name} - {p}/{t} ({type_symbol}){attachment_str}"
    if mana_str:
        return f"{card_name} ({mana_str}, {type_symbol})"
    return f"{card_name} ({type_symbol})"


def format_creature_from_card(card) -> str:
    """Format a creature Card object with current stats and attached tokens.

    Args:
        card: A Card object from the battlefield.

    Returns:
        Formatted string with P/T (including bonuses) and attached tokens.
    """
    # Get current power/toughness (including temporary and permanent bonuses)
    if card.current_power is not None:
        p = card.current_power
    else:
        p = card.power + getattr(card, "permanent_power_bonus", 0)

    if card.current_toughness is not None:
        t = card.current_toughness
    else:
        t = card.toughness + getattr(card, "permanent_toughness_bonus", 0)

    attached = getattr(card, "attached_tokens", [])

    return format_card_display(card.name, power=p, toughness=t, attached_tokens=attached)


def get_lands_by_type(battlefield: list) -> dict[str, int]:
    """Get land counts by type from battlefield."""
    lands: dict[str, int] = {}
    for card in battlefield:
        if card.card_type == CardType.LAND or card.produces_mana:
            lands[card.name] = lands.get(card.name, 0) + 1
    return lands


def get_card_text(card_name: str) -> str | None:
    """Get rules text for a card."""
    try:
        registry = CardRegistry.get_instance()
        card = registry.get(card_name)
        if hasattr(card, "rules_text") and card.rules_text:
            return card.rules_text
    except (KeyError, AttributeError):
        pass
    return None


def is_land_card(card_name: str) -> bool:
    """Check if a card is a land."""
    try:
        registry = CardRegistry.get_instance()
        card = registry.get(card_name)
        return card.card_type == CardType.LAND
    except (KeyError, AttributeError):
        # Fallback: common land names
        land_names = {"Mountain", "Island", "Plains", "Forest", "Swamp"}
        return card_name in land_names


def is_creature_card(card_name: str) -> bool:
    """Check if a card is a creature."""
    try:
        registry = CardRegistry.get_instance()
        card = registry.get(card_name)
        return card.card_type == CardType.CREATURE
    except (KeyError, AttributeError):
        return False


def get_mana_cost(card_name: str) -> str:
    """Get mana cost text for a card."""
    try:
        registry = CardRegistry.get_instance()
        card = registry.get(card_name)
        if hasattr(card, "mana_cost") and card.mana_cost:
            return card.mana_cost.to_text()
    except (KeyError, AttributeError):
        pass
    return ""


def get_card_type_str(card_name: str) -> str:
    """Get card type string for a card."""
    try:
        registry = CardRegistry.get_instance()
        card = registry.get(card_name)
        if hasattr(card, "card_type"):
            return card.card_type.value
    except (KeyError, AttributeError):
        pass
    return ""


# =============================================================================
# Turn Headers
# =============================================================================


def print_player_turn_header(turn: int) -> None:
    """Print player turn header with green styling."""
    console.print(
        Panel(
            f"[bold green]Turn {turn} - Player[/bold green]",
            border_style="green",
            width=console.width,
        )
    )


def print_opponent_turn_header(turn: int) -> None:
    """Print opponent turn header with red styling."""
    console.print(
        Panel(
            f"[bold red]Turn {turn} - Opponent[/bold red]",
            border_style="red",
            width=console.width,
        )
    )


def print_play_draw_selection_full(player_on_play: bool) -> None:
    """Print play/draw selection with full width."""
    if player_on_play:
        result = "[bold green]You are ON THE PLAY[/bold green] (go first, no draw T1)"
        coin = "🪙 Coin flip: Heads"
    else:
        result = "[bold blue]You are ON THE DRAW[/bold blue] (go second, draw T1)"
        coin = "🪙 Coin flip: Tails"

    content = f"""[bold cyan]GAME START[/bold cyan]

{coin}

{result}"""

    console.print(
        Panel(
            content,
            title="[bold cyan]Play/Draw Selection[/bold cyan]",
            title_align="center",
            border_style="cyan",
            width=console.width,
        )
    )


# =============================================================================
# Phase Boxes
# =============================================================================


def print_phase_box(
    phase: str,
    player: str,
    actions: list[str] | None = None,
) -> None:
    """Print a phase box with actions.

    Player phases: green border
    Opponent phases: red border

    Note: Actions should already be formatted with colors from format_action_for_display.
    """
    is_player = player == "Player"
    border_style = "green" if is_player else "red"
    title_style = "bold green" if is_player else "bold red"

    content_parts = []
    if actions:
        for action in actions:
            # Actions are already formatted with colors, just add them
            content_parts.append(action)
    else:
        content_parts.append("[dim]Pass[/dim]")

    content = "\n".join(content_parts)

    console.print(
        Panel(
            content,
            title=f"[{title_style}]{phase}[/{title_style}]",
            title_align="center",
            border_style=border_style,
            width=console.width,
        )
    )


def get_new_actions_from_log(prev_info: dict, curr_info: dict) -> list[dict]:
    """Get new actions from the action log by comparing before and after."""
    prev_log = prev_info.get("action_log", [])
    curr_log = curr_info.get("action_log", [])

    # Return new actions (those in curr but not in prev)
    prev_count = len(prev_log)
    return curr_log[prev_count:]


def format_logged_action(action: dict, active_player_idx: int | None = None) -> str:
    """Format a logged action for display with proper coloring.

    Action types: yellow bold
    Card names: white (via format_card_display)
    Details/card text: dim grey
    Triggers: cyan

    Args:
        action: The action dict from the game log.
        active_player_idx: The index of the player whose turn it is (0=Player, 1=Opponent).
            If provided and the actor is different, a [Player]/[Opponent] prefix is added.
    """
    action_type = action.get("action_type", "ACTION")
    card_name = action.get("card_name", "")
    details = action.get("details", {})
    actor_idx = action.get("player", 0)

    # Determine if we need a caster prefix (when casting during opponent's turn)
    caster_prefix = ""
    if active_player_idx is not None and actor_idx != active_player_idx:
        if actor_idx == 0:
            caster_prefix = "[bold green][Player][/bold green] "
        else:
            caster_prefix = "[bold red][Opponent][/bold red] "

    def _normalize_target_label(raw_target: str) -> str:
        """Normalize target labels to Player/Opponent from viewer perspective."""
        lower = raw_target.lower()
        if lower in {"you", "player"}:
            return "Player" if actor_idx == 0 else "Opponent"
        if lower in {"opponent"}:
            return "Opponent" if actor_idx == 0 else "Player"
        return raw_target

    def _colorize(text: str, owner_idx: int | None) -> str:
        if owner_idx is None:
            return text
        color = "green" if owner_idx == 0 else "red"
        return f"[{color}]{text}[/{color}]"

    if action_type == "PLAY_LAND":
        # Show the land being played and total lands on board
        all_lands = details.get("all_lands", {})
        if all_lands:
            lands_str = ", ".join(
                f"{format_card_display(name)} x{count}" for name, count in all_lands.items()
            )
        else:
            land_count = details.get("land_count", 1)
            lands_str = f"{format_card_display(card_name)} x{land_count}"
        card_display = format_card_display(card_name)
        enters_tapped = details.get("enters_tapped", False)
        tapped_str = " [dim italic](enters tapped)[/dim italic]" if enters_tapped else ""
        return (
            f"[bold yellow]PLAY_LAND:[/bold yellow] {card_display}{tapped_str} "
            f"[dim](Board: {lands_str})[/dim]"
        )

    elif action_type == "CAST":
        triggered_abilities = details.get("triggered_abilities", [])
        target = details.get("target", "")
        # For Haughty Djinn, use graveyard count to show expected power
        gy_count = details.get("graveyard_instant_sorcery_count")

        card_disp = format_card_display(card_name, graveyard_instant_sorcery_count=gy_count)
        result = f"{caster_prefix}[bold yellow]CAST:[/bold yellow] {card_disp}"

        # Show target if any
        if target:
            target_label = _normalize_target_label(target)
            # Handle player targets vs creature targets
            if target_label in {"Opponent", "Player"}:
                target_style = "bold red" if target_label == "Opponent" else "bold green"
                result += f" targeting [{target_style}]{target_label}[/{target_style}]"
            else:
                target_power = details.get("target_power")
                target_toughness = details.get("target_toughness")
                target_tokens = details.get("target_tokens", [])
                # Determine ownership prefix for creature targets
                target_owner = details.get("target_owner")
                ownership_prefix = ""
                if target_owner is not None and details.get("target_kind") == "creature":
                    # From the perspective of the viewer (player 0)
                    if actor_idx == 0:
                        # Player is casting: target_owner != actor means opponent's creature
                        ownership_prefix = (
                            "[bold red]Opponent's[/bold red] " if target_owner != actor_idx else ""
                        )
                    else:
                        # Opponent is casting: target_owner == 0 means player's creature
                        ownership_prefix = (
                            "[bold green]Your[/bold green] " if target_owner == 0 else ""
                        )
                result += (
                    " targeting "
                    + ownership_prefix
                    + format_card_display(
                        target_label,
                        power=target_power,
                        toughness=target_toughness,
                        attached_tokens=target_tokens,
                    )
                )

        # Show card text (but not for lands)
        if not is_land_card(card_name):
            card_text = get_card_text(card_name)
            if card_text:
                result += f"\n  [dim]→ {card_text}[/dim]"

        # Show damage resolution with purple color for spell damage
        try:
            card_data = CardRegistry.get_instance().get(card_name)
        except KeyError:
            card_data = None
        if card_data and card_data.deals_damage > 0:
            dmg = card_data.deals_damage
            if target:
                result += f"\n  [bold magenta]→ Deals {dmg} damage to {target_label}[/bold magenta]"
            elif card_data.can_target_any:
                result += f"\n  [bold magenta]→ Deals {dmg} damage to Opponent[/bold magenta]"

        # Show triggered abilities
        for trigger_desc in triggered_abilities:
            result += f"\n  [cyan]→ TRIGGER: {trigger_desc}[/cyan]"

        return result

    elif action_type == "ATTACK":
        attacker_data = details.get("attacker_data", [])
        triggered_abilities = details.get("triggered_abilities", [])

        if attacker_data:
            # Use attacker_data with current power/toughness
            formatted_attackers = []
            for atk in attacker_data:
                name = atk.get("name", "")
                power = atk.get("power")
                toughness = atk.get("toughness")
                tokens = atk.get("tokens", [])
                formatted_attackers.append(
                    format_card_display(name, power, toughness, attached_tokens=tokens)
                )
            attackers_joined = ", ".join(formatted_attackers)
            result = f"[bold yellow]ATTACK:[/bold yellow] Attacking with {attackers_joined}"
        else:
            # Fallback to old format for backwards compatibility
            attackers = details.get("attackers", [card_name] if card_name else [])
            if attackers:
                formatted_attackers = [format_card_display(a) for a in attackers]
                attackers_joined = ", ".join(formatted_attackers)
                result = f"[bold yellow]ATTACK:[/bold yellow] Attacking with {attackers_joined}"
            else:
                result = "[bold yellow]ATTACK:[/bold yellow] [dim]No attackers[/dim]"

        for trigger_desc in triggered_abilities:
            result += f"\n  [cyan]→ TRIGGER: {trigger_desc}[/cyan]"

        return result

    elif action_type == "TRIGGER":
        trigger_type = details.get("trigger_type", "")
        effect_applied = details.get("effect_applied", "")
        description = details.get("description", "")

        return (
            f"[cyan]→ TRIGGER ({trigger_type}):[/cyan] [dim]{effect_applied or description}[/dim]"
        )

    elif action_type == "RESOLVE":
        target = details.get("target", "")
        new_power = details.get("new_power")
        new_toughness = details.get("new_toughness")
        tokens = details.get("tokens", [])
        buff_power = details.get("buff_power")
        buff_toughness = details.get("buff_toughness")
        buff_result_power = details.get("buff_result_power")
        buff_result_toughness = details.get("buff_result_toughness")
        role_bonus = details.get("role_bonus", 0)
        role_result_power = details.get("role_result_power")
        role_result_toughness = details.get("role_result_toughness")
        if target:
            extra_lines = []
            if buff_power is not None or buff_toughness is not None:
                buff_target = format_card_display(
                    target, power=buff_result_power, toughness=buff_result_toughness
                )
                extra_lines.append(
                    "[bold magenta]BUFF:[/bold magenta] "
                    f"{target} gets +{buff_power}/+{buff_toughness} "
                    f"(now {buff_target})"
                )
            if role_bonus:
                role_target = format_card_display(
                    target, power=role_result_power, toughness=role_result_toughness
                )
                extra_lines.append(
                    "[bold magenta]BUFF:[/bold magenta] "
                    f"{target} gets +{role_bonus}/+{role_bonus} from Monster Role "
                    f"(now {role_target})"
                )
            target_str = format_card_display(
                target,
                power=new_power,
                toughness=new_toughness,
                attached_tokens=tokens,
            )
            resolve_line = (
                "[bold yellow]RESOLVE:[/bold yellow] "
                f"{format_card_display(card_name)} → {target_str}"
            )
            if extra_lines:
                return "\n".join(extra_lines + [resolve_line])
            return resolve_line
        return f"[bold yellow]RESOLVE:[/bold yellow] {format_card_display(card_name)}"

    elif action_type == "DRAW":
        result = f"[bold yellow]DRAW:[/bold yellow] {format_card_display(card_name)}"
        # Only show card text for non-land cards
        if not is_land_card(card_name):
            card_text = get_card_text(card_name)
            if card_text:
                result += f"\n  [dim]→ {card_text}[/dim]"
        return result

    elif action_type == "DRAW_SELECTION":
        # Memory Deluge and similar "look at X, choose Y" effects
        cards_drawn = details.get("cards_drawn", [])
        count = details.get("count", 0)
        if cards_drawn:
            card_strs = ", ".join(format_card_display(c) for c in cards_drawn)
            return (
                f"[bold cyan]DRAW_SELECTION:[/bold cyan] {format_card_display(card_name)} "
                f"→ Drew {count} card(s): {card_strs}"
            )
        return (
            f"[bold cyan]DRAW_SELECTION:[/bold cyan] "
            f"{format_card_display(card_name)} → Drew {count} card(s)"
        )

    elif action_type == "BLOCK":
        block_data = details.get("block_data", [])
        if block_data:
            block_strs = []
            for entry in block_data:
                blocker_owner = entry.get("blocker_owner_idx", action.get("player"))
                attacker_owner = entry.get("attacker_owner_idx", 1 - action.get("player", 0))
                blocker = format_card_display(
                    entry.get("blocker", ""),
                    entry.get("blocker_power"),
                    entry.get("blocker_toughness"),
                    attached_tokens=entry.get("blocker_tokens", []),
                )
                attacker = format_card_display(
                    entry.get("attacker", ""),
                    entry.get("attacker_power"),
                    entry.get("attacker_toughness"),
                    attached_tokens=entry.get("attacker_tokens", []),
                )
                block_strs.append(
                    f"{_colorize(blocker, blocker_owner)} blocks "
                    f"{_colorize(attacker, attacker_owner)}"
                )
            return f"[bold yellow]BLOCK:[/bold yellow] {', '.join(block_strs)}"
        blocks = details.get("blocks", [])
        if blocks:
            block_strs = [
                f"{_colorize(format_card_display(b[0]), action.get('player'))} blocks "
                f"{_colorize(format_card_display(b[1]), 1 - action.get('player', 0))}"
                for b in blocks
            ]
            return f"[bold yellow]BLOCK:[/bold yellow] {', '.join(block_strs)}"
        return f"[bold yellow]BLOCK:[/bold yellow] {card_name} blocks"

    elif action_type == "DAMAGE":
        attacker_data = details.get("attacker_data", [])
        block_data = details.get("block_data", [])
        events = details.get("events", [])
        attacker_owner = action.get("player", 0)
        defender_owner = 1 - attacker_owner

        # If we have attacker_data, use it for proper display with current stats
        if attacker_data:
            formatted_events = []
            for atk in attacker_data:
                name = atk.get("name", "Unknown")
                power = atk.get("power", 0)
                toughness = atk.get("toughness", 0)
                damage = atk.get("damage", power)
                tokens = atk.get("tokens", [])
                formatted_creature = format_card_display(
                    name, power, toughness, attached_tokens=tokens
                )
                formatted_events.append(
                    f"{_colorize(formatted_creature, attacker_owner)} deals {damage} damage"
                )

            # Add death triggers and other events
            for event in events:
                event_str = str(event)
                if "Death trigger" in event_str:
                    formatted_events.append(f"[cyan]{event_str}[/cyan]")
                elif " dies" in event_str:
                    creature_name = event_str.replace(" dies", "").strip()
                    formatted_events.append(f"{format_card_display(creature_name)} dies")
                elif " blocked by " in event_str:
                    parts = event_str.split(" blocked by ")
                    attacker_name = parts[0].strip()
                    blocker_names = []
                    if len(parts) > 1:
                        blocker_names = [n.strip() for n in parts[1].split(",")]
                    if block_data:
                        matched = []
                        for blocker_name in blocker_names:
                            entry = next(
                                (
                                    b
                                    for b in block_data
                                    if b.get("attacker") == attacker_name
                                    and b.get("blocker") == blocker_name
                                ),
                                None,
                            )
                            if entry:
                                atk_fmt = format_card_display(
                                    entry.get("attacker", ""),
                                    entry.get("attacker_power"),
                                    entry.get("attacker_toughness"),
                                    attached_tokens=entry.get("attacker_tokens", []),
                                )
                                blk_fmt = format_card_display(
                                    entry.get("blocker", ""),
                                    entry.get("blocker_power"),
                                    entry.get("blocker_toughness"),
                                    attached_tokens=entry.get("blocker_tokens", []),
                                )
                                matched.append(
                                    f"{_colorize(atk_fmt, attacker_owner)} blocked by "
                                    f"{_colorize(blk_fmt, defender_owner)}"
                                )
                        if matched:
                            formatted_events.extend(matched)
                            continue
                    atk_fmt = format_card_display(attacker_name)
                    blk_fmt = format_card_display(", ".join(blocker_names)) if blocker_names else ""
                    formatted_events.append(
                        f"{_colorize(atk_fmt, attacker_owner)} blocked by "
                        f"{_colorize(blk_fmt, defender_owner)}"
                    )

            return f"[bold yellow]DAMAGE:[/bold yellow] {'; '.join(formatted_events)}"

        # Fallback to old parsing method
        if events:
            formatted_events = []
            for event in events:
                event_str = str(event)
                if "Death trigger" in event_str:
                    formatted_events.append(f"[cyan]{event_str}[/cyan]")
                elif " deals " in event_str and " damage" in event_str:
                    parts = event_str.split(" deals ")
                    creature_name = parts[0].strip()
                    damage_part = parts[1] if len(parts) > 1 else ""
                    formatted_creature = format_card_display(creature_name)
                    formatted_events.append(f"{formatted_creature} deals {damage_part}")
                elif " blocked by " in event_str:
                    parts = event_str.split(" blocked by ")
                    attacker_name = parts[0].strip()
                    blocker_name = parts[1].strip() if len(parts) > 1 else ""
                    atk_fmt = format_card_display(attacker_name)
                    blk_fmt = format_card_display(blocker_name)
                    formatted_events.append(f"{atk_fmt} blocked by {blk_fmt}")
                elif " dies" in event_str:
                    creature_name = event_str.replace(" dies", "").strip()
                    formatted_events.append(f"{format_card_display(creature_name)} dies")
                else:
                    formatted_events.append(event_str)
            return f"[bold yellow]DAMAGE:[/bold yellow] {'; '.join(formatted_events)}"
        return "[bold yellow]DAMAGE:[/bold yellow] [dim]No combat damage[/dim]"

    elif action_type == "UNTAP":
        untapped = details.get("untapped", [])
        untapped_data = details.get("untapped_data", [])
        lands_on_board = details.get("lands_on_board", {})

        result = "[bold yellow]UNTAP:[/bold yellow]"
        if untapped_data:
            formatted_untapped = [
                format_card_display(
                    entry.get("name", ""),
                    entry.get("power"),
                    entry.get("toughness"),
                    attached_tokens=entry.get("tokens", []),
                )
                for entry in untapped_data
            ]
            result += f" Untapped: {', '.join(formatted_untapped)}"
        elif untapped:
            formatted_untapped = [format_card_display(name) for name in untapped]
            result += f" Untapped: {', '.join(formatted_untapped)}"
        else:
            result += " [dim]No tapped permanents[/dim]"

        if lands_on_board:
            lands_str = ", ".join(
                f"{format_card_display(name)} x{count}" for name, count in lands_on_board.items()
            )
            result += f"\n  [dim](Board: {lands_str})[/dim]"

        return result

    elif action_type == "CLEANUP":
        effects_removed = details.get("effects_removed", [])
        discarded_cards = details.get("discarded_cards", [])
        lines = []

        # Build summary parts for first line
        summary_parts = []
        if effects_removed:
            creature_names = [effect.get("creature", "?") for effect in effects_removed]
            summary_parts.append(f"Buffs expire: {', '.join(creature_names)}")
        if discarded_cards:
            discard_names = [d.get("card_name", "?") for d in discarded_cards]
            summary_parts.append(f"Discard: {', '.join(discard_names)}")

        if summary_parts:
            first_line = f"[bold magenta]CLEANUP:[/bold magenta] {'; '.join(summary_parts)}"
            lines.append(first_line)

            # Detailed buff removal lines
            for effect in effects_removed:
                creature = effect.get("creature", "Unknown")
                from_p = effect.get("from_power", 0)
                from_t = effect.get("from_toughness", 0)
                to_p = effect.get("to_power", 0)
                to_t = effect.get("to_toughness", 0)
                tokens = effect.get("tokens", [])
                token_str = f" (has {', '.join(tokens)})" if tokens else ""
                lines.append(
                    f"  [dim]→ {format_card_display(creature)}: "
                    f"{from_p}/{from_t} → {to_p}/{to_t}{token_str}[/dim]"
                )

            # Detailed discard lines
            for discard in discarded_cards:
                card_name = discard.get("card_name", "Unknown")
                player_idx = discard.get("player_idx", 0)
                player_label = "Player" if player_idx == 0 else "Opponent"
                lines.append(
                    f"  [dim]→ {player_label} discards {format_card_display(card_name)} "
                    f"(hand size: {discard.get('hand_size_before', '?')} → "
                    f"{discard.get('hand_size_after', '?')})[/dim]"
                )

            return "\n".join(lines)

        return "[bold magenta]CLEANUP:[/bold magenta] [dim]No effects to remove[/dim]"

    elif action_type == "COUNTER":
        countered = details.get("countered", "Unknown spell")
        destination = details.get("destination", "graveyard")
        dest_str = "exiled" if destination == "exile" else "sent to graveyard"
        return (
            f"[bold red]COUNTERED:[/bold red] {format_card_display(card_name)} counters "
            f"{format_card_display(countered)} [dim]({dest_str})[/dim]"
        )

    elif action_type == "EXILE":
        source = details.get("source", "")
        source_str = f" [dim](by {format_card_display(source)})[/dim]" if source else ""
        return f"[bold red]EXILE:[/bold red] {format_card_display(card_name)}{source_str}"

    elif action_type == "DEATH_TRIGGER":
        trigger_desc = details.get("trigger", "")
        damage = details.get("damage", 0)
        target_idx = details.get("target_player_idx")
        if damage and target_idx is not None:
            target = "Player" if target_idx == 0 else "Opponent"
            return (
                f"[bold magenta]💀 DEATH TRIGGER:[/bold magenta] {format_card_display(card_name)} "
                f"deals {damage} damage to [bold]{target}[/bold]"
            )
        return f"[bold magenta]💀 DEATH TRIGGER:[/bold magenta] {trigger_desc}"

    else:
        return f"[bold yellow]{action_type}:[/bold yellow] {card_name}"


def infer_opponent_actions(prev_info: dict, curr_info: dict) -> list[str]:
    """Get opponent actions from the action log."""
    new_actions = get_new_actions_from_log(prev_info, curr_info)

    # Filter to only actions that happened during opponent's turn
    opponent_actions = []
    for action in new_actions:
        if action.get("active_player", action.get("player")) == 1:  # Opponent's turn
            formatted = format_logged_action(action, active_player_idx=1)
            opponent_actions.append(formatted)

    return opponent_actions


ALL_PHASES = [
    "Untap",
    "Upkeep",
    "Draw",
    "Main 1",
    "Combat",
    "Attackers",
    "Blockers",
    "Damage",
    "Main 2",
    "End",
    "Cleanup",
]


def display_complete_turn(
    turn: int,
    player: str,
    actions_by_phase: dict[str, list[str]],
    delays: dict[str, float],
    is_first_turn: bool = False,
) -> None:
    """Display a complete turn with all phases shown."""
    # Untap
    untap_actions = actions_by_phase.get("Untap", [])
    if not untap_actions:
        untap_actions = ["[dim]Untap all permanents[/dim]"]
    print_phase_box("Untap", player, untap_actions)
    time.sleep(delays["action"])

    # Upkeep
    upkeep_actions = actions_by_phase.get("Upkeep", [])
    if not upkeep_actions:
        upkeep_actions = ["[dim]Pass[/dim]"]
    print_phase_box("Upkeep", player, upkeep_actions)
    time.sleep(delays["action"])

    # Draw
    draw_actions = actions_by_phase.get("Draw", [])
    if not draw_actions:
        # On Turn 1, the first player (on the play) doesn't draw
        # The first player to act is the one whose turn it is first
        if is_first_turn:
            draw_actions = ["[dim]No draw on Turn 1 for first player[/dim]"]
        else:
            draw_actions = ["[dim]Draw a card[/dim]"]
    print_phase_box("Draw", player, draw_actions)
    time.sleep(delays["action"])

    # Main 1
    main1_actions = actions_by_phase.get("Main 1", [])
    if not main1_actions:
        main1_actions = ["[dim]Pass[/dim]"]
    print_phase_box("Main 1", player, main1_actions)
    time.sleep(delays["action"])

    # Combat - Declare Attackers
    # Check both short and full phase names for compatibility
    combat_actions = actions_by_phase.get("Combat", [])
    atk_actions = actions_by_phase.get("Combat - Declare Attackers", []) + actions_by_phase.get(
        "Attackers", []
    )
    if combat_actions:
        atk_actions = combat_actions + atk_actions
    if not atk_actions:
        atk_actions = ["[dim]No attacks[/dim]"]
    print_combat_phase_box("Declare Attackers", player, atk_actions)
    time.sleep(delays["action"])

    # Combat - Declare Blockers
    blk_actions = actions_by_phase.get("Combat - Declare Blockers", []) + actions_by_phase.get(
        "Blockers", []
    )
    if not blk_actions:
        blk_actions = ["[dim]No blockers declared[/dim]"]
    print_combat_phase_box("Declare Blockers", player, blk_actions)
    time.sleep(delays["action"])

    # Combat - Damage
    dmg_actions = actions_by_phase.get("Combat - Damage", []) + actions_by_phase.get("Damage", [])
    if not dmg_actions:
        dmg_actions = ["[dim]No combat damage[/dim]"]
    print_combat_phase_box("Damage", player, dmg_actions)
    time.sleep(delays["action"])

    # Main 2
    main2_actions = actions_by_phase.get("Main 2", [])
    if not main2_actions:
        main2_actions = ["[dim]Pass[/dim]"]
    print_phase_box("Main 2", player, main2_actions)
    time.sleep(delays["action"])

    # End Step
    end_actions = actions_by_phase.get("End", [])
    if not end_actions:
        end_actions = ["[dim]End of turn[/dim]"]
    print_phase_box("End", player, end_actions)
    time.sleep(delays["action"])

    # Cleanup Step (always show)
    cleanup_actions = actions_by_phase.get("Cleanup", [])
    if not cleanup_actions:
        cleanup_actions = ["[dim]No temporary effects to remove[/dim]"]
    print_phase_box("Cleanup", player, cleanup_actions)
    time.sleep(delays["action"])


def display_opponent_turn_from_log(
    turn: int,
    action_log: list[dict],
    delays: dict[str, float],
    env: MTGEnv,
) -> None:
    """Display opponent's complete turn using actions from the action log."""
    # Filter actions for this turn during opponent's turn (active_player=1)
    opp_actions = [
        a
        for a in action_log
        if a.get("turn") == turn and a.get("active_player", a.get("player")) == 1
    ]

    if not opp_actions:
        console.print(f"[dim]Opponent Turn {turn}: No actions recorded[/dim]")
        return

    # Group by phase
    actions_by_phase: dict[str, list[str]] = {}
    for action in opp_actions:
        phase = action.get("phase", "UNKNOWN")
        # Opponent's turn - active_player is 1 (opponent)
        formatted = format_logged_action(action, active_player_idx=1)
        action_type = action.get("action_type", "")

        # Map phase enum names to display names
        phase_display = {
            "UNTAP": "Untap",
            "UPKEEP": "Upkeep",
            "DRAW": "Draw",
            "MAIN_PRECOMBAT": "Main 1",
            "COMBAT_BEGIN": "Combat",
            "COMBAT_ATTACKERS": "Attackers",
            "COMBAT_BLOCKERS": "Blockers",
            "COMBAT_DAMAGE": "Damage",
            "MAIN_POSTCOMBAT": "Main 2",
            "END_STEP": "End",
        }.get(phase, phase)

        # Override phase for action types that belong to specific phases
        if action_type == "BLOCK":
            phase_display = "Blockers"
        elif action_type == "ATTACK":
            phase_display = "Attackers"

        if phase_display not in actions_by_phase:
            actions_by_phase[phase_display] = []
        actions_by_phase[phase_display].append(formatted)

    # Print header
    print_opponent_turn_header(turn)

    # Display turn start state
    print_turn_start_state(env, "Opponent")
    time.sleep(delays["action"])

    # Display all phases
    display_complete_turn(turn, "Opponent", actions_by_phase, delays, is_first_turn=(turn == 1))

    # Display turn end state
    print_turn_end_state(env, "Opponent")
    time.sleep(delays["action"])


def print_combat_phase_box(
    sub_phase: str,  # "Declare Attackers", "Declare Blockers", "Damage"
    player: str,
    actions: list[str] | None = None,
) -> None:
    """Print a combat sub-phase box.

    For Declare Blockers, colors are inverted (blocker's perspective).
    Note: Actions should already be formatted with colors.
    """
    is_player = player != "Player" if sub_phase == "Declare Blockers" else player == "Player"

    border_style = "green" if is_player else "red"
    title_style = "bold green" if is_player else "bold red"

    content_parts = []
    if actions:
        for action in actions:
            # Actions are already formatted with colors
            content_parts.append(action)
    else:
        content_parts.append(
            "[dim]No blockers declared[/dim]"
            if sub_phase == "Declare Blockers"
            else "[dim]No combat damage[/dim]"
            if sub_phase == "Damage"
            else "[dim]No attackers[/dim]"
        )

    content = "\n".join(content_parts)

    console.print(
        Panel(
            content,
            title=f"[{title_style}]Combat - {sub_phase}[/{title_style}]",
            title_align="center",
            border_style=border_style,
            width=console.width,
        )
    )


# =============================================================================
# Turn State Display (using existing cli_display functions)
# =============================================================================


def _format_graveyard_with_counts(graveyard: list) -> str:
    """Format graveyard with card names and type counts."""
    if not graveyard:
        return "[dim]Empty[/dim]"

    gy_parts = [format_card_display(c.name) for c in graveyard]
    type_counts: dict[str, int] = {}
    for c in graveyard:
        ct = c.card_type.value.lower() if hasattr(c, "card_type") else "unknown"
        type_counts[ct] = type_counts.get(ct, 0) + 1
    type_icons = {
        "creature": "⚔️",
        "instant": "✨",
        "sorcery": "🌟",
        "enchantment": "🔮",
        "land": "🌍",
    }
    counts_str = " ".join(f"{type_icons.get(t, '🃏')}{c}" for t, c in type_counts.items())
    return f"{', '.join(gy_parts)} ({counts_str})"


def print_turn_start_state(env: MTGEnv, active: str) -> None:
    """Print turn start state showing hands, lands, creatures, and graveyards side-by-side.

    Note: Due to action_log processing, this shows current state not true start state.
    """
    player = env.state.players[0]
    opponent = env.state.players[1]

    # Calculate half width for side-by-side display
    half_width = (console.width - 6) // 2  # Account for padding

    # Build player table
    player_table = Table(
        show_header=True,
        header_style="bold green",
        box=box.ROUNDED,
        title="[bold green]Player - Turn Start[/bold green]",
        width=half_width,
    )
    player_table.add_column("Category", style="cyan", width=18)
    player_table.add_column("Contents", style="white", overflow="fold")

    # Format player hand with count
    if player.hand:
        hand_str = ", ".join(format_card_display(c.name) for c in player.hand)
        player_table.add_row(f"🃏 Hand ({len(player.hand)})", hand_str)
    else:
        player_table.add_row("🃏 Hand (0)", "[dim]Empty[/dim]")

    # Lands on board
    p_lands = get_lands_by_type(player.battlefield)
    if p_lands:
        lands_str = ", ".join(f"{format_card_display(n)} x{c}" for n, c in p_lands.items())
        player_table.add_row(f"🌍 Lands ({sum(p_lands.values())})", lands_str)
    else:
        player_table.add_row("🌍 Lands (0)", "[dim]None[/dim]")

    # Creatures on board (with attached tokens)
    p_creatures = [c for c in player.battlefield if c.card_type == CardType.CREATURE]
    if p_creatures:
        creatures_str = ", ".join(format_creature_from_card(c) for c in p_creatures)
        player_table.add_row(f"⚔️ Creatures ({len(p_creatures)})", creatures_str)
    else:
        player_table.add_row("⚔️ Creatures (0)", "[dim]None[/dim]")

    # Graveyard with count
    gy_str = _format_graveyard_with_counts(player.graveyard)
    player_table.add_row(f"💀 Graveyard ({len(player.graveyard)})", gy_str)

    # Build opponent table
    opp_table = Table(
        show_header=True,
        header_style="bold red",
        box=box.ROUNDED,
        title="[bold red]Opponent - Turn Start[/bold red]",
        width=half_width,
    )
    opp_table.add_column("Category", style="cyan", width=22)
    opp_table.add_column("Contents", style="white", overflow="fold")

    # Show opponent's actual hand with (hidden) indicator and count
    if opponent.hand:
        hand_parts = [
            f"[dim italic]{format_card_display(c.name)}[/dim italic]" for c in opponent.hand
        ]
        opp_table.add_row(f"🃏 Hand ({len(opponent.hand)}) hidden", ", ".join(hand_parts))
    else:
        opp_table.add_row("🃏 Hand (0)", "[dim]Empty[/dim]")

    # Lands on board
    o_lands = get_lands_by_type(opponent.battlefield)
    if o_lands:
        lands_str = ", ".join(f"{format_card_display(n)} x{c}" for n, c in o_lands.items())
        opp_table.add_row(f"🌍 Lands ({sum(o_lands.values())})", lands_str)
    else:
        opp_table.add_row("🌍 Lands (0)", "[dim]None[/dim]")

    # Creatures on board (with attached tokens)
    o_creatures = [c for c in opponent.battlefield if c.card_type == CardType.CREATURE]
    if o_creatures:
        # Battlefield is public info - use normal white text
        creatures_str = ", ".join(format_creature_from_card(c) for c in o_creatures)
        opp_table.add_row(f"⚔️ Creatures ({len(o_creatures)})", creatures_str)
    else:
        opp_table.add_row("⚔️ Creatures (0)", "[dim]None[/dim]")

    # Graveyard with count
    gy_str = _format_graveyard_with_counts(opponent.graveyard)
    opp_table.add_row(f"💀 Graveyard ({len(opponent.graveyard)})", gy_str)

    console.print(Columns([player_table, opp_table], padding=(0, 2), equal=True, expand=True))


def print_turn_start_state_from_data(
    player_hand: list,
    opponent_hand: list,
    player_graveyard: list,
    opponent_graveyard: list,
    player_lands: dict | None = None,
    opponent_lands: dict | None = None,
    player_creatures: list | None = None,
    opponent_creatures: list | None = None,
) -> None:
    """Print turn start state from saved data (for accurate Turn 1 display).

    Handles both Card objects and string card names.
    """

    # Helper to format creature from Card object, dict, or string
    def _effective_stats(creature: dict) -> tuple[int, int]:
        power = (
            creature.get("base_power", 0)
            + creature.get("power_bonus", 0)
            + creature.get("temp_power_bonus", 0)
        )
        toughness = (
            creature.get("base_toughness", 0)
            + creature.get("toughness_bonus", 0)
            + creature.get("temp_toughness_bonus", 0)
        )
        return power, toughness

    def format_creature(c) -> str:
        if hasattr(c, "name"):
            # Card object - use format_creature_from_card if available
            return format_creature_from_card(c)
        if isinstance(c, dict):
            # Dict with name and attached_tokens
            power, toughness = _effective_stats(c)
            return format_card_display(
                c["name"],
                power=power,
                toughness=toughness,
                attached_tokens=c.get("attached_tokens", []),
            )
        return format_card_display(str(c))

    # Helper to get card name from either Card object, dict, or string
    def get_name(c) -> str:
        if hasattr(c, "name"):
            return c.name
        if isinstance(c, dict):
            return c["name"]
        return str(c)

    # Calculate half width for side-by-side display
    half_width = (console.width - 6) // 2  # Account for padding

    # Build player table
    player_table = Table(
        show_header=True,
        header_style="bold green",
        box=box.ROUNDED,
        title="[bold green]Player - Turn Start[/bold green]",
        width=half_width,
    )
    player_table.add_column("Category", style="cyan", width=18)
    player_table.add_column("Contents", style="white", overflow="fold")

    # Format player hand with count
    if player_hand:
        hand_str = ", ".join(format_card_display(get_name(c)) for c in player_hand)
        player_table.add_row(f"🃏 Hand ({len(player_hand)})", hand_str)
    else:
        player_table.add_row("🃏 Hand (0)", "[dim]Empty[/dim]")

    # Lands on board
    if player_lands:
        lands_str = ", ".join(f"{format_card_display(n)} x{c}" for n, c in player_lands.items())
        player_table.add_row(f"🌍 Lands ({sum(player_lands.values())})", lands_str)
    else:
        player_table.add_row("🌍 Lands (0)", "[dim]None[/dim]")

    # Creatures on board (with attached tokens if any)
    if player_creatures:
        creatures_str = ", ".join(format_creature(c) for c in player_creatures)
        player_table.add_row(f"⚔️ Creatures ({len(player_creatures)})", creatures_str)
    else:
        player_table.add_row("⚔️ Creatures (0)", "[dim]None[/dim]")

    # Graveyard with count - handle both Card objects and strings
    if player_graveyard and hasattr(player_graveyard[0], "name"):
        gy_str = _format_graveyard_with_counts(player_graveyard)
    else:
        gy_str = _format_graveyard_with_counts_from_names(player_graveyard)
    player_table.add_row(f"💀 Graveyard ({len(player_graveyard)})", gy_str)

    # Build opponent table
    opp_table = Table(
        show_header=True,
        header_style="bold red",
        box=box.ROUNDED,
        title="[bold red]Opponent - Turn Start[/bold red]",
        width=half_width,
    )
    opp_table.add_column("Category", style="cyan", width=22)
    opp_table.add_column("Contents", style="white", overflow="fold")

    # Show opponent's actual hand with (hidden) indicator and count
    if opponent_hand:
        hand_parts = [
            f"[dim italic]{format_card_display(get_name(c))}[/dim italic]" for c in opponent_hand
        ]
        opp_table.add_row(f"🃏 Hand ({len(opponent_hand)}) hidden", ", ".join(hand_parts))
    else:
        opp_table.add_row("🃏 Hand (0)", "[dim]Empty[/dim]")

    # Lands on board
    if opponent_lands:
        lands_str = ", ".join(f"{format_card_display(n)} x{c}" for n, c in opponent_lands.items())
        opp_table.add_row(f"🌍 Lands ({sum(opponent_lands.values())})", lands_str)
    else:
        opp_table.add_row("🌍 Lands (0)", "[dim]None[/dim]")

    # Creatures on board - visible to both players (with attached tokens if any)
    if opponent_creatures:
        creatures_str = ", ".join(format_creature(c) for c in opponent_creatures)
        opp_table.add_row(f"⚔️ Creatures ({len(opponent_creatures)})", creatures_str)
    else:
        opp_table.add_row("⚔️ Creatures (0)", "[dim]None[/dim]")

    # Graveyard with count - handle both Card objects and strings
    if opponent_graveyard and hasattr(opponent_graveyard[0], "name"):
        gy_str = _format_graveyard_with_counts(opponent_graveyard)
    else:
        gy_str = _format_graveyard_with_counts_from_names(opponent_graveyard)
    opp_table.add_row(f"💀 Graveyard ({len(opponent_graveyard)})", gy_str)

    console.print(Columns([player_table, opp_table], padding=(0, 2), equal=True, expand=True))


def print_turn_end_state_from_snapshot(
    player_hand: list[str],
    opponent_hand: list[str],
    player_graveyard: list[str],
    opponent_graveyard: list[str],
    player_lands: dict | None = None,
    opponent_lands: dict | None = None,
    player_creatures: list | None = None,
    opponent_creatures: list | None = None,
    env: MTGEnv | None = None,
) -> None:
    """Print turn end state showing both player and opponent from snapshot data."""
    # Calculate half width for side-by-side display
    half_width = (console.width - 6) // 2

    # Build player table
    player_table = Table(
        show_header=True,
        header_style="bold green",
        box=box.ROUNDED,
        title="[bold green]Player - Turn End[/bold green]",
        width=half_width,
    )
    player_table.add_column("Category", style="cyan", width=20)
    player_table.add_column("Contents", style="white", overflow="fold")

    if player_hand:
        hand_str = ", ".join(format_card_display(c) for c in player_hand)
        player_table.add_row(f"🃏 Hand ({len(player_hand)})", hand_str)
    else:
        player_table.add_row("🃏 Hand (0)", "[dim]Empty[/dim]")

    # Lands on board
    if player_lands:
        lands_str = ", ".join(f"{format_card_display(n)} x{c}" for n, c in player_lands.items())
        player_table.add_row(f"🌍 Lands ({sum(player_lands.values())})", lands_str)
    else:
        player_table.add_row("🌍 Lands (0)", "[dim]None[/dim]")

    # Creatures on board - use live env for stats if available (with attached tokens)
    if player_creatures:
        if env:
            # Get actual creature stats and attached tokens from environment
            creature_strs = []
            for c in env.state.players[0].battlefield:
                if c.card_type == CardType.CREATURE:
                    creature_strs.append(format_creature_from_card(c))
            creatures_str = ", ".join(creature_strs) if creature_strs else "[dim]None[/dim]"
        else:
            # Format from tracked data (dicts or strings)
            def _effective_stats(creature: dict) -> tuple[int, int]:
                power = (
                    creature.get("base_power", 0)
                    + creature.get("power_bonus", 0)
                    + creature.get("temp_power_bonus", 0)
                )
                toughness = (
                    creature.get("base_toughness", 0)
                    + creature.get("toughness_bonus", 0)
                    + creature.get("temp_toughness_bonus", 0)
                )
                return power, toughness

            creature_strs = []
            for c in player_creatures:
                if isinstance(c, dict):
                    power, toughness = _effective_stats(c)
                    creature_strs.append(
                        format_card_display(
                            c["name"],
                            power=power,
                            toughness=toughness,
                            attached_tokens=c.get("attached_tokens", []),
                        )
                    )
                else:
                    creature_strs.append(format_card_display(c))
            creatures_str = ", ".join(creature_strs)
        player_table.add_row(f"⚔️ Creatures ({len(player_creatures)})", creatures_str)
    else:
        player_table.add_row("⚔️ Creatures (0)", "[dim]None[/dim]")

    gy_str = _format_graveyard_with_counts_from_names(player_graveyard)
    player_table.add_row(f"💀 Graveyard ({len(player_graveyard)})", gy_str)

    # Build opponent table
    opp_table = Table(
        show_header=True,
        header_style="bold red",
        box=box.ROUNDED,
        title="[bold red]Opponent - Turn End[/bold red]",
        width=half_width,
    )
    opp_table.add_column("Category", style="cyan", width=22)
    opp_table.add_column("Contents", style="white", overflow="fold")

    if opponent_hand:
        hand_parts = [f"[dim italic]{format_card_display(c)}[/dim italic]" for c in opponent_hand]
        opp_table.add_row(f"🃏 Hand ({len(opponent_hand)}) hidden", ", ".join(hand_parts))
    else:
        opp_table.add_row("🃏 Hand (0)", "[dim]Empty[/dim]")

    # Lands on board
    if opponent_lands:
        lands_str = ", ".join(f"{format_card_display(n)} x{c}" for n, c in opponent_lands.items())
        opp_table.add_row(f"🌍 Lands ({sum(opponent_lands.values())})", lands_str)
    else:
        opp_table.add_row("🌍 Lands (0)", "[dim]None[/dim]")

    # Creatures on board - visible to both players (with attached tokens)
    if opponent_creatures:
        if env:
            # Get actual creature stats and attached tokens from environment
            creature_strs = []
            for c in env.state.players[1].battlefield:
                if c.card_type == CardType.CREATURE:
                    creature_strs.append(format_creature_from_card(c))
            creatures_str = ", ".join(creature_strs) if creature_strs else "[dim]None[/dim]"
        else:
            # Format from tracked data (dicts or strings)
            def _effective_stats(creature: dict) -> tuple[int, int]:
                power = (
                    creature.get("base_power", 0)
                    + creature.get("power_bonus", 0)
                    + creature.get("temp_power_bonus", 0)
                )
                toughness = (
                    creature.get("base_toughness", 0)
                    + creature.get("toughness_bonus", 0)
                    + creature.get("temp_toughness_bonus", 0)
                )
                return power, toughness

            creature_strs = []
            for c in opponent_creatures:
                if isinstance(c, dict):
                    power, toughness = _effective_stats(c)
                    creature_strs.append(
                        format_card_display(
                            c["name"],
                            power=power,
                            toughness=toughness,
                            attached_tokens=c.get("attached_tokens", []),
                        )
                    )
                else:
                    creature_strs.append(format_card_display(c))
            creatures_str = ", ".join(creature_strs)
        opp_table.add_row(f"⚔️ Creatures ({len(opponent_creatures)})", creatures_str)
    else:
        opp_table.add_row("⚔️ Creatures (0)", "[dim]None[/dim]")

    gy_str = _format_graveyard_with_counts_from_names(opponent_graveyard)
    opp_table.add_row(f"💀 Graveyard ({len(opponent_graveyard)})", gy_str)

    console.print(Columns([player_table, opp_table], padding=(0, 2), equal=True, expand=True))


def _format_graveyard_with_counts_from_names(graveyard: list[str]) -> str:
    """Format graveyard from list of card names."""
    if not graveyard:
        return "[dim]Empty[/dim]"

    # Count by type using CardRegistry
    registry = CardRegistry.get_instance()
    type_counts: dict[str, int] = {}
    card_names: list[str] = []

    for name in graveyard:
        card_names.append(format_card_display(name))
        try:
            card = registry.get(name)
            type_name = card.card_type.name if card.card_type else "UNKNOWN"
        except KeyError:
            type_name = "UNKNOWN"
        type_counts[type_name] = type_counts.get(type_name, 0) + 1

    # Type icons
    icons = {"CREATURE": "💀", "INSTANT": "✨", "SORCERY": "🌟", "ENCHANTMENT": "🔮", "LAND": "🌍"}
    type_str = " ".join(f"{icons.get(t, '📄')}{c}" for t, c in type_counts.items())

    return f"{', '.join(card_names)} ({type_str})"


def print_turn_end_state(env: MTGEnv, active: str) -> None:
    """Print turn end state showing both player and opponent side by side."""
    player = env.state.players[0]
    opponent = env.state.players[1]

    # Calculate half width for side-by-side display
    half_width = (console.width - 6) // 2

    # Build player table
    player_table = Table(
        show_header=True,
        header_style="bold green",
        box=box.ROUNDED,
        title="[bold green]Player - Turn End[/bold green]",
        width=half_width,
    )
    player_table.add_column("Category", style="cyan", width=20)
    player_table.add_column("Contents", style="white", overflow="fold")

    if player.hand:
        hand_str = ", ".join(format_card_display(c.name) for c in player.hand)
        player_table.add_row(f"🃏 Hand ({len(player.hand)})", hand_str)
    else:
        player_table.add_row("🃏 Hand (0)", "[dim]Empty[/dim]")

    # Lands on board
    p_lands = get_lands_by_type(player.battlefield)
    if p_lands:
        lands_str = ", ".join(f"{format_card_display(n)} x{c}" for n, c in p_lands.items())
        player_table.add_row(f"🌍 Lands ({sum(p_lands.values())})", lands_str)
    else:
        player_table.add_row("🌍 Lands (0)", "[dim]None[/dim]")

    # Creatures on board (with attached tokens)
    p_creatures = [c for c in player.battlefield if c.card_type == CardType.CREATURE]
    if p_creatures:
        creatures_str = ", ".join(format_creature_from_card(c) for c in p_creatures)
        player_table.add_row(f"⚔️ Creatures ({len(p_creatures)})", creatures_str)
    else:
        player_table.add_row("⚔️ Creatures (0)", "[dim]None[/dim]")

    gy_str = _format_graveyard_with_counts(player.graveyard)
    player_table.add_row(f"💀 Graveyard ({len(player.graveyard)})", gy_str)

    # Build opponent table
    opp_table = Table(
        show_header=True,
        header_style="bold red",
        box=box.ROUNDED,
        title="[bold red]Opponent - Turn End[/bold red]",
        width=half_width,
    )
    opp_table.add_column("Category", style="cyan", width=22)
    opp_table.add_column("Contents", style="white", overflow="fold")

    if opponent.hand:
        hand_parts = [
            f"[dim italic]{format_card_display(c.name)}[/dim italic]" for c in opponent.hand
        ]
        opp_table.add_row(f"🃏 Hand ({len(opponent.hand)}) hidden", ", ".join(hand_parts))
    else:
        opp_table.add_row("🃏 Hand (0)", "[dim]Empty[/dim]")

    # Lands on board
    o_lands = get_lands_by_type(opponent.battlefield)
    if o_lands:
        lands_str = ", ".join(f"{format_card_display(n)} x{c}" for n, c in o_lands.items())
        opp_table.add_row(f"🌍 Lands ({sum(o_lands.values())})", lands_str)
    else:
        opp_table.add_row("🌍 Lands (0)", "[dim]None[/dim]")

    # Creatures on board (with attached tokens)
    o_creatures = [c for c in opponent.battlefield if c.card_type == CardType.CREATURE]
    if o_creatures:
        # Battlefield is public info - use normal white text
        creatures_str = ", ".join(format_creature_from_card(c) for c in o_creatures)
        opp_table.add_row(f"⚔️ Creatures ({len(o_creatures)})", creatures_str)
    else:
        opp_table.add_row("⚔️ Creatures (0)", "[dim]None[/dim]")

    gy_str = _format_graveyard_with_counts(opponent.graveyard)
    opp_table.add_row(f"💀 Graveyard ({len(opponent.graveyard)})", gy_str)

    console.print(Columns([player_table, opp_table], padding=(0, 2), equal=True, expand=True))


# =============================================================================
# Mulligan Display
# =============================================================================


def print_mulligan_phase_detailed(
    player_hand: list,
    opponent_hand: list,
    player_keeps: bool,
    mulligan_new_hand: list | None = None,
    returned_cards: list | None = None,
    final_kept_hand: list | None = None,
    opponent_original_hand: list | None = None,
    opponent_keeps: bool = True,
    opponent_mulligan_new_hand: list | None = None,
    opponent_returned_cards: list | None = None,
    opponent_final_kept_hand: list | None = None,
) -> None:
    """Print detailed mulligan phase display with London Mulligan rules.

    London Mulligan: Draw 7 cards, then put cards on bottom equal to mulligans taken.
    """
    console.rule("[bold cyan]Mulligan Phase[/bold cyan]")

    # Player's opening hand
    console.print(f"\n[bold green]Player's Opening Hand ({len(player_hand)} cards):[/bold green]")
    for card in player_hand:
        console.print(f"  • {format_card_display(card.name)}")

    # Decision - show full process
    if player_keeps:
        console.print("\n[bold green]  ✓ Player keeps hand[/bold green]")
    else:
        console.print("\n[bold yellow]  ↻ Player mulligans[/bold yellow]")

        # Show new 7-card hand drawn
        if mulligan_new_hand:
            console.print(
                f"\n[bold yellow]New Hand Drawn ({len(mulligan_new_hand)} cards):[/bold yellow]"
            )
            for card in mulligan_new_hand:
                if hasattr(card, "name"):
                    console.print(f"  • {format_card_display(card.name)}")
                else:
                    console.print(f"  • {format_card_display(card)}")

        # Show cards returned to bottom of library
        if returned_cards:
            num_returned = len(returned_cards)
            label = "Card" if num_returned == 1 else "Cards"
            console.print(
                f"\n[bold yellow]{label} Returned to Bottom ({num_returned}):[/bold yellow]"
            )
            for card in returned_cards:
                if hasattr(card, "name"):
                    console.print(f"  • {format_card_display(card.name)}")
                else:
                    console.print(f"  • {format_card_display(card)}")

        # Show final kept hand
        if final_kept_hand:
            console.print(
                f"\n[bold green]Final Kept Hand ({len(final_kept_hand)} cards):[/bold green]"
            )
            for card in final_kept_hand:
                if hasattr(card, "name"):
                    console.print(f"  • {format_card_display(card.name)}")
                else:
                    console.print(f"  • {format_card_display(card)}")
            console.print("\n[bold green]  ✓ Player keeps hand[/bold green]")

    # Opponent's hand - use original hand for display
    opp_display_hand = opponent_original_hand if opponent_original_hand else opponent_hand
    console.print(
        f"\n[bold red]Opponent's Opening Hand ({len(opp_display_hand)} cards) (hidden):[/bold red]"
    )
    for card in opp_display_hand:
        if hasattr(card, "name"):
            console.print(f"  [dim italic]• {format_card_display(card.name)} (hidden)[/dim italic]")
        else:
            console.print(f"  [dim italic]• {format_card_display(card)} (hidden)[/dim italic]")

    if opponent_keeps:
        console.print("\n[bold red]  ✓ Opponent keeps hand[/bold red]")
    else:
        console.print("\n[bold yellow]  ↻ Opponent mulligans[/bold yellow]")

        # Show new 7-card hand drawn
        if opponent_mulligan_new_hand:
            console.print(
                f"\n[bold yellow]New Hand Drawn ({len(opponent_mulligan_new_hand)} cards) "
                "(hidden):[/bold yellow]"
            )
            for card in opponent_mulligan_new_hand:
                if hasattr(card, "name"):
                    console.print(
                        f"  [dim italic]• {format_card_display(card.name)} (hidden)[/dim italic]"
                    )
                else:
                    console.print(
                        f"  [dim italic]• {format_card_display(card)} (hidden)[/dim italic]"
                    )

        # Show cards returned to bottom
        if opponent_returned_cards:
            num_returned = len(opponent_returned_cards)
            label = "Card" if num_returned == 1 else "Cards"
            console.print(
                f"\n[bold yellow]{label} Returned to Bottom ({num_returned}) "
                f"(hidden):[/bold yellow]"
            )
            for card in opponent_returned_cards:
                if hasattr(card, "name"):
                    console.print(
                        f"  [dim italic]• {format_card_display(card.name)} (hidden)[/dim italic]"
                    )
                else:
                    console.print(
                        f"  [dim italic]• {format_card_display(card)} (hidden)[/dim italic]"
                    )

        # Show final kept hand
        if opponent_final_kept_hand:
            console.print(
                f"\n[bold red]Final Kept Hand ({len(opponent_final_kept_hand)} cards) "
                "(hidden):[/bold red]"
            )
            for card in opponent_final_kept_hand:
                if hasattr(card, "name"):
                    console.print(
                        f"  [dim italic]• {format_card_display(card.name)} (hidden)[/dim italic]"
                    )
                else:
                    console.print(
                        f"  [dim italic]• {format_card_display(card)} (hidden)[/dim italic]"
                    )
            console.print("\n[bold red]  ✓ Opponent keeps hand[/bold red]")

    console.print()


# =============================================================================
# Final Game State
# =============================================================================


def print_final_game_state(env: MTGEnv) -> None:
    """Print detailed final game state."""
    player = env.state.players[0]
    opponent = env.state.players[1]

    console.print()

    table = Table(
        title="[bold cyan]Final Game State[/bold cyan]",
        box=box.DOUBLE,
        show_header=True,
        header_style="bold",
        width=console.width,
    )
    table.add_column("", style="cyan", width=20)
    table.add_column("Player", style="green", justify="left")
    table.add_column("Opponent", style="red", justify="left")

    # Life
    table.add_row("❤️ Life", str(player.life), str(opponent.life))

    # Hand
    if player.hand:
        p_hand = ", ".join(format_card_display(c.name) for c in player.hand)
    else:
        p_hand = "[dim]Empty[/dim]"
    table.add_row("🃏 Hand", p_hand, f"{len(opponent.hand)} cards (hidden)")

    # Lands
    p_lands = get_lands_by_type(player.battlefield)
    o_lands = get_lands_by_type(opponent.battlefield)
    p_lands_str = (
        ", ".join(f"{format_card_display(n)} x{c}" for n, c in p_lands.items()) or "[dim]None[/dim]"
    )
    o_lands_str = (
        ", ".join(f"{format_card_display(n)} x{c}" for n, c in o_lands.items()) or "[dim]None[/dim]"
    )
    table.add_row("🌍 Lands", p_lands_str, o_lands_str)

    # Creatures
    p_creatures = [c for c in player.battlefield if c.card_type == CardType.CREATURE]
    o_creatures = [c for c in opponent.battlefield if c.card_type == CardType.CREATURE]
    p_c_str = (
        ", ".join(format_card_display(c.name, c.power, c.toughness) for c in p_creatures)
        or "[dim]None[/dim]"
    )
    o_c_str = (
        ", ".join(format_card_display(c.name, c.power, c.toughness) for c in o_creatures)
        or "[dim]None[/dim]"
    )
    table.add_row("⚔️ Creatures", p_c_str, o_c_str)

    # Enchantments
    p_ench = [c for c in player.battlefield if c.card_type == CardType.ENCHANTMENT]
    o_ench = [c for c in opponent.battlefield if c.card_type == CardType.ENCHANTMENT]
    p_e_str = ", ".join(format_card_display(c.name) for c in p_ench) or "[dim]None[/dim]"
    o_e_str = ", ".join(format_card_display(c.name) for c in o_ench) or "[dim]None[/dim]"
    table.add_row("🔮 Enchantments", p_e_str, o_e_str)

    # Graveyard
    if player.graveyard:
        gy_parts = [format_card_display(c.name) for c in player.graveyard]
        type_counts = {}
        for c in player.graveyard:
            ct = c.card_type.value.lower() if hasattr(c, "card_type") else "unknown"
            type_counts[ct] = type_counts.get(ct, 0) + 1
        type_icons = {
            "creature": "⚔️",
            "instant": "✨",
            "sorcery": "🌟",
            "enchantment": "🔮",
            "land": "🌍",
        }
        counts_str = " ".join(f"{type_icons.get(t, '🃏')}{c}" for t, c in type_counts.items())
        p_gy = f"{', '.join(gy_parts)} ({counts_str})"
    else:
        p_gy = "[dim]Empty[/dim]"

    if opponent.graveyard:
        gy_parts = [format_card_display(c.name) for c in opponent.graveyard]
        type_counts = {}
        for c in opponent.graveyard:
            ct = c.card_type.value.lower() if hasattr(c, "card_type") else "unknown"
            type_counts[ct] = type_counts.get(ct, 0) + 1
        type_icons = {
            "creature": "⚔️",
            "instant": "✨",
            "sorcery": "🌟",
            "enchantment": "🔮",
            "land": "🌍",
        }
        counts_str = " ".join(f"{type_icons.get(t, '🃏')}{c}" for t, c in type_counts.items())
        o_gy = f"{', '.join(gy_parts)} ({counts_str})"
    else:
        o_gy = "[dim]Empty[/dim]"

    table.add_row("💀 Graveyard", p_gy, o_gy)

    console.print(table)
    console.print(f"\n[dim]{CARD_TYPE_LEGEND}[/dim]")


# =============================================================================
# Action Formatting
# =============================================================================


def format_action_for_display(action_name: str, player: str, env: MTGEnv) -> str | None:
    """Format an action for CLI display.

    Formatting rules:
    - Action type (CAST, DRAW, ATTACK, etc.): yellow bold
    - Card names: white (via format_card_display)
    - Additional details/card text: dim grey

    Returns None for Pass actions.
    """
    if "Pass" in action_name or "Unknown" in action_name:
        return None

    player_idx = 0 if player == "Player" else 1
    state = env.state

    upper_name = action_name.upper()

    if "PLAY_LAND" in upper_name or action_name.startswith("Play"):
        parts = action_name.split(":")
        land = parts[1].strip() if len(parts) > 1 else "Land"
        lands = get_lands_by_type(state.players[player_idx].battlefield)
        # Format as "Mountain x2, Island x1"
        lands_parts = [f"{format_card_display(n)} x{c}" for n, c in lands.items()]
        lands_str = ", ".join(lands_parts) if lands_parts else format_card_display(land) + " x1"
        return (
            f"[bold yellow]PLAY_LAND:[/bold yellow] {format_card_display(land)} "
            f"[dim](Board: {lands_str})[/dim]"
        )

    if "CAST" in upper_name or action_name.startswith("Cast"):
        parts = action_name.split(":")
        spell = parts[1].strip() if len(parts) > 1 else action_name
        action_str = f"[bold yellow]CAST:[/bold yellow] {format_card_display(spell)}"
        card_text = get_card_text(spell)
        if card_text:
            action_str += f"\n  [dim]→ {card_text}[/dim]"
        return action_str

    if "ATTACK" in upper_name:
        # Parse creature names from attack action
        parts = action_name.split(":")
        if len(parts) > 1:
            # Always get actual creature names
            creatures = [
                c for c in state.players[player_idx].battlefield if c.card_type == CardType.CREATURE
            ]
            if creatures:
                formatted = ", ".join(
                    format_card_display(c.name, c.power, c.toughness) for c in creatures
                )
                return f"[bold yellow]ATTACK:[/bold yellow] Attacking with {formatted}"
            return "[bold yellow]ATTACK:[/bold yellow] [dim]No creatures to attack[/dim]"
        return f"[bold yellow]ATTACK:[/bold yellow] {action_name}"

    if "BLOCK" in upper_name:
        parts = action_name.split(":")
        if len(parts) > 1:
            blockers_str = parts[1].strip()
            return f"[bold yellow]BLOCK:[/bold yellow] {format_card_display(blockers_str)}"
        return f"[bold yellow]BLOCK:[/bold yellow] {action_name}"

    if "DRAW" in upper_name:
        parts = action_name.split(":")
        drawn = parts[1].strip() if len(parts) > 1 else "Card"
        action_str = f"[bold yellow]DRAW:[/bold yellow] {format_card_display(drawn)}"
        card_text = get_card_text(drawn)
        if card_text:
            action_str += f"\n  [dim]→ {card_text}[/dim]"
        return action_str

    if "UNTAP" in upper_name:
        return f"[bold yellow]UNTAP:[/bold yellow] [dim]{action_name}[/dim]"

    return action_name


# =============================================================================
# User Prompts
# =============================================================================


def prompt_agent_selection() -> str:
    """Prompt user to select an agent."""
    console.print("\n[bold cyan]Select Player Agent[/bold cyan]")

    available = list_agents()
    special = ["demo"]
    ready = ["random", "greedy_aggro"]
    trainable = ["ppo", "causal"]

    console.print("\n[magenta]Demo/Showcase:[/magenta]")
    console.print("  1. demo - Scripted showcase demonstrating game mechanics")

    console.print("\n[green]Ready to use:[/green]")
    for i, name in enumerate(ready, 2):
        desc = {
            "random": "Random action selection",
            "greedy_aggro": "Greedy aggressive strategy",
        }.get(name, "")
        console.print(f"  {i}. {name} - {desc}")

    console.print("\n[yellow]Requires training:[/yellow]")
    for i, name in enumerate(trainable, len(ready) + 2):
        desc = {
            "ppo": "PPO reinforcement learning",
            "causal": "Causal reinforcement learning",
        }.get(name, "")
        console.print(f"  {i}. {name} - {desc}")

    console.print("\n[dim]Train agents with: uv run python scripts/runner/run_training.py[/dim]")

    choices = special + ready + trainable
    while True:
        choice = Prompt.ask("\nEnter agent name or number", default="demo")
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
        elif choice in choices or choice in available:
            return choice
        console.print("[red]Invalid selection.[/red]")


def prompt_deck_selection(prompt_text: str, default: str) -> str:
    """Prompt user to select a deck archetype."""
    archetypes = list_archetypes()

    console.print(f"\n[bold cyan]{prompt_text}[/bold cyan]")
    for i, name in enumerate(archetypes, 1):
        console.print(f"  {i}. {name}")

    while True:
        choice = Prompt.ask("Enter archetype name or number", default=default)
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(archetypes):
                return archetypes[idx]
        elif choice in archetypes:
            return choice
        console.print("[red]Invalid selection.[/red]")


def prompt_speed_selection() -> str:
    """Prompt user to select visualization speed."""
    console.print("\n[bold cyan]Visualization Speed[/bold cyan]")
    console.print("  1. slow - 3s phases (detailed review)")
    console.print("  2. medium - 1.5s phases")
    console.print("  3. fast - 0.3s phases")
    console.print("  4. instant - no delays (testing)")

    choice = Prompt.ask(
        "Select speed",
        default="fast",
        choices=["slow", "medium", "fast", "instant", "1", "2", "3", "4"],
    )
    return {"1": "slow", "2": "medium", "3": "fast", "4": "instant"}.get(choice, choice)


def get_default_agent_for_deck(deck_name: str) -> str:
    """Return the default heuristic agent for a deck archetype."""
    from mtg.agents import heuristic_for_deck

    normalized = deck_name.lower().replace(" ", "_").replace("-", "_")
    return heuristic_for_deck(normalized) or "greedy_aggro"


def display_deck_contents(archetype_name: str, label: str) -> None:
    """Display deck contents with proper formatting."""
    try:
        archetype = get_archetype(archetype_name)
        console.print(f"\n[bold]{label} Deck: {archetype.display_name}[/bold]")
        console.print(f"[dim]{archetype.description}[/dim]")

        registry = CardRegistry.get_instance()
        creatures, spells, lands = [], [], []

        for card_name, count in archetype.card_list:
            try:
                card = registry.get(card_name)
                entry = f"{format_card_display(card_name)} x{count}"
                if card.card_type == CardType.CREATURE:
                    creatures.append(entry)
                elif card.card_type == CardType.LAND:
                    lands.append(entry)
                else:
                    spells.append(entry)
            except (KeyError, AttributeError):
                spells.append(f"{card_name} x{count}")

        if creatures:
            console.print(f"  ⚔️ Creatures: {', '.join(creatures)}")
        if spells:
            console.print(f"  ✨ Spells: {', '.join(spells)}")
        if lands:
            console.print(f"  🌍 Lands: {', '.join(lands)}")
    except Exception as e:
        console.print(f"[red]Error loading deck: {e}[/red]")


def prompt_agent_type(deck_name: str, role: str) -> tuple[str, str | None]:
    """Prompt user to select agent type for a deck.

    Automatically discovers trained models matching the selected deck.
    Models are named {agent_type}_{deck_name}, so we can auto-find them.

    Args:
        deck_name: The deck archetype name.
        role: 'Player' or 'Opponent'.

    Returns:
        Tuple of (agent_name, model_path or None).
    """
    from mtg.utils.interactive import discover_trained_models, find_model_for_deck

    # Auto-find models matching this deck
    ppo_model_path = find_model_for_deck("ppo", deck_name)
    causal_model_path = find_model_for_deck("causal", deck_name)

    has_ppo = ppo_model_path is not None
    has_causal = causal_model_path is not None

    console.print(f"\n[bold cyan]Select {role} Agent Type[/bold cyan]")
    console.print("  1. [green]Heuristic[/green] - Rule-based strategy")

    if has_ppo:
        model_file = Path(ppo_model_path).name  # type: ignore[arg-type]
        console.print(
            f"  2. [green]Reinforcement Learning[/green] - PPO trained for {deck_name} "
            f"({model_file})"
        )
    else:
        console.print(
            f"  2. [yellow]Reinforcement Learning[/yellow] - PPO (no trained model for {deck_name})"
        )

    if has_causal:
        model_file = Path(causal_model_path).name  # type: ignore[arg-type]
        console.print(
            f"  3. [green]Causal RL[/green] - Causal agent for {deck_name} ({model_file})"
        )
    else:
        console.print(f"  3. [yellow]Causal RL[/yellow] - (no trained model for {deck_name})")

    choice = Prompt.ask(
        f"Agent type for {role}",
        default="1",
        choices=["1", "2", "3", "heuristic", "rl", "causal"],
    )

    if choice in ["2", "rl"]:
        if has_ppo:
            console.print(
                f"[green]Using PPO model for {deck_name}: {Path(ppo_model_path).name}[/green]"  # type: ignore[arg-type]
            )
            return ("ppo", ppo_model_path)
        # No auto-match; check all models
        all_models = discover_trained_models()
        ppo_models = [m for m in all_models if m.get("agent_type") == "ppo"]
        if ppo_models:
            console.print(
                f"[yellow]No PPO model trained for {deck_name}. "
                f"Select from available models:[/yellow]"
            )
            model_path = _prompt_model_selection(ppo_models, "PPO")
            if model_path:
                return ("ppo", model_path)
        console.print("[yellow]No trained PPO models. Using Heuristic.[/yellow]")
        return (get_default_agent_for_deck(deck_name), None)

    elif choice in ["3", "causal"]:
        if has_causal:
            console.print(
                f"[green]Using Causal model for {deck_name}: {Path(causal_model_path).name}[/green]"  # type: ignore[arg-type]
            )
            return ("causal", causal_model_path)
        all_models = discover_trained_models()
        causal_models = [m for m in all_models if m.get("agent_type") == "causal"]
        if causal_models:
            console.print(
                f"[yellow]No Causal model trained for {deck_name}. "
                f"Select from available models:[/yellow]"
            )
            model_path = _prompt_model_selection(causal_models, "Causal")
            if model_path:
                return ("causal", model_path)
        console.print("[yellow]No trained Causal models. Using Heuristic.[/yellow]")
        return (get_default_agent_for_deck(deck_name), None)

    # Default: heuristic
    return (get_default_agent_for_deck(deck_name), None)


def _prompt_model_selection(models: list[dict], agent_type: str) -> str | None:
    """Prompt user to select a trained model.

    Args:
        models: List of model info dicts.
        agent_type: Agent type name for display.

    Returns:
        Selected model path or None.
    """
    if not models:
        return None

    console.print(f"\n[bold cyan]Select {agent_type} Model[/bold cyan]")

    for i, model in enumerate(models[:5], 1):
        name = model.get("name", "unknown")[:40]
        player_deck = model.get("player_deck", "?")
        opponent_deck = model.get("opponent_deck", "?")
        console.print(f"  {i}. {name} ({player_deck} vs {opponent_deck})")

    if len(models) > 5:
        console.print(f"  ... and {len(models) - 5} more")

    choices = [str(i) for i in range(1, min(len(models) + 1, 6))]
    choice = Prompt.ask("Select model", default="1", choices=choices)

    idx = int(choice) - 1
    return models[idx]["path"]


def _create_agent_for_gameplay(
    agent_name: str,
    model_path: str | None,
    deck_archetype: str,
) -> Any:
    """Create an agent for gameplay, handling RL agent loading.

    Args:
        agent_name: Agent type name ('ppo', 'causal', or heuristic name).
        model_path: Path to trained model (for RL agents).
        deck_archetype: Deck archetype for observation/action dims.

    Returns:
        Initialized agent instance.
    """
    from mtg.agents import CausalAgent, PPOAgent

    # For RL agents, we need to load from model path
    if agent_name == "ppo" and model_path:
        # Get dims from a temporary env (we need obs/action space sizes)
        temp_env = MTGEnv(
            deck_archetype=deck_archetype,
            opponent_archetype="mono_red_aggro",
            max_turns=10,
        )
        obs_dim = temp_env.observation_space.shape[0]
        act_dim = temp_env.action_space.n

        agent = PPOAgent(observation_dim=obs_dim, action_dim=act_dim)
        agent.load(model_path)
        console.print(f"[green]Loaded PPO model: {Path(model_path).name}[/green]")
        return agent

    elif agent_name == "causal" and model_path:
        temp_env = MTGEnv(
            deck_archetype=deck_archetype,
            opponent_archetype="mono_red_aggro",
            max_turns=10,
        )
        obs_dim = temp_env.observation_space.shape[0]
        act_dim = temp_env.action_space.n

        agent = CausalAgent(observation_dim=obs_dim, action_dim=act_dim)
        agent.load(model_path)
        console.print(f"[green]Loaded Causal model: {Path(model_path).name}[/green]")
        return agent

    # Fallback to heuristic agents
    return get_agent(agent_name)


def prompt_gameplay_config() -> GameplayConfig:
    """Prompt user for game configuration."""
    console.print("\n[bold cyan]Game Mode Selection[/bold cyan]")
    console.print(
        "  1. [green]Demo Mode[/green] - Fixed decks (Mono-Red vs Azorius Control), "
        "reproducible game"
    )
    console.print("  2. [blue]Custom Game[/blue] - Choose decks and agents")

    mode_choice = Prompt.ask(
        "Select mode",
        default="1",
        choices=["1", "2", "demo", "custom"],
    )

    is_demo = mode_choice in ["1", "demo"]

    if is_demo:
        # Demo mode: fixed decks, correct heuristic agents, reproducible seed
        console.print("\n[bold green]Demo Mode[/bold green]")
        console.print("[dim]Fixed matchup: Mono-Red Aggro vs Azorius Control[/dim]")
        console.print("[dim]Reproducible draws with fixed seed (42)[/dim]")

        player_deck = "mono_red_aggro"
        opponent_deck = "azorius_control"
        player_agent = get_default_agent_for_deck(player_deck)  # greedy_aggro
        opponent_agent = get_default_agent_for_deck(opponent_deck)  # control
        seed = 42  # Fixed seed for reproducibility

        console.print(f"\n[dim]Agents: Player → {player_agent}, Opponent → {opponent_agent}[/dim]")
        player_model_path = None
        opponent_model_path = None
    else:
        # Custom mode: choose decks, then agent types
        console.print("\n[bold blue]Custom Game[/bold blue]")

        # Step 1: Choose Player Deck
        player_deck = prompt_deck_selection("Select Player Deck", "mono_red_aggro")
        display_deck_contents(player_deck, "Player")

        # Step 2: Choose Player Agent Type
        player_agent, player_model_path = prompt_agent_type(player_deck, "Player")

        # Step 3: Choose Opponent Deck
        opponent_deck = prompt_deck_selection("Select Opponent Deck", "azorius_control")
        display_deck_contents(opponent_deck, "Opponent")

        # Step 4: Choose Opponent Agent Type
        opponent_agent, opponent_model_path = prompt_agent_type(opponent_deck, "Opponent")

        seed = None  # Random seed for variety

        console.print(f"\n[dim]Agents: Player → {player_agent}, Opponent → {opponent_agent}[/dim]")

    speed = prompt_speed_selection()
    num_turns = IntPrompt.ask("Number of turns", default=5)

    return GameplayConfig(
        player_agent=player_agent,
        opponent_agent=opponent_agent,
        player_model_path=player_model_path,
        opponent_model_path=opponent_model_path,
        player_deck=player_deck,
        opponent_deck=opponent_deck,
        speed=speed,
        num_turns=min(max(num_turns, 1), 10),
        save_report=True,
        seed=seed,
        is_demo=is_demo,
    )


# =============================================================================
# Turn Summary using cli_display.print_turn_summary
# =============================================================================


def build_turn_summary_from_snapshot(
    turn: int,
    player_actions: list,
    opponent_actions: list,
    player_life: int,
    opponent_life: int,
    player_life_change: int,
    opponent_life_change: int,
    tracked_player_hand: list[str],
    tracked_opponent_hand: list[str],
    tracked_player_lands: dict[str, int],
    tracked_opponent_lands: dict[str, int],
    tracked_player_creatures: list[str],
    tracked_opponent_creatures: list[str],
    tracked_player_graveyard: list[str],
    tracked_opponent_graveyard: list[str],
    tracked_player_exile: list[str] | None = None,
    tracked_opponent_exile: list[str] | None = None,
) -> None:
    """Build and print turn summary using tracked state data for accuracy."""
    # Build action tuples
    p_actions = list(player_actions)
    o_actions = list(opponent_actions)

    # Build hand tuples from tracked names
    p_hand = [(name, "") for name in tracked_player_hand]
    o_hand = [(name, "") for name in tracked_opponent_hand]

    # Build creature lists (formatted) from tracked creature dicts with current stats
    # Special handling for creatures with variable power (e.g., Haughty Djinn)
    def _calculate_creature_power(creature: dict, graveyard: list[str]) -> int:
        """Calculate effective power, handling variable power creatures."""
        base = (
            creature.get("base_power", 0)
            + creature.get("power_bonus", 0)
            + creature.get("temp_power_bonus", 0)
        )
        # Haughty Djinn: power equals instants/sorceries in graveyard
        if creature["name"] == "Haughty Djinn":
            registry = CardRegistry.get_instance()
            instant_sorcery_count = 0
            for card_name in graveyard:
                try:
                    card = registry.get(card_name)
                    if card.card_type in {CardType.INSTANT, CardType.SORCERY}:
                        instant_sorcery_count += 1
                except KeyError:
                    pass
            return instant_sorcery_count
        return base

    p_creatures = []
    for c in tracked_player_creatures:
        effective_power = _calculate_creature_power(c, tracked_player_graveyard)
        effective_toughness = (
            c.get("base_toughness", 0)
            + c.get("toughness_bonus", 0)
            + c.get("temp_toughness_bonus", 0)
        )
        p_creatures.append(
            format_card_display(
                c["name"],
                power=effective_power,
                toughness=effective_toughness,
                attached_tokens=c.get("attached_tokens", []),
            )
        )
    o_creatures = []
    for c in tracked_opponent_creatures:
        effective_power = _calculate_creature_power(c, tracked_opponent_graveyard)
        effective_toughness = (
            c.get("base_toughness", 0)
            + c.get("toughness_bonus", 0)
            + c.get("temp_toughness_bonus", 0)
        )
        o_creatures.append(
            format_card_display(
                c["name"],
                power=effective_power,
                toughness=effective_toughness,
                attached_tokens=c.get("attached_tokens", []),
            )
        )

    # Build graveyard tuples from tracked names (look up card type from registry)
    registry = CardRegistry.get_instance()

    def get_card_type(name: str) -> str:
        try:
            card = registry.get(name)
            return card.card_type.value if card.card_type else "unknown"
        except KeyError:
            return "unknown"

    p_graveyard = [(name, get_card_type(name)) for name in tracked_player_graveyard]
    o_graveyard = [(name, get_card_type(name)) for name in tracked_opponent_graveyard]

    # Empty enchantments (would need separate tracking if important)
    p_enchantments: list[str] = []
    o_enchantments: list[str] = []

    # Collect tokens from creatures' attached_tokens
    # Format: list of token names with the creature they're attached to
    p_tokens: list[str] = []
    for creature in tracked_player_creatures:
        for token in creature.get("attached_tokens", []):
            p_tokens.append(f"{token} (on {creature['name']})")

    o_tokens: list[str] = []
    for creature in tracked_opponent_creatures:
        for token in creature.get("attached_tokens", []):
            o_tokens.append(f"{token} (on {creature['name']})")

    print_turn_summary(
        turn=turn,
        player_actions=p_actions,
        opponent_actions=o_actions,
        player_life_change=player_life_change,
        opponent_life_change=opponent_life_change,
        player_life=player_life,
        opponent_life=opponent_life,
        player_lands=tracked_player_lands,
        opponent_lands=tracked_opponent_lands,
        player_hand=p_hand,
        opponent_hand=o_hand,
        player_creatures=p_creatures,
        opponent_creatures=o_creatures,
        player_enchantments=p_enchantments,
        opponent_enchantments=o_enchantments,
        player_graveyard=p_graveyard,
        opponent_graveyard=o_graveyard,
        player_exile=tracked_player_exile,
        opponent_exile=tracked_opponent_exile,
        player_tokens=p_tokens,
        opponent_tokens=o_tokens,
    )


# =============================================================================
# Main Game Loop
# =============================================================================


def run_game(config: GameplayConfig) -> dict[str, Any]:
    """Run a game using MTGEnv with rich visualization.

    All game logic is handled by the environment.
    The agent selects actions from legal actions.
    """
    delays = SPEED_PRESETS.get(config.speed, SPEED_PRESETS["fast"])

    # Create agents (with potential RL model loading)
    agent = _create_agent_for_gameplay(
        config.player_agent,
        config.player_model_path,
        config.player_deck,
    )
    opponent_agent_name = config.opponent_agent or get_default_agent_for_deck(config.opponent_deck)
    opponent_agent = _create_agent_for_gameplay(
        opponent_agent_name,
        config.opponent_model_path,
        config.opponent_deck,
    )

    # Enable auto_resolve when the player agent is a trained RL model.
    # The RL model was trained with auto_resolve=True (mechanical decisions
    # like mana tapping / targeting are handled automatically), so gameplay
    # must match for the policy to work correctly.
    is_rl_agent = config.player_agent in {"ppo", "causal"} and config.player_model_path
    env = MTGEnv(
        deck_archetype=config.player_deck,
        opponent_archetype=config.opponent_deck,
        max_turns=config.num_turns,
        reward_type="shaped",
        opponent_agent=opponent_agent,
        auto_resolve=bool(is_rl_agent),
    )

    # Display mode info
    if config.is_demo:
        console.print("[dim]🎲 Demo mode: Fixed seed (42) for reproducible draws[/dim]\n")

    # Initialize recorder
    recorder = GameRecorder(
        player_deck=config.player_deck,
        opponent_deck=config.opponent_deck,
        player_agent=config.player_agent.upper(),
        opponent_agent=opponent_agent_name.upper(),
    )

    # Reset environment with optional seed for reproducibility
    obs, info = env.reset(seed=config.seed)

    # === COIN FLIP ===
    # Use the environment's coin flip to keep game state consistent.
    player_on_play = env.state.player_on_play
    recorder.set_player_on_play(player_on_play)

    # === GAME START ===
    print_divider("Game Start")
    time.sleep(delays["turn"])

    console.print("[bold yellow]Determining who goes first...[/bold yellow]\n")
    time.sleep(delays["action"])
    print_play_draw_selection_full(player_on_play)
    time.sleep(delays["phase"])

    # === MULLIGAN PHASE ===
    # Capture hands immediately as Card objects AND as names
    # This ensures we have accurate snapshots before any actions are processed
    player_hand = list(env.state.players[0].hand)
    opponent_hand = list(env.state.players[1].hand)

    # IMPORTANT: Capture card names IMMEDIATELY before any env.step() processing
    # This is the "ground truth" for Turn 1 start state
    initial_player_hand_names = [c.name for c in player_hand]
    initial_opponent_hand_names = [c.name for c in opponent_hand]

    # Mulligan variables
    mulligan_new_hand = None
    returned_cards: list = []  # Track all cards returned to bottom
    final_kept_hand = None
    original_player_hand = list(player_hand)  # Save original for display

    # Mulligan loop for player - London Mulligan rules
    max_mulligans = 6  # Can mulligan down to 1 card
    mulligans_taken = 0

    while mulligans_taken < max_mulligans:
        land_count = sum(1 for c in player_hand if c.card_type == CardType.LAND)

        # Decision: Keep if 2-5 lands, or forced to keep at 1 card
        if 1 <= land_count <= 5:
            player_keeps = True
            break

        # Need to mulligan - use rules engine to shuffle and draw new 7
        player_keeps = False
        mulligans_taken += 1

        # Execute mulligan through rules engine (always draws 7 cards)
        env.state = env.rules_engine.execute_mulligan(env.state, keep=False)

        # Get the new 7-card hand from the environment
        player_hand = list(env.state.players[0].hand)
        mulligan_new_hand = list(player_hand)  # Copy for display (full 7 cards)

        # London Mulligan: Put cards on bottom equal to mulligans taken
        # Choose worst cards to return (highest CMC non-lands, or excess lands)
        cards_to_bottom = mulligans_taken
        returned_cards = []

        for _ in range(cards_to_bottom):
            if not player_hand:
                break

            non_lands = [c for c in player_hand if c.card_type != CardType.LAND]
            lands = [c for c in player_hand if c.card_type == CardType.LAND]

            # Strategy: Keep 2-3 lands, return excess lands or highest CMC spells
            if len(lands) > 3:
                # Too many lands - return one
                card_to_return = lands[-1]
            elif non_lands:
                # Return highest CMC spell
                card_to_return = max(non_lands, key=lambda c: c.mana_cost.cmc if c.mana_cost else 0)
            else:
                # Only lands left, return one
                card_to_return = player_hand[-1]

            returned_cards.append(card_to_return)
            env.state.players[0].hand.remove(card_to_return)
            env.state.players[0].deck.insert(0, card_to_return)  # Bottom of deck
            player_hand = list(env.state.players[0].hand)

        final_kept_hand = list(player_hand)

    # If we exited loop without keeping, force keep
    if not player_keeps:
        player_keeps = True
        final_kept_hand = list(player_hand)

    # After mulligan loop: update player_hand to current state and ensure final_kept_hand is set
    player_hand = list(env.state.players[0].hand)
    if mulligans_taken > 0:
        # Player mulliganed - final_kept_hand should have fewer than 7 cards
        final_kept_hand = list(player_hand)

    # Finalize player mulligan decision - this transitions priority to opponent
    env.state.mulligan_count[0] = mulligans_taken
    env.state.priority_player = 0  # Ensure priority is on player
    env.state = env.rules_engine.execute_mulligan(env.state, keep=True)

    # === OPPONENT MULLIGAN ===
    # Priority should now be on opponent after player's keep
    opponent_hand = list(env.state.players[1].hand)
    original_opponent_hand = list(opponent_hand)
    opponent_mulligan_new_hand = None
    opponent_returned_cards: list = []
    opponent_final_kept_hand = None
    opponent_mulligans_taken = 0
    opponent_keeps = False

    while opponent_mulligans_taken < max_mulligans:
        opp_land_count = sum(1 for c in opponent_hand if c.card_type == CardType.LAND)

        # Decision: Keep if 1-5 lands
        if 1 <= opp_land_count <= 5:
            opponent_keeps = True
            break

        # Need to mulligan
        opponent_keeps = False
        opponent_mulligans_taken += 1

        # Execute mulligan through rules engine (always draws 7 cards)
        env.state = env.rules_engine.execute_mulligan(env.state, keep=False)

        # Get the new 7-card hand from the environment
        opponent_hand = list(env.state.players[1].hand)
        opponent_mulligan_new_hand = list(opponent_hand)

        # London Mulligan: Put cards on bottom equal to mulligans taken
        cards_to_bottom = opponent_mulligans_taken
        opponent_returned_cards = []

        for _ in range(cards_to_bottom):
            if not opponent_hand:
                break

            non_lands = [c for c in opponent_hand if c.card_type != CardType.LAND]
            lands = [c for c in opponent_hand if c.card_type == CardType.LAND]

            if len(lands) > 3:
                card_to_return = lands[-1]
            elif non_lands:
                card_to_return = max(non_lands, key=lambda c: c.mana_cost.cmc if c.mana_cost else 0)
            else:
                card_to_return = opponent_hand[-1]

            opponent_returned_cards.append(card_to_return)
            env.state.players[1].hand.remove(card_to_return)
            env.state.players[1].deck.insert(0, card_to_return)
            opponent_hand = list(env.state.players[1].hand)

        opponent_final_kept_hand = list(opponent_hand)

    # If we exited loop without keeping, force keep
    if not opponent_keeps:
        opponent_keeps = True
        opponent_final_kept_hand = list(opponent_hand)

    # Update opponent_hand to current state
    opponent_hand = list(env.state.players[1].hand)
    if opponent_mulligans_taken > 0:
        opponent_final_kept_hand = list(opponent_hand)

    # Finalize opponent mulligan decision - this transitions to Turn 1
    env.state.mulligan_count[1] = opponent_mulligans_taken
    env.state.priority_player = 1  # Ensure priority is on opponent
    env.state = env.rules_engine.execute_mulligan(env.state, keep=True)

    # Update initial opponent hand names for tracking
    initial_opponent_hand_names = [c.name for c in opponent_hand]

    # Record initial state - use original hand for opening, current for kept
    original_hand_tuples = [
        (c.name, c.mana_cost.to_text() if c.mana_cost else "") for c in original_player_hand
    ]
    player_hand_tuples = [
        (c.name, c.mana_cost.to_text() if c.mana_cost else "") for c in player_hand
    ]

    original_opponent_hand_tuples = [
        (c.name, c.mana_cost.to_text() if c.mana_cost else "") for c in original_opponent_hand
    ]

    recorder.record_initial_state(
        player_on_play=player_on_play,
        player_opening_hand=original_hand_tuples,
        player_kept=mulligans_taken == 0,
        player_mulligan_hand=[
            (c.name, c.mana_cost.to_text() if c.mana_cost else "")
            for c in (mulligan_new_hand or [])
        ],
        player_returned_cards=[
            (c.name, c.mana_cost.to_text() if c.mana_cost else "") for c in returned_cards
        ],
        player_kept_hand=player_hand_tuples,
        opponent_opening_hand=original_opponent_hand_tuples,
        opponent_kept=opponent_mulligans_taken == 0,
        player_mulligans=mulligans_taken,
        opponent_mulligans=opponent_mulligans_taken,
    )

    # Show detailed mulligan with actual hands
    print_mulligan_phase_detailed(
        player_hand=original_player_hand,  # Show original opening hand
        opponent_hand=opponent_hand,
        player_keeps=mulligans_taken == 0,
        mulligan_new_hand=mulligan_new_hand,
        returned_cards=returned_cards,
        final_kept_hand=final_kept_hand or player_hand,
        opponent_original_hand=original_opponent_hand,
        opponent_keeps=opponent_mulligans_taken == 0,
        opponent_mulligan_new_hand=opponent_mulligan_new_hand,
        opponent_returned_cards=opponent_returned_cards,
        opponent_final_kept_hand=opponent_final_kept_hand or opponent_hand,
    )
    time.sleep(delays["phase"])

    # === MAIN GAME LOOP ===
    # After execute_mulligan(keep=True) for both players, the game should be at Turn 1
    # Refresh observation and info from the properly transitioned state
    obs = env.obs_builder.build_flat_observation(env.state, player_id=0)
    info = env._get_info()

    # Track actions from action_log - skip mulligan actions
    action_log = info.get("action_log", [])
    # Find first non-mulligan action and its turn number
    first_game_action_idx = 0
    first_turn_number = 0
    for i, act in enumerate(action_log):
        if act.get("phase") != "MULLIGAN":
            first_game_action_idx = i
            first_turn_number = act.get("turn", 0)
            break
    else:
        first_game_action_idx = len(action_log)
        first_turn_number = info.get("turn", 0)

    processed_action_count = first_game_action_idx

    # For Turn 1 "start" state, use the names captured BEFORE the mulligan loop
    # Player's hand is updated if they mulliganed (using final_kept_hand)
    # Opponent's hand uses the initial capture (7 cards)
    player_hand_for_turn1 = (
        [c.name for c in final_kept_hand] if final_kept_hand else initial_player_hand_names
    )
    # Opponent ALWAYS has their initial 7-card hand at Turn 1 start
    opponent_hand_for_turn1 = initial_opponent_hand_names

    saved_turn_start_hands = {
        1: {
            "player_hand": player_hand_for_turn1,  # Player's hand after mulligan
            "opponent_hand": opponent_hand_for_turn1,  # Opponent's 7-card hand
            "player_graveyard": [],
            "opponent_graveyard": [],
            "player_lands": {},
            "opponent_lands": {},
            "player_creatures": [],
            "opponent_creatures": [],
        }
    }

    # State tracking - start with the hands as they were at the end of mulligan phase
    # Player: uses final kept hand (6 if mulliganed, 7 otherwise)
    # Opponent: uses initial 7-card hand (captured before any processing)
    tracked_player_hand: list[str] = player_hand_for_turn1.copy()
    tracked_opponent_hand: list[str] = opponent_hand_for_turn1.copy()
    tracked_player_graveyard: list[str] = []
    tracked_opponent_graveyard: list[str] = []
    tracked_player_exile: list[str] = []
    tracked_opponent_exile: list[str] = []
    tracked_player_lands: dict[str, int] = {}
    tracked_opponent_lands: dict[str, int] = {}
    # Creature tracking: list of dicts with name, attached_tokens, power/toughness bonuses
    tracked_player_creatures: list[dict] = []
    tracked_opponent_creatures: list[dict] = []

    # Snapshot at player segment end (for Turn End display)
    segment_end_snapshot: dict = {}

    # Opponent's state at the start of player's segment (for Turn End display)
    # When showing "Player Turn End", opponent's state should be unchanged from turn start
    opponent_at_player_turn_start: dict = {
        "hand": opponent_hand_for_turn1.copy(),
        "graveyard": [],
        "lands": {},
        "creatures": [],  # Will store dicts with name, attached_tokens
    }

    # State tracking
    current_display_turn = 0
    current_display_player = ""

    # Life tracking - track current life incrementally from actions
    tracked_player_life = 20  # Current tracked life
    tracked_opponent_life = 20
    turn_start_player_life = 20  # Life at start of current turn (for delta)
    turn_start_opponent_life = 20

    # Actions by phase for current segment
    player_phase_actions: dict[str, list[str]] = {}
    opponent_phase_actions: dict[str, list[str]] = {}

    # Actions for turn summary
    turn_player_actions: list[tuple[str, str]] = []
    turn_opponent_actions: list[tuple[str, str]] = []

    step_count = 0
    max_steps = 500

    def phase_name_to_display(phase_name: str) -> str:
        """Convert phase name to display name matching HTML template."""
        return {
            "UNTAP": "Untap",
            "UPKEEP": "Upkeep",
            "DRAW": "Draw",
            "MAIN_PRECOMBAT": "Main 1",
            "COMBAT_BEGIN": "Combat",
            "COMBAT_ATTACKERS": "Combat - Declare Attackers",
            "COMBAT_BLOCKERS": "Combat - Declare Blockers",
            "COMBAT_DAMAGE": "Combat - Damage",
            "MAIN_POSTCOMBAT": "Main 2",
            "END_STEP": "End",
            "CLEANUP": "Cleanup",
        }.get(phase_name, phase_name)

    def display_player_segment(
        player: str, actions_by_phase: dict[str, list[str]], turn: int
    ) -> None:
        """Display all collected phases for a player's turn segment."""
        nonlocal segment_end_snapshot
        if not actions_by_phase:
            return
        display_complete_turn(turn, player, actions_by_phase, delays, is_first_turn=(turn == 1))
        # Use snapshot if available, otherwise use live state
        if segment_end_snapshot:
            print_turn_end_state_from_snapshot(
                segment_end_snapshot["player_hand"],
                segment_end_snapshot["opponent_hand"],
                segment_end_snapshot["player_graveyard"],
                segment_end_snapshot["opponent_graveyard"],
                segment_end_snapshot.get("player_lands"),
                segment_end_snapshot.get("opponent_lands"),
                segment_end_snapshot.get("player_creatures"),
                segment_end_snapshot.get("opponent_creatures"),
                env=env,
            )
            segment_end_snapshot = {}
        else:
            print_turn_end_state(env, player)
        time.sleep(delays["action"])

    def process_new_actions(action_log: list[dict], start_idx: int) -> int:
        """Process and display all new actions from action_log.

        Returns: The new action count (next start_idx).
        """
        nonlocal current_display_turn, current_display_player
        nonlocal player_phase_actions, opponent_phase_actions
        nonlocal turn_player_actions, turn_opponent_actions
        nonlocal turn_start_player_life, turn_start_opponent_life
        nonlocal tracked_player_life, tracked_opponent_life
        nonlocal tracked_player_hand, tracked_opponent_hand
        nonlocal tracked_player_graveyard, tracked_opponent_graveyard
        nonlocal tracked_player_exile, tracked_opponent_exile
        nonlocal tracked_player_lands, tracked_opponent_lands
        nonlocal tracked_player_creatures, tracked_opponent_creatures
        nonlocal segment_end_snapshot, opponent_at_player_turn_start

        new_actions = action_log[start_idx:]

        def _effective_stats(creature: dict) -> tuple[int, int]:
            power = (
                creature.get("base_power", 0)
                + creature.get("power_bonus", 0)
                + creature.get("temp_power_bonus", 0)
            )
            toughness = (
                creature.get("base_toughness", 0)
                + creature.get("toughness_bonus", 0)
                + creature.get("temp_toughness_bonus", 0)
            )
            return power, toughness

        def _matches_target(creature: dict, target_name: str) -> bool:
            return creature["name"] == target_name or target_name in creature["name"]

        def _find_nth_matching(
            creatures: list[dict], target_name: str, occurrence: int
        ) -> dict | None:
            count = 0
            for creature in creatures:
                if _matches_target(creature, target_name):
                    if count == occurrence:
                        return creature
                    count += 1
            return None

        def _find_by_id(creatures: list[dict], card_id: int | None) -> dict | None:
            if card_id is None:
                return None
            for creature in creatures:
                if creature.get("card_id") == card_id:
                    return creature
            return None

        for action in new_actions:
            action_turn = action.get("turn", 0)
            action_phase = action.get("phase", "")
            action_player_idx = action.get("player", 0)
            action_turn_player_idx = action.get("active_player", action_player_idx)
            action_player = "Player" if action_player_idx == 0 else "Opponent"
            action_turn_player = "Player" if action_turn_player_idx == 0 else "Opponent"

            # Skip mulligan phase actions in display
            if action_phase == "MULLIGAN":
                continue

            # Display turn = (env turn - first_turn_number) + 1
            # This ensures we start at Turn 1 regardless of mulligan processing
            display_turn = (action_turn - first_turn_number) + 1

            # Skip actions from turns beyond the max display turns
            if display_turn > config.num_turns:
                continue

            phase_display = phase_name_to_display(action_phase)

            # Override phase for action types that belong to specific phases
            action_type = action.get("action_type", "")
            if action_type == "BLOCK":
                phase_display = "Combat - Declare Blockers"
            elif action_type == "ATTACK":
                phase_display = "Combat - Declare Attackers"

            # === NEW TURN ===
            if display_turn != current_display_turn:
                # Finish the prior turn before starting the new one
                if current_display_turn > 0:
                    # Capture snapshot BEFORE displaying segment
                    segment_end_snapshot = {
                        "player_hand": tracked_player_hand.copy(),
                        "opponent_hand": tracked_opponent_hand.copy(),
                        "player_graveyard": tracked_player_graveyard.copy(),
                        "opponent_graveyard": tracked_opponent_graveyard.copy(),
                        "player_lands": tracked_player_lands.copy(),
                        "opponent_lands": tracked_opponent_lands.copy(),
                        "player_creatures": tracked_player_creatures.copy(),
                        "opponent_creatures": tracked_opponent_creatures.copy(),
                    }
                    if current_display_player == "Player" and player_phase_actions:
                        display_player_segment("Player", player_phase_actions, current_display_turn)
                        player_phase_actions = {}
                    elif current_display_player == "Opponent" and opponent_phase_actions:
                        display_player_segment(
                            "Opponent", opponent_phase_actions, current_display_turn
                        )
                        opponent_phase_actions = {}

                    # Turn summary with life deltas using tracked state
                    p_life_delta = tracked_player_life - turn_start_player_life
                    o_life_delta = tracked_opponent_life - turn_start_opponent_life

                    build_turn_summary_from_snapshot(
                        turn=current_display_turn,
                        player_actions=turn_player_actions,
                        opponent_actions=turn_opponent_actions,
                        player_life=tracked_player_life,
                        opponent_life=tracked_opponent_life,
                        player_life_change=p_life_delta,
                        opponent_life_change=o_life_delta,
                        tracked_player_hand=tracked_player_hand.copy(),
                        tracked_opponent_hand=tracked_opponent_hand.copy(),
                        tracked_player_lands=tracked_player_lands.copy(),
                        tracked_opponent_lands=tracked_opponent_lands.copy(),
                        tracked_player_creatures=tracked_player_creatures.copy(),
                        tracked_opponent_creatures=tracked_opponent_creatures.copy(),
                        tracked_player_graveyard=tracked_player_graveyard.copy(),
                        tracked_opponent_graveyard=tracked_opponent_graveyard.copy(),
                        tracked_player_exile=tracked_player_exile.copy(),
                        tracked_opponent_exile=tracked_opponent_exile.copy(),
                    )
                    # Record turn summary for HTML report
                    # Counting logic:
                    # - turn_player_actions: player's = no prefix, opponent's = [Opponent]
                    # - turn_opponent_actions: opponent's = no prefix, player's = [Player]

                    # Player's spells = player's turn (no prefix) + opp's turn ([Player])
                    player_spells = sum(
                        1
                        for _, d in turn_player_actions
                        if "CAST" in d.upper() and "[Opponent]" not in d
                    ) + sum(
                        1
                        for _, d in turn_opponent_actions
                        if "CAST" in d.upper() and "[Player]" in d
                    )
                    # Opponent's spells = player's turn ([Opponent]) + opp's turn (no prefix)
                    opponent_spells = sum(
                        1
                        for _, d in turn_player_actions
                        if "CAST" in d.upper() and "[Opponent]" in d
                    ) + sum(
                        1
                        for _, d in turn_opponent_actions
                        if "CAST" in d.upper() and "[Player]" not in d
                    )

                    # Lands can only be played on your own turn (sorcery speed)
                    player_lands = sum(
                        1
                        for _, d in turn_player_actions
                        if "PLAY_LAND" in d.upper() and "[Opponent]" not in d
                    )
                    opponent_lands = sum(
                        1
                        for _, d in turn_opponent_actions
                        if "PLAY_LAND" in d.upper() and "[Player]" not in d
                    )

                    # Creatures = CAST with ⚔️ on own turn
                    player_creature_casts = sum(
                        1
                        for _, d in turn_player_actions
                        if "CAST" in d.upper() and "⚔️" in d and "[Opponent]" not in d
                    )
                    opponent_creature_casts = sum(
                        1
                        for _, d in turn_opponent_actions
                        if "CAST" in d.upper() and "⚔️" in d and "[Player]" not in d
                    )
                    # Subtract countered creatures
                    player_creatures_countered = sum(
                        1 for _, d in turn_player_actions if "COUNTER" in d.upper() and "⚔️" in d
                    )
                    opponent_creatures_countered = sum(
                        1 for _, d in turn_opponent_actions if "COUNTER" in d.upper() and "⚔️" in d
                    )
                    player_creature_casts = max(
                        0, player_creature_casts - player_creatures_countered
                    )
                    opponent_creature_casts = max(
                        0, opponent_creature_casts - opponent_creatures_countered
                    )

                    # Cards drawn = on own turn (no prefix)
                    player_draws = sum(
                        1
                        for _, d in turn_player_actions
                        if "DRAW" in d.upper() and "[Opponent]" not in d
                    )
                    opponent_draws = sum(
                        1
                        for _, d in turn_opponent_actions
                        if "DRAW" in d.upper() and "[Player]" not in d
                    )

                    recorder.record_turn_summary(
                        turn=current_display_turn,
                        player_damage=-o_life_delta if o_life_delta < 0 else 0,
                        opponent_damage=-p_life_delta if p_life_delta < 0 else 0,
                        player_spells=player_spells,
                        opponent_spells=opponent_spells,
                        player_lands=player_lands,
                        opponent_lands=opponent_lands,
                        player_creatures=player_creature_casts,
                        opponent_creatures=opponent_creature_casts,
                        player_draws=player_draws,
                        opponent_draws=opponent_draws,
                    )

                    # Helper to calculate effective P/T from creature dict
                    def _snapshot_creature_pt(c: dict) -> tuple[int, int]:
                        """Calculate creature power/toughness from tracked dict."""
                        base_p = c.get("base_power", 0) or 0
                        base_t = c.get("base_toughness", 0) or 0
                        bonus_p = c.get("power_bonus", 0) + c.get("temp_power_bonus", 0)
                        bonus_t = c.get("toughness_bonus", 0) + c.get("temp_toughness_bonus", 0)
                        return base_p + bonus_p, base_t + bonus_t

                    def _is_instant_or_sorcery(name: str) -> bool:
                        """Check if card is instant or sorcery."""
                        try:
                            card = CardRegistry.get_instance().get(name)
                            return card.card_type in (CardType.INSTANT, CardType.SORCERY)
                        except (KeyError, AttributeError):
                            return False

                    # Record snapshot for HTML report
                    recorder.record_snapshot(
                        turn=current_display_turn,
                        phase="End",
                        active_player="Player" if action_turn_player_idx == 0 else "Opponent",
                        player_life=tracked_player_life,
                        opponent_life=tracked_opponent_life,
                        player_hand=[(n, "") for n in tracked_player_hand],
                        opponent_hand=[(n, "") for n in tracked_opponent_hand],
                        player_lands=tracked_player_lands.copy(),
                        opponent_lands=tracked_opponent_lands.copy(),
                        player_creatures=[
                            {
                                "name": c["name"],
                                "power": _snapshot_creature_pt(c)[0],
                                "toughness": _snapshot_creature_pt(c)[1],
                                "tapped": c.get("tapped", False),
                                "attached_tokens": c.get("attached_tokens", []),
                            }
                            for c in tracked_player_creatures
                        ],
                        opponent_creatures=[
                            {
                                "name": c["name"],
                                "power": _snapshot_creature_pt(c)[0],
                                "toughness": _snapshot_creature_pt(c)[1],
                                "tapped": c.get("tapped", False),
                                "attached_tokens": c.get("attached_tokens", []),
                            }
                            for c in tracked_opponent_creatures
                        ],
                        player_graveyard=[(n, "card") for n in tracked_player_graveyard],
                        opponent_graveyard=[(n, "card") for n in tracked_opponent_graveyard],
                        player_exile=tracked_player_exile.copy(),
                        opponent_exile=tracked_opponent_exile.copy(),
                        player_graveyard_instant_sorcery_count=sum(
                            1 for n in tracked_player_graveyard if _is_instant_or_sorcery(n)
                        ),
                        opponent_graveyard_instant_sorcery_count=sum(
                            1 for n in tracked_opponent_graveyard if _is_instant_or_sorcery(n)
                        ),
                    )
                    turn_player_actions = []
                    turn_opponent_actions = []
                    # Update life tracking for next turn
                    turn_start_player_life = tracked_player_life
                    turn_start_opponent_life = tracked_opponent_life
                    time.sleep(delays["turn"])

                # Start new turn
                console.print()
                console.rule(f"[bold cyan]TURN {display_turn}[/bold cyan]")
                console.print()

                # Show turn start state - use SAVED state for Turn 1 (before any actions)
                if display_turn == 1 and 1 in saved_turn_start_hands:
                    print_turn_start_state_from_data(
                        saved_turn_start_hands[1]["player_hand"],
                        saved_turn_start_hands[1]["opponent_hand"],
                        saved_turn_start_hands[1]["player_graveyard"],
                        saved_turn_start_hands[1]["opponent_graveyard"],
                        saved_turn_start_hands[1]["player_lands"],
                        saved_turn_start_hands[1]["opponent_lands"],
                        saved_turn_start_hands[1]["player_creatures"],
                        saved_turn_start_hands[1]["opponent_creatures"],
                    )
                else:
                    # Use tracked hands for accurate state
                    print_turn_start_state_from_data(
                        tracked_player_hand.copy(),
                        tracked_opponent_hand.copy(),
                        tracked_player_graveyard.copy(),
                        tracked_opponent_graveyard.copy(),
                        tracked_player_lands.copy(),
                        tracked_opponent_lands.copy(),
                        tracked_player_creatures.copy(),
                        tracked_opponent_creatures.copy(),
                    )
                time.sleep(delays["turn"])

                current_display_turn = display_turn
                current_display_player = ""
                player_phase_actions = {}
                opponent_phase_actions = {}

            # === TURN OWNER CHANGE (within same turn) ===
            if action_turn_player != current_display_player:
                # Capture snapshot BEFORE switching for Turn End display
                # For Player Turn End: use opponent's state from player turn START (unchanged)
                # For Opponent Turn End: use current tracked opponent state
                if current_display_player == "Player":
                    # Ending Player's segment - use saved opponent state
                    segment_end_snapshot = {
                        "player_hand": tracked_player_hand.copy(),
                        "opponent_hand": opponent_at_player_turn_start["hand"].copy(),
                        "player_graveyard": tracked_player_graveyard.copy(),
                        "opponent_graveyard": opponent_at_player_turn_start["graveyard"].copy(),
                        "player_lands": tracked_player_lands.copy(),
                        "opponent_lands": opponent_at_player_turn_start["lands"].copy(),
                        "player_creatures": tracked_player_creatures.copy(),
                        "opponent_creatures": opponent_at_player_turn_start["creatures"].copy(),
                    }
                else:
                    # Ending Opponent's segment or first player - use current tracked state
                    segment_end_snapshot = {
                        "player_hand": tracked_player_hand.copy(),
                        "opponent_hand": tracked_opponent_hand.copy(),
                        "player_graveyard": tracked_player_graveyard.copy(),
                        "opponent_graveyard": tracked_opponent_graveyard.copy(),
                        "player_lands": tracked_player_lands.copy(),
                        "opponent_lands": tracked_opponent_lands.copy(),
                        "player_creatures": tracked_player_creatures.copy(),
                        "opponent_creatures": tracked_opponent_creatures.copy(),
                    }

                # When switching TO Player's turn, save opponent's current state
                if action_turn_player == "Player":
                    opponent_at_player_turn_start = {
                        "hand": tracked_opponent_hand.copy(),
                        "graveyard": tracked_opponent_graveyard.copy(),
                        "lands": tracked_opponent_lands.copy(),
                        "creatures": tracked_opponent_creatures.copy(),
                    }

                # Finish the prior player's segment
                if current_display_player == "Player" and player_phase_actions:
                    display_player_segment("Player", player_phase_actions, display_turn)
                    player_phase_actions = {}
                elif current_display_player == "Opponent" and opponent_phase_actions:
                    display_player_segment("Opponent", opponent_phase_actions, display_turn)
                    opponent_phase_actions = {}

                # Print header
                if action_turn_player == "Player":
                    print_player_turn_header(display_turn)
                else:
                    print_opponent_turn_header(display_turn)

                # Show turn start state when switching players within same turn
                if current_display_player != "":  # Not first player of turn
                    # Use tracked hands for accurate state
                    print_turn_start_state_from_data(
                        tracked_player_hand.copy(),
                        tracked_opponent_hand.copy(),
                        tracked_player_graveyard.copy(),
                        tracked_opponent_graveyard.copy(),
                        tracked_player_lands.copy(),
                        tracked_opponent_lands.copy(),
                        tracked_player_creatures.copy(),
                        tracked_opponent_creatures.copy(),
                    )
                    time.sleep(delays["action"])

                current_display_player = action_turn_player

            # === Update tracked hand state based on action ===
            action_type = action.get("action_type", "").upper()
            card_name = action.get("card_name", "")
            details = action.get("details", {})
            card_type_str = details.get("card_type", "")

            if action_player_idx == 0:  # Player
                if action_type == "DRAW" and card_name:
                    tracked_player_hand.append(card_name)
                elif action_type == "PLAY_LAND" and card_name:
                    if card_name in tracked_player_hand:
                        tracked_player_hand.remove(card_name)
                    # Track land on board
                    tracked_player_lands[card_name] = tracked_player_lands.get(card_name, 0) + 1
                elif action_type == "CAST" and card_name:
                    if card_name in tracked_player_hand:
                        tracked_player_hand.remove(card_name)
                    # Track creature on board if it's a creature
                    # (Rules engine doesn't log RESOLVE for basic creatures, so we track on CAST
                    # and remove on COUNTER if it gets countered)
                    if card_type_str.upper() == "CREATURE" or is_creature_card(card_name):
                        # Get base stats from registry
                        base_power, base_toughness = 0, 0
                        try:
                            reg = CardRegistry.get_instance()
                            cdata = reg.get(card_name)
                            base_power = cdata.power
                            base_toughness = cdata.toughness
                        except (KeyError, AttributeError):
                            pass
                        tracked_player_creatures.append(
                            {
                                "name": card_name,
                                "attached_tokens": [],
                                "card_id": details.get("card_id"),
                                "power_bonus": 0,
                                "toughness_bonus": 0,
                                "temp_power_bonus": 0,
                                "temp_toughness_bonus": 0,
                                "base_power": base_power,
                                "base_toughness": base_toughness,
                            }
                        )
                    else:
                        # Instants/Sorceries go to graveyard after resolving
                        tracked_player_graveyard.append(card_name)
                elif action_type == "DIES" and card_name:
                    tracked_player_graveyard.append(card_name)
                    dead_id = details.get("card_id")
                    # Remove from creatures (prefer card_id to handle duplicates)
                    if dead_id is not None:
                        for i, creature in enumerate(tracked_player_creatures):
                            if creature.get("card_id") == dead_id:
                                tracked_player_creatures.pop(i)
                                break
                    else:
                        for i, creature in enumerate(tracked_player_creatures):
                            if creature["name"] == card_name:
                                tracked_player_creatures.pop(i)
                                break
            else:  # Opponent
                if action_type == "DRAW" and card_name:
                    tracked_opponent_hand.append(card_name)
                elif action_type == "PLAY_LAND" and card_name:
                    if card_name in tracked_opponent_hand:
                        tracked_opponent_hand.remove(card_name)
                    # Track land on board
                    tracked_opponent_lands[card_name] = tracked_opponent_lands.get(card_name, 0) + 1
                elif action_type == "CAST" and card_name:
                    if card_name in tracked_opponent_hand:
                        tracked_opponent_hand.remove(card_name)
                    # Track creature on board if it's a creature
                    if card_type_str.upper() == "CREATURE" or is_creature_card(card_name):
                        # Get base stats from registry
                        base_power, base_toughness = 0, 0
                        try:
                            reg = CardRegistry.get_instance()
                            cdata = reg.get(card_name)
                            base_power = cdata.power
                            base_toughness = cdata.toughness
                        except (KeyError, AttributeError):
                            pass
                        tracked_opponent_creatures.append(
                            {
                                "name": card_name,
                                "attached_tokens": [],
                                "card_id": details.get("card_id"),
                                "power_bonus": 0,
                                "toughness_bonus": 0,
                                "temp_power_bonus": 0,
                                "temp_toughness_bonus": 0,
                                "base_power": base_power,
                                "base_toughness": base_toughness,
                            }
                        )
                    else:
                        # Instants/Sorceries go to graveyard after resolving
                        tracked_opponent_graveyard.append(card_name)
                elif action_type == "DIES" and card_name:
                    tracked_opponent_graveyard.append(card_name)
                    dead_id = details.get("card_id")
                    # Remove from creatures (prefer card_id to handle duplicates)
                    if dead_id is not None:
                        for i, creature in enumerate(tracked_opponent_creatures):
                            if creature.get("card_id") == dead_id:
                                tracked_opponent_creatures.pop(i)
                                break
                    else:
                        for i, creature in enumerate(tracked_opponent_creatures):
                            if creature["name"] == card_name:
                                tracked_opponent_creatures.pop(i)
                                break

            # Handle COUNTER actions - remove from battlefield (if creature was added on CAST)
            # and track to exile or graveyard
            if action_type == "COUNTER":
                countered_card = details.get("countered", "")
                destination = details.get("destination", "graveyard")
                # The countered spell was cast by the opponent of the counter caster
                if countered_card:
                    if (
                        action_player_idx == 0
                    ):  # Player cast the counter -> opponent's spell was countered
                        # Remove from opponent's creatures if it was a creature
                        for i, creature in enumerate(tracked_opponent_creatures):
                            if creature["name"] == countered_card:
                                tracked_opponent_creatures.pop(i)
                                break
                        # Add to exile or graveyard
                        if destination == "exile":
                            tracked_opponent_exile.append(countered_card)
                        else:
                            tracked_opponent_graveyard.append(countered_card)
                    else:  # Opponent cast the counter -> player's spell was countered
                        # Remove from player's creatures if it was a creature
                        for i, creature in enumerate(tracked_player_creatures):
                            if creature["name"] == countered_card:
                                tracked_player_creatures.pop(i)
                                break
                        # Add to exile or graveyard
                        if destination == "exile":
                            tracked_player_exile.append(countered_card)
                        else:
                            tracked_player_graveyard.append(countered_card)

            # Apply temporary bonuses from triggered abilities (e.g., Prowess)
            if action_type == "CAST":
                triggered_abilities = details.get("triggered_abilities", [])
                if triggered_abilities:
                    creatures = (
                        tracked_player_creatures
                        if action_player_idx == 0
                        else tracked_opponent_creatures
                    )
                    trigger_counts: dict[str, int] = {}
                    for trigger_desc in triggered_abilities:
                        match = re.search(r"^(.*) gets \+(\d+)/\+(\d+)", trigger_desc)
                        if not match:
                            continue
                        creature_name = match.group(1).strip()
                        power_bonus = int(match.group(2))
                        toughness_bonus = int(match.group(3))
                        occurrence = trigger_counts.get(creature_name, 0)
                        creature = _find_nth_matching(creatures, creature_name, occurrence)
                        if creature:
                            creature["temp_power_bonus"] = (
                                creature.get("temp_power_bonus", 0) + power_bonus
                            )
                            creature["temp_toughness_bonus"] = (
                                creature.get("temp_toughness_bonus", 0) + toughness_bonus
                            )
                        trigger_counts[creature_name] = occurrence + 1

            # Apply resolved pump effects to tracked stats
            if action_type == "RESOLVE":
                target = details.get("target", "")
                target_id = details.get("target_id")
                new_power = details.get("new_power")
                new_toughness = details.get("new_toughness")
                tokens = details.get("tokens", [])
                if target and target not in {"Opponent", "Player", "opponent", "player"}:
                    target_creature = (
                        _find_by_id(tracked_player_creatures, target_id)
                        or _find_by_id(tracked_opponent_creatures, target_id)
                        or _find_nth_matching(tracked_player_creatures, target, 0)
                        or _find_nth_matching(tracked_opponent_creatures, target, 0)
                    )
                    if target_creature:
                        # Apply attached tokens (e.g., Monster Role)
                        for token in tokens:
                            if token not in target_creature["attached_tokens"]:
                                target_creature["attached_tokens"].append(token)
                                if token == "Monster Role":
                                    target_creature["power_bonus"] = (
                                        target_creature.get("power_bonus", 0) + 1
                                    )
                                    target_creature["toughness_bonus"] = (
                                        target_creature.get("toughness_bonus", 0) + 1
                                    )
                        if new_power is not None and new_toughness is not None:
                            base_power = target_creature.get("base_power", 0) + target_creature.get(
                                "power_bonus", 0
                            )
                            base_toughness = target_creature.get(
                                "base_toughness", 0
                            ) + target_creature.get("toughness_bonus", 0)
                            target_creature["temp_power_bonus"] = max(
                                0, int(new_power) - base_power
                            )
                            target_creature["temp_toughness_bonus"] = max(
                                0, int(new_toughness) - base_toughness
                            )

            # Track discards from CLEANUP (hand size limit)
            if action_type == "CLEANUP":
                discarded = details.get("discarded_cards", [])
                for discard in discarded:
                    discard_player_idx = discard.get("player_idx", 0)
                    discard_card_name = discard.get("card_name", "")
                    if discard_player_idx == 0:  # Player discarded
                        if discard_card_name in tracked_player_hand:
                            tracked_player_hand.remove(discard_card_name)
                        tracked_player_graveyard.append(discard_card_name)
                    else:  # Opponent discarded
                        if discard_card_name in tracked_opponent_hand:
                            tracked_opponent_hand.remove(discard_card_name)
                        tracked_opponent_graveyard.append(discard_card_name)
                # Clear temporary buffs at end of turn
                for creature in tracked_player_creatures:
                    creature["temp_power_bonus"] = 0
                    creature["temp_toughness_bonus"] = 0
                for creature in tracked_opponent_creatures:
                    creature["temp_power_bonus"] = 0
                    creature["temp_toughness_bonus"] = 0

            # Track life changes from DAMAGE actions (combat damage)
            if action_type == "DAMAGE":
                events = details.get("events", [])
                trigger_damage = details.get("trigger_damage", [])
                for entry in trigger_damage:
                    damage = int(entry.get("damage", 0))
                    target_idx = entry.get("target_player_idx")
                    if damage <= 0 or target_idx is None:
                        continue
                    if target_idx == 0:
                        tracked_player_life -= damage
                    else:
                        tracked_opponent_life -= damage
                for event in events:
                    # Parse damage events - format is "CardName deals X damage"
                    if "death trigger" in str(event).lower():
                        continue
                    if "deals" in event.lower() and "damage" in event.lower():
                        # Extract damage amount using regex
                        damage_match = re.search(r"deals (\d+) damage", event.lower())
                        if damage_match:
                            damage = int(damage_match.group(1))
                            # Attacker deals damage to defender
                            if action_player_idx == 0:  # Player attacking
                                tracked_opponent_life -= damage
                            else:  # Opponent attacking
                                tracked_player_life -= damage

            # Track life changes from spell damage (burn spells)
            if action_type == "CAST":
                spell_damage = details.get("deals_damage", 0)
                target = details.get("target", "")
                if spell_damage and target and "opponent" in target.lower():
                    if action_player_idx == 0:  # Player casting at opponent
                        tracked_opponent_life -= spell_damage
                    else:  # Opponent casting at "opponent" (which is the player)
                        tracked_player_life -= spell_damage

            # Format the action (pass active player to show caster prefix when casting during
            # opponent's priority, e.g., instants cast in response)
            formatted = format_logged_action(action, active_player_idx=action_turn_player_idx)

            if formatted and "PASS" not in action.get("action_type", "").upper():
                if action_turn_player == "Player":
                    if phase_display not in player_phase_actions:
                        player_phase_actions[phase_display] = []
                    player_phase_actions[phase_display].append(formatted)
                    turn_player_actions.append((phase_display, formatted.split("\n")[0]))
                else:
                    if phase_display not in opponent_phase_actions:
                        opponent_phase_actions[phase_display] = []
                    opponent_phase_actions[phase_display].append(formatted)
                    turn_opponent_actions.append((phase_display, formatted.split("\n")[0]))

                # Record for HTML report (use display phase name, not raw enum)
                # Extract main description (first line) and effects (subsequent lines)
                formatted_lines = formatted.split("\n")
                main_desc = formatted_lines[0]
                # Effects are subsequent lines that start with "→" or contain TRIGGER/BUFF
                effect_lines = []
                for line in formatted_lines[1:]:
                    clean_line = _strip_rich_markup(line).strip()
                    if clean_line:
                        effect_lines.append(clean_line)

                recorder.record_action(
                    turn=display_turn,
                    phase=phase_display,  # Use mapped display name e.g. "Main 1"
                    player=action_player,
                    action_type=action.get("action_type", "ACTION"),
                    description=main_desc,
                    active_player_turn=action_turn_player,  # Track whose turn it is
                    effects=effect_lines if effect_lines else None,
                )

        return len(action_log)

    # Main game loop
    # Use < instead of <= because display_turn = (env_turn - first_turn) + 1
    max_display_turns = config.num_turns
    max_env_turn = first_turn_number + max_display_turns - 1
    while (
        not env.state.game_over and step_count < max_steps and env.state.turn_number <= max_env_turn
    ):
        step_count += 1

        phase = info.get("phase", "Unknown")

        # Skip mulligan phase display
        if phase == "Mulligan":
            action_mask = info.get("action_mask", np.ones(24))
            legal = np.where(action_mask > 0)[0]
            if len(legal) > 0:
                obs, _, terminated, _, info = env.step(legal[0])
                if terminated:
                    break
            else:
                break
            continue

        # Get legal actions
        action_mask = info.get("action_mask", np.ones(24))
        legal = np.where(action_mask > 0)[0]

        if len(legal) == 0:
            obs, _, terminated, _, info = env.step(0)
            if terminated:
                break
            continue

        # Select action (player is always idx 0 in our setup)
        action = agent.select_action(obs, action_mask, info)

        # Execute action - this may also execute opponent's turn internally
        obs, reward, terminated, truncated, info = env.step(action)

        # Process ALL new actions (including opponent's that happened internally)
        action_log = info.get("action_log", [])
        processed_action_count = process_new_actions(action_log, processed_action_count)

        if terminated:
            break

    # === FINISH LAST SEGMENTS ===
    # Capture snapshot before displaying final segment
    segment_end_snapshot = {
        "player_hand": tracked_player_hand.copy(),
        "opponent_hand": tracked_opponent_hand.copy(),
        "player_graveyard": tracked_player_graveyard.copy(),
        "opponent_graveyard": tracked_opponent_graveyard.copy(),
        "player_lands": tracked_player_lands.copy(),
        "opponent_lands": tracked_opponent_lands.copy(),
        "player_creatures": tracked_player_creatures.copy(),
        "opponent_creatures": tracked_opponent_creatures.copy(),
    }
    if current_display_player == "Player" and player_phase_actions:
        display_player_segment("Player", player_phase_actions, current_display_turn)
    elif current_display_player == "Opponent" and opponent_phase_actions:
        display_player_segment("Opponent", opponent_phase_actions, current_display_turn)

    if current_display_turn > 0:
        # Final turn summary using tracked state
        p_life_delta = tracked_player_life - turn_start_player_life
        o_life_delta = tracked_opponent_life - turn_start_opponent_life

        build_turn_summary_from_snapshot(
            turn=current_display_turn,
            player_actions=turn_player_actions,
            opponent_actions=turn_opponent_actions,
            player_life=tracked_player_life,
            opponent_life=tracked_opponent_life,
            player_life_change=p_life_delta,
            opponent_life_change=o_life_delta,
            tracked_player_hand=tracked_player_hand.copy(),
            tracked_opponent_hand=tracked_opponent_hand.copy(),
            tracked_player_lands=tracked_player_lands.copy(),
            tracked_opponent_lands=tracked_opponent_lands.copy(),
            tracked_player_creatures=tracked_player_creatures.copy(),
            tracked_opponent_creatures=tracked_opponent_creatures.copy(),
            tracked_player_graveyard=tracked_player_graveyard.copy(),
            tracked_opponent_graveyard=tracked_opponent_graveyard.copy(),
            tracked_player_exile=tracked_player_exile.copy(),
            tracked_opponent_exile=tracked_opponent_exile.copy(),
        )
        # Record final turn summary for HTML report
        # Counting logic: same as mid-game turn summary
        # Player's spells = on player's turn (no prefix) + on opp's turn ([Player] prefix)
        player_spells = sum(
            1 for _, d in turn_player_actions if "CAST" in d.upper() and "[Opponent]" not in d
        ) + sum(1 for _, d in turn_opponent_actions if "CAST" in d.upper() and "[Player]" in d)
        # Opponent's spells = on player's turn ([Opponent] prefix) + on opp's turn (no prefix)
        opponent_spells = sum(
            1 for _, d in turn_player_actions if "CAST" in d.upper() and "[Opponent]" in d
        ) + sum(1 for _, d in turn_opponent_actions if "CAST" in d.upper() and "[Player]" not in d)

        # Lands can only be played on your own turn
        player_lands = sum(
            1 for _, d in turn_player_actions if "PLAY_LAND" in d.upper() and "[Opponent]" not in d
        )
        opponent_lands = sum(
            1 for _, d in turn_opponent_actions if "PLAY_LAND" in d.upper() and "[Player]" not in d
        )

        # Creatures = CAST with ⚔️ on own turn
        player_creature_casts = sum(
            1
            for _, d in turn_player_actions
            if "CAST" in d.upper() and "⚔️" in d and "[Opponent]" not in d
        )
        opponent_creature_casts = sum(
            1
            for _, d in turn_opponent_actions
            if "CAST" in d.upper() and "⚔️" in d and "[Player]" not in d
        )
        # Subtract countered creatures
        player_creatures_countered = sum(
            1 for _, d in turn_player_actions if "COUNTER" in d.upper() and "⚔️" in d
        )
        opponent_creatures_countered = sum(
            1 for _, d in turn_opponent_actions if "COUNTER" in d.upper() and "⚔️" in d
        )
        player_creature_casts = max(0, player_creature_casts - player_creatures_countered)
        opponent_creature_casts = max(0, opponent_creature_casts - opponent_creatures_countered)

        # Cards drawn = on own turn (no prefix)
        player_draws = sum(
            1 for _, d in turn_player_actions if "DRAW" in d.upper() and "[Opponent]" not in d
        )
        opponent_draws = sum(
            1 for _, d in turn_opponent_actions if "DRAW" in d.upper() and "[Player]" not in d
        )

        recorder.record_turn_summary(
            turn=current_display_turn,
            player_damage=-o_life_delta if o_life_delta < 0 else 0,
            opponent_damage=-p_life_delta if p_life_delta < 0 else 0,
            player_spells=player_spells,
            opponent_spells=opponent_spells,
            player_lands=player_lands,
            opponent_lands=opponent_lands,
            player_creatures=player_creature_casts,
            opponent_creatures=opponent_creature_casts,
            player_draws=player_draws,
            opponent_draws=opponent_draws,
        )

        # Helper to calculate effective P/T from creature dict
        def _final_snapshot_creature_pt(c: dict) -> tuple[int, int]:
            """Calculate creature power/toughness from tracked dict."""
            base_p = c.get("base_power", 0) or 0
            base_t = c.get("base_toughness", 0) or 0
            bonus_p = c.get("power_bonus", 0) + c.get("temp_power_bonus", 0)
            bonus_t = c.get("toughness_bonus", 0) + c.get("temp_toughness_bonus", 0)
            return base_p + bonus_p, base_t + bonus_t

        def _final_is_instant_or_sorcery(name: str) -> bool:
            """Check if card is instant or sorcery."""
            try:
                card = CardRegistry.get_instance().get(name)
                return card.card_type in (CardType.INSTANT, CardType.SORCERY)
            except (KeyError, AttributeError):
                return False

        # Record final snapshot for HTML report
        recorder.record_snapshot(
            turn=current_display_turn,
            phase="End",
            active_player="Player",
            player_life=tracked_player_life,
            opponent_life=tracked_opponent_life,
            player_hand=[(n, "") for n in tracked_player_hand],
            opponent_hand=[(n, "") for n in tracked_opponent_hand],
            player_lands=tracked_player_lands.copy(),
            opponent_lands=tracked_opponent_lands.copy(),
            player_creatures=[
                {
                    "name": c["name"],
                    "power": _final_snapshot_creature_pt(c)[0],
                    "toughness": _final_snapshot_creature_pt(c)[1],
                    "tapped": c.get("tapped", False),
                    "attached_tokens": c.get("attached_tokens", []),
                }
                for c in tracked_player_creatures
            ],
            opponent_creatures=[
                {
                    "name": c["name"],
                    "power": _final_snapshot_creature_pt(c)[0],
                    "toughness": _final_snapshot_creature_pt(c)[1],
                    "tapped": c.get("tapped", False),
                    "attached_tokens": c.get("attached_tokens", []),
                }
                for c in tracked_opponent_creatures
            ],
            player_graveyard=[(n, "card") for n in tracked_player_graveyard],
            opponent_graveyard=[(n, "card") for n in tracked_opponent_graveyard],
            player_exile=tracked_player_exile.copy(),
            opponent_exile=tracked_opponent_exile.copy(),
            player_graveyard_instant_sorcery_count=sum(
                1 for n in tracked_player_graveyard if _final_is_instant_or_sorcery(n)
            ),
            opponent_graveyard_instant_sorcery_count=sum(
                1 for n in tracked_opponent_graveyard if _final_is_instant_or_sorcery(n)
            ),
        )

    # === GAME OVER ===
    winner = info.get("game_result", "draw")
    if winner == "win":
        recorder.set_winner("Player")
    elif winner == "loss":
        recorder.set_winner("Opponent")
    else:
        recorder.set_winner("Draw")

    print_divider("Game Over")

    if winner == "win":
        console.print(
            Panel(
                "[bold green]🎉 VICTORY! 🎉[/bold green]\n\nPlayer wins the game!",
                border_style="green",
                width=console.width,
            )
        )
    elif winner == "loss":
        console.print(
            Panel(
                "[bold red]💀 DEFEAT 💀[/bold red]\n\nOpponent wins the game!",
                border_style="red",
                width=console.width,
            )
        )
    else:
        console.print(
            Panel(
                "[bold yellow]⏱️ GAME COMPLETE ⏱️[/bold yellow]\n\n"
                "Turn limit reached - Game ended with no winner.",
                border_style="yellow",
                width=console.width,
            )
        )

    time.sleep(delays["turn"])
    print_final_game_state(env)

    return {
        "winner": "Player" if winner == "win" else "Opponent" if winner == "loss" else "Draw",
        "player_life": env.state.players[0].life,
        "opponent_life": env.state.players[1].life,
        "turns_played": info.get("turn", 0),
        "recorder": recorder,
    }


# =============================================================================
# Report Generation
# =============================================================================


def save_report(result: dict[str, Any], config: GameplayConfig) -> Path | None:
    """Save game replay as HTML report."""
    recorder = result.get("recorder")
    if not recorder:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    game_dir = Path("results/gameplay") / f"{config.player_agent}_{timestamp}"
    game_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{config.player_deck}_vs_{config.opponent_deck}.html"
    output_path = game_dir / filename

    try:
        html = generate_html_report(recorder.get_replay())
        output_path.write_text(html)
        console.print(f"\n[green]Report saved to {output_path}[/green]")
        return output_path
    except Exception as e:
        console.print(f"[red]Error saving report: {e}[/red]")
        return None


# =============================================================================
# Entry Point
# =============================================================================


def main() -> int:
    """Main entry point."""
    console.clear()
    print_logo()

    console.print("\n[bold cyan]MTG-Causal-RL Gameplay[/bold cyan]")
    console.print("[dim]Play a game of Magic: The Gathering with AI agents.[/dim]\n")

    config = prompt_gameplay_config()

    console.clear()
    print_logo()

    result = run_game(config)

    if config.save_report:
        save_report(result, config)

    print_divider("Session Complete")

    return 0


if __name__ == "__main__":
    sys.exit(main())

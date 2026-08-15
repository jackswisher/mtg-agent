"""HTML gameplay report generator for MTG-Causal-RL.

Creates interactive, visually appealing HTML reports of game sessions
for post-hoc analysis and visualization.
"""

from __future__ import annotations

import contextlib
import json
import re
import typing as tp
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from mtg.env.card_definitions import CardType, get_card


def is_creature(card: tp.Any) -> bool:
    """Check if a card is a creature."""
    if card is None:
        return False
    if hasattr(card, "card_type"):
        return card.card_type == CardType.CREATURE
    return False


def _is_land_card(card_name: str) -> bool:
    """Check if a card is a land."""
    with contextlib.suppress(Exception):
        card = get_card(card_name)
        if card:
            return card.card_type == CardType.LAND
    return False


def _get_card_text(card_name: str) -> str | None:
    """Get rules text for a card (mirrors ``get_card_text`` in ``run_gameplay``)."""
    with contextlib.suppress(Exception):
        card = get_card(card_name)
        if card and hasattr(card, "rules_text") and card.rules_text:
            return card.rules_text
    return None


# Mana color icons
MANA_ICONS = {
    "W": "⚪",
    "U": "🔵",
    "B": "⚫",
    "R": "🔴",
    "G": "🟢",
    "C": "◇",
}


def format_mana_cost(mana_cost: str) -> str:
    """Format a mana cost string with icons.

    Args:
        mana_cost: Mana cost like "2RR" or "1WU".

    Returns:
        Formatted string with icons.

    """
    if not mana_cost:
        return ""

    result = []
    i = 0
    while i < len(mana_cost):
        char = mana_cost[i]
        if char.isdigit():
            # Colorless mana
            num = ""
            while i < len(mana_cost) and mana_cost[i].isdigit():
                num += mana_cost[i]
                i += 1
            result.append(f"({num})")
            continue
        elif char.upper() in MANA_ICONS:
            result.append(MANA_ICONS[char.upper()])
        i += 1

    return "".join(result)


@dataclass
class CardInfo:
    """Information about a card."""

    name: str
    mana_cost: str = ""
    card_type: str = ""


@dataclass
class GameAction:
    """Represents a single game action."""

    turn: int
    phase: str
    player: str
    action_type: str
    description: str
    active_player_turn: str = ""  # Whose turn is it when this action happens
    effects: list[str] = field(default_factory=list)
    state_changes: dict[str, tp.Any] = field(default_factory=dict)


@dataclass
class ReplayStateSnapshot:
    """Snapshot of game state at a point in time for HTML replay."""

    turn: int
    phase: str
    active_player: str
    player_life: int
    opponent_life: int
    player_hand: list[CardInfo]  # Player's actual cards in hand
    opponent_hand: list[CardInfo]  # Opponent's actual cards (shown hidden in report)
    opponent_hand_size: int  # Keep for backward compatibility
    player_lands: dict[str, int]  # Lands ON BOARD by type: {"Mountain": 2}
    opponent_lands: dict[str, int]  # Lands ON BOARD by type
    player_mana: dict[str, int]  # Mana by color: {"R": 2, "U": 1}
    opponent_mana: dict[str, int]
    player_creatures: list[dict[str, tp.Any]]
    opponent_creatures: list[dict[str, tp.Any]]
    player_tokens: list[dict[str, tp.Any]]  # Added tokens
    opponent_tokens: list[dict[str, tp.Any]]
    player_graveyard: list[dict[str, str]]
    opponent_graveyard: list[dict[str, str]]
    player_exile: list[str] = field(default_factory=list)  # Exiled cards
    opponent_exile: list[str] = field(default_factory=list)
    board_power: int = 0
    opponent_power: int = 0
    # Graveyard instant/sorcery counts for Haughty Djinn power calculation
    player_graveyard_instant_sorcery_count: int = 0
    opponent_graveyard_instant_sorcery_count: int = 0


@dataclass
class MulliganInfo:
    """Mulligan information for a player.

    In MTG mulligan:
    1. See opening_hand (7 cards)
    2. If mulligan: shuffle back, draw new 7 cards (mulligan_hand)
    3. Keep new hand but return N cards to bottom (returned_cards)
    4. Final hand = new_hand (6 cards for 1 mulligan)
    """

    opening_hand: list[CardInfo]  # Initial 7 cards
    kept: bool  # Whether player kept immediately (no mulligan)
    mulligan_hand: list[CardInfo] = field(default_factory=list)  # The NEW 7 cards after mulligan
    returned_cards: list[CardInfo] = field(default_factory=list)  # Card(s) put on bottom
    new_hand: list[CardInfo] = field(default_factory=list)  # Final kept hand (6 for 1 mull)
    mulligans_taken: int = 0


@dataclass
class InitialGameState:
    """Initial game setup information."""

    player_on_play: bool
    player_mulligan: MulliganInfo
    opponent_mulligan: MulliganInfo


@dataclass
class TurnSummary:
    """Summary statistics for a turn."""

    turn: int
    player_damage_dealt: int = 0
    opponent_damage_dealt: int = 0
    player_spells_cast: int = 0
    opponent_spells_cast: int = 0
    player_lands_played: int = 0
    opponent_lands_played: int = 0
    player_creatures_played: int = 0
    opponent_creatures_played: int = 0
    player_cards_drawn: int = 0
    opponent_cards_drawn: int = 0


@dataclass
class GameReplay:
    """Complete game replay data."""

    game_id: str
    timestamp: str
    player_deck: str
    opponent_deck: str
    player_agent: str
    opponent_agent: str
    player_on_play: bool
    winner: str
    total_turns: int
    initial_state: InitialGameState | None = None
    actions: list[GameAction] = field(default_factory=list)
    snapshots: list[ReplayStateSnapshot] = field(default_factory=list)
    turn_summaries: list[TurnSummary] = field(default_factory=list)
    metadata: dict[str, tp.Any] = field(default_factory=dict)


class GameRecorder:
    """Records game events for replay and HTML export."""

    def __init__(
        self,
        game_id: str | None = None,
        player_deck: str = "unknown",
        opponent_deck: str = "unknown",
        player_agent: str = "unknown",
        opponent_agent: str = "unknown",
    ) -> None:
        """Initialize game recorder.

        Args:
            game_id: Unique identifier for the game.
            player_deck: Name of player's deck archetype.
            opponent_deck: Name of opponent's deck archetype.
            player_agent: Name of player's agent.
            opponent_agent: Name of opponent's agent.

        """
        self.game_id = game_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.player_deck = player_deck
        self.opponent_deck = opponent_deck
        self.player_agent = player_agent
        self.opponent_agent = opponent_agent
        self.player_on_play = True
        self.winner = ""
        self.total_turns = 0
        self.initial_state: InitialGameState | None = None
        self.actions: list[GameAction] = []
        self.snapshots: list[ReplayStateSnapshot] = []
        self.turn_summaries: list[TurnSummary] = []
        self.metadata: dict[str, tp.Any] = {}
        self._start_time = datetime.now()

    def set_player_on_play(self, player_on_play: bool) -> None:
        """Set whether player is on the play."""
        self.player_on_play = player_on_play

    def record_initial_state(
        self,
        player_on_play: bool,
        player_opening_hand: list[tuple[str, str]],  # Initial 7 cards
        player_kept: bool,
        player_mulligan_hand: list[tuple[str, str]],  # New 7 cards after mulligan
        player_returned_cards: list[tuple[str, str]],  # Cards put on bottom
        player_kept_hand: list[tuple[str, str]],  # Final kept hand
        opponent_opening_hand: list[tuple[str, str]],
        opponent_kept: bool,
        player_mulligans: int = 0,
        opponent_mulligans: int = 0,
    ) -> None:
        """Record initial game state including mulligan info.

        Args:
            player_on_play: Whether player is on the play.
            player_opening_hand: Player's initial 7 cards.
            player_kept: Whether player kept immediately.
            player_mulligan_hand: Player's new 7 cards after mulligan (empty if kept).
            player_returned_cards: Cards returned to bottom of library.
            player_kept_hand: Player's final kept hand.
            opponent_opening_hand: Opponent's initial cards.
            opponent_kept: Whether opponent kept.
            player_mulligans: Number of mulligans taken by player.
            opponent_mulligans: Number of mulligans taken by opponent.

        """
        self.player_on_play = player_on_play
        self.initial_state = InitialGameState(
            player_on_play=player_on_play,
            player_mulligan=MulliganInfo(
                opening_hand=[CardInfo(name=n, mana_cost=m) for n, m in player_opening_hand],
                kept=player_kept,
                mulligan_hand=[CardInfo(name=n, mana_cost=m) for n, m in player_mulligan_hand],
                returned_cards=[CardInfo(name=n, mana_cost=m) for n, m in player_returned_cards],
                new_hand=[CardInfo(name=n, mana_cost=m) for n, m in player_kept_hand],
                mulligans_taken=player_mulligans,
            ),
            opponent_mulligan=MulliganInfo(
                opening_hand=[CardInfo(name=n, mana_cost=m) for n, m in opponent_opening_hand],
                kept=opponent_kept,
                new_hand=[],
                mulligans_taken=opponent_mulligans,
            ),
        )

    def record_action(
        self,
        turn: int,
        phase: str,
        player: str,
        action_type: str,
        description: str,
        active_player_turn: str | None = None,
        effects: list[str] | None = None,
        state_changes: dict[str, tp.Any] | None = None,
    ) -> None:
        """Record a game action.

        Args:
            turn: Current turn number.
            phase: Current game phase.
            player: Which player took the action.
            action_type: Type of action (CAST, ATTACK, etc.).
            description: Human-readable description.
            active_player_turn: Whose turn it is (for instant/block placement).
            effects: Optional list of triggered effects.
            state_changes: Optional state changes from this action.

        """
        self.actions.append(
            GameAction(
                turn=turn,
                phase=phase,
                player=player,
                action_type=action_type,
                description=description,
                active_player_turn=active_player_turn or player,
                effects=effects or [],
                state_changes=state_changes or {},
            )
        )
        self.total_turns = max(self.total_turns, turn)

    def record_snapshot(
        self,
        turn: int,
        phase: str,
        active_player: str,
        player_life: int,
        opponent_life: int,
        player_hand: list[tuple[str, str]],  # (name, mana_cost) tuples
        opponent_hand: list[tuple[str, str]],  # Opponent's actual cards
        player_lands: dict[str, int],  # BOARD lands by type
        opponent_lands: dict[str, int],  # BOARD lands by type
        player_mana: dict[str, int] | None = None,
        opponent_mana: dict[str, int] | None = None,
        player_creatures: list | None = None,
        opponent_creatures: list | None = None,
        player_tokens: list[tuple[str, int, int]] | None = None,
        opponent_tokens: list[tuple[str, int, int]] | None = None,
        player_graveyard: list[tuple[str, str]] | None = None,
        opponent_graveyard: list[tuple[str, str]] | None = None,
        player_exile: list[str] | None = None,
        opponent_exile: list[str] | None = None,
        board_power: int = 0,
        opponent_power: int = 0,
        player_graveyard_instant_sorcery_count: int = 0,
        opponent_graveyard_instant_sorcery_count: int = 0,
    ) -> None:
        """Record a game state snapshot.

        Args:
            turn: Current turn number.
            phase: Current game phase.
            active_player: Active player name.
            player_life: Player's life total.
            opponent_life: Opponent's life total.
            player_hand: Cards in player's hand as (name, mana_cost) tuples.
            opponent_hand: Cards in opponent's hand as (name, mana_cost) tuples.
            player_lands: Player's lands on board by type (dict).
            opponent_lands: Opponent's lands on board by type (dict).
            player_mana: Available mana by color.
            opponent_mana: Opponent's available mana by color.
            player_creatures: List of creatures.
            opponent_creatures: List of opponent creatures.
            player_tokens: List of (name, power, toughness).
            opponent_tokens: List of (name, power, toughness).
            player_graveyard: List of (card_name, card_type).
            opponent_graveyard: List of (card_name, card_type).
            player_exile: List of exiled card names.
            opponent_exile: List of opponent's exiled card names.
            board_power: Total creature power for player.
            opponent_power: Total creature power for opponent.
            player_graveyard_instant_sorcery_count: Count of instants/sorceries in player graveyard.
            opponent_graveyard_instant_sorcery_count: Count of instants/sorceries in opponent graveyard.

        """
        # Convert hand to list of CardInfo
        hand_list = [CardInfo(name=n, mana_cost=m) for n, m in player_hand]

        # Handle creature format flexibility
        def parse_creatures(creatures: list | None) -> list[dict]:
            if not creatures:
                return []
            result = []
            for c in creatures:
                if isinstance(c, dict):
                    result.append(c)
                elif isinstance(c, tuple):
                    if len(c) == 4:  # (name, p, t, tapped)
                        n, p, t, tap = c
                        result.append(
                            {"name": n, "power": p, "toughness": t, "tapped": tap, "mana_cost": ""}
                        )
                    elif len(c) == 5:  # (name, p, t, tapped, mana_cost)
                        n, p, t, tap, mc = c
                        result.append(
                            {"name": n, "power": p, "toughness": t, "tapped": tap, "mana_cost": mc}
                        )
                    else:
                        # Best effort
                        result.append(
                            {
                                "name": str(c[0]) if c else "Unknown",
                                "power": 0,
                                "toughness": 0,
                                "tapped": False,
                                "mana_cost": "",
                            }
                        )
            return result

        # Handle graveyard format
        def parse_graveyard(gy: list | None) -> list[dict]:
            if not gy:
                return []
            result = []
            for item in gy:
                if isinstance(item, dict):
                    result.append(item)
                elif isinstance(item, tuple) and len(item) >= 2:
                    result.append({"name": item[0], "type": item[1]})
            return result

        # Convert opponent hand to CardInfo list
        opp_hand_list = [CardInfo(name=n, mana_cost=m) for n, m in opponent_hand]

        self.snapshots.append(
            ReplayStateSnapshot(
                turn=turn,
                phase=phase,
                active_player=active_player,
                player_life=player_life,
                opponent_life=opponent_life,
                player_hand=list(hand_list),
                opponent_hand=list(opp_hand_list),
                opponent_hand_size=len(opp_hand_list),
                player_lands=dict(player_lands),  # Copy dict
                opponent_lands=dict(opponent_lands),  # Copy dict
                player_mana=dict(player_mana) if player_mana else {},
                opponent_mana=dict(opponent_mana) if opponent_mana else {},
                player_creatures=parse_creatures(player_creatures),
                opponent_creatures=parse_creatures(opponent_creatures),
                player_tokens=[
                    {"name": n, "power": p, "toughness": t} for n, p, t in (player_tokens or [])
                ],
                opponent_tokens=[
                    {"name": n, "power": p, "toughness": t} for n, p, t in (opponent_tokens or [])
                ],
                player_graveyard=parse_graveyard(player_graveyard),
                opponent_graveyard=parse_graveyard(opponent_graveyard),
                player_exile=list(player_exile) if player_exile else [],
                opponent_exile=list(opponent_exile) if opponent_exile else [],
                board_power=board_power,
                opponent_power=opponent_power,
                player_graveyard_instant_sorcery_count=player_graveyard_instant_sorcery_count,
                opponent_graveyard_instant_sorcery_count=opponent_graveyard_instant_sorcery_count,
            )
        )

    def record_turn_summary(
        self,
        turn: int,
        player_damage: int = 0,
        opponent_damage: int = 0,
        player_spells: int = 0,
        opponent_spells: int = 0,
        player_lands: int = 0,
        opponent_lands: int = 0,
        player_creatures: int = 0,
        opponent_creatures: int = 0,
        player_draws: int = 0,
        opponent_draws: int = 0,
    ) -> None:
        """Record turn summary statistics.

        Args:
            turn: Turn number.
            player_damage: Damage dealt by player this turn.
            opponent_damage: Damage dealt by opponent this turn.
            player_spells: Spells cast by player this turn (excludes lands).
            opponent_spells: Spells cast by opponent this turn (excludes lands).
            player_lands: Lands played by player this turn.
            opponent_lands: Lands played by opponent this turn.
            player_creatures: Creatures played by player this turn.
            opponent_creatures: Creatures played by opponent this turn.
            player_draws: Cards drawn by player this turn.
            opponent_draws: Cards drawn by opponent this turn.

        """
        self.turn_summaries.append(
            TurnSummary(
                turn=turn,
                player_damage_dealt=player_damage,
                opponent_damage_dealt=opponent_damage,
                player_spells_cast=player_spells,
                opponent_spells_cast=opponent_spells,
                player_lands_played=player_lands,
                opponent_lands_played=opponent_lands,
                player_creatures_played=player_creatures,
                opponent_creatures_played=opponent_creatures,
                player_cards_drawn=player_draws,
                opponent_cards_drawn=opponent_draws,
            )
        )

    def set_winner(self, winner: str) -> None:
        """Set the game winner."""
        self.winner = winner

    def add_metadata(self, key: str, value: tp.Any) -> None:
        """Add metadata to the replay."""
        self.metadata[key] = value

    def get_replay(self) -> GameReplay:
        """Get the complete game replay data."""
        return GameReplay(
            game_id=self.game_id,
            timestamp=self._start_time.isoformat(),
            player_deck=self.player_deck,
            opponent_deck=self.opponent_deck,
            player_agent=self.player_agent,
            opponent_agent=self.opponent_agent,
            player_on_play=self.player_on_play,
            winner=self.winner,
            total_turns=self.total_turns,
            initial_state=self.initial_state,
            actions=self.actions,
            snapshots=self.snapshots,
            turn_summaries=self.turn_summaries,
            metadata=self.metadata,
        )


def snapshot_from_env(env: tp.Any) -> dict[str, tp.Any]:
    """Build a ``record_snapshot``-ready dict by reading directly from ``env.state``.

    This produces the *same* rich data the gameplay CLI uses (creatures with
    P/T, attached tokens, lands as dicts, exile zones, instant/sorcery
    graveyard counts) so that reports from training and evaluation match the
    reports from the gameplay workflow.

    Args:
        env: An ``MTGEnv`` instance (unwrapped).

    Returns:
        kwargs dict suitable for ``GameRecorder.record_snapshot(**result)``.

    """
    state = env.state
    if state is None:
        return {}

    rules_engine = env.rules_engine
    player = state.players[0]
    opponent = state.players[1]

    # --- Hands (name, mana_cost) ------------------------------------------
    player_hand = [(c.name, c.mana_cost.to_text()) for c in player.hand]
    opponent_hand = [(c.name, c.mana_cost.to_text()) for c in opponent.hand]

    # --- Lands as dict[str, int] ------------------------------------------
    def _land_dict(ps: tp.Any) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in ps.battlefield:
            if c.produces_mana:
                counts[c.name] = counts.get(c.name, 0) + 1
        return counts

    player_lands = _land_dict(player)
    opponent_lands = _land_dict(opponent)

    # --- Creatures as rich dicts ------------------------------------------
    def _creature_list(ps: tp.Any) -> list[dict[str, tp.Any]]:
        out: list[dict[str, tp.Any]] = []
        for c in ps.battlefield:
            is_creature = c.card_type == CardType.CREATURE or c.card_id in ps.activated_creatures
            if not is_creature:
                continue
            power, toughness = rules_engine._get_effective_power_toughness(c, ps)
            out.append(
                {
                    "name": c.name,
                    "power": power,
                    "toughness": toughness,
                    "tapped": c.card_id in ps.tapped_permanents,
                    "mana_cost": c.mana_cost.to_text(),
                    "attached_tokens": list(c.attached_tokens),
                }
            )
        return out

    player_creatures = _creature_list(player)
    opponent_creatures = _creature_list(opponent)

    # --- Graveyard as [(name, card_type_str)] -----------------------------
    player_graveyard = [(c.name, c.card_type.value) for c in player.graveyard]
    opponent_graveyard = [(c.name, c.card_type.value) for c in opponent.graveyard]

    # --- Exile as [name, ...] ---------------------------------------------
    player_exile = [c.name for c in player.exile]
    opponent_exile = [c.name for c in opponent.exile]

    # --- Instant/sorcery counts in graveyard ------------------------------
    def _is_count(gy: list[tp.Any]) -> int:
        return sum(1 for c in gy if c.card_type in (CardType.INSTANT, CardType.SORCERY))

    p_is_count = _is_count(player.graveyard)
    o_is_count = _is_count(opponent.graveyard)

    # --- Board power ------------------------------------------------------
    board_power = sum(
        rules_engine._get_effective_power_toughness(c, player)[0]
        for c in player.battlefield
        if c.card_type == CardType.CREATURE or c.card_id in player.activated_creatures
    )
    opponent_power = sum(
        rules_engine._get_effective_power_toughness(c, opponent)[0]
        for c in opponent.battlefield
        if c.card_type == CardType.CREATURE or c.card_id in opponent.activated_creatures
    )

    # --- Phase / turn info ------------------------------------------------
    phase_names = {
        "MULLIGAN": "Mulligan",
        "UNTAP": "Untap",
        "UPKEEP": "Upkeep",
        "DRAW": "Draw",
        "MAIN_1": "Main 1",
        "BEGIN_COMBAT": "Begin Combat",
        "DECLARE_ATTACKERS": "Declare Attackers",
        "DECLARE_BLOCKERS": "Declare Blockers",
        "COMBAT_DAMAGE": "Combat Damage",
        "END_COMBAT": "End Combat",
        "MAIN_2": "Main 2",
        "END_STEP": "End Step",
        "CLEANUP": "Cleanup",
    }
    phase = phase_names.get(state.phase.name, state.phase.name)
    active_player = "Player" if state.active_player == 0 else "Opponent"

    return {
        "turn": state.turn_number,
        "phase": phase,
        "active_player": active_player,
        "player_life": player.life,
        "opponent_life": opponent.life,
        "player_hand": player_hand,
        "opponent_hand": opponent_hand,
        "player_lands": player_lands,
        "opponent_lands": opponent_lands,
        "player_creatures": player_creatures,
        "opponent_creatures": opponent_creatures,
        "player_graveyard": player_graveyard,
        "opponent_graveyard": opponent_graveyard,
        "player_exile": player_exile,
        "opponent_exile": opponent_exile,
        "board_power": board_power,
        "opponent_power": opponent_power,
        "player_graveyard_instant_sorcery_count": p_is_count,
        "opponent_graveyard_instant_sorcery_count": o_is_count,
    }


_PHASE_DISPLAY_NAMES: dict[str, str] = {
    "MULLIGAN": "Mulligan",
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
    "GAME_OVER": "End",
}


def _format_card_for_report(
    card_name: str,
    power: int | None = None,
    toughness: int | None = None,
    attached_tokens: list[str] | None = None,
) -> str:
    """Format a card name with its stats for the HTML report.

    Mirrors ``format_card_display`` from ``run_gameplay.py`` so that
    training reports and gameplay reports produce identical card strings.

    Standard format:
    - Creatures: ``"Name - P/T (mana, ⚔️)"``
    - Non-creatures: ``"Name (mana, symbol)"``
    - Lands: ``"Name (🌍)"``

    Args:
        card_name: Name of the card to format.
        power: Override power (e.g. current P/T from game state).
        toughness: Override toughness.
        attached_tokens: List of attached token names.

    Returns:
        Formatted string matching gameplay CLI output.

    """
    try:
        card = get_card(card_name)
        if card:
            mana_str = format_mana_cost(card.mana_cost.to_text()) if card.mana_cost else ""
            type_symbol = CARD_TYPE_SYMBOLS.get(card.card_type.name.lower(), "")

            attachment_str = ""
            if attached_tokens:
                token_strs = [f"🎴{token}" for token in attached_tokens]
                attachment_str = f" [{', '.join(token_strs)}]"

            if is_creature(card):
                p = power if power is not None else card.power
                t = toughness if toughness is not None else card.toughness
                if mana_str:
                    return f"{card_name} - {p}/{t} ({mana_str}, {type_symbol}){attachment_str}"
                return f"{card_name} - {p}/{t} ({type_symbol}){attachment_str}"

            if mana_str:
                return f"{card_name} ({mana_str}, {type_symbol})"
            return f"{card_name} ({type_symbol})"
    except Exception:
        pass
    if power is not None and toughness is not None:
        return f"{card_name} - {power}/{toughness} (🎴)"
    return card_name


def actions_from_env(env: tp.Any, since_idx: int = 0) -> list[dict[str, tp.Any]]:
    """Extract recorded game actions from ``env.state.action_log`` since a given index.

    Produces rich descriptions that match the quality of the gameplay CLI output,
    including card stats, targets, triggered abilities, and combat details.

    Each returned dict can be passed directly to
    ``GameRecorder.record_action(**item)``.

    Args:
        env: An ``MTGEnv`` instance (unwrapped).
        since_idx: Index into ``state.action_log`` from which to start.

    Returns:
        List of kwargs dicts for ``record_action``.

    """
    state = env.state
    if state is None:
        return []

    out: list[dict[str, tp.Any]] = []
    for a in state.action_log[since_idx:]:
        action_type = (
            a.action_type.upper() if isinstance(a.action_type, str) else str(a.action_type)
        )

        # Normalise action type
        at = action_type
        al = action_type.lower()
        if "cast_flashback" in al or "cast" in al:
            at = "CAST"
        elif "play" in al and "land" in al:
            at = "PLAY_LAND"
        elif "attack" in al:
            at = "ATTACK"
        elif "block" in al:
            at = "BLOCK"
        elif "activate" in al:
            at = "ACTIVATE"
        elif "pass" in al:
            at = "PASS"

        # Skip PASS actions (run_gameplay also filters these)
        if at == "PASS":
            continue

        player_label = "Player" if a.player == 0 else "Opponent"
        active_label = "Player" if a.active_player == 0 else "Opponent"

        desc = ""
        effects: list[str] = []

        # ── CAST ──────────────────────────────────────────────────────
        if at == "CAST" and a.card_name:
            card_disp = _format_card_for_report(a.card_name)

            # Target information
            target = a.details.get("target", "")
            target_kind = a.details.get("target_kind", "")
            target_str = ""
            if target:
                if target_kind == "player":
                    target_label = "Player" if a.details.get("target_id") == 0 else "Opponent"
                    target_str = f" targeting {target_label}"
                elif target_kind == "creature":
                    target_owner = a.details.get("target_owner")
                    if target_owner is not None and target_owner != a.player:
                        owner_prefix = "Opponent's " if a.player == 0 else "Player's "
                    else:
                        owner_prefix = ""
                    target_power = a.details.get("target_power")
                    target_toughness = a.details.get("target_toughness")
                    target_tokens = a.details.get("target_tokens", [])
                    target_disp = _format_card_for_report(
                        target,
                        power=target_power,
                        toughness=target_toughness,
                        attached_tokens=target_tokens,
                    )
                    target_str = f" targeting {owner_prefix}{target_disp}"
                else:
                    target_str = f" targeting {target}"

            desc = f"Cast {card_disp}{target_str}"

            # Card rules text (gameplay shows for ALL cards, not just non-creatures)
            if not _is_land_card(a.card_name):
                card_text = _get_card_text(a.card_name)
                if card_text:
                    effects.append(f"→ {card_text}")

            # Spell damage
            deals_damage = a.details.get("deals_damage", 0)
            if deals_damage:
                if target:
                    dmg_label = (
                        target
                        if target_kind != "player"
                        else ("Player" if a.details.get("target_id") == 0 else "Opponent")
                    )
                    effects.append(f"→ Deals {deals_damage} damage to {dmg_label}")
                else:
                    effects.append(f"→ Deals {deals_damage} damage")

            # Triggered abilities (prowess, etc.)
            for trigger_desc in a.details.get("triggered_abilities", []):
                effects.append(f"→ TRIGGER: {trigger_desc}")

        # ── PLAY LAND ─────────────────────────────────────────────────
        elif at == "PLAY_LAND" and a.card_name:
            all_lands = a.details.get("all_lands", {})
            enters_tapped = a.details.get("enters_tapped", False)
            tapped_str = " (enters tapped)" if enters_tapped else ""
            card_disp = f"{a.card_name} (🌍)"
            if all_lands:
                lands_str = ", ".join(f"{n} (🌍) x{c}" for n, c in all_lands.items())
                desc = f"PLAY_LAND: {card_disp}{tapped_str} (Board: {lands_str})"
            else:
                land_count = a.details.get("land_count", 1)
                desc = f"PLAY_LAND: {card_disp}{tapped_str} (Board: {card_disp} x{land_count})"

        # ── DRAW ──────────────────────────────────────────────────────
        elif at == "DRAW" and a.card_name:
            desc = f"Draw {_format_card_for_report(a.card_name)}"
            # Show rules text for non-land cards (matching gameplay)
            if not _is_land_card(a.card_name):
                card_text = _get_card_text(a.card_name)
                if card_text:
                    effects.append(f"→ {card_text}")

        # ── DRAW SELECTION (Memory Deluge etc.) ───────────────────────
        elif at == "DRAW_SELECTION":
            cards_drawn = a.details.get("cards_drawn", [])
            count = a.details.get("count", 0)
            card_str = _format_card_for_report(a.card_name) if a.card_name else "Effect"
            if cards_drawn:
                drawn_strs = ", ".join(_format_card_for_report(c) for c in cards_drawn)
                desc = f"{card_str} → Drew {count} card(s): {drawn_strs}"
            else:
                desc = f"{card_str} → Drew {count} card(s)"

        # ── ATTACK ────────────────────────────────────────────────────
        elif at == "ATTACK":
            attacker_data = a.details.get("attacker_data", [])
            if attacker_data:
                formatted_attackers = []
                for atk in attacker_data:
                    name = atk.get("name", "")
                    power = atk.get("power")
                    toughness = atk.get("toughness")
                    tokens = atk.get("tokens", [])
                    formatted_attackers.append(
                        _format_card_for_report(
                            name,
                            power=power,
                            toughness=toughness,
                            attached_tokens=tokens,
                        )
                    )
                desc = f"ATTACK: Attacking with {', '.join(formatted_attackers)}"
            else:
                attackers = a.details.get("attackers", [])
                if attackers:
                    formatted_attackers = [_format_card_for_report(a_name) for a_name in attackers]
                    desc = f"ATTACK: Attacking with {', '.join(formatted_attackers)}"
                elif a.card_name:
                    desc = f"ATTACK: Attacking with {_format_card_for_report(a.card_name)}"
                else:
                    desc = "ATTACK: No attackers"

            # Attack triggers
            for trigger_desc in a.details.get("triggered_abilities", []):
                effects.append(f"→ TRIGGER: {trigger_desc}")

        # ── BLOCK ─────────────────────────────────────────────────────
        elif at == "BLOCK":
            block_data = a.details.get("block_data", [])
            if block_data:
                block_strs = []
                for entry in block_data:
                    blocker = entry.get("blocker", "")
                    attacker = entry.get("attacker", "")
                    bp = entry.get("blocker_power")
                    bt = entry.get("blocker_toughness")
                    ap = entry.get("attacker_power")
                    at_ = entry.get("attacker_toughness")
                    b_disp = _format_card_for_report(blocker, power=bp, toughness=bt)
                    a_disp = _format_card_for_report(attacker, power=ap, toughness=at_)
                    block_strs.append(f"{b_disp} blocks {a_disp}")
                desc = "BLOCK: " + ", ".join(block_strs)
            else:
                blocks = a.details.get("blocks", [])
                if blocks:
                    block_strs = [
                        f"{_format_card_for_report(b[0])} blocks {_format_card_for_report(b[1])}"
                        for b in blocks
                    ]
                    desc = "BLOCK: " + ", ".join(block_strs)
                else:
                    desc = (
                        f"BLOCK: {_format_card_for_report(a.card_name)} blocks"
                        if a.card_name
                        else "BLOCK"
                    )

        # ── DAMAGE ────────────────────────────────────────────────────
        elif at == "DAMAGE":
            attacker_data = a.details.get("attacker_data", [])
            events = a.details.get("events", [])
            if attacker_data:
                dmg_parts = []
                for atk in attacker_data:
                    name = atk.get("name", "Unknown")
                    power = atk.get("power", 0)
                    toughness = atk.get("toughness", 0)
                    damage = atk.get("damage", power)
                    tokens = atk.get("tokens", [])
                    atk_disp = _format_card_for_report(
                        name,
                        power=power,
                        toughness=toughness,
                        attached_tokens=tokens,
                    )
                    dmg_parts.append(f"{atk_disp} deals {damage} damage")
                desc = "DAMAGE: " + "; ".join(dmg_parts)
                # Add combat events (blocked, dies)
                for event in events:
                    event_str = str(event)
                    if " dies" in event_str:
                        creature_name = event_str.replace(" dies", "").strip()
                        effects.append(f"💀 {creature_name} dies")
                    elif " blocked by " in event_str:
                        effects.append(event_str)
                    elif "Death trigger" in event_str:
                        effects.append(f"💀 {event_str}")
                    else:
                        effects.append(event_str)
            elif events:
                formatted_events = [str(event) for event in events]
                desc = "DAMAGE: " + "; ".join(formatted_events)
            elif a.details.get("defender_life") is not None:
                life = a.details["defender_life"]
                desc = f"DAMAGE: Deal damage (opponent at {life} life)"
            else:
                desc = "DAMAGE: Combat damage resolved"

        # ── RESOLVE ───────────────────────────────────────────────────
        elif at == "RESOLVE":
            target = a.details.get("target", "")
            new_power = a.details.get("new_power")
            new_toughness = a.details.get("new_toughness")
            buff_power = a.details.get("buff_power")
            buff_toughness = a.details.get("buff_toughness")
            tokens = a.details.get("tokens", [])
            role_bonus = a.details.get("role_bonus", 0)

            card_str = _format_card_for_report(a.card_name) if a.card_name else "Spell"
            if target:
                target_disp = _format_card_for_report(
                    target,
                    power=new_power,
                    toughness=new_toughness,
                    attached_tokens=tokens,
                )
                desc = f"RESOLVE: {card_str} → {target_disp}"

                if buff_power is not None or buff_toughness is not None:
                    bp = buff_power or 0
                    bt = buff_toughness or 0
                    brp = a.details.get("buff_result_power")
                    brt = a.details.get("buff_result_toughness")
                    if brp is not None and brt is not None:
                        buff_target = _format_card_for_report(
                            target,
                            power=brp,
                            toughness=brt,
                        )
                        buff_str = f"{target} gets +{bp}/+{bt} (now {buff_target})"
                    else:
                        buff_str = f"{target} gets +{bp}/+{bt}"
                    effects.append(f"BUFF: {buff_str}")
                if role_bonus:
                    rrp = a.details.get("role_result_power")
                    rrt = a.details.get("role_result_toughness")
                    if rrp is not None and rrt is not None:
                        role_target = _format_card_for_report(
                            target,
                            power=rrp,
                            toughness=rrt,
                        )
                        role_str = (
                            f"{target} gets +{role_bonus}/+{role_bonus} from Monster Role "
                            f"(now {role_target})"
                        )
                    else:
                        role_str = f"{target} gets +{role_bonus}/+{role_bonus} from Monster Role"
                    effects.append(f"BUFF: {role_str}")
            else:
                desc = f"RESOLVE: {card_str}"

        # ── TRIGGER ───────────────────────────────────────────────────
        elif at == "TRIGGER":
            trigger_type = a.details.get("trigger_type", "")
            effect = a.details.get("effect", "")
            effect_applied = a.details.get("effect_applied", "")
            description = a.details.get("description", "")
            card_str = a.card_name or "Trigger"
            detail_str = effect_applied or description or effect or ""
            if trigger_type:
                desc = f"TRIGGER ({trigger_type}): {card_str}"
            else:
                desc = f"TRIGGER: {card_str}"
            if detail_str:
                effects.append(f"→ {detail_str}")

        # ── UNTAP ─────────────────────────────────────────────────────
        elif at == "UNTAP":
            untapped_data = a.details.get("untapped_data", [])
            untapped = a.details.get("untapped", [])
            lands = a.details.get("lands_on_board", {})
            if untapped_data:
                parts = []
                for entry in untapped_data:
                    name = entry.get("name", "")
                    power = entry.get("power")
                    toughness = entry.get("toughness")
                    tokens = entry.get("tokens", [])
                    parts.append(
                        _format_card_for_report(
                            name,
                            power=power,
                            toughness=toughness,
                            attached_tokens=tokens,
                        )
                    )
                desc = f"UNTAP: Untapped: {', '.join(parts)}"
            elif untapped:
                formatted = [_format_card_for_report(n) for n in untapped]
                desc = f"UNTAP: Untapped: {', '.join(formatted)}"
            else:
                desc = "UNTAP: No tapped permanents"
            if lands:
                lands_str = ", ".join(f"{n} (🌍) x{c}" for n, c in lands.items())
                effects.append(f"(Board: {lands_str})")

        # ── CLEANUP ───────────────────────────────────────────────────
        elif at == "CLEANUP":
            effects_removed = a.details.get("effects_removed", [])
            discarded_cards = a.details.get("discarded_cards", [])
            parts = []
            if effects_removed:
                creature_names = [e.get("creature", "?") for e in effects_removed]
                parts.append(f"Buffs expire: {', '.join(creature_names)}")
                for effect in effects_removed:
                    creature = effect.get("creature", "Unknown")
                    from_p = effect.get("from_power", 0)
                    from_t = effect.get("from_toughness", 0)
                    to_p = effect.get("to_power", 0)
                    to_t = effect.get("to_toughness", 0)
                    creature_disp = _format_card_for_report(
                        creature,
                        power=to_p,
                        toughness=to_t,
                    )
                    effects.append(f"→ {creature_disp}: {from_p}/{from_t} → {to_p}/{to_t}")
            if discarded_cards:
                discard_names = [d.get("card_name", "?") for d in discarded_cards]
                parts.append(f"Discard: {', '.join(discard_names)}")
                for discard in discarded_cards:
                    cname = discard.get("card_name", "Unknown")
                    pidx = discard.get("player_idx", 0)
                    plabel = "Player" if pidx == 0 else "Opponent"
                    effects.append(
                        f"→ {plabel} discards {cname} "
                        f"(hand: {discard.get('hand_size_before', '?')} → "
                        f"{discard.get('hand_size_after', '?')})"
                    )
            desc = "CLEANUP: " + "; ".join(parts) if parts else "CLEANUP: No effects to remove"

        # ── DIES ──────────────────────────────────────────────────────
        elif at == "DIES":
            source = a.details.get("source", "")
            cause = a.details.get("cause", "")
            card_name = a.card_name or "Creature"
            owner_label = f"[{player_label}]" if player_label else ""
            if source:
                desc = f"{owner_label} DIES: {card_name} (from {source})"
            elif cause:
                desc = f"{owner_label} DIES: {card_name} ({cause})"
            else:
                desc = f"{owner_label} DIES: {card_name}"

        # ── DEATH_TRIGGER ─────────────────────────────────────────────
        elif at == "DEATH_TRIGGER":
            trigger_desc = a.details.get("trigger", "")
            damage = a.details.get("damage", 0)
            target_idx = a.details.get("target_player_idx")
            card_str = _format_card_for_report(a.card_name) if a.card_name else "Creature"
            if damage and target_idx is not None:
                target_name = "Player" if target_idx == 0 else "Opponent"
                desc = f"💀 DEATH TRIGGER: {card_str} deals {damage} damage to {target_name}"
            elif trigger_desc:
                desc = f"💀 DEATH TRIGGER: {trigger_desc}"
            else:
                desc = f"💀 DEATH TRIGGER: {card_str}"

        # ── COUNTER ───────────────────────────────────────────────────
        elif at == "COUNTER":
            countered = a.details.get("countered", "Unknown spell")
            destination = a.details.get("destination", "graveyard")
            dest_str = "exiled" if destination == "exile" else "sent to graveyard"
            card_str = _format_card_for_report(a.card_name) if a.card_name else "Counterspell"
            desc = f"{card_str} counters {countered} ({dest_str})"

        # ── EXILE ─────────────────────────────────────────────────────
        elif at == "EXILE":
            source = a.details.get("source", "")
            card_str = _format_card_for_report(a.card_name) if a.card_name else "Card"
            desc = f"EXILE: {card_str} (by {source})" if source else f"EXILE: {card_str}"

        # ── EXILE_ALL ─────────────────────────────────────────────────
        elif at == "EXILE_ALL":
            count = a.details.get("count", 0)
            card_str = a.card_name or "Effect"
            desc = f"EXILE ALL: {card_str} exiles {count} nonland permanent(s)"

        # ── ACTIVATE ──────────────────────────────────────────────────
        elif at == "ACTIVATE":
            mana_cost = a.details.get("mana_cost", "")
            card_str = _format_card_for_report(a.card_name) if a.card_name else "Ability"
            desc = f"Activate {card_str}"
            if mana_cost:
                desc += f" ({format_mana_cost(mana_cost)})"

        # ── SEARCH_LAND ──────────────────────────────────────────────
        elif at == "SEARCH_LAND":
            found = a.details.get("found_land", a.card_name or "land")
            desc = f"Search library → put {found} onto the battlefield"

        # ── ETB_PUT_CREATURES ─────────────────────────────────────────
        elif at == "ETB_PUT_CREATURES":
            count = a.details.get("count", 0)
            card_str = a.card_name or "Effect"
            desc = f"{card_str} creates {count} token creature(s)"

        # ── ETB_DRAW ──────────────────────────────────────────────────
        elif at == "ETB_DRAW":
            count = a.details.get("count", 0)
            card_str = a.card_name or "Effect"
            desc = f"{card_str} draws {count} card(s)"

        # ── FALLBACK ──────────────────────────────────────────────────
        else:
            card_part = f" {a.card_name}" if a.card_name else ""
            desc = f"{at}{card_part}"
            if a.details:
                key_details = []
                for k in ["target", "target_kind", "deals_damage", "effect", "description"]:
                    if k in a.details and a.details[k]:
                        key_details.append(f"{a.details[k]}")
                if key_details:
                    desc += " (" + ", ".join(key_details) + ")"

        # Map raw GamePhase enum names to HTML-friendly display names
        phase_display = _PHASE_DISPLAY_NAMES.get(a.phase, a.phase)

        # Override phase for actions that belong to specific combat sub-phases
        # (matches run_gameplay behaviour so attacks/blocks show in the right section)
        if at == "ATTACK":
            phase_display = "Combat - Declare Attackers"
        elif at == "BLOCK":
            phase_display = "Combat - Declare Blockers"

        out.append(
            {
                "turn": a.turn,
                "phase": phase_display,
                "player": player_label,
                "action_type": at,
                "description": desc,
                "active_player_turn": active_label,
                "effects": effects if effects else None,
            }
        )
    return out


def turn_summary_from_env(env: tp.Any, turn: int) -> dict[str, tp.Any]:
    """Build ``record_turn_summary`` kwargs from ``env.state.turn_actions``.

    Computes per-turn stats (damage, spells, lands, creatures, draws) from
    the action log, matching the gameplay CLI's end-of-turn summary.

    Args:
        env: An ``MTGEnv`` instance (unwrapped).
        turn: The turn number to summarise.

    Returns:
        kwargs dict for ``GameRecorder.record_turn_summary(**result)``.

    """
    state = env.state
    if state is None:
        return {"turn": turn}

    turn_data = state.turn_actions.get(turn, {})

    def _count(player_idx: int) -> dict[str, int]:
        actions = turn_data.get(player_idx, [])
        spells = 0
        lands = 0
        creatures = 0
        damage = 0
        draws = 0
        for a in actions:
            at = (a.action_type if isinstance(a.action_type, str) else "").lower()
            if "cast" in at:
                spells += 1
                # Check if creature
                if a.card_name:
                    try:
                        card = get_card(a.card_name)
                        if card and card.card_type == CardType.CREATURE:
                            creatures += 1
                    except (KeyError, AttributeError):
                        pass
            elif "play" in at and "land" in at:
                lands += 1
            elif "draw" in at:
                draws += 1
            # Damage from details
            if a.details and "damage" in a.details:
                with contextlib.suppress(ValueError, TypeError):
                    damage += int(a.details["damage"])
        return {
            "spells": spells,
            "lands": lands,
            "creatures": creatures,
            "damage": damage,
            "draws": draws,
        }

    p = _count(0)
    o = _count(1)

    return {
        "turn": turn,
        "player_damage": p["damage"],
        "opponent_damage": o["damage"],
        "player_spells": p["spells"],
        "opponent_spells": o["spells"],
        "player_lands": p["lands"],
        "opponent_lands": o["lands"],
        "player_creatures": p["creatures"],
        "opponent_creatures": o["creatures"],
        "player_draws": p["draws"],
        "opponent_draws": o["draws"],
    }


def _generate_html_header() -> str:
    """Generate HTML header with styles."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MTG-Causal-RL Game Replay</title>
    <style>
        :root {
            --bg-dark: #1a1b26;
            --bg-card: #24283b;
            --bg-hover: #2f3549;
            --accent-blue: #7aa2f7;
            --accent-green: #9ece6a;
            --accent-red: #f7768e;
            --accent-yellow: #e0af68;
            --accent-purple: #bb9af7;
            --text-primary: #c0caf5;
            --text-secondary: #565f89;
            --text-dim: #414868;
            --border: #3b4261;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            background: var(--bg-dark);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        header {
            text-align: center;
            margin-bottom: 2rem;
            padding: 2rem;
            background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-dark) 100%);
            border-radius: 12px;
            border: 1px solid var(--border);
        }

        h1 {
            font-size: 2.5rem;
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        }

        h2 {
            font-size: 1.5rem;
            color: var(--accent-blue);
            margin: 1.5rem 0 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--border);
        }

        .game-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1.5rem;
        }

        .info-card {
            background: var(--bg-card);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid var(--border);
        }

        .info-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }

        .info-value {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--accent-blue);
        }

        /* Initial Game Section */
        .initial-game {
            margin: 2rem 0;
            background: var(--bg-card);
            border-radius: 12px;
            border: 1px solid var(--border);
            padding: 1.5rem;
        }

        .initial-game h2 {
            margin-top: 0;
            color: var(--accent-purple);
        }

        .play-draw-box {
            background: var(--bg-hover);
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            text-align: center;
        }

        .play-draw-box .coin {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }

        .mulligan-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }

        .mulligan-box {
            background: var(--bg-hover);
            border-radius: 8px;
            padding: 1rem;
        }

        .mulligan-box.player-side {
            border-left: 4px solid var(--accent-green);
        }

        .mulligan-box.opponent-side {
            border-left: 4px solid var(--accent-red);
        }

        .mulligan-box h3 {
            font-size: 1rem;
            margin-bottom: 0.75rem;
        }

        .hand-cards {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 0.75rem 0;
        }

        .card-tag {
            background: var(--bg-card);
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
        }

        .card-tag.hidden-card {
            opacity: 0.3;
            font-style: italic;
        }

        .mana-cost {
            font-size: 0.75rem;
        }

        .decision-tag {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.85rem;
        }

        .decision-tag.keep {
            background: var(--accent-green);
            color: var(--bg-dark);
        }

        .decision-tag.mulligan {
            background: var(--accent-red);
            color: var(--bg-dark);
        }

        /* Timeline */
        .timeline {
            position: relative;
            margin: 2rem 0;
        }

        .turn-section {
            margin-bottom: 2rem;
            background: var(--bg-card);
            border-radius: 12px;
            border: 1px solid var(--border);
            overflow: hidden;
        }

        .turn-header {
            background: linear-gradient(90deg, var(--accent-blue) 0%, var(--accent-purple) 100%);
            color: var(--bg-dark);
            padding: 1rem 1.5rem;
            font-weight: 700;
            font-size: 1.25rem;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .turn-header:hover {
            filter: brightness(1.1);
        }

        .turn-content {
            padding: 1.5rem;
        }

        /* Phase display */
        .player-turn-block, .opponent-turn-block {
            margin-bottom: 1.5rem;
            padding: 1rem;
            border-radius: 8px;
        }

        .player-turn-block {
            background: linear-gradient(90deg, rgba(158, 206, 106, 0.1) 0%, transparent 100%);
            border-left: 4px solid var(--accent-green);
        }

        .opponent-turn-block {
            background: linear-gradient(90deg, rgba(247, 118, 142, 0.1) 0%, transparent 100%);
            border-left: 4px solid var(--accent-red);
        }

        .turn-block-header {
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
        }

        .player-turn-block .turn-block-header {
            color: var(--accent-green);
        }

        .opponent-turn-block .turn-block-header {
            color: var(--accent-red);
        }

        .phase-section {
            margin-bottom: 1rem;
            background: var(--bg-hover);
            border-radius: 8px;
            padding: 0.75rem 1rem;
        }

        .phase-section:last-child {
            margin-bottom: 0;
        }

        .phase-header {
            color: var(--accent-yellow);
            font-weight: 600;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Declare Blockers shows in opposite color (blocker's perspective) */
        .player-turn-block .phase-header.blockers-phase { color: var(--accent-red); }
        .opponent-turn-block .phase-header.blockers-phase { color: var(--accent-green); }

        .action {
            background: var(--bg-card);
            border-radius: 6px;
            padding: 0.5rem 0.75rem;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: flex-start;
            gap: 0.5rem;
        }

        .action.opponent-action {
            border-left: 3px solid var(--accent-red);
            background: linear-gradient(90deg, rgba(247, 118, 142, 0.1), var(--bg-card));
        }

        .action.response-action {
            margin-left: 1.5rem;
            border: 1px dashed var(--accent-yellow);
        }

        .action-icon {
            font-size: 1rem;
            min-width: 1.25rem;
            text-align: center;
        }

        .action-content {
            flex: 1;
        }

        .action-desc {
            color: var(--text-primary);
            font-size: 0.9rem;
        }

        .action-effects {
            margin-top: 0.25rem;
            font-size: 0.8rem;
            color: var(--accent-purple);
        }

        .no-actions {
            color: var(--text-dim);
            font-size: 0.85rem;
            font-style: italic;
        }

        /* Hand display */
        .hand-section {
            margin: 1rem 0;
            padding: 0.75rem;
            background: var(--bg-card);
            border-radius: 8px;
            border: 1px dashed var(--border);
        }

        .hand-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }

        /* State grid */
        .state-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-top: 1rem;
        }

        .player-state {
            background: var(--bg-hover);
            border-radius: 8px;
            padding: 1rem;
        }

        .player-state.player-side {
            border-left: 4px solid var(--accent-green);
        }

        .player-state.opponent-side {
            border-left: 4px solid var(--accent-red);
        }

        .player-name {
            font-weight: 600;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .stats-row {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            margin-bottom: 0.5rem;
        }

        .stat {
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }

        .stat-icon {
            font-size: 1rem;
        }

        .stat-value {
            font-weight: 600;
        }

        .life { color: var(--accent-red); }
        .cards { color: var(--accent-blue); }
        .lands { color: var(--accent-green); }
        .power { color: var(--accent-yellow); }
        .mana { color: var(--accent-purple); }

        .state-section {
            margin: 0.4rem 0;
            padding: 0.3rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }

        .state-section .section-label {
            color: var(--text-secondary);
            font-size: 0.85rem;
            margin-right: 0.5rem;
        }

        .state-section .empty {
            color: var(--text-secondary);
            font-style: italic;
            opacity: 0.6;
        }

        .mana-pool {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            padding: 0.5rem;
            background: var(--bg-card);
            border-radius: 4px;
            margin-top: 0.5rem;
        }

        .mana-item {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            font-size: 0.9rem;
        }

        .lands-detail {
            margin-top: 0.5rem;
            padding: 0.5rem;
            background: var(--bg-card);
            border-radius: 4px;
            font-size: 0.85rem;
        }

        .creature-list, .graveyard-list, .token-list {
            margin-top: 0.75rem;
            padding-top: 0.75rem;
            border-top: 1px solid var(--border);
        }

        .section-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.5rem;
        }

        .creature-tag, .gy-tag, .token-tag {
            display: inline-block;
            background: var(--bg-card);
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
            margin-right: 0.5rem;
            margin-bottom: 0.25rem;
        }

        .creature-tag.tapped {
            opacity: 0.6;
        }

        .token-tag {
            border: 1px dashed var(--accent-yellow);
        }

        .gy-tag.creature { border-left: 2px solid var(--accent-red); }
        .gy-tag.instant, .gy-tag.sorcery { border-left: 2px solid var(--accent-blue); }

        /* Legend Box - Premium Styling */
        .legend-box {
            margin: 1.5rem 0;
            padding: 1.25rem;
            background: linear-gradient(135deg, var(--bg-hover) 0%, var(--bg-card) 100%);
            border-radius: 12px;
            border: 1px solid var(--border);
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }

        .legend-box h3 {
            color: var(--accent-yellow);
            font-size: 0.85rem;
            margin-bottom: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .legend-box h3::before {
            content: "📖";
        }

        .legend-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
        }

        .legend-category {
            background: var(--bg-dark);
            padding: 0.75rem;
            border-radius: 8px;
            border: 1px solid var(--border);
        }

        .legend-category h4 {
            color: var(--text-secondary);
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.25rem;
        }

        .legend-items {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem 1rem;
        }

        .legend-item {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            font-size: 0.8rem;
            color: var(--text-primary);
        }

        .legend-item .icon {
            font-size: 1rem;
        }

        /* Turn summary - CLI-style */
        .turn-summary {
            margin-top: 1.5rem;
            padding: 1.25rem;
            background: linear-gradient(135deg, rgba(139,233,253,0.05) 0%, rgba(189,147,249,0.05) 100%);
            border-radius: 12px;
            border: 2px solid var(--accent-purple);
            position: relative;
        }

        .turn-summary::before {
            content: "📊";
            position: absolute;
            top: -12px;
            left: 20px;
            background: var(--bg-dark);
            padding: 0 0.5rem;
            font-size: 1.2rem;
        }

        .turn-summary h4 {
            font-size: 1rem;
            color: var(--accent-yellow);
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            text-align: center;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }

        .summary-col {
            padding: 1rem;
            background: var(--bg-card);
            border-radius: 8px;
        }

        .summary-col h5 {
            font-size: 0.85rem;
            margin-bottom: 0.75rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
        }

        .summary-col.player {
            border-left: 4px solid var(--accent-green);
        }
        .summary-col.player h5 { color: var(--accent-green); }

        .summary-col.opponent {
            border-left: 4px solid var(--accent-red);
        }
        .summary-col.opponent h5 { color: var(--accent-red); }

        .summary-stat {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.9rem;
            padding: 0.4rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }

        .summary-stat:last-child {
            border-bottom: none;
        }

        .summary-stat .label {
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .summary-stat .value {
            font-weight: 600;
            font-size: 1rem;
        }

        .summary-stat .value.positive { color: var(--accent-green); }
        .summary-stat .value.negative { color: var(--accent-red); }
        .summary-stat .value.neutral { color: var(--text-dim); }

        /* Life bar */
        .life-bar-container {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-top: 0.5rem;
        }

        .life-bar {
            flex: 1;
            height: 8px;
            background: var(--bg-dark);
            border-radius: 4px;
            overflow: hidden;
        }

        .life-bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }

        .life-bar-fill.player { background: linear-gradient(90deg, var(--accent-green) 0%, var(--accent-blue) 100%); }
        .life-bar-fill.opponent { background: linear-gradient(90deg, var(--accent-red) 0%, var(--accent-yellow) 100%); }

        /* Final Game Summary */
        .final-game-summary {
            margin: 3rem 0;
            padding: 2rem;
            background: linear-gradient(135deg, var(--bg-hover) 0%, var(--bg-card) 100%);
            border-radius: 16px;
            border: 2px solid var(--accent-purple);
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        }

        .final-game-summary h2 {
            text-align: center;
            font-size: 1.5rem;
            color: var(--accent-yellow);
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
        }

        .final-stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-bottom: 2rem;
        }

        .final-player-panel {
            background: var(--bg-card);
            border-radius: 12px;
            padding: 1.5rem;
        }

        .final-player-panel.player { border: 2px solid var(--accent-green); }
        .final-player-panel.opponent { border: 2px solid var(--accent-red); }

        .final-player-panel h3 {
            font-size: 1.2rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .final-player-panel.player h3 { color: var(--accent-green); }
        .final-player-panel.opponent h3 { color: var(--accent-red); }

        .final-life-display {
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            margin: 1rem 0;
        }

        .final-life-display.player { color: var(--accent-green); }
        .final-life-display.opponent { color: var(--accent-red); }

        .final-board-state {
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border);
        }

        .final-board-row {
            display: flex;
            align-items: flex-start;
            margin: 0.5rem 0;
            font-size: 0.9rem;
        }

        .final-board-row .icon {
            width: 24px;
            text-align: center;
            margin-right: 0.5rem;
        }

        .final-board-row .label {
            color: var(--text-secondary);
            min-width: 100px;
        }

        .final-board-row .items {
            flex: 1;
            color: var(--text-primary);
        }

        .final-board-row .items.empty {
            color: var(--text-dim);
            font-style: italic;
        }

        /* Tapped indicator */
        .tapped-indicator {
            color: var(--accent-yellow);
            font-size: 0.75rem;
            margin-left: 0.25rem;
        }

        /* Action type badges */
        .action-type-badge {
            display: inline-block;
            padding: 0.15rem 0.4rem;
            border-radius: 3px;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            margin-right: 0.5rem;
        }

        .action-type-badge.cast { background: var(--accent-yellow); color: var(--bg-dark); }
        .action-type-badge.play_land { background: var(--accent-green); color: var(--bg-dark); }
        .action-type-badge.attack { background: var(--accent-red); color: white; }
        .action-type-badge.draw { background: var(--accent-blue); color: white; }
        .action-type-badge.counter { background: var(--accent-purple); color: white; }
        .action-type-badge.damage { background: #ff6b6b; color: white; }

        /* Turn Start Display */
        .turn-start-display {
            margin: 1rem 0;
            padding: 1rem;
            background: linear-gradient(135deg, rgba(139,233,253,0.1) 0%, rgba(189,147,249,0.1) 100%);
            border-radius: 12px;
            border: 1px solid var(--border);
        }

        .turn-start-display h4 {
            text-align: center;
            color: var(--accent-cyan);
            margin-bottom: 1rem;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }

        .turn-start-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }

        .turn-start-panel {
            background: var(--bg-card);
            border-radius: 8px;
            padding: 1rem;
        }

        .turn-start-panel.player { border-left: 3px solid var(--accent-green); }
        .turn-start-panel.opponent { border-left: 3px solid var(--accent-red); }

        .turn-start-panel h5 {
            font-size: 0.85rem;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .turn-start-panel.player h5 { color: var(--accent-green); }
        .turn-start-panel.opponent h5 { color: var(--accent-red); }

        .start-row {
            display: flex;
            align-items: flex-start;
            margin: 0.4rem 0;
            font-size: 0.85rem;
            padding: 0.3rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }

        .start-row:last-child { border-bottom: none; }

        .start-row .category {
            min-width: 120px;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }

        .start-row .contents {
            flex: 1;
            color: var(--text-primary);
        }

        /* Board State Sections */
        .board-states-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-top: 1.5rem;
        }

        .board-state {
            background: var(--bg-card);
            border-radius: 8px;
            padding: 1rem;
        }

        .board-state.player-board { border-left: 4px solid var(--accent-green); }
        .board-state.opponent-board { border-left: 4px solid var(--accent-red); }

        .board-state h5 {
            font-size: 0.9rem;
            margin-bottom: 0.75rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
        }

        .board-state.player-board h5 { color: var(--accent-green); }
        .board-state.opponent-board h5 { color: var(--accent-red); }

        .life-display {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 1rem;
            padding: 0.5rem;
            background: var(--bg-dark);
            border-radius: 6px;
        }

        .life-display .life-icon { font-size: 1.2rem; }
        .life-display .life-value { font-size: 1.5rem; font-weight: 700; min-width: 40px; }
        .life-display .life-bar { flex: 1; }

        .board-row {
            display: flex;
            align-items: flex-start;
            margin: 0.35rem 0;
            font-size: 0.85rem;
            line-height: 1.4;
        }

        .board-row .icon {
            width: 20px;
            text-align: center;
            margin-right: 0.35rem;
        }

        .board-row .label {
            color: var(--text-secondary);
            min-width: 100px;
        }

        .board-row .items {
            flex: 1;
            color: var(--text-primary);
            word-wrap: break-word;
        }

        .board-row .items.empty {
            color: var(--text-dim);
            font-style: italic;
        }

        /* Summary Stats Grid */
        .summary-stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-bottom: 1rem;
        }

        /* Life Change Indicators */
        .life-change { margin-left: 0.5rem; font-size: 0.85rem; }
        .life-change.positive { color: var(--accent-green); }
        .life-change.negative { color: var(--accent-red); }
        .life-change.neutral { color: var(--text-dim); }

        /* Inline Legend */
        .legend-inline {
            margin-top: 1rem;
            padding-top: 0.75rem;
            border-top: 1px solid var(--border);
            text-align: center;
            font-size: 0.75rem;
            color: var(--text-dim);
        }

        /* Card Info Text */
        .card-info-text {
            font-size: 0.75rem;
            color: var(--accent-purple);
            font-style: italic;
            margin-top: 0.25rem;
            padding-left: 1.5rem;
        }

        /* Final Game Summary Enhanced */
        .final-stat-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }

        .final-stat-row .stat-label {
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .final-stat-row .stat-value {
            font-weight: 600;
            font-size: 1.1rem;
        }

        .final-stat-row .stat-value.positive {
            color: var(--accent-green);
        }

        .final-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--border), transparent);
            margin: 1rem 0;
        }

        .final-game-footer {
            text-align: center;
            margin-top: 2rem;
            padding: 1.5rem;
            background: var(--bg-card);
            border-radius: 8px;
            border: 1px solid var(--border);
        }

        .game-stats-row {
            display: inline-flex;
            gap: 1.5rem;
            align-items: center;
            font-size: 0.95rem;
            color: var(--text-secondary);
        }

        .game-stats-row strong {
            color: var(--text-primary);
        }

        .game-stats-row .player-color {
            color: var(--accent-green);
        }

        .game-stats-row .opponent-color {
            color: var(--accent-red);
        }

        .game-stats-row .divider {
            color: var(--border);
        }

        .empty {
            color: var(--text-dim);
            font-style: italic;
        }

        /* Hidden hand styling */
        .hidden-hand {
            color: var(--text-secondary);
            font-style: italic;
            opacity: 0.7;
        }

        .hidden-hand::before {
            content: "(hidden) ";
            color: var(--text-dim);
            font-size: 0.8em;
        }

        /* Result subtitle */
        .result-subtitle {
            text-align: center;
            color: var(--text-secondary);
            font-size: 1rem;
            margin-top: -0.5rem;
            margin-bottom: 1.5rem;
        }

        /* Winner panel highlight */
        .final-player-panel.winner {
            box-shadow: 0 0 20px rgba(158, 206, 106, 0.3);
        }

        .final-player-panel.winner h3::after {
            content: " 👑";
        }

        /* Deck name display */
        .deck-name {
            text-align: center;
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-top: -0.5rem;
            margin-bottom: 0.75rem;
            font-style: italic;
        }

        .winner-banner {
            text-align: center;
            padding: 2rem;
            margin: 2rem 0;
            background: linear-gradient(135deg, var(--accent-green) 0%, var(--accent-blue) 100%);
            color: var(--bg-dark);
            border-radius: 12px;
            font-size: 1.5rem;
            font-weight: 700;
        }

        .winner-banner.player-wins {
            background: linear-gradient(135deg, var(--accent-green) 0%, var(--accent-blue) 100%);
        }

        .winner-banner.opponent-wins {
            background: linear-gradient(135deg, var(--accent-red) 0%, var(--accent-yellow) 100%);
        }
        .winner-banner.draw {
            background: linear-gradient(135deg, var(--accent-yellow) 0%, var(--accent-purple) 100%);
            color: var(--bg-dark);
        }

        footer {
            text-align: center;
            padding: 2rem;
            color: var(--text-dim);
            font-size: 0.85rem;
        }

        footer a {
            color: var(--accent-blue);
            text-decoration: none;
        }

        footer a:hover {
            text-decoration: underline;
        }

        /* Returned cards in mulligan */
        .returned-cards .card-tag.returned {
            background: rgba(247, 118, 142, 0.2);
            border: 1px dashed var(--accent-red);
            text-decoration: line-through;
            opacity: 0.7;
        }

        /* Graveyard with type counts */
        .graveyard-summary {
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-top: 0.5rem;
            padding: 0.25rem 0.5rem;
            background: var(--bg-dark);
            border-radius: 4px;
            display: inline-block;
        }

        /* Opponent hand (hidden) */
        .opponent-hand-hidden {
            font-style: italic;
            color: var(--text-dim);
            font-size: 0.85rem;
        }

        @media (max-width: 768px) {
            body { padding: 1rem; }
            .state-grid, .mulligan-section, .summary-grid { grid-template-columns: 1fr; }
            h1 { font-size: 1.75rem; }
        }
    </style>
</head>
<body>
"""


def _generate_html_footer() -> str:
    """Generate HTML footer."""
    return """
    <footer>
        <p>Generated by <a href="https://github.com/your-repo/mtg-causal-rl">MTG-Causal-RL</a></p>
        <p>A Causal Reinforcement Learning Benchmark for Magic: The Gathering</p>
    </footer>
</body>
</html>
"""


def _action_icon(action_type: str) -> str:
    """Get icon for action type."""
    icons = {
        "CAST": "✨",
        "CAST_FLASHBACK": "✨",
        "PLAY_LAND": "🌍",
        "ATTACK": "⚔️",
        "BLOCK": "🛡️",
        "DRAW": "🃏",
        "DRAW_SELECTION": "🃏",
        "PASS": "➡️",
        "MULLIGAN": "🔄",
        "DAMAGE": "💥",
        "INSTANT": "⚡",
        "PROWESS": "🔥",
        "UNTAP": "🔓",
        "UPKEEP": "⏰",
        "TRIGGER": "⭐",
        "DEATH_TRIGGER": "💀",
        "DIES": "💀",
        "COUNTER": "🚫",
        "COUNTERED": "🚫",
        "EXILE": "⛔",
        "EXILE_ALL": "⛔",
        "BUFF": "💪",
        "RESOLVE": "✅",
        "CLEANUP": "🧹",
        "ACTIVATE": "⚡",
        "SEARCH_LAND": "🔍",
        "ETB_PUT_CREATURES": "⚔️",
        "ETB_DRAW": "🃏",
    }
    return icons.get(action_type.upper(), "•")


def _format_lands(lands: dict[str, int]) -> str:
    """Format lands dictionary for display with type symbol."""
    if not lands:
        return "None"
    parts = []
    for land_name, count in lands.items():
        parts.append(f"{land_name} (🌍) x{count}")
    return ", ".join(parts)


# Card type symbols matching CLI
CARD_TYPE_SYMBOLS = {
    "creature": "⚔️",
    "instant": "✨",
    "sorcery": "🌟",
    "enchantment": "🔮",
    "artifact": "⚙️",
    "land": "🌍",
    "planeswalker": "👤",
    "token": "🎴",
}


def _format_card_html(
    name: str,
    mana_cost: str = "",
    card_type: str = "",
    power: int | None = None,
    toughness: int | None = None,
    hidden: bool = False,
) -> str:
    """Format a card for HTML display matching CLI format.

    Automatically looks up card data from CardRegistry if not provided.

    Args:
        name: Card name.
        mana_cost: Mana cost string (e.g., "1R"). If empty, looks up from registry.
        card_type: Card type (e.g., "creature", "instant"). If empty, looks up from registry.
        power: Power (for creatures). If None, looks up from registry.
        toughness: Toughness (for creatures). If None, looks up from registry.
        hidden: Whether this is a hidden card (opponent's hand).

    Returns:
        HTML-formatted card string.

    """
    # Try to look up card data from registry if not provided
    try:
        from mtg.env.card_definitions import get_card

        card = get_card(name)
        if card:
            if not mana_cost and card.mana_cost:
                mana_cost = card.mana_cost.to_text()
            if not card_type:
                card_type = card.card_type.name.lower() if card.card_type else ""
            if power is None and card.power is not None:
                power = card.power
            if toughness is None and card.toughness is not None:
                toughness = card.toughness
    except Exception:
        pass

    mana_str = format_mana_cost(mana_cost) if mana_cost else ""
    type_symbol = CARD_TYPE_SYMBOLS.get(card_type.lower(), "") if card_type else ""

    # Build display: Name - P/T (mana, type) for creatures only, Name (mana, type) for non-creatures
    # Note: Tokens that are creatures would have power/toughness, but artifact tokens (like Map) don't
    parts = [name]
    is_creature_type = card_type.lower() == "creature"
    if (
        power is not None
        and toughness is not None
        and is_creature_type
        and (power > 0 or toughness > 0)
    ):
        parts.append(f" - {power}/{toughness}")

    meta_parts = []
    if mana_str:
        meta_parts.append(mana_str)
    if type_symbol:
        meta_parts.append(type_symbol)

    if meta_parts:
        parts.append(f" ({', '.join(meta_parts)})")

    card_text = "".join(parts)
    hidden_class = " hidden-card" if hidden else ""
    return f'<span class="card-tag{hidden_class}">{card_text}</span>'


def _format_graveyard_html(graveyard: list[dict[str, str]]) -> str:
    """Format graveyard for HTML display with proper card formatting and type counts."""
    if not graveyard:
        return '<span class="empty">Empty</span>'

    html_parts = []

    # Format each card with type symbol and mana cost
    for g in graveyard:
        name = g.get("name", "Unknown")
        card_type = g.get("type", "").lower()
        type_symbol = CARD_TYPE_SYMBOLS.get(card_type, "")

        # Try to get mana cost from card registry
        mana_str = ""
        try:
            from mtg.env.card_definitions import get_card

            card = get_card(name)
            if card and card.mana_cost:
                mana_str = format_mana_cost(card.mana_cost.to_text())
        except Exception:
            pass

        # Build display: Name (mana, type_symbol)
        if mana_str and type_symbol:
            display = f"{name} ({mana_str}, {type_symbol})"
        elif type_symbol:
            display = f"{name} ({type_symbol})"
        elif mana_str:
            display = f"{name} ({mana_str})"
        else:
            display = name

        html_parts.append(f'<span class="gy-tag {card_type}">{display}</span>')

    # Add type counts in parentheses
    type_counts: dict[str, int] = {}
    for g in graveyard:
        card_type = g.get("type", "other").lower()
        type_counts[card_type] = type_counts.get(card_type, 0) + 1

    count_parts = []
    for t, c in type_counts.items():
        icon = CARD_TYPE_SYMBOLS.get(t, "•")
        count_parts.append(f"{icon}{c}")

    if count_parts:
        html_parts.append(f'<span class="graveyard-summary">({" ".join(count_parts)})</span>')

    return " ".join(html_parts)


def _format_mana_pool(mana: dict[str, int]) -> str:
    """Format mana pool for display."""
    if not mana or all(v == 0 for v in mana.values()):
        return ""
    parts = []
    for color, amount in mana.items():
        if amount > 0:
            icon = MANA_ICONS.get(color, color)
            parts.append(f"{icon}x{amount}")
    return " ".join(parts)


def _get_card_text_for_action(action_type: str, description: str) -> str:
    """Get the card rules text for CAST/DRAW actions on non-land cards.

    Args:
        action_type: The action type (CAST, DRAW, etc.).
        description: The action description.

    Returns:
        The card rules text, or empty string if not applicable.

    """
    if action_type.upper() not in ("CAST", "DRAW", "INSTANT"):
        return ""

    try:
        from mtg.env.card_definitions import get_card

        # Extract card name from description
        # Format: "Cast CardName - P/T (mana, type)" or "Draw CardName (mana, type)"
        # or "Cast CardName (mana, type)" for non-creatures
        # Card name ends at " - " (for P/T), " (" (for mana), or end of string

        # Remove the action prefix (Cast/Draw)
        desc_lower = description.lower()
        if desc_lower.startswith("cast ") or desc_lower.startswith("draw "):
            rest = description[5:]
        else:
            return ""

        # Extract card name: ends at " - " (creature stats) or " (" (mana cost)
        if " - " in rest:
            card_name = rest.split(" - ")[0].strip()
        elif " (" in rest:
            card_name = rest.split(" (")[0].strip()
        else:
            card_name = rest.strip()

        card = get_card(card_name)
        if card and card.rules_text and card.card_type.name.lower() != "land":
            card_type_name = card.card_type.name.capitalize()
            return f"{card_type_name}: {card.rules_text}"
    except Exception:
        pass

    return ""


def _strip_rich_markup(text: str) -> str:
    """Strip Rich markup tags from text while preserving content.

    Handles patterns like [bold red]text[/bold red], [dim]text[/dim], etc.
    Preserves [🎴...] token indicators which are NOT Rich markup.

    Args:
        text: Text potentially containing Rich markup.

    Returns:
        Plain text with markup removed.

    """
    # Remove Rich-style tags: [tag], [/tag], [tag attr]
    # BUT preserve [🎴...] which are token indicators, not Rich markup
    # Rich tags start with letters or /, token indicators start with 🎴
    cleaned = re.sub(r"\[(?!🎴)[^\]]*\]", "", text)
    # Clean up any double spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _format_action_description(description: str) -> str:
    """Format an action description by replacing card names with proper formatting.

    Recognizes patterns like:
    - "Cast CardName"
    - "Play CardName"
    - "Draw CardName"
    - "CardName attacks"
    - "CardName blocks"
    - "[Opponent] Cast..." (caster indication)
    - "targeting Opponent's CardName" (target ownership)
    - "[🎴Monster Role]" (token suffix - preserved)

    Args:
        description: The action description to format (may contain Rich markup).

    Returns:
        Formatted description with card names properly displayed.

    """
    # First strip Rich markup to get clean text
    description = _strip_rich_markup(description)

    # For ATTACK/DAMAGE/UNTAP actions with multiple creatures, preserve as-is
    # (token suffixes may be embedded in middle of creature list)
    if any(desc_type in description.upper() for desc_type in ["ATTACK:", "DAMAGE:", "UNTAP:"]):
        return description

    # Extract and preserve any token suffixes like [🎴Monster Role]
    token_suffix_match = re.search(r"(\s*\[🎴[^\]]+\])", description)
    token_suffix = token_suffix_match.group(1) if token_suffix_match else ""
    # Remove token suffix from description for parsing, will add back later
    if token_suffix:
        description = description.replace(token_suffix, "")

    def is_creature(card) -> bool:
        """Check if a card is a creature type."""
        if not card or not card.card_type:
            return False
        return card.card_type.name.lower() == "creature"

    # Try to extract and format card names from common action patterns
    # Handle both "CAST:" (from CLI) and "Cast" (simple) formats
    try:
        from mtg.env.card_definitions import get_card

        # Pattern: "CAST: CardName" or "Cast CardName" with optional P/T and targeting
        # Extract P/T from description if present (for dynamic power like Haughty Djinn)
        cast_match = re.match(
            r"CAST:?\s+([A-Za-z\s'\-]+?)(?:\s*[-–]\s*(\d+)/(\d+))?(?:\s*\([^)]*\))?(?:\s+targeting\s+(.*))?$",
            description,
            re.IGNORECASE,
        )
        if cast_match:
            card_name = cast_match.group(1).strip()
            desc_power = cast_match.group(2)  # P/T from description (may be dynamic)
            desc_toughness = cast_match.group(3)
            target_info = cast_match.group(4)
            card = get_card(card_name)
            if card:
                mana_str = format_mana_cost(card.mana_cost.to_text()) if card.mana_cost else ""
                type_symbol = CARD_TYPE_SYMBOLS.get(card.card_type.name.lower(), "")
                target_str = f" targeting {target_info}" if target_info else ""
                # Use P/T from description if available (preserves dynamic values)
                if desc_power and desc_toughness:
                    return f"Cast {card_name} - {desc_power}/{desc_toughness} ({mana_str}, {type_symbol}){token_suffix}{target_str}"
                elif is_creature(card) and card.power is not None and card.toughness is not None:
                    return f"Cast {card_name} - {card.power}/{card.toughness} ({mana_str}, {type_symbol}){token_suffix}{target_str}"
                else:
                    return f"Cast {card_name} ({mana_str}, {type_symbol}){token_suffix}{target_str}"

        # Pattern: "DRAW: CardName" or "Draw CardName" with optional P/T
        draw_match = re.match(
            r"DRAW:?\s+([A-Za-z\s'\-]+?)(?:\s*[-–]\s*(\d+)/(\d+))?(?:\s*\([^)]*\))?$",
            description,
            re.IGNORECASE,
        )
        if draw_match:
            card_name = draw_match.group(1).strip()
            desc_power = draw_match.group(2)
            desc_toughness = draw_match.group(3)
            card = get_card(card_name)
            if card:
                mana_str = format_mana_cost(card.mana_cost.to_text()) if card.mana_cost else ""
                type_symbol = CARD_TYPE_SYMBOLS.get(card.card_type.name.lower(), "")
                # Use P/T from description if available (preserves dynamic values)
                if desc_power and desc_toughness:
                    return f"Draw {card_name} - {desc_power}/{desc_toughness} ({mana_str}, {type_symbol}){token_suffix}"
                elif is_creature(card) and card.power is not None and card.toughness is not None:
                    return f"Draw {card_name} - {card.power}/{card.toughness} ({mana_str}, {type_symbol}){token_suffix}"
                else:
                    meta_parts = [p for p in [mana_str, type_symbol] if p]
                    meta_str = ", ".join(meta_parts) if meta_parts else ""
                    base = f"Draw {card_name} ({meta_str})" if meta_str else f"Draw {card_name}"
                    return f"{base}{token_suffix}"

        # Pattern: "PLAY_LAND: CardName" or "Play CardName"
        # Handles: "PLAY_LAND: Mountain (🌍) (Board: Mountain (🌍) x1)"
        # Extract both the card name AND the board state if present
        play_match = re.match(
            r"(?:PLAY_LAND:?\s*|Play\s+)([A-Za-z][A-Za-z\s'\-]*?)(?:\s*\([^)]*\))?\s*(?:\(enters tapped\))?\s*(?:\(Board:\s*([^)]+)\))?",
            description,
            re.IGNORECASE,
        )
        if play_match:
            card_name = play_match.group(1).strip()
            board_state = play_match.group(2)
            card = get_card(card_name)
            if card and card.card_type.name.lower() == "land":
                type_symbol = CARD_TYPE_SYMBOLS.get(card.card_type.name.lower(), "")
                # Check if enters tapped
                enters_tapped = "(enters tapped) " if "enters tapped" in description.lower() else ""
                # Include board state like CLI
                if board_state:
                    return f"Play {card_name} ({type_symbol}) {enters_tapped}(Board: {board_state}){token_suffix}"
                else:
                    return f"Play {card_name} ({type_symbol}) {enters_tapped}{token_suffix}".strip()

        # Pattern: "CardName attacks" or "Attack with CardName"
        attack_match = re.match(
            r"(?:Attack with\s+)?([A-Za-z\s']+?)\s+attacks?", description, re.IGNORECASE
        )
        if attack_match:
            card_name = attack_match.group(1).strip()
            card = get_card(card_name)
            if card and is_creature(card) and card.power is not None:
                mana_str = format_mana_cost(card.mana_cost.to_text()) if card.mana_cost else ""
                type_symbol = CARD_TYPE_SYMBOLS.get(card.card_type.name.lower(), "")
                return f"{card_name} - {card.power}/{card.toughness} ({mana_str}, {type_symbol}){token_suffix} attacks"

    except Exception:
        pass

    # Return the original description with token suffix preserved
    return f"{description}{token_suffix}" if token_suffix else description


def _generate_initial_game_section(replay: GameReplay) -> str:
    """Generate the Initial Game section HTML."""
    if not replay.initial_state:
        return ""

    html_parts: list[str] = []
    html_parts.append('<div class="initial-game">')
    html_parts.append("<h2>Initial Game Setup</h2>")

    # Play/Draw box
    on_play = "Player" if replay.player_on_play else "Opponent"
    on_draw = "Opponent" if replay.player_on_play else "Player"
    html_parts.append(f"""
        <div class="play-draw-box">
            <div class="coin">🪙</div>
            <div><strong>{on_play}</strong> wins the coin flip</div>
            <div><strong>{on_play}</strong> chooses to go first (on the play)</div>
            <div style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 0.5rem;">
                {on_draw} will be on the draw (draws on Turn 1)
            </div>
        </div>
    """)

    # Mulligan section
    html_parts.append('<div class="mulligan-section">')

    # Player mulligan
    pm = replay.initial_state.player_mulligan
    html_parts.append('<div class="mulligan-box player-side">')
    html_parts.append("<h3>Player Mulligan</h3>")

    html_parts.append('<div class="hand-label">Opening Hand (7 cards):</div>')
    html_parts.append('<div class="hand-cards">')
    for card in pm.opening_hand:
        # Use automatic lookup - just pass the card name
        html_parts.append(_format_card_html(card.name))
    html_parts.append("</div>")

    if pm.kept:
        html_parts.append('<span class="decision-tag keep">KEEP</span>')
    else:
        html_parts.append('<span class="decision-tag mulligan">MULLIGAN</span>')
        html_parts.append(
            f'<div style="margin-top: 0.75rem; color: var(--text-secondary);">Mulligans taken: {pm.mulligans_taken}</div>'
        )

        # Show new hand after mulligan
        if pm.mulligan_hand:
            html_parts.append(
                f'<div class="hand-label" style="margin-top: 0.75rem;">New Hand After Mulligan ({len(pm.mulligan_hand)} cards):</div>'
            )
            html_parts.append('<div class="hand-cards">')
            for card in pm.mulligan_hand:
                html_parts.append(_format_card_html(card.name))
            html_parts.append("</div>")

        # Show cards returned to bottom of library
        if pm.returned_cards:
            html_parts.append(
                f'<div class="hand-label" style="margin-top: 0.5rem; color: var(--accent-red);">Card(s) Returned to Bottom ({len(pm.returned_cards)}):</div>'
            )
            html_parts.append('<div class="hand-cards returned-cards">')
            for card in pm.returned_cards:
                # Get formatted card display without the span wrapper
                formatted = _format_card_html(card.name)
                # Extract the content between the span tags
                inner_content = formatted.replace('<span class="card-tag">', "").replace(
                    "</span>", ""
                )
                html_parts.append(
                    f'<span class="card-tag returned">{inner_content} → returned</span>'
                )
            html_parts.append("</div>")

        # Show final kept hand
        if pm.new_hand:
            html_parts.append(
                f'<div class="hand-label" style="margin-top: 0.75rem;">Final Kept Hand ({len(pm.new_hand)} cards):</div>'
            )
            html_parts.append('<div class="hand-cards">')
            for card in pm.new_hand:
                html_parts.append(_format_card_html(card.name))
            html_parts.append("</div>")

    html_parts.append("</div>")  # player mulligan box

    # Opponent mulligan (hidden/transparent)
    om = replay.initial_state.opponent_mulligan
    html_parts.append('<div class="mulligan-box opponent-side">')
    html_parts.append("<h3>Opponent Mulligan</h3>")

    html_parts.append('<div class="hand-label">Opening Hand (hidden):</div>')
    html_parts.append('<div class="hand-cards">')
    for card in om.opening_hand:
        # Show cards but with hidden-card styling (semi-transparent)
        html_parts.append(_format_card_html(card.name, hidden=True))
    html_parts.append("</div>")

    if om.kept:
        html_parts.append(
            f'<span class="decision-tag keep">KEEP ({len(om.opening_hand)} cards)</span>'
        )
    else:
        html_parts.append('<span class="decision-tag mulligan">MULLIGAN</span>')

    html_parts.append("</div>")  # opponent mulligan box
    html_parts.append("</div>")  # mulligan-section
    html_parts.append("</div>")  # initial-game

    return "".join(html_parts)


def generate_html_report(
    replay: GameReplay,
    output_path: Path | None = None,
) -> str:
    """Generate an HTML report from a game replay.

    Args:
        replay: The game replay data.
        output_path: Optional path to save the HTML file.

    Returns:
        The generated HTML as a string.

    """
    html_parts: list[str] = []
    html_parts.append(_generate_html_header())

    # Header section
    html_parts.append('<div class="container">')
    html_parts.append("<header>")
    html_parts.append("<h1>MTG-Causal-RL Game Replay</h1>")
    html_parts.append(f'<p style="color: var(--text-secondary)">Game ID: {replay.game_id}</p>')

    # Game info cards
    html_parts.append('<div class="game-info">')

    info_items = [
        ("Player Deck", replay.player_deck),
        ("Opponent Deck", replay.opponent_deck),
        ("Player Agent", replay.player_agent),
        ("Opponent Agent", replay.opponent_agent),
        ("On the Play", "Player" if replay.player_on_play else "Opponent"),
        ("Total Turns", str(replay.total_turns)),
    ]

    for label, value in info_items:
        html_parts.append(f"""
            <div class="info-card">
                <div class="info-label">{label}</div>
                <div class="info-value">{value}</div>
            </div>
        """)

    html_parts.append("</div>")  # game-info

    # Premium Legend Box
    html_parts.append("""
    <div class="legend-box">
        <h3>Icon Reference Guide</h3>
        <div class="legend-grid">
            <div class="legend-category">
                <h4>Card Types</h4>
                <div class="legend-items">
                    <span class="legend-item"><span class="icon">⚔️</span>Creature</span>
                    <span class="legend-item"><span class="icon">✨</span>Instant</span>
                    <span class="legend-item"><span class="icon">🌟</span>Sorcery</span>
                    <span class="legend-item"><span class="icon">🔮</span>Enchantment</span>
                    <span class="legend-item"><span class="icon">⚙️</span>Artifact</span>
                    <span class="legend-item"><span class="icon">🌍</span>Land</span>
                    <span class="legend-item"><span class="icon">👤</span>Planeswalker</span>
                    <span class="legend-item"><span class="icon">🎴</span>Token</span>
                </div>
            </div>
            <div class="legend-category">
                <h4>Mana Colors</h4>
                <div class="legend-items">
                    <span class="legend-item"><span class="icon">🔴</span>Red</span>
                    <span class="legend-item"><span class="icon">🔵</span>Blue</span>
                    <span class="legend-item"><span class="icon">⚪</span>White</span>
                    <span class="legend-item"><span class="icon">⚫</span>Black</span>
                    <span class="legend-item"><span class="icon">🟢</span>Green</span>
                    <span class="legend-item"><span class="icon">◇</span>Colorless</span>
                </div>
            </div>
            <div class="legend-category">
                <h4>Game Zones & Stats</h4>
                <div class="legend-items">
                    <span class="legend-item"><span class="icon">❤️</span>Life Total</span>
                    <span class="legend-item"><span class="icon">🃏</span>Hand</span>
                    <span class="legend-item"><span class="icon">💀</span>Graveyard</span>
                    <span class="legend-item"><span class="icon">✨</span>Exile</span>
                    <span class="legend-item"><span class="icon">📍</span>Tapped</span>
                    <span class="legend-item"><span class="icon">💥</span>Damage</span>
                </div>
            </div>
        </div>
    </div>
    """)

    html_parts.append("</header>")

    # Winner banner
    if replay.winner:
        if replay.winner == "Player":
            winner_class = "player-wins"
            winner_text = "Victory: Player Wins!"
        elif replay.winner == "Opponent":
            winner_class = "opponent-wins"
            winner_text = "Defeat: Opponent Wins!"
        else:
            winner_class = "draw"
            winner_text = "Draw: Turn Limit Reached"
        html_parts.append(f'<div class="winner-banner {winner_class}">')
        html_parts.append(winner_text)
        html_parts.append("</div>")

    # Initial Game Section (after winner banner)
    html_parts.append(_generate_initial_game_section(replay))

    # Timeline
    html_parts.append('<div class="timeline">')

    # Group actions by turn and active player
    actions_by_turn: dict[int, dict[str, list[GameAction]]] = {}
    for action in replay.actions:
        if action.turn not in actions_by_turn:
            actions_by_turn[action.turn] = {"Player": [], "Opponent": []}
        # Use active_player_turn to determine which column to show the action in
        active_turn = action.active_player_turn or action.player
        actions_by_turn[action.turn][active_turn].append(action)

    # Group snapshots by turn
    snapshots_by_turn: dict[int, list[ReplayStateSnapshot]] = {}
    for snap in replay.snapshots:
        if snap.turn not in snapshots_by_turn:
            snapshots_by_turn[snap.turn] = []
        snapshots_by_turn[snap.turn].append(snap)

    # Get turn summaries by turn
    summaries_by_turn = {ts.turn: ts for ts in replay.turn_summaries}

    for turn in sorted(actions_by_turn.keys()):
        turn_actions = actions_by_turn[turn]
        turn_snapshots = snapshots_by_turn.get(turn, [])
        turn_summary = summaries_by_turn.get(turn)
        # Get end-of-turn snapshot for this turn
        end_snapshot = turn_snapshots[-1] if turn_snapshots else None

        html_parts.append('<div class="turn-section">')
        html_parts.append(f'<div class="turn-header">Turn {turn}</div>')
        html_parts.append('<div class="turn-content">')

        # Turn Start Display
        html_parts.append(
            _generate_turn_start_display(turn, snapshots_by_turn.get(turn - 1, []), replay)
        )

        # Determine turn order based on who is on the play
        # When opponent is on the play, show opponent first
        if replay.player_on_play:
            first_player, second_player = "Player", "Opponent"
        else:
            first_player, second_player = "Opponent", "Player"

        # Render both halves of the turn in correct order
        for current_player in [first_player, second_player]:
            is_player_block = current_player == "Player"
            actions_for_player = turn_actions.get(current_player, [])

            # For opponent block, filter to only show actions during opponent's turn
            if not is_player_block:
                actions_for_player = [
                    a for a in actions_for_player if a.active_player_turn == "Opponent"
                ]

            if not actions_for_player:
                continue

            block_class = "player-turn-block" if is_player_block else "opponent-turn-block"
            block_header = "Player's Turn" if is_player_block else "Opponent's Turn"

            html_parts.append(f'<div class="{block_class}">')
            html_parts.append(f'<div class="turn-block-header">{block_header}</div>')

            # Group by phase
            phases_in_order = [
                "Untap",
                "Upkeep",
                "Draw",
                "Main 1",
                "Combat",
                "Combat - Declare Attackers",
                "Combat - Declare Blockers",
                "Combat - Damage",
                "Main 2",
                "End",
                "Cleanup",
            ]
            actions_by_phase: dict[str, list[GameAction]] = {}
            for act in actions_for_player:
                phase_key = act.phase
                if phase_key.lower().startswith("combat") and phase_key not in phases_in_order:
                    phase_key = "Combat"
                if phase_key not in actions_by_phase:
                    actions_by_phase[phase_key] = []
                actions_by_phase[phase_key].append(act)

            for phase in phases_in_order:
                phase_actions = actions_by_phase.get(phase, [])
                if phase == "Combat":
                    continue

                html_parts.append('<div class="phase-section">')
                # Add blockers-phase class for Declare Blockers (inverted colors)
                phase_class = (
                    "phase-header blockers-phase"
                    if phase == "Combat - Declare Blockers"
                    else "phase-header"
                )
                html_parts.append(f'<div class="{phase_class}">{phase}</div>')

                if phase_actions:
                    for act in phase_actions:
                        icon = _action_icon(act.action_type)
                        is_response = act.player != current_player
                        action_class = (
                            "action opponent-action response-action" if is_response else "action"
                        )
                        formatted_desc = _format_action_description(act.description)
                        # Effects already come from CLI with → prefix
                        effects = list(act.effects) if act.effects else []

                        # Only add land text for PLAY_LAND if no effects exist
                        if act.action_type.upper() == "PLAY_LAND" and not effects:
                            land_text = _get_land_text(act.description)
                            if land_text:
                                effects.append(land_text)

                        response_prefix = (
                            f"[{'Opponent' if is_player_block else 'Player'}] "
                            if is_response
                            else ""
                        )
                        # Effects already include → prefix from CLI, don't add another
                        html_parts.append(f"""
                            <div class="{action_class}">
                                <div class="action-icon">{icon}</div>
                                <div class="action-content">
                                    <div class="action-desc">{response_prefix}{formatted_desc}</div>
                                    {"".join(f'<div class="action-effects">{eff}</div>' for eff in effects)}
                                </div>
                            </div>
                        """)
                else:
                    html_parts.append('<div class="no-actions">Pass</div>')

                html_parts.append("</div>")  # phase-section

            html_parts.append("</div>")  # player/opponent-turn-block

        # Turn Summary at end of turn with full board state
        if turn_summary:
            html_parts.append(_generate_turn_summary_html(turn_summary, end_snapshot))

        html_parts.append("</div>")  # turn-content
        html_parts.append("</div>")  # turn-section

    html_parts.append("</div>")  # timeline

    # Final Game Summary
    html_parts.append(_generate_final_game_summary(replay))

    # Footer
    html_parts.append("""
    <footer>
        <p>Generated by <a href="https://github.com/your-repo/mtg-causal-rl">MTG-Causal-RL</a></p>
        <p>A Causal Reinforcement Learning Benchmark for Magic: The Gathering</p>
    </footer>
</body>
</html>
    """)

    html = "".join(html_parts)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")

    return html


def _format_card_cli_style(
    card_name: str, power: int | None = None, toughness: int | None = None
) -> str:
    """Format a card in CLI style: Name - P/T (mana, type) or Name (mana, type).

    Args:
        card_name: Name of the card
        power: Optional power override (for dynamic P/T like Haughty Djinn)
        toughness: Optional toughness override

    Returns:
        Formatted string matching CLI style
    """
    try:
        card = get_card(card_name)
        if not card:
            return card_name

        type_symbol = CARD_TYPE_SYMBOLS.get(card.card_type.name.lower(), "")
        mana_str = format_mana_cost(card.mana_cost.to_text()) if card.mana_cost else ""

        # Use override P/T if provided, else use card's base values
        p = power if power is not None else card.power
        t = toughness if toughness is not None else card.toughness

        # Creatures show P/T
        if is_creature(card) and p is not None:
            if mana_str:
                return f"{card_name} - {p}/{t} ({mana_str}, {type_symbol})"
            else:
                return f"{card_name} - {p}/{t} ({type_symbol})"
        else:
            # Non-creatures
            if mana_str:
                return f"{card_name} ({mana_str}, {type_symbol})"
            else:
                return f"{card_name} ({type_symbol})"
    except Exception:
        return card_name


def _format_hand_items(hand: list, hidden: bool = False) -> str:
    """Format hand cards for display in CLI style."""
    if not hand:
        return '<span class="empty">Empty</span>'

    parts = []
    for c in hand:
        if hasattr(c, "name"):
            name = c.name
        elif isinstance(c, tuple) and len(c) >= 1:
            name = c[0]
        elif isinstance(c, dict):
            name = c.get("name", "?")
        elif isinstance(c, str):
            name = c
        else:
            name = str(c)

        # Format in CLI style
        formatted = _format_card_cli_style(name)
        parts.append(formatted)

    result = ", ".join(parts)
    if hidden:
        return f'<span class="hidden-hand">{result}</span>'
    return result


def _generate_turn_summary_html(
    summary: TurnSummary,
    snapshot: ReplayStateSnapshot | None = None,
) -> str:
    """Generate HTML for a turn summary matching CLI quality with full board state."""

    # Format values with styling
    def format_val(val: int, is_damage: bool = False) -> str:
        if val == 0:
            return '<span class="value neutral">0</span>'
        elif is_damage:
            return f'<span class="value negative">{val}</span>'
        else:
            return f'<span class="value positive">{val}</span>'

    def format_life_change(change: int) -> str:
        if change > 0:
            return f'<span class="life-change positive">(+{change})</span>'
        elif change < 0:
            return f'<span class="life-change negative">({change})</span>'
        return '<span class="life-change neutral">(0)</span>'

    def format_creature_list(
        creatures: list,
        graveyard_instant_sorcery_count: int = 0,
    ) -> str:
        if not creatures:
            return '<span class="empty">None</span>'
        parts = []
        for c in creatures:
            if isinstance(c, dict):
                name = c.get("name", "?")
                p = c.get("power", 0)
                t = c.get("toughness", 0)
                tapped = " 📍" if c.get("tapped") else ""
                attached = c.get("attached_tokens", [])
            else:
                name = str(c)
                p, t = 0, 0
                tapped = ""
                attached = []

            # ALWAYS get proper P/T from card registry as fallback
            try:
                card = get_card(name)
                if card:
                    mana_str = format_mana_cost(card.mana_cost.to_text()) if card.mana_cost else ""
                    # Use actual P/T from dict if non-zero (could be buffed), else card base
                    actual_p = p if p != 0 else (card.power or 0)
                    actual_t = t if t != 0 else (card.toughness or 0)
                    # Haughty Djinn power = instant/sorcery count in graveyard
                    if name == "Haughty Djinn":
                        actual_p = graveyard_instant_sorcery_count
                    # Build token suffix like CLI: [🎴Monster Role]
                    token_suffix = ""
                    if attached:
                        token_strs = [f"🎴{tok}" for tok in attached]
                        token_suffix = f" [{', '.join(token_strs)}]"
                    if mana_str:
                        parts.append(
                            f"{name} - {actual_p}/{actual_t} ({mana_str}, ⚔️){token_suffix}{tapped}"
                        )
                    else:
                        parts.append(f"{name} - {actual_p}/{actual_t} (⚔️){token_suffix}{tapped}")
                else:
                    parts.append(f"{name} - {p}/{t} (⚔️){tapped}")
            except Exception:
                parts.append(f"{name} - {p}/{t} (⚔️){tapped}")
        return ", ".join(parts)

    def format_lands_dict(lands: dict) -> str:
        if not lands:
            return '<span class="empty">None</span>'
        parts = [f"{name} 🌍 x{count}" for name, count in lands.items()]
        return ", ".join(parts)

    def format_graveyard(gy: list) -> str:
        if not gy:
            return '<span class="empty">Empty</span>'
        # Format each card in CLI style and count by type
        parts = []
        type_counts: dict[str, int] = {"instant": 0, "sorcery": 0, "creature": 0}
        for item in gy:
            if isinstance(item, tuple):
                name = item[0]
            elif isinstance(item, dict):
                name = item.get("name", "?")
            else:
                name = str(item)

            # Format card and detect type
            try:
                card = get_card(name)
                if card:
                    card_type = card.card_type.name.lower()
                    if card_type in type_counts:
                        type_counts[card_type] += 1
                    parts.append(_format_card_cli_style(name))
                else:
                    parts.append(name)
            except Exception:
                parts.append(name)

        # Build type summary like CLI: (✨2, 🌟1)
        type_str_parts = []
        if type_counts["instant"] > 0:
            type_str_parts.append(f"✨{type_counts['instant']}")
        if type_counts["sorcery"] > 0:
            type_str_parts.append(f"🌟{type_counts['sorcery']}")
        if type_counts["creature"] > 0:
            type_str_parts.append(f"⚔️{type_counts['creature']}")

        type_summary = f" ({', '.join(type_str_parts)})" if type_str_parts else ""
        return f"{', '.join(parts)}{type_summary}"

    # Build board state sections
    player_board = ""
    opponent_board = ""

    if snapshot:
        # Player board
        player_life = snapshot.player_life
        p_life_pct = max(0, min(100, (player_life / 20) * 100))

        player_board = f"""
        <div class="board-state player-board">
            <h5>🟢 Player Board</h5>
            <div class="life-display">
                <span class="life-icon">❤️</span>
                <span class="life-value">{player_life}</span>
                <div class="life-bar"><div class="life-bar-fill player" style="width: {p_life_pct}%"></div></div>
            </div>
            <div class="board-row"><span class="icon">🌍</span><span class="label">Lands:</span><span class="items">{format_lands_dict(snapshot.player_lands)}</span></div>
            <div class="board-row"><span class="icon">⚔️</span><span class="label">Creatures:</span><span class="items">{format_creature_list(snapshot.player_creatures, snapshot.player_graveyard_instant_sorcery_count)}</span></div>
            <div class="board-row"><span class="icon">🔮</span><span class="label">Enchantments:</span><span class="items empty">None</span></div>
            <div class="board-row"><span class="icon">⚙️</span><span class="label">Artifacts:</span><span class="items empty">None</span></div>
            <div class="board-row"><span class="icon">🎴</span><span class="label">Tokens:</span><span class="items empty">None</span></div>
            <div class="board-row"><span class="icon">🃏</span><span class="label">Hand ({len(snapshot.player_hand) if snapshot.player_hand else 0}):</span><span class="items">{_format_hand_items(snapshot.player_hand, hidden=False)}</span></div>
            <div class="board-row"><span class="icon">💀</span><span class="label">Graveyard:</span><span class="items">{format_graveyard(snapshot.player_graveyard)}</span></div>
            <div class="board-row"><span class="icon">✨</span><span class="label">Exile:</span><span class="items{" empty" if not snapshot.player_exile else ""}">{", ".join(snapshot.player_exile) if snapshot.player_exile else "None"}</span></div>
        </div>
        """

        # Opponent board
        opponent_life = snapshot.opponent_life
        o_life_pct = max(0, min(100, (opponent_life / 20) * 100))
        opp_hand_count = (
            len(snapshot.opponent_hand) if snapshot.opponent_hand else snapshot.opponent_hand_size
        )

        opponent_board = f"""
        <div class="board-state opponent-board">
            <h5>🔴 Opponent Board</h5>
            <div class="life-display">
                <span class="life-icon">❤️</span>
                <span class="life-value">{opponent_life}</span>
                <div class="life-bar"><div class="life-bar-fill opponent" style="width: {o_life_pct}%"></div></div>
            </div>
            <div class="board-row"><span class="icon">🌍</span><span class="label">Lands:</span><span class="items">{format_lands_dict(snapshot.opponent_lands)}</span></div>
            <div class="board-row"><span class="icon">⚔️</span><span class="label">Creatures:</span><span class="items">{format_creature_list(snapshot.opponent_creatures, snapshot.opponent_graveyard_instant_sorcery_count)}</span></div>
            <div class="board-row"><span class="icon">🔮</span><span class="label">Enchantments:</span><span class="items empty">None</span></div>
            <div class="board-row"><span class="icon">⚙️</span><span class="label">Artifacts:</span><span class="items empty">None</span></div>
            <div class="board-row"><span class="icon">🎴</span><span class="label">Tokens:</span><span class="items empty">None</span></div>
            <div class="board-row"><span class="icon">🃏</span><span class="label">Hand ({opp_hand_count}) hidden:</span><span class="items">{_format_hand_items(snapshot.opponent_hand, hidden=True) if snapshot.opponent_hand else f"{opp_hand_count} cards"}</span></div>
            <div class="board-row"><span class="icon">💀</span><span class="label">Graveyard:</span><span class="items">{format_graveyard(snapshot.opponent_graveyard)}</span></div>
            <div class="board-row"><span class="icon">✨</span><span class="label">Exile:</span><span class="items{" empty" if not snapshot.opponent_exile else ""}">{", ".join(snapshot.opponent_exile) if snapshot.opponent_exile else "None"}</span></div>
        </div>
        """

    return f"""
    <div class="turn-summary">
        <h4>Turn {summary.turn} Complete</h4>

        <div class="summary-stats-grid">
            <div class="summary-col player">
                <h5>🟢 Player's Turn</h5>
                <div class="summary-stat">
                    <span class="label">💥 Damage Dealt</span>
                    {format_val(summary.player_damage_dealt, True)}
                </div>
                <div class="summary-stat">
                    <span class="label">✨ Spells Cast</span>
                    {format_val(summary.player_spells_cast)}
                </div>
                <div class="summary-stat">
                    <span class="label">🌍 Lands Played</span>
                    {format_val(summary.player_lands_played)}
                </div>
                <div class="summary-stat">
                    <span class="label">⚔️ Creatures Played</span>
                    {format_val(summary.player_creatures_played)}
                </div>
                <div class="summary-stat">
                    <span class="label">🃏 Cards Drawn</span>
                    {format_val(summary.player_cards_drawn)}
                </div>
            </div>
            <div class="summary-col opponent">
                <h5>🔴 Opponent's Turn</h5>
                <div class="summary-stat">
                    <span class="label">💥 Damage Dealt</span>
                    {format_val(summary.opponent_damage_dealt, True)}
                </div>
                <div class="summary-stat">
                    <span class="label">✨ Spells Cast</span>
                    {format_val(summary.opponent_spells_cast)}
                </div>
                <div class="summary-stat">
                    <span class="label">🌍 Lands Played</span>
                    {format_val(summary.opponent_lands_played)}
                </div>
                <div class="summary-stat">
                    <span class="label">⚔️ Creatures Played</span>
                    {format_val(summary.opponent_creatures_played)}
                </div>
                <div class="summary-stat">
                    <span class="label">🃏 Cards Drawn</span>
                    {format_val(summary.opponent_cards_drawn)}
                </div>
            </div>
        </div>

        <div class="board-states-grid">
            {player_board}
            {opponent_board}
        </div>

        <div class="legend-inline">
            ⚔️=Creatures ✨=Instants 🌟=Sorceries 🔮=Enchantments ⚙️=Artifacts 🌍=Lands 👤=Planeswalkers 🎴=Tokens
        </div>
    </div>
    """


def _generate_turn_start_display(
    turn: int,
    prev_snapshots: list[ReplayStateSnapshot],
    replay: GameReplay,
) -> str:
    """Generate turn start display showing initial state for both players."""
    # Use end-of-prior-turn snapshot, or initial state for turn 1
    snap = prev_snapshots[-1] if prev_snapshots else None

    def format_creatures_for_start(creatures: list) -> str:
        if not creatures:
            return '<span class="empty">None</span>'
        parts = []
        for c in creatures:
            if isinstance(c, dict):
                name = c.get("name", "?")
                p = c.get("power", 0)
                t = c.get("toughness", 0)
            else:
                name = str(c)
                p, t = 0, 0

            # Get proper formatting from card registry - ALWAYS use card P/T if available
            try:
                card = get_card(name)
                if card:
                    mana_str = format_mana_cost(card.mana_cost.to_text()) if card.mana_cost else ""
                    # Use snapshot P/T if non-zero (creature may be buffed), else card base
                    actual_p = p if p != 0 else (card.power or 0)
                    actual_t = t if t != 0 else (card.toughness or 0)
                    if mana_str:
                        parts.append(f"{name} - {actual_p}/{actual_t} ({mana_str}, ⚔️)")
                    else:
                        parts.append(f"{name} - {actual_p}/{actual_t} (⚔️)")
                else:
                    parts.append(f"{name} - {p}/{t} (⚔️)")
            except Exception:
                parts.append(f"{name} - {p}/{t} (⚔️)")
        return ", ".join(parts)

    def format_lands_for_start(lands: dict) -> str:
        if not lands:
            return '<span class="empty">None</span>'
        parts = [f"{name} (🌍) x{count}" for name, count in lands.items()]
        return ", ".join(parts)

    def format_graveyard_for_start(gy: list) -> str:
        if not gy:
            return '<span class="empty">Empty</span>'
        # Format each card in CLI style
        parts = []
        type_counts: dict[str, int] = {"instant": 0, "sorcery": 0, "creature": 0}
        for item in gy:
            if isinstance(item, tuple):
                name = item[0]
            elif isinstance(item, dict):
                name = item.get("name", "?")
            else:
                name = str(item)

            try:
                card = get_card(name)
                if card:
                    card_type = card.card_type.name.lower()
                    if card_type in type_counts:
                        type_counts[card_type] += 1
                    parts.append(_format_card_cli_style(name))
                else:
                    parts.append(name)
            except Exception:
                parts.append(name)

        # Build type summary
        type_str_parts = []
        if type_counts["instant"] > 0:
            type_str_parts.append(f"✨{type_counts['instant']}")
        if type_counts["sorcery"] > 0:
            type_str_parts.append(f"🌟{type_counts['sorcery']}")
        if type_counts["creature"] > 0:
            type_str_parts.append(f"⚔️{type_counts['creature']}")

        type_summary = f" ({', '.join(type_str_parts)})" if type_str_parts else ""
        return f"{', '.join(parts)}{type_summary}"

    # For Turn 1, use initial mulligan hand data
    if turn == 1 and not snap and replay.initial_state:
        # Player's opening hand from mulligan
        pm = replay.initial_state.player_mulligan
        if pm.kept:
            player_hand = pm.opening_hand  # List of CardInfo
        else:
            # Used final kept hand after mulligan
            player_hand = pm.new_hand if pm.new_hand else pm.opening_hand

        # Opponent's opening hand from mulligan
        om = replay.initial_state.opponent_mulligan
        if om.kept:
            opponent_hand = om.opening_hand
        else:
            opponent_hand = om.new_hand if om.new_hand else om.opening_hand

        player_lands = {}
        player_creatures = []
        player_gy = []
        opponent_lands = {}
        opponent_creatures = []
        opponent_gy = []
        opp_hand_count = len(opponent_hand) if opponent_hand else 0
    else:
        # Build the display from snapshot
        player_hand = snap.player_hand if snap else []
        player_lands = snap.player_lands if snap else {}
        player_creatures = snap.player_creatures if snap else []
        player_gy = snap.player_graveyard if snap else []

        opponent_hand = snap.opponent_hand if snap else []
        opponent_lands = snap.opponent_lands if snap else {}
        opponent_creatures = snap.opponent_creatures if snap else []
        opponent_gy = snap.opponent_graveyard if snap else []
        opp_hand_count = (
            len(opponent_hand) if opponent_hand else (snap.opponent_hand_size if snap else 0)
        )

    return f"""
    <div class="turn-start-display">
        <h4>Turn {turn} - Start State</h4>
        <div class="turn-start-grid">
            <div class="turn-start-panel player">
                <h5>🟢 Player - Turn Start</h5>
                <div class="start-row">
                    <span class="category">🃏 Hand ({len(player_hand)})</span>
                    <span class="contents">{_format_hand_items(player_hand, hidden=False)}</span>
                </div>
                <div class="start-row">
                    <span class="category">🌍 Lands ({sum(player_lands.values()) if player_lands else 0})</span>
                    <span class="contents">{format_lands_for_start(player_lands)}</span>
                </div>
                <div class="start-row">
                    <span class="category">⚔️ Creatures ({len(player_creatures)})</span>
                    <span class="contents">{format_creatures_for_start(player_creatures)}</span>
                </div>
                <div class="start-row">
                    <span class="category">💀 Graveyard ({len(player_gy)})</span>
                    <span class="contents">{format_graveyard_for_start(player_gy)}</span>
                </div>
            </div>
            <div class="turn-start-panel opponent">
                <h5>🔴 Opponent - Turn Start</h5>
                <div class="start-row">
                    <span class="category">🃏 Hand ({opp_hand_count}) hidden</span>
                    <span class="contents">{_format_hand_items(opponent_hand, hidden=True) if opponent_hand else f"{opp_hand_count} cards"}</span>
                </div>
                <div class="start-row">
                    <span class="category">🌍 Lands ({sum(opponent_lands.values()) if opponent_lands else 0})</span>
                    <span class="contents">{format_lands_for_start(opponent_lands)}</span>
                </div>
                <div class="start-row">
                    <span class="category">⚔️ Creatures ({len(opponent_creatures)})</span>
                    <span class="contents">{format_creatures_for_start(opponent_creatures)}</span>
                </div>
                <div class="start-row">
                    <span class="category">💀 Graveyard ({len(opponent_gy)})</span>
                    <span class="contents">{format_graveyard_for_start(opponent_gy)}</span>
                </div>
            </div>
        </div>
    </div>
    """


def _get_card_text_for_cast(card_name: str) -> str:
    """Get card rules text for a card being cast."""
    # Card rules text database - only shown when casting
    card_texts = {
        "Play with Fire": "Deal 2 damage to any target. Scry 1.",
        "Lightning Strike": "Deal 3 damage to any target.",
        "Monstrous Rage": "+2/+0 and create a Monster Role attached to this creature.",
        "Monastery Swiftspear": "Haste. Prowess (Whenever you cast a noncreature spell, +1/+1 until end of turn.)",
        "Slickshot Show-Off": "Flying, haste. Prowess. Plot 1R.",
        "Phoenix Chick": "Flying, haste. Can only attack alone. Returns from graveyard when creature with 4+ power attacks.",
        "Heartfire Hero": "When this creature dies, deal damage equal to its power to any target.",
        "Haughty Djinn": "Flying. Power equal to instants/sorceries in your graveyard. Spells cost 1 less.",
        "No More Lies": "Counter target spell. Its controller may pay 3 to prevent this. Exile if countered.",
        "Make Disappear": "Counter target spell unless its controller pays 2. Casualty 1.",
        "Memory Deluge": "Look at top 4, put 2 in hand. Flashback 5UU for top 7.",
        "Get Lost": "Exile target artifact, creature, enchantment, or planeswalker. Its controller creates 2 Map tokens.",
        "Sunfall": "Exile all creatures. Create an Incubator token with +1/+1 counters for each exiled creature.",
    }

    for name, text in card_texts.items():
        if name.lower() in card_name.lower():
            return text
    return ""


def _format_mana_symbols_in_text(text: str) -> str:
    """Convert mana symbols like {W}{U} to ⚪🔵 in text."""
    mana_mapping = {
        "{W}": "⚪",
        "{U}": "🔵",
        "{B}": "⚫",
        "{R}": "🔴",
        "{G}": "🟢",
        "{C}": "◇",
        "{1}": "(1)",
        "{2}": "(2)",
        "{3}": "(3)",
        "{4}": "(4)",
        "{5}": "(5)",
    }
    for symbol, icon in mana_mapping.items():
        text = text.replace(symbol, icon)
    return text


def _get_land_text(land_name: str) -> str:
    """Get special text for lands when played."""
    land_texts = {
        "Restless Anchorage": "Enters tapped. ⚪🔵: Become 2/3 creature until end of turn.",
        "Restless Cottage": "Enters tapped. ⚫🟢: Become 4/4 creature until end of turn.",
        "Restless Ridgeline": "Enters tapped. 🔴🟢: Become 3/4 creature until end of turn.",
        "Restless Reef": "Enters tapped. 🔵⚫: Become 4/3 creature until end of turn.",
        "Restless Vents": "Enters tapped. 🔴⚪: Become 2/3 creature until end of turn.",
    }

    for name, text in land_texts.items():
        if name.lower() in land_name.lower():
            return text
    return ""


def _generate_final_game_summary(replay: GameReplay) -> str:
    """Generate final game summary with full board state."""
    # Get final snapshot if available
    final_snap = replay.snapshots[-1] if replay.snapshots else None

    # Calculate totals from turn summaries
    total_player_damage = sum(ts.player_damage_dealt for ts in replay.turn_summaries)
    total_opponent_damage = sum(ts.opponent_damage_dealt for ts in replay.turn_summaries)
    total_player_spells = sum(ts.player_spells_cast for ts in replay.turn_summaries)
    total_opponent_spells = sum(ts.opponent_spells_cast for ts in replay.turn_summaries)
    total_player_lands = sum(ts.player_lands_played for ts in replay.turn_summaries)
    total_opponent_lands = sum(ts.opponent_lands_played for ts in replay.turn_summaries)
    total_player_creatures = sum(ts.player_creatures_played for ts in replay.turn_summaries)
    total_opponent_creatures = sum(ts.opponent_creatures_played for ts in replay.turn_summaries)

    # Winner info - be explicit
    if replay.winner == "Player":
        result_icon = "🎉"
        result_text = "PLAYER WINS"
        subtitle = (
            f"{replay.player_agent} playing {replay.player_deck} defeats {replay.opponent_agent}"
        )
    elif replay.winner == "Opponent":
        result_icon = "💀"
        result_text = "OPPONENT WINS"
        subtitle = (
            f"{replay.opponent_agent} playing {replay.opponent_deck} defeats {replay.player_agent}"
        )
    else:
        result_icon = "⏱️"
        result_text = "DRAW - Turn Limit Reached"
        subtitle = f"Game ended after {replay.total_turns} turns"

    def format_creatures_final(creatures: list, graveyard_instant_sorcery_count: int = 0) -> str:
        if not creatures:
            return '<span class="empty">None</span>'
        parts = []
        for c in creatures:
            if isinstance(c, dict):
                name = c.get("name", "?")
                p = c.get("power", 0)
                t = c.get("toughness", 0)
                tapped = " 📍" if c.get("tapped") else ""
                attached = c.get("attached_tokens", [])
            else:
                name = str(c)
                p, t = 0, 0
                tapped = ""
                attached = []

            # ALWAYS get proper P/T and mana cost from card registry
            try:
                card = get_card(name)
                if card:
                    mana_str = format_mana_cost(card.mana_cost.to_text()) if card.mana_cost else ""
                    actual_p = p if p != 0 else (card.power or 0)
                    actual_t = t if t != 0 else (card.toughness or 0)
                    # Haughty Djinn power = instant/sorcery count in graveyard
                    if name == "Haughty Djinn":
                        actual_p = graveyard_instant_sorcery_count
                    # Build token suffix like CLI: [🎴Monster Role]
                    token_suffix = ""
                    if attached:
                        token_strs = [f"🎴{tok}" for tok in attached]
                        token_suffix = f" [{', '.join(token_strs)}]"
                    if mana_str:
                        parts.append(
                            f"{name} - {actual_p}/{actual_t} ({mana_str}, ⚔️){token_suffix}{tapped}"
                        )
                    else:
                        parts.append(f"{name} - {actual_p}/{actual_t} (⚔️){token_suffix}{tapped}")
                else:
                    parts.append(f"{name} - {p}/{t} (⚔️){tapped}")
            except Exception:
                parts.append(f"{name} - {p}/{t} (⚔️){tapped}")

        return ", ".join(parts) if parts else '<span class="empty">None</span>'

    def format_lands_final(lands: dict) -> str:
        if not lands:
            return '<span class="empty">None</span>'
        parts = [f"{name} (🌍) x{count}" for name, count in lands.items()]
        return ", ".join(parts)

    def format_graveyard_final(gy: list) -> str:
        if not gy:
            return '<span class="empty">Empty</span>'
        # Format each card in CLI style
        parts = []
        type_counts: dict[str, int] = {"instant": 0, "sorcery": 0, "creature": 0}
        for item in gy:
            if isinstance(item, tuple):
                name = item[0]
            elif isinstance(item, dict):
                name = item.get("name", "?")
            else:
                name = str(item)

            try:
                card = get_card(name)
                if card:
                    card_type = card.card_type.name.lower()
                    if card_type in type_counts:
                        type_counts[card_type] += 1
                    parts.append(_format_card_cli_style(name))
                else:
                    parts.append(name)
            except Exception:
                parts.append(name)

        # Build type summary
        type_str_parts = []
        if type_counts["instant"] > 0:
            type_str_parts.append(f"✨{type_counts['instant']}")
        if type_counts["sorcery"] > 0:
            type_str_parts.append(f"🌟{type_counts['sorcery']}")
        if type_counts["creature"] > 0:
            type_str_parts.append(f"⚔️{type_counts['creature']}")

        type_summary = f" ({', '.join(type_str_parts)})" if type_str_parts else ""
        return f"{', '.join(parts)}{type_summary}"

    # Player final state
    player_lands = final_snap.player_lands if final_snap else {}
    player_creatures = final_snap.player_creatures if final_snap else []
    player_hand_count = len(final_snap.player_hand) if final_snap and final_snap.player_hand else 0
    player_gy = final_snap.player_graveyard if final_snap else []
    player_exile = final_snap.player_exile if final_snap else []
    player_life = final_snap.player_life if final_snap else 20
    player_gy_count = final_snap.player_graveyard_instant_sorcery_count if final_snap else 0

    # Opponent final state
    opp_lands = final_snap.opponent_lands if final_snap else {}
    opp_creatures = final_snap.opponent_creatures if final_snap else []
    opp_gy_count = final_snap.opponent_graveyard_instant_sorcery_count if final_snap else 0
    opp_hand_count = (
        len(final_snap.opponent_hand)
        if final_snap and final_snap.opponent_hand
        else (final_snap.opponent_hand_size if final_snap else 0)
    )
    opp_gy = final_snap.opponent_graveyard if final_snap else []
    opp_exile = final_snap.opponent_exile if final_snap else []
    opp_life = final_snap.opponent_life if final_snap else 20

    # Life bar widths (percentage, handle negative life)
    player_life_pct = max(0, min(100, (max(0, player_life) / 20) * 100))
    opp_life_pct = max(0, min(100, (max(0, opp_life) / 20) * 100))

    # Format hand contents
    player_hand_display = (
        _format_hand_items(final_snap.player_hand, hidden=False)
        if final_snap and final_snap.player_hand
        else f"{player_hand_count} cards"
    )
    opp_hand_display = (
        _format_hand_items(final_snap.opponent_hand, hidden=True)
        if final_snap and final_snap.opponent_hand
        else f"{opp_hand_count} cards"
    )

    return f"""
    <div class="final-game-summary">
        <h2>{result_icon} GAME OVER - {result_text} {result_icon}</h2>
        <div class="result-subtitle">{subtitle}</div>

        <div class="final-stats-grid">
            <div class="final-player-panel player {"winner" if replay.winner == "Player" else ""}">
                <h3>🟢 Player ({replay.player_agent})</h3>
                <div class="deck-name">{replay.player_deck}</div>
                <div class="final-life-display player">❤️ {player_life}</div>
                <div class="life-bar-container">
                    <div class="life-bar">
                        <div class="life-bar-fill player" style="width: {player_life_pct}%"></div>
                    </div>
                </div>

                <div class="final-board-state">
                    <div class="final-stat-row">
                        <span class="stat-label">💥 Total Damage Dealt</span>
                        <span class="stat-value {"positive" if total_player_damage > 0 else ""}">{total_player_damage}</span>
                    </div>
                    <div class="final-stat-row">
                        <span class="stat-label">✨ Total Spells Cast</span>
                        <span class="stat-value">{total_player_spells}</span>
                    </div>
                    <div class="final-stat-row">
                        <span class="stat-label">🌍 Lands Played</span>
                        <span class="stat-value">{total_player_lands}</span>
                    </div>
                    <div class="final-stat-row">
                        <span class="stat-label">⚔️ Creatures Played</span>
                        <span class="stat-value">{total_player_creatures}</span>
                    </div>

                    <div class="final-divider"></div>

                    <div class="final-board-row">
                        <span class="icon">🌍</span>
                        <span class="label">Lands:</span>
                        <span class="items">{format_lands_final(player_lands)}</span>
                    </div>
                    <div class="final-board-row">
                        <span class="icon">⚔️</span>
                        <span class="label">Creatures:</span>
                        <span class="items">{format_creatures_final(player_creatures, player_gy_count)}</span>
                    </div>
                    <div class="final-board-row">
                        <span class="icon">🔮</span>
                        <span class="label">Enchantments:</span>
                        <span class="items empty">None</span>
                    </div>
                    <div class="final-board-row">
                        <span class="icon">⚙️</span>
                        <span class="label">Artifacts:</span>
                        <span class="items empty">None</span>
                    </div>
                    <div class="final-board-row">
                        <span class="icon">🎴</span>
                        <span class="label">Tokens:</span>
                        <span class="items empty">None</span>
                    </div>
                    <div class="final-board-row">
                        <span class="icon">🃏</span>
                        <span class="label">Hand ({player_hand_count}):</span>
                        <span class="items">{player_hand_display}</span>
                    </div>
                    <div class="final-board-row">
                        <span class="icon">💀</span>
                        <span class="label">Graveyard:</span>
                        <span class="items">{format_graveyard_final(player_gy)}</span>
                    </div>
                    <div class="final-board-row">
                        <span class="icon">✨</span>
                        <span class="label">Exile:</span>
                        <span class="items{" empty" if not player_exile else ""}">{", ".join(player_exile) if player_exile else "None"}</span>
                    </div>
                </div>
            </div>

            <div class="final-player-panel opponent {"winner" if replay.winner == "Opponent" else ""}">
                <h3>🔴 Opponent ({replay.opponent_agent})</h3>
                <div class="deck-name">{replay.opponent_deck}</div>
                <div class="final-life-display opponent">❤️ {opp_life}</div>
                <div class="life-bar-container">
                    <div class="life-bar">
                        <div class="life-bar-fill opponent" style="width: {opp_life_pct}%"></div>
                    </div>
                </div>

                <div class="final-board-state">
                    <div class="final-stat-row">
                        <span class="stat-label">💥 Total Damage Dealt</span>
                        <span class="stat-value {"positive" if total_opponent_damage > 0 else ""}">{total_opponent_damage}</span>
                    </div>
                    <div class="final-stat-row">
                        <span class="stat-label">✨ Total Spells Cast</span>
                        <span class="stat-value">{total_opponent_spells}</span>
                    </div>
                    <div class="final-stat-row">
                        <span class="stat-label">🌍 Lands Played</span>
                        <span class="stat-value">{total_opponent_lands}</span>
                    </div>
                    <div class="final-stat-row">
                        <span class="stat-label">⚔️ Creatures Played</span>
                        <span class="stat-value">{total_opponent_creatures}</span>
                    </div>

                    <div class="final-divider"></div>

                    <div class="final-board-row">
                        <span class="icon">🌍</span>
                        <span class="label">Lands:</span>
                        <span class="items">{format_lands_final(opp_lands)}</span>
                    </div>
                    <div class="final-board-row">
                        <span class="icon">⚔️</span>
                        <span class="label">Creatures:</span>
                        <span class="items">{format_creatures_final(opp_creatures, opp_gy_count)}</span>
                    </div>
                    <div class="final-board-row">
                        <span class="icon">🔮</span>
                        <span class="label">Enchantments:</span>
                        <span class="items empty">None</span>
                    </div>
                    <div class="final-board-row">
                        <span class="icon">⚙️</span>
                        <span class="label">Artifacts:</span>
                        <span class="items empty">None</span>
                    </div>
                    <div class="final-board-row">
                        <span class="icon">🎴</span>
                        <span class="label">Tokens:</span>
                        <span class="items empty">None</span>
                    </div>
                    <div class="final-board-row">
                        <span class="icon">🃏</span>
                        <span class="label">Hand ({opp_hand_count}) hidden:</span>
                        <span class="items">{opp_hand_display}</span>
                    </div>
                    <div class="final-board-row">
                        <span class="icon">💀</span>
                        <span class="label">Graveyard:</span>
                        <span class="items">{format_graveyard_final(opp_gy)}</span>
                    </div>
                    <div class="final-board-row">
                        <span class="icon">✨</span>
                        <span class="label">Exile:</span>
                        <span class="items{" empty" if not opp_exile else ""}">{", ".join(opp_exile) if opp_exile else "None"}</span>
                    </div>
                </div>
            </div>
        </div>

        <div class="final-game-footer">
            <div class="game-stats-row">
                <span>📊 Total Turns: <strong>{replay.total_turns}</strong></span>
                <span class="divider">│</span>
                <span>🎴 Player Deck: <strong class="player-color">{replay.player_deck}</strong></span>
                <span class="divider">│</span>
                <span>🎴 Opponent Deck: <strong class="opponent-color">{replay.opponent_deck}</strong></span>
            </div>
        </div>
    </div>
    """


def save_replay_json(
    replay: GameReplay,
    output_path: Path,
) -> None:
    """Save replay data as JSON for later processing.

    Args:
        replay: The game replay data.
        output_path: Path to save the JSON file.

    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict for JSON serialization
    initial_state_dict = None
    if replay.initial_state:
        initial_state_dict = {
            "player_on_play": replay.initial_state.player_on_play,
            "player_mulligan": {
                "opening_hand": [
                    {"name": c.name, "mana_cost": c.mana_cost}
                    for c in replay.initial_state.player_mulligan.opening_hand
                ],
                "kept": replay.initial_state.player_mulligan.kept,
                "new_hand": [
                    {"name": c.name, "mana_cost": c.mana_cost}
                    for c in replay.initial_state.player_mulligan.new_hand
                ],
                "mulligans_taken": replay.initial_state.player_mulligan.mulligans_taken,
            },
            "opponent_mulligan": {
                "opening_hand": [
                    {"name": c.name, "mana_cost": c.mana_cost}
                    for c in replay.initial_state.opponent_mulligan.opening_hand
                ],
                "kept": replay.initial_state.opponent_mulligan.kept,
                "new_hand": [
                    {"name": c.name, "mana_cost": c.mana_cost}
                    for c in replay.initial_state.opponent_mulligan.new_hand
                ],
                "mulligans_taken": replay.initial_state.opponent_mulligan.mulligans_taken,
            },
        }

    data = {
        "game_id": replay.game_id,
        "timestamp": replay.timestamp,
        "player_deck": replay.player_deck,
        "opponent_deck": replay.opponent_deck,
        "player_agent": replay.player_agent,
        "opponent_agent": replay.opponent_agent,
        "player_on_play": replay.player_on_play,
        "winner": replay.winner,
        "total_turns": replay.total_turns,
        "initial_state": initial_state_dict,
        "metadata": replay.metadata,
        "actions": [
            {
                "turn": a.turn,
                "phase": a.phase,
                "player": a.player,
                "action_type": a.action_type,
                "description": a.description,
                "active_player_turn": a.active_player_turn,
                "effects": a.effects,
                "state_changes": a.state_changes,
            }
            for a in replay.actions
        ],
        "snapshots": [
            {
                "turn": s.turn,
                "phase": s.phase,
                "active_player": s.active_player,
                "player_life": s.player_life,
                "opponent_life": s.opponent_life,
                "player_hand": [{"name": c.name, "mana_cost": c.mana_cost} for c in s.player_hand],
                "opponent_hand_size": s.opponent_hand_size,
                "player_lands": s.player_lands,
                "opponent_lands": s.opponent_lands,
                "player_mana": s.player_mana,
                "opponent_mana": s.opponent_mana,
                "board_power": s.board_power,
                "opponent_power": s.opponent_power,
                "player_creatures": s.player_creatures,
                "opponent_creatures": s.opponent_creatures,
                "player_tokens": s.player_tokens,
                "opponent_tokens": s.opponent_tokens,
                "player_graveyard": s.player_graveyard,
                "opponent_graveyard": s.opponent_graveyard,
            }
            for s in replay.snapshots
        ],
        "turn_summaries": [
            {
                "turn": ts.turn,
                "player_damage_dealt": ts.player_damage_dealt,
                "opponent_damage_dealt": ts.opponent_damage_dealt,
                "player_spells_cast": ts.player_spells_cast,
                "opponent_spells_cast": ts.opponent_spells_cast,
                "player_lands_played": ts.player_lands_played,
                "opponent_lands_played": ts.opponent_lands_played,
                "player_creatures_played": ts.player_creatures_played,
                "opponent_creatures_played": ts.opponent_creatures_played,
                "player_cards_drawn": ts.player_cards_drawn,
                "opponent_cards_drawn": ts.opponent_cards_drawn,
            }
            for ts in replay.turn_summaries
        ],
    }

    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_replay_json(input_path: Path) -> GameReplay:
    """Load replay data from JSON.

    Args:
        input_path: Path to the JSON file.

    Returns:
        Loaded GameReplay object.

    """
    data = json.loads(input_path.read_text(encoding="utf-8"))

    # Load initial state
    initial_state = None
    if data.get("initial_state"):
        ist = data["initial_state"]
        initial_state = InitialGameState(
            player_on_play=ist["player_on_play"],
            player_mulligan=MulliganInfo(
                opening_hand=[
                    CardInfo(name=c["name"], mana_cost=c.get("mana_cost", ""))
                    for c in ist["player_mulligan"]["opening_hand"]
                ],
                kept=ist["player_mulligan"]["kept"],
                new_hand=[
                    CardInfo(name=c["name"], mana_cost=c.get("mana_cost", ""))
                    for c in ist["player_mulligan"].get("new_hand", [])
                ],
                mulligans_taken=ist["player_mulligan"].get("mulligans_taken", 0),
            ),
            opponent_mulligan=MulliganInfo(
                opening_hand=[
                    CardInfo(name=c["name"], mana_cost=c.get("mana_cost", ""))
                    for c in ist["opponent_mulligan"]["opening_hand"]
                ],
                kept=ist["opponent_mulligan"]["kept"],
                new_hand=[
                    CardInfo(name=c["name"], mana_cost=c.get("mana_cost", ""))
                    for c in ist["opponent_mulligan"].get("new_hand", [])
                ],
                mulligans_taken=ist["opponent_mulligan"].get("mulligans_taken", 0),
            ),
        )

    actions = [
        GameAction(
            turn=a["turn"],
            phase=a["phase"],
            player=a["player"],
            action_type=a["action_type"],
            description=a["description"],
            active_player_turn=a.get("active_player_turn", a["player"]),
            effects=a.get("effects", []),
            state_changes=a.get("state_changes", {}),
        )
        for a in data.get("actions", [])
    ]

    snapshots = [
        ReplayStateSnapshot(
            turn=s["turn"],
            phase=s["phase"],
            active_player=s["active_player"],
            player_life=s["player_life"],
            opponent_life=s["opponent_life"],
            player_hand=[
                CardInfo(name=c["name"], mana_cost=c.get("mana_cost", ""))
                for c in s.get("player_hand", [])
            ],
            opponent_hand_size=s.get("opponent_hand_size", 0),
            player_lands=s.get("player_lands", {}),
            opponent_lands=s.get("opponent_lands", {}),
            player_mana=s.get("player_mana", {}),
            opponent_mana=s.get("opponent_mana", {}),
            board_power=s.get("board_power", 0),
            opponent_power=s.get("opponent_power", 0),
            player_creatures=s.get("player_creatures", []),
            opponent_creatures=s.get("opponent_creatures", []),
            player_tokens=s.get("player_tokens", []),
            opponent_tokens=s.get("opponent_tokens", []),
            player_graveyard=s.get("player_graveyard", []),
            opponent_graveyard=s.get("opponent_graveyard", []),
        )
        for s in data.get("snapshots", [])
    ]

    turn_summaries = [
        TurnSummary(
            turn=ts["turn"],
            player_damage_dealt=ts.get("player_damage_dealt", 0),
            opponent_damage_dealt=ts.get("opponent_damage_dealt", 0),
            player_spells_cast=ts.get("player_spells_cast", 0),
            opponent_spells_cast=ts.get("opponent_spells_cast", 0),
            player_lands_played=ts.get("player_lands_played", 0),
            opponent_lands_played=ts.get("opponent_lands_played", 0),
            player_creatures_played=ts.get("player_creatures_played", 0),
            opponent_creatures_played=ts.get("opponent_creatures_played", 0),
            player_cards_drawn=ts.get("player_cards_drawn", 0),
            opponent_cards_drawn=ts.get("opponent_cards_drawn", 0),
        )
        for ts in data.get("turn_summaries", [])
    ]

    return GameReplay(
        game_id=data["game_id"],
        timestamp=data["timestamp"],
        player_deck=data["player_deck"],
        opponent_deck=data["opponent_deck"],
        player_agent=data["player_agent"],
        opponent_agent=data["opponent_agent"],
        player_on_play=data["player_on_play"],
        winner=data["winner"],
        total_turns=data["total_turns"],
        initial_state=initial_state,
        actions=actions,
        snapshots=snapshots,
        turn_summaries=turn_summaries,
        metadata=data.get("metadata", {}),
    )

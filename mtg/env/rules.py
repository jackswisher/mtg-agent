"""MTG game rules implementation with priority system.

This module contains the core game rules and state management for the
MTG environment, including a simplified stack-based priority system
for instant-speed spell casting.
"""

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from mtg.env.card_definitions import (
    Card,
    CardRegistry,
    CardType,
    Keyword,
    ManaColor,
    ManaCost,
    Trigger,
    TriggerEffect,
    TriggerType,
)


class GamePhase(Enum):
    """Game phases including priority windows.

    Priority windows allow players to cast instants and activated abilities.
    """

    MULLIGAN = auto()
    UNTAP = auto()
    UPKEEP = auto()  # Priority window
    DRAW = auto()
    MAIN_PRECOMBAT = auto()  # Priority window (Main Phase 1)
    COMBAT_BEGIN = auto()  # Priority window
    COMBAT_ATTACKERS = auto()  # Priority window after attackers declared
    COMBAT_BLOCKERS = auto()  # Priority window after blockers declared
    COMBAT_DAMAGE = auto()
    MAIN_POSTCOMBAT = auto()  # Priority window (Main Phase 2)
    END_STEP = auto()  # Priority window
    CLEANUP = auto()
    GAME_OVER = auto()


class ActionType(Enum):
    """Types of actions a player can take."""

    KEEP_HAND = auto()
    MULLIGAN = auto()
    PLAY_LAND = auto()
    CAST_CREATURE = auto()
    CAST_SORCERY = auto()
    CAST_INSTANT = auto()  # Can be done at instant speed
    CAST_ENCHANTMENT = auto()
    ACTIVATE_ABILITY = auto()  # Instant speed
    ATTACK = auto()
    ATTACK_WITH = auto()  # Attack with specific creature
    BLOCK = auto()
    BLOCK_WITH = auto()  # Block with specific creature
    PASS_PRIORITY = auto()  # Pass priority to opponent
    PASS = auto()  # End phase / pass turn


# Phases where instants can be cast (priority windows)
PRIORITY_PHASES = {
    GamePhase.UPKEEP,
    GamePhase.DRAW,
    GamePhase.MAIN_PRECOMBAT,
    GamePhase.COMBAT_BEGIN,
    GamePhase.COMBAT_ATTACKERS,
    GamePhase.COMBAT_BLOCKERS,
    GamePhase.MAIN_POSTCOMBAT,
    GamePhase.END_STEP,
}

# Phases where sorcery-speed spells can be cast (only active player's main phase)
SORCERY_SPEED_PHASES = {
    GamePhase.MAIN_PRECOMBAT,
    GamePhase.MAIN_POSTCOMBAT,
}


@dataclass
class StackItem:
    """An item on the stack waiting to resolve.

    Attributes:
        source: The card/ability source.
        controller: Player index who controls this.
        targets: Target references.
        item_type: Type of stack item.
        is_flashback: If True, exile instead of graveyard after resolving.
        is_kicked: If True, the spell was kicked and gets bonus effects.
        x_value: For X spells, the value of X.

    """

    source: Card
    controller: int
    targets: list[Any] = field(default_factory=list)
    item_type: str = "spell"  # "spell", "ability", "trigger"
    is_flashback: bool = False
    is_kicked: bool = False
    x_value: int = 0


@dataclass(frozen=True)
class TargetRef:
    """Represents a target choice for a spell or ability."""

    kind: str  # "player", "creature", "permanent", "stack"
    ref_id: int
    name: str


@dataclass
class PlayerState:
    """Represents a player's game state."""

    life: int = 20
    deck: list[Card] = field(default_factory=list)
    hand: list[Card] = field(default_factory=list)
    battlefield: list[Card] = field(default_factory=list)
    graveyard: list[Card] = field(default_factory=list)
    exile: list[Card] = field(default_factory=list)  # Exile zone

    # Mana pool
    mana_pool: dict[str, int] = field(
        default_factory=lambda: {"W": 0, "U": 0, "B": 0, "R": 0, "G": 0, "C": 0}
    )

    # Game state tracking
    lands_played_this_turn: int = 0
    has_drawn_for_turn: bool = False
    spells_cast_this_turn: int = 0  # For storm, etc.
    failed_draw: bool = False  # Tried to draw from empty library → lose

    # Creature state
    tapped_permanents: set[int] = field(default_factory=set)
    summoning_sick: set[int] = field(default_factory=set)
    # Lands or permanents temporarily animated into creatures
    activated_creatures: dict[int, tuple[int, int]] = field(default_factory=dict)

    # Combat state
    declared_attackers: list[int] = field(default_factory=list)
    declared_blockers: dict[int, int] = field(default_factory=dict)  # blocker_id -> attacker_id
    # Multi-blocker tracking: attacker_id -> list of blocker_ids
    multi_blockers: dict[int, list[int]] = field(default_factory=dict)

    # Domain count (number of basic land types controlled)
    domain_count: int = 0

    def reset_for_turn(self) -> None:
        """Reset per-turn state."""
        self.lands_played_this_turn = 0
        self.has_drawn_for_turn = False
        self.spells_cast_this_turn = 0
        self.mana_pool = {"W": 0, "U": 0, "B": 0, "R": 0, "G": 0, "C": 0}
        self.declared_attackers.clear()
        self.declared_blockers.clear()
        self.multi_blockers.clear()
        # Reset crewed vehicles
        for card in self.battlefield:
            if hasattr(card, "is_crewed"):
                card.is_crewed = False

    def update_domain_count(self) -> None:
        """Update domain count based on basic land types controlled."""
        basic_types = set()
        for card in self.battlefield:
            if card.land_props:
                for land_type in card.land_props.basic_land_types:
                    basic_types.add(land_type)
        self.domain_count = len(basic_types)

    def exile_card(self, card: Card) -> None:
        """Move a card to exile."""
        self.exile.append(card)

    def untap_all(self) -> None:
        """Untap all permanents."""
        self.tapped_permanents.clear()
        # Remove summoning sickness from creatures that survived a turn
        self.summoning_sick.clear()

    def draw_card(self) -> Card | None:
        """Draw a card from the deck.

        Sets ``failed_draw`` if the library is empty, which triggers a
        game loss via ``check_game_over``.
        """
        if not self.deck:
            self.failed_draw = True
            return None
        card = self.deck.pop(0)
        self.hand.append(card)
        return card

    def draw_cards(self, n: int) -> list[Card]:
        """Draw n cards from the deck."""
        cards = []
        for _ in range(n):
            card = self.draw_card()
            if card:
                cards.append(card)
        return cards

    def shuffle_deck(self, rng: random.Random) -> None:
        """Shuffle the deck."""
        rng.shuffle(self.deck)

    def get_available_mana(self) -> dict[str, int]:
        """Get mana that can be produced from untapped lands/creatures.

        Each mana source contributes at most ONE mana.  For duals that
        can produce multiple colours, the *first* colour in their
        ``produces_mana`` list is counted (a conservative heuristic).
        The actual mana-payment engine selects sources optimally.
        """
        available: dict[str, int] = {"W": 0, "U": 0, "B": 0, "R": 0, "G": 0, "C": 0}

        for card in self.battlefield:
            if card.card_id in self.tapped_permanents:
                continue
            if card.produces_mana:
                # Each source taps for exactly one mana; pick the first colour.
                available[card.produces_mana[0].value] += 1

        return available

    def get_total_available_mana(self) -> int:
        """Get total mana available."""
        return sum(
            1
            for card in self.battlefield
            if card.produces_mana and card.card_id not in self.tapped_permanents
        )

    def tap_permanent(self, card_id: int) -> bool:
        """Tap a permanent."""
        for card in self.battlefield:
            if card.card_id == card_id and card_id not in self.tapped_permanents:
                self.tapped_permanents.add(card_id)
                return True
        return False

    def get_card_by_id(self, card_id: int) -> Card | None:
        """Find a card by ID on battlefield."""
        for card in self.battlefield:
            if card.card_id == card_id:
                return card
        return None

    def can_attack_with(self, card: Card) -> bool:
        """Check if a creature can attack."""
        if card.card_type != CardType.CREATURE and card.card_id not in self.activated_creatures:
            return False
        if card.card_id in self.tapped_permanents:
            return False
        if card.name == "Topiary Stomper":
            land_count = sum(1 for c in self.battlefield if c.produces_mana)
            if land_count < 7:
                return False
        if card.card_id in self.summoning_sick and not card.has_haste:
            return False
        return not (card.rules_text and "can't attack" in card.rules_text.lower())

    def can_block_with(self, card: Card) -> bool:
        """Check if a creature can block."""
        if card.card_type != CardType.CREATURE and card.card_id not in self.activated_creatures:
            return False
        if card.card_id in self.tapped_permanents:
            return False
        if card.name == "Topiary Stomper":
            land_count = sum(1 for c in self.battlefield if c.produces_mana)
            if land_count < 7:
                return False
        return not (card.rules_text and "can't block" in card.rules_text.lower())


@dataclass
class GameAction:
    """Represents a logged game action."""

    player: int  # 0 = player, 1 = opponent
    action_type: str  # "PLAY_LAND", "CAST", "ATTACK", "BLOCK", "DRAW", etc.
    card_name: str = ""
    phase: str = ""
    turn: int = 0
    details: dict = field(default_factory=dict)
    active_player: int = 0


@dataclass
class TriggerEvent:
    """Represents a triggered ability event.

    Attributes:
        trigger: The trigger definition from the card.
        source_card: The card that has the trigger.
        controller: Player index who controls the source (0 or 1).
        affected_cards: Cards affected by the trigger.
        effect_applied: Description of what happened.

    """

    trigger: Trigger
    source_card: Card
    controller: int
    affected_cards: list[Card] = field(default_factory=list)
    effect_applied: str = ""
    damage: int | None = None
    target_player_idx: int | None = None
    target_card_id: int | None = None


class TriggerEngine:
    """Manages triggered abilities in the game.

    This engine detects when triggers should fire and applies their effects.
    """

    @staticmethod
    def get_triggers_for_event(
        state: "GameState",
        trigger_type: TriggerType,
        player_idx: int,
    ) -> list[tuple[Card, Trigger]]:
        """Find all triggers of a given type for a player.

        Args:
            state: Current game state.
            trigger_type: The type of trigger to look for.
            player_idx: Which player's permanents to check.

        Returns:
            List of (card, trigger) tuples that should fire.

        """
        triggers_found: list[tuple[Card, Trigger]] = []
        player = state.players[player_idx]

        for card in player.battlefield:
            for trigger in card.triggers:
                if trigger.trigger_type == trigger_type:
                    triggers_found.append((card, trigger))

        return triggers_found

    @staticmethod
    def apply_trigger(
        state: "GameState",
        card: Card,
        trigger: Trigger,
        controller: int,
    ) -> tuple["GameState", TriggerEvent]:
        """Apply a trigger's effect to the game state.

        Args:
            state: Current game state.
            card: The card with the trigger.
            trigger: The trigger to apply.
            controller: Player index who controls the trigger source.

        Returns:
            Tuple of (updated state, trigger event).

        """
        event = TriggerEvent(
            trigger=trigger,
            source_card=card,
            controller=controller,
        )

        player = state.players[controller]
        opponent = state.players[1 - controller]

        # Apply effect based on type
        if trigger.effect == TriggerEffect.POWER_TOUGHNESS_BOOST:
            if trigger.target == "self":
                p_boost = trigger.effect_value
                t_boost = (
                    trigger.toughness_value if trigger.toughness_value is not None else p_boost
                )
                card.current_power = (card.current_power or card.power) + p_boost
                card.current_toughness = (card.current_toughness or card.toughness) + t_boost
                event.effect_applied = (
                    f"{card.name} gets +{p_boost}/+{t_boost}"
                    f" (now {card.current_power}/{card.current_toughness})"
                )
                event.affected_cards = [card]

        elif trigger.effect == TriggerEffect.POWER_TOUGHNESS_COUNTER:
            if trigger.target == "self":
                # Add +1/+1 counters
                card.add_plus_counter(trigger.effect_value)
                event.effect_applied = (
                    f"{card.name} gets {trigger.effect_value} +1/+1 counter(s)"
                    f" (now {card.effective_power}/{card.effective_toughness})"
                )
                event.affected_cards = [card]

        elif trigger.effect == TriggerEffect.DEAL_DAMAGE:
            damage = trigger.effect_value
            if damage == 0:
                damage = max(0, card.effective_power)
            event.damage = damage
            if trigger.target == "opponent":
                event.target_player_idx = 1 - controller
                opponent.life -= damage
                event.effect_applied = f"{card.name} deals {damage} damage to opponent"
            elif trigger.target == "any":
                target_creature = None
                if damage > 0:
                    lethal = [
                        c
                        for c in opponent.battlefield
                        if c.card_type == CardType.CREATURE and damage >= c.effective_toughness
                    ]
                    if lethal:
                        target_creature = max(lethal, key=lambda c: c.effective_power)
                if target_creature:
                    opponent.battlefield.remove(target_creature)
                    opponent.graveyard.append(target_creature)
                    event.target_card_id = target_creature.card_id
                    state.log_action(
                        player=1 - controller,
                        action_type="DIES",
                        card_name=target_creature.name,
                        details={
                            "card_id": target_creature.card_id,
                            "cause": "trigger_damage",
                        },
                    )
                    event.effect_applied = (
                        f"{card.name} deals {damage} damage to {target_creature.name}"
                    )
                    event.affected_cards = [target_creature]
                else:
                    event.target_player_idx = 1 - controller
                    opponent.life -= damage
                    event.effect_applied = f"{card.name} deals {damage} damage to opponent"

        elif trigger.effect == TriggerEffect.GAIN_LIFE:
            player.life += trigger.effect_value
            event.effect_applied = f"Controller gains {trigger.effect_value} life"

        elif trigger.effect == TriggerEffect.DRAW_CARDS:
            drawn = player.draw_cards(trigger.effect_value)
            event.effect_applied = f"Controller draws {len(drawn)} card(s)"
            event.affected_cards = drawn

        elif trigger.effect == TriggerEffect.CREATE_TOKEN:
            # Create the token and add it to battlefield
            token = TriggerEngine.create_token(trigger.token_type, controller, state)
            if token:
                player.battlefield.append(token)
                event.effect_applied = f"Create {trigger.token_type} token"
                event.affected_cards = [token]
            else:
                event.effect_applied = f"Create {trigger.token_type} token (failed)"

        elif trigger.effect == TriggerEffect.TRANSFORM:
            event.effect_applied = f"{card.name} transforms"

        elif trigger.effect == TriggerEffect.GRANT_KEYWORD:
            if trigger.keyword_granted:
                if trigger.target == "self":
                    card.keywords.add(trigger.keyword_granted)
                    event.effect_applied = f"{card.name} gains {trigger.keyword_granted.name}"
                else:
                    event.effect_applied = f"Grants {trigger.keyword_granted.name}"

        elif trigger.effect == TriggerEffect.SCRY:
            # Scry: look at top X cards, put any on bottom in any order
            # Simplified: for AI, just put the worst cards on bottom
            scry_count = trigger.effect_value
            top_cards = player.deck[:scry_count]
            if top_cards:
                # Heuristic: put non-lands on top, lands on bottom (simplified)
                keep_on_top = [c for c in top_cards if c.card_type != CardType.LAND]
                put_on_bottom = [c for c in top_cards if c.card_type == CardType.LAND]
                # Rebuild deck
                player.deck = keep_on_top + player.deck[scry_count:] + put_on_bottom
                event.effect_applied = f"Scry {scry_count}"

        elif trigger.effect == TriggerEffect.DISCARD_CARDS:
            # Discard cards (simplified: discard highest CMC cards)
            discard_count = trigger.effect_value
            target_player = opponent if trigger.target == "opponent" else player

            discarded = []
            for _ in range(discard_count):
                if target_player.hand:
                    # Simple heuristic: discard highest CMC
                    highest = max(
                        target_player.hand, key=lambda c: c.mana_cost.cmc if c.mana_cost else 0
                    )
                    target_player.hand.remove(highest)
                    target_player.graveyard.append(highest)
                    discarded.append(highest.name)
            if discarded:
                event.effect_applied = f"Discards: {', '.join(discarded)}"

        elif trigger.effect == TriggerEffect.LOSE_LIFE:
            if trigger.target == "opponent":
                opponent.life -= trigger.effect_value
                event.effect_applied = f"Opponent loses {trigger.effect_value} life"
            else:
                player.life -= trigger.effect_value
                event.effect_applied = f"Loses {trigger.effect_value} life"

        elif trigger.effect == TriggerEffect.DESTROY_PERMANENT:
            # Simplified: destroy target creature (would need proper targeting)
            event.effect_applied = "Destroy target permanent"

        elif trigger.effect == TriggerEffect.EXILE_PERMANENT:
            event.effect_applied = "Exile target permanent"

        elif trigger.effect == TriggerEffect.RETURN_TO_HAND:
            event.effect_applied = "Return to hand"

        elif trigger.effect == TriggerEffect.TAP_PERMANENT:
            event.effect_applied = "Tap target permanent"

        elif trigger.effect == TriggerEffect.CANT_ATTACK:
            event.effect_applied = "Can't attack"

        elif trigger.effect == TriggerEffect.CANT_BLOCK:
            event.effect_applied = "Can't block"

        # NOTE: We don't log trigger events here anymore because they're already
        # collected and logged as part of the CAST action's triggered_abilities.
        # This prevents triggers from appearing at the wrong position in the log.

        return state, event

    @staticmethod
    def fire_cast_noncreature_triggers(
        state: "GameState",
        caster_idx: int,
    ) -> tuple["GameState", list[TriggerEvent]]:
        """Fire all CAST_NONCREATURE triggers (Prowess, etc).

        Args:
            state: Current game state.
            caster_idx: Player who cast the spell.

        Returns:
            Tuple of (updated state, list of trigger events).

        """
        triggers = TriggerEngine.get_triggers_for_event(
            state, TriggerType.CAST_NONCREATURE, caster_idx
        )

        events = []
        for card, trigger in triggers:
            state, event = TriggerEngine.apply_trigger(state, card, trigger, caster_idx)
            events.append(event)

        return state, events

    @staticmethod
    def fire_cast_creature_triggers(
        state: "GameState",
        caster_idx: int,
    ) -> tuple["GameState", list[TriggerEvent]]:
        """Fire all CAST_CREATURE triggers (Kumano Ch II, etc).

        Args:
            state: Current game state.
            caster_idx: Player who cast the spell.

        Returns:
            Tuple of (updated state, list of trigger events).

        """
        triggers = TriggerEngine.get_triggers_for_event(
            state, TriggerType.CAST_CREATURE, caster_idx
        )

        events = []
        for card, trigger in triggers:
            state, event = TriggerEngine.apply_trigger(state, card, trigger, caster_idx)
            events.append(event)

        return state, events

    @staticmethod
    def fire_etb_triggers(
        state: "GameState",
        entering_card: Card,
        controller: int,
    ) -> tuple["GameState", list[TriggerEvent]]:
        """Fire ETB (enters the battlefield) triggers.

        Args:
            state: Current game state.
            entering_card: The card that entered.
            controller: Player who controls the entering card.

        Returns:
            Tuple of (updated state, list of trigger events).

        """
        events = []

        # Check the entering card's own ETB triggers
        for trigger in entering_card.triggers:
            if trigger.trigger_type == TriggerType.ETB_SELF:
                state, event = TriggerEngine.apply_trigger(
                    state, entering_card, trigger, controller
                )
                events.append(event)

        # Check other permanents for "whenever a creature enters" triggers
        if entering_card.card_type == CardType.CREATURE:
            triggers = TriggerEngine.get_triggers_for_event(
                state, TriggerType.ETB_CREATURE, controller
            )
            for card, trigger in triggers:
                if card != entering_card:  # Don't double-trigger
                    state, event = TriggerEngine.apply_trigger(state, card, trigger, controller)
                    events.append(event)

        # Check for landfall triggers
        if entering_card.card_type == CardType.LAND:
            triggers = TriggerEngine.get_triggers_for_event(state, TriggerType.ETB_LAND, controller)
            for card, trigger in triggers:
                state, event = TriggerEngine.apply_trigger(state, card, trigger, controller)
                events.append(event)

        return state, events

    @staticmethod
    def fire_attack_triggers(
        state: "GameState",
        attacker_card: Card,
        controller: int,
    ) -> tuple["GameState", list[TriggerEvent]]:
        """Fire attack triggers for a creature.

        Args:
            state: Current game state.
            attacker_card: The attacking creature.
            controller: Player who controls the attacker.

        Returns:
            Tuple of (updated state, list of trigger events).

        """
        events = []

        # Check the attacker's own attack triggers
        for trigger in attacker_card.triggers:
            if trigger.trigger_type == TriggerType.ATTACK_SELF:
                state, event = TriggerEngine.apply_trigger(
                    state, attacker_card, trigger, controller
                )
                events.append(event)

        return state, events

    # Token counter for unique IDs
    _token_counter: int = 10000

    @staticmethod
    def create_token(
        token_type: str,
        controller: int,
        state: "GameState",
    ) -> Card | None:
        """Create a token creature.

        Args:
            token_type: Description of token (e.g., "1/1 white Soldier creature")
            controller: Player index who will control the token.
            state: Current game state (for unique ID generation).

        Returns:
            A Card object representing the token, or None if invalid.
        """
        import re

        # Parse token type string
        # Formats: "1/1 white Soldier creature",
        # "2/2 white and black Vampire creature with lifelink"

        # Extract power/toughness
        pt_match = re.match(r"(\d+)/(\d+)", token_type)
        if not pt_match:
            return None
        power = int(pt_match.group(1))
        toughness = int(pt_match.group(2))

        # Extract colors
        colors: list[ManaColor] = []
        color_words = {
            "white": ManaColor.WHITE,
            "blue": ManaColor.BLUE,
            "black": ManaColor.BLACK,
            "red": ManaColor.RED,
            "green": ManaColor.GREEN,
        }
        for word, color in color_words.items():
            if word in token_type.lower():
                colors.append(color)

        # Extract creature type (simple heuristic)
        type_words = [
            "Soldier",
            "Vampire",
            "Spirit",
            "Goblin",
            "Zombie",
            "Angel",
            "Elemental",
            "Bird",
            "Samurai",
            "Human",
            "Elf",
            "Beast",
            "Dragon",
            "Cat",
            "Dog",
            "Knight",
        ]
        creature_types = [t for t in type_words if t.lower() in token_type.lower()]
        name = " ".join(creature_types) + " Token" if creature_types else "Creature Token"

        # Extract keywords
        keywords: set[Keyword] = set()
        if "lifelink" in token_type.lower():
            keywords.add(Keyword.LIFELINK)
        if "flying" in token_type.lower():
            keywords.add(Keyword.FLYING)
        if "vigilance" in token_type.lower():
            keywords.add(Keyword.VIGILANCE)
        if "haste" in token_type.lower():
            keywords.add(Keyword.HASTE)
        if "first strike" in token_type.lower():
            keywords.add(Keyword.FIRST_STRIKE)
        if "deathtouch" in token_type.lower():
            keywords.add(Keyword.DEATHTOUCH)
        if "trample" in token_type.lower():
            keywords.add(Keyword.TRAMPLE)

        # Generate unique token ID
        TriggerEngine._token_counter += 1
        token_id = TriggerEngine._token_counter

        # Create the token
        token = Card(
            name=name,
            card_type=CardType.CREATURE,
            power=power,
            toughness=toughness,
            keywords=keywords,
            card_id=token_id,
            is_token=True,
            token_name=token_type,
            owner_player_idx=controller,
        )

        return token

    @staticmethod
    def create_artifact_token(
        token_name: str,
        controller: int,
        state: "GameState",
    ) -> Card | None:
        """Create a noncreature artifact token by name."""
        registry = CardRegistry.get_instance()
        try:
            template = registry.get(token_name)
        except KeyError:
            return None
        token_id = TriggerEngine._token_counter
        TriggerEngine._token_counter += 1
        token = Card(
            name=template.name,
            card_type=template.card_type,
            mana_cost=ManaCost.from_string("0"),
            power=template.power,
            toughness=template.toughness,
            keywords=set(template.keywords),
            rules_text=template.rules_text,
            set_code=template.set_code,
            is_token=True,
            owner_player_idx=controller,
        )
        token.card_id = token_id
        return token

    @staticmethod
    def fire_death_triggers(
        state: "GameState",
        dying_card: Card,
        controller: int,
    ) -> tuple["GameState", list[TriggerEvent]]:
        """Fire death triggers for a creature.

        Args:
            state: Current game state.
            dying_card: The dying creature.
            controller: Player who controlled the dying card.

        Returns:
            Tuple of (updated state, list of trigger events).

        """
        events = []

        # Check the dying card's own death triggers
        for trigger in dying_card.triggers:
            if trigger.trigger_type == TriggerType.DIES_SELF:
                state, event = TriggerEngine.apply_trigger(state, dying_card, trigger, controller)
                events.append(event)

        return state, events


@dataclass
class GameState:
    """Complete game state with priority tracking."""

    # Players (0 = current agent, 1 = opponent)
    players: list[PlayerState] = field(default_factory=lambda: [PlayerState(), PlayerState()])

    # Current game state
    active_player: int = 0  # Whose turn it is
    priority_player: int = 0  # Who has priority to act
    turn_number: int = 0
    phase: GamePhase = GamePhase.MULLIGAN
    mulligan_count: list[int] = field(default_factory=lambda: [0, 0])

    # Stack
    stack: list[StackItem] = field(default_factory=list)

    # Priority tracking
    passed_priority: list[bool] = field(default_factory=lambda: [False, False])

    # Game result
    winner: int | None = None
    game_over: bool = False

    # Play/Draw
    player_on_play: bool = True  # True if player 0 is on the play

    # Configuration
    max_turns: int = 10
    starting_hand_size: int = 7

    # Action logging for visualization
    action_log: list = field(default_factory=list)  # List of GameAction
    turn_actions: dict = field(default_factory=dict)  # {turn: {player: [actions]}}

    # Pending multi-step actions (targeting, combat, mana, discard)
    pending_action_type: str | None = None
    pending_player: int | None = None
    pending_required: int = 0
    pending_selected_indices: list[int] = field(default_factory=list)
    pending_attackers: set[int] = field(default_factory=set)
    pending_block_assignments: dict[int, int] = field(default_factory=dict)
    pending_block_attacker_id: int | None = None
    pending_spell: StackItem | None = None
    pending_target_candidates: list[TargetRef] = field(default_factory=list)
    pending_mana_cost: ManaCost | None = None
    pending_mana_sources: list[int] = field(default_factory=list)
    pending_mana_chosen: list[int] = field(default_factory=list)
    pending_activation: Card | None = None

    def get_active_player(self) -> PlayerState:
        """Get the active (turn) player's state."""
        return self.players[self.active_player]

    def get_priority_player(self) -> PlayerState:
        """Get the player with priority."""
        return self.players[self.priority_player]

    def get_opponent(self, player_idx: int | None = None) -> PlayerState:
        """Get the opponent's state."""
        if player_idx is None:
            player_idx = self.active_player
        return self.players[1 - player_idx]

    def switch_priority(self) -> None:
        """Pass priority to the other player."""
        self.priority_player = 1 - self.priority_player

    def reset_priority(self) -> None:
        """Reset priority to active player."""
        self.priority_player = self.active_player
        self.passed_priority = [False, False]

    def both_players_passed(self) -> bool:
        """Check if both players passed priority in succession."""
        return all(self.passed_priority)

    def is_priority_phase(self) -> bool:
        """Check if current phase is a priority window."""
        return self.phase in PRIORITY_PHASES

    def can_cast_sorcery_speed(self) -> bool:
        """Check if sorcery-speed spells can be cast."""
        return (
            self.phase in SORCERY_SPEED_PHASES
            and self.priority_player == self.active_player
            and len(self.stack) == 0
        )

    def switch_turn(self) -> None:
        """Switch to the other player's turn.

        Turn number increments when it becomes the first player's turn again.
        First player is determined by player_on_play.
        """
        self.active_player = 1 - self.active_player
        # Increment turn when it's the first player's turn again (full cycle complete)
        first_player = 0 if self.player_on_play else 1
        if self.active_player == first_player:
            self.turn_number += 1
        self.reset_priority()

    def log_action(
        self,
        player: int,
        action_type: str,
        card_name: str = "",
        details: dict | None = None,
    ) -> None:
        """Log an action for visualization.

        Args:
            player: 0 for player, 1 for opponent.
            action_type: Type of action (PLAY_LAND, CAST, ATTACK, etc.).
            card_name: Name of the card involved.
            details: Additional details about the action.

        """
        action = GameAction(
            player=player,
            action_type=action_type,
            card_name=card_name,
            phase=self.phase.name,
            turn=self.turn_number,
            details=details or {},
            active_player=self.active_player,
        )
        self.action_log.append(action)

        # Also organize by turn and player
        turn_key = self.turn_number
        if turn_key not in self.turn_actions:
            self.turn_actions[turn_key] = {0: [], 1: []}
        self.turn_actions[turn_key][player].append(action)

    def get_turn_actions(self, turn: int, player: int) -> list:
        """Get all actions for a specific turn and player."""
        if turn in self.turn_actions:
            return self.turn_actions[turn].get(player, [])
        return []

    def get_recent_actions(self, count: int = 10) -> list:
        """Get the most recent actions."""
        return self.action_log[-count:] if self.action_log else []

    def check_game_over(self) -> bool:
        """Check if the game is over."""
        # Check life totals
        for i, player in enumerate(self.players):
            if player.life <= 0:
                self.winner = 1 - i
                self.game_over = True
                self.phase = GamePhase.GAME_OVER
                return True

        # Deck-out: lose when you attempt to draw from an empty library
        for i, player in enumerate(self.players):
            if getattr(player, "failed_draw", False):
                self.winner = 1 - i
                self.game_over = True
                self.phase = GamePhase.GAME_OVER
                return True

        # Check turn limit - game ends in a draw (no winner)
        if self.turn_number > self.max_turns:
            self.winner = None  # No winner - draw
            self.game_over = True
            self.phase = GamePhase.GAME_OVER
            return True

        return False


class RulesEngine:
    """Engine for executing game rules with priority system."""

    def _prune_battlefield_tracking(self, state: GameState) -> None:
        """Remove stale references to cards no longer on the battlefield."""
        for player in state.players:
            battlefield_ids = {c.card_id for c in player.battlefield}
            player.tapped_permanents.intersection_update(battlefield_ids)
            player.summoning_sick.intersection_update(battlefield_ids)
            player.activated_creatures = {
                cid: stats
                for cid, stats in player.activated_creatures.items()
                if cid in battlefield_ids
            }
            player.declared_attackers = [
                cid for cid in player.declared_attackers if cid in battlefield_ids
            ]
            player.declared_blockers = {
                bid: aid for bid, aid in player.declared_blockers.items() if bid in battlefield_ids
            }
            player.multi_blockers = {
                aid: [bid for bid in bids if bid in battlefield_ids]
                for aid, bids in player.multi_blockers.items()
                if bids
            }

    def __init__(self, rng: random.Random | None = None):
        """Initialize the rules engine.

        Args:
            rng: Random number generator for shuffling etc.

        """
        self.rng = rng or random.Random()

    def initialize_game(
        self,
        player_deck: list[Card],
        opponent_deck: list[Card],
        max_turns: int = 10,
    ) -> GameState:
        """Initialize a new game.

        Args:
            player_deck: The agent's deck.
            opponent_deck: The opponent's deck.
            max_turns: Maximum number of turns before game ends.

        Returns:
            Initial game state.

        """
        state = GameState(max_turns=max_turns)

        # Randomly determine who goes first
        state.player_on_play = self.rng.random() < 0.5

        # Set up decks
        state.players[0].deck = player_deck.copy()
        state.players[1].deck = opponent_deck.copy()
        for card in state.players[0].deck:
            card.owner_player_idx = 0
        for card in state.players[1].deck:
            card.owner_player_idx = 1
        # Ensure card_ids are globally unique across both players
        self._assign_unique_card_ids(
            [state.players[0].deck, state.players[1].deck],
            start_id=1,
        )

        # Shuffle decks
        state.players[0].shuffle_deck(self.rng)
        state.players[1].shuffle_deck(self.rng)

        # Draw initial hands
        state.players[0].draw_cards(state.starting_hand_size)
        state.players[1].draw_cards(state.starting_hand_size)

        # Start in mulligan phase
        # During mulligan, player 0 always decides first, then player 1
        # Active player is set based on who goes first, but priority for mulligan
        # always starts with player 0
        state.phase = GamePhase.MULLIGAN
        state.active_player = 0 if state.player_on_play else 1
        state.priority_player = 0  # Player 0 always mulligans first

        return state

    def _assign_unique_card_ids(
        self,
        decks: list[list[Card]],
        start_id: int = 1,
    ) -> int:
        """Assign globally unique card_ids across all decks."""
        next_id = start_id
        for deck in decks:
            for card in deck:
                card.card_id = next_id
                next_id += 1
        return next_id

    def execute_mulligan(self, state: GameState, keep: bool) -> GameState:
        """Execute a mulligan decision using London Mulligan rules.

        London Mulligan: Always draw 7 cards, then put cards on bottom
        equal to the number of mulligans taken.

        A maximum of 3 mulligans is enforced (keeping at least 4 cards).
        Beyond that the player is forced to keep.

        Args:
            state: Current game state.
            keep: True to keep hand, False to mulligan.

        Returns:
            Updated game state.

        """
        player_idx = state.priority_player
        player = state.players[player_idx]

        # Force keep after 3 mulligans (keep at least 4 cards)
        max_mulligans = 3
        if not keep and state.mulligan_count[player_idx] >= max_mulligans:
            keep = True

        if keep:
            if state.mulligan_count[player_idx] > 0:
                # Auto-bottom highest CMC non-land cards for all players.
                # This avoids the multi-step card selection that can stall
                # untrained RL agents, and mirrors competitive heuristics.
                for _ in range(state.mulligan_count[player_idx]):
                    if not player.hand:
                        break
                    non_lands = [c for c in player.hand if c.card_type != CardType.LAND]
                    card_to_bottom = max(
                        non_lands or player.hand,
                        key=lambda c: c.mana_cost.cmc if c.mana_cost else 0,
                    )
                    player.hand.remove(card_to_bottom)
                    player.deck.append(card_to_bottom)

            # Move to next player's mulligan or start game
            if player_idx == 0:
                state.priority_player = 1
            else:
                # Both players done, start game
                state.active_player = 0 if state.player_on_play else 1
                state.priority_player = state.active_player
                state.turn_number = 1
                state.phase = GamePhase.UNTAP
                # Execute beginning of turn for first turn
                state = self._execute_beginning_of_turn(state)
        else:
            # London Mulligan: Put hand back, shuffle, always draw 7
            player.deck.extend(player.hand)
            player.hand.clear()
            player.shuffle_deck(self.rng)

            # Track mulligan count (used later to determine cards to bottom)
            state.mulligan_count[player_idx] += 1

            # Always draw 7 cards (London Mulligan rule)
            # Cards will be put on bottom separately after player chooses
            player.draw_cards(state.starting_hand_size)

        return state

    def _apply_bottom_cards(self, state: GameState, player_idx: int) -> GameState:
        player = state.players[player_idx]
        count = state.pending_required
        selected = sorted(set(state.pending_selected_indices), reverse=True)
        moved = 0
        for idx in selected:
            if moved >= count:
                break
            if 0 <= idx < len(player.hand):
                card = player.hand.pop(idx)
                player.deck.append(card)
                moved += 1

        # Clear pending
        state.pending_action_type = None
        state.pending_player = None
        state.pending_required = 0
        state.pending_selected_indices = []

        # Continue mulligan flow
        if player_idx == 0:
            state.priority_player = 1
        else:
            state.active_player = 0 if state.player_on_play else 1
            state.priority_player = state.active_player
            state.turn_number = 1
            state.phase = GamePhase.UNTAP
            state = self._execute_beginning_of_turn(state)

        return state

    def _apply_discards(self, state: GameState, player_idx: int) -> GameState:
        player = state.players[player_idx]
        count = state.pending_required
        selected = sorted(set(state.pending_selected_indices), reverse=True)
        discarded_cards = []

        moved = 0
        for idx in selected:
            if moved >= count:
                break
            if 0 <= idx < len(player.hand):
                card = player.hand.pop(idx)
                player.graveyard.append(card)
                self._update_haughty_djinn_power(player)
                discarded_cards.append(card.name)
                moved += 1

        state.pending_action_type = None
        state.pending_player = None
        state.pending_required = 0
        state.pending_selected_indices = []

        # Finish cleanup after discards
        state = self._execute_cleanup(state)
        return state

    def advance_phase(self, state: GameState) -> GameState:
        """Advance to the next phase."""
        self._prune_battlefield_tracking(state)
        phase_order = [
            GamePhase.UNTAP,
            GamePhase.UPKEEP,
            GamePhase.DRAW,
            GamePhase.MAIN_PRECOMBAT,
            GamePhase.COMBAT_BEGIN,
            GamePhase.COMBAT_ATTACKERS,
            GamePhase.COMBAT_BLOCKERS,
            GamePhase.COMBAT_DAMAGE,
            GamePhase.MAIN_POSTCOMBAT,
            GamePhase.END_STEP,
            GamePhase.CLEANUP,
        ]

        current_idx = phase_order.index(state.phase) if state.phase in phase_order else 0
        next_idx = current_idx + 1

        if next_idx >= len(phase_order):
            # Turn ends, switch to opponent
            state.switch_turn()
            state.phase = GamePhase.UNTAP
            # Check turn limit before starting the new turn
            if state.check_game_over():
                return state
            state = self._execute_beginning_of_turn(state)
        else:
            state.phase = phase_order[next_idx]
            state.reset_priority()

            # Execute phase-specific effects
            if state.phase == GamePhase.UNTAP:
                state = self._execute_beginning_of_turn(state)
            elif state.phase == GamePhase.DRAW:
                state = self._execute_draw_phase(state)
            elif state.phase == GamePhase.COMBAT_DAMAGE:
                self._prune_battlefield_tracking(state)
                state = self._execute_combat_damage(state)
            elif state.phase == GamePhase.CLEANUP:
                state = self._execute_cleanup(state)
                if state.pending_action_type:
                    return state
                # Cleanup doesn't get priority, move to next turn
                state = self.advance_phase(state)

        return state

    def _execute_beginning_of_turn(self, state: GameState) -> GameState:
        """Execute beginning of turn effects."""
        player = state.get_active_player()

        # Track what was tapped before untapping
        tapped_permanents = list(player.tapped_permanents)
        untapped_cards = [c for c in player.battlefield if c.card_id in tapped_permanents]

        # Count lands on board
        lands_on_board = {}
        for c in player.battlefield:
            if c.produces_mana:
                lands_on_board[c.name] = lands_on_board.get(c.name, 0) + 1

        player.untap_all()
        player.reset_for_turn()
        state.reset_priority()
        state.pending_action_type = None
        state.pending_player = None
        state.pending_required = 0
        state.pending_selected_indices = []
        state.pending_attackers = set()
        state.pending_block_assignments = {}
        state.pending_block_attacker_id = None
        state.pending_spell = None
        state.pending_target_candidates = []
        state.pending_mana_cost = None
        state.pending_mana_sources = []
        state.pending_mana_chosen = []
        state.pending_activation = None

        # Log the untap action with details
        untapped_names = [c.name for c in untapped_cards]
        untapped_data = []
        for card in untapped_cards:
            power, toughness = self._get_effective_power_toughness(card, player)
            untapped_data.append(
                {
                    "name": card.name,
                    "power": power,
                    "toughness": toughness,
                    "tokens": card.attached_tokens.copy(),
                }
            )
        state.log_action(
            player=state.active_player,
            action_type="UNTAP",
            card_name="Beginning of Turn",
            details={
                "untapped": untapped_names,
                "untapped_data": untapped_data,
                "lands_on_board": lands_on_board,
            },
        )

        return state

    def _apply_sheoldred_draw_triggers(self, state: GameState, drawing_player_idx: int) -> None:
        """Apply Sheoldred, the Apocalypse draw/drain triggers.

        When the controller draws: gain 2 life.
        When the opponent draws: they lose 2 life.
        """
        for p_idx, p in enumerate(state.players):
            for card in p.battlefield:
                if card.name == "Sheoldred, the Apocalypse":
                    if drawing_player_idx == p_idx:
                        p.life += 2
                    else:
                        state.players[drawing_player_idx].life -= 2

    def _execute_draw_phase(self, state: GameState) -> GameState:
        """Execute the draw phase."""
        # Skip draw on turn 1 for player on the play
        skip_draw = state.turn_number == 1 and state.active_player == (
            0 if state.player_on_play else 1
        )

        if not skip_draw:
            player = state.get_active_player()
            drawn_card = player.draw_card()
            player.has_drawn_for_turn = True

            if drawn_card:
                state.log_action(
                    player=state.active_player,
                    action_type="DRAW",
                    card_name=drawn_card.name,
                    details={
                        "card_type": drawn_card.card_type.value,
                        "hand_size": len(player.hand),
                    },
                )
                self._apply_sheoldred_draw_triggers(state, state.active_player)

        return state

    def _execute_combat_damage(self, state: GameState) -> GameState:
        """Execute combat damage step with first strike and double strike support.

        Combat damage happens in two steps:
        1. First strike damage step: creatures with first/double strike deal damage
        2. Regular damage step: creatures without first strike deal damage
           (double strike creatures deal damage again)
        """
        attacker_idx = state.active_player
        defender_idx = 1 - attacker_idx
        attacker_state = state.get_active_player()
        defender_state = state.get_opponent()

        damage_events = []
        deaths = []  # Track (card, controller_idx) for death triggers
        attacker_data = []  # Track attacker info for display

        # Track damage marked on creatures (doesn't persist between combats)
        damage_marked: dict[int, int] = {}  # card_id -> damage taken

        # Determine if we need first strike step
        has_first_strike = any(
            Keyword.FIRST_STRIKE
            in (
                attacker_state.get_card_by_id(aid).keywords
                if attacker_state.get_card_by_id(aid)
                else set()
            )
            or Keyword.DOUBLE_STRIKE
            in (
                attacker_state.get_card_by_id(aid).keywords
                if attacker_state.get_card_by_id(aid)
                else set()
            )
            for aid in attacker_state.declared_attackers
        ) or any(
            Keyword.FIRST_STRIKE
            in (
                defender_state.get_card_by_id(bid).keywords
                if defender_state.get_card_by_id(bid)
                else set()
            )
            or Keyword.DOUBLE_STRIKE
            in (
                defender_state.get_card_by_id(bid).keywords
                if defender_state.get_card_by_id(bid)
                else set()
            )
            for bid in defender_state.declared_blockers
        )

        # Execute first strike damage step if needed
        if has_first_strike:
            state, fs_events, fs_deaths, damage_marked, fs_attacker_data = (
                self._execute_damage_step(
                    state,
                    attacker_state,
                    defender_state,
                    attacker_idx,
                    defender_idx,
                    first_strike_only=True,
                    damage_marked=damage_marked,
                )
            )
            damage_events.extend(fs_events)
            deaths.extend(fs_deaths)
            attacker_data.extend(fs_attacker_data)

            # Check for deaths and run state-based actions
            state = self._check_state_based_actions(state, deaths)
            deaths.clear()  # Deaths already processed

        # Execute regular damage step
        state, reg_events, reg_deaths, damage_marked, reg_attacker_data = self._execute_damage_step(
            state,
            attacker_state,
            defender_state,
            attacker_idx,
            defender_idx,
            first_strike_only=False,
            damage_marked=damage_marked,
        )
        damage_events.extend(reg_events)
        deaths.extend(reg_deaths)
        attacker_data.extend(reg_attacker_data)

        # Fire death triggers for all creatures that died
        trigger_damage_events = []
        for card, controller in deaths:
            state, events = TriggerEngine.fire_death_triggers(state, card, controller)
            for event in events:
                damage_events.append(f"Death trigger: {event.effect_applied}")
                if event.damage and event.target_player_idx is not None:
                    trigger_damage_events.append(
                        {
                            "damage": event.damage,
                            "target_player_idx": event.target_player_idx,
                            "source_name": event.source_card.name,
                            "source_id": event.source_card.card_id,
                        }
                    )

        # Capture current combat block assignments for accurate display
        block_data = []
        for blocker_id, attacker_id in defender_state.declared_blockers.items():
            blocker_card = defender_state.get_card_by_id(blocker_id)
            attacker_card = attacker_state.get_card_by_id(attacker_id)
            if not blocker_card or not attacker_card:
                continue
            attacker_power, attacker_toughness = self._get_effective_power_toughness(
                attacker_card, attacker_state
            )
            blocker_power, blocker_toughness = self._get_effective_power_toughness(
                blocker_card, defender_state
            )
            block_data.append(
                {
                    "attacker": attacker_card.name,
                    "attacker_id": attacker_card.card_id,
                    "attacker_power": attacker_power,
                    "attacker_toughness": attacker_toughness,
                    "attacker_tokens": attacker_card.attached_tokens.copy(),
                    "blocker": blocker_card.name,
                    "blocker_id": blocker_card.card_id,
                    "blocker_power": blocker_power,
                    "blocker_toughness": blocker_toughness,
                    "blocker_tokens": blocker_card.attached_tokens.copy(),
                    "attacker_owner_idx": attacker_idx,
                    "blocker_owner_idx": defender_idx,
                }
            )

        # Log combat damage
        if damage_events:
            state.log_action(
                player=attacker_idx,
                action_type="DAMAGE",
                card_name="Combat Damage",
                details={
                    "events": damage_events,
                    "defender_life": defender_state.life,
                    "attacker_data": attacker_data,
                    "block_data": block_data,
                    "trigger_damage": trigger_damage_events,
                },
            )

        # Combat is over - clear combat state for both players
        for player in state.players:
            player.declared_attackers.clear()
            player.declared_blockers.clear()
            player.multi_blockers.clear()
        state.pending_block_assignments = {}
        state.pending_block_attacker_id = None

        state.check_game_over()
        return state

    def _execute_damage_step(
        self,
        state: GameState,
        attacker_state: PlayerState,
        defender_state: PlayerState,
        attacker_idx: int,
        defender_idx: int,
        first_strike_only: bool,
        damage_marked: dict[int, int],
    ) -> tuple[
        GameState,
        list[str],
        list[tuple[Card, int]],
        dict[int, int],
        list[dict[str, int | str]],
    ]:
        """Execute a single damage step (first strike or regular).

        Args:
            state: Current game state.
            attacker_state: Attacking player state.
            defender_state: Defending player state.
            attacker_idx: Attacking player index.
            defender_idx: Defending player index.
            first_strike_only: If True, only first/double strike creatures deal damage.
            damage_marked: Dictionary tracking damage marked on creatures.

        Returns:
            Tuple of (state, damage_events, deaths, updated_damage_marked, attacker_damage_data).
        """
        damage_events = []
        deaths = []
        attacker_damage_data: list[dict[str, int | str]] = []

        for attacker_id in list(attacker_state.declared_attackers):
            attacker = attacker_state.get_card_by_id(attacker_id)
            if not attacker or attacker not in attacker_state.battlefield:
                continue

            # Check if this creature deals damage in this step
            has_fs = (
                Keyword.FIRST_STRIKE in attacker.keywords
                or Keyword.DOUBLE_STRIKE in attacker.keywords
            )
            has_ds = Keyword.DOUBLE_STRIKE in attacker.keywords

            if first_strike_only and not has_fs:
                continue  # Skip non-first-strikers in first strike step
            if not first_strike_only and has_fs and not has_ds:
                continue  # Skip pure first-strikers in regular step (they already dealt damage)

            # Check if blocked
            blocked_by = [
                bid for bid, aid in defender_state.declared_blockers.items() if aid == attacker_id
            ]
            # Also check multi_blockers
            if attacker_id in defender_state.multi_blockers:
                blocked_by = defender_state.multi_blockers[attacker_id]

            # Use effective stats
            attacker_power, attacker_toughness = self._get_effective_power_toughness(
                attacker, attacker_state
            )

            if blocked_by:
                blockers = [defender_state.get_card_by_id(bid) for bid in blocked_by]
                blockers = [b for b in blockers if b and b in defender_state.battlefield]

                if not blockers:
                    # All blockers died before damage; trample still goes through
                    if (
                        Keyword.TRAMPLE in attacker.keywords
                        and attacker in attacker_state.battlefield
                    ):
                        defender_state.life -= attacker_power
                        damage_events.append(
                            f"{attacker.name} trample {attacker_power} (all blockers dead)"
                        )
                        if Keyword.LIFELINK in attacker.keywords:
                            attacker_state.life += attacker_power
                            damage_events.append(f"{attacker.name} lifelink {attacker_power}")
                    continue

                damage_events.append(
                    f"{attacker.name} blocked by {', '.join(b.name for b in blockers)}"
                )

                attacker_deathtouch = Keyword.DEATHTOUCH in attacker.keywords
                attacker_lifelink = Keyword.LIFELINK in attacker.keywords
                attacker_indestructible = Keyword.INDESTRUCTIBLE in attacker.keywords

                remaining_power = attacker_power
                total_damage_dealt = 0

                # Assign damage to blockers by increasing effective toughness
                blockers_sorted = sorted(
                    blockers,
                    key=lambda b: self._get_effective_power_toughness(b, defender_state)[1],
                )
                for blocker in blockers_sorted:
                    if blocker not in defender_state.battlefield:
                        continue

                    blocker_power, blocker_toughness = self._get_effective_power_toughness(
                        blocker, defender_state
                    )

                    # Check if blocker deals damage in this step
                    blocker_fs = (
                        Keyword.FIRST_STRIKE in blocker.keywords
                        or Keyword.DOUBLE_STRIKE in blocker.keywords
                    )
                    blocker_ds = Keyword.DOUBLE_STRIKE in blocker.keywords

                    blocker_deals_damage = (first_strike_only and blocker_fs) or (
                        not first_strike_only and (not blocker_fs or blocker_ds)
                    )

                    # Attacker damages blocker
                    lethal = 1 if attacker_deathtouch else blocker_toughness
                    assigned = min(remaining_power, lethal)
                    remaining_power -= assigned
                    total_damage_dealt += assigned

                    # Mark damage on blocker
                    damage_marked[blocker.card_id] = (
                        damage_marked.get(blocker.card_id, 0) + assigned
                    )

                    # Check if blocker dies (damage >= toughness or deathtouch)
                    blocker_indestructible = Keyword.INDESTRUCTIBLE in blocker.keywords
                    if not blocker_indestructible:
                        if attacker_deathtouch and assigned > 0:
                            defender_state.battlefield.remove(blocker)
                            defender_state.graveyard.append(blocker)
                            self._handle_leave_battlefield(state, blocker)
                            state.log_action(
                                player=defender_idx,
                                action_type="DIES",
                                card_name=blocker.name,
                                details={
                                    "card_id": blocker.card_id,
                                    "cause": "combat_deathtouch",
                                },
                            )
                            damage_events.append(f"{blocker.name} dies (deathtouch)")
                            deaths.append((blocker, defender_idx))
                        elif damage_marked.get(blocker.card_id, 0) >= blocker_toughness:
                            defender_state.battlefield.remove(blocker)
                            defender_state.graveyard.append(blocker)
                            self._handle_leave_battlefield(state, blocker)
                            state.log_action(
                                player=defender_idx,
                                action_type="DIES",
                                card_name=blocker.name,
                                details={
                                    "card_id": blocker.card_id,
                                    "cause": "combat_damage",
                                },
                            )
                            damage_events.append(f"{blocker.name} dies")
                            deaths.append((blocker, defender_idx))

                    # Blocker damages attacker
                    if blocker_deals_damage and blocker_power > 0:
                        damage_marked[attacker.card_id] = (
                            damage_marked.get(attacker.card_id, 0) + blocker_power
                        )

                        if Keyword.LIFELINK in blocker.keywords:
                            defender_state.life += blocker_power
                            damage_events.append(f"{blocker.name} lifelink {blocker_power}")

                        if (
                            Keyword.DEATHTOUCH in blocker.keywords
                            and not attacker_indestructible
                            and attacker in attacker_state.battlefield
                        ):
                            attacker_state.battlefield.remove(attacker)
                            attacker_state.graveyard.append(attacker)
                            self._handle_leave_battlefield(state, attacker)
                            state.log_action(
                                player=attacker_idx,
                                action_type="DIES",
                                card_name=attacker.name,
                                details={
                                    "card_id": attacker.card_id,
                                    "cause": "combat_deathtouch",
                                },
                            )
                            damage_events.append(f"{attacker.name} dies (deathtouch)")
                            deaths.append((attacker, attacker_idx))

                # Check if attacker dies from accumulated damage
                if (
                    not attacker_indestructible
                    and attacker in attacker_state.battlefield
                    and damage_marked.get(attacker.card_id, 0) >= attacker_toughness
                ):
                    attacker_state.battlefield.remove(attacker)
                    attacker_state.graveyard.append(attacker)
                    self._handle_leave_battlefield(state, attacker)
                    state.log_action(
                        player=attacker_idx,
                        action_type="DIES",
                        card_name=attacker.name,
                        details={
                            "card_id": attacker.card_id,
                            "cause": "combat_damage",
                        },
                    )
                    damage_events.append(f"{attacker.name} dies")
                    deaths.append((attacker, attacker_idx))

                # Trample: excess damage to player
                trample_damage = 0
                if attacker in attacker_state.battlefield and Keyword.TRAMPLE in attacker.keywords:
                    excess = max(0, remaining_power)
                    if excess > 0:
                        defender_state.life -= excess
                        trample_damage = excess
                        damage_events.append(f"{attacker.name} trample {excess} damage")
                        attacker_damage_data.append(
                            {
                                "name": attacker.name,
                                "power": attacker_power,
                                "toughness": attacker_toughness,
                                "damage": excess,
                                "tokens": attacker.attached_tokens.copy(),
                            }
                        )

                # Lifelink counts ALL damage dealt (blockers + trample)
                if attacker_lifelink and (total_damage_dealt + trample_damage) > 0:
                    life_gained = total_damage_dealt + trample_damage
                    attacker_state.life += life_gained
                    damage_events.append(f"{attacker.name} lifelink {life_gained}")
            else:
                # Unblocked - damage to player (unless menace-blocked)
                menace_set = getattr(defender_state, "menace_blocked", set())
                if attacker_id in menace_set:
                    continue
                defender_state.life -= attacker_power
                damage_events.append(f"{attacker.name} deals {attacker_power} damage")
                if attacker_power > 0:
                    attacker_damage_data.append(
                        {
                            "name": attacker.name,
                            "power": attacker_power,
                            "toughness": attacker_toughness,
                            "damage": attacker_power,
                            "tokens": attacker.attached_tokens.copy(),
                        }
                    )

                # Lifelink
                if Keyword.LIFELINK in attacker.keywords:
                    attacker_state.life += attacker_power
                    damage_events.append(f"{attacker.name} lifelink {attacker_power}")

        return state, damage_events, deaths, damage_marked, attacker_damage_data

    def _check_state_based_actions(
        self, state: GameState, deaths: list[tuple[Card, int]]
    ) -> GameState:
        """Check and execute state-based actions.

        State-based actions include:
        - Creatures with 0 or less toughness die
        - Players with 0 or less life lose
        - Creatures with lethal damage marked die
        - +1/+1 and -1/-1 counters cancel out
        - Auras attached to nothing go to graveyard

        Args:
            state: Current game state.
            deaths: List to append deaths to.

        Returns:
            Updated game state.
        """
        for player_idx in [0, 1]:
            player = state.players[player_idx]

            for card in list(player.battlefield):
                if card.card_type != CardType.CREATURE:
                    continue

                # Check for 0 or less toughness
                _, toughness = self._get_effective_power_toughness(card, player)
                if toughness <= 0 and Keyword.INDESTRUCTIBLE not in card.keywords:
                    player.battlefield.remove(card)
                    player.graveyard.append(card)
                    state.log_action(
                        player=player_idx,
                        action_type="DIES",
                        card_name=card.name,
                        details={
                            "card_id": card.card_id,
                            "cause": "state_based",
                        },
                    )
                    deaths.append((card, player_idx))

            # Check for player death
            if player.life <= 0:
                state.game_over = True
                state.winner = 1 - player_idx

        return state

    def _execute_cleanup(self, state: GameState) -> GameState:
        """Execute cleanup step (no priority normally)."""
        max_hand_size = 7
        discarded_cards: list[dict] = []

        # Discard to hand size - active player discards first, then opponent
        for player_idx in [state.active_player, 1 - state.active_player]:
            player = state.players[player_idx]
            if len(player.hand) > max_hand_size:
                if player_idx == 0:
                    state.pending_action_type = "discard"
                    state.pending_player = player_idx
                    state.pending_required = len(player.hand) - max_hand_size
                    state.pending_selected_indices = []
                    return state

                while len(player.hand) > max_hand_size:
                    non_lands = [c for c in player.hand if c.card_type != CardType.LAND]
                    if non_lands:
                        card_to_discard = max(
                            non_lands, key=lambda c: c.mana_cost.cmc if c.mana_cost else 0
                        )
                    else:
                        card_to_discard = player.hand[-1]

                    player.hand.remove(card_to_discard)
                    player.graveyard.append(card_to_discard)
                    self._update_haughty_djinn_power(player)
                    discarded_cards.append(
                        {
                            "player_idx": player_idx,
                            "card_name": card_to_discard.name,
                            "hand_size_before": len(player.hand) + 1,
                            "hand_size_after": len(player.hand),
                        }
                    )

        # Clear damage and temporary effects (Prowess, pump spells)
        cleanup_effects = []
        for player_idx, player in enumerate(state.players):
            for card in player.battlefield:
                # Check if creature had temporary buffs
                if card.current_power is not None or card.current_toughness is not None:
                    old_power = card.current_power or card.power
                    old_toughness = card.current_toughness or card.toughness
                    # Reset temporary power/toughness modifications
                    card.current_power = None
                    card.current_toughness = None
                    # Calculate final stats (base + permanent bonuses)
                    final_power = card.power + card.permanent_power_bonus
                    final_toughness = card.toughness + card.permanent_toughness_bonus
                    # Log the effect wearing off
                    cleanup_effects.append(
                        {
                            "creature": card.name,
                            "from_power": old_power,
                            "from_toughness": old_toughness,
                            "to_power": final_power,
                            "to_toughness": final_toughness,
                            "has_token": len(card.attached_tokens) > 0,
                            "tokens": card.attached_tokens.copy(),
                            "player_idx": player_idx,
                        }
                    )
                else:
                    # Reset anyway (in case)
                    card.current_power = None
                    card.current_toughness = None

            player.activated_creatures.clear()

        # Always log cleanup (even if no effects wore off)
        state.log_action(
            player=state.active_player,
            action_type="CLEANUP",
            card_name="",
            details={
                "effects_removed": cleanup_effects,
                "discarded_cards": discarded_cards,
            },
        )
        return state

    def can_cast_spell(
        self,
        state: GameState,
        card: Card,
        player_idx: int | None = None,
    ) -> bool:
        """Check if a spell can be cast by a player.

        Args:
            state: Current game state.
            card: Card to cast.
            player_idx: Player index (defaults to priority player).

        Returns:
            True if the spell can be cast.

        """
        if player_idx is None:
            player_idx = state.priority_player

        player = state.players[player_idx]
        opponent = state.get_opponent(player_idx)

        if card.card_type == CardType.LAND:
            return False

        # Check if in hand
        if card not in player.hand:
            return False

        # Check mana (domain + convoke)
        cost = self._effective_mana_cost(state, card, player_idx)
        if card.has_convoke:
            cost, _ = self._apply_convoke_to_cost(player, cost)
        if not self._can_pay_mana_cost(player, cost):
            return False

        # Pump spells should only be cast during combat or main phases
        if card.is_pump_spell:
            # Pump spells need our own creatures
            our_creatures = [c for c in player.battlefield if c.card_type == CardType.CREATURE]
            if not our_creatures:
                return False
            # Pump spells are only useful during combat or main phases (for blocks)
            if state.phase not in [
                GamePhase.MAIN_PRECOMBAT,
                GamePhase.COMBAT_BEGIN,
                GamePhase.COMBAT_ATTACKERS,
                GamePhase.COMBAT_BLOCKERS,
                GamePhase.MAIN_POSTCOMBAT,
            ]:
                return False

        # Check for valid targets (non-pump spells)
        elif card.requires_creature_target:
            if not card.deals_damage and not card.is_removal:
                # Non-damage targeting spell needs our own creatures
                our_creatures = [c for c in player.battlefield if c.card_type == CardType.CREATURE]
                if not our_creatures:
                    return False
            else:
                # Removal/damage targeting creatures needs opponent creatures
                opponent_creatures = [
                    c for c in opponent.battlefield if c.card_type == CardType.CREATURE
                ]
                if not opponent_creatures:
                    return False
        elif card.can_target_nonland_permanent:
            if card.name in {"Leyline Binding", "Detention Sphere"}:
                opponent_permanents = [
                    c for c in opponent.battlefield if c.card_type != CardType.LAND
                ]
                if not opponent_permanents:
                    return False
            else:
                any_permanents = [
                    c
                    for c in player.battlefield + opponent.battlefield
                    if c.card_type != CardType.LAND
                ]
                if not any_permanents:
                    return False

        # Counterspells require a spell on the stack
        if card.is_counterspell and not state.stack:
            return False

        # Verify target availability for targeted spells
        if (
            card.requires_creature_target
            or card.can_target_any
            or card.can_target_nonland_permanent
            or card.is_counterspell
        ):
            candidates = self.get_spell_target_candidates(state, card, player_idx)
            if not candidates:
                return False

        # Instants and flash can be cast in any priority window
        if card.card_type == CardType.INSTANT:
            return state.phase in PRIORITY_PHASES
        if Keyword.FLASH in card.keywords:
            return state.phase in PRIORITY_PHASES

        # Sorcery speed: must be active player, main phase, empty stack
        return state.can_cast_sorcery_speed()

    def can_activate_ability(
        self,
        state: GameState,
        card: Card,
        player_idx: int,
    ) -> bool:
        """Check if an activated ability can be used."""
        player = state.players[player_idx]
        if not card.land_props or not card.land_props.has_activation:
            return False
        if card.card_id in player.tapped_permanents:
            return False
        if not self._can_pay_mana_cost(player, card.land_props.activation_cost):
            return False
        return state.phase in PRIORITY_PHASES

    def can_cast_flashback(
        self,
        state: GameState,
        card: Card,
        player_idx: int,
    ) -> bool:
        """Check if a spell can be cast with flashback from graveyard.

        Args:
            state: Current game state.
            card: Card to cast.
            player_idx: Player index.

        Returns:
            True if the spell can be cast with flashback.
        """
        player = state.players[player_idx]

        # Card must have flashback
        if not card.has_flashback or not card.flashback_cost:
            return False

        # Card must be in graveyard
        if card not in player.graveyard:
            return False

        # Check flashback mana cost
        if not self._can_pay_mana_cost(player, card.flashback_cost):
            return False

        # Instants can be cast anytime with priority
        if card.card_type == CardType.INSTANT:
            return state.phase in PRIORITY_PHASES

        # Sorceries need sorcery-speed timing
        return state.can_cast_sorcery_speed()

    def cast_flashback(
        self,
        state: GameState,
        card: Card,
        player_idx: int,
    ) -> GameState:
        """Cast a spell with flashback from graveyard.

        Args:
            state: Current game state.
            card: Card to cast.
            player_idx: Player index.

        Returns:
            Updated game state.
        """
        player = state.players[player_idx]

        if card not in player.graveyard or not card.flashback_cost:
            return state

        # Pay flashback cost
        sources = self._get_mana_sources(player)
        chosen = self._select_mana_sources_for_cost(sources, card.flashback_cost)
        if not chosen:
            return state

        for src in chosen:
            player.tapped_permanents.add(src.card_id)

        # Remove from graveyard (will be exiled after resolution)
        player.graveyard.remove(card)

        # Get targets
        targets = self._auto_choose_targets(state, card, player_idx)

        # Put on stack
        item = StackItem(
            source=card,
            controller=player_idx,
            targets=targets,
            is_flashback=True,
        )
        state.stack.append(item)
        state.reset_priority()

        # Log the action
        state.log_action(
            player=player_idx,
            action_type="CAST_FLASHBACK",
            card_name=card.name,
            details={"cost": card.flashback_cost.to_text()},
        )

        return state

    def get_spell_target_candidates(
        self,
        state: GameState,
        card: Card,
        player_idx: int,
    ) -> list[TargetRef]:
        """Build target candidates for a spell, respecting hexproof/protection.

        Args:
            state: Current game state.
            card: The spell being cast.
            player_idx: The player casting the spell.

        Returns:
            List of valid targets that can be targeted by this spell.
        """
        player = state.players[player_idx]
        opponent = state.get_opponent(player_idx)
        opponent_idx = 1 - player_idx
        candidates: list[TargetRef] = []

        if card.is_counterspell:
            for i, item in enumerate(state.stack):
                if item.controller != player_idx:
                    candidates.append(TargetRef(kind="stack", ref_id=i, name=item.source.name))
            return candidates

        # Get spell colors for protection checking
        spell_colors = [c.value for c in card.mana_cost.colors] if card.mana_cost else []

        if card.requires_creature_target and (card.deals_damage > 0 or card.is_removal):
            restriction = getattr(card, "target_restriction", "")
            for creature in opponent.battlefield:
                if not (
                    creature.card_type == CardType.CREATURE
                    or creature.card_id in opponent.activated_creatures
                ):
                    continue
                if not self._can_target_creature(creature, player_idx, opponent_idx, spell_colors):
                    continue
                # Go for the Throat: nonartifact creatures only
                if restriction == "nonartifact" and creature.card_type == CardType.ARTIFACT:
                    continue
                # Cut Down: total P+T <= 5
                if restriction == "pt_lte_5":
                    ep, et = self._get_effective_power_toughness(creature, opponent)
                    if ep + et > 5:
                        continue
                candidates.append(
                    TargetRef(kind="creature", ref_id=creature.card_id, name=creature.name)
                )
            return candidates

        if card.is_pump_spell or (
            card.requires_creature_target and not card.deals_damage and not card.is_removal
        ):
            # Targeting own creatures with buffs
            for creature in player.battlefield:
                if (
                    creature.card_type == CardType.CREATURE
                    or creature.card_id in player.activated_creatures
                ) and self._can_target_creature(creature, player_idx, player_idx, spell_colors):
                    # Can always target own creatures (hexproof doesn't apply)
                    candidates.append(
                        TargetRef(kind="creature", ref_id=creature.card_id, name=creature.name)
                    )
            return candidates

        if card.can_target_nonland_permanent:
            targets = (
                [opponent]
                if card.name in {"Leyline Binding", "Detention Sphere"}
                else [opponent, player]
            )
            for target_player in targets:
                controller_idx = opponent_idx if target_player is opponent else player_idx
                for permanent in target_player.battlefield:
                    if permanent.card_type == CardType.LAND:
                        continue
                    if permanent.card_type == CardType.CREATURE and not self._can_target_creature(
                        permanent, player_idx, controller_idx, spell_colors
                    ):
                        continue
                    candidates.append(
                        TargetRef(
                            kind="permanent",
                            ref_id=permanent.card_id,
                            name=permanent.name,
                        )
                    )
            return candidates

        if card.can_target_any:
            candidates.append(TargetRef(kind="player", ref_id=opponent_idx, name="Opponent"))
            candidates.append(TargetRef(kind="player", ref_id=player_idx, name="You"))
            # Own creatures
            for creature in player.battlefield:
                if (
                    creature.card_type == CardType.CREATURE
                    or creature.card_id in player.activated_creatures
                ) and self._can_target_creature(creature, player_idx, player_idx, spell_colors):
                    candidates.append(
                        TargetRef(kind="creature", ref_id=creature.card_id, name=creature.name)
                    )
            # Opponent's creatures
            for creature in opponent.battlefield:
                if (
                    creature.card_type == CardType.CREATURE
                    or creature.card_id in opponent.activated_creatures
                ) and self._can_target_creature(creature, player_idx, opponent_idx, spell_colors):
                    candidates.append(
                        TargetRef(kind="creature", ref_id=creature.card_id, name=creature.name)
                    )
            return candidates

        return candidates

    def _can_target_creature(
        self,
        creature: Card,
        caster_idx: int,
        creature_controller_idx: int,
        spell_colors: list[str],
    ) -> bool:
        """Check if a creature can be targeted by a spell.

        Args:
            creature: The creature to check.
            caster_idx: Player casting the spell.
            creature_controller_idx: Player controlling the creature.
            spell_colors: Colors of the spell being cast.

        Returns:
            True if the creature can be legally targeted.
        """
        # Hexproof - can't be targeted by opponents
        if creature.has_hexproof and caster_idx != creature_controller_idx:
            return False

        # Protection from colors
        return all(not creature.has_protection_from(color) for color in spell_colors)

    def _can_pay_mana_cost(self, player: PlayerState, cost: ManaCost) -> bool:
        """Check if untapped mana sources can pay a mana cost."""
        sources = [
            c
            for c in player.battlefield
            if c.produces_mana and c.card_id not in player.tapped_permanents
        ]
        source_colors = [{color.value for color in s.produces_mana} for s in sources]

        # Pay colored mana first, preferring the most restrictive sources
        required_colors: list[str] = (
            ["W"] * cost.white
            + ["U"] * cost.blue
            + ["B"] * cost.black
            + ["R"] * cost.red
            + ["G"] * cost.green
        )
        remaining_sources = list(range(len(source_colors)))

        for color in required_colors:
            candidates = [idx for idx in remaining_sources if color in source_colors[idx]]
            if not candidates:
                return False
            # Prefer sources with fewer color options
            best_idx = min(candidates, key=lambda i: len(source_colors[i]))
            remaining_sources.remove(best_idx)

        # Remaining sources can pay generic costs
        return len(remaining_sources) >= cost.generic

    def _creature_colors(self, card: Card) -> set[str]:
        if not card.mana_cost:
            return set()
        return {c.value for c in card.mana_cost.colors}

    def _apply_convoke_to_cost(
        self,
        player: PlayerState,
        cost: ManaCost,
    ) -> tuple[ManaCost, list[Card]]:
        """Apply convoke taps to reduce colored then generic costs."""
        candidates = self._get_convoke_candidates(player)
        if not candidates:
            return cost, []

        def _sort_key(card: Card) -> tuple[int, int]:
            can_attack = player.can_attack_with(card)
            return (1 if can_attack else 0, card.power)

        candidates.sort(key=_sort_key)
        tapped: list[Card] = []

        remaining = ManaCost(
            generic=cost.generic,
            white=cost.white,
            blue=cost.blue,
            black=cost.black,
            red=cost.red,
            green=cost.green,
        )

        color_map = [
            ("white", "W"),
            ("blue", "U"),
            ("black", "B"),
            ("red", "R"),
            ("green", "G"),
        ]

        for attr, symbol in color_map:
            needed = getattr(remaining, attr)
            while needed > 0:
                idx = next(
                    (i for i, c in enumerate(candidates) if symbol in self._creature_colors(c)),
                    None,
                )
                if idx is None:
                    break
                tapped.append(candidates.pop(idx))
                setattr(remaining, attr, getattr(remaining, attr) - 1)
                needed -= 1

        if remaining.generic > 0 and candidates:
            tap_count = min(remaining.generic, len(candidates))
            tapped.extend(candidates[:tap_count])
            remaining = remaining.reduce_generic(tap_count)

        return remaining, tapped

    def _handle_leave_battlefield(
        self,
        state: GameState,
        card: Card,
    ) -> None:
        """Handle leave-the-battlefield effects."""
        if card.name not in {"Leyline Binding", "Detention Sphere"} or not card.exiled_cards:
            return
        for exiled in list(card.exiled_cards):
            owner_idx = exiled.owner_player_idx
            if owner_idx is None:
                continue
            owner = state.players[owner_idx]
            if exiled in owner.exile:
                owner.exile.remove(exiled)
                owner.battlefield.append(exiled)
                if exiled.card_type == CardType.CREATURE and Keyword.HASTE not in exiled.keywords:
                    owner.summoning_sick.add(exiled.card_id)
        card.exiled_cards.clear()

    def _effective_mana_cost(
        self,
        state: GameState,
        card: Card,
        player_idx: int,
    ) -> ManaCost:
        """Apply cost reductions (domain, Haughty Djinn, etc.)."""
        cost = card.mana_cost
        if card.has_domain_cost_reduction:
            domain = state.players[player_idx].domain_count
            cost = cost.reduce_generic(domain)
        # Haughty Djinn: instant and sorcery spells cost 1 less
        player = state.players[player_idx]
        for c in player.battlefield:
            if c.name == "Haughty Djinn" and card.card_type in (
                CardType.INSTANT,
                CardType.SORCERY,
            ):
                cost = cost.reduce_generic(1)
        return cost

    def _get_convoke_candidates(self, player: PlayerState) -> list[Card]:
        """Get untapped creatures that can be tapped for convoke."""
        candidates = []
        for card in player.battlefield:
            if card.card_id in player.tapped_permanents:
                continue
            if card.card_type == CardType.CREATURE or card.card_id in player.activated_creatures:
                candidates.append(card)
        return candidates

    def _select_convoke_creatures(
        self,
        player: PlayerState,
        cost: ManaCost,
    ) -> tuple[ManaCost, list[Card]]:
        """Select creatures to tap for convoke and reduce cost."""
        return self._apply_convoke_to_cost(player, cost)

    def _update_haughty_djinn_power(self, player: PlayerState) -> None:
        """Update Haughty Djinn power based on instants/sorceries in graveyard."""
        instant_sorcery_count = sum(
            1 for c in player.graveyard if c.card_type in {CardType.INSTANT, CardType.SORCERY}
        )
        for card in player.battlefield:
            if card.name == "Haughty Djinn":
                card.power = instant_sorcery_count

    def _get_effective_power_toughness(
        self,
        card: Card,
        owner: PlayerState,
    ) -> tuple[int, int]:
        """Get effective power/toughness including bonuses, counters, and temporary buffs."""
        if card.card_id in owner.activated_creatures:
            base_power, base_toughness = owner.activated_creatures[card.card_id]
        else:
            base_power = card.power + card.permanent_power_bonus
            base_toughness = card.toughness + card.permanent_toughness_bonus

        # +1/+1 and -1/-1 counters
        base_power += card.plus_counters - card.minus_counters
        base_toughness += card.plus_counters - card.minus_counters

        if card.current_power is not None:
            delta = card.current_power - (card.power + card.permanent_power_bonus)
            base_power += delta
        if card.current_toughness is not None:
            delta = card.current_toughness - (card.toughness + card.permanent_toughness_bonus)
            base_toughness += delta

        return base_power, base_toughness

    def _get_mana_sources(self, player: PlayerState) -> list[Card]:
        return [
            c
            for c in player.battlefield
            if c.produces_mana and c.card_id not in player.tapped_permanents
        ]

    def _select_mana_sources_for_cost(
        self,
        sources: list[Card],
        cost: ManaCost,
    ) -> list[int]:
        source_colors = [{color.value for color in s.produces_mana} for s in sources]
        required_colors: list[str] = (
            ["W"] * cost.white
            + ["U"] * cost.blue
            + ["B"] * cost.black
            + ["R"] * cost.red
            + ["G"] * cost.green
        )

        remaining = list(range(len(source_colors)))
        chosen: list[int] = []

        for color in required_colors:
            candidates = [idx for idx in remaining if color in source_colors[idx]]
            if not candidates:
                return []
            best_idx = min(candidates, key=lambda i: len(source_colors[i]))
            remaining.remove(best_idx)
            chosen.append(sources[best_idx].card_id)

        generic_needed = cost.generic
        for idx in remaining:
            if generic_needed <= 0:
                break
            chosen.append(sources[idx].card_id)
            generic_needed -= 1

        if generic_needed > 0:
            return []

        return chosen

    def _can_pay_with_sources(self, sources: list[Card], cost: ManaCost) -> bool:
        return bool(self._select_mana_sources_for_cost(sources, cost))

    def _pay_mana_cost_with_sources(
        self,
        player: PlayerState,
        source_ids: list[int],
    ) -> None:
        for card_id in source_ids:
            player.tap_permanent(card_id)

    def start_spell_cast(
        self,
        state: GameState,
        card: Card,
        player_idx: int,
    ) -> GameState:
        """Begin casting a spell by entering target or mana selection."""
        pending = StackItem(source=card, controller=player_idx, targets=[], item_type="spell")
        state.pending_spell = pending
        state.pending_player = player_idx
        state.pending_action_type = "spell_target"
        state.pending_target_candidates = self.get_spell_target_candidates(state, card, player_idx)

        if not state.pending_target_candidates:
            state.pending_action_type = "mana_payment"
            state.pending_target_candidates = []

        state.pending_mana_cost = self._effective_mana_cost(state, card, player_idx)
        state.pending_mana_sources = [
            c.card_id for c in self._get_mana_sources(state.players[player_idx])
        ]
        state.pending_mana_chosen = []

        return state

    def start_activation(
        self,
        state: GameState,
        card: Card,
        player_idx: int,
    ) -> GameState:
        """Begin activating an ability."""
        state.pending_activation = card
        state.pending_player = player_idx
        state.pending_action_type = "mana_payment"
        state.pending_target_candidates = []
        state.pending_mana_cost = card.land_props.activation_cost if card.land_props else ManaCost()
        state.pending_mana_sources = [
            c.card_id for c in self._get_mana_sources(state.players[player_idx])
        ]
        state.pending_mana_chosen = []
        return state

    def finalize_pending_spell(self, state: GameState) -> GameState:
        """Finalize a pending spell, pay costs, and place on stack."""
        if not state.pending_spell:
            return state
        player_idx = (
            state.pending_player if state.pending_player is not None else state.priority_player
        )
        player = state.players[player_idx]
        card = state.pending_spell.source

        if card not in player.hand:
            return state

        # Pay mana (apply domain reduction, ward adjustments, and convoke)
        effective_cost = self._effective_mana_cost(state, card, player_idx)
        if state.pending_mana_cost:
            effective_cost = state.pending_mana_cost
        if card.has_convoke:
            effective_cost, convoke_creatures = self._apply_convoke_to_cost(player, effective_cost)
            for creature in convoke_creatures:
                player.tap_permanent(creature.card_id)

        if state.pending_mana_chosen:
            sources = [
                c for c in self._get_mana_sources(player) if c.card_id in state.pending_mana_chosen
            ]
            if not self._can_pay_with_sources(sources, effective_cost):
                return state
            self._pay_mana_cost_with_sources(player, state.pending_mana_chosen)
        else:
            sources = self._get_mana_sources(player)
            chosen = self._select_mana_sources_for_cost(sources, effective_cost)
            if not chosen:
                return state
            self._pay_mana_cost_with_sources(player, chosen)

        player.hand.remove(card)
        state.stack.append(state.pending_spell)

        # Fire cast triggers
        trigger_events = []
        if card.card_type == CardType.CREATURE:
            state, events = TriggerEngine.fire_cast_creature_triggers(state, player_idx)
            trigger_events.extend(events)
        else:
            state, events = TriggerEngine.fire_cast_noncreature_triggers(state, player_idx)
            trigger_events.extend(events)

        triggered_abilities = [e.effect_applied for e in trigger_events if e.effect_applied]

        details = {
            "card_type": card.card_type.value,
            "mana_cost": card.mana_cost.to_text(),
            "triggered_abilities": triggered_abilities,
            "card_id": card.card_id,
        }
        if state.pending_spell.targets:
            details["target"] = ", ".join(t.name for t in state.pending_spell.targets)
            if len(state.pending_spell.targets) == 1:
                target = state.pending_spell.targets[0]
                details["target_kind"] = target.kind
                details["target_id"] = target.ref_id
                # Determine target owner for creature targets
                if target.kind == "creature":
                    opponent = state.players[1 - player_idx]
                    if any(c.card_id == target.ref_id for c in opponent.battlefield):
                        details["target_owner"] = 1 - player_idx  # Opponent's creature
                    else:
                        details["target_owner"] = player_idx  # Own creature
                elif target.kind == "player":
                    details["target_owner"] = target.ref_id
        if card.deals_damage:
            details["deals_damage"] = card.deals_damage

        state.log_action(
            player=player_idx,
            action_type="CAST",
            card_name=card.name,
            details=details,
        )

        # Up the Beanstalk trigger: draw when casting spells with MV >= 5
        if card.mana_cost.cmc >= 5:
            for perm in player.battlefield:
                if perm.name == "Up the Beanstalk":
                    drawn = player.draw_cards(1)
                    if drawn:
                        state.log_action(
                            player=player_idx,
                            action_type="TRIGGER",
                            card_name="Up the Beanstalk",
                            details={"effect": "draw", "count": 1},
                        )

        state.passed_priority = [False, False]
        state.pending_spell = None
        state.pending_action_type = None
        state.pending_player = None
        state.pending_target_candidates = []
        state.pending_mana_cost = None
        state.pending_mana_sources = []
        state.pending_mana_chosen = []

        return state

    def finalize_pending_activation(self, state: GameState) -> GameState:
        """Finalize a pending activated ability."""
        if not state.pending_activation:
            return state
        player_idx = (
            state.pending_player if state.pending_player is not None else state.priority_player
        )
        player = state.players[player_idx]
        card = state.pending_activation

        if state.pending_mana_chosen:
            sources = [
                c for c in self._get_mana_sources(player) if c.card_id in state.pending_mana_chosen
            ]
            if not self._can_pay_with_sources(sources, card.land_props.activation_cost):
                return state
            self._pay_mana_cost_with_sources(player, state.pending_mana_chosen)
        else:
            sources = self._get_mana_sources(player)
            chosen = self._select_mana_sources_for_cost(sources, card.land_props.activation_cost)
            if not chosen:
                return state
            self._pay_mana_cost_with_sources(player, chosen)

        state.stack.append(
            StackItem(source=card, controller=player_idx, targets=[], item_type="ability")
        )

        state.log_action(
            player=player_idx,
            action_type="ACTIVATE",
            card_name=card.name,
            details={"mana_cost": card.land_props.activation_cost.to_text()},
        )

        state.passed_priority = [False, False]
        state.pending_activation = None
        state.pending_action_type = None
        state.pending_player = None
        state.pending_mana_cost = None
        state.pending_mana_sources = []
        state.pending_mana_chosen = []

        return state

    def handle_pending_action(
        self,
        state: GameState,
        action_kind: str,
        slot: int,
        player_idx: int,
    ) -> GameState:
        """Handle multi-step selections for targets, combat, discard, mana."""
        if state.pending_action_type == "mulligan_bottom":
            if action_kind == "bottom_card" and slot < len(state.players[player_idx].hand):
                if slot in state.pending_selected_indices:
                    state.pending_selected_indices.remove(slot)
                else:
                    state.pending_selected_indices.append(slot)
                # Auto-apply as soon as enough cards are selected
                if len(state.pending_selected_indices) >= state.pending_required:
                    state = self._apply_bottom_cards(state, player_idx)
            elif action_kind == "confirm":
                if len(state.pending_selected_indices) >= state.pending_required:
                    state = self._apply_bottom_cards(state, player_idx)
                elif state.pending_required > 0:
                    # Not enough cards selected; auto-select remaining from unselected
                    hand_size = len(state.players[player_idx].hand)
                    unselected = [
                        i for i in range(hand_size) if i not in state.pending_selected_indices
                    ]
                    need = state.pending_required - len(state.pending_selected_indices)
                    state.pending_selected_indices.extend(unselected[:need])
                    state = self._apply_bottom_cards(state, player_idx)
            elif action_kind == "cancel":
                state.pending_selected_indices.clear()
            return state

        if state.pending_action_type == "discard":
            if action_kind == "discard_card" and slot < len(state.players[player_idx].hand):
                if slot in state.pending_selected_indices:
                    state.pending_selected_indices.remove(slot)
                else:
                    state.pending_selected_indices.append(slot)
            elif action_kind == "confirm":
                if len(state.pending_selected_indices) >= state.pending_required:
                    state = self._apply_discards(state, player_idx)
            elif action_kind == "cancel":
                state.pending_selected_indices.clear()
            return state

        if state.pending_action_type == "spell_target":
            if action_kind == "target" and slot < len(state.pending_target_candidates):
                chosen_target = state.pending_target_candidates[slot]
                state.pending_spell.targets = [chosen_target]
                if (
                    chosen_target.kind in {"creature", "permanent"}
                    and state.pending_spell
                    and state.pending_mana_cost
                ):
                    target_card = state.players[player_idx].get_card_by_id(
                        chosen_target.ref_id
                    ) or state.get_opponent(player_idx).get_card_by_id(chosen_target.ref_id)
                    if (
                        target_card
                        and target_card.ward_cost > 0
                        and target_card.owner_player_idx is not None
                        and target_card.owner_player_idx != player_idx
                    ):
                        state.pending_mana_cost = state.pending_mana_cost.add_generic(
                            target_card.ward_cost
                        )
                state.pending_action_type = "mana_payment"
                state.pending_target_candidates = []
            elif action_kind == "cancel":
                state.pending_spell = None
                state.pending_action_type = None
                state.pending_target_candidates = []
                state.pending_mana_cost = None
                state.pending_mana_sources = []
                state.pending_mana_chosen = []
            return state

        if state.pending_action_type == "mana_payment":
            if action_kind == "mana_source" and slot < len(state.pending_mana_sources):
                card_id = state.pending_mana_sources[slot]
                if card_id in state.pending_mana_chosen:
                    state.pending_mana_chosen.remove(card_id)
                else:
                    state.pending_mana_chosen.append(card_id)
            elif action_kind == "auto_pay":
                state.pending_mana_chosen = []
                if state.pending_spell:
                    state = self.finalize_pending_spell(state)
                elif state.pending_activation:
                    state = self.finalize_pending_activation(state)
            elif action_kind == "confirm":
                if state.pending_spell:
                    state = self.finalize_pending_spell(state)
                elif state.pending_activation:
                    state = self.finalize_pending_activation(state)
            elif action_kind == "cancel":
                state.pending_spell = None
                state.pending_activation = None
                state.pending_action_type = None
                state.pending_player = None
                state.pending_target_candidates = []
                state.pending_mana_cost = None
                state.pending_mana_sources = []
                state.pending_mana_chosen = []
            return state

        if state.pending_action_type == "attack":
            player = state.players[player_idx]
            attackers = [c for c in player.battlefield if player.can_attack_with(c)]
            if action_kind == "attack_toggle" and slot < len(attackers):
                card_id = attackers[slot].card_id
                if card_id in state.pending_attackers:
                    state.pending_attackers.remove(card_id)
                else:
                    state.pending_attackers.add(card_id)
            elif action_kind == "confirm":
                if state.pending_attackers:
                    state = self.declare_attackers(state, list(state.pending_attackers))
                else:
                    state = self.advance_phase(state)
                state.pending_action_type = None
                state.pending_attackers = set()
            elif action_kind == "cancel":
                state.pending_attackers.clear()
            return state

        if state.pending_action_type == "block":
            defender = state.players[player_idx]
            attacker_state = state.get_active_player()
            attackers = [
                attacker_state.get_card_by_id(cid) for cid in attacker_state.declared_attackers
            ]
            attackers = [c for c in attackers if c]
            blockers = [c for c in defender.battlefield if defender.can_block_with(c)]

            if action_kind == "block_select_attacker" and slot < len(attackers):
                state.pending_block_attacker_id = attackers[slot].card_id
            elif action_kind == "block_select_blocker" and slot < len(blockers):
                if state.pending_block_attacker_id is None:
                    return state
                state.pending_block_assignments[blockers[slot].card_id] = (
                    state.pending_block_attacker_id
                )
            elif action_kind == "confirm":
                state = self.declare_blockers(state, state.pending_block_assignments)
                state.pending_block_assignments = {}
                state.pending_block_attacker_id = None
                state.pending_action_type = None
            elif action_kind == "cancel":
                state.pending_block_assignments = {}
                state.pending_block_attacker_id = None
            return state

        return state

    def _is_creature_buff_instant(self, card: Card) -> bool:
        """Check if an instant targets our own creature as a buff/pump."""
        return card.card_type == CardType.INSTANT and (
            card.is_pump_spell
            or (card.requires_creature_target and not card.deals_damage and not card.is_removal)
        )

    def _should_cast_creature_buff_instant(self, state: GameState, player_idx: int) -> bool:
        """Heuristic for when creature-targeting buff instants are worth casting."""
        player = state.players[player_idx]

        # Avoid forced discard at cleanup if possible
        if len(player.hand) > 7:
            return True

        is_active_player = state.active_player == player_idx
        phase = state.phase

        # Cast pump spells only during combat phases (not main phase)
        # This ensures we maximize prowess triggers and don't telegraph our play
        if is_active_player and phase in {
            GamePhase.COMBAT_BEGIN,
            GamePhase.COMBAT_ATTACKERS,
        }:
            return any(
                player.can_attack_with(c)
                for c in player.battlefield
                if c.card_type == CardType.CREATURE
            )

        # Cast if it will matter for blocks while defending
        if not is_active_player and phase in {
            GamePhase.COMBAT_ATTACKERS,
            GamePhase.COMBAT_BLOCKERS,
        }:
            opponent = state.get_active_player()
            has_attackers = bool(opponent.declared_attackers)
            has_blockers = any(
                player.can_block_with(c)
                for c in player.battlefield
                if c.card_type == CardType.CREATURE
            )
            return has_attackers and has_blockers

        return False

    def can_play_land(self, state: GameState, card: Card) -> bool:
        """Check if a land can be played."""
        if card.card_type != CardType.LAND:
            return False

        if state.priority_player != state.active_player:
            return False

        if not state.can_cast_sorcery_speed():
            return False

        player = state.get_active_player()
        if player.lands_played_this_turn >= 1:
            return False

        return card in player.hand

    def play_land(self, state: GameState, card: Card) -> GameState:
        """Play a land from hand."""
        player_idx = state.active_player
        player = state.get_active_player()

        player.hand.remove(card)
        player.battlefield.append(card)
        player.lands_played_this_turn += 1

        enters_tapped = card.enters_tapped
        # Conditional ETB tapped: enters tapped if you control > N other lands
        if card.land_props and card.land_props.enters_tapped_conditional:
            threshold = card.land_props.enters_tapped_conditional
            other_lands = sum(
                1
                for c in player.battlefield
                if c.card_type == CardType.LAND and c.card_id != card.card_id
            )
            if other_lands >= threshold:
                enters_tapped = True
        # Shocklands: pay 2 life to enter untapped (heuristic: pay if >= 5 life)
        if card.land_props and getattr(card.land_props, "life_cost", 0) > 0:
            if player.life > 4:
                player.life -= card.land_props.life_cost
            else:
                enters_tapped = True
        if enters_tapped:
            player.tapped_permanents.add(card.card_id)

        # Update domain count (number of basic land types controlled)
        player.update_domain_count()

        # Fire landfall triggers
        landfall_events = []
        landfall_triggers = TriggerEngine.get_triggers_for_event(
            state, TriggerType.ETB_LAND, player_idx
        )
        for trigger_card, trigger in landfall_triggers:
            state, event = TriggerEngine.apply_trigger(state, trigger_card, trigger, player_idx)
            if event.effect_applied:
                landfall_events.append(event.effect_applied)

        # Count lands on board by type
        all_lands: dict[str, int] = {}
        for c in player.battlefield:
            if c.produces_mana:
                all_lands[c.name] = all_lands.get(c.name, 0) + 1

        land_count = all_lands.get(card.name, 1)

        # Log the action with full land breakdown
        state.log_action(
            player=player_idx,
            action_type="PLAY_LAND",
            card_name=card.name,
            details={
                "land_count": land_count,
                "all_lands": all_lands,
                "domain_count": player.domain_count,
                "landfall_triggers": landfall_events,
                "enters_tapped": card.enters_tapped,
            },
        )

        return state

    def cast_spell(
        self,
        state: GameState,
        card: Card,
        targets: list[TargetRef] | None = None,
    ) -> GameState:
        """Cast a spell and place it on the stack (auto-targeted)."""
        player_idx = state.priority_player
        player = state.players[player_idx]

        if card not in player.hand:
            return state

        if targets is None:
            targets = self._auto_choose_targets(state, card, player_idx)

        effective_cost = self._effective_mana_cost(state, card, player_idx)
        if card.has_convoke:
            effective_cost, convoke_creatures = self._apply_convoke_to_cost(player, effective_cost)
            for creature in convoke_creatures:
                player.tap_permanent(creature.card_id)

        sources = self._get_mana_sources(player)
        chosen = self._select_mana_sources_for_cost(sources, effective_cost)
        if not chosen:
            return state

        self._pay_mana_cost_with_sources(player, chosen)
        player.hand.remove(card)

        state.stack.append(
            StackItem(source=card, controller=player_idx, targets=targets, item_type="spell")
        )

        trigger_events = []
        if card.card_type == CardType.CREATURE:
            state, events = TriggerEngine.fire_cast_creature_triggers(state, player_idx)
            trigger_events.extend(events)
        else:
            state, events = TriggerEngine.fire_cast_noncreature_triggers(state, player_idx)
            trigger_events.extend(events)

        triggered_abilities = [e.effect_applied for e in trigger_events if e.effect_applied]
        details = {
            "card_type": card.card_type.value,
            "mana_cost": card.mana_cost.to_text(),
            "triggered_abilities": triggered_abilities,
            "card_id": card.card_id,
        }
        if targets:
            details["target"] = ", ".join(t.name for t in targets)
            if len(targets) == 1:
                details["target_kind"] = targets[0].kind
                details["target_id"] = targets[0].ref_id
        if card.deals_damage:
            details["deals_damage"] = card.deals_damage
        # For Haughty Djinn, include graveyard count to show expected power
        if card.name == "Haughty Djinn":
            gy_instant_sorcery = sum(
                1 for c in player.graveyard if c.card_type in (CardType.INSTANT, CardType.SORCERY)
            )
            details["graveyard_instant_sorcery_count"] = gy_instant_sorcery

        state.log_action(
            player=player_idx,
            action_type="CAST",
            card_name=card.name,
            details=details,
        )

        state.passed_priority = [False, False]
        return state

    def _pay_mana_cost(self, player: PlayerState, card: Card) -> None:
        """Tap lands to pay a mana cost."""
        cost = card.mana_cost
        lands_to_tap: list[Card] = []

        mana_sources = [
            c
            for c in player.battlefield
            if c.produces_mana and c.card_id not in player.tapped_permanents
        ]

        # Tap colored sources for colored costs
        needed = {
            "W": cost.white,
            "U": cost.blue,
            "B": cost.black,
            "R": cost.red,
            "G": cost.green,
        }

        for source in mana_sources:
            for color in source.produces_mana:
                if needed.get(color.value, 0) > 0:
                    needed[color.value] -= 1
                    lands_to_tap.append(source)
                    break

        # Tap remaining for generic cost
        generic_needed = cost.generic
        for source in mana_sources:
            if source not in lands_to_tap and generic_needed > 0:
                lands_to_tap.append(source)
                generic_needed -= 1

        for land in lands_to_tap:
            player.tap_permanent(land.card_id)

    def _auto_choose_targets(
        self,
        state: GameState,
        card: Card,
        player_idx: int,
    ) -> list[TargetRef]:
        candidates = self.get_spell_target_candidates(state, card, player_idx)
        if not candidates:
            return []

        player = state.players[player_idx]
        opponent = state.get_opponent(player_idx)

        if card.is_counterspell:
            return candidates[:1]

        if card.is_pump_spell or (
            card.requires_creature_target and not card.deals_damage and not card.is_removal
        ):
            best = None
            best_score = -1
            for candidate in candidates:
                creature = player.get_card_by_id(candidate.ref_id)
                if creature:
                    eff_power, _ = self._get_effective_power_toughness(creature, player)
                    if eff_power > best_score:
                        best = candidate
                        best_score = eff_power
            return [best] if best else candidates[:1]

        if card.requires_creature_target:
            best = None
            best_score = -1
            for candidate in candidates:
                creature = opponent.get_card_by_id(candidate.ref_id)
                if creature:
                    eff_power, _ = self._get_effective_power_toughness(creature, opponent)
                    if eff_power > best_score:
                        best = candidate
                        best_score = eff_power
            return [best] if best else candidates[:1]

        if card.can_target_any:
            # Check for lethal first
            if opponent.life <= card.deals_damage and card.deals_damage > 0:
                return [c for c in candidates if c.kind == "player" and c.ref_id == 1 - player_idx][
                    :1
                ]

            # Calculate board state to decide target (effective power)
            my_power = sum(
                self._get_effective_power_toughness(c, player)[0]
                for c in player.battlefield
                if c.card_type == CardType.CREATURE
            )
            opp_power = sum(
                self._get_effective_power_toughness(c, opponent)[0]
                for c in opponent.battlefield
                if c.card_type == CardType.CREATURE
            )
            behind_on_board = opp_power > my_power

            if behind_on_board:
                best_creature = None
                best_power = 0
                for candidate in candidates:
                    if candidate.kind == "creature":
                        creature = opponent.get_card_by_id(candidate.ref_id)
                        if creature:
                            _, eff_tough = self._get_effective_power_toughness(creature, opponent)
                            eff_pow, _ = self._get_effective_power_toughness(creature, opponent)
                            if eff_tough <= card.deals_damage and eff_pow > best_power:
                                best_creature = candidate
                                best_power = eff_pow
                if best_creature:
                    return [best_creature]

            for candidate in candidates:
                if candidate.kind == "player" and candidate.ref_id == 1 - player_idx:
                    return [candidate]
            return candidates[:1]

        return candidates[:1]

    def resolve_top_of_stack(self, state: GameState) -> GameState:
        """Resolve the top spell/ability on the stack."""
        if not state.stack:
            return state
        item = state.stack.pop()
        if item.item_type == "spell":
            state = self._resolve_spell(state, item)
        elif item.item_type == "ability":
            state = self._resolve_ability(state, item)
        state.check_game_over()
        return state

    def _resolve_spell(self, state: GameState, item: StackItem) -> GameState:
        player_idx = item.controller
        player = state.players[player_idx]
        opponent = state.get_opponent(player_idx)
        card = item.source
        targets = item.targets or []

        if card.card_type == CardType.CREATURE:
            player.battlefield.append(card)
            if Keyword.HASTE not in card.keywords:
                player.summoning_sick.add(card.card_id)

            # Update Haughty Djinn power if it enters the battlefield
            if card.name == "Haughty Djinn":
                self._update_haughty_djinn_power(player)

            # Topiary Stomper ETB: search a basic land and put tapped
            if card.name == "Topiary Stomper":
                for i, deck_card in enumerate(player.deck):
                    if deck_card.land_props and deck_card.land_props.is_basic:
                        land = player.deck.pop(i)
                        player.battlefield.append(land)
                        if land.enters_tapped:
                            player.tapped_permanents.add(land.card_id)
                        player.update_domain_count()
                        state.log_action(
                            player=player_idx,
                            action_type="SEARCH_LAND",
                            card_name=land.name,
                            details={"source": "Topiary Stomper"},
                        )
                        break

            # Knight-Errant of Eos ETB: put up to two small creatures from top six
            if card.name == "Knight-Errant of Eos":
                top_cards = player.deck[:6]
                player.deck = player.deck[6:]
                chosen: list[Card] = []
                rest: list[Card] = []
                for top in top_cards:
                    if (
                        len(chosen) < 2
                        and top.card_type == CardType.CREATURE
                        and top.mana_cost.cmc <= 2
                    ):
                        chosen.append(top)
                    else:
                        rest.append(top)
                for creature in chosen:
                    player.battlefield.append(creature)
                    if Keyword.HASTE not in creature.keywords:
                        player.summoning_sick.add(creature.card_id)
                player.deck.extend(rest)  # bottom in order
                if chosen:
                    state.log_action(
                        player=player_idx,
                        action_type="ETB_PUT_CREATURES",
                        card_name=card.name,
                        details={"count": len(chosen), "names": [c.name for c in chosen]},
                    )

            # Atraxa ETB: reveal top 10, take one of each type (simplified)
            if card.name == "Atraxa, Grand Unifier":
                top_cards = player.deck[:10]
                player.deck = player.deck[10:]
                chosen_types: set[CardType] = set()
                kept: list[Card] = []
                rest: list[Card] = []
                for top in top_cards:
                    if top.card_type not in chosen_types:
                        kept.append(top)
                        chosen_types.add(top.card_type)
                    else:
                        rest.append(top)
                player.hand.extend(kept)
                player.deck.extend(rest)  # bottom in order
                if kept:
                    state.log_action(
                        player=player_idx,
                        action_type="ETB_DRAW",
                        card_name=card.name,
                        details={"count": len(kept)},
                    )
        elif card.card_type in [CardType.INSTANT, CardType.SORCERY]:
            target = targets[0] if targets else None
            if card.name == "Sunfall":
                exiled_count = 0
                for owner_idx in [player_idx, 1 - player_idx]:
                    owner = state.players[owner_idx]
                    for creature in list(owner.battlefield):
                        if (
                            creature.card_type == CardType.CREATURE
                            or creature.card_id in owner.activated_creatures
                        ):
                            owner.battlefield.remove(creature)
                            owner.exile.append(creature)
                            self._handle_leave_battlefield(state, creature)
                            exiled_count += 1
                if exiled_count > 0:
                    incubator = TriggerEngine.create_artifact_token(
                        "Incubator Token", player_idx, state
                    )
                    if incubator:
                        player.battlefield.append(incubator)
                state.log_action(
                    player=player_idx,
                    action_type="EXILE_ALL",
                    card_name=card.name,
                    details={"exiled": exiled_count, "incubate_value": exiled_count},
                )
            elif card.is_counterspell and target and target.kind == "stack":
                countered_spell = None
                target_idx = target.ref_id
                if isinstance(target_idx, int) and 0 <= target_idx < len(state.stack):
                    countered_spell = state.stack.pop(target_idx)
                if countered_spell:
                    countered_card = countered_spell.source
                    target_player = state.players[countered_spell.controller]
                    # Exile if "No More Lies" or similar, otherwise graveyard
                    if "exile" in card.rules_text.lower():
                        target_player.exile.append(countered_card)
                        destination = "exile"
                    else:
                        target_player.graveyard.append(countered_card)
                        destination = "graveyard"
                    state.log_action(
                        player=player_idx,
                        action_type="COUNTER",
                        card_name=card.name,
                        details={
                            "countered": countered_card.name,
                            "destination": destination,
                        },
                    )
            elif card.is_pump_spell or (
                card.requires_creature_target and not card.deals_damage and not card.is_removal
            ):
                if target and target.kind == "creature":
                    target_creature = player.get_card_by_id(target.ref_id)
                    if target_creature:
                        power_boost = 2
                        toughness_boost = 0
                        for trigger in card.triggers:
                            if trigger.effect == TriggerEffect.POWER_TOUGHNESS_BOOST:
                                power_boost = trigger.effect_value
                                toughness_boost = trigger.effect_value
                        pre_power, pre_toughness = self._get_effective_power_toughness(
                            target_creature, player
                        )
                        base_power = target_creature.power + target_creature.permanent_power_bonus
                        base_tough = (
                            target_creature.toughness + target_creature.permanent_toughness_bonus
                        )
                        if target_creature.current_power is None:
                            target_creature.current_power = base_power
                        if target_creature.current_toughness is None:
                            target_creature.current_toughness = base_tough
                        target_creature.current_power += power_boost
                        target_creature.current_toughness += toughness_boost
                        buff_result_power = pre_power + power_boost
                        buff_result_toughness = pre_toughness + toughness_boost
                        role_bonus = 0
                        role_result_power = None
                        role_result_toughness = None
                        if card.name == "Monstrous Rage":
                            role_name = "Monster Role"
                            existing_roles = target_creature.attached_tokens.count(role_name)
                            if existing_roles > 1:
                                extras = existing_roles - 1
                                target_creature.permanent_power_bonus -= extras
                                target_creature.permanent_toughness_bonus -= extras
                                target_creature.current_power -= extras
                                target_creature.current_toughness -= extras
                                pruned_tokens = []
                                kept_role = False
                                for token in target_creature.attached_tokens:
                                    if token == role_name:
                                        if kept_role:
                                            continue
                                        kept_role = True
                                    pruned_tokens.append(token)
                                target_creature.attached_tokens = pruned_tokens
                                existing_roles = 1
                            if existing_roles == 0:
                                target_creature.attached_tokens.append(role_name)
                                target_creature.permanent_power_bonus += 1
                                target_creature.permanent_toughness_bonus += 1
                                target_creature.current_power += 1
                                target_creature.current_toughness += 1
                                role_bonus = 1
                                role_result_power = buff_result_power + 1
                                role_result_toughness = buff_result_toughness + 1
                        resolved_power, resolved_toughness = self._get_effective_power_toughness(
                            target_creature, player
                        )
                        state.log_action(
                            player=player_idx,
                            action_type="RESOLVE",
                            card_name=card.name,
                            details={
                                "target": target_creature.name,
                                "target_id": target_creature.card_id,
                                "buff_power": power_boost,
                                "buff_toughness": toughness_boost,
                                "buff_result_power": buff_result_power,
                                "buff_result_toughness": buff_result_toughness,
                                "role_bonus": role_bonus,
                                "role_result_power": role_result_power,
                                "role_result_toughness": role_result_toughness,
                                "new_power": resolved_power,
                                "new_toughness": resolved_toughness,
                                "tokens": target_creature.attached_tokens.copy(),
                            },
                        )
            elif (
                card.can_target_nonland_permanent
                and target
                and target.kind in {"creature", "permanent"}
            ):
                target_perm = opponent.get_card_by_id(target.ref_id) or player.get_card_by_id(
                    target.ref_id
                )
                if target_perm and target_perm.card_type != CardType.LAND:
                    owner = player if target_perm in player.battlefield else opponent
                    owner_idx = player_idx if owner is player else 1 - player_idx
                    owner.battlefield.remove(target_perm)
                    if card.name in {"Leyline Binding", "Detention Sphere"}:
                        owner.exile.append(target_perm)
                        card.exiled_cards.append(target_perm)
                        action = "EXILE"
                    elif card.name == "Get Lost":
                        owner.exile.append(target_perm)
                        for _ in range(2):
                            map_token = TriggerEngine.create_artifact_token(
                                "Map Token", owner_idx, state
                            )
                            if map_token:
                                owner.battlefield.append(map_token)
                        action = "EXILE"
                    else:
                        owner.graveyard.append(target_perm)
                        action = "DIES"
                    self._handle_leave_battlefield(state, target_perm)
                    state.log_action(
                        player=owner_idx,
                        action_type=action,
                        card_name=target_perm.name,
                        details={"card_id": target_perm.card_id, "source": card.name},
                    )
            elif card.requires_creature_target and (card.deals_damage > 0 or card.is_removal):
                if target and target.kind == "creature":
                    target_creature = opponent.get_card_by_id(target.ref_id)
                    if target_creature and target_creature in opponent.battlefield:
                        _, eff_tough = self._get_effective_power_toughness(
                            target_creature, opponent
                        )
                        is_indestructible = Keyword.INDESTRUCTIBLE in target_creature.keywords
                        killed = False

                        if card.deals_damage > 0:
                            # Mark damage on the creature
                            target_creature.damage_marked = (
                                getattr(target_creature, "damage_marked", 0) + card.deals_damage
                            )
                            # Check lethal: damage >= effective toughness
                            if target_creature.damage_marked >= eff_tough and not is_indestructible:
                                killed = True
                        if not killed and card.is_removal and not is_indestructible:
                            killed = True

                        if killed:
                            opponent.battlefield.remove(target_creature)
                            opponent.graveyard.append(target_creature)
                            self._handle_leave_battlefield(state, target_creature)
                            cause = "spell_damage" if card.deals_damage > 0 else "spell_removal"
                            state.log_action(
                                player=1 - player_idx,
                                action_type="DIES",
                                card_name=target_creature.name,
                                details={
                                    "card_id": target_creature.card_id,
                                    "cause": cause,
                                    "source": card.name,
                                },
                            )
                            state, events = TriggerEngine.fire_death_triggers(
                                state, target_creature, 1 - player_idx
                            )
                            for event in events:
                                if event.effect_applied:
                                    state.log_action(
                                        player=1 - player_idx,
                                        action_type="DEATH_TRIGGER",
                                        card_name=target_creature.name,
                                        details={
                                            "trigger": event.effect_applied,
                                            "damage": event.damage,
                                            "target_player_idx": event.target_player_idx,
                                        },
                                    )
                                    if event.damage and event.target_player_idx is not None:
                                        state.players[event.target_player_idx].life -= event.damage
                            # Searing Blood rider: 3 damage to controller
                            if card.name == "Searing Blood":
                                opponent.life -= 3
            elif card.can_target_any:
                if target and target.kind == "player":
                    if target.ref_id == player_idx:
                        player.life -= card.deals_damage
                    else:
                        opponent.life -= card.deals_damage
                elif target and target.kind == "creature":
                    target_creature = player.get_card_by_id(
                        target.ref_id
                    ) or opponent.get_card_by_id(target.ref_id)
                    if target_creature:
                        owner = player if target_creature in player.battlefield else opponent
                        if target_creature not in owner.battlefield:
                            pass  # target already left the battlefield
                        else:
                            _, eff_tough = self._get_effective_power_toughness(
                                target_creature, owner
                            )
                            is_indestructible = Keyword.INDESTRUCTIBLE in target_creature.keywords
                            killed = False
                            if card.deals_damage > 0:
                                target_creature.damage_marked = (
                                    getattr(target_creature, "damage_marked", 0) + card.deals_damage
                                )
                                if (
                                    target_creature.damage_marked >= eff_tough
                                    and not is_indestructible
                                ):
                                    killed = True
                            if not killed and card.is_removal and not is_indestructible:
                                killed = True
                            if killed:
                                owner.battlefield.remove(target_creature)
                                owner.graveyard.append(target_creature)
                                self._handle_leave_battlefield(state, target_creature)
                                owner_idx = player_idx if owner is player else 1 - player_idx
                                state.log_action(
                                    player=owner_idx,
                                    action_type="DIES",
                                    card_name=target_creature.name,
                                    details={
                                        "card_id": target_creature.card_id,
                                        "cause": "spell_damage",
                                        "source": card.name,
                                    },
                                )
                                state, events = TriggerEngine.fire_death_triggers(
                                    state, target_creature, owner_idx
                                )
                                for event in events:
                                    if event.effect_applied:
                                        state.log_action(
                                            player=owner_idx,
                                            action_type="DEATH_TRIGGER",
                                            card_name=target_creature.name,
                                            details={
                                                "trigger": event.effect_applied,
                                                "damage": event.damage,
                                                "target_player_idx": event.target_player_idx,
                                            },
                                        )
                                        if event.damage and event.target_player_idx is not None:
                                            state.players[
                                                event.target_player_idx
                                            ].life -= event.damage
            else:
                if card.draws_cards > 0:
                    # For Memory Deluge and similar "look at X, draw Y" cards,
                    # we simplify to just drawing Y cards but log what was drawn
                    drawn_cards = player.draw_cards(card.draws_cards)
                    if drawn_cards and card.name in {"Memory Deluge", "Drawn from Dreams"}:
                        state.log_action(
                            player=player_idx,
                            action_type="DRAW_SELECTION",
                            card_name=card.name,
                            details={
                                "cards_drawn": [c.name for c in drawn_cards],
                                "count": len(drawn_cards),
                            },
                        )
                if (
                    card.deals_damage > 0
                    and not card.requires_creature_target
                    and not card.can_target_any
                ):
                    opponent.life -= card.deals_damage
                if card.gains_life > 0:
                    player.life += card.gains_life

            # Flashback: exile instead of graveyard
            if item.is_flashback:
                player.exile.append(card)
            else:
                player.graveyard.append(card)
            self._update_haughty_djinn_power(player)

        elif card.card_type == CardType.ENCHANTMENT:
            player.battlefield.append(card)
            if card.name == "Up the Beanstalk":
                drawn = player.draw_cards(1)
                if drawn:
                    state.log_action(
                        player=player_idx,
                        action_type="ETB_DRAW",
                        card_name=card.name,
                        details={"count": 1},
                    )
            if card.name in {"Leyline Binding", "Detention Sphere"} and targets:
                target = targets[0]
                if target.kind in {"creature", "permanent"}:
                    target_perm = opponent.get_card_by_id(target.ref_id)
                    if target_perm and target_perm.card_type != CardType.LAND:
                        opponent.battlefield.remove(target_perm)
                        opponent.exile.append(target_perm)
                        card.exiled_cards.append(target_perm)
                        state.log_action(
                            player=player_idx,
                            action_type="EXILE",
                            card_name=target_perm.name,
                            details={"source": card.name},
                        )
        elif card.card_type == CardType.ARTIFACT:
            player.battlefield.append(card)
        elif card.card_type == CardType.PLANESWALKER:
            player.battlefield.append(card)
            if card.name == "The Wandering Emperor":
                # Simplified activation: exile tapped creature if available, else create token.
                tapped_creatures = [
                    c
                    for c in opponent.battlefield
                    if c.card_type == CardType.CREATURE and c.card_id in opponent.tapped_permanents
                ]
                if tapped_creatures:
                    target_creature = tapped_creatures[0]
                    opponent.battlefield.remove(target_creature)
                    opponent.exile.append(target_creature)
                    self._handle_leave_battlefield(state, target_creature)
                    state.log_action(
                        player=player_idx,
                        action_type="EXILE",
                        card_name=target_creature.name,
                        details={"source": card.name},
                    )
                else:
                    token = TriggerEngine.create_token(
                        "2/2 white Samurai creature with vigilance",
                        player_idx,
                        state,
                    )
                    if token:
                        player.battlefield.append(token)
                        player.summoning_sick.add(token.card_id)
                        state.log_action(
                            player=player_idx,
                            action_type="CREATE_TOKEN",
                            card_name="Samurai Token",
                            details={"source": card.name},
                        )

        trigger_events = []
        if card.card_type in [CardType.CREATURE, CardType.ENCHANTMENT, CardType.ARTIFACT]:
            state, events = TriggerEngine.fire_etb_triggers(state, card, player_idx)
            trigger_events.extend(events)

        return state

    def _resolve_ability(self, state: GameState, item: StackItem) -> GameState:
        player_idx = item.controller
        player = state.players[player_idx]
        card = item.source
        if card.land_props and card.land_props.has_activation:
            player.activated_creatures[card.card_id] = (
                card.land_props.activation_power,
                card.land_props.activation_toughness,
            )
        return state

    def declare_attackers(self, state: GameState, attacker_ids: list[int]) -> GameState:
        """Declare attackers for combat.

        Args:
            state: Current game state.
            attacker_ids: List of creature card_ids to attack with.

        Returns:
            Updated game state.

        """
        player_idx = state.active_player
        player = state.get_active_player()

        attacker_names = []
        attacker_cards = []
        for card_id in attacker_ids:
            card = player.get_card_by_id(card_id)
            if card and player.can_attack_with(card):
                player.declared_attackers.append(card_id)
                if Keyword.VIGILANCE not in card.keywords:
                    player.tap_permanent(card_id)
                attacker_names.append(card.name)
                attacker_cards.append(card)

        # Fire attack triggers for each attacker
        attack_trigger_events = []
        for card in attacker_cards:
            state, events = TriggerEngine.fire_attack_triggers(state, card, player_idx)
            attack_trigger_events.extend(events)

        triggered_abilities = [e.effect_applied for e in attack_trigger_events if e.effect_applied]

        # Log the action with current power/toughness
        if attacker_names:
            # Build attacker data with current stats
            attacker_data = []
            for card in attacker_cards:
                power, toughness = self._get_effective_power_toughness(card, player)
                attacker_data.append(
                    {
                        "name": card.name,
                        "power": power,
                        "toughness": toughness,
                        "tokens": card.attached_tokens.copy(),
                    }
                )
            state.log_action(
                player=player_idx,
                action_type="ATTACK",
                card_name=", ".join(attacker_names),
                details={
                    "attackers": attacker_names,
                    "attacker_data": attacker_data,
                    "triggered_abilities": triggered_abilities,
                },
            )

        # Move to attackers declared phase
        state.phase = GamePhase.COMBAT_ATTACKERS
        state.reset_priority()

        return state

    def declare_blockers(self, state: GameState, blocks: dict[int, int]) -> GameState:
        """Declare blockers for combat.

        Args:
            state: Current game state.
            blocks: Dict mapping blocker_id to attacker_id.

        Returns:
            Updated game state.

        """
        defender_idx = 1 - state.active_player  # The defending player
        defender = state.get_opponent()
        attacker = state.get_active_player()

        # Build multi-blocker map: attacker_id -> list of blocker_ids
        multi_blockers: dict[int, list[int]] = {}

        block_info = []
        block_data = []
        for blocker_id, attacker_id in blocks.items():
            blocker = defender.get_card_by_id(blocker_id)
            attacker_card = attacker.get_card_by_id(attacker_id)

            if (
                blocker
                and attacker_card
                and defender.can_block_with(blocker)
                and attacker_id in attacker.declared_attackers
            ):
                # Check flying - Reach creatures can block flyers
                if Keyword.FLYING in attacker_card.keywords:
                    can_block_flyer = (
                        Keyword.FLYING in blocker.keywords or Keyword.REACH in blocker.keywords
                    )
                    if not can_block_flyer:
                        continue  # Can't block flyer without flying or reach

                # Check protection - can't block if attacker has protection from blocker's colors
                # (simplified: check if blocker has any colored mana in its cost)
                blocker_colors = blocker.mana_cost.colors if blocker.mana_cost else []
                blocked_by_protection = any(
                    attacker_card.has_protection_from(color.value) for color in blocker_colors
                )
                if blocked_by_protection:
                    continue

                defender.declared_blockers[blocker_id] = attacker_id

                # Track multi-blockers
                if attacker_id not in multi_blockers:
                    multi_blockers[attacker_id] = []
                multi_blockers[attacker_id].append(blocker_id)

                block_info.append((blocker.name, attacker_card.name))
                blocker_power, blocker_toughness = self._get_effective_power_toughness(
                    blocker, defender
                )
                attacker_power, attacker_toughness = self._get_effective_power_toughness(
                    attacker_card, attacker
                )
                block_data.append(
                    {
                        "blocker_id": blocker.card_id,
                        "attacker_id": attacker_card.card_id,
                        "blocker": blocker.name,
                        "attacker": attacker_card.name,
                        "blocker_power": blocker_power,
                        "blocker_toughness": blocker_toughness,
                        "attacker_power": attacker_power,
                        "attacker_toughness": attacker_toughness,
                        "blocker_tokens": blocker.attached_tokens.copy(),
                        "attacker_tokens": attacker_card.attached_tokens.copy(),
                    }
                )

        # Validate menace - creatures with menace must be blocked by 2+ creatures
        # Remove invalid blocks where menace creature is blocked by only 1 creature
        for attacker_id in attacker.declared_attackers:
            attacker_card = attacker.get_card_by_id(attacker_id)
            if attacker_card and Keyword.MENACE in attacker_card.keywords:
                blockers_for_attacker = multi_blockers.get(attacker_id, [])
                if len(blockers_for_attacker) == 1:
                    # Invalid block - menace requires 2+ blockers
                    # Remove this block
                    blocker_id = blockers_for_attacker[0]
                    if blocker_id in defender.declared_blockers:
                        del defender.declared_blockers[blocker_id]
                    block_info = [(b, a) for b, a in block_info if b != blocker_id]
                    block_data = [
                        entry for entry in block_data if entry.get("blocker_id") != blocker_id
                    ]
                    multi_blockers[attacker_id] = []

        # Track attackers that were blocked (even if all blockers were
        # removed due to menace).  Needed so that blocked-without-trample
        # creatures don't deal damage to the defending player.
        defender.menace_blocked: set[int] = set()
        for attacker_id in attacker.declared_attackers:
            attacker_card = attacker.get_card_by_id(attacker_id)
            if (
                attacker_card
                and Keyword.MENACE in attacker_card.keywords
                and not multi_blockers.get(attacker_id)
                and (
                    any(entry.get("attacker_id") == attacker_id for entry in block_data)
                    or attacker_id in multi_blockers
                )
            ):
                defender.menace_blocked.add(attacker_id)

        # Store multi-blockers for damage assignment
        defender.multi_blockers = multi_blockers

        # Log the action
        if block_info:
            state.log_action(
                player=defender_idx,
                action_type="BLOCK",
                card_name=", ".join(b[0] for b in block_info),
                details={"blocks": block_info, "block_data": block_data},
            )

        state.phase = GamePhase.COMBAT_BLOCKERS
        state.reset_priority()

        return state

    def can_legally_block(
        self, blocker: Card, attacker: Card, defender: PlayerState, all_blocks: dict[int, int]
    ) -> bool:
        """Check if a blocker can legally block an attacker.

        Args:
            blocker: The blocking creature.
            attacker: The attacking creature.
            defender: Defending player state.
            all_blocks: All proposed blocks.

        Returns:
            True if the block is legal.
        """
        if not defender.can_block_with(blocker):
            return False

        # Flying check
        if (
            Keyword.FLYING in attacker.keywords
            and Keyword.FLYING not in blocker.keywords
            and Keyword.REACH not in blocker.keywords
        ):
            return False

        # Protection check
        blocker_colors = blocker.mana_cost.colors if blocker.mana_cost else []
        return not any(attacker.has_protection_from(c.value) for c in blocker_colors)

    def pass_priority(self, state: GameState) -> GameState:
        """Current player passes priority.

        Args:
            state: Current game state.

        Returns:
            Updated game state.

        """
        state.passed_priority[state.priority_player] = True

        if state.both_players_passed():
            # Both passed - resolve stack or advance phase
            if state.stack:
                state = self.resolve_top_of_stack(state)
                state.reset_priority()
            else:
                state = self.advance_phase(state)
        else:
            # Pass to other player
            state.switch_priority()

        return state

    def execute_opponent_priority(self, state: GameState) -> GameState:
        """Execute opponent's priority actions (heuristic).

        This simulates the opponent responding with instants when appropriate.

        Args:
            state: Current game state.

        Returns:
            Updated game state.

        """
        if state.priority_player != 1:
            return state

        opponent = state.players[1]
        attacker_state = state.get_active_player()

        # Defend during player's attack step
        if (
            state.active_player == 0
            and state.phase == GamePhase.COMBAT_ATTACKERS
            and not state.stack
        ):
            active = state.get_active_player()
            attackers = [active.get_card_by_id(cid) for cid in active.declared_attackers]
            attackers = [c for c in attackers if c]
            blockers = [c for c in opponent.battlefield if opponent.can_block_with(c)]
            if attackers and blockers:
                blocks: dict[int, int] = {}
                for blocker in blockers:
                    legal_attackers = [
                        a
                        for a in attackers
                        if Keyword.FLYING not in a.keywords
                        or Keyword.FLYING in blocker.keywords
                        or Keyword.REACH in blocker.keywords
                    ]
                    if not legal_attackers:
                        continue
                    # Prefer favorable trades (use effective P/T)
                    b_pow = blocker.effective_power
                    b_tgh = blocker.effective_toughness
                    best = max(
                        legal_attackers,
                        key=lambda a: (
                            b_pow >= a.effective_toughness,
                            b_tgh > a.effective_power,
                            a.effective_power,
                        ),
                    )
                    blocks[blocker.card_id] = best.card_id
                if blocks:
                    return self.declare_blockers(state, blocks)

        # Opponent main phase: cast noncreature spells before combat to trigger prowess-like buffs
        if state.active_player == 1 and state.phase == GamePhase.MAIN_PRECOMBAT and not state.stack:
            attackers = [c for c in attacker_state.battlefield if attacker_state.can_attack_with(c)]
            has_noncreature_trigger = any(
                trigger.trigger_type == TriggerType.CAST_NONCREATURE
                for c in attackers
                for trigger in c.triggers
            )
            if has_noncreature_trigger:
                burn_spells = [
                    c
                    for c in opponent.hand
                    if self.can_cast_spell(state, c, player_idx=1)
                    and c.card_type == CardType.INSTANT
                    and c.deals_damage > 0
                    and c.can_target_any
                ]
                if burn_spells:
                    return self.cast_spell(state, burn_spells[0])

        # Simple heuristic: cast instants if advantageous
        for card in opponent.hand:
            if (
                card.card_type == CardType.INSTANT or Keyword.FLASH in card.keywords
            ) and self.can_cast_spell(state, card, player_idx=1):
                if self._is_creature_buff_instant(card):
                    if self._should_cast_creature_buff_instant(state, player_idx=1):
                        return self.cast_spell(state, card)
                    continue
                if card.deals_damage > 0 and state.players[0].life <= 5:
                    return self.cast_spell(state, card)

        # Declare attackers during opponent's combat if not done yet
        if (
            state.active_player == 1
            and not state.stack
            and state.phase
            in {
                GamePhase.COMBAT_BEGIN,
                GamePhase.COMBAT_ATTACKERS,
            }
        ):
            attacker_state = state.get_active_player()
            if not attacker_state.declared_attackers:
                attackers = []
                for c in attacker_state.battlefield:
                    if not attacker_state.can_attack_with(c):
                        continue
                    power, _ = self._get_effective_power_toughness(c, attacker_state)
                    if power > 0:
                        attackers.append(c.card_id)
                if attackers:
                    state = self.declare_attackers(state, attackers)
                    state = self.pass_priority(state)
                    return state

        # Pass priority
        return self.pass_priority(state)

    def execute_opponent_turn(self, state: GameState) -> GameState:
        """Execute opponent's full turn with priority considerations.

        Args:
            state: Current game state.

        Returns:
            Updated game state after opponent's turn.

        """
        if state.active_player != 1:
            return state

        opponent = state.players[1]

        # Execute phases
        while state.active_player == 1 and not state.game_over:
            if state.phase == GamePhase.UNTAP:
                state = self.advance_phase(state)

            elif state.phase == GamePhase.UPKEEP:
                state = self.pass_priority(state)

            elif state.phase == GamePhase.DRAW:
                state = self.advance_phase(state)

            elif state.phase == GamePhase.MAIN_PRECOMBAT:
                # Play a land first (no priority window for land drops).
                lands = [c for c in opponent.hand if c.card_type == CardType.LAND]
                if lands and opponent.lands_played_this_turn == 0:
                    state = self.play_land(state, lands[0])

                # Cast ONE creature, then return priority to player 0 so
                # they can respond (counter it, remove before triggers, etc.).
                # This implements the standard MTG priority window after a
                # spell is cast. Without it, control's counterspells and
                # instant-speed removal never have a legal target on the
                # opponent's turn.
                castable_creatures = [
                    c
                    for c in opponent.hand
                    if self.can_cast_spell(state, c, player_idx=1)
                    and c.card_type == CardType.CREATURE
                ]
                castable_creatures.sort(
                    key=lambda c: (0 if Keyword.HASTE in c.keywords else 1, c.mana_cost.cmc)
                )
                if castable_creatures:
                    state = self.cast_spell(state, castable_creatures[0])
                    state.priority_player = 0
                    return state

                # Then cast one burn spell targeting opponent (pre-combat).
                burn_spells = [
                    c
                    for c in opponent.hand
                    if self.can_cast_spell(state, c, player_idx=1)
                    and c.card_type == CardType.INSTANT
                    and c.deals_damage > 0
                    and c.can_target_any
                ]
                if burn_spells:
                    state = self.cast_spell(state, burn_spells[0])
                    state.priority_player = 0
                    return state

                state = self.advance_phase(state)

            elif state.phase == GamePhase.COMBAT_BEGIN:
                # Cast pump spells AND burn spells before declaring attackers
                # This triggers Prowess on creatures before they attack
                combat_spells = [
                    c
                    for c in opponent.hand
                    if self.can_cast_spell(state, c, player_idx=1)
                    and c.card_type == CardType.INSTANT
                ]
                for card in combat_spells:
                    if self._is_creature_buff_instant(
                        card
                    ) and not self._should_cast_creature_buff_instant(state, player_idx=1):
                        continue
                    if self.can_cast_spell(state, card, player_idx=1):
                        state = self.cast_spell(state, card)
                        state.priority_player = 0
                        return state

                # Declare attackers (creatures now have Prowess bonuses)
                attackers = []
                for c in opponent.battlefield:
                    if not opponent.can_attack_with(c):
                        continue
                    power, _ = self._get_effective_power_toughness(c, opponent)
                    if power > 0:
                        attackers.append(c.card_id)
                if attackers:
                    state = self.declare_attackers(state, attackers)
                    # Give player priority to respond
                    state = self.pass_priority(state)
                    return state
                else:
                    state = self.advance_phase(state)

            elif state.phase == GamePhase.COMBAT_ATTACKERS:
                # If player has priority, let them respond
                if state.priority_player == 0:
                    return state
                # If attackers were not declared yet (e.g., after combat instants), declare now
                if not state.get_active_player().declared_attackers:
                    attackers = []
                    for c in opponent.battlefield:
                        if not opponent.can_attack_with(c):
                            continue
                        power, _ = self._get_effective_power_toughness(c, opponent)
                        if power > 0:
                            attackers.append(c.card_id)
                    if attackers:
                        state = self.declare_attackers(state, attackers)
                        state = self.pass_priority(state)
                        return state
                # Pass priority to player (or advance if both have passed)
                state = self.pass_priority(state)
                return state

            elif state.phase == GamePhase.COMBAT_BLOCKERS or state.phase == GamePhase.COMBAT_DAMAGE:
                state = self.advance_phase(state)

            elif state.phase == GamePhase.MAIN_POSTCOMBAT:
                # Cast remaining spells (non-pump instants, sorceries, etc)
                castable = [
                    c
                    for c in opponent.hand
                    if self.can_cast_spell(state, c, player_idx=1)
                    and not c.is_pump_spell  # Don't cast pump spells post-combat
                ]
                for card in castable:
                    if self.can_cast_spell(state, card, player_idx=1):
                        state = self.cast_spell(state, card)
                        state.priority_player = 0
                        return state

                state = self.advance_phase(state)

            elif state.phase == GamePhase.END_STEP:
                state = self.advance_phase(state)

            else:
                state = self.advance_phase(state)

        return state

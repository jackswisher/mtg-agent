"""Deck archetype definitions for Standard 2025.

This module defines the five deck archetypes used in the benchmark,
representing the competitive Standard 2025 metagame. The system is
extensible to allow custom archetypes.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field
from enum import Enum, auto

from mtg.env.card_definitions import Card, CardRegistry, create_card_instance


class ArchetypeStrategy(Enum):
    """High-level strategic classification of deck archetypes.

    Attributes:
        AGGRO: Fast, damage-focused strategy.
        CONTROL: Reactive, card-advantage strategy.
        MIDRANGE: Balanced, adaptable strategy.
        RAMP: Mana acceleration strategy.
        COMBO: Synergy-focused strategy.

    """

    AGGRO = auto()
    CONTROL = auto()
    MIDRANGE = auto()
    RAMP = auto()
    COMBO = auto()


@dataclass
class DeckArchetype:
    """Represents a deck archetype configuration.

    Attributes:
        name: Unique identifier for the archetype.
        display_name: Human-readable name.
        description: Detailed description of the strategy.
        strategy: High-level strategic classification.
        card_list: List of (card_name, count) tuples.
        meta_share: Approximate metagame share (0-1).
        tier: Competitive tier (1, 2, or 3).
        colors: Color identity of the deck.

    """

    name: str
    display_name: str
    description: str
    strategy: ArchetypeStrategy
    card_list: list[tuple[str, int]]
    meta_share: float = 0.0
    tier: int = 1
    colors: list[str] = field(default_factory=list)

    DECK_SIZE: tp.ClassVar[int] = 60

    def build_deck(self) -> list[Card]:
        """Build a deck from the card list.

        Returns:
            List of Card instances forming the complete deck.

        """
        deck: list[Card] = []
        instance_id = 0

        for card_name, count in self.card_list:
            for _ in range(count):
                deck.append(create_card_instance(card_name, instance_id))
                instance_id += 1

        return deck

    def validate(self) -> tuple[bool, list[str]]:
        """Validate that the deck meets construction requirements.

        Returns:
            Tuple of (is_valid, list of error messages).

        """
        errors: list[str] = []

        total_cards = sum(count for _, count in self.card_list)
        if total_cards != self.DECK_SIZE:
            errors.append(f"Deck has {total_cards} cards, expected {self.DECK_SIZE}")

        card_counts: dict[str, int] = {}
        registry = CardRegistry.get_instance()

        for card_name, count in self.card_list:
            card_counts[card_name] = card_counts.get(card_name, 0) + count

            try:
                card = registry.get(card_name)
                if (
                    card.land_props
                    and not card.land_props.is_basic
                    or not (card.land_props and card.land_props.is_basic)
                ) and card_counts[card_name] > 4:
                    errors.append(f"'{card_name}' has {card_counts[card_name]} copies (max 4)")
            except KeyError:
                errors.append(f"Card '{card_name}' not found in registry")

        return len(errors) == 0, errors

    def get_land_count(self) -> int:
        """Get the number of lands in the deck.

        Returns:
            Total land count.

        """
        registry = CardRegistry.get_instance()
        count = 0
        for card_name, num in self.card_list:
            try:
                card = registry.get(card_name)
                if card.land_props is not None:
                    count += num
            except KeyError:
                pass
        return count

    def get_curve(self) -> dict[int, int]:
        """Get the mana curve distribution.

        Returns:
            Dictionary mapping CMC to card count.

        """
        registry = CardRegistry.get_instance()
        curve: dict[int, int] = {}
        for card_name, count in self.card_list:
            try:
                card = registry.get(card_name)
                if card.land_props is None:
                    cmc = card.mana_cost.cmc
                    curve[cmc] = curve.get(cmc, 0) + count
            except KeyError:
                pass
        return curve


class ArchetypeRegistry:
    """Registry for managing deck archetypes.

    Attributes:
        _archetypes: Dictionary of registered archetypes.

    """

    _instance: tp.ClassVar[ArchetypeRegistry | None] = None
    _archetypes: dict[str, DeckArchetype]

    def __init__(self) -> None:
        """Initialize an empty archetype registry."""
        self._archetypes = {}

    @classmethod
    def get_instance(cls) -> ArchetypeRegistry:
        """Get the singleton registry instance.

        Returns:
            The global ArchetypeRegistry instance.

        """
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._register_default_archetypes()
        return cls._instance

    def register(self, archetype: DeckArchetype) -> None:
        """Register an archetype.

        Args:
            archetype: Archetype to register.

        """
        self._archetypes[archetype.name] = archetype

    def get(self, name: str) -> DeckArchetype:
        """Get an archetype by name.

        Args:
            name: Archetype name.

        Returns:
            The archetype definition.

        Raises:
            KeyError: If archetype not found.

        """
        normalized = name.lower().replace(" ", "_").replace("-", "_")
        if normalized not in self._archetypes:
            available = list(self._archetypes.keys())
            raise KeyError(f"Archetype '{name}' not found. Available: {available}")
        return self._archetypes[normalized]

    def list_all(self) -> list[str]:
        """List all registered archetype names.

        Returns:
            List of archetype names.

        """
        return list(self._archetypes.keys())

    def list_by_strategy(self, strategy: ArchetypeStrategy) -> list[DeckArchetype]:
        """List archetypes by strategy type.

        Args:
            strategy: Strategy to filter by.

        Returns:
            List of matching archetypes.

        """
        return [a for a in self._archetypes.values() if a.strategy == strategy]

    def _register_default_archetypes(self) -> None:
        """Register the five Standard 2025 archetypes."""
        self._register_mono_red_aggro()
        self._register_azorius_control()
        self._register_dimir_midrange()
        self._register_domain_ramp()
        self._register_boros_convoke()

    def _register_mono_red_aggro(self) -> None:
        """Register Mono-Red Aggro archetype."""
        self.register(
            DeckArchetype(
                name="mono_red_aggro",
                display_name="Mono-Red Aggro",
                description=(
                    "A fast aggressive deck aiming to reduce the opponent's life total "
                    "quickly using efficient creatures with haste and direct damage spells."
                ),
                strategy=ArchetypeStrategy.AGGRO,
                meta_share=0.15,
                tier=1,
                colors=["R"],
                card_list=[
                    ("Mountain", 20),
                    ("Monastery Swiftspear", 4),
                    ("Heartfire Hero", 4),
                    ("Slickshot Show-Off", 4),
                    ("Phoenix Chick", 4),
                    ("Play with Fire", 4),
                    ("Lightning Strike", 4),
                    ("Monstrous Rage", 4),
                    ("Shock", 4),
                    ("Kumano Faces Kakkazan", 4),
                    ("Soul-Scar Mage", 4),
                ],
            )
        )

    def _register_azorius_control(self) -> None:
        """Register Azorius Control archetype."""
        self.register(
            DeckArchetype(
                name="azorius_control",
                display_name="Azorius Control",
                description=(
                    "A control deck leveraging counterspells, board wipes, and card "
                    "advantage to dominate the late game with powerful finishers."
                ),
                strategy=ArchetypeStrategy.CONTROL,
                meta_share=0.12,
                tier=1,
                colors=["W", "U"],
                card_list=[
                    ("Plains", 6),
                    ("Island", 6),
                    ("Adarkar Wastes", 4),
                    ("Restless Anchorage", 4),
                    ("Deserted Beach", 4),
                    ("Haughty Djinn", 4),
                    ("No More Lies", 4),
                    ("Make Disappear", 4),
                    ("Memory Deluge", 4),
                    ("Sunfall", 4),
                    ("The Wandering Emperor", 4),
                    ("Get Lost", 4),
                    ("Counterspell", 4),
                    ("Dissolve", 4),
                ],
            )
        )

    def _register_dimir_midrange(self) -> None:
        """Register Dimir Midrange archetype."""
        self.register(
            DeckArchetype(
                name="dimir_midrange",
                display_name="Dimir Midrange",
                description=(
                    "A versatile strategy balancing disruption with efficient threats, "
                    "adapting to opponents while deploying powerful creatures."
                ),
                strategy=ArchetypeStrategy.MIDRANGE,
                meta_share=0.11,
                tier=1,
                colors=["U", "B"],
                card_list=[
                    ("Swamp", 6),
                    ("Island", 5),
                    ("Underground River", 4),
                    ("Shipwreck Marsh", 4),
                    ("Restless Reef", 4),
                    ("Preacher of the Schism", 4),
                    ("Sheoldred, the Apocalypse", 4),
                    ("Faerie Mastermind", 4),
                    ("Deep-Cavern Bat", 4),
                    ("Go for the Throat", 4),
                    ("Cut Down", 4),
                    ("Duress", 4),
                    ("Make Disappear", 4),
                    ("Memory Deluge", 1),
                    ("Dissolve", 4),
                ],
            )
        )

    def _register_domain_ramp(self) -> None:
        """Register Domain Ramp archetype."""
        self.register(
            DeckArchetype(
                name="domain_ramp",
                display_name="Domain Ramp",
                description=(
                    "A multicolor deck accelerating mana to deploy powerful domain "
                    "payoffs like Atraxa and Leyline Binding."
                ),
                strategy=ArchetypeStrategy.RAMP,
                meta_share=0.08,
                tier=2,
                colors=["W", "U", "B", "R", "G"],
                card_list=[
                    ("Forest", 8),
                    ("Plains", 4),
                    ("Island", 4),
                    ("Swamp", 2),
                    ("Mountain", 2),
                    ("Adarkar Wastes", 2),
                    ("Underground River", 2),
                    ("Atraxa, Grand Unifier", 4),
                    ("Topiary Stomper", 4),
                    ("Leyline Binding", 4),
                    ("Up the Beanstalk", 4),
                    ("Sunfall", 4),
                    ("Llanowar Elves", 4),
                    ("Memory Deluge", 4),
                    ("Go for the Throat", 4),
                    ("Detention Sphere", 4),
                ],
            )
        )

    def _register_boros_convoke(self) -> None:
        """Register Boros Convoke archetype."""
        self.register(
            DeckArchetype(
                name="boros_convoke",
                display_name="Boros Convoke",
                description=(
                    "An aggressive token strategy exploiting the convoke mechanic "
                    "to deploy threats faster than the mana curve suggests."
                ),
                strategy=ArchetypeStrategy.AGGRO,
                meta_share=0.07,
                tier=2,
                colors=["R", "W"],
                card_list=[
                    ("Plains", 10),
                    ("Mountain", 6),
                    ("Battlefield Forge", 4),
                    ("Inspiring Vantage", 4),
                    ("Warden of the Inner Sky", 4),
                    ("Resolute Reinforcements", 4),
                    ("Knight-Errant of Eos", 4),
                    ("Monastery Swiftspear", 4),
                    ("Phoenix Chick", 4),
                    ("Play with Fire", 4),
                    ("Lightning Strike", 4),
                    ("Monstrous Rage", 4),
                    ("Heartfire Hero", 4),
                ],
            )
        )


def get_archetype(name: str) -> DeckArchetype:
    """Get a deck archetype by name.

    Args:
        name: Archetype name (e.g., 'mono_red_aggro', 'aggro').

    Returns:
        The corresponding DeckArchetype.

    Raises:
        KeyError: If archetype name is not recognized.

    """
    registry = ArchetypeRegistry.get_instance()

    aliases: dict[str, str] = {
        "aggro": "mono_red_aggro",
        "control": "azorius_control",
        "midrange": "dimir_midrange",
        "ramp": "domain_ramp",
        "convoke": "boros_convoke",
    }

    normalized = name.lower().replace(" ", "_").replace("-", "_")
    if normalized in aliases:
        normalized = aliases[normalized]

    return registry.get(normalized)


def list_archetypes() -> list[str]:
    """List all available archetype names.

    Returns:
        List of archetype names.

    """
    return ArchetypeRegistry.get_instance().list_all()


def register_custom_archetype(archetype: DeckArchetype) -> None:
    """Register a custom archetype.

    Args:
        archetype: Custom archetype to register.

    """
    ArchetypeRegistry.get_instance().register(archetype)

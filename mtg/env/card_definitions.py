"""Card definitions and registry for the MTG environment.

This module provides an extensible card system with a registry pattern
for defining cards, lands (with tier abstractions), and abilities.
All cards are Standard 2025 legal.
"""

from __future__ import annotations

import copy
import typing as tp
from dataclasses import dataclass, field
from enum import Enum, auto


class CardType(Enum):
    """Enumeration of MTG card types.

    Attributes:
        LAND: Mana-producing permanent.
        CREATURE: Permanent that can attack and block.
        INSTANT: Spell castable at any time.
        SORCERY: Spell castable only on your turn.
        ENCHANTMENT: Persistent spell effect.
        ARTIFACT: Colorless permanent.
        PLANESWALKER: Loyalty-based permanent.

    """

    LAND = "land"
    CREATURE = "creature"
    INSTANT = "instant"
    SORCERY = "sorcery"
    ENCHANTMENT = "enchantment"
    ARTIFACT = "artifact"
    PLANESWALKER = "planeswalker"


class ManaColor(Enum):
    """Enumeration of mana colors in MTG.

    Attributes:
        WHITE: Plains mana (W).
        BLUE: Island mana (U).
        BLACK: Swamp mana (B).
        RED: Mountain mana (R).
        GREEN: Forest mana (G).
        COLORLESS: Generic colorless mana (C).

    """

    WHITE = "W"
    BLUE = "U"
    BLACK = "B"
    RED = "R"
    GREEN = "G"
    COLORLESS = "C"


class LandTier(Enum):
    """Land tier classification for strategic abstraction.

    Attributes:
        BASIC: Basic lands (always untapped, single color).
        FAST_DUAL: Dual lands with conditional untapped entry.
        SLOW_DUAL: Dual lands that always enter tapped.
        UTILITY: Lands with activated abilities.
        PAIN: Dual lands that deal damage for colored mana.

    """

    BASIC = auto()
    FAST_DUAL = auto()
    SLOW_DUAL = auto()
    UTILITY = auto()
    PAIN = auto()


class Keyword(Enum):
    """Creature keyword abilities.

    Attributes:
        HASTE: Can attack immediately.
        FLYING: Can only be blocked by flyers/reach.
        VIGILANCE: Does not tap to attack.
        LIFELINK: Damage dealt gains life.
        DEATHTOUCH: Any damage is lethal.
        PROWESS: Gets +1/+1 when casting noncreature spells.
        FLASH: Can be cast at instant speed.
        TRAMPLE: Excess damage goes to player.
        FIRST_STRIKE: Deals damage before normal combat.
        DOUBLE_STRIKE: Deals first strike and normal damage.
        REACH: Can block creatures with flying.
        MENACE: Can only be blocked by 2+ creatures.
        HEXPROOF: Cannot be targeted by opponents.
        WARD: Counter unless opponent pays (value stored separately).
        INDESTRUCTIBLE: Cannot be destroyed by damage or destroy effects.
        PROTECTION_WHITE: Protection from white.
        PROTECTION_BLUE: Protection from blue.
        PROTECTION_BLACK: Protection from black.
        PROTECTION_RED: Protection from red.
        PROTECTION_GREEN: Protection from green.
        DEFENDER: Cannot attack.
        SKULK: Can't be blocked by creatures with greater power.
        UNDYING: Returns with +1/+1 counter when dies without one.
        PERSIST: Returns with -1/-1 counter when dies without one.

    """

    HASTE = auto()
    FLYING = auto()
    VIGILANCE = auto()
    LIFELINK = auto()
    DEATHTOUCH = auto()
    PROWESS = auto()
    FLASH = auto()
    TRAMPLE = auto()
    FIRST_STRIKE = auto()
    DOUBLE_STRIKE = auto()
    REACH = auto()
    MENACE = auto()
    HEXPROOF = auto()
    WARD = auto()
    INDESTRUCTIBLE = auto()
    PROTECTION_WHITE = auto()
    PROTECTION_BLUE = auto()
    PROTECTION_BLACK = auto()
    PROTECTION_RED = auto()
    PROTECTION_GREEN = auto()
    DEFENDER = auto()
    SKULK = auto()
    UNDYING = auto()
    PERSIST = auto()


class TriggerType(Enum):
    """Types of triggered abilities in MTG.

    These define WHEN a trigger fires.
    """

    # Spell casting triggers
    CAST_NONCREATURE = auto()  # Prowess, "whenever you cast a noncreature spell"
    CAST_CREATURE = auto()  # "whenever you cast a creature spell"
    CAST_ANY = auto()  # "whenever you cast a spell"

    # Permanent entering battlefield
    ETB_SELF = auto()  # When THIS permanent enters
    ETB_CREATURE = auto()  # When a creature enters (yours)
    ETB_LAND = auto()  # Landfall triggers

    # Death triggers
    DIES_SELF = auto()  # When THIS creature dies
    DIES_CREATURE = auto()  # When any creature dies

    # Combat triggers
    ATTACK_SELF = auto()  # When THIS creature attacks
    ATTACK_ANY = auto()  # When any creature attacks
    DEAL_DAMAGE = auto()  # When THIS deals combat damage

    # Phase triggers
    UPKEEP = auto()  # At beginning of upkeep
    END_STEP = auto()  # At beginning of end step
    DRAW_STEP = auto()  # At beginning of draw step

    # Saga triggers
    CHAPTER_I = auto()  # Saga Chapter I
    CHAPTER_II = auto()  # Saga Chapter II
    CHAPTER_III = auto()  # Saga Chapter III

    # Activated ability (not really a trigger, but for completeness)
    ACTIVATED = auto()


class TriggerEffect(Enum):
    """Types of effects that triggers can produce.

    These define WHAT happens when triggered.
    """

    # Stat modifications
    POWER_TOUGHNESS_BOOST = auto()  # +X/+X until end of turn
    POWER_TOUGHNESS_COUNTER = auto()  # +1/+1 counter
    GRANT_KEYWORD = auto()  # Grants a keyword ability

    # Damage and life
    DEAL_DAMAGE = auto()  # Deal X damage
    GAIN_LIFE = auto()  # Gain X life
    LOSE_LIFE = auto()  # Target loses X life

    # Cards
    DRAW_CARDS = auto()  # Draw X cards
    DISCARD_CARDS = auto()  # Discard X cards
    SCRY = auto()  # Scry X

    # Permanent manipulation
    CREATE_TOKEN = auto()  # Create a token
    DESTROY_PERMANENT = auto()  # Destroy a permanent
    EXILE_PERMANENT = auto()  # Exile a permanent
    RETURN_TO_HAND = auto()  # Return to hand
    RETURN_TO_BATTLEFIELD = auto()  # Return from graveyard to battlefield
    TAP_PERMANENT = auto()  # Tap a permanent

    # Special
    TRANSFORM = auto()  # Transform/flip the card
    COUNTER_SPELL = auto()  # Counter a spell
    CANT_ATTACK = auto()  # Target can't attack
    CANT_BLOCK = auto()  # Target can't block


@dataclass
class Trigger:
    """Represents a triggered ability on a card.

    Attributes:
        trigger_type: When this trigger fires.
        effect: What effect this trigger produces.
        effect_value: Numeric value for the effect (damage amount, +X/+X, etc).
        target: Who/what is affected ("self", "controller", "opponent", "all_creatures").
        keyword_granted: If GRANT_KEYWORD, which keyword.
        token_type: If CREATE_TOKEN, what kind of token.
        condition: Optional additional condition text.
        description: Human-readable description of the trigger.

    """

    trigger_type: TriggerType
    effect: TriggerEffect
    effect_value: int = 0
    toughness_value: int | None = None  # For P/T boosts; None = same as effect_value
    target: str = "self"  # "self", "controller", "opponent", "all_your_creatures"
    keyword_granted: Keyword | None = None
    token_type: str = ""  # e.g., "1/1 white Spirit"
    condition: str = ""  # Additional condition
    description: str = ""  # Human-readable description


# Pre-defined common triggers for convenience
def prowess_trigger() -> Trigger:
    """Create a Prowess trigger."""
    return Trigger(
        trigger_type=TriggerType.CAST_NONCREATURE,
        effect=TriggerEffect.POWER_TOUGHNESS_BOOST,
        effect_value=1,  # +1/+1
        target="self",
        description="Prowess (Whenever you cast a noncreature spell, this creature gets +1/+1 until end of turn.)",
    )


def etb_damage_trigger(damage: int, target: str = "any") -> Trigger:
    """Create an ETB damage trigger."""
    return Trigger(
        trigger_type=TriggerType.ETB_SELF,
        effect=TriggerEffect.DEAL_DAMAGE,
        effect_value=damage,
        target=target,
        description=f"When this enters the battlefield, deal {damage} damage to any target.",
    )


def death_trigger_draw(cards: int) -> Trigger:
    """Create a death trigger that draws cards."""
    return Trigger(
        trigger_type=TriggerType.DIES_SELF,
        effect=TriggerEffect.DRAW_CARDS,
        effect_value=cards,
        target="controller",
        description=f"When this dies, draw {cards} card{'s' if cards > 1 else ''}.",
    )


def attack_trigger_damage(damage: int) -> Trigger:
    """Create an attack trigger that deals damage."""
    return Trigger(
        trigger_type=TriggerType.ATTACK_SELF,
        effect=TriggerEffect.DEAL_DAMAGE,
        effect_value=damage,
        target="opponent",
        description=f"Whenever this attacks, deal {damage} damage to defending player.",
    )


def landfall_trigger(effect: TriggerEffect, value: int) -> Trigger:
    """Create a landfall trigger."""
    return Trigger(
        trigger_type=TriggerType.ETB_LAND,
        effect=effect,
        effect_value=value,
        target="self" if effect == TriggerEffect.POWER_TOUGHNESS_BOOST else "controller",
        description=f"Landfall — Whenever a land enters the battlefield under your control, this gets +{value}/+{value}.",
    )


@dataclass
class ManaCost:
    """Represents a mana cost for casting spells.

    Attributes:
        generic: Amount of generic (any color) mana required.
        white: Amount of white mana required.
        blue: Amount of blue mana required.
        black: Amount of black mana required.
        red: Amount of red mana required.
        green: Amount of green mana required.

    """

    generic: int = 0
    white: int = 0
    blue: int = 0
    black: int = 0
    red: int = 0
    green: int = 0

    @property
    def cmc(self) -> int:
        """Calculate converted mana cost (total mana value).

        Returns:
            Total mana required to cast this spell.

        """
        return self.generic + self.white + self.blue + self.black + self.red + self.green

    @property
    def colors(self) -> list[ManaColor]:
        """Get the colors present in this mana cost.

        Returns:
            List of colors required by this cost.

        """
        result: list[ManaColor] = []
        if self.white > 0:
            result.append(ManaColor.WHITE)
        if self.blue > 0:
            result.append(ManaColor.BLUE)
        if self.black > 0:
            result.append(ManaColor.BLACK)
        if self.red > 0:
            result.append(ManaColor.RED)
        if self.green > 0:
            result.append(ManaColor.GREEN)
        return result

    def can_pay(self, available_mana: dict[str, int]) -> bool:
        """Check if this cost can be paid with available mana.

        Args:
            available_mana: Dictionary mapping color codes to amounts.

        Returns:
            True if the cost can be paid.

        """
        if available_mana.get("W", 0) < self.white:
            return False
        if available_mana.get("U", 0) < self.blue:
            return False
        if available_mana.get("B", 0) < self.black:
            return False
        if available_mana.get("R", 0) < self.red:
            return False
        if available_mana.get("G", 0) < self.green:
            return False

        total_colored_required = self.white + self.blue + self.black + self.red + self.green
        total_available = sum(available_mana.values())
        return total_available >= total_colored_required + self.generic

    def add_generic(self, amount: int) -> ManaCost:
        """Return a copy of this cost with generic increased by amount."""
        return ManaCost(
            generic=self.generic + max(0, amount),
            white=self.white,
            blue=self.blue,
            black=self.black,
            red=self.red,
            green=self.green,
        )

    def reduce_generic(self, amount: int) -> ManaCost:
        """Return a copy of this cost with generic reduced by amount."""
        reduced = max(0, self.generic - max(0, amount))
        return ManaCost(
            generic=reduced,
            white=self.white,
            blue=self.blue,
            black=self.black,
            red=self.red,
            green=self.green,
        )

    @classmethod
    def from_string(cls, cost_str: str) -> ManaCost:
        """Parse a mana cost from string notation.

        Args:
            cost_str: Mana cost string (e.g., "2RR", "1UB", "G").

        Returns:
            Parsed ManaCost object.

        """
        cost = cls()
        i = 0
        while i < len(cost_str):
            char = cost_str[i]
            if char.isdigit():
                num_str = char
                while i + 1 < len(cost_str) and cost_str[i + 1].isdigit():
                    i += 1
                    num_str += cost_str[i]
                cost.generic = int(num_str)
            elif char == "W":
                cost.white += 1
            elif char == "U":
                cost.blue += 1
            elif char == "B":
                cost.black += 1
            elif char == "R":
                cost.red += 1
            elif char == "G":
                cost.green += 1
            i += 1
        return cost

    def to_text(self) -> str:
        """Convert mana cost to text representation.

        Returns:
            Text representation of the mana cost (e.g., "1RR", "2UB", "G").

        """
        parts = []
        if self.generic > 0:
            parts.append(str(self.generic))
        parts.append("W" * self.white)
        parts.append("U" * self.blue)
        parts.append("B" * self.black)
        parts.append("R" * self.red)
        parts.append("G" * self.green)
        return "".join(parts)


@dataclass
class LandProperties:
    """Properties specific to land cards.

    Attributes:
        produces: Colors of mana this land can produce.
        tier: Land tier for strategic classification.
        enters_tapped: Whether land enters tapped.
        enters_tapped_conditional: Turn threshold for conditional entry.
        life_cost: Life paid when tapping for colored mana.
        is_basic: Whether this is a basic land.
        basic_land_types: Basic land types this land has.
        has_activation: Whether land has an activated ability.
        activation_cost: Mana cost to activate ability.
        activation_power: Power when activated as creature.
        activation_toughness: Toughness when activated as creature.

    """

    produces: list[ManaColor] = field(default_factory=list)
    tier: LandTier = LandTier.BASIC
    enters_tapped: bool = False
    enters_tapped_conditional: int = 0
    life_cost: int = 0
    is_basic: bool = False
    basic_land_types: list[str] = field(default_factory=list)
    has_activation: bool = False
    activation_cost: ManaCost = field(default_factory=ManaCost)
    activation_power: int = 0
    activation_toughness: int = 0


@dataclass
class Card:
    """Represents an MTG card with all relevant properties.

    Attributes:
        name: Card name.
        card_type: Primary card type.
        mana_cost: Cost to cast (not applicable for lands).
        power: Creature power (0 for non-creatures).
        toughness: Creature toughness (0 for non-creatures).
        keywords: Set of keyword abilities.
        land_props: Land-specific properties.
        triggers: List of triggered abilities on this card.
        draws_cards: Number of cards drawn when resolved.
        deals_damage: Damage dealt when resolved.
        gains_life: Life gained when resolved.
        is_removal: Whether this can destroy creatures.
        is_counterspell: Whether this can counter spells.
        requires_creature_target: Whether spell requires a creature target.
        rules_text: Card's rules/oracle text for display.
        set_code: Set this card is from.
        card_id: Unique instance identifier.

    """

    name: str
    card_type: CardType
    mana_cost: ManaCost = field(default_factory=ManaCost)
    power: int = 0
    toughness: int = 0
    keywords: set[Keyword] = field(default_factory=set)
    land_props: LandProperties | None = None
    triggers: list[Trigger] = field(default_factory=list)
    draws_cards: int = 0
    deals_damage: int = 0
    gains_life: int = 0
    is_removal: bool = False
    is_counterspell: bool = False
    requires_creature_target: bool = False  # Spell ONLY targets creatures
    can_target_any: bool = False  # Spell can target creature or player
    can_target_nonland_permanent: bool = False  # Spell can target any nonland permanent
    is_pump_spell: bool = False  # Combat trick that pumps a creature
    target_restriction: str = ""  # e.g. "nonartifact", "pt_lte_5"
    rules_text: str = ""
    set_code: str = "FDN"
    card_id: int = 0
    # Current power/toughness (modified by effects like Prowess, pump spells)
    current_power: int | None = None
    current_toughness: int | None = None
    # Attached tokens/auras (e.g., Monster Role from Monstrous Rage)
    attached_tokens: list[str] = field(default_factory=list)
    # Permanent power/toughness bonuses from tokens (not cleared at end of turn)
    permanent_power_bonus: int = 0
    permanent_toughness_bonus: int = 0
    # +1/+1 and -1/-1 counters
    plus_counters: int = 0
    minus_counters: int = 0
    # Ward cost (mana or life amount)
    ward_cost: int = 0
    # Protection colors (for specific protection, not keyword-based)
    protection_colors: set[str] = field(default_factory=set)
    # Loyalty for planeswalkers
    loyalty: int = 0
    starting_loyalty: int = 0
    # Adventure properties
    adventure_name: str = ""
    adventure_cost: ManaCost | None = None
    adventure_effect: str = ""
    is_on_adventure: bool = False  # True if cast as adventure (exiled)
    # Flashback cost
    flashback_cost: ManaCost | None = None
    has_flashback: bool = False
    # Kicker cost
    kicker_cost: ManaCost | None = None
    was_kicked: bool = False
    # Transform/MDFC properties
    back_face: Card | None = None
    is_transformed: bool = False
    # Vehicle properties
    crew_cost: int = 0  # Power needed to crew
    is_vehicle: bool = False
    is_crewed: bool = False  # Currently a creature this turn
    # Domain count (cached, recalculated on land play)
    domain_count: int = 0
    # Saga properties
    saga_chapter: int = 0
    max_chapters: int = 0
    # Owner player index (for returning exiled cards)
    owner_player_idx: int = 0
    # Exiled by this card (for effects like Leyline Binding)
    exiled_cards: list[Card] = field(default_factory=list)
    # Attached to (for auras/equipment)
    attached_to: Card | None = None
    # Attachments on this creature (auras/equipment)
    attachments: list[Card] = field(default_factory=list)
    # Token indicator
    is_token: bool = False
    token_name: str = ""
    # Convoke - can tap creatures to pay generic mana
    has_convoke: bool = False
    # Domain cost reduction (e.g., Leyline Binding)
    has_domain_cost_reduction: bool = False
    # Improvise - can tap artifacts to pay generic mana
    has_improvise: bool = False
    # Affinity type (e.g., "artifact" for affinity for artifacts)
    affinity_type: str = ""

    def __hash__(self) -> int:
        """Generate hash based on name and instance ID.

        Returns:
            Hash value for this card.

        """
        return hash((self.name, self.card_id))

    def __eq__(self, other: object) -> bool:
        """Check equality with another card.

        Args:
            other: Object to compare.

        Returns:
            True if cards are equal.

        """
        if not isinstance(other, Card):
            return NotImplemented
        return self.name == other.name and self.card_id == other.card_id

    @property
    def has_haste(self) -> bool:
        """Check if creature has haste.

        Returns:
            True if creature has haste keyword.

        """
        return Keyword.HASTE in self.keywords

    @property
    def has_flash(self) -> bool:
        """Check if card has flash.

        Returns:
            True if card can be cast at instant speed.

        """
        return Keyword.FLASH in self.keywords

    @property
    def has_flying(self) -> bool:
        """Check if creature has flying."""
        return Keyword.FLYING in self.keywords

    @property
    def has_vigilance(self) -> bool:
        """Check if creature has vigilance."""
        return Keyword.VIGILANCE in self.keywords

    @property
    def has_lifelink(self) -> bool:
        """Check if creature has lifelink."""
        return Keyword.LIFELINK in self.keywords

    @property
    def has_deathtouch(self) -> bool:
        """Check if creature has deathtouch."""
        return Keyword.DEATHTOUCH in self.keywords

    @property
    def has_trample(self) -> bool:
        """Check if creature has trample."""
        return Keyword.TRAMPLE in self.keywords

    @property
    def has_first_strike(self) -> bool:
        """Check if creature has first strike."""
        return Keyword.FIRST_STRIKE in self.keywords or Keyword.DOUBLE_STRIKE in self.keywords

    @property
    def has_double_strike(self) -> bool:
        """Check if creature has double strike."""
        return Keyword.DOUBLE_STRIKE in self.keywords

    @property
    def has_reach(self) -> bool:
        """Check if creature can block flyers."""
        return Keyword.REACH in self.keywords

    @property
    def has_menace(self) -> bool:
        """Check if creature requires 2+ blockers."""
        return Keyword.MENACE in self.keywords

    @property
    def has_hexproof(self) -> bool:
        """Check if creature has hexproof."""
        return Keyword.HEXPROOF in self.keywords

    @property
    def has_ward(self) -> bool:
        """Check if creature has ward."""
        return Keyword.WARD in self.keywords

    @property
    def has_indestructible(self) -> bool:
        """Check if creature has indestructible."""
        return Keyword.INDESTRUCTIBLE in self.keywords

    @property
    def has_defender(self) -> bool:
        """Check if creature cannot attack."""
        return Keyword.DEFENDER in self.keywords

    @property
    def effective_power(self) -> int:
        """Get effective power including counters and bonuses."""
        base = self.current_power if self.current_power is not None else self.power
        return base + self.plus_counters - self.minus_counters + self.permanent_power_bonus

    @property
    def effective_toughness(self) -> int:
        """Get effective toughness including counters and bonuses."""
        base = self.current_toughness if self.current_toughness is not None else self.toughness
        return base + self.plus_counters - self.minus_counters + self.permanent_toughness_bonus

    def has_protection_from(self, color: str) -> bool:
        """Check if creature has protection from a color.

        Args:
            color: Color code (W, U, B, R, G).

        Returns:
            True if creature has protection from that color.
        """
        if color in self.protection_colors:
            return True
        keyword_map = {
            "W": Keyword.PROTECTION_WHITE,
            "U": Keyword.PROTECTION_BLUE,
            "B": Keyword.PROTECTION_BLACK,
            "R": Keyword.PROTECTION_RED,
            "G": Keyword.PROTECTION_GREEN,
        }
        return keyword_map.get(color) in self.keywords

    def can_be_targeted_by(self, source_player_idx: int, owner_idx: int) -> bool:
        """Check if this permanent can be targeted by a player.

        Args:
            source_player_idx: Index of player trying to target.
            owner_idx: Index of player who owns this permanent.

        Returns:
            True if can be targeted.
        """
        # Hexproof prevents targeting by opponents
        return not (self.has_hexproof and source_player_idx != owner_idx)

    def add_plus_counter(self, amount: int = 1) -> None:
        """Add +1/+1 counters to this creature."""
        self.plus_counters += amount

    def add_minus_counter(self, amount: int = 1) -> None:
        """Add -1/-1 counters to this creature."""
        self.minus_counters += amount
        # +1/+1 and -1/-1 counters cancel out
        cancelled = min(self.plus_counters, self.minus_counters)
        self.plus_counters -= cancelled
        self.minus_counters -= cancelled

    @property
    def produces_mana(self) -> list[ManaColor]:
        """Get mana colors this card can produce.

        Returns:
            List of mana colors, empty if not a mana source.

        """
        if self.land_props:
            return self.land_props.produces
        return []

    @property
    def enters_tapped(self) -> bool:
        """Check if this land enters tapped.

        Returns:
            True if land enters tapped.

        """
        if self.land_props:
            return self.land_props.enters_tapped
        return False

    @property
    def is_threat(self) -> bool:
        """Check if this is a threat (creature with power >= 2).

        Returns:
            True if this is considered a threat.

        """
        return self.card_type == CardType.CREATURE and self.power >= 2

    def get_triggers_by_type(self, trigger_type: TriggerType) -> list[Trigger]:
        """Get all triggers of a specific type.

        Args:
            trigger_type: The type of trigger to find.

        Returns:
            List of matching triggers.

        """
        return [t for t in self.triggers if t.trigger_type == trigger_type]

    def has_trigger(self, trigger_type: TriggerType) -> bool:
        """Check if card has a trigger of specified type.

        Args:
            trigger_type: The type of trigger to check for.

        Returns:
            True if card has at least one trigger of this type.

        """
        return any(t.trigger_type == trigger_type for t in self.triggers)

    def to_feature_vector(self) -> list[float]:
        """Convert card to a feature vector for observation space.

        Returns:
            Flat feature vector representing this card.

        """
        features: list[float] = []

        card_type_onehot = [0.0] * len(CardType)
        card_type_onehot[list(CardType).index(self.card_type)] = 1.0
        features.extend(card_type_onehot)

        features.append(float(self.mana_cost.cmc))
        features.append(float(self.power))
        features.append(float(self.toughness))

        color_vec = [0.0] * 5
        for color in self.mana_cost.colors:
            if color != ManaColor.COLORLESS:
                idx = [
                    ManaColor.WHITE,
                    ManaColor.BLUE,
                    ManaColor.BLACK,
                    ManaColor.RED,
                    ManaColor.GREEN,
                ].index(color)
                color_vec[idx] = 1.0
        features.extend(color_vec)

        keyword_vec = [0.0] * len(Keyword)
        for kw in self.keywords:
            keyword_vec[list(Keyword).index(kw)] = 1.0
        features.extend(keyword_vec)

        if self.land_props:
            produces_vec = [0.0] * 5
            for color in self.land_props.produces:
                if color != ManaColor.COLORLESS:
                    idx = [
                        ManaColor.WHITE,
                        ManaColor.BLUE,
                        ManaColor.BLACK,
                        ManaColor.RED,
                        ManaColor.GREEN,
                    ].index(color)
                    produces_vec[idx] = 1.0
            features.extend(produces_vec)
            features.append(float(self.land_props.enters_tapped))
            features.append(float(self.land_props.life_cost))
            features.append(float(self.land_props.is_basic))
            features.append(float(self.land_props.has_activation))
        else:
            features.extend([0.0] * 9)

        features.append(float(self.draws_cards))
        features.append(float(self.deals_damage))
        features.append(float(self.is_removal))
        features.append(float(self.is_counterspell))

        return features

    @staticmethod
    def get_feature_dim() -> int:
        """Get the dimension of the card feature vector.

        Returns:
            Number of features in card encoding.

        """
        return len(CardType) + 3 + 5 + len(Keyword) + 9 + 4


class CardRegistry:
    """Registry for managing and accessing card definitions.

    Attributes:
        _cards: Dictionary mapping card names to definitions.

    """

    _instance: tp.ClassVar[CardRegistry | None] = None
    _cards: dict[str, Card]

    def __init__(self) -> None:
        """Initialize an empty card registry."""
        self._cards = {}

    @classmethod
    def get_instance(cls) -> CardRegistry:
        """Get the singleton registry instance.

        Returns:
            The global CardRegistry instance.

        """
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._register_default_cards()
        return cls._instance

    def register(self, card: Card) -> None:
        """Register a card in the registry.

        Args:
            card: Card to register.

        """
        self._cards[card.name] = card

    def get(self, name: str) -> Card:
        """Get a card by name.

        Args:
            name: Card name to look up.

        Returns:
            The card definition.

        Raises:
            KeyError: If card is not found.

        """
        if name not in self._cards:
            available = list(self._cards.keys())[:10]
            raise KeyError(f"Card '{name}' not found. Available: {available}...")
        return self._cards[name]

    def get_all(self) -> dict[str, Card]:
        """Get all registered cards.

        Returns:
            Dictionary of all card names to definitions.

        """
        return self._cards.copy()

    def list_by_type(self, card_type: CardType) -> list[Card]:
        """List all cards of a given type.

        Args:
            card_type: Type to filter by.

        Returns:
            List of matching cards.

        """
        return [c for c in self._cards.values() if c.card_type == card_type]

    def list_by_color(self, color: ManaColor) -> list[Card]:
        """List all cards containing a color.

        Args:
            color: Color to filter by.

        Returns:
            List of matching cards.

        """
        return [c for c in self._cards.values() if color in c.mana_cost.colors]

    def create_instance(self, name: str, instance_id: int) -> Card:
        """Create a unique instance of a card.

        Args:
            name: Card name to instantiate.
            instance_id: Unique identifier for this instance.

        Returns:
            New card instance with unique ID.

        """
        template = self.get(name)
        return Card(
            name=template.name,
            card_type=template.card_type,
            mana_cost=template.mana_cost,
            power=template.power,
            toughness=template.toughness,
            keywords=template.keywords.copy(),
            land_props=copy.deepcopy(template.land_props) if template.land_props else None,
            triggers=list(template.triggers),
            draws_cards=template.draws_cards,
            deals_damage=template.deals_damage,
            gains_life=template.gains_life,
            is_removal=template.is_removal,
            is_counterspell=template.is_counterspell,
            requires_creature_target=template.requires_creature_target,
            can_target_any=template.can_target_any,
            can_target_nonland_permanent=template.can_target_nonland_permanent,
            is_pump_spell=template.is_pump_spell,
            target_restriction=template.target_restriction,
            rules_text=template.rules_text,
            set_code=template.set_code,
            card_id=instance_id,
            has_convoke=template.has_convoke,
            has_domain_cost_reduction=template.has_domain_cost_reduction,
            has_improvise=template.has_improvise,
            affinity_type=template.affinity_type,
        )

    def _register_default_cards(self) -> None:
        """Register all default Standard 2025 cards."""
        self._register_basic_lands()
        self._register_dual_lands()
        self._register_red_cards()
        self._register_white_cards()
        self._register_blue_cards()
        self._register_black_cards()
        self._register_green_cards()
        self._register_multicolor_cards()

    def _register_basic_lands(self) -> None:
        """Register all basic lands."""
        basics = [
            ("Plains", ManaColor.WHITE, "W", "({T}: Add {W}.)"),
            ("Island", ManaColor.BLUE, "U", "({T}: Add {U}.)"),
            ("Swamp", ManaColor.BLACK, "B", "({T}: Add {B}.)"),
            ("Mountain", ManaColor.RED, "R", "({T}: Add {R}.)"),
            ("Forest", ManaColor.GREEN, "G", "({T}: Add {G}.)"),
        ]
        for name, color, _, rules in basics:
            self.register(
                Card(
                    name=name,
                    card_type=CardType.LAND,
                    land_props=LandProperties(
                        produces=[color],
                        tier=LandTier.BASIC,
                        is_basic=True,
                        basic_land_types=[name],
                    ),
                    rules_text=rules,
                    set_code="FDN",
                )
            )

    def _register_dual_lands(self) -> None:
        """Register dual and utility lands."""
        self.register(
            Card(
                name="Battlefield Forge",
                card_type=CardType.LAND,
                land_props=LandProperties(
                    produces=[ManaColor.RED, ManaColor.WHITE],
                    tier=LandTier.PAIN,
                    life_cost=1,
                    basic_land_types=["Mountain", "Plains"],
                ),
                rules_text="{T}: Add {C}. {T}: Add {R} or {W}. Battlefield Forge deals 1 damage to you.",
                set_code="BRO",
            )
        )

        self.register(
            Card(
                name="Adarkar Wastes",
                card_type=CardType.LAND,
                land_props=LandProperties(
                    produces=[ManaColor.WHITE, ManaColor.BLUE],
                    tier=LandTier.PAIN,
                    life_cost=1,
                    basic_land_types=["Plains", "Island"],
                ),
                rules_text="{T}: Add {C}. {T}: Add {W} or {U}. Adarkar Wastes deals 1 damage to you.",
                set_code="DMU",
            )
        )

        self.register(
            Card(
                name="Underground River",
                card_type=CardType.LAND,
                land_props=LandProperties(
                    produces=[ManaColor.BLUE, ManaColor.BLACK],
                    tier=LandTier.PAIN,
                    life_cost=1,
                    basic_land_types=["Island", "Swamp"],
                ),
                rules_text="{T}: Add {C}. {T}: Add {U} or {B}. Underground River deals 1 damage to you.",
                set_code="BRO",
            )
        )

        self.register(
            Card(
                name="Deserted Beach",
                card_type=CardType.LAND,
                land_props=LandProperties(
                    produces=[ManaColor.WHITE, ManaColor.BLUE],
                    tier=LandTier.SLOW_DUAL,
                    enters_tapped=True,
                    enters_tapped_conditional=2,
                    basic_land_types=["Plains", "Island"],
                ),
                rules_text="Deserted Beach enters the battlefield tapped unless you control two or more other lands. {T}: Add {W} or {U}.",
                set_code="MID",
            )
        )

        self.register(
            Card(
                name="Shipwreck Marsh",
                card_type=CardType.LAND,
                land_props=LandProperties(
                    produces=[ManaColor.BLUE, ManaColor.BLACK],
                    tier=LandTier.SLOW_DUAL,
                    enters_tapped=True,
                    enters_tapped_conditional=2,
                    basic_land_types=["Island", "Swamp"],
                ),
                rules_text="Shipwreck Marsh enters the battlefield tapped unless you control two or more other lands. {T}: Add {U} or {B}.",
                set_code="MID",
            )
        )

        self.register(
            Card(
                name="Restless Anchorage",
                card_type=CardType.LAND,
                land_props=LandProperties(
                    produces=[ManaColor.WHITE, ManaColor.BLUE],
                    tier=LandTier.UTILITY,
                    enters_tapped=True,
                    has_activation=True,
                    activation_cost=ManaCost.from_string("1WU"),
                    activation_power=2,
                    activation_toughness=3,
                    basic_land_types=["Plains", "Island"],
                ),
                rules_text="Enters tapped. {T}: Add {W} or {U}. {1}{W}{U}: Until end of turn, Restless Anchorage becomes a 2/3 white and blue Bird creature with flying.",
                set_code="LCI",
            )
        )

        self.register(
            Card(
                name="Restless Reef",
                card_type=CardType.LAND,
                land_props=LandProperties(
                    produces=[ManaColor.BLUE, ManaColor.BLACK],
                    tier=LandTier.UTILITY,
                    enters_tapped=True,
                    has_activation=True,
                    activation_cost=ManaCost.from_string("1UB"),
                    activation_power=4,
                    activation_toughness=4,
                    basic_land_types=["Island", "Swamp"],
                ),
                rules_text="Enters tapped. {T}: Add {U} or {B}. {1}{U}{B}: Until end of turn, Restless Reef becomes a 4/4 blue and black Kraken creature.",
                set_code="LCI",
            )
        )

        self.register(
            Card(
                name="Inspiring Vantage",
                card_type=CardType.LAND,
                land_props=LandProperties(
                    produces=[ManaColor.RED, ManaColor.WHITE],
                    tier=LandTier.FAST_DUAL,
                    enters_tapped_conditional=3,
                    basic_land_types=["Mountain", "Plains"],
                ),
                rules_text="Inspiring Vantage enters the battlefield tapped unless you control two or fewer other lands. {T}: Add {R} or {W}.",
                set_code="FDN",
            )
        )

        self.register(
            Card(
                name="Hallowed Fountain",
                card_type=CardType.LAND,
                land_props=LandProperties(
                    produces=[ManaColor.WHITE, ManaColor.BLUE],
                    tier=LandTier.FAST_DUAL,
                    life_cost=2,
                    basic_land_types=["Plains", "Island"],
                ),
                rules_text="({T}: Add {W} or {U}.) As Hallowed Fountain enters, you may pay 2 life. If you don't, it enters tapped.",
                set_code="RNA",
            )
        )

        self.register(
            Card(
                name="Steam Vents",
                card_type=CardType.LAND,
                land_props=LandProperties(
                    produces=[ManaColor.BLUE, ManaColor.RED],
                    tier=LandTier.FAST_DUAL,
                    life_cost=2,
                    basic_land_types=["Island", "Mountain"],
                ),
                rules_text="({T}: Add {U} or {R}.) As Steam Vents enters, you may pay 2 life. If you don't, it enters tapped.",
                set_code="GRN",
            )
        )

    def _register_red_cards(self) -> None:
        """Register red creatures and spells."""
        self.register(
            Card(
                name="Monastery Swiftspear",
                card_type=CardType.CREATURE,
                mana_cost=ManaCost.from_string("R"),
                power=1,
                toughness=2,
                keywords={Keyword.HASTE, Keyword.PROWESS},
                triggers=[prowess_trigger()],
                rules_text="Haste. Prowess (Whenever you cast a noncreature spell, this creature gets +1/+1 until end of turn.)",
                set_code="FDN",
            )
        )

        self.register(
            Card(
                name="Heartfire Hero",
                card_type=CardType.CREATURE,
                mana_cost=ManaCost.from_string("R"),
                power=1,
                toughness=1,
                keywords={Keyword.PROWESS},
                triggers=[
                    prowess_trigger(),
                    Trigger(
                        trigger_type=TriggerType.DIES_SELF,
                        effect=TriggerEffect.DEAL_DAMAGE,
                        effect_value=0,  # Dynamic based on power
                        target="any",
                        description="When Heartfire Hero dies, it deals damage equal to its power to any target.",
                    ),
                ],
                rules_text="Prowess. When Heartfire Hero dies, it deals damage equal to its power to any target.",
                set_code="BLB",
            )
        )

        self.register(
            Card(
                name="Slickshot Show-Off",
                card_type=CardType.CREATURE,
                mana_cost=ManaCost.from_string("1R"),
                power=2,
                toughness=1,
                keywords={Keyword.FLYING, Keyword.HASTE},
                triggers=[
                    Trigger(
                        trigger_type=TriggerType.CAST_NONCREATURE,
                        effect=TriggerEffect.POWER_TOUGHNESS_BOOST,
                        effect_value=2,
                        toughness_value=0,
                        target="self",
                        description="Whenever you cast a noncreature spell, Slickshot Show-Off gets +2/+0 until end of turn.",
                    ),
                ],
                rules_text="Flying, haste. Whenever you cast a noncreature spell, Slickshot Show-Off gets +2/+0 until end of turn.",
                set_code="OTJ",
            )
        )

        self.register(
            Card(
                name="Phoenix Chick",
                card_type=CardType.CREATURE,
                mana_cost=ManaCost.from_string("R"),
                power=1,
                toughness=1,
                keywords={Keyword.FLYING, Keyword.HASTE},
                triggers=[
                    Trigger(
                        trigger_type=TriggerType.ATTACK_ANY,
                        effect=TriggerEffect.RETURN_TO_BATTLEFIELD,
                        effect_value=0,
                        target="self",
                        condition="when you attack with three or more creatures",
                        description="Whenever you attack with three or more creatures, return Phoenix Chick from your graveyard to the battlefield tapped and attacking.",
                    ),
                ],
                rules_text="Flying, haste. Phoenix Chick can't block. Whenever you attack with three or more creatures, return Phoenix Chick from your graveyard to the battlefield tapped and attacking.",
                set_code="DMU",
            )
        )

        self.register(
            Card(
                name="Goblin Guide",
                card_type=CardType.CREATURE,
                mana_cost=ManaCost.from_string("R"),
                power=2,
                toughness=2,
                keywords={Keyword.HASTE},
                triggers=[
                    Trigger(
                        trigger_type=TriggerType.ATTACK_SELF,
                        effect=TriggerEffect.DRAW_CARDS,  # Simplified: opponent may draw
                        effect_value=0,  # Conditional
                        target="opponent",
                        condition="defending player reveals top card; if land, put in hand",
                        description="Whenever Goblin Guide attacks, defending player reveals the top card of their library. If it's a land, that player puts it into their hand.",
                    ),
                ],
                rules_text="Haste. Whenever Goblin Guide attacks, defending player reveals the top card of their library. If it's a land, that player puts it into their hand.",
                set_code="ZEN",
            )
        )

        self.register(
            Card(
                name="Soul-Scar Mage",
                card_type=CardType.CREATURE,
                mana_cost=ManaCost.from_string("R"),
                power=1,
                toughness=2,
                keywords={Keyword.PROWESS},
                triggers=[prowess_trigger()],
                rules_text="Prowess. If a source you control would deal noncombat damage to a creature an opponent controls, put that many -1/-1 counters on that creature instead.",
                set_code="AKH",
            )
        )

        self.register(
            Card(
                name="Lightning Bolt",
                card_type=CardType.INSTANT,
                mana_cost=ManaCost.from_string("R"),
                deals_damage=3,
                can_target_any=True,
                rules_text="Deal 3 damage to any target.",
                set_code="FDN",
            )
        )

        self.register(
            Card(
                name="Shock",
                card_type=CardType.INSTANT,
                mana_cost=ManaCost.from_string("R"),
                deals_damage=2,
                can_target_any=True,
                rules_text="Deal 2 damage to any target.",
                set_code="FDN",
            )
        )

        self.register(
            Card(
                name="Searing Blood",
                card_type=CardType.INSTANT,
                mana_cost=ManaCost.from_string("1R"),
                deals_damage=2,
                requires_creature_target=True,
                rules_text="Deal 2 damage to target creature. When that creature dies this turn, deal 3 damage to its controller.",
                set_code="BNG",
            )
        )

        self.register(
            Card(
                name="Kumano Faces Kakkazan",
                card_type=CardType.ENCHANTMENT,
                mana_cost=ManaCost.from_string("R"),
                triggers=[
                    Trigger(
                        trigger_type=TriggerType.CHAPTER_I,
                        effect=TriggerEffect.DEAL_DAMAGE,
                        effect_value=1,
                        target="opponent",
                        description="I — Kumano Faces Kakkazan deals 1 damage to each opponent and each planeswalker they control.",
                    ),
                    Trigger(
                        trigger_type=TriggerType.CHAPTER_II,
                        effect=TriggerEffect.GRANT_KEYWORD,
                        keyword_granted=Keyword.HASTE,
                        target="next_creature",
                        condition="When you cast a creature spell this turn",
                        description="II — When you cast a creature spell this turn, it enters with a +1/+1 counter and gains haste until end of turn.",
                    ),
                    Trigger(
                        trigger_type=TriggerType.CHAPTER_III,
                        effect=TriggerEffect.TRANSFORM,
                        target="self",
                        description="III — Exile this Saga, then return it to the battlefield transformed under your control.",
                    ),
                ],
                rules_text="I — Deals 1 damage to each opponent and each planeswalker. II — When you cast a creature spell this turn, it enters with a +1/+1 counter and gains haste. III — Exile, return as Etching of Kumano.",
                set_code="NEO",
            )
        )

        self.register(
            Card(
                name="Etching of Kumano",
                card_type=CardType.CREATURE,
                mana_cost=ManaCost.from_string("R"),
                power=2,
                toughness=2,
                keywords={Keyword.HASTE},
                rules_text="Haste. Whenever a creature dealt damage by Etching of Kumano would die this turn, exile it instead.",
                set_code="NEO",
            )
        )

        self.register(
            Card(
                name="Play with Fire",
                card_type=CardType.INSTANT,
                mana_cost=ManaCost.from_string("R"),
                deals_damage=2,
                can_target_any=True,
                rules_text="Deal 2 damage to any target. If targeting a player, scry 1.",
                set_code="MID",
            )
        )

        self.register(
            Card(
                name="Lightning Strike",
                card_type=CardType.INSTANT,
                mana_cost=ManaCost.from_string("1R"),
                deals_damage=3,
                can_target_any=True,
                rules_text="Deal 3 damage to any target.",
                set_code="FDN",
            )
        )

        self.register(
            Card(
                name="Monstrous Rage",
                card_type=CardType.INSTANT,
                mana_cost=ManaCost.from_string("R"),
                requires_creature_target=True,
                is_pump_spell=True,
                rules_text="Target creature gets +2/+0 and gains trample until end of turn. Create a Monster Role token attached to it.",
                set_code="WOE",
            )
        )

    def _register_white_cards(self) -> None:
        """Register white creatures and spells."""
        self.register(
            Card(
                name="Warden of the Inner Sky",
                card_type=CardType.CREATURE,
                mana_cost=ManaCost.from_string("W"),
                power=1,
                toughness=2,
                keywords={Keyword.VIGILANCE},
                rules_text="Vigilance. Whenever Warden of the Inner Sky attacks, if it has three or more counters on it, exile it, then return it transformed.",
                set_code="LCI",
            )
        )

        self.register(
            Card(
                name="Resolute Reinforcements",
                card_type=CardType.CREATURE,
                mana_cost=ManaCost.from_string("1W"),
                power=1,
                toughness=1,
                keywords={Keyword.FLASH},
                triggers=[
                    Trigger(
                        trigger_type=TriggerType.ETB_SELF,
                        effect=TriggerEffect.CREATE_TOKEN,
                        effect_value=1,
                        token_type="1/1 white Soldier",
                        target="controller",
                        description="When Resolute Reinforcements enters the battlefield, create a 1/1 white Soldier creature token.",
                    ),
                ],
                rules_text="Flash. When Resolute Reinforcements enters the battlefield, create a 1/1 white Soldier creature token.",
                set_code="DMU",
            )
        )

        self.register(
            Card(
                name="Knight-Errant of Eos",
                card_type=CardType.CREATURE,
                mana_cost=ManaCost.from_string("4W"),
                power=4,
                toughness=4,
                has_convoke=True,
                rules_text="Convoke. When Knight-Errant of Eos enters the battlefield, look at the top six cards of your library. Put up to two creature cards with mana value 2 or less onto the battlefield.",
                set_code="MOM",
            )
        )

        self.register(
            Card(
                name="The Wandering Emperor",
                card_type=CardType.PLANESWALKER,
                mana_cost=ManaCost.from_string("2WW"),
                keywords={Keyword.FLASH},
                # Simplified: resolves as a one-shot effect (exile tapped
                # creature OR create 2/2 Samurai).  Full loyalty-ability
                # PW system is not modelled.
                rules_text="Flash. +1: Put a +1/+1 counter on up to one target creature. It gains first strike until your next turn. -1: Create a 2/2 white Samurai creature token with vigilance. -2: Exile target tapped creature.",
                set_code="NEO",
            )
        )

        self.register(
            Card(
                name="Get Lost",
                card_type=CardType.INSTANT,
                mana_cost=ManaCost.from_string("1W"),
                is_removal=True,
                can_target_nonland_permanent=True,
                rules_text="Exile target artifact, creature, enchantment, or planeswalker. Its controller creates two Map tokens.",
                set_code="LCI",
            )
        )

        self.register(
            Card(
                name="Sunfall",
                card_type=CardType.SORCERY,
                mana_cost=ManaCost.from_string("3WW"),
                is_removal=True,
                rules_text="Exile all creatures. Incubate X, where X is the number of creatures exiled this way.",
                set_code="MOM",
            )
        )

    def _register_blue_cards(self) -> None:
        """Register blue creatures and spells."""
        self.register(
            Card(
                name="Faerie Mastermind",
                card_type=CardType.CREATURE,
                mana_cost=ManaCost.from_string("1U"),
                power=2,
                toughness=1,
                keywords={Keyword.FLASH, Keyword.FLYING},
                rules_text="Flash, flying. Whenever an opponent draws their second card each turn, you draw a card. 3U: Each player draws a card.",
                set_code="MOM",
            )
        )

        self.register(
            Card(
                name="Haughty Djinn",
                card_type=CardType.CREATURE,
                mana_cost=ManaCost.from_string("1UU"),
                power=0,
                toughness=4,
                keywords={Keyword.FLYING},
                rules_text="Flying. Power equals instants and sorceries in your graveyard. Instant and sorcery spells you cast cost 1 less.",
                set_code="DMU",
            )
        )

        self.register(
            Card(
                name="No More Lies",
                card_type=CardType.INSTANT,
                mana_cost=ManaCost.from_string("WU"),
                is_counterspell=True,
                rules_text="Counter target spell unless its controller pays 3. If that spell is countered, exile it instead of putting it into its owner's graveyard.",
                set_code="MKM",
            )
        )

        self.register(
            Card(
                name="Make Disappear",
                card_type=CardType.INSTANT,
                mana_cost=ManaCost.from_string("1U"),
                is_counterspell=True,
                rules_text="Casualty 1. Counter target spell unless its controller pays 2.",
                set_code="SNC",
            )
        )

        self.register(
            Card(
                name="Memory Deluge",
                card_type=CardType.INSTANT,
                mana_cost=ManaCost.from_string("2UU"),
                draws_cards=2,
                rules_text="Look at the top X cards of your library, where X is the amount of mana spent to cast this spell. Put two of them into your hand and the rest on the bottom. Flashback 5UU.",
                set_code="MID",
            )
        )

        self.register(
            Card(
                name="Counterspell",
                card_type=CardType.INSTANT,
                mana_cost=ManaCost.from_string("UU"),
                is_counterspell=True,
                rules_text="Counter target spell.",
                set_code="FDN",
            )
        )

        self.register(
            Card(
                name="Dissolve",
                card_type=CardType.INSTANT,
                mana_cost=ManaCost.from_string("1UU"),
                is_counterspell=True,
                rules_text="Counter target spell. Scry 1.",
                set_code="THS",
            )
        )

    def _register_black_cards(self) -> None:
        """Register black creatures and spells."""
        self.register(
            Card(
                name="Deep-Cavern Bat",
                card_type=CardType.CREATURE,
                mana_cost=ManaCost.from_string("1B"),
                power=1,
                toughness=1,
                keywords={Keyword.FLYING, Keyword.LIFELINK},
                rules_text="Flying, lifelink. When Deep-Cavern Bat enters, look at target opponent's hand. You may exile a nonland card from it until Deep-Cavern Bat leaves the battlefield.",
                set_code="LCI",
            )
        )

        self.register(
            Card(
                name="Preacher of the Schism",
                card_type=CardType.CREATURE,
                mana_cost=ManaCost.from_string("2B"),
                power=2,
                toughness=4,
                keywords={Keyword.VIGILANCE},
                rules_text="Vigilance. Whenever Preacher of the Schism attacks, you may pay 2 life. If you do, create a 1/1 white and black Vampire creature token with lifelink.",
                set_code="LCI",
            )
        )

        self.register(
            Card(
                name="Sheoldred, the Apocalypse",
                card_type=CardType.CREATURE,
                mana_cost=ManaCost.from_string("2BB"),
                power=4,
                toughness=5,
                keywords={Keyword.DEATHTOUCH},
                rules_text="Deathtouch. Whenever you draw a card, you gain 2 life. Whenever an opponent draws a card, they lose 2 life.",
                set_code="DMU",
            )
        )

        self.register(
            Card(
                name="Go for the Throat",
                card_type=CardType.INSTANT,
                mana_cost=ManaCost.from_string("1B"),
                is_removal=True,
                requires_creature_target=True,
                target_restriction="nonartifact",
                rules_text="Destroy target nonartifact creature.",
                set_code="BRO",
            )
        )

        self.register(
            Card(
                name="Cut Down",
                card_type=CardType.INSTANT,
                mana_cost=ManaCost.from_string("B"),
                is_removal=True,
                requires_creature_target=True,
                target_restriction="pt_lte_5",
                rules_text="Destroy target creature with total power and toughness 5 or less.",
                set_code="DMU",
            )
        )

        self.register(
            Card(
                name="Duress",
                card_type=CardType.SORCERY,
                mana_cost=ManaCost.from_string("B"),
                rules_text="Target opponent reveals their hand. You choose a noncreature, nonland card from it. That player discards that card.",
                set_code="FDN",
            )
        )

    def _register_green_cards(self) -> None:
        """Register green creatures and spells."""
        self.register(
            Card(
                name="Llanowar Elves",
                card_type=CardType.CREATURE,
                mana_cost=ManaCost.from_string("G"),
                power=1,
                toughness=1,
                land_props=LandProperties(produces=[ManaColor.GREEN]),
                rules_text="Tap: Add G.",
                set_code="FDN",
            )
        )

        self.register(
            Card(
                name="Topiary Stomper",
                card_type=CardType.CREATURE,
                mana_cost=ManaCost.from_string("1GG"),
                power=4,
                toughness=4,
                keywords={Keyword.VIGILANCE},
                rules_text="Vigilance. When Topiary Stomper enters, search your library for a basic land card, put it onto the battlefield tapped, then shuffle. Topiary Stomper can't attack or block unless you control seven or more lands.",
                set_code="SNC",
            )
        )

        self.register(
            Card(
                name="Up the Beanstalk",
                card_type=CardType.ENCHANTMENT,
                mana_cost=ManaCost.from_string("1G"),
                draws_cards=1,
                rules_text="When Up the Beanstalk enters the battlefield, draw a card. Whenever you cast a spell with mana value 5 or greater, draw a card.",
                set_code="WOE",
            )
        )

    def _register_multicolor_cards(self) -> None:
        """Register multicolor cards."""
        self.register(
            Card(
                name="Atraxa, Grand Unifier",
                card_type=CardType.CREATURE,
                mana_cost=ManaCost.from_string("3WUBG"),
                power=7,
                toughness=7,
                keywords={Keyword.FLYING, Keyword.VIGILANCE, Keyword.DEATHTOUCH, Keyword.LIFELINK},
                rules_text="Flying, vigilance, deathtouch, lifelink. When Atraxa enters, reveal the top ten cards of your library. For each card type, put a card of that type into your hand. Put the rest on the bottom in random order.",
                set_code="ONE",
            )
        )

        self.register(
            Card(
                name="Leyline Binding",
                card_type=CardType.ENCHANTMENT,
                mana_cost=ManaCost.from_string("5W"),
                is_removal=True,
                can_target_nonland_permanent=True,
                has_domain_cost_reduction=True,
                keywords={Keyword.FLASH},
                rules_text="Flash. Domain — This spell costs 1 less to cast for each basic land type among lands you control. When Leyline Binding enters, exile target nonland permanent an opponent controls until Leyline Binding leaves the battlefield.",
                set_code="DMU",
            )
        )

        self.register(
            Card(
                name="Detention Sphere",
                card_type=CardType.ENCHANTMENT,
                mana_cost=ManaCost.from_string("1WU"),
                is_removal=True,
                can_target_nonland_permanent=True,
                rules_text="When Detention Sphere enters, exile target nonland permanent not named Detention Sphere and all other permanents with the same name as that permanent.",
                set_code="RTR",
            )
        )

        # Tokens
        self.register(
            Card(
                name="Incubator Token",
                card_type=CardType.ARTIFACT,
                mana_cost=ManaCost.from_string("0"),
                power=0,
                toughness=0,
                rules_text="{2}: Transform this artifact. (It becomes a 0/0 Phyrexian artifact creature. Put X +1/+1 counters on it equal to the incubate value.)",
                set_code="MOM",
            )
        )
        self.register(
            Card(
                name="Map Token",
                card_type=CardType.ARTIFACT,
                mana_cost=ManaCost.from_string("0"),
                power=0,
                toughness=0,
                rules_text="Map token. {1}, {T}, Sacrifice this artifact: Target creature you control explores.",
                set_code="LCI",
            )
        )

        # Only the sorcery-speed Map Token variant above is registered;
        # any second variant would be a duplicate.


def get_card_pool() -> dict[str, Card]:
    """Get the full card pool dictionary.

    Returns:
        Dictionary mapping card names to Card objects.

    """
    return CardRegistry.get_instance().get_all()


def get_card(name: str) -> Card:
    """Get a card by name from the registry.

    Args:
        name: Card name to look up.

    Returns:
        The card definition.

    """
    return CardRegistry.get_instance().get(name)


def create_card_instance(name: str, instance_id: int) -> Card:
    """Create a unique instance of a card.

    Args:
        name: Card name.
        instance_id: Unique identifier.

    Returns:
        New card instance.

    """
    return CardRegistry.get_instance().create_instance(name, instance_id)

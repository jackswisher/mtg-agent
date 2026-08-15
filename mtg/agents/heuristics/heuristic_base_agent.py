"""Shared heuristics for rule-based agents."""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

import numpy as np

from mtg.agents.base.base import BaseAgent
from mtg.env.action_mask import ActionKind
from mtg.env.card_definitions import Card, CardRegistry, CardType, Keyword


@dataclass(frozen=True)
class HeuristicStyle:
    """Behavior profile for an archetype."""

    aggression: float = 0.5
    defense: float = 0.5
    value: float = 0.5
    development: float = 0.5
    hold_up: float = 0.5
    curve: float = 0.4
    land_priority: float = 0.6
    risk_tolerance: float = 0.5


@dataclass(frozen=True)
class MulliganProfile:
    """Mulligan preferences for an archetype.

    Attributes:
        min_lands: Keep-hand minimum land count.
        max_lands: Keep-hand maximum land count.
        keep_one_land_with_early: Keep 1-land hands that have 2+ early spells.
        min_early_spells: Hands below this early-spell count get mulliganed.
        min_interaction: Slow decks demand some interaction; aggro doesn't.
        max_top_end_cards: Maximum allowed CMC>=5 cards in starting hand
            (too many expensive cards = uncastable opener). 0 = unlimited.
        max_hand_cmc_avg: Maximum average CMC of a keepable opener. If the
            average exceeds this, mulligan. 0 = unlimited.
    """

    min_lands: int = 2
    max_lands: int = 5
    keep_one_land_with_early: bool = False
    min_early_spells: int = 0
    min_interaction: int = 0
    max_top_end_cards: int = 0
    max_hand_cmc_avg: float = 0.0


class HeuristicBaseAgent(BaseAgent):
    """Base class for heuristic agents with shared decision flow."""

    def __init__(
        self,
        name: str,
        seed: int | None = None,
        style: HeuristicStyle | None = None,
        mulligan_profile: MulliganProfile | None = None,
        low_life_threshold: int = 5,
    ) -> None:
        super().__init__(name=name, deterministic=False)
        self.rng = np.random.default_rng(seed)
        self.card_registry = CardRegistry.get_instance()
        self.style = style or HeuristicStyle()
        self.mulligan_profile = mulligan_profile or MulliganProfile()
        self.low_life_threshold = low_life_threshold

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        info: dict[str, tp.Any] | None = None,
    ) -> int:
        """Select an action using heuristic scoring."""
        legal = np.where(action_mask > 0)[0]
        if len(legal) == 0:
            return 0

        if info is None:
            info = {}

        hand_size = info.get("hand_size", 0)
        keep_action = self._first_action(legal, info, ActionKind.KEEP_HAND.value)
        if keep_action is not None:
            return self._mulligan_decision(hand_size, legal, info)

        context = self._build_context(info)

        if (
            info.get("pending_action_type") is None
            and self._is_active_player(info)
            and info.get("phase_enum") in {"MAIN_PRECOMBAT", "MAIN_POSTCOMBAT"}
            and info.get("stack_size", 0) == 0
        ):
            land_action = self._best_land_action(legal, info, context)
            if land_action is not None:
                return land_action

            if info.get("phase_enum") == "MAIN_PRECOMBAT":
                creature_action = self._best_creature_action(legal, info, context)
                if creature_action is not None:
                    card = self._get_card_from_action(creature_action, info)
                    if card and self._should_prioritize_creature(card, context):
                        return creature_action

        scores: dict[int, float] = {}
        for action in legal:
            scores[int(action)] = self._score_action(int(action), info, context)

        best_score = max(scores.values())
        best_actions = sorted(a for a, s in scores.items() if s == best_score)
        # Seeded stochastic tie-break: when several actions share the
        # highest score, sample uniformly among them using ``self.rng``
        # rather than always picking the lowest action index. The RNG
        # is seeded in ``__init__`` so play stays bit-exactly
        # reproducible per ``(seed, trajectory)`` while removing the
        # structural bias toward whatever happens to come first in the
        # action schema (for example, the leftmost hand slot).
        if len(best_actions) == 1:
            return int(best_actions[0])
        idx = int(self.rng.integers(0, len(best_actions)))
        return int(best_actions[idx])

    def _first_action(
        self,
        legal: np.ndarray,
        info: dict[str, tp.Any],
        kind: str,
    ) -> int | None:
        metadata = info.get("action_metadata", {})
        for action in legal:
            if metadata.get(int(action), {}).get("kind") == kind:
                return int(action)
        return None

    def _action_kind(
        self,
        action: int,
        info: dict[str, tp.Any],
    ) -> tuple[str | None, int]:
        metadata = info.get("action_metadata", {})
        meta = metadata.get(int(action), {})
        return meta.get("kind"), int(meta.get("slot", -1))

    def _player_idx(self, info: dict[str, tp.Any]) -> int:
        """Return this agent's player index (0=player, 1=opponent)."""
        return int(info.get("player_idx", 0))

    def _is_active_player(self, info: dict[str, tp.Any]) -> bool:
        """True if this agent is the active player for the current turn."""
        return info.get("active_player_idx", 0) == self._player_idx(info)

    def _current_turn(self, info: dict[str, tp.Any]) -> int:
        """Return current game turn (1-indexed)."""
        try:
            return int(info.get("turn", 1))
        except (TypeError, ValueError):
            return 1

    def _turns_remaining(self, info: dict[str, tp.Any], max_turns: int = 20) -> int:
        """Return approximate turns remaining before game-end (capped at 1)."""
        return max(1, max_turns - self._current_turn(info))

    def _build_context(self, info: dict[str, tp.Any]) -> dict[str, tp.Any]:
        player_creatures = self._normalize_creatures(info.get("player_creatures", []))
        opponent_creatures = self._normalize_creatures(info.get("opponent_creatures", []))
        player_untapped = [c for c in player_creatures if not c["tapped"]]
        opponent_untapped = [c for c in opponent_creatures if not c["tapped"]]

        hand_cards = self._get_hand_cards(info)
        instants = [c for c in hand_cards if c.card_type == CardType.INSTANT]
        interaction = [
            c
            for c in hand_cards
            if c.is_removal
            or c.is_counterspell
            or (c.deals_damage > 0 and c.requires_creature_target)
        ]
        burn = [c for c in hand_cards if c.deals_damage > 0 and c.can_target_any]
        draw = [c for c in hand_cards if c.draws_cards > 0]

        opponent_attack_power = sum(c["power"] for c in opponent_creatures if c["power"] > 0)
        player_attack_power = sum(c["power"] for c in player_untapped if c["power"] > 0)

        return {
            "player_life": info.get("player_life", 20),
            "opponent_life": info.get("opponent_life", 20),
            "player_creatures": player_creatures,
            "opponent_creatures": opponent_creatures,
            "player_untapped": player_untapped,
            "opponent_untapped": opponent_untapped,
            "player_power": sum(c["power"] for c in player_creatures),
            "opponent_power": sum(c["power"] for c in opponent_creatures),
            "player_attack_power": player_attack_power,
            "opponent_attack_power": opponent_attack_power,
            "hand_size": info.get("hand_size", 0),
            "mana_available": info.get("mana_available", 0),
            "hand_cards": hand_cards,
            "instants": instants,
            "interaction": interaction,
            "burn": burn,
            "draw": draw,
        }

    def _best_creature_action(
        self,
        legal: np.ndarray,
        info: dict[str, tp.Any],
        context: dict[str, tp.Any],
    ) -> int | None:
        """Pick the best creature cast action available."""
        best_action = None
        best_score = float("-inf")
        for action in legal:
            kind, _slot = self._action_kind(int(action), info)
            if kind not in {ActionKind.CAST_SORCERY.value, ActionKind.CAST_INSTANT.value}:
                continue
            card = self._get_card_from_action(int(action), info)
            if card is None or card.card_type != CardType.CREATURE:
                continue
            score = self._score_card(card, info)
            if score > best_score:
                best_score = score
                best_action = int(action)
        return best_action

    def _should_prioritize_creature(
        self,
        card: Card,
        context: dict[str, tp.Any],
    ) -> bool:
        """Decide if we should cast a creature before other spells."""
        if not context["player_creatures"]:
            return True
        if card.has_haste:
            return True
        return self.style.development >= 0.6

    def _normalize_creatures(self, creatures: list) -> list[dict[str, tp.Any]]:
        """Normalize creature data from various formats.

        Handles both tuple format (name, power, toughness, tapped) and
        dictionary format {name, power, toughness, tapped, ...}.
        """
        normalized: list[dict[str, tp.Any]] = []
        for entry in creatures:
            if isinstance(entry, dict):
                # Dictionary format from _get_opponent_info
                normalized.append(
                    {
                        "name": entry.get("name", "Unknown"),
                        "power": entry.get("power", 0),
                        "toughness": entry.get("toughness", 0),
                        "tapped": entry.get("tapped", False),
                    }
                )
            elif isinstance(entry, list | tuple) and len(entry) >= 4:
                # Tuple format from _get_info
                normalized.append(
                    {
                        "name": entry[0],
                        "power": entry[1],
                        "toughness": entry[2],
                        "tapped": entry[3],
                    }
                )
        return normalized

    def _get_hand_cards(self, info: dict[str, tp.Any]) -> list[Card]:
        cards: list[Card] = []
        for entry in info.get("player_hand", []):
            if not entry:
                continue
            name = entry[0]
            try:
                cards.append(self.card_registry.get(name))
            except KeyError:
                continue
        return cards

    def _score_action(
        self,
        action: int,
        info: dict[str, tp.Any],
        context: dict[str, tp.Any],
    ) -> float:
        kind, slot = self._action_kind(action, info)
        pending = info.get("pending_action_type")

        if kind == ActionKind.PASS.value:
            return self._score_pass(info, context)

        if kind == ActionKind.PLAY_LAND.value:
            return self._score_land_play(action, info, context)

        if kind in {ActionKind.CAST_SORCERY.value, ActionKind.CAST_INSTANT.value}:
            return self._score_cast_spell(action, info, context)

        phase = info.get("phase_enum")
        is_active = self._is_active_player(info)

        if kind == ActionKind.CONFIRM.value and (
            pending == "attack" or (pending is None and phase == "COMBAT_BEGIN" and is_active)
        ):
            return self._score_attack_confirm(info, context)
        if kind == ActionKind.ATTACK_TOGGLE.value and (
            pending == "attack" or (pending is None and phase == "COMBAT_BEGIN" and is_active)
        ):
            return self._score_attack_toggle(slot, info, context)

        if kind == ActionKind.CONFIRM.value and (
            pending == "block"
            or (pending is None and phase == "COMBAT_ATTACKERS" and not is_active)
        ):
            return self._score_block_confirm(info, context)
        if kind == ActionKind.BLOCK_SELECT_ATTACKER.value and (
            pending == "block"
            or (pending is None and phase == "COMBAT_ATTACKERS" and not is_active)
        ):
            return self._score_block_attacker(slot, info, context)
        if kind == ActionKind.BLOCK_SELECT_BLOCKER.value and (
            pending == "block"
            or (pending is None and phase == "COMBAT_ATTACKERS" and not is_active)
        ):
            return self._score_blocker(slot, info, context)

        if kind == ActionKind.TARGET.value and pending == "spell_target":
            return self._score_target(slot, info, context)
        if (
            kind in {ActionKind.CONFIRM.value, ActionKind.AUTO_PAY.value}
            and pending == "mana_payment"
        ):
            return 0.5
        if kind == ActionKind.MANA_SOURCE.value and pending == "mana_payment":
            return self._score_mana_source(slot, info, context)

        if kind == ActionKind.BOTTOM_CARD.value and pending == "mulligan_bottom":
            return self._score_bottom_card(slot, info, context)
        if kind == ActionKind.DISCARD_CARD.value and pending == "discard":
            return self._score_discard_card(slot, info, context)
        if kind == ActionKind.CONFIRM.value and pending in {"mulligan_bottom", "discard"}:
            required = info.get("pending_required", 0)
            selected = info.get("pending_selected_indices", [])
            if len(selected) < required:
                return float("-inf")
            return 0.4

        if kind == ActionKind.ACTIVATE.value:
            return self._score_activation(action, info, context)

        if kind == ActionKind.CANCEL.value:
            # Cancel is only useful in specific selection situations
            if pending in {"mulligan_bottom", "discard"} and info.get("pending_required", 0) > 0:
                return -5.0  # Don't cancel mandatory selections
            if pending in {"attack", "block", None, ""}:
                # In attack/block declaration or no pending action, cancel loops forever
                # Always prefer confirm or pass to advance the game
                return -10.0
            # Allow cancel for optional target selection (to not cast a spell)
            return -0.2

        return float("-inf")

    def _get_card_from_action(self, action: int, info: dict[str, tp.Any]) -> Card | None:
        action_name = info.get("action_names", {}).get(action, "")
        for prefix in (
            "Cast: ",
            "Cast (instant): ",
            "Play land: ",
            "Bottom card: ",
            "Discard card: ",
            "Activate: ",
        ):
            if action_name.startswith(prefix):
                card_name = action_name[len(prefix) :]
                try:
                    return self.card_registry.get(card_name)
                except KeyError:
                    return None
        return None

    def _score_land_play(
        self,
        action: int,
        info: dict[str, tp.Any],
        context: dict[str, tp.Any],
    ) -> float:
        card = self._get_card_from_action(action, info)
        if card is None or not card.produces_mana:
            return float("-inf")

        score = 2.0 * self.style.land_priority
        if self._is_active_player(info) and info.get("phase_enum") in {
            "MAIN_PRECOMBAT",
            "MAIN_POSTCOMBAT",
        }:
            score += 0.8

        if card.enters_tapped:
            score -= 0.5
        else:
            score += 0.3

        current_colors = set()
        for land_name in info.get("player_lands", []):
            try:
                land = self.card_registry.get(land_name)
            except KeyError:
                continue
            for color in land.produces_mana:
                current_colors.add(color)

        needed_colors = set()
        for spell in context["hand_cards"]:
            for color in spell.mana_cost.colors:
                needed_colors.add(color)

        missing = needed_colors - current_colors
        if missing:
            for color in card.produces_mana:
                if color in missing:
                    score += 0.5

        effective_mana = context["mana_available"] + (0 if card.enters_tapped else 1)
        if effective_mana > context["mana_available"]:
            best_spell = self._best_hand_spell(context["hand_cards"], effective_mana)
            if best_spell is not None:
                score += 0.3 * self._score_card(best_spell, info)

        return score

    def _has_play_land_action(self, info: dict[str, tp.Any]) -> bool:
        action_mask = info.get("action_mask")
        action_names = info.get("action_names", {})
        if action_mask is None:
            return False
        try:
            mask = list(action_mask)
        except TypeError:
            return False
        for idx, name in action_names.items():
            if isinstance(idx, str):
                try:
                    idx = int(idx)
                except ValueError:
                    continue
            if idx < 0 or idx >= len(mask):
                continue
            if mask[idx] > 0 and name.startswith("Play land"):
                return True
        return False

    def _has_prowess_attacker(self, info: dict[str, tp.Any]) -> bool:
        for candidate in info.get("attack_candidates", []):
            card = self._safe_get_card(candidate.get("name", ""))
            if card and Keyword.PROWESS in card.keywords:
                return True
        return False

    def _has_tapped_prowess_creature(self, info: dict[str, tp.Any]) -> bool:
        for creature in self._normalize_creatures(info.get("player_creatures", [])):
            if not creature.get("tapped"):
                continue
            card = self._safe_get_card(creature.get("name", ""))
            if card and Keyword.PROWESS in card.keywords:
                return True
        return False

    def _has_castable_prowess_creature(
        self, context: dict[str, tp.Any], mana_available: int
    ) -> bool:
        for card in context.get("hand_cards", []):
            if (
                card.card_type == CardType.CREATURE
                and Keyword.PROWESS in card.keywords
                and card.mana_cost.cmc <= mana_available
            ):
                return True
        return False

    def _best_land_action(
        self,
        legal: np.ndarray,
        info: dict[str, tp.Any],
        context: dict[str, tp.Any],
    ) -> int | None:
        best_action = None
        best_score = float("-inf")
        for action in legal:
            kind, _ = self._action_kind(int(action), info)
            if kind != ActionKind.PLAY_LAND.value:
                continue
            score = self._score_land_play(int(action), info, context)
            if score > best_score:
                best_score = score
                best_action = int(action)
        return best_action

    def _best_hand_spell(self, cards: list[Card], mana_available: int) -> Card | None:
        best = None
        best_score = float("-inf")
        for card in cards:
            if card.card_type == CardType.LAND:
                continue
            if card.mana_cost.cmc > mana_available:
                continue
            score = self._score_card(card, {})
            if score > best_score:
                best_score = score
                best = card
        return best

    def _score_cast_spell(
        self,
        action: int,
        info: dict[str, tp.Any],
        context: dict[str, tp.Any],
    ) -> float:
        card = self._get_card_from_action(action, info)
        if card is None:
            return float("-inf")

        if self._should_skip_creature_buff_instant(action, info):
            return float("-inf")

        phase = info.get("phase_enum", "")
        is_active = self._is_active_player(info)
        if (
            (card.card_type == CardType.INSTANT or Keyword.FLASH in card.keywords)
            and is_active
            and phase in {"UPKEEP", "DRAW"}
            and not self._is_urgent_instant(card, context, phase=phase)
        ):
            return float("-inf")
        if (
            info.get("stack_size", 0) > 0
            and is_active
            and phase in {"MAIN_PRECOMBAT", "MAIN_POSTCOMBAT"}
            and not self._is_urgent_instant(card, context, phase=phase)
        ):
            return float("-inf")

        score = self._score_card(card, info)
        score += self._contextual_spell_bonus(card, info, context)
        if (
            is_active
            and info.get("phase_enum") in {"MAIN_PRECOMBAT", "MAIN_POSTCOMBAT"}
            and self._has_play_land_action(info)
        ):
            score -= 1.5
        if (
            card.card_type in {CardType.INSTANT, CardType.SORCERY}
            and is_active
            and info.get("phase_enum") == "MAIN_PRECOMBAT"
            and self._has_castable_prowess_creature(
                context, context["mana_available"] - card.mana_cost.cmc
            )
        ):
            urgent = False
            if card.card_type == CardType.INSTANT:
                urgent = self._is_urgent_instant(card, context, phase=phase)
            elif card.deals_damage > 0 and card.can_target_any:
                urgent = context["opponent_life"] <= card.deals_damage
            if not urgent:
                return float("-inf")
            score -= 2.0

        # Smart instant-removal/burn timing.
        #
        # Default posture: wait for END_STEP to preserve optionality.
        # Override: defensive agents (control/ramp) MUST cast removal on
        # their own turn if the opponent has creatures, because waiting means
        # taking another combat worth of damage. We therefore replace the
        # old hard `-inf` with a graded penalty that style.defense can overcome.
        if card.card_type == CardType.INSTANT and card.can_target_any:
            should_cast_now = self._should_cast_removal_now(card, info, context, phase)
            opp_has_creatures = bool(context.get("opponent_creatures"))

            if is_active:
                if not should_cast_now and not self._is_urgent_instant(card, context, phase=phase):
                    has_prowess = self._has_prowess_attacker(info)
                    has_attackers = info.get("player_attackers_available", 0) > 0
                    in_combat_with_attackers = (
                        phase in {"COMBAT_BEGIN", "COMBAT_ATTACKERS"} and has_attackers
                    )
                    if not has_prowess and not in_combat_with_attackers:
                        # Defensive agents with a creature to shoot: graded
                        # penalty. Aggro agents with no target: hard skip.
                        if self.style.defense >= 0.6 and opp_has_creatures:
                            # For control / ramp, still cast (positive EV vs tempo).
                            score -= 0.8
                        else:
                            return float("-inf")
            else:
                # Opponent's turn: strongly prefer END_STEP / CLEANUP windows.
                if (
                    phase not in {"END_STEP", "CLEANUP"}
                    and not should_cast_now
                    and not self._is_urgent_instant(card, context, phase=phase)
                ):
                    score -= 10.0

        # Sorcery-speed removal: defensive agents need to cast it on
        # their own main phase when the opponent has creatures, so
        # board wipes do not rot in hand while the agent dies to aggro.
        if (
            card.card_type == CardType.SORCERY
            and (card.is_removal or (card.deals_damage > 0 and card.requires_creature_target))
            and is_active
            and phase in {"MAIN_PRECOMBAT", "MAIN_POSTCOMBAT"}
            and context.get("opponent_creatures")
            and self.style.defense >= 0.6
        ):
            # Encourage casting sorcery removal in main phases for defensive
            # agents. Strength scales with how many enemy creatures exist.
            score += 0.6 * self.style.defense * len(context["opponent_creatures"])

        if context["mana_available"] > 0:
            utilization = min(1.0, card.mana_cost.cmc / context["mana_available"])
            score += self.style.curve * utilization

        return score

    def _contextual_spell_bonus(
        self,
        card: Card,
        info: dict[str, tp.Any],
        context: dict[str, tp.Any],
    ) -> float:
        score = 0.0
        player_life = context["player_life"]
        opponent_life = context["opponent_life"]
        opponent_power = context["opponent_power"]
        player_power = context["player_power"]
        phase = info.get("phase_enum", "")
        is_active = self._is_active_player(info)

        if card.deals_damage > 0 and card.can_target_any:
            if opponent_life <= card.deals_damage:
                score += 10.0
            score += 2.0 * self.style.aggression

        if card.is_removal or (card.deals_damage > 0 and card.requires_creature_target):
            if context["opponent_creatures"]:
                score += 1.5 * self.style.defense
            if opponent_power > player_power:
                score += 1.0 * self.style.defense

        if card.draws_cards > 0:
            if context["hand_size"] <= 2:
                score += 2.0 * self.style.value
            elif context["hand_size"] >= 6:
                score -= 0.5

        if card.card_type == CardType.CREATURE:
            if not context["player_creatures"]:
                score += 1.2 * self.style.development
            if opponent_power > player_power:
                score += 0.5 * self.style.defense
        elif card.card_type in {CardType.INSTANT, CardType.SORCERY}:
            if (
                is_active
                and phase == "MAIN_PRECOMBAT"
                and info.get("player_attackers_available", 0) > 0
            ):
                score += 1.0 * self.style.aggression
            if (
                is_active
                and phase == "MAIN_POSTCOMBAT"
                and info.get("player_attackers_available", 0) > 0
                and not self._is_urgent_instant(card, context, phase=phase)
            ):
                score -= 1.2

        if card.is_counterspell:
            if not is_active:
                score += 2.0 * self.style.hold_up
            else:
                score -= 1.0

        # Penalize casting non-urgent instants outside of optimal windows
        # (main phases and upkeep when we're active player should wait)
        if (
            card.card_type == CardType.INSTANT
            and is_active
            and phase in {"MAIN_PRECOMBAT", "MAIN_POSTCOMBAT", "UPKEEP", "DRAW"}
            and not self._is_urgent_instant(card, context, phase=phase)
        ):
            penalty = 0.8 * self.style.hold_up
            if phase == "MAIN_PRECOMBAT" and (
                self._has_prowess_attacker(info)
                or self._has_castable_prowess_creature(context, context["mana_available"])
            ):
                penalty = 0.0
            if phase == "MAIN_POSTCOMBAT" and self._has_tapped_prowess_creature(info):
                penalty = 1.6 * self.style.hold_up
            score -= penalty

        # Extra penalty for upkeep - almost never cast during upkeep
        if card.card_type == CardType.INSTANT and phase == "UPKEEP":
            score -= 2.0  # Strong penalty for upkeep casting

        if card.is_pump_spell:
            if is_active and phase == "COMBAT_BEGIN":
                score += 1.2 * self.style.aggression
            elif (
                is_active
                and phase == "MAIN_PRECOMBAT"
                and info.get("player_attackers_available", 0) > 0
            ):
                score += 0.8 * self.style.aggression
            elif phase in {"COMBAT_ATTACKERS", "COMBAT_BLOCKERS"}:
                score += 0.6 * self.style.aggression
            elif not is_active:
                score += 0.6 * self.style.defense

        if player_life <= self.low_life_threshold and card.gains_life > 0:
            score += 2.0

        return score

    def _is_urgent_instant(
        self, card: Card, context: dict[str, tp.Any], *, phase: str | None = None
    ) -> bool:
        """Check if an instant is urgent enough to cast immediately.

        Even urgent instants should wait for Main 1 if we have prowess creatures,
        because the prowess trigger is worth more than casting a turn early.
        """
        # If we're in upkeep/draw and have prowess creatures, wait for Main 1
        if phase in {"UPKEEP", "DRAW"}:
            for creature in context.get("player_creatures", []):
                c = self._safe_get_card(creature.get("name", ""))
                if c and Keyword.PROWESS in c.keywords:
                    return False
        if card.is_removal and context["opponent_power"] >= context["player_life"]:
            return True
        if card.deals_damage > 0 and card.can_target_any:
            return context["opponent_life"] <= card.deals_damage + 2
        return False

    def _should_cast_removal_now(
        self,
        card: Card,
        info: dict[str, tp.Any],
        context: dict[str, tp.Any],
        phase: str,
    ) -> bool:
        """Decide if removal should be cast NOW vs held for a better window.

        Pro-play principles:
        1. In combat, removal always trades up; cast it.
        2. ANY haste / menace / flying creature = kill on sight (virtual damage).
        3. Creatures with power >= 2 become lethal quickly: treat as threats.
        4. If life is at or below 10 OR opponent_power >= 50% of life,
           there is no time to wait; kill the biggest threat now.
        5. Preserve optionality ONLY when not under pressure and no
           threat on board deserves removal.
        """
        opponent_mana = info.get("opponent_mana_available", 0)
        opponent_creatures = context.get("opponent_creatures", [])
        player_life = context.get("player_life", 20)
        opp_attack_power = context.get("opponent_attack_power", 0)
        opponent_tapped_out = opponent_mana <= 1

        # Combat: removal is always good (kills a live attacker or an
        # eligible blocker, both high EV).
        in_combat = phase in {
            "COMBAT_BEGIN",
            "COMBAT_ATTACKERS",
            "COMBAT_BLOCKERS",
            "COMBAT_DAMAGE",
        }
        if in_combat and (
            info.get("opponent_declared_attackers", 0) > 0
            or bool(info.get("block_attacker_candidates", []))
        ):
            return True

        # Under pressure: don't hold removal while dying.
        if player_life <= 10 or opp_attack_power >= player_life // 2:
            for creature in opponent_creatures:
                if card.deals_damage >= creature.get("toughness", 99):
                    return True

        # Evaluate each creature for "kill on sight" threats.
        for creature in opponent_creatures:
            power = creature.get("power", 0)
            toughness = creature.get("toughness", 99)
            if card.deals_damage < toughness:
                continue  # can't kill this one anyway
            name = creature.get("name", "")
            creature_card = self._safe_get_card(name)
            keywords = creature_card.keywords if creature_card else set()

            has_haste = Keyword.HASTE in keywords
            has_flying = Keyword.FLYING in keywords
            has_menace = Keyword.MENACE in keywords
            has_lifelink = Keyword.LIFELINK in keywords
            has_double_strike = Keyword.DOUBLE_STRIKE in keywords
            has_trample = Keyword.TRAMPLE in keywords
            has_prowess = Keyword.PROWESS in keywords
            has_first_strike = Keyword.FIRST_STRIKE in keywords

            # Evasion or keyword threats: always kill immediately.
            if (
                has_haste
                or has_flying
                or has_menace
                or has_lifelink
                or has_double_strike
                or has_first_strike
                or has_prowess  # scales with instant/sorcery count
                or (has_trample and power >= 3)
            ):
                return True
            # Scaling threats: power >= 2 deals >= 6 damage over 3 turns.
            if power >= 2:
                return True
            # Lethal threat (power alone exceeds our life).
            if power >= player_life:
                return True

        # Still something alive and we're defensive? Clear it. A high-defense
        # agent (control / ramp) loses tempo by leaving even 1/1s alive
        # (they chip for 3-5 damage over the remaining turns).
        if self.style.defense >= 0.7 and opponent_creatures:
            return True

        # Otherwise: only cast on small creatures if opponent is tapped out
        # AND we're noticeably behind (so the trade actually matters).
        my_creatures = len(context.get("player_creatures", []))
        opp_creatures = len(opponent_creatures)
        behind = opp_creatures >= my_creatures + 2
        return opponent_tapped_out and behind

    def _current_aggression(self, info: dict[str, tp.Any]) -> float:
        """Return current aggression scalar.

        Dynamic adjustments:
        - Low life threshold: reduce aggression to prioritize survival.
        - Near-lethal opponent: ramp aggression to close the game.
        - Late-game turn boost: after turn 10, every turn we delay brings us
          closer to the 20-turn timeout (loss). This penalizes stalling.
        """
        aggression = self.style.aggression
        player_life = info.get("player_life", 20)
        opponent_life = info.get("opponent_life", 20)
        if player_life <= self.low_life_threshold:
            aggression *= 0.6
        if opponent_life <= 5:
            aggression *= 1.2
        # Late-game boost: we can't afford to time-out in a simulator with a
        # turn cap. After turn 10, scale aggression up toward the timeout.
        turn = self._current_turn(info)
        if turn > 10:
            aggression *= 1.0 + 0.06 * (turn - 10)
        return aggression

    def _score_attack_single(
        self,
        action: int,
        info: dict[str, tp.Any],
        context: dict[str, tp.Any],
    ) -> float:
        attackers = context["player_untapped"]
        _, slot = self._action_kind(action, info)
        if slot < 0 or slot >= len(attackers):
            return float("-inf")

        attacker = attackers[slot]
        damage = attacker["power"]
        aggression = self._current_aggression(info)

        blockers_remaining = max(0, len(attackers) - 1)
        expected_next = context["opponent_attack_power"]
        risk_penalty = self._attack_risk_penalty(
            expected_next, context["player_life"], blockers_remaining=blockers_remaining
        )

        score = aggression * damage - risk_penalty * 0.7
        if context["opponent_life"] <= damage:
            score += 10.0
        return score

    def _attack_risk_penalty(
        self,
        expected_damage_next: int,
        player_life: int,
        blockers_remaining: int,
    ) -> float:
        if player_life <= 0:
            return 10.0

        prevented = 0
        if blockers_remaining > 0:
            prevented = 0.5 * expected_damage_next

        net = max(0, expected_damage_next - prevented)
        danger_ratio = net / max(1, player_life)
        penalty = (1 - self.style.risk_tolerance) * danger_ratio * 8.0
        if net >= player_life:
            penalty += 8.0
        return penalty

    def _score_block_skip(self, info: dict[str, tp.Any], context: dict[str, tp.Any]) -> float:
        opponent_damage = context["opponent_attack_power"]
        if opponent_damage >= context["player_life"]:
            return -20.0
        return -0.2 * self.style.defense

    def _score_block(
        self,
        action: int,
        info: dict[str, tp.Any],
        context: dict[str, tp.Any],
    ) -> float:
        blockers = context["player_untapped"]
        _, slot = self._action_kind(action, info)
        if slot < 0 or slot >= len(blockers):
            return float("-inf")

        attacker = self._estimate_first_attacker(context["opponent_creatures"])
        if attacker is None:
            return float("-inf")

        blocker = blockers[slot]

        attacker_card = self._safe_get_card(attacker["name"])
        blocker_card = self._safe_get_card(blocker["name"])

        if (
            attacker_card
            and Keyword.FLYING in attacker_card.keywords
            and not (blocker_card and Keyword.FLYING in blocker_card.keywords)
        ):
            return -5.0

        attacker_power = attacker["power"]
        attacker_toughness = attacker["toughness"]
        blocker_power = blocker["power"]
        blocker_toughness = blocker["toughness"]

        attacker_dies = blocker_power >= attacker_toughness
        blocker_dies = attacker_power >= blocker_toughness

        score = 0.0
        if attacker_power >= context["player_life"]:
            score += 10.0

        if attacker_dies:
            score += (attacker_power + attacker_toughness) * self.style.defense
        if blocker_dies:
            score -= (blocker_power + blocker_toughness) * 0.6
        if attacker_dies and not blocker_dies:
            score += 2.0 * self.style.defense

        return score

    def _score_attack_confirm(self, info: dict[str, tp.Any], context: dict[str, tp.Any]) -> float:
        """Score confirming the attack declaration.

        Key principle: Don't confirm early if there are more good attackers to add.
        LETHAL CHECK: If all-in is lethal, only confirm when all are selected.
        """
        selected_ids = set(info.get("pending_attackers", []))
        selected_names = set(info.get("pending_attacker_names", []))
        candidates = info.get("attack_candidates", [])
        if not candidates:
            return -0.5

        # Count how many creatures are selected
        selected_count = sum(
            1
            for c in candidates
            if (
                (c.get("card_id") in selected_ids)
                or (c.get("card_id") is None and c.get("name") in selected_names)
            )
        )

        blockers_available = info.get("opponent_blockers_available", 0)

        if selected_count == 0:
            # No attackers selected - prefer to select at least one
            if blockers_available == 0:
                return -2.0  # Strongly prefer selecting when no blockers
            return -1.0  # Still prefer selecting even with blockers

        # LETHAL CHECK: If total board power is lethal, don't confirm until all are selected
        if self._is_lethal_on_board(info, context):
            unselected = len(candidates) - selected_count
            if unselected > 0:
                # Still have creatures to add: do not confirm yet.
                return -15.0 * unselected
            # All selected and it is lethal: confirm immediately.
            return 20.0

        # Calculate how many unselected creatures would survive being blocked
        # Use sorted blocker powers for realistic assessment
        opp_creatures = context.get("opponent_creatures", [])
        blocker_powers_sorted: list[int] = []
        for c in opp_creatures:
            if isinstance(c, dict):
                bp = c.get("power", 0)
            elif isinstance(c, list | tuple) and len(c) >= 2:
                bp = c[1]
            else:
                bp = 0
            blocker_powers_sorted.append(bp)
        blocker_powers_sorted.sort(reverse=True)

        # Match unselected creatures against available blockers
        # Strongest blockers are assigned first to the most threatening attackers
        unselected_creatures = []
        for c in candidates:
            is_sel = (c.get("card_id") in selected_ids) or (
                c.get("card_id") is None and c.get("name") in selected_names
            )
            if not is_sel and c.get("power", 0) > 0:
                unselected_creatures.append(c)

        # Sort unselected by power descending (opponent blocks strongest first)
        unselected_creatures.sort(key=lambda x: x.get("power", 0), reverse=True)

        # Count survivors: each unselected creature matched with a blocker
        survivors_not_selected = 0
        for i, creature in enumerate(unselected_creatures):
            toughness = creature.get("toughness", 0)
            # Which blocker would face this creature?
            # After the first `selected_count` blockers are used on selected attackers,
            # remaining blockers face unselected creatures
            blocker_idx = selected_count + i
            if blocker_idx < len(blocker_powers_sorted):
                assigned_blocker_power = blocker_powers_sorted[blocker_idx]
            else:
                # No blocker left for this creature: it gets through free.
                assigned_blocker_power = 0

            if toughness > assigned_blocker_power:
                survivors_not_selected += 1

        # Penalize confirming if there are more good attackers to select
        if survivors_not_selected > 0:
            # Each unselected survivor is lost damage opportunity
            return -1.5 * survivors_not_selected

        # All good attackers selected - calculate confirm score
        damage = sum(
            c["power"]
            for c in candidates
            if (
                (c.get("card_id") in selected_ids)
                or (c.get("card_id") is None and c.get("name") in selected_names)
            )
        )
        aggression = self._current_aggression(info)
        selected_total = len(selected_ids) if selected_ids else len(selected_names)
        risk = self._attack_risk_penalty(
            context["opponent_attack_power"],
            context["player_life"],
            blockers_remaining=max(0, len(candidates) - selected_total),
        )
        score = aggression * damage - risk
        if not info.get("opponent_creatures"):
            score += 0.6 * self.style.aggression
        if context["opponent_life"] <= damage:
            score += 12.0
        # Avoid oscillating attacker toggles when a selection is already made.
        if info.get("pending_action_type") == "attack":
            score = max(score, -0.05)
        return score

    def _is_lethal_on_board(
        self,
        info: dict[str, tp.Any],
        context: dict[str, tp.Any],
    ) -> bool:
        """Check if attacking with all creatures would be lethal."""
        candidates = info.get("attack_candidates", [])
        if not candidates:
            return False
        total_power = sum(c.get("power", 0) for c in candidates)
        opponent_life = context.get("opponent_life", 20)
        return total_power >= opponent_life and opponent_life > 0

    def _score_attack_toggle(
        self,
        slot: int,
        info: dict[str, tp.Any],
        context: dict[str, tp.Any],
    ) -> float:
        """Score toggling a creature for attack.

        Key considerations:
        0. LETHAL CHECK - if all-in is lethal, attack with everything
        1. Base score is power × aggression
        2. Bonus if no blockers available (free damage)
        3. Consider combat survival - attack if we survive being blocked
        4. Penalize attacking with creatures that would die to blockers
        """
        candidates = info.get("attack_candidates", [])
        if slot < 0 or slot >= len(candidates):
            return -1.0
        candidate = candidates[slot]
        selected_ids = set(info.get("pending_attackers", []))
        selected_names = set(info.get("pending_attacker_names", []))

        is_selected = (
            candidate.get("card_id") in selected_ids
            if candidate.get("card_id") is not None
            else candidate["name"] in selected_names
        )

        # LETHAL CHECK: If all-in attack kills the opponent, always attack
        if self._is_lethal_on_board(info, context):
            if is_selected:
                # Already selected for lethal: never deselect.
                return -10.0
            # Not yet selected: strongly select for lethal.
            return 15.0

        power = candidate["power"]
        toughness = candidate["toughness"]
        aggression = self._current_aggression(info)

        # Base score from damage potential
        base = aggression * power

        blockers_available = info.get("opponent_blockers_available", 0)

        if blockers_available == 0:
            # No blockers - free damage, always attack
            base += 0.7 * self.style.aggression
        else:
            # Blockers exist - evaluate combat survival
            # Get sorted blocker powers (descending) to match realistically
            opp_creatures = context.get("opponent_creatures", [])
            blocker_powers: list[int] = []
            best_blocker_toughness = 0
            for c in opp_creatures:
                if isinstance(c, dict):
                    bp = c.get("power", 0)
                    bt = c.get("toughness", 0)
                elif isinstance(c, list | tuple) and len(c) >= 2:
                    bp = c[1]
                    bt = c[2] if len(c) > 2 else 0
                else:
                    bp, bt = 0, 0
                blocker_powers.append(bp)
                best_blocker_toughness = max(best_blocker_toughness, bt)

            if blocker_powers:
                # Sort descending: strongest blockers first.
                blocker_powers.sort(reverse=True)
                # Use median blocker power for a more realistic assessment
                # (not every creature will be blocked by the best one)
                median_idx = len(blocker_powers) // 2
                typical_blocker_power = blocker_powers[median_idx]

                # Would we survive being blocked by a typical blocker?
                survives_block = toughness > typical_blocker_power
                # Would we kill the blocker?
                kills_blocker = power >= best_blocker_toughness

                if survives_block:
                    # We survive - good to attack
                    base += 0.5 * aggression
                    if kills_blocker:
                        # We trade favorably or kill blocker
                        base += 0.8 * aggression
                else:
                    # We would die if blocked - penalize small creatures
                    if power <= 2:
                        base -= 1.5 * self.style.defense
                    else:
                        # High power creatures might be worth trading
                        base -= 0.5 * self.style.defense

        if is_selected:
            # Already selected - should we deselect?
            # Only deselect if the original decision to attack was bad (base very negative)
            # Otherwise, prefer confirm to avoid oscillation
            if base < -2.0:
                # Bad attack - allow deselection with mild positive score
                return 0.5
            else:
                # Decent attack or neutral - strongly discourage deselection
                # Prefer confirm instead
                return -5.0
        return base

    def _score_block_confirm(self, info: dict[str, tp.Any], context: dict[str, tp.Any]) -> float:
        assignments = info.get("pending_block_assignments", {})
        if not assignments:
            return self._score_block_skip(info, context)
        score = 0.0
        attackers = {c["name"]: c for c in info.get("block_attacker_candidates", [])}
        blockers = {c["name"]: c for c in info.get("blocker_candidates", [])}
        for blocker_name, attacker_name in assignments.items():
            attacker = attackers.get(attacker_name)
            blocker = blockers.get(blocker_name)
            if attacker and blocker:
                attacker_value = attacker["power"] + attacker["toughness"]
                blocker_value = blocker["power"] + blocker["toughness"]
                if blocker["power"] >= attacker["toughness"]:
                    score += attacker_value * self.style.defense
                if attacker["power"] >= blocker["toughness"]:
                    score -= blocker_value * 0.4
        return max(score, 0.2)

    def _score_block_attacker(
        self,
        slot: int,
        info: dict[str, tp.Any],
        context: dict[str, tp.Any],
    ) -> float:
        """Score choosing which attacker to block next.

        Priorities (in order):
        1. Lethal damage prevention: the attacker whose unblocked
           damage tips total damage over the life total.
        2. Evasive / keyword threats (lifelink, double strike, flying)
           that at least one blocker CAN block.
        3. Highest power attackers that at least one blocker can kill.
        """
        if not info.get("blocker_candidates"):
            return -1.0
        if info.get("pending_block_attacker_id"):
            return -0.3
        attackers = info.get("block_attacker_candidates", [])
        if slot < 0 or slot >= len(attackers):
            return -1.0

        atk = attackers[slot]
        atk_power = atk.get("power", 0)
        atk_toughness = atk.get("toughness", 0)
        blockers = info.get("blocker_candidates", [])
        if not blockers:
            return -1.0

        # Total incoming damage vs life: pick attackers that push toward lethal.
        total_incoming = sum(a.get("power", 0) for a in attackers)
        life = context.get("player_life", 20)
        # We "must block" attackers that push damage over life (weighted by
        # how much overkill they add).
        lethal_pressure = max(0, total_incoming - life)

        atk_card = self._safe_get_card(atk.get("name", ""))
        atk_keywords = atk_card.keywords if atk_card else set()
        has_flying = Keyword.FLYING in atk_keywords
        has_lifelink = Keyword.LIFELINK in atk_keywords
        has_double_strike = Keyword.DOUBLE_STRIKE in atk_keywords
        has_trample = Keyword.TRAMPLE in atk_keywords

        # Can any of our blockers legally block this attacker?
        blockable = False
        best_blocker_trade = float("-inf")
        for b in blockers:
            b_card = self._safe_get_card(b.get("name", ""))
            b_keywords = b_card.keywords if b_card else set()
            # Flying attackers require flying / reach blockers.
            if has_flying and not (Keyword.FLYING in b_keywords or Keyword.REACH in b_keywords):
                continue
            blockable = True
            b_power = b.get("power", 0)
            b_toughness = b.get("toughness", 0)
            atk_dies = b_power >= atk_toughness
            blocker_dies = atk_power >= b_toughness
            trade = 0.0
            if atk_dies:
                trade += atk_power + atk_toughness
            if blocker_dies:
                trade -= (b_power + b_toughness) * 0.6
            best_blocker_trade = max(best_blocker_trade, trade)

        if not blockable:
            # No legal blocker: do not waste the declaration.
            return -3.0

        score = 0.0
        # Prevent lethal (strongest priority).
        if lethal_pressure > 0 or atk_power >= life:
            score += 100.0  # Dominates everything else.
        # Keyword-based threat weighting.
        if has_lifelink:
            score += 3.0
        if has_double_strike:
            score += 2.5
        if has_flying:
            score += 1.0
        if has_trample and atk_power >= 3:
            score += 1.0
        # Raw power scaled by defense style.
        score += atk_power * self.style.defense
        # Bonus if the best available trade is favorable.
        if best_blocker_trade > 0:
            score += best_blocker_trade * 0.4
        return score

    def _score_blocker(
        self,
        slot: int,
        info: dict[str, tp.Any],
        context: dict[str, tp.Any],
    ) -> float:
        """Score assigning THIS blocker to the currently-chosen attacker.

        Considerations:
        1. Prefer the blocker that survives and kills the attacker.
        2. Never block a flying attacker with a non-flier / non-reach blocker.
        3. If no clean trade exists, use the most expendable blocker to chump
           a lethal-threat attacker.
        4. Avoid trading a large creature for a small one without need.
        """
        if not info.get("pending_block_attacker_id"):
            return -1.5
        blockers = info.get("blocker_candidates", [])
        if slot < 0 or slot >= len(blockers):
            return -1.0
        # Already have an assignment in flight: do not over-commit.
        if info.get("pending_block_assignments"):
            return -0.2

        blocker = blockers[slot]
        b_power = blocker.get("power", 0)
        b_toughness = blocker.get("toughness", 0)
        b_card = self._safe_get_card(blocker.get("name", ""))
        b_keywords = b_card.keywords if b_card else set()

        # Identify which attacker we're blocking.
        target_atk = None
        target_atk_id = info.get("pending_block_attacker_id")
        for a in info.get("block_attacker_candidates", []):
            if a.get("card_id") == target_atk_id:
                target_atk = a
                break
        if target_atk is None:
            # Fallback: pick the first attacker.
            attackers = info.get("block_attacker_candidates", [])
            if not attackers:
                return -1.0
            target_atk = attackers[0]

        atk_power = target_atk.get("power", 0)
        atk_toughness = target_atk.get("toughness", 0)
        atk_card = self._safe_get_card(target_atk.get("name", ""))
        atk_keywords = atk_card.keywords if atk_card else set()

        has_flying = Keyword.FLYING in atk_keywords
        if has_flying and not (Keyword.FLYING in b_keywords or Keyword.REACH in b_keywords):
            return -10.0  # Illegal block at any price.

        blocker_dies = atk_power >= b_toughness
        atk_dies = b_power >= atk_toughness

        score = 0.0
        # Pure trade-up: we kill them and we live.
        if atk_dies and not blocker_dies:
            score += (atk_power + atk_toughness) * 1.0 + 4.0
        # Mutual kill: slightly favorable if attacker is larger.
        elif atk_dies and blocker_dies:
            score += (atk_power + atk_toughness) - (b_power + b_toughness) * 0.4 + 1.0
        # Chump-block: blocker dies, attacker lives. Fine only if attacker
        # would have been lethal or we're very low on life.
        elif blocker_dies and not atk_dies:
            life = context.get("player_life", 20)
            lethal_incoming = sum(
                a.get("power", 0) for a in info.get("block_attacker_candidates", [])
            )
            if atk_power >= life or lethal_incoming > life:
                # Chump the threat: give up toughness to buy a turn.
                score += 2.0 - b_toughness * 0.3
            else:
                # Pointless trade: discourage.
                score -= (b_power + b_toughness) * 0.5
        else:
            # Nobody dies. Tiny bonus if the block absorbs damage we need.
            score += 0.3 * self.style.defense
        return score * self.style.defense + 0.1  # tiny positive baseline

    def _score_target(
        self,
        slot: int,
        info: dict[str, tp.Any],
        context: dict[str, tp.Any],
    ) -> float:
        """Score a target for burn/removal spells using pro-play heuristics.

        Key considerations:
        - Board state: ahead vs behind determines removal vs race
        - Virtual damage: creature power * expected turns is often > one-time burn
        - Threat assessment: high-power creatures are priority targets when behind
        """
        candidates = info.get("pending_target_candidates", [])
        if slot < 0 or slot >= len(candidates):
            return -1.0
        target = candidates[slot]
        spell_name = info.get("pending_spell_name")
        if not spell_name:
            return 0.0
        try:
            card = self.card_registry.get(spell_name)
        except KeyError:
            return 0.0
        if card.is_counterspell:
            return 2.0

        # Calculate board advantage
        my_power = sum(
            c.get("power", 0)
            if isinstance(c, dict)
            else (c[1] if isinstance(c, list | tuple) else 0)
            for c in context.get("player_creatures", [])
        )
        opp_power = sum(
            c.get("power", 0)
            if isinstance(c, dict)
            else (c[1] if isinstance(c, list | tuple) else 0)
            for c in context.get("opponent_creatures", [])
        )
        behind_on_board = opp_power > my_power
        at_parity = opp_power == my_power and opp_power > 0

        if target["kind"] == "player":
            if target["name"] == "Opponent":
                if card.deals_damage > 0 and card.can_target_any:
                    # Lethal check - always go face if lethal
                    if context["opponent_life"] <= card.deals_damage:
                        return 10.0

                    # Behind on board: strongly prefer removing creatures
                    if behind_on_board:
                        return 0.5  # Very low score for face when behind

                    # At parity with creatures: prefer creature removal
                    if at_parity:
                        return 1.0

                    # Ahead on board: race - go face
                    return 3.0 + self.style.aggression
                return 2.0 + self.style.aggression
            return -1.0

        if target["kind"] == "permanent":
            try:
                target_card = self.card_registry.get(target["name"])
            except KeyError:
                return 0.0
            if target_card.card_type == CardType.CREATURE:
                target = {**target, "kind": "creature"}
            else:
                if card.is_removal:
                    if target_card.card_type == CardType.PLANESWALKER:
                        return 6.0 * self.style.defense
                    return 3.0 * self.style.defense
                return 0.0

        if target["kind"] == "creature":
            creatures = info.get("opponent_creatures", []) + info.get("player_creatures", [])
            power = 0
            has_haste = False
            for creature in creatures:
                if isinstance(creature, dict):
                    if creature.get("name") == target["name"]:
                        power = max(power, creature.get("power", 0))
                        if creature.get("has_haste"):
                            has_haste = True
                elif (
                    isinstance(creature, list | tuple)
                    and len(creature) >= 2
                    and creature[0] == target["name"]
                ):
                    power = max(power, creature[1])

            if card.is_pump_spell:
                return power * self.style.aggression

            if card.is_removal or card.deals_damage > 0:
                # Virtual damage: a creature deals `power` per turn for
                # however many turns remain. Early-game creatures are more
                # valuable to remove than late-game ones because they have
                # more turns of virtual damage ahead.
                turns_left = self._turns_remaining(info)
                # Cap at 4 (removing something that would die in combat anyway
                # isn't worth 10 turns of virtual value).
                expected_turns_alive = max(1, min(turns_left, 4))
                virtual_damage = power * expected_turns_alive

                # Haste = immediate damage (effectively + one turn's worth).
                if has_haste:
                    virtual_damage += power

                # Board state modifiers.
                board_modifier = 1.0
                if behind_on_board:
                    board_modifier = 2.0
                elif at_parity:
                    board_modifier = 1.5

                # Life total urgency: low life = removal is even more valuable.
                life = context.get("player_life", 20)
                if life <= 8:
                    board_modifier *= 1.5

                return virtual_damage * board_modifier * self.style.defense

        return 0.0

    def _score_mana_source(
        self,
        slot: int,
        info: dict[str, tp.Any],
        context: dict[str, tp.Any],
    ) -> float:
        sources = info.get("pending_mana_source_cards", [])
        if slot < 0 or slot >= len(sources):
            return -1.0
        colors = sources[slot].get("colors", [])
        return -0.2 * len(colors)

    def _score_bottom_card(
        self,
        slot: int,
        info: dict[str, tp.Any],
        context: dict[str, tp.Any],
    ) -> float:
        hand = info.get("player_hand", [])
        if slot < 0 or slot >= len(hand):
            return -1.0
        selected = set(info.get("pending_selected_indices", []))
        required = info.get("pending_required", 0)
        if slot in selected:
            return -2.0
        if len(selected) >= required and required > 0:
            return -1.0
        card_name = hand[slot][0]
        try:
            card = self.card_registry.get(card_name)
        except KeyError:
            return 0.0
        base = -self._score_card(card, info)
        if required > len(selected):
            return max(0.05, base)
        return base

    def _score_discard_card(
        self,
        slot: int,
        info: dict[str, tp.Any],
        context: dict[str, tp.Any],
    ) -> float:
        return self._score_bottom_card(slot, info, context)

    def _score_activation(
        self,
        action: int,
        info: dict[str, tp.Any],
        context: dict[str, tp.Any],
    ) -> float:
        card = self._get_card_from_action(action, info)
        if not card or not card.land_props:
            return -1.0
        power = card.land_props.activation_power
        toughness = card.land_props.activation_toughness
        score = 0.4 * power + 0.3 * toughness
        if context["opponent_power"] > context["player_power"]:
            score += 0.6 * self.style.defense
        else:
            score += 0.6 * self.style.aggression
        return score

    def _estimate_first_attacker(
        self,
        opponent_creatures: list[dict[str, tp.Any]],
    ) -> dict[str, tp.Any] | None:
        for creature in opponent_creatures:
            if creature["power"] > 0:
                return creature
        return None

    def _score_pass(self, info: dict[str, tp.Any], context: dict[str, tp.Any]) -> float:
        """Score passing priority.

        Pass is the default "do nothing" action; every other useful action
        should beat it. We only add significant weight to pass when:
        - Stack resolution is pending (priority needs to move on)
        - We genuinely want to hold interaction until opponent's turn/end step
        """
        score = 0.0
        is_active = self._is_active_player(info)
        phase = info.get("phase_enum", "")

        # Passing to resolve the stack is usually correct, but don't let it
        # drown out useful responses (like countering or removing the spell).
        if info.get("stack_size", 0) > 0:
            score += 0.6

        # Never pass up lethal attacks.
        if (
            is_active
            and phase in {"COMBAT_BEGIN", "COMBAT_ATTACKERS"}
            and info.get("attack_candidates")
            and info.get("opponent_blockers_available", 0) == 0
        ):
            score -= 1.5

        # Holding cards past the discard threshold is bad.
        if context["hand_size"] > 7:
            score -= 2.5

        # Holding up instants during our own main phases is rarely worth
        # it: that is what end-of-turn windows are for. Don't reward
        # passing unnecessarily when we have no reason to hold mana.
        if context["instants"] and (not is_active or phase == "END_STEP"):
            score += 1.2 * self.style.hold_up

        # Holding interaction for the opponent's turn.
        if context["interaction"] and not is_active:
            score += 0.8 * self.style.hold_up

        # Late-game: don't idle away turns when we could be pressuring.
        turn = self._current_turn(info)
        if is_active and turn >= 8 and context.get("hand_cards"):
            has_castable = any(
                c.card_type != CardType.LAND and c.mana_cost.cmc <= context.get("mana_available", 0)
                for c in context["hand_cards"]
            )
            if has_castable and phase in {"MAIN_PRECOMBAT", "MAIN_POSTCOMBAT"}:
                score -= 0.5 * self.style.aggression

        return score

    def _safe_get_card(self, name: str) -> Card | None:
        try:
            return self.card_registry.get(name)
        except KeyError:
            return None

    def _should_skip_creature_buff_instant(
        self,
        action: int,
        info: dict[str, tp.Any],
    ) -> bool:
        action_name = info.get("action_names", {}).get(action, "")
        prefix = "Cast (instant): "
        if not action_name.startswith(prefix):
            return False
        card_name = action_name[len(prefix) :]
        if not self._is_creature_buff_instant(card_name):
            return False
        return not self._should_cast_creature_buff_instant(info)

    def _is_creature_buff_instant(self, card_name: str) -> bool:
        try:
            card = self.card_registry.get(card_name)
        except KeyError:
            return False
        return card.card_type == CardType.INSTANT and (
            card.is_pump_spell
            or (card.requires_creature_target and not card.deals_damage and not card.is_removal)
        )

    def _should_cast_creature_buff_instant(self, info: dict[str, tp.Any]) -> bool:
        if info.get("hand_size", 0) > 7:
            return True

        is_active = self._is_active_player(info)
        phase_enum = info.get("phase_enum", "")

        if is_active and phase_enum in {
            "MAIN_PRECOMBAT",
            "COMBAT_BEGIN",
            "COMBAT_ATTACKERS",
        }:
            return info.get("player_attackers_available", 0) > 0

        if not is_active and phase_enum in {"COMBAT_ATTACKERS", "COMBAT_BLOCKERS"}:
            return (
                info.get("opponent_declared_attackers", 0) > 0
                and info.get("player_blockers_available", 0) > 0
            )

        return False

    def _mulligan_decision(
        self,
        hand_size: int,
        legal: np.ndarray,
        info: dict[str, tp.Any],
    ) -> int:
        """Decide whether to mulligan or keep the opening hand.

        Keep criteria reflect pro-play mulligan theory:
        - 0-land hands: always mulligan (unless forced to 1 card).
        - 2-5 lands (default band): preferred keep range.
        - 1-land hands: keepable only with 2+ early plays (aggro).
        - Keepable hands must also have:
            * enough early spells (min_early_spells)
            * enough interaction (min_interaction) for slower decks
            * not too many top-end cards that won't be castable in time
            * a reasonable average CMC so we can curve into something.

        Low hand sizes (<=5) short-circuit the above; at that point the
        cost of further mulligans outweighs imperfect hands.
        """
        lands = info.get("lands", 0)
        keep_action = self._first_action(legal, info, ActionKind.KEEP_HAND.value)
        mull_action = self._first_action(legal, info, ActionKind.MULLIGAN.value)

        # 0-land hands: always mulligan unless we've bottomed out.
        if lands == 0:
            if hand_size <= 1 and keep_action is not None:
                return keep_action
            if mull_action is not None:
                return mull_action

        # Low-count hands: mulligan cost gets too high below 5 cards.
        if hand_size <= 5 and lands >= 1 and keep_action is not None:
            return keep_action

        hand_cards = self._get_hand_cards(info)
        early_spells = sum(
            1 for c in hand_cards if c.card_type != CardType.LAND and c.mana_cost.cmc <= 2
        )
        interaction = sum(
            1 for c in hand_cards if c.is_removal or c.is_counterspell or c.deals_damage > 0
        )
        top_end = sum(
            1 for c in hand_cards if c.card_type != CardType.LAND and c.mana_cost.cmc >= 5
        )
        non_land_cards = [c for c in hand_cards if c.card_type != CardType.LAND]
        if non_land_cards:
            avg_cmc = sum(c.mana_cost.cmc for c in non_land_cards) / len(non_land_cards)
        else:
            avg_cmc = 0.0

        # 1-land special-case for aggro: keep with 2+ early plays (treat as
        # "keep and hope to draw a second land").
        if (
            lands == 1
            and self.mulligan_profile.keep_one_land_with_early
            and early_spells >= 2
            and keep_action is not None
        ):
            return keep_action

        # Core keep-band: land count, early spells, interaction, curve.
        if (
            lands < self.mulligan_profile.min_lands or lands > self.mulligan_profile.max_lands
        ) and mull_action is not None:
            return mull_action

        if (
            early_spells < self.mulligan_profile.min_early_spells
            and hand_size >= 6
            and mull_action is not None
        ):
            return mull_action

        if (
            interaction < self.mulligan_profile.min_interaction
            and hand_size >= 6
            and mull_action is not None
        ):
            return mull_action

        # Too many top-end cards: hand will rot without early plays.
        max_top_end = self.mulligan_profile.max_top_end_cards
        if max_top_end > 0 and top_end > max_top_end and hand_size >= 6 and mull_action is not None:
            return mull_action

        # Average CMC too high: opener cannot function on curve.
        max_cmc = self.mulligan_profile.max_hand_cmc_avg
        if max_cmc > 0.0 and avg_cmc > max_cmc and hand_size >= 6 and mull_action is not None:
            return mull_action

        if keep_action is not None:
            return keep_action

        return int(legal[0])

    def _score_card(self, card: Card, info: dict[str, tp.Any]) -> float:
        raise NotImplementedError

    def get_action_probabilities(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        info: dict[str, tp.Any] | None = None,
    ) -> np.ndarray:
        """Return probability distribution over actions (deterministic)."""
        action = self.select_action(observation, action_mask, info)
        probs = np.zeros(len(action_mask))
        probs[action] = 1.0
        return probs

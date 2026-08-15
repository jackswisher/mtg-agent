"""Action mask builder for legal action filtering with priority system."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple

import numpy as np

from mtg.env.card_definitions import Card, CardType, Keyword
from mtg.env.rules import (
    PRIORITY_PHASES,
    SORCERY_SPEED_PHASES,
    GamePhase,
    GameState,
    RulesEngine,
)


class ActionKind(Enum):
    """High-level action categories.

    Note: ``ATTACK_ALL`` is used only by the scripted opponent
    (``_execute_opponent_action``) as a shortcut to declare all legal
    attackers at once.  The RL agent uses ``ATTACK_TOGGLE`` + ``CONFIRM``
    which allows selecting any subset of attackers.
    """

    PASS = "pass"
    KEEP_HAND = "keep_hand"
    MULLIGAN = "mulligan"
    CONFIRM = "confirm"
    CANCEL = "cancel"
    AUTO_PAY = "auto_pay"
    BOTTOM_CARD = "bottom_card"
    DISCARD_CARD = "discard_card"
    PLAY_LAND = "play_land"
    CAST_SORCERY = "cast_sorcery"
    CAST_INSTANT = "cast_instant"
    ACTIVATE = "activate"
    ATTACK_TOGGLE = "attack_toggle"
    ATTACK_ALL = "attack_all"
    BLOCK_SELECT_ATTACKER = "block_select_attacker"
    BLOCK_SELECT_BLOCKER = "block_select_blocker"
    TARGET = "target"
    MANA_SOURCE = "mana_source"


class Action(NamedTuple):
    """Represents a decoded action."""

    kind: ActionKind
    slot: int = -1


@dataclass
class ActionSpaceConfig:
    """Configuration for action space.

    ``max_hand_slots`` is intentionally pinned to the same value as
    :class:`mtg.env.observation.ObservationConfig.max_hand_size` (10) so
    that every actionable hand slot in the action mask is also
    observable in the observation tensor. Without this contract the
    policy would be choosing between hand slots whose cards it cannot
    see, leaving it with no signal beyond positional bias.

    The MTG end-of-turn rule enforces ``max hand size = 7``, and
    mid-turn hand sizes greater than 10 are vanishingly rare in
    practice across all deck archetypes shipped with this codebase, so
    this cap does not materially change reachable game states.
    """

    max_hand_slots: int = 10
    max_creature_slots: int = 60
    max_permanent_slots: int = 60
    max_target_slots: int = 122


@dataclass
class ActionIndexMap:
    """Maps action kinds to index ranges."""

    max_hand_slots: int
    max_creature_slots: int
    max_permanent_slots: int
    max_target_slots: int

    pass_idx: int = 0
    keep_idx: int = 1
    mulligan_idx: int = 2
    confirm_idx: int = 3
    cancel_idx: int = 4
    auto_pay_idx: int = 5

    @property
    def bottom_start(self) -> int:
        """Start index for bottom-card actions (mulligan)."""
        return self.auto_pay_idx + 1

    @property
    def discard_start(self) -> int:
        """Start index for discard actions."""
        return self.bottom_start + self.max_hand_slots

    @property
    def play_land_start(self) -> int:
        """Start index for play-land actions."""
        return self.discard_start + self.max_hand_slots

    @property
    def cast_sorcery_start(self) -> int:
        """Start index for sorcery-speed cast actions."""
        return self.play_land_start + self.max_hand_slots

    @property
    def cast_instant_start(self) -> int:
        """Start index for instant-speed cast actions."""
        return self.cast_sorcery_start + self.max_hand_slots

    @property
    def activate_start(self) -> int:
        """Start index for activated-ability actions."""
        return self.cast_instant_start + self.max_hand_slots

    @property
    def attack_toggle_start(self) -> int:
        """Start index for attack-toggle actions."""
        return self.activate_start + self.max_permanent_slots

    @property
    def block_attacker_start(self) -> int:
        """Start index for block-attacker selection actions."""
        return self.attack_toggle_start + self.max_creature_slots

    @property
    def block_blocker_start(self) -> int:
        """Start index for block-blocker selection actions."""
        return self.block_attacker_start + self.max_creature_slots

    @property
    def target_start(self) -> int:
        """Start index for spell-target selection actions."""
        return self.block_blocker_start + self.max_creature_slots

    @property
    def mana_source_start(self) -> int:
        """Start index for mana-source tapping actions."""
        return self.target_start + self.max_target_slots

    @property
    def total_size(self) -> int:
        """Total number of discrete actions in the action space."""
        return self.mana_source_start + self.max_permanent_slots

    def action_kind(self, action_idx: int) -> ActionKind | None:
        """Map an action index to its ActionKind, or None if invalid."""
        if action_idx == self.pass_idx:
            return ActionKind.PASS
        if action_idx == self.keep_idx:
            return ActionKind.KEEP_HAND
        if action_idx == self.mulligan_idx:
            return ActionKind.MULLIGAN
        if action_idx == self.confirm_idx:
            return ActionKind.CONFIRM
        if action_idx == self.cancel_idx:
            return ActionKind.CANCEL
        if action_idx == self.auto_pay_idx:
            return ActionKind.AUTO_PAY

        if self.bottom_start <= action_idx < self.discard_start:
            return ActionKind.BOTTOM_CARD
        if self.discard_start <= action_idx < self.play_land_start:
            return ActionKind.DISCARD_CARD
        if self.play_land_start <= action_idx < self.cast_sorcery_start:
            return ActionKind.PLAY_LAND
        if self.cast_sorcery_start <= action_idx < self.cast_instant_start:
            return ActionKind.CAST_SORCERY
        if self.cast_instant_start <= action_idx < self.activate_start:
            return ActionKind.CAST_INSTANT
        if self.activate_start <= action_idx < self.attack_toggle_start:
            return ActionKind.ACTIVATE
        if self.attack_toggle_start <= action_idx < self.block_attacker_start:
            return ActionKind.ATTACK_TOGGLE
        if self.block_attacker_start <= action_idx < self.block_blocker_start:
            return ActionKind.BLOCK_SELECT_ATTACKER
        if self.block_blocker_start <= action_idx < self.target_start:
            return ActionKind.BLOCK_SELECT_BLOCKER
        if self.target_start <= action_idx < self.mana_source_start:
            return ActionKind.TARGET
        if self.mana_source_start <= action_idx < self.total_size:
            return ActionKind.MANA_SOURCE
        return None

    def action_slot(self, action_idx: int) -> int:
        """Return the slot offset within its action kind for the given index."""
        kind = self.action_kind(action_idx)
        if kind == ActionKind.BOTTOM_CARD:
            return action_idx - self.bottom_start
        if kind == ActionKind.DISCARD_CARD:
            return action_idx - self.discard_start
        if kind == ActionKind.PLAY_LAND:
            return action_idx - self.play_land_start
        if kind == ActionKind.CAST_SORCERY:
            return action_idx - self.cast_sorcery_start
        if kind == ActionKind.CAST_INSTANT:
            return action_idx - self.cast_instant_start
        if kind == ActionKind.ACTIVATE:
            return action_idx - self.activate_start
        if kind == ActionKind.ATTACK_TOGGLE:
            return action_idx - self.attack_toggle_start
        if kind == ActionKind.BLOCK_SELECT_ATTACKER:
            return action_idx - self.block_attacker_start
        if kind == ActionKind.BLOCK_SELECT_BLOCKER:
            return action_idx - self.block_blocker_start
        if kind == ActionKind.TARGET:
            return action_idx - self.target_start
        if kind == ActionKind.MANA_SOURCE:
            return action_idx - self.mana_source_start
        return -1


class ActionMaskBuilder:
    """Builds action masks for legal action filtering with priority support."""

    def __init__(
        self,
        config: ActionSpaceConfig | None = None,
        rules_engine: RulesEngine | None = None,
    ):
        self.config = config or ActionSpaceConfig()
        self.rules_engine = rules_engine or RulesEngine()
        self.index_map = ActionIndexMap(
            max_hand_slots=self.config.max_hand_slots,
            max_creature_slots=self.config.max_creature_slots,
            max_permanent_slots=self.config.max_permanent_slots,
            max_target_slots=self.config.max_target_slots,
        )

    def get_action_space_size(self) -> int:
        """Return the total number of discrete actions."""
        return self.index_map.total_size

    def _get_instant_speed_cards(self, hand: list[Card]) -> list[tuple[int, Card]]:
        result = []
        for i, card in enumerate(hand):
            if card.card_type == CardType.INSTANT or Keyword.FLASH in card.keywords:
                result.append((i, card))
        return result

    def _get_sorcery_speed_cards(self, hand: list[Card]) -> list[tuple[int, Card]]:
        result = []
        for i, card in enumerate(hand):
            if card.card_type == CardType.LAND:
                continue
            if card.card_type == CardType.INSTANT or Keyword.FLASH in card.keywords:
                continue
            result.append((i, card))
        return result

    def _ensure_non_empty(self, mask: np.ndarray) -> np.ndarray:
        """Ensure mask has at least one legal action (fallback to pass)."""
        if not mask.any():
            mask[self.index_map.pass_idx] = 1
        return mask

    def build_action_mask(self, state: GameState, player_id: int = 0) -> np.ndarray:
        """Build a binary mask of legal actions for the current game state."""
        mask = np.zeros(self.index_map.total_size, dtype=np.int8)
        player = state.players[player_id]

        if state.game_over:
            return self._ensure_non_empty(mask)

        if state.pending_action_type:
            return self._ensure_non_empty(self._build_pending_mask(state, player_id))

        has_priority = state.priority_player == player_id
        if not has_priority:
            return self._ensure_non_empty(mask)

        if state.phase == GamePhase.MULLIGAN:
            mask[self.index_map.keep_idx] = 1
            # Allow mulligan up to 3 times (London Mulligan; keep at least 4 cards)
            if state.mulligan_count[player_id] < 3:
                mask[self.index_map.mulligan_idx] = 1
            return self._ensure_non_empty(mask)

        if state.phase in PRIORITY_PHASES:
            mask[self.index_map.pass_idx] = 1

        if state.phase in PRIORITY_PHASES:
            instant_cards = self._get_instant_speed_cards(player.hand)
            for slot_idx, (_hand_idx, card) in enumerate(
                instant_cards[: self.config.max_hand_slots]
            ):
                if self.rules_engine.can_cast_spell(state, card, player_id):
                    mask[self.index_map.cast_instant_start + slot_idx] = 1

        is_active_player = state.active_player == player_id
        can_sorcery = (
            is_active_player and state.phase in SORCERY_SPEED_PHASES and len(state.stack) == 0
        )

        if can_sorcery:
            lands_in_hand = [
                (i, c) for i, c in enumerate(player.hand) if c.card_type == CardType.LAND
            ]
            for slot_idx, (_hand_idx, card) in enumerate(
                lands_in_hand[: self.config.max_hand_slots]
            ):
                if self.rules_engine.can_play_land(state, card):
                    mask[self.index_map.play_land_start + slot_idx] = 1

            sorcery_cards = self._get_sorcery_speed_cards(player.hand)
            for slot_idx, (_hand_idx, card) in enumerate(
                sorcery_cards[: self.config.max_hand_slots]
            ):
                if self.rules_engine.can_cast_spell(state, card, player_id):
                    mask[self.index_map.cast_sorcery_start + slot_idx] = 1

        if state.phase in PRIORITY_PHASES:
            activatables = [
                c for c in player.battlefield if c.land_props and c.land_props.has_activation
            ]
            for slot_idx, card in enumerate(activatables[: self.config.max_permanent_slots]):
                if self.rules_engine.can_activate_ability(state, card, player_id):
                    mask[self.index_map.activate_start + slot_idx] = 1

        if is_active_player and state.phase == GamePhase.COMBAT_BEGIN and not state.stack:
            attackers = [c for c in player.battlefield if player.can_attack_with(c)]
            if attackers:
                mask[self.index_map.confirm_idx] = 1
                mask[self.index_map.cancel_idx] = 1
                for slot_idx, _creature in enumerate(attackers[: self.config.max_creature_slots]):
                    mask[self.index_map.attack_toggle_start + slot_idx] = 1

        if not is_active_player and state.phase == GamePhase.COMBAT_ATTACKERS and not state.stack:
            opponent = state.get_active_player()
            if opponent.declared_attackers:
                mask[self.index_map.confirm_idx] = 1
                mask[self.index_map.cancel_idx] = 1
                attackers = [opponent.get_card_by_id(cid) for cid in opponent.declared_attackers]
                for slot_idx, _creature in enumerate(
                    [c for c in attackers if c][: self.config.max_creature_slots]
                ):
                    mask[self.index_map.block_attacker_start + slot_idx] = 1
                blockers = [c for c in player.battlefield if player.can_block_with(c)]
                for slot_idx, _creature in enumerate(blockers[: self.config.max_creature_slots]):
                    mask[self.index_map.block_blocker_start + slot_idx] = 1

        return self._ensure_non_empty(mask)

    def _build_pending_mask(self, state: GameState, player_id: int) -> np.ndarray:
        mask = np.zeros(self.index_map.total_size, dtype=np.int8)
        player = state.players[player_id]
        pending = state.pending_action_type

        if pending in {"mulligan_bottom", "discard"}:
            mask[self.index_map.confirm_idx] = 1
            mask[self.index_map.cancel_idx] = 1
            start = (
                self.index_map.bottom_start
                if pending == "mulligan_bottom"
                else self.index_map.discard_start
            )
            for slot_idx, _ in enumerate(player.hand[: self.config.max_hand_slots]):
                mask[start + slot_idx] = 1
            return mask

        if pending == "spell_target":
            mask[self.index_map.cancel_idx] = 1
            for slot_idx, _ in enumerate(
                state.pending_target_candidates[: self.config.max_target_slots]
            ):
                mask[self.index_map.target_start + slot_idx] = 1
            return mask

        if pending == "attack":
            mask[self.index_map.confirm_idx] = 1
            mask[self.index_map.cancel_idx] = 1
            attackers = [c for c in player.battlefield if player.can_attack_with(c)]
            for slot_idx, _ in enumerate(attackers[: self.config.max_creature_slots]):
                mask[self.index_map.attack_toggle_start + slot_idx] = 1
            return mask

        if pending == "block":
            mask[self.index_map.confirm_idx] = 1
            mask[self.index_map.cancel_idx] = 1
            opponent = state.get_active_player()
            attackers = [opponent.get_card_by_id(cid) for cid in opponent.declared_attackers]
            for slot_idx, _creature in enumerate(
                [c for c in attackers if c][: self.config.max_creature_slots]
            ):
                mask[self.index_map.block_attacker_start + slot_idx] = 1
            blockers = [c for c in player.battlefield if player.can_block_with(c)]
            for slot_idx, _ in enumerate(blockers[: self.config.max_creature_slots]):
                mask[self.index_map.block_blocker_start + slot_idx] = 1
            return mask

        if pending == "mana_payment":
            mask[self.index_map.confirm_idx] = 1
            mask[self.index_map.cancel_idx] = 1
            mask[self.index_map.auto_pay_idx] = 1
            for slot_idx, _ in enumerate(
                state.pending_mana_sources[: self.config.max_permanent_slots]
            ):
                mask[self.index_map.mana_source_start + slot_idx] = 1
            return mask

        return mask

    def decode_action(
        self,
        action_idx: int,
        state: GameState,
        player_id: int = 0,
    ) -> Action:
        """Decode an integer action index into an Action namedtuple."""
        kind = self.index_map.action_kind(action_idx)
        if kind is None:
            return Action(ActionKind.PASS)
        slot = self.index_map.action_slot(action_idx)
        return Action(kind, slot)

    def get_legal_actions(self, state: GameState, player_id: int = 0) -> list[int]:
        """Return list of legal action indices for the given game state."""
        mask = self.build_action_mask(state, player_id)
        return list(np.where(mask == 1)[0])

    def action_to_string(
        self,
        action_idx: int,
        state: GameState,
        player_id: int = 0,
    ) -> str:
        """Convert an action index to a human-readable description string."""
        action = self.decode_action(action_idx, state, player_id)
        player = state.players[player_id]
        kind = action.kind
        slot = action.slot

        if kind == ActionKind.PASS:
            return "Pass priority"
        if kind == ActionKind.KEEP_HAND:
            return "Keep hand"
        if kind == ActionKind.MULLIGAN:
            return "Mulligan"
        if kind == ActionKind.CONFIRM:
            return "Confirm selection"
        if kind == ActionKind.CANCEL:
            return "Cancel selection"
        if kind == ActionKind.AUTO_PAY:
            return "Auto-pay mana"

        if kind == ActionKind.BOTTOM_CARD:
            if slot < len(player.hand):
                return f"Bottom card: {player.hand[slot].name}"
            return "Bottom card"
        if kind == ActionKind.DISCARD_CARD:
            if slot < len(player.hand):
                return f"Discard card: {player.hand[slot].name}"
            return "Discard card"

        if kind == ActionKind.PLAY_LAND:
            lands = [c for c in player.hand if c.card_type == CardType.LAND]
            if slot < len(lands):
                return f"Play land: {lands[slot].name}"
            return "Play land"
        if kind == ActionKind.CAST_SORCERY:
            cards = self._get_sorcery_speed_cards(player.hand)
            if slot < len(cards):
                return f"Cast: {cards[slot][1].name}"
            return "Cast spell"
        if kind == ActionKind.CAST_INSTANT:
            cards = self._get_instant_speed_cards(player.hand)
            if slot < len(cards):
                return f"Cast (instant): {cards[slot][1].name}"
            return "Cast instant"
        if kind == ActionKind.ACTIVATE:
            activatables = [
                c for c in player.battlefield if c.land_props and c.land_props.has_activation
            ]
            if slot < len(activatables):
                return f"Activate: {activatables[slot].name}"
            return "Activate ability"
        if kind == ActionKind.ATTACK_TOGGLE:
            attackers = [c for c in player.battlefield if player.can_attack_with(c)]
            if slot < len(attackers):
                return f"Toggle attacker: {attackers[slot].name}"
            return "Toggle attacker"
        if kind == ActionKind.BLOCK_SELECT_ATTACKER:
            opponent = state.get_active_player()
            attackers = [opponent.get_card_by_id(cid) for cid in opponent.declared_attackers]
            attackers = [c for c in attackers if c]
            if slot < len(attackers):
                return f"Select attacker: {attackers[slot].name}"
            return "Select attacker"
        if kind == ActionKind.BLOCK_SELECT_BLOCKER:
            blockers = [c for c in player.battlefield if player.can_block_with(c)]
            if slot < len(blockers):
                return f"Select blocker: {blockers[slot].name}"
            return "Select blocker"
        if kind == ActionKind.TARGET:
            if slot < len(state.pending_target_candidates):
                target = state.pending_target_candidates[slot]
                return f"Select target: {target.name}"
            return "Select target"
        if kind == ActionKind.MANA_SOURCE:
            if slot < len(state.pending_mana_sources):
                card = state.get_priority_player().get_card_by_id(state.pending_mana_sources[slot])
                if card:
                    return f"Tap for mana: {card.name}"
            return "Tap mana source"

        return f"Unknown action: {action_idx}"

    def get_action_names(self, state: GameState, player_id: int = 0) -> dict[int, str]:
        """Return a mapping of all action indices to their string names."""
        return {
            i: self.action_to_string(i, state, player_id) for i in range(self.index_map.total_size)
        }

    def get_action_metadata(self, state: GameState) -> dict[int, dict[str, Any]]:
        """Return metadata for every action index.

        Includes card properties (is_removal, deals_damage, etc.) so the
        causal agent can map actions to SCM interventions without brittle
        card-name string matching.

        Uses the SAME filtered card lists as ``build_action_mask`` and
        ``action_to_string`` so that slot indices map to the correct card.
        """
        player = state.players[0]

        lands = [c for c in player.hand if c.card_type == CardType.LAND]
        sorcery_cards = self._get_sorcery_speed_cards(player.hand)
        instant_cards = self._get_instant_speed_cards(player.hand)
        activatables = [
            c for c in player.battlefield if c.land_props and c.land_props.has_activation
        ]
        attackers = [c for c in player.battlefield if player.can_attack_with(c)]

        metadata: dict[int, dict[str, Any]] = {}
        for i in range(self.index_map.total_size):
            kind = self.index_map.action_kind(i)
            slot = self.index_map.action_slot(i)
            if kind is None:
                continue
            entry: dict[str, Any] = {"kind": kind.value, "slot": slot}

            card = None
            if kind == ActionKind.PLAY_LAND:
                if 0 <= slot < len(lands):
                    card = lands[slot]
            elif kind == ActionKind.CAST_SORCERY:
                if 0 <= slot < len(sorcery_cards):
                    card = sorcery_cards[slot][1]
            elif kind == ActionKind.CAST_INSTANT:
                if 0 <= slot < len(instant_cards):
                    card = instant_cards[slot][1]
            elif kind == ActionKind.ACTIVATE:
                if 0 <= slot < len(activatables):
                    card = activatables[slot]
            elif kind == ActionKind.ATTACK_TOGGLE and 0 <= slot < len(attackers):
                card = attackers[slot]

            if card is not None:
                entry["card_name"] = card.name
                entry["card_type"] = card.card_type.value
                entry["action_type"] = (
                    f"cast_{card.card_type.value}"
                    if kind in (ActionKind.CAST_SORCERY, ActionKind.CAST_INSTANT)
                    else kind.value
                )
                entry["power"] = card.power
                entry["toughness"] = card.toughness
                entry["is_removal"] = card.is_removal
                entry["is_counterspell"] = card.is_counterspell
                entry["draws_cards"] = card.draws_cards
                entry["deals_damage"] = card.deals_damage
            else:
                entry["action_type"] = kind.value

            metadata[i] = entry
        return metadata

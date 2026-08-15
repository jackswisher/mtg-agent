"""Observation space builder for the MTG environment.

This module handles the construction of observations from game state,
implementing partial observability (agent cannot see opponent's hand/deck).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from mtg.env.card_definitions import Card, CardRegistry, CardType, Keyword
from mtg.env.rules import GameState


@dataclass
class ObservationConfig:
    """Configuration for observation space."""

    # Maximum counts for fixed-size observations
    max_hand_size: int = 10
    max_battlefield_size: int = 20
    max_graveyard_size: int = 20
    max_deck_cards: int = 40

    # Card encoding dimensions (7 type + 10 base + 12 keywords/role/state + 5 color pips = 34)
    card_embedding_dim: int = 34

    # Future work: encode recent actions into the observation for improved
    # partial-observability handling (e.g. predicting opponent strategy).
    # Not yet wired into build_observation / get_observation_space_shape.
    include_action_history: bool = False
    action_history_length: int = 10


class ObservationBuilder:
    """Builds observations from game state."""

    def __init__(self, config: ObservationConfig | None = None):
        """Initialize the observation builder.

        Args:
            config: Observation configuration.
        """
        self.config = config or ObservationConfig()
        self._card_to_idx = self._build_card_index()

    def get_observation_space_shape(self) -> dict[str, tuple[int, ...]]:
        """Get the shape of the observation space components."""
        cfg = self.config
        return {
            "game_state": (17,),
            "hand": (cfg.max_hand_size, cfg.card_embedding_dim),
            "battlefield_self": (cfg.max_battlefield_size, cfg.card_embedding_dim),
            "battlefield_opponent": (cfg.max_battlefield_size, cfg.card_embedding_dim),
            "graveyard_self": (cfg.max_graveyard_size, cfg.card_embedding_dim),
            "graveyard_opponent": (cfg.max_graveyard_size, cfg.card_embedding_dim),
        }

    def get_flat_observation_dim(self) -> int:
        """Get the total dimension of a flattened observation."""
        shapes = self.get_observation_space_shape()
        total = 0
        for shape in shapes.values():
            dim = 1
            for s in shape:
                dim *= s
            total += dim
        return total

    def build_observation(self, state: GameState, player_id: int = 0) -> dict[str, np.ndarray]:
        """Build an observation dict from game state.

        Args:
            state: The current game state.
            player_id: The player perspective (0 for agent).

        Returns:
            Dictionary of observation arrays.
        """
        player = state.players[player_id]
        opponent = state.players[1 - player_id]

        obs = {
            "game_state": self._encode_game_state(state, player_id),
            "hand": self._encode_card_list(player.hand, self.config.max_hand_size),
            "battlefield_self": self._encode_card_list(
                player.battlefield, self.config.max_battlefield_size, player=player
            ),
            "battlefield_opponent": self._encode_card_list(
                opponent.battlefield, self.config.max_battlefield_size, player=opponent
            ),
            "graveyard_self": self._encode_card_list(
                player.graveyard, self.config.max_graveyard_size
            ),
            "graveyard_opponent": self._encode_card_list(
                opponent.graveyard, self.config.max_graveyard_size
            ),
        }

        return obs

    def build_flat_observation(self, state: GameState, player_id: int = 0) -> np.ndarray:
        """Build a flat observation array.

        Args:
            state: The current game state.
            player_id: The player perspective (0 for agent).

        Returns:
            Flattened observation array.
        """
        obs_dict = self.build_observation(state, player_id)

        # Concatenate all arrays in a consistent order
        arrays = [
            obs_dict["game_state"].flatten(),
            obs_dict["hand"].flatten(),
            obs_dict["battlefield_self"].flatten(),
            obs_dict["battlefield_opponent"].flatten(),
            obs_dict["graveyard_self"].flatten(),
            obs_dict["graveyard_opponent"].flatten(),
        ]

        return np.concatenate(arrays).astype(np.float32)

    def _encode_game_state(self, state: GameState, player_id: int) -> np.ndarray:
        """Encode scalar game state features."""
        player = state.players[player_id]
        opponent = state.players[1 - player_id]

        own_attackers = sum(
            1
            for c in player.battlefield
            if c.card_type == CardType.CREATURE and player.can_attack_with(c)
        )
        own_blockers = sum(
            1
            for c in player.battlefield
            if c.card_type == CardType.CREATURE and player.can_block_with(c)
        )
        opp_attackers = sum(
            1
            for c in opponent.battlefield
            if c.card_type == CardType.CREATURE and opponent.can_attack_with(c)
        )
        opp_blockers = sum(
            1
            for c in opponent.battlefield
            if c.card_type == CardType.CREATURE and opponent.can_block_with(c)
        )

        features = np.array(
            [
                player.life / 20.0,
                opponent.life / 20.0,
                state.turn_number / state.max_turns,
                float(state.phase.value) / 10.0,
                len(player.hand) / 10.0,
                len(player.deck) / 40.0,
                player.get_total_available_mana() / 10.0,
                float(state.active_player == player_id),
                opponent.get_total_available_mana() / 10.0,
                len(opponent.hand) / 10.0,
                len(state.stack) / 5.0,
                float(state.pending_action_type is not None),
                own_attackers / 10.0,
                own_blockers / 10.0,
                opp_attackers / 10.0,
                opp_blockers / 10.0,
                len(opponent.declared_attackers) / 10.0,
            ],
            dtype=np.float32,
        )

        return features

    def _encode_card_list(
        self,
        cards: list[Card],
        max_size: int,
        player: Any = None,
    ) -> np.ndarray:
        """Encode a list of cards into a fixed-size array."""
        cfg = self.config
        encoded = np.zeros((max_size, cfg.card_embedding_dim), dtype=np.float32)

        for i, card in enumerate(cards[:max_size]):
            encoded[i] = self._encode_card(card, player)

        return encoded

    def _encode_card(self, card: Card, player: Any = None) -> np.ndarray:
        """Encode a single card.

        Args:
            card: The card to encode.
            player: Optional PlayerState for runtime status (tapped,
                summoning sickness). Pass for battlefield cards.
        """
        cfg = self.config
        features = np.zeros(cfg.card_embedding_dim, dtype=np.float32)

        # Card type one-hot (7 types: land, creature, instant, sorcery,
        # enchantment, artifact, planeswalker)
        type_idx = list(CardType).index(card.card_type)
        features[type_idx] = 1.0

        # Mana cost (normalized)
        features[7] = card.mana_cost.cmc / 10.0

        # Power/toughness (effective, with buffs/counters/tokens)
        if card.card_type == CardType.CREATURE:
            features[8] = card.effective_power / 10.0
            features[9] = card.effective_toughness / 10.0

        # Boolean abilities
        features[10] = float(card.has_haste)
        features[11] = float(card.has_flash)
        features[12] = float(len(card.produces_mana) > 0)

        # Effects (normalized)
        features[13] = card.draws_cards / 5.0
        features[14] = card.deals_damage / 5.0
        features[15] = card.gains_life / 5.0

        # Card identity (deterministic)
        features[16] = self._get_card_idx(card.name) / 100.0

        # Combat keyword abilities
        features[17] = float(card.has_flying)
        features[18] = float(card.has_trample)
        features[19] = float(card.has_deathtouch)
        features[20] = float(card.has_lifelink)
        features[21] = float(card.has_first_strike)
        features[22] = float(card.has_menace)
        features[23] = float(card.has_vigilance)
        features[24] = float(Keyword.PROWESS in card.keywords)

        # Card role flags
        features[25] = float(card.is_removal)
        features[26] = float(card.is_counterspell)

        # Runtime state (only meaningful for battlefield cards)
        if player is not None:
            features[27] = float(card.card_id in player.tapped_permanents)
            features[28] = float(card.card_id in player.summoning_sick)

        # Color mana costs (normalized by max expected pips)
        features[29] = card.mana_cost.white / 5.0
        features[30] = card.mana_cost.blue / 5.0
        features[31] = card.mana_cost.black / 5.0
        features[32] = card.mana_cost.red / 5.0
        features[33] = card.mana_cost.green / 5.0

        return features

    @staticmethod
    def _build_card_index() -> dict[str, int]:
        """Build a deterministic card-name -> index mapping from the registry.

        Sorted alphabetically so the encoding is stable across episodes
        and across different environment instances.
        """
        registry = CardRegistry.get_instance()
        names = sorted(registry.get_all().keys())
        return {name: idx + 1 for idx, name in enumerate(names)}

    def _get_card_idx(self, card_name: str) -> int:
        """Get the deterministic index for a card name."""
        return self._card_to_idx.get(card_name, 0)

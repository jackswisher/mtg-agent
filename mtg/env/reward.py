"""Reward calculation for the MTG environment.

This module implements various reward shaping strategies aligned with
the causal variables defined in the SCM.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from mtg.env.card_definitions import CardType
from mtg.env.rules import GamePhase, GameState


class RewardType(Enum):
    """Type of reward signal."""

    SPARSE = "sparse"  # Only win/loss at end
    SHAPED = "shaped"  # Intermediate shaping based on causal variables
    DENSE = "dense"  # Per-step rewards for all relevant events


@dataclass
class RewardConfig:
    """Configuration for reward calculation."""

    reward_type: RewardType = RewardType.SHAPED

    # Terminal rewards
    win_reward: float = 1.0
    loss_reward: float = -1.0
    draw_reward: float = -0.3  # Slight penalty to encourage trying to win

    # Per-step penalty to discourage stalling (passing priority repeatedly).
    # Small enough not to dominate, but enough to make shorter wins preferred.
    step_penalty: float = -0.005

    # Shaped rewards (per causal variable contribution)
    mana_development_weight: float = 0.1
    card_advantage_weight: float = 0.1
    board_pressure_weight: float = 0.1
    life_differential_weight: float = 0.1
    tempo_weight: float = 0.05

    # Discount factor for potential-based shaping (should match PPO gamma)
    gamma: float = 0.995

    # Dense rewards
    damage_dealt_weight: float = 0.02
    card_drawn_weight: float = 0.01
    creature_played_weight: float = 0.01


class RewardCalculator:
    """Calculates rewards from game state transitions."""

    def __init__(self, config: RewardConfig | None = None):
        """Initialize the reward calculator.

        Args:
            config: Reward configuration.
        """
        self.config = config or RewardConfig()
        self._prev_state_cache: dict[str, float] = {}

    def calculate_reward(
        self,
        prev_state: GameState,
        action: int,
        next_state: GameState,
        player_id: int = 0,
    ) -> float:
        """Calculate reward for a state transition.

        Args:
            prev_state: State before action.
            action: Action taken.
            next_state: State after action.
            player_id: Player perspective.

        Returns:
            Reward value.
        """
        cfg = self.config

        # Terminal reward
        if next_state.game_over:
            if next_state.winner == player_id:
                return cfg.win_reward
            elif next_state.winner is not None:
                return cfg.loss_reward
            else:
                return cfg.draw_reward

        # NOTE: The environment can present PASS-only steps (for
        # example, combat priority when no instants are available).
        # These mandatory steps still receive shaped reward. The
        # action-mask ensures the agent can only choose PASS, so no
        # gradient signal is wasted: MaskablePPO zeroes out log-probs
        # for masked actions and the policy update is a no-op.

        # Non-terminal reward based on reward type
        if cfg.reward_type == RewardType.SPARSE:
            reward = 0.0
        elif cfg.reward_type == RewardType.SHAPED:
            reward = self._calculate_shaped_reward(prev_state, next_state, player_id)
        elif cfg.reward_type == RewardType.DENSE:
            reward = self._calculate_dense_reward(prev_state, next_state, player_id)
        else:
            reward = cfg.step_penalty

        # Penalise passing priority when affordable spells exist.
        # Only fires during the agent's main phases (has priority and
        # can cast sorcery-speed spells).  ``can_pay`` checks mana only,
        # not per-card timing restrictions, so this is still an
        # approximation but avoids penalising forced PASS steps in
        # combat or on the opponent's turn.
        if action == 0 and cfg.reward_type != RewardType.SPARSE:
            is_main = prev_state.phase in (
                GamePhase.MAIN_PRECOMBAT,
                GamePhase.MAIN_POSTCOMBAT,
            )
            has_priority = getattr(prev_state, "priority_player", None) == player_id
            if is_main and has_priority:
                player = prev_state.players[player_id]
                avail = player.get_available_mana()
                has_castable = any(
                    c.mana_cost.can_pay(avail) for c in player.hand if c.card_type != CardType.LAND
                )
                if has_castable:
                    reward += cfg.step_penalty

        return reward

    def _calculate_shaped_reward(
        self,
        prev_state: GameState,
        next_state: GameState,
        player_id: int,
    ) -> float:
        """Calculate potential-based shaped reward.

        Uses gamma * Phi(s') - Phi(s) per Ng et al. 1999 to preserve the
        optimal policy under shaping.
        """
        prev_potential = self._state_potential(prev_state, player_id)
        next_potential = self._state_potential(next_state, player_id)

        return self.config.gamma * next_potential - prev_potential

    def _state_potential(self, state: GameState, player_id: int) -> float:
        """Calculate state potential based on causal variables."""
        cfg = self.config
        player = state.players[player_id]
        opponent = state.players[1 - player_id]

        potential = 0.0

        # Mana development (lands in play)
        mana_count = sum(1 for c in player.battlefield if c.produces_mana)
        potential += cfg.mana_development_weight * mana_count

        # Card advantage (battlefield only; opponent hand is hidden)
        card_count = len(player.battlefield)
        opp_card_count = len(opponent.battlefield)
        potential += cfg.card_advantage_weight * (card_count - opp_card_count)

        # Board pressure (effective power accounts for buffs, counters, tokens)
        total_power = sum(
            c.effective_power for c in player.battlefield if c.card_type == CardType.CREATURE
        )
        opp_power = sum(
            c.effective_power for c in opponent.battlefield if c.card_type == CardType.CREATURE
        )
        potential += cfg.board_pressure_weight * (total_power - opp_power)

        # Life differential (clamped to [-20, 20] and scaled to ~unit range)
        life_diff = max(-20.0, min(20.0, player.life - opponent.life)) / 20.0
        potential += cfg.life_differential_weight * life_diff

        # Tempo (mana efficiency)
        own_mana_pool = max(1, mana_count)
        own_spent = sum(1 for c in player.battlefield if c.card_type != CardType.LAND)
        potential += cfg.tempo_weight * (own_spent / own_mana_pool)

        return potential

    def _calculate_dense_reward(
        self,
        prev_state: GameState,
        next_state: GameState,
        player_id: int,
    ) -> float:
        """Calculate dense per-event reward."""
        cfg = self.config
        reward = 0.0

        prev_player = prev_state.players[player_id]
        next_player = next_state.players[player_id]
        prev_opp = prev_state.players[1 - player_id]
        next_opp = next_state.players[1 - player_id]

        # Damage dealt
        damage_dealt = prev_opp.life - next_opp.life
        if damage_dealt > 0:
            reward += cfg.damage_dealt_weight * damage_dealt

        # Cards drawn (hand size increase accounting for plays)
        cards_drawn = len(next_player.hand) - len(prev_player.hand)
        cards_played = len(next_player.battlefield) - len(prev_player.battlefield)
        net_cards = cards_drawn + max(0, cards_played)
        if net_cards > 0:
            reward += cfg.card_drawn_weight * net_cards

        # Creatures played
        prev_creatures = sum(1 for c in prev_player.battlefield if c.card_type == CardType.CREATURE)
        next_creatures = sum(1 for c in next_player.battlefield if c.card_type == CardType.CREATURE)
        creatures_added = next_creatures - prev_creatures
        if creatures_added > 0:
            reward += cfg.creature_played_weight * creatures_added

        # Also include shaped component
        reward += 0.5 * self._calculate_shaped_reward(prev_state, next_state, player_id)

        return reward

    def get_causal_variable_values(self, state: GameState, player_id: int = 0) -> dict[str, float]:
        """Extract causal variable values from game state.

        These correspond to the SCM variables defined in the paper:
        - Mana: Available mana production
        - CardAdv: Card advantage
        - BoardPress: Board pressure
        - Tempo: Initiative measure
        - LifeBuffer: Life total buffer

        Args:
            state: Current game state.
            player_id: Player perspective.

        Returns:
            Dictionary of causal variable values.
        """
        player = state.players[player_id]
        opponent = state.players[1 - player_id]

        # Mana: Total mana-producing permanents
        mana = sum(1 for c in player.battlefield if c.produces_mana)

        # CardAdv: Board advantage (aligned with shaped reward potential)
        card_adv = len(player.battlefield) - len(opponent.battlefield)

        # BoardPress: Net board power (effective, accounting for buffs/counters)
        own_power = sum(
            c.effective_power for c in player.battlefield if c.card_type == CardType.CREATURE
        )
        opp_power = sum(
            c.effective_power for c in opponent.battlefield if c.card_type == CardType.CREATURE
        )
        board_press = own_power - opp_power

        # Tempo: mana spent ratio differential (own efficiency - opponent efficiency)
        own_mana_pool = max(1, mana)
        own_mana_spent = sum(1 for c in player.battlefield if c.card_type != CardType.LAND)
        opp_mana_pool_raw = sum(1 for c in opponent.battlefield if c.produces_mana)
        opp_mana_pool = max(1, opp_mana_pool_raw)
        opp_mana_spent = sum(1 for c in opponent.battlefield if c.card_type != CardType.LAND)
        tempo = float(
            np.clip(
                own_mana_spent / own_mana_pool - opp_mana_spent / opp_mana_pool,
                -1.0,
                1.0,
            )
        )

        # LifeBuffer: Life differential (player life - opponent life)
        life_buffer = float(player.life - opponent.life)

        # ThreatDensity: fraction of creatures with power >= 2 among own permanents
        total_permanents = max(1, len(player.battlefield))
        threats = sum(
            1 for c in player.battlefield if c.card_type == CardType.CREATURE and c.power >= 2
        )
        threat_density = float(threats / total_permanents)

        # Mana creatures: creatures that produce mana (for SCM Mana_{t+1} equation)
        mana_creatures = sum(
            1 for c in player.battlefield if c.card_type == CardType.CREATURE and c.produces_mana
        )

        # Land drop availability (approximation: player has lands in hand)
        lands_in_hand = sum(1 for c in player.hand if c.card_type == CardType.LAND)
        land_drop = 1.0 if lands_in_hand > 0 else 0.0

        # Removal availability: does the player hold removal or burn?
        removal_avail = float(any(c.is_removal or c.deals_damage > 0 for c in player.hand))

        # Per-side raw mana counts (pre-pool clamping); needed by the
        # SCM ``tempo`` mechanism, which recomputes own/opp efficiencies
        # from these as parents. Without these keys the SCM would fall
        # back to defaults and ``tempo`` would evaluate to 0, silently
        # breaking the calibration signal for that factor.
        return {
            "mana": float(mana),
            "mana_t": float(mana),
            "mana_spent": float(own_mana_spent),
            "opp_mana": float(opp_mana_pool_raw),
            "opp_mana_spent": float(opp_mana_spent),
            "card_advantage": float(card_adv),
            "card_adv": float(card_adv),
            "board_pressure": float(board_press),
            "board_press": float(board_press),
            "tempo": tempo,
            "life_buffer": life_buffer,
            "threat_density": threat_density,
            "own_power": float(own_power),
            "opp_power": float(opp_power),
            "own_life": float(player.life),
            "opp_life": float(opponent.life),
            "mana_creatures": float(mana_creatures),
            "land_drop": land_drop,
            "removal_avail": removal_avail,
            "has_removal": removal_avail,
            "board_presence": float(len(player.battlefield)),
            "opp_board_presence": float(len(opponent.battlefield)),
            "threat_count": float(threats),
            "card_count": float(len(player.hand)),
        }

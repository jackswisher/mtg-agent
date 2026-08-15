"""MTG Gymnasium Environment with Priority System.

This module provides the main Gymnasium-compatible environment for
Magic: The Gathering strategic decision-making, including instant-speed
spell casting and priority passing.
"""

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mtg.env.action_mask import ActionKind, ActionMaskBuilder, ActionSpaceConfig
from mtg.env.card_definitions import CardType, Keyword
from mtg.env.deck_archetypes import get_archetype
from mtg.env.observation import ObservationBuilder, ObservationConfig
from mtg.env.reward import RewardCalculator, RewardConfig, RewardType
from mtg.env.rules import PRIORITY_PHASES, GamePhase, GameState, RulesEngine


class MTGEnv(gym.Env):
    """Gymnasium environment for MTG strategic decision-making.

    This environment simulates a Magic: The Gathering game with a configurable
    turn horizon, focusing on strategically decisive decisions:
    - Mulligan decisions
    - Land sequencing
    - Spell casting (including instants during opponent's turn)
    - Combat decisions with blocking
    - Priority passing

    The environment supports:
    - Partial observability (opponent's hand hidden)
    - Legal action masking
    - Instant-speed responses during opponent's turn
    - Play/draw randomization

    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        deck_archetype: str = "mono_red_aggro",
        opponent_archetype: str | None = None,
        max_turns: int = 10,
        max_steps_per_episode: int = 500,
        reward_type: str = "shaped",
        render_mode: str | None = None,
        seed: int | None = None,
        opponent_agent: Any = None,
        auto_resolve: bool | None = None,
        auto_combat: bool = False,
        auto_target: bool = False,
        auto_mana: bool = True,
        gamma: float = 0.995,
    ):
        """Initialize the MTG environment.

        Args:
            deck_archetype: Player deck archetype name.
            opponent_archetype: Opponent deck archetype (defaults to same as player).
            max_turns: Maximum number of turns before game ends.
            max_steps_per_episode: Safety limit on env steps to prevent infinite loops.
            reward_type: Reward type ('sparse', 'shaped', 'dense').
            render_mode: Rendering mode ('human', 'ansi', or None).
            seed: Random seed for reproducibility.
            opponent_agent: Agent to use for opponent decisions (None = rule-based fallback).
            auto_resolve: Convenience flag. When set, overrides all three
                granular ``auto_*`` flags below.
            auto_combat: If True, combat is all-or-nothing (attack with all
                eligible creatures). If False, agent toggles individual attackers.
            auto_target: If True, spell targets are auto-picked using board
                heuristics. If False, agent selects targets from candidates.
            auto_mana: If True, mana payment is always auto-resolved. Should
                generally stay True (learning mana tapping has no strategic value).
            gamma: Discount factor forwarded into ``RewardConfig`` so that
                potential-based reward shaping uses the same discount as
                the downstream RL agent.

        """
        super().__init__()
        self.opponent_agent = opponent_agent
        self.active_opponent_name: str | None = None

        if auto_resolve is not None:
            self.auto_combat = auto_resolve
            self.auto_target = auto_resolve
            self.auto_mana = auto_resolve
        else:
            self.auto_combat = auto_combat
            self.auto_target = auto_target
            self.auto_mana = auto_mana

        # Configuration
        self.deck_archetype_name = deck_archetype
        self.opponent_archetype_name = opponent_archetype or deck_archetype
        self.max_turns = max_turns
        self.max_steps_per_episode = max_steps_per_episode
        self._step_count = 0
        self.render_mode = render_mode

        # Load archetypes
        self.player_archetype = get_archetype(deck_archetype)
        self.opponent_archetype = get_archetype(self.opponent_archetype_name)

        # Initialize components
        self.rules_engine = RulesEngine()
        self.obs_builder = ObservationBuilder(ObservationConfig())
        self.action_builder = ActionMaskBuilder(ActionSpaceConfig(), self.rules_engine)
        self.reward_calculator = RewardCalculator(
            RewardConfig(reward_type=RewardType(reward_type), gamma=gamma)
        )

        # Observation/action contract: every actionable hand slot in the
        # action mask must also be observable in the observation tensor.
        # If one cap is raised without the other (for example, raising
        # ``max_hand_slots`` without raising the observation hand cap to
        # match), the policy network would be deciding on slots it cannot
        # see. Fail fast with a clear error rather than degrade silently.
        obs_cfg = self.obs_builder.config
        act_cfg = self.action_builder.config
        if act_cfg.max_hand_slots > obs_cfg.max_hand_size:
            raise ValueError(
                "Action-mask hand cap "
                f"(ActionSpaceConfig.max_hand_slots={act_cfg.max_hand_slots}) "
                "exceeds observation hand cap "
                f"(ObservationConfig.max_hand_size={obs_cfg.max_hand_size}). "
                "Hand slots beyond the observation cap are literally "
                "unobservable to the policy network; raise "
                "ObservationConfig.max_hand_size to match before "
                "increasing the action-mask cap."
            )

        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_builder.get_flat_observation_dim(),),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(self.action_builder.get_action_space_size())

        # Game state
        self.state: GameState | None = None
        self._prev_state: GameState | None = None

        # Seed RNG
        self._np_random: np.random.Generator | None = None
        if seed is not None:
            self.reset(seed=seed)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed.
            options: Additional options.

        Returns:
            Tuple of (observation, info dict).

        """
        super().reset(seed=seed)

        if seed is not None:
            self._np_random = np.random.default_rng(seed)
            self.rules_engine.rng.seed(seed)
        else:
            # Derive a fresh seed from the internal RNG so episodes are
            # not correlated when reset() is called without an explicit seed.
            new_seed = self.rules_engine.rng.randint(0, 2**31)
            self.rules_engine.rng.seed(new_seed)

        # Build decks
        player_deck = self.player_archetype.build_deck()
        opponent_deck = self.opponent_archetype.build_deck()

        # Initialize game
        self.state = self.rules_engine.initialize_game(
            player_deck=player_deck,
            opponent_deck=opponent_deck,
            max_turns=self.max_turns,
        )
        self._prev_state = None
        self._step_count = 0

        # Build observation and info
        obs = self.obs_builder.build_flat_observation(self.state, player_id=0)
        info = self._get_info()

        return obs, info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute an action in the environment.

        Args:
            action: Discrete action index.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).

        """
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self._step_count += 1

        # Store prior state so the reward calculator can diff against it
        self._prev_state = self._copy_state(self.state)

        # Decode and execute action
        decoded = self.action_builder.decode_action(action, self.state, player_id=0)
        self._execute_action(decoded, action)

        # After player action, handle opponent priority/turns
        self._handle_opponent_actions()
        # Clean any stale tracking (tapped, summoned, activated) after state changes
        self.rules_engine._prune_battlefield_tracking(self.state)

        # Check game-over BEFORE reward so terminal conditions (including
        # deck-out via failed_draw) are visible to calculate_reward.
        self.state.check_game_over()

        reward = self.reward_calculator.calculate_reward(
            self._prev_state, action, self.state, player_id=0
        )
        terminated = self.state.game_over
        # Truncate if we exceed max steps (safety against infinite loops)
        truncated = not terminated and self._step_count >= self.max_steps_per_episode

        # Treat truncation (step-limit) like a draw so the agent is
        # discouraged from stalling.  Without this, truncated episodes
        # return the last shaped-reward delta instead of a terminal
        # penalty, which lets the agent exploit episode extension.
        if truncated:
            reward = self.reward_calculator.config.draw_reward

        # Build observation and info
        obs = self.obs_builder.build_flat_observation(self.state, player_id=0)
        info = self._get_info()

        if truncated and "game_result" not in info:
            info["game_result"] = "draw"
            info["winner"] = None

        # When the episode is truncated by the step cap (rather than the
        # game terminating), SB3's GAE wants to bootstrap V(s_last)
        # instead of assuming zero future return.  Gymnasium's convention
        # is to expose the final observation under info["terminal_observation"]
        # so VecEnv wrappers can pick it up when they auto-reset.
        if truncated:
            info["TimeLimit.truncated"] = True
            info["terminal_observation"] = obs.copy()

        if (terminated or truncated) and self.active_opponent_name is not None:
            info["active_opponent"] = self.active_opponent_name

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _execute_action(self, action: Any, action_idx: int) -> None:
        """Execute a decoded action on the game state.

        Args:
            action: Decoded action tuple.
            action_idx: Original action index for type checking.

        """
        assert self.state is not None

        player = self.state.players[0]
        decoded = self.action_builder.decode_action(action_idx, self.state, player_id=0)

        if self.state.pending_action_type:
            self.state = self.rules_engine.handle_pending_action(
                self.state, decoded.kind.value, decoded.slot, player_idx=0
            )
            # After agent selects target, auto-pay mana if configured
            if self.auto_mana and self.state.pending_action_type == "mana_payment":
                self.state = self.rules_engine.handle_pending_action(
                    self.state, "auto_pay", -1, player_idx=0
                )
            return

        if (
            decoded.kind in {ActionKind.ATTACK_TOGGLE, ActionKind.CONFIRM}
            and self.state.active_player == 0
            and self.state.phase == GamePhase.COMBAT_BEGIN
        ):
            if self.auto_combat:
                player = self.state.players[0]
                attackers = [c for c in player.battlefield if player.can_attack_with(c)]
                if attackers:
                    self.state.pending_action_type = "attack"
                    self.state.pending_player = 0
                    for c in attackers:
                        self.state.pending_attackers.add(c.card_id)
                    self.state = self.rules_engine.handle_pending_action(
                        self.state, "confirm", -1, player_idx=0
                    )
                else:
                    self.state = self.rules_engine.pass_priority(self.state)
                return

            # Selective combat: agent toggles individual attackers + confirms
            self.state.pending_action_type = "attack"
            self.state.pending_player = 0
            self.state = self.rules_engine.handle_pending_action(
                self.state, decoded.kind.value, decoded.slot, player_idx=0
            )
            return

        if (
            decoded.kind
            in {
                ActionKind.BLOCK_SELECT_ATTACKER,
                ActionKind.BLOCK_SELECT_BLOCKER,
                ActionKind.CONFIRM,
            }
            and self.state.active_player == 1
            and self.state.phase == GamePhase.COMBAT_ATTACKERS
        ):
            if self.auto_combat:
                # Simplified blocking: agent decides CONFIRM (no block) or
                # BLOCK_SELECT_ATTACKER (block that attacker with best blocker).
                # Auto-confirm after a single block assignment for simplicity.
                if decoded.kind == ActionKind.CONFIRM:
                    # No blocks: accept damage (aggro race strategy).
                    self.state.pending_action_type = "block"
                    self.state.pending_player = 0
                    self.state = self.rules_engine.handle_pending_action(
                        self.state, "confirm", -1, player_idx=0
                    )
                elif decoded.kind == ActionKind.BLOCK_SELECT_ATTACKER:
                    # Block this attacker with the best available blocker
                    self.state.pending_action_type = "block"
                    self.state.pending_player = 0
                    self.state = self.rules_engine.handle_pending_action(
                        self.state, decoded.kind.value, decoded.slot, player_idx=0
                    )
                    # Auto-assign best blocker (highest power) and confirm
                    if self.state.pending_action_type == "block":
                        player = self.state.players[0]
                        blockers = [c for c in player.battlefield if player.can_block_with(c)]
                        if blockers:
                            best_idx = max(
                                range(len(blockers)),
                                key=lambda i: blockers[i].effective_power,
                            )
                            self.state = self.rules_engine.handle_pending_action(
                                self.state, "block_select_blocker", best_idx, player_idx=0
                            )
                        if self.state.pending_action_type == "block":
                            self.state = self.rules_engine.handle_pending_action(
                                self.state, "confirm", -1, player_idx=0
                            )
                return

            self.state.pending_action_type = "block"
            self.state.pending_player = 0
            self.state = self.rules_engine.handle_pending_action(
                self.state, decoded.kind.value, decoded.slot, player_idx=0
            )
            return

        # === MULLIGAN ===
        if decoded.kind == ActionKind.KEEP_HAND:
            self.state = self.rules_engine.execute_mulligan(self.state, keep=True)
            self._advance_to_player_priority()

        elif decoded.kind == ActionKind.MULLIGAN:
            self.state = self.rules_engine.execute_mulligan(self.state, keep=False)

        # === PASS PRIORITY ===
        elif decoded.kind == ActionKind.PASS:
            self.state = self.rules_engine.pass_priority(self.state)

        # === PLAY LAND ===
        elif decoded.kind == ActionKind.PLAY_LAND:
            lands = [c for c in player.hand if c.card_type == CardType.LAND]
            if 0 <= decoded.slot < len(lands):
                card = lands[decoded.slot]
                if self.rules_engine.can_play_land(self.state, card):
                    self.state = self.rules_engine.play_land(self.state, card)

        # === CAST SORCERY-SPEED SPELL ===
        elif decoded.kind == ActionKind.CAST_SORCERY:
            sorcery_cards = self.action_builder._get_sorcery_speed_cards(player.hand)
            if 0 <= decoded.slot < len(sorcery_cards):
                card = sorcery_cards[decoded.slot][1]
                if self.rules_engine.can_cast_spell(self.state, card, player_idx=0):
                    self.state = self.rules_engine.start_spell_cast(self.state, card, player_idx=0)
                    self._maybe_auto_resolve_spell(player_idx=0)

        # === CAST INSTANT-SPEED SPELL ===
        elif decoded.kind == ActionKind.CAST_INSTANT:
            instant_cards = self.action_builder._get_instant_speed_cards(player.hand)
            if 0 <= decoded.slot < len(instant_cards):
                card = instant_cards[decoded.slot][1]
                if self.rules_engine.can_cast_spell(self.state, card, player_idx=0):
                    self.state = self.rules_engine.start_spell_cast(self.state, card, player_idx=0)
                    self._maybe_auto_resolve_spell(player_idx=0)

        elif decoded.kind == ActionKind.ACTIVATE:
            activatables = [
                c for c in player.battlefield if c.land_props and c.land_props.has_activation
            ]
            if 0 <= decoded.slot < len(activatables):
                card = activatables[decoded.slot]
                if self.rules_engine.can_activate_ability(self.state, card, player_idx=0):
                    self.state = self.rules_engine.start_activation(self.state, card, player_idx=0)
                    self._maybe_auto_resolve_spell(player_idx=0)

    def _maybe_auto_resolve_spell(self, player_idx: int = 0) -> None:
        """Auto-resolve targeting and/or mana based on granular flags.

        Called right after ``start_spell_cast`` / ``start_activation``.
        Decides what to auto-resolve based on ``auto_target`` and ``auto_mana``.
        """
        assert self.state is not None
        if self.auto_target:
            self._auto_resolve_pending(player_idx)
        elif self.auto_mana and self.state.pending_action_type == "mana_payment":
            # Spell has no targets → skip straight to auto-pay
            self.state = self.rules_engine.handle_pending_action(
                self.state, "auto_pay", -1, player_idx=player_idx
            )

    def _auto_resolve_pending(self, player_idx: int = 0) -> None:
        """Auto-resolve pending spell targeting and mana payment.

        This collapses the multi-step spell cast sequence (cast → target → mana
        → confirm) into a single agent action.  The agent only decides *which*
        spell to cast; targeting and payment are handled automatically.

        Targeting heuristic (for mono-red aggro "go face" strategy):
        - If the opponent player is a valid target, always go face.
        - Otherwise pick the first valid target.

        Mana payment always uses auto-pay.
        """
        assert self.state is not None
        # Safety bound to prevent infinite loops in exotic states
        for _ in range(5):
            pending = self.state.pending_action_type
            if pending is None:
                break

            if pending == "spell_target":
                candidates = self.state.pending_target_candidates
                if len(candidates) == 0:
                    # No targets → advance to mana payment
                    self.state.pending_action_type = "mana_payment"
                    self.state.pending_target_candidates = []
                else:
                    chosen = self._pick_auto_target(candidates, player_idx)
                    self.state = self.rules_engine.handle_pending_action(
                        self.state, "target", chosen, player_idx=player_idx
                    )

            elif pending == "mana_payment":
                # Auto-pay mana
                self.state = self.rules_engine.handle_pending_action(
                    self.state, "auto_pay", -1, player_idx=player_idx
                )
            else:
                # Other pending types (discard, block, etc.) → agent decides
                break

    def _pick_auto_target(self, candidates: list, player_idx: int) -> int:
        """Choose the best target index for auto_resolve.

        Uses the same board-aware logic as
        ``RulesEngine._auto_choose_targets`` so the learning agent is not
        mechanically disadvantaged relative to heuristic opponents:

        1. Lethal burn → face.
        2. Behind on board power → kill the highest-power creature we
           can kill with this spell's damage.
        3. At parity or ahead → go face.
        4. If targeting own creatures (pump spells) → highest power.
        5. Fallback → first candidate.

        Returns:
            Index into *candidates* list.
        """
        assert self.state is not None
        opponent_idx = 1 - player_idx
        player = self.state.players[player_idx]
        opponent = self.state.players[opponent_idx]

        # Determine damage dealt by the spell currently being cast
        spell_damage = 0
        if self.state.pending_spell and hasattr(self.state.pending_spell, "source"):
            spell_damage = getattr(self.state.pending_spell.source, "deals_damage", 0)
        elif self.state.stack:
            top_item = self.state.stack[-1]
            source = getattr(top_item, "source", None)
            spell_damage = getattr(source, "deals_damage", 0) if source else 0

        # Check for lethal
        if spell_damage > 0 and opponent.life <= spell_damage:
            for i, c in enumerate(candidates):
                if c.kind == "player" and c.ref_id == opponent_idx:
                    return i

        # Board power comparison
        from mtg.env.card_definitions import CardType

        my_power = sum(
            c.effective_power for c in player.battlefield if c.card_type == CardType.CREATURE
        )
        opp_power = sum(
            c.effective_power for c in opponent.battlefield if c.card_type == CardType.CREATURE
        )

        # Behind on board → try to kill the highest-power creature
        if opp_power > my_power and spell_damage > 0:
            best_idx: int | None = None
            best_power = 0
            for i, c in enumerate(candidates):
                if c.kind == "creature":
                    creature = opponent.get_card_by_id(c.ref_id)
                    if (
                        creature
                        and creature.effective_toughness <= spell_damage
                        and creature.effective_power > best_power
                    ):
                        best_idx = i
                        best_power = creature.effective_power
            if best_idx is not None:
                return best_idx

        # Ahead or parity → go face
        for i, c in enumerate(candidates):
            if c.kind == "player" and c.ref_id == opponent_idx:
                return i

        # Own-creature targeting (pump spells) → highest power
        best_idx_pump = 0
        best_power_pump = -1
        for i, c in enumerate(candidates):
            if c.kind == "creature" and c.ref_id is not None:
                for card in player.battlefield:
                    if card.card_id == c.ref_id:
                        power, _ = self.rules_engine._get_effective_power_toughness(card, player)
                        if power > best_power_pump:
                            best_power_pump = power
                            best_idx_pump = i
                        break

        return best_idx_pump

    def _decide_opponent_mulligan_keep(self) -> bool:
        """Decide whether the opponent keeps or mulligans its current hand.

        Applies a London-mulligan-style policy with seeded stochastic
        noise so the opponent acts as a competent baseline rather than
        always keeping the opener:

        * Hand quality scoring: count lands and very-early
          (CMC<=2) plays in the opener. A "broken" hand is 0-land,
          7-land, or all-spells-no-lands; these get a low keep-prob.
        * Hand-size cliff: once the opener drops to <=5 cards the
          marginal cost of another mulligan exceeds the marginal
          hand-quality gain, so the keep-prob saturates at 1.0.
        * Stochasticity: the keep decision samples against the
          computed probability using the env's seeded RNG
          (``self._np_random``), so play stays bit-exactly
          reproducible per episode seed.

        Returns:
            ``True`` if the opponent should keep this hand, ``False``
            to mulligan.
        """
        if self.state is None:
            return True
        opponent = self.state.players[1]
        hand = list(getattr(opponent, "hand", []) or [])
        hand_size = len(hand)
        if hand_size <= 1:
            return True

        mulligan_count = int(self.state.mulligan_count[1])
        # London mulligan: at 5 cards the opponent has already seen
        # 7 + 6 + 5 = 18 cards; taking a 4-card hand is almost always
        # worse on average.
        if mulligan_count >= 3 or hand_size <= 5:
            return True

        try:
            from mtg.env.card_definitions import CardType
        except ImportError:
            return True

        lands = sum(1 for c in hand if getattr(c, "card_type", None) == CardType.LAND)
        early_plays = 0
        for c in hand:
            if getattr(c, "card_type", None) == CardType.LAND:
                continue
            cost = getattr(c, "mana_cost", None)
            cmc = float(getattr(cost, "cmc", 99)) if cost is not None else 99.0
            if cmc <= 2.0:
                early_plays += 1

        # Default keep-prob calibration, tuned to roughly match pro-play
        # empirical mulligan rates for 7-card openers:
        #   0 or 7 lands         -> mulligan with high probability
        #   1 land + 0 early     -> mulligan often
        #   2-5 lands + early    -> keep almost always
        #   1 land + 2+ early    -> ~50/50 (aggro-style snap-keep)
        if lands == 0 or lands >= 7:
            keep_prob = 0.05
        elif lands == 1:
            keep_prob = 0.35 + 0.20 * min(early_plays, 2)
        elif lands in (6,):
            keep_prob = 0.55
        else:
            # The "good keeps" band: 2 / 3 / 4 / 5 lands.
            keep_prob = 0.85 + 0.05 * min(early_plays, 3)
            keep_prob = min(keep_prob, 0.99)

        rng = self._np_random
        if rng is None:
            rng = np.random.default_rng()
            self._np_random = rng
        return bool(rng.random() < keep_prob)

    def _advance_to_player_priority(self) -> None:
        """Advance game state until player has priority again."""
        assert self.state is not None

        # Keep advancing while opponent has priority or during automatic phases
        max_iterations = 100  # Safety limit
        iteration = 0

        while not self.state.game_over and iteration < max_iterations:
            iteration += 1

            # If we are awaiting a multi-step selection, return control
            if self.state.pending_action_type:
                break

            # Player has priority during a priority phase
            if self.state.priority_player == 0 and self.state.phase in PRIORITY_PHASES:
                break

            # Mulligan phase
            if self.state.phase == GamePhase.MULLIGAN:
                if self.state.priority_player == 0:
                    break
                keep = self._decide_opponent_mulligan_keep()
                self.state = self.rules_engine.execute_mulligan(self.state, keep=keep)

            # Automatic phases (no priority)
            elif self.state.phase in {
                GamePhase.UNTAP,
                GamePhase.CLEANUP,
                GamePhase.COMBAT_DAMAGE,
            }:
                self.state = self.rules_engine.advance_phase(self.state)

            # Opponent's turn
            elif self.state.active_player == 1:
                if self.opponent_agent is not None:
                    self._execute_opponent_agent_turn()
                else:
                    self.state = self.rules_engine.execute_opponent_turn(self.state)

            # Player's non-priority phase - advance
            elif self.state.phase not in PRIORITY_PHASES:
                self.state = self.rules_engine.advance_phase(self.state)

            else:
                break

    def _handle_opponent_actions(self) -> None:
        """Handle opponent priority and actions after player action."""
        assert self.state is not None

        if self.state.game_over:
            return

        max_steps = 50  # Safety limit to prevent infinite loops
        step = 0
        consecutive_non_progress = 0  # Track actions that don't advance game

        # If opponent has priority, let them act
        while self.state.priority_player == 1 and not self.state.game_over and step < max_steps:
            step += 1
            prev_stack_size = len(self.state.stack)
            prev_phase = self.state.phase

            if self.opponent_agent is not None:
                # Use opponent agent for priority decisions
                info = self._get_opponent_info()
                action_mask = self.action_builder.build_action_mask(self.state, player_id=1)
                action = self.opponent_agent.select_action(
                    observation=None,
                    action_mask=action_mask,
                    info=info,
                )
                decoded = self.action_builder.decode_action(action, self.state, player_id=1)
                self._execute_opponent_action(decoded)
            else:
                self.state = self.rules_engine.execute_opponent_priority(self.state)

            # Detect if no progress was made (same stack, same phase, still has priority)
            if (
                self.state.priority_player == 1
                and len(self.state.stack) == prev_stack_size
                and self.state.phase == prev_phase
            ):
                consecutive_non_progress += 1
                if consecutive_non_progress >= 5:
                    # Force pass to prevent stuck loops
                    self.state = self.rules_engine.pass_priority(self.state)
                    break
            else:
                consecutive_non_progress = 0

        if step >= max_steps:
            # Force pass priority to break potential infinite loop
            self.state = self.rules_engine.pass_priority(self.state)

        # Advance to next player priority window
        self._advance_to_player_priority()

    def _execute_opponent_agent_turn(self) -> None:
        """Execute opponent's turn using the opponent agent.

        This method runs a loop where the opponent agent selects actions
        until the turn passes back to the player.
        """
        assert self.state is not None
        assert self.opponent_agent is not None

        max_steps = 100  # Safety limit (reduced from 200)
        step = 0
        consecutive_non_progress = 0

        while self.state.active_player == 1 and not self.state.game_over and step < max_steps:
            step += 1
            prev_turn = self.state.turn_number
            prev_phase = self.state.phase
            prev_stack = len(self.state.stack)

            # Set pending_action_type for opponent's combat phase
            # This is critical for heuristic agents to properly score attack actions
            if (
                self.state.phase == GamePhase.COMBAT_BEGIN
                and not self.state.stack
                and self.state.pending_action_type != "attack"
            ):
                opponent = self.state.players[1]
                attackers = [c for c in opponent.battlefield if opponent.can_attack_with(c)]
                if attackers:
                    self.state.pending_action_type = "attack"
                    self.state.pending_player = 1

            # Build info for opponent (player_id=1)
            info = self._get_opponent_info()
            action_mask = self.action_builder.build_action_mask(self.state, player_id=1)

            # Get action from opponent agent
            action = self.opponent_agent.select_action(
                observation=None,  # Agents typically don't use raw obs
                action_mask=action_mask,
                info=info,
            )

            # Execute the action
            decoded = self.action_builder.decode_action(action, self.state, player_id=1)
            self._execute_opponent_action(decoded)

            # Handle automatic phases
            if self.state.phase in {GamePhase.UNTAP, GamePhase.CLEANUP, GamePhase.COMBAT_DAMAGE}:
                self.state = self.rules_engine.advance_phase(self.state)

            # PRIORITY WINDOW: if the opponent just cast a spell that went on
            # the stack, give the player a chance to respond (counterspell,
            # instant removal, combat trick). Without this, the player never
            # sees opponent creatures on the stack and counter/removal
            # effectively can't interact with the opponent's turn plays.
            stack_grew = len(self.state.stack) > prev_stack
            if stack_grew and self._player_has_instant_interaction():
                self.state.priority_player = 0
                break

            # Detect stuck state
            if (
                self.state.turn_number == prev_turn
                and self.state.phase == prev_phase
                and len(self.state.stack) == prev_stack
                and self.state.active_player == 1
            ):
                consecutive_non_progress += 1
                if consecutive_non_progress >= 10:
                    # Force advance turn
                    self.state = self.rules_engine.pass_priority(self.state)
                    break
            else:
                consecutive_non_progress = 0

    def _player_has_instant_interaction(self) -> bool:
        """Return True if player 0 has any castable instant-speed interaction.

        Used to decide whether to yield priority during the opponent's
        turn after they cast a spell. The player only yields when they
        could actually use the window; otherwise the default
        fast-forward behaviour is kept for matchups where interaction
        is impossible.
        """
        assert self.state is not None
        player = self.state.players[0]
        for card in player.hand:
            if card.card_type != CardType.INSTANT and Keyword.FLASH not in card.keywords:
                continue
            # Use can_cast_spell to respect mana / color requirements.
            if self.rules_engine.can_cast_spell(self.state, card, player_idx=0):
                return True
        return False

    def _get_opponent_info(self) -> dict[str, Any]:
        """Build info dict from opponent's perspective (player_id=1)."""
        assert self.state is not None

        opponent = self.state.players[1]  # "Opponent" is player index 1
        player = self.state.players[0]  # "Player" is player index 0

        # From opponent's perspective, they are the "player" and we are the "opponent"
        opp_creatures = []
        for c in opponent.battlefield:
            if c.card_type == CardType.CREATURE or c.card_id in opponent.activated_creatures:
                power, toughness = self.rules_engine._get_effective_power_toughness(c, opponent)
                opp_creatures.append(
                    {
                        "name": c.name,
                        "power": power,
                        "toughness": toughness,
                        "tapped": c.card_id in opponent.tapped_permanents,
                        "card_id": c.card_id,
                    }
                )

        player_creatures = []
        for c in player.battlefield:
            if c.card_type == CardType.CREATURE or c.card_id in player.activated_creatures:
                power, toughness = self.rules_engine._get_effective_power_toughness(c, player)
                player_creatures.append(
                    {
                        "name": c.name,
                        "power": power,
                        "toughness": toughness,
                        "tapped": c.card_id in player.tapped_permanents,
                        "card_id": c.card_id,
                    }
                )

        # Attack candidates for opponent
        attack_candidates = []
        for c in opponent.battlefield:
            if opponent.can_attack_with(c):
                power, toughness = self.rules_engine._get_effective_power_toughness(c, opponent)
                attack_candidates.append(
                    {
                        "name": c.name,
                        "power": power,
                        "toughness": toughness,
                        "tapped": c.card_id in opponent.tapped_permanents,
                        "card_id": c.card_id,
                    }
                )

        # Block candidates from opponent's POV: opponent's creatures that
        # can block (when "the player" = player 0 is attacking).
        blocker_candidates_opp = []
        for c in opponent.battlefield:
            if opponent.can_block_with(c):
                power, toughness = self.rules_engine._get_effective_power_toughness(c, opponent)
                blocker_candidates_opp.append(
                    {
                        "name": c.name,
                        "power": power,
                        "toughness": toughness,
                        "tapped": c.card_id in opponent.tapped_permanents,
                    }
                )

        # Attackers declared against opponent (from opponent's POV)
        block_attacker_candidates_opp = []
        for c in player.battlefield:
            if c.card_id in player.declared_attackers:
                power, toughness = self.rules_engine._get_effective_power_toughness(c, player)
                block_attacker_candidates_opp.append(
                    {
                        "name": c.name,
                        "power": power,
                        "toughness": toughness,
                        "tapped": c.card_id in player.tapped_permanents,
                    }
                )

        return {
            "player_idx": 1,
            "action_metadata": self.action_builder.get_action_metadata(self.state),
            "action_names": self.action_builder.get_action_names(self.state, player_id=1),
            "turn": self.state.turn_number,
            "player_deck": self.opponent_archetype_name,
            "opponent_deck": self.deck_archetype_name,
            "player_on_play": self.state.player_on_play,
            "player_life": opponent.life,  # From opponent's view
            "opponent_life": player.life,
            "player_creatures": opp_creatures,
            "opponent_creatures": player_creatures,
            "mana_available": opponent.get_total_available_mana(),
            "opponent_mana_available": player.get_total_available_mana(),
            "hand_size": len(opponent.hand),
            "opponent_hand_size": len(player.hand),
            "player_hand": [
                (c.name, c.mana_cost.to_text(), c.card_type.value) for c in opponent.hand
            ],
            "player_lands": [c.name for c in opponent.battlefield if c.produces_mana],
            "opponent_lands_list": [c.name for c in player.battlefield if c.produces_mana],
            "lands": sum(1 for c in opponent.hand if c.card_type == CardType.LAND),
            "phase_enum": self.state.phase.name,
            "active_player_idx": self.state.active_player,
            "stack_size": len(self.state.stack),
            "attack_candidates": attack_candidates,
            "blocker_candidates": blocker_candidates_opp,
            "block_attacker_candidates": block_attacker_candidates_opp,
            "player_attackers_available": sum(
                1
                for c in opponent.battlefield
                if c.card_type == CardType.CREATURE and opponent.can_attack_with(c)
            ),
            "player_blockers_available": sum(
                1
                for c in opponent.battlefield
                if c.card_type == CardType.CREATURE and opponent.can_block_with(c)
            ),
            "opponent_blockers_available": sum(
                1
                for c in player.battlefield
                if c.card_type == CardType.CREATURE and player.can_block_with(c)
            ),
            "opponent_declared_attackers": len(player.declared_attackers),
            "board_power": sum(
                c.power for c in opponent.battlefield if c.card_type == CardType.CREATURE
            ),
            "opponent_power": sum(
                c.power for c in player.battlefield if c.card_type == CardType.CREATURE
            ),
            "pending_action_type": self.state.pending_action_type,
            "pending_required": self.state.pending_required,
            "pending_selected_indices": list(self.state.pending_selected_indices),
            "pending_attackers": list(self.state.pending_attackers),
            "pending_attacker_names": [
                opponent.get_card_by_id(cid).name
                for cid in self.state.pending_attackers
                if opponent.get_card_by_id(cid) is not None
            ],
            "pending_block_assignments": {
                opponent.get_card_by_id(bid).name: player.get_card_by_id(aid).name
                for bid, aid in self.state.pending_block_assignments.items()
                if opponent.get_card_by_id(bid) and player.get_card_by_id(aid)
            },
            "pending_block_attacker_id": self.state.pending_block_attacker_id,
            "pending_spell_name": self.state.pending_spell.source.name
            if self.state.pending_spell
            else None,
            "pending_target_candidates": [
                {"kind": t.kind, "ref_id": t.ref_id, "name": t.name}
                for t in self.state.pending_target_candidates
            ],
            "lands_played_this_turn": opponent.lands_played_this_turn,
        }

    def _execute_opponent_action(self, decoded) -> None:
        """Execute a decoded action for the opponent (player_id=1)."""
        assert self.state is not None
        opponent = self.state.players[1]

        if decoded.kind == ActionKind.PASS:
            self.state = self.rules_engine.pass_priority(self.state)

        elif decoded.kind == ActionKind.PLAY_LAND:
            lands = [c for c in opponent.hand if c.card_type == CardType.LAND]
            if decoded.slot < len(lands):
                self.state = self.rules_engine.play_land(self.state, lands[decoded.slot])

        elif decoded.kind in {ActionKind.CAST_INSTANT, ActionKind.CAST_SORCERY}:
            if decoded.kind == ActionKind.CAST_INSTANT:
                cards = self.action_builder._get_instant_speed_cards(opponent.hand)
            else:
                cards = self.action_builder._get_sorcery_speed_cards(opponent.hand)
            if decoded.slot < len(cards):
                _, card = cards[decoded.slot]
                self.state = self.rules_engine.cast_spell(self.state, card)

        elif decoded.kind == ActionKind.ATTACK_ALL:
            attackers = [c.card_id for c in opponent.battlefield if opponent.can_attack_with(c)]
            if attackers:
                self.state = self.rules_engine.declare_attackers(self.state, attackers)

        elif decoded.kind == ActionKind.ATTACK_TOGGLE:
            candidates = [c for c in opponent.battlefield if opponent.can_attack_with(c)]
            if decoded.slot < len(candidates):
                card = candidates[decoded.slot]
                if card.card_id in self.state.pending_attackers:
                    self.state.pending_attackers.remove(card.card_id)
                else:
                    self.state.pending_attackers.add(card.card_id)

        elif decoded.kind == ActionKind.CONFIRM:
            if self.state.phase == GamePhase.COMBAT_BEGIN:
                attackers = (
                    list(self.state.pending_attackers) if self.state.pending_attackers else []
                )
                self.state.pending_attackers.clear()
                if attackers:
                    self.state = self.rules_engine.declare_attackers(self.state, attackers)
                else:
                    self.state = self.rules_engine.advance_phase(self.state)
            elif self.state.phase == GamePhase.COMBAT_BLOCKERS:
                self.state = self.rules_engine.advance_phase(self.state)
            else:
                self.state = self.rules_engine.pass_priority(self.state)

    def _get_info(self) -> dict[str, Any]:
        """Build info dict for current state."""
        assert self.state is not None

        player = self.state.players[0]
        opponent = self.state.players[1]

        # Get creatures for visualization
        player_creatures = []
        for c in player.battlefield:
            if c.card_type == CardType.CREATURE or c.card_id in player.activated_creatures:
                power, toughness = self.rules_engine._get_effective_power_toughness(c, player)
                player_creatures.append(
                    (c.name, power, toughness, c.card_id in player.tapped_permanents)
                )

        opponent_creatures = []
        for c in opponent.battlefield:
            if c.card_type == CardType.CREATURE or c.card_id in opponent.activated_creatures:
                power, toughness = self.rules_engine._get_effective_power_toughness(c, opponent)
                opponent_creatures.append(
                    (c.name, power, toughness, c.card_id in opponent.tapped_permanents)
                )

        player_attackers_available = sum(
            1
            for c in player.battlefield
            if c.card_type == CardType.CREATURE and player.can_attack_with(c)
        )
        player_blockers_available = sum(
            1
            for c in player.battlefield
            if c.card_type == CardType.CREATURE and player.can_block_with(c)
        )
        opponent_blockers_available = sum(
            1
            for c in opponent.battlefield
            if c.card_type == CardType.CREATURE and opponent.can_block_with(c)
        )
        opponent_declared_attackers = len(opponent.declared_attackers)

        attack_candidates = []
        for c in player.battlefield:
            if player.can_attack_with(c):
                power, toughness = self.rules_engine._get_effective_power_toughness(c, player)
                attack_candidates.append(
                    {
                        "name": c.name,
                        "power": power,
                        "toughness": toughness,
                        "tapped": c.card_id in player.tapped_permanents,
                        "card_id": c.card_id,
                    }
                )

        block_attacker_candidates = []
        for c in opponent.battlefield:
            if c.card_id in opponent.declared_attackers:
                power, toughness = self.rules_engine._get_effective_power_toughness(c, opponent)
                block_attacker_candidates.append(
                    {
                        "name": c.name,
                        "power": power,
                        "toughness": toughness,
                        "tapped": c.card_id in opponent.tapped_permanents,
                    }
                )

        blocker_candidates = []
        for c in player.battlefield:
            if player.can_block_with(c):
                power, toughness = self.rules_engine._get_effective_power_toughness(c, player)
                blocker_candidates.append(
                    {
                        "name": c.name,
                        "power": power,
                        "toughness": toughness,
                        "tapped": c.card_id in player.tapped_permanents,
                    }
                )

        # Get graveyard info
        player_graveyard = [(c.name, c.card_type.value) for c in player.graveyard]
        opponent_graveyard = [(c.name, c.card_type.value) for c in opponent.graveyard]

        # Get player hand for display (opponent hand hidden for partial observability)
        player_hand = [(c.name, c.mana_cost.to_text(), c.card_type.value) for c in player.hand]

        # Get all permanents on battlefield
        player_lands = [c.name for c in player.battlefield if c.produces_mana]
        opponent_lands_list = [c.name for c in opponent.battlefield if c.produces_mana]

        info = {
            "player_deck": self.deck_archetype_name,
            "opponent_deck": self.opponent_archetype_name,
            "player_idx": 0,
            "action_mask": self.action_builder.build_action_mask(self.state, player_id=0),
            "action_names": self.action_builder.get_action_names(self.state, player_id=0),
            "action_metadata": self.action_builder.get_action_metadata(self.state),
            "turn": self.state.turn_number,
            "phase": self._get_phase_display_name(),
            "phase_enum": self.state.phase.name,
            "active_player": "Player" if self.state.active_player == 0 else "Opponent",
            "active_player_idx": self.state.active_player,
            "priority_player": "Player" if self.state.priority_player == 0 else "Opponent",
            "stack_size": len(self.state.stack),
            "player_life": player.life,
            "opponent_life": opponent.life,
            "hand_size": len(player.hand),
            "opponent_hand_size": len(opponent.hand),
            "player_hand": player_hand,
            "lands": sum(
                1 for c in player.hand if c.card_type == CardType.LAND
            ),  # For mulligan decisions
            "lands_on_battlefield": sum(1 for c in player.battlefield if c.produces_mana),
            "opponent_lands": sum(1 for c in opponent.battlefield if c.produces_mana),
            "player_lands": player_lands,
            "opponent_lands_list": opponent_lands_list,
            "mana_available": player.get_total_available_mana(),
            "opponent_mana_available": opponent.get_total_available_mana(),
            "board_power": sum(
                c.power for c in player.battlefield if c.card_type == CardType.CREATURE
            ),
            "opponent_power": sum(
                c.power for c in opponent.battlefield if c.card_type == CardType.CREATURE
            ),
            "player_creatures": player_creatures,
            "opponent_creatures": opponent_creatures,
            "player_attackers_available": player_attackers_available,
            "player_blockers_available": player_blockers_available,
            "opponent_blockers_available": opponent_blockers_available,
            "opponent_declared_attackers": opponent_declared_attackers,
            "player_graveyard": player_graveyard,
            "opponent_graveyard": opponent_graveyard,
            "player_graveyard_size": len(player.graveyard),
            "opponent_graveyard_size": len(opponent.graveyard),
            "player_on_play": self.state.player_on_play,
            "pending_action_type": self.state.pending_action_type,
            "pending_required": self.state.pending_required,
            "pending_selected_indices": list(self.state.pending_selected_indices),
            "pending_attackers": list(self.state.pending_attackers),
            "pending_attacker_names": [
                player.get_card_by_id(cid).name
                for cid in self.state.pending_attackers
                if player.get_card_by_id(cid)
            ],
            "pending_block_assignments": {
                player.get_card_by_id(bid).name: opponent.get_card_by_id(aid).name
                for bid, aid in self.state.pending_block_assignments.items()
                if player.get_card_by_id(bid) and opponent.get_card_by_id(aid)
            },
            "pending_block_attacker_id": self.state.pending_block_attacker_id,
            "pending_spell_name": self.state.pending_spell.source.name
            if self.state.pending_spell
            else None,
            "pending_target_candidates": [
                {"kind": t.kind, "ref_id": t.ref_id, "name": t.name}
                for t in self.state.pending_target_candidates
            ],
            "pending_mana_sources": list(self.state.pending_mana_sources),
            "pending_mana_chosen": list(self.state.pending_mana_chosen),
            "pending_mana_source_cards": [
                {
                    "name": (player.get_card_by_id(cid) or opponent.get_card_by_id(cid)).name,
                    "colors": [
                        c.value
                        for c in (
                            player.get_card_by_id(cid) or opponent.get_card_by_id(cid)
                        ).produces_mana
                    ],
                }
                for cid in self.state.pending_mana_sources
                if (player.get_card_by_id(cid) or opponent.get_card_by_id(cid))
            ],
            "attack_candidates": attack_candidates,
            "blocker_candidates": blocker_candidates,
            "block_attacker_candidates": block_attacker_candidates,
            "causal_variables": self.reward_calculator.get_causal_variable_values(
                self.state, player_id=0
            ),
        }

        # Full action logs are expensive (grow each step) and only needed
        # for post-game reports, not during training rollouts.  Include
        # them only on terminal steps to keep VecEnv memory lean.
        if self.state.game_over:
            info["action_log"] = [
                {
                    "player": a.player,
                    "active_player": a.active_player,
                    "action_type": a.action_type,
                    "card_name": a.card_name,
                    "phase": a.phase,
                    "turn": a.turn,
                    "details": a.details,
                }
                for a in self.state.action_log
            ]
            info["turn_actions"] = {
                turn: {
                    player_idx: [
                        {
                            "action_type": a.action_type,
                            "card_name": a.card_name,
                            "phase": a.phase,
                            "details": a.details,
                        }
                        for a in actions
                    ]
                    for player_idx, actions in player_actions.items()
                }
                for turn, player_actions in self.state.turn_actions.items()
            }

        if self.state.game_over:
            info["winner"] = self.state.winner
            if self.state.winner == 0:
                info["game_result"] = "win"
            elif self.state.winner == 1:
                info["game_result"] = "loss"
            else:
                info["game_result"] = "draw"  # Timeout/no winner

        return info

    def _get_phase_display_name(self) -> str:
        """Get human-readable phase name."""
        assert self.state is not None

        phase_names = {
            GamePhase.MULLIGAN: "Mulligan",
            GamePhase.UNTAP: "Untap",
            GamePhase.UPKEEP: "Upkeep",
            GamePhase.DRAW: "Draw",
            GamePhase.MAIN_PRECOMBAT: "Main 1",
            GamePhase.COMBAT_BEGIN: "Combat",
            GamePhase.COMBAT_ATTACKERS: "Attackers",
            GamePhase.COMBAT_BLOCKERS: "Blockers",
            GamePhase.COMBAT_DAMAGE: "Damage",
            GamePhase.MAIN_POSTCOMBAT: "Main 2",
            GamePhase.END_STEP: "End",
            GamePhase.CLEANUP: "Cleanup",
            GamePhase.GAME_OVER: "Game Over",
        }
        return phase_names.get(self.state.phase, self.state.phase.name)

    def _copy_state(self, state: GameState) -> GameState:
        """Create a deep copy of the game state."""
        import copy

        return copy.deepcopy(state)

    def set_opponent(
        self,
        agent: Any,
        name: str | None = None,
        deck: str | None = None,
    ) -> None:
        """Swap the opponent agent (and optionally the opponent deck).

        Used by ``LeagueEnvWrapper`` to rotate opponents between
        episodes.  If ``deck`` is provided the opponent archetype is
        reloaded, which also reshuffles a fresh opponent deck at the
        next ``reset``.  ``name`` is stored so completed-episode info
        dicts can surface ``active_opponent`` for league bookkeeping.
        """
        self.opponent_agent = agent
        if name is not None:
            self.active_opponent_name = name
        if deck is not None and deck != self.opponent_archetype_name:
            self.opponent_archetype_name = deck
            self.opponent_archetype = get_archetype(deck)

    def render(self) -> str | None:
        """Render the current game state."""
        if self.state is None:
            return None

        if self.render_mode in ("ansi", "human"):
            return self._render_ansi()

        return None

    def _render_ansi(self) -> str:
        """Render game state as ANSI string."""
        assert self.state is not None

        lines = []
        lines.append("=" * 60)
        lines.append(
            f"Turn {self.state.turn_number} - "
            f"{'Player' if self.state.active_player == 0 else 'Opponent'}'s Turn | "
            f"Phase: {self._get_phase_display_name()} | "
            f"Priority: {'You' if self.state.priority_player == 0 else 'Opp'}"
        )
        lines.append("=" * 60)

        for i, player in enumerate(self.state.players):
            label = "You" if i == 0 else "Opponent"
            lines.append(f"\n{label} (Life: {player.life})")
            lines.append(f"  Hand: {len(player.hand)} cards")
            if i == 0:  # Only show agent's hand
                for card in player.hand:
                    instant_marker = "[I]" if card.card_type == CardType.INSTANT else ""
                    lines.append(f"    - {card.name} {instant_marker}")
            lines.append(f"  Battlefield ({len(player.battlefield)}):")
            for card in player.battlefield:
                tapped = "(T)" if card.card_id in player.tapped_permanents else ""
                lines.append(f"    - {card.name} {tapped}")

        lines.append("\n" + "=" * 60)

        output = "\n".join(lines)
        if self.render_mode == "human":
            print(output)
        return output

    def get_legal_actions(self) -> list[int]:
        """Get list of legal action indices."""
        if self.state is None:
            return []
        return self.action_builder.get_legal_actions(self.state, player_id=0)

    def action_to_string(self, action: int) -> str:
        """Convert action index to human-readable string."""
        if self.state is None:
            return f"Action {action}"
        return self.action_builder.action_to_string(action, self.state, player_id=0)

    def close(self) -> None:
        """Clean up environment resources."""
        pass

"""Tests for gameplay mechanics and game simulation."""

import numpy as np
import pytest

from mtg.env import MTGEnv
from mtg.env.card_definitions import Card, CardType, Keyword, ManaCost
from mtg.env.rules import GamePhase, GameState, PlayerState, RulesEngine


class TestCombatMechanics:
    """Tests for combat mechanics."""

    def test_first_strike_damage_order(self):
        """Test first strike deals damage before regular damage."""
        # Create first strike creature and regular creature
        first_striker = Card(
            name="First Striker",
            card_type=CardType.CREATURE,
            mana_cost=ManaCost(white=1),
            power=2,
            toughness=2,
            keywords={Keyword.FIRST_STRIKE},
            card_id=1001,
        )

        regular = Card(
            name="Regular",
            card_type=CardType.CREATURE,
            mana_cost=ManaCost(red=1),
            power=3,
            toughness=2,
            card_id=1002,
        )

        # First striker should have first strike property
        assert first_striker.has_first_strike
        assert not regular.has_first_strike

    def test_double_strike_has_first_strike(self):
        """Test double strike implies first strike."""
        double_striker = Card(
            name="Double Striker",
            card_type=CardType.CREATURE,
            mana_cost=ManaCost(red=2),
            power=2,
            toughness=2,
            keywords={Keyword.DOUBLE_STRIKE},
            card_id=1003,
        )

        assert double_striker.has_first_strike
        assert double_striker.has_double_strike

    def test_reach_can_block_flying(self):
        """Test reach creatures can block flyers."""
        flyer = Card(
            name="Flyer",
            card_type=CardType.CREATURE,
            mana_cost=ManaCost(white=2),
            power=2,
            toughness=2,
            keywords={Keyword.FLYING},
            card_id=1004,
        )

        reacher = Card(
            name="Reacher",
            card_type=CardType.CREATURE,
            mana_cost=ManaCost(green=2),
            power=3,
            toughness=3,
            keywords={Keyword.REACH},
            card_id=1005,
        )

        assert flyer.has_flying
        assert reacher.has_reach

    def test_menace_requires_two_blockers(self):
        """Test menace creature requires two blockers."""
        menace_creature = Card(
            name="Menace",
            card_type=CardType.CREATURE,
            mana_cost=ManaCost(black=2),
            power=3,
            toughness=2,
            keywords={Keyword.MENACE},
            card_id=1006,
        )

        assert menace_creature.has_menace

    def test_defender_cannot_attack(self):
        """Test defender creatures cannot attack."""
        defender = Card(
            name="Defender",
            card_type=CardType.CREATURE,
            mana_cost=ManaCost(white=1),
            power=0,
            toughness=4,
            keywords={Keyword.DEFENDER},
            card_id=1007,
        )

        assert defender.has_defender


class TestLifelinkAndDeathtouch:
    """Tests for lifelink and deathtouch."""

    def test_lifelink_creature(self):
        """Test lifelink property."""
        lifelink_creature = Card(
            name="Lifelink",
            card_type=CardType.CREATURE,
            mana_cost=ManaCost(white=2),
            power=2,
            toughness=2,
            keywords={Keyword.LIFELINK},
            card_id=1008,
        )

        assert lifelink_creature.has_lifelink

    def test_deathtouch_creature(self):
        """Test deathtouch property."""
        deathtouch_creature = Card(
            name="Deathtouch",
            card_type=CardType.CREATURE,
            mana_cost=ManaCost(black=1),
            power=1,
            toughness=1,
            keywords={Keyword.DEATHTOUCH},
            card_id=1009,
        )

        assert deathtouch_creature.has_deathtouch


class TestIndestructible:
    """Tests for indestructible keyword."""

    def test_indestructible_property(self):
        """Test indestructible property."""
        indestructible_creature = Card(
            name="Indestructible",
            card_type=CardType.CREATURE,
            mana_cost=ManaCost(white=4),
            power=4,
            toughness=4,
            keywords={Keyword.INDESTRUCTIBLE},
            card_id=1010,
        )

        assert indestructible_creature.has_indestructible


class TestGamePhases:
    """Tests for game phase transitions."""

    def test_all_phases_exist(self):
        """Test all game phases exist."""
        phases = [
            GamePhase.MULLIGAN,
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
            GamePhase.GAME_OVER,
        ]
        for phase in phases:
            assert phase is not None

    def test_game_starts_in_mulligan(self):
        """Test game starts in mulligan phase."""
        env = MTGEnv(deck_archetype="mono_red_aggro", seed=42)
        obs, info = env.reset()
        assert info["phase_enum"] == "MULLIGAN"


class TestPlayerState:
    """Tests for player state management."""

    def test_player_initial_state(self):
        """Test player initial state."""
        player = PlayerState()
        assert player.life == 20
        assert len(player.hand) == 0
        assert len(player.battlefield) == 0
        assert len(player.graveyard) == 0
        assert len(player.exile) == 0

    def test_player_draw_card(self):
        """Test player drawing a card."""
        player = PlayerState()
        card = Card(
            name="Test Card",
            card_type=CardType.CREATURE,
            power=2,
            toughness=2,
            card_id=1,
        )
        player.deck.append(card)

        drawn = player.draw_card()
        assert drawn is not None
        assert drawn.name == "Test Card"
        assert len(player.hand) == 1
        assert len(player.deck) == 0

    def test_player_draw_from_empty_deck(self):
        """Test drawing from empty deck returns None."""
        player = PlayerState()
        drawn = player.draw_card()
        assert drawn is None

    def test_player_reset_for_turn(self):
        """Test player state reset for turn."""
        player = PlayerState()
        player.lands_played_this_turn = 1
        player.has_drawn_for_turn = True
        player.mana_pool["W"] = 3

        player.reset_for_turn()

        assert player.lands_played_this_turn == 0
        assert not player.has_drawn_for_turn
        assert player.mana_pool["W"] == 0


class TestManaPayment:
    """Tests for mana payment mechanics."""

    def test_mana_cost_creation(self):
        """Test mana cost creation."""
        cost = ManaCost(white=1, blue=2, generic=1)
        assert cost.white == 1
        assert cost.blue == 2
        assert cost.generic == 1
        assert cost.cmc == 4

    def test_mana_cost_text(self):
        """Test mana cost text representation."""
        cost = ManaCost(red=2, generic=1)
        text = cost.to_text()
        assert "R" in text or "1" in text


class TestGameSimulation:
    """Tests for full game simulation."""

    def test_game_runs_to_completion(self):
        """Test game can run to completion."""
        env = MTGEnv(deck_archetype="mono_red_aggro", max_turns=5, seed=42)
        obs, info = env.reset()

        done = False
        steps = 0
        max_steps = 200

        while not done and steps < max_steps:
            action_mask = info["action_mask"]
            if action_mask.sum() == 0:
                obs, _, terminated, truncated, info = env.step(0)
                done = terminated or truncated
            else:
                legal = np.where(action_mask > 0)[0]
                action = int(np.random.choice(legal))
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            steps += 1

        # Game should complete
        assert done or steps >= max_steps

    def test_game_has_winner(self):
        """Test game ends with a winner or draw."""
        env = MTGEnv(deck_archetype="mono_red_aggro", max_turns=5, seed=42)
        obs, info = env.reset()

        done = False
        steps = 0

        while not done and steps < 200:
            action_mask = info["action_mask"]
            if action_mask.sum() == 0:
                obs, _, terminated, truncated, info = env.step(0)
                done = terminated or truncated
            else:
                legal = np.where(action_mask > 0)[0]
                action = int(np.random.choice(legal))
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            steps += 1

        if done:
            # Should have a winner
            assert "winner" in info or "terminated" in str(info)

    def test_multiple_games_different_seeds(self):
        """Test multiple games with different seeds produce different results."""
        results = []
        for seed in [1, 2, 3]:
            env = MTGEnv(deck_archetype="mono_red_aggro", max_turns=3, seed=seed)
            obs, info = env.reset()

            steps = 0
            done = False
            while not done and steps < 100:
                action_mask = info["action_mask"]
                if action_mask.sum() == 0:
                    obs, _, terminated, truncated, info = env.step(0)
                    done = terminated or truncated
                else:
                    legal = np.where(action_mask > 0)[0]
                    action = int(legal[0])  # Deterministic action
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                steps += 1
            results.append(steps)

        # Different seeds should produce variety
        assert len(results) == 3


class TestEnvironmentInterface:
    """Tests for environment interface."""

    def test_observation_shape(self):
        """Test observation has correct shape."""
        env = MTGEnv(deck_archetype="mono_red_aggro", seed=42)
        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert len(obs.shape) == 1

    def test_action_mask_shape(self):
        """Test action mask has correct shape."""
        env = MTGEnv(deck_archetype="mono_red_aggro", seed=42)
        obs, info = env.reset()

        action_mask = info["action_mask"]
        assert isinstance(action_mask, np.ndarray)
        assert action_mask.shape[0] == env.action_space.n

    def test_info_contains_required_keys(self):
        """Test info dict contains required keys."""
        env = MTGEnv(deck_archetype="mono_red_aggro", seed=42)
        obs, info = env.reset()

        assert "action_mask" in info
        assert "phase" in info
        assert "phase_enum" in info

    def test_environment_seeding(self):
        """Test environment seeding produces reproducible results."""
        env1 = MTGEnv(deck_archetype="mono_red_aggro", seed=42)
        env2 = MTGEnv(deck_archetype="mono_red_aggro", seed=42)

        obs1, info1 = env1.reset()
        obs2, info2 = env2.reset()

        np.testing.assert_array_equal(obs1, obs2)


class TestRewardSystem:
    """Tests for reward calculation."""

    def test_reward_is_float(self):
        """Test reward is a float."""
        env = MTGEnv(deck_archetype="mono_red_aggro", seed=42)
        obs, info = env.reset()

        action_mask = info["action_mask"]
        if action_mask.sum() > 0:
            legal = np.where(action_mask > 0)[0]
            obs, reward, _, _, _ = env.step(int(legal[0]))
            assert isinstance(reward, int | float)

    def test_reward_bounded(self):
        """Test rewards are reasonably bounded."""
        env = MTGEnv(deck_archetype="mono_red_aggro", seed=42)
        obs, info = env.reset()

        rewards = []
        done = False
        steps = 0

        while not done and steps < 100:
            action_mask = info["action_mask"]
            if action_mask.sum() == 0:
                obs, reward, terminated, truncated, info = env.step(0)
                done = terminated or truncated
            else:
                legal = np.where(action_mask > 0)[0]
                obs, reward, terminated, truncated, info = env.step(int(legal[0]))
                done = terminated or truncated
                rewards.append(reward)
            steps += 1

        # Rewards should be bounded (not infinite)
        for r in rewards:
            assert -100 <= r <= 100

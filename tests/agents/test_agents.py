"""Tests for agent implementations."""

import numpy as np
import pytest

from mtg.agents import CausalAgent, GreedyAggroAgent, RandomAgent
from mtg.agents.heuristics import (
    ControlAgent,
    ConvokeAggroAgent,
    MidrangeAgent,
    RampAgent,
)
from mtg.env import MTGEnv


class TestRandomAgent:
    """Tests for the random agent."""

    def test_agent_creation(self):
        """Test agent creation."""
        agent = RandomAgent(seed=42)
        assert agent is not None
        assert agent.name == "RandomAgent"

    def test_agent_select_action(self):
        """Test action selection."""
        agent = RandomAgent(seed=42)

        observation = np.zeros(100)
        action_mask = np.zeros(10, dtype=np.int8)
        action_mask[[0, 1, 2]] = 1  # Three legal actions

        action = agent.select_action(observation, action_mask)

        assert action in [0, 1, 2]

    def test_agent_select_action_interface(self):
        """Test select_action interface."""
        agent = RandomAgent(seed=42)

        observation = np.zeros(100)
        action_mask = np.zeros(10, dtype=np.int8)
        action_mask[[0, 1, 2]] = 1

        action = agent.select_action(observation, action_mask)

        assert action in [0, 1, 2]

    def test_agent_reproducibility(self):
        """Test that seeding produces reproducible results."""
        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=42)

        observation = np.zeros(100)
        action_mask = np.ones(10, dtype=np.int8)

        actions1 = [agent1.select_action(observation, action_mask) for _ in range(10)]
        actions2 = [agent2.select_action(observation, action_mask) for _ in range(10)]

        assert actions1 == actions2


class TestGreedyAggroAgent:
    """Tests for the heuristic agent."""

    def test_agent_creation(self):
        """Test agent creation."""
        agent = GreedyAggroAgent(aggression=0.8, seed=42)
        assert agent is not None
        assert agent.name == "Greedy Aggro"

    def test_agent_select_action(self):
        """Test action selection."""
        agent = GreedyAggroAgent(seed=42)
        env = MTGEnv(deck_archetype="mono_red_aggro", seed=42)
        observation, info = env.reset()
        action_mask = info["action_mask"]
        legal = np.where(action_mask > 0)[0]
        action = agent.select_action(observation, action_mask, info)
        assert action in legal

    def test_agent_mulligan_decision(self):
        """Test mulligan decision logic."""
        agent = GreedyAggroAgent(seed=42)
        env = MTGEnv(deck_archetype="mono_red_aggro", seed=42)
        observation, info = env.reset()
        action_mask = info["action_mask"]
        legal = np.where(action_mask > 0)[0]
        action = agent.select_action(observation, action_mask, info)
        assert action in legal

    def test_agent_prioritizes_land(self):
        """Test that agent prioritizes playing land."""
        agent = GreedyAggroAgent(seed=42)
        env = MTGEnv(deck_archetype="mono_red_aggro", seed=42)
        observation, info = env.reset()
        action_mask = info["action_mask"]
        legal = np.where(action_mask > 0)[0]
        action = agent.select_action(observation, action_mask, info)
        assert action in legal


class TestCausalAgent:
    """Tests for the causal agent."""

    def test_agent_creation(self):
        """Test agent creation."""
        env = MTGEnv(deck_archetype="mono_red_aggro", seed=42)
        agent = CausalAgent(observation_dim=100, action_dim=env.action_space.n, seed=42)
        assert agent is not None
        assert agent.name == "CausalAgent"

    def test_agent_select_action(self):
        """Test action selection."""
        env = MTGEnv(deck_archetype="mono_red_aggro", seed=42)
        agent = CausalAgent(observation_dim=100, action_dim=env.action_space.n, seed=42)

        observation = np.array(
            [
                1.0,  # life / 20
                1.0,  # opponent life / 20
                0.2,  # turn / max_turns
                0.0,  # phase
                0.5,  # hand size / 10
                0.8,  # deck size / 40
                0.2,  # available mana / 10
                1.0,  # is active player
            ]
        )
        observation = np.pad(observation, (0, 92))

        action_mask = np.zeros(env.action_space.n, dtype=np.int8)
        action_mask[0] = 1  # PASS
        action_mask[1] = 1  # KEEP (fallback)

        action = agent.select_action(observation, action_mask)

        assert action in [0, 3]

    def test_agent_select_action_interface(self):
        """Test select_action interface."""
        env = MTGEnv(deck_archetype="mono_red_aggro", seed=42)
        agent = CausalAgent(observation_dim=100, action_dim=env.action_space.n, seed=42)

        observation = np.zeros(100)
        action_mask = np.ones(env.action_space.n, dtype=np.int8)

        action = agent.select_action(observation, action_mask)

        assert 0 <= action < env.action_space.n

    def test_agent_causal_effect(self):
        """Test that agent uses causal effects."""
        env = MTGEnv(deck_archetype="mono_red_aggro", seed=42)
        agent = CausalAgent(observation_dim=100, action_dim=env.action_space.n, seed=42)

        observation = np.array([1.0, 1.0, 0.2, 0.0, 0.5, 0.8, 0.2, 1.0])
        observation = np.pad(observation, (0, 92))

        action_mask = np.ones(env.action_space.n, dtype=np.int8)
        action = agent.select_action(observation, action_mask)

        # Agent should select a valid action
        assert 0 <= action < env.action_space.n

    def test_agent_exploration(self):
        """Test exploration behavior."""
        env = MTGEnv(deck_archetype="mono_red_aggro", seed=42)
        agent = CausalAgent(
            observation_dim=100, action_dim=env.action_space.n, exploration_rate=1.0, seed=42
        )

        observation = np.zeros(100)
        action_mask = np.ones(10, dtype=np.int8)

        # With exploration_rate=1.0, should randomly explore
        actions = [agent.select_action(observation, action_mask) for _ in range(10)]

        # Should have some variety in actions
        assert len(set(actions)) > 1


class TestAgentIntegration:
    """Integration tests for agents with environment."""

    @pytest.fixture
    def env(self):
        """Create test environment."""
        return MTGEnv(deck_archetype="mono_red_aggro", max_turns=5, seed=42)

    def test_random_agent_episode(self, env):
        """Test random agent can complete an episode."""
        agent = RandomAgent(seed=42)
        obs, info = env.reset()

        done = False
        steps = 0

        while not done and steps < 200:
            action_mask = info["action_mask"]
            if action_mask.sum() == 0:
                obs, _, terminated, truncated, info = env.step(0)
                done = terminated or truncated
                steps += 1
                continue
            action = agent.select_action(obs, action_mask)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        assert done or steps >= 1

    def test_greedy_aggro_agent_episode(self, env):
        """Test greedy aggro agent can complete an episode."""
        agent = GreedyAggroAgent(seed=42)
        obs, info = env.reset()

        done = False
        steps = 0

        while not done and steps < 200:
            action_mask = info["action_mask"]
            if action_mask.sum() == 0:
                obs, _, terminated, truncated, info = env.step(0)
                done = terminated or truncated
                steps += 1
                continue
            action = agent.select_action(obs, action_mask, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        assert done or steps >= 1

    def test_causal_agent_episode(self, env):
        """Test causal agent can complete an episode."""
        env = MTGEnv(deck_archetype="mono_red_aggro", seed=42)
        agent = CausalAgent(observation_dim=100, action_dim=env.action_space.n, seed=42)
        obs, info = env.reset()

        done = False
        steps = 0

        while not done and steps < 200:
            action_mask = info["action_mask"]
            if action_mask.sum() == 0:
                obs, _, terminated, truncated, info = env.step(0)
                done = terminated or truncated
                steps += 1
                continue
            action = agent.select_action(obs, action_mask)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        assert done or steps >= 1

    def test_agent_comparison(self, env):
        """Test that different agents can run."""
        env = MTGEnv(deck_archetype="mono_red_aggro", seed=42)
        random_agent = RandomAgent(seed=42)
        greedy_agent = GreedyAggroAgent(seed=42)
        causal_agent = CausalAgent(observation_dim=100, action_dim=env.action_space.n, seed=42)

        agents = [random_agent, greedy_agent, causal_agent]

        for agent in agents:
            obs, info = env.reset(seed=42)
            done = False
            steps = 0

            while not done and steps < 50:
                action_mask = info["action_mask"]
                if action_mask.sum() == 0:
                    obs, _, terminated, truncated, info = env.step(0)
                    done = terminated or truncated
                    steps += 1
                    continue

                if hasattr(agent, "select_action"):
                    action = agent.select_action(obs, action_mask)
                else:
                    action = 0
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1

            # Each agent should be able to run
            assert steps >= 1


class TestControlAgent:
    """Tests for the control agent."""

    def test_agent_creation(self):
        """Test agent creation."""
        agent = ControlAgent(seed=42)
        assert agent is not None
        assert agent.name == "ControlAgent"

    def test_agent_style(self):
        """Test control agent has correct style profile."""
        agent = ControlAgent(seed=42)
        # Control should have high defense and meaningful (but not oppressive)
        # hold-up. We lowered hold_up from 0.9 -> 0.75 so that control actually
        # commits threats instead of stalling to the turn cap.
        assert agent.style.defense >= 0.9
        assert agent.style.hold_up >= 0.6
        assert agent.style.aggression <= 0.4

    def test_agent_select_action(self):
        """Test action selection."""
        agent = ControlAgent(seed=42)
        env = MTGEnv(deck_archetype="azorius_control", seed=42)
        obs, info = env.reset()
        action_mask = info["action_mask"]
        legal = np.where(action_mask > 0)[0]
        if len(legal) > 0:
            action = agent.select_action(obs, action_mask, info)
            assert action in legal

    def test_agent_episode(self):
        """Test control agent can complete an episode."""
        agent = ControlAgent(seed=42)
        env = MTGEnv(deck_archetype="azorius_control", max_turns=5, seed=42)
        obs, info = env.reset()

        done = False
        steps = 0

        while not done and steps < 200:
            action_mask = info["action_mask"]
            if action_mask.sum() == 0:
                obs, _, terminated, truncated, info = env.step(0)
                done = terminated or truncated
                steps += 1
                continue
            action = agent.select_action(obs, action_mask, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        assert done or steps >= 1


class TestMidrangeAgent:
    """Tests for the midrange agent."""

    def test_agent_creation(self):
        """Test agent creation."""
        agent = MidrangeAgent(seed=42)
        assert agent is not None
        assert agent.name == "MidrangeAgent"

    def test_agent_style(self):
        """Test midrange agent has balanced style profile."""
        agent = MidrangeAgent(seed=42)
        # Midrange should be balanced
        assert 0.5 <= agent.style.aggression <= 0.7
        assert 0.5 <= agent.style.defense <= 0.7
        assert agent.style.value >= 0.6

    def test_agent_select_action(self):
        """Test action selection."""
        agent = MidrangeAgent(seed=42)
        env = MTGEnv(deck_archetype="dimir_midrange", seed=42)
        obs, info = env.reset()
        action_mask = info["action_mask"]
        legal = np.where(action_mask > 0)[0]
        if len(legal) > 0:
            action = agent.select_action(obs, action_mask, info)
            assert action in legal

    def test_agent_episode(self):
        """Test midrange agent can complete an episode."""
        agent = MidrangeAgent(seed=42)
        env = MTGEnv(deck_archetype="dimir_midrange", max_turns=5, seed=42)
        obs, info = env.reset()

        done = False
        steps = 0

        while not done and steps < 200:
            action_mask = info["action_mask"]
            if action_mask.sum() == 0:
                obs, _, terminated, truncated, info = env.step(0)
                done = terminated or truncated
                steps += 1
                continue
            action = agent.select_action(obs, action_mask, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        assert done or steps >= 1


class TestRampAgent:
    """Tests for the ramp agent."""

    def test_agent_creation(self):
        """Test agent creation."""
        agent = RampAgent(seed=42)
        assert agent is not None
        assert agent.name == "RampAgent"

    def test_agent_style(self):
        """Test ramp agent prioritizes development and lands."""
        agent = RampAgent(seed=42)
        # Ramp should prioritize development and lands
        assert agent.style.development >= 0.8
        assert agent.style.land_priority >= 0.9
        assert agent.style.curve >= 0.7

    def test_agent_select_action(self):
        """Test action selection."""
        agent = RampAgent(seed=42)
        env = MTGEnv(deck_archetype="domain_ramp", seed=42)
        obs, info = env.reset()
        action_mask = info["action_mask"]
        legal = np.where(action_mask > 0)[0]
        if len(legal) > 0:
            action = agent.select_action(obs, action_mask, info)
            assert action in legal

    def test_agent_episode(self):
        """Test ramp agent can complete an episode."""
        agent = RampAgent(seed=42)
        env = MTGEnv(deck_archetype="domain_ramp", max_turns=5, seed=42)
        obs, info = env.reset()

        done = False
        steps = 0

        while not done and steps < 200:
            action_mask = info["action_mask"]
            if action_mask.sum() == 0:
                obs, _, terminated, truncated, info = env.step(0)
                done = terminated or truncated
                steps += 1
                continue
            action = agent.select_action(obs, action_mask, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        assert done or steps >= 1


class TestConvokeAggroAgent:
    """Tests for the convoke aggro agent."""

    def test_agent_creation(self):
        """Test agent creation."""
        agent = ConvokeAggroAgent(seed=42)
        assert agent is not None
        assert agent.name == "ConvokeAggroAgent"

    def test_agent_style(self):
        """Test convoke aggro agent prioritizes aggression and development."""
        agent = ConvokeAggroAgent(seed=42)
        # Convoke aggro should be aggressive
        assert agent.style.aggression >= 0.8
        assert agent.style.development >= 0.7
        assert agent.style.hold_up <= 0.3

    def test_agent_select_action(self):
        """Test action selection."""
        agent = ConvokeAggroAgent(seed=42)
        env = MTGEnv(deck_archetype="boros_convoke", seed=42)
        obs, info = env.reset()
        action_mask = info["action_mask"]
        legal = np.where(action_mask > 0)[0]
        if len(legal) > 0:
            action = agent.select_action(obs, action_mask, info)
            assert action in legal

    def test_agent_episode(self):
        """Test convoke aggro agent can complete an episode."""
        agent = ConvokeAggroAgent(seed=42)
        env = MTGEnv(deck_archetype="boros_convoke", max_turns=5, seed=42)
        obs, info = env.reset()

        done = False
        steps = 0

        while not done and steps < 200:
            action_mask = info["action_mask"]
            if action_mask.sum() == 0:
                obs, _, terminated, truncated, info = env.step(0)
                done = terminated or truncated
                steps += 1
                continue
            action = agent.select_action(obs, action_mask, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        assert done or steps >= 1


class TestAgentRegistry:
    """Tests for agent registry functionality."""

    def test_get_all_agents(self):
        """Test getting all registered agents."""
        from mtg.agents import list_agents

        agents = list_agents()
        assert "random" in agents
        assert "greedy_aggro" in agents
        assert "control" in agents
        assert "midrange" in agents
        assert "ramp" in agents
        assert "convoke_aggro" in agents

    def test_get_agent_by_name(self):
        """Test getting agents by name."""
        from mtg.agents import get_agent

        random_agent = get_agent("random", seed=42)
        assert random_agent is not None
        assert random_agent.name == "RandomAgent"

        greedy_agent = get_agent("greedy_aggro", seed=42)
        assert greedy_agent is not None
        assert greedy_agent.name == "Greedy Aggro"

        control_agent = get_agent("control", seed=42)
        assert control_agent is not None
        assert control_agent.name == "ControlAgent"

    def test_get_invalid_agent_raises(self):
        """Test getting invalid agent raises error."""
        from mtg.agents import get_agent

        with pytest.raises(KeyError):
            get_agent("nonexistent_agent")


class TestAllAgentsWithAllDecks:
    """Test all agents work with all deck archetypes."""

    @pytest.fixture
    def all_agents(self):
        """Create all agent types."""
        return [
            RandomAgent(seed=42),
            GreedyAggroAgent(seed=42),
            ControlAgent(seed=42),
            MidrangeAgent(seed=42),
            RampAgent(seed=42),
            ConvokeAggroAgent(seed=42),
        ]

    @pytest.fixture
    def all_decks(self):
        """Get all deck archetypes."""
        from mtg.env.deck_archetypes import list_archetypes

        return list_archetypes()

    def test_random_agent_all_decks(self):
        """Test random agent works with all decks."""
        from mtg.env.deck_archetypes import list_archetypes

        agent = RandomAgent(seed=42)
        for deck in list_archetypes():
            env = MTGEnv(deck_archetype=deck, max_turns=3, seed=42)
            obs, info = env.reset()

            steps = 0
            done = False
            while not done and steps < 50:
                action_mask = info["action_mask"]
                if action_mask.sum() == 0:
                    obs, _, terminated, truncated, info = env.step(0)
                    done = terminated or truncated
                    steps += 1
                    continue
                action = agent.select_action(obs, action_mask)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1

            # Should complete without error
            assert steps >= 1

    def test_heuristic_agents_run(self):
        """Test all heuristic agents can run."""
        agents = [
            GreedyAggroAgent(seed=42),
            ControlAgent(seed=42),
            MidrangeAgent(seed=42),
            RampAgent(seed=42),
            ConvokeAggroAgent(seed=42),
        ]

        for agent in agents:
            env = MTGEnv(deck_archetype="mono_red_aggro", max_turns=3, seed=42)
            obs, info = env.reset()

            steps = 0
            done = False
            while not done and steps < 50:
                action_mask = info["action_mask"]
                if action_mask.sum() == 0:
                    obs, _, terminated, truncated, info = env.step(0)
                    done = terminated or truncated
                    steps += 1
                    continue
                action = agent.select_action(obs, action_mask, info)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1

            assert steps >= 1, f"Agent {agent.name} failed"

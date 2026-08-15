#!/usr/bin/env python3
"""Comprehensive tests for all agent/deck combinations.

This module provides both pytest tests and a standalone validation script.

Run as pytest:
    pytest tests/test_comprehensive_agent_deck_combinations.py -v

Run as standalone script (fast validation):
    python tests/test_comprehensive_agent_deck_combinations.py
"""

from __future__ import annotations

import sys
import time
from typing import Any

import numpy as np
import pytest

from mtg.agents import CausalAgent, GreedyAggroAgent, RandomAgent
from mtg.agents.heuristics import (
    ControlAgent,
    ConvokeAggroAgent,
    MidrangeAgent,
    RampAgent,
)
from mtg.agents.reinforcement_learning.ppo_agent import PPOAgent
from mtg.env import MTGEnv
from mtg.env.deck_archetypes import list_archetypes

# Constants
ALL_DECKS = list_archetypes()
QUICK_STEPS = 5  # Steps for quick validation
FULL_GAME_TURNS = 10  # Turns for full game tests


def get_agents(action_dim: int, seed: int = 42) -> dict[str, Any]:
    """Create all agent types for testing."""
    return {
        "random": RandomAgent(seed=seed),
        "greedy_aggro": GreedyAggroAgent(seed=seed),
        "control": ControlAgent(seed=seed),
        "midrange": MidrangeAgent(seed=seed),
        "ramp": RampAgent(seed=seed),
        "convoke_aggro": ConvokeAggroAgent(seed=seed),
        "ppo": PPOAgent(observation_dim=100, action_dim=action_dim, seed=seed),
        "causal": CausalAgent(observation_dim=100, action_dim=action_dim, seed=seed),
    }


def get_heuristic_agents(seed: int = 42) -> dict[str, Any]:
    """Create heuristic agents only (no RL agents)."""
    return {
        "random": RandomAgent(seed=seed),
        "greedy_aggro": GreedyAggroAgent(seed=seed),
        "control": ControlAgent(seed=seed),
        "midrange": MidrangeAgent(seed=seed),
        "ramp": RampAgent(seed=seed),
        "convoke_aggro": ConvokeAggroAgent(seed=seed),
    }


def quick_validate(env: MTGEnv, agent: Any, steps: int = QUICK_STEPS) -> bool:
    """Run a few steps and verify no crashes."""
    obs, info = env.reset()
    for _ in range(steps):
        mask = info["action_mask"]
        if mask.sum() > 0:
            action = agent.select_action(obs, mask, info)
            assert mask[action] == 1, f"Agent selected illegal action {action}"
            obs, _, term, trunc, info = env.step(action)
        else:
            obs, _, term, trunc, info = env.step(0)
        if term or trunc:
            break
    return True


def run_full_game(
    player_deck: str,
    opponent_deck: str,
    player_agent: Any,
    opponent_agent: Any,
    max_turns: int = FULL_GAME_TURNS,
    seed: int = 42,
) -> dict[str, Any]:
    """Run a full game and return results.

    Returns:
        dict with game_completed, turns_played, winner, player_life, opponent_life,
        action_log, errors
    """
    env = MTGEnv(
        deck_archetype=player_deck,
        opponent_archetype=opponent_deck,
        max_turns=max_turns,
        seed=seed,
        opponent_agent=opponent_agent,
    )

    result = {
        "player_deck": player_deck,
        "opponent_deck": opponent_deck,
        "game_completed": False,
        "turns_played": 0,
        "winner": None,
        "player_life": 20,
        "opponent_life": 20,
        "action_count": 0,
        "errors": [],
    }

    try:
        obs, info = env.reset()
        max_actions = max_turns * 100  # Safety limit

        for action_num in range(max_actions):
            mask = info["action_mask"]
            if mask.sum() > 0:
                action = player_agent.select_action(obs, mask, info)
                if mask[action] != 1:
                    result["errors"].append(f"Illegal action {action} at step {action_num}")
                    break
                obs, reward, term, trunc, info = env.step(action)
                result["action_count"] += 1
            else:
                obs, reward, term, trunc, info = env.step(0)

            result["player_life"] = info.get("player_life", 0)
            result["opponent_life"] = info.get("opponent_life", 0)

            if term or trunc:
                result["game_completed"] = True
                result["turns_played"] = env.state.turn_number if hasattr(env, "state") else 0
                if result["player_life"] <= 0:
                    result["winner"] = "opponent"
                elif result["opponent_life"] <= 0:
                    result["winner"] = "player"
                else:
                    result["winner"] = "draw/timeout"
                break

    except Exception as e:
        result["errors"].append(str(e))

    return result


# =============================================================================
# PYTEST TEST CLASSES
# =============================================================================


class TestAllAgentDeckCombinations:
    """Test all agent/deck combinations work."""

    def test_all_40_agent_deck_combinations_as_player(self):
        """All 8 agents × 5 decks = 40 combinations."""
        for deck in ALL_DECKS:
            env = MTGEnv(deck_archetype=deck, max_turns=2, seed=42)
            agents = get_agents(env.action_space.n)
            for name, agent in agents.items():
                assert quick_validate(env, agent), f"{name}/{deck} failed"

    def test_all_25_deck_matchups(self):
        """All 5 × 5 = 25 deck matchups initialize correctly."""
        for p_deck in ALL_DECKS:
            for o_deck in ALL_DECKS:
                env = MTGEnv(deck_archetype=p_deck, opponent_archetype=o_deck, max_turns=2, seed=42)
                obs, info = env.reset()
                assert info["player_life"] == 20
                assert info["opponent_life"] == 20


class TestOpponentAgents:
    """Test all agents work as opponents."""

    def test_heuristic_agents_as_opponent(self):
        """Test 6 heuristic agents × 3 sample decks = 18 combinations."""
        player = RandomAgent(seed=42)
        sample_decks = ["mono_red_aggro", "azorius_control", "domain_ramp"]

        for opp_deck in sample_decks:
            opp_agents = get_heuristic_agents(seed=123)
            for opp_name, opp_agent in opp_agents.items():
                env = MTGEnv(
                    deck_archetype="mono_red_aggro",
                    opponent_archetype=opp_deck,
                    max_turns=2,
                    seed=42,
                    opponent_agent=opp_agent,
                )
                assert quick_validate(env, player), f"{opp_name} opponent/{opp_deck} failed"


class TestMagicRulesValidation:
    """Validate Magic rules basics."""

    @pytest.mark.parametrize("deck", ALL_DECKS)
    def test_initial_state_correct(self, deck: str):
        """Each deck starts with correct initial state."""
        env = MTGEnv(deck_archetype=deck, max_turns=2, seed=42)
        _, info = env.reset()
        assert info["player_life"] == 20
        assert info["opponent_life"] == 20
        assert info["hand_size"] == 7
        assert info["phase_enum"] == "MULLIGAN"

    def test_action_masks_valid(self):
        """Action masks are properly formatted."""
        for deck in ALL_DECKS:
            env = MTGEnv(deck_archetype=deck, max_turns=2, seed=42)
            _, info = env.reset()
            mask = info["action_mask"]
            assert isinstance(mask, np.ndarray)
            assert mask.shape[0] == env.action_space.n
            assert np.all((mask == 0) | (mask == 1))


class TestMirrorMatches:
    """Test mirror matches work."""

    @pytest.mark.parametrize("deck", ALL_DECKS)
    def test_mirror_match(self, deck: str):
        """Each deck works in mirror match."""
        env = MTGEnv(
            deck_archetype=deck,
            opponent_archetype=deck,
            max_turns=2,
            seed=42,
            opponent_agent=RandomAgent(seed=123),
        )
        assert quick_validate(env, RandomAgent(seed=42))


class TestGameplayDeterminism:
    """Test determinism."""

    def test_same_seed_same_result(self):
        """Same seed produces identical action sequences."""
        runs = []
        for _ in range(2):
            env = MTGEnv(deck_archetype="mono_red_aggro", max_turns=2, seed=42)
            agent = RandomAgent(seed=42)
            obs, info = env.reset()
            actions = []
            for _ in range(10):
                mask = info["action_mask"]
                if mask.sum() > 0:
                    action = agent.select_action(obs, mask, info)
                    actions.append(action)
                    obs, _, term, trunc, info = env.step(action)
                else:
                    obs, _, term, trunc, info = env.step(0)
                if term or trunc:
                    break
            runs.append(actions)
        assert runs[0] == runs[1]


class TestFullGames:
    """Test complete games run without issues."""

    @pytest.mark.parametrize("deck", ALL_DECKS)
    def test_full_10_turn_game(self, deck: str):
        """Each deck can play a 10-turn game without errors."""
        player = GreedyAggroAgent(seed=42)
        opponent = ControlAgent(seed=123)

        result = run_full_game(
            player_deck=deck,
            opponent_deck="azorius_control",
            player_agent=player,
            opponent_agent=opponent,
            max_turns=10,
        )

        assert not result["errors"], f"Errors in {deck}: {result['errors']}"
        assert result["action_count"] > 0, f"No actions taken in {deck}"


# =============================================================================
# STANDALONE VALIDATION SCRIPT
# =============================================================================


def run_quick_validation() -> bool:
    """Run fast validation of all combinations (standalone mode)."""
    start = time.time()
    print("=" * 60)
    print("MTG-Causal-RL: Fast Validation of All Combinations")
    print("=" * 60)

    all_decks = ALL_DECKS
    print(f"\nFound {len(all_decks)} decks: {all_decks}")

    # Test 1: All deck matchups can initialize
    print("\n[1/5] Testing all 25 deck matchups initialize...", end=" ", flush=True)
    t0 = time.time()
    failures = []

    for p_deck in all_decks:
        for o_deck in all_decks:
            try:
                env = MTGEnv(deck_archetype=p_deck, opponent_archetype=o_deck, max_turns=2, seed=42)
                obs, info = env.reset()
                assert info["player_life"] == 20
                assert info["opponent_life"] == 20
            except Exception as e:
                failures.append(f"{p_deck} vs {o_deck}: {e}")

    if failures:
        print("FAILED")
        for f in failures:
            print(f"    ✗ {f}")
        return False
    print(f"PASS ({time.time() - t0:.1f}s)")

    # Test 2: All agents can select actions
    print("\n[2/5] Testing all 8 agents select valid actions...", end=" ", flush=True)
    t0 = time.time()

    env = MTGEnv(deck_archetype="mono_red_aggro", max_turns=2, seed=42)
    agents = get_agents(env.action_space.n)
    failures = []

    for name, agent in agents.items():
        try:
            obs, info = env.reset()
            mask = info["action_mask"]
            if mask.sum() > 0:
                action = agent.select_action(obs, mask, info)
                assert mask[action] == 1, "Illegal action"
        except Exception as e:
            failures.append(f"{name}: {e}")

    if failures:
        print("FAILED")
        for f in failures:
            print(f"    ✗ {f}")
        return False
    print(f"PASS ({time.time() - t0:.1f}s)")

    # Test 3: Each agent works with each deck
    print("\n[3/5] Testing all 40 agent/deck combinations (5 steps)...", end=" ", flush=True)
    t0 = time.time()
    failures = []

    for deck in all_decks:
        env = MTGEnv(deck_archetype=deck, max_turns=2, seed=42)
        agents = get_agents(env.action_space.n)

        for agent_name, agent in agents.items():
            try:
                quick_validate(env, agent)
            except Exception as e:
                failures.append(f"{agent_name}/{deck}: {e}")

    if failures:
        print("FAILED")
        for f in failures:
            print(f"    ✗ {f}")
        return False
    print(f"PASS ({time.time() - t0:.1f}s)")

    # Test 4: Opponent agents work
    print("\n[4/5] Testing 6 opponent agents with 3 deck samples...", end=" ", flush=True)
    t0 = time.time()
    failures = []

    sample_decks = ["mono_red_aggro", "azorius_control", "domain_ramp"]
    player = RandomAgent(seed=42)

    for opp_deck in sample_decks:
        opp_agents = get_heuristic_agents(seed=123)
        for opp_name, opp_agent in opp_agents.items():
            try:
                env = MTGEnv(
                    deck_archetype="mono_red_aggro",
                    opponent_archetype=opp_deck,
                    max_turns=2,
                    seed=42,
                    opponent_agent=opp_agent,
                )
                quick_validate(env, player)
            except Exception as e:
                failures.append(f"{opp_name} opp/{opp_deck}: {e}")

    if failures:
        print("FAILED")
        for f in failures:
            print(f"    ✗ {f}")
        return False
    print(f"PASS ({time.time() - t0:.1f}s)")

    # Test 5: Magic rules validation
    print("\n[5/5] Validating Magic rules...", end=" ", flush=True)
    t0 = time.time()
    failures = []

    for deck in all_decks:
        try:
            env = MTGEnv(deck_archetype=deck, max_turns=2, seed=42)
            _, info = env.reset()
            assert info["player_life"] == 20
            assert info["opponent_life"] == 20
            assert info["hand_size"] == 7
            assert info["phase_enum"] == "MULLIGAN"
            mask = info["action_mask"]
            assert isinstance(mask, np.ndarray)
            assert np.all((mask == 0) | (mask == 1))
        except Exception as e:
            failures.append(f"{deck}: {e}")

    if failures:
        print("FAILED")
        for f in failures:
            print(f"    ✗ {f}")
        return False
    print(f"PASS ({time.time() - t0:.1f}s)")

    # Summary
    total_time = time.time() - start
    print("\n" + "=" * 60)
    print(f"✓ ALL VALIDATIONS PASSED in {total_time:.1f}s")
    print("=" * 60)
    print(
        """
Summary:
  • 25 deck matchups verified
  • 8 agents verified
  • 40 agent/deck player combinations verified
  • 18 opponent agent combinations verified
  • All Magic rules (initial state, action masks) verified
"""
    )
    return True


if __name__ == "__main__":
    success = run_quick_validation()
    sys.exit(0 if success else 1)

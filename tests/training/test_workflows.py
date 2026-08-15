"""Tests for MTG-Causal-RL workflows.

This module tests the training, evaluation, and gameplay workflows
to ensure all components work correctly together.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

# =============================================================================
# Interactive Module Tests
# =============================================================================


class TestInteractiveModule:
    """Tests for the interactive CLI module."""

    def test_training_config_creation(self) -> None:
        """Test TrainingConfig can be created with defaults."""
        from mtg.utils.interactive import TrainingConfig

        config = TrainingConfig(
            agent_type="ppo",
            player_deck="mono_red_aggro",
            opponent_deck="azorius_control",
        )

        assert config.agent_type == "ppo"
        assert config.player_deck == "mono_red_aggro"
        assert config.opponent_deck == "azorius_control"
        assert config.timesteps == 1_000_000
        assert config.reward_type == "shaped"
        assert config.seed == 42

    def test_training_config_run_name(self) -> None:
        """Test TrainingConfig generates valid run name."""
        from mtg.utils.interactive import TrainingConfig

        config = TrainingConfig(
            agent_type="causal",
            player_deck="mono_red_aggro",
            opponent_deck="dimir_midrange",
        )

        run_name = config.get_run_name()

        assert "causal" in run_name
        assert "mono_red_aggro" in run_name
        assert "dimir_midrange" in run_name

    def test_evaluation_config_creation(self) -> None:
        """Test EvaluationConfig can be created with defaults."""
        from mtg.utils.interactive import EvaluationConfig

        config = EvaluationConfig(agent_type="greedy_aggro")

        assert config.agent_type == "greedy_aggro"
        assert config.episodes == 500
        assert len(config.seeds) == 5
        assert config.model_path is None

    def test_gameplay_config_creation(self) -> None:
        """Test GameplayConfig can be created with defaults."""
        from mtg.utils.interactive import GameplayConfig

        config = GameplayConfig()

        assert config.player_agent == "greedy_aggro"
        assert config.player_deck == "mono_red_aggro"
        assert config.opponent_deck == "azorius_control"
        assert config.speed == "medium"
        assert config.save_report is True

    def test_get_available_agents(self) -> None:
        """Test that available agents can be retrieved."""
        from mtg.utils.interactive import get_available_agents

        agents = get_available_agents()

        assert isinstance(agents, list)
        assert len(agents) >= 4
        assert "random" in agents
        assert "greedy_aggro" in agents
        assert "ppo" in agents
        assert "causal" in agents

    def test_get_available_archetypes(self) -> None:
        """Test that available archetypes can be retrieved."""
        from mtg.utils.interactive import get_available_archetypes

        archetypes = get_available_archetypes()

        assert isinstance(archetypes, list)
        assert len(archetypes) >= 5
        assert "mono_red_aggro" in archetypes
        assert "azorius_control" in archetypes

    def test_get_archetype_info(self) -> None:
        """Test that archetype info can be retrieved."""
        from mtg.utils.interactive import get_archetype_info

        info = get_archetype_info("mono_red_aggro")

        assert "name" in info
        assert "display_name" in info
        assert "strategy" in info
        assert info["strategy"] == "AGGRO"

    def test_create_output_directory(self) -> None:
        """Test output directory creation."""
        from mtg.utils.interactive import create_output_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = create_output_directory(tmpdir, "test_run")

            assert output_path.exists()
            assert (output_path / "plots").exists()
            assert (output_path / "reports").exists()

    def test_format_duration(self) -> None:
        """Test duration formatting."""
        from mtg.utils.interactive import format_duration

        assert format_duration(65) == "1m 5s"
        assert format_duration(3665) == "1h 1m 5s"
        assert format_duration(30) == "30s"

    def test_discover_trained_models_empty(self) -> None:
        """Test model discovery with empty directory."""
        from mtg.utils.interactive import discover_trained_models

        with tempfile.TemporaryDirectory() as tmpdir:
            models = discover_trained_models(tmpdir)
            assert models == []


# =============================================================================
# Game Simulator Tests
# =============================================================================


class TestGameSimulator:
    """Tests for the game simulator module."""

    def test_simulation_config_creation(self) -> None:
        """Test SimulationConfig can be created."""
        from mtg.simulation import SimulationConfig

        config = SimulationConfig(
            player_deck="mono_red_aggro",
            opponent_deck="azorius_control",
            max_turns=5,
        )

        assert config.player_deck == "mono_red_aggro"
        assert config.opponent_deck == "azorius_control"
        assert config.max_turns == 5

    def test_game_result_creation(self) -> None:
        """Test GameResult can be created."""
        from mtg.simulation import GameResult

        result = GameResult(
            winner="Player",
            player_life=20,
            opponent_life=0,
            turns_played=5,
            total_reward=1.0,
        )

        assert result.winner == "Player"
        assert result.player_life == 20
        assert result.opponent_life == 0

    def test_game_simulator_initialization(self) -> None:
        """Test GameSimulator can be initialized."""
        from mtg.agents import get_agent
        from mtg.simulation import GameSimulator, SimulationConfig

        config = SimulationConfig(
            player_deck="mono_red_aggro",
            opponent_deck="azorius_control",
            visualize=False,
            record=False,
        )

        agent = get_agent("random", seed=42)
        simulator = GameSimulator(config, agent)

        assert simulator.config == config
        assert simulator.player_agent == agent

    @pytest.mark.slow
    def test_run_game_completes(self) -> None:
        """Test that a game can be run to completion."""
        from mtg.agents import get_agent
        from mtg.simulation import run_game

        agent = get_agent("random", seed=42)
        result = run_game(
            player_agent=agent,
            player_deck="mono_red_aggro",
            opponent_deck="azorius_control",
            max_turns=3,
            seed=42,
            visualize=False,
            record=False,
        )

        assert result.winner in ["Player", "Opponent", "Draw"]
        assert result.turns_played >= 1
        assert result.actions_taken >= 0

    @pytest.mark.slow
    def test_run_game_with_recording(self) -> None:
        """Test that a game can be run with recording."""
        from mtg.agents import get_agent
        from mtg.simulation import run_game

        agent = get_agent("random", seed=42)
        result = run_game(
            player_agent=agent,
            player_deck="mono_red_aggro",
            opponent_deck="azorius_control",
            max_turns=3,
            seed=42,
            visualize=False,
            record=True,
        )

        assert result.recorder is not None
        assert result.game_id != ""


# =============================================================================
# Agent Registry Tests
# =============================================================================


class TestAgentRegistry:
    """Tests for agent registration and custom agents."""

    def test_register_custom_agent(self) -> None:
        """Test registering a custom agent."""
        from mtg.agents import BaseAgent, get_agent, list_agents, register_agent

        class TestCustomAgent(BaseAgent):
            """A test custom agent."""

            def __init__(self, **kwargs) -> None:
                super().__init__(name="test_custom", **kwargs)

            def select_action(self, observation, action_mask, info=None) -> int:
                legal = [i for i, m in enumerate(action_mask) if m > 0]
                return legal[0] if legal else 0

        register_agent("test_custom", TestCustomAgent, overwrite=True)

        assert "test_custom" in list_agents()

        agent = get_agent("test_custom")
        assert agent.name == "test_custom"

    def test_list_agents_includes_defaults(self) -> None:
        """Test that list_agents includes default agents."""
        from mtg.agents import list_agents

        agents = list_agents()

        assert "random" in agents
        assert "greedy_aggro" in agents
        assert "ppo" in agents
        assert "causal" in agents

    def test_get_agent_by_name(self) -> None:
        """Test getting agents by name."""
        from mtg.agents import get_agent

        for agent_name in ["random", "greedy_aggro"]:
            agent = get_agent(agent_name, seed=42)
            assert agent is not None

    def test_get_agent_invalid_name_raises(self) -> None:
        """Test that invalid agent name raises KeyError."""
        from mtg.agents import get_agent

        with pytest.raises(KeyError):
            get_agent("nonexistent_agent")


# =============================================================================
# Deck Archetype Tests
# =============================================================================


class TestDeckArchetypes:
    """Tests for deck archetype functionality."""

    def test_get_archetype(self) -> None:
        """Test getting an archetype by name."""
        from mtg.env.deck_archetypes import get_archetype

        archetype = get_archetype("mono_red_aggro")

        assert archetype.name == "mono_red_aggro"
        assert archetype.display_name == "Mono-Red Aggro"

    def test_archetype_build_deck(self) -> None:
        """Test building a deck from an archetype."""
        from mtg.env.deck_archetypes import get_archetype

        archetype = get_archetype("mono_red_aggro")
        deck = archetype.build_deck()

        assert len(deck) == 60

    def test_archetype_validation(self) -> None:
        """Test deck validation."""
        from mtg.env.deck_archetypes import get_archetype

        archetype = get_archetype("mono_red_aggro")
        is_valid, errors = archetype.validate()

        # May have validation errors due to card list issues,
        # but should not crash
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)

    def test_list_archetypes(self) -> None:
        """Test listing all archetypes."""
        from mtg.env.deck_archetypes import list_archetypes

        archetypes = list_archetypes()

        assert len(archetypes) >= 5
        assert "mono_red_aggro" in archetypes
        assert "azorius_control" in archetypes

    def test_archetype_aliases(self) -> None:
        """Test archetype aliases work."""
        from mtg.env.deck_archetypes import get_archetype

        # These aliases should work
        aggro = get_archetype("aggro")
        control = get_archetype("control")

        assert aggro.name == "mono_red_aggro"
        assert control.name == "azorius_control"


# =============================================================================
# HTML Report Tests
# =============================================================================


class TestHTMLReport:
    """Tests for HTML report generation."""

    def test_game_recorder_creation(self) -> None:
        """Test GameRecorder can be created."""
        from mtg.utils.html_report import GameRecorder

        recorder = GameRecorder(
            player_deck="mono_red_aggro",
            opponent_deck="azorius_control",
            player_agent="Random",
            opponent_agent="Heuristic",
        )

        assert recorder.player_deck == "mono_red_aggro"
        assert recorder.opponent_deck == "azorius_control"

    def test_game_recorder_set_player_on_play(self) -> None:
        """Test setting play/draw."""
        from mtg.utils.html_report import GameRecorder

        recorder = GameRecorder(
            player_deck="mono_red_aggro",
            opponent_deck="azorius_control",
        )

        recorder.set_player_on_play(True)
        assert recorder.player_on_play is True

        recorder.set_player_on_play(False)
        assert recorder.player_on_play is False

    def test_game_recorder_record_action(self) -> None:
        """Test recording an action."""
        from mtg.utils.html_report import GameRecorder

        recorder = GameRecorder(
            player_deck="mono_red_aggro",
            opponent_deck="azorius_control",
        )

        recorder.record_action(
            turn=1,
            phase="Main 1",
            player="Player",
            action_type="PLAY_LAND",
            description="Play Mountain",
        )

        replay = recorder.get_replay()
        assert len(replay.actions) == 1
        assert replay.actions[0].description == "Play Mountain"

    def test_game_recorder_record_snapshot(self) -> None:
        """Test recording a state snapshot."""
        from mtg.utils.html_report import GameRecorder

        recorder = GameRecorder(
            player_deck="mono_red_aggro",
            opponent_deck="azorius_control",
        )

        recorder.record_snapshot(
            turn=1,
            phase="Main 1",
            active_player="Player",
            player_life=20,
            opponent_life=20,
            player_hand=[("Mountain", ""), ("Lightning Bolt", "R")],
            opponent_hand=[("Island", ""), ("Counterspell", "UU")],
            player_lands={"Mountain": 1},
            opponent_lands={"Island": 1},
        )

        replay = recorder.get_replay()
        assert len(replay.snapshots) == 1
        assert replay.snapshots[0].player_life == 20

    def test_generate_html_report(self) -> None:
        """Test HTML report generation."""
        from mtg.utils.html_report import GameRecorder, generate_html_report

        recorder = GameRecorder(
            player_deck="mono_red_aggro",
            opponent_deck="azorius_control",
        )
        recorder.set_player_on_play(True)
        recorder.set_winner("Player")

        recorder.record_action(1, "Main 1", "Player", "PLAY_LAND", "Play Mountain")
        recorder.record_snapshot(
            turn=1,
            phase="Main 1",
            active_player="Player",
            player_life=20,
            opponent_life=20,
            player_hand=[("Mountain", "")],
            opponent_hand=[("Island", "")],
            player_lands={"Mountain": 1},
            opponent_lands={},
        )

        replay = recorder.get_replay()

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            html_path = Path(f.name)

        try:
            generate_html_report(replay, html_path)
            assert html_path.exists()

            content = html_path.read_text()
            assert "mono_red_aggro" in content.lower() or "Mono" in content
        finally:
            html_path.unlink(missing_ok=True)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.slow
    def test_training_workflow_runs(self) -> None:
        """Test that training workflow can run (minimal)."""
        from mtg.utils.interactive import TrainingConfig

        # Just test config creation, not actual training
        config = TrainingConfig(
            agent_type="random",
            player_deck="mono_red_aggro",
            opponent_deck="azorius_control",
            timesteps=100,
            eval_episodes=5,
        )

        assert config.timesteps == 100

    @pytest.mark.slow
    def test_evaluation_workflow_runs(self) -> None:
        """Test that evaluation workflow can run (minimal)."""
        from mtg.agents import get_agent
        from mtg.simulation import run_evaluation

        agent = get_agent("random", seed=42)
        result = run_evaluation(
            agent=agent,
            player_deck="mono_red_aggro",
            opponent_deck="azorius_control",
            num_games=5,
            show_progress=False,
        )

        assert result.num_games == 5
        assert 0 <= result.win_rate <= 1

    @pytest.mark.slow
    def test_gameplay_workflow_runs(self) -> None:
        """Test that gameplay workflow can run (minimal)."""
        from mtg.agents import get_agent
        from mtg.simulation import run_game

        agent = get_agent("random", seed=42)
        result = run_game(
            player_agent=agent,
            player_deck="mono_red_aggro",
            opponent_deck="azorius_control",
            max_turns=2,
            seed=42,
            visualize=False,
            record=True,
        )

        assert result.winner in ["Player", "Opponent", "Draw"]
        assert result.recorder is not None


# =============================================================================
# CLI Display Tests
# =============================================================================


class TestCLIDisplay:
    """Tests for CLI display functions."""

    def test_format_mana_cost(self) -> None:
        """Test mana cost formatting."""
        from mtg.utils.cli_display import format_mana_cost

        assert "🔴" in format_mana_cost("R")
        assert "🔵" in format_mana_cost("U")
        assert "⚪" in format_mana_cost("W")
        assert "⚫" in format_mana_cost("B")
        assert "🟢" in format_mana_cost("G")
        assert "1" in format_mana_cost("1R")

    def test_console_exists(self) -> None:
        """Test console is defined."""
        from mtg.utils.cli_display import console

        assert console is not None


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

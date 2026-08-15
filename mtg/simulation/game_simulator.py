"""Game simulation engine for MTG-Causal-RL.

This module provides the core game simulation that runs agents against
each other in the MTG environment, with full state tracking and recording.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mtg.utils.html_report import (
    GameRecorder,
    generate_html_report,
    save_replay_json,
)


@dataclass
class SimulationConfig:
    """Configuration for game simulation.

    Attributes:
        player_deck: Player's deck archetype.
        opponent_deck: Opponent's deck archetype.
        max_turns: Maximum turns before game ends.
        reward_type: Reward shaping type.
        seed: Random seed for reproducibility.
        visualize: Whether to output CLI visualization.
        speed: Visualization speed preset.
        record: Whether to record game for replay.

    """

    player_deck: str = "mono_red_aggro"
    opponent_deck: str = "azorius_control"
    max_turns: int = 10
    reward_type: str = "sparse"
    seed: int = 42
    visualize: bool = True
    speed: str = "medium"
    record: bool = True


@dataclass
class GameResult:
    """Result of a simulated game.

    Attributes:
        winner: Winner of the game ("Player", "Opponent", or "Draw").
        player_life: Player's final life total.
        opponent_life: Opponent's final life total.
        turns_played: Number of turns played.
        total_reward: Total reward accumulated.
        game_id: Unique identifier for the game.
        actions_taken: Number of actions taken.
        recorder: GameRecorder with full game history (if recorded).

    """

    winner: str
    player_life: int
    opponent_life: int
    turns_played: int
    total_reward: float
    game_id: str = ""
    actions_taken: int = 0
    recorder: GameRecorder | None = None


@dataclass
class EvaluationResult:
    """Result of multiple game evaluation.

    Attributes:
        agent_name: Name of the evaluated agent.
        num_games: Number of games played.
        wins: Number of wins.
        losses: Number of losses.
        draws: Number of draws.
        win_rate: Win rate as a fraction.
        avg_reward: Average reward per game.
        avg_turns: Average turns per game.
        std_reward: Standard deviation of rewards.

    """

    agent_name: str
    num_games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    win_rate: float = 0.0
    avg_reward: float = 0.0
    avg_turns: float = 0.0
    std_reward: float = 0.0


class GameSimulator:
    """Game simulation engine.

    Runs MTG games with configurable agents and deck matchups,
    providing full state tracking and optional visualization.

    Attributes:
        config: Simulation configuration.
        env: MTG environment instance.
        player_agent: Agent playing as the player.
        opponent_agent: Agent playing as the opponent.
        recorder: GameRecorder for replay generation.

    """

    def __init__(
        self,
        config: SimulationConfig,
        player_agent: tp.Any,
        opponent_agent: tp.Any | None = None,
    ) -> None:
        """Initialize the game simulator.

        Args:
            config: Simulation configuration.
            player_agent: Agent for the player.
            opponent_agent: Agent for the opponent (uses env default if None).

        """
        self.config = config
        self.player_agent = player_agent
        self.opponent_agent = opponent_agent
        self.env: tp.Any = None
        self.recorder: GameRecorder | None = None

        # Speed presets (seconds between actions)
        self._speed_presets = {
            "slow": {"phase": 5.0, "action": 5.0},
            "medium": {"phase": 3.0, "action": 3.0},
            "fast": {"phase": 1.0, "action": 1.0},
        }

    def setup(self) -> None:
        """Initialize the environment and recorder."""
        from mtg.env import MTGEnv

        self.env = MTGEnv(
            deck_archetype=self.config.player_deck,
            opponent_archetype=self.config.opponent_deck,
            max_turns=self.config.max_turns,
            reward_type=self.config.reward_type,
            seed=self.config.seed,
        )

        if self.config.record:
            self.recorder = GameRecorder(
                player_deck=self.config.player_deck,
                opponent_deck=self.config.opponent_deck,
                player_agent=getattr(self.player_agent, "name", "Player"),
                opponent_agent=getattr(self.opponent_agent, "name", "Opponent")
                if self.opponent_agent
                else "Heuristic",
            )

    def run_game(self) -> GameResult:
        """Run a single game simulation.

        Returns:
            GameResult with complete game information.

        """
        if self.env is None:
            self.setup()

        assert self.env is not None

        obs, info = self.env.reset()
        done = False
        total_reward = 0.0
        actions_taken = 0

        # Initialize recorder
        if self.recorder:
            player_on_play = info.get("player_on_play", True)
            self.recorder.set_player_on_play(player_on_play)

            # Record initial state
            self._record_state(info, 1, "Untap", "Player")

        while not done:
            action_mask = info["action_mask"]

            # Select action
            action = self.player_agent.select_action(obs, action_mask, info)

            # Get action name
            action_name = info.get("action_names", {}).get(action, f"Action {action}")

            # Get current state before stepping
            turn = info.get("turn", 1)
            phase = info.get("phase", "Main 1")
            active_player = info.get("active_player", "Player")

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            actions_taken += 1
            done = terminated or truncated

            # Record action and state
            if self.recorder:
                action_type = self._classify_action(action_name)
                self.recorder.record_action(
                    turn=turn,
                    phase=phase,
                    player=active_player,
                    action_type=action_type,
                    description=action_name,
                )
                self._record_state(info, turn, phase, active_player)

            # Visualize if enabled
            if self.config.visualize:
                self._visualize_state(info, action_name)

        # Determine winner
        result = info.get("game_result", "unknown")
        winner = "Draw"
        if result == "win":
            winner = "Player"
        elif result == "loss":
            winner = "Opponent"

        if self.recorder:
            self.recorder.set_winner(winner)

        return GameResult(
            winner=winner,
            player_life=info.get("player_life", 0),
            opponent_life=info.get("opponent_life", 0),
            turns_played=info.get("turn", 1),
            total_reward=total_reward,
            game_id=self.recorder.game_id if self.recorder else "",
            actions_taken=actions_taken,
            recorder=self.recorder,
        )

    def _classify_action(self, action_name: str) -> str:
        """Classify action type from action name.

        Args:
            action_name: Full action description.

        Returns:
            Action type classification.

        """
        name_lower = action_name.lower()
        if "cast" in name_lower:
            return "CAST"
        elif "play" in name_lower and "land" in name_lower:
            return "PLAY_LAND"
        elif "attack" in name_lower:
            return "ATTACK"
        elif "block" in name_lower:
            return "BLOCK"
        elif "draw" in name_lower:
            return "DRAW"
        elif "mulligan" in name_lower:
            return "MULLIGAN"
        else:
            return "PASS"

    def _record_state(
        self,
        info: dict[str, tp.Any],
        turn: int,
        phase: str,
        active_player: str,
    ) -> None:
        """Record current game state.

        Args:
            info: Environment info dictionary.
            turn: Current turn number.
            phase: Current game phase.
            active_player: Currently active player.

        """
        if not self.recorder:
            return

        # Build simplified hand representation
        hand_size = info.get("hand_size", 0)
        player_hand = [("Unknown", "") for _ in range(hand_size)]

        opponent_hand_size = info.get("opponent_hand_size", 0)
        opponent_hand = [("Unknown", "") for _ in range(opponent_hand_size)]

        # Build land dict
        player_lands_count = info.get("lands", 0)
        player_lands = {"Land": player_lands_count}

        opponent_lands_count = info.get("opponent_lands", 0)
        opponent_lands = {"Land": opponent_lands_count}

        self.recorder.record_snapshot(
            turn=turn,
            phase=phase,
            active_player=active_player,
            player_life=info.get("player_life", 20),
            opponent_life=info.get("opponent_life", 20),
            player_hand=player_hand,
            opponent_hand=opponent_hand,
            player_lands=player_lands,
            opponent_lands=opponent_lands,
            player_creatures=info.get("player_creatures", []),
            opponent_creatures=info.get("opponent_creatures", []),
            player_graveyard=info.get("player_graveyard", []),
            opponent_graveyard=info.get("opponent_graveyard", []),
            board_power=info.get("board_power", 0),
            opponent_power=info.get("opponent_power", 0),
        )

    def _visualize_state(
        self,
        info: dict[str, tp.Any],
        action_name: str,
    ) -> None:
        """Visualize current game state in CLI.

        Args:
            info: Environment info dictionary.
            action_name: Last action taken.

        """
        import time

        from mtg.utils.cli_display import print_game_state

        delays = self._speed_presets.get(self.config.speed, self._speed_presets["medium"])

        print_game_state(
            turn=info.get("turn", 1),
            phase=info.get("phase", "Main 1"),
            player_life=info.get("player_life", 20),
            opponent_life=info.get("opponent_life", 20),
            hand_size=info.get("hand_size", 0),
            lands=info.get("lands", 0),
            board_power=info.get("board_power", 0),
            opponent_lands=info.get("opponent_lands", 0),
            opponent_power=info.get("opponent_power", 0),
            opponent_hand_size=info.get("opponent_hand_size", 0),
            last_action=action_name,
            active_player=info.get("active_player", "Player"),
            mana_available=info.get("mana_available", 0),
            player_creatures=info.get("player_creatures", []),
            opponent_creatures=info.get("opponent_creatures", []),
            player_graveyard=info.get("player_graveyard", []),
            opponent_graveyard=info.get("opponent_graveyard", []),
        )

        delay = delays["action"] if "pass" not in action_name.lower() else delays["phase"]
        time.sleep(delay)

    def save_report(
        self,
        result: GameResult,
        output_dir: str | Path,
    ) -> Path:
        """Save game replay report.

        Args:
            result: Game result with recorder.
            output_dir: Directory to save report.

        Returns:
            Path to saved report directory.

        """
        if not result.recorder:
            raise ValueError("Game was not recorded, cannot save report")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        replay = result.recorder.get_replay()

        html_path = output_path / "replay.html"
        json_path = output_path / "replay.json"

        generate_html_report(replay, html_path)
        save_replay_json(replay, json_path)

        return output_path


def run_game(
    player_agent: tp.Any,
    player_deck: str = "mono_red_aggro",
    opponent_deck: str = "azorius_control",
    opponent_agent: tp.Any | None = None,
    max_turns: int = 10,
    seed: int = 42,
    visualize: bool = False,
    record: bool = True,
) -> GameResult:
    """Convenience function to run a single game.

    Args:
        player_agent: Agent for the player.
        player_deck: Player's deck archetype.
        opponent_deck: Opponent's deck archetype.
        opponent_agent: Agent for opponent (optional).
        max_turns: Maximum turns.
        seed: Random seed.
        visualize: Whether to show CLI visualization.
        record: Whether to record for replay.

    Returns:
        GameResult with game information.

    """
    config = SimulationConfig(
        player_deck=player_deck,
        opponent_deck=opponent_deck,
        max_turns=max_turns,
        seed=seed,
        visualize=visualize,
        record=record,
    )

    simulator = GameSimulator(config, player_agent, opponent_agent)
    return simulator.run_game()


def run_evaluation(
    agent: tp.Any,
    player_deck: str = "mono_red_aggro",
    opponent_deck: str = "azorius_control",
    num_games: int = 100,
    seeds: list[int] | None = None,
    show_progress: bool = True,
) -> EvaluationResult:
    """Run evaluation across multiple games.

    Args:
        agent: Agent to evaluate.
        player_deck: Player's deck archetype.
        opponent_deck: Opponent's deck archetype.
        num_games: Number of games to play.
        seeds: Random seeds (one per game, or cycles if fewer).
        show_progress: Whether to show progress bar.

    Returns:
        EvaluationResult with aggregate statistics.

    """
    seeds = seeds or [42]
    rewards: list[float] = []
    turns: list[int] = []
    wins = 0
    losses = 0
    draws = 0

    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

    progress_context = (
        Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )
        if show_progress
        else None
    )

    def run_games() -> None:
        nonlocal wins, losses, draws

        iterator = range(num_games)
        if progress_context:
            task = progress_context.add_task("Evaluating...", total=num_games)

        for i in iterator:
            seed = seeds[i % len(seeds)]
            config = SimulationConfig(
                player_deck=player_deck,
                opponent_deck=opponent_deck,
                seed=seed + i,
                visualize=False,
                record=False,
            )

            simulator = GameSimulator(config, agent)
            result = simulator.run_game()

            rewards.append(result.total_reward)
            turns.append(result.turns_played)

            if result.winner == "Player":
                wins += 1
            elif result.winner == "Opponent":
                losses += 1
            else:
                draws += 1

            if progress_context:
                progress_context.advance(task)

    if progress_context:
        with progress_context:
            run_games()
    else:
        run_games()

    return EvaluationResult(
        agent_name=getattr(agent, "name", "Unknown"),
        num_games=num_games,
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate=wins / num_games if num_games > 0 else 0.0,
        avg_reward=float(np.mean(rewards)) if rewards else 0.0,
        avg_turns=float(np.mean(turns)) if turns else 0.0,
        std_reward=float(np.std(rewards)) if rewards else 0.0,
    )


__all__ = [
    "GameSimulator",
    "GameResult",
    "SimulationConfig",
    "EvaluationResult",
    "run_game",
    "run_evaluation",
]

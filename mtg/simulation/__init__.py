"""Game simulation module for MTG-Causal-RL.

This module provides abstract game simulation that can run any agent
against any opponent with any deck matchup.
"""

from mtg.simulation.game_simulator import (
    GameResult,
    GameSimulator,
    SimulationConfig,
    run_evaluation,
    run_game,
)

__all__ = [
    "GameSimulator",
    "GameResult",
    "SimulationConfig",
    "run_game",
    "run_evaluation",
]

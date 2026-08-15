"""Random agent for baseline comparison.

This agent selects uniformly at random from legal actions,
providing a lower bound on performance.
"""

from __future__ import annotations

import typing as tp

import numpy as np

from mtg.agents.base.base import BaseAgent


class RandomAgent(BaseAgent):
    """Agent that selects actions uniformly at random.

    This serves as a baseline for comparison. The agent respects
    action masking, only selecting from legal actions.

    Attributes:
        rng: Random number generator.

    """

    def __init__(
        self,
        seed: int | None = None,
    ) -> None:
        """Initialize the random agent.

        Args:
            seed: Optional random seed for reproducibility.

        """
        super().__init__(name="RandomAgent", deterministic=False)
        self.rng = np.random.default_rng(seed)

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        info: dict[str, tp.Any] | None = None,
    ) -> int:
        """Select a random legal action.

        Args:
            observation: Current state (unused).
            action_mask: Binary mask of legal actions.
            info: Optional info (unused).

        Returns:
            Randomly selected legal action index.

        """
        legal_actions = np.where(action_mask > 0)[0]
        if len(legal_actions) == 0:
            return 0
        return int(self.rng.choice(legal_actions))

    def get_action_probabilities(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
    ) -> np.ndarray:
        """Get uniform distribution over legal actions.

        Args:
            observation: Current state (unused).
            action_mask: Binary mask of legal actions.

        Returns:
            Uniform probability over legal actions.

        """
        n_actions = len(action_mask)
        legal_actions = np.where(action_mask > 0)[0]
        probs = np.zeros(n_actions)
        if len(legal_actions) > 0:
            probs[legal_actions] = 1.0 / len(legal_actions)
        return probs

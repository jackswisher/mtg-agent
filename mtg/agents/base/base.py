"""Base agent interface and registry for MTG-Causal-RL.

This module defines the abstract base class that all agents must implement,
providing a consistent interface for the benchmark. It also includes an
AgentRegistry for registering and instantiating custom agents.
"""

from __future__ import annotations

import typing as tp
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class AgentRegistry:
    """Registry for managing agent types.

    Allows users to register custom agent classes and instantiate them by name.
    This enables easy extension of the benchmark with new agent implementations.

    Attributes:
        _agents: Dictionary mapping agent names to classes.

    """

    _instance: tp.ClassVar[AgentRegistry | None] = None
    _agents: dict[str, type]

    def __init__(self) -> None:
        """Initialize an empty agent registry."""
        self._agents = {}

    @classmethod
    def get_instance(cls) -> AgentRegistry:
        """Get the singleton registry instance.

        Returns:
            The global AgentRegistry instance.

        """
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._register_default_agents()
        return cls._instance

    def register(
        self,
        name: str,
        agent_class: type,
        overwrite: bool = False,
    ) -> None:
        """Register an agent class.

        Args:
            name: Name to register the agent under.
            agent_class: Agent class (must inherit from BaseAgent).
            overwrite: Whether to overwrite existing registration.

        Raises:
            ValueError: If name exists and overwrite is False.
            TypeError: If agent_class doesn't inherit from BaseAgent.

        """
        if name in self._agents and not overwrite:
            raise ValueError(f"Agent '{name}' already registered. Use overwrite=True to replace.")

        if not (isinstance(agent_class, type) and issubclass(agent_class, BaseAgent)):
            raise TypeError(f"Agent class must inherit from BaseAgent, got {agent_class}")

        normalized = name.lower().replace("-", "_")
        self._agents[normalized] = agent_class

    def get(self, name: str) -> type:
        """Get an agent class by name.

        Args:
            name: Registered agent name.

        Returns:
            The agent class.

        Raises:
            KeyError: If agent not found.

        """
        normalized = name.lower().replace("-", "_")
        if normalized not in self._agents:
            available = list(self._agents.keys())
            raise KeyError(f"Agent '{name}' not found. Available: {available}")
        return self._agents[normalized]

    def create(
        self,
        name: str,
        **kwargs: tp.Any,
    ) -> BaseAgent:
        """Create an agent instance by name.

        Args:
            name: Registered agent name.
            **kwargs: Arguments to pass to agent constructor.

        Returns:
            Instantiated agent.

        """
        agent_class = self.get(name)
        return agent_class(**kwargs)

    def list_agents(self) -> list[str]:
        """List all registered agent names.

        Returns:
            List of agent names.

        """
        return list(self._agents.keys())

    def _register_default_agents(self) -> None:
        """Register the built-in agents."""
        pass


class BaseAgent(ABC):
    """Abstract base class for MTG agents.

    All agents in the benchmark must inherit from this class and implement
    the required methods for action selection and training.

    Attributes:
        name: Human-readable agent name.
        deterministic: Whether agent acts deterministically.

    """

    def __init__(self, name: str, deterministic: bool = False) -> None:
        """Initialize the base agent.

        Args:
            name: Agent identifier.
            deterministic: Whether to act deterministically.

        """
        self.name = name
        self.deterministic = deterministic

    @abstractmethod
    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        info: dict[str, tp.Any] | None = None,
    ) -> int:
        """Select an action given the current observation.

        Args:
            observation: Current state observation vector.
            action_mask: Binary mask of legal actions.
            info: Optional additional info from environment.

        Returns:
            Selected action index.

        """
        pass

    def learn(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        info: dict[str, tp.Any] | None = None,
    ) -> dict[str, float] | None:
        """Update the agent from a transition.

        Args:
            observation: State before action.
            action: Action taken.
            reward: Reward received.
            next_observation: State after action.
            done: Whether episode ended.
            info: Optional additional info.

        Returns:
            Optional dictionary of training metrics.

        """
        return None

    def reset(self) -> None:  # noqa: B027
        """Reset agent state for a new episode."""
        pass

    def save(self, path: str | Path) -> None:  # noqa: B027
        """Save agent to disk.

        Args:
            path: Path to save location.

        """
        pass

    def load(self, path: str | Path) -> None:  # noqa: B027
        """Load agent from disk.

        Args:
            path: Path to saved agent.

        """
        pass

    def get_action_probabilities(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
    ) -> np.ndarray:
        """Get probability distribution over actions.

        Args:
            observation: Current state observation.
            action_mask: Binary mask of legal actions.

        Returns:
            Probability distribution over actions.

        """
        n_actions = len(action_mask)
        legal_actions = np.where(action_mask > 0)[0]
        probs = np.zeros(n_actions)
        if len(legal_actions) > 0:
            probs[legal_actions] = 1.0 / len(legal_actions)
        return probs

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            Agent description string.

        """
        return f"{self.__class__.__name__}(name='{self.name}')"

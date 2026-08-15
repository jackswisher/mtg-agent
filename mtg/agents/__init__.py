"""Agent implementations for MTG-Causal-RL.

This module provides agent types for the benchmark:
- RandomAgent: Uniform random baseline
- GreedyAggroAgent: Greedy aggressive strategy baseline
- PPOAgent: Model-free RL agent
- CausalAgent: Causal RL agent (our contribution)

Users can also register custom agents using the AgentRegistry.
"""

from mtg.agents.base.base import AgentRegistry, BaseAgent
from mtg.agents.causal.causal_agent import CausalAgent
from mtg.agents.causal.cgfa_agent import CGFAAgent, CGFAScalarOnlyAgent
from mtg.agents.heuristics.control_agent import ControlAgent
from mtg.agents.heuristics.convoke_aggro_agent import ConvokeAggroAgent
from mtg.agents.heuristics.greedy_aggro_agent import GreedyAggroAgent
from mtg.agents.heuristics.midrange_agent import MidrangeAgent
from mtg.agents.heuristics.ramp_agent import RampAgent
from mtg.agents.heuristics.random_agent import RandomAgent
from mtg.agents.reinforcement_learning.ppo_agent import PPOAgent


def _register_default_agents() -> None:
    """Register all built-in agents with the registry."""
    registry = AgentRegistry.get_instance()
    registry.register("random", RandomAgent, overwrite=True)
    registry.register("greedy_aggro", GreedyAggroAgent, overwrite=True)
    registry.register("control", ControlAgent, overwrite=True)
    registry.register("midrange", MidrangeAgent, overwrite=True)
    registry.register("ramp", RampAgent, overwrite=True)
    registry.register("convoke_aggro", ConvokeAggroAgent, overwrite=True)
    registry.register("ppo", PPOAgent, overwrite=True)
    registry.register("causal", CausalAgent, overwrite=True)
    registry.register("cgfa", CGFAAgent, overwrite=True)
    # Architecture-matched scalar-only ablation: same network capacity
    # as ``cgfa`` (per-factor heads + gate MLP still constructed) but
    # with every CGFA learning signal pinned to zero so improvements
    # over plain ``ppo`` cannot be attributed to "more parameters".
    registry.register("cgfa_scalar_only", CGFAScalarOnlyAgent, overwrite=True)


_register_default_agents()


# Canonical mapping from a player deck archetype to the heuristic agent that
# is strategically appropriate for it. Used by training/evaluation/gameplay to
# pair a deck with a sensible non-RL opponent (e.g. running ``greedy_aggro``
# against an Azorius Control deck makes no strategic sense).
DECK_TO_HEURISTIC: dict[str, str] = {
    "mono_red_aggro": "greedy_aggro",
    "boros_convoke": "convoke_aggro",
    "azorius_control": "control",
    "dimir_midrange": "midrange",
    "domain_ramp": "ramp",
}


def heuristic_for_deck(deck: str) -> str | None:
    """Return the canonical heuristic agent name for a player deck.

    Args:
        deck: Deck archetype name (e.g. ``"mono_red_aggro"``).

    Returns:
        The matched heuristic agent name (e.g. ``"greedy_aggro"``), or
        ``None`` if no canonical heuristic is defined for this deck.

    Example:
        >>> heuristic_for_deck("mono_red_aggro")
        'greedy_aggro'
        >>> heuristic_for_deck("azorius_control")
        'control'

    """
    return DECK_TO_HEURISTIC.get(deck)


def register_agent(name: str, agent_class: type, overwrite: bool = False) -> None:
    """Register a custom agent class.

    Args:
        name: Name to register the agent under.
        agent_class: Agent class (must inherit from BaseAgent).
        overwrite: Whether to overwrite existing registration.

    Example:
        >>> from mtg.agents import BaseAgent, register_agent
        >>> class MyAgent(BaseAgent):
        ...     def select_action(self, obs, mask, info=None):
        ...         return 0  # Always pass
        >>> register_agent("my_agent", MyAgent)

    """
    AgentRegistry.get_instance().register(name, agent_class, overwrite)


def get_agent(name: str, **kwargs) -> BaseAgent:
    """Create an agent instance by name.

    Args:
        name: Registered agent name.
        **kwargs: Arguments to pass to agent constructor.

    Returns:
        Instantiated agent.

    Example:
        >>> agent = get_agent("random", seed=42)
        >>> agent = get_agent("greedy_aggro", aggression=0.8)

    """
    return AgentRegistry.get_instance().create(name, **kwargs)


def list_agents() -> list[str]:
    """List all registered agent names.

    Returns:
        List of available agent names.

    Example:
        >>> list_agents()
        ['random', 'greedy_aggro', 'ppo', 'causal']

    """
    return AgentRegistry.get_instance().list_agents()


__all__ = [
    "DECK_TO_HEURISTIC",
    "BaseAgent",
    "AgentRegistry",
    "RandomAgent",
    "GreedyAggroAgent",
    "ControlAgent",
    "MidrangeAgent",
    "RampAgent",
    "ConvokeAggroAgent",
    "PPOAgent",
    "CausalAgent",
    "CGFAAgent",
    "CGFAScalarOnlyAgent",
    "register_agent",
    "get_agent",
    "list_agents",
    "heuristic_for_deck",
]

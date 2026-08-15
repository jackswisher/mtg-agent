"""MTG-Causal-RL: A Causal Reinforcement Learning Benchmark for Magic: The Gathering."""

__version__ = "0.1.0"

from mtg.causal import CausalSCM
from mtg.env import MTGEnv

__all__ = ["MTGEnv", "CausalSCM", "__version__"]

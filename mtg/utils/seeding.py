"""Seeding utilities for reproducibility.

This module provides utilities for setting random seeds across
all relevant libraries for reproducible experiments.
"""

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set random seed for all libraries.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Set PyTorch seed if available
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # For deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Get a numpy random generator.

    Args:
        seed: Optional seed for the generator.

    Returns:
        Numpy random generator.
    """
    return np.random.default_rng(seed)


def get_python_rng(seed: int | None = None) -> random.Random:
    """Get a Python random generator.

    Args:
        seed: Optional seed for the generator.

    Returns:
        Python random generator.
    """
    rng = random.Random()
    if seed is not None:
        rng.seed(seed)
    return rng


class SeedSequence:
    """Generator for reproducible seed sequences.

    Useful for generating different seeds for multiple experiments
    while maintaining reproducibility.
    """

    def __init__(self, base_seed: int = 42):
        """Initialize the seed sequence.

        Args:
            base_seed: Base seed for the sequence.
        """
        self.base_seed = base_seed
        self._counter = 0
        self._rng = np.random.default_rng(base_seed)

    def next(self) -> int:
        """Get the next seed in the sequence.

        Returns:
            Next seed value.
        """
        seed = int(self._rng.integers(0, 2**31))
        self._counter += 1
        return seed

    def get_seeds(self, n: int) -> list[int]:
        """Get n seeds from the sequence.

        Args:
            n: Number of seeds to generate.

        Returns:
            List of seeds.
        """
        return [self.next() for _ in range(n)]

    def reset(self) -> None:
        """Reset the sequence to the beginning."""
        self._rng = np.random.default_rng(self.base_seed)
        self._counter = 0

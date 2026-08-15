"""Shared pytest fixtures and configuration for the MTG-Causal-RL test suite.

Lives at the root of the ``tests/`` package so every subdirectory
(``env``, ``agents``, ``causal``, ``cgfa``, ``training``, ``research``)
inherits the fixtures and the matplotlib non-interactive backend.
"""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Force a non-interactive matplotlib backend for the whole test session.

    Several CGFA / research tests render figures (calibration plot, case
    study, headline comparison).  Without the Agg backend they would try
    to open a GUI window during ``pytest`` runs on headless CI hosts.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
    except ImportError:
        # matplotlib is optional for the env / agents tests; the CGFA
        # and research tests already importorskip on it explicitly.
        pass


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_env_kwargs() -> dict:
    """Cheap MTGEnv kwargs for fast environment-level unit tests.

    Reduces the per-game compute by capping turns and forcing
    auto-resolve so each step does not recurse through full
    combat/targeting decisions.  Use this whenever the test only needs
    the env to run a few steps; do not use it for training-loop
    correctness checks.
    """
    return {
        "deck_archetype": "mono_red_aggro",
        "opponent_archetype": "mono_red_aggro",
        "max_turns": 10,
        "max_steps_per_episode": 200,
        "seed": 0,
    }


@pytest.fixture
def deterministic_seed() -> int:
    """Fixed seed for tests that must be byte-identical across runs."""
    return 0

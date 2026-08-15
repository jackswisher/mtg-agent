"""Experiment configuration module.

This module provides experiment configurations for reproducible
research on the MTG-Causal-RL benchmark.

Programmatic ablation/transfer suites live alongside the YAML configs:

* :func:`mtg.experiments.ablation.default_cgfa_ablation_variants`
  defines the canonical 6-point CGFA-PPO ablation suite used by
  ``scripts/research/ablation_sweep.py``.
* :func:`mtg.experiments.transfer.default_transfer_split` defines the
  canonical train/held-out opponent split used by
  ``scripts/research/transfer_eval.py``.
"""

from pathlib import Path

from mtg.experiments.ablation import (
    DEFAULT_VARIANT_NAMES,
    AblationVariant,
    default_cgfa_ablation_variants,
    load_ablation_variants,
    save_ablation_variants,
    variants_by_name,
)

EXPERIMENTS_DIR = Path(__file__).parent

CONFIG_PATH = EXPERIMENTS_DIR / "config.yaml"
ABLATIONS_PATH = EXPERIMENTS_DIR / "ablations.yaml"
GENERALIZATION_PATH = EXPERIMENTS_DIR / "generalization.yaml"


def get_config_path(name: str) -> Path:
    """Get path to a config file.

    Args:
        name: Config name (config, ablations, generalization).

    Returns:
        Path to config file.
    """
    configs = {
        "config": CONFIG_PATH,
        "main": CONFIG_PATH,
        "ablations": ABLATIONS_PATH,
        "generalization": GENERALIZATION_PATH,
    }

    if name not in configs:
        raise ValueError(f"Unknown config: {name}. Available: {list(configs.keys())}")

    return configs[name]


__all__ = [
    "ABLATIONS_PATH",
    "CONFIG_PATH",
    "DEFAULT_VARIANT_NAMES",
    "EXPERIMENTS_DIR",
    "GENERALIZATION_PATH",
    "AblationVariant",
    "default_cgfa_ablation_variants",
    "get_config_path",
    "load_ablation_variants",
    "save_ablation_variants",
    "variants_by_name",
]

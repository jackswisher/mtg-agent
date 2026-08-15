"""Ablation specifications for CGFA-PPO experimental validation.

This module defines the canonical 6-point ablation suite that validates
the CGFA-PPO contribution by isolating its components:

1. ``ppo``: vanilla MaskablePPO (no causal anything).
2. ``causal``: CausalAgent (PPO + auxiliary CWM losses but no
   per-factor advantage decomposition).
3. ``cgfa_scalar_only``: architecture-matched scalar PPO ablation. The
   per-factor heads and gate MLP are identical to ``cgfa_full``, but
   every CGFA loss coefficient is pinned to zero.  Controls for
   "more parameters" so improvements over ``ppo`` cannot be attributed
   to extra network capacity.
4. ``cgfa_no_gate``: CGFA with the residual gate disabled
   (``learnable_gate=False``, ``cgfa_alpha=1.0``).  Shows whether
   per-state mixing matters.
5. ``cgfa_no_cal``: CGFA with intervention calibration disabled
   (``intervention_calibration_coef=0.0``).  Shows whether SCM
   alignment of A_k matters.
6. ``cgfa_full``: CGFA-PPO with all components enabled.

Each variant maps onto an existing agent type registered in
:data:`mtg.training.train.AGENT_REGISTRY` plus a dictionary of
``agent_kwargs`` forwarded to the agent constructor.  The ablation
suite reuses the entire training/evaluation pipeline with no
per-variant code changes; the only thing that differs is the kwargs
dict.

Usage::

    from mtg.experiments.ablation import default_cgfa_ablation_variants

    for variant in default_cgfa_ablation_variants():
        cfg = TrainingConfig(
            agent_type=variant.agent_type,
            agent_kwargs=variant.agent_kwargs,
            ...
        )
        Trainer(cfg).setup(env_factory).train()
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Canonical names of the six ablation variants.  Kept ordered so a
# downstream report renders "vanilla -> full CGFA" left-to-right.
DEFAULT_VARIANT_NAMES: tuple[str, ...] = (
    "ppo",
    "causal",
    "cgfa_scalar_only",
    "cgfa_no_gate",
    "cgfa_no_cal",
    "cgfa_full",
)

STRESS_VARIANT_NAMES: tuple[str, ...] = (
    *DEFAULT_VARIANT_NAMES,
    "cgfa_no_scm_init",
    "cgfa_interventional_cal",
)


@dataclass
class AblationVariant:
    """A single agent configuration in the ablation suite.

    Attributes:
        name: Unique identifier for the variant (used for filesystem
            output dirs and figure labels).
        agent_type: Key into :data:`mtg.training.train.AGENT_REGISTRY`.
        agent_kwargs: Keyword arguments forwarded to the agent
            constructor.  Empty dict = use the agent's defaults.
        description: One-line human-readable description for reports.
    """

    name: str
    agent_type: str
    agent_kwargs: dict[str, tp.Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict[str, tp.Any]:
        """Return a YAML-serialisable representation."""
        return {
            "name": self.name,
            "agent_type": self.agent_type,
            "agent_kwargs": dict(self.agent_kwargs),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, tp.Any]) -> AblationVariant:
        """Build an :class:`AblationVariant` from a YAML/JSON dict."""
        return cls(
            name=str(data["name"]),
            agent_type=str(data["agent_type"]),
            agent_kwargs=dict(data.get("agent_kwargs") or {}),
            description=str(data.get("description") or ""),
        )


def default_cgfa_ablation_variants() -> list[AblationVariant]:
    """Return the canonical 6-point CGFA ablation suite.

    The variants are returned in a fixed order so figures/tables render
    consistently across runs.
    """
    return [
        AblationVariant(
            name="ppo",
            agent_type="ppo",
            agent_kwargs={},
            description=(
                "Vanilla MaskablePPO with no causal world model and no "
                "per-factor value decomposition.  Lower bound for the suite."
            ),
        ),
        AblationVariant(
            name="causal",
            agent_type="causal",
            agent_kwargs={},
            description=(
                "PPO + Causal World Model auxiliary losses.  Tests whether "
                "auxiliary causal supervision alone (without factored "
                "advantages) is enough to beat vanilla PPO."
            ),
        ),
        AblationVariant(
            name="cgfa_scalar_only",
            agent_type="cgfa_scalar_only",
            agent_kwargs={},
            description=(
                "Architecture-matched scalar PPO. Per-factor value heads "
                "and the residual-gate MLP are still present in the network, "
                "but every CGFA loss coefficient is pinned to zero "
                "(factor_value_coef=0, intervention_calibration_coef=0, "
                "gate_entropy_coef=0, learnable_gate=False, cgfa_alpha=0).  "
                "Controls for parameter count so any CGFA win cannot be "
                "attributed to extra capacity."
            ),
        ),
        AblationVariant(
            name="cgfa_no_gate",
            agent_type="cgfa",
            agent_kwargs={
                "cgfa_alpha": 1.0,
                "learnable_gate": False,
                "state_conditional_gate": False,
                "intervention_calibration_coef": 0.1,
            },
            description=(
                "CGFA with the residual gate frozen at alpha=1 (pure "
                "factored advantage, no scalar mixing).  Tests whether "
                "per-state advantage mixing is necessary."
            ),
        ),
        AblationVariant(
            name="cgfa_no_cal",
            agent_type="cgfa",
            agent_kwargs={
                "cgfa_alpha": 0.5,
                "learnable_gate": True,
                "state_conditional_gate": True,
                "intervention_calibration_coef": 0.0,
            },
            description=(
                "CGFA with intervention-calibration loss disabled.  Tests "
                "whether SCM-alignment of per-factor advantages "
                "(via Pearson maximisation against eps_k) is necessary."
            ),
        ),
        AblationVariant(
            name="cgfa_full",
            agent_type="cgfa",
            agent_kwargs={
                "cgfa_alpha": 0.5,
                "learnable_gate": True,
                "state_conditional_gate": True,
                "intervention_calibration_coef": 0.1,
            },
            description=(
                "Full CGFA-PPO: per-factor decomposition + state-conditional "
                "residual gate + intervention calibration.  The proposed "
                "method."
            ),
        ),
    ]


def stress_cgfa_ablation_variants() -> list[AblationVariant]:
    """Return the canonical suite plus reviewer-facing CGFA stress tests."""
    return [
        *default_cgfa_ablation_variants(),
        AblationVariant(
            name="cgfa_no_scm_init",
            agent_type="cgfa",
            agent_kwargs={
                "cgfa_alpha": 0.5,
                "learnable_gate": True,
                "state_conditional_gate": True,
                "intervention_calibration_coef": 0.1,
                "init_blend_from_scm": False,
            },
            description=(
                "Full CGFA losses and gate, but factor mixture weights are not "
                "initialised from SCM win-prob weights. Stress-tests whether "
                "the structural prior matters beyond extra heads."
            ),
        ),
        AblationVariant(
            name="cgfa_interventional_cal",
            agent_type="cgfa",
            agent_kwargs={
                "cgfa_alpha": 0.5,
                "learnable_gate": True,
                "state_conditional_gate": True,
                "intervention_calibration_coef": 0.1,
                "calibration_mode": "interventional",
            },
            description=(
                "CGFA with the experimental action-metadata interventional "
                "calibration target. Used as a stress/improvement variant, not "
                "as a silent replacement for the factual CGFA baseline."
            ),
        ),
    ]


def variants_by_name(
    names: tp.Iterable[str],
    *,
    available: tp.Iterable[AblationVariant] | None = None,
) -> list[AblationVariant]:
    """Filter a variant list down to ``names``, preserving order.

    Args:
        names: Variant names to keep.
        available: Source list of variants (defaults to the canonical
            6-point suite).

    Raises:
        KeyError: If any name in ``names`` is not present.
    """
    pool = list(available) if available is not None else default_cgfa_ablation_variants()
    by_name = {v.name: v for v in pool}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise KeyError(f"Unknown ablation variants: {missing}.  Available: {sorted(by_name)}")
    return [by_name[n] for n in names]


def load_ablation_variants(path: str | Path) -> list[AblationVariant]:
    """Load a list of :class:`AblationVariant` from a YAML file.

    The expected schema is::

        variants:
          - name: <str>
            agent_type: <str>
            agent_kwargs: {<key>: <value>, ...}
            description: <str>
          - ...

    A flat list (no top-level ``variants`` key) is also accepted.
    """
    raw = yaml.safe_load(Path(path).read_text())
    items = raw["variants"] if isinstance(raw, dict) and "variants" in raw else raw
    if not isinstance(items, list):
        raise ValueError(f"Expected a list of variants in {path}; got {type(items).__name__}")
    return [AblationVariant.from_dict(item) for item in items]


def save_ablation_variants(
    variants: tp.Iterable[AblationVariant],
    path: str | Path,
) -> None:
    """Serialise ``variants`` to a YAML file (round-trips with ``load``)."""
    payload = {"variants": [v.to_dict() for v in variants]}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False))

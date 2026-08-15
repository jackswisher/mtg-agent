"""Tests for the canonical 6-point CGFA-PPO ablation suite.

Covers:
* ``AblationVariant`` round-trips through dict/YAML.
* ``default_cgfa_ablation_variants`` shape, names, and that each variant
  resolves to an existing agent type in the agent registry.
* ``variants_by_name`` correctness and error handling.
* ``load_ablation_variants`` / ``save_ablation_variants`` round-trip.
* The runner builds ``SweepRun`` objects with the variant's
  ``agent_kwargs`` attached so the trainer forwards them.
* CLI ``--variants ppo cgfa_full`` correctly subsets the suite.
* The architecture-matched ``cgfa_scalar_only`` variant uses the
  ``cgfa_scalar_only`` agent type so the per-factor heads are
  constructed but receive no learning signal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mtg.experiments import (
    DEFAULT_VARIANT_NAMES,
    AblationVariant,
    default_cgfa_ablation_variants,
    load_ablation_variants,
    save_ablation_variants,
    variants_by_name,
)

# ---------------------------------------------------------------------------
# AblationVariant: shape and round-trip
# ---------------------------------------------------------------------------


def test_ablation_variant_round_trips_via_dict() -> None:
    """``from_dict(to_dict(v)) == v`` for an ``AblationVariant``."""
    original = AblationVariant(
        name="cgfa_no_gate",
        agent_type="cgfa",
        agent_kwargs={"learnable_gate": False, "cgfa_alpha": 1.0},
        description="No gate",
    )
    payload = original.to_dict()
    rebuilt = AblationVariant.from_dict(payload)
    assert rebuilt == original


def test_ablation_variant_default_kwargs_is_empty() -> None:
    """Empty agent_kwargs is the default and is still serialisable."""
    v = AblationVariant(name="ppo", agent_type="ppo")
    assert v.agent_kwargs == {}
    assert v.to_dict()["agent_kwargs"] == {}


def test_ablation_variant_from_dict_handles_missing_fields() -> None:
    """``from_dict`` accepts dicts without optional fields."""
    v = AblationVariant.from_dict({"name": "ppo", "agent_type": "ppo"})
    assert v.name == "ppo"
    assert v.agent_type == "ppo"
    assert v.agent_kwargs == {}
    assert v.description == ""


# ---------------------------------------------------------------------------
# default_cgfa_ablation_variants
# ---------------------------------------------------------------------------


def test_default_variants_have_canonical_names_in_order() -> None:
    """The default 6-point suite must match ``DEFAULT_VARIANT_NAMES``."""
    variants = default_cgfa_ablation_variants()
    assert tuple(v.name for v in variants) == DEFAULT_VARIANT_NAMES


def test_default_variants_have_six_distinct_entries() -> None:
    """Suite has exactly 6 variants with unique names."""
    variants = default_cgfa_ablation_variants()
    names = [v.name for v in variants]
    assert len(variants) == 6
    assert len(set(names)) == 6


def test_cgfa_scalar_only_variant_uses_dedicated_agent_type() -> None:
    """The ``cgfa_scalar_only`` variant routes through the matched-arch agent.

    Picking a different ``agent_type`` (e.g. ``"cgfa"`` with kwargs) would
    bypass the hard pinning of CGFA coefficients to zero, opening the
    door to "matched architecture" being subtly different from "all CGFA
    learning signals off".  This test guards that contract.
    """
    by_name = {v.name: v for v in default_cgfa_ablation_variants()}
    assert "cgfa_scalar_only" in by_name
    assert by_name["cgfa_scalar_only"].agent_type == "cgfa_scalar_only"


def test_cgfa_scalar_only_agent_pins_all_cgfa_coefs_to_zero() -> None:
    """Construct the agent and assert every CGFA loss coef / blend is zero."""
    pytest.importorskip("torch")

    from mtg.agents import CGFAScalarOnlyAgent

    agent = CGFAScalarOnlyAgent(observation_dim=64, action_dim=8, seed=0)
    assert agent.cgfa_alpha == 0.0
    assert agent.learnable_gate is False
    assert agent.factor_value_coef == 0.0
    assert agent.intervention_calibration_coef == 0.0
    assert agent.gate_entropy_coef == 0.0


def test_default_variants_resolve_to_registered_agent_types() -> None:
    """Every variant's ``agent_type`` is in the canonical agent registry."""
    from mtg.training.train import AGENT_REGISTRY

    variants = default_cgfa_ablation_variants()
    for v in variants:
        assert (
            v.agent_type in AGENT_REGISTRY
        ), f"variant {v.name!r} references unknown agent_type {v.agent_type!r}"


def test_default_variants_isolate_each_cgfa_component() -> None:
    """The cgfa_no_gate, cgfa_no_cal, cgfa_full variants disable distinct components.

    cgfa_no_gate freezes the gate; cgfa_no_cal disables calibration; cgfa_full enables both.
    """
    by_name = {v.name: v for v in default_cgfa_ablation_variants()}

    no_gate = by_name["cgfa_no_gate"]
    no_cal = by_name["cgfa_no_cal"]
    full = by_name["cgfa_full"]

    assert no_gate.agent_kwargs["learnable_gate"] is False
    assert no_gate.agent_kwargs.get("intervention_calibration_coef", 0.0) > 0.0

    assert no_cal.agent_kwargs["learnable_gate"] is True
    assert no_cal.agent_kwargs["intervention_calibration_coef"] == 0.0

    assert full.agent_kwargs["learnable_gate"] is True
    assert full.agent_kwargs["intervention_calibration_coef"] > 0.0


# ---------------------------------------------------------------------------
# variants_by_name
# ---------------------------------------------------------------------------


def test_variants_by_name_filters_and_preserves_order() -> None:
    """The returned list matches the requested order, not the suite order."""
    selected = variants_by_name(["cgfa_full", "ppo"])
    assert [v.name for v in selected] == ["cgfa_full", "ppo"]


def test_variants_by_name_raises_on_unknown_variant() -> None:
    """Asking for a non-existent variant raises a clear error."""
    with pytest.raises(KeyError, match="Unknown ablation variants"):
        variants_by_name(["does_not_exist"])


def test_variants_by_name_uses_provided_pool_when_given() -> None:
    """The ``available`` keyword overrides the default suite."""
    custom = [
        AblationVariant(name="alpha", agent_type="ppo"),
        AblationVariant(name="beta", agent_type="ppo"),
    ]
    selected = variants_by_name(["beta"], available=custom)
    assert [v.name for v in selected] == ["beta"]


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------


def test_save_and_load_variants_round_trip(tmp_path: Path) -> None:
    """``load_ablation_variants(save_ablation_variants(...))`` returns the input."""
    variants = default_cgfa_ablation_variants()
    yaml_path = tmp_path / "variants.yaml"
    save_ablation_variants(variants, yaml_path)
    assert yaml_path.exists()

    loaded = load_ablation_variants(yaml_path)
    assert len(loaded) == len(variants)
    for src, dst in zip(variants, loaded, strict=True):
        assert src.name == dst.name
        assert src.agent_type == dst.agent_type
        assert src.agent_kwargs == dst.agent_kwargs
        assert src.description == dst.description


def test_load_variants_accepts_flat_list(tmp_path: Path) -> None:
    """A YAML file that's a bare list (no ``variants:`` key) is also accepted."""
    yaml_path = tmp_path / "flat.yaml"
    yaml_path.write_text("- {name: a, agent_type: ppo}\n- {name: b, agent_type: ppo}\n")
    loaded = load_ablation_variants(yaml_path)
    assert [v.name for v in loaded] == ["a", "b"]


def test_load_variants_rejects_non_list(tmp_path: Path) -> None:
    """A scalar root is not a valid variants file."""
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text("just a string\n")
    with pytest.raises(ValueError, match="Expected a list"):
        load_ablation_variants(yaml_path)


# ---------------------------------------------------------------------------
# The bundled YAML matches the programmatic suite (single source of truth)
# ---------------------------------------------------------------------------


def test_bundled_ablations_yaml_matches_programmatic_suite() -> None:
    """``mtg/experiments/ablations.yaml`` mirrors ``default_cgfa_ablation_variants``."""
    from mtg.experiments import ABLATIONS_PATH

    on_disk = load_ablation_variants(ABLATIONS_PATH)
    in_code = default_cgfa_ablation_variants()
    assert [v.name for v in on_disk] == [v.name for v in in_code]
    for d, c in zip(on_disk, in_code, strict=True):
        assert d.agent_type == c.agent_type
        assert d.agent_kwargs == c.agent_kwargs


# ---------------------------------------------------------------------------
# Runner: variants -> SweepRuns
# ---------------------------------------------------------------------------


def test_runner_builds_sweep_runs_with_variant_kwargs() -> None:
    """Each ``SweepRun`` carries the variant's ``agent_kwargs``."""
    from scripts.research.ablation_sweep import _build_variant_runs

    variant = AblationVariant(
        name="cgfa_no_cal",
        agent_type="cgfa",
        agent_kwargs={"intervention_calibration_coef": 0.0},
    )
    runs = _build_variant_runs(
        variant,
        player_decks=["mono_red_aggro"],
        seeds=[42, 123],
        opponents=["azorius_control"],
    )
    assert len(runs) == 2
    for r in runs:
        assert r.agent == "cgfa"
        assert r.player_deck == "mono_red_aggro"
        assert r.opponents == ["azorius_control"]
        assert r.agent_kwargs == {"intervention_calibration_coef": 0.0}
        # Output dirs encode the variant + deck + seed for filesystem clarity.
        assert r.output_dir.startswith("cgfa_no_cal__")


def test_runner_resolve_variants_subsets_default_suite() -> None:
    """``--variants ppo cgfa_full`` returns exactly those two in order."""
    from scripts.research.ablation_sweep import _resolve_variants

    selected = _resolve_variants(["ppo", "cgfa_full"], variants_yaml=None)
    assert [v.name for v in selected] == ["ppo", "cgfa_full"]


def test_runner_resolve_variants_returns_all_when_keyword_all() -> None:
    """``--variants all`` returns the full suite."""
    from scripts.research.ablation_sweep import _resolve_variants

    selected = _resolve_variants(["all"], variants_yaml=None)
    assert [v.name for v in selected] == list(DEFAULT_VARIANT_NAMES)


def test_runner_resolve_variants_loads_from_yaml(tmp_path: Path) -> None:
    """``--variants-yaml`` overrides the default suite source."""
    from scripts.research.ablation_sweep import _resolve_variants

    yaml_path = tmp_path / "custom.yaml"
    save_ablation_variants(
        [
            AblationVariant(name="custom_a", agent_type="ppo"),
            AblationVariant(name="custom_b", agent_type="cgfa"),
        ],
        yaml_path,
    )
    selected = _resolve_variants(["all"], variants_yaml=yaml_path)
    assert [v.name for v in selected] == ["custom_a", "custom_b"]


# ---------------------------------------------------------------------------
# TrainingConfig propagates agent_kwargs through to the engine config
# ---------------------------------------------------------------------------


def test_cli_training_config_propagates_agent_kwargs_to_engine() -> None:
    """CLI ``TrainingConfig.agent_kwargs`` reaches ``EngineTrainingConfig``."""
    from mtg.utils.interactive import TrainingConfig as CLITrainingConfig

    cli_cfg = CLITrainingConfig(
        agent_type="cgfa",
        player_deck="mono_red_aggro",
        opponent_deck="azorius_control",
        agent_kwargs={"learnable_gate": False, "cgfa_alpha": 0.7},
    )
    engine_cfg = cli_cfg.to_engine_config()
    assert engine_cfg.agent_type == "cgfa"
    assert engine_cfg.agent_kwargs == {
        "learnable_gate": False,
        "cgfa_alpha": 0.7,
    }

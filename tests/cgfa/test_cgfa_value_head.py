"""Tests for the FactorSpec + multi-head causal value module."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mtg.agents.reinforcement_learning.cgfa import (  # noqa: E402
    DEFAULT_FACTOR_NAMES,
    CausalValueHead,
    FactorSpec,
    extract_factor_values,
    factor_blend_from_scm_weights,
)
from mtg.causal.scm import SCMWeights  # noqa: E402

# ---------------------------------------------------------------------------
# FactorSpec
# ---------------------------------------------------------------------------


def test_factor_spec_default_dimensions() -> None:
    """Default FactorSpec has K = 6 factors with valid blend/scale shapes."""
    spec = FactorSpec()
    assert spec.n_factors == len(DEFAULT_FACTOR_NAMES)
    assert spec.blend_init.shape == (spec.n_factors,)
    assert spec.scale.shape == (spec.n_factors,)


def test_factor_spec_validates_blend_shape() -> None:
    """Constructing a FactorSpec with mismatched blend length raises."""
    with pytest.raises(ValueError):
        FactorSpec(blend_init=np.ones(2, dtype=np.float32))


def test_factor_spec_normalises_rewards_by_scale() -> None:
    """normalised_rewards divides element-wise and ignores zero scales."""
    scale = np.array([1.0, 2.0, 0.0, 4.0, 0.5, 1.0], dtype=np.float32)
    spec = FactorSpec(scale=scale)
    raw = np.array([2.0, 4.0, 5.0, 8.0, 1.0, 0.5], dtype=np.float32)
    out = spec.normalised_rewards(raw)
    expected = np.array([2.0, 2.0, 5.0, 2.0, 2.0, 0.5], dtype=np.float32)
    np.testing.assert_allclose(out, expected)


def test_extract_factor_values_handles_missing_keys() -> None:
    """Missing causal-variable keys default to 0 instead of raising."""
    spec = FactorSpec()
    out = extract_factor_values({"card_adv": 3.0, "tempo": -0.5}, spec)
    assert out.shape == (spec.n_factors,)
    idx = spec.names.index("card_adv")
    assert out[idx] == pytest.approx(3.0)
    idx = spec.names.index("tempo")
    assert out[idx] == pytest.approx(-0.5)
    # Untouched factors are zero
    idx = spec.names.index("life_buffer")
    assert out[idx] == pytest.approx(0.0)


def test_extract_factor_values_handles_none() -> None:
    """A None causal_variables dict yields all zeros."""
    spec = FactorSpec()
    out = extract_factor_values(None, spec)
    np.testing.assert_array_equal(out, np.zeros(spec.n_factors, dtype=np.float32))


def test_factor_blend_from_scm_weights_matches_winprob_weights() -> None:
    """Default blend init mirrors the SCMWeights values per factor name."""
    weights = SCMWeights()
    blend = factor_blend_from_scm_weights(weights)
    expected = np.array(
        [
            weights.card_adv_weight,
            weights.board_press_weight,
            weights.tempo_weight,
            weights.life_buffer_weight,
            weights.threat_density_weight,
            weights.removal_avail_weight,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(blend, expected)


# ---------------------------------------------------------------------------
# CausalValueHead
# ---------------------------------------------------------------------------


def test_value_head_returns_per_factor_tensor_only() -> None:
    """The head returns only ``V_factors [B, K]`` (no scalar branch)."""
    head = CausalValueHead(latent_dim=8, n_factors=4)
    latent = torch.randn(5, 8)
    v_factors = head(latent)
    assert isinstance(v_factors, torch.Tensor)
    assert v_factors.shape == (5, 4)


def test_value_head_handles_unbatched_input() -> None:
    """A 1D latent vector is treated as a batch of size 1."""
    head = CausalValueHead(latent_dim=8, n_factors=3)
    latent = torch.randn(8)
    v_factors = head(latent)
    assert v_factors.shape == (1, 3)


def test_value_head_no_longer_constructs_dead_scalar_head() -> None:
    """``scalar_head`` must NOT exist as a child module of CausalValueHead.

    The scalar value branch lives on the policy itself
    (``policy.value_net``); duplicating it inside the value head would
    add parameters and compute for no learning signal because the
    duplicate ``v_scalar`` would be discarded by every consumer.
    """
    head = CausalValueHead(latent_dim=8, n_factors=4)
    assert not hasattr(head, "scalar_head"), (
        "CausalValueHead.scalar_head must not exist; "
        "consumers should use SB3's `policy.value_net` for V_scalar."
    )
    # And nothing in the named parameters should mention scalar_head.
    for name, _ in head.named_parameters():
        assert "scalar_head" not in name, f"Found stray scalar_head parameter: {name}"


def test_value_head_aggregate_is_softmax_weighted_sum() -> None:
    """aggregate() returns sum_k softmax(blend)_k * V_k(s)."""
    blend = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    head = CausalValueHead(latent_dim=4, n_factors=3, blend_init=blend)
    weights = head.mixture_weights.detach().numpy()
    np.testing.assert_allclose(
        weights,
        np.exp(blend) / np.exp(blend).sum(),
        rtol=1e-5,
    )

    v_factors = torch.tensor([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]], dtype=torch.float32)
    agg = head.aggregate(v_factors).detach().numpy()
    w = weights
    expected = np.array(
        [
            w[0] * 0.1 + w[1] * 0.2 + w[2] * 0.3,
            w[0] * 1.0 + w[1] * 2.0 + w[2] * 3.0,
        ]
    )
    np.testing.assert_allclose(agg, expected, rtol=1e-5)


def test_value_head_blend_is_learnable_parameter() -> None:
    """factor_blend appears in parameters() and supports gradient updates."""
    head = CausalValueHead(latent_dim=4, n_factors=3)
    assert head.factor_blend.requires_grad
    params = list(head.parameters())
    assert any(p is head.factor_blend for p in params)

    latent = torch.randn(2, 4, requires_grad=False)
    v_factors = head(latent)
    loss = head.aggregate(v_factors).sum()
    loss.backward()
    assert head.factor_blend.grad is not None
    assert head.factor_blend.grad.shape == (3,)


def test_value_head_rejects_wrong_blend_shape() -> None:
    """Mismatched blend_init length raises immediately."""
    with pytest.raises(ValueError):
        CausalValueHead(latent_dim=4, n_factors=3, blend_init=np.zeros(5, dtype=np.float32))


def test_value_head_initialised_from_scm_weights_matches_blend() -> None:
    """Initialising via SCM weights stores them in the parameter."""
    spec_blend = factor_blend_from_scm_weights(SCMWeights())
    head = CausalValueHead(
        latent_dim=4,
        n_factors=len(spec_blend),
        blend_init=spec_blend,
    )
    np.testing.assert_allclose(
        head.factor_blend.detach().numpy(),
        spec_blend,
        rtol=1e-6,
    )


# ---------------------------------------------------------------------------
# Residual gate
# ---------------------------------------------------------------------------


def test_value_head_state_conditional_gate_returns_per_state_values() -> None:
    """State-conditional gate yields a (batch,) tensor in (0, 1)."""
    head = CausalValueHead(
        latent_dim=8,
        n_factors=3,
        residual_gate_init=0.5,
        state_conditional_gate=True,
    )
    latent = torch.randn(7, 8)
    gate = head.compute_gate(latent)
    assert gate.shape == (7,)
    assert th_in_unit_open_interval(gate)


def test_value_head_state_conditional_gate_initialises_at_residual_gate_init() -> None:
    """At init, gate(any latent) ~= residual_gate_init since the final layer is bias-only."""
    head = CausalValueHead(
        latent_dim=8,
        n_factors=3,
        residual_gate_init=0.7,
        state_conditional_gate=True,
    )
    latent = torch.randn(16, 8) * 5.0  # large magnitude so the bias dominates
    gate = head.compute_gate(latent)
    np.testing.assert_allclose(
        gate.detach().numpy(),
        np.full((16,), 0.7, dtype=np.float32),
        atol=1e-5,
    )


def test_value_head_scalar_gate_initialises_at_residual_gate_init() -> None:
    """Non-state-conditional gate matches residual_gate_init at init."""
    head = CausalValueHead(
        latent_dim=8,
        n_factors=3,
        residual_gate_init=0.3,
        state_conditional_gate=False,
    )
    latent = torch.randn(5, 8)
    gate = head.compute_gate(latent)
    assert gate.shape == (5,)
    np.testing.assert_allclose(
        gate.detach().numpy(),
        np.full((5,), 0.3, dtype=np.float32),
        atol=1e-5,
    )


def test_value_head_scalar_gate_is_learnable() -> None:
    """The non-state-conditional gate logit appears in parameters and gets a gradient."""
    head = CausalValueHead(
        latent_dim=4,
        n_factors=3,
        residual_gate_init=0.5,
        state_conditional_gate=False,
    )
    assert head.residual_gate_logit.requires_grad
    latent = torch.randn(3, 4)
    gate = head.compute_gate(latent)
    gate.sum().backward()
    assert head.residual_gate_logit.grad is not None
    assert head.residual_gate_logit.grad.shape == ()


def test_value_head_state_conditional_gate_is_learnable() -> None:
    """The state-conditional gate's MLP receives gradients."""
    head = CausalValueHead(
        latent_dim=4,
        n_factors=3,
        residual_gate_init=0.5,
        state_conditional_gate=True,
    )
    latent = torch.randn(3, 4)
    gate = head.compute_gate(latent)
    gate.sum().backward()
    final_layer = head.gate_net[-1]
    assert final_layer.bias.grad is not None
    assert final_layer.bias.grad.shape == (1,)
    # The first linear layer should receive a non-zero gradient too,
    # because its outputs feed into a tanh that is non-saturated near
    # zero and into the bias-only final layer (we manually zeroed the
    # final layer's *weight*, but gradients still flow through tanh).
    first_layer = head.gate_net[0]
    assert first_layer.weight.grad is not None
    assert first_layer.weight.grad.shape == first_layer.weight.shape


def th_in_unit_open_interval(x: torch.Tensor) -> bool:  # type: ignore[name-defined]
    """Helper: assert all values of x are in (0, 1)."""
    return bool((x > 0).all().item() and (x < 1).all().item())

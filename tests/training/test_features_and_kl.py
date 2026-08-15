"""Tests for Phase B additions: set encoder, adaptive KL, value clip.

These are small, self-contained unit tests that do not require a full
training run.  They validate shape / permutation / behaviour contracts
so regressions surface without burning a smoke-train budget.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gym_spaces = pytest.importorskip("gymnasium.spaces")

from mtg.agents.reinforcement_learning.features import (  # noqa: E402
    MTGFeaturesExtractor,
    ObservationSplitter,
    ZonePooler,
    zone_mask,
)
from mtg.training.callbacks import AdaptiveKLCallback  # noqa: E402


@pytest.fixture
def splitter() -> ObservationSplitter:
    """Default ObservationSplitter fixture."""
    return ObservationSplitter()


def _make_fake_obs(batch: int, splitter: ObservationSplitter) -> torch.Tensor:
    flat = splitter.expected_flat_dim
    return torch.randn(batch, flat)


def test_splitter_shapes_match_config(splitter: ObservationSplitter) -> None:
    """Splitter returns the expected per-zone shapes for a batched obs."""
    obs = _make_fake_obs(4, splitter)
    out = splitter(obs)
    assert out["game_state"].shape == (4, splitter.game_state_dim)
    assert out["hand"].shape == (4, splitter.hand_size, splitter.card_dim)
    assert out["bf_self"].shape == (
        4,
        splitter.battlefield_size,
        splitter.card_dim,
    )
    assert out["bf_opp"].shape == (
        4,
        splitter.battlefield_size,
        splitter.card_dim,
    )
    assert out["gy_self"].shape == (
        4,
        splitter.graveyard_size,
        splitter.card_dim,
    )
    assert out["gy_opp"].shape == (
        4,
        splitter.graveyard_size,
        splitter.card_dim,
    )


def test_zone_mask_flags_empty_slots(splitter: ObservationSplitter) -> None:
    """All-zero slots should be flagged by the zone mask."""
    obs = _make_fake_obs(2, splitter)
    zones = splitter(obs)
    hand = zones["hand"].clone()
    hand[0, 3, :] = 0.0
    mask = zone_mask(hand)
    assert mask[0, 3].item() is True
    assert mask[0, 0].item() is False


def test_zone_pooler_respects_mask() -> None:
    """Fully-masked rows must pool to zero; unmasked rows must not."""
    torch.manual_seed(0)
    pooler = ZonePooler(hidden_dim=16, n_heads=2)
    slots = torch.randn(3, 5, 16)
    # Entire row 2 is empty: pooled output must be all zeros.
    all_empty = torch.tensor(
        [
            [False, False, False, False, False],
            [False, True, True, True, True],
            [True, True, True, True, True],
        ]
    )
    out = pooler(slots, all_empty)
    assert out.shape == (3, 16)
    assert torch.allclose(out[2], torch.zeros_like(out[2]))
    # Non-empty row should be non-zero with high probability.
    assert not torch.allclose(out[0], torch.zeros_like(out[0]))


def test_features_extractor_forward_shape() -> None:
    """Forward pass returns ``[B, features_dim]`` for the expected flat obs."""
    splitter = ObservationSplitter()
    obs_space = gym_spaces.Box(
        low=-np.inf, high=np.inf, shape=(splitter.expected_flat_dim,), dtype=np.float32
    )
    extractor = MTGFeaturesExtractor(obs_space, features_dim=128, hidden_dim=32, n_heads=2)
    obs = _make_fake_obs(4, splitter)
    out = extractor(obs)
    assert out.shape == (4, 128)


def test_features_extractor_hand_permutation_invariance() -> None:
    """Reordering hand slots must not change the output embedding.

    This is the entire point of the set encoder; if it ever starts
    being sensitive to slot index the whole Phase B architectural
    claim in the paper is wrong.
    """
    torch.manual_seed(42)
    splitter = ObservationSplitter()
    obs_space = gym_spaces.Box(
        low=-np.inf, high=np.inf, shape=(splitter.expected_flat_dim,), dtype=np.float32
    )
    extractor = MTGFeaturesExtractor(obs_space, features_dim=64, hidden_dim=32, n_heads=2)
    extractor.eval()

    obs = _make_fake_obs(1, splitter)
    zones_a = splitter(obs)

    # Permute the hand slots and re-flatten to a new obs tensor.
    perm = torch.randperm(splitter.hand_size)
    hand_permuted = zones_a["hand"][:, perm, :]

    def _rebuild(hand: torch.Tensor) -> torch.Tensor:
        parts = [
            zones_a["game_state"],
            hand.reshape(1, -1),
            zones_a["bf_self"].reshape(1, -1),
            zones_a["bf_opp"].reshape(1, -1),
            zones_a["gy_self"].reshape(1, -1),
            zones_a["gy_opp"].reshape(1, -1),
        ]
        return torch.cat(parts, dim=-1)

    obs_orig = _rebuild(zones_a["hand"])
    obs_perm = _rebuild(hand_permuted)

    with torch.no_grad():
        out_orig = extractor(obs_orig)
        out_perm = extractor(obs_perm)

    torch.testing.assert_close(out_orig, out_perm, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Adaptive KL
# ---------------------------------------------------------------------------


class _FakeLogger:
    def __init__(self, kl: float):
        self.name_to_value = {"train/approx_kl": kl}


class _FakeModel:
    def __init__(self, clip: float = 0.2, kl: float = 0.02):
        self.clip_range = clip
        self.learning_rate = 3e-4
        self.logger = _FakeLogger(kl)


def test_adaptive_kl_decreases_clip_when_kl_too_high() -> None:
    """High measured KL shrinks the PPO clip range."""
    cb = AdaptiveKLCallback(
        target_kl=0.02, decrease_factor=0.5, increase_factor=2.0, min_clip=0.01, max_clip=1.0
    )
    cb.model = _FakeModel(clip=0.2, kl=0.1)
    cb._on_rollout_end()
    # clip should have been reduced multiplicatively
    new_clip = cb._current_clip()
    assert new_clip is not None
    assert new_clip < 0.2


def test_adaptive_kl_increases_clip_when_kl_too_low() -> None:
    """Low measured KL relaxes (increases) the PPO clip range."""
    cb = AdaptiveKLCallback(
        target_kl=0.02, decrease_factor=0.5, increase_factor=2.0, min_clip=0.01, max_clip=1.0
    )
    cb.model = _FakeModel(clip=0.2, kl=0.001)
    cb._on_rollout_end()
    new_clip = cb._current_clip()
    assert new_clip is not None
    assert new_clip > 0.2


def test_adaptive_kl_respects_bounds() -> None:
    """Clip updates should always be clamped into ``[min_clip, max_clip]``."""
    cb = AdaptiveKLCallback(
        target_kl=0.02,
        decrease_factor=0.01,
        increase_factor=100.0,
        min_clip=0.05,
        max_clip=0.3,
    )
    m = _FakeModel(clip=0.2, kl=10.0)  # kl extremely high -> pushed to min
    cb.model = m
    cb._on_rollout_end()
    assert abs(cb._current_clip() - 0.05) < 1e-6

    m2 = _FakeModel(clip=0.2, kl=1e-6)  # kl extremely low -> pushed to max
    cb.model = m2
    cb._on_rollout_end()
    assert abs(cb._current_clip() - 0.3) < 1e-6


def test_adaptive_kl_no_change_in_band() -> None:
    """KL inside the target band should leave the clip range untouched."""
    cb = AdaptiveKLCallback(target_kl=0.02, min_clip=0.05, max_clip=0.3)
    cb.model = _FakeModel(clip=0.2, kl=0.02)  # inside the no-op band
    cb._on_rollout_end()
    assert abs(cb._current_clip() - 0.2) < 1e-6

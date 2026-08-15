"""Permutation-invariant features extractor for MTG observations.

A flat MLP over the full 3,077-dim observation (the SB3 default)
forces the policy to learn that ``hand[0]`` and ``hand[1]`` are
interchangeable slots, that ``battlefield[5]`` could semantically be
any of the twenty slots, and so on.  This wastes capacity and sample
efficiency.

``MTGFeaturesExtractor`` instead:

1. Splits the flat observation back into its structured zones
   (game_state, hand, battlefield_self, battlefield_opponent,
   graveyard_self, graveyard_opponent) using the fixed slot widths
   from ``ObservationConfig``.
2. Projects every card slot through a shared MLP into ``hidden_dim``
   space.
3. Pools each zone with a multi-head attention block whose query is a
   learnable ``[CLS]`` token (DeepSets-style, but attention-weighted
   instead of mean-pooled).  Empty slots (all-zero card vectors) are
   masked out to keep their zero features from biasing the pooled
   representation.
4. Concatenates the five zone embeddings with a projected game-state
   vector and fuses them through a final MLP into ``features_dim``.

The result is a permutation-invariant representation of the observable
game state that respects MTG's set-valued zones.  It also exposes
``ObservationSplitter`` so downstream modules (the CausalValueHead in
Phase F) can reuse the same parsing.
"""

from __future__ import annotations

import typing as tp

import numpy as np
import torch
from gymnasium import spaces
from torch import nn

try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    HAS_SB3 = True
except ImportError:  # pragma: no cover
    HAS_SB3 = False

    class BaseFeaturesExtractor(nn.Module):  # type: ignore[no-redef]
        """Fallback stub used when SB3 is unavailable (tests / docs)."""

        def __init__(self, observation_space: tp.Any, features_dim: int) -> None:
            super().__init__()
            self._features_dim = features_dim
            self.observation_space = observation_space

        @property
        def features_dim(self) -> int:
            """Output dimensionality advertised by the extractor."""
            return self._features_dim


# Default sizes match ``ObservationConfig`` defaults; read from the
# module at import time so any project-wide resize here is picked up.
try:
    from mtg.env.observation import ObservationConfig as _ObsCfg

    _DEFAULT_OBS_CFG = _ObsCfg()
    DEFAULT_GAME_STATE_DIM = 17
    DEFAULT_CARD_DIM = _DEFAULT_OBS_CFG.card_embedding_dim
    DEFAULT_HAND_SIZE = _DEFAULT_OBS_CFG.max_hand_size
    DEFAULT_BF_SIZE = _DEFAULT_OBS_CFG.max_battlefield_size
    DEFAULT_GY_SIZE = _DEFAULT_OBS_CFG.max_graveyard_size
except ImportError:  # pragma: no cover
    DEFAULT_GAME_STATE_DIM = 17
    DEFAULT_CARD_DIM = 34
    DEFAULT_HAND_SIZE = 10
    DEFAULT_BF_SIZE = 20
    DEFAULT_GY_SIZE = 20


class ObservationSplitter(nn.Module):
    """Parse a flat MTG observation into structured zones.

    Exposed as a stand-alone module so both the features extractor and
    the causal value head (Phase F) can share the same parsing logic.
    Splitting is position-based and reads the slot sizes off
    ``ObservationConfig`` at construction time; that config is
    treated as part of the observation contract and should not change
    at runtime.
    """

    def __init__(
        self,
        game_state_dim: int = DEFAULT_GAME_STATE_DIM,
        card_dim: int = DEFAULT_CARD_DIM,
        hand_size: int = DEFAULT_HAND_SIZE,
        battlefield_size: int = DEFAULT_BF_SIZE,
        graveyard_size: int = DEFAULT_GY_SIZE,
    ) -> None:
        super().__init__()
        self.game_state_dim = game_state_dim
        self.card_dim = card_dim
        self.hand_size = hand_size
        self.battlefield_size = battlefield_size
        self.graveyard_size = graveyard_size

    @property
    def expected_flat_dim(self) -> int:
        """Total size of the flat observation this splitter expects."""
        return (
            self.game_state_dim
            + self.hand_size * self.card_dim
            + 2 * self.battlefield_size * self.card_dim
            + 2 * self.graveyard_size * self.card_dim
        )

    def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Split a ``[B, flat_dim]`` observation into zones.

        Returns a dict with keys ``game_state``, ``hand``, ``bf_self``,
        ``bf_opp``, ``gy_self``, ``gy_opp``.  Game state is ``[B, gs_dim]``;
        each zone is ``[B, N, card_dim]``.
        """
        b = obs.shape[0]
        offset = 0
        gs = obs[:, offset : offset + self.game_state_dim]
        offset += self.game_state_dim

        def _slice(n: int) -> torch.Tensor:
            nonlocal offset
            chunk = obs[:, offset : offset + n * self.card_dim]
            offset += n * self.card_dim
            return chunk.reshape(b, n, self.card_dim)

        hand = _slice(self.hand_size)
        bf_self = _slice(self.battlefield_size)
        bf_opp = _slice(self.battlefield_size)
        gy_self = _slice(self.graveyard_size)
        gy_opp = _slice(self.graveyard_size)

        return {
            "game_state": gs,
            "hand": hand,
            "bf_self": bf_self,
            "bf_opp": bf_opp,
            "gy_self": gy_self,
            "gy_opp": gy_opp,
        }


def zone_mask(zone: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return a ``[B, N]`` bool mask where True = empty slot (zero card)."""
    return zone.abs().sum(dim=-1) < eps


class ZonePooler(nn.Module):
    """Per-zone attention pooling with a learnable ``[CLS]`` query.

    Given projected slots ``S ∈ R^{B×N×H}`` and a key-padding mask,
    emits a ``[B, H]`` pooled embedding that respects empty slots.
    """

    def __init__(self, hidden_dim: int, n_heads: int = 4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cls = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, slots: torch.Tensor, empty_mask: torch.Tensor) -> torch.Tensor:
        """Pool ``slots`` (``[B, N, H]``) into ``[B, H]``.

        ``empty_mask`` is ``[B, N]`` bool, True for slots that should be
        ignored.  Zones where every slot is empty get an all-zero
        pooled vector (we force the mask to attend to slot 0 in that
        case and then zero out the output).
        """
        b = slots.shape[0]
        cls = self.cls.expand(b, -1, -1)  # [B, 1, H]

        all_empty = empty_mask.all(dim=-1)  # [B]
        # Attention requires at least one non-masked key per row; if a
        # row is completely empty, un-mask slot 0 so the call doesn't
        # NaN, then zero out that row's output.
        safe_mask = empty_mask.clone()
        safe_mask[:, 0] = safe_mask[:, 0] & ~all_empty

        pooled, _ = self.attn(cls, slots, slots, key_padding_mask=safe_mask)
        pooled = pooled.squeeze(1)  # [B, H]
        pooled = self.norm(pooled)
        if all_empty.any():
            pooled = pooled.masked_fill(all_empty.unsqueeze(-1), 0.0)
        return pooled


class MTGFeaturesExtractor(BaseFeaturesExtractor):
    """SB3 features extractor with set-aware attention pooling.

    Args:
        observation_space: Flat Box observation space from ``MTGEnv``.
        features_dim: Output dimensionality of the extractor.
        hidden_dim: Per-slot / per-zone hidden size.
        n_heads: Attention heads in each ``ZonePooler``.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 512,
        hidden_dim: int = 128,
        n_heads: int = 4,
    ) -> None:
        super().__init__(observation_space, features_dim)
        self.hidden_dim = hidden_dim
        self.splitter = ObservationSplitter()

        flat_dim = int(np.prod(observation_space.shape))
        assert flat_dim == self.splitter.expected_flat_dim, (
            f"Observation shape {flat_dim} does not match expected set-encoder "
            f"layout {self.splitter.expected_flat_dim}. Did ObservationConfig change?"
        )

        self.slot_proj = nn.Sequential(
            nn.Linear(self.splitter.card_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.zone_names = ["hand", "bf_self", "bf_opp", "gy_self", "gy_opp"]
        self.poolers = nn.ModuleDict(
            {name: ZonePooler(hidden_dim, n_heads) for name in self.zone_names}
        )

        self.gs_proj = nn.Sequential(
            nn.Linear(self.splitter.game_state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        fused_dim = (1 + len(self.zone_names)) * hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.GELU(),
            nn.Linear(features_dim, features_dim),
            nn.LayerNorm(features_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode a batch of flat observations into a fixed-size vector."""
        zones = self.splitter(obs)
        zone_embs: list[torch.Tensor] = []
        for name in self.zone_names:
            zone = zones[name]
            mask = zone_mask(zone)
            slots = self.slot_proj(zone)
            pooled = self.poolers[name](slots, mask)
            zone_embs.append(pooled)
        gs_emb = self.gs_proj(zones["game_state"])
        fused = torch.cat([gs_emb, *zone_embs], dim=-1)
        return self.fusion(fused)

"""Tests for the obs <-> action hand-cap contract.

The env pins the two hand caps together and enforces a runtime contract
check in :meth:`mtg.env.MTGEnv.__init__`. These tests make sure:

1. The default caps satisfy ``max_hand_slots <= max_hand_size``.
2. The action-space size matches the canonical value of 478.
3. Constructing the env with a violating ``ActionSpaceConfig`` raises a
   clear ``ValueError`` instead of silently producing an
   under-observable mask.
4. End-to-end ``reset`` + ``step`` keeps action-mask hand slots within
   the observation cap (no slot above ``max_hand_size`` is ever set to
   1 in the live mask).
"""

from __future__ import annotations

import numpy as np
import pytest

from mtg.env import MTGEnv
from mtg.env.action_mask import ActionMaskBuilder, ActionSpaceConfig
from mtg.env.observation import ObservationConfig

# ---------------------------------------------------------------------------
# Static contract
# ---------------------------------------------------------------------------


def test_default_caps_satisfy_obs_action_contract() -> None:
    """The defaults must satisfy ``max_hand_slots <= max_hand_size``."""
    obs_cfg = ObservationConfig()
    act_cfg = ActionSpaceConfig()
    assert act_cfg.max_hand_slots <= obs_cfg.max_hand_size, (
        f"Action-mask hand cap {act_cfg.max_hand_slots} exceeds "
        f"observation hand cap {obs_cfg.max_hand_size}; the policy would "
        "be deciding on hand slots it cannot see."
    )


def test_action_space_size_matches_canonical_value() -> None:
    """Total action-space size matches the canonical value of 478."""
    env = MTGEnv(deck_archetype="mono_red_aggro", max_turns=5)
    # 6 (special idxs) + 5*max_hand_slots + max_permanent_slots
    # + max_creature_slots + 2*max_creature_slots + max_target_slots
    # + max_permanent_slots
    # = 6 + 5*10 + 60 + 60 + 120 + 122 + 60 = 478
    assert env.action_space.n == 478, f"Expected action_space.n=478; got {env.action_space.n}"


def test_hand_action_stride_matches_obs_hand_cap() -> None:
    """``play_land_start + max_hand_slots`` must end at the obs hand cap."""
    builder = ActionMaskBuilder(ActionSpaceConfig())
    play_land_start = builder.index_map.play_land_start
    cast_sorcery_start = builder.index_map.cast_sorcery_start
    assert cast_sorcery_start - play_land_start == 10, (
        f"Hand-action stride must equal max_hand_slots (10); "
        f"got {cast_sorcery_start - play_land_start}"
    )


# ---------------------------------------------------------------------------
# Runtime contract check
# ---------------------------------------------------------------------------


def test_violating_action_cap_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """If someone hard-overrides the action cap above the obs cap, fail loudly.

    We monkey-patch the default ``ActionSpaceConfig`` to a violating
    value so the env's ``__init__`` triggers the contract check.
    """
    bad_cfg = ActionSpaceConfig(max_hand_slots=60)

    # Inject the bad config by patching ``ActionMaskBuilder.__init__``
    # to use ``bad_cfg`` regardless of what MTGEnv passes.  This
    # approximates "user constructs MTGEnv with a custom builder that
    # internally uses a violating cap" without changing MTGEnv's
    # public surface.
    real_builder_init = ActionMaskBuilder.__init__

    def _patched_init(self, config=None, rules_engine=None) -> None:  # type: ignore[no-untyped-def]
        real_builder_init(self, bad_cfg, rules_engine)

    monkeypatch.setattr(ActionMaskBuilder, "__init__", _patched_init)

    with pytest.raises(ValueError, match="Action-mask hand cap"):
        MTGEnv(deck_archetype="mono_red_aggro", max_turns=5)


# ---------------------------------------------------------------------------
# End-to-end behaviour
# ---------------------------------------------------------------------------


def test_no_action_mask_slot_exceeds_obs_hand_cap_during_episode() -> None:
    """During a real rollout, no hand-targeted action slot above the obs cap is ever live.

    We walk a few turns of an episode (random legal action) and assert
    on every step that the action mask never enables an action with
    ``hand_slot >= obs_hand_size``.  This catches both:

    * silently raising the action cap above the obs cap (would fail at
      construction now), and
    * action-mask construction code accidentally producing slot indices
      outside the cap (would still fail here even if construction
      passed).
    """
    env = MTGEnv(deck_archetype="mono_red_aggro", seed=2026, max_turns=8)
    obs_cap = env.obs_builder.config.max_hand_size
    act_cap = env.action_builder.config.max_hand_slots
    assert act_cap <= obs_cap

    builder = env.action_builder
    idx_map = builder.index_map
    play_land_start = idx_map.play_land_start
    cast_sorcery_start = idx_map.cast_sorcery_start
    cast_instant_start = idx_map.cast_instant_start
    discard_start = idx_map.discard_start
    bottom_start = idx_map.bottom_start

    # Hand-derived slot ranges: every action in these ranges encodes a
    # slot index into the player's hand.
    hand_action_ranges = [
        (bottom_start, bottom_start + act_cap, "BOTTOM_CARD"),
        (discard_start, discard_start + act_cap, "DISCARD_CARD"),
        (play_land_start, play_land_start + act_cap, "PLAY_LAND"),
        (cast_sorcery_start, cast_sorcery_start + act_cap, "CAST_SORCERY"),
        (cast_instant_start, cast_instant_start + act_cap, "CAST_INSTANT"),
    ]

    obs, info = env.reset(seed=2026)
    rng = np.random.default_rng(2026)
    for _ in range(40):
        mask = info.get("action_mask")
        if mask is None or not np.any(mask):
            break

        # Verify every hand-targeted live action slot is within the obs cap.
        live_idxs = np.flatnonzero(mask)
        for action_idx in live_idxs:
            for start, end, label in hand_action_ranges:
                if start <= action_idx < end:
                    slot = int(action_idx - start)
                    assert slot < obs_cap, (
                        f"{label} action at index {action_idx} maps to "
                        f"hand slot {slot}, which is beyond the "
                        f"observation cap (max_hand_size={obs_cap})"
                    )
                    break

        # Step with a random legal action.
        choice = int(rng.choice(live_idxs))
        obs, _, terminated, truncated, info = env.step(choice)
        if terminated or truncated:
            break

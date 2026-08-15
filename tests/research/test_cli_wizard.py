"""Tests for the ``mtg-research`` interactive wizard.

These do not exercise the full interactive prompt loop.  Instead they
spy on :func:`scripts.research.cli._run_pipeline` to assert the
invariants the wizard MUST hold for paper-quality comparisons: chiefly
that an A/B run feeds both arms the same ``agency_mode`` (so they run
in the same MDP) and the same seeds (so paired bootstrap is
well-defined).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture
def _stub_user_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive the wizard's prompts deterministically.

    Sequence:
    * mode = 2 (A/B comparison)
    * agency_choice = 3 (full); specifically NOT the default
    * remaining IntPrompts: budget, n_envs, eval_episodes, training_mode

    Confirms default to True to keep the wizard moving forward.
    """
    int_inputs = iter(
        [
            2,  # mode (A/B comparison)
            10_000,  # timesteps_per_opponent
            1,  # training_mode = round-robin (selected because >1 opponent)
            3,  # agency_choice = "full"
            1,  # n_envs (auto)
            10,  # eval_episodes
        ]
    )
    str_inputs = iter(
        [
            "exp_test",  # experiment name
            "all",  # player decks (defaults)
            "all",  # opponents (defaults)
            "42 123",  # seeds
        ]
    )

    def fake_int_ask(prompt, default=None, **_kwargs):  # type: ignore[no-untyped-def]
        try:
            return next(int_inputs)
        except StopIteration:
            return default

    def fake_str_ask(prompt, default=None, **_kwargs):  # type: ignore[no-untyped-def]
        try:
            return next(str_inputs)
        except StopIteration:
            return default

    def fake_confirm_ask(prompt, default=True, **_kwargs):  # type: ignore[no-untyped-def]
        return bool(default)

    from scripts.research import cli as cli_mod

    monkeypatch.setattr(cli_mod, "IntPrompt", type("X", (), {"ask": staticmethod(fake_int_ask)}))
    monkeypatch.setattr(cli_mod, "Prompt", type("X", (), {"ask": staticmethod(fake_str_ask)}))
    monkeypatch.setattr(
        cli_mod,
        "Confirm",
        type("X", (), {"ask": staticmethod(fake_confirm_ask)}),
    )


def test_ab_wizard_uses_same_agency_mode_for_both_arms(_stub_user_inputs: None) -> None:
    """A/B comparison runs PPO and Causal in the SAME MDP.

    The wizard must thread the user-selected ``agency_mode`` to both
    arms; if it pinned one arm to a different mode the two arms would
    operate in different action spaces and silently invalidate the
    paired-bootstrap significance claim downstream. This test re-runs
    the wizard with the user explicitly choosing ``"full"`` agency and
    asserts that both arms (and the cross-aggregate) see
    ``agency_mode="full"``.
    """
    from scripts.research import cli as cli_mod

    captured_calls: list[dict[str, object]] = []

    def _spy_run_pipeline(**kwargs: object) -> int:
        captured_calls.append(kwargs)
        return 0

    def _spy_aggregate(**_kwargs: object) -> None:
        return None

    with (
        patch.object(cli_mod, "_run_pipeline", side_effect=_spy_run_pipeline),
        patch("scripts.research.aggregate.aggregate", side_effect=_spy_aggregate),
    ):
        rc = cli_mod.interactive_wizard()

    assert rc == 0
    assert len(captured_calls) == 2, "A/B wizard must invoke _run_pipeline twice (PPO, Causal)"

    arm_a, arm_b = captured_calls
    assert arm_a["agency_mode"] == "full", (
        f"PPO arm received agency_mode={arm_a['agency_mode']!r}; "
        "the wizard must NOT pin it to 'auto' when the user requests 'full'."
    )
    assert (
        arm_b["agency_mode"] == "full"
    ), f"Causal arm received agency_mode={arm_b['agency_mode']!r}; expected user-chosen 'full'."

    assert arm_a["seeds"] == arm_b["seeds"], (
        f"A/B arms have mismatched seeds {arm_a['seeds']!r} vs "
        f"{arm_b['seeds']!r}; paired-bootstrap requires identical seeds."
    )
    assert arm_a["player_decks"] == arm_b["player_decks"]
    assert arm_a["opponents"] == arm_b["opponents"]
    assert arm_a["training_mode"] == arm_b["training_mode"]
    assert arm_a["timesteps_per_opponent"] == arm_b["timesteps_per_opponent"]
    assert arm_a["max_turns"] == arm_b["max_turns"]
    assert arm_a["reward_type"] == arm_b["reward_type"]

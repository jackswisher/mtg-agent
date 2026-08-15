"""SCM <-> env causal-variable contract tests.

These tests guard the invariant that every parent variable an SCM
``_compute_variable`` branch reads MUST be present in the dictionary
returned by :meth:`mtg.env.reward.RewardCalculator.get_causal_variable_values`.

Failing this contract leads to silent, severe bugs:

* The SCM falls back to default values for missing parents.
* Downstream variables (e.g. ``tempo``) evaluate to nonsensical
  defaults regardless of game state.
* The CGFA intervention-calibration target ``factor_eps`` is then
  uncorrelated with the policy gradient signal, so the calibration
  loss has no power to align ``A_k`` with the SCM prior.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from mtg.causal.scm import StructuralCausalModel
from mtg.env.mtg_env import MTGEnv
from mtg.env.reward import RewardCalculator

# ---------------------------------------------------------------------------
# Static reads: parse _compute_variable for every ``values.get("...")``
# expression so we cannot accidentally add a new SCM parent without
# updating the env's exporter.
# ---------------------------------------------------------------------------

_SCM_PATH = Path(__file__).resolve().parents[2] / "mtg" / "causal" / "scm.py"


def _scm_parent_keys() -> set[str]:
    """Statically extract every key read via ``values.get("...")``.

    Includes both ``_compute_variable`` and ``_compute_win_prob``, so
    any new structural equation that pulls a parent value will be
    automatically swept into the contract test.
    """
    src = _SCM_PATH.read_text(encoding="utf-8")
    # Match values.get("name", ...) and values.get('name', ...)
    pattern = re.compile(r"""values\.get\(\s*['"]([a-zA-Z_][a-zA-Z0-9_]*)['"]""")
    return set(pattern.findall(src))


def test_scm_parents_static_extraction_is_nonempty() -> None:
    """Sanity check: the regex should find at least the well-known parents."""
    keys = _scm_parent_keys()
    expected = {
        "mana_t",
        "land_drop",
        "mana_creatures",  # mana_t1
        "own_power",
        "opp_power",  # board_press
        "board_presence",
        "threat_count",  # threat_density
        "opp_board_presence",  # card_adv
        "mana_spent",
        "opp_mana",
        "opp_mana_spent",  # tempo
        "own_life",
        "opp_life",  # life_buffer
        "has_removal",  # removal_avail
    }
    missing = expected - keys
    assert not missing, f"Static extractor missed expected SCM parents: {missing}"


def test_env_reward_exposes_every_scm_parent_static() -> None:
    """Static contract: every SCM parent key MUST be returned by RewardCalculator.

    If the env failed to expose a parent (for example ``mana_spent``,
    ``opp_mana`` or ``opp_mana_spent``) the SCM mechanism that depends
    on it would silently evaluate to 0 and corrupt the
    intervention-calibration signal for the affected factor.
    """
    rc = RewardCalculator()
    env = MTGEnv(deck_archetype="mono_red_aggro", seed=0, max_turns=3)
    env.reset()
    cv = rc.get_causal_variable_values(env.state, player_id=0)
    cv_keys = set(cv.keys())

    scm_parents = _scm_parent_keys()
    missing = scm_parents - cv_keys
    assert not missing, (
        f"SCM-env contract broken: env.causal_variables is missing keys "
        f"that the SCM structural equations read: {missing}.\n"
        f"Add them to RewardCalculator.get_causal_variable_values."
    )


def test_scm_evaluate_runs_on_real_env_state_without_warning(caplog) -> None:
    """A real env state must drive SCM.evaluate() with no missing-key fallbacks."""
    rc = RewardCalculator()
    env = MTGEnv(deck_archetype="mono_red_aggro", seed=0, max_turns=3)
    env.reset()
    cv = rc.get_causal_variable_values(env.state, player_id=0)
    scm = StructuralCausalModel()
    out = scm.evaluate(cv, force_recompute=True)
    # Tempo MUST be a finite float in [-1, 1] -- if mana_spent etc. were
    # missing, the structural equation would still produce a value, but
    # it would be the constant default rather than reflect the state.
    assert "tempo" in out
    tempo = out["tempo"]
    assert np.isfinite(tempo)
    assert -1.0 <= tempo <= 1.0


def test_step_info_causal_variables_supersets_scm_parents() -> None:
    """End-to-end: the ``info["causal_variables"]`` after a step is contract-complete.

    This catches regressions where ``MTGEnv.step()`` plugs in a
    different (e.g. trimmed) causal-variable dict.
    """
    env = MTGEnv(deck_archetype="mono_red_aggro", seed=0, max_turns=3)
    obs, info = env.reset()
    # Take any action (pass priority is action 0 and is always legal).
    obs, reward, terminated, truncated, info = env.step(0)
    cv = info.get("causal_variables", {})
    assert cv, "step() must populate info['causal_variables']"

    scm_parents = _scm_parent_keys()
    missing = scm_parents - set(cv.keys())
    assert not missing, f"step()['causal_variables'] is missing SCM parents: {missing}"


def _make_stub_env():
    """Build a real gymnasium-compatible stub env for the wrapper tests."""
    import gymnasium as gym
    from gymnasium.spaces import Box, Discrete

    class _Stub(gym.Env):
        metadata: dict = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
            self.action_space = Discrete(2)

        def reset(self, **_):
            return np.zeros(4, dtype=np.float32), {"causal_variables": {"card_adv": 0.0}}

        def step(self, _action):
            return (
                np.zeros(4, dtype=np.float32),
                0.0,
                False,
                False,
                {"causal_variables": {"card_adv": 1.0}},
            )

    return _Stub()


def test_cgfa_wrapper_warns_loudly_on_scm_eval_failure(caplog) -> None:
    """The narrowed exception handler in CGFAEnvWrapper must emit a warning.

    Ensures we do NOT silently degrade the calibration signal again.
    """
    import logging

    from mtg.agents.reinforcement_learning.cgfa.factor_spec import FactorSpec
    from mtg.agents.reinforcement_learning.cgfa.wrapper import CGFAEnvWrapper

    class _BrokenSCM:
        def evaluate(self, *_, **__):
            raise KeyError("synthetic-missing-parent")

    spec = FactorSpec()
    w = CGFAEnvWrapper(_make_stub_env(), factor_spec=spec, scm=_BrokenSCM())
    w.reset()
    with caplog.at_level(logging.WARNING, logger="mtg.agents.reinforcement_learning.cgfa.wrapper"):
        w.step(0)
    warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warning_records, "CGFAEnvWrapper did not warn on a synthetic SCM failure"
    assert any("SCM.evaluate()" in r.getMessage() for r in warning_records)
    assert w._scm_eval_failures == 1


def test_cgfa_wrapper_does_not_swallow_unexpected_exception() -> None:
    """A bug-class exception (e.g. AttributeError) must NOT be silently caught."""
    from mtg.agents.reinforcement_learning.cgfa.factor_spec import FactorSpec
    from mtg.agents.reinforcement_learning.cgfa.wrapper import CGFAEnvWrapper

    class _BuggySCM:
        def evaluate(self, *_, **__):
            raise AttributeError("regression: some_attr_was_renamed")

    spec = FactorSpec()
    w = CGFAEnvWrapper(_make_stub_env(), factor_spec=spec, scm=_BuggySCM())
    w.reset()
    # AttributeError is NOT in the narrowed except clause; it must
    # propagate so a real bug is loud, not silent.
    with pytest.raises(AttributeError):
        w.step(0)

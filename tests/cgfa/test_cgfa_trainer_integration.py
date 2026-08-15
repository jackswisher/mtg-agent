"""End-to-end CGFA-PPO + Trainer + MTGEnv smoke test.

Validates that the canonical :class:`mtg.training.train.Trainer`
integrates :class:`CGFAMaskablePPO` correctly:

* The agent registry resolves ``"cgfa"`` to :class:`CGFAAgent`.
* :class:`CGFAEnvWrapper` is automatically applied to each rollout
  env (so per-factor signals reach the rollout buffer).
* :class:`CGFAMaskablePPO.train` runs end-to-end without errors.
* All CGFA logger keys (gate stats, per-factor calibration,
  per-factor credit shares) are populated after one update.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mtg.agents import CGFAAgent  # noqa: E402
from mtg.agents.causal.cgfa_agent import CGFAAgent as DirectCGFAAgent  # noqa: E402
from mtg.training.train import Trainer, TrainingConfig  # noqa: E402


def _make_trainer(**overrides) -> Trainer:
    cfg = TrainingConfig(
        agent_type="cgfa",
        deck_archetype="mono_red_aggro",
        opponent_archetype="mono_red_aggro",
        reward_type="shaped",
        max_turns=15,
        max_steps_per_episode=200,
        auto_combat=True,
        auto_target=True,
        total_timesteps=512,
        n_envs=1,
        seed=0,
        enable_checkpointing=False,
        enable_periodic_eval=False,
        enable_episode_logger=False,
        enable_entropy_schedule=False,
        enable_early_stopping=False,
        enable_adaptive_kl=False,
        experiment_name="test_cgfa_trainer",
        agent_kwargs={
            "n_steps": 128,
            "batch_size": 64,
            "n_epochs": 2,
            "learning_rate": 3e-4,
            "intervention_calibration_coef": 0.1,
            "learnable_gate": True,
            "state_conditional_gate": True,
            **overrides,
        },
    )
    return Trainer(cfg)


def test_trainer_resolves_cgfa_agent_class() -> None:
    """The agent registry maps ``cgfa`` to ``CGFAAgent``."""
    trainer = _make_trainer()
    trainer.setup()
    try:
        assert isinstance(trainer.agent, CGFAAgent)
        assert isinstance(trainer.agent, DirectCGFAAgent)
    finally:
        trainer.close()


def test_trainer_wraps_env_with_cgfa_env_wrapper() -> None:
    """The vec env wraps each rollout env with :class:`CGFAEnvWrapper`.

    We can only inspect the wrapper hierarchy on a ``DummyVecEnv``
    (n_envs=1) where the inner env is a live Python object.
    """
    from mtg.agents.reinforcement_learning.cgfa import CGFAEnvWrapper

    trainer = _make_trainer()
    trainer.setup()
    try:
        inner_env = trainer.env.envs[0]
        # Walk the wrapper chain looking for CGFAEnvWrapper.
        found = False
        cur = inner_env
        while cur is not None:
            if isinstance(cur, CGFAEnvWrapper):
                found = True
                break
            cur = getattr(cur, "env", None)
        assert found, "CGFAEnvWrapper was not applied to the rollout env"
    finally:
        trainer.close()


def test_trainer_train_emits_cgfa_logging_keys() -> None:
    """After one CGFA-PPO update, all CGFA logging keys are present."""
    trainer = _make_trainer()
    trainer.setup()
    try:
        trainer.train()
        keys = set(trainer.agent.model.logger.name_to_value.keys())

        # Gate stats
        for k in ("cgfa/gate/mean", "cgfa/gate/std", "cgfa/gate/min", "cgfa/gate/max"):
            assert k in keys, f"missing CGFA gate key: {k}"

        # Per-factor diagnostics
        spec = trainer.agent.factor_spec
        for name in spec.names:
            assert f"cgfa/blend/{name}" in keys
            assert f"cgfa/factor_corr/{name}" in keys
            assert f"cgfa/factor_sign_agree/{name}" in keys
            assert f"cgfa/factor_contribution/{name}" in keys
            assert f"cgfa/factor_share/{name}" in keys
            assert f"cgfa/factor_adv/{name}/mean" in keys
            assert f"cgfa/factor_adv/{name}/std" in keys
            assert f"cgfa/factor_ret/{name}/mean" in keys

        # Sanity: shares are non-negative and (when nonzero) sum to 1.
        shares = [
            trainer.agent.model.logger.name_to_value[f"cgfa/factor_share/{n}"] for n in spec.names
        ]
        assert all(s >= 0 for s in shares)
        if sum(shares) > 0:
            assert sum(shares) == pytest.approx(1.0, rel=1e-4)
    finally:
        trainer.close()

"""Tests for the league / PFSP infrastructure."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mtg.training.callbacks import LeagueCallback
from mtg.training.env_factory import EnvConfig, LeagueEnvWrapper, create_env
from mtg.training.league import (
    DEFAULT_ELO,
    League,
    LeagueConfig,
    PFSPSampler,
    elo_update,
    expected_score,
)
from mtg.utils.interactive import EvaluationConfig as InteractiveEvalConfig
from mtg.utils.interactive import TrainingConfig as InteractiveTrainingConfig

# ---------------------------------------------------------------------------
# Elo / PFSP primitives
# ---------------------------------------------------------------------------


def test_expected_score_symmetry() -> None:
    """Equal ratings must predict exactly 0.5."""
    assert abs(expected_score(1500.0, 1500.0) - 0.5) < 1e-9


def test_elo_update_rewards_underdog_win() -> None:
    """Underdog wins close the rating gap more than equal wins.

    A weaker player beating a stronger one gains more than the
    symmetric rating gap.
    """
    r_a, r_b = 1400.0, 1600.0
    new_a, new_b = elo_update(r_a, r_b, score_a=1.0, k=32.0)
    assert new_a > r_a
    assert new_b < r_b
    gap_before = r_b - r_a
    gap_after = new_b - new_a
    assert gap_after < gap_before


def test_elo_update_is_zero_sum() -> None:
    """Elo rating updates must preserve total rating (zero-sum property)."""
    r_a, r_b = 1500.0, 1500.0
    new_a, new_b = elo_update(r_a, r_b, score_a=1.0, k=16.0)
    total_before = r_a + r_b
    total_after = new_a + new_b
    assert abs(total_before - total_after) < 1e-9


def test_pfsp_prefers_near_equal_opponents() -> None:
    """PFSP must under-sample dominated opponents.

    Stronger-than-learner opponents should be sampled more often than
    opponents the learner dominates.  This is the whole PFSP point.
    """
    sampler = PFSPSampler(p=2.0, eps=0.0, rng=np.random.default_rng(0))
    learner = 1500.0
    ratings = [1200.0, 1500.0, 1700.0]  # dominated, even, harder
    counts = [0, 0, 0]
    for _ in range(5000):
        counts[sampler.sample_index(learner, ratings)] += 1
    # Dominated opponent should be least sampled; equal/harder opponents
    # should each get substantially more attention.
    assert counts[0] < counts[1]
    assert counts[0] < counts[2]


# ---------------------------------------------------------------------------
# League bookkeeping
# ---------------------------------------------------------------------------


class _NullAgent:
    """Placeholder agent used for pool bookkeeping tests."""

    def select_action(self, *_args, **_kwargs) -> int:
        return 0


def test_league_add_and_sample() -> None:
    """Adding heuristics to the pool and sampling should produce a valid entry."""
    league = League(LeagueConfig(sampling="uniform"), rng=np.random.default_rng(1))
    league.add_heuristic("a", "deck_a", _NullAgent())
    league.add_heuristic("b", "deck_b", _NullAgent())
    entry = league.sample_opponent()
    assert entry.name in {"a", "b"}
    assert entry.resolve_agent() is not None


def test_league_record_match_updates_ratings() -> None:
    """``record_match`` should update learner + opponent ratings consistently."""
    league = League(LeagueConfig(elo_k=16.0))
    league.add_heuristic("opp", "deck", _NullAgent())

    league.record_match("opp", win=True)
    assert league.learner_rating > DEFAULT_ELO
    assert league.get("opp").rating < DEFAULT_ELO
    # The *sum* of all ratings must be preserved by Elo.
    total = league.learner_rating + league.get("opp").rating
    assert abs(total - 2 * DEFAULT_ELO) < 1e-9


def test_league_pfsp_distribution_shift_after_learning() -> None:
    """PFSP shifts focus away from opponents the learner has crushed.

    After the learner wins a lot, PFSP should direct attention away
    from crushed opponents toward the remaining hard ones.
    """
    league = League(
        LeagueConfig(elo_k=32.0, pfsp_p=2.0, pfsp_eps=0.0, sampling="pfsp"),
        rng=np.random.default_rng(0),
    )
    league.add_heuristic("easy", "deck_easy", _NullAgent())
    league.add_heuristic("hard", "deck_hard", _NullAgent())

    # Learner crushes "easy" 30 games in a row, stays even vs "hard".
    for _ in range(30):
        league.record_match("easy", win=True)

    counts = {"easy": 0, "hard": 0}
    for _ in range(3000):
        counts[league.sample_opponent().name] += 1
    assert counts["hard"] > counts["easy"]


def test_league_snapshot_eviction(tmp_path: Path) -> None:
    """Exceeding ``max_historical`` should evict the oldest historical entry."""
    league = League(
        LeagueConfig(max_historical=2, snapshot_dir=tmp_path, keep_snapshots_on_disk=False)
    )

    class _Saveable:
        def save(self, path: str) -> None:
            Path(path + ".zip").write_text("snapshot")

    for i in range(4):
        league.add_snapshot(f"snap_{i}", _Saveable(), deck="learner")
    historicals = [e for e in league.pool if e.is_historical]
    assert len(historicals) == 2
    assert {e.name for e in historicals} == {"snap_2", "snap_3"}


# ---------------------------------------------------------------------------
# LeagueEnvWrapper integration
# ---------------------------------------------------------------------------


def test_league_env_wrapper_rotates_opponents_on_reset() -> None:
    """Each ``reset()`` should install the sampled opponent on the env."""
    cfg = EnvConfig(
        player_deck="mono_red_aggro",
        opponent_deck="azorius_control",
        max_turns=3,
        max_steps_per_episode=20,
    )
    env = create_env(cfg)
    league = League(LeagueConfig(sampling="uniform"), rng=np.random.default_rng(2))
    league.add_heuristic("opp_a", "mono_red_aggro", _NullAgent())
    league.add_heuristic("opp_b", "azorius_control", _NullAgent())
    wrapper = LeagueEnvWrapper(env, league)

    wrapper.reset(seed=0)
    first = env.active_opponent_name
    # Force the other opponent to be sampled
    wrapper.reset(seed=0)
    second_candidates = {"opp_a", "opp_b"}
    assert first in second_candidates
    assert env.active_opponent_name in second_candidates


# ---------------------------------------------------------------------------
# Interactive -> engine config conversion (C3)
# ---------------------------------------------------------------------------


def test_interactive_training_config_to_engine_multi_opp() -> None:
    """Multi-opponent CLI config should enable league training by default."""
    cli_cfg = InteractiveTrainingConfig(
        agent_type="ppo",
        player_deck="mono_red_aggro",
        opponent_deck="azorius_control,simic_ramp",
        timesteps=500,
        n_envs=1,
    )
    engine_cfg = cli_cfg.to_engine_config()
    assert engine_cfg.agent_type == "ppo"
    assert engine_cfg.deck_archetype == "mono_red_aggro"
    assert engine_cfg.opponent_archetype == "azorius_control"  # first one
    assert engine_cfg.enable_league is True
    assert engine_cfg.league_opponents == ["azorius_control", "simic_ramp"]


def test_interactive_training_config_to_engine_single_opp() -> None:
    """Single-opponent CLI config should default to league disabled."""
    cli_cfg = InteractiveTrainingConfig(
        agent_type="ppo",
        player_deck="mono_red_aggro",
        opponent_deck="azorius_control",
        timesteps=500,
    )
    engine_cfg = cli_cfg.to_engine_config()
    assert engine_cfg.enable_league is False
    assert engine_cfg.opponent_archetype == "azorius_control"


def test_interactive_eval_config_to_engine() -> None:
    """CLI EvaluationConfig should yield a canonical engine config."""
    cli_cfg = InteractiveEvalConfig(
        agent_type="greedy_aggro",
        player_deck="mono_red_aggro",
        opponent_deck="azorius_control,simic_ramp",
        episodes=50,
        seeds=[7, 11, 13],
    )
    engine_cfg = cli_cfg.to_engine_config()
    assert engine_cfg.deck_archetype == "mono_red_aggro"
    assert engine_cfg.opponent_archetype == "azorius_control"
    assert engine_cfg.n_episodes == 50
    assert engine_cfg.seed == 7


# ---------------------------------------------------------------------------
# MTGEnv.set_opponent + info["active_opponent"]
# ---------------------------------------------------------------------------


def test_set_opponent_updates_active_name() -> None:
    """``MTGEnv.set_opponent`` should update the active opponent name."""
    cfg = EnvConfig(
        player_deck="mono_red_aggro",
        opponent_deck="azorius_control",
        max_turns=3,
        max_steps_per_episode=20,
    )
    env = create_env(cfg)
    env.set_opponent(_NullAgent(), name="custom_opp")
    assert env.active_opponent_name == "custom_opp"
    assert env.opponent_agent is not None


# ---------------------------------------------------------------------------
# Trainer wiring: League RNG follows TrainingConfig.seed
# ---------------------------------------------------------------------------


def test_trainer_seeds_league_rng_from_training_config(tmp_path: Path) -> None:
    """``Trainer._build_league`` must seed the league RNG from the training seed.

    Without this wiring, two runs that share every other seed would
    still get different opponent draws because the league would default
    to a fresh ``np.random.default_rng()``.
    """
    from mtg.training.train import Trainer, TrainingConfig

    cfg = TrainingConfig(
        agent_type="random",
        deck_archetype="mono_red_aggro",
        opponent_archetype="azorius_control",
        total_timesteps=1,
        n_envs=1,
        max_turns=2,
        max_steps_per_episode=4,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpts"),
        experiment_name="league_seed_check",
        enable_league=True,
        league_opponents=["azorius_control", "simic_ramp"],
        seed=12345,
        enable_periodic_eval=False,
        enable_episode_logger=False,
        enable_checkpointing=False,
        enable_entropy_schedule=False,
        enable_early_stopping=False,
        enable_vec_normalize=False,
    )
    trainer = Trainer(cfg)
    league_a = trainer._build_league()
    trainer.close()

    trainer_b = Trainer(cfg)
    league_b = trainer_b._build_league()
    trainer_b.close()

    # Same training seed -> identical opponent draws via PFSP.
    ratings = [1500.0, 1450.0, 1550.0]
    draws_a = [league_a._pfsp.sample_index(1500.0, ratings) for _ in range(64)]
    draws_b = [league_b._pfsp.sample_index(1500.0, ratings) for _ in range(64)]
    assert draws_a == draws_b, (
        "League PFSP draws diverged between runs with the same training "
        "seed; League rng is not being seeded from TrainingConfig.seed."
    )


# ---------------------------------------------------------------------------
# LeagueCallback: periodic flush of league.json
# ---------------------------------------------------------------------------


class _FakeMatchModel:
    """Minimal stub used to inject info dicts through ``LeagueCallback._on_step``."""

    def __init__(self) -> None:
        self.num_timesteps = 0


def test_league_callback_flushes_periodically(tmp_path: Path) -> None:
    """``LeagueCallback`` must persist league.json every N matches, not only at end.

    Mid-run crashes would otherwise discard the entire match history;
    this regression test pins the periodic-flush behaviour by driving
    the callback through a small batch of synthetic matches.
    """
    league = League(LeagueConfig(sampling="uniform"), rng=np.random.default_rng(0))
    league.add_heuristic("opp_a", "mono_red_aggro", _NullAgent())
    league.add_heuristic("opp_b", "azorius_control", _NullAgent())

    cb = LeagueCallback(
        league=league,
        snapshot_interval=None,
        log_dir=tmp_path,
        flush_every_n_matches=3,
        verbose=0,
    )
    cb.model = _FakeMatchModel()
    cb.locals = {
        "dones": np.array([True]),
        "infos": [{"active_opponent": "opp_a", "winner": 0}],
    }

    target = tmp_path / "league.json"
    assert not target.exists()

    for _ in range(2):
        cb._on_step()
    assert not target.exists(), "Flushed too early"

    cb._on_step()
    assert target.exists(), "Periodic flush should have written league.json"
    payload = json.loads(target.read_text())
    assert payload["match_history"], "Match history must persist via the periodic flush"
    assert len(payload["match_history"]) == 3

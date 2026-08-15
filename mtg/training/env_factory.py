"""Unified environment factory used by training and evaluation.

This is the single source of truth for how MTG environments are
constructed.  Both ``Trainer`` and ``Evaluator`` route through here so
that training and evaluation cannot drift into different MDPs (different
``auto_combat`` / ``auto_target`` settings, different step limits, etc.)
without the user explicitly asking for it.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnvConfig:
    """Canonical MDP configuration shared between training and evaluation.

    Any field that changes the reachable action / observation space or
    the transition dynamics lives here.  By pinning these into one
    dataclass and passing it through both sides we avoid the classic
    train/eval drift bug where an agent is trained under simplified
    combat but evaluated under full agency (or vice versa).

    Attributes:
        player_deck: Player's deck archetype.
        opponent_deck: Opponent's deck archetype (single opponent).
        reward_type: Reward shaping type ("sparse", "shaped", "dense").
        max_turns: Maximum MTG turns per game.
        max_steps_per_episode: Safety cap on env steps (truncation).
        auto_combat: If True, attacks are all-or-nothing.  Agent chooses
            whether to attack at all; the engine then attacks with every
            eligible creature.
        auto_target: If True, spell targets are auto-selected using
            board heuristics.  Agent only chooses which spell to cast.
        auto_mana: If True, mana payment is auto-resolved. Should stay
            True in almost all cases; learning mana tapping has no
            strategic signal.
        use_heuristic_opponent: If True, use a heuristic opponent agent
            selected by ``heuristic_for_deck``.  If False, the built-in
            scripted opponent in the engine is used.
        seed: Base random seed.
    """

    player_deck: str
    opponent_deck: str
    reward_type: str = "shaped"
    max_turns: int = 20
    max_steps_per_episode: int = 500
    auto_combat: bool = False
    auto_target: bool = False
    auto_mana: bool = True
    use_heuristic_opponent: bool = True
    seed: int = 42
    # Single source of truth for discount factor.  ``RewardConfig.gamma``
    # (potential-based shaping) and PPO's ``gamma`` must match for
    # potential-based shaping to preserve the optimal policy; pinning them
    # here avoids the classic subtle bug of drifting them apart.
    gamma: float = 0.995

    def with_overrides(self, **kwargs: tp.Any) -> EnvConfig:
        """Return a copy with overridden fields."""
        from dataclasses import replace

        return replace(self, **kwargs)

    def describe(self) -> str:
        """One-line human-readable description (for logs / eval headers)."""
        agency = "full"
        if self.auto_combat and self.auto_target:
            agency = "auto"
        elif self.auto_combat or self.auto_target:
            agency = "mixed"
        return (
            f"EnvConfig(deck={self.player_deck} vs {self.opponent_deck}, "
            f"reward={self.reward_type}, max_turns={self.max_turns}, "
            f"max_steps={self.max_steps_per_episode}, agency={agency}, "
            f"heuristic_opp={self.use_heuristic_opponent}, seed={self.seed})"
        )


def get_default_agent_for_deck(deck_name: str) -> str:
    """Return the heuristic agent name associated with a deck archetype."""
    from mtg.agents import heuristic_for_deck

    normalized = deck_name.lower().replace(" ", "_").replace("-", "_")
    return heuristic_for_deck(normalized) or "greedy_aggro"


def create_env(config: EnvConfig, seed_offset: int = 0) -> tp.Any:
    """Create a single non-vectorised MTG environment from an EnvConfig."""
    from mtg.agents import get_agent
    from mtg.env import MTGEnv

    opponent_agent = None
    if config.use_heuristic_opponent:
        opponent_agent = get_agent(get_default_agent_for_deck(config.opponent_deck))

    env = MTGEnv(
        deck_archetype=config.player_deck,
        opponent_archetype=config.opponent_deck,
        max_turns=config.max_turns,
        max_steps_per_episode=config.max_steps_per_episode,
        reward_type=config.reward_type,
        seed=config.seed + seed_offset,
        auto_combat=config.auto_combat,
        auto_target=config.auto_target,
        auto_mana=config.auto_mana,
        opponent_agent=opponent_agent,
        gamma=config.gamma,
    )
    if config.use_heuristic_opponent:
        env.active_opponent_name = get_default_agent_for_deck(config.opponent_deck)
    return env


try:
    import gymnasium as _gym

    _WrapperBase = _gym.Wrapper
except ImportError:  # pragma: no cover - gymnasium is a hard dep for training

    class _WrapperBase:  # type: ignore[no-redef]
        """Fallback wrapper base used when gymnasium is unavailable."""

        def __init__(self, env: tp.Any):
            self.env = env


class LeagueEnvWrapper(_WrapperBase):
    """Gymnasium wrapper that samples opponents from a ``League``.

    On every ``reset()`` the wrapper queries ``league.sample_opponent()``
    and installs the chosen opponent on the underlying ``MTGEnv`` via
    ``set_opponent`` before delegating to the inner reset.  All other
    calls are passed through unchanged.  ``LeagueCallback`` reads the
    ``info["active_opponent"]`` field that the wrapper causes
    ``MTGEnv.step`` to surface on episode termination, and uses that
    to update Elo ratings.
    """

    def __init__(self, env: tp.Any, league: tp.Any, use_deck_from_entry: bool = False):
        super().__init__(env)
        self.league = league
        self.use_deck_from_entry = use_deck_from_entry

    def reset(self, **kwargs: tp.Any) -> tp.Any:
        """Install a fresh opponent then delegate to the inner reset."""
        entry = self.league.sample_opponent()
        agent = entry.resolve_agent()
        deck = entry.deck if self.use_deck_from_entry else None
        inner = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        inner.set_opponent(agent, name=entry.name, deck=deck)
        return self.env.reset(**kwargs)


def make_env_fn(
    config: EnvConfig,
    seed_offset: int = 0,
    league: tp.Any = None,
    cgfa_factor_spec: tp.Any = None,
    cgfa_scm: tp.Any = None,
) -> tp.Callable[[], tp.Any]:
    """Return a callable that creates an ActionMasker-wrapped MTG env.

    Used by ``SubprocVecEnv`` which requires factories rather than live envs.
    If ``league`` is provided, each created env is additionally wrapped in
    ``LeagueEnvWrapper`` so opponents are rotated between episodes.

    When ``cgfa_factor_spec`` is provided, each created env is additionally
    wrapped in :class:`CGFAEnvWrapper` (after any LeagueEnvWrapper but
    before ActionMasker) so :class:`CGFAMaskablePPO` can read per-factor
    rewards / SCM-predicted deltas from ``info``.
    """

    def _init() -> tp.Any:
        env = create_env(config, seed_offset=seed_offset)
        if league is not None:
            env = LeagueEnvWrapper(env, league)
        if cgfa_factor_spec is not None:
            from mtg.agents.reinforcement_learning.cgfa import CGFAEnvWrapper

            env = CGFAEnvWrapper(env, factor_spec=cgfa_factor_spec, scm=cgfa_scm)
        try:
            from sb3_contrib.common.wrappers import ActionMasker

            from mtg.agents.reinforcement_learning.ppo_agent import _mask_fn
        except ImportError:
            return env
        return ActionMasker(env, _mask_fn)

    return _init


def create_vec_env(
    config: EnvConfig,
    n_envs: int = 1,
    normalize: bool = False,
    norm_obs: bool = True,
    norm_reward: bool = True,
    clip_obs: float = 10.0,
    clip_reward: float = 10.0,
    league: tp.Any = None,
    cgfa_factor_spec: tp.Any = None,
    cgfa_scm: tp.Any = None,
) -> tp.Any:
    """Create a vectorised training environment from an EnvConfig.

    n_envs == 1   -> ``DummyVecEnv`` (single-process, easier to debug).
    n_envs  > 1   -> ``SubprocVecEnv`` for parallel rollout collection.

    If ``normalize`` is True the returned VecEnv is additionally wrapped in
    ``VecNormalize`` with running-mean/var normalisation on observations
    and optional reward normalisation.  Callers should save the running
    stats alongside the policy so evaluation uses identical normalisation.

    When ``league`` is provided, every underlying env is wrapped in a
    ``LeagueEnvWrapper`` so opponents are drawn from the league
    between episodes. A League is stateful and mutating it from
    multiple subprocesses is not supported, so callers should pass
    ``n_envs=1`` when using ``league``.

    When ``cgfa_factor_spec`` is provided, every underlying env is also
    wrapped in :class:`CGFAEnvWrapper` so the per-factor signals required
    by :class:`CGFAMaskablePPO` reach the rollout buffer.  ``cgfa_scm`` is
    optional and only affects the SCM-predicted ``factor_eps`` used by
    the calibration auxiliary.
    """
    env_fns = [
        make_env_fn(
            config,
            seed_offset=i,
            league=league,
            cgfa_factor_spec=cgfa_factor_spec,
            cgfa_scm=cgfa_scm,
        )
        for i in range(max(1, n_envs))
    ]
    if n_envs <= 1:
        from stable_baselines3.common.vec_env import DummyVecEnv

        vec_env = DummyVecEnv(env_fns)
    else:
        from stable_baselines3.common.vec_env import SubprocVecEnv

        vec_env = SubprocVecEnv(env_fns)

    if normalize:
        from stable_baselines3.common.vec_env import VecNormalize

        vec_env = VecNormalize(
            vec_env,
            norm_obs=norm_obs,
            norm_reward=norm_reward,
            clip_obs=clip_obs,
            clip_reward=clip_reward,
            gamma=config.gamma,
        )
    return vec_env


def make_obs_normaliser_from_vec_normalize(
    vec_normalize: tp.Any,
) -> tp.Callable[[np.ndarray], np.ndarray]:
    """Build a per-obs normaliser from a (possibly frozen) ``VecNormalize``.

    Mirrors :meth:`VecNormalize.normalize_obs` for the (Box, non-Dict)
    case so a single-env evaluation loop can apply training-time
    statistics without instantiating a full vec env.

    Use this from inside ``EvalCallback`` to keep periodic evaluation
    in the same observation distribution as training.
    """
    if vec_normalize is None or not hasattr(vec_normalize, "obs_rms"):
        return lambda obs: obs
    obs_rms = vec_normalize.obs_rms
    clip_obs = float(getattr(vec_normalize, "clip_obs", 10.0))
    epsilon = float(getattr(vec_normalize, "epsilon", 1e-8))

    def _normalise(obs: np.ndarray) -> np.ndarray:
        normalised = (obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon)
        return np.clip(normalised, -clip_obs, clip_obs).astype(obs.dtype)

    return _normalise


def maybe_load_vec_normalize_for_eval(
    vec_env: tp.Any,
    run_dir: tp.Any,
    *,
    filename: str = "vec_normalize.pkl",
) -> tp.Any:
    """Wrap ``vec_env`` with the frozen ``VecNormalize`` stats from ``run_dir``.

    This is the evaluation-time counterpart to the ``VecNormalize``
    wrapping that ``Trainer`` does at training time when
    ``enable_vec_normalize`` is True. Without it, a policy trained on
    normalised observations would be evaluated on raw observations,
    introducing a train/eval distribution-shift bug.

    Behaviour:

    * If ``run_dir/<filename>`` does not exist, returns ``vec_env``
      unchanged (no-op for runs that did not enable normalization).
    * If it exists, loads the running-mean/var statistics from disk,
      attaches them to ``vec_env``, and freezes them
      (``training=False``, ``norm_reward=False``) so:

      - obs are normalised with the **same** statistics seen at train
        time (no online drift during eval),
      - rewards are NOT normalised (we want the raw scale for
        bookkeeping / win-rate analysis).

    Args:
        vec_env: The VecEnv to wrap (typically built by
            :func:`create_vec_env` with ``normalize=False``).
        run_dir: Directory containing the saved ``vec_normalize.pkl``.
            Accepts ``str`` or ``pathlib.Path``.
        filename: Name of the saved stats file.  Default matches the
            convention used by :class:`mtg.training.train.Trainer`.

    Returns:
        Either the original ``vec_env`` (no stats found) or a frozen
        :class:`VecNormalize` wrapper around it.
    """
    from pathlib import Path

    path = Path(run_dir) / filename
    if not path.exists():
        return vec_env

    from stable_baselines3.common.vec_env import VecNormalize

    vec_env = VecNormalize.load(str(path), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env


def evaluate_policy_on_env(
    env: tp.Any,
    agent: tp.Any,
    n_episodes: int,
    max_steps_per_episode: int | None = None,
    deterministic: bool = True,
    obs_normaliser: tp.Callable[[np.ndarray], np.ndarray] | None = None,
    seed_offset: int = 0,
) -> dict[str, tp.Any]:
    """Run ``n_episodes`` evaluation games on a live (non-vectorised) env.

    Returns raw per-episode rewards, wins and lengths alongside aggregate
    metrics.  Callers can derive bootstrap CIs / per-seed statistics from
    the raw lists.

    Args:
        env: Live MTG env (typically wrapped in ``ActionMasker``).
        agent: Agent with ``select_action`` and (optional) ``deterministic`` attr.
        n_episodes: Number of evaluation episodes.
        max_steps_per_episode: Hard step cap; defaults to env attribute.
        deterministic: Whether to set ``agent.deterministic``.
        obs_normaliser: Optional callable applied to every observation
            (after ``reset`` and after every ``step``). Used to plug in
            the frozen ``VecNormalize`` running statistics from
            training, so a policy trained on normalised observations
            is evaluated on the same distribution.
        seed_offset: Offset added to per-episode seeds so multiple
            evaluation passes can use disjoint seeds.
    """
    prev_deterministic = getattr(agent, "deterministic", None)
    if hasattr(agent, "deterministic"):
        agent.deterministic = deterministic

    step_cap = max_steps_per_episode or getattr(env.unwrapped, "max_steps_per_episode", 500)

    def _maybe_norm(obs: np.ndarray) -> np.ndarray:
        return obs_normaliser(obs) if obs_normaliser is not None else obs

    rewards: list[float] = []
    lengths: list[int] = []
    wins: list[bool] = []

    try:
        for ep in range(n_episodes):
            obs, info = env.reset(seed=seed_offset + ep)
            obs = _maybe_norm(obs)
            done = False
            ep_reward = 0.0
            ep_length = 0
            while not done and ep_length < step_cap:
                action_mask = info.get(
                    "action_mask",
                    np.ones(env.action_space.n, dtype=bool),
                )
                action = agent.select_action(obs, action_mask, info)
                obs, r, terminated, truncated, info = env.step(action)
                obs = _maybe_norm(obs)
                ep_reward += float(r)
                ep_length += 1
                done = terminated or truncated
            rewards.append(ep_reward)
            lengths.append(ep_length)
            wins.append(_is_win(info))
    finally:
        if prev_deterministic is not None and hasattr(agent, "deterministic"):
            agent.deterministic = prev_deterministic

    wins_arr = np.asarray(wins, dtype=float)
    rewards_arr = np.asarray(rewards, dtype=float)
    lengths_arr = np.asarray(lengths, dtype=float)

    return {
        "n_episodes": n_episodes,
        "win_rate": float(wins_arr.mean()) if len(wins_arr) else 0.0,
        "mean_reward": float(rewards_arr.mean()) if len(rewards_arr) else 0.0,
        "std_reward": float(rewards_arr.std()) if len(rewards_arr) else 0.0,
        "mean_length": float(lengths_arr.mean()) if len(lengths_arr) else 0.0,
        "std_length": float(lengths_arr.std()) if len(lengths_arr) else 0.0,
        "rewards": rewards,
        "lengths": lengths,
        "wins": wins,
    }


def _is_win(info: dict[str, tp.Any]) -> bool:
    result = info.get("game_result")
    if isinstance(result, str):
        return result.lower() == "win"
    winner = info.get("winner")
    if isinstance(winner, str):
        return winner.lower() in {"player", "win", "agent"}
    if isinstance(winner, int | np.integer):
        return winner == 0
    return False

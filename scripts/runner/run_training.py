#!/usr/bin/env python3
"""Training script for MTG-Causal-RL agents.

This script provides both interactive and command-line modes for training
RL agents (PPO, Causal) on the MTG environment with configurable parameters.

Features:
- Only trainable agents (PPO, Causal) are selectable
- Multi-opponent training for robust agents
- Live training display with metrics
- Post-training evaluation with progress bar
- Intuitive model naming: {agent_type}_{player_deck}

Usage:
    # Interactive mode (prompts for all settings)
    uv run python scripts/runner/run_training.py --interactive

    # Command-line mode
    uv run python scripts/runner/run_training.py --agent ppo --deck mono_red_aggro

    # Multi-opponent training (recommended)
    uv run python scripts/runner/run_training.py --agent ppo --deck mono_red_aggro --opponent all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import typing as tp
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from mtg.utils.cli_display import (
    TrainingDisplay,
    console,
    print_divider,
    print_evaluation_results,
    print_logo,
)
from mtg.utils.interactive import (
    TrainingConfig,
    confirm_config,
    create_output_directory,
    format_duration,
    get_available_archetypes,
    prompt_training_config,
)

# ─── Round-robin training constants ────────────────────────────────────────
# Episodes between opponent swaps in round-robin mode.  Swaps are deferred
# to PPO rollout boundaries so each rollout buffer stays opponent-homogeneous
# (preserving GAE advantage estimates).
RR_ROTATE_EVERY: int = 50
# Rough heuristic for MTG episode length with auto_resolve.  Used to size
# PPO rollouts so a single rollout naturally contains ≈ ``RR_ROTATE_EVERY``
# episodes — i.e., the rotation cadence promised by the console message
# is honored in practice rather than dominated by the rollout boundary.
RR_AVG_EP_LEN: int = 70
# Floor on n_steps when auto-tuning for round-robin: a rollout buffer
# smaller than this destabilizes PPO regardless of the rotation goal.
RR_MIN_N_STEPS: int = 128


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.

    """
    parser = argparse.ArgumentParser(
        description="Train an RL agent on MTG-Causal-RL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode selection
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode with prompts",
    )

    # Agent configuration (only trainable agents)
    parser.add_argument(
        "--agent",
        type=str,
        default="ppo",
        choices=["ppo", "causal"],
        help="Agent type to train",
    )

    # Deck configuration
    parser.add_argument(
        "--deck",
        type=str,
        default="mono_red_aggro",
        choices=get_available_archetypes(),
        help="Agent deck archetype",
    )

    parser.add_argument(
        "--opponent",
        type=str,
        default="all",
        help="Opponent deck(s): 'all' for multi-opponent, or a specific archetype name",
    )

    # Training parameters
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help=(
            "Total training timesteps. Recommended: 500K (quick test), "
            "1M (standard), 2M+ (paper-quality)."
        ),
    )

    parser.add_argument(
        "--reward",
        type=str,
        default="shaped",
        choices=["sparse", "shaped", "dense"],
        help="Reward type",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help=("Max MTG turns per game. 5 (fast), 10 (quick iteration), 20 (standard, default)."),
    )

    parser.add_argument(
        "--n-envs",
        type=str,
        default="4",
        help=(
            "Number of parallel environments for rollout collection. "
            "Use 'auto' to match available CPU cores, or 1 to disable parallelism."
        ),
    )

    parser.add_argument(
        "--training-mode",
        type=str,
        default="round-robin",
        choices=["round-robin", "sequential"],
        help=(
            "Multi-opponent training mode. 'round-robin' interleaves opponents "
            "(recommended); 'sequential' trains full budget per opponent in order."
        ),
    )

    # Agency mode
    parser.add_argument(
        "--agency",
        type=str,
        default="auto",
        choices=["auto", "full", "curriculum"],
        help=(
            "Agent decision agency mode. "
            "'auto': simplified combat/targeting (fastest learning). "
            "'full': agent controls individual attackers + spell targets (needs 3M+ steps). "
            "'curriculum': auto for 70%% of budget then full for 30%% "
            "(recommended for the Causal RL agent; not needed for vanilla PPO baseline)."
        ),
    )

    # Output configuration
    parser.add_argument(
        "--output",
        type=str,
        default="results/trained_agents",
        help="Output directory for saved models and artifacts",
    )

    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Final evaluation episodes per opponent",
    )
    parser.add_argument(
        "--quick-eval-episodes",
        type=int,
        default=20,
        help=(
            "Episodes per opponent for the in-line 'Quick evaluation' "
            "shown right after training (deterministic; sanity check only)."
        ),
    )

    # Sample report configuration
    parser.add_argument(
        "--sample-games",
        type=int,
        default=3,
        help="Number of sample games per opponent to record as HTML reports after training",
    )

    parser.add_argument(
        "--sample-opponents",
        type=str,
        default="",
        help="Opponents for sample games (comma-separated). Default: same as training opponents",
    )

    return parser.parse_args()


def _resolve_n_envs(value: str) -> int:
    """Resolve ``--n-envs`` value: ``'auto'`` -> CPU count, else parse int."""
    if value.strip().lower() == "auto":
        import os

        return os.cpu_count() or 4
    return int(value)


def args_to_config(args: argparse.Namespace) -> TrainingConfig:
    """Convert command-line args to TrainingConfig.

    Args:
        args: Parsed arguments.

    Returns:
        TrainingConfig instance.

    """
    # Handle 'all' opponent
    if args.opponent == "all":
        opponent_deck = ",".join(get_available_archetypes())
    else:
        opponent_deck = args.opponent

    return TrainingConfig(
        agent_type=args.agent,
        player_deck=args.deck,
        opponent_deck=opponent_deck,
        timesteps=args.timesteps,
        reward_type=args.reward,
        seed=args.seed,
        max_turns=args.max_turns,
        n_envs=_resolve_n_envs(args.n_envs),
        training_mode=args.training_mode,
        agency_mode=args.agency,
        eval_episodes=args.eval_episodes,
        quick_eval_episodes=args.quick_eval_episodes,
        sample_games=args.sample_games,
        sample_opponents=args.sample_opponents,
        output_dir=args.output,
    )


def get_default_agent_for_deck(deck_name: str) -> str:
    """Return the default heuristic agent for a deck archetype.

    Matches the logic from run_gameplay.py to ensure consistency.

    Args:
        deck_name: Deck archetype name.

    Returns:
        Heuristic agent name.

    """
    from mtg.agents import heuristic_for_deck

    normalized = deck_name.lower().replace(" ", "_").replace("-", "_")
    return heuristic_for_deck(normalized) or "greedy_aggro"


def create_env(
    player_deck: str,
    opponent_deck: str,
    reward_type: str = "shaped",
    seed: int = 42,
    max_turns: int = 10,
    max_steps_per_episode: int = 500,
    use_heuristic_opponent: bool = True,
    auto_combat: bool = False,
    auto_target: bool = False,
    cgfa_factor_spec: tp.Any = None,
    cgfa_scm: tp.Any = None,
    cgfa_calibration_mode: str = "factual",
) -> tp.Any:
    """Create and configure the training environment.

    Args:
        player_deck: Player deck archetype.
        opponent_deck: Opponent deck archetype.
        reward_type: Reward shaping type.
        seed: Random seed.
        max_turns: Maximum MTG turns per game.
        max_steps_per_episode: Safety limit on steps per episode.
        use_heuristic_opponent: If True, use heuristic agent for opponent
            (matches gameplay). If False, use built-in opponent logic.
        auto_combat: If True, attacks are all-or-nothing. If False (default),
            agent selects individual attackers.
        auto_target: If True, spell targets are auto-picked. If False
            (default), agent selects targets from candidates.
        cgfa_factor_spec: Optional :class:`FactorSpec` from
            :mod:`mtg.agents.reinforcement_learning.cgfa`. When provided
            the env is wrapped in :class:`CGFAEnvWrapper` so per-factor
            signals (``factor_values`` / ``factor_rewards`` /
            ``factor_eps``) appear on every ``info`` dict, which the
            CGFA-PPO algorithm reads from the rollout buffer.
        cgfa_scm: Optional :class:`StructuralCausalModel` used by
            :class:`CGFAEnvWrapper` to compute ``factor_eps``. Ignored
            when ``cgfa_factor_spec`` is None.
        cgfa_calibration_mode: CGFA factor-epsilon target mode passed to
            :class:`CGFAEnvWrapper` (``"factual"`` or ``"interventional"``).

    Returns:
        Configured MTG environment.

    """
    from mtg.agents import get_agent
    from mtg.env import MTGEnv

    opponent_agent = None
    if use_heuristic_opponent:
        agent_name = get_default_agent_for_deck(opponent_deck)
        opponent_agent = get_agent(agent_name)

    env = MTGEnv(
        deck_archetype=player_deck,
        opponent_archetype=opponent_deck,
        max_turns=max_turns,
        max_steps_per_episode=max_steps_per_episode,
        reward_type=reward_type,
        seed=seed,
        auto_combat=auto_combat,
        auto_target=auto_target,
        opponent_agent=opponent_agent,
    )
    if cgfa_factor_spec is not None:
        from mtg.agents.reinforcement_learning.cgfa import CGFAEnvWrapper

        env = CGFAEnvWrapper(
            env,
            factor_spec=cgfa_factor_spec,
            scm=cgfa_scm,
            calibration_mode=cgfa_calibration_mode,
        )
    return env


def make_env_fn(
    player_deck: str,
    opponent_deck: str,
    reward_type: str,
    seed: int,
    max_turns: int,
    max_steps_per_episode: int,
    auto_combat: bool = False,
    auto_target: bool = False,
    cgfa_factor_spec: tp.Any = None,
    cgfa_scm: tp.Any = None,
    cgfa_calibration_mode: str = "factual",
) -> tp.Callable[[], tp.Any]:
    """Return a factory that creates a single ActionMasker-wrapped MTG env.

    Used by SubprocVecEnv which needs callables, not pre-built envs.
    MaskablePPO requires ActionMasker wrapping on each sub-env.

    When ``cgfa_factor_spec`` is provided, the env is additionally wrapped
    in :class:`CGFAEnvWrapper` *before* the ActionMasker so per-factor
    signals appear on every ``info`` dict that reaches the rollout buffer.
    """
    from sb3_contrib.common.wrappers import ActionMasker

    from mtg.agents.reinforcement_learning.ppo_agent import _mask_fn

    def _init() -> tp.Any:
        env = create_env(
            player_deck,
            opponent_deck,
            reward_type,
            seed,
            max_turns=max_turns,
            max_steps_per_episode=max_steps_per_episode,
            auto_combat=auto_combat,
            auto_target=auto_target,
            cgfa_factor_spec=cgfa_factor_spec,
            cgfa_scm=cgfa_scm,
            cgfa_calibration_mode=cgfa_calibration_mode,
        )
        return ActionMasker(env, _mask_fn)

    return _init


def create_vec_env(
    player_deck: str,
    opponent_deck: str,
    reward_type: str,
    seed: int,
    max_turns: int,
    max_steps_per_episode: int,
    n_envs: int = 1,
    auto_combat: bool = False,
    auto_target: bool = False,
    cgfa_factor_spec: tp.Any = None,
    cgfa_scm: tp.Any = None,
    cgfa_calibration_mode: str = "factual",
) -> tp.Any:
    """Create a vectorized training environment.

    When n_envs > 1, uses SubprocVecEnv for parallel rollout collection.
    When n_envs == 1, wraps a single env in DummyVecEnv so the returned
    object always satisfies the VecEnv interface (callers like the
    round-robin proxy depend on attributes such as ``num_envs``).

    When ``cgfa_factor_spec`` is provided, every sub-env is wrapped in
    :class:`CGFAEnvWrapper` so per-factor signals appear on every
    ``info`` dict produced by the vec env (required by CGFA-PPO).
    """
    env_fns = [
        make_env_fn(
            player_deck,
            opponent_deck,
            reward_type,
            seed + i,
            max_turns,
            max_steps_per_episode,
            auto_combat=auto_combat,
            auto_target=auto_target,
            cgfa_factor_spec=cgfa_factor_spec,
            cgfa_scm=cgfa_scm,
            cgfa_calibration_mode=cgfa_calibration_mode,
        )
        for i in range(max(1, n_envs))
    ]
    if n_envs <= 1:
        from stable_baselines3.common.vec_env import DummyVecEnv

        return DummyVecEnv(env_fns)

    from stable_baselines3.common.vec_env import SubprocVecEnv

    return SubprocVecEnv(env_fns)


def create_agent(
    agent_type: str,
    env: tp.Any,
    seed: int = 42,
    total_timesteps: int = 100_000,
    cgfa_factor_spec: tp.Any = None,
    cgfa_scm: tp.Any = None,
    is_round_robin: bool = False,
    agent_kwargs: dict[str, tp.Any] | None = None,
) -> tp.Any:
    """Create and initialize a trainable agent.

    The PPO rollout buffer size (n_steps) is adapted to the training budget
    so that multiple gradient updates happen even with small step counts.

    Args:
        agent_type: Agent type ('ppo', 'causal', or 'cgfa').
        env: Training environment. For CGFA, this must already be wrapped
            with :class:`CGFAEnvWrapper` (callers should use
            ``create_vec_env(..., cgfa_factor_spec=spec, cgfa_scm=scm)``).
        seed: Random seed.
        total_timesteps: Total training budget (used to tune n_steps).
        cgfa_factor_spec: Required when ``agent_type == "cgfa"``. The
            same :class:`FactorSpec` instance used to build the env so
            the agent's per-factor heads match the env's per-factor
            signals.
        cgfa_scm: Required when ``agent_type == "cgfa"``. The same
            :class:`StructuralCausalModel` instance used to build the env
            so the intervention-calibration loss compares the same SCM.
        is_round_robin: When ``True``, ``n_steps`` is auto-tuned so each
            PPO rollout naturally contains ≈ :data:`RR_ROTATE_EVERY`
            episodes (≈ ``RR_ROTATE_EVERY * RR_AVG_EP_LEN`` wall steps).
            This honors the round-robin rotation cadence promised in
            ``_train_round_robin``: opponent swaps are deferred to rollout
            boundaries, so a rollout much larger than the cadence target
            collapses many "every 50 episodes" triggers into a single
            late swap.  Required for fair multi-opponent interleaving.
        agent_kwargs: Extra keyword arguments forwarded to the selected
            agent constructor after runner-managed rollout settings.

    Returns:
        Initialized agent.

    """
    from mtg.agents.causal.causal_agent import CausalAgent
    from mtg.agents.causal.cgfa_agent import CGFAAgent
    from mtg.agents.reinforcement_learning.ppo_agent import PPOAgent

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    extra_agent_kwargs = dict(agent_kwargs or {})

    n_envs_actual = getattr(env, "num_envs", 1)

    if is_round_robin:
        # Round-robin: size each rollout to ≈ RR_ROTATE_EVERY episodes so
        # the swap promised "every N episodes" actually fires every (or
        # nearly every) rollout boundary.  Without this, the default
        # n_steps=4096 with n_envs=15 produces 61k-step rollouts that
        # only swap every ~900 episodes — the agent over-fits to one
        # opponent for whole rollouts at a time.
        target_rollout_steps = RR_ROTATE_EVERY * RR_AVG_EP_LEN
        n_steps = max(RR_MIN_N_STEPS, target_rollout_steps // max(n_envs_actual, 1))
    else:
        # n_steps must cover several full episodes so PPO sees complete
        # win/loss trajectories for proper credit assignment.  MTG
        # episodes are typically 70-120 steps with auto_resolve, so 2048
        # captures ~20+ episodes.  For very small budgets, fall back to
        # something sensible.
        n_steps = max(512, min(4096, total_timesteps // 4))
    batch_size = min(256, n_steps)
    # Ensure rollout buffer (n_steps * n_envs) is divisible by batch_size
    # to avoid SB3 truncated-minibatch warnings and gradient noise.
    rollout_size = n_steps * n_envs_actual
    if rollout_size % batch_size != 0:
        batch_size = max(1, rollout_size // (rollout_size // batch_size))

    if agent_type == "cgfa":
        if cgfa_factor_spec is None or cgfa_scm is None:
            raise ValueError(
                "create_agent: cgfa agent requires cgfa_factor_spec and cgfa_scm; "
                "the env must also be built with the same instances via "
                "create_vec_env(..., cgfa_factor_spec=..., cgfa_scm=...)."
            )
        # These are runner-managed so the env wrapper and agent share the
        # exact same instances without passing duplicate constructor kwargs.
        extra_agent_kwargs.pop("factor_spec", None)
        extra_agent_kwargs.pop("scm", None)
        agent = CGFAAgent(
            observation_dim=obs_dim,
            action_dim=act_dim,
            seed=seed,
            n_steps=n_steps,
            batch_size=batch_size,
            factor_spec=cgfa_factor_spec,
            scm=cgfa_scm,
            **extra_agent_kwargs,
        )
    elif agent_type == "causal":
        agent = CausalAgent(
            observation_dim=obs_dim,
            action_dim=act_dim,
            seed=seed,
            **extra_agent_kwargs,
        )
        # Configure the internal PPO agent's hyperparameters
        agent.base_agent.n_steps = n_steps
        agent.base_agent.batch_size = batch_size
    else:
        agent = PPOAgent(
            observation_dim=obs_dim,
            action_dim=act_dim,
            seed=seed,
            n_steps=n_steps,
            batch_size=batch_size,
            **extra_agent_kwargs,
        )

    agent.initialize_model(env)
    return agent


def _anneal_entropy(model: tp.Any, num_timesteps: int) -> None:
    """Update the model's entropy coefficient based on training progress.

    Uses ``PPOAgent.get_ent_coef_for_progress`` if available.  The
    denominator is ``_entropy_total_budget`` (set once at the start of
    training to cover all phases) so entropy anneals smoothly across
    curriculum phases instead of resetting per ``learn()`` call.
    """
    if model is None:
        return
    total = getattr(model, "_entropy_total_budget", 0)
    if total <= 0:
        total = getattr(model, "_total_timesteps", 0)
    if total <= 0:
        return
    progress = max(0.0, 1.0 - num_timesteps / total)
    ppo_agent = getattr(model, "_ppo_agent_ref", None)
    if ppo_agent is not None and hasattr(ppo_agent, "get_ent_coef_for_progress"):
        model.ent_coef = ppo_agent.get_ent_coef_for_progress(progress)


def _maybe_attach_cgfa_callback(
    config: TrainingConfig,
    base_callback: tp.Any,
    output_dir: Path,
) -> tp.Any:
    """Wrap ``base_callback`` with ``CGFACalibrationCallback`` for CGFA runs.

    The wrapping makes the CGFA-PPO trainer emit ``cgfa_calibration.csv``
    alongside the run artefacts (consumed by
    ``mtg-research calibration-plot``).  For non-CGFA agents this is a
    no-op and ``base_callback`` is returned unchanged.
    """
    if config.agent_type != "cgfa":
        return base_callback
    try:
        from stable_baselines3.common.callbacks import CallbackList
    except ImportError:  # pragma: no cover - sb3_contrib only path
        from sb3_contrib.common.callbacks import CallbackList  # type: ignore[no-redef]
    from mtg.agents.reinforcement_learning.cgfa import CGFACalibrationCallback

    cgfa_cb = CGFACalibrationCallback(log_dir=str(output_dir / "cgfa"))
    return CallbackList([base_callback, cgfa_cb])


def create_sb3_callback(
    display: TrainingDisplay,
    update_every: int = 64,
    on_step_hook: tp.Callable[[int], None] | None = None,
    causal_agent: tp.Any = None,
) -> tp.Any:
    """Create an SB3-compatible callback that updates the training display.

    This bridges SB3's callback system with our Rich live display,
    reporting timesteps, episodes, win rate, reward, and FPS.

    Args:
        display: TrainingDisplay instance to update.
        update_every: Update the display every N steps. Lower = more responsive.
        on_step_hook: Optional callable(num_timesteps) invoked every display
            update, e.g. for periodic checkpointing.
        causal_agent: Optional CausalAgent whose WinProbLearner should be
            fed game outcomes at the end of each episode.

    Returns:
        SB3 BaseCallback instance.

    """
    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError:
        from sb3_contrib.common.callbacks import BaseCallback  # type: ignore[no-redef]

    # Import once at callback-creation time, not inside the hot loop
    _causal_transition_cls = None
    _causal_vars_to_array = None
    if causal_agent is not None and hasattr(causal_agent, "causal_world_model"):
        from mtg.causal.causal_world_model import CausalTransition, causal_vars_to_array

        _causal_transition_cls = CausalTransition
        _causal_vars_to_array = causal_vars_to_array

    class DisplayCallback(BaseCallback):
        """SB3 callback that updates the Rich training display."""

        def __init__(self) -> None:
            super().__init__(verbose=0)
            self._episode_rewards: list[float] = []
            self._episode_wins: list[float] = []
            self._episode_lengths: list[int] = []
            self._current_rewards: dict[int, float] = {}
            self._current_lengths: dict[int, int] = {}
            self._start_time = time.time()
            self._last_update = 0
            self._last_loss: float = 0.0
            self._prev_causal_vars: dict[int, np.ndarray] = {}

        def _on_step(self) -> bool:
            _anneal_entropy(self.model, self.num_timesteps)

            dones = self.locals.get("dones", np.array([False]))
            infos = self.locals.get("infos", [{}])
            rewards = self.locals.get("rewards", np.array([0.0]))
            obs_tensor = self.locals.get("obs_tensor")
            actions = self.locals.get("actions")

            for i in range(len(dones)):
                r = float(rewards[i]) if i < len(rewards) else 0.0
                self._current_rewards[i] = self._current_rewards.get(i, 0.0) + r
                self._current_lengths[i] = self._current_lengths.get(i, 0) + 1

                # Record CWM transitions from rollout data
                if _causal_transition_cls is not None:
                    info = infos[i] if i < len(infos) else {}
                    cv = info.get("causal_variables", {})
                    if cv and i in self._prev_causal_vars:
                        prev_cv = self._prev_causal_vars[i]
                        curr_cv = _causal_vars_to_array(cv)
                        obs_np = (
                            obs_tensor[i].cpu().numpy()
                            if obs_tensor is not None and i < len(obs_tensor)
                            else np.zeros(causal_agent.observation_dim)
                        )
                        act = int(actions[i]) if actions is not None and i < len(actions) else 0
                        is_terminal = bool(dones[i])
                        outcome = 0.5
                        if is_terminal:
                            game_res = info.get("game_result", "")
                            if game_res == "win":
                                outcome = 1.0
                            elif game_res == "loss":
                                outcome = 0.0
                            # draw stays 0.5
                        causal_agent.causal_world_model.record_transition(
                            _causal_transition_cls(
                                obs=obs_np,
                                action=act,
                                causal_vars=prev_cv,
                                next_causal_vars=curr_cv,
                                terminal=is_terminal,
                                outcome=outcome,
                            )
                        )
                    if cv:
                        self._prev_causal_vars[i] = _causal_vars_to_array(cv)

                if bool(dones[i]):
                    ep_reward = self._current_rewards.pop(i, 0.0)
                    ep_length = self._current_lengths.pop(i, 0)
                    self._episode_rewards.append(ep_reward)
                    self._episode_lengths.append(ep_length)

                    info = infos[i] if i < len(infos) else {}
                    terminal_info = info.get("terminal_info", info)
                    game_result = terminal_info.get("game_result", "")
                    self._episode_wins.append(1.0 if game_result == "win" else 0.0)

                    if causal_agent is not None and game_result in ("win", "loss"):
                        cv = terminal_info.get("causal_variables", {})
                        if cv:
                            mapped = causal_agent._to_scm_causal_vars(cv, terminal_info)
                            causal_agent.record_game_outcome(
                                mapped, 1.0 if game_result == "win" else 0.0
                            )
                    if i in self._prev_causal_vars:
                        del self._prev_causal_vars[i]

            # Extract loss from SB3's internal logger after each training update.
            # PPO/MaskablePPO logs train/loss, train/policy_gradient_loss,
            # train/value_loss, etc. to self.model.logger.name_to_value.
            loss = 0.0
            if hasattr(self, "model") and self.model is not None:
                logger = getattr(self.model, "logger", None)
                if logger is not None:
                    name_to_value = getattr(logger, "name_to_value", {})
                    # Total loss (sum of policy + value + entropy losses)
                    loss = name_to_value.get("train/loss", 0.0)
                    if loss == 0.0:
                        # Fallback: sum individual components
                        pg_loss = abs(
                            name_to_value.get(
                                "train/policy_gradient_loss",
                                0.0,
                            )
                        )
                        v_loss = name_to_value.get("train/value_loss", 0.0)
                        ent_loss = abs(
                            name_to_value.get(
                                "train/entropy_loss",
                                0.0,
                            )
                        )
                        loss = pg_loss + v_loss + ent_loss

            # PPO total loss can be negative (entropy is subtracted),
            # so track it whenever it has been computed (non-zero).
            if loss != 0.0:
                self._last_loss = loss

            # Update display frequently for responsive feedback
            if self.num_timesteps - self._last_update >= update_every:
                self._last_update = self.num_timesteps
                elapsed = time.time() - self._start_time
                fps = self.num_timesteps / max(elapsed, 1e-6)

                n_episodes = len(self._episode_rewards)
                recent = min(50, n_episodes)

                win_rate = float(np.mean(self._episode_wins[-recent:])) if recent > 0 else 0.0
                avg_reward = float(np.mean(self._episode_rewards[-recent:])) if recent > 0 else 0.0
                avg_length = float(np.mean(self._episode_lengths[-recent:])) if recent > 0 else 0.0

                display.update(
                    timesteps=self.num_timesteps,
                    episodes=n_episodes,
                    win_rate=win_rate,
                    avg_reward=avg_reward,
                    episode_length=avg_length,
                    loss=self._last_loss if self._last_loss != 0.0 else None,
                    fps=fps,
                )

                if on_step_hook is not None:
                    on_step_hook(self.num_timesteps)

            return True

        def _on_rollout_end(self) -> None:
            """Train the CausalWorldModel after each PPO rollout."""
            if causal_agent is not None and hasattr(causal_agent, "train_causal_world_model"):
                causal_agent.train_causal_world_model()

    return DisplayCallback()


def evaluate_agent(
    env: tp.Any,
    agent: tp.Any,
    n_episodes: int,
    max_steps_per_episode: int = 500,
    base_seed: int | None = 42,
    obs_normaliser: tp.Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, float]:
    """Evaluate an agent over multiple episodes with progress display.

    Each episode resets the environment with a deterministic seed
    ``base_seed + episode_idx`` so results are reproducible across runs
    (matches the formal :class:`mtg.training.evaluate.Evaluator`
    behaviour). Set ``base_seed=None`` to disable seeded resets and
    fall back to non-deterministic episode draws.

    If the trained model used ``VecNormalize`` for observation
    statistics, pass an ``obs_normaliser`` callable (e.g. produced via
    :func:`mtg.training.env_factory.make_obs_normaliser_from_vec_normalize`)
    so the policy sees normalised observations at eval time. Without
    this the train and eval distributions silently disagree and
    inflate apparent regression risk.

    Args:
        env: Evaluation environment.
        agent: Agent to evaluate.
        n_episodes: Number of episodes.
        max_steps_per_episode: Maximum steps per episode to prevent hangs.
        base_seed: Base seed for per-episode resets.  ``None`` disables
            seeding.
        obs_normaliser: Optional callable applied to every observation
            before the agent sees it.

    Returns:
        Dictionary of evaluation metrics.
    """
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeRemainingColumn,
    )

    prev_deterministic = getattr(agent, "deterministic", None)
    agent.deterministic = True

    wins = 0
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []

    def _norm(obs: np.ndarray) -> np.ndarray:
        return obs_normaliser(obs) if obs_normaliser is not None else obs

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, complete_style="green"),
        TaskProgressColumn(),
        TextColumn("•"),
        TextColumn("[green]Win: {task.fields[win_rate]}[/green]"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Evaluating",
            total=n_episodes,
            win_rate="0.0%",
        )

        for ep in range(n_episodes):
            reset_seed = None if base_seed is None else int(base_seed) + ep
            if reset_seed is None:
                obs, info = env.reset()
            else:
                obs, info = env.reset(seed=reset_seed)
            obs = _norm(obs)
            done = False
            ep_reward = 0.0
            ep_length = 0

            while not done and ep_length < max_steps_per_episode:
                action_mask = info.get(
                    "action_mask",
                    np.ones(env.action_space.n, dtype=bool),
                )
                action = agent.select_action(obs, action_mask, info)
                obs, reward, terminated, truncated, info = env.step(action)
                obs = _norm(obs)
                ep_reward += reward
                ep_length += 1
                done = terminated or truncated

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)

            if info.get("game_result") == "win":
                wins += 1

            # Update progress
            current_wr = wins / (ep + 1)
            progress.update(
                task,
                advance=1,
                win_rate=f"{current_wr:.1%}",
            )

    if prev_deterministic is not None:
        agent.deterministic = prev_deterministic

    wr = wins / n_episodes
    # Wilson score interval for binomial proportion (95% CI)
    z = 1.96
    denom = 1 + z**2 / n_episodes
    margin = z * np.sqrt((wr * (1 - wr) + z**2 / (4 * n_episodes)) / n_episodes) / denom
    return {
        "win_rate": wr,
        "win_rate_ci95": float(margin),
        "avg_reward": float(np.mean(episode_rewards)),
        "reward_std": float(np.std(episode_rewards)),
        "avg_episode_length": float(np.mean(episode_lengths)),
        "episode_length_std": float(np.std(episode_lengths)),
        "n_episodes": int(n_episodes),
        "base_seed": None if base_seed is None else int(base_seed),
    }


def generate_training_plots(
    output_dir: Path,
    display: TrainingDisplay | None,
    eval_results: dict[str, dict[str, float]],
    config: TrainingConfig,
    all_displays: dict[str, TrainingDisplay] | None = None,
) -> list[Path]:
    """Generate training curve plots and save to output directory.

    Args:
        output_dir: Output directory.
        display: TrainingDisplay with metrics history (for single-opponent).
        eval_results: Evaluation results per opponent.
        config: Training configuration.
        all_displays: Per-opponent displays (for multi-opponent training).

    Returns:
        List of saved plot paths.

    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        console.print("[yellow]matplotlib not installed, skipping plots[/yellow]")
        return []

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    saved: list[Path] = []

    # Determine which displays to plot
    if all_displays and len(all_displays) > 1:
        # Multi-opponent: plot separate curves for each
        displays_to_plot = all_displays
        title_suffix = " (Multi-Opponent)"
    elif display is not None:
        # Single opponent: use the provided display
        displays_to_plot = {display.opponent: display}
        title_suffix = ""
    else:
        return saved

    # Check if we have any history
    if not any(d.metrics.history for d in displays_to_plot.values()):
        return saved

    # Color palette for multiple opponents
    colors = [
        "#2A9D8F",  # Teal
        "#E63946",  # Red
        "#457B9D",  # Blue
        "#E9C46A",  # Yellow
        "#F77F00",  # Orange
        "#9B59B6",  # Purple
        "#1ABC9C",  # Turquoise
    ]

    # --- Plot 1: Training Curves (win rate, reward, episode length) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Training: {config.agent_type.upper()} on {config.player_deck}{title_suffix}",
        fontsize=14,
        fontweight="bold",
    )

    def _rolling_mean(values: list[float], window: int) -> list[float]:
        """Centered rolling mean that preserves length; robust to short series."""
        if not values:
            return []
        n = len(values)
        w = max(1, min(window, n))
        out: list[float] = []
        for i in range(n):
            lo = max(0, i - w // 2)
            hi = min(n, lo + w)
            lo = max(0, hi - w)
            out.append(float(sum(values[lo:hi]) / max(1, hi - lo)))
        return out

    def _plot_with_smoothing(
        ax: tp.Any,
        values: list[float],
        color: str,
        label: str,
        x_offset: int = 0,
    ) -> None:
        if not values:
            return
        window = max(5, min(51, len(values) // 20))
        smoothed = _rolling_mean(values, window)
        xs = range(x_offset, x_offset + len(values))
        ax.plot(xs, values, color=color, linewidth=0.7, alpha=0.18)
        ax.plot(xs, smoothed, color=color, linewidth=2.0, label=label, alpha=0.95)

    # Plot each opponent's metrics (raw faded + smoothed rolling mean bold)
    for idx, (opp_name, disp) in enumerate(displays_to_plot.items()):
        history = disp.metrics.history
        if not history:
            continue

        color = colors[idx % len(colors)]
        label = opp_name.replace("_", " ").title()

        wr_data = [h.get("win_rate", 0) for h in history]
        _plot_with_smoothing(axes[0, 0], wr_data, color, label)

        rw_data = [h.get("avg_reward", 0) for h in history]
        _plot_with_smoothing(axes[0, 1], rw_data, color, label)

        el_data = [h.get("episode_length", 0) for h in history]
        _plot_with_smoothing(axes[1, 0], el_data, color, label)

        loss_data = [h.get("loss", 0) for h in history]
        nonzero_indices = [i for i, v in enumerate(loss_data) if v is not None and v != 0]
        if nonzero_indices:
            start_idx = nonzero_indices[0]
            trimmed_loss = loss_data[start_idx:]
            _plot_with_smoothing(axes[1, 1], trimmed_loss, color, label, x_offset=start_idx)

    # Configure axes
    axes[0, 0].set_title("Win Rate")
    axes[0, 0].set_ylabel("Win Rate")
    axes[0, 0].set_ylim(-0.05, 1.05)
    axes[0, 0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    axes[0, 0].grid(True, alpha=0.3)
    if len(displays_to_plot) > 1:
        axes[0, 0].legend(loc="best", fontsize=8)

    axes[0, 1].set_title("Average Reward")
    axes[0, 1].set_ylabel("Reward")
    axes[0, 1].grid(True, alpha=0.3)
    if len(displays_to_plot) > 1:
        axes[0, 1].legend(loc="best", fontsize=8)

    axes[1, 0].set_title("Episode Length")
    axes[1, 0].set_ylabel("Steps")
    axes[1, 0].set_xlabel("Training Step (×32)")
    axes[1, 0].grid(True, alpha=0.3)
    if len(displays_to_plot) > 1:
        axes[1, 0].legend(loc="best", fontsize=8)

    # Loss plot
    if not any(
        d.metrics.history
        for d in displays_to_plot.values()
        if any(h.get("loss", 0) != 0 for h in d.metrics.history)
    ):
        axes[1, 1].text(
            0.5,
            0.5,
            "No loss data\n(needs more timesteps\nfor PPO to train)",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
            fontsize=10,
            color="gray",
        )
    axes[1, 1].set_title("Loss")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].set_xlabel("Training Step (×32)")
    axes[1, 1].grid(True, alpha=0.3)
    if len(displays_to_plot) > 1:
        axes[1, 1].legend(loc="best", fontsize=8)

    plt.tight_layout()
    path = plots_dir / "training_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # --- Plot 2: Evaluation Results Bar Chart with 95% CI error bars ---
    if eval_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        opponents = list(eval_results.keys())
        win_rates = [eval_results[o].get("win_rate", 0) for o in opponents]
        cis = [eval_results[o].get("win_rate_ci95", 0) for o in opponents]
        n_eps = [eval_results[o].get("n_episodes", config.eval_episodes) for o in opponents]

        colors = ["#2A9D8F" if wr >= 0.5 else "#E63946" for wr in win_rates]
        bars = ax.bar(
            range(len(opponents)),
            win_rates,
            color=colors,
            edgecolor="white",
            yerr=cis,
            capsize=6,
            ecolor="#333333",
            error_kw={"linewidth": 1.5, "alpha": 0.8},
        )

        for bar, wr, ci in zip(bars, win_rates, cis, strict=False):
            ax.annotate(
                f"{wr:.0%}\n±{ci:.0%}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + ci + 0.02),
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_xticks(range(len(opponents)))
        ax.set_xticklabels(
            [o.replace("_", " ").title() for o in opponents],
            rotation=25,
            ha="right",
        )
        ax.set_ylabel("Win Rate (mean over eval episodes)", fontweight="medium")
        ax.set_title(
            f"Evaluation: {config.agent_type.upper()} vs Opponents "
            f"(n={n_eps[0] if n_eps else config.eval_episodes} episodes/opponent, "
            f"95% Wilson CI)",
            fontweight="bold",
            fontsize=11,
        )
        ax.set_ylim(0, 1.20)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% baseline")
        ax.legend(loc="upper right")
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        path = plots_dir / "evaluation_results.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    return saved


def generate_sample_reports(
    agent: tp.Any,
    config: TrainingConfig,
    output_dir: Path,
    max_steps: int = 500,
    auto_combat: bool = True,
    auto_target: bool = True,
) -> list[Path]:
    """Generate HTML game reports by running sample games with the trained agent.

    Reads directly from ``env.state`` via the shared ``snapshot_from_env``
    and ``actions_from_env`` helpers so that reports are as rich and detailed
    as those produced by the gameplay workflow (creatures with P/T, tokens,
    exile zones, graveyard instant/sorcery counts, etc.).

    Args:
        agent: Trained agent to evaluate.
        config: Training configuration (includes sample_games, sample_opponents).
        output_dir: Root output directory for this training run.
        max_steps: Maximum steps per game.
        auto_combat: Whether to auto-resolve combat decisions.
        auto_target: Whether to auto-resolve targeting decisions.

    Returns:
        List of paths to saved HTML reports.

    """
    from mtg.utils.html_report import (
        GameRecorder,
        actions_from_env,
        generate_html_report,
        save_replay_json,
        snapshot_from_env,
        turn_summary_from_env,
    )

    n_games = config.sample_games
    if n_games <= 0:
        return []

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    opponent_decks = config.sample_opponent_decks
    games_per_opp = n_games

    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
    )

    total_games = games_per_opp * len(opponent_decks)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30, complete_style="green"),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        prog_task = progress.add_task("Recording sample games", total=total_games)

        game_num = 0
        for opp_deck in opponent_decks:
            env = create_env(
                config.player_deck,
                opp_deck,
                "sparse",
                config.seed + 5000,
                max_turns=config.max_turns,
                max_steps_per_episode=max_steps,
                auto_combat=auto_combat,
                auto_target=auto_target,
            )

            for _g in range(games_per_opp):
                game_num += 1
                recorder = GameRecorder(
                    player_deck=config.player_deck,
                    opponent_deck=opp_deck,
                    player_agent=config.agent_type.upper(),
                    opponent_agent="Heuristic",
                )

                obs, info = env.reset()
                done = False
                step_count = 0
                action_log_cursor = 0  # Track which actions we've already recorded
                prev_turn = 0

                player_on_play = info.get("player_on_play", True)
                recorder.set_player_on_play(player_on_play)

                # Record initial snapshot (post-reset, pre-action)
                snap = snapshot_from_env(env)
                if snap:
                    recorder.record_snapshot(**snap)

                while not done and step_count < max_steps:
                    action_mask = info.get(
                        "action_mask",
                        np.ones(env.action_space.n, dtype=bool),
                    )
                    action = agent.select_action(obs, action_mask, info)

                    obs, reward, terminated, truncated, info = env.step(action)
                    step_count += 1
                    done = terminated or truncated

                    # Record any new actions from the engine's action log
                    new_actions = actions_from_env(env, since_idx=action_log_cursor)
                    for act_kw in new_actions:
                        recorder.record_action(**act_kw)
                    if env.state:
                        action_log_cursor = len(env.state.action_log)

                    # Record a snapshot after the step
                    snap = snapshot_from_env(env)
                    if snap:
                        # Record turn summary when turn changes
                        current_turn = snap.get("turn", 0)
                        if current_turn > prev_turn and prev_turn > 0:
                            ts = turn_summary_from_env(env, prev_turn)
                            recorder.record_turn_summary(**ts)
                        prev_turn = current_turn

                        recorder.record_snapshot(**snap)

                # Record final turn summary
                if prev_turn > 0:
                    ts = turn_summary_from_env(env, prev_turn)
                    recorder.record_turn_summary(**ts)

                # Set winner
                result = info.get("game_result", "unknown")
                if result == "win":
                    recorder.set_winner("Player")
                elif result == "loss":
                    recorder.set_winner("Opponent")
                else:
                    recorder.set_winner("Draw")

                # Save HTML and JSON
                game_dir = reports_dir / f"game_{game_num}_{opp_deck}"
                game_dir.mkdir(parents=True, exist_ok=True)

                replay = recorder.get_replay()
                html_path = game_dir / "replay.html"
                json_path = game_dir / "replay.json"

                try:
                    generate_html_report(replay, html_path)
                    save_replay_json(replay, json_path)
                    saved.append(html_path)
                except Exception as e:
                    console.print(
                        f"[dim yellow]Warning: Could not save report "
                        f"for game {game_num}: {e}[/dim yellow]"
                    )

                progress.update(prog_task, advance=1)

    return saved


def save_training_artifacts(
    output_dir: Path,
    config: TrainingConfig,
    metrics: dict[str, tp.Any],
    training_time: float,
) -> None:
    """Save training configuration and metrics.

    Args:
        output_dir: Output directory.
        config: Training configuration.
        metrics: Training and evaluation metrics.
        training_time: Total training time in seconds.

    """
    # Save configuration
    config_path = output_dir / "config.yaml"
    mode = getattr(config, "training_mode", "sequential")
    n_opps = len(config.opponent_decks) if config.is_multi_opponent else 1
    total_steps = config.timesteps * n_opps if mode == "round-robin" else config.timesteps * n_opps
    config_dict = {
        "agent_type": config.agent_type,
        "agent_kwargs": dict(getattr(config, "agent_kwargs", {}) or {}),
        "player_deck": config.player_deck,
        "opponent_deck": config.opponent_deck,
        "timesteps_per_opponent": config.timesteps,
        "total_timesteps": total_steps,
        "n_opponents": n_opps,
        "reward_type": config.reward_type,
        "seed": config.seed,
        "max_turns": config.max_turns,
        "n_envs": getattr(config, "n_envs", 1),
        "training_mode": mode,
        "agency_mode": getattr(config, "agency_mode", "auto"),
        "sample_games": getattr(config, "sample_games", 3),
        "eval_episodes": config.eval_episodes,
        "quick_eval_episodes": getattr(config, "quick_eval_episodes", 20),
        "training_time_seconds": training_time,
        "timestamp": datetime.now().isoformat(),
    }
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    console.print(f"[dim]Artifacts saved to {output_dir}/[/dim]")


def _train_round_robin(
    agent: tp.Any,
    config: TrainingConfig,
    opponent_decks: list[str],
    total_timesteps: int,
    training_reward_type: str,
    train_max_steps: int,
    n_envs: int,
    checkpoint_fn: tp.Callable[[int], None] | None = None,
    auto_combat: bool | None = None,
    auto_target: bool | None = None,
    reset_timesteps: bool = True,
    output_dir: Path | None = None,
    cgfa_factor_spec: tp.Any = None,
    cgfa_scm: tp.Any = None,
) -> tuple[dict[str, TrainingDisplay], TrainingDisplay | None, list[dict[str, tp.Any]]]:
    """Run round-robin multi-opponent training.

    Uses a single ``learn()`` call with a custom callback that rotates the
    opponent environment every :data:`RR_ROTATE_EVERY` episodes, giving
    balanced exposure across all matchups.

    The effective cadence is ``max(RR_ROTATE_EVERY, rollout_episodes)``
    because swaps are deferred to PPO rollout boundaries to keep each
    rollout buffer opponent-homogeneous (preserving GAE).  When the agent
    is built via :func:`create_agent` with ``is_round_robin=True``, the
    rollout size is auto-tuned so a rollout naturally contains
    ≈ :data:`RR_ROTATE_EVERY` episodes — i.e., the cadence is honored.

    Returns:
        Tuple of (all_displays, last_display, all_metrics).

    """
    rotate_every = RR_ROTATE_EVERY

    # If the agent is a CausalAgent, feed game outcomes to its WinProbLearner.
    from mtg.agents.causal.causal_agent import CausalAgent

    _causal_agent: CausalAgent | None = agent if isinstance(agent, CausalAgent) else None

    # One live display shown in the terminal: its opponent label is
    # updated as the callback rotates. Separate "shadow" displays per
    # opponent accumulate history for the training-curves plot.
    # Stamp the rotation suffix from the start so users can see round-robin
    # is active even before the first swap fires (which only happens at the
    # end of the first PPO rollout that crosses the 50-episode threshold).
    live_display = TrainingDisplay(
        agent_name=config.agent_type.upper(),
        total_timesteps=total_timesteps,
        deck=config.player_deck,
        opponent=f"{opponent_decks[0]}  (rotation 0, cycle 1)",
    )
    all_displays: dict[str, TrainingDisplay] = {}
    for opp_deck in opponent_decks:
        all_displays[opp_deck] = TrainingDisplay(
            agent_name=config.agent_type.upper(),
            total_timesteps=total_timesteps,
            deck=config.player_deck,
            opponent=opp_deck,
        )

    # Allow explicit override (used by curriculum Phase 2)
    if auto_combat is None:
        auto_combat = config.auto_combat
    if auto_target is None:
        auto_target = config.auto_target

    # Build all opponent envs upfront
    opp_envs: list[tp.Any] = []
    for i, opp_deck in enumerate(opponent_decks):
        opp_envs.append(
            create_vec_env(
                config.player_deck,
                opp_deck,
                training_reward_type,
                config.seed + i,
                max_turns=config.max_turns,
                max_steps_per_episode=train_max_steps,
                n_envs=n_envs,
                auto_combat=auto_combat,
                auto_target=auto_target,
                cgfa_factor_spec=cgfa_factor_spec,
                cgfa_scm=cgfa_scm,
                cgfa_calibration_mode=str(
                    getattr(config, "agent_kwargs", {}).get("calibration_mode", "factual")
                ),
            )
        )

    # Proxy VecEnv that delegates to whichever opponent env is active.
    # collect_rollouts captures `env` as a local variable, so swapping
    # self.env on the model doesn't affect the current rollout.  By
    # keeping one stable proxy object and swapping its delegate, the
    # local reference inside collect_rollouts stays valid.
    from stable_baselines3.common.vec_env import VecEnv as _VecEnv

    class _ProxyVecEnv(_VecEnv):
        """Thin proxy that delegates to whichever env is currently active."""

        def __init__(self, initial_env: tp.Any) -> None:
            self._delegate = initial_env
            super().__init__(
                initial_env.num_envs,
                initial_env.observation_space,
                initial_env.action_space,
            )

        def set_delegate(self, env: tp.Any) -> None:
            """Swap the underlying environment."""
            self._delegate = env

        def reset(self) -> np.ndarray:
            return self._delegate.reset()

        def step_async(self, actions: np.ndarray) -> None:
            self._delegate.step_async(actions)

        def step_wait(self) -> tuple:
            return self._delegate.step_wait()

        def close(self) -> None:
            pass  # lifetime managed by opp_envs list

        def env_method(self, method_name: str, *args: tp.Any, **kwargs: tp.Any) -> list:
            return self._delegate.env_method(method_name, *args, **kwargs)

        def get_attr(self, attr_name: str, indices: tp.Any = None) -> list:
            return self._delegate.get_attr(attr_name, indices)

        def set_attr(self, attr_name: str, value: tp.Any, indices: tp.Any = None) -> None:
            self._delegate.set_attr(attr_name, value, indices)

        def seed(self, seed: int | None = None) -> list:
            return self._delegate.seed(seed)

        def env_is_wrapped(self, wrapper_class: type, indices: tp.Any = None) -> list[bool]:
            return self._delegate.env_is_wrapped(wrapper_class, indices)

    proxy_env = _ProxyVecEnv(opp_envs[0])

    # Custom callback that rotates opponents and feeds per-opponent displays
    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError:
        from sb3_contrib.common.callbacks import BaseCallback  # type: ignore[no-redef]

    # Import CWM helpers once for round-robin too
    _rr_causal_transition_cls = None
    _rr_causal_vars_to_array = None
    if _causal_agent is not None and hasattr(_causal_agent, "causal_world_model"):
        from mtg.causal.causal_world_model import CausalTransition, causal_vars_to_array

        _rr_causal_transition_cls = CausalTransition
        _rr_causal_vars_to_array = causal_vars_to_array

    class RoundRobinCallback(BaseCallback):
        """Rotate opponent every N episodes via the proxy env.

        Swaps are **deferred to rollout boundaries** so each PPO rollout
        buffer contains transitions from a single opponent, keeping GAE
        advantage estimates unbiased.  When the episode count triggers a
        rotation, a ``_pending_swap`` flag is set; the actual delegate
        swap happens in ``_on_rollout_end``.
        """

        def __init__(self) -> None:
            super().__init__(verbose=0)
            self._opp_idx = 0
            self._episode_count = 0
            self._rotation_count = 0
            self._pending_swap = False
            self._current_rewards: dict[int, float] = {}
            self._current_lengths: dict[int, int] = {}
            self._start_time = time.time()
            self._last_update = 0
            self._last_loss: float = 0.0
            self._per_opp_wins: dict[str, list[float]] = {d: [] for d in opponent_decks}
            self._per_opp_rewards: dict[str, list[float]] = {d: [] for d in opponent_decks}
            self._per_opp_lengths: dict[str, list[int]] = {d: [] for d in opponent_decks}
            self._episode_rewards: list[float] = []
            self._episode_wins: list[float] = []
            self._episode_lengths: list[int] = []
            self._prev_causal_vars: dict[int, np.ndarray] = {}

        @property
        def _current_opp(self) -> str:
            return opponent_decks[self._opp_idx]

        def _swap_env(self) -> None:
            """Swap to the next opponent via the proxy.

            Called at rollout boundaries (``_on_rollout_end``) so each
            rollout buffer stays opponent-homogeneous.
            """
            self._opp_idx = (self._opp_idx + 1) % len(opponent_decks)
            self._rotation_count += 1
            proxy_env.set_delegate(opp_envs[self._opp_idx])
            cycle = self._rotation_count // len(opponent_decks) + 1
            live_display.opponent = (
                f"{self._current_opp}  (rotation {self._rotation_count}, cycle {cycle})"
            )

        def _on_step(self) -> bool:
            _anneal_entropy(self.model, self.num_timesteps)

            dones = self.locals.get("dones", np.array([False]))
            infos = self.locals.get("infos", [{}])
            rewards = self.locals.get("rewards", np.array([0.0]))
            obs_tensor = self.locals.get("obs_tensor")
            actions = self.locals.get("actions")

            for i in range(len(dones)):
                r = float(rewards[i]) if i < len(rewards) else 0.0
                self._current_rewards[i] = self._current_rewards.get(i, 0.0) + r
                self._current_lengths[i] = self._current_lengths.get(i, 0) + 1

                # Record CWM transitions (same logic as DisplayCallback)
                if _rr_causal_transition_cls is not None:
                    info = infos[i] if i < len(infos) else {}
                    cv = info.get("causal_variables", {})
                    if cv and i in self._prev_causal_vars:
                        prev_cv = self._prev_causal_vars[i]
                        curr_cv = _rr_causal_vars_to_array(cv)
                        obs_np = (
                            obs_tensor[i].cpu().numpy()
                            if obs_tensor is not None and i < len(obs_tensor)
                            else np.zeros(_causal_agent.observation_dim)
                        )
                        act = int(actions[i]) if actions is not None and i < len(actions) else 0
                        is_terminal = bool(dones[i])
                        outcome = 0.5
                        if is_terminal:
                            game_res = info.get("game_result", "")
                            if game_res == "win":
                                outcome = 1.0
                            elif game_res == "loss":
                                outcome = 0.0
                        _causal_agent.causal_world_model.record_transition(
                            _rr_causal_transition_cls(
                                obs=obs_np,
                                action=act,
                                causal_vars=prev_cv,
                                next_causal_vars=curr_cv,
                                terminal=is_terminal,
                                outcome=outcome,
                            )
                        )
                    if cv:
                        self._prev_causal_vars[i] = _rr_causal_vars_to_array(cv)

                if bool(dones[i]):
                    ep_reward = self._current_rewards.pop(i, 0.0)
                    ep_length = self._current_lengths.pop(i, 0)

                    info = infos[i] if i < len(infos) else {}
                    terminal_info = info.get("terminal_info", info)
                    game_result = terminal_info.get("game_result", "")
                    win = 1.0 if game_result == "win" else 0.0

                    if _causal_agent is not None and game_result in ("win", "loss"):
                        cv = terminal_info.get("causal_variables", {})
                        if cv:
                            mapped = _causal_agent._to_scm_causal_vars(cv, terminal_info)
                            _causal_agent.record_game_outcome(mapped, win)

                    opp = self._current_opp
                    self._per_opp_wins[opp].append(win)
                    self._per_opp_rewards[opp].append(ep_reward)
                    self._per_opp_lengths[opp].append(ep_length)

                    self._episode_rewards.append(ep_reward)
                    self._episode_wins.append(win)
                    self._episode_lengths.append(ep_length)
                    self._episode_count += 1

                    # Reset prev causal vars on episode end
                    if i in self._prev_causal_vars:
                        del self._prev_causal_vars[i]

                    if self._episode_count % rotate_every == 0 and len(opponent_decks) > 1:
                        self._pending_swap = True

            # Extract loss
            loss = 0.0
            if hasattr(self, "model") and self.model is not None:
                logger = getattr(self.model, "logger", None)
                if logger is not None:
                    name_to_value = getattr(logger, "name_to_value", {})
                    loss = name_to_value.get("train/loss", 0.0)
                    if loss == 0.0:
                        pg_loss = abs(name_to_value.get("train/policy_gradient_loss", 0.0))
                        v_loss = name_to_value.get("train/value_loss", 0.0)
                        ent_loss = abs(name_to_value.get("train/entropy_loss", 0.0))
                        loss = pg_loss + v_loss + ent_loss
            if loss != 0.0:
                self._last_loss = loss

            # Update per-opponent shadow displays (for plotting) and the
            # live display (shown in the terminal) with the current opponent.
            if self.num_timesteps - self._last_update >= 32:
                self._last_update = self.num_timesteps
                elapsed = time.time() - self._start_time
                fps = self.num_timesteps / max(elapsed, 1e-6)

                for opp_deck in opponent_decks:
                    disp = all_displays[opp_deck]
                    wins = self._per_opp_wins[opp_deck]
                    rews = self._per_opp_rewards[opp_deck]
                    lens = self._per_opp_lengths[opp_deck]
                    recent = min(50, len(wins))
                    if recent > 0:
                        disp.update(
                            timesteps=self.num_timesteps,
                            episodes=len(wins),
                            win_rate=float(np.mean(wins[-recent:])),
                            avg_reward=float(np.mean(rews[-recent:])),
                            episode_length=float(np.mean(lens[-recent:])),
                            loss=self._last_loss if self._last_loss != 0.0 else None,
                            fps=fps,
                        )

                # Push the *current* opponent's stats to the live display
                opp = self._current_opp
                wins = self._per_opp_wins[opp]
                rews = self._per_opp_rewards[opp]
                lens = self._per_opp_lengths[opp]
                recent = min(50, len(wins))
                if recent > 0:
                    # NOTE: loss shown is the global PPO update loss,
                    # not per-opponent.  Rotation rate depends on
                    # n_envs × mean episode length (by design).
                    live_display.update(
                        timesteps=self.num_timesteps,
                        episodes=len(wins),
                        win_rate=float(np.mean(wins[-recent:])),
                        avg_reward=float(np.mean(rews[-recent:])),
                        episode_length=float(np.mean(lens[-recent:])),
                        loss=self._last_loss if self._last_loss != 0.0 else None,
                        fps=fps,
                    )
                else:
                    live_display.update(timesteps=self.num_timesteps, fps=fps)

                if checkpoint_fn is not None:
                    checkpoint_fn(self.num_timesteps)

            return True

        def _on_rollout_end(self) -> None:
            """Apply deferred opponent swap and train CWM.

            Resets ``_last_obs`` so the next rollout bootstraps from the
            new opponent's environment, not stale observations.
            """
            # Train CWM after each rollout (same as DisplayCallback)
            if _causal_agent is not None and hasattr(_causal_agent, "train_causal_world_model"):
                _causal_agent.train_causal_world_model()

            if self._pending_swap:
                self._swap_env()
                self._pending_swap = False
                if self.model is not None:
                    self.model._last_obs = proxy_env.reset()
                    self.model._last_episode_starts = np.ones((proxy_env.num_envs,), dtype=bool)

    # Estimate the maximum number of opponent swaps this budget can support.
    # A swap is gated by TWO things:
    #   1. PPO rollout boundary  -> ceiling = total / (n_steps * n_envs) - 1
    #   2. Episodes-since-last-swap >= rotate_every  -> ceiling = total_eps / rotate_every - 1
    # The realistic limit is min(boundary_cap, episode_cap). Average MTG
    # episode length with auto-resolve is ~70 steps (RR_AVG_EP_LEN), so
    # we use that as the heuristic for episode_cap. Far from perfect but
    # catches the most common "I set --timesteps too low" pitfall.
    _model = getattr(agent, "model", None) or getattr(
        getattr(agent, "base_agent", None), "model", None
    )
    _n_steps = int(getattr(_model, "n_steps", 2048)) if _model is not None else 2048
    _avg_ep_len = RR_AVG_EP_LEN
    _rollout_steps = _n_steps * max(n_envs, 1)
    _rollout_eps = max(1, _rollout_steps // _avg_ep_len)
    # Effective cadence: a swap fires no sooner than `rotate_every`
    # episodes AND no sooner than the next rollout boundary.  When
    # `_rollout_eps > rotate_every`, the rollout boundary dominates and
    # the agent over-fits to one opponent for whole rollouts at a time.
    _effective_cadence_eps = max(rotate_every, _rollout_eps)
    _boundary_swaps = max(0, total_timesteps // _rollout_steps - 1)
    _episode_swaps = max(0, (total_timesteps // _avg_ep_len) // rotate_every - 1)
    _max_swaps = min(_boundary_swaps, _episode_swaps)
    _opps_reachable = min(len(opponent_decks), _max_swaps + 1)
    _safe_budget = (rotate_every * _avg_ep_len) * len(opponent_decks)

    print_divider(f"Round-Robin Training ({len(opponent_decks)} opponents)")
    console.print(
        f"[dim]Rotating opponent every ~{_effective_cadence_eps} episodes "
        f"(target {rotate_every}; PPO rollout = {_rollout_steps:,} steps "
        f"≈ {_rollout_eps} episodes). "
        f"Total budget: {total_timesteps:,} steps "
        f"≈ {_max_swaps + 1} swaps.[/dim]"
    )
    if _rollout_eps > rotate_every * 2:
        # Rollout is so large that swaps are heavily bottlenecked by the
        # PPO boundary, not the cadence target.  This usually means the
        # agent was built without ``is_round_robin=True`` (or n_envs is
        # tiny so even a small n_steps still produces a big rollout).
        console.print(
            f"[bold yellow]\u26a0 Rotation cadence is rollout-bound: "
            f"each rollout ≈ {_rollout_eps} episodes >> target {rotate_every}.[/]\n"
            f"[dim]Agent will over-fit to one opponent for ~{_rollout_eps} "
            f"episodes between swaps.  Build the agent via "
            f"create_agent(..., is_round_robin=True) to auto-tune n_steps.[/]"
        )
    if _opps_reachable < len(opponent_decks):
        console.print(
            f"[bold yellow]\u26a0 Budget too small for a full round-robin cycle.[/]\n"
            f"[dim]Estimate: ~{_max_swaps} swaps reachable (rollout cap "
            f"{_boundary_swaps}, episode cap {_episode_swaps}); only "
            f"{_opps_reachable}/{len(opponent_decks)} opponents will see any "
            f"training and the rest will report 0% in the post-training "
            f"quick-eval. Bump --timesteps to >= {_safe_budget:,} for a single "
            f"full cycle, or 5-10x that for meaningful round-robin learning.[/]"
        )
    console.print()

    # Point the agent's model at the proxy so collect_rollouts
    # uses it (and the callback can swap the delegate mid-rollout).
    model = getattr(agent, "model", None) or getattr(
        getattr(agent, "base_agent", None), "model", None
    )
    if model is not None:
        model.set_env(proxy_env, force_reset=False)
        model._last_obs = proxy_env.reset()
        model._last_episode_starts = np.ones((proxy_env.num_envs,), dtype=bool)

    live_display.start()
    last_display = live_display

    rr_callback = RoundRobinCallback()
    train_callback = (
        _maybe_attach_cgfa_callback(config, rr_callback, output_dir)
        if output_dir is not None
        else rr_callback
    )
    try:
        agent.train(
            total_timesteps=total_timesteps,
            callback=train_callback,
            progress_bar=False,
            reset_num_timesteps=reset_timesteps,
        )
    except Exception as e:
        live_display.stop()
        console.print(f"[bold red]Training error: {e}[/bold red]")
        raise

    live_display.stop()

    # Clean up subprocess envs
    import contextlib

    for env in opp_envs:
        with contextlib.suppress(Exception):
            env.close()

    # Summary
    n_ep = rr_callback._episode_count
    wr = float(np.mean(rr_callback._episode_wins[-50:])) if n_ep > 0 else 0.0
    console.print(
        f"  ✓ Completed: {n_ep} episodes total, "
        f"overall win rate: [{'green' if wr > 0.5 else 'red'}]{wr:.1%}[/]"
    )
    for opp_deck in opponent_decks:
        opp_wins = rr_callback._per_opp_wins[opp_deck]
        opp_wr = float(np.mean(opp_wins[-50:])) if opp_wins else 0.0
        console.print(
            f"    vs {opp_deck}: {len(opp_wins)} episodes, "
            f"win rate: [{'green' if opp_wr > 0.5 else 'red'}]{opp_wr:.1%}[/]"
        )

    # Quick eval per opponent (deterministic; final stats come from
    # evaluate_agent later with config.eval_episodes)
    quick_n = int(getattr(config, "quick_eval_episodes", 20))
    all_metrics: list[dict[str, tp.Any]] = []
    for opp_idx, opp_deck in enumerate(opponent_decks):
        console.print(f"\n[dim]Quick evaluation vs {opp_deck} ({quick_n} episodes)...[/dim]")
        eval_env = create_env(
            config.player_deck,
            opp_deck,
            "sparse",
            config.seed + 1000 + opp_idx,
            max_turns=config.max_turns,
            max_steps_per_episode=train_max_steps,
            auto_combat=auto_combat,
            auto_target=auto_target,
        )
        eval_result = evaluate_agent(
            eval_env,
            agent,
            quick_n,
            max_steps_per_episode=train_max_steps,
            base_seed=config.seed + 1000 + opp_idx,
        )
        all_metrics.append(
            {
                "opponent": opp_deck,
                "win_rate": eval_result["win_rate"],
                "avg_reward": eval_result["avg_reward"],
            }
        )
        console.print(
            f"  Win rate vs {opp_deck}: "
            f"[{'green' if eval_result['win_rate'] > 0.5 else 'red'}]"
            f"{eval_result['win_rate']:.1%}[/]"
        )

    return all_displays, last_display, all_metrics


def train_agent(
    config: TrainingConfig,
) -> tuple[tp.Any, dict[str, tp.Any]]:
    """Execute training workflow.

    Supports multi-opponent training: if multiple opponent decks are specified,
    the agent trains against each opponent for the full timestep budget.

    Args:
        config: Training configuration.

    Returns:
        Tuple of (trained_agent, metrics_dict).

    """
    run_name = config.get_run_name()
    output_dir = create_output_directory(config.output_dir, run_name)

    opponent_decks = config.opponent_decks
    is_multi = config.is_multi_opponent
    full_steps_per_opponent = config.timesteps
    n_envs = getattr(config, "n_envs", 1)
    training_mode = getattr(config, "training_mode", "round-robin")
    agency_mode = getattr(config, "agency_mode", "auto")
    use_round_robin = is_multi and training_mode == "round-robin"

    # For curriculum mode, Phase 1 gets 70% of the budget (auto),
    # Phase 2 gets 30% (full agency).  Other modes use the full budget.
    if agency_mode == "curriculum":
        phase1_steps_per_opp = int(full_steps_per_opponent * 0.7)
        phase2_steps_per_opp = full_steps_per_opponent - phase1_steps_per_opp
        steps_per_opponent = phase1_steps_per_opp
    else:
        steps_per_opponent = full_steps_per_opponent
        phase2_steps_per_opp = 0

    # --timesteps is always a per-opponent budget.
    # In round-robin the total is scaled up so each opponent gets equal exposure.
    if is_multi:
        n_opps = len(opponent_decks)
        if training_mode == "round-robin":
            total_timesteps = steps_per_opponent * n_opps
            console.print(
                f"[bold green]Multi-opponent training (round-robin): "
                f"{n_opps} opponents × {steps_per_opponent:,} steps each "
                f"= {total_timesteps:,} total steps[/bold green]"
            )
        else:
            total_timesteps = steps_per_opponent
            console.print(
                f"[bold green]Multi-opponent training (sequential): "
                f"{n_opps} opponents × {steps_per_opponent:,} steps each "
                f"= {steps_per_opponent * n_opps:,} total[/bold green]"
            )
        for opp in opponent_decks:
            console.print(f"  • {opp}")
        console.print()
    else:
        total_timesteps = steps_per_opponent

    # Warn about small per-opponent budgets
    if steps_per_opponent < 100_000:
        console.print(
            f"[bold yellow]⚠️  Low training budget ({steps_per_opponent:,} steps per opponent). "
            f"A 10-turn MTG game ≈ 80-120 env steps, so this budget only "
            f"covers ~{steps_per_opponent // 100} episodes.[/bold yellow]"
        )
        console.print(
            "[dim]Recommended: 500K (quick test), 1M (standard), 2M+ (paper-quality)[/dim]\n"
        )

    if agency_mode == "curriculum":
        console.print(
            f"[bold magenta]📚 Curriculum learning: "
            f"Phase 1 ({phase1_steps_per_opp:,} steps, auto) → "
            f"Phase 2 ({phase2_steps_per_opp:,} steps, full agency)[/bold magenta]\n"
        )

    # Resolve agency flags from mode.  For "curriculum" the initial phase
    # uses auto; full agency is enabled in phase 2.
    auto_combat = agency_mode in ("auto", "curriculum")
    auto_target = agency_mode in ("auto", "curriculum")

    # Scale step limit by turn cap.  With full agency (manual combat +
    # targeting) each turn has more decision points, so we allow more steps.
    steps_per_turn = 50 if agency_mode == "full" else 30
    train_max_steps = max(300, config.max_turns * steps_per_turn)

    # Use the configured reward type. Shaped rewards give richer per-step
    # signal (life-delta, board advantage) while sparse gives only game
    # outcome (+1 win / -1 loss / -0.3 draw).  Stalling is prevented by
    # the tight step budget (max_turns * steps_per_turn) and heuristic
    # opponents that play aggressively.
    training_reward_type = config.reward_type

    # CGFA agents need a FactorSpec / SCM that is shared between the env
    # wrapper (which writes per-factor signals to ``info``) and the agent
    # (which constructs per-factor heads sized to ``factor_spec.K``).
    cgfa_factor_spec = None
    cgfa_scm = None
    agent_kwargs = dict(config.agent_kwargs)
    cgfa_calibration_mode = str(agent_kwargs.get("calibration_mode", "factual"))
    if config.agent_type == "cgfa":
        from mtg.agents.reinforcement_learning.cgfa import FactorSpec
        from mtg.causal.scm import StructuralCausalModel

        cgfa_factor_spec = agent_kwargs.get("factor_spec") or FactorSpec()
        cgfa_scm = agent_kwargs.get("scm") or StructuralCausalModel()

    # Create initial env with first opponent
    console.print("[bold green]Creating environment...[/]")
    if n_envs > 1:
        console.print(f"[dim]Using {n_envs} parallel environments (SubprocVecEnv)[/dim]")
    env = create_vec_env(
        config.player_deck,
        opponent_decks[0],
        training_reward_type,
        config.seed,
        max_turns=config.max_turns,
        max_steps_per_episode=train_max_steps,
        n_envs=n_envs,
        auto_combat=auto_combat,
        auto_target=auto_target,
        cgfa_factor_spec=cgfa_factor_spec,
        cgfa_scm=cgfa_scm,
        cgfa_calibration_mode=cgfa_calibration_mode,
    )

    console.print("[bold green]Creating agent...[/]")
    agent = create_agent(
        config.agent_type,
        env,
        config.seed,
        total_timesteps=steps_per_opponent,
        cgfa_factor_spec=cgfa_factor_spec,
        cgfa_scm=cgfa_scm,
        is_round_robin=use_round_robin,
        agent_kwargs=agent_kwargs,
    )
    # Store full training budget for entropy annealing so it spans all
    # phases (curriculum phase 1 + 2) without resetting per learn() call.
    n_opps = len(opponent_decks) if is_multi else 1
    _entropy_budget = full_steps_per_opponent * n_opps
    model = getattr(agent, "model", None) or getattr(
        getattr(agent, "base_agent", None), "model", None
    )
    if model is not None:
        model._entropy_total_budget = _entropy_budget
    console.print()

    # Periodic checkpoints so long runs are recoverable
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_interval = max(50_000, steps_per_opponent // 5)
    _last_checkpoint_step = 0

    def _maybe_checkpoint(timesteps: int) -> None:
        nonlocal _last_checkpoint_step
        if timesteps - _last_checkpoint_step >= checkpoint_interval:
            _last_checkpoint_step = timesteps
            ckpt_path = checkpoint_dir / f"checkpoint_{timesteps}"
            agent.save(ckpt_path)

    # Detect CausalAgent for WinProbLearner feeding
    from mtg.agents.causal.causal_agent import CausalAgent

    _causal_agent_ref: CausalAgent | None = agent if isinstance(agent, CausalAgent) else None

    start_time = time.time()
    all_metrics: list[dict[str, tp.Any]] = []
    last_display: TrainingDisplay | None = None
    all_displays: dict[str, TrainingDisplay] = {}

    if use_round_robin:
        # ── Round-robin: single learn() call, rotate opponents in callback ──
        all_displays, last_display, all_metrics = _train_round_robin(
            agent=agent,
            config=config,
            opponent_decks=opponent_decks,
            total_timesteps=total_timesteps,
            training_reward_type=training_reward_type,
            train_max_steps=train_max_steps,
            n_envs=n_envs,
            checkpoint_fn=_maybe_checkpoint,
            auto_combat=auto_combat,
            auto_target=auto_target,
            output_dir=output_dir,
            cgfa_factor_spec=cgfa_factor_spec,
            cgfa_scm=cgfa_scm,
        )
    else:
        # ── Sequential: full budget per opponent ──
        for opp_idx, opp_deck in enumerate(opponent_decks):
            if is_multi:
                console.print(
                    f"\n[bold cyan]Training vs {opp_deck} "
                    f"({opp_idx + 1}/{len(opponent_decks)})[/bold cyan]"
                )

            if opp_idx > 0:
                import contextlib

                with contextlib.suppress(Exception):
                    env.close()
                env = create_vec_env(
                    config.player_deck,
                    opp_deck,
                    training_reward_type,
                    config.seed + opp_idx,
                    max_turns=config.max_turns,
                    max_steps_per_episode=train_max_steps,
                    n_envs=n_envs,
                    auto_combat=auto_combat,
                    auto_target=auto_target,
                    cgfa_factor_spec=cgfa_factor_spec,
                    cgfa_scm=cgfa_scm,
                    cgfa_calibration_mode=cgfa_calibration_mode,
                )
                if hasattr(agent, "set_env"):
                    agent.set_env(env)
                else:
                    agent.initialize_model(env)

            display = TrainingDisplay(
                agent_name=config.agent_type.upper(),
                total_timesteps=steps_per_opponent,
                deck=config.player_deck,
                opponent=opp_deck,
            )
            last_display = display
            all_displays[opp_deck] = display

            print_divider(f"Training vs {opp_deck}")

            display.start()
            sb3_callback = create_sb3_callback(
                display,
                update_every=32,
                on_step_hook=_maybe_checkpoint,
                causal_agent=_causal_agent_ref,
            )
            train_callback = _maybe_attach_cgfa_callback(config, sb3_callback, output_dir)
            try:
                agent.train(
                    total_timesteps=steps_per_opponent,
                    callback=train_callback,
                    progress_bar=False,
                    reset_num_timesteps=(opp_idx == 0),
                )
            except Exception as e:
                display.stop()
                console.print(f"[bold red]Training error: {e}[/bold red]")
                raise

            display.stop()

            n_ep = len(sb3_callback._episode_rewards)
            wr = float(np.mean(sb3_callback._episode_wins[-50:])) if n_ep > 0 else 0.0
            console.print(
                f"  ✓ Completed: {n_ep} episodes, "
                f"win rate: [{'green' if wr > 0.5 else 'red'}]{wr:.1%}[/]"
            )

            quick_n = int(getattr(config, "quick_eval_episodes", 20))
            console.print(f"\n[dim]Quick evaluation vs {opp_deck} ({quick_n} episodes)...[/dim]")
            eval_env = create_env(
                config.player_deck,
                opp_deck,
                "sparse",
                config.seed + 1000 + opp_idx,
                max_turns=config.max_turns,
                max_steps_per_episode=train_max_steps,
                auto_combat=auto_combat,
                auto_target=auto_target,
            )
            eval_result = evaluate_agent(
                eval_env,
                agent,
                quick_n,
                max_steps_per_episode=train_max_steps,
                base_seed=config.seed + 1000 + opp_idx,
            )
            all_metrics.append(
                {
                    "opponent": opp_deck,
                    "win_rate": eval_result["win_rate"],
                    "avg_reward": eval_result["avg_reward"],
                }
            )
            console.print(
                f"  Win rate vs {opp_deck}: "
                f"[{'green' if eval_result['win_rate'] > 0.5 else 'red'}]"
                f"{eval_result['win_rate']:.1%}[/]"
            )

    # ── Curriculum Phase 2: fine-tune with full agency ──
    if agency_mode == "curriculum" and phase2_steps_per_opp > 0:
        print_divider("Curriculum Phase 2: Full Agency Fine-tuning")
        console.print(
            f"[bold magenta]Switching to full agency (selective combat + manual targeting) "
            f"for {phase2_steps_per_opp:,} steps per opponent[/bold magenta]\n"
        )

        # Increase step limit for full agency episodes
        train_max_steps_p2 = max(300, config.max_turns * 50)
        auto_combat = False
        auto_target = False

        if is_multi:
            n_opps = len(opponent_decks)
            if training_mode == "round-robin":
                p2_total = phase2_steps_per_opp * n_opps
                all_displays_p2, _, p2_metrics = _train_round_robin(
                    agent=agent,
                    config=config,
                    opponent_decks=opponent_decks,
                    total_timesteps=p2_total,
                    training_reward_type=training_reward_type,
                    train_max_steps=train_max_steps_p2,
                    n_envs=n_envs,
                    checkpoint_fn=_maybe_checkpoint,
                    auto_combat=False,
                    auto_target=False,
                    reset_timesteps=False,
                )
                # Merge curriculum Phase 2 displays with a "[P2]" suffix so
                # the Phase 1 metrics remain visible alongside them.
                for k, v in all_displays_p2.items():
                    all_displays[f"{k} [P2]"] = v
                all_metrics.extend(p2_metrics)
            else:
                p2_env: tp.Any = None
                for opp_idx, opp_deck in enumerate(opponent_decks):
                    console.print(
                        f"\n[bold cyan]Phase 2 vs {opp_deck} "
                        f"({opp_idx + 1}/{len(opponent_decks)})[/bold cyan]"
                    )
                    if p2_env is not None:
                        import contextlib

                        with contextlib.suppress(Exception):
                            p2_env.close()
                    p2_env = create_vec_env(
                        config.player_deck,
                        opp_deck,
                        training_reward_type,
                        config.seed + 100 + opp_idx,
                        max_turns=config.max_turns,
                        max_steps_per_episode=train_max_steps_p2,
                        n_envs=n_envs,
                        auto_combat=False,
                        auto_target=False,
                        cgfa_factor_spec=cgfa_factor_spec,
                        cgfa_scm=cgfa_scm,
                        cgfa_calibration_mode=cgfa_calibration_mode,
                    )
                    if hasattr(agent, "set_env"):
                        agent.set_env(p2_env)
                    else:
                        agent.initialize_model(p2_env)

                    display = TrainingDisplay(
                        agent_name=config.agent_type.upper(),
                        total_timesteps=phase2_steps_per_opp,
                        deck=config.player_deck,
                        opponent=f"{opp_deck} [P2]",
                    )
                    display.start()
                    sb3_callback = create_sb3_callback(
                        display,
                        update_every=32,
                        on_step_hook=_maybe_checkpoint,
                        causal_agent=_causal_agent_ref,
                    )
                    train_callback = _maybe_attach_cgfa_callback(config, sb3_callback, output_dir)
                    try:
                        agent.train(
                            total_timesteps=phase2_steps_per_opp,
                            callback=train_callback,
                            progress_bar=False,
                            reset_num_timesteps=False,
                        )
                    except Exception as e:
                        display.stop()
                        console.print(f"[bold red]Phase 2 training error: {e}[/bold red]")
                        raise
                    display.stop()
        else:
            p2_env = create_vec_env(
                config.player_deck,
                opponent_decks[0],
                training_reward_type,
                config.seed + 100,
                max_turns=config.max_turns,
                max_steps_per_episode=train_max_steps_p2,
                n_envs=n_envs,
                auto_combat=False,
                auto_target=False,
                cgfa_factor_spec=cgfa_factor_spec,
                cgfa_scm=cgfa_scm,
                cgfa_calibration_mode=cgfa_calibration_mode,
            )
            if hasattr(agent, "set_env"):
                agent.set_env(p2_env)
            else:
                agent.initialize_model(p2_env)

            display = TrainingDisplay(
                agent_name=config.agent_type.upper(),
                total_timesteps=phase2_steps_per_opp,
                deck=config.player_deck,
                opponent=f"{opponent_decks[0]} [P2]",
            )
            display.start()
            sb3_callback = create_sb3_callback(
                display,
                update_every=32,
                on_step_hook=_maybe_checkpoint,
                causal_agent=_causal_agent_ref,
            )
            train_callback = _maybe_attach_cgfa_callback(config, sb3_callback, output_dir)
            try:
                agent.train(
                    total_timesteps=phase2_steps_per_opp,
                    callback=train_callback,
                    progress_bar=False,
                    reset_num_timesteps=False,
                )
            except Exception as e:
                display.stop()
                console.print(f"[bold red]Phase 2 training error: {e}[/bold red]")
                raise
            display.stop()

        # Update train_max_steps for eval to use full agency step limit
        train_max_steps = train_max_steps_p2
        console.print("\n[bold green]✓ Curriculum Phase 2 complete[/bold green]")

    training_time = time.time() - start_time

    # Save model
    model_name = config.get_model_name()
    model_path = output_dir / f"{model_name}.zip"
    console.print(f"\n[bold green]Saving model to {model_path}[/]")
    agent.save(model_path)

    # Full evaluation uses the final agency settings
    if agency_mode == "curriculum":
        auto_combat = False
        auto_target = False

    # Full evaluation
    print_divider("Final Evaluation")
    console.print(f"[bold cyan]Evaluating over {config.eval_episodes} episodes per opponent...[/]")

    eval_results: dict[str, dict[str, float]] = {}
    for opp_idx, opp_deck in enumerate(opponent_decks):
        eval_env = create_env(
            config.player_deck,
            opp_deck,
            "sparse",
            config.seed + 2000 + opp_idx,
            max_turns=config.max_turns,
            max_steps_per_episode=train_max_steps,
            auto_combat=auto_combat,
            auto_target=auto_target,
        )
        episodes_per_opp = max(1, config.eval_episodes)
        console.print(f"\n[cyan]vs {opp_deck} ({episodes_per_opp} episodes):[/cyan]")
        result = evaluate_agent(
            eval_env,
            agent,
            episodes_per_opp,
            max_steps_per_episode=train_max_steps,
            base_seed=config.seed + 2000 + opp_idx,
        )
        eval_results[opp_deck] = result

    # Compile metrics
    n_opps_final = len(opponent_decks) if is_multi else 1
    mode_final = getattr(config, "training_mode", "sequential")
    total_steps_final = config.timesteps * n_opps_final

    # Persist per-opponent step-by-step history so plots can be regenerated later.
    # The TrainingDisplay.metrics.history contains one dict per PPO update with
    # keys like "win_rate", "avg_reward", "episode_length", "loss".
    per_opponent_history: dict[str, list[dict[str, tp.Any]]] = {}
    for opp_name, disp in all_displays.items():
        try:
            per_opponent_history[opp_name] = list(disp.metrics.history)
        except (AttributeError, TypeError):
            per_opponent_history[opp_name] = []

    metrics: dict[str, tp.Any] = {
        "training": {
            "timesteps_per_opponent": config.timesteps,
            "total_timesteps": total_steps_final,
            "training_mode": mode_final,
            "training_time_seconds": training_time,
            "training_time_formatted": format_duration(training_time),
            "opponents": opponent_decks,
            "per_opponent": all_metrics,
            "per_opponent_history": per_opponent_history,
        },
        "evaluation": eval_results,
    }

    # Save artifacts
    save_training_artifacts(output_dir, config, metrics, training_time)

    # Generate training plots
    print_divider("Generating Plots")
    if last_display is not None or all_displays:
        plots = generate_training_plots(
            output_dir,
            last_display,
            eval_results,
            config,
            all_displays=all_displays if is_multi else None,
        )
        if plots:
            for p in plots:
                console.print(f"  📊 Saved: {p}")
        else:
            console.print("  [dim]No plots generated.[/dim]")

    # Generate sample game reports (HTML replays)
    print_divider("Generating Sample Reports")
    sample_opps = config.sample_opponent_decks
    console.print(
        f"[bold cyan]Recording {config.sample_games} sample game(s) "
        f"vs {', '.join(sample_opps)}...[/]"
    )
    report_paths = generate_sample_reports(
        agent,
        config,
        output_dir,
        max_steps=500,
        auto_combat=auto_combat,
        auto_target=auto_target,
    )
    if report_paths:
        for rp in report_paths:
            console.print(f"  📄 Report: {rp}")
    else:
        console.print("  [dim]No reports generated.[/dim]")

    # Print results table
    print_evaluation_results(
        eval_results,
        title="Final Training Results",
    )

    console.print(f"\n[dim]Total time: {format_duration(training_time)}[/]")
    console.print(f"[dim]Model saved: {model_path}[/]")
    console.print(f"[dim]Artifacts: {output_dir}/[/]")
    console.print(
        f"[dim]To use in gameplay: select [bold]{config.agent_type}[/bold] agent "
        f"with [bold]{config.player_deck}[/bold] deck[/dim]"
    )

    return agent, metrics


def main_interactive() -> int:
    """Run training in interactive mode.

    Returns:
        Exit code (0 for success).

    """
    console.clear()
    print_logo()

    console.print("\n[bold cyan]Interactive Training Mode[/]")
    console.print("[dim]Answer the prompts to configure training.[/dim]\n")

    # Get configuration interactively
    config = prompt_training_config()

    # Confirm configuration
    if not confirm_config(config, "Training"):
        console.print("[yellow]Training cancelled.[/yellow]")
        return 1

    # Run training
    train_agent(config)

    return 0


def main_cli(args: argparse.Namespace) -> int:
    """Run training with command-line arguments.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code (0 for success).

    """
    console.clear()
    print_logo()

    # Convert args to config
    config = args_to_config(args)

    print_divider("Configuration")
    console.print(f"[bold cyan]Agent:[/] {config.agent_type}")
    console.print(f"[bold cyan]Timesteps:[/] {config.timesteps:,} per opponent")
    console.print(f"[bold cyan]Agent Deck:[/] {config.player_deck}")
    if config.is_multi_opponent:
        n_opps = len(config.opponent_decks)
        mode = getattr(config, "training_mode", "round-robin")
        total = config.timesteps * n_opps if mode == "round-robin" else config.timesteps * n_opps
        console.print(
            f"[bold cyan]Opponents:[/] {n_opps} archetypes (multi-opponent) → {total:,} total steps"
        )
        console.print(f"[bold cyan]Training Mode:[/] {mode}")
    else:
        console.print(f"[bold cyan]Opponent:[/] {config.opponent_deck}")
    console.print(f"[bold cyan]Reward:[/] {config.reward_type}")
    console.print(f"[bold cyan]Max Turns:[/] {config.max_turns}")
    console.print(f"[bold cyan]Parallel Envs:[/] {config.n_envs}")
    agency_labels = {
        "auto": "Auto (simplified combat and targeting; fastest learning)",
        "full": "Full (agent selects attackers and spell targets; needs 3M+ steps)",
        "curriculum": "Curriculum (auto 70% then full 30%; designed for Causal RL agent)",
    }
    console.print(
        f"[bold cyan]Agency:[/] {agency_labels.get(config.agency_mode, config.agency_mode)}"
    )
    console.print(f"[bold cyan]Seed:[/] {config.seed}")
    console.print()

    # Run training
    train_agent(config)

    return 0


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success).

    """
    args = parse_args()

    if args.interactive:
        return main_interactive()
    else:
        return main_cli(args)


if __name__ == "__main__":
    sys.exit(main())

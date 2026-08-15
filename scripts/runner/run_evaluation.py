#!/usr/bin/env python3
"""Evaluation script for MTG-Causal-RL agents.

This script provides both interactive and command-line modes for evaluating
trained or baseline agents on the benchmark, producing metrics and reports.

Usage:
    # Interactive mode (prompts for all settings)
    uv run python scripts/runner/run_evaluation.py --interactive

    # Command-line mode (specify settings)
    uv run python scripts/runner/run_evaluation.py --agent ppo \
        --model-path results/trained_agents/ppo_model.zip

    # Evaluate all baseline agents
    uv run python scripts/runner/run_evaluation.py --agent all

    # Demo mode (sample results visualization)
    uv run python scripts/runner/run_evaluation.py --demo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import typing as tp
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from rich import box
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from mtg.utils.cli_display import (
    console,
    print_divider,
    print_evaluation_results,
    print_game_state,
    print_logo,
    print_mulligan_state,
    print_play_draw_selection,
)
from mtg.utils.html_report import (
    GameRecorder,
    generate_html_report,
    save_replay_json,
)
from mtg.utils.interactive import (
    EvaluationConfig,
    confirm_config,
    create_output_directory,
    discover_trained_models,
    format_duration,
    get_available_archetypes,
    print_section_header,
    prompt_evaluation_config,
)

# Speed presets for visualization (seconds)
SPEED_PRESETS = {
    "slow": {"phase": 5.0, "action": 5.0},
    "medium": {"phase": 3.0, "action": 3.0},
    "fast": {"phase": 1.0, "action": 1.0},
}
DELAYS = SPEED_PRESETS["fast"]


def generate_evaluation_plots(
    output_dir: Path,
    results: dict[str, dict[str, tp.Any]],
    config: EvaluationConfig,
) -> list[Path]:
    """Generate evaluation plots and save them under ``output_dir / plots``.

    Creates two figures:
    - ``win_rate_comparison.png``: win-rate bars with error bars.
    - ``reward_comparison.png``: average reward bars with error bars.
    """
    if not results:
        return []

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        console.print("[yellow]matplotlib not installed, skipping evaluation plots[/yellow]")
        return []

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # Stable ordering for reproducible plot output
    ordered_items = list(results.items())
    labels = []
    win_rates = []
    win_stds = []
    rewards = []
    reward_stds = []

    for _key, row in ordered_items:
        agent = str(row.get("agent", "agent")).upper()
        opp = str(row.get("opponent", "opponent")).replace("_", " ").title()
        if config.agent_type == "all":
            label = f"{agent} vs {opp}" if config.is_multi_opponent else agent
        else:
            label = opp if config.is_multi_opponent else agent
        labels.append(label)
        win_rates.append(float(row.get("win_rate", 0.0)))
        win_stds.append(float(row.get("win_rate_std", 0.0)))
        rewards.append(float(row.get("avg_reward", 0.0)))
        reward_stds.append(float(row.get("reward_std", 0.0)))

    x = np.arange(len(labels))
    width = max(8.0, min(20.0, 0.9 * len(labels) + 6.0))

    # Plot 1: Win rate comparison
    fig, ax = plt.subplots(figsize=(width, 5.5))
    colors = ["#2A9D8F" if wr >= 0.5 else "#E63946" for wr in win_rates]
    bars = ax.bar(
        x,
        win_rates,
        yerr=win_stds,
        capsize=4,
        color=colors,
        edgecolor="white",
        alpha=0.9,
    )
    for bar, wr in zip(bars, win_rates, strict=False):
        ax.annotate(
            f"{wr:.0%}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50%")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    ax.set_title("Evaluation Win Rate Comparison", fontweight="bold")
    plt.tight_layout()
    win_path = plots_dir / "win_rate_comparison.png"
    fig.savefig(win_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(win_path)

    # Plot 2: Reward comparison
    fig, ax = plt.subplots(figsize=(width, 5.5))
    reward_colors = ["#457B9D" if rw >= 0 else "#E76F51" for rw in rewards]
    bars = ax.bar(
        x,
        rewards,
        yerr=reward_stds,
        capsize=4,
        color=reward_colors,
        edgecolor="white",
        alpha=0.9,
    )
    for bar, rw in zip(bars, rewards, strict=False):
        y = rw + (0.03 if rw >= 0 else -0.05)
        va = "bottom" if rw >= 0 else "top"
        ax.annotate(
            f"{rw:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, y),
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Average Reward")
    ax.axhline(y=0.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_title("Evaluation Reward Comparison", fontweight="bold")
    plt.tight_layout()
    reward_path = plots_dir / "reward_comparison.png"
    fig.savefig(reward_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(reward_path)

    return saved


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.

    """
    parser = argparse.ArgumentParser(
        description="Evaluate agents on MTG-Causal-RL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode selection
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode with prompts",
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run evaluation demo with sample results (no actual evaluation)",
    )

    # Agent configuration
    parser.add_argument(
        "--agent",
        type=str,
        default="all",
        help="Agent type to evaluate ('all' for all registered agents)",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model (for ppo/causal)",
    )

    # Deck configuration
    parser.add_argument(
        "--deck",
        type=str,
        default="mono_red_aggro",
        choices=get_available_archetypes(),
        help="Player deck archetype",
    )

    parser.add_argument(
        "--opponent",
        type=str,
        default="azorius_control",
        help="Opponent deck(s): 'all' for all archetypes, or comma-separated names",
    )

    # Evaluation parameters
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help=(
            "Total evaluation episodes per opponent "
            "(distributed across seeds, not split across opponents)"
        ),
    )

    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Max MTG turns per game (should match training setting)",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789, 1000],
        help="Random seeds for evaluation",
    )

    # Output configuration
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluations",
        help="Output directory for results",
    )

    # Visualization options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show live game state visualization during evaluation",
    )

    parser.add_argument(
        "--show-games",
        type=int,
        default=0,
        help="Number of games to visualize in detail (default: 0)",
    )
    parser.add_argument(
        "--show-games-opponents",
        type=str,
        default="",
        help=(
            "Opponents for sample games/reports (comma-separated, empty = all evaluated opponents)"
        ),
    )

    parser.add_argument(
        "--save-reports",
        action="store_true",
        help="Save HTML gameplay reports for visualized games",
    )

    parser.add_argument(
        "--speed",
        type=str,
        choices=["slow", "medium", "fast"],
        default="fast",
        help="Visualization speed: slow (5s), medium (3s), fast (1s)",
    )

    return parser.parse_args()


def args_to_config(args: argparse.Namespace) -> EvaluationConfig:
    """Convert command-line args to EvaluationConfig.

    Args:
        args: Parsed arguments.

    Returns:
        EvaluationConfig instance.

    """
    # Handle 'all' opponent
    if args.opponent == "all":
        opponent_deck = ",".join(get_available_archetypes())
    else:
        opponent_deck = args.opponent

    return EvaluationConfig(
        agent_type=args.agent,
        model_path=args.model_path,
        player_deck=args.deck,
        opponent_deck=opponent_deck,
        episodes=args.episodes,
        max_turns=args.max_turns,
        seeds=args.seeds,
        save_reports=args.save_reports,
        show_games=args.show_games,
        show_games_opponents=args.show_games_opponents,
        output_dir=args.output,
        verbose=args.verbose,
    )


def create_env(
    player_deck: str,
    opponent_deck: str,
    seed: int,
    max_turns: int = 10,
    use_heuristic_opponent: bool = True,
) -> tp.Any:
    """Create evaluation environment.

    Args:
        player_deck: Player's deck archetype.
        opponent_deck: Opponent's deck archetype.
        seed: Random seed.
        max_turns: Maximum MTG turns per game.
        use_heuristic_opponent: If True, use deck-matched heuristic opponent agent.

    Returns:
        Configured MTG environment.

    """
    from mtg.agents import get_agent, heuristic_for_deck
    from mtg.env import MTGEnv

    normalized = opponent_deck.lower().replace(" ", "_").replace("-", "_")
    opponent_agent_name = heuristic_for_deck(normalized) or "greedy_aggro"
    opponent_agent = get_agent(opponent_agent_name, seed=seed) if use_heuristic_opponent else None

    return MTGEnv(
        deck_archetype=player_deck,
        opponent_archetype=opponent_deck,
        max_turns=max_turns,
        max_steps_per_episode=500,
        reward_type="sparse",
        seed=seed,
        auto_resolve=True,
        opponent_agent=opponent_agent,
    )


def create_agent(
    agent_type: str,
    obs_dim: int,
    act_dim: int,
    model_path: str | None,
    seed: int,
) -> tp.Any:
    """Create an agent for evaluation.

    Args:
        agent_type: Type of agent.
        obs_dim: Observation dimension.
        act_dim: Action dimension.
        model_path: Optional path to trained model.
        seed: Random seed.

    Returns:
        Agent instance.

    """
    from mtg.agents import get_agent, list_agents

    try:
        if agent_type in ["ppo", "causal"]:
            agent = get_agent(
                agent_type,
                observation_dim=obs_dim,
                action_dim=act_dim,
                seed=seed,
            )
            if model_path:
                agent.load(model_path)
        elif agent_type == "greedy_aggro":
            agent = get_agent(agent_type, aggression=0.7, seed=seed)
        else:
            agent = get_agent(agent_type, seed=seed)
    except KeyError as err:
        available = list_agents()
        raise ValueError(f"Unknown agent '{agent_type}'. Available: {available}") from err

    return agent


def _build_eval_obs_normaliser(
    vec_normalize_path: str | None,
    env: tp.Any,
) -> tp.Callable[[np.ndarray], np.ndarray] | None:
    """Load ``vec_normalize.pkl`` and return a per-obs normaliser.

    Returns ``None`` if the file does not exist or the path is None,
    so callers can pass the result directly without wrapping in
    additional null-checks.  The normaliser applies the same
    ``(obs - mean) / sqrt(var + eps)`` -> clip -> cast pipeline used
    by SB3's ``VecNormalize.normalize_obs``.
    """
    if not vec_normalize_path:
        return None
    from pathlib import Path

    path = Path(vec_normalize_path)
    if not path.exists():
        return None
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from mtg.training.env_factory import (
        make_obs_normaliser_from_vec_normalize,
    )

    scratch = DummyVecEnv([lambda: env])
    loaded = VecNormalize.load(str(path), scratch)
    return make_obs_normaliser_from_vec_normalize(loaded)


def evaluate_single(
    env: tp.Any,
    agent: tp.Any,
    n_episodes: int,
    max_steps_per_episode: int = 500,
    obs_normaliser: tp.Callable[[np.ndarray], np.ndarray] | None = None,
    base_seed: int = 0,
) -> dict[str, float]:
    """Evaluate a single agent.

    Args:
        env: Evaluation environment.
        agent: Agent to evaluate.
        n_episodes: Number of episodes.
        max_steps_per_episode: Safety limit to prevent infinite loops.
        obs_normaliser: Optional callable applied to every observation,
            used to plug in the frozen ``VecNormalize`` running stats
            from training.
        base_seed: Per-episode seeds will be ``base_seed + ep`` so each
            (agent, opponent, seed) tuple has its own deterministic
            episode draw. Without this, ``env.reset()`` (no seed)
            could return the same opponent draw for every agent in
            the comparison loop.

    Returns:
        Evaluation metrics.
    """
    wins = 0
    rewards: list[float] = []
    lengths: list[int] = []

    def _maybe_norm(obs: np.ndarray) -> np.ndarray:
        return obs_normaliser(obs) if obs_normaliser is not None else obs

    for ep in range(n_episodes):
        obs, info = env.reset(seed=base_seed + ep)
        obs = _maybe_norm(obs)
        done = False
        ep_reward = 0.0
        ep_len = 0

        while not done and ep_len < max_steps_per_episode:
            action_mask = info.get("action_mask", np.ones(env.action_space.n, dtype=bool))
            action = agent.select_action(obs, action_mask, info)
            obs, reward, terminated, truncated, info = env.step(action)
            obs = _maybe_norm(obs)
            ep_reward += reward
            ep_len += 1
            done = terminated or truncated

        rewards.append(ep_reward)
        lengths.append(ep_len)
        if info.get("game_result") == "win":
            wins += 1

    return {
        "win_rate": wins / n_episodes,
        "win_rate_std": np.sqrt(wins / n_episodes * (1 - wins / n_episodes) / n_episodes),
        "avg_reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "avg_length": float(np.mean(lengths)),
    }


def run_visualized_game(
    env: tp.Any,
    agent: tp.Any,
    agent_name: str,
    game_num: int,
    player_deck: str = "unknown",
    opponent_deck: str = "unknown",
    save_report: bool = False,
    output_dir: Path | None = None,
    obs_normaliser: tp.Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, tp.Any]:
    """Run a single game with visualization.

    Args:
        env: Game environment.
        agent: Agent to play.
        agent_name: Name of the agent.
        game_num: Game number for display.
        player_deck: Player deck archetype name.
        opponent_deck: Opponent deck archetype name.
        save_report: Whether to save an HTML report.
        output_dir: Directory for saving reports.
        obs_normaliser: Optional callable that mirrors the training-time
            observation normalisation (e.g. ``VecNormalize`` mean/var).
            Applied to every observation before feeding it to ``agent``
            so the visualised rollout matches the agent's training
            distribution.  See :func:`_build_eval_obs_normaliser`.

    Returns:
        Game result information.

    """
    console.print(f"\n[bold cyan]Game {game_num} - {agent_name}[/bold cyan]")
    print_divider("")

    from mtg.utils.html_report import (
        actions_from_env,
        snapshot_from_env,
        turn_summary_from_env,
    )

    obs, info = env.reset()
    if obs_normaliser is not None:
        obs = obs_normaliser(obs)
    done = False
    ep_reward = 0.0
    step = 0
    action_history: list[str] = []
    action_log_cursor = 0
    prev_turn = 0

    # Initialize game recorder for HTML report
    recorder = GameRecorder(
        player_deck=player_deck,
        opponent_deck=opponent_deck,
        player_agent=agent_name,
        opponent_agent="Heuristic",
    )

    # Show play/draw selection
    player_on_play = info.get("player_on_play", True)
    recorder.set_player_on_play(player_on_play)
    print_play_draw_selection(player_on_play)
    time.sleep(DELAYS["action"])

    # Show mulligan phase
    hand_size = info.get("hand_size", 7)
    print_mulligan_state(
        hand_size=hand_size,
        mulligan_count=0,
        action="Opening hand",
        player_name="Player",
    )
    time.sleep(DELAYS["phase"])

    # Record initial snapshot
    snap = snapshot_from_env(env)
    if snap:
        recorder.record_snapshot(**snap)

    while not done:
        action_mask = info["action_mask"]
        action = agent.select_action(obs, action_mask, info)

        # Get action name if available
        action_name = info.get("action_names", {}).get(action, f"Action {action}")
        action_history.append(action_name)

        obs, reward, terminated, truncated, info = env.step(action)
        if obs_normaliser is not None:
            obs = obs_normaliser(obs)
        ep_reward += reward
        step += 1
        done = terminated or truncated

        # Record new actions from engine action log
        new_actions = actions_from_env(env, since_idx=action_log_cursor)
        for act_kw in new_actions:
            recorder.record_action(**act_kw)
        if env.state:
            action_log_cursor = len(env.state.action_log)

        # Record rich state snapshot directly from env.state
        snap = snapshot_from_env(env)
        if snap:
            current_turn = snap.get("turn", 0)
            if current_turn > prev_turn and prev_turn > 0:
                ts = turn_summary_from_env(env, prev_turn)
                recorder.record_turn_summary(**ts)
            prev_turn = current_turn
            recorder.record_snapshot(**snap)

        # CLI display (from info dict, lightweight)
        turn = info.get("turn", 1)
        phase = info.get("phase", "Main 1")
        active_player = info.get("active_player", "Player")
        player_life = info.get("player_life", 20)
        opponent_life = info.get("opponent_life", 20)
        hand_size = info.get("hand_size", 0)
        lands = info.get("lands_on_battlefield", 0)
        board_power = info.get("board_power", 0)

        print_game_state(
            turn=turn,
            phase=phase,
            player_life=player_life,
            opponent_life=opponent_life,
            hand_size=hand_size,
            lands=lands,
            board_power=board_power,
            opponent_lands=info.get("opponent_lands", 0),
            opponent_power=info.get("opponent_power", 0),
            opponent_hand_size=info.get("opponent_hand_size", 0),
            last_action=action_name,
            action_history=action_history[-5:],
            active_player=active_player,
            mana_available=info.get("mana_available", 0),
            player_creatures=info.get("player_creatures", []),
            opponent_creatures=info.get("opponent_creatures", []),
            player_graveyard=info.get("player_graveyard", []),
            opponent_graveyard=info.get("opponent_graveyard", []),
        )
        time.sleep(DELAYS["phase"] if "pass" in action_name.lower() else DELAYS["action"])

    # Record final turn summary
    if prev_turn > 0:
        ts = turn_summary_from_env(env, prev_turn)
        recorder.record_turn_summary(**ts)

    # Show result
    result = info.get("game_result", "unknown")
    if result == "win":
        console.print(f"\n[bold green]VICTORY! (+{ep_reward:.2f})[/bold green]")
        recorder.set_winner("Player")
    elif result == "loss":
        console.print(f"\n[bold red]DEFEAT ({ep_reward:.2f})[/bold red]")
        recorder.set_winner("Opponent")
    else:
        console.print(f"\n[bold yellow]DRAW ({ep_reward:.2f})[/bold yellow]")
        recorder.set_winner("Draw")

    # Save HTML report if requested
    if save_report and output_dir:
        replay = recorder.get_replay()
        report_dir = output_dir / "reports" / f"game_{game_num}_{opponent_deck}"
        report_dir.mkdir(parents=True, exist_ok=True)

        html_path = report_dir / "replay.html"
        json_path = report_dir / "replay.json"

        generate_html_report(replay, html_path)
        save_replay_json(replay, json_path)

        console.print(f"\n[dim]Report saved: {report_dir}/[/dim]")

    return {
        "result": result,
        "reward": ep_reward,
        "steps": step,
    }


def run_report_game(
    env: tp.Any,
    agent: tp.Any,
    agent_name: str,
    game_num: int,
    player_deck: str,
    opponent_deck: str,
    output_dir: Path,
    obs_normaliser: tp.Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, tp.Any]:
    """Run one silent game and save HTML/JSON report artifacts."""
    from mtg.utils.html_report import (
        GameRecorder,
        actions_from_env,
        generate_html_report,
        save_replay_json,
        snapshot_from_env,
        turn_summary_from_env,
    )

    obs, info = env.reset()
    if obs_normaliser is not None:
        obs = obs_normaliser(obs)
    done = False
    ep_reward = 0.0
    step = 0
    action_log_cursor = 0
    prev_turn = 0

    recorder = GameRecorder(
        player_deck=player_deck,
        opponent_deck=opponent_deck,
        player_agent=agent_name,
        opponent_agent="Heuristic",
    )
    player_on_play = info.get("player_on_play", True)
    recorder.set_player_on_play(player_on_play)

    snap = snapshot_from_env(env)
    if snap:
        recorder.record_snapshot(**snap)

    while not done:
        action_mask = info.get("action_mask", np.ones(env.action_space.n, dtype=bool))
        action = agent.select_action(obs, action_mask, info)
        obs, reward, terminated, truncated, info = env.step(action)
        if obs_normaliser is not None:
            obs = obs_normaliser(obs)
        ep_reward += reward
        step += 1
        done = terminated or truncated

        new_actions = actions_from_env(env, since_idx=action_log_cursor)
        for act_kw in new_actions:
            recorder.record_action(**act_kw)
        if env.state:
            action_log_cursor = len(env.state.action_log)

        snap = snapshot_from_env(env)
        if snap:
            current_turn = snap.get("turn", 0)
            if current_turn > prev_turn and prev_turn > 0:
                ts = turn_summary_from_env(env, prev_turn)
                recorder.record_turn_summary(**ts)
            prev_turn = current_turn
            recorder.record_snapshot(**snap)

    if prev_turn > 0:
        ts = turn_summary_from_env(env, prev_turn)
        recorder.record_turn_summary(**ts)

    result = info.get("game_result", "unknown")
    if result == "win":
        recorder.set_winner("Player")
    elif result == "loss":
        recorder.set_winner("Opponent")
    else:
        recorder.set_winner("Draw")

    replay = recorder.get_replay()
    report_dir = output_dir / "reports" / f"game_{game_num}_{opponent_deck}"
    report_dir.mkdir(parents=True, exist_ok=True)
    html_path = report_dir / "replay.html"
    json_path = report_dir / "replay.json"
    generate_html_report(replay, html_path)
    save_replay_json(replay, json_path)

    return {
        "result": result,
        "reward": ep_reward,
        "steps": step,
    }


def run_evaluation(
    config: EvaluationConfig,
) -> dict[str, dict[str, tp.Any]]:
    """Run evaluation workflow.

    Supports multi-opponent evaluation. Results are keyed by
    ``{agent_type}`` (single opponent) or ``{agent_type}_vs_{opponent}``
    (multi-opponent).

    Args:
        config: Evaluation configuration.

    Returns:
        Dictionary of results per agent (and per opponent if multi-opp).

    """
    from mtg.agents import list_agents as get_all_agents

    # Determine which agents to evaluate
    agents_to_eval = get_all_agents() if config.agent_type == "all" else [config.agent_type]
    opponent_decks = config.opponent_decks

    # Create output directory
    run_name = config.get_run_name()
    output_dir = create_output_directory(config.output_dir, run_name)

    results: dict[str, dict[str, tp.Any]] = {}
    start_time = time.time()

    print_divider("Running Evaluation")
    if config.is_multi_opponent:
        console.print(
            f"[dim]Evaluating against {len(opponent_decks)} opponents: "
            f"{', '.join(opponent_decks)}[/dim]\n"
        )
        console.print(
            f"[dim]Episode budget: {config.episodes} per opponent "
            f"(distributed across {len(config.seeds)} seed(s))[/dim]\n"
        )
    else:
        console.print(
            f"[dim]Episode budget: {config.episodes} total "
            f"(distributed across {len(config.seeds)} seed(s))[/dim]\n"
        )

    total_units = len(agents_to_eval) * len(opponent_decks) * len(config.seeds)

    # Create progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, complete_style="green"),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        overall_task = progress.add_task("[cyan]Overall Progress", total=total_units)

        for agent_type in agents_to_eval:
            for opp_deck in opponent_decks:
                label = (
                    f"{agent_type.upper()} vs {opp_deck}"
                    if config.is_multi_opponent
                    else agent_type.upper()
                )
                agent_task = progress.add_task(
                    f"[yellow]  {label}",
                    total=len(config.seeds),
                )

                seed_results: list[dict[str, float]] = []

                for seed in config.seeds:
                    env = create_env(
                        config.player_deck,
                        opp_deck,
                        seed,
                        max_turns=config.max_turns,
                    )

                    obs_dim = env.observation_space.shape[0]
                    act_dim = env.action_space.n

                    agent = create_agent(
                        agent_type,
                        obs_dim,
                        act_dim,
                        config.model_path if agent_type == config.agent_type else None,
                        seed,
                    )
                    # Apply frozen VecNormalize stats from training
                    # run. Only meaningful when this agent was actually
                    # trained with normalization, hence the per-agent
                    # filter. The function is a no-op when the path is
                    # missing or invalid.
                    obs_normaliser = _build_eval_obs_normaliser(
                        config.vec_normalize_path if agent_type == config.agent_type else None,
                        env,
                    )
                    # Mirror training semantics: each opponent gets the full episode budget.
                    # We only distribute that budget across seeds.
                    episodes_per_seed = max(1, config.episodes // len(config.seeds))
                    metrics = evaluate_single(
                        env,
                        agent,
                        episodes_per_seed,
                        obs_normaliser=obs_normaliser,
                        base_seed=seed * 1000,
                    )
                    seed_results.append(metrics)

                    progress.advance(agent_task)
                    progress.advance(overall_task)

                avg_win_rate = np.mean([r["win_rate"] for r in seed_results])
                std_win_rate = np.std([r["win_rate"] for r in seed_results])
                avg_reward = np.mean([r["avg_reward"] for r in seed_results])
                std_reward = np.std([r["avg_reward"] for r in seed_results])

                result_key = (
                    f"{agent_type}_vs_{opp_deck}" if config.is_multi_opponent else agent_type
                )
                results[result_key] = {
                    "agent": agent_type,
                    "opponent": opp_deck,
                    "win_rate": float(avg_win_rate),
                    "win_rate_std": float(std_win_rate),
                    "avg_reward": float(avg_reward),
                    "reward_std": float(std_reward),
                    "per_seed": seed_results,
                }

                progress.update(agent_task, description=f"[green]  {label}")

    # Save results
    eval_time = time.time() - start_time

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "config": {
                    "player_deck": config.player_deck,
                    "opponent_decks": opponent_decks,
                    "episodes": config.episodes,
                    "episodes_per_opponent": config.episodes,
                    "episodes_per_seed": max(1, config.episodes // len(config.seeds)),
                    "max_turns": config.max_turns,
                    "seeds": config.seeds,
                    "model_path": config.model_path,
                    "evaluation_time_seconds": eval_time,
                    "timestamp": datetime.now().isoformat(),
                },
                "results": results,
            },
            f,
            indent=2,
        )

    console.print(f"\n[bold green]Results saved to {results_path}[/]")

    # Print results table
    print_divider("Results Summary")
    title = (
        f"Benchmark: {config.player_deck} vs {', '.join(opponent_decks)}"
        if config.is_multi_opponent
        else f"Benchmark: {config.player_deck} vs {opponent_decks[0]}"
    )
    print_evaluation_results(results, title=title)

    console.print(f"[dim]Evaluation time: {format_duration(eval_time)}[/]")

    # Generate evaluation plots
    print_divider("Generating Plots")
    plot_paths = generate_evaluation_plots(output_dir, results, config)
    if plot_paths:
        for p in plot_paths:
            console.print(f"  📊 Saved: {p}")
    else:
        console.print("  [dim]No plots generated.[/dim]")

    # Optional post-eval game generation
    if config.show_games > 0:
        viz_opponents = config.show_games_opponent_decks
        if config.save_reports and not config.verbose:
            print_divider("Generating Sample Reports")
            console.print(
                f"\n[bold]Recording {config.show_games} game(s) per opponent "
                f"({', '.join(viz_opponents)})...[/]\n"
            )
            total_reports = len(agents_to_eval) * len(viz_opponents) * config.show_games
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40, complete_style="green"),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("[cyan]Generating reports", total=total_reports)
                for agent_type in agents_to_eval:
                    for opp_deck in viz_opponents:
                        env = create_env(
                            config.player_deck,
                            opp_deck,
                            42,
                            max_turns=config.max_turns,
                        )
                        obs_dim = env.observation_space.shape[0]
                        act_dim = env.action_space.n
                        agent = create_agent(
                            agent_type,
                            obs_dim,
                            act_dim,
                            config.model_path if agent_type == config.agent_type else None,
                            42,
                        )
                        obs_normaliser = _build_eval_obs_normaliser(
                            config.vec_normalize_path if agent_type == config.agent_type else None,
                            env,
                        )
                        for game_num in range(1, config.show_games + 1):
                            run_report_game(
                                env=env,
                                agent=agent,
                                agent_name=agent_type.upper(),
                                game_num=game_num,
                                player_deck=config.player_deck,
                                opponent_deck=opp_deck,
                                output_dir=output_dir,
                                obs_normaliser=obs_normaliser,
                            )
                            progress.advance(task)
            console.print(f"[green]Reports saved under {output_dir / 'reports'}[/green]")
        else:
            print_divider("Detailed Game Visualizations & Reports")
            console.print(
                f"\n[bold]Recording {config.show_games} games per opponent "
                f"({', '.join(viz_opponents)})...[/]\n"
            )

            for agent_type in agents_to_eval:
                for opp_deck in viz_opponents:
                    env = create_env(
                        config.player_deck,
                        opp_deck,
                        42,
                        max_turns=config.max_turns,
                    )

                    obs_dim = env.observation_space.shape[0]
                    act_dim = env.action_space.n

                    agent = create_agent(
                        agent_type,
                        obs_dim,
                        act_dim,
                        config.model_path if agent_type == config.agent_type else None,
                        42,
                    )
                    obs_normaliser = _build_eval_obs_normaliser(
                        config.vec_normalize_path if agent_type == config.agent_type else None,
                        env,
                    )

                    for game_num in range(1, config.show_games + 1):
                        run_visualized_game(
                            env=env,
                            agent=agent,
                            agent_name=agent_type.upper(),
                            game_num=game_num,
                            player_deck=config.player_deck,
                            opponent_deck=opp_deck,
                            save_report=config.save_reports,
                            output_dir=output_dir,
                            obs_normaliser=obs_normaliser,
                        )

    return results


def demo_evaluation_results() -> None:
    """Demonstrate evaluation results table with sample data.

    This is useful for showcasing the evaluation visualization without
    running actual evaluations.
    """
    print_divider("Evaluation Results Demo")
    console.print("\n[dim]Displaying sample benchmark results...[/]\n")

    results = {
        "causal": {
            "win_rate": 0.672,
            "win_rate_std": 0.019,
            "avg_reward": 0.344,
            "reward_std": 0.082,
        },
        "ppo": {
            "win_rate": 0.587,
            "win_rate_std": 0.024,
            "avg_reward": 0.174,
            "reward_std": 0.095,
        },
        "greedy_aggro": {
            "win_rate": 0.423,
            "win_rate_std": 0.031,
            "avg_reward": -0.154,
            "reward_std": 0.089,
        },
        "random": {
            "win_rate": 0.251,
            "win_rate_std": 0.028,
            "avg_reward": -0.498,
            "reward_std": 0.075,
        },
    }

    print_evaluation_results(results, title="Full Benchmark Results")

    # Show trained models discovery
    print_section_header("Trained Models Discovery")
    models = discover_trained_models()

    if models:
        table = Table(title="Available Trained Models", box=box.ROUNDED)
        table.add_column("Name", style="cyan")
        table.add_column("Agent", style="green")
        table.add_column("Matchup", style="yellow")
        table.add_column("Path", style="dim")

        for model in models[:5]:
            table.add_row(
                model["name"][:30],
                model.get("agent_type", "?"),
                f"{model.get('player_deck', '?')} vs {model.get('opponent_deck', '?')}",
                model["path"][:40] + "...",
            )

        console.print(table)
    else:
        console.print("[dim]No trained models found in results/trained_agents/[/dim]")


def main_interactive() -> int:
    """Run evaluation in interactive mode.

    Returns:
        Exit code (0 for success).

    """
    global DELAYS

    console.clear()
    print_logo()

    console.print("\n[bold cyan]Interactive Evaluation Mode[/]")
    console.print("[dim]Answer the prompts to configure evaluation.[/dim]\n")

    # Get configuration interactively
    config = prompt_evaluation_config()

    # Confirm configuration
    if not confirm_config(config, "Evaluation"):
        console.print("[yellow]Evaluation cancelled.[/yellow]")
        return 1

    DELAYS = SPEED_PRESETS["fast"]

    # Run evaluation
    run_evaluation(config)

    return 0


def main_cli(args: argparse.Namespace) -> int:
    """Run evaluation with command-line arguments.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code (0 for success).

    """
    global DELAYS
    DELAYS = SPEED_PRESETS[args.speed]

    console.clear()
    print_logo()

    # Convert args to config
    config = args_to_config(args)

    print_divider("Evaluation Configuration")
    console.print(f"[bold cyan]Agent(s):[/] {config.agent_type}")
    console.print(f"[bold cyan]Matchup:[/] {config.player_deck} vs {config.opponent_deck}")
    console.print(f"[bold cyan]Episodes:[/] {config.episodes}")
    console.print(f"[bold cyan]Seeds:[/] {config.seeds}")
    if config.model_path:
        console.print(f"[bold cyan]Model:[/] {config.model_path}")
    console.print()

    # Run evaluation
    run_evaluation(config)

    return 0


def main_demo() -> int:
    """Run evaluation demo (sample results visualization).

    Returns:
        Exit code (0 for success).

    """
    console.clear()
    print_logo()

    console.print("\n[bold cyan]Evaluation Results Demo[/]")
    console.print("[dim]This demonstrates the evaluation results display.[/dim]\n")

    demo_evaluation_results()

    console.print("\n[bold green]Demo complete![/]")
    console.print("[dim]Run without --demo to evaluate actual agents.[/dim]\n")

    return 0


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success).

    """
    args = parse_args()

    if args.demo:
        return main_demo()
    elif args.interactive:
        return main_interactive()
    else:
        return main_cli(args)


if __name__ == "__main__":
    sys.exit(main())

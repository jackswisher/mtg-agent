r"""Why-did-the-agent-do-that case study for CGFA-PPO.

Runs a single deterministic episode with a trained CGFA-PPO checkpoint
and records, for every step, the **per-factor decomposition of the
critic** along with the SCM's per-factor predicted change.  Outputs:

* ``case_study_steps.csv``: one row per step with columns:

      ``turn, action, reward, done,
       v_scalar, gate, V_<factor_k>, A_<factor_k>, eps_<factor_k>,
       blended_advantage, scalar_advantage``

* ``case_study.png``: a 3-panel figure:

  1. Stacked bar of per-factor critic value V_k(s) per turn (positive
     contributions in colour, negative below the axis).
  2. Stacked bar of per-factor advantage A_k(s,a) per turn, the
     "credit attribution" each factor receives for the action played.
  3. Lines of SCM-predicted per-factor change eps_k(s,a) overlaid on
     A_k for all factors (visualises the calibration).

The single-episode rollout is deterministic and uses the same env config
the model was trained with so the figure is reproducible.

Usage::

    uv run python -m scripts.research.case_study \
        --model-path results/research/.../models/.../final_model.zip \
        --output-dir figures/case_study/cgfa_full_seed42 \
        --player-deck mono_red_aggro \
        --opponent-deck azorius_control \
        --max-turns 20 \
        --episode-seed 42
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as th

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mtg.agents.causal.cgfa_agent import CGFAAgent
from mtg.agents.reinforcement_learning.cgfa import (
    CGFAEnvWrapper,
    FactorSpec,
)
from mtg.training.env_factory import EnvConfig, create_env

# High-contrast, colourblind-friendly palette for dense stacked factors.
_PALETTE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
]


# ---------------------------------------------------------------------------
# Episode rollout
# ---------------------------------------------------------------------------


def _wrap_for_cgfa(env, factor_spec: FactorSpec, scm):
    """Wrap a raw env with :class:`CGFAEnvWrapper`.

    The case-study script applies the action mask manually below, so we
    do not stack an :class:`ActionMasker` on top of the CGFA wrapper.
    """
    return CGFAEnvWrapper(env, factor_spec=factor_spec, scm=scm)


def _action_masks_from_info(env, info: dict) -> np.ndarray:
    """Return a boolean action-mask from info or env.action_mask()."""
    mask = info.get("action_mask") if info else None
    if mask is None and hasattr(env, "action_mask"):
        try:
            mask = env.action_mask()
        except TypeError:
            mask = None
    if mask is None and hasattr(env.unwrapped, "action_mask"):
        try:
            mask = env.unwrapped.action_mask()
        except TypeError:
            mask = None
    if mask is None:
        mask = np.ones(env.action_space.n, dtype=bool)
    return np.asarray(mask, dtype=bool)


def _format_deck_name(deck: str) -> str:
    """Format a deck identifier for figure titles."""
    return deck.replace("_", " ").title()


def rollout_one_episode(
    *,
    agent: CGFAAgent,
    env,
    deterministic: bool = True,
    max_steps: int = 200,
    episode_seed: int = 42,
) -> tuple[list[dict], dict]:
    """Run one deterministic episode and return per-step records + outcome.

    Each step record is a dict with the columns described in the module
    docstring (everything required to materialise the case-study figure).
    The second return value is an ``outcome`` dict carrying the
    episode-level summary (game result, total reward, total steps,
    truncation flag) so the figure title can disclose whether the case
    study is a win, a loss, or a draw.
    """
    rows: list[dict] = []
    obs, info = env.reset(seed=episode_seed)

    factor_names = list(agent.factor_spec.names)
    n_factors = agent.factor_spec.n_factors
    blend = agent.factor_spec.blend_init.astype(np.float64)
    blend = blend / max(blend.sum(), 1e-8)

    model = agent.model
    if model is None:
        raise RuntimeError("CGFAAgent.model is None; load the checkpoint first via agent.load(...)")
    policy = model.policy

    step = 0
    done = False
    truncated_flag = False
    last_info: dict = dict(info or {})
    total_reward = 0.0
    while not done and step < max_steps:
        mask = _action_masks_from_info(env, info)
        # --- Critic decomposition --------------------------------------
        obs_tensor, _ = policy.obs_to_tensor(np.asarray(obs))
        with th.no_grad():
            v_scalar_t, v_factors_t = policy.predict_factor_values(obs_tensor)
            gate_t = policy.predict_gate(obs_tensor)
        v_scalar = float(v_scalar_t.cpu().numpy()[0])
        v_factors = v_factors_t.cpu().numpy()[0]
        gate = float(gate_t.cpu().numpy()[0])

        # --- Action selection ------------------------------------------
        action, _ = model.predict(
            np.asarray(obs),
            deterministic=deterministic,
            action_masks=mask,
        )
        action = int(action)

        # --- Step --------------------------------------------------------
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        truncated_flag = bool(truncated)
        last_info = dict(info or {})
        total_reward += float(reward)

        # --- Per-factor advantage ---------------------------------------
        # A_k is computed on-policy as the one-step TD residual:
        #   A_k(s,a) ~= r_k + gamma * V_k(s') - V_k(s)
        # (This is consistent with the GAE used in training when gae_lambda=0
        #  and is the simplest interpretable per-step credit signal.)
        factor_rewards = np.asarray(info.get("factor_rewards", np.zeros(n_factors)), dtype=float)
        factor_eps = np.asarray(info.get("factor_eps", np.zeros(n_factors)), dtype=float)
        if not done:
            with th.no_grad():
                next_obs_tensor, _ = policy.obs_to_tensor(np.asarray(next_obs))
                _, v_next_factors_t = policy.predict_factor_values(next_obs_tensor)
            v_next_factors = v_next_factors_t.cpu().numpy()[0]
        else:
            v_next_factors = np.zeros_like(v_factors)
        gamma = float(getattr(model, "gamma", 0.99))
        a_factors = factor_rewards + gamma * v_next_factors - v_factors

        # The blended scalar / factor advantage used in the actual update.
        scalar_advantage = float(reward + gamma * (0.0 if done else v_scalar) - v_scalar)
        weighted = float((blend * a_factors).sum())
        blended_advantage = (1.0 - gate) * scalar_advantage + gate * weighted

        row: dict = {
            "step": step,
            "turn": info.get("turn", step) if info else step,
            "action": action,
            "action_name": _action_label(env, info, action),
            "reward": float(reward),
            "done": int(done),
            "v_scalar": v_scalar,
            "gate": gate,
            "blended_advantage": float(blended_advantage),
            "scalar_advantage": scalar_advantage,
        }
        for k, name in enumerate(factor_names):
            row[f"V_{name}"] = float(v_factors[k])
            row[f"A_{name}"] = float(a_factors[k])
            row[f"eps_{name}"] = float(factor_eps[k])
            row[f"r_{name}"] = float(factor_rewards[k])
        rows.append(row)
        obs = next_obs
        step += 1

    outcome: dict = {
        "game_result": last_info.get("game_result", "unknown"),
        "total_reward": float(total_reward),
        "total_steps": int(step),
        "truncated": bool(truncated_flag),
        "max_steps_reached": int(step) >= int(max_steps),
        "episode_seed": int(episode_seed),
    }
    return rows, outcome


def _action_label(env, info, action: int) -> str:
    """Best-effort human-readable label for an action.

    Returns ``str(action)`` if no descriptor is available.
    """
    if info and isinstance(info.get("action_name"), str):
        return str(info["action_name"])
    descr = getattr(env.unwrapped, "describe_action", None)
    if callable(descr):
        try:
            return str(descr(action))
        except Exception:  # pragma: no cover (best-effort only)
            pass
    return f"action_{action}"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _mean_rows_by_turn(rows: list[dict], factor_names: list[str]) -> list[dict]:
    """Collapse per-decision rows into one plotted row per turn.

    The CSV remains per decision, but the figure is a turn-level summary.
    Without this aggregation, multiple decisions from the same turn are drawn
    at the same x-position and create misleading internal stripes in the bars.
    """
    if not rows:
        return []

    fields = [f"{prefix}{name}" for prefix in ("V_", "A_", "eps_") for name in factor_names]
    by_turn: dict[int, list[dict]] = {}
    for row in rows:
        by_turn.setdefault(int(row["turn"]), []).append(row)

    aggregated: list[dict] = []
    for turn, turn_rows in sorted(by_turn.items()):
        out: dict[str, float | int] = {"turn": turn}
        for field in fields:
            vals = [float(row[field]) for row in turn_rows if field in row]
            out[field] = float(np.mean(vals)) if vals else 0.0
        aggregated.append(out)
    return aggregated


def _stacked_bar_signed(
    ax: plt.Axes,
    rows: list[dict],
    prefix: str,
    factor_names: list[str],
    title: str,
    ylabel: str,
) -> None:
    """Stacked bar where positive contributions stack up, negatives stack down.

    ``prefix`` is one of ``"V_"`` (critic decomposition) or ``"A_"``
    (per-factor advantage decomposition).
    """
    if not rows:
        return
    turns = [r["turn"] for r in rows]
    matrix = np.array(
        [[float(r[f"{prefix}{name}"]) for name in factor_names] for r in rows]
    )  # (T, K)

    pos_running = np.zeros(len(rows))
    neg_running = np.zeros(len(rows))
    for k, name in enumerate(factor_names):
        col = matrix[:, k]
        pos = np.where(col > 0, col, 0.0)
        neg = np.where(col < 0, col, 0.0)
        ax.bar(
            turns,
            pos,
            bottom=pos_running,
            color=_PALETTE[k % len(_PALETTE)],
            edgecolor="white",
            linewidth=0.35,
            label=name,
        )
        ax.bar(
            turns,
            neg,
            bottom=neg_running,
            color=_PALETTE[k % len(_PALETTE)],
            edgecolor="white",
            linewidth=0.35,
        )
        pos_running = pos_running + pos
        neg_running = neg_running + neg

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Turn")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.2)


def _calibration_overlay(
    ax: plt.Axes,
    rows: list[dict],
    factor_names: list[str],
) -> None:
    """Overlay learned per-factor advantage A_k vs the SCM-predicted eps_k.

    Plots **every** factor in ``factor_names`` with one colour per factor,
    drawing ``A_k`` as a solid line and ``eps_k`` as a dashed line in the
    same colour.  Factors are ordered by importance (largest L1 mass of
    ``A_k`` summed over the episode) so the most-explanatory factors are
    drawn last and stay on top in the legend.
    """
    if not rows:
        return
    importance = {name: float(np.sum([abs(r[f"A_{name}"]) for r in rows])) for name in factor_names}
    ranked = sorted(factor_names, key=lambda n: importance[n])

    turns = [r["turn"] for r in rows]
    for k, name in enumerate(ranked):
        a_series = [r[f"A_{name}"] for r in rows]
        eps_series = [r[f"eps_{name}"] for r in rows]
        color = _PALETTE[k % len(_PALETTE)]
        ax.plot(turns, a_series, color=color, lw=1.8, label=rf"$A_{{{name}}}$ (learned)")
        ax.plot(
            turns,
            eps_series,
            color=color,
            lw=1.4,
            linestyle="--",
            label=rf"$\hat{{\epsilon}}_{{{name}}}$ (SCM)",
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Per-factor advantage / SCM eps")
    ax.set_title(
        f"Calibration overlay: learned A_k vs SCM eps_k (all {len(factor_names)} factors)",
        fontweight="bold",
    )
    # Two columns keep the legend compact when many factors are plotted.
    ax.legend(loc="best", fontsize=7, ncol=2, framealpha=0.85)
    ax.grid(alpha=0.2)


def _format_outcome_for_title(outcome: dict | None) -> str:
    """Render ``outcome`` as a compact human-readable suffix."""
    if not outcome:
        return ""
    parts: list[str] = []
    result = str(outcome.get("game_result", "")).lower()
    if result == "win":
        parts.append("WIN")
    elif result == "loss":
        parts.append("LOSS")
    elif result == "draw":
        parts.append("DRAW")
    elif result:
        parts.append(result.upper())
    if outcome.get("max_steps_reached"):
        parts.append("step-cap reached")
    elif outcome.get("truncated"):
        parts.append("truncated")
    if "total_reward" in outcome:
        parts.append(f"R={float(outcome['total_reward']):+.2f}")
    if "total_steps" in outcome:
        parts.append(f"T={int(outcome['total_steps'])} steps")
    return ", ".join(parts)


def render_case_study(
    rows: list[dict],
    output_path: Path,
    factor_names: list[str],
    title_suffix: str = "",
    *,
    outcome: dict | None = None,
) -> Path:
    """Render the 3-panel case-study PNG and write it to ``output_path``.

    The episode ``outcome`` dict (returned by :func:`rollout_one_episode`)
    is formatted into the figure suptitle so reviewers can see at a
    glance whether the case study is a win or a loss.
    """
    plot_rows = _mean_rows_by_turn(rows, factor_names)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), facecolor="white")
    for ax in axes:
        ax.set_facecolor("white")

    _stacked_bar_signed(
        axes[0],
        plot_rows,
        prefix="V_",
        factor_names=factor_names,
        title="Mean critic decomposition by turn",
        ylabel=r"Per-factor critic $V_k(s)$",
    )
    _stacked_bar_signed(
        axes[1],
        plot_rows,
        prefix="A_",
        factor_names=factor_names,
        title="Mean per-factor advantage by turn",
        ylabel=r"Per-factor advantage $A_k(s,a)$",
    )
    _calibration_overlay(axes[2], plot_rows, factor_names)

    suptitle = "CGFA-PPO case study"
    outcome_str = _format_outcome_for_title(outcome)
    if title_suffix and outcome_str:
        suptitle = f"{suptitle}: {title_suffix} [{outcome_str}]"
    elif title_suffix:
        suptitle = f"{suptitle}: {title_suffix}"
    elif outcome_str:
        suptitle = f"{suptitle} [{outcome_str}]"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_case_study_csv(
    rows: list[dict],
    csv_path: Path,
    *,
    outcome: dict | None = None,
) -> Path:
    """Persist all per-step records to a wide CSV (one row per step).

    If ``outcome`` is provided it is also written to a sibling JSON file
    ``case_study_outcome.json`` so the figure can be regenerated later
    (via ``--from-csv``) without losing the win/loss/draw label.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return csv_path
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if outcome is not None:
        import json

        outcome_path = csv_path.with_name("case_study_outcome.json")
        outcome_path.write_text(json.dumps(outcome, indent=2, default=str))
    return csv_path


def read_case_study_outcome(csv_path: Path) -> dict | None:
    """Load the sibling ``case_study_outcome.json`` for a case-study CSV.

    Returns ``None`` if the sidecar file is missing; callers must treat
    this as "outcome unknown" rather than as an error.
    """
    import json

    outcome_path = Path(csv_path).with_name("case_study_outcome.json")
    if not outcome_path.exists():
        return None
    try:
        return json.loads(outcome_path.read_text())
    except (OSError, ValueError):
        return None


def read_case_study_csv(csv_path: Path) -> tuple[list[dict], list[str]]:
    """Reverse of :func:`write_case_study_csv`: load rows + factor names from disk.

    Recovers numeric types so the loaded ``rows`` are drop-in compatible
    with :func:`render_case_study` (i.e. the figure can be regenerated
    without re-rolling out the episode).

    Returns:
        ``(rows, factor_names)`` where ``factor_names`` is derived from
        the ``V_<name>`` columns in the CSV header (the canonical source
        of truth for which factors the run used).
    """
    rows: list[dict] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"empty or malformed case-study CSV: {csv_path}")
        factor_names = [name[len("V_") :] for name in reader.fieldnames if name.startswith("V_")]
        # Columns we know are numeric.  Anything outside this set
        # (e.g. ``action_name``) is preserved as-is.
        int_cols = {"step", "turn", "action", "done"}
        for raw in reader:
            row: dict = {}
            for k, v in raw.items():
                if v is None or v == "":
                    row[k] = v
                    continue
                if k in int_cols:
                    try:
                        row[k] = int(float(v))
                    except (TypeError, ValueError):
                        row[k] = v
                    continue
                # Numeric-looking columns get cast to float; otherwise keep string.
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    row[k] = v
            rows.append(row)
    return rows, factor_names


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        prog="case_study",
        description=(
            "Run one deterministic CGFA-PPO episode and produce a case-study "
            "table + figure decomposing the critic and advantage per factor."
        ),
    )
    # ``--model-path`` is required for fresh rollouts but ignored when
    # ``--from-csv`` is used (re-render only, no model needed).
    p.add_argument("--model-path", type=Path, required=False, default=None)
    p.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help=(
            "Skip the rollout and re-render the case-study figure from an "
            "existing case_study_steps.csv. The CSV's V_<factor> columns "
            "are the source of truth for the factor list."
        ),
    )
    p.add_argument("--output-dir", type=Path, default=Path("figures/case_study"))
    p.add_argument(
        "--player-deck",
        default="mono_red_aggro",
        help="Player deck archetype (matches mtg.env.deck_archetypes).",
    )
    p.add_argument(
        "--opponent-deck",
        default="azorius_control",
        help="Opponent deck archetype.",
    )
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument("--episode-seed", type=int, default=42)
    p.add_argument(
        "--agency",
        choices=["auto", "full"],
        default="auto",
        help=(
            "Player agency mode.  'auto' = engine resolves combat/targeting; "
            "'full' = the model selects every micro-decision."
        ),
    )
    p.add_argument(
        "--reward-type",
        choices=["sparse", "shaped"],
        default="sparse",
        help=(
            "Reward type for the case-study rollout.  Default 'sparse' "
            "matches the eval pipeline so factor decompositions reflect "
            "the same outcome-driven signal as the headline win-rate "
            "tables.  Use 'shaped' only if you want to inspect the "
            "potential-based shaping channel itself."
        ),
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy (default: True).",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Hard cap on the number of agent steps (safety net).",
    )
    return p.parse_args()


def main() -> int:
    """Entry point for ``python -m scripts.research.case_study``."""
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suptitle = (
        f"{_format_deck_name(args.player_deck)} player deck vs "
        f"{_format_deck_name(args.opponent_deck)} opponent "
        f"(seed {args.episode_seed})"
    )

    # --- Replot-from-disk path: no env, no model, no rollout. ------------
    if args.from_csv is not None:
        if not args.from_csv.exists():
            print(f"ERROR: --from-csv path does not exist: {args.from_csv}")
            return 1
        rows, factor_names = read_case_study_csv(args.from_csv)
        if not rows:
            print(f"ERROR: --from-csv file has no rows: {args.from_csv}")
            return 1
        outcome = read_case_study_outcome(args.from_csv)
        fig_path = render_case_study(
            rows,
            args.output_dir / "case_study.png",
            factor_names,
            title_suffix=suptitle,
            outcome=outcome,
        )
        print(f"wrote {fig_path}  (replot from {args.from_csv})")
        return 0

    # --- Fresh-rollout path: requires a trained model. -------------------
    if args.model_path is None:
        print("ERROR: --model-path is required (or pass --from-csv to replot from disk)")
        return 1

    auto = args.agency == "auto"
    env_cfg = EnvConfig(
        player_deck=args.player_deck,
        opponent_deck=args.opponent_deck,
        max_turns=args.max_turns,
        auto_combat=auto,
        auto_target=auto,
        reward_type=args.reward_type,
        seed=args.episode_seed,
    )
    raw_env = create_env(env_cfg)
    factor_spec = FactorSpec()
    from mtg.causal.scm import StructuralCausalModel

    scm = StructuralCausalModel()
    env = _wrap_for_cgfa(raw_env, factor_spec, scm)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    agent = CGFAAgent(
        observation_dim=obs_dim,
        action_dim=act_dim,
        factor_spec=factor_spec,
        scm=scm,
    )
    agent.load(args.model_path)

    rows, outcome = rollout_one_episode(
        agent=agent,
        env=env,
        deterministic=args.deterministic,
        max_steps=args.max_steps,
        episode_seed=args.episode_seed,
    )

    csv_path = write_case_study_csv(rows, args.output_dir / "case_study_steps.csv", outcome=outcome)
    fig_path = render_case_study(
        rows,
        args.output_dir / "case_study.png",
        list(agent.factor_spec.names),
        title_suffix=suptitle,
        outcome=outcome,
    )

    print(f"wrote {csv_path}")
    print(f"wrote {fig_path}")
    print(
        f"  outcome: {outcome.get('game_result')} "
        f"(reward={outcome.get('total_reward'):+.2f}, "
        f"steps={outcome.get('total_steps')}, "
        f"reward_type={args.reward_type})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

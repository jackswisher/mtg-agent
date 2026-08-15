"""Publication-grade visualizations for MTG-Causal-RL.

This module provides sleek, publication-quality figures for the benchmark,
including causal graphs, learning curves, and comparison plots.
"""

from __future__ import annotations

import typing as tp
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# =============================================================================
# Publication Style Configuration
# =============================================================================

PUBLICATION_STYLE = {
    # Figure size for single column (3.25") and double column (6.75")
    "figure.figsize": (6.75, 4.5),
    "figure.dpi": 150,
    "figure.facecolor": "white",
    # Fonts - use serif for publication
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    # Lines and markers
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    # Axes
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    # Legend
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    # Save settings
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

# Color palette - inspired by scientific publications
COLORS = {
    "primary": "#2E4057",  # Deep blue
    "secondary": "#048A81",  # Teal
    "accent": "#E63946",  # Red accent
    "highlight": "#F4A261",  # Orange highlight
    "muted": "#8D99AE",  # Muted gray-blue
    "success": "#2A9D8F",  # Green
    "warning": "#E9C46A",  # Yellow
    "danger": "#E76F51",  # Coral
    # Agent colors
    "random": "#8D99AE",
    "heuristic": "#457B9D",
    "ppo": "#E63946",
    "causal": "#2A9D8F",
    # Layer colors for SCM
    "resource": "#A8DADC",
    "board_state": "#457B9D",
    "strategic": "#1D3557",
    "outcome": "#E63946",
}

AGENT_MARKERS = {
    "random": "o",
    "heuristic": "s",
    "ppo": "^",
    "causal": "D",
}


def apply_publication_style() -> None:
    """Apply publication style to matplotlib."""
    plt.rcParams.update(PUBLICATION_STYLE)


def create_learning_curve(
    data: dict[str, dict[str, tp.Any]],
    metric: str = "win_rate",
    title: str = "Learning Curves",
    save_path: Path | None = None,
    show_confidence: bool = True,
) -> plt.Figure:
    """Create polished learning-curve figures suitable for papers.

    Args:
        data: Dict mapping agent name to {steps, mean, std}.
        metric: Metric being plotted.
        title: Figure title.
        save_path: Optional path to save figure.
        show_confidence: Whether to show confidence bands.

    Returns:
        Matplotlib figure.

    """
    apply_publication_style()

    fig, ax = plt.subplots(figsize=(6.75, 4))

    for agent_name, values in data.items():
        steps = values["steps"]
        mean = values["mean"]
        std = values.get("std", np.zeros_like(mean))

        color = COLORS.get(agent_name.lower(), COLORS["muted"])
        marker = AGENT_MARKERS.get(agent_name.lower(), "o")

        # Plot mean line
        ax.plot(
            steps,
            mean,
            color=color,
            marker=marker,
            markevery=max(1, len(steps) // 8),
            label=agent_name,
            linewidth=2,
            markersize=7,
        )

        # Confidence band
        if show_confidence and np.any(std > 0):
            ax.fill_between(
                steps,
                mean - std,
                mean + std,
                color=color,
                alpha=0.15,
            )

    ax.set_xlabel("Training Steps", fontweight="medium")
    ax.set_ylabel(metric.replace("_", " ").title(), fontweight="medium")
    ax.set_title(title, fontweight="bold", pad=10)

    # Format x-axis with K/M suffixes
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"{x / 1000:.0f}K" if x < 1e6 else f"{x / 1e6:.1f}M")
    )

    ax.legend(loc="lower right", framealpha=0.95)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def create_comparison_bar(
    results: dict[str, dict[str, float]],
    metrics: list[str] | None = None,
    title: str = "Agent Comparison",
    save_path: Path | None = None,
) -> plt.Figure:
    """Create a polished comparison bar chart suitable for papers.

    Args:
        results: Dict mapping agent name to {metric: value}.
        metrics: List of metrics to show.
        title: Figure title.
        save_path: Optional path to save.

    Returns:
        Matplotlib figure.

    """
    apply_publication_style()

    if metrics is None:
        metrics = ["win_rate", "avg_reward"]

    agents = list(results.keys())
    n_agents = len(agents)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics, strict=False):
        values = [results[a].get(metric, 0) for a in agents]
        stds = [results[a].get(f"{metric}_std", 0) for a in agents]
        colors = [COLORS.get(a.lower(), COLORS["muted"]) for a in agents]

        bars = ax.bar(
            range(n_agents),
            values,
            yerr=stds,
            capsize=4,
            color=colors,
            edgecolor="white",
            linewidth=1.5,
            error_kw={"linewidth": 1.5, "capthick": 1.5},
        )

        # Add value labels on bars
        for bar, val, std in zip(bars, values, stds, strict=False):
            height = bar.get_height()
            label = f"{val:.1%}" if metric == "win_rate" else f"{val:.2f}"
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, height + std + 0.02),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="medium",
            )

        ax.set_xticks(range(n_agents))
        ax.set_xticklabels(agents, rotation=0)
        ax.set_ylabel(metric.replace("_", " ").title(), fontweight="medium")
        ax.set_ylim(bottom=0, top=max(values) * 1.25)

    fig.suptitle(title, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def create_scm_diagram(
    save_path: Path | None = None,
    style: str = "modern",
) -> plt.Figure:
    """Create a polished SCM diagram suitable for papers.

    Args:
        save_path: Optional path to save.
        style: Visual style ('modern', 'classic').

    Returns:
        Matplotlib figure.

    """
    apply_publication_style()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-0.5, 4.5)
    ax.axis("off")

    # Layer definitions
    layers = {
        "RESOURCE": {
            "y": 4,
            "color": COLORS["resource"],
            "nodes": [
                ("mana_t", 2),
                ("mana_t+1", 5),
                ("card_count", 8),
            ],
        },
        "BOARD STATE": {
            "y": 3,
            "color": COLORS["board_state"],
            "nodes": [
                ("board_presence", 2),
                ("board_press", 5),
                ("threat_density", 8),
            ],
        },
        "STRATEGIC": {
            "y": 2,
            "color": COLORS["strategic"],
            "nodes": [
                ("card_adv", 1.5),
                ("tempo", 4),
                ("life_buffer", 6.5),
                ("removal", 9),
            ],
        },
        "OUTCOME": {
            "y": 1,
            "color": COLORS["outcome"],
            "nodes": [
                ("win_prob", 5),
            ],
        },
    }

    # Draw layer backgrounds
    for layer_name, layer_info in layers.items():
        y = layer_info["y"]
        rect = FancyBboxPatch(
            (-0.5, y - 0.4),
            11,
            0.8,
            boxstyle="round,pad=0.05,rounding_size=0.1",
            facecolor=layer_info["color"],
            alpha=0.15,
            edgecolor=layer_info["color"],
            linewidth=1.5,
        )
        ax.add_patch(rect)

        # Layer label
        ax.text(
            -0.3,
            y,
            layer_name,
            fontsize=8,
            fontweight="bold",
            color=layer_info["color"],
            rotation=90,
            va="center",
            ha="right",
        )

    # Draw nodes
    node_positions = {}
    for _layer_name, layer_info in layers.items():
        y = layer_info["y"]
        color = layer_info["color"]

        for node_name, x in layer_info["nodes"]:
            node_positions[node_name] = (x, y)

            # Node circle
            circle = plt.Circle(
                (x, y),
                0.3,
                facecolor="white",
                edgecolor=color,
                linewidth=2,
                zorder=10,
            )
            ax.add_patch(circle)

            # Node label
            display_name = node_name.replace("_", "\n")
            ax.text(
                x,
                y,
                display_name,
                fontsize=7,
                fontweight="medium",
                ha="center",
                va="center",
                zorder=11,
            )

    # Define edges
    edges = [
        ("mana_t", "mana_t+1"),
        ("mana_t", "board_presence"),
        ("mana_t", "board_press"),
        ("board_presence", "threat_density"),
        ("board_presence", "board_press"),
        ("card_count", "card_adv"),
        ("board_presence", "card_adv"),
        ("mana_t", "tempo"),
        ("board_press", "tempo"),
        ("board_press", "life_buffer"),
        ("card_count", "removal"),
        ("mana_t", "removal"),
        ("card_adv", "win_prob"),
        ("tempo", "win_prob"),
        ("life_buffer", "win_prob"),
        ("threat_density", "win_prob"),
        ("board_press", "win_prob"),
    ]

    # Draw edges
    for start, end in edges:
        if start in node_positions and end in node_positions:
            x1, y1 = node_positions[start]
            x2, y2 = node_positions[end]

            # Adjust for node radius
            dx = x2 - x1
            dy = y2 - y1
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0:
                x1 += 0.3 * dx / dist
                y1 += 0.3 * dy / dist
                x2 -= 0.3 * dx / dist
                y2 -= 0.3 * dy / dist

            arrow = FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=12,
                color="#555555",
                linewidth=1,
                alpha=0.7,
                connectionstyle="arc3,rad=0.1",
                zorder=5,
            )
            ax.add_patch(arrow)

    # Title
    ax.set_title(
        "Structural Causal Model for MTG Strategic Decisions",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")

    return fig


def create_generalization_heatmap(
    results: dict[str, dict[str, float]],
    agents: list[str],
    matchups: list[tuple[str, str]],
    save_path: Path | None = None,
) -> plt.Figure:
    """Create heatmap showing generalization across matchups.

    Args:
        results: Results dictionary.
        agents: List of agent names.
        matchups: List of (deck, opponent) tuples.
        save_path: Optional save path.

    Returns:
        Matplotlib figure.

    """
    apply_publication_style()

    n_agents = len(agents)
    n_matchups = len(matchups)

    # Build data matrix
    data = np.zeros((n_agents, n_matchups))
    for i, agent in enumerate(agents):
        for j, (deck, opp) in enumerate(matchups):
            key = f"{agent}_{deck}_vs_{opp}"
            data[i, j] = results.get(key, {}).get("win_rate", 0.5)

    # Create custom colormap
    colors = ["#E63946", "#F1FAEE", "#2A9D8F"]
    cmap = LinearSegmentedColormap.from_list("winrate", colors)

    fig, ax = plt.subplots(figsize=(8, 4))

    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Labels
    ax.set_xticks(range(n_matchups))
    ax.set_xticklabels([f"{d[:3]}v{o[:3]}" for d, o in matchups], rotation=45, ha="right")
    ax.set_yticks(range(n_agents))
    ax.set_yticklabels(agents)

    # Add text annotations
    for i in range(n_agents):
        for j in range(n_matchups):
            text = f"{data[i, j]:.0%}"
            color = "white" if data[i, j] < 0.3 or data[i, j] > 0.7 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Win Rate", fontweight="medium")

    ax.set_title("Generalization Across Matchups", fontweight="bold", pad=10)
    ax.set_xlabel("Matchup", fontweight="medium")
    ax.set_ylabel("Agent", fontweight="medium")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig

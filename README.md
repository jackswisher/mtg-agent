# MTG-Causal-RL

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A Causal Reinforcement Learning Benchmark for Magic: The Gathering**

MTG-Causal-RL is a Gymnasium benchmark for studying causal reinforcement
learning in strategic card games. It combines partial observability, legal
action masking, five Standard-style deck archetypes, and an explicit
Structural Causal Model (SCM) over strategic game variables.

---

## Table of Contents

- [Key Features](#key-features)
- [Benchmark Configuration](#benchmark-configuration)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
  - [Out-of-the-Box Workflows](#out-of-the-box-workflows)
- [The Causal Model](#the-causal-model)
- [Action Space](#action-space)
- [Observation Space](#observation-space)
- [Environment Configuration](#environment-configuration)
- [Deck Archetypes](#deck-archetypes)
- [Agents](#agents)
  - [Causal Agent Features](#causal-agent-features)
  - [Causal Graph-Factored Advantage (CGFA-PPO)](#causal-graph-factored-advantage-cgfa-ppo)
  - [Agent Structure (Folders)](#agent-structure-folders)
  - [Using the Agent Registry](#using-the-agent-registry)
- [Workflows](#workflows)
  - [1. Training (`mtg-train`)](#1-training-mtg-train)
  - [2. Evaluation (`mtg-eval`)](#2-evaluation-mtg-eval)
  - [3. Gameplay Demo (`mtg-gameplay`)](#3-gameplay-demo-mtg-gameplay)
  - [4. Research Pipeline (`mtg-research`)](#4-research-pipeline-mtg-research)
  - [5. The CGFA Benchmark Pipeline](#5-the-cgfa-benchmark-pipeline)
  - [Replotting figures from disk](#replotting-figures-from-disk)
  - [HTML Gameplay Reports](#html-gameplay-reports)
- [Reproducing Benchmark Results](#reproducing-benchmark-results)
  - [One-command reproduction (`mtg-research paper`)](#one-command-reproduction-mtg-research-paper)
  - [Smoke recipe (a few hours, sanity check)](#smoke-recipe-a-few-hours-sanity-check)
  - [Optional CGFA signal pre-flight](#optional-cgfa-signal-pre-flight)
  - [Full benchmark reproduction](#full-benchmark-reproduction)
  - [Resuming an interrupted benchmark run](#resuming-an-interrupted-benchmark-run)
  - [Exporting figures and tables](#exporting-figures-and-tables)
  - [Expected Output Tree](#expected-output-tree)
- [Testing](#testing)
  - [Running Tests](#running-tests)
  - [Test Categories](#test-categories)
- [Project Structure](#project-structure)
- [Visualizations](#visualizations)
  - [Benchmark Figures](#benchmark-figures)
  - [CLI Experience](#cli-experience)
  - [Programmatic Visualization](#programmatic-visualization)
- [Extending the Benchmark](#extending-the-benchmark)
  - [Adding a Custom Card](#adding-a-custom-card)
  - [Adding a Custom Archetype](#adding-a-custom-archetype)
  - [Adding a Custom Agent](#adding-a-custom-agent)
  - [Advanced Custom Agent (with Learning)](#advanced-custom-agent-with-learning)
  - [Adding a Causal Variable](#adding-a-causal-variable)
  - [Future Directions](#future-directions)
- [Tutorial](#tutorial)
- [Citation](#citation)
- [Related Work](#related-work)
- [License](#license)
- [Contributing](#contributing)

---

## Key Features

- **Causal benchmark interface**: 11 SCM variables across resource, board,
  strategic, and outcome layers.
- **MTG-inspired environment**: 56-card Standard 2025 pool, 60-card decks,
  five competitive archetypes, hidden hands, and legal action masking.
- **Gymnasium compatible**: Use the environment with standard RL tooling.
- **Baselines included**: Random, heuristic, PPO, CausalAgent, and CGFA-PPO.
- **Extensible by design**: Add cards, archetypes, agents, reward schemes,
  or causal variables without changing the core API.
- **Reproducible benchmark pipeline**: Multi-seed sweeps, ablations,
  transfer evaluation, calibration plots, case studies, and statistical
  aggregation through the `mtg-research` CLI.
- **Tutorial notebook**: A guided walkthrough in `tutorial/getting_started.ipynb`.

---

## Benchmark Configuration

| Component | Specification |
|-----------|---------------|
| Card Pool | 56 unique cards (Standard 2025 legal) |
| Deck Size | 60 cards |
| Archetypes | 5 (Mono-Red Aggro, Azorius Control, Dimir Midrange, Domain Ramp, Boros Convoke) |
| Max Turns | 10 (default; configurable 1 to 20) |
| Action Space | 478 discrete actions with legal action masking (typically 2 to 15 legal per state) |
| Observation | Partial (own hand, battlefields, life totals, graveyard) |

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| gymnasium | ≥0.29.0 | RL environment interface |
| stable-baselines3 | ≥2.2.0 | PPO implementation |
| torch | ≥2.0.0 | Neural network backend |
| networkx | ≥3.0 | Causal graph operations |
| numpy | ≥1.24.0 | Numerical operations |
| rich | ≥13.0.0 | CLI visualization |
| matplotlib | ≥3.7.0 | Figure generation |

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/anonymous/mtg-causal-rl.git
cd mtg-causal-rl

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e ".[dev]"

# Verify installation
python -c "from mtg.env import MTGEnv; print('Installation successful!')"
```

### Basic Usage

```python
from mtg.env import MTGEnv
from mtg.agents import RandomAgent, GreedyAggroAgent, CausalAgent

# Create environment
env = MTGEnv(
    deck_archetype="mono_red_aggro",
    opponent_archetype="azorius_control",
    max_turns=10,
    reward_type="shaped",
)

# Create agent
agent = RandomAgent(seed=42)

# Run episode
obs, info = env.reset(seed=42)
done = False

while not done:
    action_mask = info["action_mask"]
    action = agent.select_action(obs, action_mask, info)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print(f"Result: {info.get('game_result', 'unknown')}")
```

### Out-of-the-Box Workflows

These commands cover the common paths without reading the rest of the README.
Run them from the repository root after installation.

#### Play a Game

```bash
uv run mtg-gameplay
```

Starts the interactive gameplay demo with terminal visualization and optional
HTML replay output.

#### Train a PPO Agent

```bash
uv run mtg-train \
    --agent ppo \
    --deck mono_red_aggro \
    --opponent azorius_control \
    --timesteps 100000
```

Trains a deck-specific PPO agent and writes checkpoints, metrics, plots, and
sample replays under `results/trained_agents/`.

#### Evaluate Baselines

```bash
uv run mtg-eval \
    --agent all \
    --opponent all \
    --episodes 100
```

Runs the registered agents against the benchmark opponents and writes summary
JSON/CSV files plus comparison plots under `results/evaluations/`.

#### Run a Benchmark Smoke Test

```bash
uv run mtg-research paper \
    --experiment-name smoke_$(date +%Y%m%d_%H%M) \
    --agents ppo cgfa \
    --player-decks mono_red_aggro azorius_control \
    --opponents mono_red_aggro azorius_control dimir_midrange \
    --heldout-opponents domain_ramp boros_convoke \
    --transfer-mode fixed \
    --seeds 42 123 456 \
    --timesteps-per-opponent 200000 \
    --eval-episodes 100
```

Runs the full benchmark workflow at a small budget: headline comparison,
ablation, transfer, calibration plot, case study, and cross-source aggregation.

#### Export Figures and Tables

```bash
EXP=<your_experiment_name>
uv run python -m scripts.research.export_paper_bundle \
    --experiment-name $EXP \
    --output-dir results/research/$EXP/benchmark_bundle
```

Collects generated figures, LaTeX table fragments, and raw JSON provenance into
one portable directory.

---

## The Causal Model

The environment exposes an explicit SCM for MTG strategic reasoning:

```
┌────────────────────────────────────────────────────────────┐
│                    RESOURCE LAYER                          │
│    mana_t ──────┬──────────> mana_t1                       │
│                 │                                          │
│    card_count ──┼───────────────────────────────────────┐  │
└─────────────────┼───────────────────────────────────────┼──┘
                  │                                       │
┌─────────────────▼───────────────────────────────────────┼──┐
│                    BOARD STATE LAYER                    │  │
│    board_presence ────────> threat_density              │  │
│         │                                               │  │
│         └────────> board_press                          │  │
└─────────────────────────────────────────────────────────┼──┘
                                                          │
┌─────────────────────────────────────────────────────────▼──┐
│                    STRATEGIC LAYER                         │
│    card_adv    tempo    life_buffer    removal_avail       │
└─────────────────────────────────────────────────────────┬──┘
                                                          │
┌─────────────────────────────────────────────────────────▼──┐
│                    OUTCOME LAYER                           │
│                       win_prob                             │
└────────────────────────────────────────────────────────────┘
```

### Causal Variables

Operational definitions (the env publishes these exact quantities via
`mtg.env.reward.RewardCalculator.get_causal_variable_values`):

| Variable | Layer | Description |
|----------|-------|-------------|
| `mana_t` | Resource | Number of mana-producing permanents controlled |
| `mana_t1` | Resource | SCM prediction of mana next turn (`mana_t + land_drop + mana_creatures`) |
| `card_count` | Resource | Cards in hand |
| `land_drop` | Resource | 1 if at least one land is in hand (drop available), else 0 |
| `mana_spent` | Resource | Number of non-land permanents controlled (proxy for committed mana) |
| `board_presence` | Board State | Number of permanents controlled |
| `board_press` | Board State | Net creature power (own minus opponent) |
| `threat_density` | Board State | Fraction of threats vs total permanents |
| `card_adv` | Strategic | Card advantage relative to opponent |
| `tempo` | Strategic | Mana efficiency differential, clipped to `[-1, 1]` |
| `life_buffer` | Strategic | Own life minus opponent's life |
| `removal_avail` | Strategic | 1 if at least one removal spell is in hand, else 0 |
| `win_prob` | Outcome | Estimated win probability under the SCM logistic |

---

## Action Space

The environment has **478 discrete actions** with legal action masking. A
typical state exposes only 2 to 15 legal actions, including instant-speed
spells during opponent priority windows.

### Core Actions

| Action ID | Action | Speed | Description |
|-----------|--------|-------|-------------|
| 0 | `PASS` | n/a | Pass priority to opponent |
| 1 | `KEEP_HAND` | n/a | Keep current hand (mulligan phase) |
| 2 | `MULLIGAN` | n/a | Mulligan to fewer cards |

### Sorcery-Speed Actions (Your Main Phase Only)

| Action ID | Action | Description |
|-----------|--------|-------------|
| 3-5 | `PLAY_LAND[0-2]` | Play a land from hand |
| 6-10 | `CAST_SORCERY[0-4]` | Cast creature, sorcery, or enchantment |

### Instant-Speed Actions (Any Priority Window)

| Action ID | Action | Description |
|-----------|--------|-------------|
| 11-15 | `CAST_INSTANT[0-4]` | Cast instant or flash creature |

### Combat Actions

| Action ID | Action | Description |
|-----------|--------|-------------|
| 16-35 | `ATTACK_TOGGLE[0-19]` | Toggle creature as attacker (+ CONFIRM) |
| 36-55 | `BLOCK_SELECT_ATTACKER[0-19]` | Select attacker to assign blocker |
| 56-75 | `BLOCK_SELECT_BLOCKER[0-19]` | Assign blocker to selected attacker |

### Priority System

The environment implements a **simplified priority system** inspired by official MTG rules:

- **Priority Windows**: Upkeep, Main 1, Combat, Blockers, Main 2, End Step
- **Instant-Speed Casting**: Cast instants during any priority phase, including the opponent's turn
- **Response Windows**: After attackers are declared, defend with instants or blockers
- **Stack Resolution**: Spells resolve in LIFO order (simplified for the benchmark)

```python
# Example: Casting an instant on opponent's turn
obs, info = env.step(action)

# Check if we have priority during opponent's combat
if info["priority_player"] == "Player" and info["phase"] == "Attackers":
    # Cast an instant in response
    instant_action = 11  # CAST_INSTANT_0
    if info["action_mask"][instant_action]:
        obs, reward, done, trunc, info = env.step(instant_action)
```

**Action masking:** only legal actions are available. Read the current binary
mask from `info["action_mask"]`.

---

## Observation Space

The observation is a **3077-dimensional** continuous vector built from fixed-size card embeddings:

| Component | Shape | Description |
|-----------|-------|-------------|
| Game state | 17 | Life totals, turn/phase, mana, hand sizes, attacker/blocker counts, stack, declared attackers |
| Own hand | 10 × 34 | Card embeddings for hand (padded to max 10) |
| Own battlefield | 20 × 34 | Card embeddings for own permanents |
| Opponent battlefield | 20 × 34 | Card embeddings for opponent's visible permanents |
| Own graveyard | 20 × 34 | Card embeddings for own graveyard |
| Opponent graveyard | 20 × 34 | Card embeddings for opponent's graveyard |

**Card embedding (34 features):** type, mana cost, effective power/toughness,
keyword abilities, card effects, tapped/summoning-sick state, and color pips.

**Partial Observability**: The opponent's hand is hidden; only battlefield and public information is visible.

---

## Environment Configuration

```python
env = MTGEnv(
    deck_archetype="mono_red_aggro",
    opponent_archetype="azorius_control",
    max_turns=20,
    reward_type="shaped",          # "sparse", "shaped", or "dense"
    render_mode="ansi",            # None, "ansi", or "human"
    seed=42,
    auto_combat=False,             # False = agent selects individual attackers
    auto_target=False,             # False = agent picks spell targets
    auto_mana=True,                # True = mana payment is always automatic
)
```

### Reward Types

| Type | Description |
|------|-------------|
| `sparse` | +1 for win, -1 for loss, 0 otherwise |
| `shaped` | Intermediate rewards based on causal variables |
| `dense` | Per-action rewards based on game state changes |

---

## Deck Archetypes

| Archetype | Strategy | Tier | Meta Share |
|-----------|----------|------|------------|
| **Mono-Red Aggro** | Fast damage with haste creatures | 1 | ~15% |
| **Azorius Control** | Counterspells and board wipes | 1 | ~12% |
| **Dimir Midrange** | Disruption with efficient threats | 1 | ~11% |
| **Domain Ramp** | Mana acceleration to powerful spells | 2 | ~8% |
| **Boros Convoke** | Token synergies with convoke | 2 | ~7% |

---

## Agents

The benchmark includes multiple baseline agents spanning heuristic and RL:

| Agent | Type | Description |
|-------|------|-------------|
| `RandomAgent` | Baseline | Uniform random over legal actions |
| `GreedyAggroAgent` | Heuristic | Aggressive, tempo-focused strategy |
| `ConvokeAggroAgent` | Heuristic | Aggro tuned for Boros Convoke |
| `ControlAgent` | Heuristic | Counterspell/removal prioritization |
| `MidrangeAgent` | Heuristic | Threat-first, removal-second balance |
| `RampAgent` | Heuristic | Ramp into high-impact threats |
| `PPOAgent` | Model-Free RL | Proximal Policy Optimization with action masking |
| `CausalAgent` | Causal RL | PPO with a learned CausalWorldModel (CWM) for auxiliary causal transition prediction and SCM-informed action selection |
| `CGFAAgent` (`cgfa`) | Causal RL | **CGFA-PPO**: PPO with SCM-factored value heads, a residual advantage gate, and an intervention-calibration loss. See [Causal Graph-Factored Advantage (CGFA-PPO)](#causal-graph-factored-advantage-cgfa-ppo). |

### Causal Agent Features

The `CausalAgent` combines PPO with a learned CausalWorldModel (CWM). The CWM
learns causal transition dynamics during training and informs action selection
alongside SCM-based counterfactual scoring, planning, and online weight updates.
When rollouts are too sparse for reliable CWM estimates, the agent falls back to
SCM-based reasoning:

```python
from mtg.agents import CausalAgent

agent = CausalAgent(
    observation_dim=3077,
    action_dim=478,
    causal_weight=0.6,              # 0=pure RL, 1=pure causal
    exploration_rate=0.1,           # Annealed to 0.01 over training
    exploration_rate_end=0.01,
    exploration_anneal_steps=50_000,
    planning_depth=3,               # Multi-step SCM lookahead
    counterfactual_samples=16,      # Stochastic counterfactual perturbations
    log_decisions=True,             # Enable decision logging for analysis
    seed=42,
)

# After running episodes, analyze causal reasoning
stats = agent.get_causal_stats()
print(f"Decisions: {stats['total_decisions']}")
print(f"Avg causal effect: {stats['avg_causal_effect']:.3f}")
print(f"Causal preferred ratio: {stats['causal_preferred_ratio']:.2%}")

# Save the decision log for downstream analysis
agent.save_decision_log("results/causal_decisions.json")
```

### Causal Graph-Factored Advantage (CGFA-PPO)

`CGFAAgent` is a drop-in PPO replacement that uses the SCM as a
credit-assignment scaffold rather than only a prior on observations. The key
building blocks are:

| Component | What it does | Where it lives |
|-----------|--------------|----------------|
| Factorised value head | Predicts a per-factor value `V_k(s)` for every SCM outcome-layer parent in addition to PPO's scalar baseline `V(s)`. The scalar critic still drives the policy gradient baseline; `V_k` is used by the per-factor GAE that feeds the gated advantage. | `mtg/agents/reinforcement_learning/cgfa/policy.py` |
| Per-factor reward channels | Adds an independent per-factor reward stream `r_k = phi_k(s_{t+1}) - phi_k(s_t)` where `phi_k` projects the state onto SCM variable `k`. These channels run alongside (not as a partition of) the scalar shaped reward. | `mtg/agents/reinforcement_learning/cgfa/wrapper.py` |
| Per-factor advantages | Computes `A_k = G_k - V_k` with GAE on each factor's reward stream. | `mtg/agents/reinforcement_learning/cgfa/ppo.py` |
| State-conditional residual gate | A small head `g(s) in (0, 1)` blends the scalar PPO advantage with the SCM-weighted sum of per-factor advantages, `A_used = (1 - g(s)) * A_scalar + g(s) * sum_k w_k * A_k`, so the model can fall back to vanilla PPO whenever the SCM decomposition is unhelpful. | `mtg/agents/reinforcement_learning/cgfa/policy.py` |
| Intervention calibration | Auxiliary loss `L_cal = -PearsonCorr(A_k, eps_k)` where `eps_k` is the SCM's structural residual; encourages each `A_k` to track the unexplained component of factor `k`. | `mtg/agents/reinforcement_learning/cgfa/ppo.py` |

```python
from mtg.agents import CGFAAgent

agent = CGFAAgent(
    observation_dim=3077,
    action_dim=478,
    cgfa_alpha=1.0,                       # weight on SCM-factored advantage
    learnable_gate=True,                  # toggle off for the cgfa_no_gate ablation
    intervention_calibration_coef=0.1,    # set to 0.0 for the cgfa_no_cal ablation
    seed=42,
)
```

CGFA-PPO logs `cgfa/calibration_pearson_*`, `cgfa/credit_share_*`, and
`cgfa/gate_mean` during training; these are dumped to
`logs/<exp>/cgfa/cgfa_calibration.csv` by the bundled
`CGFACalibrationCallback` for downstream visualisation (see
[`mtg-research calibration-plot`](#5-the-cgfa-benchmark-pipeline)) and the
case-study tool.

### Agent Structure (Folders)

Agents are grouped by type under `mtg/agents/`:

```
mtg/agents/
├── base/                   # BaseAgent + AgentRegistry
├── heuristics/             # Heuristic baselines
├── reinforcement_learning/ # RL baselines (PPO)
└── causal/                 # Causal agents
```

### Using the Agent Registry

```python
from mtg.agents import list_agents, get_agent, register_agent

# List available agents
print(list_agents())
# ['random', 'greedy_aggro', 'control', 'midrange', 'ramp',
#  'convoke_aggro', 'ppo', 'causal', 'cgfa']

# Create an agent by name
agent = get_agent("greedy_aggro", aggression=0.8, seed=42)

# Register your own agent (see Extension section below)
register_agent("my_agent", MyAgentClass)
```

---

## Workflows

MTG-Causal-RL exposes five CLI workflows. All deck/opponent arguments accept
`mono_red_aggro`, `azorius_control`, `dimir_midrange`, `domain_ramp`, and
`boros_convoke`. Trainable agents are `ppo`, `causal`, and `cgfa`; heuristic
agents are `random`, `greedy_aggro`, `control`, `midrange`, `ramp`, and
`convoke_aggro`.

| # | Workflow | When to use |
|---|----------|-------------|
| 1 | [Training](#1-training-mtg-train) | Train one agent (interactive or CLI) |
| 2 | [Evaluation](#2-evaluation-mtg-eval) | Evaluate trained or baseline agents |
| 3 | [Gameplay Demo](#3-gameplay-demo-mtg-gameplay) | Watch sample games and produce HTML replays |
| 4 | [Research Pipeline](#4-research-pipeline-mtg-research) | Multi-seed sweeps + paired-bootstrap statistical comparisons |
| 5 | [CGFA Benchmark Pipeline](#5-the-cgfa-benchmark-pipeline) | Ablation, transfer, calibration, and case-study diagnostics |

---

### 1. Training (`mtg-train`)

Train PPO or Causal RL agents against heuristic opponents.

```bash
uv run mtg-train [OPTIONS]
# or
uv run python scripts/runner/run_training.py [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-i` / `--interactive` | flag | `False` | Interactive mode with guided prompts |
| `--agent` | `str` | `ppo` | Agent type: `ppo` or `causal` |
| `--deck` | `str` | `mono_red_aggro` | Player deck archetype |
| `--opponent` | `str` | `all` | Opponent(s): `all`, or a specific archetype |
| `--timesteps` | `int` | `1000000` | Total training timesteps |
| `--reward` | `str` | `shaped` | Reward type: `sparse`, `shaped`, `dense` |
| `--seed` | `int` | `42` | Random seed |
| `--max-turns` | `int` | `20` | Max MTG turns per game |
| `--n-envs` | `str` | `4` | Parallel environments for rollout collection (`auto` = all CPU cores, `1` = no parallelism) |
| `--training-mode` | `str` | `round-robin` | Multi-opponent mode: `round-robin` or `sequential` |
| `--agency` | `str` | `auto` | Agent decision agency: `auto` (simplified), `full` (selective combat + targeting, needs 3M+), `curriculum` (auto 70% → full 30%, designed for Causal RL agent) |
| `--output` | `str` | `results/trained_agents` | Output directory |
| `--eval-episodes` | `int` | `100` | Post-training evaluation episodes per opponent |
| `--sample-games` | `int` | `3` | Sample games to record as HTML reports |
| `--sample-opponents` | `str` | `""` | Opponents for sample games (comma-separated; default: training opponents) |

**Budget semantics:** `--timesteps` is per opponent. In `sequential` mode each
opponent receives the full budget independently. In `round-robin` mode
(default), total training is scaled to `timesteps × n_opponents` so each
opponent gets equal exposure.

**Output artifacts** (saved to `{output}/{run_name}/`):

| Artifact | Path |
|----------|------|
| Saved model | `{agent}_{deck}.zip` |
| Config | `config.yaml` |
| Metrics | `metrics.json` |
| Training curves plot | `plots/training_curves.png` |
| Evaluation results plot | `plots/evaluation_results.png` |
| Sample game replays | `reports/game_{n}_{opponent}/replay.html` + `replay.json` |

**Examples:**

```bash
# Train PPO against a single opponent (1M steps, round-robin irrelevant)
uv run mtg-train --agent ppo --deck mono_red_aggro --opponent azorius_control

# Train causal agent against all opponents (round-robin, 2M steps)
uv run mtg-train --agent causal --opponent all --timesteps 2000000

# Sequential training for ablation (full budget per opponent)
uv run mtg-train --opponent all --training-mode sequential --timesteps 500000

# Curriculum: auto first, then fine-tune with full agency (designed for Causal RL agent)
uv run mtg-train --agent causal --agency curriculum --timesteps 2000000

# Full agency from scratch (needs large budget)
uv run mtg-train --agency full --timesteps 5000000

# Interactive mode with guided prompts
uv run mtg-train --interactive
```

#### Deck-Specific Agents

Agents specialize to one player deck and one opponent pool.

| Aspect | Behavior |
|--------|----------|
| **Deck specialization** | The agent learns to play one deck optimally |
| **Opponent adaptation** | The agent learns patterns against one opponent archetype |
| **Observation space** | Tied to the specific deck's cards and strategies |

This keeps training tractable while still supporting transfer experiments
across unseen opponent archetypes.

---

### 2. Evaluation (`mtg-eval`)

Evaluate one or all agents across matchups and seeds with statistical aggregation.

```bash
uv run mtg-eval [OPTIONS]
# or
uv run python scripts/runner/run_evaluation.py [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-i` / `--interactive` | flag | `False` | Interactive mode |
| `--demo` | flag | `False` | Show sample results table (no actual games) |
| `--agent` | `str` | `all` | Agent to evaluate (`all` for every registered agent) |
| `--model-path` | `str` | `None` | Path to trained `.zip` model (for `ppo`/`causal`) |
| `--deck` | `str` | `mono_red_aggro` | Player deck archetype |
| `--opponent` | `str` | `azorius_control` | Opponent(s): `all`, or comma-separated names |
| `--episodes` | `int` | `500` | Total episodes per opponent (split across seeds) |
| `--max-turns` | `int` | `10` | Max MTG turns (should match training) |
| `--seeds` | `int` (multiple) | `42 123 456 789 1000` | Random seeds for statistical robustness |
| `--output` | `str` | `results/evaluations` | Output directory |
| `--verbose` | flag | `False` | Live game state visualization |
| `--show-games` | `int` | `0` | Games to visualize in detail |
| `--show-games-opponents` | `str` | `""` | Opponents for visualized games (comma-separated) |
| `--save-reports` | flag | `False` | Save HTML gameplay reports |
| `--speed` | `str` | `fast` | Visualization speed: `slow` (5s), `medium` (3s), `fast` (1s) |

**Budget semantics:** `--episodes` is per opponent and split only across seeds
(`episodes_per_seed = episodes / num_seeds`).

**Output artifacts** (saved to `{output}/{run_name}/`):

| Artifact | Path |
|----------|------|
| Results JSON | `results.json` |
| Win rate comparison | `plots/win_rate_comparison.png` |
| Reward comparison | `plots/reward_comparison.png` |
| Game replays | `reports/game_{n}_{opponent}/replay.html` + `replay.json` |

**Examples:**

```bash
# Evaluate all agents against all opponents
uv run mtg-eval --agent all --opponent all --episodes 500

# Evaluate a trained PPO model with HTML reports
uv run mtg-eval --agent ppo --model-path results/trained_agents/run/ppo_mono_red_aggro.zip \
    --opponent all --show-games 3 --save-reports

# Quick demo with sample results (no games run)
uv run mtg-eval --demo

# Test generalization to a new opponent
uv run mtg-eval --agent ppo --model-path results/trained_agents/run/ppo_mono_red_aggro.zip \
    --opponent dimir_midrange
```

---

### 3. Gameplay Demo (`mtg-gameplay`)

Play or watch a single interactive game with Rich terminal visualization.

```bash
uv run mtg-gameplay
# or
uv run python scripts/runner/run_gameplay.py
```

This workflow is fully interactive (no CLI flags). It prompts for:

| Setting | Options | Default |
|---------|---------|---------|
| Mode | Demo (preset decks/agents) or Custom (pick everything) | Demo |
| Speed | `slow`, `medium`, `fast`, `instant` | `fast` |
| Turns | 1 to 10 | 5 |

**Output artifacts:**

| Artifact | Path |
|----------|------|
| HTML replay | `results/gameplay/{agent}_{timestamp}/{player_deck}_vs_{opponent_deck}.html` |

**Features:**
- Rich CLI visualization with phase-by-phase display
- Animated turn progression through all 7 MTG phases
- HTML replay report generation
- Any agent type vs any agent type, any deck matchup

---

### 4. Research Pipeline (`mtg-research`)

`mtg-research` is the resumable benchmark pipeline. Use it for multi-seed
training sweeps, high-fidelity evaluation, statistical aggregation, and figures.
Each stage writes machine-readable artifacts, so later stages can be re-run
without repeating training.

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  mtg-research    │ ─► │  mtg-research    │ ─► │  mtg-research    │
│  train           │    │  eval            │    │  aggregate       │
│  (agent x deck   │    │  (per-episode    │    │  (figures, LaTeX │
│   x seed runs)   │    │   metrics + CSV) │    │   tables, tests) │
└──────────────────┘    └──────────────────┘    └──────────────────┘
        writes                  writes                  writes
   sweep_manifest.yaml      eval_results.json       figures/, tables/,
   <run-dirs>/              eval_episodes.csv       aggregated_results.json
                            eval_summary.csv
```

| Command | What it does |
|---------|-------------|
| `mtg-research -i` | Interactive wizard that configures and runs the full pipeline |
| `mtg-research pipeline ...` | Runs Stages 1+2+3 end-to-end from CLI flags |
| `mtg-research train ...` | Stage 1 only (multi-seed training sweep) |
| `mtg-research eval ...` | Stage 2 only (high-fidelity evaluation + baselines) |
| `mtg-research aggregate ...` | Stage 3 only (figures + paired-bootstrap tests) |
| `mtg-research ablation ...` | End-to-end 6-point CGFA ablation suite (`ppo`, `causal`, `cgfa_scalar_only`, `cgfa_no_gate`, `cgfa_no_cal`, `cgfa_full`) |
| `mtg-research transfer ...` | Train on K opponents, evaluate on held-out opponents, and report the generalisation gap |
| `mtg-research calibration-plot ...` | Render the CGFA intervention-calibration diagnostic figure from one or more `cgfa_calibration.csv` files |
| `mtg-research case-study ...` | Run a deterministic CGFA episode and dump a per-step factor-attribution table + 3-panel "why-did-the-agent-do-that" figure |
| `mtg-research --help` | List subcommands |

#### Interactive Mode (`mtg-research -i`)

The easiest entry point: it walks you through every decision and then executes
the full pipeline:

```bash
uv run mtg-research -i
```

The wizard prompts for:

| # | Step | Notes |
|---|------|-------|
| 1 | Mode | `single method`, `A/B comparison` (matched seeds), or `quick smoke test` |
| 2 | Experiment name | Timestamped default |
| 3 | Agent(s) | Locked to PPO+Causal in A/B mode; otherwise pick one |
| 4 | Player decks | Multi-select from the 5 archetypes |
| 5 | Opponents | Multi-select (default = all 5) |
| 6 | Seeds | `42 123 456` default; 3+ recommended for paired-bootstrap |
| 7 | Training budget | Timesteps per opponent |
| 8 | Training mode | `round-robin` (default) or `sequential` |
| 9 | Agency | `auto` (recommended for PPO), `curriculum` (for Causal), or `full` |
| 10 | Parallel envs | `0`/`auto` = all CPU cores - 1 |
| 11 | Eval episodes | 500 default (keeps CIs at ±5% or tighter) |
| 12 | Baselines | Yes/No; when Yes (default) each player deck is auto-paired with `random` + its canonical heuristic (e.g. `azorius_control` -> `random` + `control`). Use the advanced override only for special ablations. |

In **A/B comparison** mode the wizard trains PPO and Causal at the **same
seeds**, evaluates both, and then runs a paired-bootstrap cross-aggregation
automatically (produces `tables/significance.tex` with p-values per cell).

#### End-to-end example (non-interactive)

Scripted equivalent of the A/B wizard:

```bash
# PPO baseline end-to-end (train + eval + aggregate)
# Baselines auto-pair: mono_red_aggro -> random + greedy_aggro
uv run mtg-research pipeline \
    --experiment-name ppo_baseline_v1 \
    --agents ppo --player-decks mono_red_aggro \
    --seeds 42 123 456 \
    --timesteps-per-opponent 2000000 \
    --agency auto \
    --eval-episodes 500

# Causal RL end-to-end at the SAME seeds (for paired comparison)
uv run mtg-research pipeline \
    --experiment-name causal_v1 \
    --agents causal --player-decks mono_red_aggro \
    --seeds 42 123 456 \
    --timesteps-per-opponent 2000000 \
    --agency curriculum \
    --eval-episodes 500 --no-baselines  # skip duplicate baseline runs

# Cross-sweep aggregation with paired-bootstrap significance tests
uv run mtg-research aggregate \
    --eval-results \
        results/research/ppo_baseline_v1/eval/eval_results.json \
        results/research/causal_v1/eval/eval_results.json \
    --source-labels ppo causal \
    --baseline-agent ppo \
    --output-dir results/research/comparison_ppo_vs_causal
```

You can also run each stage individually with `mtg-research train`,
`mtg-research eval`, and `mtg-research aggregate`. The per-stage options are
documented below.

#### Stage 1: `mtg-research train`

Sweep over `(agent x player_deck x seed)`. Resumable: re-running skips any
combination whose model already exists (matched by reading each subdir's
`config.yaml`).

| Option | Default | Description |
|--------|---------|-------------|
| `--experiment-name` | timestamp | Sub-directory name under `--output-root` |
| `--agents` | `ppo` | One or more of: `ppo`, `causal` |
| `--player-decks` | `mono_red_aggro` | One or more deck archetypes |
| `--seeds` | `42 123 456` | Random seeds (one trained model per seed) |
| `--opponents` | all 5 | Opponent decks for round-robin training |
| `--timesteps-per-opponent` | `2000000` | Per-opponent training budget |
| `--training-mode` | `round-robin` | `round-robin` or `sequential` |
| `--agency` | `auto` | `auto`, `full`, or `curriculum` (curriculum is designed for the Causal RL agent) |
| `--n-envs` | `auto` | Parallel environments (`auto` = CPU-1) |
| `--output-root` | `results/research` | Parent directory for the experiment |
| `--force` | `False` | Re-train even if a matching model exists |
| `--quick` | `False` | ~3-min smoke test |

#### Stage 2: `mtg-research eval`

Re-evaluate every model in a sweep at high episode count and optionally
include heuristic/random baselines. Writes per-episode CSV for downstream
pandas analysis.

| Option | Default | Description |
|--------|---------|-------------|
| positional `experiment_dir` | required | Path to a sweep directory |
| `--eval-episodes` | `500` | Episodes per (model, opponent) cell |
| `--no-baselines` | off | Skip baseline evaluation (only score trained models) |
| `--baseline-agents` | unset (auto) | **Advanced.** Explicit list of baseline agent names applied to every deck. When omitted (default), each player deck is auto-paired with `random` + the canonical heuristic from `mtg.agents.DECK_TO_HEURISTIC` (e.g. `azorius_control` -> `random` + `control`). |
| `--extra-player-decks` | none | Decks to evaluate baselines on, beyond what was trained |
| `--max-turns` | `20` | Match what was used at training time |
| `--agency` | `auto` | Match the agency used at training time |

Outputs (under `<experiment_dir>/eval/`):

| Artifact | Description |
|----------|-------------|
| `eval_results.json` | Aggregated per (agent, deck, seed, opponent) summary |
| `eval_summary.csv`  | One row per cell with mean/CI |
| `eval_episodes.csv` | One row per episode (wide format) |
| `eval_manifest.yaml`| Config and timing of the evaluation |

#### Stage 3: `mtg-research aggregate`

Aggregate one or more `eval_results.json` files into polished figures,
LaTeX tables, and **paired-bootstrap significance tests** across seeds.

| Option | Default | Description |
|--------|---------|-------------|
| `--eval-results` | required | One or more paths to `eval_results.json` |
| `--output-dir`   | required | Where to write figures and tables |
| `--baseline-agent` | none | Agent treated as the comparison baseline (e.g. `ppo`) |
| `--source-labels` | derived | Optional human-readable label per source |

Outputs (under `--output-dir`):

| Artifact | Description |
|----------|-------------|
| `figures/win_rate_by_opponent.png` | Grouped bars: opponent x source, mean ± 95% CI across seeds |
| `figures/headline_comparison.png`  | One bar per (source, agent, deck) |
| `figures/per_matchup_heatmap.png`  | (agent, deck) x opponent matrix |
| `tables/headline.tex`              | LaTeX booktabs table (mean ± SEM) |
| `tables/significance.tex`          | Paired-bootstrap CI + p-value per cell |
| `aggregated_results.json`          | Machine-readable everything |

#### Statistical primitives (`stats.py`)

Reusable building blocks for any custom analysis. Importable from any script:

```python
from scripts.research.stats import (
    wilson_ci,             # CI on a proportion (e.g. win rate)
    bootstrap_mean_ci,     # percentile bootstrap CI on the mean
    paired_bootstrap_test, # paired diff CI + p-value (preferred for seed-paired)
    welch_ttest,           # unpaired, unequal-variance t-test
    wilcoxon_signed_rank,  # non-parametric paired test
    holm_bonferroni,       # multiple-testing correction
)
```

#### Recommended evaluation protocol

1. Use at least 3 seeds per `(agent, deck)` cell; use 5+ when reporting
   bootstrap p-values.
2. Use the same seeds across compared sweeps so paired tests are valid.
3. Include random and heuristic baselines unless running a targeted ablation.
4. Use paired-bootstrap tests for "X beats Y" claims. The pipeline reports
   paired CIs and p-values in `tables/significance.tex`.
5. Increase `--eval-episodes` for contested cells where the CI width is the
   limiting factor.

#### Smoke test (~3 minutes)

The fastest way to verify the full pipeline works end-to-end. Interactive
wizard option 3 does this; the CLI equivalent is:

```bash
uv run mtg-research train --quick
uv run mtg-research eval results/research/smoke_test \
    --eval-episodes 10 --baseline-agents random   # tiny override for speed
uv run mtg-research aggregate \
    --eval-results results/research/smoke_test/eval/eval_results.json \
    --output-dir results/research/smoke_test/aggregated --baseline-agent ppo
```

---

### 5. The CGFA Benchmark Pipeline

CGFA-PPO adds four benchmark diagnostics beyond train/eval/aggregate:
ablation, transfer, calibration, and case study. Each diagnostic is available
as a standalone `mtg-research` subcommand.

| Subcommand | What it produces | Typical use |
|------------|------------------|-------------|
| `mtg-research ablation` | One train/eval sub-sweep per variant plus cross-variant figures and tables | Which CGFA component matters? |
| `mtg-research transfer` | In-distribution and held-out evaluation plus a paired generalisation-gap report | Does the agent transfer to unseen opponents? |
| `mtg-research calibration-plot` | Per-factor correlation, credit-share, and gate trajectories | Is the calibration objective engaged? |
| `mtg-research case-study` | Per-step factor attribution CSV plus a 3-panel diagnostic figure | Why did the agent choose an action? |

#### 5.1 6-Point Ablation Suite (`mtg-research ablation`)

Trains and evaluates the canonical six variants:
`ppo` (vanilla MaskablePPO), `causal` (PPO + CWM), `cgfa_scalar_only`
(architecture-matched scalar PPO with all CGFA losses pinned to zero,
isolating the effect of extra parameters), `cgfa_no_gate` (gate frozen
at 1), `cgfa_no_cal` (intervention-calibration coefficient set to 0),
and `cgfa_full` (everything on). The variant definitions live in
[`mtg/experiments/ablation.py`](mtg/experiments/ablation.py) and are
mirrored in [`mtg/experiments/ablations.yaml`](mtg/experiments/ablations.yaml).
For robustness checks, `--variants stress` adds `cgfa_no_scm_init` and
`cgfa_interventional_cal` without changing the canonical six-point suite.

```bash
# Full suite, 3 seeds, 1M steps/opponent, 300-500 eval episodes/cell
uv run mtg-research ablation \
    --experiment-name cgfa_ablation_v1 \
    --player-decks mono_red_aggro \
    --seeds 42 123 456 \
    --opponents mono_red_aggro azorius_control dimir_midrange \
    --timesteps-per-opponent 1000000 \
    --eval-episodes 300

# Optional stress suite for robustness checks
uv run mtg-research ablation --variants stress ...

# Subset (e.g. just PPO vs CGFA-full)
uv run mtg-research ablation --variants ppo cgfa_full ...
```

#### 5.2 Transfer Experiment (`mtg-research transfer`)

Trains every (agent, deck, seed) on a **training opponent set** and
then evaluates the same models twice: once against the training set
(`<exp>/eval/`) and once against a disjoint **held-out opponent set**
(`<exp>/eval_heldout/`). The transfer report pairs the (deck, seed)
cells across the two splits and computes a per-agent generalisation
gap with a paired-bootstrap CI:

```bash
uv run mtg-research transfer \
    --experiment-name cgfa_transfer_v1 \
    --agents ppo cgfa \
    --player-decks mono_red_aggro \
    --seeds 42 123 456 \
    --train-opponents mono_red_aggro azorius_control dimir_midrange \
    --heldout-opponents domain_ramp boros_convoke \
    --timesteps-per-opponent 1000000 \
    --eval-episodes 500
```

Outputs land under `<exp>/transfer/`:

| Artefact | Description |
|----------|-------------|
| `transfer_report.json`     | Per-agent in-dist + held-out means, gap, paired-bootstrap CI / p-value, plus per (deck, seed) pairs |
| `transfer_summary.csv`     | Long-form one row per (agent, deck, seed, opponent, split) cell |
| `transfer_per_opponent.csv`| Per held-out opponent x agent mean win rate + 95% bootstrap CI |
| `figures/transfer_gap.png` | 2-panel figure: in-dist vs held-out bars + signed gap with CI |

A `--smoke` flag runs the entire pipeline (training + 2 eval passes +
report) end-to-end on a tiny budget for CI / regression testing.

#### 5.3 Calibration Plot (`mtg-research calibration-plot`)

Renders a 3-panel diagnostic PNG from one or more `cgfa_calibration.csv`
files emitted during CGFA training:

```bash
uv run mtg-research calibration-plot \
    results/trained_agents/<cgfa_run>/cgfa/cgfa_calibration.csv \
    --output results/figures/cgfa_calibration.png
```

#### 5.4 Case Study (`mtg-research case-study`)

Runs a deterministic CGFA rollout and emits per-step value, advantage, and
calibration diagnostics:

```bash
uv run mtg-research case-study \
    --model results/trained_agents/<cgfa_run>/best_model.zip \
    --player-deck mono_red_aggro --opponent-deck azorius_control \
    --episode-seed 7 \
    --csv results/case_study/episode7.csv \
    --figure results/case_study/episode7.png
```

---

### Replotting figures from disk

Figures are regenerable from persisted JSON or CSV. Use these commands to
restyle plots without re-running training or evaluation.

| Figure family | Source of truth on disk | Replot command |
|---|---|---|
| Training curves (`training_curves.png`) + post-train eval (`evaluation_results.png`) | `<run>/metrics.json` (training run) | `uv run python -m scripts.runner.regenerate_plots <run>` |
| Standalone eval (`win_rate_comparison.png` + `reward_comparison.png`) | `<run>/results.json` (eval run) | `uv run python -m scripts.runner.regenerate_plots <run>` (auto-detects `results.json`) |
| Research aggregation (`win_rate_by_opponent.png`, `headline_comparison.png`, `per_matchup_heatmap.png`, headline / significance LaTeX) | one or more `eval/eval_results.json` | `uv run mtg-research aggregate --eval-results <path/to/eval_results.json> --output-dir <new_dir>` |
| CGFA calibration diagnostic (`cgfa_calibration.png`) | `<train_run>/cgfa/cgfa_calibration.csv` | `uv run mtg-research calibration-plot <csv> --output <out.png>` |
| Case study (`case_study.png`) | `<output>/case_study_steps.csv` | `uv run python -m scripts.research.case_study --from-csv <csv> --output-dir <out_dir>` |
| Transfer gap (`transfer_gap.png`) | `<exp>/transfer/transfer_report.json` | `uv run python -m scripts.research.transfer_sweep --from-report <json> [--output <out.png>]` |

Notes:

- `regenerate_plots` auto-detects whether the given directory holds
  `metrics.json` (training run) or `results.json` (eval run) and renders
  the appropriate set of figures. It will render both if both files are
  present.
- The training-curves plot uses smoothed rolling means (raw faded behind
  the smoothed line); the eval plot includes 95% Wilson-CI error bars
  and sample-size annotations. Runs created before per-update history
  was persisted to `metrics.json` will only regenerate the eval bar
  chart.
- `case_study --from-csv` and `transfer_sweep --from-report` skip all
  training, evaluation, env construction, and model loading: they are
  pure load-and-render entry points, so they're safe to run on any
  machine without a GPU.

---

### HTML Gameplay Reports

HTML reports are generated by gameplay, training, and evaluation workflows:
- **Gameplay:** `results/gameplay/...`
- **Training sample reports:** `results/trained_agents/{run_name}/reports/...`
- **Evaluation visualized games:** `results/evaluations/{run_name}/reports/...`

These provide interactive replays of game sessions showing:
- Turn-by-turn action timeline
- Player/opponent life, hand, lands, mana, and board power
- Creatures on the battlefield (with tapped status)
- Graveyard contents (organized by card type)
- Winner and game metadata

```python
from mtg.utils import GameRecorder, generate_html_report

recorder = GameRecorder(
    player_deck="Mono-Red Aggro",
    opponent_deck="Azorius Control",
    player_agent="PPO",
    opponent_agent="Heuristic",
)

recorder.record_action(turn=1, phase="Main 1", player="Player",
                       action_type="CAST", description="Cast Lightning Bolt")

replay = recorder.get_replay()
generate_html_report(replay, Path("results/gameplay/my_game.html"))
```

---

## Reproducing Benchmark Results

All benchmark results are reproduced via the composable research
pipeline (multi-seed sweeps + paired-bootstrap significance tests).
See [Section 4](#4-research-pipeline-mtg-research) for the generic
guide and [Section 5](#5-the-cgfa-benchmark-pipeline) for the
CGFA-specific subcommands.

### One-command reproduction (`mtg-research paper`)

The `paper` subcommand reproduces the benchmark figures and tables in one
resumable workflow: headline comparison, ablation, transfer, calibration,
case study, and cross-source significance aggregation.

Recommended order:

1. [**Smoke recipe**](#smoke-recipe-a-few-hours-sanity-check) — verifies the
   full workflow at a small budget.
2. [**Optional CGFA signal pre-flight**](#optional-cgfa-signal-pre-flight) —
   two targeted PPO-vs-CGFA cells at full per-cell budget.
3. [**Full benchmark reproduction**](#full-benchmark-reproduction) —
   the canonical high-budget protocol for the released benchmark results.

#### Smoke recipe (a few hours, sanity check)

```bash
# Intentionally overrides the full benchmark defaults.
# Use this to confirm the pipeline runs end-to-end on your hardware.
uv run mtg-research paper \
    --experiment-name smoke_$(date +%Y%m%d_%H%M) \
    --agents ppo cgfa \
    --player-decks mono_red_aggro azorius_control \
    --opponents mono_red_aggro azorius_control dimir_midrange \
    --heldout-opponents domain_ramp boros_convoke \
    --transfer-mode fixed \
    --seeds 42 123 456 \
    --timesteps-per-opponent 200000 \
    --eval-episodes 100
```

The smoke recipe is a pipeline and environment sanity check, not a
benchmark-quality result. It uses only 3 seeds, 200K timesteps per
opponent, 100 evaluation episodes, and a fixed transfer split.

#### Optional CGFA signal pre-flight

Before launching the full all-deck benchmark sweep, the dedicated
pre-flight command trains only two high-signal PPO-vs-CGFA cells
with paired seeds, 1M timesteps, 100 eval episodes, and CGFA's
interventional calibration target:

```bash
uv run mtg-research preflight-signal \
    --experiment-name preflight_cgfa_$(date +%Y%m%d_%H%M) \
    --seeds 42 123 456 789 1024 \
    --timesteps-per-opponent 1000000 \
    --eval-episodes 100 \
    --cgfa-calibration-mode interventional
```

Outputs land under `results/research/<EXP>/azorius_vs_dimir/` and
`results/research/<EXP>/monored_mirror/`. Inspect each
`aggregated/aggregated_results.json` and `aggregated/tables/significance.tex`.
The headline rows include within-source `ppo` vs `cgfa` tests, so a
single eval file answers whether CGFA is separating from PPO on that
cell. A practical go/no-go rule: proceed if `azorius_vs_dimir` is
positive by roughly 5 percentage points and not obviously noisy; pause
if both cells are tied or negative after 1M timesteps.

#### Full benchmark reproduction

Use this command to reproduce the high-budget benchmark results. Keep the
defaults unless you are intentionally defining a new benchmark variant.

```bash
# Full benchmark reproduction: every change from the smoke recipe is annotated.
EXP=full_$(date +%Y%m%d_%H%M)
uv run mtg-research paper \
    --experiment-name $EXP \
    --agents ppo cgfa \
    --player-decks mono_red_aggro azorius_control dimir_midrange domain_ramp boros_convoke \
    --ablation-decks mono_red_aggro azorius_control dimir_midrange domain_ramp boros_convoke \
    --transfer-decks mono_red_aggro azorius_control dimir_midrange domain_ramp boros_convoke \
    --opponents mono_red_aggro dimir_midrange domain_ramp \
    --heldout-opponents azorius_control boros_convoke \
    --transfer-mode leave-one-out \
    --seeds 42 123 456 789 1024 2048 4096 \
    --timesteps-per-opponent 1000000 \
    --eval-episodes 100 \
    --cgfa-calibration-mode interventional
```

| Flag (vs smoke recipe) | Why |
|------------------------|-----|
| `--player-decks ...` | Evaluates learned agents on all 5 benchmark archetypes. |
| `--ablation-decks ...`, `--transfer-decks ...` | Keeps ablation and transfer aligned with headline evaluation. |
| `--transfer-mode leave-one-out` | Rotates each opponent into the held-out slot and avoids fixed-split bias. |
| `--seeds ...` (7) | Enables paired-bootstrap p-values (`n_s >= 5`) and improves power on borderline cells. |
| `--timesteps-per-opponent 1000000` | Uses a substantially larger training budget than the smoke recipe. |
| `--eval-episodes 100` | Balances evaluation precision and runtime; re-evaluate contested cells with more episodes if needed. |
| `--cgfa-calibration-mode interventional` | Uses action-conditional SCM intervention targets for CGFA calibration. |
| Train/held-out split | Defines the deck pool that leave-one-out rotates over. |

**Stages** (six, run in order, all under `results/research/$EXP/`):

1. `headline` (Fig. 3 + Tab. 1)
2. `ablation` (Fig. 4 + Tab. 2 + per-variant `eval_results.json`)
3. `transfer` (Fig. 5; **5 LOO folds** under `transfer/loo_<deck>/`)
4. `calibration` (Fig. 6, post-hoc render from CGFA logs)
5. `case-study` (Fig. 7, single deterministic episode)
6. `cross-source` (the **paired-bootstrap roll-up** across every
   `eval_results.json` produced by stages 1-3 — yields the unified
   Holm–Bonferroni-adjusted significance table at
   `results/research/$EXP/cross_source/tables/significance.tex`)

**Protocol interpretation:** use win-rate tables for performance claims and
calibration/case-study artifacts for mechanism claims. The aggregation pipeline
applies Holm-Bonferroni correction separately within each planned comparison
family.

The Mono-Red mirror is retained because it is a hard symmetric stress test for
CGFA. Use per-factor logs and case-study traces to understand failures rather
than removing difficult matchups.

**Optional CGFA pre-flight:** `mtg-research preflight-signal` trains two paired,
resumable PPO-vs-CGFA cells and writes the same within-source significance table
used by the full workflow.

**Pre-launch checklist** (do these once, before launching):

```bash
# 1. Verify the smoke recipe works end-to-end first. A clean smoke run
#    should complete in hours, not days; use the smoke-recipe flags above.
uv run mtg-research paper \
    --experiment-name smoke \
    --agents ppo cgfa \
    --player-decks mono_red_aggro azorius_control \
    --opponents mono_red_aggro azorius_control dimir_midrange \
    --heldout-opponents domain_ramp boros_convoke \
    --transfer-mode fixed \
    --seeds 42 123 456 \
    --timesteps-per-opponent 200000 \
    --eval-episodes 100 \
    --dry-run

# 2. Print the full benchmark plan without launching anything; confirm
#    the 5 LOO folds, the 6 stages, and the cost warnings look right.
uv run mtg-research paper \
    --experiment-name $EXP \
    --agents ppo cgfa \
    --player-decks mono_red_aggro azorius_control dimir_midrange domain_ramp boros_convoke \
    --ablation-decks mono_red_aggro azorius_control dimir_midrange domain_ramp boros_convoke \
    --transfer-decks mono_red_aggro azorius_control dimir_midrange domain_ramp boros_convoke \
    --opponents mono_red_aggro dimir_midrange domain_ramp \
    --heldout-opponents azorius_control boros_convoke \
    --transfer-mode leave-one-out \
    --seeds 42 123 456 789 1024 2048 4096 \
    --timesteps-per-opponent 1000000 \
    --eval-episodes 100 \
    --cgfa-calibration-mode interventional \
    --dry-run

# 3. Make sure you have enough free space on the results disk.
#    The all-deck full benchmark is large; start with 200+ GB free.
#    Each (agent, deck, seed) cell ships periodic checkpoints; the
#    7-seed budget grows the on-disk footprint roughly proportionally.
df -h results/

# 4. Wrap the launch in a terminal multiplexer so an editor crash, a
#    dropped SSH session, or the host sleeping cannot kill it. See
#    "Resuming an interrupted benchmark run" below for canonical tmux
#    and nohup snippets.
```

**Estimated wall-time:** the smoke recipe should finish in hours. The full
benchmark uses all 5 player decks, 7 paired seeds, 1M steps per opponent, and
5 leave-one-out transfer folds, so expect a long unattended run. Runtime depends
heavily on hardware, parallelism, and checkpoint reuse.

**Verifying the run finished cleanly** (run after the `paper` command exits):

```bash
# Every stage should leave at least one canonical artefact on disk.
EXP=<your_experiment_name>
ROOT=results/research/$EXP

# Headline produced its aggregated bundle?
test -f $ROOT/headline/aggregated/aggregated_results.json && echo "headline OK"

# All 6 ablation variants ran 7 seeds each? (7 .zip checkpoints per variant)
for v in ppo causal cgfa_scalar_only cgfa_no_gate cgfa_no_cal cgfa_full; do
  n=$(ls $ROOT/ablation/$v/*/*.zip 2>/dev/null | grep -v checkpoints/ | wc -l | tr -d ' ')
  echo "ablation/$v: $n / 7 trained models"
done

# All 5 LOO folds produced a transfer report?
ls $ROOT/transfer/loo_*/transfer/transfer_report.json | wc -l   # expect 5

# Cross-source aggregate produced the unified significance table?
test -f $ROOT/cross_source/tables/significance.tex && echo "cross_source OK"
```

If any of those checks fail, re-run the **same command** to resume.
The `train_sweep` resume logic skips every `(agent, deck, seed)` cell
that already has a saved checkpoint; `eval_sweep` always re-runs but
is cheap; `aggregate` is cheap and idempotent. See
[Resuming an interrupted benchmark run](#resuming-an-interrupted-benchmark-run)
for deeper details.

#### Exporting figures and tables

After a benchmark run finishes, collect figures, LaTeX tables, and raw numbers
into one portable directory:

```bash
EXP=<your_experiment_name>
uv run python -m scripts.research.export_paper_bundle \
    --experiment-name $EXP \
    --output-dir results/research/$EXP/benchmark_bundle

# Optional for leave-one-out transfer: choose which fold is exposed as
# fig5_transfer_gap.png in the bundle.
uv run python -m scripts.research.export_paper_bundle \
    --experiment-name $EXP \
    --transfer-fold loo_azorius_control \
    --output-dir results/research/$EXP/benchmark_bundle
```

The output directory contains `figures/`, `tables/`, and `raw_numbers/` with
stable filenames. Use it directly in a LaTeX project or for independent checks;
the raw JSON files provide provenance for every generated table value.

The export script emits the following table fragments:

| Bundle file                              | Source / generator |
| ---------------------------------------- | ------------------ |
| `tables/tab1_headline.tex`               | Copied from `cross_source/tables/headline.tex` (per-(source, agent, deck) marginal win rate + 95 % CI). |
| `tables/tab2_ablation.tex`               | Copied from `ablation/aggregated/tables/headline.tex` (per-variant marginal win rate + 95 % CI). |
| `tables/tab3_main_results.tex`           | **Generated** from `headline/aggregated/aggregated_results.json` (or `cross_source/aggregated_results.json` when present): per-deck PPO-vs-CGFA-PPO mean, paired-bootstrap CI, $n_s$, $\Delta$, $p_\text{boot}$, $p_\text{Holm}$. |
| `tables/tab4_transfer_gap.tex`           | **Generated** from `transfer/transfer/transfer_report.json` and any `transfer/loo_*/transfer/transfer_report.json`: per-agent in-dist / held-out / $\Delta$ / 95 % CI / $p_\text{boot}$ / $p_\text{Holm}$. |
| `tables/tabA_reference_anchors.tex`      | Static reference-anchor descriptions (deck-specialised heuristic vs random); not regenerated. |

#### Useful flags

| Flag | Effect |
|------|--------|
| `--dry-run` | Print the planned per-stage commands and exit (no training, no eval). |
| `--only headline ablation` | Run only the listed stages (mutually exclusive with `--skip`). Choices: `headline`, `ablation`, `transfer`, `calibration`, `case-study`, `cross-source`. |
| `--skip transfer` | Skip the listed stages (`headline`, `ablation`, `transfer`, `calibration`, `case-study`, `cross-source`). |
| `--ablation-decks DECK [...]` | Override the player decks for the ablation suite (default: all `--player-decks` entries). |
| `--transfer-decks DECK [...]` | Override the player decks for the transfer experiment (default: all `--player-decks` entries). |
| `--transfer-mode {fixed,leave-one-out}` | `fixed` trains once and evaluates on held-out opponents; `leave-one-out` rotates each opponent into the held-out slot. |
| `--cgfa-calibration-mode {factual,interventional}` | CGFA factor-epsilon target. The full benchmark recipe uses `interventional`, matching the pre-flight command. |
| `--case-study-{player,opponent}-deck DECK` | Decks for the qualitative case-study rollout. |
| `--case-study-seed N` | Episode seed for the case-study rollout (default 7). |
| `--no-baselines` | Skip random + heuristic baselines in the headline + ablation eval passes (faster). |
| `--force` | Re-train models even if a checkpoint already exists on disk. |

The pipeline runs six stages in order: `headline` → `ablation` → `transfer` →
`calibration` → `case-study` → `cross-source`.

#### Resuming an interrupted benchmark run

`mtg-research paper` is fully resumable. Re-run the exact same command and
every completed `(agent, deck, seed)` checkpoint is reused:

```bash
# Resume: identical flags to the original launch.
uv run mtg-research paper \
    --experiment-name $EXP \
    --agents ppo cgfa \
    --player-decks mono_red_aggro azorius_control \
    --opponents mono_red_aggro azorius_control dimir_midrange \
    --heldout-opponents domain_ramp boros_convoke \
    --seeds 42 123 456 \
    --timesteps-per-opponent 200000 \
    --eval-episodes 100
```

The pipeline prints `↻ skipped (model exists) <run-dir>` for reused
checkpoints. Skip complete stages with `--skip`:

```bash
# Headline already done; only redo ablation + transfer + calibration + case study.
uv run mtg-research paper \
    --experiment-name $EXP \
    --skip headline
```

What is and isn't re-done on resume:

| Artefact | Behaviour |
|----------|-----------|
| Trained model with `config.yaml` matching `(agent, deck, seed)` | **Skipped.** Existing run is reused. |
| Interrupted training run with no `config.yaml` yet | **Re-trained from scratch** in a new timestamped sub-dir. The partial dir is harmless garbage you can `rm -rf`. |
| Eval passes (`eval/`, `eval_heldout/`) | **Always re-run.** Cheap and deterministic; keeps numbers consistent across the whole sweep. |
| Aggregates, figures, LaTeX tables | **Always re-run.** Picks up any new evals automatically. |

Resume detection is at `(agent, deck, seed)` granularity, not per timestep. A
partial training run restarts from 0 timesteps; a completed run is skipped.

**Pre-resume check**, to see what's already on disk before resuming:

```bash
# Per-variant seed count (3/3 means all seeds finished training).
EXP=<your_experiment_name>
for v in ppo causal cgfa_scalar_only cgfa_no_gate cgfa_no_cal cgfa_full; do
  done=$(ls -d results/research/$EXP/ablation/$v/*/config.yaml 2>/dev/null | wc -l)
  echo "$v: $done/3 seeds trained"
done

# Or have `paper` print exactly what it plans to do, no execution.
uv run mtg-research paper --experiment-name $EXP --dry-run
```

Use `--force` to deliberately re-train everything (for example after
bumping a hyperparameter and wanting fresh checkpoints).

> **Tip: protect long runs against terminal exits.** Wrap the launch
> in a terminal multiplexer so an editor crash or a closed terminal
> cannot kill the run:
>
> ```bash
> # tmux (macOS: brew install tmux)
> tmux new -s mtg-benchmark
> source .venv/bin/activate
> uv run mtg-research paper --experiment-name $EXP ...
> # detach: Ctrl+b d   |   reattach: tmux attach -t mtg-benchmark
>
> # Or, no install, also keeps macOS from idle-sleeping during the run:
> mkdir -p logs && LOG=logs/run_$(date +%Y%m%d_%H%M).log
> nohup caffeinate -i uv run mtg-research paper --experiment-name $EXP ... > "$LOG" 2>&1 &
> disown && tail -f "$LOG"
> ```

Output layout with `--experiment-name X`:

```
results/research/
└── X/                                     # one master dir per benchmark run
    ├── headline/                          # Stage 1: PPO vs CGFA paired comparison
    │   ├── eval/eval_results.json
    │   └── aggregated/                    # win-rate figures, headline LaTeX tables
    ├── ablation/                          # Stage 2: 6-variant CGFA ablation
    │   ├── ppo/ ... cgfa_full/            # one sub-sweep per variant
    │   ├── aggregated/                    # tables/significance.tex, figures/
    │   ├── figures/cgfa_calibration.png   # Stage 4 (auto-rendered)
    │   └── case_study/                    # Stage 5 (auto-rendered)
    └── transfer/                          # Stage 3: in-dist + held-out evaluation
        └── transfer/transfer_report.json
```

The interactive wizard is also available:

```bash
uv run mtg-research -i
```

Scripted equivalent for the **headline PPO vs Causal vs CGFA** comparison:

```bash
# PPO baseline end-to-end at 3 seeds
# Baselines auto-pair to each player deck (random + canonical heuristic).
uv run mtg-research pipeline \
    --experiment-name ppo_baseline_v1 --agents ppo \
    --seeds 42 123 456 --timesteps-per-opponent 2000000 --agency auto \
    --eval-episodes 500

# Causal RL end-to-end at the SAME seeds
uv run mtg-research pipeline \
    --experiment-name causal_v1 --agents causal \
    --seeds 42 123 456 --timesteps-per-opponent 2000000 --agency curriculum \
    --eval-episodes 500 --no-baselines  # skip duplicate baseline runs

# CGFA-PPO at the SAME seeds
uv run mtg-research pipeline \
    --experiment-name cgfa_v1 --agents cgfa \
    --seeds 42 123 456 --timesteps-per-opponent 2000000 --agency curriculum \
    --eval-episodes 500 --no-baselines

# Cross-sweep aggregation with paired-bootstrap significance tests
uv run mtg-research aggregate \
    --eval-results results/research/ppo_baseline_v1/eval/eval_results.json \
                   results/research/causal_v1/eval/eval_results.json \
                   results/research/cgfa_v1/eval/eval_results.json \
    --source-labels ppo causal cgfa --baseline-agent ppo \
    --output-dir results/research/comparison_ppo_vs_causal_vs_cgfa
```

The one-command workflow stages the CGFA-specific diagnostics automatically.
Run `mtg-research ablation`, `mtg-research transfer`,
`mtg-research calibration-plot`, or `mtg-research case-study` directly when
you only need one artifact family.

### Expected Output Tree

A `mtg-research paper` run produces **one self-contained master
directory**: one benchmark run, one path you can `tar`, `mv`, share,
or `rm -rf`. Files marked **(skip marker)** are what the resume logic
checks to decide whether a `(agent, deck, seed)` cell is already
finished (see [Resuming an interrupted benchmark run](#resuming-an-interrupted-benchmark-run)).

```
results/research/
└── X/                                            # --experiment-name X
    │
    ├── headline/                                # Stage 1   (--skip headline)
    │   ├── sweep_manifest.yaml                  # one row per (agent, deck, seed)
    │   ├── ppo_mono_red_aggro_vs_multi_<ts>/    # one run-dir per row
    │   │   ├── config.yaml                      # ← skip marker (parsed for agent/deck/seed)
    │   │   ├── ppo_mono_red_aggro.zip           # ← skip marker (final model: <agent>_<deck>.zip)
    │   │   ├── metrics.json
    │   │   ├── checkpoints/                     # periodic checkpoints during training
    │   │   │   └── checkpoint_<steps>.zip       # one per checkpoint interval
    │   │   ├── plots/                           # per-run training + evaluation curves
    │   │   │   ├── training_curves.png
    │   │   │   └── evaluation_results.png
    │   │   └── cgfa/cgfa_calibration.csv        # CGFA only: per-update calibration log
    │   ├── cgfa_mono_red_aggro_vs_multi_<ts>/   # ... (one per other agent, deck, seed)
    │   ├── eval/                                # in-distribution evaluation (always re-run)
    │   │   ├── eval_results.json                # per-(agent, deck, seed, opp) summaries + baselines
    │   │   ├── eval_summary.csv
    │   │   ├── eval_episodes.csv                # per-episode records
    │   │   └── eval_manifest.yaml
    │   └── aggregated/                          # paired-bootstrap figures + LaTeX tables
    │       ├── figures/
    │       │   ├── win_rate_by_opponent.png
    │       │   ├── headline_comparison.png      # Fig. 3
    │       │   └── per_matchup_heatmap.png
    │       ├── tables/headline.tex              # per-(source, agent, deck) headline table
    │       ├── tables/significance.tex          # includes within-source PPO-vs-CGFA rows
    │       └── aggregated_results.json          # everything machine-readable
    │
    ├── ablation/                                # Stage 2   (--skip ablation)
    │   ├── ablation_variants.yaml               # 6-variant manifest
    │   ├── ppo/                                 # one variant sub-sweep, same shape as
    │   ├── causal/                              # `headline/` above (sweep_manifest +
    │   ├── cgfa_scalar_only/                    # run-dirs + eval/), one per variant
    │   ├── cgfa_no_gate/
    │   ├── cgfa_no_cal/
    │   ├── cgfa_full/
    │   ├── aggregated/                          # cross-variant aggregation
    │   │   ├── figures/                         # Fig. 4 (ablation bars, per-variant deltas)
    │   │   ├── tables/significance.tex          # paired-bootstrap p-values vs cgfa_full
    │   │   └── aggregated_results.json
    │   ├── figures/cgfa_calibration.png         # Fig. 6   (Stage 4, auto-rendered)
    │   └── case_study/                          # Fig. 7   (Stage 5, auto-rendered)
    │       ├── case_study.png
    │       ├── case_study_steps.csv             # per-step factor attributions
    │       └── case_study_outcome.json
    │
    ├── transfer/                                # Stage 3   (--skip transfer)
    │   │                                        # ↓↓↓ FIXED-SPLIT MODE (--transfer-mode fixed) ↓↓↓
    │   ├── sweep_manifest.yaml
    │   ├── ppo_<deck>_vs_train_<ts>/            # train sweep (same shape as headline run-dir;
    │   ├── cgfa_<deck>_vs_train_<ts>/           # trained ONLY on --train-opponents)
    │   ├── eval/                                # in-distribution: vs --train-opponents
    │   ├── eval_heldout/                        # held-out: vs --heldout-opponents
    │   └── transfer/                            # cross-split aggregation
    │       ├── transfer_report.json             # per-agent gap + paired-bootstrap CI
    │       ├── transfer_summary.csv             # long-form per (agent, deck, seed, split, opp)
    │       ├── transfer_per_opponent.csv        # per held-out opponent x agent CI
    │       └── figures/transfer_gap.png         # Fig. 5
    │   │                                        # ↓↓↓ LEAVE-ONE-OUT MODE (--transfer-mode leave-one-out) ↓↓↓
    │   └── loo_<deck>/                          # one fold per opponent in the pool;
    │       ├── sweep_manifest.yaml              # each fold trains a fresh sweep on N-1
    │       ├── ppo_<deck>_vs_train_<ts>/        # opponents and evals on the held-out 1.
    │       ├── cgfa_<deck>_vs_train_<ts>/       # Same internal shape as the fixed-split tree
    │       ├── eval/                            # above, just nested one extra level deep.
    │       ├── eval_heldout/                    # Cost: ~5x training rollouts on a 5-deck pool.
    │       └── transfer/                        # Aggregated across folds by Stage 6 below.
    │           ├── transfer_report.json
    │           ├── transfer_summary.csv
    │           ├── transfer_per_opponent.csv
    │           └── figures/transfer_gap.png
    │
    └── cross_source/                            # Stage 6   (--skip cross-source)
        ├── figures/                             # paired-bootstrap roll-up over EVERY
        │   ├── win_rate_by_opponent.png         # eval_results.json under this benchmark run
        │   ├── headline_comparison.png          # (headline + every ablation variant +
        │   └── per_matchup_heatmap.png          # both transfer splits, including all LOO folds).
        ├── tables/
        │   ├── headline.tex                     # one row per (source, agent, deck) cell
        │   └── significance.tex                 # Holm-Bonferroni-adjusted p-values
        └── aggregated_results.json              # ↑ headline benchmark significance table
```

**Standalone subcommands** (`mtg-research pipeline`, `ablation`,
`transfer`) write the same internal shape, but **without** the
master-dir wrapper. Output lives directly at
`results/research/<experiment-name>/...`. Use the master-dir layout
when you need a single end-to-end `paper` run; use the flat layout
when you're iterating on a single stage.

**Cross-experiment comparison** (`mtg-research aggregate ...`) writes to
its own dir:

```
results/research/comparison_{a}_vs_{b}/
├── figures/
│   ├── win_rate_by_opponent.png
│   ├── headline_comparison.png
│   └── per_matchup_heatmap.png
├── tables/
│   ├── headline.tex
│   └── significance.tex
└── aggregated_results.json                      # everything machine-readable
```

---

## Testing

The test suite covers the environment, agents, SCM/CWM components,
training/evaluation loops, and benchmark pipeline utilities.

### Running Tests

```bash
# Run all tests (using uv)
uv run pytest

# Run all tests (standard)
pytest tests/

# Run with coverage report
uv run pytest --cov=mtg --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_env.py -v

# Run specific test class
uv run pytest tests/test_agents.py::TestGreedyAggroAgent -v

# Include slow tests (evaluation workflows)
uv run pytest -m ""

# Run only slow tests
uv run pytest -m slow
```

### Test Categories

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_env.py` | 32 | Environment mechanics, observation/action spaces, keywords, counters, protection, exile, tokens, domain |
| `test_agents.py` | 39 | All agent implementations (Random, GreedyAggro, Control, Midrange, Ramp, ConvokeAggro, Causal), agent registry |
| `test_gameplay.py` | 24 | Combat mechanics, game phases, player state, mana payment, game simulation, rewards |
| `test_causal_model.py` | 11 | SCM structure, causal variables, interventions |
| `test_workflows.py` | 28 | Training, evaluation, gameplay workflows, HTML reports, CLI display |

---

## Project Structure

```
mtg-causal-rl/
├── mtg/                          # Main package
│   ├── env/                      # Environment implementation
│   │   ├── mtg_env.py           # Gymnasium environment
│   │   ├── card_definitions.py  # Card registry and definitions
│   │   ├── deck_archetypes.py   # Archetype definitions
│   │   ├── rules.py             # Game rules engine
│   │   ├── action_mask.py       # Legal action computation
│   │   ├── observation.py       # Observation building
│   │   └── reward.py            # Reward calculation
│   ├── causal/                   # Causal model
│   │   ├── scm.py               # Structural causal model
│   │   ├── interventions.py     # do-calculus operations
│   │   └── counterfactuals.py   # Counterfactual queries
│   ├── agents/                   # Agent implementations
│   │   ├── base/                # BaseAgent + AgentRegistry
│   │   ├── heuristics/          # Heuristic baselines
│   │   ├── reinforcement_learning/ # RL baselines
│   │   └── causal/              # Causal agents
│   ├── simulation/               # Game simulation
│   │   └── game_simulator.py    # Abstract game runner
│   ├── utils/                    # Utilities
│   │   ├── visualization.py     # Publication-style figures
│   │   ├── cli_display.py       # Rich CLI interface
│   │   ├── html_report.py       # HTML replay generation
│   │   └── interactive.py       # Interactive CLI prompts
│   └── training/                 # Training utilities
├── scripts/
│   ├── runner/                   # Day-to-day workflows
│   │   ├── run_training.py      # Training workflow (single run)
│   │   ├── run_evaluation.py    # Evaluation workflow (single agent)
│   │   ├── run_gameplay.py      # Gameplay workflow (all agents)
│   │   └── regenerate_plots.py  # Re-render plots from a saved run
│   ├── research/                 # Composable research pipeline (mtg-research)
│   │   ├── cli.py               # mtg-research entry point (subcommands + wizard)
│   │   ├── train_sweep.py       # Stage 1: multi-seed training sweep
│   │   ├── eval_sweep.py        # Stage 2: high-fidelity evaluation
│   │   ├── aggregate.py         # Stage 3: figures + LaTeX + significance tests
│   │   └── stats.py             # Wilson CI, paired bootstrap, Wilcoxon, ...
│   └── pre-commit/               # Code quality hooks
├── tutorial/                     # Tutorial notebooks
│   └── getting_started.ipynb    # Comprehensive tutorial
├── tests/                        # Test suite
│   ├── test_env.py              # Environment tests
│   ├── test_agents.py           # Agent tests
│   ├── test_causal_model.py     # SCM tests
│   └── test_workflows.py        # Workflow tests
├── results/                      # Experiment outputs
│   ├── trained_agents/          # Saved models
│   ├── evaluations/             # Evaluation results
│   ├── gameplay/                # Game replays
│   └── plots/                   # Generated figures
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

---

## Visualizations

MTG-Causal-RL ships publication-style plots and terminal visualizations. See
[Workflows](#workflows) for the CLI entry points.

### Benchmark Figures

Every benchmark result figure is produced automatically by the
[`mtg-research paper`](#one-command-reproduction-mtg-research-paper)
pipeline and packaged into a portable `figures/` + `tables/` +
`raw_numbers/` bundle by the
[`export_paper_bundle`](#exporting-figures-and-tables) script. The
same bundle can be reused as the figure pack for any follow-up
writeup that builds on this benchmark.

#### What's in each figure

- **Fig. 3 — Headline comparison** (`figures/fig3a_headline_comparison.png` + `figures/fig3b_per_matchup_heatmap.png`) — overall win rate per (agent, deck) with 95% percentile-bootstrap CIs and a per-matchup heatmap.
- **Fig. 4 — Six-point ablation** (`figures/fig4_ablation.png`) — `ppo` / `causal` / `cgfa_scalar_only` / `cgfa_no_gate` / `cgfa_no_cal` / `cgfa_full` overall win rates side-by-side, with planned source-pair significance against `cgfa_full`.
- **Fig. 5 — Held-out transfer** (`figures/fig5_transfer_gap.png`) — in-distribution vs held-out mean win rate per agent, with paired-bootstrap CI on the gap.
- **Fig. 6 — Calibration trajectory** (`figures/fig6_cgfa_calibration.png`) — three-panel CGFA-PPO diagnostic: per-factor Pearson(Â<sub>k</sub>, ε<sub>k</sub>), per-factor credit share, and residual gate ḡ(s) over training.
- **Fig. 7 — Case study** (`figures/fig7_case_study.png`) — single deterministic episode showing per-factor critic V<sub>k</sub>(s), per-factor advantage A<sub>k</sub>(s,a), and the calibration overlay turn-by-turn.

The Structural Causal Model (SCM) diagram is not experiment output; generate it
directly from `mtg.utils.visualization`:

```python
from mtg.utils.visualization import create_scm_diagram

fig = create_scm_diagram()
fig.savefig("scm_diagram.pdf", dpi=300, bbox_inches="tight")
```

### CLI Experience

Features:

- Live training metrics with sparklines and trend indicators.
- Mulligan, battlefield, life-total, and phase displays.
- Action history and turn summaries for both players.
- Styled result tables and optional HTML replays.

The game visualization includes:
- **Play/Draw Selection** - Coin flip to determine who goes first
- **Mulligan Phase** - Visual card display for both players
- **All 7 Phases** - Untap → Upkeep → Draw → Main 1 → Combat → Main 2 → End
- **Both Players' States** - Life bars, hand size, lands, mana, and power
- **Creature Lists** - Names with power/toughness and tapped status
- **Action History** - Recent actions with highlighted current action
- **Turn Summaries** - Side-by-side comparison after each turn
- **Speed Control** - Slow (5s), Medium (3s), or Fast (1s) delays

### Programmatic Visualization

```python
from mtg.utils.visualization import (
    apply_publication_style,
    create_scm_diagram,
    create_learning_curve,
    create_comparison_bar,
)

# Apply publication style
apply_publication_style()

# Create SCM diagram
fig = create_scm_diagram()
fig.savefig("scm.pdf", dpi=300)

# Create learning curves
fig = create_learning_curve(
    data={"Agent1": {"steps": steps, "mean": means, "std": stds}},
    metric="win_rate",
    title="Win Rate vs Training Steps",
)
```

---

## Extending the Benchmark

### Adding a Custom Card

```python
from mtg.env.card_definitions import (
    Card, CardType, ManaCost, Keyword, CardRegistry
)

# Define the card
my_card = Card(
    name="Custom Creature",
    card_type=CardType.CREATURE,
    mana_cost=ManaCost.from_string("2R"),
    power=3,
    toughness=2,
    keywords={Keyword.HASTE},
)

# Register it
registry = CardRegistry.get_instance()
registry.register(my_card)
```

### Adding a Custom Archetype

```python
from mtg.env.deck_archetypes import (
    DeckArchetype, ArchetypeStrategy, register_custom_archetype
)

my_deck = DeckArchetype(
    name="custom_aggro",
    display_name="Custom Aggro",
    description="My custom aggressive deck",
    strategy=ArchetypeStrategy.AGGRO,
    card_list=[
        ("Mountain", 22),
        ("Monastery Swiftspear", 4),
        # ... 60 total cards
    ],
)

register_custom_archetype(my_deck)
```

### Adding a Custom Agent

1. Place your agent in the appropriate folder:
   - `mtg/agents/heuristics/` for hand-crafted heuristics
   - `mtg/agents/reinforcement_learning/` for learned agents
   - `mtg/agents/causal/` for causal agents

2. Import and register it in `mtg/agents/__init__.py` so it shows up in `list_agents()`.

3. Optionally create an instance directly or use the registry.

```python
from mtg.agents import BaseAgent, register_agent, get_agent
import numpy as np

# 1. Define your agent by subclassing BaseAgent
class GreedyAgent(BaseAgent):
    """Agent that always takes the highest-index legal action."""

    def __init__(self, seed=None):
        super().__init__(name="GreedyAgent", deterministic=True)

    def select_action(self, observation, action_mask, info=None):
        legal_actions = np.where(action_mask > 0)[0]
        if len(legal_actions) == 0:
            return 0
        return int(legal_actions[-1])  # Take highest legal action

# 2. Register your agent
register_agent("greedy", GreedyAgent)

# 3. Use it via the registry
agent = get_agent("greedy", seed=42)

# 4. Or use it directly in training/evaluation
# python scripts/runner/run_evaluation.py --agent greedy
```

### Advanced Custom Agent (with Learning)

```python
from mtg.agents import BaseAgent, register_agent
import numpy as np

class MyLearningAgent(BaseAgent):
    """Custom agent with Q-learning."""

    def __init__(self, observation_dim, action_dim, seed=None):
        super().__init__(name="MyLearningAgent", deterministic=False)
        self.q_table = {}
        self.epsilon = 0.1
        self.lr = 0.1
        self.gamma = 0.99
        self._rng = np.random.default_rng(seed)

    def select_action(self, observation, action_mask, info=None):
        obs_key = tuple(observation.round(2))
        legal = np.where(action_mask > 0)[0]

        if self._rng.random() < self.epsilon:
            return int(self._rng.choice(legal))

        if obs_key not in self.q_table:
            self.q_table[obs_key] = np.zeros(len(action_mask))

        q_values = self.q_table[obs_key].copy()
        q_values[action_mask == 0] = -np.inf
        return int(np.argmax(q_values))

    def learn(self, obs, action, reward, next_obs, done, info=None):
        obs_key = tuple(obs.round(2))
        next_key = tuple(next_obs.round(2))

        if obs_key not in self.q_table:
            self.q_table[obs_key] = np.zeros(478)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(478)

        target = reward + self.gamma * np.max(self.q_table[next_key]) * (1 - done)
        self.q_table[obs_key][action] += self.lr * (target - self.q_table[obs_key][action])

        return {"td_error": target - self.q_table[obs_key][action]}

# Register and use
register_agent("my_qlearning", MyLearningAgent)
```

### Adding a Causal Variable

```python
from mtg.causal.scm import CausalVariable, CausalLayer, StructuralCausalModel

scm = StructuralCausalModel()

custom_var = CausalVariable(
    name="burn_potential",
    layer=CausalLayer.STRATEGIC,
    var_type="continuous",
    description="Direct damage potential from hand",
    parents=["card_count", "mana_t"],
)

scm.variables.add(custom_var)
```

### Future Directions

The benchmark currently uses deck-specific training. Natural extensions include:

- Multi-deck self-play with asymmetric deck matchups.
- Population-based training over diverse agent/deck pairs.
- Metagame optimization where deck choice is part of the policy.
- Curriculum learning across opponent difficulty and archetype diversity.
- Transfer learning between deck archetypes using the SCM interface.

---

## Tutorial

For a comprehensive walkthrough, see the Jupyter notebook:

```bash
# Open the tutorial
jupyter notebook tutorial/getting_started.ipynb
```

The tutorial covers:
1. **Environment basics** - Creating and interacting with the MTG environment
2. **Deck archetypes** - Exploring the 5 competitive decks
3. **Causal model** - Understanding the SCM structure
4. **Running agents** - Evaluating baseline and trained agents
5. **Agent registry** - Using and extending the agent system
6. **Custom agents** - Building your own agents step-by-step
7. **Visualizations** - Creating publication-ready figures

---

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{anonymous2026mtgcausal,
  title={{MTG}-Causal-{RL}: A Causal Reinforcement Learning Benchmark for Magic: The Gathering},
  author={{Anonymous}},
  year={2026},
  note={TBD}
}
```

---

## Related Work

This benchmark builds on foundational work in:

- **Causal RL**: [CausalWorld](https://arxiv.org/abs/2010.04296), [Invariant Causal Prediction for MDPs](https://arxiv.org/abs/2006.06635)
- **Game AI**: [AlphaGo](https://www.nature.com/articles/nature16961), [Pluribus](https://www.science.org/doi/10.1126/science.aay2400)
- **MTG Complexity**: [MTG is Turing Complete](https://arxiv.org/abs/1904.09828)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contributing

Contributions are welcome! Please see our contributing guidelines and ensure code passes pre-commit checks:

```bash
pre-commit install
pre-commit run --all-files
```

---

**Made for the Causal RL and Magic: The Gathering research community**

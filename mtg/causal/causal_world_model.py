"""Learned causal world model for MTG strategic decision-making.

This module implements a neural network that learns the causal transition
dynamics of strategic MTG variables from experience. Unlike the hand-coded
SCM structural equations, this model is trained online during PPO training,
providing:

1. **Learned causal effects**: The model predicts how each action changes
   causal variables (mana, board pressure, etc.), replacing brittle
   string-matching heuristics.

2. **Causal auxiliary loss**: Training the model alongside PPO forces the
   shared observation encoder to learn causal representations, improving
   sample efficiency and generalization.

3. **Win probability estimation**: A separate head predicts P(win) from
   causal variable values, providing a learned potential function for
   reward shaping that adapts over training.

This is the core technical contribution that makes the system genuine
causal RL (training-time integration) rather than inference-time heuristic.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

CAUSAL_VAR_NAMES = (
    "mana",
    "card_advantage",
    "board_pressure",
    "tempo",
    "life_buffer",
    "threat_density",
    "removal_avail",
)

N_CAUSAL_VARS = len(CAUSAL_VAR_NAMES)


@dataclass
class CausalTransition:
    """A single (obs, action, causal_t, causal_t+1, outcome) transition."""

    obs: np.ndarray
    action: int
    causal_vars: np.ndarray
    next_causal_vars: np.ndarray
    terminal: bool = False
    outcome: float = 0.0  # 1.0 win, 0.0 loss, 0.5 ongoing


class CausalWorldModel(nn.Module):
    """Neural network that learns causal variable transition dynamics.

    Architecture:
        Encoder: obs ⊕ action_onehot → hidden → hidden
        Delta head: hidden → Δ(causal_var) for each causal variable
        WinProb head: causal_vars → P(win)

    The delta head learns how actions change causal variables.
    The win-prob head learns a value function in causal variable space.
    Together they enable computing the causal effect of any action on
    win probability: ΔP(win) = WinProb(CV + Δ(CV)) - WinProb(CV).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        buffer_size: int = 20_000,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_causal = N_CAUSAL_VARS

        # Action embedding (more efficient than one-hot for large action spaces)
        self.action_embed = nn.Embedding(action_dim, 32)

        # Shared encoder for observation + action
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Predict causal variable deltas
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_causal),
        )

        # Predict win probability from causal variable values
        self.win_prob_head = nn.Sequential(
            nn.Linear(self.n_causal, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self._buffer: deque[CausalTransition] = deque(maxlen=buffer_size)
        self._outcome_buffer: deque[tuple[np.ndarray, float]] = deque(maxlen=buffer_size)
        self._train_steps = 0

    def predict_causal_delta(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Predict change in causal variables for (obs, action) pairs.

        Args:
            obs: Observation tensor, shape (batch, obs_dim).
            action: Action indices, shape (batch,) as long tensor.

        Returns:
            Predicted deltas, shape (batch, n_causal).
        """
        act_emb = self.action_embed(action)
        x = torch.cat([obs, act_emb], dim=-1)
        h = self.encoder(x)
        return self.delta_head(h)

    def predict_win_prob(self, causal_vars: torch.Tensor) -> torch.Tensor:
        """Predict win probability from causal variable values.

        Args:
            causal_vars: Causal variable tensor, shape (batch, n_causal).

        Returns:
            Win probability, shape (batch, 1).
        """
        return torch.sigmoid(self.win_prob_head(causal_vars))

    def estimate_action_effects(
        self,
        obs: np.ndarray,
        legal_actions: np.ndarray,
        current_causal: np.ndarray,
    ) -> np.ndarray:
        """Estimate causal effect on win probability for each legal action.

        For each action a, computes:
            ΔWP(a) = WinProb(CV + Δ(CV|a)) - WinProb(CV)

        Args:
            obs: Current observation vector.
            legal_actions: Array of legal action indices.
            current_causal: Current causal variable values.

        Returns:
            Array of estimated win probability deltas, one per legal action.
        """
        self.eval()
        with torch.no_grad():
            n = len(legal_actions)
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).expand(n, -1)
            act_t = torch.tensor(legal_actions, dtype=torch.long)
            cv_t = torch.tensor(current_causal, dtype=torch.float32).unsqueeze(0).expand(n, -1)

            deltas = self.predict_causal_delta(obs_t, act_t)
            new_cv = cv_t + deltas

            current_wp = self.predict_win_prob(cv_t)
            new_wp = self.predict_win_prob(new_cv)

            effects = (new_wp - current_wp).squeeze(-1).numpy()
        self.train()
        return effects

    def record_transition(self, transition: CausalTransition) -> None:
        """Store a transition for training."""
        self._buffer.append(transition)
        if transition.terminal:
            self._outcome_buffer.append((transition.next_causal_vars, transition.outcome))

    @property
    def has_enough_data(self) -> bool:
        """Whether enough transitions have been collected to train."""
        return len(self._buffer) >= 128

    def train_on_buffer(
        self,
        batch_size: int = 128,
        n_steps: int = 16,
    ) -> dict[str, float]:
        """Train the model on buffered transitions.

        Returns:
            Dictionary of loss metrics.
        """
        if not self.has_enough_data:
            return {"delta_loss": 0.0, "wp_loss": 0.0}

        self.train()
        total_delta_loss = 0.0
        total_wp_loss = 0.0

        for _ in range(n_steps):
            batch = random.sample(list(self._buffer), min(batch_size, len(self._buffer)))

            obs_batch = torch.tensor(np.array([t.obs for t in batch]), dtype=torch.float32)
            act_batch = torch.tensor(np.array([t.action for t in batch]), dtype=torch.long)
            cv_batch = torch.tensor(np.array([t.causal_vars for t in batch]), dtype=torch.float32)
            next_cv_batch = torch.tensor(
                np.array([t.next_causal_vars for t in batch]), dtype=torch.float32
            )

            # Causal delta prediction loss
            pred_delta = self.predict_causal_delta(obs_batch, act_batch)
            true_delta = next_cv_batch - cv_batch
            delta_loss = functional.mse_loss(pred_delta, true_delta)

            # Win probability loss (from terminal outcomes)
            wp_loss = torch.tensor(0.0)
            if len(self._outcome_buffer) >= 32:
                wp_batch = random.sample(
                    list(self._outcome_buffer),
                    min(64, len(self._outcome_buffer)),
                )
                wp_cv = torch.tensor(np.array([s[0] for s in wp_batch]), dtype=torch.float32)
                wp_y = torch.tensor(
                    np.array([s[1] for s in wp_batch]), dtype=torch.float32
                ).unsqueeze(-1)
                wp_pred = self.predict_win_prob(wp_cv)
                wp_loss = functional.binary_cross_entropy(wp_pred, wp_y)

            loss = delta_loss + 0.5 * wp_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()

            total_delta_loss += delta_loss.item()
            total_wp_loss += wp_loss.item()

        self._train_steps += 1

        return {
            "delta_loss": total_delta_loss / n_steps,
            "wp_loss": total_wp_loss / n_steps,
            "buffer_size": len(self._buffer),
            "outcome_buffer_size": len(self._outcome_buffer),
        }

    def get_learned_scm_weights(self) -> dict[str, float]:
        """Extract the learned win-prob weights for SCM synchronization.

        Maps the win_prob_head's first-layer weights back to SCM variable
        weights, enabling the hand-coded SCM to benefit from learned data.
        """
        with torch.no_grad():
            importance = self.win_prob_head[0].weight.data.abs().mean(dim=0)
            total = importance.sum().item() + 1e-8

        return {
            name: float(importance[i].item() / total) for i, name in enumerate(CAUSAL_VAR_NAMES)
        }

    def state_dict_for_save(self) -> dict:
        """Lightweight state dict for checkpointing (no buffer)."""
        return {
            "model_state": self.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "train_steps": self._train_steps,
        }

    def load_state_from(self, state: dict) -> None:
        """Restore from checkpoint."""
        self.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self._train_steps = state.get("train_steps", 0)


def causal_vars_to_array(cv_dict: dict[str, float]) -> np.ndarray:
    """Convert a causal variable dict to a fixed-order numpy array."""
    return np.array([cv_dict.get(k, 0.0) for k in CAUSAL_VAR_NAMES], dtype=np.float32)

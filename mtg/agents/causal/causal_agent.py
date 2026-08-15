"""Causal RL agent with a learned causal world model.

This agent performs causal reinforcement learning by combining a PPO
policy with an auxiliary causal world model and an interventional
scoring layer over a declared structural causal model (SCM).  Three
mechanisms make this causal RL rather than a heuristic:

1. **Learned Causal World Model (CWM)**: A separate neural network is
   trained alongside PPO on ``(obs, action, causal_vars,
   next_causal_vars)`` transitions so the model learns how actions
   change the typed causal variables.  This network has its **own**
   parameters and does *not* share a trunk with the PPO policy.

2. **Learned Causal Action Scoring**: At decision time, the CWM's
   predicted next-step causal variables are fed into the SCM
   structural equation for ``win_prob`` to estimate each action's
   downstream effect.  The final score is a blend of the PPO log-prob
   and the CWM-induced causal effect.

3. **SCM Weight Synchronization**: The ``WinProbLearner`` online-fits
   logistic-regression weights for the SCM ``win_prob`` equation from
   terminal game outcomes, so the structural equation stays
   calibrated to the current training distribution.

Key design: PPO trains the policy as usual.  An auxiliary callback
trains the CausalWorldModel after each PPO rollout.  At inference the
CWM replaces the SCM's fallback hand-coded intervention mapping.
"""

from __future__ import annotations

import json
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mtg.agents.base.base import BaseAgent
from mtg.agents.reinforcement_learning.ppo_agent import PPOAgent
from mtg.causal.causal_world_model import (
    CAUSAL_VAR_NAMES,
    CausalTransition,
    CausalWorldModel,
    causal_vars_to_array,
)
from mtg.causal.scm import StructuralCausalModel


@dataclass
class CausalDecisionLog:
    """Log entry for a causal decision."""

    step: int
    action_taken: int
    legal_actions: list[int]
    policy_scores: list[float]
    causal_effects: list[float]
    combined_scores: list[float]
    causal_vars: dict[str, float]
    learned_effects: list[float] = field(default_factory=list)


class CausalAgent(BaseAgent):
    """Agent combining PPO with a learned causal world model.

    During training, the CausalWorldModel learns causal transition
    dynamics from experience.  During inference, it scores actions by
    their predicted causal effect on win probability.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        causal_weight: float = 0.6,
        exploration_rate: float = 0.1,
        exploration_rate_end: float = 0.01,
        exploration_anneal_steps: int = 50_000,
        mirror_opponent_deck: str = "mono_red_aggro",
        mirror_causal_weight: float = 0.72,
        mirror_exploration_rate: float = 0.01,
        intervention_samples: int = 16,
        log_decisions: bool = False,
        seed: int | None = None,
        cwm_hidden_dim: int = 128,
        cwm_lr: float = 1e-3,
        use_scm_fallback: bool = True,
        use_learned_cwm: bool = True,
        **deprecated_kwargs: tp.Any,
    ) -> None:
        """Initialize the causal agent.

        Args:
            observation_dim: Dimension of observation space.
            action_dim: Number of discrete actions.
            causal_weight: Blend weight for causal vs policy scores.
            exploration_rate: Initial epsilon-greedy rate.
            exploration_rate_end: Final epsilon-greedy rate.
            exploration_anneal_steps: Steps over which epsilon anneals.
            mirror_opponent_deck: Deck name for mirror-specific tuning.
            mirror_causal_weight: Causal weight override for mirror matchups.
            mirror_exploration_rate: Exploration rate for mirror matchups.
            intervention_samples: Number of Monte Carlo samples used when
                estimating interventional effects with additive noise.  Our
                SCM is deterministic, so this is *not* a proper
                counterfactual estimator (which would require abduction of
                exogenous noise); it is an interventional robustness check.
            log_decisions: Whether to log decision details for analysis.
            seed: Random seed.
            cwm_hidden_dim: Hidden layer size for the CausalWorldModel.
            cwm_lr: Learning rate for the CausalWorldModel.
            use_scm_fallback: Use hand-coded SCM when learned model lacks data.
            use_learned_cwm: If False, never use the learned CWM for action
                scoring (SCM interventional path only).
            **deprecated_kwargs: Accepted for backwards compatibility only.
                ``counterfactual_samples`` is mapped to
                ``intervention_samples``; ``planning_depth`` is silently
                ignored (it was never wired up).
        """
        super().__init__(name="CausalAgent", deterministic=False)

        if "counterfactual_samples" in deprecated_kwargs:
            intervention_samples = int(deprecated_kwargs.pop("counterfactual_samples"))
        deprecated_kwargs.pop("planning_depth", None)
        if deprecated_kwargs:
            raise TypeError(
                f"Unexpected keyword arguments for CausalAgent: {sorted(deprecated_kwargs)}"
            )

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.causal_weight = causal_weight
        self.exploration_rate_start = exploration_rate
        self.exploration_rate_end = exploration_rate_end
        self.exploration_anneal_steps = exploration_anneal_steps
        self.mirror_opponent_deck = mirror_opponent_deck
        self.mirror_causal_weight = mirror_causal_weight
        self.mirror_exploration_rate = mirror_exploration_rate
        self.intervention_samples = intervention_samples
        self.log_decisions = log_decisions
        self.seed = seed
        self.use_scm_fallback = use_scm_fallback
        self.use_learned_cwm = use_learned_cwm

        self.scm = StructuralCausalModel()
        self.base_agent = PPOAgent(
            observation_dim=observation_dim,
            action_dim=action_dim,
            seed=seed,
        )
        self.causal_world_model = CausalWorldModel(
            obs_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=cwm_hidden_dim,
            lr=cwm_lr,
        )

        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._prev_causal_vars: np.ndarray | None = None
        self._prev_obs: np.ndarray | None = None
        self._prev_action: int | None = None
        self.decision_log: list[CausalDecisionLog] = []
        self.cwm_metrics: dict[str, float] = {}

    def initialize_model(self, env: tp.Any) -> None:
        """Initialize the base RL model."""
        self.base_agent.initialize_model(env)

    def set_env(self, env: tp.Any) -> None:
        """Swap the training environment while preserving learned weights."""
        self.base_agent.set_env(env)

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        info: dict[str, tp.Any] | None = None,
    ) -> int:
        """Select action using learned causal reasoning."""
        self._step_count += 1
        legal = np.where(action_mask > 0)[0]

        if len(legal) == 0:
            return 0

        # Record transition from the prior step for CWM training
        self._record_transition_if_ready(observation, info)

        effective_exploration = self._effective_exploration_rate(info)
        if not self.deterministic and self._rng.random() < effective_exploration:
            self._update_prev_state(observation, info, int(self._rng.choice(legal)))
            return self._prev_action  # type: ignore[return-value]

        if info is None:
            action = self.base_agent.select_action(observation, action_mask, info)
            self._update_prev_state(observation, info, action)
            return action

        causal_vars = self._extract_causal_vars(info)
        if causal_vars is None:
            action = self.base_agent.select_action(observation, action_mask, info)
            self._update_prev_state(observation, info, action)
            return action

        effective_causal_weight = self._effective_causal_weight(info)
        scores, policy_scores, causal_effects, learned_fx = self._score_actions(
            legal,
            causal_vars,
            observation,
            action_mask,
            info,
            effective_causal_weight,
        )

        best_idx = int(np.argmax(scores))
        action = int(legal[best_idx])

        if self.log_decisions:
            self.decision_log.append(
                CausalDecisionLog(
                    step=self._step_count,
                    action_taken=action,
                    legal_actions=legal.tolist(),
                    policy_scores=policy_scores.tolist(),
                    causal_effects=causal_effects.tolist(),
                    combined_scores=scores.tolist(),
                    causal_vars=dict(zip(CAUSAL_VAR_NAMES, causal_vars.tolist(), strict=False)),
                    learned_effects=learned_fx.tolist(),
                )
            )

        self._update_prev_state(observation, info, action)
        return action

    # ------------------------------------------------------------------
    # Action scoring
    # ------------------------------------------------------------------

    def _score_actions(
        self,
        legal_actions: np.ndarray,
        causal_vars: np.ndarray,
        observation: np.ndarray,
        action_mask: np.ndarray,
        info: dict[str, tp.Any],
        causal_weight: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Score legal actions combining policy and causal effects."""
        n = len(legal_actions)
        policy_scores = np.zeros(n)
        causal_effects = np.zeros(n)
        learned_effects = np.zeros(n)

        # Get policy log-probs
        policy_probs = self.base_agent.get_action_probabilities(observation, action_mask)
        for i, action in enumerate(legal_actions):
            policy_scores[i] = np.log(policy_probs[action] + 1e-8)

        # Get causal effects from learned model (primary) or SCM (fallback)
        if self.use_learned_cwm and self.causal_world_model.has_enough_data:
            learned_effects = self.causal_world_model.estimate_action_effects(
                observation,
                legal_actions,
                causal_vars,
            )
            causal_effects = learned_effects
        elif self.use_scm_fallback:
            action_metadata = info.get("action_metadata", {})
            for i, action in enumerate(legal_actions):
                meta = self._get_action_meta(action_metadata, int(action))
                causal_effects[i] = self._scm_estimate_effect(causal_vars, meta, info)

        # Normalize and combine
        policy_norm = self._normalize_scores(policy_scores)
        causal_norm = self._normalize_scores(causal_effects)
        combined = (
            (1 - causal_weight) * policy_norm + causal_weight * causal_norm + 0.05 * policy_scores
        )
        combined = combined - combined.max()
        return combined, policy_scores, causal_effects, learned_effects

    def _scm_estimate_effect(
        self,
        causal_vars: np.ndarray,
        action_meta: dict[str, tp.Any],
        info: dict[str, tp.Any],
    ) -> float:
        """SCM-based causal effect estimate (fallback when CWM lacks data).

        Uses card *properties* (is_removal, deals_damage, etc.) instead
        of brittle card-name string matching.
        """
        scm_vars = self._to_scm_causal_vars_from_array(causal_vars, info)
        intervention = self._action_to_intervention(scm_vars, action_meta, info)
        if not intervention:
            return 0.0

        factual = self.scm.evaluate(scm_vars, force_recompute=False)
        current_wp = factual.get("win_prob", 0.5)

        effects: list[float] = []
        for _ in range(max(1, self.intervention_samples)):
            result = self.scm.do_intervention(scm_vars, intervention)
            noise = (
                {k: self._rng.normal(0, 0.03) for k in intervention}
                if self.intervention_samples > 1
                else {}
            )
            if noise:
                for k, v in noise.items():
                    if k in result:
                        result[k] = result[k] + v
                result = self.scm.evaluate(result, force_recompute=True)
            effects.append(result.get("win_prob", 0.5) - current_wp)

        return float(np.mean(effects))

    def _action_to_intervention(
        self,
        causal_vars: dict[str, float],
        action_meta: dict[str, tp.Any],
        info: dict[str, tp.Any],
    ) -> dict[str, float]:
        """Map action to SCM intervention using card properties (not names)."""
        intervention: dict[str, float] = {}
        action_type = str(action_meta.get("action_type", "")).lower()
        card_type = str(action_meta.get("card_type", "")).lower()

        if action_type == "pass":
            return intervention

        if action_type == "play_land":
            intervention["mana_t"] = causal_vars.get("mana_t", 0) + 1
            intervention["mana_t1"] = intervention["mana_t"] + 1
        elif action_type == "cast_creature" or card_type == "creature":
            power = action_meta.get("power", 2)
            intervention["board_presence"] = causal_vars.get("board_presence", 0) + 1
            intervention["board_press"] = causal_vars.get("board_press", 0) + max(1.0, power / 2.0)
            intervention["threat_density"] = min(
                1.0,
                causal_vars.get("threat_density", 0) + power * 0.1,
            )
            intervention["tempo"] = causal_vars.get("tempo", 0) + 0.1
        elif action_type in ("cast_instant", "cast_sorcery"):
            # Use card properties instead of name matching
            is_removal = bool(action_meta.get("is_removal", False))
            is_counterspell = bool(action_meta.get("is_counterspell", False))
            draws_cards = int(action_meta.get("draws_cards", 0))
            deals_damage = int(action_meta.get("deals_damage", 0))

            if is_removal or deals_damage > 0:
                intervention["removal_avail"] = max(
                    0,
                    causal_vars.get("removal_avail", 1) - 1,
                )
                intervention["board_press"] = causal_vars.get("board_press", 0) + 0.8
                intervention["threat_density"] = min(
                    1.0,
                    causal_vars.get("threat_density", 0.5) + 0.1,
                )
            if draws_cards > 0:
                intervention["card_adv"] = causal_vars.get("card_adv", 0) + draws_cards
            if is_counterspell:
                intervention["tempo"] = causal_vars.get("tempo", 0) + 0.1
                intervention["board_press"] = causal_vars.get("board_press", 0) + 0.5
        elif action_type in ("attack", "attack_toggle", "attack_all"):
            board_power = causal_vars.get("board_presence", 0) * 2
            opponent_life = info.get("opponent_life", 20)
            intervention["board_press"] = causal_vars.get("board_press", 0) + board_power * 0.2
            if opponent_life - board_power <= 0:
                intervention["win_prob"] = 1.0

        return intervention

    # ------------------------------------------------------------------
    # Transition recording (for CWM training)
    # ------------------------------------------------------------------

    def _record_transition_if_ready(
        self,
        current_obs: np.ndarray,
        info: dict[str, tp.Any] | None,
    ) -> None:
        """Record a causal transition from the prior step."""
        if (
            self._prev_obs is not None
            and self._prev_causal_vars is not None
            and self._prev_action is not None
            and info is not None
        ):
            current_cv = self._extract_causal_vars(info)
            if current_cv is not None:
                terminal = info.get("game_result") is not None
                outcome = 0.5
                if terminal:
                    game_res = info.get("game_result", "")
                    if game_res == "win":
                        outcome = 1.0
                    elif game_res == "loss":
                        outcome = 0.0
                    # draw stays 0.5
                self.causal_world_model.record_transition(
                    CausalTransition(
                        obs=self._prev_obs,
                        action=self._prev_action,
                        causal_vars=self._prev_causal_vars,
                        next_causal_vars=current_cv,
                        terminal=terminal,
                        outcome=outcome,
                    )
                )

    def _update_prev_state(
        self,
        obs: np.ndarray,
        info: dict[str, tp.Any] | None,
        action: int,
    ) -> None:
        """Cache current state for next-step transition recording."""
        self._prev_obs = obs.copy()
        self._prev_action = action
        if info is not None:
            cv = self._extract_causal_vars(info)
            self._prev_causal_vars = cv
        else:
            self._prev_causal_vars = None

    # ------------------------------------------------------------------
    # Causal variable extraction
    # ------------------------------------------------------------------

    def _extract_causal_vars(
        self,
        info: dict[str, tp.Any],
    ) -> np.ndarray | None:
        """Extract causal variable array from info dict."""
        env_vars = info.get("causal_variables", {})
        if not env_vars:
            return None
        return causal_vars_to_array(env_vars)

    def _to_scm_causal_vars_from_array(
        self,
        cv_array: np.ndarray,
        info: dict[str, tp.Any],
    ) -> dict[str, float]:
        """Map env causal array + info to full SCM variable dict.

        Populates all parent variables that the SCM structural equations
        reference (``own_power``, ``opp_power``, ``threat_count``, etc.)
        so that ``evaluate(force_recompute=True)`` produces correct results.
        """
        mapped: dict[str, float] = {
            "mana_t": float(cv_array[0]),
            "card_adv": float(cv_array[1]),
            "board_press": float(cv_array[2]),
            "tempo": float(cv_array[3]),
            "life_buffer": float(cv_array[4]),
            "threat_density": float(cv_array[5]),
        }
        if len(cv_array) > 6:
            mapped["removal_avail"] = float(cv_array[6])

        player_creatures = info.get("player_creatures", []) if info else []
        opp_creatures = info.get("opponent_creatures", []) if info else []
        mapped["board_presence"] = float(len(player_creatures))
        mapped["opp_board_presence"] = float(len(opp_creatures))
        mapped["card_count"] = float(info.get("hand_size", 0) if info else 0)
        mapped["own_life"] = float(info.get("player_life", 20) if info else 20)
        mapped["opp_life"] = float(info.get("opponent_life", 20) if info else 20)

        # Prefer the env-provided removal_avail from causal_variables when
        # ``cv_array`` did not already populate it.
        if "removal_avail" not in mapped or not info:
            env_cv = info.get("causal_variables", {}) if info else {}
            mapped["removal_avail"] = float(env_cv.get("removal_avail", 0))

        # Parent variables required by SCM structural equations
        mapped["own_power"] = float(info.get("board_power", 0) if info else 0)
        mapped["opp_power"] = float(info.get("opponent_power", 0) if info else 0)
        mapped["threat_count"] = float(sum(1 for c in player_creatures if len(c) > 1 and c[1] >= 2))
        # Tempo equation parents: approximate mana_spent as non-land permanents
        lands_on_bf = float(info.get("lands_on_battlefield", 0) if info else 0)
        opp_lands = float(info.get("opponent_lands", 0) if info else 0)
        mapped["mana_spent"] = max(0, mapped["board_presence"] - lands_on_bf)
        mapped["opp_mana"] = max(1, opp_lands)
        mapped["opp_mana_spent"] = max(0, float(len(opp_creatures)))

        return mapped

    # ------------------------------------------------------------------
    # Matchup-aware hyperparameters
    # ------------------------------------------------------------------

    def _is_mirror_match(self, info: dict[str, tp.Any] | None) -> bool:
        if not info:
            return False
        player_deck = str(info.get("player_deck", "")).lower()
        opponent_deck = str(info.get("opponent_deck", "")).lower()
        mirror = self.mirror_opponent_deck.lower()
        return player_deck == mirror and opponent_deck == mirror

    def _effective_causal_weight(self, info: dict[str, tp.Any] | None) -> float:
        return self.mirror_causal_weight if self._is_mirror_match(info) else self.causal_weight

    @property
    def exploration_rate(self) -> float:
        """Current exploration rate after annealing."""
        progress = min(1.0, self._step_count / max(1, self.exploration_anneal_steps))
        return (
            self.exploration_rate_start
            + (self.exploration_rate_end - self.exploration_rate_start) * progress
        )

    def _effective_exploration_rate(self, info: dict[str, tp.Any] | None) -> float:
        return (
            self.mirror_exploration_rate if self._is_mirror_match(info) else self.exploration_rate
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _get_action_meta(
        self,
        action_metadata: dict[int | str, dict[str, tp.Any]],
        action: int,
    ) -> dict[str, tp.Any]:
        if action in action_metadata:
            return action_metadata[action]
        action_str = str(action)
        if action_str in action_metadata:
            return action_metadata[action_str]
        return {}

    def _normalize_scores(self, values: np.ndarray) -> np.ndarray:
        if len(values) == 0:
            return values
        mean = float(np.mean(values))
        std = float(np.std(values))
        if std < 1e-8:
            return values - mean
        return (values - mean) / std

    # ------------------------------------------------------------------
    # Training / save / load
    # ------------------------------------------------------------------

    def train(
        self,
        total_timesteps: int,
        callback: tp.Any | None = None,
        progress_bar: bool = True,
        reset_num_timesteps: bool = True,
    ) -> None:
        """Train the base PPO agent.

        The CausalWorldModel is trained via the CausalAuxCallback
        (created externally in the training pipeline).
        """
        self.base_agent.train(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=progress_bar,
            reset_num_timesteps=reset_num_timesteps,
        )

    def train_causal_world_model(self) -> dict[str, float]:
        """Explicitly train the CWM on collected transitions.

        Called by the training callback after each PPO rollout.
        """
        metrics = self.causal_world_model.train_on_buffer()
        self.cwm_metrics = metrics
        return metrics

    def save(self, path: str | Path) -> None:
        """Save agent (PPO model + CWM state)."""
        path = Path(path)
        self.base_agent.save(path)
        cwm_path = path.with_suffix(".cwm.pt")
        import torch

        torch.save(self.causal_world_model.state_dict_for_save(), cwm_path)

    def load(self, path: str | Path) -> None:
        """Load agent (PPO model + CWM state if available)."""
        path = Path(path)
        self.base_agent.load(path)
        cwm_path = path.with_suffix(".cwm.pt")
        if cwm_path.exists():
            import torch

            state = torch.load(cwm_path, map_location="cpu", weights_only=True)
            self.causal_world_model.load_state_from(state)

    def get_action_probabilities(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
    ) -> np.ndarray:
        """Return action probabilities from the base PPO policy."""
        return self.base_agent.get_action_probabilities(observation, action_mask)

    def record_game_outcome(self, causal_vars: dict[str, float], outcome: float) -> None:
        """Record a game outcome for WinProb weight learning."""
        self.scm.record_outcome(causal_vars, outcome)

    def _to_scm_causal_vars(
        self,
        env_vars: dict[str, tp.Any],
        info: dict[str, tp.Any],
    ) -> dict[str, float]:
        """Map environment causal variables to SCM variable names.

        Kept for backwards compatibility with the training pipeline which
        calls ``agent._to_scm_causal_vars(cv, terminal_info)``.
        """
        if not env_vars:
            return {}
        mapped: dict[str, float] = {
            "mana_t": float(env_vars.get("mana", 0.0)),
            "card_adv": float(env_vars.get("card_advantage", 0.0)),
            "board_press": float(env_vars.get("board_pressure", 0.0)),
            "tempo": float(env_vars.get("tempo", 0.0)),
            "life_buffer": float(env_vars.get("life_buffer", 0.0)),
            "threat_density": float(env_vars.get("threat_density", 0.0)),
            "removal_avail": float(env_vars.get("removal_avail", 0.0)),
        }
        player_creatures = info.get("player_creatures", []) if info else []
        opp_creatures = info.get("opponent_creatures", []) if info else []
        mapped["board_presence"] = float(len(player_creatures))
        mapped["opp_board_presence"] = float(len(opp_creatures))
        mapped["card_count"] = float(info.get("hand_size", 0) if info else 0)
        mapped["own_life"] = float(info.get("player_life", 20) if info else 20)
        mapped["opp_life"] = float(info.get("opponent_life", 20) if info else 20)
        return mapped

    # ------------------------------------------------------------------
    # Decision log / stats
    # ------------------------------------------------------------------

    def save_decision_log(self, path: str | Path) -> None:
        """Serialize the decision log to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        log_data = [
            {
                "step": e.step,
                "action_taken": e.action_taken,
                "legal_actions": e.legal_actions,
                "policy_scores": e.policy_scores,
                "causal_effects": e.causal_effects,
                "combined_scores": e.combined_scores,
                "causal_vars": e.causal_vars,
                "learned_effects": e.learned_effects,
            }
            for e in self.decision_log
        ]
        with open(path, "w") as f:
            json.dump(log_data, f, indent=2)

    def clear_decision_log(self) -> None:
        """Clear stored decision log entries and reset step counter."""
        self.decision_log.clear()
        self._step_count = 0

    def get_causal_stats(self) -> dict[str, float]:
        """Compute summary statistics from the decision log."""
        if not self.decision_log:
            return {}
        causal_effects = [
            e.causal_effects[e.legal_actions.index(e.action_taken)]
            for e in self.decision_log
            if e.action_taken in e.legal_actions
        ]
        policy_scores = [
            e.policy_scores[e.legal_actions.index(e.action_taken)]
            for e in self.decision_log
            if e.action_taken in e.legal_actions
        ]
        causal_preferred = sum(
            1
            for e in self.decision_log
            if e.action_taken in e.legal_actions
            and e.causal_effects[e.legal_actions.index(e.action_taken)]
            > e.policy_scores[e.legal_actions.index(e.action_taken)]
        )
        return {
            "total_decisions": len(self.decision_log),
            "avg_causal_effect": float(np.mean(causal_effects)) if causal_effects else 0.0,
            "std_causal_effect": float(np.std(causal_effects)) if causal_effects else 0.0,
            "avg_policy_score": float(np.mean(policy_scores)) if policy_scores else 0.0,
            "causal_preferred_ratio": (
                causal_preferred / len(self.decision_log) if self.decision_log else 0.0
            ),
            "cwm_buffer_size": self.causal_world_model._buffer.__len__(),
            "cwm_train_steps": self.causal_world_model._train_steps,
            **self.cwm_metrics,
        }

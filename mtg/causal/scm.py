"""Structural Causal Model for MTG strategic decision-making.

This module implements the SCM that captures causal relationships
between strategic MTG variables. Variables are organized in layers:
Resources -> Board State -> Strategic Position -> Outcome.

The WinProb weights are learned via online logistic regression over
game outcomes, updated periodically during training.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

import networkx as nx
import numpy as np


class CausalLayer(Enum):
    """Causal layers in the MTG SCM.

    Attributes:
        RESOURCE: Mana and card resources.
        BOARD_STATE: Battlefield presence and power.
        STRATEGIC: High-level strategic position.
        OUTCOME: Win probability.

    """

    RESOURCE = auto()
    BOARD_STATE = auto()
    STRATEGIC = auto()
    OUTCOME = auto()


@dataclass
class CausalVariable:
    """A variable in the structural causal model.

    Attributes:
        name: Unique identifier for the variable.
        layer: Causal layer this variable belongs to.
        var_type: Type of variable ('continuous', 'discrete', 'binary').
        min_val: Minimum possible value.
        max_val: Maximum possible value.
        description: Human-readable description.
        parents: Names of parent variables in the causal graph.

    """

    name: str
    layer: CausalLayer
    var_type: str = "continuous"
    min_val: float = float("-inf")
    max_val: float = float("inf")
    description: str = ""
    parents: list[str] = field(default_factory=list)

    def normalize(self, value: float) -> float:
        """Normalize a value to [0, 1] range.

        Args:
            value: Raw value to normalize.

        Returns:
            Normalized value in [0, 1].

        """
        if self.min_val == float("-inf") or self.max_val == float("inf"):
            return np.tanh(value)
        return (value - self.min_val) / (self.max_val - self.min_val + 1e-8)


class CausalVariableSet:
    """Collection of causal variables with topological ordering.

    Attributes:
        variables: Dictionary of variable name to CausalVariable.

    """

    def __init__(self) -> None:
        """Initialize the variable set with default MTG variables."""
        self.variables: dict[str, CausalVariable] = {}
        self._register_default_variables()

    def add(self, variable: CausalVariable) -> None:
        """Add a variable to the set.

        Args:
            variable: Variable to add.

        """
        self.variables[variable.name] = variable

    def get(self, name: str) -> CausalVariable:
        """Get a variable by name.

        Args:
            name: Variable name.

        Returns:
            The causal variable.

        Raises:
            KeyError: If variable not found.

        """
        if name not in self.variables:
            raise KeyError(f"Variable '{name}' not found")
        return self.variables[name]

    def list_by_layer(self, layer: CausalLayer) -> list[CausalVariable]:
        """List variables in a specific layer.

        Args:
            layer: Layer to filter by.

        Returns:
            List of variables in that layer.

        """
        return [v for v in self.variables.values() if v.layer == layer]

    def get_topological_order(self) -> list[str]:
        """Get variables in topological order.

        Returns:
            List of variable names in causal order.

        """
        graph = nx.DiGraph()
        for name, var in self.variables.items():
            graph.add_node(name)
            for parent in var.parents:
                if parent in self.variables:
                    graph.add_edge(parent, name)
        return list(nx.topological_sort(graph))

    def _register_default_variables(self) -> None:
        """Register the default MTG causal variables.

        **Invariant:** every non-exogenous variable's ``parents`` list
        must exactly match the symbols read by its structural equation
        in :meth:`StructuralCausalModel._compute_variable`.  If you add
        a new term to an equation you must also add it as a parent here,
        otherwise :meth:`do_intervention` will silently fail to mark
        that node as affected by an upstream intervention.
        """
        # --- Exogenous / observed leaves ---------------------------------
        # These are the "raw" game-state readings that the env feeds into
        # the SCM.  They have no parents because they are observed directly
        # from the engine rather than computed from other causal variables.
        # Operational definitions: the descriptions below match what
        # ``mtg.env.reward.RewardCalculator.get_causal_variable_values``
        # actually publishes per step, so the structural equations below
        # operate on the same quantities the env exports. Renaming a
        # variable requires updating both sides plus the contract test
        # in ``tests/env/test_scm_env_contract.py``.
        for name, desc, lo, hi, vtype in [
            ("mana_t", "Number of mana-producing permanents controlled", 0, 10, "discrete"),
            ("card_count", "Cards in hand", 0, 15, "discrete"),
            ("board_presence", "Number of permanents controlled", 0, 10, "discrete"),
            ("opp_board_presence", "Opponent's permanent count", 0, 10, "discrete"),
            ("own_power", "Sum of own creature power on battlefield", 0, 40, "continuous"),
            ("opp_power", "Sum of opponent creature power on battlefield", 0, 40, "continuous"),
            ("threat_count", "Number of our creatures classified as threats", 0, 10, "discrete"),
            (
                "land_drop",
                "1 if at least one land is in hand (drop available), else 0",
                0,
                1,
                "binary",
            ),
            ("mana_creatures", "Mana-producing creatures on battlefield", 0, 10, "discrete"),
            (
                "mana_spent",
                "Number of non-land permanents controlled (proxy for committed mana)",
                0,
                10,
                "discrete",
            ),
            (
                "opp_mana",
                "Number of mana-producing permanents the opponent controls",
                0,
                10,
                "discrete",
            ),
            (
                "opp_mana_spent",
                "Number of opponent non-land permanents (proxy for committed mana)",
                0,
                10,
                "discrete",
            ),
            ("own_life", "Own life total", 0, 40, "discrete"),
            ("opp_life", "Opponent's life total", 0, 40, "discrete"),
            ("has_removal", "1 if at least one removal spell is in hand", 0, 1, "binary"),
        ]:
            self.add(
                CausalVariable(
                    name=name,
                    layer=CausalLayer.RESOURCE
                    if name
                    in {
                        "mana_t",
                        "card_count",
                        "land_drop",
                        "mana_creatures",
                        "mana_spent",
                        "opp_mana",
                        "opp_mana_spent",
                        "has_removal",
                    }
                    else CausalLayer.BOARD_STATE,
                    var_type=vtype,
                    min_val=lo,
                    max_val=hi,
                    description=desc,
                    parents=[],
                )
            )

        # --- Derived variables --------------------------------------------
        # Each ``parents`` list below mirrors the exact reads performed by
        # the structural equation in ``_compute_variable``.
        self.add(
            CausalVariable(
                name="mana_t1",
                layer=CausalLayer.RESOURCE,
                var_type="discrete",
                min_val=0,
                max_val=10,
                description="Expected mana next turn",
                parents=["mana_t", "land_drop", "mana_creatures"],
            )
        )

        self.add(
            CausalVariable(
                name="board_press",
                layer=CausalLayer.BOARD_STATE,
                var_type="continuous",
                min_val=-20,
                max_val=20,
                description="Net creature power (own - opponent)",
                parents=["own_power", "opp_power"],
            )
        )

        self.add(
            CausalVariable(
                name="threat_density",
                layer=CausalLayer.BOARD_STATE,
                var_type="continuous",
                min_val=0,
                max_val=1,
                description="Fraction of threats vs total permanents",
                parents=["board_presence", "threat_count"],
            )
        )

        self.add(
            CausalVariable(
                name="card_adv",
                layer=CausalLayer.STRATEGIC,
                var_type="continuous",
                min_val=-10,
                max_val=10,
                description="Board-presence card advantage (own - opp permanents)",
                parents=["board_presence", "opp_board_presence"],
            )
        )

        self.add(
            CausalVariable(
                name="tempo",
                layer=CausalLayer.STRATEGIC,
                var_type="continuous",
                min_val=-1,
                max_val=1,
                description="Mana efficiency and initiative",
                parents=["mana_t", "mana_spent", "opp_mana", "opp_mana_spent"],
            )
        )

        self.add(
            CausalVariable(
                name="life_buffer",
                layer=CausalLayer.STRATEGIC,
                var_type="continuous",
                min_val=-20,
                max_val=20,
                description="Own life minus opponent life",
                parents=["own_life", "opp_life"],
            )
        )

        self.add(
            CausalVariable(
                name="removal_avail",
                layer=CausalLayer.STRATEGIC,
                var_type="binary",
                min_val=0,
                max_val=1,
                description="Whether removal spell is available (from has_removal)",
                parents=["has_removal"],
            )
        )

        self.add(
            CausalVariable(
                name="win_prob",
                layer=CausalLayer.OUTCOME,
                var_type="continuous",
                min_val=0,
                max_val=1,
                description="Estimated win probability",
                parents=[
                    "card_adv",
                    "board_press",
                    "tempo",
                    "life_buffer",
                    "threat_density",
                    "removal_avail",
                ],
            )
        )


@dataclass
class SCMWeights:
    """Learned weights for the SCM structural equations.

    Attributes:
        card_adv_weight: Weight for card advantage in win prob.
        board_press_weight: Weight for board pressure in win prob.
        tempo_weight: Weight for tempo in win prob.
        life_buffer_weight: Weight for life buffer in win prob.
        threat_density_weight: Weight for threat density in win prob.
        removal_avail_weight: Weight for removal availability in win prob.
        bias: Bias term for win probability.

    """

    card_adv_weight: float = 0.3
    board_press_weight: float = 0.25
    tempo_weight: float = 0.15
    life_buffer_weight: float = 0.2
    threat_density_weight: float = 0.1
    removal_avail_weight: float = 0.08
    bias: float = 0.0


class WinProbLearner:
    """Online logistic regression learner for WinProb weights.

    Accumulates (feature_vector, outcome) pairs from completed games
    and periodically fits a logistic regression to update the SCM weights.
    """

    FEATURE_NAMES = (
        "card_adv",
        "board_press",
        "tempo",
        "life_buffer",
        "threat_density",
        "removal_avail",
    )

    def __init__(self, buffer_size: int = 2000, update_interval: int = 200) -> None:
        self._buffer: deque[tuple[np.ndarray, float]] = deque(maxlen=buffer_size)
        self._update_interval = update_interval
        self._samples_since_update = 0

    def record(self, causal_vars: dict[str, float], outcome: float) -> None:
        """Record a game outcome for learning.

        Args:
            causal_vars: Terminal causal variable values.
            outcome: 1.0 for win, 0.0 for loss.
        """
        features = np.array([causal_vars.get(k, 0.0) for k in self.FEATURE_NAMES])
        self._buffer.append((features, outcome))
        self._samples_since_update += 1

    def should_update(self) -> bool:
        """Return True when enough new samples have been collected to refit."""
        return self._samples_since_update >= self._update_interval and len(self._buffer) >= 50

    def fit(self, current_weights: SCMWeights) -> SCMWeights:
        """Fit logistic regression and return updated weights.

        Falls back to current weights if fitting fails or data is insufficient.
        """
        if len(self._buffer) < 50:
            return current_weights

        features = np.array([s[0] for s in self._buffer])
        y = np.array([s[1] for s in self._buffer])

        if len(np.unique(y)) < 2:
            return current_weights

        # Standardize features for stable fitting
        mean = features.mean(axis=0)
        std = features.std(axis=0) + 1e-8
        features_norm = (features - mean) / std

        # Simple gradient-based logistic regression (no sklearn dependency)
        w = np.zeros(features_norm.shape[1])
        bias = 0.0
        lr = 0.01
        for _ in range(200):
            logits = features_norm @ w + bias
            preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -10, 10)))
            error = preds - y
            w -= lr * (features_norm.T @ error) / len(y)
            bias -= lr * error.mean()

        # Convert back to original scale
        w_orig = w / std

        self._samples_since_update = 0
        return SCMWeights(
            card_adv_weight=float(w_orig[0]),
            board_press_weight=float(w_orig[1]),
            tempo_weight=float(w_orig[2]),
            life_buffer_weight=float(w_orig[3]),
            threat_density_weight=float(w_orig[4]),
            removal_avail_weight=float(w_orig[5]) if len(w_orig) > 5 else 0.08,
            bias=float(bias - (mean / std) @ w),
        )

    @property
    def sample_count(self) -> int:
        """Number of observation samples currently stored."""
        return len(self._buffer)


class StructuralCausalModel:
    """Structural Causal Model for MTG strategic reasoning.

    Attributes:
        variables: Collection of causal variables.
        weights: Learned weights for structural equations.
        graph: NetworkX graph representing causal structure.
        win_prob_learner: Online learner for WinProb weights.

    """

    def __init__(
        self,
        weights: SCMWeights | None = None,
        learn_win_prob: bool = True,
    ) -> None:
        """Initialize the SCM.

        Args:
            weights: Optional custom weights for structural equations.
            learn_win_prob: Whether to learn WinProb weights from game outcomes.

        """
        self.variables = CausalVariableSet()
        self.weights = weights or SCMWeights()
        self.graph = self._build_graph()
        self.win_prob_learner = WinProbLearner() if learn_win_prob else None

    def record_outcome(self, causal_vars: dict[str, float], outcome: float) -> None:
        """Record a game outcome for WinProb weight learning.

        Args:
            causal_vars: Terminal causal variable values.
            outcome: 1.0 for win, 0.0 for loss.
        """
        if self.win_prob_learner is not None:
            self.win_prob_learner.record(causal_vars, outcome)
            if self.win_prob_learner.should_update():
                self.weights = self.win_prob_learner.fit(self.weights)

    def _build_graph(self) -> nx.DiGraph:
        """Build the causal graph from variables.

        Returns:
            NetworkX DiGraph representing causal structure.

        """
        graph = nx.DiGraph()
        for name, var in self.variables.variables.items():
            graph.add_node(name, layer=var.layer, var_type=var.var_type)
            for parent in var.parents:
                if parent in self.variables.variables:
                    graph.add_edge(parent, name)
        return graph

    def evaluate(
        self,
        observations: dict[str, float],
        force_recompute: bool = True,
    ) -> dict[str, float]:
        """Evaluate all causal variables given observations.

        When ``force_recompute`` is True (the default), downstream variables
        are always recomputed from their parents via structural equations,
        even if they already appear in *observations*.  This is critical for
        multi-step SCM look-ahead: after updating an intervention variable,
        all descendant values must propagate correctly.

        Args:
            observations: Dictionary of observed variable values.
            force_recompute: If True, recompute every variable that has a
                structural equation, overwriting any value already in
                *observations* (except root / exogenous nodes).

        Returns:
            Dictionary of all variable values including derived.

        """
        result = observations.copy()
        order = self.variables.get_topological_order()

        for var_name in order:
            var = self.variables.get(var_name)
            has_parents = bool(var.parents)
            if force_recompute and has_parents or var_name not in result:
                result[var_name] = self._compute_variable(var_name, result)

        return result

    def _compute_variable(
        self,
        var_name: str,
        values: dict[str, float],
    ) -> float:
        """Compute a derived variable from parent values.

        Args:
            var_name: Name of variable to compute.
            values: Current known values.

        Returns:
            Computed value for the variable.

        """
        if var_name == "mana_t1":
            land_drop = values.get("land_drop", 1.0)
            mana_creatures = values.get("mana_creatures", 0.0)
            return values.get("mana_t", 0) + land_drop + mana_creatures

        if var_name == "board_press":
            return values.get("own_power", 0) - values.get("opp_power", 0)

        if var_name == "threat_density":
            presence = values.get("board_presence", 0)
            threats = values.get("threat_count", 0)
            return threats / max(1, presence)

        if var_name == "card_adv":
            # Aligned with env: battlefield count differential.
            return values.get("board_presence", 0) - values.get("opp_board_presence", 0)

        if var_name == "tempo":
            mana = values.get("mana_t", 1)
            spent = values.get("mana_spent", 0)
            opp_mana = values.get("opp_mana", 1)
            opp_spent = values.get("opp_mana_spent", 0)
            own_eff = spent / max(1, mana)
            opp_eff = opp_spent / max(1, opp_mana)
            return np.clip(own_eff - opp_eff, -1, 1)

        if var_name == "life_buffer":
            own_life = values.get("own_life", 20)
            opp_life = values.get("opp_life", 20)
            return own_life - opp_life

        if var_name == "removal_avail":
            return float(values.get("has_removal", 0))

        if var_name == "win_prob":
            return self._compute_win_prob(values)

        # Variables with parents in the causal graph but no explicit
        # structural equation (e.g. board_presence, card_count): preserve
        # the observed value rather than defaulting to zero.
        return values.get(var_name, 0.0)

    def _compute_win_prob(self, values: dict[str, float]) -> float:
        """Compute win probability using structural equation.

        Args:
            values: Dictionary of variable values.

        Returns:
            Estimated win probability in (0, 1).

        """
        card_adv = self.variables.get("card_adv").normalize(values.get("card_adv", 0))
        board_press = self.variables.get("board_press").normalize(values.get("board_press", 0))
        tempo = values.get("tempo", 0)
        life_buffer = self.variables.get("life_buffer").normalize(values.get("life_buffer", 0))
        threat_density = values.get("threat_density", 0)
        removal_avail = values.get("removal_avail", 0)

        logit = (
            self.weights.card_adv_weight * card_adv
            + self.weights.board_press_weight * board_press
            + self.weights.tempo_weight * tempo
            + self.weights.life_buffer_weight * life_buffer
            + self.weights.threat_density_weight * threat_density
            + self.weights.removal_avail_weight * removal_avail
            + self.weights.bias
        )

        return float(1 / (1 + np.exp(-logit)))

    def do_intervention(
        self,
        observations: dict[str, float],
        intervention: dict[str, float],
    ) -> dict[str, float]:
        """Apply a do-intervention and compute downstream effects.

        Clamps the intervention variables and recomputes all downstream
        (descendant) variables via their structural equations.  Non-descendant
        variables retain their observed values.

        Args:
            observations: Current variable values.
            intervention: Variables to intervene on with new values.

        Returns:
            New variable values after intervention.

        """
        result = observations.copy()
        result.update(intervention)

        # Collect all descendants of intervened variables
        intervened_vars = set(intervention.keys())
        affected: set[str] = set()
        for iv in intervened_vars:
            if iv in self.variables.variables:
                affected |= self.get_descendants(iv)

        order = self.variables.get_topological_order()
        for var_name in order:
            if var_name in intervened_vars:
                continue
            if var_name in affected or var_name not in result:
                result[var_name] = self._compute_variable(var_name, result)

        return result

    def interventional_prediction(
        self,
        factual: dict[str, float],
        intervention: dict[str, float],
    ) -> dict[str, float]:
        """Predict variable values under a hypothetical intervention.

        This performs Pearl's *interventional* query: compute P(Y | do(X=x))
        by clamping intervention variables and propagating through structural
        equations.  It is **not** a full three-step counterfactual (which
        would require abduction of exogenous noise, then intervention, then
        prediction).  For our deterministic SCM the two coincide, but we
        use the interventional name to be precise.

        Args:
            factual: Current observed variable values.
            intervention: Variables to set (do-operator).

        Returns:
            Variable values after intervention.

        """
        return self.do_intervention(factual, intervention)

    # Backwards compatibility alias
    counterfactual = interventional_prediction

    def get_causal_effect(
        self,
        treatment_var: str,
        outcome_var: str,
        observations: dict[str, float],
        treatment_values: tuple[float, float] = (0.0, 1.0),
    ) -> float:
        """Estimate causal effect of treatment on outcome.

        Args:
            treatment_var: Variable to treat.
            outcome_var: Outcome to measure.
            observations: Current state.
            treatment_values: (control, treatment) values.

        Returns:
            Estimated causal effect (ATE).

        """
        control_val, treat_val = treatment_values

        control_obs = self.do_intervention(observations, {treatment_var: control_val})
        treat_obs = self.do_intervention(observations, {treatment_var: treat_val})

        control_outcome = control_obs.get(outcome_var, 0)
        treat_outcome = treat_obs.get(outcome_var, 0)

        return treat_outcome - control_outcome

    def is_d_separated(
        self,
        x: str,
        y: str,
        z: set[str] | None = None,
    ) -> bool:
        """Check if X and Y are d-separated given Z.

        Args:
            x: First variable.
            y: Second variable.
            z: Conditioning set.

        Returns:
            True if d-separated.

        """
        if z is None:
            z = set()
        return nx.d_separated(self.graph, {x}, {y}, z)

    def get_parents(self, var_name: str) -> list[str]:
        """Get parent variables.

        Args:
            var_name: Variable name.

        Returns:
            List of parent variable names.

        """
        return list(self.graph.predecessors(var_name))

    def get_children(self, var_name: str) -> list[str]:
        """Get child variables.

        Args:
            var_name: Variable name.

        Returns:
            List of child variable names.

        """
        return list(self.graph.successors(var_name))

    def get_ancestors(self, var_name: str) -> set[str]:
        """Get all ancestor variables.

        Args:
            var_name: Variable name.

        Returns:
            Set of ancestor variable names.

        """
        return nx.ancestors(self.graph, var_name)

    def get_descendants(self, var_name: str) -> set[str]:
        """Get all descendant variables.

        Args:
            var_name: Variable name.

        Returns:
            Set of descendant variable names.

        """
        return nx.descendants(self.graph, var_name)


# Alias for backwards compatibility
CausalSCM = StructuralCausalModel

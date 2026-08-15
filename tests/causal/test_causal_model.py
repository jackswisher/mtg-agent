"""Tests for the causal modeling module."""

import numpy as np
import pytest

from mtg.causal import CausalLayer, CausalVariable, CausalVariableSet, StructuralCausalModel
from mtg.env import MTGEnv


class TestCausalVariables:
    """Tests for causal variables."""

    def test_causal_variable_creation(self):
        """Test creating a causal variable."""
        var = CausalVariable(
            name="test_var",
            layer=CausalLayer.RESOURCE,
            var_type="continuous",
            min_val=0.0,
            max_val=1.0,
            description="Test variable",
        )

        assert var.name == "test_var"
        assert var.var_type == "continuous"
        assert var.layer == CausalLayer.RESOURCE

    def test_causal_variable_set_default(self):
        """Test default causal variable set."""
        var_set = CausalVariableSet()

        assert len(var_set.variables) > 0
        # Check for expected variables
        assert "mana_t" in var_set.variables

    def test_causal_variable_set_topology(self):
        """Test topological ordering."""
        var_set = CausalVariableSet()
        order = var_set.get_topological_order()

        # Order should contain at least some variables
        assert len(order) > 0


class TestStructuralCausalModel:
    """Tests for the structural causal model."""

    def test_scm_creation(self):
        """Test SCM creation."""
        scm = StructuralCausalModel()
        assert scm is not None
        assert scm.variables is not None

    def test_scm_graph(self):
        """Test SCM graph structure."""
        scm = StructuralCausalModel()

        # Graph should exist
        assert scm.graph is not None
        assert scm.graph.is_directed()

    def test_scm_has_variables(self):
        """Test SCM has expected variables."""
        scm = StructuralCausalModel()

        # Should have key strategic variables
        var_names = [v.name for v in scm.variables.variables.values()]
        assert "mana_t" in var_names

    def test_scm_evaluate_exists(self):
        """Test SCM has evaluate method."""
        scm = StructuralCausalModel()

        # Should have evaluate method
        assert hasattr(scm, "evaluate")

    def test_scm_evaluate(self):
        """Test SCM evaluation with observations."""
        scm = StructuralCausalModel()

        observations = {
            "mana_t": 3.0,
            "card_advantage": 1.0,
        }

        result = scm.evaluate(observations)
        assert isinstance(result, dict)


class TestCausalIntegration:
    """Integration tests for the causal module."""

    def test_scm_with_agent(self):
        """Test SCM integration with causal agent."""
        from mtg.agents import CausalAgent

        env = MTGEnv(deck_archetype="mono_red_aggro", seed=42)
        agent = CausalAgent(observation_dim=100, action_dim=env.action_space.n, seed=42)

        # Agent should have an SCM
        assert hasattr(agent, "scm")

        # Agent should be able to select actions
        obs = np.zeros(100)
        action_mask = np.ones(env.action_space.n, dtype=np.int8)
        action = agent.select_action(obs, action_mask)

        assert 0 <= action < env.action_space.n

    def test_causal_variable_normalization(self):
        """Test variable normalization."""
        var = CausalVariable(
            name="test",
            layer=CausalLayer.RESOURCE,
            min_val=0.0,
            max_val=10.0,
        )

        # Test normalization (allow floating point tolerance)
        assert var.normalize(0.0) == pytest.approx(0.0, abs=1e-6)
        assert var.normalize(10.0) == pytest.approx(1.0, abs=1e-6)
        assert var.normalize(5.0) == pytest.approx(0.5, abs=1e-6)

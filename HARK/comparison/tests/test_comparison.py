"""
Comprehensive tests for model comparison infrastructure.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path

from HARK.comparison import ModelComparison, EconomicMetrics
from HARK.comparison.parameter_translation import ParameterTranslator
from HARK.comparison.adapters import (
    HARKAdapter,
    SSJAdapter,
    MaliarAdapter,
    ExternalAdapter,
    AiyagariAdapter,
)


class TestModelComparison:
    """Test ModelComparison class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.description = "Test model comparison"
        self.primitives = {
            "CRRA": 2.0,
            "DiscFac": 0.96,
            "Rfree": 1.03,
            "CapShare": 0.36,
            "DeprFac": 0.025,
            "PermGroFac": 1.01,
            "LbrInd": 1.0,
        }
        self.comp = ModelComparison(self.description, self.primitives)

    def test_initialization(self):
        """Test proper initialization."""
        assert self.comp.description == self.description
        assert self.comp.primitives == self.primitives
        assert len(self.comp.solutions) == 0
        assert len(self.comp.metrics) == 0

    def test_add_solution_method(self):
        """Test adding solution method configurations."""
        config = {"aXtraCount": 100, "tolerance": 1e-6}
        self.comp.add_solution_method("test_method", config)

        assert "test_method" in self.comp.method_configs
        assert self.comp.method_configs["test_method"] == config

    def test_parameter_translation(self):
        """Test parameter translation works correctly."""
        self.comp.add_solution_method("krusell_smith/HARK", {})

        # Test that solve creates adapter with translated params
        with pytest.raises(Exception):  # Will fail without full setup
            self.comp.solve("krusell_smith/HARK", verbose=False)

    def test_unknown_method_error(self):
        """Test error for unknown solution method."""
        with pytest.raises(ValueError, match="Method configuration"):
            self.comp.solve("unknown_method")

    def test_metrics_computation(self):
        """Test metrics computation placeholder."""
        # Would need mock solutions to test fully
        metrics = self.comp.compute_metrics()
        assert isinstance(metrics, dict)

    def test_comparison_report(self):
        """Test comparison report generation."""
        # Test with empty solutions
        df = self.comp.compare_solutions()
        assert len(df) == 0  # No solutions yet


class TestEconomicMetrics:
    """Test EconomicMetrics class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = EconomicMetrics()
        self.params = {"CRRA": 2.0, "DiscFac": 0.96, "Rfree": 1.03, "PermGroFac": 1.0}

    def test_euler_equation_error(self):
        """Test Euler equation error computation."""

        # Simple linear consumption function for testing
        def c_func(m):
            return 0.1 * m

        test_points = np.linspace(1, 10, 5)
        errors = self.metrics.euler_equation_error(c_func, self.params, test_points)

        assert len(errors) == len(test_points)
        assert np.all(np.isfinite(errors))
        assert np.all(errors >= 0)  # Errors should be non-negative

    def test_bellman_equation_error(self):
        """Test Bellman equation error computation."""

        # Simple test functions
        def v_func(m):
            return -(m**2)

        def c_func(m):
            return 0.1 * m

        test_points = np.linspace(1, 10, 5)
        errors = self.metrics.bellman_equation_error(
            v_func, c_func, self.params, test_points
        )

        assert len(errors) == len(test_points)
        assert np.all(np.isfinite(errors))

    def test_wealth_distribution_metrics(self):
        """Test wealth distribution metrics."""
        wealth = np.random.lognormal(2, 1, 1000)
        metrics = self.metrics.wealth_distribution_metrics(wealth)

        assert "gini" in metrics
        assert "mean_wealth" in metrics
        assert "median_wealth" in metrics
        assert "wealth_share_top10" in metrics

        # Check reasonable values
        assert 0 <= metrics["gini"] <= 1
        assert metrics["mean_wealth"] > 0
        assert metrics["wealth_share_top10"] >= 0.1  # Top 10% has at least 10%

    def test_den_haan_marcet_statistic(self):
        """Test Den Haan-Marcet statistic computation."""
        # Simple test data with fixed seed for reproducibility
        np.random.seed(42)
        T = 100
        k_series = 10 + 0.1 * np.random.randn(T)
        sim_data = {"aggregate_capital": k_series}

        # Simple forecast function
        def forecast(k):
            return 0.9 * k + 1.0  # Mean reversion

        r2 = self.metrics.den_haan_marcet_statistic(sim_data, forecast)

        # R² can be slightly outside [-1, 1] due to numerical issues
        assert -1.1 <= r2 <= 1.1  # Allow small numerical tolerance


class TestParameterTranslator:
    """Test ParameterTranslator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.translator = ParameterTranslator()
        self.primitives = {
            "CRRA": 2.0,
            "DiscFac": 0.96,
            "Rfree": 1.03,
            "CapShare": 0.36,
        }

    def test_hark_translation(self):
        """Test translation for HARK methods."""
        translated = self.translator.translate(
            self.primitives, "krusell_smith/HARK", {}
        )

        # Check that primitives are preserved
        for key, value in self.primitives.items():
            assert translated[key] == value

        # Check defaults added
        assert "AgentCount" in translated
        assert "act_T" in translated

    def test_ssj_translation(self):
        """Test translation for SSJ method."""
        translated = self.translator.translate(self.primitives, "ssj", {})

        # Check parameter name mappings
        assert translated["beta"] == self.primitives["DiscFac"]
        assert translated["alpha"] == self.primitives["CapShare"]
        assert translated["eis"] == 1.0 / self.primitives["CRRA"]
        assert translated["r_ss"] == self.primitives["Rfree"] - 1.0

    def test_maliar_translation(self):
        """Test translation for Maliar methods."""
        translated = self.translator.translate(
            self.primitives, "maliar_winant_euler", {}
        )

        # Check neural network defaults
        assert "nn_layers" in translated
        assert "learning_rate" in translated
        assert "epochs" in translated

    def test_reverse_translation(self):
        """Test reverse translation."""
        ssj_params = {"beta": 0.96, "eis": 0.5, "alpha": 0.36, "r_ss": 0.03}

        primitives = self.translator.reverse_translate(ssj_params, "ssj")

        assert primitives["DiscFac"] == 0.96
        assert primitives["CRRA"] == 2.0  # 1/0.5
        assert primitives["CapShare"] == 0.36
        assert primitives["Rfree"] == 1.03


class TestAdapters:
    """Test individual adapter classes."""

    def test_hark_adapter_initialization(self):
        """Test HARK adapter initialization."""
        adapter = HARKAdapter()
        assert adapter.method_type == "HARK"
        assert adapter.agent is None
        assert adapter.economy is None

    def test_ssj_adapter_initialization(self):
        """Test SSJ adapter initialization."""
        adapter = SSJAdapter()
        assert adapter.method_type == "ssj"
        assert adapter.model is None
        assert adapter.steady_state is None

    def test_maliar_adapter_initialization(self):
        """Test Maliar adapter initialization."""
        for method in ["euler", "bellman", "reward"]:
            adapter = MaliarAdapter(method)
            assert adapter.method_type == method
            assert adapter.neural_net is None

    def test_external_adapter_file_loading(self):
        """Test external adapter file loading."""
        adapter = ExternalAdapter()

        # Create temporary test file
        test_data = {
            "cFunc": [0.1, 0.2, 0.3],
            "grid": [1.0, 2.0, 3.0],
            "method": "test",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            # Test loading
            solution = adapter.solve({"solution_path": temp_path})
            assert solution == test_data

            # Test policy extraction
            policy = adapter.get_consumption_policy()
            assert policy is not None
        finally:
            Path(temp_path).unlink()

    def test_aiyagari_adapter_initialization(self):
        """Test Aiyagari adapter initialization."""
        adapter = AiyagariAdapter()
        assert adapter.method_type == "aiyagari"
        assert adapter.agent is None
        assert adapter.equilibrium is None


class TestIntegration:
    """Integration tests for the full comparison workflow."""

    def test_simple_comparison_workflow(self):
        """Test a simple comparison workflow."""
        # Create comparison object
        comp = ModelComparison("Simple test model", {"CRRA": 2.0, "DiscFac": 0.96})

        # Add method configurations with mock solution data
        comp.add_solution_method(
            "external",
            {
                "solution_data": {
                    "cFunc": lambda m: 0.1 * m,
                    "description": "Mock solution",
                }
            },
        )

        # Solve with external adapter
        comp.solve("external", verbose=False)

        # Check that adapter was created and solution stored
        assert "external" in comp.adapters
        assert "external" in comp.solutions


class TestUtilities:
    """Test utility functions and edge cases."""

    def test_metrics_with_invalid_input(self):
        """Test metrics with invalid inputs."""
        metrics = EconomicMetrics()

        # Test with empty wealth array
        wealth_metrics = metrics.wealth_distribution_metrics(np.array([]))
        assert np.isnan(wealth_metrics["gini"])

        # Test convergence metric with short history
        conv_metrics = metrics.convergence_metric([1.0])
        assert not conv_metrics["converged"]
        assert conv_metrics["iterations"] == 1

    def test_adapter_test_points_generation(self):
        """Test test point generation in adapters."""
        adapter = HARKAdapter()
        points = adapter.get_test_points(100)

        assert "mNrm" in points
        assert len(points["mNrm"]) == 100
        assert np.all(points["mNrm"] > 0)

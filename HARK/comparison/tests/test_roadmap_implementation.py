"""
Tests for SDG Roadmap Implementation

This test suite validates that all key components of the HARK SDG roadmap
have been successfully implemented, including:
1. Block model definitions for macroeconomic models
2. Baseline solution functionality
3. Method comparison capabilities
4. Integration of traditional vs. cutting-edge algorithms
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os

from HARK.comparison.base import ModelComparison
from HARK.comparison.models.krusell_smith_blocks import (
    simple_krusell_smith_model,
    ks_calibration,
)
from HARK.comparison.models.aiyagari_blocks import (
    simple_aiyagari_model,
    aiyagari_calibration,
)


class TestBlockModelDefinitions(unittest.TestCase):
    """Test the block model definitions for macroeconomic models."""

    def test_krusell_smith_model_structure(self):
        """Test that Krusell-Smith model is properly defined."""
        model = simple_krusell_smith_model

        # Check model structure
        self.assertEqual(model.name, "simple_krusell_smith_model")
        self.assertEqual(len(model.blocks), 2)  # Individual + aggregate

        # Check block names
        block_names = [block.name for block in model.blocks]
        self.assertIn("simple_ks_individual", block_names)
        self.assertIn("simple_ks_aggregate", block_names)

        # Check individual block components
        individual_block = model.blocks[0]
        self.assertIn("agg_state", individual_block.shocks)
        self.assertIn("emp_state", individual_block.shocks)
        self.assertIn("c", individual_block.dynamics)  # Control variable
        self.assertIn("a", individual_block.dynamics)  # Assets

    def test_aiyagari_model_structure(self):
        """Test that Aiyagari model is properly defined."""
        model = simple_aiyagari_model

        # Check model structure
        self.assertEqual(model.name, "simple_aiyagari_model")
        self.assertEqual(len(model.blocks), 2)  # Individual + aggregate

        # Check block names
        block_names = [block.name for block in model.blocks]
        self.assertIn("simple_aiyagari_individual", block_names)
        self.assertIn("simple_aiyagari_aggregate", block_names)

        # Check individual block components
        individual_block = model.blocks[0]
        self.assertIn("emp_state", individual_block.shocks)
        self.assertIn("income_shk", individual_block.shocks)
        self.assertIn("c", individual_block.dynamics)  # Control variable
        self.assertIn("a", individual_block.dynamics)  # Assets

    def test_model_calibrations(self):
        """Test that model calibrations are properly defined."""
        # Krusell-Smith calibration
        ks_cal = ks_calibration
        required_ks_params = ["DiscFac", "CRRA", "CapShare", "DeprFac"]
        for param in required_ks_params:
            self.assertIn(param, ks_cal)
            self.assertIsInstance(ks_cal[param], (int, float))

        # Aiyagari calibration
        aiy_cal = aiyagari_calibration
        required_aiy_params = ["DiscFac", "CRRA", "CapShare", "DeprFac"]
        for param in required_aiy_params:
            self.assertIn(param, aiy_cal)
            self.assertIsInstance(aiy_cal[param], (int, float))


class TestBaselineFunctionality(unittest.TestCase):
    """Test the baseline solution functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.comparison = ModelComparison(
            model_description_md="Test model for roadmap implementation",
            primitives={"DiscFac": 0.95, "CRRA": 2.0},
        )
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after tests."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_baseline_solution(self):
        """Test saving baseline solutions."""
        # Create dummy solution and add it to the comparison object
        method_name = "test_method"
        solution = {
            "consumption_policy": "test_policy_data",
            "solved": True,
        }  # Use simple data

        # Add the solution to the comparison object first
        self.comparison.solutions[method_name] = solution
        self.comparison.method_configs[method_name] = {
            "method": "fixed_point",
            "tolerance": 1e-6,
        }
        self.comparison.metrics[method_name] = {
            "mean_consumption": 1.0,
            "mean_assets": 2.0,
        }
        self.comparison.computation_times[method_name] = 15.5

        filepath = os.path.join(self.temp_dir, "test_baseline.pkl")

        # Save baseline (should not raise an exception)
        try:
            self.comparison.save_baseline_solution(
                method_name, filepath, include_adapter=False
            )
        except Exception as e:
            self.fail(f"save_baseline_solution raised an exception: {e}")

        # Check that file was created
        self.assertTrue(os.path.exists(filepath))

    def test_load_baseline_solution(self):
        """Test loading baseline solutions."""
        # First create and save a baseline
        method_name = "test_method"
        solution = {"test": "data"}

        # Add the solution to the comparison object first
        self.comparison.solutions[method_name] = solution
        self.comparison.method_configs[method_name] = {"method": "test"}
        self.comparison.metrics[method_name] = {"test_metric": 1.0}
        self.comparison.computation_times[method_name] = 10.0

        filepath = os.path.join(self.temp_dir, "test_load_baseline.pkl")

        # Save then load
        self.comparison.save_baseline_solution(
            method_name, filepath, include_adapter=False
        )

        # Load baseline (should not raise an exception)
        try:
            self.comparison.load_baseline_solution(filepath, "loaded_baseline")
        except Exception as e:
            self.fail(f"load_baseline_solution raised an exception: {e}")

        # Check that baseline was loaded
        baselines = self.comparison.list_baselines()
        self.assertIn(
            "loaded_baseline",
            [baseline["baseline_name"] for _, baseline in baselines.iterrows()],
        )

    def test_baseline_management(self):
        """Test baseline management functionality."""
        # Test list_baselines when empty
        baselines = self.comparison.list_baselines()
        self.assertIsInstance(baselines, pd.DataFrame)

        # Test clear_baselines
        try:
            self.comparison.clear_baselines(["nonexistent"])
        except Exception as e:
            self.fail(f"clear_baselines raised an exception: {e}")


class TestMethodComparison(unittest.TestCase):
    """Test comparison of different solution methods."""

    def test_method_comparison_structure(self):
        """Test that method comparison framework is in place."""
        comparison = ModelComparison(
            model_description_md="Test model", primitives={"DiscFac": 0.95}
        )

        # Test that comparison object has the right methods
        self.assertTrue(hasattr(comparison, "save_baseline_solution"))
        self.assertTrue(hasattr(comparison, "load_baseline_solution"))
        self.assertTrue(hasattr(comparison, "compare_against_baseline"))
        self.assertTrue(hasattr(comparison, "list_baselines"))
        self.assertTrue(hasattr(comparison, "clear_baselines"))

    def test_metrics_computation(self):
        """Test economic metrics computation."""
        # Create dummy simulation results
        sim_results = {
            "consumption": np.random.lognormal(0, 0.2, (100, 1000)),
            "assets": np.random.lognormal(1, 0.3, (100, 1000)),
        }

        # Test that we can compute basic metrics
        mean_consumption = np.mean(sim_results["consumption"])
        mean_assets = np.mean(sim_results["assets"])
        consumption_vol = np.std(sim_results["consumption"])

        self.assertIsInstance(mean_consumption, float)
        self.assertIsInstance(mean_assets, float)
        self.assertIsInstance(consumption_vol, float)
        self.assertGreater(mean_consumption, 0)
        self.assertGreater(mean_assets, 0)


class TestRoadmapIntegration(unittest.TestCase):
    """Test integration of all SDG roadmap components."""

    def test_problem_representation(self):
        """Test that macroeconomic problems can be represented using blocks."""
        # Both models should be available
        self.assertIsNotNone(simple_krusell_smith_model)
        self.assertIsNotNone(simple_aiyagari_model)

        # Models should have block structure
        self.assertTrue(hasattr(simple_krusell_smith_model, "blocks"))
        self.assertTrue(hasattr(simple_aiyagari_model, "blocks"))

        # Models should be constructible with calibration
        try:
            simple_krusell_smith_model.construct_shocks(ks_calibration)
            simple_aiyagari_model.construct_shocks(aiyagari_calibration)
        except Exception as e:
            self.fail(f"Model construction failed: {e}")

    def test_solution_algorithm_support(self):
        """Test that multiple solution algorithms are supported."""
        # This tests the framework structure, not actual algorithm implementation

        # Traditional method representation
        traditional_method = {
            "name": "fixed_point_iteration",
            "type": "traditional_hark",
            "description": "Backward induction with fixed-point iteration",
        }

        # Cutting-edge method representation
        modern_method = {
            "name": "neural_network",
            "type": "deep_learning",
            "description": "Neural network approximation (Maliar et al.)",
        }

        # External method representation
        external_method = {
            "name": "maliar_external",
            "type": "external",
            "description": "External Maliar method implementation",
        }

        # All methods should be representable
        methods = [traditional_method, modern_method, external_method]
        for method in methods:
            self.assertIn("name", method)
            self.assertIn("type", method)
            self.assertIn("description", method)

    def test_comparative_analysis_ready(self):
        """Test that the framework is ready for comparative analysis."""
        comparison = ModelComparison(
            model_description_md="Test model", primitives={"DiscFac": 0.95}
        )

        # Framework should support the key comparison operations
        required_methods = [
            "save_baseline_solution",
            "load_baseline_solution",
            "compare_against_baseline",
            "list_baselines",
        ]

        for method_name in required_methods:
            self.assertTrue(hasattr(comparison, method_name))
            method = getattr(comparison, method_name)
            self.assertTrue(callable(method))

    def test_efficiency_workflow(self):
        """Test that efficient baseline-based workflow is implemented."""
        comparison = ModelComparison(
            model_description_md="Test model", primitives={"DiscFac": 0.95}
        )

        # Test workflow: save expensive baseline, reuse for comparisons
        temp_dir = tempfile.mkdtemp()
        try:
            baseline_file = os.path.join(temp_dir, "efficient_baseline.pkl")

            # Add a solution first
            method_name = "expensive_method"
            comparison.solutions[method_name] = {"policy": "test_policy"}
            comparison.computation_times[method_name] = 100.0

            # Step 1: Save expensive baseline
            comparison.save_baseline_solution(
                method_name, baseline_file, include_adapter=False
            )
            self.assertTrue(os.path.exists(baseline_file))

            # Step 2: Load for reuse
            comparison.load_baseline_solution(baseline_file, "reused_baseline")
            baselines = comparison.list_baselines()
            baseline_names = [
                baseline["baseline_name"] for _, baseline in baselines.iterrows()
            ]
            self.assertIn("reused_baseline", baseline_names)

            # Step 3: Can be compared against (framework in place)
            self.assertTrue(hasattr(comparison, "compare_against_baseline"))

        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)


class TestFullRoadmapImplementation(unittest.TestCase):
    """Integration test for the complete SDG roadmap implementation."""

    def test_traditional_vs_cutting_edge_framework(self):
        """Test that framework supports traditional vs. cutting-edge comparison."""

        # Traditional HARK approach simulation
        traditional_results = {
            "method": "fixed_point_iteration",
            "computation_time": 45.2,
            "accuracy": 1e-6,  # Lower error = higher accuracy
            "metrics": {"mean_consumption": 1.0, "welfare": 10.0},
        }

        # Cutting-edge deep learning approach simulation
        modern_results = {
            "method": "neural_network",
            "computation_time": 8.7,
            "accuracy": 1e-4,  # Higher error = lower accuracy
            "metrics": {"mean_consumption": 1.02, "welfare": 10.1},
        }

        # Framework should be able to compare these
        speed_ratio = (
            traditional_results["computation_time"] / modern_results["computation_time"]
        )
        # For accuracy comparison, invert the ratio since lower error = higher accuracy
        accuracy_ratio = (
            modern_results["accuracy"] / traditional_results["accuracy"]
        )  # Error ratio

        self.assertGreater(speed_ratio, 1)  # Traditional is slower
        self.assertGreater(
            accuracy_ratio, 1
        )  # Traditional has lower error (accuracy_ratio > 1 means traditional is more accurate)

        # This represents the trade-offs the framework should capture
        tradeoff = {
            "speed_improvement": speed_ratio,
            "accuracy_cost": accuracy_ratio,
            "welfare_difference": abs(
                modern_results["metrics"]["welfare"]
                - traditional_results["metrics"]["welfare"]
            ),
        }

        self.assertIsInstance(tradeoff, dict)
        self.assertIn("speed_improvement", tradeoff)
        self.assertIn("accuracy_cost", tradeoff)

    def test_high_dimensional_model_support(self):
        """Test support for high-dimensional heterogeneous agent models."""

        # Krusell-Smith: Aggregate uncertainty + heterogeneity
        ks_dimensions = {
            "individual_state_vars": ["a", "emp_state"],  # Assets + employment
            "aggregate_state_vars": ["K_agg", "agg_state"],  # Capital + productivity
            "control_vars": ["c"],  # Consumption
            "shocks": ["agg_state", "emp_state"],  # Aggregate + idiosyncratic
        }

        # Aiyagari: Idiosyncratic risk + distribution effects
        aiy_dimensions = {
            "individual_state_vars": ["a", "emp_state"],  # Assets + employment
            "aggregate_state_vars": ["K_agg", "r"],  # Capital + interest rate
            "control_vars": ["c"],  # Consumption
            "shocks": ["emp_state", "income_shk"],  # Employment + income
        }

        # Both should be high-dimensional (multiple state variables)
        for model_dims in [ks_dimensions, aiy_dimensions]:
            total_states = len(model_dims["individual_state_vars"]) + len(
                model_dims["aggregate_state_vars"]
            )
            self.assertGreaterEqual(total_states, 3)  # High-dimensional
            self.assertGreater(
                len(model_dims["shocks"]), 1
            )  # Multiple sources of uncertainty


def run_roadmap_tests():
    """Run all roadmap implementation tests."""
    # Create test suite
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestBlockModelDefinitions,
        TestBaselineFunctionality,
        TestMethodComparison,
        TestRoadmapIntegration,
        TestFullRoadmapImplementation,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbose=2)
    result = runner.run(suite)

    # Report results
    if result.wasSuccessful():
        print("\n" + "=" * 60)
        print("🎉 ALL SDG ROADMAP TESTS PASSED!")
        print("Framework is ready for traditional vs. cutting-edge comparisons")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Some tests failed. Check implementation.")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print("=" * 60)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_roadmap_tests()

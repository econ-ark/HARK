"""
Tests for baseline solution functionality in ModelComparison.
"""

import pytest
import tempfile
import os
from pathlib import Path

from HARK.comparison.base import ModelComparison


class TestBaselineFunctionality:
    """Test suite for baseline solution save/load functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.primitives = {
            "DiscFac": 0.96,
            "CRRA": 2.0,
            "Rfree": 1.03,
            "TranShkStd": 0.2,
            "PermShkStd": 0.1,
        }

        self.description = "Test model for baseline functionality"

        self.comparison = ModelComparison(
            model_description_md=self.description, primitives=self.primitives
        )

    def test_save_baseline_no_solution(self):
        """Test that saving baseline fails when method not solved."""
        with pytest.raises(ValueError, match="has not been solved yet"):
            self.comparison.save_baseline_solution("nonexistent_method")

    def test_save_and_load_baseline_basic(self):
        """Test basic save and load functionality."""
        # Mock a solution with pickle-able objects
        mock_solution = {
            "policy_function": "mock_policy_string",
            "value_function": "mock_value_string",
            "solved": True,
        }

        # Add mock solution and related data
        method_name = "test_method"
        self.comparison.solutions[method_name] = mock_solution
        self.comparison.method_configs[method_name] = {"param1": "value1"}
        self.comparison.computation_times[method_name] = 5.0
        self.comparison.metrics[method_name] = {"metric1": 0.5}

        # Test saving
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "test_baseline.pkl"

            saved_path = self.comparison.save_baseline_solution(
                method_name, str(filepath)
            )

            assert saved_path == str(filepath)
            assert filepath.exists()

            # Test loading
            loaded_data = self.comparison.load_baseline_solution(
                str(filepath), "loaded_baseline"
            )

            # Verify loaded data
            assert loaded_data["method"] == method_name
            assert loaded_data["primitives"] == self.primitives
            assert loaded_data["method_config"] == {"param1": "value1"}
            assert loaded_data["computation_time"] == 5.0
            assert loaded_data["metrics"] == {"metric1": 0.5}

            # Verify baseline is available in comparison object
            assert "loaded_baseline" in self.comparison.baseline_solutions
            assert "loaded_baseline" in self.comparison.solutions

    def test_load_baseline_file_not_found(self):
        """Test loading baseline with non-existent file."""
        with pytest.raises(FileNotFoundError):
            self.comparison.load_baseline_solution("nonexistent_file.pkl")

    def test_default_filepath_generation(self):
        """Test that default filepath is generated correctly."""
        # Mock a solution
        method_name = "test/method_with_slashes"
        self.comparison.solutions[method_name] = {"test": "data"}

        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory to avoid cluttering
            original_cwd = os.getcwd()
            os.chdir(temp_dir)

            try:
                saved_path = self.comparison.save_baseline_solution(method_name)
                expected_filename = "baseline_solution_test_method_with_slashes.pkl"
                assert saved_path == expected_filename
                assert Path(expected_filename).exists()
            finally:
                os.chdir(original_cwd)

    def test_list_baselines_empty(self):
        """Test listing baselines when none are loaded."""
        baselines_df = self.comparison.list_baselines()
        assert baselines_df.empty

    def test_list_baselines_with_data(self):
        """Test listing baselines with loaded data."""
        # Mock baseline data
        baseline_data = {
            "method": "test_method",
            "timestamp": 1234567890.0,
            "computation_time": 5.0,
            "metrics": {"metric1": 0.5},
            "simulation_results": {"result1": [1, 2, 3]},
            "solution": {"policy": "mock_policy"},
        }

        self.comparison.baseline_solutions["test_baseline"] = baseline_data
        self.comparison.adapters["test_baseline"] = "mock_adapter"

        baselines_df = self.comparison.list_baselines()

        assert len(baselines_df) == 1
        assert baselines_df.iloc[0]["baseline_name"] == "test_baseline"
        assert baselines_df.iloc[0]["original_method"] == "test_method"
        assert baselines_df.iloc[0]["computation_time"] == 5.0
        assert baselines_df.iloc[0]["has_metrics"]
        assert baselines_df.iloc[0]["has_simulation"]
        assert baselines_df.iloc[0]["has_adapter"]

    def test_clear_baselines(self):
        """Test clearing baseline solutions."""
        # Add mock baselines
        baseline_names = ["baseline1", "baseline2"]
        for name in baseline_names:
            self.comparison.baseline_solutions[name] = {"test": "data"}
            self.comparison.solutions[name] = {"test": "solution"}
            self.comparison.metrics[name] = {"test": "metric"}

        # Clear specific baseline
        self.comparison.clear_baselines(["baseline1"])

        assert "baseline1" not in self.comparison.baseline_solutions
        assert "baseline2" in self.comparison.baseline_solutions
        assert "baseline1" not in self.comparison.solutions
        assert "baseline2" in self.comparison.solutions

        # Clear all baselines
        self.comparison.clear_baselines()

        assert len(self.comparison.baseline_solutions) == 0
        assert "baseline2" not in self.comparison.solutions

    def test_compare_against_baseline_no_baseline(self):
        """Test comparison against non-existent baseline."""
        with pytest.raises(ValueError, match="Baseline 'nonexistent' not found"):
            self.comparison.compare_against_baseline("nonexistent")

    def test_compare_against_baseline_success(self):
        """Test successful comparison against baseline."""
        # Setup baseline
        baseline_data = {
            "method": "baseline_method",
            "solution": {"policy": "mock_policy"},
            "metrics": {"metric1": 1.0, "metric2": 2.0},
            "computation_time": 10.0,
            "primitives": self.primitives,
        }

        baseline_name = "test_baseline"
        self.comparison.baseline_solutions[baseline_name] = baseline_data
        self.comparison.solutions[baseline_name] = baseline_data["solution"]
        self.comparison.metrics[baseline_name] = baseline_data["metrics"]
        self.comparison.computation_times[baseline_name] = baseline_data[
            "computation_time"
        ]

        # Setup comparison method
        method_name = "test_method"
        self.comparison.solutions[method_name] = {"policy": "mock_policy"}
        self.comparison.metrics[method_name] = {"metric1": 1.1, "metric2": 1.8}
        self.comparison.computation_times[method_name] = 5.0

        # Perform comparison
        comparison_df = self.comparison.compare_against_baseline(baseline_name)

        assert len(comparison_df) == 1
        row = comparison_df.iloc[0]

        assert row["method"] == method_name
        assert row["baseline"] == baseline_name
        assert row["computation_time"] == 5.0
        assert row["baseline_computation_time"] == 10.0
        assert row["time_ratio"] == 0.5
        assert row["metric1"] == 1.1
        assert row["metric1_baseline"] == 1.0
        assert abs(row["metric1_rel_diff"] - 0.1) < 1e-10
        assert row["metric2"] == 1.8
        assert row["metric2_baseline"] == 2.0
        assert abs(row["metric2_rel_diff"] - (-0.1)) < 1e-10

    def test_compare_against_baseline_no_methods(self):
        """Test comparison when no methods are available."""
        # Setup baseline only
        baseline_data = {
            "method": "baseline_method",
            "solution": {"policy": "mock_policy"},
            "metrics": {"metric1": 1.0},
            "computation_time": 10.0,
            "primitives": self.primitives,
        }

        baseline_name = "test_baseline"
        self.comparison.baseline_solutions[baseline_name] = baseline_data
        self.comparison.solutions[baseline_name] = baseline_data["solution"]
        self.comparison.metrics[baseline_name] = baseline_data["metrics"]

        # No other methods available
        with pytest.warns(UserWarning, match="No methods available for comparison"):
            comparison_df = self.comparison.compare_against_baseline(baseline_name)

        assert comparison_df.empty

    def test_baseline_with_different_primitives(self):
        """Test loading baseline with different primitives shows warning."""
        # Create baseline with different primitives
        different_primitives = self.primitives.copy()
        different_primitives["DiscFac"] = 0.98  # Different value

        baseline_data = {
            "method": "baseline_method",
            "solution": {"policy": "mock_policy"},
            "primitives": different_primitives,
            "metrics": {},
            "timestamp": 1234567890.0,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "test_baseline.pkl"

            # Mock the pickle save/load
            import pickle

            with open(filepath, "wb") as f:
                pickle.dump(baseline_data, f)

            # Loading should show warning about different primitives
            with pytest.warns(UserWarning, match="different primitives"):
                self.comparison.load_baseline_solution(str(filepath), "test_baseline")

    def test_save_baseline_with_adapter(self):
        """Test saving baseline with adapter included."""
        method_name = "test_method"
        mock_solution = {"policy": "mock_policy"}
        mock_adapter = "mock_adapter"

        self.comparison.solutions[method_name] = mock_solution
        self.comparison.adapters[method_name] = mock_adapter

        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "test_baseline.pkl"

            # Save with adapter
            self.comparison.save_baseline_solution(
                method_name, str(filepath), include_adapter=True
            )

            # Load and verify adapter is included
            loaded_data = self.comparison.load_baseline_solution(
                str(filepath), "loaded_baseline"
            )

            assert "adapter" in loaded_data
            assert "loaded_baseline" in self.comparison.adapters

    def test_save_baseline_without_adapter(self):
        """Test saving baseline without adapter."""
        method_name = "test_method"
        mock_solution = {"policy": "mock_policy"}
        mock_adapter = "mock_adapter"

        self.comparison.solutions[method_name] = mock_solution
        self.comparison.adapters[method_name] = mock_adapter

        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "test_baseline.pkl"

            # Save without adapter
            self.comparison.save_baseline_solution(
                method_name, str(filepath), include_adapter=False
            )

            # Load and verify adapter is not included
            loaded_data = self.comparison.load_baseline_solution(
                str(filepath), "loaded_baseline"
            )

            assert "adapter" not in loaded_data
            assert "loaded_baseline" not in self.comparison.adapters

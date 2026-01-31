"""
Base model comparison infrastructure for HARK.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import warnings
from copy import deepcopy
import time
import pickle
from pathlib import Path

from HARK.metric import MetricObject
from .metrics import EconomicMetrics
from .parameter_translation import ParameterTranslator


class ModelComparison(MetricObject):
    """
    A class to compare different solution methods for heterogeneous agent models.

    This class provides a unified interface for solving the same economic model
    using different computational methods and comparing their results.

    Parameters
    ----------
    model_description_md : str
        Markdown description of the models being compared
    primitives : dict
        Economic parameters (preferences, shocks, production, etc.)

    Attributes
    ----------
    description : str
        Model description
    primitives : dict
        Economic primitive parameters
    solutions : dict
        Stored solutions by method name
    metrics : dict
        Computed metrics by method name
    adapters : dict
        Solution adapters by method name
    baseline_solutions : dict
        Stored baseline solutions loaded from disk
    """

    distance_criteria = ["primitives"]

    def __init__(self, model_description_md: str, primitives: dict):
        """Initialize ModelComparison instance."""
        self.description = model_description_md
        self.primitives = deepcopy(primitives)
        self.solutions = {}
        self.metrics = {}
        self.adapters = {}
        self.method_configs = {}
        self.simulation_results = {}
        self.computation_times = {}
        self.baseline_solutions = {}

        # Parameter translator
        self.param_translator = ParameterTranslator()

        # Economic metrics calculator
        self.metric_calculator = EconomicMetrics()

    def save_baseline_solution(
        self, method: str, filepath: str = None, include_adapter: bool = True
    ) -> str:
        """
        Save a baseline solution to disk for later comparison.

        Parameters
        ----------
        method : str
            Name of the method whose solution to save as baseline
        filepath : str, optional
            Path to save the baseline solution. If None, uses default naming
        include_adapter : bool
            Whether to include the adapter in the saved baseline

        Returns
        -------
        filepath : str
            Path where the baseline was saved

        Raises
        ------
        ValueError
            If the method has not been solved yet
        """
        if method not in self.solutions:
            raise ValueError(
                f"Method {method} has not been solved yet. "
                f"Solve it first using solve('{method}')."
            )

        # Create default filepath if not provided
        if filepath is None:
            # Clean method name for filename
            safe_method = method.replace("/", "_").replace("\\", "_")
            filepath = f"baseline_solution_{safe_method}.pkl"

        # Prepare baseline data
        baseline_data = {
            "method": method,
            "solution": deepcopy(self.solutions[method]),
            "primitives": deepcopy(self.primitives),
            "method_config": deepcopy(self.method_configs.get(method, {})),
            "computation_time": self.computation_times.get(method, None),
            "metrics": deepcopy(self.metrics.get(method, {})),
            "simulation_results": deepcopy(self.simulation_results.get(method, {})),
            "description": self.description,
            "timestamp": time.time(),
            "version_info": {
                "python_version": str(type(self).__module__),
                "class_name": type(self).__name__,
            },
        }

        # Include adapter if requested and available
        if include_adapter and method in self.adapters:
            baseline_data["adapter"] = deepcopy(self.adapters[method])

        # Save to disk
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(baseline_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Baseline solution for {method} saved to {filepath}")
        return str(filepath)

    def load_baseline_solution(self, filepath: str, baseline_name: str = None) -> dict:
        """
        Load a baseline solution from disk.

        Parameters
        ----------
        filepath : str
            Path to the saved baseline solution file
        baseline_name : str, optional
            Name to use for the loaded baseline. If None, uses original method name

        Returns
        -------
        baseline_data : dict
            Loaded baseline solution data

        Raises
        ------
        FileNotFoundError
            If the baseline file doesn't exist
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Baseline solution file not found: {filepath}")

        # Load baseline data
        with open(filepath, "rb") as f:
            baseline_data = pickle.load(f)

        # Determine baseline name
        if baseline_name is None:
            baseline_name = f"baseline_{baseline_data['method']}"

        # Store in baseline_solutions
        self.baseline_solutions[baseline_name] = baseline_data

        # Optionally restore adapter if it was saved
        if "adapter" in baseline_data:
            self.adapters[baseline_name] = baseline_data["adapter"]

        # Add to solutions for comparison
        self.solutions[baseline_name] = baseline_data["solution"]
        self.metrics[baseline_name] = baseline_data.get("metrics", {})
        self.simulation_results[baseline_name] = baseline_data.get(
            "simulation_results", {}
        )
        self.computation_times[baseline_name] = baseline_data.get(
            "computation_time", 0.0
        )

        print(f"Baseline solution loaded from {filepath} as '{baseline_name}'")

        # Check if primitives match
        if baseline_data["primitives"] != self.primitives:
            warnings.warn(
                "Loaded baseline has different primitives than current model. "
                "This may affect comparison validity."
            )

        return baseline_data

    def compare_against_baseline(
        self,
        baseline_name: str,
        methods: List[str] = None,
        metrics_to_compare: List[str] = None,
    ) -> pd.DataFrame:
        """
        Compare methods against a loaded baseline solution.

        Parameters
        ----------
        baseline_name : str
            Name of the baseline solution to compare against
        methods : list of str, optional
            Methods to compare. If None, compares all solved methods
        metrics_to_compare : list of str, optional
            Specific metrics to compare. If None, compares all available metrics

        Returns
        -------
        comparison_df : pd.DataFrame
            DataFrame with comparison results
        """
        if baseline_name not in self.baseline_solutions:
            raise ValueError(
                f"Baseline '{baseline_name}' not found. "
                f"Load it first using load_baseline_solution()."
            )

        # Get methods to compare
        if methods is None:
            methods = [m for m in self.solutions.keys() if m != baseline_name]
        else:
            methods = [m for m in methods if m in self.solutions and m != baseline_name]

        if not methods:
            warnings.warn("No methods available for comparison against baseline.")
            return pd.DataFrame()

        # Ensure metrics are computed for all methods
        all_methods = [baseline_name] + methods
        for method in all_methods:
            if method not in self.metrics or not self.metrics[method]:
                self.compute_metrics(method)

        # Create comparison DataFrame
        comparison_data = []
        baseline_metrics = self.metrics[baseline_name]

        for method in methods:
            method_metrics = self.metrics[method]
            row = {"method": method, "baseline": baseline_name}

            # Add computation time comparison
            baseline_time = self.computation_times.get(baseline_name, 0.0)
            method_time = self.computation_times.get(method, 0.0)
            row["computation_time"] = method_time
            row["baseline_computation_time"] = baseline_time
            if baseline_time > 0:
                row["time_ratio"] = method_time / baseline_time

            # Compare metrics
            if metrics_to_compare is None:
                # Use all common metrics
                common_metrics = set(baseline_metrics.keys()) & set(
                    method_metrics.keys()
                )
            else:
                common_metrics = set(metrics_to_compare)

            for metric in common_metrics:
                if metric in baseline_metrics and metric in method_metrics:
                    baseline_val = baseline_metrics[metric]
                    method_val = method_metrics[metric]

                    row[f"{metric}"] = method_val
                    row[f"{metric}_baseline"] = baseline_val

                    # Calculate relative difference
                    if baseline_val != 0:
                        row[f"{metric}_rel_diff"] = (
                            method_val - baseline_val
                        ) / baseline_val
                    else:
                        row[f"{metric}_abs_diff"] = method_val - baseline_val

            comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def list_baselines(self) -> pd.DataFrame:
        """
        List all loaded baseline solutions.

        Returns
        -------
        baselines_df : pd.DataFrame
            DataFrame with information about loaded baselines
        """
        if not self.baseline_solutions:
            print("No baseline solutions loaded.")
            return pd.DataFrame()

        baselines_data = []
        for name, data in self.baseline_solutions.items():
            row = {
                "baseline_name": name,
                "original_method": data["method"],
                "timestamp": pd.to_datetime(data["timestamp"], unit="s"),
                "computation_time": data.get("computation_time", "N/A"),
                "has_metrics": bool(data.get("metrics", {})),
                "has_simulation": bool(data.get("simulation_results", {})),
                "has_adapter": name in self.adapters,
            }
            baselines_data.append(row)

        return pd.DataFrame(baselines_data)

    def clear_baselines(self, baseline_names: List[str] = None):
        """
        Clear loaded baseline solutions from memory.

        Parameters
        ----------
        baseline_names : list of str, optional
            Names of baselines to clear. If None, clears all baselines
        """
        if baseline_names is None:
            baseline_names = list(self.baseline_solutions.keys())

        for name in baseline_names:
            if name in self.baseline_solutions:
                del self.baseline_solutions[name]

                # Also remove from other dictionaries
                if name in self.solutions:
                    del self.solutions[name]
                if name in self.metrics:
                    del self.metrics[name]
                if name in self.simulation_results:
                    del self.simulation_results[name]
                if name in self.adapters:
                    del self.adapters[name]
                if name in self.computation_times:
                    del self.computation_times[name]

        print(f"Cleared {len(baseline_names)} baseline solution(s)")

    def add_solution_method(self, name: str, method_config: dict):
        """
        Add a solution method configuration.

        Parameters
        ----------
        name : str
            Name of the solution method
        method_config : dict
            Method-specific parameters (grids, neural net config, etc.)
        """
        self.method_configs[name] = deepcopy(method_config)

    def solve(self, method: str, verbose: bool = True) -> dict:
        """
        Solve using specified method.

        Available methods:
        - 'krusell_smith/HARK': Fixed-point iteration
        - 'maliar_winant_euler': Deep learning Euler equation
        - 'maliar_winant_bellman': Deep learning Bellman equation
        - 'maliar_winant_reward': Deep learning lifetime reward
        - 'ssj': Sequence Space Jacobian
        - 'aiyagari/HARK': Standard Aiyagari solution

        Parameters
        ----------
        method : str
            Solution method identifier
        verbose : bool
            Whether to print progress information

        Returns
        -------
        solution : dict
            Solution dictionary with method-specific results
        """
        if verbose:
            print(f"Solving model using {method}...")

        start_time = time.time()

        # Get method configuration
        if method not in self.method_configs:
            raise ValueError(
                f"Method configuration for {method} not found. "
                f"Please add configuration using add_solution_method()."
            )

        # Create adapter for the method
        adapter = self._create_adapter(method)
        self.adapters[method] = adapter

        # Translate parameters for this method
        method_params = self.param_translator.translate(
            self.primitives, method, self.method_configs[method]
        )

        # Solve using adapter
        try:
            solution = adapter.solve(method_params)
            self.solutions[method] = solution

            # Record computation time
            self.computation_times[method] = time.time() - start_time

            if verbose:
                print(
                    f"Solution completed in {self.computation_times[method]:.2f} seconds"
                )

        except Exception as e:
            warnings.warn(f"Solution method {method} failed with error: {str(e)}")
            raise

        return solution

    def simulate_policy(
        self,
        method: str = None,
        periods: int = 1000,
        num_agents: int = 10000,
        burn_in: int = 100,
        seed: int = 0,
    ) -> Dict[str, pd.DataFrame]:
        """
        Simulate the model forward using the solved policy function.

        Parameters
        ----------
        method : str, optional
            Which solution method to simulate (if None, simulate all)
        periods : int
            Number of periods to simulate
        num_agents : int
            Number of agents in simulation
        burn_in : int
            Number of burn-in periods to discard
        seed : int
            Random seed for simulation

        Returns
        -------
        results : dict
            Dictionary mapping method names to DataFrames with simulation results
        """
        methods_to_simulate = [method] if method else list(self.solutions.keys())

        for method_name in methods_to_simulate:
            if method_name not in self.solutions:
                warnings.warn(
                    f"No solution found for {method_name}, skipping simulation"
                )
                continue

            adapter = self.adapters[method_name]

            # Run simulation
            sim_results = adapter.simulate(
                periods=periods + burn_in, num_agents=num_agents, seed=seed
            )

            # Discard burn-in and store results
            for key in sim_results:
                if hasattr(sim_results[key], "__len__"):
                    sim_results[key] = sim_results[key][burn_in:]

            self.simulation_results[method_name] = sim_results

        return self.simulation_results

    def compute_metrics(self, method: str = None) -> Dict[str, Dict[str, float]]:
        """
        Compute error metrics for solution(s).

        Parameters
        ----------
        method : str, optional
            Which method to compute metrics for (if None, compute for all)

        Returns
        -------
        metrics : dict
            Dictionary mapping method names to metric dictionaries
        """
        methods_to_evaluate = [method] if method else list(self.solutions.keys())

        for method_name in methods_to_evaluate:
            if method_name not in self.solutions:
                warnings.warn(f"No solution found for {method_name}, skipping metrics")
                continue

            adapter = self.adapters[method_name]
            solution = self.solutions[method_name]

            # Initialize metrics dict for this method
            method_metrics = {}

            # Compute Euler equation errors if applicable
            if (
                hasattr(adapter, "get_consumption_policy")
                and adapter.get_consumption_policy() is not None
            ):
                try:
                    euler_errors = self.metric_calculator.euler_equation_error(
                        adapter.get_consumption_policy(),
                        self.primitives,
                        adapter.get_test_points(),
                    )
                    method_metrics["max_euler_error"] = np.max(euler_errors)
                    method_metrics["mean_euler_error"] = np.mean(euler_errors)
                    method_metrics["euler_error_percentiles"] = {
                        "50": np.percentile(euler_errors, 50),
                        "90": np.percentile(euler_errors, 90),
                        "99": np.percentile(euler_errors, 99),
                    }
                except Exception as e:
                    warnings.warn(
                        f"Could not compute Euler errors for {method_name}: {str(e)}"
                    )

            # Compute Bellman equation errors if value function available
            if (
                hasattr(adapter, "get_value_function")
                and adapter.get_value_function() is not None
            ):
                try:
                    bellman_errors = self.metric_calculator.bellman_equation_error(
                        adapter.get_value_function(),
                        adapter.get_consumption_policy(),
                        self.primitives,
                        adapter.get_test_points(),
                    )
                    method_metrics["max_bellman_error"] = np.max(bellman_errors)
                    method_metrics["mean_bellman_error"] = np.mean(bellman_errors)
                except Exception as e:
                    warnings.warn(
                        f"Could not compute Bellman errors for {method_name}: {str(e)}"
                    )

            # Compute Den Haan-Marcet statistics if we have simulations and aggregate law
            if (
                method_name in self.simulation_results
                and hasattr(adapter, "get_aggregate_law")
                and adapter.get_aggregate_law() is not None
            ):
                try:
                    dhm_stat = self.metric_calculator.den_haan_marcet_statistic(
                        self.simulation_results[method_name],
                        adapter.get_aggregate_law(),
                    )
                    method_metrics["den_haan_marcet_r2"] = dhm_stat
                except Exception as e:
                    warnings.warn(
                        f"Could not compute DH-M statistic for {method_name}: {str(e)}"
                    )

            # Compute wealth distribution metrics if we have simulations
            if method_name in self.simulation_results:
                try:
                    if "assets" in self.simulation_results[method_name]:
                        wealth_metrics = (
                            self.metric_calculator.wealth_distribution_metrics(
                                self.simulation_results[method_name]["assets"]
                            )
                        )
                        method_metrics.update(wealth_metrics)
                except Exception as e:
                    warnings.warn(
                        f"Could not compute wealth metrics for {method_name}: {str(e)}"
                    )

            # Add computation time
            method_metrics["computation_time"] = self.computation_times.get(
                method_name, np.nan
            )

            # Store metrics
            self.metrics[method_name] = method_metrics

        return self.metrics

    def compare_solutions(
        self,
        methods: List[str] = None,
        save_report: bool = False,
        report_path: str = "comparison_report.md",
    ) -> pd.DataFrame:
        """
        Generate comparison report across methods.

        Parameters
        ----------
        methods : list, optional
            Methods to compare (if None, compare all solved)
        save_report : bool
            Whether to save a markdown report
        report_path : str
            Path to save the report

        Returns
        -------
        comparison : pd.DataFrame
            DataFrame with comparison metrics
        """
        methods_to_compare = methods if methods else list(self.solutions.keys())

        # Ensure metrics are computed
        for method in methods_to_compare:
            if method not in self.metrics:
                self.compute_metrics(method)

        # Create comparison DataFrame
        comparison_data = []
        for method in methods_to_compare:
            if method in self.metrics:
                row = {"method": method}
                row.update(self._flatten_metrics(self.metrics[method]))
                comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        # Generate markdown report if requested
        if save_report:
            self._generate_markdown_report(comparison_df, report_path)

        return comparison_df

    def _create_adapter(self, method: str):
        """Factory method to create appropriate adapter."""
        # Import adapters here to avoid circular imports
        from .adapters import (
            HARKAdapter,
            SSJAdapter,
            MaliarAdapter,
            ExternalAdapter,
            AiyagariAdapter,
        )

        adapter_map = {
            "krusell_smith/HARK": HARKAdapter,
            "ssj": SSJAdapter,
            "maliar_winant_euler": MaliarAdapter,
            "maliar_winant_bellman": MaliarAdapter,
            "maliar_winant_reward": MaliarAdapter,
            "aiyagari/HARK": AiyagariAdapter,
            "external": ExternalAdapter,
        }

        adapter_class = adapter_map.get(method)
        if adapter_class is None:
            raise ValueError(f"Unknown method: {method}")

        return adapter_class(
            method_type=method.split("/")[-1] if "/" in method else method
        )

    def _flatten_metrics(self, metrics_dict: dict, prefix: str = "") -> dict:
        """Flatten nested metrics dictionary for DataFrame creation."""
        flat_dict = {}
        for key, value in metrics_dict.items():
            if isinstance(value, dict):
                flat_dict.update(self._flatten_metrics(value, f"{prefix}{key}_"))
            else:
                flat_dict[f"{prefix}{key}"] = value
        return flat_dict

    def _generate_markdown_report(self, comparison_df: pd.DataFrame, path: str):
        """Generate a markdown comparison report."""
        with open(path, "w") as f:
            f.write("# Model Comparison Report\n\n")
            f.write(f"## Model Description\n\n{self.description}\n\n")
            f.write("## Primitive Parameters\n\n")
            f.write("```python\n")
            for key, value in self.primitives.items():
                f.write(f"{key}: {value}\n")
            f.write("```\n\n")
            f.write("## Comparison Results\n\n")
            f.write(comparison_df.to_markdown(index=False))
            f.write("\n\n## Detailed Metrics by Method\n\n")
            for method, metrics in self.metrics.items():
                f.write(f"### {method}\n\n")
                for metric_name, metric_value in metrics.items():
                    f.write(f"- **{metric_name}**: {metric_value}\n")
                f.write("\n")

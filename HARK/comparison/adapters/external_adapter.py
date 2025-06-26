"""
Adapter for external solution methods.
"""

from typing import Callable, Optional, Union
import numpy as np
from copy import deepcopy
import pickle
import json
import warnings
from pathlib import Path

from .base_adapter import SolutionAdapter


class ExternalAdapter(SolutionAdapter):
    """
    Adapter for loading solutions from external sources.

    This adapter allows loading pre-computed solutions from files
    (pickle, numpy, JSON) or from external repositories/URLs.
    """

    def __init__(self, method_type: str = "external"):
        """Initialize external adapter."""
        super().__init__(method_type)
        self.external_solution = None
        self.solution_metadata = {}

    def solve(self, params: dict) -> dict:
        """
        Load solution from external source.

        Parameters
        ----------
        params : dict
            Must contain 'solution_path' or 'solution_data'

        Returns
        -------
        solution : dict
            Loaded solution data
        """
        self.params = deepcopy(params)

        if "solution_path" in params:
            self.external_solution = self._load_from_file(params["solution_path"])
        elif "solution_data" in params:
            self.external_solution = params["solution_data"]
        elif "github_url" in params:
            self.external_solution = self._load_from_github(params["github_url"])
        else:
            raise ValueError(
                "External adapter requires 'solution_path', "
                "'solution_data', or 'github_url' in params"
            )

        # Extract metadata if available
        if isinstance(self.external_solution, dict):
            self.solution_metadata = {
                k: v
                for k, v in self.external_solution.items()
                if k.startswith("meta_") or k in ["description", "method", "source"]
            }

        self.solution = self.external_solution
        return self.solution

    def _load_from_file(self, path: Union[str, Path]) -> dict:
        """Load solution from file."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Solution file not found: {path}")

        # Determine file type and load
        if path.suffix == ".pkl" or path.suffix == ".pickle":
            with open(path, "rb") as f:
                data = pickle.load(f)
        elif path.suffix == ".npz":
            data = dict(np.load(path))
        elif path.suffix == ".json":
            with open(path, "r") as f:
                data = json.load(f)
        elif path.suffix == ".npy":
            data = {"array": np.load(path)}
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        return data

    def _load_from_github(self, url: str) -> dict:
        """Load solution from GitHub or URL."""
        try:
            import requests
        except ImportError:
            raise ImportError("requests package required for loading from URLs")

        warnings.warn(f"Loading solution from URL: {url}")

        # Download the file
        response = requests.get(url)
        response.raise_for_status()

        # Try to parse as JSON first
        try:
            data = response.json()
        except:
            # Otherwise save temporarily and load
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name

            # Guess file type from URL
            if ".pkl" in url or ".pickle" in url:
                with open(tmp_path, "rb") as f:
                    data = pickle.load(f)
            else:
                warnings.warn("Could not determine file type from URL")
                data = {"raw_content": response.content}

        return data

    def get_consumption_policy(self) -> Optional[Callable]:
        """Extract consumption policy from external solution."""
        if self.external_solution is None:
            return None

        # Try different possible keys
        policy_keys = ["consumption_policy", "cFunc", "policy", "c_policy"]

        for key in policy_keys:
            if key in self.external_solution:
                policy_data = self.external_solution[key]

                # If it's already a function, return it
                if callable(policy_data):
                    return policy_data

                # If it's an array, create interpolator
                elif isinstance(policy_data, (np.ndarray, list)):
                    return self._create_interpolator(policy_data)

                # If it's a dict with grid and values
                elif isinstance(policy_data, dict):
                    if "grid" in policy_data and "values" in policy_data:
                        return self._create_interpolator(
                            policy_data["values"], policy_data["grid"]
                        )

        return None

    def get_value_function(self) -> Optional[Callable]:
        """Extract value function from external solution."""
        if self.external_solution is None:
            return None

        # Try different possible keys
        value_keys = ["value_function", "vFunc", "value", "v_func"]

        for key in value_keys:
            if key in self.external_solution:
                value_data = self.external_solution[key]

                if callable(value_data):
                    return value_data
                elif isinstance(value_data, (np.ndarray, list)):
                    return self._create_interpolator(value_data)
                elif isinstance(value_data, dict):
                    if "grid" in value_data and "values" in value_data:
                        return self._create_interpolator(
                            value_data["values"], value_data["grid"]
                        )

        return None

    def get_aggregate_law(self) -> Optional[Callable]:
        """Extract aggregate law from external solution."""
        if self.external_solution is None:
            return None

        # Try different possible keys
        agg_keys = ["aggregate_law", "AFunc", "agg_law", "forecast_rule"]

        for key in agg_keys:
            if key in self.external_solution:
                agg_data = self.external_solution[key]

                if callable(agg_data):
                    return agg_data

                # If it's coefficients for a polynomial
                elif isinstance(agg_data, (list, np.ndarray)):
                    coeffs = np.asarray(agg_data)
                    if coeffs.ndim == 1:
                        # Simple polynomial
                        def poly_law(k, state=0):
                            return np.polyval(coeffs, k)

                        return poly_law
                    else:
                        # State-contingent polynomial
                        def state_poly_law(k, state=0):
                            return np.polyval(coeffs[int(state)], k)

                        return state_poly_law

        return None

    def simulate(self, periods: int, num_agents: int, seed: int = 0) -> dict:
        """Load or generate simulation from external solution."""
        if self.external_solution is None:
            raise ValueError("Must load solution before simulating")

        # Check if simulation data is included
        if "simulation" in self.external_solution:
            return self.external_solution["simulation"]

        # Otherwise try to run simulation with available policy
        policy = self.get_consumption_policy()
        if policy is None:
            warnings.warn("No policy function found for simulation")
            return {"periods": periods, "num_agents": num_agents}

        # Simple simulation placeholder
        np.random.seed(seed)

        # Initialize
        m = 10 * np.ones(num_agents) + np.random.randn(num_agents)
        c_history = np.zeros((periods, num_agents))
        a_history = np.zeros((periods, num_agents))

        for t in range(periods):
            c = policy(m)
            a = m - c

            c_history[t] = c
            a_history[t] = a

            # Simple dynamics
            m = 1.03 * a + np.exp(np.random.randn(num_agents) * 0.1)

        return {
            "consumption": c_history,
            "assets": a_history,
            "cNrm": c_history,
            "aNrm": a_history,
        }

    def _create_interpolator(self, values, grid=None):
        """Create interpolation function from grid data."""
        from HARK.interpolation import LinearInterp

        values = np.asarray(values)

        if grid is None:
            # Create default grid
            grid = np.linspace(0.1, 50.0, len(values))
        else:
            grid = np.asarray(grid)

        # Create interpolator
        if values.ndim == 1:
            return LinearInterp(grid, values)
        else:
            # Multi-dimensional - return first slice as placeholder
            warnings.warn("Multi-dimensional interpolation simplified to 1D")
            return LinearInterp(grid, values[0])

    def get_test_points(self, n_points: int = 1000) -> dict:
        """Generate test points based on available grids."""
        if self.external_solution is not None:
            # Look for grid information
            if "state_grid" in self.external_solution:
                grid = self.external_solution["state_grid"]
                if len(grid) >= n_points:
                    indices = np.linspace(0, len(grid) - 1, n_points, dtype=int)
                    return {"mNrm": grid[indices]}
                else:
                    return {"mNrm": grid}

        return super().get_test_points(n_points)

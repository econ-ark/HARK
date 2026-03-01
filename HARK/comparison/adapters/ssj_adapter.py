"""
Adapter for Sequence Space Jacobian (SSJ) solution method.
"""

from typing import Callable, Optional
import numpy as np
from copy import deepcopy
import warnings

from .base_adapter import SolutionAdapter


class SSJAdapter(SolutionAdapter):
    """
    Adapter for Sequence Space Jacobian solution method.

    This adapter interfaces with the SSJ toolkit to solve heterogeneous
    agent models using the Jacobian-based approach.
    """

    def __init__(self, method_type: str = "ssj"):
        """Initialize SSJ adapter."""
        super().__init__(method_type)
        self.model = None
        self.steady_state = None
        self.jacobian = None

    def solve(self, params: dict) -> dict:
        """
        Solve using SSJ method.

        Parameters
        ----------
        params : dict
            Model parameters in SSJ format

        Returns
        -------
        solution : dict
            Solution dictionary with SSJ results
        """
        self.params = deepcopy(params)

        try:
            # Import SSJ components (may not be installed)
            # from sequence_jacobian import create_model, SteadyStateDict
            from HARK.ConsumptionSaving.ConsNewKeynesianModel import (
                NewKeynesianConsumerType,
            )
        except ImportError:
            warnings.warn(
                "sequence_jacobian package not found. SSJ adapter requires it."
            )
            raise ImportError(
                "Please install sequence_jacobian: pip install sequence-jacobian"
            )

        # Create HANK agent for SSJ
        hank_dict = self._create_hank_dict(params)
        self.agent = NewKeynesianConsumerType(**hank_dict)

        # Create SSJ model components
        blocks = self._create_ssj_blocks(params)

        # Create the model
        self.model = create_model(blocks, name="KrusellSmith")

        # Find steady state
        calib_params = self._get_calibration_params(params)
        unknowns = {"beta": 0.98}  # Discount factor to target
        targets = {"r": params.get("r_ss", 0.01)}  # Interest rate target

        self.steady_state = self.model.solve_steady_state(
            calib_params, unknowns, targets, solver="broyden_custom"
        )

        # Compute Jacobians
        exogenous = ["Z"]  # TFP shock
        unknowns = ["r"]  # Interest rate
        targets = ["asset_mkt"]  # Asset market clearing

        T = 300  # Time horizon for Jacobians
        self.jacobian = self.model.jacobian(
            self.steady_state, exogenous, unknowns, targets, T=T
        )

        # Package solution
        solution = {
            "model": self.model,
            "steady_state": self.steady_state,
            "jacobian": self.jacobian,
            "agent": self.agent,
        }

        self.solution = solution
        return solution

    def _create_hank_dict(self, params: dict) -> dict:
        """Create HANK agent parameters from SSJ parameters."""
        hank_dict = {
            "CRRA": 1.0 / params.get("eis", 0.5),  # EIS to CRRA
            "DiscFac": params.get("beta", 0.98),
            "LivPrb": [0.99375],
            "PermGroFac": [1.00],
            "PermShkStd": [0.06],
            "PermShkCount": 5,
            "TranShkStd": [0.2],
            "TranShkCount": 5,
            "UnempPrb": 0.0,
            "IncUnemp": 0.0,
            "Rfree": [1 + params.get("r_ss", 0.01)],
            "wage": [params.get("w_ss", 0.7)],
            "tax_rate": [0.0],
            "labor": [params.get("L_ss", 1.0)],
            "BoroCnstArt": 0.0,
            "aXtraMin": 0.001,
            "aXtraMax": 50,
            "aXtraCount": 50,
            "aXtraExtra": None,
            "vFuncBool": False,
            "CubicBool": False,
        }

        # Update with any provided parameters
        for key in ["PermShkStd", "TranShkStd", "aXtraMax", "aXtraCount"]:
            if key in params:
                hank_dict[key] = params[key]

        return hank_dict

    def _create_ssj_blocks(self, params: dict) -> list:
        """Create SSJ model blocks."""
        # This is a simplified version - full implementation would need
        # proper block definitions
        blocks = []

        # Household block (using HANK agent)
        household = {
            "name": "household",
            "agent": self.agent,
            "inputs": ["r", "w", "Z"],
            "outputs": ["C", "A"],
        }
        blocks.append(household)

        # Firm block
        firm = {
            "name": "firm",
            "inputs": ["Z", "K", "L"],
            "outputs": ["Y", "r", "w"],
            "parameters": {
                "alpha": params.get("alpha", 0.36),
                "delta": params.get("delta", 0.025),
            },
        }
        blocks.append(firm)

        # Market clearing
        market = {
            "name": "market",
            "equations": {"asset_mkt": "A - K", "goods_mkt": "Y - C - delta * K"},
        }
        blocks.append(market)

        return blocks

    def _get_calibration_params(self, params: dict) -> dict:
        """Extract calibration parameters for SSJ."""
        return {
            "eis": params.get("eis", 0.5),
            "beta": params.get("beta", 0.98),
            "alpha": params.get("alpha", 0.36),
            "delta": params.get("delta", 0.025),
            "L_ss": params.get("L_ss", 1.0),
            "Z": 1.0,  # Normalized TFP
        }

    def get_consumption_policy(self) -> Optional[Callable]:
        """Return consumption policy function."""
        if self.agent is not None and hasattr(self.agent, "solution"):
            return self.agent.solution[0].cFunc
        return None

    def get_value_function(self) -> Optional[Callable]:
        """Return value function if available."""
        # SSJ typically doesn't compute value functions
        return None

    def get_aggregate_law(self) -> Optional[Callable]:
        """Return aggregate law of motion."""
        if self.jacobian is not None:
            # Create a linear approximation using Jacobians
            def aggregate_forecast(k, z_shock=0):
                # Simplified - use Jacobian for linear forecast
                k_ss = self.steady_state["K"]
                deviation = k - k_ss
                # This is highly simplified - real implementation would
                # use full Jacobian dynamics
                return k_ss + 0.9 * deviation + 0.1 * z_shock

            return aggregate_forecast
        return None

    def simulate(self, periods: int, num_agents: int, seed: int = 0) -> dict:
        """Run simulation using SSJ methods."""
        if self.model is None:
            raise ValueError("Must solve model before simulating")

        # SSJ simulation would typically use the model's built-in methods
        # This is a simplified placeholder
        np.random.seed(seed)

        # Generate aggregate shocks
        z_shocks = np.random.normal(0, 0.01, periods)

        # Simulate aggregate dynamics using impulse responses
        # This is highly simplified - real SSJ would use full model
        k_path = np.zeros(periods)
        k_path[0] = self.steady_state["K"]

        for t in range(1, periods):
            k_path[t] = (
                0.95 * k_path[t - 1] + 0.05 * self.steady_state["K"] + z_shocks[t]
            )

        # For individual simulations, would need to use the agent
        # This is a placeholder
        results = {
            "aggregate_capital": k_path,
            "z_shocks": z_shocks,
            "periods": periods,
        }

        return results

    def get_test_points(self, n_points: int = 1000) -> dict:
        """Generate test points for SSJ model."""
        if self.steady_state is not None:
            # Use points around steady state
            k_ss = self.steady_state.get("K", 10.0)
            k_min = 0.5 * k_ss
            k_max = 1.5 * k_ss

            return {
                "K": np.linspace(k_min, k_max, n_points),
                "Z": np.ones(n_points),  # Normalized TFP
            }
        else:
            return super().get_test_points(n_points)

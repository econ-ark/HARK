"""
Adapter for Aiyagari model solution methods.
"""

from typing import Callable, Optional
import numpy as np
from copy import deepcopy
import warnings

from .base_adapter import SolutionAdapter
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType


class AiyagariAdapter(SolutionAdapter):
    """
    Adapter for Aiyagari (1994) model solutions.

    This adapter handles the standard incomplete markets model
    with idiosyncratic income risk and borrowing constraints.
    """

    def __init__(self, method_type: str = "aiyagari"):
        """Initialize Aiyagari adapter."""
        super().__init__(method_type)
        self.agent = None
        self.equilibrium = None

    def solve(self, params: dict) -> dict:
        """
        Solve Aiyagari model.

        Parameters
        ----------
        params : dict
            Model parameters including income process and preferences

        Returns
        -------
        solution : dict
            Solution with policy functions and equilibrium values
        """
        self.params = deepcopy(params)

        # Check if we need to find equilibrium interest rate
        if params.get("find_equilibrium", True):
            solution = self._solve_equilibrium(params)
        else:
            # Just solve at given interest rate
            solution = self._solve_partial(params)

        self.solution = solution
        return solution

    def _solve_partial(self, params: dict) -> dict:
        """Solve agent's problem at given interest rate."""
        # Create IndShockConsumerType agent
        agent_params = self._prepare_agent_params(params)
        self.agent = IndShockConsumerType(**agent_params)

        # Solve the agent's problem
        self.agent.solve()

        # Compute stationary distribution if requested
        if params.get("compute_distribution", True):
            self._compute_stationary_distribution()

        # Package solution
        solution = {
            "agent": self.agent,
            "cFunc": self.agent.solution[0].cFunc,
            "vFunc": self.agent.solution[0].vFunc
            if hasattr(self.agent.solution[0], "vFunc")
            else None,
            "Rfree": agent_params["Rfree"],
            "wage": params.get("wage", 1.0),
        }

        return solution

    def _solve_equilibrium(self, params: dict) -> dict:
        """Find equilibrium interest rate."""
        from scipy.optimize import brentq

        # Parameters for equilibrium
        target_K = params.get("target_capital", None)
        r_min = params.get("r_min", 0.001)
        r_max = params.get("r_max", 0.04)
        tol = params.get("equilibrium_tol", 1e-4)

        # Production function parameters
        alpha = params.get("CapShare", 0.36)
        delta = params.get("DeprFac", 0.08)

        def excess_demand(r):
            """Compute excess demand for capital at interest rate r."""
            # Set interest rate
            params_r = params.copy()
            params_r["Rfree"] = 1 + r
            params_r["find_equilibrium"] = False

            # Solve at this interest rate
            self._solve_partial(params_r)

            # Get aggregate capital supply from households
            K_supply = self._compute_aggregate_capital()

            # If target capital specified, use it
            if target_K is not None:
                K_demand = target_K
            else:
                # Otherwise compute from firm's FOC
                # r = alpha * (K/L)^(alpha-1) - delta
                # => K/L = ((r + delta) / alpha)^(1/(alpha-1))
                L = params.get("LbrInd", 1.0)
                K_L_ratio = ((r + delta) / alpha) ** (1 / (alpha - 1))
                K_demand = K_L_ratio * L

            return K_supply - K_demand

        # Find equilibrium interest rate
        try:
            r_eq = brentq(excess_demand, r_min, r_max, xtol=tol)
        except ValueError:
            warnings.warn(f"Could not find equilibrium in range [{r_min}, {r_max}]")
            r_eq = 0.02  # Default

        # Solve at equilibrium rate
        params_eq = params.copy()
        params_eq["Rfree"] = 1 + r_eq
        params_eq["find_equilibrium"] = False
        solution = self._solve_partial(params_eq)

        # Add equilibrium information
        solution["r_equilibrium"] = r_eq
        solution["K_equilibrium"] = self._compute_aggregate_capital()

        # Compute equilibrium wage from marginal product
        K = solution["K_equilibrium"]
        L = params.get("LbrInd", 1.0)
        solution["wage_equilibrium"] = (1 - alpha) * (K / L) ** alpha

        self.equilibrium = {"r": r_eq, "K": K, "w": solution["wage_equilibrium"]}

        return solution

    def _prepare_agent_params(self, params: dict) -> dict:
        """Prepare parameters for IndShockConsumerType."""
        # Map Aiyagari parameters to HARK parameters
        agent_params = {
            "CRRA": params.get("CRRA", 2.0),
            "DiscFac": params.get("DiscFac", 0.96),
            "LivPrb": [params.get("LivPrb", 1.0)],
            "PermGroFac": [params.get("PermGroFac", 1.0)],
            "Rfree": params.get("Rfree", 1.03),
            "BoroCnstArt": params.get("BoroCnstArt", 0.0),
            "vFuncBool": params.get("vFuncBool", True),
            "CubicBool": params.get("CubicBool", False),
            "T_cycle": 1,
            "cycles": 0,  # Infinite horizon
        }

        # Income process parameters
        if "TranShkStd" in params:
            agent_params["TranShkStd"] = [params["TranShkStd"]]
            agent_params["TranShkCount"] = params.get("TranShkCount", 7)
        else:
            # Default: no transitory shocks
            agent_params["TranShkStd"] = [0.0]
            agent_params["TranShkCount"] = 1

        if "PermShkStd" in params:
            agent_params["PermShkStd"] = [params["PermShkStd"]]
            agent_params["PermShkCount"] = params.get("PermShkCount", 7)
        else:
            # Default: small permanent shocks
            agent_params["PermShkStd"] = [0.01]
            agent_params["PermShkCount"] = 3

        # Asset grid
        agent_params["aXtraMin"] = params.get("aXtraMin", 0.001)
        agent_params["aXtraMax"] = params.get("aXtraMax", 50.0)
        agent_params["aXtraCount"] = params.get("aXtraCount", 48)
        agent_params["aXtraNestFac"] = params.get("aXtraNestFac", 3)

        # Unemployment parameters (if specified)
        if "UnempPrb" in params:
            agent_params["UnempPrb"] = params["UnempPrb"]
            agent_params["IncUnemp"] = params.get("IncUnemp", 0.0)
        else:
            agent_params["UnempPrb"] = 0.0
            agent_params["IncUnemp"] = 0.0

        return agent_params

    def _compute_stationary_distribution(self, sim_periods: int = 1000):
        """Compute stationary distribution of assets."""
        if self.agent is None:
            return

        # Set up simulation
        self.agent.T_sim = sim_periods
        if not hasattr(self.agent, "AgentCount") or self.agent.AgentCount < 10000:
            self.agent.AgentCount = 10000

        # Initialize and simulate
        self.agent.initialize_sim()
        self.agent.simulate()

        # Extract final period distribution
        self.asset_distribution = self.agent.history["aNrm"][-1]

    def _compute_aggregate_capital(self) -> float:
        """Compute aggregate capital from distribution."""
        if hasattr(self, "asset_distribution"):
            return np.mean(self.asset_distribution)
        elif self.agent is not None and hasattr(self.agent, "history"):
            # Use last period of simulation
            return np.mean(self.agent.history["aNrm"][-1])
        else:
            warnings.warn("No distribution available, using approximate calculation")
            # Approximate with policy function
            m_grid = np.linspace(0.1, 20.0, 1000)
            c_vals = self.agent.solution[0].cFunc(m_grid)
            a_vals = m_grid - c_vals
            return np.mean(a_vals)

    def get_consumption_policy(self) -> Optional[Callable]:
        """Return consumption policy function."""
        if self.agent is not None and hasattr(self.agent, "solution"):
            return self.agent.solution[0].cFunc
        return None

    def get_value_function(self) -> Optional[Callable]:
        """Return value function if available."""
        if self.agent is not None and hasattr(self.agent, "solution"):
            if hasattr(self.agent.solution[0], "vFunc"):
                return self.agent.solution[0].vFunc
        return None

    def get_aggregate_law(self) -> Optional[Callable]:
        """Aiyagari model has no aggregate shocks, so no aggregate law."""
        return None

    def simulate(self, periods: int, num_agents: int, seed: int = 0) -> dict:
        """Run simulation of Aiyagari model."""
        if self.agent is None:
            raise ValueError("Must solve model before simulating")

        # Set simulation parameters
        self.agent.T_sim = periods
        self.agent.AgentCount = num_agents
        self.agent.seed = seed

        # Initialize and simulate
        self.agent.initialize_sim()
        self.agent.simulate()

        # Extract results
        results = {
            "mNrm": self.agent.history["mNrm"],
            "cNrm": self.agent.history["cNrm"],
            "aNrm": self.agent.history["aNrm"],
            "assets": self.agent.history["aNrm"],
        }

        # Add equilibrium values if available
        if self.equilibrium is not None:
            results["r_equilibrium"] = self.equilibrium["r"]
            results["K_equilibrium"] = self.equilibrium["K"]
            results["w_equilibrium"] = self.equilibrium["w"]

        return results

    def get_test_points(self, n_points: int = 1000) -> dict:
        """Generate test points for Aiyagari model."""
        if self.agent is not None:
            # Use the asset grid bounds
            m_min = (
                self.agent.solution[0].mNrmMin
                if hasattr(self.agent.solution[0], "mNrmMin")
                else 0.1
            )
            m_max = self.agent.aXtraMax + 1.0  # Add some income

            return {"mNrm": np.linspace(m_min, m_max, n_points)}
        else:
            return super().get_test_points(n_points)

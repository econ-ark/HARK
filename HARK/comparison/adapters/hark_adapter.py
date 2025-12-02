"""
Adapter for native HARK solution methods.
"""

from typing import Callable, Optional
import numpy as np
from copy import deepcopy

from .base_adapter import SolutionAdapter
from HARK.ConsumptionSaving.ConsAggShockModel import (
    KrusellSmithType,
    KrusellSmithEconomy,
)
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType


class HARKAdapter(SolutionAdapter):
    """
    Adapter for HARK's native solution methods.

    This adapter wraps HARK's AgentType and Market classes to provide
    a uniform interface for the comparison framework.
    """

    def __init__(self, method_type: str = "HARK"):
        """Initialize HARK adapter."""
        super().__init__(method_type)
        self.agent = None
        self.economy = None

    def solve(self, params: dict) -> dict:
        """
        Solve using HARK's native methods.

        Parameters
        ----------
        params : dict
            Model parameters in HARK format

        Returns
        -------
        solution : dict
            Solution dictionary with HARK results
        """
        self.params = deepcopy(params)

        # Determine which model to solve
        if "krusell_smith" in self.method_type.lower():
            return self._solve_krusell_smith(params)
        elif "aiyagari" in self.method_type.lower():
            return self._solve_aiyagari(params)
        else:
            # Default to basic consumption-savings
            return self._solve_indshock(params)

    def _solve_krusell_smith(self, params: dict) -> dict:
        """Solve Krusell-Smith model using HARK."""
        # Create economy
        self.economy = KrusellSmithEconomy(
            agents=[],
            **{
                k: v
                for k, v in params.items()
                if k
                in [
                    "act_T",
                    "tolerance",
                    "DampingFac",
                    "T_discard",
                    "DiscFac",
                    "CRRA",
                    "LbrInd",
                    "CapShare",
                    "DeprFac",
                    "ProdB",
                    "ProdG",
                    "DurMeanB",
                    "DurMeanG",
                    "SpellMeanB",
                    "SpellMeanG",
                    "UrateB",
                    "UrateG",
                    "RelProbBG",
                    "RelProbGB",
                ]
            },
        )

        # Create agents
        agent_params = {
            k: v
            for k, v in params.items()
            if k in ["DiscFac", "CRRA", "LbrInd", "AgentCount"]
        }
        self.agent = KrusellSmithType(**agent_params)
        self.agent.get_economy_data(self.economy)
        self.agent.construct()

        # Add agents to economy
        self.economy.agents = [self.agent]

        # Solve the model
        self.economy.make_Mrkv_history()
        self.economy.solve()

        # Package solution
        solution = {
            "agent": self.agent,
            "economy": self.economy,
            "AFunc": self.economy.AFunc,
            "history": self.economy.history,
        }

        self.solution = solution
        return solution

    def _solve_aiyagari(self, params: dict) -> dict:
        """Solve Aiyagari model using HARK."""
        # Create agent
        agent_params = {
            k: v for k, v in params.items() if k not in ["act_T", "tolerance"]
        }
        self.agent = IndShockConsumerType(**agent_params)

        # Solve the agent's problem
        self.agent.solve()

        # For full Aiyagari, would need to find equilibrium interest rate
        # This is a simplified version
        solution = {
            "agent": self.agent,
            "solution": self.agent.solution,
            "cFunc": self.agent.solution[0].cFunc,
            "vFunc": self.agent.solution[0].vFunc
            if hasattr(self.agent.solution[0], "vFunc")
            else None,
        }

        self.solution = solution
        return solution

    def _solve_indshock(self, params: dict) -> dict:
        """Solve basic IndShock model."""
        self.agent = IndShockConsumerType(**params)
        self.agent.solve()

        solution = {
            "agent": self.agent,
            "solution": self.agent.solution,
            "cFunc": self.agent.solution[0].cFunc,
            "vFunc": self.agent.solution[0].vFunc
            if hasattr(self.agent.solution[0], "vFunc")
            else None,
        }

        self.solution = solution
        return solution

    def get_consumption_policy(self) -> Optional[Callable]:
        """Return consumption policy function."""
        if self.solution is None:
            return None

        if "cFunc" in self.solution:
            return self.solution["cFunc"]
        elif self.agent is not None and hasattr(self.agent, "solution"):
            # For KS model, return state-contingent policies
            if hasattr(self.agent.solution[0], "cFunc"):
                if isinstance(self.agent.solution[0].cFunc, list):
                    # Return function that selects based on state
                    def state_contingent_policy(m, state=0):
                        return self.agent.solution[0].cFunc[int(state)](m)

                    return state_contingent_policy
                else:
                    return self.agent.solution[0].cFunc
        return None

    def get_value_function(self) -> Optional[Callable]:
        """Return value function if available."""
        if self.solution is None:
            return None

        if "vFunc" in self.solution and self.solution["vFunc"] is not None:
            return self.solution["vFunc"]
        elif self.agent is not None and hasattr(self.agent, "solution"):
            if hasattr(self.agent.solution[0], "vFunc"):
                return self.agent.solution[0].vFunc
        return None

    def get_aggregate_law(self) -> Optional[Callable]:
        """Return aggregate law of motion if applicable."""
        if self.economy is not None and hasattr(self.economy, "AFunc"):
            # For KS model, AFunc is a list of functions for each state
            if isinstance(self.economy.AFunc, list):

                def aggregate_forecast(k, state):
                    return self.economy.AFunc[int(state)](k)

                return aggregate_forecast
            else:
                return self.economy.AFunc
        return None

    def simulate(self, periods: int, num_agents: int, seed: int = 0) -> dict:
        """Run simulation using HARK's simulation methods."""
        if self.agent is None:
            raise ValueError("Must solve model before simulating")

        # Set up simulation parameters
        self.agent.T_sim = periods
        self.agent.AgentCount = num_agents
        self.agent.seed = seed

        # Initialize and run simulation
        self.agent.initialize_sim()
        self.agent.simulate()

        # Extract results
        if hasattr(self.agent, "history") and self.agent.history:
            results = {
                "mNrm": self.agent.history.get("mNrm", np.zeros((periods, num_agents))),
                "cNrm": self.agent.history.get("cNrm", np.zeros((periods, num_agents))),
                "aNrm": self.agent.history.get("aNrm", np.zeros((periods, num_agents))),
            }
        else:
            # If simulation didn't produce history, create placeholder
            results = {
                "mNrm": np.zeros((periods, num_agents)),
                "cNrm": np.zeros((periods, num_agents)),
                "aNrm": np.zeros((periods, num_agents)),
            }

        # Add aggregate variables if available
        if hasattr(self.agent, "history"):
            if "AaggNow" in self.agent.history:
                results["AaggNow"] = self.agent.history["AaggNow"]
            if "MaggNow" in self.agent.history:
                results["MaggNow"] = self.agent.history["MaggNow"]

        # For market models, get economy history
        if self.economy is not None and hasattr(self.economy, "history"):
            results.update(
                {k: v for k, v in self.economy.history.items() if k not in results}
            )

        # Rename assets to standard name
        if "aNrm" in results:
            results["assets"] = results["aNrm"]

        return results

    def get_test_points(self, n_points: int = 1000) -> dict:
        """Generate test points based on the state space."""
        if self.agent is not None and hasattr(self.agent, "solution"):
            # Use the agent's grid if available
            if hasattr(self.agent.solution[0], "mNrmMin"):
                m_min = self.agent.solution[0].mNrmMin
            else:
                m_min = 0.1
            m_max = 50.0

            return {"mNrm": np.linspace(m_min, m_max, n_points)}
        else:
            return super().get_test_points(n_points)

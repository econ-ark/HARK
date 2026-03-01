"""
Base adapter class for solution methods.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional
import numpy as np


class SolutionAdapter(ABC):
    """
    Base class for adapting different solution methods.

    This abstract class defines the interface that all solution method
    adapters must implement to work with the ModelComparison framework.
    """

    def __init__(self, method_type: str = ""):
        """
        Initialize adapter.

        Parameters
        ----------
        method_type : str
            Type/variant of the method (e.g., 'euler', 'bellman', 'reward')
        """
        self.method_type = method_type
        self.solution = None
        self.params = None

    @abstractmethod
    def solve(self, params: dict) -> dict:
        """
        Solve the model using this method.

        Parameters
        ----------
        params : dict
            Model parameters in method-specific format

        Returns
        -------
        solution : dict
            Solution dictionary with method-specific results
        """
        pass

    @abstractmethod
    def get_consumption_policy(self) -> Optional[Callable]:
        """
        Return consumption policy function in standard format.

        Returns
        -------
        policy : callable or None
            Function mapping states to consumption decisions
        """
        pass

    @abstractmethod
    def get_value_function(self) -> Optional[Callable]:
        """
        Return value function if available.

        Returns
        -------
        value_func : callable or None
            Value function mapping states to values
        """
        pass

    @abstractmethod
    def get_aggregate_law(self) -> Optional[Callable]:
        """
        Return aggregate law of motion if applicable.

        Returns
        -------
        agg_law : callable or None
            Function predicting next period's aggregate state
        """
        pass

    @abstractmethod
    def simulate(self, periods: int, num_agents: int, seed: int = 0) -> dict:
        """
        Run simulation with common interface.

        Parameters
        ----------
        periods : int
            Number of periods to simulate
        num_agents : int
            Number of agents in simulation
        seed : int
            Random seed

        Returns
        -------
        results : dict
            Simulation results with standard keys
        """
        pass

    def get_test_points(self, n_points: int = 1000) -> dict:
        """
        Generate test points for metric evaluation.

        Parameters
        ----------
        n_points : int
            Number of test points to generate

        Returns
        -------
        test_points : dict
            Dictionary with test point arrays
        """
        # Default implementation - can be overridden
        if hasattr(self, "state_grid"):
            return {"mNrm": self.state_grid}
        else:
            # Generate default grid
            m_min = 0.1
            m_max = 50.0
            return {"mNrm": np.linspace(m_min, m_max, n_points)}

    def extract_solution_info(self) -> dict:
        """
        Extract standardized information from solution.

        Returns
        -------
        info : dict
            Dictionary with solution information
        """
        info = {"method_type": self.method_type, "solved": self.solution is not None}

        if self.solution is not None:
            # Add method-specific information
            if hasattr(self.solution, "time_to_solve"):
                info["solve_time"] = self.solution.time_to_solve
            if hasattr(self.solution, "iterations"):
                info["iterations"] = self.solution.iterations

        return info

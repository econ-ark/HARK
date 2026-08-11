"""
Solution classes for labeled consumption-saving models.

This module contains the value function and solution classes that use
xarray for labeled, multidimensional data handling.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import xarray as xr

from HARK.metric import MetricObject
from HARK.rewards import UtilityFuncCRRA

__all__ = [
    "ValueFuncCRRALabeled",
    "ConsumerSolutionLabeled",
]


class ValueFuncCRRALabeled(MetricObject):
    """
    Value function interpolation using xarray for labeled arrays.

    This class enables value function interpolation and derivative computation
    using xarray's labeled data structures. It stores the value function in
    inverse form for numerical stability with CRRA utility.

    Parameters
    ----------
    dataset : xr.Dataset
        Underlying dataset containing variables:
        - "v": value function
        - "v_der": marginal value function
        - "v_inv": inverse of value function
        - "v_der_inv": inverse of marginal value function
    CRRA : float
        Coefficient of relative risk aversion. Must be non-negative and finite.

    Raises
    ------
    ValueError
        If CRRA is negative or not finite.
    """

    def __init__(self, dataset: xr.Dataset, CRRA: float) -> None:
        if not np.isfinite(CRRA):
            raise ValueError(f"CRRA must be finite, got {CRRA}")
        if CRRA < 0:
            raise ValueError(f"CRRA must be non-negative, got {CRRA}")

        # Validate dataset structure
        if not isinstance(dataset, xr.Dataset):
            raise TypeError(f"dataset must be xr.Dataset, got {type(dataset)}")

        required_vars = {"v", "v_der", "v_inv", "v_der_inv"}
        missing_vars = required_vars - set(dataset.data_vars)
        if missing_vars:
            raise ValueError(
                f"Dataset missing required variables: {missing_vars}. "
                f"Required: {required_vars}"
            )

        self.dataset = dataset
        self.CRRA = CRRA
        self.u = UtilityFuncCRRA(CRRA)

    def __call__(self, state: Mapping[str, np.ndarray]) -> xr.DataArray:
        """
        Evaluate value function at given state via interpolation.

        Interpolates the inverse value function and then inverts to get
        the value function, which is more numerically stable for CRRA utility.

        Parameters
        ----------
        state : Mapping[str, np.ndarray]
            State variables to evaluate value function at. Must contain
            all coordinates present in the dataset.

        Returns
        -------
        xr.DataArray
            Value function evaluated at the given state.

        Raises
        ------
        KeyError
            If state is missing required coordinates.
        """
        state_dict = self._validate_state(state)

        result = self.u(
            self.dataset["v_inv"].interp(
                state_dict,
                assume_sorted=True,
                kwargs={"fill_value": "extrapolate"},
            )
        )

        result.name = "v"
        result.attrs = self.dataset["v"].attrs

        return result

    def derivative(self, state: Mapping[str, np.ndarray]) -> xr.DataArray:
        """
        Evaluate marginal value function at given state via interpolation.

        Interpolates the inverse marginal value function and then inverts
        to get the marginal value function.

        Parameters
        ----------
        state : Mapping[str, np.ndarray]
            State variables to evaluate marginal value function at.

        Returns
        -------
        xr.DataArray
            Marginal value function evaluated at the given state.

        Raises
        ------
        KeyError
            If state is missing required coordinates.
        """
        state_dict = self._validate_state(state)

        result = self.u.der(
            self.dataset["v_der_inv"].interp(
                state_dict,
                assume_sorted=True,
                kwargs={"fill_value": "extrapolate"},
            )
        )

        result.name = "v_der"
        result.attrs = self.dataset["v"].attrs

        return result

    def evaluate(self, state: Mapping[str, np.ndarray]) -> xr.Dataset:
        """
        Interpolate all data variables in the dataset at given state.

        Parameters
        ----------
        state : Mapping[str, np.ndarray]
            State variables to evaluate all data variables at.

        Returns
        -------
        xr.Dataset
            All interpolated data variables at the given state.

        Raises
        ------
        KeyError
            If state is missing required coordinates.
        """
        state_dict = self._validate_state(state)

        result = self.dataset.interp(
            state_dict,
            kwargs={"fill_value": None},
        )
        result.attrs = self.dataset["v"].attrs

        return result

    def _validate_state(self, state: Mapping[str, np.ndarray]) -> dict:
        """
        Validate state and extract required coordinates.

        Parameters
        ----------
        state : Mapping[str, np.ndarray]
            State to validate. Must be a dict or xr.Dataset.

        Returns
        -------
        dict
            Dictionary containing only the coordinates present in both
            the dataset and the input state.

        Raises
        ------
        TypeError
            If state is not a dict or xr.Dataset.
        KeyError
            If a required coordinate is missing from state.
        """
        if not isinstance(state, (xr.Dataset, dict)):
            raise TypeError(f"state must be a dict or xr.Dataset, got {type(state)}")

        state_dict = {}
        for coord in self.dataset.coords.keys():
            if coord not in state:
                raise KeyError(
                    f"Required coordinate '{coord}' not found in state. "
                    f"Available keys: {list(state.keys())}"
                )
            state_dict[coord] = state[coord]

        return state_dict


class ConsumerSolutionLabeled(MetricObject):
    """
    Solution to a labeled consumption-saving problem.

    This class represents the complete solution to a one-period
    consumption-saving problem, containing the value function,
    policy function (consumption), and continuation value function.

    Parameters
    ----------
    value : ValueFuncCRRALabeled
        Value function for this period.
    policy : xr.Dataset
        Policy function (consumption as function of state).
    continuation : ValueFuncCRRALabeled or None
        Continuation value function (value of post-decision state).
        Can be None for terminal period solutions.
    attrs : dict, optional
        Additional attributes of the solution, such as minimum
        normalized market resources. Default is None.
    """

    def __init__(
        self,
        value: ValueFuncCRRALabeled,
        policy: xr.Dataset,
        continuation: ValueFuncCRRALabeled | None,
        attrs: dict | None = None,
    ) -> None:
        # Type validation
        if not isinstance(value, ValueFuncCRRALabeled):
            raise TypeError(f"value must be ValueFuncCRRALabeled, got {type(value)}")
        if not isinstance(policy, xr.Dataset):
            raise TypeError(f"policy must be xr.Dataset, got {type(policy)}")
        if continuation is not None and not isinstance(
            continuation, ValueFuncCRRALabeled
        ):
            raise TypeError(
                f"continuation must be ValueFuncCRRALabeled or None, got {type(continuation)}"
            )

        self.value = value
        self.policy = policy
        self.continuation = continuation
        self.attrs = attrs if attrs is not None else {}

    def distance(self, other: ConsumerSolutionLabeled) -> float:
        """
        Compute the maximum absolute difference between two solutions.

        This method is used to check for convergence in infinite horizon
        problems by comparing value functions across iterations.

        Parameters
        ----------
        other : ConsumerSolutionLabeled
            Other solution to compare to.

        Returns
        -------
        float
            Maximum absolute difference between value functions.

        Raises
        ------
        TypeError
            If other is not a ConsumerSolutionLabeled.
        ValueError
            If one or both solutions have no value function.
        """
        if not isinstance(other, ConsumerSolutionLabeled):
            raise TypeError(
                f"Cannot compute distance with {type(other)}. "
                f"Expected ConsumerSolutionLabeled."
            )
        if self.value is None or other.value is None:
            raise ValueError(
                "Cannot compute distance: one or both solutions have no value function"
            )

        value = self.value.dataset
        other_value = other.value.dataset.interp_like(value)

        return float(np.max(np.abs(value - other_value).to_array()))

"""
Transition functions for labeled consumption-saving models.

This module implements the Strategy pattern for state transitions,
allowing different model types to share the same solver structure
while varying only the transition dynamics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from types import SimpleNamespace

    from HARK.rewards import UtilityFuncCRRA

    from .solution import ValueFuncCRRALabeled

__all__ = [
    "Transitions",
    "PerfectForesightTransitions",
    "IndShockTransitions",
    "RiskyAssetTransitions",
    "FixedPortfolioTransitions",
    "PortfolioTransitions",
]


def _validate_shock_keys(
    shocks: dict[str, Any], required_keys: set[str], class_name: str
) -> None:
    """
    Validate that shocks dictionary contains required keys.

    Parameters
    ----------
    shocks : dict
        Shock dictionary to validate.
    required_keys : set
        Set of required key names.
    class_name : str
        Name of the class for error messages.

    Raises
    ------
    KeyError
        If any required key is missing from shocks.
    """
    missing_keys = required_keys - set(shocks.keys())
    if missing_keys:
        raise KeyError(
            f"{class_name} requires shock keys {required_keys} but got {set(shocks.keys())}. "
            f"Missing: {missing_keys}. "
            f"Ensure the shock distribution has the correct variable names."
        )


@runtime_checkable
class Transitions(Protocol):
    """
    Protocol defining the interface for model-specific transitions.

    Each model type (PerfForesight, IndShock, RiskyAsset, etc.) implements
    this protocol with its specific transition dynamics. The transitions
    include:
    - post_state: How savings today become resources tomorrow
    - continuation: How to compute continuation value from post-state
    """

    requires_shocks: bool

    def post_state(
        self,
        post_state: dict[str, Any],
        shocks: dict[str, Any] | None,
        params: SimpleNamespace,
    ) -> dict[str, Any]:
        """Transform post-decision state to next period's state."""
        ...

    def continuation(
        self,
        post_state: dict[str, Any],
        shocks: dict[str, Any] | None,
        value_next: ValueFuncCRRALabeled,
        params: SimpleNamespace,
        utility: UtilityFuncCRRA,
    ) -> dict[str, Any]:
        """Compute continuation value from post-decision state."""
        ...


class PerfectForesightTransitions:
    """
    Transitions for perfect foresight consumption model.

    In perfect foresight, there are no shocks. Next period's market
    resources depend only on savings, risk-free return, and growth.

    State transition: mNrm_{t+1} = aNrm_t * Rfree / PermGroFac + 1
    """

    requires_shocks: bool = False

    def post_state(
        self,
        post_state: dict[str, Any],
        shocks: dict[str, Any] | None,
        params: SimpleNamespace,
    ) -> dict[str, Any]:
        """
        Transform savings to next period's market resources.

        Parameters
        ----------
        post_state : dict
            Post-decision state with 'aNrm' (normalized assets).
        shocks : dict or None
            Not used for perfect foresight.
        params : SimpleNamespace
            Parameters including Rfree and PermGroFac.

        Returns
        -------
        dict
            Next state with 'mNrm' (normalized market resources).
        """
        next_state = {}
        next_state["mNrm"] = post_state["aNrm"] * params.Rfree / params.PermGroFac + 1
        return next_state

    def continuation(
        self,
        post_state: dict[str, Any],
        shocks: dict[str, Any] | None,
        value_next: ValueFuncCRRALabeled,
        params: SimpleNamespace,
        utility: UtilityFuncCRRA,
    ) -> dict[str, Any]:
        """
        Compute continuation value for perfect foresight model.

        Parameters
        ----------
        post_state : dict
            Post-decision state with 'aNrm'.
        shocks : dict or None
            Not used for perfect foresight.
        value_next : ValueFuncCRRALabeled
            Next period's value function.
        params : SimpleNamespace
            Parameters including CRRA, Rfree, PermGroFac.
        utility : UtilityFuncCRRA
            Utility function for inverse operations.

        Returns
        -------
        dict
            Continuation value variables including v, v_der, v_inv, v_der_inv.
        """
        variables = {}
        next_state = self.post_state(post_state, shocks, params)
        variables.update(next_state)

        # Value scaled by permanent income growth
        variables["v"] = params.PermGroFac ** (1 - params.CRRA) * value_next(next_state)

        # Marginal value scaled by return and growth
        variables["v_der"] = (
            params.Rfree
            * params.PermGroFac ** (-params.CRRA)
            * value_next.derivative(next_state)
        )

        variables["v_inv"] = utility.inv(variables["v"])
        variables["v_der_inv"] = utility.derinv(variables["v_der"])

        variables["contributions"] = variables["v"]
        variables["value"] = np.sum(variables["v"])

        return variables


class IndShockTransitions:
    """
    Transitions for model with idiosyncratic income shocks.

    Adds permanent and transitory income shocks to the transition.

    State transition: mNrm_{t+1} = aNrm_t * Rfree / (PermGroFac * perm) + tran
    """

    requires_shocks: bool = True
    _required_shock_keys: set[str] = {"perm", "tran"}

    def post_state(
        self,
        post_state: dict[str, Any],
        shocks: dict[str, Any],
        params: SimpleNamespace,
    ) -> dict[str, Any]:
        """
        Transform savings to next period's market resources with income shocks.

        Parameters
        ----------
        post_state : dict
            Post-decision state with 'aNrm'.
        shocks : dict
            Income shocks with 'perm' and 'tran'.
        params : SimpleNamespace
            Parameters including Rfree and PermGroFac.

        Returns
        -------
        dict
            Next state with 'mNrm'.

        Raises
        ------
        KeyError
            If required shock keys are missing.
        """
        _validate_shock_keys(shocks, self._required_shock_keys, "IndShockTransitions")
        next_state = {}
        next_state["mNrm"] = (
            post_state["aNrm"] * params.Rfree / (params.PermGroFac * shocks["perm"])
            + shocks["tran"]
        )
        return next_state

    def continuation(
        self,
        post_state: dict[str, Any],
        shocks: dict[str, Any],
        value_next: ValueFuncCRRALabeled,
        params: SimpleNamespace,
        utility: UtilityFuncCRRA,
    ) -> dict[str, Any]:
        """
        Compute continuation value with income shocks.

        Parameters
        ----------
        post_state : dict
            Post-decision state with 'aNrm'.
        shocks : dict
            Income shocks with 'perm' and 'tran'.
        value_next : ValueFuncCRRALabeled
            Next period's value function.
        params : SimpleNamespace
            Parameters including CRRA, Rfree, PermGroFac.
        utility : UtilityFuncCRRA
            Utility function for inverse operations.

        Returns
        -------
        dict
            Continuation value variables.
        """
        variables = {}
        next_state = self.post_state(post_state, shocks, params)
        variables.update(next_state)

        # Permanent income scaling
        psi = params.PermGroFac * shocks["perm"]
        variables["psi"] = psi

        variables["v"] = psi ** (1 - params.CRRA) * value_next(next_state)
        variables["v_der"] = (
            params.Rfree * psi ** (-params.CRRA) * value_next.derivative(next_state)
        )

        variables["contributions"] = variables["v"]
        variables["value"] = np.sum(variables["v"])

        return variables


class RiskyAssetTransitions:
    """
    Transitions for model with risky asset returns.

    Savings earn a stochastic risky return instead of risk-free rate.

    State transition: mNrm_{t+1} = aNrm_t * risky / (PermGroFac * perm) + tran
    """

    requires_shocks: bool = True
    _required_shock_keys: set[str] = {"perm", "tran", "risky"}

    def post_state(
        self,
        post_state: dict[str, Any],
        shocks: dict[str, Any],
        params: SimpleNamespace,
    ) -> dict[str, Any]:
        """
        Transform savings with risky asset return.

        Parameters
        ----------
        post_state : dict
            Post-decision state with 'aNrm'.
        shocks : dict
            Shocks with 'perm', 'tran', and 'risky'.
        params : SimpleNamespace
            Parameters including PermGroFac.

        Returns
        -------
        dict
            Next state with 'mNrm'.

        Raises
        ------
        KeyError
            If required shock keys are missing.
        """
        _validate_shock_keys(shocks, self._required_shock_keys, "RiskyAssetTransitions")
        next_state = {}
        next_state["mNrm"] = (
            post_state["aNrm"] * shocks["risky"] / (params.PermGroFac * shocks["perm"])
            + shocks["tran"]
        )
        return next_state

    def continuation(
        self,
        post_state: dict[str, Any],
        shocks: dict[str, Any],
        value_next: ValueFuncCRRALabeled,
        params: SimpleNamespace,
        utility: UtilityFuncCRRA,
    ) -> dict[str, Any]:
        """
        Compute continuation value with risky asset.

        The marginal value is scaled by the risky return instead of Rfree.

        Parameters
        ----------
        post_state : dict
            Post-decision state with 'aNrm'.
        shocks : dict
            Shocks with 'perm', 'tran', and 'risky'.
        value_next : ValueFuncCRRALabeled
            Next period's value function.
        params : SimpleNamespace
            Parameters including CRRA, PermGroFac.
        utility : UtilityFuncCRRA
            Utility function for inverse operations.

        Returns
        -------
        dict
            Continuation value variables.
        """
        variables = {}
        next_state = self.post_state(post_state, shocks, params)
        variables.update(next_state)

        psi = params.PermGroFac * shocks["perm"]
        variables["psi"] = psi

        variables["v"] = psi ** (1 - params.CRRA) * value_next(next_state)
        # Risky return scales marginal value
        variables["v_der"] = (
            shocks["risky"] * psi ** (-params.CRRA) * value_next.derivative(next_state)
        )

        variables["contributions"] = variables["v"]
        variables["value"] = np.sum(variables["v"])

        return variables


class FixedPortfolioTransitions:
    """
    Transitions for model with fixed portfolio allocation.

    Agent allocates a fixed share to risky asset, earning portfolio return.

    Portfolio return: rPort = Rfree + (risky - Rfree) * RiskyShareFixed
    State transition: mNrm_{t+1} = aNrm_t * rPort / (PermGroFac * perm) + tran
    """

    requires_shocks: bool = True
    _required_shock_keys: set[str] = {"perm", "tran", "risky"}

    def post_state(
        self,
        post_state: dict[str, Any],
        shocks: dict[str, Any],
        params: SimpleNamespace,
    ) -> dict[str, Any]:
        """
        Transform savings with fixed portfolio return.

        Parameters
        ----------
        post_state : dict
            Post-decision state with 'aNrm'.
        shocks : dict
            Shocks with 'perm', 'tran', and 'risky'.
        params : SimpleNamespace
            Parameters including Rfree, PermGroFac, RiskyShareFixed.

        Returns
        -------
        dict
            Next state with 'mNrm', 'rDiff', 'rPort'.

        Raises
        ------
        KeyError
            If required shock keys are missing.
        """
        _validate_shock_keys(
            shocks, self._required_shock_keys, "FixedPortfolioTransitions"
        )
        next_state = {}
        next_state["rDiff"] = shocks["risky"] - params.Rfree
        next_state["rPort"] = (
            params.Rfree + next_state["rDiff"] * params.RiskyShareFixed
        )
        next_state["mNrm"] = (
            post_state["aNrm"]
            * next_state["rPort"]
            / (params.PermGroFac * shocks["perm"])
            + shocks["tran"]
        )
        return next_state

    def continuation(
        self,
        post_state: dict[str, Any],
        shocks: dict[str, Any],
        value_next: ValueFuncCRRALabeled,
        params: SimpleNamespace,
        utility: UtilityFuncCRRA,
    ) -> dict[str, Any]:
        """
        Compute continuation value with fixed portfolio.

        The marginal value is scaled by the portfolio return.

        Parameters
        ----------
        post_state : dict
            Post-decision state with 'aNrm'.
        shocks : dict
            Shocks with 'perm', 'tran', and 'risky'.
        value_next : ValueFuncCRRALabeled
            Next period's value function.
        params : SimpleNamespace
            Parameters including CRRA, PermGroFac.
        utility : UtilityFuncCRRA
            Utility function for inverse operations.

        Returns
        -------
        dict
            Continuation value variables.
        """
        variables = {}
        next_state = self.post_state(post_state, shocks, params)
        variables.update(next_state)

        psi = params.PermGroFac * shocks["perm"]
        variables["psi"] = psi

        variables["v"] = psi ** (1 - params.CRRA) * value_next(next_state)
        # Portfolio return scales marginal value
        variables["v_der"] = (
            next_state["rPort"]
            * psi ** (-params.CRRA)
            * value_next.derivative(next_state)
        )

        variables["contributions"] = variables["v"]
        variables["value"] = np.sum(variables["v"])

        return variables


class PortfolioTransitions:
    """
    Transitions for model with optimal portfolio choice.

    Agent optimally chooses risky share (stigma) each period.

    Portfolio return: rPort = Rfree + (risky - Rfree) * stigma
    State transition: mNrm_{t+1} = aNrm_t * rPort / (PermGroFac * perm) + tran

    Also computes derivatives for portfolio optimization:
    - dvda: derivative of value wrt assets
    - dvds: derivative of value wrt risky share
    """

    requires_shocks: bool = True
    _required_shock_keys: set[str] = {"perm", "tran", "risky"}

    def post_state(
        self,
        post_state: dict[str, Any],
        shocks: dict[str, Any],
        params: SimpleNamespace,
    ) -> dict[str, Any]:
        """
        Transform savings with optimal portfolio return.

        Parameters
        ----------
        post_state : dict
            Post-decision state with 'aNrm' and 'stigma' (risky share).
        shocks : dict
            Shocks with 'perm', 'tran', and 'risky'.
        params : SimpleNamespace
            Parameters including Rfree, PermGroFac.

        Returns
        -------
        dict
            Next state with 'mNrm', 'rDiff', 'rPort'.

        Raises
        ------
        KeyError
            If required shock keys are missing.
        """
        _validate_shock_keys(shocks, self._required_shock_keys, "PortfolioTransitions")
        next_state = {}
        next_state["rDiff"] = shocks["risky"] - params.Rfree
        next_state["rPort"] = params.Rfree + next_state["rDiff"] * post_state["stigma"]
        next_state["mNrm"] = (
            post_state["aNrm"]
            * next_state["rPort"]
            / (params.PermGroFac * shocks["perm"])
            + shocks["tran"]
        )
        return next_state

    def continuation(
        self,
        post_state: dict[str, Any],
        shocks: dict[str, Any],
        value_next: ValueFuncCRRALabeled,
        params: SimpleNamespace,
        utility: UtilityFuncCRRA,
    ) -> dict[str, Any]:
        """
        Compute continuation value with optimal portfolio.

        Also computes derivatives needed for portfolio optimization:
        - dvda: used for consumption FOC
        - dvds: used for portfolio FOC (should equal 0 at optimum)

        Parameters
        ----------
        post_state : dict
            Post-decision state with 'aNrm' and 'stigma'.
        shocks : dict
            Shocks with 'perm', 'tran', and 'risky'.
        value_next : ValueFuncCRRALabeled
            Next period's value function.
        params : SimpleNamespace
            Parameters including CRRA, PermGroFac.
        utility : UtilityFuncCRRA
            Utility function for inverse operations.

        Returns
        -------
        dict
            Continuation value variables including dvda and dvds.
        """
        variables = {}
        next_state = self.post_state(post_state, shocks, params)
        variables.update(next_state)

        psi = params.PermGroFac * shocks["perm"]
        variables["psi"] = psi

        variables["v"] = psi ** (1 - params.CRRA) * value_next(next_state)
        variables["v_der"] = psi ** (-params.CRRA) * value_next.derivative(next_state)

        # Derivatives for portfolio optimization
        variables["dvda"] = next_state["rPort"] * variables["v_der"]
        variables["dvds"] = (
            next_state["rDiff"] * post_state["aNrm"] * variables["v_der"]
        )

        variables["contributions"] = variables["v"]
        variables["value"] = np.sum(variables["v"])

        return variables

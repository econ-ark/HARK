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


def _base_continuation(
    transitions,
    post_state: dict[str, Any],
    shocks: dict[str, Any],
    value_next: "ValueFuncCRRALabeled",
    params: "SimpleNamespace",
    return_factor: Any,
) -> dict[str, Any]:
    """
    Shared computation kernel for stochastic continuation methods.

    Computes the next state, permanent income scaling (psi), value (v),
    marginal value (v_der), contributions, and aggregate value that are
    common to all stochastic transition classes.

    Parameters
    ----------
    transitions : Transitions instance
        The calling transitions object, whose ``post_state`` method is
        used to map the post-decision state forward through shocks.
    post_state : dict
        Post-decision state (e.g. containing 'aNrm').
    shocks : dict
        Realized shocks for this quadrature node (must include 'perm').
    value_next : ValueFuncCRRALabeled
        Next period's value function, callable and with a ``derivative``
        method.
    params : SimpleNamespace
        Model parameters; must expose ``PermGroFac`` and ``CRRA``.
    return_factor : scalar, array, or callable
        Factor that scales the marginal value ``v_der``.  Pass
        ``params.Rfree`` for IndShock, ``shocks["risky"]`` for
        RiskyAsset, ``1.0`` for Portfolio (which applies its own scaling
        afterward), or a callable ``(next_state) -> factor`` when the
        factor depends on ``next_state`` (e.g. ``lambda ns: ns["rPort"]``
        for FixedPortfolio).

    Returns
    -------
    dict
        Variables dict containing: all entries from ``next_state``,
        ``psi``, ``v``, ``v_der``, ``contributions``, and ``value``.
    """
    variables = {}
    next_state = transitions.post_state(post_state, shocks, params)
    variables.update(next_state)

    psi = params.PermGroFac * shocks["perm"]
    variables["psi"] = psi

    # Allow return_factor to depend on next_state without a second post_state call.
    if callable(return_factor):
        factor = return_factor(next_state)
    else:
        factor = return_factor

    variables["v"] = psi ** (1 - params.CRRA) * value_next(next_state)
    variables["v_der"] = (
        factor * psi ** (-params.CRRA) * value_next.derivative(next_state)
    )

    variables["contributions"] = variables["v"]
    variables["value"] = np.sum(variables["v"])

    return variables


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
        return _base_continuation(
            self, post_state, shocks, value_next, params, params.Rfree
        )


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
        return _base_continuation(
            self, post_state, shocks, value_next, params, shocks["risky"]
        )


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
        return _base_continuation(
            self, post_state, shocks, value_next, params, lambda ns: ns["rPort"]
        )


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

        Uses ``return_factor=1.0`` so that ``v_der`` is unscaled.
        Then adds ``dvda`` (portfolio return times ``v_der``) for the
        consumption FOC and ``dvds`` (excess return times assets times
        ``v_der``) for the portfolio FOC (should equal 0 at optimum).

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
        variables = _base_continuation(
            self, post_state, shocks, value_next, params, 1.0
        )

        # Derivatives for portfolio optimization
        variables["dvda"] = variables["rPort"] * variables["v_der"]
        variables["dvds"] = variables["rDiff"] * post_state["aNrm"] * variables["v_der"]

        return variables

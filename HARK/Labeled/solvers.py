"""
Solvers for labeled consumption-saving models.

This module implements the Template Method pattern for the Endogenous Grid
Method (EGM) algorithm. The base solver defines the algorithm skeleton,
while concrete solvers override specific hook methods for their model type.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from HARK.metric import MetricObject
from HARK.rewards import UtilityFuncCRRA

from .solution import ConsumerSolutionLabeled, ValueFuncCRRALabeled
from .transitions import (
    FixedPortfolioTransitions,
    IndShockTransitions,
    PerfectForesightTransitions,
    PortfolioTransitions,
    RiskyAssetTransitions,
)

if TYPE_CHECKING:
    from HARK.distributions import DiscreteDistributionLabeled

__all__ = [
    "BaseLabeledSolver",
    "ConsPerfForesightLabeledSolver",
    "ConsIndShockLabeledSolver",
    "ConsRiskyAssetLabeledSolver",
    "ConsFixedPortfolioLabeledSolver",
    "ConsPortfolioLabeledSolver",
]


class BaseLabeledSolver(MetricObject):
    """
    Base solver implementing Template Method pattern for EGM algorithm.

    This class provides the algorithm skeleton for solving consumption-saving
    problems using the Endogenous Grid Method. Subclasses customize behavior
    by:
    1. Setting TransitionsClass to specify model-specific transitions
    2. Overriding hook methods for model-specific logic

    Template Method: solve()
    Hook Methods:
        - create_params_namespace(): Add model-specific parameters
        - calculate_borrowing_constraint(): Model-specific constraint logic
        - create_post_state(): Add extra state dimensions (e.g., stigma)
        - create_continuation_function(): Handle shock integration

    Parameters
    ----------
    solution_next : ConsumerSolutionLabeled
        Solution to next period's problem.
    LivPrb : float
        Survival probability.
    DiscFac : float
        Intertemporal discount factor.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree : float
        Risk-free interest factor.
    PermGroFac : float
        Permanent income growth factor.
    BoroCnstArt : float or None
        Artificial borrowing constraint.
    aXtraGrid : np.ndarray
        Grid of end-of-period asset values above minimum.
    **kwargs
        Additional model-specific parameters.

    Raises
    ------
    ValueError
        If CRRA is invalid or aXtraGrid is malformed.
    """

    # Class-level strategy specification - override in subclasses
    TransitionsClass: type = PerfectForesightTransitions

    def __init__(
        self,
        solution_next: ConsumerSolutionLabeled,
        LivPrb: float,
        DiscFac: float,
        CRRA: float,
        Rfree: float,
        PermGroFac: float,
        BoroCnstArt: float | None,
        aXtraGrid: np.ndarray,
        **kwargs,
    ) -> None:
        # Input validation - solution_next
        if solution_next is None:
            raise ValueError("solution_next cannot be None")
        if not isinstance(solution_next, ConsumerSolutionLabeled):
            raise TypeError(
                f"solution_next must be ConsumerSolutionLabeled, got {type(solution_next)}"
            )
        if "m_nrm_min" not in solution_next.attrs:
            raise ValueError(
                "solution_next.attrs must contain 'm_nrm_min'. "
                "Use make_solution_terminal_labeled() to create valid terminal solutions."
            )

        # Input validation - CRRA
        if not np.isfinite(CRRA):
            raise ValueError(f"CRRA must be finite, got {CRRA}")
        if CRRA < 0:
            raise ValueError(f"CRRA must be non-negative, got {CRRA}")

        # Input validation - economic parameters
        if LivPrb <= 0 or LivPrb > 1:
            raise ValueError(f"LivPrb must be in (0, 1], got {LivPrb}")
        if DiscFac <= 0:
            raise ValueError(f"DiscFac must be positive, got {DiscFac}")
        if Rfree <= 0:
            raise ValueError(f"Rfree must be positive, got {Rfree}")
        if PermGroFac <= 0:
            raise ValueError(f"PermGroFac must be positive, got {PermGroFac}")

        # Input validation - asset grid
        aXtraGrid = np.asarray(aXtraGrid)
        if len(aXtraGrid) == 0:
            raise ValueError("aXtraGrid cannot be empty")
        if np.any(aXtraGrid < 0):
            raise ValueError("aXtraGrid values must be non-negative")
        if not np.all(np.diff(aXtraGrid) > 0):
            raise ValueError("aXtraGrid must be strictly increasing")

        # Store parameters
        self.solution_next = solution_next
        self.LivPrb = LivPrb
        self.DiscFac = DiscFac
        self.CRRA = CRRA
        self.Rfree = Rfree
        self.PermGroFac = PermGroFac
        self.BoroCnstArt = BoroCnstArt
        self.aXtraGrid = aXtraGrid

        # Initialize utility function
        self.u = UtilityFuncCRRA(CRRA)

        # Initialize transitions strategy
        self.transitions = self.TransitionsClass()

        # Store additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    # =========================================================================
    # TEMPLATE METHOD - The algorithm skeleton
    # =========================================================================

    def solve(self) -> ConsumerSolutionLabeled:
        """
        Solve the consumption-saving problem using EGM.

        This is the template method that defines the algorithm skeleton.
        It calls hook methods that subclasses can override.

        Returns
        -------
        ConsumerSolutionLabeled
            Solution containing value function, policy, and continuation.
        """
        self.prepare_to_solve()
        self.endogenous_grid_method()
        return self.solution

    # =========================================================================
    # HOOK METHODS - Override in subclasses for customization
    # =========================================================================

    def create_params_namespace(self) -> SimpleNamespace:
        """
        Create parameters namespace.

        Override in subclasses to add model-specific parameters.

        Returns
        -------
        SimpleNamespace
            Parameters for this period's problem.
        """
        return SimpleNamespace(
            Discount=self.DiscFac * self.LivPrb,
            CRRA=self.CRRA,
            Rfree=self.Rfree,
            PermGroFac=self.PermGroFac,
        )

    def calculate_borrowing_constraint(self) -> None:
        """
        Calculate the natural borrowing constraint.

        Override in shock models to account for minimum shock realizations.
        Sets self.BoroCnstNat.
        """
        self.BoroCnstNat = (
            (self.solution_next.attrs["m_nrm_min"] - 1)
            * self.params.PermGroFac
            / self.params.Rfree
        )

    def create_post_state(self) -> xr.Dataset:
        """
        Create the post-decision state grid.

        Override in portfolio models to add the risky share dimension.

        Returns
        -------
        xr.Dataset
            Post-decision state grid.
        """
        if self.nat_boro_cnst:
            a_grid = self.aXtraGrid + self.m_nrm_min
        else:
            a_grid = np.append(0.0, self.aXtraGrid) + self.m_nrm_min

        aVec = xr.DataArray(
            a_grid,
            name="aNrm",
            dims=("aNrm"),
            attrs={"long_name": "savings", "state": True},
        )
        return xr.Dataset({"aNrm": aVec})

    def create_continuation_function(self) -> ValueFuncCRRALabeled:
        """
        Create the continuation value function.

        Override in shock models to integrate over shock distributions.

        Returns
        -------
        ValueFuncCRRALabeled
            Continuation value function.
        """
        value_next = self.solution_next.value

        v_end = self.transitions.continuation(
            self.post_state, None, value_next, self.params, self.u
        )
        v_end = xr.Dataset(v_end).drop_vars(["mNrm"])

        v_end["v_inv"] = self.u.inv(v_end["v"])
        v_end["v_der_inv"] = self.u.derinv(v_end["v_der"])

        borocnst = self.borocnst.drop_vars(["mNrm"]).expand_dims("aNrm")
        if self.nat_boro_cnst:
            v_end = xr.merge([borocnst, v_end], join="outer", compat="no_conflicts")

        return ValueFuncCRRALabeled(v_end, self.params.CRRA)

    # =========================================================================
    # CORE METHODS - Shared implementation, rarely overridden
    # =========================================================================

    def prepare_to_solve(self) -> None:
        """Prepare solver state before running EGM."""
        self.params = self.create_params_namespace()
        self.calculate_borrowing_constraint()
        self.define_boundary_constraint()
        self.post_state = self.create_post_state()

    def define_boundary_constraint(self) -> None:
        """Define borrowing constraint boundary conditions."""
        if self.BoroCnstArt is None or self.BoroCnstArt <= self.BoroCnstNat:
            self.m_nrm_min = self.BoroCnstNat
            self.nat_boro_cnst = True
            self.borocnst = xr.Dataset(
                coords={"mNrm": self.m_nrm_min, "aNrm": self.m_nrm_min},
                data_vars={
                    "cNrm": 0.0,
                    "v": -np.inf,
                    "v_inv": 0.0,
                    "reward": -np.inf,
                    "marginal_reward": np.inf,
                    "v_der": np.inf,
                    "v_der_inv": 0.0,
                },
            )
        else:
            self.m_nrm_min = self.BoroCnstArt
            self.nat_boro_cnst = False
            self.borocnst = xr.Dataset(
                coords={"mNrm": self.m_nrm_min, "aNrm": self.m_nrm_min},
                data_vars={"cNrm": 0.0},
            )

    def state_transition(
        self, state: dict[str, Any], action: dict[str, Any], params: SimpleNamespace
    ) -> dict[str, Any]:
        """Compute post-decision state from state and action."""
        return {"aNrm": state["mNrm"] - action["cNrm"]}

    def reverse_transition(
        self,
        post_state: dict[str, Any],
        action: dict[str, Any],
        params: SimpleNamespace,
    ) -> dict[str, Any]:
        """Recover state from post-decision state and action (for EGM)."""
        return {"mNrm": post_state["aNrm"] + action["cNrm"]}

    def egm_transition(
        self,
        post_state: dict[str, Any],
        continuation: ValueFuncCRRALabeled,
        params: SimpleNamespace,
    ) -> dict[str, Any]:
        """Compute optimal action using first-order condition (EGM)."""
        return {
            "cNrm": self.u.derinv(params.Discount * continuation.derivative(post_state))
        }

    def value_transition(
        self,
        action: dict[str, Any],
        state: dict[str, Any],
        continuation: ValueFuncCRRALabeled,
        params: SimpleNamespace,
    ) -> dict[str, Any]:
        """Compute value function variables from action, state, and continuation."""
        variables = {}
        post_state = self.state_transition(state, action, params)
        variables.update(post_state)

        variables["reward"] = self.u(action["cNrm"])
        variables["v"] = variables["reward"] + params.Discount * continuation(
            post_state
        )
        variables["v_inv"] = self.u.inv(variables["v"])

        variables["marginal_reward"] = self.u.der(action["cNrm"])
        variables["v_der"] = variables["marginal_reward"]
        variables["v_der_inv"] = action["cNrm"]

        variables["contributions"] = variables["v"]
        variables["value"] = np.sum(variables["v"])

        return variables

    def _continuation_for_expectation(
        self,
        shocks: dict[str, Any],
        post_state: dict[str, Any],
        value_next: ValueFuncCRRALabeled,
        params: SimpleNamespace,
    ) -> dict[str, Any]:
        """
        Wrapper for continuation transition compatible with expected().

        This method adapts the transitions.continuation() interface to work
        with the expected() function from DiscreteDistributionLabeled.

        Parameters
        ----------
        shocks : dict[str, Any]
            Shock realizations (e.g., perm, tran, risky).
        post_state : dict[str, Any]
            Post-decision state (e.g., aNrm).
        value_next : ValueFuncCRRALabeled
            Next period's value function.
        params : SimpleNamespace
            Model parameters.

        Returns
        -------
        dict[str, Any]
            Continuation value variables.
        """
        return self.transitions.continuation(
            post_state, shocks, value_next, params, self.u
        )

    def endogenous_grid_method(self) -> None:
        """Execute the Endogenous Grid Method algorithm."""
        wfunc = self.create_continuation_function()

        # Check for numerical issues in continuation function
        if np.any(~np.isfinite(wfunc.dataset["v_der_inv"].values)):
            warnings.warn(
                "Continuation value function contains NaN or Inf values. "
                "This may indicate invalid parameters (CRRA too high, "
                "PermGroFac issues, or extreme shock realizations).",
                RuntimeWarning,
                stacklevel=2,
            )

        # EGM: Get optimal actions from first-order condition
        acted = self.egm_transition(self.post_state, wfunc, self.params)
        state = self.reverse_transition(self.post_state, acted, self.params)

        # Check for numerical issues in EGM results
        if np.any(acted["cNrm"] < 0):
            warnings.warn(
                "EGM produced negative consumption values. "
                "Check discount factor and interest rate parameters.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Swap dimensions for state-based indexing
        action = xr.Dataset(acted).swap_dims({"aNrm": "mNrm"})
        state = xr.Dataset(state).swap_dims({"aNrm": "mNrm"})

        egm_dataset = xr.merge([action, state])

        if not self.nat_boro_cnst:
            egm_dataset = xr.concat(
                [self.borocnst, egm_dataset], dim="mNrm", data_vars="all"
            )

        # Compute values
        values = self.value_transition(egm_dataset, egm_dataset, wfunc, self.params)
        egm_dataset.update(values)

        if self.nat_boro_cnst:
            egm_dataset = xr.concat(
                [self.borocnst, egm_dataset],
                dim="mNrm",
                data_vars="all",
                combine_attrs="no_conflicts",
            )

        egm_dataset = egm_dataset.drop_vars("aNrm")

        # Build solution
        vfunc = ValueFuncCRRALabeled(
            egm_dataset[["v", "v_der", "v_inv", "v_der_inv"]], self.params.CRRA
        )
        pfunc = egm_dataset[["cNrm"]]

        self.solution = ConsumerSolutionLabeled(
            value=vfunc,
            policy=pfunc,
            continuation=wfunc,
            attrs={"m_nrm_min": self.m_nrm_min, "dataset": egm_dataset},
        )


class ConsPerfForesightLabeledSolver(BaseLabeledSolver):
    """
    Solver for perfect foresight consumption model.

    Uses PerfectForesightTransitions - no shocks, risk-free return only.
    """

    TransitionsClass = PerfectForesightTransitions


class ConsIndShockLabeledSolver(BaseLabeledSolver):
    """
    Solver for consumption model with idiosyncratic income shocks.

    Uses IndShockTransitions and integrates continuation value over
    the income shock distribution.

    Additional Parameters
    ---------------------
    IncShkDstn : DiscreteDistributionLabeled
        Distribution of income shocks with 'perm' and 'tran' variables.
    """

    TransitionsClass = IndShockTransitions

    def __init__(
        self,
        solution_next: ConsumerSolutionLabeled,
        IncShkDstn: DiscreteDistributionLabeled,
        LivPrb: float,
        DiscFac: float,
        CRRA: float,
        Rfree: float,
        PermGroFac: float,
        BoroCnstArt: float | None,
        aXtraGrid: np.ndarray,
        **kwargs,
    ) -> None:
        self.IncShkDstn = IncShkDstn
        super().__init__(
            solution_next=solution_next,
            LivPrb=LivPrb,
            DiscFac=DiscFac,
            CRRA=CRRA,
            Rfree=Rfree,
            PermGroFac=PermGroFac,
            BoroCnstArt=BoroCnstArt,
            aXtraGrid=aXtraGrid,
            **kwargs,
        )

    def calculate_borrowing_constraint(self) -> None:
        """Calculate constraint accounting for minimum shock realizations."""
        PermShkMinNext = np.min(self.IncShkDstn.atoms[0])
        TranShkMinNext = np.min(self.IncShkDstn.atoms[1])

        self.BoroCnstNat = (
            (self.solution_next.attrs["m_nrm_min"] - TranShkMinNext)
            * (self.params.PermGroFac * PermShkMinNext)
            / self.params.Rfree
        )

    def create_continuation_function(self) -> ValueFuncCRRALabeled:
        """Create continuation function by integrating over income shocks."""
        value_next = self.solution_next.value

        v_end = self.IncShkDstn.expected(
            func=self._continuation_for_expectation,
            post_state=self.post_state,
            value_next=value_next,
            params=self.params,
        )

        v_end["v_inv"] = self.u.inv(v_end["v"])
        v_end["v_der_inv"] = self.u.derinv(v_end["v_der"])

        borocnst = self.borocnst.drop_vars(["mNrm"]).expand_dims("aNrm")
        if self.nat_boro_cnst:
            v_end = xr.merge([borocnst, v_end], join="outer", compat="no_conflicts")

        return ValueFuncCRRALabeled(v_end, self.params.CRRA)


class ConsRiskyAssetLabeledSolver(BaseLabeledSolver):
    """
    Solver for consumption model with risky asset.

    Uses RiskyAssetTransitions - all savings earn stochastic risky return.

    Additional Parameters
    ---------------------
    ShockDstn : DiscreteDistributionLabeled
        Joint distribution of income and risky return shocks.
    """

    TransitionsClass = RiskyAssetTransitions

    def __init__(
        self,
        solution_next: ConsumerSolutionLabeled,
        ShockDstn: DiscreteDistributionLabeled,
        LivPrb: float,
        DiscFac: float,
        CRRA: float,
        Rfree: float,
        PermGroFac: float,
        BoroCnstArt: float | None,
        aXtraGrid: np.ndarray,
        **kwargs,
    ) -> None:
        self.ShockDstn = ShockDstn
        super().__init__(
            solution_next=solution_next,
            LivPrb=LivPrb,
            DiscFac=DiscFac,
            CRRA=CRRA,
            Rfree=Rfree,
            PermGroFac=PermGroFac,
            BoroCnstArt=BoroCnstArt,
            aXtraGrid=aXtraGrid,
            **kwargs,
        )

    def calculate_borrowing_constraint(self) -> None:
        """Calculate constraint with artificial borrowing constraint."""
        self.BoroCnstArt = 0.0
        self.IncShkDstn = self.ShockDstn

        PermShkMinNext = np.min(self.ShockDstn.atoms[0])
        TranShkMinNext = np.min(self.ShockDstn.atoms[1])

        self.BoroCnstNat = (
            (self.solution_next.attrs["m_nrm_min"] - TranShkMinNext)
            * (self.params.PermGroFac * PermShkMinNext)
            / self.params.Rfree
        )

    def create_continuation_function(self) -> ValueFuncCRRALabeled:
        """Create continuation function integrating over shock distribution."""
        value_next = self.solution_next.value

        v_end = self.ShockDstn.expected(
            func=self._continuation_for_expectation,
            post_state=self.post_state,
            value_next=value_next,
            params=self.params,
        )

        v_end["v_inv"] = self.u.inv(v_end["v"])
        v_end["v_der_inv"] = self.u.derinv(v_end["v_der"])

        borocnst = self.borocnst.drop_vars(["mNrm"]).expand_dims("aNrm")
        if self.nat_boro_cnst:
            v_end = xr.merge([borocnst, v_end], join="outer", compat="no_conflicts")

        v_end = v_end.transpose("aNrm", ...)

        return ValueFuncCRRALabeled(v_end, self.params.CRRA)


class ConsFixedPortfolioLabeledSolver(ConsRiskyAssetLabeledSolver):
    """
    Solver for consumption model with fixed portfolio allocation.

    Uses FixedPortfolioTransitions - agent allocates fixed share to risky asset.

    Additional Parameters
    ---------------------
    RiskyShareFixed : float
        Fixed share of savings allocated to risky asset.
    """

    TransitionsClass = FixedPortfolioTransitions

    def __init__(
        self,
        solution_next: ConsumerSolutionLabeled,
        ShockDstn: DiscreteDistributionLabeled,
        LivPrb: float,
        DiscFac: float,
        CRRA: float,
        Rfree: float,
        PermGroFac: float,
        BoroCnstArt: float | None,
        aXtraGrid: np.ndarray,
        RiskyShareFixed: float,
        **kwargs,
    ) -> None:
        # Validate RiskyShareFixed
        if RiskyShareFixed < 0 or RiskyShareFixed > 1:
            raise ValueError(
                f"RiskyShareFixed must be in [0, 1], got {RiskyShareFixed}"
            )

        self.RiskyShareFixed = RiskyShareFixed
        super().__init__(
            solution_next=solution_next,
            ShockDstn=ShockDstn,
            LivPrb=LivPrb,
            DiscFac=DiscFac,
            CRRA=CRRA,
            Rfree=Rfree,
            PermGroFac=PermGroFac,
            BoroCnstArt=BoroCnstArt,
            aXtraGrid=aXtraGrid,
            **kwargs,
        )

    def create_params_namespace(self) -> SimpleNamespace:
        """Add RiskyShareFixed to parameters."""
        params = super().create_params_namespace()
        params.RiskyShareFixed = self.RiskyShareFixed
        return params


class ConsPortfolioLabeledSolver(ConsRiskyAssetLabeledSolver):
    """
    Solver for consumption model with optimal portfolio choice.

    Uses PortfolioTransitions - agent optimally chooses risky share each period.
    The optimal share is found by solving the portfolio first-order condition.

    Additional Parameters
    ---------------------
    ShareGrid : np.ndarray
        Grid of risky share values to search over.
    """

    TransitionsClass = PortfolioTransitions

    def __init__(
        self,
        solution_next: ConsumerSolutionLabeled,
        ShockDstn: DiscreteDistributionLabeled,
        LivPrb: float,
        DiscFac: float,
        CRRA: float,
        Rfree: float,
        PermGroFac: float,
        BoroCnstArt: float | None,
        aXtraGrid: np.ndarray,
        ShareGrid: np.ndarray,
        **kwargs,
    ) -> None:
        # Validate ShareGrid
        ShareGrid = np.asarray(ShareGrid)
        if len(ShareGrid) == 0:
            raise ValueError("ShareGrid cannot be empty")
        if np.any(ShareGrid < 0) or np.any(ShareGrid > 1):
            raise ValueError("ShareGrid values must be in [0, 1]")
        if not np.all(np.diff(ShareGrid) > 0):
            raise ValueError("ShareGrid must be strictly increasing")

        self.ShareGrid = ShareGrid
        super().__init__(
            solution_next=solution_next,
            ShockDstn=ShockDstn,
            LivPrb=LivPrb,
            DiscFac=DiscFac,
            CRRA=CRRA,
            Rfree=Rfree,
            PermGroFac=PermGroFac,
            BoroCnstArt=BoroCnstArt,
            aXtraGrid=aXtraGrid,
            **kwargs,
        )

    def create_post_state(self) -> xr.Dataset:
        """Add risky share dimension to post-decision state."""
        post_state = super().create_post_state()
        post_state["stigma"] = xr.DataArray(
            self.ShareGrid, dims=["stigma"], attrs={"long_name": "risky share"}
        )
        return post_state

    def create_continuation_function(self) -> ValueFuncCRRALabeled:
        """
        Create continuation function with optimal portfolio choice.

        First computes continuation value over the (aNrm, stigma) grid,
        then finds the optimal stigma for each aNrm level.
        """
        # Get continuation value over full (aNrm, stigma) grid
        wfunc = super().create_continuation_function()

        dvds = wfunc.dataset["dvds"].values

        # Find optimal share using linear interpolation on FOC
        crossing = np.logical_and(dvds[:, 1:] <= 0.0, dvds[:, :-1] >= 0.0)
        share_idx = np.argmax(crossing, axis=1)
        a_idx = np.arange(self.post_state["aNrm"].size)

        bottom_share = self.ShareGrid[share_idx]
        top_share = self.ShareGrid[share_idx + 1]
        bottom_foc = dvds[a_idx, share_idx]
        top_foc = dvds[a_idx, share_idx + 1]

        # Linear interpolation with division-by-zero protection
        denominator = top_foc - bottom_foc
        fallback_mask = np.abs(denominator) <= 1e-12
        if np.any(fallback_mask):
            n_fallbacks = np.sum(fallback_mask)
            warnings.warn(
                f"Portfolio optimization used fallback interpolation for {n_fallbacks} "
                f"grid points due to near-zero FOC difference. "
                f"Consider refining ShareGrid for more accurate results.",
                RuntimeWarning,
                stacklevel=2,
            )
        alpha = np.where(
            ~fallback_mask,
            1.0 - top_foc / denominator,
            0.5,
        )
        opt_share = (1.0 - alpha) * bottom_share + alpha * top_share

        # Handle corner solutions
        opt_share[dvds[:, -1] > 0.0] = 1.0  # Want more than 100% risky
        opt_share[dvds[:, 0] < 0.0] = 0.0  # Want less than 0% risky

        if not self.nat_boro_cnst:
            # At aNrm = 0 the portfolio share is irrelevant; 1.0 is limit as a --> 0
            opt_share[0] = 1.0

        opt_share = xr.DataArray(
            opt_share,
            coords={"aNrm": self.post_state["aNrm"].values},
            dims=["aNrm"],
            attrs={"long_name": "optimal risky share"},
        )

        # Evaluate continuation at optimal share
        v_end = wfunc.evaluate({"aNrm": self.post_state["aNrm"], "stigma": opt_share})
        v_end = v_end.reset_coords(names="stigma")

        wfunc = ValueFuncCRRALabeled(v_end, self.params.CRRA)

        # Remove stigma from post_state for EGM
        self.post_state = self.post_state.drop_vars("stigma")

        return wfunc

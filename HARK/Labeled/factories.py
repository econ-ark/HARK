"""
Factory functions for creating labeled solutions and distributions.

These functions create terminal period solutions and convert standard
HARK distributions to labeled versions for use with labeled solvers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from HARK.Calibration.Assets.AssetProcesses import (
    combine_IncShkDstn_and_RiskyDstn,
    make_lognormal_RiskyDstn,
)
from HARK.Calibration.Income.IncomeProcesses import (
    construct_lognormal_income_process_unemployment,
)
from HARK.distributions import DiscreteDistributionLabeled
from HARK.rewards import UtilityFuncCRRA

from .solution import ConsumerSolutionLabeled, ValueFuncCRRALabeled

if TYPE_CHECKING:
    from numpy.random import Generator

__all__ = [
    "make_solution_terminal_labeled",
    "make_labeled_inc_shk_dstn",
    "make_labeled_risky_dstn",
    "make_labeled_shock_dstn",
]


def make_solution_terminal_labeled(
    CRRA: float,
    aXtraGrid: np.ndarray,
) -> ConsumerSolutionLabeled:
    """
    Construct the terminal period solution for a labeled consumption model.

    In the terminal period, the optimal policy is to consume all resources.
    This function creates the value function and policy function for this
    terminal period.

    Parameters
    ----------
    CRRA : float
        Coefficient of relative risk aversion.
    aXtraGrid : np.ndarray
        Grid of assets above minimum. Used to construct the state grid.

    Returns
    -------
    ConsumerSolutionLabeled
        Terminal period solution with value and policy functions.

    Raises
    ------
    ValueError
        If CRRA is invalid or aXtraGrid is malformed.
    """
    # Input validation
    if not np.isfinite(CRRA):
        raise ValueError(f"CRRA must be finite, got {CRRA}")
    if CRRA <= 0:
        raise ValueError(f"CRRA must be positive, got {CRRA}")

    aXtraGrid = np.asarray(aXtraGrid)
    if len(aXtraGrid) == 0:
        raise ValueError("aXtraGrid cannot be empty")
    if np.any(aXtraGrid < 0):
        raise ValueError("aXtraGrid values must be non-negative")
    if not np.all(np.diff(aXtraGrid) > 0):
        raise ValueError("aXtraGrid must be strictly increasing")

    u = UtilityFuncCRRA(CRRA)

    # Create state grid
    mNrm = xr.DataArray(
        np.append(0.0, aXtraGrid),
        name="mNrm",
        dims=("mNrm"),
        attrs={"long_name": "cash_on_hand"},
    )
    state = xr.Dataset({"mNrm": mNrm})

    # Optimal decision: consume everything in terminal period
    cNrm = xr.DataArray(
        mNrm,
        name="cNrm",
        dims=state.dims,
        coords=state.coords,
        attrs={"long_name": "consumption"},
    )

    # Compute value function variables
    v = u(cNrm)
    v.name = "v"
    v.attrs = {"long_name": "value function"}

    v_der = u.der(cNrm)
    v_der.name = "v_der"
    v_der.attrs = {"long_name": "marginal value function"}

    v_inv = cNrm.copy()
    v_inv.name = "v_inv"
    v_inv.attrs = {"long_name": "inverse value function"}

    v_der_inv = cNrm.copy()
    v_der_inv.name = "v_der_inv"
    v_der_inv.attrs = {"long_name": "inverse marginal value function"}

    dataset = xr.Dataset(
        {
            "cNrm": cNrm,
            "v": v,
            "v_der": v_der,
            "v_inv": v_inv,
            "v_der_inv": v_der_inv,
        }
    )

    vfunc = ValueFuncCRRALabeled(dataset[["v", "v_der", "v_inv", "v_der_inv"]], CRRA)

    solution_terminal = ConsumerSolutionLabeled(
        value=vfunc,
        policy=dataset[["cNrm"]],
        continuation=None,
        attrs={"m_nrm_min": 0.0},
    )
    return solution_terminal


def make_labeled_inc_shk_dstn(
    T_cycle: int,
    PermShkStd: list[float],
    PermShkCount: int,
    TranShkStd: list[float],
    TranShkCount: int,
    T_retire: int,
    UnempPrb: float,
    IncUnemp: float,
    UnempPrbRet: float,
    IncUnempRet: float,
    RNG: Generator,
    neutral_measure: bool = False,
) -> list[DiscreteDistributionLabeled]:
    """
    Create labeled income shock distributions.

    Wrapper around construct_lognormal_income_process_unemployment that
    converts the resulting distributions to labeled versions.

    Parameters
    ----------
    T_cycle : int
        Number of periods in the cycle.
    PermShkStd : list[float]
        Standard deviation of permanent shocks by period.
    PermShkCount : int
        Number of permanent shock points.
    TranShkStd : list[float]
        Standard deviation of transitory shocks by period.
    TranShkCount : int
        Number of transitory shock points.
    T_retire : int
        Period of retirement (0 means never retire).
    UnempPrb : float
        Probability of unemployment during working life.
    IncUnemp : float
        Income during unemployment.
    UnempPrbRet : float
        Probability of "unemployment" in retirement.
    IncUnempRet : float
        Income during retirement "unemployment".
    RNG : Generator
        Random number generator.
    neutral_measure : bool, optional
        Whether to use risk-neutral measure. Default False.

    Returns
    -------
    list[DiscreteDistributionLabeled]
        List of labeled income shock distributions, one per period.

    Raises
    ------
    ValueError
        If input parameters fail validation checks.
    """
    # Input validation
    if T_cycle <= 0:
        raise ValueError(f"T_cycle must be positive, got {T_cycle}")
    if PermShkCount <= 0:
        raise ValueError(f"PermShkCount must be positive, got {PermShkCount}")
    if TranShkCount <= 0:
        raise ValueError(f"TranShkCount must be positive, got {TranShkCount}")
    if len(PermShkStd) == 0:
        raise ValueError("PermShkStd cannot be empty")
    if len(TranShkStd) == 0:
        raise ValueError("TranShkStd cannot be empty")
    if not (0 <= UnempPrb <= 1):
        raise ValueError(f"UnempPrb must be in [0, 1], got {UnempPrb}")
    if not (0 <= UnempPrbRet <= 1):
        raise ValueError(f"UnempPrbRet must be in [0, 1], got {UnempPrbRet}")
    if RNG is None:
        raise ValueError("RNG cannot be None")

    IncShkDstnBase = construct_lognormal_income_process_unemployment(
        T_cycle,
        PermShkStd,
        PermShkCount,
        TranShkStd,
        TranShkCount,
        T_retire,
        UnempPrb,
        IncUnemp,
        UnempPrbRet,
        IncUnempRet,
        RNG,
        neutral_measure,
    )

    IncShkDstn = []
    for i in range(len(IncShkDstnBase.dstns)):
        IncShkDstn.append(
            DiscreteDistributionLabeled.from_unlabeled(
                IncShkDstnBase[i],
                name="Distribution of Shocks to Income",
                var_names=["perm", "tran"],
            )
        )
    return IncShkDstn


def make_labeled_risky_dstn(
    T_cycle: int,
    RiskyAvg: float,
    RiskyStd: float,
    RiskyCount: int,
    RNG: Generator,
) -> DiscreteDistributionLabeled:
    """
    Create a labeled risky asset return distribution.

    Wrapper around make_lognormal_RiskyDstn that converts the result
    to a labeled distribution.

    Parameters
    ----------
    T_cycle : int
        Number of periods in the cycle.
    RiskyAvg : float
        Mean risky return.
    RiskyStd : float
        Standard deviation of risky return.
    RiskyCount : int
        Number of risky return points.
    RNG : Generator
        Random number generator.

    Returns
    -------
    DiscreteDistributionLabeled
        Labeled distribution of risky asset returns.

    Raises
    ------
    ValueError
        If input parameters fail validation checks.
    """
    # Input validation
    if T_cycle <= 0:
        raise ValueError(f"T_cycle must be positive, got {T_cycle}")
    if RiskyAvg <= 0:
        raise ValueError(f"RiskyAvg must be positive, got {RiskyAvg}")
    if RiskyStd < 0:
        raise ValueError(f"RiskyStd must be non-negative, got {RiskyStd}")
    if RiskyCount <= 0:
        raise ValueError(f"RiskyCount must be positive, got {RiskyCount}")
    if RNG is None:
        raise ValueError("RNG cannot be None")

    RiskyDstnBase = make_lognormal_RiskyDstn(
        T_cycle, RiskyAvg, RiskyStd, RiskyCount, RNG
    )

    RiskyDstn = DiscreteDistributionLabeled.from_unlabeled(
        RiskyDstnBase,
        name="Distribution of Risky Asset Returns",
        var_names=["risky"],
    )
    return RiskyDstn


def make_labeled_shock_dstn(
    T_cycle: int,
    IncShkDstn: list[DiscreteDistributionLabeled],
    RiskyDstn: DiscreteDistributionLabeled,
) -> list[DiscreteDistributionLabeled]:
    """
    Create labeled joint shock distributions.

    Combines income shock and risky return distributions into a joint
    distribution with labeled variables.

    Parameters
    ----------
    T_cycle : int
        Number of periods in the cycle.
    IncShkDstn : list[DiscreteDistributionLabeled]
        List of income shock distributions.
    RiskyDstn : DiscreteDistributionLabeled
        Risky asset return distribution.

    Returns
    -------
    list[DiscreteDistributionLabeled]
        List of labeled joint shock distributions, one per period.

    Raises
    ------
    ValueError
        If input parameters fail validation checks.
    """
    # Input validation
    if T_cycle <= 0:
        raise ValueError(f"T_cycle must be positive, got {T_cycle}")
    if IncShkDstn is None or len(IncShkDstn) == 0:
        raise ValueError("IncShkDstn cannot be None or empty")
    if RiskyDstn is None:
        raise ValueError("RiskyDstn cannot be None")

    ShockDstnBase = combine_IncShkDstn_and_RiskyDstn(T_cycle, RiskyDstn, IncShkDstn)

    ShockDstn = []
    for i in range(len(ShockDstnBase.dstns)):
        ShockDstn.append(
            DiscreteDistributionLabeled.from_unlabeled(
                ShockDstnBase[i],
                name="Distribution of Shocks to Income and Risky Asset Returns",
                var_names=["perm", "tran", "risky"],
            )
        )
    return ShockDstn

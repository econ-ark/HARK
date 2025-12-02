"""
Consumption-saving models that also include medical spending.
"""

from copy import deepcopy

import numpy as np
from scipy.stats import norm
from scipy.special import erfc
from scipy.optimize import brentq

from HARK import AgentType
from HARK.Calibration.Income.IncomeProcesses import (
    construct_lognormal_income_process_unemployment,
    get_PermShkDstn_from_IncShkDstn,
    get_TranShkDstn_from_IncShkDstn,
    make_AR1_style_pLvlNextFunc,
    make_pLvlGrid_by_simulation,
    make_basic_pLvlPctiles,
    make_persistent_income_process_dict,
)
from HARK.ConsumptionSaving.ConsIndShockModel import (
    make_lognormal_kNrm_init_dstn,
    make_lognormal_pLvl_init_dstn,
)
from HARK.ConsumptionSaving.ConsGenIncProcessModel import (
    PersistentShockConsumerType,
    VariableLowerBoundFunc2D,
)
from HARK.ConsumptionSaving.ConsIndShockModel import ConsumerSolution
from HARK.distributions import (
    Lognormal,
    MultivariateLogNormal,
    add_discrete_outcome_constant_mean,
    expected,
)
from HARK.interpolation import (
    BilinearInterp,
    BilinearInterpOnInterp1D,
    ConstantFunction,
    CubicInterp,
    LinearInterp,
    LinearInterpOnInterp1D,
    LowerEnvelope3D,
    MargMargValueFuncCRRA,
    MargValueFuncCRRA,
    TrilinearInterp,
    UpperEnvelope,
    ValueFuncCRRA,
    VariableLowerBoundFunc3D,
)
from HARK.metric import MetricObject
from HARK.rewards import (
    CRRAutility,
    CRRAutilityP,
    CRRAutility_inv,
    CRRAutility_invP,
    CRRAutilityP_inv,
    CRRAutilityPP,
    UtilityFuncCRRA,
)
from HARK.utilities import NullFunc, make_grid_exp_mult, make_assets_grid, get_it_from

__all__ = [
    "MedShockPolicyFunc",
    "cThruXfunc",
    "MedThruXfunc",
    "MedShockConsumerType",
    "make_lognormal_MedShkDstn",
]

utility_inv = CRRAutility_inv
utilityP_inv = CRRAutilityP_inv
utility = CRRAutility
utility_invP = CRRAutility_invP
utilityPP = CRRAutilityPP


class MedShockPolicyFunc(MetricObject):
    """
    Class for representing the policy function in the medical shocks model: opt-
    imal consumption and medical care for given market resources, permanent income,
    and medical need shock.  Always obeys Con + MedPrice*Med = optimal spending.

    Parameters
    ----------
    xFunc : function
        Optimal total spending as a function of market resources, permanent
        income, and the medical need shock.
    xLvlGrid : np.array
        1D array of total expenditure levels.
    MedShkGrid : np.array
        1D array of medical shocks.
    MedPrice : float
        Relative price of a unit of medical care.
    CRRAcon : float
        Coefficient of relative risk aversion for consumption.
    CRRAmed : float
        Coefficient of relative risk aversion for medical care.
    xLvlCubicBool : boolean
        Indicator for whether cubic spline interpolation (rather than linear)
        should be used in the xLvl dimension.
    MedShkCubicBool : boolean
        Indicator for whether bicubic interpolation should be used; only
        operative when xLvlCubicBool=True.
    """

    distance_criteria = ["xFunc", "cFunc", "MedPrice"]

    def __init__(
        self,
        xFunc,
        xLvlGrid,
        MedShkGrid,
        MedPrice,
        CRRAcon,
        CRRAmed,
        xLvlCubicBool=False,
        MedShkCubicBool=False,
    ):
        # Store some of the inputs in self
        self.MedPrice = MedPrice
        self.xFunc = xFunc

        # Calculate optimal consumption at each combination of mLvl and MedShk.
        cLvlGrid = np.zeros(
            (xLvlGrid.size, MedShkGrid.size)
        )  # Initialize consumption grid
        for i in range(xLvlGrid.size):
            xLvl = xLvlGrid[i]
            for j in range(MedShkGrid.size):
                MedShk = MedShkGrid[j]
                if xLvl == 0:  # Zero consumption when mLvl = 0
                    cLvl = 0.0
                elif MedShk == 0:  # All consumption when MedShk = 0
                    cLvl = xLvl
                else:

                    def optMedZeroFunc(c):
                        return (MedShk / MedPrice) ** (-1.0 / CRRAcon) * (
                            (xLvl - c) / MedPrice
                        ) ** (CRRAmed / CRRAcon) - c

                    # Find solution to FOC
                    cLvl = brentq(optMedZeroFunc, 0.0, xLvl)
                cLvlGrid[i, j] = cLvl

        # Construct the consumption function and medical care function
        if xLvlCubicBool:
            if MedShkCubicBool:
                raise NotImplementedError("Bicubic interpolation not yet implemented")
            else:
                xLvlGrid_tiled = np.tile(
                    np.reshape(xLvlGrid, (xLvlGrid.size, 1)), (1, MedShkGrid.size)
                )
                MedShkGrid_tiled = np.tile(
                    np.reshape(MedShkGrid, (1, MedShkGrid.size)), (xLvlGrid.size, 1)
                )
                dfdx = (
                    (CRRAmed / (CRRAcon * MedPrice))
                    * (MedShkGrid_tiled / MedPrice) ** (-1.0 / CRRAcon)
                    * ((xLvlGrid_tiled - cLvlGrid) / MedPrice)
                    ** (CRRAmed / CRRAcon - 1.0)
                )
                dcdx = dfdx / (dfdx + 1.0)
                # approximation; function goes crazy otherwise
                dcdx[0, :] = dcdx[1, :]
                dcdx[:, 0] = 1.0  # no Med when MedShk=0, so all x is c
                cFromxFunc_by_MedShk = []
                for j in range(MedShkGrid.size):
                    cFromxFunc_by_MedShk.append(
                        CubicInterp(xLvlGrid, cLvlGrid[:, j], dcdx[:, j])
                    )
                cFunc = LinearInterpOnInterp1D(cFromxFunc_by_MedShk, MedShkGrid)
        else:
            cFunc = BilinearInterp(cLvlGrid, xLvlGrid, MedShkGrid)
        self.cFunc = cFunc

    def __call__(self, mLvl, pLvl, MedShk):
        """
        Evaluate optimal consumption and medical care at given levels of market
        resources, permanent income, and medical need shocks.

        Parameters
        ----------
        mLvl : np.array
            Market resource levels.
        pLvl : np.array
            Permanent income levels; should be same size as mLvl.
        MedShk : np.array
            Medical need shocks; should be same size as mLvl.

        Returns
        -------
        cLvl : np.array
            Optimal consumption for each point in (xLvl,MedShk).
        Med : np.array
            Optimal medical care for each point in (xLvl,MedShk).
        """
        xLvl = self.xFunc(mLvl, pLvl, MedShk)
        cLvl = self.cFunc(xLvl, MedShk)
        Med = (xLvl - cLvl) / self.MedPrice
        return cLvl, Med

    def derivativeX(self, mLvl, pLvl, MedShk):
        """
        Evaluate the derivative of consumption and medical care with respect to
        market resources at given levels of market resources, permanent income,
        and medical need shocks.

        Parameters
        ----------
        mLvl : np.array
            Market resource levels.
        pLvl : np.array
            Permanent income levels; should be same size as mLvl.
        MedShk : np.array
            Medical need shocks; should be same size as mLvl.

        Returns
        -------
        dcdm : np.array
            Derivative of consumption with respect to market resources for each
            point in (xLvl,MedShk).
        dMeddm : np.array
            Derivative of medical care with respect to market resources for each
            point in (xLvl,MedShk).
        """
        xLvl = self.xFunc(mLvl, pLvl, MedShk)
        dxdm = self.xFunc.derivativeX(mLvl, pLvl, MedShk)
        dcdx = self.cFunc.derivativeX(xLvl, MedShk)
        dcdm = dxdm * dcdx
        dMeddm = (dxdm - dcdm) / self.MedPrice
        return dcdm, dMeddm

    def derivativeY(self, mLvl, pLvl, MedShk):
        """
        Evaluate the derivative of consumption and medical care with respect to
        permanent income at given levels of market resources, permanent income,
        and medical need shocks.

        Parameters
        ----------
        mLvl : np.array
            Market resource levels.
        pLvl : np.array
            Permanent income levels; should be same size as mLvl.
        MedShk : np.array
            Medical need shocks; should be same size as mLvl.

        Returns
        -------
        dcdp : np.array
            Derivative of consumption with respect to permanent income for each
            point in (xLvl,MedShk).
        dMeddp : np.array
            Derivative of medical care with respect to permanent income for each
            point in (xLvl,MedShk).
        """
        xLvl = self.xFunc(mLvl, pLvl, MedShk)
        dxdp = self.xFunc.derivativeY(mLvl, pLvl, MedShk)
        dcdx = self.cFunc.derivativeX(xLvl, MedShk)
        dcdp = dxdp * dcdx
        dMeddp = (dxdp - dcdp) / self.MedPrice
        return dcdp, dMeddp

    def derivativeZ(self, mLvl, pLvl, MedShk):
        """
        Evaluate the derivative of consumption and medical care with respect to
        medical need shock at given levels of market resources, permanent income,
        and medical need shocks.

        Parameters
        ----------
        mLvl : np.array
            Market resource levels.
        pLvl : np.array
            Permanent income levels; should be same size as mLvl.
        MedShk : np.array
            Medical need shocks; should be same size as mLvl.

        Returns
        -------
        dcdShk : np.array
            Derivative of consumption with respect to medical need for each
            point in (xLvl,MedShk).
        dMeddShk : np.array
            Derivative of medical care with respect to medical need for each
            point in (xLvl,MedShk).
        """
        xLvl = self.xFunc(mLvl, pLvl, MedShk)
        dxdShk = self.xFunc.derivativeZ(mLvl, pLvl, MedShk)
        dcdx = self.cFunc.derivativeX(xLvl, MedShk)
        dcdShk = dxdShk * dcdx + self.cFunc.derivativeY(xLvl, MedShk)
        dMeddShk = (dxdShk - dcdShk) / self.MedPrice
        return dcdShk, dMeddShk


class cThruXfunc(MetricObject):
    """
    Class for representing consumption function derived from total expenditure
    and consumption.

    Parameters
    ----------
    xFunc : function
        Optimal total spending as a function of market resources, permanent
        income, and the medical need shock.
    cFunc : function
        Optimal consumption as a function of total spending and the medical
        need shock.
    """

    distance_criteria = ["xFunc", "cFunc"]

    def __init__(self, xFunc, cFunc):
        self.xFunc = xFunc
        self.cFunc = cFunc

    def __call__(self, mLvl, pLvl, MedShk):
        """
        Evaluate optimal consumption at given levels of market resources, perma-
        nent income, and medical need shocks.

        Parameters
        ----------
        mLvl : np.array
            Market resource levels.
        pLvl : np.array
            Permanent income levels; should be same size as mLvl.
        MedShk : np.array
            Medical need shocks; should be same size as mLvl.

        Returns
        -------
        cLvl : np.array
            Optimal consumption for each point in (xLvl,MedShk).
        """
        xLvl = self.xFunc(mLvl, pLvl, MedShk)
        cLvl = self.cFunc(xLvl, MedShk)
        return cLvl

    def derivativeX(self, mLvl, pLvl, MedShk):
        """
        Evaluate the derivative of consumption with respect to market resources
        at given levels of market resources, permanent income, and medical need
        shocks.

        Parameters
        ----------
        mLvl : np.array
            Market resource levels.
        pLvl : np.array
            Permanent income levels; should be same size as mLvl.
        MedShk : np.array
            Medical need shocks; should be same size as mLvl.

        Returns
        -------
        dcdm : np.array
            Derivative of consumption with respect to market resources for each
            point in (xLvl,MedShk).
        """
        xLvl = self.xFunc(mLvl, pLvl, MedShk)
        dxdm = self.xFunc.derivativeX(mLvl, pLvl, MedShk)
        dcdx = self.cFunc.derivativeX(xLvl, MedShk)
        dcdm = dxdm * dcdx
        return dcdm

    def derivativeY(self, mLvl, pLvl, MedShk):
        """
        Evaluate the derivative of consumption and medical care with respect to
        permanent income at given levels of market resources, permanent income,
        and medical need shocks.

        Parameters
        ----------
        mLvl : np.array
            Market resource levels.
        pLvl : np.array
            Permanent income levels; should be same size as mLvl.
        MedShk : np.array
            Medical need shocks; should be same size as mLvl.

        Returns
        -------
        dcdp : np.array
            Derivative of consumption with respect to permanent income for each
            point in (xLvl,MedShk).
        """
        xLvl = self.xFunc(mLvl, pLvl, MedShk)
        dxdp = self.xFunc.derivativeY(mLvl, pLvl, MedShk)
        dcdx = self.cFunc.derivativeX(xLvl, MedShk)
        dcdp = dxdp * dcdx
        return dcdp

    def derivativeZ(self, mLvl, pLvl, MedShk):
        """
        Evaluate the derivative of consumption and medical care with respect to
        medical need shock at given levels of market resources, permanent income,
        and medical need shocks.

        Parameters
        ----------
        mLvl : np.array
            Market resource levels.
        pLvl : np.array
            Permanent income levels; should be same size as mLvl.
        MedShk : np.array
            Medical need shocks; should be same size as mLvl.

        Returns
        -------
        dcdShk : np.array
            Derivative of consumption with respect to medical need for each
            point in (xLvl,MedShk).
        """
        xLvl = self.xFunc(mLvl, pLvl, MedShk)
        dxdShk = self.xFunc.derivativeZ(mLvl, pLvl, MedShk)
        dcdx = self.cFunc.derivativeX(xLvl, MedShk)
        dcdShk = dxdShk * dcdx + self.cFunc.derivativeY(xLvl, MedShk)
        return dcdShk


class MedThruXfunc(MetricObject):
    """
    Class for representing medical care function derived from total expenditure
    and consumption.

    Parameters
    ----------
    xFunc : function
        Optimal total spending as a function of market resources, permanent
        income, and the medical need shock.
    cFunc : function
        Optimal consumption as a function of total spending and the medical
        need shock.
    MedPrice : float
        Relative price of a unit of medical care.
    """

    distance_criteria = ["xFunc", "cFunc", "MedPrice"]

    def __init__(self, xFunc, cFunc, MedPrice):
        self.xFunc = xFunc
        self.cFunc = cFunc
        self.MedPrice = MedPrice

    def __call__(self, mLvl, pLvl, MedShk):
        """
        Evaluate optimal medical care at given levels of market resources,
        permanent income, and medical need shock.

        Parameters
        ----------
        mLvl : np.array
            Market resource levels.
        pLvl : np.array
            Permanent income levels; should be same size as mLvl.
        MedShk : np.array
            Medical need shocks; should be same size as mLvl.

        Returns
        -------
        Med : np.array
            Optimal medical care for each point in (xLvl,MedShk).
        """
        xLvl = self.xFunc(mLvl, pLvl, MedShk)
        Med = (xLvl - self.cFunc(xLvl, MedShk)) / self.MedPrice
        return Med

    def derivativeX(self, mLvl, pLvl, MedShk):
        """
        Evaluate the derivative of consumption and medical care with respect to
        market resources at given levels of market resources, permanent income,
        and medical need shocks.

        Parameters
        ----------
        mLvl : np.array
            Market resource levels.
        pLvl : np.array
            Permanent income levels; should be same size as mLvl.
        MedShk : np.array
            Medical need shocks; should be same size as mLvl.

        Returns
        -------
        dcdm : np.array
            Derivative of consumption with respect to market resources for each
            point in (xLvl,MedShk).
        dMeddm : np.array
            Derivative of medical care with respect to market resources for each
            point in (xLvl,MedShk).
        """
        xLvl = self.xFunc(mLvl, pLvl, MedShk)
        dxdm = self.xFunc.derivativeX(mLvl, pLvl, MedShk)
        dcdx = self.cFunc.derivativeX(xLvl, MedShk)
        dcdm = dxdm * dcdx
        dMeddm = (dxdm - dcdm) / self.MedPrice
        return dMeddm

    def derivativeY(self, mLvl, pLvl, MedShk):
        """
        Evaluate the derivative of medical care with respect to permanent income
        at given levels of market resources, permanent income, and medical need
        shocks.

        Parameters
        ----------
        mLvl : np.array
            Market resource levels.
        pLvl : np.array
            Permanent income levels; should be same size as mLvl.
        MedShk : np.array
            Medical need shocks; should be same size as mLvl.

        Returns
        -------
        dMeddp : np.array
            Derivative of medical care with respect to permanent income for each
            point in (xLvl,MedShk).
        """
        xLvl = self.xFunc(mLvl, pLvl, MedShk)
        dxdp = self.xFunc.derivativeY(mLvl, pLvl, MedShk)
        dMeddp = (dxdp - dxdp * self.cFunc.derivativeX(xLvl, MedShk)) / self.MedPrice
        return dMeddp

    def derivativeZ(self, mLvl, pLvl, MedShk):
        """
        Evaluate the derivative of medical care with respect to medical need
        shock at given levels of market resources, permanent income, and medical
        need shocks.

        Parameters
        ----------
        mLvl : np.array
            Market resource levels.
        pLvl : np.array
            Permanent income levels; should be same size as mLvl.
        MedShk : np.array
            Medical need shocks; should be same size as mLvl.

        Returns
        -------
        dMeddShk : np.array
            Derivative of medical care with respect to medical need for each
            point in (xLvl,MedShk).
        """
        xLvl = self.xFunc(mLvl, pLvl, MedShk)
        dxdShk = self.xFunc.derivativeZ(mLvl, pLvl, MedShk)
        dcdx = self.cFunc.derivativeX(xLvl, MedShk)
        dcdShk = dxdShk * dcdx + self.cFunc.derivativeY(xLvl, MedShk)
        dMeddShk = (dxdShk - dcdShk) / self.MedPrice
        return dMeddShk


def make_market_resources_grid(mNrmMin, mNrmMax, mNrmNestFac, mNrmCount, mNrmExtra):
    """
    Constructor for mNrmGrid that aliases make_assets_grid.
    """
    return make_assets_grid(mNrmMin, mNrmMax, mNrmCount, mNrmExtra, mNrmNestFac)


def make_capital_grid(kLvlMin, kLvlMax, kLvlCount, kLvlOrder):
    """
    Constructor for kLvlGrid, using a simple "invertible" format.
    """
    base_grid = np.linspace(0.0, 1.0, kLvlCount) ** kLvlOrder
    kLvlGrid = (kLvlMax - kLvlMin) * base_grid + kLvlMin
    return kLvlGrid


def reformat_bequest_motive(BeqMPC, BeqInt, CRRA):
    """
    Reformats interpretable bequest motive parameters (terminal intercept and MPC)
    into parameters that are easily useable in math (shifter and scaler).
    """
    BeqParamDict = {
        "BeqFac": BeqMPC ** (-CRRA),
        "BeqShift": BeqInt / BeqMPC,
    }
    return BeqParamDict


def make_lognormal_MedShkDstn(
    T_cycle,
    MedShkAvg,
    MedShkStd,
    MedShkCount,
    MedShkCountTail,
    RNG,
    MedShkTailBound=[0.0, 0.9],
):
    r"""
    Constructs discretized lognormal distributions of medical preference shocks
    for each period in the cycle.

    .. math::
        \text{ medShk}_t \sim \exp(\mathcal{N}(\textbf{MedShkStd}^2)) \\
        \mathbb{E}[\text{medShk}_t]=\textbf{MedShkAvg}


    Parameters
    ----------
    T_cycle : int
        Number of non-terminal periods in the agent's cycle.
    MedShkAvg : [float]
        Mean of medical needs shock in each period of the problem.
    MedShkStd : [float]
        Standard deviation of log medical needs shock in each period of the problem.
    MedShkCount : int
        Number of equiprobable nodes in the "body" of the discretization.
    MedShkCountTail : int
        Number of nodes in each "tail" of the discretization.
    RNG : RandomState
        The AgentType's internal random number generator.
    MedShkTailBound : [float,float]
        CDF bounds for the tail of the discretization.

    Returns
    -------
    MedShkDstn : [DiscreteDistribuion]
    """
    MedShkDstn = []  # empty list for medical shock distribution each period
    for t in range(T_cycle):
        # get shock distribution parameters
        MedShkAvg_t = MedShkAvg[t]
        MedShkStd_t = MedShkStd[t]
        MedShkDstn_t = Lognormal(
            mu=np.log(MedShkAvg_t) - 0.5 * MedShkStd_t**2,
            sigma=MedShkStd_t,
            seed=RNG.integers(0, 2**31 - 1),
        ).discretize(
            N=MedShkCount,
            method="equiprobable",
            tail_N=MedShkCountTail,
            tail_bound=MedShkTailBound,
        )
        MedShkDstn_t = add_discrete_outcome_constant_mean(
            MedShkDstn_t, 0.0, 0.0, sort=True
        )  # add point at zero with no probability
        MedShkDstn.append(MedShkDstn_t)
    return MedShkDstn


def make_continuous_MedShockDstn(
    MedShkLogMean, MedShkLogStd, MedCostLogMean, MedCostLogStd, MedCorr, T_cycle, RNG
):
    """
    Construct a time-varying list of bivariate lognormals for the medical shocks
    distribution. This representation uses fully continuous distributions, with
    no discretization in either dimension.

    Parameters
    ----------
    MedShkLogMean : [float]
        Age-varying list of means of log medical needs (utility) shocks.
    MedShkLogStd : [float]
        Age-varying list of standard deviations of log medical needs (utility) shocks.
    MedCostLogMean : [float]
        Age-varying list of means of log medical expense shocks.
    MedCostLogStd : [float]
        Age-varying list of standard deviations of log medical expense shocks..
    MedCorr : [float]
        Age-varying correlation coefficient between log medical expenses and utility shocks.
    T_cycle : int
        Number of periods in the agent's sequence.
    RNG : RandomState
        Random number generator for this type.

    Returns
    -------
    MedShockDstn : [MultivariateLognormal]
        Age-varying list of bivariate lognormal distributions, ordered as (MedCost,MedShk).
    """
    MedShockDstn = []
    for t in range(T_cycle):
        s1 = MedCostLogStd[t]
        s2 = MedShkLogStd[t]
        diag = MedCorr[t] * s1 * s2
        S = np.array([[s1**2, diag], [diag, s2**2]])
        M = np.array([MedCostLogMean[t], MedShkLogMean[t]])
        seed_t = RNG.integers(0, 2**31 - 1)
        dstn_t = MultivariateLogNormal(mu=M, Sigma=S, seed=seed_t)
        MedShockDstn.append(dstn_t)
    return MedShockDstn


def make_MedShock_solution_terminal(
    CRRA, CRRAmed, MedShkDstn, MedPrice, aXtraGrid, pLvlGrid, CubicBool
):
    """
    Construct the terminal period solution for this type.  Similar to other models,
    optimal behavior involves spending all available market resources; however,
    the agent must split his resources between consumption and medical care.

    Parameters
    ----------
    None

    Returns:
    --------
    None
    """
    # Take last period data, whichever way time is flowing
    MedPrice = MedPrice[-1]
    MedShkVals = MedShkDstn[-1].atoms.flatten()
    MedShkPrbs = MedShkDstn[-1].pmv

    # Initialize grids of medical need shocks, market resources, and optimal consumption
    MedShkGrid = MedShkVals
    xLvlMin = np.min(aXtraGrid) * np.min(pLvlGrid)
    xLvlMax = np.max(aXtraGrid) * np.max(pLvlGrid)
    xLvlGrid = make_grid_exp_mult(xLvlMin, xLvlMax, 3 * aXtraGrid.size, 8)
    trivial_grid = np.array([0.0, 1.0])  # Trivial grid

    # Make the policy functions for the terminal period
    xFunc_terminal = TrilinearInterp(
        np.array([[[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]]),
        trivial_grid,
        trivial_grid,
        trivial_grid,
    )
    policyFunc_terminal = MedShockPolicyFunc(
        xFunc_terminal,
        xLvlGrid,
        MedShkGrid,
        MedPrice,
        CRRA,
        CRRAmed,
        xLvlCubicBool=CubicBool,
    )
    cFunc_terminal = cThruXfunc(xFunc_terminal, policyFunc_terminal.cFunc)
    MedFunc_terminal = MedThruXfunc(xFunc_terminal, policyFunc_terminal.cFunc, MedPrice)

    # Calculate optimal consumption on a grid of market resources and medical shocks
    mLvlGrid = xLvlGrid
    mLvlGrid_tiled = np.tile(
        np.reshape(mLvlGrid, (mLvlGrid.size, 1)), (1, MedShkGrid.size)
    )
    pLvlGrid_tiled = np.ones_like(
        mLvlGrid_tiled
    )  # permanent income irrelevant in terminal period
    MedShkGrid_tiled = np.tile(
        np.reshape(MedShkVals, (1, MedShkGrid.size)), (mLvlGrid.size, 1)
    )
    cLvlGrid, MedGrid = policyFunc_terminal(
        mLvlGrid_tiled, pLvlGrid_tiled, MedShkGrid_tiled
    )

    # Integrate marginal value across shocks to get expected marginal value
    vPgrid = cLvlGrid ** (-CRRA)
    vPgrid[np.isinf(vPgrid)] = 0.0  # correct for issue at bottom edges
    PrbGrid = np.tile(np.reshape(MedShkPrbs, (1, MedShkGrid.size)), (mLvlGrid.size, 1))
    vP_expected = np.sum(vPgrid * PrbGrid, axis=1)

    # Construct the marginal (marginal) value function for the terminal period
    vPnvrs = vP_expected ** (-1.0 / CRRA)
    vPnvrs[0] = 0.0
    vPnvrsFunc = BilinearInterp(
        np.tile(np.reshape(vPnvrs, (vPnvrs.size, 1)), (1, trivial_grid.size)),
        mLvlGrid,
        trivial_grid,
    )
    vPfunc_terminal = MargValueFuncCRRA(vPnvrsFunc, CRRA)
    vPPfunc_terminal = MargMargValueFuncCRRA(vPnvrsFunc, CRRA)

    # Integrate value across shocks to get expected value
    vGrid = utility(cLvlGrid, rho=CRRA) + MedShkGrid_tiled * utility(
        MedGrid, rho=CRRAmed
    )
    # correct for issue when MedShk=0
    vGrid[:, 0] = utility(cLvlGrid[:, 0], rho=CRRA)
    vGrid[np.isinf(vGrid)] = 0.0  # correct for issue at bottom edges
    v_expected = np.sum(vGrid * PrbGrid, axis=1)

    # Construct the value function for the terminal period
    vNvrs = utility_inv(v_expected, rho=CRRA)
    vNvrs[0] = 0.0
    vNvrsP = vP_expected * utility_invP(v_expected, rho=CRRA)
    # TODO: Figure out MPCmax in this model
    vNvrsP[0] = 0.0
    tempFunc = CubicInterp(mLvlGrid, vNvrs, vNvrsP)
    vNvrsFunc = LinearInterpOnInterp1D([tempFunc, tempFunc], trivial_grid)
    vFunc_terminal = ValueFuncCRRA(vNvrsFunc, CRRA)

    # Make and return the terminal period solution
    solution_terminal = ConsumerSolution(
        cFunc=cFunc_terminal,
        vFunc=vFunc_terminal,
        vPfunc=vPfunc_terminal,
        vPPfunc=vPPfunc_terminal,
        hNrm=0.0,
        mNrmMin=0.0,
    )
    solution_terminal.MedFunc = MedFunc_terminal
    solution_terminal.policyFunc = policyFunc_terminal
    # Track absolute human wealth and minimum market wealth by permanent income
    solution_terminal.hLvl = ConstantFunction(0.0)
    solution_terminal.mLvlMin = ConstantFunction(0.0)
    return solution_terminal


###############################################################################


def solve_one_period_ConsMedShock(
    solution_next,
    IncShkDstn,
    MedShkDstn,
    LivPrb,
    DiscFac,
    CRRA,
    CRRAmed,
    Rfree,
    MedPrice,
    pLvlNextFunc,
    BoroCnstArt,
    aXtraGrid,
    pLvlGrid,
    vFuncBool,
    CubicBool,
):
    """
    Class for solving the one period problem for the "medical shocks" model, in
    which consumers receive shocks to permanent and transitory income as well as
    shocks to "medical need"-- multiplicative utility shocks for a second good.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncShkDstn : distribution.Distribution
        A discrete approximation to the income process between the period being
        solved and the one immediately following (in solution_next).
    MedShkDstn : distribution.Distribution
        Discrete distribution of the multiplicative utility shifter for medical care.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion for composite consumption.
    CRRAmed : float
        Coefficient of relative risk aversion for medical care.
    Rfree : float
        Risk free interest factor on end-of-period assets.
    MedPrice : float
        Price of unit of medical care relative to unit of consumption.
    pLvlNextFunc : float
        Expected permanent income next period as a function of current pLvl.
    BoroCnstArt: float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.
    aXtraGrid: np.array
        Array of "extra" end-of-period (normalized) asset values-- assets
        above the absolute minimum acceptable level.
    pLvlGrid: np.array
        Array of permanent income levels at which to solve the problem.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear inter-
        polation.

    Returns
    -------
    solution_now : ConsumerSolution
        Solution to this period's consumption-saving problem.
    """
    # Define the utility functions for this period
    uFunc = UtilityFuncCRRA(CRRA)
    uMed = UtilityFuncCRRA(CRRAmed)  # Utility function for medical care
    DiscFacEff = DiscFac * LivPrb  # "effective" discount factor

    # Unpack next period's income shock distribution
    ShkPrbsNext = IncShkDstn.pmv
    PermShkValsNext = IncShkDstn.atoms[0]
    TranShkValsNext = IncShkDstn.atoms[1]
    PermShkMinNext = np.min(PermShkValsNext)
    TranShkMinNext = np.min(TranShkValsNext)
    MedShkPrbs = MedShkDstn.pmv
    MedShkVals = MedShkDstn.atoms.flatten()

    # Calculate the probability that we get the worst possible income draw
    IncNext = PermShkValsNext * TranShkValsNext
    WorstIncNext = PermShkMinNext * TranShkMinNext
    WorstIncPrb = np.sum(ShkPrbsNext[IncNext == WorstIncNext])
    # WorstIncPrb is the "Weierstrass p" concept: the odds we get the WORST thing

    # Unpack next period's (marginal) value function
    vFuncNext = solution_next.vFunc  # This is None when vFuncBool is False
    vPfuncNext = solution_next.vPfunc
    vPPfuncNext = solution_next.vPPfunc  # This is None when CubicBool is False

    # Update the bounding MPCs and PDV of human wealth:
    PatFac = ((Rfree * DiscFacEff) ** (1.0 / CRRA)) / Rfree
    try:
        MPCminNow = 1.0 / (1.0 + PatFac / solution_next.MPCmin)
    except:
        MPCminNow = 0.0
    mLvlMinNext = solution_next.mLvlMin

    # TODO: Deal with this unused code for the upper bound of MPC (should be a function now)
    # Ex_IncNext = np.dot(ShkPrbsNext, TranShkValsNext * PermShkValsNext)
    # hNrmNow = 0.0
    # temp_fac = (WorstIncPrb ** (1.0 / CRRA)) * PatFac
    # MPCmaxNow = 1.0 / (1.0 + temp_fac / solution_next.MPCmax)

    # Define some functions for calculating future expectations
    def calc_pLvl_next(S, p):
        return pLvlNextFunc(p) * S["PermShk"]

    def calc_mLvl_next(S, a, p_next):
        return Rfree * a + p_next * S["TranShk"]

    def calc_hLvl(S, p):
        pLvl_next = calc_pLvl_next(S, p)
        hLvl = S["TranShk"] * pLvl_next + solution_next.hLvl(pLvl_next)
        return hLvl

    def calc_v_next(S, a, p):
        pLvl_next = calc_pLvl_next(S, p)
        mLvl_next = calc_mLvl_next(S, a, pLvl_next)
        v_next = vFuncNext(mLvl_next, pLvl_next)
        return v_next

    def calc_vP_next(S, a, p):
        pLvl_next = calc_pLvl_next(S, p)
        mLvl_next = calc_mLvl_next(S, a, pLvl_next)
        vP_next = vPfuncNext(mLvl_next, pLvl_next)
        return vP_next

    def calc_vPP_next(S, a, p):
        pLvl_next = calc_pLvl_next(S, p)
        mLvl_next = calc_mLvl_next(S, a, pLvl_next)
        vPP_next = vPPfuncNext(mLvl_next, pLvl_next)
        return vPP_next

    # Construct human wealth level as a function of productivity pLvl
    hLvlGrid = 1.0 / Rfree * expected(calc_hLvl, IncShkDstn, args=(pLvlGrid))
    hLvlNow = LinearInterp(np.insert(pLvlGrid, 0, 0.0), np.insert(hLvlGrid, 0, 0.0))

    # Make temporary grids of income shocks and next period income values
    ShkCount = TranShkValsNext.size
    pLvlCount = pLvlGrid.size
    PermShkVals_temp = np.tile(
        np.reshape(PermShkValsNext, (1, ShkCount)), (pLvlCount, 1)
    )
    TranShkVals_temp = np.tile(
        np.reshape(TranShkValsNext, (1, ShkCount)), (pLvlCount, 1)
    )
    pLvlNext_temp = (
        np.tile(
            np.reshape(pLvlNextFunc(pLvlGrid), (pLvlCount, 1)),
            (1, ShkCount),
        )
        * PermShkVals_temp
    )

    # Find the natural borrowing constraint for each persistent income level
    aLvlMin_candidates = (
        mLvlMinNext(pLvlNext_temp) - TranShkVals_temp * pLvlNext_temp
    ) / Rfree
    aLvlMinNow = np.max(aLvlMin_candidates, axis=1)
    BoroCnstNat = LinearInterp(
        np.insert(pLvlGrid, 0, 0.0), np.insert(aLvlMinNow, 0, 0.0)
    )

    # Define the minimum allowable mLvl by pLvl as the greater of the natural and artificial borrowing constraints
    if BoroCnstArt is not None:
        BoroCnstArt = LinearInterp(np.array([0.0, 1.0]), np.array([0.0, BoroCnstArt]))
        mLvlMinNow = UpperEnvelope(BoroCnstArt, BoroCnstNat)
    else:
        mLvlMinNow = BoroCnstNat

    # Make the constrained total spending function: spend all market resources
    trivial_grid = np.array([0.0, 1.0])  # Trivial grid
    spendAllFunc = TrilinearInterp(
        np.array([[[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]]),
        trivial_grid,
        trivial_grid,
        trivial_grid,
    )
    xFuncNowCnst = VariableLowerBoundFunc3D(spendAllFunc, mLvlMinNow)

    # Define grids of pLvl and aLvl on which to compute future expectations
    pLvlCount = pLvlGrid.size
    aNrmCount = aXtraGrid.size
    MedCount = MedShkVals.size
    pLvlNow = np.tile(pLvlGrid, (aNrmCount, 1)).transpose()
    aLvlNow = np.tile(aXtraGrid, (pLvlCount, 1)) * pLvlNow + BoroCnstNat(pLvlNow)
    # shape = (pLvlCount,aNrmCount)
    if pLvlGrid[0] == 0.0:  # aLvl turns out badly if pLvl is 0 at bottom
        aLvlNow[0, :] = aXtraGrid

    # Calculate end-of-period marginal value of assets
    EndOfPrd_vP = (
        DiscFacEff * Rfree * expected(calc_vP_next, IncShkDstn, args=(aLvlNow, pLvlNow))
    )

    # If the value function has been requested, construct the end-of-period vFunc
    if vFuncBool:
        # Compute expected value from end-of-period states
        EndOfPrd_v = expected(calc_v_next, IncShkDstn, args=(aLvlNow, pLvlNow))
        EndOfPrd_v *= DiscFacEff

        # Transformed value through inverse utility function to "decurve" it
        EndOfPrd_vNvrs = uFunc.inv(EndOfPrd_v)
        EndOfPrd_vNvrsP = EndOfPrd_vP * uFunc.derinv(EndOfPrd_v, order=(0, 1))

        # Add points at mLvl=zero
        EndOfPrd_vNvrs = np.concatenate(
            (np.zeros((pLvlCount, 1)), EndOfPrd_vNvrs), axis=1
        )
        EndOfPrd_vNvrsP = np.concatenate(
            (
                np.reshape(EndOfPrd_vNvrsP[:, 0], (pLvlCount, 1)),
                EndOfPrd_vNvrsP,
            ),
            axis=1,
        )
        # This is a very good approximation, vNvrsPP = 0 at the asset minimum

        # Make a temporary aLvl grid for interpolating the end-of-period value function
        aLvl_temp = np.concatenate(
            (
                np.reshape(BoroCnstNat(pLvlGrid), (pLvlGrid.size, 1)),
                aLvlNow,
            ),
            axis=1,
        )

        # Make an end-of-period value function for each persistent income level in the grid
        EndOfPrd_vNvrsFunc_list = []
        for p in range(pLvlCount):
            EndOfPrd_vNvrsFunc_list.append(
                CubicInterp(
                    aLvl_temp[p, :] - BoroCnstNat(pLvlGrid[p]),
                    EndOfPrd_vNvrs[p, :],
                    EndOfPrd_vNvrsP[p, :],
                )
            )
        EndOfPrd_vNvrsFuncBase = LinearInterpOnInterp1D(
            EndOfPrd_vNvrsFunc_list, pLvlGrid
        )

        # Re-adjust the combined end-of-period value function to account for the
        # natural borrowing constraint shifter and "re-curve" it
        EndOfPrd_vNvrsFunc = VariableLowerBoundFunc2D(
            EndOfPrd_vNvrsFuncBase, BoroCnstNat
        )
        EndOfPrd_vFunc = ValueFuncCRRA(EndOfPrd_vNvrsFunc, CRRA)

    # Solve the first order condition to get optimal consumption and medical
    # spending, then find the endogenous mLvl gridpoints
    # Calculate endogenous gridpoints and controls
    cLvlNow = np.tile(
        np.reshape(uFunc.derinv(EndOfPrd_vP, order=(1, 0)), (1, pLvlCount, aNrmCount)),
        (MedCount, 1, 1),
    )
    MedBaseNow = np.tile(
        np.reshape(
            uMed.derinv(MedPrice * EndOfPrd_vP, order=(1, 0)),
            (1, pLvlCount, aNrmCount),
        ),
        (MedCount, 1, 1),
    )
    MedShkVals_tiled = np.tile(  # This includes CRRA adjustment
        np.reshape(MedShkVals ** (1.0 / CRRAmed), (MedCount, 1, 1)),
        (1, pLvlCount, aNrmCount),
    )
    MedLvlNow = MedShkVals_tiled * MedBaseNow
    aLvlNow_tiled = np.tile(
        np.reshape(aLvlNow, (1, pLvlCount, aNrmCount)), (MedCount, 1, 1)
    )
    xLvlNow = cLvlNow + MedPrice * MedLvlNow
    mLvlNow = xLvlNow + aLvlNow_tiled

    # Limiting consumption is zero as m approaches the natural borrowing constraint
    x_for_interpolation = np.concatenate(
        (np.zeros((MedCount, pLvlCount, 1)), xLvlNow), axis=-1
    )
    temp = np.tile(
        BoroCnstNat(np.reshape(pLvlGrid, (1, pLvlCount, 1))),
        (MedCount, 1, 1),
    )
    m_for_interpolation = np.concatenate((temp, mLvlNow), axis=-1)

    # Make a 3D array of permanent income for interpolation
    p_for_interpolation = np.tile(
        np.reshape(pLvlGrid, (1, pLvlCount, 1)), (MedCount, 1, aNrmCount + 1)
    )

    MedShkVals_tiled = np.tile(  # This does *not* have the CRRA adjustment
        np.reshape(MedShkVals, (MedCount, 1, 1)), (1, pLvlCount, aNrmCount)
    )

    # Build the set of cFuncs by pLvl, gathered in a list
    xFunc_by_pLvl_and_MedShk = []  # Initialize the empty list of lists of 1D xFuncs
    if CubicBool:
        # Calculate end-of-period marginal marginal value of assets
        vPP_fac = DiscFacEff * Rfree * Rfree
        EndOfPrd_vPP = expected(calc_vPP_next, IncShkDstn, args=(aLvlNow, pLvlNow))
        EndOfPrd_vPP *= vPP_fac
        EndOfPrd_vPP = np.tile(
            np.reshape(EndOfPrd_vPP, (1, pLvlCount, aNrmCount)), (MedCount, 1, 1)
        )

        # Calculate the MPC and MPM at each gridpoint
        dcda = EndOfPrd_vPP / uFunc.der(np.array(cLvlNow), order=2)
        dMedda = EndOfPrd_vPP / (MedShkVals_tiled * uMed.der(MedLvlNow, order=2))
        dMedda[0, :, :] = 0.0  # dMedda goes crazy when MedShk=0
        MPC = dcda / (1.0 + dcda + MedPrice * dMedda)
        MPM = dMedda / (1.0 + dcda + MedPrice * dMedda)

        # Convert to marginal propensity to spend
        MPX = MPC + MedPrice * MPM
        MPX = np.concatenate(
            (np.reshape(MPX[:, :, 0], (MedCount, pLvlCount, 1)), MPX), axis=2
        )  # NEED TO CALCULATE MPM AT NATURAL BORROWING CONSTRAINT
        MPX[0, :, 0] = 1.0

        # Loop over each permanent income level and medical shock and make a cubic xFunc
        xFunc_by_pLvl_and_MedShk = []  # Initialize the empty list of lists of 1D xFuncs
        for i in range(pLvlCount):
            temp_list = []
            pLvl_i = p_for_interpolation[0, i, 0]
            mLvlMin_i = BoroCnstNat(pLvl_i)
            for j in range(MedCount):
                m_temp = m_for_interpolation[j, i, :] - mLvlMin_i
                x_temp = x_for_interpolation[j, i, :]
                MPX_temp = MPX[j, i, :]
                temp_list.append(CubicInterp(m_temp, x_temp, MPX_temp))
            xFunc_by_pLvl_and_MedShk.append(deepcopy(temp_list))

    # Basic version: use linear interpolation within a pLvl and MedShk
    else:
        # Loop over pLvl and then MedShk within that
        for i in range(pLvlCount):
            temp_list = []
            pLvl_i = p_for_interpolation[0, i, 0]
            mLvlMin_i = BoroCnstNat(pLvl_i)
            for j in range(MedCount):
                m_temp = m_for_interpolation[j, i, :] - mLvlMin_i
                x_temp = x_for_interpolation[j, i, :]
                temp_list.append(LinearInterp(m_temp, x_temp))
            xFunc_by_pLvl_and_MedShk.append(deepcopy(temp_list))

    # Combine the nested list of linear xFuncs into a single function
    pLvl_temp = p_for_interpolation[0, :, 0]
    MedShk_temp = MedShkVals_tiled[:, 0, 0]
    xFuncUncBase = BilinearInterpOnInterp1D(
        xFunc_by_pLvl_and_MedShk, pLvl_temp, MedShk_temp
    )
    xFuncNowUnc = VariableLowerBoundFunc3D(xFuncUncBase, BoroCnstNat)
    # Re-adjust for lower bound of natural borrowing constraint

    # Combine the constrained and unconstrained functions into the true consumption function
    xFuncNow = LowerEnvelope3D(xFuncNowUnc, xFuncNowCnst)

    # Transform the expenditure function into policy functions for consumption and medical care
    aug_factor = 2
    xLvlGrid = make_grid_exp_mult(
        np.min(x_for_interpolation),
        np.max(x_for_interpolation),
        aug_factor * aNrmCount,
        8,
    )
    policyFuncNow = MedShockPolicyFunc(
        xFuncNow,
        xLvlGrid,
        MedShkVals,
        MedPrice,
        CRRA,
        CRRAmed,
        xLvlCubicBool=CubicBool,
    )
    cFuncNow = cThruXfunc(xFuncNow, policyFuncNow.cFunc)
    MedFuncNow = MedThruXfunc(xFuncNow, policyFuncNow.cFunc, MedPrice)

    # Make the marginal value function by integrating over medical shocks
    # Make temporary grids to evaluate the consumption function
    temp_grid = np.tile(
        np.reshape(aXtraGrid, (aNrmCount, 1, 1)), (1, pLvlCount, MedCount)
    )
    aMinGrid = np.tile(
        np.reshape(mLvlMinNow(pLvlGrid), (1, pLvlCount, 1)),
        (aNrmCount, 1, MedCount),
    )
    pGrid = np.tile(np.reshape(pLvlGrid, (1, pLvlCount, 1)), (aNrmCount, 1, MedCount))
    mGrid = temp_grid * pGrid + aMinGrid
    if pLvlGrid[0] == 0:
        mGrid[:, 0, :] = np.tile(np.reshape(aXtraGrid, (aNrmCount, 1)), (1, MedCount))
    MedShkGrid = np.tile(
        np.reshape(MedShkVals, (1, 1, MedCount)), (aNrmCount, pLvlCount, 1)
    )
    probsGrid = np.tile(
        np.reshape(MedShkPrbs, (1, 1, MedCount)), (aNrmCount, pLvlCount, 1)
    )

    # Get optimal consumption (and medical care) for each state
    cGrid, MedGrid = policyFuncNow(mGrid, pGrid, MedShkGrid)

    # Calculate expected marginal value by "integrating" across medical shocks
    vPgrid = uFunc.der(cGrid)
    vPnow = np.sum(vPgrid * probsGrid, axis=2)

    # Add vPnvrs=0 at m=mLvlMin to close it off at the bottom (and vNvrs=0)
    mGrid_small = np.concatenate(
        (np.reshape(mLvlMinNow(pLvlGrid), (1, pLvlCount)), mGrid[:, :, 0])
    )
    vPnvrsNow = np.concatenate(
        (np.zeros((1, pLvlCount)), uFunc.derinv(vPnow, order=(1, 0)))
    )

    # Calculate expected value by "integrating" across medical shocks
    if vFuncBool:
        # interpolation error sometimes makes Med < 0 (barely), so fix that
        MedGrid = np.maximum(MedGrid, 1e-100)
        # interpolation error sometimes makes tiny violations, so fix that
        aGrid = np.maximum(mGrid - cGrid - MedPrice * MedGrid, aMinGrid)
        vGrid = uFunc(cGrid) + MedShkGrid * uMed(MedGrid) + EndOfPrd_vFunc(aGrid, pGrid)
        vNow = np.sum(vGrid * probsGrid, axis=2)

        # Switch to pseudo-inverse value and add a point at bottom
        vNvrsNow = np.concatenate((np.zeros((1, pLvlCount)), uFunc.inv(vNow)), axis=0)
        vNvrsPnow = vPnow * uFunc.derinv(vNow, order=(0, 1))
        vNvrsPnow = np.concatenate((np.zeros((1, pLvlCount)), vNvrsPnow), axis=0)

    # Construct the pseudo-inverse value and marginal value functions over mLvl,pLvl
    vPnvrsFunc_by_pLvl = []
    vNvrsFunc_by_pLvl = []
    # Make a pseudo inverse marginal value function for each pLvl
    for j in range(pLvlCount):
        pLvl = pLvlGrid[j]
        m_temp = mGrid_small[:, j] - mLvlMinNow(pLvl)
        vPnvrs_temp = vPnvrsNow[:, j]
        vPnvrsFunc_by_pLvl.append(LinearInterp(m_temp, vPnvrs_temp))
        if vFuncBool:
            vNvrs_temp = vNvrsNow[:, j]
            vNvrsP_temp = vNvrsPnow[:, j]
            vNvrsFunc_by_pLvl.append(CubicInterp(m_temp, vNvrs_temp, vNvrsP_temp))

    # Combine those functions across pLvls, and adjust for the lower bound of mLvl
    vPnvrsFuncBase = LinearInterpOnInterp1D(vPnvrsFunc_by_pLvl, pLvlGrid)
    vPnvrsFunc = VariableLowerBoundFunc2D(vPnvrsFuncBase, mLvlMinNow)
    if vFuncBool:
        vNvrsFuncBase = LinearInterpOnInterp1D(vNvrsFunc_by_pLvl, pLvlGrid)
        vNvrsFunc = VariableLowerBoundFunc2D(vNvrsFuncBase, mLvlMinNow)

    # "Re-curve" the (marginal) value function
    vPfuncNow = MargValueFuncCRRA(vPnvrsFunc, CRRA)
    if vFuncBool:
        vFuncNow = ValueFuncCRRA(vNvrsFunc, CRRA)
    else:
        vFuncNow = NullFunc()

    # If using cubic spline interpolation, construct the marginal marginal value function
    if CubicBool:
        vPPfuncNow = MargMargValueFuncCRRA(vPfuncNow.cFunc, CRRA)
    else:
        vPPfuncNow = NullFunc()

    # Package and return the solution object
    solution_now = ConsumerSolution(
        cFunc=cFuncNow,
        vFunc=vFuncNow,
        vPfunc=vPfuncNow,
        vPPfunc=vPPfuncNow,
        mNrmMin=0.0,  # Not a normalized model, mLvlMin will be added below
        hNrm=0.0,  # Not a normalized model, hLvl will be added below
        MPCmin=MPCminNow,
        MPCmax=0.0,  # This should be a function, need to make it
    )
    solution_now.hLvl = hLvlNow
    solution_now.mLvlMin = mLvlMinNow
    solution_now.MedFunc = MedFuncNow
    solution_now.policyFunc = policyFuncNow
    return solution_now


###############################################################################

# Make a constructor dictionary for the general income process consumer type
medshock_constructor_dict = {
    "IncShkDstn": construct_lognormal_income_process_unemployment,
    "PermShkDstn": get_PermShkDstn_from_IncShkDstn,
    "TranShkDstn": get_TranShkDstn_from_IncShkDstn,
    "aXtraGrid": make_assets_grid,
    "pLvlPctiles": make_basic_pLvlPctiles,
    "pLvlGrid": make_pLvlGrid_by_simulation,
    "pLvlNextFunc": make_AR1_style_pLvlNextFunc,
    "MedShkDstn": make_lognormal_MedShkDstn,
    "solution_terminal": make_MedShock_solution_terminal,
    "kNrmInitDstn": make_lognormal_kNrm_init_dstn,
    "pLvlInitDstn": make_lognormal_pLvl_init_dstn,
}

# Make a dictionary with parameters for the default constructor for kNrmInitDstn
default_kNrmInitDstn_params = {
    "kLogInitMean": -6.0,  # Mean of log initial capital
    "kLogInitStd": 1.0,  # Stdev of log initial capital
    "kNrmInitCount": 15,  # Number of points in initial capital discretization
}

# Make a dictionary with parameters for the default constructor for pLvlInitDstn
default_pLvlInitDstn_params = {
    "pLogInitMean": 0.0,  # Mean of log permanent income
    "pLogInitStd": 0.4,  # Stdev of log permanent income
    "pLvlInitCount": 15,  # Number of points in initial capital discretization
}

# Default parameters to make IncShkDstn using construct_lognormal_income_process_unemployment
default_IncShkDstn_params = {
    "PermShkStd": [0.1],  # Standard deviation of log permanent income shocks
    "PermShkCount": 7,  # Number of points in discrete approximation to permanent income shocks
    "TranShkStd": [0.1],  # Standard deviation of log transitory income shocks
    "TranShkCount": 7,  # Number of points in discrete approximation to transitory income shocks
    "UnempPrb": 0.05,  # Probability of unemployment while working
    "IncUnemp": 0.3,  # Unemployment benefits replacement rate while working
    "T_retire": 0,  # Period of retirement (0 --> no retirement)
    "UnempPrbRet": 0.005,  # Probability of "unemployment" while retired
    "IncUnempRet": 0.0,  # "Unemployment" benefits when retired
}

# Default parameters to make aXtraGrid using make_assets_grid
default_aXtraGrid_params = {
    "aXtraMin": 0.001,  # Minimum end-of-period "assets above minimum" value
    "aXtraMax": 30,  # Maximum end-of-period "assets above minimum" value
    "aXtraNestFac": 3,  # Exponential nesting factor for aXtraGrid
    "aXtraCount": 32,  # Number of points in the grid of "assets above minimum"
    "aXtraExtra": [0.005, 0.01],  # Additional other values to add in grid (optional)
}

# Default parameters to make pLvlGrid using make_basic_pLvlPctiles
default_pLvlPctiles_params = {
    "pLvlPctiles_count": 19,  # Number of points in the "body" of the grid
    "pLvlPctiles_bound": [0.05, 0.95],  # Percentile bounds of the "body"
    "pLvlPctiles_tail_count": 4,  # Number of points in each tail of the grid
    "pLvlPctiles_tail_order": np.e,  # Scaling factor for points in each tail
}

# Default parameters to make pLvlGrid using make_trivial_pLvlNextFunc
default_pLvlGrid_params = {
    "pLvlInitMean": 0.0,  # Mean of log initial permanent income
    "pLvlInitStd": 0.4,  # Standard deviation of log initial permanent income *MUST BE POSITIVE*
    # "pLvlPctiles": pLvlPctiles,  # Percentiles of permanent income to use for the grid
    "pLvlExtra": [
        0.0001
    ],  # Additional permanent income points to automatically add to the grid, optional
}

# Default parameters to make pLvlNextFunc using make_AR1_style_pLvlNextFunc
default_pLvlNextFunc_params = {
    "PermGroFac": [1.0],  # Permanent income growth factor
    "PrstIncCorr": 0.98,  # Correlation coefficient on (log) persistent income
}

# Default parameters to make MedShkDstn using make_lognormal_MedShkDstn
default_MedShkDstn_params = {
    "MedShkAvg": [0.1],  # Average of medical need shocks
    "MedShkStd": [4.0],  # Standard deviation of (log) medical need shocks
    "MedShkCount": 5,  # Number of medical shock points in "body"
    "MedShkCountTail": 15,  # Number of medical shock points in "tail" (upper only)
    "MedPrice": [1.5],  # Relative price of a unit of medical care
}

# Make a dictionary to specify a medical shocks consumer type
init_medical_shocks = {
    # BASIC HARK PARAMETERS REQUIRED TO SOLVE THE MODEL
    "cycles": 1,  # Finite, non-cyclic model
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    "pseudo_terminal": False,  # Terminal period really does exist
    "constructors": medshock_constructor_dict,  # See dictionary above
    # PRIMITIVE RAW PARAMETERS REQUIRED TO SOLVE THE MODEL
    "CRRA": 2.0,  # Coefficient of relative risk aversion on consumption
    "CRRAmed": 3.0,  # Coefficient of relative risk aversion on medical care
    "Rfree": [1.03],  # Interest factor on retained assets
    "DiscFac": 0.96,  # Intertemporal discount factor
    "LivPrb": [0.99],  # Survival probability after each period
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint
    "vFuncBool": False,  # Whether to calculate the value function during solution
    "CubicBool": False,  # Whether to use cubic spline interpolation when True
    # (Uses linear spline interpolation for cFunc when False)
    # PARAMETERS REQUIRED TO SIMULATE THE MODEL
    "AgentCount": 10000,  # Number of agents of this type
    "T_age": None,  # Age after which simulated agents are automatically killed
    "PermGroFacAgg": 1.0,  # Aggregate permanent income growth factor
    # (The portion of PermGroFac attributable to aggregate productivity growth)
    "NewbornTransShk": False,  # Whether Newborns have transitory shock
    # ADDITIONAL OPTIONAL PARAMETERS
    "PerfMITShk": False,  # Do Perfect Foresight MIT Shock
    # (Forces Newborns to follow solution path of the agent they replaced if True)
    "neutral_measure": False,  # Whether to use permanent income neutral measure (see Harmenberg 2021)
}
init_medical_shocks.update(default_IncShkDstn_params)
init_medical_shocks.update(default_aXtraGrid_params)
init_medical_shocks.update(default_pLvlPctiles_params)
init_medical_shocks.update(default_pLvlGrid_params)
init_medical_shocks.update(default_MedShkDstn_params)
init_medical_shocks.update(default_pLvlNextFunc_params)
init_medical_shocks.update(default_pLvlInitDstn_params)
init_medical_shocks.update(default_kNrmInitDstn_params)


class MedShockConsumerType(PersistentShockConsumerType):
    r"""
    A consumer type based on GenIncShockConsumerType, with two types of consumption goods (medical and nonmedical) and random shocks to medical utility.

    .. math::
        \begin{eqnarray*}
        V_t(M_t,P_t,\eta_t) &=& \max_{C_t, med_t} U_t(C_t, med_t; \eta_t) + \beta (1-\mathsf{D}_{t+1}) \mathbb{E} [V_{t+1}(M_{t+1}, P_{t+1}, \text{medShk}_{t+1})], \\
        A_t &=& M_t - X_t, \\
        X_t &=& C_t +med_t \textbf{ medPrice}_t,\\
        A_t/P_t &\geq& \underline{a}, \\
        P_{t+1} &=& \Gamma_{t+1}(P_t)\psi_{t+1}, \\
        Y_{t+1} &=& P_{t+1} \theta_{t+1}
        M_{t+1} &=& R A_t + Y_{t+1}, \\
        (\psi_{t+1},\theta_{t+1}) &\sim& F_{t+1},\\
        \eta_t &~\sim& G_t,\\
        U_t(C, med; \eta) &=& \frac{C^{1-\rho}}{1-\rho} +\eta \frac{med^{1-\nu}}{1-\nu}.
        \end{eqnarray*}


    Constructors
    ------------
    IncShkDstn: Constructor, :math:`\psi`, :math:`\theta`
        The agent's income shock distributions.
        Its default constructor is :func:`HARK.Calibration.Income.IncomeProcesses.construct_lognormal_income_process_unemployment`
    aXtraGrid: Constructor
        The agent's asset grid.
        Its default constructor is :func:`HARK.utilities.make_assets_grid`
    pLvlNextFunc: Constructor
        An arbitrary function used to evolve the GenIncShockConsumerType's permanent income
        Its default constructor is :func:`HARK.Calibration.Income.IncomeProcesses.make_trivial_pLvlNextFunc`
    pLvlGrid: Constructor
        The agent's pLvl grid
        Its default constructor is :func:`HARK.Calibration.Income.IncomeProcesses.make_pLvlGrid_by_simulation`
    pLvlPctiles: Constructor
        The agents income level percentile grid
        Its default constructor is :func:`HARK.Calibration.Income.IncomeProcesses.make_basic_pLvlPctiles`
    MedShkDstn: Constructor, :math:`\text{medShk}`
        The agent's Medical utility shock distribution.
        Its default constructor is :func:`HARK.ConsumptionSaving.ConsMedModel.make_lognormal_MedShkDstn`

    Solving Parameters
    ------------------
    cycles: int
        0 specifies an infinite horizon model, 1 specifies a finite model.
    T_cycle: int
        Number of periods in the cycle for this agent type.
    CRRA: float, :math:`\rho`
        Coefficient of Relative Risk Aversion for consumption.
    CRRAmed: float, :math:`\nu`
        Coefficient of Relative Risk Aversion for medical care.
    Rfree: float or list[float], time varying, :math:`\mathsf{R}`
        Risk Free interest rate. Pass a list of floats to make Rfree time varying.
    DiscFac: float, :math:`\beta`
        Intertemporal discount factor.
    LivPrb: list[float], time varying, :math:`1-\mathsf{D}`
        Survival probability after each period.
    PermGroFac: list[float], time varying, :math:`\Gamma`
        Permanent income growth factor.
    BoroCnstArt: float, :math:`\underline{a}`
        The minimum Asset/Perminant Income ratio, None to ignore.
    vFuncBool: bool
        Whether to calculate the value function during solution.
    CubicBool: bool
        Whether to use cubic spline interpoliation.

    Simulation Parameters
    ---------------------
    AgentCount: int
        Number of agents of this kind that are created during simulations.
    T_age: int
        Age after which to automatically kill agents, None to ignore.
    T_sim: int, required for simulation
        Number of periods to simulate.
    track_vars: list[strings]
        List of variables that should be tracked when running the simulation.
        For this agent, the options are 'Med', 'MedShk', 'PermShk', 'TranShk', 'aLvl', 'cLvl', 'mLvl', 'pLvl', and 'who_dies'.

        PermShk is the agent's permanent income shock

        MedShk is the agent's medical utility shock

        TranShk is the agent's transitory income shock

        aLvl is the nominal asset level

        cLvl is the nominal consumption level

        Med is the nominal medical spending level

        mLvl is the nominal market resources

        pLvl is the permanent income level

        who_dies is the array of which agents died
    aNrmInitMean: float
        Mean of Log initial Normalized Assets.
    aNrmInitStd: float
        Std of Log initial Normalized Assets.
    pLvlInitMean: float
        Mean of Log initial permanent income.
    pLvlInitStd: float
        Std of Log initial permanent income.
    PermGroFacAgg: float
        Aggregate permanent income growth factor (The portion of PermGroFac attributable to aggregate productivity growth).
    PerfMITShk: boolean
        Do Perfect Foresight MIT Shock (Forces Newborns to follow solution path of the agent they replaced if True).
    NewbornTransShk: boolean
        Whether Newborns have transitory shock.

    Attributes
    ----------
    solution: list[Consumer solution object]
        Created by the :func:`.solve` method. Finite horizon models create a list with T_cycle+1 elements, for each period in the solution.
        Infinite horizon solutions return a list with T_cycle elements for each period in the cycle.

        Unlike other models with this solution type, this model's variables are NOT normalized.
        The solution functions additionally depend on the permanent income level and the medical shock.
        For example, :math:`C=\text{cFunc}(M,P,MedShk)`.
        hNrm has been replaced by hLvl which is a function of permanent income.
        MPC max has not yet been implemented for this class. It will be a function of permanent income.

        This solution has two additional functions
        :math:`\text{Med}=\text{MedFunc}(M,P,\text{MedShk})`: returns the agent's spending on Medical care

        :math:`[C,Med]=\text{policyFunc}(M,P,\text{MedShk})`: returns the agent's spending on consumption and Medical care as numpy arrays

        Visit :class:`HARK.ConsumptionSaving.ConsIndShockModel.ConsumerSolution` for more information about the solution.
    history: Dict[Array]
        Created by running the :func:`.simulate()` method.
        Contains the variables in track_vars. Each item in the dictionary is an array with the shape (T_sim,AgentCount).
        Visit :class:`HARK.core.AgentType.simulate` for more information.
    """

    default_ = {
        "params": init_medical_shocks,
        "solver": solve_one_period_ConsMedShock,
        "model": "ConsMedShock.yaml",
    }

    time_vary_ = PersistentShockConsumerType.time_vary_ + ["MedPrice", "MedShkDstn"]
    time_inv_ = PersistentShockConsumerType.time_inv_ + ["CRRAmed"]
    shock_vars_ = PersistentShockConsumerType.shock_vars_ + ["MedShk"]
    state_vars = PersistentShockConsumerType.state_vars + ["mLvl"]
    distributions = [
        "IncShkDstn",
        "PermShkDstn",
        "TranShkDstn",
        "kNrmInitDstn",
        "pLvlInitDstn",
        "MedShkDstn",
    ]

    def pre_solve(self):
        self.construct("solution_terminal")

    def get_shocks(self):
        """
        Gets permanent and transitory income shocks for this period as well as medical need shocks
        and the price of medical care.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Get permanent and transitory income shocks
        PersistentShockConsumerType.get_shocks(self)

        # Initialize medical shock array and relative price array
        MedShkNow = np.zeros(self.AgentCount)
        MedPriceNow = np.zeros(self.AgentCount)

        # Get shocks for each period of the cycle
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            N = np.sum(these)
            if N > 0:
                MedShkNow[these] = self.MedShkDstn[t].draw(N)
                MedPriceNow[these] = self.MedPrice[t]
        self.shocks["MedShk"] = MedShkNow
        self.shocks["MedPrice"] = MedPriceNow

    def get_controls(self):
        """
        Calculates consumption and medical care for each consumer of this type
        using the consumption and medical care functions.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        cLvlNow = np.zeros(self.AgentCount) + np.nan
        MedNow = np.zeros(self.AgentCount) + np.nan
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            cLvlNow[these], MedNow[these] = self.solution[t].policyFunc(
                self.state_now["mLvl"][these],
                self.state_now["pLvl"][these],
                self.shocks["MedShk"][these],
            )
        self.controls["cLvl"] = cLvlNow
        self.controls["Med"] = MedNow
        return None

    def get_poststates(self):
        """
        Calculates end-of-period assets for each consumer of this type.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.state_now["aLvl"] = (
            self.state_now["mLvl"]
            - self.controls["cLvl"]
            - self.shocks["MedPrice"] * self.controls["Med"]
        )

        # moves now to prev
        AgentType.get_poststates(self)


###############################################################################


class ConsMedExtMargSolution(MetricObject):
    """
    Representation of the solution to one period's problem in the extensive margin
    medical expense model. If no inputs are passed, a trivial object is constructed,
    which can be used as the pseudo-terminal solution.

    Parameters
    ----------
    vFunc_by_pLvl : [function]
        List of beginning-of-period value functions over kLvl, by pLvl.
    vPfunc_by_pLvl : [function]
        List of beginning-of-period marginal functions over kLvl, by pLvl.
    cFunc_by_pLvl : [function]
        List of consumption functions over bLvl, by pLvl.
    vNvrsFuncMid_by_pLvl : [function]
        List of pseudo-inverse value function for consumption phase over bLvl, by pLvl.
    ExpMedFunc : function
        Expected medical care as a function of mLvl and pLvl, just before medical
        shock is realized.
    CareProbFunc : function
        Probability of getting medical treatment as a function of mLvl and pLvl,
        just before medical shock is realized.
    pLvl : np.array
        Grid of permanent income levels during the period (after shocks).
    CRRA : float
        Coefficient of relative risk aversion
    """

    distance_criteria = ["cFunc"]

    def __init__(
        self,
        vFunc_by_pLvl=None,
        vPfunc_by_pLvl=None,
        cFunc_by_pLvl=None,
        vNvrsFuncMid_by_pLvl=None,
        ExpMedFunc=None,
        CareProbFunc=None,
        pLvl=None,
        CRRA=None,
    ):
        self.pLvl = pLvl
        self.CRRA = CRRA
        if vFunc_by_pLvl is None:
            self.vFunc_by_pLvl = pLvl.size * [ConstantFunction(0.0)]
        else:
            self.vFunc_by_pLvl = vFunc_by_pLvl
        if vPfunc_by_pLvl is None:
            self.vPfunc_by_pLvl = pLvl.size * [ConstantFunction(0.0)]
        else:
            self.vPfunc_by_pLvl = vPfunc_by_pLvl
        if cFunc_by_pLvl is not None:
            self.cFunc = LinearInterpOnInterp1D(cFunc_by_pLvl, pLvl)
        else:
            self.cFunc = None
        if vNvrsFuncMid_by_pLvl is not None:
            vNvrsFuncMid = LinearInterpOnInterp1D(vNvrsFuncMid_by_pLvl, pLvl)
            self.vFuncMid = ValueFuncCRRA(vNvrsFuncMid, CRRA, illegal_value=-np.inf)
        if ExpMedFunc is not None:
            self.ExpMedFunc = ExpMedFunc
        if CareProbFunc is not None:
            self.CareProbFunc = CareProbFunc


def make_MedExtMarg_solution_terminal(pLogCount):
    """
    Construct a trivial pseudo-terminal solution for the extensive margin medical
    spending model: a list of constant zero functions for (marginal) value. The
    only piece of information needed for this is how many such functions to include.
    """
    pLvl_terminal = np.arange(pLogCount)
    solution_terminal = ConsMedExtMargSolution(pLvl=pLvl_terminal)
    return solution_terminal


###############################################################################


def solve_one_period_ConsMedExtMarg(
    solution_next,
    DiscFac,
    CRRA,
    BeqFac,
    BeqShift,
    Rfree,
    LivPrb,
    MedShkLogMean,
    MedShkLogStd,
    MedCostLogMean,
    MedCostLogStd,
    MedCorr,
    MedCostBot,
    MedCostTop,
    MedCostCount,
    aNrmGrid,
    pLogGrid,
    pLvlMean,
    TranShkDstn,
    pLogMrkvArray,
    mNrmGrid,
    kLvlGrid,
):
    """
    Solve one period of the "extensive margin medical care" model. Each period, the
    agent receives a persistent and transitory shock to income, and then a medical
    shock with two components: utility and cost. He makes a binary choice between
    paying the cost in medical expenses or suffering the utility loss, then makes
    his ordinary consumption-saving decision (technically made simultaneously, but
    solved as if sequential). This version has one health state and no insurance choice
    and hardcodes a liquidity constraint.

    Parameters
    ----------
    solution_next : ConsMedExtMargSolution
        Solution to the succeeding period's problem.
    DiscFac : float
        Intertemporal discount factor.
    CRRA : float
        Coefficient of relative risk aversion.
    BeqFac : float
        Scaling factor for bequest motive.
    BeqShift : float
        Shifter for bequest motive.
    Rfree : float
        Risk free return factor on saving.
    LivPrb : float
        Survival probability from this period to the next one.
    MedShkLogMean : float
        Mean of log utility shocks, assumed to be lognormally distributed.
    MedShkLogStd : float
        Stdev of log utility shocks, assumed to be lognormally distributed.
    MedCostLogMean : float
        Mean of log medical expense shocks, assumed to be lognormally distributed.
    MedCostLogStd : float
        Stdev of log medical expense shocks, assumed to be lognormally distributed.
    MedCorr : float
        Correlation coefficient betwen log utility shocks and log medical expense
        shocks, assumed to be joint normal (in logs).
    MedCostBot : float
        Lower bound of medical costs to consider, as standard deviations of log
        expenses away from the mean.
    MedCostTop : float
        Upper bound of medical costs to consider, as standard deviations of log
        expenses away from the mean.
    MedCostCount : int
        Number of points to use when discretizing MedCost
    aNrmGrid : np.array
        Exogenous grid of end-of-period assets, normalized by income level.
    pLogGrid : np.array
        Exogenous grid of *deviations from mean* log income level.
    pLvlMean : float
        Mean income level at this age, for generating actual income levels from
        pLogGrid as pLvl = pLvlMean * np.exp(pLogGrid).
    TranShkDstn : DiscreteDistribution
        Discretized transitory income shock distribution.
    pLogMrkvArray : np.array
        Markov transition array from beginning-of-period (prior) income levels
        to this period's levels. Pre-computed by (e.g.) Tauchen's method.
    mNrmGrid : np.array
        Exogenous grid of decision-time normalized market resources,
    kLvlGrid : np.array
        Beginning-of-period capital grid (spanning zero to very high wealth).

    Returns
    -------
    solution_now : ConsMedExtMargSolution
        Representation of the solution to this period's problem, including the
        beginning-of-period (marginal) value function by pLvl, the consumption
        function by pLvl, and the (pseudo-inverse) value function for the consumption
        phase (as a list by pLvl).
    """
    # Define (marginal) utility and bequest motive functions
    u = lambda x: CRRAutility(x, rho=CRRA)
    uP = lambda x: CRRAutilityP(x, rho=CRRA)
    W = lambda x: BeqFac * u(x + BeqShift)
    Wp = lambda x: BeqFac * uP(x + BeqShift)
    n = lambda x: CRRAutility_inv(x, rho=CRRA)

    # Make grids of pLvl x aLvl
    pLvl = np.exp(pLogGrid) * pLvlMean
    aLvl = np.dot(
        np.reshape(aNrmGrid, (aNrmGrid.size, 1)), np.reshape(pLvl, (1, pLvl.size))
    )
    aLvl = np.concatenate([np.zeros((1, pLvl.size)), aLvl])  # add zero entries

    # Evaluate end-of-period marginal value at each combination of pLvl x aLvl
    pLvlCount = pLvl.size
    EndOfPrd_vP = np.empty_like(aLvl)
    EndOfPrd_v = np.empty_like(aLvl)
    for j in range(pLvlCount):
        EndOfPrdvFunc_this_pLvl = solution_next.vFunc_by_pLvl[j]
        EndOfPrdvPfunc_this_pLvl = solution_next.vPfunc_by_pLvl[j]
        EndOfPrd_v[:, j] = DiscFac * LivPrb * EndOfPrdvFunc_this_pLvl(aLvl[:, j])
        EndOfPrd_vP[:, j] = DiscFac * LivPrb * EndOfPrdvPfunc_this_pLvl(aLvl[:, j])
    EndOfPrd_v += (1.0 - LivPrb) * W(aLvl)
    EndOfPrd_vP += (1.0 - LivPrb) * Wp(aLvl)

    # Calculate optimal consumption for each (aLvl,pLvl) gridpoint, roll back to bLvl
    cLvl = CRRAutilityP_inv(EndOfPrd_vP, CRRA)
    bLvl = aLvl + cLvl

    # Construct consumption functions over bLvl for each pLvl
    cFunc_by_pLvl = []
    for j in range(pLvlCount):
        cFunc_j = LinearInterp(
            np.insert(bLvl[:, j], 0, 0.0), np.insert(cLvl[:, j], 0, 0.0)
        )
        cFunc_by_pLvl.append(cFunc_j)

    # Construct pseudo-inverse value functions over bLvl for each pLvl
    v_mid = u(cLvl) + EndOfPrd_v  # value of reaching consumption phase
    vNvrsFuncMid_by_pLvl = []
    for j in range(pLvlCount):
        b_cnst = np.linspace(0.001, 0.95, 10) * bLvl[0, j]  # constrained wealth levels
        c_cnst = b_cnst
        v_cnst = u(c_cnst) + EndOfPrd_v[0, j]
        b_temp = np.concatenate([b_cnst, bLvl[:, j]])
        v_temp = np.concatenate([v_cnst, v_mid[:, j]])
        vNvrs_temp = n(v_temp)
        vNvrsFunc_j = LinearInterp(
            np.insert(b_temp, 0, 0.0), np.insert(vNvrs_temp, 0, 0.0)
        )
        vNvrsFuncMid_by_pLvl.append(vNvrsFunc_j)

    # Make a grid of (log) medical expenses (and probs), cross it with (mLvl,pLvl)
    MedCostBaseGrid = np.linspace(MedCostBot, MedCostTop, MedCostCount)
    MedCostLogGrid = MedCostLogMean + MedCostBaseGrid * MedCostLogStd
    MedCostGrid = np.exp(MedCostLogGrid)
    mLvl_base = np.dot(
        np.reshape(mNrmGrid, (mNrmGrid.size, 1)), np.reshape(pLvl, (1, pLvlCount))
    )
    mLvl = np.reshape(mLvl_base, (mNrmGrid.size, pLvlCount, 1))
    bLvl_if_care = mLvl - np.reshape(MedCostGrid, (1, 1, MedCostCount))
    bLvl_if_not = mLvl_base

    # Calculate mean (log) utility shock for each MedCost gridpoint, and conditional stdev
    MedShkLog_cond_mean = MedShkLogMean + MedCorr * MedShkLogStd * MedCostBaseGrid
    MedShkLog_cond_mean = np.reshape(MedShkLog_cond_mean, (1, MedCostCount))
    MedShkLog_cond_std = np.sqrt(MedShkLogStd**2 * (1.0 - MedCorr**2))
    MedShk_cond_mean = np.exp(MedShkLog_cond_mean + 0.5 * MedShkLog_cond_std**2)

    # Initialize (marginal) value function arrays over (mLvl,pLvl,MedCost)
    v_at_Dcsn = np.empty_like(bLvl_if_care)
    vP_at_Dcsn = np.empty_like(bLvl_if_care)
    care_prob_array = np.empty_like(bLvl_if_care)
    for j in range(pLvlCount):
        # Evaluate value function for (bLvl,pLvl_j), including MedCost=0
        v_if_care = u(vNvrsFuncMid_by_pLvl[j](bLvl_if_care[:, j, :]))
        v_if_not = np.reshape(
            u(vNvrsFuncMid_by_pLvl[j](bLvl_if_not[:, j])), (mNrmGrid.size, 1)
        )
        cant_pay = bLvl_if_care[:, j, :] <= 0.0
        v_if_care[cant_pay] = -np.inf

        # Find value difference at each gridpoint, convert to MedShk stdev; find prob of care
        v_diff = v_if_not - v_if_care
        log_v_diff = np.log(v_diff)
        crit_stdev = (log_v_diff - MedShkLog_cond_mean) / MedShkLog_cond_std
        prob_no_care = norm.cdf(crit_stdev)
        prob_get_care = 1.0 - prob_no_care
        care_prob_array[:, j, :] = prob_get_care

        # Calculate expected MedShk conditional on not getting medical care
        crit_z = crit_stdev - MedShkLog_cond_std
        MedShk_no_care_cond_mean = 0.5 * MedShk_cond_mean * erfc(crit_z) / prob_no_care

        # Compute expected (marginal) value over MedShk for each (mLvl,pLvl_j,MedCost)
        v_if_care[cant_pay] = 0.0
        v_at_Dcsn[:, j, :] = (
            prob_no_care * (v_if_not - MedShk_no_care_cond_mean)
            + prob_get_care * v_if_care
        )
        vP_if_care = uP(cFunc_by_pLvl[j](bLvl_if_care[:, j, :]))
        vP_if_not = np.reshape(
            uP(cFunc_by_pLvl[j](bLvl_if_not[:, j])), (mNrmGrid.size, 1)
        )
        vP_if_care[cant_pay] = 0.0
        MedShk_rate_of_change = (
            norm.pdf(crit_stdev) * (vP_if_care - vP_if_not) * MedShk_no_care_cond_mean
        )
        vP_at_Dcsn[:, j, :] = (
            prob_no_care * vP_if_not
            + prob_get_care * vP_if_care
            + MedShk_rate_of_change
        )

    # Compute expected (marginal) value over MedCost for each (mLvl,pLvl)
    temp_grid = np.linspace(MedCostBot, MedCostTop, MedCostCount)
    MedCost_pmv = norm.pdf(temp_grid)
    MedCost_pmv /= np.sum(MedCost_pmv)
    MedCost_probs = np.reshape(MedCost_pmv, (1, 1, MedCostCount))
    v_before_shk = np.sum(v_at_Dcsn * MedCost_probs, axis=2)
    vP_before_shk = np.sum(vP_at_Dcsn * MedCost_probs, axis=2)
    vNvrs_before_shk = n(v_before_shk)
    vPnvrs_before_shk = CRRAutilityP_inv(vP_before_shk, CRRA)

    # Compute expected medical expenses at each state space point
    ExpCare_all = care_prob_array * np.reshape(MedCostGrid, (1, 1, MedCostCount))
    ExpCare = np.sum(ExpCare_all * MedCost_probs, axis=2)
    ProbCare = np.sum(care_prob_array * MedCost_probs, axis=2)
    ExpCareFunc_by_pLvl = []
    CareProbFunc_by_pLvl = []
    for j in range(pLvlCount):
        m_temp = np.insert(mLvl_base[:, j], 0, 0.0)
        EC_temp = np.insert(ExpCare[:, j], 0, 0.0)
        prob_temp = np.insert(ProbCare[:, j], 0, 0.0)
        ExpCareFunc_by_pLvl.append(LinearInterp(m_temp, EC_temp))
        CareProbFunc_by_pLvl.append(LinearInterp(m_temp, prob_temp))
    ExpCareFunc = LinearInterpOnInterp1D(ExpCareFunc_by_pLvl, pLvl)
    ProbCareFunc = LinearInterpOnInterp1D(CareProbFunc_by_pLvl, pLvl)

    # Fixing kLvlGrid, compute expected (marginal) value over TranShk for each (kLvl,pLvl)
    v_by_kLvl_and_pLvl = np.empty((kLvlGrid.size, pLvlCount))
    vP_by_kLvl_and_pLvl = np.empty((kLvlGrid.size, pLvlCount))
    for j in range(pLvlCount):
        p = pLvl[j]

        # Make (marginal) value functions over mLvl for this pLvl
        m_temp = np.insert(mLvl_base[:, j], 0, 0.0)
        vNvrs_temp = np.insert(vNvrs_before_shk[:, j], 0, 0.0)
        vPnvrs_temp = np.insert(vPnvrs_before_shk[:, j], 0, 0.0)
        vNvrsFunc_temp = LinearInterp(m_temp, vNvrs_temp)
        vPnvrsFunc_temp = LinearInterp(m_temp, vPnvrs_temp)
        vFunc_temp = lambda x: u(vNvrsFunc_temp(x))
        vPfunc_temp = lambda x: uP(vPnvrsFunc_temp(x))

        # Compute expectation over TranShkDstn
        v = lambda TranShk, kLvl: vFunc_temp(kLvl + TranShk * p)
        vP = lambda TranShk, kLvl: vPfunc_temp(kLvl + TranShk * p)
        v_by_kLvl_and_pLvl[:, j] = expected(v, TranShkDstn, args=(kLvlGrid,))
        vP_by_kLvl_and_pLvl[:, j] = expected(vP, TranShkDstn, args=(kLvlGrid,))

    # Compute expectation over persistent shocks by using pLvlMrkvArray
    v_arvl = np.dot(v_by_kLvl_and_pLvl, pLogMrkvArray.T)
    vP_arvl = np.dot(vP_by_kLvl_and_pLvl, pLogMrkvArray.T)
    vNvrs_arvl = n(v_arvl)
    vPnvrs_arvl = CRRAutilityP_inv(vP_arvl, CRRA)

    # Construct "arrival" (marginal) value function by pLvl
    vFuncArvl_by_pLvl = []
    vPfuncArvl_by_pLvl = []
    for j in range(pLvlCount):
        vNvrsFunc_temp = LinearInterp(kLvlGrid, vNvrs_arvl[:, j])
        vPnvrsFunc_temp = LinearInterp(kLvlGrid, vPnvrs_arvl[:, j])
        vFuncArvl_by_pLvl.append(ValueFuncCRRA(vNvrsFunc_temp, CRRA))
        vPfuncArvl_by_pLvl.append(MargValueFuncCRRA(vPnvrsFunc_temp, CRRA))

    # Gather elements and return the solution object
    solution_now = ConsMedExtMargSolution(
        vFunc_by_pLvl=vFuncArvl_by_pLvl,
        vPfunc_by_pLvl=vPfuncArvl_by_pLvl,
        cFunc_by_pLvl=cFunc_by_pLvl,
        vNvrsFuncMid_by_pLvl=vNvrsFuncMid_by_pLvl,
        pLvl=pLvl,
        CRRA=CRRA,
        ExpMedFunc=ExpCareFunc,
        CareProbFunc=ProbCareFunc,
    )
    return solution_now


###############################################################################


# Define a dictionary of constructors for the extensive margin medical spending model
med_ext_marg_constructors = {
    "pLvlNextFunc": make_AR1_style_pLvlNextFunc,
    "IncomeProcessDict": make_persistent_income_process_dict,
    "pLogGrid": get_it_from("IncomeProcessDict"),
    "pLvlMean": get_it_from("IncomeProcessDict"),
    "pLogMrkvArray": get_it_from("IncomeProcessDict"),
    "IncShkDstn": construct_lognormal_income_process_unemployment,
    "PermShkDstn": get_PermShkDstn_from_IncShkDstn,
    "TranShkDstn": get_TranShkDstn_from_IncShkDstn,
    "BeqFac": get_it_from("BeqParamDict"),
    "BeqShift": get_it_from("BeqParamDict"),
    "BeqParamDict": reformat_bequest_motive,
    "aNrmGrid": make_assets_grid,
    "mNrmGrid": make_market_resources_grid,
    "kLvlGrid": make_capital_grid,
    "solution_terminal": make_MedExtMarg_solution_terminal,
    "kNrmInitDstn": make_lognormal_kNrm_init_dstn,
    "pLvlInitDstn": make_lognormal_pLvl_init_dstn,
    "MedShockDstn": make_continuous_MedShockDstn,
}

# Make a dictionary with default bequest motive parameters
default_BeqParam_dict = {
    "BeqMPC": 0.1,  # Hypothetical "MPC at death"
    "BeqInt": 1.0,  # Intercept term for hypothetical "consumption function at death"
}

# Make a dictionary with parameters for the default constructor for kNrmInitDstn
default_kNrmInitDstn_params_ExtMarg = {
    "kLogInitMean": -6.0,  # Mean of log initial capital
    "kLogInitStd": 1.0,  # Stdev of log initial capital
    "kNrmInitCount": 15,  # Number of points in initial capital discretization
}

# Default parameters to make IncomeProcessDict using make_persistent_income_process_dict;
# some of these are used by construct_lognormal_income_process_unemployment as well
default_IncomeProcess_params = {
    "PermShkStd": [0.1],  # Standard deviation of log permanent income shocks
    "PermShkCount": 7,  # Number of points in discrete approximation to permanent income shocks
    "TranShkStd": [0.1],  # Standard deviation of log transitory income shocks
    "TranShkCount": 7,  # Number of points in discrete approximation to transitory income shocks
    "UnempPrb": 0.05,  # Probability of unemployment while working
    "IncUnemp": 0.3,  # Unemployment benefits replacement rate while working
    "T_retire": 0,  # Period of retirement (0 --> no retirement)
    "UnempPrbRet": 0.005,  # Probability of "unemployment" while retired
    "IncUnempRet": 0.0,  # "Unemployment" benefits when retired
    "pLogInitMean": 0.0,  # Mean of log initial permanent income
    "pLogInitStd": 0.4,  # Standard deviation of log initial permanent income *MUST BE POSITIVE*
    "pLvlInitCount": 25,  # Number of discrete nodes in initial permanent income level dstn
    "PermGroFac": [1.0],  # Permanent income growth factor
    "PrstIncCorr": 0.98,  # Correlation coefficient on (log) persistent income
    "pLogCount": 45,  # Number of points in persistent income grid each period
    "pLogRange": 3.5,  # Upper/lower bound of persistent income, in unconditional standard deviations
}

# Default parameters to make aNrmGrid using make_assets_grid
default_aNrmGrid_params = {
    "aXtraMin": 0.001,  # Minimum end-of-period "assets above minimum" value
    "aXtraMax": 40.0,  # Maximum end-of-period "assets above minimum" value
    "aXtraNestFac": 2,  # Exponential nesting factor for aXtraGrid
    "aXtraCount": 96,  # Number of points in the grid of "assets above minimum"
    "aXtraExtra": [0.005, 0.01],  # Additional other values to add in grid (optional)
}

# Default parameters to make mLvlGrid using make_market_resources_grid
default_mNrmGrid_params = {
    "mNrmMin": 0.001,
    "mNrmMax": 40.0,
    "mNrmNestFac": 2,
    "mNrmCount": 72,
    "mNrmExtra": None,
}

# Default parameters to make kLvlGrid using make_capital_grid
default_kLvlGrid_params = {
    "kLvlMin": 0.0,
    "kLvlMax": 200,
    "kLvlCount": 250,
    "kLvlOrder": 2.0,
}

# Default "basic" parameters
med_ext_marg_basic_params = {
    "constructors": med_ext_marg_constructors,
    "cycles": 1,  # Lifecycle by default
    "T_cycle": 1,  # Number of periods in the default sequence
    "T_age": None,
    "AgentCount": 10000,  # Number of agents to simulate
    "DiscFac": 0.96,  # intertemporal discount factor
    "CRRA": 1.5,  # coefficient of relative risk aversion
    "Rfree": [1.02],  # risk free interest factor
    "LivPrb": [0.99],  # survival probability
    "MedCostBot": -3.1,  # lower bound of medical cost distribution, in stdevs
    "MedCostTop": 5.2,  # upper bound of medical cost distribution, in stdevs
    "MedCostCount": 84,  # number of nodes in medical cost discretization
    "MedShkLogMean": [-2.0],  # mean of log utility shocks
    "MedShkLogStd": [1.5],  # standard deviation of log utility shocks
    "MedCostLogMean": [-1.0],  # mean of log medical expenses
    "MedCostLogStd": [1.0],  # standard deviation of log medical expenses
    "MedCorr": [0.3],  # correlation coefficient between utility shock and expenses
    "PermGroFacAgg": 1.0,  # Aggregate productivity growth rate (leave at 1)
    "NewbornTransShk": True,  # Whether "newborns" have a transitory income shock
}

# Combine the dictionaries into a single default dictionary
init_med_ext_marg = med_ext_marg_basic_params.copy()
init_med_ext_marg.update(default_IncomeProcess_params)
init_med_ext_marg.update(default_aNrmGrid_params)
init_med_ext_marg.update(default_mNrmGrid_params)
init_med_ext_marg.update(default_kLvlGrid_params)
init_med_ext_marg.update(default_kNrmInitDstn_params_ExtMarg)
init_med_ext_marg.update(default_BeqParam_dict)


class MedExtMargConsumerType(PersistentShockConsumerType):
    r"""
    Class for representing agents in the extensive margin medical expense model.
    Such agents have labor income dynamics identical to the "general income process"
    model (permanent income is not normalized out), and also experience a medical
    shock with two components: medical cost and utility loss. They face a binary
    choice of whether to pay the cost or suffer the loss, then make a consumption-
    saving decision as normal. To simplify the computation, the joint distribution
    of medical shocks is specified as bivariate lognormal. This can be loosened to
    accommodate insurance contracts as mappings from total to out-of-pocket expenses.
    Can also be extended to include a health process.

    .. math::
        \begin{eqnarray*}
        V_t(M_t,P_t) &=& \max_{C_t, D_t} U_t(C_t) - (1-D_t) \eta_t + \beta (1-\mathsf{D}_{t+1}) \mathbb{E} [V_{t+1}(M_{t+1}, P_{t+1}], \\
        A_t &=& M_t - C_t - D_t med_t,  \\
        A_t/ &\geq& 0, \\
        D_t &\in& \{0,1\}, \\
        P_{t+1} &=& \Gamma_{t+1}(P_t)\psi_{t+1}, \\
        Y_{t+1} &=& P_{t+1} \theta_{t+1}
        M_{t+1} &=& R A_t + Y_{t+1}, \\
        (\psi_{t+1},\theta_{t+1}) &\sim& F_{t+1},\\
        (med_t,\eta_t) &~\sim& G_t,\\
        U_t(C) &=& \frac{C^{1-\rho}}{1-\rho}.
        \end{eqnarray*}
    """

    default_ = {
        "params": init_med_ext_marg,
        "solver": solve_one_period_ConsMedExtMarg,
        "model": "ConsExtMargMed.yaml",
    }

    time_vary_ = [
        "Rfree",
        "LivPrb",
        "MedShkLogMean",
        "MedShkLogStd",
        "MedCostLogMean",
        "MedCostLogStd",
        "MedCorr",
        "pLogGrid",
        "pLvlMean",
        "TranShkDstn",
        "pLogMrkvArray",
        "pLvlNextFunc",
        "IncShkDstn",
        "MedShockDstn",
    ]
    time_inv_ = [
        "DiscFac",
        "CRRA",
        "BeqFac",
        "BeqShift",
        "MedCostBot",
        "MedCostTop",
        "MedCostCount",
        "aNrmGrid",
        "mNrmGrid",
        "kLvlGrid",
    ]
    shock_vars = ["PermShk", "TranShk", "MedShk", "MedCost"]

    def get_shocks(self):
        """
        Gets permanent and transitory income shocks for this period as well as
        medical need and cost shocks.
        """
        # Get permanent and transitory income shocks
        PersistentShockConsumerType.get_shocks(self)

        # Initialize medical shock array and cost of care array
        MedShkNow = np.zeros(self.AgentCount)
        MedCostNow = np.zeros(self.AgentCount)

        # Get shocks for each period of the cycle
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            if np.any(these):
                N = np.sum(these)
                dstn_t = self.MedShockDstn[t]
                draws_t = dstn_t.draw(N)
                MedCostNow[these] = draws_t[0, :]
                MedShkNow[these] = draws_t[1, :]
        self.shocks["MedShk"] = MedShkNow
        self.shocks["MedCost"] = MedCostNow

    def get_controls(self):
        """
        Finds consumption for each agent, along with whether or not they get care.
        """
        # Initialize output
        cLvlNow = np.empty(self.AgentCount)
        CareNow = np.zeros(self.AgentCount, dtype=bool)

        # Get states and shocks
        mLvl = self.state_now["mLvl"]
        pLvl = self.state_now["pLvl"]
        MedCost = self.shocks["MedCost"]
        MedShk = self.shocks["MedShk"]

        # Find remaining resources with and without care
        bLvl_no_care = mLvl
        bLvl_with_care = mLvl - MedCost

        # Get controls for each period of the cycle
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            if np.any(these):
                vFunc_t = self.solution[t].vFuncMid
                cFunc_t = self.solution[t].cFunc

                v_no_care = vFunc_t(bLvl_no_care[these], pLvl[these]) - MedShk[these]
                v_if_care = vFunc_t(bLvl_with_care[these], pLvl[these])
                get_care = v_if_care > v_no_care

                b_temp = bLvl_no_care[these]
                b_temp[get_care] = bLvl_with_care[get_care]
                cLvlNow[these] = cFunc_t(b_temp, pLvl[these])
                CareNow[these] = get_care

        # Store the results
        self.controls["cLvl"] = cLvlNow
        self.controls["Care"] = CareNow

    def get_poststates(self):
        """
        Calculates end-of-period assets for each consumer of this type.
        """
        self.state_now["MedLvl"] = self.shocks["MedCost"] * self.controls["Care"]
        self.state_now["aLvl"] = (
            self.state_now["mLvl"] - self.controls["cLvl"] - self.state_now["MedLvl"]
        )
        # Move now to prev
        AgentType.get_poststates(self)

"""
Consumption-saving models that also include medical spending.
"""

from copy import deepcopy

import numpy as np
from scipy.optimize import brentq

from HARK import AgentType
from HARK.Calibration.Income.IncomeProcesses import (
    construct_lognormal_income_process_unemployment,
    get_PermShkDstn_from_IncShkDstn,
    get_TranShkDstn_from_IncShkDstn,
    make_AR1_style_pLvlNextFunc,
    make_pLvlGrid_by_simulation,
    make_basic_pLvlPctiles,
)
from HARK.ConsumptionSaving.ConsGenIncProcessModel import (
    PersistentShockConsumerType,
    VariableLowerBoundFunc2D,
)
from HARK.ConsumptionSaving.ConsIndShockModel import ConsumerSolution
from HARK.distribution import Lognormal, add_discrete_outcome_constant_mean, expected
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
    CRRAutility_inv,
    CRRAutility_invP,
    CRRAutilityP_inv,
    CRRAutilityPP,
    UtilityFuncCRRA,
)
from HARK.utilities import NullFunc, make_grid_exp_mult, make_assets_grid

__all__ = [
    "MedShockPolicyFunc",
    "cThruXfunc",
    "MedThruXfunc",
    "MedShockConsumerType",
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
                raise NotImplementedError()("Bicubic interpolation not yet implemented")
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
        return dcdm, dMeddm

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


def make_lognormal_MedShkDstn(
    T_cycle,
    MedShkAvg,
    MedShkStd,
    MedShkCount,
    MedShkCountTail,
    RNG,
    MedShkTailBound=[0.0, 0.9],
):
    """
    Constructs discretized lognormal distributions of medical preference shocks
    for each period in the cycle.

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
            mu=np.log(MedShkAvg_t) - 0.5 * MedShkStd_t**2, sigma=MedShkStd_t
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

# Default parameters to make MedShkDstn using make_lognormal_MedShkDstn
default_MedShkDstn_params = {
    "MedShkAvg": [0.001],  # Average of medical need shocks
    "MedShkStd": [5.0],  # Standard deviation of (log) medical need shocks
    "MedShkCount": 5,  # Number of medical shock points in "body"
    "MedShkCountTail": 15,  # Number of medical shock points in "tail" (upper only)
    "MedPrice": [1.5],  # Relative price of a unit of medical care
}

# Default parameters to make pLvlNextFunc using make_AR1_style_pLvlNextFunc
default_pLvlNextFunc_params = {
    "PermGroFac": [1.0],  # Permanent income growth factor
    "PrstIncCorr": 0.98,  # Correlation coefficient on (log) persistent income
}

# Make a dictionary to specify a medical shocks consumer type
init_medical_shocks = {
    # BASIC HARK PARAMETERS REQUIRED TO SOLVE THE MODEL
    "cycles": 1,  # Finite, non-cyclic model
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    "constructors": medshock_constructor_dict,  # See dictionary above
    # PRIMITIVE RAW PARAMETERS REQUIRED TO SOLVE THE MODEL
    "CRRA": 2.0,  # Coefficient of relative risk aversion on consumption
    "CRRAmed": 3.0,  # Coefficient of relative risk aversion on medical care
    "Rfree": 1.03,  # Interest factor on retained assets
    "DiscFac": 0.96,  # Intertemporal discount factor
    "LivPrb": [0.98],  # Survival probability after each period
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint
    "vFuncBool": False,  # Whether to calculate the value function during solution
    "CubicBool": False,  # Whether to use cubic spline interpolation when True
    # (Uses linear spline interpolation for cFunc when False)
    # PARAMETERS REQUIRED TO SIMULATE THE MODEL
    "AgentCount": 10000,  # Number of agents of this type
    "T_age": None,  # Age after which simulated agents are automatically killed
    "aNrmInitMean": 0.0,  # Mean of log initial assets
    "aNrmInitStd": 1.0,  # Standard deviation of log initial assets
    "pLvlInitMean": 0.0,  # Mean of log initial permanent income
    "pLvlInitStd": 0.0,  # Standard deviation of log initial permanent income
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


class MedShockConsumerType(PersistentShockConsumerType):
    """
    A class to represent agents who consume two goods: ordinary composite consumption
    and medical care; both goods yield CRRAutility, and the coefficients on the
    goods might be different.  Agents expect to receive shocks to permanent and
    transitory income as well as multiplicative shocks to utility from medical care.

    See init_med_shock for a dictionary of the keywords
    that should be passed to the constructor.

    Parameters
    ----------
    cycles : int
        Number of times the sequence of periods should be solved.
    """

    default_params_ = init_medical_shocks
    shock_vars_ = PersistentShockConsumerType.shock_vars_ + ["MedShk"]
    state_vars = PersistentShockConsumerType.state_vars + ["mLvl"]

    def __init__(self, **kwds):
        params = self.default_params_.copy()
        params.update(kwds)

        AgentType.__init__(self, **params)
        self.time_vary = deepcopy(self.time_vary_)
        self.time_inv = deepcopy(self.time_inv_)
        self.shock_vars = deepcopy(self.shock_vars_)

        self.solve_one_period = solve_one_period_ConsMedShock
        self.add_to_time_inv("CRRAmed")
        self.add_to_time_vary("MedPrice")
        self.update()

        self.state_now["aLvl"] = None
        self.state_prev["aLvl"] = None
        self.state_now["mLvl"] = None
        self.state_prev["mLvl"] = None

    def pre_solve(self):
        self.update_solution_terminal()

    def update(self):
        """
        Update the income process, the assets grid, the permanent income grid,
        the medical shock distribution, and the terminal solution.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.update_Rfree()
        self.update_income_process()
        self.update_assets_grid()
        self.update_pLvlNextFunc()
        self.update_pLvlGrid()
        self.update_med_shock_process()
        self.update_solution_terminal()

    def update_med_shock_process(self):
        """
        Constructs discrete distributions of medical preference shocks for each
        period in the cycle.  Distributions are saved as attribute MedShkDstn,
        which is added to time_vary.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.construct("MedShkDstn")
        self.add_to_time_vary("MedShkDstn")

    def reset_rng(self):
        """
        Reset the RNG behavior of this type.  This method is called automatically
        by initialize_sim(), ensuring that each simulation run uses the same sequence
        of random shocks; this is necessary for structural estimation to work.
        This method extends PersistentShockConsumerType.reset_rng() to also reset
        elements of MedShkDstn.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        PersistentShockConsumerType.reset_rng(self)

        # Reset MedShkDstn if it exists (it might not because reset_rng is called at init)
        if hasattr(self, "MedShkDstn"):
            for dstn in self.MedShkDstn:
                dstn.reset()

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
        PersistentShockConsumerType.get_shocks(
            self
        )  # Get permanent and transitory income shocks
        MedShkNow = np.zeros(self.AgentCount)  # Initialize medical shock array
        # Initialize relative price array
        MedPriceNow = np.zeros(self.AgentCount)
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
        Calculates consumption and medical care for each consumer of this type using the consumption
        and medical care functions.

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

        return None


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

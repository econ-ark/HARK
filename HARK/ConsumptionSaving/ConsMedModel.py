"""
Consumption-saving models that also include medical spending.
"""
import numpy as np
from scipy.optimize import brentq
from HARK import AgentType, MetricObject, make_one_period_oo_solver
from HARK.distribution import add_discrete_outcome_constant_mean, Lognormal
from HARK.utilities import (
    CRRAutilityP_inv,
    CRRAutility,
    CRRAutility_inv,
    CRRAutility_invP,
    CRRAutilityPP,
    make_grid_exp_mult,
    NullFunc,
)
from HARK.ConsumptionSaving.ConsIndShockModel import ConsumerSolution
from HARK.interpolation import (
    BilinearInterpOnInterp1D,
    TrilinearInterp,
    BilinearInterp,
    CubicInterp,
    LinearInterp,
    LowerEnvelope3D,
    UpperEnvelope,
    LinearInterpOnInterp1D,
    VariableLowerBoundFunc3D,
    ValueFuncCRRA,
    MargValueFuncCRRA,
    MargMargValueFuncCRRA,
)
from HARK.ConsumptionSaving.ConsGenIncProcessModel import (
    ConsGenIncProcessSolver,
    PersistentShockConsumerType,
    VariableLowerBoundFunc2D,
    init_persistent_shocks,
)
from copy import deepcopy

__all__ = [
    "MedShockPolicyFunc",
    "cThruXfunc",
    "MedThruXfunc",
    "MedShockConsumerType",
    "ConsMedShockSolver",
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
                    optMedZeroFunc = (
                        lambda c: (MedShk / MedPrice) ** (-1.0 / CRRAcon)
                        * ((xLvl - c) / MedPrice) ** (CRRAmed / CRRAcon)
                        - c
                    )
                    cLvl = brentq(optMedZeroFunc, 0.0, xLvl)  # Find solution to FOC
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
                dcdx[0, :] = dcdx[1, :]  # approximation; function goes crazy otherwise
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


###############################################################################

# -----------------------------------------------------------------------------
# ----- Define additional parameters for the medical shocks model -------------
# -----------------------------------------------------------------------------
CRRA = 2.0
CRRAmed = 1.5 * CRRA  # Coefficient of relative risk aversion for medical care
MedShkAvg = [0.001]  # Average of medical need shocks
MedShkStd = [5.0]  # Standard deviation of (log) medical need shocks
MedShkCount = 5  # Number of medical shock points in "body"
MedShkCountTail = 15  # Number of medical shock points in "tail" (upper only)
MedPrice = [1.5]  # Relative price of a unit of medical care

# Make a dictionary for the "medical shocks" model
init_medical_shocks = init_persistent_shocks.copy()
init_medical_shocks["CRRAmed"] = CRRAmed
init_medical_shocks["MedShkAvg"] = MedShkAvg
init_medical_shocks["MedShkStd"] = MedShkStd
init_medical_shocks["MedShkCount"] = MedShkCount
init_medical_shocks["MedShkCountTail"] = MedShkCountTail
init_medical_shocks["MedPrice"] = MedPrice
init_medical_shocks["aXtraCount"] = 32


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

    shock_vars_ = PersistentShockConsumerType.shock_vars_ + ["MedShk"]
    state_vars = PersistentShockConsumerType.state_vars + ["mLvl"]

    def __init__(self, **kwds):
        params = init_medical_shocks.copy()
        params.update(kwds)

        PersistentShockConsumerType.__init__(self, **params)
        self.solve_one_period = make_one_period_oo_solver(ConsMedShockSolver)
        self.add_to_time_inv("CRRAmed")
        self.add_to_time_vary("MedPrice")

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
        MedShkDstn = []  # empty list for medical shock distribution each period
        for t in range(self.T_cycle):
            MedShkAvgNow = self.MedShkAvg[t]  # get shock distribution parameters
            MedShkStdNow = self.MedShkStd[t]
            MedShkDstnNow = Lognormal(
                mu=np.log(MedShkAvgNow) - 0.5 * MedShkStdNow**2, sigma=MedShkStdNow
            ).approx(
                N=self.MedShkCount, tail_N=self.MedShkCountTail, tail_bound=[0, 0.9]
            )
            MedShkDstnNow = add_discrete_outcome_constant_mean(
                MedShkDstnNow, 0.0, 0.0, sort=True
            )  # add point at zero with no probability
            MedShkDstn.append(MedShkDstnNow)
        self.MedShkDstn = MedShkDstn
        self.add_to_time_vary("MedShkDstn")

    def update_solution_terminal(self):
        """
        Update the terminal period solution for this type.  Similar to other models,
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
        MedPrice = self.MedPrice[-1]
        MedShkVals = self.MedShkDstn[-1].data.flatten()
        MedShkPrbs = self.MedShkDstn[-1].prob

        # Initialize grids of medical need shocks, market resources, and optimal consumption
        MedShkGrid = MedShkVals
        xLvlMin = np.min(self.aXtraGrid) * np.min(self.pLvlGrid)
        xLvlMax = np.max(self.aXtraGrid) * np.max(self.pLvlGrid)
        xLvlGrid = make_grid_exp_mult(xLvlMin, xLvlMax, 3 * self.aXtraGrid.size, 8)
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
            self.CRRA,
            self.CRRAmed,
            xLvlCubicBool=self.CubicBool,
        )
        cFunc_terminal = cThruXfunc(xFunc_terminal, policyFunc_terminal.cFunc)
        MedFunc_terminal = MedThruXfunc(
            xFunc_terminal, policyFunc_terminal.cFunc, MedPrice
        )

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
        vPgrid = cLvlGrid ** (-self.CRRA)
        vPgrid[np.isinf(vPgrid)] = 0.0  # correct for issue at bottom edges
        PrbGrid = np.tile(
            np.reshape(MedShkPrbs, (1, MedShkGrid.size)), (mLvlGrid.size, 1)
        )
        vP_expected = np.sum(vPgrid * PrbGrid, axis=1)

        # Construct the marginal (marginal) value function for the terminal period
        vPnvrs = vP_expected ** (-1.0 / self.CRRA)
        vPnvrs[0] = 0.0
        vPnvrsFunc = BilinearInterp(
            np.tile(np.reshape(vPnvrs, (vPnvrs.size, 1)), (1, trivial_grid.size)),
            mLvlGrid,
            trivial_grid,
        )
        vPfunc_terminal = MargValueFuncCRRA(vPnvrsFunc, self.CRRA)
        vPPfunc_terminal = MargMargValueFuncCRRA(vPnvrsFunc, self.CRRA)

        # Integrate value across shocks to get expected value
        vGrid = utility(cLvlGrid, gam=self.CRRA) + MedShkGrid_tiled * utility(
            MedGrid, gam=self.CRRAmed
        )
        vGrid[:, 0] = utility(
            cLvlGrid[:, 0], gam=self.CRRA
        )  # correct for issue when MedShk=0
        vGrid[np.isinf(vGrid)] = 0.0  # correct for issue at bottom edges
        v_expected = np.sum(vGrid * PrbGrid, axis=1)

        # Construct the value function for the terminal period
        vNvrs = utility_inv(v_expected, gam=self.CRRA)
        vNvrs[0] = 0.0
        vNvrsP = vP_expected * utility_invP(
            v_expected, gam=self.CRRA
        )  # NEED TO FIGURE OUT MPC MAX IN THIS MODEL
        vNvrsP[0] = 0.0
        tempFunc = CubicInterp(mLvlGrid, vNvrs, vNvrsP)
        vNvrsFunc = LinearInterpOnInterp1D([tempFunc, tempFunc], trivial_grid)
        vFunc_terminal = ValueFuncCRRA(vNvrsFunc, self.CRRA)

        # Make the terminal period solution
        self.solution_terminal.cFunc = cFunc_terminal
        self.solution_terminal.MedFunc = MedFunc_terminal
        self.solution_terminal.policyFunc = policyFunc_terminal
        self.solution_terminal.vPfunc = vPfunc_terminal
        self.solution_terminal.vFunc = vFunc_terminal
        self.solution_terminal.vPPfunc = vPPfunc_terminal
        self.solution_terminal.hNrm = 0.0  # Don't track normalized human wealth
        self.solution_terminal.hLvl = lambda p: np.zeros_like(
            p
        )  # But do track absolute human wealth by permanent income
        self.solution_terminal.mLvlMin = lambda p: np.zeros_like(
            p
        )  # And minimum allowable market resources by perm inc

    def update_pLvlGrid(self):
        """
        Update the grid of permanent income levels.  Currently only works for
        infinite horizon models (cycles=0) and lifecycle models (cycles=1).  Not
        clear what to do about cycles>1.  Identical to version in persistent
        shocks model, but pLvl=0 is manually added to the grid (because there is
        no closed form lower-bounding cFunc for pLvl=0).

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Run basic version of this method
        PersistentShockConsumerType.update_pLvlGrid(self)
        for j in range(len(self.pLvlGrid)):  # Then add 0 to the bottom of each pLvlGrid
            this_grid = self.pLvlGrid[j]
            self.pLvlGrid[j] = np.insert(this_grid, 0, 0.0001)

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
        MedPriceNow = np.zeros(self.AgentCount)  # Initialize relative price array
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


class ConsMedShockSolver(ConsGenIncProcessSolver):
    """
    Class for solving the one period problem for the "medical shocks" model, in
    which consumers receive shocks to permanent and transitory income as well as
    shocks to "medical need"-- multiplicative utility shocks for a second good.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncShkDstn : distribution.Distribution
        A discrete
        approximations to the income process between the period being solved
        and the one immediately following (in solution_next).
    MedShkDstn : distribution.Distribution
        Discrete distribution of the multiplicative utility shifter for med-
        ical care.
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
    """

    def __init__(
        self,
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
        Constructor for a new solver for a one period problem with idiosyncratic
        shocks to permanent and transitory income and shocks to medical need.
        """
        self.solution_next = solution_next
        self.IncShkDstn = IncShkDstn
        self.MedShkDstn = MedShkDstn
        self.LivPrb = LivPrb
        self.DiscFac = DiscFac
        self.CRRA = CRRA
        self.CRRAmed = CRRAmed
        self.Rfree = Rfree
        self.MedPrice = MedPrice
        self.pLvlNextFunc = pLvlNextFunc
        self.BoroCnstArt = BoroCnstArt
        self.aXtraGrid = aXtraGrid
        self.pLvlGrid = pLvlGrid
        self.vFuncBool = vFuncBool
        self.CubicBool = CubicBool
        self.PermGroFac = 0.0
        self.def_utility_funcs()

    def set_and_update_values(self, solution_next, IncShkDstn, LivPrb, DiscFac):
        """
        Unpacks some of the inputs (and calculates simple objects based on them),
        storing the results in self for use by other methods.  These include:
        income shocks and probabilities, medical shocks and probabilities, next
        period's marginal value function (etc), the probability of getting the
        worst income shock next period, the patience factor, human wealth, and
        the bounding MPCs.

        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one period problem.
        IncShkDstn : distribution.Distribution
            A discrete
            approximation to the income process between the period being solved
            and the one immediately following (in solution_next).
        LivPrb : float
            Survival probability; likelihood of being alive at the beginning of
            the succeeding period.
        DiscFac : float
            Intertemporal discount factor for future utility.

        Returns
        -------
        None
        """
        # Run basic version of this method
        ConsGenIncProcessSolver.set_and_update_values(
            self, self.solution_next, self.IncShkDstn, self.LivPrb, self.DiscFac
        )

        # Also unpack the medical shock distribution
        self.MedShkPrbs = self.MedShkDstn.prob
        self.MedShkVals = self.MedShkDstn.data.flatten()

    def def_utility_funcs(self):
        """
        Defines CRRA utility function for this period (and its derivatives,
        and their inverses), saving them as attributes of self for other methods
        to use.  Extends version from ConsIndShock models by also defining inverse
        marginal utility function over medical care.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        ConsGenIncProcessSolver.def_utility_funcs(self)  # Do basic version
        self.uMedPinv = lambda Med: utilityP_inv(Med, gam=self.CRRAmed)
        self.uMed = lambda Med: utility(Med, gam=self.CRRAmed)
        self.uMedPP = lambda Med: utilityPP(Med, gam=self.CRRAmed)

    def def_BoroCnst(self, BoroCnstArt):
        """
        Defines the constrained portion of the consumption function as cFuncNowCnst,
        an attribute of self.  Uses the artificial and natural borrowing constraints.

        Parameters
        ----------
        BoroCnstArt : float or None
            Borrowing constraint for the minimum allowable (normalized) assets
            to end the period with.  If it is less than the natural borrowing
            constraint at a particular permanent income level, then it is irrelevant;
            BoroCnstArt=None indicates no artificial borrowing constraint.

        Returns
        -------
        None
        """
        # Find minimum allowable end-of-period assets at each permanent income level
        PermIncMinNext = self.PermShkMinNext * self.pLvlNextFunc(self.pLvlGrid)
        IncLvlMinNext = PermIncMinNext * self.TranShkMinNext
        aLvlMin = (
            self.solution_next.mLvlMin(PermIncMinNext) - IncLvlMinNext
        ) / self.Rfree

        # Make a function for the natural borrowing constraint by permanent income
        BoroCnstNat = LinearInterp(
            np.insert(self.pLvlGrid, 0, 0.0), np.insert(aLvlMin, 0, 0.0)
        )
        self.BoroCnstNat = BoroCnstNat

        # Define the minimum allowable level of market resources by permanent income
        if self.BoroCnstArt is not None:
            BoroCnstArt = LinearInterp([0.0, 1.0], [0.0, self.BoroCnstArt])
            self.mLvlMinNow = UpperEnvelope(BoroCnstNat, BoroCnstArt)
        else:
            self.mLvlMinNow = BoroCnstNat

        # Make the constrained total spending function: spend all market resources
        trivial_grid = np.array([0.0, 1.0])  # Trivial grid
        spendAllFunc = TrilinearInterp(
            np.array([[[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]]),
            trivial_grid,
            trivial_grid,
            trivial_grid,
        )
        self.xFuncNowCnst = VariableLowerBoundFunc3D(spendAllFunc, self.mLvlMinNow)

        self.mNrmMinNow = (
            0.0  # Needs to exist so as not to break when solution is created
        )
        self.MPCmaxEff = (
            0.0  # Actually might vary by p, but no use formulating as a function
        )

    def get_points_for_interpolation(self, EndOfPrdvP, aLvlNow):
        """
        Finds endogenous interpolation points (x,m) for the expenditure function.

        Parameters
        ----------
        EndOfPrdvP : np.array
            Array of end-of-period marginal values.
        aLvlNow : np.array
            Array of end-of-period asset values that yield the marginal values
            in EndOfPrdvP.

        Returns
        -------
        x_for_interpolation : np.array
            Total expenditure points for interpolation.
        m_for_interpolation : np.array
            Corresponding market resource points for interpolation.
        p_for_interpolation : np.array
            Corresponding permanent income points for interpolation.
        """
        # Get size of each state dimension
        mCount = aLvlNow.shape[1]
        pCount = aLvlNow.shape[0]
        MedCount = self.MedShkVals.size

        # Calculate endogenous gridpoints and controls
        cLvlNow = np.tile(
            np.reshape(self.uPinv(EndOfPrdvP), (1, pCount, mCount)), (MedCount, 1, 1)
        )
        MedBaseNow = np.tile(
            np.reshape(self.uMedPinv(self.MedPrice * EndOfPrdvP), (1, pCount, mCount)),
            (MedCount, 1, 1),
        )
        MedShkVals_tiled = np.tile(
            np.reshape(self.MedShkVals ** (1.0 / self.CRRAmed), (MedCount, 1, 1)),
            (1, pCount, mCount),
        )
        MedLvlNow = MedShkVals_tiled * MedBaseNow
        aLvlNow_tiled = np.tile(
            np.reshape(aLvlNow, (1, pCount, mCount)), (MedCount, 1, 1)
        )
        xLvlNow = cLvlNow + self.MedPrice * MedLvlNow
        mLvlNow = xLvlNow + aLvlNow_tiled

        # Limiting consumption is zero as m approaches the natural borrowing constraint
        x_for_interpolation = np.concatenate(
            (np.zeros((MedCount, pCount, 1)), xLvlNow), axis=-1
        )
        temp = np.tile(
            self.BoroCnstNat(np.reshape(self.pLvlGrid, (1, self.pLvlGrid.size, 1))),
            (MedCount, 1, 1),
        )
        m_for_interpolation = np.concatenate((temp, mLvlNow), axis=-1)

        # Make a 3D array of permanent income for interpolation
        p_for_interpolation = np.tile(
            np.reshape(self.pLvlGrid, (1, pCount, 1)), (MedCount, 1, mCount + 1)
        )

        # Store for use by cubic interpolator
        self.cLvlNow = cLvlNow
        self.MedLvlNow = MedLvlNow
        self.MedShkVals_tiled = np.tile(
            np.reshape(self.MedShkVals, (MedCount, 1, 1)), (1, pCount, mCount)
        )

        return x_for_interpolation, m_for_interpolation, p_for_interpolation

    def use_points_for_interpolation(self, xLvl, mLvl, pLvl, MedShk, interpolator):
        """
        Constructs a basic solution for this period, including the consumption
        function and marginal value function.

        Parameters
        ----------
        xLvl : np.array
            Total expenditure points for interpolation.
        mLvl : np.array
            Corresponding market resource points for interpolation.
        pLvl : np.array
            Corresponding permanent income level points for interpolation.
        MedShk : np.array
            Corresponding medical need shocks for interpolation.
        interpolator : function
            A function that constructs and returns a consumption function.

        Returns
        -------
        solution_now : ConsumerSolution
            The solution to this period's consumption-saving problem, with a
            consumption function, marginal value function, and minimum m.
        """
        # Construct the unconstrained total expenditure function
        xFuncNowUnc = interpolator(mLvl, pLvl, MedShk, xLvl)
        xFuncNowCnst = self.xFuncNowCnst
        xFuncNow = LowerEnvelope3D(xFuncNowUnc, xFuncNowCnst)

        # Transform the expenditure function into policy functions for consumption and medical care
        aug_factor = 2
        xLvlGrid = make_grid_exp_mult(
            np.min(xLvl), np.max(xLvl), aug_factor * self.aXtraGrid.size, 8
        )
        policyFuncNow = MedShockPolicyFunc(
            xFuncNow,
            xLvlGrid,
            self.MedShkVals,
            self.MedPrice,
            self.CRRA,
            self.CRRAmed,
            xLvlCubicBool=self.CubicBool,
        )
        cFuncNow = cThruXfunc(xFuncNow, policyFuncNow.cFunc)
        MedFuncNow = MedThruXfunc(xFuncNow, policyFuncNow.cFunc, self.MedPrice)

        # Make the marginal value function (and the value function if vFuncBool=True)
        vFuncNow, vPfuncNow = self.make_v_and_vP_funcs(policyFuncNow)

        # Pack up the solution and return it
        solution_now = ConsumerSolution(
            cFunc=cFuncNow, vFunc=vFuncNow, vPfunc=vPfuncNow, mNrmMin=self.mNrmMinNow
        )
        solution_now.MedFunc = MedFuncNow
        solution_now.policyFunc = policyFuncNow
        return solution_now

    def make_v_and_vP_funcs(self, policyFunc):
        """
        Constructs the marginal value function for this period.

        Parameters
        ----------
        policyFunc : function
            Consumption and medical care function for this period, defined over
            market resources, permanent income level, and the medical need shock.

        Returns
        -------
        vFunc : function
            Value function for this period, defined over market resources and
            permanent income.
        vPfunc : function
            Marginal value (of market resources) function for this period, defined
            over market resources and permanent income.
        """
        # Get state dimension sizes
        mCount = self.aXtraGrid.size
        pCount = self.pLvlGrid.size
        MedCount = self.MedShkVals.size

        # Make temporary grids to evaluate the consumption function
        temp_grid = np.tile(
            np.reshape(self.aXtraGrid, (mCount, 1, 1)), (1, pCount, MedCount)
        )
        aMinGrid = np.tile(
            np.reshape(self.mLvlMinNow(self.pLvlGrid), (1, pCount, 1)),
            (mCount, 1, MedCount),
        )
        pGrid = np.tile(
            np.reshape(self.pLvlGrid, (1, pCount, 1)), (mCount, 1, MedCount)
        )
        mGrid = temp_grid * pGrid + aMinGrid
        if self.pLvlGrid[0] == 0:
            mGrid[:, 0, :] = np.tile(
                np.reshape(self.aXtraGrid, (mCount, 1)), (1, MedCount)
            )
        MedShkGrid = np.tile(
            np.reshape(self.MedShkVals, (1, 1, MedCount)), (mCount, pCount, 1)
        )
        probsGrid = np.tile(
            np.reshape(self.MedShkPrbs, (1, 1, MedCount)), (mCount, pCount, 1)
        )

        # Get optimal consumption (and medical care) for each state
        cGrid, MedGrid = policyFunc(mGrid, pGrid, MedShkGrid)

        # Calculate expected value by "integrating" across medical shocks
        if self.vFuncBool:
            MedGrid = np.maximum(
                MedGrid, 1e-100
            )  # interpolation error sometimes makes Med < 0 (barely)
            aGrid = np.maximum(
                mGrid - cGrid - self.MedPrice * MedGrid, aMinGrid
            )  # interpolation error sometimes makes tiny violations
            vGrid = (
                self.u(cGrid)
                + MedShkGrid * self.uMed(MedGrid)
                + self.EndOfPrdvFunc(aGrid, pGrid)
            )
            vNow = np.sum(vGrid * probsGrid, axis=2)

        # Calculate expected marginal value by "integrating" across medical shocks
        vPgrid = self.uP(cGrid)
        vPnow = np.sum(vPgrid * probsGrid, axis=2)

        # Add vPnvrs=0 at m=mLvlMin to close it off at the bottom (and vNvrs=0)
        mGrid_small = np.concatenate(
            (np.reshape(self.mLvlMinNow(self.pLvlGrid), (1, pCount)), mGrid[:, :, 0])
        )
        vPnvrsNow = np.concatenate((np.zeros((1, pCount)), self.uPinv(vPnow)))
        if self.vFuncBool:
            vNvrsNow = np.concatenate((np.zeros((1, pCount)), self.uinv(vNow)), axis=0)
            vNvrsPnow = vPnow * self.uinvP(vNow)
            vNvrsPnow = np.concatenate((np.zeros((1, pCount)), vNvrsPnow), axis=0)

        # Construct the pseudo-inverse value and marginal value functions over mLvl,pLvl
        vPnvrsFunc_by_pLvl = []
        vNvrsFunc_by_pLvl = []
        for j in range(
            pCount
        ):  # Make a pseudo inverse marginal value function for each pLvl
            pLvl = self.pLvlGrid[j]
            m_temp = mGrid_small[:, j] - self.mLvlMinNow(pLvl)
            vPnvrs_temp = vPnvrsNow[:, j]
            vPnvrsFunc_by_pLvl.append(LinearInterp(m_temp, vPnvrs_temp))
            if self.vFuncBool:
                vNvrs_temp = vNvrsNow[:, j]
                vNvrsP_temp = vNvrsPnow[:, j]
                vNvrsFunc_by_pLvl.append(CubicInterp(m_temp, vNvrs_temp, vNvrsP_temp))
        vPnvrsFuncBase = LinearInterpOnInterp1D(vPnvrsFunc_by_pLvl, self.pLvlGrid)
        vPnvrsFunc = VariableLowerBoundFunc2D(
            vPnvrsFuncBase, self.mLvlMinNow
        )  # adjust for the lower bound of mLvl
        if self.vFuncBool:
            vNvrsFuncBase = LinearInterpOnInterp1D(vNvrsFunc_by_pLvl, self.pLvlGrid)
            vNvrsFunc = VariableLowerBoundFunc2D(
                vNvrsFuncBase, self.mLvlMinNow
            )  # adjust for the lower bound of mLvl

        # "Re-curve" the (marginal) value function
        vPfunc = MargValueFuncCRRA(vPnvrsFunc, self.CRRA)
        if self.vFuncBool:
            vFunc = ValueFuncCRRA(vNvrsFunc, self.CRRA)
        else:
            vFunc = NullFunc()

        return vFunc, vPfunc

    def make_linear_xFunc(self, mLvl, pLvl, MedShk, xLvl):
        """
        Constructs the (unconstrained) expenditure function for this period using
        bilinear interpolation (over permanent income and the medical shock) among
        an array of linear interpolations over market resources.

        Parameters
        ----------
        mLvl : np.array
            Corresponding market resource points for interpolation.
        pLvl : np.array
            Corresponding permanent income level points for interpolation.
        MedShk : np.array
            Corresponding medical need shocks for interpolation.
        xLvl : np.array
            Expenditure points for interpolation, corresponding to those in mLvl,
            pLvl, and MedShk.

        Returns
        -------
        xFuncUnc : BilinearInterpOnInterp1D
            Unconstrained total expenditure function for this period.
        """
        # Get state dimensions
        pCount = mLvl.shape[1]
        MedCount = mLvl.shape[0]

        # Loop over each permanent income level and medical shock and make a linear xFunc
        xFunc_by_pLvl_and_MedShk = []  # Initialize the empty list of lists of 1D xFuncs
        for i in range(pCount):
            temp_list = []
            pLvl_i = pLvl[0, i, 0]
            mLvlMin_i = self.BoroCnstNat(pLvl_i)
            for j in range(MedCount):
                m_temp = mLvl[j, i, :] - mLvlMin_i
                x_temp = xLvl[j, i, :]
                temp_list.append(LinearInterp(m_temp, x_temp))
            xFunc_by_pLvl_and_MedShk.append(deepcopy(temp_list))

        # Combine the nested list of linear xFuncs into a single function
        pLvl_temp = pLvl[0, :, 0]
        MedShk_temp = MedShk[:, 0, 0]
        xFuncUncBase = BilinearInterpOnInterp1D(
            xFunc_by_pLvl_and_MedShk, pLvl_temp, MedShk_temp
        )
        xFuncUnc = VariableLowerBoundFunc3D(xFuncUncBase, self.BoroCnstNat)
        return xFuncUnc

    def make_cubic_xFunc(self, mLvl, pLvl, MedShk, xLvl):
        """
        Constructs the (unconstrained) expenditure function for this period using
        bilinear interpolation (over permanent income and the medical shock) among
        an array of cubic interpolations over market resources.

        Parameters
        ----------
        mLvl : np.array
            Corresponding market resource points for interpolation.
        pLvl : np.array
            Corresponding permanent income level points for interpolation.
        MedShk : np.array
            Corresponding medical need shocks for interpolation.
        xLvl : np.array
            Expenditure points for interpolation, corresponding to those in mLvl,
            pLvl, and MedShk.

        Returns
        -------
        xFuncUnc : BilinearInterpOnInterp1D
            Unconstrained total expenditure function for this period.
        """
        # Get state dimensions
        pCount = mLvl.shape[1]
        MedCount = mLvl.shape[0]

        # Calculate the MPC and MPM at each gridpoint
        EndOfPrdvPP = (
            self.DiscFacEff
            * self.Rfree
            * self.Rfree
            * np.sum(
                self.vPPfuncNext(self.mLvlNext, self.pLvlNext) * self.ShkPrbs_temp,
                axis=0,
            )
        )
        EndOfPrdvPP = np.tile(
            np.reshape(EndOfPrdvPP, (1, pCount, EndOfPrdvPP.shape[1])), (MedCount, 1, 1)
        )
        dcda = EndOfPrdvPP / self.uPP(np.array(self.cLvlNow))
        dMedda = EndOfPrdvPP / (self.MedShkVals_tiled * self.uMedPP(self.MedLvlNow))
        dMedda[0, :, :] = 0.0  # dMedda goes crazy when MedShk=0
        MPC = dcda / (1.0 + dcda + self.MedPrice * dMedda)
        MPM = dMedda / (1.0 + dcda + self.MedPrice * dMedda)

        # Convert to marginal propensity to spend
        MPX = MPC + self.MedPrice * MPM
        MPX = np.concatenate(
            (np.reshape(MPX[:, :, 0], (MedCount, pCount, 1)), MPX), axis=2
        )  # NEED TO CALCULATE MPM AT NATURAL BORROWING CONSTRAINT
        MPX[0, :, 0] = self.MPCmaxNow

        # Loop over each permanent income level and medical shock and make a cubic xFunc
        xFunc_by_pLvl_and_MedShk = []  # Initialize the empty list of lists of 1D xFuncs
        for i in range(pCount):
            temp_list = []
            pLvl_i = pLvl[0, i, 0]
            mLvlMin_i = self.BoroCnstNat(pLvl_i)
            for j in range(MedCount):
                m_temp = mLvl[j, i, :] - mLvlMin_i
                x_temp = xLvl[j, i, :]
                MPX_temp = MPX[j, i, :]
                temp_list.append(CubicInterp(m_temp, x_temp, MPX_temp))
            xFunc_by_pLvl_and_MedShk.append(deepcopy(temp_list))

        # Combine the nested list of cubic xFuncs into a single function
        pLvl_temp = pLvl[0, :, 0]
        MedShk_temp = MedShk[:, 0, 0]
        xFuncUncBase = BilinearInterpOnInterp1D(
            xFunc_by_pLvl_and_MedShk, pLvl_temp, MedShk_temp
        )
        xFuncUnc = VariableLowerBoundFunc3D(xFuncUncBase, self.BoroCnstNat)
        return xFuncUnc

    def make_basic_solution(self, EndOfPrdvP, aLvl, interpolator):
        """
        Given end of period assets and end of period marginal value, construct
        the basic solution for this period.

        Parameters
        ----------
        EndOfPrdvP : np.array
            Array of end-of-period marginal values.
        aLvl : np.array
            Array of end-of-period asset values that yield the marginal values
            in EndOfPrdvP.
        interpolator : function
            A function that constructs and returns a consumption function.

        Returns
        -------
        solution_now : ConsumerSolution
            The solution to this period's consumption-saving problem, with a
            consumption function, marginal value function, and minimum m.
        """
        xLvl, mLvl, pLvl = self.get_points_for_interpolation(EndOfPrdvP, aLvl)
        MedShk_temp = np.tile(
            np.reshape(self.MedShkVals, (self.MedShkVals.size, 1, 1)),
            (1, mLvl.shape[1], mLvl.shape[2]),
        )
        solution_now = self.use_points_for_interpolation(
            xLvl, mLvl, pLvl, MedShk_temp, interpolator
        )
        return solution_now

    def add_vPPfunc(self, solution):
        """
        Adds the marginal marginal value function to an existing solution, so
        that the next solver can evaluate vPP and thus use cubic interpolation.

        Parameters
        ----------
        solution : ConsumerSolution
            The solution to this single period problem, which must include the
            consumption function.

        Returns
        -------
        solution : ConsumerSolution
            The same solution passed as input, but with the marginal marginal
            value function for this period added as the attribute vPPfunc.
        """
        vPPfuncNow = MargMargValueFuncCRRA(solution.vPfunc.cFunc, self.CRRA)
        solution.vPPfunc = vPPfuncNow
        return solution

    def solve(self):
        """
        Solves a one period consumption saving problem with risky income and
        shocks to medical need.

        Parameters
        ----------
        None

        Returns
        -------
        solution : ConsumerSolution
            The solution to the one period problem, including a consumption
            function, medical spending function ( both defined over market re-
            sources, permanent income, and medical shock), a marginal value func-
            tion (defined over market resources and permanent income), and human
            wealth as a function of permanent income.
        """
        aLvl, trash = self.prepare_to_calc_EndOfPrdvP()
        EndOfPrdvP = self.calc_EndOfPrdvP()
        if self.vFuncBool:
            self.make_EndOfPrdvFunc(EndOfPrdvP)
        if self.CubicBool:
            interpolator = self.make_cubic_xFunc
        else:
            interpolator = self.make_linear_xFunc
        solution = self.make_basic_solution(EndOfPrdvP, aLvl, interpolator)
        solution = self.add_MPC_and_human_wealth(solution)
        if self.CubicBool:
            solution = self.add_vPPfunc(solution)
        return solution

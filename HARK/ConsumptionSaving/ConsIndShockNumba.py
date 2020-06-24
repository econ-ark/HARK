"""
Classes to solve canonical consumption-savings models with idiosyncratic shocks
to income.  All models here assume CRRA utility with geometric discounting, no
bequest motive, and income shocks are fully transitory or fully permanent.

It currently solves three types of models:
   1) A very basic "perfect foresight" consumption-savings model with no uncertainty.
   2) A consumption-savings model with risk over transitory and permanent income shocks.
   3) The model described in (2), with an interest rate for debt that differs
      from the interest rate for savings.

See NARK for information on variable naming conventions.
See HARK documentation for mathematical descriptions of the models being solved.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from interpolation import interp
from numba import njit
from quantecon.optimize import newton_secant

from HARK import makeOnePeriodOOSolver
from HARK.ConsumptionSaving.ConsIndShockModel import (
    PerfForesightConsumerType,
    ConsPerfForesightSolver,
    IndShockConsumerType,
    ConsIndShockSolverBasic,
    ConsIndShockSetup,
)
from HARK.interpolation import LinearInterp

__all__ = [
    "ConsPerfForesightSolverNumba",
    "PerfForesightConsumerTypeNumba",
    "ConsIndShockSolverBasicNumba",
    "ConsIndShockSolverNumba",
    "ConsIndShockSetupNumba",
]


@njit
def makePFcFuncNumba(
    model_BoroCnstArt,
    PermGroFac,
    Rfree,
    sn_hNrm,
    DiscFacEff,
    CRRA,
    sn_MPCmin,
    sn_cFunc_x_list,
    sn_cFunc_y_list,
    MaxKinks,
):
    # Use a local value of BoroCnstArt to prevent comparing None and float below.
    if model_BoroCnstArt is None:
        BoroCnstArt = -np.inf
    else:
        BoroCnstArt = model_BoroCnstArt

    # Calculate human wealth this period
    hNrmNow = (PermGroFac / Rfree) * (sn_hNrm + 1.0)  # return

    # Calculate the lower bound of the marginal propensity to consume
    PatFac = ((Rfree * DiscFacEff) ** (1.0 / CRRA)) / Rfree
    MPCmin = 1.0 / (1.0 + PatFac / sn_MPCmin)  # return

    # Extract the discrete kink points in next period's consumption function;
    # don't take the last one, as it only defines the extrapolation and is not a kink.
    mNrmNext = sn_cFunc_x_list[:-1]
    cNrmNext = sn_cFunc_y_list[:-1]

    # Calculate the end-of-period asset values that would reach those kink points
    # next period, then invert the first order condition to get consumption. Then
    # find the endogenous gridpoint (kink point) today that corresponds to each kink
    aNrmNow = (PermGroFac / Rfree) * (mNrmNext - 1.0)
    cNrmNow = (DiscFacEff * Rfree) ** (-1.0 / CRRA) * (PermGroFac * cNrmNext)
    mNrmNow = aNrmNow + cNrmNow

    # Add an additional point to the list of gridpoints for the extrapolation,
    # using the new value of the lower bound of the MPC.
    mNrmNow = np.append(mNrmNow, mNrmNow[-1] + 1.0)
    cNrmNow = np.append(cNrmNow, cNrmNow[-1] + MPCmin)

    # If the artificial borrowing constraint binds, combine the constrained and
    # unconstrained consumption functions.
    if BoroCnstArt > mNrmNow[0]:
        # Find the highest index where constraint binds
        cNrmCnst = mNrmNow - BoroCnstArt
        CnstBinds = cNrmCnst < cNrmNow
        idx = np.where(CnstBinds)[0][-1]

        if idx < (mNrmNow.size - 1):
            # If it is not the *very last* index, find the the critical level
            # of mNrm where the artificial borrowing contraint begins to bind.
            d0 = cNrmNow[idx] - cNrmCnst[idx]
            d1 = cNrmCnst[idx + 1] - cNrmNow[idx + 1]
            m0 = mNrmNow[idx]
            m1 = mNrmNow[idx + 1]
            alpha = d0 / (d0 + d1)
            mCrit = m0 + alpha * (m1 - m0)

            # Adjust the grids of mNrm and cNrm to account for the borrowing constraint.
            cCrit = mCrit - BoroCnstArt
            mNrmNow = np.concatenate(
                (np.array([BoroCnstArt, mCrit]), mNrmNow[(idx + 1) :])
            )
            cNrmNow = np.concatenate((np.array([0.0, cCrit]), cNrmNow[(idx + 1) :]))

        else:
            # If it *is* the very last index, then there are only three points
            # that characterize the consumption function: the artificial borrowing
            # constraint, the constraint kink, and the extrapolation point.
            mXtra = cNrmNow[-1] - cNrmCnst[-1] / (1.0 - MPCmin)
            mCrit = mNrmNow[-1] + mXtra
            cCrit = mCrit - BoroCnstArt
            mNrmNow = np.array([BoroCnstArt, mCrit, mCrit + 1.0])
            cNrmNow = np.array([0.0, cCrit, cCrit + MPCmin])

    # If the mNrm and cNrm grids have become too large, throw out the last
    # kink point, being sure to adjust the extrapolation.
    if mNrmNow.size > MaxKinks:
        mNrmNow = np.concatenate((mNrmNow[:-2], np.array([mNrmNow[-3] + 1.0])))
        cNrmNow = np.concatenate((cNrmNow[:-2], np.array([cNrmNow[-3] + MPCmin])))

    # Construct the consumption function as a linear interpolation.
    # s_.cFunc = LinearInterp(mNrmNow, cNrmNow)

    # Calculate the upper bound of the MPC as the slope of the bottom segment.
    MPCmax = (cNrmNow[1] - cNrmNow[0]) / (mNrmNow[1] - mNrmNow[0])  # return

    # Add two attributes to enable calculation of steady state market resources.
    ExIncNext = 1.0  # Perfect foresight income of 1
    mNrmMinNow = mNrmNow[0]  # Relabeling for compatibility with addSSmNrm

    return (
        hNrmNow,
        MPCmin,
        MPCmax,
        mNrmNow,
        cNrmNow,
        ExIncNext,
        mNrmMinNow,
    )


@njit
def searchSSfuncNumba(m, PermGroFac, Rfree, ExIncNext, mNrmNow, cNrmNow):
    # Make a linear function of all combinations of c and m that yield mNext = mNow
    mZeroChange = (1.0 - PermGroFac / Rfree) * m + (PermGroFac / Rfree) * ExIncNext
    # Find the steady state level of market resources
    res = interp(mNrmNow, cNrmNow, m) - mZeroChange
    # A zero of this is SS market resources
    return res


@njit
def addSSmNrmNumba(
    PermGroFac, Rfree, ExIncNext, mNrmMinNow, mNrmNow, cNrmNow,
):
    m_init_guess = mNrmMinNow + ExIncNext
    # Minimum market resources plus next income is okay starting guess
    try:
        mNrmSS = newton_secant(
            searchSSfuncNumba,
            m_init_guess,
            args=(PermGroFac, Rfree, ExIncNext, mNrmNow, cNrmNow),
        )[0]
    except:
        mNrmSS = None

    return mNrmSS


@njit
def makeCubiccFuncNumba(DiscFacEff, Rfree, PermGroFac, CRRA, PermShkVals_temp):
    EndOfPrdvPP = (
        DiscFacEff
        * Rfree
        * Rfree
        * PermGroFac ** (-CRRA - 1.0)
        * np.sum(
            PermShkVals_temp ** (-CRRA - 1.0)
            * s_vPPfuncNext(s_mNrmNext)
            * s_ShkPrbs_temp,
            axis=0,
        )
    )
    dcda = EndOfPrdvPP / s_uPP(np.array(cNrm[1:]))
    MPC = dcda / (dcda + 1.0)
    MPC = np.insert(MPC, 0, s_MPCmaxNow)

    return EndOfPrdvP, MPC


@njit
def prepareToCalcEndOfPrdvPNumba(
    aXtraGrid,
    BoroCnstNat,
    PermShkValsNext,
    TranShkValsNext,
    ShkPrbsNext,
    PermGroFac,
    Rfree,
):
    def tile(A, reps):
        return A.repeat(reps[0]).reshape(A.size, -1).transpose()

    # We define aNrmNow all the way from BoroCnstNat up to max(s_aXtraGrid)
    # even if BoroCnstNat < BoroCnstArt, so we can construct the consumption
    # function as the lower envelope of the (by the artificial borrowing con-
    # straint) uconstrained consumption function, and the artificially con-
    # strained consumption function.
    aNrmNow = np.asarray(aXtraGrid) + BoroCnstNat
    ShkCount = TranShkValsNext.size
    aNrm_temp = tile(aNrmNow, (ShkCount, 1))

    # Tile arrays of the income shocks and put them into useful shapes
    aNrmCount = aNrmNow.shape[0]
    PermShkVals_temp = (tile(PermShkValsNext, (aNrmCount, 1))).transpose()
    TranShkVals_temp = (tile(TranShkValsNext, (aNrmCount, 1))).transpose()
    ShkPrbs_temp = (tile(ShkPrbsNext, (aNrmCount, 1))).transpose()

    # Get cash on hand next period
    mNrmNext = Rfree / (PermGroFac * PermShkVals_temp) * aNrm_temp + TranShkVals_temp
    # CDC 20191205: This should be divided by LivPrb[0] for Blanchard insurance

    # Store and report the results
    return PermShkVals_temp, ShkPrbs_temp, mNrmNext, aNrmNow


@njit
def setAndUpdateValuesNumba(
    DiscFac,
    LivPrb,
    IncomeDstn_pmf,
    IncomeDstn_X,
    Rfree,
    CRRA,
    sn_MPCmin,
    sn_hNrm,
    sn_MPCmax,
    PermGroFac,
):
    DiscFacEff = DiscFac * LivPrb  # "effective" discount factor #return
    ShkPrbsNext = IncomeDstn_pmf  # return
    PermShkValsNext = IncomeDstn_X[0]  # return
    TranShkValsNext = IncomeDstn_X[1]  # return
    PermShkMinNext = np.min(PermShkValsNext)  # return
    TranShkMinNext = np.min(TranShkValsNext)  # return

    WorstIncPrb = np.sum(
        ShkPrbsNext[
            (PermShkValsNext * TranShkValsNext) == (PermShkMinNext * TranShkMinNext)
        ]
    )  # return

    # Update the bounding MPCs and PDV of human wealth:
    PatFac = ((Rfree * DiscFacEff) ** (1.0 / CRRA)) / Rfree  # return
    MPCminNow = 1.0 / (1.0 + PatFac / sn_MPCmin)  # return
    ExIncNext = np.dot(ShkPrbsNext, TranShkValsNext * PermShkValsNext)  # return
    hNrmNow = PermGroFac / Rfree * (ExIncNext + sn_hNrm)  # return
    MPCmaxNow = 1.0 / (
        1.0 + (WorstIncPrb ** (1.0 / CRRA)) * PatFac / sn_MPCmax
    )  # return

    cFuncLimitIntercept = MPCminNow * hNrmNow  # return
    cFuncLimitSlope = MPCminNow  # return

    return (
        DiscFacEff,
        ShkPrbsNext,
        PermShkValsNext,
        TranShkValsNext,
        PermShkMinNext,
        TranShkMinNext,
        WorstIncPrb,
        PatFac,
        MPCminNow,
        ExIncNext,
        hNrmNow,
        MPCmaxNow,
        cFuncLimitIntercept,
        cFuncLimitSlope,
    )


@njit
def defBoroCnstNumba(
    BoroCnstArt,
    sn_mNrmMin,
    TranShkMinNext,
    PermGroFac,
    PermShkMinNext,
    Rfree,
    MPCmaxNow,
):
    # Calculate the minimum allowable value of money resources in this period
    BoroCnstNat = (
        (sn_mNrmMin - TranShkMinNext) * (PermGroFac * PermShkMinNext) / Rfree
    )  # return

    # Note: need to be sure to handle BoroCnstArt==None appropriately.
    # In Py2, this would evaluate to 5.0:  np.max([None, 5.0]).
    # However in Py3, this raises a TypeError. Thus here we need to directly
    # address the situation in which BoroCnstArt == None:
    if BoroCnstArt is None:
        mNrmMinNow = BoroCnstNat  # return
    else:
        mNrmMinNow = np.max(np.array([BoroCnstNat, BoroCnstArt]))  # return
    if BoroCnstNat < mNrmMinNow:
        MPCmaxEff = 1.0  # If actually constrained, MPC near limit is 1 #return
    else:
        MPCmaxEff = MPCmaxNow

    return BoroCnstNat, mNrmMinNow, MPCmaxEff


class ConsPerfForesightSolverNumba(ConsPerfForesightSolver):
    """
    A class for solving a one period perfect foresight consumption-saving problem.
    An instance of this class is created by the function solvePerfForesight in each period.
    """

    def makePFcFunc(self):
        """
        Makes the (linear) consumption function for this period.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        (
            self.hNrmNow,
            self.MPCmin,
            self.MPCmax,
            self.mNrmNow,
            self.cNrmNow,
            self.ExIncNext,
            self.mNrmMinNow,
        ) = makePFcFuncNumba(
            self.BoroCnstArt,
            self.PermGroFac,
            self.Rfree,
            self.solution_next.hNrm,
            self.DiscFacEff,
            self.CRRA,
            self.solution_next.MPCmin,
            self.solution_next.cFunc.x_list,
            self.solution_next.cFunc.y_list,
            self.MaxKinks,
        )

        self.cFunc = LinearInterp(self.mNrmNow, self.cNrmNow)

    def addSSmNrm(self, solution):
        """
        Finds steady state (normalized) market resources and adds it to the
        solution.  This is the level of market resources such that the expectation
        of market resources in the next period is unchanged.  This value doesn't
        necessarily exist.

        Parameters
        ----------
        solution : ConsumerSolution
            Solution to this period's problem, which must have attribute cFunc.
        Returns
        -------
        solution : ConsumerSolution
            Same solution that was passed, but now with the attribute mNrmSS.
        """

        # Add mNrmSS to the solution and return it
        solution.mNrmSS = addSSmNrmNumba(
            self.PermGroFac,
            self.Rfree,
            self.ExIncNext,
            self.mNrmMinNow,
            self.mNrmNow,
            self.cNrmNow,
        )

        return solution


class ConsIndShockSetupNumba(ConsIndShockSetup, ConsPerfForesightSolverNumba):
    """
    A superclass for solvers of one period consumption-saving problems with
    constant relative risk aversion utility and permanent and transitory shocks
    to income.  Has methods to set up but not solve the one period problem.
    """

    def setAndUpdateValues(self, solution_next, IncomeDstn, LivPrb, DiscFac):
        """
        Unpacks some of the inputs (and calculates simple objects based on them),
        storing the results in self for use by other methods.  These include:
        income shocks and probabilities, next period's marginal value function
        (etc), the probability of getting the worst income shock next period,
        the patience factor, human wealth, and the bounding MPCs.

        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one period problem.
        IncomeDstn : distribution.DiscreteDistribution
            A DiscreteDistribution with a pmf
            and two point value arrays in X, order:
            permanent shocks, transitory shocks.
        LivPrb : float
            Survival probability; likelihood of being alive at the beginning of
            the succeeding period.
        DiscFac : float
            Intertemporal discount factor for future utility.

        Returns
        -------
        None
        """

        (
            self.DiscFacEff,
            self.ShkPrbsNext,
            self.PermShkValsNext,
            self.TranShkValsNext,
            self.PermShkMinNext,
            self.TranShkMinNext,
            self.WorstIncPrb,
            self.PatFac,
            self.MPCminNow,
            self.ExIncNext,
            self.hNrmNow,
            self.MPCmaxNow,
            self.cFuncLimitIntercept,
            self.cFuncLimitSlope,
        ) = setAndUpdateValuesNumba(
            DiscFac,
            LivPrb,
            IncomeDstn.pmf,
            np.array(IncomeDstn.X),
            self.Rfree,
            self.CRRA,
            solution_next.MPCmin,
            solution_next.hNrm,
            solution_next.MPCmax,
            self.PermGroFac,
        )

        self.vPfuncNext = solution_next.vPfunc

        if self.CubicBool:
            self.vPPfuncNext = solution_next.vPPfunc

        if self.vFuncBool:
            self.vFuncNext = solution_next.vFunc

    def defBoroCnst(self, BoroCnstArt):
        """
            Defines the constrained portion of the consumption function as cFuncNowCnst,
            an attribute of self.  Uses the artificial and natural borrowing constraints.

            Parameters
            ----------
            BoroCnstArt : float or None
                Borrowing constraint for the minimum allowable assets to end the
                period with.  If it is less than the natural borrowing constraint,
                then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
                rowing constraint.

            Returns
            -------
            none
            """

        self.BoroCnstNat, self.mNrmMinNow, self.MPCmaxEff = defBoroCnstNumba(
            BoroCnstArt,
            self.solution_next.mNrmMin,
            self.TranShkMinNext,
            self.PermGroFac,
            self.PermShkMinNext,
            self.Rfree,
            self.MPCmaxNow,
        )

        # Define the borrowing constraint (limiting consumption function)
        self.cFuncNowCnst = LinearInterp(
            np.array([self.mNrmMinNow, self.mNrmMinNow + 1]), np.array([0.0, 1.0])
        )


class ConsIndShockSolverBasicNumba(ConsIndShockSetupNumba, ConsIndShockSolverBasic):
    """
    This class solves a single period of a standard consumption-saving problem,
    using linear interpolation and without the ability to calculate the value
    function.  ConsIndShockSolver inherits from this class and adds the ability
    to perform cubic interpolation and to calculate the value function.

    Note that this class does not have its own initializing method.  It initial-
    izes the same problem in the same way as ConsIndShockSetup, from which it
    inherits.
    """

    def prepareToCalcEndOfPrdvP(self):
        """
        Prepare to calculate end-of-period marginal value by creating an array
        of market resources that the agent could have next period, considering
        the grid of end-of-period assets and the distribution of shocks he might
        experience next period.

        Parameters
        ----------
        none

        Returns
        -------
        aNrmNow : np.array
            A 1D array of end-of-period assets; also stored as attribute of self.
        """

        (
            self.PermShkVals_temp,
            self.ShkPrbs_temp,
            self.mNrmNext,
            self.aNrmNow,
        ) = prepareToCalcEndOfPrdvPNumba(
            self.aXtraGrid,
            self.BoroCnstNat,
            self.PermShkValsNext,
            self.TranShkValsNext,
            self.ShkPrbsNext,
            self.PermGroFac,
            self.Rfree,
        )

        return self.aNrmNow


class PerfForesightConsumerTypeNumba(PerfForesightConsumerType):
    def __init__(self, **kwargs):
        PerfForesightConsumerType.__init__(self, **kwargs)

        self.solveOnePeriod = makeOnePeriodOOSolver(ConsPerfForesightSolverNumba)


class IndShockConsumerTypeNumba(IndShockConsumerType):
    def __init__(self, **kwargs):
        IndShockConsumerType.__init__(self, **kwargs)

        # Add consumer-type specific objects, copying to create independent versions
        if (not self.CubicBool) and (not self.vFuncBool):
            solver = ConsIndShockSolverBasicNumba
        # else:  # Use the "advanced" solver if either is requested
        #     solver = ConsIndShockSolverNumba

        self.solveOnePeriod = makeOnePeriodOOSolver(solver)

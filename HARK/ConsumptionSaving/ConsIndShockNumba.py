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
)
from HARK.interpolation import LinearInterp
from HARK.numba import (
    CRRAutility,
    CRRAutilityP,
    CRRAutilityPP,
    CRRAutilityP_inv,
    CRRAutility_invP,
    CRRAutility_inv,
    CRRAutilityP_invP,
)

utility = CRRAutility
utilityP = CRRAutilityP
utilityPP = CRRAutilityPP
utilityP_inv = CRRAutilityP_inv
utility_invP = CRRAutility_invP
utility_inv = CRRAutility_inv
utilityP_invP = CRRAutilityP_invP


@njit
def makePFcFuncNumba(
    s_BoroCnstArt,
    s_PermGroFac,
    s_Rfree,
    s_solution_next_hNrm,
    s_DiscFacEff,
    s_CRRA,
    s_solution_next_MPCmin,
    s_solution_next_cFunc_x_list,
    s_solution_next_cFunc_y_list,
    s_MaxKinks,
):
    # Use a local value of BoroCnstArt to prevent comparing None and float below.
    if s_BoroCnstArt is None:
        BoroCnstArt = -np.inf
    else:
        BoroCnstArt = s_BoroCnstArt

    # Calculate human wealth this period
    s_hNrmNow = (s_PermGroFac / s_Rfree) * (s_solution_next_hNrm + 1.0)  # return

    # Calculate the lower bound of the marginal propensity to consume
    PatFac = ((s_Rfree * s_DiscFacEff) ** (1.0 / s_CRRA)) / s_Rfree
    s_MPCmin = 1.0 / (1.0 + PatFac / s_solution_next_MPCmin)  # return

    # Extract the discrete kink points in next period's consumption function;
    # don't take the last one, as it only defines the extrapolation and is not a kink.
    mNrmNext = s_solution_next_cFunc_x_list[:-1]
    cNrmNext = s_solution_next_cFunc_y_list[:-1]

    # Calculate the end-of-period asset values that would reach those kink points
    # next period, then invert the first order condition to get consumption. Then
    # find the endogenous gridpoint (kink point) today that corresponds to each kink
    aNrmNow = (s_PermGroFac / s_Rfree) * (mNrmNext - 1.0)
    cNrmNow = (s_DiscFacEff * s_Rfree) ** (-1.0 / s_CRRA) * (s_PermGroFac * cNrmNext)
    mNrmNow = aNrmNow + cNrmNow

    # Add an additional point to the list of gridpoints for the extrapolation,
    # using the new value of the lower bound of the MPC.
    mNrmNow = np.append(mNrmNow, mNrmNow[-1] + 1.0)
    cNrmNow = np.append(cNrmNow, cNrmNow[-1] + s_MPCmin)

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
            mXtra = cNrmNow[-1] - cNrmCnst[-1] / (1.0 - s_MPCmin)
            mCrit = mNrmNow[-1] + mXtra
            cCrit = mCrit - BoroCnstArt
            mNrmNow = np.array([BoroCnstArt, mCrit, mCrit + 1.0])
            cNrmNow = np.array([0.0, cCrit, cCrit + s_MPCmin])

    # If the mNrm and cNrm grids have become too large, throw out the last
    # kink point, being sure to adjust the extrapolation.
    if mNrmNow.size > s_MaxKinks:
        s_mNrmNow = np.concatenate((mNrmNow[:-2], np.array([mNrmNow[-3] + 1.0])))
        s_cNrmNow = np.concatenate((cNrmNow[:-2], np.array([cNrmNow[-3] + s_MPCmin])))

    # Construct the consumption function as a linear interpolation.
    # s_.cFunc = LinearInterp(mNrmNow, cNrmNow)

    # Calculate the upper bound of the MPC as the slope of the bottom segment.
    s_MPCmax = (cNrmNow[1] - cNrmNow[0]) / (mNrmNow[1] - mNrmNow[0])  # return

    # Add two attributes to enable calculation of steady state market resources.
    s_ExIncNext = 1.0  # Perfect foresight income of 1
    s_mNrmMinNow = mNrmNow[0]  # Relabeling for compatibility with addSSmNrm

    return (
        s_hNrmNow,
        s_MPCmin,
        s_MPCmax,
        mNrmNow,
        cNrmNow,
        s_ExIncNext,
        s_mNrmMinNow,
    )


@njit
def searchSSfuncNumba(m, s_PermGroFac, s_Rfree, s_ExIncNext, s_mNrmNow, s_cNrmNow):
    # Make a linear function of all combinations of c and m that yield mNext = mNow
    mZeroChange = (1.0 - s_PermGroFac / s_Rfree) * m + (
        s_PermGroFac / s_Rfree
    ) * s_ExIncNext
    # Find the steady state level of market resources
    res = interp(s_mNrmNow, s_cNrmNow, m) - mZeroChange
    # A zero of this is SS market resources
    return res


@njit
def addSSmNrmNumba(
    s_PermGroFac, s_Rfree, s_ExIncNext, s_mNrmMinNow, s_mNrmNow, s_cNrmNow,
):
    m_init_guess = s_mNrmMinNow + s_ExIncNext
    # Minimum market resources plus next income is okay starting guess
    try:
        mNrmSS = newton_secant(
            searchSSfuncNumba,
            m_init_guess,
            args=(s_PermGroFac, s_Rfree, s_ExIncNext, s_mNrmNow, s_cNrmNow),
        )
    except:
        mNrmSS = None

    return mNrmSS


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


class PerfForesightConsumerTypeNumba(PerfForesightConsumerType):
    def __init__(self, **kwargs):
        PerfForesightConsumerType.__init__(self, **kwargs)

        self.solveOnePeriod = makeOnePeriodOOSolver(ConsPerfForesightSolverNumba)

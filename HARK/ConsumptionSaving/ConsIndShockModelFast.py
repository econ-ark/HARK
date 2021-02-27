"""
Classes to solve canonical consumption-savings models with idiosyncratic shocks
to income.  All models here assume CRRA utility with geometric discounting, no
bequest motive, and income shocks are fully transitory or fully permanent.

It currently solves three types of models:
   1) A very basic "perfect foresight" consumption-savings model with no uncertainty.
   2) A consumption-savings model with risk over transitory and permanent income shocks.
   3) The model described in (2), with an interest rate for debt that differs
      from the interest rate for savings. #todo

See NARK https://HARK.githhub.io/Documentation/NARK for information on variable naming conventions.
See HARK documentation for mathematical descriptions of the models being solved.
"""

from copy import deepcopy

import numpy as np
from interpolation import interp
from numba import njit
from quantecon.optimize import newton_secant

from HARK import make_one_period_oo_solver, MetricObject
from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsumerSolution,
    ConsPerfForesightSolver,
    ConsIndShockSolverBasic,
    PerfForesightConsumerType,
    IndShockConsumerType,
)
from HARK.interpolation import (
    LinearInterp,
    LowerEnvelope,
    CubicInterp,
    ValueFuncCRRA,
    MargValueFuncCRRA,
    MargMargValueFuncCRRA
)
from HARK.numba import (
    CRRAutility,
    CRRAutilityP,
    CRRAutilityPP,
    CRRAutilityP_inv,
    CRRAutility_invP,
    CRRAutility_inv,
    CRRAutilityP_invP,
)
from HARK.numba import linear_interp_fast, cubic_interp_fast, linear_interp_deriv_fast

__all__ = [
    "PerfForesightSolution",
    "IndShockSolution",
    "ConsPerfForesightSolverFast",
    "ConsIndShockSolverBasicFast",
    "ConsIndShockSolverFast",
    "PerfForesightConsumerTypeFast",
    "IndShockConsumerTypeFast",
]

utility = CRRAutility
utilityP = CRRAutilityP
utilityPP = CRRAutilityPP
utilityP_inv = CRRAutilityP_inv
utility_invP = CRRAutility_invP
utility_inv = CRRAutility_inv
utilityP_invP = CRRAutilityP_invP


# =====================================================================
# === Classes that help solve consumption-saving models ===
# =====================================================================


class PerfForesightSolution(MetricObject):
    """
    A class representing the solution of a single period of a consumption-saving
    perfect foresight problem.

    Here and elsewhere in the code, Nrm indicates that variables are normalized
    by permanent income.

    Parameters
    ----------
    mNrm: np.array
        (Normalized) corresponding market resource points for interpolation.
    cNrm : np.array
        (Normalized) consumption points for interpolation.
    vFuncNvrsSlope: float
        Constant slope of inverse value vFuncNvrs
    mNrmMin : float
        The minimum allowable market resources for this period; the consump-
        tion function (etc) are undefined for m < mNrmMin.
    hNrm : float
        Human wealth after receiving income this period: PDV of all future
        income, ignoring mortality.
    MPCmin : float
        Infimum of the marginal propensity to consume this period.
        MPC --> MPCmin as m --> infinity.
    MPCmax : float
        Supremum of the marginal propensity to consume this period.
        MPC --> MPCmax as m --> mNrmMin.
    """

    distance_criteria = ["cNrm", "mNrm"]

    def __init__(
        self,
        mNrm=np.array([0.0, 1.0]),
        cNrm=np.array([0.0, 1.0]),
        vFuncNvrsSlope=0.0,
        mNrmMin=0.0,
        hNrm=0.0,
        MPCmin=1.0,
        MPCmax=1.0,
    ):
        self.mNrm = mNrm
        self.cNrm = cNrm
        self.vFuncNvrsSlope = vFuncNvrsSlope
        self.mNrmMin = mNrmMin
        self.hNrm = hNrm
        self.MPCmin = MPCmin
        self.MPCmax = MPCmax


class IndShockSolution(MetricObject):
    """
    A class representing the solution of a single period of a consumption-saving
    idiosyncratic shocks to permanent and transitory income problem.

    Parameters
    ----------
    mNrm: np.array
        (Normalized) corresponding market resource points for interpolation.
    cNrm : np.array
        (Normalized) consumption points for interpolation.
    vFuncNvrsSlope: float
        Constant slope of inverse value vFuncNvrs
    mNrmMin : float
        The minimum allowable market resources for this period; the consump-
        tion function (etc) are undefined for m < mNrmMin.
    hNrm : float
        Human wealth after receiving income this period: PDV of all future
        income, ignoring mortality.
    MPCmin : float
        Infimum of the marginal propensity to consume this period.
        MPC --> MPCmin as m --> infinity.
    MPCmax : float
        Supremum of the marginal propensity to consume this period.
        MPC --> MPCmax as m --> mNrmMin.
    """

    distance_criteria = ["cNrm", "mNrm", "mNrmMin"]

    def __init__(
        self,
        mNrm=np.linspace(0, 1),
        cNrm=np.linspace(0, 1),
        cFuncLimitIntercept=None,
        cFuncLimitSlope=None,
        mNrmMin=0.0,
        hNrm=0.0,
        MPCmin=1.0,
        MPCmax=1.0,
        ExIncNext=0.0,
        MPC=None,
        mNrmGrid=None,
        vNvrs=None,
        vNvrsP=None,
        MPCminNvrs=None,
    ):
        self.mNrm = mNrm
        self.cNrm = cNrm
        self.cFuncLimitIntercept = cFuncLimitIntercept
        self.cFuncLimitSlope = cFuncLimitSlope
        self.mNrmMin = mNrmMin
        self.hNrm = hNrm
        self.MPCmin = MPCmin
        self.MPCmax = MPCmax
        self.ExIncNext = ExIncNext
        self.mNrmGrid = mNrmGrid
        self.vNvrs = vNvrs
        self.MPCminNvrs = MPCminNvrs
        self.MPC = MPC
        self.vNvrsP = vNvrsP


# =====================================================================
# === Classes and functions that solve consumption-saving models ===
# =====================================================================


@njit(cache=True)
def _searchSSfunc(m, Rfree, PermGroFac, mNrm, cNrm, ExIncNext):
    # Make a linear function of all combinations of c and m that yield mNext = mNow
    mZeroChange = (1.0 - PermGroFac / Rfree) * m + (PermGroFac / Rfree) * ExIncNext

    # Find the steady state level of market resources
    res = interp(mNrm, cNrm, m) - mZeroChange
    # A zero of this is SS market resources
    return res


# @njit(cache=True) can't cache because of use of globals, perhaps newton_secant?
@njit
def _addSSmNrmNumba(Rfree, PermGroFac, mNrm, cNrm, mNrmMin, ExIncNext, _searchSSfunc):
    """
    Finds steady state (normalized) market resources and adds it to the
    solution.  This is the level of market resources such that the expectation
    of market resources in the next period is unchanged.  This value doesn't
    necessarily exist.
    """

    # Minimum market resources plus next income is okay starting guess
    m_init_guess = mNrmMin + ExIncNext

    mNrmSS = newton_secant(
        _searchSSfunc,
        m_init_guess,
        args=(Rfree, PermGroFac, mNrm, cNrm, ExIncNext),
        disp=False,
    )

    if mNrmSS.converged:
        return mNrmSS.root
    else:
        return None


@njit(cache=True)
def _solveConsPerfForesightNumba(
    DiscFac,
    LivPrb,
    CRRA,
    Rfree,
    PermGroFac,
    BoroCnstArt,
    MaxKinks,
    mNrmNext,
    cNrmNext,
    hNrmNext,
    MPCminNext,
):
    """
    Makes the (linear) consumption function for this period.
    """

    DiscFacEff = DiscFac * LivPrb

    # Calculate human wealth this period
    hNrmNow = (PermGroFac / Rfree) * (hNrmNext + 1.0)

    # Calculate the lower bound of the marginal propensity to consume
    PatFac = ((Rfree * DiscFacEff) ** (1.0 / CRRA)) / Rfree
    MPCmin = 1.0 / (1.0 + PatFac / MPCminNext)

    # Extract the discrete kink points in next period's consumption function;
    # don't take the last one, as it only defines the extrapolation and is not a kink.
    mNrmNext = mNrmNext[:-1]
    cNrmNext = cNrmNext[:-1]

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

    # Calculate the upper bound of the MPC as the slope of the bottom segment.
    MPCmax = (cNrmNow[1] - cNrmNow[0]) / (mNrmNow[1] - mNrmNow[0])

    # Add attributes to enable calculation of steady state market resources.
    # Relabeling for compatibility with addSSmNrm
    mNrmMinNow = mNrmNow[0]

    # See the PerfForesightConsumerType.ipynb documentation notebook for the derivations
    vFuncNvrsSlope = MPCmin ** (-CRRA / (1.0 - CRRA))

    return (
        mNrmNow,
        cNrmNow,
        vFuncNvrsSlope,
        mNrmMinNow,
        hNrmNow,
        MPCmin,
        MPCmax,
    )


class ConsPerfForesightSolverFast(ConsPerfForesightSolver):
    """
    A class for solving a one period perfect foresight consumption-saving problem.
    An instance of this class is created by the function solvePerfForesight in each period.
    """

    def solve(self):
        """
        Solves the one period perfect foresight consumption-saving problem.

        Parameters
        ----------
        None

        Returns
        -------
        solution : PerfForesightSolution
            The solution to this period's problem.
        """

        # Use a local value of BoroCnstArt to prevent comparing None and float below.
        if self.BoroCnstArt is None:
            BoroCnstArt = -np.inf
        else:
            BoroCnstArt = self.BoroCnstArt

        (
            self.mNrmNow,
            self.cNrmNow,
            self.vFuncNvrsSlope,
            self.mNrmMinNow,
            self.hNrmNow,
            self.MPCmin,
            self.MPCmax,
        ) = _solveConsPerfForesightNumba(
            self.DiscFac,
            self.LivPrb,
            self.CRRA,
            self.Rfree,
            self.PermGroFac,
            BoroCnstArt,
            self.MaxKinks,
            self.solution_next.mNrm,
            self.solution_next.cNrm,
            self.solution_next.hNrm,
            self.solution_next.MPCmin,
        )

        solution = PerfForesightSolution(
            self.mNrmNow,
            self.cNrmNow,
            self.vFuncNvrsSlope,
            self.mNrmMinNow,
            self.hNrmNow,
            self.MPCmin,
            self.MPCmax,
        )
        return solution


@njit(cache=True)
def _np_tile(A, reps):
    return A.repeat(reps[0]).reshape(A.size, -1).transpose()


@njit(cache=True)
def _np_insert(arr, obj, values, axis=-1):
    return np.append(np.array(values), arr)


@njit(cache=True)
def _prepareToSolveConsIndShockNumba(
    DiscFac,
    LivPrb,
    CRRA,
    Rfree,
    PermGroFac,
    BoroCnstArt,
    aXtraGrid,
    hNrmNext,
    mNrmMinNext,
    MPCminNext,
    MPCmaxNext,
    PermShkValsNext,
    TranShkValsNext,
    ShkPrbsNext,
):
    """
    Unpacks some of the inputs (and calculates simple objects based on them),
    storing the results in self for use by other methods.  These include:
    income shocks and probabilities, next period's marginal value function
    (etc), the probability of getting the worst income shock next period,
    the patience factor, human wealth, and the bounding MPCs.

    Defines the constrained portion of the consumption function as cFuncNowCnst,
    an attribute of self.  Uses the artificial and natural borrowing constraints.

    """

    DiscFacEff = DiscFac * LivPrb  # "effective" discount factor
    PermShkMinNext = np.min(PermShkValsNext)
    TranShkMinNext = np.min(TranShkValsNext)
    WorstIncPrb = np.sum(
        ShkPrbsNext[
            (PermShkValsNext * TranShkValsNext) == (PermShkMinNext * TranShkMinNext)
        ]
    )

    # Update the bounding MPCs and PDV of human wealth:
    PatFac = ((Rfree * DiscFacEff) ** (1.0 / CRRA)) / Rfree
    MPCminNow = 1.0 / (1.0 + PatFac / MPCminNext)
    ExIncNext = np.dot(ShkPrbsNext, TranShkValsNext * PermShkValsNext)
    hNrmNow = PermGroFac / Rfree * (ExIncNext + hNrmNext)
    MPCmaxNow = 1.0 / (1.0 + (WorstIncPrb ** (1.0 / CRRA)) * PatFac / MPCmaxNext)

    cFuncLimitIntercept = MPCminNow * hNrmNow
    cFuncLimitSlope = MPCminNow

    # Calculate the minimum allowable value of money resources in this period
    BoroCnstNat = (mNrmMinNext - TranShkMinNext) * (PermGroFac * PermShkMinNext) / Rfree

    # Note: need to be sure to handle BoroCnstArt==None appropriately.
    # In Py2, this would evaluate to 5.0:  np.max([None, 5.0]).
    # However in Py3, this raises a TypeError. Thus here we need to directly
    # address the situation in which BoroCnstArt == None:
    if BoroCnstArt is None:
        mNrmMinNow = BoroCnstNat
    else:
        mNrmMinNow = np.max(np.array([BoroCnstNat, BoroCnstArt]))
    if BoroCnstNat < mNrmMinNow:
        MPCmaxEff = 1.0  # If actually constrained, MPC near limit is 1
    else:
        MPCmaxEff = MPCmaxNow

    """
    Prepare to calculate end-of-period marginal value by creating an array
    of market resources that the agent could have next period, considering
    the grid of end-of-period assets and the distribution of shocks he might
    experience next period.
    """

    # We define aNrmNow all the way from BoroCnstNat up to max(self.aXtraGrid)
    # even if BoroCnstNat < BoroCnstArt, so we can construct the consumption
    # function as the lower envelope of the (by the artificial borrowing con-
    # straint) uconstrained consumption function, and the artificially con-
    # strained consumption function.
    aNrmNow = np.asarray(aXtraGrid) + BoroCnstNat
    ShkCount = TranShkValsNext.size
    aNrm_temp = _np_tile(aNrmNow, (ShkCount, 1))

    # Tile arrays of the income shocks and put them into useful shapes
    aNrmCount = aNrmNow.shape[0]
    PermShkVals_temp = (_np_tile(PermShkValsNext, (aNrmCount, 1))).transpose()
    TranShkVals_temp = (_np_tile(TranShkValsNext, (aNrmCount, 1))).transpose()
    ShkPrbs_temp = (_np_tile(ShkPrbsNext, (aNrmCount, 1))).transpose()

    # Get cash on hand next period
    mNrmNext = Rfree / (PermGroFac * PermShkVals_temp) * aNrm_temp + TranShkVals_temp
    # CDC 20191205: This should be divided by LivPrb[0] for Blanchard insurance

    return (
        DiscFacEff,
        BoroCnstNat,
        cFuncLimitIntercept,
        cFuncLimitSlope,
        mNrmMinNow,
        hNrmNow,
        MPCminNow,
        MPCmaxNow,
        MPCmaxEff,
        ExIncNext,
        mNrmNext,
        PermShkVals_temp,
        ShkPrbs_temp,
        aNrmNow,
    )


@njit(cache=True)
def _solveConsIndShockLinearNumba(
    mNrmMinNext,
    mNrmNext,
    CRRA,
    mNrmUnc,
    cNrmUnc,
    DiscFacEff,
    Rfree,
    PermGroFac,
    PermShkVals_temp,
    ShkPrbs_temp,
    aNrmNow,
    BoroCnstNat,
    cFuncInterceptNext,
    cFuncSlopeNext,
):
    """
    Calculate end-of-period marginal value of assets at each point in aNrmNow.
    Does so by taking a weighted sum of next period marginal values across
    income shocks (in a preconstructed grid self.mNrmNext).
    """

    mNrmCnst = np.array([mNrmMinNext, mNrmMinNext + 1])
    cNrmCnst = np.array([0.0, 1.0])
    cFuncNextCnst = linear_interp_fast(mNrmNext.flatten(), mNrmCnst, cNrmCnst)
    cFuncNextUnc = linear_interp_fast(
        mNrmNext.flatten(), mNrmUnc, cNrmUnc, cFuncInterceptNext, cFuncSlopeNext
    )
    cFuncNext = np.minimum(cFuncNextCnst, cFuncNextUnc)
    vPfuncNext = utilityP(cFuncNext, CRRA).reshape(mNrmNext.shape)

    EndOfPrdvP = (
        DiscFacEff
        * Rfree
        * PermGroFac ** (-CRRA)
        * np.sum(PermShkVals_temp ** (-CRRA) * vPfuncNext * ShkPrbs_temp, axis=0)
    )

    # Finds interpolation points (c,m) for the consumption function.

    cNrmNow = utilityP_inv(EndOfPrdvP, CRRA)
    mNrmNow = cNrmNow + aNrmNow

    # Limiting consumption is zero as m approaches mNrmMin
    cNrm = _np_insert(cNrmNow, 0, 0.0, axis=-1)
    mNrm = _np_insert(mNrmNow, 0, BoroCnstNat, axis=-1)

    return (cNrm, mNrm, EndOfPrdvP)


class ConsIndShockSolverBasicFast(ConsIndShockSolverBasic):
    """
    This class solves a single period of a standard consumption-saving problem,
    using linear interpolation and without the ability to calculate the value
    function.  ConsIndShockSolver inherits from this class and adds the ability
    to perform cubic interpolation and to calculate the value function.

    Note that this class does not have its own initializing method.  It initial-
    izes the same problem in the same way as ConsIndShockSetup, from which it
    inherits.
    """

    def prepareToSolve(self):
        """
        Perform preparatory work before calculating the unconstrained consumption
        function.
        Parameters
        ----------
        none
        Returns
        -------
        none
        """

        self.ShkPrbsNext = self.IncShkDstn.pmf
        self.PermShkValsNext = self.IncShkDstn.X[0]
        self.TranShkValsNext = self.IncShkDstn.X[1]

        (
            self.DiscFacEff,
            self.BoroCnstNat,
            self.cFuncLimitIntercept,
            self.cFuncLimitSlope,
            self.mNrmMinNow,
            self.hNrmNow,
            self.MPCminNow,
            self.MPCmaxNow,
            self.MPCmaxEff,
            self.ExIncNext,
            self.mNrmNext,
            self.PermShkVals_temp,
            self.ShkPrbs_temp,
            self.aNrmNow,
        ) = _prepareToSolveConsIndShockNumba(
            self.DiscFac,
            self.LivPrb,
            self.CRRA,
            self.Rfree,
            self.PermGroFac,
            self.BoroCnstArt,
            self.aXtraGrid,
            self.solution_next.hNrm,
            self.solution_next.mNrmMin,
            self.solution_next.MPCmin,
            self.solution_next.MPCmax,
            self.PermShkValsNext,
            self.TranShkValsNext,
            self.ShkPrbsNext,
        )

    def solve(self):
        """
        Solves a one period consumption saving problem with risky income.
        Parameters
        ----------
        None
        Returns
        -------
        solution : ConsumerSolution
            The solution to the one period problem.
        """

        self.cNrm, self.mNrm, self.EndOfPrdvP = _solveConsIndShockLinearNumba(
            self.solution_next.mNrmMin,
            self.mNrmNext,
            self.CRRA,
            self.solution_next.mNrm,
            self.solution_next.cNrm,
            self.DiscFacEff,
            self.Rfree,
            self.PermGroFac,
            self.PermShkVals_temp,
            self.ShkPrbs_temp,
            self.aNrmNow,
            self.BoroCnstNat,
            self.solution_next.cFuncLimitIntercept,
            self.solution_next.cFuncLimitSlope,
        )

        # Pack up the solution and return it
        solution = IndShockSolution(
            self.mNrm,
            self.cNrm,
            self.cFuncLimitIntercept,
            self.cFuncLimitSlope,
            self.mNrmMinNow,
            self.hNrmNow,
            self.MPCminNow,
            self.MPCmaxEff,
            self.ExIncNext,
        )

        return solution


@njit(cache=True)
def _solveConsIndShockCubicNumba(
    mNrmMinNext,
    mNrmNext,
    mNrmUnc,
    cNrmUnc,
    MPCNext,
    cFuncInterceptNext,
    cFuncSlopeNext,
    CRRA,
    DiscFacEff,
    Rfree,
    PermGroFac,
    PermShkVals_temp,
    ShkPrbs_temp,
    aNrmNow,
    BoroCnstNat,
    MPCmaxNow,
):
    mNrmCnst = np.array([mNrmMinNext, mNrmMinNext + 1])
    cNrmCnst = np.array([0.0, 1.0])
    cFuncNextCnst, MPCNextCnst = linear_interp_deriv_fast(
        mNrmNext.flatten(), mNrmCnst, cNrmCnst
    )
    cFuncNextUnc, MPCNextUnc = cubic_interp_fast(
        mNrmNext.flatten(),
        mNrmUnc,
        cNrmUnc,
        MPCNext,
        cFuncInterceptNext,
        cFuncSlopeNext,
    )

    cFuncNext = np.where(cFuncNextCnst <= cFuncNextUnc, cFuncNextCnst, cFuncNextUnc)

    vPfuncNext = utilityP(cFuncNext, CRRA).reshape(mNrmNext.shape)

    EndOfPrdvP = (
        DiscFacEff
        * Rfree
        * PermGroFac ** (-CRRA)
        * np.sum(PermShkVals_temp ** (-CRRA) * vPfuncNext * ShkPrbs_temp, axis=0)
    )
    # Finds interpolation points (c,m) for the consumption function.

    cNrmNow = EndOfPrdvP ** (-1.0 / CRRA)
    mNrmNow = cNrmNow + aNrmNow

    # Limiting consumption is zero as m approaches mNrmMin
    cNrm = _np_insert(cNrmNow, 0, 0.0, axis=-1)
    mNrm = _np_insert(mNrmNow, 0, BoroCnstNat, axis=-1)

    """
    Makes a cubic spline interpolation of the unconstrained consumption
    function for this period.
    """

    MPCinterpNext = np.where(cFuncNextCnst <= cFuncNextUnc, MPCNextCnst, MPCNextUnc)
    vPPfuncNext = (MPCinterpNext * utilityPP(cFuncNext, CRRA)).reshape(mNrmNext.shape)

    EndOfPrdvPP = (
        DiscFacEff
        * Rfree
        * Rfree
        * PermGroFac ** (-CRRA - 1.0)
        * np.sum(PermShkVals_temp ** (-CRRA - 1.0) * vPPfuncNext * ShkPrbs_temp, axis=0)
    )
    dcda = EndOfPrdvPP / utilityPP(cNrm[1:], CRRA)
    MPC = dcda / (dcda + 1.0)
    MPC = _np_insert(MPC, 0, MPCmaxNow)

    return cNrm, mNrm, MPC, EndOfPrdvP


@njit(cache=True)
def _cFuncCubic(aXtraGrid, mNrmMinNow, mNrmNow, cNrmNow, MPCNow, MPCminNow, hNrmNow):
    mNrmGrid = mNrmMinNow + aXtraGrid
    mNrmCnst = np.array([mNrmMinNow, mNrmMinNow + 1])
    cNrmCnst = np.array([0.0, 1.0])
    cFuncNowCnst = linear_interp_fast(mNrmGrid.flatten(), mNrmCnst, cNrmCnst)
    cFuncNowUnc, MPCNowUnc = cubic_interp_fast(
        mNrmGrid.flatten(), mNrmNow, cNrmNow, MPCNow, MPCminNow * hNrmNow, MPCminNow
    )

    cNrmNow = np.where(cFuncNowCnst <= cFuncNowUnc, cFuncNowCnst, cFuncNowUnc)

    return cNrmNow, mNrmGrid


@njit(cache=True)
def _cFuncLinear(aXtraGrid, mNrmMinNow, mNrmNow, cNrmNow, MPCminNow, hNrmNow):
    mNrmGrid = mNrmMinNow + aXtraGrid
    mNrmCnst = np.array([mNrmMinNow, mNrmMinNow + 1])
    cNrmCnst = np.array([0.0, 1.0])
    cFuncNowCnst = linear_interp_fast(mNrmGrid.flatten(), mNrmCnst, cNrmCnst)
    cFuncNowUnc = linear_interp_fast(
        mNrmGrid.flatten(), mNrmNow, cNrmNow, MPCminNow * hNrmNow, MPCminNow
    )

    cNrmNow = np.where(cFuncNowCnst <= cFuncNowUnc, cFuncNowCnst, cFuncNowUnc)

    return cNrmNow, mNrmGrid


@njit(cache=True)
def _addvFuncNumba(
    mNrmNext,
    mNrmGridNext,
    vNvrsNext,
    vNvrsPNext,
    MPCminNvrsNext,
    hNrmNext,
    CRRA,
    PermShkVals_temp,
    PermGroFac,
    DiscFacEff,
    ShkPrbs_temp,
    EndOfPrdvP,
    aNrmNow,
    BoroCnstNat,
    mNrmGrid,
    cFuncNow,
    mNrmMinNow,
    MPCmaxEff,
    MPCminNow,
):
    """
    Construct the end-of-period value function for this period, storing it
    as an attribute of self for use by other methods.
    """

    # vFunc always cubic

    vNvrsFuncNow, _ = cubic_interp_fast(
        mNrmNext.flatten(),
        mNrmGridNext,
        vNvrsNext,
        vNvrsPNext,
        MPCminNvrsNext * hNrmNext,
        MPCminNvrsNext,
    )

    vFuncNext = utility(vNvrsFuncNow, CRRA).reshape(mNrmNext.shape)

    VLvlNext = (
        PermShkVals_temp ** (1.0 - CRRA) * PermGroFac ** (1.0 - CRRA)
    ) * vFuncNext
    EndOfPrdv = DiscFacEff * np.sum(VLvlNext * ShkPrbs_temp, axis=0)

    # value transformed through inverse utility
    EndOfPrdvNvrs = utility_inv(EndOfPrdv, CRRA)
    EndOfPrdvNvrsP = EndOfPrdvP * utility_invP(EndOfPrdv, CRRA)
    EndOfPrdvNvrs = _np_insert(EndOfPrdvNvrs, 0, 0.0)

    # This is a very good approximation, vNvrsPP = 0 at the asset minimum
    EndOfPrdvNvrsP = _np_insert(EndOfPrdvNvrsP, 0, EndOfPrdvNvrsP[0])
    aNrm_temp = _np_insert(aNrmNow, 0, BoroCnstNat)

    """
    Creates the value function for this period, defined over market resources m.
    self must have the attribute EndOfPrdvFunc in order to execute.
    """

    # Compute expected value and marginal value on a grid of market resources

    aNrmNow = mNrmGrid - cFuncNow

    EndOfPrdvNvrsFunc, _ = cubic_interp_fast(
        aNrmNow, aNrm_temp, EndOfPrdvNvrs, EndOfPrdvNvrsP
    )
    EndOfPrdvFunc = utility(EndOfPrdvNvrsFunc, CRRA)

    vNrmNow = utility(cFuncNow, CRRA) + EndOfPrdvFunc
    vPnow = utilityP(cFuncNow, CRRA)

    # Construct the beginning-of-period value function
    vNvrs = utility_inv(vNrmNow, CRRA)  # value transformed through inverse utility
    vNvrsP = vPnow * utility_invP(vNrmNow, CRRA)
    mNrmGrid = _np_insert(mNrmGrid, 0, mNrmMinNow)
    vNvrs = _np_insert(vNvrs, 0, 0.0)
    vNvrsP = _np_insert(vNvrsP, 0, MPCmaxEff ** (-CRRA / (1.0 - CRRA)))
    MPCminNvrs = MPCminNow ** (-CRRA / (1.0 - CRRA))

    return (
        mNrmGrid,
        vNvrs,
        vNvrsP,
        MPCminNvrs,
    )


@njit
def _addSSmNrmIndNumba(
    PermGroFac, Rfree, ExIncNext, mNrmMin, mNrm, cNrm, MPC, MPCmin, hNrm, _searchfunc,
):
    """
    Finds steady state (normalized) market resources and adds it to the
    solution.  This is the level of market resources such that the expectation
    of market resources in the next period is unchanged.  This value doesn't
    necessarily exist.
    """

    # Minimum market resources plus next income is okay starting guess
    m_init_guess = mNrmMin + ExIncNext

    mNrmSS = newton_secant(
        _searchfunc,
        m_init_guess,
        args=(PermGroFac, Rfree, ExIncNext, mNrmMin, mNrm, cNrm, MPC, MPCmin, hNrm),
        disp=False,
    )

    if mNrmSS.converged:
        return mNrmSS.root
    else:
        return None


@njit(cache=True)
def _searchSSfuncLinear(
    m, PermGroFac, Rfree, ExIncNext, mNrmMin, mNrm, cNrm, MPC, MPCmin, hNrm
):
    # Make a linear function of all combinations of c and m that yield mNext = mNow
    mZeroChange = (1.0 - PermGroFac / Rfree) * m + (PermGroFac / Rfree) * ExIncNext

    mNrmCnst = np.array([mNrmMin, mNrmMin + 1])
    cNrmCnst = np.array([0.0, 1.0])
    cFuncNowCnst = linear_interp_fast(np.array([m]), mNrmCnst, cNrmCnst)
    cFuncNowUnc = linear_interp_fast(np.array([m]), mNrm, cNrm, MPCmin * hNrm, MPCmin)

    cNrmNow = np.where(cFuncNowCnst <= cFuncNowUnc, cFuncNowCnst, cFuncNowUnc)

    # Find the steady state level of market resources
    res = cNrmNow[0] - mZeroChange
    # A zero of this is SS market resources
    return res


@njit(cache=True)
def _searchSSfuncCubic(
    m, PermGroFac, Rfree, ExIncNext, mNrmMin, mNrm, cNrm, MPC, MPCmin, hNrm
):
    # Make a linear function of all combinations of c and m that yield mNext = mNow
    mZeroChange = (1.0 - PermGroFac / Rfree) * m + (PermGroFac / Rfree) * ExIncNext

    mNrmCnst = np.array([mNrmMin, mNrmMin + 1])
    cNrmCnst = np.array([0.0, 1.0])
    cFuncNowCnst = linear_interp_fast(np.array([m]), mNrmCnst, cNrmCnst)
    cFuncNowUnc, MPCNowUnc = cubic_interp_fast(
        np.array([m]), mNrm, cNrm, MPC, MPCmin * hNrm, MPCmin
    )

    cNrmNow = np.where(cFuncNowCnst <= cFuncNowUnc, cFuncNowCnst, cFuncNowUnc)

    # Find the steady state level of market resources
    res = cNrmNow[0] - mZeroChange
    # A zero of this is SS market resources
    return res


class ConsIndShockSolverFast(ConsIndShockSolverBasicFast):
    """
    This class solves a single period of a standard consumption-saving problem.
    It inherits from ConsIndShockSolverBasic, adding the ability to perform cubic
    interpolation and to calculate the value function.
    """

    def solve(self):
        """
        Solves a one period consumption saving problem with risky income.
        Parameters
        ----------
        None
        Returns
        -------
        solution : ConsumerSolution
            The solution to the one period problem.
        """

        if self.CubicBool:
            (
                self.cNrm,
                self.mNrm,
                self.MPC,
                self.EndOfPrdvP,
            ) = _solveConsIndShockCubicNumba(
                self.solution_next.mNrmMin,
                self.mNrmNext,
                self.solution_next.mNrm,
                self.solution_next.cNrm,
                self.solution_next.MPC,
                self.solution_next.cFuncLimitIntercept,
                self.solution_next.cFuncLimitSlope,
                self.CRRA,
                self.DiscFacEff,
                self.Rfree,
                self.PermGroFac,
                self.PermShkVals_temp,
                self.ShkPrbs_temp,
                self.aNrmNow,
                self.BoroCnstNat,
                self.MPCmaxNow,
            )
            # Pack up the solution and return it
            solution = IndShockSolution(
                self.mNrm,
                self.cNrm,
                self.cFuncLimitIntercept,
                self.cFuncLimitSlope,
                self.mNrmMinNow,
                self.hNrmNow,
                self.MPCminNow,
                self.MPCmaxEff,
                self.ExIncNext,
                self.MPC,
            )
        else:
            self.cNrm, self.mNrm, self.EndOfPrdvP = _solveConsIndShockLinearNumba(
                self.solution_next.mNrmMin,
                self.mNrmNext,
                self.CRRA,
                self.solution_next.mNrm,
                self.solution_next.cNrm,
                self.DiscFacEff,
                self.Rfree,
                self.PermGroFac,
                self.PermShkVals_temp,
                self.ShkPrbs_temp,
                self.aNrmNow,
                self.BoroCnstNat,
                self.solution_next.cFuncLimitIntercept,
                self.solution_next.cFuncLimitSlope,
            )

            # Pack up the solution and return it
            solution = IndShockSolution(
                self.mNrm,
                self.cNrm,
                self.cFuncLimitIntercept,
                self.cFuncLimitSlope,
                self.mNrmMinNow,
                self.hNrmNow,
                self.MPCminNow,
                self.MPCmaxEff,
                self.ExIncNext,
            )

        if self.vFuncBool:

            if self.CubicBool:
                self.cFuncNow, self.mNrmGrid = _cFuncCubic(
                    self.aXtraGrid,
                    self.mNrmMinNow,
                    self.mNrm,
                    self.cNrm,
                    self.MPC,
                    self.MPCminNow,
                    self.hNrmNow,
                )
            else:
                self.cFuncNow, self.mNrmGrid = _cFuncLinear(
                    self.aXtraGrid,
                    self.mNrmMinNow,
                    self.mNrm,
                    self.cNrm,
                    self.MPCminNow,
                    self.hNrmNow,
                )

            self.mNrmGrid, self.vNvrs, self.vNvrsP, self.MPCminNvrs = _addvFuncNumba(
                self.mNrmNext,
                self.solution_next.mNrmGrid,
                self.solution_next.vNvrs,
                self.solution_next.vNvrsP,
                self.solution_next.MPCminNvrs,
                self.solution_next.hNrm,
                self.CRRA,
                self.PermShkVals_temp,
                self.PermGroFac,
                self.DiscFacEff,
                self.ShkPrbs_temp,
                self.EndOfPrdvP,
                self.aNrmNow,
                self.BoroCnstNat,
                self.mNrmGrid,
                self.cFuncNow,
                self.mNrmMinNow,
                self.MPCmaxEff,
                self.MPCminNow,
            )

            # Pack up the solution and return it

            solution.mNrmGrid = self.mNrmGrid
            solution.vNvrs = self.vNvrs
            solution.vNvrsP = self.vNvrsP
            solution.MPCminNvrs = self.MPCminNvrs

        return solution


# ============================================================================
# == Classes for representing types of consumer agents (and things they do) ==
# ============================================================================


class PerfForesightConsumerTypeFast(PerfForesightConsumerType):
    """
    A perfect foresight consumer type who has no uncertainty other than mortality.
    His problem is defined by a coefficient of relative risk aversion, intertemporal
    discount factor, interest factor, an artificial borrowing constraint (maybe)
    and time sequences of the permanent income growth rate and survival probability.
    """

    # Define some universal values for all consumer types
    solution_terminal_ = PerfForesightSolution()

    def __init__(self, **kwargs):
        PerfForesightConsumerType.__init__(self, **kwargs)

        self.solve_one_period = make_one_period_oo_solver(ConsPerfForesightSolverFast)

    def updateSolutionTerminal(self):
        """
        Update the terminal period solution.  This method should be run when a
        new AgentType is created or when CRRA changes.
        """

        self.solution_terminal_cs = ConsumerSolution(
            cFunc=self.cFunc_terminal_,
            vFunc=ValueFuncCRRA(self.cFunc_terminal_, self.CRRA),
            vPfunc=MargValueFuncCRRA(self.cFunc_terminal_, self.CRRA),
            vPPfunc=MargMargValueFuncCRRA(self.cFunc_terminal_, self.CRRA),
            mNrmMin=0.0,
            hNrm=0.0,
            MPCmin=1.0,
            MPCmax=1.0,
        )

    def post_solve(self):
        self.solution_fast = deepcopy(self.solution)

        if self.cycles == 0:
            terminal = 1
        else:
            terminal = self.cycles
            self.solution[terminal] = self.solution_terminal_cs

        for i in range(terminal):
            solution = self.solution[i]

            # Construct the consumption function as a linear interpolation.
            cFunc = LinearInterp(solution.mNrm, solution.cNrm)

            """
            Defines the value and marginal value functions for this period.
            Uses the fact that for a perfect foresight CRRA utility problem,
            if the MPC in period t is :math:`\kappa_{t}`, and relative risk
            aversion :math:`\rho`, then the inverse value vFuncNvrs has a
            constant slope of :math:`\kappa_{t}^{-\rho/(1-\rho)}` and
            vFuncNvrs has value of zero at the lower bound of market resources
            mNrmMin.  See PerfForesightConsumerType.ipynb documentation notebook
            for a brief explanation and the links below for a fuller treatment.

            https://econ.jhu.edu/people/ccarroll/public/lecturenotes/consumption/PerfForesightCRRA/#vFuncAnalytical
            https://econ.jhu.edu/people/ccarroll/SolvingMicroDSOPs/#vFuncPF
            """

            vFuncNvrs = LinearInterp(
                np.array([solution.mNrmMin, solution.mNrmMin + 1.0]),
                np.array([0.0, solution.vFuncNvrsSlope]),
            )
            vFunc = ValueFuncCRRA(vFuncNvrs, self.CRRA)
            vPfunc = MargValueFuncCRRA(cFunc, self.CRRA)

            consumer_solution = ConsumerSolution(
                cFunc=cFunc,
                vFunc=vFunc,
                vPfunc=vPfunc,
                mNrmMin=solution.mNrmMin,
                hNrm=solution.hNrm,
                MPCmin=solution.MPCmin,
                MPCmax=solution.MPCmax,
            )

            ExIncNext = 1.0  # Perfect foresight income of 1

            # Add mNrmSS to the solution and return it
            consumer_solution.mNrmSS = _addSSmNrmNumba(
                self.Rfree,
                self.PermGroFac[i],
                solution.mNrm,
                solution.cNrm,
                solution.mNrmMin,
                ExIncNext,
                _searchSSfunc,
            )

            self.solution[i] = consumer_solution


class IndShockConsumerTypeFast(IndShockConsumerType, PerfForesightConsumerTypeFast):
    solution_terminal_ = IndShockSolution()

    def __init__(self, **kwargs):
        IndShockConsumerType.__init__(self, **kwargs)

        # Add consumer-type specific objects, copying to create independent versions
        if (not self.CubicBool) and (not self.vFuncBool):
            solver = ConsIndShockSolverBasicFast
        else:  # Use the "advanced" solver if either is requested
            solver = ConsIndShockSolverFast

        self.solve_one_period = make_one_period_oo_solver(solver)

    def updateSolutionTerminal(self):
        PerfForesightConsumerTypeFast.updateSolutionTerminal(self)
        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            self.solution_terminal.MPC = np.array([1.0, 1.0])
            self.solution_terminal.MPCminNvrs = 0.0
            self.solution_terminal.vNvrs = utility(np.linspace(0.0, 1.0), self.CRRA)
            self.solution_terminal.vNvrsP = utilityP(np.linspace(0.0, 1.0), self.CRRA)
            self.solution_terminal.mNrmGrid = np.linspace(0.0, 1.0)

    def post_solve(self):
        self.solution_fast = deepcopy(self.solution)

        if self.cycles == 0:
            cycles = 1
        else:
            cycles = self.cycles
            self.solution[-1] = self.solution_terminal_cs

        for i in range(cycles):
            for j in range(self.T_cycle):
                solution = self.solution[i * self.T_cycle + j]

                # Define the borrowing constraint (limiting consumption function)
                cFuncNowCnst = LinearInterp(
                    np.array([solution.mNrmMin, solution.mNrmMin + 1]),
                    np.array([0.0, 1.0]),
                )

                """
                Constructs a basic solution for this period, including the consumption
                function and marginal value function.
                """

                if self.CubicBool:
                    # Makes a cubic spline interpolation of the unconstrained consumption
                    # function for this period.
                    cFuncNowUnc = CubicInterp(
                        solution.mNrm,
                        solution.cNrm,
                        solution.MPC,
                        solution.cFuncLimitIntercept,
                        solution.cFuncLimitSlope,
                    )
                else:
                    # Makes a linear interpolation to represent the (unconstrained) consumption function.
                    # Construct the unconstrained consumption function
                    cFuncNowUnc = LinearInterp(
                        solution.mNrm,
                        solution.cNrm,
                        solution.cFuncLimitIntercept,
                        solution.cFuncLimitSlope,
                    )

                # Combine the constrained and unconstrained functions into the true consumption function
                cFuncNow = LowerEnvelope(cFuncNowUnc, cFuncNowCnst)

                # Make the marginal value function and the marginal marginal value function
                vPfuncNow = MargValueFuncCRRA(cFuncNow, self.CRRA)

                # Pack up the solution and return it
                consumer_solution = ConsumerSolution(
                    cFunc=cFuncNow,
                    vPfunc=vPfuncNow,
                    mNrmMin=solution.mNrmMin,
                    hNrm=solution.hNrm,
                    MPCmin=solution.MPCmin,
                    MPCmax=solution.MPCmax,
                )

                if self.vFuncBool:
                    vNvrsFuncNow = CubicInterp(
                        solution.mNrmGrid,
                        solution.vNvrs,
                        solution.vNvrsP,
                        solution.MPCminNvrs * solution.hNrm,
                        solution.MPCminNvrs,
                    )
                    vFuncNow = ValueFuncCRRA(vNvrsFuncNow, self.CRRA)

                    consumer_solution.vFunc = vFuncNow

                if self.CubicBool or self.vFuncBool:
                    _searchFunc = (
                        _searchSSfuncCubic if self.CubicBool else _searchSSfuncLinear
                    )
                    # Add mNrmSS to the solution and return it
                    consumer_solution.mNrmSS = _addSSmNrmIndNumba(
                        self.PermGroFac[j],
                        self.Rfree,
                        solution.ExIncNext,
                        solution.mNrmMin,
                        solution.mNrm,
                        solution.cNrm,
                        solution.MPC,
                        solution.MPCmin,
                        solution.hNrm,
                        _searchFunc,
                    )

                self.solution[i * self.T_cycle + j] = consumer_solution

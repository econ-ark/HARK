"""
Classes to solve canonical consumption-savings models with idiosyncratic shocks
to income.  All models here assume CRRA utility with geometric discounting, no
bequest motive, and income shocks are fully transitory or fully permanent.

It currently solves three types of models:
   1) A very basic "perfect foresight" consumption-savings model with no uncertainty.
   2) A consumption-savings model with risk over transitory and permanent income shocks.
   3) The model described in (2), with an interest rate for debt that differs
      from the interest rate for savings. #todo

See NARK https://github.com/econ-ark/HARK/blob/master/docs/NARK/NARK.pdf for information on variable naming conventions.
See HARK documentation for mathematical descriptions of the models being solved.
"""

import warnings
from copy import deepcopy

import numpy as np
from interpolation import interp
from numba import njit
from quantecon.optimize import newton_secant

from HARK import make_one_period_oo_solver
from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsumerSolution,
    IndShockConsumerType,
    PerfForesightConsumerType,
    init_perfect_foresight,
    init_idiosyncratic_shocks,
)
from HARK.ConsumptionSaving.LegacyOOsolvers import (
    ConsIndShockSolverBasic,
    ConsPerfForesightSolver,
)
from HARK.interpolation import (
    CubicInterp,
    LinearInterp,
    LowerEnvelope,
    MargValueFuncCRRA,
    MargMargValueFuncCRRA,
    ValueFuncCRRA,
)
from HARK.metric import MetricObject
from HARK.numba_tools import (
    CRRAutility,
    CRRAutility_inv,
    CRRAutility_invP,
    CRRAutilityP,
    CRRAutilityP_inv,
    CRRAutilityP_invP,
    CRRAutilityPP,
    cubic_interp_fast,
    linear_interp_deriv_fast,
    linear_interp_fast,
)
from HARK.utilities import NullFunc

__all__ = [
    "PerfForesightSolution",
    "IndShockSolution",
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
# === Terminal solution grid parameters for numba compatibility ===
# =====================================================================
# These parameters define the grid used to initialize vNvrs and vNvrsP arrays
# in the terminal period solution. The grid needs to cover the range of mNrmNext
# values that may be encountered during backward induction.

# Minimum value for terminal grid (near-zero to avoid division issues)
TERMINAL_GRID_MIN = 1e-6

# Maximum value for terminal grid (should be large enough to cover typical
# mNrmNext values during backward induction; 100 covers most standard cases)
TERMINAL_GRID_MAX = 100.0

# Number of points in the terminal grid (more points = better interpolation
# accuracy but slightly higher memory usage)
TERMINAL_GRID_SIZE = 200


# =====================================================================
# === Classes that help solve consumption-saving models ===
# =====================================================================


class PerfForesightSolution(MetricObject):
    r"""
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
        Constant slope of inverse value ``vFuncNvrs``
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
        mNrm=None,
        cNrm=None,
        cFuncLimitIntercept=None,
        cFuncLimitSlope=None,
        mNrmMin=0.0,
        hNrm=0.0,
        MPCmin=1.0,
        MPCmax=1.0,
        Ex_IncNext=0.0,
        MPC=None,
        mNrmGrid=None,
        vNvrs=None,
        vNvrsP=None,
        MPCminNvrs=None,
    ):
        self.mNrm = mNrm if mNrm is not None else np.linspace(0, 1)
        self.cNrm = cNrm if cNrm is not None else np.linspace(0, 1)
        self.cFuncLimitIntercept = cFuncLimitIntercept
        self.cFuncLimitSlope = cFuncLimitSlope
        self.mNrmMin = mNrmMin
        self.hNrm = hNrm
        self.MPCmin = MPCmin
        self.MPCmax = MPCmax
        self.Ex_IncNext = Ex_IncNext
        self.mNrmGrid = mNrmGrid
        self.vNvrs = vNvrs
        self.MPCminNvrs = MPCminNvrs
        self.MPC = MPC
        self.vNvrsP = vNvrsP


# =====================================================================
# === Classes and functions that solve consumption-saving models ===
# =====================================================================


def make_solution_terminal_fast(solution_terminal_class, CRRA):
    """
    Construct the terminal period solution for the fast solver.

    At terminal period, consumer consumes everything: c = m.
    Therefore (for CRRA != 1):
    - v(m) = u(m)
    - vNvrs(m) = u_inv(v(m)) = u_inv(u(m)) = m
    - vNvrsP = d(vNvrs)/dm = 1
    - MPCmin = 1 (consume everything)
    - MPCminNvrs = 1 (since MPCmin = 1)

    Note: This function requires CRRA != 1 because the vNvrs transformation
    vNvrs(m) = u_inv(u(m)) = m only holds for CRRA utility. For log utility
    (CRRA = 1), u(c) = log(c) and the inverse differs fundamentally.

    Parameters
    ----------
    solution_terminal_class : class
        The solution class to instantiate (PerfForesightSolution or IndShockSolution).
    CRRA : float
        Coefficient of relative risk aversion.

    Returns
    -------
    solution_terminal : solution_terminal_class
        The terminal period solution with properly initialized arrays for numba.
    """
    solution_terminal = solution_terminal_class()

    # Terminal consumption function: c = m
    cFunc_terminal = LinearInterp([0.0, 1.0], [0.0, 1.0])
    solution_terminal.cFunc = cFunc_terminal
    solution_terminal.vFunc = ValueFuncCRRA(cFunc_terminal, CRRA)
    solution_terminal.vPfunc = MargValueFuncCRRA(cFunc_terminal, CRRA)
    solution_terminal.vPPfunc = MargMargValueFuncCRRA(cFunc_terminal, CRRA)

    # MPC is 1 everywhere at terminal (consume everything)
    solution_terminal.MPC = np.array([1.0, 1.0])

    # At terminal, MPCmin = 1 (consume everything), so MPCminNvrs = 1
    solution_terminal.MPCminNvrs = 1.0

    # Create grid that covers typical mNrmNext range during backward induction
    # Uses module-level constants for configurability
    mNrmGrid = np.linspace(TERMINAL_GRID_MIN, TERMINAL_GRID_MAX, TERMINAL_GRID_SIZE)
    solution_terminal.mNrmGrid = mNrmGrid

    # At terminal: vNvrs(m) = u_inv(u(m)) = m (since c = m, for CRRA != 1)
    solution_terminal.vNvrs = mNrmGrid.copy()

    # vNvrsP = d(vNvrs)/dm = d(m)/dm = 1 everywhere
    solution_terminal.vNvrsP = np.ones_like(mNrmGrid)

    # hNrm = 0 at terminal (no future income)
    solution_terminal.hNrm = 0.0

    return solution_terminal


@njit(cache=True)
def _find_mNrmStE(m, Rfree, PermGroFac, mNrm, cNrm, Ex_IncNext):  # pragma: nocover
    # Make a linear function of all combinations of c and m that yield mNext = mNow
    mZeroChange = (1.0 - PermGroFac / Rfree) * m + (PermGroFac / Rfree) * Ex_IncNext

    # Find the steady state level of market resources
    res = interp(mNrm, cNrm, m) - mZeroChange
    # A zero of this is SS market resources
    return res


# @njit(cache=True) can't cache because of use of globals, perhaps newton_secant?
@njit
def _add_mNrmStENumba(
    Rfree, PermGroFac, mNrm, cNrm, mNrmMin, Ex_IncNext, _find_mNrmStE
):  # pragma: nocover
    """
    Finds steady state (normalized) market resources and adds it to the
    solution.  This is the level of market resources such that the expectation
    of market resources in the next period is unchanged.  This value doesn't
    necessarily exist.
    """

    # Minimum market resources plus next income is okay starting guess
    m_init_guess = mNrmMin + Ex_IncNext

    mNrmStE = newton_secant(
        _find_mNrmStE,
        m_init_guess,
        args=(Rfree, PermGroFac, mNrm, cNrm, Ex_IncNext),
        disp=False,
    )

    if mNrmStE.converged:
        return mNrmStE.root
    else:
        return None


@njit(cache=True, parallel=True)
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
):  # pragma: nocover
    """
    Makes the (linear) consumption function for this period.
    """

    DiscFacEff = DiscFac * LivPrb

    # Calculate human wealth this period
    hNrmNow = (PermGroFac / Rfree) * (hNrmNext + 1.0)

    # Calculate the lower bound of the marginal propensity to consume
    APF = ((Rfree * DiscFacEff) ** (1.0 / CRRA)) / Rfree
    MPCmin = 1.0 / (1.0 + APF / MPCminNext)

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
            mXtra = (cNrmNow[-1] - cNrmCnst[-1]) / (1.0 - MPCmin)
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
    # Relabeling for compatibility with add_mNrmStE
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
def _np_tile(A, reps):  # pragma: nocover
    return A.repeat(reps[0]).reshape(A.size, -1).transpose()


@njit(cache=True)
def _np_insert(arr, obj, values, axis=-1):  # pragma: nocover
    return np.append(np.array(values), arr)


@njit(cache=True, parallel=True)
def _prepare_to_solveConsIndShockNumba(
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
):  # pragma: nocover
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
    APF = ((Rfree * DiscFacEff) ** (1.0 / CRRA)) / Rfree
    MPCminNow = 1.0 / (1.0 + APF / MPCminNext)
    Ex_IncNext = np.dot(ShkPrbsNext, TranShkValsNext * PermShkValsNext)
    hNrmNow = PermGroFac / Rfree * (Ex_IncNext + hNrmNext)
    MPCmaxNow = 1.0 / (1.0 + (WorstIncPrb ** (1.0 / CRRA)) * APF / MPCmaxNext)

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
        Ex_IncNext,
        mNrmNext,
        PermShkVals_temp,
        ShkPrbs_temp,
        aNrmNow,
    )


@njit(cache=True, parallel=True)
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
):  # pragma: nocover
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

    def prepare_to_solve(self):
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

        self.ShkPrbsNext = self.IncShkDstn.pmv
        self.PermShkValsNext = self.IncShkDstn.atoms[0]
        self.TranShkValsNext = self.IncShkDstn.atoms[1]

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
            self.Ex_IncNext,
            self.mNrmNext,
            self.PermShkVals_temp,
            self.ShkPrbs_temp,
            self.aNrmNow,
        ) = _prepare_to_solveConsIndShockNumba(
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
            self.Ex_IncNext,
        )

        return solution


@njit(cache=True, parallel=True)
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
):  # pragma: nocover
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
def _cFuncCubic(
    aXtraGrid, mNrmMinNow, mNrmNow, cNrmNow, MPCNow, MPCminNow, hNrmNow
):  # pragma: nocover
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
def _cFuncLinear(
    aXtraGrid, mNrmMinNow, mNrmNow, cNrmNow, MPCminNow, hNrmNow
):  # pragma: nocover
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
def _add_vFuncNumba(
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
):  # pragma: nocover
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
def _add_mNrmStEIndNumba(
    PermGroFac,
    Rfree,
    Ex_IncNext,
    mNrmMin,
    mNrm,
    cNrm,
    MPC,
    MPCmin,
    hNrm,
    _searchfunc,
):  # pragma: nocover
    """
    Finds steady state (normalized) market resources and adds it to the
    solution.  This is the level of market resources such that the expectation
    of market resources in the next period is unchanged.  This value doesn't
    necessarily exist.
    """

    # Minimum market resources plus next income is okay starting guess
    m_init_guess = mNrmMin + Ex_IncNext

    mNrmStE = newton_secant(
        _searchfunc,
        m_init_guess,
        args=(PermGroFac, Rfree, Ex_IncNext, mNrmMin, mNrm, cNrm, MPC, MPCmin, hNrm),
        disp=False,
    )

    if mNrmStE.converged:
        return mNrmStE.root
    else:
        return None


@njit(cache=True)
def _find_mNrmStELinear(
    m, PermGroFac, Rfree, Ex_IncNext, mNrmMin, mNrm, cNrm, MPC, MPCmin, hNrm
):  # pragma: nocover
    # Make a linear function of all combinations of c and m that yield mNext = mNow
    mZeroChange = (1.0 - PermGroFac / Rfree) * m + (PermGroFac / Rfree) * Ex_IncNext

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
def _find_mNrmStECubic(
    m, PermGroFac, Rfree, Ex_IncNext, mNrmMin, mNrm, cNrm, MPC, MPCmin, hNrm
):  # pragma: nocover
    # Make a linear function of all combinations of c and m that yield mNext = mNow
    mZeroChange = (1.0 - PermGroFac / Rfree) * m + (PermGroFac / Rfree) * Ex_IncNext

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
    r"""
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
                self.Ex_IncNext,
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
                self.Ex_IncNext,
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

            self.mNrmGrid, self.vNvrs, self.vNvrsP, self.MPCminNvrs = _add_vFuncNumba(
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


init_perfect_foresight_fast = init_perfect_foresight.copy()
perf_foresight_constructor_dict = init_perfect_foresight["constructors"].copy()
perf_foresight_constructor_dict["solution_terminal"] = make_solution_terminal_fast
init_perfect_foresight_fast["constructors"] = perf_foresight_constructor_dict


class PerfForesightConsumerTypeFast(PerfForesightConsumerType):
    r"""
    A version of the perfect foresight consumer type speed up by numba.

    Note: This fast solver does not support CRRA=1 (log utility) due to the
    mathematical singularity in the inverse value function transformation.
    Use the standard PerfForesightConsumerType for log utility.
    """

    solution_terminal_class = PerfForesightSolution
    default_ = {
        "params": init_perfect_foresight_fast,
        "solver": make_one_period_oo_solver(ConsPerfForesightSolverFast),
        "model": "ConsPerfForesight.yaml",
    }

    def pre_solve(self):
        """
        Perform pre-solve checks and setup.

        Raises
        ------
        ValueError
            If CRRA equals 1 (log utility), which is not supported by the fast solver.

        Warns
        -----
        UserWarning
            If CRRA is very close to 1 (between 0.99 and 1.01), which may cause
            numerical instability.
        """
        if np.isclose(self.CRRA, 1.0):
            raise ValueError(
                "PerfForesightConsumerTypeFast does not support CRRA=1 (log utility) "
                "due to mathematical singularities in the numba-optimized solver. "
                "Please use PerfForesightConsumerType instead for log utility preferences."
            )
        # Warn for CRRA values that are close to 1 but not caught by np.isclose
        if 0.99 < self.CRRA < 1.01 and not np.isclose(self.CRRA, 1.0):
            warnings.warn(
                f"CRRA={self.CRRA} is very close to 1, which may cause numerical "
                "instability. Consider using the standard solver or a CRRA value "
                "further from 1.",
                UserWarning,
                stacklevel=2,
            )
        # Call parent's pre_solve
        super().pre_solve()

    def post_solve(self):
        self.solution_fast = deepcopy(self.solution)

        if self.cycles == 0:
            terminal = 1
        else:
            terminal = self.cycles
            self.solution[terminal] = self.solution_terminal

        for i in range(terminal):
            solution = self.solution[i]

            # Construct the consumption function as a linear interpolation.
            cFunc = LinearInterp(solution.mNrm, solution.cNrm)

            """
            Defines the value and marginal value functions for this period.
            Uses the fact that for a perfect foresight CRRA utility problem,
            if the MPC in period t is :math:`\\kappa_{t}`, and relative risk
            aversion :math:`\rho`, then the inverse value vFuncNvrs has a
            constant slope of :math:`\\kappa_{t}^{-\rho/(1-\rho)}` and
            vFuncNvrs has value of zero at the lower bound of market resources
            mNrmMin.  See PerfForesightConsumerType.ipynb documentation notebook
            for a brief explanation and the links below for a fuller treatment.

            https://www.econ2.jhu.edu/people/ccarroll/public/lecturenotes/consumption/PerfForesightCRRA/#vFuncAnalytical
            https://www.econ2.jhu.edu/people/ccarroll/SolvingMicroDSOPs/#vFuncPF
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

            Ex_IncNext = 1.0  # Perfect foresight income of 1

            # Add mNrmStE to the solution and return it
            consumer_solution.mNrmStE = _add_mNrmStENumba(
                self.Rfree[i],
                self.PermGroFac[i],
                solution.mNrm,
                solution.cNrm,
                solution.mNrmMin,
                Ex_IncNext,
                _find_mNrmStE,
            )

            self.solution[i] = consumer_solution


###############################################################################


def select_fast_solver(CubicBool, vFuncBool):
    if (not CubicBool) and (not vFuncBool):
        solver = ConsIndShockSolverBasicFast
    else:  # Use the "advanced" solver if either is requested
        solver = ConsIndShockSolverFast
    solve_one_period = make_one_period_oo_solver(solver)
    return solve_one_period


init_idiosyncratic_shocks_fast = init_idiosyncratic_shocks.copy()
ind_shock_fast_constructor_dict = init_idiosyncratic_shocks["constructors"].copy()
ind_shock_fast_constructor_dict["solution_terminal"] = make_solution_terminal_fast
ind_shock_fast_constructor_dict["solve_one_period"] = select_fast_solver
init_idiosyncratic_shocks_fast["constructors"] = ind_shock_fast_constructor_dict


class IndShockConsumerTypeFast(IndShockConsumerType, PerfForesightConsumerTypeFast):
    r"""
    A version of the idiosyncratic shock consumer type sped up by numba.

    If CubicBool and vFuncBool are both set to false it's further optimized.

    Note: This fast solver does not support CRRA=1 (log utility) due to the
    mathematical singularity in the inverse value function transformation.
    Use the standard IndShockConsumerType for log utility.
    """

    solution_terminal_class = IndShockSolution
    default_ = {
        "params": init_idiosyncratic_shocks_fast,
        "solver": NullFunc(),
        "model": "ConsIndShock.yaml",
    }

    def pre_solve(self):
        """
        Perform pre-solve checks and setup.

        Raises
        ------
        ValueError
            If CRRA equals 1 (log utility), which is not supported by the fast solver.

        Warns
        -----
        UserWarning
            If CRRA is very close to 1 (between 0.99 and 1.01), which may cause
            numerical instability.
        """
        if np.isclose(self.CRRA, 1.0):
            raise ValueError(
                "IndShockConsumerTypeFast does not support CRRA=1 (log utility) "
                "due to mathematical singularities in the numba-optimized solver. "
                "Please use IndShockConsumerType instead for log utility preferences."
            )
        # Warn for CRRA values that are close to 1 but not caught by np.isclose
        if 0.99 < self.CRRA < 1.01 and not np.isclose(self.CRRA, 1.0):
            warnings.warn(
                f"CRRA={self.CRRA} is very close to 1, which may cause numerical "
                "instability. Consider using the standard solver or a CRRA value "
                "further from 1.",
                UserWarning,
                stacklevel=2,
            )
        # Call parent's pre_solve
        super().pre_solve()

    def post_solve(self):
        self.solution_fast = deepcopy(self.solution)

        if self.cycles == 0:
            cycles = 1
        else:
            cycles = self.cycles
            self.solution[-1] = init_idiosyncratic_shocks["constructors"][
                "solution_terminal"
            ](self.CRRA)

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
                        _find_mNrmStECubic if self.CubicBool else _find_mNrmStELinear
                    )
                    # Add mNrmStE to the solution and return it
                    consumer_solution.mNrmStE = _add_mNrmStEIndNumba(
                        self.PermGroFac[j],
                        self.Rfree[j],
                        solution.Ex_IncNext,
                        solution.mNrmMin,
                        solution.mNrm,
                        solution.cNrm,
                        solution.MPC,
                        solution.MPCmin,
                        solution.hNrm,
                        _searchFunc,
                    )

                self.solution[i * self.T_cycle + j] = consumer_solution

        if (self.cycles == 0) and (self.T_cycle == 1):
            self.calc_stable_points(force=True)

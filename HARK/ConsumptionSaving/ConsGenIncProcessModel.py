"""
Classes to solve consumption-saving models with idiosyncratic shocks to income
in which shocks are not necessarily fully transitory or fully permanent.  Extends
ConsIndShockModel by explicitly tracking persistent income as a state variable,
and allows (log) persistent income to follow an AR1 process rather than random walk.
"""

import numpy as np

from HARK import AgentType, NullFunc
from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsumerSolution,
    IndShockConsumerType,
    init_idiosyncratic_shocks,
)
from HARK.distribution import Lognormal, Uniform, expected
from HARK.interpolation import (
    BilinearInterp,
    CubicInterp,
    LinearInterp,
    LinearInterpOnInterp1D,
    LowerEnvelope2D,
    MargMargValueFuncCRRA,
    MargValueFuncCRRA,
    UpperEnvelope,
    ValueFuncCRRA,
    VariableLowerBoundFunc2D,
)
from HARK.metric import MetricObject
from HARK.rewards import (
    CRRAutility,
    CRRAutility_inv,
    CRRAutility_invP,
    CRRAutilityP,
    CRRAutilityP_inv,
    CRRAutilityP_invP,
    CRRAutilityPP,
    UtilityFuncCRRA,
)
from HARK.utilities import get_percentiles

__all__ = [
    "pLvlFuncAR1",
    "GenIncProcessConsumerType",
    "IndShockExplicitPermIncConsumerType",
    "PersistentShockConsumerType",
    "init_explicit_perm_inc",
    "init_persistent_shocks",
]

utility = CRRAutility
utilityP = CRRAutilityP
utilityPP = CRRAutilityPP
utilityP_inv = CRRAutilityP_inv
utility_invP = CRRAutility_invP
utility_inv = CRRAutility_inv
utilityP_invP = CRRAutilityP_invP


class pLvlFuncAR1(MetricObject):
    """
    A class for representing AR1-style persistent income growth functions.

    Parameters
    ----------
    pLogMean : float
        Log persistent income level toward which we are drawn.
    PermGroFac : float
        Autonomous (e.g. life cycle) pLvl growth (does not AR1 decay).
    Corr : float
        Correlation coefficient on log income.
    """

    def __init__(self, pLogMean, PermGroFac, Corr):
        self.pLogMean = pLogMean
        self.LogGroFac = np.log(PermGroFac)
        self.Corr = Corr

    def __call__(self, pLvlNow):
        """
        Returns expected persistent income level next period as a function of
        this period's persistent income level.

        Parameters
        ----------
        pLvlNow : np.array
            Array of current persistent income levels.

        Returns
        -------
        pLvlNext : np.array
            Identically shaped array of next period persistent income levels.
        """
        pLvlNext = np.exp(
            self.Corr * np.log(pLvlNow)
            + (1.0 - self.Corr) * self.pLogMean
            + self.LogGroFac
        )
        return pLvlNext


###############################################################################


def solve_one_period_ConsGenIncProcess(
    solution_next,
    IncShkDstn,
    LivPrb,
    DiscFac,
    CRRA,
    Rfree,
    pLvlNextFunc,
    BoroCnstArt,
    aXtraGrid,
    pLvlGrid,
    vFuncBool,
    CubicBool,
):
    """
    Solves one one period problem of a consumer who experiences persistent and
    transitory shocks to his income.  Unlike in ConsIndShock, consumers do not
    necessarily have the same predicted level of p next period as this period
    (after controlling for growth).  Instead, they have  a function that translates
    current persistent income into expected next period persistent income (subject
    to shocks).

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncShkDstn : distribution.Distribution
        A discrete
        approximation to the income process between the period being solved
        and the one immediately following (in solution_next). Order: event
        probabilities, persistent shocks, transitory shocks.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree : float
        Risk free interest factor on end-of-period assets.
    pLvlNextFunc : float
        Expected persistent income next period as a function of current pLvl.
    BoroCnstArt: float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.
    aXtraGrid: np.array
        Array of "extra" end-of-period (normalized) asset values-- assets
        above the absolute minimum acceptable level.
    pLvlGrid: np.array
        Array of persistent income levels at which to solve the problem.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear interpolation.

    Returns
    -------
    solution_now : ConsumerSolution
        Solution to this period's consumption-saving problem.
    """
    # Define the utility function for this period
    uFunc = UtilityFuncCRRA(CRRA)
    DiscFacEff = DiscFac * LivPrb  # "effective" discount factor

    # Unpack next period's income shock distribution
    ShkPrbsNext = IncShkDstn.pmv
    PermShkValsNext = IncShkDstn.atoms[0]
    TranShkValsNext = IncShkDstn.atoms[1]
    PermShkMinNext = np.min(PermShkValsNext)
    TranShkMinNext = np.min(TranShkValsNext)

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

    # Define the constrained consumption function as "consume all" shifted by mLvlMin
    cFuncNowCnstBase = BilinearInterp(
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
    )
    cFuncNowCnst = VariableLowerBoundFunc2D(cFuncNowCnstBase, mLvlMinNow)

    # Define grids of pLvl and aLvl on which to compute future expectations
    pLvlCount = pLvlGrid.size
    aNrmCount = aXtraGrid.size
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

    # Solve the first order condition to get optimal consumption, then find the
    # endogenous gridpoints
    cLvlNow = uFunc.derinv(EndOfPrd_vP, order=(1, 0))
    mLvlNow = cLvlNow + aLvlNow

    # Limiting consumption is zero as m approaches mNrmMin
    c_for_interpolation = np.concatenate((np.zeros((pLvlCount, 1)), cLvlNow), axis=-1)
    m_for_interpolation = np.concatenate(
        (
            BoroCnstNat(np.reshape(pLvlGrid, (pLvlCount, 1))),
            mLvlNow,
        ),
        axis=-1,
    )

    # Limiting consumption is MPCmin*mLvl as p approaches 0
    m_temp = np.reshape(m_for_interpolation[0, :], (1, m_for_interpolation.shape[1]))
    m_for_interpolation = np.concatenate((m_temp, m_for_interpolation), axis=0)
    c_for_interpolation = np.concatenate(
        (MPCminNow * m_temp, c_for_interpolation), axis=0
    )

    # Make an array of corresponding pLvl values, adding an additional column for
    # the mLvl points at the lower boundary *and* an extra row for pLvl=0.
    p_for_interpolation = np.concatenate(
        (np.reshape(pLvlGrid, (pLvlCount, 1)), pLvlNow), axis=-1
    )
    p_for_interpolation = np.concatenate(
        (np.zeros((1, m_for_interpolation.shape[1])), p_for_interpolation)
    )

    # Build the set of cFuncs by pLvl, gathered in a list
    cFunc_by_pLvl_list = []  # list of consumption functions for each pLvl
    if CubicBool:
        # Calculate end-of-period marginal marginal value of assets
        vPP_fac = DiscFacEff * Rfree * Rfree
        EndOfPrd_vPP = expected(calc_vPP_next, IncShkDstn, args=(aLvlNow, pLvlNow))
        EndOfPrd_vPP *= vPP_fac

        # Calculate the MPC at each gridpoint
        dcda = EndOfPrd_vPP / uFunc.der(np.array(c_for_interpolation[1:, 1:]), order=2)
        MPC = dcda / (dcda + 1.0)
        MPC = np.concatenate((np.reshape(MPC[:, 0], (MPC.shape[0], 1)), MPC), axis=1)

        # Stick an extra row of MPC values at pLvl=zero
        MPC = np.concatenate((MPCminNow * np.ones((1, aNrmCount + 1)), MPC), axis=0)

        # Make cubic consumption function with respect to mLvl for each persistent income level
        for j in range(p_for_interpolation.shape[0]):
            pLvl_j = p_for_interpolation[j, 0]
            m_temp = m_for_interpolation[j, :] - BoroCnstNat(pLvl_j)

            # Make a cubic consumption function for this pLvl
            c_temp = c_for_interpolation[j, :]
            MPC_temp = MPC[j, :]
            if pLvl_j > 0:
                cFunc_by_pLvl_list.append(
                    CubicInterp(
                        m_temp,
                        c_temp,
                        MPC_temp,
                        lower_extrap=True,
                        slope_limit=MPCminNow,
                        intercept_limit=MPCminNow * hLvlNow(pLvl_j),
                    )
                )
            else:  # When pLvl=0, cFunc is linear
                cFunc_by_pLvl_list.append(
                    LinearInterp(m_temp, c_temp, lower_extrap=True)
                )

    # Basic version: use linear interpolation within a pLvl
    else:
        # Loop over pLvl values and make an mLvl for each one
        for j in range(p_for_interpolation.shape[0]):
            pLvl_j = p_for_interpolation[j, 0]
            m_temp = m_for_interpolation[j, :] - BoroCnstNat(pLvl_j)

            # Make a linear consumption function for this pLvl
            c_temp = c_for_interpolation[j, :]
            if pLvl_j > 0:
                cFunc_by_pLvl_list.append(
                    LinearInterp(
                        m_temp,
                        c_temp,
                        lower_extrap=True,
                        slope_limit=MPCminNow,
                        intercept_limit=MPCminNow * hLvlNow(pLvl_j),
                    )
                )
            else:
                cFunc_by_pLvl_list.append(
                    LinearInterp(m_temp, c_temp, lower_extrap=True)
                )

    # Combine all linear cFuncs into one function
    pLvl_list = p_for_interpolation[:, 0]
    cFuncUncBase = LinearInterpOnInterp1D(cFunc_by_pLvl_list, pLvl_list)
    cFuncNowUnc = VariableLowerBoundFunc2D(cFuncUncBase, BoroCnstNat)
    # Re-adjust for lower bound of natural borrowing constraint

    # Combine the constrained and unconstrained functions into the true consumption function
    cFuncNow = LowerEnvelope2D(cFuncNowUnc, cFuncNowCnst)

    # Make the marginal value function
    vPfuncNow = MargValueFuncCRRA(cFuncNow, CRRA)

    # If using cubic spline interpolation, construct the marginal marginal value function
    if CubicBool:
        vPPfuncNow = MargMargValueFuncCRRA(cFuncNow, CRRA)
    else:
        vPPfuncNow = NullFunc()

    # If the value function has been requested, construct it now
    if vFuncBool:
        # Compute expected value and marginal value on a grid of market resources
        # Tile pLvl across m values
        pLvl_temp = np.tile(pLvlGrid, (aNrmCount, 1))
        mLvl_temp = (
            np.tile(mLvlMinNow(pLvlGrid), (aNrmCount, 1))
            + np.tile(np.reshape(aXtraGrid, (aNrmCount, 1)), (1, pLvlCount)) * pLvl_temp
        )
        cLvl_temp = cFuncNow(mLvl_temp, pLvl_temp)
        aLvl_temp = mLvl_temp - cLvl_temp
        v_temp = uFunc(cLvl_temp) + EndOfPrd_vFunc(aLvl_temp, pLvl_temp)
        vP_temp = uFunc.der(cLvl_temp)

        # Calculate pseudo-inverse value and its first derivative (wrt mLvl)
        vNvrs_temp = uFunc.inv(v_temp)  # value transformed through inverse utility
        vNvrsP_temp = vP_temp * uFunc.derinv(v_temp, order=(0, 1))

        # Add data at the lower bound of m
        mLvl_temp = np.concatenate(
            (np.reshape(mLvlMinNow(pLvlGrid), (1, pLvlCount)), mLvl_temp), axis=0
        )
        vNvrs_temp = np.concatenate((np.zeros((1, pLvlCount)), vNvrs_temp), axis=0)
        vNvrsP_temp = np.concatenate(
            (np.reshape(vNvrsP_temp[0, :], (1, vNvrsP_temp.shape[1])), vNvrsP_temp),
            axis=0,
        )

        # Add data at the lower bound of p
        MPCminNvrs = MPCminNow ** (-CRRA / (1.0 - CRRA))
        m_temp = np.reshape(mLvl_temp[:, 0], (aNrmCount + 1, 1))
        mLvl_temp = np.concatenate((m_temp, mLvl_temp), axis=1)
        vNvrs_temp = np.concatenate((MPCminNvrs * m_temp, vNvrs_temp), axis=1)
        vNvrsP_temp = np.concatenate(
            (MPCminNvrs * np.ones((aNrmCount + 1, 1)), vNvrsP_temp), axis=1
        )

        # Construct the pseudo-inverse value function
        vNvrsFunc_list = []
        for j in range(pLvlCount + 1):
            pLvl = np.insert(pLvlGrid, 0, 0.0)[j]
            vNvrsFunc_list.append(
                CubicInterp(
                    mLvl_temp[:, j] - mLvlMinNow(pLvl),
                    vNvrs_temp[:, j],
                    vNvrsP_temp[:, j],
                    MPCminNvrs * hLvlNow(pLvl),
                    MPCminNvrs,
                )
            )
        vNvrsFuncBase = LinearInterpOnInterp1D(
            vNvrsFunc_list, np.insert(pLvlGrid, 0, 0.0)
        )  # Value function "shifted"
        vNvrsFuncNow = VariableLowerBoundFunc2D(vNvrsFuncBase, mLvlMinNow)

        # "Re-curve" the pseudo-inverse value function into the value function
        vFuncNow = ValueFuncCRRA(vNvrsFuncNow, CRRA)

    else:
        vFuncNow = NullFunc()

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
    return solution_now





###############################################################################

# -----------------------------------------------------------------------------
# ----- Define additional parameters for the persistent shocks model ----------
# -----------------------------------------------------------------------------

pLvlPctiles = np.concatenate(
    (
        [0.001, 0.005, 0.01, 0.03],
        np.linspace(0.05, 0.95, num=19),
        [0.97, 0.99, 0.995, 0.999],
    )
)
PrstIncCorr = 0.98  # Serial correlation coefficient for permanent income

# Make a dictionary for the "explicit permanent income" idiosyncratic shocks model
init_explicit_perm_inc = init_idiosyncratic_shocks.copy()
init_explicit_perm_inc["pLvlPctiles"] = pLvlPctiles
init_explicit_perm_inc["pLvlInitStd"] = 0.4  # This *must* be nonzero
# long run permanent income growth doesn't work yet
init_explicit_perm_inc["PermGroFac"] = [1.0]
init_explicit_perm_inc["aXtraMax"] = 30
init_explicit_perm_inc["aXtraExtra"] = np.array([0.005, 0.01])


class GenIncProcessConsumerType(IndShockConsumerType):
    """
    A consumer type with idiosyncratic shocks to persistent and transitory income.
    His problem is defined by a sequence of income distributions, survival prob-
    abilities, and persistent income growth functions, as well as time invariant
    values for risk aversion, discount factor, the interest rate, the grid of
    end-of-period assets, and an artificial borrowing constraint.

    See init_explicit_perm_inc for a dictionary of the
    keywords that should be passed to the constructor.

    Parameters
    ----------
    cycles : int
        Number of times the sequence of periods should be solved.
    """

    cFunc_terminal_ = BilinearInterp(
        np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0.0, 1.0]), np.array([0.0, 1.0])
    )
    solution_terminal_ = ConsumerSolution(
        cFunc=cFunc_terminal_, mNrmMin=0.0, hNrm=0.0, MPCmin=1.0, MPCmax=1.0
    )

    state_vars = ["pLvl", "mLvl", "aLvl"]

    def __init__(self, **kwds):
        params = init_explicit_perm_inc.copy()
        params.update(kwds)

        # Initialize a basic ConsumerType
        IndShockConsumerType.__init__(self, **params)
        self.solve_one_period = solve_one_period_ConsGenIncProcess

        # a poststate?
        self.state_now["aLvl"] = None
        self.state_prev["aLvl"] = None

        # better way to do this...
        self.state_now["mLvl"] = None
        self.state_prev["mLvl"] = None

    def pre_solve(self):
        self.update_solution_terminal()

    def update(self):
        """
        Update the income process, the assets grid, the persistent income grid,
        and the terminal solution.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        IndShockConsumerType.update(self)
        self.update_pLvlNextFunc()
        self.update_pLvlGrid()

    def update_solution_terminal(self):
        """
        Update the terminal period solution.  This method should be run when a
        new AgentType is created or when CRRA changes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.solution_terminal.vFunc = ValueFuncCRRA(self.cFunc_terminal_, self.CRRA)
        self.solution_terminal.vPfunc = MargValueFuncCRRA(
            self.cFunc_terminal_, self.CRRA
        )
        self.solution_terminal.vPPfunc = MargMargValueFuncCRRA(
            self.cFunc_terminal_, self.CRRA
        )
        self.solution_terminal.hNrm = 0.0  # Don't track normalized human wealth
        self.solution_terminal.hLvl = lambda p: np.zeros_like(p)
        # But do track absolute human wealth by persistent income
        self.solution_terminal.mLvlMin = lambda p: np.zeros_like(p)
        # And minimum allowable market resources by perm inc

    def update_pLvlNextFunc(self):
        """
        A dummy method that creates a trivial pLvlNextFunc attribute that has
        no persistent income dynamics.  This method should be overwritten by
        subclasses in order to make (e.g.) an AR1 income process.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pLvlNextFuncBasic = LinearInterp(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        self.pLvlNextFunc = self.T_cycle * [pLvlNextFuncBasic]
        self.add_to_time_vary("pLvlNextFunc")

    def install_retirement_func(self):
        """
        Installs a special pLvlNextFunc representing retirement in the correct
        element of self.pLvlNextFunc.  Draws on the attributes T_retire and
        pLvlNextFuncRet.  If T_retire is zero or pLvlNextFuncRet does not
        exist, this method does nothing.  Should only be called from within the
        method update_pLvlNextFunc, which ensures that time is flowing forward.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if (not hasattr(self, "pLvlNextFuncRet")) or self.T_retire == 0:
            return
        t = self.T_retire
        self.pLvlNextFunc[t] = self.pLvlNextFuncRet

    def update_pLvlGrid(self):
        """
        Update the grid of persistent income levels.  Currently only works for
        infinite horizon models (cycles=0) and lifecycle models (cycles=1).  Not
        clear what to do about cycles>1 because the distribution of persistent
        income will be different within a period depending on how many cycles
        have elapsed.  This method uses a simulation approach to generate the
        pLvlGrid at each period of the cycle, drawing on the initial distribution
        of persistent income, the pLvlNextFuncs, and the attribute pLvlPctiles.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        LivPrbAll = np.array(self.LivPrb)

        # Simulate the distribution of persistent income levels by t_cycle in a lifecycle model
        if self.cycles == 1:
            pLvlNow = Lognormal(
                self.pLvlInitMean, sigma=self.pLvlInitStd, seed=31382
            ).draw(self.AgentCount)
            pLvlGrid = []  # empty list of time-varying persistent income grids
            # Calculate distribution of persistent income in each period of lifecycle
            for t in range(len(self.PermShkStd)):
                if t > 0:
                    PermShkNow = self.PermShkDstn[t - 1].draw(N=self.AgentCount)
                    pLvlNow = self.pLvlNextFunc[t - 1](pLvlNow) * PermShkNow
                pLvlGrid.append(get_percentiles(pLvlNow, percentiles=self.pLvlPctiles))

        # Calculate "stationary" distribution in infinite horizon (might vary across periods of cycle)
        elif self.cycles == 0:
            T_long = 1000  # Number of periods to simulate to get to "stationary" distribution
            pLvlNow = Lognormal(
                mu=self.pLvlInitMean, sigma=self.pLvlInitStd, seed=31382
            ).draw(self.AgentCount)
            t_cycle = np.zeros(self.AgentCount, dtype=int)
            for t in range(T_long):
                # Determine who dies and replace them with newborns
                LivPrb = LivPrbAll[t_cycle]
                draws = Uniform(seed=t).draw(self.AgentCount)
                who_dies = draws > LivPrb
                pLvlNow[who_dies] = Lognormal(
                    self.pLvlInitMean, self.pLvlInitStd, seed=t + 92615
                ).draw(np.sum(who_dies))
                t_cycle[who_dies] = 0

                for j in range(self.T_cycle):  # Update persistent income
                    these = t_cycle == j
                    PermShkTemp = self.PermShkDstn[j].draw(N=np.sum(these))
                    pLvlNow[these] = self.pLvlNextFunc[j](pLvlNow[these]) * PermShkTemp
                t_cycle = t_cycle + 1
                t_cycle[t_cycle == self.T_cycle] = 0

            # We now have a "long run stationary distribution", extract percentiles
            pLvlGrid = []  # empty list of time-varying persistent income grids
            for t in range(self.T_cycle):
                these = t_cycle == t
                pLvlGrid.append(
                    get_percentiles(pLvlNow[these], percentiles=self.pLvlPctiles)
                )

        # Throw an error if cycles>1
        else:
            assert False, "Can only handle cycles=0 or cycles=1!"

        # Store the result and add attribute to time_vary
        self.pLvlGrid = pLvlGrid
        self.add_to_time_vary("pLvlGrid")

    def sim_birth(self, which_agents):
        """
        Makes new consumers for the given indices.  Initialized variables include aNrm and pLvl, as
        well as time variables t_age and t_cycle.  Normalized assets and persistent income levels
        are drawn from lognormal distributions given by aNrmInitMean and aNrmInitStd (etc).

        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.AgentCount indicating which agents should be "born".

        Returns
        -------
        None
        """
        # Get and store states for newly born agents
        N = np.sum(which_agents)  # Number of new consumers to make
        aNrmNow_new = Lognormal(
            self.aNrmInitMean, self.aNrmInitStd, seed=self.RNG.integers(0, 2**31 - 1)
        ).draw(N)
        self.state_now["pLvl"][which_agents] = Lognormal(
            self.pLvlInitMean, self.pLvlInitStd, seed=self.RNG.integers(0, 2**31 - 1)
        ).draw(N)
        self.state_now["aLvl"][which_agents] = (
            aNrmNow_new * self.state_now["pLvl"][which_agents]
        )
        # How many periods since each agent was born
        self.t_age[which_agents] = 0
        # Which period of the cycle each agent is currently in
        self.t_cycle[which_agents] = 0

    def transition(self):
        """
        Calculates updated values of normalized market resources
        and persistent income level for each
        agent.  Uses pLvlNow, aLvlNow, PermShkNow, TranShkNow.

        Parameters
        ----------
        None

        Returns
        -------
        pLvlNow
        mLvlNow
        """
        aLvlPrev = self.state_prev["aLvl"]
        RfreeNow = self.get_Rfree()

        # Calculate new states: normalized market resources
        # and persistent income level
        pLvlNow = np.zeros_like(aLvlPrev)

        for t in range(self.T_cycle):
            these = t == self.t_cycle
            pLvlNow[these] = (
                self.pLvlNextFunc[t - 1](self.state_prev["pLvl"][these])
                * self.shocks["PermShk"][these]
            )

        # state value
        bLvlNow = RfreeNow * aLvlPrev  # Bank balances before labor income

        # Market resources after income - state value
        mLvlNow = bLvlNow + self.shocks["TranShk"] * pLvlNow

        return (pLvlNow, mLvlNow)

    def get_controls(self):
        """
        Calculates consumption for each consumer of this type using the consumption functions.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        cLvlNow = np.zeros(self.AgentCount) + np.nan
        MPCnow = np.zeros(self.AgentCount) + np.nan

        for t in range(self.T_cycle):
            these = t == self.t_cycle
            cLvlNow[these] = self.solution[t].cFunc(
                self.state_now["mLvl"][these], self.state_now["pLvl"][these]
            )
            MPCnow[these] = self.solution[t].cFunc.derivativeX(
                self.state_now["mLvl"][these], self.state_now["pLvl"][these]
            )
        self.controls["cLvl"] = cLvlNow
        self.MPCnow = MPCnow

    def get_poststates(self):
        """
        Calculates end-of-period assets for each consumer of this type.
        Identical to version in IndShockConsumerType but uses Lvl rather than Nrm variables.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.state_now["aLvl"] = self.state_now["mLvl"] - self.controls["cLvl"]
        # moves now to prev
        AgentType.get_poststates(self)


###############################################################################


class IndShockExplicitPermIncConsumerType(GenIncProcessConsumerType):
    """
    A consumer type with idiosyncratic shocks to permanent and transitory income.
    The problem is defined by a sequence of income distributions, survival prob-
    abilities, and permanent income growth rates, as well as time invariant values
    for risk aversion, discount factor, the interest rate, the grid of end-of-
    period assets, and an artificial borrowing constraint.  This agent type is
    identical to an IndShockConsumerType but for explicitly tracking pLvl as a
    state variable during solution.  There is no real economic use for it.
    """

    def update_pLvlNextFunc(self):
        """
        A method that creates the pLvlNextFunc attribute as a sequence of
        linear functions, indicating constant expected permanent income growth
        across permanent income levels.  Draws on the attribute PermGroFac, and
        installs a special retirement function when it exists.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pLvlNextFunc = []
        for t in range(self.T_cycle):
            pLvlNextFunc.append(
                LinearInterp(np.array([0.0, 1.0]), np.array([0.0, self.PermGroFac[t]]))
            )

        self.pLvlNextFunc = pLvlNextFunc
        self.add_to_time_vary("pLvlNextFunc")


###############################################################################


# Make a dictionary for the "persistent idiosyncratic shocks" model
init_persistent_shocks = init_explicit_perm_inc.copy()
init_persistent_shocks["PrstIncCorr"] = PrstIncCorr


class PersistentShockConsumerType(GenIncProcessConsumerType):
    """
    Type with idiosyncratic shocks to persistent ('Prst') and transitory income.
    The problem is defined by a sequence of income distributions, survival prob-
    abilities, and persistent income growth rates, as well as time invariant values
    for risk aversion, discount factor, the interest rate, the grid of end-of-
    period assets, an artificial borrowing constraint, and the AR1 correlation
    coefficient for (log) persistent income.

    Parameters
    ----------

    """

    def __init__(self, **kwds):
        params = init_persistent_shocks.copy()
        params.update(kwds)

        GenIncProcessConsumerType.__init__(self, **params)

    def update_pLvlNextFunc(self):
        """
        A method that creates the pLvlNextFunc attribute as a sequence of
        AR1-style functions.  Draws on the attributes PermGroFac and PrstIncCorr.
        If cycles=0, the product of PermGroFac across all periods must be 1.0,
        otherwise this method is invalid.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        pLvlNextFunc = []
        pLogMean = self.pLvlInitMean  # Initial mean (log) persistent income

        for t in range(self.T_cycle):
            pLvlNextFunc.append(
                pLvlFuncAR1(pLogMean, self.PermGroFac[t], self.PrstIncCorr)
            )
            pLogMean += np.log(self.PermGroFac[t])

        self.pLvlNextFunc = pLvlNextFunc
        self.add_to_time_vary("pLvlNextFunc")

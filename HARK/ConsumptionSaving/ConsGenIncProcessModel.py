"""
Classes to solve consumption-saving models with idiosyncratic shocks to income
in which shocks are not necessarily fully transitory or fully permanent.  Extends
ConsIndShockModel by explicitly tracking persistent income as a state variable,
and allows (log) persistent income to follow an AR1 process rather than random walk.
"""

import numpy as np

from HARK import AgentType, NullFunc
from HARK.Calibration.Income.IncomeProcesses import (
    construct_lognormal_income_process_unemployment,
    get_PermShkDstn_from_IncShkDstn,
    get_TranShkDstn_from_IncShkDstn,
    pLvlFuncAR1,
    make_trivial_pLvlNextFunc,
    make_explicit_perminc_pLvlNextFunc,
    make_AR1_style_pLvlNextFunc,
    make_pLvlGrid_by_simulation,
    make_basic_pLvlPctiles,
)
from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsumerSolution,
    IndShockConsumerType,
    make_lognormal_kNrm_init_dstn,
    make_lognormal_pLvl_init_dstn,
)
from HARK.distributions import expected
from HARK.interpolation import (
    BilinearInterp,
    ConstantFunction,
    CubicInterp,
    IdentityFunction,
    LinearInterp,
    LinearInterpOnInterp1D,
    LowerEnvelope2D,
    MargMargValueFuncCRRA,
    MargValueFuncCRRA,
    UpperEnvelope,
    ValueFuncCRRA,
    VariableLowerBoundFunc2D,
)
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
from HARK.utilities import make_assets_grid

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


###############################################################################


def make_2D_CRRA_solution_terminal(CRRA):
    """
    Construct the terminal period solution for a consumption-saving model with CRRA
    utility and two state variables: levels of market resources and permanent income.

    Parameters
    ----------
    CRRA : float
        Coefficient of relative risk aversion. This is the only relevant parameter.

    Returns
    -------
    solution_terminal : ConsumerSolution
        Terminal period solution for someone with the given CRRA.
    """
    cFunc_terminal = IdentityFunction(i_dim=0, n_dims=2)
    vFunc_terminal = ValueFuncCRRA(cFunc_terminal, CRRA)
    vPfunc_terminal = MargValueFuncCRRA(cFunc_terminal, CRRA)
    vPPfunc_terminal = MargMargValueFuncCRRA(cFunc_terminal, CRRA)
    solution_terminal = ConsumerSolution(
        cFunc=cFunc_terminal,
        vFunc=vFunc_terminal,
        vPfunc=vPfunc_terminal,
        vPPfunc=vPPfunc_terminal,
        mNrmMin=ConstantFunction(0.0),
        hNrm=ConstantFunction(0.0),
        MPCmin=1.0,
        MPCmax=1.0,
    )
    solution_terminal.hLvl = solution_terminal.hNrm
    solution_terminal.mLvlMin = solution_terminal.mNrmMin
    return solution_terminal


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
        A discrete approximation to the income shocks between the period being
        solved and the one immediately following (in solution_next).
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
        # Value function "shifted"
        vNvrsFuncBase = LinearInterpOnInterp1D(
            vNvrsFunc_list, np.insert(pLvlGrid, 0, 0.0)
        )
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

# Make a constructor dictionary for the general income process consumer type
GenIncProcessConsumerType_constructors_default = {
    "IncShkDstn": construct_lognormal_income_process_unemployment,
    "PermShkDstn": get_PermShkDstn_from_IncShkDstn,
    "TranShkDstn": get_TranShkDstn_from_IncShkDstn,
    "aXtraGrid": make_assets_grid,
    "pLvlPctiles": make_basic_pLvlPctiles,
    "pLvlGrid": make_pLvlGrid_by_simulation,
    "pLvlNextFunc": make_trivial_pLvlNextFunc,
    "solution_terminal": make_2D_CRRA_solution_terminal,
    "kNrmInitDstn": make_lognormal_kNrm_init_dstn,
    "pLvlInitDstn": make_lognormal_pLvl_init_dstn,
}

# Make a dictionary with parameters for the default constructor for kNrmInitDstn
GenIncProcessConsumerType_kNrmInitDstn_default = {
    "kLogInitMean": -12.0,  # Mean of log initial capital
    "kLogInitStd": 0.0,  # Stdev of log initial capital
    "kNrmInitCount": 15,  # Number of points in initial capital discretization
}

# Make a dictionary with parameters for the default constructor for pLvlInitDstn
GenIncProcessConsumerType_pLvlInitDstn_default = {
    "pLogInitMean": 0.0,  # Mean of log permanent income
    "pLogInitStd": 0.4,  # Stdev of log permanent income
    "pLvlInitCount": 15,  # Number of points in initial capital discretization
}

# Default parameters to make IncShkDstn using construct_lognormal_income_process_unemployment
GenIncProcessConsumerType_IncShkDstn_default = {
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
GenIncProcessConsumerType_aXtraGrid_default = {
    "aXtraMin": 0.001,  # Minimum end-of-period "assets above minimum" value
    "aXtraMax": 30,  # Maximum end-of-period "assets above minimum" value
    "aXtraNestFac": 3,  # Exponential nesting factor for aXtraGrid
    "aXtraCount": 48,  # Number of points in the grid of "assets above minimum"
    "aXtraExtra": [0.005, 0.01],  # Additional other values to add in grid (optional)
}
GenIncProcessConsumerType_pLvlNextFunc_default = {}  # Trivial function has no parameters

# Default parameters to make pLvlGrid using make_basic_pLvlPctiles
GenIncProcessConsumerType_pLvlPctiles_default = {
    "pLvlPctiles_count": 19,  # Number of points in the "body" of the grid
    "pLvlPctiles_bound": [0.05, 0.95],  # Percentile bounds of the "body"
    "pLvlPctiles_tail_count": 4,  # Number of points in each tail of the grid
    "pLvlPctiles_tail_order": np.e,  # Scaling factor for points in each tail
}

# Default parameters to make pLvlGrid using make_pLvlGrid_by_simulation
GenIncProcessConsumerType_pLvlGrid_default = {
    "pLvlExtra": None,  # Additional permanent income points to automatically add to the grid, optional
}

# Make a dictionary to specify a general income process consumer type
GenIncProcessConsumerType_solving_default = {
    # BASIC HARK PARAMETERS REQUIRED TO SOLVE THE MODEL
    "cycles": 1,  # Finite, non-cyclic model
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    "pseudo_terminal": False,  # Terminal period really does exist
    "constructors": GenIncProcessConsumerType_constructors_default,  # See dictionary above
    # PRIMITIVE RAW PARAMETERS REQUIRED TO SOLVE THE MODEL
    "CRRA": 2.0,  # Coefficient of relative risk aversion
    "Rfree": [1.03],  # Interest factor on retained assets
    "DiscFac": 0.96,  # Intertemporal discount factor
    "LivPrb": [0.98],  # Survival probability after each period
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint
    "vFuncBool": False,  # Whether to calculate the value function during solution
    "CubicBool": False,  # Whether to use cubic spline interpolation when True
    # (Uses linear spline interpolation for cFunc when False)
}
GenIncProcessConsumerType_simulation_default = {
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
GenIncProcessConsumerType_default = {}
GenIncProcessConsumerType_default.update(GenIncProcessConsumerType_kNrmInitDstn_default)
GenIncProcessConsumerType_default.update(GenIncProcessConsumerType_pLvlInitDstn_default)
GenIncProcessConsumerType_default.update(GenIncProcessConsumerType_IncShkDstn_default)
GenIncProcessConsumerType_default.update(GenIncProcessConsumerType_aXtraGrid_default)
GenIncProcessConsumerType_default.update(GenIncProcessConsumerType_pLvlNextFunc_default)
GenIncProcessConsumerType_default.update(GenIncProcessConsumerType_pLvlGrid_default)
GenIncProcessConsumerType_default.update(GenIncProcessConsumerType_pLvlPctiles_default)
GenIncProcessConsumerType_default.update(GenIncProcessConsumerType_solving_default)
GenIncProcessConsumerType_default.update(GenIncProcessConsumerType_simulation_default)
init_general_inc = GenIncProcessConsumerType_default


class GenIncProcessConsumerType(IndShockConsumerType):
    r"""
    A consumer type with idiosyncratic shocks to persistent and transitory income.
    Their problem is defined by a sequence of income distributions, survival prob-
    abilities, and persistent income growth functions, as well as time invariant
    values for risk aversion, discount factor, the interest rate, the grid of
    end-of-period assets, and an artificial borrowing constraint.

    .. math::
        \begin{eqnarray*}
        V_t(M_t,P_t) &=& \max_{C_t} U(C_t) + \beta \mathsf{S}_{t} \mathbb{E} [V_{t+1}(M_{t+1}, P_{t+1}) ], \\
        A_t &=& M_t - C_t, \\
        A_t/P_t &\geq& \underline{a}, \\
        M_{t+1} &=& R_{t+1} A_t + \theta_{t+1}, \\
        P_{t+1} &=& G_{t+1}(P_t)\psi_{t+1}, \\
        (\psi_{t+1},\theta_{t+1}) &\sim& F_{t+1}, \\
        \mathbb{E} [\psi_{t+1}] &=& 1, \\
        U(C) &=& \frac{C^{1-\rho}}{1-\rho}. \\
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

    Solving Parameters
    ------------------
    cycles: int
        0 specifies an infinite horizon model, 1 specifies a finite model.
    T_cycle: int
        Number of periods in the cycle for this agent type.
    CRRA: float, :math:`\rho`
        Coefficient of Relative Risk Aversion.
    Rfree: float or list[float], time varying, :math:`\mathsf{R}_t`
        Risk Free interest rate. Pass a list of floats to make Rfree time varying.
    DiscFac: float, :math:`\beta`
        Intertemporal discount factor.
    LivPrb: list[float], time varying, :math:`\mathsf{S}_t`
        Survival probability after each period.
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
        For this agent, the options are 'PermShk', 'TranShk', 'aLvl', 'cLvl', 'mLvl', 'pLvl', and 'who_dies'.

        PermShk is the agent's permanent income shock

        TranShk is the agent's transitory income shock

        aLvl is the nominal asset level

        cLvl is the nominal consumption level

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
        The solution functions also depend on the permanent income level. For example, :math:`C=\text{cFunc}(M,P)`.
        hNrm has been replaced by hLvl which is a function of permanent income.
        MPC max has not yet been implemented for this class. It will be a function of permanent income.

        Visit :class:`HARK.ConsumptionSaving.ConsIndShockModel.ConsumerSolution` for more information about the solution.

    history: Dict[Array]
        Created by running the :func:`.simulate()` method.
        Contains the variables in track_vars. Each item in the dictionary is an array with the shape (T_sim,AgentCount).
        Visit :class:`HARK.core.AgentType.simulate` for more information.
    """

    IncShkDstn_default = GenIncProcessConsumerType_IncShkDstn_default
    aXtraGrid_default = GenIncProcessConsumerType_aXtraGrid_default
    pLvlNextFunc_default = GenIncProcessConsumerType_pLvlNextFunc_default
    pLvlGrid_default = GenIncProcessConsumerType_pLvlGrid_default
    pLvlPctiles_default = GenIncProcessConsumerType_pLvlPctiles_default
    solving_default = GenIncProcessConsumerType_solving_default
    simulation_default = GenIncProcessConsumerType_simulation_default

    state_vars = ["kLvl", "pLvl", "mLvl", "aLvl", "aNrm"]
    time_vary_ = IndShockConsumerType.time_vary_ + ["pLvlNextFunc", "pLvlGrid"]
    default_ = {
        "params": GenIncProcessConsumerType_default,
        "solver": solve_one_period_ConsGenIncProcess,
        "model": "ConsGenIncProcess.yaml",
    }

    def pre_solve(self):
        self.construct("solution_terminal")

    def install_retirement_func(self):
        """
        Installs a special pLvlNextFunc representing retirement in the correct
        element of self.pLvlNextFunc.  Draws on the attributes T_retire and
        pLvlNextFuncRet.  If T_retire is zero or pLvlNextFuncRet does not
        exist, this method does nothing.

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
        super().sim_birth(which_agents)
        self.state_now["aLvl"][which_agents] = (
            self.state_now["aNrm"][which_agents] * self.state_now["pLvl"][which_agents]
        )

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
        kLvlNow
        pLvlNow
        mLvlNow
        """
        kLvlNow = self.state_prev["aLvl"]
        pLvlNow = np.zeros_like(kLvlNow)
        RportNow = self.get_Rport()

        # Calculate new states: normalized market resources and persistent income level
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            pLvlNow[these] = (
                self.pLvlNextFunc[t - 1](self.state_prev["pLvl"][these])
                * self.shocks["PermShk"][these]
            )

        # state value
        bLvlNow = RportNow * kLvlNow  # Bank balances before labor income

        # Market resources after income - state value
        mLvlNow = bLvlNow + self.shocks["TranShk"] * pLvlNow

        return (kLvlNow, pLvlNow, mLvlNow)

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

    def check_conditions(self, verbose=None):
        raise NotImplementedError()  # pragma: nocover

    def calc_limiting_values(self):
        raise NotImplementedError()  # pragma: nocover


###############################################################################

# Make a dictionary for the "explicit permanent income" consumer type; see parent dictionary above.
IndShockExplicitPermIncConsumerType_constructors_default = (
    GenIncProcessConsumerType_constructors_default.copy()
)
IndShockExplicitPermIncConsumerType_constructors_default["pLvlNextFunc"] = (
    make_explicit_perminc_pLvlNextFunc
)
IndShockExplicitPermIncConsumerType_IncShkDstn_default = (
    GenIncProcessConsumerType_IncShkDstn_default.copy()
)
IndShockExplicitPermIncConsumerType_kNrmInitDstn_default = (
    GenIncProcessConsumerType_kNrmInitDstn_default.copy()
)
IndShockExplicitPermIncConsumerType_pLvlInitDstn_default = (
    GenIncProcessConsumerType_pLvlInitDstn_default.copy()
)
IndShockExplicitPermIncConsumerType_aXtraGrid_default = (
    GenIncProcessConsumerType_aXtraGrid_default.copy()
)
IndShockExplicitPermIncConsumerType_pLvlNextFunc_default = (
    GenIncProcessConsumerType_pLvlNextFunc_default.copy()
)
IndShockExplicitPermIncConsumerType_pLvlGrid_default = (
    GenIncProcessConsumerType_pLvlGrid_default.copy()
)
IndShockExplicitPermIncConsumerType_pLvlPctiles_default = (
    GenIncProcessConsumerType_pLvlPctiles_default.copy()
)
IndShockExplicitPermIncConsumerType_solving_default = (
    GenIncProcessConsumerType_solving_default.copy()
)
IndShockExplicitPermIncConsumerType_solving_default["constructors"] = (
    IndShockExplicitPermIncConsumerType_constructors_default
)
IndShockExplicitPermIncConsumerType_pLvlNextFunc_default["PermGroFac"] = [1.0]
IndShockExplicitPermIncConsumerType_simulation_default = (
    GenIncProcessConsumerType_simulation_default.copy()
)

IndShockExplicitPermIncConsumerType_default = {}
IndShockExplicitPermIncConsumerType_default.update(
    IndShockExplicitPermIncConsumerType_IncShkDstn_default
)
IndShockExplicitPermIncConsumerType_default.update(
    IndShockExplicitPermIncConsumerType_kNrmInitDstn_default
)
IndShockExplicitPermIncConsumerType_default.update(
    IndShockExplicitPermIncConsumerType_pLvlInitDstn_default
)
IndShockExplicitPermIncConsumerType_default.update(
    IndShockExplicitPermIncConsumerType_aXtraGrid_default
)
IndShockExplicitPermIncConsumerType_default.update(
    IndShockExplicitPermIncConsumerType_pLvlNextFunc_default
)
IndShockExplicitPermIncConsumerType_default.update(
    IndShockExplicitPermIncConsumerType_pLvlGrid_default
)
IndShockExplicitPermIncConsumerType_default.update(
    IndShockExplicitPermIncConsumerType_pLvlPctiles_default
)
IndShockExplicitPermIncConsumerType_default.update(
    IndShockExplicitPermIncConsumerType_solving_default
)
IndShockExplicitPermIncConsumerType_default.update(
    IndShockExplicitPermIncConsumerType_simulation_default
)
init_explicit_perm_inc = IndShockExplicitPermIncConsumerType_default

# NB: Permanent income growth was not in the default dictionary for GenIncProcessConsumerType
# because its pLvlNextFunc constructor was *trivial*: no permanent income dynamics at all!
# For the "explicit permanent income" model, this parameter is added back into the dictionary.
# However, note that if this model is used in an *infinite horizon* setting, it will work
# best if the product of PermGroFac (across all periods) is 1. If it is far from 1, then the
# pLvlGrid that is constructed by the default method might not be appropriate.


class IndShockExplicitPermIncConsumerType(GenIncProcessConsumerType):
    r"""
    A consumer type based on GenIncProcessModel, where the general function
    describing the path of permanent income multiplies the current permanent
    income by the PermGroFac (:math:`\Gamma`). It's behavior is the same as
    :class:`HARK.ConsumptionSaving.ConsIndShockModel.IndShockConsumerType`, except
    that the variables aren't normalized. This makes the result less
    accurate. This Model uses a lognormal random walk income process.
    If you would like to use a different income process, use
    :class:`HARK.ConsumptionSaving.ConsGenIncProcessModel.GenIncProcessConsumerType`

    .. math::
        \begin{eqnarray*}
        V_t(M_t,P_t) &=& \max_{C_t} U(C_t) + \beta (1-\mathsf{D}_{t+1}) \mathbb{E} [V_{t+1}(M_{t+1}, P_{t+1}) ], \\
        A_t &=& M_t - C_t, \\
        A_t/P_t &\geq& \underline{a}, \\
        M_{t+1} &=& R A_t + \theta_{t+1}, \\
        P_{t+1} &=& G_{t+1}(P_t)\psi_{t+1}, \\
        (\psi_{t+1},\theta_{t+1}) &\sim& F_{t+1}, \\
        \mathbb{E} [F_{t+1}] &=& 1, \\
        U(C) &=& \frac{C^{1-\rho}}{1-\rho}. \\
        G_{t+1} (x) &=&\Gamma_{t+1} x
        \end{eqnarray*}


    Constructors
    ------------
    IncShkDstn: Constructor, :math:`\psi`, :math:`\theta`
        The agent's income shock distributions.

        Its default constructor is :func:`HARK.Calibration.Income.IncomeProcesses.construct_lognormal_income_process_unemployment`
    aXtraGrid: Constructor
        The agent's asset grid.

        Its default constructor is :func:`HARK.utilities.make_assets_grid`
    pLvlNextFunc: Constructor, (:math:`\Gamma`)
        An arbitrary function used to evolve the GenIncShockConsumerType's permanent income

        Its default constructor is :func:`HARK.Calibration.Income.IncomeProcesses.make_explicit_perminc_pLvlNextFunc`
    pLvlGrid: Constructor
        The agent's pLvl grid

        Its default constructor is :func:`HARK.Calibration.Income.IncomeProcesses.make_pLvlGrid_by_simulation`
    pLvlPctiles: Constructor
        The agents income level percentile grid

        Its default constructor is :func:`HARK.Calibration.Income.IncomeProcesses.make_basic_pLvlPctiles`

    Solving Parameters
    ------------------
    cycles: int
        0 specifies an infinite horizon model, 1 specifies a finite model.
    T_cycle: int
        Number of periods in the cycle for this agent type.
    CRRA: float, :math:`\rho`
        Coefficient of Relative Risk Aversion.
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
        For this agent, the options are 'PermShk', 'TranShk', 'aLvl', 'cLvl', 'mLvl', 'pLvl', and 'who_dies'.

        PermShk is the agent's permanent income shock

        TranShk is the agent's transitory income shock

        aLvl is the nominal asset level

        cLvl is the nominal consumption level

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
        The solution functions also depend on the permanent income level. For example, :math:`C=\text{cFunc}(M,P)`.
        hNrm has been replaced by hLvl which is a function of permanent income.
        MPC max has not yet been implemented for this class. It will be a function of permanent income.

        Visit :class:`HARK.ConsumptionSaving.ConsIndShockModel.ConsumerSolution` for more information about the solution.

    history: Dict[Array]
        Created by running the :func:`.simulate()` method.
        Contains the variables in track_vars. Each item in the dictionary is an array with the shape (T_sim,AgentCount).
        Visit :class:`HARK.core.AgentType.simulate` for more information.
    """

    default_ = {
        "params": init_explicit_perm_inc,
        "solver": solve_one_period_ConsGenIncProcess,
        "model": "ConsGenIncProcess.yaml",
    }


###############################################################################

# Make a dictionary for the "persistent idiosyncratic shocks" consumer type; see parent dictionary above.

PersistentShockConsumerType_constructors_default = (
    GenIncProcessConsumerType_constructors_default.copy()
)
PersistentShockConsumerType_constructors_default["pLvlNextFunc"] = (
    make_AR1_style_pLvlNextFunc
)
PersistentShockConsumerType_kNrmInitDstn_default = (
    IndShockExplicitPermIncConsumerType_kNrmInitDstn_default.copy()
)
PersistentShockConsumerType_pLvlInitDstn_default = (
    IndShockExplicitPermIncConsumerType_pLvlInitDstn_default.copy()
)
PersistentShockConsumerType_IncShkDstn_default = (
    IndShockExplicitPermIncConsumerType_IncShkDstn_default.copy()
)
PersistentShockConsumerType_aXtraGrid_default = (
    IndShockExplicitPermIncConsumerType_aXtraGrid_default.copy()
)
PersistentShockConsumerType_pLvlNextFunc_default = (
    IndShockExplicitPermIncConsumerType_pLvlNextFunc_default.copy()
)
PersistentShockConsumerType_pLvlGrid_default = (
    IndShockExplicitPermIncConsumerType_pLvlGrid_default.copy()
)
PersistentShockConsumerType_pLvlPctiles_default = (
    IndShockExplicitPermIncConsumerType_pLvlPctiles_default.copy()
)
PersistentShockConsumerType_solving_default = (
    IndShockExplicitPermIncConsumerType_solving_default.copy()
)
PersistentShockConsumerType_solving_default["constructors"] = (
    PersistentShockConsumerType_constructors_default
)
PersistentShockConsumerType_pLvlNextFunc_default["PrstIncCorr"] = 0.98
PersistentShockConsumerType_simulation_default = (
    IndShockExplicitPermIncConsumerType_simulation_default.copy()
)

PersistentShockConsumerType_default = {}
PersistentShockConsumerType_default.update(
    PersistentShockConsumerType_IncShkDstn_default
)
PersistentShockConsumerType_default.update(
    PersistentShockConsumerType_kNrmInitDstn_default
)
PersistentShockConsumerType_default.update(
    PersistentShockConsumerType_pLvlInitDstn_default
)
PersistentShockConsumerType_default.update(
    PersistentShockConsumerType_aXtraGrid_default
)
PersistentShockConsumerType_default.update(
    PersistentShockConsumerType_pLvlNextFunc_default
)
PersistentShockConsumerType_default.update(PersistentShockConsumerType_pLvlGrid_default)
PersistentShockConsumerType_default.update(
    PersistentShockConsumerType_pLvlPctiles_default
)
PersistentShockConsumerType_default.update(PersistentShockConsumerType_solving_default)
PersistentShockConsumerType_default.update(
    PersistentShockConsumerType_simulation_default
)
init_persistent_shocks = PersistentShockConsumerType_default


class PersistentShockConsumerType(GenIncProcessConsumerType):
    r"""
    A consumer type based on GenIncProcessModel, where the log permanent income follows an AR1 process.
    If you would like to use a different income process, use
    :class:`HARK.ConsumptionSaving.ConsGenIncProcessModel.GenIncProcessConsumerType`

    .. math::
        \begin{eqnarray*}
        V_t(M_t,P_t) &=& \max_{C_t} U(C_t) + \beta (1-\mathsf{D}_{t+1}) \mathbb{E} [V_{t+1}(M_{t+1}, P_{t+1}) ], \\
        A_t &=& M_t - C_t, \\
        A_t/P_t &\geq& \underline{a}, \\
        M_{t+1} &=& R A_t + \theta_{t+1}, \\
        p_{t+1} &=& G_{t+1}(P_t)\psi_{t+1}, \\
        (\psi_{t+1},\theta_{t+1}) &\sim& F_{t+1}, \\
        \mathbb{E} [F_{t+1}] &=& 1, \\
        U(C) &=& \frac{C^{1-\rho}}{1-\rho}, \\
        log(G_{t+1} (x)) &=&\varphi log(x) + (1-\varphi) log(\overline{P}_{t})+log(\Gamma_{t+1}) + log(\psi_{t+1}), \\
        \overline{P}_{t+1} &=& \overline{P}_{t} \Gamma_{t+1} \\
        \end{eqnarray*}

    Constructors
    ------------
    IncShkDstn: Constructor, :math:`\psi`, :math:`\theta`
        The agent's income shock distributions.
        Its default constructor is :func:`HARK.Calibration.Income.IncomeProcesses.construct_lognormal_income_process_unemployment`
    aXtraGrid: Constructor
        The agent's asset grid.
        Its default constructor is :func:`HARK.utilities.make_assets_grid`
    pLvlNextFunc: Constructor, (:math:`\Gamma`, :math:`\varphi`)
        An arbitrary function used to evolve the GenIncShockConsumerType's permanent income
        Its default constructor is :func:`HARK.Calibration.Income.IncomeProcesses.make_AR1_style_pLvlNextFunc`
    pLvlGrid: Constructor
        The agent's pLvl grid
        Its default constructor is :func:`HARK.Calibration.Income.IncomeProcesses.make_pLvlGrid_by_simulation`
    pLvlPctiles: Constructor
        The agents income level percentile grid
        Its default constructor is :func:`HARK.Calibration.Income.IncomeProcesses.make_basic_pLvlPctiles`

    Solving Parameters
    ------------------
    cycles: int
        0 specifies an infinite horizon model, 1 specifies a finite model.
    T_cycle: int
        Number of periods in the cycle for this agent type.
    CRRA: float, :math:`\rho`
        Coefficient of Relative Risk Aversion.
    Rfree: list[float], time varying, :math:`\mathsf{R}`
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
        For this agent, the options are 'PermShk', 'TranShk', 'aLvl', 'cLvl', 'mLvl', 'pLvl', and 'who_dies'.

        PermShk is the agent's permanent income shock

        TranShk is the agent's transitory income shock

        aLvl is the nominal asset level

        cLvl is the nominal consumption level

        mLvl is the nominal market resources

        pLvl is the permanent income level

        who_dies is the array of which agents died
    kLogInitMean: float
        Mean of Log initial Normalized Assets.
    kLogInitStd: float
        Std of Log initial Normalized Assets.
    pLogInitMean: float
        Mean of Log initial permanent income.
    pLogInitStd: float
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
        The solution functions also depend on the permanent income level. For example, :math:`C=\text{cFunc}(M,P)`.
        hNrm has been replaced by hLvl which is a function of permanent income.
        MPC max has not yet been implemented for this class. It will be a function of permanent income.

        Visit :class:`HARK.ConsumptionSaving.ConsIndShockModel.ConsumerSolution` for more information about the solution.

    history: Dict[Array]
        Created by running the :func:`.simulate()` method.
        Contains the variables in track_vars. Each item in the dictionary is an array with the shape (T_sim,AgentCount).
        Visit :class:`HARK.core.AgentType.simulate` for more information.
    """

    default_ = {
        "params": init_persistent_shocks,
        "solver": solve_one_period_ConsGenIncProcess,
        "model": "ConsGenIncProcess.yaml",
    }

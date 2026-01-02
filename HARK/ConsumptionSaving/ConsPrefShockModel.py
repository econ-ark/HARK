"""
Extensions to ConsIndShockModel concerning models with preference shocks.
It currently only two models:

1) An extension of ConsIndShock, but with an iid lognormal multiplicative shock each period.
2) A combination of (1) and ConsKinkedR, demonstrating how to construct a new model
   by inheriting from multiple classes.
"""

import numpy as np

from HARK import NullFunc
from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsumerSolution,
    IndShockConsumerType,
    KinkedRconsumerType,
    make_assets_grid,
    make_lognormal_kNrm_init_dstn,
    make_lognormal_pLvl_init_dstn,
)
from HARK.Calibration.Income.IncomeProcesses import (
    construct_lognormal_income_process_unemployment,
    get_PermShkDstn_from_IncShkDstn,
    get_TranShkDstn_from_IncShkDstn,
)
from HARK.distributions import MeanOneLogNormal, expected
from HARK.interpolation import (
    IdentityFunction,
    CubicInterp,
    LinearInterp,
    LinearInterpOnInterp1D,
    LowerEnvelope,
    MargValueFuncCRRA,
    MargMargValueFuncCRRA,
    ValueFuncCRRA,
)
from HARK.rewards import UtilityFuncCRRA

__all__ = [
    "PrefShockConsumerType",
    "KinkyPrefConsumerType",
    "make_lognormal_PrefShkDstn",
]


def make_pref_shock_solution_terminal(CRRA):
    """
    Construct the terminal period solution for a consumption-saving model with
    CRRA utility and two state variables. The consumption function depends *only*
    on the first dimension, representing market resources.

    Parameters
    ----------
    CRRA : float
        Coefficient of relative risk aversion. This is the only relevant parameter.

    Returns
    -------
    solution_terminal : ConsumerSolution
        Terminal period solution for someone with the given CRRA.
    """
    cFunc_terminal = IdentityFunction(i_dim=0, n_dims=2)  # c=m at t=T
    vFunc_terminal = ValueFuncCRRA(cFunc_terminal, CRRA)
    vPfunc_terminal = MargValueFuncCRRA(cFunc_terminal, CRRA)
    vPPfunc_terminal = MargMargValueFuncCRRA(cFunc_terminal, CRRA)
    solution_terminal = ConsumerSolution(
        cFunc=cFunc_terminal,
        vFunc=vFunc_terminal,
        vPfunc=vPfunc_terminal,
        vPPfunc=vPPfunc_terminal,
        mNrmMin=0.0,
        hNrm=0.0,
        MPCmin=1.0,
        MPCmax=1.0,
    )
    return solution_terminal


def make_lognormal_PrefShkDstn(
    T_cycle,
    PrefShkStd,
    PrefShkCount,
    RNG,
    PrefShk_tail_N=0,
    PrefShk_tail_order=np.e,
    PrefShk_tail_bound=[0.02, 0.98],
):
    r"""
    Make a discretized mean one lognormal preference shock distribution for each
    period of the agent's problem.

    .. math::
        \eta_t \sim \mathcal{N}(-\textbf{PrefShkStd}_{t}^{2}/2,\textbf{PrefShkStd}_{t}^2)

    Parameters
    ----------
    T_cycle : int
        Number of non-terminal periods in the agent's cycle.
    PrefShkStd : [float]
        Standard deviation of log preference shocks in each period.
    PrefShkCount : int
        Number of equiprobable preference shock nodes in the "body" of the distribution.
    RNG : RandomState
        The AgentType's internal random number generator.
    PrefShk_tail_N : int
        Number of shock nodes in each "tail" of the distribution (optional).
    PrefShk_tail_order : float
        Scaling factor for tail nodes (optional).
    PrefShk_tail_bound : [float,float]
        CDF bounds for tail nodes (optional).

    Returns
    -------
    PrefShkDstn : [DiscreteDistribution]
        List of discretized lognormal distributions for shocks.
    """
    PrefShkDstn = []  # discrete distributions of preference shocks
    for t in range(T_cycle):
        PrefShkStd = PrefShkStd[t]
        new_dstn = MeanOneLogNormal(
            sigma=PrefShkStd, seed=RNG.integers(0, 2**31 - 1)
        ).discretize(
            N=PrefShkCount,
            method="equiprobable",
            tail_N=PrefShk_tail_N,
            tail_order=PrefShk_tail_order,
            tail_bound=PrefShk_tail_bound,
        )
        PrefShkDstn.append(new_dstn)
    return PrefShkDstn


###############################################################################


def solve_one_period_ConsPrefShock(
    solution_next,
    IncShkDstn,
    PrefShkDstn,
    LivPrb,
    DiscFac,
    CRRA,
    Rfree,
    PermGroFac,
    BoroCnstArt,
    aXtraGrid,
    vFuncBool,
    CubicBool,
):
    """
    Solves one period of a consumption-saving model with idiosyncratic shocks to
    permanent and transitory income, with one risk free asset and CRRA utility.
    The consumer also faces iid preference shocks as a multiplicative shifter to
    their marginal utility of consumption.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to the succeeding one period problem.
    IncShkDstn : distribution.Distribution
        A discrete approximation to the income process between the period being
        solved and the one immediately following (in solution_next). Order:
        permanent shocks, transitory shocks.
    PrefShkDstn : distribution.Distribution
        Discrete distribution of the multiplicative utility shifter.  Order:
        probabilities, preference shocks.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree : float
        Risk free interest factor on end-of-period assets.
    PermGroGac : float
        Expected permanent income growth factor at the end of this period.
    BoroCnstArt: float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.  If it is less than the natural borrowing constraint,
        then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
        rowing constraint.
    aXtraGrid: np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear inter-
        polation.

    Returns
    -------
    solution: ConsumerSolution
        The solution to the single period consumption-saving problem.  Includes
        a consumption function cFunc (using linear splines), a marginal value
        function vPfunc, a minimum acceptable level of normalized market re-
        sources mNrmMin, normalized human wealth hNrm, and bounding MPCs MPCmin
        and MPCmax.  It might also have a value function vFunc.  The consumption
        function is defined over normalized market resources and the preference
        shock, c = cFunc(m,PrefShk), but the (marginal) value function is defined
        unconditionally on the shock, just before it is revealed.
    """
    # Define the current period utility function and effective discount factor
    uFunc = UtilityFuncCRRA(CRRA)
    DiscFacEff = DiscFac * LivPrb  # "effective" discount factor

    # Unpack next period's income and preference shock distributions
    ShkPrbsNext = IncShkDstn.pmv
    PermShkValsNext = IncShkDstn.atoms[0]
    TranShkValsNext = IncShkDstn.atoms[1]
    PermShkMinNext = np.min(PermShkValsNext)
    TranShkMinNext = np.min(TranShkValsNext)
    PrefShkPrbs = PrefShkDstn.pmv
    PrefShkVals = PrefShkDstn.atoms.flatten()

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
    Ex_IncNext = np.dot(ShkPrbsNext, TranShkValsNext * PermShkValsNext)
    hNrmNow = PermGroFac / Rfree * (Ex_IncNext + solution_next.hNrm)
    temp_fac = (WorstIncPrb ** (1.0 / CRRA)) * PatFac
    MPCmaxNow = 1.0 / (1.0 + temp_fac / solution_next.MPCmax)

    # Calculate the minimum allowable value of money resources in this period
    PermGroFacEffMin = (PermGroFac * PermShkMinNext) / Rfree
    BoroCnstNat = (solution_next.mNrmMin - TranShkMinNext) * PermGroFacEffMin

    # Set the minimum allowable (normalized) market resources based on the natural
    # and artificial borrowing constraints
    if BoroCnstArt is None:
        mNrmMinNow = BoroCnstNat
    else:
        mNrmMinNow = np.max([BoroCnstNat, BoroCnstArt])

    # Set the upper limit of the MPC (at mNrmMinNow) based on whether the natural
    # or artificial borrowing constraint actually binds
    if BoroCnstNat < mNrmMinNow:
        MPCmaxEff = 1.0  # If actually constrained, MPC near limit is 1
    else:
        MPCmaxEff = MPCmaxNow  # Otherwise, it's the MPC calculated above

    # Define the borrowing-constrained consumption function
    cFuncNowCnst = LinearInterp(
        np.array([mNrmMinNow, mNrmMinNow + 1.0]), np.array([0.0, 1.0])
    )

    # Construct the assets grid by adjusting aXtra by the natural borrowing constraint
    aNrmNow = np.asarray(aXtraGrid) + BoroCnstNat

    # Define local functions for taking future expectations
    def calc_mNrmNext(S, a, R):
        return R / (PermGroFac * S["PermShk"]) * a + S["TranShk"]

    def calc_vNext(S, a, R):
        return (S["PermShk"] ** (1.0 - CRRA) * PermGroFac ** (1.0 - CRRA)) * vFuncNext(
            calc_mNrmNext(S, a, R)
        )

    def calc_vPnext(S, a, R):
        return S["PermShk"] ** (-CRRA) * vPfuncNext(calc_mNrmNext(S, a, R))

    def calc_vPPnext(S, a, R):
        return S["PermShk"] ** (-CRRA - 1.0) * vPPfuncNext(calc_mNrmNext(S, a, R))

    # Calculate end-of-period marginal value of assets at each gridpoint
    vPfacEff = DiscFacEff * Rfree * PermGroFac ** (-CRRA)
    EndOfPrdvP = vPfacEff * expected(calc_vPnext, IncShkDstn, args=(aNrmNow, Rfree))

    # Find optimal consumption corresponding to each aNrm, PrefShk combination
    cNrm_base = uFunc.derinv(EndOfPrdvP, order=(1, 0))
    PrefShkCount = PrefShkVals.size
    PrefShk_temp = np.tile(
        np.reshape(PrefShkVals ** (1.0 / CRRA), (PrefShkCount, 1)),
        (1, cNrm_base.size),
    )
    cNrmNow = np.tile(cNrm_base, (PrefShkCount, 1)) * PrefShk_temp
    mNrmNow = cNrmNow + np.tile(aNrmNow, (PrefShkCount, 1))
    # These are the endogenous gridpoints, as usual

    # Add the bottom point to the c and m arrays
    m_for_interpolation = np.concatenate(
        (BoroCnstNat * np.ones((PrefShkCount, 1)), mNrmNow), axis=1
    )
    c_for_interpolation = np.concatenate((np.zeros((PrefShkCount, 1)), cNrmNow), axis=1)

    # Construct the consumption function as a cubic or linear spline interpolation
    # for each value of PrefShk, interpolated across those values.
    if CubicBool:
        # This is not yet supported, not sure why we never got to it
        raise ValueError(
            "Cubic interpolation is not yet supported by the preference shock model!"
        )

    # Make the preference-shock specific consumption functions
    cFuncs_by_PrefShk = []
    for j in range(PrefShkCount):
        MPCmin_j = MPCminNow * PrefShkVals[j] ** (1.0 / CRRA)
        cFunc_this_shk = LowerEnvelope(
            LinearInterp(
                m_for_interpolation[j, :],
                c_for_interpolation[j, :],
                intercept_limit=hNrmNow * MPCmin_j,
                slope_limit=MPCmin_j,
            ),
            cFuncNowCnst,
        )
        cFuncs_by_PrefShk.append(cFunc_this_shk)

    # Combine the list of consumption functions into a single interpolation
    cFuncNow = LinearInterpOnInterp1D(cFuncs_by_PrefShk, PrefShkVals)

    # Make the ex ante marginal value function (before the preference shock)
    m_grid = aXtraGrid + mNrmMinNow
    vP_vec = np.zeros_like(m_grid)
    for j in range(PrefShkCount):  # numeric integration over the preference shock
        vP_vec += (
            uFunc.der(cFuncs_by_PrefShk[j](m_grid)) * PrefShkPrbs[j] * PrefShkVals[j]
        )
    vPnvrs_vec = uFunc.derinv(vP_vec, order=(1, 0))
    vPfuncNow = MargValueFuncCRRA(LinearInterp(m_grid, vPnvrs_vec), CRRA)

    # Define this period's marginal marginal value function
    vPPfuncNow = NullFunc()  # Dummy object, cubic interpolation not implemented

    # Construct this period's value function if requested
    if vFuncBool:
        # Calculate end-of-period value, its derivative, and their pseudo-inverse
        EndOfPrdv = DiscFacEff * expected(calc_vNext, IncShkDstn, args=(aNrmNow, Rfree))
        EndOfPrdvNvrs = uFunc.inv(
            EndOfPrdv
        )  # value transformed through inverse utility
        EndOfPrdvNvrsP = EndOfPrdvP * uFunc.derinv(EndOfPrdv, order=(0, 1))
        EndOfPrdvNvrs = np.insert(EndOfPrdvNvrs, 0, 0.0)
        EndOfPrdvNvrsP = np.insert(EndOfPrdvNvrsP, 0, EndOfPrdvNvrsP[0])
        # This is a very good approximation, vNvrsPP = 0 at the asset minimum

        # Construct the end-of-period value function
        aNrm_temp = np.insert(aNrmNow, 0, BoroCnstNat)
        EndOfPrd_vNvrsFunc = CubicInterp(aNrm_temp, EndOfPrdvNvrs, EndOfPrdvNvrsP)
        EndOfPrd_vFunc = ValueFuncCRRA(EndOfPrd_vNvrsFunc, CRRA)

        # Compute expected value and marginal value on a grid of market resources,
        # accounting for all of the discrete preference shocks
        mNrm_temp = mNrmMinNow + aXtraGrid
        v_temp = np.zeros_like(mNrm_temp)
        vP_temp = np.zeros_like(mNrm_temp)
        for j in range(PrefShkCount):
            this_shock = PrefShkVals[j]
            this_prob = PrefShkPrbs[j]
            cNrm_temp = cFuncNow(mNrm_temp, this_shock * np.ones_like(mNrm_temp))
            aNrm_temp = mNrm_temp - cNrm_temp
            v_temp += this_prob * (
                this_shock * uFunc(cNrm_temp) + EndOfPrd_vFunc(aNrm_temp)
            )
            vP_temp += this_prob * this_shock * uFunc.der(cNrm_temp)

        # Construct the beginning-of-period value function
        # value transformed through inverse utility
        vNvrs_temp = uFunc.inv(v_temp)
        vNvrsP_temp = vP_temp * uFunc.derinv(v_temp, order=(0, 1))
        mNrm_temp = np.insert(mNrm_temp, 0, mNrmMinNow)
        vNvrs_temp = np.insert(vNvrs_temp, 0, 0.0)
        vNvrsP_temp = np.insert(vNvrsP_temp, 0, MPCmaxEff ** (-CRRA / (1.0 - CRRA)))
        MPCminNvrs = MPCminNow ** (-CRRA / (1.0 - CRRA))
        vNvrsFuncNow = CubicInterp(
            mNrm_temp, vNvrs_temp, vNvrsP_temp, MPCminNvrs * hNrmNow, MPCminNvrs
        )
        vFuncNow = ValueFuncCRRA(vNvrsFuncNow, CRRA)

    else:
        vFuncNow = NullFunc()  # Dummy object

    # Create and return this period's solution
    solution_now = ConsumerSolution(
        cFunc=cFuncNow,
        vFunc=vFuncNow,
        vPfunc=vPfuncNow,
        vPPfunc=vPPfuncNow,
        mNrmMin=mNrmMinNow,
        hNrm=hNrmNow,
        MPCmin=MPCminNow,
        MPCmax=MPCmaxEff,
    )
    return solution_now


###############################################################################


def solve_one_period_ConsKinkyPref(
    solution_next,
    IncShkDstn,
    PrefShkDstn,
    LivPrb,
    DiscFac,
    CRRA,
    Rboro,
    Rsave,
    PermGroFac,
    BoroCnstArt,
    aXtraGrid,
    vFuncBool,
    CubicBool,
):
    """
    Solves one period of a consumption-saving model with idiosyncratic shocks to
    permanent and transitory income, with a risk free asset and CRRA utility.
    In this variation, the interest rate on borrowing Rboro exceeds the interest
    rate on saving Rsave. The consumer also faces iid preference shocks as a multi-
    plicative shifter to their marginal utility of consumption.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncShkDstn : distribution.Distribution
        A discrete approximation to the income process between the period being
        solved and the one immediately following (in solution_next).
    PrefShkDstn : distribution.Distribution
        Discrete distribution of the multiplicative utility shifter.  Order:
        probabilities, preference shocks.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
    Rboro: float
        Interest factor on assets between this period and the succeeding
        period when assets are negative.
    Rsave: float
        Interest factor on assets between this period and the succeeding
        period when assets are positive.
    PermGroFac : float
        Expected permanent income growth factor at the end of this period.
    BoroCnstArt: float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.  If it is less than the natural borrowing constraint,
        then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
        rowing constraint.
    aXtraGrid: np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear inter-
        polation.

    Returns
    -------
    solution_now : ConsumerSolution
        The solution to the single period consumption-saving problem.  Includes
        a consumption function cFunc (using linear splines), a marginal value
        function vPfunc, a minimum acceptable level of normalized market re-
        sources mNrmMin, normalized human wealth hNrm, and bounding MPCs MPCmin
        and MPCmax.  It might also have a value function vFunc.  The consumption
        function is defined over normalized market resources and the preference
        shock, c = cFunc(m,PrefShk), but the (marginal) value function is defined
        unconditionally on the shock, just before it is revealed.
    """
    # Verifiy that there is actually a kink in the interest factor
    assert Rboro >= Rsave, (
        "Interest factor on debt less than interest factor on savings!"
    )
    # If the kink is in the wrong direction, code should break here. If there's
    # no kink at all, then just use the ConsIndShockModel solver.
    if Rboro == Rsave:
        solution_now = solve_one_period_ConsPrefShock(
            solution_next,
            IncShkDstn,
            PrefShkDstn,
            LivPrb,
            DiscFac,
            CRRA,
            Rboro,
            PermGroFac,
            BoroCnstArt,
            aXtraGrid,
            vFuncBool,
            CubicBool,
        )
        return solution_now

    # Define the current period utility function and effective discount factor
    uFunc = UtilityFuncCRRA(CRRA)
    DiscFacEff = DiscFac * LivPrb  # "effective" discount factor

    # Unpack next period's income and preference shock distributions
    ShkPrbsNext = IncShkDstn.pmv
    PermShkValsNext = IncShkDstn.atoms[0]
    TranShkValsNext = IncShkDstn.atoms[1]
    PermShkMinNext = np.min(PermShkValsNext)
    TranShkMinNext = np.min(TranShkValsNext)
    PrefShkPrbs = PrefShkDstn.pmv
    PrefShkVals = PrefShkDstn.atoms.flatten()

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
    PatFac = ((Rsave * DiscFacEff) ** (1.0 / CRRA)) / Rsave
    PatFacAlt = ((Rboro * DiscFacEff) ** (1.0 / CRRA)) / Rboro
    try:
        MPCminNow = 1.0 / (1.0 + PatFac / solution_next.MPCmin)
    except:
        MPCminNow = 0.0
    Ex_IncNext = np.dot(ShkPrbsNext, TranShkValsNext * PermShkValsNext)
    hNrmNow = (PermGroFac / Rsave) * (Ex_IncNext + solution_next.hNrm)
    temp_fac = (WorstIncPrb ** (1.0 / CRRA)) * PatFacAlt
    MPCmaxNow = 1.0 / (1.0 + temp_fac / solution_next.MPCmax)

    # Calculate the minimum allowable value of money resources in this period
    PermGroFacEffMin = (PermGroFac * PermShkMinNext) / Rboro
    BoroCnstNat = (solution_next.mNrmMin - TranShkMinNext) * PermGroFacEffMin

    # Set the minimum allowable (normalized) market resources based on the natural
    # and artificial borrowing constraints
    if BoroCnstArt is None:
        mNrmMinNow = BoroCnstNat
    else:
        mNrmMinNow = np.max([BoroCnstNat, BoroCnstArt])

    # Set the upper limit of the MPC (at mNrmMinNow) based on whether the natural
    # or artificial borrowing constraint actually binds
    if BoroCnstNat < mNrmMinNow:
        MPCmaxEff = 1.0  # If actually constrained, MPC near limit is 1
    else:
        MPCmaxEff = MPCmaxNow  # Otherwise, it's the MPC calculated above

    # Define the borrowing-constrained consumption function
    cFuncNowCnst = LinearInterp(
        np.array([mNrmMinNow, mNrmMinNow + 1.0]), np.array([0.0, 1.0])
    )

    # Construct the assets grid by adjusting aXtra by the natural borrowing constraint
    aNrmNow = np.sort(
        np.hstack((np.asarray(aXtraGrid) + mNrmMinNow, np.array([0.0, 0.0])))
    )

    # Make a 1D array of the interest factor at each asset gridpoint
    Rfree = Rsave * np.ones_like(aNrmNow)
    Rfree[aNrmNow < 0] = Rboro
    i_kink = np.argwhere(aNrmNow == 0.0)[0][0]
    Rfree[i_kink] = Rboro

    # Define local functions for taking future expectations
    def calc_mNrmNext(S, a, R):
        return R / (PermGroFac * S["PermShk"]) * a + S["TranShk"]

    def calc_vNext(S, a, R):
        return (S["PermShk"] ** (1.0 - CRRA) * PermGroFac ** (1.0 - CRRA)) * vFuncNext(
            calc_mNrmNext(S, a, R)
        )

    def calc_vPnext(S, a, R):
        return S["PermShk"] ** (-CRRA) * vPfuncNext(calc_mNrmNext(S, a, R))

    def calc_vPPnext(S, a, R):
        return S["PermShk"] ** (-CRRA - 1.0) * vPPfuncNext(calc_mNrmNext(S, a, R))

    # Calculate end-of-period marginal value of assets at each gridpoint
    vPfacEff = DiscFacEff * Rfree * PermGroFac ** (-CRRA)
    EndOfPrdvP = vPfacEff * expected(calc_vPnext, IncShkDstn, args=(aNrmNow, Rfree))

    # Find optimal consumption corresponding to each aNrm, PrefShk combination
    cNrm_base = uFunc.derinv(EndOfPrdvP, order=(1, 0))
    PrefShkCount = PrefShkVals.size
    PrefShk_temp = np.tile(
        np.reshape(PrefShkVals ** (1.0 / CRRA), (PrefShkCount, 1)),
        (1, cNrm_base.size),
    )
    cNrmNow = np.tile(cNrm_base, (PrefShkCount, 1)) * PrefShk_temp
    mNrmNow = cNrmNow + np.tile(aNrmNow, (PrefShkCount, 1))
    # These are the endogenous gridpoints, as usual

    # Add the bottom point to the c and m arrays
    m_for_interpolation = np.concatenate(
        (BoroCnstNat * np.ones((PrefShkCount, 1)), mNrmNow), axis=1
    )
    c_for_interpolation = np.concatenate((np.zeros((PrefShkCount, 1)), cNrmNow), axis=1)

    # Construct the consumption function as a cubic or linear spline interpolation
    # for each value of PrefShk, interpolated across those values.
    if CubicBool:
        # This is not yet supported, not sure why we never got to it
        raise ValueError(
            "Cubic interpolation is not yet supported by the preference shock model!"
        )

    # Make the preference-shock specific consumption functions
    cFuncs_by_PrefShk = []
    for j in range(PrefShkCount):
        MPCmin_j = MPCminNow * PrefShkVals[j] ** (1.0 / CRRA)
        cFunc_this_shk = LowerEnvelope(
            LinearInterp(
                m_for_interpolation[j, :],
                c_for_interpolation[j, :],
                intercept_limit=hNrmNow * MPCmin_j,
                slope_limit=MPCmin_j,
            ),
            cFuncNowCnst,
        )
        cFuncs_by_PrefShk.append(cFunc_this_shk)

    # Combine the list of consumption functions into a single interpolation
    cFuncNow = LinearInterpOnInterp1D(cFuncs_by_PrefShk, PrefShkVals)

    # Make the ex ante marginal value function (before the preference shock)
    m_grid = aXtraGrid + mNrmMinNow
    vP_vec = np.zeros_like(m_grid)
    for j in range(PrefShkCount):  # numeric integration over the preference shock
        vP_vec += (
            uFunc.der(cFuncs_by_PrefShk[j](m_grid)) * PrefShkPrbs[j] * PrefShkVals[j]
        )
    vPnvrs_vec = uFunc.derinv(vP_vec, order=(1, 0))
    vPfuncNow = MargValueFuncCRRA(LinearInterp(m_grid, vPnvrs_vec), CRRA)

    # Define this period's marginal marginal value function
    vPPfuncNow = NullFunc()  # Dummy object, cubic interpolation not implemented

    # Construct this period's value function if requested
    if vFuncBool:
        # Calculate end-of-period value, its derivative, and their pseudo-inverse
        EndOfPrdv = DiscFacEff * expected(calc_vNext, IncShkDstn, args=(aNrmNow, Rfree))
        EndOfPrdvNvrs = uFunc.inv(
            EndOfPrdv
        )  # value transformed through inverse utility
        EndOfPrdvNvrsP = EndOfPrdvP * uFunc.derinv(EndOfPrdv, order=(0, 1))
        EndOfPrdvNvrs = np.insert(EndOfPrdvNvrs, 0, 0.0)
        EndOfPrdvNvrsP = np.insert(EndOfPrdvNvrsP, 0, EndOfPrdvNvrsP[0])
        # This is a very good approximation, vNvrsPP = 0 at the asset minimum

        # Construct the end-of-period value function
        aNrm_temp = np.insert(aNrmNow, 0, BoroCnstNat)
        EndOfPrd_vNvrsFunc = CubicInterp(aNrm_temp, EndOfPrdvNvrs, EndOfPrdvNvrsP)
        EndOfPrd_vFunc = ValueFuncCRRA(EndOfPrd_vNvrsFunc, CRRA)

        # Compute expected value and marginal value on a grid of market resources,
        # accounting for all of the discrete preference shocks
        mNrm_temp = mNrmMinNow + aXtraGrid
        v_temp = np.zeros_like(mNrm_temp)
        vP_temp = np.zeros_like(mNrm_temp)
        for j in range(PrefShkCount):
            this_shock = PrefShkVals[j]
            this_prob = PrefShkPrbs[j]
            cNrm_temp = cFuncNow(mNrm_temp, this_shock * np.ones_like(mNrm_temp))
            aNrm_temp = mNrm_temp - cNrm_temp
            v_temp += this_prob * (
                this_shock * uFunc(cNrm_temp) + EndOfPrd_vFunc(aNrm_temp)
            )
            vP_temp += this_prob * this_shock * uFunc.der(cNrm_temp)

        # Construct the beginning-of-period value function
        # value transformed through inverse utility
        vNvrs_temp = uFunc.inv(v_temp)
        vNvrsP_temp = vP_temp * uFunc.derinv(v_temp, order=(0, 1))
        mNrm_temp = np.insert(mNrm_temp, 0, mNrmMinNow)
        vNvrs_temp = np.insert(vNvrs_temp, 0, 0.0)
        vNvrsP_temp = np.insert(vNvrsP_temp, 0, MPCmaxEff ** (-CRRA / (1.0 - CRRA)))
        MPCminNvrs = MPCminNow ** (-CRRA / (1.0 - CRRA))
        vNvrsFuncNow = CubicInterp(
            mNrm_temp, vNvrs_temp, vNvrsP_temp, MPCminNvrs * hNrmNow, MPCminNvrs
        )
        vFuncNow = ValueFuncCRRA(vNvrsFuncNow, CRRA)

    else:
        vFuncNow = NullFunc()  # Dummy object

    # Create and return this period's solution
    solution_now = ConsumerSolution(
        cFunc=cFuncNow,
        vFunc=vFuncNow,
        vPfunc=vPfuncNow,
        vPPfunc=vPPfuncNow,
        mNrmMin=mNrmMinNow,
        hNrm=hNrmNow,
        MPCmin=MPCminNow,
        MPCmax=MPCmaxEff,
    )
    return solution_now


###############################################################################

# Make a dictionary of constructors for the preference shock model
PrefShockConsumerType_constructors_default = {
    "IncShkDstn": construct_lognormal_income_process_unemployment,
    "PermShkDstn": get_PermShkDstn_from_IncShkDstn,
    "TranShkDstn": get_TranShkDstn_from_IncShkDstn,
    "aXtraGrid": make_assets_grid,
    "PrefShkDstn": make_lognormal_PrefShkDstn,
    "solution_terminal": make_pref_shock_solution_terminal,
    "kNrmInitDstn": make_lognormal_kNrm_init_dstn,
    "pLvlInitDstn": make_lognormal_pLvl_init_dstn,
}

# Make a dictionary with parameters for the default constructor for kNrmInitDstn
PrefShockConsumerType_kNrmInitDstn_default = {
    "kLogInitMean": -12.0,  # Mean of log initial capital
    "kLogInitStd": 0.0,  # Stdev of log initial capital
    "kNrmInitCount": 15,  # Number of points in initial capital discretization
}

# Make a dictionary with parameters for the default constructor for pLvlInitDstn
PrefShockConsumerType_pLvlInitDstn_default = {
    "pLogInitMean": 0.0,  # Mean of log permanent income
    "pLogInitStd": 0.0,  # Stdev of log permanent income
    "pLvlInitCount": 15,  # Number of points in initial capital discretization
}

# Default parameters to make IncShkDstn using construct_lognormal_income_process_unemployment
PrefShockConsumerType_IncShkDstn_default = {
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

# Default parameters to make aXtraGrid using construct_assets_grid

PrefShockConsumerType_aXtraGrid_default = {
    "aXtraMin": 0.001,  # Minimum end-of-period "assets above minimum" value
    "aXtraMax": 20,  # Maximum end-of-period "assets above minimum" value
    "aXtraNestFac": 3,  # Exponential nesting factor for aXtraGrid
    "aXtraCount": 48,  # Number of points in the grid of "assets above minimum"
    "aXtraExtra": None,  # Additional other values to add in grid (optional)
}

# Default parameters to make PrefShkDstn using make_lognormal_PrefShkDstn

PrefShockConsumerType_PrefShkDstn_default = {
    "PrefShkCount": 12,  # Number of points in discrete approximation to preference shock dist
    "PrefShk_tail_N": 4,  # Number of "tail points" on each end of pref shock dist
    "PrefShkStd": [0.30],  # Standard deviation of utility shocks
}

# Make a dictionary to specify an preference shocks consumer type
PrefShockConsumerType_solving_default = {
    # BASIC HARK PARAMETERS REQUIRED TO SOLVE THE MODEL
    "cycles": 1,  # Finite, non-cyclic model
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    "pseudo_terminal": False,  # Terminal period really does exist
    "constructors": PrefShockConsumerType_constructors_default,  # See dictionary above
    # PRIMITIVE RAW PARAMETERS REQUIRED TO SOLVE THE MODEL
    "CRRA": 2.0,  # Coefficient of relative risk aversion
    "Rfree": [1.03],  # Interest factor on retained assets
    "DiscFac": 0.96,  # Intertemporal discount factor
    "LivPrb": [0.98],  # Survival probability after each period
    "PermGroFac": [1.01],  # Permanent income growth factor
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint
    "vFuncBool": False,  # Whether to calculate the value function during solution
    "CubicBool": False,  # Whether to use cubic spline interpolation when True
    # (Uses linear spline interpolation for cFunc when False)
}
PrefShockConsumerType_simulation_default = {
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

PrefShockConsumerType_default = {}
PrefShockConsumerType_default.update(PrefShockConsumerType_IncShkDstn_default)
PrefShockConsumerType_default.update(PrefShockConsumerType_aXtraGrid_default)
PrefShockConsumerType_default.update(PrefShockConsumerType_PrefShkDstn_default)
PrefShockConsumerType_default.update(PrefShockConsumerType_kNrmInitDstn_default)
PrefShockConsumerType_default.update(PrefShockConsumerType_pLvlInitDstn_default)
PrefShockConsumerType_default.update(PrefShockConsumerType_solving_default)
PrefShockConsumerType_default.update(PrefShockConsumerType_simulation_default)
init_preference_shocks = (
    PrefShockConsumerType_default  # So models that aren't updated don't break
)


class PrefShockConsumerType(IndShockConsumerType):
    r"""
    A consumer type based on IndShockConsumerType, with multiplicative shocks to utility each period.

    .. math::
        \newcommand{\CRRA}{\rho}
        \newcommand{\DiePrb}{\mathsf{D}}
        \newcommand{\PermGroFac}{\Gamma}
        \newcommand{\Rfree}{\mathsf{R}}
        \newcommand{\DiscFac}{\beta}
        \begin{align*}
        v_t(m_t,\eta_t) &=\max_{c_t} \eta_{t} u(c_t) + \DiscFac (1 - \DiePrb_{t+1}) \mathbb{E}_{t} \left[ (\PermGroFac_{t+1} \psi_{t+1})^{1-\CRRA} v_{t+1}(m_{t+1},\eta_{t+1}) \right], \\
        & \text{s.t.}  \\
        a_t &= m_t - c_t, \\
        a_t &\geq \underline{a}, \\
        m_{t+1} &= a_t \Rfree_{t+1}/(\PermGroFac_{t+1} \psi_{t+1}) + \theta_{t+1}, \\
        (\psi_{t+1},\theta_{t+1},\eta_{t+1}) &\sim F_{t+1}, \\
        \mathbb{E}[\psi]=\mathbb{E}[\theta] &= 1, \\
        u(c) &= \frac{c^{1-\CRRA}}{1-\CRRA} \\
        \end{align*}


    Constructors
    ------------
    IncShkDstn: Constructor, :math:`\psi`, :math:`\theta`
        The agent's income shock distributions.

        Its default constructor is :func:`HARK.Calibration.Income.IncomeProcesses.construct_lognormal_income_process_unemployment`
    aXtraGrid: Constructor
        The agent's asset grid.

        Its default constructor is :func:`HARK.utilities.make_assets_grid`
    PrefShkDstn: Constructor, :math:`\eta`
        The agent's preference shock distributions.

        Its default constuctor is :func:`HARK.ConsumptionSaving.ConsPrefShockModel.make_lognormal_PrefShkDstn`

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
        For this agent, the options are 'PermShk', 'PrefShk', 'TranShk', 'aLvl', 'aNrm', 'bNrm', 'cNrm', 'mNrm', 'pLvl', and 'who_dies'.

        PermShk is the agent's permanent income shock

        PrefShk is the agent's preference shock

        TranShk is the agent's transitory income shock

        aLvl is the nominal asset level

        aNrm is the normalized assets

        bNrm is the normalized resources without this period's labor income

        cNrm is the normalized consumption

        mNrm is the normalized market resources

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

        For this model, cFunc is defined over normalized market resources and :math:`\eta`, cNrm = cFunc(mNrm, :math:`\eta`).

        Visit :class:`HARK.ConsumptionSaving.ConsIndShockModel.ConsumerSolution` for more information about the solution.
    history: Dict[Array]
        Created by running the :func:`.simulate()` method.
        Contains the variables in track_vars. Each item in the dictionary is an array with the shape (T_sim,AgentCount).
        Visit :class:`HARK.core.AgentType.simulate` for more information.
    """

    IncShkDstn_defaults = PrefShockConsumerType_IncShkDstn_default
    aXtraGrid_defaults = PrefShockConsumerType_aXtraGrid_default
    PrefShkDstn_defaults = PrefShockConsumerType_PrefShkDstn_default
    solving_defaults = PrefShockConsumerType_solving_default
    simulation_defaults = PrefShockConsumerType_simulation_default
    default_ = {
        "params": PrefShockConsumerType_default,
        "solver": solve_one_period_ConsPrefShock,
        "model": "ConsMarkov.yaml",
    }

    shock_vars_ = IndShockConsumerType.shock_vars_ + ["PrefShk"]
    time_vary_ = IndShockConsumerType.time_vary_ + ["PrefShkDstn"]
    distributions = [
        "IncShkDstn",
        "PermShkDstn",
        "TranShkDstn",
        "kNrmInitDstn",
        "pLvlInitDstn",
        "PrefShkDstn",
    ]

    def pre_solve(self):
        self.construct("solution_terminal")

    def reset_rng(self):
        """
        Reset the RNG behavior of this type.  This method is called automatically
        by initialize_sim(), ensuring that each simulation run uses the same sequence
        of random shocks; this is necessary for structural estimation to work.
        This method extends IndShockConsumerType.reset_rng() to also reset elements
        of PrefShkDstn.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        IndShockConsumerType.reset_rng(self)

        # Reset PrefShkDstn if it exists (it might not because reset_rng is called at init)
        if hasattr(self, "PrefShkDstn"):
            for dstn in self.PrefShkDstn:
                dstn.reset()

    def get_shocks(self):
        """
        Gets permanent and transitory income shocks for this period as well as preference shocks.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        IndShockConsumerType.get_shocks(
            self
        )  # Get permanent and transitory income shocks
        PrefShkNow = np.zeros(self.AgentCount)  # Initialize shock array
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            N = np.sum(these)
            if N > 0:
                PrefShkNow[these] = self.PrefShkDstn[t].draw(N)
        self.shocks["PrefShk"] = PrefShkNow

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
        cNrmNow = np.zeros(self.AgentCount) + np.nan
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            cNrmNow[these] = self.solution[t].cFunc(
                self.state_now["mNrm"][these], self.shocks["PrefShk"][these]
            )
        self.controls["cNrm"] = cNrmNow
        return None

    def calc_bounding_values(self):  # pragma: nocover
        """
        NOT YET IMPLEMENTED FOR THIS CLASS
        """
        raise NotImplementedError()

    def make_euler_error_func(self, mMax=100, approx_inc_dstn=True):  # pragma: nocover
        """
        NOT YET IMPLEMENTED FOR THIS CLASS
        """
        raise NotImplementedError()

    def check_conditions(self, verbose=None):
        raise NotImplementedError()  # pragma: nocover

    def calc_limiting_values(self):
        raise NotImplementedError()  # pragma: nocover


###############################################################################

# Specify default parameters that differ in "kinky preference" model compared to base PrefShockConsumerType
kinky_pref_different_params = {
    "Rboro": 1.20,  # Interest factor on assets when borrowing, a < 0
    "Rsave": 1.02,  # Interest factor on assets when saving, a > 0
    "BoroCnstArt": None,  # Kinked R only matters if borrowing is allowed
}
KinkyPrefConsumerType_constructors_default = (
    PrefShockConsumerType_constructors_default.copy()
)
KinkyPrefConsumerType_IncShkDstn_default = (
    PrefShockConsumerType_IncShkDstn_default.copy()
)
KinkyPrefConsumerType_pLvlInitDstn_default = (
    PrefShockConsumerType_pLvlInitDstn_default.copy()
)
KinkyPrefConsumerType_kNrmInitDstn_default = (
    PrefShockConsumerType_kNrmInitDstn_default.copy()
)
KinkyPrefConsumerType_aXtraGrid_default = PrefShockConsumerType_aXtraGrid_default.copy()
KinkyPrefConsumerType_PrefShkDstn_default = (
    PrefShockConsumerType_PrefShkDstn_default.copy()
)
KinkyPrefConsumerType_solving_default = PrefShockConsumerType_solving_default.copy()
KinkyPrefConsumerType_solving_default["constructors"] = (
    KinkyPrefConsumerType_constructors_default
)
KinkyPrefConsumerType_simulation_default = (
    PrefShockConsumerType_simulation_default.copy()
)
KinkyPrefConsumerType_solving_default.update(kinky_pref_different_params)

# Make a dictionary to specify a "kinky preference" consumer
KinkyPrefConsumerType_default = {}
KinkyPrefConsumerType_default.update(KinkyPrefConsumerType_IncShkDstn_default)
KinkyPrefConsumerType_default.update(KinkyPrefConsumerType_aXtraGrid_default)
KinkyPrefConsumerType_default.update(KinkyPrefConsumerType_PrefShkDstn_default)
KinkyPrefConsumerType_default.update(KinkyPrefConsumerType_kNrmInitDstn_default)
KinkyPrefConsumerType_default.update(KinkyPrefConsumerType_pLvlInitDstn_default)
KinkyPrefConsumerType_default.update(KinkyPrefConsumerType_solving_default)
KinkyPrefConsumerType_default.update(KinkyPrefConsumerType_simulation_default)
init_kinky_pref = KinkyPrefConsumerType_default


class KinkyPrefConsumerType(PrefShockConsumerType, KinkedRconsumerType):
    r"""
    A consumer type based on PrefShockConsumerType, with different
    interest rates for saving (:math:`\mathsf{R}_{save}`) and borrowing
    (:math:`\mathsf{R}_{boro}`).

    Solver for this class is currently only compatible with linear spline interpolation.

    .. math::
        \newcommand{\CRRA}{\rho}
        \newcommand{\DiePrb}{\mathsf{D}}
        \newcommand{\PermGroFac}{\Gamma}
        \newcommand{\Rfree}{\mathsf{R}}
        \newcommand{\DiscFac}{\beta}
        \begin{align*}
        v_t(m_t,\eta_t) &= \max_{c_t} \eta_{t} u(c_t) + \DiscFac (1-\DiePrb_{t+1})  \mathbb{E}_{t} \left[(\PermGroFac_{t+1}\psi_{t+1})^{1-\CRRA} v_{t+1}(m_{t+1},\eta_{t+1}) \right], \\
        a_t &= m_t - c_t, \\
        a_t &\geq \underline{a}, \\
        m_{t+1} &= \Rfree_t/(\PermGroFac_{t+1} \psi_{t+1}) a_t + \theta_{t+1}, \\
        \Rfree_t &= \begin{cases}
        \Rfree_{boro} & \text{if } a_t < 0\\
        \Rfree_{save} & \text{if } a_t \geq 0,
        \end{cases}\\
        \Rfree_{boro} &> \Rfree_{save}, \\
        (\psi_{t+1},\theta_{t+1},\eta_{t+1}) &\sim F_{t+1}, \\
        \mathbb{E}[\psi]=\mathbb{E}[\theta] &= 1. \\
        u(c) &= \frac{c^{1-\CRRA}}{1-\CRRA} \\
        \end{align*}


    Constructors
    ------------
    IncShkDstn: Constructor, :math:`\psi`, :math:`\theta`
        The agent's income shock distributions.

        It's default constructor is :func:`HARK.Calibration.Income.IncomeProcesses.construct_lognormal_income_process_unemployment`
    aXtraGrid: Constructor
        The agent's asset grid.

        It's default constructor is :func:`HARK.utilities.make_assets_grid`
    PrefShkDstn: Constructor, :math:`\eta`
        The agent's preference shock distributions.

        It's default constuctor is :func:`HARK.ConsumptionSaving.ConsPrefShockModel.make_lognormal_PrefShkDstn`

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
    Rboro: float, :math:`\mathsf{R}_{boro}`
        Risk Free interest rate when assets are negative.
    Rsave: float, :math:`\mathsf{R}_{save}`
        Risk Free interest rate when assets are positive.
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
        For this agent, the options are 'PermShk', 'PrefShk', 'TranShk', 'aLvl', 'aNrm', 'bNrm', 'cNrm', 'mNrm', 'pLvl', and 'who_dies'.

        PermShk is the agent's permanent income shock

        PrefShk is the agent's preference shock

        TranShk is the agent's transitory income shock

        aLvl is the nominal asset level

        aNrm is the normalized assets

        bNrm is the normalized resources without this period's labor income

        cNrm is the normalized consumption

        mNrm is the normalized market resources

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

        For this model, cFunc is defined over normalized market resources and :math:`\eta`, cNrm = cFunc(mNrm, :math:`\eta`).

        Visit :class:`HARK.ConsumptionSaving.ConsIndShockModel.ConsumerSolution` for more information about the solution.
    history: Dict[Array]
        Created by running the :func:`.simulate()` method.
        Contains the variables in track_vars. Each item in the dictionary is an array with the shape (T_sim,AgentCount).
        Visit :class:`HARK.core.AgentType.simulate` for more information.
    """

    IncShkDstn_defaults = KinkyPrefConsumerType_IncShkDstn_default
    aXtraGrid_defaults = KinkyPrefConsumerType_aXtraGrid_default
    PrefShkDstn_defaults = KinkyPrefConsumerType_PrefShkDstn_default
    solving_defaults = KinkyPrefConsumerType_solving_default
    simulation_defaults = KinkyPrefConsumerType_simulation_default
    default_ = {
        "params": KinkyPrefConsumerType_default,
        "solver": solve_one_period_ConsKinkyPref,
    }

    time_inv_ = IndShockConsumerType.time_inv_ + ["Rboro", "Rsave"]
    distributions = [
        "IncShkDstn",
        "PermShkDstn",
        "TranShkDstn",
        "kNrmInitDstn",
        "pLvlInitDstn",
        "PrefShkDstn",
    ]

    def pre_solve(self):
        self.construct("solution_terminal")

    def get_Rport(self):  # Specify which get_Rport to use
        return KinkedRconsumerType.get_Rport(self)

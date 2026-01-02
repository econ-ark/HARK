"""
Classes to solve consumption-saving models with a bequest motive and
idiosyncratic shocks to income and wealth. All models here assume
separable CRRA utility of consumption and Stone-Geary utility of
savings with geometric discounting of the continuation value and
shocks to income that have transitory and/or permanent components.

It currently solves two types of models:
    1) A standard lifecycle model with a terminal and/or accidental bequest motive.
    2) A portfolio choice model with a terminal and/or accidental bequest motive.
"""

import numpy as np

from HARK import NullFunc
from HARK.Calibration.Income.IncomeProcesses import (
    construct_lognormal_income_process_unemployment,
    get_PermShkDstn_from_IncShkDstn,
    get_TranShkDstn_from_IncShkDstn,
)
from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsumerSolution,
    IndShockConsumerType,
    make_basic_CRRA_solution_terminal,
    make_lognormal_kNrm_init_dstn,
    make_lognormal_pLvl_init_dstn,
)
from HARK.Calibration.Assets.AssetProcesses import (
    make_lognormal_RiskyDstn,
    combine_IncShkDstn_and_RiskyDstn,
    calc_ShareLimit_for_CRRA,
)
from HARK.ConsumptionSaving.ConsPortfolioModel import (
    PortfolioConsumerType,
    PortfolioSolution,
    make_portfolio_solution_terminal,
    make_AdjustDstn,
)
from HARK.ConsumptionSaving.ConsRiskyAssetModel import make_simple_ShareGrid
from HARK.distributions import expected
from HARK.interpolation import (
    BilinearInterp,
    ConstantFunction,
    CubicInterp,
    IdentityFunction,
    LinearInterp,
    LinearInterpOnInterp1D,
    LowerEnvelope,
    MargMargValueFuncCRRA,
    MargValueFuncCRRA,
    ValueFuncCRRA,
)
from HARK.rewards import UtilityFuncCRRA, UtilityFuncStoneGeary
from HARK.utilities import make_assets_grid


def make_bequest_solution_terminal(
    CRRA, BeqCRRATerm, BeqFacTerm, BeqShiftTerm, aXtraGrid
):
    """
    Make the terminal period solution when there is a warm glow bequest motive with
    Stone-Geary form utility. If there is no warm glow bequest motive (BeqFacTerm = 0),
    then the terminal period solution is identical to ConsIndShock.

    Parameters
    ----------
    CRRA : float
        Coefficient on relative risk aversion over consumption.
    BeqCRRATerm : float
        Coefficient on relative risk aversion in the terminal warm glow bequest motive.
    BeqFacTerm : float
        Scaling factor for the terminal warm glow bequest motive.
    BeqShiftTerm : float
        Stone-Geary shifter term for the terminal warm glow bequest motive.
    aXtraGrid : np.array
        Set of assets-above-minimum to be used in the solution.

    Returns
    -------
    solution_terminal : ConsumerSolution
        Terminal period solution when there is a warm glow bequest.
    """
    if BeqFacTerm == 0.0:  # No terminal bequest
        solution_terminal = make_basic_CRRA_solution_terminal(CRRA)
        return solution_terminal

    utility = UtilityFuncCRRA(CRRA)
    warm_glow = UtilityFuncStoneGeary(
        BeqCRRATerm,
        factor=BeqFacTerm,
        shifter=BeqShiftTerm,
    )

    aNrmGrid = np.append(0.0, aXtraGrid) if BeqShiftTerm != 0.0 else aXtraGrid
    cNrmGrid = utility.derinv(warm_glow.der(aNrmGrid))
    vGrid = utility(cNrmGrid) + warm_glow(aNrmGrid)
    cNrmGridW0 = np.append(0.0, cNrmGrid)
    mNrmGridW0 = np.append(0.0, aNrmGrid + cNrmGrid)
    vNvrsGridW0 = np.append(0.0, utility.inv(vGrid))

    cFunc_term = LinearInterp(mNrmGridW0, cNrmGridW0)
    vNvrsFunc_term = LinearInterp(mNrmGridW0, vNvrsGridW0)
    vFunc_term = ValueFuncCRRA(vNvrsFunc_term, CRRA)
    vPfunc_term = MargValueFuncCRRA(cFunc_term, CRRA)
    vPPfunc_term = MargMargValueFuncCRRA(cFunc_term, CRRA)

    solution_terminal = ConsumerSolution(
        cFunc=cFunc_term,
        vFunc=vFunc_term,
        vPfunc=vPfunc_term,
        vPPfunc=vPPfunc_term,
        mNrmMin=0.0,
        hNrm=0.0,
        MPCmax=1.0,
        MPCmin=1.0,
    )
    return solution_terminal


def make_warmglow_portfolio_solution_terminal(
    CRRA, BeqCRRATerm, BeqFacTerm, BeqShiftTerm, aXtraGrid
):
    """
    Make the terminal period solution when there is a warm glow bequest motive with
    Stone-Geary form utility and portfolio choice. If there is no warm glow bequest
    motive (BeqFacTerm = 0), then the terminal period solution is identical to ConsPortfolio.

    Parameters
    ----------
    CRRA : float
        Coefficient on relative risk aversion over consumption.
    BeqCRRATerm : float
        Coefficient on relative risk aversion in the terminal warm glow bequest motive.
    BeqFacTerm : float
        Scaling factor for the terminal warm glow bequest motive.
    BeqShiftTerm : float
        Stone-Geary shifter term for the terminal warm glow bequest motive.
    aXtraGrid : np.array
        Set of assets-above-minimum to be used in the solution.

    Returns
    -------
    solution_terminal : ConsumerSolution
        Terminal period solution when there is a warm glow bequest and portfolio choice.
    """
    if BeqFacTerm == 0.0:  # No terminal bequest
        solution_terminal = make_portfolio_solution_terminal(CRRA)
        return solution_terminal

    # Solve the terminal period problem when there is no portfolio choice
    solution_terminal_no_port = make_bequest_solution_terminal(
        CRRA, BeqCRRATerm, BeqFacTerm, BeqShiftTerm, aXtraGrid
    )

    # Take consumption function from the no portfolio choice solution
    cFuncAdj_terminal = solution_terminal_no_port.cFunc
    cFuncFxd_terminal = lambda m, s: solution_terminal_no_port(m)

    # Risky share is irrelevant-- no end-of-period assets; set to zero
    ShareFuncAdj_terminal = ConstantFunction(0.0)
    ShareFuncFxd_terminal = IdentityFunction(i_dim=1, n_dims=2)

    # Value function is simply utility from consuming market resources
    vFuncAdj_terminal = solution_terminal_no_port.vFunc
    vFuncFxd_terminal = lambda m, s: solution_terminal_no_port.cFunc(m)

    # Marginal value of market resources is marg utility at the consumption function
    vPfuncAdj_terminal = solution_terminal_no_port.vPfunc
    dvdmFuncFxd_terminal = lambda m, s: solution_terminal_no_port.vPfunc(m)
    # No future, no marg value of Share
    dvdsFuncFxd_terminal = ConstantFunction(0.0)

    # Construct the terminal period solution
    solution_terminal = PortfolioSolution(
        cFuncAdj=cFuncAdj_terminal,
        ShareFuncAdj=ShareFuncAdj_terminal,
        vFuncAdj=vFuncAdj_terminal,
        vPfuncAdj=vPfuncAdj_terminal,
        cFuncFxd=cFuncFxd_terminal,
        ShareFuncFxd=ShareFuncFxd_terminal,
        vFuncFxd=vFuncFxd_terminal,
        dvdmFuncFxd=dvdmFuncFxd_terminal,
        dvdsFuncFxd=dvdsFuncFxd_terminal,
    )
    return solution_terminal


###############################################################################


def solve_one_period_ConsWarmBequest(
    solution_next,
    IncShkDstn,
    LivPrb,
    DiscFac,
    CRRA,
    Rfree,
    PermGroFac,
    BoroCnstArt,
    aXtraGrid,
    BeqCRRA,
    BeqFac,
    BeqShift,
    CubicBool,
    vFuncBool,
):
    """
    Solves one period of a consumption-saving model with idiosyncratic shocks to
    permanent and transitory income, with one risk free asset and CRRA utility.
    The consumer also has a "warm glow" bequest motive in which they gain additional
    utility based on their terminal wealth upon death.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncShkDstn : distribution.Distribution
        A discrete approximation to the income process between the period being
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
    PermGroFac : float
        Expected permanent income growth factor at the end of this period.
    BoroCnstArt: float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.  If it is less than the natural borrowing constraint,
        then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
        rowing constraint.
    aXtraGrid : np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.
    BeqCRRA : float
        Coefficient of relative risk aversion for warm glow bequest motive.
    BeqFac : float
        Multiplicative intensity factor for the warm glow bequest motive.
    BeqShift : float
        Stone-Geary shifter in the warm glow bequest motive.
    CubicBool : bool
        An indicator for whether the solver should use cubic or linear interpolation.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.

    Returns
    -------
    solution_now : ConsumerSolution
        Solution to this period's consumption-saving problem with income risk.
    """
    # Define the current period utility function and effective discount factor
    uFunc = UtilityFuncCRRA(CRRA)
    DiscFacEff = DiscFac * LivPrb  # "effective" discount factor
    BeqFacEff = (1.0 - LivPrb) * BeqFac  # "effective" bequest factor
    warm_glow = UtilityFuncStoneGeary(BeqCRRA, BeqFacEff, BeqShift)

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
    Ex_IncNext = np.dot(ShkPrbsNext, TranShkValsNext * PermShkValsNext)
    hNrmNow = PermGroFac / Rfree * (Ex_IncNext + solution_next.hNrm)
    temp_fac = (WorstIncPrb ** (1.0 / CRRA)) * PatFac
    MPCmaxNow = 1.0 / (1.0 + temp_fac / solution_next.MPCmax)
    cFuncLimitIntercept = MPCminNow * hNrmNow
    cFuncLimitSlope = MPCminNow

    # Calculate the minimum allowable value of money resources in this period
    PermGroFacEffMin = (PermGroFac * PermShkMinNext) / Rfree
    BoroCnstNat = (solution_next.mNrmMin - TranShkMinNext) * PermGroFacEffMin
    BoroCnstNat = np.max([BoroCnstNat, -BeqShift])

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
    EndOfPrdvP += warm_glow.der(aNrmNow)

    # Invert the first order condition to find optimal cNrm from each aNrm gridpoint
    cNrmNow = uFunc.derinv(EndOfPrdvP, order=(1, 0))
    mNrmNow = cNrmNow + aNrmNow  # Endogenous mNrm gridpoints

    # Limiting consumption is zero as m approaches mNrmMin
    c_for_interpolation = np.insert(cNrmNow, 0, 0.0)
    m_for_interpolation = np.insert(mNrmNow, 0, BoroCnstNat)

    # Construct the consumption function as a cubic or linear spline interpolation
    if CubicBool:
        # Calculate end-of-period marginal marginal value of assets at each gridpoint
        vPPfacEff = DiscFacEff * Rfree * Rfree * PermGroFac ** (-CRRA - 1.0)
        EndOfPrdvPP = vPPfacEff * expected(
            calc_vPPnext, IncShkDstn, args=(aNrmNow, Rfree)
        )
        EndOfPrdvPP += warm_glow.der(aNrmNow, order=2)
        dcda = EndOfPrdvPP / uFunc.der(np.array(cNrmNow), order=2)
        MPC = dcda / (dcda + 1.0)
        MPC_for_interpolation = np.insert(MPC, 0, MPCmaxNow)

        # Construct the unconstrained consumption function as a cubic interpolation
        cFuncNowUnc = CubicInterp(
            m_for_interpolation,
            c_for_interpolation,
            MPC_for_interpolation,
            cFuncLimitIntercept,
            cFuncLimitSlope,
        )
    else:
        # Construct the unconstrained consumption function as a linear interpolation
        cFuncNowUnc = LinearInterp(
            m_for_interpolation,
            c_for_interpolation,
            cFuncLimitIntercept,
            cFuncLimitSlope,
        )

    # Combine the constrained and unconstrained functions into the true consumption function.
    # LowerEnvelope should only be used when BoroCnstArt is True
    cFuncNow = LowerEnvelope(cFuncNowUnc, cFuncNowCnst, nan_bool=False)

    # Make the marginal value function
    vPfuncNow = MargValueFuncCRRA(cFuncNow, CRRA)

    # Define this period's marginal marginal value function
    if CubicBool:
        vPPfuncNow = MargMargValueFuncCRRA(cFuncNow, CRRA)
    else:
        vPPfuncNow = NullFunc()  # Dummy object

    # Construct this period's value function if requested
    if vFuncBool:
        # Calculate end-of-period value, its derivative, and their pseudo-inverse
        EndOfPrdv = DiscFacEff * expected(calc_vNext, IncShkDstn, args=(aNrmNow, Rfree))
        EndOfPrdv += warm_glow(aNrmNow)
        EndOfPrdvNvrs = uFunc.inv(EndOfPrdv)
        # value transformed through inverse utility
        EndOfPrdvNvrsP = EndOfPrdvP * uFunc.derinv(EndOfPrdv, order=(0, 1))
        EndOfPrdvNvrs = np.insert(EndOfPrdvNvrs, 0, 0.0)
        EndOfPrdvNvrsP = np.insert(EndOfPrdvNvrsP, 0, EndOfPrdvNvrsP[0])
        # This is a very good approximation, vNvrsPP = 0 at the asset minimum

        # Construct the end-of-period value function
        aNrm_temp = np.insert(aNrmNow, 0, BoroCnstNat)
        EndOfPrd_vNvrsFunc = CubicInterp(aNrm_temp, EndOfPrdvNvrs, EndOfPrdvNvrsP)
        EndOfPrd_vFunc = ValueFuncCRRA(EndOfPrd_vNvrsFunc, CRRA)

        # Compute expected value and marginal value on a grid of market resources
        mNrm_temp = mNrmMinNow + aXtraGrid
        cNrm_temp = cFuncNow(mNrm_temp)
        aNrm_temp = mNrm_temp - cNrm_temp
        v_temp = uFunc(cNrm_temp) + EndOfPrd_vFunc(aNrm_temp)
        vP_temp = uFunc.der(cNrm_temp)

        # Construct the beginning-of-period value function
        vNvrs_temp = uFunc.inv(v_temp)  # value transformed through inv utility
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


def solve_one_period_ConsPortfolioWarmGlow(
    solution_next,
    IncShkDstn,
    RiskyDstn,
    LivPrb,
    DiscFac,
    CRRA,
    Rfree,
    PermGroFac,
    BoroCnstArt,
    aXtraGrid,
    ShareGrid,
    AdjustPrb,
    ShareLimit,
    vFuncBool,
    DiscreteShareBool,
    BeqCRRA,
    BeqFac,
    BeqShift,
):
    """
    Solve one period of a consumption-saving problem with portfolio allocation
    between a riskless and risky asset. This function handles various sub-cases
    or variations on the problem, including the possibility that the agent does
    not necessarily get to update their portfolio share in every period, or that
    they must choose a discrete rather than continuous risky share.

    Parameters
    ----------
    solution_next : PortfolioSolution
        Solution to next period's problem.
    ShockDstn : Distribution
        Joint distribution of permanent income shocks, transitory income shocks,
        and risky returns.  This is only used if the input IndepDstnBool is False,
        indicating that income and return distributions can't be assumed to be
        independent.
    IncShkDstn : Distribution
        Discrete distribution of permanent income shocks and transitory income
        shocks. This is only used if the input IndepDstnBool is True, indicating
        that income and return distributions are independent.
    RiskyDstn : Distribution
       Distribution of risky asset returns. This is only used if the input
       IndepDstnBool is True, indicating that income and return distributions
       are independent.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree : float
        Risk free interest factor on end-of-period assets.
    PermGroFac : float
        Expected permanent income growth factor at the end of this period.
    BoroCnstArt: float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.  In this model, it is *required* to be zero.
    aXtraGrid: np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.
    ShareGrid : np.array
        Array of risky portfolio shares on which to define the interpolation
        of the consumption function when Share is fixed. Also used when the
        risky share choice is specified as discrete rather than continuous.
    AdjustPrb : float
        Probability that the agent will be able to update his portfolio share.
    ShareLimit : float
        Limiting lower bound of risky portfolio share as mNrm approaches infinity.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.
    DiscreteShareBool : bool
        Indicator for whether risky portfolio share should be optimized on the
        continuous [0,1] interval using the FOC (False), or instead only selected
        from the discrete set of values in ShareGrid (True).  If True, then
        vFuncBool must also be True.
    IndepDstnBool : bool
        Indicator for whether the income and risky return distributions are in-
        dependent of each other, which can speed up the expectations step.
    BeqCRRA : float
        Coefficient of relative risk aversion for warm glow bequest motive.
    BeqFac : float
        Multiplicative intensity factor for the warm glow bequest motive.
    BeqShift : float
        Stone-Geary shifter in the warm glow bequest motive.

    Returns
    -------
    solution_now : PortfolioSolution
        Solution to this period's problem.
    """
    # Make sure the individual is liquidity constrained.  Allowing a consumer to
    # borrow *and* invest in an asset with unbounded (negative) returns is a bad mix.
    if BoroCnstArt != 0.0:
        raise ValueError("PortfolioConsumerType must have BoroCnstArt=0.0!")

    # Make sure that if risky portfolio share is optimized only discretely, then
    # the value function is also constructed (else this task would be impossible).
    if DiscreteShareBool and (not vFuncBool):
        raise ValueError(
            "PortfolioConsumerType requires vFuncBool to be True when DiscreteShareBool is True!"
        )

    # Define the current period utility function and effective discount factor
    uFunc = UtilityFuncCRRA(CRRA)
    DiscFacEff = DiscFac * LivPrb  # "effective" discount factor
    BeqFacEff = (1.0 - LivPrb) * BeqFac  # "effective" bequest factor
    warm_glow = UtilityFuncStoneGeary(BeqCRRA, BeqFacEff, BeqShift)

    # Unpack next period's solution for easier access
    vPfuncAdj_next = solution_next.vPfuncAdj
    dvdmFuncFxd_next = solution_next.dvdmFuncFxd
    dvdsFuncFxd_next = solution_next.dvdsFuncFxd
    vFuncAdj_next = solution_next.vFuncAdj
    vFuncFxd_next = solution_next.vFuncFxd

    # Set a flag for whether the natural borrowing constraint is zero, which
    # depends on whether the smallest transitory income shock is zero
    BoroCnstNat_iszero = (np.min(IncShkDstn.atoms[1]) == 0.0) or (
        BeqFac != 0.0 and BeqShift == 0.0
    )

    # Prepare to calculate end-of-period marginal values by creating an array
    # of market resources that the agent could have next period, considering
    # the grid of end-of-period assets and the distribution of shocks he might
    # experience next period.

    # Unpack the risky return shock distribution
    Risky_next = RiskyDstn.atoms
    RiskyMax = np.max(Risky_next)
    RiskyMin = np.min(Risky_next)

    # bNrm represents R*a, balances after asset return shocks but before income.
    # This just uses the highest risky return as a rough shifter for the aXtraGrid.
    if BoroCnstNat_iszero:
        aNrmGrid = aXtraGrid
        bNrmGrid = np.insert(RiskyMax * aXtraGrid, 0, RiskyMin * aXtraGrid[0])
    else:
        # Add an asset point at exactly zero
        aNrmGrid = np.insert(aXtraGrid, 0, 0.0)
        bNrmGrid = RiskyMax * np.insert(aXtraGrid, 0, 0.0)

    # Get grid and shock sizes, for easier indexing
    aNrmCount = aNrmGrid.size
    ShareCount = ShareGrid.size

    # Make tiled arrays to calculate future realizations of mNrm and Share when integrating over IncShkDstn
    bNrmNext, ShareNext = np.meshgrid(bNrmGrid, ShareGrid, indexing="ij")

    # Define functions that are used internally to evaluate future realizations
    def calc_mNrm_next(S, b):
        """
        Calculate future realizations of market resources mNrm from the income
        shock distribution S and normalized bank balances b.
        """
        return b / (S["PermShk"] * PermGroFac) + S["TranShk"]

    def calc_dvdm_next(S, b, z):
        """
        Evaluate realizations of marginal value of market resources next period,
        based on the income distribution S, values of bank balances bNrm, and
        values of the risky share z.
        """
        mNrm_next = calc_mNrm_next(S, b)
        dvdmAdj_next = vPfuncAdj_next(mNrm_next)

        if AdjustPrb < 1.0:
            # Expand to the same dimensions as mNrm
            Share_next_expanded = z + np.zeros_like(mNrm_next)
            dvdmFxd_next = dvdmFuncFxd_next(mNrm_next, Share_next_expanded)
            # Combine by adjustment probability
            dvdm_next = AdjustPrb * dvdmAdj_next + (1.0 - AdjustPrb) * dvdmFxd_next
        else:  # Don't bother evaluating if there's no chance that portfolio share is fixed
            dvdm_next = dvdmAdj_next

        dvdm_next = (S["PermShk"] * PermGroFac) ** (-CRRA) * dvdm_next
        return dvdm_next

    def calc_dvds_next(S, b, z):
        """
        Evaluate realizations of marginal value of risky share next period, based
        on the income distribution S, values of bank balances bNrm, and values of
        the risky share z.
        """
        mNrm_next = calc_mNrm_next(S, b)

        # No marginal value of Share if it's a free choice!
        dvdsAdj_next = np.zeros_like(mNrm_next)

        if AdjustPrb < 1.0:
            # Expand to the same dimensions as mNrm
            Share_next_expanded = z + np.zeros_like(mNrm_next)
            dvdsFxd_next = dvdsFuncFxd_next(mNrm_next, Share_next_expanded)
            # Combine by adjustment probability
            dvds_next = AdjustPrb * dvdsAdj_next + (1.0 - AdjustPrb) * dvdsFxd_next
        else:  # Don't bother evaluating if there's no chance that portfolio share is fixed
            dvds_next = dvdsAdj_next

        dvds_next = (S["PermShk"] * PermGroFac) ** (1.0 - CRRA) * dvds_next
        return dvds_next

    # Calculate end-of-period marginal value of assets and shares at each point
    # in aNrm and ShareGrid. Does so by taking expectation of next period marginal
    # values across income and risky return shocks.

    # Calculate intermediate marginal value of bank balances by taking expectations over income shocks
    dvdb_intermed = expected(calc_dvdm_next, IncShkDstn, args=(bNrmNext, ShareNext))
    dvdbNvrs_intermed = uFunc.derinv(dvdb_intermed, order=(1, 0))
    dvdbNvrsFunc_intermed = BilinearInterp(dvdbNvrs_intermed, bNrmGrid, ShareGrid)
    dvdbFunc_intermed = MargValueFuncCRRA(dvdbNvrsFunc_intermed, CRRA)

    # Calculate intermediate marginal value of risky portfolio share by taking expectations over income shocks
    dvds_intermed = expected(calc_dvds_next, IncShkDstn, args=(bNrmNext, ShareNext))
    dvdsFunc_intermed = BilinearInterp(dvds_intermed, bNrmGrid, ShareGrid)

    # Make tiled arrays to calculate future realizations of bNrm and Share when integrating over RiskyDstn
    aNrmNow, ShareNext = np.meshgrid(aNrmGrid, ShareGrid, indexing="ij")

    # Define functions for calculating end-of-period marginal value
    def calc_EndOfPrd_dvda(S, a, z):
        """
        Compute end-of-period marginal value of assets at values a, conditional
        on risky asset return S and risky share z.
        """
        # Calculate future realizations of bank balances bNrm
        Rxs = S - Rfree  # Excess returns
        Rport = Rfree + z * Rxs  # Portfolio return
        bNrm_next = Rport * a

        # Ensure shape concordance
        z_rep = z + np.zeros_like(bNrm_next)

        # Calculate and return dvda
        EndOfPrd_dvda = Rport * dvdbFunc_intermed(bNrm_next, z_rep)
        return EndOfPrd_dvda

    def EndOfPrddvds_dist(S, a, z):
        """
        Compute end-of-period marginal value of risky share at values a, conditional
        on risky asset return S and risky share z.
        """
        # Calculate future realizations of bank balances bNrm
        Rxs = S - Rfree  # Excess returns
        Rport = Rfree + z * Rxs  # Portfolio return
        bNrm_next = Rport * a

        # Make the shares match the dimension of b, so that it can be vectorized
        z_rep = z + np.zeros_like(bNrm_next)

        # Calculate and return dvds
        EndOfPrd_dvds = Rxs * a * dvdbFunc_intermed(
            bNrm_next, z_rep
        ) + dvdsFunc_intermed(bNrm_next, z_rep)
        return EndOfPrd_dvds

    # Evaluate realizations of value and marginal value after asset returns are realized

    # Calculate end-of-period marginal value of assets by taking expectations
    EndOfPrd_dvda = DiscFacEff * expected(
        calc_EndOfPrd_dvda, RiskyDstn, args=(aNrmNow, ShareNext)
    )
    warm_glow_der = warm_glow.der(aNrmNow)
    EndOfPrd_dvda += np.where(np.isnan(warm_glow_der), 0.0, warm_glow_der)
    EndOfPrd_dvdaNvrs = uFunc.derinv(EndOfPrd_dvda)

    # Calculate end-of-period marginal value of risky portfolio share by taking expectations
    EndOfPrd_dvds = DiscFacEff * expected(
        EndOfPrddvds_dist, RiskyDstn, args=(aNrmNow, ShareNext)
    )

    # Make the end-of-period value function if the value function is requested
    if vFuncBool:

        def calc_v_intermed(S, b, z):
            """
            Calculate "intermediate" value from next period's bank balances, the
            income shocks S, and the risky asset share.
            """
            mNrm_next = calc_mNrm_next(S, b)

            vAdj_next = vFuncAdj_next(mNrm_next)
            if AdjustPrb < 1.0:
                vFxd_next = vFuncFxd_next(mNrm_next, z)
                # Combine by adjustment probability
                v_next = AdjustPrb * vAdj_next + (1.0 - AdjustPrb) * vFxd_next
            else:  # Don't bother evaluating if there's no chance that portfolio share is fixed
                v_next = vAdj_next

            v_intermed = (S["PermShk"] * PermGroFac) ** (1.0 - CRRA) * v_next
            return v_intermed

        # Calculate intermediate value by taking expectations over income shocks
        v_intermed = expected(calc_v_intermed, IncShkDstn, args=(bNrmNext, ShareNext))

        # Construct the "intermediate value function" for this period
        vNvrs_intermed = uFunc.inv(v_intermed)
        vNvrsFunc_intermed = BilinearInterp(vNvrs_intermed, bNrmGrid, ShareGrid)
        vFunc_intermed = ValueFuncCRRA(vNvrsFunc_intermed, CRRA)

        def calc_EndOfPrd_v(S, a, z):
            # Calculate future realizations of bank balances bNrm
            Rxs = S - Rfree
            Rport = Rfree + z * Rxs
            bNrm_next = Rport * a

            # Make an extended share_next of the same dimension as b_nrm so
            # that the function can be vectorized
            z_rep = z + np.zeros_like(bNrm_next)

            EndOfPrd_v = vFunc_intermed(bNrm_next, z_rep)
            return EndOfPrd_v

        # Calculate end-of-period value by taking expectations
        EndOfPrd_v = DiscFacEff * expected(
            calc_EndOfPrd_v, RiskyDstn, args=(aNrmNow, ShareNext)
        )
        EndOfPrd_v += warm_glow(aNrmNow)
        EndOfPrd_vNvrs = uFunc.inv(EndOfPrd_v)

        # Now make an end-of-period value function over aNrm and Share
        EndOfPrd_vNvrsFunc = BilinearInterp(EndOfPrd_vNvrs, aNrmGrid, ShareGrid)
        EndOfPrd_vFunc = ValueFuncCRRA(EndOfPrd_vNvrsFunc, CRRA)
        # This will be used later to make the value function for this period

    # Find the optimal risky asset share either by choosing the best value among
    # the discrete grid choices, or by satisfying the FOC with equality (continuous)
    if DiscreteShareBool:
        # If we're restricted to discrete choices, then portfolio share is
        # the one with highest value for each aNrm gridpoint
        opt_idx = np.argmax(EndOfPrd_v, axis=1)
        ShareAdj_now = ShareGrid[opt_idx]

        # Take cNrm at that index as well... and that's it!
        cNrmAdj_now = EndOfPrd_dvdaNvrs[np.arange(aNrmCount), opt_idx]

    else:
        # Now find the optimal (continuous) risky share on [0,1] by solving the first
        # order condition EndOfPrd_dvds == 0.
        FOC_s = EndOfPrd_dvds  # Relabel for convenient typing

        # For each value of aNrm, find the value of Share such that FOC_s == 0
        crossing = np.logical_and(FOC_s[:, 1:] <= 0.0, FOC_s[:, :-1] >= 0.0)
        share_idx = np.argmax(crossing, axis=1)
        # This represents the index of the segment of the share grid where dvds flips
        # from positive to negative, indicating that there's a zero *on* the segment

        # Calculate the fractional distance between those share gridpoints where the
        # zero should be found, assuming a linear function; call it alpha
        a_idx = np.arange(aNrmCount)
        bot_s = ShareGrid[share_idx]
        top_s = ShareGrid[share_idx + 1]
        bot_f = FOC_s[a_idx, share_idx]
        top_f = FOC_s[a_idx, share_idx + 1]
        bot_c = EndOfPrd_dvdaNvrs[a_idx, share_idx]
        top_c = EndOfPrd_dvdaNvrs[a_idx, share_idx + 1]
        alpha = 1.0 - top_f / (top_f - bot_f)

        # Calculate the continuous optimal risky share and optimal consumption
        ShareAdj_now = (1.0 - alpha) * bot_s + alpha * top_s
        cNrmAdj_now = (1.0 - alpha) * bot_c + alpha * top_c

        # If agent wants to put more than 100% into risky asset, he is constrained.
        # Likewise if he wants to put less than 0% into risky asset, he is constrained.
        constrained_top = FOC_s[:, -1] > 0.0
        constrained_bot = FOC_s[:, 0] < 0.0

        # Apply those constraints to both risky share and consumption (but lower
        # constraint should never be relevant)
        ShareAdj_now[constrained_top] = 1.0
        ShareAdj_now[constrained_bot] = 0.0
        cNrmAdj_now[constrained_top] = EndOfPrd_dvdaNvrs[constrained_top, -1]
        cNrmAdj_now[constrained_bot] = EndOfPrd_dvdaNvrs[constrained_bot, 0]

    # When the natural borrowing constraint is *not* zero, then aNrm=0 is in the
    # grid, but there's no way to "optimize" the portfolio if a=0, and consumption
    # can't depend on the risky share if it doesn't meaningfully exist. Apply
    # a small fix to the bottom gridpoint (aNrm=0) when this happens.
    if not BoroCnstNat_iszero:
        ShareAdj_now[0] = 1.0
        cNrmAdj_now[0] = EndOfPrd_dvdaNvrs[0, -1]

    # Construct functions characterizing the solution for this period

    # Calculate the endogenous mNrm gridpoints when the agent adjusts his portfolio,
    # then construct the consumption function when the agent can adjust his share
    mNrmAdj_now = np.insert(aNrmGrid + cNrmAdj_now, 0, 0.0)
    cNrmAdj_now = np.insert(cNrmAdj_now, 0, 0.0)
    cFuncAdj_now = LinearInterp(mNrmAdj_now, cNrmAdj_now)

    # Construct the marginal value (of mNrm) function when the agent can adjust
    vPfuncAdj_now = MargValueFuncCRRA(cFuncAdj_now, CRRA)

    # Construct the consumption function when the agent *can't* adjust the risky
    # share, as well as the marginal value of Share function
    cFuncFxd_by_Share = []
    dvdsFuncFxd_by_Share = []
    for j in range(ShareCount):
        cNrmFxd_temp = np.insert(EndOfPrd_dvdaNvrs[:, j], 0, 0.0)
        mNrmFxd_temp = np.insert(aNrmGrid + cNrmFxd_temp[1:], 0, 0.0)
        dvdsFxd_temp = np.insert(EndOfPrd_dvds[:, j], 0, EndOfPrd_dvds[0, j])
        cFuncFxd_by_Share.append(LinearInterp(mNrmFxd_temp, cNrmFxd_temp))
        dvdsFuncFxd_by_Share.append(LinearInterp(mNrmFxd_temp, dvdsFxd_temp))
    cFuncFxd_now = LinearInterpOnInterp1D(cFuncFxd_by_Share, ShareGrid)
    dvdsFuncFxd_now = LinearInterpOnInterp1D(dvdsFuncFxd_by_Share, ShareGrid)

    # The share function when the agent can't adjust his portfolio is trivial
    ShareFuncFxd_now = IdentityFunction(i_dim=1, n_dims=2)

    # Construct the marginal value of mNrm function when the agent can't adjust his share
    dvdmFuncFxd_now = MargValueFuncCRRA(cFuncFxd_now, CRRA)

    # Construct the optimal risky share function when adjusting is possible.
    # The interpolation method depends on whether the choice is discrete or continuous.
    if DiscreteShareBool:
        # If the share choice is discrete, the "interpolated" share function acts
        # like a step function, with jumps at the midpoints of mNrm gridpoints.
        # Because an actual step function would break our (assumed continuous) linear
        # interpolator, there's a *tiny* region with extremely high slope.
        mNrmAdj_mid = (mNrmAdj_now[2:] + mNrmAdj_now[1:-1]) / 2
        mNrmAdj_plus = mNrmAdj_mid * (1.0 + 1e-12)
        mNrmAdj_comb = (np.transpose(np.vstack((mNrmAdj_mid, mNrmAdj_plus)))).flatten()
        mNrmAdj_comb = np.append(np.insert(mNrmAdj_comb, 0, 0.0), mNrmAdj_now[-1])
        Share_comb = (np.transpose(np.vstack((ShareAdj_now, ShareAdj_now)))).flatten()
        ShareFuncAdj_now = LinearInterp(mNrmAdj_comb, Share_comb)

    else:
        # If the share choice is continuous, just make an ordinary interpolating function
        if BoroCnstNat_iszero:
            Share_lower_bound = ShareLimit
        else:
            Share_lower_bound = 1.0
        ShareAdj_now = np.insert(ShareAdj_now, 0, Share_lower_bound)
        ShareFuncAdj_now = LinearInterp(mNrmAdj_now, ShareAdj_now, ShareLimit, 0.0)

    # Add the value function if requested
    if vFuncBool:
        # Create the value functions for this period, defined over market resources
        # mNrm when agent can adjust his portfolio, and over market resources and
        # fixed share when agent can not adjust his portfolio.

        # Construct the value function when the agent can adjust his portfolio
        mNrm_temp = aXtraGrid  # Just use aXtraGrid as our grid of mNrm values
        cNrm_temp = cFuncAdj_now(mNrm_temp)
        aNrm_temp = mNrm_temp - cNrm_temp
        Share_temp = ShareFuncAdj_now(mNrm_temp)
        v_temp = uFunc(cNrm_temp) + EndOfPrd_vFunc(aNrm_temp, Share_temp)
        vNvrs_temp = uFunc.inv(v_temp)
        vNvrsP_temp = uFunc.der(cNrm_temp) * uFunc.inverse(v_temp, order=(0, 1))
        vNvrsFuncAdj = CubicInterp(
            np.insert(mNrm_temp, 0, 0.0),  # x_list
            np.insert(vNvrs_temp, 0, 0.0),  # f_list
            np.insert(vNvrsP_temp, 0, vNvrsP_temp[0]),  # dfdx_list
        )
        # Re-curve the pseudo-inverse value function
        vFuncAdj_now = ValueFuncCRRA(vNvrsFuncAdj, CRRA)

        # Construct the value function when the agent *can't* adjust his portfolio
        mNrm_temp, Share_temp = np.meshgrid(aXtraGrid, ShareGrid)
        cNrm_temp = cFuncFxd_now(mNrm_temp, Share_temp)
        aNrm_temp = mNrm_temp - cNrm_temp
        v_temp = uFunc(cNrm_temp) + EndOfPrd_vFunc(aNrm_temp, Share_temp)
        vNvrs_temp = uFunc.inv(v_temp)
        vNvrsP_temp = uFunc.der(cNrm_temp) * uFunc.inverse(v_temp, order=(0, 1))
        vNvrsFuncFxd_by_Share = []
        for j in range(ShareCount):
            vNvrsFuncFxd_by_Share.append(
                CubicInterp(
                    np.insert(mNrm_temp[:, 0], 0, 0.0),  # x_list
                    np.insert(vNvrs_temp[:, j], 0, 0.0),  # f_list
                    np.insert(vNvrsP_temp[:, j], 0, vNvrsP_temp[j, 0]),  # dfdx_list
                )
            )
        vNvrsFuncFxd = LinearInterpOnInterp1D(vNvrsFuncFxd_by_Share, ShareGrid)
        vFuncFxd_now = ValueFuncCRRA(vNvrsFuncFxd, CRRA)

    else:  # If vFuncBool is False, fill in dummy values
        vFuncAdj_now = NullFunc()
        vFuncFxd_now = NullFunc()

    # Package and return the solution
    solution_now = PortfolioSolution(
        cFuncAdj=cFuncAdj_now,
        ShareFuncAdj=ShareFuncAdj_now,
        vPfuncAdj=vPfuncAdj_now,
        vFuncAdj=vFuncAdj_now,
        cFuncFxd=cFuncFxd_now,
        ShareFuncFxd=ShareFuncFxd_now,
        dvdmFuncFxd=dvdmFuncFxd_now,
        dvdsFuncFxd=dvdsFuncFxd_now,
        vFuncFxd=vFuncFxd_now,
        AdjPrb=AdjustPrb,
    )
    return solution_now


###############################################################################

# Make a dictionary of constructors for the warm glow bequest model
warmglow_constructor_dict = {
    "IncShkDstn": construct_lognormal_income_process_unemployment,
    "PermShkDstn": get_PermShkDstn_from_IncShkDstn,
    "TranShkDstn": get_TranShkDstn_from_IncShkDstn,
    "aXtraGrid": make_assets_grid,
    "solution_terminal": make_bequest_solution_terminal,
    "kNrmInitDstn": make_lognormal_kNrm_init_dstn,
    "pLvlInitDstn": make_lognormal_pLvl_init_dstn,
}

# Make a dictionary with parameters for the default constructor for kNrmInitDstn
default_kNrmInitDstn_params = {
    "kLogInitMean": -12.0,  # Mean of log initial capital
    "kLogInitStd": 0.0,  # Stdev of log initial capital
    "kNrmInitCount": 15,  # Number of points in initial capital discretization
}

# Make a dictionary with parameters for the default constructor for pLvlInitDstn
default_pLvlInitDstn_params = {
    "pLogInitMean": 0.0,  # Mean of log permanent income
    "pLogInitStd": 0.0,  # Stdev of log permanent income
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
    "aXtraMax": 20,  # Maximum end-of-period "assets above minimum" value
    "aXtraNestFac": 3,  # Exponential nesting factor for aXtraGrid
    "aXtraCount": 48,  # Number of points in the grid of "assets above minimum"
    "aXtraExtra": None,  # Additional other values to add in grid (optional)
}

# Make a dictionary to specify awarm glow bequest consumer type
init_warm_glow = {
    # BASIC HARK PARAMETERS REQUIRED TO SOLVE THE MODEL
    "cycles": 1,  # Finite, non-cyclic model
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    "constructors": warmglow_constructor_dict,  # See dictionary above
    # PRIMITIVE RAW PARAMETERS REQUIRED TO SOLVE THE MODEL
    "CRRA": 2.0,  # Coefficient of relative risk aversion on consumption
    "Rfree": [1.03],  # Interest factor on retained assets
    "DiscFac": 0.96,  # Intertemporal discount factor
    "LivPrb": [0.98],  # Survival probability after each period
    "PermGroFac": [1.01],  # Permanent income growth factor
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint
    "BeqCRRA": 2.0,  # Coefficient of relative risk aversion for bequest motive
    "BeqFac": 40.0,  # Scaling factor for bequest motive
    "BeqShift": 0.0,  # Stone-Geary shifter term for bequest motive
    "BeqCRRATerm": 2.0,  # Coefficient of relative risk aversion for bequest motive, terminal period only
    "BeqFacTerm": 40.0,  # Scaling factor for bequest motive, terminal period only
    "BeqShiftTerm": 0.0,  # Stone-Geary shifter term for bequest motive, terminal period only
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
init_warm_glow.update(default_IncShkDstn_params)
init_warm_glow.update(default_aXtraGrid_params)
init_warm_glow.update(default_kNrmInitDstn_params)
init_warm_glow.update(default_pLvlInitDstn_params)

# Make a dictionary with bequest motives turned off
init_accidental_bequest = init_warm_glow.copy()
init_accidental_bequest["BeqFac"] = 0.0
init_accidental_bequest["BeqShift"] = 0.0
init_accidental_bequest["BeqFacTerm"] = 0.0
init_accidental_bequest["BeqShiftTerm"] = 0.0

# Make a dictionary that has *only* a terminal period bequest
init_warm_glow_terminal_only = init_warm_glow.copy()
init_warm_glow_terminal_only["BeqFac"] = 0.0
init_warm_glow_terminal_only["BeqShift"] = 0.0


class BequestWarmGlowConsumerType(IndShockConsumerType):
    r"""
    A consumer type with based on IndShockConsumerType, with an additional bequest motive.
    They gain utility for any wealth they leave when they die, according to a Stone-Geary utility.

    .. math::
        \newcommand{\CRRA}{\rho}
        \newcommand{\DiePrb}{\mathsf{D}}
        \newcommand{\PermGroFac}{\Gamma}
        \newcommand{\Rfree}{\mathsf{R}}
        \newcommand{\DiscFac}{\beta}
        \begin{align*}
        v_t(m_t) &= \max_{c_t}u(c_t) + \DiePrb_{t+1} u_{Beq}(a_t)+\DiscFac (1 - \DiePrb_{t+1}) \mathbb{E}_{t} \left[ (\PermGroFac_{t+1} \psi_{t+1})^{1-\CRRA} v_{t+1}(m_{t+1}) \right], \\
        & \text{s.t.}  \\
        a_t &= m_t - c_t, \\
        a_t &\geq \underline{a}, \\
        m_{t+1} &= a_t \Rfree_{t+1}/(\PermGroFac_{t+1} \psi_{t+1}) + \theta_{t+1}, \\
        (\psi_{t+1},\theta_{t+1}) &\sim F_{t+1}, \\
        \mathbb{E}[\psi]=\mathbb{E}[\theta] &= 1, \\
        u(c) &= \frac{c^{1-\CRRA}}{1-\CRRA} \\
        u_{Beq} (a) &= \textbf{BeqFac} \frac{(a+\textbf{BeqShift})^{1-\CRRA_{Beq}}}{1-\CRRA_{Beq}} \\
        \end{align*}


    Constructors
    ------------
    IncShkDstn: Constructor, :math:`\psi`, :math:`\theta`
        The agent's income shock distributions.

        It's default constructor is :func:`HARK.Calibration.Income.IncomeProcesses.construct_lognormal_income_process_unemployment`
    aXtraGrid: Constructor
        The agent's asset grid.

        It's default constructor is :func:`HARK.utilities.make_assets_grid`

    Solving Parameters
    ------------------
    cycles: int
        0 specifies an infinite horizon model, 1 specifies a finite model.
    T_cycle: int
        Number of periods in the cycle for this agent type.
    CRRA: float, :math:`\rho`
        Coefficient of Relative Risk Aversion.
    BeqCRRA: float, :math:`\rho_{Beq}`
        Coefficient of Relative Risk Aversion for the bequest motive.
        If this value isn't the same as CRRA, then the model can only be represented as a Bellman equation.
        This may cause unintented behavior.
    BeqCRRATerm: float, :math:`\rho_{Beq}`
        The Coefficient of Relative Risk Aversion for the bequest motive, but only in the terminal period.
        In most cases this should be the same as beqCRRA.
    BeqShift: float, :math:`\textbf{BeqShift}`
        The Shift term from the bequest motive's utility function.
        If this value isn't 0, then the model can only be represented as a Bellman equation.
        This may cause unintented behavior.
    BeqShiftTerm: float, :math:`\textbf{BeqShift}`
        The shift term from the bequest motive's utility function, in the terminal period.
        In most cases this should be the same as beqShift
    BeqFac: float, :math:`\textbf{BeqFac}`
        The weight for the bequest's utility function.
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
        For this agent, the options are 'PermShk', 'TranShk', 'aLvl', 'aNrm', 'bNrm', 'cNrm', 'mNrm', 'pLvl', and 'who_dies'.

        PermShk is the agent's permanent income shock

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

        Visit :class:`HARK.ConsumptionSaving.ConsIndShockModel.ConsumerSolution` for more information about the solution.
    history: Dict[Array]
        Created by running the :func:`.simulate()` method.
        Contains the variables in track_vars. Each item in the dictionary is an array with the shape (T_sim,AgentCount).
        Visit :class:`HARK.core.AgentType.simulate` for more information.
    """

    time_inv_ = IndShockConsumerType.time_inv_ + ["BeqCRRA", "BeqShift", "BeqFac"]
    default_ = {
        "params": init_accidental_bequest,
        "solver": solve_one_period_ConsWarmBequest,
        "model": "ConsIndShock.yaml",
    }

    def pre_solve(self):
        self.construct("solution_terminal")

    def check_conditions(self, verbose=None):
        raise NotImplementedError()  # pragma: nocover

    def calc_limiting_values(self):
        raise NotImplementedError()  # pragma: nocover


###############################################################################


# Make a dictionary of constructors for the portfolio choice consumer type
portfolio_bequest_constructor_dict = {
    "IncShkDstn": construct_lognormal_income_process_unemployment,
    "PermShkDstn": get_PermShkDstn_from_IncShkDstn,
    "TranShkDstn": get_TranShkDstn_from_IncShkDstn,
    "aXtraGrid": make_assets_grid,
    "RiskyDstn": make_lognormal_RiskyDstn,
    "ShockDstn": combine_IncShkDstn_and_RiskyDstn,
    "ShareLimit": calc_ShareLimit_for_CRRA,
    "ShareGrid": make_simple_ShareGrid,
    "AdjustDstn": make_AdjustDstn,
    "kNrmInitDstn": make_lognormal_kNrm_init_dstn,
    "pLvlInitDstn": make_lognormal_pLvl_init_dstn,
    "solution_terminal": make_warmglow_portfolio_solution_terminal,
}

# Make a dictionary with parameters for the default constructor for kNrmInitDstn
default_kNrmInitDstn_params = {
    "kLogInitMean": -12.0,  # Mean of log initial capital
    "kLogInitStd": 0.0,  # Stdev of log initial capital
    "kNrmInitCount": 15,  # Number of points in initial capital discretization
}

# Make a dictionary with parameters for the default constructor for pLvlInitDstn
default_pLvlInitDstn_params = {
    "pLogInitMean": 0.0,  # Mean of log permanent income
    "pLogInitStd": 0.0,  # Stdev of log permanent income
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
    "aXtraMax": 100,  # Maximum end-of-period "assets above minimum" value
    "aXtraNestFac": 1,  # Exponential nesting factor for aXtraGrid
    "aXtraCount": 200,  # Number of points in the grid of "assets above minimum"
    "aXtraExtra": None,  # Additional other values to add in grid (optional)
}

# Default parameters to make RiskyDstn with make_lognormal_RiskyDstn (and uniform ShareGrid)
default_RiskyDstn_and_ShareGrid_params = {
    "RiskyAvg": 1.08,  # Mean return factor of risky asset
    "RiskyStd": 0.18362634887,  # Stdev of log returns on risky asset
    "RiskyCount": 5,  # Number of integration nodes to use in approximation of risky returns
    "ShareCount": 25,  # Number of discrete points in the risky share approximation
}

# Make a dictionary to specify a risky asset consumer type
init_portfolio_bequest = {
    # BASIC HARK PARAMETERS REQUIRED TO SOLVE THE MODEL
    "cycles": 1,  # Finite, non-cyclic model
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    "constructors": portfolio_bequest_constructor_dict,  # See dictionary above
    # PRIMITIVE RAW PARAMETERS REQUIRED TO SOLVE THE MODEL
    "CRRA": 5.0,  # Coefficient of relative risk aversion
    "Rfree": [1.03],  # Return factor on risk free asset
    "DiscFac": 0.90,  # Intertemporal discount factor
    "LivPrb": [0.98],  # Survival probability after each period
    "PermGroFac": [1.01],  # Permanent income growth factor
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint
    "BeqCRRA": 2.0,  # Coefficient of relative risk aversion for bequest motive
    "BeqFac": 40.0,  # Scaling factor for bequest motive
    "BeqShift": 0.0,  # Stone-Geary shifter term for bequest motive
    "BeqCRRATerm": 2.0,  # Coefficient of relative risk aversion for bequest motive, terminal period only
    "BeqFacTerm": 40.0,  # Scaling factor for bequest motive, terminal period only
    "BeqShiftTerm": 0.0,  # Stone-Geary shifter term for bequest motive, terminal period only
    "DiscreteShareBool": False,  # Whether risky asset share is restricted to discrete values
    "vFuncBool": False,  # Whether to calculate the value function during solution
    "CubicBool": False,  # Whether to use cubic spline interpolation when True
    # (Uses linear spline interpolation for cFunc when False)
    "IndepDstnBool": True,  # Indicator for whether return & income shocks are independent
    "PortfolioBool": True,  # Whether this agent has portfolio choice
    "PortfolioBisect": False,  # What does this do?
    "AdjustPrb": 1.0,  # Probability that the agent can update their risky portfolio share each period
    "RiskyShareFixed": None,  # This just needs to exist because of inheritance, does nothing
    "sim_common_Rrisky": True,  # Whether risky returns have a shared/common value across agents
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
init_portfolio_bequest.update(default_kNrmInitDstn_params)
init_portfolio_bequest.update(default_pLvlInitDstn_params)
init_portfolio_bequest.update(default_IncShkDstn_params)
init_portfolio_bequest.update(default_aXtraGrid_params)
init_portfolio_bequest.update(default_RiskyDstn_and_ShareGrid_params)


class BequestWarmGlowPortfolioType(PortfolioConsumerType):
    r"""
    A consumer type with based on PortfolioConsumerType, with an additional bequest motive.
    They gain utility for any wealth they leave when they die, according to a Stone-Geary utility.

    .. math::
        \newcommand{\CRRA}{\rho}
        \newcommand{\DiePrb}{\mathsf{D}}
        \newcommand{\PermGroFac}{\Gamma}
        \newcommand{\Rfree}{\mathsf{R}}
        \newcommand{\DiscFac}{\beta}
        \begin{align*}
        v_t(m_t,S_t) &= \max_{c_t,S^{*}_t} u(c_t) + \DiePrb_{t+1} u_{Beq}(a_t)+ \DiscFac (1-\DiePrb_{t+1})  \mathbb{E}_{t} \left[(\PermGroFac_{t+1}\psi_{t+1})^{1-\CRRA} v_{t+1}(m_{t+1},S_{t+1}) \right], \\
        & \text{s.t.}  \\
        a_t &= m_t - c_t, \\
        a_t &\geq \underline{a}, \\
        m_{t+1} &= \mathsf{R}_{t+1}/(\PermGroFac_{t+1} \psi_{t+1}) a_t + \theta_{t+1}, \\
        \mathsf{R}_{t+1} &=S_t\phi_{t+1}\mathbf{R}_{t+1}+ (1-S_t)\mathsf{R}_{t+1}, \\
        S_{t+1} &= \begin{cases}
        S^{*}_t & \text{if } p_t < \wp\\
        S_t & \text{if } p_t \geq \wp,
        \end{cases}\\
        (\psi_{t+1},\theta_{t+1},\phi_{t+1},p_t) &\sim F_{t+1}, \\
        \mathbb{E}[\psi]=\mathbb{E}[\theta] &= 1. \\
        u(c) &= \frac{c^{1-\CRRA}}{1-\CRRA} \\
        u_{Beq} (a) &= \textbf{BeqFac} \frac{(a+\textbf{BeqShift})^{1-\CRRA_{Beq}}}{1-\CRRA_{Beq}} \\
        \end{align*}


    Constructors
    ------------
    IncShkDstn: Constructor, :math:`\psi`, :math:`\theta`
        The agent's income shock distributions.

        It's default constructor is :func:`HARK.Calibration.Income.IncomeProcesses.construct_lognormal_income_process_unemployment`
    aXtraGrid: Constructor
        The agent's asset grid.

        It's default constructor is :func:`HARK.utilities.make_assets_grid`
    ShareGrid: Constructor
        The agent's risky asset share grid

        It's default constructor is :func:`HARK.ConsumptionSaving.ConsRiskyAssetModel.make_simple_ShareGrid`
    RiskyDstn: Constructor, :math:`\phi`
        The agent's asset shock distribution for risky assets.

        It's default constructor is :func:`HARK.Calibration.Assets.AssetProcesses.make_lognormal_RiskyDstn`

    Solving Parameters
    ------------------
    cycles: int
        0 specifies an infinite horizon model, 1 specifies a finite model.
    T_cycle: int
        Number of periods in the cycle for this agent type.
    CRRA: float, :math:`\rho`
        Coefficient of Relative Risk Aversion.
    BeqCRRA: float, :math:`\rho_{Beq}`
        Coefficient of Relative Risk Aversion for the bequest motive.
        If this value isn't the same as CRRA, then the model can only be represented as a Bellman equation.
        This may cause unintented behavior.
    BeqCRRATerm: float, :math:`\rho_{Beq}`
        The Coefficient of Relative Risk Aversion for the bequest motive, but only in the terminal period.
        In most cases this should be the same as beqCRRA.
    BeqShift: float, :math:`\textbf{BeqShift}`
        The Shift term from the bequest motive's utility function.
        If this value isn't 0, then the model can only be represented as a Bellman equation.
        This may cause unintented behavior.
    BeqShiftTerm: float, :math:`\textbf{BeqShift}`
        The shift term from the bequest motive's utility function, in the terminal period.
        In most cases this should be the same as beqShift
    BeqFac: float, :math:`\textbf{BeqFac}`
        The weight for the bequest's utility function.
    Rfree: float or list[float], time varying, :math:`\mathsf{R}`
        Risk Free interest rate. Pass a list of floats to make Rfree time varying.
    DiscFac: float, :math:`\beta`
        Intertemporal discount factor.
    LivPrb: list[float], time varying, :math:`1-\mathsf{D}`
        Survival probability after each period.
    PermGroFac: list[float], time varying, :math:`\Gamma`
        Permanent income growth factor.
    BoroCnstArt: float, default=0.0, :math:`\underline{a}`
        The minimum Asset/Perminant Income ratio. for this agent, BoroCnstArt must be 0.
    vFuncBool: bool
        Whether to calculate the value function during solution.
    CubicBool: bool
        Whether to use cubic spline interpoliation.
    AdjustPrb: float or list[float], time varying
        Must be between 0 and 1. Probability that the agent can update their risky portfolio share each period. Pass a list of floats to make AdjustPrb time varying.

    Simulation Parameters
    ---------------------
    sim_common_Rrisky: Boolean
        Whether risky returns have a shared/common value across agents. If True, Risky return's can't be time varying.
    AgentCount: int
        Number of agents of this kind that are created during simulations.
    T_age: int
        Age after which to automatically kill agents, None to ignore.
    T_sim: int, required for simulation
        Number of periods to simulate.
    track_vars: list[strings]
        List of variables that should be tracked when running the simulation.
        For this agent, the options are 'Adjust', 'PermShk', 'Risky', 'TranShk', 'aLvl', 'aNrm', 'bNrm', 'cNrm', 'mNrm', 'pLvl', and 'who_dies'.

        Adjust is the array of which agents can adjust

        PermShk is the agent's permanent income shock

        Risky is the agent's risky asset shock

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

        Visit :class:`HARK.ConsumptionSaving.ConsPortfolioModel.PortfolioSolution` for more information about the solution.

    history: Dict[Array]
        Created by running the :func:`.simulate()` method.
        Contains the variables in track_vars. Each item in the dictionary is an array with the shape (T_sim,AgentCount).
        Visit :class:`HARK.core.AgentType.simulate` for more information.
    """

    time_inv_ = PortfolioConsumerType.time_inv_ + ["BeqCRRA", "BeqShift", "BeqFac"]
    default_ = {
        "params": init_portfolio_bequest,
        "solver": solve_one_period_ConsPortfolioWarmGlow,
        "model": "ConsRiskyAsset.yaml",
    }

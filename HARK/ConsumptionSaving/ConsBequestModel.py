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

from copy import deepcopy

import numpy as np

from HARK import NullFunc
from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsumerSolution,
    IndShockConsumerType,
    init_idiosyncratic_shocks,
)
from HARK.ConsumptionSaving.ConsPortfolioModel import (
    PortfolioConsumerType,
    PortfolioSolution,
    init_portfolio,
)
from HARK.distribution import expected
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


class BequestWarmGlowConsumerType(IndShockConsumerType):
    time_inv_ = IndShockConsumerType.time_inv_ + ["BeqCRRA", "BeqShift", "BeqFac"]

    def __init__(self, **kwds):
        params = init_accidental_bequest.copy()
        params.update(kwds)

        super().__init__(**params)

        self.solve_one_period = solve_one_period_ConsWarmBequest

    def update_solution_terminal(self):
        if self.BeqFacTerm == 0.0:  # No terminal bequest
            super().update_solution_terminal()
        else:
            utility = UtilityFuncCRRA(self.CRRA)

            warm_glow = UtilityFuncStoneGeary(
                self.BeqCRRATerm,
                factor=self.BeqFacTerm,
                shifter=self.BeqShiftTerm,
            )

            aNrmGrid = (
                np.append(0.0, self.aXtraGrid)
                if self.BeqShiftTerm != 0.0
                else self.aXtraGrid
            )
            cNrmGrid = utility.derinv(warm_glow.der(aNrmGrid))
            vGrid = utility(cNrmGrid) + warm_glow(aNrmGrid)
            cNrmGridW0 = np.append(0.0, cNrmGrid)
            mNrmGridW0 = np.append(0.0, aNrmGrid + cNrmGrid)
            vNvrsGridW0 = np.append(0.0, utility.inv(vGrid))

            cFunc_term = LinearInterp(mNrmGridW0, cNrmGridW0)
            vNvrsFunc_term = LinearInterp(mNrmGridW0, vNvrsGridW0)
            vFunc_term = ValueFuncCRRA(vNvrsFunc_term, self.CRRA)
            vPfunc_term = MargValueFuncCRRA(cFunc_term, self.CRRA)
            vPPfunc_term = MargMargValueFuncCRRA(cFunc_term, self.CRRA)

            self.solution_terminal.cFunc = cFunc_term
            self.solution_terminal.vFunc = vFunc_term
            self.solution_terminal.vPfunc = vPfunc_term
            self.solution_terminal.vPPfunc = vPPfunc_term
            self.solution_terminal.mNrmMin = 0.0


class BequestWarmGlowPortfolioType(PortfolioConsumerType):
    time_inv_ = PortfolioConsumerType.time_inv_ + ["BeqCRRA", "BeqShift", "BeqFac"]

    def __init__(self, **kwds):
        params = init_portfolio_bequest.copy()
        params.update(kwds)

        self.IndepDstnBool = True

        super().__init__(**params)

        self.solve_one_period = solve_one_period_ConsPortfolioWarmGlow

    def update_solution_terminal(self):
        if self.BeqFacTerm == 0.0:  # No terminal bequest
            super().update_solution_terminal()
        else:
            utility = UtilityFuncCRRA(self.CRRA)

            warm_glow = UtilityFuncStoneGeary(
                self.BeqCRRATerm,
                factor=self.BeqFacTerm,
                shifter=self.BeqShiftTerm,
            )

            aNrmGrid = (
                np.append(0.0, self.aXtraGrid)
                if self.BeqShiftTerm != 0.0
                else self.aXtraGrid
            )
            cNrmGrid = utility.derinv(warm_glow.der(aNrmGrid))
            vGrid = utility(cNrmGrid) + warm_glow(aNrmGrid)
            cNrmGridW0 = np.append(0.0, cNrmGrid)
            mNrmGridW0 = np.append(0.0, aNrmGrid + cNrmGrid)
            vNvrsGridW0 = np.append(0.0, utility.inv(vGrid))

            cFunc_term = LinearInterp(mNrmGridW0, cNrmGridW0)
            vNvrsFunc_term = LinearInterp(mNrmGridW0, vNvrsGridW0)
            vFunc_term = ValueFuncCRRA(vNvrsFunc_term, self.CRRA)
            vPfunc_term = MargValueFuncCRRA(cFunc_term, self.CRRA)
            vPPfunc_term = MargMargValueFuncCRRA(cFunc_term, self.CRRA)

            self.solution_terminal.cFunc = cFunc_term
            self.solution_terminal.vFunc = vFunc_term
            self.solution_terminal.vPfunc = vPfunc_term
            self.solution_terminal.vPPfunc = vPPfunc_term
            self.solution_terminal.mNrmMin = 0.0

            # Consume all market resources: c_T = m_T
            cFuncAdj_terminal = self.solution_terminal.cFunc
            cFuncFxd_terminal = lambda m, s: self.solution_terminal.cFunc(m)

            # Risky share is irrelevant-- no end-of-period assets; set to zero
            ShareFuncAdj_terminal = ConstantFunction(0.0)
            ShareFuncFxd_terminal = IdentityFunction(i_dim=1, n_dims=2)

            # Value function is simply utility from consuming market resources
            vFuncAdj_terminal = self.solution_terminal.vFunc
            vFuncFxd_terminal = lambda m, s: self.solution_terminal.vFunc(m)

            # Marginal value of market resources is marg utility at the consumption function
            vPfuncAdj_terminal = self.solution_terminal.vPfunc
            dvdmFuncFxd_terminal = lambda m, s: self.solution_terminal.vPfunc(m)
            # No future, no marg value of Share
            dvdsFuncFxd_terminal = ConstantFunction(0.0)

            # Construct the terminal period solution
            self.solution_terminal = PortfolioSolution(
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

    # This is a point at which (a,c,share) have consistent length. Take the
    # snapshot for storing the grid and values in the solution.
    save_points = {
        "a": deepcopy(aNrmGrid),
        "eop_dvda_adj": uFunc.der(cNrmAdj_now),
        "share_adj": deepcopy(ShareAdj_now),
        "share_grid": deepcopy(ShareGrid),
        "eop_dvda_fxd": uFunc.der(EndOfPrd_dvda),
        "eop_dvds_fxd": EndOfPrd_dvds,
    }

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
        # WHAT IS THIS STUFF FOR??
        aGrid=save_points["a"],
        Share_adj=save_points["share_adj"],
        EndOfPrddvda_adj=save_points["eop_dvda_adj"],
        ShareGrid=save_points["share_grid"],
        EndOfPrddvda_fxd=save_points["eop_dvda_fxd"],
        EndOfPrddvds_fxd=save_points["eop_dvds_fxd"],
    )
    return solution_now


init_accidental_bequest = init_idiosyncratic_shocks.copy()
init_accidental_bequest["BeqCRRA"] = init_idiosyncratic_shocks["CRRA"]
init_accidental_bequest["BeqFac"] = 0.0
init_accidental_bequest["BeqShift"] = 0.0
init_accidental_bequest["BeqCRRATerm"] = init_idiosyncratic_shocks["CRRA"]
init_accidental_bequest["BeqFacTerm"] = 0.0
init_accidental_bequest["BeqShiftTerm"] = 0.0

init_warm_glow_terminal_only = init_accidental_bequest.copy()
init_warm_glow_terminal_only["BeqCRRATerm"] = init_idiosyncratic_shocks["CRRA"]
init_warm_glow_terminal_only["BeqFacTerm"] = 40.0  # kid lives 40yr after bequest
init_warm_glow_terminal_only["BeqShiftTerm"] = 0.0

init_warm_glow = init_warm_glow_terminal_only.copy()
init_warm_glow["BeqCRRA"] = init_idiosyncratic_shocks["CRRA"]
init_warm_glow["BeqFac"] = 40.0
init_warm_glow["BeqShift"] = 0.0

init_portfolio_bequest = init_accidental_bequest.copy()
init_portfolio_bequest.update(init_portfolio)

"""Classes to solve consumption-saving models with a bequest motive and
idiosyncratic shocks to income and wealth. All models here assume
separable CRRA utility of consumption and Stone-Geary utility of
savings with geometric discounting of the continuation value and
shocks to income that have transitory and/or permanent components.

It currently solves 2 types of models:
    1) A standard lifecycle model with a terminal and/or accidental bequest motive.
    3) A portfolio choice model with a terminal and/or accidental bequest motive.
"""

import numpy as np
from HARK import NullFunc
from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsIndShockSolver,
    ConsumerSolution,
    IndShockConsumerType,
    init_idiosyncratic_shocks,
    init_lifecycle,
)
from HARK.ConsumptionSaving.ConsPortfolioModel import (
    ConsPortfolioSolver,
    PortfolioConsumerType,
    PortfolioSolution,
    init_portfolio,
)
from HARK.core import make_one_period_oo_solver
from HARK.distribution import expected
from HARK.interpolation import (
    ConstantFunction,
    IdentityFunction,
    LinearInterp,
    LowerEnvelope,
    MargMargValueFuncCRRA,
    MargValueFuncCRRA,
    ValueFuncCRRA,
)
from HARK.rewards import UtilityFuncCRRA, UtilityFuncStoneGeary


class BequestWarmGlowConsumerType(IndShockConsumerType):
    time_inv_ = IndShockConsumerType.time_inv_ + [
        "BeqCRRA",
        "BeqShift",
    ]

    time_vary_ = IndShockConsumerType.time_vary_ + [
        "BeqFac",
    ]

    def __init__(self, **kwds):
        params = init_wealth_in_utility.copy()
        params.update(kwds)

        super().__init__(**params)

        self.solve_one_period = solve_one_period_ConsWarmBequest

    def update(self):
        super().update()
        self.update_parameters()

    def update_parameters(self):
        if not isinstance(self.BeqCRRA, (int, float)):
            raise ValueError("Bequest CRRA parameter must be a single value.")

        if isinstance(self.BeqFac, (int, float)):
            self.BeqFac = [self.BeqFac] * self.T_cycle
        elif len(self.BeqFac) == 1:
            self.BeqFac *= self.T_cycle
        elif len(self.BeqFac) != self.T_cycle:
            raise ValueError(
                "Bequest relative value parameter must be a single value or a list of length T_cycle",
            )

        if not isinstance(self.BeqShift, (int, float)):
            raise ValueError("Bequest Stone-Geary parameter must be a single value.")

    def update_solution_terminal(self):
        if self.TermBeqFac == 0.0:  # No terminal bequest
            super().update_solution_terminal()
        else:
            utility = UtilityFuncCRRA(self.CRRA)

            warm_glow = UtilityFuncStoneGeary(
                self.TermBeqCRRA,
                factor=self.TermBeqFac,
                shifter=self.TermBeqShift,
            )

            aNrmGrid = (
                np.append(0.0, self.aXtraGrid)
                if self.TermBeqShift != 0.0
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
    time_inv_ = IndShockConsumerType.time_inv_ + [
        "BeqCRRA",
        "BeqShift",
    ]

    time_vary_ = IndShockConsumerType.time_vary_ + [
        "BeqFac",
    ]

    def __init__(self, **kwds):
        params = init_portfolio_bequest.copy()
        params.update(kwds)

        self.IndepDstnBool = True

        super().__init__(**params)

        self.solve_one_period = make_one_period_oo_solver(
            BequestWarmGlowPortfolioSolver,
        )

    def update(self):
        super().update()
        self.update_parameters()

    def update_parameters(self):
        if not isinstance(self.BeqCRRA, (int, float)):
            raise ValueError("Bequest CRRA parameter must be a single value.")

        if isinstance(self.BeqFac, (int, float)):
            self.BeqFac = [self.BeqFac] * self.T_cycle
        elif len(self.BeqFac) == 1:
            self.BeqFac *= self.T_cycle
        elif len(self.BeqFac) != self.T_cycle:
            raise ValueError(
                "Bequest relative value parameter must be a single value or a list of length T_cycle",
            )

        if not isinstance(self.BeqShift, (int, float)):
            raise ValueError("Bequest Stone-Geary parameter must be a single value.")

    def update_solution_terminal(self):
        if self.TermBeqFac == 0.0:  # No terminal bequest
            super().update_solution_terminal()
        else:
            utility = UtilityFuncCRRA(self.CRRA)

            warm_glow = UtilityFuncStoneGeary(
                self.TermBeqCRRA,
                factor=self.TermBeqFac,
                shifter=self.TermBeqShift,
            )

            aNrmGrid = (
                np.append(0.0, self.aXtraGrid)
                if self.TermBeqShift != 0.0
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
):
    """
    Solves one period of a consumption-saving model with idiosyncratic shocks to
    permanent and transitory income, with one risk free asset and CRRA utility.
    The consumer also has a "warm glow" bequest motive in which they gain additional
    utility based on their terminal wealth upon death. Currently does not support
    value function construction nor cubic spline interpolation, in order to match
    behavior of existing OO-solver.

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

    # Make the marginal value function and the marginal marginal value function
    vPfuncNow = MargValueFuncCRRA(cFuncNow, CRRA)
    vPPfuncNow = NullFunc()  # Dummy object
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


class BequestWarmGlowConsumerSolver(ConsIndShockSolver):
    def __init__(
        self,
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
    ):
        self.BeqCRRA = BeqCRRA
        self.BeqFac = BeqFac
        self.BeqShift = BeqShift
        vFuncBool = False
        CubicBool = False

        super().__init__(
            solution_next,
            IncShkDstn,
            LivPrb,
            DiscFac,
            CRRA,
            Rfree,
            PermGroFac,
            BoroCnstArt,
            aXtraGrid,
            vFuncBool,
            CubicBool,
        )

    def def_utility_funcs(self):
        super().def_utility_funcs()

        BeqFacEff = (1.0 - self.LivPrb) * self.BeqFac

        self.warm_glow = UtilityFuncStoneGeary(self.BeqCRRA, BeqFacEff, self.BeqShift)

    def def_BoroCnst(self, BoroCnstArt):
        self.BoroCnstNat = (
            (self.solution_next.mNrmMin - self.TranShkMinNext)
            * (self.PermGroFac * self.PermShkMinNext)
            / self.Rfree
        )

        self.BoroCnstNat = np.max([self.BoroCnstNat, -self.BeqShift])

        if BoroCnstArt is None:
            self.mNrmMinNow = self.BoroCnstNat
        else:
            self.mNrmMinNow = np.max([self.BoroCnstNat, BoroCnstArt])
        if self.BoroCnstNat < self.mNrmMinNow:
            self.MPCmaxEff = 1.0
        else:
            self.MPCmaxEff = self.MPCmaxNow

        self.cFuncNowCnst = LinearInterp(
            np.array([self.mNrmMinNow, self.mNrmMinNow + 1]), np.array([0.0, 1.0])
        )

    def calc_EndOfPrdvP(self):
        EndofPrdvP = super().calc_EndOfPrdvP()

        return EndofPrdvP + self.warm_glow.der(self.aNrmNow)


class BequestWarmGlowPortfolioSolver(ConsPortfolioSolver):
    def __init__(
        self,
        solution_next,
        ShockDstn,
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
        BeqCRRA,
        BeqFac,
        BeqShift,
    ):
        self.BeqCRRA = BeqCRRA
        self.BeqFac = BeqFac
        self.BeqShift = BeqShift
        vFuncBool = False
        DiscreteShareBool = False
        IndepDstnBool = True

        super().__init__(
            solution_next,
            ShockDstn,
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
            vFuncBool,
            AdjustPrb,
            DiscreteShareBool,
            ShareLimit,
            IndepDstnBool,
        )

    def def_utility_funcs(self):
        super().def_utility_funcs()

        self.warm_glow = UtilityFuncStoneGeary(self.BeqCRRA, self.BeqFac, self.BeqShift)

    def calc_EndOfPrdvP(self):
        super().calc_EndOfPrdvP()

        self.EndofPrddvda = self.EndOfPrddvda + self.warm_glow.der(self.aNrm_tiled)
        self.EndOfPrddvdaNvrs = self.uPinv(self.EndOfPrddvda)


init_wealth_in_utility = init_idiosyncratic_shocks.copy()
init_wealth_in_utility["BeqCRRA"] = init_idiosyncratic_shocks["CRRA"]
init_wealth_in_utility["BeqFac"] = 1.0
init_wealth_in_utility["BeqShift"] = 0.0
init_wealth_in_utility["TermBeqCRRA"] = init_idiosyncratic_shocks["CRRA"]
init_wealth_in_utility["TermBeqFac"] = 0.0  # ignore bequest motive in terminal period
init_wealth_in_utility["TermBeqShift"] = 0.0

init_warm_glow = init_lifecycle.copy()
init_warm_glow["TermBeqCRRA"] = init_lifecycle["CRRA"]
init_warm_glow["TermBeqFac"] = 1.0
init_warm_glow["TermBeqShift"] = 0.0

init_accidental_bequest = init_warm_glow.copy()
init_accidental_bequest["BeqFac"] = 1.0  # Value of bequest relative to consumption
init_accidental_bequest["BeqShift"] = 0.0  # Shifts the utility function
init_accidental_bequest["BeqCRRA"] = init_lifecycle["CRRA"]

init_portfolio_bequest = init_accidental_bequest.copy()
init_portfolio_bequest.update(init_portfolio)

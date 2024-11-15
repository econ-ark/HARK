"""
Classes for representing agents with two financial assets: one liquid and one
illiquid. The illiquid asset has a higher return factor, but is less accessible
in some sense. In the basic model, withdrawals from the illiquid asset incur a
proportional penalty.
"""

from copy import copy
import numpy as np
import matplotlib.pyplot as plt

from HARK.metric import MetricObject
from HARK.rewards import UtilityFuncCRRA
from HARK.distribution import expected
from HARK.interpolation import (
    LinearInterp,
    LinearInterpOnInterp1D,
    MargValueFuncCRRA,
    VariableLowerBoundFunc2D,
    IdentityFunction,
    LowerEnvelope2D,
    UpperEnvelope,
)
from HARK.utilities import make_assets_grid, plot_funcs
from HARK.ConsumptionSaving.ConsIndShockModel import (
    KinkedRconsumerType,
    PerfForesightConsumerType,
)
from HARK.Calibration.Income.IncomeProcesses import (
    construct_lognormal_income_process_unemployment,
    get_PermShkDstn_from_IncShkDstn,
    get_TranShkDstn_from_IncShkDstn,
)

###############################################################################

class ConFromAssetsFunction(MetricObject):
    """
    A trivial class for wrapping an "assets function" in a consumption function.
    """
    def __init__(self, aFunc):
        self.aFunc = aFunc
        
    def __call__(self,x,y):
        return x - self.aFunc(x,y)
    
    def derivativeX(self,x,y):
        return 1. - self.aFunc.derivativeX(x,y)
    
    def derivativeY(self,x,y):
        return -self.aFunc.derivativeY(x,y)


class DepositFunction(MetricObject):
    """
    A special class for representing optimal net deposits as a function of liquid
    and illiquid market resources (mNrm, nNrm).

    Parameters
    ----------
    LowerBound : function
        Function that yields the lower boundary of the region of inaction with
        respect to mNrm given nNrm.
    UpperBound : function
        Function that yields the upper boundary of the region of inaction with
        respect to mNrm given nNrm.
    CashDeposit : function
        Function that yields the optimal kNrm to hold as a function of total wealth
        when making a deposit.
    CashWithdraw : function
        Function that yields the optimal kNrm to hold as a function of adjusted
        total wealth when making a withdrawal.
    Bottom : float
        Lowest value of total adjusted wealth for which CashWithdraw is defined.
        Below this, the agent should withdraw all of his illiquid wealth.
    Penalty : float
        Scaling penalty on making withdrawals from the illiquid asset.

    Returns
    -------
    None
    """

    distance_criteria = [
        "LowerBound",
        "UpperBound",
        "CashDeposit",
        "CashWithdraw",
        "Bottom",
        "Penalty",
    ]

    def __init__(
        self, LowerBound, UpperBound, CashDeposit, CashWithdraw, Bottom, Penalty
    ):
        self.LowerBound = LowerBound
        self.UpperBound = UpperBound
        self.CashDeposit = CashDeposit
        self.CashWithdraw = CashWithdraw
        self.Bottom = Bottom
        self.Penalty = Penalty

    def __call__(self, mNrm, nNrm):
        mNrmX = np.array(mNrm)
        nNrmX = np.array(nNrm)
        kNrm = mNrmX.copy()
        deposit = mNrmX > self.UpperBound(nNrmX)
        withdraw = mNrmX < self.LowerBound(nNrmX)
        oNrm = mNrmX[deposit] + nNrmX[deposit]
        kNrm[deposit] = self.CashDeposit(oNrm)
        oNrmAdj = mNrmX[withdraw] + nNrmX[withdraw] / (1.0 + self.Penalty)
        kNrm_temp = self.CashWithdraw(oNrmAdj)
        below_bottom = oNrmAdj <= self.Bottom
        kNrm_temp[below_bottom] = oNrmAdj[below_bottom]  # withdraw everything
        kNrm[withdraw] = kNrm_temp
        dNrm = mNrmX - kNrm  # Will be zero for "do nothing"
        return dNrm

    def derivativeX(self, mNrm, nNrm, eps=1e-8):
        dLo = self.__call__(mNrm - eps, nNrm)
        dHi = self.__call__(mNrm + eps, nNrm)
        return (dHi - dLo) / (2 * eps)

    def derivativeY(self, mNrm, nNrm, eps=1e-8):
        dLo = self.__call__(mNrm, nNrm - eps)
        dHi = self.__call__(mNrm, nNrm + eps)
        return (dHi - dLo) / (2 * eps)


class SpecialMargValueFunction(MetricObject):
    """
    A class with a special representation of the decision-time marginal value
    function over liquid and illiquid market resources.

    Parameters
    ----------
    dFunc : function
        Optimal net deposits as a function of liquid and illiquid market resources (mNrm, nNrm).
    dvdk : function
        Marginal value of middle-of-period liquid kash-on-hand, as a function of
        middle-of-period kash-on-hand kNrm and middle-of-period illiquid assets bNrm.
    dvdb : function
        Marginal value of middle-of-period illiquid assets, as a function of
        middle-of-period kash-on-hand kNrm and middle-of-period illiquid assets bNrm.
    Penalty : float
        Scaling penalty on making withdrawals from the illiquid asset.

    Returns
    -------
    None
    """

    distance_criteria = ["dFunc", "dvdk", "dvdb", "cFunc_Deposit", "cFunc_Withdraw", "Penalty", "CRRA"]

    def __init__(self, dFunc, dvdk, dvdb, cFunc_Deposit, cFunc_Withdraw, Penalty, CRRA):
        self.dFunc = dFunc
        self.dvdk = dvdk
        self.dvdb = dvdb
        self.cFunc_Deposit = cFunc_Deposit
        self.cFunc_Withdraw = cFunc_Withdraw
        self.Penalty = Penalty
        self.CRRA = CRRA

    def __call__(self, mNrm, nNrm):
        """
        Returns marginal values of beginning-of-period liquid and illiquid market resources
        dvdm and dvdn as a function of liquid and illiquid market resources (mNrm, nNrm).
        """
        dNrm = self.dFunc(mNrm, nNrm)
        withdraw = dNrm < 0.0
        deposit = dNrm > 0.0
        dNrmAdj = dNrm * (1.0 + self.Penalty * withdraw)
        kNrm = mNrm - dNrm
        bNrm = np.maximum(nNrm + dNrmAdj, 0.0)
        dvdm = self.dvdk(kNrm, bNrm)
        dvdn = self.dvdb(kNrm, bNrm)
        withdraw = np.logical_and(withdraw, bNrm > 0.)
        if np.any(deposit):
            cNrm_dep = self.cFunc_Deposit(bNrm[deposit])
            dvdm[deposit] = cNrm_dep**-self.CRRA
            dvdn[deposit] = dvdm[deposit]
        if np.any(withdraw):
            cNrm_wdw = self.cFunc_Withdraw(bNrm[withdraw])
            dvdm[withdraw] = cNrm_wdw**-self.CRRA
            dvdn[withdraw] = dvdm[withdraw] / (1.0 + self.Penalty)
        return dvdm, dvdn

    def dvdm(self, mNrm, nNrm):
        """
        Returns only marginal value of liquid market resources as a function of (mNrm,nNrm).
        """
        dvdm, trash = self.__call__(mNrm, nNrm)
        return dvdm

    def dvdn(self, mNrm, nNrm):
        """
        Returns only marginal value of illiquid market resources as a function of (mNrm,nNrm).
        """
        trash, dvdn = self.__call__(mNrm, nNrm)
        return dvdn


class IlliquidConsumerSolution(MetricObject):
    """
    A class for representing one period's solution to the liquid-illiquid asset
    consumption-saving problem. Just stores the marginal value function, consumption
    function, and deposit function.
    """

    distance_criteria = ["cFunc", "dFunc"]

    def __init__(self, MargValueFunc, cFunc, dFunc, mNrmMin, MPCmin=None, hNrm=None, cFunc_Deposit=None, cFunc_Withdraw=None):
        self.MargValueFunc = MargValueFunc
        self.cFunc = cFunc
        self.dFunc = dFunc
        self.mNrmMin = mNrmMin
        self.MPCmin = MPCmin
        self.hNrm = hNrm
        self.cFunc_Deposit = cFunc_Deposit
        self.cFunc_Withdraw = cFunc_Withdraw


def make_basic_illiquid_solution_terminal(CRRA):
    """
    Function that makes a trivial terminal period in which liquid and illiquid
    assets are summed together, then consumed. Need to improve later.

    Parameters
    ----------
    CRRA : float
        Coefficient of relative risk aversion.

    Returns
    -------
    solution_terminal : IlliquidConsumerSolution
        A trivial terminal period solution.
    """
    rho = CRRA
    MargValueFunc_terminal = lambda m, n: ((m + n) ** -rho, (m + n) ** -rho)
    cFunc_terminal = lambda k, b: k + b
    dFunc_terminal = lambda m, n: -n
    mNrmMin_terminal = LinearInterp(np.array([0.0, 1.0]), np.array([0.0, -1.0]))
    solution_terminal = IlliquidConsumerSolution(
        MargValueFunc=MargValueFunc_terminal,
        cFunc=cFunc_terminal,
        dFunc=dFunc_terminal,
        mNrmMin=mNrmMin_terminal,
    )
    return solution_terminal


def make_PF_illiquid_solution_terminal(CRRA, DiscFac, Rilqd):
    """
    Function that makes a trivial terminal period in which liquid and illiquid
    assets are summed together, then passed to an infinite horizon PF problem.

    Parameters
    ----------
    CRRA : float
        Coefficient of relative risk aversion.
    DiscFac : float
        Intertemporal discount factor.
    Rilqd : float
        Rate of return on illiquid asset holdings.

    Returns
    -------
    solution_terminal : IlliquidConsumerSolution
        A trivial terminal period solution.
    """
    rho = CRRA
    LivPrbPF = 0.95
    PermGroFacPF = 1.0

    PF_dict = {
        "cycles": 0,
        "BoroCnstArt": 0.0,
        "Rfree": Rilqd,
        "CRRA": rho,
        "DiscFac": DiscFac,
        "LivPrb": [LivPrbPF],
        "PermGroFac": [PermGroFacPF],
    }
    PFtype = PerfForesightConsumerType(**PF_dict)
    PFtype.solve()
    vPfunc_terminal = PFtype.solution[0].vPfunc
    mNrmMin_terminal = PFtype.solution[0].mNrmMin

    MargValueFunc_terminal = lambda m, n: (
        vPfunc_terminal(m + n),
        vPfunc_terminal(m + n),
    )
    cFunc_terminal = lambda k, b: k + b
    dFunc_terminal = lambda m, n: -n
    mNrmMin_terminal = LinearInterp(
        np.array([0.0, 1.0]), np.array([mNrmMin_terminal, mNrmMin_terminal - 1.0])
    )
    solution_terminal = IlliquidConsumerSolution(
        MargValueFunc=MargValueFunc_terminal,
        cFunc=cFunc_terminal,
        dFunc=dFunc_terminal,
        mNrmMin=mNrmMin_terminal,
        MPCmin=PFtype.solution[0].MPCmin,
        hNrm=PFtype.solution[0].hNrm,
    )
    return solution_terminal


def make_illiquid_assets_grid(bNrmMin, bNrmMax, bNrmCount, bNrmExtra, bNrmNestFac):
    """
    Simple constructor that wraps make_assets_grid, applying it to illiquid assets.
    """
    bNrmGrid = make_assets_grid(bNrmMin, bNrmMax, bNrmCount, bNrmExtra, bNrmNestFac)
    return bNrmGrid


###############################################################################


def calc_boro_const_nat(mNrmLowerBoundNext, bGrid, Y, R0, R1, Gamma):
    """Calculate the natural borrowing constraint on an array of bNrm values.

    Args:
        mNrmLowerBoundNext (float): Minimum normalized market resources next period.
        bGrid (array): Values of illiquid assets at which to calculate the natural borrowing constraint.
        Y (DiscreteDstn): Distribution of shocks to income.
        R0 (float): Risk free interest factor on liquid assets.
        R1 (float): Risk free interest factor on illiquid assets.
        Gamma (float): Permanent income growth factor.
    """
    perm, tran = Y.atoms
    perm = np.reshape(perm, (perm.size, 1))
    tran = np.reshape(tran, (tran.size, 1))
    bGridX = np.reshape(bGrid, (1, bGrid.size))
    G = Gamma * perm
    n_next = R1 * bGridX / G
    m_min_next = mNrmLowerBoundNext(n_next)
    a_now = G * (m_min_next - tran) / R0
    return np.max(a_now, axis=0)


def calc_next_period_state(Y, a, b, Rboro, Rsave, Rilqd, Gamma):
    """
    Calculate the distribution of next period's (m,n) states.
    """
    psi = Y["PermShk"]
    theta = Y["TranShk"]
    G = Gamma * psi
    R = Rsave * np.ones_like(a)
    R[a < 0] = Rboro
    m = R / G * a + theta
    n = Rilqd * b / G + 0.0 * theta
    return m, n


def calc_marg_values_next(
    Y, a, b, rho, Rboro, Rsave, Rilqd, Gamma, marg_value_func_next
):
    """
    Calculate next period's marginal values of liquid and illiquid market resources.
    """
    psi = Y["PermShk"]
    theta = Y["TranShk"]
    G = Gamma * psi
    R = Rsave * np.ones_like(a)
    R[a < 0] = Rboro
    m = R / G * a + theta
    n = Rilqd / G * b + 0.0 * theta
    fac = G**-rho
    dvdm, dvdn = marg_value_func_next(m, n)
    dvdm *= fac
    dvdn *= fac
    return dvdm, dvdn


def solve_one_period_basic_illiquid(
    solution_next,
    IncShkDstn,
    LivPrb,
    DiscFac,
    CRRA,
    Rsave,
    Rboro,
    Rilqd,
    IlqdPenalty,
    PermGroFac,
    BoroCnstArt,
    aXtraGrid,
    bNrmGrid,
):
    """
    Solves one period of a consumption-saving model with one liquid and one illiquid
    asset. The liquid asset has a very low (or zero) rate of return for saving and
    a high rate for borrowing. The illiquid asset has a moderate return to saving
    and cannot be used for borrowing; deposits can be freely made into the illiquid
    asset, but withdrawals incur a proportional penalty. The agent faces a typical
    permanent-transitory labor income risk structure and has additively time-
    separable, constant relative risk aversion preferences, permitting the typical
    normalization by permanent income.

    Parameters
    ----------
    solution_next : IlliquidConsumerSolution
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
    Rboro: float
        Interest factor on assets between this period and the succeeding
        period when liquid assets are negative.
    Rsave : float
        Interest factor on assets between this period and the succeeding
        period when liquid assets are positive.
    Rilqd : float
        Interest factor on illiquid assets.
    IlqdPenalty : float
        Penalty factor on withdrawals from the illiquid asset (into the liquid)
        asset. If the agent withdraws X dollars, then it costs (1+IlqdPenalty)*X
        in terms of reduced illiquid assets.
    PermGroFac : float
        Expected permanent income growth factor at the end of this period.
    BoroCnstArt : float or None
        Borrowing constraint for the minimum allowable liquid assets to end the
        period with.  If it is less than the natural borrowing constraint,
        then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
        rowing constraint.
    aXtraGrid : np.array
        Array of "extra" end-of-period liquid asset values-- assets above the
        absolute minimum acceptable level.
    bNrmGrid : np.array
        Array of end-of-period illiquid asset values. Because borrowing is not
        allowed for illiquid assets, this grid does not need to be adjusted for
        the "minimum acceptable"-- it's always zero.

    Returns
    -------
    solution_now : IlliquidConsumerSolution
        Solution to this period's consumption-saving problem with illiquid assets.
    """
    # Verifiy that there is actually a kink in the interest factor
    assert (
        Rboro >= Rsave
    ), "Interest factor on debt less than interest factor on savings!"
    # If the kink is in the wrong direction, code should break here

    # Define the current period utility function and effective discount factor
    uFunc = UtilityFuncCRRA(CRRA)
    DiscFacEff = DiscFac * LivPrb  # "effective" discount factor

    # Calculate patience factor and lower bound of MPC
    PatFac = ((Rilqd * DiscFacEff) ** (1.0 / CRRA)) / Rilqd
    MPCminNow = 1.0 / (1.0 + PatFac / solution_next.MPCmin)

    # Calculate updated human wealth
    Ex_IncNext = expected(lambda x: x["PermShk"] * x["TranShk"], IncShkDstn)
    hNrmNow = (PermGroFac / Rilqd) * (solution_next.hNrm + Ex_IncNext)

    # Unpack next period's marginal value function
    MargValueFuncNext = solution_next.MargValueFunc

    # Calculate the natural borrowing constraint by illiquid assets
    BoroCnstNat = calc_boro_const_nat(
        solution_next.mNrmMin, bNrmGrid, IncShkDstn, Rboro, Rilqd, PermGroFac
    )
    BoroCnstNatFunc = LinearInterp(bNrmGrid, BoroCnstNat)

    plt.plot(bNrmGrid, BoroCnstNat, "-b")
    plt.show()

    # Determine whether the natural borrowing constraint or the tradeoff is steeper
    tradeoff_slope = -1.0 / (1.0 + IlqdPenalty)
    tradeoff_func = LinearInterp(
        [0.0, 1.0], [BoroCnstNat[0], BoroCnstNat[0] + tradeoff_slope]
    )

    # Make the cross product of the liquid asset grid and illiquid asset grid
    aCount = aXtraGrid.size
    bCount = bNrmGrid.size
    aNrmNow = np.zeros((aCount + 2, bCount))
    bNrmNow = np.tile(np.reshape(bNrmGrid, (1, bCount)), (aCount + 2, 1))
    for j in range(bCount):
        temp_grid = BoroCnstNat[j] + aXtraGrid
        i = np.searchsorted(temp_grid, 0.0)
        temp_grid = np.insert(temp_grid, i, 0.0)
        temp_grid = np.insert(temp_grid, i, -1e-16)
        aNrmNow[:, j] = temp_grid

    # Compute end-of-period marginal value of liquid and illiquid assets
    # EndOfPrd_dvda, EndOfPrd_dvdb = expected(
    #     calc_marg_values_next,
    #     IncShkDstn,
    #     args=(
    #         aNrmNow,
    #         bNrmNow,
    #         CRRA,
    #         Rboro,
    #         Rsave,
    #         Rilqd,
    #         PermGroFac,
    #         MargValueFuncNext,
    #     ),
    # )
    aNrm_temp = np.reshape(aNrmNow, (aCount + 2, bCount, 1))
    bNrm_temp = np.reshape(bNrmNow, (aCount + 2, bCount, 1))
    ShkCount = IncShkDstn.atoms.shape[-1]
    PermShk = np.reshape(IncShkDstn.atoms[0,], (1, 1, ShkCount))
    TranShk = np.reshape(IncShkDstn.atoms[1,], (1, 1, ShkCount))
    ShkPrbs = np.tile(
        np.reshape(IncShkDstn.pmv, (1, 1, ShkCount)), (aCount + 2, bCount, 1)
    )

    GroFac = PermGroFac * PermShk
    Rliqd = Rsave * np.ones_like(aNrm_temp)
    Rliqd[aNrm_temp < 0] = Rboro
    mNrm_temp = Rliqd / GroFac * aNrm_temp + TranShk
    nNrm_temp = Rilqd / GroFac * bNrm_temp + 0.0 * TranShk
    temp_fac = GroFac**-CRRA
    dvdm, dvdn = MargValueFuncNext(mNrm_temp, nNrm_temp)
    dvdm *= temp_fac
    dvdn *= temp_fac

    EndOfPrd_dvda = np.sum(dvdm * ShkPrbs, axis=2)
    EndOfPrd_dvdb = np.sum(dvdn * ShkPrbs, axis=2)

    # Rescale expected marginal value by discount factor and return factor
    Rliqd = Rsave * np.ones_like(EndOfPrd_dvda)
    Rliqd[aNrmNow < 0] = Rboro
    EndOfPrd_dvda *= DiscFacEff * Rliqd
    EndOfPrd_dvdb *= DiscFacEff * Rilqd

    # Find optimal consumption implied by each (a,b) end-of-period state by
    # inverting the first order condition, as well as endogenous gridpoints in k.
    cNrmNow = uFunc.derinv(EndOfPrd_dvda, order=(1, 0))
    kNrmNow = cNrmNow + aNrmNow  # Endogenous kNrm gridpoints

    LB_next = solution_next.mNrmMin(nNrm_temp)
    violations = mNrm_temp < LB_next

    bad = np.argwhere(np.isnan(dvdm))
    if np.any(bad):
        J = 10
        for j in range(J):
            print(
                "m=",
                mNrm_temp[bad[j, 0], bad[j, 1], bad[j, 2]],
                "n=",
                nNrm_temp[bad[j, 0], bad[j, 1], bad[j, 2]],
            )

    print("cNrm", np.sum(np.isnan(cNrmNow)), np.sum(np.isinf(cNrmNow)))
    print("dvdm", np.sum(np.isnan(dvdm)), np.sum(np.isinf(dvdm)))
    print("violations", np.sum(violations))

    for j in range(bCount):
        plt.plot(kNrmNow[:, j], cNrmNow[:, j])
    plt.show()

    # Construct consumption function and pseudo-inverse marginal value of illiquid
    # market resources functions
    dvdbNvrs = uFunc.derinv(EndOfPrd_dvdb, order=(1, 0))
    cFunc_by_bNrm_list = []
    aFunc_by_bNrm_list = []
    dvdbNvrsFunc_by_bNrm_list = []
    for j in range(bCount):
        kNrm_temp = np.insert(kNrmNow[:, j] - BoroCnstNat[j], 0, 0.0)
        aNrm_temp = np.insert(aNrmNow[:, j], 0, BoroCnstNat[j])
        cNrm_temp = np.insert(cNrmNow[:, j], 0, 0.0)
        dvdbNvrs_temp = np.insert(dvdbNvrs[:, j], 0, 0.0)
        intercept_temp = MPCminNow * (hNrmNow + bNrmGrid[j] + BoroCnstNat[j])
        cFunc_by_bNrm_list.append(
            LinearInterp(kNrm_temp, cNrm_temp, intercept_temp, MPCminNow)
        )
        aFunc_by_bNrm_list.append(
            LinearInterp(kNrm_temp, aNrm_temp)
        )
        dvdbNvrsFunc_by_bNrm_list.append(LinearInterp(kNrm_temp, dvdbNvrs_temp))
    #cFuncUncBase = LinearInterpOnInterp1D(cFunc_by_bNrm_list, bNrmGrid)
    #cFuncUnc = VariableLowerBoundFunc2D(cFuncUncBase, BoroCnstNatFunc)
    cFuncCnstBase = IdentityFunction(i_dim=0, n_dims=2)
    aFuncUncBase = LinearInterpOnInterp1D(aFunc_by_bNrm_list, bNrmGrid)
    aFuncUnc = VariableLowerBoundFunc2D(aFuncUncBase, BoroCnstNatFunc)
    cFuncUnc = ConFromAssetsFunction(aFuncUnc)
    if (BoroCnstArt is not None) and (BoroCnstArt > -np.inf):
        borrowing_constraint = LinearInterp(
            [0.0, 1.0], [BoroCnstArt, BoroCnstArt - 1 / (1.0 + IlqdPenalty)]
        )
    else:
        borrowing_constraint = BoroCnstNatFunc
    cFuncCnst = VariableLowerBoundFunc2D(cFuncCnstBase, borrowing_constraint)
    cFuncNow = LowerEnvelope2D(cFuncUnc, cFuncCnst, nan_bool=False)
    dvdbNvrsFuncBase = LinearInterpOnInterp1D(dvdbNvrsFunc_by_bNrm_list, bNrmGrid)
    dvdbNvrsFunc = VariableLowerBoundFunc2D(dvdbNvrsFuncBase, BoroCnstNatFunc)

    # Make the lower bound of mNrm as a function of nNrm
    mNrmMinNow = UpperEnvelope(BoroCnstNatFunc, tradeoff_func)
    mNrmMinNow = tradeoff_func

    # Construct the marginal value functions for middle-of-period liquid and illiquid resource
    dvdkFunc = MargValueFuncCRRA(cFuncNow, CRRA)
    dvdbFunc = MargValueFuncCRRA(dvdbNvrsFunc, CRRA)

    # The consumption stage has now been solved, and we can begin work on the deposit stage.

    # Initialize arrays to hold the optimal zero deposit and optimal zero withdrawal values
    OptZeroDeposit = np.zeros(bCount) + np.nan
    OptZeroWithdraw = np.zeros(bCount) + np.nan

    # Loop over bNrm values and find places where the FOC holds with equality.
    # The version here will be *very* slow, can improve later once it works.
    RatioTarg = -1.0 / CRRA * np.log(1.0 + IlqdPenalty)
    LogNvrsRatio = np.log(cNrmNow / dvdbNvrs)
    for j in range(bCount):
        LogNvrsRatio_temp = LogNvrsRatio[:, j]
        kNrm_temp = kNrmNow[:, j]
        descending = LogNvrsRatio_temp[1:] - LogNvrsRatio_temp[:-1] < -0.001
        try:
            crash
            cut = np.argwhere(descending)[0][0]
        except:
            cut = -1

        # plt.plot(kNrm_temp, LogNvrsRatio_temp, '-b')
        # plt.plot(kNrm_temp, np.zeros_like(kNrm_temp), '--k')
        # plt.plot(kNrm_temp, RatioTarg * np.ones_like(kNrm_temp), '--k')
        # plt.show()

        idx = np.searchsorted(LogNvrsRatio_temp[:cut], 0.0)
        k0 = kNrm_temp[idx - 1]
        k1 = kNrm_temp[idx]
        x0 = LogNvrsRatio_temp[idx - 1]
        x1 = LogNvrsRatio_temp[idx]
        # print(cut,x0,x1,k0,k1)
        alpha = (0.0 - x0) / (x1 - x0)
        OptZeroDeposit[j] = (1.0 - alpha) * k0 + alpha * k1

        idx = np.searchsorted(LogNvrsRatio_temp[:cut], RatioTarg)
        k0 = kNrm_temp[idx - 1]
        k1 = kNrm_temp[idx]
        x0 = LogNvrsRatio_temp[idx - 1]
        x1 = LogNvrsRatio_temp[idx]
        # print(cut,x0,x1,k0,k1)
        alpha = (RatioTarg - x0) / (x1 - x0)
        OptZeroWithdraw[j] = (1.0 - alpha) * k0 + alpha * k1

        # Define the functions to be searched for FOC solutions
        # dvdkFunc_temp = MargValueFuncCRRA(cFunc_by_bNrm_list[j], CRRA)
        # dvdbFunc_temp = MargValueFuncCRRA(dvdbNvrsFunc_by_bNrm_list[j], CRRA)
        # LogRatioFunc = lambda x: np.log(dvdkFunc_temp(x) / dvdbFunc_temp(x))
        # LogRatioFuncAlt = lambda x: LogRatioFunc(x) - RatioTarg

        # # Define the bounds of the search in k
        # if j < 10:
        #     kBotD = kNrmNow[0, j] - BoroCnstNat[j]
        #     kTopD = kBotD + 3.0 #kNrmNow[-1, j] - BoroCnstNat[j]
        #     kBotW = kNrmNow[0, j] - BoroCnstNat[j]
        #     kTopW = kNrmNow[-1, j] - BoroCnstNat[j]
        # else:
        #     kBotD = OptZeroDeposit[j - 1] - BoroCnstNat[j - 1]
        #     kTopD = kBotD + 1.0
        #     kBotW = OptZeroWithdraw[j - 1] - BoroCnstNat[j - 1]

        # # Perform a bounded search for optimal zero withdrawal and deposit
        # print(kBotD, kTopD, LogRatioFunc(kBotD), LogRatioFunc(kTopD))
        # #plot_funcs(LogRatioFunc, kNrmNow[0, j] - BoroCnstNat[j], kNrmNow[-1, j] - BoroCnstNat[j])
        # OptZeroDeposit[j] = (
        #     root_scalar(
        #         LogRatioFunc,
        #         bracket=(kBotD, kTopD),
        #         xtol=1e-8,
        #         rtol=1e-8,
        #         method="brentq",
        #     ).root
        #     + BoroCnstNat[j]
        # )
        # # print(bNrmGrid[j], 'deposit', OptZeroDeposit[j])
        # if j > 1:
        #     kTopW = OptZeroDeposit[j] - BoroCnstNat[j]
        # try:
        #     OptZeroWithdraw[j] = (
        #         root_scalar(
        #             LogRatioFuncAlt,
        #             bracket=(kBotW, kTopW),
        #             xtol=1e-8,
        #             rtol=1e-8,
        #             method="brentq",
        #         ).root
        #         + BoroCnstNat[j]
        #     )
        # except:
        #     OptZeroWithdraw[j] = BoroCnstNat[j]
        # print(bNrmGrid[j], "deposit", OptZeroDeposit[j], "withdraw", OptZeroWithdraw[j])

    plt.plot(bNrmGrid, OptZeroDeposit, "-b")
    plt.plot(bNrmGrid, OptZeroWithdraw, "-r")
    plt.xlabel('illiquid assets bNrm')
    plt.ylabel('liquid assets kNrm')
    plt.show()
    
    # Construct linear interpolations of the boundaries of the region of inaction
    InactionUpperBoundFunc = LinearInterp(bNrmGrid, OptZeroDeposit)
    InactionLowerBoundFunc = LinearInterp(bNrmGrid, OptZeroWithdraw)

    # Construct the "optimal kash on hand when depositing" function
    TotalWealth = OptZeroDeposit + bNrmGrid
    CashFunc_Deposit = LinearInterp(TotalWealth, OptZeroDeposit)

    # Construct the "optimal kash on hand when withdrawing" function
    TotalWealthAdj = OptZeroWithdraw + bNrmGrid / (1.0 + IlqdPenalty)
    CashFunc_Withdraw = LinearInterp(TotalWealthAdj, OptZeroWithdraw)
    
    # Construct simplified consumption functions when withdrawing or depositing
    cNrm_dep = cFuncNow(OptZeroDeposit, bNrmGrid)
    cNrm_wdw = cFuncNow(OptZeroWithdraw, bNrmGrid)
    cFunc_Deposit = LinearInterp(bNrmGrid, cNrm_dep)
    cFunc_Withdraw = LinearInterp(bNrmGrid, cNrm_wdw)
    
    plt.plot(bNrmGrid, cNrm_dep, "-b")
    plt.plot(bNrmGrid, cNrm_wdw, "-r")
    plt.xlabel('illiquid assets bNrm')
    plt.ylabel('consumption cNrm')
    plt.show()
    
    # Construct the deposit function as a special representation
    dFuncNow = DepositFunction(
        InactionLowerBoundFunc,
        InactionUpperBoundFunc,
        CashFunc_Deposit,
        CashFunc_Withdraw,
        TotalWealthAdj[0],
        IlqdPenalty,
    )

    # Construct the marginal value function
    MargValueFuncNow = SpecialMargValueFunction(
        dFuncNow, dvdkFunc, dvdbFunc, cFunc_Deposit, cFunc_Withdraw, IlqdPenalty, CRRA,
    )

    # Package and return the solution object
    solution_now = IlliquidConsumerSolution(
        MargValueFunc=MargValueFuncNow,
        cFunc=cFuncNow,
        dFunc=dFuncNow,
        mNrmMin=mNrmMinNow,
        MPCmin=MPCminNow,
        hNrm=hNrmNow,
        cFunc_Deposit=cFunc_Deposit,
        cFunc_Withdraw=cFunc_Withdraw,
    )
    return solution_now


###############################################################################

# Make a dictionary of constructors for the idiosyncratic income shocks model
basic_illiquid_constructor_dict = {
    "IncShkDstn": construct_lognormal_income_process_unemployment,
    "PermShkDstn": get_PermShkDstn_from_IncShkDstn,
    "TranShkDstn": get_TranShkDstn_from_IncShkDstn,
    "aXtraGrid": make_assets_grid,
    "bNrmGrid": make_illiquid_assets_grid,
    "solution_terminal": make_PF_illiquid_solution_terminal,
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
    "aXtraMin": 1e-3,  # Minimum end-of-period "assets above minimum" value
    "aXtraMax": 24,  # Maximum end-of-period "assets above minimum" value
    "aXtraNestFac": 1,  # Linear spacing for aXtraGrid
    "aXtraCount": 200,  # Number of points in the grid of "assets above minimum"
    "aXtraExtra": [5e-3, 1e-2],  # Additional other values to add in grid (optional)
}

# Default parameters to make bNrmGrid using make_illiquid_assets_grid
default_bNrmGrid_params = {
    "bNrmMin": 0.0,  # Minimum end-of-period "assets above minimum" value
    "bNrmMax": 16,  # Maximum end-of-period "assets above minimum" value
    "bNrmNestFac": 1,  # Exponential spacing for bNrmGrid
    "bNrmCount": 96,  # Number of points in the grid of "assets above minimum"
    "bNrmExtra": None,  # Additional other values to add in grid (optional)
}

# Make a dictionary to specify an idiosyncratic income shocks consumer type
init_basic_illiquid = {
    # BASIC HARK PARAMETERS REQUIRED TO SOLVE THE MODEL
    "cycles": 1,  # Finite, non-cyclic model
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    "constructors": basic_illiquid_constructor_dict,  # See dictionary above
    # PRIMITIVE RAW PARAMETERS REQUIRED TO SOLVE THE MODEL
    "CRRA": 2.0,  # Coefficient of relative risk aversion
    "Rboro": 1.20,  # Interest factor on liquid assets when borrowing, a < 0
    "Rsave": 1.00,  # Interest factor on liquid assets when saving, a > 0
    "Rilqd": 1.03,  # Interest factor on illiquid assets b
    "IlqdPenalty": 0.15,  # Proportional penalty on withdrawing from illiquid asset
    "DiscFac": 0.96,  # Intertemporal discount factor
    "LivPrb": [0.98],  # Survival probability after each period
    "PermGroFac": [1.01],  # Permanent income growth factor
    "BoroCnstArt": None,  # Artificial borrowing constraint
    # (Uses linear spline interpolation for cFunc when False)
    # PARAMETERS REQUIRED TO SIMULATE THE MODEL
    "AgentCount": 10000,  # Number of agents of this type
    "T_age": None,  # Age after which simulated agents are automatically killed
    "aNrmInitMean": 0.0,  # Mean of log initial assets
    "aNrmInitStd": 1.0,  # Standard deviation of log initial assets
    "pLvlInitMean": 0.0,  # Mean of log initial permanent income
    "pLvlInitStd": 0.0,  # Standard deviation of log initial permanent income
    "PermGroFacAgg": 1.0,  # Aggregate permanent income growth factor
    # (The portion of PermGroFac attributable to aggregate productivity growth)
    "NewbornTransShk": False,  # Whether Newborns have transitory shock
    # ADDITIONAL OPTIONAL PARAMETERS
    "PerfMITShk": False,  # Do Perfect Foresight MIT Shock
    # (Forces Newborns to follow solution path of the agent they replaced if True)
    "neutral_measure": False,  # Whether to use permanent income neutral measure (see Harmenberg 2021)
}
init_basic_illiquid.update(default_IncShkDstn_params)
init_basic_illiquid.update(default_aXtraGrid_params)
init_basic_illiquid.update(default_bNrmGrid_params)


class BasicIlliquidConsumerType(KinkedRconsumerType):
    """
    A consumer type that faces idiosyncratic shocks to income and has a different
    interest factor on saving vs borrowing.  Extends IndShockConsumerType, with
    very small changes.  Solver for this class is currently only compatible with
    linear spline interpolation.

    Same parameters as AgentType.


    Parameters
    ----------
    """

    time_inv_ = copy(KinkedRconsumerType.time_inv_)
    time_inv_ += ["Rilqd", "bNrmGrid", "IlqdPenalty"]

    def __init__(self, **kwds):
        params = init_basic_illiquid.copy()
        params.update(kwds)

        # Initialize a basic AgentType
        super().__init__(**params)

        # Add consumer-type specific objects, copying to create independent versions
        self.solve_one_period = solve_one_period_basic_illiquid

    def update(self):
        super().update()
        self.construct("bNrmGrid")

    def cFuncSimple(self, t, mNrm, nNrm):
        dFunc = self.solution[t].dFunc
        cFunc = self.solution[t].cFunc
        dNrm = dFunc(mNrm, nNrm)
        kNrm = mNrm - dNrm
        withdraw = dNrm < 0.0
        deposit = dNrm > 0.0
        temp = 1.0 + self.IlqdPenalty * withdraw
        bNrm = np.maximum(nNrm + temp * dNrm, 0.0)
        cNrm = cFunc(kNrm, bNrm)
        withdraw = np.logical_and(withdraw, bNrm > 0.)
        if np.any(deposit):
            cNrm_dep = self.solution[t].cFunc_Deposit(bNrm[deposit])
            cNrm[deposit] = cNrm_dep
        if np.any(withdraw):
            cNrm_wdw = self.solution[t].cFunc_Withdraw(bNrm[withdraw])
            cNrm[withdraw] = cNrm_wdw
        return cNrm

    def MPCfuncSimple(self, t, mNrm, nNrm):
        dFunc = self.solution[t].dFunc
        cFunc = self.solution[t].cFunc
        dNrm = dFunc(mNrm, nNrm)
        kNrm = mNrm - dNrm
        temp = 1.0 + self.IlqdPenalty * (dNrm < 0.0)
        bNrm = np.maximum(nNrm + temp * dNrm, 0.0)

        dcdk = cFunc.derivativeX(kNrm, bNrm)
        dcdb = cFunc.derivativeY(kNrm, bNrm)
        dDdm = dFunc.derivativeX(mNrm, nNrm)
        dkdm = 1.0 - dDdm
        dbdm = temp * dDdm
        dcdm = dcdk * dkdm + dcdb * dbdm
        MPC = dcdm

        return MPC


if __name__ == "__main__":
    MyType = BasicIlliquidConsumerType()
    MyType.cycles = 3
    MyType.solve()

    B = 6.0
    f = lambda x: MyType.MPCfuncSimple(0, x, B * np.ones_like(x))
    g = lambda x: MyType.cFuncSimple(0, x, B * np.ones_like(x))
    h = lambda x: MyType.solution[0].dFunc(x, B * np.ones_like(x))
    z = lambda x: (x + MyType.solution[0].hNrm + B) * MyType.solution[0].MPCmin
    c = lambda x: MyType.solution[0].cFunc(x, B * np.ones_like(x))

    plot_funcs([g], -7.5, 12)


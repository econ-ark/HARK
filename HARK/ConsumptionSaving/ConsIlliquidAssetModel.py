"""
Classes for representing agents with two financial assets: one liquid and one
illiquid. The illiquid asset has a higher return factor, but is less accessible
in some sense. In the basic model, withdrawals from the illiquid asset incur a
proportional penalty.
"""

# from copy import copy, deepcopy
import numpy as np
from scipy.optimize import brentq

from HARK.metric import MetricObject
from HARK.rewards import UtilityFuncCRRA

# from HARK.distribution import expected
from HARK.interpolation import (
    LinearInterp,
    LinearInterpOnInterp1D,
    MargValueFuncCRRA,
    VariableLowerBoundFunc2D,
    IdentityFunction,
    LowerEnvelope2D,
)
# from HARK.ConsumptionSaving.ConsIndShockModel import KinkedRconsumerType

###############################################################################


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
        "Penalty",
    ]

    def __init__(self, LowerBound, UpperBound, CashDeposit, CashWithdraw, Penalty):
        self.LowerBound = LowerBound
        self.UpperBound = UpperBound
        self.CashDeposit = CashDeposit
        self.CashWithdraw = CashWithdraw
        self.Penalty = Penalty

    def __call__(self, mNrm, nNrm):
        kNrm = mNrm.copy()
        deposit = mNrm > self.UpperBound(mNrm)
        withdraw = mNrm < self.LowerBound(nNrm)
        oNrm = mNrm[deposit] + nNrm[deposit]
        kNrm[deposit] = self.CashDeposit(oNrm)
        oNrmAdj = mNrm[withdraw] + nNrm[withdraw] / (1.0 + self.Penalty)
        kNrm[withdraw] = self.CashWithDraw(oNrmAdj)
        dNrm = mNrm - kNrm  # Will be zero for "do nothing"
        return dNrm


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

    distance_criteria = ["dFunc", "dvdk", "dvdb", "Penalty"]

    def __init__(self, dFunc, dvdk, dvdb, Penalty):
        self.dFunc = dFunc
        self.dvdk = dvdk
        self.dvdb = dvdb
        self.Penalty = Penalty

    def __call__(self, mNrm, nNrm):
        """
        Returns marginal values of beginning-of-period liquid and illiquid market resources
        dvdm and dvdn as a function of liquid and illiquid market resources (mNrm, nNrm).
        """
        dNrm = self.dFunc(mNrm, nNrm)
        withdraw = dNrm < 0.0
        dNrmAdj = dNrm + self.Penalty * withdraw
        kNrm = mNrm - dNrm
        bNrm = nNrm + dNrmAdj
        dvdm = self.dvdk(kNrm, bNrm)
        dvdn = self.dvdb(kNrm, bNrm)
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

    def __init__(self, MargValueFunc, cFunc, dFunc):
        self.MargValueFunc = MargValueFunc
        self.cFunc = cFunc
        self.dFunc = dFunc


###############################################################################


def calc_boro_const_nat(mNrmLowerBoundNext, bGrid, mMinFunc, Y, R, Gamma):
    """Calculate the natural borrowing constraint on an array of bNrm values.

    Args:
        mNrmLowerBoundNext (float): Minimum normalized market resources next period.
        Y (DiscreteDstn): Distribution of shocks to income.
        R (float): Risk free interest factor.
        Gamma (float): Permanent income growth factor.
    """
    perm, tran = Y.atoms
    temp_fac = (Gamma * np.min(perm)) / R
    return (mMinFunc(bGrid) - np.min(tran)) * temp_fac


def calc_next_period_state(Y, a, b, Rboro, Rsave, Rilqd, Gamma):
    """
    Calculate the distribution of next period's (m,n) states.
    """
    psi = Y["Perm"]
    theta = Y["Tran"]
    G = Gamma * psi
    R = Rsave * np.ones_like(a)
    R[a < 0] = Rboro
    m = R / G * a + theta
    n = Rilqd * b
    return m, n


def calc_marg_values_next(
    Y, a, b, rho, Rboro, Rsave, Rilqd, Gamma, marg_value_func_next
):
    """
    Calculate next period's marginal values of liquid and illiquid market resources.
    """
    psi = Y["Perm"]
    theta = Y["Tran"]
    G = Gamma * psi
    R = Rsave * np.ones_like(a)
    R[a < 0] = Rboro
    m = R / G * a + theta
    n = Rilqd * b
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

    # Unpack next period's marginal value function
    MargValueFuncNext = solution_next.MargValueFunc

    # Calculate the natural borrowing constraint by illiquid assets
    BoroCnstNat = calc_boro_const_nat(
        solution_next.mNrmMin, bNrmGrid, IncShkDstn, Rboro, PermGroFac
    )
    BoroCnstNatFunc = LinearInterp(bNrmGrid, BoroCnstNat)

    # Make the cross product of the liquid asset grid and illiquid asset grid
    aCount = aXtraGrid.size
    bCount = bNrmGrid.size
    aNrmNow = np.zeros((aCount + 2, bCount))
    bNrmNow = np.tile(np.reshape(bNrmGrid, (1, bCount)), (aCount, 1))
    for j in range(bCount):
        temp_grid = BoroCnstNat[j] + aXtraGrid
        i = np.searchsorted(temp_grid, 0.0)
        temp_grid = np.insert(temp_grid, 0.0, i)
        temp_grid = np.insert(temp_grid, -1e16, i)
        aNrmNow[:, j] = temp_grid

    # Compute end-of-period marginal value of liquid and illiquid assets
    EndOfPrd_dvda, EndOfPrd_dvdb = calc_marg_values_next(
        IncShkDstn,
        aNrmNow,
        bNrmNow,
        CRRA,
        Rboro,
        Rsave,
        Rilqd,
        PermGroFac,
        MargValueFuncNext,
    )
    Rliqd = Rsave * np.ones_like(EndOfPrd_dvda)
    # Rescale expected marginal value by discount factor and return factor
    Rilqd[aNrmNow < 0] = Rboro
    EndOfPrd_dvda *= DiscFacEff * Rliqd
    EndOfPrd_dvdb *= DiscFacEff * Rilqd

    # Find optimal consumption implied by each (a,b) end-of-period state by
    # inverting the first order condition, as well as endogenous gridpoints in k.
    cNrmNow = uFunc.derinv(EndOfPrd_dvda, order=(1, 0))
    kNrmNow = cNrmNow + aNrmNow  # Endogenous kNrm gridpoints

    # Construct consumption function and pseudo-inverse marginal value of illiquid
    # market resources functions
    dvdbNvrs = uFunc.derinv(EndOfPrd_dvdb, order=(1, 0))
    cFunc_by_bNrm_list = []
    dvdbNvrsFunc_by_bNrm_list = []
    for j in range(bCount):
        kNrm_temp = np.insert(kNrmNow[:, j] - BoroCnstNat[j], 0, 0.0)
        cNrm_temp = np.insert(cNrmNow[:, j], 0, 0.0)
        dvdbNvrs_temp = np.insert(dvdbNvrs[:, j], 0, 0.0)
        cFunc_by_bNrm_list.append(LinearInterp(kNrm_temp, cNrm_temp))
        dvdbNvrsFunc_by_bNrm_list.append(LinearInterp(kNrm_temp, dvdbNvrs_temp))
    cFuncUncBase = LinearInterpOnInterp1D(cFunc_by_bNrm_list, bNrmGrid)
    cFuncCnstBase = IdentityFunction(i_dim=0, n_dims=2)
    cFuncBase = LowerEnvelope2D(cFuncUncBase, cFuncCnstBase)
    cFuncNow = VariableLowerBoundFunc2D(cFuncBase, BoroCnstNatFunc)
    dvdbNvrsFuncBase = LinearInterpOnInterp1D(dvdbNvrsFunc_by_bNrm_list, bNrmGrid)
    dvdbNvrsFunc = VariableLowerBoundFunc2D(dvdbNvrsFuncBase, BoroCnstNatFunc)

    # Construct the marginal value functions for middle-of-period liquid and illiquid resource
    dvdkFunc = MargValueFuncCRRA(cFuncNow)
    dvdbFunc = MargValueFuncCRRA(dvdbNvrsFunc)

    # The consumption stage has now been solved, and we can begin work on the deposit stage.

    # Initialize arrays to hold the optimal zero deposit and optimal zero withdrawal values
    OptZeroDeposit = np.zeros(bCount) + np.nan
    OptZeroWithdraw = np.zeros(bCount) + np.nan

    # Loop over bNrm values and find places where the FOC holds with equality.
    # The version here will be *very* slow, can improve later once it works.
    RatioTarg = np.log(1.0 + IlqdPenalty)
    for j in range(bCount):
        # Define the functions to be searched for FOC solutions
        dvdkFunc_temp = MargValueFuncCRRA(cFunc_by_bNrm_list[j], CRRA)
        dvdbFunc_temp = MargValueFuncCRRA(dvdbNvrsFunc_by_bNrm_list[j], CRRA)
        LogRatioFunc = lambda x: np.log(dvdkFunc_temp(x) / dvdbFunc_temp(x))
        LogRatioFuncAlt = lambda x: LogRatioFunc(x) - RatioTarg

        # Define the bounds of the search in k
        kBot = kNrmNow[0, j] - BoroCnstNat[j]
        kTop = kNrmNow[-1, j] - BoroCnstNat[j]

        # Perform a bounded search for optimal zero withdrawal and deposit
        OptZeroDeposit[j] = (
            brentq(LogRatioFunc, kBot, kTop, xtol=1e-8, rtol=1e-8) + BoroCnstNat[j]
        )
        OptZeroWithdraw[j] = (
            brentq(LogRatioFuncAlt, kBot, kTop, xtol=1e-8, rtol=1e-8) + BoroCnstNat[j]
        )

    # Construct linear interpolations of the boundaries of the region of inaction
    InactionUpperBoundFunc = LinearInterp(bNrmGrid, OptZeroDeposit)
    InactionLowerBoundFunc = LinearInterp(bNrmGrid, OptZeroWithdraw)

    # Construct the "optimal kash on hand when depositing" function
    TotalWealth = OptZeroDeposit + bNrmGrid
    CashFunc_Deposit = LinearInterp(TotalWealth, OptZeroDeposit)

    # Construct the "optimal kash on hand when withdrawing" function
    TotalWealthAdj = OptZeroWithdraw + bNrmGrid / (1.0 + IlqdPenalty)
    CashFunc_Withdraw = LinearInterp(TotalWealthAdj, OptZeroWithdraw)

    # Construct the deposit function as a special representation
    dFuncNow = DepositFunction(
        InactionLowerBoundFunc,
        InactionUpperBoundFunc,
        CashFunc_Deposit,
        CashFunc_Withdraw,
        IlqdPenalty,
    )

    # Construct the marginal value function
    MargValueFuncNow = SpecialMargValueFunction(
        dFuncNow, dvdkFunc, dvdbFunc, IlqdPenalty
    )

    # Package and return the solution object
    solution_now = IlliquidConsumerSolution(
        MargValueFunc=MargValueFuncNow, cFunc=cFuncNow, dFunc=dFuncNow
    )
    return solution_now

"""
This file contains classes and functions for representing, solving, and simulating
agents who must allocate their resources among consumption, saving in a risk-free
asset (with a low return), and saving in a risky asset (with higher average return).
"""

from copy import deepcopy

import numpy as np

from HARK import NullFunc
from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    make_lognormal_pLvl_init_dstn,
    make_lognormal_kNrm_init_dstn,
)
from HARK.Calibration.Assets.AssetProcesses import (
    make_lognormal_RiskyDstn,
    combine_IncShkDstn_and_RiskyDstn,
    calc_ShareLimit_for_CRRA,
)
from HARK.Calibration.Income.IncomeProcesses import (
    construct_lognormal_income_process_unemployment,
    get_PermShkDstn_from_IncShkDstn,
    get_TranShkDstn_from_IncShkDstn,
)
from HARK.ConsumptionSaving.ConsRiskyAssetModel import (
    RiskyAssetConsumerType,
    make_simple_ShareGrid,
    make_AdjustDstn,
)
from HARK.distributions import expected
from HARK.interpolation import (
    BilinearInterp,
    ConstantFunction,
    CubicInterp,
    IdentityFunction,
    LinearInterp,
    LinearInterpOnInterp1D,
    MargValueFuncCRRA,
    ValueFuncCRRA,
)
from HARK.metric import MetricObject
from HARK.rewards import UtilityFuncCRRA
from HARK.utilities import make_assets_grid

__all__ = [
    "PortfolioSolution",
    "PortfolioConsumerType",
]


# Define a class to represent the single period solution of the portfolio choice problem
class PortfolioSolution(MetricObject):
    r"""
    A class for representing the single period solution of the portfolio choice model.

    Parameters
    ----------
    cFuncAdj : Interp1D
        Consumption function over normalized market resources when the agent is able
        to adjust their portfolio shares: :math:`c_t=\text{cFuncAdj} (m_t)`.
    ShareFuncAdj : Interp1D
        Risky share function over normalized market resources when the agent is able
        to adjust their portfolio shares: :math:`S_t=\text{ShareFuncAdj} (m_t)`.
    vFuncAdj : ValueFuncCRRA
        Value function over normalized market resources when the agent is able to
        adjust their portfolio shares: :math:`v_t=\text{vFuncAdj} (m_t)`.
    vPfuncAdj : MargValueFuncCRRA
        Marginal value function over normalized market resources when the agent is able
        to adjust their portfolio shares: :math:`v'_t=\text{vPFuncAdj} (m_t)`.
    cFuncFxd : Interp2D
        Consumption function over normalized market resources and risky portfolio share
        when the agent is NOT able to adjust their portfolio shares, so they are fixed:
        :math:`c_t=\text{cFuncFxd} (m_t,S_t)`.
    ShareFuncFxd : Interp2D
        Risky share function over normalized market resources and risky portfolio share
        when the agent is NOT able to adjust their portfolio shares, so they are fixed.
        This should always be an IdentityFunc, by definition.
    vFuncFxd : ValueFuncCRRA
        Value function over normalized market resources and risky portfolio share when
        the agent is NOT able to adjust their portfolio shares, so they are fixed:
        :math:`v_t=\text{vFuncFxd}(m_t,S_t)`.
    dvdmFuncFxd : MargValueFuncCRRA
        The derivative of the value function with respect to normalized market
        resources when the agent is Not able to adjust their portfolio shares,
        so they are fixed: :math:`\frac{dv_t}{dm_t}=\text{vFuncFxd}(m_t,S_t)`.
    dvdsFuncFxd : MargValueFuncCRRA
        The derivative of the value function with respect to risky asset share
        when the agent is Not able to adjust their portfolio shares,so they are
        fixed: :math:`\frac{dv_t}{dS_t}=\text{vFuncFxd}(m_t,S_t)`.
    aGrid: np.array
        End-of-period-assets grid used to find the solution.
    Share_adj: np.array
        Optimal portfolio share associated with each aGrid point: :math:`S^{*}_t=\text{vFuncFxd}(m_t)`.
    EndOfPrddvda_adj: np.array
        Marginal value of end-of-period resources associated with each aGrid
        point.
    ShareGrid: np.array
        Grid for the portfolio share that is used to solve the model.
    EndOfPrddvda_fxd: np.array
        Marginal value of end-of-period resources associated with each
        (aGrid x sharegrid) combination, for the agent who can not adjust his
        portfolio.
    AdjustPrb: float
        Probability that the agent will be able to adjust his portfolio
        next period.
    """

    distance_criteria = ["vPfuncAdj"]

    def __init__(
        self,
        cFuncAdj=None,
        ShareFuncAdj=None,
        vFuncAdj=None,
        vPfuncAdj=None,
        cFuncFxd=None,
        ShareFuncFxd=None,
        vFuncFxd=None,
        dvdmFuncFxd=None,
        dvdsFuncFxd=None,
        aGrid=None,
        Share_adj=None,
        EndOfPrddvda_adj=None,
        ShareGrid=None,
        EndOfPrddvda_fxd=None,
        EndOfPrddvds_fxd=None,
        AdjPrb=None,
    ):
        # Change any missing function inputs to NullFunc
        if cFuncAdj is None:
            cFuncAdj = NullFunc()
        if cFuncFxd is None:
            cFuncFxd = NullFunc()
        if ShareFuncAdj is None:
            ShareFuncAdj = NullFunc()
        if ShareFuncFxd is None:
            ShareFuncFxd = NullFunc()
        if vFuncAdj is None:
            vFuncAdj = NullFunc()
        if vFuncFxd is None:
            vFuncFxd = NullFunc()
        if vPfuncAdj is None:
            vPfuncAdj = NullFunc()
        if dvdmFuncFxd is None:
            dvdmFuncFxd = NullFunc()
        if dvdsFuncFxd is None:
            dvdsFuncFxd = NullFunc()

        # Set attributes of self
        self.cFuncAdj = cFuncAdj
        self.cFuncFxd = cFuncFxd
        self.ShareFuncAdj = ShareFuncAdj
        self.ShareFuncFxd = ShareFuncFxd
        self.vFuncAdj = vFuncAdj
        self.vFuncFxd = vFuncFxd
        self.vPfuncAdj = vPfuncAdj
        self.dvdmFuncFxd = dvdmFuncFxd
        self.dvdsFuncFxd = dvdsFuncFxd
        self.aGrid = aGrid
        self.Share_adj = Share_adj
        self.EndOfPrddvda_adj = EndOfPrddvda_adj
        self.ShareGrid = ShareGrid
        self.EndOfPrddvda_fxd = EndOfPrddvda_fxd
        self.EndOfPrddvds_fxd = EndOfPrddvds_fxd
        self.AdjPrb = AdjPrb


###############################################################################


def make_portfolio_solution_terminal(CRRA):
    """
    Solves the terminal period of the portfolio choice problem.  The solution is
    trivial, as usual: consume all market resources, and put nothing in the risky
    asset (because you have nothing anyway).

    Parameters
    ----------
    CRRA : float
        Coefficient of relative risk aversion.

    Returns
    -------
    solution_terminal : PortfolioSolution
        Terminal period solution for a consumption-saving problem with portfolio
        choice and CRRA utility.
    """
    # Consume all market resources: c_T = m_T
    cFuncAdj_terminal = IdentityFunction()
    cFuncFxd_terminal = IdentityFunction(i_dim=0, n_dims=2)

    # Risky share is irrelevant-- no end-of-period assets; set to zero
    ShareFuncAdj_terminal = ConstantFunction(0.0)
    ShareFuncFxd_terminal = IdentityFunction(i_dim=1, n_dims=2)

    # Value function is simply utility from consuming market resources
    vFuncAdj_terminal = ValueFuncCRRA(cFuncAdj_terminal, CRRA)
    vFuncFxd_terminal = ValueFuncCRRA(cFuncFxd_terminal, CRRA)

    # Marginal value of market resources is marg utility at the consumption function
    vPfuncAdj_terminal = MargValueFuncCRRA(cFuncAdj_terminal, CRRA)
    dvdmFuncFxd_terminal = MargValueFuncCRRA(cFuncFxd_terminal, CRRA)
    dvdsFuncFxd_terminal = ConstantFunction(0.0)  # No future, no marg value of Share

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
    solution_terminal.hNrm = 0.0
    solution_terminal.MPCmin = 1.0
    return solution_terminal


def calc_radj(shock, share_limit, rfree, crra):
    """Expected rate of return adjusted by CRRA

    Args:
        shock (DiscreteDistribution): Distribution of risky asset returns
        share_limit (float): limiting lower bound of risky portfolio share
        rfree (float): Risk free interest rate
        crra (float): Coefficient of relative risk aversion
    """
    rport = share_limit * shock + (1.0 - share_limit) * rfree
    return rport ** (1.0 - crra)


def calc_human_wealth(shocks, perm_gro_fac, share_limit, rfree, crra, h_nrm_next):
    """Calculate human wealth this period given human wealth next period.

    Args:
        shocks (DiscreteDistribution): Joint distribution of shocks to income and returns.
        perm_gro_fac (float): Permanent income growth factor
        share_limit (float): limiting lower bound of risky portfolio share
        rfree (float): Risk free interest rate
        crra (float): Coefficient of relative risk aversion
        h_nrm_next (float): Human wealth next period
    """
    perm_shk_fac = perm_gro_fac * shocks["PermShk"]
    rport = share_limit * shocks["Risky"] + (1.0 - share_limit) * rfree
    hNrm = (perm_shk_fac / rport**crra) * (shocks["TranShk"] + h_nrm_next)
    return hNrm


def calc_m_nrm_next(shocks, b_nrm, perm_gro_fac):
    """
    Calculate future realizations of market resources mNrm from the income
    shock distribution "shocks" and normalized bank balances b.
    """
    return b_nrm / (shocks["PermShk"] * perm_gro_fac) + shocks["TranShk"]


def calc_dvdx_next(
    shocks,
    b_nrm,
    share,
    adjust_prob,
    perm_gro_fac,
    crra,
    vp_func_adj,
    dvdm_func_fxd,
    dvds_func_fxd,
):
    """
    Evaluate realizations of marginal values next period, based
    on the income distribution "shocks", values of bank balances bNrm, and values of
    the risky share z.
    """
    m_nrm = calc_m_nrm_next(shocks, b_nrm, perm_gro_fac)
    dvdm_adj = vp_func_adj(m_nrm)
    # No marginal value of shockshare if it's a free choice!
    dvds_adj = np.zeros_like(m_nrm)

    if adjust_prob < 1.0:
        # Expand to the same dimensions as mNrm
        share_exp = np.full_like(m_nrm, share)
        dvdm_fxd = dvdm_func_fxd(m_nrm, share_exp)
        dvds_fxd = dvds_func_fxd(m_nrm, share_exp)
        # Combine by adjustment probability
        dvdm = adjust_prob * dvdm_adj + (1.0 - adjust_prob) * dvdm_fxd
        dvds = adjust_prob * dvds_adj + (1.0 - adjust_prob) * dvds_fxd
    else:  # Don't bother evaluating if there's no chance that portfolio share is fixed
        dvdm = dvdm_adj
        dvds = dvds_adj

    perm_shk_fac = shocks["PermShk"] * perm_gro_fac
    dvdm = perm_shk_fac ** (-crra) * dvdm
    dvds = perm_shk_fac ** (1.0 - crra) * dvds

    return dvdm, dvds


def calc_end_of_prd_dvdx(shocks, a_nrm, share, rfree, dvdb_func, dvds_func):
    """
    Compute end-of-period marginal values at values a, conditional
    on risky asset return shocks and risky share z.
    """
    # Calculate future realizations of bank balances bNrm
    ex_ret = shocks - rfree  # Excess returns
    r_port = rfree + share * ex_ret  # Portfolio return
    b_nrm = r_port * a_nrm
    # Ensure shape concordance
    share_exp = np.full_like(b_nrm, share)

    # Calculate and return dvda, dvds
    dvda = r_port * dvdb_func(b_nrm, share_exp)
    dvds = ex_ret * a_nrm * dvdb_func(b_nrm, share_exp) + dvds_func(b_nrm, share_exp)
    return dvda, dvds


def calc_v_intermed(
    shocks, b_nrm, share, adjust_prob, perm_gro_fac, crra, v_func_adj, v_func_fxd
):
    """
    Calculate "intermediate" value from next period's bank balances, the
    income shocks shocks, and the risky asset share.
    """
    m_nrm = calc_m_nrm_next(shocks, b_nrm, perm_gro_fac)

    v_adj = v_func_adj(m_nrm)
    if adjust_prob < 1.0:
        v_fxd = v_func_fxd(m_nrm, share)
        # Combine by adjustment probability
        v_next = adjust_prob * v_adj + (1.0 - adjust_prob) * v_fxd
    else:  # Don't bother evaluating if there's no chance that portfolio share is fixed
        v_next = v_adj

    v_intermed = (shocks["PermShk"] * perm_gro_fac) ** (1.0 - crra) * v_next
    return v_intermed


def calc_end_of_prd_v(shocks, a_nrm, share, rfree, v_func):
    """Compute end-of-period values."""
    # Calculate future realizations of bank balances bNrm
    ex_ret = shocks - rfree
    r_port = rfree + share * ex_ret
    b_rnm = r_port * a_nrm

    # Make an extended share_next of the same dimension as b_nrm so
    # that the function can be vectorized
    share_exp = np.full_like(b_rnm, share)

    return v_func(b_rnm, share_exp)


def calc_m_nrm_next_joint(shocks, a_nrm, share, rfree, perm_gro_fac):
    """
    Calculate future realizations of market resources mNrm from the shock
    distribution shocks, normalized end-of-period assets a, and risky share z.
    """
    # Calculate future realizations of bank balances bNrm
    ex_ret = shocks["Risky"] - rfree
    r_port = rfree + share * ex_ret
    b_nrm = r_port * a_nrm
    return b_nrm / (shocks["PermShk"] * perm_gro_fac) + shocks["TranShk"]


def calc_end_of_prd_dvdx_joint(
    shocks,
    a_nrm,
    share,
    rfree,
    adjust_prob,
    perm_gro_fac,
    crra,
    vp_func_adj,
    dvdm_func_fxd,
    dvds_func_fxd,
):
    """
    Evaluate end-of-period marginal value of assets and risky share based
    on the shock distribution S, values of bend of period assets a, and
    risky share z.
    """
    m_nrm = calc_m_nrm_next_joint(shocks, a_nrm, share, rfree, perm_gro_fac)
    ex_ret = shocks["Risky"] - rfree
    r_port = rfree + share * ex_ret
    dvdm_adj = vp_func_adj(m_nrm)
    # No marginal value of Share if it's a free choice!
    dvds_adj = np.zeros_like(m_nrm)

    if adjust_prob < 1.0:
        # Expand to the same dimensions as mNrm
        share_exp = np.full_like(m_nrm, share)
        dvdm_fxd = dvdm_func_fxd(m_nrm, share_exp)
        dvds_fxd = dvds_func_fxd(m_nrm, share_exp)
        # Combine by adjustment probability
        dvdm_next = adjust_prob * dvdm_adj + (1.0 - adjust_prob) * dvdm_fxd
        dvds_next = adjust_prob * dvds_adj + (1.0 - adjust_prob) * dvds_fxd
    else:  # Don't bother evaluating if there's no chance that portfolio share is fixed
        dvdm_next = dvdm_adj
        dvds_next = dvds_adj

    perm_shk_fac = shocks["PermShk"] * perm_gro_fac
    temp_fac = perm_shk_fac ** (-crra) * dvdm_next
    eop_dvda = r_port * temp_fac
    eop_dvds = ex_ret * a_nrm * temp_fac + perm_shk_fac ** (1 - crra) * dvds_next

    return eop_dvda, eop_dvds


def calc_end_of_prd_v_joint(
    shocks, a_nrm, share, rfree, adjust_prob, perm_gro_fac, crra, v_func_adj, v_func_fxd
):
    """
    Evaluate end-of-period value, based on the shock distribution S, values
    of bank balances bNrm, and values of the risky share z.
    """
    m_nrm = calc_m_nrm_next_joint(shocks, a_nrm, share, rfree, perm_gro_fac)
    v_adj = v_func_adj(m_nrm)

    if adjust_prob < 1.0:
        # Expand to the same dimensions as mNrm
        share_exp = np.full_like(m_nrm, share)
        v_fxd = v_func_fxd(m_nrm, share_exp)
        # Combine by adjustment probability
        v_next = adjust_prob * v_adj + (1.0 - adjust_prob) * v_fxd
    else:  # Don't bother evaluating if there's no chance that portfolio share is fixed
        v_next = v_adj

    return (shocks["PermShk"] * perm_gro_fac) ** (1.0 - crra) * v_next


def solve_one_period_ConsPortfolio(
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
    vFuncBool,
    DiscreteShareBool,
    IndepDstnBool,
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

    # Unpack next period's solution for easier access
    vPfuncAdj_next = solution_next.vPfuncAdj
    dvdmFuncFxd_next = solution_next.dvdmFuncFxd
    dvdsFuncFxd_next = solution_next.dvdsFuncFxd
    vFuncAdj_next = solution_next.vFuncAdj
    vFuncFxd_next = solution_next.vFuncFxd

    # Set a flag for whether the natural borrowing constraint is zero, which
    # depends on whether the smallest transitory income shock is zero
    BoroCnstNat_iszero = np.min(IncShkDstn.atoms[1]) == 0.0

    # Prepare to calculate end-of-period marginal values by creating an array
    # of market resources that the agent could have next period, considering
    # the grid of end-of-period assets and the distribution of shocks he might
    # experience next period.

    # Unpack the risky return shock distribution
    Risky_next = RiskyDstn.atoms
    RiskyMax = np.max(Risky_next)
    RiskyMin = np.min(Risky_next)

    # Perform an alternate calculation of the absolute patience factor when
    # returns are risky. This uses the Merton-Samuelson limiting risky share,
    # which is what's relevant as mNrm goes to infinity.

    R_adj = expected(calc_radj, RiskyDstn, args=(ShareLimit, Rfree, CRRA))[0]
    PatFac = (DiscFacEff * R_adj) ** (1.0 / CRRA)
    MPCminNow = 1.0 / (1.0 + PatFac / solution_next.MPCmin)

    # Also perform an alternate calculation for human wealth under risky returns

    # This correctly accounts for risky returns and risk aversion
    hNrmNow = (
        expected(
            calc_human_wealth,
            ShockDstn,
            args=(PermGroFac, ShareLimit, Rfree, CRRA, solution_next.hNrm),
        )
        / R_adj
    )

    # Set the terms of the limiting linear consumption function as mNrm goes to infinity
    cFuncLimitIntercept = MPCminNow * hNrmNow
    cFuncLimitSlope = MPCminNow

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

    # If the income shock distribution is independent from the risky return distribution,
    # then taking end-of-period expectations can proceed in a two part process: First,
    # construct an "intermediate" value function by integrating out next period's income
    # shocks, *then* compute end-of-period expectations by integrating out return shocks.
    # This method is lengthy to code, but can be significantly faster.
    if IndepDstnBool:
        # Make tiled arrays to calculate future realizations of mNrm and Share when integrating over IncShkDstn
        bNrmNext, ShareNext = np.meshgrid(bNrmGrid, ShareGrid, indexing="ij")

        # Define functions that are used internally to evaluate future realizations

        # Calculate end-of-period marginal value of assets and shares at each point
        # in aNrm and ShareGrid. Does so by taking expectation of next period marginal
        # values across income and risky return shocks.

        # Calculate intermediate marginal value of bank balances and risky portfolio share
        # by taking expectations over income shocks

        dvdb_intermed, dvds_intermed = expected(
            calc_dvdx_next,
            IncShkDstn,
            args=(
                bNrmNext,
                ShareNext,
                AdjustPrb,
                PermGroFac,
                CRRA,
                vPfuncAdj_next,
                dvdmFuncFxd_next,
                dvdsFuncFxd_next,
            ),
        )

        dvdbNvrs_intermed = uFunc.derinv(dvdb_intermed, order=(1, 0))
        dvdbNvrsFunc_intermed = BilinearInterp(dvdbNvrs_intermed, bNrmGrid, ShareGrid)
        dvdbFunc_intermed = MargValueFuncCRRA(dvdbNvrsFunc_intermed, CRRA)

        dvdsFunc_intermed = BilinearInterp(dvds_intermed, bNrmGrid, ShareGrid)

        # Make tiled arrays to calculate future realizations of bNrm and Share when integrating over RiskyDstn
        aNrmNow, ShareNext = np.meshgrid(aNrmGrid, ShareGrid, indexing="ij")

        # Define functions for calculating end-of-period marginal value

        # Evaluate realizations of value and marginal value after asset returns are realized

        # Calculate end-of-period marginal value of assets  and risky portfolio share
        # by taking expectations

        EndOfPrd_dvda, EndOfPrd_dvds = DiscFacEff * expected(
            calc_end_of_prd_dvdx,
            RiskyDstn,
            args=(aNrmNow, ShareNext, Rfree, dvdbFunc_intermed, dvdsFunc_intermed),
        )

        EndOfPrd_dvdaNvrs = uFunc.derinv(EndOfPrd_dvda)

        # Make the end-of-period value function if the value function is requested
        if vFuncBool:
            # Calculate intermediate value by taking expectations over income shocks
            v_intermed = expected(
                calc_v_intermed,
                IncShkDstn,
                args=(
                    bNrmNext,
                    ShareNext,
                    AdjustPrb,
                    PermGroFac,
                    CRRA,
                    vFuncAdj_next,
                    vFuncFxd_next,
                ),
            )

            # Construct the "intermediate value function" for this period
            vNvrs_intermed = uFunc.inv(v_intermed)
            vNvrsFunc_intermed = BilinearInterp(vNvrs_intermed, bNrmGrid, ShareGrid)
            vFunc_intermed = ValueFuncCRRA(vNvrsFunc_intermed, CRRA)

            # Calculate end-of-period value by taking expectations
            EndOfPrd_v = DiscFacEff * expected(
                calc_end_of_prd_v,
                RiskyDstn,
                args=(aNrmNow, ShareNext, Rfree, vFunc_intermed),
            )
            EndOfPrd_vNvrs = uFunc.inv(EndOfPrd_v)

            # Now make an end-of-period value function over aNrm and Share
            EndOfPrd_vNvrsFunc = BilinearInterp(EndOfPrd_vNvrs, aNrmGrid, ShareGrid)
            EndOfPrd_vFunc = ValueFuncCRRA(EndOfPrd_vNvrsFunc, CRRA)
            # This will be used later to make the value function for this period

    # If the income shock distribution and risky return distribution are *NOT*
    # independent, then computation of end-of-period expectations are simpler in
    # code, but might take longer to execute
    else:
        # Make tiled arrays to calculate future realizations of mNrm and Share when integrating over IncShkDstn
        aNrmNow, ShareNext = np.meshgrid(aNrmGrid, ShareGrid, indexing="ij")

        # Define functions that are used internally to evaluate future realizations

        # Evaluate realizations of value and marginal value after asset returns are realized

        # Calculate end-of-period marginal value of assets and risky share by taking expectations
        EndOfPrd_dvda, EndOfPrd_dvds = DiscFacEff * expected(
            calc_end_of_prd_dvdx_joint,
            ShockDstn,
            args=(
                aNrmNow,
                ShareNext,
                Rfree,
                AdjustPrb,
                PermGroFac,
                CRRA,
                vPfuncAdj_next,
                dvdmFuncFxd_next,
                dvdsFuncFxd_next,
            ),
        )
        EndOfPrd_dvdaNvrs = uFunc.derinv(EndOfPrd_dvda)

        # Construct the end-of-period value function if requested
        if vFuncBool:
            # Calculate end-of-period value, its derivative, and their pseudo-inverse
            EndOfPrd_v = DiscFacEff * expected(
                calc_end_of_prd_v_joint,
                ShockDstn,
                args=(
                    aNrmNow,
                    ShareNext,
                    Rfree,
                    AdjustPrb,
                    PermGroFac,
                    CRRA,
                    vFuncAdj_next,
                    vFuncFxd_next,
                ),
            )
            EndOfPrd_vNvrs = uFunc.inv(EndOfPrd_v)

            # value transformed through inverse utility
            EndOfPrd_vNvrsP = EndOfPrd_dvda * uFunc.derinv(EndOfPrd_v, order=(0, 1))

            # Construct the end-of-period value function
            EndOfPrd_vNvrsFunc_by_Share = []
            for j in range(ShareCount):
                EndOfPrd_vNvrsFunc_by_Share.append(
                    CubicInterp(
                        aNrmNow[:, j], EndOfPrd_vNvrs[:, j], EndOfPrd_vNvrsP[:, j]
                    )
                )
            EndOfPrd_vNvrsFunc = LinearInterpOnInterp1D(
                EndOfPrd_vNvrsFunc_by_Share, ShareGrid
            )
            EndOfPrd_vFunc = ValueFuncCRRA(EndOfPrd_vNvrsFunc, CRRA)

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
        aNrm_temp = np.maximum(mNrm_temp - cNrm_temp, 0.0)  # Fix tiny violations
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
    solution_now.hNrm = hNrmNow
    solution_now.MPCmin = MPCminNow
    return solution_now


###############################################################################

# Make a dictionary of constructors for the portfolio choice consumer type
PortfolioConsumerType_constructors_default = {
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
    "solution_terminal": make_portfolio_solution_terminal,
}

# Make a dictionary with parameters for the default constructor for kNrmInitDstn
PortfolioConsumerType_kNrmInitDstn_default = {
    "kLogInitMean": -12.0,  # Mean of log initial capital
    "kLogInitStd": 0.0,  # Stdev of log initial capital
    "kNrmInitCount": 15,  # Number of points in initial capital discretization
}

# Make a dictionary with parameters for the default constructor for pLvlInitDstn
PortfolioConsumerType_pLvlInitDstn_default = {
    "pLogInitMean": 0.0,  # Mean of log permanent income
    "pLogInitStd": 0.0,  # Stdev of log permanent income
    "pLvlInitCount": 15,  # Number of points in initial capital discretization
}

# Default parameters to make IncShkDstn using construct_lognormal_income_process_unemployment
PortfolioConsumerType_IncShkDstn_default = {
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
PortfolioConsumerType_aXtraGrid_default = {
    "aXtraMin": 0.001,  # Minimum end-of-period "assets above minimum" value
    "aXtraMax": 100,  # Maximum end-of-period "assets above minimum" value
    "aXtraNestFac": 1,  # Exponential nesting factor for aXtraGrid
    "aXtraCount": 200,  # Number of points in the grid of "assets above minimum"
    "aXtraExtra": None,  # Additional other values to add in grid (optional)
}

# Default parameters to make RiskyDstn with make_lognormal_RiskyDstn (and uniform ShareGrid)
PortfolioConsumerType_RiskyDstn_default = {
    "RiskyAvg": 1.08,  # Mean return factor of risky asset
    "RiskyStd": 0.18362634887,  # Stdev of log returns on risky asset
    "RiskyCount": 5,  # Number of integration nodes to use in approximation of risky returns
}
PortfolioConsumerType_ShareGrid_default = {
    "ShareCount": 25  # Number of discrete points in the risky share approximation
}

# Make a dictionary to specify a risky asset consumer type
PortfolioConsumerType_solving_default = {
    # BASIC HARK PARAMETERS REQUIRED TO SOLVE THE MODEL
    "cycles": 1,  # Finite, non-cyclic model
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    "constructors": PortfolioConsumerType_constructors_default,  # See dictionary above
    # PRIMITIVE RAW PARAMETERS REQUIRED TO SOLVE THE MODEL
    "CRRA": 5.0,  # Coefficient of relative risk aversion
    "Rfree": [1.03],  # Return factor on risk free asset
    "DiscFac": 0.90,  # Intertemporal discount factor
    "LivPrb": [0.98],  # Survival probability after each period
    "PermGroFac": [1.01],  # Permanent income growth factor
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint
    "DiscreteShareBool": False,  # Whether risky asset share is restricted to discrete values
    "PortfolioBool": True,  # This *must* be set to True; only exists because of inheritance
    "PortfolioBisect": False,  # What does this do?
    "IndepDstnBool": True,  # Whether return and income shocks are independent
    "vFuncBool": False,  # Whether to calculate the value function during solution
    "CubicBool": False,  # Whether to use cubic spline interpolation when True
    # (Uses linear spline interpolation for cFunc when False)
    "AdjustPrb": 1.0,  # Probability that the agent can update their risky portfolio share each period
    "RiskyShareFixed": None,  # This does nothing in this model; only exists because of inheritance
    "sim_common_Rrisky": True,  # Whether risky returns have a shared/common value across agents
}
PortfolioConsumerType_simulation_default = {
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
PortfolioConsumerType_default = {}
PortfolioConsumerType_default.update(PortfolioConsumerType_solving_default)
PortfolioConsumerType_default.update(PortfolioConsumerType_simulation_default)
PortfolioConsumerType_default.update(PortfolioConsumerType_kNrmInitDstn_default)
PortfolioConsumerType_default.update(PortfolioConsumerType_pLvlInitDstn_default)
PortfolioConsumerType_default.update(PortfolioConsumerType_aXtraGrid_default)
PortfolioConsumerType_default.update(PortfolioConsumerType_ShareGrid_default)
PortfolioConsumerType_default.update(PortfolioConsumerType_IncShkDstn_default)
PortfolioConsumerType_default.update(PortfolioConsumerType_RiskyDstn_default)
init_portfolio = PortfolioConsumerType_default


class PortfolioConsumerType(RiskyAssetConsumerType):
    r"""
    A consumer type based on IndShockRiskyAssetConsumerType, with portfolio optimization.
    The agent is only able to change their risky asset share with a certain probability.

    .. math::
        \newcommand{\CRRA}{\rho}
        \newcommand{\DiePrb}{\mathsf{D}}
        \newcommand{\PermGroFac}{\Gamma}
        \newcommand{\Rfree}{\mathsf{R}}
        \newcommand{\DiscFac}{\beta}
        \begin{align*}
        v_t(m_t,S_t) &= \max_{c_t,S^{*}_t} u(c_t) + \DiscFac (1-\DiePrb_{t+1})  \mathbb{E}_{t} \left[(\PermGroFac_{t+1}\psi_{t+1})^{1-\CRRA} v_{t+1}(m_{t+1},S_{t+1}) \right], \\
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
        \mathbb{E}[\psi]=\mathbb{E}[\theta] &= 1.\\
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
    ShareGrid: Constructor
        The agent's risky asset share grid

        Its default constructor is :func:`HARK.ConsumptionSaving.ConsRiskyAssetModel.make_simple_ShareGrid`
    RiskyDstn: Constructor, :math:`\phi`
        The agent's asset shock distribution for risky assets.

        Its default constructor is :func:`HARK.Calibration.Assets.AssetProcesses.make_lognormal_RiskyDstn`

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

    IncShkDstn_default = PortfolioConsumerType_IncShkDstn_default
    aXtraGrid_default = PortfolioConsumerType_aXtraGrid_default
    ShareGrid_default = PortfolioConsumerType_ShareGrid_default
    RiskyDstn_default = PortfolioConsumerType_RiskyDstn_default
    solving_default = PortfolioConsumerType_solving_default
    simulation_default = PortfolioConsumerType_simulation_default

    default_ = {
        "params": PortfolioConsumerType_default,
        "solver": solve_one_period_ConsPortfolio,
        "model": "ConsPortfolio.yaml",
    }

    time_inv_ = deepcopy(RiskyAssetConsumerType.time_inv_)
    time_inv_ = time_inv_ + ["DiscreteShareBool"]

    def initialize_sim(self):
        """
        Initialize the state of simulation attributes.  Simply calls the same method
        for IndShockConsumerType, then sets the type of AdjustNow to bool.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # these need to be set because "post states",
        # but are a control variable and shock, respectively
        self.controls["Share"] = np.zeros(self.AgentCount)
        RiskyAssetConsumerType.initialize_sim(self)

    def sim_birth(self, which_agents):
        """
        Create new agents to replace ones who have recently died; takes draws of
        initial aNrm and pLvl, as in ConsIndShockModel, then sets Share and Adjust
        to zero as initial values.
        Parameters
        ----------
        which_agents : np.array
            Boolean array of size AgentCount indicating which agents should be "born".

        Returns
        -------
        None
        """
        IndShockConsumerType.sim_birth(self, which_agents)

        self.controls["Share"][which_agents] = 0.0
        # here a shock is being used as a 'post state'
        self.shocks["Adjust"][which_agents] = False

    def get_controls(self):
        """
        Calculates consumption cNrmNow and risky portfolio share ShareNow using
        the policy functions in the attribute solution.  These are stored as attributes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        cNrmNow = np.zeros(self.AgentCount) + np.nan
        ShareNow = np.zeros(self.AgentCount) + np.nan

        # Loop over each period of the cycle, getting controls separately depending on "age"
        for t in range(self.T_cycle):
            these = t == self.t_cycle

            # Get controls for agents who *can* adjust their portfolio share
            those = np.logical_and(these, self.shocks["Adjust"])
            cNrmNow[those] = self.solution[t].cFuncAdj(self.state_now["mNrm"][those])
            ShareNow[those] = self.solution[t].ShareFuncAdj(
                self.state_now["mNrm"][those]
            )

            # Get controls for agents who *can't* adjust their portfolio share
            those = np.logical_and(these, np.logical_not(self.shocks["Adjust"]))
            cNrmNow[those] = self.solution[t].cFuncFxd(
                self.state_now["mNrm"][those], self.controls["Share"][those]
            )
            ShareNow[those] = self.solution[t].ShareFuncFxd(
                self.state_now["mNrm"][those], self.controls["Share"][those]
            )  # this just returns same share as before

        # Store controls as attributes of self
        self.controls["cNrm"] = cNrmNow
        self.controls["Share"] = ShareNow

    def check_conditions(self, verbose=None):
        raise NotImplementedError()

    def calc_limiting_values(self):
        raise NotImplementedError()

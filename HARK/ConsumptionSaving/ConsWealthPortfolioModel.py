from copy import deepcopy

import numpy as np
from HARK.ConsumptionSaving.ConsPortfolioModel import (
    PortfolioConsumerType,
    init_portfolio,
    PortfolioSolution,
)
from HARK.distribution import expected
from HARK.interpolation import (
    BilinearInterp,
    CubicInterp,
    LinearInterp,
    MargValueFuncCRRA,
    ValueFuncCRRA,
)
from HARK.rewards import UtilityFuncCRRA
from HARK.utilities import NullFunc


import numpy as np
from HARK.interpolation import LinearInterp


class ChiFromOmegaFunction:
    """
    A class for representing a function that takes in values of omega = EndOfPrdvP / aNrm
    and returns the corresponding optimal chi = cNrm / aNrm. The only parameters
    that matter for this transformation are the coefficient of relative risk
    aversion rho and the share of wealth in the Cobb-Douglas aggregator delta.

    Parameters
    ----------
    rho : float
        Coefficient of relative risk aversion.
    delta : float
        Share for wealth in the Cobb-Douglas aggregator in CRRA utility function.
    N : int, optional
        Number of interpolating gridpoints to use (default 501).
    z_bound : float, optional
        Absolute value on the auxiliary variable z's boundary (default 15).
        z represents values that are input into a logit transformation
        scaled by the upper bound of chi, which yields chi values.
    """

    def __init__(self, CRRA, WealthShare, N=501, z_bound=15):
        self.CRRA = CRRA
        self.WealthShare = WealthShare
        self.N = N
        self.z_bound = z_bound

        self.update()

    def f(self, x):
        """
        Define the relationship between chi and omega, and evaluate on the vector
        """
        return x ** (1 - self.WealthShare) * (
            (1 - self.WealthShare) * x ** (-self.WealthShare)
            - self.WealthShare * x ** (1 - self.WealthShare)
        ) ** (-1 / self.CRRA)

    def update(self):
        """
        Construct the underlying interpolation of log(omega) on z.
        """
        # Make vectors of chi and z
        chi_limit = (1.0 - self.WealthShare) / self.WealthShare
        z_vec = np.linspace(-self.z_bound, self.z_bound, self.N)
        exp_z = np.exp(z_vec)
        chi_vec = chi_limit * exp_z / (1 + exp_z)

        omega_vec = self.f(chi_vec)
        log_omega_vec = np.log(omega_vec)

        # Construct the interpolant
        zFromLogOmegaFunc = LinearInterp(log_omega_vec, z_vec, lower_extrap=True)

        # Store the function and limit as attributes
        self.func = zFromLogOmegaFunc
        self.limit = chi_limit

    def __call__(self, omega):
        """
        Calculate optimal values of chi = cNrm / aNrm from values of omega.

        Parameters
        ----------
        omega : np.array
            One or more values of omega = EndOfPrdvP / aNrm.

        Returns
        -------
        chi : np.array
            Identically shaped array with optimal chi values.
        """
        z = self.func(np.log(omega))
        exp_z = np.exp(z)
        chi = self.limit * exp_z / (1 + exp_z)
        return np.nan_to_num(chi)


class WealthPortfolioConsumerType(PortfolioConsumerType):
    time_inv_ = deepcopy(PortfolioConsumerType.time_inv_)
    time_inv_ = time_inv_ + ["WealthShare", "WealthShift", "ChiFunc"]

    def __init__(self, **kwds):
        params = init_wealth_portfolio.copy()
        params.update(kwds)
        kwds = params

        # Initialize a basic portfolio consumer type
        super().__init__(**kwds)

        self.solve_one_period = solve_one_period_WealthPortfolio

        if self.WealthShare == 0.0:
            self.ChiFunc = None
        else:
            self.ChiFunc = ChiFromOmegaFunction(self.CRRA, self.WealthShare)


def utility(c, a, CRRA, share=0.0, intercept=0.0):
    w = a + intercept
    return (c ** (1 - share) * w**share) ** (1 - CRRA) / (1 - CRRA)


def dudc(c, a, CRRA, share=0.0, intercept=0.0):
    u = utility(c, a, CRRA, share, intercept)
    return u * (1 - CRRA) * (1 - share) / c


def duda(c, a, CRRA, share=0.0, intercept=0.0):
    u = utility(c, a, CRRA, share, intercept)
    return u * (1 - CRRA) * share / (a + intercept)


def du2dc2(c, a, CRRA, share=0.0, intercept=0.0):
    u = utility(c, a, CRRA, share, intercept)
    return u * (1 - CRRA) * (share - 1) * ((1 - CRRA) * (share - 1) + 1) / c**2


def du2dadc(c, a, CRRA, share=0.0, intercept=0.0):
    u = utility(c, a, CRRA, share, intercept)
    w = a + intercept
    return u * (1 - CRRA) * share * (share - 1) * (CRRA - 1) / (c * w)


def du_diff(c, a, CRRA, share=0.0, intercept=0.0):
    ufac = utility(c, a, CRRA, share, intercept) * (1 - CRRA)
    dudc = ufac * (1 - share) / c

    if share == 0:
        return dudc
    else:
        duda = ufac * share / (a + intercept)

    return dudc - duda


def du2_diff(c, a=None, CRRA=None, share=None, intercept=None, vp_a=None):
    ufac = utility(c, a, CRRA, share, intercept) * (1 - CRRA)
    w = a + intercept

    dudcdc = ufac * (share - 1) * ((1 - CRRA) * (share - 1) + 1) / c**2
    dudadc = ufac * share * (share - 1) * (CRRA - 1) / (c * w)

    return dudcdc - dudadc


def du2_jac(c, a, CRRA, share, intercept, vp_a):
    du2_diag = du2_diff(c, a, CRRA, share, intercept, vp_a)
    return np.diag(du2_diag)


def chi_ratio(c, a, intercept):
    return c / (a + intercept)


def chi_func(chi, CRRA, share):
    return chi ** (1 - share) * (
        (1 - share) * chi ** (-share) - share * chi ** (1 - share)
    ) ** (-1 / CRRA)


def euler(c, a, CRRA, share, intercept, vp_a):
    dufac = du_diff(c, a, CRRA, share, intercept)
    return dufac - vp_a


def euler2(c, a=None, CRRA=None, share=None, intercept=None, vp_a=None):
    return euler(c, a, CRRA, share, intercept, vp_a) ** 2


def euler2_diff(c, a=None, CRRA=None, share=None, intercept=None, vp_a=None):
    return (
        2
        * euler(c, a, CRRA, share, intercept, vp_a)
        * du2_diff(c, a, CRRA, share, intercept)
    )


def calc_m_nrm_next(shocks, b_nrm, perm_gro_fac):
    """
    Calculate future realizations of market resources mNrm from the income
    shock distribution S and normalized bank balances b.
    """
    return b_nrm / (shocks["PermShk"] * perm_gro_fac) + shocks["TranShk"]


def calc_dvdm_next(shocks, b_nrm, perm_gro_fac, crra, vp_func):
    """
    Evaluate realizations of marginal value of market resources next period,
    based on the income distribution S and values of bank balances bNrm
    """
    m_nrm = calc_m_nrm_next(shocks, b_nrm, perm_gro_fac)
    perm_shk_fac = shocks["PermShk"] * perm_gro_fac
    return perm_shk_fac ** (-crra) * vp_func(m_nrm)


def calc_end_dvda(shocks, a_nrm, share, rfree, dvdb_func):
    """
    Compute end-of-period marginal value of assets at values a, conditional
    on risky asset return S and risky share z.
    """
    # Calculate future realizations of bank balances bNrm
    ex_ret = shocks - rfree  # Excess returns
    rport = rfree + share * ex_ret  # Portfolio return
    b_nrm = rport * a_nrm

    # Calculate and return dvda
    return rport * dvdb_func(b_nrm)


def calc_end_dvds(shocks, a_nrm, share, rfree, dvdb_func):
    """
    Compute end-of-period marginal value of risky share at values a,
    conditional on risky asset return S and risky share z.
    """
    # Calculate future realizations of bank balances bNrm
    ex_ret = shocks - rfree  # Excess returns
    rport = rfree + share * ex_ret  # Portfolio return
    b_nrm = rport * a_nrm

    # Calculate and return dvds (second term is all zeros)
    return ex_ret * a_nrm * dvdb_func(b_nrm)


def calc_end_dvdx(shocks, a_nrm, share, rfree, dvdb_func):
    ex_ret = shocks - rfree  # Excess returns
    rport = rfree + share * ex_ret  # Portfolio return
    b_nrm = rport * a_nrm

    # Calculate and return dvds (second term is all zeros)
    dvdb = dvdb_func(b_nrm)
    dvda = rport * dvdb
    dvds = ex_ret * a_nrm * dvdb
    return dvda, dvds


def calc_med_v(shocks, b_nrm, perm_gro_fac, crra, v_func):
    """
    Calculate "intermediate" value from next period's bank balances, the
    income shocks S, and the risky asset share.
    """
    m_nrm = calc_m_nrm_next(shocks, b_nrm, perm_gro_fac)
    v_next = v_func(m_nrm)
    return (shocks["PermShk"] * perm_gro_fac) ** (1.0 - crra) * v_next


def calc_end_v(shocks, a_nrm, share, rfree, v_func):
    # Calculate future realizations of bank balances bNrm
    ex_ret = shocks - rfree
    rport = rfree + share * ex_ret
    b_nrm = rport * a_nrm

    return v_func(b_nrm)


def solve_one_period_WealthPortfolio(
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
    ShareLimit,
    vFuncBool,
    WealthShare,
    WealthShift,
    ChiFunc,
):
    # Make sure the individual is liquidity constrained.  Allowing a consumer to
    # borrow *and* invest in an asset with unbounded (negative) returns is a bad mix.
    if BoroCnstArt != 0.0:
        raise ValueError("PortfolioConsumerType must have BoroCnstArt=0.0!")

    # Define the current period utility function and effective discount factor
    uFunc = UtilityFuncCRRA(CRRA)
    DiscFacEff = DiscFac * LivPrb  # "effective" discount factor

    # Unpack next period's solution for easier access
    vp_func_next = solution_next.vPfuncAdj
    v_func_next = solution_next.vFuncAdj

    # Set a flag for whether the natural borrowing constraint is zero, which
    # depends on whether the smallest transitory income shock is zero
    BoroCnstNat_iszero = (np.min(IncShkDstn.atoms[1]) == 0.0) or (
        WealthShare != 0.0 and WealthShift == 0.0
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

    # Make tiled arrays to calculate future realizations of mNrm and Share when integrating over IncShkDstn
    bNrmNext = bNrmGrid

    # Define functions that are used internally to evaluate future realizations

    # Calculate end-of-period marginal value of assets and shares at each point
    # in aNrm and ShareGrid. Does so by taking expectation of next period marginal
    # values across income and risky return shocks.

    # Calculate intermediate marginal value of bank balances by taking expectations over income shocks
    med_dvdb = expected(
        calc_dvdm_next,
        IncShkDstn,
        args=(bNrmNext, PermGroFac, CRRA, vp_func_next),
    )
    med_dvdb_nvrs = uFunc.derinv(med_dvdb, order=(1, 0))
    med_dvdb_nvrs_func = LinearInterp(bNrmGrid, med_dvdb_nvrs)
    med_dvdb_func = MargValueFuncCRRA(med_dvdb_nvrs_func, CRRA)

    # Make tiled arrays to calculate future realizations of bNrm and Share when integrating over RiskyDstn
    aNrmNow, ShareNext = np.meshgrid(aNrmGrid, ShareGrid, indexing="ij")

    # Define functions for calculating end-of-period marginal value

    # Evaluate realizations of value and marginal value after asset returns are realized

    end_dvda, end_dvds = DiscFacEff * expected(
        calc_end_dvdx,
        RiskyDstn,
        args=(aNrmNow, ShareNext, Rfree, med_dvdb_func),
    )

    end_dvda_nvrs = uFunc.derinv(end_dvda)

    # Now find the optimal (continuous) risky share on [0,1] by solving the first
    # order condition end_dvds == 0.
    focs = end_dvds  # Relabel for convenient typing

    # For each value of aNrm, find the value of Share such that focs == 0
    crossing = np.logical_and(focs[:, 1:] <= 0.0, focs[:, :-1] >= 0.0)
    share_idx = np.argmax(crossing, axis=1)
    # This represents the index of the segment of the share grid where dvds flips
    # from positive to negative, indicating that there's a zero *on* the segment

    # Calculate the fractional distance between those share gridpoints where the
    # zero should be found, assuming a linear function; call it alpha
    a_idx = np.arange(aNrmCount)
    bot_s = ShareGrid[share_idx]
    top_s = ShareGrid[share_idx + 1]
    bot_f = focs[a_idx, share_idx]
    top_f = focs[a_idx, share_idx + 1]
    bot_c = end_dvda_nvrs[a_idx, share_idx]
    top_c = end_dvda_nvrs[a_idx, share_idx + 1]
    bot_dvda = end_dvda[a_idx, share_idx]
    top_dvda = end_dvda[a_idx, share_idx + 1]
    alpha = 1.0 - top_f / (top_f - bot_f)

    # Calculate the continuous optimal risky share and optimal consumption
    Share_now = (1.0 - alpha) * bot_s + alpha * top_s
    end_dvda_nvrs_now = (1.0 - alpha) * bot_c + alpha * top_c
    end_dvda_now = (1.0 - alpha) * bot_dvda + alpha * top_dvda

    # If agent wants to put more than 100% into risky asset, he is constrained.
    # Likewise if he wants to put less than 0% into risky asset, he is constrained.
    constrained_top = focs[:, -1] > 0.0
    constrained_bot = focs[:, 0] < 0.0

    # Apply those constraints to both risky share and consumption (but lower
    # constraint should never be relevant)
    Share_now[constrained_top] = 1.0
    Share_now[constrained_bot] = 0.0
    end_dvda_nvrs_now[constrained_top] = end_dvda_nvrs[constrained_top, -1]
    end_dvda_nvrs_now[constrained_bot] = end_dvda_nvrs[constrained_bot, 0]
    end_dvda_now[constrained_top] = end_dvda[constrained_top, -1]
    end_dvda_now[constrained_bot] = end_dvda[constrained_bot, 0]

    # When the natural borrowing constraint is *not* zero, then aNrm=0 is in the
    # grid, but there's no way to "optimize" the portfolio if a=0, and consumption
    # can't depend on the risky share if it doesn't meaningfully exist. Apply
    # a small fix to the bottom gridpoint (aNrm=0) when this happens.
    if not BoroCnstNat_iszero:
        Share_now[0] = 1.0
        end_dvda_nvrs_now[0] = end_dvda_nvrs[0, -1]
        end_dvda_now[0] = end_dvda[0, -1]

    # Construct functions characterizing the solution for this period

    # Now this is where we look for optimal C
    # for each a in the agrid find corresponding c that satisfies the euler equation

    if WealthShare == 0.0:
        cNrm_now = end_dvda_nvrs_now
    else:
        omega = end_dvda_nvrs_now / (aNrmGrid + WealthShift)
        cNrm_now = ChiFunc(omega) * (aNrmGrid + WealthShift)

    # Calculate the endogenous mNrm gridpoints when the agent adjusts his portfolio,
    # then construct the consumption function when the agent can adjust his share
    mNrm_now = np.insert(aNrmGrid + cNrm_now, 0, 0.0)
    cNrm_now = np.insert(cNrm_now, 0, 0.0)
    cFuncNow = LinearInterp(mNrm_now, cNrm_now)

    dudc_now = dudc(cNrm_now, mNrm_now - cNrm_now, CRRA, WealthShare, WealthShift)
    dudc_nvrs_now = uFunc.derinv(dudc_now, order=(1, 0))
    dudc_nvrs_func_now = LinearInterp(mNrm_now, dudc_nvrs_now)

    # Construct the marginal value (of mNrm) function
    vPfuncNow = MargValueFuncCRRA(dudc_nvrs_func_now, CRRA)

    # If the share choice is continuous, just make an ordinary interpolating function
    if BoroCnstNat_iszero:
        Share_lower_bound = ShareLimit
    else:
        Share_lower_bound = 1.0
    Share_now = np.insert(Share_now, 0, Share_lower_bound)
    ShareFuncNow = LinearInterp(mNrm_now, Share_now, ShareLimit, 0.0)

    # Add the value function if requested
    if vFuncBool:
        # Calculate intermediate value by taking expectations over income shocks
        med_v = expected(
            calc_med_v, IncShkDstn, args=(bNrmNext, PermGroFac, CRRA, v_func_next)
        )

        # Construct the "intermediate value function" for this period
        med_v_nvrs = uFunc.inv(med_v)
        med_v_nvrs_func = LinearInterp(bNrmGrid, med_v_nvrs)
        med_v_func = ValueFuncCRRA(med_v_nvrs_func, CRRA)

        # Calculate end-of-period value by taking expectations
        end_v = DiscFacEff * expected(
            calc_end_v,
            RiskyDstn,
            args=(aNrmNow, ShareNext, PermGroFac, CRRA, med_v_func),
        )
        end_v_nvrs = uFunc.inv(end_v)

        # Now make an end-of-period value function over aNrm and Share
        end_v_nvrs_func = BilinearInterp(end_v_nvrs, aNrmGrid, ShareGrid)
        end_v_func = ValueFuncCRRA(end_v_nvrs_func, CRRA)
        # This will be used later to make the value function for this period

        # Create the value functions for this period, defined over market resources
        # mNrm when agent can adjust his portfolio, and over market resources and
        # fixed share when agent can not adjust his portfolio.

        # Construct the value function
        mNrm_temp = aXtraGrid  # Just use aXtraGrid as our grid of mNrm values
        cNrm_temp = cFuncNow(mNrm_temp)
        aNrm_temp = np.maximum(mNrm_temp - cNrm_temp, 0.0)  # Fix tiny violations
        Share_temp = ShareFuncNow(mNrm_temp)
        v_temp = uFunc(cNrm_temp) + end_v_func(aNrm_temp, Share_temp)
        vNvrs_temp = uFunc.inv(v_temp)
        vNvrsP_temp = uFunc.der(cNrm_temp) * uFunc.inverse(v_temp, order=(0, 1))
        vNvrsFunc = CubicInterp(
            np.insert(mNrm_temp, 0, 0.0),  # x_list
            np.insert(vNvrs_temp, 0, 0.0),  # f_list
            np.insert(vNvrsP_temp, 0, vNvrsP_temp[0]),  # dfdx_list
        )
        # Re-curve the pseudo-inverse value function
        vFuncNow = ValueFuncCRRA(vNvrsFunc, CRRA)

    else:  # If vFuncBool is False, fill in dummy values
        vFuncNow = NullFunc()

    # Package and return the solution
    solution_now = PortfolioSolution(
        cFuncAdj=cFuncNow,
        ShareFuncAdj=ShareFuncNow,
        vPfuncAdj=vPfuncNow,
        vFuncAdj=vFuncNow,
    )
    return solution_now


init_wealth_portfolio = init_portfolio.copy()
init_wealth_portfolio["WealthShare"] = 0.5
init_wealth_portfolio["WealthShift"] = 0.1

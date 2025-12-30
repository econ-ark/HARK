from copy import deepcopy

import numpy as np
from HARK.ConsumptionSaving.ConsPortfolioModel import (
    PortfolioConsumerType,
    PortfolioSolution,
    make_portfolio_solution_terminal,
)
from HARK.distributions import expected
from HARK.interpolation import (
    BilinearInterp,
    ConstantFunction,
    CubicInterp,
    LinearInterp,
    MargValueFuncCRRA,
    ValueFuncCRRA,
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
    make_simple_ShareGrid,
    make_AdjustDstn,
)
from HARK.ConsumptionSaving.ConsWealthUtilityModel import (
    make_ChiFromOmega_function,
)
from HARK.ConsumptionSaving.ConsIndShockModel import (
    make_lognormal_kNrm_init_dstn,
    make_lognormal_pLvl_init_dstn,
)
from HARK.rewards import UtilityFuncCRRA
from HARK.utilities import NullFunc, make_assets_grid


def utility(c, a, CRRA, share=0.0, intercept=0.0):
    w = a + intercept
    return (c ** (1 - share) * w**share) ** (1 - CRRA) / (1 - CRRA)


def dudc(c, a, CRRA, share=0.0, intercept=0.0):
    u = utility(c, a, CRRA, share, intercept)
    return u * (1 - CRRA) * (1 - share) / c


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


###############################################################################


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
    """
    TODO: Fill in this missing docstring.

    Parameters
    ----------
    solution_next : TYPE
        DESCRIPTION.
    IncShkDstn : TYPE
        DESCRIPTION.
    RiskyDstn : TYPE
        DESCRIPTION.
    LivPrb : TYPE
        DESCRIPTION.
    DiscFac : TYPE
        DESCRIPTION.
    CRRA : TYPE
        DESCRIPTION.
    Rfree : TYPE
        DESCRIPTION.
    PermGroFac : TYPE
        DESCRIPTION.
    BoroCnstArt : TYPE
        DESCRIPTION.
    aXtraGrid : TYPE
        DESCRIPTION.
    ShareGrid : TYPE
        DESCRIPTION.
    ShareLimit : TYPE
        DESCRIPTION.
    vFuncBool : TYPE
        DESCRIPTION.
    WealthShare : TYPE
        DESCRIPTION.
    WealthShift : TYPE
        DESCRIPTION.
    ChiFunc : TYPE
        DESCRIPTION.

    Returns
    -------
    solution_now : TYPE
        DESCRIPTION.

    """
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
            args=(aNrmNow, ShareNext, Rfree, med_v_func),
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


###############################################################################

# Make a dictionary of constructors for the wealth-in-utility portfolio choice consumer type
WealthPortfolioConsumerType_constructors_default = {
    "IncShkDstn": construct_lognormal_income_process_unemployment,
    "PermShkDstn": get_PermShkDstn_from_IncShkDstn,
    "TranShkDstn": get_TranShkDstn_from_IncShkDstn,
    "aXtraGrid": make_assets_grid,
    "RiskyDstn": make_lognormal_RiskyDstn,
    "ShockDstn": combine_IncShkDstn_and_RiskyDstn,
    "ShareLimit": calc_ShareLimit_for_CRRA,
    "ShareGrid": make_simple_ShareGrid,
    "ChiFunc": make_ChiFromOmega_function,
    "AdjustDstn": make_AdjustDstn,
    "kNrmInitDstn": make_lognormal_kNrm_init_dstn,
    "pLvlInitDstn": make_lognormal_pLvl_init_dstn,
    "solution_terminal": make_portfolio_solution_terminal,
}

# Default parameters to make IncShkDstn using construct_lognormal_income_process_unemployment
WealthPortfolioConsumerType_IncShkDstn_default = {
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
WealthPortfolioConsumerType_aXtraGrid_default = {
    "aXtraMin": 0.001,  # Minimum end-of-period "assets above minimum" value
    "aXtraMax": 100,  # Maximum end-of-period "assets above minimum" value
    "aXtraNestFac": 1,  # Exponential nesting factor for aXtraGrid
    "aXtraCount": 200,  # Number of points in the grid of "assets above minimum"
    "aXtraExtra": None,  # Additional other values to add in grid (optional)
}

# Default parameters to make RiskyDstn with make_lognormal_RiskyDstn (and uniform ShareGrid)
WealthPortfolioConsumerType_RiskyDstn_default = {
    "RiskyAvg": 1.08,  # Mean return factor of risky asset
    "RiskyStd": 0.18362634887,  # Stdev of log returns on risky asset
    "RiskyCount": 5,  # Number of integration nodes to use in approximation of risky returns
}

WealthPortfolioConsumerType_ShareGrid_default = {
    "ShareCount": 25  # Number of discrete points in the risky share approximation
}

# Default parameters to make ChiFunc with make_ChiFromOmega_function
WealthPortfolioConsumerType_ChiFunc_default = {
    "ChiFromOmega_N": 501,  # Number of gridpoints in chi-from-omega function
    "ChiFromOmega_bound": 15,  # Highest gridpoint to use for it
}

# Make a dictionary with parameters for the default constructor for kNrmInitDstn
WealthPortfolioConsumerType_kNrmInitDstn_default = {
    "kLogInitMean": -12.0,  # Mean of log initial capital
    "kLogInitStd": 0.0,  # Stdev of log initial capital
    "kNrmInitCount": 15,  # Number of points in initial capital discretization
}

# Make a dictionary with parameters for the default constructor for pLvlInitDstn
WealthPortfolioConsumerType_pLvlInitDstn_default = {
    "pLogInitMean": 0.0,  # Mean of log permanent income
    "pLogInitStd": 0.0,  # Stdev of log permanent income
    "pLvlInitCount": 15,  # Number of points in initial capital discretization
}

# Make a dictionary to specify a risky asset consumer type
WealthPortfolioConsumerType_solving_default = {
    # BASIC HARK PARAMETERS REQUIRED TO SOLVE THE MODEL
    "cycles": 1,  # Finite, non-cyclic model
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    "constructors": WealthPortfolioConsumerType_constructors_default,  # See dictionary above
    # PRIMITIVE RAW PARAMETERS REQUIRED TO SOLVE THE MODEL
    "CRRA": 5.0,  # Coefficient of relative risk aversion
    "Rfree": [1.03],  # Return factor on risk free asset
    "DiscFac": 0.90,  # Intertemporal discount factor
    "LivPrb": [0.98],  # Survival probability after each period
    "PermGroFac": [1.01],  # Permanent income growth factor
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint
    "WealthShare": 0.5,  # Share of wealth in Cobb-Douglas aggregator in utility function
    "WealthShift": 0.1,  # Shifter for wealth in utility function
    "DiscreteShareBool": False,  # Whether risky asset share is restricted to discrete values
    "PortfolioBool": True,  # Whether there is portfolio choice
    "PortfolioBisect": False,  # This is a mystery parameter
    "IndepDstnBool": True,  # Whether income and return shocks are independent
    "vFuncBool": False,  # Whether to calculate the value function during solution
    "CubicBool": False,  # Whether to use cubic spline interpolation when True
    # (Uses linear spline interpolation for cFunc when False)
    "AdjustPrb": 1.0,  # Probability that the agent can update their risky portfolio share each period
    "RiskyShareFixed": None,  # This just needs to exist because of inheritance, does nothing
    "sim_common_Rrisky": True,  # Whether risky returns have a shared/common value across agents
}
WealthPortfolioConsumerType_simulation_default = {
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

# Assemble the default dictionary
WealthPortfolioConsumerType_default = {}
WealthPortfolioConsumerType_default.update(WealthPortfolioConsumerType_solving_default)
WealthPortfolioConsumerType_default.update(
    WealthPortfolioConsumerType_simulation_default
)
WealthPortfolioConsumerType_default.update(
    WealthPortfolioConsumerType_aXtraGrid_default
)
WealthPortfolioConsumerType_default.update(
    WealthPortfolioConsumerType_ShareGrid_default
)
WealthPortfolioConsumerType_default.update(
    WealthPortfolioConsumerType_IncShkDstn_default
)
WealthPortfolioConsumerType_default.update(
    WealthPortfolioConsumerType_RiskyDstn_default
)
WealthPortfolioConsumerType_default.update(WealthPortfolioConsumerType_ChiFunc_default)
WealthPortfolioConsumerType_default.update(
    WealthPortfolioConsumerType_kNrmInitDstn_default
)
WealthPortfolioConsumerType_default.update(
    WealthPortfolioConsumerType_pLvlInitDstn_default
)
init_wealth_portfolio = WealthPortfolioConsumerType_default

###############################################################################


class WealthPortfolioConsumerType(PortfolioConsumerType):
    """
    TODO: This docstring is missing and needs to be written.
    """

    time_inv_ = deepcopy(PortfolioConsumerType.time_inv_)
    time_inv_ = time_inv_ + [
        "WealthShare",
        "WealthShift",
        "ChiFunc",
        "RiskyDstn",
    ]
    default_ = {
        "params": init_wealth_portfolio,
        "solver": solve_one_period_WealthPortfolio,
        "model": "ConsRiskyAsset.yaml",
    }

    def pre_solve(self):
        self.construct("solution_terminal")
        self.solution_terminal.ShareFunc = ConstantFunction(1.0)

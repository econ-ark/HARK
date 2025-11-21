"""
This module has consumption-saving models with wealth directly in the utility function.
Unlike the model in ConsWealthPortfolioModel.py, these models do not have portfolio
allocation between a risky and riskless asset. Two AgentType subclasses will be covered:

1) WealthUtilityConsumerType: Agents who face transitory and permanent shocks to labor
   income and who can save in a riskless asset, and have CRRA preferences over a Cobb-
   Douglas composition of consumption and retained assets. This is WealthPortfolioConsumerType
   with no portfolio allocation.

2) CapitalistSpiritConsumerType: Agents who face transitory and permanent shocks to labor
   income and who can save in a riskless asset, and have *additive* CRRA preferences over
   consumption and assets, with a *lower* CRRA for assets than consumption.
"""

import numpy as np
from copy import deepcopy

from HARK.distributions import expected
from HARK.interpolation import (
    LinearInterp,
    LowerEnvelope,
    ValueFuncCRRA,
    MargValueFuncCRRA,
)
from HARK.Calibration.Income.IncomeProcesses import (
    construct_lognormal_income_process_unemployment,
    get_PermShkDstn_from_IncShkDstn,
    get_TranShkDstn_from_IncShkDstn,
)
from HARK.ConsumptionSaving.ConsIndShockModel import (
    calc_boro_const_nat,
    calc_m_nrm_min,
    make_lognormal_kNrm_init_dstn,
    make_lognormal_pLvl_init_dstn,
    IndShockConsumerType,
    ConsumerSolution,
    make_basic_CRRA_solution_terminal,
)
from HARK.rewards import UtilityFuncCRRA
from HARK.utilities import NullFunc, make_assets_grid


class ChiFromOmegaFunction:
    """
    A class for representing a function that takes in values of omega = EndOfPrdvPnvrs / aNrm
    and returns the corresponding optimal chi = cNrm / aNrm. The only parameters
    that matter for this transformation are the coefficient of relative risk
    aversion (rho) and the share of wealth in the Cobb-Douglas aggregator (delta).

    Parameters
    ----------
    CRRA : float
        Coefficient of relative risk aversion.
    WealthShare : float
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
        r = self.CRRA
        d = self.WealthShare
        return x ** (1 - d) * ((1 - d) * x ** (-d) - d * x ** (1 - d)) ** (-1 / r)

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


# Trivial constructor function
def make_ChiFromOmega_function(CRRA, WealthShare, ChiFromOmega_N, ChiFromOmega_bound):
    if WealthShare == 0.0:
        return NullFunc()
    return ChiFromOmegaFunction(
        CRRA, WealthShare, N=ChiFromOmega_N, z_bound=ChiFromOmega_bound
    )


def utility(c, a, CRRA, share=0.0, intercept=0.0):
    w = a + intercept
    return (c ** (1 - share) * w**share) ** (1 - CRRA) / (1 - CRRA)


def dudc(c, a, CRRA, share=0.0, intercept=0.0):
    u = utility(c, a, CRRA, share, intercept)
    return u * (1 - CRRA) * (1 - share) / c


def calc_m_nrm_next(shocks, a_nrm, G, R):
    """
    Calculate future realizations of market resources mNrm from the income
    shock distribution S and end-of-period assets a_nrm.
    """
    return R * a_nrm / (shocks["PermShk"] * G) + shocks["TranShk"]


def calc_dvdm_next(shocks, a_nrm, G, R, rho, vp_func):
    """
    Evaluate realizations of marginal value of market resources next period,
    based on the income distribution S and values of end-of-period assets a_nrm
    """
    m_nrm = calc_m_nrm_next(shocks, a_nrm, G, R)
    perm_shk_fac = shocks["PermShk"] * G
    return perm_shk_fac ** (-rho) * vp_func(m_nrm)


def calc_v_next(shocks, a_nrm, G, R, rho, v_func):
    """
    Evaluate realizations of value of market resources next period, based on the
    income distribution S and values of end-of-period assets a_nrm.
    """
    m_nrm = calc_m_nrm_next(shocks, a_nrm, G, R)
    v_next = v_func(m_nrm)
    return (shocks["PermShk"] * G) ** (1.0 - rho) * v_next


def solve_one_period_WealthUtility(
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
    WealthShare,
    WealthShift,
    ChiFunc,
):
    """
    Solve one period of the wealth-in-utility consumption-saving problem, conditional
    on the solution to the succeeding period.

    Parameters
    ----------
    solution_next : ConsumerSolution
        Solution to the succeeding period's problem, which must include a vPfunc,
        among other attributes.
    IncShkDstn : Distribution
        Distribution of next period's permanent and transitory income shocks, discretized.
    LivPrb : float
        Survival probability at the end of this period.
    DiscFac : float
        Intertemporal discount factor.
    CRRA : float
        Coefficient of relative risk aversion on composite of consumption and wealth.
    Rfree : float
        Risk-free return factor on retained assets.
    PermGroFac : float
        Expected growth rate of permanent income from this period to the next.
    BoroCnstArt : float or None
        Artificial borrowing constraint on retained assets.
    aXtraGrid : np.array
        Grid of end-of-period assets values.
    vFuncBool : bool
        Indicator for whether the value function should be constructed.
    CubicBool : bool
        Indicator for whether the consumption function should use cubic spline
        interpolation (True) or linear splines (False).
    WealthShare : float
        Share of wealth in the Cobb-Douglas composition of consumption and wealth,
        which should be between 0 and 1.
    WealthShift : float
        Shifter on wealth in the Cobb-Douglas composition, which should be non-negative.
    ChiFunc : function
        Mapping from omega = EndOfPrdvPnvrs / aNrm to the optimal chi = cNrm / aNrm.

    Returns
    -------
    solution_now : ConsumerSolution
        Solution to this period's problem, including the consumption function cFunc.

    """
    # Raise an error if cubic interpolation was requested
    if CubicBool:
        raise NotImplementedError(
            "Cubic interpolation hasn't been programmed for the wealth in utility model yet."
        )

    # Define the current period utility function and effective discount factor
    uFunc = UtilityFuncCRRA(CRRA)
    DiscFacEff = DiscFac * LivPrb  # "effective" discount factor

    # Unpack next period's solution for easier access
    vPfuncNext = solution_next.vPfunc
    vFuncNext = solution_next.vFunc

    # Calculate the minimum allowable value of money resources in this period
    BoroCnstNat = calc_boro_const_nat(
        solution_next.mNrmMin, IncShkDstn, Rfree, PermGroFac
    )
    BoroCnstNat = np.maximum(-WealthShift, BoroCnstNat)

    # Set the minimum allowable (normalized) market resources based on the natural
    # and artificial borrowing constraints
    mNrmMinNow = calc_m_nrm_min(BoroCnstArt, BoroCnstNat)

    # Make a grid of end-of-period assets
    aNrmGrid = aXtraGrid + BoroCnstNat

    # Calculate marginal value of end-of-period assets by taking expectations over income shocks
    exp_dvdm = expected(
        calc_dvdm_next,
        IncShkDstn,
        args=(aNrmGrid, PermGroFac, Rfree, CRRA, vPfuncNext),
    )
    dvda_end_of_prd = DiscFacEff * Rfree * exp_dvdm
    dvda_nvrs = uFunc.derinv(dvda_end_of_prd, order=(1, 0))

    # Calculate optimal consumption for each end-of-period assets value
    if WealthShare == 0.0:
        cNrm_now = dvda_nvrs
    else:
        wealth_temp = aXtraGrid + WealthShift
        omega = dvda_nvrs / wealth_temp
        cNrm_now = ChiFunc(omega) * wealth_temp

    # Calculate the endogenous mNrm gridpoints, then construct the consumption function
    mNrm_now = np.insert(aNrmGrid + cNrm_now, 0, BoroCnstNat)
    cNrm_now = np.insert(cNrm_now, 0, 0.0)
    cFuncUnc = LinearInterp(mNrm_now, cNrm_now)
    cFuncCnst = LinearInterp([mNrmMinNow, mNrmMinNow + 1.0], [0.0, 1.0])
    cFuncNow = LowerEnvelope(cFuncUnc, cFuncCnst)

    # Calculate marginal value of market resources as the marginal utility of consumption
    m_temp = aXtraGrid.copy() + mNrmMinNow
    c_temp = cFuncNow(m_temp)
    a_temp = m_temp - c_temp
    dudc_now = dudc(c_temp, a_temp, CRRA, WealthShare, WealthShift)
    dudc_nvrs_now = np.insert(uFunc.derinv(dudc_now, order=(1, 0)), 0, 0.0)
    dudc_nvrs_func_now = LinearInterp(np.insert(m_temp, 0, mNrmMinNow), dudc_nvrs_now)

    # Construct the marginal value (of mNrm) function
    vPfuncNow = MargValueFuncCRRA(dudc_nvrs_func_now, CRRA)

    # Add the value function if requested
    if vFuncBool:
        EndOfPrd_v = expected(
            calc_v_next, IncShkDstn, args=(a_temp, PermGroFac, Rfree, CRRA, vFuncNext)
        )
        EndOfPrd_v *= DiscFacEff
        u_now = utility(c_temp, a_temp, CRRA, WealthShare, WealthShift)
        v_now = u_now + EndOfPrd_v
        vNvrs_now = np.insert(uFunc.inverse(v_now), 0, 0.0)
        vNvrsFunc = LinearInterp(np.insert(m_temp, 0, mNrmMinNow), vNvrs_now)
        vFuncNow = ValueFuncCRRA(vNvrsFunc, CRRA)
    else:
        vFuncNow = NullFunc()

    # Package and return the solution
    solution_now = ConsumerSolution(
        cFunc=cFuncNow,
        vPfunc=vPfuncNow,
        vFunc=vFuncNow,
        mNrmMin=mNrmMinNow,
    )
    return solution_now


###############################################################################

# Make a dictionary of constructors for the wealth-in-utility portfolio choice consumer type
WealthUtility_constructors_default = {
    "IncShkDstn": construct_lognormal_income_process_unemployment,
    "PermShkDstn": get_PermShkDstn_from_IncShkDstn,
    "TranShkDstn": get_TranShkDstn_from_IncShkDstn,
    "aXtraGrid": make_assets_grid,
    "ChiFunc": make_ChiFromOmega_function,
    "kNrmInitDstn": make_lognormal_kNrm_init_dstn,
    "pLvlInitDstn": make_lognormal_pLvl_init_dstn,
    "solution_terminal": make_basic_CRRA_solution_terminal,
}

# Default parameters to make IncShkDstn using construct_lognormal_income_process_unemployment
WealthUtility_IncShkDstn_default = {
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
WealthUtility_aXtraGrid_default = {
    "aXtraMin": 0.001,  # Minimum end-of-period "assets above minimum" value
    "aXtraMax": 20,  # Maximum end-of-period "assets above minimum" value
    "aXtraNestFac": 2,  # Exponential nesting factor for aXtraGrid
    "aXtraCount": 48,  # Number of points in the grid of "assets above minimum"
    "aXtraExtra": None,  # Additional other values to add in grid (optional)
}

# Default parameters to make ChiFunc with make_ChiFromOmega_function
WealthUtility_ChiFunc_default = {
    "ChiFromOmega_N": 501,  # Number of gridpoints in chi-from-omega function
    "ChiFromOmega_bound": 15,  # Highest gridpoint to use for it
}

# Make a dictionary with parameters for the default constructor for kNrmInitDstn
WealthUtility_kNrmInitDstn_default = {
    "kLogInitMean": -12.0,  # Mean of log initial capital
    "kLogInitStd": 0.0,  # Stdev of log initial capital
    "kNrmInitCount": 15,  # Number of points in initial capital discretization
}

# Make a dictionary with parameters for the default constructor for pLvlInitDstn
WealthUtility_pLvlInitDstn_default = {
    "pLogInitMean": 0.0,  # Mean of log permanent income
    "pLogInitStd": 0.0,  # Stdev of log permanent income
    "pLvlInitCount": 15,  # Number of points in initial capital discretization
}

# Make a dictionary to specify a risky asset consumer type
WealthUtility_solving_default = {
    # BASIC HARK PARAMETERS REQUIRED TO SOLVE THE MODEL
    "cycles": 1,  # Finite, non-cyclic model
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    "constructors": WealthUtility_constructors_default,  # See dictionary above
    # PRIMITIVE RAW PARAMETERS REQUIRED TO SOLVE THE MODEL
    "CRRA": 2.0,  # Coefficient of relative risk aversion
    "Rfree": [1.03],  # Return factor on risk free asset
    "DiscFac": 0.96,  # Intertemporal discount factor
    "LivPrb": [0.98],  # Survival probability after each period
    "PermGroFac": [1.01],  # Permanent income growth factor
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint
    "WealthShare": 0.5,  # Share of wealth in Cobb-Douglas aggregator in utility function
    "WealthShift": 0.1,  # Shifter for wealth in utility function
    "vFuncBool": False,  # Whether to calculate the value function during solution
    "CubicBool": False,  # Whether to use cubic spline interpolation when True
}
WealthUtility_simulation_default = {
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

WealthUtilityConsumerType_default = {}
WealthUtilityConsumerType_default.update(WealthUtility_solving_default)
WealthUtilityConsumerType_default.update(WealthUtility_simulation_default)
WealthUtilityConsumerType_default.update(WealthUtility_kNrmInitDstn_default)
WealthUtilityConsumerType_default.update(WealthUtility_pLvlInitDstn_default)
WealthUtilityConsumerType_default.update(WealthUtility_aXtraGrid_default)
WealthUtilityConsumerType_default.update(WealthUtility_IncShkDstn_default)
WealthUtilityConsumerType_default.update(WealthUtility_ChiFunc_default)
init_wealth_utility = WealthUtilityConsumerType_default


class WealthUtilityConsumerType(IndShockConsumerType):
    r"""
    Class for representing consumers who face idiosyncratic income risk and can save in
    a risk-free asset, and have CRRA preferences over a Cobb-Douglas composite of assets
    and consumption.

    .. math::
        \newcommand{\CRRA}{\rho}
        \newcommand{\LivPrb}{\mathsf{S}}
        \newcommand{\PermGroFac}{\Gamma}
        \newcommand{\Rfree}{\mathsf{R}}
        \newcommand{\DiscFac}{\beta}
        \newcommand{\WealthShare}{\alpha}
        \newcommand{\WealthShift}{\omega}
        \begin{align*}
        v_t(m_t) &= \max_{c_t}u(x_t) + \DiscFac \LivPrb_{t+1} \mathbb{E}_{t} \left[ (\PermGroFac_{t+1} \psi_{t+1})^{1-\CRRA} v_{t+1}(m_{t+1}) \right], \\
        & \text{s.t.}  \\
        x_t &= (a_t + \WealthShift)^\WealthShare c_t^{1-\WealthShare}, \\
        a_t &= m_t - c_t, \\
        a_t &\geq \underline{a}, \\
        m_{t+1} &= a_t \Rfree_{t+1}/(\PermGroFac_{t+1} \psi_{t+1}) + \theta_{t+1}, \\
        (\psi_{t+1},\theta_{t+1}) &\sim F_{t+1}, \\
        \mathbb{E}[\psi]=\mathbb{E}[\theta] &= 1, \\
        u(c) &= \frac{c^{1-\CRRA}}{1-\CRRA}
        \end{align*}

    """

    time_inv_ = deepcopy(IndShockConsumerType.time_inv_)
    time_inv_ = time_inv_ + [
        "WealthShare",
        "WealthShift",
        "ChiFunc",
    ]
    default_ = {
        "params": init_wealth_utility,
        "solver": solve_one_period_WealthUtility,
        "model": "ConsIndShock.yaml",
    }

    def pre_solve(self):
        self.construct("solution_terminal")

    def make_euler_error_func(self, mMax=100, approx_inc_dstn=True):  # pragma: nocover
        raise NotImplementedError()

    def check_conditions(self, verbose):  # pragma: nocover
        raise NotImplementedError()

    def calc_limiting_values(self):  # pragma: nocover
        raise NotImplementedError()

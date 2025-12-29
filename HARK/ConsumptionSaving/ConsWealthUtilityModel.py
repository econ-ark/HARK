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
    ConstantFunction,
    LinearInterp,
    CubicInterp,
    LowerEnvelope,
    LowerEnvelope2D,
    ValueFuncCRRA,
    MargValueFuncCRRA,
    MargMargValueFuncCRRA,
    UpperEnvelope,
    BilinearInterp,
    VariableLowerBoundFunc2D,
    LinearInterpOnInterp1D,
)
from HARK.Calibration.Income.IncomeProcesses import (
    construct_lognormal_income_process_unemployment,
    get_PermShkDstn_from_IncShkDstn,
    get_TranShkDstn_from_IncShkDstn,
    make_AR1_style_pLvlNextFunc,
    make_pLvlGrid_by_simulation,
    make_basic_pLvlPctiles,
)
from HARK.ConsumptionSaving.ConsIndShockModel import (
    calc_boro_const_nat,
    calc_m_nrm_min,
    make_lognormal_kNrm_init_dstn,
    make_lognormal_pLvl_init_dstn,
    IndShockConsumerType,
    ConsumerSolution,
)
from HARK.ConsumptionSaving.ConsGenIncProcessModel import (
    GenIncProcessConsumerType,
)
from HARK.rewards import UtilityFuncCRRA, CRRAutility
from HARK.utilities import NullFunc, make_assets_grid


def make_terminal_solution_for_wealth_in_utility(CRRA, WealthShare, WealthShift):
    """
    Construct the terminal period solution for a consumption-saving model with
    CRRA utility over a composite of wealth and consumption.

    Parameters
    ----------
    CRRA : float
        Coefficient of relative risk aversion.
    WealthShare : float
        Wealth's share in the Cobb-Douglas aggregator.
    WealthShift : float
        Additive shifter for wealth in the Cobb-Douglas aggregator.

    Returns
    -------
    solution_terminal : ConsumerSolution
        Terminal period solution for someone with the given CRRA.
    """
    if (WealthShift > 0.0) and (WealthShare > 0.0):
        m_cusp = (1 - WealthShare) / WealthShare * WealthShift
        m_terminal = np.array([0.0, m_cusp, m_cusp + 1.0])
        c_terminal = np.array([0.0, m_cusp, m_cusp + (1.0 - WealthShare)])
    else:
        m_terminal = np.array([0.0, 1.0])
        c_terminal = np.array([0.0, 1.0 - WealthShare])

    cFunc_terminal = LinearInterp(m_terminal, c_terminal)
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
        MPCmin=1.0 - WealthShare,
        MPCmax=1.0,
    )
    return solution_terminal


def make_2D_CRRA_solution_empty(CRRA):
    """
    Construct the pseudo-terminal period solution for a consumption-saving model with CRRA
    utility and two state variables: levels of market resources and permanent income.
    All functions return zero everywhere.

    Parameters
    ----------
    CRRA : float
        Coefficient of relative risk aversion. This is the only relevant parameter.

    Returns
    -------
    solution_terminal : ConsumerSolution
        Terminal period solution for someone with the given CRRA.
    """
    cFunc_terminal = ConstantFunction(0.0)
    vFunc_terminal = ConstantFunction(0.0)
    vPfunc_terminal = ConstantFunction(0.0)
    vPPfunc_terminal = ConstantFunction(0.0)
    solution_terminal = ConsumerSolution(
        cFunc=cFunc_terminal,
        vFunc=vFunc_terminal,
        vPfunc=vPfunc_terminal,
        vPPfunc=vPPfunc_terminal,
        mNrmMin=ConstantFunction(0.0),
        hNrm=ConstantFunction(0.0),
        MPCmin=1.0,
        MPCmax=1.0,
    )
    solution_terminal.hLvl = solution_terminal.hNrm
    solution_terminal.mLvlMin = solution_terminal.mNrmMin
    return solution_terminal


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
        return chi


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
    "solution_terminal": make_terminal_solution_for_wealth_in_utility,
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
    "pseudo_terminal": False,  # solution_terminal really is part of solution
    # PRIMITIVE RAW PARAMETERS REQUIRED TO SOLVE THE MODEL
    "CRRA": 2.0,  # Coefficient of relative risk aversion
    "Rfree": [1.03],  # Return factor on risk free asset
    "DiscFac": 0.96,  # Intertemporal discount factor
    "LivPrb": [0.98],  # Survival probability after each period
    "PermGroFac": [1.01],  # Permanent income growth factor
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint
    "WealthShare": 0.2,  # Share of wealth in Cobb-Douglas aggregator in utility function
    "WealthShift": 0.0,  # Shifter for wealth in utility function
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
        \newcommand{\WealthShift}{\xi}

        \begin{equation*}
        v_t(m_t) = \max_{c_t}u(x_t) + \DiscFac \LivPrb_{t} \mathbb{E}_{t} \left[ (\PermGroFac_{t+1} \psi_{t+1})^{1-\CRRA} v_{t+1}(m_{t+1}) \right] ~~\text{s.t.}
        \end{equation*}
        \begin{align*}
        x_t &=& (a_t + \WealthShift)^\WealthShare c_t^{1-\WealthShare}, \\
        a_t &=& m_t - c_t, \\
        a_t &\geq& \underline{a}, \\
        m_{t+1} &=& a_t \Rfree_{t+1}/(\PermGroFac_{t+1} \psi_{t+1}) + \theta_{t+1}, \\
        (\psi_{t+1},\theta_{t+1}) &\sim& F_{t+1}, \\
        \mathbb{E}[\psi] &=& 1, \\
        u(x) &=& \frac{x^{1-\CRRA}}{1-\CRRA}.
        \end{align*}

    Constructors
    ------------
    IncShkDstn: Constructor, :math:`\psi`, :math:`\theta`
        The agent's income shock distributions.

        Its default constructor is :func:`HARK.Calibration.Income.IncomeProcesses.construct_lognormal_income_process_unemployment`
    aXtraGrid: Constructor
        The agent's asset grid.

        Its default constructor is :func:`HARK.utilities.make_assets_grid`

    Solving Parameters
    ------------------
    cycles: int
        0 specifies an infinite horizon model, 1 specifies a finite model.
    T_cycle: int
        Number of periods in the cycle for this agent type.
    CRRA: float, :math:`\rho`
        Coefficient of Relative Risk Aversion.
    WealthShift : float, :math:`\xi`
        Additive shifter for wealth in the utility function.
    WealthShare : float, :math:`\alpha`
        Cobb-Douglas share for wealth in the utility function.
    Rfree: float or list[float], time varying, :math:`\mathsf{R}`
        Risk Free interest rate. Pass a list of floats to make Rfree time varying.
    DiscFac: float, :math:`\beta`
        Intertemporal discount factor.
    LivPrb: list[float], time varying, :math:`1-\mathsf{D}`
        Survival probability after each period.
    PermGroFac: list[float], time varying, :math:`\Gamma`
        Permanent income growth factor.
    BoroCnstArt: float, :math:`\underline{a}`
        The minimum Asset/Permanent Income ratio, None to ignore.
    vFuncBool: bool
        Whether to calculate the value function during solution.
    CubicBool: bool
        Whether to use cubic spline interpolation.

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


###############################################################################


def solve_one_period_CapitalistSpirit(
    solution_next,
    IncShkDstn,
    LivPrb,
    DiscFac,
    CRRA,
    WealthCurve,
    WealthFac,
    WealthShift,
    Rfree,
    pLvlNextFunc,
    BoroCnstArt,
    aXtraGrid,
    pLvlGrid,
    vFuncBool,
    CubicBool,
):
    """
    Solves one one period problem of a consumer who experiences persistent and
    transitory shocks to his income. Unlike in ConsIndShock, consumers do not
    necessarily have the same predicted level of p next period as this period
    (after controlling for growth).  Instead, they have  a function that translates
    current persistent income into expected next period persistent income (subject
    to shocks).

    Moreover, the agent's preferences follow the "capitalist spirit" model, so that
    end-of-period assets yield additively separable CRRA utility with a *lower*
    coefficient than for consumption. This causes the saving rate to increase as
    wealth increases, eventually approaching 100%.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncShkDstn : distribution.Distribution
        A discrete approximation to the income shocks between the period being
        solved and the one immediately following (in solution_next).
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion for consumption.
    WealthCurve : float
        Ratio of CRRA for consumption to CRRA for wealth. Must be strictly between
        zero and one.
    WealthFac : float
        Weighting factor on utility of wealth relative to utility of consumption.
        Should be non-negative.
    WealthShift : float
        Shifter for wealth when calculating utility. Should be non-negative.
    Rfree : float
        Risk free interest factor on end-of-period assets.
    pLvlNextFunc : float
        Expected persistent income next period as a function of current pLvl.
    BoroCnstArt: float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.
    aXtraGrid: np.array
        Array of "extra" end-of-period (normalized) asset values-- assets
        above the absolute minimum acceptable level.
    pLvlGrid: np.array
        Array of persistent income levels at which to solve the problem.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear interpolation.

    Returns
    -------
    solution_now : ConsumerSolution
        Solution to this period's consumption-saving problem.
    """
    if WealthCurve >= 1.0:
        raise ValueError("WealthCurve must be less than 1!")
    if WealthCurve <= 0.0:
        raise ValueError("WealthCurve must be greater than 0!")
    if WealthFac < 0.0:
        raise ValueError("WealthFac cannot be negative!")
    if WealthShift < 0.0:
        raise ValueError("WealthShift cannot be negative!")

    # Define the utility functions for this period
    uFuncCon = UtilityFuncCRRA(CRRA)
    DiscFacEff = DiscFac * LivPrb  # "effective" discount factor
    CRRAwealth = CRRA * WealthCurve
    uFuncWealth = lambda a: WealthFac * CRRAutility(a + WealthShift, rho=CRRAwealth)

    if vFuncBool and (CRRA >= 1.0) and (CRRAwealth < 1.0):
        raise ValueError(
            "Can't construct a good representation of value function when rho > 1 > nu!"
        )

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
    mLvlMinNext = solution_next.mLvlMin
    hLvlNext = solution_next.hLvl

    # Define some functions for calculating future expectations
    def calc_pLvl_next(S, p):
        return pLvlNextFunc(p) * S["PermShk"]

    def calc_mLvl_next(S, a, p_next):
        return Rfree * a + p_next * S["TranShk"]

    def calc_hLvl(S, p):
        pLvl_next = pLvlNextFunc(p) * S["PermShk"]
        hLvl = S["TranShk"] * pLvl_next + hLvlNext(pLvl_next)
        return hLvl

    def calc_v_next(S, a, p):
        pLvl_next = calc_pLvl_next(S, p)
        mLvl_next = calc_mLvl_next(S, a, pLvl_next)
        v_next = vFuncNext(mLvl_next, pLvl_next)
        return v_next

    def calc_vP_next(S, a, p):
        pLvl_next = pLvlNextFunc(p) * S["PermShk"]
        mLvl_next = calc_mLvl_next(S, a, pLvl_next)
        vP_next = vPfuncNext(mLvl_next, pLvl_next)
        return vP_next

    # Construct human wealth level as a function of productivity pLvl
    hLvlGrid = 1.0 / Rfree * expected(calc_hLvl, IncShkDstn, args=(pLvlGrid))
    hLvlNow = LinearInterp(np.insert(pLvlGrid, 0, 0.0), np.insert(hLvlGrid, 0, 0.0))

    # Make temporary grids of income shocks and next period income values
    ShkCount = TranShkValsNext.size
    pLvlCount = pLvlGrid.size
    PermShkVals_temp = np.tile(
        np.reshape(PermShkValsNext, (1, ShkCount)), (pLvlCount, 1)
    )
    TranShkVals_temp = np.tile(
        np.reshape(TranShkValsNext, (1, ShkCount)), (pLvlCount, 1)
    )
    pLvlNext_temp = (
        np.tile(
            np.reshape(pLvlNextFunc(pLvlGrid), (pLvlCount, 1)),
            (1, ShkCount),
        )
        * PermShkVals_temp
    )

    # Find the natural borrowing constraint for each persistent income level
    aLvlMin_candidates = (
        mLvlMinNext(pLvlNext_temp) - TranShkVals_temp * pLvlNext_temp
    ) / Rfree
    aLvlMinNow = np.max(aLvlMin_candidates, axis=1)
    aLvlMinNow = np.maximum(aLvlMinNow, -WealthShift)
    BoroCnstNat = LinearInterp(
        np.insert(pLvlGrid, 0, 0.0), np.insert(aLvlMinNow, 0, 0.0)
    )

    # Define the minimum allowable mLvl by pLvl as the greater of the natural and artificial borrowing constraints
    if BoroCnstArt is not None:
        BoroCnstArt = LinearInterp(np.array([0.0, 1.0]), np.array([0.0, BoroCnstArt]))
        mLvlMinNow = UpperEnvelope(BoroCnstArt, BoroCnstNat)
    else:
        mLvlMinNow = BoroCnstNat

    # Define the constrained consumption function as "consume all" shifted by mLvlMin
    cFuncNowCnstBase = BilinearInterp(
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
    )
    cFuncNowCnst = VariableLowerBoundFunc2D(cFuncNowCnstBase, mLvlMinNow)

    # Define grids of pLvl and aLvl on which to compute future expectations
    pLvlCount = pLvlGrid.size
    aNrmCount = aXtraGrid.size
    pLvlNow = np.tile(pLvlGrid, (aNrmCount, 1)).transpose()
    aLvlNow = np.tile(aXtraGrid, (pLvlCount, 1)) * pLvlNow + BoroCnstNat(pLvlNow)
    # shape = (pLvlCount,aNrmCount)
    if pLvlGrid[0] == 0.0:  # aLvl turns out badly if pLvl is 0 at bottom
        aLvlNow[0, :] = aXtraGrid

    # Calculate end-of-period marginal value of assets
    EndOfPrd_vP = (
        DiscFacEff * Rfree * expected(calc_vP_next, IncShkDstn, args=(aLvlNow, pLvlNow))
    )

    # Add in marginal utility of assets through the capitalist spirit function
    dvda = EndOfPrd_vP + WealthFac * (aLvlNow + WealthShift) ** (-CRRAwealth)

    # If the value function has been requested, construct the end-of-period vFunc
    if vFuncBool:
        # Compute expected value from end-of-period states
        EndOfPrd_v = expected(calc_v_next, IncShkDstn, args=(aLvlNow, pLvlNow))
        EndOfPrd_v *= DiscFacEff

        # Transformed value through inverse utility function to "decurve" it
        EndOfPrd_vNvrs = uFuncCon.inv(EndOfPrd_v)
        EndOfPrd_vNvrsP = EndOfPrd_vP * uFuncCon.derinv(EndOfPrd_v, order=(0, 1))

        # Add points at mLvl=zero
        EndOfPrd_vNvrs = np.concatenate(
            (np.zeros((pLvlCount, 1)), EndOfPrd_vNvrs), axis=1
        )
        EndOfPrd_vNvrsP = np.concatenate(
            (
                np.reshape(EndOfPrd_vNvrsP[:, 0], (pLvlCount, 1)),
                EndOfPrd_vNvrsP,
            ),
            axis=1,
        )
        # This is a very good approximation, vNvrsPP = 0 at the asset minimum

        # Make a temporary aLvl grid for interpolating the end-of-period value function
        aLvl_temp = np.concatenate(
            (
                np.reshape(BoroCnstNat(pLvlGrid), (pLvlGrid.size, 1)),
                aLvlNow,
            ),
            axis=1,
        )

        # Make an end-of-period value function for each persistent income level in the grid
        EndOfPrd_vNvrsFunc_list = []
        for p in range(pLvlCount):
            EndOfPrd_vNvrsFunc_list.append(
                CubicInterp(
                    aLvl_temp[p, :] - BoroCnstNat(pLvlGrid[p]),
                    EndOfPrd_vNvrs[p, :],
                    EndOfPrd_vNvrsP[p, :],
                )
            )
        EndOfPrd_vNvrsFuncBase = LinearInterpOnInterp1D(
            EndOfPrd_vNvrsFunc_list, pLvlGrid
        )

        # Re-adjust the combined end-of-period value function to account for the
        # natural borrowing constraint shifter and "re-curve" it
        EndOfPrd_vNvrsFunc = VariableLowerBoundFunc2D(
            EndOfPrd_vNvrsFuncBase, BoroCnstNat
        )
        EndOfPrd_vFunc = ValueFuncCRRA(EndOfPrd_vNvrsFunc, CRRA)
        if isinstance(vFuncNext, ConstantFunction):
            EndOfPrd_vFunc = ConstantFunction(vFuncNext.value)

    # Solve the first order condition to get optimal consumption, then find the
    # endogenous gridpoints
    cLvlNow = uFuncCon.derinv(dvda, order=(1, 0))
    mLvlNow = cLvlNow + aLvlNow

    # Limiting consumption is zero as m approaches mNrmMin
    c_for_interpolation = np.concatenate((np.zeros((pLvlCount, 1)), cLvlNow), axis=-1)
    m_for_interpolation = np.concatenate(
        (
            BoroCnstNat(np.reshape(pLvlGrid, (pLvlCount, 1))),
            mLvlNow,
        ),
        axis=-1,
    )

    # Make an array of corresponding pLvl values, adding an additional column for
    # the mLvl points at the lower boundary
    p_for_interpolation = np.concatenate(
        (np.reshape(pLvlGrid, (pLvlCount, 1)), pLvlNow), axis=-1
    )

    # Build the set of cFuncs by pLvl, gathered in a list
    cFunc_by_pLvl_list = []  # list of consumption functions for each pLvl

    # Loop over pLvl values and make an mLvl for each one
    for j in range(p_for_interpolation.shape[0]):
        pLvl_j = p_for_interpolation[j, 0]
        m_temp = m_for_interpolation[j, :] - BoroCnstNat(pLvl_j)

        # Make a linear consumption function for this pLvl
        c_temp = c_for_interpolation[j, :]
        if pLvl_j > 0:
            cFunc_by_pLvl_list.append(
                LinearInterp(
                    m_temp,
                    c_temp,
                    lower_extrap=True,
                )
            )
        else:
            cFunc_by_pLvl_list.append(LinearInterp(m_temp, c_temp, lower_extrap=True))

    # Combine all linear cFuncs into one function
    pLvl_list = p_for_interpolation[:, 0]
    cFuncUncBase = LinearInterpOnInterp1D(cFunc_by_pLvl_list, pLvl_list)
    cFuncNowUnc = VariableLowerBoundFunc2D(cFuncUncBase, BoroCnstNat)
    # Re-adjust for lower bound of natural borrowing constraint

    # Combine the constrained and unconstrained functions into the true consumption function
    cFuncNow = LowerEnvelope2D(cFuncNowUnc, cFuncNowCnst)

    # Make the marginal value function
    vPfuncNow = MargValueFuncCRRA(cFuncNow, CRRA)

    # If the value function has been requested, construct it now
    if vFuncBool:
        # Compute expected value and marginal value on a grid of market resources
        # Tile pLvl across m values
        pLvl_temp = np.tile(pLvlGrid, (aNrmCount, 1))
        mLvl_temp = (
            np.tile(mLvlMinNow(pLvlGrid), (aNrmCount, 1))
            + np.tile(np.reshape(aXtraGrid, (aNrmCount, 1)), (1, pLvlCount)) * pLvl_temp
        )
        cLvl_temp = cFuncNow(mLvl_temp, pLvl_temp)
        aLvl_temp = mLvl_temp - cLvl_temp
        u_now = uFuncCon(cLvl_temp) + uFuncWealth(aLvl_temp)
        v_temp = u_now + EndOfPrd_vFunc(aLvl_temp, pLvl_temp)
        vP_temp = uFuncCon.der(cLvl_temp)

        # Calculate pseudo-inverse value and its first derivative (wrt mLvl)
        vNvrs_temp = uFuncCon.inv(v_temp)  # value transformed through inverse utility
        vNvrsP_temp = vP_temp * uFuncCon.derinv(v_temp, order=(0, 1))

        # Add data at the lower bound of m
        mLvl_temp = np.concatenate(
            (np.reshape(mLvlMinNow(pLvlGrid), (1, pLvlCount)), mLvl_temp), axis=0
        )
        vNvrs_temp = np.concatenate((np.zeros((1, pLvlCount)), vNvrs_temp), axis=0)
        vNvrsP_temp = np.concatenate(
            (np.reshape(vNvrsP_temp[0, :], (1, vNvrsP_temp.shape[1])), vNvrsP_temp),
            axis=0,
        )

        # Construct the pseudo-inverse value function
        vNvrsFunc_list = []
        for j in range(pLvlCount):
            pLvl = pLvlGrid[j]
            vNvrsFunc_list.append(
                CubicInterp(
                    mLvl_temp[:, j] - mLvlMinNow(pLvl),
                    vNvrs_temp[:, j],
                    vNvrsP_temp[:, j],
                )
            )
        # Value function "shifted"
        vNvrsFuncBase = LinearInterpOnInterp1D(vNvrsFunc_list, pLvlGrid)
        vNvrsFuncNow = VariableLowerBoundFunc2D(vNvrsFuncBase, mLvlMinNow)

        # "Re-curve" the pseudo-inverse value function into the value function
        vFuncNow = ValueFuncCRRA(vNvrsFuncNow, CRRA)

    else:
        vFuncNow = NullFunc()

    # Dummy out the marginal marginal value function
    vPPfuncNow = NullFunc()

    # Package and return the solution object
    solution_now = ConsumerSolution(
        cFunc=cFuncNow,
        vFunc=vFuncNow,
        vPfunc=vPfuncNow,
        vPPfunc=vPPfuncNow,
        mNrmMin=0.0,  # Not a normalized model, mLvlMin will be added below
        hNrm=0.0,  # Not a normalized model, hLvl will be added below
        MPCmax=0.0,  # This should be a function, need to make it
    )
    solution_now.hLvl = hLvlNow
    solution_now.mLvlMin = mLvlMinNow
    return solution_now


###############################################################################


# Make a constructor dictionary for the capitalist spirit consumer type
CapitalistSpirit_constructors_default = {
    "IncShkDstn": construct_lognormal_income_process_unemployment,
    "PermShkDstn": get_PermShkDstn_from_IncShkDstn,
    "TranShkDstn": get_TranShkDstn_from_IncShkDstn,
    "aXtraGrid": make_assets_grid,
    "pLvlPctiles": make_basic_pLvlPctiles,
    "pLvlGrid": make_pLvlGrid_by_simulation,
    "pLvlNextFunc": make_AR1_style_pLvlNextFunc,
    "solution_terminal": make_2D_CRRA_solution_empty,
    "kNrmInitDstn": make_lognormal_kNrm_init_dstn,
    "pLvlInitDstn": make_lognormal_pLvl_init_dstn,
}

# Make a dictionary with parameters for the default constructor for kNrmInitDstn
CapitalistSpirit_kNrmInitDstn_default = {
    "kLogInitMean": -12.0,  # Mean of log initial capital
    "kLogInitStd": 0.0,  # Stdev of log initial capital
    "kNrmInitCount": 15,  # Number of points in initial capital discretization
}

# Make a dictionary with parameters for the default constructor for pLvlInitDstn
CapitalistSpirit_pLvlInitDstn_default = {
    "pLogInitMean": 0.0,  # Mean of log permanent income
    "pLogInitStd": 0.4,  # Stdev of log permanent income
    "pLvlInitCount": 15,  # Number of points in initial capital discretization
}

# Default parameters to make IncShkDstn using construct_lognormal_income_process_unemployment
CapitalistSpirit_IncShkDstn_default = {
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
CapitalistSpirit_aXtraGrid_default = {
    "aXtraMin": 0.001,  # Minimum end-of-period "assets above minimum" value
    "aXtraMax": 50.0,  # Maximum end-of-period "assets above minimum" value
    "aXtraNestFac": 2,  # Exponential nesting factor for aXtraGrid
    "aXtraCount": 72,  # Number of points in the grid of "assets above minimum"
    "aXtraExtra": [0.005, 0.01],  # Additional other values to add in grid (optional)
}

# Default parameters to make pLvlGrid using make_basic_pLvlPctiles
CapitalistSpirit_pLvlPctiles_default = {
    "pLvlPctiles_count": 19,  # Number of points in the "body" of the grid
    "pLvlPctiles_bound": [0.05, 0.95],  # Percentile bounds of the "body"
    "pLvlPctiles_tail_count": 4,  # Number of points in each tail of the grid
    "pLvlPctiles_tail_order": np.e,  # Scaling factor for points in each tail
}

# Default parameters to make pLvlGrid using make_pLvlGrid_by_simulation
CapitalistSpirit_pLvlGrid_default = {
    "pLvlExtra": None,  # Additional permanent income points to automatically add to the grid, optional
}

CapitalistSpirit_pLvlNextFunc_default = {
    "PrstIncCorr": 0.98,  # Persistence factor for "permanent" shocks
    "PermGroFac": [1.00],  # Expected permanent income growth factor
}

# Make a dictionary to specify a general income process consumer type
CapitalistSpirit_solving_default = {
    # BASIC HARK PARAMETERS REQUIRED TO SOLVE THE MODEL
    "cycles": 1,  # Finite, non-cyclic model
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    "pseudo_terminal": True,  # solution_terminal is not actually part of solution
    "constructors": CapitalistSpirit_constructors_default,  # See dictionary above
    # PRIMITIVE RAW PARAMETERS REQUIRED TO SOLVE THE MODEL
    "CRRA": 2.0,  # Coefficient of relative risk aversion
    "Rfree": [1.03],  # Interest factor on retained assets
    "DiscFac": 0.96,  # Intertemporal discount factor
    "LivPrb": [0.98],  # Survival probability after each period
    "WealthFac": 1.0,  # Relative weight on utility of wealth
    "WealthShift": 0.0,  # Additive shifter for wealth in utility function
    "WealthCurve": 0.8,  # Ratio of CRRA for wealth to CRRA for consumption
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint
    "vFuncBool": False,  # Whether to calculate the value function during solution
    "CubicBool": False,  # Whether to use cubic spline interpolation when True
}

CapitalistSpirit_simulation_default = {
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
CapitalistSpirit_default = {}
CapitalistSpirit_default.update(CapitalistSpirit_kNrmInitDstn_default)
CapitalistSpirit_default.update(CapitalistSpirit_pLvlInitDstn_default)
CapitalistSpirit_default.update(CapitalistSpirit_IncShkDstn_default)
CapitalistSpirit_default.update(CapitalistSpirit_aXtraGrid_default)
CapitalistSpirit_default.update(CapitalistSpirit_pLvlNextFunc_default)
CapitalistSpirit_default.update(CapitalistSpirit_pLvlGrid_default)
CapitalistSpirit_default.update(CapitalistSpirit_pLvlPctiles_default)
CapitalistSpirit_default.update(CapitalistSpirit_solving_default)
CapitalistSpirit_default.update(CapitalistSpirit_simulation_default)
init_capitalist_spirit = CapitalistSpirit_default


class CapitalistSpiritConsumerType(GenIncProcessConsumerType):
    r"""
    Class for representing consumers who have "capitalist spirit" preferences,
    yielding CRRA utility from consumption and wealth, additively. Importantly,
    the risk aversion coefficient for wealth is *lower* than for consumption, so
    the agent's saving rate approaches 100% as they become arbitrarily rich.

    .. math::
        \begin{eqnarray*}
        V_t(M_t,P_t) &=& \max_{C_t} U(C_t, A_t) + \beta (1-\mathsf{D}_{t+1}) \mathbb{E} [V_{t+1}(M_{t+1}, P_{t+1}) ], \\
        A_t &=& M_t - C_t, \\
        A_t/P_t &\geq& \underline{a}, \\
        M_{t+1} &=& R A_t + \theta_{t+1}, \\
        p_{t+1} &=& G_{t+1}(P_t)\psi_{t+1}, \\
        (\psi_{t+1},\theta_{t+1}) &\sim& F_{t+1}, \\
        \mathbb{E} [F_{t+1}] &=& 1, \\
        U(C) &=& \frac{C^{1-\rho}}{1-\rho} + \alpha \frac{(A + \xi)^{1-\nu}}{1-\nu}, \\
        log(G_{t+1} (x)) &=&\varphi log(x) + (1-\varphi) log(\overline{P}_{t})+log(\Gamma_{t+1}) + log(\psi_{t+1}), \\
        \overline{P}_{t+1} &=& \overline{P}_{t} \Gamma_{t+1} \\
        \end{eqnarray*}

    Constructors
    ------------
    IncShkDstn: Constructor, :math:`\psi`, :math:`\theta`
        The agent's income shock distributions.
        Its default constructor is :func:`HARK.Calibration.Income.IncomeProcesses.construct_lognormal_income_process_unemployment`
    aXtraGrid: Constructor
        The agent's asset grid.
        Its default constructor is :func:`HARK.utilities.make_assets_grid`
    pLvlNextFunc: Constructor, (:math:`\Gamma`, :math:`\varphi`)
        An arbitrary function used to evolve the GenIncShockConsumerType's permanent income
        Its default constructor is :func:`HARK.Calibration.Income.IncomeProcesses.make_AR1_style_pLvlNextFunc`
    pLvlGrid: Constructor
        The agent's pLvl grid
        Its default constructor is :func:`HARK.Calibration.Income.IncomeProcesses.make_pLvlGrid_by_simulation`
    pLvlPctiles: Constructor
        The agents income level percentile grid
        Its default constructor is :func:`HARK.Calibration.Income.IncomeProcesses.make_basic_pLvlPctiles`

    Solving Parameters
    ------------------
    cycles: int
        0 specifies an infinite horizon model, 1 specifies a finite model.
    T_cycle: int
        Number of periods in the cycle for this agent type.
    CRRA: float, :math:`\rho`
        Coefficient of Relative Risk Aversion.
    WealthFac : float, :math:`\alpha`
        Weighting factor on utility of wealth.
    WealthShift : float :math:`\xi`
        Additive shifter for wealth in utility function.
    WealthCurve : float
        CRRA for wealth as a proportion of ordinary CRRA; must be in (0,1): CRRAwealth = WealthCurve *CRRA.
    Rfree: list[float], time varying, :math:`\mathsf{R}`
        Risk Free interest rate. Pass a list of floats to make Rfree time varying.
    DiscFac: float, :math:`\beta`
        Intertemporal discount factor.
    LivPrb: list[float], time varying, :math:`1-\mathsf{D}`
        Survival probability after each period.
    PermGroFac: list[float], time varying, :math:`\Gamma`
        Permanent income growth factor.
    BoroCnstArt: float, :math:`\underline{a}`
        The minimum Asset/Permanent Income ratio, None to ignore.
    vFuncBool: bool
        Whether to calculate the value function during solution.
    CubicBool: bool
        Whether to use cubic spline interpolation.

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
        For this agent, the options are 'PermShk', 'TranShk', 'aLvl', 'cLvl', 'mLvl', 'pLvl', and 'who_dies'.

        PermShk is the agent's permanent income shock

        TranShk is the agent's transitory income shock

        aLvl is the nominal asset level

        cLvl is the nominal consumption level

        mLvl is the nominal market resources

        pLvl is the permanent income level

        who_dies is the array of which agents died
    kLogInitMean: float
        Mean of Log initial Normalized Assets.
    kLogInitStd: float
        Std of Log initial Normalized Assets.
    pLogInitMean: float
        Mean of Log initial permanent income.
    pLogInitStd: float
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

        Unlike other models with this solution type, this model's variables are NOT normalized.
        The solution functions also depend on the permanent income level. For example, :math:`C=\text{cFunc}(M,P)`.
        hNrm has been replaced by hLvl which is a function of permanent income.
        MPC max has not yet been implemented for this class. It will be a function of permanent income.

        Visit :class:`HARK.ConsumptionSaving.ConsIndShockModel.ConsumerSolution` for more information about the solution.

    history: Dict[Array]
        Created by running the :func:`.simulate()` method.
        Contains the variables in track_vars. Each item in the dictionary is an array with the shape (T_sim,AgentCount).
        Visit :class:`HARK.core.AgentType.simulate` for more information.
    """

    time_inv_ = GenIncProcessConsumerType.time_inv_ + [
        "WealthCurve",
        "WealthFac",
        "WealthShift",
    ]

    default_ = {
        "params": init_capitalist_spirit,
        "solver": solve_one_period_CapitalistSpirit,
        "model": "ConsGenIncProcess.yaml",
    }

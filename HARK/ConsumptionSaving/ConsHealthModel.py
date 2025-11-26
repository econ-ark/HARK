"""
Classes to represent consumers who make decisions about health investment. The
first model here is adapted from White (2015).
"""

import numpy as np
from HARK.core import AgentType
from HARK.distributions import (
    expected,
    combine_indep_dstns,
    Uniform,
    DiscreteDistribution,
    DiscreteDistributionLabeled,
)
from HARK.Calibration.Income.IncomeProcesses import construct_lognormal_wage_dstn
from HARK.rewards import CRRAutility, CRRAutility_inv
from HARK.interpolation import Curvilinear2DInterp
from HARK.utilities import make_assets_grid
from HARK.ConsumptionSaving.ConsIndShockModel import make_lognormal_kNrm_init_dstn

###############################################################################


# Define a function that yields health produced from investment
def eval_health_prod(n, alpha, gamma):
    return (gamma / alpha) * n**alpha


# Define a function that yields health produced from investment
def eval_marg_health_prod(n, alpha, gamma):
    return gamma * n ** (alpha - 1.0)


# Define a function for computing expectations over next period's (marginal) value
# from the perspective of end-of-period states, conditional on survival
def calc_exp_next(shock, a, H, R, rho, alpha, gamma, funcs):
    m_next = R * a + shock["WageRte"] * H
    h_next = (1.0 - shock["DeprRte"]) * H
    vNvrs_next, c_next, n_next = funcs(m_next, h_next)
    dvdm_next = c_next**-rho
    dvdh_next = dvdm_next / (gamma * n_next ** (alpha - 1.0))
    v_next = CRRAutility(vNvrs_next, rho=rho)
    dvda = R * dvdm_next
    dvdH = (1.0 - shock["DeprRte"]) * (shock["WageRte"] * dvdm_next + dvdh_next)
    return v_next, dvda, dvdH


###############################################################################


def solve_one_period_ConsBasicHealth(
    solution_next,
    DiscFac,
    Rfree,
    CRRA,
    HealthProdExp,
    HealthProdFac,
    DieProbMax,
    ShockDstn,
    aLvlGrid,
    HLvlGrid,
    constrained_N,
):
    """
    Solve one period of the basic health investment / consumption-saving model
    using the endogenous grid method. Policy functions are the consumption function
    cFunc and the health investment function nFunc.

    Parameters
    ----------
    solution_next : Curvilinear2DInterp
        Solution to the succeeding period's problem, represented as a multi-function
        interpolant with entries vNvrsFunc, cFunc, and nFunc.
    DiscFac : float
        Intertemporal discount factor, representing beta.
    Rfree : float
        Risk-free rate of return on retained assets.
    CRRA : float
        Coefficient of relative risk aversion, representing rho. Assumed to be
        constant across periods. Should be strictly between 0 and 1.
    HealthProdExp : float
        Exponent in health production function; should be strictly b/w 0 and 1.
        This corresponds to alpha in White (2015).
    HealthProdFac : float
        Scaling factor in health production function; should be strictly positive.
        This corresponds to gamma in White (2015).
    DieProbMax : float
        Maximum death probability at the end of this period, if HLvl were exactly zero.
    ShockDstn : DiscreteDistribution
        Joint distribution of income and depreciation values that could realize
        at the start of the next period.
    aLvlGrid : np.array
        Grid of end-of-period assets (after all actions are accomplished).
    HLvlGrid : np.array
        Grid of end-of-period post-investment health.
    constrained_N : int
        Number of additional interpolation nodes to put in the mLvl dimension
        on the liquidity-constrained portion of the consumption function.

    Returns
    -------
    solution_now : dict
        Solution to this period's problem, including policy functions cFunc and
        nFunc, as well as (marginal) value functions vFunc, dvdmFunc, and dvdhFunc.
    """
    # Determine whether there is a liquidity-constrained portion of the policy functions
    WageRte_min = np.min(ShockDstn.atoms[0])
    constrained = WageRte_min > 0.0

    # Adjust the assets grid if liquidity constraint will bind somewhere
    aLvlGrid_temp = np.insert(aLvlGrid, 0, 0.0) if constrained else aLvlGrid

    # Make meshes of end-of-period states aLvl and HLvl
    (aLvl, HLvl) = np.meshgrid(aLvlGrid_temp, HLvlGrid, indexing="ij")

    # Calculate expected (marginal) value conditional on survival
    v_next_exp, dvdm_next_exp, dvdh_next_exp = expected(
        calc_exp_next,
        ShockDstn,
        args=(aLvl, HLvl, Rfree, CRRA, HealthProdExp, HealthProdFac, solution_next),
    )

    # Calculate (marginal) survival probabilities
    LivPrb = 1.0 - DieProbMax / (1.0 + HLvl)
    MargLivPrb = DieProbMax / (1.0 + HLvl) ** 2.0

    # Calculate end-of-period expectations
    EndOfPrd_v = DiscFac * (LivPrb * v_next_exp)
    EndOfPrd_dvda = DiscFac * (LivPrb * dvdm_next_exp)
    EndOfPrd_dvdH = DiscFac * (LivPrb * dvdh_next_exp + MargLivPrb * v_next_exp)
    vP_ratio = EndOfPrd_dvda / EndOfPrd_dvdH

    # Invert the first order conditions to find optimal controls when unconstrained
    cLvl = EndOfPrd_dvda ** (-1.0 / CRRA)
    nLvl = (vP_ratio / HealthProdFac) ** (1.0 / (HealthProdExp - 1.0))

    # If there is a liquidity constrained portion, find additional controls on it
    if constrained:
        # Make the grid of constrained health investment by scaling cusp values
        N = constrained_N  # to shorten next line
        frac_grid = np.reshape(np.linspace(0.01, 0.99, num=N), (N, 1))
        nLvl_at_cusp = np.reshape(nLvl[0, :], (1, HLvlGrid.size))
        nLvl_cnst = frac_grid * nLvl_at_cusp

        # Solve intraperiod FOC to get constrained consumption
        marg_health_cnst = eval_marg_health_prod(
            nLvl_cnst, HealthProdExp, HealthProdFac
        )
        cLvl_cnst = (EndOfPrd_dvdH[0, :] * marg_health_cnst) ** (-1.0 / CRRA)

        # Define "constrained end of period states" and continuation value
        aLvl_cnst = np.zeros((N, HLvlGrid.size))
        HLvl_cnst = np.tile(np.reshape(HLvlGrid, (1, HLvlGrid.size)), (N, 1))
        EndOfPrd_v_cnst = np.tile(
            np.reshape(EndOfPrd_v[0, :], (1, HLvlGrid.size)), (N, 1)
        )

        # Combine constrained and unconstrained arrays
        aLvl = np.concatenate([aLvl_cnst, aLvl], axis=0)
        HLvl = np.concatenate([HLvl_cnst, HLvl], axis=0)
        cLvl = np.concatenate([cLvl_cnst, cLvl], axis=0)
        nLvl = np.concatenate([nLvl_cnst, nLvl], axis=0)
        EndOfPrd_v = np.concatenate([EndOfPrd_v_cnst, EndOfPrd_v], axis=0)

    # Invert intratemporal transitions to find endogenous gridpoints
    mLvl = aLvl + cLvl + nLvl
    hLvl = HLvl - eval_health_prod(nLvl, HealthProdExp, HealthProdFac)

    # Calculate (pseudo-inverse) value as of decision-time
    Value = CRRAutility(cLvl, rho=CRRA) + EndOfPrd_v
    vNvrs = CRRAutility_inv(Value, rho=CRRA)

    # Add points at the lower boundary of mLvl for each function
    Zeros = np.zeros((1, HLvlGrid.size))
    mLvl = np.concatenate((Zeros, mLvl), axis=0)
    hLvl = np.concatenate((np.reshape(hLvl[0, :], (1, HLvlGrid.size)), hLvl), axis=0)
    cLvl = np.concatenate((Zeros, cLvl), axis=0)
    nLvl = np.concatenate((Zeros, nLvl), axis=0)
    vNvrs = np.concatenate((Zeros, vNvrs), axis=0)

    # Construct solution as a multi-interpolation
    solution_now = Curvilinear2DInterp([vNvrs, cLvl, nLvl], mLvl, hLvl)
    return solution_now


###############################################################################


def make_solution_terminal_ConsBasicHealth():
    """
    Constructor for the terminal period solution for the basic health investment
    model. The trivial solution is to consume all market resources and invest
    nothing in health. Takes no parameters because CRRA is irrelevant: pseudo-inverse
    value is returned rather than value, and the former is just cLvl = mLvl.

    The solution representation for this model is a multiple output function that
    takes market resources and health capital level as inputs and returns pseudo-
    inverse value, consumption level, and health investment level in that order.
    """
    return lambda mLvl, hLvl: (mLvl, mLvl, np.zeros_like(mLvl))


def make_health_grid(hLvlMin, hLvlMax, hLvlCount):
    """
    Make a uniform grid of health capital levels.

    Parameters
    ----------
    hLvlMin : float
        Lower bound on health capital level; should almost surely be zero.
    hLvlMax : float
        Upper bound on health capital level.
    hLvlCount : int
        Number of points in uniform health capital level grid.

    Returns
    -------
    hLvlGrid : np.array
        Uniform grid of health capital levels
    """
    return np.linspace(hLvlMin, hLvlMax, hLvlCount)


def make_uniform_depreciation_dstn(
    T_cycle, DeprRteMean, DeprRteSpread, DeprRteCount, RNG
):
    """
    Constructor for DeprRteDstn that makes uniform distributions that vary by age.

    Parameters
    ----------
    T_cycle : int
        Number of periods in the agent's sequence or cycle.
    DeprRteMean : [float]
        Age-varying list (or array) of mean depreciation rates.
    DeprRteSpread : [float]
        Age-varying list (or array) of half-widths of depreciate rate distribution.
    DeprRteCount : int
        Number of equiprobable nodes in each distribution.
    RNG : np.random.RandomState
        Agent's internal random number generator.

    Returns
    -------
    DeprRteDstn : [DiscreteDistribution]
        List of age-dependent discrete approximations to the depreciate rate distribution.
    """
    if len(DeprRteMean) != T_cycle:
        raise ValueError("DeprRteMean must have length T_cycle!")
    if len(DeprRteSpread) != T_cycle:
        raise ValueError("DeprRteSpread must have length T_cycle!")

    DeprRteDstn = []
    probs = DeprRteCount**-1.0 * np.ones(DeprRteCount)
    for t in range(T_cycle):
        bot = DeprRteMean[t] - DeprRteSpread[t]
        top = DeprRteMean[t] + DeprRteSpread[t]
        vals = np.linspace(bot, top, DeprRteCount)
        DeprRteDstn.append(
            DiscreteDistribution(
                pmv=probs,
                atoms=vals,
                seed=RNG.integers(0, 2**31 - 1),
            )
        )
    return DeprRteDstn


def combine_indep_wage_and_depr_dstns(T_cycle, WageRteDstn, DeprRteDstn, RNG):
    """
    Combine univariate distributions of wage rate realizations and depreciation
    rate realizations at each age, treating them as independent.

    Parameters
    ----------
    T_cycle : int
        Number of periods in the agent's sequence of periods (cycle).
    WageRteDstn : [DiscreteDistribution]
        Age-dependent list of wage rate realizations; should have length T_cycle.
    DeprRteDstn : [DiscreteDistribution]
        Age-dependent list of health depreciation rate realizatiosn; should have
        length T_cycle.
    RNG : np.random.RandomState
        Internal random number generator for the AgentType instance.

    Returns
    -------
    ShockDstn : [DiscreteDistribution]
        Age-dependent bivariate distribution with joint realizations of income
        and health depreciation rates.
    """
    if len(WageRteDstn) != T_cycle:
        raise ValueError(
            "IncShkDstn must be a list of distributions of length T_cycle!"
        )
    if len(DeprRteDstn) != T_cycle:
        raise ValueError(
            "DeprRteDstn must be a list of distributions of length T_cycle!"
        )

    ShockDstn = []
    for t in range(T_cycle):
        temp_dstn = combine_indep_dstns(
            WageRteDstn[t], DeprRteDstn[t], seed=RNG.integers(0, 2**31 - 1)
        )
        temp_dstn_alt = DiscreteDistributionLabeled.from_unlabeled(
            dist=temp_dstn,
            name="wage and depreciation shock distribution",
            var_names=["WageRte", "DeprRte"],
        )
        ShockDstn.append(temp_dstn_alt)
    return ShockDstn


def make_logistic_polynomial_die_prob(T_cycle, DieProbMaxCoeffs):
    """
    Constructor for DieProbMax, the age-varying list of maximum death probabilities
    (if health is zero). Builds the list as the logistic function evaluated on a
    polynomial of model age, given polynomial coefficients. Logistic function is
    applied to ensure probabilities are always between zero and one.

    Parameters
    ----------
    T_cycle : int
        Number of periods in the agent's sequence of periods (cycle).
    DieProbMaxCoeffs : np.array
        List or vector of polynomial coefficients for maximum death probability.

    Returns
    -------
    DieProbMax : [float]
        Age-varying list of maximum death probabilities (if health were zero).
    """
    age_vec = np.arange(T_cycle)
    DieProbMax = (1.0 + np.exp(-np.polyval(DieProbMaxCoeffs, age_vec))) ** (-1.0)
    return DieProbMax.tolist()


def make_uniform_HLvl_init_dstn(HLvlInitMin, HLvlInitMax, HLvlInitCount, RNG):
    """
    Constructor for HLvlInitDstn that builds a uniform distribution for initial
    health capital at model birth.

    Parameters
    ----------
    HLvlInitMin : float
        Lower bound of initial health capital distribution.
    HLvlInitMax : float
        Upper bound of initial health capital distribution
    HLvlInitCount : int
        Number of discretized nodes in initial health capital distribution.
    RNG : np.random.RandomState
        Agent's internal RNG.

    Returns
    -------
    HLvlInitDstn : DiscreteDistribution
        Discretized uniform distribution of initial health capital level.
    """
    dstn = Uniform(bot=HLvlInitMin, top=HLvlInitMax, seed=RNG.integers(0, 2**31 - 1))
    HLvlInitDstn = dstn.discretize(HLvlInitCount, endpoints=True)
    return HLvlInitDstn


###############################################################################

# Make a dictionary of default constructor functions
basic_health_constructors = {
    "WageRteDstn": construct_lognormal_wage_dstn,
    "DeprRteDstn": make_uniform_depreciation_dstn,
    "ShockDstn": combine_indep_wage_and_depr_dstns,
    "aLvlGrid": make_assets_grid,
    "HLvlGrid": make_health_grid,
    "DieProbMax": make_logistic_polynomial_die_prob,
    "HLvlInitDstn": make_uniform_HLvl_init_dstn,
    "kLvlInitDstn": make_lognormal_kNrm_init_dstn,
    "solution_terminal": make_solution_terminal_ConsBasicHealth,
}

# Make a dictionary of default parameters for depreciation rate distribution
default_DeprRteDstn_params = {
    "DeprRteMean": [0.05],  # Mean of uniform depreciation distribution
    "DeprRteSpread": [0.05],  # Half-width of uniform depreciation distribution
    "DeprRteCount": 7,  # Number of nodes in discrete approximation
}

# Make a dictionary of default parameters for wage rate distribution
default_WageRteDstn_params = {
    "WageRteMean": [0.1],  # Age-varying mean of wage rate
    "WageRteStd": [0.1],  # Age-varying stdev of wage rate
    "WageRteCount": 7,  # Number of nodes to use in discrete approximation
    "UnempPrb": 0.07,  # Probability of unemployment
    "IncUnemp": 0.3,  # Income when unemployed
}

# Make a dictionary of default parameters for assets grid
default_aLvlGrid_params = {
    "aXtraMin": 1e-5,  # Minimum value of end-of-period assets grid
    "aXtraMax": 100.0,  # Maximum value of end-of-period assets grid
    "aXtraCount": 44,  # Number of nodes in base assets grid
    "aXtraNestFac": 1,  # Level of exponential nesting for assets grid
    "aXtraExtra": [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],  # Extra assets nodes
}

# Make a dictionary of default parameters for health capital grid
default_hLvlGrid_params = {
    "hLvlMin": 0.0,  # Minimum value of health capital grid (leave at zero)
    "hLvlMax": 50.0,  # Maximum value of health capital grid
    "hLvlCount": 50,  # Number of nodes in health capital grid
}

# Make a dictionary of default parameters for maximum death probability
default_DieProbMax_params = {
    "DieProbMaxCoeffs": [0.0],  # Logistic-polynomial coefficients on age
}

# Make a dictionary with parameters for the default constructor for kNrmInitDstn
default_kLvlInitDstn_params = {
    "kLogInitMean": -12.0,  # Mean of log initial capital
    "kLogInitStd": 0.0,  # Stdev of log initial capital
    "kNrmInitCount": 15,  # Number of points in initial capital discretization
}

# Make a dictionary with parameters for the default constructor for HLvlInitDstn
default_HLvlInitDstn_params = {
    "HLvlInitMin": 1.0,  # Lower bound of initial health capital
    "HLvlInitMax": 2.0,  # Upper bound of initial health capital
    "HLvlInitCount": 15,  # Number of points in initial health capital discretization
}

# Make a dictionary of default parameters for the health investment model
basic_health_simple_params = {
    "constructors": basic_health_constructors,
    "DiscFac": 0.95,  # Intertemporal discount factor
    "Rfree": [1.03],  # Risk-free asset return factor
    "CRRA": 0.5,  # Coefficient of relative risk aversion
    "HealthProdExp": 0.35,  # Exponent on health production function
    "HealthProdFac": 1.0,  # Factor on health production function
    "constrained_N": 7,  # Number of points on liquidity constrained portion
    "T_cycle": 1,  # Number of periods in default cycle
    "cycles": 1,  # Number of cycles
    "T_age": None,  # Maximum lifetime length override
    "AgentCount": 10000,  # Number of agents to simulate
}

# Assemble the default parameters dictionary
init_basic_health = {}
init_basic_health.update(basic_health_simple_params)
init_basic_health.update(default_DeprRteDstn_params)
init_basic_health.update(default_WageRteDstn_params)
init_basic_health.update(default_aLvlGrid_params)
init_basic_health.update(default_hLvlGrid_params)
init_basic_health.update(default_DieProbMax_params)
init_basic_health.update(default_kLvlInitDstn_params)
init_basic_health.update(default_HLvlInitDstn_params)


class BasicHealthConsumerType(AgentType):
    r"""
    A class to represent consumers who can save in a risk-free asset and invest
    in health capital via a health production function. The model is a slight
    alteration of the one from White (2015), which was in turn lifted from Ludwig
    and Schoen. In this variation, survival probability depends on post-investment
    health capital, rather than next period's health capital realization.

    Each period, the agent chooses consumption $c_t$ and health investment $n_t$.
    Consumption yields utility via CRRA function, while investment yields additional
    health capital via production function $f(n_t)$. The agent faces a mortality
    risk that depends on their post-investment health $H_t = h_t + g(n_t)$, as
    well as income risk through wage rate $\omega_t$ and health capital depreciation
    rate $\delta_t$. Health capital also serves as human capital in the sense that
    the agent earns more income when $h_t$ is higher.

    Unlike most other HARK models, this one is *not* normalized with respect to
    permanent income-- indeed, there is no "permanent income" in this simple model.
    As parametric restrictions, the solver requires that $\rho < 1$ so that utility
    is positive everywhere. This restriction ensures that the first order conditions
    are necessary and sufficient to characterize the solution when not liquidity-
    constrained. The liquidity constrained portion of the policy function is handled.

    .. math::
        \newcommand{\CRRA}{\rho}
        \newcommand{\DiePrb}{\mathsf{D}}
        \newcommand{\PermGroFac}{\Gamma}
        \newcommand{\Rfree}{\mathsf{R}}
        \newcommand{\DiscFac}{\beta}
        \newcommand{\DeprRte}{\delta}
        \newcommand{\WageRte}{\omega}
        \begin{align*}
        v_t(m_t, h_t) &= \max_{c_t, n_t}u(c_t) + \DiscFac (1 - \DiePrb_{t}) v_{t+1}(m_{t+1}, h_{t+1}), \\
        & \text{s.t.}  \\
        H_t &= h_t + g(n_t), \\
        a_t &= m_t - c_t - n_t, \\
        \DiePrb_t = \phi_t / (1 + H_t), \\
        h_{t+1} &= (1-\DeprRte_{t+1}) H_t, \\
        y_{t+1} &= \omega_{t+1} h_{t+1}, \\
        m_{t+1} &= \Rfree_{t+1} a_t + y_{t+1}, \\
        u(c) &= \frac{c^{1-\CRRA}}{1-\CRRA}, \\
        g(n) = (\gamma / \alpha) n^{\alpha}, \\
        (\WageRte_{t+1}, \DeprRte_{t+1}) \sim F_{t+1}.
        \end{align*}

    Solving Parameters
    ------------------
    cycles: int
        0 specifies an infinite horizon model, 1 specifies a finite model.
    T_cycle: int
        Number of periods in the cycle for this agent type.
    CRRA: float, :math:`\rho`
        Coefficient of Relative Risk Aversion.
    Rfree: list[float], time varying, :math:`\mathsf{R}`
        Risk-free interest rate by age.
    DiscFac: float, :math:`\beta`
        Intertemporal discount factor.
    DieProbMax: list[float], time varying, :math:`\phi`
        Maximum death probability by age, if $H_t=0$.
    HealthProdExp : float, :math:`\alpha`
        Exponent in health production function; should be strictly b/w 0 and 1.
    HealthProdFac : float, :math:`\gamma`
        Scaling factor in health production function; should be strictly positive.
    ShockDstn : DiscreteDistribution, time varying
        Joint distribution of income and depreciation values that could realize
        at the start of the next period.
    aLvlGrid : np.array
        Grid of end-of-period assets (after all actions are accomplished).
    HLvlGrid : np.array
        Grid of end-of-period post-investment health.


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
        For this agent, the options are 'kLvl', 'yLvl', 'mLvl', 'hLvl', 'cLvl',
        'nLvl', 'WageRte', 'DeprRte',  'aLvl', 'HLvl'.

        kLvl : Beginning-of-period capital holdings, equivalent to aLvl_{t-1}

        yLvl : Labor income, the wage rate times health capital.

        mLvl : Market resources, the interest factor times capital holdings, plus labor income.

        hLvl : Health or human capital level at decision-time.

        cLvl : Consumption level.

        nLvl : Health investment level.

        WageRte : Wage rate this period.

        DeprRte : Health capital depreciation rate this period.

        aLvl : End-of-period assets: market resources less consumption and investment.

        HLvl : End-of-period health capital: health capital plus produced health.

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

    default_ = {
        "params": init_basic_health,
        "solver": solve_one_period_ConsBasicHealth,
    }
    time_vary_ = ["Rfree", "DieProbMax", "ShockDstn"]
    time_inv_ = [
        "DiscFac",
        "CRRA",
        "HealthProdExp",
        "HealthProdFac",
        "aLvlGrid",
        "HLvlGrid",
        "constrained_N",
    ]
    state_vars = ["kLvl", "yLvl", "mLvl", "hLvl", "aLvl", "HLvl"]
    shock_vars_ = ["WageRte", "DeprRte"]
    distributions = ["ShockDstn", "kLvlInitDstn", "HLvlInitDstn"]

    def sim_death(self):
        """
        Draw mortality shocks for all agents, marking some for death and replacement.

        Returns
        -------
        which_agents : np.array
            Boolean array of size AgentCount, indicating who dies now.
        """
        # Calculate agent-specific death probability
        phi = np.array(self.DieProbMax)[self.t_cycle]
        DieProb = phi / (1.0 + self.state_now["HLvl"])

        # Draw mortality shocks and mark who dies
        N = self.AgentCount
        DeathShks = self.RNG.random(N)
        which_agents = DeathShks < DieProb
        if self.T_age is not None:  # Kill agents that have lived for too many periods
            too_old = self.t_age >= self.T_age
            which_agents = np.logical_or(which_agents, too_old)
        return which_agents

    def sim_birth(self, which_agents):
        """
        Makes new consumers for the given indices.  Initialized variables include
        kLvl and HLvl, as well as time variables t_age and t_cycle.

        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.AgentCount indicating which agents should be "born".

        Returns
        -------
        None
        """
        N = np.sum(which_agents)
        kLvl_newborns = self.kLvlInitDstn.draw(N)
        HLvl_newborns = self.HLvlInitDstn.draw(N)
        self.state_now["aLvl"][which_agents] = kLvl_newborns
        self.state_now["HLvl"][which_agents] = HLvl_newborns
        self.t_age[which_agents] = 0
        self.t_cycle[which_agents] = 0

    def get_shocks(self):
        """
        Draw wage and depreciation rate shocks for all simulated agents.
        """
        WageRte_now = np.empty(self.AgentCount)
        DeprRte_now = np.empty(self.AgentCount)
        for t in range(self.T_cycle):
            these = self.t_cycle == t
            dstn = self.ShockDstn[t - 1]
            N = np.sum(these)
            Shocks = dstn.draw(N)
            WageRte_now[these] = Shocks[0, :]
            DeprRte_now[these] = Shocks[1, :]
        self.shocks["WageRte"] = WageRte_now
        self.shocks["DeprRte"] = DeprRte_now

    def transition(self):
        """
        Find current market resources and health capital from prior health capital
        and the drawn shocks.
        """
        kLvlNow = self.state_prev["aLvl"]
        HLvlPrev = self.state_prev["HLvl"]
        RfreeNow = np.array(self.Rfree)[self.t_cycle - 1]
        hLvlNow = (1.0 - self.shocks["DeprRte"]) * HLvlPrev
        yLvlNow = self.shocks["WageRte"] * hLvlNow
        mLvlNow = RfreeNow * kLvlNow + yLvlNow
        return kLvlNow, yLvlNow, mLvlNow, hLvlNow

    def get_controls(self):
        """
        Evaluate consumption and health investment functions conditional on
        current state and model age, yielding controls cLvl and nLvl.
        """
        # This intentionally has the same bug with cycles > 1 as all our other
        # models. It will be fixed all in one PR.
        mLvl = self.state_now["mLvl"]
        hLvl = self.state_now["hLvl"]
        cLvl = np.empty(self.AgentCount)
        nLvl = np.empty(self.AgentCount)
        for t in range(self.T_cycle):
            these = self.t_cycle == t
            func_t = self.solution[t]
            trash, cLvl[these], nLvl[these] = func_t(mLvl[these], hLvl[these])
        self.controls["cLvl"] = cLvl
        self.controls["nLvl"] = nLvl

    def get_poststates(self):
        """
        Calculate end-of-period retained assets and post-investment health.
        """
        self.state_now["aLvl"] = (
            self.state_now["mLvl"] - self.controls["cLvl"] - self.controls["nLvl"]
        )
        self.state_now["HLvl"] = (
            self.state_now["hLvl"]
            + (self.HealthProdFac / self.HealthProdExp)
            * self.controls["nLvl"] ** self.HealthProdExp
        )

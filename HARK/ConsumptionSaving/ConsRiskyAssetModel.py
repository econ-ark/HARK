"""
This file contains a class that adds a risky asset with a log-normal return
factor to IndShockConsumerType. It is meant as a container of methods for dealing
with risky assets that will be useful to models what will inherit from it.
"""

import numpy as np

from HARK import NullFunc
from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsumerSolution,
    IndShockConsumerType,
    make_basic_CRRA_solution_terminal,
    make_lognormal_kNrm_init_dstn,
    make_lognormal_pLvl_init_dstn,
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
from HARK.distributions import (
    Bernoulli,
    expected,
    IndexDistribution,
)
from HARK.interpolation import (
    BilinearInterp,
    ConstantFunction,
    LinearInterp,
    LinearInterpOnInterp1D,
    LowerEnvelope,
    CubicInterp,
    MargMargValueFuncCRRA,
    MargValueFuncCRRA,
    ValueFuncCRRA,
)
from HARK.rewards import UtilityFuncCRRA
from HARK.utilities import make_assets_grid

###############################################################################


def make_simple_ShareGrid(ShareCount):
    """
    Make a uniformly spaced grid on the unit interval, representing risky asset shares.

    Parameters
    ----------
    ShareCount : int
        Number of points in the grid.

    Returns
    -------
    ShareGrid : np.array
    """
    ShareGrid = np.linspace(0.0, 1.0, ShareCount)
    return ShareGrid


def select_risky_solver(PortfolioBool):
    """
    Trivial constructor function that chooses between two solvers.
    """
    if PortfolioBool:
        solve_one_period = solve_one_period_ConsPortChoice
    else:
        solve_one_period = solve_one_period_ConsIndShockRiskyAsset
    return solve_one_period


def make_AdjustDstn(AdjustPrb, T_cycle, RNG):
    """
    Make the distribution of "allowed to adjust" outcomes (a Bernoulli dstn) that
    could depend on age.

    Parameters
    ----------
    AdjustPrb : float or [float]
        Probability of being allowed to adjust portfolio allocation, by period of cycle.
    T_cycle : int
        Number of periods in the cycle.
    RNG : RandomState
        Instance's own random number generator.

    Returns
    -------
    AdjustDstn : BernoulliDistribution or IndexDistribution
        Distribution object for whether agents can update their portfolios.
    """
    if type(AdjustPrb) is list and (len(AdjustPrb) == T_cycle):
        AdjustDstn = IndexDistribution(
            Bernoulli, {"p": AdjustPrb}, seed=RNG.integers(0, 2**31 - 1)
        )
    elif type(AdjustPrb) is list:
        raise AttributeError(
            "If AdjustPrb is time-varying, it must have length of T_cycle!"
        )
    else:
        AdjustDstn = Bernoulli(p=AdjustPrb, seed=RNG.integers(0, 2**31 - 1))
    return AdjustDstn


###############################################################################

# Make a dictionary of constructors for the risky asset model
IndShockRiskyAssetConsumerType_constructor_default = {
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
    "solution_terminal": make_basic_CRRA_solution_terminal,
    "solve_one_period": select_risky_solver,
}

# Make a dictionary with parameters for the default constructor for kNrmInitDstn
IndShockRiskyAssetConsumerType_kNrmInitDstn_default = {
    "kLogInitMean": -12.0,  # Mean of log initial capital
    "kLogInitStd": 0.0,  # Stdev of log initial capital
    "kNrmInitCount": 15,  # Number of points in initial capital discretization
}

# Make a dictionary with parameters for the default constructor for pLvlInitDstn
IndShockRiskyAssetConsumerType_pLvlInitDstn_default = {
    "pLogInitMean": 0.0,  # Mean of log permanent income
    "pLogInitStd": 0.0,  # Stdev of log permanent income
    "pLvlInitCount": 15,  # Number of points in initial capital discretization
}

# Default parameters to make IncShkDstn using construct_lognormal_income_process_unemployment
IndShockRiskyAssetConsumerType_IncShkDstn_default = {
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
IndShockRiskyAssetConsumerType_aXtraGrid_default = {
    "aXtraMin": 0.001,  # Minimum end-of-period "assets above minimum" value
    "aXtraMax": 20,  # Maximum end-of-period "assets above minimum" value
    "aXtraNestFac": 3,  # Exponential nesting factor for aXtraGrid
    "aXtraCount": 48,  # Number of points in the grid of "assets above minimum"
    "aXtraExtra": None,  # Additional other values to add in grid (optional)
}

# Default parameters to make RiskyDstn with make_lognormal_RiskyDstn
IndShockRiskyAssetConsumerType_RiskyDstn_default = {
    "RiskyAvg": 1.0803701891,  # Mean return factor of risky asset
    "RiskyStd": 0.162927447983,  # Stdev of log returns on risky asset
    "RiskyCount": 5,  # Number of integration nodes to use in approximation of risky returns
}
# Risky return factor moments are based on SP500 real returns from Shiller's
# "chapter 26" data, which can be found at https://www.econ.yale.edu/~shiller/data.htm
# Access it through the internet archive
# We've (will) rounded them to the nearest .01

# Default parameters to make RiskyDstn with make_simple_ShareGrid
IndShockRiskyAssetConsumerType_ShareGrid_default = {
    "ShareCount": 25,  # Number of discrete points in the risky share approximation
}

# Make a dictionary to specify a risky asset consumer type
IndShockRiskyAssetConsumerType_solving_default = {
    # BASIC HARK PARAMETERS REQUIRED TO SOLVE THE MODEL
    "cycles": 1,  # Finite, non-cyclic model
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    "constructors": IndShockRiskyAssetConsumerType_constructor_default,  # See dictionary above
    # PRIMITIVE RAW PARAMETERS REQUIRED TO SOLVE THE MODEL
    "CRRA": 2.0,  # Coefficient of relative risk aversion
    "Rfree": [1.03],  # Return factor on risk free asset (not used by this type)
    "DiscFac": 0.96,  # Intertemporal discount factor
    "LivPrb": [0.98],  # Survival probability after each period
    "PermGroFac": [1.01],  # Permanent income growth factor
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint
    "vFuncBool": False,  # Whether to calculate the value function during solution
    "CubicBool": False,  # Whether to use cubic spline interpolation when True
    # (Uses linear spline interpolation for cFunc when False)
    "RiskyShareFixed": 1.0,  # Fixed share of risky asset when PortfolioBool is False
    "AdjustPrb": 1.0,  # Probability that the agent can update their risky portfolio share each period
    "IndepDstnBool": True,  # Whether return and income shocks are independent
    "PortfolioBool": False,  # Whether this instance can choose portfolio shares
    "PortfolioBisect": False,  # What does this do?
    "pseudo_terminal": False,
}
IndShockRiskyAssetConsumerType_simulation_default = {
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
    "sim_common_Rrisky": True,  # Whether risky returns have a shared/common value across agents
}
IndShockRiskyAssetConsumerType_default = {}
IndShockRiskyAssetConsumerType_default.update(
    IndShockRiskyAssetConsumerType_IncShkDstn_default
)
IndShockRiskyAssetConsumerType_default.update(
    IndShockRiskyAssetConsumerType_RiskyDstn_default
)
IndShockRiskyAssetConsumerType_default.update(
    IndShockRiskyAssetConsumerType_aXtraGrid_default
)
IndShockRiskyAssetConsumerType_default.update(
    IndShockRiskyAssetConsumerType_ShareGrid_default
)
IndShockRiskyAssetConsumerType_default.update(
    IndShockRiskyAssetConsumerType_solving_default
)
IndShockRiskyAssetConsumerType_default.update(
    IndShockRiskyAssetConsumerType_simulation_default
)
IndShockRiskyAssetConsumerType_default.update(
    IndShockRiskyAssetConsumerType_kNrmInitDstn_default
)
IndShockRiskyAssetConsumerType_default.update(
    IndShockRiskyAssetConsumerType_pLvlInitDstn_default
)
init_risky_asset = IndShockRiskyAssetConsumerType_default


class IndShockRiskyAssetConsumerType(IndShockConsumerType):
    r"""
    A consumer type based on IndShockConsumerType, that has access to a risky asset for their savings. The
    risky asset has lognormal returns that are possibly correlated with his
    income shocks.

    If PortfolioBool is False, then the risky asset share is always one.
    Otherwise the agent can optimize their risky asset share.

    .. math::
        \newcommand{\CRRA}{\rho}
        \newcommand{\DiePrb}{\mathsf{D}}
        \newcommand{\PermGroFac}{\Gamma}
        \newcommand{\Rfree}{\mathsf{R}}
        \newcommand{\DiscFac}{\beta}
        \begin{align*}
        v_t(m_t) &= \max_{c_t,S_t} u(c_t) + \DiscFac (1-\DiePrb_{t+1})  \mathbb{E}_{t} \left[(\PermGroFac_{t+1}\psi_{t+1})^{1-\CRRA} v_{t+1}(m_{t+1}) \right], \\
        & \text{s.t.}  \\
        a_t &= m_t - c_t, \\
        a_t &\geq \underline{a}, \\
        m_{t+1} &= \mathsf{R}_{t+1}/(\PermGroFac_{t+1} \psi_{t+1}) a_t + \theta_{t+1}, \\
        \mathsf{R}_{t+1} &=S_t\phi_{t+1}\mathbf{R}_{t+1}+ (1-S_t)\mathsf{R}_{t+1}, \\
        (\psi_{t+1},\theta_{t+1},\phi_{t+1}) &\sim F_{t+1}, \\
        \mathbb{E}[\psi]=\mathbb{E}[\theta] &= 1. \\
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
    PortfolioBool: Boolean
        Determines whether agent will use portfolio optimization or they only have access to risky assets. If false, the risky share is always one.

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
        Infinite horizon solutions return a list with T_cycle elements for each period in the cycle. If PortfolioBool is True, the solution also contains ShareFunc.

        If PortfolioBool is True, the solution also contains:
        ShareFunc - The asset share function for this period, defined over normalized market resources :math:`S=ShareFunc(mNrm)`.

        Visit :class:`HARK.ConsumptionSaving.ConsIndShockModel.ConsumerSolution` for more information about the solution.
    history: Dict[Array]
        Created by running the :func:`.simulate()` method.
        Contains the variables in track_vars. Each item in the dictionary is an array with the shape (T_sim,AgentCount).
        Visit :class:`HARK.core.AgentType.simulate` for more information.
    """

    IncShkDstn_default = IndShockRiskyAssetConsumerType_IncShkDstn_default
    RiskyDstn_default = IndShockRiskyAssetConsumerType_RiskyDstn_default
    aXtraGrid_default = IndShockRiskyAssetConsumerType_aXtraGrid_default
    ShareGrid_default = IndShockRiskyAssetConsumerType_ShareGrid_default
    solving_default = IndShockRiskyAssetConsumerType_solving_default
    simulation_default = IndShockRiskyAssetConsumerType_simulation_default  # So sphinx documents defaults
    default_ = {
        "params": IndShockRiskyAssetConsumerType_default,
        "solver": NullFunc(),
        "model": "ConsRiskyAsset.yaml",
    }

    time_inv_ = IndShockConsumerType.time_inv_ + [
        "PortfolioBisect",
        "ShareGrid",
        "PortfolioBool",
        "IndepDstnBool",
        "RiskyShareFixed",
    ]
    time_vary_ = IndShockConsumerType.time_vary_ + ["ShockDstn", "ShareLimit"]
    shock_vars_ = IndShockConsumerType.shock_vars_ + ["Adjust", "Risky"]
    distributions = [
        "IncShkDstn",
        "PermShkDstn",
        "TranShkDstn",
        "RiskyDstn",
        "ShockDstn",
        "kNrmInitDstn",
        "pLvlInitDstn",
        "RiskyDstn",
    ]

    def pre_solve(self):
        self.construct("solution_terminal")
        self.update_timing()
        self.solution_terminal.ShareFunc = ConstantFunction(1.0)

    def update_timing(self):
        """
        This method simply ensures that a few attributes that could be in either
        time_inv or time_vary are appropriately labeled.
        """
        if type(self.AdjustDstn) is IndexDistribution:
            self.add_to_time_vary("AdjustPrb")
            self.del_from_time_inv("AdjustPrb")
        else:
            self.add_to_time_inv("AdjustPrb")
            self.del_from_time_vary("AdjustPrb")
        if hasattr(self.RiskyDstn, "__getitem__"):
            self.add_to_time_vary("RiskyDstn")
        else:
            self.add_to_time_inv("RiskyDstn")
        if type(self.ShareLimit) is list:
            self.add_to_time_vary("ShareLimit")
            self.del_from_time_inv("ShareLimit")
        else:
            self.add_to_time_inv("ShareLimit")
            self.del_from_time_vary("ShareLimit")

    def get_Rport(self):
        """
        Calculates realized return factor for each agent, using the attributes Rfree,
        RiskyNow, and ShareNow.

        Parameters
        ----------
        None

        Returns
        -------
        Rport : np.array
            Array of size AgentCount with each simulated agent's realized portfolio
            return factor.  Will be used by get_states() to calculate mNrmNow, where it
            will be mislabeled as "Rfree".
        """

        RfreeNow = super().get_Rport()
        RiskyNow = self.shocks["Risky"]
        if self.PortfolioBool:
            ShareNow = self.controls["Share"]
        else:
            ShareNow = np.ones_like(RiskyNow)  # Only asset is risky asset

        Rport = ShareNow * RiskyNow + (1.0 - ShareNow) * RfreeNow
        self.Rport = Rport
        return Rport

    def get_Risky(self):
        """
        Draws a new risky return factor.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # How we draw the shocks depends on whether their distribution is time-varying
        if "RiskyDstn" in self.time_vary:
            if self.sim_common_Rrisky:
                raise AttributeError(
                    "If sim_common_Rrisky is True, RiskyDstn cannot be time-varying!"
                )

            else:
                # Make use of the IndexDistribution.draw() method
                self.shocks["Risky"] = self.RiskyDstn.draw(
                    np.maximum(self.t_cycle - 1, 0)
                    if self.cycles == 1
                    else self.t_cycle
                )

        else:
            # Draw either a common economy-wide return, or one for each agent
            if self.sim_common_Rrisky:
                self.shocks["Risky"] = self.RiskyDstn.draw(1)
            else:
                self.shocks["Risky"] = self.RiskyDstn.draw(self.AgentCount)

    def get_Adjust(self):
        """
        Sets the attribute Adjust as a boolean array of size AgentCount, indicating
        whether each agent is able to adjust their risky portfolio share this period.
        Uses the attribute AdjustPrb to draw from a Bernoulli distribution.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if "AdjustPrb" in self.time_vary:
            self.shocks["Adjust"] = self.AdjustDstn.draw(
                np.maximum(self.t_cycle - 1, 0) if self.cycles == 1 else self.t_cycle
            )
        else:
            self.shocks["Adjust"] = self.AdjustDstn.draw(self.AgentCount)

    def initialize_sim(self):
        """
        Initialize the state of simulation attributes.  Simply calls the same
        method for IndShockConsumerType, then initializes the new states/shocks
        Adjust and Share.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.shocks["Adjust"] = np.zeros(self.AgentCount, dtype=bool)
        # Initialize Share to default value; will be updated in get_controls()
        self.controls["Share"] = np.ones(self.AgentCount)
        IndShockConsumerType.initialize_sim(self)

    def get_shocks(self):
        """
        Draw idiosyncratic income shocks, just as for IndShockConsumerType, then draw
        a single common value for the risky asset return.  Also draws whether each
        agent is able to adjust their portfolio this period.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        IndShockConsumerType.get_shocks(self)
        self.get_Risky()
        self.get_Adjust()

    def get_controls(self):
        """
        Calculates consumption for each consumer of this type using the consumption functions;
        also calculates risky asset share when PortfolioBool=True

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        cNrmNow = np.full(self.AgentCount, np.nan)
        MPCnow = np.full(self.AgentCount, np.nan)
        ShareNow = np.full(self.AgentCount, np.nan)
        for t in np.unique(self.t_cycle):
            idx = self.t_cycle == t
            if np.any(idx):
                mNrm = self.state_now["mNrm"][idx]
                cNrmNow[idx], MPCnow[idx] = self.solution[t].cFunc.eval_with_derivative(
                    mNrm
                )
                if self.PortfolioBool:
                    ShareNow[idx] = self.solution[t].ShareFunc(mNrm)
                else:
                    ShareNow[idx] = self.RiskyShareFixed
        self.controls["cNrm"] = cNrmNow
        self.controls["Share"] = ShareNow
        self.MPCnow = MPCnow

    def check_conditions(self, verbose=None):
        raise NotImplementedError()  # pragma: nocover

    def calc_limiting_values(self):
        raise NotImplementedError()  # pragma: nocover


# This is to preserve compatibility with old name
RiskyAssetConsumerType = IndShockRiskyAssetConsumerType


###############################################################################


def solve_one_period_ConsPortChoice(
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
    ShareLimit,
    vFuncBool,
    IndepDstnBool,
):
    """
    Solve one period of a consumption-saving problem with portfolio allocation
    between a riskless and risky asset. This function handles only the most
    fundamental portfolio choice problem: frictionless reallocation of the
    portfolio each period as a continuous choice.

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
    ShareLimit : float
        Limiting lower bound of risky portfolio share as mNrm approaches infinity.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.
    IndepDstnBool : bool
        Indicator for whether the income and risky return distributions are in-
        dependent of each other, which can speed up the expectations step.

    Returns
    -------
    solution_now : PortfolioSolution
        Solution to this period's problem.

    :meta private:
    """
    # Make sure the individual is liquidity constrained.  Allowing a consumer to
    # borrow *and* invest in an asset with unbounded (negative) returns is a bad mix.
    if BoroCnstArt != 0.0:
        raise ValueError("PortfolioConsumerType must have BoroCnstArt=0.0!")

    # Define the current period utility function and effective discount factor
    uFunc = UtilityFuncCRRA(CRRA)
    DiscFacEff = DiscFac * LivPrb  # "effective" discount factor

    # Unpack next period's solution for easier access
    vPfunc_next = solution_next.vPfunc
    vFunc_next = solution_next.vFunc

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
    def calc_Radj(R):
        Rport = ShareLimit * R + (1.0 - ShareLimit) * Rfree
        return Rport ** (1.0 - CRRA)

    R_adj = expected(calc_Radj, RiskyDstn)[0]
    PatFac = (DiscFacEff * R_adj) ** (1.0 / CRRA)
    MPCminNow = 1.0 / (1.0 + PatFac / solution_next.MPCmin)

    # Also perform an alternate calculation for human wealth under risky returns
    def calc_hNrm(S):
        Risky = S["Risky"]
        PermShk = S["PermShk"]
        TranShk = S["TranShk"]
        G = PermGroFac * PermShk
        Rport = ShareLimit * Risky + (1.0 - ShareLimit) * Rfree
        hNrm = (G / Rport**CRRA) * (TranShk + solution_next.hNrm)
        return hNrm

    # This correctly accounts for risky returns and risk aversion
    hNrmNow = expected(calc_hNrm, ShockDstn) / R_adj

    # This basic equation works if there's no correlation among shocks
    # hNrmNow = (PermGroFac/Rfree)*(1 + solution_next.hNrm)

    # Define the terms for the limiting linear consumption function as m gets very big
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
        bNrmNext = bNrmGrid

        # Define functions that are used internally to evaluate future realizations
        def calc_mNrm_next(S, b):
            """
            Calculate future realizations of market resources mNrm from the income
            shock distribution S and normalized bank balances b.
            """
            return b / (S["PermShk"] * PermGroFac) + S["TranShk"]

        def calc_dvdm_next(S, b):
            """
            Evaluate realizations of marginal value of market resources next period,
            based on the income distribution S and values of bank balances bNrm
            """
            mNrm_next = calc_mNrm_next(S, b)
            G = S["PermShk"] * PermGroFac
            dvdm_next = G ** (-CRRA) * vPfunc_next(mNrm_next)
            return dvdm_next

        # Calculate end-of-period marginal value of assets and shares at each point
        # in aNrm and ShareGrid. Does so by taking expectation of next period marginal
        # values across income and risky return shocks.

        # Calculate intermediate marginal value of bank balances by taking expectations over income shocks
        dvdb_intermed = expected(calc_dvdm_next, IncShkDstn, args=(bNrmNext))
        dvdbNvrs_intermed = uFunc.derinv(dvdb_intermed, order=(1, 0))

        dvdbNvrsFunc_intermed = LinearInterp(bNrmGrid, dvdbNvrs_intermed)
        dvdbFunc_intermed = MargValueFuncCRRA(dvdbNvrsFunc_intermed, CRRA)

        # The intermediate marginal value of risky portfolio share is zero in this
        # model because risky share is flexible: we can just change it next period,
        # so there is no marginal value of Share once the return is realized.
        dvdsFunc_intermed = ConstantFunction(0.0)  # all zeros

        # Make tiled arrays to calculate future realizations of bNrm and Share when integrating over RiskyDstn
        aNrmNow, ShareNext = np.meshgrid(aNrmGrid, ShareGrid, indexing="ij")

        # Define functions for calculating end-of-period marginal value
        def calc_EndOfPrd_dvda(R, a, z):
            """
            Compute end-of-period marginal value of assets at values a, conditional
            on risky asset return R and risky share z.
            """
            # Calculate future realizations of bank balances bNrm
            Rxs = R - Rfree  # Excess returns
            Rport = Rfree + z * Rxs  # Portfolio return
            bNrm_next = Rport * a

            # Calculate and return dvda
            EndOfPrd_dvda = Rport * dvdbFunc_intermed(bNrm_next)
            return EndOfPrd_dvda

        def calc_EndOfPrd_dvds(R, a, z):
            """
            Compute end-of-period marginal value of risky share at values a, conditional
            on risky asset return S and risky share z.
            """
            # Calculate future realizations of bank balances bNrm
            Rxs = R - Rfree  # Excess returns
            Rport = Rfree + z * Rxs  # Portfolio return
            bNrm_next = Rport * a

            # Calculate and return dvds (second term is all zeros)
            EndOfPrd_dvds = Rxs * a * dvdbFunc_intermed(bNrm_next) + dvdsFunc_intermed(
                bNrm_next
            )
            return EndOfPrd_dvds

        TempDstn = RiskyDstn  # relabeling for below

        # Evaluate realizations of value and marginal value after asset returns are realized

        # Calculate end-of-period marginal value of risky portfolio share by taking expectations
        EndOfPrd_dvds = DiscFacEff * expected(
            calc_EndOfPrd_dvds, RiskyDstn, args=(aNrmNow, ShareNext)
        )

        # Make the end-of-period value function if the value function is requested
        if vFuncBool:

            def calc_v_intermed(S, b):
                """
                Calculate "intermediate" value from next period's bank balances, the
                income shocks S, and the risky asset share.
                """
                mNrm_next = calc_mNrm_next(S, b)
                v_next = vFunc_next(mNrm_next)
                v_intermed = (S["PermShk"] * PermGroFac) ** (1.0 - CRRA) * v_next
                return v_intermed

            # Calculate intermediate value by taking expectations over income shocks
            v_intermed = expected(calc_v_intermed, IncShkDstn, args=(bNrmNext))

            # Construct the "intermediate value function" for this period
            vNvrs_intermed = uFunc.inv(v_intermed)
            vNvrsFunc_intermed = LinearInterp(bNrmGrid, vNvrs_intermed)
            vFunc_intermed = ValueFuncCRRA(vNvrsFunc_intermed, CRRA)

            def calc_EndOfPrd_v(S, a, z):
                # Calculate future realizations of bank balances bNrm
                Rxs = S - Rfree
                Rport = Rfree + z * Rxs
                bNrm_next = Rport * a

                EndOfPrd_v = vFunc_intermed(bNrm_next)
                return EndOfPrd_v

            # Calculate end-of-period value by taking expectations
            EndOfPrd_v = DiscFacEff * expected(
                calc_EndOfPrd_v, RiskyDstn, args=(aNrmNow, ShareNext)
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
        def calc_mNrm_next(S, a, z):
            """
            Calculate future realizations of market resources mNrm from the shock
            distribution S, normalized end-of-period assets a, and risky share z.
            """
            # Calculate future realizations of bank balances bNrm
            Rxs = S["Risky"] - Rfree
            Rport = Rfree + z * Rxs
            bNrm_next = Rport * a
            mNrm_next = bNrm_next / (S["PermShk"] * PermGroFac) + S["TranShk"]
            return mNrm_next

        def calc_EndOfPrd_dvdx(S, a, z):
            """
            Evaluate end-of-period marginal value of assets and risky share based
            on the shock distribution S, normalized end-of-period assets a, and
            risky share z.
            """
            mNrm_next = calc_mNrm_next(S, a, z)
            Rxs = S["Risky"] - Rfree
            Rport = Rfree + z * Rxs
            dvdm_next = vPfunc_next(mNrm_next)
            # No marginal value of Share if it's a free choice!
            dvds_next = np.zeros_like(mNrm_next)

            EndOfPrd_dvda = Rport * (S["PermShk"] * PermGroFac) ** (-CRRA) * dvdm_next
            EndOfPrd_dvds = (
                Rxs * a * (S["PermShk"] * PermGroFac) ** (-CRRA) * dvdm_next
                + (S["PermShk"] * PermGroFac) ** (1 - CRRA) * dvds_next
            )

            return EndOfPrd_dvda, EndOfPrd_dvds

        def calc_EndOfPrd_v(S, a, z):
            """
            Evaluate end-of-period value, based on the shock distribution S, values
            of bank balances bNrm, and values of the risky share z.
            """
            mNrm_next = calc_mNrm_next(S, a, z)
            v_next = vFunc_next(mNrm_next)
            EndOfPrd_v = (S["PermShk"] * PermGroFac) ** (1.0 - CRRA) * v_next
            return EndOfPrd_v

        calc_EndOfPrd_dvda = lambda S, a, z: calc_EndOfPrd_dvdx(S, a, z)[0]
        calc_EndOfPrd_dvds = lambda S, a, z: calc_EndOfPrd_dvdx(S, a, z)[1]
        TempDstn = ShockDstn

        # Evaluate realizations of value and marginal value after asset returns are realized

        # Calculate end-of-period marginal value of assets and risky share by taking expectations
        EndOfPrd_dvda, EndOfPrd_dvds = DiscFacEff * expected(
            calc_EndOfPrd_dvdx, ShockDstn, args=(aNrmNow, ShareNext)
        )
        EndOfPrd_dvdaNvrs = uFunc.derinv(EndOfPrd_dvda)

        # Construct the end-of-period value function if requested
        if vFuncBool:
            # Calculate end-of-period value, its derivative, and their pseudo-inverse
            EndOfPrd_v = DiscFacEff * expected(
                calc_EndOfPrd_v, ShockDstn, args=(aNrmNow, ShareNext)
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

    # Now find the optimal (continuous) risky share on [0,1] by solving the first
    # order condition EndOfPrd_dvds == 0.
    FOC_s = EndOfPrd_dvds  # Relabel for convenient typing

    # If agent wants to put more than 100% into risky asset, he is constrained.
    # Likewise if he wants to put less than 0% into risky asset, he is constrained.
    constrained_top = FOC_s[:, -1] > 0.0
    constrained_bot = FOC_s[:, 0] < 0.0
    constrained = np.logical_or(constrained_top, constrained_bot)
    a_idx = np.arange(aNrmCount)

    # For each value of aNrm, find the value of Share such that FOC_s == 0
    crossing = np.logical_and(FOC_s[:, 1:] <= 0.0, FOC_s[:, :-1] >= 0.0)
    share_idx = np.argmax(crossing, axis=1)

    for k in range(3):
        # This represents the index of the segment of the share grid where dvds flips
        # from positive to negative, indicating that there's a zero *on* the segment.
        # The exception is for aNrm values that are flagged as constrained, for which
        # there will be no crossing point and we can just use the boundary value.

        # Now that we have a *range* for the location of the optimal share, we can
        # do a refined search for the optimal share at each aNrm value where there
        # is an interior solution (not constrained). We now make a refined ShareGrid
        # that has *different* values on it for each aNrm value.
        bot_s = ShareNext[a_idx, share_idx]
        top_s = ShareNext[a_idx, share_idx + 1]
        for j in range(aNrmCount):
            if constrained[j]:
                continue
            ShareNext[j, :] = np.linspace(bot_s[j], top_s[j], ShareCount)

        # Now evaluate end-of-period marginal value of risky share on the refined grid
        EndOfPrd_dvds = DiscFacEff * expected(
            calc_EndOfPrd_dvds, TempDstn, args=(aNrmNow, ShareNext)
        )
        these = np.logical_not(constrained)
        FOC_s[these, :] = EndOfPrd_dvds[these, :]  # Relabel for convenient typing

        # Look for "crossing points" again
        crossing = np.logical_and(FOC_s[these, 1:] <= 0.0, FOC_s[these, :-1] >= 0.0)
        share_idx[these] = np.argmax(crossing, axis=1)

    # Calculate end-of-period marginal value of assets on the refined grid
    EndOfPrd_dvda = DiscFacEff * expected(
        calc_EndOfPrd_dvda, TempDstn, args=(aNrmNow, ShareNext)
    )
    EndOfPrd_dvdaNvrs = uFunc.derinv(EndOfPrd_dvda)

    # Calculate the fractional distance between those share gridpoints where the
    # zero should be found, assuming a linear function; call it alpha
    bot_s = ShareNext[a_idx, share_idx]
    top_s = ShareNext[a_idx, share_idx + 1]
    bot_f = FOC_s[a_idx, share_idx]
    top_f = FOC_s[a_idx, share_idx + 1]
    bot_c = EndOfPrd_dvdaNvrs[a_idx, share_idx]
    top_c = EndOfPrd_dvdaNvrs[a_idx, share_idx + 1]
    alpha = 1.0 - top_f / (top_f - bot_f)

    # Calculate the continuous optimal risky share and optimal consumption
    Share_now = (1.0 - alpha) * bot_s + alpha * top_s
    cNrm_now = (1.0 - alpha) * bot_c + alpha * top_c

    # If agent wants to put more than 100% into risky asset, he is constrained.
    # Likewise if he wants to put less than 0% into risky asset, he is constrained.
    constrained_top = FOC_s[:, -1] > 0.0
    constrained_bot = FOC_s[:, 0] < 0.0

    # Apply the constraints to both risky share and consumption (but lower
    # constraint should never be relevant)
    Share_now[constrained_top] = 1.0
    Share_now[constrained_bot] = 0.0
    cNrm_now[constrained_top] = EndOfPrd_dvdaNvrs[constrained_top, -1]
    cNrm_now[constrained_bot] = EndOfPrd_dvdaNvrs[constrained_bot, 0]

    # When the natural borrowing constraint is *not* zero, then aNrm=0 is in the
    # grid, but there's no way to "optimize" the portfolio if a=0, and consumption
    # can't depend on the risky share if it doesn't meaningfully exist. Apply
    # a small fix to the bottom gridpoint (aNrm=0) when this happens.
    if not BoroCnstNat_iszero:
        Share_now[0] = 1.0
        cNrm_now[0] = EndOfPrd_dvdaNvrs[0, -1]

    # Construct functions characterizing the solution for this period

    # Calculate the endogenous mNrm gridpoints when the agent adjusts his portfolio,
    # then construct the consumption function when the agent can adjust his share
    mNrm_now = np.insert(aNrmGrid + cNrm_now, 0, 0.0)
    cNrm_now = np.insert(cNrm_now, 0, 0.0)
    cFunc_now = LinearInterp(mNrm_now, cNrm_now, cFuncLimitIntercept, cFuncLimitSlope)

    # Construct the marginal value (of mNrm) function
    vPfunc_now = MargValueFuncCRRA(cFunc_now, CRRA)

    # If the share choice is continuous, just make an ordinary interpolating function
    if BoroCnstNat_iszero:
        Share_lower_bound = ShareLimit
    else:
        Share_lower_bound = 1.0
    Share_now = np.insert(Share_now, 0, Share_lower_bound)
    ShareFunc_now = LinearInterp(mNrm_now, Share_now, ShareLimit, 0.0)

    # Add the value function if requested
    if vFuncBool:
        # Create the value functions for this period, defined over market resources
        # mNrm when agent can adjust his portfolio, and over market resources and
        # fixed share when agent can not adjust his portfolio.

        # Construct the value function
        mNrm_temp = aXtraGrid  # Just use aXtraGrid as our grid of mNrm values
        cNrm_temp = cFunc_now(mNrm_temp)
        aNrm_temp = np.maximum(mNrm_temp - cNrm_temp, 0.0)  # Fix tiny violations
        Share_temp = ShareFunc_now(mNrm_temp)
        v_temp = uFunc(cNrm_temp) + EndOfPrd_vFunc(aNrm_temp, Share_temp)
        vNvrs_temp = uFunc.inv(v_temp)
        vNvrsP_temp = uFunc.der(cNrm_temp) * uFunc.inverse(v_temp, order=(0, 1))
        vNvrsFunc = CubicInterp(
            np.insert(mNrm_temp, 0, 0.0),  # x_list
            np.insert(vNvrs_temp, 0, 0.0),  # f_list
            np.insert(vNvrsP_temp, 0, vNvrsP_temp[0]),  # dfdx_list
        )
        # Re-curve the pseudo-inverse value function
        vFunc_now = ValueFuncCRRA(vNvrsFunc, CRRA)

    else:  # If vFuncBool is False, fill in dummy values
        vFunc_now = NullFunc()

    # Package and return the solution
    solution_now = ConsumerSolution(
        cFunc=cFunc_now,
        vPfunc=vPfunc_now,
        vFunc=vFunc_now,
        hNrm=hNrmNow,
        MPCmin=MPCminNow,
    )
    solution_now.ShareFunc = ShareFunc_now
    return solution_now


###############################################################################


def solve_one_period_ConsIndShockRiskyAsset(
    solution_next,
    IncShkDstn,
    RiskyDstn,
    ShockDstn,
    LivPrb,
    DiscFac,
    Rfree,
    CRRA,
    PermGroFac,
    BoroCnstArt,
    aXtraGrid,
    RiskyShareFixed,
    vFuncBool,
    CubicBool,
    IndepDstnBool,
):
    """
    Solves one period of a consumption-saving model with idiosyncratic shocks to
    permanent and transitory income, with one risky asset and CRRA utility.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncShkDstn : Distribution
        Discrete distribution of permanent income shocks and transitory income
        shocks. This is only used if the input IndepDstnBool is True, indicating
        that income and return distributions are independent.
    RiskyDstn : Distribution
       Distribution of risky asset returns. This is only used if the input
       IndepDstnBool is True, indicating that income and return distributions
       are independent.
    ShockDstn : Distribution
        Joint distribution of permanent income shocks, transitory income shocks,
        and risky returns.  This is only used if the input IndepDstnBool is False,
        indicating that income and return distributions can't be assumed to be
        independent.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    Rfree : float
        Risk free interest factor on end-of-period assets.
    CRRA : float
        Coefficient of relative risk aversion.
    PermGroFac : float
        Expected permanent income growth factor at the end of this period.
    BoroCnstArt: float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.  If it is less than the natural borrowing constraint,
        then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
        rowing constraint.
    aXtraGrid: np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.
    RiskyShareFixed : float
        Fixed fraction of end-of-period assets that are allocated to the risky asset.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear interpolation.
    IndepDstnBool : bool
        Indicator for whether the income and risky return distributions are in-
        dependent of each other, which can speed up the expectations step.

    Returns
    -------
    solution_now : ConsumerSolution
        Solution to this period's consumption-saving problem with income risk.

    :meta private:
    """
    # Do a quick validity check; don't want to allow borrowing with risky returns
    if BoroCnstArt != 0.0:
        raise ValueError("RiskyAssetConsumerType must have BoroCnstArt=0.0!")

    # Define the current period utility function and effective discount factor
    uFunc = UtilityFuncCRRA(CRRA)
    DiscFacEff = DiscFac * LivPrb  # "effective" discount factor

    # Unpack next period's income shock distribution
    ShkPrbsNext = ShockDstn.pmv
    PermShkValsNext = ShockDstn.atoms[0]
    TranShkValsNext = ShockDstn.atoms[1]
    RiskyValsNext = ShockDstn.atoms[2]
    PermShkMinNext = np.min(PermShkValsNext)
    TranShkMinNext = np.min(TranShkValsNext)
    RiskyMinNext = np.min(RiskyValsNext)
    RiskyMaxNext = np.max(RiskyValsNext)

    # Unpack next period's (marginal) value function
    vFuncNext = solution_next.vFunc  # This is None when vFuncBool is False
    vPfuncNext = solution_next.vPfunc
    vPPfuncNext = solution_next.vPPfunc  # This is None when CubicBool is False

    # Perform an alternate calculation of the absolute patience factor when returns are risky
    def calc_Radj(R):
        Rport = RiskyShareFixed * R + (1.0 - RiskyShareFixed) * Rfree
        return Rport ** (1.0 - CRRA)

    R_adj = expected(calc_Radj, RiskyDstn)
    PatFac = (DiscFacEff * R_adj) ** (1.0 / CRRA)
    MPCminNow = 1.0 / (1.0 + PatFac / solution_next.MPCmin)
    MPCminNow = MPCminNow[0]

    # Also perform an alternate calculation for human wealth under risky returns
    def calc_hNrm(S):
        Risky = S["Risky"]
        PermShk = S["PermShk"]
        TranShk = S["TranShk"]
        G = PermGroFac * PermShk
        Rport = RiskyShareFixed * Risky + (1.0 - RiskyShareFixed) * Rfree
        hNrm = (G / Rport**CRRA) * (TranShk + solution_next.hNrm)
        return hNrm

    # This correctly accounts for risky returns and risk aversion
    hNrmNow = expected(calc_hNrm, ShockDstn) / R_adj
    hNrmNow = hNrmNow[0]

    # The above attempts to pin down the limiting consumption function for this
    # model, however it is not clear why it creates bugs, so for now we allow
    # for a linear extrapolation beyond the last asset point
    cFuncLimitIntercept = MPCminNow * hNrmNow
    cFuncLimitSlope = MPCminNow

    # Calculate the minimum allowable value of market resources in this period
    BoroCnstNat_cand = (
        (solution_next.mNrmMin - TranShkValsNext)
        * (PermGroFac * PermShkValsNext)
        / RiskyValsNext
    )
    BoroCnstNat = np.max(BoroCnstNat_cand)  # Must be at least this

    # Set a flag for whether the natural borrowing constraint is zero, which
    # depends on whether the smallest transitory income shock is zero
    BoroCnstNat_iszero = np.min(IncShkDstn.atoms[1]) == 0.0

    # Set the minimum allowable (normalized) market resources based on the natural
    # and artificial borrowing constraints
    if BoroCnstArt is None:
        mNrmMinNow = BoroCnstNat
    else:
        mNrmMinNow = np.max([BoroCnstNat, BoroCnstArt])

    # The MPCmax code is a bit unusual here, and possibly "harmlessly wrong".
    # The "worst event" should depend on the risky return factor as well as
    # income shocks. However, the natural borrowing constraint is only ever
    # relevant in this model when it's zero, so the MPC at mNrm is only relevant
    # in the case where risky returns don't matter at all (because a=0).

    # Calculate the probability that we get the worst possible income draw
    IncNext = PermShkValsNext * TranShkValsNext
    WorstIncNext = PermShkMinNext * TranShkMinNext
    WorstIncPrb = np.sum(ShkPrbsNext[IncNext == WorstIncNext])
    # WorstIncPrb is the "Weierstrass p" concept: the odds we get the WORST thing

    # Update the upper bounding MPC as market resources approach the lower bound
    temp_fac = (WorstIncPrb ** (1.0 / CRRA)) * PatFac
    MPCmaxNow = 1.0 / (1.0 + temp_fac / solution_next.MPCmax)

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

    # Big methodological split here: whether the income and return distributions are independent.
    # Calculation of end-of-period marginal (marginal) value uses different approaches
    if IndepDstnBool:
        # bNrm represents R*a, balances after asset return shocks but before income.
        # This just uses the highest risky return as a rough shifter for the aXtraGrid.
        if BoroCnstNat_iszero:
            bNrmNow = np.insert(
                RiskyMaxNext * aXtraGrid, 0, RiskyMinNext * aXtraGrid[0]
            )
            aNrmNow = aXtraGrid.copy()
        else:
            # Add a bank balances point at exactly zero
            bNrmNow = RiskyMaxNext * np.insert(aXtraGrid, 0, 0.0)
            aNrmNow = np.insert(aXtraGrid, 0, 0.0)

        # Define local functions for taking future expectations when the interest
        # factor *is* independent from the income shock distribution. These go
        # from "bank balances" bNrm = R * aNrm to t+1 realizations.
        def calc_mNrmNext(S, b):
            return b / (PermGroFac * S["PermShk"]) + S["TranShk"]

        def calc_vNext(S, b):
            return S["PermShk"] ** (1.0 - CRRA) * vFuncNext(calc_mNrmNext(S, b))

        def calc_vPnext(S, b):
            return S["PermShk"] ** (-CRRA) * vPfuncNext(calc_mNrmNext(S, b))

        def calc_vPPnext(S, b):
            return S["PermShk"] ** (-CRRA - 1.0) * vPPfuncNext(calc_mNrmNext(S, b))

        # Calculate marginal value of bank balances at each gridpoint
        vPfacEff = PermGroFac ** (-CRRA)
        Intermed_vP = vPfacEff * expected(calc_vPnext, IncShkDstn, args=(bNrmNow))
        Intermed_vPnvrs = uFunc.derinv(Intermed_vP, order=(1, 0))

        if BoroCnstNat_iszero:
            Intermed_vPnvrs = np.insert(Intermed_vPnvrs, 0, 0.0)
            bNrm_temp = np.insert(bNrmNow, 0, 0.0)
        else:
            bNrm_temp = bNrmNow.copy()

        # If using cubic spline interpolation, also calculate "intermediate"
        # marginal marginal value of bank balances
        if CubicBool:
            vPPfacEff = PermGroFac ** (-CRRA - 1.0)
            Intermed_vPP = vPPfacEff * expected(
                calc_vPPnext, IncShkDstn, args=(bNrmNow)
            )
            Intermed_vPnvrsP = Intermed_vPP * uFunc.derinv(Intermed_vP, order=(1, 1))
            if BoroCnstNat_iszero:
                Intermed_vPnvrsP = np.insert(Intermed_vPnvrsP, 0, Intermed_vPnvrsP[0])

            # Make a cubic spline intermediate pseudo-inverse marginal value function
            Intermed_vPnvrsFunc = CubicInterp(
                bNrm_temp,
                Intermed_vPnvrs,
                Intermed_vPnvrsP,
                lower_extrap=True,
            )
            Intermed_vPPfunc = MargMargValueFuncCRRA(Intermed_vPnvrsFunc, CRRA)
        else:
            # Make a linear interpolation intermediate pseudo-inverse marginal value function
            Intermed_vPnvrsFunc = LinearInterp(
                bNrm_temp, Intermed_vPnvrs, lower_extrap=True
            )

        # "Recurve" the intermediate pseudo-inverse marginal value function
        Intermed_vPfunc = MargValueFuncCRRA(Intermed_vPnvrsFunc, CRRA)

        # If the value function is requested, calculate "intermediate" value
        if vFuncBool:
            vFacEff = PermGroFac ** (1.0 - CRRA)
            Intermed_v = vFacEff * expected(calc_vNext, IncShkDstn, args=(bNrmNow))
            Intermed_vNvrs = uFunc.inv(Intermed_v)
            # value transformed through inverse utility
            Intermed_vNvrsP = Intermed_vP * uFunc.derinv(Intermed_v, order=(0, 1))
            if BoroCnstNat_iszero:
                Intermed_vNvrs = np.insert(Intermed_vNvrs, 0, 0.0)
                Intermed_vNvrsP = np.insert(Intermed_vNvrsP, 0, Intermed_vNvrsP[0])
                # This is a very good approximation, vNvrsPP = 0 at the asset minimum

            # Make a cubic spline intermediate pseudo-inverse value function
            Intermed_vNvrsFunc = CubicInterp(bNrm_temp, Intermed_vNvrs, Intermed_vNvrsP)

            # "Recurve" the intermediate pseudo-inverse value function
            Intermed_vFunc = ValueFuncCRRA(Intermed_vNvrsFunc, CRRA)

        # We have "intermediate" (marginal) value functions defined over bNrm,
        # so now we want to take expectations over Risky realizations at each aNrm.

        # Begin by re-defining transition functions for taking expectations, which are all very simple!
        Z = RiskyShareFixed  # for shorter notation

        def calc_bNrmNext(R, a):
            Rport = Z * R + (1 - Z) * Rfree
            return Rport * a

        def calc_vNext(R, a):
            return Intermed_vFunc(calc_bNrmNext(R, a))

        def calc_vPnext(R, a):
            Rport = Z * R + (1 - Z) * Rfree
            return Rport * Intermed_vPfunc(calc_bNrmNext(R, a))

        def calc_vPPnext(R, a):
            Rport = Z * R + (1 - Z) * Rfree
            return Rport * Rport * Intermed_vPPfunc(calc_bNrmNext(R, a))

        # Calculate end-of-period marginal value of assets at each gridpoint
        EndOfPrdvP = DiscFacEff * expected(calc_vPnext, RiskyDstn, args=(aNrmNow))

        # Invert the first order condition to find optimal cNrm from each aNrm gridpoint
        cNrmNow = uFunc.derinv(EndOfPrdvP, order=(1, 0))
        mNrmNow = cNrmNow + aNrmNow  # Endogenous mNrm gridpoints

        # Calculate the MPC at each gridpoint if using cubic spline interpolation
        if CubicBool:
            # Calculate end-of-period marginal marginal value of assets at each gridpoint
            EndOfPrdvPP = DiscFacEff * expected(calc_vPPnext, RiskyDstn, args=(aNrmNow))
            dcda = EndOfPrdvPP / uFunc.der(np.array(cNrmNow), order=2)
            MPC = dcda / (dcda + 1.0)
            MPC_for_interpolation = np.insert(MPC, 0, MPCmaxNow)

        # Limiting consumption is zero as m approaches mNrmMin
        c_for_interpolation = np.insert(cNrmNow, 0, 0.0)
        m_for_interpolation = np.insert(mNrmNow, 0, BoroCnstNat)

        # Construct the end-of-period value function if requested
        if vFuncBool:
            # Calculate end-of-period value, its derivative, and their pseudo-inverse
            EndOfPrdv = DiscFacEff * expected(calc_vNext, RiskyDstn, args=(aNrmNow))
            EndOfPrdvNvrs = uFunc.inv(EndOfPrdv)
            # value transformed through inverse utility
            EndOfPrdvNvrsP = EndOfPrdvP * uFunc.derinv(EndOfPrdv, order=(0, 1))

            # Construct the end-of-period value function
            if BoroCnstNat_iszero:
                EndOfPrdvNvrs = np.insert(EndOfPrdvNvrs, 0, 0.0)
                EndOfPrdvNvrsP = np.insert(EndOfPrdvNvrsP, 0, EndOfPrdvNvrsP[0])
                # This is a very good approximation, vNvrsPP = 0 at the asset minimum
                aNrm_temp = np.insert(aNrmNow, 0, BoroCnstNat)
            else:
                aNrm_temp = aNrmNow.copy()

            EndOfPrd_vNvrsFunc = CubicInterp(aNrm_temp, EndOfPrdvNvrs, EndOfPrdvNvrsP)
            EndOfPrd_vFunc = ValueFuncCRRA(EndOfPrd_vNvrsFunc, CRRA)

    # NON-INDEPENDENT METHOD BEGINS HERE
    else:
        # Construct the assets grid by adjusting aXtra by the natural borrowing constraint
        # aNrmNow = np.asarray(aXtraGrid) + BoroCnstNat
        if BoroCnstNat_iszero:
            aNrmNow = aXtraGrid
        else:
            # Add an asset point at exactly zero
            aNrmNow = np.insert(aXtraGrid, 0, 0.0)

        # Define local functions for taking future expectations when the interest
        # factor is *not* independent from the income shock distribution
        Z = RiskyShareFixed  # for shorter notation

        def calc_mNrmNext(S, a):
            Risky = S["Risky"]
            Rport = Z * Risky + (1 - Z) * Rfree
            return Rport / (PermGroFac * S["PermShk"]) * a + S["TranShk"]

        def calc_vNext(S, a):
            return S["PermShk"] ** (1.0 - CRRA) * vFuncNext(calc_mNrmNext(S, a))

        def calc_vPnext(S, a):
            Risky = S["Risky"]
            Rport = Z * Risky + (1 - Z) * Rfree
            return Rport * S["PermShk"] ** (-CRRA) * vPfuncNext(calc_mNrmNext(S, a))

        def calc_vPPnext(S, a):
            Risky = S["Risky"]
            Rport = Z * Risky + (1 - Z) * Rfree
            return (
                (Rport**2)
                * S["PermShk"] ** (-CRRA - 1.0)
                * vPPfuncNext(calc_mNrmNext(S, a))
            )

        # Calculate end-of-period marginal value of assets at each gridpoint
        vPfacEff = DiscFacEff * PermGroFac ** (-CRRA)
        EndOfPrdvP = vPfacEff * expected(calc_vPnext, ShockDstn, args=(aNrmNow))

        # Invert the first order condition to find optimal cNrm from each aNrm gridpoint
        cNrmNow = uFunc.derinv(EndOfPrdvP, order=(1, 0))
        mNrmNow = cNrmNow + aNrmNow  # Endogenous mNrm gridpoints

        # Calculate the MPC at each gridpoint if using cubic spline interpolation
        if CubicBool:
            # Calculate end-of-period marginal marginal value of assets at each gridpoint
            vPPfacEff = DiscFacEff * PermGroFac ** (-CRRA - 1.0)
            EndOfPrdvPP = vPPfacEff * expected(calc_vPPnext, ShockDstn, args=(aNrmNow))
            dcda = EndOfPrdvPP / uFunc.der(np.array(cNrmNow), order=2)
            MPC = dcda / (dcda + 1.0)
            MPC_for_interpolation = np.insert(MPC, 0, MPCmaxNow)

        # Limiting consumption is zero as m approaches mNrmMin
        c_for_interpolation = np.insert(cNrmNow, 0, 0.0)
        m_for_interpolation = np.insert(mNrmNow, 0, BoroCnstNat)

        # Construct the end-of-period value function if requested
        if vFuncBool:
            # Calculate end-of-period value, its derivative, and their pseudo-inverse
            vFacEff = DiscFacEff * PermGroFac ** (1.0 - CRRA)
            EndOfPrdv = vFacEff * expected(calc_vNext, ShockDstn, args=(aNrmNow))
            EndOfPrdvNvrs = uFunc.inv(EndOfPrdv)
            # value transformed through inverse utility
            EndOfPrdvNvrsP = EndOfPrdvP * uFunc.derinv(EndOfPrdv, order=(0, 1))

            # Construct the end-of-period value function
            if BoroCnstNat_iszero:
                EndOfPrdvNvrs = np.insert(EndOfPrdvNvrs, 0, 0.0)
                EndOfPrdvNvrsP = np.insert(EndOfPrdvNvrsP, 0, EndOfPrdvNvrsP[0])
                # This is a very good approximation, vNvrsPP = 0 at the asset minimum
                aNrm_temp = np.insert(aNrmNow, 0, BoroCnstNat)
            else:
                aNrm_temp = aNrmNow.copy()

            EndOfPrd_vNvrsFunc = CubicInterp(aNrm_temp, EndOfPrdvNvrs, EndOfPrdvNvrsP)
            EndOfPrd_vFunc = ValueFuncCRRA(EndOfPrd_vNvrsFunc, CRRA)

    # Construct the consumption function; this uses the same method whether the
    # income distribution is independent from the return distribution
    if CubicBool:
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

    # Make the marginal value function and the marginal marginal value function
    vPfuncNow = MargValueFuncCRRA(cFuncNow, CRRA)

    # Define this period's marginal marginal value function
    if CubicBool:
        vPPfuncNow = MargMargValueFuncCRRA(cFuncNow, CRRA)
    else:
        vPPfuncNow = NullFunc()  # Dummy object

    # Construct this period's value function if requested. This version is set
    # up for the non-independent distributions, need to write a faster version.
    if vFuncBool:
        # Compute expected value and marginal value on a grid of market resources
        mNrm_temp = mNrmMinNow + aXtraGrid
        cNrm_temp = cFuncNow(mNrm_temp)
        aNrm_temp = np.maximum(mNrm_temp - cNrm_temp, 0.0)  # fix tiny errors
        v_temp = uFunc(cNrm_temp) + EndOfPrd_vFunc(aNrm_temp)
        vP_temp = uFunc.der(cNrm_temp)

        # Construct the beginning-of-period value function
        vNvrs_temp = uFunc.inv(v_temp)  # value transformed through inv utility
        vNvrsP_temp = vP_temp * uFunc.derinv(v_temp, order=(0, 1))
        mNrm_temp = np.insert(mNrm_temp, 0, mNrmMinNow)
        vNvrs_temp = np.insert(vNvrs_temp, 0, 0.0)
        vNvrsP_temp = np.insert(vNvrsP_temp, 0, MPCmaxEff ** (-CRRA / (1.0 - CRRA)))
        # MPCminNvrs = MPCminNow ** (-CRRA / (1.0 - CRRA))
        vNvrsFuncNow = CubicInterp(mNrm_temp, vNvrs_temp, vNvrsP_temp)
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
    solution_now.ShareFunc = ConstantFunction(RiskyShareFixed)
    return solution_now

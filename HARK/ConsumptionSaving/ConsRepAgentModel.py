"""
This module contains models for solving representative agent macroeconomic models.
This stands in contrast to all other model modules in HARK, which (unsurprisingly)
take a heterogeneous agents approach.  In RA models, all attributes are either
time invariant or exist on a short cycle; models must be infinite horizon.
"""

import numpy as np
from HARK.Calibration.Income.IncomeProcesses import (
    construct_lognormal_income_process_unemployment,
    get_PermShkDstn_from_IncShkDstn,
    get_TranShkDstn_from_IncShkDstn,
)
from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsumerSolution,
    IndShockConsumerType,
    make_basic_CRRA_solution_terminal,
)
from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType
from HARK.distributions import MarkovProcess
from HARK.interpolation import LinearInterp, MargValueFuncCRRA
from HARK.utilities import make_assets_grid

__all__ = ["RepAgentConsumerType", "RepAgentMarkovConsumerType"]


def solve_ConsRepAgent(
    solution_next, DiscFac, CRRA, IncShkDstn, CapShare, DeprFac, PermGroFac, aXtraGrid
):
    """
    Solve one period of the simple representative agent consumption-saving model.

    Parameters
    ----------
    solution_next : ConsumerSolution
        Solution to the next period's problem (i.e. previous iteration).
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
    IncShkDstn : distribution.Distribution
        A discrete approximation to the income process between the period being
        solved and the one immediately following (in solution_next). Order:
        permanent shocks, transitory shocks.
    CapShare : float
        Capital's share of income in Cobb-Douglas production function.
    DeprFac : float
        Depreciation rate for capital.
    PermGroFac : float
        Expected permanent income growth factor at the end of this period.
    aXtraGrid : np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.  In this model, the minimum acceptable
        level is always zero.

    Returns
    -------
    solution_now : ConsumerSolution
        Solution to this period's problem (new iteration).
    """
    # Unpack next period's solution and the income distribution
    vPfuncNext = solution_next.vPfunc
    ShkPrbsNext = IncShkDstn.pmv
    PermShkValsNext = IncShkDstn.atoms[0]
    TranShkValsNext = IncShkDstn.atoms[1]

    # Make tiled versions of end-of-period assets, shocks, and probabilities
    aNrmNow = aXtraGrid
    aNrmCount = aNrmNow.size
    ShkCount = ShkPrbsNext.size
    aNrm_tiled = np.tile(np.reshape(aNrmNow, (aNrmCount, 1)), (1, ShkCount))

    # Tile arrays of the income shocks and put them into useful shapes
    PermShkVals_tiled = np.tile(
        np.reshape(PermShkValsNext, (1, ShkCount)), (aNrmCount, 1)
    )
    TranShkVals_tiled = np.tile(
        np.reshape(TranShkValsNext, (1, ShkCount)), (aNrmCount, 1)
    )
    ShkPrbs_tiled = np.tile(np.reshape(ShkPrbsNext, (1, ShkCount)), (aNrmCount, 1))

    # Calculate next period's capital-to-permanent-labor ratio under each combination
    # of end-of-period assets and shock realization
    kNrmNext = aNrm_tiled / (PermGroFac * PermShkVals_tiled)

    # Calculate next period's market resources
    KtoLnext = kNrmNext / TranShkVals_tiled
    RfreeNext = 1.0 - DeprFac + CapShare * KtoLnext ** (CapShare - 1.0)
    wRteNext = (1.0 - CapShare) * KtoLnext**CapShare
    mNrmNext = RfreeNext * kNrmNext + wRteNext * TranShkVals_tiled

    # Calculate end-of-period marginal value of assets for the RA
    vPnext = vPfuncNext(mNrmNext)
    EndOfPrdvP = DiscFac * np.sum(
        RfreeNext
        * (PermGroFac * PermShkVals_tiled) ** (-CRRA)
        * vPnext
        * ShkPrbs_tiled,
        axis=1,
    )

    # Invert the first order condition to get consumption, then find endogenous gridpoints
    cNrmNow = EndOfPrdvP ** (-1.0 / CRRA)
    mNrmNow = aNrmNow + cNrmNow

    # Construct the consumption function and the marginal value function
    cFuncNow = LinearInterp(np.insert(mNrmNow, 0, 0.0), np.insert(cNrmNow, 0, 0.0))
    vPfuncNow = MargValueFuncCRRA(cFuncNow, CRRA)

    # Construct and return the solution for this period
    solution_now = ConsumerSolution(cFunc=cFuncNow, vPfunc=vPfuncNow)
    return solution_now


def solve_ConsRepAgentMarkov(
    solution_next,
    MrkvArray,
    DiscFac,
    CRRA,
    IncShkDstn,
    CapShare,
    DeprFac,
    PermGroFac,
    aXtraGrid,
):
    """
    Solve one period of the simple representative agent consumption-saving model.
    This version supports a discrete Markov process.

    Parameters
    ----------
    solution_next : ConsumerSolution
        Solution to the next period's problem (i.e. previous iteration).
    MrkvArray : np.array
        Markov transition array between this period and next period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
    IncShkDstn : [distribution.Distribution]
        A list of discrete approximations to the income process between the
        period being solved and the one immediately following (in solution_next).
        Order: event probabilities, permanent shocks, transitory shocks.
    CapShare : float
        Capital's share of income in Cobb-Douglas production function.
    DeprFac : float
        Depreciation rate of capital.
    PermGroFac : [float]
        Expected permanent income growth factor for each state we could be in
        next period.
    aXtraGrid : np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.  In this model, the minimum acceptable
        level is always zero.

    Returns
    -------
    solution_now : ConsumerSolution
        Solution to this period's problem (new iteration).
    """
    # Define basic objects
    StateCount = MrkvArray.shape[0]
    aNrmNow = aXtraGrid
    aNrmCount = aNrmNow.size
    EndOfPrdvP_cond = np.zeros((StateCount, aNrmCount)) + np.nan

    # Loop over *next period* states, calculating conditional EndOfPrdvP
    for j in range(StateCount):
        # Define next-period-state conditional objects
        vPfuncNext = solution_next.vPfunc[j]
        ShkPrbsNext = IncShkDstn[j].pmv
        PermShkValsNext = IncShkDstn[j].atoms[0]
        TranShkValsNext = IncShkDstn[j].atoms[1]

        # Make tiled versions of end-of-period assets, shocks, and probabilities
        ShkCount = ShkPrbsNext.size
        aNrm_tiled = np.tile(np.reshape(aNrmNow, (aNrmCount, 1)), (1, ShkCount))

        # Tile arrays of the income shocks and put them into useful shapes
        PermShkVals_tiled = np.tile(
            np.reshape(PermShkValsNext, (1, ShkCount)), (aNrmCount, 1)
        )
        TranShkVals_tiled = np.tile(
            np.reshape(TranShkValsNext, (1, ShkCount)), (aNrmCount, 1)
        )
        ShkPrbs_tiled = np.tile(np.reshape(ShkPrbsNext, (1, ShkCount)), (aNrmCount, 1))

        # Calculate next period's capital-to-permanent-labor ratio under each combination
        # of end-of-period assets and shock realization
        kNrmNext = aNrm_tiled / (PermGroFac[j] * PermShkVals_tiled)

        # Calculate next period's market resources
        KtoLnext = kNrmNext / TranShkVals_tiled
        RfreeNext = 1.0 - DeprFac + CapShare * KtoLnext ** (CapShare - 1.0)
        wRteNext = (1.0 - CapShare) * KtoLnext**CapShare
        mNrmNext = RfreeNext * kNrmNext + wRteNext * TranShkVals_tiled

        # Calculate end-of-period marginal value of assets for the RA
        vPnext = vPfuncNext(mNrmNext)
        EndOfPrdvP_cond[j, :] = DiscFac * np.sum(
            RfreeNext
            * (PermGroFac[j] * PermShkVals_tiled) ** (-CRRA)
            * vPnext
            * ShkPrbs_tiled,
            axis=1,
        )

    # Apply the Markov transition matrix to get unconditional end-of-period marginal value
    EndOfPrdvP = np.dot(MrkvArray, EndOfPrdvP_cond)

    # Construct the consumption function and marginal value function for each discrete state
    cFuncNow_list = []
    vPfuncNow_list = []
    for i in range(StateCount):
        # Invert the first order condition to get consumption, then find endogenous gridpoints
        cNrmNow = EndOfPrdvP[i, :] ** (-1.0 / CRRA)
        mNrmNow = aNrmNow + cNrmNow

        # Construct the consumption function and the marginal value function
        cFuncNow_list.append(
            LinearInterp(np.insert(mNrmNow, 0, 0.0), np.insert(cNrmNow, 0, 0.0))
        )
        vPfuncNow_list.append(MargValueFuncCRRA(cFuncNow_list[-1], CRRA))

    # Construct and return the solution for this period
    solution_now = ConsumerSolution(cFunc=cFuncNow_list, vPfunc=vPfuncNow_list)
    return solution_now


###############################################################################

# Make a dictionary of constructors for the representative agent model
repagent_constructor_dict = {
    "IncShkDstn": construct_lognormal_income_process_unemployment,
    "PermShkDstn": get_PermShkDstn_from_IncShkDstn,
    "TranShkDstn": get_TranShkDstn_from_IncShkDstn,
    "aXtraGrid": make_assets_grid,
    "solution_terminal": make_basic_CRRA_solution_terminal,
}

# Default parameters to make IncShkDstn using construct_lognormal_income_process_unemployment
default_IncShkDstn_params = {
    "PermShkStd": [0.1],  # Standard deviation of log permanent income shocks
    "PermShkCount": 7,  # Number of points in discrete approximation to permanent income shocks
    "TranShkStd": [0.1],  # Standard deviation of log transitory income shocks
    "TranShkCount": 7,  # Number of points in discrete approximation to transitory income shocks
    "UnempPrb": 0.00,  # Probability of unemployment while working
    "IncUnemp": 0.0,  # Unemployment benefits replacement rate while working
    "T_retire": 0,  # Period of retirement (0 --> no retirement)
    "UnempPrbRet": 0.005,  # Probability of "unemployment" while retired
    "IncUnempRet": 0.0,  # "Unemployment" benefits when retired
}

# Default parameters to make aXtraGrid using make_assets_grid
default_aXtraGrid_params = {
    "aXtraMin": 0.001,  # Minimum end-of-period "assets above minimum" value
    "aXtraMax": 20,  # Maximum end-of-period "assets above minimum" value
    "aXtraNestFac": 3,  # Exponential nesting factor for aXtraGrid
    "aXtraCount": 48,  # Number of points in the grid of "assets above minimum"
    "aXtraExtra": None,  # Additional other values to add in grid (optional)
}

# Make a dictionary to specify a representative agent consumer type
init_rep_agent = {
    # BASIC HARK PARAMETERS REQUIRED TO SOLVE THE MODEL
    "cycles": 0,  # Finite, non-cyclic model
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    "constructors": repagent_constructor_dict,  # See dictionary above
    # PRIMITIVE RAW PARAMETERS REQUIRED TO SOLVE THE MODEL
    "CRRA": 2.0,  # Coefficient of relative risk aversion
    "Rfree": 1.03,  # Interest factor on retained assets
    "DiscFac": 0.96,  # Intertemporal discount factor
    "LivPrb": [1.0],  # Survival probability after each period
    "PermGroFac": [1.01],  # Permanent income growth factor
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint
    "DeprFac": 0.05,  # Depreciation rate for capital
    "CapShare": 0.36,  # Capital's share in Cobb-Douglas production function
    "vFuncBool": False,  # Whether to calculate the value function during solution
    "CubicBool": False,  # Whether to use cubic spline interpolation when True
    # (Uses linear spline interpolation for cFunc when False)
    # PARAMETERS REQUIRED TO SIMULATE THE MODEL
    "AgentCount": 1,  # Number of agents of this type
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
init_rep_agent.update(default_IncShkDstn_params)
init_rep_agent.update(default_aXtraGrid_params)


class RepAgentConsumerType(IndShockConsumerType):
    """
    A class for representing representative agents with inelastic labor supply.

    Parameters
    ----------

    """

    time_inv_ = IndShockConsumerType.time_inv_ + ["CapShare", "DeprFac"]

    def __init__(self, **kwds):
        params = init_rep_agent.copy()
        params.update(kwds)

        IndShockConsumerType.__init__(self, **params)
        self.AgentCount = 1  # Hardcoded, because this is rep agent
        self.solve_one_period = solve_ConsRepAgent
        self.del_from_time_inv("Rfree", "BoroCnstArt", "vFuncBool", "CubicBool")

    def pre_solve(self):
        self.update_solution_terminal()

    def get_states(self):
        """
        TODO: replace with call to transition

        Calculates updated values of normalized market resources and permanent income level.
        Uses pLvlNow, aNrmNow, PermShkNow, TranShkNow.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pLvlPrev = self.state_prev["pLvl"]
        aNrmPrev = self.state_prev["aNrm"]

        # Calculate new states: normalized market resources and permanent income level
        self.pLvlNow = pLvlPrev * self.shocks["PermShk"]  # Same as in IndShockConsType
        self.kNrmNow = aNrmPrev / self.shocks["PermShk"]
        self.yNrmNow = self.kNrmNow**self.CapShare * self.shocks["TranShk"] ** (
            1.0 - self.CapShare
        )
        self.Rfree = (
            1.0
            + self.CapShare
            * self.kNrmNow ** (self.CapShare - 1.0)
            * self.shocks["TranShk"] ** (1.0 - self.CapShare)
            - self.DeprFac
        )
        self.wRte = (
            (1.0 - self.CapShare)
            * self.kNrmNow**self.CapShare
            * self.shocks["TranShk"] ** (-self.CapShare)
        )
        self.mNrmNow = self.Rfree * self.kNrmNow + self.wRte * self.shocks["TranShk"]


###############################################################################

# Define the default dictionary for a markov representative agent type
init_markov_rep_agent = init_rep_agent.copy()
init_markov_rep_agent["PermGroFac"] = [[0.97, 1.03]]
init_markov_rep_agent["MrkvArray"] = np.array([[0.99, 0.01], [0.01, 0.99]])
init_markov_rep_agent["Mrkv"] = 0


class RepAgentMarkovConsumerType(RepAgentConsumerType):
    """
    A class for representing representative agents with inelastic labor supply
    and a discrete MarkovState

    Parameters
    ----------
    """

    time_inv_ = RepAgentConsumerType.time_inv_ + ["MrkvArray"]

    def __init__(self, **kwds):
        params = init_markov_rep_agent.copy()
        params.update(kwds)

        RepAgentConsumerType.__init__(self, **params)
        self.solve_one_period = solve_ConsRepAgentMarkov

    def pre_solve(self):
        self.update_solution_terminal()

    def initialize_sim(self):
        # self.shocks["Mrkv"] = np.zeros(self.AgentCount, dtype=int)
        RepAgentConsumerType.initialize_sim(self)
        self.shocks["Mrkv"] = self.Mrkv

    def update_solution_terminal(self):
        """
        Update the terminal period solution.  This method should be run when a
        new AgentType is created or when CRRA changes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        RepAgentConsumerType.update_solution_terminal(self)

        # Make replicated terminal period solution
        StateCount = self.MrkvArray.shape[0]
        self.solution_terminal.cFunc = StateCount * [self.cFunc_terminal_]
        self.solution_terminal.vPfunc = StateCount * [self.solution_terminal.vPfunc]
        self.solution_terminal.mNrmMin = StateCount * [self.solution_terminal.mNrmMin]

    def reset_rng(self):
        MarkovConsumerType.reset_rng(self)

    def get_shocks(self):
        """
        Draws a new Markov state and income shocks for the representative agent.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.shocks["Mrkv"] = MarkovProcess(
            self.MrkvArray, seed=self.RNG.integers(0, 2**31 - 1)
        ).draw(self.shocks["Mrkv"])

        t = self.t_cycle[0]
        i = self.shocks["Mrkv"]
        IncShkDstnNow = self.IncShkDstn[t - 1][i]  # set current income distribution
        PermGroFacNow = self.PermGroFac[t - 1][i]  # and permanent growth factor
        # Get random draws of income shocks from the discrete distribution
        EventDraw = IncShkDstnNow.draw_events(1)
        PermShkNow = (
            IncShkDstnNow.atoms[0][EventDraw] * PermGroFacNow
        )  # permanent "shock" includes expected growth
        TranShkNow = IncShkDstnNow.atoms[1][EventDraw]
        self.shocks["PermShk"] = np.array(PermShkNow)
        self.shocks["TranShk"] = np.array(TranShkNow)

    def get_controls(self):
        """
        Calculates consumption for the representative agent using the consumption functions.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        t = self.t_cycle[0]
        i = self.shocks["Mrkv"]
        self.controls["cNrm"] = self.solution[t].cFunc[i](self.mNrmNow)

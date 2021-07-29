"""
This module contains models for solving representative agent macroeconomic models.
This stands in contrast to all other model modules in HARK, which (unsurprisingly)
take a heterogeneous agents approach.  In RA models, all attributes are either
time invariant or exist on a short cycle; models must be infinite horizon.
"""
import numpy as np
from HARK.interpolation import LinearInterp, MargValueFuncCRRA
from HARK.distribution import (MarkovProcess, Uniform)
from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    ConsumerSolution,
    init_idiosyncratic_shocks,
)
from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType

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
        A discrete
        approximation to the income process between the period being solved
        and the one immediately following (in solution_next). Order: 
        permanent shocks, transitory shocks.
    CapShare : float
        Capital's share of income in Cobb-Douglas production function.
    DeprFac : float
        Depreciation rate of capital.
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
    ShkPrbsNext = IncShkDstn.pmf
    PermShkValsNext = IncShkDstn.X[0]
    TranShkValsNext = IncShkDstn.X[1]

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
    wRteNext = (1.0 - CapShare) * KtoLnext ** CapShare
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
        A list of discrete
        approximations to the income process between the period being solved
        and the one immediately following (in solution_next). Order: event
        probabilities, permanent shocks, transitory shocks.
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
        ShkPrbsNext = IncShkDstn[j].pmf
        PermShkValsNext = IncShkDstn[j].X[0]
        TranShkValsNext = IncShkDstn[j].X[1]

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
        wRteNext = (1.0 - CapShare) * KtoLnext ** CapShare
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
        pLvlPrev = self.state_prev['pLvl']
        aNrmPrev = self.state_prev['aNrm']

        # Calculate new states: normalized market resources and permanent income level
        self.pLvlNow = (
            pLvlPrev * self.shocks['PermShk']
        )  # Same as in IndShockConsType
        self.kNrmNow = aNrmPrev / self.shocks['PermShk']
        self.yNrmNow = self.kNrmNow ** self.CapShare * self.shocks['TranShk'] ** (
            1.0 - self.CapShare
        )
        self.Rfree = (
            1.0
            + self.CapShare
            * self.kNrmNow ** (self.CapShare - 1.0)
            * self.shocks['TranShk'] ** (1.0 - self.CapShare)
            - self.DeprFac
        )
        self.wRte = (
            (1.0 - self.CapShare)
            * self.kNrmNow ** self.CapShare
            * self.shocks['TranShk'] ** (-self.CapShare)
        )
        self.mNrmNow = self.Rfree * self.kNrmNow + self.wRte * self.shocks['TranShk']


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
        #self.shocks["Mrkv"] = np.zeros(self.AgentCount, dtype=int)
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
            self.MrkvArray,
            seed=self.RNG.randint(0, 2 ** 31 - 1)
            ).draw(self.shocks["Mrkv"])

        t = self.t_cycle[0]
        i = self.shocks["Mrkv"]
        IncShkDstnNow = self.IncShkDstn[t - 1][i]  # set current income distribution
        PermGroFacNow = self.PermGroFac[t - 1][i]  # and permanent growth factor
        # Get random draws of income shocks from the discrete distribution
        EventDraw = IncShkDstnNow.draw_events(1)
        PermShkNow = (
            IncShkDstnNow.X[0][EventDraw] * PermGroFacNow
        )  # permanent "shock" includes expected growth
        TranShkNow = IncShkDstnNow.X[1][EventDraw]
        self.shocks['PermShk'] = np.array(PermShkNow)
        self.shocks['TranShk'] = np.array(TranShkNow)

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
        self.controls['cNrm'] = self.solution[t].cFunc[i](self.mNrmNow)


# Define the default dictionary for a representative agent type
init_rep_agent = init_idiosyncratic_shocks.copy()
init_rep_agent["DeprFac"] = 0.05
init_rep_agent["CapShare"] = 0.36
init_rep_agent["UnempPrb"] = 0.0
init_rep_agent["LivPrb"] = [1.0]

# Define the default dictionary for a markov representative agent type
init_markov_rep_agent = init_rep_agent.copy()
init_markov_rep_agent["PermGroFac"] = [[0.97, 1.03]]
init_markov_rep_agent["MrkvArray"] = np.array([[0.99, 0.01], [0.01, 0.99]])
init_markov_rep_agent["Mrkv"] = 0

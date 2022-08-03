"""
Consumption-saving models with aggregate productivity shocks as well as idiosyn-
cratic income shocks.  Currently only contains one microeconomic model with a
basic solver.  Also includes a subclass of Market called CobbDouglas economy,
used for solving "macroeconomic" models with aggregate shocks.
"""
import numpy as np
import scipy.stats as stats
from HARK.interpolation import (
    LinearInterp,
    LinearInterpOnInterp1D,
    ConstantFunction,
    IdentityFunction,
    VariableLowerBoundFunc2D,
    BilinearInterp,
    LowerEnvelope2D,
    UpperEnvelope,
    MargValueFuncCRRA,
)
from HARK.utilities import (
    CRRAutility,
    CRRAutilityP,
    CRRAutilityPP,
    CRRAutilityP_inv,
    CRRAutility_invP,
    CRRAutility_inv,
    make_grid_exp_mult,
)
from HARK.distribution import (
    MarkovProcess,
    MeanOneLogNormal,
    Uniform,
    combine_indep_dstns,
    calc_expectation,
)
from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsumerSolution,
    IndShockConsumerType,
    init_idiosyncratic_shocks,
)
from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType
from HARK import MetricObject, Market, AgentType
from copy import deepcopy
import matplotlib.pyplot as plt

__all__ = [
    "AggShockConsumerType",
    "AggShockMarkovConsumerType",
    "CobbDouglasEconomy",
    "SmallOpenEconomy",
    "CobbDouglasMarkovEconomy",
    "SmallOpenMarkovEconomy",
    "AggregateSavingRule",
    "AggShocksDynamicRule",
    "init_agg_shocks",
    "init_agg_mrkv_shocks",
    "init_cobb_douglas",
    "init_mrkv_cobb_douglas",
]

utility = CRRAutility
utilityP = CRRAutilityP
utilityPP = CRRAutilityPP
utilityP_inv = CRRAutilityP_inv
utility_invP = CRRAutility_invP
utility_inv = CRRAutility_inv


class MargValueFunc2D(MetricObject):
    """
    A class for representing a marginal value function in models where the
    standard envelope condition of dvdm(m,M) = u'(c(m,M)) holds (with CRRA utility).
    """

    distance_criteria = ["cFunc", "CRRA"]

    def __init__(self, cFunc, CRRA):
        """
        Constructor for a new marginal value function object.

        Parameters
        ----------
        cFunc : function
            A real function representing the marginal value function composed
            with the inverse marginal utility function, defined on normalized individual market
            resources and aggregate market resources-to-labor ratio: uP_inv(vPfunc(m,M)).
            Called cFunc because when standard envelope condition applies,
            uP_inv(vPfunc(m,M)) = cFunc(m,M).
        CRRA : float
            Coefficient of relative risk aversion.

        Returns
        -------
        new instance of MargValueFunc
        """
        self.cFunc = deepcopy(cFunc)
        self.CRRA = CRRA

    def __call__(self, m, M):
        return utilityP(self.cFunc(m, M), gam=self.CRRA)


###############################################################################

# Make a dictionary to specify an aggregate shocks consumer
init_agg_shocks = init_idiosyncratic_shocks.copy()
del init_agg_shocks["Rfree"]  # Interest factor is endogenous in agg shocks model
del init_agg_shocks["CubicBool"]  # Not supported yet for agg shocks model
del init_agg_shocks["vFuncBool"]  # Not supported yet for agg shocks model
init_agg_shocks["PermGroFac"] = [1.0]
# Grid of capital-to-labor-ratios (factors)
MgridBase = np.array(
    [0.1, 0.3, 0.6, 0.8, 0.9, 0.98, 1.0, 1.02, 1.1, 1.2, 1.6, 2.0, 3.0]
)
init_agg_shocks["MgridBase"] = MgridBase
init_agg_shocks["aXtraCount"] = 24
init_agg_shocks["aNrmInitStd"] = 0.0
init_agg_shocks["LivPrb"] = [0.98]


class AggShockConsumerType(IndShockConsumerType):
    """
    A class to represent consumers who face idiosyncratic (transitory and per-
    manent) shocks to their income and live in an economy that has aggregate
    (transitory and permanent) shocks to labor productivity.  As the capital-
    to-labor ratio varies in the economy, so does the wage rate and interest
    rate.  "Aggregate shock consumers" have beliefs about how the capital ratio
    evolves over time and take aggregate shocks into account when making their
    decision about how much to consume.
    """

    def __init__(self, **kwds):
        """
        Make a new instance of AggShockConsumerType, an extension of
        IndShockConsumerType.  Sets appropriate solver and input lists.
        """
        params = init_agg_shocks.copy()
        params.update(kwds)

        AgentType.__init__(
            self,
            solution_terminal=deepcopy(IndShockConsumerType.solution_terminal_),
            pseudo_terminal=False,
            **params
        )

        # Add consumer-type specific objects, copying to create independent versions
        self.time_vary = deepcopy(IndShockConsumerType.time_vary_)
        self.time_inv = deepcopy(IndShockConsumerType.time_inv_)
        self.del_from_time_inv("Rfree", "vFuncBool", "CubicBool")

        self.solve_one_period = solveConsAggShock
        self.update()

    def reset(self):
        """
        Initialize this type for a new simulated history of K/L ratio.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.initialize_sim()
        self.state_now["aLvlNow"] = self.kInit * np.ones(
            self.AgentCount
        )  # Start simulation near SS
        self.state_now["aNrm"] = (
            self.state_now["aLvlNow"] / self.state_now["pLvl"]
        )  # ???

    def pre_solve(self):
        #        AgentType.pre_solve()
        self.update_solution_terminal()

    def update_solution_terminal(self):
        """
        Updates the terminal period solution for an aggregate shock consumer.
        Only fills in the consumption function and marginal value function.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        cFunc_terminal = BilinearInterp(
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        )
        vPfunc_terminal = MargValueFuncCRRA(cFunc_terminal, self.CRRA)
        mNrmMin_terminal = ConstantFunction(0)
        self.solution_terminal = ConsumerSolution(
            cFunc=cFunc_terminal, vPfunc=vPfunc_terminal, mNrmMin=mNrmMin_terminal
        )

    def get_economy_data(self, economy):
        """
        Imports economy-determined objects into self from a Market.
        Instances of AggShockConsumerType "live" in some macroeconomy that has
        attributes relevant to their microeconomic model, like the relationship
        between the capital-to-labor ratio and the interest and wage rates; this
        method imports those attributes from an "economy" object and makes them
        attributes of the ConsumerType.

        Parameters
        ----------
        economy : Market
            The "macroeconomy" in which this instance "lives".  Might be of the
            subclass CobbDouglasEconomy, which has methods to generate the
            relevant attributes.

        Returns
        -------
        None
        """
        self.T_sim = (
            economy.act_T
        )  # Need to be able to track as many periods as economy runs
        self.kInit = economy.kSS  # Initialize simulation assets to steady state
        self.aNrmInitMean = np.log(
            0.00000001
        )  # Initialize newborn assets to nearly zero
        self.Mgrid = (
            economy.MSS * self.MgridBase
        )  # Aggregate market resources grid adjusted around SS capital ratio
        self.AFunc = economy.AFunc  # Next period's aggregate savings function
        self.Rfunc = economy.Rfunc  # Interest factor as function of capital ratio
        self.wFunc = economy.wFunc  # Wage rate as function of capital ratio
        self.DeprFac = economy.DeprFac  # Rate of capital depreciation
        self.PermGroFacAgg = (
            economy.PermGroFacAgg
        )  # Aggregate permanent productivity growth
        self.add_AggShkDstn(
            economy.AggShkDstn
        )  # Combine idiosyncratic and aggregate shocks into one dstn
        self.add_to_time_inv(
            "Mgrid", "AFunc", "Rfunc", "wFunc", "DeprFac", "PermGroFacAgg"
        )

    def add_AggShkDstn(self, AggShkDstn):
        """
        Updates attribute IncShkDstn by combining idiosyncratic shocks with aggregate shocks.

        Parameters
        ----------
        AggShkDstn : [np.array]
            Aggregate productivity shock distribution.  First element is proba-
            bilities, second element is agg permanent shocks, third element is
            agg transitory shocks.

        Returns
        -------
        None
        """
        if len(self.IncShkDstn[0].data) > 2:
            self.IncShkDstn = self.IncShkDstnWithoutAggShocks
        else:
            self.IncShkDstnWithoutAggShocks = self.IncShkDstn
        self.IncShkDstn = [
            combine_indep_dstns(self.IncShkDstn[t], AggShkDstn)
            for t in range(self.T_cycle)
        ]

    def sim_birth(self, which_agents):
        """
        Makes new consumers for the given indices.  Initialized variables include aNrm and pLvl, as
        well as time variables t_age and t_cycle.  Normalized assets and permanent income levels
        are drawn from lognormal distributions given by aNrmInitMean and aNrmInitStd (etc).

        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.AgentCount indicating which agents should be "born".

        Returns
        -------
        None
        """
        IndShockConsumerType.sim_birth(self, which_agents)
        if "aLvl" in self.state_now and self.state_now["aLvl"] is not None:
            self.state_now["aLvl"][which_agents] = (
                self.state_now["aNrm"][which_agents]
                * self.state_now["pLvl"][which_agents]
            )
        else:
            self.state_now["aLvl"] = self.state_now["aNrm"] * self.state_now["pLvl"]

    def sim_death(self):
        """
        Randomly determine which consumers die, and distribute their wealth among the survivors.
        This method only works if there is only one period in the cycle.

        Parameters
        ----------
        None

        Returns
        -------
        who_dies : np.array(bool)
            Boolean array of size AgentCount indicating which agents die.
        """
        # Divide agents into wealth groups, kill one random agent per wealth group
        #        order = np.argsort(self.aLvlNow)
        #        how_many_die = int(self.AgentCount*(1.0-self.LivPrb[0]))
        #        group_size = self.AgentCount/how_many_die # This should be an integer
        #        base_idx = self.RNG.randint(0,group_size,size=how_many_die)
        #        kill_by_rank = np.arange(how_many_die,dtype=int)*group_size + base_idx
        #        who_dies = np.zeros(self.AgentCount,dtype=bool)
        #        who_dies[order[kill_by_rank]] = True

        # Just select a random set of agents to die
        how_many_die = int(round(self.AgentCount * (1.0 - self.LivPrb[0])))
        base_bool = np.zeros(self.AgentCount, dtype=bool)
        base_bool[0:how_many_die] = True
        who_dies = self.RNG.permutation(base_bool)
        if self.T_age is not None:
            who_dies[self.t_age >= self.T_age] = True

        # Divide up the wealth of those who die, giving it to those who survive
        who_lives = np.logical_not(who_dies)
        wealth_living = np.sum(self.state_now["aLvl"][who_lives])
        wealth_dead = np.sum(self.state_now["aLvl"][who_dies])
        Ractuarial = 1.0 + wealth_dead / wealth_living
        self.state_now["aNrm"][who_lives] = (
            self.state_now["aNrm"][who_lives] * Ractuarial
        )
        self.state_now["aLvl"][who_lives] = (
            self.state_now["aLvl"][who_lives] * Ractuarial
        )
        return who_dies

    def get_Rfree(self):
        """
        Returns an array of size self.AgentCount with self.RfreeNow in every entry.

        Parameters
        ----------
        None

        Returns
        -------
        RfreeNow : np.array
             Array of size self.AgentCount with risk free interest rate for each agent.
        """
        RfreeNow = self.RfreeNow * np.ones(self.AgentCount)
        return RfreeNow

    def get_shocks(self):
        """
        Finds the effective permanent and transitory shocks this period by combining the aggregate
        and idiosyncratic shocks of each type.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        IndShockConsumerType.get_shocks(self)  # Update idiosyncratic shocks
        self.shocks["TranShk"] = (
            self.shocks["TranShk"] * self.TranShkAggNow * self.wRteNow
        )
        self.shocks["PermShk"] = self.shocks["PermShk"] * self.PermShkAggNow

    def get_controls(self):
        """
        Calculates consumption for each consumer of this type using the consumption functions.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        cNrmNow = np.zeros(self.AgentCount) + np.nan
        MPCnow = np.zeros(self.AgentCount) + np.nan
        MaggNow = self.get_MaggNow()
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            cNrmNow[these] = self.solution[t].cFunc(
                self.state_now["mNrm"][these], MaggNow[these]
            )
            MPCnow[these] = self.solution[t].cFunc.derivativeX(
                self.state_now["mNrm"][these], MaggNow[these]
            )  # Marginal propensity to consume

        self.controls["cNrm"] = cNrmNow
        self.MPCnow = MPCnow
        return None

    def get_MaggNow(self):  # This function exists to be overwritten in StickyE model
        return self.MaggNow * np.ones(self.AgentCount)

    def market_action(self):
        """
        In the aggregate shocks model, the "market action" is to simulate one
        period of receiving income and choosing how much to consume.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.simulate(1)

    def calc_bounding_values(self):
        """
        Calculate human wealth plus minimum and maximum MPC in an infinite
        horizon model with only one period repeated indefinitely.  Store results
        as attributes of self.  Human wealth is the present discounted value of
        expected future income after receiving income this period, ignoring mort-
        ality.  The maximum MPC is the limit of the MPC as m --> mNrmMin.  The
        minimum MPC is the limit of the MPC as m --> infty.

        NOT YET IMPLEMENTED FOR THIS CLASS

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        raise NotImplementedError()

    def make_euler_error_func(self, mMax=100, approx_inc_dstn=True):
        """
        Creates a "normalized Euler error" function for this instance, mapping
        from market resources to "consumption error per dollar of consumption."
        Stores result in attribute eulerErrorFunc as an interpolated function.
        Has option to use approximate income distribution stored in self.IncShkDstn
        or to use a (temporary) very dense approximation.

        NOT YET IMPLEMENTED FOR THIS CLASS

        Parameters
        ----------
        mMax : float
            Maximum normalized market resources for the Euler error function.
        approx_inc_dstn : Boolean
            Indicator for whether to use the approximate discrete income distri-
            bution stored in self.IncShkDstn[0], or to use a very accurate
            discrete approximation instead.  When True, uses approximation in
            IncShkDstn; when False, makes and uses a very dense approximation.

        Returns
        -------
        None

        Notes
        -----
        This method is not used by any other code in the library. Rather, it is here
        for expository and benchmarking purposes.

        """
        raise NotImplementedError()


# This example makes a high risk, low growth state and a low risk, high growth state
MrkvArray = np.array([[0.90, 0.10], [0.04, 0.96]])
PermShkAggStd = [
    0.012,
    0.006,
]  # Standard deviation of log aggregate permanent shocks by state
TranShkAggStd = [
    0.006,
    0.003,
]  # Standard deviation of log aggregate transitory shocks by state
PermGroFacAgg = [0.98, 1.02]  # Aggregate permanent income growth factor

# Make a dictionary to specify a Markov aggregate shocks consumer
init_agg_mrkv_shocks = init_agg_shocks.copy()
init_agg_mrkv_shocks["MrkvArray"] = MrkvArray


class AggShockMarkovConsumerType(AggShockConsumerType):
    """
    A class for representing ex ante heterogeneous "types" of consumers who
    experience both aggregate and idiosyncratic shocks to productivity (both
    permanent and transitory), who lives in an environment where the macroeconomic
    state is subject to Markov-style discrete state evolution.
    """

    def __init__(self, **kwds):
        params = init_agg_mrkv_shocks.copy()
        params.update(kwds)
        kwds = params
        AggShockConsumerType.__init__(self, **kwds)

        self.shocks["Mrkv"] = None

        self.add_to_time_inv("MrkvArray")
        self.solve_one_period = solve_ConsAggMarkov

    def add_AggShkDstn(self, AggShkDstn):
        """
        Variation on AggShockConsumerType.add_AggShkDstn that handles the Markov
        state. AggShkDstn is a list of aggregate productivity shock distributions
        for each Markov state.
        """
        if len(self.IncShkDstn[0][0].data) > 2:
            self.IncShkDstn = self.IncShkDstnWithoutAggShocks
        else:
            self.IncShkDstnWithoutAggShocks = self.IncShkDstn

        IncShkDstnOut = []
        N = self.MrkvArray.shape[0]
        for t in range(self.T_cycle):
            IncShkDstnOut.append(
                [
                    combine_indep_dstns(self.IncShkDstn[t][n], AggShkDstn[n])
                    for n in range(N)
                ]
            )
        self.IncShkDstn = IncShkDstnOut

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
        AggShockConsumerType.update_solution_terminal(self)

        # Make replicated terminal period solution
        StateCount = self.MrkvArray.shape[0]
        self.solution_terminal.cFunc = StateCount * [self.solution_terminal.cFunc]
        self.solution_terminal.vPfunc = StateCount * [self.solution_terminal.vPfunc]
        self.solution_terminal.mNrmMin = StateCount * [self.solution_terminal.mNrmMin]

    def reset_rng(self):
        MarkovConsumerType.reset_rng(self)

    def get_shocks(self):
        """
        Gets permanent and transitory income shocks for this period.  Samples from IncShkDstn for
        each period in the cycle.  This is a copy-paste from IndShockConsumerType, with the
        addition of the Markov macroeconomic state.  Unfortunately, the get_shocks method for
        MarkovConsumerType cannot be used, as that method assumes that MrkvNow is a vector
        with a value for each agent, not just a single int.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        PermShkNow = np.zeros(self.AgentCount)  # Initialize shock arrays
        TranShkNow = np.zeros(self.AgentCount)
        newborn = self.t_age == 0
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            N = np.sum(these)
            if N > 0:
                IncShkDstnNow = self.IncShkDstn[t - 1][
                    self.shocks["Mrkv"]
                ]  # set current income distribution
                PermGroFacNow = self.PermGroFac[t - 1]  # and permanent growth factor

                # Get random draws of income shocks from the discrete distribution
                ShockDraws = IncShkDstnNow.draw(N, exact_match=True)
                # Permanent "shock" includes expected growth
                PermShkNow[these] = ShockDraws[0] * PermGroFacNow
                TranShkNow[these] = ShockDraws[1]

        # That procedure used the *last* period in the sequence for newborns, but that's not right
        # Redraw shocks for newborns, using the *first* period in the sequence.  Approximation.
        N = np.sum(newborn)
        if N > 0:
            these = newborn
            IncShkDstnNow = self.IncShkDstn[0][
                self.shocks["Mrkv"]
            ]  # set current income distribution
            PermGroFacNow = self.PermGroFac[0]  # and permanent growth factor

            # Get random draws of income shocks from the discrete distribution
            ShockDraws = IncShkDstnNow.draw(N, exact_match=True)

            # Permanent "shock" includes expected growth
            PermShkNow[these] = ShockDraws[0] * PermGroFacNow
            TranShkNow[these] = ShockDraws[1]

        # Store the shocks in self
        self.EmpNow = np.ones(self.AgentCount, dtype=bool)
        self.EmpNow[TranShkNow == self.IncUnemp] = False
        self.shocks["TranShk"] = TranShkNow * self.TranShkAggNow * self.wRteNow
        self.shocks["PermShk"] = PermShkNow * self.PermShkAggNow

    def get_controls(self):
        """
        Calculates consumption for each consumer of this type using the consumption functions.
        For this AgentType class, MrkvNow is the same for all consumers.  However, in an
        extension with "macroeconomic inattention", consumers might misperceive the state
        and thus act as if they are in different states.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        cNrmNow = np.zeros(self.AgentCount) + np.nan
        MPCnow = np.zeros(self.AgentCount) + np.nan
        MaggNow = self.get_MaggNow()
        MrkvNow = self.getMrkvNow()

        StateCount = self.MrkvArray.shape[0]
        MrkvBoolArray = np.zeros((StateCount, self.AgentCount), dtype=bool)
        for i in range(StateCount):
            MrkvBoolArray[i, :] = i == MrkvNow

        for t in range(self.T_cycle):
            these = t == self.t_cycle
            for i in range(StateCount):
                those = np.logical_and(these, MrkvBoolArray[i, :])
                cNrmNow[those] = self.solution[t].cFunc[i](
                    self.state_now["mNrm"][those], MaggNow[those]
                )
                # Marginal propensity to consume
                MPCnow[those] = (
                    self.solution[t]
                    .cFunc[i]
                    .derivativeX(self.state_now["mNrm"][those], MaggNow[those])
                )
        self.controls["cNrm"] = cNrmNow
        self.MPCnow = MPCnow
        return None

    def getMrkvNow(self):  # This function exists to be overwritten in StickyE model
        return self.shocks["Mrkv"] * np.ones(self.AgentCount, dtype=int)


init_KS_agents = {
    "T_cycle": 1,
    "DiscFac": 0.99,
    "CRRA": 1.0,
    "LbrInd": 1.0,
    "aMin": 0.001,
    "aMax": 50.0,
    "aCount": 32,
    "aNestFac": 2,
    "MgridBase": np.array(
        [0.1, 0.3, 0.6, 0.8, 0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1, 1.2, 1.6, 2.0, 3.0]
    ),
    "AgentCount": 5000,
}


class KrusellSmithType(AgentType):
    """
    A class for representing agents in the seminal Krusell-Smith (1998) model from
    the paper "Income and Wealth Heterogeneity in the Macroeconomy".  All default
    parameters have been set to match those in the paper, but the equilibrium object
    is perceptions of aggregate assets as a function of aggregate market resources
    in each macroeconomic state (bad=0, good=1), rather than aggregate capital as
    a function of previous aggregate capital.  This choice was made so that some
    of the code from HARK's other HA-macro models can be used.
    """

    def __init__(self, **kwds):
        """
        Make a new instance of the Krusell-Smith type.
        """
        params = init_KS_agents.copy()
        params.update(kwds)

        AgentType.__init__(self, pseudo_terminal=False, **params)

        # Add consumer-type specific objects
        self.time_vary = []
        self.time_inv = [
            "DiscFac",
            "CRRA",
        ]
        # need better handling of this
        self.state_now = {"aNow": None, "mNow": None, "EmpNow": None}
        self.state_prev = {"aNow": None, "mNow": None, "EmpNow": None}

        self.shock_vars = {"Mrkv": None}

        self.solve_one_period = solve_KrusellSmith
        self.update()

    def pre_solve(self):
        self.update()
        self.precompute_arrays()

    def update(self):
        """
        Construct objects used during solution from primitive parameters.
        """
        self.make_grid()
        self.update_solution_terminal()

    def get_economy_data(self, Economy):
        """
        Imports economy-determined objects into self from a Market.

        Parameters
        ----------
        Economy : KrusellSmithEconomy
            The "macroeconomy" in which this instance "lives".

        Returns
        -------
        None
        """
        self.T_sim = (
            Economy.act_T
        )  # Need to be able to track as many periods as economy runs
        self.kInit = Economy.KSS  # Initialize simulation assets to steady state
        self.MrkvInit = Economy.sow_init[
            "Mrkv"
        ]  # Starting Markov state for the macroeconomy
        self.Mgrid = (
            Economy.MSS * self.MgridBase
        )  # Aggregate market resources grid adjusted around SS capital ratio
        self.AFunc = Economy.AFunc  # Next period's aggregate savings function
        self.DeprFac = Economy.DeprFac  # Rate of capital depreciation
        self.CapShare = Economy.CapShare  # Capital's share of production
        self.LbrInd = Economy.LbrInd  # Idiosyncratic labor supply (when employed)
        self.UrateB = Economy.UrateB  # Unemployment rate in bad state
        self.UrateG = Economy.UrateG  # Unemployment rate in good state
        self.ProdB = Economy.ProdB  # Total factor productivity in bad state
        self.ProdG = Economy.ProdG  # Total factor productivity in good state
        self.MrkvIndArray = (
            Economy.MrkvIndArray
        )  # Transition probabilities among discrete states
        self.MrkvAggArray = (
            Economy.MrkvArray
        )  # Transition probabilities among aggregate discrete states
        self.add_to_time_inv(
            "Mgrid",
            "AFunc",
            "DeprFac",
            "CapShare",
            "UrateB",
            "LbrInd",
            "UrateG",
            "ProdB",
            "ProdG",
            "MrkvIndArray",
            "MrkvAggArray",
        )

    def make_grid(self):
        """
        Construct the attribute aXtraGrid from the primitive attributes aMin,
        aMax, aCount, aNestFac.
        """
        self.aGrid = make_grid_exp_mult(
            self.aMin, self.aMax, self.aCount, self.aNestFac
        )
        self.add_to_time_inv("aGrid")

    def update_solution_terminal(self):
        """
        Construct the trivial terminal period solution (initial guess).
        """
        cFunc_terminal = 4 * [IdentityFunction(n_dims=2)]
        vPfunc_terminal = [
            MargValueFuncCRRA(cFunc_terminal[j], self.CRRA) for j in range(4)
        ]
        self.solution_terminal = ConsumerSolution(
            cFunc=cFunc_terminal, vPfunc=vPfunc_terminal
        )

    def precompute_arrays(self):
        """
        Construct the attributes ProbArray, mNextArray, MnextArray, and RnextArray,
        which will be used by the one period solver.
        """
        # Get array sizes
        aCount = self.aGrid.size
        Mcount = self.Mgrid.size

        # Make tiled array of end-of-period idiosyncratic assets (order: a, M, s, s')
        aNow_tiled = np.tile(
            np.reshape(self.aGrid, [aCount, 1, 1, 1]), [1, Mcount, 4, 4]
        )

        # Make arrays of end-of-period aggregate assets (capital next period)
        AnowB = self.AFunc[0](self.Mgrid)
        AnowG = self.AFunc[1](self.Mgrid)
        KnextB = np.tile(np.reshape(AnowB, [1, Mcount, 1, 1]), [1, 1, 1, 4])
        KnextG = np.tile(np.reshape(AnowG, [1, Mcount, 1, 1]), [1, 1, 1, 4])
        Knext = np.concatenate((KnextB, KnextB, KnextG, KnextG), axis=2)

        # Make arrays of aggregate labor and TFP next period
        Lnext = np.zeros((1, Mcount, 4, 4))  # shape (1,Mcount,4,4)
        Lnext[0, :, :, 0:2] = (1.0 - self.UrateB) * self.LbrInd
        Lnext[0, :, :, 2:4] = (1.0 - self.UrateG) * self.LbrInd
        Znext = np.zeros((1, Mcount, 4, 4))
        Znext[0, :, :, 0:2] = self.ProdB
        Znext[0, :, :, 2:4] = self.ProdG

        # Calculate (net) interest factor and wage rate next period
        KtoLnext = Knext / Lnext
        Rnext = 1.0 + Znext * CapShare * KtoLnext ** (CapShare - 1.0) - DeprFac
        Wnext = Znext * (1.0 - CapShare) * KtoLnext**CapShare

        # Calculate aggregate market resources next period
        Ynext = Znext * Knext**CapShare * Lnext ** (1.0 - CapShare)
        Mnext = (1.0 - DeprFac) * Knext + Ynext

        # Tile the interest, wage, and aggregate market resources arrays
        Rnext_tiled = np.tile(Rnext, [aCount, 1, 1, 1])
        Wnext_tiled = np.tile(Wnext, [aCount, 1, 1, 1])
        Mnext_tiled = np.tile(Mnext, [aCount, 1, 1, 1])

        # Make an array of idiosyncratic labor supply next period
        lNext_tiled = np.zeros([aCount, Mcount, 4, 4])
        lNext_tiled[:, :, :, 1] = self.LbrInd
        lNext_tiled[:, :, :, 3] = self.LbrInd

        # Calculate idiosyncratic market resources next period
        mNext = Rnext_tiled * aNow_tiled + Wnext_tiled * lNext_tiled

        # Make a tiled array of transition probabilities
        Probs_tiled = np.tile(
            np.reshape(self.MrkvIndArray, [1, 1, 4, 4]), [aCount, Mcount, 1, 1]
        )

        # Store the attributes that will be used by the solver
        self.ProbArray = Probs_tiled
        self.mNextArray = mNext
        self.MnextArray = Mnext_tiled
        self.RnextArray = Rnext_tiled
        self.add_to_time_inv("ProbArray", "mNextArray", "MnextArray", "RnextArray")

    def make_emp_idx_arrays(self):
        """
        Construct the attributes emp_permute and unemp_permute, each of which is
        a 2x2 nested list of boolean arrays.  The j,k-th element of emp_permute
        represents the employment states this period for agents who were employed
        last period when the macroeconomy is transitioning from state j to state k.
        Likewise, j,k-th element of unemp_permute represents the employment states
        this period for agents who were unemployed last period when the macro-
        economy is transitioning from state j to state k.  These attributes are
        referenced during simulation, when they are randomly permuted in order to
        maintain exact unemployment rates in each period.
        """
        # Get counts of employed and unemployed agents in each macroeconomic state
        B_unemp_N = int(np.round(self.UrateB * self.AgentCount))
        B_emp_N = self.AgentCount - B_unemp_N
        G_unemp_N = int(np.round(self.UrateG * self.AgentCount))
        G_emp_N = self.AgentCount - G_unemp_N

        # Bad-bad transition indices
        BB_stay_unemp_N = int(
            np.round(B_unemp_N * self.MrkvIndArray[0, 0] / self.MrkvAggArray[0, 0])
        )
        BB_become_unemp_N = B_unemp_N - BB_stay_unemp_N
        BB_stay_emp_N = int(
            np.round(B_emp_N * self.MrkvIndArray[1, 1] / self.MrkvAggArray[0, 0])
        )
        BB_become_emp_N = B_emp_N - BB_stay_emp_N
        BB_unemp_permute = np.concatenate(
            [
                np.ones(BB_become_emp_N, dtype=bool),
                np.zeros(BB_stay_unemp_N, dtype=bool),
            ]
        )
        BB_emp_permute = np.concatenate(
            [
                np.ones(BB_stay_emp_N, dtype=bool),
                np.zeros(BB_become_unemp_N, dtype=bool),
            ]
        )

        # Bad-good transition indices
        BG_stay_unemp_N = int(
            np.round(B_unemp_N * self.MrkvIndArray[0, 2] / self.MrkvAggArray[0, 1])
        )
        BG_become_unemp_N = G_unemp_N - BG_stay_unemp_N
        BG_stay_emp_N = int(
            np.round(B_emp_N * self.MrkvIndArray[1, 3] / self.MrkvAggArray[0, 1])
        )
        BG_become_emp_N = G_emp_N - BG_stay_emp_N
        BG_unemp_permute = np.concatenate(
            [
                np.ones(BG_become_emp_N, dtype=bool),
                np.zeros(BG_stay_unemp_N, dtype=bool),
            ]
        )
        BG_emp_permute = np.concatenate(
            [
                np.ones(BG_stay_emp_N, dtype=bool),
                np.zeros(BG_become_unemp_N, dtype=bool),
            ]
        )

        # Good-bad transition indices
        GB_stay_unemp_N = int(
            np.round(G_unemp_N * self.MrkvIndArray[2, 0] / self.MrkvAggArray[1, 0])
        )
        GB_become_unemp_N = B_unemp_N - GB_stay_unemp_N
        GB_stay_emp_N = int(
            np.round(G_emp_N * self.MrkvIndArray[3, 1] / self.MrkvAggArray[1, 0])
        )
        GB_become_emp_N = B_emp_N - GB_stay_emp_N
        GB_unemp_permute = np.concatenate(
            [
                np.ones(GB_become_emp_N, dtype=bool),
                np.zeros(GB_stay_unemp_N, dtype=bool),
            ]
        )
        GB_emp_permute = np.concatenate(
            [
                np.ones(GB_stay_emp_N, dtype=bool),
                np.zeros(GB_become_unemp_N, dtype=bool),
            ]
        )

        # Good-good transition indices
        GG_stay_unemp_N = int(
            np.round(G_unemp_N * self.MrkvIndArray[2, 2] / self.MrkvAggArray[1, 1])
        )
        GG_become_unemp_N = G_unemp_N - GG_stay_unemp_N
        GG_stay_emp_N = int(
            np.round(G_emp_N * self.MrkvIndArray[3, 3] / self.MrkvAggArray[1, 1])
        )
        GG_become_emp_N = G_emp_N - GG_stay_emp_N
        GG_unemp_permute = np.concatenate(
            [
                np.ones(GG_become_emp_N, dtype=bool),
                np.zeros(GG_stay_unemp_N, dtype=bool),
            ]
        )
        GG_emp_permute = np.concatenate(
            [
                np.ones(GG_stay_emp_N, dtype=bool),
                np.zeros(GG_become_unemp_N, dtype=bool),
            ]
        )

        # Store transition matrices as attributes of self
        self.unemp_permute = [
            [BB_unemp_permute, BG_unemp_permute],
            [GB_unemp_permute, GG_unemp_permute],
        ]
        self.emp_permute = [
            [BB_emp_permute, BG_emp_permute],
            [GB_emp_permute, GG_emp_permute],
        ]

    def reset(self):
        self.initialize_sim()

    def market_action(self):
        self.simulate(1)

    def initialize_sim(self):
        self.shocks["Mrkv"] = self.MrkvInit
        AgentType.initialize_sim(self)
        self.state_now["EmpNow"] = self.state_now["EmpNow"].astype(bool)
        self.make_emp_idx_arrays()

    def sim_birth(self, which):
        """
        Create newborn agents with randomly drawn employment states.  This will
        only ever be called by initialize_sim() at the start of a new simulation
        history, as the Krusell-Smith model does not have death and replacement.
        The sim_death() method does not exist, as AgentType's default of "no death"
        is the correct behavior for the model.
        """
        N = np.sum(which)
        if N == 0:
            return

        if self.shocks["Mrkv"] == 0:
            unemp_N = int(np.round(self.UrateB * N))
            emp_N = self.AgentCount - unemp_N
        elif self.shocks["Mrkv"] == 1:
            unemp_N = int(np.round(self.UrateG * N))
            emp_N = self.AgentCount - unemp_N
        else:
            assert False, "Illegal macroeconomic state: MrkvNow must be 0 or 1"
        EmpNew = np.concatenate(
            [np.zeros(unemp_N, dtype=bool), np.ones(emp_N, dtype=bool)]
        )

        self.state_now["EmpNow"][which] = self.RNG.permutation(EmpNew)
        self.state_now["aNow"][which] = self.kInit

    def get_shocks(self):
        """
        Get new idiosyncratic employment states based on the macroeconomic state.
        """
        # Get boolean arrays for current employment states
        employed = self.state_prev["EmpNow"].copy().astype(bool)
        unemployed = np.logical_not(employed)

        # derive from past employment rate rather than store previous value
        mrkv_prev = int((unemployed.sum() / float(self.AgentCount)) != self.UrateB)

        # Transition some agents between unemployment and employment
        emp_permute = self.emp_permute[mrkv_prev][self.shocks["Mrkv"]]
        unemp_permute = self.unemp_permute[mrkv_prev][self.shocks["Mrkv"]]
        # TODO: replace poststate_vars functionality with shocks here
        EmpNow = self.state_now["EmpNow"]

        # It's really this permutation that is the shock...
        # This apparatus is trying to 'exact match' the 'internal' Markov process.
        EmpNow[employed] = self.RNG.permutation(emp_permute)
        EmpNow[unemployed] = self.RNG.permutation(unemp_permute)

    def get_states(self):
        """
        Get each agent's idiosyncratic state, their household market resources.
        """
        self.state_now["mNow"] = (
            self.Rnow * self.state_prev["aNow"]
            + self.Wnow * self.LbrInd * self.state_now["EmpNow"]
        )

    def get_controls(self):
        """
        Get each agent's consumption given their current state.'
        """
        employed = self.state_now["EmpNow"].copy().astype(bool)
        unemployed = np.logical_not(employed)

        # Get the discrete index for (un)employed agents
        if self.shocks["Mrkv"] == 0:  # Bad macroeconomic conditions
            unemp_idx = 0
            emp_idx = 1
        elif self.shocks["Mrkv"] == 1:  # Good macroeconomic conditions
            unemp_idx = 2
            emp_idx = 3
        else:
            assert False, "Illegal macroeconomic state: MrkvNow must be 0 or 1"

        # Get consumption for each agent using the appropriate consumption function
        cNow = np.zeros(self.AgentCount)
        Mnow = self.Mnow * np.ones(self.AgentCount)
        cNow[unemployed] = self.solution[0].cFunc[unemp_idx](
            self.state_now["mNow"][unemployed], Mnow[unemployed]
        )
        cNow[employed] = self.solution[0].cFunc[emp_idx](
            self.state_now["mNow"][employed], Mnow[employed]
        )
        self.controls["cNow"] = cNow

    def get_poststates(self):
        """
        Gets each agent's retained assets after consumption.
        """
        self.state_now["aNow"] = self.state_now["mNow"] - self.controls["cNow"]


###############################################################################


def solveConsAggShock(
    solution_next,
    IncShkDstn,
    LivPrb,
    DiscFac,
    CRRA,
    PermGroFac,
    PermGroFacAgg,
    aXtraGrid,
    BoroCnstArt,
    Mgrid,
    AFunc,
    Rfunc,
    wFunc,
):
    """
    Solve one period of a consumption-saving problem with idiosyncratic and
    aggregate shocks (transitory and permanent).  This is a basic solver that
    can't handle cubic splines, nor can it calculate a value function.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to the succeeding one period problem.
    IncShkDstn : distribution.Distribution
        A discrete
        approximation to the income process between the period being solved
        and the one immediately following (in solution_next). Order:
        idiosyncratic permanent shocks, idiosyncratic transitory
        shocks, aggregate permanent shocks, aggregate transitory shocks.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
    PermGroFac : float
        Expected permanent income growth factor at the end of this period.
    PermGroFacAgg : float
        Expected aggregate productivity growth factor.
    aXtraGrid : np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.
    BoroCnstArt : float
        Artificial borrowing constraint; minimum allowable end-of-period asset-to-
        permanent-income ratio.  Unlike other models, this *can't* be None.
    Mgrid : np.array
        A grid of aggregate market resourses to permanent income in the economy.
    AFunc : function
        Aggregate savings as a function of aggregate market resources.
    Rfunc : function
        The net interest factor on assets as a function of capital ratio k.
    wFunc : function
        The wage rate for labor as a function of capital-to-labor ratio k.
    DeprFac : float
        Capital Depreciation Rate

    Returns
    -------
    solution_now : ConsumerSolution
        The solution to the single period consumption-saving problem.  Includes
        a consumption function cFunc (linear interpolation over linear interpola-
        tions) and marginal value function vPfunc.
    """
    # Unpack next period's solution
    vPfuncNext = solution_next.vPfunc
    mNrmMinNext = solution_next.mNrmMin

    # Unpack the income shocks
    ShkPrbsNext = IncShkDstn.prob
    PermShkValsNext = IncShkDstn.data[0]
    TranShkValsNext = IncShkDstn.data[1]
    PermShkAggValsNext = IncShkDstn.data[2]
    TranShkAggValsNext = IncShkDstn.data[3]
    ShkCount = ShkPrbsNext.size

    # Make the grid of end-of-period asset values, and a tiled version
    aNrmNow = aXtraGrid
    aCount = aNrmNow.size
    Mcount = Mgrid.size
    aXtra_tiled = np.tile(np.reshape(aNrmNow, (1, aCount, 1)), (Mcount, 1, ShkCount))

    # Make tiled versions of the income shocks
    # Dimension order: Mnow, aNow, Shk
    ShkPrbsNext_tiled = np.tile(
        np.reshape(ShkPrbsNext, (1, 1, ShkCount)), (Mcount, aCount, 1)
    )
    PermShkValsNext_tiled = np.tile(
        np.reshape(PermShkValsNext, (1, 1, ShkCount)), (Mcount, aCount, 1)
    )
    TranShkValsNext_tiled = np.tile(
        np.reshape(TranShkValsNext, (1, 1, ShkCount)), (Mcount, aCount, 1)
    )
    PermShkAggValsNext_tiled = np.tile(
        np.reshape(PermShkAggValsNext, (1, 1, ShkCount)), (Mcount, aCount, 1)
    )
    TranShkAggValsNext_tiled = np.tile(
        np.reshape(TranShkAggValsNext, (1, 1, ShkCount)), (Mcount, aCount, 1)
    )

    # Calculate returns to capital and labor in the next period
    AaggNow_tiled = np.tile(
        np.reshape(AFunc(Mgrid), (Mcount, 1, 1)), (1, aCount, ShkCount)
    )
    kNext_array = AaggNow_tiled / (
        PermGroFacAgg * PermShkAggValsNext_tiled
    )  # Next period's aggregate capital/labor ratio
    kNextEff_array = (
        kNext_array / TranShkAggValsNext_tiled
    )  # Same thing, but account for *transitory* shock
    R_array = Rfunc(kNextEff_array)  # Interest factor on aggregate assets
    Reff_array = (
        R_array / LivPrb
    )  # Effective interest factor on individual assets *for survivors*
    wEff_array = (
        wFunc(kNextEff_array) * TranShkAggValsNext_tiled
    )  # Effective wage rate (accounts for labor supply)
    PermShkTotal_array = (
        PermGroFac * PermGroFacAgg * PermShkValsNext_tiled * PermShkAggValsNext_tiled
    )  # total / combined permanent shock
    Mnext_array = (
        kNext_array * R_array + wEff_array
    )  # next period's aggregate market resources

    # Find the natural borrowing constraint for each value of M in the Mgrid.
    # There is likely a faster way to do this, but someone needs to do the math:
    # is aNrmMin determined by getting the worst shock of all four types?
    aNrmMin_candidates = (
        PermGroFac
        * PermGroFacAgg
        * PermShkValsNext_tiled[:, 0, :]
        * PermShkAggValsNext_tiled[:, 0, :]
        / Reff_array[:, 0, :]
        * (
            mNrmMinNext(Mnext_array[:, 0, :])
            - wEff_array[:, 0, :] * TranShkValsNext_tiled[:, 0, :]
        )
    )
    aNrmMin_vec = np.max(aNrmMin_candidates, axis=1)
    BoroCnstNat_vec = aNrmMin_vec
    aNrmMin_tiled = np.tile(
        np.reshape(aNrmMin_vec, (Mcount, 1, 1)), (1, aCount, ShkCount)
    )
    aNrmNow_tiled = aNrmMin_tiled + aXtra_tiled

    # Calculate market resources next period (and a constant array of capital-to-labor ratio)
    mNrmNext_array = (
        Reff_array * aNrmNow_tiled / PermShkTotal_array
        + TranShkValsNext_tiled * wEff_array
    )

    # Find marginal value next period at every income shock realization and every aggregate market resource gridpoint
    vPnext_array = (
        Reff_array
        * PermShkTotal_array ** (-CRRA)
        * vPfuncNext(mNrmNext_array, Mnext_array)
    )

    # Calculate expectated marginal value at the end of the period at every asset gridpoint
    EndOfPrdvP = DiscFac * LivPrb * np.sum(vPnext_array * ShkPrbsNext_tiled, axis=2)

    # Calculate optimal consumption from each asset gridpoint
    cNrmNow = EndOfPrdvP ** (-1.0 / CRRA)
    mNrmNow = aNrmNow_tiled[:, :, 0] + cNrmNow

    # Loop through the values in Mgrid and make a linear consumption function for each
    cFuncBaseByM_list = []
    for j in range(Mcount):
        c_temp = np.insert(cNrmNow[j, :], 0, 0.0)  # Add point at bottom
        m_temp = np.insert(mNrmNow[j, :] - BoroCnstNat_vec[j], 0, 0.0)
        cFuncBaseByM_list.append(LinearInterp(m_temp, c_temp))
        # Add the M-specific consumption function to the list

    # Construct the overall unconstrained consumption function by combining the M-specific functions
    BoroCnstNat = LinearInterp(
        np.insert(Mgrid, 0, 0.0), np.insert(BoroCnstNat_vec, 0, 0.0)
    )
    cFuncBase = LinearInterpOnInterp1D(cFuncBaseByM_list, Mgrid)
    cFuncUnc = VariableLowerBoundFunc2D(cFuncBase, BoroCnstNat)

    # Make the constrained consumption function and combine it with the unconstrained component
    cFuncCnst = BilinearInterp(
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array([BoroCnstArt, BoroCnstArt + 1.0]),
        np.array([0.0, 1.0]),
    )
    cFuncNow = LowerEnvelope2D(cFuncUnc, cFuncCnst)

    # Make the minimum m function as the greater of the natural and artificial constraints
    mNrmMinNow = UpperEnvelope(BoroCnstNat, ConstantFunction(BoroCnstArt))

    # Construct the marginal value function using the envelope condition
    vPfuncNow = MargValueFuncCRRA(cFuncNow, CRRA)

    # Pack up and return the solution
    solution_now = ConsumerSolution(
        cFunc=cFuncNow, vPfunc=vPfuncNow, mNrmMin=mNrmMinNow
    )
    return solution_now


def solve_ConsAggShock_new(
    solution_next,
    IncShkDstn,
    LivPrb,
    DiscFac,
    CRRA,
    PermGroFac,
    PermGroFacAgg,
    aXtraGrid,
    BoroCnstArt,
    Mgrid,
    AFunc,
    Rfunc,
    wFunc,
    DeprFac,
):
    """
    Solve one period of a consumption-saving problem with idiosyncratic and
    aggregate shocks (transitory and permanent).  This is a basic solver that
    can't handle cubic splines, nor can it calculate a value function. This
    version uses calc_expectation to reduce code clutter.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to the succeeding one period problem.
    IncShkDstn : distribution.Distribution
        A discrete
        approximation to the income process between the period being solved
        and the one immediately following (in solution_next). Order:
        idiosyncratic permanent shocks, idiosyncratic transitory
        shocks, aggregate permanent shocks, aggregate transitory shocks.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
    PermGroFac : float
        Expected permanent income growth factor at the end of this period.
    PermGroFacAgg : float
        Expected aggregate productivity growth factor.
    aXtraGrid : np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.
    BoroCnstArt : float
        Artificial borrowing constraint; minimum allowable end-of-period asset-to-
        permanent-income ratio.  Unlike other models, this *can't* be None.
    Mgrid : np.array
        A grid of aggregate market resourses to permanent income in the economy.
    AFunc : function
        Aggregate savings as a function of aggregate market resources.
    Rfunc : function
        The net interest factor on assets as a function of capital ratio k.
    wFunc : function
        The wage rate for labor as a function of capital-to-labor ratio k.
    DeprFac : float
        Capital Depreciation Rate

    Returns
    -------
    solution_now : ConsumerSolution
        The solution to the single period consumption-saving problem.  Includes
        a consumption function cFunc (linear interpolation over linear interpola-
        tions) and marginal value function vPfunc.
    """
    # Unpack the income shocks and get grid sizes
    PermShkValsNext = IncShkDstn.data[0]
    TranShkValsNext = IncShkDstn.data[1]
    PermShkAggValsNext = IncShkDstn.data[2]
    TranShkAggValsNext = IncShkDstn.data[3]
    aCount = aXtraGrid.size
    Mcount = Mgrid.size

    # Define a function that calculates M_{t+1} from M_t and the aggregate shocks;
    # the function also returns the wage rate and effective interest factor
    def calcAggObjects(M, Psi, Theta):
        A = AFunc(M)  # End-of-period aggregate assets (normalized)
        kNext = A / (PermGroFacAgg * Psi)  # Next period's aggregate capital/labor ratio
        kNextEff = kNext / Theta  # Same thing, but account for *transitory* shock
        R = Rfunc(kNextEff)  # Interest factor on aggregate assets
        wEff = (
            wFunc(kNextEff) * Theta
        )  # Effective wage rate (accounts for labor supply)
        Reff = R / LivPrb  # Account for redistribution of decedents' wealth
        Mnext = kNext * R + wEff  # Next period's aggregate market resources
        return Mnext, Reff, wEff

    # Define a function that evaluates R*v'(m_{t+1},M_{t+1}) from a_t, M_t, and the income shocks
    def vPnextFunc(a, M, psi, theta, Psi, Theta):
        Mnext, Reff, wEff = calcAggObjects(M, Psi, Theta)
        PermShkTotal = (
            PermGroFac * PermGroFacAgg * psi * Psi
        )  # Total / combined permanent shock
        mNext = Reff * a / PermShkTotal + theta * wEff  # Idiosyncratic market resources
        vPnext = Reff * PermShkTotal ** (-CRRA) * solution_next.vPfunc(mNext, Mnext)
        return vPnext

    # Make an array of a_t values at which to calculate end-of-period marginal value of assets
    BoroCnstNat_vec = np.zeros(Mcount)  # Natural borrowing constraint at each M_t
    aNrmNow = np.zeros((aCount, Mcount))
    for j in range(Mcount):
        Mnext, Reff, wEff = calcAggObjects(
            Mgrid[j], PermShkAggValsNext, TranShkAggValsNext
        )
        aNrmMin_cand = (
            PermGroFac * PermGroFacAgg * PermShkValsNext * PermShkAggValsNext / Reff
        ) * (solution_next.mNrmMin(Mnext) - wEff * TranShkValsNext)
        aNrmMin = np.max(aNrmMin_cand)  # Lowest valid a_t value for this M_t
        aNrmNow[:, j] = aNrmMin + aXtraGrid
        BoroCnstNat_vec[j] = aNrmMin

    # Compute end-of-period marginal value of assets
    MaggNow = np.tile(np.reshape(Mgrid, (1, Mcount)), (aCount, 1))  # Tiled Mgrid
    EndOfPrdvP = (
        DiscFac * LivPrb * calc_expectation(IncShkDstn, vPnextFunc, [aNrmNow, MaggNow])
    )

    # Calculate optimal consumption from each asset gridpoint and endogenous m_t gridpoint
    cNrmNow = EndOfPrdvP ** (-1.0 / CRRA)
    mNrmNow = aNrmNow + cNrmNow

    # Loop through the values in Mgrid and make a linear consumption function for each
    cFuncBaseByM_list = []
    for j in range(Mcount):
        c_temp = np.insert(cNrmNow[:, j], 0, 0.0)  # Add point at bottom
        m_temp = np.insert(mNrmNow[:, j] - BoroCnstNat_vec[j], 0, 0.0)
        cFuncBaseByM_list.append(LinearInterp(m_temp, c_temp))

    # Construct the overall unconstrained consumption function by combining the M-specific functions
    BoroCnstNat = LinearInterp(
        np.insert(Mgrid, 0, 0.0), np.insert(BoroCnstNat_vec, 0, 0.0)
    )
    cFuncBase = LinearInterpOnInterp1D(cFuncBaseByM_list, Mgrid)
    cFuncUnc = VariableLowerBoundFunc2D(cFuncBase, BoroCnstNat)

    # Make the constrained consumption function and combine it with the unconstrained component
    cFuncCnst = BilinearInterp(
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array([BoroCnstArt, BoroCnstArt + 1.0]),
        np.array([0.0, 1.0]),
    )
    cFuncNow = LowerEnvelope2D(cFuncUnc, cFuncCnst)

    # Make the minimum m function as the greater of the natural and artificial constraints
    mNrmMinNow = UpperEnvelope(BoroCnstNat, ConstantFunction(BoroCnstArt))

    # Construct the marginal value function using the envelope condition
    vPfuncNow = MargValueFunc2D(cFuncNow, CRRA)

    # Pack up and return the solution
    solution_now = ConsumerSolution(
        cFunc=cFuncNow, vPfunc=vPfuncNow, mNrmMin=mNrmMinNow
    )
    return solution_now


###############################################################################


def solve_ConsAggMarkov(
    solution_next,
    IncShkDstn,
    LivPrb,
    DiscFac,
    CRRA,
    MrkvArray,
    PermGroFac,
    PermGroFacAgg,
    aXtraGrid,
    BoroCnstArt,
    Mgrid,
    AFunc,
    Rfunc,
    wFunc,
):
    """
    Solve one period of a consumption-saving problem with idiosyncratic and
    aggregate shocks (transitory and permanent).  Moreover, the macroeconomic
    state follows a Markov process that determines the income distribution and
    aggregate permanent growth factor. This is a basic solver that can't handle
    cubic splines, nor can it calculate a value function.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to the succeeding one period problem.
    IncShkDstn : [distribution.Distribution]
        A list of
        discrete approximations to the income process between the period being
        solved and the one immediately following (in solution_next). Order:
        idisyncratic permanent shocks, idiosyncratic transitory
        shocks, aggregate permanent shocks, aggregate transitory shocks.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
    MrkvArray : np.array
        Markov transition matrix between discrete macroeconomic states.
        MrkvArray[i,j] is probability of being in state j next period conditional
        on being in state i this period.
    PermGroFac : float
        Expected permanent income growth factor at the end of this period,
        for the *individual*'s productivity.
    PermGroFacAgg : [float]
        Expected aggregate productivity growth in each Markov macro state.
    aXtraGrid : np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.
    BoroCnstArt : float
        Artificial borrowing constraint; minimum allowable end-of-period asset-to-
        permanent-income ratio.  Unlike other models, this *can't* be None.
    Mgrid : np.array
        A grid of aggregate market resourses to permanent income in the economy.
    AFunc : [function]
        Aggregate savings as a function of aggregate market resources, for each
        Markov macro state.
    Rfunc : function
        The net interest factor on assets as a function of capital ratio k.
    wFunc : function
        The wage rate for labor as a function of capital-to-labor ratio k.
    DeprFac : float
        Capital Depreciation Rate

    Returns
    -------
    solution_now : ConsumerSolution
        The solution to the single period consumption-saving problem.  Includes
        a consumption function cFunc (linear interpolation over linear interpola-
        tions) and marginal value function vPfunc.
    """
    # Get sizes of grids
    aCount = aXtraGrid.size
    Mcount = Mgrid.size
    StateCount = MrkvArray.shape[0]

    # Loop through next period's states, assuming we reach each one at a time.
    # Construct EndOfPrdvP_cond functions for each state.
    EndOfPrdvPfunc_cond = []
    BoroCnstNat_cond = []
    for j in range(StateCount):
        # Unpack next period's solution
        vPfuncNext = solution_next.vPfunc[j]
        mNrmMinNext = solution_next.mNrmMin[j]

        # Unpack the income shocks
        ShkPrbsNext = IncShkDstn[j].prob
        PermShkValsNext = IncShkDstn[j].data[0]
        TranShkValsNext = IncShkDstn[j].data[1]
        PermShkAggValsNext = IncShkDstn[j].data[2]
        TranShkAggValsNext = IncShkDstn[j].data[3]
        ShkCount = ShkPrbsNext.size
        aXtra_tiled = np.tile(
            np.reshape(aXtraGrid, (1, aCount, 1)), (Mcount, 1, ShkCount)
        )

        # Make tiled versions of the income shocks
        # Dimension order: Mnow, aNow, Shk
        ShkPrbsNext_tiled = np.tile(
            np.reshape(ShkPrbsNext, (1, 1, ShkCount)), (Mcount, aCount, 1)
        )
        PermShkValsNext_tiled = np.tile(
            np.reshape(PermShkValsNext, (1, 1, ShkCount)), (Mcount, aCount, 1)
        )
        TranShkValsNext_tiled = np.tile(
            np.reshape(TranShkValsNext, (1, 1, ShkCount)), (Mcount, aCount, 1)
        )
        PermShkAggValsNext_tiled = np.tile(
            np.reshape(PermShkAggValsNext, (1, 1, ShkCount)), (Mcount, aCount, 1)
        )
        TranShkAggValsNext_tiled = np.tile(
            np.reshape(TranShkAggValsNext, (1, 1, ShkCount)), (Mcount, aCount, 1)
        )

        # Make a tiled grid of end-of-period aggregate assets.  These lines use
        # next prd state j's aggregate saving rule to get a relevant set of Aagg,
        # which will be used to make an interpolated EndOfPrdvP_cond function.
        # After constructing these functions, we will use the aggregate saving
        # rule for *current* state i to get values of Aagg at which to evaluate
        # these conditional marginal value functions.  In the strange, maybe even
        # impossible case where the aggregate saving rules differ wildly across
        # macro states *and* there is "anti-persistence", so that the macro state
        # is very likely to change each period, then this procedure will lead to
        # an inaccurate solution because the grid of Aagg values on which the
        # conditional marginal value functions are constructed is not relevant
        # to the values at which it will actually be evaluated.
        AaggGrid = AFunc[j](Mgrid)
        AaggNow_tiled = np.tile(
            np.reshape(AaggGrid, (Mcount, 1, 1)), (1, aCount, ShkCount)
        )

        # Calculate returns to capital and labor in the next period
        kNext_array = AaggNow_tiled / (
            PermGroFacAgg[j] * PermShkAggValsNext_tiled
        )  # Next period's aggregate capital to labor ratio
        kNextEff_array = (
            kNext_array / TranShkAggValsNext_tiled
        )  # Same thing, but account for *transitory* shock
        R_array = Rfunc(kNextEff_array)  # Interest factor on aggregate assets
        Reff_array = (
            R_array / LivPrb
        )  # Effective interest factor on individual assets *for survivors*
        wEff_array = (
            wFunc(kNextEff_array) * TranShkAggValsNext_tiled
        )  # Effective wage rate (accounts for labor supply)
        PermShkTotal_array = (
            PermGroFac
            * PermGroFacAgg[j]
            * PermShkValsNext_tiled
            * PermShkAggValsNext_tiled
        )  # total / combined permanent shock
        Mnext_array = (
            kNext_array * R_array + wEff_array
        )  # next period's aggregate market resources

        # Find the natural borrowing constraint for each value of M in the Mgrid.
        # There is likely a faster way to do this, but someone needs to do the math:
        # is aNrmMin determined by getting the worst shock of all four types?
        aNrmMin_candidates = (
            PermGroFac
            * PermGroFacAgg[j]
            * PermShkValsNext_tiled[:, 0, :]
            * PermShkAggValsNext_tiled[:, 0, :]
            / Reff_array[:, 0, :]
            * (
                mNrmMinNext(Mnext_array[:, 0, :])
                - wEff_array[:, 0, :] * TranShkValsNext_tiled[:, 0, :]
            )
        )
        aNrmMin_vec = np.max(aNrmMin_candidates, axis=1)
        BoroCnstNat_vec = aNrmMin_vec
        aNrmMin_tiled = np.tile(
            np.reshape(aNrmMin_vec, (Mcount, 1, 1)), (1, aCount, ShkCount)
        )
        aNrmNow_tiled = aNrmMin_tiled + aXtra_tiled

        # Calculate market resources next period (and a constant array of capital-to-labor ratio)
        mNrmNext_array = (
            Reff_array * aNrmNow_tiled / PermShkTotal_array
            + TranShkValsNext_tiled * wEff_array
        )

        # Find marginal value next period at every income shock
        # realization and every aggregate market resource gridpoint
        vPnext_array = (
            Reff_array
            * PermShkTotal_array ** (-CRRA)
            * vPfuncNext(mNrmNext_array, Mnext_array)
        )

        # Calculate expectated marginal value at the end of the period at every asset gridpoint
        EndOfPrdvP = DiscFac * LivPrb * np.sum(vPnext_array * ShkPrbsNext_tiled, axis=2)

        # Make the conditional end-of-period marginal value function
        BoroCnstNat = LinearInterp(
            np.insert(AaggGrid, 0, 0.0), np.insert(BoroCnstNat_vec, 0, 0.0)
        )
        EndOfPrdvPnvrs = np.concatenate(
            (np.zeros((Mcount, 1)), EndOfPrdvP ** (-1.0 / CRRA)), axis=1
        )
        EndOfPrdvPnvrsFunc_base = BilinearInterp(
            np.transpose(EndOfPrdvPnvrs), np.insert(aXtraGrid, 0, 0.0), AaggGrid
        )
        EndOfPrdvPnvrsFunc = VariableLowerBoundFunc2D(
            EndOfPrdvPnvrsFunc_base, BoroCnstNat
        )
        EndOfPrdvPfunc_cond.append(MargValueFuncCRRA(EndOfPrdvPnvrsFunc, CRRA))
        BoroCnstNat_cond.append(BoroCnstNat)

    # Prepare some objects that are the same across all current states
    aXtra_tiled = np.tile(np.reshape(aXtraGrid, (1, aCount)), (Mcount, 1))
    cFuncCnst = BilinearInterp(
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array([BoroCnstArt, BoroCnstArt + 1.0]),
        np.array([0.0, 1.0]),
    )

    # Now loop through *this* period's discrete states, calculating end-of-period
    # marginal value (weighting across state transitions), then construct consumption
    # and marginal value function for each state.
    cFuncNow = []
    vPfuncNow = []
    mNrmMinNow = []
    for i in range(StateCount):
        # Find natural borrowing constraint for this state by Aagg
        AaggNow = AFunc[i](Mgrid)
        aNrmMin_candidates = np.zeros((StateCount, Mcount)) + np.nan
        for j in range(StateCount):
            if MrkvArray[i, j] > 0.0:  # Irrelevant if transition is impossible
                aNrmMin_candidates[j, :] = BoroCnstNat_cond[j](AaggNow)
        aNrmMin_vec = np.nanmax(aNrmMin_candidates, axis=0)
        BoroCnstNat_vec = aNrmMin_vec

        # Make tiled grids of aNrm and Aagg
        aNrmMin_tiled = np.tile(np.reshape(aNrmMin_vec, (Mcount, 1)), (1, aCount))
        aNrmNow_tiled = aNrmMin_tiled + aXtra_tiled
        AaggNow_tiled = np.tile(np.reshape(AaggNow, (Mcount, 1)), (1, aCount))

        # Loop through feasible transitions and calculate end-of-period marginal value
        EndOfPrdvP = np.zeros((Mcount, aCount))
        for j in range(StateCount):
            if MrkvArray[i, j] > 0.0:
                temp = EndOfPrdvPfunc_cond[j](aNrmNow_tiled, AaggNow_tiled)
                EndOfPrdvP += MrkvArray[i, j] * temp

        # Calculate consumption and the endogenous mNrm gridpoints for this state
        cNrmNow = EndOfPrdvP ** (-1.0 / CRRA)
        mNrmNow = aNrmNow_tiled + cNrmNow

        # Loop through the values in Mgrid and make a piecewise linear consumption function for each
        cFuncBaseByM_list = []
        for n in range(Mcount):
            c_temp = np.insert(cNrmNow[n, :], 0, 0.0)  # Add point at bottom
            m_temp = np.insert(mNrmNow[n, :] - BoroCnstNat_vec[n], 0, 0.0)
            cFuncBaseByM_list.append(LinearInterp(m_temp, c_temp))
            # Add the M-specific consumption function to the list

        # Construct the unconstrained consumption function by combining the M-specific functions
        BoroCnstNat = LinearInterp(
            np.insert(Mgrid, 0, 0.0), np.insert(BoroCnstNat_vec, 0, 0.0)
        )
        cFuncBase = LinearInterpOnInterp1D(cFuncBaseByM_list, Mgrid)
        cFuncUnc = VariableLowerBoundFunc2D(cFuncBase, BoroCnstNat)

        # Combine the constrained consumption function with unconstrained component
        cFuncNow.append(LowerEnvelope2D(cFuncUnc, cFuncCnst))

        # Make the minimum m function as the greater of the natural and artificial constraints
        mNrmMinNow.append(UpperEnvelope(BoroCnstNat, ConstantFunction(BoroCnstArt)))

        # Construct the marginal value function using the envelope condition
        vPfuncNow.append(MargValueFuncCRRA(cFuncNow[-1], CRRA))

    # Pack up and return the solution
    solution_now = ConsumerSolution(
        cFunc=cFuncNow, vPfunc=vPfuncNow, mNrmMin=mNrmMinNow
    )
    return solution_now


###############################################################################


def solve_KrusellSmith(
    solution_next,
    DiscFac,
    CRRA,
    aGrid,
    Mgrid,
    mNextArray,
    MnextArray,
    ProbArray,
    RnextArray,
):
    """
    Solve the one period problem of an agent in Krusell & Smith's canonical 1998 model.
    Because this model is so specialized and only intended to be used with a very narrow
    case, many arrays can be precomputed, making the code here very short.  See the
    method KrusellSmithType.precompute_arrays() for details.

    Parameters
    ----------
    solution_next : ConsumerSolution
        Representation of the solution to next period's problem, including the
        discrete-state-conditional consumption function and marginal value function.
    DiscFac : float
        Intertemporal discount factor.
    CRRA : float
        Coefficient of relative risk aversion.
    aGrid : np.array
        Array of end-of-period asset values.
    Mgrid : np.array
        A grid of aggregate market resources in the economy.
    mNextArray : np.array
        Precomputed array of next period's market resources attained from every
        end-of-period state in the exogenous grid crossed with every shock that
        might attain.  Has shape [aCount, Mcount, 4, 4] ~ [a, M, s, s'].
    MnextArray : np.array
        Precomputed array of next period's aggregate market resources attained
        from every end-of-period state in the exogenous grid crossed with every
        shock that might attain.  Corresponds to mNextArray.
    ProbArray : np.array
        Tiled array of transition probabilities among discrete states.  Every
        slice [i,j,:,:] is identical and translated from MrkvIndArray.
    RnextArray : np.array
        Tiled array of net interest factors next period, attained from every
        end-of-period state crossed with every shock that might attain.

    Returns
    -------
    solution_now : ConsumerSolution
        Representation of this period's solution to the Krusell-Smith model.
    """
    # Loop over next period's state realizations, computing marginal value of market resources
    vPnext = np.zeros_like(mNextArray)
    for j in range(4):
        vPnext[:, :, :, j] = solution_next.vPfunc[j](
            mNextArray[:, :, :, j], MnextArray[:, :, :, j]
        )

    # Compute end-of-period marginal value of assets
    EndOfPrdvP = DiscFac * np.sum(RnextArray * vPnext * ProbArray, axis=3)

    # Invert the first order condition to find optimal consumption
    cNow = EndOfPrdvP ** (-1.0 / CRRA)

    # Find the endogenous gridpoints
    aCount = aGrid.size
    Mcount = Mgrid.size
    aNow = np.tile(np.reshape(aGrid, [aCount, 1, 1]), [1, Mcount, 4])
    mNow = aNow + cNow

    # Insert zeros at the bottom of both cNow and mNow arrays (consume nothing)
    cNow = np.concatenate([np.zeros([1, Mcount, 4]), cNow], axis=0)
    mNow = np.concatenate([np.zeros([1, Mcount, 4]), mNow], axis=0)

    # Construct the consumption and marginal value function for each discrete state
    cFunc_by_state = []
    vPfunc_by_state = []
    for j in range(4):
        cFunc_by_M = [LinearInterp(mNow[:, k, j], cNow[:, k, j]) for k in range(Mcount)]
        cFunc_j = LinearInterpOnInterp1D(cFunc_by_M, Mgrid)
        vPfunc_j = MargValueFuncCRRA(cFunc_j, CRRA)
        cFunc_by_state.append(cFunc_j)
        vPfunc_by_state.append(vPfunc_j)

    # Package and return the solution
    solution_now = ConsumerSolution(cFunc=cFunc_by_state, vPfunc=vPfunc_by_state)
    return solution_now


###############################################################################

CRRA = 2.0
DiscFac = 0.96

# Parameters for a Cobb-Douglas economy
PermGroFacAgg = 1.00  # Aggregate permanent income growth factor
PermShkAggCount = (
    3  # Number of points in discrete approximation to aggregate permanent shock dist
)
TranShkAggCount = (
    3  # Number of points in discrete approximation to aggregate transitory shock dist
)
PermShkAggStd = 0.0063  # Standard deviation of log aggregate permanent shocks
TranShkAggStd = 0.0031  # Standard deviation of log aggregate transitory shocks
DeprFac = 0.025  # Capital depreciation rate
CapShare = 0.36  # Capital's share of income
DiscFacPF = DiscFac  # Discount factor of perfect foresight calibration
CRRAPF = CRRA  # Coefficient of relative risk aversion of perfect foresight calibration
intercept_prev = 0.0  # Intercept of aggregate savings function
slope_prev = 1.0  # Slope of aggregate savings function
verbose_cobb_douglas = (
    True  # Whether to print solution progress to screen while solving
)
T_discard = 200  # Number of simulated "burn in" periods to discard when updating AFunc
DampingFac = 0.5  # Damping factor when updating AFunc; puts DampingFac weight on old params, rest on new
max_loops = 20  # Maximum number of AFunc updating loops to allow


# Make a dictionary to specify a Cobb-Douglas economy
init_cobb_douglas = {
    "PermShkAggCount": PermShkAggCount,
    "TranShkAggCount": TranShkAggCount,
    "PermShkAggStd": PermShkAggStd,
    "TranShkAggStd": TranShkAggStd,
    "DeprFac": DeprFac,
    "CapShare": CapShare,
    "DiscFac": DiscFacPF,
    "CRRA": CRRAPF,
    "PermGroFacAgg": PermGroFacAgg,
    "AggregateL": 1.0,
    "intercept_prev": intercept_prev,
    "slope_prev": slope_prev,
    "verbose": verbose_cobb_douglas,
    "T_discard": T_discard,
    "DampingFac": DampingFac,
    "max_loops": max_loops,
}


class CobbDouglasEconomy(Market):
    """
    A class to represent an economy with a Cobb-Douglas aggregate production
    function over labor and capital, extending HARK.Market.  The "aggregate
    market process" for this market combines all individuals' asset holdings
    into aggregate capital, yielding the interest factor on assets and the wage
    rate for the upcoming period.

    Note: The current implementation assumes a constant labor supply, but
    this will be generalized in the future.

    Parameters
    ----------
    agents : [ConsumerType]
        List of types of consumers that live in this economy.
    tolerance: float
        Minimum acceptable distance between "dynamic rules" to consider the
        solution process converged.  Distance depends on intercept and slope
        of the log-linear "next capital ratio" function.
    act_T : int
        Number of periods to simulate when making a history of of the market.
    """

    def __init__(self, agents=None, tolerance=0.0001, act_T=1200, **kwds):
        agents = agents if agents is not None else list()
        params = init_cobb_douglas.copy()
        params["sow_vars"] = [
            "MaggNow",
            "AaggNow",
            "RfreeNow",
            "wRteNow",
            "PermShkAggNow",
            "TranShkAggNow",
            "KtoLnow",
        ]
        params.update(kwds)

        Market.__init__(
            self,
            agents=agents,
            reap_vars=["aLvl", "pLvl"],
            track_vars=["MaggNow", "AaggNow"],
            dyn_vars=["AFunc"],
            tolerance=tolerance,
            act_T=act_T,
            **params
        )
        self.update()

        # Use previously hardcoded values for AFunc updating if not passed
        # as part of initialization dictionary.  This is to prevent a last
        # minute update to HARK before a release from having a breaking change.
        if not hasattr(self, "DampingFac"):
            self.DampingFac = 0.5
        if not hasattr(self, "max_loops"):
            self.max_loops = 20
        if not hasattr(self, "T_discard"):
            self.T_discard = 200
        if not hasattr(self, "verbose"):
            self.verbose = True

    def mill_rule(self, aLvl, pLvl):
        """
        Function to calculate the capital to labor ratio, interest factor, and
        wage rate based on each agent's current state.  Just calls calc_R_and_W().

        See documentation for calc_R_and_W for more information.
        """
        return self.calc_R_and_W(aLvl, pLvl)

    def calc_dynamics(self, MaggNow, AaggNow):
        """
        Calculates a new dynamic rule for the economy: end of period savings as
        a function of aggregate market resources.  Just calls calc_AFunc().

        See documentation for calc_AFunc for more information.
        """
        return self.calc_AFunc(MaggNow, AaggNow)

    def update(self):
        """
        Use primitive parameters (and perfect foresight calibrations) to make
        interest factor and wage rate functions (of capital to labor ratio),
        as well as discrete approximations to the aggregate shock distributions.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.kSS = (
            (
                self.get_PermGroFacAggLR() ** (self.CRRA) / self.DiscFac
                - (1.0 - self.DeprFac)
            )
            / self.CapShare
        ) ** (1.0 / (self.CapShare - 1.0))
        self.KtoYSS = self.kSS ** (1.0 - self.CapShare)
        self.wRteSS = (1.0 - self.CapShare) * self.kSS ** (self.CapShare)
        self.RfreeSS = (
            1.0 + self.CapShare * self.kSS ** (self.CapShare - 1.0) - self.DeprFac
        )
        self.MSS = self.kSS * self.RfreeSS + self.wRteSS
        self.convertKtoY = lambda KtoY: KtoY ** (
            1.0 / (1.0 - self.CapShare)
        )  # converts K/Y to K/L
        self.Rfunc = lambda k: (
            1.0 + self.CapShare * k ** (self.CapShare - 1.0) - self.DeprFac
        )
        self.wFunc = lambda k: ((1.0 - self.CapShare) * k ** (self.CapShare))

        self.sow_init["KtoLnow"] = self.kSS
        self.sow_init["MaggNow"] = self.kSS
        self.sow_init["AaggNow"] = self.kSS
        self.sow_init["RfreeNow"] = self.Rfunc(self.kSS)
        self.sow_init["wRteNow"] = self.wFunc(self.kSS)
        self.sow_init["PermShkAggNow"] = 1.0
        self.sow_init["TranShkAggNow"] = 1.0
        self.make_AggShkDstn()
        self.AFunc = AggregateSavingRule(self.intercept_prev, self.slope_prev)

    def get_PermGroFacAggLR(self):
        """
        A trivial function that returns self.PermGroFacAgg.  Exists to be overwritten
        and extended by ConsAggShockMarkov model.

        Parameters
        ----------
        None

        Returns
        -------
        PermGroFacAggLR : float
            Long run aggregate permanent income growth, which is the same thing
            as aggregate permanent income growth.
        """
        return self.PermGroFacAgg

    def make_AggShkDstn(self):
        """
        Creates the attributes TranShkAggDstn, PermShkAggDstn, and AggShkDstn.
        Draws on attributes TranShkAggStd, PermShkAddStd, TranShkAggCount, PermShkAggCount.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.TranShkAggDstn = MeanOneLogNormal(sigma=self.TranShkAggStd).approx(
            N=self.TranShkAggCount
        )
        self.PermShkAggDstn = MeanOneLogNormal(sigma=self.PermShkAggStd).approx(
            N=self.PermShkAggCount
        )
        self.AggShkDstn = combine_indep_dstns(self.PermShkAggDstn, self.TranShkAggDstn)

    def reset(self):
        """
        Reset the economy to prepare for a new simulation.  Sets the time index
        of aggregate shocks to zero and runs Market.reset().

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.Shk_idx = 0
        Market.reset(self)

    def make_AggShkHist(self):
        """
        Make simulated histories of aggregate transitory and permanent shocks.
        Histories are of length self.act_T, for use in the general equilibrium
        simulation.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        sim_periods = self.act_T
        Events = np.arange(self.AggShkDstn.prob.size)  # just a list of integers
        EventDraws = self.AggShkDstn.draw(N=sim_periods, data=Events)
        PermShkAggHist = self.AggShkDstn.data[0][EventDraws]
        TranShkAggHist = self.AggShkDstn.data[1][EventDraws]

        # Store the histories
        self.PermShkAggHist = PermShkAggHist * self.PermGroFacAgg
        self.TranShkAggHist = TranShkAggHist

    def calc_R_and_W(self, aLvlNow, pLvlNow):
        """
        Calculates the interest factor and wage rate this period using each agent's
        capital stock to get the aggregate capital ratio.

        Parameters
        ----------
        aLvlNow : [np.array]
            Agents' current end-of-period assets.  Elements of the list correspond
            to types in the economy, entries within arrays to agents of that type.

        Returns
        -------
        MaggNow : float
            Aggregate market resources for this period normalized by mean permanent income
        AaggNow : float
            Aggregate savings for this period normalized by mean permanent income
        RfreeNow : float
            Interest factor on assets in the economy this period.
        wRteNow : float
            Wage rate for labor in the economy this period.
        PermShkAggNow : float
            Permanent shock to aggregate labor productivity this period.
        TranShkAggNow : float
            Transitory shock to aggregate labor productivity this period.
        KtoLnow : float
            Capital-to-labor ratio in the economy this period.
        """
        # Calculate aggregate savings
        AaggPrev = np.mean(np.array(aLvlNow)) / np.mean(
            pLvlNow
        )  # End-of-period savings from last period
        # Calculate aggregate capital this period
        AggregateK = np.mean(np.array(aLvlNow))  # ...becomes capital today
        # This version uses end-of-period assets and
        # permanent income to calculate aggregate capital, unlike the Mathematica
        # version, which first applies the idiosyncratic permanent income shocks
        # and then aggregates.  Obviously this is mathematically equivalent.

        # Get this period's aggregate shocks
        PermShkAggNow = self.PermShkAggHist[self.Shk_idx]
        TranShkAggNow = self.TranShkAggHist[self.Shk_idx]
        self.Shk_idx += 1

        AggregateL = np.mean(pLvlNow) * PermShkAggNow

        # Calculate the interest factor and wage rate this period
        KtoLnow = AggregateK / AggregateL
        self.KtoYnow = KtoLnow ** (1.0 - self.CapShare)
        RfreeNow = self.Rfunc(KtoLnow / TranShkAggNow)
        wRteNow = self.wFunc(KtoLnow / TranShkAggNow)
        MaggNow = KtoLnow * RfreeNow + wRteNow * TranShkAggNow
        self.KtoLnow = KtoLnow  # Need to store this as it is a sow variable

        # Package the results into an object and return it
        return (
            MaggNow,
            AaggPrev,
            RfreeNow,
            wRteNow,
            PermShkAggNow,
            TranShkAggNow,
            KtoLnow,
        )

    def calc_AFunc(self, MaggNow, AaggNow):
        """
        Calculate a new aggregate savings rule based on the history
        of the aggregate savings and aggregate market resources from a simulation.

        Parameters
        ----------
        MaggNow : [float]
            List of the history of the simulated aggregate market resources for an economy.
        AaggNow : [float]
            List of the history of the simulated aggregate savings for an economy.

        Returns
        -------
        (unnamed) : CapDynamicRule
            Object containing a new savings rule
        """
        verbose = self.verbose
        discard_periods = (
            self.T_discard
        )  # Throw out the first T periods to allow the simulation to approach the SS
        update_weight = (
            1.0 - self.DampingFac
        )  # Proportional weight to put on new function vs old function parameters
        total_periods = len(MaggNow)

        # Regress the log savings against log market resources
        logAagg = np.log(AaggNow[discard_periods:total_periods])
        logMagg = np.log(MaggNow[discard_periods - 1 : total_periods - 1])
        slope, intercept, r_value, p_value, std_err = stats.linregress(logMagg, logAagg)

        # Make a new aggregate savings rule by combining the new regression parameters
        # with the previous guess
        intercept = (
            update_weight * intercept + (1.0 - update_weight) * self.intercept_prev
        )
        slope = update_weight * slope + (1.0 - update_weight) * self.slope_prev
        AFunc = AggregateSavingRule(
            intercept, slope
        )  # Make a new next-period capital function

        # Save the new values as "previous" values for the next iteration
        self.intercept_prev = intercept
        self.slope_prev = slope

        # Print the new parameters
        if verbose:
            print(
                "intercept="
                + str(intercept)
                + ", slope="
                + str(slope)
                + ", r-sq="
                + str(r_value**2)
            )

        return AggShocksDynamicRule(AFunc)


class SmallOpenEconomy(Market):
    """
    A class for representing a small open economy, where the wage rate and interest rate are
    exogenously determined by some "global" rate.  However, the economy is still subject to
    aggregate productivity shocks.

    Parameters
    ----------
    agents : [ConsumerType]
        List of types of consumers that live in this economy.
    tolerance: float
        Minimum acceptable distance between "dynamic rules" to consider the
        solution process converged.  Distance depends on intercept and slope
        of the log-linear "next capital ratio" function.
    act_T : int
        Number of periods to simulate when making a history of of the market.
    """

    def __init__(self, agents=None, tolerance=0.0001, act_T=1000, **kwds):
        agents = agents if agents is not None else list()
        Market.__init__(
            self,
            agents=agents,
            sow_vars=[
                "MaggNow",
                "AaggNow",
                "RfreeNow",
                "wRteNow",
                "PermShkAggNow",
                "TranShkAggNow",
                "KtoLnow",
            ],
            reap_vars=[],
            track_vars=["MaggNow", "AaggNow", "KtoLnow"],
            dyn_vars=[],
            tolerance=tolerance,
            act_T=act_T,
        )
        self.assign_parameters(**kwds)
        self.update()

    def update(self):
        """
        Use primitive parameters to set basic objects.
        This is an extremely stripped-down version
        of update for CobbDouglasEconomy.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        self.kSS = 1.0
        self.MSS = 1.0
        self.sow_init["KtoLnow_init"] = self.kSS
        self.Rfunc = ConstantFunction(self.Rfree)
        self.wFunc = ConstantFunction(self.wRte)
        self.sow_init["RfreeNow"] = self.Rfunc(self.kSS)
        self.sow_init["wRteNow"] = self.wFunc(self.kSS)
        self.sow_init["MaggNow"] = self.kSS
        self.sow_init["AaggNow"] = self.kSS
        self.sow_init["PermShkAggNow"] = 1.0
        self.sow_init["TranShkAggNow"] = 1.0
        self.make_AggShkDstn()
        self.AFunc = ConstantFunction(1.0)

    def make_AggShkDstn(self):
        """
        Creates the attributes TranShkAggDstn, PermShkAggDstn, and AggShkDstn.
        Draws on attributes TranShkAggStd, PermShkAddStd, TranShkAggCount, PermShkAggCount.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.TranShkAggDstn = MeanOneLogNormal(sigma=self.TranShkAggStd).approx(
            N=self.TranShkAggCount
        )
        self.PermShkAggDstn = MeanOneLogNormal(sigma=self.PermShkAggStd).approx(
            N=self.PermShkAggCount
        )
        self.AggShkDstn = combine_indep_dstns(self.PermShkAggDstn, self.TranShkAggDstn)

    def mill_rule(self):
        """
        No aggregation occurs for a small open economy, because the wage and interest rates are
        exogenously determined.  However, aggregate shocks may occur.

        See documentation for get_AggShocks() for more information.
        """
        return self.get_AggShocks()

    def calc_dynamics(self, KtoLnow):
        """
        Calculates a new dynamic rule for the economy, which is just an empty object.
        There is no "dynamic rule" for a small open economy, because K/L does not generate w and R.
        """
        return MetricObject()

    def reset(self):
        """
        Reset the economy to prepare for a new simulation.  Sets the time index of aggregate shocks
        to zero and runs Market.reset().  This replicates the reset method for CobbDouglasEconomy;
        future version should create parent class of that class and this one.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.Shk_idx = 0
        Market.reset(self)

    def make_AggShkHist(self):
        """
        Make simulated histories of aggregate transitory and permanent shocks. Histories are of
        length self.act_T, for use in the general equilibrium simulation.  This replicates the same
        method for CobbDouglasEconomy; future version should create parent class.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        sim_periods = self.act_T
        Events = np.arange(self.AggShkDstn.prob.size)  # just a list of integers
        EventDraws = self.AggShkDstn.draw(N=sim_periods, data=Events)
        PermShkAggHist = self.AggShkDstn.data[0][EventDraws]
        TranShkAggHist = self.AggShkDstn.data[1][EventDraws]

        # Store the histories
        self.PermShkAggHist = PermShkAggHist
        self.TranShkAggHist = TranShkAggHist

    def get_AggShocks(self):
        """
        Returns aggregate state variables and shocks for this period.  The capital-to-labor ratio
        is irrelevant and thus treated as constant, and the wage and interest rates are also
        constant.  However, aggregate shocks are assigned from a prespecified history.

        Parameters
        ----------
        None

        Returns
        -------
        MaggNow : float
            Aggregate market resources for this period normalized by mean permanent income
        AaggNow : float
            Aggregate savings for this period normalized by mean permanent income
        RfreeNow : float
            Interest factor on assets in the economy this period.
        wRteNow : float
            Wage rate for labor in the economy this period.
        PermShkAggNow : float
            Permanent shock to aggregate labor productivity this period.
        TranShkAggNow : float
            Transitory shock to aggregate labor productivity this period.
        KtoLnow : float
            Capital-to-labor ratio in the economy this period.

        """
        # Get this period's aggregate shocks
        PermShkAggNow = self.PermShkAggHist[self.Shk_idx]
        TranShkAggNow = self.TranShkAggHist[self.Shk_idx]
        self.Shk_idx += 1

        # Factor prices are constant
        RfreeNow = self.Rfunc(1.0 / PermShkAggNow)
        wRteNow = self.wFunc(1.0 / PermShkAggNow)

        # Aggregates are irrelavent
        AaggNow = 1.0
        MaggNow = 1.0
        KtoLnow = 1.0 / PermShkAggNow

        return (
            MaggNow,
            AaggNow,
            RfreeNow,
            wRteNow,
            PermShkAggNow,
            TranShkAggNow,
            KtoLnow,
        )


# Make a dictionary to specify a Markov Cobb-Douglas economy
init_mrkv_cobb_douglas = init_cobb_douglas.copy()
init_mrkv_cobb_douglas["PermShkAggStd"] = [0.012, 0.006]
init_mrkv_cobb_douglas["TranShkAggStd"] = [0.006, 0.003]
init_mrkv_cobb_douglas["PermGroFacAgg"] = [0.98, 1.02]
init_mrkv_cobb_douglas["MrkvArray"] = MrkvArray
init_mrkv_cobb_douglas["MrkvNow_init"] = 0
init_mrkv_cobb_douglas["slope_prev"] = 2 * [slope_prev]
init_mrkv_cobb_douglas["intercept_prev"] = 2 * [intercept_prev]


class CobbDouglasMarkovEconomy(CobbDouglasEconomy):
    """
    A class to represent an economy with a Cobb-Douglas aggregate production
    function over labor and capital, extending HARK.Market.  The "aggregate
    market process" for this market combines all individuals' asset holdings
    into aggregate capital, yielding the interest factor on assets and the wage
    rate for the upcoming period.  This small extension incorporates a Markov
    state for the "macroeconomy", so that the shock distribution and aggregate
    productivity growth factor can vary over time.

    Parameters
    ----------
    agents : [ConsumerType]
        List of types of consumers that live in this economy.
    tolerance: float
        Minimum acceptable distance between "dynamic rules" to consider the
        solution process converged.  Distance depends on intercept and slope
        of the log-linear "next capital ratio" function.
    act_T : int
        Number of periods to simulate when making a history of of the market.
    """

    def __init__(
        self,
        agents=None,
        tolerance=0.0001,
        act_T=1200,
        sow_vars=[
            "MaggNow",
            "AaggNow",
            "RfreeNow",
            "wRteNow",
            "PermShkAggNow",
            "TranShkAggNow",
            "KtoLnow",
            "Mrkv",  # This one is new
        ],
        **kwds
    ):
        agents = agents if agents is not None else list()
        params = init_mrkv_cobb_douglas.copy()
        params.update(kwds)

        CobbDouglasEconomy.__init__(
            self,
            agents=agents,
            tolerance=tolerance,
            act_T=act_T,
            sow_vars=sow_vars,
            **params
        )

        self.sow_init["Mrkv"] = params["MrkvNow_init"]

    def update(self):
        """
        Use primitive parameters (and perfect foresight calibrations) to make
        interest factor and wage rate functions (of capital to labor ratio),
        as well as discrete approximations to the aggregate shock distributions.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        CobbDouglasEconomy.update(self)
        StateCount = self.MrkvArray.shape[0]
        AFunc_all = []
        for i in range(StateCount):
            AFunc_all.append(
                AggregateSavingRule(self.intercept_prev[i], self.slope_prev[i])
            )
        self.AFunc = AFunc_all

    def get_PermGroFacAggLR(self):
        """
        Calculates and returns the long run permanent income growth factor.  This
        is the average growth factor in self.PermGroFacAgg, weighted by the long
        run distribution of Markov states (as determined by self.MrkvArray).

        Parameters
        ----------
        None

        Returns
        -------
        PermGroFacAggLR : float
            Long run aggregate permanent income growth factor
        """
        # Find the long run distribution of Markov states
        w, v = np.linalg.eig(np.transpose(self.MrkvArray))
        idx = (np.abs(w - 1.0)).argmin()
        x = v[:, idx].astype(float)
        LR_dstn = x / np.sum(x)

        # Return the weighted average of aggregate permanent income growth factors
        PermGroFacAggLR = np.dot(LR_dstn, np.array(self.PermGroFacAgg))
        return PermGroFacAggLR

    def make_AggShkDstn(self):
        """
        Creates the attributes TranShkAggDstn, PermShkAggDstn, and AggShkDstn.
        Draws on attributes TranShkAggStd, PermShkAddStd, TranShkAggCount, PermShkAggCount.
        This version accounts for the Markov macroeconomic state.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        TranShkAggDstn = []
        PermShkAggDstn = []
        AggShkDstn = []
        StateCount = self.MrkvArray.shape[0]

        for i in range(StateCount):
            TranShkAggDstn.append(
                MeanOneLogNormal(sigma=self.TranShkAggStd[i]).approx(
                    N=self.TranShkAggCount
                )
            )
            PermShkAggDstn.append(
                MeanOneLogNormal(sigma=self.PermShkAggStd[i]).approx(
                    N=self.PermShkAggCount
                )
            )
            AggShkDstn.append(
                combine_indep_dstns(PermShkAggDstn[-1], TranShkAggDstn[-1])
            )

        self.TranShkAggDstn = TranShkAggDstn
        self.PermShkAggDstn = PermShkAggDstn
        self.AggShkDstn = AggShkDstn

    def make_AggShkHist(self):
        """
        Make simulated histories of aggregate transitory and permanent shocks.
        Histories are of length self.act_T, for use in the general equilibrium
        simulation.  Draws on history of aggregate Markov states generated by
        internal call to make_Mrkv_history().

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.make_Mrkv_history()  # Make a (pseudo)random sequence of Markov states
        sim_periods = self.act_T

        # For each Markov state in each simulated period, draw the aggregate shocks
        # that would occur in that state in that period
        StateCount = self.MrkvArray.shape[0]
        PermShkAggHistAll = np.zeros((StateCount, sim_periods))
        TranShkAggHistAll = np.zeros((StateCount, sim_periods))
        for i in range(StateCount):
            AggShockDraws = self.AggShkDstn[i].draw(N=sim_periods)
            PermShkAggHistAll[i, :] = AggShockDraws[0, :]
            TranShkAggHistAll[i, :] = AggShockDraws[1, :]

        # Select the actual history of aggregate shocks based on the sequence
        # of Markov states that the economy experiences
        PermShkAggHist = np.zeros(sim_periods)
        TranShkAggHist = np.zeros(sim_periods)
        for i in range(StateCount):
            these = i == self.MrkvNow_hist
            PermShkAggHist[these] = PermShkAggHistAll[i, these] * self.PermGroFacAgg[i]
            TranShkAggHist[these] = TranShkAggHistAll[i, these]

        # Store the histories
        self.PermShkAggHist = PermShkAggHist
        self.TranShkAggHist = TranShkAggHist

    def make_Mrkv_history(self):
        """
        Makes a history of macroeconomic Markov states, stored in the attribute
        MrkvNow_hist.  This version ensures that each state is reached a sufficient
        number of times to have a valid sample for calc_dynamics to produce a good
        dynamic rule.  It will sometimes cause act_T to be increased beyond its
        initially specified level.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if hasattr(self, "loops_max"):
            loops_max = self.loops_max
        else:  # Maximum number of loops; final act_T never exceeds act_T*loops_max
            loops_max = 10

        state_T_min = 50  # Choose minimum number of periods in each state for a valid Markov sequence
        logit_scale = (
            0.2  # Scaling factor on logit choice shocks when jumping to a new state
        )
        # Values close to zero make the most underrepresented states very likely to visit, while
        # large values of logit_scale make any state very likely to be jumped to.

        # Reset act_T to the level actually specified by the user
        if hasattr(self, "act_T_orig"):
            act_T = self.act_T_orig
        else:  # Or store it for the first time
            self.act_T_orig = self.act_T
            act_T = self.act_T

        # Find the long run distribution of Markov states
        w, v = np.linalg.eig(np.transpose(self.MrkvArray))
        idx = (np.abs(w - 1.0)).argmin()
        x = v[:, idx].astype(float)
        LR_dstn = x / np.sum(x)

        # Initialize the Markov history and set up transitions
        MrkvNow_hist = np.zeros(self.act_T_orig, dtype=int)
        loops = 0
        go = True
        MrkvNow = self.sow_init["Mrkv"]
        t = 0
        StateCount = self.MrkvArray.shape[0]

        # Add histories until each state has been visited at least state_T_min times
        while go:
            draws = Uniform(seed=loops).draw(N=self.act_T_orig)
            markov_process = MarkovProcess(self.MrkvArray, seed=loops)
            for s in range(self.act_T_orig):  # Add act_T_orig more periods
                MrkvNow_hist[t] = MrkvNow
                MrkvNow = markov_process.draw(MrkvNow)
                t += 1

            # Calculate the empirical distribution
            state_T = np.zeros(StateCount)
            for i in range(StateCount):
                state_T[i] = np.sum(MrkvNow_hist == i)

            # Check whether each state has been visited state_T_min times
            if np.all(state_T >= state_T_min):
                go = False  # If so, terminate the loop
                continue

            # Choose an underrepresented state to "jump" to
            if np.any(
                state_T == 0
            ):  # If any states have *never* been visited, randomly choose one of those
                never_visited = np.where(np.array(state_T == 0))[0]
                MrkvNow = np.random.choice(never_visited)
            else:  # Otherwise, use logit choice probabilities to visit an underrepresented state
                emp_dstn = state_T / act_T
                ratios = LR_dstn / emp_dstn
                ratios_adj = ratios - np.max(ratios)
                ratios_exp = np.exp(ratios_adj / logit_scale)
                ratios_sum = np.sum(ratios_exp)
                jump_probs = ratios_exp / ratios_sum
                cum_probs = np.cumsum(jump_probs)
                MrkvNow = np.searchsorted(cum_probs, draws[-1])

            loops += 1
            # Make the Markov state history longer by act_T_orig periods
            if loops >= loops_max:
                go = False
                print(
                    "make_Mrkv_history reached maximum number of loops without generating a valid sequence!"
                )
            else:
                MrkvNow_new = np.zeros(self.act_T_orig, dtype=int)
                MrkvNow_hist = np.concatenate((MrkvNow_hist, MrkvNow_new))
                act_T += self.act_T_orig

        # Store the results as attributes of self
        self.MrkvNow_hist = MrkvNow_hist
        self.act_T = act_T

    def mill_rule(self, aLvl, pLvl):
        """
        Function to calculate the capital to labor ratio, interest factor, and
        wage rate based on each agent's current state.  Just calls calc_R_and_W()
        and adds the Markov state index.

        See documentation for calc_R_and_W for more information.

        Params
        -------
        aLvl : float
        pLvl : float

        Returns
        -------
        Mnow : float
            Aggregate market resources for this period.
        Aprev : float
            Aggregate savings for the prior period.
        KtoLnow : float
            Capital-to-labor ratio in the economy this period.
        Rnow : float
            Interest factor on assets in the economy this period.
        Wnow : float
            Wage rate for labor in the economy this period.
        MrkvNow : int
            Binary indicator for bad (0) or good (1) macroeconomic state.
        """
        MrkvNow = self.MrkvNow_hist[self.Shk_idx]
        temp = self.calc_R_and_W(aLvl, pLvl)

        return temp + (MrkvNow,)

    def calc_AFunc(self, MaggNow, AaggNow):
        """
        Calculate a new aggregate savings rule based on the history of the
        aggregate savings and aggregate market resources from a simulation.
        Calculates an aggregate saving rule for each macroeconomic Markov state.

        Parameters
        ----------
        MaggNow : [float]
            List of the history of the simulated  aggregate market resources for an economy.
        AaggNow : [float]
            List of the history of the simulated  aggregate savings for an economy.

        Returns
        -------
        (unnamed) : CapDynamicRule
            Object containing new saving rules for each Markov state.
        """
        verbose = self.verbose
        discard_periods = (
            self.T_discard
        )  # Throw out the first T periods to allow the simulation to approach the SS
        update_weight = (
            1.0 - self.DampingFac
        )  # Proportional weight to put on new function vs old function parameters
        total_periods = len(MaggNow)

        # Trim the histories of M_t and A_t and convert them to logs
        logAagg = np.log(AaggNow[discard_periods:total_periods])
        logMagg = np.log(MaggNow[discard_periods - 1 : total_periods - 1])
        MrkvHist = self.MrkvNow_hist[discard_periods - 1 : total_periods - 1]

        # For each Markov state, regress A_t on M_t and update the saving rule
        AFunc_list = []
        rSq_list = []
        for i in range(self.MrkvArray.shape[0]):
            these = i == MrkvHist
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                logMagg[these], logAagg[these]
            )

            # Make a new aggregate savings rule by combining the new regression parameters
            # with the previous guess
            intercept = (
                update_weight * intercept
                + (1.0 - update_weight) * self.intercept_prev[i]
            )
            slope = update_weight * slope + (1.0 - update_weight) * self.slope_prev[i]
            AFunc_list.append(
                AggregateSavingRule(intercept, slope)
            )  # Make a new next-period capital function
            rSq_list.append(r_value**2)

            # Save the new values as "previous" values for the next iteration
            self.intercept_prev[i] = intercept
            self.slope_prev[i] = slope

        # Print the new parameters
        if verbose:
            print(
                "intercept="
                + str(self.intercept_prev)
                + ", slope="
                + str(self.slope_prev)
                + ", r-sq="
                + str(rSq_list)
            )

        return AggShocksDynamicRule(AFunc_list)


class SmallOpenMarkovEconomy(CobbDouglasMarkovEconomy, SmallOpenEconomy):
    """
    A class for representing a small open economy, where the wage rate and interest rate are
    exogenously determined by some "global" rate.  However, the economy is still subject to
    aggregate productivity shocks.  This version supports a discrete Markov state.  All
    methods in this class inherit from the two parent classes.
    """

    def __init__(self, agents=None, tolerance=0.0001, act_T=1000, **kwds):
        agents = agents if agents is not None else list()
        CobbDouglasMarkovEconomy.__init__(
            self, agents=agents, tolerance=tolerance, act_T=act_T, **kwds
        )
        self.reap_vars = []
        self.dyn_vars = []

    def update(self):
        SmallOpenEconomy.update(self)
        StateCount = self.MrkvArray.shape[0]
        self.AFunc = StateCount * [IdentityFunction()]

    def make_AggShkDstn(self):
        CobbDouglasMarkovEconomy.make_AggShkDstn(self)

    def mill_rule(self):
        MrkvNow = self.MrkvNow_hist[self.Shk_idx]
        temp = SmallOpenEconomy.get_AggShocks(self)
        temp(MrkvNow=MrkvNow)
        return temp

    def calc_dynamics(self, KtoLnow):
        return MetricObject()

    def make_AggShkHist(self):
        CobbDouglasMarkovEconomy.make_AggShkHist(self)


init_KS_economy = {
    "verbose": True,
    "act_T": 11000,
    "T_discard": 1000,
    "DampingFac": 0.5,
    "intercept_prev": [0.0, 0.0],
    "slope_prev": [1.0, 1.0],
    "DiscFac": 0.99,
    "CRRA": 1.0,
    "LbrInd": 0.3271,  # Not listed in KS (1998), but Alan Lujan got this number indirectly from KS
    "ProdB": 0.99,
    "ProdG": 1.01,
    "CapShare": 0.36,
    "DeprFac": 0.025,
    "DurMeanB": 8.0,
    "DurMeanG": 8.0,
    "SpellMeanB": 2.5,
    "SpellMeanG": 1.5,
    "UrateB": 0.10,
    "UrateG": 0.04,
    "RelProbBG": 0.75,
    "RelProbGB": 1.25,
    "MrkvNow_init": 0,
}


class KrusellSmithEconomy(Market):
    """
    A class to represent an economy in the special Krusell-Smith (1998) model.
    This model replicates the one presented in the JPE article "Income and Wealth
    Heterogeneity in the Macroeconomy", with its default parameters set to match
    those in the paper.

    Parameters
    ----------
    agents : [ConsumerType]
        List of types of consumers that live in this economy.
    tolerance: float
        Minimum acceptable distance between "dynamic rules" to consider the
        solution process converged.  Distance depends on intercept and slope
        of the log-linear "next capital ratio" function.
    act_T : int
        Number of periods to simulate when making a history of of the market.
    """

    def __init__(self, agents=None, tolerance=0.0001, **kwds):
        agents = agents if agents is not None else list()
        params = deepcopy(init_KS_economy)
        params.update(kwds)

        Market.__init__(
            self,
            agents=agents,
            tolerance=tolerance,
            sow_vars=["Mnow", "Aprev", "Mrkv", "Rnow", "Wnow"],
            reap_vars=["aNow", "EmpNow"],
            track_vars=["Mrkv", "Aprev", "Mnow", "Urate"],
            dyn_vars=["AFunc"],
            **params
        )
        self.update()

    def update(self):
        """
        Construct trivial initial guesses of the aggregate saving rules, as well
        as the perfect foresight steady state and associated objects.
        """
        StateCount = 2
        AFunc_all = [
            AggregateSavingRule(self.intercept_prev[j], self.slope_prev[j])
            for j in range(StateCount)
        ]
        self.AFunc = AFunc_all
        self.KtoLSS = (
            (1.0**self.CRRA / self.DiscFac - (1.0 - self.DeprFac)) / self.CapShare
        ) ** (1.0 / (self.CapShare - 1.0))
        self.KSS = self.KtoLSS * self.LbrInd
        self.KtoYSS = self.KtoLSS ** (1.0 - self.CapShare)
        self.WSS = (1.0 - self.CapShare) * self.KtoLSS ** (self.CapShare)
        self.RSS = (
            1.0 + self.CapShare * self.KtoLSS ** (self.CapShare - 1.0) - self.DeprFac
        )
        self.MSS = self.KSS * self.RSS + self.WSS * self.LbrInd
        self.convertKtoY = lambda KtoY: KtoY ** (
            1.0 / (1.0 - self.CapShare)
        )  # converts K/Y to K/L
        self.rFunc = lambda k: self.CapShare * k ** (self.CapShare - 1.0)
        self.Wfunc = lambda k: ((1.0 - self.CapShare) * k ** (self.CapShare))
        self.sow_init["KtoLnow"] = self.KtoLSS
        self.sow_init["Mnow"] = self.MSS
        self.sow_init["Aprev"] = self.KSS
        self.sow_init["Rnow"] = self.RSS
        self.sow_init["Wnow"] = self.WSS
        self.PermShkAggNow_init = 1.0
        self.TranShkAggNow_init = 1.0
        self.sow_init["Mrkv"] = 0
        self.make_MrkvArray()

    def reset(self):
        """
        Reset the economy to prepare for a new simulation.  Sets the time index
        of aggregate shocks to zero and runs Market.reset().
        """
        self.Shk_idx = 0
        Market.reset(self)

    def make_MrkvArray(self):
        """
        Construct the attributes MrkvAggArray and MrkvIndArray from the primitive
        attributes DurMeanB, DurMeanG, SpellMeanB, SpellMeanG, UrateB, UrateG,
        RelProbGB, and RelProbBG.
        """
        # Construct aggregate Markov transition probabilities
        ProbBG = 1.0 / self.DurMeanB
        ProbGB = 1.0 / self.DurMeanG
        ProbBB = 1.0 - ProbBG
        ProbGG = 1.0 - ProbGB
        MrkvAggArray = np.array([[ProbBB, ProbBG], [ProbGB, ProbGG]])

        # Construct idiosyncratic Markov transition probabilities
        # ORDER: BU, BE, GU, GE
        MrkvIndArray = np.zeros((4, 4))

        # BAD-BAD QUADRANT
        MrkvIndArray[0, 1] = ProbBB * 1.0 / self.SpellMeanB
        MrkvIndArray[0, 0] = ProbBB * (1 - 1.0 / self.SpellMeanB)
        MrkvIndArray[1, 0] = self.UrateB / (1.0 - self.UrateB) * MrkvIndArray[0, 1]
        MrkvIndArray[1, 1] = ProbBB - MrkvIndArray[1, 0]

        # GOOD-GOOD QUADRANT
        MrkvIndArray[2, 3] = ProbGG * 1.0 / self.SpellMeanG
        MrkvIndArray[2, 2] = ProbGG * (1 - 1.0 / self.SpellMeanG)
        MrkvIndArray[3, 2] = self.UrateG / (1.0 - self.UrateG) * MrkvIndArray[2, 3]
        MrkvIndArray[3, 3] = ProbGG - MrkvIndArray[3, 2]

        # BAD-GOOD QUADRANT
        MrkvIndArray[0, 2] = self.RelProbBG * MrkvIndArray[2, 2] / ProbGG * ProbBG
        MrkvIndArray[0, 3] = ProbBG - MrkvIndArray[0, 2]
        MrkvIndArray[1, 2] = (
            ProbBG * self.UrateG - self.UrateB * MrkvIndArray[0, 2]
        ) / (1.0 - self.UrateB)
        MrkvIndArray[1, 3] = ProbBG - MrkvIndArray[1, 2]

        # GOOD-BAD QUADRANT
        MrkvIndArray[2, 0] = self.RelProbGB * MrkvIndArray[0, 0] / ProbBB * ProbGB
        MrkvIndArray[2, 1] = ProbGB - MrkvIndArray[2, 0]
        MrkvIndArray[3, 0] = (
            ProbGB * self.UrateB - self.UrateG * MrkvIndArray[2, 0]
        ) / (1.0 - self.UrateG)
        MrkvIndArray[3, 1] = ProbGB - MrkvIndArray[3, 0]

        # Test for valid idiosyncratic transition probabilities
        assert np.all(
            MrkvIndArray >= 0.0
        ), "Invalid idiosyncratic transition probabilities!"
        self.MrkvArray = MrkvAggArray
        self.MrkvIndArray = MrkvIndArray

    def make_Mrkv_history(self):
        """
        Makes a history of macroeconomic Markov states, stored in the attribute
        MrkvNow_hist.  This variable is binary (0 bad, 1 good) in the KS model.
        """
        # Initialize the Markov history and set up transitions
        self.MrkvNow_hist = np.zeros(self.act_T, dtype=int)
        MrkvNow = self.MrkvNow_init

        markov_process = MarkovProcess(self.MrkvArray, seed=0)
        for s in range(self.act_T):  # Add act_T_orig more periods
            self.MrkvNow_hist[s] = MrkvNow
            MrkvNow = markov_process.draw(MrkvNow)

    def mill_rule(self, aNow, EmpNow):
        """
        Method to calculate the capital to labor ratio, interest factor, and
        wage rate based on each agent's current state.  Just calls calc_R_and_W().

        See documentation for calc_R_and_W for more information.

        Returns
        -------
        Mnow : float
            Aggregate market resources for this period.
        Aprev : float
            Aggregate savings for the prior period.
        MrkvNow : int
            Binary indicator for bad (0) or good (1) macroeconomic state.
        Rnow : float
            Interest factor on assets in the economy this period.
        Wnow : float
            Wage rate for labor in the economy this period.
        """

        return self.calc_R_and_W(aNow, EmpNow)

    def calc_dynamics(self, Mnow, Aprev):
        """
        Method to update perceptions of the aggregate saving rule in each
        macroeconomic state; just calls calc_AFunc.
        """
        return self.calc_AFunc(Mnow, Aprev)

    def calc_R_and_W(self, aNow, EmpNow):
        """
        Calculates the interest factor and wage rate this period using each agent's
        capital stock to get the aggregate capital ratio.

        Parameters
        ----------
        aNow : [np.array]
            Agents' current end-of-period assets.  Elements of the list correspond
            to types in the economy, entries within arrays to agents of that type.
        EmpNow [np.array]
            Agents' binary employment states.  Not actually used in computation of
            interest and wage rates, but stored in the history to verify that the
            idiosyncratic unemployment probabilities are behaving as expected.

        Returns
        -------
        Mnow : float
            Aggregate market resources for this period.
        Aprev : float
            Aggregate savings for the prior period.
        MrkvNow : int
            Binary indicator for bad (0) or good (1) macroeconomic state.
        Rnow : float
            Interest factor on assets in the economy this period.
        Wnow : float
            Wage rate for labor in the economy this period.
        """
        # Calculate aggregate savings
        Aprev = np.mean(np.array(aNow))  # End-of-period savings from last period
        # Calculate aggregate capital this period
        AggK = Aprev  # ...becomes capital today

        # Calculate unemployment rate
        Urate = 1.0 - np.mean(np.array(EmpNow))
        self.Urate = Urate  # This is the unemployment rate for the *prior* period

        # Get this period's TFP and labor supply
        MrkvNow = self.MrkvNow_hist[self.Shk_idx]
        if MrkvNow == 0:
            Prod = self.ProdB
            AggL = (1.0 - self.UrateB) * self.LbrInd
        elif MrkvNow == 1:
            Prod = self.ProdG
            AggL = (1.0 - self.UrateG) * self.LbrInd
        self.Shk_idx += 1

        # Calculate the interest factor and wage rate this period
        KtoLnow = AggK / AggL
        Rnow = 1.0 + Prod * self.rFunc(KtoLnow) - self.DeprFac
        Wnow = Prod * self.Wfunc(KtoLnow)
        Mnow = Rnow * AggK + Wnow * AggL
        self.KtoLnow = KtoLnow  # Need to store this as it is a sow variable

        # Returns a tuple of these values
        return Mnow, Aprev, MrkvNow, Rnow, Wnow

    def calc_AFunc(self, Mnow, Aprev):
        """
        Calculate a new aggregate savings rule based on the history of the
        aggregate savings and aggregate market resources from a simulation.
        Calculates an aggregate saving rule for each macroeconomic Markov state.

        Parameters
        ----------
        Mnow : [float]
            List of the history of the simulated aggregate market resources for an economy.
        Anow : [float]
            List of the history of the simulated aggregate savings for an economy.

        Returns
        -------
        (unnamed) : CapDynamicRule
            Object containing new saving rules for each Markov state.
        """
        verbose = self.verbose
        discard_periods = (
            self.T_discard
        )  # Throw out the first T periods to allow the simulation to approach the SS
        update_weight = (
            1.0 - self.DampingFac
        )  # Proportional weight to put on new function vs old function parameters
        total_periods = len(Mnow)

        # Trim the histories of M_t and A_t and convert them to logs
        logAagg = np.log(Aprev[discard_periods:total_periods])
        logMagg = np.log(Mnow[discard_periods - 1 : total_periods - 1])
        MrkvHist = self.MrkvNow_hist[discard_periods - 1 : total_periods - 1]

        # For each Markov state, regress A_t on M_t and update the saving rule
        AFunc_list = []
        rSq_list = []
        for i in range(self.MrkvArray.shape[0]):
            these = i == MrkvHist
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                logMagg[these], logAagg[these]
            )

            # Make a new aggregate savings rule by combining the new regression parameters
            # with the previous guess
            intercept = (
                update_weight * intercept
                + (1.0 - update_weight) * self.intercept_prev[i]
            )
            slope = update_weight * slope + (1.0 - update_weight) * self.slope_prev[i]
            AFunc_list.append(
                AggregateSavingRule(intercept, slope)
            )  # Make a new next-period capital function
            rSq_list.append(r_value**2)

            # Save the new values as "previous" values for the next iteration
            self.intercept_prev[i] = intercept
            self.slope_prev[i] = slope

        # Print the new parameters
        if verbose:
            print(
                "intercept="
                + str(self.intercept_prev)
                + ", slope="
                + str(self.slope_prev)
                + ", r-sq="
                + str(rSq_list)
            )

        return AggShocksDynamicRule(AFunc_list)


class AggregateSavingRule(MetricObject):
    """
    A class to represent agent beliefs about aggregate saving at the end of this period (AaggNow) as
    a function of (normalized) aggregate market resources at the beginning of the period (MaggNow).

    Parameters
    ----------
    intercept : float
        Intercept of the log-linear capital evolution rule.
    slope : float
        Slope of the log-linear capital evolution rule.
    """

    def __init__(self, intercept, slope):
        self.intercept = intercept
        self.slope = slope
        self.distance_criteria = ["slope", "intercept"]

    def __call__(self, Mnow):
        """
        Evaluates aggregate savings as a function of the aggregate market resources this period.

        Parameters
        ----------
        Mnow : float
            Aggregate market resources this period.

        Returns
        -------
        Aagg : Expected aggregate savings this period.
        """
        Aagg = np.exp(self.intercept + self.slope * np.log(Mnow))
        return Aagg


class AggShocksDynamicRule(MetricObject):
    """
    Just a container class for passing the dynamic rule in the aggregate shocks model to agents.

    Parameters
    ----------
    AFunc : CapitalEvoRule
        Aggregate savings as a function of aggregate market resources.
    """

    def __init__(self, AFunc):
        self.AFunc = AFunc
        self.distance_criteria = ["AFunc"]

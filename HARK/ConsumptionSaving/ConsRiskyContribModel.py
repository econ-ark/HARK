"""
This file contains classes and functions for representing, solving, and simulating
agents who must allocate their resources among consumption, saving in a risk-free
asset (with a low return), and saving in a risky asset (with higher average return).
"""
import numpy as np
from copy import deepcopy
from HARK import MetricObject, NullFunc  # Basic HARK features
from HARK.ConsumptionSaving.ConsIndShockModel import (
    utility,  # CRRA utility function
    utility_inv,  # Inverse CRRA utility function
    utilityP,  # CRRA marginal utility function
    utilityP_inv,  # Inverse CRRA marginal utility function
    init_lifecycle
)

from HARK.ConsumptionSaving.ConsRiskyAssetModel import (
    RiskyAssetConsumerType,
    risky_asset_parms,
    init_risky_asset,
)

from HARK.distribution import calc_expectation

from HARK.interpolation import (
    LinearInterp,  # Piecewise linear interpolation
    BilinearInterp,  # 2D interpolator
    TrilinearInterp,  # 3D interpolator
    ConstantFunction,  # Interpolator-like class that returns constant value
    IdentityFunction,  # Interpolator-like class that returns one of its arguments
    ValueFuncCRRA,
    MargValueFuncCRRA,
    DiscreteInterp,
)

from HARK.utilities import make_grid_exp_mult


class RiskyContribConsumerType(RiskyAssetConsumerType):
    """
    A consumer type with idiosyncratic shocks to permanent and transitory income,
    who can save in both a risk-free and a risky asset but faces frictions to
    moving funds between them. The agent can only consume out of his risk-free
    asset.
    
    The frictions are:
        - A proportional tax on funds moved from the risky to the risk-free
         asset.
        - A stochastic inability to move funds between his accounts.
    
    To partially avoid the second friction, the agent can commit to have a
    fraction of his labor income, which is usually deposited in his risk-free
    account, diverted to his risky account. He can change this fraction
    only in periods where he is able to move funds between accounts.
    """

    time_inv_ = deepcopy(RiskyAssetConsumerType.time_inv_)
    time_inv_ = time_inv_ + ["DiscreteShareBool"]

    state_vars = RiskyAssetConsumerType.state_vars + [
        "nNrm",
        "mNrmTilde",
        "nNrmTilde",
        "Share",
    ]
    shock_vars_ = RiskyAssetConsumerType.shock_vars_

    def __init__(self, cycles=1, verbose=False, quiet=False, **kwds):

        params = init_risky_contrib.copy()
        params.update(kwds)
        kwds = params

        # Initialize a basic consumer type
        RiskyAssetConsumerType.__init__(
            self, cycles=cycles, verbose=verbose, quiet=quiet, **kwds
        )

        # The model is solved and simulated spliting each of the agent's
        # decisions into its own "stage". The stages in chronological order
        # are
        # - Reb: asset-rebalancing stage.
        # - Sha: definition of the income contribution share.
        # - Cns: consumption stage.
        self.stages = ["Reb", "Sha", "Cns"]

        # Each stage has its own states and controls, and its methods
        # to find them.
        self.set_states = {
            "Reb": self.get_states_Reb,
            "Sha": self.get_states_Sha,
            "Cns": self.get_states_Cns,
        }

        self.set_controls = {
            "Reb": self.get_controls_Reb,
            "Sha": self.get_controls_Sha,
            "Cns": self.get_controls_Cns,
        }

        # Set the solver for the portfolio model, and update various constructed attributes
        self.solve_one_period = solveRiskyContrib
        self.update()

    def pre_solve(self):
        self.update_solution_terminal()

    def update(self):

        RiskyAssetConsumerType.update(self)
        self.update_share_grid()
        self.update_d_grid()
        self.update_nNrm_grid()
        self.update_mNrm_grid()
        self.update_tau()

    def update_solution_terminal(self):
        """
        Solves the terminal period. The solution is trivial.
        Cns: agent will consume all of his liquid resources.
        Sha: irrelevant as there is no "next" period.
        Reb: agent will shift all of his resources to the risk-free asset.
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # Construct the terminal solution backwards.

        # Start with the consumption stage. All liquid resources are consumed.
        cFunc_term = IdentityFunction(i_dim=0, n_dims=3)
        vFuncCns_term = ValueFuncCRRA(cFunc_term, CRRA=self.CRRA)
        # Marginal values
        dvdmFuncCns_term = MargValueFuncCRRA(cFunc_term, CRRA=self.CRRA)
        dvdnFuncCns_term = ConstantFunction(0.0)
        dvdsFuncCns_term = ConstantFunction(0.0)

        CnsStageSol = RiskyContribCnsSolution(
            # Consumption stage
            vFuncCns=vFuncCns_term,
            cFunc=cFunc_term,
            dvdmFuncCns=dvdmFuncCns_term,
            dvdnFuncCns=dvdnFuncCns_term,
            dvdsFuncCns=dvdsFuncCns_term,
        )

        # Share stage

        # It's irrelevant because there is no future period. Set share to 0.
        # Create a dummy 2-d consumption function to get value function and marginal
        c2d = IdentityFunction(i_dim=0, n_dims=2)
        ShaStageSol = RiskyContribShaSolution(
            # Adjust
            vFuncShaAdj=ValueFuncCRRA(c2d, CRRA=self.CRRA),
            ShareFuncAdj=ConstantFunction(0.0),
            dvdmFuncShaAdj=MargValueFuncCRRA(c2d, CRRA=self.CRRA),
            dvdnFuncShaAdj=ConstantFunction(0.0),
            # Fixed
            vFuncShaFxd=vFuncCns_term,
            ShareFuncFxd=IdentityFunction(i_dim=2, n_dims=3),
            dvdmFuncShaFxd=dvdmFuncCns_term,
            dvdnFuncShaFxd=dvdnFuncCns_term,
            dvdsFuncShaFxd=dvdsFuncCns_term,
        )

        # Rabalancing stage

        # Adjusting agent:
        # Withdraw everything from the pension fund and consume everything
        DFuncAdj_term = ConstantFunction(-1.0)

        # Find the withdrawal penalty. If it is time-varying, assume it takes
        # the same value as in the last non-terminal period
        if type(self.tau) is list:
            tau = self.tau[-1]
        else:
            tau = self.tau

        # Value and marginal value function of the adjusting agent
        vFuncRebAdj_term = ValueFuncCRRA(lambda m, n: m + n / (1 + tau), self.CRRA)
        dvdmFuncRebAdj_term = MargValueFuncCRRA(
            lambda m, n: m + n / (1 + tau), self.CRRA
        )
        # A marginal unit of n will be withdrawn and put into m. Then consumed.
        dvdnFuncRebAdj_term = lambda m, n: dvdmFuncRebAdj_term(m, n) / (1 + tau)

        RebStageSol = RiskyContribRebSolution(
            # Rebalancing stage
            vFuncRebAdj=vFuncRebAdj_term,
            DFuncAdj=DFuncAdj_term,
            dvdmFuncRebAdj=dvdmFuncRebAdj_term,
            dvdnFuncRebAdj=dvdnFuncRebAdj_term,
            # Adjusting stage
            vFuncRebFxd=vFuncCns_term,
            DFuncFxd=ConstantFunction(0.0),
            dvdmFuncRebFxd=dvdmFuncCns_term,
            dvdnFuncRebFxd=dvdnFuncCns_term,
            dvdsFuncRebFxd=dvdsFuncCns_term,
        )

        # Construct the terminal period solution
        self.solution_terminal = RiskyContribSolution(
            RebStageSol, ShaStageSol, CnsStageSol
        )

    def update_tau(self):
        """
        Checks that the tax rate on risky-to-risk-free flows has the appropriate
        length adds it to time_(in)vary

        Returns
        -------
        None.

        """
        if type(self.tau) is list and (len(self.tau) == self.T_cycle):
            self.add_to_time_vary("tau")
        elif type(self.tau) is list:
            raise AttributeError(
                "If tau is time-varying, it must have length of T_cycle!"
            )
        else:
            self.add_to_time_inv("tau")

    def update_share_grid(self):
        """
        Creates grid for the income contribution share.
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.ShareGrid = np.linspace(0.0, self.ShareMax, self.ShareCount)
        self.add_to_time_inv("ShareGrid")

    def update_d_grid(self):
        """
        Creates grid for the rebalancing flow between assets. This flow is
        normalized as a ratio.
        - If d > 0, d*mNrm flows towards the risky asset.
        - If d < 0, d*nNrm (pre-tax) flows towards the risk-free asset.

        Returns
        -------
        None.

        """
        self.dGrid = np.linspace(0, 1, self.dCount)
        self.add_to_time_inv("dGrid")

    def update_nNrm_grid(self):
        """
        Updates the agent's iliquid assets grid by constructing a
        multi-exponentially spaced grid of nNrm values.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None.
        """
        # Extract parameters
        nNrmMin = self.nNrmMin
        nNrmMax = self.nNrmMax
        nNrmCount = self.nNrmCount
        exp_nest = self.nNrmNestFac
        # Create grid
        nNrmGrid = make_grid_exp_mult(
            ming=nNrmMin, maxg=nNrmMax, ng=nNrmCount, timestonest=exp_nest
        )
        # Assign and set it as time invariant
        self.nNrmGrid = nNrmGrid
        self.add_to_time_inv("nNrmGrid")

    def update_mNrm_grid(self):
        """
        Updates the agent's liquid assets exogenous grid by constructing a
        multi-exponentially spaced grid of mNrm values.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None.
        """
        # Extract parameters
        mNrmMin = self.mNrmMin
        mNrmMax = self.mNrmMax
        mNrmCount = self.mNrmCount
        exp_nest = self.mNrmNestFac
        # Create grid
        mNrmGrid = make_grid_exp_mult(
            ming=mNrmMin, maxg=mNrmMax, ng=mNrmCount, timestonest=exp_nest
        )
        # Assign and set it as time invariant
        self.mNrmGrid = mNrmGrid
        self.add_to_time_inv("mNrmGrid")

    def initialize_sim(self):
        """
        Initialize the state of simulation attributes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        RiskyAssetConsumerType.initialize_sim(self)
        self.state_now["Share"] = np.zeros(self.AgentCount)

    def sim_birth(self, which_agents):
        """
        Create new agents to replace ones who have recently died; takes draws of
        initial aNrm and pLvl, as in ConsIndShockModel, then sets Share, Adjust
        and post-rebalancing risky asset nNrmTilde to zero as initial values.
        Parameters
        ----------
        which_agents : np.array
            Boolean array of size AgentCount indicating which agents should be "born".

        Returns
        -------
        None
        """

        RiskyAssetConsumerType.sim_birth(self, which_agents)
        self.state_now["Share"][which_agents] = 0.0
        self.state_now["nNrmTilde"][which_agents] = 0.0

    def sim_one_period(self):
        """
        Simulates one period for this type.
        
        Has to be re-defined instead of using AgentType.sim_one_period() because
        of the "stages" structure.
        
        Parameters
        ----------
        None
        Returns
        -------
        None
        """

        if not hasattr(self, "solution"):
            raise Exception(
                "Model instance does not have a solution stored. To simulate, it is necessary"
                " to run the `solve()` method of the class first."
            )

        # Mortality adjusts the agent population
        self.get_mortality()  # Replace some agents with "newborns"

        # Make state_now into state_prev, clearing state_now
        for var in self.state_now:
            self.state_prev[var] = self.state_now[var]

            if isinstance(self.state_now[var], np.ndarray):
                self.state_now[var] = np.empty(self.AgentCount)
            else:
                # Probably an aggregate variable. It may be getting set by the Market.
                pass

        if self.read_shocks:  # If shock histories have been pre-specified, use those
            self.read_shocks_from_history()
        else:  # Otherwise, draw shocks as usual according to subclass-specific method
            self.get_shocks()

        # Sequentially get states and controls of every stage
        for s in self.stages:
            self.set_states[s]()
            self.set_controls[s]()

        self.get_post_states()

        # Advance time for all agents
        self.t_age = self.t_age + 1  # Age all consumers by one period
        self.t_cycle = self.t_cycle + 1  # Age all consumers within their cycle
        self.t_cycle[
            self.t_cycle == self.T_cycle
        ] = 0  # Resetting to zero for those who have reached the end

    def get_states_Reb(self):
        """
        Get states for the first "stage": rebalancing.
        """

        pLvlPrev = self.state_prev["pLvl"]
        aNrmPrev = self.state_prev["aNrm"]
        SharePrev = self.state_prev["Share"]
        nNrmTildePrev = self.state_prev["nNrmTilde"]
        Rfree = self.Rfree
        Rrisk = self.shocks["Risky"]

        # Calculate new states:

        # Permanent income
        self.state_now["pLvl"] = pLvlPrev * self.shocks["PermShk"]
        self.state_now["PlvlAgg"] = self.state_prev["PlvlAgg"] * self.PermShkAggNow

        # Assets: mNrm and nNrm

        # Compute the effective growth factor of each asset
        RfEff = Rfree / self.shocks["PermShk"]
        RrEff = Rrisk / self.shocks["PermShk"]

        bNrm = RfEff * aNrmPrev  # Liquid balances before labor income
        gNrm = RrEff * nNrmTildePrev  # Iliquid balances before labor income

        # Liquid balances after labor income
        self.state_now["mNrm"] = bNrm + self.shocks["TranShk"] * (1 - SharePrev)
        # Iliquid balances after labor income
        self.state_now["nNrm"] = gNrm + self.shocks["TranShk"] * SharePrev

        return None

    def get_controls_Reb(self):
        """
        Get controls for the first stage: rebalancing
        """
        DNrm = np.zeros(self.AgentCount) + np.nan

        # Loop over each period of the cycle, getting controls separately depending on "age"
        for t in range(self.T_cycle):

            # Find agents in this period-stage
            these = t == self.t_cycle

            # Get controls for agents who *can* adjust.
            those = np.logical_and(these, self.shocks["Adjust"])
            DNrm[those] = (
                self.solution[t]
                .stageSols["Reb"]
                .DFuncAdj(self.state_now["mNrm"][those], self.state_now["nNrm"][those])
            )

            # Get Controls for agents who *can't* adjust.
            those = np.logical_and(these, np.logical_not(self.shocks["Adjust"]))
            DNrm[those] = (
                self.solution[t]
                .stageSols["Reb"]
                .DFuncFxd(
                    self.state_now["mNrm"][those],
                    self.state_now["nNrm"][those],
                    self.state_prev["Share"][those],
                )
            )

        # Store controls as attributes of self
        self.controls["DNrm"] = DNrm

    def get_states_Sha(self):
        """
        Get states for the second "stage": choosing the contribution share.
        """

        # Post-states are assets after rebalancing

        if not "tau" in self.time_vary:

            mNrmTilde, nNrmTilde = rebalance_assets(
                self.controls["DNrm"],
                self.state_now["mNrm"],
                self.state_now["nNrm"],
                self.tau,
            )

        else:

            # Initialize
            mNrmTilde = np.zeros_like(self.state_now["mNrm"]) + np.nan
            nNrmTilde = np.zeros_like(self.state_now["mNrm"]) + np.nan

            # Loop over each period of the cycle, getting controls separately depending on "age"
            for t in range(self.T_cycle):

                # Find agents in this period-stage
                these = t == self.t_cycle

                if np.sum(these) > 0:
                    tau = self.tau[t]

                    mNrmTilde[these], nNrmTilde[these] = rebalance_assets(
                        self.controls["DNrm"][these],
                        self.state_now["mNrm"][these],
                        self.state_now["nNrm"][these],
                        tau,
                    )

        self.state_now["mNrmTilde"] = mNrmTilde
        self.state_now["nNrmTilde"] = nNrmTilde

    def get_controls_Sha(self):
        """
        Get controls for the second "stage": choosing the contribution share.
        """

        Share = np.zeros(self.AgentCount) + np.nan

        # Loop over each period of the cycle, getting controls separately depending on "age"
        for t in range(self.T_cycle):

            # Find agents in this period-stage
            these = t == self.t_cycle

            # Get controls for agents who *can* adjust.
            those = np.logical_and(these, self.shocks["Adjust"])
            Share[those] = (
                self.solution[t]
                .stageSols["Sha"]
                .ShareFuncAdj(
                    self.state_now["mNrmTilde"][those],
                    self.state_now["nNrmTilde"][those],
                )
            )

            # Get Controls for agents who *can't* adjust.
            those = np.logical_and(these, np.logical_not(self.shocks["Adjust"]))
            Share[those] = (
                self.solution[t]
                .stageSols["Sha"]
                .ShareFuncFxd(
                    self.state_now["mNrmTilde"][those],
                    self.state_now["nNrmTilde"][those],
                    self.state_prev["Share"][those],
                )
            )

        # Store controls as attributes of self
        self.controls["Share"] = Share

    def get_states_Cns(self):
        """
        Get states for the third "stage": consumption.
        """

        # Contribution share becomes a state in the consumption problem
        self.state_now["Share"] = self.controls["Share"]

    def get_controls_Cns(self):
        """
        Get controls for the third "stage": consumption.
        """

        cNrm = np.zeros(self.AgentCount) + np.nan

        # Loop over each period of the cycle, getting controls separately depending on "age"
        for t in range(self.T_cycle):

            # Find agents in this period-stage
            these = t == self.t_cycle

            # Get consumption
            cNrm[these] = (
                self.solution[t]
                .stageSols["Cns"]
                .cFunc(
                    self.state_now["mNrmTilde"][these],
                    self.state_now["nNrmTilde"][these],
                    self.state_now["Share"][these],
                )
            )

        # Store controls as attributes of self
        # Since agents might be willing to end the period with a = 0, make
        # sure consumption does not go over m because of some numerical error.
        self.controls["cNrm"] = np.minimum(cNrm, self.state_now["mNrmTilde"])

    def get_post_states(self):
        """
        Set variables that are not a state to any problem but need to be
        computed in order to interact with shocks and produce next period's
        states.
        """
        self.state_now["aNrm"] = self.state_now["mNrmTilde"] - self.controls["cNrm"]


# %% Classes for RiskyContrib type solution objects

# Class for asset adjustment stage solution
class RiskyContribRebSolution(MetricObject):
    """
    A class for representing the solution to the asset-rebalancing stage of
    the 'RiskyContrib' model.
    
    Parameters
    ----------
    vFuncRebAdj : ValueFunc2D
        Stage value function over normalized liquid resources and normalized
        iliquid resources when the agent is able to adjust his portfolio.
    DFuncAdj : Interp2D
        Deposit function over normalized liquid resources and normalized
        iliquid resources when the agent is able to adjust his portfolio.
    dvdmFuncRebAdj : MargValueFunc2D
        Marginal value over normalized liquid resources when the agent is able
        to adjust his portfolio.
    dvdnFuncRebAdj : MargValueFunc2D
        Marginal value over normalized liquid resources when the agent is able
        to adjust his portfolio.
    vFuncRebFxd : ValueFunc3D
        Stage value function over normalized liquid resources, normalized
        iliquid resources, and income contribution share when the agent is
        not able to adjust his portfolio.
    DFuncFxd : Interp2D
        Deposit function over normalized liquid resources, normalized iliquid
        resources, and income contribution share when the agent is not able to
        adjust his portfolio.
        Must be ConstantFunction(0.0)
    dvdmFuncRebFxd : MargValueFunc3D
        Marginal value over normalized liquid resources when the agent is not
        able to adjust his portfolio.
    dvdnFuncRebFxd : MargValueFunc3D
        Marginal value over normalized iliquid resources when the agent is not
        able to adjust his portfolio.
    dvdsFuncRebFxd : Interp3D
        Marginal value function over income contribution share when the agent
        is not able to ajust his portfolio.
    """

    distance_criteria = ["dvdmFuncRebAdj", "dvdnFuncRebAdj"]

    def __init__(
        self,
        # Rebalancing stage, adjusting
        vFuncRebAdj=None,
        DFuncAdj=None,
        dvdmFuncRebAdj=None,
        dvdnFuncRebAdj=None,
        # Rebalancing stage, fixed
        vFuncRebFxd=None,
        DFuncFxd=None,
        dvdmFuncRebFxd=None,
        dvdnFuncRebFxd=None,
        dvdsFuncRebFxd=None,
    ):

        # Rebalancing stage
        if vFuncRebAdj is None:
            vFuncRebAdj = NullFunc()
        if DFuncAdj is None:
            DFuncAdj = NullFunc()
        if dvdmFuncRebAdj is None:
            dvdmFuncRebAdj = NullFunc()
        if dvdnFuncRebAdj is None:
            dvdnFuncRebAdj = NullFunc()

        if vFuncRebFxd is None:
            vFuncRebFxd = NullFunc()
        if DFuncFxd is None:
            DFuncFxd = NullFunc()
        if dvdmFuncRebFxd is None:
            dvdmFuncRebFxd = NullFunc()
        if dvdnFuncRebFxd is None:
            dvdnFuncRebFxd = NullFunc()
        if dvdsFuncRebFxd is None:
            dvdsFuncRebFxd = NullFunc()

        # Components of the adjusting problem
        self.vFuncRebAdj = vFuncRebAdj
        self.DFuncAdj = DFuncAdj
        self.dvdmFuncRebAdj = dvdmFuncRebAdj
        self.dvdnFuncRebAdj = dvdnFuncRebAdj

        # Components of the fixed problem
        self.vFuncRebFxd = vFuncRebFxd
        self.DFuncFxd = DFuncFxd
        self.dvdmFuncRebFxd = dvdmFuncRebFxd
        self.dvdnFuncRebFxd = dvdnFuncRebFxd
        self.dvdsFuncRebFxd = dvdsFuncRebFxd


# Class for the contribution share stage solution
class RiskyContribShaSolution(MetricObject):
    """
    A class for representing the solution to the contribution-share stage of
    the 'RiskyContrib' model.
    
    Parameters
    ----------
    vFuncShaAdj : ValueFunc2D
        Stage value function over normalized liquid resources and normalized
        iliquid resources when the agent is able to adjust his portfolio.
    ShareFuncAdj : Interp2D
        Income contribution share function over normalized liquid resources
        and normalized iliquid resources when the agent is able to adjust his
        portfolio.
    dvdmFuncShaAdj : MargValueFunc2D
        Marginal value function over normalized liquid resources when the agent
        is able to adjust his portfolio.
    dvdnFuncShaAdj : MargValueFunc2D
        Marginal value function over normalized iliquid resources when the
        agent is able to adjust his portfolio.
    vFuncShaFxd : ValueFunc3D
        Stage value function over normalized liquid resources, normalized
        iliquid resources, and income contribution share when the agent is not
        able to adjust his portfolio.
    ShareFuncFxd : Interp3D
        Income contribution share function over normalized liquid resources,
        iliquid resources, and income contribution share when the agent is not
        able to adjust his portfolio.
        Should be an IdentityFunc.
    dvdmFuncShaFxd : MargValueFunc3D
        Marginal value function over normalized liquid resources when the agent
        is not able to adjust his portfolio.
    dvdnFuncShaFxd : MargValueFunc3D
        Marginal value function over normalized iliquid resources when the
        agent is not able to adjust his portfolio.
    dvdsFuncShaFxd : Interp3D
        Marginal value function over income contribution share when the agent
        is not able to adjust his portfolio
    """

    distance_criteria = ["dvdmFuncShaAdj", "dvdnFuncShaAdj"]

    def __init__(
        self,
        # Contribution stage, adjust
        vFuncShaAdj=None,
        ShareFuncAdj=None,
        dvdmFuncShaAdj=None,
        dvdnFuncShaAdj=None,
        # Contribution stage, fixed
        vFuncShaFxd=None,
        ShareFuncFxd=None,
        dvdmFuncShaFxd=None,
        dvdnFuncShaFxd=None,
        dvdsFuncShaFxd=None,
    ):

        # Contribution stage, adjust
        if vFuncShaAdj is None:
            vFuncShaAdj = NullFunc()
        if ShareFuncAdj is None:
            ShareFuncAdj = NullFunc()
        if dvdmFuncShaAdj is None:
            dvdmFuncShaAdj = NullFunc()
        if dvdnFuncShaAdj is None:
            dvdnFuncShaAdj = NullFunc()

        # Contribution stage, fixed
        if vFuncShaFxd is None:
            vFuncShaFxd = NullFunc()
        if ShareFuncFxd is None:
            ShareFuncFxd = NullFunc()
        if dvdmFuncShaFxd is None:
            dvdmFuncShaFxd = NullFunc()
        if dvdnFuncShaFxd is None:
            dvdnFuncShaFxd = NullFunc()
        if dvdsFuncShaFxd is None:
            dvdsFuncShaFxd = NullFunc()

        # Set attributes of self
        self.vFuncShaAdj = vFuncShaAdj
        self.ShareFuncAdj = ShareFuncAdj
        self.dvdmFuncShaAdj = dvdmFuncShaAdj
        self.dvdnFuncShaAdj = dvdnFuncShaAdj

        self.vFuncShaFxd = vFuncShaFxd
        self.ShareFuncFxd = ShareFuncFxd
        self.dvdmFuncShaFxd = dvdmFuncShaFxd
        self.dvdnFuncShaFxd = dvdnFuncShaFxd
        self.dvdsFuncShaFxd = dvdsFuncShaFxd


# Class for the consumption stage solution
class RiskyContribCnsSolution(MetricObject):
    """
    A class for representing the solution to the consumption stage of the
    'RiskyContrib' model.
    
    Parameters
    ----------
    vFuncCns : ValueFunc3D
        Stage-value function over normalized liquid resources, normalized
        iliquid resources, and income contribution share.
    cFunc : Interp3D
        Consumption function over normalized liquid resources, normalized
        iliquid resources, and income contribution share.
    dvdmFuncCns : MargValueFunc3D
        Marginal value function over normalized liquid resources.
    dvdnFuncCns : MargValueFunc3D
        Marginal value function over normalized iliquid resources.
    dvdsFuncCns : Interp3D
        Marginal value function over income contribution share.
    """

    distance_criteria = ["dvdmFuncCns", "dvdnFuncCns"]

    def __init__(
        self,
        # Consumption stage
        vFuncCns=None,
        cFunc=None,
        dvdmFuncCns=None,
        dvdnFuncCns=None,
        dvdsFuncCns=None,
    ):

        if vFuncCns is None:
            vFuncCns = NullFunc()
        if cFunc is None:
            cFunc = NullFunc()
        if dvdmFuncCns is None:
            dvdmFuncCns = NullFunc()
        if dvdnFuncCns is None:
            dvdmFuncCns = NullFunc()
        if dvdsFuncCns is None:
            dvdsFuncCns = NullFunc()

        self.vFuncCns = vFuncCns
        self.cFunc = cFunc
        self.dvdmFuncCns = dvdmFuncCns
        self.dvdnFuncCns = dvdnFuncCns
        self.dvdsFuncCns = dvdsFuncCns


# Class for the solution of a whole period
class RiskyContribSolution(MetricObject):
    """
    A class for representing the solution to a full time-period of the
    'RiskyContrib' agent type's problem.
    
    Parameters
    ----------
    Reb : RiskyContribRebSolution
        Solution to the period's rebalancing stage.
    Sha : RiskyContribShaSolution
        Solution to the period's contribution-share stage.
    Cns : RiskyContribCnsSolution
        Solution to the period's consumption stage.
    """

    # Solutions are to be compared on the basis of their sub-period solutions
    distance_criteria = ["stageSols"]

    def __init__(self, Reb, Sha, Cns):

        # Dictionary of stage solutions
        self.stageSols = {"Reb": Reb, "Sha": Sha, "Cns": Cns}


# %% Auxiliary functions and transition equations for the RiskyContrib model.


def rebalance_assets(d, m, n, tau):
    """
    A function that produces post-rebalancing assets for given initial assets,
    rabalancing action, and tax rate.

    Parameters
    ----------
    d : np.array
        Array with rebalancing decisions. d > 0 represents depositing d*m into
        the risky asset account. d<0 represents withdrawing |d|*n (pre-tax)
        from the risky account into the risky account.
    m : np.array
        Initial risk-free assets.
    n : np.array
        Initial risky assets.
    tau : float
        Tax rate on flows from the risky to the risk-free asset.

    Returns
    -------
    mTil : np.array
        Post-rebalancing risk-free assets.
    nTil : np.arrat
        Post-rebalancing risky assets.

    """
    # Initialize
    mTil = np.zeros_like(m) + np.nan
    nTil = np.zeros_like(m) + np.nan

    # Contributions
    inds = d >= 0
    mTil[inds] = m[inds] * (1 - d[inds])
    nTil[inds] = n[inds] + m[inds] * d[inds]

    # Withdrawals
    inds = d < 0
    mTil[inds] = m[inds] - d[inds] * n[inds] * (1 - tau)
    nTil[inds] = n[inds] * (1 + d[inds])

    return (mTil, nTil)


# Transition equations for the consumption stage
def m_nrm_next(shocks, aNrm, Share, Rfree, PermGroFac):
    """
    Given end-of-period balances and contribution share and the
    start-of-next-period shocks, figure out next period's normalized riskless
    assets

    Parameters
    ----------
    shocks : np.array
        Length-3 array with the stochastic shocks that get realized between the
        end of the current period and the start of next period. Their order is
        (0) permanent income shock, (1) transitory income shock, (2) risky
        asset return.
    aNrm : float
        End-of-period risk-free asset balances.
    Share : float
        End-of-period income deduction share.
    Rfree : float
        Risk-free return factor.
    PermGroFac : float
        Permanent income growth factor.

    Returns
    -------
    m_nrm_tp1 : float
        Next-period normalized riskless balance.

    """
    # Extract shocks
    perm_shk = shocks[0]
    tran_shk = shocks[1]

    m_nrm_tp1 = Rfree * aNrm / (perm_shk * PermGroFac) + (1.0 - Share) * tran_shk

    return m_nrm_tp1


def n_nrm_next(shocks, nNrm, Share, PermGroFac):
    """
    Given end-of-period balances and contribution share and the
    start-of-next-period shocks, figure out next period's normalized risky
    assets

    Parameters
    ----------
    shocks : np.array
        Length-3 array with the stochastic shocks that get realized between the
        end of the current period and the start of next period. Their order is
        (0) permanent income shock, (1) transitory income shock, (2) risky
        asset return.
    nNrm : float
        End-of-period risky asset balances.
    Share : float
        End-of-period income deduction share.
    PermGroFac : float
        Permanent income growth factor.

    Returns
    -------
    n_nrm_tp1 : float
        Next-period normalized risky balance.

    """

    # Extract shocks
    perm_shk = shocks[0]
    tran_shk = shocks[1]
    R_risky = shocks[2]

    n_nrm_tp1 = R_risky * nNrm / (perm_shk * PermGroFac) + Share * tran_shk

    return n_nrm_tp1


def end_of_period_derivs(
    shocks,
    a,
    nTil,
    s,
    dvdm_next,
    dvdn_next,
    dvds_next,
    CRRA,
    PermGroFac,
    Rfree,
    DiscFac,
    LivPrb,
    v_next=None,
):
    """
    While solving his problem in a given period, the agent must estimate the
    expected continuation value of ending the current period in a given state.
    Most importantly, he must know the derivatives of that function.
    This method is a step in computing these expected values. It computes the
    end-of-period derivatives (and optionally the value) of the continuation
    function, conditional on shocks. This is so that the expectations can be
    calculated later by integrating over shocks.

    Parameters
    ----------
    shocks : np.array
        Length-3 array with the stochastic shocks that get realized between the
        end of the current period and the start of next period. Their order is
        (0) permanent income shock, (1) transitory income shock, (2) risky
        asset return.
    a : float
        end-of-period risk-free assets.
    nTil : float
        end-of-period risky assets.
    s : float
        end-of-period income deduction share.
    dvdm_next : 3D function
        Next-period's marginal value of riskless assets.
    dvdn_next : 3D function
        Next-period's marginal value of risky assets.
    dvds_next : 3D function
        Next-period's marginal value of income-deduction share.
    CRRA : float
        Coefficient of relative risk aversion.
    PermGroFac : float
        Permanent income deterministic growth factor.
    Rfree : float
        Risk-free return factor.
    DiscFac : float
        Time-preference discount factor.
    LivPrb : float
        Survival probability.
    v_next : 3D function, optional
        Next-period's value function. The default is None.

    Returns
    -------
    np.array
        Array with end-of-period-value derivatives conditional on next
        period's shocks. Order
        (0) riskless assets, (1) risky assets, (2) income deduction share
        Optionally, the level of end-of-period value is added in position (3).

    """

    temp_fac_A = utilityP(shocks[0] * PermGroFac, CRRA)
    temp_fac_B = (shocks[0] * PermGroFac) ** (1.0 - CRRA)

    # Find next-period asset balances
    m_next = m_nrm_next(shocks, a, s, Rfree, PermGroFac)
    n_next = n_nrm_next(shocks, nTil, s, PermGroFac)

    # Interpolate next-period-value derivatives
    dvdm_tp1 = dvdm_next(m_next, n_next, s)
    dvdn_tp1 = dvdn_next(m_next, n_next, s)
    if shocks[1] == 0:
        dvds_tp1 = dvds_next(m_next, n_next, s)
    else:
        dvds_tp1 = shocks[1] * (dvdn_tp1 - dvdm_tp1) + dvds_next(m_next, n_next, s)

    # Discount next-period-value derivatives to current period

    # Liquid resources
    end_of_prd_dvda = DiscFac * Rfree * LivPrb * temp_fac_A * dvdm_tp1
    # Iliquid resources
    end_of_prd_dvdn = DiscFac * shocks[2] * LivPrb * temp_fac_A * dvdn_tp1
    # Contribution share
    end_of_prd_dvds = DiscFac * LivPrb * temp_fac_B * dvds_tp1

    # End of period value function, if needed
    if v_next is not None:
        end_of_prd_v = DiscFac * LivPrb * temp_fac_B * v_next(m_next, n_next, s)
        return np.stack(
            [end_of_prd_dvda, end_of_prd_dvdn, end_of_prd_dvds, end_of_prd_v]
        )
    else:
        return np.stack([end_of_prd_dvda, end_of_prd_dvdn, end_of_prd_dvds])


# %% RiskyContrib solvers

# Consumption stage solver
def solveRiskyContribCnsStage(
    solution_next,
    ShockDstn,
    LivPrb,
    DiscFac,
    CRRA,
    Rfree,
    PermGroFac,
    BoroCnstArt,
    aXtraGrid,
    nNrmGrid,
    mNrmGrid,
    ShareGrid,
    vFuncBool,
    AdjustPrb,
    DiscreteShareBool,
    **unused_params
):

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

    # Define temporary functions for utility and its derivative and inverse
    u = lambda x: utility(x, CRRA)
    uPinv = lambda x: utilityP_inv(x, CRRA)
    uInv = lambda x: utility_inv(x, CRRA)

    # Unpack next period's solution
    vFuncRebAdj_next = solution_next.vFuncRebAdj
    dvdmFuncRebAdj_next = solution_next.dvdmFuncRebAdj
    dvdnFuncRebAdj_next = solution_next.dvdnFuncRebAdj

    vFuncRebFxd_next = solution_next.vFuncRebFxd
    dvdmFuncRebFxd_next = solution_next.dvdmFuncRebFxd
    dvdnFuncRebFxd_next = solution_next.dvdnFuncRebFxd
    dvdsFuncRebFxd_next = solution_next.dvdsFuncRebFxd

    # STEP ONE
    # Find end-of-period (continuation) value function and its derivatives.

    # It's possible for the agent to end with 0 iliquid assets regardless of
    # future income and probability of adjustment.
    nNrmGrid = np.concatenate([np.array([0.0]), nNrmGrid])

    # Now, under which parameters do we need to consider the possibility
    # of the agent ending with 0 liquid assets?:
    # -If he has guaranteed positive income next period.
    # -If he is sure he can draw on iliquid assets even if income and liquid
    #  assets are 0.
    # If none of these is true, he will not allow his end-of-period liquid
    # assets to be 0
    TranShks_next = ShockDstn.X[1]
    zero_bound = np.min(TranShks_next) == 0.0
    if (not zero_bound) or (zero_bound and AdjustPrb == 1.0):
        aNrmGrid = np.insert(aXtraGrid, 0, 0.0)
    else:
        # aNrmGrid = aXtraGrid
        aNrmGrid = np.insert(aXtraGrid, 0, 0.0)

    # Create tiled arrays with conforming dimensions. These are used
    # to compute expectations at every grid combinations
    # Convention will be (a,n,s)
    aNrm_tiled, nNrm_tiled, Share_tiled = np.meshgrid(
        aNrmGrid, nNrmGrid, ShareGrid, indexing="ij"
    )

    # Evaluate realizations of the derivatives and levels of next period's
    # value function

    # The agent who can adjust starts at the "contrib" stage, the one who can't
    # starts at the Fxd stage.

    # We are interested in marginal values before the realization of the
    # adjustment random variable. Compute those objects
    if AdjustPrb < 1.0:

        dvdm_next = lambda m, n, s: AdjustPrb * dvdmFuncRebAdj_next(m, n) + (
            1.0 - AdjustPrb
        ) * dvdmFuncRebFxd_next(m, n, s)
        dvdn_next = lambda m, n, s: AdjustPrb * dvdnFuncRebAdj_next(m, n) + (
            1.0 - AdjustPrb
        ) * dvdnFuncRebFxd_next(m, n, s)
        dvds_next = lambda m, n, s: (1.0 - AdjustPrb) * dvdsFuncRebFxd_next(m, n, s)

        # Value function if needed
        if vFuncBool:

            v_next = lambda m, n, s: AdjustPrb * vFuncRebAdj_next(m, n) + (
                1.0 - AdjustPrb
            ) * vFuncRebFxd_next(m, n, s)

    else:

        dvdm_next = lambda m, n, s: dvdmFuncRebAdj_next(m, n)
        dvdn_next = lambda m, n, s: dvdnFuncRebAdj_next(m, n)
        dvds_next = ConstantFunction(0.0)

        if vFuncBool:
            v_next = lambda m, n, s: vFuncRebAdj_next(m, n)

    # Find end of period derivatives and value as discounted expectations of
    # next period's derivatives and value.
    # Create a function to recover the derivatives (and possible value) of the
    # end of period value function conditional on states and shocks.
    end_of_period_ds_func = lambda shocks, a, n, s: end_of_period_derivs(
        shocks,
        a,
        n,
        s,
        dvdm_next,
        dvdn_next,
        dvds_next,
        CRRA,
        PermGroFac,
        Rfree,
        DiscFac,
        LivPrb,
        v_next=v_next if vFuncBool else None,
    )

    # Then integrate over shocks
    EndOfPrd_derivs = calc_expectation(
        ShockDstn, end_of_period_ds_func, aNrm_tiled, nNrm_tiled, Share_tiled
    )[:, :, :, :, 0]

    # Unpack results
    EndOfPrddvdaNvrs = uPinv(EndOfPrd_derivs[0])
    EndOfPrddvdnNvrs = uPinv(EndOfPrd_derivs[1])
    EndOfPrddvds = EndOfPrd_derivs[2]
    if vFuncBool:
        EndOfPrdvNvrs = uInv(EndOfPrd_derivs[3])

        # Construct an interpolator for EndOfPrdV. It will be used later.
        EndOfPrdvFunc = ValueFuncCRRA(
            TrilinearInterp(EndOfPrdvNvrs, aNrmGrid, nNrmGrid, ShareGrid), CRRA
        )

    # STEP TWO:
    # Solve the consumption problem and create interpolators for c, vCns,
    # and its derivatives.

    # Apply EGM over liquid resources at every (n,s) to find consumption.
    c_end = EndOfPrddvdaNvrs
    mNrm_end = aNrm_tiled + c_end

    # Now construct interpolators for c and the derivatives of vCns.
    # The m grid is different for every (n,s). We interpolate the object of
    # interest on the regular m grid for every (n,s). At the end we will have
    # values of the functions of interest on a regular (m,n,s) grid. We use
    # trilinear interpolation on those points.

    # Expand the regular m grid to contain 0.
    mNrmGrid = np.insert(mNrmGrid, 0, 0)

    # Dimensions might have changed, so re-create tiled arrays
    mNrm_tiled, nNrm_tiled, Share_tiled = np.meshgrid(
        mNrmGrid, nNrmGrid, ShareGrid, indexing="ij"
    )

    # Initialize arrays
    c_vals = np.zeros_like(mNrm_tiled)
    dvdnNvrs_vals = np.zeros_like(mNrm_tiled)
    dvds_vals = np.zeros_like(mNrm_tiled)

    nNrm_N = nNrmGrid.size
    Share_N = ShareGrid.size
    for nInd in range(nNrm_N):
        for sInd in range(Share_N):

            # Extract the endogenous m grid for particular (n,s).
            m_ns = mNrm_end[:, nInd, sInd]

            # Check if there is a natural constraint
            if m_ns[0] == 0.0:

                # There's no need to insert points since we have m==0.0

                # c
                c_vals[:, nInd, sInd] = LinearInterp(m_ns, c_end[:, nInd, sInd])(
                    mNrmGrid
                )

                # dvdnNvrs
                dvdnNvrs_vals[:, nInd, sInd] = LinearInterp(
                    m_ns, EndOfPrddvdnNvrs[:, nInd, sInd]
                )(mNrmGrid)

                # dvds
                dvds_vals[:, nInd, sInd] = LinearInterp(
                    m_ns, EndOfPrddvds[:, nInd, sInd]
                )(mNrmGrid)

            else:

                # We know that:
                # -The lowest gridpoints of both a and n are 0.
                # -Consumption at m < m0 is m.
                # -dvdnFxd at (m,n) for m < m0(n) is dvdnFxd(m0,n)
                # -Same is true for dvdsFxd

                m_ns = np.concatenate([np.array([0]), m_ns])

                # c
                c_vals[:, nInd, sInd] = LinearInterp(
                    m_ns, np.concatenate([np.array([0]), c_end[:, nInd, sInd]])
                )(mNrmGrid)

                # dvdnNvrs
                dvdnNvrs_vals[:, nInd, sInd] = LinearInterp(
                    m_ns,
                    np.concatenate(
                        [
                            np.array([EndOfPrddvdnNvrs[0, nInd, sInd]]),
                            EndOfPrddvdnNvrs[:, nInd, sInd],
                        ]
                    ),
                )(mNrmGrid)

                # dvds
                dvds_vals[:, nInd, sInd] = LinearInterp(
                    m_ns,
                    np.concatenate(
                        [
                            np.array([EndOfPrddvds[0, nInd, sInd]]),
                            EndOfPrddvds[:, nInd, sInd],
                        ]
                    ),
                )(mNrmGrid)

    # With the arrays filled, create 3D interpolators

    # Consumption interpolator
    cFunc = TrilinearInterp(c_vals, mNrmGrid, nNrmGrid, ShareGrid)
    # dvdmCns interpolator
    dvdmFuncCns = MargValueFuncCRRA(cFunc, CRRA)
    # dvdnCns interpolator
    dvdnNvrsFunc = TrilinearInterp(dvdnNvrs_vals, mNrmGrid, nNrmGrid, ShareGrid)
    dvdnFuncCns = MargValueFuncCRRA(dvdnNvrsFunc, CRRA)
    # dvdsCns interpolator
    dvdsFuncCns = TrilinearInterp(dvds_vals, mNrmGrid, nNrmGrid, ShareGrid)

    # Compute value function if needed
    if vFuncBool:
        # Consumption in the regular grid
        aNrm_reg = mNrm_tiled - c_vals
        vCns = u(c_vals) + EndOfPrdvFunc(aNrm_reg, nNrm_tiled, Share_tiled)
        vNvrsCns = uInv(vCns)
        vNvrsFuncCns = TrilinearInterp(vNvrsCns, mNrmGrid, nNrmGrid, ShareGrid)
        vFuncCns = ValueFuncCRRA(vNvrsFuncCns, CRRA)
    else:
        vFuncCns = NullFunc()

    # Assemble solution
    solution = RiskyContribCnsSolution(
        vFuncCns=vFuncCns,
        cFunc=cFunc,
        dvdmFuncCns=dvdmFuncCns,
        dvdnFuncCns=dvdnFuncCns,
        dvdsFuncCns=dvdsFuncCns,
    )

    return solution


# Solver for the contribution stage
def solveRiskyContribShaStage(
    solution_next,
    CRRA,
    AdjustPrb,
    mNrmGrid,
    nNrmGrid,
    ShareGrid,
    DiscreteShareBool,
    vFuncBool,
    **unused_params
):

    # Unpack solution from the next sub-stage
    vFuncCns_next = solution_next.vFuncCns
    cFunc_next = solution_next.cFunc
    dvdmFuncCns_next = solution_next.dvdmFuncCns
    dvdnFuncCns_next = solution_next.dvdnFuncCns
    dvdsFuncCns_next = solution_next.dvdsFuncCns

    uPinv = lambda x: utilityP_inv(x, CRRA)

    # Create tiled grids

    # Add 0 to the m and n grids
    nNrmGrid = np.concatenate([np.array([0.0]), nNrmGrid])
    nNrm_N = len(nNrmGrid)
    mNrmGrid = np.concatenate([np.array([0.0]), mNrmGrid])
    mNrm_N = len(mNrmGrid)

    if AdjustPrb == 1.0:
        # If the readjustment probability is 1, set the share to 0:
        # - If there is a withdrawal tax: better for the agent to observe
        #   income before rebalancing.
        # - If there is no tax: all shares should yield the same value.
        mNrm_tiled, nNrm_tiled = np.meshgrid(mNrmGrid, nNrmGrid, indexing="ij")

        optIdx = np.zeros_like(mNrm_tiled, dtype=int)
        optShare = ShareGrid[optIdx]

        if vFuncBool:
            vNvrsSha = vFuncCns_next.func(mNrm_tiled, nNrm_tiled, optShare)

    else:

        # Figure out optimal share by evaluating all alternatives at all
        # (m,n) combinations
        m_idx_tiled, n_idx_tiled = np.meshgrid(
            np.arange(mNrm_N), np.arange(nNrm_N), indexing="ij"
        )

        mNrm_tiled, nNrm_tiled, Share_tiled = np.meshgrid(
            mNrmGrid, nNrmGrid, ShareGrid, indexing="ij"
        )

        if DiscreteShareBool:

            # Evaluate value function to optimize over shares.
            # Do it in inverse space
            vNvrs = vFuncCns_next.func(mNrm_tiled, nNrm_tiled, Share_tiled)

            # Find the optimal share at each (m,n).
            optIdx = np.argmax(vNvrs, axis=2)

            # Compute objects needed for the value function and its derivatives
            vNvrsSha = vNvrs[m_idx_tiled, n_idx_tiled, optIdx]
            optShare = ShareGrid[optIdx]

            # Project grids
            mNrm_tiled = mNrm_tiled[:, :, 0]
            nNrm_tiled = nNrm_tiled[:, :, 0]

        else:

            # Evaluate the marginal value of the contribution share at
            # every (m,n,s) gridpoint
            dvds = dvdsFuncCns_next(mNrm_tiled, nNrm_tiled, Share_tiled)

            # If the derivative is negative at the lowest share, then s[0] is optimal
            constrained_bot = dvds[:, :, 0] <= 0.0
            # If it is poitive at the highest share, then s[-1] is optimal
            constrained_top = dvds[:, :, -1] >= 0.0

            # Find indices at which the derivative crosses 0 for the 1st time
            # will be 0 if it never does, but "constrained_top/bot" deals with that
            crossings = np.logical_and(dvds[:, :, :-1] >= 0.0, dvds[:, :, 1:] <= 0.0)
            idx = np.argmax(crossings, axis=2)

            # Linearly interpolate the optimal share
            idx1 = idx + 1
            slopes = (
                dvds[m_idx_tiled, n_idx_tiled, idx1]
                - dvds[m_idx_tiled, n_idx_tiled, idx]
            ) / (ShareGrid[idx1] - ShareGrid[idx])
            optShare = ShareGrid[idx] - dvds[m_idx_tiled, n_idx_tiled, idx] / slopes

            # Replace the ones we knew were constrained
            optShare[constrained_bot] = ShareGrid[0]
            optShare[constrained_top] = ShareGrid[-1]

            # Project grids
            mNrm_tiled = mNrm_tiled[:, :, 0]
            nNrm_tiled = nNrm_tiled[:, :, 0]

            # Evaluate the inverse value function at the optimal shares
            if vFuncBool:
                vNvrsSha = vFuncCns_next.func(mNrm_tiled, nNrm_tiled, optShare)

    dvdmNvrsSha = cFunc_next(mNrm_tiled, nNrm_tiled, optShare)
    dvdnSha = dvdnFuncCns_next(mNrm_tiled, nNrm_tiled, optShare)
    dvdnNvrsSha = uPinv(dvdnSha)

    # Interpolators

    # Value function if needed
    if vFuncBool:
        vNvrsFuncSha = BilinearInterp(vNvrsSha, mNrmGrid, nNrmGrid)
        vFuncSha = ValueFuncCRRA(vNvrsFuncSha, CRRA)
    else:
        vFuncSha = NullFunc()

    # Contribution share function
    if DiscreteShareBool:
        ShareFunc = DiscreteInterp(
            BilinearInterp(optIdx, mNrmGrid, nNrmGrid), ShareGrid
        )
    else:
        ShareFunc = BilinearInterp(optShare, mNrmGrid, nNrmGrid)

    # Derivatives
    dvdmNvrsFuncSha = BilinearInterp(dvdmNvrsSha, mNrmGrid, nNrmGrid)
    dvdmFuncSha = MargValueFuncCRRA(dvdmNvrsFuncSha, CRRA)
    dvdnNvrsFuncSha = BilinearInterp(dvdnNvrsSha, mNrmGrid, nNrmGrid)
    dvdnFuncSha = MargValueFuncCRRA(dvdnNvrsFuncSha, CRRA)

    solution = RiskyContribShaSolution(
        vFuncShaAdj=vFuncSha,
        ShareFuncAdj=ShareFunc,
        dvdmFuncShaAdj=dvdmFuncSha,
        dvdnFuncShaAdj=dvdnFuncSha,
        # The fixed agent does nothing at this stage,
        # so his value functions are the next problem's
        vFuncShaFxd=vFuncCns_next,
        ShareFuncFxd=IdentityFunction(i_dim=2, n_dims=3),
        dvdmFuncShaFxd=dvdmFuncCns_next,
        dvdnFuncShaFxd=dvdnFuncCns_next,
        dvdsFuncShaFxd=dvdsFuncCns_next,
    )

    return solution


# Solver for the asset rebalancing stage
def solveRiskyContribRebStage(
    solution_next, CRRA, tau, nNrmGrid, mNrmGrid, dGrid, vFuncBool, **unused_params
):

    # Extract next stage's solution
    vFuncAdj_next = solution_next.vFuncShaAdj
    dvdmFuncAdj_next = solution_next.dvdmFuncShaAdj
    dvdnFuncAdj_next = solution_next.dvdnFuncShaAdj

    vFuncFxd_next = solution_next.vFuncShaFxd
    dvdmFuncFxd_next = solution_next.dvdmFuncShaFxd
    dvdnFuncFxd_next = solution_next.dvdnFuncShaFxd
    dvdsFuncFxd_next = solution_next.dvdsFuncShaFxd

    uPinv = lambda x: utilityP_inv(x, CRRA)

    # Create tiled grids

    # Add 0 to the m and n grids
    nNrmGrid = np.concatenate([np.array([0.0]), nNrmGrid])
    nNrm_N = len(nNrmGrid)
    mNrmGrid = np.concatenate([np.array([0.0]), mNrmGrid])
    mNrm_N = len(mNrmGrid)
    d_N = len(dGrid)

    # Duplicate d so that possible values are -dGrid,dGrid. Duplicate 0 is
    # intentional since the tax causes a discontinuity. We need the value
    # from the left and right.
    dGrid = np.concatenate((-1 * np.flip(dGrid), dGrid))

    # It will be useful to pre-evaluate marginals at every (m,n,d) combination

    # Create tiled arrays for every d,m,n option
    d_N2 = len(dGrid)
    d_tiled, mNrm_tiled, nNrm_tiled = np.meshgrid(
        dGrid, mNrmGrid, nNrmGrid, indexing="ij"
    )

    # Get post-rebalancing assets.
    m_tilde, n_tilde = rebalance_assets(d_tiled, mNrm_tiled, nNrm_tiled, tau)

    # Now the marginals, in inverse space
    dvdmNvrs = dvdmFuncAdj_next.cFunc(m_tilde, n_tilde)
    dvdnNvrs = dvdnFuncAdj_next.cFunc(m_tilde, n_tilde)

    # Pre-evaluate the inverse of (1-tau)
    taxNvrs = uPinv(1 - tau)
    # Create a tiled array of the tax
    taxNvrs_tiled = np.tile(
        np.reshape(
            np.concatenate([np.repeat(taxNvrs, d_N), np.ones(d_N, dtype=np.double)]),
            (d_N2, 1, 1),
        ),
        (1, mNrm_N, nNrm_N),
    )

    # The FOC is dvdn = tax*dvdm or dvdnNvrs = taxNvrs*dvdmNvrs
    dvdDNvrs = dvdnNvrs - taxNvrs_tiled * dvdmNvrs
    # The optimal d will be at the first point where dvdD < 0. The inverse
    # transformation flips the sign.

    # If the derivative is negative (inverse positive) at the lowest d,
    # then d == -1.0 is optimal
    constrained_bot = dvdDNvrs[0, :, :] >= 0.0
    # If it is positive (inverse negative) at the highest d, then d[-1] = 1.0
    # is optimal
    constrained_top = dvdDNvrs[-1, :, :,] <= 0.0

    # Find indices at which the derivative crosses 0 for the 1st time
    # will be 0 if it never does, but "constrained_top/bot" deals with that
    crossings = np.logical_and(dvdDNvrs[:-1, :, :] <= 0.0, dvdDNvrs[1:, :, :] >= 0.0)
    idx = np.argmax(crossings, axis=0)

    m_idx_tiled, n_idx_tiled = np.meshgrid(
        np.arange(mNrm_N), np.arange(nNrm_N), indexing="ij"
    )

    # Linearly interpolate the optimal withdrawal percentage d
    idx1 = idx + 1
    slopes = (
        dvdDNvrs[idx1, m_idx_tiled, n_idx_tiled]
        - dvdDNvrs[idx, m_idx_tiled, n_idx_tiled]
    ) / (dGrid[idx1] - dGrid[idx])
    dOpt = dGrid[idx] - dvdDNvrs[idx, m_idx_tiled, n_idx_tiled] / slopes

    # Replace the ones we knew were constrained
    dOpt[constrained_bot] = dGrid[0]
    dOpt[constrained_top] = dGrid[-1]

    # Find m_tilde and n_tilde
    mtil_opt, ntil_opt = rebalance_assets(dOpt, mNrm_tiled[0], nNrm_tiled[0], tau)

    # Now the derivatives. These are not straight forward because of corner
    # solutions with partial derivatives that change the limits. The idea then
    # is to evaluate the possible uses of the marginal unit of resources and
    # take the maximum.

    # An additional unit of m
    marg_m = dvdmFuncAdj_next(mtil_opt, ntil_opt)
    # An additional unit of n kept in n
    marg_n = dvdnFuncAdj_next(mtil_opt, ntil_opt)
    # An additional unit of n withdrawn to m
    marg_n_to_m = marg_m * (1 - tau)

    # Marginal value is the maximum of the marginals in their possible uses
    dvdmAdj = np.maximum(marg_m, marg_n)
    dvdmNvrsAdj = uPinv(dvdmAdj)
    dvdnAdj = np.maximum(marg_n, marg_n_to_m)
    dvdnNvrsAdj = uPinv(dvdnAdj)

    # Interpolators

    # Value
    if vFuncBool:
        vNvrsAdj = vFuncAdj_next.func(mtil_opt, ntil_opt)
        vNvrsFuncAdj = BilinearInterp(vNvrsAdj, mNrmGrid, nNrmGrid)
        vFuncAdj = ValueFuncCRRA(vNvrsFuncAdj, CRRA)
    else:
        vFuncAdj = NullFunc()

    # Marginals
    dvdmFuncAdj = MargValueFuncCRRA(
        BilinearInterp(dvdmNvrsAdj, mNrmGrid, nNrmGrid), CRRA
    )
    dvdnFuncAdj = MargValueFuncCRRA(
        BilinearInterp(dvdnNvrsAdj, mNrmGrid, nNrmGrid), CRRA
    )

    # Decison
    DFuncAdj = BilinearInterp(dOpt, mNrmGrid, nNrmGrid)

    solution = RiskyContribRebSolution(
        # Rebalancing stage adjusting
        vFuncRebAdj=vFuncAdj,
        DFuncAdj=DFuncAdj,
        dvdmFuncRebAdj=dvdmFuncAdj,
        dvdnFuncRebAdj=dvdnFuncAdj,
        # Rebalancing stage fixed (nothing happens, so value functions are
        # the ones from the next stage)
        vFuncRebFxd=vFuncFxd_next,
        DFuncFxd=ConstantFunction(0.0),
        dvdmFuncRebFxd=dvdmFuncFxd_next,
        dvdnFuncRebFxd=dvdnFuncFxd_next,
        dvdsFuncRebFxd=dvdsFuncFxd_next,
    )

    return solution


def solveRiskyContrib(
    solution_next,
    ShockDstn,
    LivPrb,
    DiscFac,
    CRRA,
    Rfree,
    PermGroFac,
    tau,
    BoroCnstArt,
    aXtraGrid,
    nNrmGrid,
    mNrmGrid,
    ShareGrid,
    dGrid,
    vFuncBool,
    AdjustPrb,
    DiscreteShareBool,
    IndepDstnBool,
):

    # Pack parameters to be passed to stage-specific solvers
    kws = {
        "ShockDstn": ShockDstn,
        "LivPrb": LivPrb,
        "DiscFac": DiscFac,
        "CRRA": CRRA,
        "Rfree": Rfree,
        "PermGroFac": PermGroFac,
        "tau": tau,
        "BoroCnstArt": BoroCnstArt,
        "aXtraGrid": aXtraGrid,
        "nNrmGrid": nNrmGrid,
        "mNrmGrid": mNrmGrid,
        "ShareGrid": ShareGrid,
        "dGrid": dGrid,
        "vFuncBool": vFuncBool,
        "AdjustPrb": AdjustPrb,
        "DiscreteShareBool": DiscreteShareBool,
        "IndepDstnBool": IndepDstnBool,
    }

    # Stages of the problem in chronological order
    Stages = ["Reb", "Sha", "Cns"]
    n_stages = len(Stages)
    # Solvers, indexed by stage names
    Solvers = {
        "Reb": solveRiskyContribRebStage,
        "Sha": solveRiskyContribShaStage,
        "Cns": solveRiskyContribCnsStage,
    }

    # Initialize empty solution
    stageSols = {}
    # Solve stages backwards
    for i in reversed(range(n_stages)):
        stage = Stages[i]

        # In the last stage, the next solution is the first stage of the next
        # period. Otherwise, its the next stage of his period.
        if i == n_stages - 1:
            sol_next_stage = solution_next.stageSols[Stages[0]]
        else:
            sol_next_stage = stageSols[Stages[i + 1]]

        # Solve
        stageSols[stage] = Solvers[stage](sol_next_stage, **kws)

    # Assemble stage solutions into period solution
    periodSol = RiskyContribSolution(**stageSols)

    return periodSol

# %% Base risky-contrib dictionaries

risky_contrib_params = {
    # Preferences. The points of the model are more evident for more risk
    # averse and impatient agents
    "CRRA": 5.0,
    "DiscFac": 0.90,
    # Artificial borrowing constraint must be on
    "BoroCnstArt": 0.0,

    # Grids go up high wealth/P ratios and are less clustered at the bottom.
    "aXtraMax": 250,
    "aXtraCount": 50,
    "aXtraNestFac": 1,
    
    # Same goes for the new grids of the model
    "mNrmMin": 1e-6,
    "mNrmMax": 250,
    "mNrmCount": 50,
    "mNrmNestFac": 1,

    "nNrmMin": 1e-6,
    "nNrmMax": 250,
    "nNrmCount": 50,
    "nNrmNestFac": 1,
    
    # Income deduction/contribution share grid
    "ShareCount": 10,
    "ShareMax": 0.9,
    "DiscreteShareBool": False,
    
    # Grid for finding the optimal rebalancing flow
    "dCount": 20
}

# Infinite horizon version
init_risky_contrib = init_risky_asset.copy()
init_risky_contrib.update(risky_contrib_params)

# Lifecycle version
init_risky_contrib_lifecycle = init_lifecycle.copy()
init_risky_contrib_lifecycle.update(risky_asset_parms)
init_risky_contrib_lifecycle.update(risky_contrib_params)
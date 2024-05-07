"""
This file contains classes and functions for representing, solving, and simulating
a consumer type with idiosyncratic shocks to permanent and transitory income,
who can save in both a risk-free and a risky asset but faces frictions to
moving funds between them. The agent can only consume out of his risk-free
asset.

The model is described in detail in the REMARK:
https://econ-ark.org/materials/riskycontrib

@software{mateo_velasquez_giraldo_2021_4977915,
  author       = {Mateo Velásquez-Giraldo},
  title        = {{Mv77/RiskyContrib: A Two-Asset Savings Model with
                   an Income-Contribution Scheme}},
  month        = jun,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v1.0.1},
  doi          = {10.5281/zenodo.4977915},
  url          = {https://doi.org/10.5281/zenodo.4977915}
}

"""
from copy import deepcopy

import numpy as np

from HARK import MetricObject, NullFunc  # Basic HARK features
from HARK.ConsumptionSaving.ConsIndShockModel import utility  # CRRA utility function
from HARK.ConsumptionSaving.ConsIndShockModel import (
    utility_inv,  # Inverse CRRA utility function
)
from HARK.ConsumptionSaving.ConsIndShockModel import (
    utilityP,  # CRRA marginal utility function
)
from HARK.ConsumptionSaving.ConsIndShockModel import (
    utilityP_inv,  # Inverse CRRA marginal utility function
)
from HARK.ConsumptionSaving.ConsIndShockModel import init_lifecycle
from HARK.ConsumptionSaving.ConsRiskyAssetModel import (
    RiskyAssetConsumerType,
    init_risky_asset,
    risky_asset_parms,
)
from HARK.distribution import calc_expectation
from HARK.interpolation import BilinearInterp  # 2D interpolator
from HARK.interpolation import (
    ConstantFunction,  # Interpolator-like class that returns constant value
)
from HARK.interpolation import (
    IdentityFunction,  # Interpolator-like class that returns one of its arguments
)
from HARK.interpolation import LinearInterp  # Piecewise linear interpolation
from HARK.interpolation import TrilinearInterp  # 3D interpolator
from HARK.interpolation import DiscreteInterp, MargValueFuncCRRA, ValueFuncCRRA
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
    time_inv_ = time_inv_ + ["DiscreteShareBool", "joint_dist_solver"]

    # The new state variables (over those in ConsIndShock) are:
    # - nMrm: start-of-period risky resources.
    # - mNrmTilde: post-rebalancing risk-free resources.
    # - nNrmTilde: post-rebalancing risky resources.
    # - Share: income-deduction share.
    # For details, see
    # https://github.com/Mv77/RiskyContrib/blob/main/RiskyContrib.pdf
    state_vars = RiskyAssetConsumerType.state_vars + [
        "gNrm",
        "nNrm",
        "mNrmTilde",
        "nNrmTilde",
        "Share",
    ]
    shock_vars_ = RiskyAssetConsumerType.shock_vars_

    def __init__(self, verbose=False, quiet=False, joint_dist_solver=False, **kwds):
        params = init_risky_contrib.copy()
        params.update(kwds)
        kwds = params

        # Initialize a basic consumer type
        RiskyAssetConsumerType.__init__(self, verbose=verbose, quiet=quiet, **kwds)

        # The model is solved and simulated spliting each of the agent's
        # decisions into its own "stage". The stages in chronological order
        # are
        # - Reb: asset-rebalancing stage.
        # - Sha: definition of the income contribution share.
        # - Cns: consumption stage.
        self.stages = ["Reb", "Sha", "Cns"]

        # Each stage has its own states and controls, and its methods
        # to find them.
        self.get_states = {
            "Reb": self.get_states_Reb,
            "Sha": self.get_states_Sha,
            "Cns": self.get_states_Cns,
        }

        self.get_controls = {
            "Reb": self.get_controls_Reb,
            "Sha": self.get_controls_Sha,
            "Cns": self.get_controls_Cns,
        }

        # The model can be solved more quickly if income and risky returns are
        # independent. However, people might want to use the general solver
        # even when they are independent for debugging and testing.
        self.joint_dist_solver = joint_dist_solver

        # Set the solver for the portfolio model, and update various constructed attributes
        self.solve_one_period = solveRiskyContrib
        self.update()

    def pre_solve(self):
        self.update_solution_terminal()

    def update(self):
        RiskyAssetConsumerType.update(self)
        self.update_share_grid()
        self.update_dfrac_grid()
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
        vFunc_Cns_term = ValueFuncCRRA(cFunc_term, CRRA=self.CRRA)
        # Marginal values
        dvdmFunc_Cns_term = MargValueFuncCRRA(cFunc_term, CRRA=self.CRRA)
        dvdnFunc_Cns_term = ConstantFunction(0.0)
        dvdsFunc_Cns_term = ConstantFunction(0.0)

        Cns_stage_sol = RiskyContribCnsSolution(
            # Consumption stage
            vFunc=vFunc_Cns_term,
            cFunc=cFunc_term,
            dvdmFunc=dvdmFunc_Cns_term,
            dvdnFunc=dvdnFunc_Cns_term,
            dvdsFunc=dvdsFunc_Cns_term,
        )

        # Share stage

        # It's irrelevant because there is no future period. Set share to 0.
        # Create a dummy 2-d consumption function to get value function and marginal
        c2d = IdentityFunction(i_dim=0, n_dims=2)
        Sha_stage_sol = RiskyContribShaSolution(
            # Adjust
            vFunc_Adj=ValueFuncCRRA(c2d, CRRA=self.CRRA),
            ShareFunc_Adj=ConstantFunction(0.0),
            dvdmFunc_Adj=MargValueFuncCRRA(c2d, CRRA=self.CRRA),
            dvdnFunc_Adj=ConstantFunction(0.0),
            # Fixed
            vFunc_Fxd=vFunc_Cns_term,
            ShareFunc_Fxd=IdentityFunction(i_dim=2, n_dims=3),
            dvdmFunc_Fxd=dvdmFunc_Cns_term,
            dvdnFunc_Fxd=dvdnFunc_Cns_term,
            dvdsFunc_Fxd=dvdsFunc_Cns_term,
        )

        # Rebalancing stage

        # Adjusting agent:
        # Withdraw everything from the pension fund and consume everything
        dfracFunc_Adj_term = ConstantFunction(-1.0)

        # Find the withdrawal penalty. If it is time-varying, assume it takes
        # the same value as in the last non-terminal period
        if type(self.tau) is list:
            tau = self.tau[-1]
        else:
            tau = self.tau

        # Value and marginal value function of the adjusting agent
        vFunc_Reb_Adj_term = ValueFuncCRRA(lambda m, n: m + n / (1 + tau), self.CRRA)
        dvdmFunc_Reb_Adj_term = MargValueFuncCRRA(
            lambda m, n: m + n / (1 + tau), self.CRRA
        )
        # A marginal unit of n will be withdrawn and put into m. Then consumed.
        dvdnFunc_Reb_Adj_term = lambda m, n: dvdmFunc_Reb_Adj_term(m, n) / (1 + tau)

        Reb_stage_sol = RiskyContribRebSolution(
            # Rebalancing stage
            vFunc_Adj=vFunc_Reb_Adj_term,
            dfracFunc_Adj=dfracFunc_Adj_term,
            dvdmFunc_Adj=dvdmFunc_Reb_Adj_term,
            dvdnFunc_Adj=dvdnFunc_Reb_Adj_term,
            # Adjusting stage
            vFunc_Fxd=vFunc_Cns_term,
            dfracFunc_Fxd=ConstantFunction(0.0),
            dvdmFunc_Fxd=dvdmFunc_Cns_term,
            dvdnFunc_Fxd=dvdnFunc_Cns_term,
            dvdsFunc_Fxd=dvdsFunc_Cns_term,
        )

        # Construct the terminal period solution
        self.solution_terminal = RiskyContribSolution(
            Reb_stage_sol, Sha_stage_sol, Cns_stage_sol
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

    def update_dfrac_grid(self):
        """
        Creates grid for the rebalancing flow between assets. This flow is
        normalized as a ratio.
        - If d > 0, d*mNrm flows towards the risky asset.
        - If d < 0, d*nNrm (pre-tax) flows towards the risk-free asset.

        Returns
        -------
        None.

        """
        self.dfracGrid = np.linspace(0, 1, self.dCount)
        self.add_to_time_inv("dfracGrid")

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
            self.get_states[s]()
            self.get_controls[s]()

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

        self.state_now["bNrm"] = RfEff * aNrmPrev  # Liquid balances before labor income
        self.state_now["gNrm"] = (
            RrEff * nNrmTildePrev
        )  # Iliquid balances before labor income

        # Liquid balances after labor income
        self.state_now["mNrm"] = self.state_now["bNrm"] + self.shocks["TranShk"] * (
            1 - SharePrev
        )
        # Iliquid balances after labor income
        self.state_now["nNrm"] = (
            self.state_now["gNrm"] + self.shocks["TranShk"] * SharePrev
        )

        return None

    def get_controls_Reb(self):
        """
        Get controls for the first stage: rebalancing
        """
        dfrac = np.zeros(self.AgentCount) + np.nan

        # Loop over each period of the cycle, getting controls separately depending on "age"
        for t in range(self.T_cycle):
            # Find agents in this period-stage
            these = t == self.t_cycle

            # Get controls for agents who *can* adjust.
            those = np.logical_and(these, self.shocks["Adjust"])
            dfrac[those] = (
                self.solution[t]
                .stage_sols["Reb"]
                .dfracFunc_Adj(
                    self.state_now["mNrm"][those], self.state_now["nNrm"][those]
                )
            )

            # Get Controls for agents who *can't* adjust.
            those = np.logical_and(these, np.logical_not(self.shocks["Adjust"]))
            dfrac[those] = (
                self.solution[t]
                .stage_sols["Reb"]
                .dfracFunc_Fxd(
                    self.state_now["mNrm"][those],
                    self.state_now["nNrm"][those],
                    self.state_prev["Share"][those],
                )
            )

        # Limit dfrac to [-1,1] to prevent negative balances. Values outside
        # the range can come from extrapolation.
        self.controls["dfrac"] = np.minimum(np.maximum(dfrac, -1), 1.0)

    def get_states_Sha(self):
        """
        Get states for the second "stage": choosing the contribution share.
        """

        # Post-states are assets after rebalancing

        if not "tau" in self.time_vary:
            mNrmTilde, nNrmTilde = rebalance_assets(
                self.controls["dfrac"],
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
                        self.controls["dfrac"][these],
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
                .stage_sols["Sha"]
                .ShareFunc_Adj(
                    self.state_now["mNrmTilde"][those],
                    self.state_now["nNrmTilde"][those],
                )
            )

            # Get Controls for agents who *can't* adjust.
            those = np.logical_and(these, np.logical_not(self.shocks["Adjust"]))
            Share[those] = (
                self.solution[t]
                .stage_sols["Sha"]
                .ShareFunc_Fxd(
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
                .stage_sols["Cns"]
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
    vFunc_Adj : ValueFunc2D
        Stage value function over normalized liquid resources and normalized
        iliquid resources when the agent is able to adjust his portfolio.
    dfracFunc_Adj : Interp2D
        Deposit function over normalized liquid resources and normalized
        iliquid resources when the agent is able to adjust his portfolio.
    dvdmFunc_Adj : MargValueFunc2D
        Marginal value over normalized liquid resources when the agent is able
        to adjust his portfolio.
    dvdnFunc_Adj : MargValueFunc2D
        Marginal value over normalized liquid resources when the agent is able
        to adjust his portfolio.
    vFunc_Fxd : ValueFunc3D
        Stage value function over normalized liquid resources, normalized
        iliquid resources, and income contribution share when the agent is
        not able to adjust his portfolio.
    dfracFunc_Fxd : Interp2D
        Deposit function over normalized liquid resources, normalized iliquid
        resources, and income contribution share when the agent is not able to
        adjust his portfolio.
        Must be ConstantFunction(0.0)
    dvdmFunc_Fxd : MargValueFunc3D
        Marginal value over normalized liquid resources when the agent is not
        able to adjust his portfolio.
    dvdnFunc_Fxd : MargValueFunc3D
        Marginal value over normalized iliquid resources when the agent is not
        able to adjust his portfolio.
    dvdsFunc_Fxd : Interp3D
        Marginal value function over income contribution share when the agent
        is not able to ajust his portfolio.
    """

    distance_criteria = ["dvdmFunc_Adj", "dvdnFunc_Adj"]

    def __init__(
        self,
        # Rebalancing stage, adjusting
        vFunc_Adj=None,
        dfracFunc_Adj=None,
        dvdmFunc_Adj=None,
        dvdnFunc_Adj=None,
        # Rebalancing stage, fixed
        vFunc_Fxd=None,
        dfracFunc_Fxd=None,
        dvdmFunc_Fxd=None,
        dvdnFunc_Fxd=None,
        dvdsFunc_Fxd=None,
    ):
        # Rebalancing stage
        if vFunc_Adj is None:
            vFunc_Adj = NullFunc()
        if dfracFunc_Adj is None:
            dfracFunc_Adj = NullFunc()
        if dvdmFunc_Adj is None:
            dvdmFunc_Adj = NullFunc()
        if dvdnFunc_Adj is None:
            dvdnFunc_Adj = NullFunc()

        if vFunc_Fxd is None:
            vFunc_Fxd = NullFunc()
        if dfracFunc_Fxd is None:
            dfracFunc_Fxd = NullFunc()
        if dvdmFunc_Fxd is None:
            dvdmFunc_Fxd = NullFunc()
        if dvdnFunc_Fxd is None:
            dvdnFunc_Fxd = NullFunc()
        if dvdsFunc_Fxd is None:
            dvdsFunc_Fxd = NullFunc()

        # Components of the adjusting problem
        self.vFunc_Adj = vFunc_Adj
        self.dfracFunc_Adj = dfracFunc_Adj
        self.dvdmFunc_Adj = dvdmFunc_Adj
        self.dvdnFunc_Adj = dvdnFunc_Adj

        # Components of the fixed problem
        self.vFunc_Fxd = vFunc_Fxd
        self.dfracFunc_Fxd = dfracFunc_Fxd
        self.dvdmFunc_Fxd = dvdmFunc_Fxd
        self.dvdnFunc_Fxd = dvdnFunc_Fxd
        self.dvdsFunc_Fxd = dvdsFunc_Fxd


# Class for the contribution share stage solution
class RiskyContribShaSolution(MetricObject):
    """
    A class for representing the solution to the contribution-share stage of
    the 'RiskyContrib' model.

    Parameters
    ----------
    vFunc_Adj : ValueFunc2D
        Stage value function over normalized liquid resources and normalized
        iliquid resources when the agent is able to adjust his portfolio.
    ShareFunc_Adj : Interp2D
        Income contribution share function over normalized liquid resources
        and normalized iliquid resources when the agent is able to adjust his
        portfolio.
    dvdmFunc_Adj : MargValueFunc2D
        Marginal value function over normalized liquid resources when the agent
        is able to adjust his portfolio.
    dvdnFunc_Adj : MargValueFunc2D
        Marginal value function over normalized iliquid resources when the
        agent is able to adjust his portfolio.
    vFunc_Fxd : ValueFunc3D
        Stage value function over normalized liquid resources, normalized
        iliquid resources, and income contribution share when the agent is not
        able to adjust his portfolio.
    ShareFunc_Fxd : Interp3D
        Income contribution share function over normalized liquid resources,
        iliquid resources, and income contribution share when the agent is not
        able to adjust his portfolio.
        Should be an IdentityFunc.
    dvdmFunc_Fxd : MargValueFunc3D
        Marginal value function over normalized liquid resources when the agent
        is not able to adjust his portfolio.
    dvdnFunc_Fxd : MargValueFunc3D
        Marginal value function over normalized iliquid resources when the
        agent is not able to adjust his portfolio.
    dvdsFunc_Fxd : Interp3D
        Marginal value function over income contribution share when the agent
        is not able to adjust his portfolio
    """

    distance_criteria = ["dvdmFunc_Adj", "dvdnFunc_Adj"]

    def __init__(
        self,
        # Contribution stage, adjust
        vFunc_Adj=None,
        ShareFunc_Adj=None,
        dvdmFunc_Adj=None,
        dvdnFunc_Adj=None,
        # Contribution stage, fixed
        vFunc_Fxd=None,
        ShareFunc_Fxd=None,
        dvdmFunc_Fxd=None,
        dvdnFunc_Fxd=None,
        dvdsFunc_Fxd=None,
    ):
        # Contribution stage, adjust
        if vFunc_Adj is None:
            vFunc_Adj = NullFunc()
        if ShareFunc_Adj is None:
            ShareFunc_Adj = NullFunc()
        if dvdmFunc_Adj is None:
            dvdmFunc_Adj = NullFunc()
        if dvdnFunc_Adj is None:
            dvdnFunc_Adj = NullFunc()

        # Contribution stage, fixed
        if vFunc_Fxd is None:
            vFunc_Fxd = NullFunc()
        if ShareFunc_Fxd is None:
            ShareFunc_Fxd = NullFunc()
        if dvdmFunc_Fxd is None:
            dvdmFunc_Fxd = NullFunc()
        if dvdnFunc_Fxd is None:
            dvdnFunc_Fxd = NullFunc()
        if dvdsFunc_Fxd is None:
            dvdsFunc_Fxd = NullFunc()

        # Set attributes of self
        self.vFunc_Adj = vFunc_Adj
        self.ShareFunc_Adj = ShareFunc_Adj
        self.dvdmFunc_Adj = dvdmFunc_Adj
        self.dvdnFunc_Adj = dvdnFunc_Adj

        self.vFunc_Fxd = vFunc_Fxd
        self.ShareFunc_Fxd = ShareFunc_Fxd
        self.dvdmFunc_Fxd = dvdmFunc_Fxd
        self.dvdnFunc_Fxd = dvdnFunc_Fxd
        self.dvdsFunc_Fxd = dvdsFunc_Fxd


# Class for the consumption stage solution
class RiskyContribCnsSolution(MetricObject):
    """
    A class for representing the solution to the consumption stage of the
    'RiskyContrib' model.

    Parameters
    ----------
    vFunc : ValueFunc3D
        Stage-value function over normalized liquid resources, normalized
        iliquid resources, and income contribution share.
    cFunc : Interp3D
        Consumption function over normalized liquid resources, normalized
        iliquid resources, and income contribution share.
    dvdmFunc : MargValueFunc3D
        Marginal value function over normalized liquid resources.
    dvdnFunc : MargValueFunc3D
        Marginal value function over normalized iliquid resources.
    dvdsFunc : Interp3D
        Marginal value function over income contribution share.
    """

    distance_criteria = ["dvdmFunc", "dvdnFunc"]

    def __init__(
        self,
        # Consumption stage
        vFunc=None,
        cFunc=None,
        dvdmFunc=None,
        dvdnFunc=None,
        dvdsFunc=None,
    ):
        if vFunc is None:
            vFunc = NullFunc()
        if cFunc is None:
            cFunc = NullFunc()
        if dvdmFunc is None:
            dvdmFunc = NullFunc()
        if dvdnFunc is None:
            dvdmFunc = NullFunc()
        if dvdsFunc is None:
            dvdsFunc = NullFunc()

        self.vFunc = vFunc
        self.cFunc = cFunc
        self.dvdmFunc = dvdmFunc
        self.dvdnFunc = dvdnFunc
        self.dvdsFunc = dvdsFunc


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
    distance_criteria = ["stage_sols"]

    def __init__(self, Reb, Sha, Cns):
        # Dictionary of stage solutions
        self.stage_sols = {"Reb": Reb, "Sha": Sha, "Cns": Cns}


# %% Auxiliary functions and transition equations for the RiskyContrib model.


def rebalance_assets(d, m, n, tau):
    """
    A function that produces post-rebalancing assets for given initial assets,
    rebalancing action, and tax rate.

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


# %% RiskyContrib solvers


# Consumption stage solver
def solve_RiskyContrib_Cns(
    solution_next,
    ShockDstn,
    IncShkDstn,
    RiskyDstn,
    IndepDstnBool,
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
    joint_dist_solver,
    **unused_params
):
    """
    Solves the consumption stage of the agent's problem

    Parameters
    ----------
    solution_next : RiskyContribRebSolution
        Solution to the first stage of the next period in the agent's problem.
    ShockDstn : DiscreteDistribution
        Joint distribution of next period's (0) permanent income shock, (1)
        transitory income shock, and (2) risky asset return factor.
    IncShkDstn : DiscreteDistribution
        Joint distribution of next period's (0) permanent income shock and (1)
        transitory income shock.
    RiskyDstn : DiscreteDistribution
        Distribution of next period's risky asset return factor.
    IndepDstnBool : bool
        Indicates whether the income and risky return distributions are
        independent.
    LivPrb : float
        Probability of surviving until next period.
    DiscFac : float
        Time-preference discount factor.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree : float
        Risk-free return factor.
    PermGroFac : float
        Deterministic permanent income growth factor.
    BoroCnstArt : float
        Minimum allowed market resources (must be 0).
    aXtraGrid : numpy array
        Exogenous grid for end-of-period risk free resources.
    nNrmGrid : numpy array
        Exogenous grid for risky resources.
    mNrmGrid : numpy array
        Exogenous grid for risk-free resources.
    ShareGrid : numpt array
        Exogenous grid for the income contribution share.
    vFuncBool : bool
        Boolean that determines wether the value function's level needs to be
        computed.
    AdjustPrb : float
        Probability thet the agent will be able to adjust his portfolio next
        period.
    DiscreteShareBool : bool
        Boolean that determines whether only a discrete set of contribution
        shares (ShareGrid) is allowed.
    joint_dist_solver: bool
        Should the general solver be used even if income and returns are
        independent?

    Returns
    -------
    solution : RiskyContribCnsSolution
        Solution to the agent's consumption stage problem.

    """
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
    vFunc_Reb_Adj_next = solution_next.vFunc_Adj
    dvdmFunc_Reb_Adj_next = solution_next.dvdmFunc_Adj
    dvdnFunc_Reb_Adj_next = solution_next.dvdnFunc_Adj

    vFunc_Reb_Fxd_next = solution_next.vFunc_Fxd
    dvdmFunc_Reb_Fxd_next = solution_next.dvdmFunc_Fxd
    dvdnFunc_Reb_Fxd_next = solution_next.dvdnFunc_Fxd
    dvdsFunc_Reb_Fxd_next = solution_next.dvdsFunc_Fxd

    # STEP ONE
    # Find end-of-period (continuation) value function and its derivatives.

    # Start by constructing functions for next-period's pre-adjustment-shock
    # expected value functions
    if AdjustPrb < 1.0:
        dvdm_next = lambda m, n, s: AdjustPrb * dvdmFunc_Reb_Adj_next(m, n) + (
            1.0 - AdjustPrb
        ) * dvdmFunc_Reb_Fxd_next(m, n, s)
        dvdn_next = lambda m, n, s: AdjustPrb * dvdnFunc_Reb_Adj_next(m, n) + (
            1.0 - AdjustPrb
        ) * dvdnFunc_Reb_Fxd_next(m, n, s)
        dvds_next = lambda m, n, s: (1.0 - AdjustPrb) * dvdsFunc_Reb_Fxd_next(m, n, s)

        # Value function if needed
        if vFuncBool:
            v_next = lambda m, n, s: AdjustPrb * vFunc_Reb_Adj_next(m, n) + (
                1.0 - AdjustPrb
            ) * vFunc_Reb_Fxd_next(m, n, s)

    else:
        dvdm_next = lambda m, n, s: dvdmFunc_Reb_Adj_next(m, n)
        dvdn_next = lambda m, n, s: dvdnFunc_Reb_Adj_next(m, n)
        dvds_next = ConstantFunction(0.0)

        if vFuncBool:
            v_next = lambda m, n, s: vFunc_Reb_Adj_next(m, n)

    if IndepDstnBool and not joint_dist_solver:
        # If income and returns are independent we can use the law of iterated
        # expectations to speed up the computation of end-of-period derivatives

        # Define "post-return variables"
        # b_aux = aNrm * R
        # g_aux = nNrmTilde * Rtilde
        # and create a function that interpolates end-of-period marginal values
        # as functions of those and the contribution share

        def post_return_derivs(inc_shocks, b_aux, g_aux, s):
            perm_shk = inc_shocks[0]
            tran_shk = inc_shocks[1]

            temp_fac_A = utilityP(perm_shk * PermGroFac, CRRA)
            temp_fac_B = (perm_shk * PermGroFac) ** (1.0 - CRRA)

            # Find next-period asset balances
            m_next = b_aux / (perm_shk * PermGroFac) + (1.0 - s) * tran_shk
            n_next = g_aux / (perm_shk * PermGroFac) + s * tran_shk

            # Interpolate next-period-value derivatives
            dvdm_tp1 = dvdm_next(m_next, n_next, s)
            dvdn_tp1 = dvdn_next(m_next, n_next, s)
            if tran_shk == 0:
                dvds_tp1 = dvds_next(m_next, n_next, s)
            else:
                dvds_tp1 = tran_shk * (dvdn_tp1 - dvdm_tp1) + dvds_next(
                    m_next, n_next, s
                )

            # Discount next-period-value derivatives to current period

            # Liquid resources
            pr_dvda = temp_fac_A * dvdm_tp1
            # Iliquid resources
            pr_dvdn = temp_fac_A * dvdn_tp1
            # Contribution share
            pr_dvds = temp_fac_B * dvds_tp1

            # End of period value function, if needed
            if vFuncBool:
                pr_v = temp_fac_B * v_next(m_next, n_next, s)
                return np.stack([pr_dvda, pr_dvdn, pr_dvds, pr_v])
            else:
                return np.stack([pr_dvda, pr_dvdn, pr_dvds])

        # Define grids
        b_aux_grid = np.concatenate([np.array([0.0]), Rfree * aXtraGrid])
        g_aux_grid = np.concatenate(
            [np.array([0.0]), max(RiskyDstn.atoms.flatten()) * nNrmGrid]
        )

        # Create tiled arrays with conforming dimensions.
        b_aux_tiled, g_aux_tiled, Share_tiled = np.meshgrid(
            b_aux_grid, g_aux_grid, ShareGrid, indexing="ij"
        )

        # Find end of period derivatives and value as expectations of (discounted)
        # next period's derivatives and value.
        pr_derivs = calc_expectation(
            IncShkDstn, post_return_derivs, b_aux_tiled, g_aux_tiled, Share_tiled
        )

        # Unpack results and create interpolators
        pr_dvdb_func = MargValueFuncCRRA(
            TrilinearInterp(uPinv(pr_derivs[0]), b_aux_grid, g_aux_grid, ShareGrid),
            CRRA,
        )
        pr_dvdg_func = MargValueFuncCRRA(
            TrilinearInterp(uPinv(pr_derivs[1]), b_aux_grid, g_aux_grid, ShareGrid),
            CRRA,
        )
        pr_dvds_func = TrilinearInterp(pr_derivs[2], b_aux_grid, g_aux_grid, ShareGrid)

        if vFuncBool:
            pr_vFunc = ValueFuncCRRA(
                TrilinearInterp(uInv(pr_derivs[3]), b_aux_grid, g_aux_grid, ShareGrid),
                CRRA,
            )

        # Now construct a function that produces end-of-period derivatives
        # given the risky return draw
        def end_of_period_derivs(risky_ret, a, nTil, s):
            """
            Computes the end-of-period derivatives (and optionally the value) of the
            continuation value function, conditional on risky returns. This is so that the
            expectations can be calculated by integrating over risky returns.

            Parameters
            ----------
            risky_ret : float
                Risky return factor
            a : float
                end-of-period risk-free assets.
            nTil : float
                end-of-period risky assets.
            s : float
                end-of-period income deduction share.
            """

            # Find next-period asset balances
            b_aux = a * Rfree
            g_aux = nTil * risky_ret

            # Interpolate post-return derivatives
            pr_dvdb = pr_dvdb_func(b_aux, g_aux, s)
            pr_dvdg = pr_dvdg_func(b_aux, g_aux, s)
            pr_dvds = pr_dvds_func(b_aux, g_aux, s)

            # Discount

            # Liquid resources
            end_of_prd_dvda = DiscFac * Rfree * LivPrb * pr_dvdb
            # Iliquid resources
            end_of_prd_dvdn = DiscFac * risky_ret * LivPrb * pr_dvdg
            # Contribution share
            end_of_prd_dvds = DiscFac * LivPrb * pr_dvds

            # End of period value function, i11f needed
            if vFuncBool:
                end_of_prd_v = DiscFac * LivPrb * pr_vFunc(b_aux, g_aux, s)
                return np.stack(
                    [end_of_prd_dvda, end_of_prd_dvdn, end_of_prd_dvds, end_of_prd_v]
                )
            else:
                return np.stack([end_of_prd_dvda, end_of_prd_dvdn, end_of_prd_dvds])

    else:
        # If income and returns are not independent, we just integrate over
        # them jointly.

        # Construct a function that evaluates and discounts them given a
        # vector of return and income shocks and an end-of-period state
        def end_of_period_derivs(shocks, a, nTil, s):
            """
            Computes the end-of-period derivatives (and optionally the value) of the
            continuation value function, conditional on shocks. This is so that the
            expectations can be calculated by integrating over shocks.

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
                dvds_tp1 = shocks[1] * (dvdn_tp1 - dvdm_tp1) + dvds_next(
                    m_next, n_next, s
                )

            # Discount next-period-value derivatives to current period

            # Liquid resources
            end_of_prd_dvda = DiscFac * Rfree * LivPrb * temp_fac_A * dvdm_tp1
            # Iliquid resources
            end_of_prd_dvdn = DiscFac * shocks[2] * LivPrb * temp_fac_A * dvdn_tp1
            # Contribution share
            end_of_prd_dvds = DiscFac * LivPrb * temp_fac_B * dvds_tp1

            # End of period value function, i11f needed
            if vFuncBool:
                end_of_prd_v = DiscFac * LivPrb * temp_fac_B * v_next(m_next, n_next, s)
                return np.stack(
                    [end_of_prd_dvda, end_of_prd_dvdn, end_of_prd_dvds, end_of_prd_v]
                )
            else:
                return np.stack([end_of_prd_dvda, end_of_prd_dvdn, end_of_prd_dvds])

    # Now find the expected values on a (a, nTil, s) grid

    # The "inversion" machinery can deal with assets of 0 even if there is a
    # natural borrowing constraint, so include zeros.
    nNrmGrid = np.concatenate([np.array([0.0]), nNrmGrid])
    aNrmGrid = np.concatenate([np.array([0.0]), aXtraGrid])

    # Create tiled arrays with conforming dimensions.
    aNrm_tiled, nNrm_tiled, Share_tiled = np.meshgrid(
        aNrmGrid, nNrmGrid, ShareGrid, indexing="ij"
    )

    # Find end of period derivatives and value as expectations of (discounted)
    # next period's derivatives and value.
    eop_derivs = calc_expectation(
        RiskyDstn if IndepDstnBool and not joint_dist_solver else ShockDstn,
        end_of_period_derivs,
        aNrm_tiled,
        nNrm_tiled,
        Share_tiled,
    )

    # Unpack results
    eop_dvdaNvrs = uPinv(eop_derivs[0])
    eop_dvdnNvrs = uPinv(eop_derivs[1])
    eop_dvds = eop_derivs[2]
    if vFuncBool:
        eop_vNvrs = uInv(eop_derivs[3])

        # Construct an interpolator for eop_V. It will be used later.
        eop_vFunc = ValueFuncCRRA(
            TrilinearInterp(eop_vNvrs, aNrmGrid, nNrmGrid, ShareGrid), CRRA
        )

    # STEP TWO:
    # Solve the consumption problem and create interpolators for c, vCns,
    # and its derivatives.

    # Apply EGM over liquid resources at every (n,s) to find consumption.
    c_end = eop_dvdaNvrs
    mNrm_end = aNrm_tiled + c_end

    # Now construct interpolators for c and the derivatives of vCns.
    # The m grid is different for every (n,s). We interpolate the object of
    # interest on the regular m grid for every (n,s). At the end we will have
    # values of the functions of interest on a regular (m,n,s) grid. We use
    # trilinear interpolation on those points.

    # Expand the exogenous m grid to contain 0.
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
                    m_ns, eop_dvdnNvrs[:, nInd, sInd]
                )(mNrmGrid)

                # dvds
                dvds_vals[:, nInd, sInd] = LinearInterp(m_ns, eop_dvds[:, nInd, sInd])(
                    mNrmGrid
                )

            else:
                # We know that:
                # -The lowest gridpoints of both a and n are 0.
                # -Consumption at m < m0 is m.
                # -dvdn_Fxd at (m,n) for m < m0(n) is dvdn_Fxd(m0,n)
                # -Same is true for dvds_Fxd

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
                            np.array([eop_dvdnNvrs[0, nInd, sInd]]),
                            eop_dvdnNvrs[:, nInd, sInd],
                        ]
                    ),
                )(mNrmGrid)

                # dvds
                dvds_vals[:, nInd, sInd] = LinearInterp(
                    m_ns,
                    np.concatenate(
                        [
                            np.array([eop_dvds[0, nInd, sInd]]),
                            eop_dvds[:, nInd, sInd],
                        ]
                    ),
                )(mNrmGrid)

    # With the arrays filled, create 3D interpolators

    # Consumption interpolator
    cFunc = TrilinearInterp(c_vals, mNrmGrid, nNrmGrid, ShareGrid)
    # dvdmCns interpolator
    dvdmFunc_Cns = MargValueFuncCRRA(cFunc, CRRA)
    # dvdnCns interpolator
    dvdnNvrsFunc = TrilinearInterp(dvdnNvrs_vals, mNrmGrid, nNrmGrid, ShareGrid)
    dvdnFunc_Cns = MargValueFuncCRRA(dvdnNvrsFunc, CRRA)
    # dvdsCns interpolator
    dvdsFunc_Cns = TrilinearInterp(dvds_vals, mNrmGrid, nNrmGrid, ShareGrid)

    # Compute value function if needed
    if vFuncBool:
        # Consumption in the regular grid
        aNrm_reg = mNrm_tiled - c_vals
        vCns = u(c_vals) + eop_vFunc(aNrm_reg, nNrm_tiled, Share_tiled)
        vNvrsCns = uInv(vCns)
        vNvrsFunc_Cns = TrilinearInterp(vNvrsCns, mNrmGrid, nNrmGrid, ShareGrid)
        vFunc_Cns = ValueFuncCRRA(vNvrsFunc_Cns, CRRA)
    else:
        vFunc_Cns = NullFunc()

    # Assemble solution
    solution = RiskyContribCnsSolution(
        vFunc=vFunc_Cns,
        cFunc=cFunc,
        dvdmFunc=dvdmFunc_Cns,
        dvdnFunc=dvdnFunc_Cns,
        dvdsFunc=dvdsFunc_Cns,
    )

    return solution


# Solver for the contribution stage
def solve_RiskyContrib_Sha(
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
    """
    Solves the income-contribution-share stag of the agent's problem

    Parameters
    ----------
    solution_next : RiskyContribCnsSolution
        Solution to the agent's consumption stage problem that follows.
    CRRA : float
        Coefficient of relative risk aversion.
    AdjustPrb : float
        Probability that the agent will be able to rebalance his portfolio
        next period.
    mNrmGrid : numpy array
        Exogenous grid for risk-free resources.
    nNrmGrid : numpy array
        Exogenous grid for risky resources.
    ShareGrid : numpy array
        Exogenous grid for the income contribution share.
    DiscreteShareBool : bool
        Boolean that determines whether only a discrete set of contribution
        shares (ShareGrid) is allowed.
    vFuncBool : bool
        Determines whether the level of the value function is computed.

    Yields
    ------
    solution : RiskyContribShaSolution
        Solution to the income-contribution-share stage of the agent's problem.

    """
    # Unpack solution from the next sub-stage
    vFunc_Cns_next = solution_next.vFunc
    cFunc_next = solution_next.cFunc
    dvdmFunc_Cns_next = solution_next.dvdmFunc
    dvdnFunc_Cns_next = solution_next.dvdnFunc
    dvdsFunc_Cns_next = solution_next.dvdsFunc

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

        opt_idx = np.zeros_like(mNrm_tiled, dtype=int)
        opt_Share = ShareGrid[opt_idx]

        if vFuncBool:
            vNvrsSha = vFunc_Cns_next.vFuncNvrs(mNrm_tiled, nNrm_tiled, opt_Share)

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
            vNvrs = vFunc_Cns_next.vFuncNvrs(mNrm_tiled, nNrm_tiled, Share_tiled)

            # Find the optimal share at each (m,n).
            opt_idx = np.argmax(vNvrs, axis=2)

            # Compute objects needed for the value function and its derivatives
            vNvrsSha = vNvrs[m_idx_tiled, n_idx_tiled, opt_idx]
            opt_Share = ShareGrid[opt_idx]

            # Project grids
            mNrm_tiled = mNrm_tiled[:, :, 0]
            nNrm_tiled = nNrm_tiled[:, :, 0]

        else:
            # Evaluate the marginal value of the contribution share at
            # every (m,n,s) gridpoint
            dvds = dvdsFunc_Cns_next(mNrm_tiled, nNrm_tiled, Share_tiled)

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
            opt_Share = ShareGrid[idx] - dvds[m_idx_tiled, n_idx_tiled, idx] / slopes

            # Replace the ones we knew were constrained
            opt_Share[constrained_bot] = ShareGrid[0]
            opt_Share[constrained_top] = ShareGrid[-1]

            # Project grids
            mNrm_tiled = mNrm_tiled[:, :, 0]
            nNrm_tiled = nNrm_tiled[:, :, 0]

            # Evaluate the inverse value function at the optimal shares
            if vFuncBool:
                vNvrsSha = vFunc_Cns_next.func(mNrm_tiled, nNrm_tiled, opt_Share)

    dvdmNvrsSha = cFunc_next(mNrm_tiled, nNrm_tiled, opt_Share)
    dvdnSha = dvdnFunc_Cns_next(mNrm_tiled, nNrm_tiled, opt_Share)
    dvdnNvrsSha = uPinv(dvdnSha)

    # Interpolators

    # Value function if needed
    if vFuncBool:
        vNvrsFunc_Sha = BilinearInterp(vNvrsSha, mNrmGrid, nNrmGrid)
        vFunc_Sha = ValueFuncCRRA(vNvrsFunc_Sha, CRRA)
    else:
        vFunc_Sha = NullFunc()

    # Contribution share function
    if DiscreteShareBool:
        ShareFunc = DiscreteInterp(
            BilinearInterp(opt_idx, mNrmGrid, nNrmGrid), ShareGrid
        )
    else:
        ShareFunc = BilinearInterp(opt_Share, mNrmGrid, nNrmGrid)

    # Derivatives
    dvdmNvrsFunc_Sha = BilinearInterp(dvdmNvrsSha, mNrmGrid, nNrmGrid)
    dvdmFunc_Sha = MargValueFuncCRRA(dvdmNvrsFunc_Sha, CRRA)
    dvdnNvrsFunc_Sha = BilinearInterp(dvdnNvrsSha, mNrmGrid, nNrmGrid)
    dvdnFunc_Sha = MargValueFuncCRRA(dvdnNvrsFunc_Sha, CRRA)

    solution = RiskyContribShaSolution(
        vFunc_Adj=vFunc_Sha,
        ShareFunc_Adj=ShareFunc,
        dvdmFunc_Adj=dvdmFunc_Sha,
        dvdnFunc_Adj=dvdnFunc_Sha,
        # The fixed agent does nothing at this stage,
        # so his value functions are the next problem's
        vFunc_Fxd=vFunc_Cns_next,
        ShareFunc_Fxd=IdentityFunction(i_dim=2, n_dims=3),
        dvdmFunc_Fxd=dvdmFunc_Cns_next,
        dvdnFunc_Fxd=dvdnFunc_Cns_next,
        dvdsFunc_Fxd=dvdsFunc_Cns_next,
    )

    return solution


# Solver for the asset rebalancing stage
def solve_RiskyContrib_Reb(
    solution_next, CRRA, tau, nNrmGrid, mNrmGrid, dfracGrid, vFuncBool, **unused_params
):
    """
    Solves the asset-rebalancing-stage of the agent's problem

    Parameters
    ----------
    solution_next : RiskyContribShaSolution
        Solution to the income-contribution-share stage problem that follows.
    CRRA : float
        Coefficient of relative risk aversion.
    tau : float
        Tax rate on risky asset withdrawals.
    nNrmGrid : numpy array
        Exogenous grid for risky resources.
    mNrmGrid : numpy array
        Exogenous grid for risk-free resources.
    dfracGrid : numpy array
        Grid for rebalancing flows. The final grid will be equivalent to
        [-nNrm*dfracGrid, dfracGrid*mNrm].
    vFuncBool : bool
        Determines whether the level of th value function must be computed.

    Returns
    -------
    solution : RiskyContribShaSolution
        Solution to the asset-rebalancing stage of the agent's problem.

    """
    # Extract next stage's solution
    vFunc_Adj_next = solution_next.vFunc_Adj
    dvdmFunc_Adj_next = solution_next.dvdmFunc_Adj
    dvdnFunc_Adj_next = solution_next.dvdnFunc_Adj

    vFunc_Fxd_next = solution_next.vFunc_Fxd
    dvdmFunc_Fxd_next = solution_next.dvdmFunc_Fxd
    dvdnFunc_Fxd_next = solution_next.dvdnFunc_Fxd
    dvdsFunc_Fxd_next = solution_next.dvdsFunc_Fxd

    uPinv = lambda x: utilityP_inv(x, CRRA)

    # Create tiled grids

    # Add 0 to the m and n grids
    nNrmGrid = np.concatenate([np.array([0.0]), nNrmGrid])
    nNrm_N = len(nNrmGrid)
    mNrmGrid = np.concatenate([np.array([0.0]), mNrmGrid])
    mNrm_N = len(mNrmGrid)
    d_N = len(dfracGrid)

    # Duplicate d so that possible values are -dfracGrid,dfracGrid. Duplicate 0 is
    # intentional since the tax causes a discontinuity. We need the value
    # from the left and right.
    dfracGrid = np.concatenate((-1 * np.flip(dfracGrid), dfracGrid))

    # It will be useful to pre-evaluate marginals at every (m,n,d) combination

    # Create tiled arrays for every d,m,n option
    d_N2 = len(dfracGrid)
    d_tiled, mNrm_tiled, nNrm_tiled = np.meshgrid(
        dfracGrid, mNrmGrid, nNrmGrid, indexing="ij"
    )

    # Get post-rebalancing assets.
    m_tilde, n_tilde = rebalance_assets(d_tiled, mNrm_tiled, nNrm_tiled, tau)

    # Now the marginals, in inverse space
    dvdmNvrs = dvdmFunc_Adj_next.cFunc(m_tilde, n_tilde)
    dvdnNvrs = dvdnFunc_Adj_next.cFunc(m_tilde, n_tilde)

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
    constrained_top = (
        dvdDNvrs[
            -1,
            :,
            :,
        ]
        <= 0.0
    )

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
    ) / (dfracGrid[idx1] - dfracGrid[idx])
    dfrac_opt = dfracGrid[idx] - dvdDNvrs[idx, m_idx_tiled, n_idx_tiled] / slopes

    # Replace the ones we knew were constrained
    dfrac_opt[constrained_bot] = dfracGrid[0]
    dfrac_opt[constrained_top] = dfracGrid[-1]

    # Find m_tilde and n_tilde
    mtil_opt, ntil_opt = rebalance_assets(dfrac_opt, mNrm_tiled[0], nNrm_tiled[0], tau)

    # Now the derivatives. These are not straight forward because of corner
    # solutions with partial derivatives that change the limits. The idea then
    # is to evaluate the possible uses of the marginal unit of resources and
    # take the maximum.

    # An additional unit of m
    marg_m = dvdmFunc_Adj_next(mtil_opt, ntil_opt)
    # An additional unit of n kept in n
    marg_n = dvdnFunc_Adj_next(mtil_opt, ntil_opt)
    # An additional unit of n withdrawn to m
    marg_n_to_m = marg_m * (1 - tau)

    # Marginal value is the maximum of the marginals in their possible uses
    dvdm_Adj = np.maximum(marg_m, marg_n)
    dvdmNvrs_Adj = uPinv(dvdm_Adj)
    dvdn_Adj = np.maximum(marg_n, marg_n_to_m)
    dvdnNvrs_Adj = uPinv(dvdn_Adj)

    # Interpolators

    # Value
    if vFuncBool:
        vNvrs_Adj = vFunc_Adj_next.vFuncNvrs(mtil_opt, ntil_opt)
        vNvrsFunc_Adj = BilinearInterp(vNvrs_Adj, mNrmGrid, nNrmGrid)
        vFunc_Adj = ValueFuncCRRA(vNvrsFunc_Adj, CRRA)
    else:
        vFunc_Adj = NullFunc()

    # Marginals
    dvdmFunc_Adj = MargValueFuncCRRA(
        BilinearInterp(dvdmNvrs_Adj, mNrmGrid, nNrmGrid), CRRA
    )
    dvdnFunc_Adj = MargValueFuncCRRA(
        BilinearInterp(dvdnNvrs_Adj, mNrmGrid, nNrmGrid), CRRA
    )

    # Decison
    dfracFunc_Adj = BilinearInterp(dfrac_opt, mNrmGrid, nNrmGrid)

    solution = RiskyContribRebSolution(
        # Rebalancing stage adjusting
        vFunc_Adj=vFunc_Adj,
        dfracFunc_Adj=dfracFunc_Adj,
        dvdmFunc_Adj=dvdmFunc_Adj,
        dvdnFunc_Adj=dvdnFunc_Adj,
        # Rebalancing stage fixed (nothing happens, so value functions are
        # the ones from the next stage)
        vFunc_Fxd=vFunc_Fxd_next,
        dfracFunc_Fxd=ConstantFunction(0.0),
        dvdmFunc_Fxd=dvdmFunc_Fxd_next,
        dvdnFunc_Fxd=dvdnFunc_Fxd_next,
        dvdsFunc_Fxd=dvdsFunc_Fxd_next,
    )

    return solution


def solveRiskyContrib(
    solution_next,
    ShockDstn,
    IncShkDstn,
    RiskyDstn,
    IndepDstnBool,
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
    dfracGrid,
    vFuncBool,
    AdjustPrb,
    DiscreteShareBool,
    joint_dist_solver,
):
    """
    Solve a full period (with its three stages) of the agent's problem

    Parameters
    ----------
    solution_next : RiskyContribSolution
        Solution to next period's problem.
    ShockDstn : DiscreteDistribution
        Joint distribution of next period's (0) permanent income shock, (1)
        transitory income shock, and (2) risky asset return factor.
    IncShkDstn : DiscreteDistribution
        Joint distribution of next period's (0) permanent income shock and (1)
        transitory income shock.
    RiskyDstn : DiscreteDistribution
        Distribution of next period's risky asset return factor.
    IndepDstnBool : bool
        Indicates whether the income and risky return distributions are
        independent.
    LivPrb : float
        Probability of surviving until next period.
    DiscFac : float
        Time-preference discount factor.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree : float
        Risk-free return factor.
    PermGroFac : float
        Deterministic permanent income growth factor.
    tau : float
        Tax rate on risky asset withdrawals.
    BoroCnstArt : float
        Minimum allowed market resources (must be 0).
    aXtraGrid : numpy array
        Exogenous grid for end-of-period risk free resources.
    nNrmGrid : numpy array
        Exogenous grid for risky resources.
    mNrmGrid : numpy array
        Exogenous grid for risk-free resources.
    ShareGrid : numpy array
        Exogenous grid for the income contribution share.
    dfracGrid : numpy array
        Grid for rebalancing flows. The final grid will be equivalent to
        [-nNrm*dfracGrid, dfracGrid*mNrm].
    vFuncBool : bool
        Determines whether the level of th value function must be computed.
    AdjustPrb : float
        Probability that the agent will be able to rebalance his portfolio
        next period.
    DiscreteShareBool : bool
        Boolean that determines whether only a discrete set of contribution
        shares (ShareGrid) is allowed.
    joint_dist_solver: bool
        Should the general solver be used even if income and returns are
        independent?

    Returns
    -------
    periodSol : RiskyContribSolution
        Solution to the agent's current-period problem.

    """
    # Pack parameters to be passed to stage-specific solvers
    kws = {
        "ShockDstn": ShockDstn,
        "IncShkDstn": IncShkDstn,
        "RiskyDstn": RiskyDstn,
        "IndepDstnBool": IndepDstnBool,
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
        "dfracGrid": dfracGrid,
        "vFuncBool": vFuncBool,
        "AdjustPrb": AdjustPrb,
        "DiscreteShareBool": DiscreteShareBool,
        "joint_dist_solver": joint_dist_solver,
    }

    # Stages of the problem in chronological order
    Stages = ["Reb", "Sha", "Cns"]
    n_stages = len(Stages)
    # Solvers, indexed by stage names
    Solvers = {
        "Reb": solve_RiskyContrib_Reb,
        "Sha": solve_RiskyContrib_Sha,
        "Cns": solve_RiskyContrib_Cns,
    }

    # Initialize empty solution
    stage_sols = {}
    # Solve stages backwards
    for i in reversed(range(n_stages)):
        stage = Stages[i]

        # In the last stage, the next solution is the first stage of the next
        # period. Otherwise, its the next stage of his period.
        if i == n_stages - 1:
            sol_next_stage = solution_next.stage_sols[Stages[0]]
        else:
            sol_next_stage = stage_sols[Stages[i + 1]]

        # Solve
        stage_sols[stage] = Solvers[stage](sol_next_stage, **kws)

    # Assemble stage solutions into period solution
    periodSol = RiskyContribSolution(**stage_sols)

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
    "dCount": 20,
}

# Infinite horizon version
init_risky_contrib = init_risky_asset.copy()
init_risky_contrib.update(risky_contrib_params)

# Lifecycle version
init_risky_contrib_lifecycle = init_lifecycle.copy()
init_risky_contrib_lifecycle.update(risky_asset_parms)
init_risky_contrib_lifecycle.update(risky_contrib_params)

"""
This file contains a class that adds a risky asset with a log-normal return
factor to IndShockConsumerType.
This class is not a fully specified model and therefore has no solution or
simulation methods. It is meant as a container of methods for dealing with
risky assets that will be useful to models what will inherit from it.
"""
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from HARK import make_one_period_oo_solver
from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsIndShockSolver,
    ConsumerSolution,
    IndShockConsumerType,  # PortfolioConsumerType inherits from it
    init_idiosyncratic_shocks,  # Baseline dictionary to build on
)
from HARK.distribution import (
    DiscreteDistribution,
    calc_expectation,
    combine_indep_dstns,
    IndexDistribution,
    Lognormal,
    Bernoulli,
)
from HARK.interpolation import LinearInterp, MargValueFuncCRRA, ValueFuncCRRA


class RiskyAssetConsumerType(IndShockConsumerType):
    """
    A consumer type that has access to a risky asset for his savings. The
    risky asset has lognormal returns that are possibly correlated with his
    income shocks.

    There is a friction that prevents the agent from adjusting his portfolio
    at any given period with an exogenously given probability.
    The meaning of "adjusting his portfolio" depends on the particular model.
    """

    time_inv_ = deepcopy(IndShockConsumerType.time_inv_)

    shock_vars_ = IndShockConsumerType.shock_vars_ + ["Adjust", "Risky"]

    def __init__(self, verbose=False, quiet=False, **kwds):
        params = init_risky_asset.copy()
        params.update(kwds)
        kwds = params

        # Initialize a basic consumer type
        IndShockConsumerType.__init__(self, verbose=verbose, quiet=quiet, **kwds)

        # These method must be overwritten by classes that inherit from
        # RiskyAssetConsumerType
        self.solve_one_period = make_one_period_oo_solver(ConsRiskySolver)

    def pre_solve(self):
        self.update_solution_terminal()

    def update(self):

        IndShockConsumerType.update(self)
        self.update_AdjustPrb()
        self.update_RiskyDstn()
        self.update_ShockDstn()

    def update_RiskyDstn(self):
        """
        Creates the attributes RiskyDstn from the primitive attributes RiskyAvg,
        RiskyStd, and RiskyCount, approximating the (perceived) distribution of
        returns in each period of the cycle.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Determine whether this instance has time-varying risk perceptions
        if (
            (type(self.RiskyAvg) is list)
            and (type(self.RiskyStd) is list)
            and (len(self.RiskyAvg) == len(self.RiskyStd))
            and (len(self.RiskyAvg) == self.T_cycle)
        ):
            self.add_to_time_vary("RiskyAvg", "RiskyStd")
        elif (type(self.RiskyStd) is list) or (type(self.RiskyAvg) is list):
            raise AttributeError(
                "If RiskyAvg is time-varying, then RiskyStd must be as well, and they must both have length of T_cycle!"
            )
        else:
            self.add_to_time_inv("RiskyAvg", "RiskyStd")

        # Generate a discrete approximation to the risky return distribution if the
        # agent has age-varying beliefs about the risky asset
        if "RiskyAvg" in self.time_vary:
            self.RiskyDstn = IndexDistribution(
                Lognormal.from_mean_std,
                {"mean": self.RiskyAvg, "std": self.RiskyStd},
                seed=self.RNG.randint(0, 2 ** 31 - 1),
            ).approx(self.RiskyCount)

            self.add_to_time_vary("RiskyDstn")

        # Generate a discrete approximation to the risky return distribution if the
        # agent does *not* have age-varying beliefs about the risky asset (base case)
        else:
            self.RiskyDstn = Lognormal.from_mean_std(
                self.RiskyAvg, self.RiskyStd,
            ).approx(self.RiskyCount)
            self.add_to_time_inv("RiskyDstn")

    def update_ShockDstn(self):
        """
        Combine the income shock distribution (over PermShk and TranShk) with the
        risky return distribution (RiskyDstn) to make a new attribute called ShockDstn.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if "RiskyDstn" in self.time_vary:
            self.ShockDstn = [
                combine_indep_dstns(self.IncShkDstn[t], self.RiskyDstn[t])
                for t in range(self.T_cycle)
            ]
        else:
            self.ShockDstn = [
                combine_indep_dstns(self.IncShkDstn[t], self.RiskyDstn)
                for t in range(self.T_cycle)
            ]
        self.add_to_time_vary("ShockDstn")

        # Mark whether the risky returns and income shocks are independent (they are)
        self.IndepDstnBool = True
        self.add_to_time_inv("IndepDstnBool")

    def update_AdjustPrb(self):
        """
        Checks and updates the exogenous probability of the agent being allowed
        to rebalance his portfolio/contribution scheme. It can be time varying.

        Parameters
        ------
        None.

        Returns
        -------
        None.

        """
        if type(self.AdjustPrb) is list and (len(self.AdjustPrb) == self.T_cycle):
            self.add_to_time_vary("AdjustPrb")
        elif type(self.AdjustPrb) is list:
            raise AttributeError(
                "If AdjustPrb is time-varying, it must have length of T_cycle!"
            )
        else:
            self.add_to_time_inv("AdjustPrb")

    def get_Risky(self):
        """
        Sets the attribute Risky as a single draw from a lognormal distribution.
        Uses the attributes RiskyAvgTrue and RiskyStdTrue if RiskyAvg is time-varying,
        else just uses the single values from RiskyAvg and RiskyStd.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if "RiskyDstn" in self.time_vary:
            RiskyAvg = self.RiskyAvgTrue
            RiskyStd = self.RiskyStdTrue
        else:
            RiskyAvg = self.RiskyAvg
            RiskyStd = self.RiskyStd

        self.shocks["Risky"] = Lognormal.from_mean_std(
            RiskyAvg, RiskyStd, seed=self.RNG.randint(0, 2 ** 31 - 1)
        ).draw(1)

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
        self.shocks["Adjust"] = IndexDistribution(
            Bernoulli, {"p": self.AdjustPrb}, seed=self.RNG.randint(0, 2 ** 31 - 1)
        ).draw(self.t_cycle)

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


####################################################################################################
####################################################################################################


@dataclass
class ConsRiskySolver(ConsIndShockSolver):
    solution_next: ConsumerSolution
    IncShkDstn: DiscreteDistribution
    TranShkDstn: DiscreteDistribution
    PermShkDstn: DiscreteDistribution
    RiskyDstn: DiscreteDistribution
    LivPrb: float
    DiscFac: float
    CRRA: float
    Rfree: float
    PermGroFac: float
    BoroCnstArt: float
    aXtraGrid: np.array
    vFuncBool: bool
    CubicBool: bool

    def __post_init__(self):
        self.def_utility_funcs()

        # Make sure the individual is liquidity constrained.  Allowing a consumer to
        # borrow *and* invest in an asset with unbounded (negative) returns is a bad mix.
        if self.BoroCnstArt != 0.0:
            raise ValueError("RiskyAssetConsumerType must have BoroCnstArt=0.0!")

        if self.CubicBool:
            raise NotImplementedError(
                "RiskyAssetConsumerType does not implement cubic interpolation yet!"
            )

    def def_BoroCnst(self, BoroCnstArt):

        # Natural Borrowing Constraint is always 0.0 in risky return models
        self.BoroCnstNat = 0.0
        # minimum normalized cash on hand is 0.0
        self.mNrmMinNow = 0.0

        self.MPCmaxEff = self.MPCmaxNow

        # Define the borrowing constraint (limiting consumption function)
        self.cFuncNowCnst = LinearInterp(
            np.array([self.mNrmMinNow, self.mNrmMinNow + 1]), np.array([0.0, 1.0])
        )

    def calc_EndOfPrdvP(self):
        """
        Calculate end-of-period marginal value of assets at each point in aNrmNow.
        Does so by taking a weighted sum of next period marginal values across
        income shocks (in a preconstructed grid self.mNrmNext).

        Parameters
        ----------
        none

        Returns
        -------
        EndOfPrdvP : np.array
            A 1D array of end-of-period marginal value of assets
        """

        self.BoroCnstNat = 0.0
        self.aNrmNow = np.asarray(self.aXtraGrid) + self.BoroCnstNat

        self.agrid = self.aXtraGrid
        self.bgrid = np.append(
            self.agrid[0] * self.RiskyDstn.X.min(), self.agrid * self.RiskyDstn.X.max()
        )
        self.wgrid = np.append(
            self.bgrid[0] / (self.PermGroFac * self.PermShkDstn.X.max()),
            self.bgrid / (self.PermGroFac * self.PermShkDstn.X.min()),
        )

        # calculate expectation with respect to transitory shock

        def preTranShkvPfunc(tran_shk, w_nrm):
            return self.vPfuncNext(w_nrm + tran_shk)

        preTranShkvP = calc_expectation(self.TranShkDstn, preTranShkvPfunc, self.wgrid)
        preTranShkvPNvrs = self.uPinv(preTranShkvP)
        # Need to add value of 0 at borrowing constraint
        preTranShkvPNvrsFunc = LinearInterp(
            np.append(0, self.wgrid), np.append(0, preTranShkvPNvrs)
        )
        self.preTranShkvPfunc = MargValueFuncCRRA(preTranShkvPNvrsFunc, self.CRRA)

        # calculate expectation with respect to permanent shock

        def prePermShkvPfunc(perm_shk, b_nrm):
            shock = perm_shk * self.PermGroFac
            return shock ** (-self.CRRA) * self.preTranShkvPfunc(b_nrm / shock)

        prePermShkvP = calc_expectation(self.PermShkDstn, prePermShkvPfunc, self.bgrid)
        prePermShkvPNvrs = self.uPinv(prePermShkvP)
        # Need to add value of 0 at borrowing constraint
        prePermShkvPNvrsFunc = LinearInterp(
            np.append(0, self.bgrid), np.append(0, prePermShkvPNvrs)
        )
        self.prePermShkvPfunc = MargValueFuncCRRA(prePermShkvPNvrsFunc, self.CRRA)

        # calculate expectation with respect to risky shock

        def preRiskyShkvPfunc(risky_shk, a_nrm):
            return risky_shk * self.prePermShkvPfunc(a_nrm * risky_shk)

        preRiskyShkvP = self.DiscFacEff * calc_expectation(
            self.RiskyDstn, preRiskyShkvPfunc, self.agrid
        )
        preRiskyShkvPNvrs = self.uPinv(preRiskyShkvP)
        # Need to add value of 0 at borrowing constraint
        preRiskyShkvPNvrsFunc = LinearInterp(
            np.append(0, self.agrid), np.append(0, preRiskyShkvPNvrs)
        )
        self.preRiskyShkvPfunc = MargValueFuncCRRA(preRiskyShkvPNvrsFunc, self.CRRA)

        EndOfPrdvP = preRiskyShkvP

        return EndOfPrdvP

    def make_EndOfPrdvFunc(self, EndOfPrdvP):
        """
        Construct the end-of-period value function for this period, storing it
        as an attribute of self for use by other methods.

        Parameters
        ----------
        EndOfPrdvP : np.array
            Array of end-of-period marginal value of assets corresponding to the
            asset values in self.aNrmNow.

        Returns
        -------
        none
        """

        def preTranShkvFunc(tran_shk, w_nrm):
            return self.vFuncNext(w_nrm + tran_shk)

        preTranShkv = calc_expectation(self.TranShkDstn, preTranShkvFunc, self.wgrid)
        # value transformed through inverse utility
        preTranShkvNvrs = self.uinv(preTranShkv)
        preTranShkvNvrsFunc = LinearInterp(
            np.append(0, self.wgrid), np.append(0, preTranShkvNvrs)
        )
        self.preTranShkvFunc = ValueFuncCRRA(preTranShkvNvrsFunc, self.CRRA)

        def prePermShkvFunc(perm_shk, b_nrm):
            shock = perm_shk * self.PermGroFac
            return shock ** (1.0 - self.CRRA) * self.preTranShkvFunc(b_nrm / shock)

        prePermShkv = calc_expectation(self.PermShkDstn, prePermShkvFunc, self.bgrid)
        # value transformed through inverse utility
        prePermShkvNvrs = self.uinv(prePermShkv)
        prePermShkvNvrsFunc = LinearInterp(
            np.append(0, self.bgrid), np.append(0, prePermShkvNvrs)
        )
        self.prePermShkvFunc = ValueFuncCRRA(prePermShkvNvrsFunc, self.CRRA)

        def preRiskyShkvFunc(risky_shk, a_nrm):
            return self.prePermShkvFunc(risky_shk * a_nrm)

        preRiskyShkv = self.DiscFacEff * calc_expectation(
            self.RiskyDstn, preRiskyShkvFunc, self.agrid
        )
        # value transformed through inverse utility
        preRiskyShkvNvrs = self.uinv(preRiskyShkv)
        preRiksyShkNvrsFunc = LinearInterp(
            np.append(0, self.agrid), np.append(0, preRiskyShkvNvrs)
        )
        self.preRiskyShkvFunc = ValueFuncCRRA(preRiksyShkNvrsFunc, self.CRRA)

        self.EndOfPrdvFunc = self.preRiskyShkvFunc


# %% Initial parameter sets

# %% Base risky asset dictionary

risky_asset_parms = {
    # Risky return factor moments. Based on SP500 real returns from Shiller's
    # "chapter 26" data, which can be found at http://www.econ.yale.edu/~shiller/data.htm
    "RiskyAvg": 1.080370891,
    "RiskyStd": 0.177196585,
    # Number of integration nodes to use in approximation of risky returns
    "RiskyCount": 5,
    # Probability that the agent can adjust their portfolio each period
    "AdjustPrb": 1.0,
}

# Make a dictionary to specify a risky asset consumer type
init_risky_asset = init_idiosyncratic_shocks.copy()
init_risky_asset.update(risky_asset_parms)

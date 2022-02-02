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
                self.RiskyAvg, self.RiskyStd
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
    ShockDstn: DiscreteDistribution
    LivPrb: float
    DiscFac: float
    CRRA: float
    Rfree: float
    PermGroFac: float
    BoroCnstArt: float
    aXtraGrid: np.array
    vFuncBool: bool
    CubicBool: bool
    IndepDstnBool: bool

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

    def set_and_update_values(self, solution_next, IncShkDstn, LivPrb, DiscFac):
        super().set_and_update_values(solution_next, IncShkDstn, LivPrb, DiscFac)

        # overwrite PatFac

        def pat_fac_func(shock):
            return shock ** (1.0 - self.CRRA)

        self.PatFac = (
            calc_expectation(self.RiskyDstn, pat_fac_func) * self.DiscFacEff
        ) ** (1.0 / self.CRRA)

        self.MPCminNow = 1.0 / (1.0 + self.PatFac / solution_next.MPCmin)

        self.hNrmNow = (
            self.PermGroFac / self.Rfree * (self.Ex_IncNext + solution_next.hNrm)
        )
        self.MPCmaxNow = 1.0 / (
            1.0
            + (self.WorstIncPrb ** (1.0 / self.CRRA))
            * self.PatFac
            / solution_next.MPCmax
        )

        self.cFuncLimitIntercept = self.MPCminNow * self.hNrmNow
        self.cFuncLimitSlope = self.MPCminNow

    def def_BoroCnst(self, BoroCnstArt):

        # Natural Borrowing Constraint is always 0.0 in risky return models
        self.BoroCnstNat = -self.TranShkDstn.X.min()
        # Flag for whether the natural borrowing constraint is zero
        self.zero_bound = self.BoroCnstNat == BoroCnstArt

        # minimum normalized cash on hand is 0.0
        self.mNrmMinNow = BoroCnstArt

        if self.BoroCnstNat < self.mNrmMinNow:
            self.MPCmaxEff = 1.0  # If actually constrained, MPC near limit is 1
        else:
            self.MPCmaxEff = self.MPCmaxNow

        # Define the borrowing constraint (limiting consumption function)
        self.cFuncNowCnst = LinearInterp(
            np.array([self.mNrmMinNow, self.mNrmMinNow + 1]), np.array([0.0, 1.0])
        )

    def prepare_to_calc_EndOfPrdvP(self):

        if self.zero_bound:
            aNrmNow = self.aXtraGrid
            bNrmNext = np.append(
                aNrmNow[0] * self.RiskyDstn.X.min(), aNrmNow * self.RiskyDstn.X.max(),
            )
            wNrmNext = np.append(
                bNrmNext[0] / (self.PermGroFac * self.PermShkDstn.X.max()),
                bNrmNext / (self.PermGroFac * self.PermShkDstn.X.min()),
            )
        else:
            aNrmNow = np.append(self.BoroCnstArt, self.aXtraGrid)
            bNrmNext = aNrmNow * self.RiskyDstn.X.max()
            wNrmNext = bNrmNext / (self.PermGroFac * self.PermShkDstn.X.min())

        self.aNrmNow = aNrmNow
        self.bNrmNext = bNrmNext
        self.wNrmNext = wNrmNext

        return self.aNrmNow

    def calc_ExpMargValueFunc(self, dstn, func, grid):

        vals = calc_expectation(dstn, func, grid)
        nvrs = self.uPinv(vals)
        nvrsFunc = LinearInterp(grid, nvrs)
        margValueFunc = MargValueFuncCRRA(nvrsFunc, self.CRRA)

        return margValueFunc, vals

    def calc_ExpValueFunc(self, dstn, func, grid):

        vals = calc_expectation(dstn, func, grid)
        nvrs = self.uinv(vals)
        nvrsFunc = LinearInterp(grid, nvrs)
        valueFunc = ValueFuncCRRA(nvrsFunc, self.CRRA)

        return valueFunc, vals

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

        if self.IndepDstnBool:

            # calculate expectation with respect to transitory shock

            def preTranShkvPfunc(tran_shk, w_nrm):
                return self.vPfuncNext(w_nrm + tran_shk)

            self.preTranShkvPfunc, _ = self.calc_ExpMargValueFunc(
                self.TranShkDstn, preTranShkvPfunc, self.wNrmNext
            )

            # calculate expectation with respect to permanent shock

            def prePermShkvPfunc(perm_shk, b_nrm):
                shock = perm_shk * self.PermGroFac
                return shock ** (-self.CRRA) * self.preTranShkvPfunc(b_nrm / shock)

            self.prePermShkvPfunc, _ = self.calc_ExpMargValueFunc(
                self.PermShkDstn, prePermShkvPfunc, self.bNrmNext
            )

            # calculate expectation with respect to risky shock

            def preRiskyShkvPfunc(risky_shk, a_nrm):
                return (
                    self.DiscFacEff
                    * risky_shk
                    * self.prePermShkvPfunc(a_nrm * risky_shk)
                )

            self.preRiskyShkvPfunc, EndOfPrdvP = self.calc_ExpMargValueFunc(
                self.RiskyDstn, preRiskyShkvPfunc, self.aNrmNow
            )

            self.EndOfPrdvPFunc = self.preRiskyShkvPfunc

        else:

            def vP_next(shocks, a_nrm):
                perm_shk = shocks[0] * self.PermGroFac
                mNrm_next = a_nrm * shocks[2] / perm_shk + shocks[1]
                return (
                    self.DiscFacEff
                    * shocks[2]
                    * perm_shk ** (-self.CRRA)
                    * self.vPfuncNext(mNrm_next)
                )

            self.EndOfPrdvPFunc, EndOfPrdvP = self.calc_ExpMargValueFunc(
                self.ShockDstn, vP_next, self.aNrmNow
            )

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

        if self.IndepDstnBool:

            def preTranShkvFunc(tran_shk, w_nrm):
                return self.vFuncNext(w_nrm + tran_shk)

            self.preTranShkvFunc, _ = self.calc_ExpValueFunc(
                self.TranShkDstn, preTranShkvFunc, self.wNrmNext
            )

            def prePermShkvFunc(perm_shk, b_nrm):
                shock = perm_shk * self.PermGroFac
                return shock ** (1.0 - self.CRRA) * self.preTranShkvFunc(b_nrm / shock)

            self.prePermShkvFunc, _ = self.calc_ExpValueFunc(
                self.PermShkDstn, prePermShkvFunc, self.bNrmNext
            )

            def preRiskyShkvFunc(risky_shk, a_nrm):
                return self.DiscFacEff * self.prePermShkvFunc(risky_shk * a_nrm)

            self.preRiskyShkvFunc, EndOfPrdv = self.calc_ExpValueFunc(
                self.RiskyDstn, preRiskyShkvFunc, self.aNrmNow
            )

            self.EndOfPrdvFunc = self.preRiskyShkvFunc

        else:

            def v_next(shocks, a_nrm):
                perm_shk = shocks[0] * self.PermGroFac
                mNrm_next = a_nrm * shocks[2] / perm_shk + shocks[1]
                return (
                    self.DiscFacEff
                    * perm_shk ** (1 - self.CRRA)
                    * self.vFuncNext(mNrm_next)
                )

            self.EndOfPrdvFunc, EndOfPrdv = self.calc_ExpValueFunc(
                self.ShockDstn, v_next, self.aNrmNow
            )


            )


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

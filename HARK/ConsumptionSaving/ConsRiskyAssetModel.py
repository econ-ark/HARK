"""
This file contains a class that adds a risky asset with a log-normal return
factor to IndShockConsumerType.
This class is not a fully specified model and therefore has no solution or
simulation methods. It is meant as a container of methods for dealing with
risky assets that will be useful to models what will inherit from it.
"""
import numpy as np
from copy import deepcopy
from HARK.utilities import NullFunc
from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,  # PortfolioConsumerType inherits from it
    init_idiosyncratic_shocks,  # Baseline dictionary to build on
)

from HARK.distribution import (
    combine_indep_dstns,
    IndexDistribution,
    Lognormal,
    Bernoulli,
)

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

    def __init__(self, cycles=1, verbose=False, quiet=False, **kwds):
        params = init_risky_asset.copy()
        params.update(kwds)
        kwds = params

        # Initialize a basic consumer type
        IndShockConsumerType.__init__(
            self, cycles=cycles, verbose=verbose, quiet=quiet, **kwds
        )

        # These method must be overwritten by classes that inherit from
        # RiskyAssetConsumerType
        self.solve_one_period = NullFunc()

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
                {
                    'mean' : self.RiskyAvg,
                    'std' : self.RiskyStd
                },
                seed=self.RNG.randint(0, 2 ** 31 - 1)
            ).approx(
                self.RiskyCount
            )

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
        RiskyAvgSqrd = RiskyAvg ** 2
        RiskyVar = RiskyStd ** 2

        mu = np.log(RiskyAvg / (np.sqrt(1.0 + RiskyVar / RiskyAvgSqrd)))
        sigma = np.sqrt(np.log(1.0 + RiskyVar / RiskyAvgSqrd))
        self.shocks["Risky"] = Lognormal(
            mu, sigma, seed=self.RNG.randint(0, 2 ** 31 - 1)
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
        self.shocks['Adjust'] = IndexDistribution(
            Bernoulli,
            {'p' : self.AdjustPrb},
            seed=self.RNG.randint(0, 2 ** 31 - 1)
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
    "AdjustPrb": 1.0
}

# Make a dictionary to specify a risky asset consumer type
init_risky_asset = init_idiosyncratic_shocks.copy()
init_risky_asset.update(risky_asset_parms)
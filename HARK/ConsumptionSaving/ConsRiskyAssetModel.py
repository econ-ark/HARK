"""
This file contains a class that adds a risky asset with a log-normal return
factor to IndShockConsumerType.
This class is not a fully specified model and therefore has no solution or
simulation methods. It is meant as a container of methods for dealing with
risky assets that will be useful to models what will inherit from it.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar, root_scalar

from HARK import make_one_period_oo_solver
from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsIndShockSolver,
    ConsumerSolution,
    IndShockConsumerType,  # PortfolioConsumerType inherits from it
    init_idiosyncratic_shocks,  # Baseline dictionary to build on
)
from HARK.distribution import (
    DiscreteDistribution,
    add_discrete_outcome_constant_mean,
    calc_expectation,
    combine_indep_dstns,
    IndexDistribution,
    Lognormal,
    Bernoulli,
)
from HARK.interpolation import (
    LinearInterp,
    MargValueFuncCRRA,
    ValueFuncCRRA,
    ConstantFunction,
)

# RiskyAssetIndShockConsumerType
class RiskyAssetIndShockConsumerType(IndShockConsumerType):
    """
    A consumer type that has access to a risky asset for his savings. The
    risky asset has lognormal returns that are possibly correlated with his
    income shocks.

    There is a friction that prevents the agent from adjusting his portfolio
    at any given period with an exogenously given probability.
    The meaning of "adjusting his portfolio" depends on the particular model.
    """

    shock_vars_ = IndShockConsumerType.shock_vars_ + ["Adjust", "Risky"]

    def __init__(self, verbose=False, quiet=False, **kwds):
        params = init_risky_asset.copy()
        params.update(kwds)
        kwds = params

        if not hasattr(self, "PortfolioBool"):
            self.PortfolioBool = False

        # Initialize a basic consumer type
        IndShockConsumerType.__init__(self, verbose=verbose, quiet=quiet, **kwds)

        # These method must be overwritten by classes that inherit from
        # RiskyAssetConsumerType
        if self.PortfolioBool:
            solver = ConsPortfolioRiskyAssetIndShkSolver  # optimize over shares
        else:
            solver = ConsRiskyAssetIndShkSolver  # risky share of 1

        self.solve_one_period = make_one_period_oo_solver(solver)

    def pre_solve(self):
        self.update_solution_terminal()

        if self.PortfolioBool:
            self.solution_terminal.ShareFunc = ConstantFunction(1.0)

    def update(self):

        IndShockConsumerType.update(self)
        self.update_AdjustPrb()
        self.update_RiskyDstn()
        self.update_ShockDstn()

        if self.PortfolioBool:
            self.update_ShareLimit()
            self.update_ShareGrid()

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
                seed=self.RNG.randint(0, 2**31 - 1),
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

    def update_ShareLimit(self):
        """
        Creates the attribute ShareLimit, representing the limiting lower bound of
        risky portfolio share as mNrm goes to infinity.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if "RiskyDstn" in self.time_vary:
            self.ShareLimit = []
            for t in range(self.T_cycle):
                RiskyDstn = self.RiskyDstn[t]
                temp_f = lambda s: -((1.0 - self.CRRA) ** -1) * np.dot(
                    (self.Rfree + s * (RiskyDstn.X - self.Rfree)) ** (1.0 - self.CRRA),
                    RiskyDstn.pmf,
                )
                SharePF = minimize_scalar(temp_f, bounds=(0.0, 1.0), method="bounded").x
                self.ShareLimit.append(SharePF)
            self.add_to_time_vary("ShareLimit")

        else:
            RiskyDstn = self.RiskyDstn
            temp_f = lambda s: -((1.0 - self.CRRA) ** -1) * np.dot(
                (self.Rfree + s * (RiskyDstn.X - self.Rfree)) ** (1.0 - self.CRRA),
                RiskyDstn.pmf,
            )
            SharePF = minimize_scalar(temp_f, bounds=(0.0, 1.0), method="bounded").x
            self.ShareLimit = SharePF
            self.add_to_time_inv("ShareLimit")

    def update_ShareGrid(self):
        """
        Creates the attribute ShareGrid as an evenly spaced grid on [0.,1.], using
        the primitive parameter ShareCount.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.ShareGrid = np.linspace(0.0, 1.0, self.ShareCount)
        self.add_to_time_inv("ShareGrid")

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
            RiskyAvg, RiskyStd, seed=self.RNG.randint(0, 2**31 - 1)
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
            Bernoulli, {"p": self.AdjustPrb}, seed=self.RNG.randint(0, 2**31 - 1)
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


# This is to preserve compatibility with old name
RiskyAssetConsumerType = RiskyAssetIndShockConsumerType


class RiskyReturnGivenFixedPortfolioShareConsumerType(RiskyAssetIndShockConsumerType):
    time_vary_ = RiskyAssetIndShockConsumerType.time_vary_ + ["RiskyShareFixed"]

    def __init__(self, verbose=False, quiet=False, **kwds):
        params = init_risky_share_fixed.copy()
        params.update(kwds)
        kwds = params

        # Initialize a basic consumer type
        RiskyAssetIndShockConsumerType.__init__(
            self, verbose=verbose, quiet=quiet, **kwds
        )

        self.solve_one_period = make_one_period_oo_solver(
            ConsFixedPortfolioRiskyAssetIndShkSolver
        )


####################################################################################################
####################################################################################################


@dataclass
class ConsRiskyAssetIndShkSolver(ConsIndShockSolver):
    """
    Solver for an agent that can save in an asset that has a risky return.
    """

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
        """
        Unpacks some of the inputs (and calculates simple objects based on them),
        storing the results in self for use by other methods.  These include:
        income shocks and probabilities, next period's marginal value function
        (etc), the probability of getting the worst income shock next period,
        the patience factor, human wealth, and the bounding MPCs.

        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one period problem.
        IncShkDstn : distribution.DiscreteDistribution
            A DiscreteDistribution with a pmf
            and two point value arrays in X, order:
            permanent shocks, transitory shocks.
        LivPrb : float
            Survival probability; likelihood of being alive at the beginning of
            the succeeding period.
        DiscFac : float
            Intertemporal discount factor for future utility.

        Returns
        -------
        None
        """

        super().set_and_update_values(solution_next, IncShkDstn, LivPrb, DiscFac)

        # Absolute Patience Factor for the model with risk free return is defined at
        # https://econ-ark.github.io/BufferStockTheory/BufferStockTheory3.html#APFacDefn

        # overwrite APFac
        # The corresponding Absolute Patience Factor when the return factor is risky is defined
        # implicitly in #TODO add link

        def APFac_func(shock):  # link to notes
            return ((shock * self.DiscFacEff) ** (1.0 / self.CRRA)) / shock

        self.APFac = calc_expectation(self.RiskyDstn, APFac_func)

        self.MPCminNow = 1.0 / (1.0 + self.APFac / solution_next.MPCmin)

        # overwrite human wealth function

        def h_nrm_func(shocks):  # link to notes
            return (
                self.PermGroFac
                / shocks[2]
                * (shocks[0] * shocks[1] + solution_next.hNrm)
            )

        self.hNrmNow = calc_expectation(self.ShockDstn, h_nrm_func)

        self.MPCmaxNow = 1.0 / (
            1.0
            + (self.WorstIncPrb ** (1.0 / self.CRRA))
            * self.APFac
            / solution_next.MPCmax
        )

        # let upper extrapolation be linear for now

        self.cFuncLimitIntercept = None
        self.cFuncLimitSlope = None

    def def_BoroCnst(self, BoroCnstArt):
        """
        Defines the constrained portion of the consumption function as cFuncNowCnst,
        an attribute of self.  Uses the artificial and natural borrowing constraints.

        Parameters
        ----------
        BoroCnstArt : float or None
            Borrowing constraint for the minimum allowable assets to end the
            period with.  If it is less than the natural borrowing constraint,
            then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
            rowing constraint.

        Returns
        -------
        none
        """

        # Calculate the minimum allowable value of money resources in this period
        self.BoroCnstNat = (
            (self.solution_next.mNrmMin - self.TranShkDstn.X.min())
            * (self.PermGroFac * self.PermShkDstn.X.min())
            / self.RiskyDstn.X.max()
        )

        # Flag for whether the natural borrowing constraint is zero
        self.zero_bound = self.BoroCnstNat == BoroCnstArt

        if BoroCnstArt is None:
            self.mNrmMinNow = self.BoroCnstNat
        else:
            self.mNrmMinNow = np.max([self.BoroCnstNat, BoroCnstArt])
        if self.BoroCnstNat < self.mNrmMinNow:
            self.MPCmaxEff = 1.0  # If actually constrained, MPC near limit is 1
        else:
            self.MPCmaxEff = self.MPCmaxNow

        # Define the borrowing constraint (limiting consumption function)
        self.cFuncNowCnst = LinearInterp(
            np.array([self.mNrmMinNow, self.mNrmMinNow + 1]), np.array([0.0, 1.0])
        )

    def prepare_to_calc_EndOfPrdvP(self):
        """
        Prepare to calculate end-of-period marginal value by creating an array
        of market resources that the agent could have next period, considering
        the grid of end-of-period assets and the distribution of shocks he might
        experience next period.

        Parameters
        ----------
        none

        Returns
        -------
        aNrmNow : np.array
            A 1D array of end-of-period assets; also stored as attribute of self.
        """

        if self.zero_bound:
            # if zero is BoroCnstNat, do not evaluate at 0.0
            aNrmNow = self.aXtraGrid
            bNrmNext = np.append(
                aNrmNow[0] * self.RiskyDstn.X.min(),
                aNrmNow * self.RiskyDstn.X.max(),
            )
            wNrmNext = np.append(
                bNrmNext[0] / (self.PermGroFac * self.PermShkDstn.X.max()),
                bNrmNext / (self.PermGroFac * self.PermShkDstn.X.min()),
            )
        else:
            # add zero to aNrmNow
            aNrmNow = np.append(self.BoroCnstArt, self.aXtraGrid)
            bNrmNext = aNrmNow * self.RiskyDstn.X.max()
            wNrmNext = bNrmNext / (self.PermGroFac * self.PermShkDstn.X.min())

        self.aNrmNow = aNrmNow
        self.bNrmNext = bNrmNext
        self.wNrmNext = wNrmNext

        return self.aNrmNow

    def calc_ExpMargValueFunc(self, dstn, func, grid):
        """
        Calculate Expected Marginal Value Function given a distribution,
        a function, and a set of interpolation nodes.
        """

        vals = calc_expectation(dstn, func, grid)
        nvrs = self.uPinv(vals)
        nvrsFunc = LinearInterp(grid, nvrs)
        margValueFunc = MargValueFuncCRRA(nvrsFunc, self.CRRA)

        return margValueFunc, vals

    def calc_preIncShkvPfunc(self, vPfuncNext):
        """
        Calculate Expected Marginal Value Function before the
        realization of income shocks.
        """

        # calculate expectation with respect to transitory shock

        def preTranShkvPfunc(tran_shk, w_nrm):
            return vPfuncNext(w_nrm + tran_shk)

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

        preIncShkvPfunc = self.prePermShkvPfunc

        return preIncShkvPfunc

    def calc_preRiskyShkvPfunc(self, preIncShkvPfunc):
        """
        Calculate Expected Marginal Value Function before
        the realization of the risky return.
        """

        # calculate expectation with respect to risky shock

        def preRiskyShkvPfunc(risky_shk, a_nrm):
            return self.DiscFacEff * risky_shk * preIncShkvPfunc(a_nrm * risky_shk)

        self.preRiskyShkvPfunc, EndOfPrdvP = self.calc_ExpMargValueFunc(
            self.RiskyDstn, preRiskyShkvPfunc, self.aNrmNow
        )

        self.EndOfPrdvPfunc = self.preRiskyShkvPfunc

        return EndOfPrdvP

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

            preIncShkvPfunc = self.calc_preIncShkvPfunc(self.vPfuncNext)

            EndOfPrdvP = self.calc_preRiskyShkvPfunc(preIncShkvPfunc)

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

            self.EndOfPrdvPfunc, EndOfPrdvP = self.calc_ExpMargValueFunc(
                self.ShockDstn, vP_next, self.aNrmNow
            )

        return EndOfPrdvP

    def calc_ExpValueFunc(self, dstn, func, grid):
        """
        Calculate Expected Value Function given distribution,
        function, and interpolating nodes.
        """

        vals = calc_expectation(dstn, func, grid)
        nvrs = self.uinv(vals)
        nvrsFunc = LinearInterp(grid, nvrs)
        valueFunc = ValueFuncCRRA(nvrsFunc, self.CRRA)

        return valueFunc, vals

    def calc_preIncShkvFunc(self, vFuncNext):
        """
        Calculate Expected Value Function prior to realization
        of income uncertainty.
        """

        def preTranShkvFunc(tran_shk, w_nrm):
            return vFuncNext(w_nrm + tran_shk)

        self.preTranShkvFunc, _ = self.calc_ExpValueFunc(
            self.TranShkDstn, preTranShkvFunc, self.wNrmNext
        )

        def prePermShkvFunc(perm_shk, b_nrm):
            shock = perm_shk * self.PermGroFac
            return shock ** (1.0 - self.CRRA) * self.preTranShkvFunc(b_nrm / shock)

        self.prePermShkvFunc, _ = self.calc_ExpValueFunc(
            self.PermShkDstn, prePermShkvFunc, self.bNrmNext
        )

        preIncShkvFunc = self.prePermShkvFunc

        return preIncShkvFunc

    def calc_preRiskyShkvFunc(self, preIncShkvFunc):
        """
        Calculate Expected Value Function prior to
        realization of risky return.
        """

        def preRiskyShkvFunc(risky_shk, a_nrm):
            return self.DiscFacEff * preIncShkvFunc(risky_shk * a_nrm)

        self.preRiskyShkvFunc, EndOfPrdv = self.calc_ExpValueFunc(
            self.RiskyDstn, preRiskyShkvFunc, self.aNrmNow
        )

        self.EndOfPrdvFunc = self.preRiskyShkvFunc

        return EndOfPrdv

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

            preIncShkvFunc = self.calc_preIncShkvFunc(self.vFuncNext)

            EndOfPrdv = self.calc_preRiskyShkvFunc(preIncShkvFunc)

        else:

            def v_next(shocks, a_nrm):
                perm_shk = shocks[0] * self.PermGroFac
                mNrm_next = a_nrm * shocks[2] / perm_shk + shocks[1]
                return (
                    self.DiscFacEff
                    * perm_shk ** (1.0 - self.CRRA)
                    * self.vFuncNext(mNrm_next)
                )

            self.EndOfPrdvFunc, EndOfPrdv = self.calc_ExpValueFunc(
                self.ShockDstn, v_next, self.aNrmNow
            )


@dataclass
class ConsPortfolioRiskyAssetIndShkSolver(ConsRiskyAssetIndShkSolver):
    ShareGrid: np.array
    ShareLimit: float

    def prepare_to_calc_EndOfPrdvP(self):
        """
        Prepare to calculate end-of-period marginal value by creating an array
        of market resources that the agent could have next period, considering
        the grid of end-of-period assets and the distribution of shocks he might
        experience next period.

        Parameters
        ----------
        none

        Returns
        -------
        aNrmNow : np.array
            A 1D array of end-of-period assets; also stored as attribute of self.
        """

        super().prepare_to_calc_EndOfPrdvP()

        self.aMat, self.sMat = np.meshgrid(self.aNrmNow, self.ShareGrid, indexing="ij")

        return self.aNrmNow

    def optimize_share(self, EndOfPrddvds):
        """
        Optimize the risky share of portfolio given End of Period
        Marginal Value wrt a given risky share. Returns optimal share
        and End of Period Marginal Value of Liquid Assets at the optimal share.
        """

        # For each value of aNrm, find the value of Share such that FOC-Share == 0.
        crossing = np.logical_and(
            EndOfPrddvds[:, 1:] <= 0.0, EndOfPrddvds[:, :-1] >= 0.0
        )
        share_idx = np.argmax(crossing, axis=1)
        a_idx = np.arange(self.aNrmNow.size)
        bot_s = self.ShareGrid[share_idx]
        top_s = self.ShareGrid[share_idx + 1]
        bot_f = EndOfPrddvds[a_idx, share_idx]
        top_f = EndOfPrddvds[a_idx, share_idx + 1]

        alpha = 1.0 - top_f / (top_f - bot_f)

        risky_share_optimal = (1.0 - alpha) * bot_s + alpha * top_s

        # If agent wants to put more than 100% into risky asset, he is constrained
        constrained_top = EndOfPrddvds[:, -1] > 0.0
        # Likewise if he wants to put less than 0% into risky asset
        constrained_bot = EndOfPrddvds[:, 0] < 0.0

        # For values of aNrm at which the agent wants to put
        # more than 100% into risky asset, constrain them
        risky_share_optimal[constrained_top] = 1.0
        risky_share_optimal[constrained_bot] = 0.0

        if not self.zero_bound:
            # aNrm=0, so there's no way to "optimize" the portfolio
            risky_share_optimal[0] = 1.0

        return risky_share_optimal

    def calc_preRiskyShkvPfunc(self, preIncShkvPfunc):
        """
        Calculate Expected Marginal Value Function before
        the realization of the risky return.
        """

        # Optimize portfolio share

        def endOfPrddvds(risky_shk, a_nrm, share):
            r_diff = risky_shk - self.Rfree
            r_port = self.Rfree + r_diff * share
            b_nrm = a_nrm * r_port
            return a_nrm * r_diff * preIncShkvPfunc(b_nrm)

        if True:

            EndOfPrddvds = calc_expectation(
                self.RiskyDstn, endOfPrddvds, self.aMat, self.sMat
            )
            EndOfPrddvds = EndOfPrddvds[:, :, 0]

            self.risky_share_optimal = self.optimize_share(EndOfPrddvds)

        else:

            def obj(share, a_nrm):
                return calc_expectation(self.RiskyDstn, endOfPrddvds, a_nrm, share)

            risky_share_optimal = np.empty_like(self.aNrmNow)

            for ai in range(self.aNrmNow.size):
                a_nrm = self.aNrmNow[ai]
                if a_nrm == 0:
                    risky_share_optimal[ai] = 1.0
                else:
                    try:
                        sol = root_scalar(
                            obj, bracket=[self.ShareLimit, 1.0], args=(a_nrm,)
                        )

                        if sol.converged:
                            risky_share_optimal[ai] = sol.root
                        else:
                            risky_share_optimal[ai] = 1.0

                    except ValueError:
                        risky_share_optimal[ai] = 1.0

            self.risky_share_optimal = risky_share_optimal

        def endOfPrddvda(risky_shk, a_nrm, share):
            r_diff = risky_shk - self.Rfree
            r_port = self.Rfree + r_diff * share
            b_nrm = a_nrm * r_port
            return r_port * preIncShkvPfunc(b_nrm)

        EndOfPrddvda = self.DiscFacEff * calc_expectation(
            self.RiskyDstn, endOfPrddvda, self.aNrmNow, self.risky_share_optimal
        )
        EndOfPrddvdaNvrs = self.uPinv(EndOfPrddvda)
        EndOfPrddvdaNvrsFunc = LinearInterp(self.aNrmNow, EndOfPrddvdaNvrs)
        EndOfPrddvdaFunc = MargValueFuncCRRA(EndOfPrddvdaNvrsFunc, self.CRRA)

        return EndOfPrddvda

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

            preIncShkvPfunc = self.calc_preIncShkvPfunc(self.vPfuncNext)

            EndOfPrdvP = self.calc_preRiskyShkvPfunc(preIncShkvPfunc)

        else:

            def endOfPrddvds(shocks, a_nrm, share):
                r_diff = shocks[2] - self.Rfree
                r_port = self.Rfree + r_diff * share
                b_nrm = a_nrm * r_port
                p_shk = self.PermGroFac * shocks[0]
                m_nrm = b_nrm / p_shk + shocks[1]

                return r_diff * a_nrm * p_shk ** (-self.CRRA) * self.vPfuncNext(m_nrm)

            EndOfPrddvds = calc_expectation(
                self.RiskyDstn, endOfPrddvds, self.aMat, self.sMat
            )
            EndOfPrddvds = EndOfPrddvds[:, :, 0]

            self.risky_share_optimal = self.optimize_share(EndOfPrddvds)

            def endOfPrddvda(shocks, a_nrm, share):
                r_diff = shocks[2] - self.Rfree
                r_port = self.Rfree + r_diff * share
                b_nrm = a_nrm * r_port
                p_shk = self.PermGroFac * shocks[0]
                m_nrm = b_nrm / p_shk + shocks[1]

                return r_port * p_shk ** (-self.CRRA) * self.vPfuncNext(m_nrm)

            EndOfPrddvda = self.DiscFacEff * calc_expectation(
                self.RiskyDstn, endOfPrddvda, self.aNrmNow, self.risky_share_optimal
            )

            EndOfPrddvdaNvrs = self.uPinv(EndOfPrddvda)
            EndOfPrddvdaNvrsFunc = LinearInterp(self.aNrmNow, EndOfPrddvdaNvrs)
            self.EndOfPrddvdaFunc = MargValueFuncCRRA(EndOfPrddvdaNvrsFunc, self.CRRA)

            EndOfPrdvP = EndOfPrddvda

        return EndOfPrdvP

    def add_ShareFunc(self, solution):
        """
        Construct the risky share function twice, once with respect
        to End of Period which depends on Liquid assets, and another
        with respect to Beginning of Period which depends on Cash on Hand.
        """

        if self.zero_bound:
            # add zero back on agrid
            self.EndOfPrdShareFunc = LinearInterp(
                np.append(0.0, self.aNrmNow),
                np.append(1.0, self.risky_share_optimal),
                intercept_limit=self.ShareLimit,
                slope_limit=0.0,
            )
        else:
            self.EndOfPrdShareFunc = LinearInterp(
                self.aNrmNow,
                self.risky_share_optimal,
                intercept_limit=self.ShareLimit,
                slope_limit=0.0,
            )

        self.ShareFunc = LinearInterp(
            np.append(0.0, self.mNrmNow),
            np.append(1.0, self.risky_share_optimal),
            intercept_limit=self.ShareLimit,
            slope_limit=0.0,
        )

        solution.EndOfPrdShareFunc = self.EndOfPrdShareFunc
        solution.ShareFunc = self.ShareFunc

        return solution

    def solve(self):
        solution = super().solve()

        solution = self.add_ShareFunc(solution)

        return solution


@dataclass
class ConsFixedPortfolioRiskyAssetIndShkSolver(ConsIndShockSolver):
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
    RiskyShareFixed: float
    PermGroFac: float
    BoroCnstArt: float
    aXtraGrid: np.array
    vFuncBool: bool
    CubicBool: bool
    IndepDstnBool: bool

    def __post_init__(self):
        self.def_utility_funcs()

    def r_port(self, shock):
        return self.Rfree + (shock - self.Rfree) * self.RiskyShareFixed

    def set_and_update_values(self, solution_next, IncShkDstn, LivPrb, DiscFac):
        """
        Unpacks some of the inputs (and calculates simple objects based on them),
        storing the results in self for use by other methods.  These include:
        income shocks and probabilities, next period's marginal value function
        (etc), the probability of getting the worst income shock next period,
        the patience factor, human wealth, and the bounding MPCs.

        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one period problem.
        IncShkDstn : distribution.DiscreteDistribution
            A DiscreteDistribution with a pmf
            and two point value arrays in X, order:
            permanent shocks, transitory shocks.
        LivPrb : float
            Survival probability; likelihood of being alive at the beginning of
            the succeeding period.
        DiscFac : float
            Intertemporal discount factor for future utility.

        Returns
        -------
        None
        """

        super().set_and_update_values(solution_next, IncShkDstn, LivPrb, DiscFac)

        # overwrite APFac

        def APFac_func(shock):
            # link to lecture notes #TODO
            r_port = self.r_port(shock)
            return ((r_port * self.DiscFacEff) ** (1.0 / self.CRRA)) / r_port

        self.APFac = calc_expectation(self.RiskyDstn, APFac_func)

        self.MPCminNow = 1.0 / (1.0 + self.APFac / solution_next.MPCmin)

        # overwrite human wealth

        def h_nrm_func(shock):
            # link to lecture notes #TODO
            r_port = self.r_port(shock)
            return self.PermGroFac / r_port * (self.Ex_IncNext + solution_next.hNrm)

        self.hNrmNow = calc_expectation(self.RiskyDstn, h_nrm_func)

        self.MPCmaxNow = 1.0 / (
            1.0
            + (self.WorstIncPrb ** (1.0 / self.CRRA))
            * self.APFac
            / solution_next.MPCmax
        )

        self.cFuncLimitIntercept = None
        self.cFuncLimitSlope = None

    def def_BoroCnst(self, BoroCnstArt):
        """
        Defines the constrained portion of the consumption function as cFuncNowCnst,
        an attribute of self.  Uses the artificial and natural borrowing constraints.

        Parameters
        ----------
        BoroCnstArt : float or None
            Borrowing constraint for the minimum allowable assets to end the
            period with.  If it is less than the natural borrowing constraint,
            then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
            rowing constraint.

        Returns
        -------
        none
        """

        # in worst case scenario, debt gets highest return possible
        self.RPortMax = (
            self.Rfree + (self.RiskyDstn.X.max() - self.Rfree) * self.RiskyShareFixed
        )

        # Calculate the minimum allowable value of money resources in this period
        self.BoroCnstNat = (
            (self.solution_next.mNrmMin - self.TranShkDstn.X.min())
            * (self.PermGroFac * self.PermShkDstn.X.min())
            / self.RPortMax
        )

        if BoroCnstArt is None:
            self.mNrmMinNow = self.BoroCnstNat
        else:
            self.mNrmMinNow = np.max([self.BoroCnstNat, BoroCnstArt])
        if self.BoroCnstNat < self.mNrmMinNow:
            self.MPCmaxEff = 1.0  # If actually constrained, MPC near limit is 1
        else:
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

        def vp_next(shocks, a_nrm):
            r_port = self.r_port(shocks[2])
            p_shk = self.PermGroFac * shocks[0]
            m_nrm_next = a_nrm * r_port / p_shk + shocks[1]
            return r_port * p_shk ** (-self.CRRA) * self.vPfuncNext(m_nrm_next)

        EndOfPrdvP = self.DiscFacEff * calc_expectation(
            self.ShockDstn, vp_next, self.aNrmNow
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

        def v_next(shocks, a_nrm):
            r_port = self.Rfree + (shocks[2] - self.Rfree) * self.RiskyShareFixed
            m_nrm_next = r_port / (self.PermGroFac * shocks[0]) * a_nrm + shocks[1]
            return shocks[0] ** (1.0 - self.CRRA) * self.vFuncNext(m_nrm_next)

        EndOfPrdv = (
            self.DiscFacEff
            * self.PermGroFac ** (1.0 - self.CRRA)
            * calc_expectation(self.ShockDstn, v_next, self.aNrmNow)
        )
        # value transformed through inverse utility
        EndOfPrdvNvrs = self.uinv(EndOfPrdv)
        aNrm_temp = np.insert(self.aNrmNow, 0, self.BoroCnstNat)
        EndOfPrdvNvrsFunc = LinearInterp(aNrm_temp, EndOfPrdvNvrs)
        self.EndOfPrdvFunc = ValueFuncCRRA(EndOfPrdvNvrsFunc, self.CRRA)


# Initial parameter sets

# Base risky asset dictionary

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

# Number of discrete points in the risky share approximation
init_risky_asset["ShareCount"] = 25

init_risky_share_fixed = init_risky_asset.copy()
init_risky_share_fixed["RiskyShareFixed"] = [0.0]


class RiskyAssetShkDiscrete(DiscreteDistribution):
    pass


class RiskyAssetShkLognorm(RiskyAssetShkDiscrete):
    """
    A one-period distribution of a multiplicative lognormal risky asset shock.

    Parameters
    ----------
    mu : float
        Mean of the lognormal distribution.
    sigma : float
        Standard deviation of the lognormal distribution.
    n_approx : int
        Number of points to use in the discrete approximation.
    seed : int, optional
        Random seed. The default is 0.

    Returns
    -------
    RiskyShkDstn : DiscreteDistribution
        Risky asset shock distribution.

    """

    def __init__(self, mu, sigma, n_approx, seed=0):
        # Construct an auxiliary discretized normal
        logn_approx = Lognormal.from_mean_std(mean=mu, std=sigma).approx(
            n_approx if sigma > 0.0 else 1, tail_N=0
        )

        super().__init__(pmf=logn_approx.pmf, X=logn_approx.X, seed=seed)


class RiskyAssetShkWithDisaster(RiskyAssetShkDiscrete):
    def __init__(self, pmf, X, DisasterPrb, DisasterShk, seed=0):
        super().__init__(pmf, X, seed)

        if DisasterPrb > 0.0:
            dstn_approx = add_discrete_outcome_constant_mean(
                dstn_approx, p=DisasterPrb, x=DisasterShk
            )


class RiskyAssetShkLognormWithDisaster(RiskyAssetShkLognorm):
    """
    A one-period distribution for risky return shocks that are a mixture
    between a log-normal and a single-value disaster shock.

    Parameters
    ----------
    mu : float
        Mean of the lognormal distribution.
    sigma : float
        Standard deviation of the log-shock.
    DisasterPrb : float
        Probability of the "disaster" shock.
    DisasterShk : float
        Return shock in the "disaster" state.
    n_approx : int
        Number of points to use in the discrete approximation.
    seed : int, optional
        Random seed. The default is 0.

    Returns
    -------
    RiskyShkDstn : DiscreteDistribution
        Risky asset shock distribution.

    """

    def __init__(self, mu, sigma, DisasterPrb, DisasterShk, n_approx, seed=0):

        super().__init__(
            mu, sigma, n_approx, pmf=dstn_approx.pmf, X=dstn_approx.X, seed=seed
        )

        if DisasterPrb > 0.0:
            dstn_approx = add_discrete_outcome_constant_mean(
                dstn_approx, p=DisasterPrb, x=DisasterShk
            )


class JointLognormalPermShkAndRiskyShk(DiscreteDistribution):
    pass

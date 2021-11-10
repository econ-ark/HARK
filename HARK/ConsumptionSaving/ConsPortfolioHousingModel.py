"""
This file contains classes and functions for representing, solving, and simulating agents
who must allocate their resources among consumption, risky or rental housing, saving in a
risk-free asset (with a low return), and saving in a risky asset (with higher average return).
"""
from copy import copy, deepcopy

import numpy as np
from numba import njit, prange
from scipy.optimize import minimize_scalar

from HARK import MetricObject, make_one_period_oo_solver, NullFunc
from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    utility,
    utilityP,
    utilityP_inv,
    utility_inv,
    utility_invP,
)
from HARK.ConsumptionSaving.ConsPortfolioModel import (
    PortfolioSolution,
    PortfolioConsumerType,
    init_portfolio,
    ConsPortfolioSolver,
)
from HARK.distribution import (
    Lognormal,
    combine_indep_dstns,
    calc_expectation,
    Bernoulli,
)
from HARK.interpolation import (
    LinearInterp,
    IdentityFunction,
    ValueFuncCRRA,
    LinearInterpOnInterp1D,
    BilinearInterp,
    MargValueFuncCRRA,
    TrilinearInterp,
    CubicInterp,
)


class PortfolioRiskyHousingSolution(MetricObject):
    distance_criteria = ["vPfuncRnt", "vPfuncHse"]

    def __init__(
        self,
        cFuncRnt=NullFunc(),
        hseFuncRnt=NullFunc(),
        totExpFuncRnt=NullFunc(),
        ShareFuncRnt=NullFunc(),
        vFuncRnt=NullFunc(),
        vPfuncRnt=NullFunc(),
        cFuncHse=NullFunc(),
        ShareFuncHse=NullFunc(),
        vFuncHse=NullFunc(),
        vPfuncHse=NullFunc(),
    ):
        # Set attributes of self
        self.cFuncRnt = cFuncRnt
        self.hseFuncRnt = hseFuncRnt
        self.totExpFuncRnt = totExpFuncRnt
        self.cFuncHse = cFuncHse
        self.ShareFuncRnt = ShareFuncRnt
        self.ShareFuncHse = ShareFuncHse
        self.vFuncRnt = vFuncRnt
        self.vFuncHse = vFuncHse
        self.vPfuncRnt = vPfuncRnt
        self.vPfuncHse = vPfuncHse


class PortfolioRentalHousingType(PortfolioConsumerType):
    """
    A consumer type with rental housing and a portfolio choice. This agent type has
    log-normal return factors. Their problem is defined by a coefficient of relative
    risk aversion, share of expenditures spent on rental housing, intertemporal
    discount factor, risk-free interest factor, and time sequences of permanent income
    growth rate, survival probability, and permanent and transitory income shock
    standard deviations (in logs).  The agent may also invest in a risky asset, which
    has a higher average return than the risk-free asset. He *might* have age-varying
    beliefs about the risky-return; if he does, then "true" values of the risky
    asset's return distribution must also be specified.
    """

    time_inv_ = deepcopy(PortfolioConsumerType.time_inv_)
    time_inv_ = time_inv_ + ["RntHseShare"]

    def __init__(self, cycles=1, verbose=False, quiet=False, **kwds):
        params = init_portfolio_housing.copy()
        params.update(kwds)
        kwds = params

        # Initialize a basic consumer type
        PortfolioConsumerType.__init__(
            self, cycles=cycles, verbose=verbose, quiet=quiet, **kwds
        )

        self.solve_one_period = make_one_period_oo_solver(
            ConsPortfolioRentalHousingSolver
        )

        if not hasattr(self, "RntHseShare"):
            raise Exception(
                "Portfolio Choice with Risky Housing must have a RntHseShare parameter."
            )

    def update(self):
        IndShockConsumerType.update(self)
        self.update_AdjustPrb()
        self.update_human_wealth()
        self.update_RiskyShares()
        self.update_RiskyDstn()
        self.update_ShockDstn()
        self.update_ShareGrid()
        self.update_ShareLimit()

    def update_solution_terminal(self):
        PortfolioConsumerType.update_solution_terminal(self)
        self.solution_terminal.hNrm = 0

    def update_human_wealth(self):
        hNrm = np.empty(self.T_cycle + 1)
        hNrm[-1] = 0.0
        for t in range(self.T_cycle - 1, -1, -1):
            IncShkDstn = self.IncShkDstn[t]
            ShkPrbsNext = IncShkDstn.pmf
            PermShkValsNext = IncShkDstn.X[0]
            TranShkValsNext = IncShkDstn.X[1]

            # Calculate human wealth this period
            Ex_IncNext = np.dot(ShkPrbsNext, TranShkValsNext * PermShkValsNext)
            hNrm[t] = self.PermGroFac[t] / self.Rfree * (Ex_IncNext + hNrm[t + 1])

        self.hNrm = hNrm

    def update_RiskyShares(self):

        if self.ExRiskyShareBool:
            if type(self.ExRiskyShare) is list:
                if len(self.ExRiskyShare) == self.T_cycle:
                    self.add_to_time_vary("ExRiskyShare")
                else:
                    raise AttributeError(
                        "If ExRiskyShare is time-varying, it must have length of T_cycle!"
                    )
            else:
                self.add_to_time_inv("ExRiskyShare")

        if "ExRiskyShare" in self.time_vary:
            self.RiskyAvg = []
            self.RiskyStd = []
            for t in range(self.T_cycle):
                mean = self.RiskyAvgTrue
                std = self.RiskyStdTrue
                mean_squared = mean ** 2
                variance = std ** 2
                mu = np.log(mean_squared / (np.sqrt(mean_squared + variance)))
                sigma = np.sqrt(np.log(1.0 + variance / mean_squared))

                ratio = (self.WlthNrmAvg[t] + self.hNrm[t]) / (
                    self.CRRA * self.ExRiskyShare[t] * self.WlthNrmAvg[t]
                )

                if self.FixRiskyAvg and self.FixRiskyStd:
                    # This case ignores exogenous risky shares as option parameters indicate
                    # fixing both RiskyAvg and RiskyStd to their true values
                    self.RiskyAvg.append(self.RiskyAvgTrue)
                    self.RiskyStd.append(self.RiskyStdTrue)
                elif self.FixRiskyStd:
                    # There is no analytical solution for this case, so we look for a numerical one
                    risky_share = (
                        lambda x: np.log(x / self.Rfree)
                        * (1.0 + self.hNrm[t] / self.WlthNrmAvg[t])
                        / (self.CRRA * np.log(1 + variance / x ** 2))
                        - self.ExRiskyShare[t]
                    )

                    res = minimize_scalar(
                        risky_share, bounds=(mean, 2), method="bounded"
                    )
                    RiskyAvg = res.x

                    self.RiskyAvg.append(RiskyAvg)
                    self.RiskyStd.append(self.RiskyStdTrue)
                elif self.FixRiskyAvg:
                    # This case has an analytical solution

                    RiskyVar = ((mean / self.Rfree) ** ratio - 1) * mean_squared

                    self.RiskyAvg.append(self.RiskyAvgTrue)
                    self.RiskyStd.append(np.sqrt(RiskyVar))
                else:
                    # There are 2 ways to do this one, but not implemented yet
                    raise NotImplementedError(
                        "The case when RiskyAvg and RiskyStd are both not fixed is not implemented yet."
                    )

    def post_solve(self):

        for i in range(self.T_age):
            TotalExpAdj = copy(self.solution[i].cFuncAdj)
            self.solution[i].TotalExpAdj = TotalExpAdj

            if isinstance(TotalExpAdj, LinearInterp):

                x_list = TotalExpAdj.x_list
                y_list = TotalExpAdj.y_list

                self.solution[i].cFuncAdj = LinearInterp(
                    x_list, (1 - self.RntHseShare) * y_list
                )
                self.solution[i].hFuncAdj = LinearInterp(
                    x_list, self.RntHseShare * y_list
                )

            elif isinstance(TotalExpAdj, IdentityFunction):

                x_list = np.array([0, 1])
                y_list = np.array([0, 1])

                self.solution[i].cFuncAdj = LinearInterp(
                    x_list, (1 - self.RntHseShare) * y_list
                )
                self.solution[i].hFuncAdj = LinearInterp(
                    x_list, self.RntHseShare * y_list
                )


class ConsPortfolioRentalHousingSolver(MetricObject):
    def __init__(
        self,
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
        vFuncBool,
        AdjustPrb,
        DiscreteShareBool,
        ShareLimit,
        IndepDstnBool,
    ):
        self.solution_next = solution_next
        self.ShockDstn = ShockDstn
        self.IncShkDstn = IncShkDstn
        self.RiskyDstn = RiskyDstn
        self.LivPrb = LivPrb
        self.DiscFac = DiscFac
        self.CRRA = CRRA
        self.Rfree = Rfree
        self.PermGroFac = PermGroFac
        self.BoroCnstArt = BoroCnstArt
        self.aXtraGrid = aXtraGrid
        self.ShareGrid = ShareGrid
        self.vFuncBool = vFuncBool
        self.AdjustPrb = AdjustPrb
        self.DiscreteShareBool = DiscreteShareBool
        self.ShareLimit = ShareLimit
        self.IndepDstnBool = IndepDstnBool

    def add_human_wealth(self):
        self.ShkPrbsNext = self.IncShkDstn.pmf
        self.PermShkValsNext = self.IncShkDstn.X[0]
        self.TranShkValsNext = self.IncShkDstn.X[1]

        # Calculate human wealth this period
        self.Ex_IncNext = np.dot(
            self.ShkPrbsNext, self.TranShkValsNext * self.PermShkValsNext
        )
        self.hNrmNow = (
            self.PermGroFac / self.Rfree * (self.Ex_IncNext + self.solution_next.hNrm)
        )

        return self.hNrmNow

    def solve(self):
        solution = ConsPortfolioSolver(
            self.solution_next,
            self.ShockDstn,
            self.IncShkDstn,
            self.RiskyDstn,
            self.LivPrb,
            self.DiscFac,
            self.CRRA,
            self.Rfree,
            self.PermGroFac,
            self.BoroCnstArt,
            self.aXtraGrid,
            self.ShareGrid,
            self.vFuncBool,
            self.AdjustPrb,
            self.DiscreteShareBool,
            self.ShareLimit,
            self.IndepDstnBool,
        ).solve()

        solution.hNrm = self.add_human_wealth()

        return solution


class PortfolioRiskyHousingType(PortfolioConsumerType):
    time_inv_ = deepcopy(PortfolioConsumerType.time_inv_)
    time_inv_ += ["HouseShare", "HseDiscFac", "RntHseShare", "HseInitPrice"]
    time_vary_ = deepcopy(PortfolioConsumerType.time_vary_)
    time_vary_ += ["RentPrb", "HseGroFac"]
    shock_vars_ = PortfolioConsumerType.shock_vars_ + ["RntShk", "HouseShk"]
    state_vars = PortfolioConsumerType.state_vars + ["haveHse", "hNrm"]
    track_vars = ["mNrm", "hNrm", "haveHse", "cNrm", "aNrm", "pLvl", "aLvl", "Share"]

    def __init__(self, cycles=1, verbose=False, quiet=False, **kwds):
        params = init_portfolio_risky_housing.copy()
        params.update(kwds)
        kwds = params

        # Initialize a basic consumer type
        PortfolioConsumerType.__init__(
            self, cycles=cycles, verbose=verbose, quiet=quiet, **kwds
        )

        self.solve_one_period = make_one_period_oo_solver(
            ConsPortfolioRiskyHousingSolver
        )

    def update_HouseDstn(self):
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
            (type(self.HouseAvg) is list)
            and (type(self.HouseStd) is list)
            and (len(self.HouseAvg) == len(self.HouseStd))
            and (len(self.HouseAvg) == self.T_cycle)
        ):
            self.add_to_time_vary("HouseAvg", "HouseStd")
        elif (type(self.HouseStd) is list) or (type(self.HouseAvg) is list):
            raise AttributeError(
                "If HouseAvg is time-varying, then HouseStd must be as well, and they must both have length of T_cycle!"
            )
        else:
            self.add_to_time_inv("HouseAvg", "HouseStd")

        # Generate a discrete approximation to the risky return distribution if the
        # agent has age-varying beliefs about the risky asset
        if "HouseAvg" in self.time_vary:
            self.HouseDstn = []
            for t in range(self.T_cycle):
                self.HouseDstn.append(
                    Lognormal.from_mean_std(self.HouseAvg[t], self.HouseStd[t]).approx(
                        self.HouseShkCount
                    )
                )
            self.add_to_time_vary("HouseDstn")

        # Generate a discrete approximation to the risky return distribution if the
        # agent does *not* have age-varying beliefs about the risky asset (base case)
        else:
            self.HouseDstn = Lognormal.from_mean_std(
                self.HouseAvg,
                self.HouseStd,
            ).approx(self.HouseShkCount)
            self.add_to_time_inv("HouseDstn")

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
        if "HouseDstn" in self.time_vary:
            self.ShockDstn = [
                combine_indep_dstns(self.IncShkDstn[t], self.HouseDstn[t])
                for t in range(self.T_cycle)
            ]
        else:
            self.ShockDstn = [
                combine_indep_dstns(self.IncShkDstn[t], self.HouseDstn)
                for t in range(self.T_cycle)
            ]
        self.add_to_time_vary("ShockDstn")

        # Mark whether the risky returns, income shocks, and housing shocks are independent (they are)
        self.IndepDstnBool = True
        self.add_to_time_inv("IndepDstnBool")

    def update(self):
        IndShockConsumerType.update(self)
        self.update_AdjustPrb()
        self.update_RiskyDstn()
        self.update_HouseDstn()
        self.update_ShockDstn()
        self.update_ShareGrid()
        self.update_HouseGrid()
        self.update_ShareLimit()

    def update_solution_terminal(self):
        PortfolioConsumerType.update_solution_terminal(self)

        solution = portfolio_to_housing(self.solution_terminal, self.RntHseShare)

        self.solution_terminal = solution

    def update_HouseGrid(self):
        """
        Creates the attribute HouseGrid as an evenly spaced grid on [HouseMin,HouseMax], using
        the primitive parameter HouseCount.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.HouseGrid = np.linspace(self.HouseMin, self.HouseMax, self.HouseCount)
        self.add_to_time_inv("HouseGrid")

    def get_HouseShk(self):
        """
        Sets the attribute HouseShk as a single draw from a lognormal distribution.
        Uses the attributes HouseAvg and HouseStd.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        HouseAvg = self.HouseAvg
        HouseStd = self.HouseStd
        HouseAvgSqrd = HouseAvg ** 2
        HouseVar = HouseStd ** 2

        mu = np.log(HouseAvg / (np.sqrt(1.0 + HouseVar / HouseAvgSqrd)))
        sigma = np.sqrt(np.log(1.0 + HouseVar / HouseAvgSqrd))
        self.shocks["HouseShk"] = Lognormal(
            mu, sigma, seed=self.RNG.randint(0, 2 ** 31 - 1)
        ).draw(1)

    def get_RentShk(self):
        """
        Sets the attribute RentShk as a boolean array of size AgentCount, indicating
        whether each agent is forced to liquidate their house this period.
        Uses the attribute RentPrb to draw from a Bernoulli distribution.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if not ("RentPrb" in self.time_vary):

            self.shocks["RentShk"] = Bernoulli(
                self.RentPrb, seed=self.RNG.randint(0, 2 ** 31 - 1)
            ).draw(self.AgentCount)

        else:

            RntShk = np.zeros(self.AgentCount, dtype=bool)  # Initialize shock array
            for t in range(self.T_cycle):
                these = t == self.t_cycle
                N = np.sum(these)
                if N > 0:
                    if t == 0:
                        RentPrb = 0.0
                    else:
                        RentPrb = self.RentPrb[t - 1]
                    RntShk[these] = Bernoulli(
                        RentPrb, seed=self.RNG.randint(0, 2 ** 31 - 1)
                    ).draw(N)

            self.shocks["RentShk"] = RntShk

    def get_shocks(self):
        """
        Draw shocks as in PortfolioConsumerType, then draw
        a single common value for the House price shock. Also draws whether each
        agent is forced to rent next period.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        PortfolioConsumerType.get_shocks(self)
        self.get_HouseShk()
        self.get_RentShk()

    def get_states(self):
        PortfolioConsumerType.get_states(self)

        # previous house size
        hNrmPrev = self.state_prev["hNrm"]

        # new house size
        self.state_now["hNrm"] = (
            np.array(self.HseGroFac)[self.t_cycle] * hNrmPrev / self.shocks["PermShk"]
        )

        # cash on hand in case of liquidation
        mRntNrmNow = (
            self.state_now["mNrm"] + self.state_now["hNrm"] * self.shocks["HouseShk"]
        )

        # find index for households that were previously homeowners but
        # will no longer be homeowners next period
        # state_prev["haveHse"] = True and
        # shocks["RentShk"] = True
        trans_idx = np.logical_and(self.state_prev["haveHse"], self.shocks["RentShk"])

        # only change state for agents who were previously homeowners
        # they may stay homeowners or become renters
        self.state_now["haveHse"] = self.state_prev["haveHse"].copy()
        self.state_now["haveHse"][trans_idx] = False

        # if households went from homeowner to renter, they
        # receive their liquidation value as cash on hand
        self.state_now["mNrm"][trans_idx] = mRntNrmNow[trans_idx]

        return None

    def get_controls(self):
        """
        Calculates consumption cNrmNow and risky portfolio share ShareNow using
        the policy functions in the attribute solution.  These are stored as attributes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        cNrmNow = np.zeros(self.AgentCount) + np.nan
        ShareNow = np.zeros(self.AgentCount) + np.nan

        # Loop over each period of the cycle, getting controls separately depending on "age"
        for t in range(self.T_cycle):
            these = t == self.t_cycle

            # Get controls for agents who are renters
            those = np.logical_and(these, self.shocks["RentShk"])
            cNrmNow[those] = self.solution[t].cFuncRnt(self.state_now["mNrm"][those])
            ShareNow[those] = self.solution[t].ShareFuncRnt(
                self.state_now["mNrm"][those]
            )

            # Get Controls for agents who are homeowners
            those = np.logical_and(these, np.logical_not(self.shocks["RentShk"]))
            cNrmNow[those] = self.solution[t].cFuncHse(
                self.state_now["mNrm"][those], self.state_now["hNrm"][those]
            )
            ShareNow[those] = self.solution[t].ShareFuncHse(
                self.state_now["mNrm"][those], self.state_now["hNrm"][those]
            )

        # Store controls as attributes of self
        self.controls["cNrm"] = cNrmNow
        self.controls["Share"] = ShareNow

    def sim_birth(self, which_agents):
        """
        Create new agents to replace ones who have recently died; takes draws of
        initial aNrm and pLvl, as in PortfolioConsumerType, then sets RentShk
        to zero as initial values.
        Parameters
        ----------
        which_agents : np.array
            Boolean array of size AgentCount indicating which agents should be "born".

        Returns
        -------
        None
        """

        # Get and store states for newly born agents

        # for now, agents start being homeowners and
        # the distribution of houses is uniform
        self.state_now["haveHse"][which_agents] = True
        N = np.sum(which_agents)  # Number of new consumers to make
        self.state_now["hNrm"][which_agents] = np.linspace(1.0, 10.0, N)

        PortfolioConsumerType.sim_birth(self, which_agents)

    def initialize_sim(self):
        """
        Initialize the state of simulation attributes.  Simply calls the same method
        for PortfolioConsumerType, then sets the type of RentShk to bool.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.state_now["haveHse"] = np.zeros(self.AgentCount, dtype=bool)
        PortfolioConsumerType.initialize_sim(self)

    def get_poststates(self):
        """
        Calculates end-of-period assets for each consumer of this type.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.state_now["aNrm"] = (
            self.state_now["mNrm"]
            - self.controls["cNrm"]
            - (np.array(self.HseGroFac)[self.t_cycle] - (1 - self.HseDiscFac))
            * self.HseInitPrice
            * self.state_now["hNrm"]
        )
        # Useful in some cases to precalculate asset level
        self.state_now["aLvl"] = self.state_now["aNrm"] * self.state_now["pLvl"]


class MargValueFuncHousing(MetricObject):
    distance_criteria = ["cFunc", "CRRA"]

    def __init__(self, cFunc, HouseGrid, CRRA, HouseShare):
        self.cFunc = deepcopy(cFunc)
        self.hseGrid = HouseGrid
        self.CRRA = CRRA
        self.HouseShare = HouseShare

    def __call__(self, m_nrm, h_nrm):
        """
        Evaluate the marginal value function at given levels of market resources m.

        Parameters
        ----------
        cFuncArgs : floats or np.arrays
            Values of the state variables at which to evaluate the marginal
            value function.

        Returns
        -------
        vP : float or np.array
            Marginal lifetime value of beginning this period with state
            cFuncArgs
        """
        c_opt = self.cFunc(m_nrm, h_nrm)
        x_comp = c_opt ** (1 - self.HouseShare) * h_nrm ** self.HouseShare
        return utilityP(x_comp, gam=self.CRRA) * (h_nrm / c_opt) ** self.HouseShare


class ConsPortfolioRiskyHousingSolver(ConsPortfolioSolver):
    """
    Define an object-oriented one period solver.
    Solve the one period problem for a portfolio-choice consumer.
    This solver is used when the income and risky return shocks
    are independent and the allowed optimal share is continuous.

    Parameters
    ----------
    solution_next : PortfolioSolution
        Solution to next period's problem.
    ShockDstn : [np.array]
        List with four arrays: discrete probabilities, permanent income shocks,
        transitory income shocks, and risky returns.  This is only used if the
        input IndepDstnBool is False, indicating that income and return distributions
        can't be assumed to be independent.
    IncShkDstn : distribution.Distribution
        Discrete distribution of permanent income shocks
        and transitory income shocks.  This is only used if the input IndepDsntBool
        is True, indicating that income and return distributions are independent.
    RiskyDstn : [np.array]
        List with two arrays: discrete probabilities and risky asset returns. This
        is only used if the input IndepDstnBool is True, indicating that income
        and return distributions are independent.
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
        of the consumption function when Share is fixed.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.
    AdjustPrb : float
        Probability that the agent will be able to update his portfolio share.
    DiscreteShareBool : bool
        Indicator for whether risky portfolio share should be optimized on the
        continuous [0,1] interval using the FOC (False), or instead only selected
        from the discrete set of values in ShareGrid (True).  If True, then
        vFuncBool must also be True.
    ShareLimit : float
        Limiting lower bound of risky portfolio share as mNrm approaches infinity.
    IndepDstnBool : bool
        Indicator for whether the income and risky return distributions are in-
        dependent of each other, which can speed up the expectations step.
    """

    def __init__(
        self,
        solution_next,
        ShockDstn,
        IncShkDstn,
        RiskyDstn,
        HouseDstn,
        LivPrb,
        DiscFac,
        CRRA,
        Rfree,
        PermGroFac,
        HseGroFac,
        HseDiscFac,
        HseInitPrice,
        HouseShare,
        RntHseShare,
        BoroCnstArt,
        aXtraGrid,
        ShareGrid,
        HouseGrid,
        vFuncBool,
        RentPrb,
        DiscreteShareBool,
        ShareLimit,
    ):
        """
        Constructor for portfolio choice problem solver.
        """

        self.solution_next = solution_next
        self.ShockDstn = ShockDstn
        self.IncShkDstn = IncShkDstn
        self.RiskyDstn = RiskyDstn
        self.HouseDstn = HouseDstn
        self.LivPrb = LivPrb
        self.DiscFac = DiscFac
        self.CRRA = CRRA
        self.Rfree = Rfree
        self.PermGroFac = PermGroFac
        self.HseGroFac = HseGroFac
        self.HseDiscFac = HseDiscFac
        self.HouseShare = HouseShare
        self.HseInitPrice = HseInitPrice
        self.RntHseShare = RntHseShare
        self.BoroCnstArt = BoroCnstArt
        self.aXtraGrid = aXtraGrid
        self.ShareGrid = ShareGrid
        self.HouseGrid = HouseGrid
        self.vFuncBool = vFuncBool
        self.RentPrb = RentPrb
        self.DiscreteShareBool = DiscreteShareBool
        self.ShareLimit = ShareLimit

        # Make sure the individual is liquidity constrained.  Allowing a consumer to
        # borrow *and* invest in an asset with unbounded (negative) returns is a bad mix.
        if self.BoroCnstArt != 0.0:
            raise ValueError("PortfolioConsumerType must have BoroCnstArt=0.0!")

        # Make sure that if risky portfolio share is optimized only discretely, then
        # the value function is also constructed (else this task would be impossible).
        if self.DiscreteShareBool and (not self.vFuncBool):
            raise ValueError(
                "PortfolioConsumerType requires vFuncBool to be True when DiscreteShareBool is True!"
            )

        self.def_utility_funcs()

    def def_utility_funcs(self):
        """
        Define temporary functions for utility and its derivative and inverse
        """

        self.u = lambda x: utility(x, self.CRRA)
        self.uP = lambda x: utilityP(x, self.CRRA)
        self.uPinv = lambda x: utilityP_inv(x, self.CRRA)
        self.uinv = lambda x: utility_inv(x, self.CRRA)
        self.uinvP = lambda x: utility_invP(x, self.CRRA)

    def set_and_update_values(self):
        """
        Unpacks some of the inputs (and calculates simple objects based on them),
        storing the results in self for use by other methods.
        """

        # Unpack next period's solution
        self.vPfuncRnt_next = self.solution_next.vPfuncRnt
        self.vPfuncHse_next = self.solution_next.vPfuncHse
        self.vFuncRnt_next = self.solution_next.vFuncRnt
        self.vFuncHse_next = self.solution_next.vFuncHse

        # Unpack the shock distribution
        self.TranShks_next = self.IncShkDstn.X[1]
        self.Risky_next = self.RiskyDstn.X

        # Flag for whether the natural borrowing constraint is zero
        self.zero_bound = np.min(self.TranShks_next) == 0.0
        self.RiskyMax = np.max(self.Risky_next)
        self.RiskyMin = np.min(self.Risky_next)

        self.tmp_fac_A = (
            ((1.0 - self.RntHseShare) ** (1.0 - self.RntHseShare))
            * (self.RntHseShare ** self.RntHseShare)
        ) ** (1.0 - self.CRRA)

        # Shock positions in ShockDstn
        self.PermShkPos = 0
        self.TranShkPos = 1
        self.HseShkPos = 2

    def prepare_to_solve(self):
        """
        Perform preparatory work.
        """

        self.set_and_update_values()

    def prepare_to_calc_EndOfPrdvP(self):
        """
        Prepare to calculate end-of-period marginal values by creating an array
        of market resources that the agent could have next period, considering
        the grid of end-of-period assets and the distribution of shocks he might
        experience next period.
        """

        # bNrm represents R*a, balances after asset return shocks but before income.
        # This just uses the highest risky return as a rough shifter for the aXtraGrid.
        if self.zero_bound:
            self.aNrmGrid = self.aXtraGrid
            self.bNrmGrid = np.insert(
                self.RiskyMax * self.aXtraGrid,
                0,
                self.RiskyMin * self.aXtraGrid[0],
            )
        else:
            # Add an asset point at exactly zero
            self.aNrmGrid = np.insert(self.aXtraGrid, 0, 0.0)
            self.bNrmGrid = self.RiskyMax * np.insert(self.aXtraGrid, 0, 0.0)

        # Get grid and shock sizes, for easier indexing
        self.aNrm_N = self.aNrmGrid.size
        self.Share_N = self.ShareGrid.size
        self.House_N = self.HouseGrid.size

        # Make tiled arrays to calculate future realizations of mNrm and Share when integrating over IncShkDstn
        self.bNrm_tiled, self.House_tiled = np.meshgrid(
            self.bNrmGrid, self.HouseGrid, indexing="ij"
        )

        self.aNrm_2tiled, self.House_2tiled = np.meshgrid(
            self.aNrmGrid, self.HouseGrid, indexing="ij"
        )

        # Make tiled arrays to calculate future realizations of bNrm and Share when integrating over RiskyDstn
        self.aNrm_3tiled, self.House_3tiled, self.Share_3tiled = np.meshgrid(
            self.aNrmGrid, self.HouseGrid, self.ShareGrid, indexing="ij"
        )

    def m_nrm_next(self, shocks, b_nrm):
        """
        Calculate future realizations of market resources
        """

        return (
            b_nrm / (self.PermGroFac * shocks[self.PermShkPos])
            + shocks[self.TranShkPos]
        )

    def hse_nrm_next(self, shocks, hse_nrm):
        """
        Calculate future realizations of house size
        """

        return self.HseGroFac * hse_nrm / shocks[self.PermShkPos]

    def m_rnt_nrm_next(self, shocks, m_nrm, hse_nrm):
        """
        Calculate future realizations of market resources
        including house liquidation
        """

        return m_nrm + shocks[self.HseShkPos] * hse_nrm

    def calc_EndOfPrdvP(self):
        """
        Calculate end-of-period marginal value of assets and shares at each point
        in aNrm and ShareGrid. Does so by taking expectation of next period marginal
        values across income and risky return shocks.
        """

        def dvdb_dist(shocks, b_nrm, hse_nrm):
            """
            Evaluate realizations of marginal value of market resources next period
            """

            mNrm_next = self.m_nrm_next(shocks, b_nrm)
            hseNrm_next = self.hse_nrm_next(shocks, hse_nrm)
            mRntNrm_next = self.m_rnt_nrm_next(shocks, mNrm_next, hseNrm_next)

            dvdmRnt_next = self.tmp_fac_A * self.vPfuncRnt_next(mRntNrm_next)
            if self.RentPrb < 1.0:
                dvdmHse_next = self.vPfuncHse_next(mNrm_next, hseNrm_next)
                # Combine by adjustment probability
                dvdm_next = (
                    self.RentPrb * dvdmRnt_next + (1.0 - self.RentPrb) * dvdmHse_next
                )
            else:  # Don't bother evaluating if there's no chance that household keeps house
                dvdm_next = dvdmRnt_next

            return (self.PermGroFac * shocks[self.PermShkPos]) ** (
                -self.CRRA
            ) * dvdm_next

        # Evaluate realizations of marginal value of risky share next period
        # No marginal value of Share if it's a free choice!

        # Calculate intermediate marginal value of bank balances by taking expectations over income shocks
        dvdb_intermed = calc_expectation(
            self.ShockDstn, dvdb_dist, self.bNrm_tiled, self.House_tiled
        )
        dvdb_intermed = dvdb_intermed[:, :, 0]
        dvdbNvrs_intermed = self.uPinv(dvdb_intermed)
        dvdbNvrsFunc_intermed = BilinearInterp(
            dvdbNvrs_intermed, self.bNrmGrid, self.HouseGrid
        )
        dvdbFunc_intermed = MargValueFuncCRRA(dvdbNvrsFunc_intermed, self.CRRA)

        def EndOfPrddvda_dist(shock, a_nrm, hse_nrm, share):
            # Calculate future realizations of bank balances bNrm
            Rxs = shock - self.Rfree
            Rport = self.Rfree + share * Rxs
            b_nrm_next = Rport * a_nrm

            return Rport * dvdbFunc_intermed(b_nrm_next, hse_nrm)

        def EndOfPrddvds_dist(shock, a_nrm, hse_nrm, share):
            # Calculate future realizations of bank balances bNrm
            Rxs = shock - self.Rfree
            Rport = self.Rfree + share * Rxs
            b_nrm_next = Rport * a_nrm
            # No marginal value of Share if it's a free choice!
            return Rxs * a_nrm * dvdbFunc_intermed(b_nrm_next, hse_nrm)

        # Calculate end-of-period marginal value of assets by taking expectations
        EndOfPrddvda = (
            self.DiscFac
            * self.LivPrb
            * calc_expectation(
                self.RiskyDstn,
                EndOfPrddvda_dist,
                self.aNrm_3tiled,
                self.House_3tiled,
                self.Share_3tiled,
            )
        )
        EndOfPrddvda = EndOfPrddvda[:, :, :, 0]

        temp_fac_hse = (1.0 - self.HouseShare) * self.House_3tiled ** (
            self.HouseShare * (1.0 - self.CRRA)
        )
        c_opt = EndOfPrddvda / temp_fac_hse
        self.c_opt = c_opt ** (
            1 / (-self.CRRA * (1.0 - self.HouseShare) - self.HouseShare)
        )

        # Calculate end-of-period marginal value of risky portfolio share by taking expectations
        EndOfPrddvds = (
            self.DiscFac
            * self.LivPrb
            * calc_expectation(
                self.RiskyDstn,
                EndOfPrddvds_dist,
                self.aNrm_3tiled,
                self.House_3tiled,
                self.Share_3tiled,
            )
        )
        EndOfPrddvds = EndOfPrddvds[:, :, :, 0]
        self.EndOfPrddvds = EndOfPrddvds

    def optimize_share(self):
        """
        Optimization of Share on continuous interval [0,1]
        """

        # Initialize to putting everything in safe asset
        self.Share_now = np.zeros((self.aNrm_N, self.House_N))
        self.cNrmHse_now = np.zeros((self.aNrm_N, self.House_N))
        # For each value of hNrm, find the value of Share such that FOC-Share == 0.
        for h in range(self.House_N):
            # For values of aNrm at which the agent wants to put more than 100% into risky asset, constrain them
            FOC_s = self.EndOfPrddvds[:, h]
            # If agent wants to put more than 100% into risky asset, he is constrained
            constrained_top = FOC_s[:, -1] > 0.0
            # Likewise if he wants to put less than 0% into risky asset
            constrained_bot = FOC_s[:, 0] < 0.0
            # so far FOC never greater than 0.0
            self.Share_now[constrained_top, h] = 1.0
            if not self.zero_bound:
                # aNrm=0, so there's no way to "optimize" the portfolio
                self.Share_now[0, h] = 1.0
                # Consumption when aNrm=0 does not depend on Share
                self.cNrmHse_now[0, h] = self.c_opt[0, h, -1]
                # Mark as constrained so that there is no attempt at optimization
                constrained_top[0] = True

            # Get consumption when share-constrained
            self.cNrmHse_now[constrained_top, h] = self.c_opt[constrained_top, h, -1]
            self.cNrmHse_now[constrained_bot, h] = self.c_opt[constrained_bot, h, 0]
            # For each value of aNrm, find the value of Share such that FOC-Share == 0.
            # This loop can probably be eliminated, but it's such a small step that it won't speed things up much.
            crossing = np.logical_and(FOC_s[:, 1:] <= 0.0, FOC_s[:, :-1] >= 0.0)
            for j in range(self.aNrm_N):
                if not (constrained_top[j] or constrained_bot[j]):
                    idx = np.argwhere(crossing[j, :])[0][0]
                    bot_s = self.ShareGrid[idx]
                    top_s = self.ShareGrid[idx + 1]
                    bot_f = FOC_s[j, idx]
                    top_f = FOC_s[j, idx + 1]
                    bot_c = self.c_opt[j, h, idx]
                    top_c = self.c_opt[j, h, idx + 1]
                    alpha = 1.0 - top_f / (top_f - bot_f)
                    self.Share_now[j, h] = (1.0 - alpha) * bot_s + alpha * top_s
                    self.cNrmHse_now[j, h] = (1.0 - alpha) * bot_c + alpha * top_c

    def optimize_share_discrete(self):
        # Major method fork: discrete vs continuous choice of risky portfolio share
        if self.DiscreteShareBool:
            # Optimization of Share on the discrete set ShareGrid
            opt_idx = np.argmax(self.EndOfPrdv, axis=2)
            # Best portfolio share is one with highest value
            Share_now = self.ShareGrid[opt_idx]
            # Take cNrm at that index as well
            cNrmHse_now = self.c_opt[
                np.arange(self.aNrm_N), np.arange(self.House_N), opt_idx
            ]
            if not self.zero_bound:
                # aNrm=0, so there's no way to "optimize" the portfolio
                Share_now[0] = 1.0
                # Consumption when aNrm=0 does not depend on Share
                cNrmHse_now[0] = self.c_opt[0, :, -1]

    def make_basic_solution(self):
        """
        Given end of period assets and end of period marginal values, construct
        the basic solution for this period.
        """

        # Calculate the endogenous mNrm gridpoints when the agent adjusts his portfolio
        self.mNrmHse_now = (
            self.aNrm_2tiled
            + self.cNrmHse_now
            + (self.HseGroFac - (1.0 - self.HseDiscFac))
            * self.HseInitPrice
            * self.House_2tiled
        )

        self.mNrmMin = (
            (self.HseGroFac - (1.0 - self.HseDiscFac))
            * self.HseInitPrice
            * self.HouseGrid
        )

        # Construct the consumption function when the agent can adjust
        cNrmHse_by_hse = []
        cNrmHse_now = np.insert(self.cNrmHse_now, 0, 0.0, axis=0)
        mNrmHse_now_temp = np.insert(self.mNrmHse_now, 0, self.mNrmMin, axis=0)
        for h in range(self.House_N):
            cNrmHse_by_hse.append(
                LinearInterp(mNrmHse_now_temp[:, h], cNrmHse_now[:, h])
            )

        self.cFuncHse_now = LinearInterpOnInterp1D(cNrmHse_by_hse, self.HouseGrid)

        # Construct the marginal value (of mNrm) function when the agent can adjust
        # this needs to be reworked
        self.vPfuncHse_now = MargValueFuncHousing(
            self.cFuncHse_now, self.HouseGrid, self.CRRA, self.HouseShare
        )

    def make_ShareFuncHse(self):
        """
        Construct the risky share function when the agent can adjust
        """

        if self.zero_bound:
            Share_lower_bound = self.ShareLimit
        else:
            Share_lower_bound = 1.0
        Share_now = np.insert(self.Share_now, 0, Share_lower_bound, axis=0)
        mNrmHse_now_temp = np.insert(self.mNrmHse_now, 0, self.mNrmMin, axis=0)
        ShareFuncHse_by_hse = []
        for j in range(self.House_N):
            ShareFuncHse_by_hse.append(
                LinearInterp(
                    mNrmHse_now_temp[:, j],
                    Share_now[:, j],
                    intercept_limit=self.ShareLimit,
                    slope_limit=0.0,
                )
            )
        self.ShareFuncHse_now = LinearInterpOnInterp1D(
            ShareFuncHse_by_hse, self.HouseGrid
        )

    def make_ShareFuncHse_discrete(self):
        # TODO
        mNrmHse_mid = (self.mNrmHse_now[1:] + self.mNrmHse_now[:-1]) / 2
        mNrmHse_plus = mNrmHse_mid * (1.0 + 1e-12)
        mNrmHse_comb = (np.transpose(np.vstack((mNrmHse_mid, mNrmHse_plus)))).flatten()
        mNrmHse_comb = np.append(np.insert(mNrmHse_comb, 0, 0.0), self.mNrmHse_now[-1])
        Share_comb = (
            np.transpose(np.vstack((self.Share_now, self.Share_now)))
        ).flatten()
        self.ShareFuncHse_now = LinearInterp(mNrmHse_comb, Share_comb)

    def add_vFunc(self):
        """
        Creates the value function for this period and adds it to the solution.
        """

        self.make_EndOfPrdvFunc()
        self.make_vFunc()

    def make_EndOfPrdvFunc(self):
        """
        Construct the end-of-period value function for this period, storing it
        as an attribute of self for use by other methods.
        """

        # If the value function has been requested, evaluate realizations of value
        def v_intermed_dist(shocks, b_nrm, hse_nrm):
            mNrm_next = self.m_nrm_next(shocks, b_nrm)
            hseNrm_next = self.hse_nrm_next(shocks, hse_nrm)
            mRntNrm = self.m_rnt_nrm_next(shocks, mNrm_next, hseNrm_next)

            vRnt_next = self.tmp_fac_A * self.vFuncRnt_next(mRntNrm)
            if self.RentPrb < 1.0:
                # Combine by adjustment probability
                vHse_next = self.vFuncHse_next(mNrm_next, hseNrm_next)
                v_next = self.RentPrb * vRnt_next + (1.0 - self.RentPrb) * vHse_next
            else:  # Don't bother evaluating if there's no chance that household keeps house
                v_next = vRnt_next

            return (self.PermGroFac * shocks[self.PermShkPos]) ** (
                1.0 - self.CRRA
            ) * v_next

        # Calculate intermediate value by taking expectations over income shocks
        v_intermed = calc_expectation(
            self.ShockDstn, v_intermed_dist, self.bNrm_tiled, self.House_tiled
        )
        v_intermed = v_intermed[:, :, 0]
        vNvrs_intermed = self.uinv(v_intermed)
        vNvrsFunc_intermed = BilinearInterp(
            vNvrs_intermed, self.bNrmGrid, self.HouseGrid
        )
        vFunc_intermed = ValueFuncCRRA(vNvrsFunc_intermed, self.CRRA)

        def EndOfPrdv_dist(shock, a_nrm, hse_nrm, share):
            # Calculate future realizations of bank balances bNrm
            Rxs = shock - self.Rfree
            Rport = self.Rfree + share * Rxs
            b_nrm_next = Rport * a_nrm

            return vFunc_intermed(b_nrm_next, hse_nrm)

        # Calculate end-of-period value by taking expectations
        self.EndOfPrdv = (
            self.DiscFac
            * self.LivPrb
            * calc_expectation(
                self.RiskyDstn,
                EndOfPrdv_dist,
                self.aNrm_3tiled,
                self.House_3tiled,
                self.Share_3tiled,
            )
        )
        self.EndOfPrdv = self.EndOfPrdv[:, :, :, 0]
        self.EndOfPrdvNvrs = self.uinv(self.EndOfPrdv)

    def make_vFunc(self):
        """
        Creates the value functions for this period, defined over market
        resources m when agent can adjust his portfolio, and over market
        resources and fixed share when agent can not adjust his portfolio.
        self must have the attribute EndOfPrdvFunc in order to execute.
        """

        # First, make an end-of-period value function over aNrm and Share
        EndOfPrdvNvrsFunc = TrilinearInterp(
            self.EndOfPrdvNvrs, self.aNrmGrid, self.HouseGrid, self.ShareGrid
        )
        EndOfPrdvFunc = ValueFuncCRRA(EndOfPrdvNvrsFunc, self.CRRA)

        # Construct the value function when the agent can adjust his portfolio
        # Just use aXtraGrid as our grid of mNrm values
        mNrm = self.aXtraGrid

        mNrm_tiled, House_tiled = np.meshgrid(mNrm, self.HouseGrid, indexing="ij")

        cNrm = self.cFuncHse_now(mNrm_tiled, House_tiled)

        aNrm = (
            mNrm_tiled
            - (self.HseGroFac - (1.0 - self.HseDiscFac))
            * self.HseInitPrice
            * House_tiled
            - cNrm
        )

        Share_temp = self.ShareFuncHse_now(mNrm_tiled, House_tiled)
        # EndOfPrdvFunc needs to be 3D

        x_comp = (cNrm ** (1.0 - self.HouseShare)) * (House_tiled ** self.HouseShare)

        v_temp = self.u(x_comp) + EndOfPrdvFunc(aNrm, House_tiled, Share_temp)

        vNvrs_temp = self.uinv(v_temp)
        vNvrsP_temp = self.uP(x_comp) * self.uinvP(v_temp)

        vNvrsFuncHse_by_House = []

        for j in range(self.House_N):
            vNvrsFuncHse_by_House.append(
                CubicInterp(
                    np.insert(mNrm, 0, 0.0),  # x_list
                    np.insert(vNvrs_temp[:, j], 0, 0.0),  # f_list
                    np.insert(vNvrsP_temp[:, j], 0, vNvrsP_temp[0, j]),  # dfdx_list
                )
            )
        vNvrsFuncHse = LinearInterpOnInterp1D(vNvrsFuncHse_by_House, self.HouseGrid)
        # Re-curve the pseudo-inverse value function
        self.vFuncHse_now = ValueFuncCRRA(vNvrsFuncHse, self.CRRA)

    def solve_retired_renter_problem(self):
        sn = self.solution_next

        portfolio_sn = housing_to_portfolio(sn)

        AdjPrb = 1.0
        IndepDstnBool = True

        portfolio_solution = ConsPortfolioSolver(
            portfolio_sn,
            self.ShockDstn,
            self.IncShkDstn,
            self.RiskyDstn,
            self.LivPrb,
            self.DiscFac,
            self.CRRA,
            self.Rfree,
            self.PermGroFac,
            self.BoroCnstArt,
            self.aXtraGrid,
            self.ShareGrid,
            self.vFuncBool,
            AdjPrb,
            self.DiscreteShareBool,
            self.ShareLimit,
            IndepDstnBool,
        ).solve()

        self.rental_solution = portfolio_to_housing(
            portfolio_solution, self.RntHseShare
        )

    def solve_retired_homeowner_problem(self):
        """
        Solve the one period problem for a portfolio-choice consumer.

        Returns
        -------
        solution_now : PortfolioSolution
        The solution to the single period consumption-saving with portfolio choice
        problem.  Includes two consumption and risky share functions: one for when
        the agent can adjust his portfolio share (Adj) and when he can't (Fxd).
        """

        # Make arrays of end-of-period assets and end-of-period marginal values
        self.prepare_to_calc_EndOfPrdvP()
        self.calc_EndOfPrdvP()

        # Construct a basic solution for this period
        self.optimize_share()
        self.make_basic_solution()
        self.make_ShareFuncHse()

        # Add the value function if requested
        if self.vFuncBool:
            self.add_vFunc()
        else:  # If vFuncBool is False, fill in dummy values
            self.vFuncRnt_now = NullFunc()
            self.vFuncHse_now = NullFunc()

    def make_portfolio_housing_solution(self):
        self.solution = PortfolioRiskyHousingSolution(
            cFuncRnt=self.rental_solution.cFuncRnt,
            hseFuncRnt=self.rental_solution.hseFuncRnt,
            totExpFuncRnt=self.rental_solution.totExpFuncRnt,
            ShareFuncRnt=self.rental_solution.ShareFuncRnt,
            vFuncRnt=self.rental_solution.vFuncRnt,
            vPfuncRnt=self.rental_solution.vPfuncRnt,
            cFuncHse=self.cFuncHse_now,
            ShareFuncHse=self.ShareFuncHse_now,
            vFuncHse=self.vFuncHse_now,
            vPfuncHse=self.vPfuncHse_now,
        )

    def solve(self):
        self.solve_retired_renter_problem()
        self.solve_retired_homeowner_problem()

        self.make_portfolio_housing_solution()

        self.solution.RentPrb = self.RentPrb

        return self.solution

    @classmethod
    def from_agent(cls, agent, solution_next=None, t=-1):
        if solution_next is None:
            solution_next = agent.solution_terminal

        return cls(
            solution_next,
            agent.ShockDstn[t],
            agent.IncShkDstn[t],
            agent.RiskyDstn,
            agent.HouseDstn,
            agent.LivPrb[t],
            agent.DiscFac,
            agent.CRRA,
            agent.Rfree,
            agent.PermGroFac[t],
            agent.HseGroFac[t],
            agent.HseDiscFac,
            agent.HseInitPrice,
            agent.HouseShare,
            agent.RntHseShare,
            agent.BoroCnstArt,
            agent.aXtraGrid,
            agent.ShareGrid,
            agent.HouseGrid,
            agent.vFuncBool,
            agent.RentPrb[t],
            agent.DiscreteShareBool,
            agent.ShareLimit,
        )


@njit(parallel=True, cache=True)
def opt_continuous_share(EndOfPrddvds, EndOfPrddvdaNvrs, zero_bound, ShareGrid):
    """
    Optimization of Share on continuous interval [0,1]
    """

    # Obtain output dimensions
    aNrm_N = EndOfPrddvds.shape[0]
    House_N = EndOfPrddvds.shape[1]

    # Initialize to putting everything in safe asset
    Share_now = np.zeros((aNrm_N, House_N))
    cNrmHse_now = np.zeros((aNrm_N, House_N))

    # For values of aNrm at which the agent wants to put more than 100% into risky asset, constrain them
    FOC_s = np.ascontiguousarray(EndOfPrddvds)
    c_opt = np.ascontiguousarray(EndOfPrddvdaNvrs)
    # If agent wants to put more than 100% into risky asset, he is constrained
    constrained_top = FOC_s[:, :, -1] > 0.0
    # Likewise if he wants to put less than 0% into risky asset
    constrained_bot = FOC_s[:, :, 0] < 0.0

    if not zero_bound:
        # aNrm=0, so there's no way to "optimize" the portfolio
        Share_now[0] = 1.0
        # Consumption when aNrm=0 does not depend on Share
        cNrmHse_now[0] = c_opt[0, :, -1]
        # Mark as constrained so that there is no attempt at optimization
        constrained_top[0] = True

    crossing = np.logical_and(FOC_s[:, :, 1:] <= 0.0, FOC_s[:, :, :-1] >= 0.0)
    # For each value of aNrm, find the value of Share such that FOC-Share == 0.
    for j in prange(aNrm_N):
        # For each value of hNrm, find the value of Share such that FOC-Share == 0.
        for k in prange(House_N):
            if not (constrained_top[j, k] or constrained_bot[j, k]):
                idx = np.argwhere(crossing[j, k, :])[0][0]
                bot_s = ShareGrid[idx]
                top_s = ShareGrid[idx + 1]
                bot_f = FOC_s[j, k, idx]
                top_f = FOC_s[j, k, idx + 1]
                bot_c = c_opt[j, k, idx]
                top_c = c_opt[j, k, idx + 1]
                alpha = 1.0 - top_f / (top_f - bot_f)
                Share_now[j, k] = (1.0 - alpha) * bot_s + alpha * top_s
                cNrmHse_now[j, k] = (1.0 - alpha) * bot_c + alpha * top_c
            elif constrained_top[j, k]:
                # so far FOC never greater than 0.0
                Share_now[j, k] = 1.0
                # Get consumption when share-constrained
                cNrmHse_now[j, k] = c_opt[j, k, -1]
            elif constrained_bot[j, k]:
                # Get consumption when share-constrained
                cNrmHse_now[j, k] = c_opt[j, k, 0]

    return Share_now, cNrmHse_now


def portfolio_to_housing(ps, RntHseShare):
    totExpFuncRnt = ps.cFuncAdj
    if isinstance(totExpFuncRnt, LinearInterp):
        x_list = totExpFuncRnt.x_list
        y_list = totExpFuncRnt.y_list
        cFuncRnt = LinearInterp(x_list, (1 - RntHseShare) * y_list)
        hseFuncRnt = LinearInterp(x_list, RntHseShare * y_list)
    elif isinstance(totExpFuncRnt, IdentityFunction):
        x_list = np.array([0, 1])
        y_list = np.array([0, 1])
        cFuncRnt = LinearInterp(x_list, (1 - RntHseShare) * y_list)
        hseFuncRnt = LinearInterp(x_list, RntHseShare * y_list)

    return PortfolioRiskyHousingSolution(
        cFuncRnt=cFuncRnt,
        hseFuncRnt=hseFuncRnt,
        totExpFuncRnt=totExpFuncRnt,
        ShareFuncRnt=ps.ShareFuncAdj,
        vFuncRnt=ps.vFuncAdj,
        vPfuncRnt=ps.vPfuncAdj,
    )


def housing_to_portfolio(sn):
    return PortfolioSolution(
        cFuncAdj=sn.cFuncRnt,
        ShareFuncAdj=sn.ShareFuncRnt,
        vPfuncAdj=sn.vPfuncRnt,
        vFuncAdj=sn.vFuncRnt,
        AdjPrb=1.0,
    )


def life_cycle_by_years(lc_dict, years):
    lc_ret = lc_dict.copy()
    n = len(lc_dict["LivPrb"]) // years
    lc_ret["Rfree"] = lc_dict["Rfree"] ** years
    lc_ret["DiscFac"] = lc_dict["DiscFac"] ** years

    PermGroFac = []
    for split in np.array_split(lc_dict["PermGroFac"], n):
        PermGroFac.append(np.prod(split))
    lc_ret["PermGroFac"] = PermGroFac

    lc_ret["T_age"] = lc_dict["T_age"] // years + 1
    lc_ret["T_cycle"] = lc_dict["T_cycle"] // years

    PermShkStd = []
    PermShkStd_temp = np.array(lc_dict["PermShkStd"])
    for split in np.array_split(PermShkStd_temp, n):
        PermShkStd.append(np.sqrt((split ** 2).sum()))
    lc_ret["PermShkStd"] = PermShkStd

    return lc_ret


# init_portfolio_housing = life_cycle_by_years(dict_portfolio, 5)
init_portfolio_housing = init_portfolio.copy()

T_cycle = init_portfolio_housing["T_cycle"]
T_retire = init_portfolio_housing["T_retire"]

init_portfolio_housing["LivPrb"] = [1.0] * T_cycle
# Standard deviation of log transitory income shocks
init_portfolio_housing["TranShkStd"] = [0.0] * T_cycle
# Number of points in discrete approximation to transitory income shocks
init_portfolio_housing["TranShkCount"] = 1
# Probability of unemployment while working
init_portfolio_housing["UnempPrb"] = 0.0
# Probability of "unemployment" while retired
init_portfolio_housing["UnempPrbRet"] = 0.0
init_portfolio_housing["ExRiskyShareBool"] = False
init_portfolio_housing["ExRiskyShare"] = [1.0] * 7 + [0.5] * 7
init_portfolio_housing["FixRiskyAvg"] = True
init_portfolio_housing["FixRiskyStd"] = False
init_portfolio_housing["WlthNrmAvg"] = np.linspace(1.0, 20.0, 14)
init_portfolio_housing["RntHseShare"] = 0.3

init_portfolio_risky_housing = init_portfolio.copy()
init_portfolio_risky_housing["LivPrb"] = [1.0] * T_cycle
# Standard deviation of log transitory income shocks
init_portfolio_risky_housing["TranShkStd"] = [0.0] * T_cycle
# Number of points in discrete approximation to transitory income shocks
init_portfolio_risky_housing["TranShkCount"] = 1
# Probability of unemployment while working
init_portfolio_risky_housing["UnempPrb"] = 0.0
# Probability of "unemployment" while retired
init_portfolio_risky_housing["UnempPrbRet"] = 0.0
init_portfolio_risky_housing["RntHseShare"] = 0.3
init_portfolio_risky_housing["HouseAvg"] = 1.0
init_portfolio_risky_housing["HouseStd"] = 0.2
init_portfolio_risky_housing["HouseShkCount"] = 7
init_portfolio_risky_housing["HouseShare"] = 0.3
init_portfolio_risky_housing["HouseMin"] = 1.0
init_portfolio_risky_housing["HouseMax"] = 10.0
init_portfolio_risky_housing["HouseCount"] = 10
init_portfolio_risky_housing["HseInitPrice"] = 1.0
init_portfolio_risky_housing["HseGroFac"] = [1.01] * T_cycle
init_portfolio_risky_housing["HseDiscFac"] = 0.01
init_portfolio_risky_housing["RentPrb"] = list(np.linspace(0.0, 1.0, T_cycle))
# init_portfolio_risky_housing["RentPrb"] = [0.0] * (T_retire - 1) + list(
#     np.linspace(0.0, 1.0, T_cycle - T_retire + 1)
# )
init_portfolio_risky_housing["vFuncBool"] = False
init_portfolio_risky_housing["aXtraMax"] = 1000
init_portfolio_risky_housing["aXtraCount"] = 1000
init_portfolio_risky_housing["aXtraNestFac"] = 1

del init_portfolio_risky_housing["cycles"]


def portfolio_housing_params(
    CRRA=10,
    DiscFac=0.96,
    T_cycle=14,
    T_retire=7,
    PermShkStd=0.1029563,
    PermShkCount=7,
    Rfree=1.02,
    RiskyAvg=1.08,
    RiskyStd=0.157,
    RiskyCount=7,
    HouseAvg=1.0,
    HouseStd=0.2,
    HouseShkCount=7,
    HseGroFac=1.01,
    HseDiscFac=0.01,
    HseInitPrice=1.0,
    repl_fac=0.68212,
):
    params = init_portfolio_risky_housing.copy()
    params["CRRA"] = CRRA
    params["DiscFac"] = DiscFac
    params["T_retire"] = T_retire
    params["T_cycle"] = T_cycle
    params["T_age"] = T_cycle + 1
    params["LivPrb"] = [1.0] * T_cycle
    params["PermGroFac"] = (
        [1.0] * (T_retire - 1) + [repl_fac] + [1.0] * (T_cycle - T_retire)
    )
    params["PermShkStd"] = [PermShkStd] * T_cycle
    params["PermShkCount"] = PermShkCount
    params["TranShkStd"] = [0.0] * T_cycle
    params["TranShkCount"] = 1
    params["HseGroFac"] = [HseGroFac] * T_cycle
    params["HseDiscFac"] = HseDiscFac
    params["HseInitPrice"] = HseInitPrice
    params["Rfree"] = Rfree
    params["RiskyAvg"] = RiskyAvg
    params["RiskyStd"] = RiskyStd
    params["RiskyCount"] = RiskyCount
    params["HouseAvg"] = HouseAvg
    params["HouseStd"] = HouseStd
    params["HouseShkCount"] = HouseShkCount
    params["RentPrb"] = [0.0] * (T_retire - 1) + list(
        np.linspace(0.0, 1.0, T_cycle - T_retire + 1)
    )

    return params

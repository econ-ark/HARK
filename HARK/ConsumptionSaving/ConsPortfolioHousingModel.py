"""
This file contains classes and functions for representing, solving, and simulating agents
who must allocate their resources among consumption, risky or rental housing, saving in a
risk-free asset (with a low return), and saving in a risky asset (with higher average return).
"""
from copy import deepcopy

import numpy as np

from HARK import make_one_period_oo_solver
from HARK.ConsumptionSaving.ConsPortfolioModel import (
    PortfolioConsumerType,
    init_portfolio,
    ConsPortfolioSolver,
)
from HARK.interpolation import (
    LinearInterp,
)


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

    def __init__(self, verbose=False, quiet=False, **kwds):
        params = init_portfolio_rental.copy()
        params.update(kwds)
        kwds = params

        # Initialize a basic consumer type
        PortfolioConsumerType.__init__(self, verbose=verbose, quiet=quiet, **kwds)

        self.solve_one_period = make_one_period_oo_solver(
            ConsPortfolioRentalHousingSolver
        )

        if not hasattr(self, "RntHseShare"):
            raise ValueError(
                "Portfolio Choice with Rental Housing must have a RntHseShare parameter."
            )

    def update_solution_terminal(self):
        """
        Solves the terminal period of the portfolio choice problem.  The solution is
        trivial, as usual: consume all market resources, and put nothing in the risky
        asset (because you have nothing anyway).

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Consume all market resources: c_T = m_T

        PortfolioConsumerType.update_solution_terminal(self)

        mNrm = np.array([0.0, 1.0])
        xFunc_terminal = LinearInterp(mNrm, mNrm)
        cFunc_terminal = LinearInterp(mNrm, (1 - self.RntHseShare) * mNrm)
        hFunc_terminal = LinearInterp(mNrm, self.RntHseShare * mNrm)

        # Construct the terminal period solution
        self.solution_terminal.xFunc = xFunc_terminal
        self.solution_terminal.cFunc = cFunc_terminal
        self.solution_terminal.hFunc = hFunc_terminal
        self.solution_terminal.hNrm = 0.0


class ConsPortfolioRentalHousingSolver(ConsPortfolioSolver):
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
        DiscreteShareBool,
        ShareLimit,
        IndepDstnBool,
        RntHseShare,
    ):
        """
        Constructor for portfolio choice problem solver.
        """

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
        self.DiscreteShareBool = DiscreteShareBool
        self.ShareLimit = ShareLimit
        self.IndepDstnBool = IndepDstnBool
        self.RntHseShare = RntHseShare
        self.AdjustPrb = 1.0

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

        self.def_utility_funcs()

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

    def make_porfolio_solution(self):
        ConsPortfolioSolver.make_porfolio_solution(self)

        xNrm_now = np.insert(self.cNrmAdj_now, 0, 0.0)
        cNrm_now = (1 - self.RntHseShare) * xNrm_now
        hNrm_now = self.RntHseShare * xNrm_now
        mNrm_now = np.insert(self.mNrmAdj_now, 0, 0.0)
        self.solution.xFunc = LinearInterp(mNrm_now, xNrm_now)
        self.solution.cFunc = LinearInterp(mNrm_now, cNrm_now)
        self.solution.hFunc = LinearInterp(mNrm_now, hNrm_now)

        self.solution.hNrm = self.add_human_wealth()


init_portfolio_rental = init_portfolio.copy()
init_portfolio_rental["RntHseShare"] = 0.3

"""
This file has one agent type that solves the portfolio choice problem in a slightly
different way. It imports from legacy OO solver code as well as the portfolio model.
"""

from HARK import make_one_period_oo_solver
from HARK.ConsumptionSaving.ConsPortfolioModel import (
    PortfolioConsumerType,
    init_portfolio,
)
from HARK.ConsumptionSaving.LegacyOOsolvers import ConsSequentialPortfolioSolver


class SequentialPortfolioConsumerType(PortfolioConsumerType):
    def __init__(self, verbose=False, quiet=False, **kwds):
        params = init_portfolio.copy()
        params.update(kwds)
        kwds = params

        # Initialize a basic consumer type
        PortfolioConsumerType.__init__(self, verbose=verbose, quiet=quiet, **kwds)

        # Set the solver for the portfolio model, and update various constructed attributes
        self.solve_one_period = make_one_period_oo_solver(ConsSequentialPortfolioSolver)

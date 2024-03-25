"""
This file has one small AgentType that inherits from IndShockRiskyAssetConsumerType.
These agents have a fixed portfolio share and use a legacy object-oriented solver.
"""

from HARK.ConsumptionSaving.ConsRiskyAssetModel import (
    IndShockRiskyAssetConsumerType,
    init_risky_share_fixed,
)
from HARK import make_one_period_oo_solver
from HARK.ConsumptionSaving.LegacyOOsolvers import (
    ConsFixedPortfolioIndShkRiskyAssetSolver,
)


class FixedPortfolioShareRiskyAssetConsumerType(IndShockRiskyAssetConsumerType):
    time_vary_ = IndShockRiskyAssetConsumerType.time_vary_ + ["RiskyShareFixed"]

    def __init__(self, verbose=False, quiet=False, **kwds):
        params = init_risky_share_fixed.copy()
        params.update(kwds)
        kwds = params

        # Initialize a basic consumer type
        IndShockRiskyAssetConsumerType.__init__(
            self, verbose=verbose, quiet=quiet, **kwds
        )

        self.solve_one_period = make_one_period_oo_solver(
            ConsFixedPortfolioIndShkRiskyAssetSolver
        )

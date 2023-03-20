import numpy as np

from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsIndShockSolver,
    IndShockConsumerType,
    init_lifecycle,
)
from HARK.ConsumptionSaving.ConsPortfolioModel import (
    PortfolioConsumerType,
    init_portfolio,
)
from HARK.core import make_one_period_oo_solver
from HARK.rewards import (
    StoneGearyCRRAutility,
    StoneGearyCRRAutilityP,
    StoneGearyCRRAutilityPP,
    UtilityFuncStoneGeary,
)


class TerminalBequestWarmGlowConsumerType(IndShockConsumerType):
    def __init__(self, **kwds):
        params = init_warm_glow.copy()
        params.update(kwds)

        super().__init__(**params)

        self.pseudo_terminal = True

    def update_solution_terminal(self):
        DiscFacEff = self.BeqRelVal / self.DiscFac
        TranShkMin = np.min(self.TranShkDstn[0].atoms)
        StoneGearyEff = self.BeqStoneGeary - TranShkMin

        warm_glow = UtilityFuncStoneGeary(self.BeqCRRA, DiscFacEff, StoneGearyEff)

        self.solution_terminal.cFunc = lambda m: m - TranShkMin
        self.solution_terminal.vFunc = lambda m: warm_glow(m)
        self.solution_terminal.vPfunc = lambda m: warm_glow.der(m)
        self.solution_terminal.vPPfunc = lambda m: warm_glow.der(m, order=2)
        self.solution_terminal.mNrmMin = np.maximum(TranShkMin, -StoneGearyEff)


class TerminalBequestWarmGlowPortfolioType(
    PortfolioConsumerType, TerminalBequestWarmGlowConsumerType
):
    def __init__(self, **kwds):
        params = init_warm_glow.copy()
        params.update(kwds)

        super().__init__(**params)


class AccidentalBequestWarmGlowConsumerType(TerminalBequestWarmGlowConsumerType):
    def __init_(self, **kwds):
        super().__init__(**kwds)

        self.solve_one_period = make_one_period_oo_solver(
            AccidentalBequestWarmGlowSolver
        )


class AccidentalBequestWarmGlowPortfolioType:
    pass


class AccidentalBequestWarmGlowSolver(ConsIndShockSolver):
    pass


init_warm_glow = init_lifecycle.copy()
init_warm_glow["BeqRelVal"] = 1.0  # Value of bequest relative to consumption
init_warm_glow["BeqStoneGeary"] = 0.0  # Shifts the utility function
init_warm_glow["BeqCRRA"] = init_lifecycle["CRRA"]


init_portfolio_bequest = init_warm_glow.copy()
init_portfolio_bequest.update(init_portfolio)

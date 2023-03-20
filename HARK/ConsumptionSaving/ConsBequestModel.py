import numpy as np

from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    init_lifecycle,
)
from HARK.ConsumptionSaving.ConsPortfolioModel import (
    PortfolioConsumerType,
    init_portfolio,
)
from HARK.rewards import (
    StoneGearyCRRAutility,
    StoneGearyCRRAutilityP,
    StoneGearyCRRAutilityPP,
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

        self.solution_terminal.cFunc = lambda m: m - TranShkMin
        self.solution_terminal.vFunc = lambda m: DiscFacEff * StoneGearyCRRAutility(
            m, self.BeqCRRA, StoneGearyEff
        )
        self.solution_terminal.vPfunc = lambda m: DiscFacEff * StoneGearyCRRAutilityP(
            m, self.BeqCRRA, StoneGearyEff
        )
        self.solution_terminal.vPPfunc = lambda m: DiscFacEff * StoneGearyCRRAutilityPP(
            m, self.BeqCRRA, StoneGearyEff
        )
        self.solution_terminal.mNrmMin = np.maximum(TranShkMin, -StoneGearyEff)


class TerminalBequestWarmGlowPortfolioType(
    PortfolioConsumerType, TerminalBequestWarmGlowConsumerType
):
    def __init__(self, **kwds):
        params = init_warm_glow.copy()
        params.update(kwds)

        super().__init__(**params)


class AccidentalBequestWarmGlowConsumerType:
    pass


class AccidentalBequestWarmGlowPortfolioType:
    pass


init_warm_glow = init_lifecycle.copy()
init_warm_glow["BeqRelVal"] = 1.0  # Value of bequest relative to consumption
init_warm_glow["BeqStoneGeary"] = 0.0  # Shifts the utility function
init_warm_glow["BeqCRRA"] = init_lifecycle["CRRA"]


init_portfolio_bequest = init_warm_glow.copy()
init_portfolio_bequest.update(init_portfolio)

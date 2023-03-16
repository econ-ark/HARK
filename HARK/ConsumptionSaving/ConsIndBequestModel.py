import numpy as np

from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    init_lifecycle,
)
from HARK.rewards import (
    StoneGearyCRRAutility,
    StoneGearyCRRAutilityP,
    StoneGearyCRRAutilityPP,
)


class TerminalBequestConsumerType(IndShockConsumerType):
    def __init__(self, **kwds):
        params = init_terminal_bequest.copy()
        params.update(kwds)

        super().__init__(**params)

    def update_solution_terminal(self):
        DiscFacEff = self.BeqDiscFac / self.DiscFac
        ShifterEff = self.BeqShifter - 1.0

        self.solution_terminal.cFunc = lambda m: m
        self.solution_terminal.vFunc = lambda m: DiscFacEff * StoneGearyCRRAutility(
            m, self.BeqCRRA, ShifterEff
        )
        self.solution_terminal.vPfunc = lambda m: DiscFacEff * StoneGearyCRRAutilityP(
            m, self.BeqCRRA, ShifterEff
        )
        self.solution_terminal.vPPfunc = lambda m: DiscFacEff * StoneGearyCRRAutilityPP(
            m, self.BeqCRRA, ShifterEff
        )
        self.solution_terminal.mNrmMin = np.maximum(1.0, -ShifterEff)


init_terminal_bequest = init_lifecycle.copy()
init_terminal_bequest["BeqDiscFac"] = 1.0
init_terminal_bequest["BeqShifter"] = 0.0
init_terminal_bequest["BeqCRRA"] = init_lifecycle["CRRA"]

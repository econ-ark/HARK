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
        self.solution_terminal.cFunc = lambda m: m
        self.solution_terminal.vFunc = (
            lambda m: self.BeqDiscFac
            * StoneGearyCRRAutility(m, self.BeqCRRA, self.BeqShifter)
        )
        self.solution_terminal.vPfunc = (
            lambda m: self.BeqDiscFac
            * StoneGearyCRRAutilityP(m, self.BeqCRRA, self.BeqShifter)
        )
        self.solution_terminal.vPPfunc = (
            lambda m: self.BeqDiscFac
            * StoneGearyCRRAutilityPP(m, self.BeqCRRA, self.BeqShifter)
        )


init_terminal_bequest = init_lifecycle.copy()
init_terminal_bequest["BeqDiscFac"] = 1.0
init_terminal_bequest["BeqShifter"] = 0.0
init_terminal_bequest["BeqCRRA"] = init_terminal_bequest["CRRA"]

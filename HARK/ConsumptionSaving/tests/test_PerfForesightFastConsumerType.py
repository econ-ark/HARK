from HARK.ConsumptionSaving.ConsIndShockModel import PerfForesightConsumerType
from HARK.ConsumptionSaving.ConsIndShockModelFast import PerfForesightConsumerTypeFast
from HARK.ConsumptionSaving.tests.test_PerfForesightConsumerType import (
    testPerfForesightConsumerType,
)


class testPerfForesightFastConsumerType(testPerfForesightConsumerType):
    def setUp(self):
        self.agent = PerfForesightConsumerTypeFast()
        self.agent_slow = PerfForesightConsumerType()
        self.agent_infinite = PerfForesightConsumerTypeFast(cycles=0)
        self.agent_infinite_slow = PerfForesightConsumerType(cycles=0)

        PF_dictionary = {
            "CRRA": 2.5,
            "DiscFac": 0.96,
            "Rfree": 1.03,
            "LivPrb": [0.98],
            "PermGroFac": [1.01],
            "T_cycle": 1,
            "cycles": 0,
            "AgentCount": 10000,
        }
        self.agent_alt = PerfForesightConsumerTypeFast(**PF_dictionary)

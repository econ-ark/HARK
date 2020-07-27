from HARK.ConsumptionSaving.ConsIndShockFastModel import PerfForesightFastConsumerType
from HARK.ConsumptionSaving.ConsIndShockModel import PerfForesightConsumerType
from HARK.ConsumptionSaving.tests.test_PerfForesightConsumerType import (
    testPerfForesightConsumerType,
)


class testPerfForesightFastConsumerType(testPerfForesightConsumerType):
    def setUp(self):
        self.agent = PerfForesightFastConsumerType()
        self.agent_slow = PerfForesightConsumerType()
        self.agent_infinite = PerfForesightFastConsumerType(cycles=0)
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
        self.agent_alt = PerfForesightFastConsumerType(**PF_dictionary)

    def test_equality(self):
        self.assertTrue(self.agent == self.agent_slow)
        self.assertFalse(self.agent == self.agent_infinite)
        self.assertFalse(self.agent == self.agent_alt)
        self.assertTrue(self.agent_infinite == self.agent_infinite_slow)
        self.assertFalse(self.agent_infinite == self.agent_alt)

        self.agent.solve()
        self.agent_slow.solve()
        self.agent_infinite.solve()
        self.agent_infinite_slow.solve()
        self.agent_alt.solve()

        self.assertTrue(self.agent.solution == self.agent_slow.solution)
        self.assertFalse(self.agent.solution == self.agent_infinite.solution)
        self.assertFalse(self.agent.solution == self.agent_alt.solution)
        self.assertTrue(
            self.agent_infinite.solution == self.agent_infinite_slow.solution
        )
        self.assertFalse(self.agent_infinite.solution == self.agent_alt.solution)

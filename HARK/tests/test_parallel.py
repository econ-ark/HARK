import unittest
from HARK.ConsumptionSaving.ConsIndShockModel import PerfForesightConsumerType
from HARK.parallel import multiThreadCommandsFake, multiThreadCommands, runCommands


class testParallel(unittest.TestCase):
    def setUp(self):
        self.agent = PerfForesightConsumerType()
        self.agents = 5 * [self.agent]

    def test_multiThreadCommandsFake(self):
        # check None return if it passes
        self.assertIsNone(multiThreadCommandsFake(self.agents, ["solve()"]))
        # check if an undefined method of agent is called
        self.assertRaises(
            AttributeError, multiThreadCommandsFake, self.agents, ["foobar"]
        )

    def test_multiThreadCommands(self):
        # check None return if it passes
        self.assertIsNone(multiThreadCommands(self.agents, ["solve()"]))
        # check if an undefined method of agent is called
        self.assertRaises(
            AttributeError, multiThreadCommandsFake, self.agents, ["foobar"]
        )

    def test_runCommands(self):
        self.assertEquals(runCommands(self.agent, ["solve()"]), self.agent)
        # check if an undefined method of agent is called
        self.assertRaises(AttributeError, runCommands, self.agent, ["foobar()"])

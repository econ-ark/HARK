import unittest
from HARK.ConsumptionSaving.ConsIndShockModel import PerfForesightConsumerType
from HARK.parallel import multi_thread_commands_fake, multi_thread_commands, run_commands


class testParallel(unittest.TestCase):
    def setUp(self):
        self.agent = PerfForesightConsumerType()
        self.agents = 5 * [self.agent]

    def test_multi_thread_commands_fake(self):
        # check None return if it passes
        self.assertIsNone(multi_thread_commands_fake(self.agents, ["solve()"]))
        # check if an undefined method of agent is called
        self.assertRaises(
            AttributeError, multi_thread_commands_fake, self.agents, ["foobar"]
        )

    def test_multi_thread_commands(self):
        # check None return if it passes
        self.assertIsNone(multi_thread_commands(self.agents, ["solve()"]))
        # check if an undefined method of agent is called
        self.assertRaises(
            AttributeError, multi_thread_commands_fake, self.agents, ["foobar"]
        )

    def test_run_commands(self):
        self.assertEquals(run_commands(self.agent, ["solve()"]), self.agent)
        # check if an undefined method of agent is called
        self.assertRaises(AttributeError, run_commands, self.agent, ["foobar()"])

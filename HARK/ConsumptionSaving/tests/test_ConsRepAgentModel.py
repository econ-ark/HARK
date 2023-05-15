import unittest

from HARK.ConsumptionSaving.ConsRepAgentModel import (
    RepAgentConsumerType,
    RepAgentMarkovConsumerType,
)
from HARK.tests import HARK_PRECISION


class testRepAgentConsumerType(unittest.TestCase):
    def setUp(self):
        self.agent = RepAgentConsumerType()
        self.agent.solve()

    def test_solution(self):
        self.assertAlmostEqual(
            self.agent.solution[0].cFunc(10).tolist(), 1.71306, places=HARK_PRECISION
        )

    def test_simulation(self):
        # Simulate the representative agent model
        self.agent.T_sim = 100
        self.agent.track_vars = ["cNrm", "mNrm", "Rfree", "wRte"]
        self.agent.initialize_sim()
        self.agent.simulate()


class testRepAgentMarkovConsumerType(unittest.TestCase):
    def setUp(self):
        self.agent = RepAgentMarkovConsumerType()
        self.agent.IncShkDstn = [2 * [self.agent.IncShkDstn[0]]]
        self.agent.solve()

    def test_solution(self):
        self.assertAlmostEqual(
            self.agent.solution[0].cFunc[0](10).tolist(), 1.38295, places=HARK_PRECISION
        )

    def test_simulation(self):
        # Simulate the representative agent model
        self.agent.T_sim = 100
        self.agent.track_vars = ["cNrm", "mNrm", "Rfree", "wRte", "Mrkv"]
        self.agent.initialize_sim()
        self.agent.simulate()

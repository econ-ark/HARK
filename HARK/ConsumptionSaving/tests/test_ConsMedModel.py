import unittest

from HARK.ConsumptionSaving.ConsMedModel import MedShockConsumerType


class testMedShockConsumerType(unittest.TestCase):
    def setUp(self):
        self.agent = MedShockConsumerType()

    def test_solution(self):
        self.agent.solve()

        self.agent.T_sim = 10
        self.agent.track_vars = ["mLvl", "cLvl", "Med"]
        self.agent.make_shock_history()
        self.agent.initialize_sim()
        self.agent.simulate()

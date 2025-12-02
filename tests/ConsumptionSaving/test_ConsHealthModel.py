import unittest
from tests import HARK_PRECISION
from HARK.ConsumptionSaving.ConsHealthModel import BasicHealthConsumerType


class testBasicHealthConsumerType(unittest.TestCase):
    def setUp(self):
        self.agent = BasicHealthConsumerType(cycles=10, T_age=10)
        self.agent.solve()

    def test_solution(self):
        policy_func = self.agent.solution[0]
        mLvl = 10.0
        hLvl = 20.0
        vNvrs, cLvl, nLvl = policy_func(mLvl, hLvl)
        self.assertAlmostEqual(vNvrs, 209.3933, places=HARK_PRECISION)
        self.assertAlmostEqual(cLvl, 4.1810, places=HARK_PRECISION)
        self.assertAlmostEqual(nLvl, 0.7857, places=HARK_PRECISION)

    def test_simulation(self):
        self.agent.T_sim = 10
        self.agent.track_vars = ["mLvl", "cLvl", "hLvl", "nLvl"]
        self.agent.make_shock_history()
        self.agent.initialize_sim()
        self.agent.simulate()

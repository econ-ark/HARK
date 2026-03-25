import unittest
from tests import HARK_PRECISION
from HARK.ConsumptionSaving.ConsHabitPortfolioModel import HabitPortfolioConsumerType


class testHabitPortfolioConsumerType(unittest.TestCase):
    def setUp(self):
        self.agent = HabitPortfolioConsumerType(cycles=10)
        self.agent.solve()

    def test_cFunc(self):
        cFunc = self.agent.solution[0]["cFunc"]
        self.assertAlmostEqual(cFunc(10.0, 1.0), 1.76346, places=HARK_PRECISION)

    def test_ShareFunc(self):
        ShareFunc = self.agent.solution[0]["ShareFunc"]
        self.assertAlmostEqual(ShareFunc(10.0, 1.0), 1.0, places=HARK_PRECISION)

    def test_terminal_period(self):
        sol_last = self.agent.solution[-1]
        self.assertAlmostEqual(sol_last["cFunc"](5.0, 1.0), 5.0, places=HARK_PRECISION)

    def test_simulation(self):
        self.agent.T_sim = 10
        self.agent.initialize_sym()
        self.agent.symulate()

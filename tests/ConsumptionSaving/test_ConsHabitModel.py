import unittest
from tests import HARK_PRECISION
from HARK.ConsumptionSaving.ConsHabitModel import (
    HabitConsumerType,
    HabitPortfolioConsumerType,
)


class testHabitConsumerType(unittest.TestCase):
    def setUp(self):
        self.agent = HabitConsumerType(cycles=10)
        self.agent.solve()

    def test_solution(self):
        cFunc = self.agent.solution[0]["cFunc"]
        mNrm = 10.0
        hNrm = 1.0
        self.assertAlmostEqual(cFunc(mNrm, hNrm), 1.72773, places=HARK_PRECISION)

    def test_simulation(self):
        self.agent.T_sim = 10
        self.agent.initialize_sym()
        self.agent.symulate()

    def test_invalid(self):
        self.assertRaises(ValueError, HabitConsumerType, HabitRte=1.2)
        self.assertRaises(ValueError, HabitConsumerType, HabitRte=0.0)
        self.assertRaises(ValueError, HabitConsumerType, HabitWgt=1.2)
        self.assertRaises(ValueError, HabitConsumerType, HabitWgt=0.0)


class testHabitPortfolioConsumerType(unittest.TestCase):
    def setUp(self):
        self.agent = HabitPortfolioConsumerType(CRRA=5.0, cycles=10)
        self.agent.solve()

    def test_cFunc(self):
        cFunc = self.agent.solution[0]["cFunc"]
        self.assertAlmostEqual(cFunc(10.0, 1.0), 1.70776, places=HARK_PRECISION)

    def test_ShareFunc(self):
        ShareFunc = self.agent.solution[0]["ShareFunc"]
        self.assertAlmostEqual(ShareFunc(10.0, 1.0), 0.71368, places=HARK_PRECISION)

    def test_terminal_period(self):
        sol_last = self.agent.solution[-1]
        self.assertAlmostEqual(sol_last["cFunc"](5.0, 1.0), 5.0, places=HARK_PRECISION)

    def test_simulation(self):
        self.agent.T_sim = 10
        self.agent.initialize_sym()
        self.agent.symulate()

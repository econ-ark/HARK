import unittest
from tests import HARK_PRECISION
from HARK.ConsumptionSaving.ConsHabitModel import HabitConsumerType


class testHabitConsumerType(unittest.TestCase):
    def setUp(self):
        self.agent = HabitConsumerType(cycles=10)
        self.agent.solve()

    def test_solution(self):
        cFunc = self.agent.solution[0]["cFunc"]
        mNrm = 10.0
        hNrm = 1.0
        self.assertAlmostEqual(cFunc(mNrm, hNrm), 1.7289, places=HARK_PRECISION)

    def test_simulation(self):
        self.agent.T_sim = 10
        self.agent.initialize_sym()
        self.agent.symulate()

    def test_invalid(self):
        self.assertRaises(ValueError, HabitConsumerType, HabitRte=1.2)
        self.assertRaises(ValueError, HabitConsumerType, HabitRte=0.0)
        self.assertRaises(ValueError, HabitConsumerType, HabitWgt=1.2)
        self.assertRaises(ValueError, HabitConsumerType, HabitWgt=0.0)

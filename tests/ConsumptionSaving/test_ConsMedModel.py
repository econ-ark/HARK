import unittest
from tests import HARK_PRECISION
from HARK.ConsumptionSaving.ConsMedModel import MedShockConsumerType


class testMedShockConsumerType(unittest.TestCase):
    def setUp(self):
        self.agent = MedShockConsumerType()
        self.agent.vFuncBool = True
        self.agent.solve()

    def test_solution(self):
        cFunc = self.agent.solution[0].cFunc
        MedFunc = self.agent.solution[0].MedFunc
        mLvl = 10.0
        pLvl = 2.0
        Shk = 1.5
        self.assertAlmostEqual(
            cFunc(mLvl, pLvl, Shk).tolist(), 4.0056, places=HARK_PRECISION
        )
        self.assertAlmostEqual(
            MedFunc(mLvl, pLvl, Shk).tolist(), 2.40487, places=HARK_PRECISION
        )

    def test_value(self):
        vFunc = self.agent.solution[0].vFunc
        mLvl = 10.0
        pLvl = 2.0
        self.assertAlmostEqual(vFunc(mLvl, pLvl), -0.36032, places=HARK_PRECISION)

    def test_simulation(self):
        self.agent.T_sim = 10
        self.agent.track_vars = ["mLvl", "cLvl", "Med"]
        self.agent.make_shock_history()
        self.agent.initialize_sim()
        self.agent.simulate()

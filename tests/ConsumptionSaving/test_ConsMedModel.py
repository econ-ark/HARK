import unittest
from tests import HARK_PRECISION
from HARK.ConsumptionSaving.ConsMedModel import (
    MedShockConsumerType,
    MedExtMargConsumerType,
)


class testMedShockConsumerType(unittest.TestCase):
    def setUp(self):
        self.agent = MedShockConsumerType(vFuncBool=True)
        self.agent.solve()

    def test_solution(self):
        cFunc = self.agent.solution[0]["PolicyFunc"].cFunc
        MedFunc = self.agent.solution[0]["PolicyFunc"].MedFunc
        mLvl = 10.0
        pLvl = 2.0
        Shk = 1.5
        self.assertAlmostEqual(
            cFunc(mLvl, pLvl, Shk).tolist(), 3.5044, places=HARK_PRECISION
        )
        self.assertAlmostEqual(
            MedFunc(mLvl, pLvl, Shk).tolist(), 2.10620, places=HARK_PRECISION
        )

    def test_unpack(self):
        # This test is relevant because solution representation is a dictionary
        self.agent.unpack("vFunc")

    def test_value(self):
        vFunc = self.agent.solution[0]["vFunc"]
        mLvl = 10.0
        pLvl = 2.0
        self.assertAlmostEqual(vFunc(mLvl, pLvl), -0.38395, places=HARK_PRECISION)

    def test_simulation(self):
        self.agent.T_sim = 10
        self.agent.track_vars = ["mLvl", "cLvl", "MedLvl"]
        self.agent.make_shock_history()
        self.agent.initialize_sim()
        self.agent.simulate()

    def test_cubic(self):
        CubicType = MedShockConsumerType(CubicBool=True)
        self.assertRaises(NotImplementedError, CubicType.solve)


class testMedExtMargConsumerType(unittest.TestCase):
    def setUp(self):
        self.agent = MedExtMargConsumerType()
        self.agent.solve()

    def test_solution(self):
        cFunc = self.agent.solution[0].cFunc
        MedFunc = self.agent.solution[0].ExpMedFunc
        mLvl = 10.0
        pLvl = 2.0
        self.assertAlmostEqual(cFunc(mLvl, pLvl).tolist(), 10.0, places=HARK_PRECISION)
        self.assertAlmostEqual(
            MedFunc(mLvl, pLvl).tolist(), 0.52176, places=HARK_PRECISION
        )

    def test_value(self):
        # Use middle index to avoid hardcoded assumptions about grid size
        pLvl_idx = len(self.agent.solution[0].vFunc_by_pLvl) // 2
        vFunc = self.agent.solution[0].vFunc_by_pLvl[pLvl_idx]
        mLvl = 10.0
        self.assertAlmostEqual(vFunc(mLvl), -1.23397, places=HARK_PRECISION)

    def test_simulation(self):
        self.agent.T_sim = 10
        self.agent.track_vars = ["mLvl", "cLvl", "Med"]
        self.agent.make_shock_history()
        self.agent.initialize_sim()
        self.agent.simulate()

    def test_IH_constructors(self):
        self.agent.cycles = 0
        self.agent.construct()

    def test_describe_constructors(self):
        self.agent.describe_constructors()

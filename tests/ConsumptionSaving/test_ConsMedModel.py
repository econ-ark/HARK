import unittest
from tests import HARK_PRECISION
from HARK.ConsumptionSaving.ConsMedModel import (
    MedShockConsumerType,
    MedExtMargConsumerType,
)


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

    def test_cubic(self):
        CubicType = MedShockConsumerType(CubicBool=True)
        CubicType.solve()
        cFunc = CubicType.solution[0].cFunc
        MedFunc = CubicType.solution[0].MedFunc
        mLvl = 10.0
        pLvl = 2.0
        Shk = 1.5
        self.assertAlmostEqual(
            cFunc(mLvl, pLvl, Shk).tolist(), 4.00158, places=HARK_PRECISION
        )
        self.assertAlmostEqual(
            MedFunc(mLvl, pLvl, Shk).tolist(), 2.4088, places=HARK_PRECISION
        )

    def test_derivatives(self):
        policyFunc = self.agent.solution[0].policyFunc
        cFunc = self.agent.solution[0].cFunc
        MedFunc = self.agent.solution[0].MedFunc
        mLvl = 10.0
        pLvl = 2.0
        Shk = 0.5
        query = (mLvl, pLvl, Shk)
        eps = 1e-9
        cLvl, Med = policyFunc(*query)

        c_alt, Med_alt = policyFunc(mLvl + eps, pLvl, Shk)
        dcdm_targ = (c_alt - cLvl) / eps
        dMeddm_targ = (Med_alt - Med) / eps
        dcdm, dMeddm = policyFunc.derivativeX(*query)
        self.assertAlmostEqual(dcdm, dcdm_targ, places=HARK_PRECISION)
        self.assertAlmostEqual(dMeddm, dMeddm_targ, places=HARK_PRECISION)

        c_alt, Med_alt = policyFunc(mLvl, pLvl + eps, Shk)
        dcdp_targ = (c_alt - cLvl) / eps
        dMeddp_targ = (Med_alt - Med) / eps
        dcdp, dMeddp = policyFunc.derivativeY(*query)
        # self.assertAlmostEqual(dcdp, dcdp_targ, delta=1e-2)
        # self.assertAlmostEqual(dMeddp, dMeddp_targ, delta=1e-2)

        c_alt, Med_alt = policyFunc(mLvl, pLvl, Shk + eps)
        dcdShk_targ = (c_alt - cLvl) / eps
        dMeddShk_targ = (Med_alt - Med) / eps
        dcdShk, dMeddShk = policyFunc.derivativeZ(*query)
        # self.assertAlmostEqual(dcdShk, dcdShk_targ, delta=1e-2)
        # self.assertAlmostEqual(dMeddShk, dMeddShk_targ, delta=1e-2)

        c_alt = cFunc(mLvl + eps, pLvl, Shk)
        dcdm_targ = (c_alt - cLvl) / eps
        dcdm_a = cFunc.derivativeX(*query)
        self.assertAlmostEqual(dcdm_a, dcdm_targ, places=HARK_PRECISION)
        self.assertAlmostEqual(dcdm_a, dcdm)

        c_alt = cFunc(mLvl, pLvl + eps, Shk)
        dcdp_targ = (c_alt - cLvl) / eps
        dcdp_a = cFunc.derivativeY(*query)
        self.assertAlmostEqual(dcdp_a, dcdp)

        c_alt = cFunc(mLvl, pLvl, Shk + eps)
        dcdShk_targ = (c_alt - cLvl) / eps
        dcdShk_a = cFunc.derivativeZ(*query)
        self.assertAlmostEqual(dcdShk_a, dcdShk)

        Med_alt = MedFunc(mLvl + eps, pLvl, Shk)
        dMeddm_targ = (Med_alt - Med) / eps
        dMeddm_a = MedFunc.derivativeX(*query)
        self.assertAlmostEqual(dMeddm_a, dMeddm_targ, places=HARK_PRECISION)
        self.assertAlmostEqual(dMeddm_a, dMeddm)

        Med_alt = MedFunc(mLvl, pLvl + eps, Shk)
        dMeddp_targ = (Med_alt - Med) / eps
        dMeddp_a = MedFunc.derivativeY(*query)
        self.assertAlmostEqual(dMeddp_a, dMeddp)

        Med_alt = MedFunc(mLvl, pLvl, Shk + eps)
        dMeddShk_targ = (Med_alt - Med) / eps
        dMeddShk_a = MedFunc.derivativeZ(*query)
        self.assertAlmostEqual(dMeddShk_a, dMeddShk)


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
        self.agent.track_vars = ["mLvl", "cLvl", "MedLvl"]
        self.agent.make_shock_history()
        self.agent.initialize_sim()
        self.agent.simulate()

    def test_IH_constructors(self):
        self.agent.cycles = 0
        self.agent.construct()

    def test_describe_constructors(self):
        self.agent.describe_constructors()

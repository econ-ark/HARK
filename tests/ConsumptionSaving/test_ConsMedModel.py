import unittest

import numpy as np

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

    def test_aNrm_is_written_every_period(self):
        # This class overrides get_poststates, so it does not inherit the
        # GenIncProcess line that defines aNrm; it has to call
        # set_aNrm_from_levels itself. Without that every continuing agent's
        # aNrm goes unwritten -- 1600 of 1600 cells on this fixture.
        self.agent.T_sim = 8
        self.agent.AgentCount = 200
        self.agent.track_vars = ["aNrm", "aLvl", "pLvl"]
        self.agent.make_shock_history()
        self.agent.initialize_sim()
        self.agent.simulate()

        aNrm = self.agent.history["aNrm"]
        implied = self.agent.history["aLvl"] / self.agent.history["pLvl"]
        self.assertEqual(aNrm.size, 1600)
        self.assertTrue(np.all(np.isfinite(aNrm)))
        self.assertTrue(np.allclose(aNrm, implied, rtol=1e-12, atol=0.0))

    def test_state_vars_has_no_duplicates(self):
        # state_vars used to append "mLvl", which the parent list already
        # carried, making it longer than the set of states it names.
        self.assertEqual(len(self.agent.state_vars), len(set(self.agent.state_vars)))

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

    def test_aNrm_is_written_every_period(self):
        # Second get_poststates override in this file; same requirement.
        self.agent.T_sim = 8
        self.agent.AgentCount = 200
        self.agent.track_vars = ["aNrm", "aLvl", "pLvl"]
        self.agent.make_shock_history()
        self.agent.initialize_sim()
        self.agent.simulate()

        aNrm = self.agent.history["aNrm"]
        implied = self.agent.history["aLvl"] / self.agent.history["pLvl"]
        self.assertEqual(aNrm.size, 1600)
        self.assertTrue(np.all(np.isfinite(aNrm)))
        self.assertTrue(np.allclose(aNrm, implied, rtol=1e-12, atol=0.0))

    def test_IH_constructors(self):
        self.agent.cycles = 0
        self.agent.construct()

    def test_describe_constructors(self):
        self.agent.describe_constructors()

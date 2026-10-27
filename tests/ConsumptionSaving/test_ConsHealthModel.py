import unittest

import numpy as np

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

    def test_solution_holds_no_nans(self):
        # Ending a period with no assets and no health made the health
        # investment at that corner nan, and the zero-health column with it
        # (issue #1864).
        for solution in self.agent.solution:
            for values in (solution.x_values, solution.y_values, solution.f_values):
                self.assertTrue(np.all(np.isfinite(np.asarray(values))))

    def test_policy_is_finite_near_zero_health(self):
        policy_func = self.agent.solution[0]
        mLvl = np.array([0.001, 0.01, 0.05, 0.2, 0.01])
        hLvl = np.array([0.0, 0.0, 0.0, 0.0, 0.5])
        vNvrs, cLvl, nLvl = policy_func(mLvl, hLvl)
        self.assertTrue(np.all(cLvl > 0.0))
        self.assertTrue(np.all(nLvl > 0.0))
        # these agents are constrained: they spend all their resources
        np.testing.assert_allclose(cLvl + nLvl, mLvl, rtol=1e-12)

    def test_infinite_horizon_converges(self):
        # With nans in the solution the distance between iterates was nan, and
        # the solve stopped after two cycles.
        agent = BasicHealthConsumerType(cycles=0, tolerance=1e-4)
        agent.solve()
        self.assertGreater(agent.completed_cycles, 2)
        self.assertTrue(np.isfinite(agent.solution_distance))
        self.assertLessEqual(agent.solution_distance, 1e-4)

    def test_simulation(self):
        self.agent.T_sim = 10
        self.agent.track_vars = ["mLvl", "cLvl", "hLvl", "nLvl"]
        self.agent.make_shock_history()
        self.agent.initialize_sim()
        self.agent.simulate()

    def test_invalid(self):
        ThisType = BasicHealthConsumerType()
        ThisType.DeprRteMean = [0.05, 0.06]
        self.assertRaises(ValueError, ThisType.construct)
        ThisType.DeprRteMean = [0.05]
        ThisType.DeprRteSpread = [0.05, 0.06]
        self.assertRaises(ValueError, ThisType.construct)

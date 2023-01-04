import unittest

import numpy as np

from HARK.ConsumptionSaving.ConsPrefShockModel import (
    KinkyPrefConsumerType,
    PrefShockConsumerType,
)
from HARK.tests import HARK_PRECISION


class testPrefShockConsumerType(unittest.TestCase):
    def setUp(self):

        self.agent = PrefShockConsumerType()
        self.agent.cycles = 0
        self.agent.solve()

    def test_solution(self):

        self.assertEqual(self.agent.solution[0].mNrmMin, 0)
        m = np.linspace(self.agent.solution[0].mNrmMin, 5, 200)

        self.assertAlmostEqual(
            self.agent.PrefShkDstn[0].atoms[0, 5], 0.69047, places=HARK_PRECISION
        )

        self.assertAlmostEqual(
            self.agent.solution[0].cFunc(m, np.ones_like(m))[35],
            0.81239,
            places=HARK_PRECISION,
        )

        self.assertAlmostEqual(
            self.agent.solution[0].cFunc.derivativeX(m, np.ones_like(m))[35],
            0.44974,
            places=HARK_PRECISION,
        )

    def test_simulation(self):
        self.agent.T_sim = 10
        self.agent.track_vars = ["cNrm", "PrefShk"]
        self.agent.make_shock_history()  # This is optional
        self.agent.initialize_sim()
        self.agent.simulate()

        # simulation test -- seed/generator specific
        # self.assertAlmostEqual(self.agent.history["cNrm"][0][5], 0.73660, place = HARK_PRECISION)

        self.assertEqual(
            self.agent.shock_history["PrefShk"][0][5],
            self.agent.history["PrefShk"][0][5],
        )

        # simulation test -- seed/generator specific
        # self.assertAlmostEqual(self.agent.history["PrefShk"][0][5], 0.49094, place = HARK_PRECISION)


class testKinkyPrefConsumerType(unittest.TestCase):
    def setUp(self):

        self.agent = KinkyPrefConsumerType()
        self.agent.cycles = 0  # Infinite horizon
        self.agent.solve()

    def test_solution(self):
        self.assertAlmostEqual(
            self.agent.solution[0].mNrmMin, -0.75552, places=HARK_PRECISION
        )

        m = np.linspace(self.agent.solution[0].mNrmMin, 5, 200)

        self.assertAlmostEqual(
            self.agent.PrefShkDstn[0].atoms[0, 5], 0.69047, places=HARK_PRECISION
        )

        c = self.agent.solution[0].cFunc(m, np.ones_like(m))
        self.assertAlmostEqual(c[5], 0.13238, places=HARK_PRECISION)

        k = self.agent.solution[0].cFunc.derivativeX(m, np.ones_like(m))
        self.assertAlmostEqual(k[5], 0.91443, places=HARK_PRECISION)

        self.agent.solution[0].vFunc
        self.agent.solution[0].mNrmMin

    def test_simulation(self):
        self.agent.T_sim = 10
        self.agent.track_vars = ["cNrm", "PrefShk"]
        self.agent.initialize_sim()
        self.agent.simulate()

        # simulation test -- seed/generator specific
        # self.assertAlmostEqual(self.agent.history["cNrm"][0][5], 0.77171, place = HARK_PRECISION)

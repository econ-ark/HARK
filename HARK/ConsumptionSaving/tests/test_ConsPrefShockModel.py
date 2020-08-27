from HARK.ConsumptionSaving.ConsPrefShockModel import (
    PrefShockConsumerType,
    KinkyPrefConsumerType,
)
import numpy as np
import unittest


class testPrefShockConsumerType(unittest.TestCase):
    def setUp(self):

        self.agent = PrefShockConsumerType()
        self.agent.cycles = 0
        self.agent.solve()

    def test_solution(self):

        self.assertEqual(self.agent.solution[0].mNrmMin, 0)
        m = np.linspace(self.agent.solution[0].mNrmMin, 5, 200)

        self.assertAlmostEqual(self.agent.PrefShkDstn[0].X[5], 0.69046812)

        self.assertAlmostEqual(
            self.agent.solution[0].cFunc(m, np.ones_like(m))[35], 0.8123891603954809
        )

        self.assertAlmostEqual(
            self.agent.solution[0].cFunc.derivativeX(m, np.ones_like(m))[35],
            0.44973706445183886,
        )

    def test_simulation(self):
        self.agent.T_sim = 10
        self.agent.track_vars = ["cNrmNow", "PrefShkNow"]
        self.agent.makeShockHistory()  # This is optional
        self.agent.initializeSim()
        self.agent.simulate()

        self.assertAlmostEqual(
            self.agent.history['cNrmNow'][0][5],
            0.7366020536567589
        )

        self.assertEqual(
            self.agent.shock_history['PrefShkNow'][0][5],
            self.agent.history['PrefShkNow'][0][5]
        )

        self.assertEqual(
            self.agent.history['PrefShkNow'][0][5],
            0.4909415933881665
        )

class testKinkyPrefConsumerType(unittest.TestCase):
    def setUp(self):

        self.agent = KinkyPrefConsumerType()
        self.agent.cycles = 0  # Infinite horizon
        self.agent.solve()

    def test_solution(self):
        self.assertAlmostEqual(self.agent.solution[0].mNrmMin, -0.7555156106287383)

        m = np.linspace(self.agent.solution[0].mNrmMin, 5, 200)

        self.assertAlmostEqual(self.agent.PrefShkDstn[0].X[5], 0.6904681186891202)

        c = self.agent.solution[0].cFunc(m, np.ones_like(m))
        self.assertAlmostEqual(c[5], 0.13237946)

        k = self.agent.solution[0].cFunc.derivativeX(m, np.ones_like(m))
        self.assertAlmostEqual(k[5], 0.91443463)

        self.agent.solution[0].vFunc
        self.agent.solution[0].mNrmMin

    def test_simulation(self):
        self.agent.T_sim = 10
        self.agent.track_vars = ["cNrmNow", "PrefShkNow"]
        self.agent.initializeSim()
        self.agent.simulate()

        self.assertAlmostEqual(
            self.agent.history['cNrmNow'][0][5],
            0.7717096928111515
        )

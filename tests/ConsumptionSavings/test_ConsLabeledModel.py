import unittest

from HARK.ConsumptionSaving.ConsIndShockModel import init_lifecycle
from HARK.ConsumptionSaving.ConsLabeledModel import (
    IndShockLabeledType,
    PerfForesightLabeledType,
    PortfolioLabeledType,
)
from HARK.tests import HARK_PRECISION


class test_PerfForesightLabeledType(unittest.TestCase):
    def setUp(self):
        self.agent = PerfForesightLabeledType()

    def test_default_solution(self):
        self.agent.solve()
        c = self.agent.solution[0].policy["cNrm"].to_numpy()
        m = self.agent.solution[0].policy["mNrm"].to_numpy()

        self.assertAlmostEqual(m[0], -0.97087, places=HARK_PRECISION)
        self.assertAlmostEqual(m[1], -0.95866, places=HARK_PRECISION)
        self.assertEqual(c[0], 0)
        self.assertAlmostEqual(c[1], 0.01120, places=HARK_PRECISION)


class test_IndShockConsumerType(unittest.TestCase):
    def setUp(self):
        LifecycleExample = IndShockLabeledType(**init_lifecycle)
        LifecycleExample.cycles = 1
        LifecycleExample.solve()

        self.agent = LifecycleExample

    def test_IndShockLabeledType(self):
        solution = [
            self.agent.solution[i].policy["cNrm"].interp({"mNrm": 1}).to_numpy()
            for i in range(10)
        ]

        self.assertAlmostEqual(solution[9], 0.79454, places=HARK_PRECISION)
        self.assertAlmostEqual(solution[8], 0.79414, places=HARK_PRECISION)
        self.assertAlmostEqual(solution[7], 0.79274, places=HARK_PRECISION)
        self.assertAlmostEqual(solution[0], 0.75088, places=HARK_PRECISION)
        self.assertAlmostEqual(solution[1], 0.75891, places=HARK_PRECISION)
        self.assertAlmostEqual(solution[2], 0.76845, places=HARK_PRECISION)


class test_PortfolioLabeledType(unittest.TestCase):
    def setUp(self):
        # Create portfolio choice consumer type
        self.agent = PortfolioLabeledType()
        self.agent.cycles = 0

        # Solve the model under the given parameters

        self.agent.solve()

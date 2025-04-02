import unittest

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
        self.agent = IndShockLabeledType(cycles=10)

    def test_IndShockLabeledType(self):
        self.agent.solve()
        c = self.agent.solution[0].policy["cNrm"].to_numpy()
        m = self.agent.solution[0].policy["mNrm"].to_numpy()

        self.assertAlmostEqual(c[4], 0.47038, places=HARK_PRECISION)
        self.assertAlmostEqual(m[4], -0.72898, places=HARK_PRECISION)


class test_PortfolioLabeledType(unittest.TestCase):
    def setUp(self):
        # Create portfolio choice consumer type
        self.agent = PortfolioLabeledType()
        self.agent.cycles = 0
        self.agent.solve()


# Note that this ^^ test is not run because it has a setUp() method but no other
# methods. The risky asset-based models in ConsLabeledModel are untested and
# might not have worked before changes were made in Feb 2025. They surely do
# not work now.

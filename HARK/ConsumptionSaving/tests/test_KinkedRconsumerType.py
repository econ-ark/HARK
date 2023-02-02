import unittest

from HARK.ConsumptionSaving.ConsIndShockModel import KinkedRconsumerType
from HARK.tests import HARK_PRECISION


class testKinkedRConsumerType(unittest.TestCase):
    def test_liquidity_constraint(self):
        KinkyExample = KinkedRconsumerType()
        KinkyExample.cycles = 0

        # The consumer cannot borrow more than 0.4
        # times their permanent income
        KinkyExample.BoroCnstArt = -0.4

        # Solve the consumer's problem
        KinkyExample.solve()

        self.assertAlmostEqual(
            KinkyExample.solution[0].cFunc(1).tolist(), 0.96233, places=HARK_PRECISION
        )

        self.assertAlmostEqual(
            KinkyExample.solution[0].cFunc(4).tolist(), 1.34377, places=HARK_PRECISION
        )

        KinkyExample.BoroCnstArt = -0.2
        KinkyExample.solve()

        self.assertAlmostEqual(
            KinkyExample.solution[0].cFunc(1).tolist(), 0.93469, places=HARK_PRECISION
        )

        self.assertAlmostEqual(
            KinkyExample.solution[0].cFunc(4).tolist(), 1.34014, places=HARK_PRECISION
        )

import unittest

from HARK.ConsumptionSaving.ConsIndShockModel import KinkedRconsumerType
from tests import HARK_PRECISION


class testKinkedRConsumerType(unittest.TestCase):
    def test_liquidity_constraint(self):
        KinkyExample = KinkedRconsumerType(cycles=0)

        # The consumer cannot borrow more than 0.4
        # times their permanent income
        KinkyExample.BoroCnstArt = -0.4

        # Solve the consumer's problem
        KinkyExample.solve()

        self.assertAlmostEqual(
            KinkyExample.solution[0].cFunc(1).tolist(), 0.96161, places=HARK_PRECISION
        )

        self.assertAlmostEqual(
            KinkyExample.solution[0].cFunc(4).tolist(), 1.34274, places=HARK_PRECISION
        )

        KinkyExample.BoroCnstArt = -0.2
        KinkyExample.solve()

        self.assertAlmostEqual(
            KinkyExample.solution[0].cFunc(1).tolist(), 0.93444, places=HARK_PRECISION
        )

        self.assertAlmostEqual(
            KinkyExample.solution[0].cFunc(4).tolist(), 1.33927, places=HARK_PRECISION
        )

    def test_cubic_and_vFunc(self):
        CubicExample = KinkedRconsumerType(cycles=0, vFuncBool=True, CubicBool=True)
        CubicExample.solve()
        cFunc = CubicExample.solution[0].cFunc
        vFunc = CubicExample.solution[0].vFunc

        m = 3.0
        self.assertAlmostEqual(cFunc(m), 1.25611, places=HARK_PRECISION)
        self.assertAlmostEqual(vFunc(m), -15.3711, places=HARK_PRECISION)

    def test_calc_bounding_values(self):
        KinkyExample = KinkedRconsumerType(cycles=0)
        KinkyExample.calc_bounding_values()

    def test_default(self):
        BoopType = KinkedRconsumerType()
        BasicType = KinkedRconsumerType(Rboro=BoopType.Rsave)
        BasicType.solve()

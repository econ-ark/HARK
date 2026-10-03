import pickle
import unittest
from copy import deepcopy

import numpy as np

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
        self.assertAlmostEqual(cFunc(m), 1.25612, places=HARK_PRECISION)
        # -15.3711 until the kink segment was fixed to lie on the 45-degree line (its
        # dip below cash-on-hand reached the fixed point): now -15.37090.
        self.assertAlmostEqual(vFunc(m), -15.37090, places=HARK_PRECISION)

    def test_cubic_kink_segment_is_the_45_degree_line(self):
        """Between the two zero-asset points the unconstrained consumption function is c = m
        (end-of-period assets stay at zero): slope one inside the segment, the borrowing- and
        saving-side MPCs just outside it; and the patch survives deepcopy and pickling."""
        for BoroCnstArt in (None, 0.0):
            agent = KinkedRconsumerType(cycles=0, CubicBool=True)
            if BoroCnstArt is not None:
                agent.BoroCnstArt = BoroCnstArt
            agent.solve()
            cFunc = agent.solution[0].cFunc
            unc = cFunc.functions[0]
            on_line = np.flatnonzero(np.abs(unc.y_list - unc.x_list) < 1e-9)
            self.assertEqual(on_line.size, 2)
            m_lo, m_hi = unc.x_list[on_line]
            m = np.linspace(m_lo, m_hi, 101)
            for f in (cFunc, deepcopy(cFunc), pickle.loads(pickle.dumps(cFunc))):
                np.testing.assert_allclose(f(m), m, rtol=0, atol=1e-12)
                np.testing.assert_allclose(f.derivative(m[1:-1]), 1.0, rtol=0, atol=1e-10)
            eps = 1e-6 * (m_hi - m_lo)
            self.assertLess(unc.derivative(m_hi + eps), 0.99)  # the saving-side MPC above the kink
            self.assertLess(unc.derivative(m_lo - eps), 0.99)  # the borrowing-side MPC below it

    def test_calc_bounding_values(self):
        KinkyExample = KinkedRconsumerType(cycles=0)
        KinkyExample.calc_bounding_values()

    def test_default(self):
        BoopType = KinkedRconsumerType()
        BasicType = KinkedRconsumerType(Rboro=BoopType.Rsave)
        BasicType.solve()

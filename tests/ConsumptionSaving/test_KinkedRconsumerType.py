import pickle
import unittest
from unittest import mock
from copy import deepcopy

import numpy as np

from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    KinkedRconsumerType,
)
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
            KinkyExample.solution[0].cFunc(1).tolist(), 0.96135, places=HARK_PRECISION
        )

        self.assertAlmostEqual(
            KinkyExample.solution[0].cFunc(4).tolist(), 1.34204, places=HARK_PRECISION
        )

        KinkyExample.BoroCnstArt = -0.2
        KinkyExample.solve()

        self.assertAlmostEqual(
            KinkyExample.solution[0].cFunc(1).tolist(), 0.93424, places=HARK_PRECISION
        )

        self.assertAlmostEqual(
            KinkyExample.solution[0].cFunc(4).tolist(), 1.33858, places=HARK_PRECISION
        )

    def test_cubic_refused(self):
        # Cubic Hermite interpolation is refused with a sum over discrete shock atoms
        CubicExample = KinkedRconsumerType(cycles=0, CubicBool=True)
        self.assertRaises(ValueError, CubicExample.solve)

    # The two tests below exercise the cubic branch itself, which the pairing rule
    # makes unreachable, so they solve with the check disabled
    @mock.patch(
        "HARK.ConsumptionSaving.ConsIndShockModel.check_cubic_pairing",
        lambda *args, **kwargs: None,
    )
    def test_cubic_and_vFunc(self):
        CubicExample = KinkedRconsumerType(cycles=0, vFuncBool=True, CubicBool=True)
        CubicExample.solve()
        cFunc = CubicExample.solution[0].cFunc
        vFunc = CubicExample.solution[0].vFunc

        m = 3.0
        # 1.25551 until the natural constraint came from the shock laws' infima
        self.assertAlmostEqual(cFunc(m), 1.25420, places=HARK_PRECISION)
        # -15.3711 until the kink segment was fixed to lie on the 45-degree line (its
        # dip below cash-on-hand reached the fixed point), then -15.37090; -15.38487
        # since HARK's default income shocks are the truncated lognormal;
        # -15.38712 since the natural constraint comes from the laws' infima.
        self.assertAlmostEqual(vFunc(m), -15.39614, places=HARK_PRECISION)

    @mock.patch(
        "HARK.ConsumptionSaving.ConsIndShockModel.check_cubic_pairing",
        lambda *args, **kwargs: None,
    )
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
                np.testing.assert_allclose(
                    f.derivative(m[1:-1]), 1.0, rtol=0, atol=1e-10
                )
            eps = 1e-6 * (m_hi - m_lo)
            self.assertLess(
                unc.derivative(m_hi + eps), 0.99
            )  # the saving-side MPC above the kink
            self.assertLess(
                unc.derivative(m_lo - eps), 0.99
            )  # the borrowing-side MPC below it

    def test_calc_bounding_values(self):
        KinkyExample = KinkedRconsumerType(cycles=0)
        KinkyExample.calc_bounding_values()

    def test_default(self):
        BoopType = KinkedRconsumerType()
        BasicType = KinkedRconsumerType(Rboro=BoopType.Rsave)
        BasicType.solve()


class testKinkedRNaturalLimitFromTheInfima(unittest.TestCase):
    """
    KinkedR's natural borrowing constraint comes from the infima of the true
    shock distributions, at the borrowing rate, where they are bounded below away
    from zero, as IndShock's does, and binds on the atoms as an artificial
    constraint does.
    """

    # HARK's lognormals bounded at three standard deviations of their logs,
    # whatever HARK's default is
    BOUNDED = dict(PermShk_inf=-3.0, PermShk_sup=3.0, TranShk_inf=-3.0, TranShk_sup=3.0)

    def test_the_limit_at_the_borrowing_rate(self):
        agent = KinkedRconsumerType(cycles=0, BoroCnstArt=None, **self.BOUNDED)
        agent.solve()
        perm_inf, tran_inf = np.ravel(agent.IncShkDstn[0].limit["infimum"])
        kappa = agent.PermGroFac[0] * perm_inf / agent.Rboro
        self.assertAlmostEqual(
            agent.solution[0].mNrmMin, -tran_inf * kappa / (1.0 - kappa), places=10
        )
        self.assertEqual(agent.solution[0].MPCmax, 1.0)
        # The atoms' own limit allows more borrowing than a draw from the true
        # laws could repay
        perm_min, tran_min = np.min(agent.IncShkDstn[0].atoms, axis=1)
        kappa = agent.PermGroFac[0] * perm_min / agent.Rboro
        atoms_limit = -tran_min * kappa / (1.0 - kappa)
        self.assertLess(atoms_limit, agent.solution[0].mNrmMin - 0.2)

    def test_equal_rates_take_the_indshock_solver(self):
        m = np.linspace(0.0, 10.0, 201)
        kinky = KinkedRconsumerType(cycles=1, Rboro=1.03, Rsave=1.03, BoroCnstArt=None)
        kinky.solve()
        plain = IndShockConsumerType(cycles=1, Rfree=[1.03], BoroCnstArt=None)
        plain.solve()
        self.assertEqual(kinky.solution[0].mNrmMin, plain.solution[0].mNrmMin)
        np.testing.assert_allclose(
            kinky.solution[0].cFunc(m), plain.solution[0].cFunc(m), rtol=1e-12
        )

    def test_laws_unbounded_below_keep_the_atoms_limit(self):
        # A law unbounded below records an infimum of zero, where the infima would
        # allow no borrowing at all; the atoms' limit stands, at the borrowing
        # rate, with both shocks unbounded and with the permanent shock alone
        unbounded = dict(
            PermShk_inf=-np.inf,
            PermShk_sup=np.inf,
            TranShk_inf=-np.inf,
            TranShk_sup=np.inf,
        )
        unbounded_perm = dict(PermShk_inf=-np.inf, PermShk_sup=np.inf)
        # Both shocks unbounded halt at the employed-floor check (testEmployedFloor)
        for kwds in (unbounded_perm,):
            agent = KinkedRconsumerType(cycles=0, BoroCnstArt=None, **kwds)
            agent.solve()
            perm_min, tran_min = np.min(agent.IncShkDstn[0].atoms, axis=1)
            kappa = agent.PermGroFac[0] * perm_min / agent.Rboro
            self.assertLess(agent.solution[0].mNrmMin, 0.0)
            self.assertAlmostEqual(
                agent.solution[0].mNrmMin, -tran_min * kappa / (1.0 - kappa), places=10
            )

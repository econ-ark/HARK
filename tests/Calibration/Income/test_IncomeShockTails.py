"""
Tests for bounded income shocks in IndShock's income constructor: the bounds
PermShk_inf, PermShk_sup, TranShk_inf and TranShk_sup, in standard deviations of
the underlying normal, with None (the default) or an infinite bound for no
bound, and the treatment
of the tails at the bounds, PermShk_tails and TranShk_tails: "truncated" (the
default) or "jumpless".
"""

import unittest

import numpy as np

from HARK.Calibration.Income.IncomeProcesses import (
    BufferStockIncShkDstn,
    LognormPermIncShk,
    MixtureTranIncShk,
    construct_lognormal_income_process_unemployment,
    get_PermShkLaw_from_IncShkDstn,
    get_TranShkLaw_from_IncShkDstn,
)
from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    init_lifecycle,
)
from HARK.distributions import (
    LogNormalTruncatedJumplessMeanOne,
    LogNormalTruncatedMeanOne,
    PointMassAtInfimum,
)

BOUNDS = {
    "PermShk_inf": -3.0,
    "PermShk_sup": 3.0,
    "TranShk_inf": -3.0,
    "TranShk_sup": 3.0,
}
TAILS = {"PermShk_tails": "truncated", "TranShk_tails": "truncated"}
# The jumpless law's bound at sigma = 0.1 and a bound of three standard deviations
JUMPLESS_BOUND = 0.734030


def sd_of_log(law):
    m1 = law.expect(np.log)
    return np.sqrt(law.expect(lambda x: np.log(x) ** 2) - m1**2)


class testIncomeShockTails(unittest.TestCase):
    def test_no_bounds_is_the_lognormal_process(self):
        plain = BufferStockIncShkDstn(0.1, 0.1, 7, 7, 0.05, 0.3)
        none = BufferStockIncShkDstn(
            0.1,
            0.1,
            7,
            7,
            0.05,
            0.3,
            **{key: None for key in BOUNDS},
            **TAILS,
        )
        np.testing.assert_array_equal(plain.atoms, none.atoms)
        np.testing.assert_array_equal(plain.pmv, none.pmv)
        agent = IndShockConsumerType()
        keys = BOUNDS.keys() | TAILS.keys()
        self.assertEqual(agent.IncShkDstn.conditional.keys() & keys, set())

    def test_infinite_bound_is_no_bound(self):
        plain = BufferStockIncShkDstn(0.1, 0.1, 7, 7, 0.05, 0.3)
        infinite = BufferStockIncShkDstn(
            0.1,
            0.1,
            7,
            7,
            0.05,
            0.3,
            PermShk_inf=-np.inf,
            PermShk_sup=np.inf,
            TranShk_inf=-np.inf,
            TranShk_sup=np.inf,
            **TAILS,
        )
        np.testing.assert_array_equal(plain.atoms, infinite.atoms)
        np.testing.assert_array_equal(plain.pmv, infinite.pmv)
        # An infinite bound on one side leaves the bound on the other alone
        one = LognormPermIncShk(0.1, 7, bot=-3.0, top=np.inf)
        below = LognormPermIncShk(0.1, 7, bot=-3.0)
        np.testing.assert_array_equal(one.atoms, below.atoms)
        np.testing.assert_array_equal(one.pmv, below.pmv)
        self.assertEqual(one.limit["supremum"], below.limit["supremum"])

    def test_permanent_shock_truncated(self):
        shock = LognormPermIncShk(0.1, 7, bot=-3.0, top=3.0, tails="truncated")
        law = shock.limit["dist"]
        self.assertIsInstance(law, LogNormalTruncatedMeanOne)
        self.assertAlmostEqual(sd_of_log(law), 0.1, places=12)
        expected = law.discretize(7, method="equiprobable")
        np.testing.assert_array_equal(shock.atoms, expected.atoms)
        self.assertEqual(len(shock.atoms[0]), 7)
        self.assertAlmostEqual(np.dot(shock.pmv, shock.atoms[0]), 1.0, places=14)
        # the truncation points are the law's infimum and supremum, at three
        # standard deviations of the underlying normal
        self.assertEqual(shock.limit["infimum"][0], law.infimum[0])
        self.assertEqual(shock.limit["supremum"][0], law.supremum[0])
        z = (np.log([law.infimum[0], law.supremum[0]]) - law.mu) / law.sigma
        np.testing.assert_allclose(z, [-3.0, 3.0], rtol=1e-12)
        self.assertTrue(
            np.all(shock.atoms > shock.limit["infimum"])
            and np.all(shock.atoms < shock.limit["supremum"])
        )

    def test_permanent_shock_jumpless(self):
        shock = LognormPermIncShk(0.1, 7, bot=-3.0, tails="jumpless")
        law = shock.limit["dist"]
        self.assertIsInstance(law, LogNormalTruncatedJumplessMeanOne)
        self.assertAlmostEqual(law._law.variance() * law.sigma**2, 0.01, places=15)
        expected = law.discretize(7, method="equiprobable")
        np.testing.assert_array_equal(shock.atoms, expected.atoms)
        self.assertEqual(len(shock.atoms[0]), 7)
        self.assertAlmostEqual(np.dot(shock.pmv, shock.atoms[0]), 1.0, places=14)
        # bounded below, where the density is zero, and not above
        self.assertAlmostEqual(shock.limit["infimum"][0], JUMPLESS_BOUND, places=6)
        self.assertEqual(shock.limit["supremum"][0], np.inf)
        self.assertEqual(law.pdf(law.infimum[0]), 0.0)
        self.assertTrue(np.all(shock.atoms > shock.limit["infimum"]))
        with self.assertRaises(ValueError):
            LognormPermIncShk(0.1, 7, bot=-3.0, top=3.0, tails="jumpless")
        # an infinite upper bound is no bound
        unbounded = LognormPermIncShk(0.1, 7, bot=-3.0, top=np.inf, tails="jumpless")
        np.testing.assert_array_equal(unbounded.atoms, shock.atoms)

    def test_point_mass_at_the_permanent_shocks_infimum(self):
        shock = LognormPermIncShk(
            0.1, 7, bot=-3.0, top=3.0, tails="truncated", infimum_prb=0.002
        )
        law = shock.limit["dist"]
        self.assertIsInstance(law, PointMassAtInfimum)
        self.assertIsInstance(law.base, LogNormalTruncatedMeanOne)
        atoms, pmv = shock.atoms[0], shock.pmv
        self.assertEqual(len(atoms), 8)
        self.assertEqual(atoms[0], shock.limit["infimum"][0])
        self.assertEqual(pmv[0], 0.002)
        self.assertAlmostEqual(np.dot(pmv, atoms), 1.0, places=14)
        # the lowest value keeps its probability whatever the number of atoms
        for n in (3, 25):
            other = LognormPermIncShk(
                0.1, n, bot=-3.0, top=3.0, tails="truncated", infimum_prb=0.002
            )
            self.assertEqual((other.atoms[0][0], other.pmv[0]), (atoms[0], pmv[0]))
        # it needs a lower bound
        with self.assertRaises(ValueError):
            LognormPermIncShk(0.1, 7, infimum_prb=0.002)
        # and through the constructor, whose laws can be read back
        args = dict(
            T_cycle=1,
            PermShkStd=[0.1],
            PermShkCount=7,
            TranShkStd=[0.1],
            TranShkCount=7,
            T_retire=0,
            UnempPrb=0.05,
            IncUnemp=0.3,
            UnempPrbRet=0.0,
            IncUnempRet=0.0,
            PermShk_inf=-3.0,
            PermShk_sup=3.0,
            TranShk_inf=-3.0,
            TranShk_sup=3.0,
            PermShk_tails="truncated",
            TranShk_tails="truncated",
        )
        for prb, kind in (
            (0.002, PointMassAtInfimum),
            (None, LogNormalTruncatedMeanOne),
        ):
            IncShkDstn = construct_lognormal_income_process_unemployment(
                RNG=np.random.default_rng(0), PermShkMinPrb=prb, **args
            )
            self.assertIsInstance(get_PermShkLaw_from_IncShkDstn(IncShkDstn)[0], kind)
            tran = get_TranShkLaw_from_IncShkDstn(IncShkDstn)[0]
            self.assertIsInstance(tran, LogNormalTruncatedMeanOne)
            self.assertAlmostEqual(float(tran.mean()), 1.0, places=12)

    def test_one_sided_bound(self):
        shock = LognormPermIncShk(0.1, 7, bot=-3.0)
        self.assertEqual(len(shock.atoms[0]), 7)
        self.assertEqual(shock.limit["supremum"][0], np.inf)
        self.assertGreater(shock.limit["infimum"][0], 0.0)
        self.assertAlmostEqual(np.dot(shock.pmv, shock.atoms[0]), 1.0, places=14)

    def test_transitory_shock_with_unemployment(self):
        scale = (1.0 - 0.05 * 0.3) / (1.0 - 0.05)
        shock = MixtureTranIncShk(0.1, 0.05, 0.3, 7, bot=-3.0, top=3.0)
        employed = LogNormalTruncatedMeanOne.from_sd_log(0.1, -3.0, 3.0).discretize(
            7, method="equiprobable"
        )
        np.testing.assert_allclose(
            shock.atoms[0], np.r_[0.3, scale * employed.atoms[0]], rtol=1e-15
        )
        np.testing.assert_allclose(shock.pmv, np.r_[0.05, 0.95 * employed.pmv])
        self.assertAlmostEqual(np.dot(shock.pmv, shock.atoms[0]), 1.0, places=14)
        self.assertEqual(shock.limit["infimum"][0], 0.3)
        # the employed law's supremum is rescaled with its atoms
        sup = shock.limit["supremum"][0]
        self.assertAlmostEqual(sup / (scale * employed.limit["supremum"][0]), 1.0, 14)
        self.assertTrue(np.all(shock.atoms[0] <= sup * (1.0 + 1e-15)))

    def test_unknown_tails_refused(self):
        with self.assertRaises(ValueError):
            LognormPermIncShk(0.1, 7, bot=-3.0, top=3.0, tails="censored")
        with self.assertRaises(ValueError):
            MixtureTranIncShk(0.1, 0.05, 0.3, 7, tails="clipped")
        with self.assertRaises(ValueError):
            construct_lognormal_income_process_unemployment(
                T_cycle=1,
                PermShkStd=[0.1],
                PermShkCount=7,
                TranShkStd=[0.1],
                TranShkCount=7,
                T_retire=0,
                UnempPrb=0.05,
                IncUnemp=0.3,
                UnempPrbRet=0.005,
                IncUnempRet=0.0,
                RNG=np.random.default_rng(0),
                TranShk_tails="Truncated",
            )

    def test_agent(self):
        law = LogNormalTruncatedMeanOne.from_sd_log(0.1, -3.0, 3.0)
        inf, sup = law.infimum[0], law.supremum[0]
        agent = IndShockConsumerType(cycles=0, **BOUNDS)
        shocks = agent.IncShkDstn[0]
        perm, tran = shocks.atoms
        self.assertAlmostEqual(np.dot(shocks.pmv, perm), 1.0, places=13)
        self.assertAlmostEqual(np.dot(shocks.pmv, tran), 1.0, places=13)
        self.assertTrue(np.all(perm >= inf - 1e-6) and np.all(perm <= sup + 1e-6))
        agent.solve()
        self.assertTrue(np.isfinite(agent.solution[0].cFunc(1.0)))

    def test_lifecycle(self):
        # the bounds apply in every working period; retirement keeps its points
        plain = IndShockConsumerType(**init_lifecycle)
        agent = IndShockConsumerType(**{**init_lifecycle, **BOUNDS})
        for t in range(agent.T_cycle):
            sigma = init_lifecycle["PermShkStd"][t]
            perm = agent.IncShkDstn[t].atoms[0]
            if sigma > 0.0:
                bounds = LogNormalTruncatedMeanOne.from_sd_log(sigma, -3.0, 3.0)
                self.assertTrue(np.all(perm >= bounds.infimum[0] * (1 - 1e-15)))
                self.assertTrue(np.all(perm <= bounds.supremum[0] * (1 + 1e-15)))
            else:
                np.testing.assert_array_equal(perm, plain.IncShkDstn[t].atoms[0])

    def test_bounds_by_period(self):
        # One bound for each period gives the same process as one for every period
        T = init_lifecycle["T_cycle"]
        every = IndShockConsumerType(**{**init_lifecycle, **BOUNDS})
        each = IndShockConsumerType(
            **{**init_lifecycle, **{key: [b] * T for key, b in BOUNDS.items()}}
        )
        for t in range(T):
            np.testing.assert_array_equal(
                every.IncShkDstn[t].atoms, each.IncShkDstn[t].atoms
            )
        # and the bounds can differ by period
        varying = IndShockConsumerType(
            **{**init_lifecycle, **BOUNDS, "PermShk_inf": [-2.0] + [-3.0] * (T - 1)}
        )
        self.assertGreater(
            varying.IncShkDstn[0].limit["infimum"][0],
            every.IncShkDstn[0].limit["infimum"][0],
        )
        np.testing.assert_array_equal(
            varying.IncShkDstn[1].atoms, every.IncShkDstn[1].atoms
        )
        # a one-period model takes a list of one
        IndShockConsumerType(cycles=0, PermShk_inf=[-3.0])

    def test_bounds_of_another_length_refused(self):
        with self.assertRaisesRegex(ValueError, "T_cycle = 1"):
            IndShockConsumerType(cycles=0, PermShk_inf=[-3.0, -3.0])

    def test_nan_bound_refused(self):
        with self.assertRaisesRegex(ValueError, "TranShk_sup is nan"):
            IndShockConsumerType(cycles=0, TranShk_sup=np.nan)
        with self.assertRaisesRegex(ValueError, "bot is nan"):
            LognormPermIncShk(0.1, 7, bot=np.nan)

    def test_bounds_on_the_wrong_side_refused(self):
        # A lower bound at or above the mean, or an upper one at or below it, is
        # refused under either treatment of the tails
        for tails in ("truncated", "jumpless"):
            with self.assertRaisesRegex(ValueError, "lower bound"):
                LognormPermIncShk(0.1, 7, bot=2.0, tails=tails)
            with self.assertRaisesRegex(ValueError, "lower bound"):
                LognormPermIncShk(0.1, 7, bot=0.0, tails=tails)
        with self.assertRaisesRegex(ValueError, "upper bound"):
            LognormPermIncShk(0.1, 7, bot=-3.0, top=-1.0)
        with self.assertRaisesRegex(ValueError, "bounded below only"):
            LognormPermIncShk(0.1, 7, bot=-3.0, top=3.0, tails="jumpless")

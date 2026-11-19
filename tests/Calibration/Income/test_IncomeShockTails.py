"""
Tests for bounded income shocks in IndShock's income constructor: the bounds
PermShk_inf, PermShk_sup, TranShk_inf and TranShk_sup, in standard deviations of
the underlying normal, with None (the default) for the default bounds of the
treatment of the tails and an infinite bound for no bound, and the treatment
of the tails at the bounds, PermShk_tails and TranShk_tails: "tapered" (the
default) or "pooled".
"""

import unittest

import numpy as np
from scipy.stats import norm

from HARK.Calibration.Income.IncomeProcesses import (
    BufferStockIncShkDstn,
    LognormPermIncShk,
    MixtureTranIncShk,
    construct_lognormal_income_process_unemployment,
    construct_markov_lognormal_income_process_unemployment,
)
from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    init_lifecycle,
)
from HARK.distributions import (
    LogNormalMeanOne,
    LogNormalPooledTailsMeanOne,
    LogNormalTaperedMeanOne,
)

BOUNDS = {
    "PermShk_inf": -3.0,
    "PermShk_sup": 3.0,
    "TranShk_inf": -3.0,
    "TranShk_sup": 3.0,
}
POOLED = {"PermShk_tails": "pooled", "TranShk_tails": "pooled"}

# The laws' bounds at sigma = 0.1 and bounds of three standard deviations
TAPERED_BOUNDS = (0.736861, 1.343574)
POOLED_BOUNDS = (0.716774, 1.382230)


def sd_of_log(law):
    m1 = law.expect(np.log)
    return np.sqrt(law.expect(lambda x: np.log(x) ** 2) - m1**2)


class testIncomeShockTails(unittest.TestCase):
    def test_no_bounds_is_the_lognormal_process(self):
        plain = BufferStockIncShkDstn(0.1, 0.1, 7, 7, 0.05, 0.3)
        for tails in ("tapered", "pooled"):
            none = BufferStockIncShkDstn(
                0.1,
                0.1,
                7,
                7,
                0.05,
                0.3,
                **{key: None for key in BOUNDS},
                PermShk_tails=tails,
                TranShk_tails=tails,
            )
            np.testing.assert_array_equal(plain.atoms, none.atoms)
            np.testing.assert_array_equal(plain.pmv, none.pmv)
        # In the agent's income constructor None takes the default bounds, and
        # an infinite bound is no bound: with infinite bounds the agent carries
        # none of the bounds' parameters
        agent = IndShockConsumerType(
            **{key: np.inf if key.endswith("sup") else -np.inf for key in BOUNDS}
        )
        keys = BOUNDS.keys() | POOLED.keys()
        self.assertEqual(agent.IncShkDstn.conditional.keys() & keys, set())

    def test_permanent_shock_tapered(self):
        shock = LognormPermIncShk(0.1, 7, bot=-3.0, top=3.0)
        law = shock.limit["dist"]
        self.assertIsInstance(law, LogNormalTaperedMeanOne)
        self.assertAlmostEqual(law._law.variance() * law.sigma**2, 0.01, places=15)
        expected = law.discretize(7, method="equiprobable")
        np.testing.assert_array_equal(shock.atoms, expected.atoms)
        self.assertEqual(len(shock.atoms[0]), 7)
        self.assertAlmostEqual(np.dot(shock.pmv, shock.atoms[0]), 1.0, places=14)
        self.assertAlmostEqual(shock.limit["infimum"][0], TAPERED_BOUNDS[0], places=6)
        self.assertAlmostEqual(shock.limit["supremum"][0], TAPERED_BOUNDS[1], places=6)
        self.assertTrue(
            np.all(shock.atoms > shock.limit["infimum"])
            and np.all(shock.atoms < shock.limit["supremum"])
        )

    def test_permanent_shock_pooled(self):
        shock = LognormPermIncShk(0.1, 7, bot=-3.0, top=3.0, tails="pooled")
        law = shock.limit["dist"]
        self.assertIsInstance(law, LogNormalPooledTailsMeanOne)
        self.assertAlmostEqual(sd_of_log(law), 0.1, places=12)
        atoms, pmv = shock.atoms[0], shock.pmv
        # the pooled points, with their probabilities, beside 7 atoms between the bounds
        self.assertEqual(len(atoms), 9)
        self.assertEqual(atoms[0], shock.limit["infimum"][0])
        self.assertEqual(atoms[-1], shock.limit["supremum"][0])
        self.assertAlmostEqual(pmv[0], norm.cdf(-3.0), places=15)
        self.assertAlmostEqual(pmv[-1], norm.sf(3.0), places=15)
        self.assertAlmostEqual(np.dot(pmv, atoms), 1.0, places=14)
        self.assertAlmostEqual(shock.limit["infimum"][0], POOLED_BOUNDS[0], places=6)
        self.assertAlmostEqual(shock.limit["supremum"][0], POOLED_BOUNDS[1], places=6)
        # the lowest value keeps its probability whatever the number of atoms
        for n in (3, 25):
            other = LognormPermIncShk(0.1, n, bot=-3.0, top=3.0, tails="pooled")
            self.assertEqual(other.atoms[0][0], atoms[0])
            self.assertEqual(other.pmv[0], pmv[0])

    def test_one_sided_bound(self):
        for tails, n_atoms in (("tapered", 7), ("pooled", 8)):
            shock = LognormPermIncShk(0.1, 7, bot=-3.0, tails=tails)
            self.assertEqual(len(shock.atoms[0]), n_atoms)
            self.assertEqual(shock.limit["supremum"][0], np.inf)
            self.assertGreater(shock.limit["infimum"][0], 0.0)
            self.assertAlmostEqual(np.dot(shock.pmv, shock.atoms[0]), 1.0, places=14)

    def test_transitory_shock_with_unemployment(self):
        scale = (1.0 - 0.05 * 0.3) / (1.0 - 0.05)
        for tails, law in (
            ("tapered", LogNormalTaperedMeanOne),
            ("pooled", LogNormalPooledTailsMeanOne),
        ):
            shock = MixtureTranIncShk(0.1, 0.05, 0.3, 7, bot=-3.0, top=3.0, tails=tails)
            employed = law.from_sd_log(0.1, -3.0, 3.0).discretize(
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
            self.assertAlmostEqual(
                sup / (scale * employed.limit["supremum"][0]), 1.0, 14
            )
            self.assertTrue(np.all(shock.atoms[0] <= sup * (1.0 + 1e-15)))
            if tails == "pooled":
                # the upper pooled point is the top atom and the supremum
                self.assertAlmostEqual(shock.atoms[0][-1] / sup, 1.0, places=15)

    def test_pooled_transitory_shock_without_unemployment(self):
        # With no unemployment, the pooled lower tail is the lowest value, with
        # positive probability
        shock = MixtureTranIncShk(0.1, 0.0, 0.0, 7, bot=-3.0, top=3.0, tails="pooled")
        self.assertEqual(shock.atoms[0][0], shock.limit["infimum"][0])
        self.assertAlmostEqual(shock.pmv[0], norm.cdf(-3.0), places=15)
        self.assertAlmostEqual(np.dot(shock.pmv, shock.atoms[0]), 1.0, places=14)

    def test_unknown_tails_refused(self):
        with self.assertRaises(ValueError):
            LognormPermIncShk(0.1, 7, bot=-3.0, top=3.0, tails="censored")
        with self.assertRaises(ValueError):
            MixtureTranIncShk(0.1, 0.05, 0.3, 7, tails="truncated")
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
                TranShk_tails="Pooled",
            )

    def test_agent(self):
        for tails, (inf, sup) in (
            ({}, TAPERED_BOUNDS),
            (POOLED, POOLED_BOUNDS),
        ):
            agent = IndShockConsumerType(cycles=0, **BOUNDS, **tails)
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
        for tails, law in (
            ({}, LogNormalTaperedMeanOne),
            (POOLED, LogNormalPooledTailsMeanOne),
        ):
            agent = IndShockConsumerType(**{**init_lifecycle, **BOUNDS, **tails})
            for t in range(agent.T_cycle):
                sigma = init_lifecycle["PermShkStd"][t]
                perm = agent.IncShkDstn[t].atoms[0]
                if sigma > 0.0:
                    bounds = law.from_sd_log(sigma, -3.0, 3.0)
                    self.assertTrue(np.all(perm >= bounds.infimum[0] * (1 - 1e-15)))
                    self.assertTrue(np.all(perm <= bounds.supremum[0] * (1 + 1e-15)))
                else:
                    np.testing.assert_array_equal(perm, plain.IncShkDstn[t].atoms[0])


def income_laws(**kwds):
    """The permanent and employed transitory laws the income constructor records."""
    dstn = construct_lognormal_income_process_unemployment(
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
        **kwds,
    )[0]
    perm, tran = dstn.limit["dist"]
    return perm.limit["dist"], tran.limit["dist"].limit["dist"]


class testDefaultBounds(unittest.TestCase):
    """
    HARK's default income shocks are bounded: a bound left None is the default
    for the treatment of the tails, SHOCK_BOUND_DEFAULT. A tapered shock is zero
    at three standard deviations, tapered from two; a pooled one is pooled
    beyond two. An infinite bound is no bound.
    """

    def test_default_is_tapered_from_two_to_three(self):
        for law in income_laws():
            self.assertIsInstance(law, LogNormalTaperedMeanOne)
            self.assertEqual((law.bot, law.top), (-3.0, 3.0))
            self.assertEqual((law._law.c_lo, law._law.c_hi), (-2.0, 2.0))

    def test_pooled_default_beyond_two(self):
        perm, tran = income_laws(PermShk_tails="pooled")
        self.assertIsInstance(perm, LogNormalPooledTailsMeanOne)
        self.assertEqual((perm.bot, perm.top), (-2.0, 2.0))
        self.assertIsInstance(tran, LogNormalTaperedMeanOne)
        self.assertEqual((tran.bot, tran.top), (-3.0, 3.0))

    def test_given_bounds_are_kept_and_infinite_ones_are_none(self):
        perm, tran = income_laws(
            PermShk_inf=-np.inf, PermShk_sup=np.inf, TranShk_inf=-2.5
        )
        self.assertIs(type(perm), LogNormalMeanOne)
        self.assertEqual((tran.bot, tran.top), (-2.5, 3.0))

    def test_the_default_agent(self):
        agent = IndShockConsumerType(cycles=0)
        law = agent.IncShkDstn[0].limit["dist"][0].limit["dist"]
        self.assertIsInstance(law, LogNormalTaperedMeanOne)
        self.assertEqual((law.bot, law.top), (-3.0, 3.0))

    def test_markov_states_are_bounded_as_indshock_is(self):
        # A one-state Markov income process gets IndShock's laws and atoms, bounds
        # and tails included, whether left at the default or given
        common = dict(T_cycle=1, PermShkCount=7, TranShkCount=7, T_retire=0)
        common.update(UnempPrbRet=None, IncUnempRet=None)
        for kwds in ({}, {"PermShk_tails": "pooled", "TranShk_inf": -2.5}):
            ind = construct_lognormal_income_process_unemployment(
                PermShkStd=[0.1],
                TranShkStd=[0.1],
                UnempPrb=0.05,
                IncUnemp=0.3,
                RNG=np.random.default_rng(0),
                **common,
                **kwds,
            )[0]
            mrkv = construct_markov_lognormal_income_process_unemployment(
                PermShkStd=np.array([[0.1]]),
                TranShkStd=np.array([[0.1]]),
                UnempPrb=np.array([0.05]),
                IncUnemp=np.array([0.3]),
                RNG=np.random.default_rng(0),
                **common,
                **kwds,
            )[0][0]
            for a, b in zip(ind.atoms, mrkv.atoms):
                np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(ind.pmv, mrkv.pmv)
            for law_i, law_m in zip(ind.limit["dist"], mrkv.limit["dist"]):
                self.assertIs(type(law_i), type(law_m))
            np.testing.assert_array_equal(ind.limit["infimum"], mrkv.limit["infimum"])

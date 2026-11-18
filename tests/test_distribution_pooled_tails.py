import unittest

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from HARK.distributions import (
    DiscreteDistribution,
    LogNormal,
    LogNormalPooledTails,
    LogNormalPooledTailsMeanOne,
)


def lognormal_tail(mu, sigma, lo, hi, f=lambda x: x, epsabs=1e-15):
    """The integral of f against LogNormal(mu, sigma) for lo < Z < hi, by quadrature."""

    def integrand(z):
        w = norm.pdf(z)
        return 0.0 if w == 0.0 else f(np.exp(mu + sigma * z)) * w

    value, _ = quad(
        integrand,
        lo,
        hi,
        epsabs=epsabs,
        epsrel=1e-13,
        limit=400,
    )
    return value


class LogNormalPooledTailsTests(unittest.TestCase):
    """
    The lognormal with pooled tails: the lognormal's own law between its cutoffs,
    and each tail beyond a cutoff a single point with the tail's probability at
    the tail's conditional mean.
    """

    cases = [(0.1, -3.0, 3.0), (0.2, -2.0, 2.0), (0.5, -2.5, 3.5), (1.0, -3.0, 2.0)]

    def test_probability_and_mean(self):
        for sigma, bot, top in self.cases:
            d = LogNormalPooledTails(mu=0.3, sigma=sigma, bot=bot, top=top)
            self.assertAlmostEqual(
                d.infimum_prb + d.body_prb + d.supremum_prb, 1.0, places=15
            )
            mean = np.exp(0.3 + 0.5 * sigma**2)
            self.assertAlmostEqual(d.mean() / mean, 1.0, places=13)
            self.assertAlmostEqual(d.moment(1) / mean, 1.0, places=13)
            self.assertAlmostEqual(d.expect(lambda x: x) / mean, 1.0, places=12)
            self.assertAlmostEqual(d.expect(lambda x: 1.0 + 0.0 * x), 1.0, places=12)

    def test_the_points_are_the_tails_conditional_means(self):
        for sigma, bot, top in self.cases:
            d = LogNormalPooledTails(mu=-0.2, sigma=sigma, bot=bot, top=top)
            self.assertAlmostEqual(d.infimum_prb, norm.cdf(bot), places=15)
            self.assertAlmostEqual(d.supremum_prb, norm.sf(top), places=15)
            lower = lognormal_tail(-0.2, sigma, -np.inf, bot) / norm.cdf(bot)
            upper = lognormal_tail(-0.2, sigma, top, np.inf) / norm.sf(top)
            self.assertAlmostEqual(d.infimum[0] / lower, 1.0, places=11)
            self.assertAlmostEqual(d.supremum[0] / upper, 1.0, places=11)
            # each point lies beyond its cutoff
            self.assertLess(d.infimum[0], np.exp(-0.2 + sigma * bot))
            self.assertGreater(d.supremum[0], np.exp(-0.2 + sigma * top))

    def test_not_censoring(self):
        # Censoring would put each tail's probability at its cutoff and move the mean
        d = LogNormalPooledTailsMeanOne(sigma=0.2, bot=-2.0, top=2.0)
        lo, hi = np.exp(d.mu + d.sigma * d.bot), np.exp(d.mu + d.sigma * d.top)
        censored_mean = (
            d.body_prb * d.body.mean() + d.infimum_prb * lo + d.supremum_prb * hi
        )
        self.assertLess(censored_mean, 1.0 - 1e-3)
        self.assertAlmostEqual(d.mean(), 1.0, places=14)

    def test_a_mean_preserving_contraction(self):
        # E[f(X)] is no larger than the lognormal's for convex f
        convex = [lambda x: 1.0 / x, lambda x: x**2, lambda x: -np.log(x)]
        for sigma, bot, top in self.cases:
            d = LogNormalPooledTails(sigma=sigma, bot=bot, top=top)
            for f in convex:
                pooled = d.expect(f)
                full = lognormal_tail(0.0, sigma, -np.inf, np.inf, f)
                self.assertLess(pooled, full)

    def test_the_lognormal_between_the_cutoffs(self):
        for sigma, bot, top in self.cases:
            d = LogNormalPooledTails(mu=0.1, sigma=sigma, bot=bot, top=top)
            ln = LogNormal(0.1, sigma)
            x = np.exp(0.1 + sigma * np.linspace(bot, top, 41)[1:-1])
            np.testing.assert_allclose(d.pdf(x), ln.pdf(x), rtol=1e-13)
            np.testing.assert_allclose(d.cdf(x), ln.cdf(x), rtol=1e-13, atol=1e-16)

    def test_steps_at_the_points(self):
        for sigma, bot, top in self.cases:
            d = LogNormalPooledTails(sigma=sigma, bot=bot, top=top)
            inf, sup = d.infimum[0], d.supremum[0]
            lo, hi = np.exp(sigma * bot), np.exp(sigma * top)
            # nothing below the lower point, nor between a point and its cutoff
            self.assertEqual(d.cdf(inf * (1 - 1e-12)), 0.0)
            for x in (inf, 0.5 * (inf + lo), lo * (1 - 1e-12)):
                self.assertAlmostEqual(d.cdf(x), norm.cdf(bot), places=14)
            for x in (hi, 0.5 * (hi + sup), sup * (1 - 1e-12)):
                self.assertAlmostEqual(d.cdf(x), norm.cdf(top), places=14)
            self.assertEqual(d.cdf(sup), 1.0)
            self.assertEqual(d.pdf(0.5 * (inf + lo)), 0.0)
            self.assertEqual(d.pdf(0.5 * (hi + sup)), 0.0)

    def test_quantiles(self):
        for sigma, bot, top in self.cases:
            d = LogNormalPooledTails(sigma=sigma, bot=bot, top=top)
            p_lo, p_hi = norm.cdf(bot), norm.sf(top)
            self.assertEqual(d.ppf(0.0), d.infimum[0])
            self.assertEqual(d.ppf(0.5 * p_lo), d.infimum[0])
            self.assertEqual(d.ppf(p_lo), d.infimum[0])
            self.assertEqual(d.ppf(1.0 - 0.5 * p_hi), d.supremum[0])
            self.assertEqual(d.ppf(1.0), d.supremum[0])
            q = np.linspace(p_lo, 1.0 - p_hi, 23)[1:-1]
            np.testing.assert_allclose(
                d.ppf(q), LogNormal(0.0, sigma).ppf(q), rtol=1e-12
            )
            np.testing.assert_allclose(d.cdf(d.ppf(q)), q, rtol=1e-12)

    def test_the_generic_quantile_route(self):
        # The expectation through the quantile function, as a solver reading the
        # law only through its generic interface would take it
        d = LogNormalPooledTailsMeanOne(sigma=0.2, bot=-2.0, top=2.0)
        p_lo, p_hi = d.infimum_prb, d.supremum_prb
        value, _ = quad(
            d.ppf, 0.0, 1.0, points=[p_lo, 1 - p_hi], limit=200, epsabs=1e-14
        )
        self.assertAlmostEqual(value, 1.0, places=11)

    def test_equiprobable_discretization(self):
        for sigma, bot, top in self.cases:
            d = LogNormalPooledTailsMeanOne(sigma=sigma, bot=bot, top=top)
            for n in (1, 2, 7, 25):
                dd = d.discretize(n)
                atoms, pmv = dd.atoms[0], dd.pmv
                self.assertEqual(len(atoms), n + 2)
                self.assertEqual(atoms[0], d.infimum[0])
                self.assertEqual(atoms[-1], d.supremum[0])
                self.assertEqual(pmv[0], d.infimum_prb)
                self.assertEqual(pmv[-1], d.supremum_prb)
                np.testing.assert_allclose(pmv[1:-1], np.ones(n) * d.body_prb / n)
                self.assertAlmostEqual(pmv.sum(), 1.0, places=14)
                self.assertAlmostEqual(np.dot(pmv, atoms), 1.0, places=13)
                self.assertTrue(np.all(np.diff(atoms) > 0.0))
                np.testing.assert_array_equal(dd.limit["infimum"], d.infimum)
                np.testing.assert_array_equal(dd.limit["supremum"], d.supremum)
            # a pooled point is already its side's bound
            self.assertEqual(len(d.discretize(7, endpoints=True).atoms[0]), 9)
            self.assertIs(d.discretize(7).limit["dist"], d)

    def test_draws(self):
        d = LogNormalPooledTailsMeanOne(sigma=0.2, bot=-2.0, top=2.0, seed=11)
        n = 200000
        x = d.draw(n)
        self.assertEqual(x.min(), d.infimum[0])
        self.assertEqual(x.max(), d.supremum[0])
        se = np.sqrt(d.infimum_prb * (1 - d.infimum_prb) / n)
        self.assertAlmostEqual(np.mean(x == d.infimum[0]), d.infimum_prb, delta=4 * se)
        self.assertAlmostEqual(
            np.mean(x == d.supremum[0]), d.supremum_prb, delta=4 * se
        )
        self.assertAlmostEqual(x.mean(), 1.0, delta=4 * 0.2 / np.sqrt(n))

    def test_mean_one_and_the_sd_of_the_log(self):
        for sd, bot, top in ((0.05, -3.0, 3.0), (0.1, -2.0, 2.0), (0.3, -3.0, 3.0)):
            d = LogNormalPooledTailsMeanOne.from_sd_log(sd, bot=bot, top=top)
            self.assertAlmostEqual(d.mean(), 1.0, places=13)
            m1 = d.expect(np.log)
            m2 = d.expect(lambda x: np.log(x) ** 2)
            self.assertAlmostEqual(np.sqrt(m2 - m1**2) / sd, 1.0, places=10)

    def test_one_side_pooled(self):
        d = LogNormalPooledTails(sigma=0.3, bot=-2.0, top=None)
        ln = LogNormal(0.0, 0.3)
        self.assertEqual(d.supremum_prb, 0.0)
        self.assertEqual(d.supremum[0], np.inf)
        x = np.exp(0.3 * np.array([-1.9, 0.0, 3.0, 6.0]))
        np.testing.assert_allclose(d.pdf(x), ln.pdf(x), rtol=1e-13)
        self.assertAlmostEqual(d.mean() / ln.mean(), 1.0, places=13)
        dd = d.discretize(5)
        self.assertEqual(len(dd.atoms[0]), 6)
        self.assertEqual(dd.atoms[0][0], d.infimum[0])
        ends = d.discretize(5, endpoints=True)
        self.assertEqual(ends.atoms[0][-1], np.inf)
        self.assertEqual(ends.pmv[-1], 0.0)
        u = LogNormalPooledTails(mu=0.1, sigma=0.3, bot=None, top=None)
        self.assertEqual(u.infimum_prb, 0.0)
        self.assertEqual(u.infimum[0], 0.0)
        np.testing.assert_allclose(u.pdf(x), LogNormal(0.1, 0.3).pdf(x), rtol=1e-13)
        np.testing.assert_allclose(u.cdf(x), LogNormal(0.1, 0.3).cdf(x), rtol=1e-13)

    def test_far_tails_keep_their_digits(self):
        d = LogNormalPooledTails(sigma=0.5, bot=-9.0, top=9.0)
        # relative tolerance only, since the tails' integrals are of order 1e-19
        lower = lognormal_tail(0.0, 0.5, -np.inf, -9.0, epsabs=0.0) / norm.cdf(-9.0)
        upper = lognormal_tail(0.0, 0.5, 9.0, np.inf, epsabs=0.0) / norm.sf(9.0)
        self.assertAlmostEqual(d.infimum[0] / lower, 1.0, places=12)
        self.assertAlmostEqual(d.supremum[0] / upper, 1.0, places=12)

    def test_point_and_refusals(self):
        point = LogNormalPooledTailsMeanOne(sigma=0.0)
        self.assertIsInstance(point, DiscreteDistribution)
        np.testing.assert_allclose(point.atoms, [[1.0]])
        with self.assertRaises(ValueError):
            LogNormalPooledTails(sigma=0.1, bot=1.0, top=1.0)
        with self.assertRaises(ValueError):
            LogNormalPooledTails(sigma=-0.1)
        with self.assertRaises(ValueError):
            LogNormalPooledTails(sigma=np.array([0.1, 0.2]))

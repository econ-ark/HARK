import unittest

import numpy as np
from scipy.integrate import quad
from scipy.stats import beta, norm

from HARK.distributions import (
    DiscreteDistribution,
    LogNormal,
    LogNormalTapered,
    LogNormalTaperedMeanOne,
)
from HARK.distributions.continuous import _taper_moment


def integrate(fn, pieces):
    """The integral of fn over consecutive pieces, each by adaptive quadrature."""
    return sum(
        quad(fn, a, b, epsabs=1e-15, epsrel=1e-13, limit=400)[0]
        for a, b in zip(pieces[:-1], pieces[1:])
    )


class LogNormalTaperedTests(unittest.TestCase):
    """
    The tapered lognormal: the lognormal's own density between its tapered
    tails, a power taper over the last standard deviation before each bound,
    zero at its bounds, with the lognormal's probability in each tail and the
    lognormal's mean.
    """

    cases = [(0.1, -3.0, 3.0), (0.2, -3.0, 3.0), (0.5, -3.0, 4.0), (1.0, -4.0, 3.5)]

    def pieces(self, d):
        """The support's ends and the tapered tails' starts, in x."""
        z = [d.bot, d.bot + 1.0, d.top - 1.0, d.top]
        return [np.exp(d.mu + d.sigma * v) for v in z]

    def test_probability_and_mean(self):
        for sigma, bot, top in self.cases:
            d = LogNormalTapered(mu=0.3, sigma=sigma, bot=bot, top=top)
            pieces = self.pieces(d)
            self.assertAlmostEqual(integrate(d.pdf, pieces), 1.0, places=12)
            mean = np.exp(0.3 + 0.5 * sigma**2)
            self.assertAlmostEqual(
                integrate(lambda x: x * d.pdf(x), pieces) / mean, 1.0, places=12
            )
            self.assertAlmostEqual(d.mean() / mean, 1.0, places=12)
            self.assertAlmostEqual(d.moment(1) / mean, 1.0, places=12)

    def test_each_tail_carries_the_lognormal_tail(self):
        for sigma, bot, top in self.cases:
            d = LogNormalTapered(sigma=sigma, bot=bot, top=top)
            inf, lo, hi, sup = self.pieces(d)
            # The lognormal's tails beyond bot + 1 and top - 1, in closed form
            mean = np.exp(0.5 * sigma**2)
            for a, b, tail_prob, tail_mean in (
                (inf, lo, norm.cdf(bot + 1), mean * norm.cdf(bot + 1 - sigma)),
                (hi, sup, norm.sf(top - 1), mean * norm.sf(top - 1 - sigma)),
            ):
                self.assertAlmostEqual(
                    integrate(d.pdf, [a, b]) / tail_prob, 1.0, places=10
                )
                self.assertAlmostEqual(
                    integrate(lambda x: x * d.pdf(x), [a, b]) / tail_mean,
                    1.0,
                    places=10,
                )

    def test_the_lognormal_between_the_tails(self):
        for sigma, bot, top in self.cases:
            d = LogNormalTapered(mu=-0.2, sigma=sigma, bot=bot, top=top)
            ln = LogNormal(-0.2, sigma)
            # the cuts themselves are steps, so stay just inside them
            z = np.linspace(bot + 1.0, top - 1.0, 41)
            z[0], z[-1] = z[0] + 1e-9, z[-1] - 1e-9
            x = np.exp(-0.2 + sigma * z)
            np.testing.assert_allclose(d.pdf(x), ln.pdf(x), rtol=1e-13)
            np.testing.assert_allclose(d.cdf(x), ln.cdf(x), rtol=1e-13, atol=1e-16)

    def test_the_step_and_zero_at_the_bounds(self):
        for sigma, bot, top in self.cases:
            d = LogNormalTapered(sigma=sigma, bot=bot, top=top)
            inf, lo, hi, sup = self.pieces(d)
            # at the start of each tapered tail, the density of Z steps from the
            # normal's to the taper's height, (eta + 1) times the tail's probability
            for cut, tail, inside, outside in (
                (lo, d._law.lower, 1 + 1e-12, 1 - 1e-12),
                (hi, d._law.upper, 1 - 1e-12, 1 + 1e-12),
            ):
                step = d.pdf(cut * outside) / d.pdf(cut * inside)
                self.assertAlmostEqual(step, tail.height / norm.pdf(tail.cut), places=8)
                self.assertAlmostEqual(
                    tail.height, (tail.eta + 1) * norm.sf(abs(tail.cut)), places=15
                )
            self.assertEqual(d.pdf(d.infimum[0] * (1 - 1e-12)), 0.0)
            self.assertEqual(d.pdf(d.supremum[0] * (1 + 1e-12)), 0.0)
            # and it goes to zero as the bounds approach
            for bound, cut, sign in ((d.infimum[0], lo, 1), (d.supremum[0], hi, -1)):
                near = [d.pdf(bound * (1 + sign * eps)) for eps in (1e-4, 1e-8, 1e-12)]
                self.assertTrue(near[0] > near[1] > near[2] > 0.0)
                self.assertLess(near[2], 1e-3 * d.pdf(cut))
            np.testing.assert_allclose(d.infimum, [np.exp(sigma * bot)], rtol=1e-15)
            np.testing.assert_allclose(d.supremum, [np.exp(sigma * top)], rtol=1e-15)

    def test_the_density_falls_toward_each_bound(self):
        for sigma, bot, top in self.cases:
            d = LogNormalTapered(sigma=sigma, bot=bot, top=top)
            inf, lo, hi, sup = self.pieces(d)
            upper = d.pdf(np.linspace(hi, sup, 400))
            lower = d.pdf(np.linspace(lo, inf, 400))
            self.assertTrue(np.all(np.diff(upper) < 0.0))
            self.assertTrue(np.all(np.diff(lower) < 0.0))

    def test_quantiles_invert_the_cdf(self):
        q = np.array([1e-9, 1e-4, 0.01, 0.02, 0.2, 0.5, 0.8, 0.98, 0.99, 1 - 1e-6])
        for sigma, bot, top in self.cases:
            d = LogNormalTapered(sigma=sigma, bot=bot, top=top)
            x = d.ppf(q)
            self.assertTrue(np.all(np.diff(x) > 0.0))
            np.testing.assert_allclose(d.cdf(x), q, rtol=1e-9, atol=1e-15)
            self.assertEqual(d.ppf(0.0), d.infimum[0])
            self.assertEqual(d.ppf(1.0), d.supremum[0])

    def test_equiprobable_discretization(self):
        for sigma, bot, top in self.cases:
            d = LogNormalTaperedMeanOne(sigma=sigma, bot=bot, top=top)
            for n in (1, 2, 7, 25):
                dd = d.discretize(n)
                atoms = dd.atoms[0]
                np.testing.assert_allclose(dd.pmv, np.ones(n) / n)
                self.assertAlmostEqual(np.dot(dd.pmv, atoms), 1.0, places=13)
                self.assertTrue(np.all(np.diff(atoms) > 0.0))
                self.assertGreater(atoms[0], d.infimum[0])
                self.assertLess(atoms[-1], d.supremum[0])
            ends = d.discretize(7, endpoints=True)
            self.assertEqual(ends.atoms[0][0], d.infimum[0])
            self.assertEqual(ends.atoms[0][-1], d.supremum[0])
            self.assertIs(ends.limit["dist"], d)

    def test_draws(self):
        d = LogNormalTaperedMeanOne(sigma=0.2, seed=11)
        x = d.draw(200000)
        self.assertGreaterEqual(x.min(), d.infimum[0])
        self.assertLessEqual(x.max(), d.supremum[0])
        self.assertAlmostEqual(x.mean(), 1.0, delta=4 * 0.2 / np.sqrt(200000))

    def test_mean_one_and_the_sd_of_the_log(self):
        for sd in (0.05, 0.1, 0.3):
            d = LogNormalTaperedMeanOne.from_sd_log(sd)
            self.assertAlmostEqual(d.mean(), 1.0, places=13)
            pieces = self.pieces(d)
            m1 = integrate(lambda x: np.log(x) * d.pdf(x), pieces)
            m2 = integrate(lambda x: np.log(x) ** 2 * d.pdf(x), pieces)
            self.assertAlmostEqual(np.sqrt(m2 - m1**2) / sd, 1.0, places=9)
            self.assertGreater(d.sigma, sd)

    def test_one_side_untapered(self):
        d = LogNormalTapered(sigma=0.3, bot=None, top=3.0)
        ln = LogNormal(0.0, 0.3)
        x = np.exp(0.3 * np.array([-6.0, -3.0, 0.0, 1.9]))
        np.testing.assert_allclose(d.pdf(x), ln.pdf(x), rtol=1e-13)
        self.assertEqual(d.infimum[0], 0.0)
        self.assertAlmostEqual(d.mean() / ln.mean(), 1.0, places=12)
        u = LogNormalTapered(mu=0.1, sigma=0.3, bot=None, top=None)
        np.testing.assert_allclose(u.pdf(x), LogNormal(0.1, 0.3).pdf(x), rtol=1e-13)

    def test_point_and_refusals(self):
        point = LogNormalTaperedMeanOne(sigma=0.0)
        self.assertIsInstance(point, DiscreteDistribution)
        np.testing.assert_allclose(point.atoms, [[1.0]])
        with self.assertRaises(ValueError):
            LogNormalTapered(sigma=0.1, bot=-0.5, top=1.0)  # less than two apart
        with self.assertRaises(ValueError):
            LogNormalTapered(sigma=0.1, bot=-1.0, top=1.0)  # tails start at the median
        with self.assertRaises(ValueError):
            # the lower tail's mean is too far out for a taper over one sd
            LogNormalTapered(sigma=0.1, bot=-2.0, top=3.0)
        LogNormalTapered(sigma=0.1, bot=-2.5, top=3.0)
        LogNormalTapered(sigma=0.8, bot=-3.0, top=3.0)
        with self.assertRaises(ValueError):
            LogNormalTapered(sigma=0.81, bot=-3.0, top=3.0)  # its lower tail would rise
        LogNormalTapered(sigma=1.27, bot=-3.5, top=3.5)  # farther out, both fall
        with self.assertRaises(ValueError):
            LogNormalTapered(sigma=1.28, bot=-3.5, top=3.5)
        with self.assertRaises(ValueError):
            LogNormalTapered(sigma=-0.1)

    def test_the_tail_against_its_formula(self):
        # Past its start c, the density of Z is (eta + 1) P (1 - t)**eta, so the
        # distance past c has the distribution Beta(1, eta + 1) within the tail
        for sigma in (0.1, 0.5):
            d = LogNormalTapered(sigma=sigma)
            for tail in (d._law.upper, d._law.lower):
                t = np.array([1e-9, 0.1, 0.5, 0.9, 1.0])  # the cut itself is a step
                z = tail.cut + tail.outward * t
                prob = norm.sf(abs(tail.cut))
                expected = (tail.eta + 1) * prob * (1 - t) ** tail.eta
                np.testing.assert_allclose(d._law.pdf(z), expected, rtol=1e-13)
                beyond = d._law.cdf(z) if tail.outward < 0 else 1 - d._law.cdf(z)
                np.testing.assert_allclose(
                    beyond, prob * beta.sf(t, 1, tail.eta + 1), rtol=1e-12, atol=1e-18
                )

    def test_the_taper_moment_against_quadrature(self):
        for eta in (0.3, 0.7, 2.5):
            for k in (-1.0, -0.1, 0.1, 1.0):
                for s in (0.25, 1.0):
                    ref = quad(
                        lambda t: (eta + 1) * (1 - t) ** eta * np.exp(k * t),
                        1 - s,
                        1,
                        epsabs=0.0,
                        epsrel=1e-13,
                    )[0]
                    self.assertAlmostEqual(_taper_moment(eta, k, s) / ref, 1.0, 13)

import unittest

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from HARK.distributions import (
    DiscreteDistribution,
    LogNormal,
    LogNormalTruncated,
    LogNormalTruncatedJumpless,
    LogNormalTruncatedJumplessMeanOne,
)
from HARK.distributions.continuous import _jumpless_lognorm, _jumpless_normal


def integrate(fn, pieces):
    """The integral of fn over consecutive pieces, each by adaptive quadrature."""
    return sum(
        quad(fn, a, b, epsabs=1e-15, epsrel=1e-13, limit=400)[0]
        for a, b in zip(pieces[:-1], pieces[1:])
    )


class LogNormalTruncatedJumplessTests(unittest.TestCase):
    """
    The jumpless truncated lognormal: truncated below, its density zero at the
    bound and rising linearly from it, the lognormal's own away from the bound,
    with no upper bound.
    """

    cases = [(0.1, -3.0), (0.2, -2.0), (0.5, -3.5), (1.0, -2.5)]

    @staticmethod
    def pieces(d):
        # from the bound up, split where the density changes character
        z = [d.bot, d.bot + 0.5, d.bot + 2.0, 0.0, 3.0, 8.0, 14.0]
        return [np.exp(d.mu + d.sigma * v) for v in z]

    def test_probability_and_mean(self):
        for sigma, bot in self.cases:
            d = LogNormalTruncatedJumpless(mu=0.3, sigma=sigma, bot=bot)
            pieces = self.pieces(d)
            self.assertAlmostEqual(integrate(d.pdf, pieces), 1.0, places=12)
            mean = integrate(lambda x: x * d.pdf(x), pieces)
            self.assertAlmostEqual(d.mean() / mean, 1.0, places=11)
            self.assertAlmostEqual(d.moment(1) / mean, 1.0, places=11)
            # in closed form, and a little above the lognormal's
            b = -bot
            closed = (
                np.exp(0.3 + 0.5 * sigma**2)
                * (norm.cdf(b + sigma) - np.exp(-2 * b * sigma) * norm.cdf(sigma - b))
                / (norm.cdf(b) - norm.cdf(-b))
            )
            self.assertAlmostEqual(d.moment(1) / closed, 1.0, places=13)
            self.assertGreater(d.moment(1), np.exp(0.3 + 0.5 * sigma**2))

    def test_the_density_is_the_normals_less_its_reflection(self):
        for sigma, bot in self.cases:
            d = LogNormalTruncatedJumpless(mu=-0.2, sigma=sigma, bot=bot)
            z = bot + np.r_[np.geomspace(1e-12, 1.0, 40), np.linspace(1.0, 9.0, 30)]
            dist = z - bot  # the distance from the bound that z holds
            x = np.exp(d.mu + sigma * z)
            # phi(z) - phi(2 bot - z) = phi(z) (1 - exp(2 bot (z - bot)))
            expected = norm.pdf(z) * -np.expm1(2.0 * bot * dist) / d.mass
            np.testing.assert_allclose(d._law.pdf(z), expected, rtol=1e-12)
            # x recovers z to rounding, so close to the bound its distance from
            # the bound, and the density with it, is known only to that rounding
            np.testing.assert_allclose(
                d.pdf(x) * sigma * x, expected, rtol=1e-9, atol=4e-16
            )
            self.assertAlmostEqual(d.mass, 1.0 - 2.0 * norm.cdf(bot), places=15)

    def test_no_jump_at_the_bound(self):
        for sigma, bot in self.cases:
            d = LogNormalTruncatedJumpless(sigma=sigma, bot=bot)
            lo = d.infimum[0]
            self.assertEqual(d.pdf(lo), 0.0)
            self.assertEqual(d.pdf(0.5 * lo), 0.0)
            self.assertEqual(d.cdf(lo), 0.0)
            self.assertEqual(d.sf(lo), 1.0)
            # the density rises linearly from zero, where the truncated
            # lognormal's jumps to a positive value
            slope = -2.0 * bot * norm.pdf(bot) / d.mass / (sigma * lo) ** 2
            for step in (1e-6, 1e-5):
                h = lo * step
                self.assertAlmostEqual(d.pdf(lo + h) / (slope * h), 1.0, places=3)
            jump = LogNormalTruncated(sigma=sigma, bot=bot, top=None)
            self.assertGreater(jump.pdf(jump.infimum[0] * (1 + 1e-12)), 0.0)
            self.assertEqual(d.infimum[0], jump.infimum[0])
            self.assertEqual(d.supremum[0], np.inf)

    def test_the_upper_tail_is_the_lognormals(self):
        for sigma, bot in self.cases:
            d = LogNormalTruncatedJumpless(sigma=sigma, bot=bot)
            ln = LogNormal(0.0, sigma)
            for z in (0.0, 2.0, 5.0):
                x = np.exp(sigma * z)
                # all of the lognormal's density but the reflection's share of it
                share = -np.expm1(2.0 * bot * (z - bot))
                self.assertAlmostEqual(
                    d.pdf(x) * d.mass / (ln.pdf(x) * share), 1.0, places=12
                )
            x = np.exp(sigma * 6.0)
            self.assertAlmostEqual(d.sf(x) * d.mass / ln.sf(x), 1.0, places=9)

    def test_cdf_against_quadrature(self):
        for sigma, bot in self.cases:
            d = LogNormalTruncatedJumpless(sigma=sigma, bot=bot)
            law = d._law
            b = -bot

            def density(s):  # of the distance s above the bound, in z
                return 2.0 * norm.pdf(b) * np.exp(-0.5 * s * s) * np.sinh(b * s)

            # on both sides of where the series hands over to the closed form
            for dist in (
                1e-9,
                1e-6,
                1e-3,
                0.5 * law.near,
                0.999 * law.near,
                1.001 * law.near,
                2 * law.near,
                1.0,
                3.0,
            ):
                z = bot + dist
                held = z - bot  # the distance z holds, to its rounding
                ref = quad(density, 0.0, held, epsabs=0.0, epsrel=1e-13)[0] / law.mass
                self.assertAlmostEqual(law.cdf(z) / ref, 1.0, places=10)
            z = np.array([bot + 0.3, -1.0, 0.0, 1.5, 4.0])
            np.testing.assert_allclose(law.cdf(z) + law.sf(z), 1.0, rtol=0, atol=2e-16)
            # the probability approaches the bound with the square of the distance
            near = law.cdf(bot + np.array([1e-5, 2e-5]))
            self.assertAlmostEqual(near[1] / near[0], 4.0, places=8)

    def test_quantiles_invert_the_cdf(self):
        q = np.array([1e-12, 1e-9, 1e-4, 0.01, 0.2, 0.5, 0.8, 0.99, 1 - 1e-6])
        for sigma, bot in self.cases:
            d = LogNormalTruncatedJumpless(sigma=sigma, bot=bot)
            x = d.ppf(q)
            self.assertTrue(np.all(np.diff(x) > 0.0))
            np.testing.assert_allclose(d.cdf(x)[2:], q[2:], rtol=1e-12, atol=1e-15)
            # close to the bound a value holds its distance from it only to rounding
            np.testing.assert_allclose(d._law.cdf(d._law.ppf(q)), q, rtol=1e-7)
            np.testing.assert_allclose(d.sf(x), 1.0 - q, rtol=1e-9, atol=1e-15)
            self.assertEqual(d.ppf(0.0), d.infimum[0])
            self.assertEqual(d.ppf(1.0), np.inf)
            self.assertTrue(np.isnan(d.ppf(-0.1)) and np.isnan(d.ppf(1.1)))

    def test_moments_of_the_log(self):
        for bot in (-0.5, -2.0, -3.0, -4.5):
            law = _jumpless_normal(bot)
            top = 12.0
            mass = integrate(law.pdf, [bot, bot + 1.0, 0.0, 4.0, top])
            mean = integrate(lambda z: z * law.pdf(z), [bot, bot + 1.0, 0.0, 4.0, top])
            second = integrate(
                lambda z: z * z * law.pdf(z), [bot, bot + 1.0, 0.0, 4.0, top]
            )
            self.assertAlmostEqual(mass, 1.0, places=12)
            self.assertAlmostEqual(law.mean(), mean, places=12)
            self.assertAlmostEqual(law.variance(), second - mean**2, places=11)
        # at three standard deviations: a mean of 0.0081 and a variance of 0.9779
        law = _jumpless_normal(-3.0)
        self.assertAlmostEqual(law.mean(), 0.008121, places=6)
        self.assertAlmostEqual(law.variance(), 0.977869, places=6)

    def test_equiprobable_discretization(self):
        for sigma, bot in self.cases:
            d = LogNormalTruncatedJumplessMeanOne(sigma=sigma, bot=bot)
            for n in (1, 2, 7, 25):
                dd = d.discretize(n)
                atoms = dd.atoms[0]
                np.testing.assert_allclose(dd.pmv, np.ones(n) / n)
                self.assertAlmostEqual(np.dot(dd.pmv, atoms), 1.0, places=13)
                self.assertTrue(np.all(np.diff(atoms) > 0.0))
                self.assertGreater(atoms[0], d.infimum[0])
            ends = d.discretize(7, endpoints=True)
            self.assertEqual(ends.atoms[0][0], d.infimum[0])
            self.assertEqual(ends.atoms[0][-1], np.inf)
            self.assertIs(ends.limit["dist"], d)
            self.assertEqual(d.discretize(7).limit["infimum"][0], d.infimum[0])

    def test_draws(self):
        d = LogNormalTruncatedJumplessMeanOne(sigma=0.2, seed=11)
        x = d.draw(200000)
        self.assertGreaterEqual(x.min(), d.infimum[0])
        self.assertAlmostEqual(x.mean(), 1.0, delta=4 * 0.2 / np.sqrt(200000))

    def test_mean_one_and_the_sd_of_the_log(self):
        for sd in (0.05, 0.1, 0.3):
            for bot in (-2.0, -3.0):
                d = LogNormalTruncatedJumplessMeanOne.from_sd_log(sd, bot=bot)
                self.assertAlmostEqual(d.mean(), 1.0, places=13)
                pieces = self.pieces(d)
                m1 = integrate(lambda x: np.log(x) * d.pdf(x), pieces)
                m2 = integrate(lambda x: np.log(x) ** 2 * d.pdf(x), pieces)
                self.assertAlmostEqual(np.sqrt(m2 - m1**2) / sd, 1.0, places=9)
                self.assertGreater(d.sigma, sd)
        # HARK's default permanent shock: a lower bound of 0.7340
        d = LogNormalTruncatedJumplessMeanOne.from_sd_log(0.1)
        self.assertAlmostEqual(d.infimum[0], 0.734030, places=6)
        self.assertAlmostEqual(d.sigma, 0.10112525, places=8)

    def test_the_bound_its_sign_and_refusals(self):
        # the bound lies below the mean, so its distance can be given unsigned
        a = LogNormalTruncatedJumpless(mu=0.1, sigma=0.3, bot=-2.0)
        b = LogNormalTruncatedJumpless(mu=0.1, sigma=0.3, bot=2.0)
        self.assertEqual(a.bot, -2.0)
        self.assertEqual(b.bot, -2.0)
        self.assertEqual(a.infimum[0], b.infimum[0])
        self.assertEqual(a.infimum[0], np.exp(0.1 - 2.0 * 0.3))
        for bad in (0.0, None, np.inf, -np.inf, np.nan):
            with self.assertRaises(ValueError):
                LogNormalTruncatedJumpless(sigma=0.3, bot=bad)
            with self.assertRaises(ValueError):
                LogNormalTruncatedJumplessMeanOne(sigma=0.3, bot=bad)
        with self.assertRaises(ValueError):
            LogNormalTruncatedJumpless(sigma=-0.1)
        with self.assertRaises(ValueError):
            LogNormalTruncatedJumpless(mu=[0.0, 0.1], sigma=0.3)
        point = LogNormalTruncatedJumpless(mu=0.4, sigma=0.0)
        self.assertIsInstance(point, DiscreteDistribution)
        self.assertEqual(point.atoms[0][0], np.exp(0.4))
        one = LogNormalTruncatedJumplessMeanOne(sigma=0.0)
        self.assertEqual(one.atoms[0][0], 1.0)

    def test_its_own_functions_match_scipys(self):
        # The density, CDF, survival and quantile functions the law computes
        # from the law of its log are those of scipy's generic path through
        # _jumpless_lognorm, inside and outside the support, to rounding
        for sigma, bot in self.cases:
            d = LogNormalTruncatedJumpless(mu=0.3, sigma=sigma, bot=bot)
            lo = d.infimum[0]
            x = np.r_[0.5 * lo, lo, lo * np.exp(sigma * np.linspace(1e-3, 8.0, 25))]
            args = dict(s=sigma, lo=bot, scale=np.exp(0.3))
            # (at the bound itself scipy's path sees its rounding)
            np.testing.assert_allclose(
                d.pdf(x), _jumpless_lognorm.pdf(x, **args), rtol=1e-12, atol=1e-14
            )
            np.testing.assert_allclose(
                d.cdf(x), _jumpless_lognorm.cdf(x, **args), rtol=1e-12, atol=1e-14
            )
            np.testing.assert_allclose(
                d.sf(x), _jumpless_lognorm.sf(x, **args), rtol=1e-12, atol=1e-14
            )
            q = np.array([0.0, 1e-6, 0.3, 0.9, 1.0])
            np.testing.assert_allclose(
                d.ppf(q), _jumpless_lognorm.ppf(q, **args), rtol=1e-12
            )
            self.assertAlmostEqual(d.var() / (d.moment(2) - d.moment(1) ** 2), 1.0, 10)


if __name__ == "__main__":
    unittest.main()

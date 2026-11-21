import unittest

import numpy as np

from HARK.distributions import (
    LogNormalMeanOne,
    LogNormalTruncatedMeanOne,
    PointMassAtInfimum,
)


class PointMassAtInfimumTests(unittest.TestCase):
    """A base law with a point mass at its infimum, the mean kept at one."""

    def setUp(self):
        self.base = LogNormalTruncatedMeanOne.from_sd_log(0.1, bot=-3.0, top=3.0)
        self.prb = 0.01
        self.law = PointMassAtInfimum(self.base, self.prb)

    def test_bounds_and_scale(self):
        inf, sup = self.base.infimum[0], self.base.supremum[0]
        self.assertEqual(self.law.infimum[0], inf)
        self.assertEqual(self.law.infimum_prb, self.prb)
        scale = (1.0 - self.prb * inf) / (1.0 - self.prb)
        self.assertAlmostEqual(self.law.scale, scale, places=15)
        self.assertAlmostEqual(self.law.supremum[0], scale * sup, places=14)

    def test_mean_one(self):
        self.assertAlmostEqual(self.law.mean(), 1.0, places=12)
        self.assertAlmostEqual(self.law.expect(lambda x: x), 1.0, places=12)

    def test_cdf_and_quantiles(self):
        inf = self.law.infimum[0]
        self.assertEqual(self.law.cdf(inf * (1.0 - 1e-12)), 0.0)
        self.assertAlmostEqual(self.law.cdf(inf), self.prb, places=15)
        self.assertEqual(self.law.ppf(self.prb / 2.0), inf)
        q = 0.5
        expected = self.law.scale * self.base.ppf((q - self.prb) / (1.0 - self.prb))
        self.assertAlmostEqual(float(self.law.ppf(q)), float(expected), places=14)
        x = float(self.law.ppf(0.3))
        self.assertAlmostEqual(float(self.law.cdf(x)), 0.3, places=12)

    def test_discretization_keeps_the_point(self):
        for n in (3, 7, 25):
            approx = self.law.discretize(n, method="equiprobable")
            atoms, pmv = np.ravel(approx.atoms), np.asarray(approx.pmv)
            self.assertEqual(len(atoms), n + 1)
            self.assertEqual(atoms[0], self.law.infimum[0])
            self.assertEqual(pmv[0], self.prb)
            self.assertAlmostEqual(pmv.sum(), 1.0, places=14)
            self.assertAlmostEqual(np.dot(pmv, atoms), 1.0, places=13)
            self.assertEqual(approx.limit["infimum"][0], self.law.infimum[0])
            self.assertEqual(approx.limit["supremum"][0], self.law.supremum[0])

    def test_draws(self):
        draws = self.law.rvs(size=20000, random_state=np.random.default_rng(0))
        self.assertTrue(np.all(draws >= self.law.infimum[0]))
        self.assertAlmostEqual(
            np.mean(draws == self.law.infimum[0]), self.prb, delta=0.003
        )

    def test_refusals(self):
        for prb in (0.0, 1.0, 1.5, -0.1):
            with self.assertRaises(ValueError):
                PointMassAtInfimum(self.base, prb)
        with self.assertRaises(ValueError):
            PointMassAtInfimum(LogNormalMeanOne(0.1), 0.01)  # infimum zero


if __name__ == "__main__":
    unittest.main()

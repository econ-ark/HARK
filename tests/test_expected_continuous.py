"""
Tests for expected_continuous(): the expectation of a function over independent
distributions, by Gauss-Legendre quadrature in pieces, cut at each unit step of a
law's standard-normal coordinate and where the function is not smooth, against
closed forms and against scipy's adaptive quadrature.
"""

import unittest

import numpy as np
from scipy.integrate import quad
from scipy.stats import gamma

from HARK.distributions import (
    DiscreteDistribution,
    DiscreteDistributionLabeled,
    LogNormalMeanOne,
    LogNormalTruncatedMeanOne,
    Normal,
    Uniform,
    Weibull,
    combine_indep_dstns,
    expected,
    expected_continuous,
)
from HARK.distributions.utils import _coordinate


class Mixture:
    """A mixture as expected_continuous reads one: weights and components."""

    def __init__(self, weights, components):
        self.weights = weights
        self.components = components


class OwnCoordinate:
    """A law known only through the standard-normal coordinate it supplies."""

    def __init__(self, law):
        self._law = law

    def normal_coordinate(self):
        return self._law.normal_coordinate()


class QuantileOnly:
    """A law known only through its quantile function and CDF."""

    def __init__(self, law):
        self._law = law

    def ppf(self, q):
        return self._law.ppf(q)

    def cdf(self, x):
        return self._law.cdf(x)


def reference(f, law, points=None):
    """E[f(X)] by adaptive quadrature against the law's density."""
    lo, hi = law.ppf(0.0), law.ppf(1.0)
    lo = law.ppf(1e-16) if not np.isfinite(lo) else lo
    hi = law.ppf(1.0 - 1e-16) if not np.isfinite(hi) else hi
    inside = None if points is None else [p for p in points if lo < p < hi] or None
    return quad(
        lambda x: f(x) * law.pdf(x), lo, hi, points=inside, epsabs=1e-14, limit=500
    )[0]


class testExpectedContinuous(unittest.TestCase):
    def test_means_and_mass(self):
        # to 5e-9 at the default nodes, four a piece, and to rounding at eight
        laws = [
            LogNormalMeanOne(0.1),
            LogNormalMeanOne(0.3),
            LogNormalTruncatedMeanOne.from_sd_log(0.1, -3.0, 3.0),
            LogNormalTruncatedMeanOne.from_sd_log(0.2, -2.0, None),
            Normal(1.5, 0.3),
            Uniform(0.2, 1.7),
            Weibull(1.0, 2.0),
            Weibull(1.0, 0.7),
        ]
        for law in laws:
            for nodes, places in ((4, 8), (8, 12)):
                with self.subTest(law=type(law).__name__, nodes=nodes):
                    mean = expected_continuous(dstns=law, nodes=nodes)
                    self.assertEqual(mean.shape, (1,))
                    self.assertAlmostEqual(
                        mean[0] / float(law.mean()), 1.0, places=places
                    )
                    mass = expected_continuous(
                        lambda x: np.ones_like(x[0]), law, nodes=nodes
                    )
                    self.assertAlmostEqual(float(mass), 1.0, places=places)

    def test_kinked_integrand_split_at_the_kink(self):
        law = LogNormalMeanOne(0.2)
        k = 1.05

        def payoff(x):
            return np.maximum(x[0] - k, 0.0)

        exact = reference(lambda x: max(x - k, 0.0), law, points=[k])
        for nodes, places in ((4, 7), (8, 12)):
            split = expected_continuous(
                payoff, law, splits=[lambda: np.array([k])], nodes=nodes
            )
            self.assertAlmostEqual(float(split), exact, places=places)
            unsplit = expected_continuous(payoff, law, nodes=nodes)
            self.assertGreater(abs(float(unsplit) - exact), 1e-6)

    def test_split_moving_with_the_arguments(self):
        law = LogNormalTruncatedMeanOne.from_sd_log(0.2, -3.0, 3.0)
        k = np.array([0.5, 0.9, 1.0, 1.2, 2.5])  # some outside the support
        result = expected_continuous(
            lambda x, k: np.maximum(x[0] - k, 0.0),
            law,
            args=(k,),
            splits=[lambda k: k[..., None]],
            nodes=8,
        )
        self.assertEqual(result.shape, k.shape)
        for i, ki in enumerate(k):
            exact = reference(lambda x: max(x - ki, 0.0), law, points=[ki])
            self.assertAlmostEqual(result[i], exact, places=12)

    def test_two_laws_with_splits_in_both(self):
        # The integrand has a kink where b / psi + theta = m_K. The theta
        # integral is split there, and the psi integral where that kink reaches
        # either end of theta's support.
        psi = LogNormalTruncatedMeanOne.from_sd_log(0.1, -3.0, 3.0)
        theta = LogNormalTruncatedMeanOne.from_sd_log(0.15, -3.0, 3.0)
        m_K = 1.0
        b = np.array([0.02, 0.1, 0.2, 0.3])

        def func(S, b):
            return np.maximum(m_K - (b / S["PermShk"] + S["TranShk"]), 0.0) ** 2

        def psi_cuts(b):
            ends = np.array([theta.infimum[0], theta.supremum[0]])
            with np.errstate(divide="ignore"):
                return b[..., None] / (m_K - ends)

        def theta_cuts(p, b):
            return (m_K - b / p)[..., None]

        result = expected_continuous(
            func,
            [psi, theta],
            args=(b,),
            labels=["PermShk", "TranShk"],
            splits=[psi_cuts, theta_cuts],
            nodes=[8, 8],
        )
        lo_t, hi_t = theta.infimum[0], theta.supremum[0]
        for i, bi in enumerate(b):

            def given_psi(p):
                cut = m_K - bi / p
                pts = [cut] if lo_t < cut < hi_t else None
                return quad(
                    lambda t: max(m_K - (bi / p + t), 0.0) ** 2 * theta.pdf(t),
                    lo_t,
                    hi_t,
                    points=pts,
                    epsabs=1e-15,
                    limit=500,
                )[0]

            pts = [bi / (m_K - e) for e in (lo_t, hi_t) if m_K > e]
            exact = reference(given_psi, psi, points=pts)
            self.assertAlmostEqual(result[i] / exact, 1.0, places=10)

    def test_discrete_laws_match_expected(self):
        # With discrete laws it is expected() over their independent combination
        perm = LogNormalMeanOne(0.1).discretize(7, method="equiprobable")
        tran = LogNormalMeanOne(0.1).discretize(5, method="equiprobable")
        joint = combine_indep_dstns(perm, tran)
        labeled = DiscreteDistributionLabeled.from_unlabeled(
            joint, var_names=["PermShk", "TranShk"]
        )
        a = np.linspace(0.0, 3.0, 11)

        def vp(S, a):
            return (a / S["PermShk"] + S["TranShk"]) ** -2.0

        np.testing.assert_allclose(
            expected_continuous(
                vp, [perm, tran], args=(a,), labels=["PermShk", "TranShk"]
            ),
            expected(vp, labeled, args=(a,)),
            rtol=1e-14,
        )

    def test_mixture_component_by_component(self):
        law = LogNormalTruncatedMeanOne.from_sd_log(0.1, -3.0, 3.0)
        point = DiscreteDistribution([1.0], [0.3])
        mixture = Mixture([0.05, 0.95], [point, law])
        mean = expected_continuous(dstns=mixture, nodes=8)[0]
        self.assertAlmostEqual(mean, 0.05 * 0.3 + 0.95 * 1.0, places=12)
        k = 0.95
        split = expected_continuous(
            lambda x: np.maximum(x[0] - k, 0.0),
            mixture,
            splits=[lambda: np.array([k])],
            nodes=8,
        )
        exact = 0.95 * reference(lambda x: max(x - k, 0.0), law, points=[k])
        self.assertAlmostEqual(float(split), exact, places=12)

    def test_cuts_outside_the_support_change_nothing(self):
        law = LogNormalTruncatedMeanOne.from_sd_log(0.1, -3.0, 3.0)

        def f(x):
            return x[0] ** -3.0

        plain = expected_continuous(f, law)
        for cuts in ([0.1], [5.0], [np.nan], [-1.0, 7.0]):
            with self.subTest(cuts=cuts):
                cut = expected_continuous(f, law, splits=[lambda: np.array(cuts)])
                self.assertAlmostEqual(float(cut), float(plain), places=14)

    def test_coordinate_and_its_steps(self):
        # A law's integral is cut at each unit step of its standard-normal
        # coordinate: the one it supplies through normal_coordinate(), as normal
        # and lognormal laws do, or else its normal score, cut at eight
        steps = list(range(-8, 9))
        cases = {
            "truncated": (
                LogNormalTruncatedMeanOne.from_sd_log(0.1, -3.0, 3.0),
                [-3, -2, -1, 0, 1, 2, 3],
            ),
            "truncated below": (
                LogNormalTruncatedMeanOne.from_sd_log(0.1, -2.5, None),
                [-2.5] + steps[6:],
            ),
            "lognormal": (LogNormalMeanOne(0.1), steps),
            "uniform": (Uniform(0.6, 1.4), steps),
        }
        for name, (law, edges) in cases.items():
            with self.subTest(law=name):
                self.assertEqual([float(e) for e in _coordinate(law)[0]], edges)
        # any law can supply its coordinate, and is integrated in it
        law = LogNormalTruncatedMeanOne.from_sd_log(0.1, -3.0, 3.0)
        own = OwnCoordinate(law)
        self.assertEqual([float(e) for e in _coordinate(own)[0]], cases["truncated"][1])
        for f in (lambda x: x[0], lambda x: x[0] ** -2):
            self.assertEqual(
                float(expected_continuous(f, own)), float(expected_continuous(f, law))
            )

    def test_own_coordinate_and_normal_score_agree(self):
        # Normal and lognormal laws, truncated or not, supply their own
        # standard-normal coordinate; known only through their quantile functions
        # and CDFs, the same laws are integrated in their normal score, cut at its
        # own unit steps, and give the same expectations to the rule's accuracy:
        # to 1e-7 at four nodes a piece, and to rounding at eight
        laws = [
            Normal(1.5, 0.3),
            LogNormalMeanOne(0.2),
            LogNormalTruncatedMeanOne.from_sd_log(0.2, -3.0, 3.0),
            LogNormalTruncatedMeanOne.from_sd_log(0.2, -2.0, None),
        ]
        for law in laws:
            generic = QuantileOnly(law)
            k = float(np.squeeze(law.mean()))

            def payoff(x):
                return np.maximum(x[0] - k, 0.0)

            for nodes, places in ((4, 7), (8, 12)):
                with self.subTest(law=type(law).__name__, nodes=nodes):
                    for f in (
                        lambda x: x[0],
                        lambda x: x[0] ** 2,
                        lambda x: np.exp(x[0]),
                    ):
                        self.assertAlmostEqual(
                            float(expected_continuous(f, generic, nodes=nodes)[()])
                            / float(expected_continuous(f, law, nodes=nodes)[()]),
                            1.0,
                            places=places,
                        )
                    kinked = [
                        float(
                            expected_continuous(
                                payoff, d, splits=[lambda: np.array([k])], nodes=nodes
                            )
                        )
                        for d in (law, generic)
                    ]
                    self.assertAlmostEqual(
                        kinked[1] / kinked[0], 1.0, places=places - 1
                    )

    def test_laws_that_are_not_lognormal(self):
        # a uniform, and a law known only by its quantile function and CDF, against
        # their moments and scipy's adaptive quadrature: in the coordinate reached
        # through the quantile function, to 1e-7 at four nodes a piece and to
        # rounding at eight
        k = 1.1
        cases = {
            "uniform": (Uniform(0.6, 1.4), Uniform(0.6, 1.4), 1.0 + 0.8**2 / 12.0),
            "quantile function": (
                QuantileOnly(gamma(3.0, scale=1.0 / 3.0)),  # a mean of one
                gamma(3.0, scale=1.0 / 3.0),
                4.0 / 3.0,
            ),
        }
        for name, (law, density, second_moment) in cases.items():
            exact = reference(lambda x: max(x - k, 0.0), density, points=[k])
            for nodes, tol in ((4, 1e-7), (8, 1e-12)):
                with self.subTest(law=name, nodes=nodes):
                    mean, second = expected_continuous(
                        lambda x: np.stack([x[0], x[0] ** 2]), law, nodes=nodes
                    )
                    self.assertAlmostEqual(float(mean), 1.0, delta=tol)
                    self.assertAlmostEqual(float(second), second_moment, delta=tol)
                    split = expected_continuous(
                        lambda x: np.maximum(x[0] - k, 0.0),
                        law,
                        splits=[lambda: np.array([k])],
                        nodes=nodes,
                    )
                    self.assertAlmostEqual(float(split), exact, delta=tol)

    def test_errors(self):
        law = LogNormalMeanOne(0.1)
        with self.assertRaises(TypeError):
            expected_continuous(dstns=object())
        with self.assertRaises(ValueError):
            expected_continuous(dstns=[law, law], nodes=[16])
        with self.assertRaises(ValueError):
            expected_continuous(dstns=[law], labels=["a", "b"])
        joint = combine_indep_dstns(
            law.discretize(3, method="equiprobable"),
            law.discretize(3, method="equiprobable"),
        )
        with self.assertRaises(ValueError):
            expected_continuous(dstns=joint)

"""
Tests for expected_continuous(): the expectation of a function over independent
distributions, by Gauss-Legendre quadrature in pieces, cut at each unit step of a
law's standard-normal coordinate and where the function is not smooth, against
closed forms and against scipy's adaptive quadrature.
"""

import unittest

import numpy as np
from scipy.integrate import quad
from scipy.stats import beta, gamma, norm, poisson

from HARK.distributions import (
    Bernoulli,
    DiscreteDistribution,
    DiscreteDistributionLabeled,
    LogNormalMeanOne,
    LogNormalTruncated,
    LogNormalTruncatedJumplessMeanOne,
    LogNormalTruncatedMeanOne,
    Normal,
    PointMassAtInfimum,
    Uniform,
    Weibull,
    combine_indep_dstns,
    end_probability,
    expected,
    expected_continuous,
    graded_quadrature,
    law_bounds,
    law_end,
    scaled_law,
    step_values,
)
from HARK.distributions.utils import STEPS_WITHIN, _coordinate, _ScaledLaw


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


class QuantileLaw:
    """A law known only through its quantile function, CDF, survival function and density."""

    def __init__(self, law):
        self._law = law

    def ppf(self, q):
        return self._law.ppf(q)

    def cdf(self, x):
        return self._law.cdf(x)

    def sf(self, x):
        return self._law.sf(x)

    def pdf(self, x):
        return self._law.pdf(x)


def log_distance_quad(f, lo, hi, pole):
    """
    The integral of f over [lo, hi] by adaptive quadrature in u = log(x - pole),
    in which a near-pole at pole, just below lo, is smooth.
    """

    def g(u):
        return f(pole + np.exp(u)) * np.exp(u)

    u_lo, u_hi = np.log(lo - pole), np.log(hi - pole)
    return quad(g, u_lo, u_hi, epsabs=0.0, epsrel=1e-12, limit=200)[0]


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

    def test_means_with_labels(self):
        # without a function it gives the means, with labels as without
        laws = [LogNormalMeanOne(0.1), Uniform(0.2, 1.7)]
        np.testing.assert_array_equal(
            expected_continuous(dstns=laws, labels=["a", "b"]),
            expected_continuous(dstns=laws),
        )

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

    def test_discrete_laws_that_are_not_discrete_distributions(self):
        # A Bernoulli holds its atoms and probabilities without being a
        # DiscreteDistribution, and is summed over them; a discrete law known
        # only through its quantile function and CDF is refused, since its
        # quantile function is a step function
        law = Bernoulli(0.3)
        self.assertEqual(float(expected_continuous(dstns=law)[0]), 0.3)
        self.assertAlmostEqual(
            float(expected_continuous(lambda x: np.exp(2.0 * x[0]), law)),
            0.7 + 0.3 * np.exp(2.0),
            places=14,
        )
        with self.assertRaises(TypeError):
            expected_continuous(dstns=poisson(3.0))

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

    def test_cuts_outside_the_support_in_some_rows(self):
        # Rows whose cut points fall outside the support, or are NaN, beside a
        # row cut inside it: each row's expectation is the one it has alone,
        # also where the integrand is infinite at an end of the support, since
        # no node lies on an end
        law = LogNormalTruncatedMeanOne.from_sd_log(0.2, -3.0, 3.0)
        lo, hi = law.infimum[0], law.supremum[0]
        k = np.array([0.9, 0.1, 5.0, np.nan])

        def cuts(k):
            return k[..., None]

        for name, f in (
            ("infinite at the infimum", lambda x, k: (x[0] - lo) ** -0.5),
            ("infinite at the supremum", lambda x, k: (hi - x[0]) ** -0.5),
        ):
            with self.subTest(integrand=name):
                together = expected_continuous(
                    f, law, args=(k,), splits=[cuts], nodes=8
                )
                self.assertTrue(np.all(np.isfinite(together)))
                alone = [
                    expected_continuous(
                        f, law, args=(k[i : i + 1],), splits=[cuts], nodes=8
                    )[0]
                    for i in range(k.size)
                ]
                np.testing.assert_allclose(together, alone, rtol=1e-14)
                uncut = float(expected_continuous(f, law, args=(0.0,), nodes=8))
                np.testing.assert_allclose(together[1:], uncut, rtol=1e-14)

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

    def test_breakpoints_are_edges(self):
        # A law's breakpoints are edges of its pieces in z, beside the unit
        # steps; here a truncated lognormal that declares one at z = 0.5
        class WithBreakpoint(LogNormalTruncatedMeanOne):
            def breakpoints(self):
                return np.exp(self.mu + self.sigma * np.array([0.5]))

        edges = np.array(_coordinate(WithBreakpoint(0.2, -3.0, 3.0))[0])
        self.assertLess(np.min(np.abs(edges - 0.5)), 1e-12)
        self.assertTrue(np.all(np.diff(edges) > 1e-9))
        # on the unit steps they add no edge, and most laws have none
        steps = np.array(_coordinate(LogNormalTruncatedMeanOne(0.2, -3.0, 3.0))[0])
        self.assertTrue(np.all(np.isin(np.round(steps, 12), np.arange(-8.0, 9.0))))
        self.assertEqual(LogNormalMeanOne(0.2).breakpoints().size, 0)
        self.assertEqual(LogNormalTruncatedMeanOne(0.2).breakpoints().size, 0)

    def test_point_mass_at_the_infimum(self):
        # A law with a point mass at its infimum is reached through its quantile
        # function, which jumps from the infimum to where the law's continuous
        # part starts. That value is the law's breakpoint, and so an edge of its
        # pieces: with the jump inside a piece instead, four nodes a piece miss
        # these moments by ten times as much or more
        base = LogNormalTruncatedMeanOne.from_sd_log(0.1, -3.0, 3.0)

        class JumpNotDeclared(PointMassAtInfimum):
            def breakpoints(self):
                return np.array([])

        for prb in (0.005, 0.05):
            law = PointMassAtInfimum(base, prb)
            np.testing.assert_allclose(law.breakpoints(), law.scale * law.infimum)
            edges = np.array(_coordinate(law)[0])
            self.assertLess(np.min(np.abs(edges - norm.ppf(prb))), 1e-12)
            for order in (1.0, -2.0):
                exact = law.expect(lambda x: x**order)

                def miss(law, nodes):
                    got = expected_continuous(lambda x: x[0] ** order, law, nodes=nodes)
                    return abs(float(got) / exact - 1.0)

                with self.subTest(prb=prb, order=order):
                    undeclared = JumpNotDeclared(base, prb)
                    self.assertLess(10.0 * miss(law, 4), miss(undeclared, 4))
                    self.assertLess(miss(law, 16), 5e-8)

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


class testLawTools(unittest.TestCase):
    """
    The tools behind expected_continuous() for laws in general: the ends of a
    law's support and how it holds probability there, laws scaled and mixed, the
    values at which a law's integral is cut, and the rule graded toward a
    near-pole beyond an end of the support.
    """

    def test_scaled_law_for_any_law(self):
        # a law of no family that rescales in closed form: its quantile function,
        # CDF, density and bounds, all scaled
        law = Uniform(0.6, 1.4)
        scaled = scaled_law(law, 1.05)
        self.assertIsInstance(scaled, _ScaledLaw)
        self.assertAlmostEqual(scaled.mean(), 1.05, places=14)
        np.testing.assert_allclose(law_bounds(scaled), (0.63, 1.47), rtol=1e-14)
        q = np.linspace(0.01, 0.99, 7)
        np.testing.assert_allclose(scaled.cdf(scaled.ppf(q)), q, rtol=1e-12)
        x = np.linspace(0.64, 1.46, 9)
        np.testing.assert_allclose(scaled.pdf(x), 1.0 / 0.84, rtol=1e-12)
        # a truncated lognormal is scaled the same way, as the same family with its
        # mu moved would be, and keeps the coordinate it supplies, and so the
        # pieces its integral is cut into
        tl = LogNormalTruncatedMeanOne.from_sd_log(0.1, -3.0, 3.0)
        scaled = scaled_law(tl, 1.05)
        self.assertIsInstance(scaled, _ScaledLaw)
        family = LogNormalTruncated(tl.mu + np.log(1.05), tl.sigma, bot=-3.0, top=3.0)
        np.testing.assert_allclose(scaled.ppf(q), family.ppf(q), rtol=1e-12)
        x = family.ppf(q)
        for f in ("cdf", "pdf"):
            np.testing.assert_allclose(
                getattr(scaled, f)(x), getattr(family, f)(x), rtol=1e-11
            )
        edges, value, _, _ = _coordinate(scaled)
        self.assertEqual([float(e) for e in edges], [-3, -2, -1, 0, 1, 2, 3])
        z = np.array(edges)  # the underlying normal's coordinate, scaled values
        np.testing.assert_allclose(value(z), np.exp(family.mu + family.sigma * z))
        self.assertFalse(
            hasattr(_ScaledLaw(QuantileLaw(tl), 1.05), "normal_coordinate")
        )
        # a discrete law's atoms are scaled, and a mixture component by component
        atoms = scaled_law(DiscreteDistribution([0.5, 0.5], [1.0, 2.0]), 1.05)
        np.testing.assert_allclose(atoms.atoms[0], [1.05, 2.1])
        mixture = scaled_law(Mixture([0.4, 0.6], [law, tl]), 1.05)
        self.assertEqual(len(mixture.components), 2)
        self.assertIsInstance(mixture.components[1], _ScaledLaw)
        self.assertAlmostEqual(float(mixture.infimum[0]), 0.63, places=14)
        self.assertIs(scaled_law(tl, 1.0), tl)

    def test_law_ends(self):
        # How a law holds probability at each end, read through its quantile
        # function and density alone: with a positive density (1) where it is
        # truncated or uniform, with one rising like a power from zero (2, 3) at
        # the jumpless law's bound and the ends of a beta law and a gamma law's
        # zero, not at all (None) where it reaches the end through a tail; a mass
        # (0) at a point
        truncated = LogNormalTruncatedMeanOne.from_sd_log(0.1, -3.0, 3.0)
        jumpless = LogNormalTruncatedJumplessMeanOne.from_sd_log(0.1)
        cases = {
            "truncated": (truncated, 1, 1),
            "truncated, wide": (LogNormalTruncated(0.0, 3.0, bot=-1.0, top=5.0), 1, 1),
            "truncated above": (
                LogNormalTruncated(0.0, 0.1, bot=None, top=3.0),
                None,
                1,
            ),
            "truncated below": (
                LogNormalTruncated(0.0, 0.1, bot=-3.0, top=None),
                1,
                None,
            ),
            "jumpless": (jumpless, 2, None),
            "lognormal": (LogNormalMeanOne(0.1), None, None),
            "lognormal, wide": (LogNormalMeanOne(2.0), None, None),
            "uniform": (Uniform(0.6, 1.4), 1, 1),
            "beta": (QuantileLaw(beta(2.0, 3.0, loc=0.4, scale=1.5)), 2, 3),
            "gamma": (QuantileLaw(gamma(3.0)), 3, None),
            "point": (DiscreteDistribution([1.0], [0.3]), 0, 0),
        }
        for name, (law, lower, upper) in cases.items():
            with self.subTest(law=name):
                self.assertEqual(law_end(law, upper=False)[1], lower)
                self.assertEqual(law_end(law, upper=True)[1], upper)

    def test_end_probability_and_step_values(self):
        # the probability a law holds at an end itself: an atom's, all of a
        # point's, none for a density; a mixture's components' that reach it
        three = DiscreteDistribution([0.2, 0.5, 0.3], [1.0, 2.0, 3.0])
        self.assertEqual(end_probability(three, upper=False), 0.2)
        self.assertEqual(end_probability(three, upper=True), 0.3)
        self.assertEqual(
            end_probability(DiscreteDistribution([1.0], [2.0]), False), 1.0
        )
        tl = LogNormalTruncatedMeanOne.from_sd_log(0.1, -3.0, 3.0)
        self.assertEqual(end_probability(tl, upper=False), 0.0)
        point = DiscreteDistribution([1.0], [0.3])
        self.assertAlmostEqual(
            end_probability(Mixture([0.4, 0.6], [point, tl]), False), 0.4
        )
        # the values at which a law's integral is cut: its coordinate's steps
        # within STEPS_WITHIN of its center, with the ends it holds probability
        # at; a discrete law's atoms
        z = np.arange(-STEPS_WITHIN, STEPS_WITHIN + 1.0)
        np.testing.assert_allclose(step_values(tl), np.exp(tl.mu + tl.sigma * z))
        np.testing.assert_array_equal(step_values(three), [1.0, 2.0, 3.0])

    def test_graded_rule_in_either_coordinate(self):
        # A truncated lognormal is graded in the standard-normal coordinate it
        # supplies; known only by its quantile function, CDF and density, the
        # same law is graded in its values, and both give the integrals of an
        # inverse distance to a pole beyond the lower end, with cuts or without
        law = LogNormalTruncatedMeanOne.from_sd_log(0.1, -3.0, 3.0)
        lo, hi = law.infimum[0], law.supremum[0]
        pole = lo - np.array([1e-6, 1e-3, 1e-2])
        exact = [
            log_distance_quad(lambda x, p=p: law.pdf(x) / (x - p), lo, hi, p)
            for p in pole
        ]
        cuts = np.broadcast_to(lo + np.array([1e-7, 1e-4, 0.01, 0.1]), (3, 4))
        for name, form in (("its coordinate", law), ("its values", QuantileLaw(law))):
            for nodes, rtol in ((4, 2e-8), (8, 2e-12)):
                for cut in (None, cuts):
                    with self.subTest(graded=name, nodes=nodes, cuts=cut is not None):
                        values, weights, graded = graded_quadrature(
                            form, nodes, pole, cut
                        )
                        self.assertTrue(np.all(graded))
                        E = np.sum(weights / (values - pole[:, None]), axis=-1)
                        np.testing.assert_allclose(E, exact, rtol=rtol)
        # a pole far from the support is not graded, and the rule is then the
        # plain one, the integral of the law's density to rounding
        values, weights, graded = graded_quadrature(law, 4, np.array([lo - 10.0]))
        self.assertFalse(np.any(graded))
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=8)

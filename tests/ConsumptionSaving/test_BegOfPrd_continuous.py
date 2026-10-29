"""
Tests for calc_BegOfPrd_continuous(), the expectations over next period's income
shocks against their continuous laws, and for the helpers that read the laws and
next period's kink. The references are independent of the rule's own split
points: nested adaptive quadrature (scipy's quad) for a policy smooth but for a
kink found by root-finding, and Gauss-Legendre split at every knot of a solved
policy as well as at its kink.
"""

import unittest
from unittest import mock

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import beta

from HARK.Calibration.Income import IncomeProcesses
from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsumerSolution,
    IndShockConsumerType,
    _bounds,
    _graded_quadrature,
    _scaled_law,
    _ScaledLaw,
    calc_BegOfPrd_continuous,
    get_income_shock_laws,
    get_mNrm_kinks,
)
from HARK.distributions import (
    DiscreteDistribution,
    DiscreteDistributionLabeled,
    LogNormalTruncatedJumplessMeanOne,
    LogNormalTruncatedMeanOne,
    LogNormalTruncated,
    Uniform,
    add_discrete_outcome_constant_mean,
    combine_indep_dstns,
)
from HARK.distributions.utils import _coordinate
from HARK.interpolation import HARKinterpolator1D, LinearInterp, LowerEnvelope

# Most of these tests check the expectation against exact references written for
# truncated lognormals, whose density is the normal's between the bounds: the
# income constructor's default treatment of the tails, at three standard
# deviations, which the module's setup fixes for these tests.
TRUNCATED = mock.patch.dict(
    IncomeProcesses._SHOCK_TAILS, {"truncated": LogNormalTruncatedMeanOne}
)
TRUNCATED_BOUND = mock.patch.dict(
    IncomeProcesses.SHOCK_BOUND_DEFAULT, {"truncated": 3.0}
)


def setUpModule():
    TRUNCATED.start()
    TRUNCATED_BOUND.start()


def tearDownModule():
    TRUNCATED_BOUND.stop()
    TRUNCATED.stop()


BOUNDS = {
    "PermShk_inf": -3.0,
    "PermShk_sup": 3.0,
    "TranShk_inf": -3.0,
    "TranShk_sup": 3.0,
    "PermShk_tails": "truncated",
    "TranShk_tails": "truncated",
}
# The same lower bounds with jumpless tails, which have no upper bound
JUMPLESS = {
    "PermShk_inf": -3.0,
    "TranShk_inf": -3.0,
    "PermShk_tails": "jumpless",
    "TranShk_tails": "jumpless",
}
X, W = leggauss(200)
X40, W40 = leggauss(40)


def pieces(lo, hi, cuts, X=X, W=W):
    """Gauss-Legendre nodes and weights on [lo, hi], split at the cuts inside it."""
    edges = np.sort(np.r_[lo, [c for c in cuts if lo < c < hi], hi])
    x = [0.5 * (b - a) * X + 0.5 * (a + b) for a, b in zip(edges[:-1], edges[1:])]
    w = [0.5 * (b - a) * W for a, b in zip(edges[:-1], edges[1:])]
    return np.concatenate(x), np.concatenate(w)


# The smooth policy below, c(m) = min(m, 0.25 + 0.5 m^0.7), and its kink
M_KINK = brentq(lambda m: m - (0.25 + 0.5 * m**0.7), 0.1, 2.0)


def adaptive_reference(a, perm, tran, R, G, rho):
    """
    E1 and E2 for the smooth policy by nested adaptive quadrature against the
    laws' densities, with breakpoints where m' reaches M_KINK.
    """
    point = isinstance(tran, DiscreteDistribution)
    lo_p, hi_p = _bounds(perm)
    # where a law's density is not smooth: its breakpoints()
    perm_breaks = list(perm.breakpoints()) if hasattr(perm, "breakpoints") else []

    def marginals(m, psi, k):
        c = m if m < M_KINK else 0.25 + 0.5 * m**0.7
        mpc = 1.0 if m < M_KINK else 0.35 * m**-0.3
        vP = c ** (-rho)
        return (
            psi ** (-rho) * vP if k == 0 else -rho * psi ** (-rho - 1.0) * vP * mpc / c
        )

    out = np.zeros((2, a.size))
    for i, ai in enumerate(a):
        base = R * ai / G
        if point:
            x = tran.atoms[0][0]
            ends = [x]
        else:
            lo_t, hi_t = _bounds(tran)
            tran_breaks = (
                list(tran.breakpoints()) if hasattr(tran, "breakpoints") else []
            )
            ends = [lo_t, hi_t] + tran_breaks
        breaks = [base / (M_KINK - e) for e in ends if M_KINK > e] + perm_breaks
        breaks = [b for b in breaks if lo_p < b < hi_p] or None
        for k in (0, 1):
            if point:

                def outer(psi):
                    return marginals(base / psi + x, psi, k) * perm.pdf(psi)

            else:

                def outer(psi):
                    cut = M_KINK - base / psi
                    inside = [t for t in [cut] + tran_breaks if lo_t < t < hi_t]
                    inside = inside or None
                    inner = quad(
                        lambda t: marginals(base / psi + t, psi, k) * tran.pdf(t),
                        lo_t,
                        hi_t,
                        points=inside,
                        epsabs=0.0,
                        epsrel=1e-13,
                        limit=200,
                    )[0]
                    return inner * perm.pdf(psi)

            out[k, i] = quad(
                outer, lo_p, hi_p, points=breaks, epsabs=0.0, epsrel=1e-12, limit=200
            )[0]
    return out


def knot_reference(solution, a, perm, tran, R, G, rho):
    """
    E1 and E2 by Gauss-Legendre in the shocks' values, 40 nodes a piece, split
    where m' reaches the solution's kink or any knot of its consumption function,
    so that the integrand is smooth on every piece.
    """
    unconstrained = solution.cFunc.functions[0]
    knots = np.asarray(unconstrained.x_list, dtype=float)
    knots = knots[knots > solution.mNrmMin]

    def above_the_line(m):
        """How far the unconstrained function lies above c = m - mNrmMin."""
        return unconstrained(np.array([m]))[0] - (m - solution.mNrmMin)

    if hasattr(solution, "mNrmKinks"):  # recorded, empty where none
        kinks = np.r_[solution.mNrmKinks, knots]
    elif above_the_line(knots[0]) * above_the_line(5.0) < 0.0:
        kinks = np.r_[brentq(above_the_line, knots[0], 5.0), knots]
    else:  # below the line from the lowest point on, as where the natural one binds
        kinks = knots
    point = isinstance(tran, DiscreteDistribution)
    if point:
        ends = [tran.atoms[0][0]]
    else:
        ends = list(_bounds(tran))
    out = np.zeros((2, a.size))
    for i, ai in enumerate(a):
        base = R * ai / G
        # where m' reaches a knot at an end, for assets of either sign
        cuts = [base / (k - e) for k in kinks for e in ends if k != e]
        psi, wp = pieces(*_bounds(perm), cuts, X40, W40)
        wp = wp * perm.pdf(psi)
        if point:
            theta, wt = np.full((psi.size, 1), ends[0]), np.ones((psi.size, 1))
        else:
            rows = [pieces(*ends, kinks - base / s, X40, W40) for s in psi]
            n = max(r[0].size for r in rows)
            theta = np.array(
                [np.pad(t, (0, n - t.size), constant_values=1.0) for t, _ in rows]
            )
            wt = np.array([np.pad(w * tran.pdf(t), (0, n - t.size)) for t, w in rows])
        m = base / psi[:, None] + theta
        c, mpc = solution.cFunc.eval_with_derivative(m)
        vP = c ** (-rho)
        weight = wp[:, None] * wt
        out[0, i] = np.sum(weight * psi[:, None] ** (-rho) * vP)
        out[1, i] = np.sum(weight * -rho * psi[:, None] ** (-rho - 1.0) * vP * mpc / c)
    return out


class SmoothConsumption(HARKinterpolator1D):
    """c(m) = min(m, 0.25 + 0.5 m^0.7): smooth but for the constraint's kink."""

    def _evaluate(self, m):
        return np.minimum(m, 0.25 + 0.5 * m**0.7)

    def _der(self, m):
        return np.where(m < 0.25 + 0.5 * m**0.7, 1.0, 0.35 * m**-0.3)

    def _evalAndDer(self, m):
        return self._evaluate(m), self._der(m)


class Mixture:
    """A mixture as the continuous expectation reads one: weights and components."""

    def __init__(self, weights, components):
        self.weights = weights
        self.components = components


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


M_MIN = -0.5  # the natural limit of the policy below


class NaturalConsumption(HARKinterpolator1D):
    """c(m) = d / (1 + d / 2), d = m - M_MIN: zero at the natural limit, no kink."""

    def _evaluate(self, m):
        d = m - M_MIN
        return d / (1.0 + 0.5 * d)

    def _der(self, m):
        return 1.0 / (1.0 + 0.5 * (m - M_MIN)) ** 2

    def _evalAndDer(self, m):
        return self._evaluate(m), self._der(m)


class testBegOfPrdContinuous(unittest.TestCase):
    def setUp(self):
        self.agent = IndShockConsumerType(
            cycles=1, T_cycle=1, BoroCnstArt=0.0, CubicBool=True, **BOUNDS
        )
        self.perm, self.tran, self.p, self.x = get_income_shock_laws(
            self.agent.IncShkDstn[0]
        )
        self.R, self.G = self.agent.Rfree[0], self.agent.PermGroFac[0]
        self.rho = self.agent.CRRA
        self.a = np.array([0.001, 0.05, 0.2, 0.5, 1.0, 3.0, 10.0, 20.0])
        self.smooth = ConsumerSolution(cFunc=SmoothConsumption(), mNrmMin=0.0)

    def expectations(self, solution, tran, **kwargs):
        return calc_BegOfPrd_continuous(
            self.a,
            solution,
            self.perm,
            tran,
            self.R,
            self.G,
            self.rho,
            CubicBool=True,
            **kwargs,
        )

    def test_laws_from_the_income_distribution(self):
        self.assertIsInstance(self.perm, LogNormalTruncatedMeanOne)
        self.assertIsInstance(self.tran, _ScaledLaw)  # the employed law, rescaled
        self.assertIsInstance(self.tran.law, LogNormalTruncated)
        self.assertEqual((self.p, self.x), (0.05, 0.3))
        kappa = (1.0 - 0.05 * 0.3) / (1.0 - 0.05)
        self.assertAlmostEqual(float(self.tran.mean()), kappa, places=13)
        plain = IndShockConsumerType(UnempPrb=0.0, **JUMPLESS)  # jumpless laws
        perm, tran, p, x = get_income_shock_laws(plain.IncShkDstn[0])
        self.assertIsInstance(perm, LogNormalTruncatedJumplessMeanOne)
        self.assertIsInstance(tran, LogNormalTruncatedJumplessMeanOne)
        self.assertEqual(p, 0.0)
        with self.assertRaises(ValueError):
            get_income_shock_laws(DiscreteDistribution([1.0], [1.0]))

    def test_kinks(self):
        self.assertEqual(get_mNrm_kinks(self.smooth).size, 1)
        kink = get_mNrm_kinks(self.smooth)[0]
        self.assertAlmostEqual(kink, 0.25 + 0.5 * kink**0.7, places=12)
        self.agent.solve()
        self.assertEqual(get_mNrm_kinks(self.agent.solution[-1]).size, 0)  # terminal
        recorded = ConsumerSolution(cFunc=SmoothConsumption(), mNrmMin=0.0)
        recorded.mNrmKinks = np.array([0.7])
        np.testing.assert_array_equal(get_mNrm_kinks(recorded), [0.7])
        # a constraint binding up to m = 60, found beyond the first window of 50
        high = ConsumerSolution(
            cFunc=LowerEnvelope(
                LinearInterp(np.array([0.0, 100.0]), np.array([30.0, 80.0])),
                LinearInterp(np.array([0.0, 1.0]), np.array([0.0, 1.0])),
            ),
            mNrmMin=0.0,
        )
        self.assertAlmostEqual(get_mNrm_kinks(high)[0], 60.0, places=10)

    def test_against_adaptive_quadrature(self):
        self.assertAlmostEqual(get_mNrm_kinks(self.smooth)[0], M_KINK, places=12)
        point = DiscreteDistribution([1.0], [self.x])
        a = self.a[[1, 3, 5, 7]]
        for tran in (self.tran, point):
            exact = adaptive_reference(a, self.perm, tran, self.R, self.G, self.rho)
            # to 2e-8 at four nodes a piece, and to rounding at eight
            for nodes, rtol in ((4, 2e-8), (8, 1e-13)):
                branch = "unemployed" if tran is point else "employed"
                with self.subTest(branch=branch, nodes=nodes):
                    E1, E2, E0 = calc_BegOfPrd_continuous(
                        a,
                        self.smooth,
                        self.perm,
                        tran,
                        self.R,
                        self.G,
                        self.rho,
                        True,
                        PermShkQuadCount=nodes,
                        TranShkQuadCount=nodes,
                    )
                    np.testing.assert_allclose(E1, exact[0], rtol=rtol)
                    np.testing.assert_allclose(E2, exact[1], rtol=rtol)
                    self.assertIsNone(E0)

    def test_against_adaptive_quadrature_on_the_jumpless_laws(self):
        # The constructor's jumpless laws, bounded three standard deviations
        # below, where their density is zero, and not above: integrated in their
        # own coordinate, against its density
        agent = IndShockConsumerType(cycles=1, T_cycle=1, **JUMPLESS)
        perm, tran, p, x = get_income_shock_laws(agent.IncShkDstn[0])
        self.assertIsInstance(perm, LogNormalTruncatedJumplessMeanOne)
        self.assertIsInstance(tran.law, LogNormalTruncatedJumplessMeanOne)
        # one asset level: the nested adaptive reference is slow on these laws
        a = self.a[[5]]
        exact = adaptive_reference(a, perm, tran, self.R, self.G, self.rho)
        for nodes, rtol in ((4, 1e-6), (8, 1e-11)):
            with self.subTest(nodes=nodes):
                E1, E2, _ = calc_BegOfPrd_continuous(
                    a,
                    self.smooth,
                    perm,
                    tran,
                    self.R,
                    self.G,
                    self.rho,
                    True,
                    PermShkQuadCount=nodes,
                    TranShkQuadCount=nodes,
                )
                np.testing.assert_allclose(E1, exact[0], rtol=rtol)
                np.testing.assert_allclose(E2, exact[1], rtol=rtol)

    def test_geometric_convergence_in_the_nodes(self):
        a = self.a[[1, 3, 5, 7]]
        exact = adaptive_reference(a, self.perm, self.tran, self.R, self.G, self.rho)
        errors = []
        for n in (2, 3, 4):
            E1 = calc_BegOfPrd_continuous(
                a,
                self.smooth,
                self.perm,
                self.tran,
                self.R,
                self.G,
                self.rho,
                PermShkQuadCount=n,
                TranShkQuadCount=n,
            )[0]
            errors.append(np.max(np.abs(E1 / exact[0] - 1.0)))
        # each node more a piece gains more than a factor of fifty
        self.assertGreater(errors[0] / errors[1], 50.0)
        self.assertGreater(errors[1] / errors[2], 50.0)
        self.assertLess(errors[2], 1e-8)

    def test_solved_cubic_policy(self):
        # next period's MPC is kinked at its knots, where the reference splits and
        # the rule does not: more nodes a piece still gain, but no longer reach
        # rounding (at most 6e-8 at four, 1e-9 at eight)
        self.agent.solve()
        solution = self.agent.solution[0]
        exact = knot_reference(
            solution, self.a, self.perm, self.tran, self.R, self.G, self.rho
        )
        for nodes, rtol in ((4, 2e-7), (8, 3e-9)):
            with self.subTest(nodes=nodes):
                E1, E2, _ = self.expectations(
                    solution,
                    self.tran,
                    PermShkQuadCount=nodes,
                    TranShkQuadCount=nodes,
                )
                np.testing.assert_allclose(E1, exact[0], rtol=rtol)
                np.testing.assert_allclose(E2, exact[1], rtol=rtol)

    def test_solved_cubic_policy_with_other_laws(self):
        # against the reference split at every knot, which the rule is not (at
        # most 2.3e-7 at four nodes a piece, 1e-7 at eight), with a uniform
        # permanent law and, for the transitory, a uniform and a law known only by
        # its quantile function, CDF and density
        self.agent.solve()
        solution = self.agent.solution[0]
        perm = Uniform(0.85, 1.15)
        for name, tran in {
            "uniform": Uniform(0.6, 1.4),
            "quantile function": QuantileLaw(beta(2.0, 3.0, loc=0.4, scale=1.5)),
        }.items():
            exact = knot_reference(
                solution, self.a, perm, tran, self.R, self.G, self.rho
            )
            for nodes, rtol in ((4, 3e-7), (8, 3e-7)):
                with self.subTest(transitory=name, nodes=nodes):
                    E1, E2, _ = calc_BegOfPrd_continuous(
                        self.a,
                        solution,
                        perm,
                        tran,
                        self.R,
                        self.G,
                        self.rho,
                        True,
                        PermShkQuadCount=nodes,
                        TranShkQuadCount=nodes,
                    )
                    np.testing.assert_allclose(E1, exact[0], rtol=rtol)
                    np.testing.assert_allclose(E2, exact[1], rtol=rtol)

    def test_solved_policy_below_zero_assets(self):
        # With no artificial constraint, assets fall below zero toward the natural
        # limit, where m' reaches next period's knots from the other side of the
        # transitory law's ends and of the unemployment point; next period's policy
        # is solved over atoms
        agent = IndShockConsumerType(
            cycles=1, T_cycle=1, CubicBool=True, BoroCnstArt=None, **BOUNDS
        )
        agent.cycles = 3
        agent.solve()
        solution = agent.solution[0]
        self.assertLess(solution.mNrmMin, 0.0)
        lo_p, lo_t = self.perm.infimum[0], self.tran.infimum[0]
        point = DiscreteDistribution([1.0], [self.x])
        for name, tran in (("employed", self.tran), ("unemployed", point)):
            worst = lo_t if tran is self.tran else self.x
            limit = (solution.mNrmMin - worst) * self.G * lo_p / self.R
            a = limit + np.array([0.05, 0.2, 0.5])
            self.assertTrue(np.all(a < 0.0))
            exact = knot_reference(
                solution, a, self.perm, tran, self.R, self.G, self.rho
            )
            # the reference splits at next period's knots, the rule not: at most
            # 9e-6 at four nodes a piece and 5e-7 at eight, in E2 near the limit
            for nodes, rtol in ((4, 3e-5), (8, 2e-6)):
                with self.subTest(branch=name, nodes=nodes):
                    E1, E2, _ = calc_BegOfPrd_continuous(
                        a,
                        solution,
                        self.perm,
                        tran,
                        self.R,
                        self.G,
                        self.rho,
                        True,
                        PermShkQuadCount=nodes,
                        TranShkQuadCount=nodes,
                    )
                    np.testing.assert_allclose(E1, exact[0], rtol=rtol)
                    np.testing.assert_allclose(E2, exact[1], rtol=rtol)

    def test_near_the_natural_limit(self):
        # Next period's consumption falls to zero at its minimum, with no kink.
        # For the unemployed, a point, assets just above the natural limit land
        # some of them just above that minimum, where the permanent integral is
        # steep; it is split geometrically there, for a permanent law bounded at
        # both ends or below only
        natural = ConsumerSolution(cFunc=NaturalConsumption(), mNrmMin=M_MIN)
        self.assertEqual(get_mNrm_kinks(natural).size, 0)
        point = DiscreteDistribution([1.0], [self.x])
        for name, perm in {
            "both ends": self.perm,
            "below only": LogNormalTruncatedMeanOne.from_sd_log(0.1, -2.0, None),
        }.items():
            with self.subTest(bounded=name):
                self.check_near_the_natural_limit(natural, perm, point)

    def check_near_the_natural_limit(self, natural, perm, point):
        lo = perm.infimum[0]
        hi = min(perm.supremum[0], float(perm.ppf(1.0 - 1e-16)))
        limit = (M_MIN - self.x) * self.G * lo / self.R  # the worst landing is M_MIN
        # from 1e-6 up: nearer the limit, m' - M_MIN loses digits to cancellation
        a = limit + np.array([1e-6, 1e-4, 1e-2, 0.1, 1.0])
        Es = {
            nodes: calc_BegOfPrd_continuous(
                a,
                natural,
                perm,
                point,
                self.R,
                self.G,
                self.rho,
                True,
                PermShkQuadCount=nodes,
                TranShkQuadCount=nodes,
            )
            for nodes in (4, 8)
        }
        for i, ai in enumerate(a):
            base = self.R * ai / self.G

            def integrand(psi, k):
                d = base / psi + self.x - M_MIN
                c, mpc = d / (1.0 + 0.5 * d), 1.0 / (1.0 + 0.5 * d) ** 2
                vP = psi ** (-self.rho) * c ** (-self.rho)
                return (vP if k == 0 else -self.rho * vP * mpc / (psi * c)) * (
                    perm.pdf(psi)
                )

            # breakpoints spaced geometrically above the steep end
            near = lo + (hi - lo) * np.geomspace(1e-10, 0.5, 12)
            for k in (0, 1):
                exact = quad(
                    integrand,
                    lo,
                    hi,
                    args=(k,),
                    points=near,
                    epsabs=0.0,
                    epsrel=1e-12,
                    limit=1000,
                )[0]
                for nodes, places in ((4, 7), (8, 9)):
                    self.assertAlmostEqual(Es[nodes][k][i] / exact, 1.0, places=places)

    def test_at_the_corner_of_the_support(self):
        # Unemployment insurance above the lowest employed income: the employed
        # branch's corner, the permanent shock's lower end with the transitory
        # shock's, sets the natural limit, and both of its integrals are graded
        # toward the near-pole there
        agent = IndShockConsumerType(IncUnemp=0.6, TranShkStd=[0.3], **BOUNDS)
        perm, tran, _, x = get_income_shock_laws(agent.IncShkDstn[0])
        lo_p, hi_p = perm.infimum[0], perm.supremum[0]
        lo_t, hi_t = tran.infimum[0], tran.supremum[0]
        self.assertLess(lo_t, x)
        natural = ConsumerSolution(cFunc=NaturalConsumption(), mNrmMin=M_MIN)
        limit = (M_MIN - lo_t) * self.G * lo_p / self.R  # the corner lands on M_MIN
        a = limit + np.array([1e-6, 1e-3, 0.1])
        E1s = {
            nodes: calc_BegOfPrd_continuous(
                a,
                natural,
                perm,
                tran,
                self.R,
                self.G,
                self.rho,
                PermShkQuadCount=nodes,
                TranShkQuadCount=nodes,
            )[0]
            for nodes in (4, 8)
        }

        def log_distance_quad(f, lo, hi, pole):
            # adaptive quadrature in u = log(x - pole), in which a near-pole at
            # pole, just below lo, is smooth
            def g(u):
                return f(pole + np.exp(u)) * np.exp(u)

            u_lo, u_hi = np.log(lo - pole), np.log(hi - pole)
            return quad(g, u_lo, u_hi, epsabs=0.0, epsrel=1e-12, limit=200)[0]

        for i, ai in enumerate(a):
            base = self.R * ai / self.G

            def given_psi(psi):
                def f(t):
                    d = base / psi + t - M_MIN
                    return (d / (1 + 0.5 * d)) ** -self.rho * tran.pdf(t)

                inner = log_distance_quad(f, lo_t, hi_t, M_MIN - base / psi)
                return psi ** (-self.rho) * inner * perm.pdf(psi)

            pole = base / (M_MIN - lo_t)  # where the corner's landing is M_MIN
            exact = log_distance_quad(given_psi, lo_p, hi_p, pole)
            for nodes, places in ((4, 7), (8, 10)):
                self.assertAlmostEqual(E1s[nodes][i] / exact, 1.0, places=places)

    def test_scaled_law_for_any_law(self):
        # a law of no family that rescales in closed form: its quantile function,
        # CDF, density and bounds, all scaled
        law = Uniform(0.6, 1.4)
        scaled = _scaled_law(law, 1.05)
        self.assertIsInstance(scaled, _ScaledLaw)
        self.assertAlmostEqual(scaled.mean(), 1.05, places=14)
        np.testing.assert_allclose(_bounds(scaled), (0.63, 1.47), rtol=1e-14)
        q = np.linspace(0.01, 0.99, 7)
        np.testing.assert_allclose(scaled.cdf(scaled.ppf(q)), q, rtol=1e-12)
        x = np.linspace(0.64, 1.46, 9)
        np.testing.assert_allclose(scaled.pdf(x), 1.0 / 0.84, rtol=1e-12)
        # a truncated lognormal is scaled the same way, as the same family with its
        # mu moved would be, and keeps the coordinate it supplies, and so the
        # pieces its integral is cut into
        tl = LogNormalTruncatedMeanOne.from_sd_log(0.1, -3.0, 3.0)
        scaled = _scaled_law(tl, 1.05)
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
        # a mixture is scaled component by component
        mixture = _scaled_law(Mixture([0.4, 0.6], [law, tl]), 1.05)
        self.assertEqual(len(mixture.components), 2)
        self.assertIsInstance(mixture.components[1], _ScaledLaw)
        self.assertAlmostEqual(float(mixture.infimum[0]), 0.63, places=14)

    def test_income_laws_with_a_uniform_employed_law(self):
        # unemployment's rescaling of the employed transitory law, for a law that is
        # not lognormal
        tran = add_discrete_outcome_constant_mean(
            Uniform(0.6, 1.4).discretize(7), 0.3, 0.05, sort=True
        )
        joint = DiscreteDistributionLabeled.from_unlabeled(
            combine_indep_dstns(Uniform(0.85, 1.15).discretize(7), tran),
            var_names=["PermShk", "TranShk"],
        )
        perm, employed, p, x = get_income_shock_laws(joint)
        self.assertIsInstance(perm, Uniform)
        self.assertEqual((p, x), (0.05, 0.3))
        kappa = (1.0 - 0.05 * 0.3) / (1.0 - 0.05)
        self.assertAlmostEqual(employed.mean(), kappa, places=14)
        np.testing.assert_allclose(_bounds(employed), (0.6 * kappa, 1.4 * kappa))

    def test_against_adaptive_quadrature_with_other_laws(self):
        # uniform laws, and a transitory law known only by its quantile function
        perm = Uniform(0.85, 1.15)
        laws = {
            "uniform": Uniform(0.6, 1.4),
            "quantile function": QuantileLaw(beta(2.0, 3.0, loc=0.4, scale=1.5)),
            "unemployed": DiscreteDistribution([1.0], [0.3]),
        }
        a = self.a[[1, 3, 5, 7]]
        for name, tran in laws.items():
            with self.subTest(transitory=name):
                exact = adaptive_reference(a, perm, tran, self.R, self.G, self.rho)
                # at the default nodes, four a piece, and at eight
                for nodes, rtol in (((4, 4), 2e-7), ((8, 8), 1e-12)):
                    E1, E2, _ = calc_BegOfPrd_continuous(
                        a,
                        self.smooth,
                        perm,
                        tran,
                        self.R,
                        self.G,
                        self.rho,
                        True,
                        PermShkQuadCount=nodes[0],
                        TranShkQuadCount=nodes[1],
                    )
                    np.testing.assert_allclose(E1, exact[0], rtol=rtol)
                    np.testing.assert_allclose(E2, exact[1], rtol=rtol)

    def test_near_the_natural_limit_with_other_laws(self):
        # the graded rule in the laws' own values, for laws with a density at their
        # ends: unemployment's point at the worst, and the employed corner
        natural = ConsumerSolution(cFunc=NaturalConsumption(), mNrmMin=M_MIN)
        perm = Uniform(0.85, 1.15)
        lo_p, hi_p = 0.85, 1.15
        rho = self.rho
        for name, tran in {
            "unemployed": DiscreteDistribution([1.0], [0.3]),
            "employed": Uniform(0.6, 1.4),
        }.items():
            with self.subTest(branch=name):
                point = isinstance(tran, DiscreteDistribution)
                lo_t, hi_t = (0.3, 0.3) if point else (0.6, 1.4)
                limit = (M_MIN - lo_t) * self.G * lo_p / self.R
                a = limit + np.array([1e-6, 1e-3, 0.1])
                E1s = {
                    nodes: calc_BegOfPrd_continuous(
                        a,
                        natural,
                        perm,
                        tran,
                        self.R,
                        self.G,
                        rho,
                        PermShkQuadCount=nodes,
                        TranShkQuadCount=nodes,
                    )[0]
                    for nodes in (4, 8)
                }
                for i, ai in enumerate(a):
                    base = self.R * ai / self.G

                    def given_psi(psi):
                        def f(t):
                            d = base / psi + t - M_MIN
                            return (d / (1 + 0.5 * d)) ** -rho * (
                                1.0 if point else tran.pdf(t)
                            )

                        inner = (
                            f(lo_t)
                            if point
                            else log_distance_quad(f, lo_t, hi_t, M_MIN - base / psi)
                        )
                        return psi ** (-rho) * inner * perm.pdf(psi)

                    pole = base / (M_MIN - lo_t)
                    exact = log_distance_quad(given_psi, lo_p, hi_p, pole)
                    for nodes, places in ((4, 7), (8, 9)):
                        self.assertAlmostEqual(
                            E1s[nodes][i] / exact, 1.0, places=places
                        )

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
                        values, weights, graded = _graded_quadrature(
                            form, nodes, pole, cut
                        )
                        self.assertTrue(np.all(graded))
                        E = np.sum(weights / (values - pole[:, None]), axis=-1)
                        np.testing.assert_allclose(E, exact, rtol=rtol)

    def test_degenerate_permanent_shock(self):
        one = DiscreteDistribution([1.0], [1.0])
        kink = get_mNrm_kinks(self.smooth)
        for nodes, places in ((4, 8), (8, 12)):
            E1 = calc_BegOfPrd_continuous(
                self.a,
                self.smooth,
                one,
                self.tran,
                self.R,
                self.G,
                self.rho,
                PermShkQuadCount=nodes,
                TranShkQuadCount=nodes,
            )[0]
            for i, ai in enumerate(self.a):
                base = self.R * ai / self.G
                t, w = pieces(self.tran.infimum[0], self.tran.supremum[0], kink - base)
                c = self.smooth.cFunc(base + t)
                exact = np.sum(w * self.tran.pdf(t) * c ** (-self.rho))
                self.assertAlmostEqual(E1[i] / exact, 1.0, places=places)

    def test_value(self):
        agent = IndShockConsumerType(
            cycles=1, T_cycle=1, BoroCnstArt=0.0, vFuncBool=True, **BOUNDS
        )
        agent.solve()
        solution = agent.solution[0]
        E0s = {
            nodes: calc_BegOfPrd_continuous(
                self.a,
                solution,
                self.perm,
                self.tran,
                self.R,
                self.G,
                self.rho,
                vFuncBool=True,
                PermShkQuadCount=nodes,
                TranShkQuadCount=nodes,
            )[2]
            for nodes in (4, 8)
        }
        kink = get_mNrm_kinks(solution)
        for i, ai in enumerate(self.a[[1, 4, 6]]):
            base = self.R * ai / self.G
            cuts = [
                base / (k - e)
                for k in kink
                for e in (self.tran.infimum[0], self.tran.supremum[0])
                if k > e
            ]
            psi, wp = pieces(self.perm.infimum[0], self.perm.supremum[0], cuts)
            total = 0.0
            for s, ws in zip(psi, wp * self.perm.pdf(psi)):
                t, wt = pieces(
                    self.tran.infimum[0], self.tran.supremum[0], kink - base / s
                )
                v = solution.vFunc(base / s + t)
                total += (
                    ws
                    * (self.G * s) ** (1.0 - self.rho)
                    * np.sum(wt * self.tran.pdf(t) * v)
                )
            # the solved value function's knots kink its slope, as in the policy
            # test; at the lowest of these assets, four nodes a piece are within
            # about 1e-8, eight within about 1e-9
            for nodes, places in ((4, 7), (8, 8)):
                self.assertAlmostEqual(
                    E0s[nodes][[1, 4, 6][i]] / total, 1.0, places=places
                )

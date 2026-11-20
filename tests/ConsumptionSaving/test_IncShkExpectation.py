"""
Tests for IndShockConsumerType's option IncShkExpectation: "atoms", the default,
and "continuous", which takes the expectation over the income shocks against
their laws with solve_one_period_ConsIndShock_continuous.
"""

import pickle
import unittest
from copy import deepcopy

import numpy as np
from scipy.stats import beta, gamma

from HARK.ConsumptionSaving.ConsIndShockModel import (
    TRANSITION_SPLIT,
    ConsumerSolution,
    IndShockConsumerType,
    KinkedRconsumerType,
    _bounds,
    _calc_BegOfPrd_IndShock_continuous,
    _law_end,
    _mpc_transition,
    _natural_limit_assets,
    _reaches_natural_limit,
    calc_BegOfPrd_continuous,
    get_income_shock_laws,
    get_mNrm_kinks,
    solve_one_period_ConsIndShock,
    solve_one_period_ConsIndShock_continuous,
)
from HARK.ConsumptionSaving.ConsPrefShockModel import PrefShockConsumerType
from HARK.distributions import (
    DiscreteDistribution,
    DiscreteDistributionLabeled,
    LogNormalMeanOne,
    LogNormalTruncatedMeanOne,
    LogNormalTruncated,
    PointMassAtInfimum,
    Uniform,
    add_discrete_outcome_constant_mean,
    combine_indep_dstns,
)
from HARK.interpolation import CubicInterp, LinearInterp
from HARK.utilities import get_arg_names

# The jumpless laws, bounded three standard deviations below, on which these
# tests run; HARK's default is the truncated one (testTruncatedDefault, below)
BOUNDS = {
    "PermShk_inf": -3.0,
    "TranShk_inf": -3.0,
    "PermShk_tails": "jumpless",
    "TranShk_tails": "jumpless",
}
# The truncated laws, bounded three standard deviations either side
TRUNCATED_BOUNDS = {
    "PermShk_inf": -3.0,
    "PermShk_sup": 3.0,
    "TranShk_inf": -3.0,
    "TranShk_sup": 3.0,
    "PermShk_tails": "truncated",
    "TranShk_tails": "truncated",
}
JUMPLESS_TAILS = {"PermShk_tails": "jumpless", "TranShk_tails": "jumpless"}


def largest_rise(cFunc, m):
    """How far the MPC climbs anywhere to the right: zero if cFunc is concave."""
    mpc = cFunc.derivative(m)
    return np.max(np.maximum.accumulate(mpc[::-1])[::-1] - mpc)


def knots_in_transition(solution, CubicBool, first):
    """How many of a solution's gridpoints lie in the transition located on it,
    where its MPC falls from near 1."""
    d_lo, d_hi = _mpc_transition(solution.cFunc, solution.mNrmMin, first, CubicBool)
    d = np.asarray(solution.cFunc.functions[0].x_list) - solution.mNrmMin
    return np.count_nonzero((d >= d_lo * (1 - 1e-9)) & (d <= d_hi * (1 + 1e-9)))


def falling_mpc(d, d0, kappa=0.1, width=0.1):
    """A consumption function from zero at d = 0 whose MPC falls from 1 to kappa
    around d0, and its MPC."""
    fall = np.logaddexp(0.0, (d - d0) / width) - np.logaddexp(0.0, -d0 / width)
    mpc = kappa + (1.0 - kappa) / (1.0 + np.exp((d - d0) / width))
    return kappa * d + (1.0 - kappa) * (d - width * fall), mpc


class Mixture:
    """A mixture as the continuous expectation reads one: weights and components."""

    def __init__(self, weights, components):
        self.weights = weights
        self.components = components
        self.infimum = np.array([min(c.infimum[0] for c in components)])
        self.supremum = np.array([max(c.supremum[0] for c in components)])


class QuantileLaw:
    """A law known only through its quantile function, CDF and density."""

    def __init__(self, law):
        self._law = law

    def ppf(self, q):
        return self._law.ppf(q)

    def cdf(self, x):
        return self._law.cdf(x)

    def pdf(self, x):
        return self._law.pdf(x)


def quantile_income(IncShkDstn):
    """
    IndShock's income distribution with the laws it records replaced by laws known
    only through their quantile functions, CDFs and densities.
    """
    perm, tran = IncShkDstn.limit["dist"]
    perm.limit["dist"] = QuantileLaw(perm.limit["dist"])
    employed = tran.limit["dist"]
    employed.limit["dist"] = QuantileLaw(employed.limit["dist"])
    return IncShkDstn


def uniform_income(UnempPrb):
    """
    IndShock's income from uniform laws, discretized as its income constructor
    discretizes its own, with unemployment paying 0.3 with probability UnempPrb.
    """
    tran = Uniform(0.6, 1.4).discretize(7)
    if UnempPrb > 0.0:
        tran = add_discrete_outcome_constant_mean(tran, 0.3, UnempPrb, sort=True)
    return DiscreteDistributionLabeled.from_unlabeled(
        combine_indep_dstns(Uniform(0.85, 1.15).discretize(7), tran),
        var_names=["PermShk", "TranShk"],
    )


class testIncShkExpectation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = IndShockConsumerType(
            cycles=0, CubicBool=True, IncShkExpectation="continuous", **BOUNDS
        )
        cls.agent.solve()

    def test_atoms_is_the_default(self):
        default = IndShockConsumerType(cycles=0)
        default.solve()
        atoms = IndShockConsumerType(cycles=0, IncShkExpectation="atoms")
        atoms.solve()
        self.assertIs(default.solve_one_period, solve_one_period_ConsIndShock)
        m = np.linspace(0.0, 20.0, 2001)
        np.testing.assert_array_equal(
            default.solution[0].cFunc(m), atoms.solution[0].cFunc(m)
        )
        self.assertFalse(hasattr(default.solution[0], "mNrmKinks"))

    def test_continuous_solution(self):
        agent = self.agent
        self.assertIs(agent.solve_one_period, solve_one_period_ConsIndShock_continuous)
        self.assertLess(agent.solution_distance, agent.tolerance)
        solution = agent.solution[0]
        # the kink is where the constraint stops binding: c = m there (mNrmMin = 0),
        # the endogenous gridpoint of the constraint itself, a knot
        kink = solution.mNrmKinks
        self.assertEqual(kink.size, 1)
        self.assertAlmostEqual(float(solution.cFunc(kink)[0]), kink[0], places=14)
        self.assertEqual(kink[0], solution.cFunc.functions[0].x_list[1])
        np.testing.assert_array_equal(
            get_mNrm_kinks(solution), kink
        )  # read back, not searched
        self.assertEqual(solution.mNrmKink, kink[0])  # as the atoms' solver records it
        # no concavity violation, and none of the shape-preserving cubic's tension
        m = np.linspace(kink[0] + 1e-6, 20.0, 200001)
        self.assertLessEqual(largest_rise(solution.cFunc, m), 1e-12)
        np.testing.assert_array_equal(solution.cFunc.functions[0].tension, 0.0)

    def test_no_concavity_violation_on_a_fine_grid(self):
        # 200 gridpoints, where over the atoms a plain cubic violates concavity
        agent = IndShockConsumerType(
            cycles=1,
            T_cycle=1,
            CubicBool=True,
            IncShkExpectation="continuous",
            aXtraCount=200,
            **BOUNDS,
        )
        agent.cycles = 10
        agent.solve()
        cFunc = agent.solution[0].cFunc
        m = np.linspace(0.8, 3.0, 200001)
        self.assertLessEqual(largest_rise(cFunc, m), 1e-12)
        self.assertFalse(np.any(cFunc.functions[0].tension > 0.0))

    def test_value_function(self):
        agent = IndShockConsumerType(
            cycles=1,
            T_cycle=1,
            vFuncBool=True,
            IncShkExpectation="continuous",
            **BOUNDS,
        )
        agent.cycles = 5
        agent.solve()
        solution = agent.solution[0]
        # the envelope condition, v'(m) = u'(c(m)), to the accuracy of the value
        # function's interpolation (over the atoms it holds to 6e-4 here)
        m = np.linspace(1.0, 10.0, 19)
        h = 1e-5
        dv = (solution.vFunc(m + h) - solution.vFunc(m - h)) / (2 * h)
        np.testing.assert_allclose(dv, solution.cFunc(m) ** (-agent.CRRA), rtol=1e-3)

    def test_degenerate_permanent_shock(self):
        agent = IndShockConsumerType(
            cycles=1,
            T_cycle=1,
            PermShkStd=[0.0],
            IncShkExpectation="continuous",
            TranShk_inf=-3.0,
            TranShk_sup=3.0,
        )
        agent.cycles = 5
        agent.solve()
        self.assertTrue(
            np.all(np.isfinite(agent.solution[0].cFunc(np.linspace(0, 10, 11))))
        )

    def test_solver_follows_the_option(self):
        # Set when the agent is made, when the option is assigned and in
        # pre_solve, for code that runs solve_one_period itself
        agent = IndShockConsumerType(
            cycles=1,
            T_cycle=1,
            CubicBool=True,
            IncShkExpectation="continuous",
            **BOUNDS,
        )
        self.assertIs(agent.solve_one_period, solve_one_period_ConsIndShock_continuous)
        agent.assign_parameters(IncShkExpectation="atoms")
        self.assertIs(agent.solve_one_period, solve_one_period_ConsIndShock)
        agent.IncShkExpectation = "continuous"
        agent.pre_solve()
        self.assertIs(agent.solve_one_period, solve_one_period_ConsIndShock_continuous)
        args = {}
        for name in get_arg_names(agent.solve_one_period):
            if name in agent.time_vary:
                args[name] = getattr(agent, name)[0]
            elif name in agent.time_inv:
                args[name] = getattr(agent, name)
        direct = agent.solve_one_period(solution_next=agent.solution_terminal, **args)
        agent.solve()
        m = np.linspace(0.0, 20.0, 201)
        np.testing.assert_array_equal(direct.cFunc(m), agent.solution[0].cFunc(m))

    def test_back_to_atoms(self):
        agent = deepcopy(self.agent)
        agent.IncShkExpectation = "atoms"
        agent.solve()
        self.assertIs(agent.solve_one_period, solve_one_period_ConsIndShock)
        fresh = IndShockConsumerType(cycles=0, CubicBool=True, **BOUNDS)
        fresh.solve()
        m = np.linspace(0.0, 20.0, 2001)
        np.testing.assert_array_equal(
            agent.solution[0].cFunc(m), fresh.solution[0].cFunc(m)
        )

    def test_pickling(self):
        # the solver pickles by reference, and the solution and the agent copy
        self.assertIs(
            pickle.loads(pickle.dumps(solve_one_period_ConsIndShock_continuous)),
            solve_one_period_ConsIndShock_continuous,
        )
        m = np.linspace(0.0, 20.0, 201)
        solution = self.agent.solution[0]
        for copy in (pickle.loads(pickle.dumps(solution)), deepcopy(solution)):
            np.testing.assert_array_equal(copy.cFunc(m), solution.cFunc(m))
            np.testing.assert_array_equal(copy.mNrmKinks, solution.mNrmKinks)
        agent = deepcopy(self.agent)
        self.assertIs(agent.solve_one_period, solve_one_period_ConsIndShock_continuous)

    def test_natural_constraint(self):
        # No artificial constraint: the natural one, from the laws' infima, binds,
        # and consumption falls to zero at it with no kink
        agent = IndShockConsumerType(
            cycles=0,
            CubicBool=True,
            BoroCnstArt=None,
            IncShkExpectation="continuous",
            **BOUNDS,
        )
        agent.solve()
        # the asset grid, placed by next period's transition, settles with the
        # solution
        self.assertLess(agent.solution_distance, agent.tolerance)
        self.assertLess(agent.completed_cycles, 200)
        solution = agent.solution[0]
        limit = solution.mNrmMin
        self.assertLess(limit, 0.0)
        self.assertEqual(solution.mNrmKinks.size, 0)
        self.assertTrue(np.isnan(solution.mNrmKink))
        self.assertEqual(float(solution.cFunc(np.array([limit]))[0]), 0.0)
        m = limit + np.r_[np.geomspace(1e-6, 0.1, 61), np.linspace(0.11, 20.0, 2000)]
        self.assertTrue(np.all(solution.cFunc(m) > 0.0))
        self.assertLessEqual(largest_rise(solution.cFunc, m), 1e-12)
        # its points land in the transition of the solution itself
        first = agent.aXtraGrid[0]
        self.assertGreaterEqual(
            knots_in_transition(solution, True, first), TRANSITION_SPLIT[2]
        )
        # accurate at the default nodes, four a piece, against six
        fine = IndShockConsumerType(
            cycles=0,
            CubicBool=True,
            BoroCnstArt=None,
            IncShkExpectation="continuous",
            PermShkQuadCount=6,
            TranShkQuadCount=6,
            **BOUNDS,
        )
        fine.solve()
        np.testing.assert_allclose(
            solution.cFunc(m), fine.solution[0].cFunc(m), rtol=0.0, atol=2e-6
        )

    def test_same_limit_as_the_atoms(self):
        # The natural borrowing constraint belongs to the model, from the laws'
        # infima: the continuous expectation has the one 7 and 11 atoms have,
        # in every period of a finite horizon
        limits = []
        for kwds in [
            dict(PermShkCount=7, TranShkCount=7),
            dict(PermShkCount=11, TranShkCount=11),
            dict(IncShkExpectation="continuous"),
        ]:
            agent = IndShockConsumerType(cycles=2, BoroCnstArt=None, **kwds)
            agent.solve()
            # The last solution is the terminal period's
            limits.append([s.mNrmMin for s in agent.solution[:-1]])
        np.testing.assert_allclose(limits[1], limits[0], rtol=0, atol=1e-12)
        np.testing.assert_allclose(limits[2], limits[0], rtol=0, atol=1e-12)

    def test_natural_constraint_linear(self):
        # The same with linear interpolation, where the transition is read off a
        # secant of next period's consumption function: the asset grid settles,
        # and the infinite horizon converges
        agent = IndShockConsumerType(
            cycles=0,
            CubicBool=False,
            BoroCnstArt=None,
            IncShkExpectation="continuous",
            **BOUNDS,
        )
        agent.solve()
        self.assertLess(agent.solution_distance, agent.tolerance)
        self.assertLess(agent.completed_cycles, 200)
        solution = agent.solution[0]
        self.assertEqual(solution.mNrmKinks.size, 0)
        first = agent.aXtraGrid[0]
        self.assertGreaterEqual(
            knots_in_transition(solution, False, first), TRANSITION_SPLIT[2]
        )

    def test_transition_moves_continuously(self):
        # As the transition of next period's consumption function, where its MPC
        # falls from near 1, slides past its gridpoints, its located edges move in
        # steps all of about the same size, linear or cubic. Read off a linear
        # interpolant's derivative, or at the scanned distances themselves, they
        # stall and jump instead, and an asset grid placed by them can cycle
        # forever over an infinite horizon
        d = np.r_[
            0.0,
            np.geomspace(1e-4, 0.4, 12),
            np.linspace(0.45, 1.5, 22),
            np.linspace(1.6, 20.0, 30),
        ]
        for cubic in (False, True):
            edges = []
            for d0 in np.linspace(0.6, 0.65, 26):
                c, mpc = falling_mpc(d, d0)
                cFunc = CubicInterp(d, c, mpc) if cubic else LinearInterp(d, c)
                edges.append(_mpc_transition(cFunc, 0.0, 0.001, cubic))
            steps = np.diff(np.array(edges), axis=0)
            self.assertTrue(np.all(steps > 0.0))
            self.assertTrue(np.all(steps.max(axis=0) < 3.0 * steps.min(axis=0)))

    def test_natural_limit_assets(self):
        # Where next period's consumption function has a transition, the 17
        # points are placed by it: 2 close to the limit, 4 below the transition, 8
        # across it, from the assets at its lower edge to those at its upper one,
        # and 3 above it
        first, limit = 0.001, -0.5
        d = np.r_[0.0, np.geomspace(1e-4, 0.4, 12), np.linspace(0.45, 20.0, 60)]
        for cubic in (False, True):
            c, mpc = falling_mpc(d, 0.6)
            cFunc = (
                CubicInterp(d + limit, c, mpc) if cubic else LinearInterp(d + limit, c)
            )
            solution = ConsumerSolution(cFunc=cFunc, mNrmMin=limit)
            points = _natural_limit_assets(solution, limit, first, cubic) - limit
            self.assertEqual(points.size, sum(TRANSITION_SPLIT))
            self.assertTrue(np.all(np.diff(points) > 0.0))
            d_lo, d_hi = _mpc_transition(cFunc, limit, first, cubic)
            a_lo, a_hi = np.array([d_lo, d_hi]) - cFunc(limit + np.array([d_lo, d_hi]))
            across = points[
                (points >= a_lo * (1 - 1e-9)) & (points <= a_hi * (1 + 1e-9))
            ]
            self.assertEqual(across.size, TRANSITION_SPLIT[2])
            self.assertEqual(np.count_nonzero(points < a_lo * (1 - 1e-9)), 6)
            self.assertEqual(np.count_nonzero(points > a_hi * (1 + 1e-9)), 3)
        # With no transition, as when next period is the terminal one, the 17
        # fixed points
        terminal = ConsumerSolution(
            cFunc=LinearInterp(np.array([limit, limit + 1.0]), np.array([0.0, 1.0])),
            mNrmMin=limit,
        )
        for cubic in (False, True):
            np.testing.assert_allclose(
                _natural_limit_assets(terminal, limit, first, cubic),
                limit + first * np.geomspace(1e-6, 100.0, 17),
            )

    def test_natural_constraint_value_function(self):
        agent = IndShockConsumerType(
            cycles=1,
            T_cycle=1,
            vFuncBool=True,
            BoroCnstArt=None,
            IncShkExpectation="continuous",
            **BOUNDS,
        )
        agent.cycles = 5
        agent.solve()
        solution = agent.solution[0]
        # the envelope condition, v'(m) = u'(c(m)), to the accuracy of the value
        # function's interpolation, coarser near this limit, where the
        # end-of-period value falls steeply (its level: the next test)
        m = solution.mNrmMin + np.linspace(1.0, 10.0, 19)
        h = 1e-5
        dv = (solution.vFunc(m + h) - solution.vFunc(m - h)) / (2 * h)
        np.testing.assert_allclose(dv, solution.cFunc(m) ** (-agent.CRRA), rtol=2e-3)

    def test_natural_limit_value_level(self):
        # Under the jumpless laws only the unemployed branch, with the permanent
        # shock at its infimum, reaches next period's limit, and the permanent law
        # approaches its infimum with order 2, above CRRA - 1 and not above CRRA at
        # the default CRRA of 2: there the
        # expectation of marginal value diverges, and the natural limit binds with
        # consumption zero at it, but the expectation of value converges, so the
        # end-of-period value at the limit is finite. A truncated law, whose density
        # is positive at its end, leaves it minus infinity
        agent = IndShockConsumerType(
            cycles=1,
            T_cycle=1,
            CubicBool=True,
            vFuncBool=True,
            BoroCnstArt=None,
            IncShkExpectation="continuous",
            **JUMPLESS_TAILS,
        )
        laws = get_income_shock_laws(agent.IncShkDstn[0])
        R, G, rho = agent.Rfree[0], agent.PermGroFac[0], agent.CRRA
        for law, value_diverges in (
            (laws[0], False),
            (LogNormalTruncatedMeanOne.from_sd_log(0.1, -3.0, 3.0), True),
        ):
            with self.subTest(permanent=type(law).__name__):
                at_limit = (law,) + tuple(laws[1:])
                worst = _law_end(law, upper=False)[0]
                a = (0.0 - laws[3]) * G * worst / R  # lands on a limit of zero
                self.assertTrue(_reaches_natural_limit(a, 0.0, at_limit, R, G, rho))
                self.assertEqual(
                    _reaches_natural_limit(a, 0.0, at_limit, R, G, rho - 1.0),
                    value_diverges,
                )
        # The value's level, not only its slope: this period's value against the
        # Bellman operator applied to next period's, near the limit and above it.
        # With the end-of-period value pinned at minus infinity at the limit, the
        # error near it was of order one in the inverse value after three periods
        # (2e-3 to 4.6), and grew from one period to the next
        for benefit in (0.3, 0.6):
            with self.subTest(IncUnemp=benefit):
                agent = IndShockConsumerType(
                    cycles=1,
                    T_cycle=1,
                    CubicBool=True,
                    vFuncBool=True,
                    BoroCnstArt=None,
                    IncUnemp=benefit,
                    IncShkExpectation="continuous",
                    **JUMPLESS_TAILS,
                )
                agent.cycles = 3
                agent.solve()
                now, after = agent.solution[0], agent.solution[1]
                d = np.r_[np.geomspace(1e-6, 0.1, 6), np.linspace(0.5, 10.0, 5)]
                m = now.mNrmMin + d
                c = now.cFunc(m)
                E0 = _calc_BegOfPrd_IndShock_continuous(
                    m - c, after, agent.IncShkDstn[0], R, G, rho, False, True, 12, 12
                )[2]
                Tv = (
                    c ** (1.0 - rho) / (1.0 - rho)
                    + agent.DiscFac * agent.LivPrb[0] * E0
                )

                def inverse(v):
                    return ((1.0 - rho) * v) ** (1.0 / (1.0 - rho))

                # within 6e-5 here
                np.testing.assert_allclose(
                    inverse(now.vFunc(m)), inverse(Tv), rtol=5e-4
                )

    def test_no_unemployment_income(self):
        # Zero income in unemployment puts the natural limit at zero, where the
        # artificial constraint is too: no kink, with or without that constraint
        for constraint in (0.0, None):
            with self.subTest(BoroCnstArt=constraint):
                agent = IndShockConsumerType(
                    cycles=0,
                    CubicBool=True,
                    IncUnemp=0.0,
                    BoroCnstArt=constraint,
                    IncShkExpectation="continuous",
                    **BOUNDS,
                )
                agent.solve()
                self.assertLess(agent.solution_distance, agent.tolerance)
                solution = agent.solution[0]
                self.assertEqual(solution.mNrmMin, 0.0)
                self.assertEqual(solution.mNrmKinks.size, 0)
                self.assertEqual(float(solution.cFunc(np.array([0.0]))[0]), 0.0)
                m = np.r_[np.geomspace(1e-6, 0.1, 61), np.linspace(0.11, 20.0, 2000)]
                self.assertTrue(np.all(solution.cFunc(m) > 0.0))
                self.assertLessEqual(largest_rise(solution.cFunc, m), 1e-12)

    def test_limit_with_finite_marginal_value(self):
        # Untruncated laws reach their infimum, zero, only through a tail: at the
        # natural limit, zero, the marginal value stays finite, so the limit binds
        # as a constraint with a kink, and it makes no difference that no
        # artificial constraint is set
        solutions = []
        for constraint in (0.0, None):
            agent = IndShockConsumerType(
                cycles=1,
                T_cycle=1,
                CubicBool=True,
                BoroCnstArt=constraint,
                IncShkExpectation="continuous",
            )
            agent.cycles = 3
            agent.solve()
            solutions.append(agent.solution[0])
        for solution in solutions:
            self.assertEqual(solution.mNrmMin, 0.0)
            self.assertEqual(solution.mNrmKinks.size, 1)
            self.assertGreater(solution.mNrmKinks[0], 0.0)
        m = np.linspace(0.0, 20.0, 2001)
        np.testing.assert_array_equal(solutions[0].cFunc(m), solutions[1].cFunc(m))

    def test_natural_limit_rule(self):
        # The marginal value at the lowest assets is infinite where the worst
        # landing carries a mass, or a positive density on each of the d laws
        # setting it with CRRA >= d; a law reaching its end through a tail adds none
        R, G = 1.03, 1.01
        perm = LogNormalTruncatedMeanOne.from_sd_log(0.1, -3.0, 3.0)
        tran = LogNormalTruncatedMeanOne.from_sd_log(0.1, -3.0, 3.0)

        def reaches(laws, CRRA, above=0.0):
            # at the natural limit for the laws' worst incomes, plus above
            psi, theta = laws[0], laws[1]
            psi_lo = (
                float(np.min(psi.atoms))
                if hasattr(psi, "atoms")
                else float(psi.infimum[0])
            )
            theta_lo = min(
                float(theta.infimum[0]), laws[3] if laws[2] > 0.0 else np.inf
            )
            limit = -theta_lo * G * psi_lo / R
            return _reaches_natural_limit(limit + above, 0.0, laws, R, G, CRRA)

        # unemployment, a point, at the worst: a density in the permanent shock
        self.assertTrue(reaches((perm, tran, 0.05, 0.3), 2.0))
        self.assertFalse(reaches((perm, tran, 0.05, 0.3), 2.0, above=0.01))
        # a discrete permanent law holds its end atom with mass, so the
        # unemployed land there with positive probability, whatever CRRA
        atoms = perm.discretize(7, method="equiprobable")
        self.assertTrue(reaches((atoms, tran, 0.05, 0.3), 0.5))
        # no unemployment: densities on both laws, d = 2
        self.assertTrue(reaches((perm, tran, 0.0, 0.0), 2.0))
        self.assertFalse(reaches((perm, tran, 0.0, 0.0), 1.5))
        # a mixture truncated as a whole holds the density of the component
        # reaching its end; untruncated below, it reaches zero through tails
        mixed = Mixture(
            [0.4, 0.6],
            [
                LogNormalTruncated(-0.05, 0.1, bot=-3.0, top=3.0),
                LogNormalTruncated(0.05, 0.1, bot=-3.0, top=3.0),
            ],
        )
        self.assertTrue(reaches((perm, mixed, 0.0, 0.0), 2.0))
        tails = Mixture([1.0], [LogNormalTruncated(0.0, 0.1, bot=None, top=3.0)])
        self.assertFalse(reaches((perm, tails, 0.0, 0.0), 2.0))
        # untruncated lognormals approach zero only through their tails
        plain = LogNormalMeanOne(0.1)
        self.assertFalse(
            _reaches_natural_limit(0.0, 0.0, (plain, plain, 0.0, 0.0), R, G, 2.0)
        )

    def test_law_ends(self):
        # How a law holds probability at each end, read through its quantile
        # function and density alone: with a positive density (1) where it is
        # truncated or uniform, with one rising like a power from zero (2, 3) at
        # the jumpless law's bound, the ends of a beta law and a gamma law's zero,
        # not at all (None) where it reaches the end through a tail; a mass (0)
        # at a point, and at the infimum where PointMassAtInfimum puts one
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
            "mass at a jumpless law's infimum": (
                PointMassAtInfimum(jumpless, 0.02),
                0,
                None,
            ),
        }
        for name, (law, lower, upper) in cases.items():
            with self.subTest(law=name):
                self.assertEqual(_law_end(law, upper=False)[1], lower)
                self.assertEqual(_law_end(law, upper=True)[1], upper)

    def test_natural_limit_rule_for_other_laws(self):
        # The same rule for laws that are not lognormal: a uniform law holds each
        # end with a positive density, so it adds 1 to the d laws that set the
        # worst landing, and a beta(2, 3) law, rising linearly from its lowest
        # value, adds 2
        R, G = 1.03, 1.01
        perm, tran = Uniform(0.85, 1.15), Uniform(0.6, 1.4)
        rising = QuantileLaw(beta(2.0, 3.0, loc=0.4, scale=1.5))

        def reaches(laws, CRRA, above=0.0):
            # at the natural limit for the laws' worst incomes, plus above
            theta_lo = min(_bounds(laws[1])[0], laws[3] if laws[2] > 0.0 else np.inf)
            limit = -theta_lo * G * _bounds(laws[0])[0] / R
            return _reaches_natural_limit(limit + above, 0.0, laws, R, G, CRRA)

        # unemployment, a point, at the worst: the permanent law alone, d = 1
        self.assertTrue(reaches((perm, tran, 0.05, 0.3), 1.0))
        self.assertFalse(reaches((perm, tran, 0.05, 0.3), 0.9))
        self.assertFalse(reaches((perm, tran, 0.05, 0.3), 2.0, above=0.01))
        # no unemployment: both laws, d = 2
        self.assertTrue(reaches((perm, tran, 0.0, 0.0), 2.0))
        self.assertFalse(reaches((perm, tran, 0.0, 0.0), 1.9))
        # the rising law, as the transitory, d = 3; as the permanent, with
        # unemployment, d = 2
        self.assertTrue(reaches((perm, rising, 0.0, 0.0), 3.0))
        self.assertFalse(reaches((perm, rising, 0.0, 0.0), 2.9))
        self.assertTrue(reaches((rising, tran, 0.05, 0.3), 2.0))
        self.assertFalse(reaches((rising, tran, 0.05, 0.3), 1.9))

    def test_solution_with_other_laws(self):
        # IndShock's income from uniform laws: the solver reads the laws behind
        # the atoms through their quantile functions, CDFs and densities
        agent = IndShockConsumerType(
            cycles=0, CubicBool=True, IncShkExpectation="continuous", **JUMPLESS_TAILS
        )
        agent.IncShkDstn = [uniform_income(0.05)]
        agent.solve()
        self.assertLess(agent.solution_distance, agent.tolerance)
        solution = agent.solution[0]
        kink = solution.mNrmKinks
        self.assertEqual(kink.size, 1)
        self.assertAlmostEqual(float(solution.cFunc(kink)[0]), kink[0], places=14)
        m = np.linspace(kink[0] + 1e-6, 20.0, 20001)
        self.assertLessEqual(largest_rise(solution.cFunc, m), 1e-12)

    def test_natural_constraint_with_other_laws(self):
        # With no artificial constraint, the natural one binds with no kink where
        # CRRA >= d, the number of uniform laws setting the worst landing: d = 1
        # with unemployment at the worst, d = 2 without it; for CRRA below d, the
        # natural limit binds as a constraint, with a kink
        for UnempPrb, CRRA, kinked in (
            (0.05, 2.0, False),
            (0.0, 2.0, False),
            (0.0, 1.5, True),
        ):
            with self.subTest(UnempPrb=UnempPrb, CRRA=CRRA):
                solutions = []
                for nodes in ((4, 4), (6, 6)) if not kinked else ((4, 4),):
                    agent = IndShockConsumerType(
                        cycles=1,
                        T_cycle=1,
                        CubicBool=True,
                        BoroCnstArt=None,
                        CRRA=CRRA,
                        IncShkExpectation="continuous",
                        PermShkQuadCount=nodes[0],
                        TranShkQuadCount=nodes[1],
                        **JUMPLESS_TAILS,
                    )
                    agent.IncShkDstn = [uniform_income(UnempPrb)]
                    agent.cycles = 5
                    agent.solve()
                    solutions.append(agent.solution[0])
                solution = solutions[0]
                limit = solution.mNrmMin
                self.assertLess(limit, 0.0)
                self.assertEqual(solution.mNrmKinks.size, 1 if kinked else 0)
                self.assertEqual(float(solution.cFunc(np.array([limit]))[0]), 0.0)
                m = (
                    limit
                    + np.r_[np.geomspace(1e-6, 0.1, 61), np.linspace(0.11, 20.0, 2000)]
                )
                self.assertTrue(np.all(solution.cFunc(m) > 0.0))
                self.assertLessEqual(largest_rise(solution.cFunc, m), 1e-12)
                if kinked:  # on the constraint, c = m - mNrmMin, up to the kink
                    below = np.linspace(limit, solution.mNrmKinks[0], 11)
                    np.testing.assert_allclose(
                        solution.cFunc(below), below - limit, rtol=0.0, atol=1e-14
                    )
                else:  # accurate at the default nodes, against many more
                    np.testing.assert_allclose(
                        solution.cFunc(m), solutions[1].cFunc(m), rtol=0.0, atol=1e-7
                    )

    def test_expectation_with_a_law_known_only_by_its_quantile_function(self):
        # A narrow permanent shock and a wide transitory one, both truncated at
        # three standard deviations, with unemployment of 0.044 paying 0.6: given
        # as truncated lognormals, integrated in the coordinate they supply, and
        # as laws known only by their quantile functions, CDFs and densities,
        # integrated in their normal scores, two quadratures of the same
        # integrals, they agree to the rule's accuracy on next period's policy as
        # this solver makes it. Measured: E1 within 1.4e-5 at four nodes a piece
        # and 1.3e-6 at eight; E2, whose integrand has a kink at each of next
        # period's knots (the MPC of a cubic), within 5e-4 and 9.4e-5
        agent = IndShockConsumerType(
            cycles=1,
            T_cycle=1,
            CubicBool=True,
            IncShkExpectation="continuous",
            UnempPrb=0.044,
            IncUnemp=0.6,
            PermShkStd=[0.001**0.5],
            TranShkStd=[0.132**0.5],
            **TRUNCATED_BOUNDS,
        )
        agent.cycles = 5
        agent.solve()
        solution = agent.solution[0]
        perm, tran, p, x = get_income_shock_laws(agent.IncShkDstn[0])
        R, G = agent.Rfree[0], agent.PermGroFac[0]
        a = np.r_[0.0, np.asarray(agent.aXtraGrid)]
        point = DiscreteDistribution([1.0], [x])
        for name, laws in (
            ("employed", (perm, tran, QuantileLaw(perm), QuantileLaw(tran))),
            ("unemployed", (perm, point, QuantileLaw(perm), point)),
        ):
            for nodes, rtol in ((4, (5e-5, 2e-3)), (8, (5e-6, 3e-4))):
                with self.subTest(branch=name, nodes=nodes):
                    E = [
                        calc_BegOfPrd_continuous(
                            a,
                            solution,
                            P,
                            T,
                            R,
                            G,
                            agent.CRRA,
                            True,
                            PermShkQuadCount=nodes,
                            TranShkQuadCount=nodes,
                        )
                        for P, T in (laws[:2], laws[2:])
                    ]
                    np.testing.assert_allclose(E[1][0], E[0][0], rtol=rtol[0])
                    np.testing.assert_allclose(E[1][1], E[0][1], rtol=rtol[1])

    def test_solution_with_a_law_known_only_by_its_quantile_function(self):
        # The same laws given to the solver both ways, truncated ones: two
        # quadratures of the same integrals, so the solutions agree to the rule's
        # accuracy, measured here within 2.7e-6 in consumption and 1.5e-4 in the
        # MPC
        params = dict(
            cycles=1,
            T_cycle=1,
            CubicBool=True,
            IncShkExpectation="continuous",
            UnempPrb=0.044,
            IncUnemp=0.6,
            PermShkStd=[0.001**0.5],
            TranShkStd=[0.132**0.5],
            **TRUNCATED_BOUNDS,
        )
        solutions = []
        for known in ("family", "quantile function"):
            agent = IndShockConsumerType(**params)
            if known == "quantile function":
                agent.IncShkDstn = [quantile_income(agent.IncShkDstn[0])]
            agent.cycles = 5
            agent.solve()
            solutions.append(agent.solution[0])
        m = np.linspace(0.0, 20.0, 2001)
        c, mpc = solutions[0].cFunc.eval_with_derivative(m)
        c_q, mpc_q = solutions[1].cFunc.eval_with_derivative(m)
        np.testing.assert_allclose(c_q, c, rtol=0.0, atol=1e-5)
        np.testing.assert_allclose(mpc_q, mpc, rtol=0.0, atol=5e-4)

    def test_errors(self):
        with self.assertRaises(ValueError):
            IndShockConsumerType(IncShkExpectation="exact").solve()
        with self.assertRaises(NotImplementedError):  # KinkedR's solver has no option
            KinkedRconsumerType(IncShkExpectation="continuous").solve()
        # nor PrefShock's, whose pre_solve does not call IndShockConsumerType's
        with self.assertRaises(NotImplementedError):
            PrefShockConsumerType(IncShkExpectation="continuous").solve()

"""
Tests for IndShockConsumerType's option IncShkExpectation: "atoms", the default,
and "continuous", which takes the expectation over the income shocks against
their laws with solve_one_period_ConsIndShock_continuous.
"""

import pickle
import unittest
import warnings
from copy import deepcopy
from unittest import mock

import numpy as np
from scipy.stats import beta

from HARK.Calibration.Income import IncomeProcesses
from HARK.ConsumptionSaving.ConsIndShockModel import (
    KINK_NEAR_ASSETS,
    TRANSITION_SPLIT,
    ConsumerSolution,
    IndShockConsumerType,
    KinkedRconsumerType,
    _calc_BegOfPrd_IndShock_continuous,
    _mpc_transition,
    _natural_limit_assets,
    _reaches_natural_limit,
    _refine_for_concavity,
    _worst_income_mass,
    calc_BegOfPrd_continuous,
    calc_boro_const_nat,
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
    LogNormalTruncatedJumplessMeanOne,
    LogNormalTruncatedMeanOne,
    LogNormalTruncated,
    PointMassAtInfimum,
    Uniform,
    add_discrete_outcome_constant_mean,
    combine_indep_dstns,
    law_bounds,
    law_end,
)
from HARK.interpolation import CubicInterp, LinearInterp
from HARK.utilities import get_arg_names

# The jumpless laws, bounded three standard deviations below, on which these
# tests run; with the continuous expectation HARK's default is the jumpless one
# (testDefaultLaws, below)
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
TRUNCATED_TAILS = {"PermShk_tails": "truncated", "TranShk_tails": "truncated"}
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
        # no concavity violation, the plain cubic through the gridpoints, with no
        # refinement needed
        m = np.linspace(kink[0] + 1e-6, 20.0, 200001)
        self.assertLessEqual(largest_rise(solution.cFunc, m), 1e-12)
        refinement = solution.ConcavityRefinement
        self.assertFalse(np.any(refinement["depth"]) or np.any(refinement["level"]))

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
        # Over the atoms the cubic is refused (check_cubic_pairing); the linear
        # solution is the atoms' own, bit for bit
        agent = deepcopy(self.agent)
        agent.IncShkExpectation = "atoms"
        with self.assertRaises(ValueError):
            agent.solve()
        self.assertIs(agent.solve_one_period, solve_one_period_ConsIndShock)
        agent.CubicBool = False
        agent.solve()
        fresh = IndShockConsumerType(cycles=0, **BOUNDS)
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

    def test_worst_income_with_no_benefit(self):
        # With no unemployment benefit, income in that event is zero whatever the
        # permanent shock, so the worst income has the event's probability and the
        # MPC at the natural limit is below one, as over the atoms
        for expectation in ("atoms", "continuous"):
            with self.subTest(expectation=expectation):
                agent = IndShockConsumerType(
                    cycles=1,
                    IncUnemp=0.0,
                    BoroCnstArt=None,
                    IncShkExpectation=expectation,
                    **BOUNDS,
                )
                agent.solve()
                solution = agent.solution[0]
                PatFac = (agent.Rfree[0] * agent.DiscFac * agent.LivPrb[0]) ** (
                    1.0 / agent.CRRA
                ) / agent.Rfree[0]
                MPCmax = 1.0 / (1.0 + agent.UnempPrb ** (1.0 / agent.CRRA) * PatFac)
                self.assertEqual(solution.mNrmMin, 0.0)
                self.assertAlmostEqual(solution.MPCmax, MPCmax, places=12)

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
        # concave, the plain cubic, where a few segments' expectations were taken
        # more accurately (and gridpoints added where that was not enough)
        self.assertLessEqual(largest_rise(solution.cFunc, m), 1e-12)
        self.assertTrue(np.any(solution.ConcavityRefinement["level"] > 0))
        # its points land in the transition of the solution itself
        first = agent.aXtraGrid[0]
        self.assertGreaterEqual(
            knots_in_transition(solution, True, first), TRANSITION_SPLIT[2]
        )
        # accurate at the default nodes, four a piece, against six: with the
        # jumpless laws, within 1.2e-7 at worst, at the top of the range, where
        # six are within 1e-8 of eight
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
        # infima: on one law, the continuous expectation has the one 7 and 11
        # atoms have, in every period of a finite horizon
        # with the bounds given: over the atoms a bound left None is no bound
        for tails in (TRUNCATED_BOUNDS, BOUNDS):
            limits = []
            for kwds in [
                dict(PermShkCount=7, TranShkCount=7),
                dict(PermShkCount=11, TranShkCount=11),
                dict(IncShkExpectation="continuous"),
            ]:
                agent = IndShockConsumerType(
                    cycles=2, BoroCnstArt=None, **tails, **kwds
                )
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

    def test_refinement_for_concavity(self):
        # A consumption function whose MPC falls sharply from 1 to 0.1 near
        # d = 0.3: through gridpoints with its exact levels and slopes the data are
        # concave, but on the coarse grid the cubic is not, until the grid is
        # bisected where it fails; the grid then settles, started from its state
        nodes = (4, 4)

        def exact(a):
            a = np.asarray(a, dtype=float)
            c = 0.1 * a + 0.9 * (
                a
                - 0.02
                * (np.logaddexp(0.0, (a - 0.3) / 0.02) - np.logaddexp(0.0, -15.0))
            )
            mpc = 0.1 + 0.9 / (1.0 + np.exp((a - 0.3) / 0.02))
            return a, c, mpc

        def expect(a, PermNodes, TranNodes):
            return np.asarray(a, dtype=float), np.asarray(a, dtype=float), None

        def endogenous(a, E1, E2):
            return exact(a)

        def violations(a):
            m, c, mpc = exact(a)
            x, y, d = np.r_[0.0, m], np.r_[0.0, c], np.r_[1.0, mpc]
            S = np.diff(y) / np.diff(x)
            bad = (S < (d[:-1] + 2.0 * d[1:]) / 3.0 - 1e-12) | (
                S > (2.0 * d[:-1] + d[1:]) / 3.0 + 1e-12
            )
            return np.count_nonzero(bad[1:])

        base = np.geomspace(0.01, 10.0, 12)
        self.assertGreater(violations(base), 0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            a, E, state = _refine_for_concavity(
                base,
                expect(base, *nodes),
                None,
                expect,
                endogenous,
                nodes,
                1.0,
                1.0,
                0.0,
            )
        self.assertEqual(violations(a), 0)
        self.assertGreater(a.size, base.size)
        self.assertTrue(np.all(np.diff(a) > 0.0))
        np.testing.assert_array_equal(E[0], a)
        self.assertTrue(np.any(state["depth"] > 0))
        # started from its own state, it adds nothing more
        a2, _, state2 = _refine_for_concavity(
            base, expect(base, *nodes), state, expect, endogenous, nodes, 1.0, 1.0, 0.0
        )
        np.testing.assert_array_equal(a2, a)
        np.testing.assert_array_equal(state2["depth"], state["depth"])

    def test_refinement_warns_at_its_end(self):
        # Where the data themselves are not concave, as with an MPC that rises,
        # neither more nodes nor more gridpoints help: the refinement warns once
        # it has run through its steps, rather than looping
        def expect(a, PermNodes, TranNodes):
            return np.asarray(a, dtype=float), np.asarray(a, dtype=float), None

        def endogenous(a, E1, E2):
            a = np.asarray(a, dtype=float)
            return a, 0.5 * a + 0.05 * np.sin(4.0 * a), 0.5 + 0.2 * np.cos(4.0 * a)

        base = np.linspace(0.1, 5.0, 8)
        with self.assertWarns(UserWarning):
            _refine_for_concavity(
                base,
                expect(base, 4, 4),
                None,
                expect,
                endogenous,
                (4, 4),
                0.0,
                0.7,
                0.0,
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
                worst = law_end(law, upper=False)[0]
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
        # as a constraint with a kink. Under the continuous expectation the laws
        # are untruncated only with infinite bounds; here with no unemployment
        # event, since an employed law unbounded below would reach below
        # unemployment income. With an artificial constraint at the limit, the
        # edge, the model is the artificial configuration and solves; with none,
        # its worst transitory shock carries no probability, which the continuous
        # expectation refuses where the natural limit binds
        unbounded = {
            "PermShk_inf": -np.inf,
            "PermShk_sup": np.inf,
            "TranShk_inf": -np.inf,
            "TranShk_sup": np.inf,
            "UnempPrb": 0.0,
        }
        agents = {
            constraint: IndShockConsumerType(
                cycles=1,
                T_cycle=1,
                CubicBool=True,
                BoroCnstArt=constraint,
                IncShkExpectation="continuous",
                **unbounded,
            )
            for constraint in (0.0, None)
        }
        for agent in agents.values():
            agent.cycles = 3
        agents[0.0].solve()
        solution = agents[0.0].solution[0]
        self.assertEqual(solution.mNrmMin, 0.0)
        self.assertEqual(solution.mNrmKinks.size, 1)
        self.assertGreater(solution.mNrmKinks[0], 0.0)
        with self.assertRaisesRegex(ValueError, "worst transitory shock to carry"):
            agents[None].solve()

    def test_power_segment_above_a_kink_at_the_natural_limit(self):
        # Where the natural constraint binds, the lowest assets still land on next
        # period's minimum cash-on-hand, and the MPC falls from its value at the
        # kink like a power of the distance. With k the power with which the
        # permanent shock's probability approaches its lowest value (1 for the
        # truncated law, 2 for the jumpless one, the default here): for CRRA in
        # [k - 1, k) the
        # MPC leaves one with no step; for CRRA in [k - 2, k - 1) it steps down at
        # the kink and then falls like a power. No cubic follows either, so the
        # segment above the kink takes the power's shape, the asset grid takes
        # points close above the constraint, and the solution is concave
        cases = (
            ({}, 1.5, True),
            ({}, 1.2, True),
            ({}, 0.8, False),
            (TRUNCATED_TAILS, 0.8, True),
            (TRUNCATED_TAILS, 0.5, True),
        )
        for tails, CRRA, no_step in cases:
            with self.subTest(tails=tails, CRRA=CRRA):
                agent = IndShockConsumerType(
                    cycles=1,
                    T_cycle=1,
                    CubicBool=True,
                    BoroCnstArt=None,
                    CRRA=CRRA,
                    IncShkExpectation="continuous",
                    **tails,
                )
                agent.cycles = 6
                with warnings.catch_warnings():
                    warnings.simplefilter("error")  # no segment left not concave
                    agent.solve()
                solution = agent.solution[0]
                unconstrained = solution.cFunc.functions[0]
                x = np.asarray(unconstrained.x_list)
                floor, kink = solution.mNrmMin, solution.mNrmKinks[0]
                # the constraint binds: positive consumption at the lowest assets
                self.assertGreater(kink, floor + 0.1)
                self.assertEqual(x[1], kink)
                # the MPC above the kink, and the segment that carries its fall
                self.assertEqual(list(unconstrained.power_segments), [1])
                if no_step:
                    self.assertEqual(solution.MPCaboveKink, 1.0)
                else:
                    self.assertLess(solution.MPCaboveKink, 0.9)
                # the grid's points close above the constraint, in assets
                assets = x[2:] - np.asarray(unconstrained.y_list)[2:] - floor
                near = agent.aXtraGrid[0] * np.array(KINK_NEAR_ASSETS)
                for point in near:
                    self.assertLess(np.min(np.abs(assets / point - 1.0)), 1e-6)
                # increasing and concave, from the kink up
                m = np.unique(
                    np.r_[
                        kink + np.geomspace(1e-12, 1.0, 20001),
                        np.linspace(kink, 30.0, 30001),
                    ]
                )
                self.assertLessEqual(largest_rise(solution.cFunc, m), 1e-12)
                mpc = solution.cFunc.derivative(m)
                self.assertTrue(np.all(mpc > 0.0) and np.all(mpc <= 1.0))
                self.assertGreater(mpc[1], 0.99 if no_step else 0.5)
                # the line c = m - mNrmMin up to the kink
                below = np.linspace(floor, kink, 11)
                np.testing.assert_allclose(
                    solution.cFunc(below), below - floor, rtol=0, atol=1e-12
                )

    def test_no_power_segment_elsewhere(self):
        # The power segment belongs to a kink at the natural limit. Where the
        # limit does not bind there is no kink, and above an artificial
        # constraint's kink every landing point stays off next period's minimum
        for constraint, kinks in ((None, 0), (0.0, 1), (-0.2, 1)):
            with self.subTest(BoroCnstArt=constraint):
                agent = IndShockConsumerType(
                    cycles=1,
                    T_cycle=1,
                    CubicBool=True,
                    BoroCnstArt=constraint,
                    IncShkExpectation="continuous",
                )
                agent.cycles = 4
                agent.solve()
                solution = agent.solution[0]
                self.assertEqual(solution.mNrmKinks.size, kinks)
                self.assertFalse(
                    getattr(solution.cFunc.functions[0], "power_segments", {})
                )

    def test_worst_transitory_point_mass(self):
        # Where the natural limit binds (no artificial constraint, or one below
        # it), the continuous expectation needs the worst transitory shock to be a
        # point mass; the edge and an artificial constraint above the limit are
        # the artificial configuration; the solver over atoms is not checked
        def solve(**kwds):
            agent = IndShockConsumerType(cycles=1, T_cycle=1, **kwds)
            agent.cycles = 1
            agent.solve()
            return agent

        continuous = {"IncShkExpectation": "continuous"}
        # an unemployment event, with a positive benefit or none
        for benefit in (0.3, 0.0):
            solve(BoroCnstArt=None, IncUnemp=benefit, **continuous)
        # no unemployment event: the jumpless employed law has no mass at its end
        for constraint in (None, -5.0):
            with self.subTest(BoroCnstArt=constraint):
                with self.assertRaisesRegex(
                    ValueError, "worst transitory shock to carry"
                ):
                    solve(BoroCnstArt=constraint, UnempPrb=0.0, **continuous)
        # a constraint at the natural limit from the laws' infima, the edge, as the
        # solver computes it from the terminal period, or above it
        agent = solve(BoroCnstArt=0.0, UnempPrb=0.0, **continuous)
        limit = calc_boro_const_nat(
            0.0,
            agent.IncShkDstn[0],
            agent.Rfree[0],
            agent.PermGroFac[0],
            use_infimum=True,
        )
        self.assertLess(limit, 0.0)
        solve(BoroCnstArt=limit, UnempPrb=0.0, **continuous)
        # over atoms the same model solves
        solve(BoroCnstArt=None, UnempPrb=0.0)

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

    def test_law_end_of_a_point_mass_at_the_infimum(self):
        # A point mass at a law's infimum (PointMassAtInfimum) is a mass (0) at
        # that end; the other laws' ends are tested with law_end itself, in
        # tests/test_expected_continuous.py
        law = PointMassAtInfimum(
            LogNormalTruncatedJumplessMeanOne.from_sd_log(0.1), 0.02
        )
        self.assertEqual(law_end(law, upper=False)[1], 0)
        self.assertIsNone(law_end(law, upper=True)[1])

    def test_point_mass_at_the_natural_limit(self):
        # A permanent shock with a point mass at its infimum (PermShkMinPrb)
        # makes the worst income, unemployment with the permanent shock at its
        # point, have positive probability: at the natural limit consumption falls
        # to zero with no kink at any CRRA, and the MPC there is the bound the
        # solution reports, which follows the worst income's probability period
        # by period
        mass = 0.02
        for CRRA in (1.5, 2.0, 5.0):
            for UnempPrb in (0.05,):
                with self.subTest(CRRA=CRRA, UnempPrb=UnempPrb):
                    agent = IndShockConsumerType(
                        cycles=1,
                        T_cycle=1,
                        CRRA=CRRA,
                        UnempPrb=UnempPrb,
                        BoroCnstArt=None,
                        IncShkExpectation="continuous",
                        PermShkMinPrb=mass,
                    )
                    agent.cycles = 3
                    agent.solve()
                    solution = agent.solution[0]
                    laws = get_income_shock_laws(agent.IncShkDstn[0])
                    worst = UnempPrb * mass
                    self.assertAlmostEqual(_worst_income_mass(laws), worst, places=12)
                    self.assertEqual(get_mNrm_kinks(solution).size, 0)
                    # three periods before the last, 1 / MPC = 1 + q + q^2 + q^3,
                    # with q = (DiscFac LivPrb Rfree p)^(1 / CRRA) / Rfree
                    R = agent.Rfree[0]
                    q = (agent.DiscFac * agent.LivPrb[0] * R * worst) ** (1.0 / CRRA)
                    q /= R
                    bound = 1.0 / (1.0 + q + q**2 + q**3)
                    self.assertAlmostEqual(solution.MPCmax, bound, places=10)
                    just_above = np.array([solution.mNrmMin + 1e-7])
                    self.assertAlmostEqual(
                        solution.cFunc.derivative(just_above)[0], bound, places=4
                    )
        # laws that hold no probability at their lowest values put none on the
        # worst income, unless unemployment pays nothing
        for tails in ("truncated", "jumpless"):
            for benefit, worst in ((0.3, 0.0), (0.0, 0.05)):
                with self.subTest(tails=tails, IncUnemp=benefit):
                    agent = IndShockConsumerType(
                        PermShk_tails=tails,
                        TranShk_tails=tails,
                        IncUnemp=benefit,
                        IncShkExpectation="continuous",
                    )
                    laws = get_income_shock_laws(agent.IncShkDstn[0])
                    self.assertEqual(_worst_income_mass(laws), worst)

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
            theta_lo = min(law_bounds(laws[1])[0], laws[3] if laws[2] > 0.0 else np.inf)
            limit = -theta_lo * G * law_bounds(laws[0])[0] / R
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
        # with unemployment at the worst. Without it, d = 2, the worst transitory
        # shock carries no probability, which the continuous expectation refuses
        # where the natural limit binds (the rule itself for d = 2:
        # test_natural_limit_rule)
        for CRRA in (2.0, 1.5):
            with self.subTest(UnempPrb=0.0, CRRA=CRRA):
                agent = IndShockConsumerType(
                    cycles=1,
                    T_cycle=1,
                    CubicBool=True,
                    BoroCnstArt=None,
                    CRRA=CRRA,
                    IncShkExpectation="continuous",
                    **JUMPLESS_TAILS,
                )
                agent.IncShkDstn = [uniform_income(0.0)]
                with self.assertRaisesRegex(
                    ValueError, "worst transitory shock to carry"
                ):
                    agent.solve()
        for UnempPrb, CRRA, kinked in ((0.05, 2.0, False),):
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

    def test_rescaled_law_keeps_its_breakpoints(self):
        # The employed transitory law, rescaled for unemployment's mean, keeps
        # its law's breakpoints, scaled, so the expectation cuts there; here the
        # jumpless laws, given breakpoints two standard deviations from the median
        class WithBreakpoints(LogNormalTruncatedJumplessMeanOne):
            def breakpoints(self):
                return np.exp(self.mu + self.sigma * np.array([-2.0, 2.0]))

        with mock.patch.dict(
            IncomeProcesses._SHOCK_TAILS, {"jumpless": WithBreakpoints}
        ):
            agent = IndShockConsumerType(
                cycles=0,
                IncShkExpectation="continuous",
                TranShk_inf=-3.5,
                **JUMPLESS_TAILS,
            )
        perm, tran, p, x = get_income_shock_laws(agent.IncShkDstn[0])
        law = agent.IncShkDstn[0].limit["dist"][1].limit["dist"].limit["dist"]
        self.assertIsInstance(law, WithBreakpoints)
        scale = (1.0 - p * x) / (1.0 - p)
        self.assertEqual(tran.breakpoints().size, 2)
        np.testing.assert_allclose(tran.breakpoints(), scale * law.breakpoints())
        np.testing.assert_allclose(
            perm.breakpoints(), np.exp(perm.mu + perm.sigma * np.array([-2.0, 2.0]))
        )

    def test_errors(self):
        with self.assertRaises(ValueError):
            IndShockConsumerType(IncShkExpectation="exact").solve()
        with self.assertRaises(NotImplementedError):  # KinkedR's solver has no option
            KinkedRconsumerType(IncShkExpectation="continuous").solve()
        # nor PrefShock's, whose pre_solve does not call IndShockConsumerType's
        with self.assertRaises(NotImplementedError):
            PrefShockConsumerType(IncShkExpectation="continuous").solve()


class testDefaultLaws(unittest.TestCase):
    """
    With the continuous expectation HARK's default income shocks are jumpless:
    lognormals truncated below at three standard deviations whose density is
    zero at the bound. The continuous expectation solves on them, and on the
    truncated lognormals that are the atoms' default, with the artificial
    constraint and with the natural one, and the cubic is concave.
    """

    laws = (
        ({}, LogNormalTruncatedJumplessMeanOne),
        (TRUNCATED_TAILS, LogNormalTruncatedMeanOne),
    )

    def test_the_default_is_jumpless(self):
        agent = IndShockConsumerType(cycles=0, IncShkExpectation="continuous")
        for law in get_income_shock_laws(agent.IncShkDstn[0])[:2]:
            law = getattr(law, "law", law)  # the employed law is rescaled
            self.assertIsInstance(law, LogNormalTruncatedJumplessMeanOne)
            self.assertEqual((law.bot, law.top), (-3.0, np.inf))
        # the atoms' default stays truncated
        atoms = IndShockConsumerType(cycles=0)
        self.assertIsInstance(
            get_income_shock_laws(atoms.IncShkDstn[0])[0], LogNormalTruncatedMeanOne
        )
        # the default has no upper bound, so one given is refused
        with self.assertRaises(ValueError):
            IndShockConsumerType(
                cycles=0, IncShkExpectation="continuous", PermShk_sup=3.0
            )

    def test_artificial_constraint(self):
        for tails, law in self.laws:
            with self.subTest(law=law.__name__):
                agent = IndShockConsumerType(
                    cycles=0, CubicBool=True, IncShkExpectation="continuous", **tails
                )
                self.assertIsInstance(
                    get_income_shock_laws(agent.IncShkDstn[0])[0], law
                )
                agent.solve()
                self.assertLess(agent.solution_distance, agent.tolerance)
                solution = agent.solution[0]
                kink = solution.mNrmKinks[0]
                m = np.linspace(kink + 1e-6, 20.0, 200001)
                self.assertLessEqual(largest_rise(solution.cFunc, m), 1e-12)

    def test_natural_constraint(self):
        for tails, law in self.laws:
            with self.subTest(law=law.__name__):
                agent = IndShockConsumerType(
                    cycles=0,
                    CubicBool=True,
                    BoroCnstArt=None,
                    IncShkExpectation="continuous",
                    **tails,
                )
                self.assertIsInstance(
                    get_income_shock_laws(agent.IncShkDstn[0])[0], law
                )
                agent.solve()
                self.assertLess(agent.solution_distance, agent.tolerance)
                solution = agent.solution[0]
                limit = solution.mNrmMin
                m = (
                    limit
                    + np.r_[
                        np.geomspace(1e-6, 0.1, 1001), np.linspace(0.1, 20.0, 200001)
                    ]
                )
                self.assertTrue(np.all(solution.cFunc(m) > 0.0))
                self.assertLessEqual(largest_rise(solution.cFunc, m), 1e-12)

    def test_no_jump_in_curvature_at_the_image_of_the_kink(self):
        # From one level of assets the unemployed land on next period's kink at
        # the permanent shock's lowest value: the lower end of the kink's image.
        # A truncated law's density is positive at its bound, so the slope in
        # assets of E2, the expectation of next period's second derivative of
        # value, jumps there, and with it the second derivative of this
        # period's consumption function. The jumpless law's density is zero at
        # its bound, and the slope is continuous: what is left of the difference
        # between its two sides falls with the square of the step
        jump = {}
        for tails, law in self.laws:
            agent = IndShockConsumerType(
                cycles=0, CubicBool=True, IncShkExpectation="continuous", **tails
            )
            agent.solve()
            solution = agent.solution[0]
            perm, _, _, benefit = get_income_shock_laws(agent.IncShkDstn[0])
            image = (
                (solution.mNrmKinks[0] - benefit)
                * agent.PermGroFac[0]
                * law_bounds(perm)[0]
                / agent.Rfree[0]
            )
            for h in (1e-3, 1e-4):
                _, E2, _ = _calc_BegOfPrd_IndShock_continuous(
                    image + h * np.arange(-2, 3),
                    solution,
                    agent.IncShkDstn[0],
                    agent.Rfree[0],
                    agent.PermGroFac[0],
                    agent.CRRA,
                    True,
                    False,
                    8,
                    8,
                    True,
                )
                below = (3 * E2[2] - 4 * E2[1] + E2[0]) / (2 * h)
                above = (-3 * E2[2] + 4 * E2[3] - E2[4]) / (2 * h)
                jump[law, h] = abs(above - below) / abs(below)
        truncated, jumpless = (
            LogNormalTruncatedMeanOne,
            LogNormalTruncatedJumplessMeanOne,
        )
        # a jump of about one percent of the slope, whatever the step
        self.assertGreater(jump[truncated, 1e-4], 1e-2)
        self.assertAlmostEqual(jump[truncated, 1e-3] / jump[truncated, 1e-4], 1.0, 2)
        # none: a difference smaller by four orders, falling with the step
        self.assertLess(jump[jumpless, 1e-4], 1e-6)
        self.assertLess(jump[jumpless, 1e-4], 0.02 * jump[jumpless, 1e-3])

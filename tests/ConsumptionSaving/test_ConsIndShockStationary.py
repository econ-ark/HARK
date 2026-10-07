"""Tests of the direct stationary solve of the ConsIndShock problem
(``stationary_method="anderson"`` on ``IndShockConsumerType``).

The stationary solution is the fixed point backward induction converges to,
so every test compares against a stock solve run to a tight tolerance.  The
quarterly calibration below has a limiting MPC near one percent, where the
stock loop needs about two thousand cycles; that is the case the accelerated
solve exists for.
"""

import unittest

import numpy as np

from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsIndShockStationaryProblem,
    IndShockConsumerType,
    KinkedRconsumerType,
    PerfForesightConsumerType,
    init_idiosyncratic_shocks,
)
from HARK.distributions import expected
from HARK.stationary import solve_stationary_problem

QUARTERLY = dict(
    init_idiosyncratic_shocks,
    cycles=0,
    DiscFac=0.99,
    Rfree=[1.01],
    PermGroFac=[1.0025],
    LivPrb=[0.995],
    PermShkStd=[0.06],
    TranShkStd=[0.2],
    UnempPrb=0.05,
    IncUnemp=0.3,
    verbose=False,
)


def solve_stock(tolerance=1e-12, **overrides):
    agent = IndShockConsumerType(**dict(QUARTERLY, **overrides))
    agent.tolerance = tolerance
    agent.solve()
    return agent


def solve_anderson(tolerance=1e-11, **overrides):
    agent = IndShockConsumerType(
        **dict(QUARTERLY, stationary_method="anderson", **overrides)
    )
    agent.tolerance = tolerance
    agent.solve()
    return agent


def gap_c(a, b, m):
    return float(np.max(np.abs(a.solution[0].cFunc(m) - b.solution[0].cFunc(m))))


class testStationaryLinear(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stock = solve_stock()
        cls.m = np.linspace(cls.stock.solution[0].mNrmMin + 0.05, 40.0, 400)
        cls.anderson = solve_anderson()

    def test_reaches_the_fixed_point_of_backward_induction(self):
        self.assertLess(gap_c(self.anderson, self.stock, self.m), 1e-8)

    def test_scalars_sit_at_their_limits(self):
        """The stock loop stops on the marginal value function while human
        wealth and the limiting MPC are still drifting geometrically; the
        stationary solve iterates them to their limits, which have closed
        forms."""
        agent = self.anderson
        solution = agent.solution[0]
        R, G, LivPrb = agent.Rfree[0], agent.PermGroFac[0], agent.LivPrb[0]
        Ex_IncNext = expected(
            lambda x: x["PermShk"] * x["TranShk"], agent.IncShkDstn[0]
        )
        hNrm = (G / R) * Ex_IncNext / (1.0 - G / R)
        MPCmin = 1.0 - (R * agent.DiscFac * LivPrb) ** (1.0 / agent.CRRA) / R
        self.assertAlmostEqual(solution.hNrm, hNrm, places=8)
        self.assertAlmostEqual(solution.MPCmin, MPCmin, places=10)
        self.assertAlmostEqual(solution.hNrm, self.stock.solution[0].hNrm, places=3)
        self.assertAlmostEqual(solution.MPCmin, self.stock.solution[0].MPCmin, places=7)
        self.assertEqual(solution.mNrmMin, self.stock.solution[0].mNrmMin)

    def test_uses_far_fewer_sweeps(self):
        info = self.anderson.stationary_info
        self.assertTrue(info["converged"])
        self.assertFalse(info["fallback"])
        self.assertEqual(info["method"], "anderson")
        self.assertLess(10 * info["total_sweeps"], self.stock.completed_cycles)
        self.assertEqual(self.anderson.completed_cycles, info["total_sweeps"])

    def test_default_tolerance_is_comparably_accurate(self):
        """Both routes stop on the largest change between sweeps, which bounds
        the distance to the fixed point only up to the contraction rate; at
        HARK's default tolerance the stock loop is off by about 1e-4 in
        consumption on this calibration.  The accelerated solve must land
        within an order of magnitude of that, in far fewer sweeps."""
        stock = solve_stock(tolerance=1e-6)
        anderson = solve_anderson(tolerance=1e-6)
        self.assertLessEqual(
            gap_c(anderson, self.stock, self.m), 10.0 * gap_c(stock, self.stock, self.m)
        )
        self.assertLess(anderson.stationary_info["sweeps"], stock.completed_cycles)

    def test_plain_iteration_of_the_problem_reaches_the_same_point(self):
        agent = self.stock
        problem = ConsIndShockStationaryProblem(
            agent.IncShkDstn[0],
            agent.LivPrb[0],
            agent.DiscFac,
            agent.CRRA,
            agent.Rfree[0],
            agent.PermGroFac[0],
            agent.BoroCnstArt,
            agent.aXtraGrid,
            False,
            agent.solution_terminal,
        )
        plain, info = solve_stationary_problem(problem, method="iterate", tol=1e-11)
        self.assertTrue(info["converged"])
        self.assertLess(
            float(
                np.max(
                    np.abs(plain.cFunc(self.m) - self.stock.solution[0].cFunc(self.m))
                )
            ),
            1e-8,
        )


class testStationaryCubicAndValue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stock = solve_stock(CubicBool=True, vFuncBool=True)
        cls.anderson = solve_anderson(CubicBool=True, vFuncBool=True)
        cls.m = np.linspace(cls.stock.solution[0].mNrmMin + 0.05, 40.0, 400)

    def test_cubic_policy_matches(self):
        self.assertLess(gap_c(self.anderson, self.stock, self.m), 1e-8)

    def test_value_function_matches(self):
        """Compared through the inverse-utility transform the two value
        functions carry, relative to its level; the value itself is unbounded
        below at the bottom of the grid."""
        ours = self.anderson.solution[0].vFunc.vFuncNvrs(self.m)
        theirs = self.stock.solution[0].vFunc.vFuncNvrs(self.m)
        self.assertLess(float(np.max(np.abs(ours / theirs - 1.0))), 1e-6)

    def test_records_every_stage(self):
        info = self.anderson.stationary_info
        self.assertTrue(info["converged"])
        self.assertGreater(info["warm_start_sweeps"], 0)
        self.assertTrue(info["value"]["converged"])
        self.assertEqual(
            info["total_sweeps"],
            info["sweeps"] + info["warm_start_sweeps"] + info["value"]["sweeps"],
        )
        self.assertLess(5 * info["total_sweeps"], self.stock.completed_cycles)


class testStationaryDispatch(unittest.TestCase):
    def test_default_path_is_unchanged(self):
        agent = IndShockConsumerType(**QUARTERLY)
        self.assertEqual(agent.stationary_method, "iterate")
        self.assertEqual(agent.stationary_options, {})
        agent.solve()
        self.assertFalse(hasattr(agent, "stationary_info"))

    def test_options_reach_the_method(self):
        agent = solve_anderson(stationary_options={"depth": 3})
        self.assertTrue(agent.stationary_info["converged"])
        agent = IndShockConsumerType(
            **dict(
                QUARTERLY,
                stationary_method="anderson",
                stationary_options={"memory": 3},
            )
        )
        with self.assertRaises(ValueError):
            agent.solve()

    def test_finite_horizon_is_refused(self):
        agent = IndShockConsumerType(
            **dict(QUARTERLY, cycles=1, stationary_method="anderson")
        )
        with self.assertRaises(ValueError):
            agent.solve()

    def test_unknown_method_is_refused(self):
        agent = IndShockConsumerType(**dict(QUARTERLY, stationary_method="newton"))
        with self.assertRaises(ValueError):
            agent.solve()

    def test_multi_period_cycle_is_refused(self):
        two_periods = dict(
            QUARTERLY,
            T_cycle=2,
            Rfree=[1.01] * 2,
            LivPrb=[0.995] * 2,
            PermGroFac=[1.0025] * 2,
            PermShkStd=[0.06] * 2,
            TranShkStd=[0.2] * 2,
            stationary_method="anderson",
        )
        agent = IndShockConsumerType(**two_periods)
        with self.assertRaises(NotImplementedError):
            agent.solve()

    def test_other_solvers_are_refused(self):
        kinked = KinkedRconsumerType(cycles=0, stationary_method="anderson")
        with self.assertRaises(NotImplementedError):
            kinked.solve()
        perfect_foresight = PerfForesightConsumerType(
            cycles=0, stationary_method="anderson"
        )
        with self.assertRaises(NotImplementedError):
            perfect_foresight.solve()

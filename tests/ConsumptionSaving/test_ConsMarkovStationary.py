"""Tests of the direct stationary solve of the ConsMarkov problem
(``stationary_method="anderson"`` on ``MarkovConsumerType``), against stock
solves run to a tight tolerance."""

import unittest

import numpy as np

from HARK.ConsumptionSaving.ConsMarkovModel import (
    MarkovConsumerType,
    init_indshk_markov,
)

# The default two-state calibration, made patient, with returns that differ
# by state so that every scalar the solver carries is an array.
PATIENT = dict(
    init_indshk_markov,
    cycles=0,
    DiscFac=0.99,
    Rfree=[np.array([1.02, 1.04])],
    verbose=False,
)


def solve_stock(tolerance=1e-12, **overrides):
    agent = MarkovConsumerType(**dict(PATIENT, **overrides))
    agent.tolerance = tolerance
    agent.solve()
    return agent


def solve_anderson(tolerance=1e-11, **overrides):
    agent = MarkovConsumerType(
        **dict(PATIENT, stationary_method="anderson", **overrides)
    )
    agent.tolerance = tolerance
    agent.solve()
    return agent


def grids(agent):
    return [np.linspace(m + 0.05, 40.0, 400) for m in agent.solution[0].mNrmMin]


def gap_c(a, b):
    return max(
        float(np.max(np.abs(fa(m) - fb(m))))
        for fa, fb, m in zip(a.solution[0].cFunc, b.solution[0].cFunc, grids(b))
    )


class testMarkovStationary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stock = solve_stock()
        cls.anderson = solve_anderson()

    def test_reaches_the_fixed_point_in_every_state(self):
        self.assertLess(gap_c(self.anderson, self.stock), 1e-8)

    def test_scalars_are_arrays_at_their_limits(self):
        ours, theirs = self.anderson.solution[0], self.stock.solution[0]
        for name in ("hNrm", "MPCmin", "MPCmax", "mNrmMin"):
            self.assertEqual(np.shape(getattr(ours, name)), (2,))
        np.testing.assert_allclose(ours.hNrm, theirs.hNrm, rtol=1e-4)
        np.testing.assert_allclose(ours.MPCmin, theirs.MPCmin, rtol=1e-6)
        np.testing.assert_array_equal(ours.mNrmMin, theirs.mNrmMin)

    def test_uses_far_fewer_sweeps(self):
        info = self.anderson.stationary_info
        self.assertTrue(info["converged"])
        self.assertFalse(info["fallback"])
        self.assertLess(5 * info["total_sweeps"], self.stock.completed_cycles)

    def test_cubic_and_value_match(self):
        stock = solve_stock(CubicBool=True, vFuncBool=True)
        anderson = solve_anderson(CubicBool=True, vFuncBool=True)
        self.assertLess(gap_c(anderson, stock), 1e-8)
        for fa, fb, m in zip(
            anderson.solution[0].vFunc, stock.solution[0].vFunc, grids(stock)
        ):
            self.assertLess(
                float(np.max(np.abs(fa.vFuncNvrs(m) / fb.vFuncNvrs(m) - 1.0))), 1e-6
            )
        info = anderson.stationary_info
        self.assertGreater(info["warm_start_sweeps"], 0)
        self.assertTrue(info["value"]["converged"])

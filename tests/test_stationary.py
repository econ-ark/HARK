"""Tests of ``HARK.stationary``: the problem interface, the method registry and
the driver, on a toy problem whose fixed point is known."""

import unittest
import warnings

import numpy as np

from HARK.stationary import (
    STATIONARY_METHODS,
    StationaryProblem,
    iterate,
    register_stationary_method,
    solve_stationary_problem,
)


class Contraction(StationaryProblem):
    """``x <- rates * x + shift``, with one rate near one; the fixed point is
    ``shift / (1 - rates)`` and ``assemble`` wraps the vector in a dict."""

    def __init__(self, reject_above=None):
        self.rates = np.array([0.99, 0.9, 0.5])
        self.shift = np.array([1.0, 2.0, 3.0])
        self.x0 = np.zeros(3)
        self.reject_above = reject_above

    @property
    def fixed_point(self):
        return self.shift / (1.0 - self.rates)

    def sweep(self, x):
        return self.rates * x + self.shift

    def assemble(self, x):
        return {"x": x}

    def admissible(self, x):
        if self.reject_above is not None and np.any(x > self.reject_above):
            return "above the cap"
        return None


def stuck(problem, tol, maxit, options=None, verbose=False):
    """A method that gives up at once, to exercise the driver's fallback."""
    return np.array(problem.x0, dtype=float), {
        "method": "stuck",
        "sweeps": 1,
        "converged": False,
        "move": np.inf,
        "restarts": 0,
    }


def relabelled(problem, tol, maxit, options=None, verbose=False):
    x, info = iterate(problem, tol, maxit)
    return x, {**info, "method": "relabelled"}


class testStationaryDriver(unittest.TestCase):
    def test_both_methods_reach_the_fixed_point(self):
        for method in ("iterate", "anderson"):
            problem = Contraction()
            solution, info = solve_stationary_problem(problem, method=method, tol=1e-11)
            np.testing.assert_allclose(solution["x"], problem.fixed_point, rtol=1e-8)
            self.assertTrue(info["converged"])
            self.assertEqual(info["method"], method)
            self.assertFalse(info["fallback"])

    def test_anderson_uses_far_fewer_sweeps(self):
        _, plain = solve_stationary_problem(Contraction(), method="iterate", tol=1e-11)
        _, mixed = solve_stationary_problem(Contraction(), method="anderson", tol=1e-11)
        self.assertGreater(plain["sweeps"], 50 * mixed["sweeps"])

    def test_unknown_method_is_refused(self):
        with self.assertRaises(ValueError):
            solve_stationary_problem(Contraction(), method="newton")

    def test_unknown_option_is_refused(self):
        with self.assertRaises(ValueError):
            solve_stationary_problem(
                Contraction(), method="anderson", options={"memory": 5}
            )

    def test_a_registered_method_is_used(self):
        register_stationary_method("relabelled", relabelled)
        try:
            problem = Contraction()
            solution, info = solve_stationary_problem(
                problem, method="relabelled", tol=1e-9
            )
            self.assertEqual(info["method"], "relabelled")
            np.testing.assert_allclose(solution["x"], problem.fixed_point, rtol=1e-6)
        finally:
            del STATIONARY_METHODS["relabelled"]

    def test_a_failed_method_falls_back_to_plain_iteration(self):
        register_stationary_method("stuck", stuck)
        try:
            problem = Contraction()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                solution, info = solve_stationary_problem(
                    problem, method="stuck", tol=1e-9
                )
            self.assertTrue(info["fallback"])
            self.assertEqual(info["method"], "iterate")
            self.assertEqual(info["attempted"]["method"], "stuck")
            self.assertTrue(any("falling back" in str(w.message) for w in caught))
            np.testing.assert_allclose(solution["x"], problem.fixed_point, rtol=1e-6)
        finally:
            del STATIONARY_METHODS["stuck"]

    def test_a_rejected_fixed_point_of_plain_iteration_raises(self):
        problem = Contraction(reject_above=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaises(ValueError):
                solve_stationary_problem(problem, method="anderson", tol=1e-9)

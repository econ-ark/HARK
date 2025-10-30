"""
This file implements unit tests for HARK.estimation, a fairly sparse module.
"""

import unittest
import numpy as np

from HARK.estimation import minimize_nelder_mead, minimize_powell, parallelNelderMead

# Objective function is simple 3D quadratic with minimum at (2,3,-4)
true_min = np.array([2.0, 3.0, -4.0])


def alt_func(q):
    return q[0] ** 2 - 4 * q[0] + q[1] ** 2 - 6 * q[1] + q[2] ** 2 + 8 * q[2]


class testMinimizer(unittest.TestCase):
    params = {"minimizer": None}
    args = {"verbose": True}

    def test_min(self):
        if self.params["minimizer"] is None:
            return

        xyz_init = np.array([0.0, 0.0, 0.0])
        minimizer = self.params["minimizer"]
        output = minimizer(alt_func, xyz_init, **self.args)
        diff = output - true_min
        dist = np.sqrt(np.dot(diff, diff))
        self.assertTrue(dist < 1e-4)


class testPowell(testMinimizer):
    params = {"minimizer": minimize_powell}


class testNelderMead(testMinimizer):
    params = {"minimizer": minimize_nelder_mead}


class testParallelNelderMead(testMinimizer):
    params = {"minimizer": parallelNelderMead}
    args = {"P": 2, "ftol": 1e-12}

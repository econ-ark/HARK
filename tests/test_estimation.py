"""
This file implements unit tests for HARK.estimation, a fairly sparse module.
"""

import unittest
import numpy as np
from copy import copy

from HARK.estimation import (
    minimize_nelder_mead,
    minimize_powell,
    parallelNelderMead,
    bootstrap_sample_from_data,
)

# Objective function is simple 3D quadratic with minimum at (2,3,-4)
true_min = np.array([2.0, 3.0, -4.0])


def my_func(q):
    return q[0] ** 2 - 4 * q[0] + q[1] ** 2 - 6 * q[1] + q[2] ** 2 + 8 * q[2]


class testMinimizer(unittest.TestCase):
    params = {"minimizer": None}
    args = {"verbose": True}

    def test_min(self):
        if self.params["minimizer"] is None:
            return

        xyz_init = np.array([0.0, 0.0, 0.0])
        minimizer = self.params["minimizer"]
        output = minimizer(my_func, xyz_init, **self.args)
        diff = output - true_min
        dist = np.sqrt(np.dot(diff, diff))
        self.assertTrue(dist < 1e-4)


class testPowell(testMinimizer):
    params = {"minimizer": minimize_powell}


class testNelderMead(testMinimizer):
    params = {"minimizer": minimize_nelder_mead}


class testParallelNelderMead(testMinimizer):
    params = {"minimizer": parallelNelderMead}
    args = {"P": 2, "ftol": 1e-16}


class testNonParallelNelderMead(testMinimizer):
    params = {"minimizer": parallelNelderMead}
    args = {"P": 1, "ftol": 1e-16, "xtol": 1e-16, "maxthreads": 1}


class testExcessParallelNelderMead(testMinimizer):
    params = {"minimizer": parallelNelderMead}
    args = {"P": 20, "ftol": 1e-16, "xtol": 1e-16}


class testPNM_read_write(unittest.TestCase):
    def test_read_and_write(self):
        # These run for 20 iterations and then save progress
        first_args = {
            "verbose": True,
            "maxiter": 20,
            "name": "test_progress",
            "savefreq": 10,
            "P": 2,
            "ftol": 1e-12,
        }

        # These resume from the prior point and then continue to completion
        second_args = copy(first_args)
        second_args["maxiter"] = np.inf
        second_args["resume"] = True

        xyz_init = np.array([0.0, 0.0, 0.0])
        minimizer = parallelNelderMead

        output1 = minimizer(my_func, xyz_init, **first_args)
        diff = output1 - true_min
        dist = np.sqrt(np.dot(diff, diff))
        self.assertTrue(dist > 1e-4)

        output2 = minimizer(my_func, xyz_init, **second_args)
        diff = output2 - true_min
        dist = np.sqrt(np.dot(diff, diff))
        self.assertTrue(dist < 1e-4)


class testBootStrapSample(unittest.TestCase):
    def test_function(self):
        data = np.random.rand(1000, 5)
        bootstrap = bootstrap_sample_from_data(data)
        self.assertAlmostEqual(bootstrap.shape[0], 1000)
        self.assertAlmostEqual(bootstrap.shape[1], 5)

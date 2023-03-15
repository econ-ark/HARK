import unittest
from math import isclose

import numpy as np

from HARK.interpolation import MultivariateInterp


class TestMultivariateInterp(unittest.TestCase):
    def setUp(self):
        # create test data

        self.grids = [
            np.linspace(-1, 1, 4),
            np.linspace(-2, 2, 5),
            np.linspace(-3, 3, 6),
        ]

        grids = np.meshgrid(*self.grids, indexing="ij")

        self.values = grids[0] + grids[1] + grids[2]

        self.interpolator_cpu = MultivariateInterp(
            self.values, self.grids, target="cpu"
        )
        self.interpolator_parallel = MultivariateInterp(
            self.values, self.grids, target="parallel"
        )

    def test_interpolation_values(self):
        # check that interpolation values match expected values
        args = [0.1, 1.5, 2.9]
        expected_value = np.sum(args)

        value_cpu = self.interpolator_cpu(*args)
        value_parallel = self.interpolator_parallel(*args)

        self.assertTrue(isclose(value_cpu, expected_value, rel_tol=1e-9))
        self.assertTrue(isclose(value_parallel, expected_value, rel_tol=1e-9))

    def test_invalid_target_error(self):
        # check that specifying an invalid target raises an error
        with self.assertRaises(ValueError):
            _ = MultivariateInterp(self.values, self.grids, target="invalid")

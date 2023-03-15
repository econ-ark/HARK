import unittest

import numpy as np

from HARK.interpolation import MultivariateInterp, WarpedInterpOnInterp2D


def function(*args):
    mats = np.meshgrid(*args, indexing="ij")

    return np.sum(mats, axis=0)


class TestMultivariateInterp(unittest.TestCase):
    def setUp(self):
        # create test data

        self.grids = [
            np.linspace(0, 1, 10),
            np.linspace(0, 1, 11),
            np.linspace(0, 1, 12),
        ]

        self.args = [
            np.linspace(0, 1, 11),
            np.linspace(0, 1, 12),
            np.linspace(0, 1, 13),
        ]

    def test_interpolation_values(self):
        # check that interpolation values match expected values

        interpolator2D_cpu = MultivariateInterp(
            function(*self.grids[0:2]), self.grids[0:2], target="cpu"
        )
        interpolator2D_parallel = MultivariateInterp(
            function(*self.grids[0:2]), self.grids[0:2], target="parallel"
        )
        interpolator3D_cpu = MultivariateInterp(
            function(*self.grids), self.grids, target="cpu"
        )
        interpolator3D_parallel = MultivariateInterp(
            function(*self.grids), self.grids, target="parallel"
        )

        val2D_cpu = interpolator2D_cpu(*np.meshgrid(*self.args[0:2], indexing="ij"))
        val2D_parallel = interpolator2D_parallel(
            *np.meshgrid(*self.args[0:2], indexing="ij")
        )
        val3D_cpu = interpolator3D_cpu(*np.meshgrid(*self.args, indexing="ij"))
        val3D_parallel = interpolator3D_parallel(
            *np.meshgrid(*self.args, indexing="ij")
        )

        self.assertTrue(np.allclose(val2D_cpu, function(*self.args[0:2])))
        self.assertTrue(np.allclose(val2D_parallel, function(*self.args[0:2])))
        self.assertTrue(np.allclose(val3D_cpu, function(*self.args)))
        self.assertTrue(np.allclose(val3D_parallel, function(*self.args)))

import unittest

import numpy as np

from HARK.interpolation import UnstructuredInterp


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

        interpolator2D = UnstructuredInterp(
            function(*self.grids[0:2]), [*np.meshgrid(*self.grids[0:2], indexing="ij")]
        )

        interpolator3D = UnstructuredInterp(
            function(*self.grids), [*np.meshgrid(*self.grids, indexing="ij")]
        )

        val2D = interpolator2D(*np.meshgrid(*self.args[0:2], indexing="ij"))

        val3D = interpolator3D(*np.meshgrid(*self.args, indexing="ij"))

        self.assertTrue(np.allclose(val2D, function(*self.args[0:2])))
        self.assertTrue(np.allclose(val3D, function(*self.args)))

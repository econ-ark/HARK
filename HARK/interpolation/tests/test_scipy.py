import unittest

import numpy as np

from HARK.interpolation import UnstructuredInterp


class TestUnstructuredInterp(unittest.TestCase):
    def test_initialization(self):
        x = np.array([1, 2, 3, 4])
        y = np.array([2, 3, 4, 5])
        z = np.array([3, 4, 5, 6])

        values = np.column_stack((x, y, z))
        grids = np.column_stack(np.meshgrid(x, y)).reshape(-1, 2)

        interp = UnstructuredInterp(values=values, grids=grids, method="linear")

        self.assertTrue(np.array_equal(interp.values, values))
        self.assertTrue(np.array_equal(interp.grids, grids))
        self.assertEqual(interp.method, "linear")

    def test_invalid_interpolation_method(self):
        interp = None
        try:
            x = np.array([1, 2, 3, 4])
            y = np.array([2, 3, 4, 5])
            z = np.array([3, 4, 5, 6])

            values = np.column_stack((x, y, z))
            grids = np.column_stack(np.meshgrid(x, y)).reshape(-1, 2)

            interp = UnstructuredInterp(values=values, grids=grids, method="invalid")
        except ValueError as e:
            self.assertEqual(str(e), "Invalid interpolation method.")

        self.assertIsNone(interp)

    def test_unknown_interpolation_method(self):
        interp = None
        try:
            x = np.array([1, 2, 3, 4])
            y = np.array([2, 3, 4, 5])
            z = np.array([3, 4, 5, 6])

            values = np.column_stack((x, y, z))
            grids = np.column_stack(np.meshgrid(x, y)).reshape(-1, 2)

            interp = UnstructuredInterp(
                values=values, grids=grids, method="some_method"
            )
        except ValueError as e:
            self.assertEqual(
                str(e),
                "Unknown interpolation method some_method for 2 dimensional data.",
            )

        self.assertIsNone(interp)

    def test_rbf_method_with_2D_grid(self):
        x = np.array([0, 1, 2, 3])
        y = np.array([4, 5, 6, 7])
        z = np.array([1, 2, 3, 4])

        values = np.column_stack((x, y, z))
        grids = np.column_stack(np.meshgrid(x, y)).reshape(-1, 2)

        interp = UnstructuredInterp(values=values, grids=grids, method="rbf")

        coords = np.array([[0.5, 5], [2.25, 6.5], [3.3, 6.8]])
        expected_output = interp.interpolator(coords).reshape(
            3,
        )

        np.testing.assert_array_almost_equal(
            expected_output, np.array([1.23606798, 3.27887228, 3.75178997]), decimal=5
        )

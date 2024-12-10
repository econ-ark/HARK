"""
This file implements unit tests for interpolation methods
"""

import unittest

import numpy as np

from HARK.interpolation import BilinearInterp
from HARK.interpolation import CubicHermiteInterp as CubicInterp
from HARK.interpolation import LinearInterp, QuadlinearInterp, TrilinearInterp
from HARK.interpolation import IdentityFunction


class testsLinearInterp(unittest.TestCase):
    """tests for LinearInterp, currently tests for uneven length of
    x and y with user input as lists, arrays, arrays with column orientation
    """

    def setUp(self):
        self.x_list = [1, 2, 3]
        self.y_list = [3, 4]
        self.z_list = [3, 4, 5]
        self.x_array = np.array(self.x_list)
        self.y_array = np.array(self.y_list)
        self.z_array = np.array(self.z_list)
        self.x_array_t = self.x_array.reshape(len(self.x_array), 1)
        self.y_array_t = self.y_array.reshape(len(self.y_array), 1)
        self.z_array_t = self.z_array.reshape(len(self.z_array), 1)

    def test_uneven_length(self):
        self.assertRaises(ValueError, LinearInterp, self.x_list, self.y_list)
        self.assertRaises(ValueError, LinearInterp, self.x_array, self.y_array)
        self.assertRaises(ValueError, LinearInterp, self.x_array_t, self.y_array_t)

    def test_same_length(self):
        linear = LinearInterp(self.x_list, self.z_list)
        self.assertEqual(linear(1.5), 3.5)
        linear = LinearInterp(self.x_array, self.z_array)
        self.assertEqual(linear(1.5), 3.5)
        linear = LinearInterp(self.x_array_t, self.z_array_t)
        self.assertEqual(linear(1.5), 3.5)


class testsCubicInterp(unittest.TestCase):
    """tests for CubicInterp, currently tests for uneven length of
    x, y and derivative with user input as lists, arrays, arrays with column orientation
    """

    def setUp(self):
        self.x_list = [1, 2, 3]
        self.y_list = [1, 4]
        self.dydx_list = [2, 4, 6]
        self.z_list = [1, 4, 9]
        self.x_array = np.array(self.x_list)
        self.y_array = np.array(self.y_list)
        self.dydx_array = np.array(self.dydx_list)
        self.z_array = np.array(self.z_list)
        self.x_array_t = self.x_array.reshape(len(self.x_array), 1)
        self.y_array_t = self.y_array.reshape(len(self.y_array), 1)
        self.dydx_array_t = self.dydx_array.reshape(len(self.dydx_array), 1)
        self.z_array_t = self.z_array.reshape(len(self.z_array), 1)

    def test_uneven_length(self):
        self.assertRaises(
            ValueError, CubicInterp, self.x_list, self.y_list, self.dydx_list
        )
        self.assertRaises(
            ValueError, CubicInterp, self.x_array, self.y_array, self.dydx_array
        )
        self.assertRaises(
            ValueError, CubicInterp, self.x_array_t, self.y_array_t, self.dydx_array_t
        )

    def test_same_length(self):
        cube = CubicInterp(self.x_list, self.z_list, self.dydx_list)
        self.assertEqual(cube(1.5), 2.25)
        cube = CubicInterp(self.x_array, self.z_array, self.dydx_array)
        self.assertEqual(cube(1.5), 2.25)
        cube = CubicInterp(self.x_array_t, self.z_array_t, self.dydx_array_t)
        self.assertEqual(cube(1.5), 2.25)


class testsBilinearInterp(unittest.TestCase):
    """tests for BilinearInterp, currently tests for uneven length of
    x, y, f(x,y) with user input as arrays, arrays with column orientation
    """

    def setUp(self):
        self.f_array = np.array([[2, 4], [4, 6]])
        self.x_array = np.array([1, 3])
        self.y_array = np.array([1, 3])
        self.z_array = np.array([1, 2, 3])
        self.z_array_t = self.z_array.reshape(len(self.z_array), 1)
        self.y_array_t = self.y_array.reshape(len(self.y_array), 1)

    def test_uneven_length(self):
        self.assertRaises(
            ValueError, BilinearInterp, self.f_array, self.x_array, self.z_array
        )
        self.assertRaises(
            ValueError, BilinearInterp, self.f_array, self.x_array, self.z_array_t
        )

    def test_same_length(self):
        bilinear = BilinearInterp(self.f_array, self.x_array, self.y_array)
        self.assertEqual(bilinear(2, 2), 4.0)
        bilinear = BilinearInterp(self.f_array, self.x_array, self.y_array_t)
        self.assertEqual(bilinear(2, 2), 4.0)


class testsTrilinearInterp(unittest.TestCase):
    """tests for TrilinearInterp, currently tests for uneven length of
    x, y, z, f(x, y, z) with user input as arrays, arrays with column orientation
    """

    def setUp(self):
        self.f_array = np.array([[[3, 5], [5, 7]], [[5, 7], [7, 10]]])
        self.x_array = np.array([1, 3])
        self.y_array = np.array([1, 3])
        self.z_array = np.array([1, 3])
        self.fail_array = np.array([1, 2, 3])
        self.fail_array_t = self.z_array.reshape(len(self.z_array), 1)
        self.y_array_t = self.y_array.reshape(len(self.y_array), 1)

    def test_uneven_length(self):
        self.assertRaises(
            ValueError,
            TrilinearInterp,
            self.f_array,
            self.x_array,
            self.y_array,
            self.fail_array,
        )
        self.assertRaises(
            ValueError,
            TrilinearInterp,
            self.f_array,
            self.x_array,
            self.fail_array,
            self.fail_array_t,
        )

    def test_same_length(self):
        bilinear = TrilinearInterp(
            self.f_array, self.x_array, self.y_array, self.z_array
        )
        self.assertEqual(bilinear(1, 2, 2), 5.0)
        bilinear = TrilinearInterp(
            self.f_array, self.x_array, self.y_array_t, self.z_array
        )
        self.assertEqual(bilinear(1, 2, 2), 5.0)


class testsQuadlinearInterp(unittest.TestCase):
    """tests for TrilinearInterp, currently tests for uneven length of
    w, x, y, z, f(w, x, y, z) with user input as arrays, arrays with column orientation
    """

    def setUp(self):
        self.f_array = np.array(
            [
                [[[4, 6], [6, 8]], [[6, 8], [8, 11]]],
                [[[6, 8], [8, 10]], [[8, 10], [10, 13]]],
            ]
        )
        self.x_array = np.array([1, 3])
        self.y_array = np.array([1, 3])
        self.z_array = np.array([1, 3])
        self.w_array = np.array([1, 3])
        self.fail_array = np.array([1, 2, 3])
        self.fail_array_t = self.z_array.reshape(len(self.z_array), 1)
        self.y_array_t = self.y_array.reshape(len(self.y_array), 1)

    def test_uneven_length(self):
        self.assertRaises(
            ValueError,
            QuadlinearInterp,
            self.f_array,
            self.x_array,
            self.y_array,
            self.fail_array,
            self.w_array,
        )
        self.assertRaises(
            ValueError,
            QuadlinearInterp,
            self.f_array,
            self.x_array,
            self.fail_array,
            self.fail_array_t,
            self.w_array,
        )

    def test_same_length(self):
        bilinear = QuadlinearInterp(
            self.f_array, self.w_array, self.x_array, self.y_array, self.z_array
        )
        self.assertEqual(bilinear(1, 2, 1, 2), 6.0)
        bilinear = QuadlinearInterp(
            self.f_array, self.w_array, self.x_array, self.y_array_t, self.z_array
        )
        self.assertEqual(bilinear(1, 2, 1, 2), 6.0)


class test_IdentityFunction(unittest.TestCase):
    """
    Tests evaluation and derivatives of IdentityFunction class.
    """

    def setUp(self):
        self.IF1D = IdentityFunction()
        self.IF2Da = IdentityFunction(i_dim=0, n_dims=2)
        self.IF2Db = IdentityFunction(i_dim=1, n_dims=2)
        self.IF3Da = IdentityFunction(i_dim=0, n_dims=3)
        self.IF3Db = IdentityFunction(i_dim=2, n_dims=3)
        self.X = 3 * np.ones(100)
        self.Y = 4 * np.ones(100)
        self.Z = 5 * np.ones(100)
        self.zero = np.zeros(100)
        self.one = np.ones(100)

    def test_eval(self):
        self.assertEqual(self.X, self.IF1D(self.X))
        self.assertEqual(self.X, self.IF2Da(self.X, self.Y))
        self.assertEqual(self.Y, self.IF2Db(self.X, self.Y))
        self.assertEqual(self.X, self.IF3Da(self.X, self.Y, self.Z))
        self.assertEqual(self.Z, self.IF3Db(self.X, self.Y, self.Z))

    def test_der(self):
        self.assertEqual(self.one, self.IF1D.der(self.X))

        self.assertEqual(self.one, self.IF2Da.derX(self.X, self.Y))
        self.assertEqual(self.zero, self.IF2Da.derY(self.X, self.Y))

        self.assertEqual(self.zero, self.IF2Db.derX(self.X, self.Y))
        self.assertEqual(self.one, self.IF2Db.derY(self.X, self.Y))

        self.assertEqual(self.one, self.IF3Da.derX(self.X, self.Y, self.Z))
        self.assertEqual(self.zero, self.IF3Da.derY(self.X, self.Y, self.Z))
        self.assertEqual(self.zero, self.IF3Da.derZ(self.X, self.Y, self.Z))

        self.assertEqual(self.zero, self.IF3Db.derX(self.X, self.Y, self.Z))
        self.assertEqual(self.zero, self.IF3Db.derY(self.X, self.Y, self.Z))
        self.assertEqual(self.one, self.IF3Db.derZ(self.X, self.Y, self.Z))

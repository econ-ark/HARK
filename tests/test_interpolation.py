"""
This file implements unit tests for interpolation methods
"""

from HARK.interpolation import (
    IdentityFunction,
    LinearInterp,
    BilinearInterp,
    TrilinearInterp,
    QuadlinearInterp,
    LinearInterpOnInterp1D,
    Curvilinear2DInterp,
)
from HARK.interpolation import CubicHermiteInterp as CubicInterp

import numpy as np
import unittest


class TestInterp1D(unittest.TestCase):
    """
    A class for testing subclasses of HARKinterpolator1D. To use it, define a subclass
    of this class and give it the class to be tested in the attribute interpolator_.
    Also define a method called make_interpolant that builds an interpolant for the
    test function stored in attribute function.
    """

    tol = 1e-6  # change this at the subclass level if necessary

    def setUp(self):
        """
        The test function for 1D interpolators is the natural log function.
        """
        f = np.log
        X = np.linspace(1.5, 15.0, 100)
        fX = f(X)
        self.function = f
        self.test_vals = X
        self.targets = fX
        self.make_interpolant()

    def make_interpolant(self):
        """
        Each subclass needs to overwrite this method with one that *actually*
        constructs an interpolant based on the function.
        """
        self.interpolant = None  # This is so the base class can "pass" tests

    def test_eval(self):
        if self.interpolant is None:
            return
        output = self.interpolant(self.test_vals)
        diffs = output - self.targets
        acc = np.mean(np.abs(diffs))
        self.assertTrue(acc < self.tol)

    def test_der(self):
        if self.interpolant is None:
            return
        # Doesn't actually check values of derivative, just whether it runs
        # and whether they are all real values
        derivs = self.interpolant.derivative(self.test_vals)
        self.assertTrue(np.all(np.logical_not(np.isnan(derivs))))
        self.assertTrue(np.all(np.logical_not(np.isinf(derivs))))


class TestInterp2D(unittest.TestCase):
    """
    A class for testing subclasses of HARKinterpolator2D. To use it, define a subclass
    of this class and give it the class to be tested in the attribute interpolator_.
    Also define a method called make_interpolant that builds an interpolant for the
    test function stored in attribute function.
    """

    tol = 1e-6  # change this at the subclass level if necessary

    def setUp(self):
        """
        The test function for 2D interpolators is f(x,y) = log(3x + 5y)
        """
        f = lambda x, y: np.log(3 * x + 5 * y)
        RNG = np.random.RandomState(seed=10212026)
        N = 500
        X = 8 * RNG.rand(N) + 0.5  # 0.5 to 8.5
        Y = 4 * RNG.rand(N) + 0.2  # 0.2 to 4.2
        fXY = f(X, Y)
        self.function = f
        self.test_vals = (X, Y)
        self.targets = fXY
        self.make_interpolant()

    def make_interpolant(self):
        """
        Each subclass needs to overwrite this method with one that *actually*
        constructs an interpolant based on the function.
        """
        self.interpolant = None  # This is so the base class can "pass" tests

    def test_eval(self):
        if self.interpolant is None:
            return
        output = self.interpolant(*self.test_vals)
        diffs = output - self.targets
        acc = np.mean(np.abs(diffs))
        self.assertTrue(acc < self.tol)

    def test_derX(self):
        if self.interpolant is None:
            return
        # Doesn't actually check values of derivative, just whether it runs
        # and whether they are all real values
        derivs = self.interpolant.derivativeX(*self.test_vals)
        self.assertTrue(np.all(np.logical_not(np.isnan(derivs))))
        self.assertTrue(np.all(np.logical_not(np.isinf(derivs))))

    def test_derY(self):
        if self.interpolant is None:
            return
        # Doesn't actually check values of derivative, just whether it runs
        # and whether they are all real values
        derivs = self.interpolant.derivativeY(*self.test_vals)
        self.assertTrue(np.all(np.logical_not(np.isnan(derivs))))
        self.assertTrue(np.all(np.logical_not(np.isinf(derivs))))


###############################################################################


class TestLinearInterp(TestInterp1D):
    interpolator_ = LinearInterp
    tol = 1e-5

    def make_interpolant(self):
        bot = 0.001
        top = 20.0
        order = 3.0
        x_grid = (top - bot) * np.linspace(0.0, 1.0, 501) ** order + bot
        y_grid = self.function(x_grid)
        self.interpolant = self.interpolator_(x_grid, y_grid)


class TestCubicInterp(TestInterp1D):
    interpolator_ = CubicInterp

    def make_interpolant(self):
        bot = 0.001
        top = 20.0
        order = 3.0
        x_grid = (top - bot) * np.linspace(0.0, 1.0, 201) ** order + bot
        y_grid = self.function(x_grid)
        dydx_grid = x_grid ** (-1.0)  # derivative of log
        self.interpolant = self.interpolator_(x_grid, y_grid, dydx_grid)


class TestBilinearInterp(TestInterp2D):
    interpolator_ = BilinearInterp
    tol = 1e-5

    def make_interpolant(self):
        xbot = 0.3
        xtop = 10.0
        xorder = 3.0
        ybot = 0.1
        ytop = 6.0
        yorder = 2.5
        x_grid = (xtop - xbot) * np.linspace(0.0, 1.0, 401) ** xorder + xbot
        y_grid = (ytop - ybot) * np.linspace(0.0, 1.0, 301) ** yorder + ybot
        xmesh, ymesh = np.meshgrid(x_grid, y_grid, indexing="ij")
        values = self.function(xmesh, ymesh)
        self.interpolant = self.interpolator_(values, x_grid, y_grid)


class TestLinearInterpOnInterp1D(TestInterp2D):
    interpolator_ = LinearInterpOnInterp1D
    tol = 2e-5

    def make_interpolant(self):
        xbot = 0.3
        xtop = 10.0
        xorder = 3.0
        ybot = 0.1
        ytop = 6.0
        yorder = 2.5
        y_grid = (ytop - ybot) * np.linspace(0.0, 1.0, 301) ** yorder + ybot
        x_interpolators = []
        RNG = np.random.RandomState(seed=8675309)
        for j in range(y_grid.size):
            temp = np.sort(RNG.rand(401))
            x_grid = (xtop - xbot) * temp**xorder + xbot
            f_temp = self.function(x_grid, y_grid[j] * np.ones_like(x_grid))
            x_interpolators.append(LinearInterp(x_grid, f_temp))
        self.interpolant = self.interpolator_(x_interpolators, y_grid)


class TestCurvilinear2DInterp(TestInterp2D):
    interpolator_ = Curvilinear2DInterp
    tol = 1e-4

    def make_interpolant(self):
        xbot = 0.3
        xtop = 10.0
        ybot = 0.1
        ytop = 6.0
        x_grid = np.linspace(xbot, xtop, 401)
        y_grid = np.linspace(ybot, ytop, 301)
        xmesh, ymesh = np.meshgrid(x_grid, y_grid, indexing="ij")
        RNG = np.random.RandomState(seed=5318008)
        x_adj = 0.005 * RNG.rand(401, 301) - 0.0025
        y_adj = 0.004 * RNG.rand(401, 301) - 0.002
        xmesh += x_adj
        ymesh += y_adj
        values = self.function(xmesh, ymesh)
        self.interpolant = self.interpolator_(values, xmesh, ymesh)


###############################################################################


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

    def testPreCompute(self):
        f = LinearInterp(self.x_array, self.z_array, lower_extrap=True)
        g = LinearInterp(
            self.x_array, self.z_array, lower_extrap=True, pre_compute=True
        )
        X = np.linspace(0.0, 4.0, 101)
        self.assertTrue(np.all(np.isclose(f(X), g(X))))

    def testIndexer(self):
        bot = 1.0
        top = 10.0
        N = 30
        order = 2.5
        grid = (top - bot) * np.linspace(0.0, 1.0, N) ** order + bot

        def my_indexer(x):
            below = x < bot
            above = x > top
            these = np.logical_not(np.logical_or(below, above))
            i = np.zeros_like(x, dtype=int)
            i[these] = np.ceil(
                (N - 1) * ((x[these] - bot) / (top - bot)) ** (1.0 / order)
            )
            i[below] = 1
            i[above] = N - 1
            return i

        f = np.log
        data = f(grid)
        g = LinearInterp(grid, data)  # default indexing with searchsorted
        h = LinearInterp(grid, data, indexer=my_indexer)  # custom indexing

        X = np.linspace(2, 11, 101)
        self.assertTrue(np.all(np.isclose(g(X), h(X))))


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
        assert np.all(self.X == self.IF1D(self.X))
        assert np.all(self.X == self.IF2Da(self.X, self.Y))
        assert np.all(self.Y == self.IF2Db(self.X, self.Y))
        assert np.all(self.X == self.IF3Da(self.X, self.Y, self.Z))
        assert np.all(self.Z == self.IF3Db(self.X, self.Y, self.Z))

    def test_der(self):
        assert np.all(self.one == self.IF1D.derivative(self.X))

        assert np.all(self.one == self.IF2Da.derivativeX(self.X, self.Y))
        assert np.all(self.zero == self.IF2Da.derivativeY(self.X, self.Y))

        assert np.all(self.zero == self.IF2Db.derivativeX(self.X, self.Y))
        assert np.all(self.one == self.IF2Db.derivativeY(self.X, self.Y))

        assert np.all(self.one == self.IF3Da.derivativeX(self.X, self.Y, self.Z))
        assert np.all(self.zero == self.IF3Da.derivativeY(self.X, self.Y, self.Z))
        assert np.all(self.zero == self.IF3Da.derivativeZ(self.X, self.Y, self.Z))

        assert np.all(self.zero == self.IF3Db.derivativeX(self.X, self.Y, self.Z))
        assert np.all(self.zero == self.IF3Db.derivativeY(self.X, self.Y, self.Z))
        assert np.all(self.one == self.IF3Db.derivativeZ(self.X, self.Y, self.Z))

from operator import index
from typing import no_type_check_decorator
import unittest
import numpy as np

from HARK.interpolation import LinearInterp, BilinearInterp
from HARK.econforgeinterp import LinearFast, DecayInterp
from HARK.core import distance_metric


class CompareLinearInterp(unittest.TestCase):
    """ 
    Compares output and properties with the basic linearinterp
    """

    def setUp(self):

        self.n_grid_points = 100
        self.n_eval_points = 3

    def compare_output(self, x, y, eval_points):

        h_interp = LinearInterp(x, y, lower_extrap=True)
        e_interp = LinearFast(y, [x])

        h_vals = h_interp(eval_points)
        e_vals = e_interp(eval_points)
        self.assertTrue(np.allclose(h_vals, e_vals))

    def test_outputs(self):
        """
        Tests that function calls using the base hark and
        econforge implementations return the same.
        """
        # Square function
        x = np.linspace(0, 10, self.n_grid_points)
        y = np.power(x, 2)

        # Interpolation
        in_points = np.linspace(2, 8, self.n_eval_points)
        self.compare_output(x, y, in_points)

        # Extrapolation
        ex_points = np.linspace(-10, 20, self.n_eval_points)
        self.compare_output(x, y, ex_points)

    def test_metric(self):
        """
        Tests that the interpolator metric called on a pair of
        interpolator objects is the same for hark and econforge
        interps
        """
        n_points = 10

        x0 = np.exp(np.linspace(0, 1, n_points))
        y0 = np.sin(x0)

        x1 = np.linspace(-10, 20, n_points)
        y1 = 0.5 * x1 + 2

        # Distance with HARK
        h_dist = distance_metric(LinearInterp(x0, y0), LinearInterp(x1, y1))
        # Distance with econforge
        e_dist = distance_metric(LinearFast(y0, [x0]), LinearFast(y1, [x1]))

        self.assertAlmostEqual(h_dist, e_dist)


class CompareBilinearInterp(unittest.TestCase):
    """ 
    Compares output and properties with the basic linearinterp
    """

    def setUp(self):

        self.n_grid_points = 100
        self.n_eval_points = 3

    def compare_output(self, x, y, z, eval_x, eval_y):

        h_interp = BilinearInterp(z, x, y)
        e_interp = LinearFast(z, [x, y])

        h_vals = h_interp(eval_x, eval_y)
        e_vals = e_interp(eval_x, eval_y)
        self.assertTrue(np.allclose(h_vals, e_vals))

    def test_outputs(self):
        """
        Tests that function calls using the base hark and
        econforge implementations return the same.
        """
        # Sum of squares function
        x_grid = np.linspace(0, 10, self.n_grid_points)
        y_grid = np.linspace(0, 10, self.n_grid_points)

        x_tiled, y_tiled = np.meshgrid(x_grid, y_grid, indexing="ij")

        z = np.power(x_tiled, 2) + np.power(y_tiled, 2)

        # Interpolation
        x_in, y_in = np.meshgrid(
            np.linspace(2, 8, self.n_eval_points),
            np.linspace(2, 8, self.n_eval_points),
            indexing="ij",
        )
        self.compare_output(x_grid, y_grid, z, x_in, y_in)

        # Extrapolation
        x_ex, y_ex = np.meshgrid(
            np.linspace(-10, 20, self.n_eval_points),
            np.linspace(-10, 20, self.n_eval_points),
            indexing="ij",
        )
        self.compare_output(x_grid, y_grid, z, x_ex, y_ex)

    def test_metric(self):
        """
        Tests that the interpolator metric called on a pair of
        interpolator objects is the same for hark and econforge
        interps
        """
        n_points = 10

        x0 = np.exp(np.linspace(0, 1, n_points))
        y0 = np.linspace(3, 4, n_points)
        x0_t, y0_t = np.meshgrid(x0, y0, indexing="ij")
        z0 = np.sin(x0_t) + y0_t

        x1 = np.linspace(-10, 20, n_points)
        y1 = 0.5 * x1 + 2
        x1_t, y1_t = np.meshgrid(x1, y1, indexing="ij")
        z1 = x1_t + y1_t

        # Distance with HARK
        h_dist = distance_metric(BilinearInterp(z0, x0, y0), BilinearInterp(z1, x1, y1))
        # Distance with econforge
        e_dist = distance_metric(LinearFast(z0, [x0, y0]), LinearFast(z1, [x1, y1]))

        self.assertAlmostEqual(h_dist, e_dist)


# Derivative tests


class Check1DDerivatives(unittest.TestCase):
    """ 
    Checks derivatives in a 1D interpolator
    """

    def setUp(self):

        pass

    def test_linear(self):

        # A linear function on a non-uniform grid
        x = np.exp(np.linspace(0, 2, 6))
        y0 = 2
        slope = 1.0
        y = y0 + slope * x

        interp = LinearFast(y, [x])

        eval_points = np.array([1.5, 2.5, 3.5, 4.5])

        grad = interp.gradient(eval_points)[0]
        result = np.ones_like(eval_points) * slope

        self.assertTrue(np.allclose(grad, result))

    def test_nonlinear(self):

        # A non linear function on uniform grid
        x = np.linspace(-10, 10, 21) * (1 / 2) * np.pi
        y = np.sin(x)
        interp = LinearFast(y, [x])

        eval_points = np.array([-1, -0.5, 0, 0.5, 1]) * np.pi
        grad = interp.gradient(eval_points)[0]

        # 1-order must be + or - pi/2
        result = np.array([-1, -1, 1, 1, -1]) * 2 / np.pi

        self.assertTrue(np.allclose(grad, result))

    def test_nonlinear_approx(self):

        # A non linear function on uniform grid
        x = np.linspace(-10, 10, 10000)
        y = np.power(x, 3)
        interp = LinearFast(y, [x])
        eval_points = np.linspace(-5, 5, 10)

        grad = interp.gradient(eval_points)[0]
        result = np.power(eval_points, 2) * 3.0

        self.assertTrue(np.allclose(grad, result, atol=0.02))


class Check2DDerivatives(unittest.TestCase):
    """ 
    Checks derivatives in a 2D interpolator
    """

    def setUp(self):

        pass

    def test_linear(self):

        # A linear function on a non-uniform grid
        x = np.exp(np.linspace(0, 2, 6))
        y = np.power(np.linspace(0, 5, 10), 2)
        x_tiled, y_tiled = np.meshgrid(x, y, indexing="ij")

        inter = 1
        slope_x = 2
        slope_y = -3
        z = inter + slope_x * x_tiled + slope_y * y_tiled

        interp = LinearFast(z, [x, y])

        # Evaluation points
        n_eval = 7
        x_ev, y_ev = np.meshgrid(
            np.linspace(-20, 20, n_eval), np.linspace(5, -5, n_eval), indexing="ij"
        )

        # Gradient
        grad = interp.gradient(x_ev, y_ev)

        # (1,0) must be x slope
        # (0,1) must be y slope
        self.assertTrue(np.allclose(grad[0], np.ones_like(x_ev) * slope_x))
        self.assertTrue(np.allclose(grad[1], np.ones_like(y_ev) * slope_y))

    def test_nonlinear_approx(self):

        # A non linear function on uniform grid
        n_grid = 100
        x = np.linspace(1, 5, n_grid)
        y = np.linspace(1, 5, n_grid)

        x_tiled, y_tiled = np.meshgrid(x, y, indexing="ij")
        z = np.sin(x_tiled) * np.log(y_tiled)

        interp = LinearFast(z, [x, y])

        # Evaluation points
        n_eval = 15
        x_ev, y_ev = np.meshgrid(
            np.linspace(2, 4, n_eval), np.linspace(4, 2, n_eval), indexing="ij",
        )

        # Get function and 1st derivatives
        grad = interp.gradient(x_ev, y_ev)

        # (1,0) must be cos(x) * ln(y)
        self.assertTrue(np.allclose(grad[0], np.cos(x_ev) * np.log(y_ev), atol=0.02))
        # (0,1) must be sin(x) / y
        self.assertTrue(np.allclose(grad[1], np.sin(x_ev) * (1 / y_ev), atol=0.02))


class TestLinearDecay(unittest.TestCase):
    """ 
    Checks the linear interpolators with limiting extrapolators
    """

    def setUp(self):

        # Set up a bilinear extrapolator for a simple function
        self.x = np.linspace(0, 10, 11)
        self.y = np.linspace(0, 10, 11)
        x_t, y_t = np.meshgrid(self.x, self.y, indexing="ij")
        z = 2 * x_t + y_t

        # Base interpolator
        interp = LinearFast(z, [self.x, self.y], extrap_mode="linear")

        # An extrapolator that limits to the same function
        self.interp_same = DecayInterp(
            interp,
            limit_fun=lambda x, y: 2 * x + y,
            limit_grad=lambda x, y: [np.ones_like(x) * 2, np.ones_like(y) * 1],
            extrap_method="decay_hark",
        )

        # another that limits to a shifted function with different slopes
        self.interp_shift = DecayInterp(
            interp,
            limit_fun=lambda x, y: np.sqrt(x),
            limit_grad=lambda x, y: [0.5 * np.power(x, -0.5), np.zeros_like(y)],
            extrap_method="decay_hark",
        )

    def test_extrap(self):

        x = np.linspace(5, 60, 100)
        y = np.linspace(5, 60, 100)
        x_t, y_t = np.meshgrid(x, y, indexing="ij")

        z_same = self.interp_same(x_t, y_t)
        z_shift = self.interp_shift(x_t, y_t)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(x_t, y_t, z_same)
        plt.show()

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(x_t, y_t, z_shift)
        plt.show()
        print("hi!")

    def test_compare_smooth_with_LinearInterp(self):

        lim_slope = 0.1
        lim_inter = 1.0

        x = np.linspace(0, 10, 20)
        y = lim_inter + lim_slope * x - 1 / (0.2 * x + 1)

        base_lim_interp = LinearInterp(
            x, y, intercept_limit=lim_inter, slope_limit=lim_slope
        )
        efor_lim_interp = DecayInterp(
            LinearFast(y, [x]),
            limit_fun=lambda x: lim_inter + lim_slope * x,
            limit_grad=lambda x: [lim_slope * np.ones_like(x)],
            extrap_method="decay_hark",
        )

        x_eval = np.linspace(0, 20, 200)
        base_vals = base_lim_interp(x_eval)
        efor_vals = efor_lim_interp(x_eval)

        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(x_eval, base_vals, label="base")
        plt.plot(x_eval, efor_vals, label="efor")
        plt.legend()
        plt.show()

        self.assertTrue(np.allclose(base_vals, efor_vals))

    def test_extrap_1D(self):

        x = np.linspace(0, 2, 50)
        interp = LinearFast(np.sqrt(x), [x])

        limit_fun = lambda x: 1 + -2 * x
        limit_grad = lambda x: [-2 * np.ones_like(x)]

        dec_pr = DecayInterp(
            interp,
            limit_fun=limit_fun,
            limit_grad=limit_grad,
            extrap_method="decay_prop",
        )
        dec_pa = DecayInterp(
            interp, limit_fun=limit_fun, limit_grad=limit_grad, extrap_method="paste",
        )
        dec_ha = DecayInterp(
            interp,
            limit_fun=limit_fun,
            limit_grad=limit_grad,
            extrap_method="decay_hark",
        )
        x_ev = np.linspace(0, 10, 200)

        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(x_ev, dec_pr(x_ev), label="prop")
        plt.plot(x_ev, dec_pa(x_ev), label="past")
        plt.plot(x_ev, dec_ha(x_ev), label="decay_hark")
        plt.plot(x_ev, limit_fun(x_ev), "--", label="limit")
        plt.legend()
        plt.show()
        print("hi!")

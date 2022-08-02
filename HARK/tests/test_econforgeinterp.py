from typing import no_type_check_decorator
import unittest
import numpy as np

from HARK.interpolation import LinearInterp, BilinearInterp
from HARK.econforgeinterp import LinearFast, LinearFastDecay
from HARK.core import distance_metric

from interpolation.splines import extrap_options as xto

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


class TestLinearDecay(unittest.TestCase):
    """ 
    Checks the linear interpolators with limiting extrapolators
    """

    def setUp(self):

        # Set up a bilinear extrapolator for a simple function
        self.x = np.linspace(0,10,11)
        self.y = np.linspace(0,10,11)
        x_t, y_t = np.meshgrid(self.x, self.y, indexing='ij')
        z = 2*x_t + y_t

        # And extrapolator that limits to the same function
        self.interp_same = LinearFastDecay(
            z,
            [self.x, self.y],
            limit_func=lambda x, y: 2*x + y,
            extrap_options=xto.LINEAR,
        )

        # another that limits to a shifted function with different slopes
        self.interp_shift = LinearFastDecay(
            z,
            [self.x, self.y],
            limit_func=lambda x, y: np.sqrt(x),
            extrap_options=xto.LINEAR,
        )
    
    def test_extrap(self):

        x = np.linspace(5,30,50)
        y = np.linspace(5,30,50)
        x_t, y_t = np.meshgrid(x,y,indexing='ij')

        z_same = self.interp_same(x_t,y_t)
        z_shift = self.interp_shift(x_t, y_t)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(x_t, y_t, z_same)
        plt.show()

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(x_t, y_t, z_shift)
        plt.show()
        print('hi!')
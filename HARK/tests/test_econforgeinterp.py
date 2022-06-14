from typing import no_type_check_decorator
import unittest
import numpy as np

from HARK.interpolation import LinearInterp
from HARK.econforgeinterp import LinearFast
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

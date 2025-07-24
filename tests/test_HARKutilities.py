"""
This file implements unit tests to check HARK/utilities.py
"""

# Bring in modules we need
import unittest

import matplotlib.pyplot as plt
import numpy as np

from HARK.rewards import (
    CRRAutility,
    CRRAutilityP,
    CRRAutilityPP,
    CRRAutilityPPP,
    CRRAutilityPPPP,
)
from HARK.utilities import make_assets_grid, make_figs, make_grid_exp_mult


class testsForHARKutilities(unittest.TestCase):
    def setUp(self):
        self.c_vals = np.linspace(0.5, 10.0, 20)
        self.CRRA_vals = np.linspace(1.0, 10.0, 10)

    def first_diff_approx(self, func, x, delta, *args):
        """
        Take the first (centered) difference approximation to the derivative of a function.

        """
        return (func(x + delta, *args) - func(x - delta, *args)) / (2.0 * delta)

    def derivative_func_comparison(self, deriv, func):
        """
        This method computes the first difference approximation to the derivative of a function
        "func" and the (supposedly) closed-form derivative of that function ("deriv") over a
        grid.  It then checks that these two things are "close enough."
        """

        # Loop through different values of consumption
        for c in self.c_vals:
            # Loop through different values of risk aversion
            for CRRA in self.CRRA_vals:
                # Calculate the difference between the derivative of the function and the
                # first difference approximation to that derivative.
                diff = abs(
                    deriv(c, CRRA) - self.first_diff_approx(func, c, 0.000001, CRRA)
                )

                # Make sure the derivative and its approximation are close
                self.assertLess(diff, 0.01)

    def test_CRRAutilityP(self):
        # Test the first derivative of the utility function
        self.derivative_func_comparison(CRRAutilityP, CRRAutility)

    def test_CRRAutilityPP(self):
        # Test the second derivative of the utility function
        self.derivative_func_comparison(CRRAutilityPP, CRRAutilityP)

    def test_CRRAutilityPPP(self):
        # Test the third derivative of the utility function
        self.derivative_func_comparison(CRRAutilityPPP, CRRAutilityPP)

    def test_CRRAutilityPPPP(self):
        # Test the fourth derivative of the utility function
        self.derivative_func_comparison(CRRAutilityPPPP, CRRAutilityPPP)

    def test_asset_grid(self):
        # Test linear asset grid
        params = {
            "aXtraMin": 0.0,
            "aXtraMax": 1.0,
            "aXtraCount": 5,
            "aXtraExtra": None,
            "aXtraNestFac": -1,
        }

        aXtraGrid = make_assets_grid(**params)
        test = np.unique(np.diff(aXtraGrid).round(decimals=3))
        self.assertEqual(test.size, 1)

    def test_make_figs(self):
        # Test the make_figs() function with a trivial output
        plt.figure()
        plt.plot(np.linspace(1, 5, 40), np.linspace(4, 8, 40))
        make_figs("test", True, False, target_dir="")
        plt.clf()

    def test_make_grid_exp_mult_negative_min(self):
        # Test make_grid_exp_mult with negative minimum values (Issue #1576)
        # This should not produce NaN values
        result = make_grid_exp_mult(-1, 3, 5)
        self.assertFalse(np.any(np.isnan(result)))
        self.assertEqual(len(result), 5)
        self.assertAlmostEqual(result[0], -1.0, places=6)
        self.assertAlmostEqual(result[-1], 3.0, places=6)
        
        # Test another negative case
        result2 = make_grid_exp_mult(-0.5, 2, 4, 10)
        self.assertFalse(np.any(np.isnan(result2)))
        self.assertEqual(len(result2), 4)
        self.assertAlmostEqual(result2[0], -0.5, places=6)
        self.assertAlmostEqual(result2[-1], 2.0, places=6)
        
        # Test exponential spacing with negative minimum
        result3 = make_grid_exp_mult(-1, 3, 5, timestonest=0)
        self.assertFalse(np.any(np.isnan(result3)))
        self.assertEqual(len(result3), 5)
        self.assertAlmostEqual(result3[0], -1.0, places=6)
        self.assertAlmostEqual(result3[-1], 3.0, places=6)

    def test_make_grid_exp_mult_proportional(self):
        # Test that the function produces proportional grids (Issue #1576)
        # Grid for [0,1] should be proportional to grids for other intervals
        grid_01 = make_grid_exp_mult(0, 1, 5, 10)
        grid_24 = make_grid_exp_mult(2, 4, 5, 10)
        
        # grid_24 should equal 2 + 2 * grid_01
        expected = 2 + 2 * grid_01
        np.testing.assert_allclose(grid_24, expected, rtol=1e-10)
        
        # Test with negative bounds
        grid_neg = make_grid_exp_mult(-3, -1, 5, 10)
        expected_neg = -3 + 2 * grid_01
        np.testing.assert_allclose(grid_neg, expected_neg, rtol=1e-10)

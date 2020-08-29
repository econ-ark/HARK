"""
This file implements unit tests to check HARK/utilities.py
"""
from __future__ import print_function, division
from __future__ import absolute_import

import HARK.utilities

# Bring in modules we need
import unittest
import numpy as np


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
        self.derivative_func_comparison(
            HARK.utilities.CRRAutilityP, HARK.utilities.CRRAutility
        )

    def test_CRRAutilityPP(self):
        # Test the second derivative of the utility function
        self.derivative_func_comparison(
            HARK.utilities.CRRAutilityPP, HARK.utilities.CRRAutilityP
        )

    def test_CRRAutilityPPP(self):
        # Test the third derivative of the utility function
        self.derivative_func_comparison(
            HARK.utilities.CRRAutilityPPP, HARK.utilities.CRRAutilityPP
        )

    def test_CRRAutilityPPPP(self):
        # Test the fourth derivative of the utility function
        self.derivative_func_comparison(
            HARK.utilities.CRRAutilityPPPP, HARK.utilities.CRRAutilityPPP
        )

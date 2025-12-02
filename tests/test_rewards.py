"""
This file implements unit tests to check HARK/rewards.py
"""

# Bring in modules we need
import unittest
import numpy as np

from HARK.rewards import (
    CRRAutility,
    CRRAutility_inv,
    CRRAutilityP,
    CRRAutilityPP,
    CRRAutilityPPP,
    CRRAutilityPPPP,
    CARAutility,
    CARAutilityP,
    CARAutilityPP,
    CARAutilityPPP,
    CARAutilityP_inv,
    CARAutility_inv,
    CARAutilityP_invP,
    UtilityFuncCARA,
    UtilityFuncCRRA,
    UtilityFuncStoneGeary,
    UtilityFuncCobbDouglas,
    UtilityFuncCobbDouglasCRRA,
    UtilityFuncConstElastSubs,
    UtilityFunction,
)


class testsForCRRA(unittest.TestCase):
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

    def test_structure(self):
        U = UtilityFuncCRRA(2.5)
        x = 2.0
        a = U(x)
        b = U(x, order=0)
        self.assertAlmostEqual(a, b)
        c = U(x, order=1)
        d = U(x, order=2)
        e = U(x, order=3)
        f = U(x, order=4)
        self.assertRaises(ValueError, U, x, 5)
        self.assertRaises(ValueError, U.inverse, x, (2, 1))


class testsForCARA(unittest.TestCase):
    def setUp(self):
        self.c_vals = np.linspace(0.5, 10.0, 20)
        self.CARA_vals = np.linspace(0.005, 0.5, 21)

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
            for CARA in self.CARA_vals:
                # Calculate the difference between the derivative of the function and the
                # first difference approximation to that derivative.
                diff = abs(
                    deriv(c, CARA) - self.first_diff_approx(func, c, 0.000001, CARA)
                )

                # Make sure the derivative and its approximation are close
                self.assertLess(diff, 0.01)

    def test_CARAutilityP(self):
        # Test the first derivative of the utility function
        self.derivative_func_comparison(CARAutilityP, CARAutility)

    def test_CARAutilityPP(self):
        # Test the second derivative of the utility function
        self.derivative_func_comparison(CARAutilityPP, CARAutilityP)

    def test_CARAutilityPPP(self):
        # Test the third derivative of the utility function
        self.derivative_func_comparison(CARAutilityPPP, CARAutilityPP)

    def test_CARAutility_inv(self):
        # Test the inverse of the utility function
        C = np.linspace(0.01, 20.0, 201)
        X = CARAutility_inv(CARAutility(C, 0.05), 0.05)
        self.assertTrue(np.all(np.isclose(C, X)))

    def test_CARAutilityP_inv(self):
        # Test the inverse of the marginal utility function
        C = np.linspace(0.01, 20.0, 201)
        X = CARAutilityP_inv(CARAutilityP(C, 0.05), 0.05)
        self.assertTrue(np.all(np.isclose(C, X)))

    def test_CARAutilityP_invP(self):
        # Test the first derivative of the inverse of the marginal utility function
        self.derivative_func_comparison(CARAutilityP_invP, CARAutilityP_inv)

    def test_structure(self):
        U = UtilityFuncCARA(0.05)
        x = 2.0
        y = 0.6
        a = U(x)
        b = U(x, order=0)
        self.assertAlmostEqual(a, b)
        c = U(x, order=1)
        d = U(x, order=2)
        e = U(x, order=3)
        self.assertRaises(ValueError, U, x, 4)
        f = U.inverse(y, order=(0, 0))
        g = U.inverse(y, order=(1, 0))
        h = U.inverse(y, order=(0, 1))
        i = U.inverse(y, order=(1, 1))
        j = U.derinv(y)
        self.assertRaises(ValueError, U.inverse, y, (2, 1))


class testsForStoneGeary(unittest.TestCase):
    def setUp(self):
        self.U = UtilityFuncStoneGeary(2.0, shifter=1.0)
        self.x = 5.0

    def test_eval(self):
        x = self.x
        U = self.U
        a = U(x)
        b = U(x, order=0)
        self.assertAlmostEqual(a, b)
        c = U(x, order=1)
        d = U(x, order=2)
        self.assertRaises(ValueError, U, x, 3)

    def test_inverse(self):
        x = self.x
        U = self.U
        Uinv = U.inverse
        a = Uinv(x)
        b = Uinv(x, order=(0, 0))
        self.assertAlmostEqual(a, b)
        c = Uinv(x, order=(0, 1))
        d = Uinv(x, order=(1, 0))
        e = Uinv(x, order=(1, 1))
        self.assertRaises(ValueError, Uinv, x, (2, 1))


class testsForCobbDouglas(unittest.TestCase):
    def setUp(self):
        self.U = UtilityFuncCobbDouglas(c_share=0.7, d_bar=0.1)
        self.x = 5.0
        self.y = 1.0

    def test_funcs(self):
        x = self.x
        y = self.y
        U = self.U
        a = U(x, y)
        b = U.derivative(x, y, axis=0)
        c = U.derivative(x, y, axis=1)
        d = U.inverse(x, y)
        self.assertRaises(ValueError, U.derivative, x, y, 2)


class testsForCobbDouglasCRRA(unittest.TestCase):
    def setUp(self):
        self.U = UtilityFuncCobbDouglasCRRA(CRRA=2.5, c_share=0.7, d_bar=0.1)
        self.x = 5.0
        self.y = 1.0

    def test_funcs(self):
        x = self.x
        y = self.y
        U = self.U
        a = U(x, y)
        b = U.derivative(x, y, axis=0)
        c = U.derivative(x, y, axis=1)
        d = U.inverse(x, y)
        self.assertRaises(ValueError, U.derivative, x, y, 2)


class testsForCES(unittest.TestCase):
    def setUp(self):
        self.U = UtilityFuncConstElastSubs(shares=[0.4, 0.5, 0.1], subs=0.3)
        self.x = np.array([12.0, 7.0, 3.0])

    def test_funcs(self):
        U = self.U
        x = self.x
        a = U(x)
        b = U.derivative(x, 0)
        c = U.derivative(x, 1)
        d = U.derivative(x, 2)


class testsForUtilityFunction(unittest.TestCase):
    def test_valid(self):
        u = lambda c: CRRAutility(c, 3.0)
        uP = lambda c: CRRAutilityP(c, 3.0)
        uinv = lambda x: CRRAutility_inv(x, 3.0)
        U = UtilityFunction(u, uP, uinv)

        x = 5.0
        a = U(x)
        b = U.der(x)
        c = U.inv(-x)

    def test_invalid(self):
        u = lambda c: CRRAutility(c, 3.0)
        U = UtilityFunction(u)

        x = 5.0
        a = U(x)
        self.assertRaises(NotImplementedError, U.der, x)
        self.assertRaises(NotImplementedError, U.inv, -x)

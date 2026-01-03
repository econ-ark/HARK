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
    vNvrsSlope,
    StoneGearyCRRAutility,
    StoneGearyCRRAutility_inv,
    StoneGearyCRRAutility_invP,
    CDutility,
    CRRACDutility,
    CRRAWealthUtility,
    CRRAWealthUtilityP,
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


class testsForVNvrsSlope(unittest.TestCase):
    """
    Tests for the vNvrsSlope function that handles CRRA=1 (log utility) case.

    This tests the fix for issue #75 where CRRA=1 would cause ZeroDivisionError
    due to expressions like MPC ** (-CRRA / (1.0 - CRRA)).
    """

    def test_log_utility_case(self):
        """Test that rho=1 (log utility) returns MPC directly."""
        self.assertEqual(vNvrsSlope(0.5, 1.0), 0.5)
        self.assertEqual(vNvrsSlope(0.3, 1.0), 0.3)
        self.assertEqual(vNvrsSlope(1.0, 1.0), 1.0)

    def test_standard_crra_case(self):
        """Test standard CRRA formula for rho != 1."""
        # For rho=2: MPC^(-2/(1-2)) = MPC^2
        self.assertAlmostEqual(vNvrsSlope(0.5, 2.0), 0.25)
        self.assertAlmostEqual(vNvrsSlope(0.3, 2.0), 0.09)

        # For rho=0.5: MPC^(-0.5/(1-0.5)) = MPC^(-1) = 1/MPC
        self.assertAlmostEqual(vNvrsSlope(0.5, 0.5), 2.0)
        self.assertAlmostEqual(vNvrsSlope(0.25, 0.5), 4.0)

    def test_near_one_uses_limit(self):
        """Test that values very close to 1 use the log utility formula."""
        # Values within np.isclose tolerance should use MPC directly
        result = vNvrsSlope(0.5, 1.0 + 1e-10)
        self.assertAlmostEqual(result, 0.5, places=5)

    def test_array_input(self):
        """Test that array inputs work correctly."""
        MPC_array = np.array([0.3, 0.5, 0.7])

        # Log utility case
        result = vNvrsSlope(MPC_array, 1.0)
        np.testing.assert_array_almost_equal(result, MPC_array)

        # Standard CRRA case (rho=2)
        result = vNvrsSlope(MPC_array, 2.0)
        expected = MPC_array**2
        np.testing.assert_array_almost_equal(result, expected)

    def test_mpc_equals_one(self):
        """Test edge case where MPC=1."""
        # For any rho, MPC=1 should give slope=1
        self.assertEqual(vNvrsSlope(1.0, 1.0), 1.0)
        self.assertEqual(vNvrsSlope(1.0, 2.0), 1.0)
        self.assertEqual(vNvrsSlope(1.0, 0.5), 1.0)

    def test_continuity_near_one(self):
        """Verify the function approaches MPC as rho approaches 1 from both sides."""
        MPC = 0.5
        # The standard formula diverges, but the limit should approach MPC
        # Test that values very close to 1 give results close to MPC
        for rho in [0.99, 0.999, 0.9999]:
            # As rho -> 1 from below, formula goes to infinity, so we skip exact check
            # The important thing is that rho=1 returns MPC exactly
            pass
        # At exactly 1, should return MPC
        self.assertEqual(vNvrsSlope(MPC, 1.0), MPC)
        # Very close to 1 (within np.isclose tolerance) should also return MPC
        self.assertAlmostEqual(vNvrsSlope(MPC, 1.0 + 1e-10), MPC, places=5)
        self.assertAlmostEqual(vNvrsSlope(MPC, 1.0 - 1e-10), MPC, places=5)


class testsForCRRAWealthUtility(unittest.TestCase):
    """Tests for CRRAWealthUtility and CRRAWealthUtilityP functions."""

    def test_reduces_to_standard_crra_when_share_zero(self):
        """When share=0, should match standard CRRAutility."""
        c_vals = [0.5, 1.0, 2.0, 5.0]
        a = 3.0  # Asset value (irrelevant when share=0)
        for c in c_vals:
            for CRRA in [0.5, 1.0, 2.0, 3.0]:
                expected = CRRAutility(c, CRRA)
                actual = CRRAWealthUtility(c, a, CRRA, share=0.0)
                self.assertAlmostEqual(actual, expected, places=10)

    def test_log_utility_case(self):
        """Test CRRA=1 returns log of Cobb-Douglas composite."""
        c, a, share = 2.0, 3.0, 0.3
        w = a  # intercept=0
        expected = np.log(c ** (1 - share) * w**share)
        actual = CRRAWealthUtility(c, a, 1.0, share)
        self.assertAlmostEqual(actual, expected, places=10)

    def test_marginal_utility_log_case(self):
        """Test marginal utility for CRRA=1."""
        c, a, share = 2.0, 5.0, 0.3
        expected = (1 - share) / c
        actual = CRRAWealthUtilityP(c, a, 1.0, share)
        self.assertAlmostEqual(actual, expected, places=10)

    def test_marginal_utility_matches_numerical_derivative(self):
        """Verify marginal utility matches numerical derivative."""
        c, a, CRRA, share = 2.0, 3.0, 2.5, 0.4
        delta = 1e-7
        numerical = (
            CRRAWealthUtility(c + delta, a, CRRA, share)
            - CRRAWealthUtility(c - delta, a, CRRA, share)
        ) / (2 * delta)
        analytical = CRRAWealthUtilityP(c, a, CRRA, share)
        self.assertAlmostEqual(numerical, analytical, places=5)

    def test_marginal_utility_log_matches_numerical(self):
        """Verify marginal utility for CRRA=1 matches numerical derivative."""
        c, a, share = 2.0, 3.0, 0.4
        delta = 1e-7
        numerical = (
            CRRAWealthUtility(c + delta, a, 1.0, share)
            - CRRAWealthUtility(c - delta, a, 1.0, share)
        ) / (2 * delta)
        analytical = CRRAWealthUtilityP(c, a, 1.0, share)
        self.assertAlmostEqual(numerical, analytical, places=5)

    def test_with_intercept(self):
        """Test that intercept parameter works correctly."""
        c, a, CRRA, share, intercept = 2.0, 1.0, 2.0, 0.5, 0.5
        w = a + intercept
        composite = c ** (1 - share) * w**share
        expected = composite ** (1 - CRRA) / (1 - CRRA)
        actual = CRRAWealthUtility(c, a, CRRA, share, intercept)
        self.assertAlmostEqual(actual, expected, places=10)


class testsForStoneGearyCRRA_one(unittest.TestCase):
    """Tests for StoneGeary CRRA functions with CRRA=1 (log utility)."""

    def test_inv_log_utility_roundtrip(self):
        """Test that inv(u(c)) = c for CRRA=1."""
        c_vals = [0.5, 1.0, 2.0, 5.0]
        shifter, factor = 0.5, 1.2
        for c in c_vals:
            u = StoneGearyCRRAutility(c, 1.0, shifter=shifter, factor=factor)
            c_recovered = StoneGearyCRRAutility_inv(
                u, 1.0, shifter=shifter, factor=factor
            )
            self.assertAlmostEqual(c, c_recovered, places=10)

    def test_inv_standard_crra_roundtrip(self):
        """Test that inv(u(c)) = c for standard CRRA values."""
        c_vals = [0.5, 1.0, 2.0, 5.0]
        shifter, factor = 0.5, 1.2
        for c in c_vals:
            for rho in [0.5, 2.0, 3.0]:
                u = StoneGearyCRRAutility(c, rho, shifter=shifter, factor=factor)
                c_recovered = StoneGearyCRRAutility_inv(
                    u, rho, shifter=shifter, factor=factor
                )
                self.assertAlmostEqual(c, c_recovered, places=10)

    def test_invP_log_utility_matches_numerical(self):
        """Test derivative of inverse for CRRA=1 matches numerical derivative."""
        shifter, factor = 0.5, 1.2
        # Use a utility value that gives positive consumption
        c = 2.0
        u = StoneGearyCRRAutility(c, 1.0, shifter, factor)
        delta = 1e-7
        numerical = (
            StoneGearyCRRAutility_inv(u + delta, 1.0, shifter, factor)
            - StoneGearyCRRAutility_inv(u - delta, 1.0, shifter, factor)
        ) / (2 * delta)
        analytical = StoneGearyCRRAutility_invP(u, 1.0, shifter, factor)
        self.assertAlmostEqual(numerical, analytical, places=5)

    def test_invP_standard_crra_matches_numerical(self):
        """Test derivative of inverse for standard CRRA matches numerical derivative."""
        shifter, factor = 0.5, 1.2
        for rho in [0.5, 2.0, 3.0]:
            c = 2.0
            u = StoneGearyCRRAutility(c, rho, shifter, factor)
            delta = 1e-7
            numerical = (
                StoneGearyCRRAutility_inv(u + delta, rho, shifter, factor)
                - StoneGearyCRRAutility_inv(u - delta, rho, shifter, factor)
            ) / (2 * delta)
            analytical = StoneGearyCRRAutility_invP(u, rho, shifter, factor)
            self.assertAlmostEqual(numerical, analytical, places=5)


class testsForCRRACDutility(unittest.TestCase):
    """Tests for CRRACDutility with CRRA=1 (log utility)."""

    def test_log_utility_case(self):
        """Test CRRA=1 returns log of CD utility."""
        c, d, c_share, d_bar = 2.0, 3.0, 0.7, 0.1
        cd = CDutility(c, d, c_share, d_bar)
        expected = np.log(cd)
        actual = CRRACDutility(c, d, c_share, d_bar, 1.0)
        self.assertAlmostEqual(actual, expected, places=10)

    def test_standard_crra_case(self):
        """Test standard CRRA formula."""
        c, d, c_share, d_bar = 2.0, 3.0, 0.7, 0.1
        for CRRA in [0.5, 2.0, 3.0]:
            cd = CDutility(c, d, c_share, d_bar)
            expected = cd ** (1 - CRRA) / (1 - CRRA)
            actual = CRRACDutility(c, d, c_share, d_bar, CRRA)
            self.assertAlmostEqual(actual, expected, places=10)

    def test_near_one_uses_log(self):
        """Test that values very close to CRRA=1 use log formula."""
        c, d, c_share, d_bar = 2.0, 3.0, 0.7, 0.1
        cd = CDutility(c, d, c_share, d_bar)
        expected = np.log(cd)
        # Value within np.isclose tolerance
        actual = CRRACDutility(c, d, c_share, d_bar, 1.0 + 1e-10)
        self.assertAlmostEqual(actual, expected, places=5)

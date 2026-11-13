"""
The convergence gate for HARK._interp_kernels: run with numpy as their array
library, the kernels must reproduce LinearInterp and BilinearInterp bit for bit.

The JAX backend evaluates with these kernels. While this gate passes, the numpy
classes could evaluate with them too, leaving one implementation of each formula.
These tests need no JAX.
"""

import unittest

import numpy as np

from HARK import _interp_kernels as kernels
from HARK.interpolation import (
    BilinearInterp,
    LinearInterp,
    get_backend,
    set_backend,
    using_backend,
)


def assert_bits_equal(test, a, b):
    """Equal as IEEE bit patterns, which also compares NaNs and signed zeros."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    test.assertEqual(a.shape, b.shape)
    np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))


def exp_grid(lo, hi, n):
    """An exponentially spaced grid, the kind HARK's solvers use."""
    return lo + (np.exp(np.linspace(0.0, np.log(hi - lo + 1.0), n)) - 1.0)


class testLinearKernelGate(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.x_list = exp_grid(0.001, 20.0, 40)
        self.y_list = np.log(1.0 + self.x_list) + 0.3 * np.sqrt(self.x_list)
        inside = rng.uniform(0.001, 20.0, 500)
        outside = np.array([-5.0, -1e-9, 0.0, 20.0 + 1e-9, 25.0, 1e3])
        self.x = np.concatenate([inside, self.x_list, outside, [np.nan]])

    def _check(self, interp, decay):
        for want_y, want_dydx in [(True, False), (False, True), (True, True)]:
            expected = interp._evalOrDer(self.x, want_y, want_dydx)
            y, dydx = kernels.linear_1d(
                np,
                interp.x_list,
                interp.y_list,
                self.x,
                want_y,
                want_dydx,
                interp.lower_extrap,
                decay,
            )
            got = [out for out in (y, dydx) if out is not None]
            self.assertEqual(len(got), len(expected))
            for g, e in zip(got, expected):
                assert_bits_equal(self, g, e)

    def test_plain(self):
        for lower_extrap in (False, True):
            interp = LinearInterp(self.x_list, self.y_list, lower_extrap=lower_extrap)
            self.assertFalse(interp.decay_extrap)
            self._check(interp, None)

    def test_decay_extrapolation(self):
        # A limiting linear function above the grid, as a solved consumption function has
        slope_limit = 0.05
        intercept_limit = self.y_list[-1] - slope_limit * self.x_list[-1] + 0.2
        for lower_extrap in (False, True):
            interp = LinearInterp(
                self.x_list,
                self.y_list,
                intercept_limit=intercept_limit,
                slope_limit=slope_limit,
                lower_extrap=lower_extrap,
            )
            self.assertTrue(interp.decay_extrap)
            decay = (
                interp.decay_extrap_A,
                interp.decay_extrap_B,
                interp.intercept_limit,
                interp.slope_limit,
            )
            self._check(interp, decay)


class testBilinearKernelGate(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.x_list = exp_grid(0.0, 10.0, 25)
        self.y_list = np.linspace(0.5, 2.0, 9)
        self.f_values = (
            np.log1p(np.outer(self.x_list, self.y_list))
            + np.sqrt(self.x_list)[:, None] * self.y_list[None, :] ** 0.5
        )
        n = 400
        self.x = np.concatenate([rng.uniform(-1.0, 12.0, n), self.x_list[:9], [np.nan]])
        self.y = np.concatenate([rng.uniform(0.0, 2.5, n), self.y_list, [1.0]])

    def test_level_and_derivatives(self):
        interp = BilinearInterp(self.f_values, self.x_list, self.y_list)
        for which, method in [
            ("f", interp._evaluate),
            ("dfdx", interp._derX),
            ("dfdy", interp._derY),
        ]:
            got = kernels.bilinear_2d(
                np, self.f_values, self.x_list, self.y_list, self.x, self.y, which
            )
            assert_bits_equal(self, got, method(self.x, self.y))

    def test_unknown_output(self):
        with self.assertRaises(ValueError):
            kernels.bilinear_2d(
                np, self.f_values, self.x_list, self.y_list, self.x, self.y, "dfdz"
            )


class testBackendSelectionWithoutJAX(unittest.TestCase):
    """What can be checked without JAX installed."""

    def test_default_is_numpy(self):
        self.assertEqual(get_backend(), ("numpy", None))
        interp = LinearInterp(np.array([0.0, 1.0]), np.array([0.0, 2.0]))
        self.assertIs(type(interp), LinearInterp)

    def test_unknown_backend_refused(self):
        with self.assertRaises(ValueError):
            set_backend("torch")
        with self.assertRaises(ValueError):
            LinearInterp(np.array([0.0, 1.0]), np.array([0.0, 2.0]), backend="torch")
        self.assertEqual(get_backend(), ("numpy", None))

    def test_numpy_float32_refused(self):
        with self.assertRaises(NotImplementedError):
            LinearInterp(np.array([0.0, 1.0]), np.array([0.0, 2.0]), dtype="float32")
        with self.assertRaises(NotImplementedError):
            with using_backend("numpy", dtype="float32"):
                pass

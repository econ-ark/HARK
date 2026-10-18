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
    LinearInterpOnInterp1D,
    LowerEnvelope2D,
    get_backend,
    set_backend,
    using_backend,
)
from HARK.rewards import CRRAutilityP, CRRAutilityPP


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


def stack_rows(interps):
    """The stacked, +inf-padded rows ``linear_rows`` takes, from numpy
    LinearInterps, as JaxLinearInterpOnInterp1D builds them."""
    lengths = [f.x_list.size for f in interps]
    width = max(lengths)
    x_rows = np.full((len(interps), width), np.inf)
    y_rows = np.zeros((len(interps), width))
    decay = np.zeros((4, len(interps)))
    for k, f in enumerate(interps):
        n = lengths[k]
        x_rows[k, :n] = f.x_list
        y_rows[k, :n] = f.y_list
        y_rows[k, n:] = f.y_list[-1]
        if f.decay_extrap:
            decay[:, k] = [
                f.decay_extrap_A,
                f.decay_extrap_B,
                f.intercept_limit,
                f.slope_limit,
            ]
    has_decay = np.array([f.decay_extrap for f in interps])
    lower_rows = np.array([f.lower_extrap for f in interps])
    return x_rows, y_rows, np.array(lengths), lower_rows, (has_decay, *decay)


class testLinearOnLinearKernelGate(unittest.TestCase):
    """Inner grids of different lengths, mixed extrapolation, some rows decaying."""

    def setUp(self):
        rng = np.random.default_rng(4)
        self.y_list = np.array([0.5, 0.8, 1.0, 1.4, 2.0])
        self.inner = []
        for k, (n, lower) in enumerate(
            [(12, False), (30, True), (7, False), (21, True), (16, False)]
        ):
            x = exp_grid(0.01 * (k + 1), 15.0 + k, n)
            y = np.log1p(x) * (1.0 + 0.1 * k)
            kwargs = {"lower_extrap": lower}
            if k in (1, 3):
                kwargs["slope_limit"] = 0.04
                kwargs["intercept_limit"] = y[-1] - 0.04 * x[-1] + 0.1
            self.inner.append(LinearInterp(x, y, **kwargs))
        knots = np.concatenate([f.x_list for f in self.inner])
        n_q = knots.size + 400
        self.x = np.concatenate(
            [knots, rng.uniform(-1.0, 25.0, 400), [-2.0, 30.0, np.nan, 3.0]]
        )
        self.y = np.concatenate([rng.uniform(0.4, 2.1, n_q), [1.0, 1.0, 1.0, np.nan]])
        # Queries exactly on the outer grid, too
        self.y[: self.y_list.size] = self.y_list

    def test_level_and_derivatives(self):
        interp = LinearInterpOnInterp1D(self.inner, self.y_list)
        x_rows, y_rows, n_rows, lower_rows, decay_rows = stack_rows(self.inner)
        for which, method in [
            ("f", interp._evaluate),
            ("dfdx", interp._derX),
            ("dfdy", interp._derY),
        ]:
            got = kernels.linear_on_linear_2d(
                np,
                x_rows,
                y_rows,
                n_rows,
                self.y_list,
                self.x,
                self.y,
                which,
                lower_rows,
                decay_rows,
            )
            assert_bits_equal(self, got, method(self.x, self.y))


class testEnvelopeAndCRRAKernelGate(unittest.TestCase):
    def test_lower_envelope(self):
        rng = np.random.default_rng(5)
        x_list = np.linspace(0.0, 5.0, 11)
        y_list = np.linspace(0.0, 2.0, 5)
        f1 = BilinearInterp(np.add.outer(x_list, y_list), x_list, y_list)
        f2 = BilinearInterp(
            np.sqrt(np.add.outer(4.0 * x_list, y_list) + 1.0), x_list, y_list
        )
        x = np.concatenate([rng.uniform(-0.5, 5.5, 300), x_list])
        y = np.concatenate([rng.uniform(-0.2, 2.2, 300), np.full(11, 1.0)])
        for nan_bool in (True, False):
            env = LowerEnvelope2D(f1, f2, nan_bool=nan_bool)
            values = np.stack([f1(x, y), f2(x, y)], axis=1)
            assert_bits_equal(
                self,
                kernels.envelope_level(np, values, nan_bool),
                env._evaluate(x, y),
            )
            for attr, method in [
                ("derivativeX", env._derX),
                ("derivativeY", env._derY),
            ]:
                partials = np.stack(
                    [getattr(f1, attr)(x, y), getattr(f2, attr)(x, y)], axis=1
                )
                assert_bits_equal(
                    self,
                    kernels.envelope_partial(np, values, partials, nan_bool),
                    method(x, y),
                )

    def test_crra(self):
        c = np.concatenate([np.linspace(0.01, 5.0, 200), [-1.0, 0.0, np.nan]])
        for rho in (1.0, 2.0, 0.5):
            with np.errstate(divide="ignore", invalid="ignore"):
                assert_bits_equal(
                    self, kernels.crra_marg_utility(np, c, rho), CRRAutilityP(c, rho)
                )
                assert_bits_equal(
                    self,
                    kernels.crra_marg_marg_utility(np, c, rho),
                    CRRAutilityPP(c, rho),
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

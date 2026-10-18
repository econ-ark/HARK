"""
The JAX backend of LinearInterp and BilinearInterp, checked against the numpy
backend at both precisions. Skipped when JAX is not installed.

This module turns on JAX's 64-bit mode when it is imported, so that float64 can
be tested; the float32 tests ask for float32 explicitly. The flag is process-wide,
so other JAX code run in the same test worker sees it on.
"""

import copy
import pickle
import unittest

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from HARK.interpolation import (  # noqa: E402
    BilinearInterp,
    LinearInterp,
    get_backend,
    set_backend,
    using_backend,
)

# The documented tolerances against the numpy backend. Levels are held to RTOL of
# their scale. A derivative is a difference of nearby values divided by the cell
# width, so its rounding error is about eps * scale / width: derivatives are held
# to DERIV_EPS machine epsilons of the value scale over the narrowest cell.
RTOL_64 = 1e-14
RTOL_32 = 1e-6
DERIV_EPS = 16


def exp_grid(lo, hi, n):
    return lo + (np.exp(np.linspace(0.0, np.log(hi - lo + 1.0), n)) - 1.0)


def assert_matches(got, expected, rtol):
    """``got`` within ``rtol`` of ``expected``, relative to the scale of
    ``expected`` so that values near zero are not held to a relative bound."""
    got = np.asarray(got, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    scale = np.nanmax(np.abs(expected))
    np.testing.assert_allclose(got, expected, rtol=rtol, atol=rtol * scale)


def assert_derivative_matches(got, expected, dtype, values, *grids):
    """``got`` within the derivative tolerance above of ``expected``."""
    width = min(np.min(np.diff(grid)) for grid in grids)
    atol = DERIV_EPS * np.finfo(dtype).eps * np.max(np.abs(values)) / width
    got = np.asarray(got, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    np.testing.assert_allclose(got, expected, rtol=0, atol=atol)


class testLinearInterpJAX(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        self.x_list = exp_grid(0.001, 20.0, 40)
        self.y_list = np.log(1.0 + self.x_list) + 0.3 * np.sqrt(self.x_list)
        self.x = np.concatenate(
            [rng.uniform(-1.0, 30.0, 500), self.x_list, [-5.0, 25.0, np.nan]]
        )
        slope_limit = 0.05
        self.decay_args = dict(
            intercept_limit=self.y_list[-1] - slope_limit * self.x_list[-1] + 0.2,
            slope_limit=slope_limit,
        )

    def _pair(self, dtype, **kwargs):
        numpy_interp = LinearInterp(self.x_list, self.y_list, **kwargs)
        jax_interp = LinearInterp(
            self.x_list, self.y_list, backend="jax", dtype=dtype, **kwargs
        )
        return numpy_interp, jax_interp

    def test_parity(self):
        for dtype, rtol in [("float64", RTOL_64), ("float32", RTOL_32)]:
            for kwargs in [{}, {"lower_extrap": True}, self.decay_args]:
                with self.subTest(dtype=dtype, kwargs=kwargs):
                    f, g = self._pair(dtype, **kwargs)
                    self.assertEqual(g(self.x).dtype, np.dtype(dtype))
                    assert_matches(g(self.x), f(self.x), rtol)
                    assert_derivative_matches(
                        g.derivative(self.x),
                        f.derivative(self.x),
                        dtype,
                        self.y_list,
                        self.x_list,
                    )
                    y, dydx = g.eval_with_derivative(self.x)
                    y0, dydx0 = f.eval_with_derivative(self.x)
                    assert_matches(y, y0, rtol)
                    assert_derivative_matches(
                        dydx, dydx0, dtype, self.y_list, self.x_list
                    )

    def test_class_and_shapes(self):
        f, g = self._pair("float64")
        self.assertIsInstance(g, LinearInterp)
        self.assertIsInstance(g(2.0), jax.Array)
        self.assertEqual(g(2.0).shape, ())
        q = self.x[:12].reshape(3, 4)
        self.assertEqual(g(q).shape, (3, 4))
        assert_matches(g(q), f(q), RTOL_64)

    def test_inside_jit_vmap_and_grad(self):
        f, g = self._pair("float64", **self.decay_args)
        q = self.x[np.isfinite(self.x) & (self.x > self.x_list[0])]
        jitted = jax.jit(lambda interp, q: interp(q))(g, q)
        assert_matches(jitted, f(q), RTOL_64)
        slopes = jax.vmap(jax.grad(lambda x: g(x)))(q)
        assert_matches(slopes, f.derivative(q), RTOL_64)

    def test_constructed_inside_jit_and_vmap(self):
        # Built from traced arrays, with no round trip to the host
        q = self.x[np.isfinite(self.x)]
        f = LinearInterp(self.x_list, self.y_list, lower_extrap=True)
        built = jax.jit(
            lambda x, y, q: LinearInterp(x, y, lower_extrap=True, backend="jax")(q)
        )(self.x_list, self.y_list, q)
        assert_matches(built, f(q), RTOL_64)
        scales = np.array([1.0, 2.0, 3.0])
        per_state = jax.vmap(
            lambda s: LinearInterp(self.x_list, s * self.y_list, backend="jax")(q)
        )(scales)
        for k, s in enumerate(scales):
            assert_matches(
                per_state[k], LinearInterp(self.x_list, s * self.y_list)(q), RTOL_64
            )
        # The decay extrapolation depends on the values, so it needs concrete grids
        with self.assertRaises(NotImplementedError):
            jax.jit(
                lambda y: LinearInterp(
                    self.x_list, y, backend="jax", **self.decay_args
                )(q)
            )(self.y_list)

    def test_copy_and_pickle(self):
        f, g = self._pair("float32", **self.decay_args)
        for h in (copy.deepcopy(g), pickle.loads(pickle.dumps(g))):
            self.assertIs(type(h), type(g))
            np.testing.assert_array_equal(np.asarray(h(self.x)), np.asarray(g(self.x)))

    def test_unsupported_options_refused(self):
        with self.assertRaises(NotImplementedError):
            LinearInterp(self.x_list, self.y_list, pre_compute=True, backend="jax")
        with self.assertRaises(NotImplementedError):
            LinearInterp(self.x_list, self.y_list, indexer=lambda x: x, backend="jax")

        class MyInterp(LinearInterp):
            pass

        with self.assertRaises(NotImplementedError):
            MyInterp(self.x_list, self.y_list, backend="jax")

    def test_float64_needs_x64(self):
        jax.config.update("jax_enable_x64", False)
        try:
            with self.assertRaises(ValueError):
                LinearInterp(self.x_list, self.y_list, backend="jax")
            g = LinearInterp(self.x_list, self.y_list, backend="jax", dtype="float32")
            self.assertEqual(g(self.x).dtype, np.float32)
        finally:
            jax.config.update("jax_enable_x64", True)


class testBilinearInterpJAX(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.x_list = exp_grid(0.0, 10.0, 25)
        self.y_list = np.linspace(0.5, 2.0, 9)
        self.f_values = np.log1p(np.outer(self.x_list, self.y_list))
        self.x = rng.uniform(-1.0, 12.0, 400)
        self.y = rng.uniform(0.0, 2.5, 400)

    def test_parity(self):
        f = BilinearInterp(self.f_values, self.x_list, self.y_list)
        for dtype, rtol in [("float64", RTOL_64), ("float32", RTOL_32)]:
            with self.subTest(dtype=dtype):
                g = BilinearInterp(
                    self.f_values, self.x_list, self.y_list, backend="jax", dtype=dtype
                )
                self.assertIsInstance(g, BilinearInterp)
                assert_matches(g(self.x, self.y), f(self.x, self.y), rtol)
                grids = (self.f_values, self.x_list, self.y_list)
                assert_derivative_matches(
                    g.derivativeX(self.x, self.y),
                    f.derivativeX(self.x, self.y),
                    dtype,
                    *grids,
                )
                assert_derivative_matches(
                    g.derivativeY(self.x, self.y),
                    f.derivativeY(self.x, self.y),
                    dtype,
                    *grids,
                )
                # A scalar broadcasts against an array, as in the numpy class
                assert_matches(g(self.x, 1.0), f(self.x, 1.0), rtol)

    def test_constructed_inside_jit(self):
        f = BilinearInterp(self.f_values, self.x_list, self.y_list)
        out = jax.jit(
            lambda v, x, y: BilinearInterp(v, x, y, backend="jax")(self.x, self.y)
        )(self.f_values, self.x_list, self.y_list)
        assert_matches(out, f(self.x, self.y), RTOL_64)

    def test_inside_jit(self):
        f = BilinearInterp(self.f_values, self.x_list, self.y_list)
        g = BilinearInterp(self.f_values, self.x_list, self.y_list, backend="jax")
        out = jax.jit(lambda interp, x, y: interp(x, y))(g, self.x, self.y)
        assert_matches(out, f(self.x, self.y), RTOL_64)


class testBackendDefault(unittest.TestCase):
    def setUp(self):
        self.x_list = np.array([0.0, 1.0, 3.0])
        self.y_list = np.array([0.0, 2.0, 3.0])

    def test_using_backend(self):
        with using_backend("jax", dtype="float32"):
            self.assertEqual(get_backend(), ("jax", "float32"))
            g = LinearInterp(self.x_list, self.y_list)
            self.assertEqual(g.dtype, np.float32)
            # An explicit numpy backend still gets float64
            self.assertIs(
                type(LinearInterp(self.x_list, self.y_list, backend="numpy")),
                LinearInterp,
            )
        self.assertEqual(get_backend(), ("numpy", None))
        self.assertIs(type(LinearInterp(self.x_list, self.y_list)), LinearInterp)

    def test_set_backend(self):
        try:
            set_backend("jax")
            g = BilinearInterp(np.eye(3), self.x_list, self.x_list)
            self.assertIsNot(type(g), BilinearInterp)
            self.assertEqual(g.dtype, np.float64)
        finally:
            set_backend("numpy")
        self.assertEqual(get_backend(), ("numpy", None))

    def test_numpy_objects_unchanged_by_default(self):
        # Copying a numpy interpolator under a JAX default keeps it numpy
        f = LinearInterp(self.x_list, self.y_list)
        with using_backend("jax"):
            self.assertIs(type(copy.deepcopy(f)), LinearInterp)
            self.assertIs(type(pickle.loads(pickle.dumps(f))), LinearInterp)

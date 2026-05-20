"""
Bit-precision parity tests for ``HARK.interpolation_jax`` vs the numpy/numba
implementations in ``HARK.interpolation``.

Each test uses fixed-seed random inputs and asserts the JAX result matches the
existing HARK implementation to a tight relative tolerance (1e-10 default).

Skipped if JAX is not installed.
"""

import os
import unittest

import numpy as np

try:
    os.environ.setdefault("JAX_ENABLE_X64", "True")
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from HARK import interpolation_jax as ij

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

from HARK.interpolation import (
    LinearInterp,
    BilinearInterp,
    LinearInterpOnInterp1D,
)


PRECISION = 1e-10


def _assert_close(jax_val, hark_val, tol=PRECISION, name=""):
    jax_val = np.asarray(jax_val)
    hark_val = np.asarray(hark_val)
    finite_match = np.isfinite(jax_val) == np.isfinite(hark_val)
    assert finite_match.all(), (
        f"{name}: NaN pattern mismatch — "
        f"JAX has {(~np.isfinite(jax_val)).sum()} NaNs, "
        f"HARK has {(~np.isfinite(hark_val)).sum()} NaNs"
    )
    finite = np.isfinite(jax_val)
    if not finite.any():
        return
    rel = np.abs(
        (jax_val[finite] - hark_val[finite])
        / np.maximum(np.abs(hark_val[finite]), 1e-15)
    )
    assert rel.max() < tol, (
        f"{name}: max relative diff = {rel.max():.2e} (> tol {tol:.0e})"
    )


@unittest.skipUnless(HAS_JAX, "JAX not installed")
class TestLinearInterp1DJax(unittest.TestCase):
    """``linear_interp_1d`` vs ``HARK.interpolation.LinearInterp``."""

    def test_no_lower_extrap(self):
        rng = np.random.default_rng(42)
        x_grid = np.sort(rng.uniform(0, 10, 20))
        y_vals = np.sin(x_grid) + 0.1 * x_grid
        x_query = np.concatenate(
            [
                rng.uniform(x_grid.min(), x_grid.max(), 50),
                np.array([x_grid.min() - 1.0, x_grid.min() - 0.001]),
                np.array([x_grid.max() + 0.001, x_grid.max() + 5.0]),
                x_grid,
            ]
        )
        hark = LinearInterp(x_grid, y_vals, lower_extrap=False)(x_query)
        jaxv = ij.linear_interp_1d(
            x_grid, y_vals, jnp.asarray(x_query), lower_extrap=False
        )
        _assert_close(jaxv, hark, name="LinearInterp no decay extrap")

    def test_lower_extrap(self):
        rng = np.random.default_rng(43)
        x_grid = np.sort(rng.uniform(0, 10, 15))
        y_vals = np.exp(-x_grid)
        x_query = rng.uniform(-5, 15, 100)
        hark = LinearInterp(x_grid, y_vals, lower_extrap=True)(x_query)
        jaxv = ij.linear_interp_1d(
            x_grid, y_vals, jnp.asarray(x_query), lower_extrap=True
        )
        _assert_close(jaxv, hark, name="LinearInterp lower_extrap")


@unittest.skipUnless(HAS_JAX, "JAX not installed")
class TestBilinearInterpJax(unittest.TestCase):
    """``bilinear_interp`` and ``bilinear_interp_derX`` vs ``BilinearInterp``."""

    def test_value(self):
        rng = np.random.default_rng(44)
        x_grid = np.sort(rng.uniform(0, 10, 12))
        y_grid = np.sort(rng.uniform(0, 5, 8))
        Xg, Yg = np.meshgrid(x_grid, y_grid, indexing="ij")
        f_vals = np.sin(Xg) * np.cos(Yg) + 0.1 * Xg
        x_query = rng.uniform(-1, 11, 80)
        y_query = rng.uniform(-0.5, 6, 80)
        hark = BilinearInterp(f_vals, x_grid, y_grid)(x_query, y_query)
        jaxv = ij.bilinear_interp(
            f_vals,
            x_grid,
            y_grid,
            jnp.asarray(x_query),
            jnp.asarray(y_query),
        )
        _assert_close(jaxv, hark, name="BilinearInterp value")

    def test_derivativeX(self):
        rng = np.random.default_rng(45)
        x_grid = np.sort(rng.uniform(0, 10, 10))
        y_grid = np.sort(rng.uniform(0, 5, 7))
        Xg, Yg = np.meshgrid(x_grid, y_grid, indexing="ij")
        f_vals = Xg**2 + Yg**2
        x_query = rng.uniform(1, 9, 50)
        y_query = rng.uniform(0.5, 4.5, 50)
        hark = BilinearInterp(f_vals, x_grid, y_grid).derivativeX(x_query, y_query)
        jaxv = ij.bilinear_interp_derX(
            f_vals,
            x_grid,
            y_grid,
            jnp.asarray(x_query),
            jnp.asarray(y_query),
        )
        _assert_close(jaxv, hark, name="BilinearInterp.derivativeX")


@unittest.skipUnless(HAS_JAX, "JAX not installed")
class TestLinearInterpOnInterp1DJax(unittest.TestCase):
    """``linear_interp_on_interp_1d_*`` vs ``LinearInterpOnInterp1D``."""

    def test_shared_xgrid(self):
        rng = np.random.default_rng(46)
        x_grid = np.sort(rng.uniform(0, 10, 15))
        y_grid = np.sort(rng.uniform(0, 5, 4))
        f_vals = np.zeros((len(y_grid), len(x_grid)))
        for k, y in enumerate(y_grid):
            f_vals[k] = np.sin(x_grid) + 0.3 * y * x_grid
        x_query = rng.uniform(x_grid.min(), x_grid.max(), 40)
        y_query = rng.uniform(y_grid.min(), y_grid.max(), 40)
        inner = [
            LinearInterp(x_grid, f_vals[k], lower_extrap=True)
            for k in range(len(y_grid))
        ]
        hark = LinearInterpOnInterp1D(inner, y_grid)(x_query, y_query)
        jaxv = ij.linear_interp_on_interp_1d_shared_xgrid(
            x_grid,
            y_grid,
            f_vals,
            jnp.asarray(x_query),
            jnp.asarray(y_query),
        )
        _assert_close(jaxv, hark, name="LinearInterpOnInterp1D shared x_grid")

    def test_per_y_xgrid(self):
        rng = np.random.default_rng(47)
        y_grid = np.sort(rng.uniform(0, 5, 4))
        Nx = 12
        x_grids = np.zeros((len(y_grid), Nx))
        f_vals = np.zeros((len(y_grid), Nx))
        for k, y in enumerate(y_grid):
            x_grids[k] = np.sort(rng.uniform(0, 10, Nx))
            f_vals[k] = np.sin(x_grids[k]) + 0.3 * y * x_grids[k]
        x_query = rng.uniform(2, 8, 30)
        y_query = rng.uniform(y_grid.min(), y_grid.max(), 30)
        inner = [
            LinearInterp(x_grids[k], f_vals[k], lower_extrap=True)
            for k in range(len(y_grid))
        ]
        hark = LinearInterpOnInterp1D(inner, y_grid)(x_query, y_query)
        jaxv = ij.linear_interp_on_interp_1d_general(
            x_grids,
            y_grid,
            f_vals,
            jnp.asarray(x_query),
            jnp.asarray(y_query),
        )
        _assert_close(jaxv, hark, name="LinearInterpOnInterp1D per-y x_grid")


@unittest.skipUnless(HAS_JAX, "JAX not installed")
class TestLowerEnvelope2DJax(unittest.TestCase):
    """``lower_envelope_2d_apply`` vs element-wise numpy minimum."""

    def test_element_wise_min(self):
        rng = np.random.default_rng(48)
        N = 50
        f1 = rng.uniform(0, 10, N)
        f2 = rng.uniform(0, 10, N)
        f3 = rng.uniform(0, 10, N)
        jaxv = ij.lower_envelope_2d_apply(
            jnp.asarray(f1), jnp.asarray(f2), jnp.asarray(f3)
        )
        expected = np.minimum(np.minimum(f1, f2), f3)
        _assert_close(jaxv, expected, name="LowerEnvelope2D min")


@unittest.skipUnless(HAS_JAX, "JAX not installed")
class TestCRRAHelpersJax(unittest.TestCase):
    """``marg_value_to_consumption`` round-trip via ``consumption_to_marg_value``."""

    def test_inverse_roundtrip(self):
        rng = np.random.default_rng(49)
        rho = 2.0
        c = rng.uniform(0.1, 5.0, 30)
        vP = c ** (-rho)
        c_back = ij.marg_value_to_consumption(jnp.asarray(vP), rho)
        _assert_close(c_back, c, name="CRRA inverse u'^-1")


if __name__ == "__main__":
    unittest.main()

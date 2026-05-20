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


# ============================================================
# Edge-case coverage (broader matrix follow-up)
# ============================================================


@unittest.skipUnless(HAS_JAX, "JAX not installed")
class TestLinearInterp1DEdgeCases(unittest.TestCase):
    """Edge cases for ``linear_interp_1d`` parity."""

    def test_two_point_grid(self):
        """Minimum-size grid (n=2): everything is the single segment."""
        x_grid = np.array([0.0, 1.0])
        y_vals = np.array([3.0, 5.0])
        x_query = np.array([-0.5, 0.0, 0.25, 0.5, 1.0, 1.5])
        hark = LinearInterp(x_grid, y_vals, lower_extrap=True)(x_query)
        jaxv = ij.linear_interp_1d(
            x_grid, y_vals, jnp.asarray(x_query), lower_extrap=True
        )
        _assert_close(jaxv, hark, name="LinearInterp n=2 grid")

    def test_constant_function(self):
        """All y_vals equal: result must be constant."""
        x_grid = np.linspace(0, 10, 30)
        y_vals = np.full(30, 7.42)
        x_query = np.linspace(-5, 15, 50)
        hark = LinearInterp(x_grid, y_vals, lower_extrap=True)(x_query)
        jaxv = ij.linear_interp_1d(
            x_grid, y_vals, jnp.asarray(x_query), lower_extrap=True
        )
        _assert_close(jaxv, hark, name="LinearInterp constant y")

    def test_single_query(self):
        """Scalar query (shape ()): result should also be scalar."""
        rng = np.random.default_rng(101)
        x_grid = np.sort(rng.uniform(0, 10, 20))
        y_vals = np.sin(x_grid)
        for x_q in (0.001, 5.0, 9.999):
            hark = float(LinearInterp(x_grid, y_vals, lower_extrap=True)(x_q))
            jaxv = float(
                ij.linear_interp_1d(x_grid, y_vals, jnp.asarray(x_q), lower_extrap=True)
            )
            self.assertAlmostEqual(jaxv, hark, places=8, msg=f"scalar query x={x_q}")

    def test_large_query_array(self):
        """1e5 query points — vectorization correctness, not timing."""
        rng = np.random.default_rng(102)
        x_grid = np.sort(rng.uniform(0, 100, 50))
        y_vals = np.cos(x_grid)
        x_query = rng.uniform(0, 100, 100_000)
        hark = LinearInterp(x_grid, y_vals, lower_extrap=True)(x_query)
        jaxv = ij.linear_interp_1d(
            x_grid, y_vals, jnp.asarray(x_query), lower_extrap=True
        )
        _assert_close(jaxv, hark, name="LinearInterp 100k queries")

    def test_strictly_decreasing_values_increasing_grid(self):
        """Monotone-decreasing y on monotone-increasing x grid."""
        x_grid = np.linspace(0.1, 10, 25)
        y_vals = 1.0 / x_grid  # strictly decreasing
        x_query = np.linspace(0.5, 9.5, 50)
        hark = LinearInterp(x_grid, y_vals, lower_extrap=True)(x_query)
        jaxv = ij.linear_interp_1d(
            x_grid, y_vals, jnp.asarray(x_query), lower_extrap=True
        )
        _assert_close(jaxv, hark, name="LinearInterp 1/x decreasing y")


@unittest.skipUnless(HAS_JAX, "JAX not installed")
class TestBilinearInterpEdgeCases(unittest.TestCase):
    """Edge cases for ``bilinear_interp`` parity."""

    def test_two_by_two_grid(self):
        """Minimum-size 2D grid: a single bilinear patch."""
        x_grid = np.array([0.0, 1.0])
        y_grid = np.array([0.0, 1.0])
        f_vals = np.array([[1.0, 2.0], [3.0, 5.0]])  # bilinear in x and y
        x_query = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        y_query = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        hark = BilinearInterp(f_vals, x_grid, y_grid)(x_query, y_query)
        jaxv = ij.bilinear_interp(
            f_vals,
            x_grid,
            y_grid,
            jnp.asarray(x_query),
            jnp.asarray(y_query),
        )
        _assert_close(jaxv, hark, name="BilinearInterp 2x2 grid")

    def test_constant_2d(self):
        """f_vals all equal: bilinear result must be the constant."""
        x_grid = np.linspace(0, 10, 8)
        y_grid = np.linspace(0, 5, 6)
        f_vals = np.full((8, 6), -3.14)
        rng = np.random.default_rng(103)
        x_query = rng.uniform(-1, 11, 25)
        y_query = rng.uniform(-1, 6, 25)
        hark = BilinearInterp(f_vals, x_grid, y_grid)(x_query, y_query)
        jaxv = ij.bilinear_interp(
            f_vals,
            x_grid,
            y_grid,
            jnp.asarray(x_query),
            jnp.asarray(y_query),
        )
        _assert_close(jaxv, hark, name="BilinearInterp constant 2D")

    def test_exact_grid_points(self):
        """Queries at exact grid corners must return the corner values."""
        rng = np.random.default_rng(104)
        x_grid = np.sort(rng.uniform(0, 10, 8))
        y_grid = np.sort(rng.uniform(0, 5, 6))
        f_vals = rng.uniform(-2, 2, (8, 6))
        Xg, Yg = np.meshgrid(x_grid, y_grid, indexing="ij")
        x_query = Xg.flatten()
        y_query = Yg.flatten()
        hark = BilinearInterp(f_vals, x_grid, y_grid)(x_query, y_query)
        jaxv = ij.bilinear_interp(
            f_vals,
            x_grid,
            y_grid,
            jnp.asarray(x_query),
            jnp.asarray(y_query),
        )
        _assert_close(jaxv, hark, name="BilinearInterp exact corners")


@unittest.skipUnless(HAS_JAX, "JAX not installed")
class TestCRRAHelpersEdgeCases(unittest.TestCase):
    """Edge cases for CRRA helpers."""

    def test_log_utility_rho_1(self):
        """CRRA=1 (log utility) round-trip."""
        rng = np.random.default_rng(105)
        rho = 1.0
        c = rng.uniform(0.1, 5.0, 30)
        vP = c ** (-rho)
        c_back = ij.marg_value_to_consumption(jnp.asarray(vP), rho)
        _assert_close(c_back, c, name="CRRA rho=1 (log)")

    def test_high_curvature_rho_6(self):
        """High curvature rho=6 round-trip — vP can be very large for small c."""
        rng = np.random.default_rng(106)
        rho = 6.0
        c = rng.uniform(0.5, 5.0, 30)  # avoid c→0 where vP→inf
        vP = c ** (-rho)
        c_back = ij.marg_value_to_consumption(jnp.asarray(vP), rho)
        _assert_close(c_back, c, name="CRRA rho=6 (high curvature)")


if __name__ == "__main__":
    unittest.main()

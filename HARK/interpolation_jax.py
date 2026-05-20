"""
JAX implementations of HARK interpolation primitives.

This is an opt-in alternative to the numpy/numba implementations in
``HARK.interpolation``. The functions here are designed for use in JAX-jitted
solvers (CPU or GPU). They are functional (data + query arguments, no classes)
so that they can be ``vmap``-ed, ``jit``-ed, and differentiated.

Every function here has a bit-precision counterpart in ``HARK.interpolation``;
see ``HARK/tests/test_interpolation_jax.py`` for the parity tests.

JAX is an optional dependency. Importing this module without JAX installed
raises ``ImportError`` with installation instructions.
"""

from __future__ import annotations

try:
    import jax
    import jax.numpy as jnp
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "HARK.interpolation_jax requires JAX. "
        "Install with: pip install jax  (or: pip install 'jax[cuda12]' for GPU)"
    ) from _e


# ============================================================
# 1-D linear interpolation — counterpart to HARK.interpolation.LinearInterp
# ============================================================


def linear_interp_1d(x_grid, y_vals, x_query, lower_extrap=False):
    """
    1-D linear interpolation, matching ``HARK.interpolation.LinearInterp``
    with ``decay_extrap=False`` (the default).

    Parameters
    ----------
    x_grid : array_like, shape (N,)
        Sorted strictly-increasing grid of x values.
    y_vals : array_like, shape (N,)
        Function values at the grid points.
    x_query : array_like
        Query x values (any shape).
    lower_extrap : bool, default False
        If False, queries below ``x_grid[0]`` return NaN (matches HARK default).
        If True, linear extrapolation is used at both ends.

    Returns
    -------
    array_like
        Interpolated y values, same shape as ``x_query``.

    Notes
    -----
    Mirrors HARK semantics exactly: ``searchsorted`` is run on ``x_grid[:-1]``,
    so above-top queries hit segment (N-2, N-1) and receive linear
    extrapolation rather than NaN regardless of ``lower_extrap``.
    """
    x_grid = jnp.asarray(x_grid)
    y_vals = jnp.asarray(y_vals)
    n = x_grid.shape[0]
    i = jnp.clip(jnp.searchsorted(x_grid[:-1], x_query, side="left"), 1, n - 1)
    alpha = (x_query - x_grid[i - 1]) / (x_grid[i] - x_grid[i - 1])
    y = (1 - alpha) * y_vals[i - 1] + alpha * y_vals[i]
    if not lower_extrap:
        y = jnp.where(x_query < x_grid[0], jnp.nan, y)
    return y


# ============================================================
# 2-D bilinear interpolation — counterpart to HARK.interpolation.BilinearInterp
# ============================================================


def bilinear_interp(f_values, x_grid, y_grid, x_query, y_query):
    """
    2-D bilinear interpolation, matching ``HARK.interpolation.BilinearInterp``.

    Parameters
    ----------
    f_values : array_like, shape (Nx, Ny)
        ``f_values[i, j] = f(x_grid[i], y_grid[j])``.
    x_grid : array_like, shape (Nx,)
    y_grid : array_like, shape (Ny,)
    x_query, y_query : array_like
        Query points, same shape.

    Returns
    -------
    array_like
        Interpolated values, same shape as ``x_query``.

    Notes
    -----
    HARK clips both directions to a valid segment, so queries outside the grid
    extrapolate using the boundary segment.
    """
    f_values = jnp.asarray(f_values)
    x_grid = jnp.asarray(x_grid)
    y_grid = jnp.asarray(y_grid)
    Nx = x_grid.shape[0]
    Ny = y_grid.shape[0]
    x_pos = jnp.clip(jnp.searchsorted(x_grid, x_query, side="left"), 1, Nx - 1)
    y_pos = jnp.clip(jnp.searchsorted(y_grid, y_query, side="left"), 1, Ny - 1)
    alpha = (x_query - x_grid[x_pos - 1]) / (x_grid[x_pos] - x_grid[x_pos - 1])
    beta = (y_query - y_grid[y_pos - 1]) / (y_grid[y_pos] - y_grid[y_pos - 1])
    f00 = f_values[x_pos - 1, y_pos - 1]
    f01 = f_values[x_pos - 1, y_pos]
    f10 = f_values[x_pos, y_pos - 1]
    f11 = f_values[x_pos, y_pos]
    return (
        (1 - alpha) * (1 - beta) * f00
        + (1 - alpha) * beta * f01
        + alpha * (1 - beta) * f10
        + alpha * beta * f11
    )


def bilinear_interp_derX(f_values, x_grid, y_grid, x_query, y_query):
    """
    Partial derivative ∂f/∂x of bilinear interp, matching
    ``HARK.interpolation.BilinearInterp.derivativeX``.

    Parameters mirror :func:`bilinear_interp`.
    """
    f_values = jnp.asarray(f_values)
    x_grid = jnp.asarray(x_grid)
    y_grid = jnp.asarray(y_grid)
    Nx = x_grid.shape[0]
    Ny = y_grid.shape[0]
    x_pos = jnp.clip(jnp.searchsorted(x_grid, x_query, side="left"), 1, Nx - 1)
    y_pos = jnp.clip(jnp.searchsorted(y_grid, y_query, side="left"), 1, Ny - 1)
    beta = (y_query - y_grid[y_pos - 1]) / (y_grid[y_pos] - y_grid[y_pos - 1])
    dfdx = (
        ((1 - beta) * f_values[x_pos, y_pos - 1] + beta * f_values[x_pos, y_pos])
        - (
            (1 - beta) * f_values[x_pos - 1, y_pos - 1]
            + beta * f_values[x_pos - 1, y_pos]
        )
    ) / (x_grid[x_pos] - x_grid[x_pos - 1])
    return dfdx


# ============================================================
# LinearInterpOnInterp1D — counterpart to HARK.interpolation.LinearInterpOnInterp1D
# ============================================================


def linear_interp_on_interp_1d_shared_xgrid(x_grid, y_grid, f_values, x_query, y_query):
    """
    LinearInterpOnInterp1D when all inner 1-D interpolators share ``x_grid``.

    Equivalent to bilinear interpolation with axes swapped (HARK's
    LinearInterpOnInterp1D stores ``f_values`` with the y-axis first).

    Parameters
    ----------
    x_grid : array_like, shape (Nx,)
    y_grid : array_like, shape (Ny,)
    f_values : array_like, shape (Ny, Nx)
        ``f_values[k, j] = inner_interpolator_k(x_grid[j])``.
    x_query, y_query : array_like
        Query points, same shape.
    """
    return bilinear_interp(f_values.T, x_grid, y_grid, x_query, y_query)


def linear_interp_on_interp_1d_general(x_grids, y_grid, f_values, x_query, y_query):
    """
    LinearInterpOnInterp1D when each inner 1-D interpolator has its own
    ``x_grid``.

    Parameters
    ----------
    x_grids : array_like, shape (Ny, Nx)
        Separate x-grid per y atom.
    y_grid : array_like, shape (Ny,)
        y values for each inner interpolator.
    f_values : array_like, shape (Ny, Nx)
        ``f_values[k, j] = inner_interpolator_k(x_grids[k, j])``.
    x_query, y_query : array_like
        Query points, same shape.

    Notes
    -----
    Inner 1-D interpolations use ``lower_extrap=True`` to match HARK's
    LinearInterpOnInterp1D, which calls the inner interpolators on raw queries.
    """
    x_grids = jnp.asarray(x_grids)
    y_grid = jnp.asarray(y_grid)
    f_values = jnp.asarray(f_values)
    Ny = y_grid.shape[0]
    y_pos = jnp.clip(jnp.searchsorted(y_grid, y_query, side="left"), 1, Ny - 1)
    alpha = (y_query - y_grid[y_pos - 1]) / (y_grid[y_pos] - y_grid[y_pos - 1])

    def per_query_lo(xq, idx):
        return linear_interp_1d(
            x_grids[idx - 1], f_values[idx - 1], xq, lower_extrap=True
        )

    def per_query_hi(xq, idx):
        return linear_interp_1d(x_grids[idx], f_values[idx], xq, lower_extrap=True)

    f_lo = jax.vmap(per_query_lo)(x_query, y_pos)
    f_hi = jax.vmap(per_query_hi)(x_query, y_pos)
    return (1 - alpha) * f_lo + alpha * f_hi


# ============================================================
# LowerEnvelope2D — counterpart to HARK.interpolation.LowerEnvelope2D
# ============================================================


def lower_envelope_2d_apply(*fvals):
    """
    Element-wise minimum across multiple function-value arrays, matching
    ``HARK.interpolation.LowerEnvelope2D`` semantics.

    Parameters
    ----------
    *fvals : tuple of array_like
        Function values at common query points.

    Returns
    -------
    array_like
        Element-wise minimum, same shape as inputs.
    """
    out = fvals[0]
    for f in fvals[1:]:
        out = jnp.minimum(out, f)
    return out


# ============================================================
# VariableLowerBoundFunc2D — counterpart to HARK.interpolation.VariableLowerBoundFunc2D
# ============================================================


def variable_lower_bound_eval(inner_fn, lb_values_at_y, x_query, y_query):
    """
    Evaluate ``inner_fn(x - lb(y), y)``.

    Matches ``HARK.interpolation.VariableLowerBoundFunc2D`` when the caller
    has pre-evaluated ``lb(y)`` at the query y values.

    Parameters
    ----------
    inner_fn : callable
        A function ``inner_fn(x, y) -> z`` that accepts JAX arrays.
    lb_values_at_y : array_like
        ``lb(y)`` evaluated at each ``y_query``.
    x_query, y_query : array_like
        Query points.
    """
    return inner_fn(x_query - lb_values_at_y, y_query)


# ============================================================
# CRRA marginal-utility helpers — counterpart to MargValueFuncCRRA
# ============================================================


def marg_value_to_consumption(vP, rho):
    """
    Invert CRRA marginal utility: return ``c`` such that ``u'(c) = vP``.

    For CRRA utility, ``u'(c) = c^(-rho)``, so ``c = vP^(-1/rho)``.
    """
    return vP ** (-1.0 / rho)


def consumption_to_marg_value(c, rho):
    """
    Forward CRRA marginal utility: ``u'(c) = c^(-rho)``.
    """
    return c ** (-rho)

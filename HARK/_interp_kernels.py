"""Interpolation kernels written once for numpy and JAX.

Each kernel takes its array library as its first argument, ``xp`` (``numpy`` or
``jax.numpy``), and uses only operations both provide: ``searchsorted``,
``maximum``, ``clip``, ``where``, ``exp``, fancy indexing and arithmetic. Every
formula keeps the order of operations of the numpy class it mirrors, so that with
``xp=numpy`` a kernel reproduces that class bit for bit. ``tests/test_interp_kernels.py``
checks this, and it is the gate for one day pointing the numpy classes at these
kernels. Today only the JAX backend (``HARK._interp_jax``) calls them.

JAX arrays cannot be written in place, so where the numpy class patches entries
(``y[below] = nan``) a kernel selects with ``xp.where``. Selection returns the same
floating-point values that the patch stores.
"""


def linear_1d(xp, x_list, y_list, x, want_y, want_dydx, lower_extrap, decay=None):
    """
    Level and/or slope of a linear interpolant, as ``LinearInterp._evalOrDer``.

    Parameters
    ----------
    xp : module
        The array library, ``numpy`` or ``jax.numpy``.
    x_list, y_list : array
        The interpolant's knots, as stored by ``LinearInterp``.
    x : array
        One-dimensional array of query points.
    want_y, want_dydx : bool
        Which outputs to compute; the other is returned as None.
    lower_extrap : bool
        Whether queries below ``x_list[0]`` are extrapolated (True) or NaN.
    decay : tuple or None
        ``(decay_extrap_A, decay_extrap_B, intercept_limit, slope_limit)`` when the
        interpolant decays to a limiting linear function above its top knot, as
        ``LinearInterp`` does when ``decay_extrap`` is True; None otherwise.

    Returns
    -------
    y, dydx : array or None
    """
    i = xp.maximum(xp.searchsorted(x_list[:-1], x), 1)
    x_lo = x_list[i - 1]
    x_hi = x_list[i]
    y_lo = y_list[i - 1]
    y_hi = y_list[i]
    if want_y:
        alpha = (x - x_lo) / (x_hi - x_lo)
        y = (1.0 - alpha) * y_lo + alpha * y_hi
    else:
        y = None
    dydx = (y_hi - y_lo) / (x_hi - x_lo) if want_dydx else None

    if not lower_extrap:
        below = x < x_list[0]
        if y is not None:
            y = xp.where(below, xp.nan, y)
        if dydx is not None:
            dydx = xp.where(below, xp.nan, dydx)

    if decay is not None:
        decay_A, decay_B, intercept_limit, slope_limit = decay
        above = x > x_list[-1]
        # Zero below the top knot, so the discarded entries cannot overflow exp
        x_temp = xp.where(above, x - x_list[-1], 0.0)
        decay_val = decay_A * xp.exp(-decay_B * x_temp)
        if y is not None:
            y = xp.where(above, intercept_limit + slope_limit * x - decay_val, y)
        if dydx is not None:
            dydx = xp.where(above, slope_limit + decay_B * decay_val, dydx)
    return y, dydx


def _locate_clipped(xp, grid, values):
    """``searchsorted`` clipped into ``[1, n - 1]``, as ``interpolation._locate_clipped``."""
    return xp.clip(xp.searchsorted(grid, values), 1, grid.shape[0] - 1)


def _cell_fraction(grid, idx, queries):
    """Position of ``queries`` within ``[grid[idx-1], grid[idx]]``, as
    ``interpolation._cell_fraction``."""
    lower = grid[idx - 1]
    return (queries - lower) / (grid[idx] - lower)


def bilinear_2d(xp, f_values, x_list, y_list, x, y, which):
    """
    Level or a partial derivative of a bilinear interpolant, as ``BilinearInterp``.

    Parameters
    ----------
    xp : module
        The array library, ``numpy`` or ``jax.numpy``.
    f_values, x_list, y_list : array
        The interpolant's values and grids, as stored by ``BilinearInterp``.
    x, y : array
        One-dimensional arrays of query points, of equal length.
    which : str
        ``"f"`` for the level, ``"dfdx"`` or ``"dfdy"`` for a partial derivative.

    Returns
    -------
    array
    """
    x_pos = _locate_clipped(xp, x_list, x)
    y_pos = _locate_clipped(xp, y_list, y)
    if which == "f":
        alpha = _cell_fraction(x_list, x_pos, x)
        beta = _cell_fraction(y_list, y_pos, y)
        return (
            (1 - alpha) * (1 - beta) * f_values[x_pos - 1, y_pos - 1]
            + (1 - alpha) * beta * f_values[x_pos - 1, y_pos]
            + alpha * (1 - beta) * f_values[x_pos, y_pos - 1]
            + alpha * beta * f_values[x_pos, y_pos]
        )
    if which == "dfdx":
        beta = _cell_fraction(y_list, y_pos, y)
        return (
            ((1 - beta) * f_values[x_pos, y_pos - 1] + beta * f_values[x_pos, y_pos])
            - (
                (1 - beta) * f_values[x_pos - 1, y_pos - 1]
                + beta * f_values[x_pos - 1, y_pos]
            )
        ) / (x_list[x_pos] - x_list[x_pos - 1])
    if which == "dfdy":
        alpha = _cell_fraction(x_list, x_pos, x)
        return (
            ((1 - alpha) * f_values[x_pos - 1, y_pos] + alpha * f_values[x_pos, y_pos])
            - (
                (1 - alpha) * f_values[x_pos - 1, y_pos - 1]
                + alpha * f_values[x_pos, y_pos - 1]
            )
        ) / (y_list[y_pos] - y_list[y_pos - 1])
    raise ValueError(f"which must be 'f', 'dfdx' or 'dfdy', not {which!r}.")

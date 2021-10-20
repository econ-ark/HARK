import numpy as np
from numba import njit, vectorize

from HARK.utilities import (
    CRRAutility,
    CRRAutilityP,
    CRRAutilityPP,
    CRRAutilityP_inv,
    CRRAutility_invP,
    CRRAutility_inv,
    CRRAutilityP_invP,
)

CRRAutility = vectorize(CRRAutility, cache=True)
CRRAutilityP = vectorize(CRRAutilityP, cache=True)
CRRAutilityPP = vectorize(CRRAutilityPP, cache=True)
CRRAutilityP_inv = vectorize(CRRAutilityP_inv, cache=True)
CRRAutility_invP = vectorize(CRRAutility_invP, cache=True)
CRRAutility_inv = vectorize(CRRAutility_inv, cache=True)
CRRAutilityP_invP = vectorize(CRRAutilityP_invP, cache=True)


@njit(cache=True, error_model="numpy")
def _interp_decay(x0, x_list, y_list, intercept_limit, slope_limit, lower_extrap):
    # Make a decay extrapolation
    slope_at_top = (y_list[-1] - y_list[-2]) / (x_list[-1] - x_list[-2])
    level_diff = intercept_limit + slope_limit * x_list[-1] - y_list[-1]
    slope_diff = slope_limit - slope_at_top

    decay_extrap_A = level_diff
    decay_extrap_B = -slope_diff / level_diff
    intercept_limit = intercept_limit
    slope_limit = slope_limit

    i = np.maximum(np.searchsorted(x_list[:-1], x0), 1)
    alpha = (x0 - x_list[i - 1]) / (x_list[i] - x_list[i - 1])
    y0 = (1.0 - alpha) * y_list[i - 1] + alpha * y_list[i]

    if not lower_extrap:
        below_lower_bound = x0 < x_list[0]
        y0[below_lower_bound] = np.nan

    above_upper_bound = x0 > x_list[-1]
    x_temp = x0[above_upper_bound] - x_list[-1]

    y0[above_upper_bound] = (
        intercept_limit
        + slope_limit * x0[above_upper_bound]
        - decay_extrap_A * np.exp(-decay_extrap_B * x_temp)
    )

    return y0


@njit(cache=True, error_model="numpy")
def _interp_linear(x0, x_list, y_list, lower_extrap):
    i = np.maximum(np.searchsorted(x_list[:-1], x0), 1)
    alpha = (x0 - x_list[i - 1]) / (x_list[i] - x_list[i - 1])
    y0 = (1.0 - alpha) * y_list[i - 1] + alpha * y_list[i]

    if not lower_extrap:
        below_lower_bound = x0 < x_list[0]
        y0[below_lower_bound] = np.nan

    return y0


@njit(cache=True, error_model="numpy")
def linear_interp_fast(
    x0, x_list, y_list, intercept_limit=None, slope_limit=None, lower_extrap=False
):
    if intercept_limit is None and slope_limit is None:
        return _interp_linear(x0, x_list, y_list, lower_extrap)
    else:
        return _interp_decay(
            x0, x_list, y_list, intercept_limit, slope_limit, lower_extrap
        )


@njit(cache=True, error_model="numpy")
def _interp_linear_deriv(x0, x_list, y_list, lower_extrap):
    i = np.maximum(np.searchsorted(x_list[:-1], x0), 1)
    alpha = (x0 - x_list[i - 1]) / (x_list[i] - x_list[i - 1])
    y0 = (1.0 - alpha) * y_list[i - 1] + alpha * y_list[i]
    dydx = (y_list[i] - y_list[i - 1]) / (x_list[i] - x_list[i - 1])

    if not lower_extrap:
        below_lower_bound = x0 < x_list[0]
        y0[below_lower_bound] = np.nan
        dydx[below_lower_bound] = np.nan

    return y0, dydx


@njit(cache=True, error_model="numpy")
def _interp_decay_deriv(x0, x_list, y_list, intercept_limit, slope_limit, lower_extrap):
    # Make a decay extrapolation
    slope_at_top = (y_list[-1] - y_list[-2]) / (x_list[-1] - x_list[-2])
    level_diff = intercept_limit + slope_limit * x_list[-1] - y_list[-1]
    slope_diff = slope_limit - slope_at_top

    decay_extrap_A = level_diff
    decay_extrap_B = -slope_diff / level_diff
    intercept_limit = intercept_limit
    slope_limit = slope_limit

    i = np.maximum(np.searchsorted(x_list[:-1], x0), 1)
    alpha = (x0 - x_list[i - 1]) / (x_list[i] - x_list[i - 1])
    y0 = (1.0 - alpha) * y_list[i - 1] + alpha * y_list[i]
    dydx = (y_list[i] - y_list[i - 1]) / (x_list[i] - x_list[i - 1])

    if not lower_extrap:
        below_lower_bound = x0 < x_list[0]
        y0[below_lower_bound] = np.nan
        dydx[below_lower_bound] = np.nan

    above_upper_bound = x0 > x_list[-1]
    x_temp = x0[above_upper_bound] - x_list[-1]

    y0[above_upper_bound] = (
        intercept_limit
        + slope_limit * x0[above_upper_bound]
        - decay_extrap_A * np.exp(-decay_extrap_B * x_temp)
    )

    dydx[above_upper_bound] = slope_limit + decay_extrap_B * decay_extrap_A * np.exp(
        -decay_extrap_B * x_temp
    )

    return y0, dydx


@njit(cache=True, error_model="numpy")
def linear_interp_deriv_fast(
    x0, x_list, y_list, intercept_limit=None, slope_limit=None, lower_extrap=False
):
    if intercept_limit is None and slope_limit is None:
        return _interp_linear_deriv(x0, x_list, y_list, lower_extrap)
    else:
        return _interp_decay_deriv(
            x0, x_list, y_list, intercept_limit, slope_limit, lower_extrap
        )


@njit(cache=True, error_model="numpy")
def _spline_decay(
    x_init, x_list, y_list, dydx_list, intercept_limit, slope_limit, lower_extrap
):
    n = x_list.size

    coeffs = np.empty((n + 1, 4))

    # Define lower extrapolation as linear function (or just NaN)
    if lower_extrap:
        coeffs[0] = np.array([y_list[0], dydx_list[0], 0, 0])
    else:
        coeffs[0] = np.array([np.nan, np.nan, np.nan, np.nan])

    # Calculate interpolation coefficients on segments mapped to [0,1]
    xdiff = np.diff(x_list)
    ydiff = np.diff(y_list)
    dydx0 = dydx_list[:-1] * xdiff
    dydx1 = dydx_list[1:] * xdiff
    coeffs[1:-1, 0] = y_list[:-1]
    coeffs[1:-1, 1] = dydx0
    coeffs[1:-1, 2] = 3 * ydiff - 2 * dydx0 - dydx1
    coeffs[1:-1, 3] = -2 * ydiff + dydx0 + dydx1

    # Calculate extrapolation coefficients as a decay toward limiting function y = mx+b
    gap = slope_limit * x_list[n - 1] + intercept_limit - y_list[n - 1]
    slope = slope_limit - dydx_list[n - 1]
    if (gap != 0) and (slope <= 0):
        coeffs[-1] = np.array([intercept_limit, slope_limit, gap, slope / gap])
    elif slope > 0:
        # fixing a problem when slope is positive
        coeffs[-1] = np.array([intercept_limit, slope_limit, 0, 0])
    else:
        coeffs[-1] = np.array([intercept_limit, slope_limit, gap, 0])

    m = len(x_init)
    pos = np.searchsorted(x_list, x_init)
    y = np.zeros(m)
    if y.size > 0:
        out_bot = pos == 0
        out_top = pos == n
        in_bnds = np.logical_not(np.logical_or(out_bot, out_top))

        # Do the "in bounds" evaluation points
        i = pos[in_bnds]
        coeffs_in = coeffs[i, :]
        alpha = (x_init[in_bnds] - x_list[i - 1]) / (x_list[i] - x_list[i - 1])
        y[in_bnds] = coeffs_in[:, 0] + alpha * (
            coeffs_in[:, 1] + alpha * (coeffs_in[:, 2] + alpha * coeffs_in[:, 3])
        )

        # Do the "out of bounds" evaluation points
        y[out_bot] = coeffs[0, 0] + coeffs[0, 1] * (x_init[out_bot] - x_list[0])
        alpha = x_init[out_top] - x_list[n - 1]
        y[out_top] = (
            coeffs[n, 0]
            + x_init[out_top] * coeffs[n, 1]
            - coeffs[n, 2] * np.exp(alpha * coeffs[n, 3])
        )

    dydx = np.zeros(m)
    if dydx.size > 0:
        out_bot = pos == 0
        out_top = pos == n
        in_bnds = np.logical_not(np.logical_or(out_bot, out_top))

        # Do the "in bounds" evaluation points
        i = pos[in_bnds]
        coeffs_in = coeffs[i, :]
        alpha = (x_init[in_bnds] - x_list[i - 1]) / (x_list[i] - x_list[i - 1])
        dydx[in_bnds] = (
            coeffs_in[:, 1]
            + alpha * (2 * coeffs_in[:, 2] + alpha * 3 * coeffs_in[:, 3])
        ) / (x_list[i] - x_list[i - 1])

        # Do the "out of bounds" evaluation points
        dydx[out_bot] = coeffs[0, 1]
        alpha = x_init[out_top] - x_list[n - 1]
        dydx[out_top] = coeffs[n, 1] - coeffs[n, 2] * coeffs[n, 3] * np.exp(
            alpha * coeffs[n, 3]
        )

    return y, dydx


@njit(cache=True, error_model="numpy")
def cubic_interp_fast(
    x0,
    x_list,
    y_list,
    dydx_list,
    intercept_limit=None,
    slope_limit=None,
    lower_extrap=False,
):
    if intercept_limit is None and slope_limit is None:
        slope = dydx_list[-1]
        intercept = y_list[-1] - slope * x_list[-1]

        return _spline_decay(
            x0, x_list, y_list, dydx_list, intercept, slope, lower_extrap
        )
    else:
        return _spline_decay(
            x0, x_list, y_list, dydx_list, intercept_limit, slope_limit, lower_extrap
        )

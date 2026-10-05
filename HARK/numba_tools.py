import numpy as np
from HARK._numba import njit

from HARK.interpolation import (
    _cubic_segment_index,
    _cubic_lower_eval,
    _cubic_upper_eval,
    _cubic_upper_row,
)
from HARK.rewards import (
    CRRAutility_X,
    CRRAutility_inv,
    CRRAutility_invP,
    CRRAutilityP_X,
    CRRAutilityP_inv,
    CRRAutilityP_invP,
    CRRAutilityPP_X,
)

CRRAutility = njit(CRRAutility_X, cache=True)
CRRAutilityP = njit(CRRAutilityP_X, cache=True)
CRRAutilityPP = njit(CRRAutilityPP_X, cache=True)
CRRAutilityP_inv = njit(CRRAutilityP_inv, cache=True)
CRRAutility_invP = njit(CRRAutility_invP, cache=True)
CRRAutility_inv = njit(CRRAutility_inv, cache=True)
CRRAutilityP_invP = njit(CRRAutilityP_invP, cache=True)

# The cubic extrapolation and segment rules of HARK.interpolation.CubicInterp
cubic_segment_index = njit(_cubic_segment_index, cache=True)
cubic_upper_row = njit(_cubic_upper_row, cache=True)
cubic_lower_eval = njit(_cubic_lower_eval, cache=True)
cubic_upper_eval = njit(_cubic_upper_eval, cache=True)


@njit(cache=True, error_model="numpy")
def _decay_extrap_coeffs(
    x_list, y_list, intercept_limit, slope_limit
):  # pragma: no cover
    # Coefficients for decay extrapolation toward the line y = slope*x + intercept.
    slope_at_top = (y_list[-1] - y_list[-2]) / (x_list[-1] - x_list[-2])
    level_diff = intercept_limit + slope_limit * x_list[-1] - y_list[-1]
    slope_diff = slope_limit - slope_at_top
    decay_extrap_A = level_diff
    decay_extrap_B = -slope_diff / level_diff
    return decay_extrap_A, decay_extrap_B


@njit(cache=True, error_model="numpy")
def _interp_linear(x0, x_list, y_list, lower_extrap):  # pragma: no cover
    i = np.maximum(np.searchsorted(x_list[:-1], x0), 1)
    alpha = (x0 - x_list[i - 1]) / (x_list[i] - x_list[i - 1])
    y0 = (1.0 - alpha) * y_list[i - 1] + alpha * y_list[i]

    if not lower_extrap:
        below_lower_bound = x0 < x_list[0]
        y0[below_lower_bound] = np.nan

    return y0


@njit(cache=True, error_model="numpy")
def _apply_y_decay_extrap(
    x0, y0, x_list, intercept_limit, slope_limit, decay_extrap_A, decay_extrap_B
):  # pragma: no cover
    above_upper_bound = x0 > x_list[-1]
    x_temp = x0[above_upper_bound] - x_list[-1]
    y0[above_upper_bound] = (
        intercept_limit
        + slope_limit * x0[above_upper_bound]
        - decay_extrap_A * np.exp(-decay_extrap_B * x_temp)
    )
    return above_upper_bound, x_temp


@njit(cache=True, error_model="numpy")
def _interp_decay(
    x0, x_list, y_list, intercept_limit, slope_limit, lower_extrap
):  # pragma: no cover
    decay_extrap_A, decay_extrap_B = _decay_extrap_coeffs(
        x_list, y_list, intercept_limit, slope_limit
    )
    y0 = _interp_linear(x0, x_list, y_list, lower_extrap)
    _apply_y_decay_extrap(
        x0, y0, x_list, intercept_limit, slope_limit, decay_extrap_A, decay_extrap_B
    )
    return y0


@njit(cache=True, error_model="numpy")
def linear_interp_fast(
    x0, x_list, y_list, intercept_limit=None, slope_limit=None, lower_extrap=False
):  # pragma: no cover
    if intercept_limit is None and slope_limit is None:
        return _interp_linear(x0, x_list, y_list, lower_extrap)
    else:
        return _interp_decay(
            x0, x_list, y_list, intercept_limit, slope_limit, lower_extrap
        )


@njit(cache=True, error_model="numpy")
def _interp_linear_deriv(x0, x_list, y_list, lower_extrap):  # pragma: no cover
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
def _interp_decay_deriv(
    x0, x_list, y_list, intercept_limit, slope_limit, lower_extrap
):  # pragma: no cover
    decay_extrap_A, decay_extrap_B = _decay_extrap_coeffs(
        x_list, y_list, intercept_limit, slope_limit
    )
    y0, dydx = _interp_linear_deriv(x0, x_list, y_list, lower_extrap)
    above_upper_bound, x_temp = _apply_y_decay_extrap(
        x0, y0, x_list, intercept_limit, slope_limit, decay_extrap_A, decay_extrap_B
    )
    dydx[above_upper_bound] = slope_limit + decay_extrap_B * decay_extrap_A * np.exp(
        -decay_extrap_B * x_temp
    )
    return y0, dydx


@njit(cache=True, error_model="numpy")
def linear_interp_deriv_fast(
    x0, x_list, y_list, intercept_limit=None, slope_limit=None, lower_extrap=False
):  # pragma: no cover
    if intercept_limit is None and slope_limit is None:
        return _interp_linear_deriv(x0, x_list, y_list, lower_extrap)
    else:
        return _interp_decay_deriv(
            x0, x_list, y_list, intercept_limit, slope_limit, lower_extrap
        )


@njit(cache=True, error_model="numpy")
def _spline_decay(
    x_init, x_list, y_list, dydx_list, intercept_limit, slope_limit, lower_extrap
):  # pragma: no cover
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

    # Element by element: a tuple of mixed numeric types cannot fill a float64 row
    b_lim, m_lim, gap, decay = cubic_upper_row(
        x_list[n - 1], y_list[n - 1], dydx_list[n - 1], intercept_limit, slope_limit
    )
    coeffs[-1, 0] = b_lim
    coeffs[-1, 1] = m_lim
    coeffs[-1, 2] = gap
    coeffs[-1, 3] = decay

    m = len(x_init)
    pos = cubic_segment_index(x_list, x_init)
    y = np.zeros(m)
    dydx = np.zeros(m)

    if m > 0:
        out_bot = pos == 0
        out_top = pos == n
        in_bnds = np.logical_not(np.logical_or(out_bot, out_top))

        # In-bounds evaluation
        i = pos[in_bnds]
        coeffs_in = coeffs[i, :]
        alpha_in = (x_init[in_bnds] - x_list[i - 1]) / (x_list[i] - x_list[i - 1])
        y[in_bnds] = coeffs_in[:, 0] + alpha_in * (
            coeffs_in[:, 1] + alpha_in * (coeffs_in[:, 2] + alpha_in * coeffs_in[:, 3])
        )
        dydx[in_bnds] = (
            coeffs_in[:, 1]
            + alpha_in * (2 * coeffs_in[:, 2] + alpha_in * 3 * coeffs_in[:, 3])
        ) / (x_list[i] - x_list[i - 1])

        # Out-of-bounds: bottom
        y_bot, dydx_bot = cubic_lower_eval(x_init[out_bot], x_list[0], coeffs[0])
        y[out_bot] = y_bot
        dydx[out_bot] = dydx_bot

        # Out-of-bounds: top
        y_top, dydx_top = cubic_upper_eval(x_init[out_top], x_list[n - 1], coeffs[n])
        y[out_top] = y_top
        dydx[out_top] = dydx_top

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
):  # pragma: no cover
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

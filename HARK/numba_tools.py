import numpy as np
from HARK._numba import njit

from HARK.interpolation import (
    _cubic_end_rows,
    _cubic_locate,
    _cubic_lower_eval,
    _cubic_poly_slope,
    _cubic_poly_value,
    _cubic_segment_coeffs,
    _cubic_top_tangent,
    _cubic_upper_eval,
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

# The cubic interpolation rules of HARK.interpolation.CubicInterp
cubic_end_rows = njit(_cubic_end_rows, cache=True)
cubic_locate = njit(_cubic_locate, cache=True)
cubic_lower_eval = njit(_cubic_lower_eval, cache=True)
cubic_poly_slope = njit(_cubic_poly_slope, cache=True)
cubic_poly_value = njit(_cubic_poly_value, cache=True)
cubic_segment_coeffs = njit(_cubic_segment_coeffs, cache=True)
cubic_top_tangent = njit(_cubic_top_tangent, cache=True)
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
    lower_row, upper_row = cubic_end_rows(
        x_list, y_list, dydx_list, intercept_limit, slope_limit, lower_extrap
    )
    coeffs = np.empty((x_list.size + 1, 4))
    coeffs[0] = lower_row
    coeffs[1:-1] = cubic_segment_coeffs(x_list, y_list, dydx_list)
    coeffs[-1] = upper_row

    out_bot, out_top, in_bnds, i, alpha, span = cubic_locate(x_list, x_init)
    coeffs_in = coeffs[i]
    y = np.zeros(x_init.size)
    dydx = np.zeros(x_init.size)
    y[in_bnds] = cubic_poly_value(coeffs_in, alpha)
    dydx[in_bnds] = cubic_poly_slope(coeffs_in, alpha, span)

    y_bot, dydx_bot = cubic_lower_eval(x_init[out_bot], x_list[0], coeffs[0])
    y[out_bot] = y_bot
    dydx[out_bot] = dydx_bot
    y_top, dydx_top = cubic_upper_eval(x_init[out_top], x_list[-1], coeffs[-1])
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
        intercept, slope = cubic_top_tangent(x_list, y_list, dydx_list)
        return _spline_decay(
            x0, x_list, y_list, dydx_list, intercept, slope, lower_extrap
        )
    else:
        return _spline_decay(
            x0, x_list, y_list, dydx_list, intercept_limit, slope_limit, lower_extrap
        )

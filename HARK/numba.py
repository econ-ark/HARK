import numpy as np
from numba import njit, vectorize, int32, float64, boolean, types, optional
from numba.experimental import jitclass

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

linear_specs = {
    "x_list": float64[::1],
    "y_list": float64[::1],
    "intercept_limit": optional(float64),
    "slope_limit": optional(float64),
    "lower_extrap": boolean,
    "distance_criteria": types.ListType(types.string),
    "x_n": int32,
    "decay_extrap": boolean,
    "decay_extrap_A": float64,
    "decay_extrap_B": float64,
}


@jitclass(linear_specs)
class LinearInterpFast(object):
    """
    A "from scratch" 1D linear interpolation class.  Allows for linear or decay
    extrapolation (approaching a limiting linear function from below).

    NOTE: When no input is given for the limiting linear function, linear
    extrapolation is used above the highest gridpoint.

    Parameters
    ----------
    x_list : np.array
        List of x values composing the grid.
    y_list : np.array
        List of y values, representing f(x) at the points in x_list.
    intercept_limit : float
        Intercept of limiting linear function.
    slope_limit : float
        Slope of limiting linear function.
    lower_extrap : boolean
        Indicator for whether lower extrapolation is allowed.  False means
        f(x) = NaN for x < min(x_list); True means linear extrapolation.
    """

    # Class members are not yet supported in jitclass: distance_criteria is moved to object attribute

    def __init__(
        self, x_list, y_list, intercept_limit=None, slope_limit=None, lower_extrap=False
    ):
        # self.distance_criteria = typed.List(["x_list", "y_list"])
        self.x_list = x_list  # checks are not possible, must already be flat array
        self.y_list = y_list  # checks are not possible, must already be flat array

        self.lower_extrap = lower_extrap
        self.x_n = self.x_list.size

        # Make a decay extrapolation
        if intercept_limit is not None and slope_limit is not None:
            slope_at_top = (y_list[-1] - y_list[-2]) / (x_list[-1] - x_list[-2])
            level_diff = intercept_limit + slope_limit * x_list[-1] - y_list[-1]
            slope_diff = slope_limit - slope_at_top
            # If the model that can handle uncertainty has been calibrated with
            # with uncertainty set to zero, the 'extrapolation' will blow up
            # Guard against that and nearby problems by testing slope equality
            if not np.isclose(slope_limit, slope_at_top, atol=1e-15):
                self.decay_extrap_A = level_diff
                self.decay_extrap_B = -slope_diff / level_diff
                self.intercept_limit = intercept_limit
                self.slope_limit = slope_limit
                self.decay_extrap = True
            else:
                self.decay_extrap = False
        else:
            self.decay_extrap = False

    def _get_index(self, x):
        return np.maximum(np.searchsorted(self.x_list[:-1], x), 1)

    def _evaluate(self, x, i):
        """
        Returns the level of the interpolated function at each value in x.  Only
        called internally by HARKinterpolator1D.__call__ (etc).
        """

        alpha = (x - self.x_list[i - 1]) / (self.x_list[i] - self.x_list[i - 1])
        y = (1.0 - alpha) * self.y_list[i - 1] + alpha * self.y_list[i]

        if not self.lower_extrap:
            below_lower_bound = x < self.x_list[0]
            y[below_lower_bound] = np.nan

        if self.decay_extrap:
            above_upper_bound = x > self.x_list[-1]
            x_temp = x[above_upper_bound] - self.x_list[-1]
            y[above_upper_bound] = (
                self.intercept_limit
                + self.slope_limit * x[above_upper_bound]
                - self.decay_extrap_A * np.exp(-self.decay_extrap_B * x_temp)
            )

        return y

    def _der(self, x, i):
        """
        Returns the first derivative of the interpolated function at each value
        in x. Only called internally by HARKinterpolator1D.derivative (etc).
        """

        dydx = (self.y_list[i] - self.y_list[i - 1]) / (
            self.x_list[i] - self.x_list[i - 1]
        )

        if not self.lower_extrap:
            below_lower_bound = x < self.x_list[0]
            dydx[below_lower_bound] = np.nan

        if self.decay_extrap:
            above_upper_bound = x > self.x_list[-1]
            x_temp = x[above_upper_bound] - self.x_list[-1]

            dydx[
                above_upper_bound
            ] = self.slope_limit + self.decay_extrap_B * self.decay_extrap_A * np.exp(
                -self.decay_extrap_B * x_temp
            )

        return dydx

    def eval(self, x):
        i = self._get_index(x)
        y = self._evaluate(x, i)

        return y

    def derivative(self, x):
        i = self._get_index(x)
        dydx = self._der(x, i)

        return dydx

    def eval_with_derivative(self, x):
        i = self._get_index(x)
        y = self._evaluate(x, i)
        dydx = self._der(x, i)

        return y, dydx


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

import numpy as np
from numba import njit, types, vectorize

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


@njit(cache=True)
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


@njit(cache=True)
def _interp_linear(x0, x_list, y_list, lower_extrap):
    i = np.maximum(np.searchsorted(x_list[:-1], x0), 1)
    alpha = (x0 - x_list[i - 1]) / (x_list[i] - x_list[i - 1])
    y0 = (1.0 - alpha) * y_list[i - 1] + alpha * y_list[i]

    if not lower_extrap:
        below_lower_bound = x0 < x_list[0]
        y0[below_lower_bound] = np.nan

    return y0


@njit(cache=True)
def LinearInterpFast(
    x0, x_list, y_list, intercept_limit=None, slope_limit=None, lower_extrap=False
):
    if intercept_limit is None and slope_limit is None:
        return _interp_linear(x0, x_list, y_list, lower_extrap)
    else:
        return _interp_decay(
            x0, x_list, y_list, intercept_limit, slope_limit, lower_extrap
        )

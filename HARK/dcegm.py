"""
Functions for working with the discrete-continuous EGM (DCEGM) algorithm as
described in "The endogenous grid method for discrete-continuous dynamic
choice models with (or without) taste shocks" by Iskhakov et al. (2016)
[https://doi.org/10.3982/QE643 and ijrsDCEGM2017 in our Zotero]

Example can be found in https://github.com/econ-ark/DemARK/blob/master/notebooks/DCEGM-Upper-Envelope.ipynb
"""
import numpy as np
from HARK.interpolation import LinearInterp
from interpolation import interp
from numba import jit, njit, typeof


@njit("Tuple((float64,float64))(float64[:], float64[:], float64[:])", cache=True)
def calc_linear_crossing(m, left_v, right_v):
    """
    Computes the intersection between two line segments, defined by two common
    x points, and the values of both segments at both x points. The intercept
    is only found if it happens between the two x coordinates

    Parameters
    ----------
    m : list or np.array, length 2
        The two common x coordinates. m[0] < m[1] is assumed 
    left_v :list or np.array, length 2
        y values of the two segments at m[0]
    right_v : list or np.array, length 2
        y values of the two segments at m[1]

    Returns
    -------
    (m_int, v_int):  a tuple with the corrdinates of the intercept.
    if there is no intercept in the interval [m[0],m[1]], (nan,nan)

    """

    # Find slopes of both segments
    delta_m = m[1] - m[0]
    s0 = (right_v[0] - left_v[0]) / delta_m
    s1 = (right_v[1] - left_v[1]) / delta_m

    if s1 == s0:
        # If they have the same slope, they can only cross if they perfectly
        # overlap. In this case, return the left extreme
        if left_v[0] == left_v[1]:
            return (m[0], left_v[0])
        else:
            return (np.nan, np.nan)
    else:
        # Find h where intercept happens at m[0] + h
        h = (left_v[0] - left_v[1]) / (s1 - s0)

        # Return the crossing if it happens between the given x coordinates.
        # If not, return nan
        if h >= 0 and h <= (m[1] - m[0]):
            return (m[0] + h, left_v[0] + h * s0)
        else:
            return (np.nan, np.nan)


@njit(
    "Tuple((float64[:,:],int64[:,:]))(float64[:], float64[:,:], int64[:])", cache=True
)
def calc_cross_points(mGrid, condVs, optIdx):
    """
    Given a grid of m values, a matrix of the conditional values of different
    actions at every grid point, and a vector indicating the optimal action
    at each grid point, this function computes the coordinates of the
    crossing points that happen when the optimal action changes

    Parameters
    ----------
    mGrid : np.array
        Market resources grid.
    condVs : np.array must have as many rows as possible discrete actions, and
             as many columns as m gridpoints there are.
        Conditional value functions

    optIdx : np.array of indices
        Optimal decision at each grid point
    Returns
    -------
    xing_points: 2D np.array
        Crossing points, each in its own row as an [m, v] pair.

    segments: np.array with two columns and as many rows as xing points.
        Each row represents a crossing point. The first column is the index
        of the optimal action to the left, and the second, to the right.
    """
    # Compute differences in the optimal index,
    # to find positions of choice-changes
    diff_max = np.append(optIdx[1:] - optIdx[:-1], 0)
    idx_change = np.where(diff_max != 0)[0]

    # If no crossings, return an empty list
    if len(idx_change) == 0:
        points = np.zeros((0, 2), dtype=np.float64)
        segments = np.zeros((0, 2), dtype=np.int64)
        return points, segments
    else:

        # To find the crossing points we need the extremes of the intervals in
        # which they happen, and the two candidate segments evaluated in both
        # extremes. switchMs[0] has the left points and switchMs[1] the right
        # points of these intervals.
        switchMs = np.stack((mGrid[idx_change], mGrid[idx_change + 1]), axis=1)

        # Store the indices of the two segments involved in the changes, by
        # looking at the argmax in the switching possitions.
        # Columns are [0]: left extreme, [1]: right extreme,
        # Rows are individual crossing points.
        segments = np.stack((optIdx[idx_change], optIdx[idx_change + 1]), axis=1)

        # Get values of segments on both the left and the right
        left_v = np.zeros_like(segments, dtype=np.float64)
        right_v = np.zeros_like(segments, dtype=np.float64)
        for i, idx in enumerate(idx_change):

            left_v[i, 0] = condVs[idx, segments[i, 0]]
            left_v[i, 1] = condVs[idx, segments[i, 1]]

            right_v[i, 0] = condVs[idx + 1, segments[i, 0]]
            right_v[i, 1] = condVs[idx + 1, segments[i, 1]]

        # A valid crossing must have both switching segments well defined at the
        # encompassing gridpoints. Filter those that do not.
        valid = np.repeat(False, len(idx_change))
        for i in range(len(valid)):
            valid[i] = np.logical_and(
                ~np.isnan(left_v[i, :]).any(), ~np.isnan(right_v[i, :]).any()
            )

        if not np.any(valid):

            points = np.zeros((0, 2), dtype=np.float64)
            segments = np.zeros((0, 2), dtype=np.int64)
            return points, segments

        else:

            segments = segments[valid, :]
            switchMs = switchMs[valid, :]
            left_v = left_v[valid, :]
            right_v = right_v[valid, :]

            # Find crossing points. Returns a list (m,v) tuples. Keep m's only
            xing_points = [
                calc_linear_crossing(switchMs[i, :], left_v[i, :], right_v[i, :])
                for i in range(segments.shape[0])
            ]

            xing_array = np.asarray(xing_points)

            return xing_array, segments


def calc_prim_kink(mGrid, vTGrids, choices):
    """
    Parameters
    ----------
    mGrid : np.array
        Common m grid
    vTGrids : [np.array], length  = # choices, each element has length = len(mGrid)
        value functions evaluated on the common m grid.
    choices : [np.array], length  = # choices, each element has length = len(mGrid)
        Optimal choices. In the form of choice probability vectors that must
        be degenerate

    Returns
    -------
    kinks: [(mCoord, vTCoor)]
        list of kink points
    segments: [(left, right)]
        List of the same length as kinks, where each element is a tuple
        indicating which segments are optimal on each side of the kink.
    """

    # Construct a vector with the optimal choice at each m point
    optChoice = np.zeros_like(mGrid, dtype=np.int64)
    for i in range(len(vTGrids)):
        idx = choices[i] == 1
        optChoice[idx] = i

    return calc_cross_points(mGrid, vTGrids.T, optChoice)


def calc_nondecreasing_segments(x, v):
    """
    """

    starts = [0]
    ends = []
    for i in range(1, len(x)):

        # Check if grid decreases in x or v
        x_dec = x[i] < x[i - 1]
        v_dec = v[i] < v[i - 1]

        if x_dec or v_dec:

            ends.append(i - 1)
            starts.append(i)

        i = i + 1

    # The last segment always ends in the last point
    ends.append(len(v) - 1)

    return np.array(starts), np.array(ends)


def upper_envelope(segments, calc_crossings=True):

    n_seg = len(segments)

    # Collect the x points of all segments in an ordered array
    x = np.sort(np.concatenate([x[0] for x in segments]))

    # Interpolate on all points using all segments
    y_cond = np.zeros((n_seg, len(x)))
    for i in range(n_seg):

        if len(segments[i][0]) == 1:
            # If the segment is a single point, we can only know its value
            # at the observed point.
            row = np.repeat(np.nan, len(x))
            ind = np.searchsorted(x, segments[i][0][0])
            row[ind] = segments[i][1][0]
        else:
            # If the segment has more than one point, we can interpolate
            row = interp(segments[i][0], segments[i][1], x)
            extrap = np.logical_or(x < segments[i][0][0], x > segments[i][0][-1])
            row[extrap] = np.nan

        y_cond[i, :] = row

    # Take the maximum to get the upper envelope
    env_inds = np.nanargmax(y_cond, 0)
    y = y_cond[env_inds, range(len(x))]

    # Get crossing points if needed
    if calc_crossings:

        xing_points, xing_lines = calc_cross_points(x, y_cond.T, env_inds)

        if len(xing_points) > 0:

            # Extract x and y coordinates
            xing_x = np.array([p[0] for p in xing_points])
            xing_y = np.array([p[1] for p in xing_points])

            # To capture the discontinuity, we'll add the successors of xing_x to
            # the grid
            succ = np.nextafter(xing_x, xing_x + 1)

            # Collect points to add to grids
            xtra_x = np.concatenate([xing_x, succ])
            # if there is a crossing, y will be the same on both segments
            xtra_y = np.concatenate([xing_y, xing_y])
            xtra_lines = np.concatenate([xing_lines[:, 0], xing_lines[:, 1]])

            # Insert them
            idx = np.searchsorted(x, xtra_x)
            x = np.insert(x, idx, xtra_x)
            y = np.insert(y, idx, xtra_y)
            env_inds = np.insert(env_inds, idx, xtra_lines)

    return x, y, env_inds

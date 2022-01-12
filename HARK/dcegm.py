"""
Functions for working with the discrete-continuous EGM (DCEGM) algorithm as
described in "The endogenous grid method for discrete-continuous dynamic
choice models with (or without) taste shocks" by Iskhakov et al. (2016)
[https://doi.org/10.3982/QE643 and ijrsDCEGM2017 in our Zotero]

Example can be found in https://github.com/econ-ark/DemARK/blob/master/notebooks/DCEGM-Upper-Envelope.ipynb
"""
import numpy as np
from interpolation import interp
from numba import njit


@njit("Tuple((float64,float64))(float64[:], float64[:], float64[:])", cache=True)
def calc_linear_crossing(x, left_y, right_y):
    """
    Computes the intersection between two line segments, defined by two common
    x points, and the values of both segments at both x points. The intercept
    is only found if it happens between the two x coordinates.

    Parameters
    ----------
    x : np.array, length 2
        The two common x coordinates. x[0] < x[1] is assumed 
    left_y : np.array, length 2
        y values of the two segments at x[0]
    right_y : np.array, length 2
        y values of the two segments at x[1]

    Returns
    -------
    (m_int, v_int):  a tuple with the corrdinates of the intercept.
    if there is no intercept in the interval [x[0],x[1]], (nan,nan)

    """

    # Find slopes of both segments
    delta_x = x[1] - x[0]
    s0 = (right_y[0] - left_y[0]) / delta_x
    s1 = (right_y[1] - left_y[1]) / delta_x

    if s1 == s0:
        # If they have the same slope, they can only cross if they perfectly
        # overlap. In this case, return the left extreme
        if left_y[0] == left_y[1]:
            return (x[0], left_y[0])
        else:
            return (np.nan, np.nan)
    else:
        # Find h where intercept happens at m[0] + h
        h = (left_y[0] - left_y[1]) / (s1 - s0)

        # Return the crossing if it happens between the given x coordinates.
        # If not, return nan
        if h >= 0 and h <= (x[1] - x[0]):
            return (x[0] + h, left_y[0] + h * s0)
        else:
            return (np.nan, np.nan)


@njit(
    "Tuple((float64[:,:],int64[:,:]))(float64[:], float64[:,:], int64[:])", cache=True
)
def calc_cross_points(x_grid, cond_ys, opt_idx):
    """
    Given a grid of x values, a matrix with the values of different line segments
    evaluated on the x grid, and a vector indicating the choice of a segment
    at each grid point, this function computes the coordinates of the
    crossing points that happen when the choice of segment changes.
    
    The purpose of the function is to take (x,y) lines that are defined piece-
    wise, and at every gap in x where the "piece" changes, find the point where
    the two "pieces" involved in the change would intercept.
    
    Adding these points to our piece-wise approximation will improve it, since
    it will eliminate interpolation between points that belong to different
    "pieces".
    
    Parameters
    ----------
    x_grid : np.array
        Grid of x values.
    cond_ys : 2-D np.array. Must have as many rows as possible segments, and
             len(x_grid) columns.
        cond_ys[i,j] contains the value of segment (or "piece") i at x_grid[j].
        Entries can be nan if the segment is not defined at a particular point.
    opt_idx : np.array of indices, must have length len(x_grid).
            Indicates what segment is to be used at each x gridpoint. The value
            of the piecewise function at x_grid[k] is cond_ys[opt_idx[k],k].
    
    Returns
    -------
    xing_points: 2D np.array
        Crossing points, each in its own row as an [x, y] pair.

    segments: np.array with two columns and as many rows as xing points.
        Each row represents a crossing point. The first column is the index
        of the segment used to the left, and the second, to the right.
    """

    # Compute differences in the optimal index,
    # to find positions of segment-changes
    diff_max = np.append(opt_idx[1:] - opt_idx[:-1], 0)
    idx_change = np.where(diff_max != 0)[0]

    # If no changes, return empty arrays
    if len(idx_change) == 0:

        points = np.zeros((0, 2), dtype=np.float64)
        segments = np.zeros((0, 2), dtype=np.int64)
        return points, segments

    else:

        # To find the crossing points we need the extremes of the intervals in
        # which they happen, and the two candidate segments evaluated in both
        # extremes. switch_interv[0] has the left points and switch_interv[1]
        # the right points of these intervals.
        switch_interv = np.stack((x_grid[idx_change], x_grid[idx_change + 1]), axis=1)

        # Store the indices of the two segments involved in the changes.
        # Columns are [0]: left extreme, [1]: right extreme,
        # Rows are individual crossing points.
        segments = np.stack((opt_idx[idx_change], opt_idx[idx_change + 1]), axis=1)

        # Get values of segments on both the left and the right
        left_y = np.zeros_like(segments, dtype=np.float64)
        right_y = np.zeros_like(segments, dtype=np.float64)

        for i, idx in enumerate(idx_change):

            left_y[i, 0] = cond_ys[segments[i, 0], idx]
            left_y[i, 1] = cond_ys[segments[i, 1], idx]

            right_y[i, 0] = cond_ys[segments[i, 0], idx + 1]
            right_y[i, 1] = cond_ys[segments[i, 1], idx + 1]

        # A valid crossing must have both switching segments well defined at the
        # encompassing gridpoints. Filter those that do not.
        valid = np.repeat(False, len(idx_change))
        for i in range(len(valid)):
            valid[i] = np.logical_and(
                ~np.isnan(left_y[i, :]).any(), ~np.isnan(right_y[i, :]).any()
            )

        if not np.any(valid):

            # If there are no valid crossings, return empty arrays.
            points = np.zeros((0, 2), dtype=np.float64)
            segments = np.zeros((0, 2), dtype=np.int64)
            return points, segments

        else:

            # Otherwise, subset valid crossings
            segments = segments[valid, :]
            switch_interv = switch_interv[valid, :]
            left_y = left_y[valid, :]
            right_y = right_y[valid, :]

            # Find crossing points.
            xing_points = [
                calc_linear_crossing(switch_interv[i, :], left_y[i, :], right_y[i, :])
                for i in range(segments.shape[0])
            ]

            xing_array = np.asarray(xing_points)

            return xing_array, segments


def calc_nondecreasing_segments(x, y):
    """
    Given a sequence of (x,y) points, this function finds the start and end
    indices of its largest non-decreasing segments.
    
    A non-decreasing segment is a sub-sequence of points
    {(x_0, y_0),...,(x_n,y_n)} such that for all 0 <= i,j <= n,
    If j>=i then x_j >= x_i and y_j >= y_i

    Parameters
    ----------
    x : 1D np.array of floats
        x coordinates of the sequence of points.
    y : 1D np.array of floats
        y coordinates of the sequence of points.

    Returns
    -------
    starts : 1D np.array of ints
        Indices where a new non-decreasing segment starts.
    ends : 1D np.array of ints
        Indices where a non-decreasing segment ends.

    """

    if len(x) == 0 or len(y) == 0 or len(y) != len(x):
        raise Exception("x and y must be non-empty arrays of the same size.")

    # Initialize
    starts = [0]
    ends = []

    for i in range(1, len(x)):

        # Check if grid decreases in x or v
        x_dec = x[i] < x[i - 1]
        y_dec = y[i] < y[i - 1]

        if x_dec or y_dec:

            ends.append(i - 1)
            starts.append(i)

        i = i + 1

    # The last segment always ends in the last point
    ends.append(len(y) - 1)

    starts = np.array(starts)
    ends = np.array(ends)

    return starts, ends


def upper_envelope(segments, calc_crossings=True):
    """
    Finds the upper envelope of a list of non-decreasing segments

    Parameters
    ----------
    segments : list of segments. Segments are tuples of arrays, with item[0]
        containing the x coordninates and item[1] the y coordinates of the
        points that confrom the segment item.
    calc_crossings : Bool, optional
        Indicates whether the crossing points at which the "upper" segment
        changes should be computed. The default is True.

    Returns
    -------
    x : np.array of floats
        x coordinates of the points that conform the upper envelope.
    y : np.array of floats
        y coordinates of the points that conform the upper envelope.
    env_inds : np array of ints
        Array of the same length as x and y. It indicates which of the
        provided segments is the "upper" one at every returned (x,y) point.

    """
    n_seg = len(segments)

    # Collect the x points of all segments in an ordered array, removing duplicates
    x = np.unique(np.concatenate([x[0] for x in segments]))

    # Interpolate all segments on every x point, without extrapolating.
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

    # Take the maximum to get the upper envelope.
    env_inds = np.nanargmax(y_cond, 0)
    y = y_cond[env_inds, range(len(x))]

    # Get crossing points if needed
    if calc_crossings:

        xing_points, xing_lines = calc_cross_points(x, y_cond, env_inds)

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

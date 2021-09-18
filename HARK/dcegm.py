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

@njit('Tuple((float64,float64))(float64[:], float64[:], float64[:])', cache = True)
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

@njit('Tuple((float64[:,:],int64[:,:]))(float64[:], float64[:,:], int64[:])', cache = True)
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
        points = np.zeros((0,2), dtype = np.float64)
        segments = np.zeros((0,2), dtype = np.int64)
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
            
            left_v[i,0] = condVs[idx, segments[i,0]]
            left_v[i,1] = condVs[idx, segments[i,1]]
            
            right_v[i,0] = condVs[idx + 1, segments[i,0]]
            right_v[i,1] = condVs[idx + 1, segments[i,1]]
            
        # A valid crossing must have both switching segments well defined at the
        # encompassing gridpoints. Filter those that do not.
        valid = np.repeat(False, len(idx_change))
        for i in range(len(valid)):    
            valid[i] = np.logical_and(~np.isnan(left_v[i,:]).any(), ~np.isnan(right_v[i,:]).any())
        
        if not np.any(valid):
            
            points = np.zeros((0,2), dtype = np.float64)
            segments = np.zeros((0,2), dtype = np.int64)
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
    for i in range(1,len(x)):
        
        # Check if grid decreases in x or v
        x_dec = x[i] < x[i - 1]
        v_dec = v[i] < v[i - 1]

        if x_dec or v_dec:
            
            ends.append(i-1)
            starts.append(i)

        i = i + 1

    # The last segment always starts in the last point
    ends.append(len(v) - 1)

    return np.array(starts), np.array(ends)


# think! nanargmax makes everythign super ugly because numpy changed the wraning
# in all nan slices to a valueerror...it's nans, aaarghgghg

def calc_multiline_envelope(M, C, V_T, commonM, find_crossings=False):
    """
    Do the envelope step of the DCEGM algorithm. Takes in market ressources,
    consumption levels, and inverse values from the EGM step. These represent
    (m, c) pairs that solve the necessary first order conditions. This function
    calculates the optimal (m, c, v_t) pairs on the commonM grid.

    Parameters
    ----------
    M : np.array
        market ressources from EGM step
    C : np.array
        consumption from EGM step
    V_T : np.array
        transformed values at the EGM grid
    commonM : np.array
        common grid to do upper envelope calculations on
    find_crossings: boolean
        should the exact crossing points of segments be computed and added to
        the grids?
    Returns
    -------


    """
    m_len = len(commonM)
    rise, fall = calc_nondecreasing_segments(M, V_T)

    num_kinks = len(fall)  # number of kinks / falling EGM grids

    # Use these segments to sequentially find upper envelopes. commonVARNAME
    # means the VARNAME evaluated on the common grid with a cloumn for each kink
    # discovered in calc_nondecreasing_segments. This means that commonVARNAME is a matrix
    # common grid length-by-number of segments to consider. In the end, we'll
    # use nanargmax over the columns to pick out the best (transformed) values.
    # This is why we fill the arrays with np.nan's.
    commonV_T = np.empty((m_len, num_kinks))
    commonV_T[:] = np.nan
    commonC = np.empty((m_len, num_kinks))
    commonC[:] = np.nan

    # Now, loop over all segments as defined by the "kinks" or the combination
    # of "rise" and "fall" indeces. These (rise[j], fall[j]) pairs define regions.

    if find_crossings:
        # We'll save V_T and C interpolating functions to aid crossing points later
        vT_funcs = []
        c_funcs = []

    for j in range(num_kinks):
        # Find points in the common grid that are in the range of the points in
        # the interval defined by (rise[j], fall[j]).
        below = M[rise[j]] >= commonM  # boolean array of bad indeces below
        above = M[fall[j]] <= commonM  # boolen array of bad indeces above
        in_range = below + above == 0  # pick out elements that are neither

        # based in in_range, find the relevant ressource values to interpolate
        m_eval = commonM[in_range]

        # create range of indeces in the input arrays
        idxs = range(rise[j], fall[j] + 1)
        # grab ressource values at the relevant indeces
        m_idx_j = M[idxs]

        # If we need to compute the xing points, create and store the
        # interpolating functions. Else simply create them.
        if find_crossings:
            # Create and store interpolating functions
            vT_funcs.append(LinearInterp(m_idx_j, V_T[idxs], lower_extrap=True))
            c_funcs.append(LinearInterp(m_idx_j, C[idxs], lower_extrap=True))

            vT_fun = vT_funcs[-1]
            c_fun = c_funcs[-1]
        else:
            vT_fun = LinearInterp(m_idx_j, V_T[idxs], lower_extrap=True)
            c_fun = LinearInterp(m_idx_j, C[idxs], lower_extrap=True)

        # re-interpolate to common grid
        commonV_T[in_range, j] = vT_fun(m_eval)  # NOQA
        commonC[in_range, j] = c_fun(
            m_eval
        )  # NOQA Interpolat econsumption also. May not be nesserary

    # for each row in the commonV_T matrix, see if all entries are np.nan. This
    # would mean that we have no valid value here, so we want to use this boolean
    # vector to filter out irrelevant entries of commonV_T.
    row_all_nan = np.array([np.all(np.isnan(row)) for row in commonV_T])
    # Now take the max of all these line segments.
    idx_max = np.zeros(commonM.size, dtype=int)
    idx_max[row_all_nan == False] = np.nanargmax(
        commonV_T[row_all_nan == False], axis=1
    )

    # prefix with upper for variable that are "upper enveloped"
    upperV_T = np.zeros(m_len)

    # Set the non-nan rows to the maximum over columns
    upperV_T[row_all_nan == False] = np.nanmax(
        commonV_T[row_all_nan == False, :], axis=1
    )
    # Set the rest to nan
    upperV_T[row_all_nan] = np.nan

    # Add the zero point in the bottom
    if np.isnan(upperV_T[0]):
        # in transformed space space, utility of zero-consumption (-inf) is 0.0
        upperV_T[0] = 0.0
        # commonM[0] is typically 0, so this is safe, but maybe it should be 0.0
        commonC[0] = commonM[0]

    # Extrapolate if NaNs are introduced due to the common grid
    # going outside all the sub-line segments
    IsNaN = np.isnan(upperV_T)
    upperV_T[IsNaN] = LinearInterp(commonM[IsNaN == False], upperV_T[IsNaN == False])(
        commonM[IsNaN]
    )
    LastBeforeNaN = np.append(np.diff(IsNaN) > 0, 0)
    LastId = LastBeforeNaN * idx_max  # Find last id-number
    idx_max[IsNaN] = LastId[IsNaN]
    # Linear index used to get optimal consumption based on "id"  from max
    ncols = commonC.shape[1]
    rowidx = np.cumsum(ncols * np.ones(len(commonM), dtype=int)) - ncols
    idx_linear = np.unravel_index(rowidx + idx_max, commonC.shape)
    upperC = commonC[idx_linear]
    upperC[IsNaN] = LinearInterp(commonM[IsNaN == 0], upperC[IsNaN == 0])(
        commonM[IsNaN]
    )

    upperM = commonM.copy()  # anticipate this TODO

    # If crossing points are requested, compute them. Else just return the
    # envelope.
    if not find_crossings:

        return upperM, upperC, upperV_T

    else:

        xing_points, segments = calc_cross_points(commonM, commonV_T, idx_max)
        # keep only the m component of crossing points.
        xing_points = [x[0] for x in xing_points]

        if len(xing_points) > 0:

            # Now construct a set of points that need to be added to the grid in
            # order to handle the discontinuities. To points per discontinuity:
            # one to the left and one to the right.
            num_crosses = len(xing_points)
            add_m = np.empty((num_crosses, 2))
            add_m[:] = np.nan
            add_vT = np.empty((num_crosses, 2))
            add_vT[:] = np.nan
            add_c = np.empty((num_crosses, 2))
            add_c[:] = np.nan

            # Fill the list of points interpolating left and right (2 points per
            # crossing)
            for i in range(num_crosses):
                # Left part of the discontinuity
                ml = xing_points[i]
                add_m[i, 0] = ml
                add_vT[i, 0] = vT_funcs[segments[i, 0]](ml)
                add_c[i, 0] = c_funcs[segments[i, 0]](ml)

                # Right part of the discontinuity
                mr = np.nextafter(ml, np.inf)
                add_m[i, 1] = mr
                add_vT[i, 1] = vT_funcs[segments[i, 1]](mr)
                add_c[i, 1] = c_funcs[segments[i, 1]](mr)

            # Flatten arrays
            add_m = add_m.flatten()
            add_vT = add_vT.flatten()
            add_c = add_c.flatten()

            # Filter any points already in the grid
            idxIncluded = np.isin(add_m, upperM)
            add_m = add_m[~idxIncluded]
            add_vT = add_vT[~idxIncluded]
            add_c = add_c[~idxIncluded]

            # Find positions at which new points must go
            insertIdx = np.searchsorted(upperM, add_m)

            # Insert
            upperM = np.insert(upperM, insertIdx, add_m)
            upperC = np.insert(upperC, insertIdx, add_c)
            upperV_T = np.insert(upperV_T, insertIdx, add_vT)

        return upperM, upperC, upperV_T, xing_points

# %% New methods

def upper_envelope(segments, calc_crossings = True):
    
    n_seg = len(segments)
    
    # Collect the x points of all segments in an ordered array
    x = np.sort(np.concatenate([x[0] for x in segments]))
    
    # Interpolate on all points using all segments
    y_cond = np.zeros((n_seg, len(x)))
    for i in range(n_seg):
        
        row = interp(segments[i][0], segments[i][1], x)
        extrap = np.logical_or(x < segments[i][0][0], x > segments[i][0][-1])
        row[extrap] = np.nan
        
        y_cond[i,:] = row
        
    # Take the maximum to get the upper envelope
    env_inds = np.nanargmax(y_cond, 0)
    y = y_cond[env_inds,range(len(x))]
    
    # Get crossing points if needed
    if calc_crossings:
        
        xing_points, xing_lines = calc_cross_points(x, y_cond.T, env_inds)
        
        if len(xing_points) > 0:
            # Extract x and y coordinates
            print(xing_points)
            xing_x = np.array([p[0] for p in xing_points])
            xing_y = np.array([p[1] for p in xing_points])
            
            # To capture the discontinuity, we'll add the successors of xing_x to
            # the grid
            succ = np.nextafter(xing_x, xing_x + 1)
            
            # Collect points to add to grids
            xtra_x = np.concatenate([xing_x, succ])
            # if there is a crossing, y will be the same on both segments
            xtra_y = np.concatenate([xing_y, xing_y]) 
            xtra_lines = np.concatenate([xing_lines[:,0], xing_lines[:,1]])
            
            # Insert them
            idx = np.searchsorted(x, xtra_x)
            x = np.insert(x, idx, xtra_x)
            y = np.insert(y, idx, xtra_y)
            env_inds = np.insert(env_inds, idx, xtra_lines)
        
    return x, y, env_inds
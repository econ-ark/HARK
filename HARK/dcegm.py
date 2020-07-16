"""
Functions for working with the discrete-continuous EGM (DCEGM) algorithm as
described in "The endogenous grid method for discrete-continuous dynamic
choice models with (or without) taste shocks" by Iskhakov et al. (2016)
[https://doi.org/10.3982/QE643 and ijrsDCEGM2017 in our Zotero]

Example can be found in https://github.com/econ-ark/DemARK/blob/master/notebooks/DCEGM-Upper-Envelope.ipynb
"""
import numpy as np
from HARK.interpolation import LinearInterp


def calcSegments(x, v):
    """
    Find index vectors `rise` and `fall` such that `rise` holds the indeces `i`
    such that x[i+1]>x[i] and `fall` holds indeces `j` such that either
    - x[j+1] < x[j] or,
    - x[j]>x[j-1] and v[j]<v[j-1].

    The vectors are essential to the DCEGM algorithm, as they definite the
    relevant intervals to be used to construct the upper envelope of potential
    solutions to the (necessary) first order conditions.

    Parameters
    ----------
    x : np.ndarray
        array of points where `v` is evaluated
    v : np.ndarray
        array of values of some function of `x`

    Returns
    -------
    rise : np.ndarray
        see description above
    fall : np.ndarray
        see description above
    """
    # NOTE: assumes that the first segment is in fact increasing (forced in EGM
    # by augmentation with the constrained segment).
    # elements in common grid g

    # Identify index intervals of falling and rising regions
    # We need these to construct the upper envelope because we need to discard
    # solutions from the inverted Euler equations that do not represent optimal
    # choices (the FOCs are only necessary in these models).
    #
    # `fall` is a vector of indeces that represent the first elements in all
    # of the falling segments (the curve can potentially fold several times)
    fall = np.empty(0, dtype=int)  # initialize with empty and then add the last point below while-loop

    rise = np.array([0])  # Initialize such thatthe lowest point is the first grid point
    i = 1  # Initialize
    while i <= len(x) - 2:
        # Check if the next (`ip1` stands for i plus 1) grid point is below the
        # current one, such that the line is folding back.
        ip1_falls = x[i+1] < x[i]  # true if grid decreases on index increment
        i_rose = x[i] > x[i-1]  # true if grid decreases on index decrement
        val_fell = v[i] < v[i-1]  # true if value rises on index decrement

        if (ip1_falls and i_rose) or (val_fell and i_rose):

            # we are in a region where the endogenous grid is decreasing or
            # the value function rises by stepping back in the grid.
            fall = np.append(fall, i)  # add the index to the vector

            # We now iterate from the current index onwards until we find point
            # where resources rises again. Unfortunately, we need to check
            # each points, as there can be multiple spells of falling endogenous
            # grids, so we cannot use bisection or some other fast algorithm.
            k = i
            while x[k+1] < x[k]:
                k = k + 1
            # k now holds either the next index the starts a new rising
            # region, or it holds the length of M, `m_len`.

            rise = np.append(rise, k)

            # Set the index to the point where resources again is rising
            i = k

        i = i + 1

    # Add the last index for convenience (then all segments are complete, as
    # len(fall) == len(rise), and we can form them by range(rise[j], fall[j]+1).
    fall = np.append(fall, len(v)-1)

    return rise, fall
# think! nanargmax makes everythign super ugly because numpy changed the wraning
# in all nan slices to a valueerror...it's nans, aaarghgghg

def calcLinearCrossing(m,left_v, right_v):
    """
    Computes the intersection between two line segments, defined by two common
    x points, and the values of both segments at both x points

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
    if there is no intercept in the interval [m[0],m[1]], (None,None)

    """
    
    # Find slopes of both segments
    delta_m = m[1] - m[0]
    s0 = (right_v[0] - left_v[0])/delta_m
    s1 = (right_v[1] - left_v[1])/delta_m
    
    if s1 == s0:
        if left_v[0] == left_v[1]:
            return (m[0],left_v[0])
        else:
            return (None, None)
    else:
        # Find h where intercept happens at m[0] + h
        h = (left_v[0] - left_v[1])/(s1-s0)
        if (h >= 0 and h <= (m[1]-m[0])):
            return (m[0]+h, left_v[0] + h*s0)

def calcMultilineEnvelope(M, C, V_T, commonM):
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

    Returns
    -------


    """
    m_len = len(commonM)
    rise, fall = calcSegments(M, V_T)

    num_kinks = len(fall)  # number of kinks / falling EGM grids

    # Use these segments to sequentially find upper envelopes. commonVARNAME
    # means the VARNAME evaluated on the common grid with a cloumn for each kink
    # discovered in calcSegments. This means that commonVARNAME is a matrix
    # common grid length-by-number of segments to consider. In the end, we'll
    # use nanargmax over the columns to pick out the best (transformed) values.
    # This is why we fill the arrays with np.nan's.
    commonV_T = np.empty((m_len, num_kinks))
    commonV_T[:] = np.nan
    commonC = np.empty((m_len, num_kinks))
    commonC[:] = np.nan

    # Now, loop over all segments as defined by the "kinks" or the combination
    # of "rise" and "fall" indeces. These (rise[j], fall[j]) pairs define regions.
    # We'll save V_T and C interpolating functions to aid crossing points later
    vT_funcs = []
    c_funcs  = []
    for j in range(num_kinks):
        # Find points in the common grid that are in the range of the points in
        # the interval defined by (rise[j], fall[j]).
        below = M[rise[j]] >= commonM  # boolean array of bad indeces below
        above = M[fall[j]] <= commonM  # boolen array of bad indeces above
        in_range = below + above == 0  # pick out elements that are neither

        # create range of indeces in the input arrays
        idxs = range(rise[j], fall[j]+1)
        # grab ressource values at the relevant indeces
        m_idx_j = M[idxs]
        
        # Create and store interpolating functions
        vT_funcs.append(LinearInterp(m_idx_j, V_T[idxs], lower_extrap=True))
        c_funcs.append(LinearInterp(m_idx_j, C[idxs], lower_extrap=True))
        
        # based in in_range, find the relevant ressource values to interpolate
        m_eval = commonM[in_range]

        # re-interpolate to common grid
        commonV_T[in_range, j] = vT_funcs[-1](m_eval) # NOQA
        commonC[in_range, j]   = c_funcs[-1](m_eval) # NOQA Interpolat econsumption also. May not be nesserary
    
    # for each row in the commonV_T matrix, see if all entries are np.nan. This
    # would mean that we have no valid value here, so we want to use this boolean
    # vector to filter out irrelevant entries of commonV_T.
    row_all_nan = np.array([np.all(np.isnan(row)) for row in commonV_T])
    # Now take the max of all these line segments.
    idx_max = np.zeros(commonM.size, dtype=int)
    idx_max[row_all_nan == False] = np.nanargmax(commonV_T[row_all_nan == False], axis=1)
    
    # Compute differences, to find positions of segment-changes (will be
    # used at the end)
    diff_max = np.insert(np.diff(idx_max),len(idx_max)-1,0)
    
    # prefix with upper for variable that are "upper enveloped"
    upperV_T = np.zeros(m_len)

    # Set the non-nan rows to the maximum over columns
    upperV_T[row_all_nan == False] = np.nanmax(commonV_T[row_all_nan == False, :], axis=1)
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
    upperV_T[IsNaN] = LinearInterp(commonM[IsNaN == False], upperV_T[IsNaN == False])(commonM[IsNaN])
    LastBeforeNaN = np.append(np.diff(IsNaN) > 0, 0)
    LastId = LastBeforeNaN*idx_max  # Find last id-number
    idx_max[IsNaN] = LastId[IsNaN]
    # Linear index used to get optimal consumption based on "id"  from max
    ncols = commonC.shape[1]
    rowidx = np.cumsum(ncols*np.ones(len(commonM), dtype=int))-ncols
    idx_linear = np.unravel_index(rowidx+idx_max, commonC.shape)
    upperC = commonC[idx_linear]
    upperC[IsNaN] = LinearInterp(commonM[IsNaN == 0], upperC[IsNaN == 0])(commonM[IsNaN])
    
    upperM = commonM.copy()  # anticipate this TODO
    
    # Now we'll find and insert the kink points if there are any
    
    # There is a change of segment if the argmax changes
    # (will deal with nan's later). [0] b/c np.where returns a tuple
    idx_change = np.where(diff_max != 0)[0]
    
    # If there is any change
    if len(idx_change) > 0:
        
        # To find the crossing points we need the extremes of the intervals in
        # which they happen, and the two candidate segments evaluated in both
        # extremes. switchMs[0] has the left points and switchMs[1] the right
        # points of these intervals.
        switchMs = np.stack([commonM[idx_change], commonM[idx_change + 1]],
                            axis = 1)
        
        # Store the indices of the two segments involved in the changes, by
        # looking at the argmax in the switching possitions.
        # Columns are [0]: left extreme, [1]: right extreme,
        # Rows are individual crossing points.
        segments = np.stack([idx_max[idx_change], idx_max[idx_change + 1]],
                            axis = 1)
    
        # Values of both segments on the left point
        left_v  = np.stack([commonV_T[idx_change, segments[:,0]],
                            commonV_T[idx_change, segments[:,1]]], axis = 1)
        # and the right point
        right_v = np.stack([commonV_T[idx_change+1, segments[:,0]],
                            commonV_T[idx_change+1, segments[:,1]]], axis = 1)
        
        # A valid crossing must have both switching segments well defined at the
        # encompassing gridpoints. Filter those that do not.
        valid = np.logical_and(~np.isnan(left_v).any(axis = 1),
                               ~np.isnan(right_v).any(axis = 1))
        
        segments = segments[valid,:]
        switchMs = switchMs[valid,:]
        left_v   = left_v[valid,:]
        right_v  = right_v[valid,:]
        
        # Find crossing points. Returns a list (m,v) tuples.
        xing_points = [calcLinearCrossing(switchMs[i,:],
                                          left_v[i,:], right_v[i,:])
                       for i in range(segments.shape[0])]
        
        # Now construct a set of points that need to be added to the grid in
        # order to handle the discontinuities. To points per discontinuity:
        # one to the left and one to the right.
        num_crosses = len(xing_points)
        add_m     = np.empty((num_crosses,2))
        add_m[:]  = np.nan
        add_vT    = np.empty((num_crosses,2))
        add_vT[:] = np.nan
        add_c     = np.empty((num_crosses,2))
        add_c[:]  = np.nan
        
        # Fill the list of points interpolating left and right (2 points per
        # crossing)
        for i in range(num_crosses):
            # Left part of the discontinuity
            ml = xing_points[i][0]
            add_m[i,0] = ml
            add_vT[i,0] = vT_funcs[segments[i,0]](ml)
            add_c[i,0] = c_funcs[segments[i,0]](ml)
            
            # Right part of the discontinuity
            mr = np.nextafter(ml, np.inf)
            add_m[i,1]  = mr
            add_vT[i,1] = vT_funcs[segments[i,1]](mr)
            add_c[i,1]  = c_funcs[segments[i,1]](mr)
        
        # Flatten arrays
        add_m = add_m.flatten()
        add_vT = add_vT.flatten()
        add_c = add_c.flatten()        

        # Filter any points already in the grid
        idxIncluded = np.isin(add_m, upperM)
        add_m  = add_m[~idxIncluded]
        add_vT = add_vT[~idxIncluded]
        add_c  = add_c[~idxIncluded]
        
        # Find positions at which new points must go
        insertIdx = np.searchsorted(upperM, add_m)
        
        # Insert
        upperM   = np.insert(upperM, insertIdx, add_m)
        upperC   = np.insert(upperC, insertIdx, add_c)
        upperV_T = np.insert(upperV_T, insertIdx, add_vT)
    
    return upperM, upperC, upperV_T

"""
General purpose  / miscellaneous functions.  Includes functions to approximate
continuous distributions with discrete ones, utility functions (and their
derivatives), manipulation of discrete distributions, and basic plotting tools.
"""
import cProfile
import functools
import os
import pstats
import re
import warnings

import numba
import numpy as np  # Python's numeric library, abbreviated "np"
from scipy.interpolate import interp1d

# try:
#     import matplotlib.pyplot as plt                 # Python's plotting library
# except ImportError:
#     import sys
#     exception_type, value, traceback = sys.exc_info()
#     raise ImportError('HARK must be used in a graphical environment.', exception_type, value, traceback)


def memoize(obj):
    """
    A decorator to (potentially) make functions more efficient.

    With this decorator, functions will "remember" if they have been evaluated with given inputs
    before.  If they have, they will "remember" the outputs that have already been calculated
    for those inputs, rather than calculating them again.
    """
    cache = obj._cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]

    return memoizer


# ==============================================================================
# ============== Some basic function tools  ====================================
# ==============================================================================


def get_arg_names(function):
    """
    Returns a list of strings naming all of the arguments for the passed function.

    Parameters
    ----------
    function : function
        A function whose argument names are wanted.

    Returns
    -------
    argNames : [string]
        The names of the arguments of function.
    """
    argCount = function.__code__.co_argcount
    argNames = function.__code__.co_varnames[:argCount]
    return argNames


class NullFunc(object):
    """
    A trivial class that acts as a placeholder "do nothing" function.
    """

    def __call__(self, *args):
        """
        Returns meaningless output no matter what the input(s) is.  If no input,
        returns None.  Otherwise, returns an array of NaNs (or a single NaN) of
        the same size as the first input.
        """
        if len(args) == 0:
            return None
        else:
            arg = args[0]
            if hasattr(arg, "shape"):
                return np.zeros_like(arg) + np.nan
            else:
                return np.nan

    def distance(self, other):
        """
        Trivial distance metric that only cares whether the other object is also
        an instance of NullFunc.  Intentionally does not inherit from HARKobject
        as this might create dependency problems.

        Parameters
        ----------
        other : any
            Any object for comparison to this instance of NullFunc.

        Returns
        -------
        (unnamed) : float
            The distance between self and other.  Returns 0 if other is also a
            NullFunc; otherwise returns an arbitrary high number.
        """
        try:
            if other.__class__ is self.__class__:
                return 0.0
            else:
                return 1000.0
        except:
            return 10000.0


# =======================================================
# ================ Other useful functions ===============
# =======================================================


def construct_assets_grid(parameters):
    """
    Constructs the base grid of post-decision states, representing end-of-period
    assets above the absolute minimum.

    All parameters are passed as attributes of the single input parameters.  The
    input can be an instance of a ConsumerType, or a custom Parameters class.

    Parameters
    ----------
    aXtraMin:                  float
        Minimum value for the a-grid
    aXtraMax:                  float
        Maximum value for the a-grid
    aXtraCount:                 int
        Size of the a-grid
    aXtraExtra:                [float]
        Extra values for the a-grid.
    exp_nest:               int
        Level of nesting for the exponentially spaced grid.
        If -1, the grid is linearly spaced.

    Returns
    -------
    aXtraGrid:     np.ndarray
        Base array of values for the post-decision-state grid.
    """
    # Unpack the parameters
    aXtraMin = parameters.aXtraMin
    aXtraMax = parameters.aXtraMax
    aXtraCount = parameters.aXtraCount
    aXtraExtra = parameters.aXtraExtra
    exp_nest = parameters.aXtraNestFac

    # Set up post decision state grid:
    if exp_nest == -1:
        aXtraGrid = np.linspace(aXtraMin, aXtraMax, aXtraCount)
    elif exp_nest >= 0:
        aXtraGrid = make_grid_exp_mult(
            ming=aXtraMin, maxg=aXtraMax, ng=aXtraCount, timestonest=exp_nest
        )
    else:
        raise ValueError(
            "aXtraNestFac not recognized in __init__."
            + "Please ensure aXtraNestFac is either -1 or a positive integer."
        )

    # Add in additional points for the grid:
    for a in aXtraExtra:
        if a is not None and a not in aXtraGrid:
            j = aXtraGrid.searchsorted(a)
            aXtraGrid = np.insert(aXtraGrid, j, a)

    return aXtraGrid


# ==============================================================================
# ============== Functions for generating state space grids  ===================
# ==============================================================================


def make_grid_exp_mult(ming, maxg, ng, timestonest=20):
    """
    Make a multi-exponentially spaced grid.

    Parameters
    ----------
    ming : float
        Minimum value of the grid
    maxg : float
        Maximum value of the grid
    ng : int
        The number of grid points
    timestonest : int
        the number of times to nest the exponentiation

    Returns
    -------
    points : np.array
        A multi-exponentially spaced grid

    Original Matab code can be found in Chris Carroll's
    [Solution Methods for Microeconomic Dynamic Optimization Problems]
    (http://www.econ2.jhu.edu/people/ccarroll/solvingmicrodsops/) toolkit.
    Latest update: 01 May 2015
    """
    if timestonest > 0:
        Lming = ming
        Lmaxg = maxg
        for j in range(timestonest):
            Lming = np.log(Lming + 1)
            Lmaxg = np.log(Lmaxg + 1)
        Lgrid = np.linspace(Lming, Lmaxg, ng)
        grid = Lgrid
        for j in range(timestonest):
            grid = np.exp(grid) - 1
    else:
        Lming = np.log(ming)
        Lmaxg = np.log(maxg)
        Lstep = (Lmaxg - Lming) / (ng - 1)
        Lgrid = np.arange(Lming, Lmaxg + 0.000001, Lstep)
        grid = np.exp(Lgrid)
    return grid


# ==============================================================================
# ============== Uncategorized general functions  ===================
# ==============================================================================


def calc_weighted_avg(data, weights):
    """
    Generates a weighted average of simulated data.  The Nth row of data is averaged
    and then weighted by the Nth element of weights in an aggregate average.

    Parameters
    ----------
    data : numpy.array
        An array of data with N rows of J floats
    weights : numpy.array
        A length N array of weights for the N rows of data.

    Returns
    -------
    weighted_sum : float
        The weighted sum of the data.
    """
    data_avg = np.mean(data, axis=1)
    weighted_sum = np.dot(data_avg, weights)
    return weighted_sum


def get_percentiles(data, weights=None, percentiles=None, presorted=False):
    """
    Calculates the requested percentiles of (weighted) data.  Median by default.

    Parameters
    ----------
    data : numpy.array
        A 1D array of float data.
    weights : np.array
        A weighting vector for the data.
    percentiles : [float]
        A list or numpy.array of percentiles to calculate for the data.  Each element should
        be in (0,1).
    presorted : boolean
        Indicator for whether data has already been sorted.

    Returns
    -------
    pctl_out : numpy.array
        The requested percentiles of the data.
    """
    if percentiles is None:
        percentiles = [0.5]
    else:
        if (
            not isinstance(percentiles, (list, np.ndarray))
            or min(percentiles) <= 0
            or max(percentiles) >= 1
        ):
            raise ValueError(
                "Percentiles should be a list or numpy array of floats between 0 and 1"
            )

    if data.size < 2:
        return np.zeros(np.array(percentiles).shape) + np.nan

    if weights is None:  # Set equiprobable weights if none were passed
        weights = np.ones(data.size) / float(data.size)

    if presorted:  # Sort the data if it is not already
        data_sorted = data
        weights_sorted = weights
    else:
        order = np.argsort(data)
        data_sorted = data[order]
        weights_sorted = weights[order]

    cum_dist = np.cumsum(weights_sorted) / np.sum(
        weights_sorted
    )  # cumulative probability distribution

    # Calculate the requested percentiles by interpolating the data over the
    # cumulative distribution, then evaluating at the percentile values
    inv_CDF = interp1d(cum_dist, data_sorted, bounds_error=False, assume_sorted=True)
    pctl_out = inv_CDF(percentiles)
    return pctl_out


def get_lorenz_shares(data, weights=None, percentiles=None, presorted=False):
    """
    Calculates the Lorenz curve at the requested percentiles of (weighted) data.
    Median by default.

    Parameters
    ----------
    data : numpy.array
        A 1D array of float data.
    weights : numpy.array
        A weighting vector for the data.
    percentiles : [float]
        A list or numpy.array of percentiles to calculate for the data.  Each element should
        be in (0,1).
    presorted : boolean
        Indicator for whether data has already been sorted.

    Returns
    -------
    lorenz_out : numpy.array
        The requested Lorenz curve points of the data.
    """
    if percentiles is None:
        percentiles = [0.5]
    else:
        if (
            not isinstance(percentiles, (list, np.ndarray))
            or min(percentiles) <= 0
            or max(percentiles) >= 1
        ):
            raise ValueError(
                "Percentiles should be a list or numpy array of floats between 0 and 1"
            )
    if weights is None:  # Set equiprobable weights if none were given
        weights = np.ones(data.size)

    if presorted:  # Sort the data if it is not already
        data_sorted = data
        weights_sorted = weights
    else:
        order = np.argsort(data)
        data_sorted = data[order]
        weights_sorted = weights[order]

    cum_dist = np.cumsum(weights_sorted) / np.sum(
        weights_sorted
    )  # cumulative probability distribution
    temp = data_sorted * weights_sorted
    cum_data = np.cumsum(temp) / sum(temp)  # cumulative ownership shares

    # Calculate the requested Lorenz shares by interpolating the cumulative ownership
    # shares over the cumulative distribution, then evaluating at requested points
    lorenzFunc = interp1d(cum_dist, cum_data, bounds_error=False, assume_sorted=True)
    lorenz_out = lorenzFunc(percentiles)
    return lorenz_out


def calc_subpop_avg(data, reference, cutoffs, weights=None):
    """
    Calculates the average of (weighted) data between cutoff percentiles of a
    reference variable.

    Parameters
    ----------
    data : numpy.array
        A 1D array of float data.
    reference : numpy.array
        A 1D array of float data of the same length as data.
    cutoffs : [(float,float)]
        A list of doubles with the lower and upper percentile bounds (should be
        in [0,1]).
    weights : numpy.array
        A weighting vector for the data.

    Returns
    -------
    slice_avg
        The (weighted) average of data that falls within the cutoff percentiles
        of reference.

    """
    if weights is None:  # Set equiprobable weights if none were given
        weights = np.ones(data.size)

    # Sort the data and generate a cumulative distribution
    order = np.argsort(reference)
    data_sorted = data[order]
    weights_sorted = weights[order]
    cum_dist = np.cumsum(weights_sorted) / np.sum(weights_sorted)

    # For each set of cutoffs, calculate the average of data that falls within
    # the cutoff percentiles of reference
    slice_avg = []
    for j in range(len(cutoffs)):
        bot = np.searchsorted(cum_dist, cutoffs[j][0])
        top = np.searchsorted(cum_dist, cutoffs[j][1])
        slice_avg.append(
            np.sum(data_sorted[bot:top] * weights_sorted[bot:top])
            / np.sum(weights_sorted[bot:top])
        )
    return slice_avg


def kernel_regression(x, y, bot=None, top=None, N=500, h=None):
    """
    Performs a non-parametric Nadaraya-Watson 1D kernel regression on given data
    with optionally specified range, number of points, and kernel bandwidth.

    Parameters
    ----------
    x : np.array
        The independent variable in the kernel regression.
    y : np.array
        The dependent variable in the kernel regression.
    bot : float
        Minimum value of interest in the regression; defaults to min(x).
    top : float
        Maximum value of interest in the regression; defaults to max(y).
    N : int
        Number of points to compute.
    h : float
        The bandwidth of the (Epanechnikov) kernel. To-do: GENERALIZE.

    Returns
    -------
    regression : LinearInterp
        A piecewise locally linear kernel regression: y = f(x).
    """
    # Fix omitted inputs
    if bot is None:
        bot = np.min(x)
    if top is None:
        top = np.max(x)
    if h is None:
        h = 2.0 * (top - bot) / float(N)  # This is an arbitrary default

    # Construct a local linear approximation
    x_vec = np.linspace(bot, top, num=N)
    y_vec = np.zeros_like(x_vec) + np.nan
    for j in range(N):
        x_here = x_vec[j]
        weights = epanechnikov_kernel(x, x_here, h)
        y_vec[j] = np.dot(weights, y) / np.sum(weights)
    regression = interp1d(x_vec, y_vec, bounds_error=False, assume_sorted=True)
    return regression


def epanechnikov_kernel(x, ref_x, h=1.0):
    """
    The Epanechnikov kernel.

    Parameters
    ----------
    x : np.array
        Values at which to evaluate the kernel
    x_ref : float
        The reference point
    h : float
        Kernel bandwidth

    Returns
    -------
    out : np.array
        Kernel values at each value of x
    """
    u = (x - ref_x) / h  # Normalize distance by bandwidth
    these = np.abs(u) <= 1.0  # Kernel = 0 outside [-1,1]
    out = np.zeros_like(x)  # Initialize kernel output
    out[these] = 0.75 * (1.0 - u[these] ** 2.0)  # Evaluate kernel
    return out


@numba.njit
def jump_to_grid_1D(m_vals, probs, Dist_mGrid):
    """
    Distributes values onto a predefined grid, maintaining the means.


    Parameters
    ----------
    m_vals: np.array
            Market resource values

    probs: np.array
            Shock probabilities associated with combinations of m_vals.
            Can be thought of as the probability mass function  of (m_vals).

    dist_mGrid : np.array
            Grid over normalized market resources

    Returns
    -------
    probGrid.flatten(): np.array
             Probabilities of each gridpoint on the combined grid of market resources

    """

    probGrid = np.zeros(len(Dist_mGrid))
    mIndex = np.digitize(m_vals, Dist_mGrid) - 1
    mIndex[m_vals <= Dist_mGrid[0]] = -1
    mIndex[m_vals >= Dist_mGrid[-1]] = len(Dist_mGrid) - 1

    for i in range(len(m_vals)):
        if mIndex[i] == -1:
            mlowerIndex = 0
            mupperIndex = 0
            mlowerWeight = 1.0
            mupperWeight = 0.0
        elif mIndex[i] == len(Dist_mGrid) - 1:
            mlowerIndex = -1
            mupperIndex = -1
            mlowerWeight = 1.0
            mupperWeight = 0.0
        else:
            mlowerIndex = mIndex[i]
            mupperIndex = mIndex[i] + 1
            mlowerWeight = (Dist_mGrid[mupperIndex] - m_vals[i]) / (
                Dist_mGrid[mupperIndex] - Dist_mGrid[mlowerIndex]
            )
            mupperWeight = 1.0 - mlowerWeight

        probGrid[mlowerIndex] += probs[i] * mlowerWeight
        probGrid[mupperIndex] += probs[i] * mupperWeight

    return probGrid.flatten()


@numba.njit
def jump_to_grid_2D(m_vals, perm_vals, probs, dist_mGrid, dist_pGrid):
    """
    Distributes values onto a predefined grid, maintaining the means. m_vals and perm_vals are realizations of market resources and permanent income while
    dist_mGrid and dist_pGrid are the predefined grids of market resources and permanent income, respectively. That is, m_vals and perm_vals do not necesarily lie on their
    respective grids. Returns probabilities of each gridpoint on the combined grid of market resources and permanent income.


    Parameters
    ----------
    m_vals: np.array
            Market resource values

    perm_vals: np.array
            Permanent income values

    probs: np.array
            Shock probabilities associated with combinations of m_vals and perm_vals.
            Can be thought of as the probability mass function  of (m_vals, perm_vals).

    dist_mGrid : np.array
            Grid over normalized market resources

    dist_pGrid : np.array
            Grid over permanent income
    Returns
    -------
    probGrid.flatten(): np.array
             Probabilities of each gridpoint on the combined grid of market resources and permanent income
    """

    probGrid = np.zeros((len(dist_mGrid), len(dist_pGrid)))

    # Maybe use np.searchsorted as opposed to np.digitize
    mIndex = (
        np.digitize(m_vals, dist_mGrid) - 1
    )  # Array indicating in which bin each values of m_vals lies in relative to dist_mGrid. Bins lie between between point of Dist_mGrid.
    # For instance, if mval lies between dist_mGrid[4] and dist_mGrid[5] it is in bin 4 (would be 5 if 1 was not subtracted in the previous line).
    mIndex[
        m_vals <= dist_mGrid[0]
    ] = (
        -1
    )  # if the value is less than the smallest value on dist_mGrid assign it an index of -1
    mIndex[m_vals >= dist_mGrid[-1]] = (
        len(dist_mGrid) - 1
    )  # if value if greater than largest value on dist_mGrid assign it an index of the length of the grid minus 1

    # the following three lines hold the same intuition as above
    pIndex = np.digitize(perm_vals, dist_pGrid) - 1
    pIndex[perm_vals <= dist_pGrid[0]] = -1
    pIndex[perm_vals >= dist_pGrid[-1]] = len(dist_pGrid) - 1

    for i in range(len(m_vals)):
        if (
            mIndex[i] == -1
        ):  # if mval is below smallest gridpoint, then assign it a weight of 1.0 for lower weight.
            mlowerIndex = 0
            mupperIndex = 0
            mlowerWeight = 1.0
            mupperWeight = 0.0
        elif (
            mIndex[i] == len(dist_mGrid) - 1
        ):  # if mval is greater than maximum gridpoint, then assign the following weights
            mlowerIndex = -1
            mupperIndex = -1
            mlowerWeight = 1.0
            mupperWeight = 0.0
        else:  # Standard case where mval does not lie past any extremes
            # identify which two points on the grid the mval is inbetween
            mlowerIndex = mIndex[i]
            mupperIndex = mIndex[i] + 1
            # Assign weight to the indices that bound the m_vals point. Intuitively, an mval perfectly between two points on the mgrid will assign a weight of .5 to the gridpoint above and below
            mlowerWeight = (dist_mGrid[mupperIndex] - m_vals[i]) / (
                dist_mGrid[mupperIndex] - dist_mGrid[mlowerIndex]
            )  # Metric to determine weight of gridpoint/index below. Intuitively, mvals that are close to gridpoint/index above are assigned a smaller mlowerweight.
            mupperWeight = 1.0 - mlowerWeight  # weight of gridpoint/ index above

        # Same logic as above except the weights here concern the permanent income grid
        if pIndex[i] == -1:
            plowerIndex = 0
            pupperIndex = 0
            plowerWeight = 1.0
            pupperWeight = 0.0
        elif pIndex[i] == len(dist_pGrid) - 1:
            plowerIndex = -1
            pupperIndex = -1
            plowerWeight = 1.0
            pupperWeight = 0.0
        else:
            plowerIndex = pIndex[i]
            pupperIndex = pIndex[i] + 1
            plowerWeight = (dist_pGrid[pupperIndex] - perm_vals[i]) / (
                dist_pGrid[pupperIndex] - dist_pGrid[plowerIndex]
            )
            pupperWeight = 1.0 - plowerWeight

        # Compute probabilities of each gridpoint on the combined market resources and permanent income grid by looping through each point on the combined market resources and permanent income grid,
        # assigning probabilities to each gridpoint based off the probabilities of the surrounding mvals and pvals and their respective weights placed on the gridpoint.
        # Note* probs[i] is the probability of mval AND pval occurring

        probGrid[mlowerIndex][plowerIndex] += (
            probs[i] * mlowerWeight * plowerWeight
        )  # probability of gridpoint below mval and pval
        probGrid[mlowerIndex][pupperIndex] += (
            probs[i] * mlowerWeight * pupperWeight
        )  # probability of gridpoint below mval and above pval
        probGrid[mupperIndex][plowerIndex] += (
            probs[i] * mupperWeight * plowerWeight
        )  # probability of gridpoint above mval and below pval
        probGrid[mupperIndex][pupperIndex] += (
            probs[i] * mupperWeight * pupperWeight
        )  # probability of gridpoint above mval and above pval

    return probGrid.flatten()


@numba.njit(parallel=True)
def gen_tran_matrix_1D(
    dist_mGrid, bNext, shk_prbs, perm_shks, tran_shks, LivPrb, NewBornDist
):
    """
    Computes Transition Matrix across normalized market resources.
    This function is built to non-stochastic simulate the IndShockConsumerType.
    This function is used exclusively when Harmenberg Neutral Measure is applied and/or if permanent income is not a state variable
    For more information, see https://econ-ark.org/materials/harmenberg-aggregation?launch

    Parameters
    ----------
    dist_mGrid : np.array
        Grid over normalized market resources

    bNext : np.array
        Grid over bank balances

    shk_prbs : np.array
        Array of shock probabilities over combinations of permanent and transitory shocks

    perm_shks : np.array
        Array of shocks to permanent income. Shocks should follow Harmenberg neutral measure

    tran_shks : np.array
        Array of shocks to transitory

    LivPrb : float
        Probability of not dying

    NewBornDist : np.array
        array representing distribution of newborns across grid of normalized market resources and grid of permanent income.

    Returns
    -------
    TranMatrix : np.array
        Transition Matrix over normalized market resources grid.


    """

    TranMatrix = np.zeros((len(dist_mGrid), len(dist_mGrid)))
    for i in numba.prange(len(dist_mGrid)):
        mNext_ij = (
            bNext[i] / perm_shks + tran_shks
        )  # Compute next period's market resources given todays bank balances bnext[i]
        TranMatrix[:, i] += (
            LivPrb * jump_to_grid_1D(mNext_ij, shk_prbs, dist_mGrid)
            + (1.0 - LivPrb) * NewBornDist
        )  # this is the transition matrix if given you are unemployed today and unemployed tomorrow so you assume the unemployed consumption policy
    return TranMatrix


@numba.njit(parallel=True)
def gen_tran_matrix_2D(
    dist_mGrid, dist_pGrid, bNext, shk_prbs, perm_shks, tran_shks, LivPrb, NewBornDist
):
    """
    Computes Transition Matrix over normalized market resources and permanent income.
    This function is built to non-stochastic simulate the IndShockConsumerType.

    Parameters
    ----------
    dist_mGrid : np.array
        Grid over normalized market resources

    dist_pGrid : np.array
        Grid over permanent income

    bNext : np.array
        Grid over bank balances

    shk_prbs : np.array
        Array of shock probabilities over combinations of perm and tran shocks

    perm_shks : np.array
        Array of shocks to permanent income

    tran_shks : np.array
        Array of shocks to transitory income

    LivPrb : float
        Probability of not dying

    NewBornDist : np.array
         array representing distribution of newborns across grid of normalized market resources and grid of permanent income.

    Returns
    -------
    TranMatrix : np.array
        Transition Matrix over normalized market resources grid and permanent income grid
    """
    TranMatrix = np.zeros(
        (len(dist_mGrid) * len(dist_pGrid), len(dist_mGrid) * len(dist_pGrid))
    )
    for i in numba.prange(len(dist_mGrid)):
        for j in numba.prange(len(dist_pGrid)):
            mNext_ij = (
                bNext[i] / perm_shks + tran_shks
            )  # Compute next period's market resources given todays bank balances bnext[i]
            pNext_ij = (
                dist_pGrid[j] * perm_shks
            )  # Computes next period's permanent income level by applying permanent income shock
            TranMatrix[:, i * len(dist_pGrid) + j] += (
                LivPrb
                * jump_to_grid_2D(mNext_ij, pNext_ij, shk_prbs, dist_mGrid, dist_pGrid)
                + (1.0 - LivPrb) * NewBornDist
            )  # this is the transition matrix if given you are unemployed today and unemployed tomorrow so you assume the unemployed consumption policy
    return TranMatrix


# ==============================================================================
# ============== Some basic plotting tools  ====================================
# ==============================================================================


def plot_funcs(functions, bottom, top, N=1000, legend_kwds=None):
    """
    Plots 1D function(s) over a given range.

    Parameters
    ----------
    functions : [function] or function
        A single function, or a list of functions, to be plotted.
    bottom : float
        The lower limit of the domain to be plotted.
    top : float
        The upper limit of the domain to be plotted.
    N : int
        Number of points in the domain to evaluate.
    legend_kwds: None, or dictionary
        If not None, the keyword dictionary to pass to plt.legend

    Returns
    -------
    none
    """
    import matplotlib.pyplot as plt

    if type(functions) == list:
        function_list = functions
    else:
        function_list = [functions]

    for function in function_list:
        x = np.linspace(bottom, top, N, endpoint=True)
        y = function(x)
        plt.plot(x, y)
    plt.xlim([bottom, top])
    if legend_kwds is not None:
        plt.legend(**legend_kwds)
    plt.show()


def plot_funcs_der(functions, bottom, top, N=1000, legend_kwds=None):
    """
    Plots the first derivative of 1D function(s) over a given range.

    Parameters
    ----------
    function : function
        A function or list of functions, the derivatives of which are to be plotted.
    bottom : float
        The lower limit of the domain to be plotted.
    top : float
        The upper limit of the domain to be plotted.
    N : int
        Number of points in the domain to evaluate.
    legend_kwds: None, or dictionary
        If not None, the keyword dictionary to pass to plt.legend

    Returns
    -------
    none
    """
    import matplotlib.pyplot as plt

    if type(functions) == list:
        function_list = functions
    else:
        function_list = [functions]

    step = (top - bottom) / N
    for function in function_list:
        x = np.arange(bottom, top, step)
        y = function.derivative(x)
        plt.plot(x, y)
    plt.xlim([bottom, top])
    if legend_kwds is not None:
        plt.legend(**legend_kwds)
    plt.show()


def determine_platform():
    """Untility function to return the platform currenlty in use.

    Returns
    ---------
    pf: str
        'darwin' (MacOS), 'debian'(debian Linux) or 'win' (windows)
    """
    import platform

    pform = platform.system().lower()
    if "darwin" in pform:
        pf = "darwin"  # MacOS
    elif "debian" in pform:
        pf = "debian"  # Probably cloud (MyBinder, CoLab, ...)
    elif "ubuntu" in pform:
        pf = "debian"  # Probably cloud (MyBinder, CoLab, ...)
    elif "win" in pform:
        pf = "win"
    elif "linux" in pform:
        pf = "linux"
    else:
        raise ValueError("Not able to find out the platform.")
    return pf


def test_latex_installation(pf):
    """Test to check if latex is installed on the machine.

    Parameters
    -----------
    pf: str (platform)
        output of determine_platform()

    Returns
    --------
    bool: Boolean
        True if latex found, else installed in the case of debian
        otherwise ImportError raised to direct user to install latex manually
    """
    # Test whether latex is installed (some of the figures require it)
    from distutils.spawn import find_executable

    latexExists = False

    if find_executable("latex"):
        latexExists = True
        return True

    if not latexExists:
        print("Some of the figures below require a full installation of LaTeX")
        # If running on Mac or Win, user can be assumed to be able to install
        # any missing packages in response to error messages; but not on cloud
        # so load LaTeX by hand (painfully slowly)
        if "debian" in pf:  # CoLab and MyBinder are both ubuntu
            print("Installing LaTeX now; please wait 3-5 minutes")
            from IPython.utils import io

            with io.capture_output() as captured:  # Hide hideously long output
                os.system("apt-get update")
                os.system(
                    "apt-get install texlive texlive-latex-extra texlive-xetex dvipng"
                )
                latexExists = True
            return True
        else:
            raise ImportError(
                "Please install a full distribution of LaTeX on your computer then rerun. \n \
            A full distribution means textlive, texlive-latex-extras, texlive-xetex, dvipng, and ghostscript"
            )


def in_ipynb():
    """If the ipython process contains 'terminal' assume not in a notebook.

    Returns
    --------
    bool: Boolean
          True if called from a jupyter notebook, else False
    """
    try:
        if "terminal" in str(type(get_ipython())):
            return False
        else:
            return True
    except NameError:
        return False


def setup_latex_env_notebook(pf, latexExists):
    """This is needed for use of the latex_envs notebook extension
    which allows the use of environments in Markdown.

    Parameters
    -----------
    pf: str (platform)
        output of determine_platform()
    """
    import os

    import matplotlib.pyplot as plt

    plt.rc("font", family="serif")
    plt.rc("text", usetex=latexExists)
    if latexExists:
        latex_preamble = (
            r"\usepackage{amsmath}\usepackage{amsfonts}"
            r"\usepackage[T1]{fontenc}"
            r"\providecommand{\Ex}{\mathbb{E}}"
            r"\providecommand{\StE}{\check}"
            r"\providecommand{\Trg}{\hat}"
            r"\providecommand{\PermGroFac}{\Gamma}"
            r"\providecommand{\cLev}{\pmb{\mathrm{c}}}"
            r"\providecommand{\mLev}{\pmb{\mathrm{m}}}"
            r"\providecommand{\Rfree}{\mathsf{R}}"
            r"\providecommand{\DiscFac}{\beta}"
            r"\providecommand{\CRRA}{\rho}"
            r"\providecommand{\MPC}{\kappa}"
            r"\providecommand{\UnempPrb}{\wp}"
        )
        # Latex expects paths to be separated by /. \ might result in pieces
        # being interpreted as commands.
        latexdefs_path = os.getcwd().replace(os.path.sep, "/") + "/latexdefs.tex"
        if os.path.isfile(latexdefs_path):
            latex_preamble = latex_preamble + r"\input{" + latexdefs_path + r"}"
        else:  # the required latex_envs package needs this file to exist even if it is empty
            from pathlib import Path

            Path(latexdefs_path).touch()
        plt.rcParams["text.latex.preamble"] = latex_preamble


def make_figs(figure_name, saveFigs, drawFigs, target_dir="Figures"):
    """Utility function to save figure in multiple formats and display the image.

    Parameters
    ----------
    figure_name: str
                name of the figure
    saveFigs: bool
              True if the figure needs to be written to disk else False
    drawFigs: bool
              True if the figure should be displayed using plt.draw()
    target_dir: str, default = 'Figures/'
              Name of folder to save figures to in the current directory

    """
    import matplotlib.pyplot as plt

    if saveFigs:
        import os

        # Where to put any figures that the user wants to save
        my_file_path = os.getcwd()  # Find pathname to this file:
        Figures_dir = os.path.join(
            my_file_path, "{}".format(target_dir)
        )  # LaTeX document assumes figures will be here
        if not os.path.exists(Figures_dir):
            os.makedirs(Figures_dir)  # If dir does not exist, create it
        # Save the figures in several formats
        print("Saving figure {} in {}".format(figure_name, target_dir))
        plt.savefig(
            os.path.join(target_dir, "{}.jpg".format(figure_name)),
            metadata={"CreationDate": None},
        )  # For web/html
        plt.savefig(
            os.path.join(target_dir, "{}.png".format(figure_name)),
            metadata={"CreationDate": None},
        )  # For web/html
        plt.savefig(
            os.path.join(target_dir, "{}.pdf".format(figure_name)),
            metadata={"CreationDate": None},
        )  # For LaTeX
        plt.savefig(
            os.path.join(target_dir, "{}.svg".format(figure_name)),
            metadata={"Date": None},
        )  # For html5
    # Make sure it's possible to plot it by checking for GUI
    if drawFigs and find_gui():
        plt.ion()  # Counterintuitively, you want interactive mode on if you don't want to interact
        plt.draw()  # Change to false if you want to pause after the figure
        plt.pause(2)


def find_gui():
    """Quick fix to check if matplotlib is running in a GUI environment.

    Returns
    -------
    bool: Boolean
          True if it's a GUI environment, False if not.
    """
    try:
        import matplotlib.pyplot as plt
    except:
        return False
    if plt.get_backend() == "Agg":
        return False
    return True


def benchmark(
    agent_type, sort_by="tottime", max_print=10, filename="restats", return_output=False
):
    """
    Profiling tool for HARK models. Calling `benchmark` on agents calls the solver for
    the agents and provides time to solve as well as the top `max_print` function calls
    in terms of  `sort_by`. Optionally allows for saving a text copy of the profile
    as well as returning the `Stats` object for further inspection.

    For more details on the python profilers, see
    https://docs.python.org/3/library/profile.html#the-stats-class

    Parameters
    ----------
    agent_type: AgentType
            A HARK AgentType with a solve() method.
    sort_by: string
            A string to sort the stats by.
    max_print: int
            Number of lines to print
    filename: string
            Optional filename to save output.
    return_output: bool
            Boolean to determine whether Stats object should be returned.

    Returns
    -------
    stats: Stats (optional)
          Profiling object with call statistics.
    """

    agent = agent_type
    cProfile.run("agent.solve()", filename)
    stats = pstats.Stats(filename)
    stats.strip_dirs()
    stats.sort_stats(sort_by)
    stats.print_stats(max_print)
    if return_output:
        return stats


simpledec = re.compile(r"\d*\.\d{8,20}")


def mround(match):
    return "{:.5f}".format(float(match.group()))


def round_in_file(filename):
    with open(filename, "r+") as file:
        filetext = file.read()
        filetext = re.sub(simpledec, mround, filetext)
        file.seek(0)
        file.write(filetext)
        file.truncate()


def files_in_dir(mypath):
    return [
        os.path.join(mypath, f)
        for f in os.listdir(mypath)
        if os.path.isfile(os.path.join(mypath, f))
    ]

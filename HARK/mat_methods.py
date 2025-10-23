from typing import List

import numpy as np
from numba import njit


@njit
def ravel_index(
    ind_mat: np.ndarray, dims: np.ndarray
) -> np.ndarray:  # pragma: no cover
    """
    This function takes a matrix of indices, and a vector of dimensions, and
    returns a vector of corresponding flattened indices
    """
    # Initialize indices
    r_ind = np.zeros(ind_mat.shape[1:], dtype=np.int64)
    # Find index multipliers
    cdims = np.concatenate((np.cumprod(dims[1:][::-1])[::-1], np.array([1])))
    for i, cdim in enumerate(cdims):
        r_ind += ind_mat[i] * cdim

    return r_ind


@njit
def multidim_get_lower_index(
    points: np.ndarray, grids: List[np.ndarray], dims: np.ndarray
) -> np.ndarray:  # pragma: no cover
    """
    Get the lower index for each point in a multidimensional grid.

    Parameters
    ----------
    points : np.ndarray
        The points for which to find the lower index.
    grids : List[np.ndarray]
        The grids for each dimension.
    dims : np.ndarray
        The dimensions of the grids.

    Returns
    -------
    np.ndarray
        The indices of the lower grid point for each point in each dimension.
    """
    inds = np.empty_like(points, dtype=np.int64)
    for i, grid in enumerate(grids):
        inds[:, i] = np.minimum(
            np.searchsorted(grid, points[:, i], side="right") - 1, dims[i] - 2
        )

    return inds


@njit
def fwd_and_bwd_diffs(
    points: np.ndarray, grids: List[np.ndarray], inds: np.ndarray
) -> np.ndarray:  # pragma: no cover
    """
    Computes backward and forward differences for each point in points for each grid in grids.

    Parameters
    ----------
    points : np.ndarray
        The points for which to compute the differences.
    grids : List[np.ndarray]
        The grids for each dimension.
    inds : np.ndarray
        The indices of the lower grid point for each point in each dimension.

    Returns
    -------
    np.ndarray
        A (2, ndim, npoints) matrix in which [:,i,:] is the backward and forward difference for the ith dimension.
    """
    # Preallocate
    diffs = np.empty((2, points.shape[1], points.shape[0]))

    for i, grid in enumerate(grids):
        # Backward
        diffs[0, i, :] = points[:, i] - grid[inds[i, :]]
        # Forward
        diffs[1, i, :] = grid[inds[i, :] + 1] - points[:, i]

    return diffs


@njit
def sum_weights(
    weights: np.ndarray, dims: np.ndarray, add_inds: np.ndarray
) -> np.ndarray:  # pragma: no cover
    """
    Sums the weights that correspond to each point in the grid.

    Parameters
    ----------
    weights : np.ndarray
        The weights to be summed.
    dims : np.ndarray
        The dimensions of the grid.
    add_inds : np.ndarray
        The indices of the points in the grid to which the weights correspond.

    Returns
    -------
    np.ndarray
        The sum of the weights for each point in the grid (flattened).
    """
    # Initialize arary to hold weights.
    distr = np.zeros(np.prod(dims), dtype=np.float64)

    # Add weights point by point
    for i in range(weights.shape[1]):
        distr[add_inds[:, i]] += weights[:, i]

    return distr


@njit
def denominators(
    inds: np.ndarray, grids: List[np.ndarray]
) -> np.ndarray:  # pragma: no cover
    """
    This function computes the denominators of the interpolation weights,
    which are the areas of the hypercubes of the grid that contain the points.

    Parameters
    ----------
    inds : np.ndarray
        The indices of the lower grid point for each point in each dimension.
    grids : List[np.ndarray]
        The grids for each dimension.

    Returns
    -------
    np.ndarray
        The denominators of the interpolation weights.
    """
    denoms = np.ones(inds.shape[1], dtype=np.float64)
    for i, g in enumerate(grids):
        d = np.diff(g)
        denoms *= d[inds[i, :]]
    return denoms


@njit
def get_combinations(ndim: int) -> np.ndarray:  # pragma: no cover
    """
    Produces an array with all the 2**ndim possible combinations of 0s and 1s.
    This is used later to generate all the possible combinations of backward and forward differences.

    Parameters
    ----------
    ndim : int
        The number of dimensions.

    Returns
    -------
    np.ndarray
        An array with all the 2**ndim possible combinations of 0s and 1s.
    """
    bits = np.zeros((2**ndim, ndim), dtype=np.int64)
    for i in range(ndim):
        col = (ndim - 1) - i
        for j in range(2**ndim):
            bits[j, col] = (j >> i) & 1
    return bits


@njit
def numerators(
    diffs: np.ndarray, comb_inds: np.ndarray, ndims: int, npoints: int
) -> np.ndarray:  # pragma: no cover
    """
    Finds the numerators of the interpolation weights, which are the areas of the hypercubes
    formed by the points and the grid points that contain them.

    Parameters
    ----------
    diffs : np.ndarray
        A (2, ndim, npoints) that contains the forward and backward differences of point coordinates.
        and the grid points that contain them along every dimension.
    comb_inds : np.ndarray
        An array with all the 2**ndim possible combinations of 0s and 1s (fwd and bwd differences).
    ndims : int
        The number of dimensions.
    npoints : int
        The number of points.

    Returns
    -------
    np.ndarray
        The numerators of the interpolation weights.
    """
    numers = np.ones((2**ndims, npoints), dtype=np.float64)
    for i in range(2**ndims):
        for d, j in enumerate(comb_inds[i]):
            numers[i, :] *= diffs[j, d, :]

    return numers


@njit
def mass_to_grid(
    points: np.ndarray, mass: np.ndarray, grids: List[np.ndarray]
) -> np.ndarray:  # pragma: no cover
    """
    Distributes the mass of a set of R^n points to a rectangular R^n grid,
    following the 'lottery' method.

    Parameters
    ----------
    points : np.ndarray
        shape = (#points, #dims) The points to be distributed.
    mass : np.ndarray
        shape = (#points) The mass of each point.
    grids : List[np.ndarray]
        The grids for each dimension.

    Returns
    -------
    np.ndarray
        The mass of each point in the grid. (flattened).
    """
    dims = np.array([len(g) for g in grids])
    ndims = len(grids)
    npoints = points.shape[0]

    # Trim points to maximum and minimum of grids
    grid_inf_lims = np.expand_dims(np.array([x[0] for x in grids]), 0)
    grid_sup_lims = np.expand_dims(np.array([x[-1] for x in grids]), 0)
    points = np.clip(points, grid_inf_lims, grid_sup_lims)

    # Find lower indices along every dimension
    inds = multidim_get_lower_index(points, grids, dims).T

    # Forward and backward differences
    diffs = fwd_and_bwd_diffs(points, grids, inds)

    # Matrix with combinations of forward and backward differencess
    comb_inds = get_combinations(len(grids))

    # Find denominators
    numers = numerators(diffs, comb_inds, ndims, npoints)
    denoms = denominators(inds, grids)

    # Multiply the ndim differences to find areas
    fact = mass / denoms

    # Weights add up to 1
    weights = numers * np.expand_dims(fact, 0)

    # A (ndim, 2**ndim, npoints) matrix in which [:,:,n] nth row has
    # the indices where we should add weights[:,n]
    add_inds = np.expand_dims(inds, axis=1) + (1 - np.expand_dims(comb_inds.T, -1))

    # Make indices unidimensional (to not do *inds in multidim matrices with numba)
    add_inds = ravel_index(add_inds, dims)
    distr = sum_weights(weights, dims, add_inds)

    return distr

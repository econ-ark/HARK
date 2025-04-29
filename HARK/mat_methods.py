from typing import List
from dataclasses import dataclass
import numpy as np
from numba import njit


# %% Class and methods that facilitates simulating populations in discretized
# state spaces
@dataclass
class DiscreteTransitions:
    living_tmats: list
    surv_probs: list
    life_cycle: bool = False
    newborn_dstn: np.array

    def __post_init__(self):
        if self.life_cycle:
            self.T = len(self.living_tmats) + 1
            if len(self.surv_probs) != (self.T - 1):
                raise ValueError("surv_probs must have length T-1.")
        else:
            self.T = 1

    def iterate_dstn_forward(self, dstn_init):

        if self.life_cycle:
            return _iterate_dstn_forward_lc(
                dstn_init, self.living_tmats, self.surv_probs, self.newborn_dstn
            )
        else:
            return _iterate_dstn_forward_ih(
                dstn_init[0], self.living_tmat[0], self.surv_prob[0], self.newborn_dstn
            )

    def find_conditional_age_dsnt(self, dstn_init):
        if self.life_cycle:
            return _find_conditional_age_dsnt(dstn_init, self.living_tmats)
        else:
            return [dstn_init]

    def find_steady_state_dstn(self, **kwargs):
        if self.life_cycle:
            if kwargs:
                raise ValueError(
                    "kwargs are not used in the life cycle version of find_steady_state_dstn."
                )
            return _find_steady_state_dstn_lc(
                self.surv_probs, self.newborn_dstn, self.living_tmats
            )
        else:
            return [
                _find_steady_state_dstn_ih(
                    self.newborn_dstn,
                    self.living_tmats[0],
                    self.surv_probs[0],
                    **kwargs,
                )
            ]


# Life cycle methods
@njit
def _iterate_dstn_forward_lc(dstn_init, living_tmats, surv_probs, newborn_dstn):
    new_dstn = [newborn_dstn]
    dead_mass = 0.0
    for i, (d0, tmat) in enumerate(zip(dstn_init, living_tmats)):
        new_dstn.append((surv_probs[i] * d0) @ tmat)
        dead_mass += (1.0 - surv_probs[i]) * np.sum(d0)
    dead_mass += np.sum(dstn_init[-1])
    new_dstn[0] *= dead_mass

    return new_dstn


def _find_conditional_age_dsnt(dstn_init, living_tmats):
    dstns = [dstn_init]
    for tmat in living_tmats:
        dstns.append(dstns[-1] @ tmat)
    return dstns


def _find_steady_state_dstn_lc(surv_probs, newborn_dstn, living_tmats):

    ss_age_mass = np.concatenate([np.array([1.0]), np.cumprod(surv_probs)])
    ss_age_mass /= np.sum(ss_age_mass)
    age_dstns = _find_conditional_age_dsnt(newborn_dstn, living_tmats)
    return [age_dstn * age_mass for age_dstn, age_mass in zip(age_dstns, ss_age_mass)]


# Infinite horizon methods
@njit
def _iterate_dstn_forward_ih(dstn_init, living_tmat, surv_prob, newborn_dstn):
    dead_mass = 1.0 - surv_prob
    new_dstn = surv_prob * dstn_init @ living_tmat
    new_dstn += dead_mass * newborn_dstn

    return new_dstn


@njit
def _find_steady_state_dstn_ih(
    newborn_dstn,
    living_tmat,
    surv_prob,
    max_iter=10000,
    tol=1e-10,
    normalize_every=100,
    dstn_init=None,
):
    if dstn_init is None:
        dstn = newborn_dstn
    else:
        dstn = dstn_init
    go = True
    i = 0
    while go:
        new_dstn = _iterate_dstn_forward_ih(dstn, living_tmat, surv_prob, newborn_dstn)
        if np.max(np.abs(new_dstn - dstn)) < tol:
            go = False
        dstn = new_dstn
        i += 1
        if i > max_iter:
            go = False
        # Renormalize every given number of iterations
        if i % normalize_every == 0:
            dstn /= np.sum(dstn)

    # Return as list just for compatibility with LC methods that return
    # a list of age dstns
    return dstn


# %% Methods to distribute mass to a grid


@njit
def ravel_index(ind_mat: np.ndarray, dims: np.ndarray) -> np.ndarray:
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
) -> np.ndarray:
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
) -> np.ndarray:
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
) -> np.ndarray:
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
def denominators(inds: np.ndarray, grids: List[np.ndarray]) -> np.ndarray:
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
def get_combinations(ndim: int) -> np.ndarray:
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
) -> np.ndarray:
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
) -> np.ndarray:
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

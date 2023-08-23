import numpy as np
from numba import njit
from typing import List


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


class transition_mat:
    def __init__(
        self,
        living_transitions: list,
        surv_probs: list,
        newborn_dstn: np.ndarray,
        life_cycle: bool,
    ) -> None:
        self.living_transitions = living_transitions
        self.surv_probs = surv_probs
        self.newborn_dstn = newborn_dstn
        self.life_cycle = life_cycle

        if self.life_cycle:
            assert len(self.living_transitions) == len(
                self.surv_probs
            ), "living_transitions must be a list of length len(surv_probs) + 1 if life_cycle is True"
        else:
            assert (
                len(self.living_transitions) == 1
            ), "living_transitions must be a list of length 1 if life_cycle is False"
            assert (
                len(self.surv_probs) == 1
            ), "surv_probs must be a list of length 1 if life_cycle is False"

        self.T = len(self.living_transitions) + 1

        self.grid_len = self.living_transitions[0].shape[0]

    def get_full_tmat(self):
        if self.life_cycle:
            # Life cycle
            dim = self.T * self.grid_len
            full_mat = np.zeros((dim, dim))
            for k in range(self.T - 1):
                row_init = k * self.grid_len
                row_end = row_init + self.grid_len
                # Living-to-newborn
                full_mat[row_init:row_end, : self.grid_len] += (
                    1 - self.surv_probs[k]
                ) * self.newborn_dstn[np.newaxis, :]
                # Living-to-age+1
                col_init = row_init + self.grid_len
                col_end = col_init + self.grid_len
                full_mat[row_init:row_end, col_init:col_end] += (
                    self.surv_probs[k] * self.living_transitions[k]
                )

            # In at the end of the last age, everyone turns into a newborn
            full_mat[
                (self.T - 1) * self.grid_len :, : self.grid_len
            ] += self.newborn_dstn[np.newaxis, :]

        else:
            # Infinite horizon
            full_mat = (
                self.surv_probs[0] * self.living_transitions[0]
                + (1 - self.surv_probs[0]) * self.newborn_dstn[np.newaxis, :]
            )

        return full_mat

    def post_multiply(self, mat):
        # Check dimension compatibility
        n_rows, n_cols = mat.shape
        if self.life_cycle:
            ncols_fullmat = self.T * self.grid_len
        else:
            ncols_fullmat = self.grid_len

        if n_rows != ncols_fullmat:
            raise Exception(
                "Matrix has {} rows, but should have {}".format(n_rows, ncols_fullmat)
            )

        if self.life_cycle:
            full_mat_dim = self.T * self.grid_len
            prod = np.zeros((full_mat_dim, n_cols))

            for k in range(self.T):
                row_init = k * self.grid_len
                row_end = row_init + self.grid_len
                if k < self.T - 1:
                    sp = self.surv_probs[k]
                else:
                    sp = 0.0

                for j in range(n_cols):
                    # From the newborn dstn
                    prod[row_init:row_end, j] += (1 - sp) * np.dot(
                        self.newborn_dstn[np.newaxis, :], mat[: self.grid_len, j]
                    )
                    if k < self.T - 1:
                        # From the living dstn
                        prod[row_init:row_end, j] += sp * np.dot(
                            self.living_transitions[k],
                            mat[row_end : (row_end + self.grid_len), j],
                        )

            return prod

        else:
            # Infinite horizon
            prod = self.surv_probs[0] * np.dot(self.living_transitions[0], mat)
            prod += (1 - self.surv_probs[0]) * np.dot(
                self.newborn_dstn[np.newaxis, :], mat
            )
            return prod

    def pre_multiply(self, mat):
        # Check dimension compatibility
        n_rows, n_cols = mat.shape
        if self.life_cycle:
            nrows_fullmat = self.T * self.grid_len
        else:
            nrows_fullmat = self.grid_len

        if n_cols != nrows_fullmat:
            raise Exception(
                "Matrix has {} cols, but should have {}".format(n_cols, nrows_fullmat)
            )

        if self.life_cycle:
            full_mat_dim = self.T * self.grid_len
            prod = np.zeros((n_rows, nrows_fullmat))

            for k in range(self.T):
                col_init = k * self.grid_len
                col_end = col_init + self.grid_len
                if k < self.T - 1:
                    sp = self.surv_probs[k]
                else:
                    sp = 0.0

                # Newborns contribute to first block of
                # cols
                nb_mat = np.tile(self.newborn_dstn, (self.grid_len, 1))
                prod[:, : self.grid_len] += (1 - sp) * np.dot(
                    mat[:, col_init:col_end], nb_mat
                )
                # Living contribute to other columns
                if k < self.T - 1:
                    prod[:, col_end : (col_end + self.grid_len)] += sp * np.dot(
                        mat[:, col_init:col_end], self.living_transitions[k]
                    )

            return prod

        else:
            # Infinite horizon
            prod = np.dot(
                mat,
                self.surv_probs[0] * self.living_transitions[0]
                + (1 - self.surv_probs[0]) * self.newborn_dstn[np.newaxis, :],
            )

            return prod

    def iterate_dstn_forward(self, dstn_init: np.ndarray) -> np.ndarray:
        dstn_final = self.pre_multiply(dstn_init.T).T

        return dstn_final

    def find_steady_state_dstn(
        self,
        dstn_init=None,
        tol=1e-10,
        max_iter=1000,
        check_every=10,
        normalize_every=20,
    ):
        if dstn_init is None:
            # Create an initial distribution that concentrates
            # on the first gridpoint of the first age
            dstn_init = dstn = np.zeros(
                (len(self.newborn_dstn), len(self.living_transitions))
            )
            dstn[0, 0] = 1.0

        # Initialize
        dstn = dstn_init
        err = tol + 1
        i = 0
        while err > tol and i < max_iter:
            dstn_new = self.iterate_dstn_forward(dstn)
            if i % normalize_every == 0:
                dstn_new /= np.sum(dstn_new)
            if i % check_every == 0:
                err = np.max(np.abs(dstn_new - dstn))
            dstn = dstn_new
            i += 1

        return dstn

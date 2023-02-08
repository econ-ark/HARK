import numpy as np
from numba import njit, prange, typed
from scipy.ndimage import map_coordinates

from HARK.core import MetricObject

AVAILABLE_TARGETS = ["cpu", "parallel"]

try:
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates as cupy_map_coordinates

    CUPY_AVAILABLE = True
    AVAILABLE_TARGETS.append("gpu")
except ImportError:
    CUPY_AVAILABLE = False


MC_KWARGS = {
    "order": 1,  # order of interpolation
    "mode": "nearest",  # how to handle extrapolation
    "cval": 0.0,  # value to use for extrapolation
    "prefilter": False,  # whether to prefilter input
}


class _RegularGridInterp(MetricObject):
    distance_criteria = ["values"]

    def __init__(self, values, target="cpu"):
        assert target in AVAILABLE_TARGETS, "Invalid target."
        self.target = target

        if target in ["cpu", "parallel"]:
            self.values = np.asarray(values)
        elif target == "gpu":
            self.values = cp.asarray(values)

        self.ndim = self.values.ndim  # should match number of grids
        self.shape = self.values.shape  # should match points in each grid

    def __call__(self, *args):
        if self.target in ["cpu", "parallel"]:
            args = np.asarray(args)
        elif self.target == "gpu":
            args = cp.asarray(args)

        assert (
            args.shape[0] == self.ndim
        ), "Number of arguments must match number of dimensions."

        coordinates = self._get_coordinates(args)
        return self._map_coordinates(coordinates)

    def _get_coordinates(self, args):
        raise NotImplementedError("Must be implemented by subclass.")

    def _map_coordinates(self, coordinates):
        raise NotImplementedError("Must be implemented by subclass.")


class MultivariateInterp(_RegularGridInterp):
    def __init__(self, values, grids, target="cpu", **kwargs):
        super().__init__(values, target=target)

        self.mc_kwargs = MC_KWARGS.copy()
        # update mc_kwargs with any kwargs that are in MC_KWARGS
        self.mc_kwargs.update((k, v) for k, v in kwargs.items() if k in MC_KWARGS)

        if target == "cpu":
            self.grids = [np.asarray(grid) for grid in grids]
        elif target == "parallel":
            self.grids = typed.List()
            [self.grids.append(grid) for grid in grids]
        elif target == "gpu":
            self.grids = [cp.asarray(grid) for grid in grids]

        assert self.ndim == len(
            self.grids
        ), "Number of grids must match number of dimensions."
        assert all(
            self.shape[i] == grid.size for i, grid in enumerate(self.grids)
        ), "Values shape must match points in each grid."

    def _get_coordinates(self, args):
        if self.target == "cpu":
            coordinates = np.empty_like(args)
            for dim, grid in enumerate(self.grids):  # for each dimension
                coordinates[dim] = np.interp(  # x, xp, fp (new x, x points, y values)
                    args[dim], grid, np.arange(self.shape[dim])
                )
        elif self.target == "parallel":
            coordinates = _nb_interp(self.grids, args)
        elif self.target == "gpu":
            coordinates = cp.empty_like(args)
            for dim, grid in enumerate(self.grids):  # for each dimension
                coordinates[dim] = cp.interp(  # x, xp, fp (new x, x points, y values)
                    args[dim], grid, cp.arange(self.shape[dim])
                )

        return coordinates

    def _map_coordinates(self, coordinates):
        if self.target in ["cpu", "parallel"]:
            # there is no parallelization for scipy map_coordinates
            output = map_coordinates(
                self.values, coordinates.reshape(self.ndim, -1), **self.mc_kwargs
            )

        elif self.target == "gpu":
            output = cupy_map_coordinates(
                self.values, coordinates.reshape(self.ndim, -1), **self.mc_kwargs
            )

        return output.reshape(coordinates[0].shape)


@njit(parallel=True, cache=True, fastmath=True)
def _nb_interp(grids, args):
    coordinates = np.empty_like(args)
    for dim in prange(args.shape[0]):
        coordinates[dim] = np.interp(args[dim], grids[dim], np.arange(grids[dim].size))

    return coordinates


class _CurvilinearGridInterp(_RegularGridInterp):
    def __init__(self, values, grids, target="cpu"):
        super().__init__(values, target=target)

        if target in ["cpu", "parallel"]:
            self.grids = np.asarray(grids)
        elif target == "gpu":
            self.grids = cp.asarray(grids)

        assert (
            self.ndim == self.grids[0].ndim
        ), "Number of grids must match number of dimensions."
        assert (
            self.shape == self.grids[0].shape
        ), "Values shape must match points in each grid."


class WarpedInterpOnInterp2D(_CurvilinearGridInterp):
    def __call__(self, *args):
        if self.target in ["cpu", "parallel"]:
            args = np.asarray(args)
        elif self.target == "gpu":
            args = cp.asarray(args)

        assert args.shape[0] == self.ndim, "Wrong number of arguments."

        if self.target == "cpu":
            output = self._target_cpu(args)
        elif self.target == "parallel":
            output = self._target_parallel(args)
        elif self.target == "gpu":
            output = self._target_gpu(args)

        return output

    def _target_cpu(self, args):
        shape = args[0].shape  # original shape of arguments

        # flatten arguments by dimension
        args = args.reshape((self.ndim, -1))

        # interpolate each dimension separately
        # create intermediate arrays to store results
        # the shape of each array is
        #    - the number of points in the first dimension of the values
        #    - the number of points in each dimension of the arguments
        # so, (dimensions, points in argument)
        y_intermed = np.empty((self.shape[0], args[0].size))
        z_intermed = np.empty((self.shape[0], args[0].size))

        for i in range(self.shape[0]):
            # for each dimension, interpolate the first argument
            y_intermed[i] = np.interp(args[0], self.grids[0][:, i], self.grids[1][:, i])
            z_intermed[i] = np.interp(args[0], self.grids[0][:, i], self.values[:, i])

        output = np.empty_like(args[0])

        for j in range(args[0].size):
            output[j] = np.interp(args[1][j], y_intermed[:, j], z_intermed[:, j])

        return output.reshape(shape)

    def _target_parallel(self, args):
        return nb_interp_piecewise(args, self.grids, self.values)

    def _target_gpu(self, args):
        shape = args[0].shape

        args = args.reshape((self.ndim, -1))

        y_intermed = cp.empty((self.shape[0], args[0].size))
        z_intermed = cp.empty((self.shape[0], args[0].size))

        for i in range(self.shape[0]):
            y_intermed[i] = cp.interp(args[0], self.grids[0][:, i], self.grids[1][:, i])
            z_intermed[i] = cp.interp(args[0], self.grids[0][:, i], self.values[:, i])

        output = cp.empty_like(args[0])

        for j in range(args[0].size):
            output[j] = cp.interp(args[1][j], y_intermed[:, j], z_intermed[:, j])

        return output.reshape(shape)


@njit(parallel=True, cache=True, fastmath=True)
def nb_interp_piecewise(args, grids, values):
    shape = args[0].shape

    args = args.reshape((values.ndim, -1))

    y_intermed = np.empty((values.shape[0], args[0].size))
    z_intermed = np.empty((values.shape[0], args[0].size))

    for i in prange(values.shape[0]):
        y_intermed[i] = np.interp(args[0], grids[0][:, i], grids[1][:, i])
        z_intermed[i] = np.interp(args[0], grids[0][:, i], values[:, i])

    output = np.empty_like(args[0])

    for j in prange(args[0].size):
        output[j] = np.interp(args[1][j], y_intermed[:, j], z_intermed[:, j])

    return output.reshape(shape)


class _UnstructuredGridInterp(_CurvilinearGridInterp):
    def __init__(self, values, grids, target="cpu"):
        super().__init__(values, grids, target=target)
        # remove non finite values that might result from
        # sequential endogenous grid method
        condition = np.logical_and.reduce([np.isfinite(grid) for grid in self.grids])
        condition = np.logical_and(condition, np.isfinite(self.values))
        self.values = self.values[condition]
        self.grids = self.grids[:, condition]
        self.ndim = self.grids.shape[0]

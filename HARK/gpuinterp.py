import numpy as np
from scipy.ndimage import map_coordinates

from HARK.core import MetricObject
from numba import njit, prange, typed

try:
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates as cupy_map_coordinates

    cupy_available = True
except ImportError:
    cupy_available = False


DIM_MESSAGE = "Dimension mismatch."


class MultInterp(MetricObject):

    distance_criteria = ["input", "grids"]

    def __init__(
        self,
        input,
        grids,
        order=1,
        mode="nearest",
        cval=0.0,
        prefilter=False,
        target="cpu",
    ):

        available_targets = ["cpu", "parallel"]

        if cupy_available:
            available_targets.append("gpu")

        assert target in available_targets, "Invalid target."

        if target == "cpu" or target == "parallel":
            import numpy as xp
        elif target == "gpu":
            import cupy as xp

        if target == "parallel":
            self.grids = typed.List()
            [self.grids.append(grid) for grid in grids]
        else:
            self.grids = [xp.asarray(grid) for grid in grids]

        self.input = xp.asarray(input)
        self.order = order
        self.mode = mode
        self.cval = cval
        self.prefilter = prefilter
        self.target = target

        self.ndim = input.ndim  # should match number of grids
        self.shape = input.shape  # should match points in each grid

        assert self.ndim == len(self.grids), DIM_MESSAGE
        for i in range(self.ndim):
            assert self.shape[i] == self.grids[i].size, DIM_MESSAGE

    def __call__(self, *args):

        if self.target == "cpu" or self.target == "parallel":
            import numpy as xp
        elif self.target == "gpu":
            import cupy as xp

        args = xp.asarray(args)
        assert args.shape[0] == self.ndim, DIM_MESSAGE

        coordinates = xp.empty_like(args)

        if self.target == "cpu":
            output = self._target_cpu(args, coordinates)
        elif self.target == "parallel":
            output = self._target_parallel(args, coordinates)
        elif self.target == "gpu":
            output = self._target_gpu(args, coordinates)

        return output

    def _map_coordinates(self, args, coordinates):

        output = map_coordinates(
            self.input,
            coordinates.reshape(args.shape[0], -1),
            order=self.order,
            mode=self.mode,
            cval=self.cval,
            prefilter=self.prefilter,
        ).reshape(args[0].shape)

        return output

    def _target_cpu(self, args, coordinates):

        for dim in range(args.shape[0]):
            arg_grid = self.grids[dim]
            new_args = args[dim]
            coordinates[dim] = np.interp(new_args, arg_grid, np.arange(arg_grid.size))

        output = self._map_coordinates(args, coordinates)

        return output

    def _target_parallel(self, args, coordinates):

        nb_interp(self.grids, args, coordinates)
        output = self._map_coordinates(args, coordinates)

        return output

    def _target_gpu(self, args, coordinates):

        ndim = args.shape[0]

        for dim in range(ndim):
            arg_grid = self.grids[dim]
            new_args = args[dim]
            coordinates[dim] = cp.interp(new_args, arg_grid, cp.arange(arg_grid.size))

        output = cupy_map_coordinates(
            self.input,
            coordinates.reshape(ndim, -1),
            order=self.order,
            mode=self.mode,
            cval=self.cval,
            prefilter=self.prefilter,
        ).reshape(args[0].shape)

        return output


@njit(parallel=True, cache=True, fastmath=True)
def nb_interp(grids, args, coordinates):

    for dim in prange(args.shape[0]):
        arg_grid = grids[dim]
        new_args = args[dim]
        coordinates[dim] = np.interp(new_args, arg_grid, np.arange(arg_grid.size))

    return coordinates

import numpy as np

from HARK.core import MetricObject
from HARK.interpolation import LinearInterp
from numba import njit, prange, typed

try:
    import cupy as cp

    cupy_available = True
except ImportError:
    cupy_available = False

DIM_MESSAGE = "Dimension mismatch."


class PiecewiseAffineInterp2D(MetricObject):
    def __init__(self, values, grids):

        self.values = np.asarray(values)
        self.grids = np.asarray(grids)

        x_vals = self.grids[0]
        y_vals = self.grids[1]
        z_vals = self.values

        self.ndim = 2
        self.shape = self.values.shape

        y_interpolators = []
        z_interpolators = []

        for i in range(self.shape[0]):
            y_interpolators.append(LinearInterp(x_vals[:, i], y_vals[:, i]))
            z_interpolators.append(LinearInterp(x_vals[:, i], z_vals[:, i]))

        self.y_interpolators = y_interpolators
        self.z_interpolators = z_interpolators

    def __call__(self, *args):

        args = np.asarray(args)
        shape = args[0].shape

        x_vals = args[0].ravel()
        y_vals = args[1].ravel()

        z_intermed = []
        y_intermed = []

        for i in range(self.shape[0]):
            y_intermed.append(self.y_interpolators[i](x_vals))
            z_intermed.append(self.z_interpolators[i](x_vals))

        y_intermed = np.asarray(y_intermed)
        z_intermed = np.asarray(z_intermed)

        output = np.empty_like(x_vals)

        for j in range(x_vals.size):
            interp = LinearInterp(y_intermed[:, j], z_intermed[:, j])
            output[j] = interp(y_vals[j])

        return output.reshape(shape)


class PiecewiseAffineInterp2DFast(MetricObject):
    def __init__(self, values: np.ndarray, grids, target="cpu"):

        available_targets = ["cpu", "parallel"]

        if cupy_available:
            available_targets.append("gpu")

        assert target in available_targets, "Invalid target."

        if target == "cpu" or target == "parallel":
            import numpy as xp
        elif target == "gpu":
            import cupy as xp

        self.values = xp.asarray(values)
        self.grids = xp.asarray(grids)
        self.target = target

        self.ndim = values.ndim
        self.shape = values.shape

    def __call__(self, *args):

        if self.target == "cpu" or self.target == "parallel":
            import numpy as xp
        elif self.target == "gpu":
            import cupy as xp

        args = xp.asarray(args)
        assert args.shape[0] == self.ndim, DIM_MESSAGE

        if self.target == "cpu":
            output = self._target_cpu(args)
        elif self.target == "parallel":
            output = self._target_parallel(args)
        elif self.target == "gpu":
            output = self._target_gpu(args)

        return output

    def _target_cpu(self, args):

        shape = args[0].shape

        args = args.reshape((self.ndim, -1))

        y_intermed = np.empty((self.shape[0], args[0].size))
        z_intermed = np.empty((self.shape[0], args[0].size))

        for i in range(self.shape[0]):
            y_intermed[i] = np.interp(args[0], self.grids[0][:, i], self.grids[1][:, i])
            z_intermed[i] = np.interp(args[0], self.grids[0][:, i], self.values[:, i])

        output = np.empty_like(args[0])

        for j in range(args[0].size):
            output[j] = np.interp(args[1][j], y_intermed[:, j], z_intermed[:, j])

        return output.reshape(shape)

    def _target_parallel(self, args):

        return nb_interp(args, self.grids, self.values)

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


@njit(parallel=True, cache=True)
def nb_interp(args, grids, values):

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

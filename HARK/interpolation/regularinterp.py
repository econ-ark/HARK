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


class MultivariateInterp(MetricObject):

    distance_criteria = ["values", "grids"]

    def __init__(
        self,
        values,
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

        self.values = xp.asarray(values)
        self.order = order
        self.mode = mode
        self.cval = cval
        self.prefilter = prefilter
        self.target = target

        self.ndim = values.ndim  # should match number of grids
        self.shape = values.shape  # should match points in each grid

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
            self.values,
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
            self.values,
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
class RegularizedPolynomialInterp(MetricObject):

    distance_criteria = ["values", "grids"]

    def __init__(self, values, grids, degree):

        assert sklearn_available, "scikit-learn is not installed."

        self.values = np.asarray(values)
        self.grids = np.asarray(grids)
        self.degree = degree

        self.ndim = self.values.ndim
        self.shape = self.values.shape

        self.models = self._set_models()

        self.X_train = np.c_[tuple(grid.ravel() for grid in self.grids)]

        y_train = np.mgrid[[slice(0, dim) for dim in self.shape]]

        self.y_train = np.c_[[y.ravel() for y in y_train]]

        for dim in range(self.ndim):
            self.models[dim].fit(self.X_train, self.y_train[dim])

    def _set_models(self):

        models = [
            make_pipeline(
                StandardScaler(),
                PolynomialFeatures(degree=self.degree),
                RidgeCV(),
            )
            for _ in range(self.ndim)
        ]

        return models

    def __call__(self, *args):

        args = np.asarray(args)

        X_test = np.c_[tuple(arg.ravel() for arg in args)]

        coordinates = np.empty_like(args)

        for dim in range(self.ndim):
            coordinates[dim] = self.models[dim].predict(X_test).reshape(args[0].shape)

        output = map_coordinates(
            self.values,
            coordinates.reshape(args.shape[0], -1),
        ).reshape(args[0].shape)

        return output


class RegularizedSplineInterp(RegularizedPolynomialInterp):
    def __init__(self, values, grids, n_knots, degree):

        self.n_knots = n_knots

        super().__init__(values, grids, degree)

    def _set_models(self):

        models = [
            make_pipeline(
                StandardScaler(),
                SplineTransformer(n_knots=self.n_knots, degree=self.degree),
                RidgeCV(),
            )
            for _ in range(self.ndim)
        ]

        return models

class SKImagePiecewiseAffineInterp(MetricObject):

    distance_criteria = ["values", "grids"]

    def __init__(self, values, grids, target="cpu"):

        if not skimage_available:
            raise ImportError(
                "PiecewiseAffineTransform requires scikit-image installed."
            )

        available_targets = ["cpu"]

        if cupy_available:
            available_targets.append("gpu")

        assert target in available_targets, "Invalid target."

        self.values = np.asarray(values)
        self.grids = np.asarray(grids)

        self.ndim = self.values.ndim

        src = np.vstack([grid.flat for grid in self.grids]).T
        coords = np.mgrid[[slice(0, dim) for dim in self.values.shape]]
        dst = np.vstack([coord.flat for coord in coords]).T

        tform = PiecewiseAffineTransform()
        tform.estimate(src, dst)

        self.tform = tform

    def __call__(self, *args):

        args = np.asarray(args)

        src_new = np.vstack([arg.flat for arg in args]).T
        coordinates = self.tform(src_new).T

        return map_coordinates(self.values, coordinates).reshape(args[0].shape)

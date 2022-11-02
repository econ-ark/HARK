import numpy as np
from HARK.core import MetricObject
from HARK.interpolation.unstructuredinterp import UnstructuredInterp
from numba import njit, prange, typed
from scipy.ndimage import map_coordinates

try:
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates as cupy_map_coordinates

    cupy_available = True
except ImportError:
    cupy_available = False


try:
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import (
        PolynomialFeatures,
        SplineTransformer,
        StandardScaler,
    )

    sklearn_available = True
except ImportError:
    sklearn_available = False

try:
    from skimage.transform import PiecewiseAffineTransform

    skimage_available = True
except ImportError:
    skimage_available = False


DIM_MESSAGE = "Dimension mismatch."

MC_KWARGS = {
    "order": 1,  # order of interpolation
    "mode": "constant",  # how to handle extrapolation
    "cval": 0.0,  # value to use for extrapolation
    "prefilter": False,  # whether to prefilter input
}


class MultivariateInterp(MetricObject):

    distance_criteria = ["values", "grids"]

    def __init__(self, values, grids, target="cpu", mc_kwargs=None):

        available_targets = ["cpu", "parallel"]

        if cupy_available:
            available_targets.append("gpu")

        assert target in available_targets, "Invalid target."

        if mc_kwargs is None:
            mc_kwargs = dict()
        self.mc_kwargs = MC_KWARGS.copy()
        self.mc_kwargs.update((k, mc_kwargs[k]) for k in mc_kwargs if k in MC_KWARGS)

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

        coordinates = self._get_coordinates(args)
        output = self._map_coordinates(coordinates)

        return output

    def _get_coordinates(self, args):

        if self.target == "cpu" or self.target == "parallel":
            import numpy as xp
        elif self.target == "gpu":
            import cupy as xp

        coordinates = xp.empty_like(args)

        if self.target == "cpu" or self.target == "gpu":
            for dim in range(self.ndim):  # for each dimension
                coordinates[dim] = xp.interp(  # x, xp, fp (new x, x points, y values)
                    args[dim], self.grids[dim], xp.arange(self.shape[dim])
                )

        elif self.target == "parallel":
            _nb_interp(self.grids, args, coordinates)

        return coordinates

    def _map_coordinates(self, coordinates):

        if self.target == "cpu" or self.target == "parallel":
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
def _nb_interp(grids, args, coordinates):

    for dim in prange(args.shape[0]):
        coordinates[dim] = np.interp(args[dim], grids[dim], np.arange(grids[dim].size))

    return coordinates


class RegularizedMultivariateInterp(MultivariateInterp):

    distance_criteria = ["values", "grids"]

    def __init__(
        self,
        values,
        grids,
        target="cpu",
        mc_kwargs=None,
        ui_kwargs=None,
    ):

        if mc_kwargs is None:
            mc_kwargs = dict()
        self.mc_kwargs = MC_KWARGS.copy()
        self.mc_kwargs.update((k, mc_kwargs[k]) for k in mc_kwargs if k in MC_KWARGS)

        self.values = np.asarray(values)
        self.grids = np.asarray(grids)
        self.target = target

        self.ndim = self.values.ndim
        self.shape = self.values.shape

        assert self.ndim == self.grids.shape[0], DIM_MESSAGE
        assert self.shape == self.grids[0].shape, DIM_MESSAGE

        # mesh of coordinates for each dimension
        coord_mesh = np.mgrid[[slice(0, dim) for dim in self.shape]]

        # densified grid and mesh of all points in all dimensions
        # this could be refined to include fewer points if it gets too big
        dense_grid = [np.unique(grid) for grid in self.grids]
        dense_mesh = np.meshgrid(*dense_grid, indexing="ij")

        # interpolator for each dimension, from grid space to coordinate space
        coord_interp = [
            UnstructuredInterp(coord_mesh[i], self.grids, interp_kwargs=ui_kwargs)
            for i in range(self.ndim)
        ]

        # coordinates for all points in dense grid
        input_coords = [coord_interp[i](*dense_mesh) for i in range(self.ndim)]

        self.coord_interp = [
            MultivariateInterp(
                input_coords[i],
                dense_grid,
                target=target,
                mc_kwargs=mc_kwargs,
            )
            for i in range(self.ndim)
        ]

    def _get_coordinates(self, args):

        coordinates = np.asarray(
            [self.coord_interp[i](*args) for i in range(self.ndim)]
        )

        return coordinates


class RegularizedPolynomialInterp(MetricObject):

    distance_criteria = ["values", "grids"]

    def __init__(self, values, grids, degree, normalize=True):

        assert sklearn_available, "scikit-learn is not installed."

        self.values = np.asarray(values)
        self.grids = np.asarray(grids)
        self.degree = degree

        self.ndim = self.values.ndim
        self.shape = self.values.shape

        assert self.ndim == self.grids.shape[0], DIM_MESSAGE
        assert self.shape == self.grids[0].shape, DIM_MESSAGE

        self.X_train = np.c_[tuple(grid.ravel() for grid in self.grids)]
        y_train = np.mgrid[[slice(0, dim) for dim in self.shape]]
        self.y_train = np.c_[[y.ravel() for y in y_train]]

        self.models = self._set_models(normalize=normalize)
        for dim in range(self.ndim):
            self.models[dim].fit(self.X_train, self.y_train[dim])

    def _set_models(self, normalize=True):

        if normalize:
            pipeline = [StandardScaler()]
        else:
            pipeline = []

        pipeline += [
            PolynomialFeatures(degree=self.degree),
            RidgeCV(alphas=np.logspace(-6, 6, 13)),
        ]

        return [make_pipeline(pipeline) for _ in range(self.ndim)]

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
    def __init__(self, values, grids, n_knots, degree, normalize=True):

        self.n_knots = n_knots

        super().__init__(values, grids, degree)

    def _set_models(self, normalize=True):

        if normalize:
            pipeline = [StandardScaler()]
        else:
            pipeline = []

        pipeline += [
            SplineTransformer(n_knots=self.n_knots, degree=self.degree),
            RidgeCV(alphas=np.logspace(-6, 6, 13)),
        ]

        return [make_pipeline(pipeline) for _ in range(self.ndim)]


class SKImagePiecewiseAffineInterp(MetricObject):

    distance_criteria = ["values", "grids"]

    def __init__(self, values, grids):

        if not skimage_available:
            raise ImportError(
                "PiecewiseAffineTransform requires scikit-image installed."
            )

        self.values = np.asarray(values)
        self.grids = np.asarray(grids)

        self.ndim = self.values.ndim
        self.shape = self.values.shape

        assert self.ndim == self.grids.shape[0], DIM_MESSAGE
        assert self.shape == self.grids[0].shape, DIM_MESSAGE

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


class WarpedInterpOnInterp2D(MetricObject):

    distance_criteria = ["values", "grids"]

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


@njit(parallel=True, cache=True)
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

import numpy as np
from scipy.ndimage import map_coordinates

from HARK.core import MetricObject


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

DIM_MESSAGE = "Dimension mismatch."


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

    def _get_coordinates(self, args):

        coordinates = np.empty_like(args)

        X_test = np.c_[tuple(arg.ravel() for arg in args)]

        for dim in range(self.ndim):
            coordinates[dim] = self.models[dim].predict(X_test).reshape(args[0].shape)

    def _map_coordinates(self, coordinates):

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


class UnstructuredPolynomialInterp(MetricObject):

    distance_criteria = ["values", "grids"]

    def __init__(self, values, grids, degree):

        self.values = np.asarray(values)
        self.grids = np.asarray(grids)
        self.degree = degree

        self.ndim = self.values.ndim
        self.shape = self.values.shape

        self.model = self._set_model()

        self.X_train = np.c_[tuple(grid.ravel() for grid in self.grids)]

        self.y_train = self.values.ravel()

        self.model.fit(self.X_train, self.y_train)

    def _set_model(self):

        return make_pipeline(
            StandardScaler(),
            PolynomialFeatures(self.degree),
            RidgeCV(alphas=np.logspace(-6, 6, 13)),
        )

    def __call__(self, *args):

        args = np.asarray(args)

        X_test = np.c_[tuple(arg.ravel() for arg in args)]

        return self.model.predict(X_test).reshape(args[0].shape)


class UnstructuredSplineInterp(UnstructuredPolynomialInterp):
    def __init__(self, values, grids, degree, n_knots=10):

        self.n_knots = n_knots

        super().__init__(values, grids, degree)

    def _set_model(self):

        return make_pipeline(
            StandardScaler(),
            SplineTransformer(n_knots=self.n_knots, degree=self.degree),
            RidgeCV(alphas=np.logspace(-6, 6, 13)),
        )

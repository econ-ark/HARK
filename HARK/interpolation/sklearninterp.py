import numpy as np
from scipy.ndimage import map_coordinates
from sklearn.linear_model import RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    Normalizer,
    PolynomialFeatures,
    SplineTransformer,
    StandardScaler,
)

from HARK.core import MetricObject

DIM_MESSAGE = "Dimension mismatch."


class _SKLearnRegressionInterp(MetricObject):

    standardize = True
    normalize = True
    distance_criteria = ["values", "grids"]

    def __init__(self, values, grids, standardize=False, normalize=False):

        self.values = np.asarray(values)
        self.grids = np.asarray(grids)

        self.standardize = standardize
        self.normalize = normalize

        self.ndim = self.values.ndim
        self.shape = self.values.shape

        assert self.ndim == self.grids.shape[0], DIM_MESSAGE
        assert self.shape == self.grids[0].shape, DIM_MESSAGE

        self.X_train = np.c_[tuple(grid.ravel() for grid in self.grids)]
        y_train = np.mgrid[[slice(0, dim) for dim in self.shape]]
        self.y_train = np.c_[[y.ravel() for y in y_train]]

        pipeline = self._set_models()

        self.models = [make_pipeline(*pipeline) for _ in range(self.ndim)]
        for dim in range(self.ndim):
            self.models[dim].fit(self.X_train, self.y_train[dim])

    def _set_models(self):

        pipeline = []

        if self.standardize:
            pipeline += [StandardScaler()]
        if self.normalize:
            pipeline += [Normalizer()]

        return pipeline

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
            coordinates.reshape(coordinates.shape[0], -1),
        ).reshape(coordinates[0].shape)

        return output


class _SKLearnFeaturesInterp(_SKLearnRegressionInterp):
    def __init__(
        self,
        values,
        grids,
        degree=3,
        n_knots=5,
        polynomial=False,
        spline=False,
        standardize=False,
        normalize=False,
    ):

        self.polynomial = polynomial
        self.spline = spline

        self.degree = degree
        self.n_knots = n_knots

        super().__init__(values, grids, standardize=standardize, normalize=normalize)

    def _set_models(self):

        pipeline = super()._set_models()

        if self.polynomial:
            pipeline += [PolynomialFeatures(degree=self.degree)]
        elif self.spline:
            pipeline += [SplineTransformer(n_knots=self.n_knots, degree=self.degree)]

        return pipeline


class RegularizedPolynomialInterp(_SKLearnRegressionInterp):
    def __init__(self, values, grids, degree, standardize=False, normalize=False):

        self.degree = degree

        super().__init__(values, grids, standardize=standardize, normalize=normalize)

    def _set_models(self):

        pipeline = super()._set_models()

        pipeline += [
            PolynomialFeatures(degree=self.degree),
            KernelRidge(),
        ]

        return pipeline


class RegularizedSplineInterp(_SKLearnRegressionInterp):
    def __init__(
        self, values, grids, n_knots, degree, standardize=False, normalize=False
    ):
        self.degree = degree
        self.n_knots = n_knots

        super().__init__(values, grids, standardize=standardize, normalize=normalize)

    def _set_models(self):

        pipeline = super()._set_models()

        pipeline += [
            SplineTransformer(n_knots=self.n_knots, degree=self.degree),
            KernelRidge(),
        ]

        return pipeline


class KernelRidgeInterp(_SKLearnRegressionInterp):
    def _set_models(self):

        pipeline = super()._set_models()

        pipeline += [KernelRidge()]

        return pipeline


class GaussianProcessInterp(_SKLearnRegressionInterp):
    def _set_models(self):
        pipeline = super()._set_models()

        pipeline += [GaussianProcessRegressor()]

        return pipeline


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

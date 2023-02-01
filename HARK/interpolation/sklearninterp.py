import numpy as np
from scipy.ndimage import map_coordinates
from sklearn.linear_model import ElasticNet, ElasticNetCV, SGDRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    Normalizer,  # scaling individual samples to have unit norm
    PolynomialFeatures,  # features’ high-order and interaction terms
    SplineTransformer,  # B-spline basis functions
    StandardScaler,  # mean removal and variance scaling
)

from HARK.core import MetricObject

DIM_MESSAGE = "Dimension mismatch."


class _CurvilinearInterp(MetricObject):

    distance_criteria = ["values", "grids"]

    def __init__(self, values, grids):

        self.values = np.asarray(values)
        self.grids = np.asarray(grids)

        self.ndim = self.values.ndim
        self.shape = self.values.shape

        assert self.ndim == self.grids.shape[0], DIM_MESSAGE
        assert self.shape == self.grids[0].shape, DIM_MESSAGE

    def __call__(self, *args):

        coordinates = self._get_coordinates(np.asarray(args))
        return self._map_coordinates(coordinates)

    def _get_coordinates(self, args):
        raise NotImplementedError("Must be implemented by subclass.")

    def _map_coordinates(self, coordinates):
        raise NotImplementedError("Must be implemented by subclass.")


class PipelineCurvilinearInterp(_CurvilinearInterp):
    def __init__(self, values, grids, pipeline):

        super().__init__(values, grids)

        self.pipeline = pipeline

        self.X_train = np.c_[tuple(grid.ravel() for grid in self.grids)]
        y_train = np.mgrid[[slice(0, dim) for dim in self.shape]]
        self.y_train = np.c_[[y.ravel() for y in y_train]]

        self.models = [make_pipeline(*pipeline) for _ in range(self.ndim)]
        for dim in range(self.ndim):
            self.models[dim].fit(self.X_train, self.y_train[dim])

    def _get_coordinates(self, args):

        X_test = np.c_[tuple(arg.ravel() for arg in args)]
        return np.array([m.predict(X_test).reshape(args[0].shape) for m in self.models])

    def _map_coordinates(self, coordinates):

        return np.reshape(
            map_coordinates(self.values, coordinates.reshape(coordinates.shape[0], -1)),
            coordinates[0].shape,
        )


class _PreprocessingCurvilinearInterp(PipelineCurvilinearInterp):
    def __init__(
        self,
        values,
        grids,
        pipeline,
        std=False,
        norm=False,
        feature=None,
        degree=3,
        n_knots=5,
    ):

        self.std = std
        self.norm = norm
        self.feature = feature
        self.degree = degree
        self.n_knots = n_knots

        if feature is None:
            pass
        elif isinstance(feature, str):
            assert isinstance(degree, int), "Degree must be an integer."
            if feature.startswith("pol"):
                pipeline = [PolynomialFeatures(degree=degree)] + pipeline
            elif feature.startswith("spl"):
                assert isinstance(n_knots, int), "n_knots must be an integer."
                pipeline = [
                    SplineTransformer(n_knots=n_knots, degree=degree)
                ] + pipeline
            else:
                raise AttributeError(f"Feature {feature} not recognized.")
        else:
            raise AttributeError(f"Feature {feature} not recognized.")

        if std:
            pipeline = [StandardScaler()] + pipeline
        if norm:
            pipeline = [Normalizer()] + pipeline

        super().__init__(values, grids, pipeline)


class GeneralizedRegressionCurvilinearInterp(_PreprocessingCurvilinearInterp):
    def __init__(self, values, grids, model="elastic-net", model_kwargs=None, **kwargs):

        if model_kwargs is None:
            model_kwargs = {}

        self.model = model
        self.model_kwargs = model_kwargs

        if model == "elastic-net":
            pipeline = [ElasticNet(**model_kwargs)]
        elif model == "elastic-net-cv":
            pipeline = [ElasticNetCV(**model_kwargs)]
        elif model == "kernel-ridge":
            pipeline = [KernelRidge(**model_kwargs)]
        elif model == "svr":
            pipeline = [SVR(**model_kwargs)]
        elif model == "sgd":
            pipeline = [SGDRegressor(**model_kwargs)]
        elif model == "gaussian-process":
            pipeline = [GaussianProcessRegressor(**model_kwargs)]
        else:
            raise AttributeError(f"Model {model} not implemented.")

        super().__init__(values, grids, pipeline, **kwargs)


class ElasticNetCurvilinearInterp(_PreprocessingCurvilinearInterp):
    def __init__(self, values, grids, model_kwargs=None, **kwargs):

        if model_kwargs is None:
            model_kwargs = {}

        self.model_kwargs = model_kwargs

        pipeline = [ElasticNet(**model_kwargs)]

        super().__init__(values, grids, pipeline, **kwargs)


class ElasticNetCVCurvilinearInterp(_PreprocessingCurvilinearInterp):
    def __init__(self, values, grids, model_kwargs=None, **kwargs):

        if model_kwargs is None:
            model_kwargs = {}

        self.model_kwargs = model_kwargs

        pipeline = [ElasticNetCV(**model_kwargs)]

        super().__init__(values, grids, pipeline, **kwargs)


class KernelRidgeCurvilinearInterp(_PreprocessingCurvilinearInterp):
    def __init__(self, values, grids, model_kwargs=None, **kwargs):

        if model_kwargs is None:
            model_kwargs = {}

        self.model_kwargs = model_kwargs

        pipeline = [KernelRidge(**model_kwargs)]

        super().__init__(values, grids, pipeline, **kwargs)


class SVRCurvilinearInterp(_PreprocessingCurvilinearInterp):
    def __init__(self, values, grids, model_kwargs=None, **kwargs):

        if model_kwargs is None:
            model_kwargs = {}

        self.model_kwargs = model_kwargs

        pipeline = [SVR(**model_kwargs)]

        super().__init__(values, grids, pipeline, **kwargs)


class SGDCurvilinearInterp(_PreprocessingCurvilinearInterp):
    def __init__(self, values, grids, model_kwargs=None, **kwargs):

        if model_kwargs is None:
            model_kwargs = {}

        self.model_kwargs = model_kwargs

        pipeline = [SGDRegressor(**model_kwargs)]

        super().__init__(values, grids, pipeline, **kwargs)


class GaussianProcessCurvilinearInterp(_PreprocessingCurvilinearInterp):
    def __init__(self, values, grids, model_kwargs=None, **kwargs):

        if model_kwargs is None:
            model_kwargs = {}

        self.model_kwargs = model_kwargs

        pipeline = [GaussianProcessRegressor(**model_kwargs)]

        super().__init__(values, grids, pipeline, **kwargs)


class _UnstructuredInterp(MetricObject):

    distance_criteria = ["values", "grids"]

    def __init__(self, values, grids):

        values = np.asarray(values)
        grids = np.asarray(grids)

        # remove non finite values that might result from
        # sequential endogenous grid method
        condition = np.logical_and.reduce([np.isfinite(grid) for grid in grids])
        condition = np.logical_and(condition, np.isfinite(values))
        self.values = values[condition]
        self.grids = np.moveaxis(grids[:, condition], -1, 0)
        self.ndim = self.grids.shape[-1]

        self.X_train = self.grids
        self.y_train = self.values


class PipelineUnstructuredInterp(_UnstructuredInterp):
    def __init__(self, values, grids, pipeline):
        super().__init__(values, grids)
        self.pipeline = pipeline
        self.model = make_pipeline(*self.pipeline)
        self.model.fit(self.X_train, self.y_train)

    def __call__(self, *args):
        args = np.asarray(args)

        X_test = np.c_[tuple(arg.ravel() for arg in args)]

        return self.model.predict(X_test).reshape(args[0].shape)


class _PreprocessingUnstructuredInterp(PipelineUnstructuredInterp):
    def __init__(
        self,
        values,
        grids,
        pipeline,
        std=False,
        norm=False,
        feature=None,
        degree=3,
        n_knots=5,
    ):

        self.std = std
        self.norm = norm
        self.feature = feature
        self.degree = degree
        self.n_knots = n_knots

        if feature is None:
            pass
        elif isinstance(feature, str):
            assert isinstance(degree, int), "Degree must be an integer."
            if feature.startswith("pol"):
                pipeline = [PolynomialFeatures(degree=degree)] + pipeline
            elif feature.startswith("spl"):
                assert isinstance(n_knots, int), "n_knots must be an integer."
                pipeline = [
                    SplineTransformer(n_knots=n_knots, degree=degree)
                ] + pipeline
            else:
                raise AttributeError(f"Feature {feature} not recognized.")
        else:
            raise AttributeError(f"Feature {feature} not recognized.")

        if std:
            pipeline = [StandardScaler()] + pipeline
        if norm:
            pipeline = [Normalizer()] + pipeline

        super().__init__(values, grids, pipeline)


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

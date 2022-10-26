import numpy as np
from HARK.core import MetricObject
from scipy.interpolate import (CloughTocher2DInterpolator,
                               LinearNDInterpolator, NearestNDInterpolator)
from scipy.ndimage import map_coordinates

try:
    import cupy as cp

    cupy_available = True
except ImportError:
    cupy_available = False

try:
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import (PolynomialFeatures, SplineTransformer,
                                       StandardScaler)

    sklearn_available = True
except ImportError:
    sklearn_available = False



class UnstructuredInterp(MetricObject):

    distance_criteria = ["values", "grids"]

    def __init__(
        self,
        values,
        grids,
        method="linear",
        rescale=False,
        fill_value=np.nan,
        # CloughTocher2DInterpolator options
        tol=1e-6,
        maxiter=400,
        # NearestNDInterpolator options
        tree_options=None,
    ):

        values = np.asarray(values)
        grids = np.asarray(grids)

        # remove non finite values that might result from
        # sequential endogenous grid method
        condition = np.logical_and.reduce([np.isfinite(grid) for grid in grids])
        condition = np.logical_and(condition, np.isfinite(values))

        self.values = values[condition]
        self.grids = grids[:, condition].T

        self.method = method
        self.rescale = rescale
        self.fill_value = fill_value
        self.tol = tol
        self.maxiter = maxiter
        self.tree_options = tree_options

        self.ndim = self.grids.shape[-1]

        assert self.ndim == values.ndim, "Dimension mismatch."

        if method == "nearest":
            interpolator = NearestNDInterpolator(
                self.grids, self.values, rescale=rescale, tree_options=tree_options
            )
        elif method == "linear":
            interpolator = LinearNDInterpolator(
                self.grids, self.values, fill_value=fill_value, rescale=rescale
            )
        elif method == "cubic" and self.ndim == 2:
            interpolator = CloughTocher2DInterpolator(
                self.grids,
                self.values,
                fill_value=fill_value,
                tol=tol,
                maxiter=maxiter,
                rescale=rescale,
            )
        else:
            raise ValueError(
                "Unknown interpolation method %r for "
                "%d dimensional data" % (method, self.ndim)
            )

        self.interpolator = interpolator

    def __call__(self, *args):

        return self.interpolator(*args)




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
            RidgeCV(),
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
            RidgeCV(),
        )

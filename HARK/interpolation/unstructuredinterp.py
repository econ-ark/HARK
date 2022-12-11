import numpy as np
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    NearestNDInterpolator,
)

from HARK.core import MetricObject

try:
    import cupy as cp

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

LNDI_KWARGS = {"fill_value": np.nan, "rescale": False}  # linear
NNDI_KWARGS = {"rescale": False, "tree_options": None}  # nearest
CT2DI_KWARGS = {  # cubic
    "fill_value": np.nan,
    "tol": 1e-06,
    "maxiter": 400,
    "rescale": False,
}
RBFI_KWARGS = {  # rbf (radial basis function)
    "neighbors": None,
    "smoothing": 0.0,
    "kernel": "thin_plate_spline",
    "epsilon": None,
    "degree": None,
}


class UnstructuredInterp(MetricObject):

    distance_criteria = ["values", "grids"]

    def __init__(
        self,
        values,
        grids,
        method="linear",
        interp_kwargs=None,
    ):

        if interp_kwargs is None:
            interp_kwargs = dict()

        values = np.asarray(values)
        grids = np.asarray(grids)

        # remove non finite values that might result from
        # sequential endogenous grid method
        condition = np.logical_and.reduce([np.isfinite(grid) for grid in grids])
        condition = np.logical_and(condition, np.isfinite(values))

        self.values = values[condition]
        self.grids = grids[:, condition].T
        self.method = method

        self.ndim = self.grids.shape[-1]

        # assert self.ndim == values.ndim, "Dimension mismatch."

        if method == "nearest":
            self.interp_kwargs = NNDI_KWARGS.copy()
            self.interp_kwargs.update(
                (k, interp_kwargs[k]) for k in interp_kwargs if k in NNDI_KWARGS
            )
            interpolator = NearestNDInterpolator
        elif method == "linear":
            self.interp_kwargs = LNDI_KWARGS.copy()
            self.interp_kwargs.update(
                (k, interp_kwargs[k]) for k in interp_kwargs if k in LNDI_KWARGS
            )
            interpolator = LinearNDInterpolator
        elif method == "cubic" and self.ndim == 2:
            self.interp_kwargs = CT2DI_KWARGS.copy()
            self.interp_kwargs.update(
                (k, interp_kwargs[k]) for k in interp_kwargs if k in CT2DI_KWARGS
            )
            interpolator = CloughTocher2DInterpolator
        else:
            raise ValueError(
                "Unknown interpolation method %r for "
                "%d dimensional data" % (method, self.ndim)
            )

        self.interpolator = interpolator(self.grids, self.values, **self.interp_kwargs)

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

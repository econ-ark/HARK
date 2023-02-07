import numpy as np
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    NearestNDInterpolator,
    RBFInterpolator,
)

from HARK.core import MetricObject

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
        **kwargs,
    ):

        if kwargs is None:
            kwargs = {}

        values = np.asarray(values)
        grids = np.asarray(grids)

        # remove non finite values that might result from
        # sequential endogenous grid method
        condition = np.logical_and.reduce([np.isfinite(grid) for grid in grids])
        condition = np.logical_and(condition, np.isfinite(values))
        self.values = values[condition]
        self.grids = np.moveaxis(grids[:, condition], -1, 0)
        self.method = method
        self.ndim = self.grids.shape[-1]

        # assert self.ndim == values.ndim, "Dimension mismatch."

        interpolator_mapping = {
            "nearest": (NNDI_KWARGS, NearestNDInterpolator),
            "linear": (LNDI_KWARGS, LinearNDInterpolator),
            "cubic": (CT2DI_KWARGS, CloughTocher2DInterpolator)
            if self.ndim == 2
            else (None, None),
            "rbf": (RBFI_KWARGS, RBFInterpolator),
        }

        self.kwargs, interpolator_class = interpolator_mapping.get(method, (None, None))

        if not self.kwargs:
            raise ValueError(
                f"Unknown interpolation method {method} for {self.ndim} dimensional data"
            )

        self.kwargs = self.kwargs.copy()
        self.kwargs.update((k, kwargs[k]) for k in kwargs if k in self.kwargs)
        self.interpolator = interpolator_class(self.grids, self.values, **self.kwargs)

    def __call__(self, *args):

        if self.method == "rbf":
            coords = np.asarray(args).reshape(self.ndim, -1).T
            return self.interpolator(coords).reshape(args[0].shape)

        return self.interpolator(*args)

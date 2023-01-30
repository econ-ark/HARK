import numpy as np
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    NearestNDInterpolator,
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

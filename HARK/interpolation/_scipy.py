import numpy as np
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    NearestNDInterpolator,
    RBFInterpolator,
)

from HARK.interpolation._multi import _UnstructuredGridInterp

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
    "kernel": "linear",
    "epsilon": None,
    "degree": None,
}

AVAILABLE_METHODS = ["nearest", "linear", "cubic", "rbf"]


class UnstructuredInterp(_UnstructuredGridInterp):
    distance_criteria = ["values", "grids"]

    def __init__(
        self,
        values,
        grids,
        method="linear",
        **kwargs,
    ):
        super().__init__(values, grids, target="cpu")

        assert method in AVAILABLE_METHODS, "Invalid interpolation method."
        self.method = method

        interpolator_mapping = {
            "nearest": (NNDI_KWARGS, NearestNDInterpolator),
            "linear": (LNDI_KWARGS, LinearNDInterpolator),
            "cubic": (CT2DI_KWARGS, CloughTocher2DInterpolator)
            if self.ndim == 2
            else (None, None),
            "rbf": (RBFI_KWARGS, RBFInterpolator),
        }

        interp_kwargs, interpolator_class = interpolator_mapping.get(
            method, (None, None)
        )

        if not interp_kwargs:
            raise ValueError(
                f"Unknown interpolation method {method} for {self.ndim} dimensional data"
            )

        self.interp_kwargs = interp_kwargs.copy()
        self.interp_kwargs.update(
            (k, v) for k, v in kwargs.items() if k in interp_kwargs
        )
        self.interpolator = interpolator_class(
            np.moveaxis(self.grids, -1, 0), self.values, **self.interp_kwargs
        )

    def __call__(self, *args):
        if self.method == "rbf":
            coords = np.asarray(args).reshape(self.ndim, -1).T
            return self.interpolator(coords).reshape(args[0].shape)

        return self.interpolator(*args)

import numpy as np
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    NearestNDInterpolator,
)
from scipy.ndimage import map_coordinates

from HARK.core import MetricObject

try:
    from skimage.transform import PiecewiseAffineTransform

    skimage_available = True
except ImportError:
    skimage_available = False

try:
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates as cupy_map_coordinates

    cupy_available = True
except ImportError:
    cupy_available = False


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


class PiecewiseAffineInterp(MetricObject):

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

        self.ndim = values.ndim

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

import numpy as np
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    NearestNDInterpolator,
)
from scipy.ndimage import map_coordinates
from skimage.transform import PiecewiseAffineTransform

from HARK.core import MetricObject


class UnstrucInterp(MetricObject):

    distance_criteria = ["input", "grids"]

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

        ndim = self.grids.shape[-1]

        if method == "nearest":
            interp = NearestNDInterpolator(
                self.grids, self.values, rescale=rescale, tree_options=tree_options
            )
        elif method == "linear":
            interp = LinearNDInterpolator(
                self.grids, self.values, fill_value=fill_value, rescale=rescale
            )
        elif method == "cubic" and ndim == 2:
            interp = CloughTocher2DInterpolator(
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
                "%d dimensional data" % (method, ndim)
            )

        self.interp = interp

    def __call__(self, *args):

        return self.interp(*args)


class PieceWiseAffineInterp(MetricObject):
    def __init__(self, input, grids):
        self.input = np.asarray(input)
        self.grids = np.asarray(grids)

        src = np.vstack([grid.flat for grid in self.grids]).T
        coords = np.mgrid[[slice(0, dim) for dim in self.input.shape]]
        dst = np.vstack([coord.flat for coord in coords]).T

        tform = PiecewiseAffineTransform()
        tform.estimate(src, dst)

        self.tform = tform

    def __call__(self, *args):

        args = np.asarray(args)

        src_new = np.vstack([arg.flat for arg in args]).T
        output = self.tform(src_new)

        x_out = output.reshape((*args[0].shape, 2))

        return map_coordinates(self.input, x_out)

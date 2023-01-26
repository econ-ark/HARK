import numpy as np
from scipy.ndimage import map_coordinates

from HARK.core import MetricObject

try:
    from skimage.transform import PiecewiseAffineTransform

    skimage_available = True
except ImportError:
    skimage_available = False

DIM_MESSAGE = "Dimension mismatch."


class SKImagePiecewiseAffineInterp(MetricObject):

    distance_criteria = ["values", "grids"]

    def __init__(self, values, grids):

        if not skimage_available:
            raise ImportError(
                "PiecewiseAffineTransform requires scikit-image installed."
            )

        self.values = np.asarray(values)
        self.grids = np.asarray(grids)

        self.ndim = self.values.ndim
        self.shape = self.values.shape

        assert self.ndim == self.grids.shape[0], DIM_MESSAGE
        assert self.shape == self.grids[0].shape, DIM_MESSAGE

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

    def _get_coordinates(self, args):

        src_new = np.vstack([arg.flat for arg in args]).T
        coordinates = self.tform(src_new).T

        return coordinates

    def _map_coordinates(self, coordinates):

        return map_coordinates(self.values, coordinates).reshape(args[0].shape)

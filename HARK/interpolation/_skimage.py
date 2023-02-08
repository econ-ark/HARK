import numpy as np
from scipy.ndimage import map_coordinates

from HARK.interpolation._multi import MC_KWARGS, _CurvilinearGridInterp

try:
    from skimage.transform import PiecewiseAffineTransform

    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

DIM_MESSAGE = "Dimension mismatch."


class PiecewiseAffineInterp(_CurvilinearGridInterp):
    distance_criteria = ["values", "grids"]

    def __init__(self, values, grids, **kwargs):
        if not SKIMAGE_AVAILABLE:
            raise ImportError(
                "PiecewiseAffineTransform requires scikit-image installed."
            )

        self.mc_kwargs = MC_KWARGS.copy()
        # update mc_kwargs with any kwargs that are in MC_KWARGS
        self.mc_kwargs.update((k, v) for k, v in kwargs.items() if k in MC_KWARGS)

        super().__init__(values, grids, target="cpu")

        source = np.reshape(self.grids, (self.ndim, -1)).T
        coordinates = np.mgrid[[slice(0, dim) for dim in self.shape]]
        destination = np.reshape(coordinates, (self.ndim, -1)).T

        interpolator = PiecewiseAffineTransform()
        interpolator.estimate(source, destination)

        self.interpolator = interpolator

    def _get_coordinates(self, args):
        input = np.reshape(args, (self.ndim, -1)).T
        output = self.interpolator(input).T
        return output.reshape(args.shape)

    def _map_coordinates(self, coordinates):
        return map_coordinates(
            self.values, coordinates.reshape(self.ndim, -1), **self.mc_kwargs
        ).reshape(coordinates[0].shape)

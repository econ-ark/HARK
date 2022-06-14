from .core import MetricObject
from interpolation.splines import eval_linear, UCGrid
import numpy as np

class LinearFast(MetricObject):

    distance_criteria = ["f_val", "grid_list"]

    def __init__(self, f_val, grids, extrap_options=None):

        self.f_val = f_val
        self.grid_list = grids
        self.Grid = UCGrid(*grids)
        self.extrap_options = extrap_options

    def __call__(self, *args):

        array_args = [np.asarray(x) for x in args]

        f = eval_linear(
            self.Grid,
            self.f_val,
            np.column_stack([x.flatten() for x in array_args]),
            self.extrap_options,
        )

        return np.reshape(f, array_args[0].shape)

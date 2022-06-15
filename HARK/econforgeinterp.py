from .core import MetricObject
from interpolation.splines import eval_linear, UCGrid
from interpolation.splines import extrap_options as xto
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


class LinearFastDecay(LinearFast):
    def __init__(self, f_val, grids, limit_func, decay_weights=None):

        self.limit_func = limit_func
        self.upper_limits = np.array([x[-1] for x in grids])

        if decay_weights is None:
            self.decay_weights = np.ones(len(grids))
        else:
            self.decay_weights = decay_weights

        super().__init__(f_val, grids)

    def grad(self, *args):
        pass

    def decay(self, x, closest_x):

        dec = np.exp(-np.dot(x - closest_x, self.decay_weights))
        return dec

    def __call__(self, *args):

        # Save the shape of the arguments
        argshape = np.asarray(args[0]).shape
        # Save in a matrix: rows are points, columns are dimensions
        col_args = np.column_stack([np.asarray(x).flatten() for x in args])

        # Get indices, points, and closest in-grid point to points that
        # require extrapolation.
        upper_ex_inds = np.any(col_args > self.upper_limits[None, :], axis=1)
        upper_ex_points = col_args[
            upper_ex_inds,
        ]
        upper_ex_nearest = np.minimum(upper_ex_points, self.upper_limits[None, :])

        # Replace extrapolations by their closest in-grid points and interpolate
        col_args[upper_ex_inds,] = upper_ex_nearest
        f = eval_linear(self.Grid, self.f_val, col_args,)

        # Get interpolations at the closest points
        upper_ex_f_nearest = f[upper_ex_inds]
        # And limiting function at the extrapolating points
        limit_f_ex = self.limit_func(*[upper_ex_points[:, i] for i in range(len(args))])

        # Combine them
        decay = self.decay(upper_ex_points, upper_ex_nearest)
        f[upper_ex_inds] = decay * upper_ex_f_nearest + (1.0 - decay) * limit_f_ex

        return np.reshape(f, argshape)

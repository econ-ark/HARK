from .core import MetricObject
from interpolation.splines import eval_linear, UCGrid
from interpolation.splines import extrap_options as xto

import numpy as np
from copy import copy


class LinearFast(MetricObject):
    """
    A class that constructs and holds all the necessary elements to
    call a multilinear interpolator from econforge.interpolator in
    a way that resembles the basic interpolators in HARK.interpolation.
    """

    distance_criteria = ["f_val", "grid_list"]

    def __init__(self, f_val, grids, extrap_options=None):
        """
        f_val: numpy.array
            An array containing the values of the function at the grid points.
            It's i-th dimension must be of the same lenght as the i-th grid.
            f_val[i,j,k] must be f(grids[0][i], grids[1][j], grids[2][k]).
        grids: [numpy.array]
            One-dimensional list of numpy arrays. It's i-th entry must be the grid
            to be used for the i-th independent variable.
        extrap_options: None or one of xto.NEAREST, xto.LINEAR, or xto.CONSTANT from
            the extrapolation options of econforge.interpolation.
            Determines how to extrapolate, using either nearest point, multilinear, or 
            constant extrapolation. The default is multilinear.
        """
        self.dim = len(grids)
        self.f_val = f_val
        self.grid_list = grids
        self.Grid = UCGrid(*grids)
        self.extrap_options = xto.LINEAR if extrap_options is None else extrap_options

    def __call__(self, *args):
        """
        Calls the interpolator.
        
        args: [numpy.array]
            List of arrays. The i-th entry contains the i-th coordinate
            of all the points to be evaluated. All entries must have the
            same shape.
        """
        array_args = [np.asarray(x) for x in args]

        f = eval_linear(
            self.Grid,
            self.f_val,
            np.column_stack([x.flatten() for x in array_args]),
            self.extrap_options,
        )

        return np.reshape(f, array_args[0].shape)


class LinearFastDecay(LinearFast):
    def __init__(
        self, f_val, grids, limit_func, decay_weights=None, extrap_options=None
    ):

        super().__init__(f_val, grids, extrap_options)

        self.limit_func = limit_func
        self.upper_limits = np.array([x[-1] for x in grids])

        if decay_weights is None:
            # By default, make weights the inverse of upper grid limits
            # so that distances will be re-expressed as proportions of
            # the upper limit
            self.decay_weights = np.abs(1 / self.upper_limits)
        else:
            self.decay_weights = decay_weights

    def decay(self, x, closest_x):

        dist = np.dot(np.abs(x - closest_x), self.decay_weights)

        weight = 1 / (1 / dist + 1)

        return weight

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

        # Find function evaluations with regular extrapolation
        f = eval_linear(self.Grid, self.f_val, col_args, self.extrap_options)

        # Get base extrapolations and limiting function at the extrapolating points
        upper_f_ex = f[upper_ex_inds]
        limit_f_ex = self.limit_func(*[upper_ex_points[:, i] for i in range(len(args))])

        # Combine them
        weight = self.decay(upper_ex_points, upper_ex_nearest)
        f[upper_ex_inds] = (1.0 - weight) * upper_f_ex + weight * limit_f_ex

        return np.reshape(f, argshape)

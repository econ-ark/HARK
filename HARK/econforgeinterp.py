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
        self, f_val, grids, limit_func, decay_weights=None, extrap_gradient=False
    ):

        super().__init__(f_val, grids)

        self.limit_func = limit_func
        self.upper_limits = np.array([x[-1] for x in grids])

        if decay_weights is None:
            self.decay_weights = np.ones(self.dim)
        else:
            self.decay_weights = decay_weights

        self.extrap_gradient = extrap_gradient

    def _grad(self, points):

        # Save in a matrix: rows are points, columns are dimensions
        col_args = copy(points)

        # Preallocate gradient
        grads = np.empty((col_args.shape[0], self.dim))

        for i, grid in enumerate(self.grid_list):

            col = copy(col_args[:, i])

            lower_inds = np.searchsorted(grid, col) - 1
            lower_inds = np.minimum(np.maximum(lower_inds, 0), len(grid) - 2)

            x_low = grid[lower_inds]
            x_upp = grid[lower_inds + 1]

            # Lower values
            col_args[:, i] = x_low
            vals_low = eval_linear(self.Grid, self.f_val, col_args,)
            # Upper values
            col_args[:, i] = x_upp
            vals_upp = eval_linear(self.Grid, self.f_val, col_args,)

            # Restore column
            col_args[:, i] = col

            # Get gradient
            grads[:, i] = (vals_upp - vals_low) / (x_upp - x_low)

        return grads

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

        # Add gradient term if requested
        if self.extrap_gradient:
            grad = self._grad(upper_ex_nearest)
            f[upper_ex_inds] += decay * np.sum(
                (upper_ex_points - upper_ex_nearest) * grad,
                axis = 1,
                keepdims=False,
            )

        return np.reshape(f, argshape)

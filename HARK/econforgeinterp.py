from HARK.core import MetricObject
from interpolation.splines import eval_linear, eval_spline, CGrid
from interpolation.splines import extrap_options as xto

import numpy as np
from copy import copy

extrap_opts = {
    "linear": xto.LINEAR,
    "nearest": xto.NEAREST,
    "constant": xto.CONSTANT,
}


class LinearFast(MetricObject):
    """
    A class that constructs and holds all the necessary elements to
    call a multilinear interpolator from econforge.interpolator in
    a way that resembles the basic interpolators in HARK.interpolation.
    """

    distance_criteria = ["f_val", "grid_list"]

    def __init__(self, f_val, grids, extrap_mode="linear"):
        """
        f_val: numpy.array
            An array containing the values of the function at the grid points.
            It's i-th dimension must be of the same lenght as the i-th grid.
            f_val[i,j,k] must be f(grids[0][i], grids[1][j], grids[2][k]).
        grids: [numpy.array]
            One-dimensional list of numpy arrays. It's i-th entry must be the grid
            to be used for the i-th independent variable.
        extrap_mode: one of 'linear', 'nearest', or 'constant'
            Determines how to extrapolate, using either nearest point, multilinear, or 
            constant extrapolation. The default is multilinear.
        """
        self.dim = len(grids)
        self.f_val = f_val
        self.grid_list = grids
        self.Grid = CGrid(*grids)

        # Set up extrapolation options
        self.extrap_mode = extrap_mode
        try:
            self.extrap_options = extrap_opts[self.extrap_mode]
        except KeyError:
            raise KeyError(
                'extrap_mode must be one of "linear", "nearest", or "costant"'
            )

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

    def _derivs(self, deriv_tuple, *args):

        # Format arguments
        array_args = [np.asarray(x) for x in args]

        # Find derivatives with respect to every dimension
        derivs = eval_spline(
            self.Grid,
            self.f_val,
            np.column_stack([x.flatten() for x in array_args]),
            out=None,
            order=1,
            diff=str(deriv_tuple),
            extrap_mode=self.extrap_mode,
        )

        # Reshape
        derivs = [derivs[:, j].reshape(args[0].shape) for j in range(self.dim)]

        return derivs

    def gradient(self, *args):

        # Form a tuple that indicates which derivatives to get
        # in the way eval_linear expects
        deriv_tup = tuple(
            tuple(1 if j == i else 0 for j in range(self.dim)) for i in range(self.dim)
        )

        return self._derivs(deriv_tup, *args)

    def _eval_and_grad(self, *args):

        # (0,0,...,0) to get the function evaluation
        eval_tup = tuple([tuple(0 for i in range(self.dim))])

        # Tuple with indicators for all the derivatives
        deriv_tup = tuple(
            tuple(1 if j == i else 0 for j in range(self.dim)) for i in range(self.dim)
        )

        results = self._derivs(eval_tup + deriv_tup, *args)

        return (results[0], results[1:])


class DecayInterp(MetricObject):

    distance_criteria = ["interp_f_val", "grid_list"]

    def __init__(self, interp, limit_fun, decay_weights=None):

        self.interp = interp
        self.limit_fun = limit_fun

        self.grid_list = self.interp.grid_list

        self.upper_limits = np.array([x[-1] for x in self.grid_list])
        self.dim = len(self.grid_list)

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
        f = self.interp(*[col_args[:, i] for i in range(self.dim)])

        # Get base extrapolations and limiting function at the extrapolating points
        upper_f_ex = f[upper_ex_inds]
        limit_f_ex = self.limit_fun(*[upper_ex_points[:, i] for i in range(len(args))])

        # Combine them
        weight = self.decay(upper_ex_points, upper_ex_nearest)
        f[upper_ex_inds] = (1.0 - weight) * upper_f_ex + weight * limit_f_ex

        return np.reshape(f, argshape)

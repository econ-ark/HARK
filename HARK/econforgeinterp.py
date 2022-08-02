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

    def gradient(self, *args):

        # Format arguments
        array_args = [np.asarray(x) for x in args]

        # Form a tuple that indicates which derivatives to get
        # in the way eval_linear expects
        deriv_tup = tuple(
            tuple(1 if j == i else 0 for j in range(self.dim)) for i in range(self.dim)
        )

        # Find derivatives with respect to every dimension
        grad = eval_spline(
            self.Grid,
            self.f_val,
            np.column_stack([x.flatten() for x in array_args]),
            out=None,
            order=1,
            diff=str(deriv_tup),
            extrap_mode=self.extrap_mode,
        )

        # Reshape
        grad = [grad[:, j].reshape(args[0].shape) for j in range(self.dim)]

        return grad

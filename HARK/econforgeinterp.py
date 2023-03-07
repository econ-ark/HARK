from copy import copy

import numpy as np
from interpolation.splines import CGrid, eval_linear, eval_spline
from interpolation.splines import extrap_options as xto

from HARK.core import MetricObject

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

        # Call the econforge function
        f = eval_linear(
            self.Grid,
            self.f_val,
            np.column_stack([x.flatten() for x in array_args]),
            self.extrap_options,
        )

        # Reshape the output to the shape of inputs
        return np.reshape(f, array_args[0].shape)

    def _derivs(self, deriv_tuple, *args):
        """
        Evaluates derivatives of the interpolator.

        Parameters
        ----------
        deriv_tuple : tuple of tuples of int
            Indicates what are the derivatives to be computed.
            It follows econforge's notation, where a derivative
            to be calculated is a tuple of length equal to the
            number of dimensions of the interpolator and entries
            in that tuple represent the order of the derivative.
            E.g. to calculate f(x,y) and df/dy(x,y) use
            deriv_tuple = ((0,0),(0,1))

        args: [numpy.array]
            List of arrays. The i-th entry contains the i-th coordinate
            of all the points to be evaluated. All entries must have the
            same shape.

        Returns
        -------
        [numpy.array]
            List of the derivatives that were requested in the same order
            as deriv_tuple. Each element has the shape of items in args.
        """

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
        derivs = [
            derivs[:, j].reshape(args[0].shape) for j, tup in enumerate(deriv_tuple)
        ]

        return derivs

    def gradient(self, *args):
        """
        Evaluates gradient of the interpolator.

        Parameters
        ----------
        args: [numpy.array]
            List of arrays. The i-th entry contains the i-th coordinate
            of all the points to be evaluated. All entries must have the
            same shape.

        Returns
        -------
        [numpy.array]
            List of the derivatives of the function with respect to each
            input, evaluated at the given points. E.g. if the interpolator
            represents 3D function f, f.gradient(x,y,z) will return
            [df/dx(x,y,z), df/dy(x,y,z), df/dz(x,y,z)]. Each element has the
            shape of items in args.
        """
        # Form a tuple that indicates which derivatives to get
        # in the way eval_linear expects
        deriv_tup = tuple(
            tuple(1 if j == i else 0 for j in range(self.dim)) for i in range(self.dim)
        )

        return self._derivs(deriv_tup, *args)

    def _eval_and_grad(self, *args):
        """
        Evaluates interpolator and its gradient.

        Parameters
        ----------
        args: [numpy.array]
            List of arrays. The i-th entry contains the i-th coordinate
            of all the points to be evaluated. All entries must have the
            same shape.

        Returns
        -------
        numpy.array
            Value of the interpolator at given arguments.
        [numpy.array]
            List of the derivatives of the function with respect to each
            input, evaluated at the given points. E.g. if the interpolator
            represents 3D function f, the list will be
            [df/dx(x,y,z), df/dy(x,y,z), df/dz(x,y,z)]. Each element has the
            shape of items in args.
        """
        # (0,0,...,0) to get the function evaluation
        eval_tup = tuple([tuple(0 for i in range(self.dim))])

        # Tuple with indicators for all the derivatives
        deriv_tup = tuple(
            tuple(1 if j == i else 0 for j in range(self.dim)) for i in range(self.dim)
        )

        results = self._derivs(eval_tup + deriv_tup, *args)

        return (results[0], results[1:])


class DecayInterp(MetricObject):
    """
    A class of interpolators that use a limiting function
    for extrapolation.

    See HARK/examples/Interpolation/DecayInterp.ipynb for examples of
    how to use this class.

    """

    distance_criteria = ["interp"]

    def __init__(
        self,
        interp,
        limit_fun,
        limit_grad=None,
        extrap_method="decay_prop",
    ):
        """

        Parameters
        ----------
        interp : N-dim LinearFast object
            Linear interpolator
        limit_fun : N-dim function
            Limiting function to be used when extrapolating
        limit_grad : function, optional
            Function that returns the gradient of the limiting function. Must
            follow the convention of LinearFast's gradients, where the gradient
            takes the form of a list of arrays. By default None
        extrap_method : str, optional
            Method to use for calculating extrapolated values. Must be one of
            "decay_prop", "decay_hark", "paste". By default "decay_prop"
            See HARK/examples/interpolation/DecayInterp.ipynb, for detailed
            explanations of each method.
        """
        self.interp = interp
        self.limit_fun = limit_fun

        self.limit_grad = limit_grad

        self.grid_list = self.interp.grid_list

        self.upper_limits = np.array([x[-1] for x in self.grid_list])
        self.dim = len(self.grid_list)

        self.extrap_methods = {
            "decay_prop": self.extrap_decay_prop,
            "decay_hark": self.extrap_decay_hark,
            "paste": self.extrap_paste,
        }

        try:
            self.extrap_fun = self.extrap_methods[extrap_method]
        except KeyError:
            raise KeyError(
                'extrap_method must be one of "decay_prop", "decay_hark", or "paste"'
            )

    def __call__(self, *args):
        """
        Calls the interpolator with decay extrapolation.

        args: [numpy.array]
            List of arrays. The i-th entry contains the i-th coordinate
            of all the points to be evaluated. All entries must have the
            same shape.
        """

        # Save the shape of the arguments
        argshape = np.asarray(args[0]).shape
        # Save in a matrix: rows are points, columns are dimensions
        col_args = np.column_stack([np.asarray(x).flatten() for x in args])

        # Get indices, points, and closest in-grid point to points that
        # require extrapolation.
        upper_ex_inds = np.any(col_args > self.upper_limits[None, :], axis=1)
        upper_ex_points = col_args[upper_ex_inds,]
        upper_ex_nearest = np.minimum(upper_ex_points, self.upper_limits[None, :])

        # Find function evaluations with regular extrapolation
        f = self.interp(*[col_args[:, i] for i in range(self.dim)])

        # Find extrapolated values with chosen method
        f[upper_ex_inds] = self.extrap_fun(upper_ex_points, upper_ex_nearest)

        return np.reshape(f, argshape)

    def extrap_decay_prop(self, x, closest_x):
        """
        "decay_prop" extrapolation method. Combines the interpolator's
        default extrapolation and the limiting function with weights
        that depend on how far from the grid the values are.

        Parameters
        ----------
        x : inputs that require extrapolation.
        closest_x : for each of the inputs that require extrapolation, contains
            the closest point that falls inside the grid.
        """
        # Evaluate base interpolator at x
        f_val_x = self.interp(*[x[:, i] for i in range(self.dim)])
        # Evaluate limiting function at x
        g_val_x = self.limit_fun(*[x[:, i] for i in range(self.dim)])

        # Find distance between points and closest in-grid point.
        decay_weights = np.abs(1 / self.upper_limits)  # Rescale as proportions
        dist = np.dot(np.abs(x - closest_x), decay_weights)
        weight = np.exp(-1 * dist)

        return weight * f_val_x + (1 - weight) * g_val_x

    def extrap_decay_hark(self, x, closest_x):
        """
        "decay_hark" extrapolation method. Takes into account the rate at
        which the interpolator and limiting function are approaching at the
        edge of the grid for combining them.

        Parameters
        ----------
        x : inputs that require extrapolation.
        closest_x : for each of the inputs that require extrapolation, contains
            the closest point that falls inside the grid.
        """

        # Evaluate limiting function at x
        g_val_x = self.limit_fun(*[x[:, i] for i in range(self.dim)])

        # Get gradients and values at the closest in-grid point
        closest_x_arglist = [closest_x[:, i][..., None] for i in range(self.dim)]

        # Interpolator
        f_val, f_grad = self.interp._eval_and_grad(*closest_x_arglist)
        f_grad = np.hstack(f_grad)
        # Limit
        g_val = self.limit_fun(*closest_x_arglist)
        g_grad = self.limit_grad(*closest_x_arglist)
        g_grad = np.hstack(g_grad)

        # Construct weights
        A = g_val - f_val
        B = np.abs(np.divide(1, A) * (g_grad - f_grad))
        # Distance weighted by B
        w_dist = np.sum(B * (x - closest_x), axis=1, keepdims=True)
        # If f and g start out together at the edge of the grid, treat
        # the point as infinitely far away so that the limiting value is used.
        w_dist[A.flatten() == 0.0, ...] = np.inf

        # Combine the limit value at x and the values at the
        # edge of the grid
        val = g_val_x[..., None] - A * np.exp(-1.0 * w_dist)

        return val.flatten()

    def extrap_paste(self, x, closest_x):
        """
        "paste" extrapolation method. Uses the limiting function
        for extrapolation, but with a vertical shift that "pastes" it
        to the interpolator at the edge of the grid.

        Parameters
        ----------
        x : inputs that require extrapolation.
        closest_x : for each of the inputs that require extrapolation, contains
            the closest point that falls inside the grid.
        """
        # Evaluate base interpolator and limit at closest x
        f_val_closest = self.interp(*[closest_x[:, i] for i in range(self.dim)])
        g_val_closest = self.limit_fun(*[closest_x[:, i] for i in range(self.dim)])

        # Evaluate limit function at x
        g_val_x = self.limit_fun(*[x[:, i] for i in range(self.dim)])

        return f_val_closest + (g_val_x - g_val_closest)

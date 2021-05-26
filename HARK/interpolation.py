"""
Custom interpolation methods for representing approximations to functions.
It also includes wrapper classes to enforce standard methods across classes.
Each interpolation class must have a distance() method that compares itself to
another instance; this is used in HARK.core's solve() method to check for solution
convergence.  The interpolator classes currently in this module inherit their
distance method from MetricObject.
"""
import numpy as np
from .core import MetricObject
from copy import deepcopy
from HARK.utilities import CRRAutility, CRRAutilityP, CRRAutilityPP
import warnings


def _isscalar(x):
    """
    Check whether x is if a scalar type, or 0-dim.

    Parameters
    ----------
    x : anything
        An input to be checked for scalar-ness.

    Returns
    -------
    is_scalar : boolean
        True if the input is a scalar, False otherwise.
    """
    return np.isscalar(x) or hasattr(x, "shape") and x.shape == ()


def _check_grid_dimensions(dimension, *args):
    if dimension == 1:
        if len(args[0]) != len(args[1]):
            raise ValueError("Grid dimensions of x and f(x) do not match")
    elif dimension == 2:
        if args[0].shape != (args[1].size, args[2].size):
            raise ValueError("Grid dimensions of x, y and f(x, y) do not match")
    elif dimension == 3:
        if args[0].shape != (args[1].size, args[2].size, args[3].size):
            raise ValueError("Grid dimensions of x, y, z and f(x, y, z) do not match")
    elif dimension == 4:
        if args[0].shape != (args[1].size, args[2].size, args[3].size, args[4].size):
            raise ValueError("Grid dimensions of x, y, z and f(x, y, z) do not match")
    else:
        raise ValueError("Dimension should be between 1 and 4 inclusive.")


def _check_flatten(dimension, *args):
    if dimension == 1:
        if isinstance(args[0], np.ndarray) and args[0].shape != args[0].flatten().shape:
            warnings.warn("input not of the size (n, ), attempting to flatten")
            return False
        else:
            return True


class HARKinterpolator1D(MetricObject):
    """
    A wrapper class for 1D interpolation methods in HARK.
    """

    distance_criteria = []

    def __call__(self, x):
        """
        Evaluates the interpolated function at the given input.

        Parameters
        ----------
        x : np.array or float
            Real values to be evaluated in the interpolated function.

        Returns
        -------
        y : np.array or float
            The interpolated function evaluated at x: y = f(x), with the same
            shape as x.
        """
        z = np.asarray(x)
        return (self._evaluate(z.flatten())).reshape(z.shape)

    def derivative(self, x):
        """
        Evaluates the derivative of the interpolated function at the given input.

        Parameters
        ----------
        x : np.array or float
            Real values to be evaluated in the interpolated function.

        Returns
        -------
        dydx : np.array or float
            The interpolated function's first derivative evaluated at x:
            dydx = f'(x), with the same shape as x.
        """
        z = np.asarray(x)
        return (self._der(z.flatten())).reshape(z.shape)

    def eval_with_derivative(self, x):
        """
        Evaluates the interpolated function and its derivative at the given input.

        Parameters
        ----------
        x : np.array or float
            Real values to be evaluated in the interpolated function.

        Returns
        -------
        y : np.array or float
            The interpolated function evaluated at x: y = f(x), with the same
            shape as x.
        dydx : np.array or float
            The interpolated function's first derivative evaluated at x:
            dydx = f'(x), with the same shape as x.
        """
        z = np.asarray(x)
        y, dydx = self._evalAndDer(z.flatten())
        return y.reshape(z.shape), dydx.reshape(z.shape)

    def _evaluate(self, x):
        """
        Interpolated function evaluator, to be defined in subclasses.
        """
        raise NotImplementedError()

    def _der(self, x):
        """
        Interpolated function derivative evaluator, to be defined in subclasses.
        """
        raise NotImplementedError()

    def _evalAndDer(self, x):
        """
        Interpolated function and derivative evaluator, to be defined in subclasses.
        """
        raise NotImplementedError()


class HARKinterpolator2D(MetricObject):
    """
    A wrapper class for 2D interpolation methods in HARK.
    """

    distance_criteria = []

    def __call__(self, x, y):
        """
        Evaluates the interpolated function at the given input.

        Parameters
        ----------
        x : np.array or float
            Real values to be evaluated in the interpolated function.
        y : np.array or float
            Real values to be evaluated in the interpolated function; must be
            the same size as x.

        Returns
        -------
        fxy : np.array or float
            The interpolated function evaluated at x,y: fxy = f(x,y), with the
            same shape as x and y.
        """
        xa = np.asarray(x)
        ya = np.asarray(y)
        return (self._evaluate(xa.flatten(), ya.flatten())).reshape(xa.shape)

    def derivativeX(self, x, y):
        """
        Evaluates the partial derivative of interpolated function with respect
        to x (the first argument) at the given input.

        Parameters
        ----------
        x : np.array or float
            Real values to be evaluated in the interpolated function.
        y : np.array or float
            Real values to be evaluated in the interpolated function; must be
            the same size as x.

        Returns
        -------
        dfdx : np.array or float
            The derivative of the interpolated function with respect to x, eval-
            uated at x,y: dfdx = f_x(x,y), with the same shape as x and y.
        """
        xa = np.asarray(x)
        ya = np.asarray(y)
        return (self._derX(xa.flatten(), ya.flatten())).reshape(xa.shape)

    def derivativeY(self, x, y):
        """
        Evaluates the partial derivative of interpolated function with respect
        to y (the second argument) at the given input.

        Parameters
        ----------
        x : np.array or float
            Real values to be evaluated in the interpolated function.
        y : np.array or float
            Real values to be evaluated in the interpolated function; must be
            the same size as x.

        Returns
        -------
        dfdy : np.array or float
            The derivative of the interpolated function with respect to y, eval-
            uated at x,y: dfdx = f_y(x,y), with the same shape as x and y.
        """
        xa = np.asarray(x)
        ya = np.asarray(y)
        return (self._derY(xa.flatten(), ya.flatten())).reshape(xa.shape)

    def _evaluate(self, x, y):
        """
        Interpolated function evaluator, to be defined in subclasses.
        """
        raise NotImplementedError()

    def _derX(self, x, y):
        """
        Interpolated function x-derivative evaluator, to be defined in subclasses.
        """
        raise NotImplementedError()

    def _derY(self, x, y):
        """
        Interpolated function y-derivative evaluator, to be defined in subclasses.
        """
        raise NotImplementedError()


class HARKinterpolator3D(MetricObject):
    """
    A wrapper class for 3D interpolation methods in HARK.
    """

    distance_criteria = []

    def __call__(self, x, y, z):
        """
        Evaluates the interpolated function at the given input.

        Parameters
        ----------
        x : np.array or float
            Real values to be evaluated in the interpolated function.
        y : np.array or float
            Real values to be evaluated in the interpolated function; must be
            the same size as x.
        z : np.array or float
            Real values to be evaluated in the interpolated function; must be
            the same size as x.

        Returns
        -------
        fxyz : np.array or float
            The interpolated function evaluated at x,y,z: fxyz = f(x,y,z), with
            the same shape as x, y, and z.
        """
        xa = np.asarray(x)
        ya = np.asarray(y)
        za = np.asarray(z)
        return (self._evaluate(xa.flatten(), ya.flatten(), za.flatten())).reshape(
            xa.shape
        )

    def derivativeX(self, x, y, z):
        """
        Evaluates the partial derivative of the interpolated function with respect
        to x (the first argument) at the given input.

        Parameters
        ----------
        x : np.array or float
            Real values to be evaluated in the interpolated function.
        y : np.array or float
            Real values to be evaluated in the interpolated function; must be
            the same size as x.
        z : np.array or float
            Real values to be evaluated in the interpolated function; must be
            the same size as x.

        Returns
        -------
        dfdx : np.array or float
            The derivative with respect to x of the interpolated function evaluated
            at x,y,z: dfdx = f_x(x,y,z), with the same shape as x, y, and z.
        """
        xa = np.asarray(x)
        ya = np.asarray(y)
        za = np.asarray(z)
        return (self._derX(xa.flatten(), ya.flatten(), za.flatten())).reshape(xa.shape)

    def derivativeY(self, x, y, z):
        """
        Evaluates the partial derivative of the interpolated function with respect
        to y (the second argument) at the given input.

        Parameters
        ----------
        x : np.array or float
            Real values to be evaluated in the interpolated function.
        y : np.array or float
            Real values to be evaluated in the interpolated function; must be
            the same size as x.
        z : np.array or float
            Real values to be evaluated in the interpolated function; must be
            the same size as x.

        Returns
        -------
        dfdy : np.array or float
            The derivative with respect to y of the interpolated function evaluated
            at x,y,z: dfdy = f_y(x,y,z), with the same shape as x, y, and z.
        """
        xa = np.asarray(x)
        ya = np.asarray(y)
        za = np.asarray(z)
        return (self._derY(xa.flatten(), ya.flatten(), za.flatten())).reshape(xa.shape)

    def derivativeZ(self, x, y, z):
        """
        Evaluates the partial derivative of the interpolated function with respect
        to z (the third argument) at the given input.

        Parameters
        ----------
        x : np.array or float
            Real values to be evaluated in the interpolated function.
        y : np.array or float
            Real values to be evaluated in the interpolated function; must be
            the same size as x.
        z : np.array or float
            Real values to be evaluated in the interpolated function; must be
            the same size as x.

        Returns
        -------
        dfdz : np.array or float
            The derivative with respect to z of the interpolated function evaluated
            at x,y,z: dfdz = f_z(x,y,z), with the same shape as x, y, and z.
        """
        xa = np.asarray(x)
        ya = np.asarray(y)
        za = np.asarray(z)
        return (self._derZ(xa.flatten(), ya.flatten(), za.flatten())).reshape(xa.shape)

    def _evaluate(self, x, y, z):
        """
        Interpolated function evaluator, to be defined in subclasses.
        """
        raise NotImplementedError()

    def _derX(self, x, y, z):
        """
        Interpolated function x-derivative evaluator, to be defined in subclasses.
        """
        raise NotImplementedError()

    def _derY(self, x, y, z):
        """
        Interpolated function y-derivative evaluator, to be defined in subclasses.
        """
        raise NotImplementedError()

    def _derZ(self, x, y, z):
        """
        Interpolated function y-derivative evaluator, to be defined in subclasses.
        """
        raise NotImplementedError()


class HARKinterpolator4D(MetricObject):
    """
    A wrapper class for 4D interpolation methods in HARK.
    """

    distance_criteria = []

    def __call__(self, w, x, y, z):
        """
        Evaluates the interpolated function at the given input.

        Parameters
        ----------
        w : np.array or float
            Real values to be evaluated in the interpolated function.
        x : np.array or float
            Real values to be evaluated in the interpolated function; must be
            the same size as w.
        y : np.array or float
            Real values to be evaluated in the interpolated function; must be
            the same size as w.
        z : np.array or float
            Real values to be evaluated in the interpolated function; must be
            the same size as w.

        Returns
        -------
        fwxyz : np.array or float
            The interpolated function evaluated at w,x,y,z: fwxyz = f(w,x,y,z),
            with the same shape as w, x, y, and z.
        """
        wa = np.asarray(w)
        xa = np.asarray(x)
        ya = np.asarray(y)
        za = np.asarray(z)
        return (
            self._evaluate(wa.flatten(), xa.flatten(), ya.flatten(), za.flatten())
        ).reshape(wa.shape)

    def derivativeW(self, w, x, y, z):
        """
        Evaluates the partial derivative with respect to w (the first argument)
        of the interpolated function at the given input.

        Parameters
        ----------
        w : np.array or float
            Real values to be evaluated in the interpolated function.
        x : np.array or float
            Real values to be evaluated in the interpolated function; must be
            the same size as w.
        y : np.array or float
            Real values to be evaluated in the interpolated function; must be
            the same size as w.
        z : np.array or float
            Real values to be evaluated in the interpolated function; must be
            the same size as w.

        Returns
        -------
        dfdw : np.array or float
            The derivative with respect to w of the interpolated function eval-
            uated at w,x,y,z: dfdw = f_w(w,x,y,z), with the same shape as inputs.
        """
        wa = np.asarray(w)
        xa = np.asarray(x)
        ya = np.asarray(y)
        za = np.asarray(z)
        return (
            self._derW(wa.flatten(), xa.flatten(), ya.flatten(), za.flatten())
        ).reshape(wa.shape)

    def derivativeX(self, w, x, y, z):
        """
        Evaluates the partial derivative with respect to x (the second argument)
        of the interpolated function at the given input.

        Parameters
        ----------
        w : np.array or float
            Real values to be evaluated in the interpolated function.
        x : np.array or float
            Real values to be evaluated in the interpolated function; must be
            the same size as w.
        y : np.array or float
            Real values to be evaluated in the interpolated function; must be
            the same size as w.
        z : np.array or float
            Real values to be evaluated in the interpolated function; must be
            the same size as w.

        Returns
        -------
        dfdx : np.array or float
            The derivative with respect to x of the interpolated function eval-
            uated at w,x,y,z: dfdx = f_x(w,x,y,z), with the same shape as inputs.
        """
        wa = np.asarray(w)
        xa = np.asarray(x)
        ya = np.asarray(y)
        za = np.asarray(z)
        return (
            self._derX(wa.flatten(), xa.flatten(), ya.flatten(), za.flatten())
        ).reshape(wa.shape)

    def derivativeY(self, w, x, y, z):
        """
        Evaluates the partial derivative with respect to y (the third argument)
        of the interpolated function at the given input.

        Parameters
        ----------
        w : np.array or float
            Real values to be evaluated in the interpolated function.
        x : np.array or float
            Real values to be evaluated in the interpolated function; must be
            the same size as w.
        y : np.array or float
            Real values to be evaluated in the interpolated function; must be
            the same size as w.
        z : np.array or float
            Real values to be evaluated in the interpolated function; must be
            the same size as w.

        Returns
        -------
        dfdy : np.array or float
            The derivative with respect to y of the interpolated function eval-
            uated at w,x,y,z: dfdy = f_y(w,x,y,z), with the same shape as inputs.
        """
        wa = np.asarray(w)
        xa = np.asarray(x)
        ya = np.asarray(y)
        za = np.asarray(z)
        return (
            self._derY(wa.flatten(), xa.flatten(), ya.flatten(), za.flatten())
        ).reshape(wa.shape)

    def derivativeZ(self, w, x, y, z):
        """
        Evaluates the partial derivative with respect to z (the fourth argument)
        of the interpolated function at the given input.

        Parameters
        ----------
        w : np.array or float
            Real values to be evaluated in the interpolated function.
        x : np.array or float
            Real values to be evaluated in the interpolated function; must be
            the same size as w.
        y : np.array or float
            Real values to be evaluated in the interpolated function; must be
            the same size as w.
        z : np.array or float
            Real values to be evaluated in the interpolated function; must be
            the same size as w.

        Returns
        -------
        dfdz : np.array or float
            The derivative with respect to z of the interpolated function eval-
            uated at w,x,y,z: dfdz = f_z(w,x,y,z), with the same shape as inputs.
        """
        wa = np.asarray(w)
        xa = np.asarray(x)
        ya = np.asarray(y)
        za = np.asarray(z)
        return (
            self._derZ(wa.flatten(), xa.flatten(), ya.flatten(), za.flatten())
        ).reshape(wa.shape)

    def _evaluate(self, w, x, y, z):
        """
        Interpolated function evaluator, to be defined in subclasses.
        """
        raise NotImplementedError()

    def _derW(self, w, x, y, z):
        """
        Interpolated function w-derivative evaluator, to be defined in subclasses.
        """
        raise NotImplementedError()

    def _derX(self, w, x, y, z):
        """
        Interpolated function w-derivative evaluator, to be defined in subclasses.
        """
        raise NotImplementedError()

    def _derY(self, w, x, y, z):
        """
        Interpolated function w-derivative evaluator, to be defined in subclasses.
        """
        raise NotImplementedError()

    def _derZ(self, w, x, y, z):
        """
        Interpolated function w-derivative evaluator, to be defined in subclasses.
        """
        raise NotImplementedError()


class IdentityFunction(MetricObject):
    """
    A fairly trivial interpolator that simply returns one of its arguments.  Useful for avoiding
    numeric error in extreme cases.

    Parameters
    ----------
    i_dim : int
        Index of the dimension on which the identity is defined.  f(*x) = x[i]
    n_dims : int
        Total number of input dimensions for this function.
    """

    distance_criteria = ["i_dim"]

    def __init__(self, i_dim=0, n_dims=1):
        self.i_dim = i_dim
        self.n_dims = n_dims

    def __call__(self, *args):
        """
        Evaluate the identity function.
        """
        return args[self.i_dim]

    def derivative(self, *args):
        """
        Returns the derivative of the function with respect to the first dimension.
        """
        if self.i_dim == 0:
            return np.ones_like(*args[0])
        else:
            return np.zeros_like(*args[0])

    def derivativeX(self, *args):
        """
        Returns the derivative of the function with respect to the X dimension.
        This is the first input whenever n_dims < 4 and the second input otherwise.
        """
        if self.n_dims >= 4:
            j = 1
        else:
            j = 0
        if self.i_dim == j:
            return np.ones_like(*args[0])
        else:
            return np.zeros_like(*args[0])

    def derivativeY(self, *args):
        """
        Returns the derivative of the function with respect to the Y dimension.
        This is the second input whenever n_dims < 4 and the third input otherwise.
        """
        if self.n_dims >= 4:
            j = 2
        else:
            j = 1
        if self.i_dim == j:
            return np.ones_like(*args[0])
        else:
            return np.zeros_like(*args[0])

    def derivativeZ(self, *args):
        """
        Returns the derivative of the function with respect to the Z dimension.
        This is the third input whenever n_dims < 4 and the fourth input otherwise.
        """
        if self.n_dims >= 4:
            j = 3
        else:
            j = 2
        if self.i_dim == j:
            return np.ones_like(*args[0])
        else:
            return np.zeros_like(*args[0])

    def derivativeW(self, *args):
        """
        Returns the derivative of the function with respect to the W dimension.
        This should only exist when n_dims >= 4.
        """
        if self.n_dims >= 4:
            j = 0
        else:
            assert (
                False
            ), "Derivative with respect to W can't be called when n_dims < 4!"
        if self.i_dim == j:
            return np.ones_like(*args[0])
        else:
            return np.zeros_like(*args[0])


class ConstantFunction(MetricObject):
    """
    A class for representing trivial functions that return the same real output for any input.  This
    is convenient for models where an object might be a (non-trivial) function, but in some variations
    that object is just a constant number.  Rather than needing to make a (Bi/Tri/Quad)-
    LinearInterpolation with trivial state grids and the same f_value in every entry, ConstantFunction
    allows the user to quickly make a constant/trivial function.  This comes up, e.g., in models
    with endogenous pricing of insurance contracts; a contract's premium might depend on some state
    variables of the individual, but in some variations the premium of a contract is just a number.

    Parameters
    ----------
    value : float
        The constant value that the function returns.
    """

    convergence_criteria = ["value"]

    def __init__(self, value):
        self.value = float(value)

    def __call__(self, *args):
        """
        Evaluate the constant function.  The first input must exist and should be an array.
        Returns an array of identical shape to args[0] (if it exists).
        """
        if (
            len(args) > 0
        ):  # If there is at least one argument, return appropriately sized array
            if _isscalar(args[0]):
                return self.value
            else:
                shape = args[0].shape
                return self.value * np.ones(shape)
        else:  # Otherwise, return a single instance of the constant value
            return self.value

    def _der(self, *args):
        """
        Evaluate the derivative of the function.  The first input must exist and should be an array.
        Returns an array of identical shape to args[0] (if it exists).  This is an array of zeros.
        """
        if len(args) > 0:
            if _isscalar(args[0]):
                return 0.0
            else:
                shape = args[0].shape
                return np.zeros(shape)
        else:
            return 0.0

    # All other derivatives are also zero everywhere, so these methods just point to derivative
    derivative = _der
    derivativeX = derivative
    derivativeY = derivative
    derivativeZ = derivative
    derivativeW = derivative
    derivativeXX = derivative


class LinearInterp(HARKinterpolator1D):
    """
    A "from scratch" 1D linear interpolation class.  Allows for linear or decay
    extrapolation (approaching a limiting linear function from below).

    NOTE: When no input is given for the limiting linear function, linear
    extrapolation is used above the highest gridpoint.

    Parameters
    ----------
    x_list : np.array
        List of x values composing the grid.
    y_list : np.array
        List of y values, representing f(x) at the points in x_list.
    intercept_limit : float
        Intercept of limiting linear function.
    slope_limit : float
        Slope of limiting linear function.
    lower_extrap : boolean
        Indicator for whether lower extrapolation is allowed.  False means
        f(x) = NaN for x < min(x_list); True means linear extrapolation.
    """

    distance_criteria = ["x_list", "y_list"]

    def __init__(
        self, x_list, y_list, intercept_limit=None, slope_limit=None, lower_extrap=False
    ):
        # Make the basic linear spline interpolation
        self.x_list = (
            np.array(x_list)
            if _check_flatten(1, x_list)
            else np.array(x_list).flatten()
        )
        self.y_list = (
            np.array(y_list)
            if _check_flatten(1, y_list)
            else np.array(y_list).flatten()
        )
        _check_grid_dimensions(1, self.y_list, self.x_list)
        self.lower_extrap = lower_extrap
        self.x_n = self.x_list.size

        # Make a decay extrapolation
        if intercept_limit is not None and slope_limit is not None:
            slope_at_top = (y_list[-1] - y_list[-2]) / (x_list[-1] - x_list[-2])
            level_diff = intercept_limit + slope_limit * x_list[-1] - y_list[-1]
            slope_diff = slope_limit - slope_at_top
            # If the model that can handle uncertainty has been calibrated with
            # with uncertainty set to zero, the 'extrapolation' will blow up
            # Guard against that and nearby problems by testing slope equality
            if not np.isclose(slope_limit, slope_at_top, atol=1e-15):
                self.decay_extrap_A = level_diff
                self.decay_extrap_B = -slope_diff / level_diff
                self.intercept_limit = intercept_limit
                self.slope_limit = slope_limit
                self.decay_extrap = True
            else:
                self.decay_extrap = False
        else:
            self.decay_extrap = False

    def _evalOrDer(self, x, _eval, _Der):
        """
        Returns the level and/or first derivative of the function at each value in
        x.  Only called internally by HARKinterpolator1D.eval_and_der (etc).

        Parameters
        ----------
        x_list : scalar or np.array
            Set of points where we want to evlauate the interpolated function and/or its derivative..
        _eval : boolean
            Indicator for whether to evalute the level of the interpolated function.
        _Der : boolean
            Indicator for whether to evaluate the derivative of the interpolated function.

        Returns
        -------
        A list including the level and/or derivative of the interpolated function where requested.
        """

        i = np.maximum(np.searchsorted(self.x_list[:-1], x), 1)
        alpha = (x - self.x_list[i - 1]) / (self.x_list[i] - self.x_list[i - 1])

        if _eval:
            y = (1.0 - alpha) * self.y_list[i - 1] + alpha * self.y_list[i]
        if _Der:
            dydx = (self.y_list[i] - self.y_list[i - 1]) / (
                self.x_list[i] - self.x_list[i - 1]
            )

        if not self.lower_extrap:
            below_lower_bound = x < self.x_list[0]

            if _eval:
                y[below_lower_bound] = np.nan
            if _Der:
                dydx[below_lower_bound] = np.nan

        if self.decay_extrap:
            above_upper_bound = x > self.x_list[-1]
            x_temp = x[above_upper_bound] - self.x_list[-1]

            if _eval:
                y[above_upper_bound] = (
                    self.intercept_limit
                    + self.slope_limit * x[above_upper_bound]
                    - self.decay_extrap_A * np.exp(-self.decay_extrap_B * x_temp)
                )

            if _Der:
                dydx[above_upper_bound] = (
                    self.slope_limit
                    + self.decay_extrap_B
                    * self.decay_extrap_A
                    * np.exp(-self.decay_extrap_B * x_temp)
                )

        output = []
        if _eval:
            output += [
                y,
            ]
        if _Der:
            output += [
                dydx,
            ]

        return output

    def _evaluate(self, x, return_indices=False):
        """
        Returns the level of the interpolated function at each value in x.  Only
        called internally by HARKinterpolator1D.__call__ (etc).
        """
        return self._evalOrDer(x, True, False)[0]

    def _der(self, x):
        """
        Returns the first derivative of the interpolated function at each value
        in x. Only called internally by HARKinterpolator1D.derivative (etc).
        """
        return self._evalOrDer(x, False, True)[0]

    def _evalAndDer(self, x):
        """
        Returns the level and first derivative of the function at each value in
        x.  Only called internally by HARKinterpolator1D.eval_and_der (etc).
        """
        y, dydx = self._evalOrDer(x, True, True)

        return y, dydx


class CubicInterp(HARKinterpolator1D):
    """
    An interpolating function using piecewise cubic splines.  Matches level and
    slope of 1D function at gridpoints, smoothly interpolating in between.
    Extrapolation above highest gridpoint approaches a limiting linear function
    if desired (linear extrapolation also enabled.)

    NOTE: When no input is given for the limiting linear function, linear
        extrapolation is used above the highest gridpoint.

    Parameters
    ----------
    x_list : np.array
        List of x values composing the grid.
    y_list : np.array
        List of y values, representing f(x) at the points in x_list.
    dydx_list : np.array
        List of dydx values, representing f'(x) at the points in x_list
    intercept_limit : float
        Intercept of limiting linear function.
    slope_limit : float
        Slope of limiting linear function.
    lower_extrap : boolean
        Indicator for whether lower extrapolation is allowed.  False means
        f(x) = NaN for x < min(x_list); True means linear extrapolation.
    """

    distance_criteria = ["x_list", "y_list", "dydx_list"]

    def __init__(
        self,
        x_list,
        y_list,
        dydx_list,
        intercept_limit=None,
        slope_limit=None,
        lower_extrap=False,
    ):
        self.x_list = (
            np.asarray(x_list)
            if _check_flatten(1, x_list)
            else np.array(x_list).flatten()
        )
        self.y_list = (
            np.asarray(y_list)
            if _check_flatten(1, y_list)
            else np.array(y_list).flatten()
        )
        self.dydx_list = (
            np.asarray(dydx_list)
            if _check_flatten(1, dydx_list)
            else np.array(dydx_list).flatten()
        )
        _check_grid_dimensions(1, self.y_list, self.x_list)
        _check_grid_dimensions(1, self.dydx_list, self.x_list)

        self.n = len(x_list)

        # Define lower extrapolation as linear function (or just NaN)
        if lower_extrap:
            self.coeffs = [[y_list[0], dydx_list[0], 0, 0]]
        else:
            self.coeffs = [[np.nan, np.nan, np.nan, np.nan]]

        # Calculate interpolation coefficients on segments mapped to [0,1]
        for i in range(self.n - 1):
            x0 = x_list[i]
            y0 = y_list[i]
            x1 = x_list[i + 1]
            y1 = y_list[i + 1]
            Span = x1 - x0
            dydx0 = dydx_list[i] * Span
            dydx1 = dydx_list[i + 1] * Span

            temp = [
                y0,
                dydx0,
                3 * (y1 - y0) - 2 * dydx0 - dydx1,
                2 * (y0 - y1) + dydx0 + dydx1,
            ]
            self.coeffs.append(temp)

        # Calculate extrapolation coefficients as a decay toward limiting function y = mx+b
        if slope_limit is None and intercept_limit is None:
            slope_limit = dydx_list[-1]
            intercept_limit = y_list[-1] - slope_limit * x_list[-1]
        gap = slope_limit * x1 + intercept_limit - y1
        slope = slope_limit - dydx_list[self.n - 1]
        if (gap != 0) and (slope <= 0):
            temp = [intercept_limit, slope_limit, gap, slope / gap]
        elif slope > 0:
            temp = [
                intercept_limit,
                slope_limit,
                0,
                0,
            ]  # fixing a problem when slope is positive
        else:
            temp = [intercept_limit, slope_limit, gap, 0]
        self.coeffs.append(temp)
        self.coeffs = np.array(self.coeffs)

    def _evaluate(self, x):
        """
        Returns the level of the interpolated function at each value in x.  Only
        called internally by HARKinterpolator1D.__call__ (etc).
        """
        if _isscalar(x):
            pos = np.searchsorted(self.x_list, x)
            if pos == 0:
                y = self.coeffs[0, 0] + self.coeffs[0, 1] * (x - self.x_list[0])
            elif pos < self.n:
                alpha = (x - self.x_list[pos - 1]) / (
                    self.x_list[pos] - self.x_list[pos - 1]
                )
                y = self.coeffs[pos, 0] + alpha * (
                    self.coeffs[pos, 1]
                    + alpha * (self.coeffs[pos, 2] + alpha * self.coeffs[pos, 3])
                )
            else:
                alpha = x - self.x_list[self.n - 1]
                y = (
                    self.coeffs[pos, 0]
                    + x * self.coeffs[pos, 1]
                    - self.coeffs[pos, 2] * np.exp(alpha * self.coeffs[pos, 3])
                )
        else:
            m = len(x)
            pos = np.searchsorted(self.x_list, x)
            y = np.zeros(m)
            if y.size > 0:
                out_bot = pos == 0
                out_top = pos == self.n
                in_bnds = np.logical_not(np.logical_or(out_bot, out_top))

                # Do the "in bounds" evaluation points
                i = pos[in_bnds]
                coeffs_in = self.coeffs[i, :]
                alpha = (x[in_bnds] - self.x_list[i - 1]) / (
                    self.x_list[i] - self.x_list[i - 1]
                )
                y[in_bnds] = coeffs_in[:, 0] + alpha * (
                    coeffs_in[:, 1]
                    + alpha * (coeffs_in[:, 2] + alpha * coeffs_in[:, 3])
                )

                # Do the "out of bounds" evaluation points
                y[out_bot] = self.coeffs[0, 0] + self.coeffs[0, 1] * (
                    x[out_bot] - self.x_list[0]
                )
                alpha = x[out_top] - self.x_list[self.n - 1]
                y[out_top] = (
                    self.coeffs[self.n, 0]
                    + x[out_top] * self.coeffs[self.n, 1]
                    - self.coeffs[self.n, 2] * np.exp(alpha * self.coeffs[self.n, 3])
                )

                y[x == self.x_list[0]] = self.y_list[0]
                
        return y

    def _der(self, x):
        """
        Returns the first derivative of the interpolated function at each value
        in x. Only called internally by HARKinterpolator1D.derivative (etc).
        """
        if _isscalar(x):
            pos = np.searchsorted(self.x_list, x)
            if pos == 0:
                dydx = self.coeffs[0, 1]
            elif pos < self.n:
                alpha = (x - self.x_list[pos - 1]) / (
                    self.x_list[pos] - self.x_list[pos - 1]
                )
                dydx = (
                    self.coeffs[pos, 1]
                    + alpha
                    * (2 * self.coeffs[pos, 2] + alpha * 3 * self.coeffs[pos, 3])
                ) / (self.x_list[pos] - self.x_list[pos - 1])
            else:
                alpha = x - self.x_list[self.n - 1]
                dydx = self.coeffs[pos, 1] - self.coeffs[pos, 2] * self.coeffs[
                    pos, 3
                ] * np.exp(alpha * self.coeffs[pos, 3])
        else:
            m = len(x)
            pos = np.searchsorted(self.x_list, x)
            dydx = np.zeros(m)
            if dydx.size > 0:
                out_bot = pos == 0
                out_top = pos == self.n
                in_bnds = np.logical_not(np.logical_or(out_bot, out_top))

                # Do the "in bounds" evaluation points
                i = pos[in_bnds]
                coeffs_in = self.coeffs[i, :]
                alpha = (x[in_bnds] - self.x_list[i - 1]) / (
                    self.x_list[i] - self.x_list[i - 1]
                )
                dydx[in_bnds] = (
                    coeffs_in[:, 1]
                    + alpha * (2 * coeffs_in[:, 2] + alpha * 3 * coeffs_in[:, 3])
                ) / (self.x_list[i] - self.x_list[i - 1])

                # Do the "out of bounds" evaluation points
                dydx[out_bot] = self.coeffs[0, 1]
                alpha = x[out_top] - self.x_list[self.n - 1]
                dydx[out_top] = self.coeffs[self.n, 1] - self.coeffs[
                    self.n, 2
                ] * self.coeffs[self.n, 3] * np.exp(alpha * self.coeffs[self.n, 3])
        return dydx

    def _evalAndDer(self, x):
        """
        Returns the level and first derivative of the function at each value in
        x.  Only called internally by HARKinterpolator1D.eval_and_der (etc).
        """
        if _isscalar(x):
            pos = np.searchsorted(self.x_list, x)
            if pos == 0:
                y = self.coeffs[0, 0] + self.coeffs[0, 1] * (x - self.x_list[0])
                dydx = self.coeffs[0, 1]
            elif pos < self.n:
                alpha = (x - self.x_list[pos - 1]) / (
                    self.x_list[pos] - self.x_list[pos - 1]
                )
                y = self.coeffs[pos, 0] + alpha * (
                    self.coeffs[pos, 1]
                    + alpha * (self.coeffs[pos, 2] + alpha * self.coeffs[pos, 3])
                )
                dydx = (
                    self.coeffs[pos, 1]
                    + alpha
                    * (2 * self.coeffs[pos, 2] + alpha * 3 * self.coeffs[pos, 3])
                ) / (self.x_list[pos] - self.x_list[pos - 1])
            else:
                alpha = x - self.x_list[self.n - 1]
                y = (
                    self.coeffs[pos, 0]
                    + x * self.coeffs[pos, 1]
                    - self.coeffs[pos, 2] * np.exp(alpha * self.coeffs[pos, 3])
                )
                dydx = self.coeffs[pos, 1] - self.coeffs[pos, 2] * self.coeffs[
                    pos, 3
                ] * np.exp(alpha * self.coeffs[pos, 3])
        else:
            m = len(x)
            pos = np.searchsorted(self.x_list, x)
            y = np.zeros(m)
            dydx = np.zeros(m)
            if y.size > 0:
                out_bot = pos == 0
                out_top = pos == self.n
                in_bnds = np.logical_not(np.logical_or(out_bot, out_top))

                # Do the "in bounds" evaluation points
                i = pos[in_bnds]
                coeffs_in = self.coeffs[i, :]
                alpha = (x[in_bnds] - self.x_list[i - 1]) / (
                    self.x_list[i] - self.x_list[i - 1]
                )
                y[in_bnds] = coeffs_in[:, 0] + alpha * (
                    coeffs_in[:, 1]
                    + alpha * (coeffs_in[:, 2] + alpha * coeffs_in[:, 3])
                )
                dydx[in_bnds] = (
                    coeffs_in[:, 1]
                    + alpha * (2 * coeffs_in[:, 2] + alpha * 3 * coeffs_in[:, 3])
                ) / (self.x_list[i] - self.x_list[i - 1])

                # Do the "out of bounds" evaluation points
                y[out_bot] = self.coeffs[0, 0] + self.coeffs[0, 1] * (
                    x[out_bot] - self.x_list[0]
                )
                dydx[out_bot] = self.coeffs[0, 1]
                alpha = x[out_top] - self.x_list[self.n - 1]
                y[out_top] = (
                    self.coeffs[self.n, 0]
                    + x[out_top] * self.coeffs[self.n, 1]
                    - self.coeffs[self.n, 2] * np.exp(alpha * self.coeffs[self.n, 3])
                )
                dydx[out_top] = self.coeffs[self.n, 1] - self.coeffs[
                    self.n, 2
                ] * self.coeffs[self.n, 3] * np.exp(alpha * self.coeffs[self.n, 3])
        return y, dydx


class BilinearInterp(HARKinterpolator2D):
    """
    Bilinear full (or tensor) grid interpolation of a function f(x,y).

    Parameters
    ----------
    f_values : numpy.array
        An array of size (x_n,y_n) such that f_values[i,j] = f(x_list[i],y_list[j])
    x_list : numpy.array
        An array of x values, with length designated x_n.
    y_list : numpy.array
        An array of y values, with length designated y_n.
    xSearchFunc : function
        An optional function that returns the reference location for x values:
        indices = xSearchFunc(x_list,x).  Default is np.searchsorted
    ySearchFunc : function
        An optional function that returns the reference location for y values:
        indices = ySearchFunc(y_list,y).  Default is np.searchsorted
    """

    distance_criteria = ["x_list", "y_list", "f_values"]

    def __init__(self, f_values, x_list, y_list, xSearchFunc=None, ySearchFunc=None):
        self.f_values = f_values
        self.x_list = (
            np.array(x_list)
            if _check_flatten(1, x_list)
            else np.array(x_list).flatten()
        )
        self.y_list = (
            np.array(y_list)
            if _check_flatten(1, y_list)
            else np.array(y_list).flatten()
        )
        _check_grid_dimensions(2, self.f_values, self.x_list, self.y_list)
        self.x_n = x_list.size
        self.y_n = y_list.size
        if xSearchFunc is None:
            xSearchFunc = np.searchsorted
        if ySearchFunc is None:
            ySearchFunc = np.searchsorted
        self.xSearchFunc = xSearchFunc
        self.ySearchFunc = ySearchFunc

    def _evaluate(self, x, y):
        """
        Returns the level of the interpolated function at each value in x,y.
        Only called internally by HARKinterpolator2D.__call__ (etc).
        """
        if _isscalar(x):
            x_pos = max(min(self.xSearchFunc(self.x_list, x), self.x_n - 1), 1)
            y_pos = max(min(self.ySearchFunc(self.y_list, y), self.y_n - 1), 1)
        else:
            x_pos = self.xSearchFunc(self.x_list, x)
            x_pos[x_pos < 1] = 1
            x_pos[x_pos > self.x_n - 1] = self.x_n - 1
            y_pos = self.ySearchFunc(self.y_list, y)
            y_pos[y_pos < 1] = 1
            y_pos[y_pos > self.y_n - 1] = self.y_n - 1
        alpha = (x - self.x_list[x_pos - 1]) / (
            self.x_list[x_pos] - self.x_list[x_pos - 1]
        )
        beta = (y - self.y_list[y_pos - 1]) / (
            self.y_list[y_pos] - self.y_list[y_pos - 1]
        )
        f = (
            (1 - alpha) * (1 - beta) * self.f_values[x_pos - 1, y_pos - 1]
            + (1 - alpha) * beta * self.f_values[x_pos - 1, y_pos]
            + alpha * (1 - beta) * self.f_values[x_pos, y_pos - 1]
            + alpha * beta * self.f_values[x_pos, y_pos]
        )
        return f

    def _derX(self, x, y):
        """
        Returns the derivative with respect to x of the interpolated function
        at each value in x,y. Only called internally by HARKinterpolator2D.derivativeX.
        """
        if _isscalar(x):
            x_pos = max(min(self.xSearchFunc(self.x_list, x), self.x_n - 1), 1)
            y_pos = max(min(self.ySearchFunc(self.y_list, y), self.y_n - 1), 1)
        else:
            x_pos = self.xSearchFunc(self.x_list, x)
            x_pos[x_pos < 1] = 1
            x_pos[x_pos > self.x_n - 1] = self.x_n - 1
            y_pos = self.ySearchFunc(self.y_list, y)
            y_pos[y_pos < 1] = 1
            y_pos[y_pos > self.y_n - 1] = self.y_n - 1
        beta = (y - self.y_list[y_pos - 1]) / (
            self.y_list[y_pos] - self.y_list[y_pos - 1]
        )
        dfdx = (
            (
                (1 - beta) * self.f_values[x_pos, y_pos - 1]
                + beta * self.f_values[x_pos, y_pos]
            )
            - (
                (1 - beta) * self.f_values[x_pos - 1, y_pos - 1]
                + beta * self.f_values[x_pos - 1, y_pos]
            )
        ) / (self.x_list[x_pos] - self.x_list[x_pos - 1])
        return dfdx

    def _derY(self, x, y):
        """
        Returns the derivative with respect to y of the interpolated function
        at each value in x,y. Only called internally by HARKinterpolator2D.derivativeY.
        """
        if _isscalar(x):
            x_pos = max(min(self.xSearchFunc(self.x_list, x), self.x_n - 1), 1)
            y_pos = max(min(self.ySearchFunc(self.y_list, y), self.y_n - 1), 1)
        else:
            x_pos = self.xSearchFunc(self.x_list, x)
            x_pos[x_pos < 1] = 1
            x_pos[x_pos > self.x_n - 1] = self.x_n - 1
            y_pos = self.ySearchFunc(self.y_list, y)
            y_pos[y_pos < 1] = 1
            y_pos[y_pos > self.y_n - 1] = self.y_n - 1
        alpha = (x - self.x_list[x_pos - 1]) / (
            self.x_list[x_pos] - self.x_list[x_pos - 1]
        )
        dfdy = (
            (
                (1 - alpha) * self.f_values[x_pos - 1, y_pos]
                + alpha * self.f_values[x_pos, y_pos]
            )
            - (
                (1 - alpha) * self.f_values[x_pos - 1, y_pos - 1]
                + alpha * self.f_values[x_pos, y_pos - 1]
            )
        ) / (self.y_list[y_pos] - self.y_list[y_pos - 1])
        return dfdy


class TrilinearInterp(HARKinterpolator3D):
    """
    Trilinear full (or tensor) grid interpolation of a function f(x,y,z).

    Parameters
    ----------
    f_values : numpy.array
        An array of size (x_n,y_n,z_n) such that f_values[i,j,k] =
        f(x_list[i],y_list[j],z_list[k])
    x_list : numpy.array
        An array of x values, with length designated x_n.
    y_list : numpy.array
        An array of y values, with length designated y_n.
    z_list : numpy.array
        An array of z values, with length designated z_n.
    xSearchFunc : function
        An optional function that returns the reference location for x values:
        indices = xSearchFunc(x_list,x).  Default is np.searchsorted
    ySearchFunc : function
        An optional function that returns the reference location for y values:
        indices = ySearchFunc(y_list,y).  Default is np.searchsorted
    zSearchFunc : function
        An optional function that returns the reference location for z values:
        indices = zSearchFunc(z_list,z).  Default is np.searchsorted
    """

    distance_criteria = ["f_values", "x_list", "y_list", "z_list"]

    def __init__(
        self,
        f_values,
        x_list,
        y_list,
        z_list,
        xSearchFunc=None,
        ySearchFunc=None,
        zSearchFunc=None,
    ):
        self.f_values = f_values
        self.x_list = (
            np.array(x_list)
            if _check_flatten(1, x_list)
            else np.array(x_list).flatten()
        )
        self.y_list = (
            np.array(y_list)
            if _check_flatten(1, y_list)
            else np.array(y_list).flatten()
        )
        self.z_list = (
            np.array(z_list)
            if _check_flatten(1, z_list)
            else np.array(z_list).flatten()
        )
        _check_grid_dimensions(3, self.f_values, self.x_list, self.y_list, self.z_list)
        self.x_n = x_list.size
        self.y_n = y_list.size
        self.z_n = z_list.size
        if xSearchFunc is None:
            xSearchFunc = np.searchsorted
        if ySearchFunc is None:
            ySearchFunc = np.searchsorted
        if zSearchFunc is None:
            zSearchFunc = np.searchsorted
        self.xSearchFunc = xSearchFunc
        self.ySearchFunc = ySearchFunc
        self.zSearchFunc = zSearchFunc

    def _evaluate(self, x, y, z):
        """
        Returns the level of the interpolated function at each value in x,y,z.
        Only called internally by HARKinterpolator3D.__call__ (etc).
        """
        if _isscalar(x):
            x_pos = max(min(self.xSearchFunc(self.x_list, x), self.x_n - 1), 1)
            y_pos = max(min(self.ySearchFunc(self.y_list, y), self.y_n - 1), 1)
            z_pos = max(min(self.zSearchFunc(self.z_list, z), self.z_n - 1), 1)
        else:
            x_pos = self.xSearchFunc(self.x_list, x)
            x_pos[x_pos < 1] = 1
            x_pos[x_pos > self.x_n - 1] = self.x_n - 1
            y_pos = self.ySearchFunc(self.y_list, y)
            y_pos[y_pos < 1] = 1
            y_pos[y_pos > self.y_n - 1] = self.y_n - 1
            z_pos = self.zSearchFunc(self.z_list, z)
            z_pos[z_pos < 1] = 1
            z_pos[z_pos > self.z_n - 1] = self.z_n - 1
        alpha = (x - self.x_list[x_pos - 1]) / (
            self.x_list[x_pos] - self.x_list[x_pos - 1]
        )
        beta = (y - self.y_list[y_pos - 1]) / (
            self.y_list[y_pos] - self.y_list[y_pos - 1]
        )
        gamma = (z - self.z_list[z_pos - 1]) / (
            self.z_list[z_pos] - self.z_list[z_pos - 1]
        )
        f = (
            (1 - alpha)
            * (1 - beta)
            * (1 - gamma)
            * self.f_values[x_pos - 1, y_pos - 1, z_pos - 1]
            + (1 - alpha)
            * (1 - beta)
            * gamma
            * self.f_values[x_pos - 1, y_pos - 1, z_pos]
            + (1 - alpha)
            * beta
            * (1 - gamma)
            * self.f_values[x_pos - 1, y_pos, z_pos - 1]
            + (1 - alpha) * beta * gamma * self.f_values[x_pos - 1, y_pos, z_pos]
            + alpha
            * (1 - beta)
            * (1 - gamma)
            * self.f_values[x_pos, y_pos - 1, z_pos - 1]
            + alpha * (1 - beta) * gamma * self.f_values[x_pos, y_pos - 1, z_pos]
            + alpha * beta * (1 - gamma) * self.f_values[x_pos, y_pos, z_pos - 1]
            + alpha * beta * gamma * self.f_values[x_pos, y_pos, z_pos]
        )
        return f

    def _derX(self, x, y, z):
        """
        Returns the derivative with respect to x of the interpolated function
        at each value in x,y,z. Only called internally by HARKinterpolator3D.derivativeX.
        """
        if _isscalar(x):
            x_pos = max(min(self.xSearchFunc(self.x_list, x), self.x_n - 1), 1)
            y_pos = max(min(self.ySearchFunc(self.y_list, y), self.y_n - 1), 1)
            z_pos = max(min(self.zSearchFunc(self.z_list, z), self.z_n - 1), 1)
        else:
            x_pos = self.xSearchFunc(self.x_list, x)
            x_pos[x_pos < 1] = 1
            x_pos[x_pos > self.x_n - 1] = self.x_n - 1
            y_pos = self.ySearchFunc(self.y_list, y)
            y_pos[y_pos < 1] = 1
            y_pos[y_pos > self.y_n - 1] = self.y_n - 1
            z_pos = self.zSearchFunc(self.z_list, z)
            z_pos[z_pos < 1] = 1
            z_pos[z_pos > self.z_n - 1] = self.z_n - 1
        beta = (y - self.y_list[y_pos - 1]) / (
            self.y_list[y_pos] - self.y_list[y_pos - 1]
        )
        gamma = (z - self.z_list[z_pos - 1]) / (
            self.z_list[z_pos] - self.z_list[z_pos - 1]
        )
        dfdx = (
            (
                (1 - beta) * (1 - gamma) * self.f_values[x_pos, y_pos - 1, z_pos - 1]
                + (1 - beta) * gamma * self.f_values[x_pos, y_pos - 1, z_pos]
                + beta * (1 - gamma) * self.f_values[x_pos, y_pos, z_pos - 1]
                + beta * gamma * self.f_values[x_pos, y_pos, z_pos]
            )
            - (
                (1 - beta)
                * (1 - gamma)
                * self.f_values[x_pos - 1, y_pos - 1, z_pos - 1]
                + (1 - beta) * gamma * self.f_values[x_pos - 1, y_pos - 1, z_pos]
                + beta * (1 - gamma) * self.f_values[x_pos - 1, y_pos, z_pos - 1]
                + beta * gamma * self.f_values[x_pos - 1, y_pos, z_pos]
            )
        ) / (self.x_list[x_pos] - self.x_list[x_pos - 1])
        return dfdx

    def _derY(self, x, y, z):
        """
        Returns the derivative with respect to y of the interpolated function
        at each value in x,y,z. Only called internally by HARKinterpolator3D.derivativeY.
        """
        if _isscalar(x):
            x_pos = max(min(self.xSearchFunc(self.x_list, x), self.x_n - 1), 1)
            y_pos = max(min(self.ySearchFunc(self.y_list, y), self.y_n - 1), 1)
            z_pos = max(min(self.zSearchFunc(self.z_list, z), self.z_n - 1), 1)
        else:
            x_pos = self.xSearchFunc(self.x_list, x)
            x_pos[x_pos < 1] = 1
            x_pos[x_pos > self.x_n - 1] = self.x_n - 1
            y_pos = self.ySearchFunc(self.y_list, y)
            y_pos[y_pos < 1] = 1
            y_pos[y_pos > self.y_n - 1] = self.y_n - 1
            z_pos = self.zSearchFunc(self.z_list, z)
            z_pos[z_pos < 1] = 1
            z_pos[z_pos > self.z_n - 1] = self.z_n - 1
        alpha = (x - self.x_list[x_pos - 1]) / (
            self.x_list[x_pos] - self.x_list[x_pos - 1]
        )
        gamma = (z - self.z_list[z_pos - 1]) / (
            self.z_list[z_pos] - self.z_list[z_pos - 1]
        )
        dfdy = (
            (
                (1 - alpha) * (1 - gamma) * self.f_values[x_pos - 1, y_pos, z_pos - 1]
                + (1 - alpha) * gamma * self.f_values[x_pos - 1, y_pos, z_pos]
                + alpha * (1 - gamma) * self.f_values[x_pos, y_pos, z_pos - 1]
                + alpha * gamma * self.f_values[x_pos, y_pos, z_pos]
            )
            - (
                (1 - alpha)
                * (1 - gamma)
                * self.f_values[x_pos - 1, y_pos - 1, z_pos - 1]
                + (1 - alpha) * gamma * self.f_values[x_pos - 1, y_pos - 1, z_pos]
                + alpha * (1 - gamma) * self.f_values[x_pos, y_pos - 1, z_pos - 1]
                + alpha * gamma * self.f_values[x_pos, y_pos - 1, z_pos]
            )
        ) / (self.y_list[y_pos] - self.y_list[y_pos - 1])
        return dfdy

    def _derZ(self, x, y, z):
        """
        Returns the derivative with respect to z of the interpolated function
        at each value in x,y,z. Only called internally by HARKinterpolator3D.derivativeZ.
        """
        if _isscalar(x):
            x_pos = max(min(self.xSearchFunc(self.x_list, x), self.x_n - 1), 1)
            y_pos = max(min(self.ySearchFunc(self.y_list, y), self.y_n - 1), 1)
            z_pos = max(min(self.zSearchFunc(self.z_list, z), self.z_n - 1), 1)
        else:
            x_pos = self.xSearchFunc(self.x_list, x)
            x_pos[x_pos < 1] = 1
            x_pos[x_pos > self.x_n - 1] = self.x_n - 1
            y_pos = self.ySearchFunc(self.y_list, y)
            y_pos[y_pos < 1] = 1
            y_pos[y_pos > self.y_n - 1] = self.y_n - 1
            z_pos = self.zSearchFunc(self.z_list, z)
            z_pos[z_pos < 1] = 1
            z_pos[z_pos > self.z_n - 1] = self.z_n - 1
        alpha = (x - self.x_list[x_pos - 1]) / (
            self.x_list[x_pos] - self.x_list[x_pos - 1]
        )
        beta = (y - self.y_list[y_pos - 1]) / (
            self.y_list[y_pos] - self.y_list[y_pos - 1]
        )
        dfdz = (
            (
                (1 - alpha) * (1 - beta) * self.f_values[x_pos - 1, y_pos - 1, z_pos]
                + (1 - alpha) * beta * self.f_values[x_pos - 1, y_pos, z_pos]
                + alpha * (1 - beta) * self.f_values[x_pos, y_pos - 1, z_pos]
                + alpha * beta * self.f_values[x_pos, y_pos, z_pos]
            )
            - (
                (1 - alpha)
                * (1 - beta)
                * self.f_values[x_pos - 1, y_pos - 1, z_pos - 1]
                + (1 - alpha) * beta * self.f_values[x_pos - 1, y_pos, z_pos - 1]
                + alpha * (1 - beta) * self.f_values[x_pos, y_pos - 1, z_pos - 1]
                + alpha * beta * self.f_values[x_pos, y_pos, z_pos - 1]
            )
        ) / (self.z_list[z_pos] - self.z_list[z_pos - 1])
        return dfdz


class QuadlinearInterp(HARKinterpolator4D):
    """
    Quadlinear full (or tensor) grid interpolation of a function f(w,x,y,z).

    Parameters
    ----------
    f_values : numpy.array
        An array of size (w_n,x_n,y_n,z_n) such that f_values[i,j,k,l] =
        f(w_list[i],x_list[j],y_list[k],z_list[l])
    w_list : numpy.array
        An array of x values, with length designated w_n.
    x_list : numpy.array
        An array of x values, with length designated x_n.
    y_list : numpy.array
        An array of y values, with length designated y_n.
    z_list : numpy.array
        An array of z values, with length designated z_n.
    wSearchFunc : function
        An optional function that returns the reference location for w values:
        indices = wSearchFunc(w_list,w).  Default is np.searchsorted
    xSearchFunc : function
        An optional function that returns the reference location for x values:
        indices = xSearchFunc(x_list,x).  Default is np.searchsorted
    ySearchFunc : function
        An optional function that returns the reference location for y values:
        indices = ySearchFunc(y_list,y).  Default is np.searchsorted
    zSearchFunc : function
        An optional function that returns the reference location for z values:
        indices = zSearchFunc(z_list,z).  Default is np.searchsorted
    """

    distance_criteria = ["f_values", "w_list", "x_list", "y_list", "z_list"]

    def __init__(
        self,
        f_values,
        w_list,
        x_list,
        y_list,
        z_list,
        wSearchFunc=None,
        xSearchFunc=None,
        ySearchFunc=None,
        zSearchFunc=None,
    ):
        self.f_values = f_values
        self.w_list = (
            np.array(w_list)
            if _check_flatten(1, w_list)
            else np.array(w_list).flatten()
        )
        self.x_list = (
            np.array(x_list)
            if _check_flatten(1, x_list)
            else np.array(x_list).flatten()
        )
        self.y_list = (
            np.array(y_list)
            if _check_flatten(1, y_list)
            else np.array(y_list).flatten()
        )
        self.z_list = (
            np.array(z_list)
            if _check_flatten(1, z_list)
            else np.array(z_list).flatten()
        )
        _check_grid_dimensions(
            4, self.f_values, self.w_list, self.x_list, self.y_list, self.z_list
        )
        self.w_n = w_list.size
        self.x_n = x_list.size
        self.y_n = y_list.size
        self.z_n = z_list.size
        if wSearchFunc is None:
            wSearchFunc = np.searchsorted
        if xSearchFunc is None:
            xSearchFunc = np.searchsorted
        if ySearchFunc is None:
            ySearchFunc = np.searchsorted
        if zSearchFunc is None:
            zSearchFunc = np.searchsorted
        self.wSearchFunc = wSearchFunc
        self.xSearchFunc = xSearchFunc
        self.ySearchFunc = ySearchFunc
        self.zSearchFunc = zSearchFunc

    def _evaluate(self, w, x, y, z):
        """
        Returns the level of the interpolated function at each value in x,y,z.
        Only called internally by HARKinterpolator4D.__call__ (etc).
        """
        if _isscalar(w):
            w_pos = max(min(self.wSearchFunc(self.w_list, w), self.w_n - 1), 1)
            x_pos = max(min(self.xSearchFunc(self.x_list, x), self.x_n - 1), 1)
            y_pos = max(min(self.ySearchFunc(self.y_list, y), self.y_n - 1), 1)
            z_pos = max(min(self.zSearchFunc(self.z_list, z), self.z_n - 1), 1)
        else:
            w_pos = self.wSearchFunc(self.w_list, w)
            w_pos[w_pos < 1] = 1
            w_pos[w_pos > self.w_n - 1] = self.w_n - 1
            x_pos = self.xSearchFunc(self.x_list, x)
            x_pos[x_pos < 1] = 1
            x_pos[x_pos > self.x_n - 1] = self.x_n - 1
            y_pos = self.ySearchFunc(self.y_list, y)
            y_pos[y_pos < 1] = 1
            y_pos[y_pos > self.y_n - 1] = self.y_n - 1
            z_pos = self.zSearchFunc(self.z_list, z)
            z_pos[z_pos < 1] = 1
            z_pos[z_pos > self.z_n - 1] = self.z_n - 1
        i = w_pos  # for convenience
        j = x_pos
        k = y_pos
        l = z_pos
        alpha = (w - self.w_list[i - 1]) / (self.w_list[i] - self.w_list[i - 1])
        beta = (x - self.x_list[j - 1]) / (self.x_list[j] - self.x_list[j - 1])
        gamma = (y - self.y_list[k - 1]) / (self.y_list[k] - self.y_list[k - 1])
        delta = (z - self.z_list[l - 1]) / (self.z_list[l] - self.z_list[l - 1])
        f = (1 - alpha) * (
            (1 - beta)
            * (
                (1 - gamma) * (1 - delta) * self.f_values[i - 1, j - 1, k - 1, l - 1]
                + (1 - gamma) * delta * self.f_values[i - 1, j - 1, k - 1, l]
                + gamma * (1 - delta) * self.f_values[i - 1, j - 1, k, l - 1]
                + gamma * delta * self.f_values[i - 1, j - 1, k, l]
            )
            + beta
            * (
                (1 - gamma) * (1 - delta) * self.f_values[i - 1, j, k - 1, l - 1]
                + (1 - gamma) * delta * self.f_values[i - 1, j, k - 1, l]
                + gamma * (1 - delta) * self.f_values[i - 1, j, k, l - 1]
                + gamma * delta * self.f_values[i - 1, j, k, l]
            )
        ) + alpha * (
            (1 - beta)
            * (
                (1 - gamma) * (1 - delta) * self.f_values[i, j - 1, k - 1, l - 1]
                + (1 - gamma) * delta * self.f_values[i, j - 1, k - 1, l]
                + gamma * (1 - delta) * self.f_values[i, j - 1, k, l - 1]
                + gamma * delta * self.f_values[i, j - 1, k, l]
            )
            + beta
            * (
                (1 - gamma) * (1 - delta) * self.f_values[i, j, k - 1, l - 1]
                + (1 - gamma) * delta * self.f_values[i, j, k - 1, l]
                + gamma * (1 - delta) * self.f_values[i, j, k, l - 1]
                + gamma * delta * self.f_values[i, j, k, l]
            )
        )
        return f

    def _derW(self, w, x, y, z):
        """
        Returns the derivative with respect to w of the interpolated function
        at each value in w,x,y,z. Only called internally by HARKinterpolator4D.derivativeW.
        """
        if _isscalar(w):
            w_pos = max(min(self.wSearchFunc(self.w_list, w), self.w_n - 1), 1)
            x_pos = max(min(self.xSearchFunc(self.x_list, x), self.x_n - 1), 1)
            y_pos = max(min(self.ySearchFunc(self.y_list, y), self.y_n - 1), 1)
            z_pos = max(min(self.zSearchFunc(self.z_list, z), self.z_n - 1), 1)
        else:
            w_pos = self.wSearchFunc(self.w_list, w)
            w_pos[w_pos < 1] = 1
            w_pos[w_pos > self.w_n - 1] = self.w_n - 1
            x_pos = self.xSearchFunc(self.x_list, x)
            x_pos[x_pos < 1] = 1
            x_pos[x_pos > self.x_n - 1] = self.x_n - 1
            y_pos = self.ySearchFunc(self.y_list, y)
            y_pos[y_pos < 1] = 1
            y_pos[y_pos > self.y_n - 1] = self.y_n - 1
            z_pos = self.zSearchFunc(self.z_list, z)
            z_pos[z_pos < 1] = 1
            z_pos[z_pos > self.z_n - 1] = self.z_n - 1
        i = w_pos  # for convenience
        j = x_pos
        k = y_pos
        l = z_pos
        beta = (x - self.x_list[j - 1]) / (self.x_list[j] - self.x_list[j - 1])
        gamma = (y - self.y_list[k - 1]) / (self.y_list[k] - self.y_list[k - 1])
        delta = (z - self.z_list[l - 1]) / (self.z_list[l] - self.z_list[l - 1])
        dfdw = (
            (
                (1 - beta)
                * (1 - gamma)
                * (1 - delta)
                * self.f_values[i, j - 1, k - 1, l - 1]
                + (1 - beta) * (1 - gamma) * delta * self.f_values[i, j - 1, k - 1, l]
                + (1 - beta) * gamma * (1 - delta) * self.f_values[i, j - 1, k, l - 1]
                + (1 - beta) * gamma * delta * self.f_values[i, j - 1, k, l]
                + beta * (1 - gamma) * (1 - delta) * self.f_values[i, j, k - 1, l - 1]
                + beta * (1 - gamma) * delta * self.f_values[i, j, k - 1, l]
                + beta * gamma * (1 - delta) * self.f_values[i, j, k, l - 1]
                + beta * gamma * delta * self.f_values[i, j, k, l]
            )
            - (
                (1 - beta)
                * (1 - gamma)
                * (1 - delta)
                * self.f_values[i - 1, j - 1, k - 1, l - 1]
                + (1 - beta)
                * (1 - gamma)
                * delta
                * self.f_values[i - 1, j - 1, k - 1, l]
                + (1 - beta)
                * gamma
                * (1 - delta)
                * self.f_values[i - 1, j - 1, k, l - 1]
                + (1 - beta) * gamma * delta * self.f_values[i - 1, j - 1, k, l]
                + beta
                * (1 - gamma)
                * (1 - delta)
                * self.f_values[i - 1, j, k - 1, l - 1]
                + beta * (1 - gamma) * delta * self.f_values[i - 1, j, k - 1, l]
                + beta * gamma * (1 - delta) * self.f_values[i - 1, j, k, l - 1]
                + beta * gamma * delta * self.f_values[i - 1, j, k, l]
            )
        ) / (self.w_list[i] - self.w_list[i - 1])
        return dfdw

    def _derX(self, w, x, y, z):
        """
        Returns the derivative with respect to x of the interpolated function
        at each value in w,x,y,z. Only called internally by HARKinterpolator4D.derivativeX.
        """
        if _isscalar(w):
            w_pos = max(min(self.wSearchFunc(self.w_list, w), self.w_n - 1), 1)
            x_pos = max(min(self.xSearchFunc(self.x_list, x), self.x_n - 1), 1)
            y_pos = max(min(self.ySearchFunc(self.y_list, y), self.y_n - 1), 1)
            z_pos = max(min(self.zSearchFunc(self.z_list, z), self.z_n - 1), 1)
        else:
            w_pos = self.wSearchFunc(self.w_list, w)
            w_pos[w_pos < 1] = 1
            w_pos[w_pos > self.w_n - 1] = self.w_n - 1
            x_pos = self.xSearchFunc(self.x_list, x)
            x_pos[x_pos < 1] = 1
            x_pos[x_pos > self.x_n - 1] = self.x_n - 1
            y_pos = self.ySearchFunc(self.y_list, y)
            y_pos[y_pos < 1] = 1
            y_pos[y_pos > self.y_n - 1] = self.y_n - 1
            z_pos = self.zSearchFunc(self.z_list, z)
            z_pos[z_pos < 1] = 1
            z_pos[z_pos > self.z_n - 1] = self.z_n - 1
        i = w_pos  # for convenience
        j = x_pos
        k = y_pos
        l = z_pos
        alpha = (w - self.w_list[i - 1]) / (self.w_list[i] - self.w_list[i - 1])
        gamma = (y - self.y_list[k - 1]) / (self.y_list[k] - self.y_list[k - 1])
        delta = (z - self.z_list[l - 1]) / (self.z_list[l] - self.z_list[l - 1])
        dfdx = (
            (
                (1 - alpha)
                * (1 - gamma)
                * (1 - delta)
                * self.f_values[i - 1, j, k - 1, l - 1]
                + (1 - alpha) * (1 - gamma) * delta * self.f_values[i - 1, j, k - 1, l]
                + (1 - alpha) * gamma * (1 - delta) * self.f_values[i - 1, j, k, l - 1]
                + (1 - alpha) * gamma * delta * self.f_values[i - 1, j, k, l]
                + alpha * (1 - gamma) * (1 - delta) * self.f_values[i, j, k - 1, l - 1]
                + alpha * (1 - gamma) * delta * self.f_values[i, j, k - 1, l]
                + alpha * gamma * (1 - delta) * self.f_values[i, j, k, l - 1]
                + alpha * gamma * delta * self.f_values[i, j, k, l]
            )
            - (
                (1 - alpha)
                * (1 - gamma)
                * (1 - delta)
                * self.f_values[i - 1, j - 1, k - 1, l - 1]
                + (1 - alpha)
                * (1 - gamma)
                * delta
                * self.f_values[i - 1, j - 1, k - 1, l]
                + (1 - alpha)
                * gamma
                * (1 - delta)
                * self.f_values[i - 1, j - 1, k, l - 1]
                + (1 - alpha) * gamma * delta * self.f_values[i - 1, j - 1, k, l]
                + alpha
                * (1 - gamma)
                * (1 - delta)
                * self.f_values[i, j - 1, k - 1, l - 1]
                + alpha * (1 - gamma) * delta * self.f_values[i, j - 1, k - 1, l]
                + alpha * gamma * (1 - delta) * self.f_values[i, j - 1, k, l - 1]
                + alpha * gamma * delta * self.f_values[i, j - 1, k, l]
            )
        ) / (self.x_list[j] - self.x_list[j - 1])
        return dfdx

    def _derY(self, w, x, y, z):
        """
        Returns the derivative with respect to y of the interpolated function
        at each value in w,x,y,z. Only called internally by HARKinterpolator4D.derivativeY.
        """
        if _isscalar(w):
            w_pos = max(min(self.wSearchFunc(self.w_list, w), self.w_n - 1), 1)
            x_pos = max(min(self.xSearchFunc(self.x_list, x), self.x_n - 1), 1)
            y_pos = max(min(self.ySearchFunc(self.y_list, y), self.y_n - 1), 1)
            z_pos = max(min(self.zSearchFunc(self.z_list, z), self.z_n - 1), 1)
        else:
            w_pos = self.wSearchFunc(self.w_list, w)
            w_pos[w_pos < 1] = 1
            w_pos[w_pos > self.w_n - 1] = self.w_n - 1
            x_pos = self.xSearchFunc(self.x_list, x)
            x_pos[x_pos < 1] = 1
            x_pos[x_pos > self.x_n - 1] = self.x_n - 1
            y_pos = self.ySearchFunc(self.y_list, y)
            y_pos[y_pos < 1] = 1
            y_pos[y_pos > self.y_n - 1] = self.y_n - 1
            z_pos = self.zSearchFunc(self.z_list, z)
            z_pos[z_pos < 1] = 1
            z_pos[z_pos > self.z_n - 1] = self.z_n - 1
        i = w_pos  # for convenience
        j = x_pos
        k = y_pos
        l = z_pos
        alpha = (w - self.w_list[i - 1]) / (self.w_list[i] - self.w_list[i - 1])
        beta = (x - self.x_list[j - 1]) / (self.x_list[j] - self.x_list[j - 1])
        delta = (z - self.z_list[l - 1]) / (self.z_list[l] - self.z_list[l - 1])
        dfdy = (
            (
                (1 - alpha)
                * (1 - beta)
                * (1 - delta)
                * self.f_values[i - 1, j - 1, k, l - 1]
                + (1 - alpha) * (1 - beta) * delta * self.f_values[i - 1, j - 1, k, l]
                + (1 - alpha) * beta * (1 - delta) * self.f_values[i - 1, j, k, l - 1]
                + (1 - alpha) * beta * delta * self.f_values[i - 1, j, k, l]
                + alpha * (1 - beta) * (1 - delta) * self.f_values[i, j - 1, k, l - 1]
                + alpha * (1 - beta) * delta * self.f_values[i, j - 1, k, l]
                + alpha * beta * (1 - delta) * self.f_values[i, j, k, l - 1]
                + alpha * beta * delta * self.f_values[i, j, k, l]
            )
            - (
                (1 - alpha)
                * (1 - beta)
                * (1 - delta)
                * self.f_values[i - 1, j - 1, k - 1, l - 1]
                + (1 - alpha)
                * (1 - beta)
                * delta
                * self.f_values[i - 1, j - 1, k - 1, l]
                + (1 - alpha)
                * beta
                * (1 - delta)
                * self.f_values[i - 1, j, k - 1, l - 1]
                + (1 - alpha) * beta * delta * self.f_values[i - 1, j, k - 1, l]
                + alpha
                * (1 - beta)
                * (1 - delta)
                * self.f_values[i, j - 1, k - 1, l - 1]
                + alpha * (1 - beta) * delta * self.f_values[i, j - 1, k - 1, l]
                + alpha * beta * (1 - delta) * self.f_values[i, j, k - 1, l - 1]
                + alpha * beta * delta * self.f_values[i, j, k - 1, l]
            )
        ) / (self.y_list[k] - self.y_list[k - 1])
        return dfdy

    def _derZ(self, w, x, y, z):
        """
        Returns the derivative with respect to z of the interpolated function
        at each value in w,x,y,z. Only called internally by HARKinterpolator4D.derivativeZ.
        """
        if _isscalar(w):
            w_pos = max(min(self.wSearchFunc(self.w_list, w), self.w_n - 1), 1)
            x_pos = max(min(self.xSearchFunc(self.x_list, x), self.x_n - 1), 1)
            y_pos = max(min(self.ySearchFunc(self.y_list, y), self.y_n - 1), 1)
            z_pos = max(min(self.zSearchFunc(self.z_list, z), self.z_n - 1), 1)
        else:
            w_pos = self.wSearchFunc(self.w_list, w)
            w_pos[w_pos < 1] = 1
            w_pos[w_pos > self.w_n - 1] = self.w_n - 1
            x_pos = self.xSearchFunc(self.x_list, x)
            x_pos[x_pos < 1] = 1
            x_pos[x_pos > self.x_n - 1] = self.x_n - 1
            y_pos = self.ySearchFunc(self.y_list, y)
            y_pos[y_pos < 1] = 1
            y_pos[y_pos > self.y_n - 1] = self.y_n - 1
            z_pos = self.zSearchFunc(self.z_list, z)
            z_pos[z_pos < 1] = 1
            z_pos[z_pos > self.z_n - 1] = self.z_n - 1
        i = w_pos  # for convenience
        j = x_pos
        k = y_pos
        l = z_pos
        alpha = (w - self.w_list[i - 1]) / (self.w_list[i] - self.w_list[i - 1])
        beta = (x - self.x_list[j - 1]) / (self.x_list[j] - self.x_list[j - 1])
        gamma = (y - self.y_list[k - 1]) / (self.y_list[k] - self.y_list[k - 1])
        dfdz = (
            (
                (1 - alpha)
                * (1 - beta)
                * (1 - gamma)
                * self.f_values[i - 1, j - 1, k - 1, l]
                + (1 - alpha) * (1 - beta) * gamma * self.f_values[i - 1, j - 1, k, l]
                + (1 - alpha) * beta * (1 - gamma) * self.f_values[i - 1, j, k - 1, l]
                + (1 - alpha) * beta * gamma * self.f_values[i - 1, j, k, l]
                + alpha * (1 - beta) * (1 - gamma) * self.f_values[i, j - 1, k - 1, l]
                + alpha * (1 - beta) * gamma * self.f_values[i, j - 1, k, l]
                + alpha * beta * (1 - gamma) * self.f_values[i, j, k - 1, l]
                + alpha * beta * gamma * self.f_values[i, j, k, l]
            )
            - (
                (1 - alpha)
                * (1 - beta)
                * (1 - gamma)
                * self.f_values[i - 1, j - 1, k - 1, l - 1]
                + (1 - alpha)
                * (1 - beta)
                * gamma
                * self.f_values[i - 1, j - 1, k, l - 1]
                + (1 - alpha)
                * beta
                * (1 - gamma)
                * self.f_values[i - 1, j, k - 1, l - 1]
                + (1 - alpha) * beta * gamma * self.f_values[i - 1, j, k, l - 1]
                + alpha
                * (1 - beta)
                * (1 - gamma)
                * self.f_values[i, j - 1, k - 1, l - 1]
                + alpha * (1 - beta) * gamma * self.f_values[i, j - 1, k, l - 1]
                + alpha * beta * (1 - gamma) * self.f_values[i, j, k - 1, l - 1]
                + alpha * beta * gamma * self.f_values[i, j, k, l - 1]
            )
        ) / (self.z_list[l] - self.z_list[l - 1])
        return dfdz


class LowerEnvelope(HARKinterpolator1D):
    """
    The lower envelope of a finite set of 1D functions, each of which can be of
    any class that has the methods __call__, derivative, and eval_with_derivative.
    Generally: it combines HARKinterpolator1Ds.

    Parameters
    ----------
    *functions : function
        Any number of real functions; often instances of HARKinterpolator1D
    nan_bool : boolean
        An indicator for whether the solver should exclude NA's when 
        forming the lower envelope
    """

    distance_criteria = ["functions"]

    def __init__(self, *functions, nan_bool=True):

        if nan_bool:
            self.compare = np.nanmin
            self.argcompare = np.nanargmin
        else:
            self.compare = np.min
            self.argcompare = np.argmin

        self.functions = []
        for function in functions:
            self.functions.append(function)
        self.funcCount = len(self.functions)

    def _evaluate(self, x):
        """
        Returns the level of the function at each value in x as the minimum among
        all of the functions.  Only called internally by HARKinterpolator1D.__call__.
        """

        if _isscalar(x):
            y = self.compare([f(x) for f in self.functions])
        else:
            m = len(x)
            fx = np.zeros((m, self.funcCount))
            for j in range(self.funcCount):
                fx[:, j] = self.functions[j](x)
            y = self.compare(fx, axis=1)
        return y

    def _der(self, x):
        """
        Returns the first derivative of the function at each value in x.  Only
        called internally by HARKinterpolator1D.derivative.
        """
        y, dydx = self._evalAndDer(x)
        return dydx  # Sadly, this is the fastest / most convenient way...

    def _evalAndDer(self, x):
        """
        Returns the level and first derivative of the function at each value in
        x.  Only called internally by HARKinterpolator1D.eval_and_der.
        """
        m = len(x)
        fx = np.zeros((m, self.funcCount))
        for j in range(self.funcCount):
            fx[:, j] = self.functions[j](x)
        i = self.argcompare(fx, axis=1)
        y = fx[np.arange(m), i]
        dydx = np.zeros_like(y)
        for j in range(self.funcCount):
            c = i == j
            dydx[c] = self.functions[j].derivative(x[c])
        return y, dydx


class UpperEnvelope(HARKinterpolator1D):
    """
    The upper envelope of a finite set of 1D functions, each of which can be of
    any class that has the methods __call__, derivative, and eval_with_derivative.
    Generally: it combines HARKinterpolator1Ds.

    Parameters
    ----------
    *functions : function
        Any number of real functions; often instances of HARKinterpolator1D
    nan_bool : boolean	
        An indicator for whether the solver should exclude NA's when forming	
        the lower envelope.
    """

    distance_criteria = ["functions"]

    def __init__(self, *functions, nan_bool=True):
        if nan_bool:
            self.compare = np.nanmax
            self.argcompare = np.nanargmax
        else:
            self.compare = np.max
            self.argcompare = np.argmax
        self.functions = []
        for function in functions:
            self.functions.append(function)
        self.funcCount = len(self.functions)

    def _evaluate(self, x):
        """
        Returns the level of the function at each value in x as the maximum among
        all of the functions.  Only called internally by HARKinterpolator1D.__call__.
        """
        if _isscalar(x):
            y = self.compare([f(x) for f in self.functions])
        else:
            m = len(x)
            fx = np.zeros((m, self.funcCount))
            for j in range(self.funcCount):
                fx[:, j] = self.functions[j](x)
            y = self.compare(fx, axis=1)
        return y

    def _der(self, x):
        """
        Returns the first derivative of the function at each value in x.  Only
        called internally by HARKinterpolator1D.derivative.
        """
        y, dydx = self._evalAndDer(x)
        return dydx  # Sadly, this is the fastest / most convenient way...

    def _evalAndDer(self, x):
        """
        Returns the level and first derivative of the function at each value in
        x.  Only called internally by HARKinterpolator1D.eval_and_der.
        """
        m = len(x)
        fx = np.zeros((m, self.funcCount))
        for j in range(self.funcCount):
            fx[:, j] = self.functions[j](x)
        i = self.argcompare(fx, axis=1)
        y = fx[np.arange(m), i]
        dydx = np.zeros_like(y)
        for j in range(self.funcCount):
            c = i == j
            dydx[c] = self.functions[j].derivative(x[c])
        return y, dydx


class LowerEnvelope2D(HARKinterpolator2D):
    """
    The lower envelope of a finite set of 2D functions, each of which can be of
    any class that has the methods __call__, derivativeX, and derivativeY.
    Generally: it combines HARKinterpolator2Ds.

    Parameters
    ----------
    *functions : function
        Any number of real functions; often instances of HARKinterpolator2D
    nan_bool : boolean	
        An indicator for whether the solver should exclude NA's when forming	
        the lower envelope.
    """

    distance_criteria = ["functions"]

    def __init__(self, *functions, nan_bool=True):
        if nan_bool:
            self.compare = np.nanmin
            self.argcompare = np.nanargmin
        else:
            self.compare = np.min
            self.argcompare = np.argmin
        self.functions = []
        for function in functions:
            self.functions.append(function)
        self.funcCount = len(self.functions)

    def _evaluate(self, x, y):
        """
        Returns the level of the function at each value in (x,y) as the minimum
        among all of the functions.  Only called internally by
        HARKinterpolator2D.__call__.
        """
        if _isscalar(x):
            f = self.compare([f(x, y) for f in self.functions])
        else:
            m = len(x)
            temp = np.zeros((m, self.funcCount))
            for j in range(self.funcCount):
                temp[:, j] = self.functions[j](x, y)
            f = self.compare(temp, axis=1)
        return f

    def _derX(self, x, y):
        """
        Returns the first derivative of the function with respect to X at each
        value in (x,y).  Only called internally by HARKinterpolator2D._derX.
        """
        m = len(x)
        temp = np.zeros((m, self.funcCount))
        for j in range(self.funcCount):
            temp[:, j] = self.functions[j](x, y)
        i = self.argcompare(temp, axis=1)
        dfdx = np.zeros_like(x)
        for j in range(self.funcCount):
            c = i == j
            dfdx[c] = self.functions[j].derivativeX(x[c], y[c])
        return dfdx

    def _derY(self, x, y):
        """
        Returns the first derivative of the function with respect to Y at each
        value in (x,y).  Only called internally by HARKinterpolator2D._derY.
        """
        m = len(x)
        temp = np.zeros((m, self.funcCount))
        for j in range(self.funcCount):
            temp[:, j] = self.functions[j](x, y)
        i = self.argcompare(temp, axis=1)
        y = temp[np.arange(m), i]
        dfdy = np.zeros_like(x)
        for j in range(self.funcCount):
            c = i == j
            dfdy[c] = self.functions[j].derivativeY(x[c], y[c])
        return dfdy


class LowerEnvelope3D(HARKinterpolator3D):
    """
    The lower envelope of a finite set of 3D functions, each of which can be of
    any class that has the methods __call__, derivativeX, derivativeY, and
    derivativeZ. Generally: it combines HARKinterpolator2Ds.

    Parameters
    ----------
    *functions : function
        Any number of real functions; often instances of HARKinterpolator3D
    nan_bool : boolean	
        An indicator for whether the solver should exclude NA's when forming	
        the lower envelope.
    """

    distance_criteria = ["functions"]

    def __init__(self, *functions, nan_bool=True):
        if nan_bool:
            self.compare = np.nanmin
            self.argcompare = np.nanargmin
        else:
            self.compare = np.min
            self.argcompare = np.argmin
        self.functions = []
        for function in functions:
            self.functions.append(function)
        self.funcCount = len(self.functions)

    def _evaluate(self, x, y, z):
        """
        Returns the level of the function at each value in (x,y,z) as the minimum
        among all of the functions.  Only called internally by
        HARKinterpolator3D.__call__.
        """
        if _isscalar(x):
            f = self.compare([f(x, y, z) for f in self.functions])
        else:
            m = len(x)
            temp = np.zeros((m, self.funcCount))
            for j in range(self.funcCount):
                temp[:, j] = self.functions[j](x, y, z)
            f = self.compare(temp, axis=1)
        return f

    def _derX(self, x, y, z):
        """
        Returns the first derivative of the function with respect to X at each
        value in (x,y,z).  Only called internally by HARKinterpolator3D._derX.
        """
        m = len(x)
        temp = np.zeros((m, self.funcCount))
        for j in range(self.funcCount):
            temp[:, j] = self.functions[j](x, y, z)
        i = self.argcompare(temp, axis=1)
        dfdx = np.zeros_like(x)
        for j in range(self.funcCount):
            c = i == j
            dfdx[c] = self.functions[j].derivativeX(x[c], y[c], z[c])
        return dfdx

    def _derY(self, x, y, z):
        """
        Returns the first derivative of the function with respect to Y at each
        value in (x,y,z).  Only called internally by HARKinterpolator3D._derY.
        """
        m = len(x)
        temp = np.zeros((m, self.funcCount))
        for j in range(self.funcCount):
            temp[:, j] = self.functions[j](x, y, z)
        i = self.argcompare(temp, axis=1)
        y = temp[np.arange(m), i]
        dfdy = np.zeros_like(x)
        for j in range(self.funcCount):
            c = i == j
            dfdy[c] = self.functions[j].derivativeY(x[c], y[c], z[c])
        return dfdy

    def _derZ(self, x, y, z):
        """
        Returns the first derivative of the function with respect to Z at each
        value in (x,y,z).  Only called internally by HARKinterpolator3D._derZ.
        """
        m = len(x)
        temp = np.zeros((m, self.funcCount))
        for j in range(self.funcCount):
            temp[:, j] = self.functions[j](x, y, z)
        i = self.argcompare(temp, axis=1)
        y = temp[np.arange(m), i]
        dfdz = np.zeros_like(x)
        for j in range(self.funcCount):
            c = i == j
            dfdz[c] = self.functions[j].derivativeZ(x[c], y[c], z[c])
        return dfdz


class VariableLowerBoundFunc2D(MetricObject):
    """
    A class for representing a function with two real inputs whose lower bound
    in the first input depends on the second input.  Useful for managing curved
    natural borrowing constraints, as occurs in the persistent shocks model.

    Parameters
    ----------
    func : function
        A function f: (R_+ x R) --> R representing the function of interest
        shifted by its lower bound in the first input.
    lowerBound : function
        The lower bound in the first input of the function of interest, as
        a function of the second input.
    """

    distance_criteria = ["func", "lowerBound"]

    def __init__(self, func, lowerBound):
        self.func = func
        self.lowerBound = lowerBound

    def __call__(self, x, y):
        """
        Evaluate the function at given state space points.

        Parameters
        ----------
        x : np.array
             First input values.
        y : np.array
             Second input values; should be of same shape as x.

        Returns
        -------
        f_out : np.array
            Function evaluated at (x,y), of same shape as inputs.
        """
        xShift = self.lowerBound(y)
        f_out = self.func(x - xShift, y)
        return f_out

    def derivativeX(self, x, y):
        """
        Evaluate the first derivative with respect to x of the function at given
        state space points.

        Parameters
        ----------
        x : np.array
             First input values.
        y : np.array
             Second input values; should be of same shape as x.

        Returns
        -------
        dfdx_out : np.array
            First derivative of function with respect to the first input,
            evaluated at (x,y), of same shape as inputs.
        """
        xShift = self.lowerBound(y)
        dfdx_out = self.func.derivativeX(x - xShift, y)
        return dfdx_out

    def derivativeY(self, x, y):
        """
        Evaluate the first derivative with respect to y of the function at given
        state space points.

        Parameters
        ----------
        x : np.array
             First input values.
        y : np.array
             Second input values; should be of same shape as x.

        Returns
        -------
        dfdy_out : np.array
            First derivative of function with respect to the second input,
            evaluated at (x,y), of same shape as inputs.
        """
        xShift, xShiftDer = self.lowerBound.eval_with_derivative(y)
        dfdy_out = self.func.derivativeY(
            x - xShift, y
        ) - xShiftDer * self.func.derivativeX(x - xShift, y)
        return dfdy_out


class VariableLowerBoundFunc3D(MetricObject):
    """
    A class for representing a function with three real inputs whose lower bound
    in the first input depends on the second input.  Useful for managing curved
    natural borrowing constraints.

    Parameters
    ----------
    func : function
        A function f: (R_+ x R^2) --> R representing the function of interest
        shifted by its lower bound in the first input.
    lowerBound : function
        The lower bound in the first input of the function of interest, as
        a function of the second input.
    """

    distance_criteria = ["func", "lowerBound"]

    def __init__(self, func, lowerBound):
        self.func = func
        self.lowerBound = lowerBound

    def __call__(self, x, y, z):
        """
        Evaluate the function at given state space points.

        Parameters
        ----------
        x : np.array
             First input values.
        y : np.array
             Second input values; should be of same shape as x.
        z : np.array
             Third input values; should be of same shape as x.

        Returns
        -------
        f_out : np.array
            Function evaluated at (x,y,z), of same shape as inputs.
        """
        xShift = self.lowerBound(y)
        f_out = self.func(x - xShift, y, z)
        return f_out

    def derivativeX(self, x, y, z):
        """
        Evaluate the first derivative with respect to x of the function at given
        state space points.

        Parameters
        ----------
        x : np.array
             First input values.
        y : np.array
             Second input values; should be of same shape as x.
        z : np.array
             Third input values; should be of same shape as x.

        Returns
        -------
        dfdx_out : np.array
            First derivative of function with respect to the first input,
            evaluated at (x,y,z), of same shape as inputs.
        """
        xShift = self.lowerBound(y)
        dfdx_out = self.func.derivativeX(x - xShift, y, z)
        return dfdx_out

    def derivativeY(self, x, y, z):
        """
        Evaluate the first derivative with respect to y of the function at given
        state space points.

        Parameters
        ----------
        x : np.array
             First input values.
        y : np.array
             Second input values; should be of same shape as x.
        z : np.array
             Third input values; should be of same shape as x.

        Returns
        -------
        dfdy_out : np.array
            First derivative of function with respect to the second input,
            evaluated at (x,y,z), of same shape as inputs.
        """
        xShift, xShiftDer = self.lowerBound.eval_with_derivative(y)
        dfdy_out = self.func.derivativeY(
            x - xShift, y, z
        ) - xShiftDer * self.func.derivativeX(x - xShift, y, z)
        return dfdy_out

    def derivativeZ(self, x, y, z):
        """
        Evaluate the first derivative with respect to z of the function at given
        state space points.

        Parameters
        ----------
        x : np.array
             First input values.
        y : np.array
             Second input values; should be of same shape as x.
        z : np.array
             Third input values; should be of same shape as x.

        Returns
        -------
        dfdz_out : np.array
            First derivative of function with respect to the third input,
            evaluated at (x,y,z), of same shape as inputs.
        """
        xShift = self.lowerBound(y)
        dfdz_out = self.func.derivativeZ(x - xShift, y, z)
        return dfdz_out


class LinearInterpOnInterp1D(HARKinterpolator2D):
    """
    A 2D interpolator that linearly interpolates among a list of 1D interpolators.

    Parameters
    ----------
    xInterpolators : [HARKinterpolator1D]
        A list of 1D interpolations over the x variable.  The nth element of
        xInterpolators represents f(x,y_values[n]).
    y_values: numpy.array
        An array of y values equal in length to xInterpolators.
    """

    distance_criteria = ["xInterpolators", "y_list"]

    def __init__(self, xInterpolators, y_values):
        self.xInterpolators = xInterpolators
        self.y_list = y_values
        self.y_n = y_values.size

    def _evaluate(self, x, y):
        """
        Returns the level of the interpolated function at each value in x,y.
        Only called internally by HARKinterpolator2D.__call__ (etc).
        """
        if _isscalar(x):
            y_pos = max(min(np.searchsorted(self.y_list, y), self.y_n - 1), 1)
            alpha = (y - self.y_list[y_pos - 1]) / (
                self.y_list[y_pos] - self.y_list[y_pos - 1]
            )
            f = (1 - alpha) * self.xInterpolators[y_pos - 1](
                x
            ) + alpha * self.xInterpolators[y_pos](x)
        else:
            m = len(x)
            y_pos = np.searchsorted(self.y_list, y)
            y_pos[y_pos > self.y_n - 1] = self.y_n - 1
            y_pos[y_pos < 1] = 1
            f = np.zeros(m) + np.nan
            if y.size > 0:
                for i in range(1, self.y_n):
                    c = y_pos == i
                    if np.any(c):
                        alpha = (y[c] - self.y_list[i - 1]) / (
                            self.y_list[i] - self.y_list[i - 1]
                        )
                        f[c] = (1 - alpha) * self.xInterpolators[i - 1](
                            x[c]
                        ) + alpha * self.xInterpolators[i](x[c])
        return f

    def _derX(self, x, y):
        """
        Returns the derivative with respect to x of the interpolated function
        at each value in x,y. Only called internally by HARKinterpolator2D.derivativeX.
        """
        if _isscalar(x):
            y_pos = max(min(np.searchsorted(self.y_list, y), self.y_n - 1), 1)
            alpha = (y - self.y_list[y_pos - 1]) / (
                self.y_list[y_pos] - self.y_list[y_pos - 1]
            )
            dfdx = (1 - alpha) * self.xInterpolators[y_pos - 1]._der(
                x
            ) + alpha * self.xInterpolators[y_pos]._der(x)
        else:
            m = len(x)
            y_pos = np.searchsorted(self.y_list, y)
            y_pos[y_pos > self.y_n - 1] = self.y_n - 1
            y_pos[y_pos < 1] = 1
            dfdx = np.zeros(m) + np.nan
            if y.size > 0:
                for i in range(1, self.y_n):
                    c = y_pos == i
                    if np.any(c):
                        alpha = (y[c] - self.y_list[i - 1]) / (
                            self.y_list[i] - self.y_list[i - 1]
                        )
                        dfdx[c] = (1 - alpha) * self.xInterpolators[i - 1]._der(
                            x[c]
                        ) + alpha * self.xInterpolators[i]._der(x[c])
        return dfdx

    def _derY(self, x, y):
        """
        Returns the derivative with respect to y of the interpolated function
        at each value in x,y. Only called internally by HARKinterpolator2D.derivativeY.
        """
        if _isscalar(x):
            y_pos = max(min(np.searchsorted(self.y_list, y), self.y_n - 1), 1)
            dfdy = (
                self.xInterpolators[y_pos](x) - self.xInterpolators[y_pos - 1](x)
            ) / (self.y_list[y_pos] - self.y_list[y_pos - 1])
        else:
            m = len(x)
            y_pos = np.searchsorted(self.y_list, y)
            y_pos[y_pos > self.y_n - 1] = self.y_n - 1
            y_pos[y_pos < 1] = 1
            dfdy = np.zeros(m) + np.nan
            if y.size > 0:
                for i in range(1, self.y_n):
                    c = y_pos == i
                    if np.any(c):
                        dfdy[c] = (
                            self.xInterpolators[i](x[c])
                            - self.xInterpolators[i - 1](x[c])
                        ) / (self.y_list[i] - self.y_list[i - 1])
        return dfdy


class BilinearInterpOnInterp1D(HARKinterpolator3D):
    """
    A 3D interpolator that bilinearly interpolates among a list of lists of 1D
    interpolators.

    Constructor for the class, generating an approximation to a function of
    the form f(x,y,z) using interpolations over f(x,y_0,z_0) for a fixed grid
    of y_0 and z_0 values.

    Parameters
    ----------
    xInterpolators : [[HARKinterpolator1D]]
        A list of lists of 1D interpolations over the x variable.  The i,j-th
        element of xInterpolators represents f(x,y_values[i],z_values[j]).
    y_values: numpy.array
        An array of y values equal in length to xInterpolators.
    z_values: numpy.array
        An array of z values equal in length to xInterpolators[0].
    """

    distance_criteria = ["xInterpolators", "y_list", "z_list"]

    def __init__(self, xInterpolators, y_values, z_values):
        self.xInterpolators = xInterpolators
        self.y_list = y_values
        self.y_n = y_values.size
        self.z_list = z_values
        self.z_n = z_values.size

    def _evaluate(self, x, y, z):
        """
        Returns the level of the interpolated function at each value in x,y,z.
        Only called internally by HARKinterpolator3D.__call__ (etc).
        """
        if _isscalar(x):
            y_pos = max(min(np.searchsorted(self.y_list, y), self.y_n - 1), 1)
            z_pos = max(min(np.searchsorted(self.z_list, z), self.z_n - 1), 1)
            alpha = (y - self.y_list[y_pos - 1]) / (
                self.y_list[y_pos] - self.y_list[y_pos - 1]
            )
            beta = (z - self.z_list[z_pos - 1]) / (
                self.z_list[z_pos] - self.z_list[z_pos - 1]
            )
            f = (
                (1 - alpha) * (1 - beta) * self.xInterpolators[y_pos - 1][z_pos - 1](x)
                + (1 - alpha) * beta * self.xInterpolators[y_pos - 1][z_pos](x)
                + alpha * (1 - beta) * self.xInterpolators[y_pos][z_pos - 1](x)
                + alpha * beta * self.xInterpolators[y_pos][z_pos](x)
            )
        else:
            m = len(x)
            y_pos = np.searchsorted(self.y_list, y)
            y_pos[y_pos > self.y_n - 1] = self.y_n - 1
            y_pos[y_pos < 1] = 1
            z_pos = np.searchsorted(self.z_list, z)
            z_pos[z_pos > self.z_n - 1] = self.z_n - 1
            z_pos[z_pos < 1] = 1
            f = np.zeros(m) + np.nan
            for i in range(1, self.y_n):
                for j in range(1, self.z_n):
                    c = np.logical_and(i == y_pos, j == z_pos)
                    if np.any(c):
                        alpha = (y[c] - self.y_list[i - 1]) / (
                            self.y_list[i] - self.y_list[i - 1]
                        )
                        beta = (z[c] - self.z_list[j - 1]) / (
                            self.z_list[j] - self.z_list[j - 1]
                        )
                        f[c] = (
                            (1 - alpha)
                            * (1 - beta)
                            * self.xInterpolators[i - 1][j - 1](x[c])
                            + (1 - alpha) * beta * self.xInterpolators[i - 1][j](x[c])
                            + alpha * (1 - beta) * self.xInterpolators[i][j - 1](x[c])
                            + alpha * beta * self.xInterpolators[i][j](x[c])
                        )
        return f

    def _derX(self, x, y, z):
        """
        Returns the derivative with respect to x of the interpolated function
        at each value in x,y,z. Only called internally by HARKinterpolator3D.derivativeX.
        """
        if _isscalar(x):
            y_pos = max(min(np.searchsorted(self.y_list, y), self.y_n - 1), 1)
            z_pos = max(min(np.searchsorted(self.z_list, z), self.z_n - 1), 1)
            alpha = (y - self.y_list[y_pos - 1]) / (
                self.y_list[y_pos] - self.y_list[y_pos - 1]
            )
            beta = (z - self.z_list[z_pos - 1]) / (
                self.z_list[z_pos] - self.z_list[z_pos - 1]
            )
            dfdx = (
                (1 - alpha)
                * (1 - beta)
                * self.xInterpolators[y_pos - 1][z_pos - 1]._der(x)
                + (1 - alpha) * beta * self.xInterpolators[y_pos - 1][z_pos]._der(x)
                + alpha * (1 - beta) * self.xInterpolators[y_pos][z_pos - 1]._der(x)
                + alpha * beta * self.xInterpolators[y_pos][z_pos]._der(x)
            )
        else:
            m = len(x)
            y_pos = np.searchsorted(self.y_list, y)
            y_pos[y_pos > self.y_n - 1] = self.y_n - 1
            y_pos[y_pos < 1] = 1
            z_pos = np.searchsorted(self.z_list, z)
            z_pos[z_pos > self.z_n - 1] = self.z_n - 1
            z_pos[z_pos < 1] = 1
            dfdx = np.zeros(m) + np.nan
            for i in range(1, self.y_n):
                for j in range(1, self.z_n):
                    c = np.logical_and(i == y_pos, j == z_pos)
                    if np.any(c):
                        alpha = (y[c] - self.y_list[i - 1]) / (
                            self.y_list[i] - self.y_list[i - 1]
                        )
                        beta = (z[c] - self.z_list[j - 1]) / (
                            self.z_list[j] - self.z_list[j - 1]
                        )
                        dfdx[c] = (
                            (1 - alpha)
                            * (1 - beta)
                            * self.xInterpolators[i - 1][j - 1]._der(x[c])
                            + (1 - alpha)
                            * beta
                            * self.xInterpolators[i - 1][j]._der(x[c])
                            + alpha
                            * (1 - beta)
                            * self.xInterpolators[i][j - 1]._der(x[c])
                            + alpha * beta * self.xInterpolators[i][j]._der(x[c])
                        )
        return dfdx

    def _derY(self, x, y, z):
        """
        Returns the derivative with respect to y of the interpolated function
        at each value in x,y,z. Only called internally by HARKinterpolator3D.derivativeY.
        """
        if _isscalar(x):
            y_pos = max(min(np.searchsorted(self.y_list, y), self.y_n - 1), 1)
            z_pos = max(min(np.searchsorted(self.z_list, z), self.z_n - 1), 1)
            beta = (z - self.z_list[z_pos - 1]) / (
                self.z_list[z_pos] - self.z_list[z_pos - 1]
            )
            dfdy = (
                (
                    (1 - beta) * self.xInterpolators[y_pos][z_pos - 1](x)
                    + beta * self.xInterpolators[y_pos][z_pos](x)
                )
                - (
                    (1 - beta) * self.xInterpolators[y_pos - 1][z_pos - 1](x)
                    + beta * self.xInterpolators[y_pos - 1][z_pos](x)
                )
            ) / (self.y_list[y_pos] - self.y_list[y_pos - 1])
        else:
            m = len(x)
            y_pos = np.searchsorted(self.y_list, y)
            y_pos[y_pos > self.y_n - 1] = self.y_n - 1
            y_pos[y_pos < 1] = 1
            z_pos = np.searchsorted(self.z_list, z)
            z_pos[z_pos > self.z_n - 1] = self.z_n - 1
            z_pos[z_pos < 1] = 1
            dfdy = np.zeros(m) + np.nan
            for i in range(1, self.y_n):
                for j in range(1, self.z_n):
                    c = np.logical_and(i == y_pos, j == z_pos)
                    if np.any(c):
                        beta = (z[c] - self.z_list[j - 1]) / (
                            self.z_list[j] - self.z_list[j - 1]
                        )
                        dfdy[c] = (
                            (
                                (1 - beta) * self.xInterpolators[i][j - 1](x[c])
                                + beta * self.xInterpolators[i][j](x[c])
                            )
                            - (
                                (1 - beta) * self.xInterpolators[i - 1][j - 1](x[c])
                                + beta * self.xInterpolators[i - 1][j](x[c])
                            )
                        ) / (self.y_list[i] - self.y_list[i - 1])
        return dfdy

    def _derZ(self, x, y, z):
        """
        Returns the derivative with respect to z of the interpolated function
        at each value in x,y,z. Only called internally by HARKinterpolator3D.derivativeZ.
        """
        if _isscalar(x):
            y_pos = max(min(np.searchsorted(self.y_list, y), self.y_n - 1), 1)
            z_pos = max(min(np.searchsorted(self.z_list, z), self.z_n - 1), 1)
            alpha = (y - self.y_list[y_pos - 1]) / (
                self.y_list[y_pos] - self.y_list[y_pos - 1]
            )
            dfdz = (
                (
                    (1 - alpha) * self.xInterpolators[y_pos - 1][z_pos](x)
                    + alpha * self.xInterpolators[y_pos][z_pos](x)
                )
                - (
                    (1 - alpha) * self.xInterpolators[y_pos - 1][z_pos - 1](x)
                    + alpha * self.xInterpolators[y_pos][z_pos - 1](x)
                )
            ) / (self.z_list[z_pos] - self.z_list[z_pos - 1])
        else:
            m = len(x)
            y_pos = np.searchsorted(self.y_list, y)
            y_pos[y_pos > self.y_n - 1] = self.y_n - 1
            y_pos[y_pos < 1] = 1
            z_pos = np.searchsorted(self.z_list, z)
            z_pos[z_pos > self.z_n - 1] = self.z_n - 1
            z_pos[z_pos < 1] = 1
            dfdz = np.zeros(m) + np.nan
            for i in range(1, self.y_n):
                for j in range(1, self.z_n):
                    c = np.logical_and(i == y_pos, j == z_pos)
                    if np.any(c):
                        alpha = (y[c] - self.y_list[i - 1]) / (
                            self.y_list[i] - self.y_list[i - 1]
                        )
                        dfdz[c] = (
                            (
                                (1 - alpha) * self.xInterpolators[i - 1][j](x[c])
                                + alpha * self.xInterpolators[i][j](x[c])
                            )
                            - (
                                (1 - alpha) * self.xInterpolators[i - 1][j - 1](x[c])
                                + alpha * self.xInterpolators[i][j - 1](x[c])
                            )
                        ) / (self.z_list[j] - self.z_list[j - 1])
        return dfdz


class TrilinearInterpOnInterp1D(HARKinterpolator4D):
    """
    A 4D interpolator that trilinearly interpolates among a list of lists of 1D interpolators.

    Constructor for the class, generating an approximation to a function of
    the form f(w,x,y,z) using interpolations over f(w,x_0,y_0,z_0) for a fixed
    grid of y_0 and z_0 values.

    Parameters
    ----------
    wInterpolators : [[[HARKinterpolator1D]]]
        A list of lists of lists of 1D interpolations over the x variable.
        The i,j,k-th element of wInterpolators represents f(w,x_values[i],y_values[j],z_values[k]).
    x_values: numpy.array
        An array of x values equal in length to wInterpolators.
    y_values: numpy.array
        An array of y values equal in length to wInterpolators[0].
    z_values: numpy.array
        An array of z values equal in length to wInterpolators[0][0]
    """

    distance_criteria = ["wInterpolators", "x_list", "y_list", "z_list"]

    def __init__(self, wInterpolators, x_values, y_values, z_values):
        self.wInterpolators = wInterpolators
        self.x_list = x_values
        self.x_n = x_values.size
        self.y_list = y_values
        self.y_n = y_values.size
        self.z_list = z_values
        self.z_n = z_values.size

    def _evaluate(self, w, x, y, z):
        """
        Returns the level of the interpolated function at each value in w,x,y,z.
        Only called internally by HARKinterpolator4D.__call__ (etc).
        """
        if _isscalar(w):
            x_pos = max(min(np.searchsorted(self.x_list, x), self.x_n - 1), 1)
            y_pos = max(min(np.searchsorted(self.y_list, y), self.y_n - 1), 1)
            z_pos = max(min(np.searchsorted(self.z_list, z), self.z_n - 1), 1)
            alpha = (x - self.x_list[x_pos - 1]) / (
                self.x_list[x_pos] - self.x_list[x_pos - 1]
            )
            beta = (y - self.y_list[y_pos - 1]) / (
                self.y_list[y_pos] - self.y_list[y_pos - 1]
            )
            gamma = (z - self.z_list[z_pos - 1]) / (
                self.z_list[z_pos] - self.z_list[z_pos - 1]
            )
            f = (
                (1 - alpha)
                * (1 - beta)
                * (1 - gamma)
                * self.wInterpolators[x_pos - 1][y_pos - 1][z_pos - 1](w)
                + (1 - alpha)
                * (1 - beta)
                * gamma
                * self.wInterpolators[x_pos - 1][y_pos - 1][z_pos](w)
                + (1 - alpha)
                * beta
                * (1 - gamma)
                * self.wInterpolators[x_pos - 1][y_pos][z_pos - 1](w)
                + (1 - alpha)
                * beta
                * gamma
                * self.wInterpolators[x_pos - 1][y_pos][z_pos](w)
                + alpha
                * (1 - beta)
                * (1 - gamma)
                * self.wInterpolators[x_pos][y_pos - 1][z_pos - 1](w)
                + alpha
                * (1 - beta)
                * gamma
                * self.wInterpolators[x_pos][y_pos - 1][z_pos](w)
                + alpha
                * beta
                * (1 - gamma)
                * self.wInterpolators[x_pos][y_pos][z_pos - 1](w)
                + alpha * beta * gamma * self.wInterpolators[x_pos][y_pos][z_pos](w)
            )
        else:
            m = len(x)
            x_pos = np.searchsorted(self.x_list, x)
            x_pos[x_pos > self.x_n - 1] = self.x_n - 1
            y_pos = np.searchsorted(self.y_list, y)
            y_pos[y_pos > self.y_n - 1] = self.y_n - 1
            y_pos[y_pos < 1] = 1
            z_pos = np.searchsorted(self.z_list, z)
            z_pos[z_pos > self.z_n - 1] = self.z_n - 1
            z_pos[z_pos < 1] = 1
            f = np.zeros(m) + np.nan
            for i in range(1, self.x_n):
                for j in range(1, self.y_n):
                    for k in range(1, self.z_n):
                        c = np.logical_and(
                            np.logical_and(i == x_pos, j == y_pos), k == z_pos
                        )
                        if np.any(c):
                            alpha = (x[c] - self.x_list[i - 1]) / (
                                self.x_list[i] - self.x_list[i - 1]
                            )
                            beta = (y[c] - self.y_list[j - 1]) / (
                                self.y_list[j] - self.y_list[j - 1]
                            )
                            gamma = (z[c] - self.z_list[k - 1]) / (
                                self.z_list[k] - self.z_list[k - 1]
                            )
                            f[c] = (
                                (1 - alpha)
                                * (1 - beta)
                                * (1 - gamma)
                                * self.wInterpolators[i - 1][j - 1][k - 1](w[c])
                                + (1 - alpha)
                                * (1 - beta)
                                * gamma
                                * self.wInterpolators[i - 1][j - 1][k](w[c])
                                + (1 - alpha)
                                * beta
                                * (1 - gamma)
                                * self.wInterpolators[i - 1][j][k - 1](w[c])
                                + (1 - alpha)
                                * beta
                                * gamma
                                * self.wInterpolators[i - 1][j][k](w[c])
                                + alpha
                                * (1 - beta)
                                * (1 - gamma)
                                * self.wInterpolators[i][j - 1][k - 1](w[c])
                                + alpha
                                * (1 - beta)
                                * gamma
                                * self.wInterpolators[i][j - 1][k](w[c])
                                + alpha
                                * beta
                                * (1 - gamma)
                                * self.wInterpolators[i][j][k - 1](w[c])
                                + alpha
                                * beta
                                * gamma
                                * self.wInterpolators[i][j][k](w[c])
                            )
        return f

    def _derW(self, w, x, y, z):
        """
        Returns the derivative with respect to w of the interpolated function
        at each value in w,x,y,z. Only called internally by HARKinterpolator4D.derivativeW.
        """
        if _isscalar(w):
            x_pos = max(min(np.searchsorted(self.x_list, x), self.x_n - 1), 1)
            y_pos = max(min(np.searchsorted(self.y_list, y), self.y_n - 1), 1)
            z_pos = max(min(np.searchsorted(self.z_list, z), self.z_n - 1), 1)
            alpha = (x - self.x_list[x_pos - 1]) / (
                self.x_list[x_pos] - self.x_list[x_pos - 1]
            )
            beta = (y - self.y_list[y_pos - 1]) / (
                self.y_list[y_pos] - self.y_list[y_pos - 1]
            )
            gamma = (z - self.z_list[z_pos - 1]) / (
                self.z_list[z_pos] - self.z_list[z_pos - 1]
            )
            dfdw = (
                (1 - alpha)
                * (1 - beta)
                * (1 - gamma)
                * self.wInterpolators[x_pos - 1][y_pos - 1][z_pos - 1]._der(w)
                + (1 - alpha)
                * (1 - beta)
                * gamma
                * self.wInterpolators[x_pos - 1][y_pos - 1][z_pos]._der(w)
                + (1 - alpha)
                * beta
                * (1 - gamma)
                * self.wInterpolators[x_pos - 1][y_pos][z_pos - 1]._der(w)
                + (1 - alpha)
                * beta
                * gamma
                * self.wInterpolators[x_pos - 1][y_pos][z_pos]._der(w)
                + alpha
                * (1 - beta)
                * (1 - gamma)
                * self.wInterpolators[x_pos][y_pos - 1][z_pos - 1]._der(w)
                + alpha
                * (1 - beta)
                * gamma
                * self.wInterpolators[x_pos][y_pos - 1][z_pos]._der(w)
                + alpha
                * beta
                * (1 - gamma)
                * self.wInterpolators[x_pos][y_pos][z_pos - 1]._der(w)
                + alpha
                * beta
                * gamma
                * self.wInterpolators[x_pos][y_pos][z_pos]._der(w)
            )
        else:
            m = len(x)
            x_pos = np.searchsorted(self.x_list, x)
            x_pos[x_pos > self.x_n - 1] = self.x_n - 1
            y_pos = np.searchsorted(self.y_list, y)
            y_pos[y_pos > self.y_n - 1] = self.y_n - 1
            y_pos[y_pos < 1] = 1
            z_pos = np.searchsorted(self.z_list, z)
            z_pos[z_pos > self.z_n - 1] = self.z_n - 1
            z_pos[z_pos < 1] = 1
            dfdw = np.zeros(m) + np.nan
            for i in range(1, self.x_n):
                for j in range(1, self.y_n):
                    for k in range(1, self.z_n):
                        c = np.logical_and(
                            np.logical_and(i == x_pos, j == y_pos), k == z_pos
                        )
                        if np.any(c):
                            alpha = (x[c] - self.x_list[i - 1]) / (
                                self.x_list[i] - self.x_list[i - 1]
                            )
                            beta = (y[c] - self.y_list[j - 1]) / (
                                self.y_list[j] - self.y_list[j - 1]
                            )
                            gamma = (z[c] - self.z_list[k - 1]) / (
                                self.z_list[k] - self.z_list[k - 1]
                            )
                            dfdw[c] = (
                                (1 - alpha)
                                * (1 - beta)
                                * (1 - gamma)
                                * self.wInterpolators[i - 1][j - 1][k - 1]._der(w[c])
                                + (1 - alpha)
                                * (1 - beta)
                                * gamma
                                * self.wInterpolators[i - 1][j - 1][k]._der(w[c])
                                + (1 - alpha)
                                * beta
                                * (1 - gamma)
                                * self.wInterpolators[i - 1][j][k - 1]._der(w[c])
                                + (1 - alpha)
                                * beta
                                * gamma
                                * self.wInterpolators[i - 1][j][k]._der(w[c])
                                + alpha
                                * (1 - beta)
                                * (1 - gamma)
                                * self.wInterpolators[i][j - 1][k - 1]._der(w[c])
                                + alpha
                                * (1 - beta)
                                * gamma
                                * self.wInterpolators[i][j - 1][k]._der(w[c])
                                + alpha
                                * beta
                                * (1 - gamma)
                                * self.wInterpolators[i][j][k - 1]._der(w[c])
                                + alpha
                                * beta
                                * gamma
                                * self.wInterpolators[i][j][k]._der(w[c])
                            )
        return dfdw

    def _derX(self, w, x, y, z):
        """
        Returns the derivative with respect to x of the interpolated function
        at each value in w,x,y,z. Only called internally by HARKinterpolator4D.derivativeX.
        """
        if _isscalar(w):
            x_pos = max(min(np.searchsorted(self.x_list, x), self.x_n - 1), 1)
            y_pos = max(min(np.searchsorted(self.y_list, y), self.y_n - 1), 1)
            z_pos = max(min(np.searchsorted(self.z_list, z), self.z_n - 1), 1)
            beta = (y - self.y_list[y_pos - 1]) / (
                self.y_list[y_pos] - self.y_list[y_pos - 1]
            )
            gamma = (z - self.z_list[z_pos - 1]) / (
                self.z_list[z_pos] - self.z_list[z_pos - 1]
            )
            dfdx = (
                (
                    (1 - beta)
                    * (1 - gamma)
                    * self.wInterpolators[x_pos][y_pos - 1][z_pos - 1](w)
                    + (1 - beta)
                    * gamma
                    * self.wInterpolators[x_pos][y_pos - 1][z_pos](w)
                    + beta
                    * (1 - gamma)
                    * self.wInterpolators[x_pos][y_pos][z_pos - 1](w)
                    + beta * gamma * self.wInterpolators[x_pos][y_pos][z_pos](w)
                )
                - (
                    (1 - beta)
                    * (1 - gamma)
                    * self.wInterpolators[x_pos - 1][y_pos - 1][z_pos - 1](w)
                    + (1 - beta)
                    * gamma
                    * self.wInterpolators[x_pos - 1][y_pos - 1][z_pos](w)
                    + beta
                    * (1 - gamma)
                    * self.wInterpolators[x_pos - 1][y_pos][z_pos - 1](w)
                    + beta * gamma * self.wInterpolators[x_pos - 1][y_pos][z_pos](w)
                )
            ) / (self.x_list[x_pos] - self.x_list[x_pos - 1])
        else:
            m = len(x)
            x_pos = np.searchsorted(self.x_list, x)
            x_pos[x_pos > self.x_n - 1] = self.x_n - 1
            y_pos = np.searchsorted(self.y_list, y)
            y_pos[y_pos > self.y_n - 1] = self.y_n - 1
            y_pos[y_pos < 1] = 1
            z_pos = np.searchsorted(self.z_list, z)
            z_pos[z_pos > self.z_n - 1] = self.z_n - 1
            z_pos[z_pos < 1] = 1
            dfdx = np.zeros(m) + np.nan
            for i in range(1, self.x_n):
                for j in range(1, self.y_n):
                    for k in range(1, self.z_n):
                        c = np.logical_and(
                            np.logical_and(i == x_pos, j == y_pos), k == z_pos
                        )
                        if np.any(c):
                            beta = (y[c] - self.y_list[j - 1]) / (
                                self.y_list[j] - self.y_list[j - 1]
                            )
                            gamma = (z[c] - self.z_list[k - 1]) / (
                                self.z_list[k] - self.z_list[k - 1]
                            )
                            dfdx[c] = (
                                (
                                    (1 - beta)
                                    * (1 - gamma)
                                    * self.wInterpolators[i][j - 1][k - 1](w[c])
                                    + (1 - beta)
                                    * gamma
                                    * self.wInterpolators[i][j - 1][k](w[c])
                                    + beta
                                    * (1 - gamma)
                                    * self.wInterpolators[i][j][k - 1](w[c])
                                    + beta * gamma * self.wInterpolators[i][j][k](w[c])
                                )
                                - (
                                    (1 - beta)
                                    * (1 - gamma)
                                    * self.wInterpolators[i - 1][j - 1][k - 1](w[c])
                                    + (1 - beta)
                                    * gamma
                                    * self.wInterpolators[i - 1][j - 1][k](w[c])
                                    + beta
                                    * (1 - gamma)
                                    * self.wInterpolators[i - 1][j][k - 1](w[c])
                                    + beta
                                    * gamma
                                    * self.wInterpolators[i - 1][j][k](w[c])
                                )
                            ) / (self.x_list[i] - self.x_list[i - 1])
        return dfdx

    def _derY(self, w, x, y, z):
        """
        Returns the derivative with respect to y of the interpolated function
        at each value in w,x,y,z. Only called internally by HARKinterpolator4D.derivativeY.
        """
        if _isscalar(w):
            x_pos = max(min(np.searchsorted(self.x_list, x), self.x_n - 1), 1)
            y_pos = max(min(np.searchsorted(self.y_list, y), self.y_n - 1), 1)
            z_pos = max(min(np.searchsorted(self.z_list, z), self.z_n - 1), 1)
            alpha = (x - self.x_list[x_pos - 1]) / (
                self.y_list[x_pos] - self.x_list[x_pos - 1]
            )
            gamma = (z - self.z_list[z_pos - 1]) / (
                self.z_list[z_pos] - self.z_list[z_pos - 1]
            )
            dfdy = (
                (
                    (1 - alpha)
                    * (1 - gamma)
                    * self.wInterpolators[x_pos - 1][y_pos][z_pos - 1](w)
                    + (1 - alpha)
                    * gamma
                    * self.wInterpolators[x_pos - 1][y_pos][z_pos](w)
                    + alpha
                    * (1 - gamma)
                    * self.wInterpolators[x_pos][y_pos][z_pos - 1](w)
                    + alpha * gamma * self.wInterpolators[x_pos][y_pos][z_pos](w)
                )
                - (
                    (1 - alpha)
                    * (1 - gamma)
                    * self.wInterpolators[x_pos - 1][y_pos - 1][z_pos - 1](w)
                    + (1 - alpha)
                    * gamma
                    * self.wInterpolators[x_pos - 1][y_pos - 1][z_pos](w)
                    + alpha
                    * (1 - gamma)
                    * self.wInterpolators[x_pos][y_pos - 1][z_pos - 1](w)
                    + alpha * gamma * self.wInterpolators[x_pos][y_pos - 1][z_pos](w)
                )
            ) / (self.y_list[y_pos] - self.y_list[y_pos - 1])
        else:
            m = len(x)
            x_pos = np.searchsorted(self.x_list, x)
            x_pos[x_pos > self.x_n - 1] = self.x_n - 1
            y_pos = np.searchsorted(self.y_list, y)
            y_pos[y_pos > self.y_n - 1] = self.y_n - 1
            y_pos[y_pos < 1] = 1
            z_pos = np.searchsorted(self.z_list, z)
            z_pos[z_pos > self.z_n - 1] = self.z_n - 1
            z_pos[z_pos < 1] = 1
            dfdy = np.zeros(m) + np.nan
            for i in range(1, self.x_n):
                for j in range(1, self.y_n):
                    for k in range(1, self.z_n):
                        c = np.logical_and(
                            np.logical_and(i == x_pos, j == y_pos), k == z_pos
                        )
                        if np.any(c):
                            alpha = (x[c] - self.x_list[i - 1]) / (
                                self.x_list[i] - self.x_list[i - 1]
                            )
                            gamma = (z[c] - self.z_list[k - 1]) / (
                                self.z_list[k] - self.z_list[k - 1]
                            )
                            dfdy[c] = (
                                (
                                    (1 - alpha)
                                    * (1 - gamma)
                                    * self.wInterpolators[i - 1][j][k - 1](w[c])
                                    + (1 - alpha)
                                    * gamma
                                    * self.wInterpolators[i - 1][j][k](w[c])
                                    + alpha
                                    * (1 - gamma)
                                    * self.wInterpolators[i][j][k - 1](w[c])
                                    + alpha * gamma * self.wInterpolators[i][j][k](w[c])
                                )
                                - (
                                    (1 - alpha)
                                    * (1 - gamma)
                                    * self.wInterpolators[i - 1][j - 1][k - 1](w[c])
                                    + (1 - alpha)
                                    * gamma
                                    * self.wInterpolators[i - 1][j - 1][k](w[c])
                                    + alpha
                                    * (1 - gamma)
                                    * self.wInterpolators[i][j - 1][k - 1](w[c])
                                    + alpha
                                    * gamma
                                    * self.wInterpolators[i][j - 1][k](w[c])
                                )
                            ) / (self.y_list[j] - self.y_list[j - 1])
        return dfdy

    def _derZ(self, w, x, y, z):
        """
        Returns the derivative with respect to z of the interpolated function
        at each value in w,x,y,z. Only called internally by HARKinterpolator4D.derivativeZ.
        """
        if _isscalar(w):
            x_pos = max(min(np.searchsorted(self.x_list, x), self.x_n - 1), 1)
            y_pos = max(min(np.searchsorted(self.y_list, y), self.y_n - 1), 1)
            z_pos = max(min(np.searchsorted(self.z_list, z), self.z_n - 1), 1)
            alpha = (x - self.x_list[x_pos - 1]) / (
                self.y_list[x_pos] - self.x_list[x_pos - 1]
            )
            beta = (y - self.y_list[y_pos - 1]) / (
                self.y_list[y_pos] - self.y_list[y_pos - 1]
            )
            dfdz = (
                (
                    (1 - alpha)
                    * (1 - beta)
                    * self.wInterpolators[x_pos - 1][y_pos - 1][z_pos](w)
                    + (1 - alpha)
                    * beta
                    * self.wInterpolators[x_pos - 1][y_pos][z_pos](w)
                    + alpha
                    * (1 - beta)
                    * self.wInterpolators[x_pos][y_pos - 1][z_pos](w)
                    + alpha * beta * self.wInterpolators[x_pos][y_pos][z_pos](w)
                )
                - (
                    (1 - alpha)
                    * (1 - beta)
                    * self.wInterpolators[x_pos - 1][y_pos - 1][z_pos - 1](w)
                    + (1 - alpha)
                    * beta
                    * self.wInterpolators[x_pos - 1][y_pos][z_pos - 1](w)
                    + alpha
                    * (1 - beta)
                    * self.wInterpolators[x_pos][y_pos - 1][z_pos - 1](w)
                    + alpha * beta * self.wInterpolators[x_pos][y_pos][z_pos - 1](w)
                )
            ) / (self.z_list[z_pos] - self.z_list[z_pos - 1])
        else:
            m = len(x)
            x_pos = np.searchsorted(self.x_list, x)
            x_pos[x_pos > self.x_n - 1] = self.x_n - 1
            y_pos = np.searchsorted(self.y_list, y)
            y_pos[y_pos > self.y_n - 1] = self.y_n - 1
            y_pos[y_pos < 1] = 1
            z_pos = np.searchsorted(self.z_list, z)
            z_pos[z_pos > self.z_n - 1] = self.z_n - 1
            z_pos[z_pos < 1] = 1
            dfdz = np.zeros(m) + np.nan
            for i in range(1, self.x_n):
                for j in range(1, self.y_n):
                    for k in range(1, self.z_n):
                        c = np.logical_and(
                            np.logical_and(i == x_pos, j == y_pos), k == z_pos
                        )
                        if np.any(c):
                            alpha = (x[c] - self.x_list[i - 1]) / (
                                self.x_list[i] - self.x_list[i - 1]
                            )
                            beta = (y[c] - self.y_list[j - 1]) / (
                                self.y_list[j] - self.y_list[j - 1]
                            )
                            dfdz[c] = (
                                (
                                    (1 - alpha)
                                    * (1 - beta)
                                    * self.wInterpolators[i - 1][j - 1][k](w[c])
                                    + (1 - alpha)
                                    * beta
                                    * self.wInterpolators[i - 1][j][k](w[c])
                                    + alpha
                                    * (1 - beta)
                                    * self.wInterpolators[i][j - 1][k](w[c])
                                    + alpha * beta * self.wInterpolators[i][j][k](w[c])
                                )
                                - (
                                    (1 - alpha)
                                    * (1 - beta)
                                    * self.wInterpolators[i - 1][j - 1][k - 1](w[c])
                                    + (1 - alpha)
                                    * beta
                                    * self.wInterpolators[i - 1][j][k - 1](w[c])
                                    + alpha
                                    * (1 - beta)
                                    * self.wInterpolators[i][j - 1][k - 1](w[c])
                                    + alpha
                                    * beta
                                    * self.wInterpolators[i][j][k - 1](w[c])
                                )
                            ) / (self.z_list[k] - self.z_list[k - 1])
        return dfdz


class LinearInterpOnInterp2D(HARKinterpolator3D):
    """
    A 3D interpolation method that linearly interpolates between "layers" of
    arbitrary 2D interpolations.  Useful for models with two endogenous state
    variables and one exogenous state variable when solving with the endogenous
    grid method.  NOTE: should not be used if an exogenous 3D grid is used, will
    be significantly slower than TrilinearInterp.

    Constructor for the class, generating an approximation to a function of
    the form f(x,y,z) using interpolations over f(x,y,z_0) for a fixed grid
    of z_0 values.

    Parameters
    ----------
    xyInterpolators : [HARKinterpolator2D]
        A list of 2D interpolations over the x and y variables.  The nth
        element of xyInterpolators represents f(x,y,z_values[n]).
    z_values: numpy.array
        An array of z values equal in length to xyInterpolators.
    """

    distance_criteria = ["xyInterpolators", "z_list"]

    def __init__(self, xyInterpolators, z_values):
        self.xyInterpolators = xyInterpolators
        self.z_list = z_values
        self.z_n = z_values.size

    def _evaluate(self, x, y, z):
        """
        Returns the level of the interpolated function at each value in x,y,z.
        Only called internally by HARKinterpolator3D.__call__ (etc).
        """
        if _isscalar(x):
            z_pos = max(min(np.searchsorted(self.z_list, z), self.z_n - 1), 1)
            alpha = (z - self.z_list[z_pos - 1]) / (
                self.z_list[z_pos] - self.z_list[z_pos - 1]
            )
            f = (1 - alpha) * self.xyInterpolators[z_pos - 1](
                x, y
            ) + alpha * self.xyInterpolators[z_pos](x, y)
        else:
            m = len(x)
            z_pos = np.searchsorted(self.z_list, z)
            z_pos[z_pos > self.z_n - 1] = self.z_n - 1
            z_pos[z_pos < 1] = 1
            f = np.zeros(m) + np.nan
            if x.size > 0:
                for i in range(1, self.z_n):
                    c = z_pos == i
                    if np.any(c):
                        alpha = (z[c] - self.z_list[i - 1]) / (
                            self.z_list[i] - self.z_list[i - 1]
                        )
                        f[c] = (1 - alpha) * self.xyInterpolators[i - 1](
                            x[c], y[c]
                        ) + alpha * self.xyInterpolators[i](x[c], y[c])
        return f

    def _derX(self, x, y, z):
        """
        Returns the derivative with respect to x of the interpolated function
        at each value in x,y,z. Only called internally by HARKinterpolator3D.derivativeX.
        """
        if _isscalar(x):
            z_pos = max(min(np.searchsorted(self.z_list, z), self.z_n - 1), 1)
            alpha = (z - self.z_list[z_pos - 1]) / (
                self.z_list[z_pos] - self.z_list[z_pos - 1]
            )
            dfdx = (1 - alpha) * self.xyInterpolators[z_pos - 1].derivativeX(
                x, y
            ) + alpha * self.xyInterpolators[z_pos].derivativeX(x, y)
        else:
            m = len(x)
            z_pos = np.searchsorted(self.z_list, z)
            z_pos[z_pos > self.z_n - 1] = self.z_n - 1
            z_pos[z_pos < 1] = 1
            dfdx = np.zeros(m) + np.nan
            if x.size > 0:
                for i in range(1, self.z_n):
                    c = z_pos == i
                    if np.any(c):
                        alpha = (z[c] - self.z_list[i - 1]) / (
                            self.z_list[i] - self.z_list[i - 1]
                        )
                        dfdx[c] = (1 - alpha) * self.xyInterpolators[i - 1].derivativeX(
                            x[c], y[c]
                        ) + alpha * self.xyInterpolators[i].derivativeX(x[c], y[c])
        return dfdx

    def _derY(self, x, y, z):
        """
        Returns the derivative with respect to y of the interpolated function
        at each value in x,y,z. Only called internally by HARKinterpolator3D.derivativeY.
        """
        if _isscalar(x):
            z_pos = max(min(np.searchsorted(self.z_list, z), self.z_n - 1), 1)
            alpha = (z - self.z_list[z_pos - 1]) / (
                self.z_list[z_pos] - self.z_list[z_pos - 1]
            )
            dfdy = (1 - alpha) * self.xyInterpolators[z_pos - 1].derivativeY(
                x, y
            ) + alpha * self.xyInterpolators[z_pos].derivativeY(x, y)
        else:
            m = len(x)
            z_pos = np.searchsorted(self.z_list, z)
            z_pos[z_pos > self.z_n - 1] = self.z_n - 1
            z_pos[z_pos < 1] = 1
            dfdy = np.zeros(m) + np.nan
            if x.size > 0:
                for i in range(1, self.z_n):
                    c = z_pos == i
                    if np.any(c):
                        alpha = (z[c] - self.z_list[i - 1]) / (
                            self.z_list[i] - self.z_list[i - 1]
                        )
                        dfdy[c] = (1 - alpha) * self.xyInterpolators[i - 1].derivativeY(
                            x[c], y[c]
                        ) + alpha * self.xyInterpolators[i].derivativeY(x[c], y[c])
        return dfdy

    def _derZ(self, x, y, z):
        """
        Returns the derivative with respect to z of the interpolated function
        at each value in x,y,z. Only called internally by HARKinterpolator3D.derivativeZ.
        """
        if _isscalar(x):
            z_pos = max(min(np.searchsorted(self.z_list, z), self.z_n - 1), 1)
            dfdz = (
                self.xyInterpolators[z_pos].derivativeX(x, y)
                - self.xyInterpolators[z_pos - 1].derivativeX(x, y)
            ) / (self.z_list[z_pos] - self.z_list[z_pos - 1])
        else:
            m = len(x)
            z_pos = np.searchsorted(self.z_list, z)
            z_pos[z_pos > self.z_n - 1] = self.z_n - 1
            z_pos[z_pos < 1] = 1
            dfdz = np.zeros(m) + np.nan
            if x.size > 0:
                for i in range(1, self.z_n):
                    c = z_pos == i
                    if np.any(c):
                        dfdz[c] = (
                            self.xyInterpolators[i](x[c], y[c])
                            - self.xyInterpolators[i - 1](x[c], y[c])
                        ) / (self.z_list[i] - self.z_list[i - 1])
        return dfdz


class BilinearInterpOnInterp2D(HARKinterpolator4D):
    """
    A 4D interpolation method that bilinearly interpolates among "layers" of
    arbitrary 2D interpolations.  Useful for models with two endogenous state
    variables and two exogenous state variables when solving with the endogenous
    grid method.  NOTE: should not be used if an exogenous 4D grid is used, will
    be significantly slower than QuadlinearInterp.

    Constructor for the class, generating an approximation to a function of
    the form f(w,x,y,z) using interpolations over f(w,x,y_0,z_0) for a fixed
    grid of y_0 and z_0 values.

    Parameters
    ----------
    wxInterpolators : [[HARKinterpolator2D]]
        A list of lists of 2D interpolations over the w and x variables.
        The i,j-th element of wxInterpolators represents
        f(w,x,y_values[i],z_values[j]).
    y_values: numpy.array
        An array of y values equal in length to wxInterpolators.
    z_values: numpy.array
        An array of z values equal in length to wxInterpolators[0].
    """

    distance_criteria = ["wxInterpolators", "y_list", "z_list"]

    def __init__(self, wxInterpolators, y_values, z_values):
        self.wxInterpolators = wxInterpolators
        self.y_list = y_values
        self.y_n = y_values.size
        self.z_list = z_values
        self.z_n = z_values.size

    def _evaluate(self, w, x, y, z):
        """
        Returns the level of the interpolated function at each value in x,y,z.
        Only called internally by HARKinterpolator4D.__call__ (etc).
        """
        if _isscalar(x):
            y_pos = max(min(np.searchsorted(self.y_list, y), self.y_n - 1), 1)
            z_pos = max(min(np.searchsorted(self.z_list, z), self.z_n - 1), 1)
            alpha = (y - self.y_list[y_pos - 1]) / (
                self.y_list[y_pos] - self.y_list[y_pos - 1]
            )
            beta = (z - self.z_list[z_pos - 1]) / (
                self.z_list[z_pos] - self.z_list[z_pos - 1]
            )
            f = (
                (1 - alpha)
                * (1 - beta)
                * self.wxInterpolators[y_pos - 1][z_pos - 1](w, x)
                + (1 - alpha) * beta * self.wxInterpolators[y_pos - 1][z_pos](w, x)
                + alpha * (1 - beta) * self.wxInterpolators[y_pos][z_pos - 1](w, x)
                + alpha * beta * self.wxInterpolators[y_pos][z_pos](w, x)
            )
        else:
            m = len(x)
            y_pos = np.searchsorted(self.y_list, y)
            y_pos[y_pos > self.y_n - 1] = self.y_n - 1
            y_pos[y_pos < 1] = 1
            z_pos = np.searchsorted(self.z_list, z)
            z_pos[z_pos > self.z_n - 1] = self.z_n - 1
            z_pos[z_pos < 1] = 1
            f = np.zeros(m) + np.nan
            for i in range(1, self.y_n):
                for j in range(1, self.z_n):
                    c = np.logical_and(i == y_pos, j == z_pos)
                    if np.any(c):
                        alpha = (y[c] - self.y_list[i - 1]) / (
                            self.y_list[i] - self.y_list[i - 1]
                        )
                        beta = (z[c] - self.z_list[j - 1]) / (
                            self.z_list[j] - self.z_list[j - 1]
                        )
                        f[c] = (
                            (1 - alpha)
                            * (1 - beta)
                            * self.wxInterpolators[i - 1][j - 1](w[c], x[c])
                            + (1 - alpha)
                            * beta
                            * self.wxInterpolators[i - 1][j](w[c], x[c])
                            + alpha
                            * (1 - beta)
                            * self.wxInterpolators[i][j - 1](w[c], x[c])
                            + alpha * beta * self.wxInterpolators[i][j](w[c], x[c])
                        )
        return f

    def _derW(self, w, x, y, z):
        """
        Returns the derivative with respect to w of the interpolated function
        at each value in w,x,y,z. Only called internally by HARKinterpolator4D.derivativeW.
        """
        # This may look strange, as we call the derivativeX() method to get the
        # derivative with respect to w, but that's just a quirk of 4D interpolations
        # beginning with w rather than x.  The derivative wrt the first dimension
        # of an element of wxInterpolators is the w-derivative of the main function.
        if _isscalar(x):
            y_pos = max(min(np.searchsorted(self.y_list, y), self.y_n - 1), 1)
            z_pos = max(min(np.searchsorted(self.z_list, z), self.z_n - 1), 1)
            alpha = (y - self.y_list[y_pos - 1]) / (
                self.y_list[y_pos] - self.y_list[y_pos - 1]
            )
            beta = (z - self.z_list[z_pos - 1]) / (
                self.z_list[z_pos] - self.z_list[z_pos - 1]
            )
            dfdw = (
                (1 - alpha)
                * (1 - beta)
                * self.wxInterpolators[y_pos - 1][z_pos - 1].derivativeX(w, x)
                + (1 - alpha)
                * beta
                * self.wxInterpolators[y_pos - 1][z_pos].derivativeX(w, x)
                + alpha
                * (1 - beta)
                * self.wxInterpolators[y_pos][z_pos - 1].derivativeX(w, x)
                + alpha * beta * self.wxInterpolators[y_pos][z_pos].derivativeX(w, x)
            )
        else:
            m = len(x)
            y_pos = np.searchsorted(self.y_list, y)
            y_pos[y_pos > self.y_n - 1] = self.y_n - 1
            y_pos[y_pos < 1] = 1
            z_pos = np.searchsorted(self.z_list, z)
            z_pos[z_pos > self.z_n - 1] = self.z_n - 1
            z_pos[z_pos < 1] = 1
            dfdw = np.zeros(m) + np.nan
            for i in range(1, self.y_n):
                for j in range(1, self.z_n):
                    c = np.logical_and(i == y_pos, j == z_pos)
                    if np.any(c):
                        alpha = (y[c] - self.y_list[i - 1]) / (
                            self.y_list[i] - self.y_list[i - 1]
                        )
                        beta = (z[c] - self.z_list[j - 1]) / (
                            self.z_list[j] - self.z_list[j - 1]
                        )
                        dfdw[c] = (
                            (1 - alpha)
                            * (1 - beta)
                            * self.wxInterpolators[i - 1][j - 1].derivativeX(w[c], x[c])
                            + (1 - alpha)
                            * beta
                            * self.wxInterpolators[i - 1][j].derivativeX(w[c], x[c])
                            + alpha
                            * (1 - beta)
                            * self.wxInterpolators[i][j - 1].derivativeX(w[c], x[c])
                            + alpha
                            * beta
                            * self.wxInterpolators[i][j].derivativeX(w[c], x[c])
                        )
        return dfdw

    def _derX(self, w, x, y, z):
        """
        Returns the derivative with respect to x of the interpolated function
        at each value in w,x,y,z. Only called internally by HARKinterpolator4D.derivativeX.
        """
        # This may look strange, as we call the derivativeY() method to get the
        # derivative with respect to x, but that's just a quirk of 4D interpolations
        # beginning with w rather than x.  The derivative wrt the second dimension
        # of an element of wxInterpolators is the x-derivative of the main function.
        if _isscalar(x):
            y_pos = max(min(np.searchsorted(self.y_list, y), self.y_n - 1), 1)
            z_pos = max(min(np.searchsorted(self.z_list, z), self.z_n - 1), 1)
            alpha = (y - self.y_list[y_pos - 1]) / (
                self.y_list[y_pos] - self.y_list[y_pos - 1]
            )
            beta = (z - self.z_list[z_pos - 1]) / (
                self.z_list[z_pos] - self.z_list[z_pos - 1]
            )
            dfdx = (
                (1 - alpha)
                * (1 - beta)
                * self.wxInterpolators[y_pos - 1][z_pos - 1].derivativeY(w, x)
                + (1 - alpha)
                * beta
                * self.wxInterpolators[y_pos - 1][z_pos].derivativeY(w, x)
                + alpha
                * (1 - beta)
                * self.wxInterpolators[y_pos][z_pos - 1].derivativeY(w, x)
                + alpha * beta * self.wxInterpolators[y_pos][z_pos].derivativeY(w, x)
            )
        else:
            m = len(x)
            y_pos = np.searchsorted(self.y_list, y)
            y_pos[y_pos > self.y_n - 1] = self.y_n - 1
            y_pos[y_pos < 1] = 1
            z_pos = np.searchsorted(self.z_list, z)
            z_pos[z_pos > self.z_n - 1] = self.z_n - 1
            z_pos[z_pos < 1] = 1
            dfdx = np.zeros(m) + np.nan
            for i in range(1, self.y_n):
                for j in range(1, self.z_n):
                    c = np.logical_and(i == y_pos, j == z_pos)
                    if np.any(c):
                        alpha = (y[c] - self.y_list[i - 1]) / (
                            self.y_list[i] - self.y_list[i - 1]
                        )
                        beta = (z[c] - self.z_list[j - 1]) / (
                            self.z_list[j] - self.z_list[j - 1]
                        )
                        dfdx[c] = (
                            (1 - alpha)
                            * (1 - beta)
                            * self.wxInterpolators[i - 1][j - 1].derivativeY(w[c], x[c])
                            + (1 - alpha)
                            * beta
                            * self.wxInterpolators[i - 1][j].derivativeY(w[c], x[c])
                            + alpha
                            * (1 - beta)
                            * self.wxInterpolators[i][j - 1].derivativeY(w[c], x[c])
                            + alpha
                            * beta
                            * self.wxInterpolators[i][j].derivativeY(w[c], x[c])
                        )
        return dfdx

    def _derY(self, w, x, y, z):
        """
        Returns the derivative with respect to y of the interpolated function
        at each value in w,x,y,z. Only called internally by HARKinterpolator4D.derivativeY.
        """
        if _isscalar(x):
            y_pos = max(min(np.searchsorted(self.y_list, y), self.y_n - 1), 1)
            z_pos = max(min(np.searchsorted(self.z_list, z), self.z_n - 1), 1)
            beta = (z - self.z_list[z_pos - 1]) / (
                self.z_list[z_pos] - self.z_list[z_pos - 1]
            )
            dfdy = (
                (
                    (1 - beta) * self.wxInterpolators[y_pos][z_pos - 1](w, x)
                    + beta * self.wxInterpolators[y_pos][z_pos](w, x)
                )
                - (
                    (1 - beta) * self.wxInterpolators[y_pos - 1][z_pos - 1](w, x)
                    + beta * self.wxInterpolators[y_pos - 1][z_pos](w, x)
                )
            ) / (self.y_list[y_pos] - self.y_list[y_pos - 1])
        else:
            m = len(x)
            y_pos = np.searchsorted(self.y_list, y)
            y_pos[y_pos > self.y_n - 1] = self.y_n - 1
            y_pos[y_pos < 1] = 1
            z_pos = np.searchsorted(self.z_list, z)
            z_pos[z_pos > self.z_n - 1] = self.z_n - 1
            z_pos[z_pos < 1] = 1
            dfdy = np.zeros(m) + np.nan
            for i in range(1, self.y_n):
                for j in range(1, self.z_n):
                    c = np.logical_and(i == y_pos, j == z_pos)
                    if np.any(c):
                        beta = (z[c] - self.z_list[j - 1]) / (
                            self.z_list[j] - self.z_list[j - 1]
                        )
                        dfdy[c] = (
                            (
                                (1 - beta) * self.wxInterpolators[i][j - 1](w[c], x[c])
                                + beta * self.wxInterpolators[i][j](w[c], x[c])
                            )
                            - (
                                (1 - beta)
                                * self.wxInterpolators[i - 1][j - 1](w[c], x[c])
                                + beta * self.wxInterpolators[i - 1][j](w[c], x[c])
                            )
                        ) / (self.y_list[i] - self.y_list[i - 1])
        return dfdy

    def _derZ(self, w, x, y, z):
        """
        Returns the derivative with respect to z of the interpolated function
        at each value in w,x,y,z. Only called internally by HARKinterpolator4D.derivativeZ.
        """
        if _isscalar(x):
            y_pos = max(min(np.searchsorted(self.y_list, y), self.y_n - 1), 1)
            z_pos = max(min(np.searchsorted(self.z_list, z), self.z_n - 1), 1)
            alpha = (y - self.y_list[y_pos - 1]) / (
                self.y_list[y_pos] - self.y_list[y_pos - 1]
            )
            dfdz = (
                (
                    (1 - alpha) * self.wxInterpolators[y_pos - 1][z_pos](w, x)
                    + alpha * self.wxInterpolators[y_pos][z_pos](w, x)
                )
                - (
                    (1 - alpha) * self.wxInterpolators[y_pos - 1][z_pos - 1](w, x)
                    + alpha * self.wxInterpolators[y_pos][z_pos - 1](w, x)
                )
            ) / (self.z_list[z_pos] - self.z_list[z_pos - 1])
        else:
            m = len(x)
            y_pos = np.searchsorted(self.y_list, y)
            y_pos[y_pos > self.y_n - 1] = self.y_n - 1
            y_pos[y_pos < 1] = 1
            z_pos = np.searchsorted(self.z_list, z)
            z_pos[z_pos > self.z_n - 1] = self.z_n - 1
            z_pos[z_pos < 1] = 1
            dfdz = np.zeros(m) + np.nan
            for i in range(1, self.y_n):
                for j in range(1, self.z_n):
                    c = np.logical_and(i == y_pos, j == z_pos)
                    if np.any(c):
                        alpha = (y[c] - self.y_list[i - 1]) / (
                            self.y_list[i] - self.y_list[i - 1]
                        )
                        dfdz[c] = (
                            (
                                (1 - alpha) * self.wxInterpolators[i - 1][j](w[c], x[c])
                                + alpha * self.wxInterpolators[i][j](w[c], x[c])
                            )
                            - (
                                (1 - alpha)
                                * self.wxInterpolators[i - 1][j - 1](w[c], x[c])
                                + alpha * self.wxInterpolators[i][j - 1](w[c], x[c])
                            )
                        ) / (self.z_list[j] - self.z_list[j - 1])
        return dfdz


class Curvilinear2DInterp(HARKinterpolator2D):
    """
    A 2D interpolation method for curvilinear or "warped grid" interpolation, as
    in White (2015).  Used for models with two endogenous states that are solved
    with the endogenous grid method.

    Parameters
    ----------
    f_values: numpy.array
        A 2D array of function values such that f_values[i,j] =
        f(x_values[i,j],y_values[i,j]).
    x_values: numpy.array
        A 2D array of x values of the same size as f_values.
    y_values: numpy.array
        A 2D array of y values of the same size as f_values.
    """

    distance_criteria = ["f_values", "x_values", "y_values"]

    def __init__(self, f_values, x_values, y_values):
        self.f_values = f_values
        self.x_values = x_values
        self.y_values = y_values
        my_shape = f_values.shape
        self.x_n = my_shape[0]
        self.y_n = my_shape[1]
        self.update_polarity()

    def update_polarity(self):
        """
        Fills in the polarity attribute of the interpolation, determining whether
        the "plus" (True) or "minus" (False) solution of the system of equations
        should be used for each sector.  Needs to be called in __init__.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        # Grab a point known to be inside each sector: the midway point between
        # the lower left and upper right vertex of each sector
        x_temp = 0.5 * (
            self.x_values[0: (self.x_n - 1), 0: (self.y_n - 1)]
            + self.x_values[1: self.x_n, 1: self.y_n]
        )
        y_temp = 0.5 * (
            self.y_values[0: (self.x_n - 1), 0: (self.y_n - 1)]
            + self.y_values[1: self.x_n, 1: self.y_n]
        )
        size = (self.x_n - 1) * (self.y_n - 1)
        x_temp = np.reshape(x_temp, size)
        y_temp = np.reshape(y_temp, size)
        y_pos = np.tile(np.arange(0, self.y_n - 1), self.x_n - 1)
        x_pos = np.reshape(
            np.tile(np.arange(0, self.x_n - 1), (self.y_n - 1, 1)).transpose(), size
        )

        # Set the polarity of all sectors to "plus", then test each sector
        self.polarity = np.ones((self.x_n - 1, self.y_n - 1), dtype=bool)
        alpha, beta = self.find_coords(x_temp, y_temp, x_pos, y_pos)
        polarity = np.logical_and(
            np.logical_and(alpha > 0, alpha < 1), np.logical_and(beta > 0, beta < 1)
        )

        # Update polarity: if (alpha,beta) not in the unit square, then that
        # sector must use the "minus" solution instead
        self.polarity = np.reshape(polarity, (self.x_n - 1, self.y_n - 1))

    def find_sector(self, x, y):
        """
        Finds the quadrilateral "sector" for each (x,y) point in the input.
        Only called as a subroutine of _evaluate().

        Parameters
        ----------
        x : np.array
            Values whose sector should be found.
        y : np.array
            Values whose sector should be found.  Should be same size as x.

        Returns
        -------
        x_pos : np.array
            Sector x-coordinates for each point of the input, of the same size.
        y_pos : np.array
            Sector y-coordinates for each point of the input, of the same size.
        """
        # Initialize the sector guess
        m = x.size
        x_pos_guess = (np.ones(m) * self.x_n / 2).astype(int)
        y_pos_guess = (np.ones(m) * self.y_n / 2).astype(int)

        # Define a function that checks whether a set of points violates a linear
        # boundary defined by (x_bound_1,y_bound_1) and (x_bound_2,y_bound_2),
        # where the latter is *COUNTER CLOCKWISE* from the former.  Returns
        # 1 if the point is outside the boundary and 0 otherwise.
        violation_check = (
            lambda x_check, y_check, x_bound_1, y_bound_1, x_bound_2, y_bound_2: (
                (y_bound_2 - y_bound_1) * x_check - (x_bound_2 - x_bound_1) * y_check
                > x_bound_1 * y_bound_2 - y_bound_1 * x_bound_2
            )
            + 0
        )

        # Identify the correct sector for each point to be evaluated
        these = np.ones(m, dtype=bool)
        max_loops = self.x_n + self.y_n
        loops = 0
        while np.any(these) and loops < max_loops:
            # Get coordinates for the four vertices: (xA,yA),...,(xD,yD)
            x_temp = x[these]
            y_temp = y[these]
            xA = self.x_values[x_pos_guess[these], y_pos_guess[these]]
            xB = self.x_values[x_pos_guess[these] + 1, y_pos_guess[these]]
            xC = self.x_values[x_pos_guess[these], y_pos_guess[these] + 1]
            xD = self.x_values[x_pos_guess[these] + 1, y_pos_guess[these] + 1]
            yA = self.y_values[x_pos_guess[these], y_pos_guess[these]]
            yB = self.y_values[x_pos_guess[these] + 1, y_pos_guess[these]]
            yC = self.y_values[x_pos_guess[these], y_pos_guess[these] + 1]
            yD = self.y_values[x_pos_guess[these] + 1, y_pos_guess[these] + 1]

            # Check the "bounding box" for the sector: is this guess plausible?
            move_down = (y_temp < np.minimum(yA, yB)) + 0
            move_right = (x_temp > np.maximum(xB, xD)) + 0
            move_up = (y_temp > np.maximum(yC, yD)) + 0
            move_left = (x_temp < np.minimum(xA, xC)) + 0

            # Check which boundaries are violated (and thus where to look next)
            c = (move_down + move_right + move_up + move_left) == 0
            move_down[c] = violation_check(
                x_temp[c], y_temp[c], xA[c], yA[c], xB[c], yB[c]
            )
            move_right[c] = violation_check(
                x_temp[c], y_temp[c], xB[c], yB[c], xD[c], yD[c]
            )
            move_up[c] = violation_check(
                x_temp[c], y_temp[c], xD[c], yD[c], xC[c], yC[c]
            )
            move_left[c] = violation_check(
                x_temp[c], y_temp[c], xC[c], yC[c], xA[c], yA[c]
            )

            # Update the sector guess based on the violations
            x_pos_next = x_pos_guess[these] - move_left + move_right
            x_pos_next[x_pos_next < 0] = 0
            x_pos_next[x_pos_next > (self.x_n - 2)] = self.x_n - 2
            y_pos_next = y_pos_guess[these] - move_down + move_up
            y_pos_next[y_pos_next < 0] = 0
            y_pos_next[y_pos_next > (self.y_n - 2)] = self.y_n - 2

            # Check which sectors have not changed, and mark them as complete
            no_move = np.array(
                np.logical_and(
                    x_pos_guess[these] == x_pos_next, y_pos_guess[these] == y_pos_next
                )
            )
            x_pos_guess[these] = x_pos_next
            y_pos_guess[these] = y_pos_next
            temp = these.nonzero()
            these[temp[0][no_move]] = False

            # Move to the next iteration of the search
            loops += 1

        # Return the output
        x_pos = x_pos_guess
        y_pos = y_pos_guess
        return x_pos, y_pos

    def find_coords(self, x, y, x_pos, y_pos):
        """
        Calculates the relative coordinates (alpha,beta) for each point (x,y),
        given the sectors (x_pos,y_pos) in which they reside.  Only called as
        a subroutine of __call__().

        Parameters
        ----------
        x : np.array
            Values whose sector should be found.
        y : np.array
            Values whose sector should be found.  Should be same size as x.
        x_pos : np.array
            Sector x-coordinates for each point in (x,y), of the same size.
        y_pos : np.array
            Sector y-coordinates for each point in (x,y), of the same size.

        Returns
        -------
        alpha : np.array
            Relative "horizontal" position of the input in their respective sectors.
        beta : np.array
            Relative "vertical" position of the input in their respective sectors.
        """
        # Calculate relative coordinates in the sector for each point
        xA = self.x_values[x_pos, y_pos]
        xB = self.x_values[x_pos + 1, y_pos]
        xC = self.x_values[x_pos, y_pos + 1]
        xD = self.x_values[x_pos + 1, y_pos + 1]
        yA = self.y_values[x_pos, y_pos]
        yB = self.y_values[x_pos + 1, y_pos]
        yC = self.y_values[x_pos, y_pos + 1]
        yD = self.y_values[x_pos + 1, y_pos + 1]
        polarity = 2.0 * self.polarity[x_pos, y_pos] - 1.0
        a = xA
        b = xB - xA
        c = xC - xA
        d = xA - xB - xC + xD
        e = yA
        f = yB - yA
        g = yC - yA
        h = yA - yB - yC + yD
        denom = d * g - h * c
        mu = (h * b - d * f) / denom
        tau = (h * (a - x) - d * (e - y)) / denom
        zeta = a - x + c * tau
        eta = b + c * mu + d * tau
        theta = d * mu
        alpha = (-eta + polarity * np.sqrt(eta ** 2.0 - 4.0 * zeta * theta)) / (
            2.0 * theta
        )
        beta = mu * alpha + tau

        # Alternate method if there are sectors that are "too regular"
        z = np.logical_or(
            np.isnan(alpha), np.isnan(beta)
        )  # These points weren't able to identify coordinates
        if np.any(z):
            these = np.isclose(
                f / b, (yD - yC) / (xD - xC)
            )  # iso-beta lines have equal slope
            if np.any(these):
                kappa = f[these] / b[these]
                int_bot = yA[these] - kappa * xA[these]
                int_top = yC[these] - kappa * xC[these]
                int_these = y[these] - kappa * x[these]
                beta_temp = (int_these - int_bot) / (int_top - int_bot)
                x_left = beta_temp * xC[these] + (1.0 - beta_temp) * xA[these]
                x_right = beta_temp * xD[these] + (1.0 - beta_temp) * xB[these]
                alpha_temp = (x[these] - x_left) / (x_right - x_left)
                beta[these] = beta_temp
                alpha[these] = alpha_temp

            # print(np.sum(np.isclose(g/c,(yD-yB)/(xD-xB))))

        return alpha, beta

    def _evaluate(self, x, y):
        """
        Returns the level of the interpolated function at each value in x,y.
        Only called internally by HARKinterpolator2D.__call__ (etc).
        """
        x_pos, y_pos = self.find_sector(x, y)
        alpha, beta = self.find_coords(x, y, x_pos, y_pos)

        # Calculate the function at each point using bilinear interpolation
        f = (
            (1 - alpha) * (1 - beta) * self.f_values[x_pos, y_pos]
            + (1 - alpha) * beta * self.f_values[x_pos, y_pos + 1]
            + alpha * (1 - beta) * self.f_values[x_pos + 1, y_pos]
            + alpha * beta * self.f_values[x_pos + 1, y_pos + 1]
        )
        return f

    def _derX(self, x, y):
        """
        Returns the derivative with respect to x of the interpolated function
        at each value in x,y. Only called internally by HARKinterpolator2D.derivativeX.
        """
        x_pos, y_pos = self.find_sector(x, y)
        alpha, beta = self.find_coords(x, y, x_pos, y_pos)

        # Get four corners data for each point
        xA = self.x_values[x_pos, y_pos]
        xB = self.x_values[x_pos + 1, y_pos]
        xC = self.x_values[x_pos, y_pos + 1]
        xD = self.x_values[x_pos + 1, y_pos + 1]
        yA = self.y_values[x_pos, y_pos]
        yB = self.y_values[x_pos + 1, y_pos]
        yC = self.y_values[x_pos, y_pos + 1]
        yD = self.y_values[x_pos + 1, y_pos + 1]
        fA = self.f_values[x_pos, y_pos]
        fB = self.f_values[x_pos + 1, y_pos]
        fC = self.f_values[x_pos, y_pos + 1]
        fD = self.f_values[x_pos + 1, y_pos + 1]

        # Calculate components of the alpha,beta --> x,y delta translation matrix
        alpha_x = (1 - beta) * (xB - xA) + beta * (xD - xC)
        alpha_y = (1 - beta) * (yB - yA) + beta * (yD - yC)
        beta_x = (1 - alpha) * (xC - xA) + alpha * (xD - xB)
        beta_y = (1 - alpha) * (yC - yA) + alpha * (yD - yB)

        # Invert the delta translation matrix into x,y --> alpha,beta
        det = alpha_x * beta_y - beta_x * alpha_y
        x_alpha = beta_y / det
        x_beta = -alpha_y / det

        # Calculate the derivative of f w.r.t. alpha and beta
        dfda = (1 - beta) * (fB - fA) + beta * (fD - fC)
        dfdb = (1 - alpha) * (fC - fA) + alpha * (fD - fB)

        # Calculate the derivative with respect to x (and return it)
        dfdx = x_alpha * dfda + x_beta * dfdb
        return dfdx

    def _derY(self, x, y):
        """
        Returns the derivative with respect to y of the interpolated function
        at each value in x,y. Only called internally by HARKinterpolator2D.derivativeX.
        """
        x_pos, y_pos = self.find_sector(x, y)
        alpha, beta = self.find_coords(x, y, x_pos, y_pos)

        # Get four corners data for each point
        xA = self.x_values[x_pos, y_pos]
        xB = self.x_values[x_pos + 1, y_pos]
        xC = self.x_values[x_pos, y_pos + 1]
        xD = self.x_values[x_pos + 1, y_pos + 1]
        yA = self.y_values[x_pos, y_pos]
        yB = self.y_values[x_pos + 1, y_pos]
        yC = self.y_values[x_pos, y_pos + 1]
        yD = self.y_values[x_pos + 1, y_pos + 1]
        fA = self.f_values[x_pos, y_pos]
        fB = self.f_values[x_pos + 1, y_pos]
        fC = self.f_values[x_pos, y_pos + 1]
        fD = self.f_values[x_pos + 1, y_pos + 1]

        # Calculate components of the alpha,beta --> x,y delta translation matrix
        alpha_x = (1 - beta) * (xB - xA) + beta * (xD - xC)
        alpha_y = (1 - beta) * (yB - yA) + beta * (yD - yC)
        beta_x = (1 - alpha) * (xC - xA) + alpha * (xD - xB)
        beta_y = (1 - alpha) * (yC - yA) + alpha * (yD - yB)

        # Invert the delta translation matrix into x,y --> alpha,beta
        det = alpha_x * beta_y - beta_x * alpha_y
        y_alpha = -beta_x / det
        y_beta = alpha_x / det

        # Calculate the derivative of f w.r.t. alpha and beta
        dfda = (1 - beta) * (fB - fA) + beta * (fD - fC)
        dfdb = (1 - alpha) * (fC - fA) + alpha * (fD - fB)

        # Calculate the derivative with respect to x (and return it)
        dfdy = y_alpha * dfda + y_beta * dfdb
        return dfdy


###############################################################################
## Functions used in discrete choice models with T1EV taste shocks ############
###############################################################################


def calc_log_sum_choice_probs(Vals, sigma):
    """
    Returns the final optimal value and choice probabilities given the choice
    specific value functions `Vals`. Probabilities are degenerate if sigma == 0.0.
    Parameters
    ----------
    Vals : [numpy.array]
        A numpy.array that holds choice specific values at common grid points.
    sigma : float
        A number that controls the variance of the taste shocks
    Returns
    -------
    V : [numpy.array]
        A numpy.array that holds the integrated value function.
    P : [numpy.array]
        A numpy.array that holds the discrete choice probabilities
    """
    # Assumes that NaNs have been replaced by -numpy.inf or similar
    if sigma == 0.0:
        # We could construct a linear index here and use unravel_index.
        Pflat = np.argmax(Vals, axis=0)

        V = np.zeros(Vals[0].shape)
        Probs = np.zeros(Vals.shape)
        for i in range(Vals.shape[0]):
            optimalIndices = Pflat == i
            V[optimalIndices] = Vals[i][optimalIndices]
            Probs[i][optimalIndices] = 1
        return V, Probs

    # else we have a taste shock
    maxV = np.max(Vals, axis=0)

    # calculate maxV+sigma*log(sum_i=1^J exp((V[i]-maxV))/sigma)
    sumexp = np.sum(np.exp((Vals - maxV) / sigma), axis=0)
    LogSumV = np.log(sumexp)
    LogSumV = maxV + sigma * LogSumV

    Probs = np.exp((Vals - LogSumV) / sigma)
    return LogSumV, Probs


def calc_choice_probs(Vals, sigma):
    """
    Returns the choice probabilities given the choice specific value functions
    `Vals`. Probabilities are degenerate if sigma == 0.0.
    Parameters
    ----------
    Vals : [numpy.array]
        A numpy.array that holds choice specific values at common grid points.
    sigma : float
        A number that controls the variance of the taste shocks
    Returns
    -------
    Probs : [numpy.array]
        A numpy.array that holds the discrete choice probabilities
    """

    # Assumes that NaNs have been replaced by -numpy.inf or similar
    if sigma == 0.0:
        # We could construct a linear index here and use unravel_index.
        Pflat = np.argmax(Vals, axis=0)
        Probs = np.zeros(Vals.shape)
        for i in range(Vals.shape[0]):
            Probs[i][Pflat == i] = 1
        return Probs

    maxV = np.max(Vals, axis=0)
    Probs = np.divide(
        np.exp((Vals - maxV) / sigma), np.sum(np.exp((Vals - maxV) / sigma), axis=0)
    )
    return Probs


def calc_log_sum(Vals, sigma):
    """
    Returns the optimal value given the choice specific value functions Vals.
    Parameters
    ----------
    Vals : [numpy.array]
        A numpy.array that holds choice specific values at common grid points.
    sigma : float
        A number that controls the variance of the taste shocks
    Returns
    -------
    V : [numpy.array]
        A numpy.array that holds the integrated value function.
    """

    # Assumes that NaNs have been replaced by -numpy.inf or similar
    if sigma == 0.0:
        # We could construct a linear index here and use unravel_index.
        V = np.amax(Vals, axis=0)
        return V

    # else we have a taste shock
    maxV = np.max(Vals, axis=0)

    # calculate maxV+sigma*log(sum_i=1^J exp((V[i]-maxV))/sigma)
    sumexp = np.sum(np.exp((Vals - maxV) / sigma), axis=0)
    LogSumV = np.log(sumexp)
    LogSumV = maxV + sigma * LogSumV
    return LogSumV

###############################################################################
# Tools for value and marginal-value functions in models where                #
# - dvdm = u'(c).                                                             #
# - u is of the CRRA family.                                                  #
###############################################################################


class ValueFuncCRRA(MetricObject):
    """
    A class for representing a value function.  The underlying interpolation is
    in the space of (state,u_inv(v)); this class "re-curves" to the value function.

    Parameters
    ----------
    vFuncNvrs : function
        A real function representing the value function composed with the
        inverse utility function, defined on the state: u_inv(vFunc(state))
    CRRA : float
        Coefficient of relative risk aversion.
    """

    distance_criteria = ["func", "CRRA"]

    def __init__(self, vFuncNvrs, CRRA):
        self.func = deepcopy(vFuncNvrs)
        self.CRRA = CRRA

    def __call__(self, *vFuncArgs):
        """
        Evaluate the value function at given levels of market resources m.

        Parameters
        ----------
        vFuncArgs : floats or np.arrays, all of the same dimensions.
            Values for the state variables. These usually start with 'm',
            market resources normalized by the level of permanent income.

        Returns
        -------
        v : float or np.array
            Lifetime value of beginning this period with the given states; has
            same size as the state inputs.
        """
        return CRRAutility(self.func(*vFuncArgs), gam=self.CRRA)


class MargValueFuncCRRA(MetricObject):
    """
    A class for representing a marginal value function in models where the
    standard envelope condition of dvdm(state) = u'(c(state)) holds (with CRRA utility).

    Parameters
    ----------
    cFunc : function.
        Its first argument must be normalized market resources m.
        A real function representing the marginal value function composed
        with the inverse marginal utility function, defined on the state
        variables: uP_inv(dvdmFunc(state)).  Called cFunc because when standard
        envelope condition applies, uP_inv(dvdm(state)) = cFunc(state).
    CRRA : float
        Coefficient of relative risk aversion.
    """

    distance_criteria = ["cFunc", "CRRA"]

    def __init__(self, cFunc, CRRA):
        self.cFunc = deepcopy(cFunc)
        self.CRRA = CRRA

    def __call__(self, *cFuncArgs):
        """
        Evaluate the marginal value function at given levels of market resources m.

        Parameters
        ----------
        cFuncArgs : floats or np.arrays
            Values of the state variables at which to evaluate the marginal
            value function.

        Returns
        -------
        vP : float or np.array
            Marginal lifetime value of beginning this period with state
            cFuncArgs
        """
        return CRRAutilityP(self.cFunc(*cFuncArgs), gam=self.CRRA)

    def derivativeX(self, *cFuncArgs):
        """
        Evaluate the derivative of the marginal value function with respect to
        market resources at given state; this is the marginal marginal value
        function.

        Parameters
        ----------
        cFuncArgs : floats or np.arrays
            State variables.

        Returns
        -------
        vPP : float or np.array
            Marginal marginal lifetime value of beginning this period with
            state cFuncArgs; has same size as inputs.

        """

        # The derivative method depends on the dimension of the function
        if isinstance(self.cFunc, (HARKinterpolator1D)):
            c, MPC = self.cFunc.eval_with_derivative(*cFuncArgs)

        elif hasattr(self.cFunc, 'derivativeX'):
            c = self.cFunc(*cFuncArgs)
            MPC = self.cFunc.derivativeX(*cFuncArgs)

        else:
            raise Exception(
                "cFunc does not have a 'derivativeX' attribute. Can't compute"
                + "marginal marginal value."
            )

        return MPC * CRRAutilityPP(c, gam=self.CRRA)


class MargMargValueFuncCRRA(MetricObject):
    """
    A class for representing a marginal marginal value function in models where
    the standard envelope condition of dvdm = u'(c(state)) holds (with CRRA utility).

    Parameters
    ----------
    cFunc : function.
        Its first argument must be normalized market resources m.
        A real function representing the marginal value function composed
        with the inverse marginal utility function, defined on the state
        variables: uP_inv(dvdmFunc(state)).  Called cFunc because when standard
        envelope condition applies, uP_inv(dvdm(state)) = cFunc(state).
    CRRA : float
        Coefficient of relative risk aversion.
    """

    distance_criteria = ["cFunc", "CRRA"]

    def __init__(self, cFunc, CRRA):
        self.cFunc = deepcopy(cFunc)
        self.CRRA = CRRA

    def __call__(self, *cFuncArgs):
        """
        Evaluate the marginal marginal value function at given levels of market
        resources m.

        Parameters
        ----------
        m : float or np.array
            Market resources (normalized by permanent income) whose marginal
            marginal value is to be found.

        Returns
        -------
        vPP : float or np.array
            Marginal marginal lifetime value of beginning this period with market
            resources m; has same size as input m.
        """

        # The derivative method depends on the dimension of the function
        if isinstance(self.cFunc, (HARKinterpolator1D)):
            c, MPC = self.cFunc.eval_with_derivative(*cFuncArgs)

        elif hasattr(self.cFunc, 'derivativeX'):
            c = self.cFunc(*cFuncArgs)
            MPC = self.cFunc.derivativeX(*cFuncArgs)

        else:
            raise Exception(
                "cFunc does not have a 'derivativeX' attribute. Can't compute"
                + "marginal marginal value."
            )

        return MPC * CRRAutilityPP(c, gam=self.CRRA)

##############################################################################
# Examples and tests
##############################################################################


def main():
    print("Sorry, HARK.interpolation doesn't actually do much on its own.")
    print("To see some examples of its interpolation methods in action, look at any")
    print("of the model modules in /ConsumptionSavingModel.  In the future, running")
    print("this module will show examples of each interpolation class.")

    from time import time
    import matplotlib.pyplot as plt

    RNG = np.random.RandomState(123)

    if False:
        x = np.linspace(1, 20, 39)
        y = np.log(x)
        dydx = 1.0 / x
        f = CubicInterp(x, y, dydx)
        x_test = np.linspace(0, 30, 200)
        y_test = f(x_test)
        plt.plot(x_test, y_test)
        plt.show()

    if False:
        def f(x, y): return 3.0 * x ** 2.0 + x * y + 4.0 * y ** 2.0
        def dfdx(x, y): return 6.0 * x + y
        def dfdy(x, y): return x + 8.0 * y

        y_list = np.linspace(0, 5, 100, dtype=float)
        xInterpolators = []
        xInterpolators_alt = []
        for y in y_list:
            this_x_list = np.sort((RNG.rand(100) * 5.0))
            this_interpolation = LinearInterp(
                this_x_list, f(this_x_list, y * np.ones(this_x_list.size))
            )
            that_interpolation = CubicInterp(
                this_x_list,
                f(this_x_list, y * np.ones(this_x_list.size)),
                dfdx(this_x_list, y * np.ones(this_x_list.size)),
            )
            xInterpolators.append(this_interpolation)
            xInterpolators_alt.append(that_interpolation)
        g = LinearInterpOnInterp1D(xInterpolators, y_list)
        h = LinearInterpOnInterp1D(xInterpolators_alt, y_list)

        rand_x = RNG.rand(100) * 5.0
        rand_y = RNG.rand(100) * 5.0
        z = (f(rand_x, rand_y) - g(rand_x, rand_y)) / f(rand_x, rand_y)
        q = (dfdx(rand_x, rand_y) - g.derivativeX(rand_x, rand_y)) / dfdx(
            rand_x, rand_y
        )
        r = (dfdy(rand_x, rand_y) - g.derivativeY(rand_x, rand_y)) / dfdy(
            rand_x, rand_y
        )
        # print(z)
        # print(q)
        # print(r)

        z = (f(rand_x, rand_y) - g(rand_x, rand_y)) / f(rand_x, rand_y)
        q = (dfdx(rand_x, rand_y) - g.derivativeX(rand_x, rand_y)) / dfdx(
            rand_x, rand_y
        )
        r = (dfdy(rand_x, rand_y) - g.derivativeY(rand_x, rand_y)) / dfdy(
            rand_x, rand_y
        )
        print(z)
        # print(q)
        # print(r)

    if False:
        f = (
            lambda x, y, z: 3.0 * x ** 2.0
            + x * y
            + 4.0 * y ** 2.0
            - 5 * z ** 2.0
            + 1.5 * x * z
        )
        def dfdx(x, y, z): return 6.0 * x + y + 1.5 * z
        def dfdy(x, y, z): return x + 8.0 * y
        def dfdz(x, y, z): return -10.0 * z + 1.5 * x

        y_list = np.linspace(0, 5, 51, dtype=float)
        z_list = np.linspace(0, 5, 51, dtype=float)
        xInterpolators = []
        for y in y_list:
            temp = []
            for z in z_list:
                this_x_list = np.sort((RNG.rand(100) * 5.0))
                this_interpolation = LinearInterp(
                    this_x_list,
                    f(
                        this_x_list,
                        y * np.ones(this_x_list.size),
                        z * np.ones(this_x_list.size),
                    ),
                )
                temp.append(this_interpolation)
            xInterpolators.append(deepcopy(temp))
        g = BilinearInterpOnInterp1D(xInterpolators, y_list, z_list)

        rand_x = RNG.rand(1000) * 5.0
        rand_y = RNG.rand(1000) * 5.0
        rand_z = RNG.rand(1000) * 5.0
        z = (f(rand_x, rand_y, rand_z) - g(rand_x, rand_y, rand_z)) / f(
            rand_x, rand_y, rand_z
        )
        q = (
            dfdx(rand_x, rand_y, rand_z) - g.derivativeX(rand_x, rand_y, rand_z)
        ) / dfdx(rand_x, rand_y, rand_z)
        r = (
            dfdy(rand_x, rand_y, rand_z) - g.derivativeY(rand_x, rand_y, rand_z)
        ) / dfdy(rand_x, rand_y, rand_z)
        p = (
            dfdz(rand_x, rand_y, rand_z) - g.derivativeZ(rand_x, rand_y, rand_z)
        ) / dfdz(rand_x, rand_y, rand_z)
        z.sort()

    if False:
        f = (
            lambda w, x, y, z: 4.0 * w * z
            - 2.5 * w * x
            + w * y
            + 6.0 * x * y
            - 10.0 * x * z
            + 3.0 * y * z
            - 7.0 * z
            + 4.0 * x
            + 2.0 * y
            - 5.0 * w
        )
        def dfdw(w, x, y, z): return 4.0 * z - 2.5 * x + y - 5.0
        def dfdx(w, x, y, z): return -2.5 * w + 6.0 * y - 10.0 * z + 4.0
        def dfdy(w, x, y, z): return w + 6.0 * x + 3.0 * z + 2.0
        def dfdz(w, x, y, z): return 4.0 * w - 10.0 * x + 3.0 * y - 7

        x_list = np.linspace(0, 5, 16, dtype=float)
        y_list = np.linspace(0, 5, 16, dtype=float)
        z_list = np.linspace(0, 5, 16, dtype=float)
        wInterpolators = []
        for x in x_list:
            temp = []
            for y in y_list:
                temptemp = []
                for z in z_list:
                    this_w_list = np.sort((RNG.rand(16) * 5.0))
                    this_interpolation = LinearInterp(
                        this_w_list,
                        f(
                            this_w_list,
                            x * np.ones(this_w_list.size),
                            y * np.ones(this_w_list.size),
                            z * np.ones(this_w_list.size),
                        ),
                    )
                    temptemp.append(this_interpolation)
                temp.append(deepcopy(temptemp))
            wInterpolators.append(deepcopy(temp))
        g = TrilinearInterpOnInterp1D(wInterpolators, x_list, y_list, z_list)

        N = 20000
        rand_w = RNG.rand(N) * 5.0
        rand_x = RNG.rand(N) * 5.0
        rand_y = RNG.rand(N) * 5.0
        rand_z = RNG.rand(N) * 5.0
        t_start = time()
        z = (f(rand_w, rand_x, rand_y, rand_z) - g(rand_w, rand_x, rand_y, rand_z)) / f(
            rand_w, rand_x, rand_y, rand_z
        )
        q = (
            dfdw(rand_w, rand_x, rand_y, rand_z)
            - g.derivativeW(rand_w, rand_x, rand_y, rand_z)
        ) / dfdw(rand_w, rand_x, rand_y, rand_z)
        r = (
            dfdx(rand_w, rand_x, rand_y, rand_z)
            - g.derivativeX(rand_w, rand_x, rand_y, rand_z)
        ) / dfdx(rand_w, rand_x, rand_y, rand_z)
        p = (
            dfdy(rand_w, rand_x, rand_y, rand_z)
            - g.derivativeY(rand_w, rand_x, rand_y, rand_z)
        ) / dfdy(rand_w, rand_x, rand_y, rand_z)
        s = (
            dfdz(rand_w, rand_x, rand_y, rand_z)
            - g.derivativeZ(rand_w, rand_x, rand_y, rand_z)
        ) / dfdz(rand_w, rand_x, rand_y, rand_z)
        t_end = time()

        z.sort()
        print(z)
        print(t_end - t_start)

    if False:
        def f(x, y): return 3.0 * x ** 2.0 + x * y + 4.0 * y ** 2.0
        def dfdx(x, y): return 6.0 * x + y
        def dfdy(x, y): return x + 8.0 * y

        x_list = np.linspace(0, 5, 101, dtype=float)
        y_list = np.linspace(0, 5, 101, dtype=float)
        x_temp, y_temp = np.meshgrid(x_list, y_list, indexing="ij")
        g = BilinearInterp(f(x_temp, y_temp), x_list, y_list)

        rand_x = RNG.rand(100) * 5.0
        rand_y = RNG.rand(100) * 5.0
        z = (f(rand_x, rand_y) - g(rand_x, rand_y)) / f(rand_x, rand_y)
        q = (f(x_temp, y_temp) - g(x_temp, y_temp)) / f(x_temp, y_temp)
        # print(z)
        # print(q)

    if False:
        f = (
            lambda x, y, z: 3.0 * x ** 2.0
            + x * y
            + 4.0 * y ** 2.0
            - 5 * z ** 2.0
            + 1.5 * x * z
        )
        def dfdx(x, y, z): return 6.0 * x + y + 1.5 * z
        def dfdy(x, y, z): return x + 8.0 * y
        def dfdz(x, y, z): return -10.0 * z + 1.5 * x

        x_list = np.linspace(0, 5, 11, dtype=float)
        y_list = np.linspace(0, 5, 11, dtype=float)
        z_list = np.linspace(0, 5, 101, dtype=float)
        x_temp, y_temp, z_temp = np.meshgrid(x_list, y_list, z_list, indexing="ij")
        g = TrilinearInterp(f(x_temp, y_temp, z_temp), x_list, y_list, z_list)

        rand_x = RNG.rand(1000) * 5.0
        rand_y = RNG.rand(1000) * 5.0
        rand_z = RNG.rand(1000) * 5.0
        z = (f(rand_x, rand_y, rand_z) - g(rand_x, rand_y, rand_z)) / f(
            rand_x, rand_y, rand_z
        )
        q = (
            dfdx(rand_x, rand_y, rand_z) - g.derivativeX(rand_x, rand_y, rand_z)
        ) / dfdx(rand_x, rand_y, rand_z)
        r = (
            dfdy(rand_x, rand_y, rand_z) - g.derivativeY(rand_x, rand_y, rand_z)
        ) / dfdy(rand_x, rand_y, rand_z)
        p = (
            dfdz(rand_x, rand_y, rand_z) - g.derivativeZ(rand_x, rand_y, rand_z)
        ) / dfdz(rand_x, rand_y, rand_z)
        p.sort()
        plt.plot(p)

    if False:
        f = (
            lambda w, x, y, z: 4.0 * w * z
            - 2.5 * w * x
            + w * y
            + 6.0 * x * y
            - 10.0 * x * z
            + 3.0 * y * z
            - 7.0 * z
            + 4.0 * x
            + 2.0 * y
            - 5.0 * w
        )
        def dfdw(w, x, y, z): return 4.0 * z - 2.5 * x + y - 5.0
        def dfdx(w, x, y, z): return -2.5 * w + 6.0 * y - 10.0 * z + 4.0
        def dfdy(w, x, y, z): return w + 6.0 * x + 3.0 * z + 2.0
        def dfdz(w, x, y, z): return 4.0 * w - 10.0 * x + 3.0 * y - 7

        w_list = np.linspace(0, 5, 16, dtype=float)
        x_list = np.linspace(0, 5, 16, dtype=float)
        y_list = np.linspace(0, 5, 16, dtype=float)
        z_list = np.linspace(0, 5, 16, dtype=float)
        w_temp, x_temp, y_temp, z_temp = np.meshgrid(
            w_list, x_list, y_list, z_list, indexing="ij"
        )
        def mySearch(trash, x): return np.floor(x / 5 * 32).astype(int)
        g = QuadlinearInterp(
            f(w_temp, x_temp, y_temp, z_temp), w_list, x_list, y_list, z_list
        )

        N = 1000000
        rand_w = RNG.rand(N) * 5.0
        rand_x = RNG.rand(N) * 5.0
        rand_y = RNG.rand(N) * 5.0
        rand_z = RNG.rand(N) * 5.0
        t_start = time()
        z = (f(rand_w, rand_x, rand_y, rand_z) - g(rand_w, rand_x, rand_y, rand_z)) / f(
            rand_w, rand_x, rand_y, rand_z
        )
        t_end = time()
        # print(z)
        print(t_end - t_start)

    if False:
        def f(x, y): return 3.0 * x ** 2.0 + x * y + 4.0 * y ** 2.0
        def dfdx(x, y): return 6.0 * x + y
        def dfdy(x, y): return x + 8.0 * y

        warp_factor = 0.01
        x_list = np.linspace(0, 5, 71, dtype=float)
        y_list = np.linspace(0, 5, 51, dtype=float)
        x_temp, y_temp = np.meshgrid(x_list, y_list, indexing="ij")
        x_adj = x_temp + warp_factor * (RNG.rand(x_list.size, y_list.size) - 0.5)
        y_adj = y_temp + warp_factor * (RNG.rand(x_list.size, y_list.size) - 0.5)
        g = Curvilinear2DInterp(f(x_adj, y_adj), x_adj, y_adj)

        rand_x = RNG.rand(1000) * 5.0
        rand_y = RNG.rand(1000) * 5.0
        t_start = time()
        z = (f(rand_x, rand_y) - g(rand_x, rand_y)) / f(rand_x, rand_y)
        q = (dfdx(rand_x, rand_y) - g.derivativeX(rand_x, rand_y)) / dfdx(
            rand_x, rand_y
        )
        r = (dfdy(rand_x, rand_y) - g.derivativeY(rand_x, rand_y)) / dfdy(
            rand_x, rand_y
        )
        t_end = time()
        z.sort()
        q.sort()
        r.sort()
        # print(z)
        print(t_end - t_start)

    if False:
        f = (
            lambda x, y, z: 3.0 * x ** 2.0
            + x * y
            + 4.0 * y ** 2.0
            - 5 * z ** 2.0
            + 1.5 * x * z
        )
        def dfdx(x, y, z): return 6.0 * x + y + 1.5 * z
        def dfdy(x, y, z): return x + 8.0 * y
        def dfdz(x, y, z): return -10.0 * z + 1.5 * x

        warp_factor = 0.01
        x_list = np.linspace(0, 5, 11, dtype=float)
        y_list = np.linspace(0, 5, 11, dtype=float)
        z_list = np.linspace(0, 5, 101, dtype=float)
        x_temp, y_temp = np.meshgrid(x_list, y_list, indexing="ij")
        xyInterpolators = []
        for j in range(z_list.size):
            x_adj = x_temp + warp_factor * (RNG.rand(x_list.size, y_list.size) - 0.5)
            y_adj = y_temp + warp_factor * (RNG.rand(x_list.size, y_list.size) - 0.5)
            z_temp = z_list[j] * np.ones(x_adj.shape)
            thisInterp = Curvilinear2DInterp(f(x_adj, y_adj, z_temp), x_adj, y_adj)
            xyInterpolators.append(thisInterp)
        g = LinearInterpOnInterp2D(xyInterpolators, z_list)

        N = 1000
        rand_x = RNG.rand(N) * 5.0
        rand_y = RNG.rand(N) * 5.0
        rand_z = RNG.rand(N) * 5.0
        z = (f(rand_x, rand_y, rand_z) - g(rand_x, rand_y, rand_z)) / f(
            rand_x, rand_y, rand_z
        )
        p = (
            dfdz(rand_x, rand_y, rand_z) - g.derivativeZ(rand_x, rand_y, rand_z)
        ) / dfdz(rand_x, rand_y, rand_z)
        p.sort()
        plt.plot(p)

    if False:
        f = (
            lambda w, x, y, z: 4.0 * w * z
            - 2.5 * w * x
            + w * y
            + 6.0 * x * y
            - 10.0 * x * z
            + 3.0 * y * z
            - 7.0 * z
            + 4.0 * x
            + 2.0 * y
            - 5.0 * w
        )
        def dfdw(w, x, y, z): return 4.0 * z - 2.5 * x + y - 5.0
        def dfdx(w, x, y, z): return -2.5 * w + 6.0 * y - 10.0 * z + 4.0
        def dfdy(w, x, y, z): return w + 6.0 * x + 3.0 * z + 2.0
        def dfdz(w, x, y, z): return 4.0 * w - 10.0 * x + 3.0 * y - 7

        warp_factor = 0.1
        w_list = np.linspace(0, 5, 16, dtype=float)
        x_list = np.linspace(0, 5, 16, dtype=float)
        y_list = np.linspace(0, 5, 16, dtype=float)
        z_list = np.linspace(0, 5, 16, dtype=float)
        w_temp, x_temp = np.meshgrid(w_list, x_list, indexing="ij")
        wxInterpolators = []
        for i in range(y_list.size):
            temp = []
            for j in range(z_list.size):
                w_adj = w_temp + warp_factor * (
                    RNG.rand(w_list.size, x_list.size) - 0.5
                )
                x_adj = x_temp + warp_factor * (
                    RNG.rand(w_list.size, x_list.size) - 0.5
                )
                y_temp = y_list[i] * np.ones(w_adj.shape)
                z_temp = z_list[j] * np.ones(w_adj.shape)
                thisInterp = Curvilinear2DInterp(
                    f(w_adj, x_adj, y_temp, z_temp), w_adj, x_adj
                )
                temp.append(thisInterp)
            wxInterpolators.append(temp)
        g = BilinearInterpOnInterp2D(wxInterpolators, y_list, z_list)

        N = 1000000
        rand_w = RNG.rand(N) * 5.0
        rand_x = RNG.rand(N) * 5.0
        rand_y = RNG.rand(N) * 5.0
        rand_z = RNG.rand(N) * 5.0

        t_start = time()
        z = (f(rand_w, rand_x, rand_y, rand_z) - g(rand_w, rand_x, rand_y, rand_z)) / f(
            rand_w, rand_x, rand_y, rand_z
        )
        t_end = time()
        z.sort()
        print(z)
        print(t_end - t_start)

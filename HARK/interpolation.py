"""
Custom interpolation methods for representing approximations to functions.
It also includes wrapper classes to enforce standard methods across classes.
Each interpolation class must have a distance() method that compares itself to
another instance; this is used in HARK.core's solve() method to check for solution
convergence.  The interpolator classes currently in this module inherit their
distance method from MetricObject.
"""

import warnings
from copy import deepcopy

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from HARK.metric import MetricObject
from HARK.rewards import CRRAutility, CRRAutilityP, CRRAutilityPP
from numba import njit


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


def _coerce_1d_grid(arr):
    """Return ``arr`` as a 1D numpy array, flattening if necessary."""
    a = np.asarray(arr)
    if a.ndim != 1:
        warnings.warn("input not of the size (n, ), attempting to flatten")
        return a.flatten()
    return a


def _broadcast_eval(inner, *args):
    """Broadcast ``args`` to a common shape, call ``inner`` on the flattened
    arrays, and reshape the result.

    Shared by the ``__call__``/``derivativeX``/``derivativeY``/... methods of
    :class:`HARKinterpolator2D`, :class:`HARKinterpolator3D`, and
    :class:`HARKinterpolator4D`.
    """
    arrs = list(np.broadcast_arrays(*[np.asarray(a) for a in args]))
    return inner(*[a.flatten() for a in arrs]).reshape(arrs[0].shape)


def _locate_clipped(grid, values, n):
    """Return ``np.searchsorted(grid, values)`` clipped into ``[1, n - 1]``.

    Shared by every interpolator that brackets queries with ``a_list[idx - 1]``
    and ``a_list[idx]``: a single clipped index per axis is enough for
    1D/2D/3D/4D evaluation and partial-derivative loops.
    """
    return np.clip(np.searchsorted(grid, values), 1, n - 1)


def _cell_fraction(grid, idx, queries):
    """Linear-cell fractional position of ``queries`` within ``[grid[idx-1], grid[idx]]``.

    Returns ``(queries - grid[idx - 1]) / (grid[idx] - grid[idx - 1])``. Works
    with ``idx`` as a scalar (interp-on-interp loops) or an integer array
    (tensor-grid interpolators).
    """
    lower = grid[idx - 1]
    return (queries - lower) / (grid[idx] - lower)


def _iter_unique_pairs(*positions):
    """Yield ``(*indices, mask)`` for each unique observed combination of axis ``positions``.

    Accepts any number of equal-length 1D position arrays. Cells that no
    query falls into are silently skipped. Yielded indices are Python
    ``int``s, safe for list indexing. No-op when no positions are passed
    or when each axis is empty.
    """
    if not positions or positions[0].size == 0:
        return
    stacked = np.column_stack(positions)
    combos, inverse = np.unique(stacked, axis=0, return_inverse=True)
    for k, combo in enumerate(combos):
        yield (*(int(v) for v in combo), inverse == k)


def _envelope_partial(envelope, args, deriv_attr):
    """Compute an envelope partial derivative.

    Evaluates each member function on the broadcast ``args`` to identify the
    active function per point (via ``envelope.argcompare``), then takes the
    requested derivative (``deriv_attr``) of the active function on its slice.
    Shared by ``LowerEnvelope2D`` and ``LowerEnvelope3D`` partial derivatives.
    """
    primary = args[0]
    temp = np.column_stack([f(*args) for f in envelope.functions])
    active = envelope.argcompare(temp, axis=1)
    out = np.zeros_like(primary)
    for j in np.unique(active):
        c = active == j
        out[c] = getattr(envelope.functions[j], deriv_attr)(*[a[c] for a in args])
    return out


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

    derivativeX = derivative  # alias

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
        Default or fallback derivative method using finite difference approximation.
        Subclasses of HARKinterpolator1D should define their own more specific method.
        """
        eps = 1e-8
        f0 = self.__call__(x)
        f1 = self.__call__(x + eps)
        dydx = (f1 - f0) / eps
        return dydx

    def _evalAndDer(self, x):
        """
        Interpolated function and derivative evaluator, to be defined in subclasses.
        Default implementation separately calls the _evaluate and _der methods, which
        might be inefficient relative to interpolator-specific implementation.
        """
        y = self._evaluate(x)
        dydx = self._der(x)
        return y, dydx

    def _init_cubic_grids(self, x_list, y_list, dydx_list):
        """
        Coerce ``x_list``, ``y_list``, ``dydx_list`` to validated 1D arrays.

        Stores them as ``self.x_list``, ``self.y_list``, ``self.dydx_list``,
        sets ``self.n``, and runs ``_check_grid_dimensions`` against ``x_list``.
        Shared between :class:`CubicInterp` and :class:`CubicHermiteInterp`.
        """
        self.x_list = _coerce_1d_grid(x_list)
        self.y_list = _coerce_1d_grid(y_list)
        self.dydx_list = _coerce_1d_grid(dydx_list)
        _check_grid_dimensions(1, self.y_list, self.x_list)
        _check_grid_dimensions(1, self.dydx_list, self.x_list)
        self.n = self.x_list.size


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
            Real values to be evaluated in the interpolated function. If both
            are arrays, they must be broadcastable to the same shape.
            Scalar inputs will be broadcast to match array inputs.

        Returns
        -------
        fxy : np.array or float
            The interpolated function evaluated at x,y: fxy = f(x,y), with the
            same shape as x and y.
        """
        return _broadcast_eval(self._evaluate, x, y)

    def derivativeX(self, x, y):
        """
        Evaluates the partial derivative of interpolated function with respect
        to x (the first argument) at the given input.

        Parameters
        ----------
        x : np.array or float
            Real values to be evaluated in the interpolated function.
        y : np.array or float
            Real values to be evaluated in the interpolated function. If both
            are arrays, they must be broadcastable to the same shape.
            Scalar inputs will be broadcast to match array inputs.

        Returns
        -------
        dfdx : np.array or float
            The derivative of the interpolated function with respect to x, eval-
            uated at x,y: dfdx = f_x(x,y), with the same shape as x and y.
        """
        return _broadcast_eval(self._derX, x, y)

    def derivativeY(self, x, y):
        """
        Evaluates the partial derivative of interpolated function with respect
        to y (the second argument) at the given input.

        Parameters
        ----------
        x : np.array or float
            Real values to be evaluated in the interpolated function.
        y : np.array or float
            Real values to be evaluated in the interpolated function. If both
            are arrays, they must be broadcastable to the same shape.
            Scalar inputs will be broadcast to match array inputs.

        Returns
        -------
        dfdy : np.array or float
            The derivative of the interpolated function with respect to y, eval-
            uated at x,y: dfdx = f_y(x,y), with the same shape as x and y.
        """
        return _broadcast_eval(self._derY, x, y)

    def _evaluate(self, x, y):
        """
        Interpolated function evaluator, to be defined in subclasses.
        """
        raise NotImplementedError()

    def _derX(self, x, y):
        """
        Default or fallback derivative with respect to x, using finite difference approximation.
        Subclasses of HARKinterpolator2D should define their own more specific method.
        """
        eps = 1e-8
        f0 = self.__call__(x, y)
        f1 = self.__call__(x + eps, y)
        dfdx = (f1 - f0) / eps
        return dfdx

    def _derY(self, x, y):
        """
        Default or fallback derivative with respect to y, using finite difference approximation.
        Subclasses of HARKinterpolator2D should define their own more specific method.
        """
        eps = 1e-8
        f0 = self.__call__(x, y)
        f1 = self.__call__(x, y + eps)
        dfdy = (f1 - f0) / eps
        return dfdy


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
            Real values to be evaluated in the interpolated function. If multiple
            inputs are arrays, they must be broadcastable to the same shape.
            Scalar inputs will be broadcast to match array inputs.
        z : np.array or float
            Real values to be evaluated in the interpolated function. If multiple
            inputs are arrays, they must be broadcastable to the same shape.
            Scalar inputs will be broadcast to match array inputs.

        Returns
        -------
        fxyz : np.array or float
            The interpolated function evaluated at x,y,z: fxyz = f(x,y,z), with
            the same shape as x, y, and z.
        """
        return _broadcast_eval(self._evaluate, x, y, z)

    def derivativeX(self, x, y, z):
        """
        Evaluates the partial derivative of the interpolated function with respect
        to x (the first argument) at the given input.

        Parameters
        ----------
        x : np.array or float
            Real values to be evaluated in the interpolated function.
        y : np.array or float
            Real values to be evaluated in the interpolated function. If multiple
            inputs are arrays, they must be broadcastable to the same shape.
            Scalar inputs will be broadcast to match array inputs.
        z : np.array or float
            Real values to be evaluated in the interpolated function. If multiple
            inputs are arrays, they must be broadcastable to the same shape.
            Scalar inputs will be broadcast to match array inputs.

        Returns
        -------
        dfdx : np.array or float
            The derivative with respect to x of the interpolated function evaluated
            at x,y,z: dfdx = f_x(x,y,z), with the same shape as x, y, and z.
        """
        return _broadcast_eval(self._derX, x, y, z)

    def derivativeY(self, x, y, z):
        """
        Evaluates the partial derivative of the interpolated function with respect
        to y (the second argument) at the given input.

        Parameters
        ----------
        x : np.array or float
            Real values to be evaluated in the interpolated function.
        y : np.array or float
            Real values to be evaluated in the interpolated function. If multiple
            inputs are arrays, they must be broadcastable to the same shape.
            Scalar inputs will be broadcast to match array inputs.
        z : np.array or float
            Real values to be evaluated in the interpolated function. If multiple
            inputs are arrays, they must be broadcastable to the same shape.
            Scalar inputs will be broadcast to match array inputs.

        Returns
        -------
        dfdy : np.array or float
            The derivative with respect to y of the interpolated function evaluated
            at x,y,z: dfdy = f_y(x,y,z), with the same shape as x, y, and z.
        """
        return _broadcast_eval(self._derY, x, y, z)

    def derivativeZ(self, x, y, z):
        """
        Evaluates the partial derivative of the interpolated function with respect
        to z (the third argument) at the given input.

        Parameters
        ----------
        x : np.array or float
            Real values to be evaluated in the interpolated function.
        y : np.array or float
            Real values to be evaluated in the interpolated function. If multiple
            inputs are arrays, they must be broadcastable to the same shape.
            Scalar inputs will be broadcast to match array inputs.
        z : np.array or float
            Real values to be evaluated in the interpolated function. If multiple
            inputs are arrays, they must be broadcastable to the same shape.
            Scalar inputs will be broadcast to match array inputs.

        Returns
        -------
        dfdz : np.array or float
            The derivative with respect to z of the interpolated function evaluated
            at x,y,z: dfdz = f_z(x,y,z), with the same shape as x, y, and z.
        """
        return _broadcast_eval(self._derZ, x, y, z)

    def _evaluate(self, x, y, z):
        """
        Interpolated function evaluator, to be defined in subclasses.
        """
        raise NotImplementedError()

    def _derX(self, x, y, z):
        """
        Default or fallback derivative with respect to x, using finite difference approximation.
        Subclasses of HARKinterpolator3D should define their own more specific method.
        """
        eps = 1e-8
        f0 = self.__call__(x, y, z)
        f1 = self.__call__(x + eps, y, z)
        dfdx = (f1 - f0) / eps
        return dfdx

    def _derY(self, x, y, z):
        """
        Default or fallback derivative with respect to y, using finite difference approximation.
        Subclasses of HARKinterpolator3D should define their own more specific method.
        """
        eps = 1e-8
        f0 = self.__call__(x, y, z)
        f1 = self.__call__(x, y + eps, z)
        dfdy = (f1 - f0) / eps
        return dfdy

    def _derZ(self, x, y, z):
        """
        Default or fallback derivative with respect to z, using finite difference approximation.
        Subclasses of HARKinterpolator3D should define their own more specific method.
        """
        eps = 1e-8
        f0 = self.__call__(x, y, z)
        f1 = self.__call__(x, y, z + eps)
        dfdz = (f1 - f0) / eps
        return dfdz


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
            Real values to be evaluated in the interpolated function. If multiple
            inputs are arrays, they must be broadcastable to the same shape.
            Scalar inputs will be broadcast to match array inputs.
        y : np.array or float
            Real values to be evaluated in the interpolated function. If multiple
            inputs are arrays, they must be broadcastable to the same shape.
            Scalar inputs will be broadcast to match array inputs.
        z : np.array or float
            Real values to be evaluated in the interpolated function. If multiple
            inputs are arrays, they must be broadcastable to the same shape.
            Scalar inputs will be broadcast to match array inputs.

        Returns
        -------
        fwxyz : np.array or float
            The interpolated function evaluated at w,x,y,z: fwxyz = f(w,x,y,z),
            with the same shape as w, x, y, and z.
        """
        return _broadcast_eval(self._evaluate, w, x, y, z)

    def derivativeW(self, w, x, y, z):
        """
        Evaluates the partial derivative with respect to w (the first argument)
        of the interpolated function at the given input.

        Parameters
        ----------
        w : np.array or float
            Real values to be evaluated in the interpolated function.
        x : np.array or float
            Real values to be evaluated in the interpolated function. If multiple
            inputs are arrays, they must be broadcastable to the same shape.
            Scalar inputs will be broadcast to match array inputs.
        y : np.array or float
            Real values to be evaluated in the interpolated function. If multiple
            inputs are arrays, they must be broadcastable to the same shape.
            Scalar inputs will be broadcast to match array inputs.
        z : np.array or float
            Real values to be evaluated in the interpolated function. If multiple
            inputs are arrays, they must be broadcastable to the same shape.
            Scalar inputs will be broadcast to match array inputs.

        Returns
        -------
        dfdw : np.array or float
            The derivative with respect to w of the interpolated function eval-
            uated at w,x,y,z: dfdw = f_w(w,x,y,z), with the same shape as inputs.
        """
        return _broadcast_eval(self._derW, w, x, y, z)

    def derivativeX(self, w, x, y, z):
        """
        Evaluates the partial derivative with respect to x (the second argument)
        of the interpolated function at the given input.

        Parameters
        ----------
        w : np.array or float
            Real values to be evaluated in the interpolated function.
        x : np.array or float
            Real values to be evaluated in the interpolated function. If multiple
            inputs are arrays, they must be broadcastable to the same shape.
            Scalar inputs will be broadcast to match array inputs.
        y : np.array or float
            Real values to be evaluated in the interpolated function. If multiple
            inputs are arrays, they must be broadcastable to the same shape.
            Scalar inputs will be broadcast to match array inputs.
        z : np.array or float
            Real values to be evaluated in the interpolated function. If multiple
            inputs are arrays, they must be broadcastable to the same shape.
            Scalar inputs will be broadcast to match array inputs.

        Returns
        -------
        dfdx : np.array or float
            The derivative with respect to x of the interpolated function eval-
            uated at w,x,y,z: dfdx = f_x(w,x,y,z), with the same shape as inputs.
        """
        return _broadcast_eval(self._derX, w, x, y, z)

    def derivativeY(self, w, x, y, z):
        """
        Evaluates the partial derivative with respect to y (the third argument)
        of the interpolated function at the given input.

        Parameters
        ----------
        w : np.array or float
            Real values to be evaluated in the interpolated function.
        x : np.array or float
            Real values to be evaluated in the interpolated function. If multiple
            inputs are arrays, they must be broadcastable to the same shape.
            Scalar inputs will be broadcast to match array inputs.
        y : np.array or float
            Real values to be evaluated in the interpolated function. If multiple
            inputs are arrays, they must be broadcastable to the same shape.
            Scalar inputs will be broadcast to match array inputs.
        z : np.array or float
            Real values to be evaluated in the interpolated function. If multiple
            inputs are arrays, they must be broadcastable to the same shape.
            Scalar inputs will be broadcast to match array inputs.

        Returns
        -------
        dfdy : np.array or float
            The derivative with respect to y of the interpolated function eval-
            uated at w,x,y,z: dfdy = f_y(w,x,y,z), with the same shape as inputs.
        """
        return _broadcast_eval(self._derY, w, x, y, z)

    def derivativeZ(self, w, x, y, z):
        """
        Evaluates the partial derivative with respect to z (the fourth argument)
        of the interpolated function at the given input.

        Parameters
        ----------
        w : np.array or float
            Real values to be evaluated in the interpolated function.
        x : np.array or float
            Real values to be evaluated in the interpolated function. If multiple
            inputs are arrays, they must be broadcastable to the same shape.
            Scalar inputs will be broadcast to match array inputs.
        y : np.array or float
            Real values to be evaluated in the interpolated function. If multiple
            inputs are arrays, they must be broadcastable to the same shape.
            Scalar inputs will be broadcast to match array inputs.
        z : np.array or float
            Real values to be evaluated in the interpolated function. If multiple
            inputs are arrays, they must be broadcastable to the same shape.
            Scalar inputs will be broadcast to match array inputs.

        Returns
        -------
        dfdz : np.array or float
            The derivative with respect to z of the interpolated function eval-
            uated at w,x,y,z: dfdz = f_z(w,x,y,z), with the same shape as inputs.
        """
        return _broadcast_eval(self._derZ, w, x, y, z)

    def _evaluate(self, w, x, y, z):
        """
        Interpolated function evaluator, to be defined in subclasses.
        """
        raise NotImplementedError()

    def _derW(self, w, x, y, z):
        """
        Default or fallback derivative with respect to w, using finite difference approximation.
        Subclasses of HARKinterpolator4D should define their own more specific method.
        """
        eps = 1e-8
        f0 = self.__call__(w, x, y, z)
        f1 = self.__call__(w + eps, x, y, z)
        dfdw = (f1 - f0) / eps
        return dfdw

    def _derX(self, w, x, y, z):
        """
        Default or fallback derivative with respect to x, using finite difference approximation.
        Subclasses of HARKinterpolator4D should define their own more specific method.
        """
        eps = 1e-8
        f0 = self.__call__(w, x, y, z)
        f1 = self.__call__(w, x + eps, y, z)
        dfdx = (f1 - f0) / eps
        return dfdx

    def _derY(self, w, x, y, z):
        """
        Default or fallback derivative with respect to y, using finite difference approximation.
        Subclasses of HARKinterpolator4D should define their own more specific method.
        """
        eps = 1e-8
        f0 = self.__call__(w, x, y, z)
        f1 = self.__call__(w, x, y + eps, z)
        dfdy = (f1 - f0) / eps
        return dfdy

    def _derZ(self, w, x, y, z):
        """
        Default or fallback derivative with respect to z, using finite difference approximation.
        Subclasses of HARKinterpolator4D should define their own more specific method.
        """
        eps = 1e-8
        f0 = self.__call__(w, x, y, z)
        f1 = self.__call__(w, x, y, z + eps)
        dfdz = (f1 - f0) / eps
        return dfdz


class IdentityFunction(MetricObject):
    """
    A fairly trivial interpolator that simply returns one of its arguments.  Useful for avoiding
    numeric error in extreme cases.

    Parameters
    ----------
    i_dim : int
        Index of the dimension on which the identity is defined.  ``f(*x) = x[i]``
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
            return np.ones_like(args[0])
        else:
            return np.zeros_like(args[0])

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
            return np.ones_like(args[0])
        else:
            return np.zeros_like(args[0])

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
            return np.ones_like(args[0])
        else:
            return np.zeros_like(args[0])

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
            return np.ones_like(args[0])
        else:
            return np.zeros_like(args[0])

    def derivativeW(self, *args):
        """
        Returns the derivative of the function with respect to the W dimension.
        This should only exist when n_dims >= 4.
        """
        if self.n_dims < 4:
            raise RuntimeError(
                "Derivative with respect to W can't be called when n_dims < 4!"
            )
        j = 0
        if self.i_dim == j:
            return np.ones_like(args[0])
        else:
            return np.zeros_like(args[0])


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

    distance_criteria = ["value"]

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

    def eval_with_derivative(self, x):
        val = self(x)
        der = self._der(x)
        return val, der

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
    lower_extrap : bool
        Indicator for whether lower extrapolation is allowed.  False means
        f(x) = NaN for x < min(x_list); True means linear extrapolation.
    pre_compute : bool
        Indicator for whether interpolation coefficients should be pre-computed
        and stored as attributes of self (default False). More memory will be used,
        and instantiation will take slightly longer, but later evaluation will
        be faster due to less arithmetic.
    indexer : function or None (default)
        If provided, a custom function that identifies the index of the interpolant
        segment for each query point. Should return results identically to the
        default behavior of np.maximum(np.searchsorted(self.x_list[:-1], x), 1).
        WARNING: User is responsible for verifying that their custom indexer is
        actually correct versus default behavior.
    decay_extrap_form : str
        Functional form of the decay toward the limiting linear function when
        ``intercept_limit`` and ``slope_limit`` are provided (irrelevant
        otherwise). ``'exp'`` (default, the long-standing behavior): the gap
        below the limiting line decays exponentially in ``x - x_list[-1]``.
        ``'powerlaw'``: the gap decays as a power law,
        ``gap(x) = A*((x + h)/(x_list[-1] + h))**(-Q)`` with pivot
        ``h = intercept_limit/slope_limit`` and ``Q = B*(x_list[-1] + h)``.
        Both forms match the level and the slope of the interpolant at the top
        gridpoint, so neither needs parameters beyond the limiting line; over a
        short span above the grid they coincide (the exponential is the
        local linearization of the power law), but the power law is the
        asymptotically correct tail for buffer-stock consumption functions,
        whose gap below the perfect-foresight asymptote ``MPCmin*(x + hNrm)``
        decays polynomially, not exponentially (``h`` is then human wealth).
        ``'powerlaw'`` requires ``slope_limit > 0``, a top knot strictly below
        the limiting line with slope strictly above ``slope_limit``, and
        ``x_list[-1] + h > 0``; if violated it warns and disables decay
        extrapolation (``decay_extrap == False``) rather than risk a divergent
        tail.

        # THEOREM-REF[BufferStockTheory-Latest @ c181870f :: theory/powerlaw-decay/final_proof.md :: §7. The computational payoff: why the compactified core is the right presentation :: The extrapolation form of record :: https://llorracc.github.io/BufferStockTheory-Latest/powerlaw-decay-theory/]
        #   The power-law gap tail C*(x + h)**(-q) is the theorem's extrapolation
        #   form of record for buffer-stock consumption functions; the
        #   exponential form is not merely inaccurate but impossible as an
        #   asymptotic form (Prop A0) — 'exp' stays only as the legacy default.
    decay_extrap_Q : float or None (default), keyword-only
        ``None``: byte-for-byte the behavior described above for both forms
        (the power-law exponent is FITTED from the top two knots as
        ``Q = B*(x_list[-1] + h)``). A positive float (requires
        ``decay_extrap_form='powerlaw'``, else ``ValueError``): use this
        EXPLICIT decay exponent instead of the fitted one. The gap amplitude
        ``A`` stays the level gap at the top knot, so the tail is
        level-matched (continuous) by construction, but its slope no longer
        matches the interpolant's top-segment slope: the derivative just above
        the top knot is ``slope_limit + Q*A/pivot``, i.e. a C1 kink of size
        ``(Q_fit - Q)*A/pivot`` relative to the fitted tangent (tiny in
        absolute terms when ``A`` is small and the pivot large, but
        sign-indefinite). Because the exponent no longer needs to be inferred
        from the top-segment slope, the slope-tangency part of the powerlaw
        validity guard is relaxed: only ``slope_limit > 0``, a top knot
        strictly below the limiting line, and a positive pivot are required —
        in particular a top slope at or below ``slope_limit`` (fitted
        ``B <= 0``, where the fitted form must disable decay) still attaches
        an explicit-Q tail. ``self.decay_extrap_Q_source`` records
        ``'explicit'`` vs ``'fitted'`` for introspection. The C1-kink
        description above applies to the ONE-TERM variant
        (``decay_extrap_terms=1``); the default two-term attachment is C1.

        # THEOREM-REF[BufferStockTheory-Latest @ c181870f :: theory/powerlaw-decay/final_proof.md :: §7. The computational payoff: why the compactified core is the right presentation :: The extrapolation form of record :: https://llorracc.github.io/BufferStockTheory-Latest/powerlaw-decay-theory/]
        #   The gap extrapolant g ~ C*(x + h)**(-q) with q = min(1, q*) is the
        #   asymptotically correct form for buffer-stock consumption functions;
        #   this keyword is the hook that lets callers pin the exponent to the
        #   theory value (or any explicit value) instead of the 2-knot fit.
    decay_extrap_terms : int, keyword-only (default 2)
        Consulted only with ``decay_extrap_Q``. ``2`` (default): the C1
        TWO-TERM attachment ``gap = A*z**(-Q) + A2*z**(-(Q+1))`` with
        ``z = (x+h)/(x_top+h)``, level- AND slope-matched at the top knot
        (``A2 = gap*(Q_fit - Q)``, ``A = gap - A2``, where ``Q_fit`` is the
        2-knot fitted exponent); it carries the theory exponent as the
        leading term, collapses EXACTLY to the one-term tail when
        ``Q_fit == Q``, and preserves below-the-line, ``c' > slope_limit``,
        and the leading exponent whenever it attaches. Guard: ``Q_fit >=
        Q + 1`` (top segment locally steeper than theory+1 — a coarse or
        non-converged grid top) warns and falls back to one term. ``1``: the
        level-matched one-term tail with the documented C1 kink.

        WHY the two-term default: it guards against Jacobian problems in
        SSJ-type (sequence-space Jacobian) approaches. Policy derivatives
        are primitive inputs to SSJ fake-news/Jacobian construction and to
        automatic or numerical differentiation through the solution; a C1
        kink at the attachment point makes those derivatives discontinuous
        for queries crossing it, producing noisy or discontinuous Jacobian
        rows for high-wealth states. The second exponent is ``Q + 1`` (NOT
        the theory-subleading pair, whose spacing ``|q* - 1|`` vanishes at
        the near-resonance calibrations and blows the amplitudes up): it is
        an attachment (boundary-layer) term absorbing exactly the one-term
        kink, not an asymptotic claim.

        # THEOREM-REF[BufferStockTheory-Latest @ 3f4b021e :: theory/powerlaw-decay/grid_design_final_spec.md :: THE SPEC (owner-proposed scheme, sharpened by F1–F8) :: F11 — The C1 two-term attachment :: https://llorracc.github.io/BufferStockTheory-Latest/powerlaw-decay-theory/grid-design-final-spec/]
        #   F11: closed-form amplitudes B = G*(Q_fit - Q), A = G*(1+Q-Q_fit)
        #   from the level+slope matching conditions; conditioning argument
        #   for Q+1 over the theory-subleading pair; property proofs
        #   (below-line, MPC floor, concavity condition, guard + fallback).
    """

    distance_criteria = ["x_list", "y_list"]

    def __init__(
        self,
        x_list,
        y_list,
        intercept_limit=None,
        slope_limit=None,
        lower_extrap=False,
        pre_compute=False,
        indexer=None,
        decay_extrap_form="exp",
        *,
        decay_extrap_Q=None,
        decay_extrap_terms=2,
    ):
        # Make the basic linear spline interpolation
        self.x_list = _coerce_1d_grid(x_list)
        self.y_list = _coerce_1d_grid(y_list)
        _check_grid_dimensions(1, self.y_list, self.x_list)
        self.lower_extrap = lower_extrap
        self.x_n = self.x_list.size
        self.indexer = indexer

        # Make a decay extrapolation
        if decay_extrap_form not in ("exp", "powerlaw"):
            raise ValueError(
                "decay_extrap_form must be 'exp' or 'powerlaw', got "
                + repr(decay_extrap_form)
            )
        self.decay_extrap_form = decay_extrap_form
        if decay_extrap_Q is not None:
            if decay_extrap_form != "powerlaw":
                raise ValueError(
                    "decay_extrap_Q requires decay_extrap_form='powerlaw'"
                )
            if intercept_limit is None or slope_limit is None:
                raise ValueError(
                    "decay_extrap_Q requires intercept_limit and slope_limit"
                )
            decay_extrap_Q = float(decay_extrap_Q)
            if not np.isfinite(decay_extrap_Q) or decay_extrap_Q <= 0.0:
                raise ValueError(
                    "decay_extrap_Q must be a positive finite float, got "
                    + repr(decay_extrap_Q)
                )
        if isinstance(decay_extrap_terms, bool) or decay_extrap_terms not in (1, 2):
            raise ValueError(
                "decay_extrap_terms must be 1 or 2, got "
                + repr(decay_extrap_terms)
            )
        if intercept_limit is not None and slope_limit is not None:
            slope_at_top = (y_list[-1] - y_list[-2]) / (x_list[-1] - x_list[-2])
            level_diff = intercept_limit + slope_limit * x_list[-1] - y_list[-1]
            slope_diff = slope_limit - slope_at_top
            if decay_extrap_Q is not None:
                # Explicit-exponent power law: level-matched at the top knot,
                # exponent supplied by the caller (relaxed guard; see docstring)
                self.intercept_limit = intercept_limit
                self.slope_limit = slope_limit
                self._init_explicit_Q_decay(
                    level_diff, slope_diff, decay_extrap_Q, decay_extrap_terms
                )
            # If the model that can handle uncertainty has been calibrated with
            # with uncertainty set to zero, the 'extrapolation' will blow up
            # Guard against that and nearby problems by testing slope equality
            elif not np.isclose(slope_limit, slope_at_top, atol=1e-15):
                self.decay_extrap_A = level_diff
                self.decay_extrap_B = -slope_diff / level_diff
                self.intercept_limit = intercept_limit
                self.slope_limit = slope_limit
                self.decay_extrap = True
                if decay_extrap_form == "powerlaw":
                    self._init_powerlaw_decay(level_diff, slope_diff)
            else:
                self.decay_extrap = False
        else:
            self.decay_extrap = False

        # Calculate interpolation coefficients now rather than at evaluation time
        if pre_compute:
            self.slopes = (self.y_list[1:] - self.y_list[:-1]) / (
                self.x_list[1:] - self.x_list[:-1]
            )
            self.intercepts = self.y_list[:-1] - self.slopes * self.x_list[:-1]

    def _init_powerlaw_decay(self, level_diff, slope_diff):
        """Set up the power-law decay tail, or fall back to no decay (with a
        warning) if the required configuration does not hold.

        The gap below the limiting line is
        ``gap(x) = A * ((x + h)/(x_top + h))**(-Q)`` with pivot
        ``h = intercept_limit/slope_limit`` and ``Q = B*(x_top + h)``, which
        matches the level AND the slope of the interpolant at the top
        gridpoint -- the same two conditions the exponential form matches, so
        no parameters beyond (``intercept_limit``, ``slope_limit``) are
        needed. A valid power-law tail requires the top knot strictly below
        the limiting line (``level_diff > 0``) and approaching it (top slope
        strictly above ``slope_limit``, i.e. ``B > 0``), plus
        ``slope_limit > 0`` and a positive pivot ``x_top + h``. For a
        converged consumption function these hold by Carroll-Kimball (1996)
        concavity, so a violation signals bad inputs; decay is then disabled
        outright rather than risk a divergent tail.

        # THEOREM-REF[BufferStockTheory-Latest @ c181870f :: theory/powerlaw-decay/final_proof.md :: §2. Model, conditions, and the imported foundations :: Carroll–Kimball 1996 :: https://llorracc.github.io/BufferStockTheory-Latest/powerlaw-decay-theory/]
        #   Imported foundations L0–L2′: a converged buffer-stock consumption
        #   function is strictly increasing and strictly concave (Carroll–Kimball
        #   1996) and approaches its PF asymptote from below with slope falling
        #   to the limiting MPC — hence level_diff > 0 and B > 0 at a valid knot.
        """
        x_top = self.x_list[-1]
        ok = self.slope_limit > 0.0 and level_diff > 0.0 and self.decay_extrap_B > 0.0
        if ok:
            pivot = x_top + self.intercept_limit / self.slope_limit
            ok = pivot > 0.0
        if not ok:
            warnings.warn(
                "LinearInterp(decay_extrap_form='powerlaw'): the top knot is "
                "not strictly below the limiting line with slope strictly "
                f"above slope_limit (level_diff={level_diff:.6g}, "
                f"slope_diff={slope_diff:.6g}, slope_limit={self.slope_limit:.6g}); "
                "disabling decay extrapolation for this interpolant."
            )
            self.decay_extrap = False
            return
        self.decay_extrap_pivot = pivot
        self.decay_extrap_Q = self.decay_extrap_B * pivot
        self.decay_extrap_Q_source = "fitted"

    def _init_explicit_Q_decay(self, level_diff, slope_diff, Q, terms=2):
        """Set up the power-law decay tail with an EXPLICIT exponent ``Q``, or
        fall back to no decay (with a warning) if the relaxed guard fails.

        With ``terms=2`` (the default; see ``decay_extrap_terms``) the gap is
        the C1 two-term attachment
        ``gap(x) = A*z**(-Q) + A2*z**(-(Q+1))``, ``z = (x+h)/(x_top+h)``,
        level- AND slope-matched at the top knot:
        ``A2 = level_diff*(Q_fit - Q)``, ``A = level_diff - A2`` (F11 closed
        forms; collapses to one term exactly when ``Q_fit == Q``); when the
        fitted rate is theory-infeasibly steep (``Q_fit >= Q + 1``, where the
        leading amplitude would turn negative) it warns and falls back to one
        term. With ``terms=1`` the gap is the one-term
        ``A * ((x + h)/(x_top + h))**(-Q)`` with ``A = level_diff``
        (level-matched only; C1 kink ``(Q_fit - Q)*A/pivot`` at the knot).
        Because ``Q`` is not inferred from the top-segment
        slope, only ``slope_limit > 0``, ``level_diff > 0`` (top knot strictly
        below the limiting line), and a positive pivot are required — NOT
        ``decay_extrap_B > 0``: a top slope at or below ``slope_limit`` (where
        the fitted form must disable decay) is exactly the rescue case an
        explicit exponent exists to serve (the two-term rescue extends the
        body smoothly; the one-term rescue kinks upward at the knot).
        """
        x_top = self.x_list[-1]
        ok = self.slope_limit > 0.0 and level_diff > 0.0
        pivot = None
        if ok:
            pivot = x_top + self.intercept_limit / self.slope_limit
            ok = pivot > 0.0
        if not ok:
            warnings.warn(
                "LinearInterp(decay_extrap_Q=...): explicit-exponent decay "
                "requires slope_limit > 0, a top knot strictly below the "
                f"limiting line, and a positive pivot (level_diff="
                f"{level_diff:.6g}, slope_limit={self.slope_limit:.6g}); "
                "disabling decay extrapolation for this interpolant."
            )
            self.decay_extrap = False
            return
        # fitted-rate diagnostic; also the slope input of the two-term form
        self.decay_extrap_B = -slope_diff / level_diff
        self.decay_extrap_pivot = pivot
        self.decay_extrap_Q = Q
        self.decay_extrap_Q_source = "explicit"
        Q_fit = self.decay_extrap_B * pivot
        A2 = level_diff * (Q_fit - Q) if np.isfinite(Q_fit) else np.nan
        if terms == 2 and np.isfinite(Q_fit) and Q_fit < Q + 1.0 and A2 != 0.0:
            # THEOREM-REF[BufferStockTheory-Latest @ 3f4b021e :: theory/powerlaw-decay/grid_design_final_spec.md :: THE SPEC (owner-proposed scheme, sharpened by F1–F8) :: F11 — The C1 two-term attachment :: https://llorracc.github.io/BufferStockTheory-Latest/powerlaw-decay-theory/grid-design-final-spec/]
            #   Level+slope matching with the theory exponent leading gives
            #   A2 = G*(Q_fit - Q), A = G - A2; the second exponent Q+1 keeps
            #   the system conditioned at near-resonance calibrations where
            #   the theory-subleading pair collides. An EXACT collapse
            #   (Q == Q_fit, A2 == 0) stores the one-term representation
            #   below instead, so it is byte-identical to terms=1 including
            #   derivatives; a non-finite Q_fit (degenerate top segment,
            #   infinite pivot) falls back rather than attach a NaN tail.
            self.decay_extrap_A2 = A2
            self.decay_extrap_A = level_diff - self.decay_extrap_A2
            self.decay_extrap_terms = 2
        else:
            if terms == 2 and not (np.isfinite(Q_fit) and Q_fit < Q + 1.0):
                warnings.warn(
                    "LinearInterp(decay_extrap_Q=..., decay_extrap_terms=2): "
                    f"the fitted rate Q_fit={Q_fit:.4g} is not finite and "
                    f"strictly below Q+1={Q + 1.0:.4g} (at or above it, the "
                    "two-term leading amplitude would be non-positive; "
                    "non-finite signals a degenerate top segment); falling "
                    "back to the one-term level-matched tail (C1 kink at "
                    "the knot). A steep fitted rate at a human-wealth-"
                    "dominated grid top is usually coordinate "
                    "amplification rather than anomalous decay (the two are "
                    "separated by the diagnosis where available): remedies "
                    "are (a) extend the grid top "
                    f"toward Q*hNrm ~= {Q * (pivot - x_top):.4g} (the "
                    "human-wealth-scale rule), or (b) wrap the body in "
                    "DecayTailInterp(decay_extrap_form='moderation_tail') "
                    "(C1 at any cut, no guard)."
                )
            self.decay_extrap_A = level_diff
            self.decay_extrap_A2 = 0.0
            self.decay_extrap_terms = 1
        self.decay_extrap = True

    def _segment_index(self, x):
        """Return the bracketing right-endpoint index for each query in ``x``."""
        if self.indexer is None:
            return np.maximum(np.searchsorted(self.x_list[:-1], x), 1)
        return self.indexer(x)

    def _segment_values(self, x, i, want_y, want_dydx):
        """Compute ``(y, dydx)`` on the linear segment to the right of ``i - 1``.

        Skipped outputs return ``None``. Returned arrays are fresh allocations
        safe for in-place patching by ``_apply_lower_bound`` /
        ``_apply_upper_decay``: in the pre-computed branch ``self.slopes[j]``
        is itself a fancy-index copy, so mutating ``dydx`` does not touch
        ``self.slopes``.
        """
        if hasattr(self, "slopes"):
            j = i - 1
            slopes_j = self.slopes[j]
            y = self.intercepts[j] + slopes_j * x if want_y else None
            dydx = slopes_j if want_dydx else None
            return y, dydx
        x_lo = self.x_list[i - 1]
        x_hi = self.x_list[i]
        y_lo = self.y_list[i - 1]
        y_hi = self.y_list[i]
        if want_y:
            alpha = (x - x_lo) / (x_hi - x_lo)
            y = (1.0 - alpha) * y_lo + alpha * y_hi
        else:
            y = None
        dydx = (y_hi - y_lo) / (x_hi - x_lo) if want_dydx else None
        return y, dydx

    def _apply_lower_bound(self, x, y, dydx):
        """In-place: mark queries below ``x_list[0]`` as NaN. ``y`` and ``dydx``
        may each be ``None`` to skip; no-op when ``self.lower_extrap`` is True."""
        if self.lower_extrap or (y is None and dydx is None):
            return
        below = x < self.x_list[0]
        if y is not None:
            y[below] = np.nan
        if dydx is not None:
            dydx[below] = np.nan

    def _apply_upper_decay(self, x, y, dydx):
        """In-place: replace queries above ``x_list[-1]`` with the limiting linear
        function minus a decaying gap (exponential or power-law, per
        ``decay_extrap_form``). ``y`` and ``dydx`` may each be ``None`` to
        skip; no-op when ``self.decay_extrap`` is False."""
        if not self.decay_extrap or (y is None and dydx is None):
            return
        above = x > self.x_list[-1]
        if not np.any(above):
            return
        x_temp = x[above] - self.x_list[-1]
        if getattr(self, "decay_extrap_form", "exp") == "powerlaw":
            if getattr(self, "decay_extrap_terms", 1) == 2:
                # C1 two-term attachment (F11): gap = A*z**(-Q) + A2*z**(-(Q+1))
                # in z = (x+h)/(x_top+h); level- and slope-matched at the knot
                # with the theory exponent leading. Same stable exp/log1p
                # evaluation; both terms underflow to the line at depth.
                lw = np.log1p(x_temp / self.decay_extrap_pivot)
                w1 = np.exp(-self.decay_extrap_Q * lw)
                w2 = np.exp(-(self.decay_extrap_Q + 1.0) * lw)
                decay = self.decay_extrap_A * w1 + self.decay_extrap_A2 * w2
                if y is not None:
                    y[above] = (
                        self.intercept_limit + self.slope_limit * x[above] - decay
                    )
                if dydx is not None:
                    # d(-gap)/dx = +(Q*A*z**(-Q) + (Q+1)*A2*z**(-(Q+1)))/(x+h)
                    dydx[above] = self.slope_limit + (
                        self.decay_extrap_Q * self.decay_extrap_A * w1
                        + (self.decay_extrap_Q + 1.0) * self.decay_extrap_A2 * w2
                    ) / (x_temp + self.decay_extrap_pivot)
                return
            # gap = A * ((x + h)/(x_top + h))**(-Q), computed via exp/log1p for
            # numerical stability. For x_temp << x_top + h it reduces to
            # A*exp(-B*x_temp): the exponential form is the local linearization
            # of this one, which is why fits over a short span above the grid
            # cannot tell them apart while the tails differ materially.
            decay = self.decay_extrap_A * np.exp(
                -self.decay_extrap_Q * np.log1p(x_temp / self.decay_extrap_pivot)
            )
            if y is not None:
                y[above] = self.intercept_limit + self.slope_limit * x[above] - decay
            if dydx is not None:
                # d(-gap)/dx = +(Q/(x + h))*gap, with x + h = x_temp + pivot
                dydx[above] = (
                    self.slope_limit
                    + self.decay_extrap_Q / (x_temp + self.decay_extrap_pivot) * decay
                )
            return
        decay = self.decay_extrap_A * np.exp(-self.decay_extrap_B * x_temp)
        if y is not None:
            y[above] = self.intercept_limit + self.slope_limit * x[above] - decay
        if dydx is not None:
            dydx[above] = self.slope_limit + self.decay_extrap_B * decay

    def _evalOrDer(self, x, _eval, _Der):
        """
        Returns the level and/or first derivative of the function at each value in
        x.  Only called internally by HARKinterpolator1D.eval_and_der (etc).

        Parameters
        ----------
        x : scalar or np.array
            Set of points where we want to evaluate the interpolated function and/or its derivative.
        _eval : boolean
            Indicator for whether to evaluate the level of the interpolated function.
        _Der : boolean
            Indicator for whether to evaluate the derivative of the interpolated function.

        Returns
        -------
        A list including the level and/or derivative of the interpolated function where requested.
        """
        i = self._segment_index(x)
        y, dydx = self._segment_values(x, i, want_y=_eval, want_dydx=_Der)
        self._apply_lower_bound(x, y, dydx)
        self._apply_upper_decay(x, y, dydx)
        output = []
        if _eval:
            output.append(y)
        if _Der:
            output.append(dydx)
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


class DecayTailInterp(HARKinterpolator1D):
    """
    A composable decay-tail wrapper over ANY 1D interpolant. At and below a
    handoff point ``x_cut`` every query is delegated, unchanged, to the wrapped
    interpolant (its own lower extrapolation and NaN semantics pass through);
    above ``x_cut`` the function is the limiting linear function
    ``intercept_limit + slope_limit*x`` minus a decaying gap -- the same tail
    family ``LinearInterp`` builds in (exponential or power-law, fitted or
    explicit exponent), with the same validity guards and the same numerically
    stable evaluation, but sourced from the wrapped function's level (and, for
    the fitted forms, slope) at ``x_cut`` instead of from the top two knots of
    a grid.

    This decouples the tail LAW from the in-grid REPRESENTATION. Previously
    the decay machinery existed only baked into ``LinearInterp``
    (``CubicInterp`` carries only the legacy exponential form, with no
    power-law option and no explicit-exponent hook), so the asymptotically
    correct power-law tail could not be attached to a cubic body, an
    econforge interpolant, or a fitted functional form. With this wrapper the
    body is swappable: anything callable on numpy arrays composes with the
    same tail. Compare :class:`HARK.econforgeinterp.DecayInterp`, which wraps
    N-dimensional econforge interpolants with ad-hoc decay-weight schemes
    toward a general limit function but provides no derivatives and not the
    power-law gap law; and ``ConsAggShockModel.make_cFunc_slice``, the policy
    layer that chooses WHICH tail to attach to consumption slices -- this
    class is the mechanism such policy layers can target.

    # THEOREM-REF[BufferStockTheory-Latest @ c181870f :: theory/powerlaw-decay/final_proof.md :: §7. The computational payoff: why the compactified core is the right presentation :: The extrapolation form of record :: https://llorracc.github.io/BufferStockTheory-Latest/powerlaw-decay-theory/]
    #   The power-law gap tail C*(x + h)**(-q) is the theorem's extrapolation
    #   form of record for buffer-stock consumption functions, and it is a
    #   property of the FUNCTION being approximated, not of the interpolation
    #   scheme used inside the grid -- which is why the tail is factored out
    #   here as a wrapper composable over any in-grid representation.

    Parameters
    ----------
    interp : callable
        The wrapped interpolant: any object mapping a numpy array of query
        points to a numpy array of values (all HARK 1D interpolants qualify).
        A ``derivative`` method is required by the FITTED tail forms
        (``decay_extrap_Q=None``) and by the DEFAULT two-term explicit mode
        (``decay_extrap_terms=2``), both of which read the slope at
        ``x_cut``; the one-term explicit mode (``decay_extrap_terms=1``)
        works on any bare callable.
    intercept_limit : float
        Intercept of the limiting linear function (required).
    slope_limit : float
        Slope of the limiting linear function (required).
    x_cut : float or None (default)
        The handoff point: queries strictly above it get the decay tail.
        Defaults to ``interp.x_list[-1]`` when the wrapped interpolant exposes
        a grid; otherwise it must be supplied. ``x_cut`` need not be a knot:
        a cut above the body's grid composes the tail with the body's own
        extrapolation (the level is read wherever the cut is), and a cut
        below the body's top TRUNCATES the body there and replaces the rest
        with the tail law -- useful for stopping a solved function at a
        certified point without rebuilding it.
    decay_extrap_form : str
        ``'powerlaw'`` (default): the gap decays as
        ``gap(x) = A*((x + h)/(x_cut + h))**(-Q)`` with pivot
        ``h = intercept_limit/slope_limit``; the asymptotically correct tail
        for buffer-stock consumption functions. ``'exp'``: the legacy
        exponential ``gap(x) = A*exp(-B*(x - x_cut))``. The exponential form
        is retained for parity with ``LinearInterp``'s long-standing default
        and is DEPRECATED here (selecting it warns): it is not merely less
        accurate but impossible as an asymptotic form for the consumption
        gap, so new code should have no reason to choose it.

        # THEOREM-REF[BufferStockTheory-Latest @ c181870f :: theory/powerlaw-decay/statement.md :: Proposition A0 (no exponential decay — GIC-free) :: https://llorracc.github.io/BufferStockTheory-Latest/powerlaw-decay-theory/statement/]
        #   Prop A0: the true gap below the perfect-foresight asymptote can
        #   never decay faster than 1/x, so any exponential tail understates
        #   it asymptotically -- the reason 'exp' is deprecated at birth in
        #   this new API while remaining LinearInterp's untouched legacy
        #   default.

        ``'moderation_tail'``: tail-only use of the Method-of-Moderation
        COORDINATES. The solved body is untouched, and this is NOT the full
        Method of Moderation solution representation (see the
        MethodOfModeration paper) -- it borrows MoM's coordinates for the
        extrapolation region only. With ``mEx = x - x_min`` and
        ``hEx = intercept_limit/slope_limit + x_min``, the gap is expressed
        as ``omega = gap/(slope_limit*hEx)`` in (0, 1) -- the position
        between the limiting ("optimist") line and the gap ceiling anchored
        at ``x_min`` (the "pessimist" line) -- and, through
        ``chi = log((1 - omega)/omega)`` and ``mu = log(mEx)``, the tail is

            ``chi(mu) = chi_cut + Q*u + (chip_cut - Q)*(1 - exp(-u))``,
            ``u = mu - mu_cut >= 0``.

        The decay law is the POWER LAW ``gap ~ mEx**(-Q)``, asymptotically
        the same law as ``'powerlaw'``; the logistic link is only the
        coordinate system carrying it (this is NOT logistic or exponential
        decay of the gap). Properties, each unit-tested: level- AND
        slope-matched (C1) at ANY cut with NO guard -- the bounded
        ``exp(-u)`` correction absorbs an arbitrarily steep local slope,
        exactly the ``Q_fit >= Q + 1`` region where the two-term power-law
        attachment must fall back to a kinked one-term tail; the gap stays
        strictly inside ``(0, slope_limit*hEx)`` for all finite queries
        (bounds by construction); and it collapses exactly to the
        pinned-slope line when ``chip_cut == Q``. The derivative floor
        ``f' > slope_limit`` holds wherever the gap is locally shrinking
        (asymptotically always); at a cut where the body's gap is WIDENING
        (``chip_cut < 0``, a configuration the fitted forms must refuse),
        C1 fidelity necessarily continues the widening before the tail
        bends toward the line -- the bounds still hold throughout. Requires
        ``decay_extrap_Q`` (there is NO fitted mode: fitting the asymptotic
        slope from the cut is definitionally the tangent extrapolation this
        form exists to correct) and ``x_min``, and always reads the body's
        derivative at the cut. Applicability violations RAISE (they signal
        inconsistent inputs) instead of warn-and-disable: the body's level
        at the cut must lie strictly between the pessimist and optimist
        lines.
    decay_extrap_Q : float or None (default), keyword-only
        ``None``: the tail is FITTED -- level- and slope-matched at ``x_cut``
        exactly as ``LinearInterp`` fits from its top two knots (the wrapped
        interpolant's ``derivative(x_cut)`` supplies the slope). A positive
        float (requires ``decay_extrap_form='powerlaw'``): use this EXPLICIT
        decay exponent; the tail is level-matched (continuous) by
        construction, and under the default ``decay_extrap_terms=2`` it is
        slope-matched (C1) as well; only the one-term variant
        (``decay_extrap_terms=1``) has the C1 kink whose size the
        ``LinearInterp.decay_extrap_Q`` documentation derives.
        The theory exponent for buffer-stock consumption functions is
        ``min(1, q*)`` from :mod:`HARK.ConsumptionSaving.pf_decay`.
    decay_extrap_terms : int, keyword-only (default 2)
        Consulted only with ``decay_extrap_Q``. ``2`` (default): the C1
        TWO-TERM attachment ``gap = A*z**(-Q) + A2*z**(-(Q+1))``, level- AND
        slope-matched at ``x_cut`` with the theory exponent leading (F11
        closed forms ``A2 = gap*(Q_fit - Q)``, ``A = gap - A2``; collapses
        exactly to one term when the local fitted rate equals ``Q``; warns
        and falls back to one term when ``Q_fit >= Q + 1``). ``1``: the
        level-matched one-term tail (C1 kink; the only explicit mode
        available to derivative-less bodies).

        WHY the two-term default: it guards against Jacobian problems in
        SSJ-type (sequence-space Jacobian) approaches -- policy derivatives
        are primitive inputs to SSJ Jacobian/fake-news construction and to
        differentiation through the solution, and a C1 kink at the
        attachment point makes them discontinuous for queries crossing the
        cut. The second exponent is ``Q + 1`` (an attachment term absorbing
        exactly the one-term kink), NOT the theory-subleading pair, whose
        spacing ``|q* - 1|`` vanishes at near-resonance calibrations.

        # THEOREM-REF[BufferStockTheory-Latest @ 3f4b021e :: theory/powerlaw-decay/grid_design_final_spec.md :: THE SPEC (owner-proposed scheme, sharpened by F1–F8) :: F11 — The C1 two-term attachment :: https://llorracc.github.io/BufferStockTheory-Latest/powerlaw-decay-theory/grid-design-final-spec/]
        #   F11: derivation of the closed-form amplitudes, the conditioning
        #   argument for Q+1, and the property proofs (below-line, MPC
        #   floor, concavity condition, guard + one-term fallback).
    x_min : float or None (default), keyword-only
        The lower support point of the moderation coordinates (for
        consumption functions: ``mNrmMin``, the "pessimist" bound anchoring
        the gap ceiling; often <= 0 under a natural borrowing constraint).
        REQUIRED by ``decay_extrap_form='moderation_tail'`` (which needs
        ``mEx = x - x_min > 0`` at the cut and
        ``hEx = intercept_limit/slope_limit + x_min > 0``). OPTIONAL with
        ``'powerlaw'``: supplying it enriches the two-term guard-trip
        warning with the exact steepness diagnosis (``s_mu``, the
        coordinate amplification factor ``1 + hEx/mEx``, and the guard-safe
        grid boundary ``mEx > Q*hEx``). Rejected with ``'exp'``.

    Notes
    -----
    LEVEL CONTINUITY AT ``x_cut`` IS AN INVARIANT OF THIS CLASS: every tail
    it can attach is level-matched to the wrapped function at the cut (the
    amplitude is always the level gap read there), so the composed function
    never jumps. There is deliberately no amplitude-override hook: imposing
    an external amplitude (e.g. a closed-form boundary value) at a
    pre-asymptotic cut forces a level discontinuity, which is never
    acceptable. Under the default two-term attachment the composed function
    is C1 at the cut as well; the one discontinuity the class can exhibit
    is the documented C1 (derivative-only) kink of the one-term
    explicit-exponent mode (``decay_extrap_terms=1``).

    Validity guards mirror ``LinearInterp``: the fitted forms require the
    level at ``x_cut`` strictly below the limiting line, approaching it
    (slope above ``slope_limit``), ``slope_limit > 0``, and a positive pivot;
    the explicit-exponent form relaxes the slope condition (its rescue case).
    On guard failure the wrapper warns and DISABLES the tail, and queries
    above ``x_cut`` simply delegate to the wrapped interpolant -- the
    composable analog of ``LinearInterp`` falling back to its naive top-
    segment extrapolation.

    # THEOREM-REF[BufferStockTheory-Latest @ c181870f :: theory/powerlaw-decay/final_proof.md :: §2. Model, conditions, and the imported foundations :: Carroll–Kimball 1996 :: https://llorracc.github.io/BufferStockTheory-Latest/powerlaw-decay-theory/]
    #   Imported foundations L0-L2': a converged buffer-stock consumption
    #   function is strictly increasing and strictly concave (Carroll-Kimball
    #   1996) and approaches its PF asymptote from below with slope falling
    #   to the limiting MPC -- the configuration the guards enforce at x_cut;
    #   a violation signals bad inputs, so the tail is refused rather than
    #   risk a divergent extrapolation.

    ``distance_criteria`` recurses into the wrapped interpolant only: with
    the limiting line and tail policy fixed, the tail parameters are
    deterministic functions of the body, so successive-iterate distances
    through the body control the total distance in solver convergence checks.

    Fine print (adversarially established): the tail parameters are a
    CONSTRUCTION-TIME snapshot -- mutating the wrapped interpolant afterward
    moves the body but not the tail. Limiting-line parameters are coerced to
    float64, so byte-parity with ``LinearInterp`` holds for float64/Python-
    float inputs (exotic dtypes like float32 round differently in
    LinearInterp's raw-dtype arithmetic). Parity also presumes a
    non-degenerate top knot: on a grid whose last two x-values coincide,
    LinearInterp's fitted tail is internally inconsistent (it level-matches a
    knot value its own evaluator does not return), while this wrapper
    level-matches what the body actually evaluates to at ``x_cut``.
    """

    distance_criteria = ["interp"]

    def __init__(
        self,
        interp,
        intercept_limit,
        slope_limit,
        x_cut=None,
        decay_extrap_form="powerlaw",
        *,
        decay_extrap_Q=None,
        decay_extrap_terms=2,
        x_min=None,
    ):
        self.interp = interp
        if decay_extrap_form not in ("exp", "powerlaw", "moderation_tail"):
            raise ValueError(
                "decay_extrap_form must be 'exp', 'powerlaw', or "
                "'moderation_tail', got " + repr(decay_extrap_form)
            )
        if isinstance(decay_extrap_terms, bool) or decay_extrap_terms not in (1, 2):
            raise ValueError(
                "decay_extrap_terms must be 1 or 2, got "
                + repr(decay_extrap_terms)
            )
        self.decay_extrap_form = decay_extrap_form
        if intercept_limit is None or slope_limit is None:
            raise ValueError(
                "DecayTailInterp requires intercept_limit and slope_limit "
                "(the limiting linear function is what the tail decays toward)"
            )
        if x_min is not None:
            if decay_extrap_form == "exp":
                raise ValueError(
                    "x_min parameterizes the moderation coordinates (and the "
                    "power-law guard diagnosis); it is meaningless with "
                    "decay_extrap_form='exp'"
                )
            x_min = float(x_min)
            if not np.isfinite(x_min):
                raise ValueError("x_min must be finite, got " + repr(x_min))
        self.decay_x_min = x_min
        if decay_extrap_Q is not None:
            if decay_extrap_form not in ("powerlaw", "moderation_tail"):
                raise ValueError(
                    "decay_extrap_Q requires decay_extrap_form='powerlaw' "
                    "or 'moderation_tail'"
                )
            decay_extrap_Q = float(decay_extrap_Q)
            if not np.isfinite(decay_extrap_Q) or decay_extrap_Q <= 0.0:
                raise ValueError(
                    "decay_extrap_Q must be a positive finite float, got "
                    + repr(decay_extrap_Q)
                )
        if decay_extrap_form == "moderation_tail":
            if decay_extrap_Q is None:
                raise ValueError(
                    "decay_extrap_form='moderation_tail' requires "
                    "decay_extrap_Q (the explicit asymptotic exponent; the "
                    "theory value is min(1, q*) from "
                    "HARK.ConsumptionSaving.pf_decay) -- there is no fitted "
                    "mode: fitting the asymptotic slope at the cut is the "
                    "tangent extrapolation this form exists to correct"
                )
            if x_min is None:
                raise ValueError(
                    "decay_extrap_form='moderation_tail' requires x_min "
                    "(the moderation coordinates' lower support point; "
                    "mNrmMin for consumption functions)"
                )
            if decay_extrap_terms == 1:
                raise ValueError(
                    "decay_extrap_terms does not apply to "
                    "decay_extrap_form='moderation_tail' (the form is C1 by "
                    "construction); leave it at its default"
                )
        if x_cut is None:
            x_list = getattr(interp, "x_list", None)
            if x_list is None:
                raise ValueError(
                    "x_cut is required: the wrapped interpolant exposes no "
                    "x_list grid to supply a default handoff point"
                )
            x_cut = x_list[-1]
        self.x_cut = float(x_cut)
        if not np.isfinite(self.x_cut):
            raise ValueError(
                "x_cut must be finite, got " + repr(x_cut)
            )
        if decay_extrap_form == "exp":
            warnings.warn(
                "DecayTailInterp(decay_extrap_form='exp'): the exponential "
                "tail is retained only for parity with LinearInterp's legacy "
                "default and is deprecated in this class; the power-law form "
                "is the asymptotically correct tail (and the default).",
                DeprecationWarning,
                stacklevel=2,
            )
        self.intercept_limit = float(intercept_limit)
        self.slope_limit = float(slope_limit)

        # Tail inputs, read from the wrapped function where LinearInterp
        # reads its top two knots: the level at x_cut always; the slope at
        # x_cut only for the fitted forms (explicit-Q needs none, so any
        # bare callable composes).
        level_at_cut = float(np.asarray(self.interp(np.array([self.x_cut]))).ravel()[0])
        if not np.isfinite(level_at_cut):
            raise ValueError(
                "DecayTailInterp: the wrapped interpolant returns a non-finite "
                f"level ({level_at_cut!r}) at x_cut={self.x_cut!r} -- x_cut is "
                "outside its usable domain (e.g. below a lower_extrap=False "
                "grid bottom); choose a cut where the body is defined"
            )
        # np.float64, NOT a Python float: LinearInterp's grid-sourced
        # level_diff is a numpy scalar, so its fitted-B division by an
        # exactly-zero gap yields inf/nan (exp form then correctly returns
        # the line; powerlaw warn-disables) -- a Python float would raise
        # ZeroDivisionError on the same reachable boundary configuration.
        level_diff = np.float64(
            self.intercept_limit + self.slope_limit * self.x_cut - level_at_cut
        )
        self.decay_extrap = False
        if decay_extrap_form == "moderation_tail":
            der = getattr(interp, "derivative", None)
            if der is None:
                raise ValueError(
                    "DecayTailInterp: the moderation tail slope-matches at "
                    "x_cut and needs the wrapped interpolant's derivative "
                    "there; this interpolant has none"
                )
            slope_at_cut = float(
                np.asarray(der(np.array([self.x_cut]))).ravel()[0]
            )
            if not np.isfinite(slope_at_cut):
                raise ValueError(
                    "DecayTailInterp: the wrapped interpolant returns a "
                    f"non-finite derivative ({slope_at_cut!r}) at "
                    f"x_cut={self.x_cut!r}; the moderation tail needs a "
                    "finite slope there"
                )
            self._init_moderation_tail(level_diff, decay_extrap_Q, slope_at_cut)
            return
        if decay_extrap_Q is not None:
            slope_at_cut = None
            if decay_extrap_terms == 2:
                der = getattr(interp, "derivative", None)
                if der is None:
                    raise ValueError(
                        "DecayTailInterp: the default two-term (C1) tail "
                        "slope-matches at x_cut and needs the wrapped "
                        "interpolant's derivative there; this interpolant "
                        "has none -- pass decay_extrap_terms=1 for the "
                        "level-matched one-term tail"
                    )
                slope_at_cut = float(
                    np.asarray(der(np.array([self.x_cut]))).ravel()[0]
                )
                if not np.isfinite(slope_at_cut):
                    raise ValueError(
                        "DecayTailInterp: the wrapped interpolant returns a "
                        f"non-finite derivative ({slope_at_cut!r}) at "
                        f"x_cut={self.x_cut!r}; the two-term tail needs a "
                        "finite slope there -- fix the cut or pass "
                        "decay_extrap_terms=1"
                    )
            self._init_explicit_tail(
                level_diff, decay_extrap_Q, decay_extrap_terms, slope_at_cut
            )
            return
        der = getattr(interp, "derivative", None)
        if der is None:
            raise ValueError(
                "DecayTailInterp: the fitted decay forms infer the tail from "
                "the wrapped interpolant's slope at x_cut, but this "
                "interpolant has no derivative method; pass decay_extrap_Q "
                "for a level-matched explicit-exponent tail instead"
            )
        slope_at_cut = float(np.asarray(der(np.array([self.x_cut]))).ravel()[0])
        if not np.isfinite(slope_at_cut):
            raise ValueError(
                "DecayTailInterp: the wrapped interpolant returns a non-finite "
                f"derivative ({slope_at_cut!r}) at x_cut={self.x_cut!r}; the "
                "fitted decay forms need a finite slope there -- fix the cut "
                "or pass decay_extrap_Q"
            )
        # Zero-uncertainty guard, exactly as in LinearInterp: a body already
        # ON the limiting line has no gap to decay.
        if not np.isclose(self.slope_limit, slope_at_cut, atol=1e-15):
            slope_diff = self.slope_limit - slope_at_cut
            self.decay_extrap_A = level_diff
            self.decay_extrap_B = -slope_diff / level_diff
            self.decay_extrap = True
            if decay_extrap_form == "powerlaw":
                self._init_powerlaw_fitted_tail(level_diff, slope_diff)

    def _init_powerlaw_fitted_tail(self, level_diff, slope_diff):
        """Fitted power-law tail setup, mirroring
        ``LinearInterp._init_powerlaw_decay`` with the top knot replaced by
        ``x_cut``: level- and slope-matched there, valid only when the body
        sits strictly below the limiting line and approaches it; otherwise
        warn and disable (queries above ``x_cut`` then delegate to the body).
        """
        ok = self.slope_limit > 0.0 and level_diff > 0.0 and self.decay_extrap_B > 0.0
        if ok:
            pivot = self.x_cut + self.intercept_limit / self.slope_limit
            ok = pivot > 0.0
        if not ok:
            warnings.warn(
                "DecayTailInterp(decay_extrap_form='powerlaw'): the wrapped "
                "interpolant at x_cut is not strictly below the limiting "
                "line with slope strictly above slope_limit (level_diff="
                f"{level_diff:.6g}, slope_diff={slope_diff:.6g}, "
                f"slope_limit={self.slope_limit:.6g}); disabling decay "
                "extrapolation -- queries above x_cut delegate to the "
                "wrapped interpolant."
            )
            self.decay_extrap = False
            return
        self.decay_extrap_pivot = pivot
        self.decay_extrap_Q = self.decay_extrap_B * pivot
        self.decay_extrap_Q_source = "fitted"

    def _init_explicit_tail(self, level_diff, Q, terms, slope_at_cut):
        """Explicit-exponent tail setup, mirroring
        ``LinearInterp._init_explicit_Q_decay`` (relaxed guard: no slope
        condition -- the rescue case an explicit exponent exists to serve).
        Level-matched at the cut unconditionally (the class invariant); with
        ``terms=2`` (default) also slope-matched (the F11 C1 attachment,
        ``A2 = gap*(Q_fit - Q)``, ``A = gap - A2``), falling back to one
        term with a warning when ``Q_fit >= Q + 1``."""
        ok = self.slope_limit > 0.0 and level_diff > 0.0
        pivot = None
        if ok:
            pivot = self.x_cut + self.intercept_limit / self.slope_limit
            ok = pivot > 0.0
        if not ok:
            warnings.warn(
                "DecayTailInterp(decay_extrap_Q=...): explicit-exponent decay "
                "requires slope_limit > 0, a level at x_cut strictly below "
                f"the limiting line, and a positive pivot (level_diff="
                f"{level_diff:.6g}, slope_limit={self.slope_limit:.6g}); "
                "disabling decay extrapolation -- queries above x_cut "
                "delegate to the wrapped interpolant."
            )
            self.decay_extrap = False
            return
        self.decay_extrap_pivot = pivot
        self.decay_extrap_Q = Q
        self.decay_extrap_Q_source = "explicit"
        if terms == 2:
            # same expressions as LinearInterp's setup, sourced from the
            # wrapped function's slope reading (byte-parity on HARK bodies)
            slope_diff = self.slope_limit - slope_at_cut
            self.decay_extrap_B = -slope_diff / level_diff
            Q_fit = self.decay_extrap_B * pivot
            A2 = level_diff * (Q_fit - Q) if np.isfinite(Q_fit) else np.nan
            if np.isfinite(Q_fit) and Q_fit < Q + 1.0 and A2 != 0.0:
                # THEOREM-REF[BufferStockTheory-Latest @ 3f4b021e :: theory/powerlaw-decay/grid_design_final_spec.md :: THE SPEC (owner-proposed scheme, sharpened by F1–F8) :: F11 — The C1 two-term attachment :: https://llorracc.github.io/BufferStockTheory-Latest/powerlaw-decay-theory/grid-design-final-spec/]
                #   F11 closed forms: level+slope matching with the theory
                #   exponent leading; an EXACT collapse (A2 == 0) stores the
                #   one-term representation (byte-identical to terms=1); a
                #   non-finite Q_fit falls back rather than attach NaN.
                self.decay_extrap_A2 = A2
                self.decay_extrap_A = level_diff - self.decay_extrap_A2
                self.decay_extrap_terms = 2
                self.decay_extrap = True
                return
            if not (np.isfinite(Q_fit) and Q_fit < Q + 1.0):
                hNrm = pivot - self.x_cut
                diag = ""
                if self.decay_x_min is not None and np.isfinite(Q_fit):
                    hEx = hNrm + self.decay_x_min
                    mEx_cut = self.x_cut - self.decay_x_min
                    if mEx_cut > 0.0 and hEx > 0.0:
                        # exact steepness decomposition:
                        # Q_fit = s_mu * (1 + hEx/mEx); guard-safe boundary
                        # mEx > Q*hEx (the human-wealth-scale rule)
                        amp = 1.0 + hEx / mEx_cut
                        diag = (
                            f" Diagnosis (x_min={self.decay_x_min:.6g}): "
                            f"s_mu={Q_fit / amp:.4g} times coordinate "
                            f"amplification {amp:.4g}; guard-safe boundary "
                            f"mEx_cut > Q*hEx = {Q * hEx:.4g}, vs mEx_cut = "
                            f"{mEx_cut:.4g} here."
                        )
                warnings.warn(
                    "DecayTailInterp(decay_extrap_Q=..., decay_extrap_terms="
                    f"2): the fitted rate Q_fit={Q_fit:.4g} is not finite "
                    f"and strictly below Q+1={Q + 1.0:.4g} (at or above it, "
                    "the two-term leading amplitude would be non-positive; "
                    "non-finite signals a degenerate reading); falling back "
                    "to the one-term level-matched tail (C1 kink at the "
                    "cut). A steep fitted rate at a human-wealth-dominated "
                    "cut is usually coordinate amplification rather than "
                    "anomalous decay (the diagnosis below separates the two "
                    "when x_min is supplied): remedies are (a) extend the "
                    "grid top toward Q*hNrm ~= "
                    f"{Q * hNrm:.4g} (the human-wealth-scale rule), or (b) "
                    "decay_extrap_form='moderation_tail' (C1 at any cut, no "
                    f"guard; requires x_min).{diag}"
                )
        else:
            # fitted-rate diagnostic is undefined without a body slope reading
            self.decay_extrap_B = np.nan
        self.decay_extrap_A = level_diff
        self.decay_extrap_A2 = 0.0
        self.decay_extrap_terms = 1
        self.decay_extrap = True

    def _init_moderation_tail(self, level_diff, Q, slope_at_cut):
        """Moderation-coordinates C1 tail setup (tail-only; NOT the full
        Method of Moderation -- see the class docstring). Level- and
        slope-matched at ANY cut with no guard: the bounded ``exp(-u)``
        correction absorbs an arbitrarily steep local slope, so there is no
        analog of the two-term ``Q_fit >= Q + 1`` fallback. Violations of
        the moderation premises signal INCONSISTENT INPUTS and raise
        (unlike the fitted/explicit power-law guards, which warn and
        disable): the wrapped body's level at the cut must lie strictly
        between the limiting ("optimist") line and the gap ceiling anchored
        at ``x_min`` (the "pessimist" line).

        # THEOREM-REF[BufferStockTheory-Latest @ c181870f :: theory/powerlaw-decay/final_proof.md :: §7. The computational payoff: why the compactified core is the right presentation :: The extrapolation form of record :: https://llorracc.github.io/BufferStockTheory-Latest/powerlaw-decay-theory/]
        #   Same decay law as the power-law forms: chi(mu) asymptotically
        #   linear with slope Q in mu = ln(x - x_min) is IDENTICALLY
        #   gap ~ mEx**(-Q); the moderation coordinates change only the
        #   ATTACHMENT (C1 at any cut, bounds built in), not the law.
        """
        x_min = self.decay_x_min
        if not self.slope_limit > 0.0:
            raise ValueError(
                "moderation_tail requires slope_limit > 0, got "
                f"{self.slope_limit!r}"
            )
        if not x_min < self.x_cut:
            raise ValueError(
                "moderation_tail requires x_min < x_cut, got "
                f"x_min={x_min!r} >= x_cut={self.x_cut!r}"
            )
        hEx = self.intercept_limit / self.slope_limit + x_min
        if not hEx > 0.0:
            raise ValueError(
                "moderation_tail requires hEx = intercept_limit/slope_limit "
                "+ x_min > 0 (the optimist-pessimist gap ceiling scale), "
                f"got hEx={hEx!r}"
            )
        ceiling = self.slope_limit * hEx
        if not 0.0 < level_diff < ceiling:
            raise ValueError(
                "moderation_tail requires the body's level at x_cut "
                "strictly between the pessimist and optimist lines: need "
                "0 < level_diff < slope_limit*hEx, got level_diff="
                f"{float(level_diff)!r} vs ceiling={float(ceiling)!r}"
            )
        mEx_cut = self.x_cut - x_min
        omega_cut = float(level_diff) / ceiling
        # chi and its mu-slope at the cut, from the body's own level and
        # slope: chip = -mEx*g'/(g*(1-omega)) with g' = slope_limit - body'
        slope_diff = self.slope_limit - slope_at_cut
        self.decay_hEx = hEx
        self.decay_gap_ceiling = ceiling
        self.decay_mEx_cut = mEx_cut
        self.decay_chi_cut = float(np.log((1.0 - omega_cut) / omega_cut))
        self.decay_chip_cut = float(
            -mEx_cut * slope_diff / (float(level_diff) * (1.0 - omega_cut))
        )
        self.decay_extrap_Q = Q
        self.decay_extrap_Q_source = "explicit"
        self.decay_extrap = True

    def _body_y(self, x):
        return np.asarray(self.interp(x), dtype=float)

    def _body_der(self, x):
        der = getattr(self.interp, "derivative", None)
        if der is not None:
            return np.asarray(der(x), dtype=float)
        # finite-difference fallback on the BODY, mirroring the default
        # HARKinterpolator1D._der (only reachable for derivative-less bodies,
        # which require explicit-Q tails)
        eps = 1e-8
        return (
            np.asarray(self.interp(x + eps), dtype=float)
            - np.asarray(self.interp(x), dtype=float)
        ) / eps

    def _body_both(self, x):
        ewd = getattr(self.interp, "eval_with_derivative", None)
        if ewd is not None:
            y, dydx = ewd(x)
            return np.asarray(y, dtype=float), np.asarray(dydx, dtype=float)
        return self._body_y(x), self._body_der(x)

    def _tail_y_der(self, x_above, want_y, want_der):
        """Tail level/derivative above ``x_cut``: the same formulas, in the
        same numerically stable arrangement, as
        ``LinearInterp._apply_upper_decay``."""
        x_temp = x_above - self.x_cut
        if self.decay_extrap_form == "moderation_tail":
            # chi(mu) = chi_cut + Q*u + (chip_cut - Q)*(1 - e^-u): stable
            # omega recovery via expm1 (chi clipped at 700, where omega has
            # already underflowed to ~1e-304 and the tail IS the line);
            # e^-u computed as the exact coordinate ratio mEx_cut/mEx.
            mEx = x_above - self.decay_x_min
            u = np.log1p(x_temp / self.decay_mEx_cut)
            eu = self.decay_mEx_cut / mEx
            chi = (
                self.decay_chi_cut
                + self.decay_extrap_Q * u
                + (self.decay_chip_cut - self.decay_extrap_Q) * (1.0 - eu)
            )
            omega = 1.0 / (2.0 + np.expm1(np.minimum(chi, 700.0)))
            y = (
                self.intercept_limit + self.slope_limit * x_above
                - self.decay_gap_ceiling * omega
                if want_y
                else None
            )
            dydx = (
                self.slope_limit
                + self.decay_gap_ceiling * omega * (1.0 - omega)
                * (
                    self.decay_extrap_Q
                    + (self.decay_chip_cut - self.decay_extrap_Q) * eu
                ) / mEx
                if want_der
                else None
            )
            return y, dydx
        if self.decay_extrap_form == "powerlaw":
            if getattr(self, "decay_extrap_terms", 1) == 2:
                # C1 two-term attachment (F11); expressions identical to
                # LinearInterp._apply_upper_decay's two-term branch
                lw = np.log1p(x_temp / self.decay_extrap_pivot)
                w1 = np.exp(-self.decay_extrap_Q * lw)
                w2 = np.exp(-(self.decay_extrap_Q + 1.0) * lw)
                decay = self.decay_extrap_A * w1 + self.decay_extrap_A2 * w2
                y = (
                    self.intercept_limit + self.slope_limit * x_above - decay
                    if want_y
                    else None
                )
                dydx = (
                    self.slope_limit + (
                        self.decay_extrap_Q * self.decay_extrap_A * w1
                        + (self.decay_extrap_Q + 1.0) * self.decay_extrap_A2 * w2
                    ) / (x_temp + self.decay_extrap_pivot)
                    if want_der
                    else None
                )
                return y, dydx
            decay = self.decay_extrap_A * np.exp(
                -self.decay_extrap_Q * np.log1p(x_temp / self.decay_extrap_pivot)
            )
            y = (
                self.intercept_limit + self.slope_limit * x_above - decay
                if want_y
                else None
            )
            dydx = (
                self.slope_limit
                + self.decay_extrap_Q / (x_temp + self.decay_extrap_pivot) * decay
                if want_der
                else None
            )
            return y, dydx
        decay = self.decay_extrap_A * np.exp(-self.decay_extrap_B * x_temp)
        y = (
            self.intercept_limit + self.slope_limit * x_above - decay
            if want_y
            else None
        )
        dydx = self.slope_limit + self.decay_extrap_B * decay if want_der else None
        return y, dydx

    def _evaluate(self, x):
        x = np.asarray(x, dtype=float)
        if not self.decay_extrap:
            return self._body_y(x)
        above = x > self.x_cut
        if not np.any(above):
            return self._body_y(x)
        y = np.empty(x.shape, dtype=float)
        body = ~above
        if np.any(body):
            y[body] = self._body_y(x[body])
        y[above], _ = self._tail_y_der(x[above], True, False)
        return y

    def _der(self, x):
        x = np.asarray(x, dtype=float)
        if not self.decay_extrap:
            return self._body_der(x)
        above = x > self.x_cut
        if not np.any(above):
            return self._body_der(x)
        dydx = np.empty(x.shape, dtype=float)
        body = ~above
        if np.any(body):
            dydx[body] = self._body_der(x[body])
        _, dydx[above] = self._tail_y_der(x[above], False, True)
        return dydx

    def _evalAndDer(self, x):
        x = np.asarray(x, dtype=float)
        if not self.decay_extrap:
            return self._body_both(x)
        above = x > self.x_cut
        if not np.any(above):
            return self._body_both(x)
        y = np.empty(x.shape, dtype=float)
        dydx = np.empty(x.shape, dtype=float)
        body = ~above
        if np.any(body):
            yb, db = self._body_both(x[body])
            y[body] = yb
            dydx[body] = db
        ya, da = self._tail_y_der(x[above], True, True)
        y[above] = ya
        dydx[above] = da
        return y, dydx


class KappaBarTailInterp(HARKinterpolator1D):
    """
    Constraint-end (maximal-MPC) tail wrapper over ANY 1D interpolant -- the
    bottom-end member of the ``DecayTailInterp`` family. At and above a knot
    ``x_knot`` every query is delegated, unchanged, to the wrapped
    interpolant; below the knot (and above the binding minimum ``mNrmMin``)
    the function is the Theorem CE constraint-end form

        c(m) = MPCmax*me - K*me**(1.0 + CRRA),      me = m - mNrmMin,
        K = (MPCmax*me_knot - y_knot)/me_knot**(1.0 + CRRA)   (value-matching),

    so the composed function is continuous at the knot by construction (the
    amplitude is always the knot's own gap below the ``MPCmax`` line -- the
    same level-continuity invariant as ``DecayTailInterp``). Queries at or
    below ``mNrmMin`` return 0.0 (consumption is zero at the constraint and
    undefined below it).

    # THEOREM-REF[BufferStockTheory-Latest @ 12b0b178 :: theory/powerlaw-decay/statement.md :: st-thm-CE :: https://llorracc.github.io/BufferStockTheory-Latest/powerlaw-decay-theory/statement/]
    #   Theorem CE: the constraint-end approach exponent is the CRRA itself
    #   (q_down = rho): c = kap_bar*me - K*me**(1+rho)*(1+o(1)) as me -> 0,
    #   with NO log-periodic prefactor (the binding one-step map is the
    #   deterministic worst-branch contraction lambda = wp**(1/rho)*Thorn_Gamma
    #   < 1). No eigenvalue problem at this end: the exponent needs no
    #   root-finder, unlike the high-wealth min(1, q*).
    # THEOREM-REF[BufferStockTheory-Latest @ 12b0b178 :: theory/powerlaw-decay/statement.md :: st-thm-CE-psi :: https://llorracc.github.io/BufferStockTheory-Latest/powerlaw-decay-theory/statement/]
    #   Theorem CE-psi (regime I): with permanent shocks the same q_down = rho
    #   law holds under the uniform-contraction criterion
    #   p_eff**(1/rho)*Thorn_Gamma < psi_min, with p_eff the worst-JOINT-atom
    #   mass (exactly HARK's WorstIncPrb accounting); outside it (regime II)
    #   q_down = min(rho, s*_+) and this form is NOT theorem-backed -- callers
    #   should gate on ``HARK.ConsumptionSaving.pf_decay.ce_psi_regime``.
    # THEOREM-REF[BufferStockTheory-Latest @ 12b0b178 :: theory/powerlaw-decay/statement.md :: st-prop-C1-psi :: https://llorracc.github.io/BufferStockTheory-Latest/powerlaw-decay-theory/statement/]
    #   kap_bar = 1 - p_eff**(1/rho)*Thorn_R (infinite horizon); finite
    #   horizon: HARK's ``calc_mpc_max`` recursion IS Prop C2
    #   (kap_bar_{T-n}**-1 = 1 + p_eff**(1/rho)*Thorn_R*kap_bar_{T-n+1}**-1,
    #   terminal anchor 1), so passing the solver's ``MPCmaxUnc`` /
    #   ``solution.MPCmax`` is exact at ANY horizon.
    # THEOREM-REF[BufferStockTheory-Latest @ 12b0b178 :: theory/powerlaw-decay/statement.md :: st-cor-C4 :: https://llorracc.github.io/BufferStockTheory-Latest/powerlaw-decay-theory/statement/]
    #   Knot placement rule (the guard message below): the knot reads the
    #   constraint asymptote to relative tolerance tol iff
    #   me_knot <= (tol*kap_bar/K)**(1/rho) -- push the grid bottom below that
    #   scale (``pf_decay.aXtraMin_from_tail_tol`` inverts it).

    The MPC behavior is the theorem's content: c'(m) = MPCmax - (1 +
    CRRA)*K*me**CRRA rises to MPCmax as me -> 0. This replaces (a) the EGM
    bottom SECANT from the constraint corner to the first gridpoint, whose
    slope understates MPCmax by exactly K*me_knot**CRRA, and (b) fitted-
    tangent lower extrapolations, whose MPC can diverge (measured
    Method-of-Moderation counterexample: MPC 3.56 at me = 1e-8).

    Modes (``strict``) and the in-solve constructor (``try_make``)
    ---------------------------------------------------------------
    ``strict=True`` (default; post-solve attachment, where a violation is a
    real diagnosis): requires the Theorem CE regime at the knot, ``K >= 0``
    (knot at or below the MPCmax line) and ``K*me_knot**CRRA < MPCmax``
    (positivity of c below the knot; monotone in me, so holding at the knot
    certifies the whole tail). Violation raises ``ValueError`` with the
    st-cor-C4 grid-rule message. ``strict=False`` admits the two-sided
    bootstrap CORRIDOR ``|K|*me_knot**CRRA < MPCmax`` (a knot ABOVE the
    MPCmax line -- K < 0 -- from a contaminated iterate or solve; the tail
    then approaches the MPCmax line from above and the inherited bias decays
    like (me/me_knot)**CRRA). ``in_regime`` records ``K >= 0``;
    ``knot_rel_deficit = K*me_knot**CRRA/MPCmax`` is the SIGNED st-cor-C4
    'tol' realized at the knot.

    ``try_make`` is the guarded constructor for IN-SOLVE use (re-anchoring
    each backward step): it returns ``None`` -- the caller keeps its default
    assembly for that step -- unless the corridor holds AND the tail's MPC
    over the exposed segment (0, me_knot] lies in ``(0, MPCmax]``:

        MPC <= MPCmax  <=>  K >= 0,
        MPC > 0 at the knot  <=>  (1 + CRRA)*K*me_knot**CRRA < MPCmax.

    The MPC-range exposure gate closes the corridor's two recorded leaks (a
    large-K corridor tail turns its MPC negative near the knot; a K < 0
    corridor tail carries MPC above MPCmax throughout), so in-solve exposure
    is always a theorem-shaped tail. Sign self-correction across iterations
    is preserved by the FALLBACK, not by exposing K < 0 tails: a refused
    iterate keeps HARK's bottom secant, which lies BELOW the true concave
    consumption function and therefore biases the next iterate's knot back
    below the MPCmax line (K > 0), re-activating the tail. (This differs
    from the reference stack's crude ``c = m`` rail, whose MPC-1 bias has the
    opposite sign and made a strict in-solve gate deadlock there.)

    Parameters
    ----------
    interp : callable
        The wrapped interpolant (any object mapping numpy query arrays to
        value arrays; all HARK 1D interpolants qualify). Queries at or above
        ``x_knot`` delegate to it; a ``derivative`` method is delegated to
        where present.
    MPCmax : float
        The maximal MPC at the constraint end: the solver's ``MPCmaxUnc`` /
        ``solution.MPCmax`` (the analytic Prop C2 recursion, exact at any
        horizon; infinite-horizon fixed point 1 - wp**(1/CRRA)*Thorn_R).
    CRRA : float
        Relative risk aversion; ALSO the tail exponent (Theorem CE).
    mNrmMin : float
        The binding minimum of market resources (``solution.mNrmMin``); 0 for
        zero-income-atom calibrations. Must lie strictly below ``x_knot``.
        Attach ONLY when the NATURAL borrowing constraint binds: with an
        artificially-constrained kink the constraint end has MPC 1 and no
        kap_bar asymptote (HARK's ``MPCmaxNow = 1.0`` override branch).
    x_knot : float
        Market resources at the attachment knot -- the first EGM gridpoint
        above the constraint corner (``m_for_interpolation[1]`` at the
        solver's assembly site).
    y_knot : float or None (default)
        Consumption at the knot. ``None`` reads the wrapped interpolant at
        ``x_knot`` (must be finite); the solver passes its own exact node
        value ``c_for_interpolation[1]``.
    strict : bool (default True)
        See Modes above.
    """

    distance_criteria = ["interp"]

    def __init__(self, interp, MPCmax, CRRA, mNrmMin, x_knot, y_knot=None,
                 strict=True):
        self.interp = interp
        self.MPCmax = float(MPCmax)
        self.CRRA = float(CRRA)
        self.mNrmMin = float(mNrmMin)
        self.x_knot = float(x_knot)
        if not (np.isfinite(self.MPCmax) and self.MPCmax > 0.0):
            raise ValueError(
                "KappaBarTailInterp: MPCmax must be a positive finite float "
                "(the solver's MPCmaxUnc / solution.MPCmax), got "
                + repr(MPCmax)
            )
        if not (np.isfinite(self.CRRA) and self.CRRA > 0.0):
            raise ValueError(
                "KappaBarTailInterp: CRRA must be a positive finite float, "
                "got " + repr(CRRA)
            )
        me_knot = self.x_knot - self.mNrmMin
        if not (np.isfinite(me_knot) and me_knot > 0.0):
            raise ValueError(
                "KappaBarTailInterp: the knot must lie strictly above "
                f"mNrmMin (x_knot={self.x_knot!r}, mNrmMin={self.mNrmMin!r})"
            )
        if y_knot is None:
            y_knot = float(
                np.asarray(self.interp(np.array([self.x_knot]))).ravel()[0]
            )
        self.y_knot = float(y_knot)
        if not np.isfinite(self.y_knot):
            raise ValueError(
                "KappaBarTailInterp: non-finite consumption at the knot "
                f"({self.y_knot!r} at x_knot={self.x_knot!r})"
            )
        # Value-matching amplitude (level continuity at the knot, the class
        # invariant): K = (MPCmax*me - c)/me**(1+CRRA).
        self.K = (self.MPCmax * me_knot - self.y_knot) / me_knot ** (
            1.0 + self.CRRA
        )
        deficit = self.K * me_knot**self.CRRA  # signed st-cor-C4 'tol'
        if strict:
            if not (self.K >= 0.0 and deficit < self.MPCmax):
                raise ValueError(
                    "KappaBarTailInterp: knot outside the constraint-end "
                    "regime -- enlarge the grid bottom (st-cor-C4: the knot "
                    "obeys the asymptote to relative tolerance tol only for "
                    "me_knot <= (tol*MPCmax/K)**(1/CRRA); K >= 0 requires "
                    "y_knot <= MPCmax*me_knot, positivity requires "
                    "K*me_knot**CRRA < MPCmax i.e. y_knot > 0; "
                    "pf_decay.aXtraMin_from_tail_tol inverts the rule). "
                    f"Got K={self.K:.6g}, me_knot={me_knot:.6g}, "
                    f"MPCmax={self.MPCmax:.6g}. With permanent shocks or a "
                    "positive worst atom also re-check MPCmax and mNrmMin "
                    "(regime gate: pf_decay.ce_psi_regime, st-rem-CE-regime)."
                )
        else:
            if not abs(deficit) < self.MPCmax:
                raise ValueError(
                    "KappaBarTailInterp(strict=False): knot outside even the "
                    "bootstrap corridor |K|*me_knot**CRRA < MPCmax (relative "
                    "distance of y_knot from the MPCmax line >= 100%) -- use "
                    "try_make, which falls back instead of constructing this."
                )
        self.strict = bool(strict)
        self.in_regime = bool(self.K >= 0.0)
        self.knot_rel_deficit = float(deficit / self.MPCmax)
        self.mpc_at_knot = float(self.MPCmax - (1.0 + self.CRRA) * deficit)

    @classmethod
    def try_make(cls, interp, MPCmax, CRRA, mNrmMin, x_knot, y_knot=None):
        """Guarded constructor for IN-SOLVE use: returns ``None`` (the caller
        keeps its default bottom assembly for that backward step) unless the
        knot admits a corridor tail whose MPC over the exposed segment lies
        in (0, MPCmax] -- the exposure gate described in the class docstring.
        Never raises on out-of-regime knots; malformed scalar inputs
        (non-positive MPCmax/CRRA, knot at or below mNrmMin, non-finite
        y_knot) also return None so a transient broken iterate cannot abort
        a solve."""
        if MPCmax is None or not np.isfinite(float(MPCmax)) or float(MPCmax) <= 0.0:
            return None
        if not np.isfinite(float(CRRA)) or float(CRRA) <= 0.0:
            return None
        me_knot = float(x_knot) - float(mNrmMin)
        if not (np.isfinite(me_knot) and me_knot > 0.0):
            return None
        if y_knot is None:
            y_knot = float(np.asarray(interp(np.array([float(x_knot)]))).ravel()[0])
        y_knot = float(y_knot)
        if not np.isfinite(y_knot):
            return None
        K = (float(MPCmax) * me_knot - y_knot) / me_knot ** (1.0 + float(CRRA))
        deficit = K * me_knot ** float(CRRA)
        # bootstrap corridor + the MPC-in-(0, MPCmax] exposure gate
        if not abs(deficit) < float(MPCmax):
            return None
        if not (K >= 0.0 and (1.0 + float(CRRA)) * deficit < float(MPCmax)):
            return None
        return cls(interp, MPCmax, CRRA, mNrmMin, x_knot, y_knot=y_knot,
                   strict=True)

    def _body_y(self, x):
        return np.asarray(self.interp(x), dtype=float)

    def _body_der(self, x):
        der = getattr(self.interp, "derivative", None)
        if der is not None:
            return np.asarray(der(x), dtype=float)
        eps = 1e-8
        return (
            np.asarray(self.interp(x + eps), dtype=float)
            - np.asarray(self.interp(x), dtype=float)
        ) / eps

    def _tail_y(self, x_below):
        me = np.maximum(x_below - self.mNrmMin, 0.0)
        return self.MPCmax * me - self.K * me ** (1.0 + self.CRRA)

    def _tail_der(self, x_below):
        # MPC of the tail: -> MPCmax as me -> 0 (Theorem CE's content); the
        # me <= 0 clip returns the limit MPCmax at/below the constraint.
        me = np.maximum(x_below - self.mNrmMin, 0.0)
        return self.MPCmax - (1.0 + self.CRRA) * self.K * me**self.CRRA

    def _evaluate(self, x):
        x = np.asarray(x, dtype=float)
        below = x < self.x_knot
        if not np.any(below):
            return self._body_y(x)
        y = np.empty(x.shape, dtype=float)
        body = ~below
        if np.any(body):
            y[body] = self._body_y(x[body])
        y[below] = self._tail_y(x[below])
        return y

    def _der(self, x):
        x = np.asarray(x, dtype=float)
        below = x < self.x_knot
        if not np.any(below):
            return self._body_der(x)
        dydx = np.empty(x.shape, dtype=float)
        body = ~below
        if np.any(body):
            dydx[body] = self._body_der(x[body])
        dydx[below] = self._tail_der(x[below])
        return dydx

    def _evalAndDer(self, x):
        return self._evaluate(x), self._der(x)


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
        self._init_cubic_grids(x_list, y_list, dydx_list)

        # Define lower extrapolation as linear function (or just NaN)
        if lower_extrap:
            lower_row = [y_list[0], dydx_list[0], 0.0, 0.0]
        else:
            lower_row = [np.nan, np.nan, np.nan, np.nan]

        # Per-segment cubic coefficients on segments mapped to [0,1] (vectorized)
        xL = self.x_list[:-1]
        xR = self.x_list[1:]
        yL = self.y_list[:-1]
        yR = self.y_list[1:]
        Span = xR - xL
        dydxL = self.dydx_list[:-1] * Span
        dydxR = self.dydx_list[1:] * Span
        seg = np.column_stack(
            [
                yL,
                dydxL,
                3 * (yR - yL) - 2 * dydxL - dydxR,
                2 * (yL - yR) + dydxL + dydxR,
            ]
        )

        # Calculate extrapolation coefficients as a decay toward limiting function y = mx+b
        x_top = self.x_list[-1]
        y_top = self.y_list[-1]
        if slope_limit is None and intercept_limit is None:
            slope_limit = self.dydx_list[-1]
            intercept_limit = y_top - slope_limit * x_top
        gap = slope_limit * x_top + intercept_limit - y_top
        slope = slope_limit - self.dydx_list[-1]
        if (gap != 0) and (slope <= 0):
            upper_row = [intercept_limit, slope_limit, gap, slope / gap]
        elif slope > 0:
            # fixing a problem when slope is positive
            upper_row = [intercept_limit, slope_limit, 0, 0]
        else:
            upper_row = [intercept_limit, slope_limit, gap, 0]

        self.coeffs = np.vstack([lower_row, seg, upper_row])

    def _classify_segments(self, x):
        """Bucket ``x`` into below-grid, above-grid, and in-bounds positions and
        precompute in-bounds coefficient slices and the local segment ``alpha``.
        Returns ``(m, out_bot, out_top, in_bnds, i, coeffs_in, alpha)``."""
        m = len(x)
        pos = np.searchsorted(self.x_list, x, side="right")
        out_bot = pos == 0
        out_top = pos == self.n
        in_bnds = np.logical_not(np.logical_or(out_bot, out_top))
        i = pos[in_bnds]
        coeffs_in = self.coeffs[i, :]
        alpha = (x[in_bnds] - self.x_list[i - 1]) / (
            self.x_list[i] - self.x_list[i - 1]
        )
        return m, out_bot, out_top, in_bnds, i, coeffs_in, alpha

    def _eval_y_outbounds(self, y, out_bot, out_top, x):
        """Apply lower/upper extrapolation values to ``y`` at out-of-bounds points."""
        y[out_bot] = self.coeffs[0, 0] + self.coeffs[0, 1] * (
            x[out_bot] - self.x_list[0]
        )
        alpha_top = x[out_top] - self.x_list[self.n - 1]
        y[out_top] = (
            self.coeffs[self.n, 0]
            + x[out_top] * self.coeffs[self.n, 1]
            - self.coeffs[self.n, 2] * np.exp(alpha_top * self.coeffs[self.n, 3])
        )
        return alpha_top

    def _eval_dydx_outbounds(self, dydx, out_bot, out_top, alpha_top):
        """Apply lower/upper extrapolation derivatives to ``dydx``."""
        dydx[out_bot] = self.coeffs[0, 1]
        dydx[out_top] = self.coeffs[self.n, 1] - self.coeffs[self.n, 2] * self.coeffs[
            self.n, 3
        ] * np.exp(alpha_top * self.coeffs[self.n, 3])

    def _evaluate(self, x):
        """
        Returns the level of the interpolated function at each value in x.  Only
        called internally by HARKinterpolator1D.__call__ (etc).
        """
        m, out_bot, out_top, in_bnds, _i, coeffs_in, alpha = self._classify_segments(x)
        y = np.zeros(m)
        if y.size > 0:
            y[in_bnds] = coeffs_in[:, 0] + alpha * (
                coeffs_in[:, 1] + alpha * (coeffs_in[:, 2] + alpha * coeffs_in[:, 3])
            )
            self._eval_y_outbounds(y, out_bot, out_top, x)
        return y

    def _der(self, x):
        """
        Returns the first derivative of the interpolated function at each value
        in x. Only called internally by HARKinterpolator1D.derivative (etc).
        """
        m, out_bot, out_top, in_bnds, i, coeffs_in, alpha = self._classify_segments(x)
        dydx = np.zeros(m)
        if dydx.size > 0:
            dydx[in_bnds] = (
                coeffs_in[:, 1]
                + alpha * (2 * coeffs_in[:, 2] + alpha * 3 * coeffs_in[:, 3])
            ) / (self.x_list[i] - self.x_list[i - 1])
            alpha_top = x[out_top] - self.x_list[self.n - 1]
            self._eval_dydx_outbounds(dydx, out_bot, out_top, alpha_top)
        return dydx

    def _evalAndDer(self, x):
        """
        Returns the level and first derivative of the function at each value in
        x.  Only called internally by HARKinterpolator1D.eval_and_der (etc).
        """
        m, out_bot, out_top, in_bnds, i, coeffs_in, alpha = self._classify_segments(x)
        y = np.zeros(m)
        dydx = np.zeros(m)
        if y.size > 0:
            y[in_bnds] = coeffs_in[:, 0] + alpha * (
                coeffs_in[:, 1] + alpha * (coeffs_in[:, 2] + alpha * coeffs_in[:, 3])
            )
            dydx[in_bnds] = (
                coeffs_in[:, 1]
                + alpha * (2 * coeffs_in[:, 2] + alpha * 3 * coeffs_in[:, 3])
            ) / (self.x_list[i] - self.x_list[i - 1])
            alpha_top = self._eval_y_outbounds(y, out_bot, out_top, x)
            self._eval_dydx_outbounds(dydx, out_bot, out_top, alpha_top)
        return y, dydx


class CubicHermiteInterp(HARKinterpolator1D):
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
        self._init_cubic_grids(x_list, y_list, dydx_list)

        self._chs = CubicHermiteSpline(
            self.x_list, self.y_list, self.dydx_list, extrapolate=None
        )
        self.coeffs = np.flip(self._chs.c.T, 1)

        # Define lower extrapolation as linear function (or just NaN)
        if lower_extrap:
            temp = np.array([self.y_list[0], self.dydx_list[0], 0, 0])
        else:
            temp = np.array([np.nan, np.nan, np.nan, np.nan])

        self.coeffs = np.vstack((temp, self.coeffs))

        x1 = self.x_list[self.n - 1]
        y1 = self.y_list[self.n - 1]

        # Calculate extrapolation coefficients as a decay toward limiting function y = mx+b
        if slope_limit is None and intercept_limit is None:
            slope_limit = self.dydx_list[-1]
            intercept_limit = self.y_list[-1] - slope_limit * self.x_list[-1]
        gap = slope_limit * x1 + intercept_limit - y1
        slope = slope_limit - self.dydx_list[self.n - 1]
        if (gap != 0) and (slope <= 0):
            temp = np.array([intercept_limit, slope_limit, gap, slope / gap])
        elif slope > 0:
            # fixing a problem when slope is positive
            temp = np.array([intercept_limit, slope_limit, 0, 0])
        else:
            temp = np.array([intercept_limit, slope_limit, gap, 0])
        self.coeffs = np.vstack((self.coeffs, temp))

    def out_of_bounds(self, x):
        out_bot = x < self.x_list[0]
        out_top = x > self.x_list[-1]
        return out_bot, out_top

    def _evaluate(self, x):
        """
        Returns the level of the interpolated function at each value in x.  Only
        called internally by HARKinterpolator1D.__call__ (etc).
        """
        out_bot, out_top = self.out_of_bounds(x)

        return self._eval_helper(x, out_bot, out_top)

    def _eval_helper(self, x, out_bot, out_top):
        y = self._chs(x)

        # Do the "out of bounds" evaluation points
        if any(out_bot):
            y[out_bot] = self.coeffs[0, 0] + self.coeffs[0, 1] * (
                x[out_bot] - self.x_list[0]
            )

        if any(out_top):
            alpha = x[out_top] - self.x_list[self.n - 1]
            y[out_top] = (
                self.coeffs[self.n, 0]
                + x[out_top] * self.coeffs[self.n, 1]
                - self.coeffs[self.n, 2] * np.exp(alpha * self.coeffs[self.n, 3])
            )

        return y

    def _der(self, x):
        """
        Returns the first derivative of the interpolated function at each value
        in x. Only called internally by HARKinterpolator1D.derivative (etc).
        """
        out_bot, out_top = self.out_of_bounds(x)

        return self._der_helper(x, out_bot, out_top)

    def _der_helper(self, x, out_bot, out_top):
        dydx = self._chs(x, nu=1)

        # Do the "out of bounds" evaluation points
        if any(out_bot):
            dydx[out_bot] = self.coeffs[0, 1]
        if any(out_top):
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
        out_bot, out_top = self.out_of_bounds(x)
        y = self._eval_helper(x, out_bot, out_top)
        dydx = self._der_helper(x, out_bot, out_top)
        return y, dydx

    def der_interp(self, nu=1):
        """
        Construct a new piecewise polynomial representing the derivative.
        See `scipy` for additional documentation:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicHermiteSpline.html
        """
        return self._chs.derivative(nu)

    def antider_interp(self, nu=1):
        """
        Construct a new piecewise polynomial representing the antiderivative.

        Antiderivative is also the indefinite integral of the function,
        and derivative is its inverse operation.

        See `scipy` for additional documentation:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicHermiteSpline.html
        """
        return self._chs.antiderivative(nu)

    def integrate(self, a, b, extrapolate=False):
        """
        Compute a definite integral over a piecewise polynomial.

        See `scipy` for additional documentation:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicHermiteSpline.html
        """
        return self._chs.integrate(a, b, extrapolate)

    def roots(self, discontinuity=True, extrapolate=False):
        """
        Find real roots of the the piecewise polynomial.

        See `scipy` for additional documentation:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicHermiteSpline.html
        """
        return self._chs.roots(discontinuity, extrapolate)

    def solve(self, y=0.0, discontinuity=True, extrapolate=False):
        """
        Find real solutions of the the equation ``pp(x) == y``.

        See `scipy` for additional documentation:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicHermiteSpline.html
        """
        return self._chs.solve(y, discontinuity, extrapolate)


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
    """

    distance_criteria = ["x_list", "y_list", "f_values"]

    def __init__(self, f_values, x_list, y_list):
        self.f_values = f_values
        self.x_list = _coerce_1d_grid(x_list)
        self.y_list = _coerce_1d_grid(y_list)
        _check_grid_dimensions(2, self.f_values, self.x_list, self.y_list)
        self.x_n = self.x_list.size
        self.y_n = self.y_list.size

    def _locate_xy_indices(self, x, y):
        """Return clamped search indices for ``x`` and ``y`` shared by ``_evaluate``,
        ``_derX``, and ``_derY``."""
        return (
            _locate_clipped(self.x_list, x, self.x_n),
            _locate_clipped(self.y_list, y, self.y_n),
        )

    def _evaluate(self, x, y):
        """
        Returns the level of the interpolated function at each value in x,y.
        Only called internally by HARKinterpolator2D.__call__ (etc).
        """
        x_pos, y_pos = self._locate_xy_indices(x, y)
        alpha = _cell_fraction(self.x_list, x_pos, x)
        beta = _cell_fraction(self.y_list, y_pos, y)
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
        x_pos, y_pos = self._locate_xy_indices(x, y)
        beta = _cell_fraction(self.y_list, y_pos, y)
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
        x_pos, y_pos = self._locate_xy_indices(x, y)
        alpha = _cell_fraction(self.x_list, x_pos, x)
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
    """

    distance_criteria = ["f_values", "x_list", "y_list", "z_list"]

    def __init__(self, f_values, x_list, y_list, z_list):
        self.f_values = f_values
        self.x_list = _coerce_1d_grid(x_list)
        self.y_list = _coerce_1d_grid(y_list)
        self.z_list = _coerce_1d_grid(z_list)
        _check_grid_dimensions(3, self.f_values, self.x_list, self.y_list, self.z_list)
        self.x_n = self.x_list.size
        self.y_n = self.y_list.size
        self.z_n = self.z_list.size

    def _locate_xyz_indices(self, x, y, z):
        """Return clamped search indices for ``x``, ``y``, ``z`` shared by
        ``_evaluate`` and the three derivative methods."""
        return (
            _locate_clipped(self.x_list, x, self.x_n),
            _locate_clipped(self.y_list, y, self.y_n),
            _locate_clipped(self.z_list, z, self.z_n),
        )

    def _evaluate(self, x, y, z):
        """
        Returns the level of the interpolated function at each value in x,y,z.
        Only called internally by HARKinterpolator3D.__call__ (etc).
        """
        x_pos, y_pos, z_pos = self._locate_xyz_indices(x, y, z)
        alpha = _cell_fraction(self.x_list, x_pos, x)
        beta = _cell_fraction(self.y_list, y_pos, y)
        gamma = _cell_fraction(self.z_list, z_pos, z)
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
        x_pos, y_pos, z_pos = self._locate_xyz_indices(x, y, z)
        beta = _cell_fraction(self.y_list, y_pos, y)
        gamma = _cell_fraction(self.z_list, z_pos, z)
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
        x_pos, y_pos, z_pos = self._locate_xyz_indices(x, y, z)
        alpha = _cell_fraction(self.x_list, x_pos, x)
        gamma = _cell_fraction(self.z_list, z_pos, z)
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
        x_pos, y_pos, z_pos = self._locate_xyz_indices(x, y, z)
        alpha = _cell_fraction(self.x_list, x_pos, x)
        beta = _cell_fraction(self.y_list, y_pos, y)
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
        An array of w values, with length designated w_n.
    x_list : numpy.array
        An array of x values, with length designated x_n.
    y_list : numpy.array
        An array of y values, with length designated y_n.
    z_list : numpy.array
        An array of z values, with length designated z_n.
    """

    distance_criteria = ["f_values", "w_list", "x_list", "y_list", "z_list"]

    def __init__(self, f_values, w_list, x_list, y_list, z_list):
        self.f_values = f_values
        self.w_list = _coerce_1d_grid(w_list)
        self.x_list = _coerce_1d_grid(x_list)
        self.y_list = _coerce_1d_grid(y_list)
        self.z_list = _coerce_1d_grid(z_list)
        _check_grid_dimensions(
            4, self.f_values, self.w_list, self.x_list, self.y_list, self.z_list
        )
        self.w_n = self.w_list.size
        self.x_n = self.x_list.size
        self.y_n = self.y_list.size
        self.z_n = self.z_list.size

    def _locate_quad_indices(self, w, x, y, z):
        """
        Return clipped lookup indices ``(i, j, k, l)`` for ``(w, x, y, z)``.

        Each axis runs ``np.searchsorted`` and clips the result into
        ``[1, n - 1]`` so that ``a_list[idx - 1]`` and ``a_list[idx]``
        always bracket the query point.
        """
        return (
            _locate_clipped(self.w_list, w, self.w_n),
            _locate_clipped(self.x_list, x, self.x_n),
            _locate_clipped(self.y_list, y, self.y_n),
            _locate_clipped(self.z_list, z, self.z_n),
        )

    def _evaluate(self, w, x, y, z):
        """
        Returns the level of the interpolated function at each value in x,y,z.
        Only called internally by HARKinterpolator4D.__call__ (etc).
        """
        i, j, k, l = self._locate_quad_indices(w, x, y, z)
        alpha = _cell_fraction(self.w_list, i, w)
        beta = _cell_fraction(self.x_list, j, x)
        gamma = _cell_fraction(self.y_list, k, y)
        delta = _cell_fraction(self.z_list, l, z)
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
        i, j, k, l = self._locate_quad_indices(w, x, y, z)
        beta = _cell_fraction(self.x_list, j, x)
        gamma = _cell_fraction(self.y_list, k, y)
        delta = _cell_fraction(self.z_list, l, z)
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
        i, j, k, l = self._locate_quad_indices(w, x, y, z)
        alpha = _cell_fraction(self.w_list, i, w)
        gamma = _cell_fraction(self.y_list, k, y)
        delta = _cell_fraction(self.z_list, l, z)
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
        i, j, k, l = self._locate_quad_indices(w, x, y, z)
        alpha = _cell_fraction(self.w_list, i, w)
        beta = _cell_fraction(self.x_list, j, x)
        delta = _cell_fraction(self.z_list, l, z)
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
        i, j, k, l = self._locate_quad_indices(w, x, y, z)
        alpha = _cell_fraction(self.w_list, i, w)
        beta = _cell_fraction(self.x_list, j, x)
        gamma = _cell_fraction(self.y_list, k, y)
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


def _init_envelope_state(obj, functions, nan_bool, lower=True):
    """Set ``compare``/``argcompare``/``functions``/``funcCount`` for an envelope."""
    if lower:
        obj.compare = np.nanmin if nan_bool else np.min
        obj.argcompare = np.nanargmin if nan_bool else np.argmin
    else:
        obj.compare = np.nanmax if nan_bool else np.max
        obj.argcompare = np.nanargmax if nan_bool else np.argmax
    obj.functions = list(functions)
    obj.funcCount = len(obj.functions)


class _Envelope1D(HARKinterpolator1D):
    """
    Base class for the lower/upper envelope of a finite set of 1D functions.

    Concrete subclasses set ``self.compare`` and ``self.argcompare`` in
    ``__init__`` (e.g. ``np.nanmin``/``np.nanargmin`` for the lower envelope,
    ``np.nanmax``/``np.nanargmax`` for the upper envelope). All evaluation
    logic is shared via ``self.compare`` and ``self.argcompare``.
    """

    distance_criteria = ["functions"]

    def _evaluate(self, x):
        """
        Returns the level of the envelope at each value in x.  Only called
        internally by HARKinterpolator1D.__call__.
        """
        fx = np.column_stack([f(x) for f in self.functions])
        return self.compare(fx, axis=1)

    def _der(self, x):
        """
        Returns the first derivative of the envelope at each value in x.  Only
        called internally by HARKinterpolator1D.derivative.
        """
        y, dydx = self._evalAndDer(x)
        return dydx  # Sadly, this is the fastest / most convenient way...

    def _evalAndDer(self, x):
        """
        Returns the level and first derivative of the envelope at each value
        in x.  Only called internally by HARKinterpolator1D.eval_and_der.
        """
        fx = np.column_stack([f(x) for f in self.functions])
        i = self.argcompare(fx, axis=1)
        y = fx[np.arange(len(x)), i]
        dydx = np.zeros_like(y)
        for j in np.unique(i):
            c = i == j
            dydx[c] = self.functions[j].derivative(x[c])
        return y, dydx


class LowerEnvelope(_Envelope1D):
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

    def __init__(self, *functions, nan_bool=True):
        _init_envelope_state(self, functions, nan_bool, lower=True)


class UpperEnvelope(_Envelope1D):
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
        the upper envelope.
    """

    def __init__(self, *functions, nan_bool=True):
        _init_envelope_state(self, functions, nan_bool, lower=False)


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
        _init_envelope_state(self, functions, nan_bool, lower=True)

    def _evaluate(self, x, y):
        """
        Returns the level of the function at each value in (x,y) as the minimum
        among all of the functions.  Only called internally by
        HARKinterpolator2D.__call__.
        """
        temp = np.column_stack([f(x, y) for f in self.functions])
        return self.compare(temp, axis=1)

    def _derX(self, x, y):
        """
        Returns the first derivative of the function with respect to X at each
        value in (x,y).  Only called internally by HARKinterpolator2D._derX.
        """
        return _envelope_partial(self, (x, y), "derivativeX")

    def _derY(self, x, y):
        """
        Returns the first derivative of the function with respect to Y at each
        value in (x,y).  Only called internally by HARKinterpolator2D._derY.
        """
        return _envelope_partial(self, (x, y), "derivativeY")


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
        _init_envelope_state(self, functions, nan_bool, lower=True)

    def _evaluate(self, x, y, z):
        """
        Returns the level of the function at each value in (x,y,z) as the minimum
        among all of the functions.  Only called internally by
        HARKinterpolator3D.__call__.
        """
        temp = np.column_stack([f(x, y, z) for f in self.functions])
        return self.compare(temp, axis=1)

    def _derX(self, x, y, z):
        """
        Returns the first derivative of the function with respect to X at each
        value in (x,y,z).  Only called internally by HARKinterpolator3D._derX.
        """
        return _envelope_partial(self, (x, y, z), "derivativeX")

    def _derY(self, x, y, z):
        """
        Returns the first derivative of the function with respect to Y at each
        value in (x,y,z).  Only called internally by HARKinterpolator3D._derY.
        """
        return _envelope_partial(self, (x, y, z), "derivativeY")

    def _derZ(self, x, y, z):
        """
        Returns the first derivative of the function with respect to Z at each
        value in (x,y,z).  Only called internally by HARKinterpolator3D._derZ.
        """
        return _envelope_partial(self, (x, y, z), "derivativeZ")


class VariableLowerBoundFunc2D(HARKinterpolator2D):
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

    def _derX(self, x, y):
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

    def _derY(self, x, y):
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


class VariableLowerBoundFunc3D(HARKinterpolator3D):
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

    def _derX(self, x, y, z):
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

    def _derY(self, x, y, z):
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

    def _derZ(self, x, y, z):
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

    def _linear_y_blend(self, x, y, eval_func):
        """Evaluate ``eval_func`` on each cell's bracketing 1D interpolators
        and combine with the y-direction linear weights ``(1 - alpha)`` and
        ``alpha``. Shared by ``_evaluate`` and ``_derX``.
        """
        m = len(x)
        y_pos = _locate_clipped(self.y_list, y, self.y_n)
        out = np.full(m, np.nan)
        for i, c in _iter_unique_pairs(y_pos):
            alpha = _cell_fraction(self.y_list, i, y[c])
            out[c] = (1 - alpha) * eval_func(
                self.xInterpolators[i - 1], x[c]
            ) + alpha * eval_func(self.xInterpolators[i], x[c])
        return out

    def _evaluate(self, x, y):
        """
        Returns the level of the interpolated function at each value in x,y.
        Only called internally by HARKinterpolator2D.__call__ (etc).
        """
        return self._linear_y_blend(x, y, lambda interp, xs: interp(xs))

    def _derX(self, x, y):
        """
        Returns the derivative with respect to x of the interpolated function
        at each value in x,y. Only called internally by HARKinterpolator2D.derivativeX.
        """
        return self._linear_y_blend(x, y, lambda interp, xs: interp._der(xs))

    def _derY(self, x, y):
        """
        Returns the derivative with respect to y of the interpolated function
        at each value in x,y. Only called internally by HARKinterpolator2D.derivativeY.
        """
        m = len(x)
        y_pos = _locate_clipped(self.y_list, y, self.y_n)
        dfdy = np.full(m, np.nan)
        for i, c in _iter_unique_pairs(y_pos):
            dfdy[c] = (
                self.xInterpolators[i](x[c]) - self.xInterpolators[i - 1](x[c])
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

    def _locate_yz_indices(self, y, z):
        """Return clipped ``searchsorted`` indices for ``y`` and ``z`` shared
        by ``_evaluate`` and the three derivative methods."""
        return (
            _locate_clipped(self.y_list, y, self.y_n),
            _locate_clipped(self.z_list, z, self.z_n),
        )

    def _bilinear_loop(self, x, y, z, eval_func):
        """Bilinear blend of ``eval_func`` across ``xInterpolators`` corners.

        Shared by ``_evaluate`` (``f(x)``) and ``_derX`` (``f._der(x)``).
        """
        m = len(x)
        y_pos, z_pos = self._locate_yz_indices(y, z)
        out = np.full(m, np.nan)
        for i, j, c in _iter_unique_pairs(y_pos, z_pos):
            alpha = _cell_fraction(self.y_list, i, y[c])
            beta = _cell_fraction(self.z_list, j, z[c])
            xc = x[c]
            out[c] = (
                (1 - alpha)
                * (1 - beta)
                * eval_func(self.xInterpolators[i - 1][j - 1], xc)
                + (1 - alpha) * beta * eval_func(self.xInterpolators[i - 1][j], xc)
                + alpha * (1 - beta) * eval_func(self.xInterpolators[i][j - 1], xc)
                + alpha * beta * eval_func(self.xInterpolators[i][j], xc)
            )
        return out

    def _evaluate(self, x, y, z):
        """
        Returns the level of the interpolated function at each value in x,y,z.
        Only called internally by HARKinterpolator3D.__call__ (etc).
        """
        return self._bilinear_loop(x, y, z, lambda f, xc: f(xc))

    def _derX(self, x, y, z):
        """
        Returns the derivative with respect to x of the interpolated function
        at each value in x,y,z. Only called internally by HARKinterpolator3D.derivativeX.
        """
        return self._bilinear_loop(x, y, z, lambda f, xc: f._der(xc))

    def _derY(self, x, y, z):
        """
        Returns the derivative with respect to y of the interpolated function
        at each value in x,y,z. Only called internally by HARKinterpolator3D.derivativeY.
        """
        m = len(x)
        y_pos, z_pos = self._locate_yz_indices(y, z)
        dfdy = np.full(m, np.nan)
        for i, j, c in _iter_unique_pairs(y_pos, z_pos):
            beta = _cell_fraction(self.z_list, j, z[c])
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
        m = len(x)
        y_pos, z_pos = self._locate_yz_indices(y, z)
        dfdz = np.full(m, np.nan)
        for i, j, c in _iter_unique_pairs(y_pos, z_pos):
            alpha = _cell_fraction(self.y_list, i, y[c])
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

    def _locate_xyz_indices(self, x, y, z):
        """Return clamped ``searchsorted`` indices for ``x``, ``y``, ``z`` shared
        by ``_evaluate`` and the four derivative methods."""
        return (
            _locate_clipped(self.x_list, x, self.x_n),
            _locate_clipped(self.y_list, y, self.y_n),
            _locate_clipped(self.z_list, z, self.z_n),
        )

    def _iter_xyz_cells(self, x, y, z, x_pos, y_pos, z_pos):
        """Yield ``(i, j, k, c, alpha, beta, gamma)`` for each non-empty cell of
        the (x, y, z) grid. Shared by ``_trilinear_loop`` and the partial-derivative
        methods, all of which use a subset of these values."""
        for i, j, k, c in _iter_unique_pairs(x_pos, y_pos, z_pos):
            alpha = _cell_fraction(self.x_list, i, x[c])
            beta = _cell_fraction(self.y_list, j, y[c])
            gamma = _cell_fraction(self.z_list, k, z[c])
            yield i, j, k, c, alpha, beta, gamma

    def _trilinear_loop(self, w, x, y, z, eval_func):
        """Trilinear interpolation over ``wInterpolators[i,j,k]`` evaluated by
        ``eval_func``. Shared by ``_evaluate`` (``f(w)``) and ``_derW`` (``f._der(w)``)."""
        m = len(x)
        x_pos, y_pos, z_pos = self._locate_xyz_indices(x, y, z)
        out = np.full(m, np.nan)
        for i, j, k, c, alpha, beta, gamma in self._iter_xyz_cells(
            x, y, z, x_pos, y_pos, z_pos
        ):
            wc = w[c]
            out[c] = (
                (1 - alpha)
                * (1 - beta)
                * (1 - gamma)
                * eval_func(self.wInterpolators[i - 1][j - 1][k - 1], wc)
                + (1 - alpha)
                * (1 - beta)
                * gamma
                * eval_func(self.wInterpolators[i - 1][j - 1][k], wc)
                + (1 - alpha)
                * beta
                * (1 - gamma)
                * eval_func(self.wInterpolators[i - 1][j][k - 1], wc)
                + (1 - alpha)
                * beta
                * gamma
                * eval_func(self.wInterpolators[i - 1][j][k], wc)
                + alpha
                * (1 - beta)
                * (1 - gamma)
                * eval_func(self.wInterpolators[i][j - 1][k - 1], wc)
                + alpha
                * (1 - beta)
                * gamma
                * eval_func(self.wInterpolators[i][j - 1][k], wc)
                + alpha
                * beta
                * (1 - gamma)
                * eval_func(self.wInterpolators[i][j][k - 1], wc)
                + alpha * beta * gamma * eval_func(self.wInterpolators[i][j][k], wc)
            )
        return out

    def _evaluate(self, w, x, y, z):
        """
        Returns the level of the interpolated function at each value in w,x,y,z.
        Only called internally by HARKinterpolator4D.__call__ (etc).
        """
        return self._trilinear_loop(w, x, y, z, lambda f, ww: f(ww))

    def _derW(self, w, x, y, z):
        """
        Returns the derivative with respect to w of the interpolated function
        at each value in w,x,y,z. Only called internally by HARKinterpolator4D.derivativeW.
        """
        return self._trilinear_loop(w, x, y, z, lambda f, ww: f._der(ww))

    def _derX(self, w, x, y, z):
        """
        Returns the derivative with respect to x of the interpolated function
        at each value in w,x,y,z. Only called internally by HARKinterpolator4D.derivativeX.
        """
        m = len(x)
        x_pos, y_pos, z_pos = self._locate_xyz_indices(x, y, z)
        dfdx = np.full(m, np.nan)
        for i, j, k, c, _alpha, beta, gamma in self._iter_xyz_cells(
            x, y, z, x_pos, y_pos, z_pos
        ):
            wc = w[c]
            dfdx[c] = (
                (
                    (1 - beta) * (1 - gamma) * self.wInterpolators[i][j - 1][k - 1](wc)
                    + (1 - beta) * gamma * self.wInterpolators[i][j - 1][k](wc)
                    + beta * (1 - gamma) * self.wInterpolators[i][j][k - 1](wc)
                    + beta * gamma * self.wInterpolators[i][j][k](wc)
                )
                - (
                    (1 - beta)
                    * (1 - gamma)
                    * self.wInterpolators[i - 1][j - 1][k - 1](wc)
                    + (1 - beta) * gamma * self.wInterpolators[i - 1][j - 1][k](wc)
                    + beta * (1 - gamma) * self.wInterpolators[i - 1][j][k - 1](wc)
                    + beta * gamma * self.wInterpolators[i - 1][j][k](wc)
                )
            ) / (self.x_list[i] - self.x_list[i - 1])
        return dfdx

    def _derY(self, w, x, y, z):
        """
        Returns the derivative with respect to y of the interpolated function
        at each value in w,x,y,z. Only called internally by HARKinterpolator4D.derivativeY.
        """
        m = len(x)
        x_pos, y_pos, z_pos = self._locate_xyz_indices(x, y, z)
        dfdy = np.full(m, np.nan)
        for i, j, k, c, alpha, _beta, gamma in self._iter_xyz_cells(
            x, y, z, x_pos, y_pos, z_pos
        ):
            wc = w[c]
            dfdy[c] = (
                (
                    (1 - alpha) * (1 - gamma) * self.wInterpolators[i - 1][j][k - 1](wc)
                    + (1 - alpha) * gamma * self.wInterpolators[i - 1][j][k](wc)
                    + alpha * (1 - gamma) * self.wInterpolators[i][j][k - 1](wc)
                    + alpha * gamma * self.wInterpolators[i][j][k](wc)
                )
                - (
                    (1 - alpha)
                    * (1 - gamma)
                    * self.wInterpolators[i - 1][j - 1][k - 1](wc)
                    + (1 - alpha) * gamma * self.wInterpolators[i - 1][j - 1][k](wc)
                    + alpha * (1 - gamma) * self.wInterpolators[i][j - 1][k - 1](wc)
                    + alpha * gamma * self.wInterpolators[i][j - 1][k](wc)
                )
            ) / (self.y_list[j] - self.y_list[j - 1])
        return dfdy

    def _derZ(self, w, x, y, z):
        """
        Returns the derivative with respect to z of the interpolated function
        at each value in w,x,y,z. Only called internally by HARKinterpolator4D.derivativeZ.
        """
        m = len(x)
        x_pos, y_pos, z_pos = self._locate_xyz_indices(x, y, z)
        dfdz = np.full(m, np.nan)
        for i, j, k, c, alpha, beta, _gamma in self._iter_xyz_cells(
            x, y, z, x_pos, y_pos, z_pos
        ):
            wc = w[c]
            dfdz[c] = (
                (
                    (1 - alpha) * (1 - beta) * self.wInterpolators[i - 1][j - 1][k](wc)
                    + (1 - alpha) * beta * self.wInterpolators[i - 1][j][k](wc)
                    + alpha * (1 - beta) * self.wInterpolators[i][j - 1][k](wc)
                    + alpha * beta * self.wInterpolators[i][j][k](wc)
                )
                - (
                    (1 - alpha)
                    * (1 - beta)
                    * self.wInterpolators[i - 1][j - 1][k - 1](wc)
                    + (1 - alpha) * beta * self.wInterpolators[i - 1][j][k - 1](wc)
                    + alpha * (1 - beta) * self.wInterpolators[i][j - 1][k - 1](wc)
                    + alpha * beta * self.wInterpolators[i][j][k - 1](wc)
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

    def _linear_z_blend(self, x, y, z, eval_func):
        """Linear blend of ``eval_func`` between consecutive ``xyInterpolators``
        layers along ``z``. Shared by ``_evaluate``, ``_derX``, ``_derY``."""
        m = len(x)
        z_pos = _locate_clipped(self.z_list, z, self.z_n)
        out = np.full(m, np.nan)
        for i, c in _iter_unique_pairs(z_pos):
            alpha = _cell_fraction(self.z_list, i, z[c])
            lower = eval_func(self.xyInterpolators[i - 1], x[c], y[c])
            upper = eval_func(self.xyInterpolators[i], x[c], y[c])
            out[c] = (1 - alpha) * lower + alpha * upper
        return out

    def _evaluate(self, x, y, z):
        """
        Returns the level of the interpolated function at each value in x,y,z.
        Only called internally by HARKinterpolator3D.__call__ (etc).
        """
        return self._linear_z_blend(x, y, z, lambda f, xv, yv: f(xv, yv))

    def _derX(self, x, y, z):
        """
        Returns the derivative with respect to x of the interpolated function
        at each value in x,y,z. Only called internally by HARKinterpolator3D.derivativeX.
        """
        return self._linear_z_blend(x, y, z, lambda f, xv, yv: f.derivativeX(xv, yv))

    def _derY(self, x, y, z):
        """
        Returns the derivative with respect to y of the interpolated function
        at each value in x,y,z. Only called internally by HARKinterpolator3D.derivativeY.
        """
        return self._linear_z_blend(x, y, z, lambda f, xv, yv: f.derivativeY(xv, yv))

    def _derZ(self, x, y, z):
        """
        Returns the derivative with respect to z of the interpolated function
        at each value in x,y,z. Only called internally by HARKinterpolator3D.derivativeZ.
        """
        m = len(x)
        z_pos = _locate_clipped(self.z_list, z, self.z_n)
        dfdz = np.full(m, np.nan)
        for i, c in _iter_unique_pairs(z_pos):
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

    def _locate_yz_indices(self, y, z):
        """Return clamped ``searchsorted`` indices for ``y`` and ``z`` shared
        by ``_evaluate`` and the four derivative methods."""
        return (
            _locate_clipped(self.y_list, y, self.y_n),
            _locate_clipped(self.z_list, z, self.z_n),
        )

    def _bilinear_loop(self, w, x, y, z, eval_func):
        """Bilinear interpolation across (y, z) layers of ``wxInterpolators``,
        with each corner evaluated by ``eval_func``. Shared by ``_evaluate``,
        ``_derW``, ``_derX`` (the latter two pick a derivative method)."""
        m = len(x)
        y_pos, z_pos = self._locate_yz_indices(y, z)
        out = np.full(m, np.nan)
        for i, j, c in _iter_unique_pairs(y_pos, z_pos):
            alpha = _cell_fraction(self.y_list, i, y[c])
            beta = _cell_fraction(self.z_list, j, z[c])
            wc, xc = w[c], x[c]
            out[c] = (
                (1 - alpha)
                * (1 - beta)
                * eval_func(self.wxInterpolators[i - 1][j - 1], wc, xc)
                + (1 - alpha) * beta * eval_func(self.wxInterpolators[i - 1][j], wc, xc)
                + alpha * (1 - beta) * eval_func(self.wxInterpolators[i][j - 1], wc, xc)
                + alpha * beta * eval_func(self.wxInterpolators[i][j], wc, xc)
            )
        return out

    def _evaluate(self, w, x, y, z):
        """
        Returns the level of the interpolated function at each value in x,y,z.
        Only called internally by HARKinterpolator4D.__call__ (etc).
        """
        return self._bilinear_loop(w, x, y, z, lambda f, wc, xc: f(wc, xc))

    def _derW(self, w, x, y, z):
        """
        Returns the derivative with respect to w of the interpolated function
        at each value in w,x,y,z. Only called internally by HARKinterpolator4D.derivativeW.
        """
        # This may look strange, as we call the derivativeX() method to get the
        # derivative with respect to w, but that's just a quirk of 4D interpolations
        # beginning with w rather than x.  The derivative wrt the first dimension
        # of an element of wxInterpolators is the w-derivative of the main function.
        return self._bilinear_loop(w, x, y, z, lambda f, wc, xc: f.derivativeX(wc, xc))

    def _derX(self, w, x, y, z):
        """
        Returns the derivative with respect to x of the interpolated function
        at each value in w,x,y,z. Only called internally by HARKinterpolator4D.derivativeX.
        """
        # This may look strange, as we call the derivativeY() method to get the
        # derivative with respect to x, but that's just a quirk of 4D interpolations
        # beginning with w rather than x.  The derivative wrt the second dimension
        # of an element of wxInterpolators is the x-derivative of the main function.
        return self._bilinear_loop(w, x, y, z, lambda f, wc, xc: f.derivativeY(wc, xc))

    def _derY(self, w, x, y, z):
        """
        Returns the derivative with respect to y of the interpolated function
        at each value in w,x,y,z. Only called internally by HARKinterpolator4D.derivativeY.
        """
        m = len(x)
        y_pos, z_pos = self._locate_yz_indices(y, z)
        dfdy = np.full(m, np.nan)
        for i, j, c in _iter_unique_pairs(y_pos, z_pos):
            beta = _cell_fraction(self.z_list, j, z[c])
            dfdy[c] = (
                (
                    (1 - beta) * self.wxInterpolators[i][j - 1](w[c], x[c])
                    + beta * self.wxInterpolators[i][j](w[c], x[c])
                )
                - (
                    (1 - beta) * self.wxInterpolators[i - 1][j - 1](w[c], x[c])
                    + beta * self.wxInterpolators[i - 1][j](w[c], x[c])
                )
            ) / (self.y_list[i] - self.y_list[i - 1])
        return dfdy

    def _derZ(self, w, x, y, z):
        """
        Returns the derivative with respect to z of the interpolated function
        at each value in w,x,y,z. Only called internally by HARKinterpolator4D.derivativeZ.
        """
        m = len(x)
        y_pos, z_pos = self._locate_yz_indices(y, z)
        dfdz = np.full(m, np.nan)
        for i, j, c in _iter_unique_pairs(y_pos, z_pos):
            alpha = _cell_fraction(self.y_list, i, y[c])
            dfdz[c] = (
                (
                    (1 - alpha) * self.wxInterpolators[i - 1][j](w[c], x[c])
                    + alpha * self.wxInterpolators[i][j](w[c], x[c])
                )
                - (
                    (1 - alpha) * self.wxInterpolators[i - 1][j - 1](w[c], x[c])
                    + alpha * self.wxInterpolators[i][j - 1](w[c], x[c])
                )
            ) / (self.z_list[j] - self.z_list[j - 1])
        return dfdz


class Curvilinear2DInterp(HARKinterpolator2D):
    """
    A 2D interpolation method for curvilinear or "warped grid" interpolation, as
    in White (2015). Used for models with two endogenous states that are solved
    with the endogenous grid method. Allows multiple function outputs, but all of
    the interpolated functions must share a common curvilinear grid.

    Parameters
    ----------
    f_values: numpy.array or [numpy.array]
        One or more 2D arrays of function values such that f_values[i,j] =
        f(x_values[i,j],y_values[i,j]).
    x_values: numpy.array
        A 2D array of x values of the same shape as f_values.
    y_values: numpy.array
        A 2D array of y values of the same shape as f_values.
    """

    distance_criteria = ["f_values", "x_values", "y_values"]

    def __init__(self, f_values, x_values, y_values):
        self.multi = isinstance(f_values, list)
        f_list = f_values if self.multi else [f_values]
        my_shape = x_values.shape
        if my_shape != y_values.shape:
            raise ValueError("y_values must have the same shape as x_values!")
        prefix = "Each element of f_values" if self.multi else "f_values"
        for arr in f_list:
            if my_shape != arr.shape:
                raise ValueError(f"{prefix} must have the same shape as x_values!")

        # Stack as (N_funcs, x_n, y_n) so per-corner indexing vectorizes
        # across functions: ``self.f_values[:, x_pos, y_pos]`` returns the
        # (N_funcs, M) corner table in one numpy call.
        self.f_values = np.stack(f_list)
        self.x_values = x_values
        self.y_values = y_values
        self.x_n, self.y_n = my_shape
        self.N_funcs = len(f_list)
        self.update_polarity()

    def _dispatch(self, x, y, inner):
        """Run ``inner`` on flattened ``(x, y)``, reshape per-function results,
        and unwrap the single-function case. Shared by ``__call__``,
        ``derivativeX``, and ``derivativeY``."""
        xa = np.asarray(x)
        ya = np.asarray(y)
        S = xa.shape
        result = inner(xa.flatten(), ya.flatten())
        output = [r.reshape(S) for r in result]
        return output if self.multi else output[0]

    def __call__(self, x, y):
        """
        Modification of HARKinterpolator2D.__call__ to account for multiple outputs.
        """
        return self._dispatch(x, y, self._evaluate)

    def derivativeX(self, x, y):
        """
        Modification of HARKinterpolator2D.derivativeX to account for multiple outputs.
        """
        return self._dispatch(x, y, self._derX)

    def derivativeY(self, x, y):
        """
        Modification of HARKinterpolator2D.derivativeY to account for multiple outputs.
        """
        return self._dispatch(x, y, self._derY)

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
            self.x_values[0 : (self.x_n - 1), 0 : (self.y_n - 1)]
            + self.x_values[1 : self.x_n, 1 : self.y_n]
        )
        y_temp = 0.5 * (
            self.y_values[0 : (self.x_n - 1), 0 : (self.y_n - 1)]
            + self.y_values[1 : self.x_n, 1 : self.y_n]
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
        Only called as a subroutine of _evaluate(), etc. Uses a numba helper
        function below to accelerate computation.

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
        x_pos, y_pos = find_sector_numba(x, y, self.x_values, self.y_values)
        return x_pos, y_pos

    def find_coords(self, x, y, x_pos, y_pos):
        """
        Calculates the relative coordinates (alpha,beta) for each point (x,y),
        given the sectors (x_pos,y_pos) in which they reside.  Only called as
        a subroutine of _evaluate(), etc. Uses a numba helper function to acc-
        elerate computation, and has a "backup method" for when the math fails.

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
        alpha, beta = find_coords_numba(
            x, y, x_pos, y_pos, self.x_values, self.y_values, self.polarity
        )

        # Alternate method if there are sectors that are "too regular"
        # These points weren't able to identify coordinates
        z = np.logical_or(np.isnan(alpha), np.isnan(beta))
        if np.any(z):
            ii = x_pos[z]
            jj = y_pos[z]
            xA = self.x_values[ii, jj]
            xB = self.x_values[ii + 1, jj]
            xC = self.x_values[ii, jj + 1]
            xD = self.x_values[ii + 1, jj + 1]
            yA = self.y_values[ii, jj]
            yB = self.y_values[ii + 1, jj]
            yC = self.y_values[ii, jj + 1]
            # yD = self.y_values[ii + 1, jj + 1]
            b = xB - xA
            f = yB - yA
            kappa = f / b
            int_bot = yA - kappa * xA
            int_top = yC - kappa * xC
            int_these = y[z] - kappa * x[z]
            beta_temp = (int_these - int_bot) / (int_top - int_bot)
            x_left = beta_temp * xC + (1.0 - beta_temp) * xA
            x_right = beta_temp * xD + (1.0 - beta_temp) * xB
            alpha_temp = (x[z] - x_left) / (x_right - x_left)
            beta[z] = beta_temp
            alpha[z] = alpha_temp

        return alpha, beta

    def _evaluate(self, x, y):
        """
        Returns the level of the interpolated function at each value in x,y.
        Only called internally by __call__ (etc).

        Returns an ``(N_funcs, M)`` array of bilinearly interpolated values.
        """
        x_pos, y_pos = self.find_sector(x, y)
        alpha, beta = self.find_coords(x, y, x_pos, y_pos)

        alpha_C = 1.0 - alpha
        beta_C = 1.0 - beta
        wA = alpha_C * beta_C
        wB = alpha * beta_C
        wC = alpha_C * beta
        wD = alpha * beta

        # Bilinear interpolation, vectorized over both queries and N_funcs.
        return (
            wA * self.f_values[:, x_pos, y_pos]
            + wB * self.f_values[:, x_pos + 1, y_pos]
            + wC * self.f_values[:, x_pos, y_pos + 1]
            + wD * self.f_values[:, x_pos + 1, y_pos + 1]
        )

    def _curvilinear_partials(self, x, y):
        """
        Compute the inverse Jacobian of the (alpha, beta) -> (x, y) curvilinear
        map at each sample point, plus the function-level (dfda, dfdb) arrays.

        Returns a 5-tuple ``(x_alpha, x_beta, y_alpha, y_beta, (dfda, dfdb))``
        where ``dfda`` and ``dfdb`` are ``(N_funcs, M)`` arrays. Used by both
        ``_derX`` and ``_derY``.
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

        # Components of the alpha,beta --> x,y delta translation matrix.
        alpha_C = 1 - alpha
        beta_C = 1 - beta
        alpha_x = beta_C * (xB - xA) + beta * (xD - xC)
        alpha_y = beta_C * (yB - yA) + beta * (yD - yC)
        beta_x = alpha_C * (xC - xA) + alpha * (xD - xB)
        beta_y = alpha_C * (yC - yA) + alpha * (yD - yB)

        # Invert the delta translation matrix into x,y --> alpha,beta.
        det = alpha_x * beta_y - beta_x * alpha_y
        x_alpha = beta_y / det
        x_beta = -alpha_y / det
        y_alpha = -beta_x / det
        y_beta = alpha_x / det

        # Function corners, vectorized over (N_funcs, M).
        fA = self.f_values[:, x_pos, y_pos]
        fB = self.f_values[:, x_pos + 1, y_pos]
        fC = self.f_values[:, x_pos, y_pos + 1]
        fD = self.f_values[:, x_pos + 1, y_pos + 1]
        dfda = beta_C * (fB - fA) + beta * (fD - fC)
        dfdb = alpha_C * (fC - fA) + alpha * (fD - fB)
        return x_alpha, x_beta, y_alpha, y_beta, (dfda, dfdb)

    def _derX(self, x, y):
        """
        Returns the derivative with respect to x of the interpolated function
        at each value in x,y. Only called internally by derivativeX.
        """
        x_alpha, x_beta, _, _, (dfda, dfdb) = self._curvilinear_partials(x, y)
        return x_alpha * dfda + x_beta * dfdb

    def _derY(self, x, y):
        """
        Returns the derivative with respect to y of the interpolated function
        at each value in x,y. Only called internally by derivativeY.
        """
        _, _, y_alpha, y_beta, (dfda, dfdb) = self._curvilinear_partials(x, y)
        return y_alpha * dfda + y_beta * dfdb


# Define a function that checks whether a set of points violates a linear boundary
# defined by (x1,y1) and (x2,y2), where the latter is *COUNTER CLOCKWISE* from the
# former. Returns 1 if the point is outside the boundary and 0 otherwise.
@njit
def boundary_check(xq, yq, x1, y1, x2, y2):  # pragma: no cover
    return int((y2 - y1) * xq - (x2 - x1) * yq > x1 * y2 - y1 * x2)


# Define a numba helper function for finding the sector in the irregular grid
@njit
def find_sector_numba(X_query, Y_query, X_values, Y_values):  # pragma: no cover
    # Initialize the sector guess
    M = X_query.size
    x_n = X_values.shape[0]
    y_n = X_values.shape[1]
    ii = int(x_n / 2)
    jj = int(y_n / 2)
    top_ii = x_n - 2
    top_jj = y_n - 2

    # Initialize the output arrays
    X_pos = np.empty(M, dtype=np.int32)
    Y_pos = np.empty(M, dtype=np.int32)

    # Identify the correct sector for each point to be evaluated
    max_loops = x_n + y_n
    for m in range(M):
        found = False
        loops = 0
        while not found and loops < max_loops:
            # Get coordinates for the four vertices: (xA,yA),...,(xD,yD)
            x0 = X_query[m]
            y0 = Y_query[m]
            xA = X_values[ii, jj]
            xB = X_values[ii + 1, jj]
            xC = X_values[ii, jj + 1]
            xD = X_values[ii + 1, jj + 1]
            yA = Y_values[ii, jj]
            yB = Y_values[ii + 1, jj]
            yC = Y_values[ii, jj + 1]
            yD = Y_values[ii + 1, jj + 1]

            # Check the "bounding box" for the sector: is this guess plausible?
            D = int(y0 < np.minimum(yA, yB))
            R = int(x0 > np.maximum(xB, xD))
            U = int(y0 > np.maximum(yC, yD))
            L = int(x0 < np.minimum(xA, xC))

            # Check which boundaries are violated (and thus where to look next)
            in_box = np.all(np.logical_not(np.array([D, R, U, L])))
            if in_box:
                D = boundary_check(x0, y0, xA, yA, xB, yB)
                R = boundary_check(x0, y0, xB, yB, xD, yD)
                U = boundary_check(x0, y0, xD, yD, xC, yC)
                L = boundary_check(x0, y0, xC, yC, xA, yA)

            # Update the sector guess based on the violations
            ii_next = np.maximum(np.minimum(ii - L + R, top_ii), 0)
            jj_next = np.maximum(np.minimum(jj - D + U, top_jj), 0)

            # Check whether sector guess changed and go to next iteration
            found = (ii == ii_next) and (jj == jj_next)
            ii = ii_next
            jj = jj_next
            loops += 1

        # Put the final sector guess into the output array
        X_pos[m] = ii
        Y_pos[m] = jj

    # Return the output
    return X_pos, Y_pos


# Define a numba helper function for finding relative coordinates within sector
@njit
def find_coords_numba(
    X_query, Y_query, X_pos, Y_pos, X_values, Y_values, polarity
):  # pragma: no cover
    M = X_query.size
    alpha = np.empty(M)
    beta = np.empty(M)

    # Calculate relative coordinates in the sector for each point
    for m in range(M):
        try:
            x0 = X_query[m]
            y0 = Y_query[m]
            ii = X_pos[m]
            jj = Y_pos[m]
            xA = X_values[ii, jj]
            xB = X_values[ii + 1, jj]
            xC = X_values[ii, jj + 1]
            xD = X_values[ii + 1, jj + 1]
            yA = Y_values[ii, jj]
            yB = Y_values[ii + 1, jj]
            yC = Y_values[ii, jj + 1]
            yD = Y_values[ii + 1, jj + 1]
            p = 2.0 * polarity[ii, jj] - 1.0
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
            tau = (h * (a - x0) - d * (e - y0)) / denom
            zeta = a - x0 + c * tau
            eta = b + c * mu + d * tau
            theta = d * mu
            alph = (-eta + p * np.sqrt(eta**2 - 4 * zeta * theta)) / (2 * theta)
            bet = mu * alph + tau
        except Exception:
            alph = np.nan
            bet = np.nan
        alpha[m] = alph
        beta[m] = bet

    return alpha, beta


class DiscreteInterp(MetricObject):
    """
    An interpolator for variables that can only take a discrete set of values.

    If the function we wish to interpolate, f(args) can take on the list of
    values discrete_vals, this class expects an interpolator for the index of
    f's value in discrete_vals.
    E.g., if f(a,b,c) = discrete_vals[5], then index_interp(a,b,c) = 5.

    Parameters
    ----------
    index_interp: HARKInterpolator
        An interpolator giving an approximation to the index of the value in
        discrete_vals that corresponds to a given set of arguments.
    discrete_vals: numpy.array
        A 1D array containing the values in the range of the discrete function
        to be interpolated.
    """

    distance_criteria = ["index_interp"]

    def __init__(self, index_interp, discrete_vals):
        self.index_interp = index_interp
        self.discrete_vals = discrete_vals
        self.n_vals = len(self.discrete_vals)

    def __call__(self, *args):
        # Interpolate indices and round to integers
        inds = np.rint(self.index_interp(*args)).astype(int)
        if type(inds) is not np.ndarray:
            inds = np.array(inds)
        # Deal with out-of range indices
        inds[inds < 0] = 0
        inds[inds >= self.n_vals] = self.n_vals - 1

        # Get values from grid
        return self.discrete_vals[inds]


class IndexedInterp(MetricObject):
    """
    An interpolator for functions whose first argument is an integer-valued index.
    Constructor takes in a list of functions as its only argument. When evaluated
    at f(i,X), interpolator returns f[i](X), where X can be any number of inputs.
    This simply provides a different interface for accessing the same functions.

    Parameters
    ----------
    functions : [Callable]
        List of one or more functions to be indexed.
    """

    distance_criteria = ["functions"]

    def __init__(self, functions):
        self.functions = functions
        self.N = len(functions)

    def __call__(self, idx, *args):
        out = np.empty(idx.shape)
        out.fill(np.nan)

        for n in range(self.N):
            these = idx == n
            if not np.any(these):
                continue
            temp = [arg[these] for arg in args]
            out[these] = self.functions[n](*temp)
        return out


###############################################################################
## Functions used in discrete choice models with T1EV taste shocks ############
###############################################################################


def _log_sum_taste_shock(Vals, sigma):
    """Stabilized log-sum-exp under a T1EV taste shock with scale ``sigma``.

    Returns ``maxV + sigma * log(sum_j exp((V_j - maxV) / sigma))``.
    Caller must ensure ``sigma != 0``.
    """
    maxV = np.max(Vals, axis=0)
    sumexp = np.sum(np.exp((Vals - maxV) / sigma), axis=0)
    return maxV + sigma * np.log(sumexp)


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
        Pflat = np.argmax(Vals, axis=0)
        V = np.max(Vals, axis=0)
        Probs = np.zeros(Vals.shape)
        np.put_along_axis(Probs, Pflat[None, ...], 1, axis=0)
        return V, Probs

    LogSumV = _log_sum_taste_shock(Vals, sigma)
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
        Pflat = np.argmax(Vals, axis=0)
        Probs = np.zeros(Vals.shape)
        np.put_along_axis(Probs, Pflat[None, ...], 1, axis=0)
        return Probs

    maxV = np.max(Vals, axis=0)
    weights = np.exp((Vals - maxV) / sigma)
    return weights / np.sum(weights, axis=0)


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
        return np.amax(Vals, axis=0)
    return _log_sum_taste_shock(Vals, sigma)


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
    illegal_value : float, optional
        If provided, value to return for "out-of-bounds" inputs that return NaN
        from the pseudo-inverse value function. Most common choice is -np.inf,
        which makes the outcome infinitely bad.
    """

    distance_criteria = ["func", "CRRA"]

    def __init__(self, vFuncNvrs, CRRA, illegal_value=None):
        self.vFuncNvrs = deepcopy(vFuncNvrs)
        self.CRRA = CRRA
        self.illegal_value = illegal_value

        if hasattr(vFuncNvrs, "grid_list"):
            self.grid_list = vFuncNvrs.grid_list
        else:
            self.grid_list = None

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
        temp = self.vFuncNvrs(*vFuncArgs)
        v = CRRAutility(temp, self.CRRA)
        if self.illegal_value is not None:
            illegal = np.isnan(temp)
            v[illegal] = self.illegal_value
        return v

    def gradient(self, *args):
        # V(s) = u(vFuncNvrs(s)), so by the chain rule
        # dV/ds_i = u'(vFuncNvrs(s)) * d vFuncNvrs / ds_i.
        NvrsGrad = self.vFuncNvrs.gradient(*args)
        marg_u = CRRAutilityP(self.vFuncNvrs(*args), self.CRRA)
        grad = [g * marg_u for g in NvrsGrad]
        return grad

    def _eval_and_grad(self, *args):
        return (self.__call__(*args), self.gradient(*args))


def _eval_c_and_mpc(cFunc, *cFuncArgs):
    """Return ``(c, MPC)`` from ``cFunc`` regardless of whether it exposes
    ``eval_with_derivative`` (1D) or a ``derivativeX`` attribute (multi-D)."""
    if isinstance(cFunc, HARKinterpolator1D):
        return cFunc.eval_with_derivative(*cFuncArgs)
    if hasattr(cFunc, "derivativeX"):
        return cFunc(*cFuncArgs), cFunc.derivativeX(*cFuncArgs)
    raise TypeError(
        "cFunc does not have a 'derivativeX' attribute. Can't compute "
        "marginal marginal value."
    )


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

        if hasattr(cFunc, "grid_list"):
            self.grid_list = cFunc.grid_list
        else:
            self.grid_list = None

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
        return CRRAutilityP(self.cFunc(*cFuncArgs), rho=self.CRRA)

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

        c, MPC = _eval_c_and_mpc(self.cFunc, *cFuncArgs)
        return MPC * CRRAutilityPP(c, rho=self.CRRA)


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

        c, MPC = _eval_c_and_mpc(self.cFunc, *cFuncArgs)
        return MPC * CRRAutilityPP(c, rho=self.CRRA)

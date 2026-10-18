"""The JAX backend of ``LinearInterp`` and ``BilinearInterp``.

``LinearInterp(..., backend="jax")`` and ``BilinearInterp(..., backend="jax")``
construct the classes here, which are subclasses of the numpy classes: they are
built and validated by the numpy constructor, then hold their grids as JAX arrays
of the requested precision and evaluate with the kernels in
``HARK._interp_kernels``. They return JAX arrays, accept traced values, and are
registered as JAX pytrees, so they can be passed into ``jax.jit``, ``jax.vmap``
and ``jax.grad`` code and called there.

The backend is meant for code compiled with JAX: solvers and simulators under
``jax.jit``, batches under ``jax.vmap``, and accelerators. Code that is not
compiled should keep the numpy backend.

This module imports JAX, and ``HARK.interpolation`` imports it only when a JAX
backend is requested.
"""

import warnings

import numpy as np

from HARK import _interp_kernels as kernels
from HARK._jax import register_pytree, require_jax, resolve_dtype
from HARK.interpolation import (
    BilinearInterp,
    LinearInterp,
    _check_grid_dimensions,
    _resolve_backend,
)

jax, jnp = require_jax()

# Raised when a value needed on the host is a tracer inside jax.jit or jax.vmap
_TRACED = (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError)


def _refuse(cls_name, option):
    raise NotImplementedError(
        f"{cls_name} does not support {option} with backend='jax'."
    )


def _jax_grid(arr, dtype):
    """``arr`` as a one-dimensional JAX array, flattening (with the numpy classes'
    warning) if necessary. JAX operations only, so traced arrays pass through."""
    a = jnp.asarray(arr, dtype=dtype)
    if a.ndim != 1:
        warnings.warn("input not of the size (n, ), attempting to flatten")
        return a.ravel()
    return a


class JaxLinearInterp(LinearInterp):
    """
    ``LinearInterp`` with the JAX backend; construct it as
    ``LinearInterp(..., backend="jax")``. Parameters are those of ``LinearInterp``,
    except that ``pre_compute`` and ``indexer`` are refused.
    """

    def __init__(
        self,
        x_list,
        y_list,
        intercept_limit=None,
        slope_limit=None,
        lower_extrap=False,
        pre_compute=False,
        indexer=None,
        *,
        backend=None,
        dtype=None,
    ):
        if pre_compute:
            _refuse("LinearInterp", "pre_compute=True")
        if indexer is not None:
            _refuse("LinearInterp", "a custom indexer")
        _, dtype = _resolve_backend("jax", dtype)
        self.dtype = resolve_dtype(dtype)

        # Built with JAX operations, so that it can be constructed inside jax.jit
        # and jax.vmap from traced arrays
        self.x_list = _jax_grid(x_list, self.dtype)
        self.y_list = _jax_grid(y_list, self.dtype)
        _check_grid_dimensions(1, self.y_list, self.x_list)
        self.lower_extrap = lower_extrap
        self.x_n = self.x_list.size
        self.indexer = None
        self.decay_extrap = False
        self._decay = None
        if intercept_limit is not None and slope_limit is not None:
            self._init_decay(x_list, y_list, intercept_limit, slope_limit)

    def _init_decay(self, x_list, y_list, intercept_limit, slope_limit):
        """Build the decay extrapolation exactly as the numpy class does. Whether
        it is used depends on the values, so it needs concrete grids."""
        try:
            ref = LinearInterp(
                np.asarray(x_list),
                np.asarray(y_list),
                intercept_limit,
                slope_limit,
                backend="numpy",
            )
        except _TRACED as exc:
            raise NotImplementedError(
                "LinearInterp with backend='jax' cannot build its decay "
                "extrapolation (intercept_limit and slope_limit) from traced "
                "values, inside jax.jit or jax.vmap; construct it outside."
            ) from exc
        self.decay_extrap = ref.decay_extrap
        if ref.decay_extrap:
            self.decay_extrap_A = ref.decay_extrap_A
            self.decay_extrap_B = ref.decay_extrap_B
            self.intercept_limit = ref.intercept_limit
            self.slope_limit = ref.slope_limit
            self._decay = tuple(
                jnp.asarray(v, dtype=self.dtype)
                for v in (
                    ref.decay_extrap_A,
                    ref.decay_extrap_B,
                    ref.intercept_limit,
                    ref.slope_limit,
                )
            )

    def _evalOrDer(self, x, _eval, _Der):
        x = jnp.asarray(x, dtype=self.dtype)
        y, dydx = kernels.linear_1d(
            jnp,
            self.x_list,
            self.y_list,
            x,
            _eval,
            _Der,
            self.lower_extrap,
            self._decay,
        )
        return [out for out in (y, dydx) if out is not None]

    def _evaluate(self, x, return_indices=False):
        return self._evalOrDer(x, True, False)[0]

    def _der(self, x):
        return self._evalOrDer(x, False, True)[0]

    def _evalAndDer(self, x):
        y, dydx = self._evalOrDer(x, True, True)
        return y, dydx

    def __call__(self, x):
        z = jnp.asarray(x, dtype=self.dtype)
        return self._evaluate(z.ravel()).reshape(z.shape)

    def derivative(self, x):
        z = jnp.asarray(x, dtype=self.dtype)
        return self._der(z.ravel()).reshape(z.shape)

    derivativeX = derivative

    def eval_with_derivative(self, x):
        z = jnp.asarray(x, dtype=self.dtype)
        y, dydx = self._evalAndDer(z.ravel())
        return y.reshape(z.shape), dydx.reshape(z.shape)


def _flatten_linear(obj):
    children = (obj.x_list, obj.y_list, obj._decay)
    aux = (obj.lower_extrap, obj.dtype.name)
    return children, aux


def _unflatten_linear(aux, children):
    obj = object.__new__(JaxLinearInterp)
    obj.x_list, obj.y_list, obj._decay = children
    obj.lower_extrap, dtype_name = aux
    obj.dtype = np.dtype(dtype_name)
    obj.decay_extrap = obj._decay is not None
    obj.indexer = None
    return obj


class JaxBilinearInterp(BilinearInterp):
    """
    ``BilinearInterp`` with the JAX backend; construct it as
    ``BilinearInterp(..., backend="jax")``.
    """

    def __init__(self, f_values, x_list, y_list, *, backend=None, dtype=None):
        _, dtype = _resolve_backend("jax", dtype)
        self.dtype = resolve_dtype(dtype)

        # Built with JAX operations, so that it can be constructed inside jax.jit
        # and jax.vmap from traced arrays
        self.f_values = jnp.asarray(f_values, dtype=self.dtype)
        self.x_list = _jax_grid(x_list, self.dtype)
        self.y_list = _jax_grid(y_list, self.dtype)
        _check_grid_dimensions(2, self.f_values, self.x_list, self.y_list)
        self.x_n = self.x_list.size
        self.y_n = self.y_list.size

    def _kernel(self, which, x, y):
        return kernels.bilinear_2d(
            jnp,
            self.f_values,
            self.x_list,
            self.y_list,
            jnp.asarray(x, dtype=self.dtype),
            jnp.asarray(y, dtype=self.dtype),
            which,
        )

    def _broadcast(self, which, x, y):
        x, y = jnp.broadcast_arrays(
            jnp.asarray(x, dtype=self.dtype), jnp.asarray(y, dtype=self.dtype)
        )
        return self._kernel(which, x.ravel(), y.ravel()).reshape(x.shape)

    def _evaluate(self, x, y):
        return self._kernel("f", x, y)

    def _derX(self, x, y):
        return self._kernel("dfdx", x, y)

    def _derY(self, x, y):
        return self._kernel("dfdy", x, y)

    def __call__(self, x, y):
        return self._broadcast("f", x, y)

    def derivativeX(self, x, y):
        return self._broadcast("dfdx", x, y)

    def derivativeY(self, x, y):
        return self._broadcast("dfdy", x, y)


def _flatten_bilinear(obj):
    return (obj.f_values, obj.x_list, obj.y_list), obj.dtype.name


def _unflatten_bilinear(aux, children):
    obj = object.__new__(JaxBilinearInterp)
    obj.f_values, obj.x_list, obj.y_list = children
    obj.dtype = np.dtype(aux)
    return obj


register_pytree(JaxLinearInterp, _flatten_linear, _unflatten_linear)
register_pytree(JaxBilinearInterp, _flatten_bilinear, _unflatten_bilinear)

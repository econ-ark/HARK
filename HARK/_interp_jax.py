"""The JAX backend of HARK's interpolators.

``LinearInterp(..., backend="jax")`` and ``BilinearInterp(..., backend="jax")``
construct the classes here, which are subclasses of the numpy classes: they are
built and validated by the numpy constructor, then hold their grids as JAX arrays
of the requested precision and evaluate with the kernels in
``HARK._interp_kernels``. Interpolators built from other functions
(``LinearInterpOnInterp1D``, ``LowerEnvelope2D``, ``VariableLowerBoundFunc2D``,
``MargValueFuncCRRA``, ``MargMargValueFuncCRRA``) follow their components: given
JAX-backed components, they construct the classes here. They return JAX arrays, accept traced values, and are
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
    LinearInterpOnInterp1D,
    LowerEnvelope2D,
    MargMargValueFuncCRRA,
    MargValueFuncCRRA,
    VariableLowerBoundFunc2D,
    _check_grid_dimensions,
    _eval_c_and_mpc,
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


def _component_dtype(cls_name, components, dtype):
    """The precision shared by a composite's components; ``dtype``, if given,
    must agree with it."""
    dtypes = {np.dtype(f.dtype) for f in components}
    if len(dtypes) != 1:
        raise ValueError(f"{cls_name}: its components differ in precision.")
    common = dtypes.pop()
    if dtype is not None and np.dtype(dtype) != common:
        raise ValueError(
            f"{cls_name}: dtype {np.dtype(dtype).name} does not match its "
            f"components' {common.name}."
        )
    return common


class _JaxBacked:
    """Marks a JAX-backed function, so that composites can follow it."""

    _hark_jax_backed = True


class _Jax2D(_JaxBacked):
    """The public methods of a JAX-backed 2D interpolator: broadcast the inputs
    with JAX, then call the class's ``_evaluate``, ``_derX`` or ``_derY``."""

    def _broadcast(self, method, x, y):
        x, y = jnp.broadcast_arrays(
            jnp.asarray(x, dtype=self.dtype), jnp.asarray(y, dtype=self.dtype)
        )
        return method(x.ravel(), y.ravel()).reshape(x.shape)

    def __call__(self, x, y):
        return self._broadcast(self._evaluate, x, y)

    def derivativeX(self, x, y):
        return self._broadcast(self._derX, x, y)

    def derivativeY(self, x, y):
        return self._broadcast(self._derY, x, y)


class JaxLinearInterp(_JaxBacked, LinearInterp):
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


class JaxBilinearInterp(_Jax2D, BilinearInterp):
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

    def _evaluate(self, x, y):
        return self._kernel("f", x, y)

    def _derX(self, x, y):
        return self._kernel("dfdx", x, y)

    def _derY(self, x, y):
        return self._kernel("dfdy", x, y)


def _flatten_bilinear(obj):
    return (obj.f_values, obj.x_list, obj.y_list), obj.dtype.name


def _unflatten_bilinear(aux, children):
    obj = object.__new__(JaxBilinearInterp)
    obj.f_values, obj.x_list, obj.y_list = children
    obj.dtype = np.dtype(aux)
    return obj


class JaxLinearInterpOnInterp1D(_Jax2D, LinearInterpOnInterp1D):
    """
    ``LinearInterpOnInterp1D`` with the JAX backend, constructed when its inner
    interpolators are JAX-backed ``LinearInterp`` objects. Their grids may differ
    in length: they are stacked into rows padded with +inf, which the kernels
    never count, so every row evaluates as its own ``LinearInterp`` does.
    """

    def __init__(self, xInterpolators, y_values, *, backend=None, dtype=None):
        LinearInterpOnInterp1D.__init__(self, xInterpolators, y_values)
        for f in xInterpolators:
            if not isinstance(f, JaxLinearInterp):
                _refuse(
                    "LinearInterpOnInterp1D",
                    f"inner interpolators of type {type(f).__name__}",
                )
        self.dtype = _component_dtype("LinearInterpOnInterp1D", xInterpolators, dtype)

        # Stacked with JAX operations, so that it can be constructed inside jax.jit
        # and jax.vmap from traced arrays
        lengths = [f.x_list.shape[0] for f in xInterpolators]
        width = max(lengths)

        def padded(row, n, fill):
            return jnp.concatenate([row, jnp.full((width - n,), fill, self.dtype)])

        self._x_rows = jnp.stack(
            [padded(f.x_list, n, jnp.inf) for f, n in zip(xInterpolators, lengths)]
        )
        self._y_rows = jnp.stack(
            [padded(f.y_list, n, f.y_list[-1]) for f, n in zip(xInterpolators, lengths)]
        )
        self._n_rows = jnp.asarray(lengths)
        self._lower_rows = jnp.asarray([f.lower_extrap for f in xInterpolators])
        self._decay_rows = None
        if any(f.decay_extrap for f in xInterpolators):
            zeros = (jnp.zeros((), self.dtype),) * 4
            decay = [f._decay if f.decay_extrap else zeros for f in xInterpolators]
            has_decay = jnp.asarray([f.decay_extrap for f in xInterpolators])
            self._decay_rows = (has_decay,) + tuple(
                jnp.stack([d[i] for d in decay]) for i in range(4)
            )
        self.y_list = jnp.asarray(y_values, dtype=self.dtype)

    def _kernel(self, which, x, y):
        return kernels.linear_on_linear_2d(
            jnp,
            self._x_rows,
            self._y_rows,
            self._n_rows,
            self.y_list,
            jnp.asarray(x, dtype=self.dtype),
            jnp.asarray(y, dtype=self.dtype),
            which,
            self._lower_rows,
            self._decay_rows,
        )

    def _evaluate(self, x, y):
        return self._kernel("f", x, y)

    def _derX(self, x, y):
        return self._kernel("dfdx", x, y)

    def _derY(self, x, y):
        return self._kernel("dfdy", x, y)


_ON_INTERP_FIELDS = (
    "xInterpolators",
    "_x_rows",
    "_y_rows",
    "_n_rows",
    "_lower_rows",
    "_decay_rows",
    "y_list",
)


def _flatten_on_interp(obj):
    children = tuple(getattr(obj, k) for k in _ON_INTERP_FIELDS)
    return children, (obj.dtype.name, obj.y_n)


def _unflatten_on_interp(aux, children):
    obj = object.__new__(JaxLinearInterpOnInterp1D)
    for k, v in zip(_ON_INTERP_FIELDS, children):
        setattr(obj, k, v)
    obj.dtype = np.dtype(aux[0])
    obj.y_n = aux[1]
    return obj


class JaxLowerEnvelope2D(_Jax2D, LowerEnvelope2D):
    """
    ``LowerEnvelope2D`` with the JAX backend, constructed when its functions are
    all JAX-backed. Every function and every partial is evaluated at every
    query, and the envelope's are selected.
    """

    def __init__(self, *functions, nan_bool=True, backend=None, dtype=None):
        LowerEnvelope2D.__init__(self, *functions, nan_bool=nan_bool)
        self.nan_bool = nan_bool
        self.dtype = _component_dtype("LowerEnvelope2D", functions, dtype)

    def _values(self, x, y):
        return jnp.stack([f(x, y) for f in self.functions], axis=1)

    def _evaluate(self, x, y):
        return kernels.envelope_level(jnp, self._values(x, y), self.nan_bool)

    def _partial(self, x, y, attr):
        partials = jnp.stack([getattr(f, attr)(x, y) for f in self.functions], axis=1)
        return kernels.envelope_partial(
            jnp, self._values(x, y), partials, self.nan_bool
        )

    def _derX(self, x, y):
        return self._partial(x, y, "derivativeX")

    def _derY(self, x, y):
        return self._partial(x, y, "derivativeY")


def _flatten_envelope(obj):
    return (tuple(obj.functions),), (obj.nan_bool, obj.dtype.name)


def _unflatten_envelope(aux, children):
    obj = object.__new__(JaxLowerEnvelope2D)
    obj.functions = list(children[0])
    obj.funcCount = len(obj.functions)
    obj.nan_bool, dtype_name = aux
    obj.dtype = np.dtype(dtype_name)
    return obj


class JaxVariableLowerBoundFunc2D(_Jax2D, VariableLowerBoundFunc2D):
    """
    ``VariableLowerBoundFunc2D`` with the JAX backend, constructed when ``func``
    and ``lowerBound`` are both JAX-backed. Its derivatives are the numpy
    class's own composition of its components'.
    """

    def __init__(self, func, lowerBound, *, backend=None, dtype=None):
        VariableLowerBoundFunc2D.__init__(self, func, lowerBound)
        if not isinstance(lowerBound, JaxLinearInterp):
            _refuse(
                "VariableLowerBoundFunc2D",
                f"a lowerBound of type {type(lowerBound).__name__}",
            )
        self.dtype = _component_dtype(
            "VariableLowerBoundFunc2D", (func, lowerBound), dtype
        )

    def _evaluate(self, x, y):
        return self.func(x - self.lowerBound(y), y)


def _flatten_bound(obj):
    return (obj.func, obj.lowerBound), obj.dtype.name


def _unflatten_bound(aux, children):
    obj = object.__new__(JaxVariableLowerBoundFunc2D)
    obj.func, obj.lowerBound = children
    obj.dtype = np.dtype(aux)
    return obj


class JaxMargValueFuncCRRA(_JaxBacked, MargValueFuncCRRA):
    """``MargValueFuncCRRA`` with the JAX backend, constructed when ``cFunc`` is
    JAX-backed."""

    def __init__(self, cFunc, CRRA, *, backend=None, dtype=None):
        # A copy of cFunc's structure, as the numpy class's deepcopy, which also
        # works on traced arrays
        self.cFunc = jax.tree.map(lambda leaf: leaf, cFunc)
        self.CRRA = CRRA
        self.grid_list = getattr(cFunc, "grid_list", None)
        self.dtype = _component_dtype("MargValueFuncCRRA", (cFunc,), dtype)

    def __call__(self, *cFuncArgs):
        return kernels.crra_marg_utility(jnp, self.cFunc(*cFuncArgs), self.CRRA)

    def derivativeX(self, *cFuncArgs):
        c, MPC = _eval_c_and_mpc(self.cFunc, *cFuncArgs)
        return MPC * kernels.crra_marg_marg_utility(jnp, c, self.CRRA)


class JaxMargMargValueFuncCRRA(_JaxBacked, MargMargValueFuncCRRA):
    """``MargMargValueFuncCRRA`` with the JAX backend, constructed when ``cFunc``
    is JAX-backed."""

    def __init__(self, cFunc, CRRA, *, backend=None, dtype=None):
        # As in JaxMargValueFuncCRRA
        self.cFunc = jax.tree.map(lambda leaf: leaf, cFunc)
        self.CRRA = CRRA
        self.dtype = _component_dtype("MargMargValueFuncCRRA", (cFunc,), dtype)

    def __call__(self, *cFuncArgs):
        c, MPC = _eval_c_and_mpc(self.cFunc, *cFuncArgs)
        return MPC * kernels.crra_marg_marg_utility(jnp, c, self.CRRA)


def _crra_pytree(cls):
    def flatten(obj):
        return (obj.cFunc,), (float(obj.CRRA), obj.dtype.name)

    def unflatten(aux, children):
        obj = object.__new__(cls)
        (obj.cFunc,) = children
        obj.CRRA = aux[0]
        obj.dtype = np.dtype(aux[1])
        obj.grid_list = getattr(obj.cFunc, "grid_list", None)
        return obj

    register_pytree(cls, flatten, unflatten)


register_pytree(JaxLinearInterp, _flatten_linear, _unflatten_linear)
register_pytree(JaxBilinearInterp, _flatten_bilinear, _unflatten_bilinear)
register_pytree(JaxLinearInterpOnInterp1D, _flatten_on_interp, _unflatten_on_interp)
register_pytree(JaxLowerEnvelope2D, _flatten_envelope, _unflatten_envelope)
register_pytree(JaxVariableLowerBoundFunc2D, _flatten_bound, _unflatten_bound)
_crra_pytree(JaxMargValueFuncCRRA)
_crra_pytree(JaxMargMargValueFuncCRRA)

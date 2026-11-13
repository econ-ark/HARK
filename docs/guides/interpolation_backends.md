# Interpolation Backends: numpy and JAX

`LinearInterp` and `BilinearInterp` can compute with [JAX](https://docs.jax.dev) instead of numpy. The JAX backend is optional: HARK never imports JAX unless it is asked for, so a default install, including the browser one, is unaffected. More interpolators will follow the same pattern.

## What it is for

A JAX-backed interpolator can be passed into code compiled with `jax.jit` and called there, and it works with `jax.vmap` and `jax.grad`. The backend is meant for that kind of code: solvers and simulators compiled with JAX, batches of many functions or agents, and GPUs.

It is not meant as a switch for code that is not compiled. Called from plain Python, every evaluation goes through JAX's dispatch, so an existing numpy solver should keep the numpy backend.

## Installing

```
pip install "econ-ark[jax]"
```

For a GPU build of JAX, follow JAX's installation instructions.

## Choosing the backend

Pass `backend` and `dtype` to the constructor:

```python
import jax
from HARK.interpolation import LinearInterp

cFunc = LinearInterp(m_grid, c_grid, backend="jax", dtype="float32")
c = jax.jit(lambda f, m: f(m))(cFunc, m_points)
```

Or set the default, for the whole process or for a block of code:

```python
from HARK.interpolation import set_backend, using_backend

set_backend("jax", dtype="float32")

with using_backend("jax", dtype="float64"):
    ...
```

`get_backend()` reports the default in effect. An explicit `backend=` argument always wins. Worker processes start with the numpy default.

A JAX-backed interpolator is an instance of its numpy class (`isinstance(cFunc, LinearInterp)` is true), returns JAX arrays, and can be copied and pickled. It can also be constructed inside `jax.jit` and `jax.vmap` code, from traced arrays, with no round trip to the host. The one exception is `LinearInterp`'s decay extrapolation (`intercept_limit` and `slope_limit`): whether it applies depends on the values, so it needs concrete grids and is refused inside a transformation.

## Precision

The JAX backend computes in `float32` or `float64`, chosen with `dtype`. 32-bit takes half the memory and bandwidth. The numpy backend computes in `float64` only.

JAX computes in 32-bit unless its 64-bit mode is on. That is a process-wide setting, made before any JAX arrays are created:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

or the environment variable `JAX_ENABLE_X64=1`. HARK never changes it for you, since it affects every JAX computation in the process. If you ask for `float64` while it is off, HARK raises an error rather than silently computing in `float32`. Use one precision within a compiled kernel, since mixing them promotes to the wider one.

What the tests guarantee, against the numpy backend on the same inputs:

- `float64`: levels within `1e-14` of their scale.
- `float32`: levels within `1e-6` of their scale.
- Derivatives: a derivative is a difference of nearby values divided by a grid cell's width, so its rounding error is about machine epsilon times the size of the values over the width of the narrowest cell. The tests allow 16 epsilons of that. In `float32`, derivatives on fine grids are therefore much less precise than levels.

Two consequences of `float32` for solvers: knots closer together than about `1e-7` of their magnitude stop being distinct, and an iteration cannot converge to a tolerance much below `1e-6`.

## What is supported

| class | JAX backend |
|---|---|
| `LinearInterp` | level, `derivative`, `eval_with_derivative`; `lower_extrap`; decay to a limiting linear function (`intercept_limit` and `slope_limit`) |
| `BilinearInterp` | level, `derivativeX`, `derivativeY` |

An option the JAX backend does not implement is refused with `NotImplementedError`, never ignored: `pre_compute` and a custom `indexer` on `LinearInterp`, and `backend="jax"` on a subclass of either class.

## A caution about processes

JAX is not safe across `fork`. A program that forks worker processes after JAX has started can deadlock. Start workers with the `spawn` method, or before JAX is first used.

## For developers: one implementation

The JAX backend evaluates with kernels in `HARK._interp_kernels`, written once for both array libraries: each takes `numpy` or `jax.numpy` as an argument and keeps the order of operations of the numpy class it mirrors. `tests/test_interp_kernels.py` checks that, run with numpy, every kernel reproduces its numpy class bit for bit. While that check passes, the numpy classes could evaluate with the same kernels, leaving one implementation of each formula. That step waits on a benchmark of the numpy path, since the kernels compute over whole arrays where the numpy classes patch only the entries that need it.

"""Optional JAX, imported only when something asks for HARK's JAX backend.

This is the second optional backend after ``HARK._numba``, and the opposite case:
``HARK.core`` needs numba's ``njit`` at import, so that module guards the import and
supplies a fallback, whereas nothing in core HARK needs JAX. So this module imports
nothing at load time. ``require_jax`` imports JAX the first time a JAX-backed object
is constructed, and ``import HARK`` never touches it, which keeps the browser install
(#1806) free of it.

Precision. JAX computes in 32-bit unless its process-wide ``jax_enable_x64`` flag is
set, and the flag has to be set before any JAX arrays exist. HARK never sets it on
the user's behalf, since that would change every other JAX computation in the
process. ``resolve_dtype`` therefore refuses float64 while the flag is off, rather
than silently computing in float32.
"""

import numpy as np

_INSTALL_HINT = (
    "HARK's JAX backend requires JAX. Install it with: pip install 'econ-ark[jax]' "
    "(or pip install jax; see the JAX documentation for GPU builds)."
)

_registered = set()


def require_jax():
    """Import JAX and return ``(jax, jax.numpy)``, or raise ImportError with install
    instructions."""
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return jax, jnp


def resolve_dtype(dtype):
    """Return the numpy dtype a JAX-backed object computes in.

    ``dtype`` is ``"float32"`` or ``"float64"`` (or the matching numpy dtype);
    ``None`` means float64. float64 is refused while JAX's x64 mode is off.
    """
    jax, _ = require_jax()
    name = np.dtype("float64" if dtype is None else dtype).name
    if name not in ("float32", "float64"):
        raise ValueError(f"dtype must be float32 or float64, not {name}.")
    if name == "float64" and not jax.config.jax_enable_x64:
        raise ValueError(
            "dtype float64 needs JAX's 64-bit mode, which is off. Set the environment "
            "variable JAX_ENABLE_X64=1 before starting Python, or call "
            "jax.config.update('jax_enable_x64', True) before creating any JAX arrays. "
            "Or pass dtype='float32'."
        )
    return np.dtype(name)


def register_pytree(cls, flatten, unflatten):
    """Register ``cls`` as a JAX pytree, once, so its instances can be passed into
    ``jax.jit``, ``jax.vmap`` and other JAX transformations."""
    if cls in _registered:
        return
    jax, _ = require_jax()
    jax.tree_util.register_pytree_node(cls, flatten, unflatten)
    _registered.add(cls)

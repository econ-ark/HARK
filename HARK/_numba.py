"""Optional numba, so that HARK imports on targets where numba has no build.

``HARK.core`` reaches ``numba`` through ``HARK.simulator``, ``HARK.SSJutils`` and
``HARK.utilities``, so without this module ``import HARK`` fails outright wherever
numba is missing. That is the situation on stock Pyodide and PyScript, whose package
set has neither ``numba`` nor ``llvmlite``, and it is also what a minimal install
without the compiled stack looks like.

Where numba is present, including JupyterLite with the emscripten-forge xeus-python
kernel, the real ``njit`` and ``prange`` are re-exported and behaviour is unchanged.

The fallback ``njit`` has to cover every calling convention HARK uses, because numba
spells three different things with one name:

    @njit                                       bare decorator
    @njit(cache=True)                           decorator factory
    @njit(parallel=True)                        decorator factory
    @njit("float64(float64[:])", cache=True)    factory with a signature string
    CRRAutility = njit(CRRAutility_X, cache=True)   direct call on a function

The last form is the dangerous one and it is live in ``HARK.numba_tools``. A guard
that only returns the function for a callable first argument *without* keywords will
hand back a decorator instead, so ``CRRAutility(c, rho)`` silently returns ``c``
rather than raising. The test is therefore whether the first argument is callable,
independent of the keywords. A signature string is not callable, so that form still
reaches the factory branch, which is what it needs.

``prange`` falls back to ``range``. numba's ``prange`` is ``range`` plus a
parallelism hint, and the loops HARK uses it in are order independent, so serial
execution is correct and only slower.
"""

from __future__ import annotations

try:
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:  # pragma: no cover - exercised only where numba is absent
    HAS_NUMBA = False

    prange = range

    def njit(*args, **kwargs):
        """No-op stand-in for ``numba.njit`` that runs the pure-Python function."""
        if args and callable(args[0]):
            # @njit, njit(func), and njit(func, cache=True)
            return args[0]
        # @njit(...), including a leading signature string
        return lambda func: func


__all__ = ["HAS_NUMBA", "njit", "prange"]

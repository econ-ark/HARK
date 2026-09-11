"""
This file implements unit tests for the optional-numba shim in HARK._numba.

Two halves. The first exercises the fallback `njit` directly, by constructing it
the way `HARK/_numba.py` does when numba is missing, so the calling conventions
are covered on every run including CI where numba is installed. The second blocks
numba with a meta-path finder and imports HARK in a subprocess, which is the only
way to see the branch the browser actually takes.
"""

import importlib.util
import inspect
import subprocess
import sys
import textwrap
import unittest

from HARK import _numba
from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    PerfForesightConsumerType,
)


def _fallback_njit(*args, **kwargs):
    """The no-op njit from HARK/_numba.py, duplicated so it can be tested directly.

    Kept in sync by test_fallback_matches_shim_source below.
    """
    if args and callable(args[0]):
        return args[0]
    return lambda func: func


def _double(x):
    return 2 * x


class TestFallbackNjitConventions(unittest.TestCase):
    """Every calling convention HARK uses must return a working function."""

    def test_bare_decorator(self):
        self.assertEqual(_fallback_njit(_double)(21), 42)

    def test_direct_call_with_kwargs(self):
        # numba_tools.py builds CRRAutility and six relatives this way. A guard
        # that also required "not kwargs" returns a decorator here, so the call
        # would yield its own argument instead of raising.
        self.assertEqual(_fallback_njit(_double, cache=True)(21), 42)

    def test_direct_call_with_several_kwargs(self):
        fn = _fallback_njit(_double, cache=True, error_model="numpy")
        self.assertEqual(fn(21), 42)

    def test_decorator_factory(self):
        self.assertEqual(_fallback_njit(cache=True)(_double)(21), 42)

    def test_decorator_factory_parallel(self):
        self.assertEqual(_fallback_njit(parallel=True)(_double)(21), 42)

    def test_signature_string_is_not_treated_as_a_function(self):
        fn = _fallback_njit("float64(float64)", cache=True)(_double)
        self.assertEqual(fn(21), 42)


class TestShimExports(unittest.TestCase):
    def test_exports_are_present(self):
        for name in ("njit", "prange", "HAS_NUMBA"):
            self.assertTrue(hasattr(_numba, name), f"HARK._numba lacks {name}")

    def test_has_numba_matches_reality(self):
        expected = importlib.util.find_spec("numba") is not None
        self.assertEqual(_numba.HAS_NUMBA, expected)

    def test_fallback_matches_shim_source(self):
        """The duplicated fallback above must match HARK/_numba.py itself."""
        source = inspect.getsource(_numba)
        self.assertIn("if args and callable(args[0]):", source)
        self.assertIn("return lambda func: func", source)


class TestWithNumbaBlocked(unittest.TestCase):
    """Import HARK in a subprocess with numba unavailable.

    A subprocess is required because the check purges numba and HARK from
    sys.modules; it cannot share an interpreter with the rest of the suite.
    """

    SCRIPT = textwrap.dedent(
        """
        import sys

        class Blocker:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "numba" or fullname.startswith("numba."):
                    # ModuleNotFoundError with .name set, matching what the
                    # interpreter raises. Code that narrows on exc.name sees
                    # the same thing here as on a platform without numba.
                    raise ModuleNotFoundError(
                        f"No module named {fullname!r}", name=fullname
                    )
                return None

        for name in [m for m in sys.modules
                     if m == "numba" or m.startswith(("numba.", "HARK"))]:
            del sys.modules[name]
        sys.meta_path.insert(0, Blocker())

        # Rejection test: find_module was removed in 3.12, so a blocker written
        # against it is inert and every assertion below would pass for nothing.
        try:
            import numba
        except ImportError:
            pass
        else:
            print("BLOCKER-INERT")
            raise SystemExit(2)

        from HARK import _numba
        assert _numba.HAS_NUMBA is False, "HAS_NUMBA true while numba is blocked"
        assert _numba.prange is range, "prange did not fall back to range"

        from HARK.ConsumptionSaving.ConsIndShockModel import (
            IndShockConsumerType,
            PerfForesightConsumerType,
        )
        pf = PerfForesightConsumerType()
        pf.solve()
        ind = IndShockConsumerType()
        ind.solve()
        print(repr(float(pf.solution[0].cFunc(1.0))))
        print(repr(float(ind.solution[0].cFunc(1.0))))
        """
    )

    def test_hark_imports_and_solves_without_numba(self):
        proc = subprocess.run(
            [sys.executable, "-c", self.SCRIPT],
            capture_output=True,
            text=True,
        )
        self.assertNotIn("BLOCKER-INERT", proc.stdout, "the numba blocker did nothing")
        self.assertEqual(proc.returncode, 0, f"stderr:\n{proc.stderr[-2000:]}")

        without = [float(line) for line in proc.stdout.split()]

        pf = PerfForesightConsumerType()
        pf.solve()
        ind = IndShockConsumerType()
        ind.solve()
        with_numba = [
            float(pf.solution[0].cFunc(1.0)),
            float(ind.solution[0].cFunc(1.0)),
        ]

        # Exact equality: the fallback runs the same pure-Python code numba
        # compiles, so any difference is a real divergence rather than rounding.
        self.assertEqual(without, with_numba)


if __name__ == "__main__":
    unittest.main()

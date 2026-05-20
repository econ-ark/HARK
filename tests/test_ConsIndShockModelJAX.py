"""
Parity tests: ``ConsIndShockModelJAX`` vs ``ConsIndShockModel``.

Each test solves the same problem with both the numpy/numba solver and the
JAX solver, then compares the resulting ``cFunc`` element-wise on a query
grid. The JAX kernel performs EGM on the same a-grid and lifts the next-
period cFunc to a 1000-point tabulation, so we expect ~1e-4 relative
agreement — tighter than EGM truncation noise but not bit-precision.

Skipped if JAX is not installed.
"""

import os
import unittest

import numpy as np

try:
    os.environ.setdefault("JAX_ENABLE_X64", "True")
    import jax

    jax.config.update("jax_enable_x64", True)
    from HARK.ConsumptionSaving.ConsIndShockModelJAX import (
        IndShockConsumerTypeJAX,
    )

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    init_idiosyncratic_shocks,
)


# Tolerance: 1e-4 relative. EGM solver truncation is typically O(1e-6) on
# the grid; the lift-to-tabulation step adds O(1e-5) interpolation error on
# top. 1e-4 is comfortably above both.
PARITY_RTOL = 1e-4


def _params_basic_infinite():
    """Standard infinite-horizon IndShock parameters from HARK defaults."""
    params = dict(init_idiosyncratic_shocks)
    params["cycles"] = 0
    params["vFuncBool"] = False
    params["CubicBool"] = False
    return params


def _params_basic_finite_horizon():
    """Finite horizon (cycles>0) variant of the basic problem.

    Uses T_cycle=1 with cycles=10 to avoid needing a hand-constructed
    per-period IncShkDstn list — HARK's default constructor builds a single
    IncShkDstn that the solver then iterates backward over ``cycles`` times.
    """
    params = dict(init_idiosyncratic_shocks)
    params["cycles"] = 10
    params["vFuncBool"] = False
    params["CubicBool"] = False
    return params


@unittest.skipUnless(HAS_JAX, "JAX not installed")
class TestConsIndShockModelJAXParity(unittest.TestCase):
    """Numerical parity between JAX and numpy solvers on standard params."""

    def test_infinite_horizon_cfunc(self):
        """Steady-state cFunc matches between JAX and numpy solvers."""
        params = _params_basic_infinite()
        agent_np = IndShockConsumerType(**params)
        agent_jax = IndShockConsumerTypeJAX(**params)
        agent_np.solve()
        agent_jax.solve()
        m_query = np.linspace(1.0, 20.0, 50)
        c_np = agent_np.solution[0].cFunc(m_query)
        c_jax = np.asarray(agent_jax.solution[0].cFunc(m_query))
        np.testing.assert_allclose(c_jax, c_np, rtol=PARITY_RTOL)

    def test_infinite_horizon_summary_scalars(self):
        """Per-period scalar summaries (hNrm, MPCmin, MPCmax) match."""
        params = _params_basic_infinite()
        agent_np = IndShockConsumerType(**params)
        agent_jax = IndShockConsumerTypeJAX(**params)
        agent_np.solve()
        agent_jax.solve()
        sol_np = agent_np.solution[0]
        sol_jax = agent_jax.solution[0]
        np.testing.assert_allclose(sol_jax.hNrm, sol_np.hNrm, rtol=PARITY_RTOL)
        np.testing.assert_allclose(sol_jax.MPCmin, sol_np.MPCmin, rtol=PARITY_RTOL)
        np.testing.assert_allclose(sol_jax.MPCmax, sol_np.MPCmax, rtol=PARITY_RTOL)

    def test_finite_horizon_cfunc_per_period(self):
        """Finite horizon (cycles=10, T_cycle=1): cFunc matches at every period."""
        params = _params_basic_finite_horizon()
        agent_np = IndShockConsumerType(**params)
        agent_jax = IndShockConsumerTypeJAX(**params)
        agent_np.solve()
        agent_jax.solve()
        m_query = np.linspace(1.0, 10.0, 30)
        for t in range(len(agent_np.solution)):
            c_np = agent_np.solution[t].cFunc(m_query)
            c_jax = np.asarray(agent_jax.solution[t].cFunc(m_query))
            np.testing.assert_allclose(
                c_jax,
                c_np,
                rtol=PARITY_RTOL,
                err_msg=f"cFunc mismatch at period t={t}",
            )

    def test_marginal_value_matches(self):
        """vPfunc(m) = cFunc(m)^(-CRRA) should match between solvers."""
        params = _params_basic_infinite()
        agent_np = IndShockConsumerType(**params)
        agent_jax = IndShockConsumerTypeJAX(**params)
        agent_np.solve()
        agent_jax.solve()
        m_query = np.linspace(1.5, 15.0, 40)
        vP_np = agent_np.solution[0].vPfunc(m_query)
        vP_jax = np.asarray(agent_jax.solution[0].vPfunc(m_query))
        np.testing.assert_allclose(vP_jax, vP_np, rtol=PARITY_RTOL)


@unittest.skipUnless(HAS_JAX, "JAX not installed")
class TestConsIndShockModelJAXNotYetSupported(unittest.TestCase):
    """Flags not yet supported by the JAX solver raise NotImplementedError."""

    def test_cubic_bool_raises(self):
        params = _params_basic_infinite()
        params["CubicBool"] = True
        agent = IndShockConsumerTypeJAX(**params)
        with self.assertRaises(NotImplementedError):
            agent.solve()

    def test_vfunc_bool_raises(self):
        params = _params_basic_infinite()
        params["vFuncBool"] = True
        agent = IndShockConsumerTypeJAX(**params)
        with self.assertRaises(NotImplementedError):
            agent.solve()


if __name__ == "__main__":
    unittest.main()

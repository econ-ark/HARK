"""
Parity tests: ``AggShockMarkovConsumerTypeJAX`` vs
``AggShockMarkovConsumerType``.

Both agents are set up identically inside a ``CobbDouglasMarkovEconomy``
(the standard test fixture from ``test_ConsAggShockModel.py``). After
solving, per-Markov-state ``cFunc`` is compared element-wise on a grid of
``(m, M)`` query points.

The JAX solver lifts ``vPfuncNext`` to a 256 × 64 bilinear tabulation each
iteration; combined with bilinear interp in the kernel, expected relative
agreement is ~1e-3 — looser than the 1-D case in PR-2 because the 2-D lift
adds error in both axes.

Skipped if JAX is not installed.
"""

import os
import unittest

import numpy as np

try:
    os.environ.setdefault("JAX_ENABLE_X64", "True")
    import jax

    jax.config.update("jax_enable_x64", True)
    from HARK.ConsumptionSaving.ConsAggShockModelJAX import (
        AggShockMarkovConsumerTypeJAX,
    )

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

from HARK.ConsumptionSaving.ConsAggShockModel import (
    AggShockMarkovConsumerType,
    CobbDouglasMarkovEconomy,
)


PARITY_RTOL = 1e-3


@unittest.skipUnless(HAS_JAX, "JAX not installed")
class TestAggShockMarkovJAXParity(unittest.TestCase):
    """JAX solver matches numpy solver on the standard test fixture."""

    @classmethod
    def setUpClass(cls):
        """Build identical numpy and JAX agents inside parallel economies."""
        # Numpy reference
        cls.agent_np = AggShockMarkovConsumerType(cycles=0, AgentCount=1000, seed=0)
        cls.economy_np = CobbDouglasMarkovEconomy(agents=[cls.agent_np], seed=0)
        cls.economy_np.give_agent_params()
        cls.agent_np.solve()

        # JAX variant — same construction, different solver
        cls.agent_jax = AggShockMarkovConsumerTypeJAX(cycles=0, AgentCount=1000, seed=0)
        cls.economy_jax = CobbDouglasMarkovEconomy(agents=[cls.agent_jax], seed=0)
        cls.economy_jax.give_agent_params()
        cls.agent_jax.solve()

    def test_cfunc_state0_at_MSS(self):
        """First-Markov-state cFunc matches at the steady-state aggregate M."""
        m_query = 10.0
        M_query = self.economy_np.MSS
        c_np = float(self.agent_np.solution[0].cFunc[0](m_query, M_query))
        c_jax = float(self.agent_jax.solution[0].cFunc[0](m_query, M_query))
        self.assertAlmostEqual(c_jax, c_np, delta=PARITY_RTOL * abs(c_np) + 1e-6)

    def test_cfunc_grid_per_state(self):
        """Per-state cFunc matches on a grid of (m, M) query points."""
        m_query = np.linspace(2.0, 15.0, 20)
        M_query = np.linspace(
            0.5 * self.economy_np.MSS,
            2.0 * self.economy_np.MSS,
            5,
        )
        n_states = len(self.agent_np.solution[0].cFunc)
        for k in range(n_states):
            for M_val in M_query:
                c_np = np.asarray(
                    self.agent_np.solution[0].cFunc[k](
                        m_query, np.full_like(m_query, M_val)
                    )
                )
                c_jax = np.asarray(
                    self.agent_jax.solution[0].cFunc[k](
                        m_query, np.full_like(m_query, M_val)
                    )
                )
                np.testing.assert_allclose(
                    c_jax,
                    c_np,
                    rtol=PARITY_RTOL,
                    err_msg=f"cFunc mismatch at Markov state k={k}, M={M_val:.3f}",
                )

    def test_cfunc_state1_at_MSS(self):
        """Second Markov state cFunc at MSS — explicit (grid test covers this implicitly)."""
        m_query = 10.0
        M_query = self.economy_np.MSS
        c_np = float(self.agent_np.solution[0].cFunc[1](m_query, M_query))
        c_jax = float(self.agent_jax.solution[0].cFunc[1](m_query, M_query))
        self.assertAlmostEqual(c_jax, c_np, delta=PARITY_RTOL * abs(c_np) + 1e-6)

    def test_mNrmMin_per_state(self):
        """Per-state mNrmMin matches on a grid of M values."""
        M_query = np.linspace(
            0.5 * self.economy_np.MSS,
            2.0 * self.economy_np.MSS,
            10,
        )
        n_states = len(self.agent_np.solution[0].mNrmMin)
        for k in range(n_states):
            mNrmMin_np = np.asarray(
                [float(self.agent_np.solution[0].mNrmMin[k](M)) for M in M_query]
            )
            mNrmMin_jax = np.asarray(
                [float(self.agent_jax.solution[0].mNrmMin[k](M)) for M in M_query]
            )
            np.testing.assert_allclose(
                mNrmMin_jax,
                mNrmMin_np,
                rtol=PARITY_RTOL,
                atol=1e-6,
                err_msg=f"mNrmMin mismatch at state k={k}",
            )

    def test_cfunc_extreme_m_low(self):
        """Queries near the borrowing constraint (low m)."""
        n_states = len(self.agent_np.solution[0].cFunc)
        M_query = self.economy_np.MSS
        for k in range(n_states):
            mNrmMin_at_MSS = float(self.agent_np.solution[0].mNrmMin[k](M_query))
            # Queries just above the constraint
            m_query = np.linspace(mNrmMin_at_MSS + 0.05, mNrmMin_at_MSS + 1.0, 20)
            c_np = np.asarray(
                self.agent_np.solution[0].cFunc[k](
                    m_query, np.full_like(m_query, M_query)
                )
            )
            c_jax = np.asarray(
                self.agent_jax.solution[0].cFunc[k](
                    m_query, np.full_like(m_query, M_query)
                )
            )
            np.testing.assert_allclose(
                c_jax,
                c_np,
                rtol=PARITY_RTOL,
                err_msg=f"cFunc near constraint mismatch at state k={k}",
            )

    def test_cfunc_extreme_m_high(self):
        """Queries at high m (well above human-wealth scale)."""
        n_states = len(self.agent_np.solution[0].cFunc)
        M_query = self.economy_np.MSS
        m_query = np.linspace(50.0, 200.0, 20)
        for k in range(n_states):
            c_np = np.asarray(
                self.agent_np.solution[0].cFunc[k](
                    m_query, np.full_like(m_query, M_query)
                )
            )
            c_jax = np.asarray(
                self.agent_jax.solution[0].cFunc[k](
                    m_query, np.full_like(m_query, M_query)
                )
            )
            np.testing.assert_allclose(
                c_jax,
                c_np,
                rtol=PARITY_RTOL,
                err_msg=f"cFunc at high m mismatch at state k={k}",
            )

    def test_cfunc_extreme_M(self):
        """Queries at very low and very high aggregate M values."""
        n_states = len(self.agent_np.solution[0].cFunc)
        m_query = 10.0
        M_extreme = [0.3 * self.economy_np.MSS, 3.0 * self.economy_np.MSS]
        for k in range(n_states):
            for M_val in M_extreme:
                c_np = float(self.agent_np.solution[0].cFunc[k](m_query, M_val))
                c_jax = float(self.agent_jax.solution[0].cFunc[k](m_query, M_val))
                self.assertAlmostEqual(
                    c_jax,
                    c_np,
                    delta=PARITY_RTOL * abs(c_np) + 1e-6,
                    msg=f"extreme M mismatch at state k={k}, M={M_val:.3f}",
                )


if __name__ == "__main__":
    unittest.main()

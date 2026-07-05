"""Tests for the opt-in perfect-foresight decay extrapolation machinery in
ConsAggShockModel (pf_mpc_min, pf_human_wealth_markov, make_cFunc_slice) and a
real-solve truth test showing the power-law tail beats the exponential one.
"""

import unittest

import numpy as np

from HARK.ConsumptionSaving.ConsAggShockModel import (
    make_cFunc_slice,
    pf_human_wealth_markov,
    pf_mpc_min,
)
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
from HARK.interpolation import LinearInterp


class TestPFBoundHelpers(unittest.TestCase):
    def test_pf_mpc_min_closed_form(self):
        R, beta, rho, L = 1.03, 0.96, 2.0, 0.98
        expect = 1.0 - (R * beta * L) ** (1.0 / rho) / R
        self.assertAlmostEqual(pf_mpc_min(R, beta, rho, L), expect, places=14)
        self.assertGreater(pf_mpc_min(R, beta, rho, L), 0.0)

    def test_pf_mpc_min_warns_when_return_impatience_fails(self):
        with self.assertWarns(UserWarning):
            val = pf_mpc_min(1.0, 1.01, 2.0)
        self.assertLessEqual(val, 0.0)

    def test_human_wealth_scalar_matches_analytic(self):
        # Single state: h = (G/R)*E / (1 - G/R) = G*E/(R - G)
        R, G, E = 1.03, 1.01, 1.0
        h = pf_human_wealth_markov(np.array([[1.0]]), R, np.array([E]), np.array([G]))
        self.assertAlmostEqual(h[0], G * E / (R - G), places=10)

    def test_human_wealth_markov_joint_fixed_point(self):
        # Two states, the second with ZERO income (deep unemployment): the
        # joint solve must give it strictly positive human wealth (future
        # re-employment), where an own-state recursion would degenerate to 0.
        M = np.array([[0.9, 0.1], [0.2, 0.8]])
        R = 1.03
        E = np.array([1.0, 0.0])
        G = np.array([1.01, 1.01])
        h = pf_human_wealth_markov(M, R, E, G)
        self.assertTrue(np.all(np.isfinite(h)))
        self.assertGreater(h[1], 0.0)
        self.assertGreater(h[0], h[1])
        # Non-circular check: h satisfies the defining fixed point
        # h_i = sum_j M[i,j] * (G_j/R) * (E_j + h_j)
        resid = h - (M * (G / R)[None, :]) @ (E + h)
        np.testing.assert_allclose(resid, 0.0, atol=1e-12)

    def test_human_wealth_nan_when_fhwc_fails(self):
        # G >= R in every state: infinite human wealth -> all-NaN + warning
        M = np.array([[0.9, 0.1], [0.2, 0.8]])
        with self.assertWarns(UserWarning):
            h = pf_human_wealth_markov(
                M, 1.03, np.array([1.0, 1.0]), np.array([1.04, 1.04])
            )
        self.assertTrue(np.all(np.isnan(h)))


class TestMakeCFuncSlice(unittest.TestCase):
    MPCmin = 0.05
    hNrm = 20.0  # PF line 0.05*(m + 20) = 1 + 0.05*m

    def line(self, m):
        return self.MPCmin * (m + self.hNrm)

    def concave_below(self, m):
        # strictly below the line, slope falling toward MPCmin from above
        return self.line(m) - 2.0 * (m + self.hNrm) ** (-0.8)

    def test_legacy_when_bounds_missing(self):
        m = np.linspace(0.0, 30.0, 40)
        c = self.concave_below(m)
        f = make_cFunc_slice(m, c)
        self.assertIsInstance(f, LinearInterp)
        self.assertFalse(f.decay_extrap)
        f2 = make_cFunc_slice(m, c, 0.05, None)
        self.assertFalse(f2.decay_extrap)

    def test_decay_attaches_powerlaw_by_default(self):
        m = np.linspace(0.0, 30.0, 40)
        c = self.concave_below(m)
        f = make_cFunc_slice(m, c, self.MPCmin, self.hNrm)
        self.assertTrue(f.decay_extrap)
        self.assertEqual(f.decay_extrap_form, "powerlaw")
        self.assertEqual(f.slope_limit, self.MPCmin)
        self.assertAlmostEqual(f.intercept_limit, self.MPCmin * self.hNrm, places=14)
        # pivot = m_top + intercept/slope = m_top + hNrm
        self.assertAlmostEqual(f.decay_extrap_pivot, 30.0 + self.hNrm, places=10)

    def test_decay_form_exp_override(self):
        m = np.linspace(0.0, 30.0, 40)
        c = self.concave_below(m)
        f = make_cFunc_slice(m, c, self.MPCmin, self.hNrm, decay_form="exp")
        self.assertTrue(f.decay_extrap)
        self.assertEqual(f.decay_extrap_form, "exp")

    def test_concavity_guard_raises_on_impossible_input(self):
        # Top knot ABOVE the PF line while the top slope has already fallen to
        # MPCmin: impossible for a converged concave consumption function.
        m = np.linspace(0.0, 30.0, 40)
        c = self.line(m) + 0.1
        with self.assertRaises(ValueError):
            make_cFunc_slice(m, c, self.MPCmin, self.hNrm)

    def test_above_line_transient_falls_back_to_legacy(self):
        # Above the line but with slope still well above MPCmin: an ordinary
        # pre-asymptotic backward-induction transient -> legacy, no raise.
        m = np.linspace(0.0, 30.0, 40)
        c = self.line(m) + 0.1 + 0.2 * (m / 30.0) ** 2
        slope_top = (c[-1] - c[-2]) / (m[-1] - m[-2])
        self.assertGreater(slope_top, self.MPCmin + 1e-6)
        f = make_cFunc_slice(m, c, self.MPCmin, self.hNrm)
        self.assertFalse(f.decay_extrap)


class TestTruncatedGridTruthIndShock(unittest.TestCase):
    """Solve a default-calibration infinite-horizon IndShock model on a DEEP
    grid (the truth), truncate its consumption function at m=40, and compare
    exponential vs power-law decay extrapolation against the truth above the
    truncated grid. Measured baseline (this exact setup): the exponential has
    destroyed ~100% of the true PF gap by m=2000 (it returns the PF line while
    the truth is still 0.25% of c below it); the power law keeps ~82% of the
    gap; max error vs truth is ~9x smaller (1.8e-3 vs 1.6e-2 relative). The
    assertions below use ~2x safety margins on those measurements.
    """

    @classmethod
    def setUpClass(cls):
        agent = IndShockConsumerType(
            cycles=0, aXtraMax=1.0e5, aXtraCount=96, aXtraNestFac=3
        )
        agent.solve()
        sol = agent.solution[0]
        cls.cT = sol.cFunc
        cls.MPCmin = sol.MPCmin
        cls.hNrm = sol.hNrm
        cls.mMin = sol.mNrmMin
        m_knots = np.linspace(cls.mMin + 0.05, 40.0, 300)
        c_knots = cls.cT(m_knots)
        cls.f_exp = LinearInterp(
            m_knots, c_knots, cls.MPCmin * cls.hNrm, cls.MPCmin
        )
        cls.f_pl = LinearInterp(
            m_knots,
            c_knots,
            cls.MPCmin * cls.hNrm,
            cls.MPCmin,
            decay_extrap_form="powerlaw",
        )
        cls.lad = np.geomspace(50.0, 2000.0, 25)
        cls.c_true = cls.cT(cls.lad)
        cls.gap_true = cls.MPCmin * (cls.lad + cls.hNrm) - cls.c_true
        cls.err_exp = np.abs(cls.f_exp(cls.lad) - cls.c_true)
        cls.err_pl = np.abs(cls.f_pl(cls.lad) - cls.c_true)

    def test_setup_is_in_the_decaying_configuration(self):
        self.assertTrue(self.f_exp.decay_extrap)
        self.assertTrue(self.f_pl.decay_extrap)
        # the true gap at the ladder top is economically real, not float noise
        self.assertGreater(self.gap_true[-1], 0.1)
        # implied power-law exponent is a sane buffer-stock value
        self.assertGreater(self.f_pl.decay_extrap_Q, 0.0)
        self.assertLess(self.f_pl.decay_extrap_Q, 2.0)

    def test_powerlaw_beats_exponential_against_truth(self):
        # max relative error: measured pl 1.8e-3 vs exp 1.6e-2 (9x)
        self.assertLess(
            np.max(self.err_pl / self.c_true),
            0.5 * np.max(self.err_exp / self.c_true),
        )
        # pointwise: measured median ratio ~8.6x
        self.assertGreater(np.median(self.err_exp / self.err_pl), 3.0)

    def test_exponential_destroys_the_gap_powerlaw_keeps_it(self):
        # measured: err_exp/gap_true = 1.000 at m=2000 (exp -> the PF line),
        # err_pl/gap_true = 0.18
        self.assertGreater(self.err_exp[-1] / self.gap_true[-1], 0.9)
        self.assertLess(self.err_pl[-1] / self.gap_true[-1], 0.5)

"""Tests for the opt-in perfect-foresight decay extrapolation machinery in
ConsAggShockModel (pf_mpc_min, pf_human_wealth_markov, make_cFunc_slice) and a
real-solve truth test showing the power-law tail beats the exponential one.
"""

import unittest
import warnings

import numpy as np

from HARK.ConsumptionSaving.ConsAggShockModel import (
    AggShockConsumerType,
    CobbDouglasEconomy,
    make_cFunc_slice,
    pf_human_wealth_markov,
    pf_mpc_min,
)
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
from HARK.ConsumptionSaving.pf_decay import PFDecayGridWarning, powerlaw_decay_params
from HARK.interpolation import LinearInterp
from tests.ConsumptionSaving.test_pf_decay import _params


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


class TestMakeCFuncSliceTheoryModes(unittest.TestCase):
    """The decay_theory/decay_Q policy of make_cFunc_slice, mode by mode.

    Pre-registered semantics (the orchestrated default policy):
      (1) decay_theory=None            -> byte-identical prior PR behavior;
      (2) decay_theory + default       -> theory exponent min(1, q*), explicit,
                                          level-matched;
      (3) decay_theory + decay_Q=None  -> fitted exponent clamped to min(1, q*)
                                          (PFDecayGridWarning when the clamp
                                          bites; inert otherwise);
      (4) decay_Q=('amplitude', B)     -> REMOVED (ValueError): level
                                          continuity at the top knot is an
                                          invariant (design ruling 2026-07-11);
      (5) theory rescues the fitted form's no-decay fallback (slope_top <=
          MPCmin with the knot below the line).
    """

    MPCmin = 0.05
    hNrm = 20.0  # PF line 0.05*(m + 20) = 1 + 0.05*m

    @classmethod
    def setUpClass(cls):
        cls.theory_lo = _params("HS")    # q* = 0.3813 < 1
        cls.theory_hi = _params("CCAP")  # q* = 1.4735 > 1, B_psi = 356.63
        assert 0.0 < cls.theory_lo.q_star < 1.0
        assert cls.theory_hi.q_star > 1.0 and cls.theory_hi.B_psi is not None

    def line(self, m):
        return self.MPCmin * (m + self.hNrm)

    def concave_below(self, m, expo=0.8):
        # strictly below the line, slope falling toward MPCmin from above;
        # fitted exponent at the top knot is ~expo
        return self.line(m) - 2.0 * (m + self.hNrm) ** (-expo)

    def knots(self, expo=0.8):
        m = np.linspace(0.0, 30.0, 40)
        return m, self.concave_below(m, expo)

    # ---- (1) decay_theory=None: byte-identical prior behavior -------------
    def test_no_theory_is_byte_identical_slicewise(self):
        probe = np.concatenate([np.linspace(0.5, 29.5, 30),
                                np.geomspace(31.0, 3000.0, 25)])
        # healthy knot: prior behavior is the fitted powerlaw attach
        m, c = self.knots()
        f_new = make_cFunc_slice(m, c, self.MPCmin, self.hNrm)
        f_old = LinearInterp(m, c, self.MPCmin * self.hNrm, self.MPCmin,
                             decay_extrap_form="powerlaw")
        np.testing.assert_array_equal(f_new(probe), f_old(probe))
        self.assertFalse(hasattr(f_new, "decay_theory"))
        # above-line transient: prior behavior is the bare fallback
        c_above = self.line(m) + 0.1 + 0.2 * (m / 30.0) ** 2
        f_new = make_cFunc_slice(m, c_above, self.MPCmin, self.hNrm)
        f_old = LinearInterp(m, c_above)
        self.assertFalse(f_new.decay_extrap)
        np.testing.assert_array_equal(f_new(probe), f_old(probe))
        # Carroll-Kimball impossible knot: still raises
        with self.assertRaises(ValueError):
            make_cFunc_slice(m, self.line(m) + 0.1, self.MPCmin, self.hNrm)

    # ---- (2) theory default: explicit min(1, q*) --------------------------
    def test_theory_default_attaches_theory_exponent(self):
        m, c = self.knots()
        f = make_cFunc_slice(m, c, self.MPCmin, self.hNrm,
                             decay_theory=self.theory_lo)
        self.assertTrue(f.decay_extrap)
        self.assertEqual(f.decay_extrap_Q_source, "explicit")
        self.assertEqual(f.decay_extrap_Q, self.theory_lo.q)
        # level-matched at the top knot
        eps = 1e-9
        lvl = float(f(np.array([m[-1] + eps]))[0])
        self.assertAlmostEqual(lvl, c[-1], places=7)
        # metadata dict attached
        meta = f.decay_theory
        self.assertEqual(meta["Q_used"], self.theory_lo.q)
        self.assertEqual(meta["q_star"], self.theory_lo.q_star)
        self.assertAlmostEqual(meta["Q_fit"], 0.8, delta=0.05)
        self.assertIsNone(meta["B_psi"])
        # q* > 1: the realized exponent is capped at 1
        f_hi = make_cFunc_slice(m, c, self.MPCmin, self.hNrm,
                                decay_theory=self.theory_hi)
        self.assertEqual(f_hi.decay_extrap_Q, 1.0)

    # ---- (3) guarded fit --------------------------------------------------
    def test_guarded_fit_inert_when_fit_below_ceiling(self):
        # fitted exponent ~0.8 < ceiling 1.0 (q* > 1): the clamp must be inert
        # and the attach identical to the plain fitted attach
        m, c = self.knots()
        probe = np.geomspace(31.0, 3000.0, 25)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            f = make_cFunc_slice(m, c, self.MPCmin, self.hNrm,
                                 decay_theory=self.theory_hi, decay_Q=None)
        self.assertFalse(
            any(issubclass(w.category, PFDecayGridWarning) for w in caught)
        )
        self.assertEqual(f.decay_extrap_Q_source, "fitted")
        f_plain = LinearInterp(m, c, self.MPCmin * self.hNrm, self.MPCmin,
                               decay_extrap_form="powerlaw")
        np.testing.assert_array_equal(f(probe), f_plain(probe))
        self.assertAlmostEqual(f.decay_theory["Q_used"], f_plain.decay_extrap_Q,
                               places=14)

    def test_guarded_fit_clamps_and_warns(self):
        m, c = self.knots()
        # (a) ceiling = q* < 1 bites on a fitted exponent ~0.8
        with self.assertWarns(PFDecayGridWarning):
            f = make_cFunc_slice(m, c, self.MPCmin, self.hNrm,
                                 decay_theory=self.theory_lo, decay_Q=None)
        self.assertEqual(f.decay_extrap_Q, self.theory_lo.q)
        self.assertEqual(f.decay_extrap_Q_source, "explicit")
        # (b) synthetic Q_fit ~1.5 > 1 knot: the Prop-A0 hard cap (ceiling 1.0
        # from the q* > 1 theory) bites
        m2, c2 = self.knots(expo=1.5)
        with self.assertWarns(PFDecayGridWarning):
            f2 = make_cFunc_slice(m2, c2, self.MPCmin, self.hNrm,
                                  decay_theory=self.theory_hi, decay_Q=None)
        self.assertEqual(f2.decay_extrap_Q, 1.0)

    # ---- explicit float ----------------------------------------------------
    def test_explicit_float_passthrough(self):
        m, c = self.knots()
        f = make_cFunc_slice(m, c, self.MPCmin, self.hNrm, decay_Q=0.6)
        self.assertTrue(f.decay_extrap)
        self.assertEqual(f.decay_extrap_Q, 0.6)
        self.assertEqual(f.decay_extrap_Q_source, "explicit")
        self.assertFalse(hasattr(f, "decay_theory"))  # no theory metadata

    # ---- (5) rescue of the no-decay fallback -------------------------------
    def test_rescue_where_fit_disables_decay(self):
        # below the line but top slope BELOW MPCmin: the fitted form must fall
        # back to naive-linear (no decay); theory attaches the explicit tail
        m = np.linspace(0.0, 30.0, 40)
        c = self.line(m) - 0.1 * (m + self.hNrm) ** 0.5
        slope_top = (c[-1] - c[-2]) / (m[-1] - m[-2])
        self.assertLess(slope_top, self.MPCmin)
        f_legacy = make_cFunc_slice(m, c, self.MPCmin, self.hNrm)
        self.assertFalse(f_legacy.decay_extrap)
        f_theory = make_cFunc_slice(m, c, self.MPCmin, self.hNrm,
                                    decay_theory=self.theory_lo)
        self.assertTrue(f_theory.decay_extrap)
        self.assertEqual(f_theory.decay_extrap_Q, self.theory_lo.q)
        # default rescue is the C1 two-term attachment: it extends the body
        # SMOOTHLY (matching its still-widening gap at the knot: an interior
        # gap maximum, not immediate decay) and only then decays to the line
        self.assertEqual(f_theory.decay_extrap_terms, 2)
        self.assertEqual(f_theory.decay_theory["terms"], 2)
        slope_top = (c[-1] - c[-2]) / (m[-1] - m[-2])
        d_above = float(f_theory.derivative(np.array([m[-1] + 1e-11]))[0])
        self.assertLess(abs(d_above - slope_top), 1e-9)  # C1, no kink
        lad = np.geomspace(50.0, 5000.0, 20)
        gap = self.line(lad) - f_theory(lad)
        gap_top = self.line(m[-1]) - c[-1]
        self.assertTrue(np.all(gap > 0.0))
        self.assertLess(gap[-1], gap_top)  # eventually decays below the knot gap
        gap_legacy = self.line(lad) - f_legacy(lad)
        self.assertGreater(gap_legacy[-1], gap[-1])
        # the ONE-TERM rescue keeps its original registrations: immediate
        # monotone decay from a level-matched knot (with the documented kink)
        f_one = make_cFunc_slice(m, c, self.MPCmin, self.hNrm,
                                 decay_theory=self.theory_lo, decay_terms=1)
        gap1 = self.line(lad) - f_one(lad)
        self.assertTrue(np.all(gap1 > 0.0))
        self.assertTrue(np.all(np.diff(gap1) < 0.0))
        self.assertLess(gap1[0], gap_top)
        # guarded fit rescues with the same ceiling exponent
        f_guard = make_cFunc_slice(m, c, self.MPCmin, self.hNrm,
                                   decay_theory=self.theory_lo, decay_Q=None)
        self.assertEqual(f_guard.decay_extrap_Q, self.theory_lo.q)

    # ---- (4) amplitude mode: REMOVED (continuity invariant) -----------------
    def test_amplitude_mode_removed_raises(self):
        # Design ruling 2026-07-11: level continuity at the top knot is an
        # INVARIANT of the decay machinery -- the former ('amplitude', B)
        # mode's guarded level jump is never attachable. The tuple form now
        # raises, pointing callers at decay_Q=1.0; that level-matched
        # exponent-1 tail remains available and jump-free.
        B0 = 25.0
        m = np.linspace(0.0, 30.0, 40)
        c = self.line(m) - B0 / (m + self.hNrm)
        with self.assertRaises(ValueError):
            make_cFunc_slice(m, c, self.MPCmin, self.hNrm,
                             decay_theory=self.theory_hi,
                             decay_Q=("amplitude", B0))
        f = make_cFunc_slice(m, c, self.MPCmin, self.hNrm,
                             decay_theory=self.theory_hi, decay_Q=1.0)
        self.assertTrue(f.decay_extrap)
        self.assertEqual(f.decay_extrap_Q_source, "explicit")
        self.assertEqual(f.decay_extrap_Q, 1.0)
        # level-matched (no jump): the tail limit at the top knot is the
        # solved value there
        lvl = float(f(np.array([m[-1] + 1e-9]))[0])
        self.assertAlmostEqual(lvl, c[-1], places=7)

    # ---- validation and refusal paths --------------------------------------
    def test_invalid_decay_Q_raises(self):
        m, c = self.knots()
        with self.assertRaises(ValueError):
            make_cFunc_slice(m, c, self.MPCmin, self.hNrm, decay_Q="bogus")
        # refuter finding (B3): decay_terms must be validated even on the
        # legacy early-return path (MPCmin/hNrm None)
        with self.assertRaises(ValueError):
            make_cFunc_slice(m, c, None, None, decay_terms=37)
        with self.assertRaises(ValueError):
            make_cFunc_slice(m, c, self.MPCmin, self.hNrm, decay_terms=True)
        with self.assertRaises(ValueError):
            make_cFunc_slice(m, c, self.MPCmin, self.hNrm,
                             decay_Q=("amplitude", -1.0))
        with self.assertRaises(ValueError):
            make_cFunc_slice(m, c, self.MPCmin, self.hNrm,
                             decay_Q=("amplitude",))
        with self.assertRaises(ValueError):
            make_cFunc_slice(m, c, self.MPCmin, self.hNrm,
                             decay_theory=self.theory_lo, decay_form="exp")

    def test_theory_with_nan_qstar_falls_back_to_guarded_fit(self):
        # FHWC-violated theory: q* is nan, so the theory default degrades to
        # the guarded fit with the GIC-free Prop-A0 ceiling 1.0 (inert here)
        theory_nan = powerlaw_decay_params(
            1.0, 1.005, 0.98, 2.0, LivPrb=0.99,
            PermShkDstn=None, TranShkDstn=(np.array([0.7, 1.3]),
                                           np.array([0.5, 0.5])),
            warn=False,
        )
        self.assertTrue(np.isnan(theory_nan.q_star))
        m, c = self.knots()
        f = make_cFunc_slice(m, c, self.MPCmin, self.hNrm,
                             decay_theory=theory_nan)
        self.assertTrue(f.decay_extrap)
        self.assertEqual(f.decay_extrap_Q_source, "fitted")
        self.assertTrue(np.isnan(f.decay_theory["q_star"]))

    def test_amplitude_ratio_logged_once_per_params_at_qstar_gt_1(self):
        theory_fresh = _params("CCAP")  # fresh object -> fresh log dedup key
        m, c = self.knots()
        with self.assertLogs("HARK.ConsumptionSaving.ConsAggShockModel",
                             level="INFO") as cm:
            make_cFunc_slice(m, c, self.MPCmin, self.hNrm,
                             decay_theory=theory_fresh)
            # second slice with the SAME params object: no second log line
            make_cFunc_slice(m, c, self.MPCmin, self.hNrm,
                             decay_theory=theory_fresh)
        ratio_lines = [r for r in cm.output if "amplitude ratio" in r]
        self.assertEqual(len(ratio_lines), 1)


class TestSolverThreading(unittest.TestCase):
    """decay_theory/decay_Q threaded through solveConsAggShock via time_inv_,
    exactly like the PR threads MPCmin/hNrm. Includes an actual-solve identity
    check: explicit decay_theory=None reproduces the default solve array-equal
    (the cross-commit byte-identity vs the pre-change HEAD was verified on the
    same probes during development; the healthy/fallback slice paths are pinned
    byte-for-byte in TestMakeCFuncSliceTheoryModes)."""

    @classmethod
    def _fresh_agent(cls, **attrs):
        agent = AggShockConsumerType(seed=0, AgentCount=100, cycles=0)
        economy = CobbDouglasEconomy(agents=[agent], seed=0, act_T=20,
                                     max_loops=1)
        economy.give_agent_params()
        for k, v in attrs.items():
            setattr(agent, k, v)
        agent.tolerance = 1e-4
        agent.solve()
        return agent, economy

    @classmethod
    def setUpClass(cls):
        cls.probe_m = np.concatenate(
            [np.linspace(0.1, 20.0, 30), np.geomspace(25.0, 5000.0, 30)]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.agent_default, cls.eco = cls._fresh_agent()
            cls.agent_explicit, _ = cls._fresh_agent(
                decay_theory=None, decay_Q="theory"
            )
            # PR opt-in bounds at the steady-state reference return
            R = float(cls.eco.Rfunc(cls.eco.kSS))
            w = float(cls.eco.wFunc(cls.eco.kSS))
            d = cls.agent_default.IncShkDstn[0]
            pmv = np.asarray(d.pmv)
            E_inc = float(np.sum(pmv * np.asarray(d.atoms[0])
                                 * np.asarray(d.atoms[1])))
            G_tot = (float(np.asarray(cls.agent_default.PermGroFac).flat[0])
                     * float(cls.agent_default.PermGroFacAgg))
            bounds = dict(
                MPCmin=pf_mpc_min(
                    R,
                    float(cls.agent_default.DiscFac),
                    float(cls.agent_default.CRRA),
                    float(np.asarray(cls.agent_default.LivPrb).flat[0]),
                ),
                hNrm=float(pf_human_wealth_markov(
                    np.array([[1.0]]), R, np.array([w * E_inc]),
                    np.array([G_tot]))[0]),
            )
            cls.agent_optin, _ = cls._fresh_agent(**bounds)
            # theory params from the agent's idiosyncratic shock marginals
            def row_marginal(row):
                vals, inv = np.unique(np.asarray(d.atoms[row]),
                                      return_inverse=True)
                p = np.zeros(len(vals))
                np.add.at(p, inv, pmv)
                return vals, p

            cls.theory = powerlaw_decay_params(
                R, G_tot,
                float(cls.agent_default.DiscFac),
                float(cls.agent_default.CRRA),
                LivPrb=float(np.asarray(cls.agent_default.LivPrb).flat[0]),
                PermShkDstn=row_marginal(0), TranShkDstn=row_marginal(1),
                warn=False,
            )
            cls.agent_theory, _ = cls._fresh_agent(
                decay_theory=cls.theory, **bounds
            )

    def _probe(self, agent):
        M = np.full_like(self.probe_m, self.eco.MSS)
        return agent.solution[0].cFunc(self.probe_m, M)

    def test_instance_defaults_exist(self):
        agent = AggShockConsumerType(seed=0, AgentCount=10, cycles=0)
        self.assertIsNone(agent.decay_theory)
        self.assertEqual(agent.decay_Q, "theory")
        self.assertIn("decay_theory", agent.time_inv_)
        self.assertIn("decay_Q", agent.time_inv_)

    def test_explicit_none_identical_to_default(self):
        np.testing.assert_array_equal(
            self._probe(self.agent_default), self._probe(self.agent_explicit)
        )

    @staticmethod
    def _slices(agent):
        # LowerEnvelope2D -> VariableLowerBoundFunc2D -> LinearInterpOnInterp1D
        return agent.solution[0].cFunc.functions[0].func.xInterpolators

    def test_theory_threading_reaches_solved_slices(self):
        # theory is well-posed at this calibration (R_SS ~ 1.042 makes the
        # default AggShock economy a q* > 1 calibration, so q = min(1, q*) = 1)
        self.assertTrue(self.theory.valid)
        self.assertGreater(self.theory.q_star, 1.0)
        self.assertEqual(self.theory.q, 1.0)
        self.assertIsNotNone(self.theory.B_psi)
        # decay_theory travelled through time_inv_ into every solved slice:
        # the metadata dict is attached with this exact params object's numbers
        for s in self._slices(self.agent_theory):
            self.assertTrue(hasattr(s, "decay_theory"))
            self.assertEqual(s.decay_theory["q_star"], self.theory.q_star)
        for s in self._slices(self.agent_optin):
            self.assertFalse(hasattr(s, "decay_theory"))

    def test_theory_inert_when_slices_are_above_line_transients(self):
        # At this GE calibration the kSS-reference PF line lies BELOW the
        # solved top knots (above-line transients on the default shallow
        # grid), so the tail policy must stay INERT: every slice falls back to
        # the legacy bare extrapolation in BOTH agents and the solves agree
        # exactly. (An engaged theory tail on a real solve is pinned at slice
        # level in TestMakeCFuncSliceTheoryModes; picking a GE reference that
        # puts the line above the knots is deliberately the caller's problem.)
        for agent in (self.agent_optin, self.agent_theory):
            for s in self._slices(agent):
                self.assertFalse(s.decay_extrap)
        c_optin = self._probe(self.agent_optin)
        c_theory = self._probe(self.agent_theory)
        self.assertTrue(np.all(np.isfinite(c_theory)))
        np.testing.assert_array_equal(c_theory, c_optin)

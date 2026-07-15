"""Tests for the optional theory-pinned tail extrapolation of IndShockConsumerType
(``decay_extrap_form`` / ``decay_extrap_form_lower``) and its supporting pieces
(``KappaBarTailInterp``, ``pf_decay.ce_psi_regime``, ``pf_decay.aXtraMin_from_tail_tol``).

This file PORTS the theorem program's extrapolator-fidelity protocol
(``theory/powerlaw-decay/verify_extrap_fidelity_checks.py`` at
BufferStockTheory-Latest @ 12b0b178) into HARK's pytest:

* NESTED GRIDS: the truth solve uses a wide dense log a-grid; small solves use
  the strict subset ``grid[k:-k]`` (subset identity asserted), so the removed
  windows at each end measure exactly what the tails must recover.
* TRUTH = the tails-in-solve solve on the pinned grid (the battery's
  pre-registered truth definition); NON-CIRCULARITY is audited by rails-only
  solves on grids EXTENDED past the comparison window (G-AUD ports), whose
  agreement with the truth on the window is both gated and used as the
  measurement FLOOR below which no fidelity ratio is gated (the battery-v2
  floor discipline).
* THE TWO ROLES: in-solve tails (the option) versus tails attached to a
  default solve's cFunc post hoc (eval-only). The in-solve variant must be
  strictly better -- the expectation channel is where the value lives.

Pre-registered gates (declared BEFORE first CI run; may be tightened, never
loosened). Values were MEASURED at authoring time (2026-07-14, linux/x86-64
float64). The ERROR gates carry >= 21x headroom above their measured values
and must sit ABOVE the printed cross-solve audit floor; the two-roles FACTOR
gates are minimum improvement factors with smaller margins over their
measured factors (CE bottom: gate 10 vs measured ~353x = 35x margin; HS top:
gate 1.3 vs measured 2.42x = 1.9x margin; HS bottom: gate 3 vs measured
~19.6x = 6.5x margin):

    CE  bottom sup rel-err (tails, k=160)   gate 1e-8   measured 4.77e-11 (audit floor 2.53e-11)
    CE  top    sup rel-err (tails, k=160)   gate 1e-6   measured 2.42e-08
    CE  in-grid contamination (tails)       gate 1e-9   measured 8.40e-12 (rails 1.64e-08)
    CE  MPC at smallest removed node        gate 1e-6   measured ~9e-12 rel dev from MPCmax
    CE  two-roles factor (bottom)           gate >= 10  measured ~353x
    HS  top    sup rel-err (tails, k=150)   gate 1e-6   measured 2.28e-08 (audit floor 2.96e-10)
    HS  bottom sup rel-err (tails, psi-general regime I)
                                            gate 2e-4   measured 9.51e-06 (rails 1.55e-03)
    HS  MPC at smallest removed excess node gate 1e-5   measured 4.47e-07
    HS  two-roles factor (top)              gate >= 1.3 measured ~2.4x
    HS  two-roles factor (bottom)           gate >= 3   measured ~19.6x
    ladder monotonicity (both calibrations): shallower trim never worse

# THEOREM-REF[BufferStockTheory-Latest @ 12b0b178 :: theory/powerlaw-decay/statement.md :: st-thm-CE :: https://llorracc.github.io/BufferStockTheory-Latest/powerlaw-decay-theory/statement/]
#   Bottom gates: c = kap_bar*me - K*me**(1+rho), MPC -> kap_bar as me -> 0
#   (q_down = rho, no root-finding at the constraint end).
# THEOREM-REF[BufferStockTheory-Latest @ 12b0b178 :: theory/powerlaw-decay/statement.md :: st-thm-CE-psi :: https://llorracc.github.io/BufferStockTheory-Latest/powerlaw-decay-theory/statement/]
#   The psi-general bottom (CAL-HS block) is theorem-backed in regime I only:
#   lambda(psi_min) = p_eff**(1/rho)*Thorn_Gamma/psi_min < 1; the regime-gate
#   tests construct a regime-II process and assert the warning + refusal
#   (st-rem-CE-regime).
# THEOREM-REF[BufferStockTheory-Latest @ 12b0b178 :: theory/powerlaw-decay/final_proof_myst.md :: eq-powerlaw :: https://llorracc.github.io/BufferStockTheory-Latest/powerlaw-decay-theory/]
#   Top gates: the gap below MPCmin*(m + hNrm) decays as wbar**(-min(1, q*)).
#   The two-roles finding and fig7/fig8 evidence:
#   https://llorracc.github.io/BufferStockTheory-Latest/powerlaw-decay-theory/extrapolators-in-practice

Byte-zero regression: with the options at their ``None`` defaults the solve
must be byte-for-byte the parent code path. The pinned probe values below were
generated at the parent commit df508470 (the theory-pin retarget commit, the
tip this PR stacks on). On the pin-generation platform (linux/x86-64 float64)
they must match EXACTLY (float equality, no tolerance); on other platforms
libm last-ulp differences shift the solved values legitimately, so the pin
comparison degrades to ``np.allclose(rtol=1e-9)`` there -- the platform-FREE
byte-zero guarantee is the in-process explicit-None-equals-stock test, which
stays exact everywhere.
"""

import platform
import sys
import unittest
import warnings

import numpy as np

from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    solve_one_period_ConsIndShock,
    solve_one_period_ConsIndShock_with_tails,
)
from HARK.ConsumptionSaving.pf_decay import (
    ConstraintEndRegimeWarning,
    aXtraMin_from_tail_tol,
    ce_psi_regime,
    powerlaw_decay_params_from_agent,
)
from HARK.distributions import DiscreteDistributionLabeled
from HARK.interpolation import (
    DecayTailInterp,
    KappaBarTailInterp,
    LinearInterp,
)

# --------------------------------------------------------------- calibrations
# CE-rho2 (bottom-primary; psi == 1 via PermShkStd=0, zero-income unemployment
# atom): the theorem program's verify_constraint_end_checks NB-rho2 calibration
# in HARK parameters. kap_bar = 1 - 0.05**0.5*Thorn_R = 0.785165538...
CE_PARS = dict(
    cycles=0,
    T_cycle=1,
    CRRA=2.0,
    Rfree=[1.04],
    DiscFac=0.96,
    LivPrb=[1.0],
    PermGroFac=[1.0],
    BoroCnstArt=None,
    vFuncBool=False,
    CubicBool=False,
    UnempPrb=0.05,
    IncUnemp=0.0,
    TranShkStd=[0.1],
    TranShkCount=7,
    PermShkStd=[0.0],
    PermShkCount=1,
    T_retire=0,
    UnempPrbRet=0.0,
    IncUnempRet=0.0,
)
# CAL-HS (top-primary; psi-general): the estimated high-school calibration of
# the theorem program's figure scripts, on HARK's own discretization (hence
# q_star = 0.3759 here vs 0.3813 on the reference stack's atom grid). Its
# worst JOINT atom is the lowest employed-income atom (theta_min < IncUnemp),
# giving lambda(psi_min) deep in regime I (statement.md st-rem-CE-regime).
HS_PARS = dict(
    cycles=0,
    T_cycle=1,
    CRRA=2.0,
    Rfree=[1.01],
    DiscFac=0.98051,
    LivPrb=[1.0 - 1.0 / 160.0],
    PermGroFac=[1.0 + 0.01812 / 4],
    BoroCnstArt=None,
    vFuncBool=False,
    CubicBool=False,
    UnempPrb=0.044,
    IncUnemp=0.7,
    TranShkStd=[np.sqrt(0.12)],
    TranShkCount=7,
    PermShkStd=[np.sqrt(0.003)],
    PermShkCount=7,
    T_retire=0,
    UnempPrbRet=0.0,
    IncUnempRet=0.0,
)

# nested-grid design (runtime-budgeted port of the battery's Na=6000 ladder)
CE_GRID = dict(aMin=1e-6, aMax=1e5, Na=800)  # ~72 nodes/decade
CE_K = 160  # ~2.2 decades trimmed per end
CE_K_SHALLOW = 80
HS_GRID = dict(aMin=1e-4, aMax=1e8, Na=900)  # ~75 nodes/decade
HS_K = 150  # 2.0 decades trimmed per end
HS_K_SHALLOW = 75
SOLVE_TOL = 1e-12
AUDIT_TOL = 1e-11  # the HS audit grid is ~2 decades longer; 1e-12 would hit
#                     core.solve_agent's max_cycles=5000 escape hatch

# pre-registered gates (see module docstring for the measured values)
GATE_CE_BOTTOM = 1e-8
GATE_CE_TOP = 1e-6
GATE_CE_CONTAM = 1e-9
GATE_CE_MPC = 1e-6
GATE_CE_TWO_ROLES = 10.0
GATE_CE_AUDIT = 1e-9  # G-AUD1 port: deep-bottom rails vs truth on window
GATE_HS_TOP = 1e-6
GATE_HS_BOTTOM = 2e-4
GATE_HS_MPC = 1e-5
GATE_HS_TWO_ROLES_TOP = 1.3  # measured 2.42x on linux/x86-64; lowered from
#   the 1.5 authoring strawman for platform robustness (the factor gates
#   carry far less headroom than the error gates -- module docstring)
GATE_HS_TWO_ROLES_BOTTOM = 3.0
GATE_HS_AUDIT = 1e-8  # G-AUD3 port: tall-top rails vs truth on window

# Byte-zero pins, generated at parent commit df508470 (see gen note in the
# module docstring). Probes: m = [0.5, 1, 2, 5, 10, 20, 40].
BYTE_ZERO_PROBES = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0])
# config A: bone-stock IndShockConsumerType(cycles=0)
BYTE_ZERO_PIN_A = [
    0.5,
    0.8652278427790324,
    1.098045258173503,
    1.3730629141572253,
    1.6897273979915504,
    2.234807386872702,
    3.2691485935849434,
]
# config B: cycles=0, CubicBool=True, vFuncBool=True, BoroCnstArt=None
BYTE_ZERO_PIN_B = [
    1.0194785800743766,
    1.08444079226619,
    1.1836154886611796,
    1.405144565610918,
    1.7072978107134946,
    2.2460708318213296,
    3.269100840528629,
]
BYTE_ZERO_PIN_B_VFUNC_AT_5 = -14.103921502643747


# ------------------------------------------------------------------- helpers
def log_grid(aMin, aMax, Na):
    return np.exp(np.linspace(np.log(aMin), np.log(aMax), Na))


def solve_agent(base_pars, grid, tol, **opts):
    pars = dict(base_pars)
    pars.update(opts)
    agent = IndShockConsumerType(**pars)
    agent.verbose = 0
    if grid is not None:
        agent.aXtraGrid = np.asarray(grid, float)
    agent.tolerance = tol
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agent.solve()
    return agent


def unwrap_m_nodes(cfunc):
    """The EGM m-node grid of an assembled cFunc (corner included), through
    any tail wrappers and the LowerEnvelope."""
    f = cfunc
    while True:
        if isinstance(f, (KappaBarTailInterp, DecayTailInterp)):
            f = f.interp
        elif hasattr(f, "functions"):
            f = f.functions[0]
        elif hasattr(f, "x_list"):
            return np.asarray(f.x_list, float)
        else:  # pragma: no cover - would signal an assembly change
            raise TypeError(type(f))


def sup_rel_err(cfunc_variant, cfunc_truth, probes):
    cv = np.asarray(cfunc_variant(probes), float)
    ct = np.asarray(cfunc_truth(probes), float)
    return float(np.max(np.abs(cv - ct) / ct))


def attach_eval_only(solution, q_eff, CRRA, do_top=True, do_bottom=True):
    """The 'tails-eval-only' variant: wrap a default (rails) solution's cFunc
    post hoc with the SAME tail laws the in-solve option uses (two-term top,
    strict kappa_bar bottom). Isolates role 2 (post-solve evaluation) from
    role 1 (in-solve expectations)."""
    m = unwrap_m_nodes(solution.cFunc)
    f = solution.cFunc
    if do_top:
        f = DecayTailInterp(
            f,
            solution.MPCmin * solution.hNrm,
            solution.MPCmin,
            x_cut=float(m[-1]),
            decay_extrap_Q=q_eff,
        )
    if do_bottom:
        # HARK's default bottom is the EGM secant, which lies BELOW the true
        # concave cFunc, so the eval-only knot has K > 0 and strict=True holds
        # (unlike the reference stack's c = m rail, which needed the corridor).
        f = KappaBarTailInterp(
            f,
            solution.MPCmax,
            CRRA,
            solution.mNrmMin,
            x_knot=float(m[1]),
            strict=True,
        )
    return f


_CACHE = {}


def ce_bundle():
    """All CE-rho2 solves, computed once and shared across tests."""
    if "CE" in _CACHE:
        return _CACHE["CE"]
    grid = log_grid(**CE_GRID)
    tails = dict(decay_extrap_form="powerlaw", decay_extrap_form_lower="kappabar")
    truth = solve_agent(CE_PARS, grid, SOLVE_TOL, **tails)
    # G-AUD1 port: rails-only, bottom extended 4 decades at equal density
    npd = (CE_GRID["Na"] - 1) / np.log10(CE_GRID["aMax"] / CE_GRID["aMin"])
    grid_aud = log_grid(
        CE_GRID["aMin"] * 1e-4, CE_GRID["aMax"], CE_GRID["Na"] + int(4 * npd)
    )
    audit = solve_agent(CE_PARS, grid_aud, SOLVE_TOL)
    out = dict(grid=grid, truth=truth, audit=audit)
    for k in (CE_K, CE_K_SHALLOW):
        sub = grid[k:-k]
        out[("rails", k)] = solve_agent(CE_PARS, sub, SOLVE_TOL)
        out[("tails", k)] = solve_agent(CE_PARS, sub, SOLVE_TOL, **tails)
    _CACHE["CE"] = out
    return out


def hs_bundle():
    """All CAL-HS solves, computed once and shared across tests."""
    if "HS" in _CACHE:
        return _CACHE["HS"]
    grid = log_grid(**HS_GRID)
    tails = dict(decay_extrap_form="powerlaw", decay_extrap_form_lower="kappabar")
    truth = solve_agent(HS_PARS, grid, SOLVE_TOL, **tails)
    # G-AUD3 port: rails-only, top extended 2 decades at equal density
    npd = (HS_GRID["Na"] - 1) / np.log10(HS_GRID["aMax"] / HS_GRID["aMin"])
    grid_aud = log_grid(
        HS_GRID["aMin"], HS_GRID["aMax"] * 1e2, HS_GRID["Na"] + int(2 * npd)
    )
    audit = solve_agent(HS_PARS, grid_aud, AUDIT_TOL)
    out = dict(grid=grid, truth=truth, audit=audit)
    for k in (HS_K,):
        sub = grid[k:-k]
        out[("rails", k)] = solve_agent(HS_PARS, sub, SOLVE_TOL)
        out[("tails", k)] = solve_agent(HS_PARS, sub, SOLVE_TOL, **tails)
    out[("tails", HS_K_SHALLOW)] = solve_agent(
        HS_PARS, grid[HS_K_SHALLOW:-HS_K_SHALLOW], SOLVE_TOL, **tails
    )
    _CACHE["HS"] = out
    return out


def bottom_window(truth_sol, small_sol, n=60):
    """Log-spaced probes in the removed-bottom window, in EXCESS coordinates
    (correct for psi-general mNrmMin < 0), strictly between the two solves'
    first EGM nodes."""
    m1_t = float(unwrap_m_nodes(truth_sol.cFunc)[1])
    m1_s = float(unwrap_m_nodes(small_sol.cFunc)[1])
    mm = max(truth_sol.mNrmMin, small_sol.mNrmMin)
    me_lo = (m1_t - truth_sol.mNrmMin) * 1.05
    me_hi = (m1_s - small_sol.mNrmMin) * 0.95
    return mm + np.exp(np.linspace(np.log(me_lo), np.log(me_hi), n))


def top_window(truth_sol, small_sol, n=80):
    m_top_t = float(unwrap_m_nodes(truth_sol.cFunc)[-1])
    m_top_s = float(unwrap_m_nodes(small_sol.cFunc)[-1])
    return np.exp(np.linspace(np.log(m_top_s * 1.05), np.log(m_top_t * 0.95), n))


def interior_window(truth_sol, small_sol, n=200):
    mm = max(truth_sol.mNrmMin, small_sol.mNrmMin)
    m1 = max(
        float(unwrap_m_nodes(truth_sol.cFunc)[1]),
        float(unwrap_m_nodes(small_sol.cFunc)[1]),
    )
    m_top_s = float(unwrap_m_nodes(small_sol.cFunc)[-1])
    return mm + np.exp(
        np.linspace(np.log((m1 - mm) * 2.0), np.log((m_top_s - mm) / 2.0), n)
    )


def make_joint_dstn(psi_atoms, psi_probs, th_atoms, th_probs):
    """Hand-built joint IncShkDstn (independent product measure), labeled the
    way the solver consumes it."""
    PSI, TH = np.meshgrid(psi_atoms, th_atoms, indexing="ij")
    P = np.outer(psi_probs, th_probs)
    atoms = np.vstack([PSI.ravel(), TH.ravel()])
    return DiscreteDistributionLabeled(
        P.ravel(), atoms, var_names=["PermShk", "TranShk"]
    )


# ============================================================ unit: the class
class TestKappaBarTailInterpUnit(unittest.TestCase):
    """Hand-computed probes and guard trips for the new interpolant (mirrors
    the reference library's self-test (ii)/(iv) cases)."""

    def body(self):
        return LinearInterp(np.array([0.5, 1.0]), np.array([0.35, 0.7]))

    def test_hand_computed_tail_values_and_delegation(self):
        # kap_bar=0.8, CRRA=2, knot (0.5, 0.35) => K = (0.4-0.35)/0.125 = 0.4
        f = KappaBarTailInterp(self.body(), 0.8, 2.0, 0.0, x_knot=0.5)
        self.assertAlmostEqual(f.K, 0.4, places=14)
        got = np.asarray(f(np.array([0.25, 0.1])), float)
        np.testing.assert_allclose(got, [0.19375, 0.0796], rtol=1e-15)
        # at/above the knot: delegated to the body
        np.testing.assert_allclose(
            np.asarray(f(np.array([0.5, 0.75])), float), [0.35, 0.525], rtol=1e-15
        )
        # level continuity at the knot (the class invariant)
        eps = 1e-12
        self.assertAlmostEqual(float(f(0.5 - eps)), 0.35, places=10)
        # explicit y_knot route agrees with the body-read route
        f2 = KappaBarTailInterp(self.body(), 0.8, 2.0, 0.0, x_knot=0.5, y_knot=0.35)
        self.assertEqual(float(f2(0.25)), float(f(0.25)))

    def test_mnrmmin_shift_and_constraint_zero(self):
        body = LinearInterp(np.array([-0.5, 0.0]), np.array([0.35, 0.7]))
        f = KappaBarTailInterp(body, 0.8, 2.0, -1.0, x_knot=-0.5)
        self.assertAlmostEqual(f.K, 0.4, places=14)
        self.assertAlmostEqual(float(f(-0.75)), 0.19375, places=15)
        # at/below the constraint: consumption 0, MPC -> MPCmax
        self.assertEqual(float(f(-1.0)), 0.0)
        self.assertEqual(float(f(-2.0)), 0.0)
        self.assertAlmostEqual(float(f.derivative(-1.0)), 0.8, places=15)

    def test_derivative_vs_central_fd_and_limit(self):
        f = KappaBarTailInterp(self.body(), 0.8, 2.0, 0.0, x_knot=0.5)
        for m in (0.35, 0.2, 0.1):
            h = 1e-7
            fd = (float(f(m + h)) - float(f(m - h))) / (2 * h)
            self.assertAlmostEqual(float(f.derivative(m)) / fd, 1.0, places=6)
        # MPC -> MPCmax from below as me -> 0 (Theorem CE's content)
        self.assertAlmostEqual(float(f.derivative(1e-12)), 0.8, places=9)
        self.assertLessEqual(float(f.derivative(0.25)), 0.8)

    def test_guards_strict_corridor_and_exposure_gate(self):
        body = LinearInterp(np.array([1.0, 2.0]), np.array([0.9, 1.8]))
        # K < 0 (knot above the kap_bar line): strict raises with the
        # st-cor-C4 grid-rule message; the corridor constructor admits it.
        with self.assertRaises(ValueError) as cm:
            KappaBarTailInterp(body, 0.8, 2.0, 0.0, x_knot=1.0)
        self.assertIn("st-cor-C4", str(cm.exception))
        f = KappaBarTailInterp(body, 0.8, 2.0, 0.0, x_knot=1.0, strict=False)
        self.assertFalse(f.in_regime)
        self.assertAlmostEqual(f.K, -0.1, places=14)
        # ... but try_make refuses it (MPC would exceed MPCmax throughout)
        self.assertIsNone(KappaBarTailInterp.try_make(body, 0.8, 2.0, 0.0, x_knot=1.0))
        # outside even the corridor: both non-strict construction and try_make
        body_far = LinearInterp(np.array([1.0, 2.0]), np.array([1.7, 3.4]))
        with self.assertRaises(ValueError):
            KappaBarTailInterp(body_far, 0.8, 2.0, 0.0, x_knot=1.0, strict=False)
        self.assertIsNone(
            KappaBarTailInterp.try_make(body_far, 0.8, 2.0, 0.0, x_knot=1.0)
        )
        # negative-MPC leak of the raw corridor (recorded v2 item): K in
        # regime (strict constructs) but (1+CRRA)*K*me**CRRA >= MPCmax, so the
        # exposure gate refuses the in-solve tail.
        body_neg = LinearInterp(np.array([1.0, 2.0]), np.array([0.3, 0.6]))
        g = KappaBarTailInterp(body_neg, 0.8, 2.0, 0.0, x_knot=1.0)
        self.assertTrue(g.in_regime)
        self.assertLess(g.mpc_at_knot, 0.0)
        self.assertIsNone(
            KappaBarTailInterp.try_make(body_neg, 0.8, 2.0, 0.0, x_knot=1.0)
        )
        # a good knot passes try_make and matches the direct construction
        ok = KappaBarTailInterp.try_make(self.body(), 0.8, 2.0, 0.0, x_knot=0.5)
        self.assertIsNotNone(ok)
        self.assertTrue(ok.in_regime)
        self.assertGreater(ok.mpc_at_knot, 0.0)
        # malformed scalars fail closed (None), never raise
        self.assertIsNone(
            KappaBarTailInterp.try_make(self.body(), np.nan, 2.0, 0.0, x_knot=0.5)
        )
        self.assertIsNone(
            KappaBarTailInterp.try_make(self.body(), 0.8, 2.0, 0.6, x_knot=0.5)
        )  # knot below mNrmMin


# ==================================================== unit: pf_decay additions
class TestCePsiRegime(unittest.TestCase):
    def test_regime_I_on_ce_process(self):
        agent = IndShockConsumerType(**CE_PARS)
        PG = float((1.04 * 0.96) ** 0.5)  # Thorn_Gamma at G=1, LivPrb=1
        reg = ce_psi_regime(agent.IncShkDstn[0], 2.0, PG)
        self.assertEqual(reg["regime"], "I")
        self.assertEqual(reg["anchor"], "st-thm-CE-psi")
        self.assertAlmostEqual(reg["p_eff"], 0.05, places=12)
        # hand value: 0.05**0.5 * PG / 1.0
        self.assertAlmostEqual(reg["lambda_min_fiber"], 0.05**0.5 * PG, places=12)

    def test_regime_II_on_fat_low_fiber(self):
        psi = np.array([0.12, 1.1552941176470588])  # E[psi] = 1
        pp = np.array([0.15, 0.85])
        th = np.array([0.0, 1.0 / 0.7])  # E[theta] = 1
        tp = np.array([0.3, 0.7])
        joint = make_joint_dstn(psi, pp, th, tp)
        PG = float((1.04 * 0.96) ** 0.5)
        reg = ce_psi_regime(joint, 2.0, PG)
        self.assertEqual(reg["regime"], "II")
        self.assertEqual(reg["anchor"], "st-rem-CE-regime")
        # zero transitory atom => p_eff = full worst-theta mass 0.3
        self.assertAlmostEqual(reg["p_eff"], 0.3, places=12)
        self.assertAlmostEqual(reg["lambda_min_fiber"], 0.3**0.5 * PG / 0.12, places=10)

    def test_worst_joint_mass_fiber_selection(self):
        # xi_min > 0: the worst JOINT atom is (psi_min, theta_min), so p_eff
        # is the fiber-selected p_w * P[psi = psi_min] (st-def-ce-psi-objects)
        psi = np.array([0.9, 1.1])
        pp = np.array([0.5, 0.5])
        th = np.array([0.5, 1.5])
        tp = np.array([0.5, 0.5])
        joint = make_joint_dstn(psi, pp, th, tp)
        reg = ce_psi_regime(joint, 2.0, 0.99)
        self.assertAlmostEqual(reg["p_eff"], 0.25, places=12)
        self.assertAlmostEqual(reg["psi_min"], 0.9, places=12)


class TestAXtraMinFromTailTol(unittest.TestCase):
    def test_hand_inversion(self):
        # K = (0.8*0.5 - 0.35)/0.5**3 = 0.4 (up to float rounding);
        # me_target = (tol*0.8/K)**(1/2)/1.5; aXtraMin = (1-0.8)*me_target
        a = aXtraMin_from_tail_tol(0.5, 0.35, 0.8, 2.0, 1e-4)
        K = (0.8 * 0.5 - 0.35) / 0.5**3
        me_target = (1e-4 * 0.8 / K) ** 0.5 / 1.5
        self.assertAlmostEqual(a, (1.0 - 0.8) * me_target, places=15)
        # explicit-K route bypasses the measurement (equal up to the last-ulp
        # difference between the measured K and the literal 0.4)
        a2 = aXtraMin_from_tail_tol(None, None, 0.8, 2.0, 1e-4, K=0.4)
        self.assertAlmostEqual(a, a2, places=15)

    def test_fail_closed(self):
        # reference node ON/ABOVE the kap_bar line: K <= 0 -> nan (st-cor-C4)
        self.assertTrue(np.isnan(aXtraMin_from_tail_tol(0.5, 0.45, 0.8, 2.0, 1e-4)))
        self.assertTrue(np.isnan(aXtraMin_from_tail_tol(0.5, 0.35, 1.2, 2.0, 1e-4)))
        self.assertTrue(np.isnan(aXtraMin_from_tail_tol(0.5, 0.35, 0.8, 2.0, np.nan)))
        # tail_tol clamped at the float64-certifiable floor 1e-6
        self.assertEqual(
            aXtraMin_from_tail_tol(0.5, 0.35, 0.8, 2.0, 1e-9),
            aXtraMin_from_tail_tol(0.5, 0.35, 0.8, 2.0, 1e-6),
        )

    def test_expost_certificate_on_ce_solve(self):
        """End-to-end: invert a coarse CE solve's bottom knot to an aXtraMin
        for tail_tol, re-solve, and verify the certificate binds."""
        tail_tol = 1e-3
        coarse = solve_agent(CE_PARS, log_grid(1e-2, 1e4, 200), 1e-10)
        sol = coarse.solution[0]
        m1 = float(unwrap_m_nodes(sol.cFunc)[1])
        me1 = m1 - sol.mNrmMin
        c1 = float(sol.cFunc(m1))
        a_min = aXtraMin_from_tail_tol(me1, c1, sol.MPCmax, 2.0, tail_tol)
        self.assertTrue(np.isfinite(a_min) and a_min > 0.0)
        fine = solve_agent(CE_PARS, log_grid(a_min, 1e4, 300), 1e-10)
        fsol = fine.solution[0]
        fm1 = float(unwrap_m_nodes(fsol.cFunc)[1])
        fme1 = fm1 - fsol.mNrmMin
        fc1 = float(fsol.cFunc(fm1))
        measured = (fsol.MPCmax - fc1 / fme1) / fsol.MPCmax
        self.assertLessEqual(
            measured,
            tail_tol,
            "ex-post certificate failed: measured %.3e > "
            "tail_tol %.1e" % (measured, tail_tol),
        )


# ================================================== option surface + byte-zero
class TestOptionSurface(unittest.TestCase):
    def test_defaults_are_none(self):
        agent = IndShockConsumerType(**CE_PARS)
        self.assertIsNone(agent.decay_extrap_form)
        self.assertIsNone(agent.decay_extrap_Q)
        self.assertIsNone(agent.decay_extrap_form_lower)

    def test_auto_exponent_is_min_one_qstar(self):
        ce = ce_bundle()
        agent = ce[("tails", CE_K)]
        params = powerlaw_decay_params_from_agent(agent, warn=False)
        self.assertEqual(agent.decay_extrap_Q, min(1.0, params.q_star))
        self.assertEqual(agent.decay_extrap_Q, 1.0)  # CE: q_star ~ 49
        hs = hs_bundle()
        agent_hs = hs[("tails", HS_K)]
        params_hs = powerlaw_decay_params_from_agent(agent_hs, warn=False)
        self.assertEqual(agent_hs.decay_extrap_Q, min(1.0, params_hs.q_star))
        self.assertLess(agent_hs.decay_extrap_Q, 1.0)  # HS: q_star ~ 0.376

    def test_user_Q_after_auto_solve_is_respected(self):
        """An explicit decay_extrap_Q assigned AFTER an auto-computed solve
        must not be clobbered by the next solve's auto refresh (which applies
        only while Q still equals the remembered auto value)."""
        grid = log_grid(1e-4, 1e3, 100)

        def fresh():
            a = IndShockConsumerType(**CE_PARS)
            a.verbose = 0
            a.aXtraGrid = grid
            a.tolerance = 1e-8
            a.decay_extrap_form = "powerlaw"
            return a

        agent = fresh()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agent.solve()
        self.assertEqual(agent.decay_extrap_Q, 1.0)  # CE auto: min(1, ~49)
        agent.decay_extrap_Q = 0.9  # user override AFTER the auto solve
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agent.solve()
        self.assertEqual(agent.decay_extrap_Q, 0.9)  # respected, not reset
        # a still-auto value keeps refreshing across solves...
        agent2 = fresh()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agent2.solve()
            agent2.solve()
        self.assertEqual(agent2.decay_extrap_Q, 1.0)
        # ...and disabling the form clears a still-auto Q (no stale exponent)
        agent2.decay_extrap_form = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agent2.solve()
        self.assertIsNone(agent2.decay_extrap_Q)

    def test_vpfunc_carries_the_tails(self):
        """The role-1 mechanism: vPfunc = u'(cFunc(m')) is what the PREVIOUS
        backward step's Euler expectation evaluates (calc_vp_next), so the
        wrapped cFunc must be what vPfunc wraps -- one wiring point, both
        roles."""
        ce = ce_bundle()
        sol = ce[("tails", CE_K)].solution[0]
        self.assertIsInstance(sol.cFunc, KappaBarTailInterp)
        self.assertIsInstance(sol.vPfunc.cFunc, KappaBarTailInterp)

    def test_option_validation(self):
        agent = IndShockConsumerType(**CE_PARS)
        agent.verbose = 0
        agent.decay_extrap_form = "bogus"
        with self.assertRaises(ValueError):
            agent.solve()
        agent2 = IndShockConsumerType(**CE_PARS)
        agent2.verbose = 0
        agent2.decay_extrap_Q = 0.5  # without decay_extrap_form='powerlaw'
        with self.assertRaises(ValueError):
            agent2.solve()

    def test_solver_requires_explicit_Q(self):
        agent = IndShockConsumerType(**CE_PARS)
        agent.verbose = 0
        agent.solve()  # default solve, to get a valid solution_next
        with self.assertRaises(ValueError):
            solve_one_period_ConsIndShock_with_tails(
                agent.solution[0],
                agent.IncShkDstn[0],
                1.0,
                0.96,
                2.0,
                1.04,
                1.0,
                None,
                agent.aXtraGrid,
                False,
                False,
                decay_extrap_form="powerlaw",  # decay_extrap_Q missing
            )

    def test_public_solver_signature_unchanged(self):
        """The long-standing public solver must keep its EXACT argument
        names: solve_one_cycle builds the solver's argument dict from those
        names, so growing the shared signature would break agents whose
        re-solve paths skip pre_solve (e.g. the HANK Jacobian machinery's
        solve(presolve=False) -- the regression that motivated the
        _with_tails split)."""
        import inspect

        legacy = list(inspect.signature(solve_one_period_ConsIndShock).parameters)
        self.assertEqual(
            legacy,
            [
                "solution_next",
                "IncShkDstn",
                "LivPrb",
                "DiscFac",
                "CRRA",
                "Rfree",
                "PermGroFac",
                "BoroCnstArt",
                "aXtraGrid",
                "vFuncBool",
                "CubicBool",
            ],
        )
        # ... and the options swap the solver in only when enabled
        ce = ce_bundle()
        self.assertIs(
            ce[("tails", CE_K)].solve_one_period,
            solve_one_period_ConsIndShock_with_tails,
        )
        self.assertIs(
            ce[("rails", CE_K)].solve_one_period, solve_one_period_ConsIndShock
        )


class TestByteZeroDefault(unittest.TestCase):
    """Options unset => the solve is byte-for-byte the parent code path.
    The pins were generated at parent commit df508470 on linux/x86-64:
    equality is EXACT there and np.allclose(rtol=1e-9) on other platforms
    (libm last-ulp differences; see the module docstring). The in-process
    explicit-None-equals-stock test below is exact on EVERY platform."""

    # exact pin equality only on the pin-generation platform
    EXACT_PINS = sys.platform == "linux" and platform.machine() == "x86_64"

    def assert_pinned(self, got, pin):
        got = np.atleast_1d(np.asarray(got, float))
        pin = np.atleast_1d(np.asarray(pin, float))
        if self.EXACT_PINS:
            self.assertEqual(got.tolist(), pin.tolist())
        else:
            np.testing.assert_allclose(got, pin, rtol=1e-9)

    def test_stock_infinite_horizon_agent_unchanged(self):
        agent = IndShockConsumerType(cycles=0)
        agent.verbose = 0
        agent.solve()
        got = np.asarray(agent.solution[0].cFunc(BYTE_ZERO_PROBES), float)
        self.assert_pinned(got, BYTE_ZERO_PIN_A)
        self.assertEqual(type(agent.solution[0].cFunc).__name__, "LowerEnvelope")

    def test_cubic_vfunc_natural_constraint_unchanged(self):
        agent = IndShockConsumerType(
            cycles=0, CubicBool=True, vFuncBool=True, BoroCnstArt=None
        )
        agent.verbose = 0
        agent.solve()
        got = np.asarray(agent.solution[0].cFunc(BYTE_ZERO_PROBES), float)
        self.assert_pinned(got, BYTE_ZERO_PIN_B)
        self.assert_pinned(
            float(agent.solution[0].vFunc(np.array([5.0]))[0]),
            BYTE_ZERO_PIN_B_VFUNC_AT_5,
        )

    def test_explicit_none_options_identical_to_stock(self):
        a = IndShockConsumerType(cycles=0)
        a.verbose = 0
        a.solve()
        b = IndShockConsumerType(
            cycles=0,
            decay_extrap_form=None,
            decay_extrap_Q=None,
            decay_extrap_form_lower=None,
        )
        b.verbose = 0
        b.solve()
        va = np.asarray(a.solution[0].cFunc(BYTE_ZERO_PROBES), float)
        vb = np.asarray(b.solution[0].cFunc(BYTE_ZERO_PROBES), float)
        self.assertEqual(va.tolist(), vb.tolist())


# ================================================== nested-grid fidelity: CE
class TestCEFidelity(unittest.TestCase):
    """CE-rho2 (bottom-primary, psi == 1, zero-income atom): the Theorem CE
    story. Gates and measured values in the module docstring."""

    @classmethod
    def setUpClass(cls):
        cls.ce = ce_bundle()
        cls.truth = cls.ce["truth"].solution[0]
        cls.rails = cls.ce[("rails", CE_K)].solution[0]
        cls.tails = cls.ce[("tails", CE_K)].solution[0]
        cls.q_eff = cls.ce[("tails", CE_K)].decay_extrap_Q
        cls.pb = bottom_window(cls.truth, cls.tails)
        cls.pt = top_window(cls.truth, cls.tails)
        # the cross-solve measurement floor: deep-bottom rails audit vs truth
        cls.floor_bottom = sup_rel_err(
            cls.ce["audit"].solution[0].cFunc, cls.truth.cFunc, cls.pb
        )

    def test_subset_identity(self):
        grid = self.ce["grid"]
        np.testing.assert_array_equal(
            np.asarray(self.ce[("tails", CE_K)].aXtraGrid), grid[CE_K:-CE_K]
        )
        self.assertGreater(grid[CE_K], grid[0])
        self.assertLess(grid[-CE_K - 1], grid[-1])

    def test_audit_non_circularity(self):
        # G-AUD1 port: the rails-only solve on the 4-decades-deeper grid must
        # agree with the tails truth ON the removed-bottom window.
        self.assertLess(
            self.floor_bottom,
            GATE_CE_AUDIT,
            "audit floor %.3e vs gate %.1e" % (self.floor_bottom, GATE_CE_AUDIT),
        )

    def test_bottom_fidelity_gate(self):
        # floor discipline: the gate must sit above the printed audit floor
        self.assertGreater(
            GATE_CE_BOTTOM,
            10.0 * self.floor_bottom,
            "gate %.1e not >= 10x audit floor %.3e"
            % (GATE_CE_BOTTOM, self.floor_bottom),
        )
        err = sup_rel_err(self.tails.cFunc, self.truth.cFunc, self.pb)
        print(
            "\nCE bottom: tails %.3e (gate %.1e, audit floor %.3e)"
            % (err, GATE_CE_BOTTOM, self.floor_bottom)
        )
        self.assertLess(err, GATE_CE_BOTTOM)

    def test_bottom_beats_rails(self):
        err_r = sup_rel_err(self.rails.cFunc, self.truth.cFunc, self.pb)
        err_x = sup_rel_err(self.tails.cFunc, self.truth.cFunc, self.pb)
        self.assertGreater(
            err_r, 10.0 * err_x, "rails %.3e vs tails %.3e" % (err_r, err_x)
        )

    def test_bottom_mpc_approaches_kappa_bar(self):
        mpc = float(self.tails.cFunc.derivative(self.pb[0]))
        dev = abs(mpc / self.tails.MPCmax - 1.0)
        print(
            "CE MPC at m=%.3e: %.12f vs MPCmax %.12f (dev %.2e)"
            % (self.pb[0], mpc, self.tails.MPCmax, dev)
        )
        self.assertLess(dev, GATE_CE_MPC)

    def test_top_fidelity_gate(self):
        err = sup_rel_err(self.tails.cFunc, self.truth.cFunc, self.pt)
        err_r = sup_rel_err(self.rails.cFunc, self.truth.cFunc, self.pt)
        print("CE top: tails %.3e rails %.3e (gate %.1e)" % (err, err_r, GATE_CE_TOP))
        self.assertLess(err, GATE_CE_TOP)
        self.assertGreater(err_r, 10.0 * err)

    def test_two_roles_bottom(self):
        # in-solve must be strictly better than the eval-only wrap; both
        # sides must be above the measurement floor for the ratio to count.
        cf_eval = attach_eval_only(self.rails, self.q_eff, 2.0)
        err_eval = sup_rel_err(cf_eval, self.truth.cFunc, self.pb)
        err_in = sup_rel_err(self.tails.cFunc, self.truth.cFunc, self.pb)
        print(
            "CE two-roles bottom: in-solve %.3e vs eval-only %.3e "
            "(factor %.1fx, floor %.3e)"
            % (err_in, err_eval, err_eval / err_in, self.floor_bottom)
        )
        self.assertGreater(err_eval, self.floor_bottom)
        self.assertGreater(err_in, 0.0)
        self.assertLess(err_in, err_eval)
        self.assertGreater(err_eval / err_in, GATE_CE_TWO_ROLES)

    def test_in_grid_contamination(self):
        pi = interior_window(self.truth, self.tails)
        c_x = sup_rel_err(self.tails.cFunc, self.truth.cFunc, pi)
        c_r = sup_rel_err(self.rails.cFunc, self.truth.cFunc, pi)
        print(
            "CE contamination: tails %.3e rails %.3e (gate %.1e)"
            % (c_x, c_r, GATE_CE_CONTAM)
        )
        self.assertLess(c_x, GATE_CE_CONTAM)
        self.assertLessEqual(c_x, c_r)

    def test_ladder_monotone(self):
        # G-EX4 port (2-point ladder): a shallower trim is never worse.
        tails_sh = self.ce[("tails", CE_K_SHALLOW)].solution[0]
        pb_sh = bottom_window(self.truth, tails_sh)
        err_sh = sup_rel_err(tails_sh.cFunc, self.truth.cFunc, pb_sh)
        err_dp = sup_rel_err(self.tails.cFunc, self.truth.cFunc, self.pb)
        print(
            "CE ladder bottom: k=%d %.3e <= k=%d %.3e"
            % (CE_K_SHALLOW, err_sh, CE_K, err_dp)
        )
        self.assertLessEqual(err_sh, err_dp)

    def test_determinism(self):
        # a fresh identical solve reproduces the cFunc byte-for-byte
        grid = self.ce["grid"][CE_K:-CE_K]
        again = solve_agent(
            CE_PARS,
            grid,
            SOLVE_TOL,
            decay_extrap_form="powerlaw",
            decay_extrap_form_lower="kappabar",
        )
        probes = np.concatenate([self.pb, self.pt])
        v1 = np.asarray(self.tails.cFunc(probes), float)
        v2 = np.asarray(again.solution[0].cFunc(probes), float)
        self.assertEqual(v1.tolist(), v2.tolist())


# ================================================== nested-grid fidelity: HS
class TestHSFidelity(unittest.TestCase):
    """CAL-HS (top-primary, psi-general): the Theorem A1/eq-powerlaw story at
    the top; the Theorem CE-psi regime-I story at the bottom (mNrmMin < 0)."""

    @classmethod
    def setUpClass(cls):
        cls.hs = hs_bundle()
        cls.truth = cls.hs["truth"].solution[0]
        cls.rails = cls.hs[("rails", HS_K)].solution[0]
        cls.tails = cls.hs[("tails", HS_K)].solution[0]
        cls.q_eff = cls.hs[("tails", HS_K)].decay_extrap_Q
        cls.pt = top_window(cls.truth, cls.tails)
        cls.pb = bottom_window(cls.truth, cls.tails)
        cls.floor_top = sup_rel_err(
            cls.hs["audit"].solution[0].cFunc, cls.truth.cFunc, cls.pt
        )

    def test_audit_non_circularity(self):
        # G-AUD3 port: rails-only on the 2-decades-taller grid vs truth
        self.assertLess(
            self.floor_top,
            GATE_HS_AUDIT,
            "audit floor %.3e vs gate %.1e" % (self.floor_top, GATE_HS_AUDIT),
        )

    def test_top_fidelity_gate(self):
        self.assertGreater(
            GATE_HS_TOP,
            10.0 * self.floor_top,
            "gate %.1e not >= 10x audit floor %.3e" % (GATE_HS_TOP, self.floor_top),
        )
        err = sup_rel_err(self.tails.cFunc, self.truth.cFunc, self.pt)
        err_r = sup_rel_err(self.rails.cFunc, self.truth.cFunc, self.pt)
        print(
            "\nHS top: tails %.3e rails %.3e (gate %.1e, audit floor %.3e)"
            % (err, err_r, GATE_HS_TOP, self.floor_top)
        )
        self.assertLess(err, GATE_HS_TOP)
        self.assertGreater(err_r, 10.0 * err)

    def test_two_roles_top(self):
        cf_eval = attach_eval_only(self.rails, self.q_eff, 2.0, do_bottom=False)
        err_eval = sup_rel_err(cf_eval, self.truth.cFunc, self.pt)
        err_in = sup_rel_err(self.tails.cFunc, self.truth.cFunc, self.pt)
        print(
            "HS two-roles top: in-solve %.3e vs eval-only %.3e "
            "(factor %.2fx, floor %.3e)"
            % (err_in, err_eval, err_eval / err_in, self.floor_top)
        )
        self.assertGreater(err_eval, self.floor_top)
        self.assertLess(err_in, err_eval)
        self.assertGreater(err_eval / err_in, GATE_HS_TWO_ROLES_TOP)

    def test_bottom_psi_general_regime_I(self):
        # the GAP-CE-psi closure case: permanent shocks, mNrmMin < 0, worst
        # JOINT atom = the lowest employed-income atom, deep regime I
        self.assertLess(self.truth.mNrmMin, 0.0)
        cf = self.tails.cFunc
        self.assertIsInstance(cf, KappaBarTailInterp)
        self.assertTrue(cf.in_regime)
        reg = ce_psi_regime(
            self.hs[("tails", HS_K)].IncShkDstn[0],
            2.0,
            float((1.01 * 0.98051 * (1.0 - 1.0 / 160.0)) ** 0.5 / (1.0 + 0.01812 / 4)),
        )
        self.assertEqual(reg["regime"], "I")
        self.assertLess(reg["lambda_min_fiber"], 0.5)
        err = sup_rel_err(self.tails.cFunc, self.truth.cFunc, self.pb)
        err_r = sup_rel_err(self.rails.cFunc, self.truth.cFunc, self.pb)
        print(
            "HS bottom (psi-general): tails %.3e rails %.3e (gate %.1e)"
            % (err, err_r, GATE_HS_BOTTOM)
        )
        self.assertLess(err, GATE_HS_BOTTOM)
        self.assertGreater(err_r, 10.0 * err)
        mpc = float(cf.derivative(self.pb[0]))
        dev = abs(mpc / self.tails.MPCmax - 1.0)
        print(
            "HS MPC at me=%.3e: dev %.2e (gate %.1e)"
            % (self.pb[0] - self.tails.mNrmMin, dev, GATE_HS_MPC)
        )
        self.assertLess(dev, GATE_HS_MPC)

    def test_two_roles_bottom(self):
        cf_eval = attach_eval_only(self.rails, self.q_eff, 2.0)
        err_eval = sup_rel_err(cf_eval, self.truth.cFunc, self.pb)
        err_in = sup_rel_err(self.tails.cFunc, self.truth.cFunc, self.pb)
        print(
            "HS two-roles bottom: in-solve %.3e vs eval-only %.3e "
            "(factor %.1fx)" % (err_in, err_eval, err_eval / err_in)
        )
        self.assertLess(err_in, err_eval)
        self.assertGreater(err_eval / err_in, GATE_HS_TWO_ROLES_BOTTOM)

    def test_ladder_monotone(self):
        tails_sh = self.hs[("tails", HS_K_SHALLOW)].solution[0]
        pt_sh = top_window(self.truth, tails_sh)
        err_sh = sup_rel_err(tails_sh.cFunc, self.truth.cFunc, pt_sh)
        err_dp = sup_rel_err(self.tails.cFunc, self.truth.cFunc, self.pt)
        print(
            "HS ladder top: k=%d %.3e <= k=%d %.3e"
            % (HS_K_SHALLOW, err_sh, HS_K, err_dp)
        )
        self.assertLessEqual(err_sh, err_dp)


# ============================================================= regime gating
class TestRegimeGate(unittest.TestCase):
    def test_regime_II_warns_and_refuses(self):
        """A regime-II income process (fat low psi fiber + big zero-income
        atom): the solve warns a ConstraintEndRegimeWarning NAMING
        st-rem-CE-regime and keeps the default bottom segment -- the returned
        cFunc equals the no-option solve exactly."""
        psi = np.array([0.12, 1.1552941176470588])
        pp = np.array([0.15, 0.85])
        th = np.array([0.0, 1.0 / 0.7])
        tp = np.array([0.3, 0.7])
        joint = make_joint_dstn(psi, pp, th, tp)
        grid = log_grid(1e-4, 1e3, 120)

        def make(**opts):
            pars = dict(CE_PARS)
            pars.update(opts)
            a = IndShockConsumerType(**pars)
            a.verbose = 0
            a.IncShkDstn = [joint]
            a.aXtraGrid = grid
            a.tolerance = 1e-8
            return a

        base = make()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base.solve()
        gated = make(decay_extrap_form_lower="kappabar")
        with warnings.catch_warnings(record=True) as wrec:
            warnings.simplefilter("always")
            gated.solve()
        hits = [w for w in wrec if issubclass(w.category, ConstraintEndRegimeWarning)]
        self.assertGreater(len(hits), 0)
        self.assertIn("st-rem-CE-regime", str(hits[0].message))
        self.assertIn("regime II", str(hits[0].message))
        # refusal = the default bottom segment, byte-for-byte
        probes = np.array([1e-4, 1e-2, 0.5, 5.0, 50.0])
        va = np.asarray(base.solution[0].cFunc(probes), float)
        vb = np.asarray(gated.solution[0].cFunc(probes), float)
        self.assertEqual(va.tolist(), vb.tolist())
        self.assertNotIsInstance(gated.solution[0].cFunc, KappaBarTailInterp)

    def test_artificial_constraint_warns_and_refuses(self):
        """With HARK's default BoroCnstArt=0.0 binding (IncUnemp > 0 makes
        BoroCnstNat < 0), the constraint end is a kink with MPC 1: the option
        must warn and keep the default assembly."""
        pars = dict(CE_PARS)
        pars.update(IncUnemp=0.3, BoroCnstArt=0.0)
        base = IndShockConsumerType(**pars)
        base.verbose = 0
        base.aXtraGrid = log_grid(1e-4, 1e3, 120)
        base.tolerance = 1e-8
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base.solve()
        pars2 = dict(pars)
        pars2.update(decay_extrap_form_lower="kappabar")
        gated = IndShockConsumerType(**pars2)
        gated.verbose = 0
        gated.aXtraGrid = log_grid(1e-4, 1e3, 120)
        gated.tolerance = 1e-8
        with warnings.catch_warnings(record=True) as wrec:
            warnings.simplefilter("always")
            gated.solve()
        hits = [w for w in wrec if issubclass(w.category, ConstraintEndRegimeWarning)]
        self.assertGreater(len(hits), 0)
        self.assertIn("artificial borrowing", str(hits[0].message))
        probes = np.array([0.1, 0.5, 5.0, 50.0])
        va = np.asarray(base.solution[0].cFunc(probes), float)
        vb = np.asarray(gated.solution[0].cFunc(probes), float)
        self.assertEqual(va.tolist(), vb.tolist())


# ============================================================== cubic path
class TestCubicPath(unittest.TestCase):
    def test_cubic_both_tails(self):
        """CubicBool=True: the top tail must be the composable DecayTailInterp
        wrap (the cubic interpolant has no powerlaw option) with x_cut at the
        top EGM knot -- required because the assembled cFunc is a LowerEnvelope
        with no x_list."""
        agent = solve_agent(
            CE_PARS,
            log_grid(1e-5, 1e4, 300),
            1e-10,
            CubicBool=True,
            decay_extrap_form="powerlaw",
            decay_extrap_form_lower="kappabar",
        )
        cf = agent.solution[0].cFunc
        self.assertIsInstance(cf, KappaBarTailInterp)
        self.assertIsInstance(cf.interp, DecayTailInterp)
        m_nodes = unwrap_m_nodes(cf)
        self.assertEqual(cf.interp.x_cut, float(m_nodes[-1]))
        self.assertEqual(cf.x_knot, float(m_nodes[1]))
        # level continuity across both attachment points
        for x in (cf.x_knot, cf.interp.x_cut):
            lo = float(cf(x * (1 - 1e-11) if x > 0 else x - 1e-11))
            hi = float(cf(x * (1 + 1e-11) if x > 0 else x + 1e-11))
            self.assertAlmostEqual(lo, hi, places=7)
        # tail MPCs: -> MPCmin from above at the top, -> MPCmax at the bottom
        sol = agent.solution[0]
        top_der = float(cf.derivative(np.array([1e7]))[0])
        self.assertGreater(top_der, sol.MPCmin)
        self.assertLess(top_der, sol.MPCmin * (1.0 + 1e-3))
        bot_der = float(cf.derivative(np.array([1e-9]))[0])
        self.assertAlmostEqual(bot_der, sol.MPCmax, places=9)

    def test_cubic_default_untouched(self):
        a = solve_agent(CE_PARS, log_grid(1e-5, 1e4, 300), 1e-10, CubicBool=True)
        self.assertEqual(type(a.solution[0].cFunc).__name__, "LowerEnvelope")


if __name__ == "__main__":
    unittest.main()

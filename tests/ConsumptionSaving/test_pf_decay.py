"""Tests for the power-law-decay theory utilities (HARK.ConsumptionSaving.pf_decay).

Pre-registered tolerances (declared here BEFORE the assertions were first run;
never weakened). The absolute anchors are the theorem program's calibration
table (theory/powerlaw-decay/figures/FIGURES.md at HAFiscal-Latest @ 71ca7c61),
built from the estimated HAFiscal calibrations HS / CTOP (College top
discount-factor atom) / CCAP (College GIC-cap atom):

    q*        = 0.3813 / 0.6942 / 1.4735      abs tol 5e-4
    B_psi     = 356.63 (CCAP only)            rel tol 5e-4 (0.05%)
    lambda_B  = 1.0096 / 1.0026 / 0.9978      abs tol 1e-3 (all near-resonance)
    E[psi^2]  = 1.002492                      abs tol 1e-5
    zeta*     = 9.19 / 3.17 / None            rel tol 1%
    E[ln A]   = -0.0113 / -0.0039 / +0.0009   abs tol 1e-4

This makes the HAFiscal calibration table a permanent HARK regression anchor
for the theory utility.

# THEOREM-REF[BufferStockTheory-Latest @ c181870f :: theory/powerlaw-decay/figures/FIGURES.md :: Calibrations]
#   Anchor provenance: the theorem program's calibration table (CAL-HS /
#   CAL-CTOP / CAL-CCAP, built from the estimated HAFiscal parameters) is the
#   source of the q*, B_psi, lambda_B and E[psi^2] targets above.

# THEOREM-REF[BufferStockTheory-Latest @ c181870f :: theory/powerlaw-decay/final_proof.md :: §6.1 How rarely is the tail visited? The dual (Kesten) root, for economists :: The reachability taxonomy :: https://llorracc.github.io/BufferStockTheory-Latest/powerlaw-decay-theory/]
#   The zeta* and E[ln A] targets above are the measured reachability table:
#   dual root 9.19 (HS) / 3.17 (College top atom) / none (GIC-cap atom, whose
#   positive log-drift leaves mortality-with-replacement to truncate its tail).
"""

import pickle
import unittest
import warnings

import numpy as np
from scipy.stats import norm

from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    PerfForesightConsumerType,
)
from HARK.ConsumptionSaving.pf_decay import (
    NearResonanceWarning,
    NoDualRootWarning,
    PFDecayConditionWarning,
    ShockCorrelationWarning,
    aXtraMax_from_tail_tol,
    aXtraMax_from_wealth_mass,
    dual_root,
    mNrm_stable_points,
    powerlaw_decay_params,
    powerlaw_decay_params_from_agent,
    powerlaw_tail_diagnostic,
    qstar_probe,
    rel_gap_at,
    resonance_constants,
)
from HARK.distributions import DiscreteDistribution

# ------------------------------------------------------------------ fixtures
# Discretizers are TEST FIXTURES, not shipped API: HARK users bring their own
# discretized distributions. These replicate the theorem program's 7-atom
# construction (equiprobable mean-one lognormal; unemployment atom with the
# employed atoms rescaled so the total mean stays 1).


def lognormal_equiprob(sigma_log, N):
    if sigma_log <= 0 or N == 1:
        return np.array([1.0]), np.array([1.0])
    u = (np.arange(N) + 0.5) / N
    z = norm.ppf(u)
    a = np.exp(sigma_log * z - 0.5 * sigma_log**2)
    p = np.full(N, 1.0 / N)
    a = a / (p * a).sum()
    return a, p


def with_unemp_atom(sigma_log, N, unemp_prob, inc_unemp=0.0):
    th_e, p_e = lognormal_equiprob(sigma_log, N)
    scale = (1.0 - unemp_prob * inc_unemp) / (1.0 - unemp_prob)
    th = np.concatenate(([inc_unemp], th_e * scale))
    p = np.concatenate(([unemp_prob], p_e * (1.0 - unemp_prob)))
    return th, p


R0, RHO, LIV = 1.01, 2.0, 1.0 - 1.0 / 160.0
PSI, PP = lognormal_equiprob(np.sqrt(0.003), 7)
TH_H, TP_H = with_unemp_atom(np.sqrt(0.12), 7, 0.044, 0.7)
TH_C, TP_C = with_unemp_atom(np.sqrt(0.12), 7, 0.027, 0.7)
G_H = 1.0 + 0.01812 / 4
G_C = 1.0 + 0.01958 / 4
BETA_HS = 0.98051
BETA_CTOP = 0.98680 + (0.99640 - 0.98680) * 13.0 / 14.0
BETA_CCAP = 1.005375

CALS = {
    "HS": dict(G=G_H, beta=BETA_HS, th=TH_H, tp=TP_H),
    "CTOP": dict(G=G_C, beta=BETA_CTOP, th=TH_C, tp=TP_C),
    "CCAP": dict(G=G_C, beta=BETA_CCAP, th=TH_C, tp=TP_C),
}
TARGETS = {
    "HS": dict(q=0.3813, lamB=1.0096, zeta=9.19, ElnA=-0.0113),
    "CTOP": dict(q=0.6942, lamB=1.0026, zeta=3.17, ElnA=-0.0039),
    "CCAP": dict(q=1.4735, lamB=0.9978, zeta=None, ElnA=+0.0009),
}
B_PSI_TARGET = 356.63
E_PSI2_TARGET = 1.002492


def _params(tag, warn=False):
    c = CALS[tag]
    return powerlaw_decay_params(
        R0, c["G"], c["beta"], RHO, LivPrb=LIV,
        PermShkDstn=(PSI, PP), TranShkDstn=(c["th"], c["tp"]), warn=warn,
    )


class TestCalibrationAnchors(unittest.TestCase):
    """FIGURES.md calibration table as a permanent regression anchor
    (tolerances pre-registered in the module docstring above)."""

    @classmethod
    def setUpClass(cls):
        cls.res = {tag: _params(tag) for tag in CALS}

    def test_q_star(self):
        for tag in CALS:
            self.assertLessEqual(
                abs(self.res[tag].q_star - TARGETS[tag]["q"]), 5e-4, tag
            )

    def test_realized_exponent_is_min_one_qstar(self):
        self.assertAlmostEqual(self.res["HS"].q, self.res["HS"].q_star, places=14)
        self.assertEqual(self.res["CCAP"].q, 1.0)

    def test_B_psi(self):
        # THEOREM-REF[BufferStockTheory-Latest @ c181870f :: theory/powerlaw-decay/alt_proof_compactified.md :: Theorem γ-B (Stage-B boundary value) :: https://llorracc.github.io/BufferStockTheory-Latest/powerlaw-decay-theory/alt-proof-compactified/]
        #   The closed-form boundary amplitude B_psi exists only at q* > 1; at the
        #   GIC-cap calibration B_psi = 356.63. At q* < 1 the Gordon denominator
        #   Rcal*Thorn_Gamma - E[psi^2] is negative and B_psi must be refused.
        self.assertIsNotNone(self.res["CCAP"].B_psi)
        self.assertLessEqual(abs(self.res["CCAP"].B_psi / B_PSI_TARGET - 1.0), 5e-4)
        self.assertIsNone(self.res["HS"].B_psi)
        self.assertIsNone(self.res["CTOP"].B_psi)

    def test_lambda_B_and_near_resonance(self):
        for tag in CALS:
            r = self.res[tag]
            self.assertLessEqual(abs(r.lambda_B - TARGETS[tag]["lamB"]), 1e-3, tag)
            self.assertTrue(r.near_resonance, tag)
            self.assertTrue(any("NEAR-RESONANCE" in w for w in r.warnings), tag)

    def test_E_psi2(self):
        for tag in CALS:
            self.assertLessEqual(
                abs(self.res[tag].E_psi2 - E_PSI2_TARGET), 1e-5, tag
            )

    def test_zeta_star(self):
        for tag in ("HS", "CTOP"):
            r = self.res[tag]
            self.assertIsNotNone(r.zeta_star, tag)
            self.assertLessEqual(
                abs(r.zeta_star / TARGETS[tag]["zeta"] - 1.0), 0.01, tag
            )
            self.assertEqual(r.dual_diagnosis, "ok", tag)
        rK = self.res["CCAP"]
        self.assertIsNone(rK.zeta_star)
        self.assertIn("positive log-drift", rK.dual_diagnosis)

    def test_E_ln_A(self):
        for tag in CALS:
            self.assertLessEqual(
                abs(self.res[tag].E_ln_A - TARGETS[tag]["ElnA"]), 1e-4, tag
            )

    def test_valid_flags(self):
        # All three estimated calibrations satisfy GIC/RIC/FHWC — the GIC cap
        # shaves beta to sit just BELOW the GIC boundary (Lambda > 0 but tiny;
        # it is the DUAL root that fails at CCAP via positive log-drift, not GIC)
        for tag in CALS:
            self.assertTrue(self.res[tag].valid, tag)
        self.assertGreater(self.res["CCAP"].Lambda, 0.0)
        self.assertLess(self.res["CCAP"].Lambda, 1e-3)


class TestHConvention(unittest.TestCase):
    """The two-human-wealth-conventions fact, pinned as tripwires.

    # THEOREM-REF[BufferStockTheory-Latest @ c181870f :: theory/powerlaw-decay/ADVERSARIAL_TESTING_GUIDE.md :: 5. LANDMINES — documented evaluation traps and silent-pass hazards :: The `h` human-wealth convention]
    #   Theorem h = 1/(Rcal-1) EXCLUDES current income = h_BST - 1 with
    #   h_BST = R/(R-Gamma). HARK's SOLVER-side hNrm (calc_human_wealth) matches
    #   the theorem convention (h*E_inc); bilt['hNrm'] from calc_limiting_values
    #   is BST-convention. These tests pin BOTH sides so a future "unification"
    #   of the two conventions trips loudly instead of silently moving one.
    """

    def test_h_equals_hBST_minus_one(self):
        # pre-registered tol: rel 1e-14 (pure algebra in longdouble)
        LD = np.longdouble
        for tag, c in CALS.items():
            r = _params(tag)
            Rcal = LD(R0) / LD(c["G"])
            h_BST = Rcal / (Rcal - 1)  # BST convention: INCLUDES current income
            dev = abs(float(LD(r.h) / (h_BST - 1)) - 1.0)
            self.assertLessEqual(dev, 1e-14, tag)

    def test_solver_hNrm_excludes_current_income(self):
        # cycles=1: exactly one period of future income, so the solver-side
        # convention is directly observable: hNrm == (G/R)*E_inc (rel 1e-10).
        agent = IndShockConsumerType(cycles=1)
        agent.solve()
        R = float(np.asarray(agent.Rfree).flat[0])
        G = float(np.asarray(agent.PermGroFac).flat[0])
        d0 = agent.IncShkDstn[0]
        E_inc = float(
            np.sum(np.asarray(d0.pmv) * np.asarray(d0.atoms[0]) * np.asarray(d0.atoms[1]))
        )
        h1 = float(agent.solution[0].hNrm)
        self.assertLessEqual(abs(h1 - (G / R) * E_inc), 1e-10 * max(1.0, h1))

    def test_bilt_hNrm_is_the_OTHER_convention_tripwire(self):
        # calc_limiting_values' bilt['hNrm'] must equal R/(R-G) — the BST
        # convention that INCLUDES current income (PF E_inc = 1). If a future
        # refactor "unifies" it with solution.hNrm (which excludes current
        # income), this assertion trips and the theory utility's compute-from-
        # primitives rule must be re-audited.
        agent = PerfForesightConsumerType(cycles=0)
        if not hasattr(agent, "bilt"):
            agent.bilt = {}
        agent.calc_limiting_values()
        R = float(np.asarray(agent.Rfree).flat[0])
        G = float(np.asarray(agent.PermGroFac).flat[0])
        h_bilt = float(agent.bilt["hNrm"])
        h_bst = R / (R - G)
        self.assertLessEqual(abs(h_bilt - h_bst), 1e-10 * h_bst)
        # and it exceeds the solver/theorem convention by exactly E_inc = 1
        self.assertGreater(h_bilt, h_bst - 1.0 + 0.5)


class TestDegenerateAndConditions(unittest.TestCase):
    def test_psi_equals_one_closed_form(self):
        # Stage A: q* = ln(Rcal)/Lambda exactly (tol 1e-12 relative)
        r = powerlaw_decay_params(
            R0, G_H, BETA_HS, RHO, LivPrb=LIV,
            PermShkDstn=None, TranShkDstn=(TH_H, TP_H), warn=False,
        )
        q_closed = np.log(r.Rcal) / r.Lambda
        self.assertLessEqual(abs(r.q_star - q_closed), 1e-12 * max(1.0, q_closed))
        self.assertEqual(r.E_psi2, 1.0)
        self.assertLessEqual(abs(r.sigma_B2 / r.Var_theta - 1.0), 1e-12)
        self.assertIsNone(r.zeta_star)
        self.assertEqual(r.P_A_gt_1, 0.0)
        self.assertIn("compact", r.dual_diagnosis)

    def test_fhwc_refusal(self):
        # FHWC violated (R <= G): clean refusal, no exception
        with self.assertWarns(PFDecayConditionWarning):
            r = powerlaw_decay_params(
                1.0, G_C, 0.98, RHO, LivPrb=LIV,
                PermShkDstn=(PSI, PP), TranShkDstn=(TH_C, TP_C),
            )
        self.assertFalse(r.FHWC)
        self.assertFalse(r.valid)
        self.assertTrue(np.isnan(r.h))
        self.assertIsNone(r.sigma_B2)
        self.assertIsNone(r.B_psi)
        self.assertIsNone(r.c_J)
        self.assertTrue(np.isnan(r.q_star))
        self.assertIn("FHWC", r.diagnosis)
        self.assertTrue(any("FHWC" in w for w in r.warnings))

    def test_gic_refusal_psi_one(self):
        # GIC violated with psi == 1: L(q) non-increasing, no (E)-root
        with self.assertWarns(PFDecayConditionWarning):
            r = powerlaw_decay_params(
                R0, G_C, 1.008, RHO, LivPrb=LIV,
                PermShkDstn=None, TranShkDstn=(TH_C, TP_C),
            )
        self.assertFalse(r.GIC)
        self.assertFalse(r.valid)
        self.assertTrue(r.RIC)
        self.assertTrue(np.isnan(r.q_star))
        self.assertNotEqual(r.diagnosis, "")
        self.assertTrue(any("GIC" in w for w in r.warnings))

    def test_gic_violated_wide_psi_still_reports_root(self):
        # With a psi spread the convex L(q) can still cross zero even though
        # GIC fails; the root is reported (with valid=False) and satisfies (E)
        # to 1e-10.
        r = powerlaw_decay_params(
            R0, G_C, 1.008, RHO, LivPrb=LIV,
            PermShkDstn=(PSI, PP), TranShkDstn=(TH_C, TP_C), warn=False,
        )
        self.assertTrue(np.isfinite(r.q_star))
        self.assertGreater(r.q_star, 0.0)
        psf, ppf = np.asarray(PSI, float), np.asarray(PP, float)
        resid = abs(
            float(np.dot(ppf, psf ** (1.0 + r.q_star)))
            - r.Rcal * r.Thorn_Gamma**r.q_star
        )
        self.assertLessEqual(resid, 1e-10)
        self.assertFalse(r.valid)

    def test_ric_violation_warns(self):
        with self.assertWarns(PFDecayConditionWarning):
            r = powerlaw_decay_params(
                1.01, 1.0, 1.03, RHO, LivPrb=1.0,
                PermShkDstn=(PSI, PP), TranShkDstn=(TH_C, TP_C),
            )
        self.assertFalse(r.RIC)
        self.assertLessEqual(r.kappa, 0.0)
        self.assertFalse(r.valid)
        self.assertTrue(any("RIC" in w for w in r.warnings))

    def test_qstar_beyond_bracket_cap_is_finite_not_false_gic(self):
        # Regression: a LEGITIMATE calibration (GIC/RIC/FHWC all hold) whose
        # (E)-root exceeds the bracket-expansion cap. psi == 1 with Lambda ~ 1e-6
        # gives analytic q* = ln(Rcal)/Lambda ~ 4963 >> the historical 1024 cap.
        # The old fixed cap returned nan + a FALSE 'GIC violated' diagnosis, which
        # poisoned the realized exponent (should be min(1, q*) = 1) and dropped B_psi.
        import math
        R, G, rho = 1.01, 1.005, 1.0            # rho=1 => Thorn = R*beta_eff
        Thorn_Gamma = math.exp(-1e-6)           # Lambda = 1e-6, deep near-resonance
        beta = (Thorn_Gamma * G) / R
        r = powerlaw_decay_params(
            R, G, beta, rho, LivPrb=1.0,
            PermShkDstn=None, TranShkDstn=(TH_C, TP_C), warn=False,
        )
        self.assertTrue(r.valid)                # GIC, RIC, FHWC all hold
        self.assertTrue(np.isfinite(r.q_star))
        self.assertAlmostEqual(r.q_star, math.log(R / G) / 1e-6, delta=1e-2)
        self.assertGreater(r.q_star, 1024.0)    # genuinely beyond the historical cap
        self.assertEqual(r.q, 1.0)              # realized exponent min(1, q*) = 1
        self.assertIsNotNone(r.B_psi)           # q* > 1 and denom > 0 => B_psi defined
        self.assertEqual(r.diagnosis, "")       # no false condition claim

    def test_dual_root_beyond_bracket_cap_is_finite(self):
        # Regression: a genuine Kesten root zeta* beyond the bracket cap. A single
        # marginally-expanding psi atom (A = Thorn_Gamma/psi just above 1) with
        # E[ln A] < 0 has a real but enormous Pareto exponent; the old fixed 1024
        # cap returned a false None ('bracket cap hit') for a tail that exists, and
        # the inline 'not reachable for finite atoms' comment was factually wrong.
        R, G, rho = 1.01, 1.0, 2.0
        Thorn_Gamma = 0.9999
        beta = (Thorn_Gamma * G) ** rho / R     # Thorn = (R*beta)^(1/rho) = Thorn_Gamma*G
        psi = np.array([0.9995, 1.0000102])
        pr = np.array([0.02, 0.98])
        psi = psi / float((pr * psi).sum())     # exact mean 1
        r = powerlaw_decay_params(
            R, G, beta, rho, LivPrb=1.0,
            PermShkDstn=(psi, pr), TranShkDstn=(TH_C, TP_C), warn=False,
        )
        self.assertIsNotNone(r.zeta_star)
        self.assertGreater(r.zeta_star, 1024.0)  # genuinely beyond the historical cap
        self.assertLess(r.E_ln_A, 0.0)           # contracts on average
        self.assertGreater(r.P_A_gt_1, 0.0)      # occasionally expands
        A = Thorn_Gamma / psi                    # satisfies E[(Thorn_Gamma/psi)^zeta] = 1
        self.assertAlmostEqual(float(np.dot(pr, A ** r.zeta_star)), 1.0, places=6)


class TestInputAcceptance(unittest.TestCase):
    def test_hark_discrete_distribution_objects(self):
        r_tuple = _params("HS")
        psi_dd = DiscreteDistribution(np.asarray(PP, float), np.asarray(PSI, float))
        th_dd = DiscreteDistribution(np.asarray(TP_H, float), np.asarray(TH_H, float))
        r_dd = powerlaw_decay_params(
            R0, G_H, BETA_HS, RHO, LivPrb=LIV,
            PermShkDstn=psi_dd, TranShkDstn=th_dd, warn=False,
        )
        self.assertLessEqual(abs(r_dd.q_star / r_tuple.q_star - 1.0), 1e-12)
        self.assertLessEqual(abs(r_dd.sigma_B2 / r_tuple.sigma_B2 - 1.0), 1e-12)

    def test_joint_equals_marginals(self):
        r_tuple = _params("HS")
        PSIJ, THJ = np.meshgrid(
            np.asarray(PSI, float), np.asarray(TH_H, float), indexing="ij"
        )
        PJ = np.outer(np.asarray(PP, float), np.asarray(TP_H, float)).ravel()
        joint = (np.vstack([PSIJ.ravel(), THJ.ravel()]), PJ)
        r_joint = powerlaw_decay_params(
            R0, G_H, BETA_HS, RHO, LivPrb=LIV, IncShkDstn=joint, warn=False
        )
        self.assertLessEqual(abs(r_joint.q_star / r_tuple.q_star - 1.0), 1e-10)
        self.assertLessEqual(abs(r_joint.sigma_B2 / r_tuple.sigma_B2 - 1.0), 1e-10)

    def test_time_varying_one_element_lists(self):
        r_tuple = _params("HS")
        r_list = powerlaw_decay_params(
            R0, G_H, BETA_HS, RHO, LivPrb=LIV,
            PermShkDstn=[(PSI, PP)], TranShkDstn=[(TH_H, TP_H)], warn=False,
        )
        self.assertEqual(r_list.q_star, r_tuple.q_star)

    def test_error_taxonomy(self):
        joint = (np.vstack([PSI, PSI]), PP)
        with self.assertRaises(ValueError):  # both joint and marginals
            powerlaw_decay_params(
                R0, G_H, BETA_HS, RHO, PermShkDstn=(PSI, PP), IncShkDstn=joint
            )
        with self.assertRaises(ValueError):  # probs don't sum to 1
            powerlaw_decay_params(
                R0, G_H, BETA_HS, RHO, PermShkDstn=(PSI, PP * 0.5),
                TranShkDstn=(TH_H, TP_H),
            )
        with self.assertRaises(ValueError):  # negative probability
            bad_p = PP.copy()
            bad_p[0], bad_p[1] = -bad_p[1], bad_p[0] + 2 * bad_p[1]
            powerlaw_decay_params(
                R0, G_H, BETA_HS, RHO, PermShkDstn=(PSI, bad_p),
                TranShkDstn=(TH_H, TP_H),
            )
        with self.assertRaises(ValueError):  # psi atom <= 0
            powerlaw_decay_params(
                R0, G_H, BETA_HS, RHO,
                PermShkDstn=(np.array([0.0, 2.0]), np.array([0.5, 0.5])),
                TranShkDstn=(TH_H, TP_H),
            )
        with self.assertRaises(ValueError):  # joint atoms wrong shape
            powerlaw_decay_params(
                R0, G_H, BETA_HS, RHO,
                IncShkDstn=(np.vstack([PSI, PSI, PSI]), PP),
            )
        with self.assertRaises(ValueError):  # non-positive scalar
            powerlaw_decay_params(-1.0, G_H, BETA_HS, RHO)

    def test_correlated_joint_warns_independent_does_not(self):
        # perfectly anticorrelated 2-atom joint: E[psi]=E[theta]=1 but
        # E[psi*theta] = 0.99 != 1
        atoms = np.array([[0.9, 1.1], [1.1, 0.9]])
        with self.assertWarns(ShockCorrelationWarning):
            powerlaw_decay_params(
                R0, G_H, BETA_HS, RHO, LivPrb=LIV,
                IncShkDstn=(atoms, np.array([0.5, 0.5])),
            )
        # independent outer product must NOT warn about correlation
        PSIJ, THJ = np.meshgrid(
            np.asarray(PSI, float), np.asarray(TH_H, float), indexing="ij"
        )
        PJ = np.outer(np.asarray(PP, float), np.asarray(TP_H, float)).ravel()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            powerlaw_decay_params(
                R0, G_H, BETA_HS, RHO, LivPrb=LIV,
                IncShkDstn=(np.vstack([PSIJ.ravel(), THJ.ravel()]), PJ),
            )
        self.assertFalse(
            any(issubclass(w.category, ShockCorrelationWarning) for w in caught)
        )

    def test_frozen_and_picklable(self):
        r = _params("HS")
        with self.assertRaises(Exception):  # frozen dataclass
            r.q_star = 0.5
        r2 = pickle.loads(pickle.dumps(r))
        self.assertEqual(r2.q_star, r.q_star)
        self.assertEqual(r2.warnings, r.warnings)


class TestWarningFilters(unittest.TestCase):
    def test_near_resonance_warning_fires_and_is_filterable(self):
        with self.assertWarns(NearResonanceWarning):
            _params("HS", warn=True)
        # production log hygiene: filtering the dedicated category silences it
        # WITHOUT touching other UserWarnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.simplefilter("ignore", category=NearResonanceWarning)
            r = powerlaw_decay_params(
                R0, G_C, BETA_CCAP, RHO, LivPrb=LIV,
                PermShkDstn=(PSI, PP), TranShkDstn=(TH_C, TP_C),
            )
        cats = [w.category for w in caught]
        self.assertFalse(any(issubclass(c, NearResonanceWarning) for c in cats))
        # CCAP's no-dual-root warning (a different category) still comes through
        self.assertTrue(any(issubclass(c, NoDualRootWarning) for c in cats))
        # and the record on the result object is complete regardless
        self.assertTrue(any("NEAR-RESONANCE" in w for w in r.warnings))

    def test_warn_false_emits_nothing_but_records(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = _params("CCAP", warn=False)
        self.assertEqual(len(caught), 0)
        # CCAP records near-resonance + no-dual-root even with warn=False
        self.assertTrue(any("NEAR-RESONANCE" in w for w in r.warnings))
        self.assertTrue(any("NO DUAL ROOT" in w for w in r.warnings))


class TestFromAgent(unittest.TestCase):
    def test_matches_direct_call(self):
        agent = IndShockConsumerType(cycles=0)
        r_agent = powerlaw_decay_params_from_agent(agent, warn=False)
        d_perm, d_tran = agent.PermShkDstn[0], agent.TranShkDstn[0]
        r_direct = powerlaw_decay_params(
            np.asarray(agent.Rfree).flat[0],
            np.asarray(agent.PermGroFac).flat[0],
            agent.DiscFac,
            agent.CRRA,
            LivPrb=np.asarray(agent.LivPrb).flat[0],
            PermShkDstn=d_perm,
            TranShkDstn=d_tran,
            warn=False,
        )
        self.assertEqual(r_agent.q_star, r_direct.q_star)
        self.assertEqual(r_agent.sigma_B2, r_direct.sigma_B2)

    def test_joint_fallback(self):
        # an agent-like object with only a joint IncShkDstn (e.g. a hand-built
        # income process) exercises the fallback path
        PSIJ, THJ = np.meshgrid(
            np.asarray(PSI, float), np.asarray(TH_H, float), indexing="ij"
        )
        PJ = np.outer(np.asarray(PP, float), np.asarray(TP_H, float)).ravel()

        class _Stub:
            Rfree = [R0]
            PermGroFac = [G_H]
            DiscFac = BETA_HS
            CRRA = RHO
            LivPrb = [LIV]
            IncShkDstn = [(np.vstack([PSIJ.ravel(), THJ.ravel()]), PJ)]

        r_stub = powerlaw_decay_params_from_agent(_Stub(), warn=False)
        r_marg = _params("HS")
        self.assertLessEqual(abs(r_stub.q_star / r_marg.q_star - 1.0), 1e-10)


class TestResonanceHelper(unittest.TestCase):
    def test_resonance_constants_at_root_found_beta(self):
        """Root-find beta so q* = 1 exactly on the College fundamentals, then
        check the theorem-program targets: C_B = 176.37 and cJ/Rcal = 1.1128
        (both rel tol 0.5%), plus the exact-resonance identity
        C_B * Lprime(1) == cJ/Rcal (rel tol 1e-10)."""

        def qstar_of(beta):
            return powerlaw_decay_params(
                R0, G_C, beta, RHO, LivPrb=LIV,
                PermShkDstn=(PSI, PP), TranShkDstn=(TH_C, TP_C), warn=False,
            ).q_star

        lo, hi = 0.99, 1.01
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if mid == lo or mid == hi:
                break
            if qstar_of(mid) < 1.0:
                lo = mid
            else:
                hi = mid
        beta_res = 0.5 * (lo + hi)
        self.assertLessEqual(abs(qstar_of(beta_res) - 1.0), 1e-8)
        rc = resonance_constants(
            R0, G_C, beta_res, RHO, LivPrb=LIV,
            PermShkDstn=(PSI, PP), TranShkDstn=(TH_C, TP_C), warn=False,
        )
        self.assertLessEqual(abs(rc["C_B"] / 176.37 - 1.0), 5e-3)
        self.assertLessEqual(abs(rc["cJ_over_Rcal"] / 1.1128 - 1.0), 5e-3)
        self.assertLessEqual(
            abs(rc["C_B"] * rc["Lprime1"] / rc["cJ_over_Rcal"] - 1.0), 1e-10
        )
        self.assertLessEqual(rc["resonance_residual"], 1e-6)


class TestTailDiagnostic(unittest.TestCase):
    """powerlaw_tail_diagnostic: the Theorem gamma-T compensated-flatness test
    as a post-solve grid diagnostic.

    Pre-registered expectations (measured during development, thresholds from
    the function's docstring, never tuned):
      * fine real solve (aXtraMax=1e5, aXtraCount=512), window [3e3, 6e4]:
        center slope(s=q) measured +0.046 -> CONFIRMED (flat_tol 0.08);
      * coarse real solve (aXtraCount=96), window [2e3, 2e4]: interpolation
        bias flattens the local exponent (center +0.23) -> PRE_ASYMPTOTIC;
        the healthy-solve contract is verdict in {CONFIRMED, PRE_ASYMPTOTIC},
        NEVER INCONSISTENT (no false positive);
      * beyond the solved grid the gap collapses below the float-cancellation
        guard -> UNMEASURABLE;
      * the h-convention trap (+E_inc on hNrm) turns the measured gap into
        gap_true + kappa*E_inc. On SYNTHETIC theorem-form data with a window
        where the constant dominates, the flat point sits near s=0, far from
        q -> INCONSISTENT (the clean signature; pinned on synthetic data
        because on a REAL solve at reachable windows the constant only
        partially dominates and the poisoning mimics a transient --
        PRE_ASYMPTOTIC with a visibly degraded center, asserted below).
    """

    @classmethod
    def setUpClass(cls):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.agent_fine = IndShockConsumerType(
                cycles=0, aXtraMax=1.0e5, aXtraCount=512, aXtraNestFac=1
            )
            cls.agent_fine.solve()
            cls.agent_coarse = IndShockConsumerType(
                cycles=0, aXtraMax=1.0e5, aXtraCount=96, aXtraNestFac=3
            )
            cls.agent_coarse.solve()
        cls.params = powerlaw_decay_params_from_agent(cls.agent_fine, warn=False)
        cls.kappa = cls.params.kappa
        cls.hN = cls.params.h * cls.params.E_inc  # theorem-convention, primitives
        cls.q = cls.params.q

    # ---------------- real-solve verdicts ----------------
    def test_confirmed_on_fine_real_solve(self):
        d = powerlaw_tail_diagnostic(
            self.agent_fine.solution[0].cFunc, self.kappa, self.hN, self.params,
            m_lo=3e3, m_hi=6e4,
        )
        self.assertEqual(d.verdict, "CONFIRMED")
        self.assertEqual(d.n_points, 40)
        # probe drifts carry the gamma-T signs: slope increasing through 0 at q
        center = d.slopes[list(d.s_grid).index(self.q)]
        self.assertLessEqual(abs(center), 0.08)
        self.assertLess(d.slopes[0], d.slopes[-1])

    def test_no_false_positive_on_coarse_real_solve(self):
        d = powerlaw_tail_diagnostic(
            self.agent_coarse.solution[0].cFunc, self.kappa, self.hN,
            self.params, m_lo=2e3, m_hi=2e4,
        )
        self.assertIn(d.verdict, ("CONFIRMED", "PRE_ASYMPTOTIC"))

    def test_unmeasurable_beyond_the_grid(self):
        d = powerlaw_tail_diagnostic(
            self.agent_fine.solution[0].cFunc, self.kappa, self.hN, self.params,
            m_lo=1e6, m_hi=1e8,
        )
        self.assertEqual(d.verdict, "UNMEASURABLE")
        self.assertLess(d.n_points, 20)

    def test_wrong_hNrm_degrades_the_center_on_a_real_solve(self):
        # +E_inc (the h_BST trap): within a reachable window the constant
        # poisoning mimics a transient; the verdict must NOT be CONFIRMED and
        # the center must move away from flat by the poisoning
        cF = self.agent_fine.solution[0].cFunc
        d_right = powerlaw_tail_diagnostic(cF, self.kappa, self.hN, self.params,
                                           m_lo=3e3, m_hi=6e4)
        d_wrong = powerlaw_tail_diagnostic(
            cF, self.kappa, self.hN + self.params.E_inc, self.params,
            m_lo=3e3, m_hi=6e4,
        )
        self.assertEqual(d_right.verdict, "CONFIRMED")
        self.assertNotEqual(d_wrong.verdict, "CONFIRMED")
        c_right = d_right.slopes[list(d_right.s_grid).index(self.q)]
        c_wrong = d_wrong.slopes[list(d_wrong.s_grid).index(self.q)]
        self.assertGreater(c_wrong, c_right + 0.1)

    # ---------------- synthetic theorem-form verdict logic ----------------
    def _synthetic(self, expo, C=2.0):
        kappa, hN = self.kappa, self.hN

        def cF(m):
            x = m + hN
            return kappa * x - C * x ** (-expo)

        return cF

    def test_confirmed_on_exact_theorem_form(self):
        d = powerlaw_tail_diagnostic(
            self._synthetic(self.q), self.kappa, self.hN, self.params,
            m_lo=50.0, m_hi=5000.0,
        )
        self.assertEqual(d.verdict, "CONFIRMED")
        center = d.slopes[list(d.s_grid).index(self.q)]
        self.assertLessEqual(abs(center), 1e-6)

    def test_pre_asymptotic_on_shallow_transient(self):
        # local exponent q - 0.3 (below q): the theorem-backed transient side
        d = powerlaw_tail_diagnostic(
            self._synthetic(self.q - 0.3), self.kappa, self.hN, self.params,
            m_lo=50.0, m_hi=5000.0,
        )
        self.assertEqual(d.verdict, "PRE_ASYMPTOTIC")

    def test_inconsistent_on_steeper_than_floor(self):
        # local exponent q + 0.6: steeper than min(1, q*) -- the Prop-A0 side
        d = powerlaw_tail_diagnostic(
            self._synthetic(self.q + 0.6), self.kappa, self.hN, self.params,
            m_lo=50.0, m_hi=5000.0,
        )
        self.assertEqual(d.verdict, "INCONSISTENT")
        self.assertIn("STEEPER", d.notes)

    def test_inconsistent_in_steeper_deadband(self):
        # Regression (docstring/code alignment): a local exponent in the band
        # (q - 2*flat_tol, q - flat_tol), i.e. center in (-0.16, -0.08) at the
        # default flat_tol=0.08. The old docstring implied INCONSISTENT only for
        # center < -2*flat_tol, leaving this band undocumented; the code (and now
        # the docstring) classify ANY steeper-than-CONFIRMED center as INCONSISTENT
        # (Prop A0: no true transient decays faster than min(1, q*)).
        d = powerlaw_tail_diagnostic(
            self._synthetic(self.q + 0.12), self.kappa, self.hN, self.params,
            m_lo=50.0, m_hi=5000.0,
        )
        center = d.slopes[list(d.s_grid).index(self.q)]
        self.assertLess(center, -0.08)          # steeper than the CONFIRMED band
        self.assertGreater(center, -0.16)       # inside the old-docstring dead-band
        self.assertEqual(d.verdict, "INCONSISTENT")
        self.assertIn("STEEPER", d.notes)

    def test_inconsistent_on_h_convention_trap_synthetic(self):
        # exact theorem-form data measured with hNrm + E_inc: the gap becomes
        # gap_true + kappa*E_inc; on a window where the constant dominates the
        # flat point sits near s = 0, far from q (q = 0.5923 > the 0.5
        # inconsistency_tol) -> INCONSISTENT, the trap detected
        self.assertGreater(self.q, 0.5)  # precondition for the far-flat-point
        d = powerlaw_tail_diagnostic(
            self._synthetic(self.q), self.kappa, self.hN + self.params.E_inc,
            self.params, m_lo=1e4, m_hi=1e6,
        )
        self.assertEqual(d.verdict, "INCONSISTENT")
        self.assertIn("wrong-exponent signature", d.notes)

    # ---------------- refusal / plumbing paths ----------------
    def test_unmeasurable_when_qstar_nan(self):
        bad = powerlaw_decay_params(
            1.0, G_C, 0.98, RHO, LivPrb=LIV,
            PermShkDstn=(PSI, PP), TranShkDstn=(TH_C, TP_C), warn=False,
        )
        d = powerlaw_tail_diagnostic(
            self._synthetic(0.5), self.kappa, self.hN, bad, m_lo=50.0, m_hi=5e3
        )
        self.assertEqual(d.verdict, "UNMEASURABLE")
        self.assertIn("q_star is nan", d.notes)

    def test_m_hi_discovery_and_requirement(self):
        # bare-callable cFunc without x_list: m_hi is required
        with self.assertRaises(ValueError):
            powerlaw_tail_diagnostic(
                self._synthetic(self.q), self.kappa, self.hN, self.params
            )
        # a LinearInterp slice carries x_list: m_hi defaults to half its top
        from HARK.interpolation import LinearInterp

        m = np.geomspace(1.0, 4.0e4, 400)
        f = LinearInterp(m, self._synthetic(self.q)(m))
        d = powerlaw_tail_diagnostic(f, self.kappa, self.hN, self.params)
        self.assertEqual(d.verdict, "CONFIRMED")
        self.assertLessEqual(d.window[1], 0.5 * 4.0e4)

    def test_trial_offsets_must_include_zero(self):
        with self.assertRaises(ValueError):
            powerlaw_tail_diagnostic(
                self._synthetic(self.q), self.kappa, self.hN, self.params,
                m_lo=50.0, m_hi=5e3, trial_offsets=(-0.15, 0.15),
            )


class TestExtentCriterion(unittest.TestCase):
    """aXtraMax_from_tail_tol + rel_gap_at: the certified grid-extent
    criterion (the surviving deliverable of the grid-placement P1 program —
    the placement scheme itself was KILLED by its pre-registered gates; see
    theory/powerlaw-decay/grid_placement_p1_frontier_of_failure.md in the
    HAFiscal-Latest repository)."""

    def test_inversion_reproduces_tolerance_on_synthetic_gap(self):
        # pure power law: rel_gap(x) = rel_ref * (x/x_ref)^(-(1+q))
        q, hN, m_ref, rel_ref, tol = 0.4, 150.0, 40.0, 0.5, 1e-4
        m_top = aXtraMax_from_tail_tol(m_ref, rel_ref, q, hN, tol, safety=1.0)
        rel_at_top = rel_ref * ((m_top + hN) / (m_ref + hN)) ** (-(1.0 + q))
        self.assertAlmostEqual(rel_at_top / tol, 1.0, places=10)

    def test_safety_extends_the_top(self):
        args = (40.0, 0.5, 0.4, 150.0, 1e-4)
        self.assertGreater(
            aXtraMax_from_tail_tol(*args, safety=1.5),
            aXtraMax_from_tail_tol(*args, safety=1.0),
        )

    def test_tol_floor_clamps_below_certifiable(self):
        a = aXtraMax_from_tail_tol(40.0, 0.5, 0.4, 150.0, 1e-12)
        b = aXtraMax_from_tail_tol(40.0, 0.5, 0.4, 150.0, 1e-6)
        self.assertEqual(a, b)

    def test_fails_closed_on_bad_inputs(self):
        self.assertTrue(np.isnan(
            aXtraMax_from_tail_tol(40.0, -0.1, 0.4, 150.0, 1e-4)))
        self.assertTrue(np.isnan(
            aXtraMax_from_tail_tol(40.0, 0.5, 0.0, 150.0, 1e-4)))
        self.assertTrue(np.isnan(
            aXtraMax_from_tail_tol(np.nan, 0.5, 0.4, 150.0, 1e-4)))

    def test_rel_gap_at_measures_the_synthetic_gap(self):
        kappa, hN = 0.02, 150.0
        gapfun = lambda m: 0.8 * ((m + hN) / hN) ** (-0.4)
        cfun = lambda m: kappa * (np.asarray(m, float) + hN) - gapfun(
            np.asarray(m, float))
        m = np.array([10.0, 100.0, 1000.0])
        got = rel_gap_at(cfun, m, kappa, hN)
        want = gapfun(m) / cfun(m)
        self.assertTrue(np.allclose(got, want, rtol=1e-12))
        # scalar in -> scalar out
        self.assertIsInstance(rel_gap_at(cfun, 10.0, kappa, hN), float)

    def test_live_extent_is_conservative_on_the_HS_anchor(self):
        # Deeper reference points imply SMALLER certified tops than shallow
        # ones inverted with the same q_eff would (the pre-asymptotic local
        # exponent rises with depth), and the inversion is monotone in tol.
        params = _params("HS")
        hN = params.h * params.E_inc
        top4 = aXtraMax_from_tail_tol(40.0, 0.75, params.q, hN, 1e-4)
        top3 = aXtraMax_from_tail_tol(40.0, 0.75, params.q, hN, 1e-3)
        self.assertGreater(top4, top3)
        self.assertGreater(top3, 40.0)

    # --- review-hardening additions (2026-07-10 adversarial pass) ---

    def test_rel_gap_at_empty_input_returns_empty(self):
        out = rel_gap_at(lambda m: np.asarray(m, float), np.array([]), 0.02,
                         150.0)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.size, 0)

    def test_bad_safety_fails_closed(self):
        self.assertTrue(np.isnan(
            aXtraMax_from_tail_tol(40.0, 0.5, 0.4, 150.0, 1e-4, safety=-1.5)))
        self.assertTrue(np.isnan(
            aXtraMax_from_tail_tol(40.0, 0.5, 0.4, 150.0, 1e-4, safety=0.0)))
        self.assertTrue(np.isnan(
            aXtraMax_from_tail_tol(40.0, 0.5, 0.4, 150.0, 1e-4,
                                   safety=np.nan)))

    def test_closed_form_route_guarantees_by_identity(self):
        # q* > 1: x_top = sqrt(B_psi/(MPCmin*tol)) makes B_psi/(MPCmin*x^2)
        # equal tol exactly; monotone in tol; live CCAP ex-post measured
        # 6.9e-5 at tol 1e-4 (see the frontier-of-failure record).
        params = _params("CCAP")
        hN = params.h * params.E_inc
        top = aXtraMax_from_tail_tol(np.nan, np.nan, np.nan, hN, 1e-4,
                                     B_psi=params.B_psi, MPCmin=params.kappa)
        x = top + hN
        self.assertAlmostEqual(params.B_psi / (params.kappa * x * x) / 1e-4,
                               1.0, places=10)
        top6 = aXtraMax_from_tail_tol(np.nan, np.nan, np.nan, hN, 1e-6,
                                      B_psi=params.B_psi, MPCmin=params.kappa)
        self.assertGreater(top6, top)
        # closed-form route with a q* < 1 calibration (B_psi None) -> nan
        self.assertTrue(np.isnan(aXtraMax_from_tail_tol(
            np.nan, np.nan, np.nan, hN, 1e-4, B_psi=_params("HS").B_psi,
            MPCmin=params.kappa)) or _params("HS").B_psi is None)


class TestQstarProbe(unittest.TestCase):
    """The operator eigen-probe: numerical q* from the model's own one-period
    backward step (no eigen-equation). Pre-registered: |q_hat - q*| <= 5e-4
    per anchor (measured 5.6e-6..5.0e-5), depth-consistency <= 1e-4
    (measured ~1e-6)."""

    def test_matches_analytic_root_on_anchors(self):
        for tag in ("HS", "CTOP", "CCAP"):
            c = CALS[tag]
            q_true = _params(tag).q_star
            q_hat, cons, diag = qstar_probe(
                R0, c["G"], c["beta"], RHO, LivPrb=LIV,
                PermShkDstn=(PSI, PP), TranShkDstn=(c["th"], c["tp"]))
            self.assertEqual(diag, "ok", tag)
            self.assertLessEqual(abs(q_hat - q_true), 5e-4, tag)
            self.assertLessEqual(cons, 1e-4, tag)

    def test_custom_one_step_hook_reproduces_primitives_mode(self):
        # portability contract: a caller-supplied backward step + PF limits
        c = CALS["HS"]
        params = _params("HS")
        hN = params.h * params.E_inc
        PSIj, THj = np.meshgrid(PSI, c["th"], indexing="ij")
        WPj = np.outer(PP, c["tp"]).ravel()
        WPj = WPj / WPj.sum()
        psi_j, th_j = PSIj.ravel(), THj.ravel()

        def my_step(c_trial, a):
            m_img = (R0 / (c["G"] * psi_j))[None, :] * a[:, None] \
                + th_j[None, :]
            rhs = c["beta"] * LIV * R0 * (
                WPj[None, :] * (c["G"] * psi_j[None, :]) ** (-RHO)
                * c_trial(m_img) ** (-RHO)).sum(1)
            return rhs ** (-1.0 / RHO)

        q_a, _, diag_a = qstar_probe(
            R0, c["G"], c["beta"], RHO, LivPrb=LIV,
            PermShkDstn=(PSI, PP), TranShkDstn=(c["th"], c["tp"]))
        q_b, _, diag_b = qstar_probe(
            R0, c["G"], c["beta"], RHO, LivPrb=LIV,
            one_step=my_step, MPCmin=params.kappa, hNrm=hN)
        self.assertEqual((diag_a, diag_b), ("ok", "ok"))
        self.assertLessEqual(abs(q_a - q_b), 1e-6)

    def test_fails_closed_without_pf_asymptote(self):
        # FHWC violated: Rcal <= 1 -> h nan -> nan + reason, no exception
        c = CALS["HS"]
        q_hat, cons, diag = qstar_probe(
            1.0, 1.02, c["beta"], RHO, LivPrb=LIV,
            PermShkDstn=(PSI, PP), TranShkDstn=(c["th"], c["tp"]))
        self.assertTrue(np.isnan(q_hat))
        self.assertIn("asymptote unavailable", diag)


class TestStablePoints(unittest.TestCase):
    """mNrm_stable_points: the two classical loci + the mortality-adjusted
    (R -> LivPrb*R) twins, with end-of-period-asset images."""

    @classmethod
    def setUpClass(cls):
        cls.c = CALS["HS"]
        cls.params = _params("HS")
        hE = cls.params.h * cls.params.E_inc
        kap = cls.params.kappa
        # synthetic concave cFunc below its PF line with a power-law gap
        cls.cf = staticmethod(
            lambda m: kap * (np.asarray(m, float) + hE)
            - 0.9 * (np.asarray(m, float) + hE) ** (-0.38) * hE ** 0.76)
        cls.sp = mNrm_stable_points(
            cls.cf, R0, cls.c["G"], LivPrb=LIV,
            PermShkDstn=(PSI, PP), TranShkDstn=(cls.c["th"], cls.c["tp"]))

    def test_roots_satisfy_their_defining_loci(self):
        sp, c = self.sp, self.c
        R, G, L = R0, c["G"], LIV
        Eth, Eip = sp.E_theta, sp.E_inv_psi
        loci = {
            sp.mNrmTrg: lambda m: m - (m - Eth) / ((R / G) * Eip),
            sp.mNrmStE: lambda m: m - (m - Eth) * G / R,
            sp.mNrmTrg_mort: lambda m: m - (m - Eth) / ((L * R / G) * Eip),
            sp.mNrmStE_mort: lambda m: m - (m - Eth) * G / (L * R),
        }
        for m_hat, locus in loci.items():
            self.assertTrue(np.isfinite(m_hat))
            resid = abs(float(self.cf(np.array([m_hat]))[0]) - locus(m_hat))
            self.assertLessEqual(resid, 1e-7 * max(1.0, m_hat))

    def test_orderings_and_a_images(self):
        sp = self.sp
        # Jensen (E[1/psi] > 1): StE below Trg; mortality shave lowers both
        self.assertLess(sp.mNrmStE, sp.mNrmTrg)
        self.assertLess(sp.mNrmTrg_mort, sp.mNrmTrg)
        self.assertLess(sp.mNrmStE_mort, sp.mNrmStE)
        # a-images are m - c(m) (the grid-relevant coordinates)
        self.assertAlmostEqual(
            sp.aNrmTrg,
            sp.mNrmTrg - float(self.cf(np.array([sp.mNrmTrg]))[0]), places=10)

    def test_LivPrb_one_degenerates_to_unadjusted(self):
        sp1 = mNrm_stable_points(
            self.cf, R0, self.c["G"], LivPrb=1.0,
            PermShkDstn=(PSI, PP), TranShkDstn=(self.c["th"], self.c["tp"]))
        self.assertAlmostEqual(sp1.mNrmTrg, sp1.mNrmTrg_mort, places=12)
        self.assertAlmostEqual(sp1.mNrmStE, sp1.mNrmStE_mort, places=12)

    def test_nan_when_no_crossing(self):
        sp = mNrm_stable_points(
            lambda m: np.full_like(np.asarray(m, float), 0.35),
            R0, self.c["G"], LivPrb=LIV,
            PermShkDstn=(PSI, PP), TranShkDstn=(self.c["th"], self.c["tp"]))
        self.assertTrue(np.isnan(sp.mNrmTrg))
        self.assertTrue(np.isnan(sp.mNrmStE))


class TestZetaL(unittest.TestCase):
    """dual_root(..., LivPrb): the mortality-augmented dual root."""

    def test_livprb_one_is_byte_compatible_none_at_cap(self):
        c = CALS["CCAP"]
        params = _params("CCAP")
        z, ElnA, PA, diag = dual_root(PSI, PP, params.Thorn_Gamma)
        self.assertIsNone(z)
        self.assertIn("positive log-drift", diag)

    def test_mortality_root_exists_at_the_cap(self):
        # the pure-GIC case: no Kesten root, but the mortality-augmented one
        # exists via the expanding branch (measured ~1.92 at the anchor)
        params = _params("CCAP")
        z, _, _, diag = dual_root(PSI, PP, params.Thorn_Gamma, LivPrb=LIV)
        self.assertIsNone(diag)
        self.assertGreater(z, 1.2)
        self.assertLess(z, 3.0)

    def test_mortality_thins_the_tail(self):
        params = _params("HS")
        z1, _, _, _ = dual_root(PSI, PP, params.Thorn_Gamma)
        zL, _, _, _ = dual_root(PSI, PP, params.Thorn_Gamma, LivPrb=LIV)
        self.assertGreater(zL, z1)

    def test_no_expanding_branch_stays_rootless(self):
        # psi == 1 under GIC: compact support; mortality cannot create a tail
        params = _params("HS")
        z, _, _, diag = dual_root(np.array([1.0]), np.array([1.0]),
                                  params.Thorn_Gamma, LivPrb=LIV)
        self.assertIsNone(z)
        self.assertIn("no expanding branch", diag)


class TestWealthMassRule(unittest.TestCase):
    """aXtraMax_from_wealth_mass on a synthetic solved cFunc (solve-free)."""

    @classmethod
    def setUpClass(cls):
        cls.c = CALS["HS"]
        params = _params("HS")
        hE = params.h * params.E_inc
        kap = params.kappa
        # harmonic blend: ~m at small m, ~kappa*(m+hE) at large m; smooth,
        # 0 < c < min(m, line) so a = m - c > 0 and the gap is positive
        cls.cf = staticmethod(
            lambda m: 1.0 / (1.0 / np.maximum(np.asarray(m, float), 1e-12)
                             + 1.0 / (kap * (np.asarray(m, float) + hE))))
        cls.a_max, cls.info = aXtraMax_from_wealth_mass(
            cls.cf, R0, cls.c["G"], cls.c["beta"], RHO, LivPrb=LIV,
            PermShkDstn=(PSI, PP), TranShkDstn=(cls.c["th"], cls.c["tp"]),
            eps_wealth=1e-4, probe_count=384)

    def test_returns_finite_adequate_top(self):
        self.assertTrue(np.isfinite(self.a_max))
        self.assertGreater(self.a_max, self.info.anchor_a)
        self.assertTrue(self.info.cover_adequate)
        self.assertEqual(self.info.diagnosis, "ok")

    def test_dial_monotonicities(self):
        t = self.info.quantile_table
        for meas in ("agent", "wealth"):
            self.assertLessEqual(t[(meas, 1e-2)], t[(meas, 1e-3)])
            self.assertLessEqual(t[(meas, 1e-3)], t[(meas, 1e-4)])
        for e in (1e-2, 1e-3, 1e-4):
            self.assertGreaterEqual(t[("wealth", e)], t[("agent", e)])

    def test_deterministic(self):
        a2, info2 = aXtraMax_from_wealth_mass(
            self.cf, R0, self.c["G"], self.c["beta"], RHO, LivPrb=LIV,
            PermShkDstn=(PSI, PP), TranShkDstn=(self.c["th"], self.c["tp"]),
            eps_wealth=1e-4, probe_count=384)
        self.assertEqual(self.a_max, a2)

    def test_wealth_measure_refuses_when_aggregate_wealth_unbounded(self):
        # cap-atom primitives with near-unit survival: zeta_L <= 1
        c = CALS["CCAP"]
        a_max, info = aXtraMax_from_wealth_mass(
            self.cf, R0, c["G"], c["beta"], RHO, LivPrb=0.9999,
            PermShkDstn=(PSI, PP), TranShkDstn=(c["th"], c["tp"]),
            eps_wealth=1e-4, probe_count=384)
        self.assertTrue(np.isnan(a_max))
        self.assertIn("REFUSED", info.diagnosis)

    def test_never_raises_on_degenerate_primitives(self):
        a_max, info = aXtraMax_from_wealth_mass(
            self.cf, 1.0, 1.02, 0.96, RHO, LivPrb=LIV,
            PermShkDstn=(PSI, PP), TranShkDstn=(self.c["th"], self.c["tp"]),
            probe_count=256)
        self.assertTrue(isinstance(info.diagnosis, str)
                        and len(info.diagnosis) > 0)

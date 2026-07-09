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
    powerlaw_decay_params,
    powerlaw_decay_params_from_agent,
    powerlaw_tail_diagnostic,
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
        # THEOREM-REF[HAFiscal-Latest @ 71ca7c61 :: theory/powerlaw-decay/alt_proof_compactified.md :: Theorem γ-B (Stage-B boundary value)]
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

    # THEOREM-REF[HAFiscal-Latest @ 71ca7c61 :: theory/powerlaw-decay/ADVERSARIAL_TESTING_GUIDE.md :: 5. LANDMINES — documented evaluation traps and silent-pass hazards :: The `h` human-wealth convention]
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

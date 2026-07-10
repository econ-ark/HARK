"""Theory utilities for the power-law decay of the buffer-stock consumption gap.

Computes, from model primitives, the quantities of the power-law-decay theorem
program that governs how a buffer-stock consumption function approaches its
perfect-foresight asymptote:

    g(x) := kappa*(m + h) - c(m)  ~  x^(-min(1, q_star))    as  x = m + h -> oo,

where ``q_star`` is the unique positive root of the eigen-equation

    (E)     E[psi^(1+q)] = Rcal * Thorn_Gamma^q,

with sharp closed-form amplitude ``B_psi`` at ``q_star > 1`` and a ``ln(x)``
resonance law exactly at ``q_star = 1``.

# THEOREM-REF[HAFiscal-Latest @ 71ca7c61 :: theory/powerlaw-decay/final_proof.md :: §0 "What is q*? (and why min(1, q*))" :: eq (E)]
#   q* = (discounting speed ln Rcal) / (descent speed Lambda) at psi == 1, and in
#   general the root of (E). The realized decay exponent is min(1, q*): the gap is
#   the sum of a here-and-now 1/x precautionary component and a near-target x^(-q*)
#   component, and the slower-fading one wins asymptotically.

# THEOREM-REF[HAFiscal-Latest @ 71ca7c61 :: theory/powerlaw-decay/final_proof.md :: §0 "What is q*? (and why min(1, q*))" :: Prop A0]
#   The gap can never decay FASTER than 1/x (a prudent consumer always collects at
#   least today's Arrow-Pratt premium) — Prop A0's floor, which makes exponential
#   decay impossible as an asymptotic form; and since the realized exponent is
#   min(1, q*), any steeper fitted exponent is theory-infeasible.

Conventions (all verified against the theorem program's reference implementations)
-----------------------------------------------------------------------------------
* ``Rcal   = R / Gamma``                       (FHWC factor; theorem's script-R)
* ``Thorn  = (beta_eff * R)^(1/rho)``          (absolute patience factor, APF)
* ``Thorn_R = Thorn / R``  (RPF),  ``Thorn_Gamma = Thorn / Gamma``  (GPF)
* ``kappa  = 1 - Thorn_R``                     (limiting MPC; == ``pf_mpc_min``)
* ``Lambda = ln(1/Thorn_Gamma)``               (log-ladder step; > 0 under GIC)
* ``h      = 1/(Rcal - 1)``                    (normalized human wealth EXCLUDING
  current income, per unit of expected income).

# THEOREM-REF[HAFiscal-Latest @ 71ca7c61 :: theory/powerlaw-decay/ADVERSARIAL_TESTING_GUIDE.md :: 5. LANDMINES — documented evaluation traps and silent-pass hazards :: The `h` human-wealth convention]
#   The h-convention trap: BST's h_BST = R/(R-Gamma) = 1 + h INCLUDES current
#   income; plugging h_BST into kappa*(m+h) sends the measured gap to -kappa*E_inc,
#   a spurious refutation. HARK itself carries BOTH conventions: the solver-side
#   ``solution.hNrm`` (recursion ``calc_human_wealth``, terminal 0) EXCLUDES current
#   income and equals ``h * E_inc`` here, while the diagnostics dict
#   ``bilt['hNrm']`` from ``calc_limiting_values`` = R/(R-G) INCLUDES it. This
#   module computes h from primitives and NEVER consumes either solved attribute
#   (``solution.hNrm`` is additionally ~11% truncated at HARK's default solve
#   tolerance, because the solution distance criterion is vPfunc-based while the
#   hNrm recursion contracts only at rate G/R per iteration).

# THEOREM-REF[HAFiscal-Latest @ 71ca7c61 :: theory/powerlaw-decay/statement.md :: 4. Remarks :: Mortality]
#   Mortality-as-impatience (Remark 9, perpetual-youth): with survival probability
#   L, replace beta by beta*L throughout the patience factors (Thorn, Thorn_R,
#   Thorn_Gamma, kappa, Lambda, q*); h and Rcal stay mortality-free. This is
#   exactly HARK's convention: DiscFacEff = DiscFac*LivPrb enters MPCmin while
#   ``calc_human_wealth`` carries no LivPrb.

Degenerate / pathological inputs (behavior contract)
-----------------------------------------------------
* ``psi == 1`` (PermShkDstn=None or a single unit atom): everything reduces to the
  transitory-only (Stage A) theory; q* equals the closed form ``ln(Rcal)/Lambda``
  and the dual root does not exist (compact-support regime).
* **FHWC violated** (``Rcal <= 1``): REFUSES CLEANLY — no exception; ``valid=False``
  with ``h = nan``, ``sigma_B2 = B_psi = c_J = None``, ``q_star = nan`` and a
  ``diagnosis`` explaining that L(0) = -ln(Rcal) >= 0 kills the (E)-root.
* **GIC violated** (``Thorn_Gamma >= 1``): ``valid=False`` and a warning, but the
  (E)-root is still REPORTED when it exists (a wide psi can make L(q) cross zero
  even outside the theorem's hypotheses); with psi == 1 no root exists.
* **RIC violated** (``Thorn_R >= 1``): ``valid=False``, warning; kappa <= 0 is
  reported as computed (the PF asymptote itself degenerates).
* ``B_psi`` is ``None`` unless ``q* > 1`` AND ``Rcal*Thorn_Gamma - E[psi^2] > 0``
  (equivalent conditions via ``lambda_B < 1``; both are guarded).
* ``zeta_star`` is ``None`` with a ``dual_diagnosis`` string when the dual root
  does not exist (positive log-drift, or no expanding branch).
* **Near-resonance warning** (``NearResonanceWarning``, filterable): fires when
  ``|lambda_B - 1| < near_resonance_band`` (default 0.01) — the calibration sits
  near the r = g (q* = 1) knife-edge where the pre-asymptotic window is longest.

Inputs (HARK-style accepted, no HARK import required)
------------------------------------------------------
Shock distributions may be given as ``(atoms, probs)`` tuples/lists of arrays, as
HARK ``DiscreteDistribution`` objects (duck-typed on ``.atoms`` / ``.pmv``; a
univariate ``atoms`` of shape (1, N) is flattened), or as one-element time-varying
lists ``[dstn]`` of either.  Alternatively pass the HARK-style JOINT ``IncShkDstn``
(atoms row 0 = permanent, row 1 = transitory); then ``sigma_B2`` is computed as
``Var(psi*(theta+h))`` directly on the joint — the master identity's Var(W) —
which equals the closed form under independence and remains the correct Var(W)
even when the joint correlates psi and theta (the theorem's hypotheses, however,
assume psi independent of theta; a ``ShockCorrelationWarning`` flags correlated
joints).

Numerics: scalar derived objects are computed in ``numpy.longdouble`` (matching
the theorem program's reference implementation ``RB4_egm_lib_B.derived``);
root-finding uses float64 bracket-expansion + bisection to floating-point
resolution.  The numpy-only root-finder is kept (rather than scipy's brentq)
because it is the exact code falsified 60/60 against the reference brentq roots
(agreement <= 5e-14 absolute on the estimated HAFiscal calibrations) and it keeps
this module dependency-free beyond numpy.

Only runtime dependency: numpy.
"""

import warnings as _warnings
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "PowerLawDecayParams",
    "powerlaw_decay_params",
    "powerlaw_decay_params_from_agent",
    "resonance_constants",
    "qstar_root",
    "dual_root",
    "TailDiagnostic",
    "powerlaw_tail_diagnostic",
    "rel_gap_at",
    "aXtraMax_from_tail_tol",
    "qstar_probe",
    "StablePoints",
    "mNrm_stable_points",
    "PFDecayConditionWarning",
    "PFDecayGridWarning",
    "NearResonanceWarning",
    "NoDualRootWarning",
    "ShockCorrelationWarning",
]

_LD = np.longdouble

# Bracket-expansion cap for both root searches. Generous (not 1024): near the
# r = g knife-edge Lambda = -ln(Thorn_Gamma) -> 0 drives the primal root
# q* = ln(Rcal)/Lambda (and, symmetrically, the dual root zeta*) arbitrarily
# large, so a tight cap would return a spurious "no root" for a perfectly valid
# calibration whose realized exponent min(1, q*) is simply 1. Both L(q) and f(z)
# are evaluated by logsumexp, so a large cap cannot overflow; the expansion loop
# still tightens the upper bracket to the smallest power of two past the root
# before bisection, so the cost of the large cap is only ~30 extra doublings in
# the (unreached) pathological limit.
_BRACKET_CAP = 1.0e12  # bracket-expansion cap for both root searches


# ----------------------------------------------------------------- warning taxonomy
class PFDecayConditionWarning(UserWarning):
    """A perfect-foresight/buffer-stock condition (FHWC, RIC, GIC) fails, so part
    of the power-law-decay theory is undefined or out of scope."""


class PFDecayGridWarning(UserWarning):
    """A fitted decay exponent or amplitude attachment looks grid-pathological
    (e.g. a fitted exponent above the theoretical ceiling min(1, q*))."""


class NearResonanceWarning(UserWarning):
    """The calibration sits near the q* = 1 (r = g) knife-edge, where asymptotic
    constants onset only at astronomically large wealth. Filterable so production
    loggers can silence it without touching other UserWarnings."""


class NoDualRootWarning(UserWarning):
    """The dual (Kesten) root zeta* does not exist for this calibration."""


class ShockCorrelationWarning(UserWarning):
    """A joint IncShkDstn correlates psi and theta; the theorem assumes
    independence (sigma_B2 is still the exact Var(W) on the joint)."""


# --------------------------------------------------------------------- input handling
def _as_atoms_probs(dstn, name):
    """Coerce a shock-distribution input to ``(atoms, probs)`` longdouble arrays.

    Accepts (atoms, probs) tuples/lists, HARK DiscreteDistribution-likes
    (``.atoms``/``.pmv``), or one-element time-varying lists of either.
    Univariate HARK atoms of shape (1, N) are flattened; probabilities are
    validated (nonnegative, sum to 1 within 1e-8, then renormalized exactly)
    and the mean is required to be 1 (1e-6), the theorem's normalization
    E[psi] = E[theta] = 1.
    """
    # unwrap one-element time-varying list [dstn]
    if isinstance(dstn, (list, tuple)) and len(dstn) == 1 and (
            hasattr(dstn[0], "pmv") or isinstance(dstn[0], (list, tuple))):
        dstn = dstn[0]
    if hasattr(dstn, "pmv") and hasattr(dstn, "atoms"):        # HARK duck-type
        atoms = np.asarray(dstn.atoms)
        probs = np.asarray(dstn.pmv)
    elif isinstance(dstn, (list, tuple)) and len(dstn) == 2:   # (atoms, probs)
        atoms = np.asarray(dstn[0])
        probs = np.asarray(dstn[1])
    else:
        raise TypeError(
            f"{name}: expected (atoms, probs) or a HARK DiscreteDistribution-like "
            f"object with .atoms/.pmv, got {type(dstn).__name__}")
    if atoms.ndim == 2 and atoms.shape[0] == 1:
        atoms = atoms[0]
    if atoms.ndim != 1:
        raise ValueError(f"{name}: expected univariate atoms, got shape {atoms.shape}")
    atoms = atoms.astype(_LD)
    probs = probs.astype(_LD)
    if atoms.shape != probs.shape:
        raise ValueError(f"{name}: atoms/probs shape mismatch "
                         f"{atoms.shape} vs {probs.shape}")
    probs = _validated_probs(probs, name)
    mean = float((probs * atoms).sum())
    if abs(mean - 1.0) > 1e-6:
        raise ValueError(f"{name}: mean {mean} != 1 (theorem normalization E[.] = 1)")
    return atoms, probs


def _validated_probs(probs, name):
    """Require nonnegative probabilities summing to 1 (tol 1e-8), then
    renormalize exactly."""
    if np.any(probs < 0):
        raise ValueError(f"{name}: probabilities must be nonnegative")
    total = float(probs.sum())
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"{name}: probabilities sum to {total:.10g}, not 1")
    return probs / probs.sum()


def _as_joint(dstn):
    """Coerce a HARK-style joint IncShkDstn to (psi_atoms, theta_atoms, probs).

    Layout per ``ConsIndShockModel.calc_limiting_values``: ``atoms[0]`` = permanent
    shock values, ``atoms[1]`` = transitory shock values, ``pmv`` = probabilities.
    Also accepts an ``(atoms_2xN, probs)`` tuple with atoms of shape (2, N).
    """
    if isinstance(dstn, (list, tuple)) and len(dstn) == 1 and hasattr(dstn[0], "pmv"):
        dstn = dstn[0]
    if hasattr(dstn, "pmv") and hasattr(dstn, "atoms"):
        atoms = np.asarray(dstn.atoms)
        probs = np.asarray(dstn.pmv)
    elif isinstance(dstn, (list, tuple)) and len(dstn) == 2:
        atoms = np.asarray(dstn[0])
        probs = np.asarray(dstn[1])
    else:
        raise TypeError("IncShkDstn: expected a joint HARK DiscreteDistribution-like "
                        "or (atoms_2xN, probs)")
    if atoms.ndim != 2 or atoms.shape[0] != 2:
        raise ValueError(f"IncShkDstn: joint IncShkDstn atoms must have shape (2, N) "
                         f"(row 0 = perm, row 1 = tran), got {atoms.shape}")
    psi = atoms[0].astype(_LD)
    th = atoms[1].astype(_LD)
    p = _validated_probs(probs.astype(_LD), "IncShkDstn")
    for nm, a in (("perm", psi), ("tran", th)):
        mean = float((p * a).sum())
        if abs(mean - 1.0) > 1e-6:
            raise ValueError(f"IncShkDstn: E[{nm}] = {mean} != 1")
    return psi, th, p


def _marginal(atoms, probs):
    """Collapse repeated atom values into a proper marginal distribution."""
    vals, inv = np.unique(np.asarray(atoms, float), return_inverse=True)
    pm = np.zeros(len(vals), dtype=_LD)
    np.add.at(pm, inv, probs)
    return vals.astype(_LD), pm


def _moments(atoms, probs):
    """(mean, variance) of a discrete distribution, longdouble."""
    m1 = (probs * atoms).sum()
    return m1, (probs * (atoms - m1) ** 2).sum()


# --------------------------------------------------------------------- root finding
def _bisect(f, lo, hi, flo, max_iter=200):
    """Bisection to floating-point resolution; requires sign(f(lo)) != sign(f(hi))."""
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi:
            break
        fm = f(mid)
        if fm == 0.0:
            return mid
        if (flo < 0.0) == (fm < 0.0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def qstar_root(psi_atoms, psi_probs, Rcal, Thorn_Gamma):
    """Unique positive root of (E): E[psi^(1+q)] = Rcal * Thorn_Gamma^q.

    # THEOREM-REF[HAFiscal-Latest @ 71ca7c61 :: theory/powerlaw-decay/final_proof.md :: §2. Model, conditions, and the imported foundations :: Lemma A5]
    #   Lemma A5: L(q) := ln E[psi^(1+q)] - ln Rcal - q*ln Thorn_Gamma is convex
    #   with L(0) = -ln Rcal < 0 under FHWC, so (E) has a unique positive root;
    #   bracket-expansion upward from q=1 then bisection finds it.

    Robust bracket-expansion + bisection, failing closed instead of raising.

    Returns ``(q_star, reason)``: ``(float, None)`` on success, ``(nan, str)`` when
    no root is available — either ``L(0) = -ln(Rcal) >= 0`` (FHWC violated) or no
    sign change up to the bracket cap (e.g. GIC violated with psi == 1, where L is
    non-increasing).
    """
    psf = np.asarray(psi_atoms, float)
    ppf = np.asarray(psi_probs, float)
    RG, PG = float(Rcal), float(Thorn_Gamma)

    ln_RG, ln_PG = np.log(RG), np.log(PG)
    ln_psf = np.log(psf)
    with np.errstate(divide="ignore"):        # a 0-prob atom -> -inf, drops out
        ln_ppf = np.log(ppf)

    def L(q):
        # L(q) = ln E[psi^(1+q)] - ln Rcal - q*ln Thorn_Gamma. The expectation is
        # formed by logsumexp so that psi atoms > 1 cannot overflow at the large q
        # reached near the r = g knife-edge (Lambda -> 0 => q* -> oo); identical to
        # log(dot(ppf, psf**(1+q))) to machine precision on well-scaled inputs.
        t = (1.0 + q) * ln_psf + ln_ppf
        tm = t.max()
        return float(tm + np.log(np.exp(t - tm).sum()) - ln_RG - q * ln_PG)

    L0 = L(0.0)
    if L0 >= 0.0:
        return float("nan"), (
            f"no (E)-root: L(0) = -ln(Rcal) = {L0:.6g} >= 0 (FHWC violated: "
            f"Rcal = {RG:.6g} <= 1)")
    hi = 1.0
    while L(hi) < 0.0 and hi < _BRACKET_CAP:
        hi *= 2.0
    if L(hi) < 0.0:
        # No crossing within the (generous) cap. L(q) is EVENTUALLY INCREASING iff
        # its large-q slope ln(psi_max) - ln(Thorn_Gamma) > 0 (the largest psi atom
        # dominates E[psi^(1+q)]); under GIC with E[psi] = 1 that always holds and a
        # finite root exists. So a miss here means one of two things, and we must
        # NOT conflate them (the old message hard-asserted GIC violation for both):
        psi_max = float(psf.max())
        if float(ln_psf.max()) - ln_PG > 0.0:
            # L increasing: the root is finite but beyond the cap — an astronomically
            # near-resonance calibration (Lambda ~ 0). The realized exponent is 1.
            return float("nan"), (
                f"(E)-root exceeds bracket cap {_BRACKET_CAP:g}: L(q) is increasing "
                f"(psi_max = {psi_max:.6g} > Thorn_Gamma = {PG:.6g}) so q* is finite "
                f"but astronomically large (Lambda = {-ln_PG:.3g} ~ 0, the r = g "
                f"knife-edge); the realized decay exponent min(1, q*) = 1")
        return float("nan"), (
            f"no (E)-root: L(q) is non-increasing (psi_max = {psi_max:.6g} <= "
            f"Thorn_Gamma = {PG:.6g}) — Thorn_Gamma >= 1 (GIC violated) or psi "
            f"degenerate")
    return _bisect(L, 0.0, hi, L0), None


def dual_root(psi_atoms, psi_probs, Thorn_Gamma):
    """Dual (Kesten) root zeta* solving E[(Thorn_Gamma/psi)^zeta] = 1.

    # THEOREM-REF[HAFiscal-Latest @ 71ca7c61 :: theory/powerlaw-decay/final_proof.md :: §6.1 How rarely is the tail visited? The dual (Kesten) root, for economists]
    #   Normalized wealth follows the Kesten recursion x' = A'x + B' with
    #   A' = Thorn_Gamma/psi'; the stationary wealth tail is Pareto with exponent
    #   zeta* solving E[A^zeta] = 1, which exists iff the multiplier contracts on
    #   average (E[ln A] < 0) and occasionally expands (P(A > 1) > 0). zeta* and
    #   q* are the two members of one Mellin family: the DUAL root governs the
    #   wealth-distribution tail, the PRIMAL root q* the consumption function.

    Returns ``(zeta_star, E_ln_A, P_A_gt_1, diagnosis)`` with ``zeta_star = None``
    and a non-None diagnosis string when the root does not exist.  Root search is
    in log space (numerically safe for large zeta) with bracket-expansion +
    bisection.  A discretized psi is lattice-arithmetic, so zeta* tail statements
    hold in the log-periodic (Kevei 2017) sense.
    """
    psf = np.asarray(psi_atoms, float)
    ppf = np.asarray(psi_probs, float)
    PG = float(Thorn_Gamma)
    ln_A = np.log(PG) - np.log(psf)
    E_ln_A = float(np.dot(ppf, ln_A))
    P_A_gt_1 = float(ppf[ln_A > 0.0].sum())
    if E_ln_A >= 0.0:
        # THEOREM-REF[HAFiscal-Latest @ 71ca7c61 :: theory/powerlaw-decay/final_proof.md :: §6.1 How rarely is the tail visited? The dual (Kesten) root, for economists :: The cap-atom exception]
        #   Positive log-drift is the GIC-cap-atom case: at the patience ceiling the
        #   mean log-step is positive, normalized wealth drifts UP on average, and
        #   only mortality-with-replacement truncates that atom's wealth tail.
        return None, E_ln_A, P_A_gt_1, (
            f"no dual root: positive log-drift E[ln(Thorn_Gamma/psi)] = "
            f"{E_ln_A:+.6g} >= 0 — normalized wealth drifts UP on average; no "
            f"stationary distribution from impatience alone (mortality/reset must "
            f"truncate the tail)")
    if P_A_gt_1 <= 0.0:
        return None, E_ln_A, P_A_gt_1, (
            "no dual root: P(Thorn_Gamma/psi > 1) = 0 — no expanding branch; the "
            "ergodic normalized-wealth support is compact (e.g. psi == 1 under "
            "GIC), no Pareto tail of any kind")
    ln_p = np.log(ppf)

    def f(z):  # ln E[A^z] via logsumexp (safe for large z)
        t = z * ln_A + ln_p
        tm = t.max()
        return float(tm + np.log(np.exp(t - tm).sum()))

    hi = 1.0
    while f(hi) < 0.0 and hi < _BRACKET_CAP:
        hi *= 2.0
    if f(hi) < 0.0:
        # f(z) ~ z*ln(A_max) -> +inf whenever P(A > 1) > 0, so a finite root ALWAYS
        # exists here and the only way to reach this branch is zeta* beyond the
        # (generous) bracket cap — an astronomically near-resonance calibration where
        # the single expanding atom has A only marginally above 1. Kept as a numerical
        # backstop (with cap = 1e12 it is unreachable for any realistic calibration;
        # a tight cap of 1024 used to make it fire on legitimate near-resonance psi).
        return None, E_ln_A, P_A_gt_1, (
            f"no dual root found in (0, {_BRACKET_CAP:g}] despite E[ln A] < 0 and "
            f"P(A > 1) > 0: zeta* exceeds the bracket cap (astronomically "
            f"near-resonance; the Pareto tail is real but its exponent is enormous)")
    # f(0) = 0 with f'(0) = E[ln A] < 0, so f < 0 just inside 0: halve down from
    # hi/2 until the lower bracket endpoint is strictly negative (covers roots < 1)
    lo = hi / 2.0
    flo = f(lo)
    while flo >= 0.0 and lo > 1e-12:
        lo /= 2.0
        flo = f(lo)
    if flo >= 0.0:  # mathematically unreachable given E[ln A] < 0; kept for safety
        return None, E_ln_A, P_A_gt_1, (
            "no dual root: lower bracket could not be established despite "
            "E[ln A] < 0 (numerical degeneracy)")
    return _bisect(f, lo, hi, flo), E_ln_A, P_A_gt_1, None


# --------------------------------------------------------------------- result object
@dataclass(frozen=True)
class PowerLawDecayParams:
    """All power-law-decay theorem quantities for one calibration.

    ``None`` = not defined for this calibration; ``nan`` = defined but not
    computable from these primitives (see ``warnings``/``diagnosis``).
    """

    # primitives (echoed)
    Rfree: float
    PermGroFac: float
    DiscFac: float
    CRRA: float
    LivPrb: float
    beta_eff: float                       # DiscFac * LivPrb (mortality-as-impatience)
    # derived patience/return objects
    Rcal: float                           # R/Gamma (mortality-free)
    Thorn: float                          # (R*beta_eff)**(1/CRRA)  (APF)
    Thorn_R: float                        # APF/R   (RIC factor)
    Thorn_Gamma: float                    # APF/Gamma  (GIC factor)
    kappa: float                          # 1 - Thorn_R == PF MPCmin
    Lambda: float                         # ln(1/Thorn_Gamma) (> 0 iff GIC)
    h: float                              # 1/(Rcal-1), per unit E[inc]; EXCLUDES
    #                                       current income == solution.hNrm/E_inc
    # shock moments
    E_inc: float                          # E[psi*theta]
    E_psi2: float
    Var_psi: float
    Var_theta: float
    # the theorem
    q_star: float                         # unique positive root of (E); nan if none
    q: float                              # min(1, q_star) — realized decay exponent
    diagnosis: str                        # '' or why q_star is nan
    sigma_B2: Optional[float]             # E[psi^2]*Var(theta) + (1+h)^2*Var(psi)
    lambda_B: float                       # E[psi^2]/(Rcal*Thorn_Gamma)
    near_resonance: bool                  # |lambda_B - 1| < near_resonance_band
    resonance_slope: Optional[float]      # kappa*(rho+1)*sigma_B2/(2*Lambda): the
    #                                       q*=1 ln-x law constant at psi == 1, for
    #                                       reference (general-psi exact constant:
    #                                       resonance_constants()['C_B'])
    B_psi: Optional[float]                # closed-form amplitude; None unless q*>1
    c_J: Optional[float]                  # kappa*(rho+1)*sigma_B2/(2*Thorn_Gamma)
    # dual (Kesten) root
    zeta_star: Optional[float]
    dual_diagnosis: str                   # 'ok' or why zeta_star is None
    E_ln_A: float
    P_A_gt_1: float
    # condition flags
    GIC: bool
    RIC: bool
    FHWC: bool
    valid: bool
    warnings: Tuple[str, ...] = ()

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------- main entry
def powerlaw_decay_params(Rfree, PermGroFac, DiscFac, CRRA, LivPrb=1.0,
                          PermShkDstn=None, TranShkDstn=None, IncShkDstn=None,
                          near_resonance_band=0.01, warn=True):
    """Compute every power-law-decay theorem quantity from model primitives.

    Parameters
    ----------
    Rfree, PermGroFac, DiscFac, CRRA : float
        Gross return R, permanent-income growth factor Gamma, discount factor
        beta, relative risk aversion rho.  Scalars (a one-element list/array is
        accepted for each, matching HARK's time-varying parameter style).
    LivPrb : float, default 1.0
        Survival probability; enters ONLY via ``beta_eff = DiscFac*LivPrb``
        (mortality-as-impatience; see the module docstring).
    PermShkDstn, TranShkDstn : optional
        Marginal shock distributions — ``(atoms, probs)`` or HARK
        DiscreteDistribution-like (or ``[dstn]``).  ``PermShkDstn=None`` means
        psi == 1 (the transitory-only theory); a degenerate theta is accepted
        but yields sigma_B2 driven by psi only.  A theta = 0 unemployment atom
        IS allowed (native to the theorem); psi atoms must be strictly positive.
    IncShkDstn : optional
        HARK-style JOINT distribution (atoms row 0 = perm, row 1 = tran).
        Mutually exclusive with PermShkDstn/TranShkDstn.  sigma_B2 is then
        ``Var(psi*(theta+h))`` on the joint (the master identity's Var(W)),
        which equals the closed form under independence and is the correct
        Var(W) even under correlation (a ``ShockCorrelationWarning`` then
        notes that the theorem's hypotheses assume independence).
    near_resonance_band : float, default 0.01
        Threshold on ``|lambda_B - 1|`` for the near-resonance warning/flag.
    warn : bool, default True
        Emit the categorized Python warnings (``PFDecayConditionWarning``,
        ``NearResonanceWarning``, ``NoDualRootWarning``,
        ``ShockCorrelationWarning``).  The same messages are always recorded in
        the returned object's ``warnings`` tuple.  Warnings fire once per call
        to this utility, never per consumption-function slice.

    Returns
    -------
    PowerLawDecayParams
        Frozen dataclass with all quantities, condition flags, and warnings.
        Never raises on GIC/RIC/FHWC violations (see the module docstring's
        behavior contract); raises ValueError/TypeError only on malformed
        inputs.
    """
    def _scalar(v, name):
        if isinstance(v, (list, tuple, np.ndarray)):
            v = np.asarray(v).ravel()
            if v.size != 1:
                raise ValueError(f"{name}: expected a scalar (or length-1 "
                                 f"list/array), got size {v.size}")
            v = v[0]
        return _LD(float(v))

    R = _scalar(Rfree, "Rfree")
    G = _scalar(PermGroFac, "PermGroFac")
    beta = _scalar(DiscFac, "DiscFac")
    rho = _scalar(CRRA, "CRRA")
    liv = _scalar(LivPrb, "LivPrb")
    if R <= 0 or G <= 0 or beta <= 0 or rho <= 0 or not (0 < liv <= 1):
        raise ValueError("Rfree, PermGroFac, DiscFac, CRRA must be positive and "
                         "LivPrb in (0, 1]")

    warn_list = []
    warn_cats = []

    def _record(msg, category):
        warn_list.append(msg)
        warn_cats.append(category)

    # ---- shock distributions
    joint = None
    if IncShkDstn is not None:
        if PermShkDstn is not None or TranShkDstn is not None:
            raise ValueError("pass IncShkDstn OR (PermShkDstn, TranShkDstn), "
                             "not both")
        psi_j, th_j, p_j = _as_joint(IncShkDstn)
        joint = (psi_j, th_j, p_j)
        psi_a, psi_p = _marginal(psi_j, p_j)
        th_a, th_p = _marginal(th_j, p_j)
        E_inc = float((p_j * psi_j * th_j).sum())
        E_prod = float((psi_p * psi_a).sum()) * float((th_p * th_a).sum())
        if abs(E_inc - E_prod) > 1e-10 * max(1.0, abs(E_inc)):
            _record(
                f"IncShkDstn psi and theta are correlated "
                f"(|E[psi*theta]-E[psi]E[theta]| = {abs(E_inc - E_prod):.2e}); "
                f"the theorem's hypotheses assume psi independent of theta. "
                f"sigma_B2 is the exact Var(W) on the joint; q_star and "
                f"zeta_star use the psi marginal.", ShockCorrelationWarning)
    else:
        if PermShkDstn is None:
            psi_a = np.array([_LD(1)])
            psi_p = np.array([_LD(1)])
        else:
            psi_a, psi_p = _as_atoms_probs(PermShkDstn, "PermShkDstn")
        if TranShkDstn is None:
            th_a = np.array([_LD(1)])
            th_p = np.array([_LD(1)])
        else:
            th_a, th_p = _as_atoms_probs(TranShkDstn, "TranShkDstn")
        E_inc = float((psi_p * psi_a).sum()) * float((th_p * th_a).sum())
    if np.any(psi_a <= 0):
        raise ValueError("PermShkDstn: psi atoms must be strictly positive "
                         "(supp psi subset of (0, inf))")

    # ---- derived patience/return objects (longdouble)
    beta_eff = beta * liv
    Thorn = (R * beta_eff) ** (1 / rho)
    Rcal = R / G
    Thorn_R = Thorn / R
    Thorn_Gamma = Thorn / G
    kappa = 1 - Thorn_R
    Lambda = -np.log(Thorn_Gamma)

    FHWC = bool(Rcal > 1)
    RIC = bool(Thorn_R < 1)
    GIC = bool(Thorn_Gamma < 1)
    if not FHWC:
        _record(f"FHWC VIOLATED: Rcal = R/Gamma = {float(Rcal):.6g} <= 1 — "
                f"human wealth h = 1/(Rcal-1) undefined/infinite; the PF "
                f"asymptote and every h-dependent quantity are unavailable",
                PFDecayConditionWarning)
    if not RIC:
        _record(f"RIC VIOLATED: Thorn_R = {float(Thorn_R):.6g} >= 1 — "
                f"kappa = 1 - Thorn_R <= 0, the PF asymptote degenerates",
                PFDecayConditionWarning)
    if not GIC:
        _record(f"GIC VIOLATED: Thorn_Gamma = {float(Thorn_Gamma):.6g} >= 1 "
                f"(Lambda = {float(Lambda):.6g} <= 0) — the theorem's "
                f"buffer-stock hypotheses fail; any reported (E)-root is "
                f"outside the theorem's scope", PFDecayConditionWarning)

    # h EXCLUDES current income: h = h_BST - 1 (see the h-convention tag in the
    # module docstring; never read solution.hNrm or bilt['hNrm'] here).
    h = 1 / (Rcal - 1) if FHWC else _LD(float("nan"))

    # ---- shock moments
    E_psi2 = float((psi_p * psi_a ** 2).sum())
    _, Var_psi = _moments(psi_a, psi_p)
    _, Var_theta = _moments(th_a, th_p)
    Var_psi, Var_theta = float(Var_psi), float(Var_theta)

    # ---- eigen-root q*, realized exponent
    q_star, q_reason = qstar_root(psi_a, psi_p, Rcal, Thorn_Gamma)
    q = min(1.0, q_star) if np.isfinite(q_star) else float("nan")

    # ---- lambda_B and near-resonance warning (h-free)
    lambda_B = float(E_psi2 / (Rcal * Thorn_Gamma))
    near_res = bool(abs(lambda_B - 1.0) < near_resonance_band)
    if near_res:
        # THEOREM-REF[HAFiscal-Latest @ 71ca7c61 :: theory/powerlaw-decay/final_proof.md :: §7. The computational payoff: why the compactified core is the right presentation :: The knife-edge window, quantified on HAFiscal's own numbers]
        #   Near the r = g knife-edge (lambda_B near 1, Lambda near 0) the
        #   asymptotic regime begins only at astronomically large wealth: at the
        #   GIC-cap calibration the compensated gap has covered only 42% of the
        #   way to B_psi at the HAFiscal grid top, 91% even at 130x the grid top.
        #   The plateau-onset scale is ln x >~ ln x_c + O(1)/(q*-1).
        _record(
            f"NEAR-RESONANCE: lambda_B = E[psi^2]/(Rcal*Thorn_Gamma) = "
            f"{lambda_B:.6f} is within {near_resonance_band:.0%} of the q* = 1 "
            f"(r = g) knife-edge — the pre-asymptotic window is long: on any "
            f"feasible grid a FITTED tail constant will understate the "
            f"closed-form amplitude B_psi, and B_psi (if defined) is approached "
            f"only at astronomically large wealth. Treat B_psi as an asymptotic "
            f"boundary value, not a fit target.", NearResonanceWarning)

    # ---- sigma_B2, B_psi, c_J (need h)
    sigma_B2 = None
    B_psi = None
    c_J = None
    resonance_slope = None
    if FHWC:
        if joint is not None:
            # Master identity Var(W), W = psi*(theta+h) - (1+h): exact on the
            # joint, equals the closed form under independence.
            psi_j, th_j, p_j = joint
            W = psi_j * (th_j + h) - (1 + h)
            EW = (p_j * W).sum()
            sigma_B2 = float((p_j * (W - EW) ** 2).sum())
        else:
            # THEOREM-REF[HAFiscal-Latest @ 71ca7c61 :: theory/powerlaw-decay/final_proof.md :: §5. Theorem III: permanent shocks — the human-wealth-revaluation channel]
            #   sigma_B^2 = Var(W) = E[psi^2]*Var(theta) + (1+h)^2*Var(psi): the
            #   (1+h)^2*Var(psi) term is the human-wealth revaluation channel — a
            #   permanent shock reprices the whole future income stream.
            sigma_B2 = float(_LD(E_psi2) * _LD(Var_theta)
                             + (1 + h) ** 2 * _LD(Var_psi))
        c_J = float(kappa * (rho + 1) * _LD(sigma_B2) / (2 * Thorn_Gamma))
        if float(Lambda) > 0.0:
            resonance_slope = float(kappa * (rho + 1) * _LD(sigma_B2)
                                    / (2 * Lambda))
        denom = float(Rcal * Thorn_Gamma) - E_psi2
        if np.isfinite(q_star) and q_star > 1.0 and denom > 0.0:
            # THEOREM-REF[HAFiscal-Latest @ 71ca7c61 :: theory/powerlaw-decay/final_proof.md :: §5. Theorem III: permanent shocks — the human-wealth-revaluation channel :: Theorem γ-B]
            #   Theorem III (= Theorem γ-B): at q* > 1,
            #   x*g(x) -> B_psi = kappa*(rho+1)*sigma_B^2 / (2*(Rcal*Thorn_Gamma - E[psi^2])),
            #   the closed-form boundary amplitude (Gordon-convergent perpetuity of
            #   precautionary premia); the denominator is positive iff lambda_B < 1
            #   iff q* > 1, so both gates below are equivalent and both guarded.
            B_psi = float(kappa * (rho + 1)) * sigma_B2 / (2.0 * denom)

    # ---- dual (Kesten) root
    zeta_star, E_ln_A, P_A_gt_1, zeta_diag = dual_root(psi_a, psi_p, Thorn_Gamma)
    if zeta_diag is not None and E_ln_A >= 0.0:
        _record(f"NO DUAL ROOT (positive log-drift): {zeta_diag}",
                NoDualRootWarning)

    if warn:
        for msg, cat in zip(warn_list, warn_cats):
            _warnings.warn(msg, cat, stacklevel=2)

    return PowerLawDecayParams(
        Rfree=float(R), PermGroFac=float(G), DiscFac=float(beta), CRRA=float(rho),
        LivPrb=float(liv), beta_eff=float(beta_eff),
        Rcal=float(Rcal), Thorn=float(Thorn), Thorn_R=float(Thorn_R),
        Thorn_Gamma=float(Thorn_Gamma), kappa=float(kappa), Lambda=float(Lambda),
        h=float(h),
        E_inc=E_inc, E_psi2=E_psi2, Var_psi=Var_psi, Var_theta=Var_theta,
        q_star=q_star, q=q, diagnosis=(q_reason or ""),
        sigma_B2=sigma_B2, lambda_B=lambda_B,
        near_resonance=near_res, resonance_slope=resonance_slope,
        B_psi=B_psi, c_J=c_J,
        zeta_star=zeta_star, dual_diagnosis=(zeta_diag or "ok"),
        E_ln_A=E_ln_A, P_A_gt_1=P_A_gt_1,
        GIC=GIC, RIC=RIC, FHWC=FHWC,
        valid=bool(GIC and RIC and FHWC),
        warnings=tuple(warn_list),
    )


# --------------------------------------------------------------- agent convenience
def _time_indexed(value, t):
    """Scalarize a possibly time-varying HARK parameter the way the solvers do:
    index a list/tuple (or an object-array/IndexDistribution-like sequence) at
    ``t``; pass scalars through."""
    if isinstance(value, (list, tuple)):
        return value[t]
    if isinstance(value, np.ndarray) and value.ndim >= 1:
        return value[t]
    return value


def powerlaw_decay_params_from_agent(agent, t=0, near_resonance_band=0.01,
                                     warn=True):
    """Convenience wrapper for IndShockConsumerType-family agents.

    Reads ``Rfree[t]``, ``PermGroFac[t]``, ``DiscFac``, ``CRRA``, ``LivPrb[t]``,
    ``PermShkDstn[t]`` and ``TranShkDstn[t]`` (falling back to the JOINT
    ``IncShkDstn[t]`` when the marginals are absent, e.g. hand-built income
    processes), scalarizing time-varying lists the same way the solvers do.

    Compute-from-primitives rule: this NEVER reads ``agent.solution[...].hNrm``
    (truncated at default solve tolerance) nor ``agent.bilt['hNrm']`` (BST
    convention, includes current income) — see the h-convention tag in the
    module docstring.  The solver-side theorem-convention human wealth
    in HARK's income units is ``params.h * params.E_inc``.
    """
    kwargs = dict(
        Rfree=_time_indexed(agent.Rfree, t),
        PermGroFac=_time_indexed(agent.PermGroFac, t),
        DiscFac=_time_indexed(agent.DiscFac, t),
        CRRA=_time_indexed(agent.CRRA, t),
        LivPrb=_time_indexed(agent.LivPrb, t),
        near_resonance_band=near_resonance_band,
        warn=warn,
    )
    perm = getattr(agent, "PermShkDstn", None)
    tran = getattr(agent, "TranShkDstn", None)
    if perm is not None and tran is not None:
        return powerlaw_decay_params(
            PermShkDstn=perm[t], TranShkDstn=tran[t], **kwargs)
    inc = getattr(agent, "IncShkDstn", None)
    if inc is None:
        raise ValueError("agent has neither (PermShkDstn, TranShkDstn) nor "
                         "IncShkDstn")
    return powerlaw_decay_params(IncShkDstn=inc[t], **kwargs)


# --------------------------------------------------------------- resonance helper
def resonance_constants(Rfree, PermGroFac, DiscFac, CRRA, LivPrb=1.0,
                        PermShkDstn=None, TranShkDstn=None, IncShkDstn=None,
                        warn=True):
    """Sharp constants of the q* = 1 resonance case.

    # THEOREM-REF[HAFiscal-Latest @ 71ca7c61 :: theory/powerlaw-decay/final_proof.md :: §4. Theorem II: the trichotomy — three things a linear recursion can do at a boundary :: Theorem γ-R]
    #   At q* = 1 (Rcal*Thorn_Gamma = 1, the r = g knife-edge) the shell recursion
    #   escapes linearly instead of contracting to a value: the gap obeys the ln-x
    #   law (x/ln x)*g(x) -> C_B — a slope, not a value. This is a knife-edge, not
    #   a neighborhood: the crossover to it is non-uniform on any window with
    #   |q*-1|*ln x = O(1).

    At exact resonance (``E[psi^2] = Rcal*Thorn_Gamma``, i.e. lambda_B = 1) the
    gap obeys ``x*g(x)/ln(x) -> C_B`` with

        C_B = kappa*(rho+1)*sigma_B^2 / (2*E[psi^2]*Lprime(1)),
        Lprime(1) = E[psi^2 * ln psi]/E[psi^2] - ln(Thorn_Gamma),

    (the psi^2-tilted mean log-step) and the per-tilted-rung Cesaro increment is
    ``c_J/Rcal``; at exact resonance ``c_J/Rcal == C_B * Lprime(1)`` identically.

    Returns a dict: ``C_B``, ``cJ_over_Rcal``, ``Lprime1``, ``lambda_B``,
    ``resonance_residual`` (= |lambda_B - 1|; the constants are the theorem's
    ONLY at exact resonance — a warning-sized residual means you are in the
    q* != 1 regime and should use B_psi or the fitted amplitude instead), plus
    the underlying ``theory`` result (a PowerLawDecayParams).
    """
    th = powerlaw_decay_params(
        Rfree, PermGroFac, DiscFac, CRRA, LivPrb=LivPrb,
        PermShkDstn=PermShkDstn, TranShkDstn=TranShkDstn, IncShkDstn=IncShkDstn,
        warn=warn)
    if th.sigma_B2 is None:
        raise ValueError("resonance_constants: sigma_B2 unavailable "
                         f"(warnings: {list(th.warnings)})")
    # rebuild the psi marginal exactly as the main entry did
    if IncShkDstn is not None:
        psi_j, _, p_j = _as_joint(IncShkDstn)
        psi_a, psi_p = _marginal(psi_j, p_j)
    elif PermShkDstn is not None:
        psi_a, psi_p = _as_atoms_probs(PermShkDstn, "PermShkDstn")
    else:
        psi_a, psi_p = np.array([_LD(1)]), np.array([_LD(1)])
    psf, ppf = np.asarray(psi_a, float), np.asarray(psi_p, float)
    E2 = float(np.dot(ppf, psf ** 2))
    Elog = float(np.dot(ppf, psf ** 2 * np.log(psf)))
    Lprime1 = Elog / E2 - float(np.log(th.Thorn_Gamma))
    C_B = th.kappa * (th.CRRA + 1.0) * th.sigma_B2 / (2.0 * E2 * Lprime1)
    cJ_over_Rcal = th.c_J / th.Rcal
    return dict(C_B=C_B, cJ_over_Rcal=cJ_over_Rcal, Lprime1=Lprime1,
                lambda_B=th.lambda_B, resonance_residual=abs(th.lambda_B - 1.0),
                theory=th)


# --------------------------------------------------------------- tail diagnostic
@dataclass
class TailDiagnostic:
    """Result of :func:`powerlaw_tail_diagnostic` (all fields per the theorem's
    Figure-4 presentation)."""

    verdict: str          # 'CONFIRMED' | 'PRE_ASYMPTOTIC' | 'INCONSISTENT'
    #                       | 'UNMEASURABLE'
    s_grid: np.ndarray    # trial exponents q + trial_offsets
    slopes: np.ndarray    # compensated slope d ln(x^s * gap)/d ln(x) per trial s
    #                       (gamma-T prediction: ~ s - q_true, flat only at the
    #                       true exponent)
    q_theory: float       # min(1, q_star) used as the tested exponent
    window: tuple         # (m_lo, m_hi) of the points that survived the guard
    n_points: int         # points surviving the gap guard
    notes: str


def powerlaw_tail_diagnostic(cFunc, MPCmin, hNrm, params, m_lo=None, m_hi=None,
                             n_pts=40, trial_offsets=(-0.15, 0.0, +0.15),
                             flat_tol=0.08, inconsistency_tol=0.5,
                             guard_rel_gap=1e-9):
    """Wrong-exponent detection: a cheap post-solve grid test (no re-solve).

    # THEOREM-REF[HAFiscal-Latest @ 71ca7c61 :: theory/powerlaw-decay/final_proof.md :: §4. Theorem II: the trichotomy — three things a linear recursion can do at a boundary :: Theorem γ-T]
    #   Theorem γ-T (wrong-exponent detection): compensate the gap by a trial
    #   exponent s — only s = min(1, q*) makes the compensated series flat
    #   (bounded with a positive limit); any other s makes it drift with sign
    #   s - min(1, q*). The compensated-flatness test below is that theorem
    #   read as a diagnostic on a solved consumption function.

    # THEOREM-REF[HAFiscal-Latest @ 71ca7c61 :: theory/powerlaw-decay/final_proof.md :: §7. The computational payoff: why the compactified core is the right presentation :: A built-in diagnostic]
    #   The grid-depth migration of measured exponents toward min(1, q*) is the
    #   theorem-backed convergence signature for solver validation; a flat point
    #   far from min(1, q*) is the actionable wrong-exponent signature (broken
    #   grid / wrong MPCmin-hNrm reference / non-converged solve).

    Sweeps ``cFunc`` once on a log-spaced window, forms the gap
    ``g(m) = MPCmin*(m + hNrm) - cFunc(m)`` against ``x = m + hNrm``, drops
    points below the float-cancellation floor (``gap/c <= guard_rel_gap`` or
    ``gap <= 0``), then fits the windowed log-log slope of the gap over >= 20
    points (NEVER the 2-knot top-slope estimator, which is grid-non-monotone).
    With a single-window fit the compensated slope is exactly affine in the
    trial exponent, ``slope(s) = s - Q_local`` with ``Q_local`` the fitted
    local exponent, so the verdict reduces to the location of the flat point
    ``s = Q_local`` relative to ``q = min(1, q_star)`` — pre-registered
    semantics (not tuned post hoc), with ``center := slope(s=q) = q - Q_local``:

    * ``UNMEASURABLE``: fewer than ``n_pts//2`` points survive the guard
      (deep-grid float cancellation), or ``params.q`` is nan (theory refused).
    * ``CONFIRMED``: ``|center| <= flat_tol`` (default 0.08; the phase-1
      measurement on a deep real solve was +0.039).
    * ``PRE_ASYMPTOTIC``: ``flat_tol < center <= inconsistency_tol`` — the
      local exponent sits BELOW min(1, q*), the theorem-backed transient side
      (the measured exponent migrates upward toward min(1, q*) with grid
      depth); expected at near-resonance calibrations, where a note is
      appended.
    * ``INCONSISTENT``: ``center > inconsistency_tol`` (default 0.5: the flat
      point is far from min(1, q*) — e.g. the h-convention trap turns the gap
      into a constant, Q_local ~ 0) OR ``center < -flat_tol`` (any local
      exponent STEEPER than min(1, q*): the impossibility-floor side, which no
      transient of the true solution produces over a measurable window — Prop A0
      forbids a gap fading faster than 1/x). Equivalently, any ``center`` outside
      the symmetric CONFIRMED band ``[-flat_tol, +flat_tol]`` on the steeper
      (negative) side is INCONSISTENT; the PRE_ASYMPTOTIC band is one-sided
      (``flat_tol < center <= inconsistency_tol``), because migration toward
      min(1, q*) only ever comes from BELOW. The actionable verdict: broken grid,
      wrong MPCmin/hNrm reference, or a non-converged solve.

    Parameters
    ----------
    cFunc : callable
        Solved 1D consumption function (a slice of a 2D solution or a 1D
        cFunc); called as ``cFunc(m_array)``.
    MPCmin : float
        PF asymptote slope kappa (from primitives, e.g. ``params.kappa``).
    hNrm : float
        Theorem-convention human wealth IN THE MODEL'S INCOME UNITS, i.e.
        ``params.h * params.E_inc`` computed from primitives. Do NOT pass
        ``solution.hNrm`` (tolerance-truncated) or ``bilt['hNrm']`` (includes
        current income); either poisons the gap — see the h-convention tag in
        the module docstring.
    params : PowerLawDecayParams
        Theory quantities for the same primitives; supplies q = min(1, q*).
    m_lo, m_hi : float, optional
        Test window. ``m_hi`` defaults to half the top of the solved grid when
        discoverable (``cFunc.x_list``), else it is required; ``m_lo``
        defaults to ``0.1*m_hi``.
    n_pts : int
        Log-spaced sweep size (>= 20 recommended).
    trial_offsets : tuple of float
        Figure-4 probe offsets around q (must include 0.0).
    flat_tol, inconsistency_tol, guard_rel_gap : float
        Pre-registered thresholds described above.

    Returns
    -------
    TailDiagnostic
    """
    if 0.0 not in tuple(trial_offsets):
        raise ValueError("trial_offsets must include 0.0 (the s = q probe)")
    q = float(getattr(params, "q", float("nan")))
    s_grid = np.array(sorted(q + np.asarray(trial_offsets, dtype=float)))
    if not np.isfinite(q):
        return TailDiagnostic(
            verdict="UNMEASURABLE", s_grid=s_grid,
            slopes=np.full(s_grid.shape, np.nan), q_theory=q,
            window=(np.nan, np.nan), n_points=0,
            notes="theory exponent undefined (q_star is nan): "
                  + (params.diagnosis or "see params.warnings"),
        )
    if m_hi is None:
        x_list = getattr(cFunc, "x_list", None)
        if x_list is None:
            raise ValueError(
                "m_hi is required when the solved grid top is not discoverable "
                "from cFunc (no x_list attribute)"
            )
        m_hi = 0.5 * float(np.asarray(x_list)[-1])
    if m_lo is None:
        m_lo = 0.1 * m_hi
    if not (0.0 < m_lo < m_hi):
        raise ValueError(f"need 0 < m_lo < m_hi, got ({m_lo}, {m_hi})")

    m = np.geomspace(m_lo, m_hi, int(n_pts))
    c = np.asarray(cFunc(m), dtype=float)
    x = m + hNrm
    gap = MPCmin * x - c
    keep = np.isfinite(gap) & np.isfinite(c) & (gap > 0.0) & (c > 0.0)
    keep &= np.where(keep, gap > guard_rel_gap * np.abs(c), False)
    n_keep = int(keep.sum())
    notes = []
    if n_keep < int(n_pts) // 2:
        return TailDiagnostic(
            verdict="UNMEASURABLE", s_grid=s_grid,
            slopes=np.full(s_grid.shape, np.nan), q_theory=q,
            window=(float(m[keep][0]), float(m[keep][-1])) if n_keep else
                   (np.nan, np.nan),
            n_points=n_keep,
            notes=f"only {n_keep}/{int(n_pts)} points survive the "
                  f"gap/c > {guard_rel_gap:g} guard (float-cancellation floor "
                  "or negative gap: wrong reference or window too deep)",
        )
    ln_x = np.log(x[keep])
    ln_gap = np.log(gap[keep])
    # windowed log-log fit over the surviving points (the >= 20-point
    # estimator; never the 2-knot top slope)
    Q_local = -float(np.polyfit(ln_x, ln_gap, 1)[0])
    slopes = s_grid - Q_local
    center = q - Q_local
    if abs(center) <= flat_tol:
        verdict = "CONFIRMED"
    elif flat_tol < center <= inconsistency_tol:
        verdict = "PRE_ASYMPTOTIC"
        if bool(getattr(params, "near_resonance", False)):
            notes.append(
                "expected at this near-resonance calibration; deepen the grid "
                "only if the exponent itself is under test"
            )
    else:
        verdict = "INCONSISTENT"
        if center < 0.0:
            notes.append(
                f"local exponent {Q_local:.3f} STEEPER than min(1, q*) = "
                f"{q:.3f}: impossibility-floor side (Prop A0)"
            )
        else:
            notes.append(
                f"flat point at s = {Q_local:.3f}, far below min(1, q*) = "
                f"{q:.3f}: wrong-exponent signature (broken grid, wrong "
                "MPCmin/hNrm reference, or non-converged solve)"
            )
    notes.insert(0, f"local exponent Q_local = {Q_local:.4f}; "
                    f"center slope(s=q) = {center:+.4f}")
    return TailDiagnostic(
        verdict=verdict, s_grid=s_grid, slopes=slopes, q_theory=q,
        window=(float(m[keep][0]), float(m[keep][-1])), n_points=n_keep,
        notes="; ".join(notes),
    )


# ---------------------------------------------------------------------------
# Certified grid-extent criterion
# ---------------------------------------------------------------------------
def rel_gap_at(cFunc, m, MPCmin, hNrm):
    """Relative consumption gap (kappa*(m + h) - c(m)) / c(m) at ``m``.

    The measurement the extent criterion consumes, and the quantity its
    ex-post certificate bounds.  ``MPCmin``/``hNrm`` must come from
    primitives (``params.kappa`` and ``params.h * params.E_inc`` in solver
    units) — never from ``solution.hNrm`` (tolerance-truncated) or
    ``bilt['hNrm']`` (h + 1 convention).
    """
    m = np.atleast_1d(np.asarray(m, float))
    c = np.asarray(cFunc(m), float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (MPCmin * (m + hNrm) - c) / c
    # scalar in -> scalar out; empty in -> empty out (never index size 0)
    return float(out[0]) if out.size == 1 else out


def aXtraMax_from_tail_tol(m_ref, rel_gap_ref, q_eff, hNrm, tail_tol,
                           safety=1.5, B_psi=None, MPCmin=None):
    """Grid-extent inversion: the m at which the RELATIVE consumption gap
    reaches ``tail_tol`` — i.e. where the grid may stop and hand the tail to
    the power-law extrapolation with a certified error bound.

    # THEOREM-REF[HAFiscal-Latest @ 71ca7c61 :: theory/powerlaw-decay/statement.md :: Theorem A1 :: leading exponent is `min(1, q*)`]
    #   gap(x) ≍ x^(-min(1,q*)) with x = m + h; with c ~ kappa*x the RELATIVE
    #   gap decays one power faster, gap/c ∝ x^(-(1+q)), so a reference
    #   measurement inverts to x_top = x_ref*(rel_gap_ref/tail_tol)^(1/(1+q)).

    Two routes, chosen by what the calibration offers:

    * **q* > 1 (closed-form GUARANTEE — preferred when available).** Pass
      ``B_psi`` and ``MPCmin`` (``params.B_psi``, ``params.kappa``): the
      compensated gap x*g(x) climbs MONOTONICALLY to B_psi, so
      ``gap(x) <= B_psi/x`` everywhere and
      ``x_top = sqrt(B_psi/(MPCmin*tail_tol))`` guarantees the relative gap
      at the top is <= tail_tol (up to an O(tail_tol) denominator
      correction).  Measured at the GIC-cap anchor: ex-post 6.9e-5 at
      tail_tol = 1e-4.
      # THEOREM-REF[HAFiscal-Latest @ 71ca7c61 :: theory/powerlaw-decay/alt_proof_compactified.md :: Theorem γ-B (Stage-B boundary value). PROVEN-HERE. :: `M_n` increasing to the plateau]
      #   W1(z) = x*g(x) extends continuously to z = 0 with boundary value
      #   B_psi, approached monotonically from below at q* > 1 — hence
      #   B_psi/x bounds the gap from above on the whole tail.
    * **q* <= 1 (measured inversion).** No closed-form amplitude exists;
      invert the measured reference gap with the power law above.  CAUTION,
      near-resonance: when ``lambda_B`` is within the warning band the
      compensated amplitude is still climbing on any feasible window, the
      relative gap decays SLOWER than the assumed power law, and the
      inversion under-sizes (measured at the near-resonance q* > 1 anchor:
      ex-post 1.8x the target, converging only slowly under repair — use the
      closed form there instead).  Measured on the estimated HAFiscal
      anchors: HS ex-post 0.40x tail_tol, College-top 0.98x (the default
      safety=1.5 just covers it).

    Ex-post certificate (the guarantee that motivates the criterion): the true
    consumption function is strictly concave with c' > MPCmin at every finite
    m (Carroll-Kimball concavity + RIC/FHWC), so the gap is positive and
    strictly DECREASING; a level-matched, monotone-decaying below-line tail
    extrapolant therefore stays, together with the true c, inside the band
    [PF-line - gap(m_top), PF-line] for every m >= m_top, giving pointwise
    |c_extrap - c_true| <= gap(m_top) and relative error <=
    rel_gap_at(cFunc, m_top, ...) <= tail_tol.  ALWAYS check it after solving
    on the delivered grid (it is the binding guarantee; the a-priori sizing is
    a heuristic); on failure, one repair step is
    ``x_top' = x_top * (measured/tail_tol)^(1/(1+q_eff))``.  The
    amplitude-jump mode of the tail law is excluded from this certificate (it
    is not level-matched).

    Parameters
    ----------
    m_ref : float
        Reference point (typically the top endogenous gridpoint of a coarse
        solve) where the relative gap was MEASURED.
    rel_gap_ref : float
        ``rel_gap_at(cFunc, m_ref, MPCmin, hNrm)`` from that solve.
    q_eff : float
        The inversion exponent's q.  Use ``min(params.q, Q_local)`` with
        Q_local the fitted local log-log gap slope of the coarse solve.  The
        min() is a GUARD for deep reference solves, where the local exponent
        can sit below min(1, q*); at shallow, h-dominated coarse windows the
        fitted slope is INFLATED (a tiny ln-x window) and the guard is
        inert — inverting with the asymptotic q is then the binding choice
        and errs conservative at q* < 1.  The guard cannot help at
        q* >= 1 (it caps at 1): use the closed-form route there.
    hNrm : float
        ``params.h * params.E_inc`` (solver units; see the h-convention
        warning in this module's docstring).
    tail_tol : float
        Target relative gap at the top.  Values below 1e-6 are clamped: the
        measured float64 gap dies into cancellation around gap/c ~ 1e-10, so
        tighter targets are not certifiable ex post.
    safety : float
        Multiplies the ratio before inversion (default 1.5) to absorb
        reference-measurement error; must be finite and positive (else nan).
        The ex-post certificate remains the binding check.
    B_psi, MPCmin : float, optional
        When BOTH are finite and positive (q* > 1 calibrations), the
        closed-form guaranteed route is used and the measured-reference
        arguments are ignored.

    Returns
    -------
    float
        The certified grid top (aXtraMax-scale, same units as m_ref), or nan
        when the inputs cannot support the inversion (non-finite or
        non-positive rel_gap_ref, q_eff, or safety).
    """
    tail_tol = max(float(tail_tol), 1.0e-6) if np.isfinite(tail_tol) \
        else float("nan")
    if not np.isfinite(tail_tol):
        return float("nan")
    if B_psi is not None and MPCmin is not None:
        B_psi, MPCmin = float(B_psi), float(MPCmin)
        if np.isfinite(B_psi) and B_psi > 0.0 and np.isfinite(MPCmin) \
                and MPCmin > 0.0 and np.isfinite(hNrm):
            x_top = np.sqrt(B_psi / (MPCmin * tail_tol))
            return float(x_top - hNrm)
        return float("nan")
    m_ref, rel_gap_ref = float(m_ref), float(rel_gap_ref)
    q_eff, hNrm, safety = float(q_eff), float(hNrm), float(safety)
    if not (np.isfinite(m_ref) and np.isfinite(rel_gap_ref)
            and rel_gap_ref > 0.0 and np.isfinite(q_eff) and q_eff > 0.0
            and np.isfinite(hNrm) and np.isfinite(safety) and safety > 0.0):
        return float("nan")
    x_ref = m_ref + hNrm
    x_top = x_ref * (safety * rel_gap_ref / tail_tol) ** (1.0 / (1.0 + q_eff))
    return float(x_top - hNrm)


# ---------------------------------------------------------------------------
# The operator eigen-probe: numerical q* without the eigen-equation
# ---------------------------------------------------------------------------
def qstar_probe(Rfree, PermGroFac, DiscFac, CRRA, LivPrb=1.0,
                PermShkDstn=None, TranShkDstn=None, IncShkDstn=None,
                x0s=(1e6, 1e7, 1e8), eps=1e-4, s_lo=0.05, s_hi=5.0,
                one_step=None, MPCmin=None, hNrm=None):
    """Measure the decay-exponent root q* NUMERICALLY from the model's own
    one-period backward operator — no closed-form eigen-equation required.

    The exponent is the eigenvalue condition of the period operator acting on
    power-law perturbations of the PF asymptote. Working in the EGM's native
    END-OF-PERIOD-ASSET coordinate (the probe never touches a grid): at deep
    a = x0 - hNrm, apply one backward step to the trial next-period function

        c_trial(m') = MPCmin*(m' + hNrm) - eps*MPCmin*x0*((m' + hNrm)/x0)^(-s),

    difference the eps > 0 and eps = 0 responses (cancels the one-period
    Arrow-Pratt premium, isolates the linear response), and root-find the s
    at which the per-period gap multiplier equals one. Normalizing the trial
    at the probe point is LOAD-BEARING: an unnormalized eps*x^(-s) trial at
    x = 1e8 is ~1e-40, beneath float64 resolution of the gap.

    # THEOREM-REF[HAFiscal-Latest @ 71ca7c61 :: theory/powerlaw-decay/final_proof.md :: §0 "What is q*? (and why min(1, q*))" :: eq (E)]
    #   The analytic eigen-equation E[psi^(1+q)] = Rcal*Thorn_Gamma^q is this
    #   probe evaluated on paper: the root of the one-period multiplier on
    #   x^(-q) gap perturbations. The probe is the model-agnostic form.

    # THEOREM-REF[HAFiscal-Latest @ 8ad5a853 :: theory/powerlaw-decay/grid_design_final_spec.md :: THE SPEC (owner-proposed scheme, sharpened by F1–F8) :: The operator eigen-probe]
    #   Validation (F9): matches the analytic q* to 5.6e-6 / 8.6e-6 / 5.0e-5
    #   at the HS / CTOP / CCAP anchors with ~1e-6 depth-consistency, while
    #   estimation FROM SOLVED VALUES is 15-40% off even on deep windows (the
    #   h-shift starves the identifying x-variation). Outside GIC the probe
    #   (true finite-x operator) and the (E)-root (idealized limit) disagree —
    #   report q_hat only alongside the condition flags.

    Portability: for the standard CRRA model, primitives suffice (the step is
    built internally). For OTHER models, pass ``one_step(c_trial, a_array) ->
    c_today_array`` (the model's backward step applied to a supplied
    next-period consumption function) together with that model's PF limits
    ``MPCmin``/``hNrm`` — anything solved by time iteration has all three.

    Returns
    -------
    (q_hat, consistency, diagnosis) : (float, float, str)
        q_hat: the multiplier's unit root (realized decay exponent is
        min(1, q_hat) — the universal 1/x premium floor, Prop A0); nan when
        no root lies in (s_lo, s_hi) or inputs fail (never raises).
        consistency: relative spread of the multiplier across the probe
        depths ``x0s`` at the root (a built-in self-check; ~1e-6 measured).
        diagnosis: 'ok' or the failure reason.
    """
    try:
        if MPCmin is None or hNrm is None:
            base = powerlaw_decay_params(
                Rfree, PermGroFac, DiscFac, CRRA, LivPrb=LivPrb,
                PermShkDstn=PermShkDstn, TranShkDstn=TranShkDstn,
                IncShkDstn=IncShkDstn, warn=False)
            MPCmin = base.kappa if MPCmin is None else MPCmin
            hNrm = base.h * base.E_inc if hNrm is None else hNrm
        MPCmin, hNrm = float(MPCmin), float(hNrm)
        if not (np.isfinite(MPCmin) and MPCmin > 0.0 and np.isfinite(hNrm)):
            return float("nan"), float("nan"), \
                "PF asymptote unavailable (MPCmin/hNrm non-finite)"
        if one_step is None:
            R = float(_time_indexed(Rfree, 0))
            G = float(_time_indexed(PermGroFac, 0))
            L = float(_time_indexed(LivPrb, 0))
            beta, rho = float(DiscFac), float(CRRA)
            if IncShkDstn is not None:
                psi_j, th_j, wp = _as_joint(IncShkDstn)
                psi_j = np.asarray(psi_j, float)
                th_j = np.asarray(th_j, float)
                wp = np.asarray(wp, float)
            else:
                if PermShkDstn is None:
                    psi_a, psi_p = np.array([1.0]), np.array([1.0])
                else:
                    psi_a, psi_p = _as_atoms_probs(PermShkDstn, "PermShkDstn")
                th_a, th_p = _as_atoms_probs(TranShkDstn, "TranShkDstn")
                PSI, TH = np.meshgrid(np.asarray(psi_a, float),
                                      np.asarray(th_a, float), indexing="ij")
                wp = np.outer(np.asarray(psi_p, float),
                              np.asarray(th_p, float)).ravel()
                psi_j, th_j = PSI.ravel(), TH.ravel()
            wp = wp / wp.sum()

            def one_step(c_trial, a):
                m_img = (R / (G * psi_j))[None, :] * a[:, None] + th_j[None, :]
                c_next = c_trial(m_img)
                rhs = beta * L * R * (wp[None, :]
                                      * (G * psi_j[None, :]) ** (-rho)
                                      * c_next ** (-rho)).sum(1)
                return rhs ** (-1.0 / rho)

        def lam_at(s, x0):
            a = np.array([x0 - hNrm])
            c0 = MPCmin * x0

            def response(e):
                def c_trial(m_img):
                    x_img = m_img + hNrm
                    return MPCmin * x_img - e * c0 * (x_img / x0) ** (-s)
                c_t = one_step(c_trial, a)
                x_t = float(a[0] + c_t[0] + hNrm)
                return float(MPCmin * x_t - c_t[0]), x_t

            g1, x_t = response(0.0)
            g2, _ = response(eps)
            return (g2 - g1) / (eps * c0 * (x_t / x0) ** (-s))

        def lam(s):
            vals = [lam_at(s, float(x0)) for x0 in x0s]
            m = float(np.mean(vals))
            return m, (float(np.std(vals) / abs(m)) if m != 0.0
                       else float("inf"))

        lo, hi = float(s_lo), float(s_hi)
        flo = lam(lo)[0] - 1.0
        fhi = lam(hi)[0] - 1.0
        if not (np.isfinite(flo) and np.isfinite(fhi)) or flo * fhi > 0:
            return float("nan"), float("nan"), \
                f"no unit-multiplier root in ({s_lo}, {s_hi}) (fails closed)"
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            fm = lam(mid)[0] - 1.0
            if flo * fm <= 0:
                hi = mid
            else:
                lo, flo = mid, fm
        q_hat = 0.5 * (lo + hi)
        return float(q_hat), lam(q_hat)[1], "ok"
    except Exception as exc:  # behavior contract: never raises
        return float("nan"), float("nan"), f"probe failed: {exc!r}"


# ---------------------------------------------------------------------------
# Stable points (targets), with the mortality-adjusted (L*R) loci
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StablePoints:
    """Roots of the consumption function against the stable-point loci, in
    BOTH coordinates: the m-roots and their end-of-period-asset images
    a = m - c(m) (the grid-relevant objects — the asset grid, the consumed
    function c(a), and everything the solver controls live in a-space)."""
    mNrmTrg: float          # E[m'] = m (needs GIC-Mod); nan if no crossing
    mNrmStE: float          # E[psi*m'] = m, balanced LEVEL growth (GIC-Raw)
    mNrmTrg_mort: float     # Delta-m=0 locus with R -> LivPrb*R (GIC-Mod-Liv)
    mNrmStE_mort: float     # balanced-growth locus with R -> LivPrb*R
    aNrmTrg: float = float("nan")
    aNrmStE: float = float("nan")
    aNrmTrg_mort: float = float("nan")
    aNrmStE_mort: float = float("nan")
    E_theta: float = float("nan")
    E_inv_psi: float = float("nan")
    notes: str = ""


def mNrm_stable_points(cFunc, Rfree, PermGroFac, LivPrb=1.0,
                       PermShkDstn=None, TranShkDstn=None, IncShkDstn=None,
                       m_hi=1.0e4):
    """All four stable points of a solved consumption function: the two
    classical loci and their mortality-adjusted twins, plus a-space images.

    Loci (independent shocks, E[psi] = 1; a(m) = m - c(m)):
      Trg  (E[m'] = m):      c(m) = m - (m - E[theta]) / ((R/Gamma)*E[1/psi])
      StE  (E[psi*m'] = m):  c(m) = m - (m - E[theta]) * Gamma/R
      *_mort: replace R by LivPrb*R. With perpetual-youth replacement
      (newborns at a = 0 drawing the same transitory income), mortality
      factors EXACTLY as this return shave in the cross-sectional mean
      dynamics, so the adjusted loci exist under GIC-Mod-Liv / GIC-Raw-Liv
      even when the unadjusted target does not (the pure-GIC case).

    # THEOREM-REF[HAFiscal-Latest @ 8ad5a853 :: theory/powerlaw-decay/grid_design_final_spec.md :: The findings ledger (each measured this arc, HS/CTOP/CCAP anchors) :: mortality EXACTLY as a return shave]
    #   Measured at the GIC-cap atom (GIC-Mod fails; unadjusted StE = 340.9):
    #   the L*R-adjusted balanced-growth root 40.18 lands on the
    #   Harmenberg-neutral ergodic mean 41.23 to 2.5% (Jensen sign correct,
    #   a(m) convex). The MEASURE picks the locus: neutral-measure
    #   aggregation has E_N[1/psi] = 1, so its mean-dynamics anchor is the
    #   StE locus; the raw cross-section carries E[1/psi] and anchors on Trg.

    ``cFunc`` is any callable m -> c (duck-typed, like
    ``powerlaw_tail_diagnostic``). Roots by bisection on
    [E[theta]/2 + 1e-6, m_hi]; nan (never an exception) when a locus does not
    cross — existence requires the corresponding growth condition.
    """
    R = float(_time_indexed(Rfree, 0))
    G = float(_time_indexed(PermGroFac, 0))
    L = float(_time_indexed(LivPrb, 0))
    if IncShkDstn is not None:
        psi_j, th_j, wp = _as_joint(IncShkDstn)
        psi_j = np.asarray(psi_j, float)
        th_j = np.asarray(th_j, float)
        wp = np.asarray(wp, float) / np.asarray(wp, float).sum()
        E_th = float((wp * th_j).sum())
        E_ip = float((wp / psi_j).sum())
    else:
        if PermShkDstn is None:
            E_ip = 1.0
        else:
            pa, pp = _as_atoms_probs(PermShkDstn, "PermShkDstn")
            E_ip = float((np.asarray(pp, float) / np.asarray(pa, float)).sum())
        ta, tp = _as_atoms_probs(TranShkDstn, "TranShkDstn")
        E_th = float((np.asarray(tp, float) * np.asarray(ta, float)).sum())

    def _root(locus):
        lo = 0.5 * E_th + 1e-6
        hi = float(m_hi)
        f = lambda m: float(np.atleast_1d(cFunc(np.array([m])))[0]) - locus(m)
        try:
            flo, fhi = f(lo), f(hi)
        except Exception:
            return float("nan")
        if not (np.isfinite(flo) and np.isfinite(fhi)) or flo * fhi > 0:
            return float("nan")
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            fm = f(mid)
            if flo * fm <= 0:
                hi = mid
            else:
                lo, flo = mid, fm
            if hi - lo < 1e-12 * max(1.0, mid):
                break
        return 0.5 * (lo + hi)

    m_trg = _root(lambda m: m - (m - E_th) / ((R / G) * E_ip))
    m_ste = _root(lambda m: m - (m - E_th) * G / R)
    m_trg_L = _root(lambda m: m - (m - E_th) / ((L * R / G) * E_ip))
    m_ste_L = _root(lambda m: m - (m - E_th) * G / (L * R))

    def _a_of(m):
        if not np.isfinite(m):
            return float("nan")
        return float(m - np.atleast_1d(cFunc(np.array([m])))[0])

    return StablePoints(
        mNrmTrg=m_trg, mNrmStE=m_ste,
        mNrmTrg_mort=m_trg_L, mNrmStE_mort=m_ste_L,
        aNrmTrg=_a_of(m_trg), aNrmStE=_a_of(m_ste),
        aNrmTrg_mort=_a_of(m_trg_L), aNrmStE_mort=_a_of(m_ste_L),
        E_theta=E_th, E_inv_psi=E_ip,
        notes="mort loci: R -> LivPrb*R (exact for cross-sectional mean "
              "dynamics under perpetual-youth a=0 newborns)")

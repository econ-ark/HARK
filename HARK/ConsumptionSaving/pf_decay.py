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
#   least today's Arrow-Pratt premium), so exponential decay is impossible as an
#   asymptotic form and any fitted exponent above min(1, q*) is theory-infeasible.

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
    "PFDecayConditionWarning",
    "PFDecayGridWarning",
    "NearResonanceWarning",
    "NoDualRootWarning",
    "ShockCorrelationWarning",
]

_LD = np.longdouble

_BRACKET_CAP = 1024.0  # bracket-expansion cap for both root searches


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

    def L(q):
        return float(np.log(np.dot(ppf, psf ** (1.0 + q)))
                     - np.log(RG) - q * np.log(PG))

    L0 = L(0.0)
    if L0 >= 0.0:
        return float("nan"), (
            f"no (E)-root: L(0) = -ln(Rcal) = {L0:.6g} >= 0 (FHWC violated: "
            f"Rcal = {RG:.6g} <= 1)")
    hi = 1.0
    while L(hi) < 0.0 and hi < _BRACKET_CAP:
        hi *= 2.0
    if L(hi) < 0.0:
        return float("nan"), (
            f"no (E)-root found in (0, {_BRACKET_CAP:g}]: L never crosses zero "
            f"(with Thorn_Gamma = {PG:.6g} >= 1 and degenerate/narrow psi, L(q) is "
            f"non-increasing — GIC violated)")
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
    if f(hi) < 0.0:  # not reachable for finite atoms with P(A>1)>0, kept for safety
        return None, E_ln_A, P_A_gt_1, (
            f"no dual root found in (0, {_BRACKET_CAP:g}] despite E[ln A] < 0 and "
            f"P(A > 1) > 0 (bracket cap hit)")
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

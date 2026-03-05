"""
Life-cycle model of housing and portfolio choice with mortgage debt.

Baseline (two-state) model from the FINRA grant proposal:
- Continuous states: m_t = M_t/P_t (cash-on-hand), d_t = D_t/H_t (LTV ratio)
- Continuous choices: c_t (consumption), varsigma_t (risky share)
- Discrete choices: tenure (own/sell/default) x participation (6 branches)
- Common income-housing shock: G_{t+1} = P_{t+1}/P_t = H_{t+1}/H_t

Value function:
    V(M,D,P,H) = P^{(1+alpha)(1-rho)} * v(m,d)

Bellman (owner):
    v_t(m,d) = max_{c,varsigma} hbar^{alpha(1-rho)} * c^{1-rho}/(1-rho)
               + beta * s_t * E[G^{(1+alpha)(1-rho)} * v_{t+1}(m',d')]

Terminal:
    v_T(m,d) = omega * (m + (1-d)*hbar)^{(1+alpha)(1-rho)} / ((1+alpha)(1-rho))
"""

from copy import deepcopy

import numpy as np

from HARK import NullFunc
from HARK.Calibration.Assets.AssetProcesses import (
    calc_ShareLimit_for_CRRA,
    combine_IncShkDstn_and_RiskyDstn,
    make_lognormal_RiskyDstn,
)
from HARK.Calibration.Income.IncomeProcesses import (
    construct_lognormal_income_process_unemployment,
    get_PermShkDstn_from_IncShkDstn,
    get_TranShkDstn_from_IncShkDstn,
)
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
from HARK.ConsumptionSaving.ConsRiskyAssetModel import make_simple_ShareGrid
from HARK.interpolation import (
    ConstantFunction,
    LinearInterp,
    LinearInterpOnInterp1D,
)
from HARK.metric import MetricObject
from HARK.utilities import make_assets_grid

__all__ = [
    "HousingPortfolioSolution",
    "HousingPortfolioConsumerType",
]


# ---------------------------------------------------------------------------
# Mortgage payment utilities
# ---------------------------------------------------------------------------


def mortgage_payment_rate(d, r_m, periods_remaining):
    """Fixed-rate mortgage payment as fraction of house value.

    Parameters
    ----------
    d : float or array
        Loan-to-value ratio D_t/H_t.
    r_m : float
        Mortgage interest rate (per period).
    periods_remaining : int
        Number of mortgage payments remaining (T^m - t).

    Returns
    -------
    pi_tilde : same shape as d
        Payment as fraction of current house value.
    """
    if periods_remaining <= 0:
        return np.zeros_like(d)
    if r_m == 0.0:
        # Zero-interest mortgage: equal principal payments
        return d / periods_remaining
    R_m = 1.0 + r_m
    factor = r_m * R_m**periods_remaining / (R_m**periods_remaining - 1.0)
    return d * factor


def ltv_next(d, r_m, periods_remaining, G):
    """LTV evolution under amortization.

    d_{t+1} = (d_t * R_m - pi_tilde_t(d_t)) / G_{t+1}
    """
    R_m = 1.0 + r_m
    pi = mortgage_payment_rate(d, r_m, periods_remaining)
    return (d * R_m - pi) / G


# ---------------------------------------------------------------------------
# Solution object
# ---------------------------------------------------------------------------


class HousingPortfolioSolution(MetricObject):
    """Single-period solution for the housing portfolio choice model.

    Stores policy and value functions for three tenure states (own, rent)
    and across the (m, d) state space for owners.

    Attributes
    ----------
    cFuncOwn : callable(m, d) -> c
        Consumption function for homeowners.
    ShareFuncOwn : callable(m, d) -> varsigma
        Risky share function for homeowners.
    vFuncOwn : callable(m, d) -> v
        Value function for homeowners.
    vPfuncOwn : callable(m, d) -> dv/dm
        Marginal value of cash-on-hand for homeowners.
    tenureFunc : callable(m, d) -> int
        Tenure choice: 0=own, 1=sell, 2=default.
    cFuncRent : callable(m) -> c
        Consumption function for renters.
    ShareFuncRent : callable(m) -> varsigma
        Risky share function for renters.
    vFuncRent : callable(m) -> v
        Value function for renters.
    vPfuncRent : callable(m) -> dv/dm
        Marginal value of cash-on-hand for renters.
    """

    distance_criteria = ["vPfuncOwn"]

    def __init__(
        self,
        cFuncOwn=None,
        ShareFuncOwn=None,
        vFuncOwn=None,
        vPfuncOwn=None,
        tenureFunc=None,
        cFuncRent=None,
        ShareFuncRent=None,
        vFuncRent=None,
        vPfuncRent=None,
    ):
        self.cFuncOwn = cFuncOwn if cFuncOwn is not None else NullFunc()
        self.ShareFuncOwn = ShareFuncOwn if ShareFuncOwn is not None else NullFunc()
        self.vFuncOwn = vFuncOwn if vFuncOwn is not None else NullFunc()
        self.vPfuncOwn = vPfuncOwn if vPfuncOwn is not None else NullFunc()
        self.tenureFunc = tenureFunc
        self.cFuncRent = cFuncRent if cFuncRent is not None else NullFunc()
        self.ShareFuncRent = (
            ShareFuncRent if ShareFuncRent is not None else NullFunc()
        )
        self.vFuncRent = vFuncRent if vFuncRent is not None else NullFunc()
        self.vPfuncRent = vPfuncRent if vPfuncRent is not None else NullFunc()
        # Store grids for diagnostics
        self.mGrid = None
        self.dGrid = None


# ---------------------------------------------------------------------------
# Terminal solution constructor
# ---------------------------------------------------------------------------


def make_housing_portfolio_solution_terminal(
    CRRA, alpha, hbar, BeqWt, **kwargs
):
    """Construct the terminal-period solution for the housing portfolio model.

    At terminal age T the household liquidates housing and bequeaths total
    net worth w_T = m_T + (1-d_T)*hbar:

        v_T(m,d) = omega * (m + (1-d)*hbar)^{(1+alpha)(1-rho)} / ((1+alpha)(1-rho))

    Parameters
    ----------
    CRRA : float
        Coefficient of relative risk aversion (rho).
    alpha : float
        Housing preference parameter.
    hbar : float
        Fixed housing-to-income ratio.
    BeqWt : float
        Bequest weight (omega).

    Returns
    -------
    solution_terminal : HousingPortfolioSolution
    """
    rho = CRRA
    gamma = (1.0 + alpha) * (1.0 - rho)  # homogeneity exponent

    # --- Owner terminal functions over (m, d) ---

    def _total_wealth(m, d):
        return np.maximum(m + (1.0 - d) * hbar, 1e-6)

    def _v_own_terminal(m, d):
        w = _total_wealth(m, d)
        return BeqWt * w**gamma / gamma

    def _vp_own_terminal(m, d):
        w = _total_wealth(m, d)
        return BeqWt * w ** (gamma - 1.0)

    def _c_own_terminal(m, d):
        # At terminal period, consume all liquid wealth
        return np.maximum(m, 1e-12)

    # --- Renter terminal functions over m only ---
    # Renter has no housing equity; bequeaths only liquid wealth
    def _v_rent_terminal(m):
        w = np.maximum(m, 1e-12)
        return BeqWt * w**gamma / gamma

    def _vp_rent_terminal(m):
        w = np.maximum(m, 1e-12)
        return BeqWt * w ** (gamma - 1.0)

    def _c_rent_terminal(m):
        return np.maximum(m, 1e-12)

    solution_terminal = HousingPortfolioSolution(
        cFuncOwn=_c_own_terminal,
        ShareFuncOwn=ConstantFunction(0.0),
        vFuncOwn=_v_own_terminal,
        vPfuncOwn=_vp_own_terminal,
        tenureFunc=None,
        cFuncRent=_c_rent_terminal,
        ShareFuncRent=ConstantFunction(0.0),
        vFuncRent=_v_rent_terminal,
        vPfuncRent=_vp_rent_terminal,
    )
    return solution_terminal


# ---------------------------------------------------------------------------
# LTV grid constructor
# ---------------------------------------------------------------------------


def make_ltv_grid(dGridCount, dMax, **kwargs):
    """Create a grid of LTV ratios from 0 to dMax.

    Parameters
    ----------
    dGridCount : int
        Number of grid points.
    dMax : float
        Maximum LTV ratio (e.g. 0.95 or 1.2 to allow underwater).

    Returns
    -------
    dGrid : np.ndarray
    """
    return np.linspace(0.0, dMax, dGridCount)


# ---------------------------------------------------------------------------
# One-period solver: renter subproblem
# ---------------------------------------------------------------------------


def solve_renter_subproblem(
    solution_next,
    IncShkDstn,
    RiskyDstn,
    LivPrb,
    DiscFac,
    CRRA,
    alpha,
    Rfree,
    PermGroFac,
    aXtraGrid,
    ShareGrid,
    hbar,
    RentRate,
    ParticCost,
    IndepDstnBool=True,
):
    """Solve the renter's one-period problem.

    .. note:: Currently assumes independent income and return shocks
       (``IndepDstnBool=True``). Correlated shocks are not yet supported.

    The renter chooses c_t, ell_t (housing services), and varsigma_t.
    With Cobb-Douglas u(c, ell) = (c * ell^alpha)^{1-rho}/(1-rho),
    optimal ell given c satisfies ell = alpha*c/(rbar), yielding indirect
    utility over total expenditure.

    For simplicity in the baseline, we assume the renter picks ell optimally
    each period, reducing to a single continuous state m with modified
    CRRA utility and a rental cost that scales consumption.

    Parameters
    ----------
    solution_next : HousingPortfolioSolution
    IncShkDstn, RiskyDstn : distributions
    LivPrb, DiscFac, CRRA, alpha, Rfree, PermGroFac : float
    aXtraGrid, ShareGrid : np.ndarray
    hbar : float
    RentRate : float
        Rental rate per unit of housing services (rbar).
    ParticCost : float
        Participation cost chi.
    IndepDstnBool : bool

    Returns
    -------
    cFuncRent, ShareFuncRent, vFuncRent, vPfuncRent : callables
    """
    rho = CRRA
    gamma = (1.0 + alpha) * (1.0 - rho)

    # With optimal housing services, the renter's problem reduces to
    # choosing c with modified utility:
    #   u_renter(c) = kappa_r * c^{gamma} / gamma
    # where kappa_r = (alpha/rbar)^{alpha(1-rho)} absorbs the optimal ell choice.
    # The Euler equation for c is standard with effective CRRA = -(gamma-1):
    # u'(c) = kappa_r * c^{gamma-1}
    # In practice, we use the indirect utility and its inverse for EGM.

    kappa_r = (alpha / RentRate) ** (alpha * (1.0 - rho))
    effective_crra = 1.0 - gamma  # = (1+alpha)(rho-1)+1 = 1-(1+alpha)(1-rho)

    DiscFacEff = DiscFac * LivPrb

    # End-of-period assets grid
    aNrmGrid = aXtraGrid

    # For the renter, next period wealth: m' = y' + a*R_port / G'
    # where R_port = varsigma*R^s + (1-varsigma)*R^f
    # and the continuation value uses vFuncRent_next (renter stays renter for now).

    # We solve a 1D EGM over (a, Share) like the standard portfolio model.
    aNrmCount = aNrmGrid.size
    ShareCount = ShareGrid.size

    # Get next-period functions
    vPfuncRent_next = solution_next.vPfuncRent

    # Compute end-of-period marginal value for each (a, Share) combination
    EndOfPrd_dvda = np.zeros((aNrmCount, ShareCount))

    for i_s, share in enumerate(ShareGrid):
        for i_a, a_nrm in enumerate(aNrmGrid):
            if a_nrm < 1e-12:
                EndOfPrd_dvda[i_a, i_s] = 1e10  # very high marginal value at 0
                continue

            # Integrate over income and return shocks
            dvda_vals = []
            for i_inc in range(IncShkDstn.pmv.size):
                perm_shk = IncShkDstn.atoms[0, i_inc]
                tran_shk = IncShkDstn.atoms[1, i_inc]
                prob_inc = IncShkDstn.pmv[i_inc]

                for i_ret in range(RiskyDstn.pmv.size):
                    risky_ret = RiskyDstn.atoms[0, i_ret]
                    prob_ret = RiskyDstn.pmv[i_ret]

                    R_port = share * risky_ret + (1.0 - share) * Rfree
                    G_next = PermGroFac * perm_shk
                    m_next = tran_shk + a_nrm * R_port / G_next

                    dvdm_next = vPfuncRent_next(m_next)
                    # Chain rule: dv/da = R_port / G * G^gamma * dvdm_next
                    # The G^gamma comes from the value function scaling
                    dvda_val = (
                        R_port / G_next * G_next**gamma * dvdm_next
                    )
                    dvda_vals.append(prob_inc * prob_ret * dvda_val)

            EndOfPrd_dvda[i_a, i_s] = DiscFacEff * np.sum(dvda_vals)

    # For each Share, do EGM to find optimal consumption
    # FOC: kappa_r * c^{gamma-1} = EndOfPrd_dvda  =>  c = (EndOfPrd_dvda / kappa_r)^{1/(gamma-1)}
    cNrmGrid = np.zeros((aNrmCount, ShareCount))
    mNrmGrid = np.zeros((aNrmCount, ShareCount))

    for i_s in range(ShareCount):
        dvda = EndOfPrd_dvda[:, i_s]
        # Invert marginal utility
        c = (dvda / kappa_r) ** (1.0 / (gamma - 1.0))
        c = np.maximum(c, 1e-12)
        cNrmGrid[:, i_s] = c
        mNrmGrid[:, i_s] = c + aNrmGrid

    # Now find optimal share for each a by checking which share gives highest value
    # For simplicity, use the share that satisfies dvds = 0 (interpolate over shares)
    # or pick the share that maximizes end-of-period value.

    # Compute end-of-period value for share optimization
    EndOfPrd_v = np.zeros((aNrmCount, ShareCount))
    for i_s, share in enumerate(ShareGrid):
        for i_a, a_nrm in enumerate(aNrmGrid):
            if a_nrm < 1e-12:
                EndOfPrd_v[i_a, i_s] = -1e10
                continue

            v_vals = []
            for i_inc in range(IncShkDstn.pmv.size):
                perm_shk = IncShkDstn.atoms[0, i_inc]
                tran_shk = IncShkDstn.atoms[1, i_inc]
                prob_inc = IncShkDstn.pmv[i_inc]

                for i_ret in range(RiskyDstn.pmv.size):
                    risky_ret = RiskyDstn.atoms[0, i_ret]
                    prob_ret = RiskyDstn.pmv[i_ret]

                    R_port = share * risky_ret + (1.0 - share) * Rfree
                    G_next = PermGroFac * perm_shk
                    m_next = tran_shk + a_nrm * R_port / G_next

                    v_next = solution_next.vFuncRent(m_next)
                    v_val = G_next**gamma * v_next
                    v_vals.append(prob_inc * prob_ret * v_val)

            EndOfPrd_v[i_a, i_s] = DiscFacEff * np.sum(v_vals)

    # Optimal share at each a: argmax over ShareGrid
    opt_share_idx = np.argmax(EndOfPrd_v, axis=1)
    opt_share = ShareGrid[opt_share_idx]

    # Extract c and m at optimal share
    c_opt = np.array([cNrmGrid[i_a, opt_share_idx[i_a]] for i_a in range(aNrmCount)])
    m_opt = np.array([mNrmGrid[i_a, opt_share_idx[i_a]] for i_a in range(aNrmCount)])

    # Handle participation cost: non-participant gets varsigma=0
    # Participation branch
    c_partic = c_opt.copy()
    m_partic = m_opt + ParticCost  # cost shifts m rightward in EGM

    # Non-participation branch: varsigma=0
    c_nopartic = cNrmGrid[:, 0].copy()  # Share=0 column
    m_nopartic = mNrmGrid[:, 0].copy()

    # Build interpolants
    # Sort by m for interpolation
    sort_p = np.argsort(m_partic)
    sort_np = np.argsort(m_nopartic)

    m_p = np.concatenate([[0.0], m_partic[sort_p]])
    c_p = np.concatenate([[0.0], c_partic[sort_p]])
    s_p = np.concatenate([[opt_share[sort_p[0]]], opt_share[sort_p]])

    m_np = np.concatenate([[0.0], m_nopartic[sort_np]])
    c_np = np.concatenate([[0.0], c_nopartic[sort_np]])

    cFunc_partic = LinearInterp(m_p, c_p)
    shareFunc_partic = LinearInterp(m_p, s_p)
    cFunc_nopartic = LinearInterp(m_np, c_np)

    # Continuation value interpolants for participation decision.
    # EndOfPrd_v[i_a, i_s] = beta*s*E[G^gamma * v_{t+1}(m')] at (a, Share).
    EndOfPrd_v_opt = np.array(
        [EndOfPrd_v[i_a, opt_share_idx[i_a]] for i_a in range(aNrmCount)]
    )
    cont_v_partic = LinearInterp(
        np.concatenate([[0.0], aNrmGrid]),
        np.concatenate([[EndOfPrd_v_opt[0]], EndOfPrd_v_opt]),
    )
    cont_v_nopartic = LinearInterp(
        np.concatenate([[0.0], aNrmGrid]),
        np.concatenate([[EndOfPrd_v[0, 0]], EndOfPrd_v[:, 0]]),
    )

    # Combined functions: compare total value (flow + continuation)
    def cFuncRent(m):
        m = np.asarray(m, dtype=float)
        scalar = m.ndim == 0
        m = np.atleast_1d(m)

        c_p = cFunc_partic(m)
        a_p = np.maximum(m - c_p - ParticCost, 0.0)
        v_p = kappa_r * c_p**gamma / gamma + cont_v_partic(a_p)

        c_np = cFunc_nopartic(m)
        a_np = np.maximum(m - c_np, 0.0)
        v_np = kappa_r * c_np**gamma / gamma + cont_v_nopartic(a_np)

        c_out = np.where(v_p > v_np, c_p, c_np)
        return float(c_out[0]) if scalar else c_out

    def ShareFuncRent(m):
        m = np.asarray(m, dtype=float)
        scalar = m.ndim == 0
        m = np.atleast_1d(m)

        c_p = cFunc_partic(m)
        a_p = np.maximum(m - c_p - ParticCost, 0.0)
        v_p = kappa_r * c_p**gamma / gamma + cont_v_partic(a_p)

        c_np = cFunc_nopartic(m)
        a_np = np.maximum(m - c_np, 0.0)
        v_np = kappa_r * c_np**gamma / gamma + cont_v_nopartic(a_np)

        s_out = np.where(v_p > v_np, shareFunc_partic(m), 0.0)
        return float(s_out[0]) if scalar else s_out

    # Build vFunc and vPfunc for renters on a fine grid.
    # Extend domain to cover tenure envelope evaluation range.
    m_fine_max = max(m_p[-1] * 1.5, aNrmGrid[-1] * 2.0)
    m_fine = np.linspace(1e-6, m_fine_max, 200)
    v_fine = np.zeros_like(m_fine)
    vp_fine = np.zeros_like(m_fine)

    for i_m, mi in enumerate(m_fine):
        c_p_i = float(cFunc_partic(mi))
        a_p_i = max(mi - c_p_i - ParticCost, 0.0)
        v_p_i = kappa_r * c_p_i**gamma / gamma + float(cont_v_partic(a_p_i))

        c_np_i = float(cFunc_nopartic(mi))
        a_np_i = max(mi - c_np_i, 0.0)
        v_np_i = kappa_r * c_np_i**gamma / gamma + float(cont_v_nopartic(a_np_i))

        if v_p_i > v_np_i:
            v_fine[i_m] = v_p_i
            vp_fine[i_m] = kappa_r * c_p_i ** (gamma - 1.0)
        else:
            v_fine[i_m] = v_np_i
            vp_fine[i_m] = kappa_r * c_np_i ** (gamma - 1.0)

    vFuncRent = LinearInterp(m_fine, v_fine)
    vPfuncRent = LinearInterp(m_fine, vp_fine)

    return cFuncRent, ShareFuncRent, vFuncRent, vPfuncRent


# ---------------------------------------------------------------------------
# One-period solver: owner subproblem
# ---------------------------------------------------------------------------


def solve_owner_subproblem(
    solution_next,
    IncShkDstn,
    RiskyDstn,
    LivPrb,
    DiscFac,
    CRRA,
    alpha,
    Rfree,
    PermGroFac,
    aXtraGrid,
    ShareGrid,
    dGrid,
    hbar,
    MortRate,
    MortPeriods,
    MaintRate,
    ParticCost,
    IndepDstnBool=True,
):
    """Solve the homeowner's one-period continuation problem (conditional on owning).

    .. note:: Currently assumes independent income and return shocks
       (``IndepDstnBool=True``). Correlated shocks are not yet supported.

    Solves the Bellman:
        v^own_t(m,d) = max_{c,varsigma} hbar^{alpha(1-rho)} c^{1-rho}/(1-rho)
                       + beta * s_t * E[G^{(1+alpha)(1-rho)} v_{t+1}(m',d')]

    Uses a 2D endogenous grid method over (a, d) with discrete share optimization.

    Returns
    -------
    cFuncOwn, ShareFuncOwn, vFuncOwn, vPfuncOwn : 2D callables over (m, d)
    """
    rho = CRRA
    gamma = (1.0 + alpha) * (1.0 - rho)

    # Flow utility multiplier from fixed housing services
    h_mult = hbar ** (alpha * (1.0 - rho))

    DiscFacEff = DiscFac * LivPrb

    aNrmCount = aXtraGrid.size
    ShareCount = ShareGrid.size
    dCount = dGrid.size

    # For each (a, d, Share), compute end-of-period value and marginal value
    EndOfPrd_dvda = np.zeros((aNrmCount, dCount, ShareCount))
    EndOfPrd_v = np.zeros((aNrmCount, dCount, ShareCount))

    r_m = MortRate
    R_m = 1.0 + r_m

    for i_d, d_now in enumerate(dGrid):
        pi_now = mortgage_payment_rate(d_now, r_m, MortPeriods)

        for i_s, share in enumerate(ShareGrid):
            for i_a, a_nrm in enumerate(aXtraGrid):
                if a_nrm < 1e-12:
                    EndOfPrd_dvda[i_a, i_d, i_s] = 1e10
                    EndOfPrd_v[i_a, i_d, i_s] = -1e10
                    continue

                dvda_accum = 0.0
                v_accum = 0.0

                for i_inc in range(IncShkDstn.pmv.size):
                    perm_shk = IncShkDstn.atoms[0, i_inc]
                    tran_shk = IncShkDstn.atoms[1, i_inc]
                    prob_inc = IncShkDstn.pmv[i_inc]

                    for i_ret in range(RiskyDstn.pmv.size):
                        risky_ret = RiskyDstn.atoms[0, i_ret]
                        prob_ret = RiskyDstn.pmv[i_ret]

                        R_port = share * risky_ret + (1.0 - share) * Rfree
                        G_next = PermGroFac * perm_shk

                        # Next-period states
                        m_next = tran_shk + a_nrm * R_port / G_next
                        d_next = (d_now * R_m - pi_now) / G_next

                        # Clamp d_next to grid bounds; the owner subproblem
                        # is conditional on staying, so the tenure envelope
                        # handles cases where exit would be optimal.
                        d_next = np.clip(d_next, dGrid[0], dGrid[-1])

                        # Owner continuation
                        dvdm_next = solution_next.vPfuncOwn(m_next, d_next)
                        v_next = solution_next.vFuncOwn(m_next, d_next)

                        prob = prob_inc * prob_ret
                        G_factor = G_next**gamma

                        dvda_accum += prob * R_port / G_next * G_factor * dvdm_next
                        v_accum += prob * G_factor * v_next

                EndOfPrd_dvda[i_a, i_d, i_s] = DiscFacEff * dvda_accum
                EndOfPrd_v[i_a, i_d, i_s] = DiscFacEff * v_accum

    # --- EGM inversion for each (d, Share) ---
    # FOC: h_mult * c^{-rho} = EndOfPrd_dvda
    # => c = (EndOfPrd_dvda / h_mult)^{-1/rho}
    cNrmGrid = np.zeros((aNrmCount, dCount, ShareCount))
    mNrmGrid = np.zeros((aNrmCount, dCount, ShareCount))

    for i_d, d_now in enumerate(dGrid):
        pi_now = mortgage_payment_rate(d_now, r_m, MortPeriods)
        mandatory_cost = pi_now * hbar + MaintRate * hbar  # in income units

        for i_s in range(ShareCount):
            dvda = EndOfPrd_dvda[:, i_d, i_s]
            c = (dvda / h_mult) ** (-1.0 / rho)
            c = np.maximum(c, 1e-12)
            cNrmGrid[:, i_d, i_s] = c
            # m = c + a + mandatory cost + participation cost (if participating)
            chi = ParticCost if ShareGrid[i_s] > 0 else 0.0
            mNrmGrid[:, i_d, i_s] = c + aXtraGrid + mandatory_cost + chi

    # --- Optimal share for each (a, d): maximize end-of-period value ---
    opt_share_idx = np.argmax(EndOfPrd_v, axis=2)  # (aNrmCount, dCount)
    opt_share = ShareGrid[opt_share_idx]

    # Extract policy at optimal share
    c_star = np.zeros((aNrmCount, dCount))
    m_star = np.zeros((aNrmCount, dCount))
    v_star = np.zeros((aNrmCount, dCount))

    for i_d in range(dCount):
        for i_a in range(aNrmCount):
            i_s = opt_share_idx[i_a, i_d]
            c_star[i_a, i_d] = cNrmGrid[i_a, i_d, i_s]
            m_star[i_a, i_d] = mNrmGrid[i_a, i_d, i_s]

    # Compute value at optimal choices
    for i_d in range(dCount):
        for i_a in range(aNrmCount):
            c = c_star[i_a, i_d]
            a = aXtraGrid[i_a]
            i_s = opt_share_idx[i_a, i_d]
            flow = h_mult * c ** (1.0 - rho) / (1.0 - rho)
            cont = EndOfPrd_v[i_a, i_d, i_s]
            v_star[i_a, i_d] = flow + cont

    # Marginal value at optimal choices
    vp_star = h_mult * c_star ** (-rho)

    # --- Build 2D interpolants over (m, d) ---
    # For each d gridpoint, build a 1D interpolant over m, then combine
    cFunc_list = []
    shareFunc_list = []
    vFunc_list = []
    vpFunc_list = []

    for i_d in range(dCount):
        m_col = m_star[:, i_d]
        c_col = c_star[:, i_d]
        s_col = opt_share[:, i_d]
        v_col = v_star[:, i_d]
        vp_col = vp_star[:, i_d]

        # Sort by m
        sort_idx = np.argsort(m_col)
        m_sorted = np.concatenate([[0.0], m_col[sort_idx]])
        c_sorted = np.concatenate([[0.0], c_col[sort_idx]])
        s_sorted = np.concatenate([[0.0], s_col[sort_idx]])
        v_sorted = np.concatenate([[-1e10], v_col[sort_idx]])
        vp_sorted = np.concatenate([[1e10], vp_col[sort_idx]])

        cFunc_list.append(LinearInterp(m_sorted, c_sorted))
        shareFunc_list.append(LinearInterp(m_sorted, s_sorted))
        vFunc_list.append(LinearInterp(m_sorted, v_sorted))
        vpFunc_list.append(LinearInterp(m_sorted, vp_sorted))

    # Wrap as 2D functions using LinearInterpOnInterp1D
    cFuncOwn = LinearInterpOnInterp1D(cFunc_list, dGrid)
    ShareFuncOwn = LinearInterpOnInterp1D(shareFunc_list, dGrid)
    vFuncOwn_raw = LinearInterpOnInterp1D(vFunc_list, dGrid)
    vPfuncOwn_raw = LinearInterpOnInterp1D(vpFunc_list, dGrid)

    return cFuncOwn, ShareFuncOwn, vFuncOwn_raw, vPfuncOwn_raw


# ---------------------------------------------------------------------------
# One-period solver: tenure choice (own vs sell vs default)
# ---------------------------------------------------------------------------


def solve_one_period_HousingPortfolio(
    solution_next,
    IncShkDstn,
    RiskyDstn,
    ShockDstn,
    LivPrb,
    DiscFac,
    CRRA,
    alpha,
    Rfree,
    PermGroFac,
    aXtraGrid,
    ShareGrid,
    dGrid,
    hbar,
    MortRate,
    MortPeriods,
    MaintRate,
    ParticCost,
    SellCost,
    DefaultPenalty,
    RentRate,
    BoroCnstArt,
    IndepDstnBool,
    ShareLimit,
    **kwargs,
):
    """Solve one period of the housing portfolio choice model.

    This function:
    1. Solves the renter subproblem (1D in m).
    2. Solves the owner continuation subproblem (2D in (m, d)).
    3. Computes sell and default values at each (m, d).
    4. Takes the upper envelope across tenure choices.

    Parameters
    ----------
    solution_next : HousingPortfolioSolution
    IncShkDstn, RiskyDstn, ShockDstn : distributions
    LivPrb, DiscFac, CRRA, alpha, Rfree, PermGroFac : float
    aXtraGrid, ShareGrid, dGrid : np.ndarray
    hbar, MortRate, MaintRate, ParticCost : float
    MortPeriods : int
    SellCost : float (kappa)
    DefaultPenalty : float (zeta, utility penalty)
    RentRate : float (rbar)
    BoroCnstArt : float
    IndepDstnBool : bool
    ShareLimit : float

    Returns
    -------
    solution_now : HousingPortfolioSolution
    """
    rho = CRRA
    gamma = (1.0 + alpha) * (1.0 - rho)

    # Step 1: Solve the renter subproblem
    cFuncRent, ShareFuncRent, vFuncRent, vPfuncRent = solve_renter_subproblem(
        solution_next,
        IncShkDstn,
        RiskyDstn,
        LivPrb,
        DiscFac,
        CRRA,
        alpha,
        Rfree,
        PermGroFac,
        aXtraGrid,
        ShareGrid,
        hbar,
        RentRate,
        ParticCost,
        IndepDstnBool,
    )

    # Step 2: Solve the owner continuation (conditional on staying)
    cFuncOwn, ShareFuncOwn, vFuncOwn_stay, vPfuncOwn_stay = solve_owner_subproblem(
        solution_next,
        IncShkDstn,
        RiskyDstn,
        LivPrb,
        DiscFac,
        CRRA,
        alpha,
        Rfree,
        PermGroFac,
        aXtraGrid,
        ShareGrid,
        dGrid,
        hbar,
        MortRate,
        MortPeriods,
        MaintRate,
        ParticCost,
        IndepDstnBool,
    )

    # Step 3: Compute sell and default values at each (m, d) grid point
    # Selling: household receives net equity (1-kappa-d)*hbar, transitions to renter
    # Default: household keeps m, suffers utility penalty zeta, transitions to renter

    # Build a fine evaluation grid for the tenure envelope
    m_eval = np.linspace(1e-6, aXtraGrid[-1] * 1.5, 100)
    d_eval = dGrid.copy()

    # Create arrays for the three tenure values
    v_own = np.zeros((m_eval.size, d_eval.size))
    v_sell = np.zeros((m_eval.size, d_eval.size))
    v_default = np.zeros((m_eval.size, d_eval.size))
    tenure_choice = np.zeros((m_eval.size, d_eval.size), dtype=int)

    for i_m, m in enumerate(m_eval):
        for i_d, d in enumerate(d_eval):
            # Own value
            v_own[i_m, i_d] = vFuncOwn_stay(m, d)

            # Sell value: net equity added to liquid wealth, become renter
            net_equity = (1.0 - SellCost - d) * hbar
            m_after_sell = m + net_equity
            if m_after_sell >= 0:
                v_sell[i_m, i_d] = vFuncRent(max(m_after_sell, 1e-6))
            else:
                v_sell[i_m, i_d] = -np.inf  # Can't sell if underwater net of costs

            # Default value: keep liquid wealth, lose house, pay utility penalty
            v_default[i_m, i_d] = vFuncRent(m) - DefaultPenalty

    # Take upper envelope
    v_stack = np.stack([v_own, v_sell, v_default], axis=2)
    tenure_choice = np.argmax(v_stack, axis=2)
    v_best = np.max(v_stack, axis=2)

    # Build final owner value and policy functions incorporating tenure choice
    # For policy: if tenure=0 (own), use owner's c and share; if 1 (sell) or 2 (default), use renter's
    c_final = np.zeros((m_eval.size, d_eval.size))
    share_final = np.zeros((m_eval.size, d_eval.size))
    vp_final = np.zeros((m_eval.size, d_eval.size))

    for i_m, m in enumerate(m_eval):
        for i_d, d in enumerate(d_eval):
            choice = tenure_choice[i_m, i_d]
            if choice == 0:  # Own
                c_final[i_m, i_d] = cFuncOwn(m, d)
                share_final[i_m, i_d] = ShareFuncOwn(m, d)
                vp_final[i_m, i_d] = vPfuncOwn_stay(m, d)
            elif choice == 1:  # Sell
                net_equity = (1.0 - SellCost - d) * hbar
                m_after = m + net_equity
                c_final[i_m, i_d] = cFuncRent(max(m_after, 1e-6))
                share_final[i_m, i_d] = ShareFuncRent(max(m_after, 1e-6))
                vp_final[i_m, i_d] = vPfuncRent(max(m_after, 1e-6))
            else:  # Default
                c_final[i_m, i_d] = cFuncRent(m)
                share_final[i_m, i_d] = ShareFuncRent(m)
                vp_final[i_m, i_d] = vPfuncRent(m)

    # Build 2D interpolants for the final (enveloped) owner functions
    cFunc_own_list = []
    shareFunc_own_list = []
    vFunc_own_list = []
    vpFunc_own_list = []
    tenure_list = []

    for i_d in range(d_eval.size):
        cFunc_own_list.append(LinearInterp(m_eval, c_final[:, i_d]))
        shareFunc_own_list.append(LinearInterp(m_eval, share_final[:, i_d]))
        vFunc_own_list.append(LinearInterp(m_eval, v_best[:, i_d]))
        vpFunc_own_list.append(LinearInterp(m_eval, vp_final[:, i_d]))
        tenure_list.append(
            LinearInterp(m_eval, tenure_choice[:, i_d].astype(float))
        )

    cFuncOwn_final = LinearInterpOnInterp1D(cFunc_own_list, d_eval)
    ShareFuncOwn_final = LinearInterpOnInterp1D(shareFunc_own_list, d_eval)
    vFuncOwn_final = LinearInterpOnInterp1D(vFunc_own_list, d_eval)
    vPfuncOwn_final = LinearInterpOnInterp1D(vpFunc_own_list, d_eval)
    tenureFunc = LinearInterpOnInterp1D(tenure_list, d_eval)

    solution_now = HousingPortfolioSolution(
        cFuncOwn=cFuncOwn_final,
        ShareFuncOwn=ShareFuncOwn_final,
        vFuncOwn=vFuncOwn_final,
        vPfuncOwn=vPfuncOwn_final,
        tenureFunc=tenureFunc,
        cFuncRent=cFuncRent,
        ShareFuncRent=ShareFuncRent,
        vFuncRent=vFuncRent,
        vPfuncRent=vPfuncRent,
    )
    solution_now.mGrid = m_eval
    solution_now.dGrid = d_eval

    return solution_now


# ---------------------------------------------------------------------------
# Calibration defaults
# ---------------------------------------------------------------------------

# Cocco (2005) calibration for a college-educated household
_life_span = 50  # age 25-75
_work_span = 40  # age 25-65, retire at 65
_retire_age = 40  # period index of retirement

# Survival probabilities (simplified; constant then declining)
_surv_probs = [0.997] * 30 + [0.99] * 10 + [0.97] * 5 + [0.94] * 5

# Permanent income growth factors (hump-shaped, then flat in retirement)
_perm_gro_fac = (
    [1.03] * 10  # fast growth early career
    + [1.02] * 10  # moderate growth
    + [1.01] * 10  # slow growth
    + [1.00] * 10  # flat late career
    + [1.00] * 10  # retirement
)

HousingPortfolioType_constructors_default = {
    "IncShkDstn": construct_lognormal_income_process_unemployment,
    "PermShkDstn": get_PermShkDstn_from_IncShkDstn,
    "TranShkDstn": get_TranShkDstn_from_IncShkDstn,
    "aXtraGrid": make_assets_grid,
    "RiskyDstn": make_lognormal_RiskyDstn,
    "ShockDstn": combine_IncShkDstn_and_RiskyDstn,
    "ShareLimit": calc_ShareLimit_for_CRRA,
    "ShareGrid": make_simple_ShareGrid,
    "dGrid": make_ltv_grid,
    "solution_terminal": make_housing_portfolio_solution_terminal,
}

HousingPortfolioType_IncShkDstn_default = {
    "PermShkStd": [0.1] * _life_span,
    "PermShkCount": 5,
    "TranShkStd": [0.1] * _life_span,
    "TranShkCount": 5,
    "UnempPrb": 0.05,
    "IncUnemp": 0.3,
    "T_retire": _retire_age,
    "UnempPrbRet": 0.005,
    "IncUnempRet": 0.0,
}

HousingPortfolioType_aXtraGrid_default = {
    "aXtraMin": 0.001,
    "aXtraMax": 50,
    "aXtraNestFac": 1,
    "aXtraCount": 48,
    "aXtraExtra": None,
}

HousingPortfolioType_RiskyDstn_default = {
    "RiskyAvg": 1.06,  # 6% mean equity return
    "RiskyStd": 0.18,  # ~18% annual volatility
    "RiskyCount": 5,
}

HousingPortfolioType_ShareGrid_default = {
    "ShareCount": 15,
}

HousingPortfolioType_dGrid_default = {
    "dGridCount": 10,
    "dMax": 1.2,  # Allow underwater mortgages
}

HousingPortfolioType_solving_default = {
    "cycles": 1,
    "T_cycle": _life_span,
    "constructors": HousingPortfolioType_constructors_default,
    # Preferences
    "CRRA": 5.0,
    "alpha": 0.2,  # Housing preference weight
    "DiscFac": 0.96,
    "LivPrb": _surv_probs,
    "PermGroFac": _perm_gro_fac,
    "Rfree": [1.02] * _life_span,
    "BoroCnstArt": 0.0,
    # Housing and mortgage
    "hbar": 3.0,  # House value = 3x annual income
    "MortRate": 0.04,  # 4% mortgage rate
    "MortPeriods": [max(30 - t, 0) for t in range(_life_span)],
    "MaintRate": 0.02,  # 2% maintenance as fraction of house value
    "SellCost": 0.06,  # 6% transaction cost on sale
    "DefaultPenalty": 0.5,  # Utility penalty for default (calibrated)
    "RentRate": 0.05,  # Annual rental rate per unit housing
    "ParticCost": 0.01,  # Participation cost (fraction of perm income)
    "BeqWt": 1.0,  # Bequest weight omega
    # Shock structure
    "IndepDstnBool": True,
    "DiscreteShareBool": True,
    "vFuncBool": True,
    "CubicBool": False,
}

HousingPortfolioType_simulation_default = {
    "AgentCount": 5000,
    "T_age": _life_span,
    "PermGroFacAgg": 1.0,
    "NewbornTransShk": False,
    "PerfMITShk": False,
}

HousingPortfolioType_kNrmInitDstn_default = {
    "kLogInitMean": -12.0,
    "kLogInitStd": 0.0,
    "kNrmInitCount": 15,
}

HousingPortfolioType_pLvlInitDstn_default = {
    "pLogInitMean": 0.0,
    "pLogInitStd": 0.0,
    "pLvlInitCount": 15,
}

# Merge all defaults
HousingPortfolioType_default = {}
HousingPortfolioType_default.update(HousingPortfolioType_solving_default)
HousingPortfolioType_default.update(HousingPortfolioType_simulation_default)
HousingPortfolioType_default.update(HousingPortfolioType_kNrmInitDstn_default)
HousingPortfolioType_default.update(HousingPortfolioType_pLvlInitDstn_default)
HousingPortfolioType_default.update(HousingPortfolioType_aXtraGrid_default)
HousingPortfolioType_default.update(HousingPortfolioType_ShareGrid_default)
HousingPortfolioType_default.update(HousingPortfolioType_dGrid_default)
HousingPortfolioType_default.update(HousingPortfolioType_IncShkDstn_default)
HousingPortfolioType_default.update(HousingPortfolioType_RiskyDstn_default)

init_housing_portfolio = HousingPortfolioType_default


# ---------------------------------------------------------------------------
# Agent type
# ---------------------------------------------------------------------------


class HousingPortfolioConsumerType(IndShockConsumerType):
    """Life-cycle consumer with housing, mortgage debt, and portfolio choice.

    This agent owns a house financed by a fixed-rate mortgage, chooses
    consumption and portfolio allocation each period, and may sell or
    default on the mortgage. After exiting homeownership, the agent
    rents housing services.

    State variables:
        m_t = M_t/P_t : normalized cash-on-hand
        d_t = D_t/H_t : loan-to-value ratio

    Choice variables:
        c_t : consumption (normalized)
        varsigma_t : risky portfolio share
        tenure : own / sell / default (discrete)
        participate : yes / no (discrete)
    """

    IncShkDstn_default = HousingPortfolioType_IncShkDstn_default
    aXtraGrid_default = HousingPortfolioType_aXtraGrid_default
    ShareGrid_default = HousingPortfolioType_ShareGrid_default
    RiskyDstn_default = HousingPortfolioType_RiskyDstn_default
    solving_default = HousingPortfolioType_solving_default
    simulation_default = HousingPortfolioType_simulation_default

    default_ = {
        "params": HousingPortfolioType_default,
        "solver": solve_one_period_HousingPortfolio,
        "track_vars": ["mNrm", "cNrm", "Share", "dNrm", "tenure"],
    }

    # Override time_inv_ to list only params our solver needs (and that are
    # time-invariant). This replaces the inherited list from IndShockConsumerType.
    time_inv_ = [
        "CRRA",
        "DiscFac",
        "BoroCnstArt",
        "aXtraGrid",
        "vFuncBool",
        "CubicBool",
        "alpha",
        "hbar",
        "MortRate",
        "MaintRate",
        "SellCost",
        "DefaultPenalty",
        "RentRate",
        "ParticCost",
        "BeqWt",
        "ShareGrid",
        "dGrid",
        "ShareLimit",
        "IndepDstnBool",
        "RiskyDstn",
        "ShockDstn",
    ]

    time_vary_ = [
        "LivPrb",
        "PermGroFac",
        "Rfree",
        "IncShkDstn",
        "PermShkDstn",
        "TranShkDstn",
        "MortPeriods",
    ]

    def __init__(self, verbose=False, quiet=True, **kwds):
        params = deepcopy(HousingPortfolioType_default)
        params.update(kwds)
        super().__init__(verbose=verbose, quiet=quiet, **params)

    def pre_solve(self):
        """Construct objects needed before solving."""
        # Build grids and distributions
        self.construct("IncShkDstn")
        self.construct("PermShkDstn")
        self.construct("TranShkDstn")
        self.construct("aXtraGrid")
        self.construct("RiskyDstn")
        self.construct("ShockDstn")
        self.construct("ShareLimit")
        self.construct("ShareGrid")
        self.construct("dGrid")
        # Build terminal solution
        self.construct("solution_terminal")

    def check_restrictions(self):
        """Check model parameter restrictions."""
        if self.DiscFac < 0:
            raise ValueError(f"DiscFac is below zero: {self.DiscFac}")
        if self.CRRA <= 1.0:
            raise ValueError(
                f"CRRA must be > 1 for this model (got {self.CRRA}). "
                "The model requires gamma = (1+alpha)(1-CRRA) < 0."
            )
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive: {self.alpha}")
        if self.hbar <= 0:
            raise ValueError(f"hbar must be positive: {self.hbar}")
        if self.RentRate <= 0:
            raise ValueError(f"RentRate must be positive: {self.RentRate}")
        if self.MortRate < 0:
            raise ValueError(f"MortRate must be non-negative: {self.MortRate}")

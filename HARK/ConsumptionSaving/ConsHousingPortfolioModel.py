"""
Life-cycle model of housing and portfolio choice with mortgage debt.

Baseline (two-state) model from the FINRA grant proposal:
- Continuous states: m_t = M_t/P_t (cash-on-hand), d_t = D_t/H_t (LTV ratio)
- Continuous choices: c_t (consumption), varsigma_t (risky share)
- Discrete choices: tenure (own / sell / default) and market participation (in / out)
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
from HARK.dcegm import calc_nondecreasing_segments, upper_envelope
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
from HARK.distributions.discrete import DiscreteDistribution
from HARK.utilities import make_assets_grid

__all__ = [
    "HousingPortfolioSolution",
    "HousingPortfolioConsumerType",
    "MarkovHousingPortfolioConsumerType",
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
        result = np.zeros_like(d)
        return float(result) if np.ndim(d) == 0 else result
    if abs(r_m) < 1e-12:
        # Zero (or near-zero) interest mortgage: equal principal payments
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


def _build_joint_shocks(inc_dstn, risky_rets_base, prob_ret_arr,
                        PermGroFac, gamma, StockIncCorr):
    """Build joint (income x return) shock arrays for vectorized integration.

    Parameters
    ----------
    inc_dstn : DiscreteDistribution
        Income shock distribution with atoms[0]=perm, atoms[1]=tran.
    risky_rets_base : np.ndarray
        Return distribution atoms.
    prob_ret_arr : np.ndarray
        Return distribution probabilities.
    PermGroFac : float
        Permanent income growth factor.
    gamma : float
        Homogeneity exponent (1+alpha)(1-rho).
    StockIncCorr : float
        Loading of log equity return on log permanent shock.

    Returns
    -------
    perm_j, tran_j, prob_j, risky_j, G_j, G_gamma_j : np.ndarray
        Joint shock arrays of length N_inc * N_ret.
    """
    perm_j = np.repeat(inc_dstn.atoms[0], risky_rets_base.size)
    tran_j = np.repeat(inc_dstn.atoms[1], risky_rets_base.size)
    prob_j = np.repeat(inc_dstn.pmv, risky_rets_base.size) * np.tile(
        prob_ret_arr, inc_dstn.pmv.size
    )
    risky_j = np.tile(risky_rets_base, inc_dstn.pmv.size)
    if StockIncCorr != 0.0:
        risky_j = risky_j * perm_j**StockIncCorr
    G_j = PermGroFac * perm_j
    G_gamma_j = G_j**gamma
    return perm_j, tran_j, prob_j, risky_j, G_j, G_gamma_j


def _renter_egm_envelope(
    EndOfPrd_dvda, EndOfPrd_v, aNrmGrid, ShareGrid,
    alpha, rho, kappa_r, gamma, ParticCost,
):
    """EGM inversion + DC-EGM participation envelope for the renter.

    Shared between base and Markov renter solvers. Takes end-of-period
    marginal value and value arrays (after integration), returns interpolated
    policy and value functions.

    Parameters
    ----------
    EndOfPrd_dvda, EndOfPrd_v : np.ndarray, shape (aNrmCount, ShareCount)
    aNrmGrid, ShareGrid : np.ndarray
    alpha, rho, kappa_r, gamma, ParticCost : float

    Returns
    -------
    cFunc, ShareFunc, vFunc, vPfunc : LinearInterp
    """
    aNrmCount = aNrmGrid.size
    ShareCount = ShareGrid.size

    # FOC: kappa_r * c^{gamma-1} = EndOfPrd_dvda
    cNrmGrid = (EndOfPrd_dvda / kappa_r) ** (1.0 / (gamma - 1.0))
    cNrmGrid = np.maximum(cNrmGrid, 1e-12)
    mNrmGrid = (1.0 + alpha) * cNrmGrid + aNrmGrid[:, np.newaxis]

    # Best share among positive shares (participation branch)
    arange_a = np.arange(aNrmCount)
    if ShareCount > 1:
        opt_pos_idx = np.argmax(EndOfPrd_v[:, 1:], axis=1) + 1
    else:
        opt_pos_idx = np.zeros(aNrmCount, dtype=int)
    c_partic = cNrmGrid[arange_a, opt_pos_idx]
    m_partic = mNrmGrid[arange_a, opt_pos_idx] + ParticCost
    s_partic = ShareGrid[opt_pos_idx]
    v_partic = (
        kappa_r * c_partic**gamma / (1.0 - rho) + EndOfPrd_v[arange_a, opt_pos_idx]
    )

    # Non-participation branch: share = 0
    c_nopartic = cNrmGrid[:, 0]
    m_nopartic = mNrmGrid[:, 0]
    v_nopartic = kappa_r * c_nopartic**gamma / (1.0 - rho) + EndOfPrd_v[:, 0]

    # DC-EGM upper envelope across participation decision
    m_env, c_env, s_env, v_env = _participation_envelope(
        m_partic, c_partic, v_partic, s_partic,
        m_nopartic, c_nopartic, v_nopartic,
    )
    vp_env = np.concatenate([[1e10], kappa_r * c_env[1:] ** (gamma - 1.0)])

    return (
        LinearInterp(m_env, c_env),
        LinearInterp(m_env, s_env),
        LinearInterp(m_env, v_env),
        LinearInterp(m_env, vp_env),
    )


def _owner_egm_envelope(
    EndOfPrd_dvda, EndOfPrd_v, aXtraGrid, ShareGrid, dGrid,
    rho, h_mult, hbar, r_m, MortPeriods, MaintRate, ParticCost,
):
    """EGM inversion + DC-EGM participation envelope for the owner.

    Shared between base and Markov owner solvers.

    Parameters
    ----------
    EndOfPrd_dvda, EndOfPrd_v : np.ndarray, shape (aNrmCount, dCount, ShareCount)
    aXtraGrid, ShareGrid, dGrid : np.ndarray
    rho, h_mult, hbar, r_m : float
    MortPeriods : int
    MaintRate, ParticCost : float

    Returns
    -------
    cFuncOwn, ShareFuncOwn, vFuncOwn, vPfuncOwn : LinearInterpOnInterp1D
    """
    aNrmCount = aXtraGrid.size
    ShareCount = ShareGrid.size
    dCount = dGrid.size

    # EGM inversion for each (d, Share)
    cNrmGrid = np.zeros((aNrmCount, dCount, ShareCount))
    mNrmGrid = np.zeros((aNrmCount, dCount, ShareCount))

    for i_d, d_now in enumerate(dGrid):
        pi_now = mortgage_payment_rate(d_now, r_m, MortPeriods)
        mandatory_cost = pi_now * hbar + MaintRate * hbar

        for i_s in range(ShareCount):
            dvda = EndOfPrd_dvda[:, i_d, i_s]
            c = (dvda / h_mult) ** (-1.0 / rho)
            c = np.maximum(c, 1e-12)
            cNrmGrid[:, i_d, i_s] = c
            chi = ParticCost if i_s > 0 else 0.0
            mNrmGrid[:, i_d, i_s] = c + aXtraGrid + mandatory_cost + chi

    # DC-EGM participation envelope for each d slice
    arange_a = np.arange(aNrmCount)
    if ShareCount > 1:
        opt_pos_idx = np.argmax(EndOfPrd_v[:, :, 1:], axis=2) + 1
    else:
        opt_pos_idx = np.zeros((aNrmCount, dCount), dtype=int)

    cFunc_list = []
    shareFunc_list = []
    vFunc_list = []
    vpFunc_list = []

    for i_d in range(dCount):
        idx_p = opt_pos_idx[:, i_d]
        c_p = cNrmGrid[arange_a, i_d, idx_p]
        m_p = mNrmGrid[arange_a, i_d, idx_p]
        s_p = ShareGrid[idx_p]
        v_p = h_mult * c_p ** (1.0 - rho) / (1.0 - rho) + EndOfPrd_v[arange_a, i_d, idx_p]

        c_np = cNrmGrid[:, i_d, 0]
        m_np = mNrmGrid[:, i_d, 0]
        v_np = h_mult * c_np ** (1.0 - rho) / (1.0 - rho) + EndOfPrd_v[:, i_d, 0]

        m_env, c_env, s_env, v_env = _participation_envelope(
            m_p, c_p, v_p, s_p, m_np, c_np, v_np,
        )
        vp_env = np.concatenate([[1e10], h_mult * c_env[1:] ** (-rho)])

        cFunc_list.append(LinearInterp(m_env, c_env))
        shareFunc_list.append(LinearInterp(m_env, s_env))
        vFunc_list.append(LinearInterp(m_env, v_env))
        vpFunc_list.append(LinearInterp(m_env, vp_env))

    return (
        LinearInterpOnInterp1D(cFunc_list, dGrid),
        LinearInterpOnInterp1D(shareFunc_list, dGrid),
        LinearInterpOnInterp1D(vFunc_list, dGrid),
        LinearInterpOnInterp1D(vpFunc_list, dGrid),
    )


def _participation_envelope(m_partic, c_partic, v_partic, s_partic,
                            m_nopartic, c_nopartic, v_nopartic):
    """Apply DC-EGM upper envelope across participation branches.

    Splits each branch into non-decreasing segments, then computes the upper
    envelope to find exact crossing points between participation and
    non-participation value functions.

    Parameters
    ----------
    m_partic, c_partic, v_partic, s_partic : np.ndarray
        Endogenous m, consumption, value, and share for participation branch.
    m_nopartic, c_nopartic, v_nopartic : np.ndarray
        Same for non-participation branch (share is 0).

    Returns
    -------
    m_env, c_env, s_env, v_env : np.ndarray
        Upper envelope arrays with boundary point (0, ...) prepended.
    """
    s_nopartic = np.zeros_like(c_nopartic)

    all_v_segs = []
    all_c_data = []
    all_s_data = []

    for m_br, c_br, v_br, s_br in [
        (m_partic, c_partic, v_partic, s_partic),
        (m_nopartic, c_nopartic, v_nopartic, s_nopartic),
    ]:
        starts, ends = calc_nondecreasing_segments(m_br, v_br)
        for j in range(len(starts)):
            sl = slice(starts[j], ends[j] + 1)
            all_v_segs.append([m_br[sl].copy(), v_br[sl].copy()])
            all_c_data.append((m_br[sl].copy(), c_br[sl].copy()))
            all_s_data.append((m_br[sl].copy(), s_br[sl].copy()))

    m_env, v_env, env_inds = upper_envelope(all_v_segs)

    # Interpolate c and share from winning segments
    c_env = np.empty_like(m_env)
    s_env = np.empty_like(m_env)
    for seg_idx in range(len(all_v_segs)):
        mask = env_inds == seg_idx
        if mask.any():
            c_env[mask] = np.interp(
                m_env[mask], all_c_data[seg_idx][0], all_c_data[seg_idx][1],
            )
            s_env[mask] = np.interp(
                m_env[mask], all_s_data[seg_idx][0], all_s_data[seg_idx][1],
            )

    # Prepend boundary
    m_env = np.concatenate([[0.0], m_env])
    c_env = np.concatenate([[0.0], c_env])
    s_env = np.concatenate([[0.0], s_env])
    v_env = np.concatenate([[-1e10], v_env])

    return m_env, c_env, s_env, v_env


def _value_envelope(m_eval, v_arrays, policy_arrays):
    """Apply upper envelope across discrete value function branches.

    Used for tenure choice (own/sell/default) and repurchase (rent/buy)
    envelopes where value functions are already interpolated on a common
    m grid and each branch is monotonically non-decreasing.

    Parameters
    ----------
    m_eval : np.ndarray
        Common m grid where all branches are evaluated.
    v_arrays : list of np.ndarray
        Value function evaluations for each branch, shape (len(m_eval),).
        Use -np.inf for infeasible points.
    policy_arrays : list of tuples of np.ndarray
        For each branch, a tuple of policy arrays (c, share, vp, ...) on m_eval.

    Returns
    -------
    m_env : np.ndarray
        Envelope m grid (with crossing points inserted).
    v_env : np.ndarray
        Envelope value.
    policy_envs : list of np.ndarray
        One array per policy variable, interpolated from the winning branch.
    env_inds : np.ndarray
        Branch index at each m_env point.
    """
    n_branches = len(v_arrays)
    n_policies = len(policy_arrays[0])

    # Build segments for upper_envelope: each branch is one segment
    # (filter out -inf to keep segments well-defined)
    segs = []
    seg_to_branch = []
    for b in range(n_branches):
        v = v_arrays[b]
        valid = np.isfinite(v) & (v > -1e100)
        if valid.any():
            segs.append([m_eval[valid].copy(), v[valid].copy()])
        else:
            # Dummy single-point segment that can never win
            segs.append([m_eval[:1].copy(), np.array([-1e200])])
        seg_to_branch.append(b)

    m_env, v_env, seg_inds = upper_envelope(segs)
    # Map segment indices back to branch indices
    env_inds = np.array([seg_to_branch[i] for i in seg_inds])

    # Interpolate policies from winning branches
    policy_envs = []
    for p in range(n_policies):
        pol = np.empty_like(m_env)
        for b in range(n_branches):
            mask = env_inds == b
            if mask.any():
                pol[mask] = np.interp(m_env[mask], m_eval, policy_arrays[b][p])
        policy_envs.append(pol)

    return m_env, v_env, policy_envs, env_inds


# ---------------------------------------------------------------------------
# Solution object
# ---------------------------------------------------------------------------


class HousingPortfolioSolution(MetricObject):
    """Single-period solution for the housing portfolio choice model.

    Stores policy and value functions for current owners (facing three
    tenure choices: stay, sell, or default) and for renters.

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
    # Renter bequest uses the same homogeneity exponent gamma=(1+alpha)(1-rho)
    # as the owner for calibration consistency. Net worth is liquid wealth alone.
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
# Markov employment: income distribution constructor
# ---------------------------------------------------------------------------


def construct_markov_income_process(
    T_cycle,
    PermShkStd,
    PermShkCount,
    TranShkStd,
    TranShkCount,
    T_retire,
    UnempIns,
    UnempPrbRet,
    IncUnempRet,
    RNG,
    neutral_measure=False,
    **kwargs,
):
    """Build per-employment-state income distributions for Markov employment.

    Returns a list of length T_cycle where each element is a list of two
    ``DiscreteDistribution`` objects: ``[employed_dstn, unemployed_dstn]``.

    Employed: standard lognormal permanent and transitory shocks (no iid
    unemployment, since unemployment is now governed by the Markov chain).
    Unemployed: same permanent shocks, deterministic transitory income equal
    to the unemployment insurance replacement rate ``UnempIns``.

    Parameters
    ----------
    UnempIns : float
        Fraction of permanent income received as unemployment insurance (mu).
    Other parameters: same as construct_lognormal_income_process_unemployment.

    Returns
    -------
    IncShkDstn_Mrkv : list of lists
        ``IncShkDstn_Mrkv[t]`` = ``[dstn_employed, dstn_unemployed]``.
    """
    # Employed distributions: standard lognormal, no iid unemployment event
    employed_dstns = construct_lognormal_income_process_unemployment(
        T_cycle=T_cycle,
        PermShkStd=PermShkStd,
        PermShkCount=PermShkCount,
        TranShkStd=TranShkStd,
        TranShkCount=TranShkCount,
        T_retire=T_retire,
        UnempPrb=0.0,
        IncUnemp=0.0,
        UnempPrbRet=UnempPrbRet,
        IncUnempRet=IncUnempRet,
        RNG=RNG,
        neutral_measure=neutral_measure,
    )

    # Unemployed distributions: same permanent shocks, transitory = UnempIns
    unemployed_dstns = []
    for t in range(T_cycle):
        emp_dstn = employed_dstns[t]
        perm_shks = emp_dstn.atoms[0]
        probs = emp_dstn.pmv

        # Marginalize over transitory shocks to get permanent-only weights
        unique_perms, inverse = np.unique(perm_shks, return_inverse=True)
        perm_probs = np.bincount(inverse, weights=probs)

        atoms = np.array(
            [unique_perms, np.full(len(unique_perms), UnempIns)]
        )
        unemp_dstn = DiscreteDistribution(perm_probs, atoms)
        unemployed_dstns.append(unemp_dstn)

    return [[employed_dstns[t], unemployed_dstns[t]] for t in range(T_cycle)]


# ---------------------------------------------------------------------------
# Markov employment: terminal solution constructor
# ---------------------------------------------------------------------------


def make_markov_housing_solution_terminal(
    CRRA, alpha, hbar, BeqWt, MrkvArray, **kwargs
):
    """Replicate the terminal solution across Markov employment states.

    The terminal bequest function does not depend on employment status, so each
    state gets an identical copy of the base terminal solution. The solution
    attributes become lists of length N (number of employment states).
    """
    base = make_housing_portfolio_solution_terminal(
        CRRA=CRRA, alpha=alpha, hbar=hbar, BeqWt=BeqWt
    )
    # MrkvArray may be a list of matrices (one per period) or a single matrix
    if isinstance(MrkvArray, list):
        N = np.asarray(MrkvArray[0]).shape[0]
    else:
        N = np.asarray(MrkvArray).shape[0]
    return HousingPortfolioSolution(
        cFuncOwn=[base.cFuncOwn] * N,
        ShareFuncOwn=[base.ShareFuncOwn] * N,
        vFuncOwn=[base.vFuncOwn] * N,
        vPfuncOwn=[base.vPfuncOwn] * N,
        tenureFunc=[base.tenureFunc] * N,
        cFuncRent=[base.cFuncRent] * N,
        ShareFuncRent=[base.ShareFuncRent] * N,
        vFuncRent=[base.vFuncRent] * N,
        vPfuncRent=[base.vPfuncRent] * N,
    )


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
    StockIncCorr=0.0,
):
    """Solve the renter's one-period problem.

    The renter chooses c_t, ell_t (housing services), and varsigma_t.
    With Cobb-Douglas preferences, optimal ell given c yields indirect
    utility u_renter(c) = kappa_r * c^gamma / (1-rho).

    Parameters
    ----------
    solution_next : HousingPortfolioSolution
    IncShkDstn, RiskyDstn : distributions
    LivPrb, DiscFac, CRRA, alpha, Rfree, PermGroFac : float
    aXtraGrid, ShareGrid : np.ndarray
    hbar, RentRate, ParticCost : float
    IndepDstnBool : bool
        Accepted for framework compatibility; shocks are always independent.
    StockIncCorr : float
        Loading of log equity return on log permanent shock; 0.0 = no correlation.

    Returns
    -------
    cFuncRent, ShareFuncRent, vFuncRent, vPfuncRent : callables
    """
    rho = CRRA
    gamma = (1.0 + alpha) * (1.0 - rho)
    kappa_r = (alpha / RentRate) ** (alpha * (1.0 - rho))
    DiscFacEff = DiscFac * LivPrb

    aNrmGrid = aXtraGrid
    aNrmCount = aNrmGrid.size
    ShareCount = ShareGrid.size

    # Precompute joint shock arrays
    perm_j, tran_j, prob_j, risky_j, G_j, G_gamma_j = _build_joint_shocks(
        IncShkDstn, RiskyDstn.atoms[0], RiskyDstn.pmv,
        PermGroFac, gamma, StockIncCorr,
    )

    # For each share, vectorize over (a, shocks)
    EndOfPrd_dvda = np.zeros((aNrmCount, ShareCount))
    EndOfPrd_v = np.zeros((aNrmCount, ShareCount))

    for i_s, share in enumerate(ShareGrid):
        R_port_j = share * risky_j + (1.0 - share) * Rfree  # (N_shk,)

        # m_next[i_a, j] = tran_j[j] + aNrmGrid[i_a] * R_port_j[j] / G_j[j]
        # Shape: (aNrmCount, N_shk)
        m_next = tran_j[np.newaxis, :] + (
            aNrmGrid[:, np.newaxis] * R_port_j[np.newaxis, :] / G_j[np.newaxis, :]
        )

        # Evaluate next-period functions vectorized
        m_next_flat = m_next.ravel()
        vP_next_flat = solution_next.vPfuncRent(m_next_flat)
        v_next_flat = solution_next.vFuncRent(m_next_flat)
        vP_next = vP_next_flat.reshape(m_next.shape)
        v_next = v_next_flat.reshape(m_next.shape)

        # Weighted sums: prob_j * (R_port/G * G^gamma * vP) and prob_j * G^gamma * v
        weight_dvda = (
            prob_j[np.newaxis, :]
            * R_port_j[np.newaxis, :]
            / G_j[np.newaxis, :]
            * G_gamma_j[np.newaxis, :]
            * vP_next
        )
        weight_v = prob_j[np.newaxis, :] * G_gamma_j[np.newaxis, :] * v_next

        EndOfPrd_dvda[:, i_s] = DiscFacEff * weight_dvda.sum(axis=1)
        EndOfPrd_v[:, i_s] = DiscFacEff * weight_v.sum(axis=1)

    # Handle a=0 boundary
    EndOfPrd_dvda[aNrmGrid < 1e-12, :] = 1e10
    EndOfPrd_v[aNrmGrid < 1e-12, :] = -1e10

    return _renter_egm_envelope(
        EndOfPrd_dvda, EndOfPrd_v, aNrmGrid, ShareGrid,
        alpha, rho, kappa_r, gamma, ParticCost,
    )


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
    StockIncCorr=0.0,
):
    """Solve the homeowner's one-period continuation problem (conditional on owning).

    The parameter ``IndepDstnBool`` is accepted but not enforced; the solver
    always treats income and return shocks as independent.

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

    EndOfPrd_dvda = np.zeros((aNrmCount, dCount, ShareCount))
    EndOfPrd_v = np.zeros((aNrmCount, dCount, ShareCount))

    r_m = MortRate
    R_m = 1.0 + r_m

    # Precompute joint shock arrays
    perm_j, tran_j, prob_j, risky_j, G_j, G_gamma_j = _build_joint_shocks(
        IncShkDstn, RiskyDstn.atoms[0], RiskyDstn.pmv,
        PermGroFac, gamma, StockIncCorr,
    )

    for i_d, d_now in enumerate(dGrid):
        pi_now = mortgage_payment_rate(d_now, r_m, MortPeriods)
        # d_next is the same for all (a, share) at this d
        d_next_j = (d_now * R_m - pi_now) / G_j  # (N_shk,)
        d_next_j = np.clip(d_next_j, dGrid[0], dGrid[-1])

        for i_s, share in enumerate(ShareGrid):
            R_port_j = share * risky_j + (1.0 - share) * Rfree  # (N_shk,)

            # m_next[i_a, j] = tran_j[j] + aNrmGrid[i_a] * R_port_j[j] / G_j[j]
            m_next = tran_j[np.newaxis, :] + (
                aXtraGrid[:, np.newaxis] * R_port_j[np.newaxis, :] / G_j[np.newaxis, :]
            )  # (aNrmCount, N_shk)

            # d_next_j is (N_shk,); tile to (aNrmCount * N_shk,) for flat eval
            m_flat = m_next.ravel()
            d_flat = np.tile(d_next_j, aNrmCount)

            vP_next = solution_next.vPfuncOwn(m_flat, d_flat).reshape(m_next.shape)
            v_next = solution_next.vFuncOwn(m_flat, d_flat).reshape(m_next.shape)

            # Weighted sums over shocks
            w_dvda = (
                prob_j[np.newaxis, :]
                * R_port_j[np.newaxis, :]
                / G_j[np.newaxis, :]
                * G_gamma_j[np.newaxis, :]
                * vP_next
            )
            w_v = prob_j[np.newaxis, :] * G_gamma_j[np.newaxis, :] * v_next

            EndOfPrd_dvda[:, i_d, i_s] = DiscFacEff * w_dvda.sum(axis=1)
            EndOfPrd_v[:, i_d, i_s] = DiscFacEff * w_v.sum(axis=1)

    # Handle a=0 boundary
    EndOfPrd_dvda[aXtraGrid < 1e-12, :, :] = 1e10
    EndOfPrd_v[aXtraGrid < 1e-12, :, :] = -1e10

    return _owner_egm_envelope(
        EndOfPrd_dvda, EndOfPrd_v, aXtraGrid, ShareGrid, dGrid,
        rho, h_mult, hbar, r_m, MortPeriods, MaintRate, ParticCost,
    )


def _compute_tenure_envelope(
    m_eval, d_eval, vFuncOwn_stay, vPfuncOwn_stay, cFuncOwn, ShareFuncOwn,
    vFuncRent, vPfuncRent, cFuncRent, ShareFuncRent,
    MortRate, MortPeriods, hbar, AffordabilityLimit, SellCost, DefaultPenalty,
):
    """Compute the tenure choice envelope over a (m, d) grid.

    Uses the DC-EGM upper envelope per d-slice to find exact crossing points
    between own, sell, and default value functions.

    Returns
    -------
    cFuncOwn_final, ShareFuncOwn_final, vFuncOwn_final, vPfuncOwn_final,
    tenureFunc : LinearInterpOnInterp1D
    """
    pi_d_vec = np.array(
        [mortgage_payment_rate(d, MortRate, MortPeriods) for d in d_eval]
    )
    affordable = pi_d_vec * hbar <= AffordabilityLimit

    # Pre-evaluate renter functions on m_eval (shared across d slices)
    v_rent_m = vFuncRent(m_eval)
    c_rent_m = cFuncRent(m_eval)
    s_rent_m = ShareFuncRent(m_eval)
    vp_rent_m = vPfuncRent(m_eval)

    # Pre-evaluate owner functions on (m_eval x d_eval) grid
    mm, dd = np.meshgrid(m_eval, d_eval, indexing="ij")
    flat_m, flat_d = mm.ravel(), dd.ravel()
    v_own_all = vFuncOwn_stay(flat_m, flat_d).reshape(mm.shape)
    c_own_all = cFuncOwn(flat_m, flat_d).reshape(mm.shape)
    s_own_all = ShareFuncOwn(flat_m, flat_d).reshape(mm.shape)
    vp_own_all = vPfuncOwn_stay(flat_m, flat_d).reshape(mm.shape)
    v_own_all[:, ~affordable] = -np.inf

    cFunc_own_list = []
    shareFunc_own_list = []
    vFunc_own_list = []
    vpFunc_own_list = []
    tenure_list = []

    for i_d in range(d_eval.size):
        d = d_eval[i_d]

        # --- Own branch (pre-computed) ---
        v_own_col = v_own_all[:, i_d]
        c_own_col = c_own_all[:, i_d] if affordable[i_d] else np.zeros_like(m_eval)
        s_own_col = s_own_all[:, i_d] if affordable[i_d] else np.zeros_like(m_eval)
        vp_own_col = vp_own_all[:, i_d] if affordable[i_d] else np.zeros_like(m_eval)

        # --- Sell branch ---
        net_equity = (1.0 - SellCost - d) * hbar
        m_after_sell = m_eval + net_equity
        m_sell_safe = np.maximum(m_after_sell, 1e-6)
        can_sell = m_after_sell >= 0
        v_sell_col = np.where(can_sell, vFuncRent(m_sell_safe), -np.inf)
        c_sell_col = np.where(can_sell, cFuncRent(m_sell_safe), 0.0)
        s_sell_col = np.where(can_sell, ShareFuncRent(m_sell_safe), 0.0)
        vp_sell_col = np.where(can_sell, vPfuncRent(m_sell_safe), 0.0)

        # --- Default branch ---
        v_def_col = v_rent_m - DefaultPenalty
        c_def_col = c_rent_m.copy()
        s_def_col = s_rent_m.copy()
        vp_def_col = vp_rent_m.copy()

        # Upper envelope across three tenure branches
        m_env, v_env, pol_envs, env_inds = _value_envelope(
            m_eval,
            [v_own_col, v_sell_col, v_def_col],
            [
                (c_own_col, s_own_col, vp_own_col),
                (c_sell_col, s_sell_col, vp_sell_col),
                (c_def_col, s_def_col, vp_def_col),
            ],
        )
        c_env, s_env, vp_env = pol_envs
        tenure_env = env_inds.astype(float)

        cFunc_own_list.append(LinearInterp(m_env, c_env))
        shareFunc_own_list.append(LinearInterp(m_env, s_env))
        vFunc_own_list.append(LinearInterp(m_env, v_env))
        vpFunc_own_list.append(LinearInterp(m_env, vp_env))
        tenure_list.append(LinearInterp(m_env, tenure_env))

    return (
        LinearInterpOnInterp1D(cFunc_own_list, d_eval),
        LinearInterpOnInterp1D(shareFunc_own_list, d_eval),
        LinearInterpOnInterp1D(vFunc_own_list, d_eval),
        LinearInterpOnInterp1D(vpFunc_own_list, d_eval),
        LinearInterpOnInterp1D(tenure_list, d_eval),
    )


def _compute_repurchase_option(
    aXtraGrid, vFuncRent, vPfuncRent, cFuncRent, ShareFuncRent,
    vFuncOwn_stay, vPfuncOwn_stay, cFuncOwn, ShareFuncOwn,
    DownPayment, MaxDTI, hbar, MortRate, MortPeriods, AffordabilityLimit,
):
    """Compute the renter repurchase option subject to origination constraints.

    If the renter can originate a mortgage (LTV <= 1 - DownPayment,
    DTI <= MaxDTI, payment affordable), the renter value is the upper
    envelope of pure renting vs buying into ownership.

    Returns
    -------
    cFuncRent_final, ShareFuncRent_final, vFuncRent_final, vPfuncRent_final
    """
    d_0 = min(1.0 - DownPayment, MaxDTI / hbar)
    d_0 = max(d_0, 0.0)
    down_cost = (1.0 - d_0) * hbar

    pi_orig = float(mortgage_payment_rate(d_0, MortRate, MortPeriods))
    can_originate = pi_orig * hbar <= AffordabilityLimit

    if d_0 > 0 and can_originate:
        m_rent_fine = np.linspace(1e-6, aXtraGrid[-1] * 2.0, 200)
        v_rent_vals = vFuncRent(m_rent_fine)
        c_rent_vals = cFuncRent(m_rent_fine)
        s_rent_vals = ShareFuncRent(m_rent_fine)
        vp_rent_vals = vPfuncRent(m_rent_fine)

        m_after_buy = m_rent_fine - down_cost
        can_buy = m_after_buy >= 1e-6
        m_buy_safe = np.where(can_buy, m_after_buy, 1e-6)
        d_0_arr = np.full_like(m_buy_safe, d_0)

        v_buy_vals = np.where(
            can_buy,
            vFuncOwn_stay(m_buy_safe, d_0_arr),
            -np.inf,
        )
        c_buy_vals = np.where(can_buy, cFuncOwn(m_buy_safe, d_0_arr), 0.0)
        s_buy_vals = np.where(can_buy, ShareFuncOwn(m_buy_safe, d_0_arr), 0.0)
        vp_buy_vals = np.where(can_buy, vPfuncOwn_stay(m_buy_safe, d_0_arr), 0.0)

        # Upper envelope of rent vs buy
        m_env, v_env, pol_envs, _ = _value_envelope(
            m_rent_fine,
            [v_rent_vals, v_buy_vals],
            [
                (c_rent_vals, s_rent_vals, vp_rent_vals),
                (c_buy_vals, s_buy_vals, vp_buy_vals),
            ],
        )
        c_env, s_env, vp_env = pol_envs

        return (
            LinearInterp(m_env, c_env),
            LinearInterp(m_env, s_env),
            LinearInterp(m_env, v_env),
            LinearInterp(m_env, vp_env),
        )
    return cFuncRent, ShareFuncRent, vFuncRent, vPfuncRent


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
    AffordabilityLimit=np.inf,
    DownPayment=0.20,
    MaxDTI=4.0,
    StockIncCorr=0.0,
    **kwargs,
):
    """Solve one period of the housing portfolio choice model.

    This function:
    1. Solves the renter subproblem (1D in m).
    2. Solves the owner continuation subproblem (2D in (m, d)).
    3. Takes the upper envelope across tenure choices (own/sell/default).
    4. Computes repurchase option for renters subject to origination constraints.

    Parameters
    ----------
    solution_next : HousingPortfolioSolution
    IncShkDstn, RiskyDstn : distributions
    ShockDstn : unused, accepted for HARK framework compatibility
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
    AffordabilityLimit : float
        Maximum payment-to-income ratio. When pi_tilde(d)*hbar exceeds this
        threshold, the household cannot stay as owner and must sell or default.
        Default ``np.inf`` (no constraint).
    DownPayment : float
        Minimum down payment fraction phi. Maximum LTV at origination is
        ``1 - DownPayment``. Default 0.20.
    MaxDTI : float
        Maximum debt-to-income ratio lambda at origination. The constraint
        ``d_0 * hbar <= MaxDTI`` limits leverage relative to income.
        Default 4.0.

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
        StockIncCorr,
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
        StockIncCorr,
    )

    # Step 3: Tenure envelope (own vs sell vs default)
    m_eval = np.linspace(1e-6, aXtraGrid[-1] * 2.0, 100)
    d_eval = dGrid

    cFuncOwn_final, ShareFuncOwn_final, vFuncOwn_final, vPfuncOwn_final, tenureFunc = (
        _compute_tenure_envelope(
            m_eval, d_eval, vFuncOwn_stay, vPfuncOwn_stay, cFuncOwn, ShareFuncOwn,
            vFuncRent, vPfuncRent, cFuncRent, ShareFuncRent,
            MortRate, MortPeriods, hbar, AffordabilityLimit, SellCost, DefaultPenalty,
        )
    )

    # Step 4: Repurchase envelope for renters
    cFuncRent_final, ShareFuncRent_final, vFuncRent_final, vPfuncRent_final = (
        _compute_repurchase_option(
            aXtraGrid, vFuncRent, vPfuncRent, cFuncRent, ShareFuncRent,
            vFuncOwn_stay, vPfuncOwn_stay, cFuncOwn, ShareFuncOwn,
            DownPayment, MaxDTI, hbar, MortRate, MortPeriods, AffordabilityLimit,
        )
    )

    solution_now = HousingPortfolioSolution(
        cFuncOwn=cFuncOwn_final,
        ShareFuncOwn=ShareFuncOwn_final,
        vFuncOwn=vFuncOwn_final,
        vPfuncOwn=vPfuncOwn_final,
        tenureFunc=tenureFunc,
        cFuncRent=cFuncRent_final,
        ShareFuncRent=ShareFuncRent_final,
        vFuncRent=vFuncRent_final,
        vPfuncRent=vPfuncRent_final,
    )
    solution_now.mGrid = m_eval
    solution_now.dGrid = d_eval

    return solution_now


# ---------------------------------------------------------------------------
# One-period solver: Markov employment renter subproblem
# ---------------------------------------------------------------------------


def solve_renter_subproblem_markov(
    solution_next,
    MrkvRow,
    IncShkDstn_list,
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
    StockIncCorr=0.0,
):
    """Solve the renter's one-period problem with Markov employment transitions.

    The continuation value integrates over employment transitions: for current
    state *i*, next-period state *j* occurs with probability ``MrkvRow[j]``,
    using income distribution ``IncShkDstn_list[j]`` and value function
    ``solution_next.vFuncRent[j]``.

    Parameters
    ----------
    solution_next : HousingPortfolioSolution
        Next-period solution with list-valued attributes (one per employment state).
    MrkvRow : array-like
        Transition probabilities from the current employment state.
    IncShkDstn_list : list of DiscreteDistribution
        Income distributions per next-period employment state.
    Other parameters : same as solve_renter_subproblem.

    Returns
    -------
    cFuncRent, ShareFuncRent, vFuncRent, vPfuncRent : callables
    """
    rho = CRRA
    gamma = (1.0 + alpha) * (1.0 - rho)
    kappa_r = (alpha / RentRate) ** (alpha * (1.0 - rho))
    DiscFacEff = DiscFac * LivPrb

    aNrmGrid = aXtraGrid
    aNrmCount = aNrmGrid.size
    ShareCount = ShareGrid.size
    n_states = len(IncShkDstn_list)

    # Compute end-of-period marginal value and value, integrating over
    # employment transitions, income shocks, and return shocks.
    EndOfPrd_dvda = np.zeros((aNrmCount, ShareCount))
    EndOfPrd_v = np.zeros((aNrmCount, ShareCount))

    risky_rets_base = RiskyDstn.atoms[0]
    prob_ret_arr = RiskyDstn.pmv

    for i_s, share in enumerate(ShareGrid):
        dvda_total = np.zeros(aNrmCount)
        v_total = np.zeros(aNrmCount)

        for e_next in range(n_states):
            trans_prob = MrkvRow[e_next]
            if trans_prob < 1e-15:
                continue
            dstn = IncShkDstn_list[e_next]
            vPf = solution_next.vPfuncRent[e_next]
            vFn = solution_next.vFuncRent[e_next]

            perm_j, tran_j, prob_j, risky_j, G_j, G_gamma_j = _build_joint_shocks(
                dstn, risky_rets_base, prob_ret_arr,
                PermGroFac, gamma, StockIncCorr,
            )
            prob_j = prob_j * trans_prob
            R_port_j = share * risky_j + (1.0 - share) * Rfree

            m_next = tran_j[np.newaxis, :] + (
                aNrmGrid[:, np.newaxis] * R_port_j[np.newaxis, :] / G_j[np.newaxis, :]
            )
            m_flat = m_next.ravel()
            vP_next = vPf(m_flat).reshape(m_next.shape)
            v_next = vFn(m_flat).reshape(m_next.shape)

            w_dvda = (
                prob_j[np.newaxis, :]
                * R_port_j[np.newaxis, :]
                / G_j[np.newaxis, :]
                * G_gamma_j[np.newaxis, :]
                * vP_next
            )
            w_v = prob_j[np.newaxis, :] * G_gamma_j[np.newaxis, :] * v_next

            dvda_total += w_dvda.sum(axis=1)
            v_total += w_v.sum(axis=1)

        EndOfPrd_dvda[:, i_s] = DiscFacEff * dvda_total
        EndOfPrd_v[:, i_s] = DiscFacEff * v_total

    # Handle a=0 boundary
    EndOfPrd_dvda[aNrmGrid < 1e-12, :] = 1e10
    EndOfPrd_v[aNrmGrid < 1e-12, :] = -1e10

    return _renter_egm_envelope(
        EndOfPrd_dvda, EndOfPrd_v, aNrmGrid, ShareGrid,
        alpha, rho, kappa_r, gamma, ParticCost,
    )


# ---------------------------------------------------------------------------
# One-period solver: Markov employment owner subproblem
# ---------------------------------------------------------------------------


def solve_owner_subproblem_markov(
    solution_next,
    MrkvRow,
    IncShkDstn_list,
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
    StockIncCorr=0.0,
):
    """Solve the homeowner's continuation problem with Markov employment.

    Continuation values integrate over employment transitions: the owner in
    employment state *i* transitions to state *j* with probability
    ``MrkvRow[j]``, using ``solution_next.vFuncOwn[j]`` as the owner value
    function for state *j*.

    Parameters
    ----------
    solution_next : HousingPortfolioSolution
        Next-period solution with list-valued attributes.
    MrkvRow : array-like
        Transition probabilities from the current employment state.
    IncShkDstn_list : list of DiscreteDistribution
        Income distributions per next-period employment state.
    Other parameters : same as solve_owner_subproblem.

    Returns
    -------
    cFuncOwn, ShareFuncOwn, vFuncOwn, vPfuncOwn : 2D callables over (m, d)
    """
    rho = CRRA
    gamma = (1.0 + alpha) * (1.0 - rho)
    h_mult = hbar ** (alpha * (1.0 - rho))
    DiscFacEff = DiscFac * LivPrb

    aNrmCount = aXtraGrid.size
    ShareCount = ShareGrid.size
    dCount = dGrid.size
    n_states = len(IncShkDstn_list)

    EndOfPrd_dvda = np.zeros((aNrmCount, dCount, ShareCount))
    EndOfPrd_v = np.zeros((aNrmCount, dCount, ShareCount))

    r_m = MortRate
    R_m = 1.0 + r_m

    risky_rets_base = RiskyDstn.atoms[0]
    prob_ret_arr = RiskyDstn.pmv

    for i_d, d_now in enumerate(dGrid):
        pi_now = mortgage_payment_rate(d_now, r_m, MortPeriods)

        for i_s, share in enumerate(ShareGrid):
            dvda_total = np.zeros(aNrmCount)
            v_total = np.zeros(aNrmCount)

            for e_next in range(n_states):
                trans_prob = MrkvRow[e_next]
                if trans_prob < 1e-15:
                    continue
                dstn = IncShkDstn_list[e_next]
                vPfOwn = solution_next.vPfuncOwn[e_next]
                vFOwn = solution_next.vFuncOwn[e_next]

                perm_j, tran_j, prob_j, risky_j, G_j, G_gamma_j = _build_joint_shocks(
                    dstn, risky_rets_base, prob_ret_arr,
                    PermGroFac, gamma, StockIncCorr,
                )
                prob_j = prob_j * trans_prob
                R_port_j = share * risky_j + (1.0 - share) * Rfree

                d_next_j = (d_now * R_m - pi_now) / G_j
                d_next_j = np.clip(d_next_j, dGrid[0], dGrid[-1])

                m_next = tran_j[np.newaxis, :] + (
                    aXtraGrid[:, np.newaxis] * R_port_j[np.newaxis, :] / G_j[np.newaxis, :]
                )
                m_flat = m_next.ravel()
                d_flat = np.tile(d_next_j, aNrmCount)

                vP_next = vPfOwn(m_flat, d_flat).reshape(m_next.shape)
                v_next = vFOwn(m_flat, d_flat).reshape(m_next.shape)

                w_dvda = (
                    prob_j[np.newaxis, :]
                    * R_port_j[np.newaxis, :]
                    / G_j[np.newaxis, :]
                    * G_gamma_j[np.newaxis, :]
                    * vP_next
                )
                w_v = prob_j[np.newaxis, :] * G_gamma_j[np.newaxis, :] * v_next

                dvda_total += w_dvda.sum(axis=1)
                v_total += w_v.sum(axis=1)

            EndOfPrd_dvda[:, i_d, i_s] = DiscFacEff * dvda_total
            EndOfPrd_v[:, i_d, i_s] = DiscFacEff * v_total

    # Handle a=0 boundary
    EndOfPrd_dvda[aXtraGrid < 1e-12, :, :] = 1e10
    EndOfPrd_v[aXtraGrid < 1e-12, :, :] = -1e10

    return _owner_egm_envelope(
        EndOfPrd_dvda, EndOfPrd_v, aXtraGrid, ShareGrid, dGrid,
        rho, h_mult, hbar, r_m, MortPeriods, MaintRate, ParticCost,
    )


# ---------------------------------------------------------------------------
# One-period solver: Markov employment housing portfolio choice
# ---------------------------------------------------------------------------


def solve_one_period_HousingPortfolioMarkov(
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
    SellCost,
    DefaultPenalty,
    RentRate,
    BoroCnstArt,
    IndepDstnBool,
    ShareLimit,
    MrkvArray,
    AffordabilityLimit=np.inf,
    DownPayment=0.20,
    MaxDTI=4.0,
    StockIncCorr=0.0,
    **kwargs,
):
    """Solve one period of the housing portfolio model with Markov employment.

    Iterates over each current employment state *i*, solving the renter and
    owner subproblems with continuation values that integrate over employment
    transitions governed by ``MrkvArray[i]``. Returns a solution with
    list-valued attributes (one entry per employment state).

    Parameters
    ----------
    solution_next : HousingPortfolioSolution
        Next-period solution with list-valued attributes.
    IncShkDstn : list of DiscreteDistribution
        Income distributions per employment state for this period.
    MrkvArray : np.ndarray
        (N, N) employment transition matrix. ``MrkvArray[i, j]`` =
        Prob(next state = j | current state = i).
    DownPayment : float
        Minimum down payment fraction phi at origination. Default 0.20.
    MaxDTI : float
        Maximum debt-to-income ratio lambda at origination. Default 4.0.
    Other parameters : same as solve_one_period_HousingPortfolio.

    Returns
    -------
    solution_now : HousingPortfolioSolution
        Solution with list-valued attributes (length N).
    """
    N = MrkvArray.shape[0]
    rho = CRRA
    gamma = (1.0 + alpha) * (1.0 - rho)

    state_solutions = []
    for i_state in range(N):
        MrkvRow = MrkvArray[i_state]

        # Step 1: Solve the renter subproblem for this employment state
        cFuncRent, ShareFuncRent, vFuncRent, vPfuncRent = (
            solve_renter_subproblem_markov(
                solution_next,
                MrkvRow,
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
                StockIncCorr,
            )
        )

        # Step 2: Solve the owner subproblem for this employment state
        cFuncOwn, ShareFuncOwn, vFuncOwn_stay, vPfuncOwn_stay = (
            solve_owner_subproblem_markov(
                solution_next,
                MrkvRow,
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
                StockIncCorr,
            )
        )

        # Step 3: Tenure envelope (own vs sell vs default)
        m_eval = np.linspace(1e-6, aXtraGrid[-1] * 2.0, 100)
        d_eval = dGrid

        cFuncOwn_2d, ShareFuncOwn_2d, vFuncOwn_2d, vPfuncOwn_2d, tenureFunc_2d = (
            _compute_tenure_envelope(
                m_eval, d_eval, vFuncOwn_stay, vPfuncOwn_stay, cFuncOwn, ShareFuncOwn,
                vFuncRent, vPfuncRent, cFuncRent, ShareFuncRent,
                MortRate, MortPeriods, hbar, AffordabilityLimit, SellCost, DefaultPenalty,
            )
        )

        # Step 4: Repurchase envelope for renters
        cFuncRent_final, ShareFuncRent_final, vFuncRent_final, vPfuncRent_final = (
            _compute_repurchase_option(
                aXtraGrid, vFuncRent, vPfuncRent, cFuncRent, ShareFuncRent,
                vFuncOwn_stay, vPfuncOwn_stay, cFuncOwn, ShareFuncOwn,
                DownPayment, MaxDTI, hbar, MortRate, MortPeriods, AffordabilityLimit,
            )
        )

        sol_i = HousingPortfolioSolution(
            cFuncOwn=cFuncOwn_2d,
            ShareFuncOwn=ShareFuncOwn_2d,
            vFuncOwn=vFuncOwn_2d,
            vPfuncOwn=vPfuncOwn_2d,
            tenureFunc=tenureFunc_2d,
            cFuncRent=cFuncRent_final,
            ShareFuncRent=ShareFuncRent_final,
            vFuncRent=vFuncRent_final,
            vPfuncRent=vPfuncRent_final,
        )
        sol_i.mGrid = m_eval
        sol_i.dGrid = d_eval
        state_solutions.append(sol_i)

    # Combine into a single solution with list-valued attributes
    combined = HousingPortfolioSolution(
        cFuncOwn=[s.cFuncOwn for s in state_solutions],
        ShareFuncOwn=[s.ShareFuncOwn for s in state_solutions],
        vFuncOwn=[s.vFuncOwn for s in state_solutions],
        vPfuncOwn=[s.vPfuncOwn for s in state_solutions],
        tenureFunc=[s.tenureFunc for s in state_solutions],
        cFuncRent=[s.cFuncRent for s in state_solutions],
        ShareFuncRent=[s.ShareFuncRent for s in state_solutions],
        vFuncRent=[s.vFuncRent for s in state_solutions],
        vPfuncRent=[s.vPfuncRent for s in state_solutions],
    )
    return combined


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
    "AffordabilityLimit": np.inf,  # Max payment-to-income ratio (no constraint)
    # Origination constraints
    "DownPayment": 0.20,  # Min down payment fraction phi (max LTV = 0.80)
    "MaxDTI": 4.0,  # Max debt-to-income ratio lambda
    # Stock-income correlation
    "StockIncCorr": 0.0,  # Loading of log equity return on log permanent shock
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
# Markov employment defaults
# ---------------------------------------------------------------------------

# Two-state Markov chain: E(mployed)=0, U(nemployed)=1
# p_u = prob of losing job, p_e = prob of regaining employment
_p_u = 0.05  # 5% chance of job loss per period
_p_e = 0.50  # 50% chance of re-employment per period (avg spell: 2 periods)
_mrkv_default = np.array([[1.0 - _p_u, _p_u], [_p_e, 1.0 - _p_e]])

MarkovHousingPortfolioType_constructors_default = {
    "IncShkDstn": construct_markov_income_process,
    "aXtraGrid": make_assets_grid,
    "RiskyDstn": make_lognormal_RiskyDstn,
    "ShareLimit": calc_ShareLimit_for_CRRA,
    "ShareGrid": make_simple_ShareGrid,
    "dGrid": make_ltv_grid,
    "solution_terminal": make_markov_housing_solution_terminal,
    # ShockDstn, PermShkDstn, TranShkDstn are omitted: the Markov solver
    # uses IncShkDstn (per-state) and RiskyDstn independently.
}

MarkovHousingPortfolioType_IncShkDstn_default = {
    "PermShkStd": [0.1] * _life_span,
    "PermShkCount": 5,
    "TranShkStd": [0.1] * _life_span,
    "TranShkCount": 5,
    "UnempIns": 0.3,  # Unemployment insurance: 30% of permanent income
    "T_retire": _retire_age,
    "UnempPrbRet": 0.005,
    "IncUnempRet": 0.0,
}

MarkovHousingPortfolioType_solving_default = {
    "cycles": 1,
    "T_cycle": _life_span,
    "constructors": MarkovHousingPortfolioType_constructors_default,
    "CRRA": 5.0,
    "alpha": 0.2,
    "DiscFac": 0.96,
    "LivPrb": _surv_probs,
    "PermGroFac": _perm_gro_fac,
    "Rfree": [1.02] * _life_span,
    "BoroCnstArt": 0.0,
    "hbar": 3.0,
    "MortRate": 0.04,
    "MortPeriods": [max(30 - t, 0) for t in range(_life_span)],
    "MaintRate": 0.02,
    "SellCost": 0.06,
    "DefaultPenalty": 0.5,
    "RentRate": 0.05,
    "ParticCost": 0.01,
    "BeqWt": 1.0,
    "AffordabilityLimit": np.inf,
    "DownPayment": 0.20,
    "MaxDTI": 4.0,
    "StockIncCorr": 0.0,
    "MrkvArray": [_mrkv_default] * _life_span,
    "IndepDstnBool": True,
    "DiscreteShareBool": True,
    "vFuncBool": True,
    "CubicBool": False,
}

MarkovHousingPortfolioType_default = {}
MarkovHousingPortfolioType_default.update(
    MarkovHousingPortfolioType_solving_default
)
MarkovHousingPortfolioType_default.update(
    HousingPortfolioType_simulation_default
)
MarkovHousingPortfolioType_default.update(
    HousingPortfolioType_kNrmInitDstn_default
)
MarkovHousingPortfolioType_default.update(
    HousingPortfolioType_pLvlInitDstn_default
)
MarkovHousingPortfolioType_default.update(HousingPortfolioType_aXtraGrid_default)
MarkovHousingPortfolioType_default.update(
    HousingPortfolioType_ShareGrid_default
)
MarkovHousingPortfolioType_default.update(HousingPortfolioType_dGrid_default)
MarkovHousingPortfolioType_default.update(
    MarkovHousingPortfolioType_IncShkDstn_default
)
MarkovHousingPortfolioType_default.update(HousingPortfolioType_RiskyDstn_default)

init_markov_housing_portfolio = MarkovHousingPortfolioType_default


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
        "AffordabilityLimit",
        "DownPayment",
        "MaxDTI",
        "StockIncCorr",
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
        self.check_restrictions()
        self.construct("IncShkDstn")
        self.construct("PermShkDstn")
        self.construct("TranShkDstn")
        self.construct("aXtraGrid")
        self.construct("RiskyDstn")
        self.construct("ShockDstn")
        self.construct("ShareLimit")
        self.construct("ShareGrid")
        self.construct("dGrid")
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
        if not (0.0 <= self.SellCost <= 1.0):
            raise ValueError(
                f"SellCost must be in [0, 1], got {self.SellCost}."
            )


class MarkovHousingPortfolioConsumerType(HousingPortfolioConsumerType):
    """Housing portfolio model with persistent Markov employment states.

    Extends the baseline model with a two-state employment chain
    e_t in {Employed, Unemployed}. Employed households receive standard
    lognormal income shocks; unemployed households receive a fraction mu
    of permanent income as unemployment insurance. Persistent unemployment
    spells create affordability stress that triggers selling or default.

    The value function is solved separately for each employment state, with
    continuation values integrating over employment transitions governed by
    the Markov transition matrix ``MrkvArray``.

    Additional parameters
    ---------------------
    MrkvArray : list of np.ndarray
        Employment transition matrix per period. Each element is (N, N)
        with ``MrkvArray[t][i, j]`` = Prob(next state = j | current = i).
        Default: 2-state chain with p_u = 0.05, p_e = 0.50.
    UnempIns : float
        Fraction of permanent income received as unemployment insurance (mu).
        Default: 0.3.
    """

    IncShkDstn_default = MarkovHousingPortfolioType_IncShkDstn_default
    solving_default = MarkovHousingPortfolioType_solving_default

    default_ = {
        "params": MarkovHousingPortfolioType_default,
        "solver": solve_one_period_HousingPortfolioMarkov,
        "track_vars": ["mNrm", "cNrm", "Share", "dNrm", "tenure", "Mrkv"],
    }

    # Override time_inv_: same as parent but without ShockDstn
    # (the Markov solver uses IncShkDstn per-state and RiskyDstn independently).
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
        "AffordabilityLimit",
        "DownPayment",
        "MaxDTI",
        "StockIncCorr",
    ]

    # Override time_vary_: drop PermShkDstn/TranShkDstn (not built for Markov),
    # add MrkvArray, keep ShockDstn out.
    time_vary_ = [
        "LivPrb",
        "PermGroFac",
        "Rfree",
        "IncShkDstn",
        "MortPeriods",
        "MrkvArray",
    ]

    def __init__(self, verbose=False, quiet=True, **kwds):
        params = deepcopy(MarkovHousingPortfolioType_default)
        params.update(kwds)
        IndShockConsumerType.__init__(
            self, verbose=verbose, quiet=quiet, **params
        )

    def pre_solve(self):
        """Construct objects needed before solving."""
        self.check_restrictions()
        self.construct("IncShkDstn")
        self.construct("aXtraGrid")
        self.construct("RiskyDstn")
        self.construct("ShareLimit")
        self.construct("ShareGrid")
        self.construct("dGrid")
        self.construct("solution_terminal")

    def check_restrictions(self):
        """Check model parameter restrictions including Markov-specific ones."""
        super().check_restrictions()
        MrkvArray = self.MrkvArray
        if isinstance(MrkvArray, list):
            MrkvArray = MrkvArray[0]
        if MrkvArray.shape[0] != MrkvArray.shape[1]:
            raise ValueError("MrkvArray must be square.")
        row_sums = MrkvArray.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError(
                f"MrkvArray rows must sum to 1, got {row_sums}."
            )

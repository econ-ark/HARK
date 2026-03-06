"""
Life-cycle model of housing and portfolio choice with collateralized borrowing.

Cocco-style (m, h) model from the FINRA grant proposal:
- Continuous states: m_t = M_t/P_t (cash-on-hand), h_t = H_t/P_t (housing ratio)
- Discrete state: P_t in {0, 1} — stock market participant (Gomes-Michaelides 2005)
- Continuous choices: c_t (consumption), varsigma_t (risky share)
- Discrete choices: tenure (own / sell / move / default) and market participation
- Discrete Markov states: employment e in {E, U}, regime nu in {B, S}
- Independent income and housing growth: G^P, G^H(nu)
- Transitory house price shock: eps^H_t ~ MeanOneLogNormal, iid each period
- Housing growth shock: eta_t ~ MeanOneLogNormal, iid each period

Participation follows Gomes-Michaelides (2005) entry cost:
- A non-participant pays fixed cost F to enter the stock market.
- A participant stays for free or exits freely.
- V^P(m,h)   = max{V^in(m,h),   V^out(m,h)}
- V^{NP}(m,h) = max{V^in(m-F,h), V^out(m,h)}
- V^in uses V^P_{t+1} as continuation; V^out uses V^{NP}_{t+1}.

Value function:
    V(M,H,P; e,nu) = P^gamma * v(m,h; e,nu)

Bellman (owner, conditional on staying):
    v^o_t(m,h) = max_{c,varsigma} (c h^alpha)^{1-rho}/(1-rho)
                 + beta s_t sum_{e',nu'} pi * E[G^gamma v_{t+1}(m',h')]

where gamma = (1+alpha)(1-rho), beta = DiscFac, s_t = LivPrb, rho = CRRA.

Tenure envelope (integrates over eps^H):
    v_t(m,h) = E_{eps^H}[max(v^own, v^sell, v^move, v^default)]

Terminal:
    v_T(m,h) = E_{eps^H}[max{omega*(m+h[(1-kappa)eps^H - lbar])^gamma/gamma,
                               omega*m^gamma/gamma - zeta}]
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
    LowerEnvelope,
)
from HARK.metric import MetricObject
from HARK.distributions import MeanOneLogNormal
from HARK.distributions.discrete import (
    DiscreteDistribution,
    DiscreteDistributionLabeled,
)
from HARK.utilities import make_assets_grid

__all__ = [
    "HousingPortfolioSolution",
    "HousingPortfolioConsumerType",
    "MarkovHousingPortfolioConsumerType",
]

# Number of points for tenure- and repurchase-envelope evaluation grids.
_N_ENVELOPE_POINTS = 100


# ---------------------------------------------------------------------------
# Housing grid and shock constructors
# ---------------------------------------------------------------------------


def make_h_grid(hGridCount=10, hMin=0.5, hMax=8.0, **kwargs):
    """Create a log-spaced grid of housing-to-income ratios.

    Parameters
    ----------
    hGridCount : int
        Number of grid points.
    hMin : float
        Minimum h (must be > 0).
    hMax : float
        Maximum h.

    Returns
    -------
    hGrid : np.ndarray
    """
    return np.exp(np.linspace(np.log(hMin), np.log(hMax), hGridCount))


def _get_eps_H(HousePriceShkDstn):
    """Extract eps^H nodes and probabilities from the house price shock distribution.

    Accepts a ``DiscreteDistribution``, ``DiscreteDistributionLabeled``,
    or ``None``.  Returns (nodes, probs) arrays.  When the distribution is
    None, returns ([1.0], [1.0]) so the tenure envelope reduces to the
    no-shock case.
    """
    if HousePriceShkDstn is None:
        return np.array([1.0]), np.array([1.0])
    return np.asarray(HousePriceShkDstn.atoms[0]), np.asarray(HousePriceShkDstn.pmv)


def make_housing_growth_shock_dstn(
    HousingGrowthShkStd=0.0, HousingGrowthShkCount=1, **kwargs
):
    """Discretize the iid housing growth shock eta.

    Uses HARK's ``MeanOneLogNormal`` distribution.  When the standard
    deviation is zero, returns a degenerate distribution at 1 (no shock).

    Parameters
    ----------
    HousingGrowthShkStd : float
        Standard deviation of log(eta).  Default 0.0 (no shock).
    HousingGrowthShkCount : int
        Number of quadrature nodes.  Default 1 (degenerate).

    Returns
    -------
    HousingGrowthShkDstn : DiscreteDistributionLabeled
    """
    discrete = MeanOneLogNormal(HousingGrowthShkStd).discretize(
        max(HousingGrowthShkCount, 1), method="equiprobable"
    )
    return DiscreteDistributionLabeled.from_unlabeled(
        discrete, var_names=["HousingGrowthShk"]
    )


def make_house_price_shock_dstn(
    HousePriceShkStd=0.0, HousePriceShkCount=1, **kwargs
):
    """Discretize the iid transitory house price shock eps^H.

    Uses HARK's ``MeanOneLogNormal`` distribution.  When the standard
    deviation is zero, returns a degenerate distribution at 1 (no shock).

    Parameters
    ----------
    HousePriceShkStd : float
        Standard deviation of log(eps^H).  Default 0.0 (no shock).
    HousePriceShkCount : int
        Number of quadrature nodes.  Default 1 (degenerate).

    Returns
    -------
    HousePriceShkDstn : DiscreteDistributionLabeled
    """
    discrete = MeanOneLogNormal(HousePriceShkStd).discretize(
        max(HousePriceShkCount, 1), method="equiprobable"
    )
    return DiscreteDistributionLabeled.from_unlabeled(
        discrete, var_names=["HousePriceShk"]
    )


def _build_joint_shocks(inc_dstn, risky_rets_base, prob_ret_arr,
                        PermGroFac, gamma, StockIncCorr,
                        housing_dstn=None, HousingIncCorr=0.0,
                        HousingStockCorr=0.0):
    """Build joint shock arrays for vectorized integration.

    When ``housing_dstn`` is provided (owner path), the outer product is
    (perm x eta x ret) = N_inc x N_eta x N_ret nodes.
    When ``housing_dstn`` is None (renter path), eta_j = 1 and nodes are
    N_inc x N_ret.

    Correlations are applied via loading factors:
      risky_j = risky_base * perm_j**StockIncCorr * eta_j**HousingStockCorr
      eta_j = eta_base * perm_j**HousingIncCorr

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
    housing_dstn : DiscreteDistribution or None
        Housing growth shock distribution. None for renter.
    HousingIncCorr : float
        Loading of log eta on log permanent shock.
    HousingStockCorr : float
        Loading of log equity return on log eta.

    Returns
    -------
    perm_j, tran_j, prob_j, risky_j, G_j, G_gamma_j, eta_j : np.ndarray
        Joint shock arrays.
    """
    perm_nodes = inc_dstn.atoms[0]
    tran_nodes = inc_dstn.atoms[1]
    prob_inc = inc_dstn.pmv
    n_inc = prob_inc.size
    n_ret = risky_rets_base.size

    if housing_dstn is not None:
        eta_nodes = np.asarray(housing_dstn.atoms[0])
        prob_eta = np.asarray(housing_dstn.pmv)
        n_eta = eta_nodes.size

        # Outer product: inc x eta x ret
        # Order: for each inc node, for each eta node, for each ret node
        perm_j = np.repeat(np.repeat(perm_nodes, n_eta), n_ret)
        tran_j = np.repeat(np.repeat(tran_nodes, n_eta), n_ret)
        prob_j = (np.repeat(np.repeat(prob_inc, n_eta), n_ret)
                  * np.tile(np.repeat(prob_eta, n_ret), n_inc)
                  * np.tile(prob_ret_arr, n_inc * n_eta))
        eta_j = np.tile(np.repeat(eta_nodes, n_ret), n_inc)
        risky_j = np.tile(risky_rets_base, n_inc * n_eta)
    else:
        # Renter: no housing shocks
        perm_j = np.repeat(perm_nodes, n_ret)
        tran_j = np.repeat(tran_nodes, n_ret)
        prob_j = np.repeat(prob_inc, n_ret) * np.tile(prob_ret_arr, n_inc)
        risky_j = np.tile(risky_rets_base, n_inc)
        eta_j = np.ones_like(perm_j)

    # Apply correlations via loading factors
    if HousingIncCorr != 0.0:
        eta_j = eta_j * perm_j ** HousingIncCorr
    if StockIncCorr != 0.0:
        risky_j = risky_j * perm_j ** StockIncCorr
    if HousingStockCorr != 0.0:
        risky_j = risky_j * eta_j ** HousingStockCorr

    G_j = PermGroFac * perm_j
    if np.any(G_j <= 0):
        raise ValueError(
            f"Non-positive growth factor G_j (min={G_j.min():.6g}). "
            "Check that PermGroFac > 0 and permanent shocks exclude zero."
        )
    G_gamma_j = G_j ** gamma
    return perm_j, tran_j, prob_j, risky_j, G_j, G_gamma_j, eta_j


def _renter_egm_envelope(
    EndOfPrd_dvda, EndOfPrd_v, aNrmGrid, ShareGrid,
    alpha, rho, kappa_r, gamma, EntryCost,
):
    """EGM inversion + DC-EGM participation envelope for the renter.

    Implements Gomes-Michaelides (2005) entry cost: the "in" (participate)
    and "out" (non-participate) branches use different continuation values
    baked into EndOfPrd arrays.  The envelope is computed twice:

    - V^P: participant can stay in or exit freely.
    - V^{NP}: non-participant must pay EntryCost to enter (m shifted by F
      on the "in" branch).

    Parameters
    ----------
    EndOfPrd_dvda, EndOfPrd_v : np.ndarray, shape (aNrmCount, ShareCount)
        End-of-period marginal value and value after integration.
        Column 0 uses V^{NP}_{t+1} continuation; columns 1+ use V^P_{t+1}.
    aNrmGrid, ShareGrid : np.ndarray
    alpha : float
        Housing preference weight (budget: m = (1+alpha)*c + a).
    rho, kappa_r, gamma : float
    EntryCost : float
        One-time entry cost F (Gomes-Michaelides 2005).

    Returns
    -------
    cFunc, ShareFunc, vFunc, vPfunc : LinearInterp
        Participant (V^P) policy and value functions.
    vFunc_NP, vPfunc_NP : LinearInterp
        Non-participant (V^{NP}) value and marginal value functions.
    """
    aNrmCount = aNrmGrid.size
    ShareCount = ShareGrid.size

    # FOC: kappa_r * c^{gamma-1} = EndOfPrd_dvda
    cNrmGrid = (EndOfPrd_dvda / kappa_r) ** (1.0 / (gamma - 1.0))
    cNrmGrid = np.maximum(cNrmGrid, 1e-12)
    mNrmGrid = (1.0 + alpha) * cNrmGrid + aNrmGrid[:, np.newaxis]

    # Best share among positive shares ("in" branch: participate)
    arange_a = np.arange(aNrmCount)
    if ShareCount > 1:
        opt_pos_idx = np.argmax(EndOfPrd_v[:, 1:], axis=1) + 1
    else:
        opt_pos_idx = np.zeros(aNrmCount, dtype=int)
    c_in = cNrmGrid[arange_a, opt_pos_idx]
    m_in = mNrmGrid[arange_a, opt_pos_idx]
    s_in = ShareGrid[opt_pos_idx]
    v_in = kappa_r * c_in ** gamma / (1.0 - rho) + EndOfPrd_v[arange_a, opt_pos_idx]

    # "out" branch: share = 0 (non-participation)
    c_out = cNrmGrid[:, 0]
    m_out = mNrmGrid[:, 0]
    v_out = kappa_r * c_out ** gamma / (1.0 - rho) + EndOfPrd_v[:, 0]

    # --- V^P envelope: participant can stay in or exit freely ---
    m_env_P, c_env_P, s_env_P, v_env_P = _participation_envelope(
        m_in, c_in, v_in, s_in,
        m_out, c_out, v_out,
    )
    vp_env_P = np.concatenate([[1e10], kappa_r * c_env_P[1:] ** (gamma - 1.0)])

    # Constrained consumption function: at a=0, budget gives c = m/(1+alpha).
    cFuncUnc = LinearInterp(m_env_P, c_env_P)
    cFuncCnst = LinearInterp(
        np.array([0.0, 1.0]), np.array([0.0, 1.0 / (1.0 + alpha)])
    )
    cFunc = LowerEnvelope(cFuncUnc, cFuncCnst)

    # --- V^{NP} envelope: non-participant pays F to enter ---
    # Shift the "in" branch m-grid rightward by EntryCost
    m_env_NP, c_env_NP, s_env_NP, v_env_NP = _participation_envelope(
        m_in + EntryCost, c_in, v_in, s_in,
        m_out, c_out, v_out,
    )
    vp_env_NP = np.concatenate([[1e10], kappa_r * c_env_NP[1:] ** (gamma - 1.0)])

    return (
        cFunc,
        LinearInterp(m_env_P, s_env_P),
        LinearInterp(m_env_P, v_env_P),
        LinearInterp(m_env_P, vp_env_P),
        LinearInterp(m_env_NP, v_env_NP),
        LinearInterp(m_env_NP, vp_env_NP),
    )


def _owner_egm_envelope(
    EndOfPrd_dvda, EndOfPrd_v, aXtraGrid, ShareGrid, hGrid,
    alpha, rho, LTV, MortRate, MaintRate, EntryCost,
):
    """EGM inversion + DC-EGM participation envelope for the owner.

    Implements Gomes-Michaelides (2005) entry cost.  The "in" (ς > 0) and
    "out" (ς = 0) branches use different continuations baked into EndOfPrd.
    Produces participant (V^P) and non-participant (V^{NP}) value functions.

    Parameters
    ----------
    EndOfPrd_dvda, EndOfPrd_v : np.ndarray, shape (aNrmCount, hCount, ShareCount)
    aXtraGrid, ShareGrid, hGrid : np.ndarray
    alpha, rho : float
    LTV : float
        Collateral fraction lbar.
    MortRate : float
        Mortgage interest rate r_m.
    MaintRate : float
    EntryCost : float
        One-time entry cost F (Gomes-Michaelides 2005).

    Returns
    -------
    cFuncOwn, ShareFuncOwn, vFuncOwn, vPfuncOwn : LinearInterpOnInterp1D
        Participant (V^P) functions.
    vFuncOwn_NP, vPfuncOwn_NP : LinearInterpOnInterp1D
        Non-participant (V^{NP}) value and marginal value.
    """
    aNrmCount = aXtraGrid.size
    ShareCount = ShareGrid.size
    hCount = hGrid.size

    # Vectorized EGM inversion across all h-slices and shares
    h_mult_all = hGrid[np.newaxis, :, np.newaxis] ** (alpha * (1.0 - rho))
    mandatory_cost_all = (MortRate * LTV + MaintRate) * hGrid[np.newaxis, :, np.newaxis]

    # FOC: c = (dvda / h_mult)^{-1/rho}  for all (a, h, s)
    cNrmGrid = np.maximum((EndOfPrd_dvda / h_mult_all) ** (-1.0 / rho), 1e-12)

    # Budget: m = c + a + mandatory_cost (no per-period participation cost)
    mNrmGrid = (
        cNrmGrid
        + aXtraGrid[:, np.newaxis, np.newaxis]
        + mandatory_cost_all
    )

    # DC-EGM participation envelope for each h slice
    arange_a = np.arange(aNrmCount)
    if ShareCount > 1:
        opt_pos_idx = np.argmax(EndOfPrd_v[:, :, 1:], axis=2) + 1
    else:
        opt_pos_idx = np.zeros((aNrmCount, hCount), dtype=int)

    cFunc_list = []
    shareFunc_list = []
    vFunc_P_list = []
    vpFunc_P_list = []
    vFunc_NP_list = []
    vpFunc_NP_list = []

    for i_h in range(hCount):
        h_mult = hGrid[i_h] ** (alpha * (1.0 - rho))
        idx_in = opt_pos_idx[:, i_h]
        c_in = cNrmGrid[arange_a, i_h, idx_in]
        m_in = mNrmGrid[arange_a, i_h, idx_in]
        s_in = ShareGrid[idx_in]
        v_in = h_mult * c_in ** (1.0 - rho) / (1.0 - rho) + EndOfPrd_v[arange_a, i_h, idx_in]

        c_out = cNrmGrid[:, i_h, 0]
        m_out = mNrmGrid[:, i_h, 0]
        v_out = h_mult * c_out ** (1.0 - rho) / (1.0 - rho) + EndOfPrd_v[:, i_h, 0]

        # V^P envelope: participant can stay in or exit freely
        m_env_P, c_env_P, s_env_P, v_env_P = _participation_envelope(
            m_in, c_in, v_in, s_in, m_out, c_out, v_out,
        )
        vp_env_P = np.concatenate([[1e10], h_mult * c_env_P[1:] ** (-rho)])

        cFunc_list.append(LinearInterp(m_env_P, c_env_P))
        shareFunc_list.append(LinearInterp(m_env_P, s_env_P))
        vFunc_P_list.append(LinearInterp(m_env_P, v_env_P))
        vpFunc_P_list.append(LinearInterp(m_env_P, vp_env_P))

        # V^{NP} envelope: non-participant pays F to enter
        m_env_NP, c_env_NP, _, v_env_NP = _participation_envelope(
            m_in + EntryCost, c_in, v_in, s_in, m_out, c_out, v_out,
        )
        vp_env_NP = np.concatenate([[1e10], h_mult * c_env_NP[1:] ** (-rho)])

        vFunc_NP_list.append(LinearInterp(m_env_NP, v_env_NP))
        vpFunc_NP_list.append(LinearInterp(m_env_NP, vp_env_NP))

    return (
        LinearInterpOnInterp1D(cFunc_list, hGrid),
        LinearInterpOnInterp1D(shareFunc_list, hGrid),
        LinearInterpOnInterp1D(vFunc_P_list, hGrid),
        LinearInterpOnInterp1D(vpFunc_P_list, hGrid),
        LinearInterpOnInterp1D(vFunc_NP_list, hGrid),
        LinearInterpOnInterp1D(vpFunc_NP_list, hGrid),
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
    c_env = np.full_like(m_env, np.nan)
    s_env = np.full_like(m_env, np.nan)
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

    Used for tenure choice (own/sell/move/default) and repurchase (rent/buy)
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
    segs = []
    for b in range(n_branches):
        v = v_arrays[b]
        valid = np.isfinite(v) & (v > -1e100)
        if valid.any():
            segs.append([m_eval[valid].copy(), v[valid].copy()])
        else:
            segs.append([m_eval[:1].copy(), np.array([-1e200])])

    m_env, v_env, seg_inds = upper_envelope(segs)
    env_inds = np.asarray(seg_inds)

    # Interpolate policies from winning branches
    policy_envs = []
    for p in range(n_policies):
        pol = np.full_like(m_env, np.nan)
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

    Stores policy and value functions for current owners (facing four
    tenure choices: stay, sell, move, or default) and for renters.
    Each set of functions exists in two versions: participant (P) and
    non-participant (NP), following the Gomes-Michaelides (2005) entry
    cost formulation.  The "plain" attributes (vFuncOwn, vFuncRent, etc.)
    are the participant versions (V^P).  The ``_NP`` suffixed attributes
    are the non-participant versions (V^{NP}).

    Attributes
    ----------
    cFuncOwn : callable(m, h) -> c
        Consumption function for homeowners (participant).
    ShareFuncOwn : callable(m, h) -> varsigma
        Risky share function for homeowners (participant).
    vFuncOwn : callable(m, h) -> v
        Value function for homeowners (participant, V^P).
    vPfuncOwn : callable(m, h) -> dv/dm
        Marginal value of cash-on-hand for homeowners (participant).
    vFuncOwn_NP : callable(m, h) -> v
        Value function for homeowners (non-participant, V^{NP}).
    vPfuncOwn_NP : callable(m, h) -> dv/dm
        Marginal value for homeowners (non-participant).
    tenureFunc : callable(m, h) -> float
        Tenure choice index (use round() to recover discrete choice):
        0=own, 1=sell, 2=move, 3=default.
    cFuncRent : callable(m) -> c
        Consumption function for renters (participant).
    ShareFuncRent : callable(m) -> varsigma
        Risky share function for renters (participant).
    vFuncRent : callable(m) -> v
        Value function for renters (participant, V^P).
    vPfuncRent : callable(m) -> dv/dm
        Marginal value of cash-on-hand for renters (participant).
    vFuncRent_NP : callable(m) -> v
        Value function for renters (non-participant, V^{NP}).
    vPfuncRent_NP : callable(m) -> dv/dm
        Marginal value for renters (non-participant).
    """

    distance_criteria = ["vPfuncOwn"]

    def __init__(
        self,
        cFuncOwn=None,
        ShareFuncOwn=None,
        vFuncOwn=None,
        vPfuncOwn=None,
        vFuncOwn_NP=None,
        vPfuncOwn_NP=None,
        tenureFunc=None,
        cFuncRent=None,
        ShareFuncRent=None,
        vFuncRent=None,
        vPfuncRent=None,
        vFuncRent_NP=None,
        vPfuncRent_NP=None,
        mNrmMin=0.0,
    ):
        self.cFuncOwn = cFuncOwn if cFuncOwn is not None else NullFunc()
        self.ShareFuncOwn = ShareFuncOwn if ShareFuncOwn is not None else NullFunc()
        self.vFuncOwn = vFuncOwn if vFuncOwn is not None else NullFunc()
        self.vPfuncOwn = vPfuncOwn if vPfuncOwn is not None else NullFunc()
        self.vFuncOwn_NP = vFuncOwn_NP if vFuncOwn_NP is not None else NullFunc()
        self.vPfuncOwn_NP = vPfuncOwn_NP if vPfuncOwn_NP is not None else NullFunc()
        self.tenureFunc = tenureFunc if tenureFunc is not None else NullFunc()
        self.cFuncRent = cFuncRent if cFuncRent is not None else NullFunc()
        self.ShareFuncRent = (
            ShareFuncRent if ShareFuncRent is not None else NullFunc()
        )
        self.vFuncRent = vFuncRent if vFuncRent is not None else NullFunc()
        self.vPfuncRent = vPfuncRent if vPfuncRent is not None else NullFunc()
        self.vFuncRent_NP = vFuncRent_NP if vFuncRent_NP is not None else NullFunc()
        self.vPfuncRent_NP = vPfuncRent_NP if vPfuncRent_NP is not None else NullFunc()
        self.mNrmMin = mNrmMin
        # Store grids for diagnostics
        self.mGrid = None
        self.hGrid = None


# ---------------------------------------------------------------------------
# Terminal solution constructor
# ---------------------------------------------------------------------------


def make_housing_portfolio_solution_terminal(
    CRRA, alpha, LTV, BeqWt, SellCost, DefaultPenalty,
    HousePriceShkDstn=None, **kwargs
):
    """Construct the terminal-period solution for the housing portfolio model.

    At terminal age T the household liquidates housing and bequeaths total
    net worth.  The owner's terminal value integrates over eps^H:

        v_T(m, h) = E_{eps^H}[max{omega*(m + h[(1-kappa)eps^H - lbar])^gamma/gamma,
                                    omega*m^gamma/gamma - zeta}]

    Parameters
    ----------
    CRRA : float
        Coefficient of relative risk aversion (rho).
    alpha : float
        Housing preference parameter.
    LTV : float
        Collateral fraction (lbar).
    BeqWt : float
        Bequest weight (omega).
    SellCost : float
        Transaction cost fraction (kappa).
    DefaultPenalty : float
        Utility penalty for default (zeta).
    HousePriceShkDstn : DiscreteDistribution or None
        Transitory house price shock distribution. None => degenerate at 1.

    Returns
    -------
    solution_terminal : HousingPortfolioSolution
    """
    rho = CRRA
    gamma = (1.0 + alpha) * (1.0 - rho)
    eps_nodes, eps_probs = _get_eps_H(HousePriceShkDstn)
    kappa = SellCost
    lbar = LTV
    zeta = DefaultPenalty

    def _v_own_terminal(m, h):
        m = np.asarray(m, dtype=float)
        h = np.asarray(h, dtype=float)
        # Broadcast over eps^H: add new axis for shock nodes
        # eps_nodes shape (n_eps,), m/h shape (...); result shape (..., n_eps)
        net_equity = h[..., np.newaxis] * (
            (1.0 - kappa) * eps_nodes[np.newaxis] - lbar
        )
        w_sell = np.maximum(m[..., np.newaxis] + net_equity, 1e-6)
        v_sell = BeqWt * w_sell ** gamma / gamma

        w_def = np.maximum(m[..., np.newaxis], 1e-6)
        v_def = BeqWt * w_def ** gamma / gamma - zeta

        # Probability-weighted expectation over eps^H
        return (eps_probs * np.maximum(v_sell, v_def)).sum(axis=-1)

    def _vp_own_terminal(m, h):
        m = np.asarray(m, dtype=float)
        h = np.asarray(h, dtype=float)
        net_equity = h[..., np.newaxis] * (
            (1.0 - kappa) * eps_nodes[np.newaxis] - lbar
        )
        w_sell_safe = np.maximum(m[..., np.newaxis] + net_equity, 1e-6)
        w_def_safe = np.maximum(m[..., np.newaxis], 1e-6)
        v_sell = BeqWt * w_sell_safe ** gamma / gamma
        v_def = BeqWt * w_def_safe ** gamma / gamma - zeta
        sell_wins = v_sell >= v_def

        vp_sell = BeqWt * w_sell_safe ** (gamma - 1.0)
        vp_def = BeqWt * w_def_safe ** (gamma - 1.0)
        return (eps_probs * np.where(sell_wins, vp_sell, vp_def)).sum(axis=-1)

    def _c_own_terminal(m, h):
        return np.maximum(np.asarray(m, dtype=float), 1e-12)

    # Renter terminal functions
    def _v_rent_terminal(m):
        w = np.maximum(m, 1e-12)
        return BeqWt * w ** gamma / gamma

    def _vp_rent_terminal(m):
        w = np.maximum(m, 1e-12)
        return BeqWt * w ** (gamma - 1.0)

    def _c_rent_terminal(m):
        return np.maximum(m, 1e-12)

    # At terminal: V^P = V^{NP} (no future, so participation status irrelevant)
    solution_terminal = HousingPortfolioSolution(
        cFuncOwn=_c_own_terminal,
        ShareFuncOwn=ConstantFunction(0.0),
        vFuncOwn=_v_own_terminal,
        vPfuncOwn=_vp_own_terminal,
        vFuncOwn_NP=_v_own_terminal,
        vPfuncOwn_NP=_vp_own_terminal,
        tenureFunc=None,
        cFuncRent=_c_rent_terminal,
        ShareFuncRent=ConstantFunction(0.0),
        vFuncRent=_v_rent_terminal,
        vPfuncRent=_vp_rent_terminal,
        vFuncRent_NP=_v_rent_terminal,
        vPfuncRent_NP=_vp_rent_terminal,
    )
    return solution_terminal


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
    """Build per-state income distributions for 4-state Markov (e,nu).

    Returns a list of length T_cycle where each element is a list of four
    ``DiscreteDistribution`` objects:
    ``[emp_boom, emp_slump, unemp_boom, unemp_slump]``.

    States 0,1 (employed) get standard lognormal shocks.
    States 2,3 (unemployed) get same permanent shocks but deterministic
    transitory income = UnempIns.
    The boom/slump distinction does not affect income distributions.

    Parameters
    ----------
    UnempIns : float
        Fraction of permanent income received as unemployment insurance.
    Other parameters: same as construct_lognormal_income_process_unemployment.

    Returns
    -------
    IncShkDstn_Mrkv : list of lists
        ``IncShkDstn_Mrkv[t]`` = ``[dstn_EB, dstn_ES, dstn_UB, dstn_US]``.
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

        unique_perms, inverse = np.unique(perm_shks, return_inverse=True)
        perm_probs = np.bincount(inverse, weights=probs)

        atoms = np.array(
            [unique_perms, np.full(len(unique_perms), UnempIns)]
        )
        unemp_dstn = DiscreteDistribution(perm_probs, atoms)
        unemployed_dstns.append(unemp_dstn)

    # 4 states: (E,B)=0, (E,S)=1, (U,B)=2, (U,S)=3
    # Boom/slump does not affect income, so states 0,1 share employed dstn
    # and states 2,3 share unemployed dstn
    return [
        [employed_dstns[t], employed_dstns[t],
         unemployed_dstns[t], unemployed_dstns[t]]
        for t in range(T_cycle)
    ]


# ---------------------------------------------------------------------------
# Markov employment: terminal solution constructor
# ---------------------------------------------------------------------------


def make_markov_housing_solution_terminal(
    CRRA, alpha, LTV, BeqWt, SellCost, DefaultPenalty,
    MrkvArray, HousePriceShkDstn=None, **kwargs
):
    """Replicate the terminal solution across Markov states.

    The terminal bequest function does not depend on employment or regime,
    so each state gets an identical copy of the base terminal solution.
    """
    base = make_housing_portfolio_solution_terminal(
        CRRA=CRRA, alpha=alpha, LTV=LTV, BeqWt=BeqWt,
        SellCost=SellCost, DefaultPenalty=DefaultPenalty,
        HousePriceShkDstn=HousePriceShkDstn,
    )
    if isinstance(MrkvArray, list):
        N = np.asarray(MrkvArray[0]).shape[0]
    else:
        N = np.asarray(MrkvArray).shape[0]
    return HousingPortfolioSolution(
        cFuncOwn=[base.cFuncOwn] * N,
        ShareFuncOwn=[base.ShareFuncOwn] * N,
        vFuncOwn=[base.vFuncOwn] * N,
        vPfuncOwn=[base.vPfuncOwn] * N,
        vFuncOwn_NP=[base.vFuncOwn_NP] * N,
        vPfuncOwn_NP=[base.vPfuncOwn_NP] * N,
        tenureFunc=[base.tenureFunc] * N,
        cFuncRent=[base.cFuncRent] * N,
        ShareFuncRent=[base.ShareFuncRent] * N,
        vFuncRent=[base.vFuncRent] * N,
        vPfuncRent=[base.vPfuncRent] * N,
        vFuncRent_NP=[base.vFuncRent_NP] * N,
        vPfuncRent_NP=[base.vPfuncRent_NP] * N,
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
    RentRate,
    EntryCost,
    IndepDstnBool=True,
    StockIncCorr=0.0,
):
    """Solve the renter's one-period problem.

    The renter chooses c_t, ell_t (housing services), and varsigma_t.
    With Cobb-Douglas preferences, optimal ell given c yields indirect
    utility u_renter(c) = kappa_r * c^gamma / (1-rho).

    Uses Gomes-Michaelides (2005) entry cost: share > 0 ("in") uses
    V^P_{t+1} continuation; share = 0 ("out") uses V^{NP}_{t+1}.

    Returns
    -------
    cFuncRent, ShareFuncRent, vFuncRent, vPfuncRent : callables
        Participant (V^P) functions.
    vFuncRent_NP, vPfuncRent_NP : callables
        Non-participant (V^{NP}) value and marginal value.
    """
    rho = CRRA
    gamma = (1.0 + alpha) * (1.0 - rho)
    kappa_r = (alpha / RentRate) ** (alpha * (1.0 - rho))
    DiscFacEff = DiscFac * LivPrb

    aNrmGrid = aXtraGrid
    aNrmCount = aNrmGrid.size
    ShareCount = ShareGrid.size

    # Precompute joint shock arrays (renter: no housing shocks)
    _, tran_j, prob_j, risky_j, G_j, G_gamma_j, _ = _build_joint_shocks(
        IncShkDstn, RiskyDstn.atoms[0], RiskyDstn.pmv,
        PermGroFac, gamma, StockIncCorr,
    )

    # Vectorize over shares: R_port has shape (ShareCount, n_shocks)
    R_port = ShareGrid[:, np.newaxis] * risky_j[np.newaxis, :] + (
        (1.0 - ShareGrid[:, np.newaxis]) * Rfree
    )

    # m_next_full: (ShareCount, aNrmCount, n_shocks)
    m_next_full = (
        tran_j[np.newaxis, np.newaxis, :]
        + aNrmGrid[np.newaxis, :, np.newaxis]
        * R_port[:, np.newaxis, :]
        / G_j[np.newaxis, np.newaxis, :]
    )
    m_flat = m_next_full.ravel()

    # GM (2005): share > 0 uses V^P, share = 0 uses V^{NP}
    # Evaluate V^P continuation for all shares (used by share > 0)
    vP_next_P = solution_next.vPfuncRent(m_flat).reshape(m_next_full.shape)
    v_next_P = solution_next.vFuncRent(m_flat).reshape(m_next_full.shape)
    # Evaluate V^{NP} continuation (used by share = 0)
    vP_next_NP = solution_next.vPfuncRent_NP(m_flat).reshape(m_next_full.shape)
    v_next_NP = solution_next.vFuncRent_NP(m_flat).reshape(m_next_full.shape)

    # Build hybrid continuation: share 0 -> NP, shares 1+ -> P
    vP_next = vP_next_P.copy()
    vP_next[0] = vP_next_NP[0]  # share index 0
    v_next = v_next_P.copy()
    v_next[0] = v_next_NP[0]

    # Integration weights: (ShareCount, 1, n_shocks)
    w_base = prob_j[np.newaxis, np.newaxis, :] * G_gamma_j[np.newaxis, np.newaxis, :]
    w_dvda = w_base * R_port[:, np.newaxis, :] / G_j[np.newaxis, np.newaxis, :] * vP_next
    w_v = w_base * v_next

    # Sum over shocks -> (ShareCount, aNrmCount), then transpose -> (aNrmCount, ShareCount)
    EndOfPrd_dvda = (DiscFacEff * w_dvda.sum(axis=2)).T
    EndOfPrd_v = (DiscFacEff * w_v.sum(axis=2)).T

    EndOfPrd_dvda[aNrmGrid < 1e-12, :] = 1e10
    EndOfPrd_v[aNrmGrid < 1e-12, :] = -1e10

    return _renter_egm_envelope(
        EndOfPrd_dvda, EndOfPrd_v, aNrmGrid, ShareGrid,
        alpha, rho, kappa_r, gamma, EntryCost,
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
    hGrid,
    LTV,
    MortRate,
    MaintRate,
    EntryCost,
    HousingGroFac,
    HousingGrowthShkDstn=None,
    IndepDstnBool=True,
    StockIncCorr=0.0,
    HousingIncCorr=0.0,
    HousingStockCorr=0.0,
):
    """Solve the homeowner's one-period continuation problem (conditional on owning).

    Solves the Bellman:
        v^own_t(m,h) = max_{c,varsigma} (c h^alpha)^{1-rho}/(1-rho)
                       + beta s_t E[G^gamma v_{t+1}(m',h')]

    Uses Gomes-Michaelides (2005) entry cost: share > 0 ("in") uses
    V^P_{t+1} continuation; share = 0 ("out") uses V^{NP}_{t+1}.

    h-transition: h' = h * HousingGroFac * eta / G^P

    Returns
    -------
    cFuncOwn, ShareFuncOwn, vFuncOwn, vPfuncOwn : 2D callables (V^P)
    vFuncOwn_NP, vPfuncOwn_NP : 2D callables (V^{NP})
    """
    rho = CRRA
    gamma = (1.0 + alpha) * (1.0 - rho)
    DiscFacEff = DiscFac * LivPrb

    aNrmCount = aXtraGrid.size
    ShareCount = ShareGrid.size
    hCount = hGrid.size

    EndOfPrd_dvda = np.zeros((aNrmCount, hCount, ShareCount))
    EndOfPrd_v = np.zeros((aNrmCount, hCount, ShareCount))

    # Precompute joint shock arrays (owner: with housing shocks)
    _, tran_j, prob_j, risky_j, G_j, G_gamma_j, eta_j = _build_joint_shocks(
        IncShkDstn, RiskyDstn.atoms[0], RiskyDstn.pmv,
        PermGroFac, gamma, StockIncCorr,
        housing_dstn=HousingGrowthShkDstn,
        HousingIncCorr=HousingIncCorr,
        HousingStockCorr=HousingStockCorr,
    )

    GH = HousingGroFac  # deterministic housing growth for this regime
    n_shocks = prob_j.size

    # Precompute R_port for all shares: shape (ShareCount, n_shocks)
    R_port = ShareGrid[:, np.newaxis] * risky_j[np.newaxis, :] + (
        (1.0 - ShareGrid[:, np.newaxis]) * Rfree
    )

    for i_h, h_now in enumerate(hGrid):
        # h' = h_now * GH * eta_j / G_j
        h_next_j = h_now * GH * eta_j / G_j
        h_next_j = np.clip(h_next_j, hGrid[0], hGrid[-1])

        # m_next: shape (ShareCount, aNrmCount, n_shocks)
        m_next = (
            tran_j[np.newaxis, np.newaxis, :]
            + aXtraGrid[np.newaxis, :, np.newaxis]
            * R_port[:, np.newaxis, :]
            / G_j[np.newaxis, np.newaxis, :]
        )
        m_flat = m_next.ravel()
        h_flat = np.tile(h_next_j, ShareCount * aNrmCount)

        # GM (2005): share > 0 uses V^P, share = 0 uses V^{NP}
        vP_next_P = solution_next.vPfuncOwn(m_flat, h_flat).reshape(m_next.shape)
        v_next_P = solution_next.vFuncOwn(m_flat, h_flat).reshape(m_next.shape)
        vP_next_NP = solution_next.vPfuncOwn_NP(m_flat, h_flat).reshape(m_next.shape)
        v_next_NP = solution_next.vFuncOwn_NP(m_flat, h_flat).reshape(m_next.shape)

        # Hybrid: share 0 -> NP, shares 1+ -> P
        vP_next = vP_next_P.copy()
        vP_next[0] = vP_next_NP[0]
        v_next = v_next_P.copy()
        v_next[0] = v_next_NP[0]

        # Integration: (ShareCount, 1, n_shocks) broadcasting
        w_base = prob_j[np.newaxis, np.newaxis, :] * G_gamma_j[np.newaxis, np.newaxis, :]
        w_dvda = w_base * R_port[:, np.newaxis, :] / G_j[np.newaxis, np.newaxis, :] * vP_next
        w_v = w_base * v_next

        # Sum over shocks -> (ShareCount, aNrmCount), transpose -> (aNrmCount, ShareCount)
        EndOfPrd_dvda[:, i_h, :] = (DiscFacEff * w_dvda.sum(axis=2)).T
        EndOfPrd_v[:, i_h, :] = (DiscFacEff * w_v.sum(axis=2)).T

    EndOfPrd_dvda[aXtraGrid < 1e-12, :, :] = 1e10
    EndOfPrd_v[aXtraGrid < 1e-12, :, :] = -1e10

    return _owner_egm_envelope(
        EndOfPrd_dvda, EndOfPrd_v, aXtraGrid, ShareGrid, hGrid,
        alpha, rho, LTV, MortRate, MaintRate, EntryCost,
    )


def _compute_tenure_envelope(
    m_eval, h_eval, vFuncOwn_stay, vPfuncOwn_stay, cFuncOwn, ShareFuncOwn,
    vFuncRent, vPfuncRent, cFuncRent, ShareFuncRent,
    LTV, MortRate, MaintRate, hGrid, AffordabilityLimit,
    SellCost, DefaultPenalty, HousePriceShkDstn=None,
):
    """Compute the tenure choice envelope over a (m, h) grid.

    For each h-slice and each transitory house price shock node eps^H,
    computes the upper envelope across own/sell/move/default.

    Returns
    -------
    cFuncOwn_final, ShareFuncOwn_final, vFuncOwn_final, vPfuncOwn_final,
    tenureFunc : LinearInterpOnInterp1D
    """
    eps_nodes, eps_probs = _get_eps_H(HousePriceShkDstn)
    n_m = len(m_eval)
    n_h = h_eval.size
    lbar = LTV
    kappa = SellCost
    zeta = DefaultPenalty

    affordable = (MortRate * lbar + MaintRate) * h_eval <= AffordabilityLimit

    # Pre-evaluate renter functions on m_eval
    v_rent_m = vFuncRent(m_eval)
    c_rent_m = cFuncRent(m_eval)
    s_rent_m = ShareFuncRent(m_eval)
    vp_rent_m = vPfuncRent(m_eval)

    # Pre-evaluate owner functions on (m_eval x h_eval)
    mm, hh = np.meshgrid(m_eval, h_eval, indexing="ij")
    flat_m, flat_h = mm.ravel(), hh.ravel()
    v_own_all = vFuncOwn_stay(flat_m, flat_h).reshape(mm.shape)
    c_own_all = cFuncOwn(flat_m, flat_h).reshape(mm.shape)
    s_own_all = ShareFuncOwn(flat_m, flat_h).reshape(mm.shape)
    vp_own_all = vPfuncOwn_stay(flat_m, flat_h).reshape(mm.shape)
    v_own_all[:, ~affordable] = -np.inf

    # Accumulate eps^H-weighted values on the common m_eval grid
    v_integrated = np.zeros((n_m, n_h))
    vp_integrated = np.zeros((n_m, n_h))

    # Precompute vp_own_all with affordability mask
    vp_own_masked = vp_own_all.copy()
    vp_own_masked[:, ~affordable] = 0.0

    # Default branch (does not depend on eps^H or h)
    v_def = v_rent_m - zeta  # shape (n_m,)
    vp_def = vp_rent_m        # shape (n_m,)

    # Down payment costs for move branch: shape (n_h_prime,)
    down_costs = (1.0 - lbar) * hGrid

    for eps_H, eps_prob in zip(eps_nodes, eps_probs):
        # Sell branch: vectorize over all h simultaneously
        # net_equity: shape (n_m, n_h) via broadcasting
        net_equity = h_eval[np.newaxis, :] * ((1.0 - kappa) * eps_H - lbar)
        m_after_sell = m_eval[:, np.newaxis] + net_equity  # (n_m, n_h)
        can_sell = m_after_sell >= 0
        m_sell_safe = np.maximum(m_after_sell, 1e-6)

        # Evaluate renter v/vp at all (m, h) sell proceeds at once
        v_sell = np.where(can_sell, vFuncRent(m_sell_safe.ravel()).reshape(n_m, n_h), -np.inf)
        vp_sell = np.where(can_sell, vPfuncRent(m_sell_safe.ravel()).reshape(n_m, n_h), 0.0)

        # Move branch: for each h_prime candidate, evaluate across all (m, h) at once
        v_move = np.full((n_m, n_h), -np.inf)
        vp_move = np.zeros((n_m, n_h))
        for i_hp, h_prime in enumerate(hGrid):
            m_after_move = m_sell_safe - down_costs[i_hp]  # (n_m, n_h)
            can_move = can_sell & (m_after_move >= 1e-6)
            if not can_move.any():
                continue
            m_move_safe = np.where(can_move, m_after_move, 1e-6)
            h_arr = np.full_like(m_move_safe, h_prime)
            v_cand = np.where(
                can_move,
                vFuncOwn_stay(m_move_safe.ravel(), h_arr.ravel()).reshape(n_m, n_h),
                -np.inf,
            )
            better = v_cand > v_move
            v_move = np.where(better, v_cand, v_move)
            vp_cand = np.where(
                can_move,
                vPfuncOwn_stay(m_move_safe.ravel(), h_arr.ravel()).reshape(n_m, n_h),
                0.0,
            )
            vp_move = np.where(better, vp_cand, vp_move)

        # Pointwise max across four branches: shape (4, n_m, n_h)
        v_stack = np.array([
            v_own_all,
            v_sell,
            v_move,
            np.broadcast_to(v_def[:, np.newaxis], (n_m, n_h)),
        ])
        winner = np.argmax(v_stack, axis=0)  # (n_m, n_h)
        idx_m, idx_h = np.meshgrid(np.arange(n_m), np.arange(n_h), indexing="ij")
        v_best = v_stack[winner, idx_m, idx_h]

        vp_stack = np.array([
            vp_own_masked,
            vp_sell,
            vp_move,
            np.broadcast_to(vp_def[:, np.newaxis], (n_m, n_h)),
        ])
        vp_best = vp_stack[winner, idx_m, idx_h]

        v_integrated += eps_prob * v_best
        vp_integrated += eps_prob * vp_best

    # Build eps^H-integrated value/marginal-value interpolants
    vFunc_own_list = []
    vpFunc_own_list = []
    for i_h in range(n_h):
        vFunc_own_list.append(LinearInterp(m_eval, v_integrated[:, i_h]))
        vpFunc_own_list.append(LinearInterp(m_eval, vp_integrated[:, i_h]))

    # Policy functions: use _value_envelope at eps^H = 1 for crossing points
    # Precompute sell branch at eps^H=1 for all h simultaneously
    net_equity_ref = h_eval[np.newaxis, :] * ((1.0 - kappa) - lbar)  # (1, n_h)
    m_after_sell_ref = m_eval[:, np.newaxis] + net_equity_ref  # (n_m, n_h)
    can_sell_ref = m_after_sell_ref >= 0
    m_sell_safe_ref = np.maximum(m_after_sell_ref, 1e-6)
    flat_ms = m_sell_safe_ref.ravel()
    v_sell_ref_all = np.where(can_sell_ref, vFuncRent(flat_ms).reshape(n_m, n_h), -np.inf)
    c_sell_ref_all = np.where(can_sell_ref, cFuncRent(flat_ms).reshape(n_m, n_h), 0.0)
    s_sell_ref_all = np.where(can_sell_ref, ShareFuncRent(flat_ms).reshape(n_m, n_h), 0.0)
    vp_sell_ref_all = np.where(can_sell_ref, vPfuncRent(flat_ms).reshape(n_m, n_h), 0.0)

    # Precompute move branch at eps^H=1 for all h simultaneously
    v_move_ref_all = np.full((n_m, n_h), -np.inf)
    c_move_ref_all = np.zeros((n_m, n_h))
    s_move_ref_all = np.zeros((n_m, n_h))
    vp_move_ref_all = np.zeros((n_m, n_h))
    for i_hp, h_prime in enumerate(hGrid):
        m_after_move = m_sell_safe_ref - down_costs[i_hp]  # (n_m, n_h)
        can_move = can_sell_ref & (m_after_move >= 1e-6)
        if not can_move.any():
            continue
        m_move_safe = np.where(can_move, m_after_move, 1e-6)
        h_arr = np.full_like(m_move_safe, h_prime)
        flat_mm = m_move_safe.ravel()
        flat_hh = h_arr.ravel()
        v_cand = np.where(
            can_move,
            vFuncOwn_stay(flat_mm, flat_hh).reshape(n_m, n_h),
            -np.inf,
        )
        better = v_cand > v_move_ref_all
        v_move_ref_all = np.where(better, v_cand, v_move_ref_all)
        c_move_ref_all = np.where(
            better,
            np.where(can_move, cFuncOwn(flat_mm, flat_hh).reshape(n_m, n_h), 0.0),
            c_move_ref_all,
        )
        s_move_ref_all = np.where(
            better,
            np.where(can_move, ShareFuncOwn(flat_mm, flat_hh).reshape(n_m, n_h), 0.0),
            s_move_ref_all,
        )
        vp_move_ref_all = np.where(
            better,
            np.where(can_move, vPfuncOwn_stay(flat_mm, flat_hh).reshape(n_m, n_h), 0.0),
            vp_move_ref_all,
        )

    # Precompute owner policies with affordability mask
    c_own_masked = c_own_all.copy()
    c_own_masked[:, ~affordable] = 0.0
    s_own_masked = s_own_all.copy()
    s_own_masked[:, ~affordable] = 0.0

    # Default branch (same for all h)
    v_def_col = v_rent_m - zeta
    c_def_col = c_rent_m
    s_def_col = s_rent_m
    vp_def_col = vp_rent_m

    cFunc_own_list = []
    shareFunc_own_list = []
    tenure_list = []

    for i_h in range(n_h):
        m_env, v_env, pol_envs, env_inds = _value_envelope(
            m_eval,
            [v_own_all[:, i_h], v_sell_ref_all[:, i_h],
             v_move_ref_all[:, i_h], v_def_col],
            [
                (c_own_masked[:, i_h], s_own_masked[:, i_h], vp_own_masked[:, i_h]),
                (c_sell_ref_all[:, i_h], s_sell_ref_all[:, i_h], vp_sell_ref_all[:, i_h]),
                (c_move_ref_all[:, i_h], s_move_ref_all[:, i_h], vp_move_ref_all[:, i_h]),
                (c_def_col, s_def_col, vp_def_col),
            ],
        )
        c_env, s_env, _ = pol_envs
        tenure_env = env_inds.astype(float)

        cFunc_own_list.append(LinearInterp(m_env, c_env))
        shareFunc_own_list.append(LinearInterp(m_env, s_env))
        tenure_list.append(LinearInterp(m_env, tenure_env))

    return (
        LinearInterpOnInterp1D(cFunc_own_list, h_eval),
        LinearInterpOnInterp1D(shareFunc_own_list, h_eval),
        LinearInterpOnInterp1D(vFunc_own_list, h_eval),
        LinearInterpOnInterp1D(vpFunc_own_list, h_eval),
        LinearInterpOnInterp1D(tenure_list, h_eval),
    )


def _compute_repurchase_option(
    aXtraGrid, vFuncRent, vPfuncRent, cFuncRent, ShareFuncRent,
    vFuncOwn_stay, vPfuncOwn_stay, cFuncOwn, ShareFuncOwn,
    LTV, MaxDTI, hGrid,
):
    """Compute the renter repurchase option.

    For each feasible h' on hGrid (subject to lbar*h' <= MaxDTI), evaluate
    buying vs renting and take the best h'.

    Returns
    -------
    cFuncRent_final, ShareFuncRent_final, vFuncRent_final, vPfuncRent_final
    """
    lbar = LTV
    m_rent_fine = np.linspace(1e-6, aXtraGrid[-1] * 2.0, 200)
    v_rent_vals = vFuncRent(m_rent_fine)
    c_rent_vals = cFuncRent(m_rent_fine)
    s_rent_vals = ShareFuncRent(m_rent_fine)
    vp_rent_vals = vPfuncRent(m_rent_fine)

    # Find best h' for buying
    v_buy_best = np.full_like(m_rent_fine, -np.inf)
    c_buy_best = np.zeros_like(m_rent_fine)
    s_buy_best = np.zeros_like(m_rent_fine)
    vp_buy_best = np.zeros_like(m_rent_fine)

    any_feasible = False
    for h_prime in hGrid:
        if lbar * h_prime > MaxDTI:
            continue
        down_cost = (1.0 - lbar) * h_prime
        m_after_buy = m_rent_fine - down_cost
        can_buy = m_after_buy >= 1e-6
        if not can_buy.any():
            continue
        any_feasible = True
        m_buy_safe = np.where(can_buy, m_after_buy, 1e-6)
        h_arr = np.full_like(m_buy_safe, h_prime)

        v_buy_vals = np.where(
            can_buy,
            vFuncOwn_stay(m_buy_safe, h_arr),
            -np.inf,
        )
        better = v_buy_vals > v_buy_best
        v_buy_best = np.where(better, v_buy_vals, v_buy_best)
        c_buy_best = np.where(
            better,
            np.where(can_buy, cFuncOwn(m_buy_safe, h_arr), 0.0),
            c_buy_best,
        )
        s_buy_best = np.where(
            better,
            np.where(can_buy, ShareFuncOwn(m_buy_safe, h_arr), 0.0),
            s_buy_best,
        )
        vp_buy_best = np.where(
            better,
            np.where(can_buy, vPfuncOwn_stay(m_buy_safe, h_arr), 0.0),
            vp_buy_best,
        )

    if any_feasible:
        m_env, v_env, pol_envs, _ = _value_envelope(
            m_rent_fine,
            [v_rent_vals, v_buy_best],
            [
                (c_rent_vals, s_rent_vals, vp_rent_vals),
                (c_buy_best, s_buy_best, vp_buy_best),
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
# One-period solver: base (non-Markov)
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
    hGrid,
    LTV,
    MortRate,
    MaintRate,
    EntryCost,
    SellCost,
    DefaultPenalty,
    RentRate,
    BoroCnstArt,
    IndepDstnBool,
    ShareLimit,
    HousingGroFac,
    AffordabilityLimit=np.inf,
    MaxDTI=4.0,
    StockIncCorr=0.0,
    HousingIncCorr=0.0,
    HousingStockCorr=0.0,
    HousePriceShkDstn=None,
    HousingGrowthShkDstn=None,
    **kwargs,
):
    """Solve one period of the housing portfolio choice model.

    This function:
    1. Solves the renter subproblem (1D in m), producing V^P and V^{NP}.
    2. Solves the owner continuation subproblem (2D in (m, h)), producing V^P and V^{NP}.
    3. Takes the upper envelope across tenure choices, separately for P and NP.
    4. Computes repurchase option for renters, separately for P and NP.

    Returns
    -------
    solution_now : HousingPortfolioSolution
    """
    # Step 1: Solve the renter subproblem (returns P and NP)
    (cFuncRent, ShareFuncRent, vFuncRent, vPfuncRent,
     vFuncRent_NP, vPfuncRent_NP) = solve_renter_subproblem(
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
        RentRate,
        EntryCost,
        IndepDstnBool,
        StockIncCorr,
    )

    # Step 2: Solve the owner continuation (returns P and NP)
    (cFuncOwn, ShareFuncOwn, vFuncOwn_stay, vPfuncOwn_stay,
     vFuncOwn_stay_NP, vPfuncOwn_stay_NP) = solve_owner_subproblem(
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
        hGrid,
        LTV,
        MortRate,
        MaintRate,
        EntryCost,
        HousingGroFac,
        HousingGrowthShkDstn,
        IndepDstnBool,
        StockIncCorr,
        HousingIncCorr,
        HousingStockCorr,
    )

    # Step 3: Tenure envelope — participant (V^P)
    m_eval = np.linspace(1e-6, aXtraGrid[-1] * 2.0, _N_ENVELOPE_POINTS)
    h_eval = hGrid

    cFuncOwn_final, ShareFuncOwn_final, vFuncOwn_final, vPfuncOwn_final, tenureFunc = (
        _compute_tenure_envelope(
            m_eval, h_eval, vFuncOwn_stay, vPfuncOwn_stay, cFuncOwn, ShareFuncOwn,
            vFuncRent, vPfuncRent, cFuncRent, ShareFuncRent,
            LTV, MortRate, MaintRate, hGrid, AffordabilityLimit,
            SellCost, DefaultPenalty, HousePriceShkDstn,
        )
    )

    # Tenure envelope — non-participant (V^{NP})
    _, _, vFuncOwn_final_NP, vPfuncOwn_final_NP, _ = (
        _compute_tenure_envelope(
            m_eval, h_eval, vFuncOwn_stay_NP, vPfuncOwn_stay_NP, cFuncOwn, ShareFuncOwn,
            vFuncRent_NP, vPfuncRent_NP, cFuncRent, ShareFuncRent,
            LTV, MortRate, MaintRate, hGrid, AffordabilityLimit,
            SellCost, DefaultPenalty, HousePriceShkDstn,
        )
    )

    # Step 4: Repurchase envelope — participant (V^P)
    cFuncRent_final, ShareFuncRent_final, vFuncRent_final, vPfuncRent_final = (
        _compute_repurchase_option(
            aXtraGrid, vFuncRent, vPfuncRent, cFuncRent, ShareFuncRent,
            vFuncOwn_stay, vPfuncOwn_stay, cFuncOwn, ShareFuncOwn,
            LTV, MaxDTI, hGrid,
        )
    )

    # Repurchase envelope — non-participant (V^{NP})
    _, _, vFuncRent_final_NP, vPfuncRent_final_NP = (
        _compute_repurchase_option(
            aXtraGrid, vFuncRent_NP, vPfuncRent_NP, cFuncRent, ShareFuncRent,
            vFuncOwn_stay_NP, vPfuncOwn_stay_NP, cFuncOwn, ShareFuncOwn,
            LTV, MaxDTI, hGrid,
        )
    )

    mNrmMin = 0.0

    solution_now = HousingPortfolioSolution(
        cFuncOwn=cFuncOwn_final,
        ShareFuncOwn=ShareFuncOwn_final,
        vFuncOwn=vFuncOwn_final,
        vPfuncOwn=vPfuncOwn_final,
        vFuncOwn_NP=vFuncOwn_final_NP,
        vPfuncOwn_NP=vPfuncOwn_final_NP,
        tenureFunc=tenureFunc,
        cFuncRent=cFuncRent_final,
        ShareFuncRent=ShareFuncRent_final,
        vFuncRent=vFuncRent_final,
        vPfuncRent=vPfuncRent_final,
        vFuncRent_NP=vFuncRent_final_NP,
        vPfuncRent_NP=vPfuncRent_final_NP,
        mNrmMin=mNrmMin,
    )
    solution_now.mGrid = m_eval
    solution_now.hGrid = h_eval

    return solution_now


# ---------------------------------------------------------------------------
# One-period solver: Markov renter subproblem
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
    RentRate,
    EntryCost,
    IndepDstnBool=True,
    StockIncCorr=0.0,
):
    """Solve the renter's one-period problem with Markov transitions.

    Uses Gomes-Michaelides (2005) entry cost: share > 0 uses V^P_{t+1},
    share = 0 uses V^{NP}_{t+1}.

    Returns
    -------
    cFuncRent, ShareFuncRent, vFuncRent, vPfuncRent : callables (V^P)
    vFuncRent_NP, vPfuncRent_NP : callables (V^{NP})
    """
    rho = CRRA
    gamma = (1.0 + alpha) * (1.0 - rho)
    kappa_r = (alpha / RentRate) ** (alpha * (1.0 - rho))
    DiscFacEff = DiscFac * LivPrb

    aNrmGrid = aXtraGrid
    aNrmCount = aNrmGrid.size
    ShareCount = ShareGrid.size
    n_states = len(IncShkDstn_list)

    risky_rets_base = RiskyDstn.atoms[0]
    prob_ret_arr = RiskyDstn.pmv

    # Precompute joint shocks per next-state (hoisted outside share loop)
    shock_data = []
    for e_next in range(n_states):
        trans_prob = MrkvRow[e_next]
        if trans_prob < 1e-15:
            continue
        dstn = IncShkDstn_list[e_next]
        _, tran_j, prob_j, risky_j, G_j, G_gamma_j, _ = _build_joint_shocks(
            dstn, risky_rets_base, prob_ret_arr,
            PermGroFac, gamma, StockIncCorr,
        )
        shock_data.append((
            tran_j, prob_j * trans_prob, risky_j, G_j, G_gamma_j,
            solution_next.vPfuncRent[e_next],
            solution_next.vFuncRent[e_next],
            solution_next.vPfuncRent_NP[e_next],
            solution_next.vFuncRent_NP[e_next],
        ))

    # Vectorize over shares: R_port shape (ShareCount, n_shocks_j)
    EndOfPrd_dvda = np.zeros((aNrmCount, ShareCount))
    EndOfPrd_v = np.zeros((aNrmCount, ShareCount))

    for (tran_j, prob_j, risky_j, G_j, G_gamma_j,
         vPf_P, vFn_P, vPf_NP, vFn_NP) in shock_data:
        R_port = ShareGrid[:, np.newaxis] * risky_j[np.newaxis, :] + (
            (1.0 - ShareGrid[:, np.newaxis]) * Rfree
        )
        # m_next: (ShareCount, aNrmCount, n_shocks)
        m_next_full = (
            tran_j[np.newaxis, np.newaxis, :]
            + aNrmGrid[np.newaxis, :, np.newaxis]
            * R_port[:, np.newaxis, :]
            / G_j[np.newaxis, np.newaxis, :]
        )
        m_flat = m_next_full.ravel()

        # GM (2005): share > 0 uses V^P, share = 0 uses V^{NP}
        vP_next_P = vPf_P(m_flat).reshape(m_next_full.shape)
        v_next_P = vFn_P(m_flat).reshape(m_next_full.shape)
        vP_next_NP = vPf_NP(m_flat).reshape(m_next_full.shape)
        v_next_NP = vFn_NP(m_flat).reshape(m_next_full.shape)

        vP_next = vP_next_P.copy()
        vP_next[0] = vP_next_NP[0]
        v_next = v_next_P.copy()
        v_next[0] = v_next_NP[0]

        w_base = prob_j[np.newaxis, np.newaxis, :] * G_gamma_j[np.newaxis, np.newaxis, :]
        w_dvda = w_base * R_port[:, np.newaxis, :] / G_j[np.newaxis, np.newaxis, :] * vP_next
        w_v = w_base * v_next

        EndOfPrd_dvda += (w_dvda.sum(axis=2)).T
        EndOfPrd_v += (w_v.sum(axis=2)).T

    EndOfPrd_dvda *= DiscFacEff
    EndOfPrd_v *= DiscFacEff
    EndOfPrd_dvda[aNrmGrid < 1e-12, :] = 1e10
    EndOfPrd_v[aNrmGrid < 1e-12, :] = -1e10

    return _renter_egm_envelope(
        EndOfPrd_dvda, EndOfPrd_v, aNrmGrid, ShareGrid,
        alpha, rho, kappa_r, gamma, EntryCost,
    )


# ---------------------------------------------------------------------------
# One-period solver: Markov owner subproblem
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
    hGrid,
    LTV,
    MortRate,
    MaintRate,
    EntryCost,
    HousingGroFac,
    HousingGrowthShkDstn=None,
    IndepDstnBool=True,
    StockIncCorr=0.0,
    HousingIncCorr=0.0,
    HousingStockCorr=0.0,
):
    """Solve the homeowner's continuation problem with Markov transitions.

    Uses Gomes-Michaelides (2005) entry cost: share > 0 uses V^P_{t+1},
    share = 0 uses V^{NP}_{t+1}.

    Returns
    -------
    cFuncOwn, ShareFuncOwn, vFuncOwn, vPfuncOwn : 2D callables (V^P)
    vFuncOwn_NP, vPfuncOwn_NP : 2D callables (V^{NP})
    """
    rho = CRRA
    gamma = (1.0 + alpha) * (1.0 - rho)
    DiscFacEff = DiscFac * LivPrb

    aNrmCount = aXtraGrid.size
    ShareCount = ShareGrid.size
    hCount = hGrid.size
    n_states = len(IncShkDstn_list)

    risky_rets_base = RiskyDstn.atoms[0]
    prob_ret_arr = RiskyDstn.pmv

    GH = HousingGroFac

    # Precompute joint shocks per next-state (hoisted outside h and share loops)
    shock_data = []
    for e_next in range(n_states):
        trans_prob = MrkvRow[e_next]
        if trans_prob < 1e-15:
            continue
        dstn = IncShkDstn_list[e_next]
        _, tran_j, prob_j, risky_j, G_j, G_gamma_j, eta_j = _build_joint_shocks(
            dstn, risky_rets_base, prob_ret_arr,
            PermGroFac, gamma, StockIncCorr,
            housing_dstn=HousingGrowthShkDstn,
            HousingIncCorr=HousingIncCorr,
            HousingStockCorr=HousingStockCorr,
        )
        shock_data.append((
            tran_j, prob_j * trans_prob, risky_j, G_j, G_gamma_j, eta_j,
            solution_next.vPfuncOwn[e_next],
            solution_next.vFuncOwn[e_next],
            solution_next.vPfuncOwn_NP[e_next],
            solution_next.vFuncOwn_NP[e_next],
        ))

    EndOfPrd_dvda = np.zeros((aNrmCount, hCount, ShareCount))
    EndOfPrd_v = np.zeros((aNrmCount, hCount, ShareCount))

    for (tran_j, prob_j, risky_j, G_j, G_gamma_j, eta_j,
         vPfOwn_P, vFOwn_P, vPfOwn_NP, vFOwn_NP) in shock_data:
        n_shocks = prob_j.size
        # R_port: (ShareCount, n_shocks)
        R_port = ShareGrid[:, np.newaxis] * risky_j[np.newaxis, :] + (
            (1.0 - ShareGrid[:, np.newaxis]) * Rfree
        )

        for i_h, h_now in enumerate(hGrid):
            h_next_j = np.clip(h_now * GH * eta_j / G_j, hGrid[0], hGrid[-1])

            # m_next: (ShareCount, aNrmCount, n_shocks)
            m_next = (
                tran_j[np.newaxis, np.newaxis, :]
                + aXtraGrid[np.newaxis, :, np.newaxis]
                * R_port[:, np.newaxis, :]
                / G_j[np.newaxis, np.newaxis, :]
            )
            m_flat = m_next.ravel()
            h_flat = np.tile(h_next_j, ShareCount * aNrmCount)

            # GM (2005): share > 0 uses V^P, share = 0 uses V^{NP}
            vP_next_P = vPfOwn_P(m_flat, h_flat).reshape(m_next.shape)
            v_next_P = vFOwn_P(m_flat, h_flat).reshape(m_next.shape)
            vP_next_NP = vPfOwn_NP(m_flat, h_flat).reshape(m_next.shape)
            v_next_NP = vFOwn_NP(m_flat, h_flat).reshape(m_next.shape)

            vP_next = vP_next_P.copy()
            vP_next[0] = vP_next_NP[0]
            v_next = v_next_P.copy()
            v_next[0] = v_next_NP[0]

            w_base = prob_j[np.newaxis, np.newaxis, :] * G_gamma_j[np.newaxis, np.newaxis, :]
            w_dvda = w_base * R_port[:, np.newaxis, :] / G_j[np.newaxis, np.newaxis, :] * vP_next
            w_v = w_base * v_next

            # (ShareCount, aNrmCount) -> transpose -> (aNrmCount, ShareCount)
            EndOfPrd_dvda[:, i_h, :] += (w_dvda.sum(axis=2)).T
            EndOfPrd_v[:, i_h, :] += (w_v.sum(axis=2)).T

    EndOfPrd_dvda *= DiscFacEff
    EndOfPrd_v *= DiscFacEff
    EndOfPrd_dvda[aXtraGrid < 1e-12, :, :] = 1e10
    EndOfPrd_v[aXtraGrid < 1e-12, :, :] = -1e10

    return _owner_egm_envelope(
        EndOfPrd_dvda, EndOfPrd_v, aXtraGrid, ShareGrid, hGrid,
        alpha, rho, LTV, MortRate, MaintRate, EntryCost,
    )


# ---------------------------------------------------------------------------
# One-period solver: Markov housing portfolio choice
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
    hGrid,
    LTV,
    MortRate,
    MaintRate,
    EntryCost,
    SellCost,
    DefaultPenalty,
    RentRate,
    BoroCnstArt,
    IndepDstnBool,
    ShareLimit,
    MrkvArray,
    HousingGroFac,
    AffordabilityLimit=np.inf,
    MaxDTI=4.0,
    StockIncCorr=0.0,
    HousingIncCorr=0.0,
    HousingStockCorr=0.0,
    HousePriceShkDstn=None,
    HousingGrowthShkDstn=None,
    **kwargs,
):
    """Solve one period of the housing portfolio model with Markov states.

    Iterates over each current state *i* in the 4-state space
    (E,B)=0, (E,S)=1, (U,B)=2, (U,S)=3.

    Parameters
    ----------
    HousingGroFac : list or float
        If list, [GH_boom, GH_slump]. If float, same for both regimes.

    Returns
    -------
    solution_now : HousingPortfolioSolution
        Solution with list-valued attributes (length N).
    """
    N = MrkvArray.shape[0]

    # Parse regime-dependent housing growth
    if isinstance(HousingGroFac, (list, np.ndarray)):
        GH_arr = np.asarray(HousingGroFac)
    else:
        GH_arr = np.array([HousingGroFac, HousingGroFac])

    state_solutions = []
    for i_state in range(N):
        MrkvRow = MrkvArray[i_state]

        # Regime index: nu_state = i_state % 2 (0=boom, 1=slump)
        nu_state = i_state % 2
        GH_now = float(GH_arr[nu_state]) if nu_state < len(GH_arr) else float(GH_arr[-1])

        # Step 1: Solve the renter subproblem (P and NP)
        (cFuncRent, ShareFuncRent, vFuncRent, vPfuncRent,
         vFuncRent_NP, vPfuncRent_NP) = (
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
                RentRate,
                EntryCost,
                IndepDstnBool,
                StockIncCorr,
            )
        )

        # Step 2: Solve the owner subproblem (P and NP)
        (cFuncOwn, ShareFuncOwn, vFuncOwn_stay, vPfuncOwn_stay,
         vFuncOwn_stay_NP, vPfuncOwn_stay_NP) = (
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
                hGrid,
                LTV,
                MortRate,
                MaintRate,
                EntryCost,
                GH_now,
                HousingGrowthShkDstn,
                IndepDstnBool,
                StockIncCorr,
                HousingIncCorr,
                HousingStockCorr,
            )
        )

        # Step 3: Tenure envelope — participant (V^P)
        m_eval = np.linspace(1e-6, aXtraGrid[-1] * 2.0, _N_ENVELOPE_POINTS)
        h_eval = hGrid

        cFuncOwn_2d, ShareFuncOwn_2d, vFuncOwn_2d, vPfuncOwn_2d, tenureFunc_2d = (
            _compute_tenure_envelope(
                m_eval, h_eval, vFuncOwn_stay, vPfuncOwn_stay, cFuncOwn, ShareFuncOwn,
                vFuncRent, vPfuncRent, cFuncRent, ShareFuncRent,
                LTV, MortRate, MaintRate, hGrid, AffordabilityLimit,
                SellCost, DefaultPenalty, HousePriceShkDstn,
            )
        )

        # Tenure envelope — non-participant (V^{NP})
        _, _, vFuncOwn_2d_NP, vPfuncOwn_2d_NP, _ = (
            _compute_tenure_envelope(
                m_eval, h_eval, vFuncOwn_stay_NP, vPfuncOwn_stay_NP,
                cFuncOwn, ShareFuncOwn,
                vFuncRent_NP, vPfuncRent_NP, cFuncRent, ShareFuncRent,
                LTV, MortRate, MaintRate, hGrid, AffordabilityLimit,
                SellCost, DefaultPenalty, HousePriceShkDstn,
            )
        )

        # Step 4: Repurchase envelope — participant (V^P)
        cFuncRent_final, ShareFuncRent_final, vFuncRent_final, vPfuncRent_final = (
            _compute_repurchase_option(
                aXtraGrid, vFuncRent, vPfuncRent, cFuncRent, ShareFuncRent,
                vFuncOwn_stay, vPfuncOwn_stay, cFuncOwn, ShareFuncOwn,
                LTV, MaxDTI, hGrid,
            )
        )

        # Repurchase envelope — non-participant (V^{NP})
        _, _, vFuncRent_final_NP, vPfuncRent_final_NP = (
            _compute_repurchase_option(
                aXtraGrid, vFuncRent_NP, vPfuncRent_NP, cFuncRent, ShareFuncRent,
                vFuncOwn_stay_NP, vPfuncOwn_stay_NP, cFuncOwn, ShareFuncOwn,
                LTV, MaxDTI, hGrid,
            )
        )

        sol_i = HousingPortfolioSolution(
            cFuncOwn=cFuncOwn_2d,
            ShareFuncOwn=ShareFuncOwn_2d,
            vFuncOwn=vFuncOwn_2d,
            vPfuncOwn=vPfuncOwn_2d,
            vFuncOwn_NP=vFuncOwn_2d_NP,
            vPfuncOwn_NP=vPfuncOwn_2d_NP,
            tenureFunc=tenureFunc_2d,
            cFuncRent=cFuncRent_final,
            ShareFuncRent=ShareFuncRent_final,
            vFuncRent=vFuncRent_final,
            vPfuncRent=vPfuncRent_final,
            vFuncRent_NP=vFuncRent_final_NP,
            vPfuncRent_NP=vPfuncRent_final_NP,
        )
        sol_i.mGrid = m_eval
        sol_i.hGrid = h_eval
        state_solutions.append(sol_i)

    combined = HousingPortfolioSolution(
        cFuncOwn=[s.cFuncOwn for s in state_solutions],
        ShareFuncOwn=[s.ShareFuncOwn for s in state_solutions],
        vFuncOwn=[s.vFuncOwn for s in state_solutions],
        vPfuncOwn=[s.vPfuncOwn for s in state_solutions],
        vFuncOwn_NP=[s.vFuncOwn_NP for s in state_solutions],
        vPfuncOwn_NP=[s.vPfuncOwn_NP for s in state_solutions],
        tenureFunc=[s.tenureFunc for s in state_solutions],
        cFuncRent=[s.cFuncRent for s in state_solutions],
        ShareFuncRent=[s.ShareFuncRent for s in state_solutions],
        vFuncRent=[s.vFuncRent for s in state_solutions],
        vPfuncRent=[s.vPfuncRent for s in state_solutions],
        vFuncRent_NP=[s.vFuncRent_NP for s in state_solutions],
        vPfuncRent_NP=[s.vPfuncRent_NP for s in state_solutions],
        mNrmMin=0.0,
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
    [1.03] * 10
    + [1.02] * 10
    + [1.01] * 10
    + [1.00] * 10
    + [1.00] * 10
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
    "hGrid": make_h_grid,
    "HousePriceShkDstn": make_house_price_shock_dstn,
    "HousingGrowthShkDstn": make_housing_growth_shock_dstn,
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
    "RiskyAvg": 1.06,
    "RiskyStd": 0.18,
    "RiskyCount": 5,
}

HousingPortfolioType_ShareGrid_default = {
    "ShareCount": 15,
}

HousingPortfolioType_hGrid_default = {
    "hGridCount": 10,
    "hMin": 0.5,
    "hMax": 8.0,
}

HousingPortfolioType_solving_default = {
    "cycles": 1,
    "T_cycle": _life_span,
    "constructors": HousingPortfolioType_constructors_default,
    # Preferences
    "CRRA": 5.0,
    "alpha": 0.2,
    "DiscFac": 0.96,
    "LivPrb": _surv_probs,
    "PermGroFac": _perm_gro_fac,
    "Rfree": [1.02] * _life_span,
    "BoroCnstArt": 0.0,
    # Housing and collateral
    "LTV": 0.80,
    "MortRate": 0.04,
    "MaintRate": 0.02,
    "SellCost": 0.06,
    "DefaultPenalty": 0.5,
    "RentRate": 0.05,
    "EntryCost": 0.01,
    "BeqWt": 1.0,
    "AffordabilityLimit": np.inf,
    # Housing growth
    "HousingGroFac": 1.02,
    "HousingGrowthShkStd": 0.0,
    "HousingGrowthShkCount": 1,
    # Origination constraints
    "MaxDTI": 4.0,
    # Correlations
    "StockIncCorr": 0.0,
    "HousingIncCorr": 0.0,
    "HousingStockCorr": 0.0,
    # Transitory house price shock
    "HousePriceShkStd": 0.0,
    "HousePriceShkCount": 1,
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
HousingPortfolioType_default.update(HousingPortfolioType_hGrid_default)
HousingPortfolioType_default.update(HousingPortfolioType_IncShkDstn_default)
HousingPortfolioType_default.update(HousingPortfolioType_RiskyDstn_default)

init_housing_portfolio = HousingPortfolioType_default

# ---------------------------------------------------------------------------
# Markov defaults
# ---------------------------------------------------------------------------

# 4-state Markov: (E,B)=0, (E,S)=1, (U,B)=2, (U,S)=3
# Employment: p_u = 0.05, p_e = 0.50
# Regime: p_sb = 0.20, p_bs = 0.30
_Pi_e = np.array([[0.95, 0.05], [0.50, 0.50]])
_Pi_nu = np.array([[0.80, 0.20], [0.30, 0.70]])
_mrkv_default_4x4 = np.kron(_Pi_e, _Pi_nu)

MarkovHousingPortfolioType_constructors_default = {
    "IncShkDstn": construct_markov_income_process,
    "aXtraGrid": make_assets_grid,
    "RiskyDstn": make_lognormal_RiskyDstn,
    "ShareLimit": calc_ShareLimit_for_CRRA,
    "ShareGrid": make_simple_ShareGrid,
    "hGrid": make_h_grid,
    "HousePriceShkDstn": make_house_price_shock_dstn,
    "HousingGrowthShkDstn": make_housing_growth_shock_dstn,
    "solution_terminal": make_markov_housing_solution_terminal,
}

MarkovHousingPortfolioType_IncShkDstn_default = {
    "PermShkStd": [0.1] * _life_span,
    "PermShkCount": 5,
    "TranShkStd": [0.1] * _life_span,
    "TranShkCount": 5,
    "UnempIns": 0.3,
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
    "LTV": 0.80,
    "MortRate": 0.04,
    "MaintRate": 0.02,
    "SellCost": 0.06,
    "DefaultPenalty": 0.5,
    "RentRate": 0.05,
    "EntryCost": 0.01,
    "BeqWt": 1.0,
    "AffordabilityLimit": np.inf,
    "MaxDTI": 4.0,
    "StockIncCorr": 0.0,
    "HousingIncCorr": 0.0,
    "HousingStockCorr": 0.0,
    "MrkvArray": [_mrkv_default_4x4] * _life_span,
    "HousingGroFac": [1.03, 1.00],  # [boom, slump]
    "HousingGrowthShkStd": 0.0,
    "HousingGrowthShkCount": 1,
    "HousePriceShkStd": 0.0,
    "HousePriceShkCount": 1,
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
MarkovHousingPortfolioType_default.update(HousingPortfolioType_hGrid_default)
MarkovHousingPortfolioType_default.update(
    MarkovHousingPortfolioType_IncShkDstn_default
)
MarkovHousingPortfolioType_default.update(HousingPortfolioType_RiskyDstn_default)

init_markov_housing_portfolio = MarkovHousingPortfolioType_default


# ---------------------------------------------------------------------------
# Agent type
# ---------------------------------------------------------------------------


class HousingPortfolioConsumerType(IndShockConsumerType):
    """Life-cycle consumer with housing, collateralized borrowing, and portfolio choice.

    State variables:
        m_t = M_t/P_t : normalized cash-on-hand
        h_t = H_t/P_t : housing-to-income ratio

    Choice variables:
        c_t : consumption (normalized)
        varsigma_t : risky portfolio share
        tenure : own / sell / move / default (discrete)
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
        "track_vars": ["mNrm", "cNrm", "Share", "hNrm", "tenure"],
    }

    time_inv_ = [
        "CRRA",
        "DiscFac",
        "BoroCnstArt",
        "aXtraGrid",
        "vFuncBool",
        "CubicBool",
        "alpha",
        "LTV",
        "MortRate",
        "MaintRate",
        "SellCost",
        "DefaultPenalty",
        "RentRate",
        "EntryCost",
        "BeqWt",
        "AffordabilityLimit",
        "MaxDTI",
        "StockIncCorr",
        "HousingIncCorr",
        "HousingStockCorr",
        "ShareGrid",
        "hGrid",
        "ShareLimit",
        "IndepDstnBool",
        "RiskyDstn",
        "ShockDstn",
        "HousePriceShkDstn",
        "HousingGrowthShkDstn",
        "HousingGroFac",
    ]

    time_vary_ = [
        "LivPrb",
        "PermGroFac",
        "Rfree",
        "IncShkDstn",
        "PermShkDstn",
        "TranShkDstn",
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
        self.construct("hGrid")
        self.construct("HousePriceShkDstn")
        self.construct("HousingGrowthShkDstn")
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
        if not (0.0 < self.LTV < 1.0):
            raise ValueError(f"LTV must be in (0, 1), got {self.LTV}")
        if self.RentRate <= 0:
            raise ValueError(f"RentRate must be positive: {self.RentRate}")
        if self.MortRate < 0:
            raise ValueError(f"MortRate must be non-negative: {self.MortRate}")
        if not (0.0 <= self.SellCost <= 1.0):
            raise ValueError(
                f"SellCost must be in [0, 1], got {self.SellCost}."
            )
        if hasattr(self, 'hMin') and self.hMin <= 0:
            raise ValueError(f"hMin must be positive: {self.hMin}")


class MarkovHousingPortfolioConsumerType(HousingPortfolioConsumerType):
    """Housing portfolio model with persistent Markov states (e, nu).

    Extends the baseline model with a 4-state Markov chain:
    (E,B)=0, (E,S)=1, (U,B)=2, (U,S)=3.

    Employment states {E,U} govern income distributions.
    Regime states {B,S} govern housing growth rates.

    Additional parameters
    ---------------------
    MrkvArray : list of np.ndarray
        4x4 transition matrix per period. Can be built from
        MrkvArrayEmploy (2x2) and MrkvArrayRegime (2x2) via Kronecker product.
    HousingGroFac : list of float
        [GH_boom, GH_slump] regime-dependent housing growth rates.
    """

    IncShkDstn_default = MarkovHousingPortfolioType_IncShkDstn_default
    solving_default = MarkovHousingPortfolioType_solving_default

    default_ = {
        "params": MarkovHousingPortfolioType_default,
        "solver": solve_one_period_HousingPortfolioMarkov,
        "track_vars": ["mNrm", "cNrm", "Share", "hNrm", "tenure", "Mrkv"],
    }

    time_inv_ = [
        "CRRA",
        "DiscFac",
        "BoroCnstArt",
        "aXtraGrid",
        "vFuncBool",
        "CubicBool",
        "alpha",
        "LTV",
        "MortRate",
        "MaintRate",
        "SellCost",
        "DefaultPenalty",
        "RentRate",
        "EntryCost",
        "BeqWt",
        "ShareGrid",
        "hGrid",
        "ShareLimit",
        "IndepDstnBool",
        "RiskyDstn",
        "AffordabilityLimit",
        "MaxDTI",
        "StockIncCorr",
        "HousingIncCorr",
        "HousingStockCorr",
        "HousePriceShkDstn",
        "HousingGrowthShkDstn",
        "HousingGroFac",
    ]

    time_vary_ = [
        "LivPrb",
        "PermGroFac",
        "Rfree",
        "IncShkDstn",
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
        # Build 4x4 MrkvArray from 2x2 components if provided
        if hasattr(self, 'MrkvArrayEmploy') and hasattr(self, 'MrkvArrayRegime'):
            Pi_e = np.asarray(self.MrkvArrayEmploy)
            Pi_nu = np.asarray(self.MrkvArrayRegime)
            combined = np.kron(Pi_e, Pi_nu)
            self.MrkvArray = [combined] * self.T_cycle
        self.construct("IncShkDstn")
        self.construct("aXtraGrid")
        self.construct("RiskyDstn")
        self.construct("ShareLimit")
        self.construct("ShareGrid")
        self.construct("hGrid")
        self.construct("HousePriceShkDstn")
        self.construct("HousingGrowthShkDstn")
        self.construct("solution_terminal")

    def check_restrictions(self):
        """Check model parameter restrictions including Markov-specific ones."""
        super().check_restrictions()
        matrices = (
            self.MrkvArray
            if isinstance(self.MrkvArray, list)
            else [self.MrkvArray]
        )
        for t, mat in enumerate(matrices):
            mat = np.asarray(mat)
            if mat.shape[0] != mat.shape[1]:
                raise ValueError(
                    f"MrkvArray[{t}] must be square, got shape {mat.shape}."
                )
            row_sums = mat.sum(axis=1)
            if not np.allclose(row_sums, 1.0):
                raise ValueError(
                    f"MrkvArray[{t}] rows must sum to 1, got {row_sums}."
                )

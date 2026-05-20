"""
JAX-accelerated solver for ``AggShockMarkovConsumerType`` — consumers facing
idiosyncratic and aggregate shocks plus a discrete Markov macro state.

This is the JAX counterpart to ``solve_ConsAggMarkov`` in
``HARK.ConsumptionSaving.ConsAggShockModel``. It follows the same drop-in
opt-in pattern as ``ConsIndShockModelJAX`` (PR-2).

JAX is an optional dependency. Importing this module without JAX installed
raises ``ImportError`` with installation instructions.

Scope of the JAX kernel
-----------------------
The two outer loops (per next-state ``j``, per current-state ``i``) stay in
plain Python/numpy. Each iteration's "expectation over shocks" is what
benefits most from JAX vectorization, so that step is JIT-compiled and
called from inside the outer loops. The cFunc and vPfunc objects returned
are built using HARK's existing numpy interpolator classes, preserving
compatibility with downstream code that walks ``solution.cFunc.functions``.

This matches the architectural pattern of ``ConsIndShockModelFast``
(numba) and of HAFiscal's existing JAX kernel for ``solve_agg_cons_markov_alt``.
"""

from __future__ import annotations

try:
    import jax
    import jax.numpy as jnp
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "HARK.ConsumptionSaving.ConsAggShockModelJAX requires JAX. "
        "Install with: pip install jax  (or 'jax[cuda12]' for GPU)"
    ) from _e

import numpy as np

from HARK.ConsumptionSaving.ConsAggShockModel import AggShockMarkovConsumerType
from HARK.interpolation import (
    BilinearInterp,
    ConstantFunction,
    LinearInterp,
    LinearInterpOnInterp1D,
    LowerEnvelope2D,
    MargValueFuncCRRA,
    UpperEnvelope,
    VariableLowerBoundFunc2D,
)
from HARK.interpolation_jax import bilinear_interp

__all__ = [
    "solve_ConsAggMarkov_jax",
    "AggShockMarkovConsumerTypeJAX",
]


# ============================================================
# Helpers — lift HARK 2D vPfuncNext to a JAX-evaluatable table
# ============================================================

_DEFAULT_M_LIFT = 512  # grid resolution for normalized m in lift
_DEFAULT_AGG_LIFT = 128  # grid resolution for aggregate M in lift


def _lift_vpfunc2d(
    vpfunc, m_min, m_max, M_min, M_max, n_m=_DEFAULT_M_LIFT, n_M=_DEFAULT_AGG_LIFT
):
    """Tabulate a 2-D vPfunc on a uniform grid for JAX bilinear evaluation.

    Returns ``(m_grid, M_grid, table)`` where
    ``table[i, k] = vpfunc(m_grid[i], M_grid[k])``.
    """
    m_grid = np.linspace(m_min, m_max, n_m)
    M_grid = np.linspace(M_min, M_max, n_M)
    Mg, mg = np.meshgrid(M_grid, m_grid)  # mg, Mg shape (n_m, n_M)
    table = np.asarray(vpfunc(mg, Mg))
    return m_grid, M_grid, table


# ============================================================
# Inner EGM kernel — JIT-ed expectation over shocks
# ============================================================


@jax.jit
def _expected_vp_next_2d(
    a_grid_2d,  # (Mcount, aCount) per-M end-of-period asset grid
    PermShks,
    TranShks,  # (ShkCount,) idiosyncratic shock atoms
    PermShkAgg,
    TranShkAgg,  # (ShkCount,) aggregate shock atoms
    pmv,  # (ShkCount,) probabilities
    PermGroFac_total,  # scalar — combined PermGroFac × PermGroFacAgg[j]
    LivPrb,  # scalar
    CRRA,  # scalar
    # next-period bilinear table for vPfuncNext[j]
    vp_m_grid,  # (n_m,) lift grid in m
    vp_M_grid,  # (n_M,) lift grid in M
    vp_table,  # (n_m, n_M) tabulated vPfuncNext[j]
    # Rfree / wage values for next period at each (Mcount, ShkCount)
    Reff_array,  # (Mcount, ShkCount)
    wEff_array,  # (Mcount, ShkCount)
    Mnext_array,  # (Mcount, ShkCount)
):
    """JIT-ed inner EGM: expected marginal value at end of this period.

    All Mcount × aCount × ShkCount work happens in a single broadcast +
    bilinear lookup, then summed over the shock axis.

    Returns ``EndOfPrdvP[Mcount, aCount]`` (NOT yet multiplied by DiscFac;
    the LivPrb factor IS included).
    """
    # Broadcast everything to (Mcount, aCount, ShkCount)
    a_b = a_grid_2d[:, :, None]  # (Mcount, aCount, 1)
    Reff_b = Reff_array[:, None, :]  # (Mcount, 1, ShkCount)
    wEff_b = wEff_array[:, None, :]
    Mnext_b = Mnext_array[:, None, :]
    Perm_b = PermShks[None, None, :]
    Tran_b = TranShks[None, None, :]
    PermAgg_b = PermShkAgg[None, None, :]
    pmv_b = pmv[None, None, :]

    PermShkTotal_b = PermGroFac_total * Perm_b * PermAgg_b  # (1, 1, ShkCount)
    mNext = (
        Reff_b * a_b / PermShkTotal_b + Tran_b * wEff_b
    )  # (Mcount, aCount, ShkCount)

    # vPnext via bilinear into the lifted table
    Mnext_full = jnp.broadcast_to(Mnext_b, mNext.shape)
    vp_next = bilinear_interp(
        vp_table,
        vp_m_grid,
        vp_M_grid,
        mNext.reshape(-1),
        Mnext_full.reshape(-1),
    ).reshape(mNext.shape)

    vP_factor = Reff_b * (PermShkTotal_b ** (-CRRA))
    EndOfPrdvP = LivPrb * jnp.sum(pmv_b * vP_factor * vp_next, axis=2)
    return EndOfPrdvP


# ============================================================
# Public solver (drop-in replacement for solve_ConsAggMarkov)
# ============================================================


def solve_ConsAggMarkov_jax(
    solution_next,
    IncShkDstn,
    LivPrb,
    DiscFac,
    CRRA,
    MrkvArray,
    PermGroFac,
    PermGroFacAgg,
    aXtraGrid,
    BoroCnstArt,
    Mgrid,
    AFunc,
    Rfunc,
    wFunc,
):
    """JAX-accelerated counterpart to ``solve_ConsAggMarkov``.

    The expectation-over-shocks step is JIT-compiled (see ``_expected_vp_next``);
    the outer loops over current/next Markov states stay in plain Python and
    use HARK's existing numpy interpolator classes for cFunc/vPfunc storage,
    matching the original solver's return shape exactly.

    Function signature matches ``solve_ConsAggMarkov`` so it can be wired
    directly into ``AggShockMarkovConsumerType.default_["solver"]``.

    See ``solve_ConsAggMarkov`` (in ``ConsAggShockModel.py``) for the math
    and the meanings of each argument.
    """
    aCount = aXtraGrid.size
    Mcount = Mgrid.size
    StateCount = MrkvArray.shape[0]
    DiscFacEff = DiscFac * LivPrb

    # ----- Pass 1: per-next-state EndOfPrdvPfunc_cond + BoroCnstNat_cond ---
    EndOfPrdvPfunc_cond = []
    BoroCnstNat_cond = []
    for j in range(StateCount):
        vPfuncNext_j = solution_next.vPfunc[j]
        mNrmMinNext_j = solution_next.mNrmMin[j]

        PermShks = np.asarray(IncShkDstn[j].atoms[0], dtype=np.float64)
        TranShks = np.asarray(IncShkDstn[j].atoms[1], dtype=np.float64)
        PermShkAgg = np.asarray(IncShkDstn[j].atoms[2], dtype=np.float64)
        TranShkAgg = np.asarray(IncShkDstn[j].atoms[3], dtype=np.float64)
        pmv = np.asarray(IncShkDstn[j].pmv, dtype=np.float64)
        ShkCount = pmv.size

        # Aggregate state grid for this next-state
        AaggGrid = AFunc[j](Mgrid)

        # k, R, w at next period, shape (Mcount, ShkCount)
        AaggNow_2d = AaggGrid[:, None]  # (Mcount, 1)
        PermAgg_2d = PermShkAgg[None, :]  # (1, ShkCount)
        TranAgg_2d = TranShkAgg[None, :]
        kNext = AaggNow_2d / (PermGroFacAgg[j] * PermAgg_2d)
        kNextEff = kNext / TranAgg_2d
        R_2d = Rfunc(kNextEff)
        Reff_2d = R_2d / LivPrb
        wEff_2d = wFunc(kNextEff) * TranAgg_2d
        Mnext_2d = kNext * R_2d + wEff_2d

        # Natural borrowing constraint per Mcount
        PermShkTotal_2d = PermGroFac * PermGroFacAgg[j] * PermAgg_2d * PermShks[None, :]
        TranShkVals_2d = TranShks[None, :]
        # Use the same convention as solve_ConsAggMarkov: pick worst shock combo by inner max
        aNrmMin_candidates = (
            PermShkTotal_2d
            / Reff_2d
            * (mNrmMinNext_j(Mnext_2d) - wEff_2d * TranShkVals_2d)
        )
        BoroCnstNat_vec = np.max(aNrmMin_candidates, axis=1)

        # Lift vPfuncNext_j to a JAX-evaluatable bilinear table.
        #
        # Range determination is subtle because vPfuncNext returns NaN for
        # m < mNrmMinNext(M), and bilinear_interp at any query inside a cell
        # that touches a NaN corner returns NaN (since 0 * NaN = NaN under
        # IEEE FP). So m_lift_min must sit at or above where the value
        # function is finite over the entire M-lift range — otherwise the
        # bottom row of the table is NaN and it poisons queries in the
        # adjacent row.
        #
        # Compute the actual m_next query range by running the same formula
        # the kernel uses, evaluated on a tiled grid. This avoids the overly-
        # conservative all-corners bound used in earlier drafts.
        aXtra_arr_for_range = np.asarray(aXtraGrid, dtype=np.float64)
        a_tiled = (
            BoroCnstNat_vec[:, None, None] + aXtra_arr_for_range[None, :, None]
        )  # (Mcount, aCount, 1)
        PermShkTotal_3d = PermShkTotal_2d[:, None, :]  # (Mcount, 1, ShkCount)
        TranShk_3d = TranShks[None, None, :]
        wEff_3d = wEff_2d[:, None, :]
        Reff_3d = Reff_2d[:, None, :]
        mNext_full = Reff_3d * a_tiled / PermShkTotal_3d + TranShk_3d * wEff_3d
        m_query_min = float(mNext_full.min())
        m_query_max = float(mNext_full.max())
        M_lift_min = float(np.min(Mnext_2d)) - 0.1
        M_lift_max = float(np.max(Mnext_2d)) + 0.1

        # Clamp m_lift_min to be at or above the maximum of mNrmMinNext_j(M)
        # over the M range — guarantees no NaN cells in the table.
        M_probe = np.linspace(M_lift_min, M_lift_max, 32)
        mNrmMin_probe = mNrmMinNext_j(M_probe)
        m_lift_min_safe = float(np.max(mNrmMin_probe)) + 1e-6
        m_lift_min = max(m_query_min - 0.05, m_lift_min_safe)
        m_lift_max = m_query_max + 1.0

        vp_m_grid, vp_M_grid, vp_table = _lift_vpfunc2d(
            vPfuncNext_j,
            m_lift_min,
            m_lift_max,
            M_lift_min,
            M_lift_max,
        )

        # End-of-period asset grid (per-Mcount, shifted by natural borrowing
        # constraint). Shape (Mcount, aCount). Passed to the JAX kernel as a
        # 2-D array so it can compute the expectation in one vectorized call.
        aXtra_arr = np.asarray(aXtraGrid, dtype=np.float64)
        a_now_2d = BoroCnstNat_vec[:, None] + aXtra_arr[None, :]  # (Mcount, aCount)

        EndOfPrdvP_no_disc = np.asarray(
            _expected_vp_next_2d(
                jnp.asarray(a_now_2d),
                jnp.asarray(PermShks),
                jnp.asarray(TranShks),
                jnp.asarray(PermShkAgg),
                jnp.asarray(TranShkAgg),
                jnp.asarray(pmv),
                float(PermGroFac * PermGroFacAgg[j]),
                float(LivPrb),
                float(CRRA),
                jnp.asarray(vp_m_grid),
                jnp.asarray(vp_M_grid),
                jnp.asarray(vp_table),
                jnp.asarray(Reff_2d),
                jnp.asarray(wEff_2d),
                jnp.asarray(Mnext_2d),
            )
        )
        EndOfPrdvP = (
            DiscFac * EndOfPrdvP_no_disc
        )  # absorb DiscFac (LivPrb already in kernel)

        # Build conditional EndOfPrdvPfunc as in original: pseudo-inverse, bilinear over (aXtra, AaggGrid)
        BoroCnstNat = LinearInterp(
            np.insert(AaggGrid, 0, 0.0),
            np.insert(BoroCnstNat_vec, 0, 0.0),
        )
        EndOfPrdvPnvrs = np.concatenate(
            (np.zeros((Mcount, 1)), EndOfPrdvP ** (-1.0 / CRRA)),
            axis=1,
        )
        EndOfPrdvPnvrsFunc_base = BilinearInterp(
            np.transpose(EndOfPrdvPnvrs),
            np.insert(np.asarray(aXtraGrid), 0, 0.0),
            AaggGrid,
        )
        EndOfPrdvPnvrsFunc = VariableLowerBoundFunc2D(
            EndOfPrdvPnvrsFunc_base,
            BoroCnstNat,
        )
        EndOfPrdvPfunc_cond.append(MargValueFuncCRRA(EndOfPrdvPnvrsFunc, CRRA))
        BoroCnstNat_cond.append(BoroCnstNat)

    # ----- Pass 2: per-current-state aggregation + cFunc construction -----
    aXtra_tiled = np.tile(np.asarray(aXtraGrid).reshape(1, aCount), (Mcount, 1))
    cFuncCnst = BilinearInterp(
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array([BoroCnstArt, BoroCnstArt + 1.0]),
        np.array([0.0, 1.0]),
    )

    cFuncNow = []
    vPfuncNow = []
    mNrmMinNow = []
    for i in range(StateCount):
        AaggNow = AFunc[i](Mgrid)
        aNrmMin_candidates = np.full((StateCount, Mcount), np.nan)
        for j in range(StateCount):
            if MrkvArray[i, j] > 0.0:
                aNrmMin_candidates[j, :] = BoroCnstNat_cond[j](AaggNow)
        aNrmMin_vec = np.nanmax(aNrmMin_candidates, axis=0)
        BoroCnstNat_vec = aNrmMin_vec

        aNrmNow_tiled = aNrmMin_vec[:, None] + aXtra_tiled
        AaggNow_tiled = np.tile(AaggNow.reshape(Mcount, 1), (1, aCount))

        EndOfPrdvP = np.zeros((Mcount, aCount))
        for j in range(StateCount):
            if MrkvArray[i, j] > 0.0:
                EndOfPrdvP += MrkvArray[i, j] * EndOfPrdvPfunc_cond[j](
                    aNrmNow_tiled,
                    AaggNow_tiled,
                )

        cNrmNow = EndOfPrdvP ** (-1.0 / CRRA)
        mNrmNow = aNrmNow_tiled + cNrmNow

        cFuncBaseByM = []
        for n in range(Mcount):
            c_temp = np.insert(cNrmNow[n, :], 0, 0.0)
            m_temp = np.insert(mNrmNow[n, :] - BoroCnstNat_vec[n], 0, 0.0)
            cFuncBaseByM.append(LinearInterp(m_temp, c_temp))

        BoroCnstNat_func = LinearInterp(
            np.insert(Mgrid, 0, 0.0),
            np.insert(BoroCnstNat_vec, 0, 0.0),
        )
        cFuncBase = LinearInterpOnInterp1D(cFuncBaseByM, Mgrid)
        cFuncUnc = VariableLowerBoundFunc2D(cFuncBase, BoroCnstNat_func)
        cFuncNow.append(LowerEnvelope2D(cFuncUnc, cFuncCnst))
        mNrmMinNow.append(
            UpperEnvelope(BoroCnstNat_func, ConstantFunction(BoroCnstArt))
        )
        vPfuncNow.append(MargValueFuncCRRA(cFuncNow[-1], CRRA))

    # ConsumerSolution import is inside ConsIndShockModel — re-use it
    from HARK.ConsumptionSaving.ConsIndShockModel import ConsumerSolution

    return ConsumerSolution(cFunc=cFuncNow, vPfunc=vPfuncNow, mNrmMin=mNrmMinNow)


# ============================================================
# Wrapper class — drop-in subclass of AggShockMarkovConsumerType
# ============================================================


class AggShockMarkovConsumerTypeJAX(AggShockMarkovConsumerType):
    """
    Drop-in JAX-accelerated subclass of ``AggShockMarkovConsumerType``.

    Identical API; ``solve()`` uses ``solve_ConsAggMarkov_jax`` instead of
    ``solve_ConsAggMarkov``.

    All caveats of the parent class apply (this type cannot be solved
    standalone; it requires a ``Market`` association — usually
    ``CobbDouglasMarkovEconomy``).
    """

    default_ = {
        **AggShockMarkovConsumerType.default_,
        "solver": solve_ConsAggMarkov_jax,
    }

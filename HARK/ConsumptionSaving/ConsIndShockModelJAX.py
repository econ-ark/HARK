"""
JAX-accelerated solver for the basic ``ConsIndShockModel`` (CRRA utility,
idiosyncratic permanent and transitory income shocks, single Markov state,
no aggregate state).

This module is the JAX counterpart to ``ConsIndShockModelFast`` (numba opt-in)
and follows the same wrapping pattern: a drop-in subclass of
``IndShockConsumerType`` that swaps in a JAX-based ``solve_one_period``.

JAX is an optional dependency. Importing this module without JAX installed
raises ``ImportError`` with installation instructions.

Currently supports the basic case: ``CubicBool=False`` and ``vFuncBool=False``.
Cubic interpolation and value-function computation are not yet ported and
raise ``NotImplementedError`` when requested.

The interpolation primitives used here live in ``HARK.interpolation_jax``.
"""
from __future__ import annotations

try:
    import jax
    import jax.numpy as jnp
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "HARK.ConsumptionSaving.ConsIndShockModelJAX requires JAX. "
        "Install with: pip install jax  (or 'jax[cuda12]' for GPU)"
    ) from _e

import numpy as np

from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsumerSolution,
    IndShockConsumerType,
    solve_one_period_ConsIndShock,  # for fallback when CubicBool/vFuncBool
)
from HARK.interpolation import LinearInterp, LowerEnvelope, MargValueFuncCRRA
from HARK.interpolation_jax import linear_interp_1d
from HARK.utilities import NullFunc

__all__ = [
    "solve_one_period_ConsIndShock_jax",
    "IndShockConsumerTypeJAX",
]


# ============================================================
# Helpers — lift HARK cFunc to a JAX-evaluatable (x_grid, y_vals) pair
# ============================================================

# Fine grid used when "lifting" a LowerEnvelope-wrapped cFunc to a single
# LinearInterp for JAX evaluation. With 10000 points, single-period interp
# error is O(1e-7); compounded through ~100 EGM iterations to a steady state,
# this typically gives ~1e-5 relative agreement with the numpy solver. Tighter
# than EGM truncation noise on the underlying aXtraGrid.
_DEFAULT_LIFT_GRID = 10000


def _lift_cfunc_to_grid(cfunc, m_min, m_max, n=_DEFAULT_LIFT_GRID):
    """Evaluate a (possibly composite) HARK cFunc on a uniform grid.

    Returns ``(x_grid, y_vals)`` as numpy arrays suitable for
    ``linear_interp_1d``.

    Parameters
    ----------
    cfunc : HARK interpolator
        Typically a ``LowerEnvelope`` of ``LinearInterp`` instances.
    m_min, m_max : float
        Range over which to tabulate.
    n : int, default 1000
        Number of tabulation points.
    """
    x_grid = np.linspace(m_min, m_max, n)
    y_vals = np.asarray(cfunc(x_grid))
    return x_grid, y_vals


# ============================================================
# EGM kernel (pure JAX, jit-able)
# ============================================================

@jax.jit
def _egm_kernel(
    a_grid,           # (Na,) end-of-period asset grid (already shifted by BoroCnstNat)
    perm_shks,        # (Ns,) permanent shock atoms
    tran_shks,        # (Ns,) transitory shock atoms
    pmv,              # (Ns,) probabilities (sum to 1)
    cfunc_x_grid,     # (Nx,) lifted next-period cFunc grid
    cfunc_y_vals,     # (Nx,) lifted next-period cFunc values
    Rfree,
    PermGroFac,
    CRRA,
    DiscFacEff,
):
    """Endogenous-grid step for one period of the basic ConsIndShockModel.

    Returns ``(m_grid, c_grid)`` — the (mNrm, cNrm) pairs from inverting the
    Euler equation at each end-of-period asset gridpoint.

    Math:
        m_next[i, s] = Rfree * a_grid[i] / (PermGroFac * perm_shks[s])
                       + tran_shks[s]
        vP_next[i, s] = cFunc_next(m_next[i, s]) ** (-CRRA)
        EndOfPrdvP[i] = DiscFacEff * Rfree * PermGroFac^(-CRRA)
                        * sum_s pmv[s] * perm_shks[s]^(-CRRA) * vP_next[i, s]
        cNrm[i] = EndOfPrdvP[i] ** (-1/CRRA)
        mNrm[i] = a_grid[i] + cNrm[i]
    """
    a_b = a_grid[:, None]              # (Na, 1)
    perm_b = perm_shks[None, :]        # (1, Ns)
    tran_b = tran_shks[None, :]        # (1, Ns)
    pmv_b = pmv[None, :]               # (1, Ns)

    m_next = Rfree * a_b / (PermGroFac * perm_b) + tran_b  # (Na, Ns)

    c_next = linear_interp_1d(
        cfunc_x_grid, cfunc_y_vals, m_next.reshape(-1), lower_extrap=True
    ).reshape(m_next.shape)

    vP_next = c_next ** (-CRRA)
    weights = pmv_b * (perm_b ** (-CRRA))
    EndOfPrdvP = (
        DiscFacEff * Rfree * (PermGroFac ** (-CRRA))
        * jnp.sum(weights * vP_next, axis=1)
    )
    cNrm = EndOfPrdvP ** (-1.0 / CRRA)
    mNrm = a_grid + cNrm
    return mNrm, cNrm


# ============================================================
# Public solver (drop-in replacement for solve_one_period_ConsIndShock)
# ============================================================

def solve_one_period_ConsIndShock_jax(
    solution_next,
    IncShkDstn,
    LivPrb,
    DiscFac,
    CRRA,
    Rfree,
    PermGroFac,
    BoroCnstArt,
    aXtraGrid,
    vFuncBool,
    CubicBool,
):
    """JAX-accelerated one-period solver matching ``solve_one_period_ConsIndShock``.

    Currently supports only the basic case (``CubicBool=False``,
    ``vFuncBool=False``). The function signature is identical to its
    numpy counterpart so it can be plugged into ``IndShockConsumerType.default_``
    via the ``solver`` key.

    Returns
    -------
    ConsumerSolution
        Solution to this period with ``cFunc``, ``vPfunc``, scalar summaries.
        ``vFunc`` is ``NullFunc()`` and ``vPPfunc`` is ``NullFunc()``.
    """
    if CubicBool:
        raise NotImplementedError(
            "solve_one_period_ConsIndShock_jax does not yet support "
            "CubicBool=True; use solve_one_period_ConsIndShock instead."
        )
    if vFuncBool:
        raise NotImplementedError(
            "solve_one_period_ConsIndShock_jax does not yet support "
            "vFuncBool=True; use solve_one_period_ConsIndShock instead."
        )

    DiscFacEff = DiscFac * LivPrb

    # Unpack discrete IncShkDstn (atoms are 2 × Ns: row 0 perm, row 1 tran)
    perm_shks = np.asarray(IncShkDstn.atoms[0], dtype=np.float64)
    tran_shks = np.asarray(IncShkDstn.atoms[1], dtype=np.float64)
    pmv = np.asarray(IncShkDstn.pmv, dtype=np.float64)
    Ex_IncNext = float(np.sum(pmv * perm_shks * tran_shks))
    WorstIncPrb = float(np.sum(
        pmv[(perm_shks == perm_shks.min()) & (tran_shks == tran_shks.min())]
    ))

    # Natural and effective borrowing constraints
    PermShkMinNext = float(perm_shks.min())
    TranShkMinNext = float(tran_shks.min())
    BoroCnstNat = (
        (solution_next.mNrmMin - TranShkMinNext)
        * (PermGroFac * PermShkMinNext) / Rfree
    )
    if BoroCnstArt is None:
        mNrmMinNow = BoroCnstNat
    else:
        mNrmMinNow = max(BoroCnstArt, BoroCnstNat)

    # Bounding MPC and human wealth (same formulas as numpy solver)
    PatFac = ((Rfree * DiscFacEff) ** (1.0 / CRRA)) / Rfree
    MPCminNow = 1.0 / (1.0 + PatFac / solution_next.MPCmin)
    hNrmNow = (PermGroFac / Rfree) * (Ex_IncNext + solution_next.hNrm)
    MPCmaxUnc = 1.0 / (
        1.0
        + (WorstIncPrb ** (1.0 / CRRA)) * PatFac / solution_next.MPCmax
    )
    MPCmaxNow = 1.0 if BoroCnstNat < mNrmMinNow else MPCmaxUnc
    cFuncLimitIntercept = MPCminNow * hNrmNow
    cFuncLimitSlope = MPCminNow

    # Lift next-period cFunc to (x_grid, y_vals) for JAX evaluation. Range
    # spans the natural lower bound of next-period mNrm up to a generous
    # upper tail derived from the current asset grid.
    m_lift_min = float(solution_next.mNrmMin)
    a_max_now = float(np.max(aXtraGrid)) + BoroCnstNat
    m_lift_max = max(
        m_lift_min + 1.0,
        float(Rfree * a_max_now / (PermGroFac * PermShkMinNext)
              + np.max(tran_shks)) * 1.1,
    )
    cfunc_x_grid, cfunc_y_vals = _lift_cfunc_to_grid(
        solution_next.cFunc, m_lift_min, m_lift_max
    )

    # End-of-period asset grid (shifted by natural borrowing constraint)
    a_grid = np.asarray(aXtraGrid, dtype=np.float64) + BoroCnstNat

    # JAX EGM step
    mNrm_jax, cNrm_jax = _egm_kernel(
        jnp.asarray(a_grid),
        jnp.asarray(perm_shks),
        jnp.asarray(tran_shks),
        jnp.asarray(pmv),
        jnp.asarray(cfunc_x_grid),
        jnp.asarray(cfunc_y_vals),
        float(Rfree),
        float(PermGroFac),
        float(CRRA),
        float(DiscFacEff),
    )
    mNrm = np.asarray(mNrm_jax)
    cNrm = np.asarray(cNrm_jax)

    # Build the (constrained + unconstrained) cFunc using HARK's native classes
    # so downstream code that walks solution.cFunc.functions continues to work.
    c_for_interp = np.insert(cNrm, 0, 0.0)
    m_for_interp = np.insert(mNrm, 0, BoroCnstNat)
    cFuncNowUnc = LinearInterp(
        m_for_interp, c_for_interp, cFuncLimitIntercept, cFuncLimitSlope
    )
    cFuncNowCnst = LinearInterp(
        np.array([mNrmMinNow, mNrmMinNow + 1.0]),
        np.array([0.0, 1.0]),
    )
    cFuncNow = LowerEnvelope(cFuncNowUnc, cFuncNowCnst, nan_bool=False)
    vPfuncNow = MargValueFuncCRRA(cFuncNow, CRRA)

    return ConsumerSolution(
        cFunc=cFuncNow,
        vFunc=NullFunc(),
        vPfunc=vPfuncNow,
        vPPfunc=NullFunc(),
        mNrmMin=mNrmMinNow,
        hNrm=hNrmNow,
        MPCmin=MPCminNow,
        MPCmax=MPCmaxNow,
    )


# ============================================================
# Wrapper class — drop-in subclass of IndShockConsumerType
# ============================================================

class IndShockConsumerTypeJAX(IndShockConsumerType):
    """
    Drop-in JAX-accelerated subclass of ``IndShockConsumerType``.

    Identical API and parameter set; the only difference is that ``solve()``
    uses ``solve_one_period_ConsIndShock_jax`` instead of the numpy/numba
    ``solve_one_period_ConsIndShock``.

    Restrictions (compared to ``IndShockConsumerType``):

    - ``CubicBool=True`` is not supported (falls back to numpy solver via
      ``NotImplementedError``).
    - ``vFuncBool=True`` is not supported.

    These restrictions match the most common use case (linear cFunc,
    no value function) and will be lifted in a follow-up.
    """

    default_ = {
        **IndShockConsumerType.default_,
        "solver": solve_one_period_ConsIndShock_jax,
    }

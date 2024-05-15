"""
Classes to solve canonical consumption-saving models with idiosyncratic shocks
to income.  All models here assume CRRA utility with geometric discounting, no
bequest motive, and income shocks that are fully transitory or fully permanent.

It currently solves three types of models:
   1) A very basic "perfect foresight" consumption-savings model with no uncertainty.
   2) A consumption-savings model with risk over transitory and permanent income shocks.
   3) The model described in (2), with an interest rate for debt that differs
      from the interest rate for savings.

See NARK https://github.com/econ-ark/HARK/blob/master/Documentation/NARK/NARK.pdf for information on variable naming conventions.
See HARK documentation for mathematical descriptions of the models being solved.
"""

from copy import copy, deepcopy

import numpy as np
from HARK.Calibration.Income.IncomeTools import (
    Cagetti_income,
    parse_income_spec,
    parse_time_params,
)
from HARK.Calibration.Income.IncomeProcesses import (
    construct_lognormal_income_process_unemployment,
    get_PermShkDstn_from_IncShkDstn,
    get_TranShkDstn_from_IncShkDstn,
)
from HARK.datasets.life_tables.us_ssa.SSATools import parse_ssa_life_table
from HARK.datasets.SCF.WealthIncomeDist.SCFDistTools import income_wealth_dists_from_scf
from HARK.distribution import (
    Lognormal,
    MeanOneLogNormal,
    Uniform,
    add_discrete_outcome_constant_mean,
    combine_indep_dstns,
    expected,
)
from HARK.interpolation import (
    CubicInterp,
    LinearInterp,
    LowerEnvelope,
    MargMargValueFuncCRRA,
    MargValueFuncCRRA,
    ValueFuncCRRA,
)
from HARK.metric import MetricObject
from HARK.rewards import (
    CRRAutility,
    CRRAutility_inv,
    CRRAutility_invP,
    CRRAutilityP,
    CRRAutilityP_inv,
    CRRAutilityP_invP,
    CRRAutilityPP,
    UtilityFuncCRRA,
)
from HARK.utilities import (
    construct_assets_grid,
    gen_tran_matrix_1D,
    gen_tran_matrix_2D,
    jump_to_grid_1D,
    jump_to_grid_2D,
    make_grid_exp_mult,
)
from scipy import sparse as sp
from scipy.optimize import newton

from HARK import (
    AgentType,
    NullFunc,
    _log,
    set_verbosity_level,
)

__all__ = [
    "ConsumerSolution",
    "PerfForesightConsumerType",
    "IndShockConsumerType",
    "KinkedRconsumerType",
    "init_perfect_foresight",
    "init_idiosyncratic_shocks",
    "init_kinked_R",
    "init_lifecycle",
    "init_cyclical",
]

utility = CRRAutility
utilityP = CRRAutilityP
utilityPP = CRRAutilityPP
utilityP_inv = CRRAutilityP_inv
utility_invP = CRRAutility_invP
utility_inv = CRRAutility_inv
utilityP_invP = CRRAutilityP_invP


# =====================================================================
# === Classes that help solve consumption-saving models ===
# =====================================================================


class ConsumerSolution(MetricObject):
    """
    A class representing the solution of a single period of a consumption-saving
    problem.  The solution must include a consumption function and marginal
    value function.

    Here and elsewhere in the code, Nrm indicates that variables are normalized
    by permanent income.

    Parameters
    ----------
    cFunc : function
        The consumption function for this period, defined over market
        resources: c = cFunc(m).
    vFunc : function
        The beginning-of-period value function for this period, defined over
        market resources: v = vFunc(m).
    vPfunc : function
        The beginning-of-period marginal value function for this period,
        defined over market resources: vP = vPfunc(m).
    vPPfunc : function
        The beginning-of-period marginal marginal value function for this
        period, defined over market resources: vPP = vPPfunc(m).
    mNrmMin : float
        The minimum allowable market resources for this period; the consump-
        tion function (etc) are undefined for m < mNrmMin.
    hNrm : float
        Human wealth after receiving income this period: PDV of all future
        income, ignoring mortality.
    MPCmin : float
        Infimum of the marginal propensity to consume this period.
        MPC --> MPCmin as m --> infinity.
    MPCmax : float
        Supremum of the marginal propensity to consume this period.
        MPC --> MPCmax as m --> mNrmMin.

    """

    distance_criteria = ["vPfunc"]

    def __init__(
        self,
        cFunc=None,
        vFunc=None,
        vPfunc=None,
        vPPfunc=None,
        mNrmMin=None,
        hNrm=None,
        MPCmin=None,
        MPCmax=None,
    ):
        # Change any missing function inputs to NullFunc
        self.cFunc = cFunc if cFunc is not None else NullFunc()
        self.vFunc = vFunc if vFunc is not None else NullFunc()
        self.vPfunc = vPfunc if vPfunc is not None else NullFunc()
        # vPFunc = NullFunc() if vPfunc is None else vPfunc
        self.vPPfunc = vPPfunc if vPPfunc is not None else NullFunc()
        self.mNrmMin = mNrmMin
        self.hNrm = hNrm
        self.MPCmin = MPCmin
        self.MPCmax = MPCmax

    def append_solution(self, new_solution):
        """
        Appends one solution to another to create a ConsumerSolution whose
        attributes are lists.  Used in ConsMarkovModel, where we append solutions
        *conditional* on a particular value of a Markov state to each other in
        order to get the entire solution.

        Parameters
        ----------
        new_solution : ConsumerSolution
            The solution to a consumption-saving problem; each attribute is a
            list representing state-conditional values or functions.

        Returns
        -------
        None
        """
        if type(self.cFunc) != list:
            # Then we assume that self is an empty initialized solution instance.
            # Begin by checking this is so.
            assert (
                NullFunc().distance(self.cFunc) == 0
            ), "append_solution called incorrectly!"

            # We will need the attributes of the solution instance to be lists.  Do that here.
            self.cFunc = [new_solution.cFunc]
            self.vFunc = [new_solution.vFunc]
            self.vPfunc = [new_solution.vPfunc]
            self.vPPfunc = [new_solution.vPPfunc]
            self.mNrmMin = [new_solution.mNrmMin]
        else:
            self.cFunc.append(new_solution.cFunc)
            self.vFunc.append(new_solution.vFunc)
            self.vPfunc.append(new_solution.vPfunc)
            self.vPPfunc.append(new_solution.vPPfunc)
            self.mNrmMin.append(new_solution.mNrmMin)


# =====================================================================
# === Classes and functions that solve consumption-saving models ===
# =====================================================================


def calc_human_wealth(h_nrm_next, perm_gro_fac, rfree, ex_inc_next):
    """Calculate human wealth this period given human wealth next period.

    Args:
        h_nrm_next (float): Normalized human wealth next period.
        perm_gro_fac (float): Permanent income growth factor.
        rfree (float): Risk free interest factor.
        ex_inc_next (float): Expected income next period.
    """
    return (perm_gro_fac / rfree) * (h_nrm_next + ex_inc_next)


def calc_patience_factor(rfree, disc_fac_eff, crra):
    """Calculate the patience factor for the agent.

    Args:
        rfree (float): Risk free interest factor.
        disc_fac_eff (float): Effective discount factor.
        crra (float): Coefficient of relative risk aversion.

    """
    return ((rfree * disc_fac_eff) ** (1.0 / crra)) / rfree


def calc_mpc_min(mpc_min_next, pat_fac):
    """Calculate the lower bound of the marginal propensity to consume.

    Args:
        mpc_min_next (float): Lower bound of the marginal propensity to
            consume next period.
        pat_fac (float): Patience factor.
    """
    return 1.0 / (1.0 + pat_fac / mpc_min_next)


def solve_one_period_ConsPF(
    solution_next,
    DiscFac,
    LivPrb,
    CRRA,
    Rfree,
    PermGroFac,
    BoroCnstArt,
    MaxKinks,
):
    """Solves one period of a basic perfect foresight consumption-saving model with
    a single risk free asset and permanent income growth.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one-period problem.
    DiscFac : float
        Intertemporal discount factor for future utility.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the next period.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree : float
        Risk free interest factor on end-of-period assets.
    PermGroFac : float
        Expected permanent income growth factor at the end of this period.
    BoroCnstArt : float or None
        Artificial borrowing constraint, as a multiple of permanent income.
        Can be None, indicating no artificial constraint.
    MaxKinks : int
        Maximum number of kink points to allow in the consumption function;
        additional points will be thrown out.  Only relevant in infinite
        horizon model with artificial borrowing constraint.

    Returns
    -------
    solution_now : ConsumerSolution
        Solution to the current period of a perfect foresight consumption-saving
        problem.

    """
    # Define the utility function and effective discount factor
    uFunc = UtilityFuncCRRA(CRRA)
    DiscFacEff = DiscFac * LivPrb  # Effective = pure x LivPrb

    # Prevent comparing None and float if there is no borrowing constraint
    # Can borrow as much as we want
    BoroCnstArt = -np.inf if BoroCnstArt is None else BoroCnstArt

    # Calculate human wealth this period
    hNrmNow = calc_human_wealth(solution_next.hNrm, PermGroFac, Rfree, 1.0)

    # Calculate the lower bound of the marginal propensity to consume
    PatFac = calc_patience_factor(Rfree, DiscFacEff, CRRA)
    MPCminNow = calc_mpc_min(solution_next.MPCmin, PatFac)

    # Extract the discrete kink points in next period's consumption function;
    # don't take the last one, as it only defines the extrapolation and is not a kink.
    mNrmNext = solution_next.cFunc.x_list[:-1]
    cNrmNext = solution_next.cFunc.y_list[:-1]
    vFuncNvrsNext = solution_next.vFunc.vFuncNvrs.y_list[:-1]
    EndOfPrdv = DiscFacEff * PermGroFac ** (1.0 - CRRA) * uFunc(vFuncNvrsNext)

    # Calculate the end-of-period asset values that would reach those kink points
    # next period, then invert the first order condition to get consumption. Then
    # find the endogenous gridpoint (kink point) today that corresponds to each kink
    aNrmNow = (PermGroFac / Rfree) * (mNrmNext - 1.0)
    cNrmNow = (DiscFacEff * Rfree) ** (-1.0 / CRRA) * (PermGroFac * cNrmNext)
    mNrmNow = aNrmNow + cNrmNow

    # Calculate (pseudo-inverse) value at each consumption kink point
    vNow = uFunc(cNrmNow) + EndOfPrdv
    vNvrsNow = uFunc.inverse(vNow)
    vNvrsSlopeMin = MPCminNow ** (-CRRA / (1.0 - CRRA))

    # Add an additional point to the list of gridpoints for the extrapolation,
    # using the new value of the lower bound of the MPC.
    mNrmNow = np.append(mNrmNow, mNrmNow[-1] + 1.0)
    cNrmNow = np.append(cNrmNow, cNrmNow[-1] + MPCminNow)
    vNvrsNow = np.append(vNvrsNow, vNvrsNow[-1] + vNvrsSlopeMin)

    # If the artificial borrowing constraint binds, combine the constrained and
    # unconstrained consumption functions.
    if BoroCnstArt > mNrmNow[0]:
        # Find the highest index where constraint binds
        cNrmCnst = mNrmNow - BoroCnstArt
        CnstBinds = cNrmCnst < cNrmNow
        idx = np.where(CnstBinds)[0][-1]

        if idx < (mNrmNow.size - 1):
            # If it is not the *very last* index, find the the critical level
            # of mNrm where the artificial borrowing contraint begins to bind.
            d0 = cNrmNow[idx] - cNrmCnst[idx]
            d1 = cNrmCnst[idx + 1] - cNrmNow[idx + 1]
            m0 = mNrmNow[idx]
            m1 = mNrmNow[idx + 1]
            alpha = d0 / (d0 + d1)
            mCrit = m0 + alpha * (m1 - m0)

            # Adjust the grids of mNrm and cNrm to account for the borrowing constraint.
            cCrit = mCrit - BoroCnstArt
            mNrmNow = np.concatenate(([BoroCnstArt, mCrit], mNrmNow[(idx + 1) :]))
            cNrmNow = np.concatenate(([0.0, cCrit], cNrmNow[(idx + 1) :]))

            # Adjust the vNvrs grid to account for the borrowing constraint
            v0 = vNvrsNow[idx]
            v1 = vNvrsNow[idx + 1]
            vNvrsCrit = v0 + alpha * (v1 - v0)
            vNvrsNow = np.concatenate(([0.0, vNvrsCrit], vNvrsNow[(idx + 1) :]))

        else:
            # If it *is* the very last index, then there are only three points
            # that characterize the consumption function: the artificial borrowing
            # constraint, the constraint kink, and the extrapolation point.
            mXtra = (cNrmNow[-1] - cNrmCnst[-1]) / (1.0 - MPCminNow)
            mCrit = mNrmNow[-1] + mXtra
            cCrit = mCrit - BoroCnstArt
            mNrmNow = np.array([BoroCnstArt, mCrit, mCrit + 1.0])
            cNrmNow = np.array([0.0, cCrit, cCrit + MPCminNow])

            # Adjust vNvrs grid for this three node structure
            mNextCrit = BoroCnstArt * Rfree + 1.0
            vNextCrit = PermGroFac ** (1.0 - CRRA) * solution_next.vFunc(mNextCrit)
            vCrit = uFunc(cCrit) + DiscFacEff * vNextCrit
            vNvrsCrit = uFunc.inverse(vCrit)
            vNvrsNow = np.array([0.0, vNvrsCrit, vNvrsCrit + vNvrsSlopeMin])

    # If the mNrm and cNrm grids have become too large, throw out the last
    # kink point, being sure to adjust the extrapolation.
    if mNrmNow.size > MaxKinks:
        mNrmNow = np.concatenate((mNrmNow[:-2], [mNrmNow[-3] + 1.0]))
        cNrmNow = np.concatenate((cNrmNow[:-2], [cNrmNow[-3] + MPCminNow]))
        vNvrsNow = np.concatenate((vNvrsNow[:-2], [vNvrsNow[-3] + vNvrsSlopeMin]))

    # Construct the consumption function as a linear interpolation.
    cFuncNow = LinearInterp(mNrmNow, cNrmNow)

    # Calculate the upper bound of the MPC as the slope of the bottom segment.
    MPCmaxNow = (cNrmNow[1] - cNrmNow[0]) / (mNrmNow[1] - mNrmNow[0])
    mNrmMinNow = mNrmNow[0]

    # Construct the (marginal) value function for this period
    # See the PerfForesightConsumerType.ipynb documentation notebook for the derivations
    vFuncNvrs = LinearInterp(mNrmNow, vNvrsNow)
    vFuncNow = ValueFuncCRRA(vFuncNvrs, CRRA)
    vPfuncNow = MargValueFuncCRRA(cFuncNow, CRRA)

    # Construct and return the solution
    solution_now = ConsumerSolution(
        cFunc=cFuncNow,
        vFunc=vFuncNow,
        vPfunc=vPfuncNow,
        mNrmMin=mNrmMinNow,
        hNrm=hNrmNow,
        MPCmin=MPCminNow,
        MPCmax=MPCmaxNow,
    )
    return solution_now


def calc_worst_inc_prob(inc_shk_dstn):
    """Calculate the probability of the worst income shock.

    Args:
        inc_shk_dstn (DiscreteDistribution): Distribution of shocks to income.
    """
    probs = inc_shk_dstn.pmv
    perm, tran = inc_shk_dstn.atoms
    income = perm * tran
    worst_inc = np.min(income)
    return np.sum(probs[income == worst_inc])


def calc_boro_const_nat(m_nrm_min_next, inc_shk_dstn, rfree, perm_gro_fac):
    """Calculate the natural borrowing constraint.

    Args:
        m_nrm_min_next (float): Minimum normalized market resources next period.
        inc_shk_dstn (DiscreteDstn): Distribution of shocks to income.
        rfree (float): Risk free interest factor.
        perm_gro_fac (float): Permanent income growth factor.
    """
    perm, tran = inc_shk_dstn.atoms
    temp_fac = (perm_gro_fac * np.min(perm)) / rfree
    return (m_nrm_min_next - np.min(tran)) * temp_fac


def calc_m_nrm_min(boro_const_art, boro_const_nat):
    """Calculate the minimum normalized market resources this period.

    Args:
        boro_const_art (float): Artificial borrowing constraint.
        boro_const_nat (float): Natural borrowing constraint.
    """
    return (
        boro_const_nat
        if boro_const_art is None
        else max(boro_const_nat, boro_const_art)
    )


def calc_mpc_max(
    mpc_max_next, worst_inc_prob, crra, pat_fac, boro_const_nat, boro_const_art
):
    """Calculate the upper bound of the marginal propensity to consume.

    Args:
        mpc_max_next (float): Upper bound of the marginal propensity to
            consume next period.
        worst_inc_prob (float): Probability of the worst income shock.
        crra (float): Coefficient of relative risk aversion.
        pat_fac (float): Patience factor.
        boro_const_nat (float): Natural borrowing constraint.
        boro_const_art (float): Artificial borrowing constraint.
    """
    temp_fac = (worst_inc_prob ** (1.0 / crra)) * pat_fac
    return 1.0 / (1.0 + temp_fac / mpc_max_next)


def calc_m_nrm_next(shock, a, rfree, perm_gro_fac):
    """Calculate normalized market resources next period.

    Args:
        shock (float): Realization of shocks to income.
        a (np.ndarray): Exogenous grid of end-of-period assets.
        rfree (float): Risk free interest factor.
        perm_gro_fac (float): Permanent income growth factor.
    """
    return rfree / (perm_gro_fac * shock["PermShk"]) * a + shock["TranShk"]


def calc_v_next(shock, a, rfree, crra, perm_gro_fac, vfunc_next):
    """Calculate continuation value function with respect to
    end-of-period assets.

    Args:
        shock (float): Realization of shocks to income.
        a (np.ndarray): Exogenous grid of end-of-period assets.
        rfree (float): Risk free interest factor.
        crra (float): Coefficient of relative risk aversion.
        perm_gro_fac (float): Permanent income growth factor.
        vfunc_next (Callable): Value function next period.
    """
    return (
        shock["PermShk"] ** (1.0 - crra) * perm_gro_fac ** (1.0 - crra)
    ) * vfunc_next(calc_m_nrm_next(shock, a, rfree, perm_gro_fac))


def calc_vp_next(shock, a, rfree, crra, perm_gro_fac, vp_func_next):
    """Calculate the continuation marginal value function with respect to
    end-of-period assets.

    Args:
        shock (float): Realization of shocks to income.
        a (np.ndarray): Exogenous grid of end-of-period assets.
        rfree (float): Risk free interest factor.
        crra (float): Coefficient of relative risk aversion.
        perm_gro_fac (float): Permanent income growth factor.
        vp_func_next (Callable): Marginal value function next period.
    """
    return shock["PermShk"] ** (-crra) * vp_func_next(
        calc_m_nrm_next(shock, a, rfree, perm_gro_fac),
    )


def calc_vpp_next(shock, a, rfree, crra, perm_gro_fac, vppfunc_next):
    """Calculate the continuation marginal marginal value function
    with respect to end-of-period assets.

    Args:
        shock (float): Realization of shocks to income.
        a (np.ndarray): Exogenous grid of end-of-period assets.
        rfree (float): Risk free interest factor.
        crra (float): Coefficient of relative risk aversion.
        perm_gro_fac (float): Permanent income growth factor.
        vppfunc_next (Callable): Marginal marginal value function next period.
    """
    return shock["PermShk"] ** (-crra - 1.0) * vppfunc_next(
        calc_m_nrm_next(shock, a, rfree, perm_gro_fac),
    )


def solve_one_period_ConsIndShock(
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
    """Solves one period of a consumption-saving model with idiosyncratic shocks to
    permanent and transitory income, with one risk free asset and CRRA utility.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncShkDstn : distribution.Distribution
        A discrete approximation to the income process between the period being
        solved and the one immediately following (in solution_next).
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree : float
        Risk free interest factor on end-of-period assets.
    PermGroFac : float
        Expected permanent income growth factor at the end of this period.
    BoroCnstArt: float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.  If it is less than the natural borrowing constraint,
        then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
        rowing constraint.
    aXtraGrid: np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear interpolation.

    Returns
    -------
    solution_now : ConsumerSolution
        Solution to this period's consumption-saving problem with income risk.

    """
    # Define the current period utility function and effective discount factor
    uFunc = UtilityFuncCRRA(CRRA)
    DiscFacEff = DiscFac * LivPrb  # "effective" discount factor

    # Calculate the probability that we get the worst possible income draw
    WorstIncPrb = calc_worst_inc_prob(IncShkDstn)
    Ex_IncNext = expected(lambda x: x["PermShk"] * x["TranShk"], IncShkDstn)
    hNrmNow = calc_human_wealth(solution_next.hNrm, PermGroFac, Rfree, Ex_IncNext)

    # Unpack next period's (marginal) value function
    vFuncNext = solution_next.vFunc  # This is None when vFuncBool is False
    vPfuncNext = solution_next.vPfunc
    vPPfuncNext = solution_next.vPPfunc  # This is None when CubicBool is False

    # Calculate the minimum allowable value of money resources in this period
    BoroCnstNat = calc_boro_const_nat(
        solution_next.mNrmMin, IncShkDstn, Rfree, PermGroFac
    )
    # Set the minimum allowable (normalized) market resources based on the natural
    # and artificial borrowing constraints
    mNrmMinNow = calc_m_nrm_min(BoroCnstArt, BoroCnstNat)

    # Update the bounding MPCs and PDV of human wealth:
    PatFac = calc_patience_factor(Rfree, DiscFacEff, CRRA)
    MPCminNow = calc_mpc_min(solution_next.MPCmin, PatFac)
    # Set the upper limit of the MPC (at mNrmMinNow) based on whether the natural
    # or artificial borrowing constraint actually binds
    MPCmaxUnc = calc_mpc_max(
        solution_next.MPCmax, WorstIncPrb, CRRA, PatFac, BoroCnstNat, BoroCnstArt
    )
    MPCmaxNow = 1.0 if BoroCnstNat < mNrmMinNow else MPCmaxUnc

    cFuncLimitIntercept = MPCminNow * hNrmNow
    cFuncLimitSlope = MPCminNow

    # Define the borrowing-constrained consumption function
    cFuncNowCnst = LinearInterp(
        np.array([mNrmMinNow, mNrmMinNow + 1.0]),
        np.array([0.0, 1.0]),
    )

    # Construct the assets grid by adjusting aXtra by the natural borrowing constraint
    aNrmNow = np.asarray(aXtraGrid) + BoroCnstNat

    # Calculate end-of-period marginal value of assets at each gridpoint
    vPfacEff = DiscFacEff * Rfree * PermGroFac ** (-CRRA)
    EndOfPrdvP = vPfacEff * expected(
        calc_vp_next,
        IncShkDstn,
        args=(aNrmNow, Rfree, CRRA, PermGroFac, vPfuncNext),
    )

    # Invert the first order condition to find optimal cNrm from each aNrm gridpoint
    cNrmNow = uFunc.derinv(EndOfPrdvP, order=(1, 0))
    mNrmNow = cNrmNow + aNrmNow  # Endogenous mNrm gridpoints

    # Limiting consumption is zero as m approaches mNrmMin
    c_for_interpolation = np.insert(cNrmNow, 0, 0.0)
    m_for_interpolation = np.insert(mNrmNow, 0, BoroCnstNat)

    # Construct the consumption function as a cubic or linear spline interpolation
    if CubicBool:
        # Calculate end-of-period marginal marginal value of assets at each gridpoint
        vPPfacEff = DiscFacEff * Rfree * Rfree * PermGroFac ** (-CRRA - 1.0)
        EndOfPrdvPP = vPPfacEff * expected(
            calc_vpp_next,
            IncShkDstn,
            args=(aNrmNow, Rfree, CRRA, PermGroFac, vPPfuncNext),
        )
        dcda = EndOfPrdvPP / uFunc.der(np.array(cNrmNow), order=2)
        MPC = dcda / (dcda + 1.0)
        MPC_for_interpolation = np.insert(MPC, 0, MPCmaxUnc)

        # Construct the unconstrained consumption function as a cubic interpolation
        cFuncNowUnc = CubicInterp(
            m_for_interpolation,
            c_for_interpolation,
            MPC_for_interpolation,
            cFuncLimitIntercept,
            cFuncLimitSlope,
        )
    else:
        # Construct the unconstrained consumption function as a linear interpolation
        cFuncNowUnc = LinearInterp(
            m_for_interpolation,
            c_for_interpolation,
            cFuncLimitIntercept,
            cFuncLimitSlope,
        )

    # Combine the constrained and unconstrained functions into the true consumption function.
    # LowerEnvelope should only be used when BoroCnstArt is True
    cFuncNow = LowerEnvelope(cFuncNowUnc, cFuncNowCnst, nan_bool=False)

    # Make the marginal value function and the marginal marginal value function
    vPfuncNow = MargValueFuncCRRA(cFuncNow, CRRA)

    # Define this period's marginal marginal value function
    if CubicBool:
        vPPfuncNow = MargMargValueFuncCRRA(cFuncNow, CRRA)
    else:
        vPPfuncNow = NullFunc()  # Dummy object

    # Construct this period's value function if requested
    if vFuncBool:
        # Calculate end-of-period value, its derivative, and their pseudo-inverse
        EndOfPrdv = DiscFacEff * expected(
            calc_v_next,
            IncShkDstn,
            args=(aNrmNow, Rfree, CRRA, PermGroFac, vFuncNext),
        )
        EndOfPrdvNvrs = uFunc.inv(
            EndOfPrdv,
        )  # value transformed through inverse utility
        EndOfPrdvNvrsP = EndOfPrdvP * uFunc.derinv(EndOfPrdv, order=(0, 1))
        EndOfPrdvNvrs = np.insert(EndOfPrdvNvrs, 0, 0.0)
        EndOfPrdvNvrsP = np.insert(EndOfPrdvNvrsP, 0, EndOfPrdvNvrsP[0])
        # This is a very good approximation, vNvrsPP = 0 at the asset minimum

        # Construct the end-of-period value function
        aNrm_temp = np.insert(aNrmNow, 0, BoroCnstNat)
        EndOfPrd_vNvrsFunc = CubicInterp(aNrm_temp, EndOfPrdvNvrs, EndOfPrdvNvrsP)
        EndOfPrd_vFunc = ValueFuncCRRA(EndOfPrd_vNvrsFunc, CRRA)

        # Compute expected value and marginal value on a grid of market resources
        mNrm_temp = mNrmMinNow + aXtraGrid
        cNrm_temp = cFuncNow(mNrm_temp)
        aNrm_temp = mNrm_temp - cNrm_temp
        v_temp = uFunc(cNrm_temp) + EndOfPrd_vFunc(aNrm_temp)
        vP_temp = uFunc.der(cNrm_temp)

        # Construct the beginning-of-period value function
        vNvrs_temp = uFunc.inv(v_temp)  # value transformed through inv utility
        vNvrsP_temp = vP_temp * uFunc.derinv(v_temp, order=(0, 1))
        mNrm_temp = np.insert(mNrm_temp, 0, mNrmMinNow)
        vNvrs_temp = np.insert(vNvrs_temp, 0, 0.0)
        vNvrsP_temp = np.insert(vNvrsP_temp, 0, MPCmaxNow ** (-CRRA / (1.0 - CRRA)))
        MPCminNvrs = MPCminNow ** (-CRRA / (1.0 - CRRA))
        vNvrsFuncNow = CubicInterp(
            mNrm_temp,
            vNvrs_temp,
            vNvrsP_temp,
            MPCminNvrs * hNrmNow,
            MPCminNvrs,
        )
        vFuncNow = ValueFuncCRRA(vNvrsFuncNow, CRRA)
    else:
        vFuncNow = NullFunc()  # Dummy object

    # Create and return this period's solution
    solution_now = ConsumerSolution(
        cFunc=cFuncNow,
        vFunc=vFuncNow,
        vPfunc=vPfuncNow,
        vPPfunc=vPPfuncNow,
        mNrmMin=mNrmMinNow,
        hNrm=hNrmNow,
        MPCmin=MPCminNow,
        MPCmax=MPCmaxNow,
    )
    return solution_now


def solve_one_period_ConsKinkedR(
    solution_next,
    IncShkDstn,
    LivPrb,
    DiscFac,
    CRRA,
    Rboro,
    Rsave,
    PermGroFac,
    BoroCnstArt,
    aXtraGrid,
    vFuncBool,
    CubicBool,
):
    """Solves one period of a consumption-saving model with idiosyncratic shocks to
    permanent and transitory income, with a risk free asset and CRRA utility.
    In this variation, the interest rate on borrowing Rboro exceeds the interest
    rate on saving Rsave.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncShkDstn : distribution.Distribution
        A discrete approximation to the income process between the period being
        solved and the one immediately following (in solution_next).
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
    Rboro: float
        Interest factor on assets between this period and the succeeding
        period when assets are negative.
    Rsave: float
        Interest factor on assets between this period and the succeeding
        period when assets are positive.
    PermGroFac : float
        Expected permanent income growth factor at the end of this period.
    BoroCnstArt: float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.  If it is less than the natural borrowing constraint,
        then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
        rowing constraint.
    aXtraGrid: np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear inter-
        polation.

    Returns
    -------
    solution_now : ConsumerSolution
        Solution to this period's consumption-saving problem with income risk.

    """
    # Verifiy that there is actually a kink in the interest factor
    assert (
        Rboro >= Rsave
    ), "Interest factor on debt less than interest factor on savings!"
    # If the kink is in the wrong direction, code should break here. If there's
    # no kink at all, then just use the ConsIndShockModel solver.
    if Rboro == Rsave:
        solution_now = solve_one_period_ConsIndShock(
            solution_next,
            IncShkDstn,
            LivPrb,
            DiscFac,
            CRRA,
            Rboro,
            PermGroFac,
            BoroCnstArt,
            aXtraGrid,
            vFuncBool,
            CubicBool,
        )
        return solution_now

    # Define the current period utility function and effective discount factor
    uFunc = UtilityFuncCRRA(CRRA)
    DiscFacEff = DiscFac * LivPrb  # "effective" discount factor

    # Calculate the probability that we get the worst possible income draw
    WorstIncPrb = calc_worst_inc_prob(IncShkDstn)
    # WorstIncPrb is the "Weierstrass p" concept: the odds we get the WORST thing
    Ex_IncNext = expected(lambda x: x["PermShk"] * x["TranShk"], IncShkDstn)
    hNrmNow = calc_human_wealth(solution_next.hNrm, PermGroFac, Rsave, Ex_IncNext)

    # Unpack next period's (marginal) value function
    vFuncNext = solution_next.vFunc  # This is None when vFuncBool is False
    vPfuncNext = solution_next.vPfunc
    vPPfuncNext = solution_next.vPPfunc  # This is None when CubicBool is False

    # Calculate the minimum allowable value of money resources in this period
    BoroCnstNat = calc_boro_const_nat(
        solution_next.mNrmMin, IncShkDstn, Rboro, PermGroFac
    )
    # Set the minimum allowable (normalized) market resources based on the natural
    # and artificial borrowing constraints
    mNrmMinNow = calc_m_nrm_min(BoroCnstArt, BoroCnstNat)

    # Update the bounding MPCs and PDV of human wealth:
    PatFacSave = calc_patience_factor(Rsave, DiscFacEff, CRRA)
    PatFacBoro = calc_patience_factor(Rboro, DiscFacEff, CRRA)
    MPCminNow = calc_mpc_min(solution_next.MPCmin, PatFacSave)
    # Set the upper limit of the MPC (at mNrmMinNow) based on whether the natural
    # or artificial borrowing constraint actually binds
    MPCmaxUnc = calc_mpc_max(
        solution_next.MPCmax, WorstIncPrb, CRRA, PatFacBoro, BoroCnstNat, BoroCnstArt
    )
    MPCmaxNow = 1.0 if BoroCnstNat < mNrmMinNow else MPCmaxUnc

    cFuncLimitIntercept = MPCminNow * hNrmNow
    cFuncLimitSlope = MPCminNow

    # Define the borrowing-constrained consumption function
    cFuncNowCnst = LinearInterp(
        np.array([mNrmMinNow, mNrmMinNow + 1.0]),
        np.array([0.0, 1.0]),
    )

    # Construct the assets grid by adjusting aXtra by the natural borrowing constraint
    aNrmNow = np.sort(
        np.hstack((np.asarray(aXtraGrid) + mNrmMinNow, np.array([0.0, 0.0]))),
    )

    # Make a 1D array of the interest factor at each asset gridpoint
    Rfree = Rsave * np.ones_like(aNrmNow)
    Rfree[aNrmNow < 0] = Rboro
    i_kink = np.argwhere(aNrmNow == 0.0)[0][0]
    Rfree[i_kink] = Rboro

    # Calculate end-of-period marginal value of assets at each gridpoint
    vPfacEff = DiscFacEff * Rfree * PermGroFac ** (-CRRA)
    EndOfPrdvP = vPfacEff * expected(
        calc_vp_next,
        IncShkDstn,
        args=(aNrmNow, Rfree, CRRA, PermGroFac, vPfuncNext),
    )

    # Invert the first order condition to find optimal cNrm from each aNrm gridpoint
    cNrmNow = uFunc.derinv(EndOfPrdvP, order=(1, 0))
    mNrmNow = cNrmNow + aNrmNow  # Endogenous mNrm gridpoints

    # Limiting consumption is zero as m approaches mNrmMin
    c_for_interpolation = np.insert(cNrmNow, 0, 0.0)
    m_for_interpolation = np.insert(mNrmNow, 0, BoroCnstNat)

    # Construct the consumption function as a cubic or linear spline interpolation
    if CubicBool:
        # Calculate end-of-period marginal marginal value of assets at each gridpoint
        vPPfacEff = DiscFacEff * Rfree * Rfree * PermGroFac ** (-CRRA - 1.0)
        EndOfPrdvPP = vPPfacEff * expected(
            calc_vpp_next,
            IncShkDstn,
            args=(aNrmNow, Rfree, CRRA, PermGroFac, vPPfuncNext),
        )
        dcda = EndOfPrdvPP / uFunc.der(np.array(cNrmNow), order=2)
        MPC = dcda / (dcda + 1.0)
        MPC_for_interpolation = np.insert(MPC, 0, MPCmaxUnc)

        # Construct the unconstrained consumption function as a cubic interpolation
        cFuncNowUnc = CubicInterp(
            m_for_interpolation,
            c_for_interpolation,
            MPC_for_interpolation,
            cFuncLimitIntercept,
            cFuncLimitSlope,
        )
        # Adjust the coefficients on the kinked portion of the cFunc
        cFuncNowUnc.coeffs[i_kink + 2] = [
            c_for_interpolation[i_kink + 1],
            m_for_interpolation[i_kink + 2] - m_for_interpolation[i_kink + 1],
            0.0,
            0.0,
        ]
    else:
        # Construct the unconstrained consumption function as a linear interpolation
        cFuncNowUnc = LinearInterp(
            m_for_interpolation,
            c_for_interpolation,
            cFuncLimitIntercept,
            cFuncLimitSlope,
        )

    # Combine the constrained and unconstrained functions into the true consumption function.
    # LowerEnvelope should only be used when BoroCnstArt is True
    cFuncNow = LowerEnvelope(cFuncNowUnc, cFuncNowCnst, nan_bool=False)

    # Make the marginal value function and the marginal marginal value function
    vPfuncNow = MargValueFuncCRRA(cFuncNow, CRRA)

    # Define this period's marginal marginal value function
    if CubicBool:
        vPPfuncNow = MargMargValueFuncCRRA(cFuncNow, CRRA)
    else:
        vPPfuncNow = NullFunc()  # Dummy object

    # Construct this period's value function if requested
    if vFuncBool:
        # Calculate end-of-period value, its derivative, and their pseudo-inverse
        EndOfPrdv = DiscFacEff * expected(
            calc_v_next,
            IncShkDstn,
            args=(aNrmNow, Rfree, CRRA, PermGroFac, vFuncNext),
        )
        EndOfPrdvNvrs = uFunc.inv(
            EndOfPrdv,
        )  # value transformed through inverse utility
        EndOfPrdvNvrsP = EndOfPrdvP * uFunc.derinv(EndOfPrdv, order=(0, 1))
        EndOfPrdvNvrs = np.insert(EndOfPrdvNvrs, 0, 0.0)
        EndOfPrdvNvrsP = np.insert(EndOfPrdvNvrsP, 0, EndOfPrdvNvrsP[0])
        # This is a very good approximation, vNvrsPP = 0 at the asset minimum

        # Construct the end-of-period value function
        aNrm_temp = np.insert(aNrmNow, 0, BoroCnstNat)
        EndOfPrdvNvrsFunc = CubicInterp(aNrm_temp, EndOfPrdvNvrs, EndOfPrdvNvrsP)
        EndOfPrdvFunc = ValueFuncCRRA(EndOfPrdvNvrsFunc, CRRA)

        # Compute expected value and marginal value on a grid of market resources
        mNrm_temp = mNrmMinNow + aXtraGrid
        cNrm_temp = cFuncNow(mNrm_temp)
        aNrm_temp = mNrm_temp - cNrm_temp
        v_temp = uFunc(cNrm_temp) + EndOfPrdvFunc(aNrm_temp)
        vP_temp = uFunc.der(cNrm_temp)

        # Construct the beginning-of-period value function
        vNvrs_temp = uFunc.inv(v_temp)  # value transformed through inv utility
        vNvrsP_temp = vP_temp * uFunc.derinv(v_temp, order=(0, 1))
        mNrm_temp = np.insert(mNrm_temp, 0, mNrmMinNow)
        vNvrs_temp = np.insert(vNvrs_temp, 0, 0.0)
        vNvrsP_temp = np.insert(vNvrsP_temp, 0, MPCmaxNow ** (-CRRA / (1.0 - CRRA)))
        MPCminNvrs = MPCminNow ** (-CRRA / (1.0 - CRRA))
        vNvrsFuncNow = CubicInterp(
            mNrm_temp,
            vNvrs_temp,
            vNvrsP_temp,
            MPCminNvrs * hNrmNow,
            MPCminNvrs,
        )
        vFuncNow = ValueFuncCRRA(vNvrsFuncNow, CRRA)
    else:
        vFuncNow = NullFunc()  # Dummy object

    # Create and return this period's solution
    solution_now = ConsumerSolution(
        cFunc=cFuncNow,
        vFunc=vFuncNow,
        vPfunc=vPfuncNow,
        vPPfunc=vPPfuncNow,
        mNrmMin=mNrmMinNow,
        hNrm=hNrmNow,
        MPCmin=MPCminNow,
        MPCmax=MPCmaxNow,
    )
    return solution_now


def make_basic_CRRA_solution_terminal(CRRA):
    """
    Construct the terminal period solution for a consumption-saving model with
    CRRA utility and only one state variable.

    Parameters
    ----------
    CRRA : float
        Coefficient of relative risk aversion. This is the only relevant parameter.

    Returns
    -------
    solution_terminal : ConsumerSolution
        Terminal period solution for someone with the given CRRA.
    """
    cFunc_terminal = LinearInterp([0.0, 1.0], [0.0, 1.0])  # c=m at t=T
    vFunc_terminal = ValueFuncCRRA(cFunc_terminal, CRRA)
    vPfunc_terminal = MargValueFuncCRRA(cFunc_terminal, CRRA)
    vPPfunc_terminal = MargMargValueFuncCRRA(cFunc_terminal, CRRA)
    solution_terminal = ConsumerSolution(
        cFunc=cFunc_terminal,
        vFunc=vFunc_terminal,
        vPfunc=vPfunc_terminal,
        vPPfunc=vPPfunc_terminal,
        mNrmMin=0.0,
        hNrm=0.0,
        MPCmin=1.0,
        MPCmax=1.0,
    )
    return solution_terminal


# ============================================================================
# == Classes for representing types of consumer agents (and things they do) ==
# ============================================================================

# Make a dictionary of constructors (very simply for perfect foresight model)
perf_foresight_constructors = {
    "solution_terminal": make_basic_CRRA_solution_terminal,
}

# Make a dictionary to specify a perfect foresight consumer type
init_perfect_foresight = {
    # BASIC HARK PARAMETERS REQUIRED TO SOLVE THE MODEL
    "cycles": 1,  # Finite, non-cyclic model
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    "constructors": perf_foresight_constructors,  # See dictionary above
    # PARAMETERS REQUIRED TO SOLVE THE MODEL
    "CRRA": 2.0,  # Coefficient of relative risk aversion
    "Rfree": 1.03,  # Interest factor on retained assets
    "DiscFac": 0.96,  # Intertemporal discount factor
    "LivPrb": [0.98],  # Survival probability after each period
    "PermGroFac": [1.01],  # Permanent income growth factor
    "BoroCnstArt": None,  # Artificial borrowing constraint
    "MaxKinks": 400,  # Maximum number of grid points to allow in cFunc
    # PARAMETERS REQUIRED TO SIMULATE THE MODEL
    "AgentCount": 10000,  # Number of agents of this type
    "T_age": None,  # Age after which simulated agents are automatically killed
    "aNrmInitMean": 0.0,  # Mean of log initial assets
    "aNrmInitStd": 1.0,  # Standard deviation of log initial assets
    "pLvlInitMean": 0.0,  # Mean of log initial permanent income
    "pLvlInitStd": 0.0,  # Standard deviation of log initial permanent income
    "PermGroFacAgg": 1.0,  # Aggregate permanent income growth factor
    # (The portion of PermGroFac attributable to aggregate productivity growth)
    # ADDITIONAL OPTIONAL PARAMETERS
    "PerfMITShk": False,  # Do Perfect Foresight MIT Shock
    # (Forces Newborns to follow solution path of the agent they replaced if True)
}


class PerfForesightConsumerType(AgentType):
    """
    A perfect foresight consumer type who has no uncertainty other than mortality.
    His problem is defined by a coefficient of relative risk aversion, intertemporal
    discount factor, interest factor, an artificial borrowing constraint (maybe)
    and time sequences of the permanent income growth rate and survival probability.
    """

    # Define some universal values for all consumer types
    cFunc_terminal_ = LinearInterp([0.0, 1.0], [0.0, 1.0])  # c=m in terminal period
    vFunc_terminal_ = LinearInterp([0.0, 1.0], [0.0, 0.0])  # This is overwritten
    solution_terminal_ = ConsumerSolution(
        cFunc=cFunc_terminal_,
        vFunc=vFunc_terminal_,
        mNrmMin=0.0,
        hNrm=0.0,
        MPCmin=1.0,
        MPCmax=1.0,
    )
    time_vary_ = ["LivPrb", "PermGroFac"]
    time_inv_ = ["CRRA", "DiscFac", "MaxKinks", "BoroCnstArt"]
    state_vars = ["pLvl", "PlvlAgg", "bNrm", "mNrm", "aNrm", "aLvl"]
    shock_vars_ = []

    def __init__(self, verbose=1, quiet=False, **kwds):
        params = init_perfect_foresight.copy()
        params.update(kwds)
        kwds = params

        # Initialize a basic AgentType
        super().__init__(
            pseudo_terminal=False,
            **kwds,
        )

        # Add consumer-type specific objects, copying to create independent versions
        self.time_vary = deepcopy(self.time_vary_)
        self.time_inv = deepcopy(self.time_inv_)
        self.shock_vars = deepcopy(self.shock_vars_)
        self.verbose = verbose
        self.quiet = quiet
        self.solve_one_period = solve_one_period_ConsPF
        set_verbosity_level((4 - verbose) * 10)
        self.bilt = {}
        self.update_Rfree()  # update interest rate if time varying

    def pre_solve(self):
        """
        Method that is run automatically just before solution by backward iteration.
        Solves the (trivial) terminal period and does a quick check on the borrowing
        constraint and MaxKinks attribute (only relevant in constrained, infinite
        horizon problems).
        """
        self.update_solution_terminal()  # Solve the terminal period problem
        if not self.quiet:
            self.check_conditions(verbose=self.verbose)

        # Fill in BoroCnstArt and MaxKinks if they're not specified or are irrelevant.
        # If no borrowing constraint specified...
        if not hasattr(self, "BoroCnstArt"):
            self.BoroCnstArt = None  # ...assume the user wanted none

        if not hasattr(self, "MaxKinks"):
            if self.cycles > 0:  # If it's not an infinite horizon model...
                self.MaxKinks = np.inf  # ...there's no need to set MaxKinks
            elif self.BoroCnstArt is None:  # If there's no borrowing constraint...
                self.MaxKinks = np.inf  # ...there's no need to set MaxKinks
            else:
                raise (
                    AttributeError(
                        "PerfForesightConsumerType requires the attribute MaxKinks to be specified when BoroCnstArt is not None and cycles == 0."
                    )
                )

    def post_solve(self):
        """
        Method that is run automatically at the end of a call to solve. Here, it
        simply calls calc_stable_points() if appropriate: an infinite horizon
        problem with a single repeated period in its cycle.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if (self.cycles == 0) and (self.T_cycle == 1):
            self.calc_stable_points()

    def check_restrictions(self):
        """
        A method to check that various restrictions are met for the model class.
        """
        if self.DiscFac < 0:
            raise Exception("DiscFac is below zero with value: " + str(self.DiscFac))

        return

    def update_solution_terminal(self):
        """
        Update the terminal period solution.  This method should be run when a
        new AgentType is created or when CRRA changes.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        self.construct("solution_terminal")

    def update_Rfree(self):
        """
        Determines whether Rfree is time-varying or fixed.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        if isinstance(self.Rfree, (int, float)):
            self.add_to_time_inv("Rfree")
        elif isinstance(self.Rfree, list):
            if len(self.Rfree) == self.T_cycle:
                if len(self.Rfree) == 1:
                    self.Rfree = self.Rfree[0]
                    self.add_to_time_inv("Rfree")
                else:
                    self.add_to_time_vary("Rfree")
            else:
                raise AttributeError(
                    "If Rfree is time-varying, it should have a length of T_cycle!"
                )
        elif isinstance(self.Rfree, np.ndarray):
            self.add_to_time_inv("Rfree")

    def unpack_cFunc(self):
        """DEPRECATED: Use solution.unpack('cFunc') instead.
        "Unpacks" the consumption functions into their own field for easier access.
        After the model has been solved, the consumption functions reside in the
        attribute cFunc of each element of ConsumerType.solution.  This method
        creates a (time varying) attribute cFunc that contains a list of consumption
        functions.
        Parameters
        ----------
        none
        Returns
        -------
        none
        """
        _log.critical(
            "unpack_cFunc is deprecated and it will soon be removed, "
            "please use unpack('cFunc') instead."
        )
        self.unpack("cFunc")

    def initialize_sim(self):
        self.PermShkAggNow = self.PermGroFacAgg  # This never changes during simulation
        self.state_now["PlvlAgg"] = 1.0
        super().initialize_sim()

    def sim_birth(self, which_agents):
        """
        Makes new consumers for the given indices.  Initialized variables include aNrm and pLvl, as
        well as time variables t_age and t_cycle.  Normalized assets and permanent income levels
        are drawn from lognormal distributions given by aNrmInitMean and aNrmInitStd (etc).

        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.AgentCount indicating which agents should be "born".

        Returns
        -------
        None
        """
        # Get and store states for newly born agents
        N = np.sum(which_agents)  # Number of new consumers to make
        self.state_now["aNrm"][which_agents] = Lognormal(
            mu=self.aNrmInitMean,
            sigma=self.aNrmInitStd,
            seed=self.RNG.integers(0, 2**31 - 1),
        ).draw(N)
        # why is a now variable set here? Because it's an aggregate.
        pLvlInitMeanNow = self.pLvlInitMean + np.log(
            self.state_now["PlvlAgg"]
        )  # Account for newer cohorts having higher permanent income
        self.state_now["pLvl"][which_agents] = Lognormal(
            pLvlInitMeanNow, self.pLvlInitStd, seed=self.RNG.integers(0, 2**31 - 1)
        ).draw(N)
        # How many periods since each agent was born
        self.t_age[which_agents] = 0

        if not hasattr(
            self, "PerfMITShk"
        ):  # If PerfMITShk not specified, let it be False
            self.PerfMITShk = False
        if not self.PerfMITShk:
            # If True, Newborns inherit t_cycle of agent they replaced (i.e. t_cycles are not reset).
            self.t_cycle[which_agents] = 0
            # Which period of the cycle each agent is currently in

        return None

    def sim_death(self):
        """
        Determines which agents die this period and must be replaced.  Uses the sequence in LivPrb
        to determine survival probabilities for each agent.

        Parameters
        ----------
        None

        Returns
        -------
        which_agents : np.array(bool)
            Boolean array of size AgentCount indicating which agents die.
        """
        # Determine who dies
        DiePrb_by_t_cycle = 1.0 - np.asarray(self.LivPrb)
        DiePrb = DiePrb_by_t_cycle[
            self.t_cycle - 1 if self.cycles == 1 else self.t_cycle
        ]  # Time has already advanced, so look back one

        # In finite-horizon problems the previous line gives newborns the
        # survival probability of the last non-terminal period. This is okay,
        # however, since they will be instantly replaced by new newborns if
        # they die.
        # See: https://github.com/econ-ark/HARK/pull/981

        DeathShks = Uniform(seed=self.RNG.integers(0, 2**31 - 1)).draw(
            N=self.AgentCount
        )
        which_agents = DeathShks < DiePrb
        if self.T_age is not None:  # Kill agents that have lived for too many periods
            too_old = self.t_age >= self.T_age
            which_agents = np.logical_or(which_agents, too_old)
        return which_agents

    def get_shocks(self):
        """
        Finds permanent and transitory income "shocks" for each agent this period.  As this is a
        perfect foresight model, there are no stochastic shocks: PermShkNow = PermGroFac for each
        agent (according to their t_cycle) and TranShkNow = 1.0 for all agents.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        PermGroFac = np.array(self.PermGroFac)
        # Cycle time has already been advanced
        self.shocks["PermShk"] = PermGroFac[self.t_cycle - 1]
        # self.shocks["PermShk"][self.t_cycle == 0] = 1. # Add this at some point
        self.shocks["TranShk"] = np.ones(self.AgentCount)

    def get_Rfree(self):
        """
        Returns an array of size self.AgentCount with self.Rfree in every entry.

        Parameters
        ----------
        None

        Returns
        -------
        RfreeNow : np.array
             Array of size self.AgentCount with risk free interest rate for each agent.
        """
        RfreeNow = np.ones(self.AgentCount)
        if "Rfree" in self.time_inv:
            RfreeNow = RfreeNow * self.Rfree
        elif "Rfree" in self.time_vary:
            for t in range(self.T_cycle):
                these = t == self.t_cycle
                RfreeNow[these] = self.Rfree[t]
        return RfreeNow

    def transition(self):
        pLvlPrev = self.state_prev["pLvl"]
        aNrmPrev = self.state_prev["aNrm"]
        RfreeNow = self.get_Rfree()

        # Calculate new states: normalized market resources and permanent income level
        # Updated permanent income level
        pLvlNow = pLvlPrev * self.shocks["PermShk"]
        # Updated aggregate permanent productivity level
        PlvlAggNow = self.state_prev["PlvlAgg"] * self.PermShkAggNow
        # "Effective" interest factor on normalized assets
        ReffNow = RfreeNow / self.shocks["PermShk"]
        bNrmNow = ReffNow * aNrmPrev  # Bank balances before labor income
        # Market resources after income
        mNrmNow = bNrmNow + self.shocks["TranShk"]

        return pLvlNow, PlvlAggNow, bNrmNow, mNrmNow, None

    def get_controls(self):
        """
        Calculates consumption for each consumer of this type using the consumption functions.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        cNrmNow = np.zeros(self.AgentCount) + np.nan
        MPCnow = np.zeros(self.AgentCount) + np.nan
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            cNrmNow[these], MPCnow[these] = self.solution[t].cFunc.eval_with_derivative(
                self.state_now["mNrm"][these]
            )
        self.controls["cNrm"] = cNrmNow

        # MPCnow is not really a control
        self.MPCnow = MPCnow
        return None

    def get_poststates(self):
        """
        Calculates end-of-period assets for each consumer of this type.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # should this be "Now", or "Prev"?!?
        self.state_now["aNrm"] = self.state_now["mNrm"] - self.controls["cNrm"]
        # Useful in some cases to precalculate asset level
        self.state_now["aLvl"] = self.state_now["aNrm"] * self.state_now["pLvl"]

        # moves now to prev
        super().get_poststates()

        return None

    def log_condition_result(self, name, result, message, verbose):
        """
        Records the result of one condition check in the attribute condition_report
        of the bilt dictionary, and in the message log.

        Parameters
        ----------
        name : string or None
             Name for the condition; if None, no test result is added to conditions.
        result : bool
             An indicator for whether the condition was passed.
        message : str
            The messages to record about the condition check.
        verbose : bool
            Indicator for whether verbose messages should be included in the report.
        """
        if name is not None:
            self.conditions[name] = result
        set_verbosity_level((4 - verbose) * 10)
        _log.info(message)
        self.bilt["conditions_report"] += message + "\n"

    def check_AIC(self, verbose=None):
        """
        Evaluate and report on the Absolute Impatience Condition.
        """
        name = "AIC"
        APFac = self.bilt["APFac"]
        result = APFac < 1.0

        messages = {
            True: f"APFac={APFac:.5f} : The Absolute Patience Factor satisfies the Absolute Impatience Condition (AIC) Þ < 1.",
            False: f"APFac={APFac:.5f} : The Absolute Patience Factor violates the Absolute Impatience Condition (AIC) Þ < 1.",
        }
        verbose = self.verbose if verbose is None else verbose
        self.log_condition_result(name, result, messages[result], verbose)

    def check_GICRaw(self, verbose=None):
        """
        Evaluate and report on the Growth Impatience Condition for the Perfect Foresight model.
        """
        name = "GICRaw"
        GPFacRaw = self.bilt["GPFacRaw"]
        result = GPFacRaw < 1.0

        messages = {
            True: f"GPFacRaw={GPFacRaw:.5f} : The Growth Patience Factor satisfies the Growth Impatience Condition (GICRaw) Þ/G < 1.",
            False: f"GPFacRaw={GPFacRaw:.5f} : The Growth Patience Factor violates the Growth Impatience Condition (GICRaw) Þ/G < 1.",
        }
        verbose = self.verbose if verbose is None else verbose
        self.log_condition_result(name, result, messages[result], verbose)

    def check_RIC(self, verbose=None):
        """
        Evaluate and report on the Return Impatience Condition.
        """
        name = "RIC"
        RPFac = self.bilt["RPFac"]
        result = RPFac < 1.0

        messages = {
            True: f"RPFac={RPFac:.5f} : The Return Patience Factor satisfies the Return Impatience Condition (RIC) Þ/R < 1.",
            False: f"RPFac={RPFac:.5f} : The Return Patience Factor violates the Return Impatience Condition (RIC) Þ/R < 1.",
        }
        verbose = self.verbose if verbose is None else verbose
        self.log_condition_result(name, result, messages[result], verbose)

    def check_FHWC(self, verbose=None):
        """
        Evaluate and report on the Finite Human Wealth Condition.
        """
        name = "FHWC"
        FHWFac = self.bilt["FHWFac"]
        result = FHWFac < 1.0

        messages = {
            True: f"FHWFac={FHWFac:.5f} : The Finite Human Wealth Factor satisfies the Finite Human Wealth Condition (FHWC) G/R < 1.",
            False: f"FHWFac={FHWFac:.5f} : The Finite Human Wealth Factor violates the Finite Human Wealth Condition (FHWC) G/R < 1.",
        }
        verbose = self.verbose if verbose is None else verbose
        self.log_condition_result(name, result, messages[result], verbose)

    def check_FVAC(self, verbose=None):
        """
        Evaluate and report on the Finite Value of Autarky Condition under perfect foresight.
        """
        name = "PFFVAC"
        PFVAFac = self.bilt["PFVAFac"]
        result = PFVAFac < 1.0

        messages = {
            True: f"PFVAFac={PFVAFac:.5f} : The Finite Value of Autarky Factor satisfies the Finite Value of Autarky Condition βG^(1-ρ) < 1.",
            False: f"PFVAFac={PFVAFac:.5f} : The Finite Value of Autarky Factor violates the Finite Value of Autarky Condition βG^(1-ρ) < 1.",
        }
        verbose = self.verbose if verbose is None else verbose
        self.log_condition_result(name, result, messages[result], verbose)

    def describe_parameters(self):
        """
        Make a string describing this instance's parameter values, including their
        representation in code and symbolically.

        Returns
        -------
        param_desc : str
            Description of parameters as a unicode string.
        """
        params_to_describe = [
            # [name, description, symbol, time varying]
            ["DiscFac", "intertemporal discount factor", "β", False],
            ["Rfree", "risk free interest factor", "R", False],
            ["PermGroFac", "permanent income growth factor", "G", True],
            ["CRRA", "coefficient of relative risk aversion", "ρ", False],
            ["LivPrb", "survival probability", "ℒ", True],
            ["APFac", "absolute patience factor", "Þ=(βℒR)^(1/ρ)", False],
        ]

        param_desc = ""
        for j in range(len(params_to_describe)):
            this_entry = params_to_describe[j]
            if this_entry[3]:
                val = getattr(self, this_entry[0])[0]
            else:
                try:
                    val = getattr(self, this_entry[0])
                except:
                    val = self.bilt[this_entry[0]]
            this_line = (
                this_entry[2]
                + f"={val:.5f} : "
                + this_entry[1]
                + " ("
                + this_entry[0]
                + ")\n"
            )
            param_desc += this_line

        return param_desc

    def calc_limiting_values(self):
        """
        Compute various scalar values that are relevant to characterizing the
        solution to an infinite horizon problem. This method should only be called
        when T_cycle=1 and cycles=0, otherwise the values generated are meaningless.
        This method adds the following values to the instance in the dictionary
        attribute called bilt.

        APFac : Absolute Patience Factor
        GPFacRaw : Growth Patience Factor
        FHWFac : Finite Human Wealth Factor
        RPFac : Return Patience Factor
        PFVAFac : Perfect Foresight Value of Autarky Factor
        cNrmPDV : Present Discounted Value of Autarky Consumption
        MPCmin : Limiting minimum MPC as market resources go to infinity
        MPCmax : Limiting maximum MPC as market resources approach minimum level.
        hNrm : Human wealth divided by permanent income.
        Delta_mNrm_ZeroFunc : Linear consumption function where expected change in market resource ratio is zero
        BalGroFunc : Linear consumption function where the level of market resources grows at the same rate as permanent income

        Returns
        -------
        None
        """
        aux_dict = self.bilt
        aux_dict["APFac"] = (self.Rfree * self.DiscFac * self.LivPrb[0]) ** (
            1 / self.CRRA
        )
        aux_dict["GPFacRaw"] = aux_dict["APFac"] / self.PermGroFac[0]
        aux_dict["FHWFac"] = self.PermGroFac[0] / self.Rfree
        aux_dict["RPFac"] = aux_dict["APFac"] / self.Rfree
        aux_dict["PFVAFac"] = (self.DiscFac * self.LivPrb[0]) * self.PermGroFac[0] ** (
            1.0 - self.CRRA
        )
        aux_dict["cNrmPDV"] = 1.0 / (1.0 - aux_dict["RPFac"])
        aux_dict["MPCmin"] = np.maximum(1.0 - aux_dict["RPFac"], 0.0)
        constrained = (
            hasattr(self, "BoroCnstArt")
            and (self.BoroCnstArt is not None)
            and (self.BoroCnstArt > -np.inf)
        )

        if constrained:
            aux_dict["MPCmax"] = 1.0
        else:
            aux_dict["MPCmax"] = aux_dict["MPCmin"]
        if aux_dict["FHWFac"] < 1.0:
            aux_dict["hNrm"] = 1.0 / (1.0 - aux_dict["FHWFac"])
        else:
            aux_dict["hNrm"] = np.inf

        # Generate the "Delta m = 0" function, which is used to find target market resources
        Ex_Rnrm = self.Rfree / self.PermGroFac[0]
        aux_dict["Delta_mNrm_ZeroFunc"] = (
            lambda m: (1.0 - 1.0 / Ex_Rnrm) * m + 1.0 / Ex_Rnrm
        )

        # Generate the "E[M_tp1 / M_t] = G" function, which is used to find balanced growth market resources
        PF_Rnrm = self.Rfree / self.PermGroFac[0]
        aux_dict["BalGroFunc"] = lambda m: (1.0 - 1.0 / PF_Rnrm) * m + 1.0 / PF_Rnrm

        self.bilt = aux_dict

    def check_conditions(self, verbose=None):
        """
        This method checks whether the instance's type satisfies the
        Absolute Impatience Condition (AIC), the Return Impatience Condition (RIC),
        the Finite Human Wealth Condition (FHWC), the perfect foresight model's
        Growth Impatience Condition (GICRaw) and Perfect Foresight Finite Value
        of Autarky Condition (FVACPF). Depending on the configuration of parameter
        values, somecombination of these conditions must be satisfied in order
        for the problem to have a nondegenerate solution. To check which conditions
        are required, in the verbose mode a reference to the relevant theoretical
        literature is made.

        Parameters
        ----------
        verbose : boolean
            Specifies different levels of verbosity of feedback. When False, it
            only reports whether the instance's type fails to satisfy a particular
            condition. When True, it reports all results, i.e. the factor values
            for all conditions.

        Returns
        -------
        None
        """
        self.conditions = {}
        self.bilt["conditions_report"] = ""
        self.degenerate = False
        verbose = self.verbose if verbose is None else verbose

        # This method only checks for the conditions for infinite horizon models
        # with a 1 period cycle. If these conditions are not met, we exit early.
        if self.cycles != 0 or self.T_cycle > 1:
            trivial_message = "No conditions report was produced because this functionality is only supported for infinite horizon models with a cycle length of 1."
            self.log_condition_result(None, None, trivial_message, verbose)
            if not self.quiet:
                _log.info(self.bilt["conditions_report"])
            return

        # Calculate some useful quantities that will be used in the condition checks
        self.calc_limiting_values()
        param_desc = self.describe_parameters()
        self.log_condition_result(None, None, param_desc, verbose)

        # Check individual conditions and add their results to the report
        self.check_AIC(verbose)
        self.check_RIC(verbose)
        self.check_GICRaw(verbose)
        self.check_FVAC(verbose)
        self.check_FHWC(verbose)
        constrained = (
            hasattr(self, "BoroCnstArt")
            and (self.BoroCnstArt is not None)
            and (self.BoroCnstArt > -np.inf)
        )

        # Exit now if verbose output was not requested.
        if not verbose:
            if not self.quiet:
                _log.info(self.bilt["conditions_report"])
            return

        # Report on the degeneracy of the consumption function solution
        if not constrained:
            if self.conditions["FHWC"]:
                RIC_message = "\nBecause the FHWC is satisfied, the solution is not c(m)=Infinity."
                if self.conditions["RIC"]:
                    RIC_message += " Because the RIC is also satisfied, the solution is also not c(m)=0 for all m, so a non-degenerate linear solution exists."
                    degenerate = False
                else:
                    RIC_message += " However, because the RIC is violated, the solution is degenerate at c(m) = 0 for all m."
                    degenerate = True
            else:
                RIC_message = "\nBecause the FHWC condition is violated and the consumer is not constrained, the solution is degenerate at c(m)=Infinity."
                degenerate = True
        else:
            if self.conditions["RIC"]:
                RIC_message = "\nBecause the RIC is satisfied and the consumer is constrained, the solution is not c(m)=0 for all m."
                if self.conditions["GICRaw"]:
                    RIC_message += " Because the GICRaw is also satisfied, the solution is non-degenerate. It is piecewise linear with an infinite number of kinks, approaching the unconstrained solution as m goes to infinity."
                    degenerate = False
                else:
                    RIC_message += " Because the GICRaw is violated, the solution is non-degenerate. It is piecewise linear with a single kink at some 0 < m < 1; it equals the unconstrained solution above that kink point and has c(m) = m below it."
                    degenerate = False
            else:
                if self.conditions["GICRaw"]:
                    RIC_message = "\nBecause the RIC is violated but the GIC is satisfied, the FHWC is necessarily also violated. In this case, the consumer's pathological patience is offset by his infinite human wealth, against which he cannot borrow arbitrarily; a non-degenerate solution exists."
                    degenerate = False
                else:
                    RIC_message = "\nBecause the RIC is violated but the FHWC is satisfied, the solution is degenerate at c(m)=0 for all m."
                    degenerate = True
        self.log_condition_result(None, None, RIC_message, verbose)

        if (
            degenerate
        ):  # All of the other checks are meaningless if the solution is degenerate
            if not self.quiet:
                _log.info(self.bilt["conditions_report"])
            return

        # Report on the consequences of the Absolute Impatience Condition
        if self.conditions["AIC"]:
            AIC_message = "\nBecause the AIC is satisfied, the absolute amount of consumption is expected to fall over time."
        else:
            AIC_message = "\nBecause the AIC is violated, the absolute amount of consumption is expected to grow over time."
        self.log_condition_result(None, None, AIC_message, verbose)

        # Report on the consequences of the Growth Impatience Condition
        if self.conditions["GICRaw"]:
            GIC_message = "\nBecause the GICRaw is satisfed, the ratio of individual wealth to permanent income is expected to fall indefinitely."
        elif self.conditions["FHWC"]:
            GIC_message = "\nBecause the GICRaw is violated but the FHWC is satisfied, the ratio of individual wealth to permanent income is expected to rise toward infinity."
        else:
            pass
            # This can never be reached! If GICRaw and FHWC both fail, then the RIC also fails, and we would have exited by this point.
        self.log_condition_result(None, None, GIC_message, verbose)

        if not self.quiet:
            _log.info(self.bilt["conditions_report"])

    def calc_stable_points(self):
        """
        If the problem is one that satisfies the conditions required for target ratios of different
        variables to permanent income to exist, and has been solved to within the self-defined
        tolerance, this method calculates the target values of market resources.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        infinite_horizon = self.cycles == 0
        single_period = self.T_cycle = 1
        if not infinite_horizon:
            _log.warning(
                "The calc_stable_points method works only for infinite horizon models."
            )
            return
        if not single_period:
            _log.warning(
                "The calc_stable_points method works only with a single infinitely repeated period."
            )
            return
        if not hasattr(self, "conditions"):
            _log.warning(
                "The calc_limiting_values method must be run before the calc_stable_points method."
            )
            return
        if not hasattr(self, "solution"):
            _log.warning(
                "The solve method must be run before the calc_stable_points method."
            )
            return

        # Extract balanced growth and delta m_t+1 = 0 functions
        BalGroFunc = self.bilt["BalGroFunc"]
        Delta_mNrm_ZeroFunc = self.bilt["Delta_mNrm_ZeroFunc"]

        # If the GICRaw holds, then there is a balanced growth market resources ratio
        if self.conditions["GICRaw"]:
            cFunc = self.solution[0].cFunc
            func_to_zero = lambda m: BalGroFunc(m) - cFunc(m)
            m0 = 1.0
            try:
                mNrmStE = newton(func_to_zero, m0)
            except:
                mNrmStE = np.nan

            # A target level of assets *might* exist even if the GICMod fails, so check no matter what
            func_to_zero = lambda m: Delta_mNrm_ZeroFunc(m) - cFunc(m)
            m0 = 1.0 if np.isnan(mNrmStE) else mNrmStE
            try:
                mNrmTrg = newton(func_to_zero, m0, maxiter=200)
            except:
                mNrmTrg = np.nan
        else:
            mNrmStE = np.nan
            mNrmTrg = np.nan

        self.solution[0].mNrmStE = mNrmStE
        self.solution[0].mNrmTrg = mNrmTrg
        self.bilt["mNrmStE"] = mNrmStE
        self.bilt["mNrmTrg"] = mNrmTrg


###############################################################################

# Make a dictionary of constructors
indshk_constructor_dict = {
    "IncShkDstn": construct_lognormal_income_process_unemployment,
    "PermShkDstn": get_PermShkDstn_from_IncShkDstn,
    "TranShkDstn": get_TranShkDstn_from_IncShkDstn,
    "aXtraGrid": construct_assets_grid,
    "solution_terminal": make_basic_CRRA_solution_terminal,
}

# Default parameters to make IncShkDstn using construct_lognormal_income_process_unemployment
default_IncShkDstn_params = {
    "PermShkStd": [0.1],  # Standard deviation of log permanent income shocks
    "PermShkCount": 7,  # Number of points in discrete approximation to permanent income shocks
    "TranShkStd": [0.1],  # Standard deviation of log transitory income shocks
    "TranShkCount": 7,  # Number of points in discrete approximation to transitory income shocks
    "UnempPrb": 0.05,  # Probability of unemployment while working
    "IncUnemp": 0.3,  # Unemployment benefits replacement rate while working
    "T_retire": 0,  # Period of retirement (0 --> no retirement)
    "UnempPrbRet": 0.005,  # Probability of "unemployment" while retired
    "IncUnempRet": 0.0,  # "Unemployment" benefits when retired
}

# Default parameters to make aXtraGrid using construct_assets_grid
default_aXtraGrid_params = {
    "aXtraMin": 0.001,  # Minimum end-of-period "assets above minimum" value
    "aXtraMax": 20,  # Maximum end-of-period "assets above minimum" value
    "aXtraNestFac": 3,  # Exponential nesting factor for aXtraGrid
    "aXtraCount": 48,  # Number of points in the grid of "assets above minimum"
    "aXtraExtra": None,  # Additional other values to add in grid (optional)
}

# Make a dictionary to specify an idiosyncratic income shocks consumer type
init_idiosyncratic_shocks = {
    # BASIC HARK PARAMETERS REQUIRED TO SOLVE THE MODEL
    "cycles": 1,  # Finite, non-cyclic model
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    "constructors": indshk_constructor_dict,  # See dictionary above
    # PRIMITIVE RAW PARAMETERS REQUIRED TO SOLVE THE MODEL
    "CRRA": 2.0,  # Coefficient of relative risk aversion
    "Rfree": 1.03,  # Interest factor on retained assets
    "DiscFac": 0.96,  # Intertemporal discount factor
    "LivPrb": [0.98],  # Survival probability after each period
    "PermGroFac": [1.01],  # Permanent income growth factor
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint
    "vFuncBool": False,  # Whether to calculate the value function during solution
    "CubicBool": False,  # Whether to use cubic spline interpolation when True
    # (Uses linear spline interpolation for cFunc when False)
    # PARAMETERS REQUIRED TO SIMULATE THE MODEL
    "AgentCount": 10000,  # Number of agents of this type
    "T_age": None,  # Age after which simulated agents are automatically killed
    "aNrmInitMean": 0.0,  # Mean of log initial assets
    "aNrmInitStd": 1.0,  # Standard deviation of log initial assets
    "pLvlInitMean": 0.0,  # Mean of log initial permanent income
    "pLvlInitStd": 0.0,  # Standard deviation of log initial permanent income
    "PermGroFacAgg": 1.0,  # Aggregate permanent income growth factor
    # (The portion of PermGroFac attributable to aggregate productivity growth)
    "NewbornTransShk": False,  # Whether Newborns have transitory shock
    # ADDITIONAL OPTIONAL PARAMETERS
    "PerfMITShk": False,  # Do Perfect Foresight MIT Shock
    # (Forces Newborns to follow solution path of the agent they replaced if True)
    "neutral_measure": False,  # Whether to use permanent income neutral measure (see Harmenberg 2021)
}
init_idiosyncratic_shocks.update(default_IncShkDstn_params)
init_idiosyncratic_shocks.update(default_aXtraGrid_params)


class IndShockConsumerType(PerfForesightConsumerType):
    """
    A consumer type with idiosyncratic shocks to permanent and transitory income.
    His problem is defined by a sequence of income distributions, survival prob-
    abilities, and permanent income growth rates, as well as time invariant values
    for risk aversion, discount factor, the interest rate, the grid of end-of-
    period assets, and an artificial borrowing constraint.

    Parameters
    ----------
    cycles : int
        Number of times the sequence of periods should be solved.
    """

    time_inv_ = PerfForesightConsumerType.time_inv_ + [
        "BoroCnstArt",
        "vFuncBool",
        "CubicBool",
    ]
    # This is in the PerfForesight model but not ConsIndShock
    time_inv_.remove("MaxKinks")
    shock_vars_ = ["PermShk", "TranShk"]

    def __init__(self, verbose=1, quiet=False, **kwds):
        params = init_idiosyncratic_shocks.copy()
        params.update(kwds)

        # Initialize a basic PerfForesightConsumerType
        super().__init__(verbose=verbose, quiet=quiet, **params)

        # Add consumer-type specific objects, copying to create independent versions
        self.solve_one_period = solve_one_period_ConsIndShock
        self.update()  # Make assets grid, income process, terminal solution

    def update_income_process(self):
        """
        Updates this agent's income process based on his own attributes.

        Parameters
        ----------
        none

        Returns:
        -----------
        none
        """
        self.construct("IncShkDstn", "PermShkDstn", "TranShkDstn")
        self.add_to_time_vary("IncShkDstn", "PermShkDstn", "TranShkDstn")

    def update_assets_grid(self):
        """
        Updates this agent's end-of-period assets grid by constructing a multi-
        exponentially spaced grid of aXtra values.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        self.construct("aXtraGrid")
        self.add_to_time_inv("aXtraGrid")

    def update(self):
        """
        Update the income process, the assets grid, and the terminal solution.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.update_income_process()
        self.update_assets_grid()
        self.update_solution_terminal()

    def reset_rng(self):
        """
        Reset the RNG behavior of this type.  This method is called automatically
        by initialize_sim(), ensuring that each simulation run uses the same sequence
        of random shocks; this is necessary for structural estimation to work.
        This method extends AgentType.reset_rng() to also reset elements of IncShkDstn.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        super().reset_rng()

        # Reset IncShkDstn if it exists (it might not because reset_rng is called at init)
        if hasattr(self, "IncShkDstn"):
            for dstn in self.IncShkDstn:
                dstn.reset()

    def get_shocks(self):
        """
        Gets permanent and transitory income shocks for this period.  Samples from IncShkDstn for
        each period in the cycle.

        Parameters
        ----------
        NewbornTransShk : boolean, optional
            Whether Newborns have transitory shock. The default is False.

        Returns
        -------
        None
        """
        NewbornTransShk = (
            self.NewbornTransShk
        )  # Whether Newborns have transitory shock. The default is False.

        PermShkNow = np.zeros(self.AgentCount)  # Initialize shock arrays
        TranShkNow = np.zeros(self.AgentCount)
        newborn = self.t_age == 0
        for t in range(self.T_cycle):
            these = t == self.t_cycle

            # temporary, see #1022
            if self.cycles == 1:
                t = t - 1

            N = np.sum(these)
            if N > 0:
                # set current income distribution
                IncShkDstnNow = self.IncShkDstn[t]
                # and permanent growth factor
                PermGroFacNow = self.PermGroFac[t]
                # Get random draws of income shocks from the discrete distribution
                IncShks = IncShkDstnNow.draw(N)

                PermShkNow[these] = (
                    IncShks[0, :] * PermGroFacNow
                )  # permanent "shock" includes expected growth
                TranShkNow[these] = IncShks[1, :]

        # That procedure used the *last* period in the sequence for newborns, but that's not right
        # Redraw shocks for newborns, using the *first* period in the sequence.  Approximation.
        N = np.sum(newborn)
        if N > 0:
            these = newborn
            # set current income distribution
            IncShkDstnNow = self.IncShkDstn[0]
            PermGroFacNow = self.PermGroFac[0]  # and permanent growth factor

            # Get random draws of income shocks from the discrete distribution
            EventDraws = IncShkDstnNow.draw_events(N)
            PermShkNow[these] = (
                IncShkDstnNow.atoms[0][EventDraws] * PermGroFacNow
            )  # permanent "shock" includes expected growth
            TranShkNow[these] = IncShkDstnNow.atoms[1][EventDraws]
        #        PermShkNow[newborn] = 1.0
        #  Whether Newborns have transitory shock. The default is False.
        if not NewbornTransShk:
            TranShkNow[newborn] = 1.0

        # Store the shocks in self
        self.EmpNow = np.ones(self.AgentCount, dtype=bool)
        self.EmpNow[TranShkNow == self.IncUnemp] = False
        self.shocks["PermShk"] = PermShkNow
        self.shocks["TranShk"] = TranShkNow

    def define_distribution_grid(
        self,
        dist_mGrid=None,
        dist_pGrid=None,
        m_density=0,
        num_pointsM=None,
        timestonest=None,
        num_pointsP=55,
        max_p_fac=30.0,
    ):
        """
        Defines the grid on which the distribution is defined. Stores the grid of market resources and permanent income as attributes of self.
        Grid for normalized market resources and permanent income may be prespecified
        as dist_mGrid and dist_pGrid, respectively. If not then default grid is computed based off given parameters.

        Parameters
        ----------
        dist_mGrid : np.array
                Prespecified grid for distribution over normalized market resources

        dist_pGrid : np.array
                Prespecified grid for distribution over permanent income.

        m_density: float
                Density of normalized market resources grid. Default value is mdensity = 0.
                Only affects grid of market resources if dist_mGrid=None.

        num_pointsM: float
                Number of gridpoints for market resources grid.

        num_pointsP: float
                 Number of gridpoints for permanent income.
                 This grid will be exponentiated by the function make_grid_exp_mult.

        max_p_fac : float
                Factor that scales the maximum value of permanent income grid.
                Larger values increases the maximum value of permanent income grid.

        Returns
        -------
        None
        """

        # If true Use Harmenberg 2021's Neutral Measure. For more information, see https://econ-ark.org/materials/harmenberg-aggregation?launch
        if not hasattr(self, "neutral_measure"):
            self.neutral_measure = False

        if num_pointsM is None:
            m_points = self.mCount
        else:
            m_points = num_pointsM

        if not isinstance(timestonest, int):
            timestonest = self.mFac
        else:
            timestonest = timestonest

        if self.cycles == 0:
            if not hasattr(dist_mGrid, "__len__"):
                mGrid = make_grid_exp_mult(
                    ming=self.mMin,
                    maxg=self.mMax,
                    ng=m_points,
                    timestonest=timestonest,
                )  # Generate Market resources grid given density and number of points

                for i in range(m_density):
                    m_shifted = np.delete(mGrid, -1)
                    m_shifted = np.insert(m_shifted, 0, 1.00000000e-04)
                    dist_betw_pts = mGrid - m_shifted
                    dist_betw_pts_half = dist_betw_pts / 2
                    new_A_grid = m_shifted + dist_betw_pts_half
                    mGrid = np.concatenate((mGrid, new_A_grid))
                    mGrid = np.sort(mGrid)

                self.dist_mGrid = mGrid

            else:
                # If grid of market resources prespecified then use as mgrid
                self.dist_mGrid = dist_mGrid

            if not hasattr(dist_pGrid, "__len__"):
                num_points = num_pointsP  # Number of permanent income gridpoints
                # Dist_pGrid is taken to cover most of the ergodic distribution
                # set variance of permanent income shocks
                p_variance = self.PermShkStd[0] ** 2
                # Maximum Permanent income value
                max_p = max_p_fac * (p_variance / (1 - self.LivPrb[0])) ** 0.5
                one_sided_grid = make_grid_exp_mult(
                    1.05 + 1e-3, np.exp(max_p), num_points, 3
                )
                self.dist_pGrid = np.append(
                    np.append(1.0 / np.fliplr([one_sided_grid])[0], np.ones(1)),
                    one_sided_grid,
                )  # Compute permanent income grid
            else:
                # If grid of permanent income prespecified then use it as pgrid
                self.dist_pGrid = dist_pGrid

            if (
                self.neutral_measure is True
            ):  # If true Use Harmenberg 2021's Neutral Measure. For more information, see https://econ-ark.org/materials/harmenberg-aggregation?launch
                self.dist_pGrid = np.array([1])

        elif self.cycles > 1:
            raise Exception(
                "define_distribution_grid requires cycles = 0 or cycles = 1"
            )

        elif self.T_cycle != 0:
            if num_pointsM is None:
                m_points = self.mCount
            else:
                m_points = num_pointsM

            if not hasattr(dist_mGrid, "__len__"):
                mGrid = make_grid_exp_mult(
                    ming=self.mMin,
                    maxg=self.mMax,
                    ng=m_points,
                    timestonest=timestonest,
                )  # Generate Market resources grid given density and number of points

                for i in range(m_density):
                    m_shifted = np.delete(mGrid, -1)
                    m_shifted = np.insert(m_shifted, 0, 1.00000000e-04)
                    dist_betw_pts = mGrid - m_shifted
                    dist_betw_pts_half = dist_betw_pts / 2
                    new_A_grid = m_shifted + dist_betw_pts_half
                    mGrid = np.concatenate((mGrid, new_A_grid))
                    mGrid = np.sort(mGrid)

                self.dist_mGrid = mGrid

            else:
                # If grid of market resources prespecified then use as mgrid
                self.dist_mGrid = dist_mGrid

            if not hasattr(dist_pGrid, "__len__"):
                self.dist_pGrid = []  # list of grids of permanent income

                for i in range(self.T_cycle):
                    num_points = num_pointsP
                    # Dist_pGrid is taken to cover most of the ergodic distribution
                    # set variance of permanent income shocks this period
                    p_variance = self.PermShkStd[i] ** 2
                    # Consider probability of staying alive this period
                    max_p = max_p_fac * (p_variance / (1 - self.LivPrb[i])) ** 0.5
                    one_sided_grid = make_grid_exp_mult(
                        1.05 + 1e-3, np.exp(max_p), num_points, 2
                    )

                    # Compute permanent income grid this period. Grid of permanent income may differ dependent on PermShkStd
                    dist_pGrid = np.append(
                        np.append(1.0 / np.fliplr([one_sided_grid])[0], np.ones(1)),
                        one_sided_grid,
                    )
                    self.dist_pGrid.append(dist_pGrid)

            else:
                # If grid of permanent income prespecified then use as pgrid
                self.dist_pGrid = dist_pGrid

            if (
                self.neutral_measure is True
            ):  # If true Use Harmenberg 2021's Neutral Measure. For more information, see https://econ-ark.org/materials/harmenberg-aggregation?launch
                self.dist_pGrid = self.T_cycle * [np.array([1])]

    def calc_transition_matrix(self, shk_dstn=None):
        """
        Calculates how the distribution of agents across market resources
        transitions from one period to the next. If finite horizon problem, then calculates
        a list of transition matrices, consumption and asset policy grids for each period of the problem.
        The transition matrix/matrices and consumption and asset policy grid(s) are stored as attributes of self.


        Parameters
        ----------
            shk_dstn: list
                list of income shock distributions. Each Income Shock Distribution should be a DiscreteDistribution Object (see Distribution.py)
        Returns
        -------
        None

        """

        if self.cycles == 0:  # Infinite Horizon Problem
            if not hasattr(shk_dstn, "pmv"):
                shk_dstn = self.IncShkDstn

            dist_mGrid = self.dist_mGrid  # Grid of market resources
            dist_pGrid = self.dist_pGrid  # Grid of permanent incomes
            # assets next period
            aNext = dist_mGrid - self.solution[0].cFunc(dist_mGrid)

            self.aPol_Grid = aNext  # Steady State Asset Policy Grid
            # Steady State Consumption Policy Grid
            self.cPol_Grid = self.solution[0].cFunc(dist_mGrid)

            # Obtain shock values and shock probabilities from income distribution
            # Bank Balances next period (Interest rate * assets)
            bNext = self.Rfree * aNext
            shk_prbs = shk_dstn[0].pmv  # Probability of shocks
            tran_shks = shk_dstn[0].atoms[1]  # Transitory shocks
            perm_shks = shk_dstn[0].atoms[0]  # Permanent shocks
            LivPrb = self.LivPrb[0]  # Update probability of staying alive

            # New borns have this distribution (assumes start with no assets and permanent income=1)
            NewBornDist = jump_to_grid_2D(
                tran_shks, np.ones_like(tran_shks), shk_prbs, dist_mGrid, dist_pGrid
            )

            if len(dist_pGrid) == 1:
                NewBornDist = jump_to_grid_1D(
                    np.ones_like(tran_shks), shk_prbs, dist_mGrid
                )
                # Compute Transition Matrix given shocks and grids.
                self.tran_matrix = gen_tran_matrix_1D(
                    dist_mGrid,
                    bNext,
                    shk_prbs,
                    perm_shks,
                    tran_shks,
                    LivPrb,
                    NewBornDist,
                )

            else:
                NewBornDist = jump_to_grid_2D(
                    np.ones_like(tran_shks),
                    np.ones_like(tran_shks),
                    shk_prbs,
                    dist_mGrid,
                    dist_pGrid,
                )

                # Generate Transition Matrix
                # Compute Transition Matrix given shocks and grids.
                self.tran_matrix = gen_tran_matrix_2D(
                    dist_mGrid,
                    dist_pGrid,
                    bNext,
                    shk_prbs,
                    perm_shks,
                    tran_shks,
                    LivPrb,
                    NewBornDist,
                )

        elif self.cycles > 1:
            raise Exception("calc_transition_matrix requires cycles = 0 or cycles = 1")

        elif self.T_cycle != 0:  # finite horizon problem
            if not hasattr(shk_dstn, "pmv"):
                shk_dstn = self.IncShkDstn

            self.cPol_Grid = []
            # List of consumption policy grids for each period in T_cycle
            self.aPol_Grid = []
            # List of asset policy grids for each period in T_cycle
            self.tran_matrix = []  # List of transition matrices

            dist_mGrid = self.dist_mGrid

            for k in range(self.T_cycle):
                if type(self.dist_pGrid) == list:
                    # Permanent income grid this period
                    dist_pGrid = self.dist_pGrid[k]
                else:
                    dist_pGrid = (
                        self.dist_pGrid
                    )  # If here then use prespecified permanent income grid

                # Consumption policy grid in period k
                Cnow = self.solution[k].cFunc(dist_mGrid)
                self.cPol_Grid.append(Cnow)  # Add to list

                aNext = dist_mGrid - Cnow  # Asset policy grid in period k
                self.aPol_Grid.append(aNext)  # Add to list

                if type(self.Rfree) == list:
                    bNext = self.Rfree[k] * aNext
                else:
                    bNext = self.Rfree * aNext

                # Obtain shocks and shock probabilities from income distribution this period
                shk_prbs = shk_dstn[k].pmv  # Probability of shocks this period
                # Transitory shocks this period
                tran_shks = shk_dstn[k].atoms[1]
                # Permanent shocks this period
                perm_shks = shk_dstn[k].atoms[0]
                # Update probability of staying alive this period
                LivPrb = self.LivPrb[k]

                if len(dist_pGrid) == 1:
                    # New borns have this distribution (assumes start with no assets and permanent income=1)
                    NewBornDist = jump_to_grid_1D(
                        np.ones_like(tran_shks), shk_prbs, dist_mGrid
                    )
                    # Compute Transition Matrix given shocks and grids.
                    TranMatrix_M = gen_tran_matrix_1D(
                        dist_mGrid,
                        bNext,
                        shk_prbs,
                        perm_shks,
                        tran_shks,
                        LivPrb,
                        NewBornDist,
                    )
                    self.tran_matrix.append(TranMatrix_M)

                else:
                    NewBornDist = jump_to_grid_2D(
                        np.ones_like(tran_shks),
                        np.ones_like(tran_shks),
                        shk_prbs,
                        dist_mGrid,
                        dist_pGrid,
                    )
                    # Compute Transition Matrix given shocks and grids.
                    TranMatrix = gen_tran_matrix_2D(
                        dist_mGrid,
                        dist_pGrid,
                        bNext,
                        shk_prbs,
                        perm_shks,
                        tran_shks,
                        LivPrb,
                        NewBornDist,
                    )
                    self.tran_matrix.append(TranMatrix)

    def calc_ergodic_dist(self, transition_matrix=None):
        """
        Calculates the ergodic distribution across normalized market resources and
        permanent income as the eigenvector associated with the eigenvalue 1.
        The distribution is stored as attributes of self both as a vector and as a reshaped array with the ij'th element representing
        the probability of being at the i'th point on the mGrid and the j'th
        point on the pGrid.

        Parameters
        ----------
        transition_matrix: List
                    list with one transition matrix whose ergordic distribution is to be solved
        Returns
        -------
        None
        """

        if not isinstance(transition_matrix, list):
            transition_matrix = [self.tran_matrix]

        eigen, ergodic_distr = sp.linalg.eigs(
            transition_matrix[0], v0=np.ones(len(transition_matrix[0])), k=1, which="LM"
        )  # Solve for ergodic distribution
        ergodic_distr = ergodic_distr.real / np.sum(ergodic_distr.real)

        self.vec_erg_dstn = ergodic_distr  # distribution as a vector
        # distribution reshaped into len(mgrid) by len(pgrid) array
        self.erg_dstn = ergodic_distr.reshape(
            (len(self.dist_mGrid), len(self.dist_pGrid))
        )

    def compute_steady_state(self):
        # Compute steady state to perturb around
        self.cycles = 0
        self.solve()

        # Use Harmenberg Measure
        self.neutral_measure = True
        self.update_income_process()

        # Non stochastic simuation
        self.define_distribution_grid()
        self.calc_transition_matrix()

        self.c_ss = self.cPol_Grid  # Normalized Consumption Policy grid
        self.a_ss = self.aPol_Grid  # Normalized Asset Policy grid

        self.calc_ergodic_dist()  # Calculate ergodic distribution
        # Steady State Distribution as a vector (m*p x 1) where m is the number of gridpoints on the market resources grid
        ss_dstn = self.vec_erg_dstn

        self.A_ss = np.dot(self.a_ss, ss_dstn)[0]
        self.C_ss = np.dot(self.c_ss, ss_dstn)[0]

        return self.A_ss, self.C_ss

    def calc_jacobian(self, shk_param, T):
        """
        Calculates the Jacobians of aggregate consumption and aggregate assets.
        Parameters that can be shocked are LivPrb, PermShkStd,TranShkStd, DiscFac,
        UnempPrb, Rfree, IncUnemp, and DiscFac.

        Parameters:
        -----------

        shk_param: string
            name of variable to be shocked

        T: int
            dimension of Jacobian Matrix. Jacobian Matrix is a TxT square Matrix


        Returns
        ----------
        CJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Consumption with respect to shk_param

        AJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Assets with respect to shk_param

        """

        # Set up finite Horizon dictionary
        params = deepcopy(self.__dict__["parameters"])
        params["T_cycle"] = T  # Dimension of Jacobian Matrix

        # Specify a dictionary of lists because problem we are solving is
        # technically finite horizon so variables can be time varying (see
        # section on fake news algorithm in
        # https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434 )
        params["LivPrb"] = params["T_cycle"] * [self.LivPrb[0]]
        params["PermGroFac"] = params["T_cycle"] * [self.PermGroFac[0]]
        params["PermShkStd"] = params["T_cycle"] * [self.PermShkStd[0]]
        params["TranShkStd"] = params["T_cycle"] * [self.TranShkStd[0]]
        params["Rfree"] = params["T_cycle"] * [self.Rfree]
        params["UnempPrb"] = params["T_cycle"] * [self.UnempPrb]
        params["IncUnemp"] = params["T_cycle"] * [self.IncUnemp]

        # Create instance of a finite horizon agent
        FinHorizonAgent = IndShockConsumerType(**params)
        FinHorizonAgent.cycles = 1  # required

        # delete Rfree from time invariant list since it varies overtime
        FinHorizonAgent.del_from_time_inv("Rfree")
        # Add Rfree to time varying list to be able to introduce time varying interest rates
        FinHorizonAgent.add_to_time_vary("Rfree")

        # Set Terminal Solution as Steady State Consumption Function
        FinHorizonAgent.solution_terminal = deepcopy(self.solution[0])

        dx = 0.0001  # Size of perturbation
        # Period in which the change in the interest rate occurs (second to last period)
        i = params["T_cycle"] - 1

        FinHorizonAgent.IncShkDstn = params["T_cycle"] * [self.IncShkDstn[0]]

        # If parameter is in time invariant list then add it to time vary list
        FinHorizonAgent.del_from_time_inv(shk_param)
        FinHorizonAgent.add_to_time_vary(shk_param)

        # this condition is because some attributes are specified as lists while other as floats
        if type(getattr(self, shk_param)) == list:
            perturbed_list = (
                (i) * [getattr(self, shk_param)[0]]
                + [getattr(self, shk_param)[0] + dx]
                + (params["T_cycle"] - i - 1) * [getattr(self, shk_param)[0]]
            )  # Sequence of interest rates the agent faces
        else:
            perturbed_list = (
                (i) * [getattr(self, shk_param)]
                + [getattr(self, shk_param) + dx]
                + (params["T_cycle"] - i - 1) * [getattr(self, shk_param)]
            )  # Sequence of interest rates the agent faces
        setattr(FinHorizonAgent, shk_param, perturbed_list)
        self.parameters[shk_param] = perturbed_list

        # Update income process if perturbed parameter enters the income shock distribution
        FinHorizonAgent.update_income_process()

        # Solve
        FinHorizonAgent.solve(run_presolve=False)

        # Use Harmenberg Neutral Measure
        FinHorizonAgent.neutral_measure = True
        FinHorizonAgent.update_income_process()

        # Calculate Transition Matrices
        FinHorizonAgent.define_distribution_grid()
        FinHorizonAgent.calc_transition_matrix()

        # Normalized consumption Policy Grids across time
        c_t = FinHorizonAgent.cPol_Grid
        a_t = FinHorizonAgent.aPol_Grid

        # Append steady state policy grid into list of policy grids as HARK does not provide the initial policy
        c_t.append(self.c_ss)
        a_t.append(self.a_ss)

        # Fake News Algorithm begins below ( To find fake news algorithm See page 2388 of https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434  )

        ##########
        # STEP 1 # of fake news algorithm, As in the paper for Curly Y and Curly D. Here the policies are over assets and consumption so we denote them as curly C and curly D.
        ##########
        a_ss = self.aPol_Grid  # steady state Asset Policy
        c_ss = self.cPol_Grid  # steady state Consumption Policy
        tranmat_ss = self.tran_matrix  # Steady State Transition Matrix

        # List of asset policies grids where households expect the shock to occur in the second to last Period
        a_t = FinHorizonAgent.aPol_Grid
        # add steady state assets to list as it does not get appended in calc_transition_matrix method
        a_t.append(self.a_ss)

        # List of consumption policies grids where households expect the shock to occur in the second to last Period
        c_t = FinHorizonAgent.cPol_Grid
        # add steady state consumption to list as it does not get appended in calc_transition_matrix method
        c_t.append(self.c_ss)

        da0_s = []  # Deviation of asset policy from steady state policy
        dc0_s = []  # Deviation of Consumption policy from steady state policy
        for i in range(T):
            da0_s.append(a_t[T - i] - a_ss)
            dc0_s.append(c_t[T - i] - c_ss)

        da0_s = np.array(da0_s)
        dc0_s = np.array(dc0_s)

        # Steady state distribution of market resources (permanent income weighted distribution)
        D_ss = self.vec_erg_dstn.T[0]
        dA0_s = []
        dC0_s = []
        for i in range(T):
            dA0_s.append(np.dot(da0_s[i], D_ss))
            dC0_s.append(np.dot(dc0_s[i], D_ss))

        dA0_s = np.array(dA0_s)
        # This is equivalent to the curly Y scalar detailed in the first step of the algorithm
        A_curl_s = dA0_s / dx

        dC0_s = np.array(dC0_s)
        C_curl_s = dC0_s / dx

        # List of computed transition matrices for each period
        tranmat_t = FinHorizonAgent.tran_matrix
        tranmat_t.append(tranmat_ss)

        # List of change in transition matrix relative to the steady state transition matrix
        dlambda0_s = []
        for i in range(T):
            dlambda0_s.append(tranmat_t[T - i] - tranmat_ss)

        dlambda0_s = np.array(dlambda0_s)

        dD0_s = []
        for i in range(T):
            dD0_s.append(np.dot(dlambda0_s[i], D_ss))

        dD0_s = np.array(dD0_s)
        D_curl_s = dD0_s / dx  # Curly D in the sequence space jacobian

        ########
        # STEP2 # of fake news algorithm
        ########

        # Expectation Vectors
        exp_vecs_a = []
        exp_vecs_c = []

        # First expectation vector is the steady state policy
        exp_vec_a = a_ss
        exp_vec_c = c_ss
        for i in range(T):
            exp_vecs_a.append(exp_vec_a)
            exp_vec_a = np.dot(tranmat_ss.T, exp_vec_a)

            exp_vecs_c.append(exp_vec_c)
            exp_vec_c = np.dot(tranmat_ss.T, exp_vec_c)

        # Turn expectation vectors into arrays
        exp_vecs_a = np.array(exp_vecs_a)
        exp_vecs_c = np.array(exp_vecs_c)

        #########
        # STEP3 # of the algorithm. In particular equation 26 of the published paper.
        #########
        # Fake news matrices
        Curl_F_A = np.zeros((T, T))  # Fake news matrix for assets
        Curl_F_C = np.zeros((T, T))  # Fake news matrix for consumption

        # First row of Fake News Matrix
        Curl_F_A[0] = A_curl_s
        Curl_F_C[0] = C_curl_s

        for i in range(T - 1):
            for j in range(T):
                Curl_F_A[i + 1][j] = np.dot(exp_vecs_a[i], D_curl_s[j])
                Curl_F_C[i + 1][j] = np.dot(exp_vecs_c[i], D_curl_s[j])

        ########
        # STEP4 #  of the algorithm
        ########

        # Function to compute jacobian matrix from fake news matrix
        def J_from_F(F):
            J = F.copy()
            for t in range(1, F.shape[0]):
                J[1:, t] += J[:-1, t - 1]
            return J

        J_A = J_from_F(Curl_F_A)
        J_C = J_from_F(Curl_F_C)

        ########
        # Additional step due to compute Zeroth Column of the Jacobian
        ########

        params = deepcopy(self.__dict__["parameters"])
        params["T_cycle"] = 2  # Dimension of Jacobian Matrix

        params["LivPrb"] = params["T_cycle"] * [self.LivPrb[0]]
        params["PermGroFac"] = params["T_cycle"] * [self.PermGroFac[0]]
        params["PermShkStd"] = params["T_cycle"] * [self.PermShkStd[0]]
        params["TranShkStd"] = params["T_cycle"] * [self.TranShkStd[0]]
        params["Rfree"] = params["T_cycle"] * [self.Rfree]
        params["UnempPrb"] = params["T_cycle"] * [self.UnempPrb]
        params["IncUnemp"] = params["T_cycle"] * [self.IncUnemp]
        params["IncShkDstn"] = params["T_cycle"] * [self.IncShkDstn[0]]
        params["cFunc_terminal_"] = deepcopy(self.solution[0].cFunc)

        # Create instance of a finite horizon agent for calculation of zeroth
        ZerothColAgent = IndShockConsumerType(**params)
        ZerothColAgent.cycles = 1  # required

        # If parameter is in time invariant list then add it to time vary list
        ZerothColAgent.del_from_time_inv(shk_param)
        ZerothColAgent.add_to_time_vary(shk_param)

        # Update income process if perturbed parameter enters the income shock distribution
        ZerothColAgent.update_income_process()

        # Solve
        ZerothColAgent.solve()

        # this condition is because some attributes are specified as lists while other as floats
        if type(getattr(self, shk_param)) == list:
            perturbed_list = [getattr(self, shk_param)[0] + dx] + (
                params["T_cycle"] - 1
            ) * [
                getattr(self, shk_param)[0]
            ]  # Sequence of interest rates the agent faces
        else:
            perturbed_list = [getattr(self, shk_param) + dx] + (
                params["T_cycle"] - 1
            ) * [getattr(self, shk_param)]
            # Sequence of interest rates the agent

        setattr(ZerothColAgent, shk_param, perturbed_list)  # Set attribute to agent
        self.parameters[shk_param] = perturbed_list

        # Use Harmenberg Neutral Measure
        ZerothColAgent.neutral_measure = True
        ZerothColAgent.update_income_process()

        # Calculate Transition Matrices
        ZerothColAgent.define_distribution_grid()
        ZerothColAgent.calc_transition_matrix()

        tranmat_t_zeroth_col = ZerothColAgent.tran_matrix
        dstn_t_zeroth_col = self.vec_erg_dstn.T[0]

        C_t_no_sim = np.zeros(T)
        A_t_no_sim = np.zeros(T)

        for i in range(T):
            if i == 0:
                dstn_t_zeroth_col = np.dot(tranmat_t_zeroth_col[i], dstn_t_zeroth_col)
            else:
                dstn_t_zeroth_col = np.dot(tranmat_ss, dstn_t_zeroth_col)

            C_t_no_sim[i] = np.dot(self.cPol_Grid, dstn_t_zeroth_col)
            A_t_no_sim[i] = np.dot(self.aPol_Grid, dstn_t_zeroth_col)

        J_A.T[0] = (A_t_no_sim - self.A_ss) / dx
        J_C.T[0] = (C_t_no_sim - self.C_ss) / dx

        return J_C, J_A

    def make_euler_error_func(self, mMax=100, approx_inc_dstn=True):
        """
        Creates a "normalized Euler error" function for this instance, mapping
        from market resources to "consumption error per dollar of consumption."
        Stores result in attribute eulerErrorFunc as an interpolated function.
        Has option to use approximate income distribution stored in self.IncShkDstn
        or to use a (temporary) very dense approximation.

        Only works on (one period) infinite horizon models at this time, will
        be generalized later.

        Parameters
        ----------
        mMax : float
            Maximum normalized market resources for the Euler error function.
        approx_inc_dstn : Boolean
            Indicator for whether to use the approximate discrete income distri-
            bution stored in self.IncShkDstn[0], or to use a very accurate
            discrete approximation instead.  When True, uses approximation in
            IncShkDstn; when False, makes and uses a very dense approximation.

        Returns
        -------
        None

        Notes
        -----
        This method is not used by any other code in the library. Rather, it is here
        for expository and benchmarking purposes.
        """
        # Get the income distribution (or make a very dense one)
        if approx_inc_dstn:
            IncShkDstn = self.IncShkDstn[0]
        else:
            TranShkDstn = MeanOneLogNormal(sigma=self.TranShkStd[0]).discretize(
                N=200,
                method="equiprobable",
                tail_N=50,
                tail_order=1.3,
                tail_bound=[0.05, 0.95],
            )
            TranShkDstn = add_discrete_outcome_constant_mean(
                TranShkDstn, self.UnempPrb, self.IncUnemp
            )
            PermShkDstn = MeanOneLogNormal(sigma=self.PermShkStd[0]).discretize(
                N=200,
                method="equiprobable",
                tail_N=50,
                tail_order=1.3,
                tail_bound=[0.05, 0.95],
            )
            IncShkDstn = combine_indep_dstns(PermShkDstn, TranShkDstn)

        # Make a grid of market resources
        mNowMin = self.solution[0].mNrmMin + 10 ** (
            -15
        )  # add tiny bit to get around 0/0 problem
        mNowMax = mMax
        mNowGrid = np.linspace(mNowMin, mNowMax, 1000)

        # Get the consumption function this period and the marginal value function
        # for next period.  Note that this part assumes a one period cycle.
        cFuncNow = self.solution[0].cFunc
        vPfuncNext = self.solution[0].vPfunc

        # Calculate consumption this period at each gridpoint (and assets)
        cNowGrid = cFuncNow(mNowGrid)
        aNowGrid = mNowGrid - cNowGrid

        # Tile the grids for fast computation
        ShkCount = IncShkDstn[0].size
        aCount = aNowGrid.size
        aNowGrid_tiled = np.tile(aNowGrid, (ShkCount, 1))
        PermShkVals_tiled = (np.tile(IncShkDstn[1], (aCount, 1))).transpose()
        TranShkVals_tiled = (np.tile(IncShkDstn[2], (aCount, 1))).transpose()
        ShkPrbs_tiled = (np.tile(IncShkDstn[0], (aCount, 1))).transpose()

        # Calculate marginal value next period for each gridpoint and each shock
        mNextArray = (
            self.Rfree / (self.PermGroFac[0] * PermShkVals_tiled) * aNowGrid_tiled
            + TranShkVals_tiled
        )
        vPnextArray = vPfuncNext(mNextArray)

        # Calculate expected marginal value and implied optimal consumption
        ExvPnextGrid = (
            self.DiscFac
            * self.Rfree
            * self.LivPrb[0]
            * self.PermGroFac[0] ** (-self.CRRA)
            * np.sum(
                PermShkVals_tiled ** (-self.CRRA) * vPnextArray * ShkPrbs_tiled, axis=0
            )
        )
        cOptGrid = ExvPnextGrid ** (
            -1.0 / self.CRRA
        )  # This is the 'Endogenous Gridpoints' step

        # Calculate Euler error and store an interpolated function
        EulerErrorNrmGrid = (cNowGrid - cOptGrid) / cOptGrid
        eulerErrorFunc = LinearInterp(mNowGrid, EulerErrorNrmGrid)
        self.eulerErrorFunc = eulerErrorFunc

    def pre_solve(self):
        self.update_solution_terminal()
        if not self.quiet:
            self.check_conditions(verbose=self.verbose)

    def describe_parameters(self):
        """
        Generate a string describing the primitive model parameters that will
        be used to calculating limiting values and factors.

        Parameters
        ----------
        None

        Returns
        -------
        param_desc : str
            Description of primitive parameters.
        """
        # Get parameter description from the perfect foresight model
        param_desc = super().describe_parameters()

        # Make a new entry for weierstrass-p (the weird formatting here is to
        # make it easier to adapt into the style of the superclass if we add more
        # parameter reports later)
        this_entry = [
            "WorstPrb",
            "probability of worst income shock realization",
            "℘",
            False,
        ]
        try:
            val = getattr(self, this_entry[0])
        except:
            val = self.bilt[this_entry[0]]
        this_line = (
            this_entry[2]
            + f"={val:.5f} : "
            + this_entry[1]
            + " ("
            + this_entry[0]
            + ")\n"
        )

        # Add in the new entry and return it
        param_desc += this_line
        return param_desc

    def calc_limiting_values(self):
        """
        Compute various scalar values that are relevant to characterizing the
        solution to an infinite horizon problem. This method should only be called
        when T_cycle=1 and cycles=0, otherwise the values generated are meaningless.
        This method adds the following values to this instance in the dictionary
        attribute called bilt.

        APFac : Absolute Patience Factor
        GPFacRaw : Growth Patience Factor
        GPFacMod : Risk-Modified Growth Patience Factor
        GPFacLiv : Mortality-Adjusted Growth Patience Factor
        GPFacLivMod : Modigliani Mortality-Adjusted Growth Patience Factor
        GPFacSdl : Szeidl Growth Patience Factor
        FHWFac : Finite Human Wealth Factor
        RPFac : Return Patience Factor
        WRPFac : Weak Return Patience Factor
        PFVAFac : Perfect Foresight Value of Autarky Factor
        VAFac : Value of Autarky Factor
        cNrmPDV : Present Discounted Value of Autarky Consumption
        MPCmin : Limiting minimum MPC as market resources go to infinity
        MPCmax : Limiting maximum MPC as market resources approach minimum level
        hNrm : Human wealth divided by permanent income.
        ELogPermShk : Expected log permanent income shock
        WorstPrb : Probability of worst income shock realization
        Delta_mNrm_ZeroFunc : Linear locus where expected change in market resource ratio is zero
        BalGroFunc : Linear consumption function where the level of market resources grows at the same rate as permanent income

        Returns
        -------
        None
        """
        super().calc_limiting_values()
        aux_dict = self.bilt

        # Calculate the risk-modified growth impatience factor
        PermShkDstn = self.PermShkDstn[0]
        inv_func = lambda x: x ** (-1.0)
        Ex_PermShkInv = expected(inv_func, PermShkDstn)[0]
        GroCompPermShk = Ex_PermShkInv ** (-1.0)
        aux_dict["GPFacMod"] = aux_dict["APFac"] / (self.PermGroFac[0] * GroCompPermShk)

        # Calculate the mortality-adjusted growth impatience factor (and version
        # with Modigiliani bequests)
        aux_dict["GPFacLiv"] = aux_dict["GPFacRaw"] * self.LivPrb[0]
        aux_dict["GPFacLivMod"] = aux_dict["GPFacLiv"] * self.LivPrb[0]

        # Calculate the risk-modified value of autarky factor
        if self.CRRA == 1.0:
            UtilCompPermShk = np.exp(expected(np.log, PermShkDstn)[0])
        else:
            CRRAfunc = lambda x: x ** (1.0 - self.CRRA)
            UtilCompPermShk = expected(CRRAfunc, PermShkDstn)[0] ** (
                1 / (1.0 - self.CRRA)
            )
        aux_dict["VAFac"] = self.DiscFac * (self.PermGroFac[0] * UtilCompPermShk) ** (
            1.0 - self.CRRA
        )

        # Calculate the expected log permanent income shock, which will be used
        # for the Szeidl variation of the Growth Impatience condition
        aux_dict["ELogPermShk"] = expected(np.log, PermShkDstn)[0]

        # Calculate the Harmenberg permanent income neutral expected log permanent
        # shock and the Harmenberg Growth Patience Factor
        Hrm_func = lambda x: x * np.log(x)
        PermShk_Hrm = np.exp(expected(Hrm_func, PermShkDstn)[0])
        aux_dict["GPFacHrm"] = aux_dict["GPFacRaw"] / PermShk_Hrm

        # Calculate the probability of the worst income shock realization
        PermShkValsNext = self.IncShkDstn[0].atoms[0]
        TranShkValsNext = self.IncShkDstn[0].atoms[1]
        ShkPrbsNext = self.IncShkDstn[0].pmv
        Ex_IncNext = np.dot(ShkPrbsNext, PermShkValsNext * TranShkValsNext)
        PermShkMinNext = np.min(PermShkValsNext)
        TranShkMinNext = np.min(TranShkValsNext)
        WorstIncNext = PermShkMinNext * TranShkMinNext
        WorstIncPrb = np.sum(
            ShkPrbsNext[(PermShkValsNext * TranShkValsNext) == WorstIncNext]
        )
        aux_dict["WorstPrb"] = WorstIncPrb

        # Calculate the weak return patience factor
        aux_dict["WRPFac"] = WorstIncPrb ** (1.0 / self.CRRA) * aux_dict["RPFac"]

        # Calculate human wealth and the infinite horizon natural borrowing constraint
        if aux_dict["FHWFac"] < 1.0:
            hNrm = Ex_IncNext / (1.0 - aux_dict["FHWFac"])
        else:
            hNrm = np.inf
        temp = PermShkMinNext * aux_dict["FHWFac"]
        BoroCnstNat = -TranShkMinNext * temp / (1.0 - temp)

        # Find the upper bound of the MPC as market resources approach the minimum
        BoroCnstArt = -np.inf if self.BoroCnstArt is None else self.BoroCnstArt
        if BoroCnstNat < BoroCnstArt:
            MPCmax = 1.0  # if natural borrowing constraint is overridden by artificial one, MPCmax is 1
        else:
            MPCmax = 1.0 - WorstIncPrb ** (1.0 / self.CRRA) * aux_dict["RPFac"]
            MPCmax = np.maximum(MPCmax, 0.0)

        # Store maximum MPC and human wealth
        aux_dict["hNrm"] = hNrm
        aux_dict["MPCmax"] = MPCmax

        # Generate the "Delta m = 0" function, which is used to find target market resources
        # This overwrites the function generated by the perfect foresight version
        Ex_Rnrm = self.Rfree / self.PermGroFac[0] * Ex_PermShkInv
        aux_dict["Delta_mNrm_ZeroFunc"] = (
            lambda m: (1.0 - 1.0 / Ex_Rnrm) * m + 1.0 / Ex_Rnrm
        )

        self.bilt = aux_dict

        self.bilt = aux_dict

    def check_GICMod(self, verbose=None):
        """
        Evaluate and report on the Risk-Modified Growth Impatience Condition.
        """
        name = "GICMod"
        GPFacMod = self.bilt["GPFacMod"]
        result = GPFacMod < 1.0

        messages = {
            True: f"GPFacMod={GPFacMod:.5f} : The Risk-Modified Growth Patience Factor satisfies the Risk-Modified Growth Impatience Condition (GICMod) Þ/(G‖Ψ‖_(-1)) < 1.",
            False: f"GPFacMod={GPFacMod:.5f} : The Risk-Modified Growth Patience Factor violates the Risk-Modified Growth Impatience Condition (GICMod) Þ/(G‖Ψ‖_(-1)) < 1.",
        }
        verbose = self.verbose if verbose is None else verbose
        self.log_condition_result(name, result, messages[result], verbose)

    def check_GICSdl(self, verbose=None):
        """
        Evaluate and report on the Szeidl variation of the Growth Impatience Condition.
        """
        name = "GICSdl"
        ELogPermShk = self.bilt["ELogPermShk"]
        result = np.log(self.bilt["GPFacRaw"]) < ELogPermShk

        messages = {
            True: f"E[log Ψ]={ELogPermShk:.5f} : The expected log permanent income shock satisfies the Szeidl Growth Impatience Condition (GICSdl) log(Þ/G) < E[log Ψ].",
            False: f"E[log Ψ]={ELogPermShk:.5f} : The expected log permanent income shock violates the Szeidl Growth Impatience Condition (GICSdl) log(Þ/G) < E[log Ψ].",
        }
        verbose = self.verbose if verbose is None else verbose
        self.log_condition_result(name, result, messages[result], verbose)

    def check_GICHrm(self, verbose=None):
        """
        Evaluate and report on the Harmenberg variation of the Growth Impatience Condition.
        """
        name = "GICHrm"
        GPFacHrm = self.bilt["GPFacHrm"]
        result = GPFacHrm < 1.0

        messages = {
            True: f"GPFacHrm={GPFacHrm:.5f} : The Harmenberg Expected Growth Patience Factor satisfies the Harmenberg Growth Normalized Impatience Condition (GICHrm) Þ/G < exp(E[Ψlog Ψ]).",
            False: f"GPFacHrm={GPFacHrm:.5f} : The Harmenberg Expected Growth Patience Factor violates the Harmenberg Growth Normalized Impatience Condition (GICHrm) Þ/G < exp(E[Ψlog Ψ]).",
        }
        verbose = self.verbose if verbose is None else verbose
        self.log_condition_result(name, result, messages[result], verbose)

    def check_GICLiv(self, verbose=None):
        """
        Evaluate and report on the Mortality-Adjusted Growth Impatience Condition.
        """
        name = "GICLiv"
        GPFacLiv = self.bilt["GPFacLiv"]
        result = GPFacLiv < 1.0

        messages = {
            True: f"GPFacLiv={GPFacLiv:.5f} : The Mortality-Adjusted Growth Patience Factor satisfies the Mortality-Adjusted Growth Impatience Condition (GICLiv) ℒÞ/G < 1.",
            False: f"GPFacLiv={GPFacLiv:.5f} : The Mortality-Adjusted Growth Patience Factor violates the Mortality-Adjusted Growth Impatience Condition (GICLiv) ℒÞ/G < 1.",
        }
        verbose = self.verbose if verbose is None else verbose
        self.log_condition_result(name, result, messages[result], verbose)

    def check_FVAC(self, verbose=None):
        """
        Evaluate and report on the Finite Value of Autarky condition in the presence of income risk.
        """
        name = "FVAC"
        VAFac = self.bilt["VAFac"]
        result = VAFac < 1.0

        messages = {
            True: f"VAFac={VAFac:.5f} : The Risk-Modified Finite Value of Autarky Factor satisfies the Risk-Modified Finite Value of Autarky Condition β(G‖Ψ‖_(1-ρ))^(1-ρ) < 1.",
            False: f"VAFac={VAFac:.5f} : The Risk-Modified Finite Value of Autarky Factor violates the Risk-Modified Finite Value of Autarky Condition β(G‖Ψ‖_(1-ρ))^(1-ρ) < 1.",
        }
        verbose = self.verbose if verbose is None else verbose
        self.log_condition_result(name, result, messages[result], verbose)

    def check_WRIC(self, verbose=None):
        """
        Evaluate and report on the Weak Return Impatience Condition.
        """
        name = "WRIC"
        WRPFac = self.bilt["WRPFac"]
        result = WRPFac < 1.0

        messages = {
            True: f"WRPFac={WRPFac:.5f} : The Weak Return Patience Factor satisfies the Weak Return Impatience Condition (WRIC) ℘ Þ/R < 1.",
            False: f"WRPFac={WRPFac:.5f} : The Weak Return Patience Factor violates the Weak Return Impatience Condition (WRIC) ℘ Þ/R < 1.",
        }
        verbose = self.verbose if verbose is None else verbose
        self.log_condition_result(name, result, messages[result], verbose)

    def check_conditions(self, verbose=None):
        """
        This method checks whether the instance's type satisfies various conditions.
        When combinations of these conditions are satisfied, the solution to the
        problem exhibits different characteristics.  (For an exposition of the
        conditions, see https://econ-ark.github.io/BufferStockTheory/)

        Parameters
        ----------
        verbose : boolean
            Specifies different levels of verbosity of feedback. When False, it only reports whether the
            instance's type fails to satisfy a particular condition. When True, it reports all results, i.e.
            the factor values for all conditions.

        Returns
        -------
        None
        """
        self.conditions = {}
        self.bilt["conditions_report"] = ""
        self.degenerate = False
        verbose = self.verbose if verbose is None else verbose

        # This method only checks for the conditions for infinite horizon models
        # with a 1 period cycle. If these conditions are not met, we exit early.
        if self.cycles != 0 or self.T_cycle > 1:
            trivial_message = "No conditions report was produced because this functionality is only supported for infinite horizon models with a cycle length of 1."
            self.log_condition_result(None, None, trivial_message, verbose)
            if not self.quiet:
                _log.info(self.bilt["conditions_report"])
            return

        # Calculate some useful quantities that will be used in the condition checks
        self.calc_limiting_values()
        param_desc = self.describe_parameters()
        self.log_condition_result(None, None, param_desc, verbose)

        # Check individual conditions and add their results to the report
        self.check_AIC(verbose)
        self.check_RIC(verbose)
        self.check_WRIC(verbose)
        self.check_GICRaw(verbose)
        self.check_GICMod(verbose)
        self.check_GICLiv(verbose)
        self.check_GICSdl(verbose)
        self.check_GICHrm(verbose)
        super().check_FVAC(verbose)
        self.check_FVAC(verbose)
        self.check_FHWC(verbose)

        # Exit now if verbose output was not requested.
        if not verbose:
            if not self.quiet:
                _log.info(self.bilt["conditions_report"])
            return

        # Report on the degeneracy of the consumption function solution
        if self.conditions["WRIC"] and self.conditions["FVAC"]:
            degen_message = "\nBecause both the WRIC and FVAC are satisfied, the recursive solution to the infinite horizon problem represents a contraction mapping on the consumption function. Thus a non-degenerate solution exists."
            degenerate = False
        elif not self.conditions["WRIC"]:
            degen_message = "\nBecause the WRIC is violated, the consumer is so pathologically patient that they will never consume at all. Thus the solution will be degenerate at c(m) = 0 for all m.\n"
            degenerate = True
        elif not self.conditions["FVAC"]:
            degen_message = "\nBecause the FVAC is violated, the recursive solution to the infinite horizon problem might not be a contraction mapping, so the produced solution might not be valid. Proceed with caution."
            degenerate = False
        self.log_condition_result(None, None, degen_message, verbose)
        self.degenerate = degenerate

        # Stop here if the solution is degenerate
        if degenerate:
            if not self.quiet:
                _log.info(self.bilt["conditions_report"])
            return

        # Report on the limiting behavior of the consumption function as m goes to infinity
        if self.conditions["RIC"]:
            if self.conditions["FHWC"]:
                RIC_message = "\nBecause both the RIC and FHWC condition are satisfied, the consumption function will approach the linear perfect foresight solution as m becomes arbitrarily large."
            else:
                RIC_message = "\nBecause the RIC is satisfied but the FHWC is violated, the GIC is satisfied."
        else:
            RIC_message = "\nBecause the RIC is violated, the FHWC condition is also violated. The consumer is pathologically impatient but has infinite expected future earnings. Thus the consumption function will not approach any linear limit as m becomes arbitrarily large, and the MPC will asymptote to zero."
        self.log_condition_result(None, None, RIC_message, verbose)

        # Report on whether a pseudo-steady-state exists at the individual level
        if self.conditions["GICRaw"]:
            GIC_message = "\nBecause the GICRaw is satisfied, there exists a pseudo-steady-state wealth ratio at which the level of wealth is expected to grow at the same rate as permanent income."
        else:
            GIC_message = "\nBecause the GICRaw is violated, there might not exist a pseudo-steady-state wealth ratio at which the level of wealth is expected to grow at the same rate as permanent income."
        self.log_condition_result(None, None, GIC_message, verbose)

        # Report on whether a target wealth ratio exists at the individual level
        if self.conditions["GICMod"]:
            GICMod_message = "\nBecause the GICMod is satisfied, expected growth of the ratio of market resources to permanent income is less than one as market resources become arbitrarily large. Hence the consumer has a target ratio of market resources to permanent income."
        else:
            GICMod_message = "\nBecause the GICMod is violated, expected growth of the ratio of market resources to permanent income exceeds one as market resources go to infinity. Hence the consumer might not have a target ratio of market resources to permanent income."
        self.log_condition_result(None, None, GICMod_message, verbose)

        # Report on whether a target level of wealth exists at the aggregate level
        if self.conditions["GICLiv"]:
            GICLiv_message = "\nBecause the GICLiv is satisfied, a target ratio of aggregate market resources to aggregate permanent income exists."
        else:
            GICLiv_message = "\nBecause the GICLiv is violated, a target ratio of aggregate market resources to aggregate permanent income might not exist."
        self.log_condition_result(None, None, GICLiv_message, verbose)

        # Report on whether invariant distributions exist
        if self.conditions["GICSdl"]:
            GICSdl_message = "\nBecause the GICSdl is satisfied, there exist invariant distributions of permanent income-normalized variables."
        else:
            GICSdl_message = "\nBecause the GICSdl is violated, there do not exist invariant distributions of permanent income-normalized variables."
        self.log_condition_result(None, None, GICSdl_message, verbose)

        # Report on whether blah blah
        if self.conditions["GICHrm"]:
            GICHrm_message = "\nBecause the GICHrm is satisfied, there exists a target ratio of the individual market resources to permanent income, under the permanent-income-neutral measure."
        else:
            GICHrm_message = "\nBecause the GICHrm is violated, there does not exist a target ratio of the individual market resources to permanent income, under the permanent-income-neutral measure.."
        self.log_condition_result(None, None, GICHrm_message, verbose)

        if not self.quiet:
            _log.info(self.bilt["conditions_report"])


###############################################################################

# Specify default parameters that differ in "kinked R" model compared to base IndShockConsumerType
kinked_R_different_params = {
    "Rboro": 1.20,  # Interest factor on assets when borrowing, a < 0
    "Rsave": 1.02,  # Interest factor on assets when saving, a > 0
    "BoroCnstArt": None,  # Kinked R only matters if borrowing is allowed
}

# Make a dictionary to specify a "kinked R" idiosyncratic shock consumer
init_kinked_R = init_idiosyncratic_shocks.copy()  # See base dictionary above
init_kinked_R.update(kinked_R_different_params)  # Update with some parameters
del init_kinked_R["Rfree"]  # Get rid of constant interest factor


class KinkedRconsumerType(IndShockConsumerType):
    """
    A consumer type that faces idiosyncratic shocks to income and has a different
    interest factor on saving vs borrowing.  Extends IndShockConsumerType, with
    very small changes.  Solver for this class is currently only compatible with
    linear spline interpolation.

    Same parameters as AgentType.


    Parameters
    ----------
    """

    time_inv_ = copy(IndShockConsumerType.time_inv_)
    time_inv_ += ["Rboro", "Rsave"]

    def __init__(self, **kwds):
        params = init_kinked_R.copy()
        params.update(kwds)

        # Initialize a basic AgentType
        super().__init__(**params)

        # Add consumer-type specific objects, copying to create independent versions
        self.solve_one_period = solve_one_period_ConsKinkedR

    def calc_bounding_values(self):
        """
        Calculate human wealth plus minimum and maximum MPC in an infinite
        horizon model with only one period repeated indefinitely.  Store results
        as attributes of self.  Human wealth is the present discounted value of
        expected future income after receiving income this period, ignoring mort-
        ality.  The maximum MPC is the limit of the MPC as m --> mNrmMin.  The
        minimum MPC is the limit of the MPC as m --> infty.  This version deals
        with the different interest rates on borrowing vs saving.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Unpack the income distribution and get average and worst outcomes
        PermShkValsNext = self.IncShkDstn[0][1]
        TranShkValsNext = self.IncShkDstn[0][2]
        ShkPrbsNext = self.IncShkDstn[0][0]
        Ex_IncNext = expected(lambda trans, perm: trans * perm, self.IncShkDstn)
        PermShkMinNext = np.min(PermShkValsNext)
        TranShkMinNext = np.min(TranShkValsNext)
        WorstIncNext = PermShkMinNext * TranShkMinNext
        WorstIncPrb = np.sum(
            ShkPrbsNext[(PermShkValsNext * TranShkValsNext) == WorstIncNext]
        )

        # Calculate human wealth and the infinite horizon natural borrowing constraint
        hNrm = (Ex_IncNext * self.PermGroFac[0] / self.Rsave) / (
            1.0 - self.PermGroFac[0] / self.Rsave
        )
        temp = self.PermGroFac[0] * PermShkMinNext / self.Rboro
        BoroCnstNat = -TranShkMinNext * temp / (1.0 - temp)

        PatFacTop = (self.DiscFac * self.LivPrb[0] * self.Rsave) ** (
            1.0 / self.CRRA
        ) / self.Rsave
        PatFacBot = (self.DiscFac * self.LivPrb[0] * self.Rboro) ** (
            1.0 / self.CRRA
        ) / self.Rboro
        if BoroCnstNat < self.BoroCnstArt:
            MPCmax = 1.0  # if natural borrowing constraint is overridden by artificial one, MPCmax is 1
        else:
            MPCmax = 1.0 - WorstIncPrb ** (1.0 / self.CRRA) * PatFacBot
        MPCmin = 1.0 - PatFacTop

        # Store the results as attributes of self
        self.hNrm = hNrm
        self.MPCmin = MPCmin
        self.MPCmax = MPCmax

    def make_euler_error_func(self, mMax=100, approx_inc_dstn=True):
        """
        Creates a "normalized Euler error" function for this instance, mapping
        from market resources to "consumption error per dollar of consumption."
        Stores result in attribute eulerErrorFunc as an interpolated function.
        Has option to use approximate income distribution stored in self.IncShkDstn
        or to use a (temporary) very dense approximation.

        SHOULD BE INHERITED FROM ConsIndShockModel

        Parameters
        ----------
        mMax : float
            Maximum normalized market resources for the Euler error function.
        approx_inc_dstn : Boolean
            Indicator for whether to use the approximate discrete income distri-
            bution stored in self.IncShkDstn[0], or to use a very accurate
            discrete approximation instead.  When True, uses approximation in
            IncShkDstn; when False, makes and uses a very dense approximation.

        Returns
        -------
        None

        Notes
        -----
        This method is not used by any other code in the library. Rather, it is here
        for expository and benchmarking purposes.
        """
        raise NotImplementedError()

    def get_Rfree(self):
        """
        Returns an array of size self.AgentCount with self.Rboro or self.Rsave in each entry, based
        on whether self.aNrmNow >< 0.

        Parameters
        ----------
        None

        Returns
        -------
        RfreeNow : np.array
             Array of size self.AgentCount with risk free interest rate for each agent.
        """
        RfreeNow = self.Rboro * np.ones(self.AgentCount)
        RfreeNow[self.state_prev["aNrm"] > 0] = self.Rsave
        return RfreeNow

    def check_conditions(self, verbose):
        """
        This empty method overwrites the version inherited from its parent class,
        IndShockConsumerType. The condition checks are not appropriate when Rfree
        has multiple values.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # raise NotImplementedError()

        pass


def apply_flat_income_tax(
    IncShkDstn, tax_rate, T_retire, unemployed_indices=None, transitory_index=2
):
    """
    Applies a flat income tax rate to all employed income states during the working
    period of life (those before T_retire).  Time runs forward in this function.

    Parameters
    ----------
    IncShkDstn : [distribution.Distribution]
        The discrete approximation to the income distribution in each time period.
    tax_rate : float
        A flat income tax rate to be applied to all employed income.
    T_retire : int
        The time index after which the agent retires.
    unemployed_indices : [int]
        Indices of transitory shocks that represent unemployment states (no tax).
    transitory_index : int
        The index of each element of IncShkDstn representing transitory shocks.

    Returns
    -------
    IncShkDstn_new : [distribution.Distribution]
        The updated income distributions, after applying the tax.
    """
    unemployed_indices = (
        unemployed_indices if unemployed_indices is not None else list()
    )
    IncShkDstn_new = deepcopy(IncShkDstn)
    i = transitory_index
    for t in range(len(IncShkDstn)):
        if t < T_retire:
            for j in range((IncShkDstn[t][i]).size):
                if j not in unemployed_indices:
                    IncShkDstn_new[t][i][j] = IncShkDstn[t][i][j] * (1 - tax_rate)
    return IncShkDstn_new


# Make a dictionary to specify a lifecycle consumer with a finite horizon

# Main calibration characteristics
birth_age = 25
death_age = 90
adjust_infl_to = 1992
# Use income estimates from Cagetti (2003) for High-school graduates
education = "HS"
income_calib = Cagetti_income[education]

# Income specification
income_params = parse_income_spec(
    age_min=birth_age,
    age_max=death_age,
    adjust_infl_to=adjust_infl_to,
    **income_calib,
    SabelhausSong=True,
)

# Initial distribution of wealth and permanent income
dist_params = income_wealth_dists_from_scf(
    base_year=adjust_infl_to, age=birth_age, education=education, wave=1995
)

# We need survival probabilities only up to death_age-1, because survival
# probability at death_age is 1.
liv_prb = parse_ssa_life_table(
    female=False, cross_sec=True, year=2004, min_age=birth_age, max_age=death_age - 1
)

# Parameters related to the number of periods implied by the calibration
time_params = parse_time_params(age_birth=birth_age, age_death=death_age)

# Update all the new parameters
init_lifecycle = copy(init_idiosyncratic_shocks)
init_lifecycle.update(time_params)
init_lifecycle.update(dist_params)
# Note the income specification overrides the pLvlInitMean from the SCF.
init_lifecycle.update(income_params)
init_lifecycle.update({"LivPrb": liv_prb})

# Make a dictionary to specify an infinite consumer with a four period cycle
init_cyclical = copy(init_idiosyncratic_shocks)
init_cyclical["PermGroFac"] = [1.1, 1.082251, 2.8, 0.3]
init_cyclical["PermShkStd"] = [0.1, 0.1, 0.1, 0.1]
init_cyclical["TranShkStd"] = [0.1, 0.1, 0.1, 0.1]
init_cyclical["LivPrb"] = 4 * [0.98]
init_cyclical["T_cycle"] = 4

"""
Classes to solve canonical consumption-saving models with idiosyncratic shocks
to income.  All models here assume CRRA utility with geometric discounting, no
bequest motive, and income shocks that are fully transitory or fully permanent.

It currently solves three types of models:
   1) A very basic "perfect foresight" consumption-savings model with no uncertainty.
   2) A consumption-savings model with risk over transitory and permanent income shocks.
   3) The model described in (2), with an interest rate for debt that differs
      from the interest rate for savings.

See NARK https://github.com/econ-ark/HARK/blob/master/docs/NARK/NARK.pdf for information on variable naming conventions.
See HARK documentation for mathematical descriptions of the models being solved.
"""

from copy import copy

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
from HARK.Calibration.life_tables.us_ssa.SSATools import parse_ssa_life_table
from HARK.Calibration.SCF.WealthIncomeDist.SCFDistTools import (
    income_wealth_dists_from_scf,
)
from HARK.distributions import (
    Lognormal,
    MeanOneLogNormal,
    Uniform,
    add_discrete_outcome_constant_mean,
    combine_indep_dstns,
    expected,
)
from HARK.interpolation import (
    LinearInterp,
    LowerEnvelope,
    MargMargValueFuncCRRA,
    MargValueFuncCRRA,
    ValueFuncCRRA,
)
from HARK.interpolation import CubicHermiteInterp as CubicInterp
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
from HARK.utilities import make_assets_grid
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
    r"""
    A class representing the solution of a single period of a consumption-saving
    problem.  The solution must include a consumption function and marginal
    value function.

    Here and elsewhere in the code, Nrm indicates that variables are normalized
    by permanent income.

    Parameters
    ----------
    cFunc : function
        The consumption function for this period, defined over normalized market
        resources: cNrm = cFunc(mNrm).
    vFunc : function
        The beginning-of-period value function for this period, defined over
        normalized market resources: vNrm = vFunc(mNrm).
    vPfunc : function
        The beginning-of-period marginal value function for this period,
        defined over normalized market resources: vNrmP = vPfunc(mNrm).
    vPPfunc : function
        The beginning-of-period marginal marginal value function for this
        period, defined over normalized market resources: vNrmPP = vPPfunc(mNrm).
    mNrmMin : float
        The minimum allowable normalized market resources for this period; the consump-
        tion function (etc) are undefined for m < mNrmMin.
    hNrm : float
        Normalized human wealth after receiving income this period: PDV of all future
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
            assert NullFunc().distance(self.cFunc) == 0, (
                "append_solution called incorrectly!"
            )

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
# == Functions for initializing newborns in consumption-saving models =
# =====================================================================


def make_lognormal_kNrm_init_dstn(kLogInitMean, kLogInitStd, kNrmInitCount, RNG):
    """
    Construct a lognormal distribution for (normalized) initial capital holdings
    of newborns, kNrm. This is the default constructor for kNrmInitDstn.

    Parameters
    ----------
    kLogInitMean : float
        Mean of log capital holdings for newborns.
    kLogInitStd : float
        Stdev of log capital holdings for newborns.
    kNrmInitCount : int
        Number of points in the discretization.
    RNG : np.random.RandomState
        Agent's internal RNG.

    Returns
    -------
    kNrmInitDstn : DiscreteDistribution
        Discretized distribution of initial capital holdings for newborns.
    """
    dstn = Lognormal(
        mu=kLogInitMean,
        sigma=kLogInitStd,
        seed=RNG.integers(0, 2**31 - 1),
    )
    kNrmInitDstn = dstn.discretize(kNrmInitCount)
    return kNrmInitDstn


def make_lognormal_pLvl_init_dstn(pLogInitMean, pLogInitStd, pLvlInitCount, RNG):
    """
    Construct a lognormal distribution for initial permanent income level of
    newborns, pLvl. This is the default constructor for pLvlInitDstn.

    Parameters
    ----------
    pLogInitMean : float
        Mean of log permanent income for newborns.
    pLogInitStd : float
        Stdev of log capital holdings for newborns.
    pLvlInitCount : int
        Number of points in the discretization.
    RNG : np.random.RandomState
        Agent's internal RNG.

    Returns
    -------
    pLvlInitDstn : DiscreteDistribution
        Discretized distribution of initial permanent income for newborns.
    """
    dstn = Lognormal(
        mu=pLogInitMean,
        sigma=pLogInitStd,
        seed=RNG.integers(0, 2**31 - 1),
    )
    pLvlInitDstn = dstn.discretize(pLvlInitCount)
    return pLvlInitDstn


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


def calc_worst_inc_prob(inc_shk_dstn, use_infimum=False):
    """Calculate the probability of the worst income shock.

    Args:
        inc_shk_dstn (DiscreteDistribution): Distribution of shocks to income.
        use_infimum (bool): Indicator for whether to try to use the infimum of the limiting (true) income distribution.
    """
    probs = inc_shk_dstn.pmv
    perm, tran = inc_shk_dstn.atoms
    income = perm * tran
    if use_infimum:
        worst_inc = np.prod(inc_shk_dstn.limit["infimum"])
    else:
        worst_inc = np.min(income)
    return np.sum(probs[income == worst_inc])


def calc_boro_const_nat(
    m_nrm_min_next, inc_shk_dstn, rfree, perm_gro_fac, use_infimum=False
):
    """Calculate the natural borrowing constraint.

    Args:
        m_nrm_min_next (float): Minimum normalized market resources next period.
        inc_shk_dstn (DiscreteDstn): Distribution of shocks to income.
        rfree (float): Risk free interest factor.
        perm_gro_fac (float): Permanent income growth factor.
        use_infimum (bool): Indicator for whether to use the infimum of the limiting (true) income distribution
    """
    if use_infimum:
        perm_min, tran_min = inc_shk_dstn.limit["infimum"]
    else:
        perm, tran = inc_shk_dstn.atoms
        perm_min = np.min(perm)
        tran_min = np.min(tran)

    temp_fac = (perm_gro_fac * perm_min) / rfree
    boro_cnst_nat = (m_nrm_min_next - tran_min) * temp_fac
    return boro_cnst_nat


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
        An indicator for whether the solver should use cubic or linear interpolation.

    Returns
    -------
    solution_now : ConsumerSolution
        Solution to this period's consumption-saving problem with income risk.

    """
    # Verifiy that there is actually a kink in the interest factor
    assert Rboro >= Rsave, (
        "Interest factor on debt less than interest factor on savings!"
    )
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
    WorstIncPrb = calc_worst_inc_prob(IncShkDstn, use_infimum=False)
    # WorstIncPrb is the "Weierstrass p" concept: the odds we get the WORST thing
    Ex_IncNext = expected(lambda x: x["PermShk"] * x["TranShk"], IncShkDstn)
    hNrmNow = calc_human_wealth(solution_next.hNrm, PermGroFac, Rsave, Ex_IncNext)

    # Unpack next period's (marginal) value function
    vFuncNext = solution_next.vFunc  # This is None when vFuncBool is False
    vPfuncNext = solution_next.vPfunc
    vPPfuncNext = solution_next.vPPfunc  # This is None when CubicBool is False

    # Calculate the minimum allowable value of money resources in this period
    BoroCnstNat = calc_boro_const_nat(
        solution_next.mNrmMin,
        IncShkDstn,
        Rboro,
        PermGroFac,
        use_infimum=False,
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
        np.hstack((np.asarray(aXtraGrid) + mNrmMinNow, np.array([0.0, 1e-15]))),
    )

    # Make a 1D array of the interest factor at each asset gridpoint
    Rfree = Rsave * np.ones_like(aNrmNow)
    Rfree[aNrmNow <= 0] = Rboro
    i_kink = np.argwhere(aNrmNow == 0.0)[0][0]

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
PerfForesightConsumerType_constructors_default = {
    "solution_terminal": make_basic_CRRA_solution_terminal,
    "kNrmInitDstn": make_lognormal_kNrm_init_dstn,
    "pLvlInitDstn": make_lognormal_pLvl_init_dstn,
}

# Make a dictionary with parameters for the default constructor for kNrmInitDstn
PerfForesightConsumerType_kNrmInitDstn_default = {
    "kLogInitMean": -12.0,  # Mean of log initial capital
    "kLogInitStd": 0.0,  # Stdev of log initial capital
    "kNrmInitCount": 15,  # Number of points in initial capital discretization
}

# Make a dictionary with parameters for the default constructor for pLvlInitDstn
PerfForesightConsumerType_pLvlInitDstn_default = {
    "pLogInitMean": 0.0,  # Mean of log permanent income
    "pLogInitStd": 0.0,  # Stdev of log permanent income
    "pLvlInitCount": 15,  # Number of points in initial capital discretization
}

# Make a dictionary to specify a perfect foresight consumer type
PerfForesightConsumerType_solving_defaults = {
    # BASIC HARK PARAMETERS REQUIRED TO SOLVE THE MODEL
    "cycles": 1,  # Finite, non-cyclic model
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    "pseudo_terminal": False,  # Terminal period really does exist
    "constructors": PerfForesightConsumerType_constructors_default,  # See dictionary above
    # PARAMETERS REQUIRED TO SOLVE THE MODEL
    "CRRA": 2.0,  # Coefficient of relative risk aversion
    "Rfree": [1.03],  # Interest factor on retained assets
    "DiscFac": 0.96,  # Intertemporal discount factor
    "LivPrb": [0.98],  # Survival probability after each period
    "PermGroFac": [1.01],  # Permanent income growth factor
    "BoroCnstArt": None,  # Artificial borrowing constraint
    "MaxKinks": 400,  # Maximum number of grid points to allow in cFunc
}
PerfForesightConsumerType_simulation_defaults = {
    # PARAMETERS REQUIRED TO SIMULATE THE MODEL
    "AgentCount": 10000,  # Number of agents of this type
    "T_age": None,  # Age after which simulated agents are automatically killed
    "PermGroFacAgg": 1.0,  # Aggregate permanent income growth factor
    # (The portion of PermGroFac attributable to aggregate productivity growth)
    # ADDITIONAL OPTIONAL PARAMETERS
    "PerfMITShk": False,  # Do Perfect Foresight MIT Shock
    # (Forces Newborns to follow solution path of the agent they replaced if True)
}
PerfForesightConsumerType_defaults = {}
PerfForesightConsumerType_defaults.update(PerfForesightConsumerType_solving_defaults)
PerfForesightConsumerType_defaults.update(
    PerfForesightConsumerType_kNrmInitDstn_default
)
PerfForesightConsumerType_defaults.update(
    PerfForesightConsumerType_pLvlInitDstn_default
)
PerfForesightConsumerType_defaults.update(PerfForesightConsumerType_simulation_defaults)
init_perfect_foresight = PerfForesightConsumerType_defaults


class PerfForesightConsumerType(AgentType):
    r"""
    A perfect foresight consumer type who has no uncertainty other than mortality.
    Their problem is defined by a coefficient of relative risk aversion (:math:`\rho`), intertemporal
    discount factor (:math:`\beta`), interest factor (:math:`\mathsf{R}`), an optional artificial borrowing constraint (:math:`\underline{a}`)
    and time sequences of the permanent income growth rate (:math:`\Gamma`) and survival probability (:math:`1-\mathsf{D}`).
    Their assets and income are normalized by permanent income.

    .. math::
        \newcommand{\CRRA}{\rho}
        \newcommand{\DiePrb}{\mathsf{D}}
        \newcommand{\PermGroFac}{\Gamma}
        \newcommand{\Rfree}{\mathsf{R}}
        \newcommand{\DiscFac}{\beta}
        \begin{align*}
        v_t(m_t) &= \max_{c_t}u(c_t) + \DiscFac (1 - \DiePrb_{t+1}) \PermGroFac_{t+1}^{1-\CRRA} v_{t+1}(m_{t+1}), \\
        & \text{s.t.}  \\
        a_t &= m_t - c_t, \\
        a_t &\geq \underline{a}, \\
        m_{t+1} &= \Rfree_{t+1} a_t/\PermGroFac_{t+1} + 1, \\
        u(c) &= \frac{c^{1-\CRRA}}{1-\CRRA}
        \end{align*}


    Solving Parameters
    ------------------
    cycles: int
        0 specifies an infinite horizon model, 1 specifies a finite model.
    T_cycle: int
        Number of periods in the cycle for this agent type.
    CRRA: float, :math:`\rho`
        Coefficient of Relative Risk Aversion.
    Rfree: float or list[float], time varying, :math:`\mathsf{R}`
        Risk Free interest rate. Pass a list of floats to make Rfree time varying.
    DiscFac: float, :math:`\beta`
        Intertemporal discount factor.
    LivPrb: list[float], time varying, :math:`1-\mathsf{D}`
        Survival probability after each period.
    PermGroFac: list[float], time varying, :math:`\Gamma`
        Permanent income growth factor.
    BoroCnstArt: float, :math:`\underline{a}`
        The minimum Asset/Perminant Income ratio, None to ignore.
    MaxKinks: int
        Maximum number of gridpoints to allow in cFunc.

    Simulation Parameters
    ---------------------
    AgentCount: int
        Number of agents of this kind that are created during simulations.
    T_age: int
        Age after which to automatically kill agents, None to ignore.
    T_sim: int, required for simulation
        Number of periods to simulate.
    track_vars: list[strings]
        List of variables that should be tracked when running the simulation.
        For this agent, the options are 'kNrm', 'aLvl', 'aNrm', 'bNrm', 'cNrm', 'mNrm', 'pLvl', and 'who_dies'.

        kNrm is beginning-of-period capital holdings (last period's assets)

        aLvl is the nominal asset level

        aNrm is the normalized assets

        bNrm is the normalized resources without this period's labor income

        cNrm is the normalized consumption

        mNrm is the normalized market resources

        pLvl is the permanent income level

        who_dies is the array of which agents died
    aNrmInitMean: float
        Mean of Log initial Normalized Assets.
    aNrmInitStd: float
        Std of Log initial Normalized Assets.
    pLvlInitMean: float
        Mean of Log initial permanent income.
    pLvlInitStd: float
        Std of Log initial permanent income.
    PermGroFacAgg: float
        Aggregate permanent income growth factor (The portion of PermGroFac attributable to aggregate productivity growth).
    PerfMITShk: boolean
        Do Perfect Foresight MIT Shock (Forces Newborns to follow solution path of the agent they replaced if True).

    Attributes
    ----------
    solution: list[Consumer solution object]
        Created by the :func:`.solve` method. Finite horizon models create a list with T_cycle+1 elements, for each period in the solution.
        Infinite horizon solutions return a list with T_cycle elements for each period in the cycle.

        Visit :class:`HARK.ConsumptionSaving.ConsIndShockModel.ConsumerSolution` for more information about the solution.
    history: Dict[Array]
        Created by running the :func:`.simulate()` method.
        Contains the variables in track_vars. Each item in the dictionary is an array with the shape (T_sim,AgentCount).
        Visit :class:`HARK.core.AgentType.simulate` for more information.
    """

    solving_defaults = PerfForesightConsumerType_solving_defaults
    simulation_defaults = PerfForesightConsumerType_simulation_defaults

    default_ = {
        "params": PerfForesightConsumerType_defaults,
        "solver": solve_one_period_ConsPF,
        "model": "ConsPerfForesight.yaml",
    }

    time_vary_ = ["LivPrb", "PermGroFac", "Rfree"]
    time_inv_ = ["CRRA", "DiscFac", "MaxKinks", "BoroCnstArt"]
    state_vars = ["kNrm", "pLvl", "bNrm", "mNrm", "aNrm", "aLvl"]
    shock_vars_ = []
    distributions = ["kNrmInitDstn", "pLvlInitDstn"]

    def pre_solve(self):
        """
        Method that is run automatically just before solution by backward iteration.
        Solves the (trivial) terminal period and does a quick check on the borrowing
        constraint and MaxKinks attribute (only relevant in constrained, infinite
        horizon problems).
        """
        self.check_restrictions()
        self.construct("solution_terminal")  # Solve the terminal period problem
        self.check_conditions(verbose=self.verbose)

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
            raise ValueError("DiscFac is below zero with value: " + str(self.DiscFac))

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
        self.state_now["aNrm"][which_agents] = self.kNrmInitDstn.draw(N)
        self.state_now["pLvl"][which_agents] = self.pLvlInitDstn.draw(N)
        self.state_now["pLvl"][which_agents] *= self.state_now["PlvlAgg"]
        self.t_age[which_agents] = 0  # How many periods since each agent was born

        # Because of the timing of the simulation system, kNrm gets written to
        # the *previous* period's aNrm after that aNrm has already been copied
        # to the history array (if it's being tracked). It will be loaded into
        # the simulation as kNrm, however, when the period is simulated.

        # If PerfMITShk not specified, let it be False
        if not hasattr(self, "PerfMITShk"):
            self.PerfMITShk = False
        if not self.PerfMITShk:
            # If True, Newborns inherit t_cycle of agent they replaced (i.e. t_cycles are not reset).
            self.t_cycle[which_agents] = 0
            # Which period of the cycle each agent is currently in

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

    def get_Rport(self):
        """
        Returns an array of size self.AgentCount with Rfree in every entry,
        representing the risk-free portfolio return

        Parameters
        ----------
        None

        Returns
        -------
        RfreeNow : np.array
             Array of size self.AgentCount with risk free interest rate for each agent.
        """
        Rfree_array = np.array(self.Rfree)
        return Rfree_array[self.t_cycle - 1]

    def transition(self):
        pLvlPrev = self.state_prev["pLvl"]
        kNrm = self.state_prev["aNrm"]
        RportNow = self.get_Rport()

        # Calculate new states: normalized market resources and permanent income level
        # Updated permanent income level
        pLvlNow = pLvlPrev * self.shocks["PermShk"]
        # "Effective" interest factor on normalized assets
        ReffNow = RportNow / self.shocks["PermShk"]
        bNrmNow = ReffNow * kNrm  # Bank balances before labor income
        # Market resources after income
        mNrmNow = bNrmNow + self.shocks["TranShk"]

        return kNrm, pLvlNow, bNrmNow, mNrmNow, None

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
        cNrmNow = np.full(self.AgentCount, np.nan)
        MPCnow = np.full(self.AgentCount, np.nan)
        for t in np.unique(self.t_cycle):
            idx = self.t_cycle == t
            if np.any(idx):
                cNrmNow[idx], MPCnow[idx] = self.solution[t].cFunc.eval_with_derivative(
                    self.state_now["mNrm"][idx]
                )
        self.controls["cNrm"] = cNrmNow

        # MPCnow is not really a control
        self.MPCnow = MPCnow

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
        self.state_now["aNrm"] = self.state_now["mNrm"] - self.controls["cNrm"]
        self.state_now["aLvl"] = self.state_now["aNrm"] * self.state_now["pLvl"]
        # Update aggregate permanent productivity level
        self.state_now["PlvlAgg"] = self.state_prev["PlvlAgg"] * self.PermShkAggNow

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
            ["Rfree", "risk free interest factor", "R", True],
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
        aux_dict["APFac"] = (self.Rfree[0] * self.DiscFac * self.LivPrb[0]) ** (
            1 / self.CRRA
        )
        aux_dict["GPFacRaw"] = aux_dict["APFac"] / self.PermGroFac[0]
        aux_dict["FHWFac"] = self.PermGroFac[0] / self.Rfree[0]
        aux_dict["RPFac"] = aux_dict["APFac"] / self.Rfree[0]
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
        Ex_Rnrm = self.Rfree[0] / self.PermGroFac[0]
        aux_dict["Delta_mNrm_ZeroFunc"] = (
            lambda m: (1.0 - 1.0 / Ex_Rnrm) * m + 1.0 / Ex_Rnrm
        )

        # Generate the "E[M_tp1 / M_t] = G" function, which is used to find balanced growth market resources
        PF_Rnrm = self.Rfree[0] / self.PermGroFac[0]
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
            pass  # pragma: nocover
            # This can never be reached! If GICRaw and FHWC both fail, then the RIC also fails, and we would have exited by this point.
        self.log_condition_result(None, None, GIC_message, verbose)

        if not self.quiet:
            _log.info(self.bilt["conditions_report"])

    def calc_stable_points(self, force=False):
        """
        If the problem is one that satisfies the conditions required for target ratios of different
        variables to permanent income to exist, and has been solved to within the self-defined
        tolerance, this method calculates the target values of market resources.

        Parameters
        ----------
        force : bool
            Indicator for whether the method should be forced to be run even if
            the agent seems to be the wrong type. Default is False.

        Returns
        -------
        None
        """
        # Child classes should not run this method
        is_perf_foresight = type(self) is PerfForesightConsumerType
        is_ind_shock = type(self) is IndShockConsumerType
        if not (is_perf_foresight or is_ind_shock or force):
            return

        infinite_horizon = self.cycles == 0
        single_period = self.T_cycle == 1
        if not infinite_horizon:
            raise ValueError(
                "The calc_stable_points method works only for infinite horizon models."
            )
        if not single_period:
            raise ValueError(
                "The calc_stable_points method works only with a single infinitely repeated period."
            )
        if not hasattr(self, "conditions"):
            raise ValueError(
                "The check_conditions method must be run before the calc_stable_points method."
            )
        if not hasattr(self, "solution"):
            raise ValueError(
                "The solve method must be run before the calc_stable_points method."
            )

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

# Make a dictionary of constructors for the idiosyncratic income shocks model
IndShockConsumerType_constructors_default = {
    "kNrmInitDstn": make_lognormal_kNrm_init_dstn,
    "pLvlInitDstn": make_lognormal_pLvl_init_dstn,
    "IncShkDstn": construct_lognormal_income_process_unemployment,
    "PermShkDstn": get_PermShkDstn_from_IncShkDstn,
    "TranShkDstn": get_TranShkDstn_from_IncShkDstn,
    "aXtraGrid": make_assets_grid,
    "solution_terminal": make_basic_CRRA_solution_terminal,
}

# Make a dictionary with parameters for the default constructor for kNrmInitDstn
IndShockConsumerType_kNrmInitDstn_default = {
    "kLogInitMean": -12.0,  # Mean of log initial capital
    "kLogInitStd": 0.0,  # Stdev of log initial capital
    "kNrmInitCount": 15,  # Number of points in initial capital discretization
}

# Make a dictionary with parameters for the default constructor for pLvlInitDstn
IndShockConsumerType_pLvlInitDstn_default = {
    "pLogInitMean": 0.0,  # Mean of log permanent income
    "pLogInitStd": 0.0,  # Stdev of log permanent income
    "pLvlInitCount": 15,  # Number of points in initial capital discretization
}

# Default parameters to make IncShkDstn using construct_lognormal_income_process_unemployment
IndShockConsumerType_IncShkDstn_default = {
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

# Default parameters to make aXtraGrid using make_assets_grid
IndShockConsumerType_aXtraGrid_default = {
    "aXtraMin": 0.001,  # Minimum end-of-period "assets above minimum" value
    "aXtraMax": 20,  # Maximum end-of-period "assets above minimum" value
    "aXtraNestFac": 3,  # Exponential nesting factor for aXtraGrid
    "aXtraCount": 48,  # Number of points in the grid of "assets above minimum"
    "aXtraExtra": None,  # Additional other values to add in grid (optional)
}

# Make a dictionary to specify an idiosyncratic income shocks consumer type
IndShockConsumerType_solving_default = {
    # BASIC HARK PARAMETERS REQUIRED TO SOLVE THE MODEL
    "cycles": 1,  # Finite, non-cyclic model
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    "pseudo_terminal": False,  # Terminal period really does exist
    "constructors": IndShockConsumerType_constructors_default,  # See dictionary above
    # PRIMITIVE RAW PARAMETERS REQUIRED TO SOLVE THE MODEL
    "CRRA": 2.0,  # Coefficient of relative risk aversion
    "Rfree": [1.03],  # Interest factor on retained assets
    "DiscFac": 0.96,  # Intertemporal discount factor
    "LivPrb": [0.98],  # Survival probability after each period
    "PermGroFac": [1.01],  # Permanent income growth factor
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint
    "vFuncBool": False,  # Whether to calculate the value function during solution
    "CubicBool": False,  # Whether to use cubic spline interpolation when True
    # (Uses linear spline interpolation for cFunc when False)
}
IndShockConsumerType_simulation_default = {
    # PARAMETERS REQUIRED TO SIMULATE THE MODEL
    "AgentCount": 10000,  # Number of agents of this type
    "T_age": None,  # Age after which simulated agents are automatically killed
    "PermGroFacAgg": 1.0,  # Aggregate permanent income growth factor
    # (The portion of PermGroFac attributable to aggregate productivity growth)
    "NewbornTransShk": False,  # Whether Newborns have transitory shock
    # ADDITIONAL OPTIONAL PARAMETERS
    "PerfMITShk": False,  # Do Perfect Foresight MIT Shock
    # (Forces Newborns to follow solution path of the agent they replaced if True)
    "neutral_measure": False,  # Whether to use permanent income neutral measure (see Harmenberg 2021)
}

IndShockConsumerType_defaults = {}
IndShockConsumerType_defaults.update(IndShockConsumerType_IncShkDstn_default)
IndShockConsumerType_defaults.update(IndShockConsumerType_kNrmInitDstn_default)
IndShockConsumerType_defaults.update(IndShockConsumerType_pLvlInitDstn_default)
IndShockConsumerType_defaults.update(IndShockConsumerType_aXtraGrid_default)
IndShockConsumerType_defaults.update(IndShockConsumerType_solving_default)
IndShockConsumerType_defaults.update(IndShockConsumerType_simulation_default)
init_idiosyncratic_shocks = IndShockConsumerType_defaults  # Here so that other models which use the old convention don't break


class IndShockConsumerType(PerfForesightConsumerType):
    r"""
    A consumer type with idiosyncratic shocks to permanent and transitory income.
    Their problem is defined by a sequence of income distributions, survival probabilities
    (:math:`\mathsf{S}`), and permanent income growth rates (:math:`\Gamma`), as well
    as time invariant values for risk aversion (:math:`\rho`), discount factor (:math:`\beta`),
    the interest rate (:math:`\mathsf{R}`), the grid of end-of-period assets, and an artificial
    borrowing constraint (:math:`\underline{a}`).

    .. math::
        \newcommand{\CRRA}{\rho}
        \newcommand{\LivPrb}{\mathsf{S}}
        \newcommand{\PermGroFac}{\Gamma}
        \newcommand{\Rfree}{\mathsf{R}}
        \newcommand{\DiscFac}{\beta}
        \begin{align*}
        v_t(m_t) &= \max_{c_t}u(c_t) + \DiscFac \LivPrb_t \mathbb{E}_{t} \left[ (\PermGroFac_{t+1} \psi_{t+1})^{1-\CRRA} v_{t+1}(m_{t+1}) \right], \\
        & \text{s.t.}  \\
        a_t &= m_t - c_t, \\
        a_t &\geq \underline{a}, \\
        m_{t+1} &= a_t \Rfree_{t+1}/(\PermGroFac_{t+1} \psi_{t+1}) + \theta_{t+1}, \\
        (\psi_{t+1},\theta_{t+1}) &\sim F_{t+1}, \\
        \mathbb{E}[\psi]=\mathbb{E}[\theta] &= 1, \\
        u(c) &= \frac{c^{1-\CRRA}}{1-\CRRA}.
        \end{align*}


    Constructors
    ------------
    IncShkDstn: Constructor, :math:`\psi`, :math:`\theta`
        The agent's income shock distributions.

        Its default constructor is :func:`HARK.Calibration.Income.IncomeProcesses.construct_lognormal_income_process_unemployment`
    aXtraGrid: Constructor
        The agent's asset grid.

        Its default constructor is :func:`HARK.utilities.make_assets_grid`

    Solving Parameters
    ------------------
    cycles: int
        0 specifies an infinite horizon model, 1 specifies a finite model.
    T_cycle: int
        Number of periods in the cycle for this agent type.
    CRRA: float, :math:`\rho`
        Coefficient of Relative Risk Aversion.
    Rfree: float or list[float], time varying, :math:`\mathsf{R}`
        Risk Free interest rate. Pass a list of floats to make Rfree time varying.
    DiscFac: float, :math:`\beta`
        Intertemporal discount factor.
    LivPrb: list[float], time varying, :math:`1-\mathsf{D}`
        Survival probability after each period.
    PermGroFac: list[float], time varying, :math:`\Gamma`
        Permanent income growth factor.
    BoroCnstArt: float, :math:`\underline{a}`
        The minimum Asset/Perminant Income ratio, None to ignore.
    vFuncBool: bool
        Whether to calculate the value function during solution.
    CubicBool: bool
        Whether to use cubic spline interpoliation.

    Simulation Parameters
    ---------------------
    AgentCount: int
        Number of agents of this kind that are created during simulations.
    T_age: int
        Age after which to automatically kill agents, None to ignore.
    T_sim: int, required for simulation
        Number of periods to simulate.
    track_vars: list[strings]
        List of variables that should be tracked when running the simulation.
        For this agent, the options are 'PermShk', 'TranShk', 'aLvl', 'aNrm', 'bNrm', 'cNrm', 'mNrm', 'pLvl', and 'who_dies'.

        PermShk is the agent's permanent income shock

        TranShk is the agent's transitory income shock

        aLvl is the nominal asset level

        aNrm is the normalized assets

        bNrm is the normalized resources without this period's labor income

        cNrm is the normalized consumption

        mNrm is the normalized market resources

        pLvl is the permanent income level

        who_dies is the array of which agents died
    aNrmInitMean: float
        Mean of Log initial Normalized Assets.
    aNrmInitStd: float
        Std of Log initial Normalized Assets.
    pLvlInitMean: float
        Mean of Log initial permanent income.
    pLvlInitStd: float
        Std of Log initial permanent income.
    PermGroFacAgg: float
        Aggregate permanent income growth factor (The portion of PermGroFac attributable to aggregate productivity growth).
    PerfMITShk: boolean
        Do Perfect Foresight MIT Shock (Forces Newborns to follow solution path of the agent they replaced if True).
    NewbornTransShk: boolean
        Whether Newborns have transitory shock.

    Attributes
    ----------
    solution: list[Consumer solution object]
        Created by the :func:`.solve` method. Finite horizon models create a list with T_cycle+1 elements, for each period in the solution.
        Infinite horizon solutions return a list with T_cycle elements for each period in the cycle.

        Visit :class:`HARK.ConsumptionSaving.ConsIndShockModel.ConsumerSolution` for more information about the solution.
    history: Dict[Array]
        Created by running the :func:`.simulate()` method.
        Contains the variables in track_vars. Each item in the dictionary is an array with the shape (T_sim,AgentCount).
        Visit :class:`HARK.core.AgentType.simulate` for more information.
    """

    IncShkDstn_defaults = IndShockConsumerType_IncShkDstn_default
    aXtraGrid_defaults = IndShockConsumerType_aXtraGrid_default
    solving_defaults = IndShockConsumerType_solving_default
    simulation_defaults = IndShockConsumerType_simulation_default
    default_ = {
        "params": IndShockConsumerType_defaults,
        "solver": solve_one_period_ConsIndShock,
        "model": "ConsIndShock.yaml",
    }

    time_inv_ = PerfForesightConsumerType.time_inv_ + [
        "vFuncBool",
        "CubicBool",
        "aXtraGrid",
    ]
    time_vary_ = PerfForesightConsumerType.time_vary_ + [
        "IncShkDstn",
        "PermShkDstn",
        "TranShkDstn",
    ]
    # This is in the PerfForesight model but not ConsIndShock
    time_inv_.remove("MaxKinks")
    shock_vars_ = ["PermShk", "TranShk"]
    distributions = [
        "IncShkDstn",
        "PermShkDstn",
        "TranShkDstn",
        "kNrmInitDstn",
        "pLvlInitDstn",
    ]

    def update_income_process(self):
        self.update("IncShkDstn", "PermShkDstn", "TranShkDstn")

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
        # Whether Newborns have transitory shock. The default is False.
        NewbornTransShk = self.NewbornTransShk

        PermShkNow = np.zeros(self.AgentCount)  # Initialize shock arrays
        TranShkNow = np.zeros(self.AgentCount)
        newborn = self.t_age == 0
        for t in np.unique(self.t_cycle):
            idx = self.t_cycle == t

            # temporary, see #1022
            if self.cycles == 1:
                t = t - 1

            N = np.sum(idx)
            if N > 0:
                # set current income distribution
                IncShkDstnNow = self.IncShkDstn[t]
                # and permanent growth factor
                PermGroFacNow = self.PermGroFac[t]
                # Get random draws of income shocks from the discrete distribution
                IncShks = IncShkDstnNow.draw(N)

                PermShkNow[idx] = (
                    IncShks[0, :] * PermGroFacNow
                )  # permanent "shock" includes expected growth
                TranShkNow[idx] = IncShks[1, :]

        # That procedure used the *last* period in the sequence for newborns, but that's not right
        # Redraw shocks for newborns, using the *first* period in the sequence.  Approximation.
        N = np.sum(newborn)
        if N > 0:
            idx = newborn
            # set current income distribution
            IncShkDstnNow = self.IncShkDstn[0]
            PermGroFacNow = self.PermGroFac[0]  # and permanent growth factor

            # Get random draws of income shocks from the discrete distribution
            EventDraws = IncShkDstnNow.draw_events(N)
            PermShkNow[idx] = (
                IncShkDstnNow.atoms[0][EventDraws] * PermGroFacNow
            )  # permanent "shock" includes expected growth
            TranShkNow[idx] = IncShkDstnNow.atoms[1][EventDraws]

        #  Whether Newborns have transitory shock. The default is False.
        if not NewbornTransShk:
            TranShkNow[newborn] = 1.0

        # Store the shocks in self
        self.shocks["PermShk"] = PermShkNow
        self.shocks["TranShk"] = TranShkNow

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
                TranShkDstn, p=self.UnempPrb, x=self.IncUnemp
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
        ShkCount = IncShkDstn.pmv.size
        aCount = aNowGrid.size
        aNowGrid_tiled = np.tile(aNowGrid, (ShkCount, 1))
        PermShkVals_tiled = (np.tile(IncShkDstn.atoms[0], (aCount, 1))).transpose()
        TranShkVals_tiled = (np.tile(IncShkDstn.atoms[1], (aCount, 1))).transpose()
        ShkPrbs_tiled = (np.tile(IncShkDstn.pmv, (aCount, 1))).transpose()

        # Calculate marginal value next period for each gridpoint and each shock
        mNextArray = (
            self.Rfree[0] / (self.PermGroFac[0] * PermShkVals_tiled) * aNowGrid_tiled
            + TranShkVals_tiled
        )
        vPnextArray = vPfuncNext(mNextArray)

        # Calculate expected marginal value and implied optimal consumption
        ExvPnextGrid = (
            self.DiscFac
            * self.Rfree[0]
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
        self.check_restrictions()
        self.construct("solution_terminal")
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
        Ex_Rnrm = self.Rfree[0] / self.PermGroFac[0] * Ex_PermShkInv
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

# Specify default parameters used in "kinked R" model

KinkedRconsumerType_IncShkDstn_default = IndShockConsumerType_IncShkDstn_default.copy()
KinkedRconsumerType_aXtraGrid_default = IndShockConsumerType_aXtraGrid_default.copy()
KinkedRconsumerType_kNrmInitDstn_default = (
    IndShockConsumerType_kNrmInitDstn_default.copy()
)
KinkedRconsumerType_pLvlInitDstn_default = (
    IndShockConsumerType_pLvlInitDstn_default.copy()
)

KinkedRconsumerType_solving_default = IndShockConsumerType_solving_default.copy()
KinkedRconsumerType_solving_default.update(
    {
        "Rboro": 1.20,  # Interest factor on assets when borrowing, a < 0
        "Rsave": 1.02,  # Interest factor on assets when saving, a > 0
        "BoroCnstArt": None,  # Kinked R only matters if borrowing is allowed
    }
)
del KinkedRconsumerType_solving_default["Rfree"]

KinkedRconsumerType_simulation_default = IndShockConsumerType_simulation_default.copy()

KinkedRconsumerType_defaults = {}
KinkedRconsumerType_defaults.update(
    KinkedRconsumerType_IncShkDstn_default
)  # Fill with some parameters
KinkedRconsumerType_defaults.update(KinkedRconsumerType_pLvlInitDstn_default)
KinkedRconsumerType_defaults.update(KinkedRconsumerType_kNrmInitDstn_default)
KinkedRconsumerType_defaults.update(KinkedRconsumerType_aXtraGrid_default)
KinkedRconsumerType_defaults.update(KinkedRconsumerType_solving_default)
KinkedRconsumerType_defaults.update(KinkedRconsumerType_simulation_default)
init_kinked_R = KinkedRconsumerType_defaults


class KinkedRconsumerType(IndShockConsumerType):
    r"""
    A consumer type based on IndShockConsumerType, with different
    interest rates for saving (:math:`\mathsf{R}_{save}`) and borrowing
    (:math:`\mathsf{R}_{boro}`).

    .. math::
        \newcommand{\CRRA}{\rho}
        \newcommand{\DiePrb}{\mathsf{D}}
        \newcommand{\PermGroFac}{\Gamma}
        \newcommand{\Rfree}{\mathsf{R}}
        \newcommand{\DiscFac}{\beta}
        \begin{align*}
        v_t(m_t) &= \max_{c_t} u(c_t) + \DiscFac (1-\DiePrb_{t+1})  \mathbb{E}_{t} \left[(\PermGroFac_{t+1}\psi_{t+1})^{1-\CRRA} v_{t+1}(m_{t+1}) \right], \\
        & \text{s.t.}  \\
        a_t &= m_t - c_t, \\
        a_t &\geq \underline{a}, \\
        m_{t+1} &= \Rfree_t/(\PermGroFac_{t+1} \psi_{t+1}) a_t + \theta_{t+1}, \\
        \Rfree_t &= \begin{cases}
        \Rfree_{boro} & \text{if } a_t < 0\\
        \Rfree_{save} & \text{if } a_t \geq 0,
        \end{cases}\\
        \Rfree_{boro} &> \Rfree_{save}, \\
        (\psi_{t+1},\theta_{t+1}) &\sim F_{t+1}, \\
        \mathbb{E}[\psi]=\mathbb{E}[\theta] &= 1.\\
        u(c) &= \frac{c^{1-\CRRA}}{1-\CRRA} \\
        \end{align*}

    Constructors
    ------------
    IncShkDstn: Constructor, :math:`\psi`, :math:`\theta`
        The agent's income shock distributions.
        Its default constructor is :func:`HARK.Calibration.Income.IncomeProcesses.construct_lognormal_income_process_unemployment`
    aXtraGrid: Constructor
        The agent's asset grid.
        Its default constructor is :func:`HARK.utilities.make_assets_grid`

    Solving Parameters
    ------------------
    cycles: int
        0 specifies an infinite horizon model, 1 specifies a finite model.
    T_cycle: int
        Number of periods in the cycle for this agent type.
    CRRA: float, :math:`\rho`
        Coefficient of Relative Risk Aversion.
    Rboro: float, :math:`\mathsf{R}_{boro}`
        Risk Free interest rate when assets are negative.
    Rsave: float, :math:`\mathsf{R}_{save}`
        Risk Free interest rate when assets are positive.
    DiscFac: float, :math:`\beta`
        Intertemporal discount factor.
    LivPrb: list[float], time varying, :math:`1-\mathsf{D}`
        Survival probability after each period.
    PermGroFac: list[float], time varying, :math:`\Gamma`
        Permanent income growth factor.
    BoroCnstArt: float, :math:`\underline{a}`
        The minimum Asset/Perminant Income ratio, None to ignore.
    vFuncBool: bool
        Whether to calculate the value function during solution.
    CubicBool: bool
        Whether to use cubic spline interpoliation.

    Simulation Parameters
    ---------------------
    AgentCount: int
        Number of agents of this kind that are created during simulations.
    T_age: int
        Age after which to automatically kill agents, None to ignore.
    T_sim: int, required for simulation
        Number of periods to simulate.
    track_vars: list[strings]
        List of variables that should be tracked when running the simulation.
        For this agent, the options are 'PermShk', 'TranShk', 'aLvl', 'aNrm', 'bNrm', 'cNrm', 'mNrm', 'pLvl', and 'who_dies'.

        PermShk is the agent's permanent income shock

        TranShk is the agent's transitory income shock

        aLvl is the nominal asset level

        aNrm is the normalized assets

        bNrm is the normalized resources without this period's labor income

        cNrm is the normalized consumption

        mNrm is the normalized market resources

        pLvl is the permanent income level

        who_dies is the array of which agents died
    aNrmInitMean: float
        Mean of Log initial Normalized Assets.
    aNrmInitStd: float
        Std of Log initial Normalized Assets.
    pLvlInitMean: float
        Mean of Log initial permanent income.
    pLvlInitStd: float
        Std of Log initial permanent income.
    PermGroFacAgg: float
        Aggregate permanent income growth factor (The portion of PermGroFac attributable to aggregate productivity growth).
    PerfMITShk: boolean
        Do Perfect Foresight MIT Shock (Forces Newborns to follow solution path of the agent they replaced if True).
    NewbornTransShk: boolean
        Whether Newborns have transitory shock.

    Attributes
    ----------
    solution: list[Consumer solution object]
        Created by the :func:`.solve` method. Finite horizon models create a list with T_cycle+1 elements, for each period in the solution.
        Infinite horizon solutions return a list with T_cycle elements for each period in the cycle.

        Visit :class:`HARK.ConsumptionSaving.ConsIndShockModel.ConsumerSolution` for more information about the solution.
    history: Dict[Array]
        Created by running the :func:`.simulate()` method.
        Contains the variables in track_vars. Each item in the dictionary is an array with the shape (T_sim,AgentCount).
        Visit :class:`HARK.core.AgentType.simulate` for more information.
    """

    IncShkDstn_defaults = KinkedRconsumerType_IncShkDstn_default
    aXtraGrid_defaults = KinkedRconsumerType_aXtraGrid_default
    solving_defaults = KinkedRconsumerType_solving_default
    simulation_defaults = KinkedRconsumerType_simulation_default
    default_ = {
        "params": KinkedRconsumerType_defaults,
        "solver": solve_one_period_ConsKinkedR,
        "model": "ConsKinkedR.yaml",
    }

    time_inv_ = copy(IndShockConsumerType.time_inv_)
    time_inv_ += ["Rboro", "Rsave"]

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
        PermShkValsNext = self.IncShkDstn[0].atoms[0]
        TranShkValsNext = self.IncShkDstn[0].atoms[1]
        ShkPrbsNext = self.IncShkDstn[0].pmv
        IncNext = PermShkValsNext * TranShkValsNext
        Ex_IncNext = np.dot(ShkPrbsNext, IncNext)
        PermShkMinNext = np.min(PermShkValsNext)
        TranShkMinNext = np.min(TranShkValsNext)
        WorstIncNext = PermShkMinNext * TranShkMinNext
        WorstIncPrb = np.sum(ShkPrbsNext[IncNext == WorstIncNext])
        # TODO: Check the math above. I think it fails for non-independent shocks

        BoroCnstArt = np.inf if self.BoroCnstArt is None else self.BoroCnstArt

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
        if BoroCnstNat < BoroCnstArt:
            MPCmax = 1.0  # if natural borrowing constraint is overridden by artificial one, MPCmax is 1
        else:
            MPCmax = 1.0 - WorstIncPrb ** (1.0 / self.CRRA) * PatFacBot
        MPCmin = 1.0 - PatFacTop

        # Store the results as attributes of self
        self.hNrm = hNrm
        self.MPCmin = MPCmin
        self.MPCmax = MPCmax

    def make_euler_error_func(self, mMax=100, approx_inc_dstn=True):  # pragma: nocover
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
        """
        raise NotImplementedError()

    def get_Rport(self):
        """
        Returns an array of size self.AgentCount with self.Rboro or self.Rsave in
        each entry, based on whether self.aNrmNow >< 0. This represents the risk-
        free portfolio return in this model.

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
        pass


###############################################################################

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
    female=False, cross_sec=True, year=2004, age_min=birth_age, age_max=death_age
)

# Parameters related to the number of periods implied by the calibration
time_params = parse_time_params(age_birth=birth_age, age_death=death_age)

# Update all the new parameters
init_lifecycle = copy(init_idiosyncratic_shocks)
del init_lifecycle["constructors"]
init_lifecycle.update(time_params)
init_lifecycle.update(dist_params)
# Note the income specification overrides the pLvlInitMean from the SCF.
init_lifecycle.update(income_params)
init_lifecycle.update({"LivPrb": liv_prb})
init_lifecycle["Rfree"] = init_lifecycle["T_cycle"] * init_lifecycle["Rfree"]

# Make a dictionary to specify an infinite consumer with a four period cycle
init_cyclical = copy(init_idiosyncratic_shocks)
init_cyclical["PermGroFac"] = [1.1, 1.082251, 2.8, 0.3]
init_cyclical["PermShkStd"] = [0.1, 0.1, 0.1, 0.1]
init_cyclical["TranShkStd"] = [0.1, 0.1, 0.1, 0.1]
init_cyclical["LivPrb"] = 4 * [0.98]
init_cyclical["Rfree"] = 4 * [1.03]
init_cyclical["T_cycle"] = 4

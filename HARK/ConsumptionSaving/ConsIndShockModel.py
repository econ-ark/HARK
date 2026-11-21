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

import warnings
from copy import copy
from dataclasses import asdict, dataclass

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
    DiscreteDistribution,
    LogNormal,
    LogNormalMeanOne,
    Uniform,
    add_discrete_outcome_constant_mean,
    cdf_invert,
    combine_indep_dstns,
    expected,
    employed_atoms,
    restrict_to_atoms,
    expected_continuous,
)
from HARK.distributions.utils import (
    Z_STEP,
    Z_UNBOUNDED,
    _coordinate,
    _split_gauss_legendre,
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
from HARK.stationary import KnotProblem, solve_stationary_problem
from HARK.utilities import make_assets_grid
from scipy.optimize import newton

from HARK import (
    AgentType,
    NullFunc,
    _log,
    set_verbosity_level,
)
from HARK.core import _implied_distance

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
    "buffer_stock_lifecycle_unskilled",
    "buffer_stock_lifecycle_operative",
    "buffer_stock_lifecycle_manager",
]

utility = CRRAutility
utilityP = CRRAutilityP
utilityPP = CRRAutilityPP
utilityP_inv = CRRAutilityP_inv
utility_invP = CRRAutility_invP
utility_inv = CRRAutility_inv
utilityP_invP = CRRAutilityP_invP


def warn_if_shuffle_voids_base_draw_cache(agent):
    """Warn when income_shuffle and the base-draw cache are both requested.

    The shuffled path picks atom counts directly rather than inverting a
    uniform per agent, so there are no base draws to record and the cache
    comes back empty.  Anything reading it, the dual-measure Q-pipeline in
    particular, then falls back to drawing independently, which is silent
    and undoes the point of the cache.  Returns True when both are set.
    """
    if not getattr(agent, "_cache_base_shock_draws", False):
        return False
    if not getattr(agent, "income_shuffle", False):
        return False
    warnings.warn(
        "income_shuffle=True and _cache_base_shock_draws=True are both set on "
        f"{type(agent).__name__}. The shuffled draw records no base uniforms, "
        "so _base_shock_draws will be empty and any consumer of it (such as "
        "the dual-measure Q-pipeline) will draw independently instead of "
        "sharing the P-measure draws. Turn one of the two off.",
        RuntimeWarning,
        stacklevel=3,
    )
    return True


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
    mNrmKink : float
        Where an artificial borrowing constraint binds, the normalized market
        resources at the kink of the consumption function, below which the
        consumer ends the period at the constraint. NaN where the natural
        borrowing constraint binds, and where the solver does not record the
        kink, so that np.isnan() answers for every solver.
    cNrmKink : float
        Normalized consumption at the kink, mNrmKink minus the constraint on
        end-of-period assets; NaN as for mNrmKink.
    MPCaboveKink : float
        The marginal propensity to consume just above the kink. Below it, on the
        constraint, the MPC is one. NaN as for mNrmKink.

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
        mNrmKink=np.nan,
        cNrmKink=np.nan,
        MPCaboveKink=np.nan,
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
        self.mNrmKink = mNrmKink
        self.cNrmKink = cNrmKink
        self.MPCaboveKink = MPCaboveKink

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


def find_kink(mNrm, cNrm, MPC, BoroCnstNat, mNrmMin):
    """
    Find the kink of a consumption function solved on an end-of-period asset
    grid that starts at a binding artificial borrowing constraint, with the
    constraint itself as a gridpoint. The unconstrained function's nodes are
    then the natural borrowing constraint, with zero consumption, followed by
    the endogenous gridpoints, the first of which is the kink: there the
    consumer ends the period with assets exactly at the constraint.

    Parameters
    ----------
    mNrm : np.array
        Market resources at the unconstrained consumption function's nodes,
        starting with the natural borrowing constraint.
    cNrm : np.array
        Consumption at those nodes.
    MPC : np.array or None
        The marginal propensity to consume at those nodes, for the cubic rule;
        None for the linear rule, whose MPC just above the kink is the slope of
        the segment that starts there.
    BoroCnstNat : float
        The natural borrowing constraint on end-of-period assets.
    mNrmMin : float
        The constraint in force, the larger of the natural and the artificial.

    Returns
    -------
    mNrmKink, cNrmKink, MPCaboveKink : float
        Market resources and consumption at the kink, and the MPC just above it;
        each NaN where the natural borrowing constraint binds, so there is no kink.

    Raises
    ------
    ValueError
        Where the artificial constraint binds but the nodes are not laid out as
        described: the first is not the natural borrowing constraint with zero
        consumption, the second does not end the period at the constraint, or,
        under the linear rule, there are fewer than three.
    """
    if not BoroCnstNat < mNrmMin:
        return np.nan, np.nan, np.nan
    if len(mNrm) < (2 if MPC is not None else 3):
        raise ValueError(
            "find_kink needs the natural borrowing constraint's node, the kink's, "
            "and, under the linear rule, the node above it: "
            f"{len(mNrm)} nodes given."
        )
    if mNrm[0] != BoroCnstNat or cNrm[0] != 0.0:
        raise ValueError(
            "find_kink needs the first node at the natural borrowing constraint, "
            f"{BoroCnstNat}, with zero consumption, not at m = {mNrm[0]}, c = {cNrm[0]}."
        )
    if abs((mNrm[1] - cNrm[1]) - mNrmMin) > 1e-10 * max(1.0, abs(mNrm[1])):
        raise ValueError(
            "find_kink needs the second node to end the period at the constraint, "
            f"{mNrmMin}: there m - c = {mNrm[1] - cNrm[1]}, so the grid does not "
            "start at the constraint."
        )
    if MPC is None:
        MPCaboveKink = (cNrm[2] - cNrm[1]) / (mNrm[2] - mNrm[1])
    else:
        MPCaboveKink = MPC[1]
    return float(mNrm[1]), float(cNrm[1]), float(MPCaboveKink)


def calc_aNrm_grid(aXtraGrid, BoroCnstNat, mNrmMin):
    """
    The end-of-period asset grid of IndShockConsumerType's one-period solvers,
    standard and fast. Written so that numba can compile it, as
    ConsIndShockModelFast does for its solver.

    Where the artificial borrowing constraint binds (BoroCnstNat < mNrmMin), the
    grid starts at that constraint, with the constraint itself as its first
    point. Gridpoints below it would build the unconstrained consumption
    function where the lower envelope always takes the constraint line instead,
    so the expectations computed there would be wasted. At the constraint itself
    the end-of-period marginal value of assets is finite, so the first-order
    condition gives consumption there directly, and that point's endogenous
    gridpoint is the kink where the constraint stops binding: a node of the
    solution, not the crossing of two interpolants (find_kink reads it).

    Below the grid, the unconstrained consumption function starts at the natural
    borrowing constraint, where consumption is zero, whichever constraint binds.
    That point keeps the interpolant defined down to mNrmMin, and the function is
    smooth at the kink, with a single slope there (the unconstrained MPC), so no
    interpolation node has to carry the kink's two slopes. The artificial
    constraint is applied afterwards, as the lower envelope of that function and
    the constraint line: at the kink the slope jumps from one, below it, to the
    unconstrained MPC above it, which a cubic Hermite node cannot carry. A
    linear interpolant could, but it is built the same way.

    Where the natural borrowing constraint binds, the grid stays above it, as
    marginal value is infinite at the natural limit itself.

    Parameters
    ----------
    aXtraGrid : np.array
        End-of-period assets above the constraint in force.
    BoroCnstNat : float
        The natural borrowing constraint.
    mNrmMin : float
        The constraint in force, the larger of the natural and the artificial.

    Returns
    -------
    aNrm : np.array
        The end-of-period asset grid.
    """
    if BoroCnstNat < mNrmMin:
        return mNrmMin + np.concatenate((np.zeros(1), aXtraGrid))
    return aXtraGrid + BoroCnstNat


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
    dstn = LogNormal(
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
    dstn = LogNormal(
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


def calc_mpc_min_closed_form(pat_fac, periods_left=np.inf):
    r"""Calculate the lower bound of the MPC of a perfect foresight consumer in
    closed form.

    This is the closed-form counterpart of :func:`calc_mpc_min`: iterating that
    one-step recursion backward from the terminal period, where the MPC is one,
    reaches this value after ``periods_left`` steps.

    Parameters
    ----------
    pat_fac : float
        Return patience factor :math:`\textbf{Þ}_R = (R \beta \mathcal{L})^{1/\rho} / R`,
        as computed by :func:`calc_patience_factor` from the effective discount
        factor and passed to :func:`calc_mpc_min` by the perfect foresight solver.
    periods_left : int or float, optional
        Number of periods remaining after this one: 0 is the terminal period,
        and ``np.inf`` (the default) gives the infinite horizon limit.

    Returns
    -------
    mpc_min : float
        Lower bound of the marginal propensity to consume, which is the MPC as
        market resources go to infinity (``MPCmin`` in the solution).

    Notes
    -----
    The recursion is :math:`\kappa_n^{-1} = 1 + \textbf{Þ}_R \kappa_{n-1}^{-1}`
    with :math:`\kappa_0 = 1`, so with :math:`n` periods left

    .. math::

        \kappa_n = \frac{1 - \textbf{Þ}_R}{1 - \textbf{Þ}_R^{n+1}},

    which is :math:`1/(n+1)` when :math:`\textbf{Þ}_R = 1`. As
    :math:`n \to \infty` it converges to :math:`1 - \textbf{Þ}_R` when the Return
    Impatience Condition (RIC, :math:`\textbf{Þ}_R < 1`) holds, and to zero when
    it fails.

    The parameters must be constant over time. For a life-cycle profile, iterate
    :func:`calc_mpc_min` instead.

    References
    ----------
    https://intertemporal-choice.github.io/content/consumption/perfforesightcrra/
    """
    if periods_left == np.inf:
        return max(1.0 - pat_fac, 0.0)
    if pat_fac == 1.0:
        return 1.0 / (periods_left + 1)
    return (1.0 - pat_fac) / (1.0 - pat_fac ** (periods_left + 1))


def calc_human_wealth_closed_form(perm_gro_fac, rfree, periods_left=np.inf):
    r"""Calculate the normalized human wealth of a perfect foresight consumer in
    closed form.

    This is the closed-form counterpart of :func:`calc_human_wealth` when income
    is one (normalized by permanent income) every period: iterating that one-step
    recursion backward from the terminal period, where human wealth is zero,
    reaches this value after ``periods_left`` steps.

    Parameters
    ----------
    perm_gro_fac : float
        Permanent income growth factor :math:`G`.
    rfree : float
        Risk free interest factor :math:`R`.
    periods_left : int or float, optional
        Number of periods remaining after this one: 0 is the terminal period,
        and ``np.inf`` (the default) gives the infinite horizon limit.

    Returns
    -------
    h_nrm : float
        Human wealth divided by permanent income, excluding this period's income,
        as in the solution's ``hNrm``. (The ``hNrm`` that ``calc_limiting_values``
        stores in ``bilt`` includes this period's income, so it is one larger
        than the infinite horizon value here.)

    Notes
    -----
    With the finite human wealth factor :math:`G/R` and :math:`n` periods left,

    .. math::

        h_n = \sum_{i=1}^{n} (G/R)^i = \frac{(G/R) \left(1 - (G/R)^n\right)}{1 - G/R},

    which is :math:`n` when :math:`G = R`. As :math:`n \to \infty` it converges
    to :math:`(G/R)/(1 - G/R)` when the Finite Human Wealth Condition (FHWC,
    :math:`G/R < 1`) holds, and is infinite when it fails.

    The parameters must be constant over time. For a life-cycle profile, iterate
    :func:`calc_human_wealth` instead.

    References
    ----------
    https://intertemporal-choice.github.io/content/consumption/perfforesightcrra/
    """
    fhw_fac = perm_gro_fac / rfree
    if periods_left == np.inf:
        return fhw_fac / (1.0 - fhw_fac) if fhw_fac < 1.0 else np.inf
    if fhw_fac == 1.0:
        return float(periods_left)
    return fhw_fac * (1.0 - fhw_fac**periods_left) / (1.0 - fhw_fac)


@dataclass(frozen=True)
class PerfForesightCondition:
    """
    The result of one perfect foresight condition.

    Attributes
    ----------
    name : str
        The condition's name, as a key of an agent's ``conditions``.
    factor_name : str
        The name of its factor, as a key of an agent's ``bilt``.
    factor : float
        The factor, which the condition compares with one.
    rate : float
        The log of the factor, negative when the condition holds.
    holds : bool
        Whether the condition holds.
    meaning : str
        What the result implies.
    """

    name: str
    factor_name: str
    factor: float
    rate: float
    holds: bool
    meaning: str


@dataclass(frozen=True)
class PerfForesightConditions:
    r"""
    The perfect foresight conditions and what they imply, for an infinite horizon
    with one repeated period. See :func:`calc_perf_foresight_conditions`.

    Attributes
    ----------
    conditions : dict
        A :class:`PerfForesightCondition` for each of AIC, RIC, GICRaw, FHWC and PFFVAC.
    regime : str
        The limit of the consumption function as the horizon extends to infinity:
        ``"linear"`` (the unconstrained solution), ``"kinked"`` (piecewise linear, with
        a binding constraint), ``"infinite"`` (consumption goes to infinity), ``"zero"``
        (to zero), or ``"income"`` (the knife edge where the Growth Patience Factor is
        exactly one and the RIC fails: consumption goes to one, where the constraint allows).
    solution_exists : bool
        Whether the limit is a non-degenerate solution: a consumption function with a
        finite value. Where the constraint binds and the RIC fails, consumption can
        converge to the kinked limit while the value does not (see
        :func:`calc_perf_foresight_conditions`); then the regime is ``"kinked"`` and
        no solution exists.
    constraint_binds : bool
        Whether the borrowing constraint can bind in the limit.
    MPCmin : float
        The limiting MPC as market resources go to infinity.
    MPCmax : float
        The limiting MPC as market resources approach their minimum.
    hNrm : float
        Limiting human wealth excluding this period's income, as in a solution's ``hNrm``.
    mNrmStable : float
        The market resources ratio to which the consumer converges when a solution
        exists and the GICRaw holds, and nan otherwise.
    """

    conditions: dict
    regime: str
    solution_exists: bool
    constraint_binds: bool
    MPCmin: float
    MPCmax: float
    hNrm: float
    mNrmStable: float

    def to_dict(self):
        """Return the results as nested dictionaries of plain values."""
        return asdict(self)


def calc_perf_foresight_conditions(
    Rfree, DiscFac, CRRA, PermGroFac, LivPrb=1.0, BoroCnstArt=None
):
    r"""Calculate all of the perfect foresight conditions and what they imply.

    This is a standalone calculation for an infinite horizon with one repeated
    period. It needs no agent, nothing in HARK calls it, and it changes nothing.

    Parameters
    ----------
    Rfree : float
        Risk free interest factor :math:`R`.
    DiscFac : float
        Intertemporal discount factor :math:`\beta`.
    CRRA : float
        Coefficient of relative risk aversion :math:`\rho`.
    PermGroFac : float
        Permanent income growth factor :math:`G`.
    LivPrb : float, optional
        Survival probability. Default 1.0.
    BoroCnstArt : float or None, optional
        Artificial borrowing constraint. Default None.

    Returns
    -------
    results : PerfForesightConditions
        The factors, rates and results of the conditions, and the limit of the solution.

    Notes
    -----
    Without a binding constraint, a linear solution exists if and only if the RIC and
    the FHWC both hold. Otherwise the Growth Patience Factor decides the limit:
    consumption goes to infinity when it is below one and to zero when it is above one.
    A constraint can bind if it is finite and above minus human wealth. Then a
    solution exists if the RIC holds. If the RIC fails, a solution exists if and only
    if the PFFVAC holds, which implies the GICRaw: under the GICRaw the consumer runs
    market resources down to the constraint and consumes permanent income from then
    on, so the value is finite only if the value of autarky is. Where the RIC fails,
    the GICRaw holds and the PFFVAC fails, consumption converges to the kinked limit
    but the value does not.

    References
    ----------
    https://intertemporal-choice.github.io/content/consumption/perfforesightcrra/
    """
    APFac = (Rfree * DiscFac * LivPrb) ** (1.0 / CRRA)
    RPFac = APFac / Rfree
    GPFacRaw = APFac / PermGroFac
    FHWFac = PermGroFac / Rfree
    PFVAFac = (DiscFac * LivPrb) * PermGroFac ** (1.0 - CRRA)

    # name: factor name, factor, meaning if it holds, meaning if it fails
    table = {
        "AIC": (
            "APFac",
            APFac,
            "The level of consumption falls over time.",
            "The level of consumption does not fall over time.",
        ),
        "RIC": (
            "RPFac",
            RPFac,
            "The limiting MPC is positive.",
            "The limiting MPC is zero.",
        ),
        "GICRaw": (
            "GPFacRaw",
            GPFacRaw,
            "Consumption grows more slowly than permanent income.",
            "Consumption does not grow more slowly than permanent income.",
        ),
        "FHWC": (
            "FHWFac",
            FHWFac,
            "Human wealth is finite.",
            "Human wealth is infinite.",
        ),
        "PFFVAC": (
            "PFVAFac",
            PFVAFac,
            "The value of autarky is finite.",
            "The value of autarky is not finite.",
        ),
    }
    conditions = {}
    for name, (factor_name, factor, if_holds, if_fails) in table.items():
        factor = float(factor)
        holds = factor < 1.0
        conditions[name] = PerfForesightCondition(
            name=name,
            factor_name=factor_name,
            factor=factor,
            rate=np.log(factor) if factor > 0.0 else -np.inf,
            holds=holds,
            meaning=if_holds if holds else if_fails,
        )
    RIC = conditions["RIC"].holds
    FHWC = conditions["FHWC"].holds
    GIC = conditions["GICRaw"].holds

    hNrm = FHWFac / (1.0 - FHWFac) if FHWC else np.inf
    MPCmin = 1.0 - RPFac if RIC else 0.0
    a_min = -np.inf if BoroCnstArt is None else BoroCnstArt
    constraint_binds = bool(a_min > -hNrm)

    if constraint_binds and (RIC or GIC):
        regime = "kinked"
    elif not constraint_binds and RIC and FHWC:
        regime = "linear"
    elif GPFacRaw > 1.0:
        regime = "zero"
    elif GPFacRaw < 1.0:
        regime = "infinite"
    else:
        regime = "income"
    PFFVAC = conditions["PFFVAC"].holds
    solution_exists = regime == "linear" or (regime == "kinked" and (RIC or PFFVAC))

    if not (solution_exists and GIC):
        mNrmStable = np.nan
    elif constraint_binds:
        mNrmStable = 1.0 + a_min * Rfree / PermGroFac
    else:
        mNrmStable = -hNrm

    return PerfForesightConditions(
        conditions=conditions,
        regime=regime,
        solution_exists=solution_exists,
        constraint_binds=constraint_binds,
        MPCmin=float(MPCmin),
        MPCmax=1.0 if constraint_binds else float(MPCmin),
        hNrm=float(hNrm),
        mNrmStable=float(mNrmStable),
    )


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

    With use_infimum=True, the probability the atoms put on the worst income
    under the true laws, the product of the infima recorded in
    inc_shk_dstn.limit["infimum"]: zero unless an atom sits there, as a point
    mass of the laws does (unemployment that pays nothing, or a discrete law's
    lowest atom). An infimum the laws reach only continuously carries an
    infinitesimal probability, which sets the natural borrowing constraint
    (calc_boro_const_nat) but no weight in any expectation. With
    use_infimum=False, the probability of the lowest atom, which moves with the
    number of atoms.

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

    The natural borrowing constraint belongs to the model, not to the method
    that solves it. With use_infimum=True it is computed from the infima of the
    true shock laws, recorded in inc_shk_dstn.limit["infimum"]: each is treated
    as a realization of infinitesimal but positive probability, positive so that
    the consumer must be able to repay even there, infinitesimal so that it
    carries no weight in any expectation. Every discretization of the same laws,
    and the continuous expectation, then has the same constraint, which
    IndShock's solver imposes where the laws are bounded below away from zero
    (calc_boro_const_nat_from_infima). A discrete law records its lowest atom as
    its infimum, so for it the two coincide. With use_infimum=False it is the atoms'
    own limit, from the lowest atoms, which moves with the number of atoms;
    IndShock's solver uses it only to anchor its unconstrained consumption
    function below the imposed constraint, a numerical device that does not
    show in the solution.

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


def laws_bounded_below(inc_shk_dstn):
    """Whether the income shocks' true laws are bounded below away from zero.

    True where every infimum that inc_shk_dstn.limit records is positive. A
    lognormal unbounded below, HARK's default law for each shock, records an
    infimum of zero, and so does unemployment that pays nothing.

    Args:
        inc_shk_dstn (DiscreteDstn): Distribution of shocks to income.

    Returns:
        bool: Whether every recorded infimum is positive.
    """
    limit = getattr(inc_shk_dstn, "limit", None) or {}
    infima = limit.get("infimum")
    return infima is not None and bool(np.all(np.ravel(infima) > 0.0))


def calc_boro_const_nat_from_infima(m_nrm_min_next, inc_shk_dstn, rfree, perm_gro_fac):
    """Calculate the natural borrowing constraint from the true laws' infima.

    Where the laws are bounded below away from zero (laws_bounded_below), the
    constraint from their infima, as calc_boro_const_nat computes it with
    use_infimum=True; otherwise None. Where an infimum is zero, as for HARK's
    default lognormals, which are unbounded below, the infima's constraint would
    allow no borrowing at all where the atoms allow some, so IndShock's solver
    keeps the atoms' own limit there, as it did before it read the infima.

    Args:
        m_nrm_min_next (float): Minimum normalized market resources next period.
        inc_shk_dstn (DiscreteDstn): Distribution of shocks to income.
        rfree (float): Risk free interest factor.
        perm_gro_fac (float): Permanent income growth factor.

    Returns:
        float or None: The constraint from the infima, or None where the laws
        are not bounded below away from zero.
    """
    if not laws_bounded_below(inc_shk_dstn):
        return None
    return calc_boro_const_nat(
        m_nrm_min_next, inc_shk_dstn, rfree, perm_gro_fac, use_infimum=True
    )


def check_worst_transitory_point_mass(inc_shk_dstn, boro_const_art, boro_const_nat_inf):
    """
    Halt where the continuous expectation would solve a model whose natural
    borrowing limit binds while its worst transitory shock carries no
    probability. Where no artificial constraint is given, or one below the
    natural limit from the shock laws' infima, the worst realization of the
    transitory shock, by its true law, must be a point mass, as an unemployment
    event is: a zero-income event, or a positive benefit at or below every
    employed income. Where an artificial constraint is at or above the natural
    limit, the model is the artificial configuration, and is not checked.

    The laws are read as get_income_shock_laws() returns them, and the employed
    law's lowest value and the probability there by _law_end(), through the
    laws' interface whatever their family.

    Args:
        inc_shk_dstn (DiscreteDistribution): Distribution of shocks to income.
        boro_const_art (float or None): Artificial borrowing constraint.
        boro_const_nat_inf (float): Natural borrowing constraint from the
            shock laws' infima.

    Raises:
        ValueError: if the rule is violated.
    """
    if boro_const_art is not None and boro_const_art >= boro_const_nat_inf:
        return
    _, TranShkLaw, UnempPrb, IncUnemp = get_income_shock_laws(inc_shk_dstn)
    if UnempPrb >= 1.0:
        return  # always unemployed: the event is the only income
    lowest, order = _law_end(TranShkLaw, upper=False)
    if UnempPrb > 0.0 and IncUnemp <= lowest:
        return  # the unemployment event, a point mass, is the worst
    if order == 0:
        return  # the employed law holds probability at its lowest value
    worst = min(lowest, IncUnemp) if UnempPrb > 0.0 else lowest
    raise ValueError(
        "Where the natural borrowing limit binds (no artificial constraint, or "
        'one below it), the continuous expectation (IncShkExpectation "continuous") '
        "needs the worst transitory shock to carry probability, as an "
        f"unemployment event does: here the lowest transitory shock, {worst:.6g}, "
        "has none. Add an unemployment event (UnempPrb > 0) whose income is at or "
        "below every employed income, impose an artificial constraint at or above "
        f"the natural limit, {boro_const_nat_inf:.6g} (BoroCnstArt), or solve over "
        'the atoms (IncShkExpectation "atoms").'
    )


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


def _bounds(law):
    """
    The ends of a continuous law's support: its infimum and supremum where it
    records them, else its quantile function at zero and one.
    """
    if hasattr(law, "infimum") and hasattr(law, "supremum"):
        return float(np.ravel(law.infimum)[0]), float(np.ravel(law.supremum)[0])
    return float(np.ravel(law.ppf(0.0))[0]), float(np.ravel(law.ppf(1.0))[0])


class _ScaledLaw:
    """
    The law of scale * X, for X distributed as a continuous law: its quantile
    function, CDF, survival function, density, bounds and breakpoints, all
    scaled, and the standard-normal coordinate the law supplies, if it supplies
    one, with its values scaled, and that coordinate's density, if the law
    supplies it.

    Parameters
    ----------
    law : Distribution
        Any law with a quantile function, CDF and density.
    scale : float
        A positive scale.
    """

    def __init__(self, law, scale):
        self.law = law
        self.scale = float(scale)
        lo, hi = _bounds(law)
        self.infimum = np.array([self.scale * lo])
        self.supremum = np.array([self.scale * hi])

    def ppf(self, q):
        return self.scale * np.asarray(self.law.ppf(q))

    def cdf(self, x):
        return self.law.cdf(np.asarray(x) / self.scale)

    def sf(self, x):
        x = np.asarray(x) / self.scale
        return self.law.sf(x) if hasattr(self.law, "sf") else 1.0 - self.law.cdf(x)

    def pdf(self, x):
        return self.law.pdf(np.asarray(x) / self.scale) / self.scale

    def mean(self):
        return self.scale * float(np.squeeze(self.law.mean()))

    def breakpoints(self):
        # the law's own, scaled, where it has them
        breaks = getattr(self.law, "breakpoints", None)
        if breaks is None:
            return np.array([])
        return self.scale * np.asarray(breaks(), dtype=float)

    @property
    def normal_coordinate(self):
        # the law's own, if it has one (AttributeError otherwise)
        own, scale = self.law.normal_coordinate, self.scale

        def normal_coordinate():
            bot, top, value, coord = own()
            return (
                bot,
                top,
                lambda z: scale * np.asarray(value(z)),
                lambda x: coord(np.asarray(x) / scale),
            )

        return normal_coordinate

    @property
    def coordinate_density(self):
        # the law's own, if it has one (AttributeError otherwise): the scale
        # moves the values, not the coordinate
        return self.law.coordinate_density


class _Mixture:
    """
    A mixture as the continuous expectation reads one: its weights and its
    components, with the support's ends.
    """

    def __init__(self, weights, components):
        self.weights = np.asarray(weights, dtype=float)
        self.components = list(components)
        ends = [
            (np.min(part.atoms), np.max(part.atoms))
            if isinstance(part, DiscreteDistribution)
            else _bounds(part)
            for weight, part in zip(self.weights, self.components)
            if weight > 0.0
        ]
        self.infimum = np.array([min(lo for lo, _ in ends)])
        self.supremum = np.array([max(hi for _, hi in ends)])


def _scaled_law(law, scale):
    """
    The law of scale * X, for X distributed as law: a discrete law's atoms
    scaled, a mixture scaled component by component, and any other law a
    _ScaledLaw, which keeps the coordinate the law supplies.
    """
    if scale == 1.0:
        return law
    if isinstance(law, DiscreteDistribution):
        return DiscreteDistribution(law.pmv, scale * law.atoms)
    if hasattr(law, "components") and hasattr(law, "weights"):
        return _Mixture(
            law.weights, [_scaled_law(part, scale) for part in law.components]
        )
    return _ScaledLaw(law, scale)


def get_income_shock_laws(IncShkDstn):
    """
    The continuous laws behind one period's joint distribution of permanent and
    transitory income shocks, read from the ``limit`` that IndShock's income
    constructor records: the permanent shock's law; the transitory shock's law
    while employed, scaled to the mean (1 - p x) / (1 - p) that the
    unemployment event leaves it; the probability p of unemployment; and the
    income x it pays. Unemployment here is IndShock's: a point mass x in the
    transitory shock, not a state of a Markov model.

    Parameters
    ----------
    IncShkDstn : DiscreteDistribution
        One period's joint income-shock distribution, as BufferStockIncShkDstn
        makes it.

    Returns
    -------
    PermShkLaw : Distribution
        The law of the permanent shock.
    TranShkLaw : Distribution
        The law of the transitory shock while employed.
    UnempPrb : float
        The probability of unemployment.
    IncUnemp : float
        The transitory income while unemployed.
    """
    limit = getattr(IncShkDstn, "limit", None) or {}
    dists = limit.get("dist")
    if limit.get("method") != "combine_indep_dstns" or len(dists) != 2:
        raise ValueError(
            "This income distribution does not record the laws its atoms come "
            "from: the continuous expectation needs IncShkDstn as IndShock's "
            "income constructor builds it."
        )
    perm, tran = dists
    PermShkLaw = perm.limit["dist"]
    if tran.limit.get("method") == "add_discrete_outcome_constant_mean":
        UnempPrb, IncUnemp = float(tran.limit["p"]), float(tran.limit["x"])
        employed = tran.limit["dist"].limit["dist"]
        scale = (1.0 - UnempPrb * IncUnemp) / (1.0 - UnempPrb)
    else:
        UnempPrb, IncUnemp = 0.0, 0.0
        employed = tran.limit["dist"]
        scale = 1.0
    return PermShkLaw, _scaled_law(employed, scale), UnempPrb, IncUnemp


def get_mNrm_kinks(solution):
    """
    The cash-on-hand points at which a solution's consumption function is not
    smooth: its recorded ``mNrmKinks``, or else the point at which its borrowing
    constraint stops binding, found by bisection on the constraint line
    c = m - mNrmMin. Empty when consumption equals cash-on-hand everywhere, as
    in the terminal period, or when the constraint binds only at the lowest
    cash-on-hand itself, as where the natural constraint binds.

    Parameters
    ----------
    solution : ConsumerSolution
        A one-period solution.

    Returns
    -------
    np.array
    """
    kinks = getattr(solution, "mNrmKinks", None)
    if kinks is not None:
        return np.asarray(kinks, dtype=float)
    cFunc = solution.cFunc
    m_min = float(solution.mNrmMin)
    lo, hi, top = m_min, 50.0, max(50.0, m_min + 50.0)
    unconstrained = getattr(cFunc, "functions", [None])[0]
    if hasattr(unconstrained, "x_list"):
        # The envelope is below the line exactly where its unconstrained part is
        x = np.asarray(unconstrained.x_list, dtype=float)
        lo, top = float(x[0]), float(x[-1])
        hi = min(top, hi)
        cFunc = unconstrained

    def below_line(m):
        return float(cFunc(np.array([m]))[0]) < m - m_min

    # The search looks below fifty first, where kinks lie in practice, then up
    # to the unconstrained part's last knot
    if not below_line(hi):
        if not (top > hi and below_line(top)):
            return np.array([])
        hi = top
    start = lo
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi:  # lo and hi are adjacent floats
            break
        if below_line(mid):
            hi = mid
        else:
            lo = mid
    kink = 0.5 * (lo + hi)
    if kink <= start + 8.0 * np.finfo(float).eps * max(1.0, abs(start)):
        return np.array([])  # below the line from the lowest point on
    return np.array([kink])


def _steps_in_values(law):
    """
    The values at which next period's kink cuts the permanent shock's integral,
    where it reaches them in the transitory shock: the transitory law's
    coordinate at each multiple of Z_STEP within STEPS_WITHIN of its center,
    and its ends where it holds probability there (_end_order); for a mixture,
    its components'; for a discrete law, its atoms. The kink leaves the
    permanent integrand a bump as wide as the transitory law, which these cuts
    resolve.
    """
    if isinstance(law, DiscreteDistribution):
        return np.unique(np.asarray(law.atoms, dtype=float))
    if hasattr(law, "components"):
        return np.unique(
            np.concatenate([_steps_in_values(part) for part in law.components])
        )
    edges, value, _, _ = _coordinate(law)
    edges = np.asarray(edges, dtype=float)
    if STEPS_WITHIN is not None:
        edges = edges[np.abs(edges) <= STEPS_WITHIN]
    lo, hi = _bounds(law)
    ends = [
        x
        for x, upper in ((lo, False), (hi, True))
        if np.isfinite(x) and _end_order(law, x, upper) is not None
    ]
    return np.unique(np.r_[np.asarray(value(edges), dtype=float), ends])


# How far from a near-pole beyond an end of a law's support, in the law's
# standard-normal coordinate, the graded rule applies: beyond it, a piece one step
# wide sees the pole at least seven half-widths away
GRADE_WITHIN = 3.0
# The step in the logarithm of the distance to the near-pole at which the graded
# part is cut: integrands there grow or fall like that distance's powers, so
# exponentially in its logarithm, and half a step keeps a few nodes a piece as
# accurate as on the unit steps of the law's coordinate
GRADE_STEP = 0.5
# How far out in the transitory law's coordinate its steps cut the permanent
# shock's integral where next period's kink reaches them: the kink's bump in the
# permanent integrand beyond three standard deviations carries too little of the
# law to matter
STEPS_WITHIN = 3.0


def _end_order(law, x, upper):
    """
    The power k with which a continuous law's probability approaches a finite end
    x of its support, P(X within h of x) ~ h^k as h falls: 1 for a density
    positive at x, 2 for one that rises linearly from zero there, and so on; None
    where the probability falls faster than any power, as through a tail. Read
    from the density's one-sided limit, f(x + h) ~ h^(k - 1) (f(x - h) at an upper
    end), at distances h falling by factors of ten from the distance that holds
    the law's first thousandth at that end: the estimates settle, their changes
    shrinking, where there is a power, and keep drifting through a tail.
    """
    q = 1.0 - 1e-3 if upper else 1e-3
    h = abs(float(np.ravel(law.ppf(np.array([q])))[0]) - x) * 10.0 ** -np.arange(14)
    h = h[h > 1e-12 * abs(x)]  # well clear of the rounding of x itself
    with np.errstate(all="ignore"):
        f = np.ravel(law.pdf(x + (-1.0 if upper else 1.0) * np.r_[h, 2.0 * h]))
        k = 1.0 + np.log(f[h.size :] / f[: h.size]) / np.log(2.0)
    inside = False  # the widest probes can leave the support
    for i in range(1, h.size):
        if not (np.isfinite(k[i - 1]) and np.isfinite(k[i])):
            if inside:
                return None  # the density vanishes toward the end: no power
            continue
        inside = True
        change, size = abs(k[i] - k[i - 1]), max(1.0, abs(k[i]))
        # an exact power, up to rounding, or changes shrinking toward one
        if change <= 1e-6 * size or (
            i > 1 and change <= 1e-4 * size and change <= 0.5 * abs(k[i - 1] - k[i - 2])
        ):
            return float(np.round(k[i])) if abs(k[i] - np.round(k[i])) < 1e-3 else k[i]
    return None  # still drifting, as through a tail


def _graded_quadrature(law, n, pole, cuts=None):
    """
    Nodes and weights against a continuous law for an integrand that may grow
    like the inverse distance to a near-pole beyond an end of the support, at
    the values ``pole`` (an array, NaN where there is none), split at the
    values ``cuts`` (pole.shape + (k,), or None), where the integrand is not
    smooth. In the law's standard-normal coordinate z, with the pole at z_p a
    distance d beyond a finite end: where d < GRADE_WITHIN, the part of the
    support within GRADE_WITHIN of the pole is integrated in u = log|z - z_p|,
    in which the near-pole is smooth, cut at each GRADE_STEP of u, at each half
    of Z_STEP in z and at the cuts; the rest in z, cut at each multiple of Z_STEP
    and at the cuts, as expected_continuous() cuts it; n Gauss-Legendre nodes a
    piece. Where a law's end is finite in its values but not in its coordinate,
    as a law reached through its quantile function has it, the same is done in
    its values, with the part from the end to the law's median graded, at an end
    the law holds probability at (_end_order), and cut at the half steps of its
    coordinate within STEPS_WITHIN of its center. Returns the values and the
    weights, each pole.shape + (m,), the weights carrying the law's density, and
    whether each row is graded.
    """
    edges, value, coord, density = _coordinate(law)
    lo, hi = edges[0], edges[-1]
    pole = np.asarray(pole, dtype=float)
    rows = pole.shape
    if cuts is None:
        cuts = np.full(rows + (0,), np.nan)
    cuts = np.asarray(cuts, dtype=float)
    in_z = lo > -Z_UNBOUNDED or hi < Z_UNBOUNDED  # a finite end in z
    with np.errstate(divide="ignore", invalid="ignore"):
        if in_z:
            t_lo, t_hi, tp = lo, hi, coord(pole)
            open_lo, open_hi = lo > -Z_UNBOUNDED, hi < Z_UNBOUNDED
            # the graded part is also cut at the coordinate's half steps
            half = 0.5 * Z_STEP
            steps = half * np.arange(np.ceil(lo / half), np.floor(hi / half) + 1.0)
            t_cuts = np.concatenate(
                [coord(cuts), np.broadcast_to(steps, rows + steps.shape)], axis=-1
            )
        else:
            t_lo, t_hi = _bounds(law)
            # in values, the graded part is also cut at the half steps of the law's
            # coordinate within STEPS_WITHIN of its center
            half = 0.5 * Z_STEP
            reach = Z_UNBOUNDED if STEPS_WITHIN is None else STEPS_WITHIN
            steps = half * np.arange(
                -np.floor(reach / half), np.floor(reach / half) + 1
            )
            steps = np.asarray(value(steps), dtype=float)
            tp = pole
            t_cuts = np.concatenate(
                [
                    np.broadcast_to(cuts, rows + cuts.shape[-1:]),
                    np.broadcast_to(steps, rows + steps.shape),
                ],
                axis=-1,
            )
            open_lo = (
                bool(np.isfinite(t_lo)) and _end_order(law, t_lo, False) is not None
            )
            open_hi = (
                bool(np.isfinite(t_hi)) and _end_order(law, t_hi, True) is not None
            )
            median = float(np.ravel(law.ppf(np.array([0.5])))[0])
        # a pole at an end, to within rounding, as at the natural limit itself, is
        # graded too
        tol_lo = 1e-12 * max(1.0, abs(t_lo))
        tol_hi = 1e-12 * max(1.0, abs(t_hi))
        below = open_lo & np.isfinite(tp) & (tp <= t_lo + tol_lo)
        above = open_hi & np.isfinite(tp) & (tp >= t_hi - tol_hi) & ~below
        d = np.where(
            below,
            np.maximum(t_lo - tp, 0.0),
            np.where(above, np.maximum(tp - t_hi, 0.0), np.inf),
        )
        if in_z:
            graded = d < GRADE_WITHIN
            g = np.where(
                below,
                np.minimum(tp + GRADE_WITHIN, t_hi),
                np.maximum(tp - GRADE_WITHIN, t_lo),
            )
            z_g = g
        else:
            graded = np.where(below, d < median - t_lo, d < t_hi - median)
            g = np.full(rows, median)
            z_g = coord(g)
    graded = graded & (below | above)

    # the rest, in z, cut at each step and at the cuts
    z_from = np.where(graded & below, z_g, lo)
    z_to = np.where(graded & above, z_g, hi)
    z_cuts = np.nan_to_num(coord(cuts), nan=lo, posinf=hi, neginf=lo)
    plain = np.concatenate(
        [
            np.broadcast_to(np.asarray(edges, dtype=float), rows + (len(edges),)),
            np.broadcast_to(z_cuts, rows + z_cuts.shape[-1:]),
            z_from[..., None],
            z_to[..., None],
        ],
        axis=-1,
    )
    plain = np.sort(np.clip(plain, z_from[..., None], z_to[..., None]), axis=-1)
    z, wz = _split_gauss_legendre(plain, n)
    x_plain, w_plain = value(z), wz * density(z)
    if not np.any(graded):
        return x_plain, w_plain, graded

    # the graded part, in u = log of the distance to the pole, cut at each step of
    # GRADE_STEP in u and at the cuts
    sign = np.where(above, -1.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        # nearer than this the nodes would round onto the end itself
        floor = 1e-14 * np.maximum(1.0, np.abs(np.where(above, t_hi, t_lo)))
        near = np.maximum(np.where(graded, d, 1.0), floor)
        u_near = np.where(graded, np.log(near), 0.0)
        u_far = np.where(graded, np.log(np.where(graded, sign * (g - tp), 1.0)), 0.0)
        u_cuts = np.log(sign[..., None] * (t_cuts - tp[..., None]))
    # cuts on the pole's side of the end, or none, fall at the graded part's end
    u_cuts = np.where(np.isfinite(u_cuts), u_cuts, u_far[..., None])
    steps = GRADE_STEP * np.arange(
        np.floor(u_near[graded].min() / GRADE_STEP) + 1.0,
        np.ceil(u_far[graded].max() / GRADE_STEP),
    )
    part = np.concatenate(
        [
            u_near[..., None],
            np.broadcast_to(steps, rows + steps.shape),
            np.broadcast_to(u_cuts, rows + u_cuts.shape[-1:]),
            u_far[..., None],
        ],
        axis=-1,
    )
    part = np.sort(np.clip(part, u_near[..., None], u_far[..., None]), axis=-1)
    u, wu = _split_gauss_legendre(part, n)
    e = np.exp(u)
    t = np.where(graded[..., None], tp[..., None] + sign[..., None] * e, t_lo)
    with np.errstate(invalid="ignore"):
        if in_z:
            x_part, w_part = value(t), wu * e * density(t)
        else:
            x_part, w_part = t, wu * e * law.pdf(t)
    w_part = np.where(graded[..., None], w_part, 0.0)
    x_part = np.where(graded[..., None], x_part, x_plain[..., :1])
    return (
        np.concatenate([x_part, x_plain], axis=-1),
        np.concatenate([w_part, w_plain], axis=-1),
        graded,
    )


def _near_limit_nodes(base, m_min, PermShkLaw, TranShkLaw, nPerm, nTran, gaps, breaks):
    """
    Nodes and weights for calc_BegOfPrd_continuous() at the assets whose landing
    points come close to next period's minimum cash-on-hand m_min, where its
    consumption is zero, or None where none do. Landing points
    m' = base / psi + theta come closest at the transitory shock's lowest value
    theta_lo, and toward m_min the integrand grows like the inverse distance to
    it: as a function of the permanent shock, a near-pole at
    psi_p = base / (m_min - theta_lo); of the transitory shock, at
    theta_p = m_min - base / psi. Each integral takes _graded_quadrature()
    toward its pole, split as calc_BegOfPrd_continuous() splits it: the
    transitory integral where m' reaches next period's non-smooth points,
    ``breaks``, and the permanent at base / gaps. For continuous laws that hold
    probability at a finite end (_end_order), the transitory one at its lowest
    value, or a transitory point, such as the income in IndShock's unemployment,
    a point mass in its transitory shock. For a mixture, a discrete permanent
    law, or a law without such an end, None, and expected_continuous()'s rule
    applies. Returns the assets graded (a boolean mask over base), and for them
    the permanent shock's values and weights, each (A, Np), and the transitory
    shock's, each (A, Np, Nt).
    """

    def held(law, upper):
        # whether the law holds probability at that finite end, as a power of the
        # distance (_end_order); not for a mixture or a discrete law
        if isinstance(law, DiscreteDistribution) or hasattr(law, "components"):
            return False
        x = _bounds(law)[1 if upper else 0]
        return bool(np.isfinite(x)) and _end_order(law, x, upper) is not None

    if not (held(PermShkLaw, False) or held(PermShkLaw, True)):
        return None
    if isinstance(TranShkLaw, DiscreteDistribution):
        atoms = np.unique(TranShkLaw.atoms[0])
        if atoms.size != 1:
            return None
        theta_lo, point = float(atoms[0]), True
    elif held(TranShkLaw, False):
        theta_lo, point = _bounds(TranShkLaw)[0], False
    else:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        psi_pole = base / (m_min - theta_lo)
        psi_cuts = base[:, None] / gaps  # those outside the support are dropped
    psi, w_psi, graded = _graded_quadrature(PermShkLaw, nPerm, psi_pole, psi_cuts)
    if not point:
        # the transitory integral's pole is closest at the permanent law's lowest
        # value
        with np.errstate(divide="ignore", invalid="ignore"):
            closest = m_min - base / _bounds(PermShkLaw)[0]
        graded = graded | _graded_quadrature(TranShkLaw, nTran, closest)[2]
    if not np.any(graded):
        return None
    psi, w_psi, base = psi[graded], w_psi[graded], base[graded]
    if point:
        return (
            graded,
            psi,
            w_psi,
            np.full(psi.shape + (1,), theta_lo),
            np.ones(psi.shape + (1,)),
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        theta_pole = m_min - base[:, None] / psi
    theta_cuts = breaks - (base[:, None] / psi)[..., None]
    theta, w_theta, _ = _graded_quadrature(TranShkLaw, nTran, theta_pole, theta_cuts)
    return graded, psi, w_psi, theta, w_theta


def calc_BegOfPrd_continuous(
    aNrm,
    solution_next,
    PermShkLaw,
    TranShkLaw,
    Rfree,
    PermGroFac,
    CRRA,
    CubicBool=False,
    vFuncBool=False,
    PermShkQuadCount=4,
    TranShkQuadCount=4,
    split_at_knots=False,
):
    """
    The expectations over next period's income shocks that a one-period solver
    needs, for one next-period state, taken against the shocks' continuous laws
    instead of summed over atoms: the counterparts of expected() over
    calc_vp_next, calc_vpp_next and calc_v_next. With
    m' = Rfree aNrm / (PermGroFac psi) + theta, they are

        E1 = E[psi^(-rho) v'(m')],
        E2 = E[psi^(-rho-1) v''(m')],
        E0 = E[(PermGroFac psi)^(1-rho) v(m')],

    with v' = c^(-rho) and v'' = -rho c^(-rho-1) c' from one evaluation of next
    period's consumption function c and its slope. The integrals are taken by
    expected_continuous(), with PermShkQuadCount and TranShkQuadCount
    Gauss-Legendre nodes a piece, cut where the integrand is not smooth, except
    at the knots of next period's consumption function: at each unit step of the
    laws' standard-normal coordinates; in the transitory shock, where m' reaches
    next period's kink (get_mNrm_kinks), at which marginal value is kinked, or
    the end of its linear first segment (mNrmLinearEnd), where the cubic's
    slope steps from that line's to the MPC; and
    in the permanent shock, where that point reaches the transitory law's unit
    steps within STEPS_WITHIN of its center, its ends, or its point. The kink
    leaves the permanent integrand a bump as wide as the transitory law, which
    those cuts resolve. The knots are not cut at, though linear interpolation
    kinks marginal value at each of them and a cubic kinks its slope: cutting
    at them as well moves the solved consumption function and its MPC by small
    fractions of the grid's own error; split_at_knots cuts at them too.
    Each asset's integrals are taken
    on their own pieces. Where m' can land just above next period's minimum
    cash-on-hand, at which its consumption is zero, at the corner of the
    shocks' support, the integrand grows like an inverse power of the distance
    to it: both integrals then take the rule graded toward that near-pole inside
    the same pieces (_near_limit_nodes, _graded_quadrature). A solver's
    expectation is the probability-weighted sum of these over the next-period
    states: in IndShock's, the employed and the unemployed.

    Parameters
    ----------
    aNrm : np.array
        End-of-period assets.
    solution_next : ConsumerSolution
        Next period's solution in this state.
    PermShkLaw : Distribution
        The permanent shock's law, continuous or a point.
    TranShkLaw : Distribution
        The transitory shock's law in this state, continuous or a point, such
        as the income paid in IndShock's unemployment, a point mass in its
        transitory shock.
    Rfree : float
        Risk free interest factor.
    PermGroFac : float
        Permanent income growth factor.
    CRRA : float
        Coefficient of relative risk aversion.
    CubicBool : bool, optional
        Whether to compute E2, for a cubic consumption function. The default is
        False.
    vFuncBool : bool, optional
        Whether to compute E0, for the value function. The default is False.
    PermShkQuadCount : int, optional
        Gauss-Legendre nodes a piece of the permanent shock's integral. The
        default is 4.
    TranShkQuadCount : int, optional
        Gauss-Legendre nodes a piece of the transitory shock's integral. The
        default is 4.
    split_at_knots : bool, optional
        Whether to cut the integrals at every knot of next period's consumption
        function as well. Each asset's integrand is then evaluated at many more
        points, in proportion to the knots its landing points cross, for the
        small change described above; IndShock's solver does not use it. The
        default is False.

    Returns
    -------
    E1 : np.array
        E[psi^(-rho) v'(m')] at each aNrm.
    E2 : np.array or None
        E[psi^(-rho-1) v''(m')] at each aNrm, when CubicBool.
    E0 : np.array or None
        E[(PermGroFac psi)^(1-rho) v(m')] at each aNrm, when vFuncBool.
    """
    shape = np.shape(aNrm)
    base = Rfree * np.atleast_1d(np.asarray(aNrm, dtype=float)) / PermGroFac
    m_min = float(solution_next.mNrmMin)
    # Where next period's marginal value is kinked: its kink, and the end of a
    # linear first segment, from its limit. Its consumption function's other
    # knots are not cut at (see above)
    breaks = np.unique(
        np.r_[
            get_mNrm_kinks(solution_next), getattr(solution_next, "mNrmLinearEnd", [])
        ]
    )
    if split_at_knots:
        # and, where asked, every knot of its consumption function, where a cubic's
        # second derivative steps: the expectations' accurate rule
        part = getattr(solution_next.cFunc, "functions", [solution_next.cFunc])[0]
        knots = np.asarray(getattr(part, "x_list", ()), dtype=float)
        breaks = np.union1d(breaks, knots[knots > m_min])
    # The permanent shock's integral is cut where that point reaches the
    # transitory law's steps, or its point
    steps = _steps_in_values(TranShkLaw)
    gaps = (breaks[:, None] - steps[None, :]).ravel()

    def psi_cuts(base):
        # where the transitory splits reach the transitory law's steps or its point,
        # for assets of either sign: those outside the support are dropped
        with np.errstate(divide="ignore", invalid="ignore"):
            return base[..., None] / gaps

    def theta_cuts(psi, base):
        return breaks - (base / psi)[..., None]

    def integrand(shocks, base):
        psi = shocks["PermShk"]
        m_next = base / psi + shocks["TranShk"]
        c_next, mpc_next = solution_next.cFunc.eval_with_derivative(m_next)
        vP = c_next ** (-CRRA)
        out = [psi ** (-CRRA) * vP]
        if CubicBool:
            out.append(-CRRA * psi ** (-CRRA - 1.0) * vP * mpc_next / c_next)
        if vFuncBool:
            out.append((PermGroFac * psi) ** (1.0 - CRRA) * solution_next.vFunc(m_next))
        return np.stack(np.broadcast_arrays(*out))

    # Landing points near next period's minimum, where its consumption is zero,
    # make the integrand a near-pole at the corner of the shocks' support, graded
    # at the assets that reach it
    near = _near_limit_nodes(
        base,
        m_min,
        PermShkLaw,
        TranShkLaw,
        PermShkQuadCount,
        TranShkQuadCount,
        gaps,
        breaks,
    )
    graded = np.zeros(base.shape, dtype=bool) if near is None else near[0]
    E = np.empty((1 + CubicBool + vFuncBool,) + base.shape)
    # One asset at a time: the assets' pieces differ in number, many permanent and
    # few transitory ones at high assets, the reverse at low, and a call pads each
    # row to the most any row has
    for i in np.flatnonzero(~graded):
        E[:, i] = expected_continuous(
            integrand,
            [PermShkLaw, TranShkLaw],
            args=(base[i : i + 1],),
            labels=["PermShk", "TranShk"],
            splits=[psi_cuts, theta_cuts],
            nodes=[PermShkQuadCount, TranShkQuadCount],
        )[:, 0]
    if near is not None:
        _, psi, w_psi, theta, w_theta = near
        f = integrand(
            {"PermShk": psi[..., None], "TranShk": theta},
            base[graded][:, None, None],
        )
        E[:, graded] = np.sum(f * (w_psi[..., None] * w_theta), axis=(-2, -1))
    E = E.reshape((E.shape[0],) + shape)
    E1 = E[0]
    E2 = E[1] if CubicBool else None
    E0 = E[-1] if vFuncBool else None
    return E1, E2, E0


def _end_mass(law, x, upper):
    """
    The probability a law holds at the finite end x of its support itself, as a
    point mass at a law's infimum (PointMassAtInfimum) holds its probability: its
    CDF at a lower end, and what its CDF leaves just below an upper end. Zero
    for a law that has no CDF.
    """
    cdf = getattr(law, "cdf", None)
    if cdf is None:
        return 0.0
    at = np.nextafter(x, -np.inf) if upper else x
    below = float(np.ravel(cdf(np.array([at])))[0])
    return 1.0 - below if upper else below


def _law_end(law, upper):
    """
    One end of a law's support, and how the law holds probability there: 0 for a
    mass, as a discrete law has at each end and PointMassAtInfimum at its
    infimum (_end_mass); otherwise the power k with which its probability
    approaches the end, P(X within h of it) ~ h^k (_end_order), 1 for a density
    positive there, as a law truncated there has; None where the law approaches
    the end only through a tail, as an untruncated lognormal approaches zero. A
    mixture law (an object with weights and components) holds its end as the
    components with weight that reach it do.
    """
    if isinstance(law, DiscreteDistribution):
        x = np.asarray(law.atoms, dtype=float)
        return (x.max() if upper else x.min()), 0
    if hasattr(law, "components") and hasattr(law, "weights"):
        parts = [
            _law_end(part, upper)
            for weight, part in zip(law.weights, law.components)
            if weight > 0.0
        ]
        x = max(p[0] for p in parts) if upper else min(p[0] for p in parts)
        held = [order for at, order in parts if at == x and order is not None]
        return x, (min(held) if held else None)
    lo, hi = _bounds(law)
    x = hi if upper else lo
    if lo == hi:
        return x, 0  # a point
    if not np.isfinite(x):
        return x, None
    if _end_mass(law, x, upper) > 1e-12:
        return x, 0  # a point mass at the end, as PointMassAtInfimum puts there
    return x, _end_order(law, x, upper)


def _end_probability(law, upper):
    """
    The probability a law holds at one end of its support itself: the atom's
    there, for a discrete law; a point mass's at the end (_end_mass); zero for a
    density.
    A mixture law holds what its components with weight that reach the end hold.
    """
    if isinstance(law, DiscreteDistribution):
        x = np.asarray(law.atoms, dtype=float).ravel()
        prb = np.asarray(law.pmv, dtype=float).ravel()
        return float(np.sum(prb[x == (x.max() if upper else x.min())]))
    if hasattr(law, "components") and hasattr(law, "weights"):
        parts = [
            (_law_end(part, upper)[0], weight * _end_probability(part, upper))
            for weight, part in zip(law.weights, law.components)
            if weight > 0.0
        ]
        x = max(at for at, _ in parts) if upper else min(at for at, _ in parts)
        return float(sum(mass for at, mass in parts if at == x))
    lo, hi = _bounds(law)
    x = hi if upper else lo
    if lo == hi:
        return 1.0  # a point
    if not np.isfinite(x):
        return 0.0
    return max(_end_mass(law, x, upper), 0.0)


def _worst_income_mass(laws):
    """
    The probability the income shocks' laws, (PermShkLaw, TranShkLaw, UnempPrb,
    IncUnemp) as get_income_shock_laws() returns them, put on the worst income
    itself, which sets the MPC at the natural borrowing constraint
    (calc_mpc_max): zero unless a point mass sits there. An unemployment event
    that pays nothing is one, whatever the permanent shock. Otherwise it is the
    probability of the lowest transitory shock (the unemployment event's, or the
    mass the employed law holds at its lowest value, whichever is the lower
    shock) times the mass the permanent shock's law holds at its lowest value,
    which a point mass at the infimum (PointMassAtInfimum) or a discrete law has
    and a density has not.
    """
    PermShkLaw, TranShkLaw, UnempPrb, IncUnemp = laws
    branches = []  # (probability, lowest transitory shock, the mass held there)
    if UnempPrb < 1.0:
        branches.append(
            (
                1.0 - UnempPrb,
                _law_end(TranShkLaw, upper=False)[0],
                _end_probability(TranShkLaw, upper=False),
            )
        )
    if UnempPrb > 0.0:
        branches.append((UnempPrb, IncUnemp, 1.0))
    lowest = min(at for _, at, _ in branches)
    tran = sum(prb * mass for prb, at, mass in branches if at == lowest)
    if lowest == 0.0:
        return tran
    return tran * _end_probability(PermShkLaw, upper=False)


def _reaches_natural_limit(aNrm, mNrmMinNext, laws, Rfree, PermGroFac, CRRA):
    """
    Whether end-of-period assets aNrm leave next period's expected marginal value
    infinite under the income shocks' laws, (PermShkLaw, TranShkLaw, UnempPrb,
    IncUnemp) as get_income_shock_laws() returns them: whether some branch of
    IndShock's income, the employed or the unemployed (a point mass in the
    transitory shock), can land on next period's minimum cash-on-hand, where
    consumption is zero, with enough probability. A mass there makes it infinite;
    so does a positive density at the end of each of the d laws that set the
    landing point otherwise, when CRRA >= d (_law_end). A law that approaches its
    end only through a tail, as an untruncated lognormal approaches zero, puts
    too little probability there.

    Where this holds at the lowest assets allowed, the natural borrowing
    constraint binds, and consumption falls to zero at it with no kink; where it
    does not, the lowest assets are a constraint that binds, with a kink.
    """
    PermShkLaw, TranShkLaw, UnempPrb, IncUnemp = laws
    branches = [TranShkLaw] if UnempPrb < 1.0 else []
    if UnempPrb > 0.0:
        branches.append(DiscreteDistribution([1.0], [IncUnemp]))
    tol = 1e-12 * max(1.0, abs(mNrmMinNext))
    for law in branches:
        landing, order = _law_end(law, upper=False)
        if order is None:
            continue
        if aNrm != 0.0:  # the permanent shock's worst end for this sign of aNrm
            psi, psi_order = _law_end(PermShkLaw, upper=aNrm > 0.0)
            if psi_order is None:
                continue
            landing += Rfree * aNrm / (PermGroFac * psi)
            order += psi_order
        if landing <= mNrmMinNext + tol and (order == 0 or CRRA >= order):
            return True
    return False


# Where the natural constraint binds with no kink, the asset grid also takes
# points placed by the transition of next period's consumption function, where
# its MPC falls from near 1 toward its value under buffer-stock saving
# (_natural_limit_assets). These constants are tuned, on five calibrations of
# that case, not derived. The fractions of the MPC's fall at the transition's
# lower and upper edges:
TRANSITION_EDGES = (0.02, 0.6)
# The distance above the limit to which the fall is measured, and the least fall
# that makes a transition:
TRANSITION_REFERENCE = 3.0
TRANSITION_LEAST_FALL = 0.05
# The half-width of the secant that reads a linear interpolant's MPC, whose
# derivative steps at each gridpoint:
TRANSITION_SECANT = 0.05
# The points close to the limit, below the transition, across it and above it;
# where the points below it start, as a fraction of its lower edge; and how far
# the points above it reach, as a multiple of its upper edge in assets:
TRANSITION_SPLIT = (2, 4, 8, 3)
TRANSITION_BELOW_START = 0.3
TRANSITION_ABOVE_TOP = 4.0


def _mpc_transition(cFunc, mNrmMin, first, CubicBool):
    """
    The transition of a consumption function cFunc that falls to zero at the
    natural limit mNrmMin with no kink, where its MPC falls from near 1 toward
    its value under buffer-stock saving: the distances d = m - mNrmMin of the
    transition's lower and upper edges, where the MPC has fallen by the
    fractions TRANSITION_EDGES of its fall from its value just above the limit
    to its value at d = TRANSITION_REFERENCE. None where the MPC falls by no
    more than TRANSITION_LEAST_FALL, as a terminal consumption function's MPC
    does not fall at all.

    The MPC is read at 800 distances log-spaced from a thousandth of first, the
    asset grid's first point, to 20: the derivative for a cubic (CubicBool), and
    for linear interpolation the secant from d - TRANSITION_SECANT to
    d + TRANSITION_SECANT. Each edge is interpolated between the two distances
    around its crossing. Both keep the edges moving continuously with cFunc: an
    edge read off a linear interpolant's derivative, or at the scanned distances
    themselves, jumps, and an asset grid placed by it can then cycle forever
    over an infinite horizon, where convergence is judged on the gridpoints.
    """
    d = np.geomspace(1e-3 * first, 20.0, 800)
    if CubicBool:
        mpc = cFunc.derivative(mNrmMin + d)
    else:
        below = np.maximum(d - TRANSITION_SECANT, 0.0)
        mpc = (cFunc(mNrmMin + d + TRANSITION_SECANT) - cFunc(mNrmMin + below)) / (
            d + TRANSITION_SECANT - below
        )
    top = mpc[0]
    fall = top - np.interp(TRANSITION_REFERENCE, d, mpc)
    if not fall > TRANSITION_LEAST_FALL:
        return None
    edges = []
    for share in TRANSITION_EDGES:
        level = top - share * fall
        crossed = np.flatnonzero(mpc < level)
        if crossed.size == 0 or crossed[0] == 0:
            return None
        i = crossed[0]
        t = (mpc[i - 1] - level) / (mpc[i - 1] - mpc[i])
        edges.append(d[i - 1] + t * (d[i] - d[i - 1]))
    return edges[0], edges[1]


def _natural_limit_assets(solution_next, mNrmMin, first, CubicBool):
    """
    The end-of-period assets the grid takes, besides its own, where the natural
    constraint at mNrmMin binds with no kink. Consumption there stays close to
    cash-on-hand down to the limit, then turns toward buffer-stock saving, and
    the end-of-period value falls steeply toward the limit: to minus infinity,
    like a logarithm, where the law reaching the limit has a positive density at
    its end, and to a finite value where its density falls to zero there fast
    enough, as the jumpless law's does at HARK's default CRRA. As an
    artificial constraint's kink is where it stops binding, that transition is
    where the natural one stops pressing: the points cover the transition, the
    first segment below it, and the end-of-period value near the limit.

    The transition is located on next period's consumption function
    (_mpc_transition), which stands in for this period's, and each distance d
    above the limit maps to the end-of-period assets left by consuming c there,
    a = d - c(d), with next period's c. With TRANSITION_SPLIT (2, 4, 8, 3) the
    points are: 2 close to the limit, at a millionth and a thousandth of first,
    the asset grid's first point above it; 4 evenly spaced in d from
    TRANSITION_BELOW_START (0.3) times the transition's lower edge up to it; 8
    log-spaced in assets across the transition; and 3 above it, up to
    TRANSITION_ABOVE_TOP (4) times its upper edge in assets. Where next
    period's consumption function has no transition, as when it is the terminal
    one, they are 17 points half a decade apart, from 1e-6 to 100 times first.

    The constants (the edges at 2% and 60% of the MPC's fall, the fall measured
    to 3 above the limit, the secant's half-width 0.05, the split 2/4/8/3, and
    0.3 and 4) are tuned, on five calibrations of the natural limit, not derived.
    """
    edges = _mpc_transition(
        solution_next.cFunc, solution_next.mNrmMin, first, CubicBool
    )
    if edges is None:
        return mNrmMin + first * np.geomspace(1e-6, 100.0, 17)
    near, below, across, above = TRANSITION_SPLIT
    d_lo, d_hi = edges
    # the points below the transition, its lower edge and its upper edge, in assets
    d = np.r_[np.linspace(TRANSITION_BELOW_START * d_lo, d_lo, below + 1), d_hi]
    floor = 1e-6 * first
    a = np.maximum(d - solution_next.cFunc(solution_next.mNrmMin + d), 10.0 * floor)
    a_lo, a_hi = a[-2], max(a[-1], 4.0 * a[-2])
    points = np.r_[
        np.geomspace(floor, 1e-3 * first, near),
        a[:-2],
        np.geomspace(a_lo, a_hi, across),
        np.geomspace(a_hi, TRANSITION_ABOVE_TOP * a_hi, above + 1)[1:],
    ]
    return mNrmMin + points


# Above a kink at the natural limit, the MPC falls from its value at the kink
# like a power of the distance, most of it within the asset grid's first point
# above the constraint (a thousandth, by default) where that power is small. The
# grid also takes these multiples of that first point: one a decade, down to a
# millionth of it, the nearest point the grid takes to a natural limit that
# does not bind (_natural_limit_assets)
KINK_NEAR_ASSETS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)

# The most times a segment of the asset grid is bisected so that the cubic
# consumption function is concave
REFINE_DEPTH = 6
# The multiples of the nodes a piece to which a segment of the asset grid rises
# where its data are not concave, the expectations not accurate enough. The
# steps raise the nodes only: cutting the integrals at next period's knots as
# well (split_at_knots) changes the expectations little, and multiplies the
# points each asset's integrand is evaluated at by about the number of knots its
# landing points cross
REFINE_NODE_STEPS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
# The step to which a failing segment's accuracy rises before it is bisected:
# twice the nodes a piece
REFINE_FIRST_STEP = 2


def _refine_for_concavity(
    base,
    E_base,
    state,
    expect,
    endogenous,
    nodes,
    m_limit,
    mpc_limit,
    a_floor,
    power_segment=False,
):
    """
    End-of-period assets, and the expectations there, for which the cubic Hermite
    consumption function through the endogenous gridpoints is concave. A cubic
    Hermite segment from (x0, y0, s0) to (x1, y1, s1), with secant S, is concave
    if and only if (s0 + 2 s1)/3 <= S <= (2 s0 + s1)/3; its data are concave if
    s0 >= S >= s1. Each segment of the cubic above the first, from the limit to
    the first gridpoint, is checked:
    - where the data are concave but the cubic is not, the segment of the base
      grid holding it is bisected (in the distance from a_floor, geometrically
      where that is positive), up to REFINE_DEPTH times;
    - where the data are not concave, the expectations are not accurate enough:
      that segment's gridpoints are taken again with more nodes a piece, the next
      of REFINE_NODE_STEPS times nodes (and so where bisection is exhausted).
    With power_segment, the segment from the first gridpoint to the second is
    not a cubic: its MPC leaves the first gridpoint's like a power of the
    distance (CubicHermiteInterp.set_power_segment), as above a kink at the
    natural borrowing constraint, and it is concave wherever its data strictly
    are, so only its data are checked.
    It stops when every segment is concave, or warns if none can be refined
    further. state, from next period's solution, holds each base segment's
    depth and node step there; the refinement starts from them and only
    deepens, so that over an infinite horizon the grid and its accuracy settle.
    expect(a, n) gives the expectations (E1, E2, E0) at assets a with n nodes a
    piece of each shock; endogenous(a, E1, E2) the gridpoints' cash-on-hand,
    consumption and MPC; E_base the expectations on base, with nodes a piece.
    Returns the assets, their expectations and the state.
    """
    n = base.size - 1
    depth, level = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    if isinstance(state, dict) and np.size(state.get("depth")) == n:
        depth = np.minimum(np.asarray(state["depth"], dtype=int), REFINE_DEPTH)
        level = np.minimum(
            np.asarray(state["level"], dtype=int), len(REFINE_NODE_STEPS) - 1
        )
    cache = {
        (x, 0): tuple(None if e is None else e[i] for e in E_base)
        for i, x in enumerate(base)
    }

    def node_count(step):
        return tuple(int(np.ceil(q * REFINE_NODE_STEPS[step])) for q in nodes)

    def points():
        a, steps = [base[:1]], [np.array([level[0]])]
        for j in range(n):
            lo, hi = base[j] - a_floor, base[j + 1] - a_floor
            t = np.arange(1, 2 ** depth[j]) / 2 ** depth[j]
            inner = a_floor + (lo * (hi / lo) ** t if lo > 0.0 else lo + (hi - lo) * t)
            a += [inner, base[j + 1 : j + 2]]
            right = level[j + 1] if j + 1 < n else level[j]
            steps += [np.full(inner.size, level[j]), np.array([max(level[j], right)])]
        return np.concatenate(a), np.concatenate(steps)

    for _ in range(n * (REFINE_DEPTH + len(REFINE_NODE_STEPS))):
        a, steps = points()
        for step in np.unique(steps):
            todo = [x for x in a[steps == step] if (x, step) not in cache]
            if todo:
                parts = expect(np.array(todo), *node_count(step))
                for i, x in enumerate(todo):
                    cache[(x, step)] = tuple(None if e is None else e[i] for e in parts)
        E = [
            None
            if cache[(a[0], steps[0])][k] is None
            else np.array([cache[(x, st)][k] for x, st in zip(a, steps)])
            for k in range(3)
        ]
        m, c, mpc = endogenous(a, E[0], E[1])
        x = np.r_[m_limit, m]
        y = np.r_[0.0, c]
        d = np.r_[mpc_limit, mpc]
        S = np.diff(y) / np.diff(x)
        tol = 1e-12
        bad = (S < (d[:-1] + 2.0 * d[1:]) / 3.0 - tol) | (
            S > (2.0 * d[:-1] + d[1:]) / 3.0 + tol
        )
        bad[0] = False  # the first segment, from the limit: see the solver
        data_concave = (d[:-1] >= S - tol) & (S >= d[1:] - tol)
        if power_segment:
            # the segment above the kink follows a power law, concave by
            # construction where its data strictly are
            bad[1] = not (d[1] > S[1] > d[2])
            data_concave[1] = not bad[1]
        if not np.any(bad):
            break
        changed = False
        for k in np.flatnonzero(bad):
            # the segment of the cubic between gridpoints k - 1 and k of a lies in
            # base segment j
            j = min(
                max(int(np.searchsorted(base, a[k - 1], side="right")) - 1, 0), n - 1
            )
            # accuracy first, to REFINE_FIRST_STEP; then gridpoints where the
            # data are concave; then more accuracy
            if level[j] < REFINE_FIRST_STEP:
                level[j] += 1
                changed = True
            elif data_concave[k] and depth[j] < REFINE_DEPTH:
                depth[j] += 1
                changed = True
            elif level[j] < len(REFINE_NODE_STEPS) - 1:
                level[j] += 1
                changed = True
        if not changed:
            warnings.warn(
                "The cubic consumption function is not concave on "
                f"{int(np.sum(bad))} segment(s) after bisecting the asset grid "
                f"{REFINE_DEPTH} times and taking the expectations with up to "
                f"{node_count(len(REFINE_NODE_STEPS) - 1)} nodes a piece."
            )
            break
    return a, E, {"depth": depth, "level": level}


def _calc_BegOfPrd_IndShock_continuous(
    aNrm,
    solution_next,
    IncShkDstn,
    Rfree,
    PermGroFac,
    CRRA,
    CubicBool,
    vFuncBool,
    PermShkQuadCount,
    TranShkQuadCount,
    split_at_knots=False,
):
    """
    IndShock's expectations against the laws IncShkDstn records: the employed
    branch's calc_BegOfPrd_continuous() with weight 1 - p, and the unemployed
    branch's, a point at the unemployment income, with weight p.
    """
    PermShkLaw, TranShkLaw, UnempPrb, IncUnemp = get_income_shock_laws(IncShkDstn)
    branches = [(1.0 - UnempPrb, TranShkLaw)]
    if UnempPrb > 0.0:
        branches.append((UnempPrb, DiscreteDistribution([1.0], [IncUnemp])))
    total = [0.0, 0.0, 0.0]
    for prob, law in branches:
        parts = calc_BegOfPrd_continuous(
            aNrm,
            solution_next,
            PermShkLaw,
            law,
            Rfree,
            PermGroFac,
            CRRA,
            CubicBool=CubicBool,
            vFuncBool=vFuncBool,
            PermShkQuadCount=PermShkQuadCount,
            TranShkQuadCount=TranShkQuadCount,
            split_at_knots=split_at_knots,
        )
        total = [
            None if part is None else t + prob * part for t, part in zip(total, parts)
        ]
    return tuple(total)


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
    return _solve_one_period_ConsIndShock(
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
    )


def solve_one_period_ConsIndShock_continuous(
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
    PermShkQuadCount=4,
    TranShkQuadCount=4,
):
    """
    Solves one period of the consumption-saving model of
    solve_one_period_ConsIndShock with the expectation over the income shocks
    taken against their continuous laws instead of summed over IncShkDstn's
    atoms: IndShockConsumerType's solver with IncShkExpectation "continuous".

    The laws are those IncShkDstn records (get_income_shock_laws): the
    permanent shock's, the transitory shock's while employed, and the income in
    unemployment, a point. Each branch's expectation is
    calc_BegOfPrd_continuous(), split where next period's cash-on-hand reaches
    its kink, which each solution records in mNrmKinks. Next period's kink then
    leaves no images in this period's consumption function.

    Three things differ from the solver over atoms. The natural borrowing
    constraint comes from the laws' infima, which is where it lies under the
    laws: from below it, a landing point m' = Rfree a / (PermGroFac psi) + theta
    could fall below next period's minimum inside the shocks' support. The
    asset grid starts at the constraint that binds, artificial or natural. Where
    the artificial one binds, the consumption function has a kink where it stops
    binding, the endogenous gridpoint of the constraint itself, at which the
    first-order condition gives consumption directly (Deaton's x*); where the
    natural one binds, consumption is zero at the limit and
    there is no kink to split at in the period before, and
    the value at the limit is minus infinity, as over atoms, though the
    end-of-period value there is finite where the expectation of next period's
    value converges at the limit, as under the jumpless laws; the asset grid then
    also takes points placed by the transition of next period's consumption
    function, where its MPC falls from near 1, and close to the limit
    (_natural_limit_assets). And the worst income
    has no mass under continuous laws, so the MPC at the natural constraint is
    one.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncShkDstn : distribution.Distribution
        A discrete approximation to the income process between the period being
        solved and the one immediately following (in solution_next), recording
        the laws its atoms come from, as IndShock's income constructor makes it.
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
        period with. If it is None, there is no artificial borrowing constraint.
    aXtraGrid: np.array
        Array of "extra" end-of-period asset values-- assets above the
        artificial borrowing constraint.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear interpolation.
        The cubic is the plain cubic Hermite through the endogenous gridpoints;
        where a segment of it would not be concave, that segment's expectations
        are taken more accurately and the asset grid is refined there
        (_refine_for_concavity), each solution recording where in
        ConcavityRefinement, from which the period before starts. Above a kink
        at the natural borrowing constraint, where the MPC falls from its value
        at the kink like a power of the distance, the segment from the kink to
        the next gridpoint takes that power's shape instead
        (CubicHermiteInterp.set_power_segment).
    PermShkQuadCount : int, optional
        Gauss-Legendre nodes a piece of the permanent shock's integral. The
        default is 4.
    TranShkQuadCount : int, optional
        Gauss-Legendre nodes a piece of the transitory shock's integral. The
        default is 4.

    Returns
    -------
    solution_now : ConsumerSolution
        Solution to this period's consumption-saving problem with income risk,
        with the cash-on-hand at which its constraint stops binding in
        mNrmKinks, empty where the natural borrowing constraint binds.
    """
    return _solve_one_period_ConsIndShock(
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
        continuous=(PermShkQuadCount, TranShkQuadCount),
    )


def _solve_one_period_ConsIndShock(
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
    continuous=None,
):
    """
    The body of solve_one_period_ConsIndShock and of
    solve_one_period_ConsIndShock_continuous: with continuous None, the
    expectation over the income shocks is over IncShkDstn's atoms; with
    continuous the pair (PermShkQuadCount, TranShkQuadCount), it is against the
    laws IncShkDstn records.
    """
    # Define the current period utility function and effective discount factor
    uFunc = UtilityFuncCRRA(CRRA)
    DiscFacEff = DiscFac * LivPrb  # "effective" discount factor

    # Calculate the probability that we get the worst possible income draw. Under
    # the laws the worst income has no mass unless a point mass sits there: an
    # unemployment event that pays nothing, whose income is zero whatever the
    # permanent shock, or the lowest values of laws that hold probability there,
    # as PointMassAtInfimum does (_worst_income_mass). Over the atoms, it is the
    # probability of the lowest, which sets the MPC at the atoms' own limit,
    # where the unconstrained consumption function is anchored
    if continuous:
        WorstIncPrb = _worst_income_mass(get_income_shock_laws(IncShkDstn))
    else:
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
    # The natural limit belongs to the model: it comes from the true laws'
    # infima, each a realization of infinitesimal but positive probability
    # (calc_boro_const_nat), so every discretization of the same laws has the
    # same one. The atoms' own limit, below it, still anchors the unconstrained
    # consumption function, where the atoms' marginal value is infinite, so the
    # infima's limit is imposed as a constraint, as BoroCnstArt is: where it is
    # above the atoms' limit it binds on the atoms, the asset grid starts at it
    # and the solution records its kink, which is the limit of the solutions as
    # the infima's probability goes to zero. Making it BoroCnstNat itself, as PR
    # #1524 did, anchored the grid where the atoms' marginal value is still
    # finite, which left the first endogenous gridpoint far above the bottom and
    # broke the value function; PR #1589 reverted that. Where the laws are not
    # bounded below away from zero, as HARK's default lognormals are not, the
    # infima's limit would allow no borrowing, so over the atoms the atoms' limit
    # stands (calc_boro_const_nat_from_infima). Under the laws themselves, as the
    # continuous expectation takes them, the infima's limit is the natural one.
    if continuous:
        BoroCnstInf = calc_boro_const_nat(
            solution_next.mNrmMin, IncShkDstn, Rfree, PermGroFac, use_infimum=True
        )
    else:
        BoroCnstInf = calc_boro_const_nat_from_infima(
            solution_next.mNrmMin, IncShkDstn, Rfree, PermGroFac
        )
    if BoroCnstInf is not None:
        mNrmMinNow = max(mNrmMinNow, BoroCnstInf)
    # Under the laws, where the natural limit binds, the worst transitory shock
    # must be a point mass
    if continuous:
        check_worst_transitory_point_mass(IncShkDstn, BoroCnstArt, BoroCnstInf)
    # Whether the continuous expectation's consumption function has a kink: where
    # the lowest assets allowed leave next period's marginal value finite, so that
    # they are a constraint that binds, as an artificial one above the natural
    # limit always is
    kinked = continuous and not _reaches_natural_limit(
        mNrmMinNow,
        solution_next.mNrmMin,
        get_income_shock_laws(IncShkDstn),
        Rfree,
        PermGroFac,
        CRRA,
    )
    # Where the natural limit itself is that constraint, the lowest assets still
    # land on next period's minimum cash-on-hand, with too little probability to
    # make marginal value infinite but enough to make its derivatives so. Next
    # period's marginal value goes like m'^(-CRRA) there, and each derivative
    # adds one to that power, so the rule is marginal value's
    # (_reaches_natural_limit) with CRRA + 1 and CRRA + 2 in its place:
    # - with CRRA + 1, the first derivative of end-of-period marginal value is
    #   infinite at the constraint, and the MPC leaves one above the kink with
    #   no step;
    # - with CRRA + 2, the second is: the MPC falls from its value at the kink
    #   like a power of the distance that the laws and CRRA set, which can be
    #   any positive number. A cubic's MPC leaves a knot linearly, so the cubic
    #   from the kink to the next gridpoint follows it only for powers near one,
    #   however short that segment; the segment takes the power's shape instead
    #   (CubicHermiteInterp.set_power_segment), and the asset grid takes points
    #   close above the constraint (KINK_NEAR_ASSETS).
    # An artificial constraint above the natural limit keeps every landing point
    # off next period's minimum, and neither holds.
    no_mpc_step, power_above_kink = (
        bool(
            kinked
            and _reaches_natural_limit(
                mNrmMinNow,
                solution_next.mNrmMin,
                get_income_shock_laws(IncShkDstn),
                Rfree,
                PermGroFac,
                CRRA + order,
            )
        )
        for order in (1.0, 2.0)
    )

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

    if continuous:
        # Construct the assets grid from the binding constraint, and take the
        # expectations against the income shocks' laws
        aNrmNow = np.asarray(aXtraGrid) + mNrmMinNow
        if kinked:
            # The grid starts at the constraint itself, where the marginal value is
            # finite: there the first-order condition gives consumption directly,
            # and its endogenous gridpoint is the kink where the constraint stops
            # binding, as in Deaton (1991) and BufferStockTheory
            aNrmNow = np.insert(aNrmNow, 0, mNrmMinNow)
            if power_above_kink:
                # and close above it, where the MPC falls from its value at the
                # kink like a power of the distance
                aNrmNow = np.union1d(
                    aNrmNow,
                    mNrmMinNow + np.asarray(aXtraGrid)[0] * np.array(KINK_NEAR_ASSETS),
                )
        else:
            # Where the natural constraint binds, consumption stays close to
            # cash-on-hand down to the limit, then turns toward buffer-stock
            # saving, and the end-of-period value falls steeply toward the limit
            # (to minus infinity, or to a finite value, by the laws' ends): the
            # grid also takes assets placed by the transition of
            # next period's consumption function, where its MPC falls from near
            # 1, covering the transition, the first segment below it and the
            # end-of-period value near the limit
            first = np.asarray(aXtraGrid)[0]
            aNrmNow = np.union1d(
                _natural_limit_assets(solution_next, mNrmMinNow, first, CubicBool),
                aNrmNow,
            )
        aNrmExp = aNrmNow

        def expect(a, PermNodes, TranNodes):
            return _calc_BegOfPrd_IndShock_continuous(
                a,
                solution_next,
                IncShkDstn,
                Rfree,
                PermGroFac,
                CRRA,
                CubicBool,
                vFuncBool,
                PermNodes,
                TranNodes,
            )

        E1, E2, E0 = expect(aNrmExp, *continuous)
        refinement = None
        if CubicBool:
            # Where the cubic through the endogenous gridpoints would not be
            # concave: more accurate expectations, and more asset gridpoints
            vPfacCont = DiscFacEff * Rfree * PermGroFac ** (-CRRA)
            vPPfacCont = DiscFacEff * Rfree * Rfree * PermGroFac ** (-CRRA - 1.0)

            def endogenous(a, E1_, E2_):
                c = uFunc.derinv(vPfacCont * E1_, order=(1, 0))
                dcda = vPPfacCont * E2_ / uFunc.der(c, order=2)
                mpc = dcda / (dcda + 1.0)
                if no_mpc_step:
                    # at the constraint itself the derivative of marginal value
                    # is infinite, whatever number its integral's nodes give
                    mpc[a == mNrmMinNow] = 1.0
                return a + c, c, mpc

            aNrmNow, (E1, E2, E0), refinement = _refine_for_concavity(
                aNrmNow,
                (E1, E2, E0),
                getattr(solution_next, "ConcavityRefinement", None),
                expect,
                endogenous,
                continuous,
                BoroCnstInf if not kinked else BoroCnstNat,
                MPCmaxUnc,
                mNrmMinNow,
                power_segment=power_above_kink,
            )
            aNrmExp = aNrmNow
    else:
        # The end-of-period assets grid, starting at a binding artificial
        # constraint (calc_aNrm_grid says why)
        aNrmNow = calc_aNrm_grid(
            np.asarray(aXtraGrid, dtype=float), BoroCnstNat, mNrmMinNow
        )

    # Calculate end-of-period marginal value of assets at each gridpoint
    vPfacEff = DiscFacEff * Rfree * PermGroFac ** (-CRRA)
    if continuous:
        EndOfPrdvP = vPfacEff * E1[-aNrmNow.size :]
    else:
        EndOfPrdvP = vPfacEff * expected(
            calc_vp_next,
            IncShkDstn,
            args=(aNrmNow, Rfree, CRRA, PermGroFac, vPfuncNext),
        )

    # Invert the first order condition to find optimal cNrm from each aNrm gridpoint
    cNrmNow = uFunc.derinv(EndOfPrdvP, order=(1, 0))
    mNrmNow = cNrmNow + aNrmNow  # Endogenous mNrm gridpoints

    # The unconstrained consumption function starts at the natural borrowing
    # constraint, where consumption is zero, whichever constraint binds; under
    # the continuous expectation, where no artificial constraint binds, that is
    # the infima's limit. Where the artificial constraint binds, this is the one
    # point below the grid: it keeps the interpolant defined down to mNrmMinNow,
    # where the lower envelope takes the constraint line instead. The kink is a
    # node of this function, which is smooth there, with a single slope (the
    # unconstrained MPC), so no interpolation node has to carry the kink's two
    # slopes.
    c_for_interpolation = np.insert(cNrmNow, 0, 0.0)
    m_limit = BoroCnstInf if continuous and not kinked else BoroCnstNat
    m_for_interpolation = np.insert(mNrmNow, 0, m_limit)

    # Construct the consumption function as a cubic or linear spline interpolation
    if CubicBool:
        # Calculate end-of-period marginal marginal value of assets at each gridpoint
        vPPfacEff = DiscFacEff * Rfree * Rfree * PermGroFac ** (-CRRA - 1.0)
        if continuous:
            EndOfPrdvPP = vPPfacEff * E2[-aNrmNow.size :]
        else:
            EndOfPrdvPP = vPPfacEff * expected(
                calc_vpp_next,
                IncShkDstn,
                args=(aNrmNow, Rfree, CRRA, PermGroFac, vPPfuncNext),
            )
        dcda = EndOfPrdvPP / uFunc.der(np.array(cNrmNow), order=2)
        MPC = dcda / (dcda + 1.0)
        if no_mpc_step:
            # at the constraint itself the derivative of marginal value is
            # infinite: the MPC leaves one above the kink with no step
            MPC[0] = 1.0
        MPC_for_interpolation = np.insert(MPC, 0, MPCmaxUnc)

        # Construct the unconstrained consumption function as a cubic interpolation;
        # under the continuous expectation, the grid is refined where it would not
        # be concave
        cFuncNowUnc = CubicInterp(
            m_for_interpolation,
            c_for_interpolation,
            MPC_for_interpolation,
            cFuncLimitIntercept,
            cFuncLimitSlope,
        )
        if continuous and not kinked:
            # From the limit to the first gridpoint, end-of-period assets stay
            # between the limit and the lowest asset gridpoint, so consumption
            # is the line c = m - mNrmMin to within that distance
            cFuncNowUnc.set_linear_segment(0)
        if power_above_kink:
            # Above a kink at the natural limit, the MPC falls from its value at
            # the kink like a power of the distance: the segment from the kink
            # to the next gridpoint takes that shape, where its data are
            # strictly concave (_refine_for_concavity warns where they are not)
            secant = (c_for_interpolation[2] - c_for_interpolation[1]) / (
                m_for_interpolation[2] - m_for_interpolation[1]
            )
            if MPC_for_interpolation[1] > secant > MPC_for_interpolation[2]:
                cFuncNowUnc.set_power_segment(1)
    else:
        # Construct the unconstrained consumption function as a linear interpolation
        cFuncNowUnc = LinearInterp(
            m_for_interpolation,
            c_for_interpolation,
            cFuncLimitIntercept,
            cFuncLimitSlope,
        )

    # Where the artificial constraint binds, the kink is the endogenous gridpoint
    # of end-of-period assets at the constraint; record it, with the MPC above it
    if continuous and not kinked:
        # Under the laws the infima's limit, though above the atoms', is the
        # natural one, where consumption falls to zero with no kink
        mNrmKink, cNrmKink, MPCaboveKink = np.nan, np.nan, np.nan
    else:
        mNrmKink, cNrmKink, MPCaboveKink = find_kink(
            m_for_interpolation,
            c_for_interpolation,
            MPC_for_interpolation if CubicBool else None,
            BoroCnstNat,
            mNrmMinNow,
        )

    # The true consumption function: the lower envelope of the unconstrained
    # function and the constraint line, which meet at the kink's node (see
    # calc_aNrm_grid)
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
        if continuous:
            # On the grid from an artificial constraint itself, where the value is
            # finite. At the natural limit the end-of-period value is minus
            # infinity, its inverse zero as over atoms, only where its expectation
            # diverges there: next period's value goes like m'^(1 - CRRA) near its
            # own limit, as its marginal value goes like m'^(-CRRA), so the rule is
            # marginal value's (_reaches_natural_limit) with CRRA - 1 in its place.
            # Elsewhere, as where the laws' densities fall to zero at their ends
            # fast enough, it is finite, and is taken at the limit itself: pinning
            # zero there would put minus infinity at the assets consumption leaves
            # just above the limit, where its MPC is near 1, and the error would
            # grow from one period to the next
            EndOfPrdv = DiscFacEff * E0
            EndOfPrdvNvrs = uFunc.inv(EndOfPrdv)
            EndOfPrdvNvrsP = vPfacEff * E1 * uFunc.derinv(EndOfPrdv, order=(0, 1))
            aNrm_temp = aNrmExp
            if not kinked:
                if _reaches_natural_limit(
                    mNrmMinNow,
                    solution_next.mNrmMin,
                    get_income_shock_laws(IncShkDstn),
                    Rfree,
                    PermGroFac,
                    CRRA - 1.0,
                ):
                    EndOfPrdvNvrsAtLimit = 0.0
                else:
                    E0AtLimit = _calc_BegOfPrd_IndShock_continuous(
                        np.array([mNrmMinNow]),
                        solution_next,
                        IncShkDstn,
                        Rfree,
                        PermGroFac,
                        CRRA,
                        False,
                        True,
                        *continuous,
                    )[2]
                    EndOfPrdvNvrsAtLimit = float(uFunc.inv(DiscFacEff * E0AtLimit)[0])
                EndOfPrdvNvrs = np.insert(EndOfPrdvNvrs, 0, EndOfPrdvNvrsAtLimit)
                EndOfPrdvNvrsP = np.insert(EndOfPrdvNvrsP, 0, EndOfPrdvNvrsP[0])
                aNrm_temp = np.insert(aNrmExp, 0, mNrmMinNow)
        else:
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
            aNrm_temp = np.insert(aNrmNow, 0, BoroCnstNat)

        # Construct the end-of-period value function
        EndOfPrd_vNvrsFunc = CubicInterp(aNrm_temp, EndOfPrdvNvrs, EndOfPrdvNvrsP)
        EndOfPrd_vFunc = ValueFuncCRRA(EndOfPrd_vNvrsFunc, CRRA)

        # Compute expected value and marginal value on a grid of market resources
        mNrm_temp = mNrmMinNow + aXtraGrid
        cNrm_temp = cFuncNow(mNrm_temp)
        aNrm_temp = mNrm_temp - cNrm_temp
        v_temp = uFunc(cNrm_temp) + EndOfPrd_vFunc(aNrm_temp)
        if continuous:
            # Also at the consumption function's own gridpoints, where the
            # end-of-period value is known without interpolation, v = u(c) +
            # DiscFacEff E0
            mNrm_temp = np.r_[mNrm_temp, mNrmNow]
            cNrm_temp = np.r_[cNrm_temp, cNrmNow]
            v_temp = np.r_[v_temp, uFunc(cNrmNow) + DiscFacEff * E0[-aNrmNow.size :]]
            order = np.argsort(mNrm_temp, kind="stable")
            order = order[np.r_[True, np.diff(mNrm_temp[order]) > 0.0]]
            mNrm_temp, cNrm_temp, v_temp = (
                mNrm_temp[order],
                cNrm_temp[order],
                v_temp[order],
            )
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
        mNrmKink=mNrmKink,
        cNrmKink=cNrmKink,
        MPCaboveKink=MPCaboveKink,
    )
    if continuous:
        # Where the constraint stops binding, the gridpoint at the constraint: the
        # period before splits there. Where the natural constraint binds,
        # consumption is zero at the limit, with no kink.
        solution_now.mNrmKinks = mNrmNow[:1] if kinked else np.array([])
        if refinement is not None:
            solution_now.ConcavityRefinement = refinement
        if CubicBool and continuous and not kinked:
            # where the linear first segment ends, its slope steps to the MPC
            solution_now.mNrmLinearEnd = np.array([m_for_interpolation[1]])
    return solution_now


solve_one_period_ConsIndShock.imposes_infimum_limit = True
solve_one_period_ConsIndShock_continuous.imposes_infimum_limit = True


class ConsIndShockStationaryProblem(KnotProblem):
    """The stationary ConsIndShock policy as a fixed point of ``solve_one_period_ConsIndShock``.

    See :class:`HARK.stationary.KnotProblem`.  This class binds the model's
    primitives and builds the continuation solution the way the solver does.
    """

    def __init__(
        self,
        IncShkDstn,
        LivPrb,
        DiscFac,
        CRRA,
        Rfree,
        PermGroFac,
        BoroCnstArt,
        aXtraGrid,
        CubicBool,
        seed_solution,
    ):
        self.args = (IncShkDstn, LivPrb, DiscFac, CRRA, Rfree, PermGroFac, BoroCnstArt)
        self.CRRA = CRRA
        super().__init__(aXtraGrid, CubicBool, seed_solution)

    def solve_one_period(self, solution_next, aXtraGrid, vFuncBool=False, cubic=None):
        cubic = self.cubic if cubic is None else cubic
        return solve_one_period_ConsIndShock(
            solution_next, *self.args, aXtraGrid, vFuncBool, cubic
        )

    def cFuncs(self, solution):
        return [solution.cFunc]

    def vFuncs(self, solution):
        return [solution.vFunc]

    def scalar_value(self, name, values):
        return float(values[0])

    def make_cFunc(self, m, c, mpc, state):
        MPCmin = self.scalars["MPCmin"][0]
        hNrm = self.scalars["hNrm"][0]
        mNrmMin = self.scalars["mNrmMin"][0]
        if mpc is None:
            cFuncUnc = LinearInterp(m, c, MPCmin * hNrm, MPCmin)
        else:
            cFuncUnc = CubicInterp(m, c, mpc, MPCmin * hNrm, MPCmin)
        cFuncCnst = LinearInterp(
            np.array([mNrmMin, mNrmMin + 1.0]), np.array([0.0, 1.0])
        )
        return LowerEnvelope(cFuncUnc, cFuncCnst, nan_bool=False)

    def make_vFunc(self, m, vNvrs, c_on_m, state):
        # The slope of the inverse value at the knots follows from the envelope
        # condition v'(m) = u'(c(m)) and the chain rule, as in the solver.
        uFunc = UtilityFuncCRRA(self.CRRA)
        vNvrsP = uFunc.der(c_on_m) * uFunc.derinv(uFunc(vNvrs), order=(0, 1))
        power = -self.CRRA / (1.0 - self.CRRA)
        MPCminNvrs = self.scalars["MPCmin"][0] ** power
        MPCmaxNvrs = self.scalars["MPCmax"][0] ** power
        vNvrsFunc = CubicInterp(
            np.insert(m, 0, self.scalars["mNrmMin"][0]),
            np.insert(vNvrs, 0, 0.0),
            np.insert(vNvrsP, 0, MPCmaxNvrs),
            MPCminNvrs * self.scalars["hNrm"][0],
            MPCminNvrs,
        )
        return ValueFuncCRRA(vNvrsFunc, self.CRRA)

    def make_continuation(self, cFuncs, vFuncs=None):
        cFunc = cFuncs[0]
        return ConsumerSolution(
            cFunc=cFunc,
            vFunc=None if vFuncs is None else vFuncs[0],
            vPfunc=MargValueFuncCRRA(cFunc, self.CRRA),
            vPPfunc=MargMargValueFuncCRRA(cFunc, self.CRRA) if self.cubic else None,
        )


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
    # The natural limit from the shock laws' infima, at the borrowing rate, where
    # they are bounded below away from zero (calc_boro_const_nat_from_infima)
    BoroCnstInf = calc_boro_const_nat_from_infima(
        solution_next.mNrmMin, IncShkDstn, Rboro, PermGroFac
    )
    # Set the minimum allowable (normalized) market resources based on the natural
    # and artificial borrowing constraints. Where the laws are bounded below away
    # from zero, the infima's limit, where it is above the atoms', is imposed as
    # an artificial constraint is, as in IndShock's solver: the atoms' limit still
    # anchors the unconstrained consumption function, where marginal value is
    # infinite. Where an infimum is zero the atoms' limit stands.
    mNrmMinNow = calc_m_nrm_min(BoroCnstArt, BoroCnstNat)
    if BoroCnstInf is not None:
        mNrmMinNow = max(mNrmMinNow, BoroCnstInf)

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

    # Construct the end-of-period assets grid from the constraint in force,
    # mNrmMinNow, the larger of the natural and artificial borrowing
    # constraints, with two points at zero for the kink in the interest factor
    # (where borrowing and saving rates differ). Unlike the standard solver's
    # grid, it has no point at an artificial constraint other than zero.
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
        # Between the two zero-asset points end-of-period assets stay at zero, so
        # consumption equals cash-on-hand: that segment is the 45-degree line,
        # with slope one inside and the borrowing- and saving-side MPCs just
        # outside. Replace its cubic by the chord. (Writing linear coefficients
        # into ``coeffs`` here, as this code once did, changed nothing once
        # CubicInterp became the scipy-backed CubicHermiteInterp: the table is
        # not what the spline evaluates, and the cubic dipped below the line.)
        cFuncNowUnc.set_linear_segment(i_kink + 1)
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


solve_one_period_ConsKinkedR.imposes_infimum_limit = True


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
    "init_shuffle": False,  # Exact-marginal initial-state draws when True (see sim_birth)
    "death_shuffle": False,  # Deterministic death counts when True (see sim_death)
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
    on_no_solution: str
        What solve does when an infinite horizon problem with one repeated period has
        no non-degenerate solution: "warn" (the default) or "raise". quiet=True
        silences the warning.

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
    kLogInitMean: float
        Mean of Log initial Normalized Assets.
    kLogInitStd: float
        Std of Log initial Normalized Assets.
    pLogInitMean: float
        Mean of Log initial permanent income.
    pLogInitStd: float
        Std of Log initial permanent income.
    PermGroFacAgg: float
        Aggregate permanent income growth factor (The portion of PermGroFac attributable to aggregate productivity growth).
        The simulation places each newborn cohort at the aggregate level, which grows by this factor every period:
        one makes a stationary economy, any other value a growing one whose levels depend on the date. The Jacobian
        tools require the same choice (see HARK.SSJutils, Two economies).
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
        "track_vars": ["aNrm", "cNrm", "mNrm", "pLvl"],
    }

    time_vary_ = ["LivPrb", "PermGroFac", "Rfree"]
    time_inv_ = ["CRRA", "DiscFac", "MaxKinks", "BoroCnstArt"]
    state_vars = ["kNrm", "pLvl", "bNrm", "mNrm", "aNrm", "aLvl"]
    shock_vars_ = []
    distributions = ["kNrmInitDstn", "pLvlInitDstn"]
    on_no_solution = "warn"

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
        self._respond_to_no_solution()

    def _respond_to_no_solution(self):
        """
        If check_conditions found that the infinite horizon problem has no solution,
        warn, or raise an error if on_no_solution is "raise". quiet=True silences the
        warning, but not the error.
        """
        if not self.degenerate:
            return
        named = [("RIC", "RPFac"), ("FHWC", "FHWFac"), ("GICRaw", "GPFacRaw")]
        # Where the constraint binds, the RIC fails and the GICRaw holds, the PFFVAC
        # decides whether a solution exists
        if (
            "PFFVAC" in self.conditions
            and not self.conditions["PFFVAC"]
            and self.conditions.get("GICRaw")
            and not self.conditions.get("RIC")
            and self._constraint_can_bind()
        ):
            named.append(("PFFVAC", "PFVAFac"))
        violated = ", ".join(
            f"{name} ({factor}={self.bilt[factor]:.5f})"
            for name, factor in named
            if not self.conditions[name]
        )
        message = (
            "This infinite horizon problem has no non-degenerate solution. Violated: "
            + violated
            + ". "
            + self.bilt["solution_verdict"]
        )
        if self.on_no_solution == "raise":
            raise ValueError(message)
        if not self.quiet:
            warnings.warn(message)

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
        if self.on_no_solution not in ("warn", "raise"):
            raise ValueError(
                'on_no_solution must be "warn" or "raise", not '
                + repr(self.on_no_solution)
            )

    def initialize_sim(self):
        self.PermShkAggNow = self.PermGroFacAgg  # This never changes during simulation
        self.state_now["PlvlAgg"] = 1.0
        # Drop base draws cached by an earlier run. They are keyed by
        # t_cycle, so a consumer reading them after the flag is turned off,
        # or after AgentCount changes, would silently pair this period's
        # agents with a previous run's uniforms.
        self._base_shock_draws = {}
        super().initialize_sim()

    def sim_birth(self, which_agents):
        """
        Makes new consumers for the given indices.  Initialized variables include aNrm and pLvl, as
        well as time variables t_age and t_cycle.  Normalized assets and permanent income levels
        are drawn from lognormal distributions given by kLogInitMean and kLogInitStd (etc).

        When ``init_shuffle`` is True (default False), draws from
        ``kNrmInitDstn`` and ``pLvlInitDstn`` use exact-marginal matching
        (``DiscreteDistribution.draw(N, shuffle=True)``) instead of iid
        sampling, eliminating cross-sectional sampling noise in the
        initial wealth/permanent-income distribution.

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
        # Passing shuffle= only when enabled keeps duck-typed continuous
        # init distributions (whose draw() lacks the kwarg) working at
        # the default.
        _kw = {"shuffle": True} if getattr(self, "init_shuffle", False) else {}
        self.state_now["aNrm"][which_agents] = self.kNrmInitDstn.draw(N, **_kw)
        self.state_now["pLvl"][which_agents] = self.pLvlInitDstn.draw(N, **_kw)
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

        if getattr(self, "death_shuffle", False):
            which_agents = self._sim_death_shuffled(DiePrb)
        else:
            DeathShks = Uniform(seed=self.RNG.integers(0, 2**31 - 1)).draw(
                N=self.AgentCount
            )
            which_agents = DeathShks < DiePrb
        if self.T_age is not None:  # Kill agents that have lived for too many periods
            too_old = self.t_age >= self.T_age
            which_agents = np.logical_or(which_agents, too_old)
        return which_agents

    def _sim_death_shuffled(self, DiePrb):
        """Deterministic death counts with random agent assignment.

        For each unique death probability in DiePrb, compute the number
        of deaths using floor-plus-remainder (so the expected count is
        unbiased) and randomly select which agents in that group die.
        This reduces binomial noise in death counts while preserving
        the expected number of deaths exactly.

        Parameters
        ----------
        DiePrb : float or np.array
            Death probability for each agent (scalar or per-agent array).

        Returns
        -------
        which_agents : np.array(bool)
            Boolean array of size AgentCount indicating which agents die.
        """
        which_agents = np.zeros(self.AgentCount, dtype=bool)
        DiePrb = np.broadcast_to(np.asarray(DiePrb), self.AgentCount)

        for p in np.unique(DiePrb):
            group = np.where(DiePrb == p)[0]
            N_group = len(group)
            # Floor-plus-remainder: unbiased expected death count.
            # This deliberately does not call
            # HARK.distributions.base.allocate_remainder_slots, despite
            # allocating a leftover slot. That function corrects a bias that
            # only appears when two or more slots are handed out, and the
            # die/survive split here can never produce more than one: its two
            # fractional parts sum to an integer, so exactly one of them is
            # nonzero. With a single slot, systematic sampling reduces to the
            # Bernoulli draw below, and routing through the general helper
            # would mean inventing a second atom for the survivors to make
            # the shapes fit.
            K_exact = N_group * p
            how_many_die = int(np.floor(K_exact))
            remainder = K_exact - how_many_die
            if remainder > 0 and self.RNG.random() < remainder:
                how_many_die += 1
            if how_many_die > 0:
                die_indices = self.RNG.choice(group, size=how_many_die, replace=False)
                which_agents[die_indices] = True
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
        of the bilt dictionary, and in the message log unless ``self.quiet`` is set.

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
        if not self.quiet:
            self._log_at_verbosity(message, verbose)
        self.bilt["conditions_report"] += message + "\n"

    def _log_at_verbosity(self, message, verbose):
        """Log a message at the level ``verbose`` selects, leaving HARK's logger as it was."""
        level = _log.level
        set_verbosity_level((4 - verbose) * 10)
        _log.info(message)
        set_verbosity_level(level)

    def _emit_conditions_report(self, verbose):
        """Emit the accumulated conditions report unless ``self.quiet`` is set."""
        if not self.quiet:
            self._log_at_verbosity(self.bilt["conditions_report"], verbose)

    def _setup_condition_check(self, verbose):
        """Initialize the conditions report and run the supported-cycle gate.

        Returns ``(verbose, should_continue)``: ``verbose`` is resolved against
        the instance default, and ``should_continue`` is False when the model
        is outside the infinite-horizon, single-period-cycle regime supported
        by ``check_conditions``. In that case the trivial-skip message has
        already been added to the conditions report, and logged unless
        ``self.quiet`` is set.
        """
        self.conditions = {}
        self.bilt["conditions_report"] = ""
        self.degenerate = False
        verbose = self.verbose if verbose is None else verbose
        if self.cycles != 0 or self.T_cycle > 1:
            self.log_condition_result(
                None,
                None,
                "No conditions report was produced because this functionality"
                " is only supported for infinite horizon models with a cycle"
                " length of 1.",
                verbose,
            )
            self._emit_conditions_report(verbose)
            return verbose, False
        self.calc_limiting_values()
        self.log_condition_result(None, None, self.describe_parameters(), verbose)
        return verbose, True

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
            True: f"PFVAFac={PFVAFac:.5f} : The Finite Value of Autarky Factor satisfies the Finite Value of Autarky Condition ℒβG^(1-ρ) < 1.",
            False: f"PFVAFac={PFVAFac:.5f} : The Finite Value of Autarky Factor violates the Finite Value of Autarky Condition ℒβG^(1-ρ) < 1.",
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
        if aux_dict["RPFac"] < 1.0:
            aux_dict["cNrmPDV"] = 1.0 / (1.0 - aux_dict["RPFac"])
        else:
            aux_dict["cNrmPDV"] = np.inf
        aux_dict["MPCmin"] = np.maximum(1.0 - aux_dict["RPFac"], 0.0)

        if self._constraint_can_bind():
            aux_dict["MPCmax"] = 1.0
        else:
            aux_dict["MPCmax"] = aux_dict["MPCmin"]
        if aux_dict["FHWFac"] < 1.0:
            aux_dict["hNrm"] = 1.0 / (1.0 - aux_dict["FHWFac"])
        else:
            aux_dict["hNrm"] = np.inf

        # Generate the "E[M_tp1 / M_t] = G" function, which is used to find balanced growth market resources
        PF_Rnrm = self.Rfree[0] / self.PermGroFac[0]
        aux_dict["BalGroFunc"] = lambda m: (1.0 - 1.0 / PF_Rnrm) * m + 1.0 / PF_Rnrm

        # The "Delta m = 0" function, which is used to find target market resources,
        # is the same function under perfect foresight
        aux_dict["Delta_mNrm_ZeroFunc"] = aux_dict["BalGroFunc"]

        self.bilt = aux_dict

    def _constraint_can_bind(self):
        """
        Whether the artificial borrowing constraint can bind in the infinite horizon
        limit. A finite constraint cannot bind if it lies at or below the natural
        borrowing constraint, which is minus human wealth excluding this period's
        income. If human wealth is infinite, every finite constraint can bind.

        Returns
        -------
        bool
        """
        BoroCnstArt = getattr(self, "BoroCnstArt", None)
        if BoroCnstArt is None or BoroCnstArt <= -np.inf:
            return False
        FHWFac = self.bilt["FHWFac"]
        if FHWFac >= 1.0:
            return True
        return BoroCnstArt > -FHWFac / (1.0 - FHWFac)

    def _describe_limiting_solution(self):
        """
        State the limit of the consumption function as the horizon extends to infinity,
        given the results of the conditions. Without a binding constraint a linear
        solution exists if and only if the RIC and the FHWC both hold; otherwise the
        Growth Patience Factor decides whether consumption goes to infinity or to zero.
        With a binding constraint, a solution exists if the RIC holds, or, where the RIC
        fails, if the PFFVAC holds (which implies the GICRaw): the consumer then runs
        market resources down to the constraint and consumes permanent income from then
        on, so the value is finite only if the value of autarky is.

        Returns
        -------
        message : str
            A statement of the limit and of the conditions that determine it.
        degenerate : bool
            Whether the limit is degenerate, so that no solution exists.
        """
        RIC = self.conditions["RIC"]
        FHWC = self.conditions["FHWC"]
        GIC = self.conditions["GICRaw"]
        GPFacRaw = self.bilt["GPFacRaw"]

        if self._constraint_can_bind():
            if RIC and GIC:
                message = "Because the RIC is satisfied and the consumer is constrained, the solution is not c(m)=0 for all m. Because the GICRaw is also satisfied, the solution is non-degenerate. It is piecewise linear with an infinite number of kinks"
                if FHWC:
                    message += ", approaching the unconstrained solution as m goes to infinity."
                else:
                    message += "."
                return message, False
            if RIC:
                return (
                    "Because the RIC is satisfied and the consumer is constrained, the solution is not c(m)=0 for all m. Because the GICRaw is violated, the solution is non-degenerate. It is piecewise linear with a single kink at some 0 < m < 1; it equals the unconstrained solution above that kink point and has c(m) = m below it.",
                    False,
                )
            if GIC and self.conditions["PFFVAC"]:
                return (
                    "Because the RIC is violated but the GIC is satisfied, the FHWC is necessarily also violated. In this case, the consumer's pathological patience is offset by his infinite human wealth, against which he cannot borrow arbitrarily: he runs market resources down to the constraint and consumes his permanent income from then on. Because the PFFVAC is also satisfied, the value of that path is finite, and a non-degenerate solution exists.",
                    False,
                )
            if GIC:
                return (
                    "Because the RIC is violated but the GIC is satisfied, the FHWC is necessarily also violated, and the consumer, who cannot borrow against his infinite human wealth, runs market resources down to the constraint and consumes his permanent income from then on. The consumption function converges, but because the PFFVAC is violated, the value of that path is not finite, so the solution is degenerate.",
                    True,
                )
            if GPFacRaw > 1.0:
                return (
                    "Because the RIC and the GICRaw are both violated, the solution is degenerate at c(m)=0 for all m.",
                    True,
                )
            return (
                "Because the RIC is violated and the Growth Patience Factor is exactly one, the solution is degenerate at c(m)=min(m-BoroCnstArt,1).",
                True,
            )

        if RIC and FHWC:
            return (
                "Because the FHWC is satisfied, the solution is not c(m)=Infinity. Because the RIC is also satisfied, the solution is also not c(m)=0 for all m, so a non-degenerate linear solution exists.",
                False,
            )
        if FHWC:
            return (
                "Because the FHWC is satisfied, the solution is not c(m)=Infinity. However, because the RIC is violated, the solution is degenerate at c(m) = 0 for all m.",
                True,
            )
        if RIC:
            return (
                "Because the FHWC condition is violated, the RIC is satisfied, and the consumer is not constrained, the solution is degenerate at c(m)=Infinity.",
                True,
            )
        if GPFacRaw < 1.0:
            return (
                "Because the FHWC and the RIC are both violated, the GICRaw decides the limit. Because the GICRaw is satisfied and the consumer is not constrained, the solution is degenerate at c(m)=Infinity.",
                True,
            )
        if GPFacRaw > 1.0:
            return (
                "Because the FHWC and the RIC are both violated, the GICRaw decides the limit. Because the GICRaw is also violated, the solution is degenerate at c(m)=0 for all m.",
                True,
            )
        return (
            "Because the FHWC and the RIC are both violated, the GICRaw decides the limit. Because the Growth Patience Factor is exactly one, the solution is degenerate at c(m)=1 for all m.",
            True,
        )

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
        verbose : int
            Which results are logged: none below 2, all of them at 2 or more.
            The conditions, the attribute ``degenerate``, and the full report in
            ``bilt["conditions_report"]`` are the same at every level.

        Returns
        -------
        None
        """
        verbose, should_continue = self._setup_condition_check(verbose)
        if not should_continue:
            return

        # Check individual conditions and add their results to the report
        self.check_AIC(verbose)
        self.check_RIC(verbose)
        self.check_GICRaw(verbose)
        self.check_FVAC(verbose)
        self.check_FHWC(verbose)

        # Report on the degeneracy of the consumption function solution
        verdict, degenerate = self._describe_limiting_solution()
        self.degenerate = degenerate
        self.bilt["solution_verdict"] = verdict
        self.log_condition_result(None, None, "\n" + verdict, verbose)

        if (
            degenerate
        ):  # All of the other checks are meaningless if the solution is degenerate
            self._emit_conditions_report(verbose)
            return

        # Report on the consequences of the Absolute Impatience Condition
        if self.conditions["AIC"]:
            AIC_message = "\nBecause the AIC is satisfied, the absolute amount of consumption is expected to fall over time."
        else:
            AIC_message = "\nBecause the AIC is violated, the absolute amount of consumption is expected to grow over time."
        self.log_condition_result(None, None, AIC_message, verbose)

        # Report on the consequences of the Growth Impatience Condition
        if self.conditions["GICRaw"] and self._constraint_can_bind():
            GIC_message = "\nBecause the GICRaw is satisfied, the ratio of individual wealth to permanent income is expected to fall until the borrowing constraint binds."
        elif self.conditions["GICRaw"]:
            GIC_message = "\nBecause the GICRaw is satisfied, the ratio of individual wealth to permanent income is expected to fall toward minus human wealth, the natural borrowing constraint."
        elif self.conditions["FHWC"]:
            GIC_message = "\nBecause the GICRaw is violated but the FHWC is satisfied, the ratio of individual wealth to permanent income is expected to rise toward infinity."
        else:
            pass  # pragma: nocover
            # This can never be reached! If GICRaw and FHWC both fail, then the RIC also fails, and we would have exited by this point.
        self.log_condition_result(None, None, GIC_message, verbose)

        self._emit_conditions_report(verbose)

    def get_perf_foresight_conditions(self):
        """
        Return all of the perfect foresight conditions and what they imply for this
        agent's parameters, using :func:`calc_perf_foresight_conditions`. This is optional:
        solve does not call it, and calling it changes nothing on the agent.

        Returns
        -------
        results : PerfForesightConditions or None
            None where ``check_conditions`` gives no report (anything but an infinite
            horizon with one repeated period), and for types with income risk, whose
            conditions differ.
        """
        if isinstance(self, IndShockConsumerType):
            return None
        if self.cycles != 0 or self.T_cycle > 1:
            return None
        return calc_perf_foresight_conditions(
            self.Rfree[0],
            self.DiscFac,
            self.CRRA,
            self.PermGroFac[0],
            self.LivPrb[0],
            getattr(self, "BoroCnstArt", None),
        )

    def convergence_report_hook(self, solution_now, solution_last, info):
        """
        The target part of the convergence report (see AgentType.report_convergence):
        the two target levels of market resources and where convergence stands at
        the knots of the consumption function nearest them.

        The knots of the consumption function are the endogenous gridpoints, pairs
        (a_i, c_i) with m_i = a_i + c_i, and next period's ratio is
        m' = R a / (G psi') + theta', so E[m' | a_i] = Rbar a_i + E[theta] exactly,
        with Rbar = (R / G) E[1 / psi]. Comparing (m_i - E[theta]) / Rbar with a_i
        therefore compares m_i with E[m' | m_i]: smaller means the household at
        m_i expects its ratio to rise (it is building wealth), larger means it
        expects it to fall, and the crossing is the individual target. Under
        perpetual youth the dead are replaced by newborns with expected ratio
        m_0 = Rbar E[kNrm_0] + E[theta], so the population's mean ratio is
        stationary where m = LivPrb (Rbar a + E[theta]) + (1 - LivPrb) m_0: the
        same comparison with the mortality-adjusted factor LivPrb Rbar (and the
        newborn term, zero when newborns start with no capital). Both are one
        vectorized sign test along the knots; the crossing is placed by linear
        interpolation between the two flanking knots, which is exact because the
        consumption function is linear between them. Under a kinked interest
        rate each knot takes the factor that applies to its assets (Rsave on
        saving, Rboro on borrowing); the solver puts a pair of knots at the kink,
        so the factor is constant between knots and the test stays exact. When
        every knot is on the building side the target lies above the grid, and
        the report says so. For each target the report gives the flanking knots,
        the change of consumption at them in the last cycle and the distance to
        the fixed point that the contraction rate implies there. Not available
        for state-dependent solutions (a list of consumption functions), for a
        solution without a consumption function cFunc, nor without income shocks
        (a perfect foresight agent). The types with a risky asset and the
        representative agent have targets too, not yet worked out here; their
        own hooks say so.

        Parameters
        ----------
        solution_now : ConsumerSolution
            The converged solution.
        solution_last : ConsumerSolution
            The iterate before it.
        info : dict
            The report so far (cycles, last_step, contraction_rate, ...).

        Returns
        -------
        extra : dict
        """
        if not hasattr(self, "IncShkDstn"):
            return {"target_note": "not available without income shocks"}
        cFunc = getattr(solution_now, "cFunc", None)
        if cFunc is None:
            return {"target_note": "not available for a solution without cFunc"}
        if isinstance(cFunc, list) or isinstance(solution_last.cFunc, list):
            return {"target_note": "not available for state-dependent solutions"}
        interp = cFunc.functions[0] if hasattr(cFunc, "functions") else cFunc
        if not hasattr(interp, "x_list"):
            return {"target_note": "not available for this consumption function"}
        m_knots = np.asarray(interp.x_list, dtype=float)
        c_knots = np.asarray(interp.y_list, dtype=float)
        keep = m_knots >= solution_now.mNrmMin
        m_knots, c_knots = m_knots[keep], c_knots[keep]
        if m_knots.size < 2:
            return {"target_note": "fewer than two knots above mNrmMin"}
        a_knots = m_knots - c_knots

        if hasattr(self, "Rboro") and hasattr(self, "Rsave"):
            # a kinked interest rate: each knot takes the factor that applies to
            # its assets (at a = 0 the two give the same m'), and newborns'
            # capital earns Rsave
            R_knots = np.where(a_knots > 0.0, self.Rsave, self.Rboro)
            R_newborn = self.Rsave
        else:
            R_knots = float(np.asarray(self.Rfree).ravel()[0])
            R_newborn = R_knots
        PermGroFac = float(np.asarray(self.PermGroFac).ravel()[0])
        LivPrb = float(np.asarray(self.LivPrb).ravel()[0])
        dstn = self.IncShkDstn[0]
        pmv = np.asarray(dstn.pmv, dtype=float)
        PermShk = np.asarray(dstn.atoms[0], dtype=float)
        TranShk = np.asarray(dstn.atoms[1], dtype=float)
        Ex_PermShkInv = float(np.sum(pmv / PermShk))
        Ex_TranShk = float(np.sum(pmv * TranShk))
        kNrmInitDstn = getattr(self, "kNrmInitDstn", None)
        if kNrmInitDstn is not None:
            Ex_kNrmInit = float(
                np.sum(
                    np.asarray(kNrmInitDstn.pmv, dtype=float)
                    * np.asarray(kNrmInitDstn.atoms, dtype=float).ravel()
                )
            )
        else:
            Ex_kNrmInit = 0.0
        Rbar = R_knots / PermGroFac * Ex_PermShkInv
        newborn_m = R_newborn / PermGroFac * Ex_PermShkInv * Ex_kNrmInit + Ex_TranShk

        r = info.get("contraction_rate", np.nan)
        extra = {"newborn_m": newborn_m}
        # gap_i > 0 where the household at m_i expects its ratio to rise (building
        # wealth), < 0 where it expects it to fall; the target is where it crosses
        comparisons = {
            "individual": (Rbar * a_knots + Ex_TranShk) - m_knots,
            "population": (
                LivPrb * (Rbar * a_knots + Ex_TranShk) + (1.0 - LivPrb) * newborn_m
            )
            - m_knots,
        }
        for name, gap in comparisons.items():
            crossing = np.where((gap[:-1] > 0.0) & (gap[1:] <= 0.0))[0]
            if crossing.size > 0:
                i0 = int(crossing[0])
                i1 = i0 + 1
                weight = gap[i0] / (gap[i0] - gap[i1])
                target = float(m_knots[i0] + weight * (m_knots[i1] - m_knots[i0]))
                above = False
            elif np.all(gap > 0.0):
                # building wealth at every knot: the target lies above the grid
                i0, i1 = m_knots.size - 2, m_knots.size - 1
                target, above = np.inf, True
            else:
                extra["target_" + name] = None
                extra["target_" + name + "_knots"] = None
                extra["target_" + name + "_above_grid"] = False
                extra["last_step_at_target_" + name] = None
                extra["implied_distance_at_target_" + name] = None
                continue
            pair = (float(m_knots[i0]), float(m_knots[i1]))
            step = max(
                abs(float(cFunc(pair[0])) - float(solution_last.cFunc(pair[0]))),
                abs(float(cFunc(pair[1])) - float(solution_last.cFunc(pair[1]))),
            )
            extra["target_" + name] = target
            extra["target_" + name + "_knots"] = pair
            extra["target_" + name + "_above_grid"] = above
            extra["last_step_at_target_" + name] = step
            extra["implied_distance_at_target_" + name] = _implied_distance(step, r)
        return extra

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

            if Delta_mNrm_ZeroFunc is BalGroFunc:
                # Under perfect foresight the two loci are the same function
                mNrmTrg = mNrmStE
            else:
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
    "PermShk_inf": None,  # Lower bound of log permanent shocks, in standard deviations (None --> no bound over the atoms, the default for the tails, -3, with IncShkExpectation "continuous"; -np.inf --> no bound)
    "PermShk_sup": None,  # Upper bound of log permanent shocks, in standard deviations (None --> no bound over the atoms, the default for the tails with IncShkExpectation "continuous", none jumpless; np.inf --> no bound)
    "TranShk_inf": None,  # Lower bound of log transitory shocks while employed, in standard deviations (None --> as PermShk_inf)
    "TranShk_sup": None,  # Upper bound of log transitory shocks while employed, in standard deviations (None --> as PermShk_sup)
    "PermShk_tails": None,  # Tails of bounded log permanent shocks: "truncated" at each bound, or "jumpless" (truncated below, the density zero at the bound) (None --> "truncated", or "jumpless" with IncShkExpectation "continuous")
    "TranShk_tails": None,  # Tails of bounded log transitory shocks while employed: "truncated" or "jumpless" (None --> as PermShk_tails)
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
    "CubicBool": False,  # Whether to use cubic spline interpolation
    "stationary_method": "iterate",  # How to solve when cycles=0: "iterate" (backward induction) or "anderson" (accelerated stationary solve)
    "stationary_options": {},  # Options of the stationary solve method (see HARK.stationary)
    "IncShkExpectation": "atoms",  # Expectation over the income shocks' atoms or "continuous" laws
    "PermShkQuadCount": 4,  # Gauss-Legendre nodes a piece of the continuous permanent shock
    "TranShkQuadCount": 4,  # Gauss-Legendre nodes a piece of the continuous transitory shock
}
IndShockConsumerType_simulation_default = {
    # PARAMETERS REQUIRED TO SIMULATE THE MODEL
    "AgentCount": 10000,  # Number of agents of this type
    "T_age": None,  # Age after which simulated agents are automatically killed
    "PermGroFacAgg": 1.0,  # Aggregate permanent income growth factor
    # (The portion of PermGroFac attributable to aggregate productivity growth)
    "NewbornTransShk": False,  # Whether Newborns have transitory shock
    "NewbornPermShk": True,  # Whether newborns have a permanent shock (and growth)
    "NewbornEmployed": True,  # Whether a newborn that draws a transitory shock is employed
    # ADDITIONAL OPTIONAL PARAMETERS
    "PerfMITShk": False,  # Do Perfect Foresight MIT Shock
    # (Forces Newborns to follow solution path of the agent they replaced if True)
    "neutral_measure": False,  # Whether to use permanent income neutral measure (see Harmenberg 2021)
    "init_shuffle": False,  # Exact-marginal initial-state draws when True (see sim_birth)
    "death_shuffle": False,  # Deterministic death counts when True (see sim_death)
    "income_shuffle": False,  # Exact per-period shock frequencies when True (see get_shocks)
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
    IncShkExpectation: str
        How the solver takes the expectation over the income shocks: "atoms", the
        default, sums over IncShkDstn's atoms; "continuous" integrates against
        the laws IncShkDstn records (see solve_one_period_ConsIndShock_continuous).
    PermShkQuadCount: int
        With IncShkExpectation "continuous", Gauss-Legendre nodes a piece of the
        permanent shock's integral; 4 by default.
    TranShkQuadCount: int
        With IncShkExpectation "continuous", Gauss-Legendre nodes a piece of the
        transitory shock's integral; 4 by default.

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
    kLogInitMean: float
        Mean of Log initial Normalized Assets.
    kLogInitStd: float
        Std of Log initial Normalized Assets.
    pLogInitMean: float
        Mean of Log initial permanent income.
    pLogInitStd: float
        Std of Log initial permanent income.
    PermGroFacAgg: float
        Aggregate permanent income growth factor (The portion of PermGroFac attributable to aggregate productivity growth).
        The simulation places each newborn cohort at the aggregate level, which grows by this factor every period:
        one makes a stationary economy, any other value a growing one whose levels depend on the date. The Jacobian
        tools require the same choice (see HARK.SSJutils, Two economies).
    PerfMITShk: boolean
        Do Perfect Foresight MIT Shock (Forces Newborns to follow solution path of the agent they replaced if True).
    NewbornTransShk: boolean
        Whether Newborns have transitory shock.
    NewbornPermShk: boolean
        Whether newborns have a permanent shock, and expected growth, in their
        first period (default True). If False, their permanent shock is 1, so
        they enter with pLvl drawn from pLvlInitDstn exactly. Read by
        ``simulate``; the model-file simulator (``initialize_sym``) refuses False.
    NewbornEmployed: boolean
        Whether a newborn that draws a transitory shock in its first period
        (NewbornTransShk True) is born employed (default True): its draw comes
        from the income distribution with the unemployment event left out
        (employed_atoms). If False, it may draw the unemployment event. A newborn
        that draws no transitory shock gets 1 either way. Read by ``simulate``.

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
        "track_vars": ["aNrm", "cNrm", "mNrm", "pLvl"],
    }
    # The one period solver for each IncShkExpectation. A subclass whose solver
    # wraps IndShock's names its own pair; one whose solver is not among them
    # takes the expectation over the atoms only.
    expectation_solvers = {
        "atoms": solve_one_period_ConsIndShock,
        "continuous": solve_one_period_ConsIndShock_continuous,
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

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.use_IncShkExpectation()

    def update_income_process(self):
        self.update("IncShkDstn", "PermShkDstn", "TranShkDstn")

    def get_shocks(self):
        """
        Gets permanent and transitory income shocks for this period.  Samples from IncShkDstn for
        each period in the cycle.

        When ``income_shuffle`` is True (default False), draws use
        ``DiscreteDistribution.draw(N, shuffle=True)`` (exact
        floor-plus-leftover shock frequencies with random assignment),
        eliminating cross-sectional sampling noise in the shock
        composition.  The default path is the original iid RNG code,
        preserved verbatim.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Whether Newborns have transitory shock. The default is False.
        NewbornTransShk = self.NewbornTransShk

        PermShkNow = np.zeros(self.AgentCount)  # Initialize shock arrays
        TranShkNow = np.zeros(self.AgentCount)
        newborn = self.t_age == 0
        _cache = getattr(self, "_cache_base_shock_draws", False)
        warn_if_shuffle_voids_base_draw_cache(self)
        base_draws_dict = {}
        for s in np.unique(self.t_cycle):
            idx = self.t_cycle == s
            t = s - 1

            N = np.sum(idx)
            if N > 0:
                # set current income distribution
                IncShkDstnNow = self.IncShkDstn[t]
                # and permanent growth factor
                PermGroFacNow = self.PermGroFac[t]
                # Draw income shocks from the discrete distribution
                if getattr(self, "income_shuffle", False):
                    ShockDraws = IncShkDstnNow.draw(N, shuffle=True)
                    PermShkNow[idx] = ShockDraws[0] * PermGroFacNow
                    TranShkNow[idx] = ShockDraws[1]
                elif _cache:
                    # Same uniforms and same inversion as draw_events:
                    # the P-stream is unchanged; the draws are recorded
                    # for dual-measure Q-CDF inversion.
                    base_draws = IncShkDstnNow._rng.uniform(size=N)
                    base_draws_dict[s] = base_draws
                    EventDraws = cdf_invert(base_draws, IncShkDstnNow.pmv)
                    PermShkNow[idx] = IncShkDstnNow.atoms[0][EventDraws] * PermGroFacNow
                    TranShkNow[idx] = IncShkDstnNow.atoms[1][EventDraws]
                else:
                    # Original RNG path, preserved bit-for-bit.
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
            # Born employed (the default): a newborn that draws a transitory
            # shock draws it with the unemployment event left out
            if NewbornTransShk and getattr(self, "NewbornEmployed", True):
                keep = employed_atoms(IncShkDstnNow)
                if keep is not None:
                    IncShkDstnNow = restrict_to_atoms(IncShkDstnNow, keep)

            # Draw income shocks from the discrete distribution
            if getattr(self, "income_shuffle", False):
                ShockDraws = IncShkDstnNow.draw(N, shuffle=True)
                PermShkNow[idx] = ShockDraws[0] * PermGroFacNow
                TranShkNow[idx] = ShockDraws[1]
            elif _cache:
                base_draws = IncShkDstnNow._rng.uniform(size=N)
                base_draws_dict["newborn"] = base_draws
                EventDraws = cdf_invert(base_draws, IncShkDstnNow.pmv)
                PermShkNow[idx] = IncShkDstnNow.atoms[0][EventDraws] * PermGroFacNow
                TranShkNow[idx] = IncShkDstnNow.atoms[1][EventDraws]
            else:
                # Original RNG path, preserved bit-for-bit.
                EventDraws = IncShkDstnNow.draw_events(N)
                PermShkNow[idx] = (
                    IncShkDstnNow.atoms[0][EventDraws] * PermGroFacNow
                )  # permanent "shock" includes expected growth
                TranShkNow[idx] = IncShkDstnNow.atoms[1][EventDraws]

        #  Whether Newborns have transitory shock. The default is False.
        if not NewbornTransShk:
            TranShkNow[newborn] = 1.0
        # Whether newborns have a permanent shock and growth. The default is True.
        if not getattr(self, "NewbornPermShk", True):
            PermShkNow[newborn] = 1.0

        if _cache:
            self._base_shock_draws = base_draws_dict

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
            TranShkDstn = LogNormalMeanOne(sigma=self.TranShkStd[0]).discretize(
                N=200,
                method="equiprobable",
                tail_N=50,
                tail_order=1.3,
                tail_bound=[0.05, 0.95],
            )
            TranShkDstn = add_discrete_outcome_constant_mean(
                TranShkDstn, p=self.UnempPrb, x=self.IncUnemp
            )
            PermShkDstn = LogNormalMeanOne(sigma=self.PermShkStd[0]).discretize(
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

    def solve_stationary(self, verbose=False):
        """Solve the infinite-horizon problem for its stationary solution directly.

        ``solve_agent`` calls this in place of iterating cycles when
        ``stationary_method`` is not ``"iterate"``.  The problem is posed as
        :class:`ConsIndShockStationaryProblem`, whose map is this model's own
        per-period solver, and handed to the method named by
        ``stationary_method`` with the options in ``stationary_options`` (see
        :mod:`HARK.stationary`); ``tolerance`` is the threshold on the largest
        change of the iterate between sweeps, as it is for backward induction.
        With ``vFuncBool`` the value function is then converged on the fixed
        policy the same way.  The record of the solve is stored in
        ``stationary_info``; its ``total_sweeps`` counts every evaluation of the
        per-period solver, warm start and value stage included, and is what
        ``completed_cycles`` reports.

        Parameters
        ----------
        verbose : bool
            Print the progress of the iteration.

        Returns
        -------
        solution : [ConsumerSolution]
            The stationary solution, as a one-element list.
        """
        if self.solve_one_period is not solve_one_period_ConsIndShock:
            solver = getattr(
                self.solve_one_period, "__name__", repr(self.solve_one_period)
            )
            raise NotImplementedError(
                "The stationary solve is implemented for the ConsIndShock problem; "
                f"{type(self).__name__} solves its periods with {solver}."
            )
        if self.T_cycle != 1:
            raise NotImplementedError(
                "The stationary solve requires a one-period cycle (T_cycle=1)."
            )
        problem = ConsIndShockStationaryProblem(
            self._first("IncShkDstn"),
            self._first("LivPrb"),
            self._first("DiscFac"),
            self.CRRA,
            self._first("Rfree"),
            self._first("PermGroFac"),
            self.BoroCnstArt,
            self.aXtraGrid,
            self.CubicBool,
            self.solution_terminal,
        )
        return self._solve_stationary_problem(problem, verbose)

    def _first(self, name):
        """A parameter's value in the (single) period of the cycle."""
        value = getattr(self, name)
        return value[0] if name in self.time_vary else value

    def _solve_stationary_problem(self, problem, verbose):
        """Run the stationary method on ``problem`` (and on its value problem)."""
        solution, info = solve_stationary_problem(
            problem,
            method=self.stationary_method,
            tol=self.tolerance,
            options=self.stationary_options,
            verbose=verbose,
        )
        total = info["sweeps"]
        if problem.warm_start is not None:
            info = {**info, "warm_start_sweeps": problem.warm_start["sweeps"]}
            total += problem.warm_start["sweeps"]
        if self.vFuncBool:
            solution, info_value = solve_stationary_problem(
                problem.value_problem(),
                method=self.stationary_method,
                tol=self.tolerance,
                options=self.stationary_options,
                verbose=verbose,
            )
            info = {**info, "value": info_value}
            total += info_value["sweeps"]
        # Every sweep is one backward-induction cycle, so the total is the
        # number to set against completed_cycles of the stock loop.
        info["total_sweeps"] = total
        self.stationary_info = info
        self.completed_cycles = total
        self.solution_distance = info["move"]
        return [solution]

    def pre_solve(self):
        self.use_IncShkExpectation()
        self.check_restrictions()
        self.construct("solution_terminal")
        # quiet silences the conditions report, but post_solve needs the conditions
        self.check_conditions(verbose=self.verbose)

    def assign_parameters(self, **kwds):
        """
        Assign parameters as AgentType.assign_parameters does, and point
        solve_one_period at the solver for IncShkExpectation if it is among them.
        """
        super().assign_parameters(**kwds)
        # While the agent is being made, its __init__ does this at the end
        if "IncShkExpectation" in kwds and "time_inv" in self.__dict__:
            self.use_IncShkExpectation()

    def solver_imposes_infimum_limit(self):
        """
        Whether this agent's one period solver imposes the natural borrowing
        constraint from the true shock laws' infima where they are bounded
        below away from zero (calc_boro_const_nat_from_infima), as its mark
        imposes_infimum_limit says. Other models' solvers still use the atoms'
        limit.

        Returns
        -------
        bool
        """
        return getattr(self.solve_one_period, "imposes_infimum_limit", False)

    def solve(self, *args, **kwds):
        """
        Solve the model, as AgentType.solve does, after setting solve_one_period
        for IncShkExpectation (use_IncShkExpectation), which reaches the
        subclasses whose pre_solve does not call this one's.
        """
        self.use_IncShkExpectation()
        return super().solve(*args, **kwds)

    def use_IncShkExpectation(self):
        """
        Point solve_one_period at the solver for IncShkExpectation in the
        class's expectation_solvers: for IndShockConsumerType,
        solve_one_period_ConsIndShock for "atoms", and
        solve_one_period_ConsIndShock_continuous for "continuous", which then
        also reads PermShkQuadCount and TranShkQuadCount. A type whose solver is
        not among them takes the expectation over the atoms, and "continuous"
        raises.

        It runs when the agent is made, when assign_parameters sets the option,
        and in pre_solve and solve, so that code calling solve_one_period after
        pre_solve gets the solver the option names, as solve does, and so do
        the subclasses whose pre_solve does not call this class's. An option set
        directly as an attribute takes effect at the next pre_solve or solve.
        """
        expectation = getattr(self, "IncShkExpectation", "atoms")
        if expectation not in ("atoms", "continuous"):
            raise ValueError(
                'IncShkExpectation must be "atoms" or "continuous", not '
                f"{expectation!r}."
            )
        solvers = type(self).expectation_solvers
        if self.solve_one_period not in solvers.values():
            if expectation == "continuous":
                raise NotImplementedError(
                    f"{type(self).__name__}'s solver takes the expectation over the "
                    'income shocks\' atoms only; IncShkExpectation "continuous" is '
                    "an option of IndShockConsumerType's solver."
                )
            return
        self.solve_one_period = solvers[expectation]
        if expectation == "continuous":
            for name in ("PermShkQuadCount", "TranShkQuadCount"):
                if name not in self.time_inv:
                    self.time_inv.append(name)

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
        aux_dict["VAFac"] = (self.DiscFac * self.LivPrb[0]) * (
            self.PermGroFac[0] * UtilCompPermShk
        ) ** (1.0 - self.CRRA)

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
        if self.solver_imposes_infimum_limit() and laws_bounded_below(
            self.IncShkDstn[0]
        ):
            # The limit implied by the true distributions' infima binds like
            # BoroCnstArt, where the laws are bounded below away from zero
            PermShkInf, TranShkInf = self.IncShkDstn[0].limit["infimum"]
            temp = PermShkInf * aux_dict["FHWFac"]
            BoroCnstArt = max(BoroCnstArt, -TranShkInf * temp / (1.0 - temp))
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
        aux_dict["Delta_mNrm_ZeroFunc"] = lambda m: (
            (1.0 - 1.0 / Ex_Rnrm) * m + 1.0 / Ex_Rnrm
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
            True: f"VAFac={VAFac:.5f} : The Risk-Modified Finite Value of Autarky Factor satisfies the Risk-Modified Finite Value of Autarky Condition ℒβ(G‖Ψ‖_(1-ρ))^(1-ρ) < 1.",
            False: f"VAFac={VAFac:.5f} : The Risk-Modified Finite Value of Autarky Factor violates the Risk-Modified Finite Value of Autarky Condition ℒβ(G‖Ψ‖_(1-ρ))^(1-ρ) < 1.",
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
        verbose, should_continue = self._setup_condition_check(verbose)
        if not should_continue:
            return

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
            self._emit_conditions_report(verbose)
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
            self._emit_conditions_report(verbose)
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
            GICHrm_message = "\nBecause the GICHrm is violated, there does not exist a target ratio of the individual market resources to permanent income, under the permanent-income-neutral measure."
        self.log_condition_result(None, None, GICHrm_message, verbose)

        self._emit_conditions_report(verbose)


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
    kLogInitMean: float
        Mean of Log initial Normalized Assets.
    kLogInitStd: float
        Std of Log initial Normalized Assets.
    pLogInitMean: float
        Mean of Log initial permanent income.
    pLogInitStd: float
        Std of Log initial permanent income.
    PermGroFacAgg: float
        Aggregate permanent income growth factor (The portion of PermGroFac attributable to aggregate productivity growth).
        The simulation places each newborn cohort at the aggregate level, which grows by this factor every period:
        one makes a stationary economy, any other value a growing one whose levels depend on the date. The Jacobian
        tools require the same choice (see HARK.SSJutils, Two economies).
    PerfMITShk: boolean
        Do Perfect Foresight MIT Shock (Forces Newborns to follow solution path of the agent they replaced if True).
    NewbornTransShk: boolean
        Whether Newborns have transitory shock.
    NewbornPermShk: boolean
        Whether newborns have a permanent shock, and expected growth, in their
        first period (default True). If False, their permanent shock is 1, so
        they enter with pLvl drawn from pLvlInitDstn exactly. Read by
        ``simulate``; the model-file simulator (``initialize_sym``) refuses False.
    NewbornEmployed: boolean
        Whether a newborn that draws a transitory shock in its first period
        (NewbornTransShk True) is born employed (default True): its draw comes
        from the income distribution with the unemployment event left out
        (employed_atoms). If False, it may draw the unemployment event. A newborn
        that draws no transitory shock gets 1 either way. Read by ``simulate``.

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
        "track_vars": ["aNrm", "cNrm", "mNrm", "pLvl"],
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
# It also carries the calibration's yearly trend as PermGroFacAgg (1.016): the
# simulation then places each newborn cohort 1.6 percent above the last, a growing
# economy whose levels depend on the date. For a stationary economy set
# PermGroFacAgg = 1.0 on the agent; the Jacobian tools require the choice to be
# made either way (see HARK.SSJutils, Two economies).
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

# Make dictionaries based on Carroll QJE (1997) lifecycle specifications
buffer_stock_lifecycle_base = {
    "CRRA": 2.0,
    "DiscFac": 0.96,
    "PermGroFacAgg": 1.02,
    "kLogInitMean": -1000.0,
    "kLogInitStd": 0.0,
    "pLogInitStd": 0.0,
    "Rfree": 49 * [1.00],
    "PermShkStd": 40 * [0.1] + 9 * [0.0],
    "TranShkStd": 40 * [0.1] + 9 * [0.0],
    "UnempPrb": 0.005,
    "IncUnemp": 0.0,
    "UnempPrbRet": 0.0005,
    "IncUnempRet": 0.0,
    "LivPrb": 49 * [1.0],
    "T_cycle": 49,
    "T_retire": 40,
    "T_age": 50,
    "AgentCount": 10000,
    "cycles": 1,
}

unskilled_update = {
    "pLogInitMean": np.log(1 / 1.03),
    "PermGroFac": 14 * [1.03] + 25 * [1.0] + [0.7] + 9 * [1.0],
}

operative_update = {
    "pLogInitMean": np.log(1 / 1.025),
    "PermGroFac": 24 * [1.025] + 15 * [1.01] + [0.7] + 9 * [1.0],
}

manager_update = {
    "pLogInitMean": np.log(1 / 1.03),
    "PermGroFac": 29 * [1.03] + 10 * [0.99] + [0.7] + 9 * [1.0],
}

# Carroll QJE (1997) life-cycle calibration for unskilled workers
buffer_stock_lifecycle_unskilled = copy(buffer_stock_lifecycle_base)
buffer_stock_lifecycle_unskilled.update(unskilled_update)

# Carroll QJE (1997) life-cycle calibration for operative workers
buffer_stock_lifecycle_operative = copy(buffer_stock_lifecycle_base)
buffer_stock_lifecycle_operative.update(operative_update)

# Carroll QJE (1997) life-cycle calibration for managerial workers
buffer_stock_lifecycle_manager = copy(buffer_stock_lifecycle_base)
buffer_stock_lifecycle_manager.update(manager_update)

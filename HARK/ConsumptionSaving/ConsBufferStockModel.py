"""
The consumption problem of Carroll's "Theoretical Foundations of Buffer Stock
Saving", as the paper states it.

:class:`BufferStockConsumerType` solves the paper's model and refuses anything
outside it. A consumer with CRRA utility (relative risk aversion above one)
lives forever, earns a constant interest factor on its assets, and receives
labor income subject to independent and identically distributed shocks:

- a permanent shock with mean one, on a bounded interval inside the positive
  reals;
- with probability ``UnempPrb``, the zero-income event, in which income is
  ``IncUnemp`` times permanent income (zero in the paper);
- otherwise a transitory draw with mean one, on a bounded interval inside the
  positive reals, scaled so that mean transitory income is one.

The conditions on the parameters are computed from the true laws of the shocks,
never from their discretization, and are reported with what the paper's
theorems conclude from them. The theorems are sufficiency results, and the
report says so where it matters. The paper's text proves them for zero income
in the zero-income event; its appendix on a positive minimum of income
(sec:ApndxPositiveMinimumIncome) proves their counterparts for positive income
there. When the permanent shock takes its lowest value with positive
probability, the consumer stays away from the borrowing limit and the MPC at
the bottom is below one; without that point mass the limit can be reached and
the MPC there is one (subsec:PositiveMinimumNoWorstEvent). With a positive
``IncUnemp`` the class by default builds a law of the first kind: the truncated
lognormal with a point mass at its infimum (:class:`PointMassAtInfimum`). The
lowest value is then the law's infimum at every number of points of the
discretization, as the lowest transitory income is the zero-income event's.

Two borrowing constraints are distinguished, and everything that depends on
which one is in force is written per scenario:

- ``"natural"``: no artificial constraint, or one at or below the natural
  borrowing constraint, so that it cannot bind. This is the paper's problem.
- ``"artificial"``: an artificial constraint above the natural one, so that it
  binds: end-of-period assets may not fall below ``BoroCnstArt``, which is at
  most zero. The worst income is then positive: either the zero-income event
  has probability zero, the paper's "liquidity constrained problem", or income
  in it is positive. For this scenario the paper's results are for the
  constraint at zero: under perfect foresight (prop:PFCExist), and under
  uncertainty for the liquidity constrained consumer and for a consumer who may
  not borrow while income in the zero-income event is positive
  (thm:existenceNoWorst, prop:NoWorstTargets). For a constraint below zero the
  report claims nothing on the paper's authority.

References
----------
Carroll, Christopher D. "Theoretical Foundations of Buffer Stock Saving."
https://econ-ark.github.io/BufferStockTheory/
"""

import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import NamedTuple

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm

from HARK import _log, set_verbosity_level
from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    calc_boro_const_nat,
    solve_one_period_ConsIndShock,
    IndShockConsumerType_aXtraGrid_default,
    IndShockConsumerType_constructors_default,
    IndShockConsumerType_kNrmInitDstn_default,
    IndShockConsumerType_pLvlInitDstn_default,
    IndShockConsumerType_simulation_default,
)
from HARK.interpolation import (
    CubicHermiteInterp,
    LinearInterp,
    LowerEnvelope,
    MargMargValueFuncCRRA,
    MargValueFuncCRRA,
)
from HARK.distributions import (
    DiscreteDistribution,
    DiscreteDistributionLabeled,
    IndexDistribution,
    MeanOneTruncatedLogNormal,
    add_discrete_outcome_constant_mean,
    combine_indep_dstns,
)

__all__ = [
    "PointMassAtInfimum",
    "BufferStockConsumerType",
    "BufferStockConditions",
    "calc_buffer_stock_conditions",
    "calc_natural_borrowing_constraint",
    "constraint_scenario",
    "init_buffer_stock",
]


###############################################################################
# The shock laws, read only through their generic interface
###############################################################################


def _law_bounds(law):
    """The infimum and supremum of a shock law."""
    if isinstance(law, DiscreteDistribution):
        atoms = np.ravel(law.atoms)
        return float(atoms.min()), float(atoms.max())
    return float(np.ravel(law.infimum)[0]), float(np.ravel(law.supremum)[0])


def _law_expect(law, func):
    """The expectation of func under a shock law, by quadrature of its quantile function."""
    if isinstance(law, DiscreteDistribution):
        return float(np.dot(law.pmv, func(np.ravel(law.atoms))))
    if isinstance(law, PointMassAtInfimum):
        return law.expect(func)
    value, _ = quad(
        lambda q: func(law.ppf(q)), 0.0, 1.0, limit=200, epsabs=1e-14, epsrel=1e-12
    )
    return float(value)


def _law_mass_at_infimum(law):
    """
    The probability that a shock law puts on its infimum: read from the atoms
    of a discrete law, from infimum_prb where a law records one, and zero for a
    continuous law.
    """
    if isinstance(law, DiscreteDistribution):
        atoms = np.ravel(law.atoms)
        return float(np.sum(np.asarray(law.pmv)[atoms == atoms.min()]))
    return float(getattr(law, "infimum_prb", 0.0))


class PointMassAtInfimum:
    """
    A shock law that takes its lowest value with positive probability.

    With probability ``prb`` the shock is the infimum of a base law. Otherwise it
    is a draw from the base law times (1 - prb * infimum) / (1 - prb), which
    keeps the mean at one when the base law's mean is one, as the draws of the
    transitory shock outside the zero-income event are scaled. The lowest value
    is the base law's infimum, whatever the number of points of a
    discretization, and the scaled draws are bounded away from it
    (ass:shocksPositiveMinimum).

    Parameters
    ----------
    base : distribution
        A continuous law with mean one, a positive infimum below one and a
        finite supremum, such as a MeanOneTruncatedLogNormal.
    prb : float
        The probability of the lowest value, strictly between zero and one.

    Attributes
    ----------
    infimum, supremum : np.array
        The bounds of the law: the base law's infimum, and its supremum scaled.
    infimum_prb : float
        The probability of the infimum.
    """

    def __init__(self, base, prb):
        inf = float(np.ravel(base.infimum)[0])
        sup = float(np.ravel(base.supremum)[0])
        self.base = base
        self.infimum_prb = float(prb)
        self.scale = (1.0 - self.infimum_prb * inf) / (1.0 - self.infimum_prb)
        self.infimum = np.array([inf])
        self.supremum = np.array([self.scale * sup])

    def ppf(self, q):
        """The quantile function."""
        q = np.asarray(q, dtype=float)
        above = np.clip((q - self.infimum_prb) / (1.0 - self.infimum_prb), 0.0, 1.0)
        return np.where(
            q <= self.infimum_prb, self.infimum[0], self.scale * self.base.ppf(above)
        )

    def expect(self, func):
        """The expectation of func under the law."""
        rest = _law_expect(self.base, lambda x: func(self.scale * x))
        return float(
            self.infimum_prb * func(self.infimum[0]) + (1.0 - self.infimum_prb) * rest
        )

    def discretize(self, N, method="equiprobable", **kwds):
        """
        The lowest value with its probability, and N points for the rest of the
        law: the base law's discretization, scaled.
        """
        rest = self.base.discretize(N, method=method, **kwds)
        atoms = np.concatenate((self.infimum, self.scale * np.ravel(rest.atoms)))
        pmv = np.concatenate(
            ([self.infimum_prb], (1.0 - self.infimum_prb) * np.asarray(rest.pmv))
        )
        limit = {
            "dist": self,
            "method": method,
            "N": N,
            "infimum": self.infimum,
            "supremum": self.supremum,
        }
        return DiscreteDistribution(
            pmv, atoms, seed=self.base.random_seed(), limit=limit
        )


def _worst_income(UnempPrb, IncUnemp, TranShkLaw):
    """
    The worst transitory income under the laws, and its probability: IncUnemp
    with probability UnempPrb when the zero-income event can happen; otherwise
    the infimum of the transitory draw, with the mass its law puts there.
    """
    if UnempPrb > 0.0:
        return float(IncUnemp), float(UnempPrb)
    return _law_bounds(TranShkLaw)[0], _law_mass_at_infimum(TranShkLaw)


def _is_point_at_one(law):
    return _law_bounds(law) == (1.0, 1.0)


###############################################################################
# Constructors
###############################################################################


def make_PermShkLaw(
    PermShkStd, PermShk_inf, PermShk_sup, PermShkMinPrb, UnempPrb, IncUnemp, RNG
):
    """
    The default law of the permanent shock: a mean-one lognormal truncated at
    PermShk_inf and PermShk_sup standard deviations of its log, whose log has
    standard deviation PermShkStd after the truncation.

    With positive income in the zero-income event the law is that lognormal
    with a point mass at its infimum (:class:`PointMassAtInfimum`), so that the
    lowest value of the permanent shock has positive probability
    (ass:shocksPositiveMinimum), of probability PermShkMinPrb; when that is
    None, the probability that a lognormal which is not truncated puts below
    the bound, norm.cdf(PermShk_inf), so that the mass the truncation removes
    at the bottom sits on the bound. PermShkMinPrb of zero gives the lognormal
    itself, with which the borrowing limit can be reached (item:NoPointMass).
    """
    law = MeanOneTruncatedLogNormal.from_sd_log(
        PermShkStd, bot=PermShk_inf, top=PermShk_sup, seed=RNG.integers(0, 2**31 - 1)
    )
    positive_minimum = UnempPrb > 0.0 and IncUnemp > 0.0
    if positive_minimum and not isinstance(law, DiscreteDistribution):
        prb = norm.cdf(PermShk_inf) if PermShkMinPrb is None else PermShkMinPrb
        if 0.0 < prb < 1.0:  # anything else is reported by check_restrictions
            return PointMassAtInfimum(law, prb)
    return law


def make_TranShkLaw(TranShkStd, TranShk_inf, TranShk_sup, RNG):
    """
    The default law of the transitory draw when income is not zero: a mean-one
    lognormal truncated at TranShk_inf and TranShk_sup standard deviations of its
    log, whose log has standard deviation TranShkStd after the truncation.
    """
    return MeanOneTruncatedLogNormal.from_sd_log(
        TranShkStd, bot=TranShk_inf, top=TranShk_sup, seed=RNG.integers(0, 2**31 - 1)
    )


def _discretize(law, count):
    if isinstance(law, DiscreteDistribution):
        return law
    return law.discretize(count, method="equiprobable")


def get_PermShkLawInf(PermShkLaw):
    """The infimum of the permanent shock's law."""
    return _law_bounds(PermShkLaw)[0]


def get_WorstInc(UnempPrb, IncUnemp, TranShkLaw):
    """
    The worst transitory income under the laws: IncUnemp when the zero-income
    event can happen, otherwise the infimum of the transitory draw.
    """
    return _worst_income(UnempPrb, IncUnemp, TranShkLaw)[0]


def make_buffer_stock_IncShkDstn(
    PermShkLaw, TranShkLaw, PermShkCount, TranShkCount, UnempPrb, IncUnemp, RNG
):
    """
    The discretized joint distribution of the permanent shock and the transitory
    shock, the latter including the zero-income event with probability UnempPrb
    and income IncUnemp; the employed draws are scaled by
    (1 - UnempPrb * IncUnemp) / (1 - UnempPrb), so mean transitory income is one.
    When UnempPrb is zero there is no such event, and no point is added for it.
    """
    perm = _discretize(PermShkLaw, PermShkCount)
    tran = _discretize(TranShkLaw, TranShkCount)
    # An inadmissible UnempPrb is reported by check_restrictions, not here
    if UnempPrb != 0.0:
        with np.errstate(divide="ignore", invalid="ignore"):
            tran = add_discrete_outcome_constant_mean(tran, x=IncUnemp, p=UnempPrb)
    joint = DiscreteDistributionLabeled.from_unlabeled(
        combine_indep_dstns(perm, tran, seed=RNG.integers(0, 2**31 - 1)),
        name="Permanent and transitory shocks to income, with the zero-income event",
        var_names=["PermShk", "TranShk"],
    )
    return IndexDistribution(distributions=[joint], seed=RNG.integers(0, 2**31 - 1))


###############################################################################
# The one-period solver
###############################################################################


def solve_one_period_ConsBufferStock(
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
    WorstInc,
    PermShkLawInf,
    cFuncSlopeAboveGrid,
):
    """
    IndShockConsumerType's one-period solver, with the natural borrowing
    constraint taken from the true laws, and, where the perfect foresight limit
    is not finite, a different extrapolation above the grid.

    That solver computes the natural constraint from the discretization, the
    lowest atom of each shock. When the worst income under the laws, WorstInc,
    is positive, the lowest atoms lie above the laws' infima, so its constraint
    is looser than the laws', and repayment would not be certain under the true
    laws. The laws' constraint,
    (mNrmMin next period - WorstInc) G PermShkLawInf / R, is then imposed as an
    artificial constraint is, unless the artificial constraint is tighter. With
    zero income in the zero-income event both natural constraints are zero, and
    nothing changes. Nor does anything change when the laws' infima are atoms
    of the discretization, as for a discrete law: the two constraints are then
    the same number, and that solver's MPC at the bottom is the laws'.

    Above its grid, that solver's consumption function approaches the perfect
    foresight solution of the current horizon. In the infinite horizon with the
    RIC or the FHWC failing, that solution does not converge as the horizon
    extends, and an extrapolation toward it would feed back into the grid. The
    class then sets cFuncSlopeAboveGrid to the limiting MPC (zero when the RIC
    fails), and the consumption function is extended above the grid along the
    line through its top gridpoint with that slope: since c is concave and its
    MPC is never below that limit, this is a lower bound on the true function.
    When cFuncSlopeAboveGrid is None, the solver's extrapolation is kept.
    """
    if WorstInc > 0.0:
        # In the arithmetic of calc_boro_const_nat, so that equal constraints
        # are equal numbers
        BoroCnstNatLaw = (solution_next.mNrmMin - WorstInc) * (
            (PermGroFac * PermShkLawInf) / Rfree
        )
        BoroCnstNatAtoms = calc_boro_const_nat(
            solution_next.mNrmMin, IncShkDstn, Rfree, PermGroFac
        )
        if BoroCnstNatLaw > BoroCnstNatAtoms and (
            BoroCnstArt is None or BoroCnstArt < BoroCnstNatLaw
        ):
            BoroCnstArt = BoroCnstNatLaw
    solution = solve_one_period_ConsIndShock(
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
    if cFuncSlopeAboveGrid is not None:
        _extend_along_a_line(solution, cFuncSlopeAboveGrid, CRRA, CubicBool)
    return solution


def _extend_along_a_line(solution, slope, CRRA, CubicBool):
    """
    Replace the consumption function's extrapolation above the grid by the line
    through the top gridpoint with the given slope, and rebuild the marginal
    value functions from it.
    """
    unconstrained, constrained = solution.cFunc.functions
    x = np.asarray(unconstrained.x_list)
    y = np.asarray(unconstrained.y_list)
    slope = float(slope)
    if CubicBool:
        unconstrained = CubicHermiteInterp(
            x,
            y,
            np.asarray(unconstrained.dydx_list),
            y[-1] - slope * x[-1],
            slope,
        )
    else:
        # A point on the line makes the last segment's slope the limiting MPC
        unconstrained = LinearInterp(
            np.append(x, x[-1] + 1.0), np.append(y, y[-1] + slope)
        )
    solution.cFunc = LowerEnvelope(unconstrained, constrained, nan_bool=False)
    solution.vPfunc = MargValueFuncCRRA(solution.cFunc, CRRA)
    if CubicBool:
        solution.vPPfunc = MargMargValueFuncCRRA(solution.cFunc, CRRA)


###############################################################################
# The conditions
###############################################################################


@dataclass(frozen=True)
class BufferStockCondition:
    """
    The result of one condition of the paper.

    Attributes
    ----------
    name : str
        The condition's name, as a key of an agent's ``conditions``.
    factor_name : str
        The name of its factor, as a key of an agent's ``bilt``.
    factor : float
        The factor, which the condition compares with one.
    holds : bool
        Whether the condition holds.
    meaning : str
        What the paper concludes from the result or, where its theorems are not
        proved for the parameters, which of them the condition is a hypothesis of.
    """

    name: str
    factor_name: str
    factor: float
    holds: bool
    meaning: str


@dataclass(frozen=True)
class BufferStockConditions:
    """
    The paper's conditions for one set of parameters, and what its theorems
    conclude from them. See :func:`calc_buffer_stock_conditions`.

    Attributes
    ----------
    conditions : dict
        A :class:`BufferStockCondition` for each of AIC, RIC, WRIC, GICRaw,
        GICMod, GICSdl, GICHrm, FHWC, PFFVAC and FVAC.
    WorstInc : float
        The worst transitory income under the laws: IncUnemp when the
        zero-income event can happen, otherwise the infimum of the transitory
        draw.
    WorstPrb : float
        The probability of the worst income event: UnempPrb when income in the
        zero-income event is zero, as in the paper; otherwise the probability
        that the transitory income and the permanent shock are both at their
        worst.
    MPCmin : float
        The limiting MPC as market resources go to infinity.
    MPCmax : float
        The limiting MPC as market resources go to their minimum.
    hNrm : float
        Human wealth as a ratio to permanent income, excluding this period's
        income, as in a solution's ``hNrm`` (the paper's h counts it, and is one
        larger).
    BoroCnstNat : float
        The natural borrowing constraint on end-of-period assets: zero when the
        worst income is zero, as in the paper.
    BoroCnstArt : float or None
        The artificial borrowing constraint, if any.
    scenario : str
        "natural" or "artificial": see :func:`constraint_scenario`.
    existence : str
        What the paper establishes about a limiting solution. "established"
        when an existence theorem of the paper applies and its hypotheses hold,
        so a non-degenerate limiting solution exists (thm:convgtobellman, under
        the WRIC and the FVAC); "fails" when the paper shows that the limiting
        solution is degenerate, as when the WRIC fails, so that consumption goes
        to zero everywhere as the horizon extends; "not established" where the
        paper proves nothing, as when the FVAC fails with the WRIC holding. With
        positive income in the zero-income event the theorems are the
        appendix's, and need minimum human wealth to be finite: when the
        permanent shock takes its lowest value with positive probability,
        thm:existencePositiveMinimum; when it does not, the borrowing limit can
        be reached and thm:existenceNoWorst needs only the FVAC. Where minimum
        human wealth is not finite, prop:InfiniteMinHumanWealth. Where an
        artificial constraint binds, the paper's results are for the constraint
        at zero: prop:PFCExist under perfect foresight; under uncertainty,
        thm:existenceNoWorst, under the FVAC, for the liquidity constrained
        consumer with no zero-income event (item:Restrained) and for a consumer
        who may not borrow while income in that event is positive
        (item:Artificial). Otherwise "not established".
    existence_basis : str
        The statement of the paper on which ``existence`` rests, or why the
        paper establishes nothing.
    target_guaranteed : bool
        Whether a theorem of the paper guarantees a unique buffer stock target:
        a limiting solution is established under uncertainty, and the GICMod
        holds (thm:target; with positive income in the zero-income event,
        thm:targetPositiveMinimum; where the borrowing limit can be reached,
        prop:NoWorstTargets).
    pseudo_target_guaranteed : bool
        Whether a theorem of the paper guarantees a unique pseudo-target: a
        limiting solution is established under uncertainty, and the GICRaw
        holds (thm:MSSBalExists; with positive income in the zero-income event,
        thm:pseudoTargetPositiveMinimum; where the borrowing limit can be
        reached, prop:NoWorstTargets).
    notes : tuple
        Further statements: about the solution above the asset grid, and about
        what the class computes where the paper states nothing.
    """

    conditions: dict
    WorstInc: float
    WorstPrb: float
    MPCmin: float
    MPCmax: float
    hNrm: float
    BoroCnstNat: float
    BoroCnstArt: object
    scenario: str
    existence: str
    existence_basis: str
    target_guaranteed: bool
    pseudo_target_guaranteed: bool
    notes: tuple

    def to_dict(self):
        """Return the results as nested dictionaries of plain values."""
        return asdict(self)


# name: factor name, factor's title, condition's title, inequality, meaning if
# it holds, meaning if it fails
_CONDITIONS = {
    "AIC": (
        "APFac",
        "Absolute Patience Factor",
        "Absolute Impatience Condition",
        "Þ < 1",
        "Under the RIC, expected consumption growth as m goes to infinity tends to Þ, below one.",
        "Under the RIC, expected consumption growth as m goes to infinity tends to Þ, not below one. No result requires the AIC.",
    ),
    "RIC": (
        "RPFac",
        "Return Patience Factor",
        "Return Impatience Condition",
        "Þ/R < 1",
        "The limiting MPC as m goes to infinity is 1 - Þ/R, which is positive.",
        "The limiting MPC as m goes to infinity is zero, and human wealth is infinite given the FVAC.",
    ),
    "WRIC": (
        "WRPFac",
        "Weak Return Patience Factor",
        "Weak Return Impatience Condition",
        "WorstPrb^(1/ρ) Þ/R < 1",
        "With the FVAC, a non-degenerate limiting solution exists.",
        "Consumption goes to zero at every m as the horizon extends: the limiting solution is degenerate.",
    ),
    "GICRaw": (
        "GPFacRaw",
        "Growth Patience Factor",
        "Growth Impatience Condition",
        "Þ/G < 1",
        "With the WRIC and the FVAC, a unique pseudo-target exists.",
        "No pseudo-target is guaranteed.",
    ),
    "GICMod": (
        "GPFacMod",
        "Modified Growth Patience Factor",
        "Strong Growth Impatience Condition",
        "Þ E[1/ψ]/G < 1",
        "With the WRIC and the FVAC, a unique buffer stock target exists.",
        "No target is guaranteed: there may be none, or two.",
    ),
    "GICSdl": (
        "GPFacSdl",
        "Szeidl Growth Patience Factor",
        "Szeidl's Growth Impatience Condition",
        "log(Þ/G) < E[log ψ]",
        "Invariant distributions of m, c and a exist across a population (Szeidl).",
        "Szeidl's sufficient condition for invariant distributions does not hold.",
    ),
    "GICHrm": (
        "GPFacHrm",
        "Harmenberg Growth Patience Factor",
        "Harmenberg's Growth Impatience Condition",
        "log(Þ/G) < E[ψ log ψ]",
        "An invariant permanent-income-weighted distribution exists (Harmenberg).",
        "Harmenberg's sufficient condition does not hold.",
    ),
    "FHWC": (
        "FHWFac",
        "Finite Human Wealth Factor",
        "Finite Human Wealth Condition",
        "G/R < 1",
        "Human wealth is finite, and the perfect foresight solution bounds consumption from above.",
        "Human wealth is infinite.",
    ),
    "PFFVAC": (
        "PFVAFac",
        "Perfect Foresight Value of Autarky Factor",
        "Perfect Foresight Finite Value of Autarky Condition",
        "β G^(1-ρ) < 1",
        "The value of autarky without risk is finite.",
        "The value of autarky without risk is not finite.",
    ),
    "FVAC": (
        "VAFac",
        "Value of Autarky Factor",
        "Finite Value of Autarky Condition",
        "β G^(1-ρ) E[ψ^(1-ρ)] < 1",
        "With the WRIC, a non-degenerate limiting solution exists.",
        "The paper proves nothing about the limiting solution.",
    ),
}

# Where the paper's theorems are not proved, a condition they need is reported
# as their hypothesis, not as what they conclude:
# name -> (the meaning when the condition holds, the meaning when it fails), with
# None for the paper's own
_NOT_FOR_THIS_PROCESS = "which is not proved for this income process."
_MEANINGS_UNPROVED = {
    "WRIC": (
        "With the FVAC, a hypothesis of the paper's existence theorem, "
        + _NOT_FOR_THIS_PROCESS,
        None,
    ),
    "FVAC": (
        "With the WRIC, a hypothesis of the paper's existence theorem, "
        + _NOT_FOR_THIS_PROCESS,
        None,
    ),
    "GICRaw": (
        "With the WRIC and the FVAC, a hypothesis of the paper's theorem on the "
        "pseudo-target, " + _NOT_FOR_THIS_PROCESS,
        None,
    ),
    "GICMod": (
        "With the WRIC and the FVAC, a hypothesis of the paper's theorem on the "
        "buffer stock target, " + _NOT_FOR_THIS_PROCESS,
        None,
    ),
}

_NOT_WHERE_IT_BINDS = (
    "which is not proved where an artificial borrowing constraint binds."
)
_WRIC_WHERE_IT_BINDS = (
    "Not needed where an artificial borrowing constraint binds: the MPC at the bottom is "
    "then one."
)
_MEANINGS_ARTIFICIAL = {
    "WRIC": (_WRIC_WHERE_IT_BINDS, _WRIC_WHERE_IT_BINDS),
    "FVAC": (
        "The value of autarky is finite. It is a hypothesis of the paper's existence "
        "theorem, " + _NOT_WHERE_IT_BINDS,
        "The value of autarky is not finite.",
    ),
    "GICRaw": (
        "A hypothesis of the paper's theorem on the pseudo-target, "
        + _NOT_WHERE_IT_BINDS,
        None,
    ),
    "GICMod": (
        "A hypothesis of the paper's theorem on the buffer stock target, "
        + _NOT_WHERE_IT_BINDS,
        None,
    ),
}
# With positive income in the zero-income event and a point mass at the
# permanent shock's lowest value, the FVAC and finite minimum human wealth
# imply the WRIC
_MEANINGS_POSITIVE_MINIMUM = {
    "WRIC": (
        (
            "With the FVAC, a non-degenerate limiting solution exists. With IncUnemp above "
            "zero, the FVAC and finite minimum human wealth imply it "
            "(remark:WRICWorstImplied)."
        ),
        (
            "Consumption goes to zero at every m as the horizon extends: the limiting "
            "solution is degenerate. With IncUnemp above zero this happens only when the "
            "FVAC fails (remark:WRICWorstImplied)."
        ),
    ),
}
# Where the borrowing limit can be reached: no point mass at the permanent
# shock's lowest value (item:NoPointMass), or a constraint at zero that binds
# (item:Restrained, item:Artificial)
_WRIC_LIMIT_REACHABLE = (
    "Not needed: the borrowing limit can be reached, and the MPC at the bottom is one "
    "(subsec:PositiveMinimumNoWorstEvent)."
)
_MEANINGS_LIMIT_REACHABLE = {
    "WRIC": (_WRIC_LIMIT_REACHABLE, _WRIC_LIMIT_REACHABLE),
    "FVAC": ("A non-degenerate limiting solution exists (thm:existenceNoWorst).", None),
    "GICRaw": (
        "With the FVAC, a unique pseudo-target exists (prop:NoWorstTargets).",
        None,
    ),
    "GICMod": (
        "With the FVAC, a unique buffer stock target exists (prop:NoWorstTargets).",
        None,
    ),
}
_MEANINGS_ARTIFICIAL_PROVED = dict(_MEANINGS_LIMIT_REACHABLE)
_MEANINGS_ARTIFICIAL_PROVED["WRIC"] = (_WRIC_WHERE_IT_BINDS, _WRIC_WHERE_IT_BINDS)
# Where minimum human wealth is not finite (prop:InfiniteMinHumanWealth)
_NEEDS_FINITE_MIN_HUMAN_WEALTH = (
    "which also needs minimum human wealth to be finite (eq:FiniteMinHumanWealth)."
)
_MEANINGS_INFINITE_MIN_HUMAN_WEALTH = {
    "WRIC": (
        "A hypothesis of the paper's existence theorem, "
        + _NEEDS_FINITE_MIN_HUMAN_WEALTH,
        (
            "With minimum human wealth not finite the paper's conclusion from its failure "
            "does not apply; prop:InfiniteMinHumanWealth says where consumption goes."
        ),
    ),
    "FVAC": (
        (
            "With minimum human wealth not finite, the problem has no non-degenerate "
            "limiting solution (prop:InfiniteMinHumanWealth)."
        ),
        (
            "With minimum human wealth not finite, the paper says where consumption goes "
            "only in some cases (prop:InfiniteMinHumanWealth)."
        ),
    ),
    "GICRaw": (
        "A hypothesis of the paper's theorem on the pseudo-target, "
        + _NEEDS_FINITE_MIN_HUMAN_WEALTH,
        None,
    ),
    "GICMod": (
        "A hypothesis of the paper's theorem on the buffer stock target, "
        + _NEEDS_FINITE_MIN_HUMAN_WEALTH,
        None,
    ),
}
# Under perfect foresight the paper says when the liquidity constraint matters
_MEANINGS_ARTIFICIAL_PERFECT_FORESIGHT = dict(_MEANINGS_ARTIFICIAL)
_MEANINGS_ARTIFICIAL_PERFECT_FORESIGHT["GICRaw"] = (
    "Under perfect foresight the liquidity constraint is relevant (eq:LiqConstrBinds).",
    (
        "Under perfect foresight the liquidity constraint is not relevant "
        "(eq:LiqConstrBinds)."
    ),
)


class _Model(NamedTuple):
    """What a verdict needs to know of the model besides its conditions."""

    UnempPrb: float
    IncUnemp: float
    BoroCnstArt: object
    PerfectForesight: bool  # no zero-income event, and both shocks always one
    PermShkMinPrb: float  # the probability of the permanent shock's lowest value
    BoroCnstNat: float  # the natural borrowing constraint
    WorstGroFac: float  # PermGroFac times the permanent shock's lowest value


class _Verdict(NamedTuple):
    """What the paper establishes for one set of parameters under one scenario."""

    existence: str
    existence_basis: str
    target_guaranteed: bool
    pseudo_target_guaranteed: bool
    MPCmax: float
    meanings: dict  # the conditions' meanings where they are not the paper's


NATURAL = "natural"
ARTIFICIAL = "artificial"


def calc_natural_borrowing_constraint(Rfree, PermGroFac, WorstInc, PermShkMin):
    """
    The natural borrowing constraint on end-of-period assets in the infinite
    horizon: repayment must be certain on the worst path, with the permanent
    shock at its infimum and transitory income at its worst, WorstInc, every
    period. It is zero when WorstInc is zero, as in the paper, and minus
    infinity when the worst path's human wealth is infinite.
    """
    if WorstInc == 0.0:
        return 0.0
    x = PermGroFac * PermShkMin / Rfree
    return -WorstInc * x / (1.0 - x) if x < 1.0 else -np.inf


def constraint_scenario(BoroCnstArt, BoroCnstNat):
    """
    Which borrowing constraint is in force: "natural" when there is no
    artificial constraint, or it is at or below the natural one and so cannot
    bind; "artificial" when it is above the natural one, so that it binds.
    """
    if BoroCnstArt is None or BoroCnstArt <= BoroCnstNat:
        return NATURAL
    return ARTIFICIAL


def _verdict_infinite_min_human_wealth(factors, holds, model):
    """
    prop:InfiniteMinHumanWealth, for positive income in the zero-income event
    with PermGroFac times the permanent shock's lowest value not below Rfree.
    Under the FVAC the problem has no non-degenerate limiting solution.
    Consumption goes to infinity at every m if Þ is below PermGroFac times that
    lowest value (part i); with a point mass there, it goes to zero if
    WorstPrb^(1/ρ) Þ is above that product or Þ is above PermGroFac (part ii).
    Without the point mass the paper states only the first statement and part
    (i). Elsewhere, with the FVAC failing, it does not say where consumption
    goes.

    Returns
    -------
    existence, basis : str
    """
    APFac = factors["AIC"]
    WorstAPFac = factors["WRIC"] * APFac / factors["RIC"]  # WorstPrb^(1/ρ) Þ
    lead = (
        "Because PermGroFac times the permanent shock's lowest value is not below "
        "Rfree, minimum human wealth is not finite (eq:FiniteMinHumanWealth)"
    )
    where = "(prop:InfiniteMinHumanWealth)"
    if APFac < model.WorstGroFac:
        direction = (
            "consumption grows without bound at every m as the horizon extends, because "
            "Þ is below PermGroFac times that lowest value"
        )
    elif model.PermShkMinPrb > 0.0 and WorstAPFac > model.WorstGroFac:
        direction = (
            "consumption goes to zero at every m as the horizon extends, because "
            "WorstPrb^(1/ρ) Þ is above PermGroFac times that lowest value"
        )
    elif model.PermShkMinPrb > 0.0 and factors["GICRaw"] > 1.0:
        direction = (
            "consumption goes to zero at every m as the horizon extends, because Þ is "
            "above PermGroFac"
        )
    else:
        direction = None
    if direction is not None:
        return (
            "fails",
            f"{lead}, and {direction}: the limiting solution is degenerate {where}.",
        )
    if holds["FVAC"]:
        return "fails", (
            f"{lead}, and under the FVAC the problem has no non-degenerate limiting "
            f"solution {where}."
        )
    return "not established", (
        f"{lead}. The FVAC fails, and the paper does not say whether consumption at a "
        f"given m goes to zero, goes to infinity or fails to converge {where}."
    )


def _verdict_natural(factors, holds, model):
    """
    What the paper's theorems establish under the natural constraint, and
    nothing beyond them.

    With zero income in the zero-income event, the paper's text: a limiting
    solution exists under the WRIC and the FVAC (thm:convgtobellman), and
    nothing is proved when the FVAC fails; a target is guaranteed by the GICMod
    as well (thm:target), a pseudo-target by the GICRaw (thm:MSSBalExists).

    With positive income there, its appendix on a positive minimum of income.
    When the permanent shock takes its lowest value with positive probability
    (ass:shocksPositiveMinimum): the same conclusions, with the WRIC using the
    probability of the worst event (eq:WRICWorst) and minimum human wealth
    finite (eq:FiniteMinHumanWealth), from thm:existencePositiveMinimum,
    thm:targetPositiveMinimum and thm:pseudoTargetPositiveMinimum; the FVAC and
    finite minimum human wealth imply that WRIC (remark:WRICWorstImplied). When
    it does not (item:NoPointMass), the borrowing limit can be reached and the
    MPC at the bottom is one: a limiting solution exists under the FVAC and
    finite minimum human wealth (thm:existenceNoWorst), with the target and the
    pseudo-target as before (prop:NoWorstTargets). When minimum human wealth is
    not finite, the problem has no non-degenerate limiting solution under the
    FVAC, and the patience factors say in some cases where consumption goes
    (prop:InfiniteMinHumanWealth).

    When the WRIC fails the limiting solution is degenerate.

    Parameters
    ----------
    factors : dict
        Each condition's factor.
    holds : dict
        Whether each condition holds.
    model : _Model
        The zero-income event, the artificial constraint, and what the verdict
        needs of the laws.

    Returns
    -------
    verdict : _Verdict
    """
    WRIC, FVAC = holds["WRIC"], holds["FVAC"]
    meanings = {}
    if model.UnempPrb == 0.0:
        # Not admitted by the class, which requires an artificial constraint
        # that binds when there is no zero-income event
        existence = "not established"
        basis = (
            "The paper's theorems for the model with uncertainty assume a zero-income "
            "event of positive probability (ass:shocks). Without one, and with no "
            "artificial borrowing constraint that binds, they establish nothing."
        )
        meanings = _MEANINGS_UNPROVED
    elif model.IncUnemp == 0.0:
        # The paper's own model
        if not WRIC:
            existence = "fails"
            basis = (
                "Because the WRIC fails, consumption goes to zero at every m as the "
                "horizon extends: the limiting solution is degenerate."
            )
        elif not FVAC:
            existence = "not established"
            basis = (
                "Because the FVAC fails, the paper does not establish a limiting solution, "
                "though the WRIC holds; its existence theorem is sufficiency only."
            )
        else:
            existence = "established"
            basis = (
                "Because the WRIC and the FVAC hold, a non-degenerate limiting solution "
                "exists (thm:convgtobellman)."
            )
    elif not np.isfinite(model.BoroCnstNat):
        # Not admitted by the class under the natural constraint
        existence, basis = _verdict_infinite_min_human_wealth(factors, holds, model)
        meanings = _MEANINGS_INFINITE_MIN_HUMAN_WEALTH
    elif model.PermShkMinPrb == 0.0:
        # The borrowing limit can be reached (item:NoPointMass)
        meanings = _MEANINGS_LIMIT_REACHABLE
        reachable = (
            "With no point mass at the permanent shock's lowest value, the borrowing "
            "limit can be reached and the MPC at the bottom is one (item:NoPointMass)."
        )
        if FVAC:
            existence = "established"
            basis = (
                f"{reachable} Because the FVAC holds and minimum human wealth is finite "
                "(eq:FiniteMinHumanWealth), a non-degenerate limiting solution exists "
                "(thm:existenceNoWorst)."
            )
        else:
            existence = "not established"
            basis = (
                f"{reachable} Because the FVAC fails, the paper does not establish a "
                "limiting solution; its existence theorem for this case is sufficiency "
                "only (thm:existenceNoWorst)."
            )
    elif not WRIC:
        existence = "fails"
        basis = (
            "Because the WRIC fails with the probability of the worst event "
            "(eq:WRICWorst) while minimum human wealth is finite "
            "(eq:FiniteMinHumanWealth), the upper bound on consumption goes to zero at "
            "every m as the horizon extends (lemma:MPCPositiveMinimum): the limiting "
            "solution is degenerate (subsec:PositiveMinimumFailures). With IncUnemp above "
            "zero this happens only when the FVAC fails (remark:WRICWorstImplied)."
        )
        meanings = _MEANINGS_POSITIVE_MINIMUM
    elif not FVAC:
        existence = "not established"
        basis = (
            "Because the FVAC fails, the paper does not establish a limiting solution, "
            "though the WRIC holds with the probability of the worst event "
            "(eq:WRICWorst); its existence theorem for positive income in the "
            "zero-income event is sufficiency only (thm:existencePositiveMinimum)."
        )
        meanings = _MEANINGS_POSITIVE_MINIMUM
    else:
        existence = "established"
        basis = (
            "Because the FVAC holds and minimum human wealth is finite "
            "(eq:FiniteMinHumanWealth), which imply the WRIC with the probability of the "
            "worst event (eq:WRICWorst, remark:WRICWorstImplied), a non-degenerate "
            "limiting solution exists (thm:existencePositiveMinimum)."
        )
        meanings = _MEANINGS_POSITIVE_MINIMUM
    established = existence == "established"
    return _Verdict(
        existence=existence,
        existence_basis=basis,
        target_guaranteed=established and holds["GICMod"],
        pseudo_target_guaranteed=established and holds["GICRaw"],
        MPCmax=1.0 - factors["WRIC"] if WRIC else 0.0,
        meanings=meanings,
    )


def _verdict_artificial(factors, holds, model):
    """
    What the paper establishes where an artificial constraint binds, and nothing
    beyond it. Its results are for the constraint that assets not be negative.

    Under perfect foresight: a non-degenerate limiting solution exists if the
    RIC holds, and, if it fails, if and only if the GICRaw holds
    (prop:PFCExist).

    Under uncertainty, its appendix on a positive minimum of income treats two
    such problems: the restrained consumer with no zero-income event, which
    with the constraint at zero is the text's liquidity constrained consumer
    (item:Restrained), and a consumer who may not borrow while income in the
    zero-income event is positive and the permanent shock takes its lowest
    value with positive probability (item:Artificial). In both a limiting
    solution exists under the FVAC (thm:existenceNoWorst), with a target under
    the GICMod and a pseudo-target under the GICRaw (prop:NoWorstTargets).

    The restrained consumer's borrowing limit below zero changes with the
    horizon, so a constant constraint below zero is not that problem. For it,
    and for any constraint below zero when the zero-income event can happen,
    the paper establishes nothing. The MPC at the bottom is one.

    Parameters and returns are those of :func:`_verdict_natural`.
    """
    at_zero = model.BoroCnstArt == 0.0
    meanings = _MEANINGS_ARTIFICIAL
    targets = False  # whether the theorems on the targets apply
    if model.PerfectForesight and at_zero:
        meanings = _MEANINGS_ARTIFICIAL_PERFECT_FORESIGHT
        problem = "the perfect foresight problem with a liquidity constraint"
        if holds["RIC"]:
            existence = "established"
            basis = (
                f"Because the RIC holds, a non-degenerate limiting solution of {problem} "
                "exists (prop:PFCExist)."
            )
        elif holds["GICRaw"]:
            existence = "established"
            basis = (
                f"Because the GICRaw holds, a non-degenerate limiting solution of {problem} "
                "exists, though the RIC fails (prop:PFCExist)."
            )
        else:
            existence = "fails"
            basis = (
                f"Because the RIC and the GICRaw both fail, the limiting solution of "
                f"{problem} is degenerate (prop:PFCExist)."
            )
    elif at_zero and (model.UnempPrb == 0.0 or model.PermShkMinPrb > 0.0):
        # With a zero-income event, a constraint at zero binds only when income
        # in the event is positive
        meanings = _MEANINGS_ARTIFICIAL_PROVED
        targets = True
        if model.UnempPrb == 0.0:
            problem = (
                "the liquidity constrained problem, with no zero-income event and no "
                "borrowing (item:Restrained with zero minimum income"
            )
        else:
            problem = (
                "the problem of a consumer who may not borrow, with positive income in the "
                "zero-income event (item:Artificial"
            )
        if holds["FVAC"]:
            existence = "established"
            basis = (
                f"Because the FVAC holds, a non-degenerate limiting solution exists for "
                f"{problem}; thm:existenceNoWorst)."
            )
        else:
            existence = "not established"
            basis = (
                f"Because the FVAC fails, the paper does not establish a limiting solution "
                f"for {problem}): its existence theorem is sufficiency only "
                "(thm:existenceNoWorst)."
            )
    else:
        existence = "not established"
        if at_zero:
            basis = (
                "With positive income in the zero-income event, the paper's result for a "
                "consumer who may not borrow assumes that the permanent shock takes its "
                "lowest value with positive probability (item:Artificial, "
                "ass:shocksPositiveMinimum). For a law without that point mass it "
                "establishes neither a limiting solution nor a target or pseudo-target."
            )
        elif model.UnempPrb == 0.0 and not model.PerfectForesight:
            basis = (
                "The paper's restrained consumer may borrow up to the minimum human wealth "
                "that an income in a zero-income event would support, a limit that changes "
                "with the horizon (item:Restrained). A constant artificial borrowing "
                "constraint below zero is not that problem, and for it the paper "
                "establishes neither a limiting solution nor a target or pseudo-target."
            )
        else:
            basis = (
                "The paper treats no artificial borrowing limit other than zero here. For "
                "an artificial borrowing constraint elsewhere it establishes neither a "
                "limiting solution nor a target or pseudo-target."
            )
        if not holds["FVAC"]:
            basis += " The FVAC fails: the value of autarky is not finite."
    established = targets and existence == "established"
    return _Verdict(
        existence=existence,
        existence_basis=basis,
        target_guaranteed=established and holds["GICMod"],
        pseudo_target_guaranteed=established and holds["GICRaw"],
        MPCmax=1.0,
        meanings=meanings,
    )


_VERDICTS = {NATURAL: _verdict_natural, ARTIFICIAL: _verdict_artificial}


def calc_buffer_stock_conditions(
    Rfree,
    DiscFac,
    CRRA,
    PermGroFac,
    UnempPrb,
    IncUnemp,
    PermShkLaw,
    TranShkLaw,
    BoroCnstArt=None,
):
    r"""
    Calculate the paper's conditions from the true laws of the shocks, and what
    its theorems conclude from them. Nothing in HARK calls this function, and it
    does not check that the parameters are admissible: see
    :meth:`BufferStockConsumerType.check_restrictions`.

    Parameters
    ----------
    Rfree : float
        Interest factor R.
    DiscFac : float
        Discount factor :math:`\beta`.
    CRRA : float
        Relative risk aversion :math:`\rho`.
    PermGroFac : float
        Growth factor of permanent income G.
    UnempPrb : float
        Probability of the zero-income event.
    IncUnemp : float
        Income in that event, as a ratio to permanent income.
    PermShkLaw : distribution
        The law of the permanent shock.
    TranShkLaw : distribution
        The law of the transitory draw when the zero-income event does not happen.
    BoroCnstArt : float or None, optional
        The artificial borrowing constraint, if any. Default None.

    Returns
    -------
    results : BufferStockConditions
    """
    R, beta, rho, G = float(Rfree), float(DiscFac), float(CRRA), float(PermGroFac)
    PermShkMin, _ = _law_bounds(PermShkLaw)

    # The worst income event: in the paper, the zero-income event, whatever the
    # permanent shock; when the worst transitory income is positive, that income
    # together with the permanent shock at its infimum.
    WorstInc, WorstPrb = _worst_income(float(UnempPrb), IncUnemp, TranShkLaw)
    PermShkMinPrb = _law_mass_at_infimum(PermShkLaw)
    if WorstInc > 0.0:
        WorstPrb *= PermShkMinPrb
    PerfectForesight = (
        float(UnempPrb) == 0.0
        and _is_point_at_one(PermShkLaw)
        and _is_point_at_one(TranShkLaw)
    )

    Ex_PermShkInv = _law_expect(PermShkLaw, lambda x: 1.0 / x)
    Ex_PermShkUtil = _law_expect(PermShkLaw, lambda x: x ** (1.0 - rho))
    Ex_LogPermShk = _law_expect(PermShkLaw, np.log)
    Ex_PermShkLogPermShk = _law_expect(PermShkLaw, lambda x: x * np.log(x))

    APFac = (R * beta) ** (1.0 / rho)
    RPFac = APFac / R
    GPFacRaw = APFac / G
    factors = {
        "AIC": APFac,
        "RIC": RPFac,
        "WRIC": WorstPrb ** (1.0 / rho) * RPFac,
        "GICRaw": GPFacRaw,
        "GICMod": GPFacRaw * Ex_PermShkInv,
        "GICSdl": GPFacRaw / np.exp(Ex_LogPermShk),
        "GICHrm": GPFacRaw / np.exp(Ex_PermShkLogPermShk),
        "FHWC": G / R,
        "PFFVAC": beta * G ** (1.0 - rho),
        "FVAC": beta * G ** (1.0 - rho) * Ex_PermShkUtil,
    }
    holds = {name: bool(factor < 1.0) for name, factor in factors.items()}

    RIC, FHWC = holds["RIC"], holds["FHWC"]
    FHWFac = factors["FHWC"]
    hNrm = FHWFac / (1.0 - FHWFac) if FHWC else np.inf

    BoroCnstNat = calc_natural_borrowing_constraint(R, G, WorstInc, PermShkMin)
    scenario = constraint_scenario(BoroCnstArt, BoroCnstNat)
    model = _Model(
        float(UnempPrb),
        float(IncUnemp),
        BoroCnstArt,
        PerfectForesight,
        PermShkMinPrb,
        float(BoroCnstNat),
        G * float(PermShkMin),
    )
    verdict = _VERDICTS[scenario](factors, holds, model)
    existence = verdict.existence

    conditions = {}
    for name, factor in factors.items():
        factor_name, _, _, _, if_holds, if_fails = _CONDITIONS[name]
        other_if_holds, other_if_fails = verdict.meanings.get(name, (None, None))
        if holds[name]:
            meaning = other_if_holds or if_holds
        else:
            meaning = other_if_fails or if_fails
        conditions[name] = BufferStockCondition(
            name=name,
            factor_name=factor_name,
            factor=float(factor),
            holds=holds[name],
            meaning=meaning,
        )

    notes = []
    if existence != "fails" and not (RIC and FHWC):
        failing = " and ".join(k for k in ("RIC", "FHWC") if not holds[k])
        notes.append(
            f"With the {failing} failing, the perfect foresight limit of consumption is not "
            "finite. Above the asset grid the consumption function is extended along the "
            "line through its top gridpoint with slope equal to the limiting MPC, a lower "
            "bound on the true function; raise aXtraMax to see more of it."
        )
    if scenario == ARTIFICIAL:
        notes.append(
            "The artificial borrowing constraint binds. Below a kink in the consumption "
            "function the consumer ends the period with assets of exactly BoroCnstArt, and "
            "the MPC there is one; the Euler equation holds with inequality there."
        )
    elif float(UnempPrb) > 0.0 and IncUnemp > 0.0:
        if PermShkMinPrb > 0.0:
            notes.append(
                "With IncUnemp > 0 the consumer may borrow, up to minimum human wealth, "
                "-BoroCnstNat (lemma:NaturalBorrowingLimit). The worst event is the "
                "zero-income event together with the permanent shock at its lowest value, "
                "of probability WorstPrb (eq:pWorst). The target and the pseudo-target "
                "can be negative, and the target lies below the pseudo-target where the "
                "consumer borrows at them (lemma:orderingPositiveMinimum)."
            )
        else:
            notes.append(
                "With IncUnemp > 0 and no point mass at the permanent shock's lowest value, "
                "the worst event has probability zero (WorstPrb). The consumer may borrow up "
                "to minimum human wealth, -BoroCnstNat, and can reach that limit: the MPC at "
                "the bottom is one (subsec:PositiveMinimumNoWorstEvent). The consumer "
                "chooses the limit itself at low enough resources if E[(ψ - ψ_min)^(-ρ)] is "
                "finite, and otherwise approaches it with a slope that tends to one; the "
                "solver's discretized law, whose lowest point lies above the infimum, has a "
                "segment at the limit, which narrows as the number of points grows. The "
                "target and the pseudo-target can be negative (prop:NoWorstTargets)."
            )

    return BufferStockConditions(
        conditions=conditions,
        WorstInc=float(WorstInc),
        WorstPrb=float(WorstPrb),
        MPCmin=max(0.0, 1.0 - RPFac) if RIC else 0.0,
        MPCmax=verdict.MPCmax,
        hNrm=float(hNrm),
        BoroCnstNat=float(BoroCnstNat),
        BoroCnstArt=BoroCnstArt,
        scenario=scenario,
        existence=existence,
        existence_basis=verdict.existence_basis,
        target_guaranteed=verdict.target_guaranteed,
        pseudo_target_guaranteed=verdict.pseudo_target_guaranteed,
        notes=tuple(notes),
    )


###############################################################################
# The agent type
###############################################################################

BufferStockConsumerType_constructors_default = dict(
    IndShockConsumerType_constructors_default
)
BufferStockConsumerType_constructors_default.update(
    {
        "PermShkLaw": make_PermShkLaw,
        "TranShkLaw": make_TranShkLaw,
        "PermShkLawInf": get_PermShkLawInf,
        "WorstInc": get_WorstInc,
        "IncShkDstn": make_buffer_stock_IncShkDstn,
    }
)

# The paper's calibration (its table of parameters)
BufferStockConsumerType_IncShkDstn_default = {
    "UnempPrb": 0.005,  # Probability of the zero-income event
    "IncUnemp": 0.0,  # Income in that event, zero in the paper
    "PermShkStd": 0.1,  # Standard deviation of log permanent shocks, after truncation
    "PermShk_inf": -3.0,  # Lower bound of log permanent shocks, in its standard deviations
    "PermShk_sup": 3.0,  # Upper bound of log permanent shocks, in its standard deviations
    "PermShkCount": 7,  # Points in the discretization of the permanent shock
    # With IncUnemp above zero, the probability of the permanent shock's lowest
    # value, its law's infimum; None for the mass the truncation removes there
    "PermShkMinPrb": None,
    "TranShkStd": 0.1,  # Standard deviation of log transitory draws, after truncation
    "TranShk_inf": -3.0,  # Lower bound of log transitory draws, in its standard deviations
    "TranShk_sup": 3.0,  # Upper bound of log transitory draws, in its standard deviations
    "TranShkCount": 7,  # Points in the discretization of the transitory draw
}

BufferStockConsumerType_solving_default = {
    "cycles": 0,  # Infinite horizon
    "T_cycle": 1,  # One period, repeated
    "pseudo_terminal": False,
    "constructors": BufferStockConsumerType_constructors_default,
    "CRRA": 2.0,  # Relative risk aversion, above one
    "Rfree": [1.04],  # Interest factor
    "DiscFac": 0.96,  # Discount factor
    "LivPrb": [1.0],  # No mortality in the paper's problem
    "PermGroFac": [1.03],  # Growth factor of permanent income
    "BoroCnstArt": None,  # No artificial constraint: the natural one is in force
    "vFuncBool": False,
    "CubicBool": False,
}

BufferStockConsumerType_defaults = {}
BufferStockConsumerType_defaults.update(BufferStockConsumerType_IncShkDstn_default)
BufferStockConsumerType_defaults.update(IndShockConsumerType_kNrmInitDstn_default)
BufferStockConsumerType_defaults.update(IndShockConsumerType_pLvlInitDstn_default)
BufferStockConsumerType_defaults.update(IndShockConsumerType_aXtraGrid_default)
# Just above the natural constraint the consumer saves only the share 1 - MPCmax
# of resources above it, and the Euler equation there is dominated by the worst
# event, whose marginal utility depends on that saving. A small error in c is
# then a large relative error in it, the larger the closer MPCmax is to one, as
# it is with positive income in the zero-income event, where the worst event has
# a small probability. The asset grid therefore has 300 points nested five times
# toward the bottom, and 60 more spaced by a constant ratio from 1e-8 to 1e-2
# above the constraint: they keep the Euler error on the grid at or below half a
# percent with linear interpolation. IndShockConsumerType's 48 points nested
# three times leave several percent in the paper's own model, and the points
# close to the constraint matter where MPCmax is within a percent of one.
BufferStockConsumerType_defaults["aXtraCount"] = 300
BufferStockConsumerType_defaults["aXtraNestFac"] = 5
BufferStockConsumerType_defaults["aXtraExtra"] = np.geomspace(1e-8, 1e-2, 60).tolist()
BufferStockConsumerType_defaults.update(BufferStockConsumerType_solving_default)
BufferStockConsumerType_defaults.update(IndShockConsumerType_simulation_default)
init_buffer_stock = BufferStockConsumerType_defaults


class BufferStockConsumerType(IndShockConsumerType):
    r"""
    The consumer of "Theoretical Foundations of Buffer Stock Saving", as the
    paper states it.

    .. math::
        \begin{align*}
        v(m_t) &= \max_{c_t} u(c_t) + \beta \mathbb{E}_t[(G\psi_{t+1})^{1-\rho} v(m_{t+1})], \\
        a_t &= m_t - c_t, \\
        m_{t+1} &= a_t R/(G\psi_{t+1}) + \xi_{t+1}, \\
        u(c) &= \frac{c^{1-\rho}}{1-\rho}, \quad \rho > 1,
        \end{align*}

    with :math:`\xi = \mu` (``IncUnemp``, zero in the paper) in the zero-income
    event, of probability :math:`\wp` (``UnempPrb``), and
    :math:`\xi = \theta (1 - \wp\mu)/(1 - \wp)` otherwise. The permanent shock
    :math:`\psi` and the draw :math:`\theta` have mean one and bounded supports
    inside the positive reals.

    The class refuses parameters outside the paper's model, computes the
    paper's conditions from the true laws, warns when a condition of the paper's
    existence theorem fails, and records the target and pseudo-target where the
    paper's theorems guarantee them. With ``IncUnemp`` above zero the theorems
    are those of the paper's appendix on a positive minimum of income, which
    assume that the permanent shock takes its lowest value with positive
    probability; the consumer may then borrow, up to minimum human wealth, and
    the target and pseudo-target can be negative. Every stable point found is
    listed in ``bilt`` as ``mNrmTrg_roots`` and ``mNrmStE_roots``.

    Solving Parameters
    ------------------
    cycles : int
        0 (the default) for the infinite horizon; a positive number of
        repetitions of the period gives a finite horizon.
    CRRA : float, :math:`\rho`
        Relative risk aversion, above one.
    Rfree : list[float], :math:`R`
        The interest factor, positive, as a list of length one.
    DiscFac : float, :math:`\beta`
        The discount factor, positive.
    PermGroFac : list[float], :math:`G`
        The growth factor of permanent income, positive, as a list of length one.
    UnempPrb : float, :math:`\wp`
        The probability of the zero-income event: strictly between zero and one,
        or zero with an artificial borrowing constraint that binds.
    IncUnemp : float, :math:`\mu`
        Income in that event, zero in the paper's text; it must be at least
        zero and below the worst employed income. With a positive value and no
        artificial constraint that binds, PermGroFac times the permanent
        shock's lowest value must be below Rfree.
    PermShkLaw, TranShkLaw : distribution
        The laws of the permanent shock and of the transitory draw. By default
        mean-one truncated lognormals built from PermShkStd, PermShk_inf,
        PermShk_sup and their transitory counterparts; with IncUnemp above zero
        the permanent shock's default law is that lognormal with a point mass
        at its infimum (:class:`PointMassAtInfimum`), unless PermShkMinPrb is
        zero. To use another law, assign it to the attribute and call
        update_income_process().
    PermShkMinPrb : float or None
        With IncUnemp above zero, the probability of the permanent shock's
        lowest value in its default law: at least zero and below one. With a
        positive value the consumer stays away from the borrowing limit; zero
        gives the lognormal itself, with which the limit can be reached and the
        MPC at the bottom is one (item:NoPointMass). None (the default) for the
        probability that a lognormal which is not truncated puts below the
        bound, norm.cdf(PermShk_inf).
    PermShkCount, TranShkCount : int
        The number of points in each law's equiprobable discretization. A point
        mass of a law is a point of its own, so the lowest values and their
        probabilities are the same at every number of points.
    BoroCnstArt : float or None
        An artificial borrowing constraint on end-of-period assets, at most
        zero. None (the default), or one at or below the natural constraint,
        cannot bind. One above the natural constraint binds; that needs the
        worst income to be positive: UnempPrb of zero, or IncUnemp above zero.
    on_no_solution : str
        "warn" (the default) or "raise": what solve does when the WRIC fails,
        so the limiting solution is degenerate. When the FVAC fails, solve
        warns. quiet=True silences warnings.

    Attributes
    ----------
    conditions : dict
        Whether each condition holds.
    bilt : dict
        The factors, the limiting MPCs, the probability of the worst event,
        the report in "conditions_report", the full record in
        "BufferStockConditions", and the stable points.
    """

    default_ = dict(IndShockConsumerType.default_)
    default_["params"] = BufferStockConsumerType_defaults
    default_["solver"] = solve_one_period_ConsBufferStock
    time_inv_ = IndShockConsumerType.time_inv_ + [
        "WorstInc",
        "PermShkLawInf",
        "cFuncSlopeAboveGrid",
    ]
    cFuncSlopeAboveGrid = None
    solving_defaults = BufferStockConsumerType_solving_default
    simulation_defaults = IndShockConsumerType_simulation_default
    on_no_solution = "warn"

    def get_buffer_stock_conditions(self):
        """
        Return the paper's conditions for this agent's parameters, from
        :func:`calc_buffer_stock_conditions`. Calling it changes nothing on the agent.
        """
        return calc_buffer_stock_conditions(
            self.Rfree[0],
            self.DiscFac,
            self.CRRA,
            self.PermGroFac[0],
            self.UnempPrb,
            self.IncUnemp,
            self.PermShkLaw,
            self.TranShkLaw,
            self.BoroCnstArt,
        )

    def constraint_scenario(self):
        """
        Which borrowing constraint is in force for this agent: see
        :func:`constraint_scenario`.
        """
        PermShkMin, _ = _law_bounds(self.PermShkLaw)
        WorstInc, _ = _worst_income(
            float(self.UnempPrb), float(self.IncUnemp), self.TranShkLaw
        )
        BoroCnstNat = calc_natural_borrowing_constraint(
            self.Rfree[0], self.PermGroFac[0], WorstInc, PermShkMin
        )
        return constraint_scenario(self.BoroCnstArt, BoroCnstNat)

    def check_restrictions(self):
        """
        Refuse parameters outside the paper's model, naming every violation. The
        restrictions that hold whatever the borrowing constraint are checked here;
        those that depend on it, in the method for the scenario in force.
        """
        problems = []

        def scalar(name):
            value = getattr(self, name)
            if isinstance(value, (list, tuple, np.ndarray)):
                if len(value) != 1:
                    problems.append(
                        f"{name} must have one value: the paper's parameters are constant."
                    )
                    return None
                value = value[0]
            return float(value)

        if self.T_cycle != 1:
            problems.append(
                "T_cycle must be 1: the paper's problem repeats one period (eq:veqnNrmRecBellman)."
            )
        CRRA = scalar("CRRA")
        if CRRA is not None and not CRRA > 1.0:
            problems.append(f"CRRA must exceed one (subsec:Setup), not {CRRA}.")
        DiscFac = scalar("DiscFac")
        if DiscFac is not None and not DiscFac > 0.0:
            problems.append(f"DiscFac must be positive (subsec:Setup), not {DiscFac}.")
        Rfree = scalar("Rfree")
        if Rfree is not None and not Rfree > 0.0:
            problems.append(f"Rfree must be positive, not {Rfree}.")
        PermGroFac = scalar("PermGroFac")
        if PermGroFac is not None and not PermGroFac > 0.0:
            problems.append(f"PermGroFac must be positive, not {PermGroFac}.")
        LivPrb = scalar("LivPrb")
        if LivPrb is not None and LivPrb != 1.0:
            problems.append(
                f"LivPrb must be 1: the paper's problem has no mortality, not {LivPrb}."
            )
        if self.on_no_solution not in ("warn", "raise"):
            problems.append(
                f'on_no_solution must be "warn" or "raise", not {self.on_no_solution!r}.'
            )

        # The laws: mean one, on a bounded interval inside the positive reals that
        # contains one (ass:shocks)
        for name in ("PermShkLaw", "TranShkLaw"):
            law = getattr(self, name)
            inf, sup = _law_bounds(law)
            mean = _law_expect(law, lambda x: x)
            if not inf > 0.0:
                problems.append(
                    f"{name}'s infimum must be positive (ass:shocks), not {inf}."
                )
            if not np.isfinite(sup):
                problems.append(
                    f"{name}'s supremum must be finite (ass:shocks): its support must be bounded."
                )
            if not inf <= 1.0 <= sup:
                problems.append(
                    f"{name}'s support must contain one (ass:shocks), not [{inf}, {sup}]."
                )
            if not abs(mean - 1.0) < 1e-8:
                problems.append(f"{name}'s mean must be one (ass:shocks), not {mean}.")

        if float(self.IncUnemp) < 0.0:
            problems.append(
                f"IncUnemp must not be negative, not {float(self.IncUnemp)}."
            )
        PermShkMinPrb = getattr(self, "PermShkMinPrb", None)
        if PermShkMinPrb is not None and not 0.0 <= PermShkMinPrb < 1.0:
            problems.append(
                f"PermShkMinPrb, the probability of the permanent shock's lowest value, "
                f"must be at least zero and below one, or None, not {PermShkMinPrb}."
            )

        restrictions = {
            NATURAL: self._restrictions_natural,
            ARTIFICIAL: self._restrictions_artificial,
        }
        restrictions[self.constraint_scenario()](problems)

        if problems:
            raise ValueError(
                "These parameters are outside the model of 'Theoretical Foundations of "
                "Buffer Stock Saving':\n- " + "\n- ".join(problems)
            )

    def _restrict_income_in_the_event(self, problems):
        """
        Income in the zero-income event is strictly below the worst employed
        income, TranShkMin (1 - UnempPrb IncUnemp) / (1 - UnempPrb), so that the
        event is the worst outcome.
        """
        UnempPrb, IncUnemp = float(self.UnempPrb), float(self.IncUnemp)
        TranShkMin, _ = _law_bounds(self.TranShkLaw)
        bound = TranShkMin / (1.0 - UnempPrb * (1.0 - TranShkMin))
        if not IncUnemp < bound:
            problems.append(
                f"IncUnemp ({IncUnemp}) must be below the worst employed income, so the "
                f"zero-income event is the worst outcome: with these laws, below {bound:.6f}."
            )

    def _restrictions_natural(self, problems):
        """
        The restrictions under the natural constraint, the paper's problem: the
        zero-income event has a probability strictly between zero and one
        (eq:TranShkDef); income in it is strictly below the worst employed
        income; and, if that income is positive, the natural constraint has a
        finite value, which needs PermGroFac times the permanent shock's lowest
        value to be below Rfree (eq:FiniteMinHumanWealth). Without it, the
        problem has no non-degenerate limiting solution under the FVAC
        (prop:InfiniteMinHumanWealth). The permanent shock may or may not take
        its lowest value with positive probability (ass:shocksPositiveMinimum,
        item:NoPointMass).
        """
        UnempPrb, IncUnemp = float(self.UnempPrb), float(self.IncUnemp)
        if not 0.0 < UnempPrb < 1.0:
            problem = (
                f"UnempPrb, the probability of the zero-income event, must be strictly "
                f"between zero and one (eq:TranShkDef), not {UnempPrb}."
            )
            if UnempPrb == 0.0:
                problem += (
                    " Zero is admissible only with an artificial borrowing constraint "
                    "that binds (BoroCnstArt)."
                )
            problems.append(problem)
            return
        self._restrict_income_in_the_event(problems)
        if not IncUnemp > 0.0:
            return
        PermShkMin, _ = _law_bounds(self.PermShkLaw)
        if not self.PermGroFac[0] * PermShkMin < self.Rfree[0]:
            problems.append(
                f"With IncUnemp above zero, PermGroFac times the lowest value of the "
                f"permanent shock ({self.PermGroFac[0] * PermShkMin:.6f}) must be below "
                f"Rfree ({self.Rfree[0]}): otherwise minimum human wealth, and so the "
                f"natural borrowing constraint, has no finite value "
                f"(eq:FiniteMinHumanWealth), and under the FVAC there is no "
                f"non-degenerate limiting solution (prop:InfiniteMinHumanWealth). An "
                f"artificial constraint at or below zero is then needed (BoroCnstArt)."
            )

    def _restrictions_artificial(self, problems):
        """
        The restrictions where an artificial constraint binds: the constraint is
        at most zero, the paper's liquidity constraint being that assets not be
        negative (sec:LiqConstrAsLimit); the zero-income event has a probability
        of at least zero and below one, zero being the paper's "liquidity
        constrained problem"; and, if the event can happen, income in it is
        strictly below the worst employed income. That the worst income is
        positive follows from the constraint's binding.
        """
        UnempPrb = float(self.UnempPrb)
        if float(self.BoroCnstArt) > 0.0:
            problems.append(
                f"BoroCnstArt ({self.BoroCnstArt}) must not be above zero: the paper's "
                f"liquidity constraint is that assets not be negative (sec:LiqConstrAsLimit)."
            )
        if not 0.0 <= UnempPrb < 1.0:
            problems.append(
                f"UnempPrb, the probability of the zero-income event, must be at least zero "
                f"and below one, not {UnempPrb}."
            )
            return
        if UnempPrb > 0.0:
            self._restrict_income_in_the_event(problems)

    def check_conditions(self, verbose=None):
        """
        Compute the paper's conditions from the true laws and record them in
        conditions, bilt and degenerate. verbose of 2 or more logs the report.
        """
        verbose = self.verbose if verbose is None else verbose
        results = self.get_buffer_stock_conditions()
        self.conditions = {
            name: condition.holds for name, condition in results.conditions.items()
        }
        for condition in results.conditions.values():
            self.bilt[condition.factor_name] = condition.factor
        self.bilt["WorstPrb"] = results.WorstPrb
        self.bilt["MPCmin"] = results.MPCmin
        self.bilt["MPCmax"] = results.MPCmax
        self.bilt["BufferStockConditions"] = results
        self.degenerate = results.existence == "fails"
        self.PFLimitFinite = results.conditions["RIC"].holds and (
            results.conditions["FHWC"].holds
        )
        # In the infinite horizon without a finite perfect foresight limit, extend
        # the consumption function above the grid along the limiting MPC
        if self.cycles == 0 and not self.PFLimitFinite:
            self.cFuncSlopeAboveGrid = results.MPCmin
        else:
            self.cFuncSlopeAboveGrid = None

        lines = []
        for name, condition in results.conditions.items():
            _, factor_title, condition_title, inequality, _, _ = _CONDITIONS[name]
            verb = "satisfies" if condition.holds else "violates"
            lines.append(
                f"{condition.factor_name}={condition.factor:.5f} : The {factor_title} "
                f"{verb} the {condition_title} ({name}) {inequality}. {condition.meaning}"
            )
        lines.append("")
        lines.append(results.existence_basis)
        for note in results.notes:
            lines.append(note)
        self.bilt["conditions_report"] = "\n".join(lines) + "\n"
        if verbose >= 2 and not self.quiet:
            level = _log.level
            set_verbosity_level((4 - verbose) * 10)
            _log.info(self.bilt["conditions_report"])
            set_verbosity_level(level)

    def _respond_to_conditions(self):
        """
        In the infinite horizon, raise or warn when the limiting solution is
        degenerate, and warn when the FVAC fails. Where nothing fails and the
        paper simply proves no theorem for the parameters, the report says so
        and there is no warning.
        """
        results = self.bilt["BufferStockConditions"]
        if self.cycles != 0 or results.existence == "established":
            return
        message = results.existence_basis
        if results.existence == "fails":
            if self.on_no_solution == "raise":
                raise ValueError(message)
        elif results.conditions["FVAC"].holds:
            return
        if not self.quiet:
            warnings.warn(message)

    def update_income_process(self):
        """
        Rebuild the discretized income process, and what the solver reads from
        the laws: the permanent shock's infimum and the worst income.
        """
        super().update_income_process()
        self.construct("PermShkLawInf", "WorstInc")

    def pre_solve(self):
        self.check_restrictions()
        # A law or a parameter of the zero-income event may have been assigned
        self.construct("PermShkLawInf", "WorstInc", "solution_terminal")
        self.check_conditions(verbose=self.verbose)
        self._respond_to_conditions()

    def post_solve(self):
        if self.cycles == 0 and self.T_cycle == 1:
            if "BufferStockConditions" not in self.bilt:
                self.check_conditions(verbose=0)
            self._restart_at_the_limits()
            self.calc_stable_points()

    def _restart_at_the_limits(self):
        """
        In the infinite horizon, once the solve has converged, solve again from
        its solution with human wealth and the limiting MPCs set to their limits.

        The solve stops when the consumption function converges on the grid,
        before the recursions for human wealth and the limiting MPCs converge, and
        above the grid the consumption function is built from them. Their limits
        are their fixed points, so started there they stay, and the consumption
        function above the grid is then the limit's. Done only where the limits
        are finite, under the FHWC and the RIC.
        """
        results = self.bilt["BufferStockConditions"]
        if not self.PFLimitFinite or results.existence == "fails":
            return
        start = deepcopy(self.solution[0])
        start.hNrm = results.hNrm
        start.MPCmin = results.MPCmin
        start.MPCmax = results.MPCmax
        cycles = self.completed_cycles
        self.solve(presolve=False, postsolve=False, from_solution=start)
        self.completed_cycles += cycles

    def calc_stable_points(self, force=False):
        """
        Find the buffer stock target and the pseudo-target of the solved model by
        a bracketed search, recording them in bilt as mNrmTrg and mNrmStE. Each is
        recorded where a theorem of the paper guarantees a unique one; otherwise
        it is nan. Every root found is listed in bilt as mNrmTrg_roots or
        mNrmStE_roots, whether or not the paper guarantees one.
        """
        if self.cycles != 0 or not hasattr(self, "solution"):
            return
        results = self.bilt["BufferStockConditions"]
        cFunc = self.solution[0].cFunc
        mNrmMin = self.solution[0].mNrmMin
        RNrm = self.Rfree[0] / self.PermGroFac[0]
        PermShkDstn = self.PermShkDstn[0]
        Ex_PermShkInv = float(
            np.dot(PermShkDstn.pmv, 1.0 / np.ravel(PermShkDstn.atoms))
        )

        # E[m'] - m is R'(m - c(m)) + 1 - m, with R' = (R/G) E[1/psi] for the
        # target and R/G for the pseudo-target (eq:mTargImplicit, eq:balgrostable)
        m_grid = mNrmMin + np.geomspace(1e-6, 1e4, 4001)
        for key, factor, guaranteed in (
            ("mNrmTrg", RNrm * Ex_PermShkInv, results.target_guaranteed),
            ("mNrmStE", RNrm, results.pseudo_target_guaranteed),
        ):

            def gap(m):
                return factor * (m - cFunc(m)) + 1.0 - m

            signs = np.sign(gap(m_grid))
            # A point of the search grid may itself be a root: with a binding
            # constraint and no risk, m = 1 is
            roots = [float(m) for m in m_grid[signs == 0]]
            for i in np.nonzero(signs[:-1] * signs[1:] < 0)[0]:
                roots.append(
                    brentq(
                        lambda m: float(gap(m)), m_grid[i], m_grid[i + 1], xtol=1e-12
                    )
                )
            roots.sort()
            self.bilt[key + "_roots"] = roots
            value = roots[0] if (guaranteed and len(roots) == 1) else np.nan
            self.bilt[key] = value
            setattr(self.solution[0], key, value)

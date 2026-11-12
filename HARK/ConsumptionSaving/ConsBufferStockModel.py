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
report says so where it matters. They are proved for zero income in the
zero-income event: with a positive ``IncUnemp`` the report gives the conditions,
but claims neither a limiting solution nor a target on the paper's authority.

Two borrowing constraints are distinguished, and everything that depends on
which one is in force is written per scenario:

- ``"natural"``: no artificial constraint, or one at or below the natural
  borrowing constraint, so that it cannot bind. This is the paper's problem,
  and the one supported.
- ``"artificial"``: an artificial constraint above the natural one, so that it
  binds. Not yet supported. Adding it means filling in three places, each
  marked "the artificial constraint": ``_verdict_artificial`` (what the
  conditions imply), ``BufferStockConsumerType._restrictions_artificial``
  (which parameters are admissible), and the tests. The solver, the stable
  points and the rest of the class already handle a binding constraint.

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

from HARK import _log, set_verbosity_level
from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
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
    value, _ = quad(
        lambda q: func(law.ppf(q)), 0.0, 1.0, limit=200, epsabs=1e-14, epsrel=1e-12
    )
    return float(value)


def _law_mass_at_infimum(law):
    """The probability that a shock law puts on its infimum: zero for a continuous law."""
    if isinstance(law, DiscreteDistribution):
        atoms = np.ravel(law.atoms)
        return float(np.sum(np.asarray(law.pmv)[atoms == atoms.min()]))
    return 0.0


###############################################################################
# Constructors
###############################################################################


def make_PermShkLaw(PermShkStd, PermShk_inf, PermShk_sup, RNG):
    """
    The default law of the permanent shock: a mean-one lognormal truncated at
    PermShk_inf and PermShk_sup standard deviations of its log, whose log has
    standard deviation PermShkStd after the truncation.
    """
    return MeanOneTruncatedLogNormal.from_sd_log(
        PermShkStd, bot=PermShk_inf, top=PermShk_sup, seed=RNG.integers(0, 2**31 - 1)
    )


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


def make_buffer_stock_IncShkDstn(
    PermShkLaw, TranShkLaw, PermShkCount, TranShkCount, UnempPrb, IncUnemp, RNG
):
    """
    The discretized joint distribution of the permanent shock and the transitory
    shock, the latter including the zero-income event with probability UnempPrb
    and income IncUnemp; the employed draws are scaled by
    (1 - UnempPrb * IncUnemp) / (1 - UnempPrb), so mean transitory income is one.
    """
    perm = _discretize(PermShkLaw, PermShkCount)
    tran = _discretize(TranShkLaw, TranShkCount)
    # An inadmissible UnempPrb is reported by check_restrictions, not here
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
    IncUnemp,
    PermShkLawInf,
    cFuncSlopeAboveGrid,
):
    """
    IndShockConsumerType's one-period solver, with the natural borrowing
    constraint taken from the true laws, and, where the perfect foresight limit
    is not finite, a different extrapolation above the grid.

    That solver computes the natural constraint from the discretization, the
    lowest atom of each shock. With positive income in the zero-income event,
    the lowest atom of the permanent shock lies above the law's infimum, so its
    constraint is looser than the law's, and repayment would not be certain
    under the true law. The law's constraint,
    (mNrmMin next period - IncUnemp) G PermShkLawInf / R, is then imposed as an
    artificial constraint is. With zero income in the event both constraints
    are zero, and nothing changes.

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
    if IncUnemp > 0.0:
        BoroCnstNatLaw = (
            (solution_next.mNrmMin - IncUnemp) * PermGroFac * PermShkLawInf / Rfree
        )
        if BoroCnstArt is None or BoroCnstArt < BoroCnstNatLaw:
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
    WorstPrb : float
        The probability of the worst income event: UnempPrb when income in the
        zero-income event is zero, as in the paper.
    MPCmin : float
        The limiting MPC as market resources go to infinity.
    MPCmax : float
        The limiting MPC as market resources go to their minimum.
    hNrm : float
        Human wealth as a ratio to permanent income, excluding this period's
        income, as in a solution's ``hNrm`` (the paper's h counts it, and is one
        larger).
    BoroCnstNat : float
        The natural borrowing constraint on end-of-period assets: zero when
        income in the zero-income event is zero.
    BoroCnstArt : float or None
        The artificial borrowing constraint, if any.
    scenario : str
        "natural" or "artificial": see :func:`constraint_scenario`.
    existence : str
        What the paper establishes about a limiting solution. "established"
        when its existence theorem applies and the theorem's hypotheses, the
        WRIC and the FVAC, hold, so a non-degenerate limiting solution exists;
        "fails" when the WRIC fails, so consumption goes to zero everywhere as
        the horizon extends; "not established" where the paper proves nothing:
        when the FVAC fails with the WRIC holding, and whenever income in the
        zero-income event is positive, whatever the conditions.
    existence_basis : str
        The statement of the paper on which ``existence`` rests, or why the
        paper establishes nothing.
    target_guaranteed : bool
        Whether a theorem of the paper guarantees a unique buffer stock target:
        income in the zero-income event is zero, and the WRIC, the FVAC and the
        GICMod hold (thm:target).
    pseudo_target_guaranteed : bool
        Whether a theorem of the paper guarantees a unique pseudo-target: income
        in the zero-income event is zero, and the WRIC, the FVAC and the GICRaw
        hold (thm:MSSBalExists).
    notes : tuple
        Further statements: about the solution above the asset grid, and about
        what the class computes where the paper states nothing.
    """

    conditions: dict
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

# With positive income in the zero-income event the paper's theorems are not
# proved, and a condition they need is reported as their hypothesis, not as what
# they conclude: name -> the meaning when the condition holds
_ONLY_FOR_ZERO_INCOME = "which is proved only for zero income in the zero-income event."
_MEANINGS_POSITIVE_INCOME = {
    "WRIC": "With the FVAC, a hypothesis of the paper's existence theorem, "
    + _ONLY_FOR_ZERO_INCOME,
    "FVAC": "With the WRIC, a hypothesis of the paper's existence theorem, "
    + _ONLY_FOR_ZERO_INCOME,
    "GICRaw": "With the WRIC and the FVAC, a hypothesis of the paper's theorem on the "
    "pseudo-target, " + _ONLY_FOR_ZERO_INCOME,
    "GICMod": "With the WRIC and the FVAC, a hypothesis of the paper's theorem on the "
    "buffer stock target, " + _ONLY_FOR_ZERO_INCOME,
}


class _Verdict(NamedTuple):
    """What the paper establishes for one set of parameters under one scenario."""

    existence: str
    existence_basis: str
    target_guaranteed: bool
    pseudo_target_guaranteed: bool
    MPCmax: float
    meanings: (
        dict  # name -> the meaning of a condition that holds, where not the paper's
    )


NATURAL = "natural"
ARTIFICIAL = "artificial"


def calc_natural_borrowing_constraint(Rfree, PermGroFac, IncUnemp, PermShkMin):
    """
    The natural borrowing constraint on end-of-period assets in the infinite
    horizon: repayment must be certain on the worst path, with the permanent
    shock at its infimum and income at IncUnemp every period. It is zero when
    IncUnemp is zero, as in the paper, and minus infinity when the worst path's
    human wealth is infinite.
    """
    if IncUnemp == 0.0:
        return 0.0
    x = PermGroFac * PermShkMin / Rfree
    return -IncUnemp * x / (1.0 - x) if x < 1.0 else -np.inf


def constraint_scenario(BoroCnstArt, BoroCnstNat):
    """
    Which borrowing constraint is in force: "natural" when there is no
    artificial constraint, or it is at or below the natural one and so cannot
    bind; "artificial" when it is above the natural one, so that it binds.
    """
    if BoroCnstArt is None or BoroCnstArt <= BoroCnstNat:
        return NATURAL
    return ARTIFICIAL


def _verdict_natural(factors, holds, IncUnemp):
    """
    What the paper's theorems establish under the natural constraint, and
    nothing beyond them. They are proved for zero income in the zero-income
    event. There, a limiting solution exists under the WRIC and the FVAC
    (thm:convgtobellman), and the paper proves nothing when the FVAC fails; a
    target is guaranteed by the GICMod as well (thm:target), a pseudo-target by
    the GICRaw (thm:MSSBalExists). With positive income in the event the paper
    proves none of these, so nothing is established or guaranteed, whatever the
    conditions. When the WRIC fails the limiting solution is degenerate.

    Parameters
    ----------
    factors : dict
        Each condition's factor.
    holds : dict
        Whether each condition holds.
    IncUnemp : float
        Income in the zero-income event.

    Returns
    -------
    verdict : _Verdict
    """
    WRIC, FVAC = holds["WRIC"], holds["FVAC"]
    papers_model = IncUnemp == 0.0
    meanings = {}
    if not WRIC:
        existence = "fails"
        basis = (
            "Because the WRIC fails, consumption goes to zero at every m as the horizon "
            "extends: the limiting solution is degenerate."
        )
        if not papers_model:
            basis += (
                " With IncUnemp > 0 the WRIC uses the probability of the zero-income event "
                "together with the permanent shock at its infimum, which the paper does "
                "not treat."
            )
    elif not papers_model:
        existence = "not established"
        basis = (
            "The paper proves its theorems for zero income in the zero-income event; of "
            "positive income there it says only that such a model could be transformed "
            "into its own (subsec:Setup). With IncUnemp > 0 it therefore establishes "
            "neither a limiting solution nor a target or pseudo-target, whatever the "
            "conditions."
        )
        if not FVAC:
            basis += " The FVAC, a hypothesis of its existence theorem, fails as well."
        meanings = _MEANINGS_POSITIVE_INCOME
    elif not FVAC:
        existence = "not established"
        basis = (
            "Because the FVAC fails, the paper does not establish a limiting solution, "
            "though the WRIC holds; its existence theorem is sufficiency only."
        )
    else:
        existence = "established"
        basis = (
            "Because the WRIC and the FVAC hold, a non-degenerate limiting solution exists "
            "(thm:convgtobellman)."
        )
    established = existence == "established"
    return _Verdict(
        existence=existence,
        existence_basis=basis,
        target_guaranteed=established and holds["GICMod"],
        pseudo_target_guaranteed=established and holds["GICRaw"],
        MPCmax=1.0 - factors["WRIC"] if WRIC else 0.0,
        meanings=meanings,
    )


def _verdict_artificial(factors, holds, IncUnemp):
    """
    What the conditions imply when an artificial constraint binds. Not yet
    written: the artificial constraint is the next step.
    """
    raise NotImplementedError(
        "An artificial borrowing constraint above the natural one is not yet supported."
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
    its theorems conclude from them. Nothing in HARK calls this function.

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
    # permanent shock; with positive income in it, that event together with the
    # permanent shock at its infimum.
    if IncUnemp == 0.0:
        WorstPrb = float(UnempPrb)
    else:
        WorstPrb = float(UnempPrb) * _law_mass_at_infimum(PermShkLaw)

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

    BoroCnstNat = calc_natural_borrowing_constraint(R, G, IncUnemp, PermShkMin)
    scenario = constraint_scenario(BoroCnstArt, BoroCnstNat)
    verdict = _VERDICTS[scenario](factors, holds, float(IncUnemp))
    existence = verdict.existence

    conditions = {}
    for name, factor in factors.items():
        factor_name, _, _, _, if_holds, if_fails = _CONDITIONS[name]
        if holds[name]:
            meaning = verdict.meanings.get(name, if_holds)
        else:
            meaning = if_fails
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
    if IncUnemp > 0.0:
        notes.append(
            "With IncUnemp > 0 the worst event is the zero-income event together with the "
            "permanent shock at its infimum. Its probability, WorstPrb, is zero for a "
            "continuous law; the limiting MPC at the bottom is then one and the WRIC holds "
            "trivially. The paper states that MPC and the WRIC for zero income in the "
            "event only."
        )

    return BufferStockConditions(
        conditions=conditions,
        WorstPrb=WorstPrb,
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
# Consumption goes to zero at the natural constraint, where it curves sharply: 200
# points keep the Euler error near the bottom below half a percent with linear
# interpolation, against several percent at IndShockConsumerType's 48
BufferStockConsumerType_defaults["aXtraCount"] = 200
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
    paper's theorems guarantee them. The theorems are proved for zero income in
    the zero-income event. With ``IncUnemp`` above zero the class solves the
    model and reports the conditions, but records no limiting solution as
    established and no target or pseudo-target as guaranteed; the stable points
    it finds are listed in ``bilt`` as ``mNrmTrg_roots`` and ``mNrmStE_roots``.

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
        The probability of the zero-income event, strictly between zero and one.
    IncUnemp : float, :math:`\mu`
        Income in that event, zero in the paper; it must be at least zero and
        below the worst employed income. The paper proves its theorems for zero
        only.
    PermShkLaw, TranShkLaw : distribution
        The laws of the permanent shock and of the transitory draw. By default
        mean-one truncated lognormals built from PermShkStd, PermShk_inf,
        PermShk_sup and their transitory counterparts. To use another law,
        assign it to the attribute and call update_income_process().
    PermShkCount, TranShkCount : int
        The number of points in each law's equiprobable discretization.
    BoroCnstArt : float or None
        An artificial borrowing constraint. None (the default), or one at or
        below the natural constraint, which cannot bind; one that would bind is
        not yet supported.
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
        "IncUnemp",
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
        BoroCnstNat = calc_natural_borrowing_constraint(
            self.Rfree[0], self.PermGroFac[0], float(self.IncUnemp), PermShkMin
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

    def _restrictions_natural(self, problems):
        """
        The restrictions under the natural constraint, the paper's problem: the
        zero-income event has a probability strictly between zero and one
        (eq:TranShkDef), and income in it is strictly below the worst employed
        income, TranShkMin (1 - UnempPrb IncUnemp) / (1 - UnempPrb), so that it
        is the worst outcome.
        """
        UnempPrb, IncUnemp = float(self.UnempPrb), float(self.IncUnemp)
        if not 0.0 < UnempPrb < 1.0:
            problems.append(
                f"UnempPrb, the probability of the zero-income event, must be strictly "
                f"between zero and one (eq:TranShkDef), not {UnempPrb}."
            )
            return
        TranShkMin, _ = _law_bounds(self.TranShkLaw)
        bound = TranShkMin / (1.0 - UnempPrb * (1.0 - TranShkMin))
        if not IncUnemp < bound:
            problems.append(
                f"IncUnemp ({IncUnemp}) must be below the worst employed income, so the "
                f"zero-income event is the worst outcome: with these laws, below {bound:.6f}."
            )

    def _restrictions_artificial(self, problems):
        """
        The restrictions when an artificial constraint binds. Not yet written: the
        artificial constraint is the next step.
        """
        problems.append(
            f"BoroCnstArt ({self.BoroCnstArt}) is above the natural borrowing constraint, "
            f"so it would bind: an artificial constraint that binds is not yet supported."
        )

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

    def pre_solve(self):
        self.check_restrictions()
        self.construct("solution_terminal")
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
        are finite, under the FHWC and the RIC, and for the MPC at the bottom only
        when income in the zero-income event is zero, where the discretization's
        worst event is the law's.
        """
        results = self.bilt["BufferStockConditions"]
        if not self.PFLimitFinite or results.existence == "fails":
            return
        start = deepcopy(self.solution[0])
        start.hNrm = results.hNrm
        start.MPCmin = results.MPCmin
        if float(self.IncUnemp) == 0.0:
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

            values = gap(m_grid)
            roots = []
            for i in np.nonzero(np.sign(values[:-1]) * np.sign(values[1:]) < 0)[0]:
                roots.append(
                    brentq(
                        lambda m: float(gap(m)), m_grid[i], m_grid[i + 1], xtol=1e-12
                    )
                )
            self.bilt[key + "_roots"] = roots
            value = roots[0] if (guaranteed and len(roots) == 1) else np.nan
            self.bilt[key] = value
            setattr(self.solution[0], key, value)

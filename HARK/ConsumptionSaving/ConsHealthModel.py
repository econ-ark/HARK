"""
Classes to represent consumers who make decisions about health investment. The
first model here is adapted from White (2015).
"""

import numpy as np
from HARK.core import AgentType
from HARK.distributions import (
    expected,
    combine_indep_dstns,
    DiscreteDistribution,
    DiscreteDistributionLabeled,
)
from HARK.income.IncomeProcesses import construct_lognormal_wage_dstn
from HARK.rewards import CRRAutility, CRRAutility_inv
from HARK.interpolation import Curvilinear2DMultiInterp
from HARK.utilities import make_assets_grid

###############################################################################


# Define a function that yields health produced from investment
def eval_health_prod(n, alpha, gamma):
    return (gamma / alpha) * n**alpha


# Define a function for computing expectations over next period's (marginal) value
# from the perspective of end-of-period states, conditional on survival
def calc_exp_next(shock, a, H, R, rho, alpha, gamma, funcs):
    m_next = R * a + shock["WageRte"] * H
    h_next = (1.0 - shock["DeprRte"]) * H
    vNvrs_next, c_next, n_next = funcs(m_next, h_next)
    dvdm_next = c_next**-rho
    dvdh_next = dvdm_next / (gamma * n_next ** (alpha - 1.0))
    v_next = CRRAutility(vNvrs_next, rho=rho)
    dvda = R * dvdm_next
    dvdH = (1.0 - shock["DeprRte"]) * (shock["WageRte"] * dvdm_next + dvdh_next)
    return v_next, dvda, dvdH


###############################################################################


def solve_one_period_ConsBasicHealth(
    solution_next,
    DiscFac,
    Rfree,
    CRRA,
    HealthProdExp,
    HealthProdFac,
    DieProbMax,
    ShockDstn,
    aLvlGrid,
    HLvlGrid,
):
    """
    Solve one period of the basic health investment / consumption-saving model
    using the endogenous grid method. Policy functions are the consumption function
    cFunc and the health investment function nFunc.

    Parameters
    ----------
    solution_next : Curvilinear2DMultiInterp
        Solution to the succeeding period's problem, represented as a multi-function
        interpolant with entries vNvrsFunc, cFunc, and nFunc.
    DiscFac : float
        Intertemporal discount factor, representing beta.
    Rfree : float
        Risk-free rate of return on retained assets.
    CRRA : float
        Coefficient of relative risk aversion, representing rho. Assumed to be
        constant across periods. Should be strictly between 0 and 1.
    HealthProdExp : float
        Exponent in health production function; should be strictly b/w 0 and 1.
        This corresponds to alpha in White (2015).
    HealthProdFac : float
        Scaling factor in health production function; should be strictly positive.
        This corresponds to gamma in White (2015).
    DieProbMax : float
        Maximum death probability at the end of this period, if HLvl were exactly zero.
    ShockDstn : DiscreteDistribution
        Joint distribution of income and depreciation values that could realize
        at the start of the next period.
    aLvlGrid : np.array
        Grid of end-of-period assets (after all actions are accomplished).
    HLvlGrid : np.array
        Grid of end-of-period post-investment health.

    Returns
    -------
    solution_now : dict
        Solution to this period's problem, including policy functions cFunc and
        nFunc, as well as (marginal) value functions vFunc, dvdmFunc, and dvdhFunc.
    """
    # Make meshes of end-of-period states aLvl and HLvl
    (aLvl, HLvl) = np.meshgrid((aLvlGrid, HLvlGrid), indexing="ij")

    # Calculate expected (marginal) value conditional on survival
    v_next_exp, dvdm_next_exp, dvdh_next_exp = expected(
        func=calc_exp_next,
        dstn=ShockDstn,
        args=(aLvl, HLvl, Rfree, CRRA, HealthProdExp, HealthProdFac, solution_next),
    )

    # Calculate (marginal) survival probabilities
    LivPrb = 1.0 - DieProbMax / (1.0 + HLvl)
    MargLivPrb = -DieProbMax / (1.0 + HLvl) ** 2.0

    # Calculate end-of-period expectations
    EndOfPrd_v = DiscFac * (LivPrb * v_next_exp)
    EndOfPrd_dvda = DiscFac * (LivPrb * dvdm_next_exp)
    EndOfPrd_dvdH = DiscFac * (LivPrb * dvdh_next_exp + MargLivPrb * v_next_exp)
    vP_ratio = EndOfPrd_dvda / EndOfPrd_dvdH

    # Invert the first order conditions to find optimal controls
    cLvl = EndOfPrd_dvda ** (-1.0 / CRRA)
    nLvl = (vP_ratio / HealthProdFac) ** (-1.0 / (HealthProdExp - 1.0))

    # Invert intratemporal transitions to find endogenous gridpoints
    mLvl = aLvl + cLvl + nLvl
    hLvl = HLvl - eval_health_prod(nLvl, HealthProdExp, HealthProdFac)

    # Calculate (pseudo-inverse) value as of decision-time
    Value = CRRAutility(cLvl, rho=CRRA) + EndOfPrd_v
    vNvrs = CRRAutility_inv(Value)

    # Add points at the lower boundary of mLvl for each function
    Zeros = np.zeros((1, HLvlGrid.size))
    mLvl = np.concatenate((Zeros, mLvl), axis=0)
    hLvl = np.concatenate((hLvl[0, :], hLvl), axis=0)
    cLvl = np.concatenate((Zeros, cLvl), axis=0)
    nLvl = np.concatenate((Zeros, nLvl), axis=0)
    vNvrs = np.concatenate((Zeros, vNvrs), axis=0)

    # Construct solution as a multi-interpolation
    solution_now = Curvilinear2DMultiInterp([vNvrs, cLvl, nLvl], mLvl, hLvl)
    return solution_now


###############################################################################


def make_solution_terminal_ConsBasicHealth():
    """
    Constructor for the terminal period solution for the basic health investment
    model. The trivial solution is to consume all market resources and invest
    nothing in health. Takes no parameters because CRRA is irrelevant: pseudo-inverse
    value is returned rather than value, and the former is just cLvl = mLvl.

    The solution representation for this model is a multiple output function that
    takes market resources and health capital level as inputs and returns pseudo-
    inverse value, consumption level, and health investment level in that order.
    """
    return lambda mLvl, hLvl: (mLvl, mLvl, np.zeros_like(mLvl))


def make_health_grid(hLvlMin, hLvlMax, hLvlCount):
    """
    Make a uniform grid of health capital levels.

    Parameters
    ----------
    hLvlMin : float
        Lower bound on health capital level; should almost surely be zero.
    hLvlMax : float
        Upper bound on health capital level.
    hLvlCount : int
        Number of points in uniform health capital level grid.

    Returns
    -------
    hLvlGrid : np.array
        Uniform grid of health capital levels
    """
    return np.linspace(hLvlMin, hLvlMax, hLvlCount)


def make_uniform_depreciation_dstn(
    T_cycle, DeprRteMean, DeprRteSpread, DeprRteCount, RNG
):
    """
    Constructor for DeprRteDstn that makes uniform distributions that vary by age.

    Parameters
    ----------
    T_cycle : int
        Number of periods in the agent's sequence or cycle.
    DeprRteMean : [float]
        Age-varying list (or array) of mean depreciation rates.
    DeprRteSpread : [float]
        Age-varying list (or array) of half-widths of depreciate rate distribution.
    DeprRteCount : int
        Number of equiprobable nodes in each distribution.
    RNG : np.random.RandomState
        Agent's internal random number generator.

    Returns
    -------
    DeprRteDstn : [DiscreteDistribution]
        List of age-dependent discrete approximations to the depreciate rate distribution.
    """
    if len(DeprRteMean) != T_cycle:
        raise ValueError("DeprRteMean must have length T_cycle!")
    if len(DeprRteSpread) != T_cycle:
        raise ValueError("DeprRteSpread must have length T_cycle!")

    DeprRteDstn = []
    probs = DeprRteCount**-1.0 * np.ones(DeprRteCount)
    for t in range(T_cycle):
        bot = DeprRteMean[t] - DeprRteSpread[t]
        top = DeprRteMean[t] + DeprRteSpread[t]
        vals = np.linspace(bot, top, DeprRteCount)
        DeprRteDstn.append(
            DiscreteDistribution(
                pmv=probs,
                atoms=vals,
                seed=RNG.integers(0, 2**31 - 1),
            )
        )
    return DeprRteDstn


def combine_indep_wage_and_depr_dstns(T_cycle, WageRteDstn, DeprRteDstn, RNG):
    """
    Combine univariate distributions of wage rate realizations and depreciation
    rate realizations at each age, treating them as independent.

    Parameters
    ----------
    T_cycle : int
        Number of periods in the agent's sequence of periods (cycle).
    WageRteDstn : [DiscreteDistribution]
        Age-dependent list of wage rate realizations; should have length T_cycle.
    DeprRteDstn : [DiscreteDistribution]
        Age-dependent list of health depreciation rate realizatiosn; should have
        length T_cycle.
    RNG : np.random.RandomState
        Internal random number generator for the AgentType instance.

    Returns
    -------
    ShockDstn : [DiscreteDistribution]
        Age-dependent bivariate distribution with joint realizations of income
        and health depreciation rates.
    """
    if len(WageRteDstn) != T_cycle:
        raise ValueError(
            "IncShkDstn must be a list of distributions of length T_cycle!"
        )
    if len(DeprRteDstn) != T_cycle:
        raise ValueError(
            "DeprRteDstn must be a list of distributions of length T_cycle!"
        )
    ShockDstnX = [
        combine_indep_dstns(
            WageRteDstn[t], DeprRteDstn[t], seed=RNG.integers(0, 2**31 - 1)
        )
        for t in range(T_cycle)
    ]
    ShockDstn = DiscreteDistributionLabeled.from_unlabeled(
        dstn=ShockDstnX,
        name="wage and depreciation shock distribution",
        var_names=["WageRte", "DeprRte"],
    )
    return ShockDstn


def make_logistic_polynomial_die_prob(T_cycle, DieProbMaxCoeffs):
    """
    Constructor for DieProbMax, the age-varying list of maximum death probabilities
    (if health is zero). Builds the list as the logistic function evaluated on a
    polynomial of model age, given polynomial coefficients. Logistic function is
    applied to ensure probabilities are always between zero and one.

    Parameters
    ----------
    T_cycle : int
        Number of periods in the agent's sequence of periods (cycle).
    DieProbMaxCoeffs : np.array
        List or vector of polynomial coefficients for maximum death probability.

    Returns
    -------
    DieProbMax : [float]
        Age-varying list of maximum death probabilities (if health were zero).
    """
    age_vec = np.arange(T_cycle)
    DieProbMax = np.polyval(DieProbMaxCoeffs, age_vec).tolist()
    return DieProbMax


###############################################################################

basic_health_constructors = {
    "solution_terminal": make_solution_terminal_ConsBasicHealth,
    "WageRteDstn": construct_lognormal_wage_dstn,
    "DeprRteDstn": make_uniform_depreciation_dstn,
    "ShockDstn": combine_indep_wage_and_depr_dstns,
    "aLvlGrid": make_assets_grid,
    "hLvlGrid": make_health_grid,
    "DieProbMax": make_logistic_polynomial_die_prob,
}

basic_health_simple_params = {
    "constructors": basic_health_constructors,
    "DiscFac": 0.96,
    "Rfree": [1.03],
    "CRRA": 0.5,
    "HealthProdExp": 0.35,
    "HealthProdFac": 1.0,
    "T_cycle": 1,
}

init_basic_health = {}
init_basic_health.update(basic_health_simple_params)


class BasicHealthConsumerType(AgentType):
    """
    A class to represent consumers who can save in a risk-free asset and invest
    in the health capital via a health production function. The model is a slight
    alteration of the one from White (2015), which was in turn lifted from Ludwig
    and Schoen. In this variation, survival probability depends on post-investment
    health capital, rather than next period's health capital realization.
    """

    defaults_ = {
        "params": init_basic_health,
        "solver": solve_one_period_ConsBasicHealth,
    }

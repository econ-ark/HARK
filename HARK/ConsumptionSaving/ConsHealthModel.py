"""
Classes to represent consumers who make decisions about health investment. The
first model here is adapted from White (2015).
"""

import numpy as np
from HARK.distributions import expected
from HARK.rewards import CRRAutility, CRRAutility_inv
from HARK.interpolation import ValueFuncCRRA, Curvilinear2DInterp

###############################################################################


# Define a function to transition from end-of-period states to succeeding states
def get_states_next(shock, a, H, R):
    m_next = R * a + shock["WageRte"] * H
    h_next = (1.0 - shock["DeprRte"]) * H
    return m_next, h_next


# Define a function that yields health produced from investment
def eval_health_prod(n, alpha, gamma):
    return (gamma / alpha) * n**alpha


# Define a function for computing expectations over next period's (marginal) value
# from the perspective of end-of-period states, conditional on survival
def calc_exp_next(shock, a, H, R, vFunc, dvdmFunc, dvdhFunc):
    m_next = R * a + shock["WageRte"] * H
    h_next = (1.0 - shock["DeprRte"]) * H
    dvdm_next = dvdmFunc(m_next, h_next)
    dvdh_next = dvdhFunc(m_next, h_next)
    v_next = vFunc(m_next, h_next)
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
    solution_next : dict
        Solution to the succeeding period's problem. Has entries for vFunc, dvdmFunc,
        dvdhFunc, nFunc, and cFunc.
    DiscFac : float
        Intertemporal discount factor, representing beta.
    Rfree : float
        Risk-free rate of return on retained assets.
    CRRA : float
        Coefficient of relative risk aversion, representing rho.
    HealthProdExp : float
        Exponent in health production function; should be strictly b/w 0 and 1.
        This corresponds to alpha in White (2015).
    HealthProdFac : float
        Scaling factor in health production function; should be strictly positive.
        This corresponds to gamma in White (2015).
    DieProbMax : float
        Maximum death probability at the end of this period, if HLvl were exactly zero.
    ShockDstn : DiscreteDistribution
        Joint distribution of wage and depreciation values that could realize
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
    # Unpack next period's solution
    vFunc_next = solution_next["vFunc"]
    dvdmFunc_next = solution_next["dvdmFunc"]
    dvdhFunc_next = solution_next["dvdhFunc"]

    # Make meshes of end-of-period states aLvl and HLvl
    (aLvl_mesh, HLvl_mesh) = np.meshgrid((aLvlGrid, HLvlGrid), indexing="ij")

    # Calculate expected (marginal) value conditional on survival
    v_next_exp, dvdm_next_exp, dvdh_next_exp = expected(
        func=calc_exp_next,
        dstn=ShockDstn,
        args=(aLvl_mesh, HLvl_mesh, Rfree, vFunc_next, dvdmFunc_next, dvdhFunc_next),
    )

    # Calculate (marginal) survival probabilities
    LivPrb = 1.0 - DieProbMax / (1.0 + HLvl_mesh)
    MargLivPrb = -DieProbMax / (1.0 + HLvl_mesh) ** 2.0

    # Calculate end-of-period expectations
    EndOfPrd_v = DiscFac * (LivPrb * v_next_exp)
    EndOfPrd_dvda = DiscFac * (LivPrb * dvdm_next_exp)
    EndOfPrd_dvdH = DiscFac * (LivPrb * dvdh_next_exp + MargLivPrb * v_next_exp)
    vP_ratio = EndOfPrd_dvda / EndOfPrd_dvdH

    # Invert the first order conditions to find optimal controls
    cLvl = EndOfPrd_dvda ** (-1.0 / CRRA)
    nLvl = (vP_ratio / HealthProdFac) ** (-1.0 / (HealthProdExp - 1.0))

    # Invert intratemporal transitions to find endogenous gridpoints
    mLvl = aLvl_mesh + cLvl + nLvl
    hLvl = HLvl_mesh - eval_health_prod(nLvl, HealthProdExp, HealthProdFac)

    # Calculate (pseudo-inverse) value as of decision-time
    Value = CRRAutility(cLvl, rho=CRRA) + EndOfPrd_v
    vNvrs = CRRAutility_inv(Value)

    # Construct policy and value functions
    cFuncNow = Curvilinear2DInterp(cLvl, mLvl, hLvl)
    nFuncNow = Curvilinear2DInterp(nLvl, mLvl, hLvl)
    vNvrsFuncNow = Curvilinear2DInterp(vNvrs, mLvl, hLvl)
    vFuncNow = ValueFuncCRRA(vNvrsFuncNow, CRRA)

    # Package and return the solution
    solution_now = {
        "cFunc": cFuncNow,
        "nFunc": nFuncNow,
        "vFunc": vFuncNow,
    }
    return solution_now

"""
A file for demonstrating the "model chunks" concept on a baby consumption-saving model.
"""

import numpy as np
from HARK.chunkymodel import Chunk
from HARK.distribution import MeanOneLogNormal, expected
from HARK.utilities import construct_assets_grid
from HARK.interpolation import LinearInterp
from HARK.ConsumptionSaving.ConsIndShockModel import MargValueFuncCRRA

###############################################################################

# Define a bunch of functions that construct various objects.


def make_baby_IncShkDstn(TranShkStd, TranShkCount):
    """
    Make an extremely basic transitory shock distribution as an equiprobable
    discrete approximation to a mean one lognormal distribution.

    Parameters
    ----------
    TranShkStd : float
        Standard deviation of log transitory income.
    TranShkCount : int
        Number of equiprobable nodes.

    Returns
    -------
    TranShkDstn : DiscreteDistribution
        Transitory shock distribution.
    """
    dstn = MeanOneLogNormal(TranShkStd)
    TranShkDstn = dstn.discretize(TranShkCount, "equiprobable")
    return TranShkDstn


def make_kGrid_as_aGrid_copy(aGrid):
    """
    Specify the grid of beginning-of-period capital as simply a copy of the end-
    of-period assets grid.

    Parameters
    ----------
    aGrid : np.array
        Grid of end-of-period assets values.

    Returns
    -------
    kGrid : np.array
        Grid of beginning-of-period capital holding values.
    """
    kGrid = aGrid.copy()
    return kGrid


def make_baby_cFunc_by_EGM(aGrid, vPfunc_EOP, CRRA, DiscFac):
    """
    Construct a simple linear interpolation consumption function given the end-
    of-period marginal value function, a grid of asset values, the coefficient
    of relative risk aversion, and the intertemporal discount factor.

    Parameters
    ----------
    aGrid : np.array
        Array of end-of-period assets values.
    vPfunc_EOP : function
        Marginal value function over end-of-period assets.
    CRRA : float
        Coefficient of relative risk aversion.
    DiscFac : float
        Intertemporal discount factor.

    Returns
    -------
    cFunc : LinearInterp
        Linear interpolation of the consumption function.
    """
    vP_EOP = vPfunc_EOP(aGrid)
    cGrid = vP_EOP ** (-1.0 / CRRA)
    mGrid = aGrid + cGrid
    cFunc = LinearInterp(np.insert(mGrid, 0, 0.0), np.insert(cGrid, 0, 0.0))
    return cFunc


def make_baby_vPfunc_at_consumption_time(cFunc, CRRA):
    """
    Construct the marginal value function (w.r.t market resources) as of the moment
    the agent makes their decision about consumption. This function simply applies
    the envelope condition.

    Parameters
    ----------
    cFunc : function
        Optimal consumption as a function of market resources.
    CRRA : float
        Coefficient of relative risk aversion.

    Returns
    -------
    vPfunc_bellman : MargValueFuncCRRA
        Marginal value of market resources, as a function of market resources,
        as of the moment that the consumption-decision is made.
    """
    vPfunc_bellman = MargValueFuncCRRA(cFunc, rho=CRRA)
    return vPfunc_bellman


def make_baby_vPfunc_at_beginning(vPfunc_bellman, kGrid, TranShkDstn, Rfree, CRRA):
    """
    Construct the beginning-of-period marginal value function over capital holdings
    by taking expectations over the Bellman marginal value function, with respect
    to transitory income shocks.

    Parameters
    ----------
    vPfunc_bellman : function
        Marginal value of market resources, as a function of market resources,
        as of the moment that the consumption-decision is made.
    kGrid : np.array
        Grid of capital holding values.
    TranShkDstn : DiscreteDistribution.
        Transitory shock distribution..
    Rfree : float
        Risk-free return factor on capital holdings.
    CRRA : float
        Coefficient of relative risk aversion.

    Returns
    -------
    vPfunc_BOP : MargValueFuncCRRA
        Marginal value of capital holdings, as a function of capital holdings, as
        of the very beginning of the period, before returns or shocks are realized.
    """
    temp_f = lambda theta, k: vPfunc_bellman(Rfree * k + theta)
    vP_BOP = Rfree * expected(temp_f, TranShkDstn, args=(kGrid))
    vPnvrs_BOP = vP_BOP ** (-1.0 / CRRA)
    vPnvrsFunc_BOP = LinearInterp(kGrid, vPnvrs_BOP)
    vPfunc_BOP = MargValueFuncCRRA(vPnvrsFunc_BOP, rho=CRRA)
    return vPfunc_BOP


###############################################################################

default_baby_constructors = {
    "TranShkDstn": make_baby_IncShkDstn,
    "aGrid": construct_assets_grid,
    "kGrid": make_kGrid_as_aGrid_copy,
    "cFunc": make_baby_cFunc_by_EGM,
    "vPfunc_bellman": make_baby_vPfunc_at_consumption_time,
    "vPfunc_BOP": make_baby_vPfunc_at_beginning,
}

default_baby_parameters = {
    "CRRA": 2.5,
    "DiscFac": 0.95,
    "Rfree": 1.02,
    "TranShkStd": 0.15,
    "TranShkCount": 7,
    "aXtraMin": 0.001,
    "aXtraMax": 20,
    "aXtraNestFac": 1,
    "aXtraCount": 48,
    "aXtraExtra": [0.0],
}


class BabyConsumptionPeriod(Chunk):
    _dynamics = "Consumption-saving model with transitory income shocks, one risk free asset, and CRRA utility. Has a hard-coded liquidity constraint."
    _requirements = ["cFunc", "TransShkDstn", "Rfree"]
    _exposes = ["vPfunc"]
    _constructors = default_baby_constructors
    _parameters = default_baby_parameters

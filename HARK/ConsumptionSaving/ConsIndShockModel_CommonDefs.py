# -*- coding: utf-8 -*-
from HARK.utilities import make_grid_exp_mult
# TODO: CDC 20210524: Below is clunky.  Presumably done with an eye toward
# writing general code that would work for any utility function; but that is
# not actually done in the actual code as written.
from HARK.utilities import CRRAutility as utility
from HARK.utilities import CRRAutilityP as utilityP
from HARK.utilities import CRRAutilityPP as utilityPP
from HARK.utilities import CRRAutilityP_inv as utilityP_inv
from HARK.utilities import CRRAutility_invP as utility_invP
from HARK.utilities import CRRAutility_inv as utility_inv
from HARK.utilities import CRRAutilityP as utilityP_invP
from HARK.interpolation import (LinearInterp,
                                ValueFuncCRRA, MargValueFuncCRRA,
                                MargMargValueFuncCRRA)
import numpy as np
from copy import deepcopy


def def_utility(stge, CRRA):
    """
    Defines CRRA utility function for this period (and its derivatives,
    and their inverses), saving them as attributes of self for other methods
    to use.

    Parameters
    ----------
    stge : ConsumerSolutionOneStateCRRA

    Returns
    -------
    none
    """
    bilt = stge

#    CRRA = bilt.parameters['CRRA']

    # utility
    bilt.u = lambda c: utility(c, CRRA)
    # marginal utility
    bilt.uP = lambda c: utilityP(c, CRRA)
    # marginal marginal utility
    bilt.uPP = lambda c: utilityPP(c, CRRA)

    # Inverses thereof
    bilt.uPinv = lambda uP: utilityP_inv(uP, CRRA)
    bilt.uPinvP = lambda uP: utilityP_invP(uP, CRRA)
    bilt.uinvP = lambda uinvP: utility_invP(uinvP, CRRA)
    bilt.uinv = lambda uinv: utility_inv(uinv, CRRA)

    return stge


def def_value_funcs(stge, CRRA):
    """
    Defines the value and function and its derivatives for this period.
    See PerfForesightConsumerType.ipynb for a brief explanation
    and the links below for a fuller treatment.

    https://github.com/llorracc/SolvingMicroDSOPs/#vFuncPF

    Parameters
    ----------
    stge

    Returns
    -------
    None

    Notes
    -------
    Uses the fact that for a perfect foresight CRRA utility problem,
    if the MPC in period t is :math:`\kappa_{t}`, and relative risk
    aversion :math:`\\rho`, then the inverse value vFuncNvrs has a
    constant slope of :math:`\\kappa_{t}^{-\\rho/(1-\\rho)}` and
    vFuncNvrs has value of zero at the lower bound of market resources
    """

#    bilt = stge.bilt
    bilt = stge
#    CRRA = bilt.parameters['CRRA']

    # See PerfForesightConsumerType.ipynb docs for derivations
    vFuncNvrsSlope = bilt.MPCmin ** (-CRRA / (1.0 - CRRA))
    vFuncNvrs = LinearInterp(
        np.array([bilt.mNrmMin, bilt.mNrmMin + 1.0]),
        np.array([0.0, vFuncNvrsSlope]),
    )
    stge.vFunc = bilt.vFunc = ValueFuncCRRA(vFuncNvrs, CRRA)
    stge.vPfunc = bilt.vPfunc = MargValueFuncCRRA(bilt.cFunc, CRRA)
    stge.vPPfunc = bilt.vPPfunc = MargMargValueFuncCRRA(bilt.cFunc, CRRA)
    return stge


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

# =======================================================
# ================ Other useful functions ===============
# =======================================================


def construct_assets_grid(parameters):
    """
    Constructs the base grid of post-decision states, representing end-of-period
    assets above the absolute minimum.

    All parameters are passed as attributes of the single input parameters.  The
    input can be an instance of a ConsumerType, or a custom Parameters class.

    Parameters
    ----------
    aXtraMin:                  float
        Minimum value for the a-grid
    aXtraMax:                  float
        Maximum value for the a-grid
    aXtraCount:                 int
        Size of the a-grid
    aXtraExtra:                [float]
        Extra values for the a-grid.
    exp_nest:               int
        Level of nesting for the exponentially spaced grid

    Returns
    -------
    aXtraGrid:     np.ndarray
        Base array of values for the post-decision-state grid.
    """
    # Unpack the parameters
    aXtraMin = parameters.aXtraMin
    aXtraMax = parameters.aXtraMax
    aXtraCount = parameters.aXtraCount
    aXtraExtra = parameters.aXtraExtra
    grid_type = "exp_mult"
    exp_nest = parameters.aXtraNestFac

    # Set up post decision state grid:
    aXtraGrid = None
    if grid_type == "linear":
        aXtraGrid = np.linspace(aXtraMin, aXtraMax, aXtraCount)
    elif grid_type == "exp_mult":
        aXtraGrid = make_grid_exp_mult(
            ming=aXtraMin, maxg=aXtraMax, ng=aXtraCount, timestonest=exp_nest
        )
    else:
        raise Exception(
            "grid_type not recognized in __init__."
            + "Please ensure grid_type is 'linear' or 'exp_mult'"
        )
    # Add in additional points for the grid:
    for a in aXtraExtra:
        if a is not None:
            if a not in aXtraGrid:
                j = aXtraGrid.searchsorted(a)
                aXtraGrid = np.insert(aXtraGrid, j, a)
    return aXtraGrid

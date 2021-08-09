# -*- coding: utf-8 -*-
from HARK.utilities import make_grid_exp_mult
from HARK.interpolation import (LinearInterp, ValueFuncCRRA, MargValueFuncCRRA,
                                MargMargValueFuncCRRA)
from HARK.utilities import (  # These are used in exec so IDE doesn't see them
    CRRAutility, CRRAutilityPP, CRRAutilityP_invP,
    CRRAutility_inv, CRRAutility_invP, CRRAutilityP, CRRAutilityP_inv)
import numpy as np
from copy import deepcopy

from types import SimpleNamespace
from ast import parse as parse  # Allow storing python stmts as objects


class ValueFunctions(SimpleNamespace):
    """Equations that define value function and related Bellman objects."""

    pass


def define_transition(soln, transition_name):
    """
    Add to Modl.Transitions the transition defined by transition_name.

    Parameters
    ----------
    soln : agent_stage_solution
        A solution with an atached Modl object.

    transition_name : str
        Name of the step in Pars.transitions_possible to execute

    Returns
    -------
    None.

    """
    transitions_possible = soln.Bilt.transitions_possible
    transition = transitions_possible[transition_name]
    transiter = {}  # It's the actor that actually does the job
    transiter['compiled'] = {}
    transiter['raw_text'] = {}
    transiter['last_val'] = {}

    for eqn_name in transition:
        transiter['raw_text'].update({eqn_name: transition[eqn_name]})
        tree = parse(transition[eqn_name], mode='exec')
        compiled = compile(tree, filename="<ast>", mode='exec')
        transiter['compiled'].update({eqn_name: compiled})

    soln.Modl.Transitions[transition_name] = transiter
    return soln


def def_utility_CRRA(soln, CRRA):
    """
    Define CRRA utility function and its relatives (derivatives, inverses).

    Saves them as attributes of self for other methods to use.

    Parameters
    ----------
    soln : ConsumerSolutionOneStateCRRA

    Returns
    -------
    none
    """
    Bilt, Pars, Modl = soln.Bilt, soln.Pars, soln.Modl
    Info = Modl.Info = {**Bilt.__dict__, **Pars.__dict__}

    Modl.Rewards = SimpleNamespace()

    Modl.Rewards.raw_text = {}
    Modl.Rewards.eqns = {}
    Modl.Rewards.vals = {}

    # Add required funcs to Modl.Info
    for func in {'CRRAutility', 'CRRAutilityP', 'CRRAutilityPP',
                 'CRRAutility_inv', 'CRRAutility_invP', 'CRRAutilityP_inv',
                 'CRRAutilityP_invP'}:
        Info[func] = globals()[func]

    # Hard-wire the passed CRRA into the utility function and its progeny
    eqns_source = {
        'u_D0':
            'u = lambda c: CRRAutility(c,' + str(CRRA) + ')',
        'u_D1':
            'u.dc = lambda c: CRRAutilityP(c, ' + str(CRRA) + ' )',
        'u_D2':
            'u.dc.dc = lambda c: CRRAutilityPP(c, ' + str(CRRA) + ')',
        'uNvrs_D0':
            'u.Nvrs = lambda u: CRRAutility_inv(u, ' + str(CRRA) + ')',
        'uNvrs_D1':
            'u.Nvrs.du = lambda u: CRRAutility_invP(u, ' + str(CRRA) + ' )',
        'u_D1_Nvrs':
            'u.dc.Nvrs = lambda uP: CRRAutilityP_inv(uP, ' + str(CRRA) + ')',
        'u_D1_Nvrs_D1':
            'u.dc.Nvrs.du = lambda uP: CRRAutilityP_invP(uP, ' + str(CRRA) + ')',
    }

    # Put the utility function in the "rewards" part of the Modl object
    for eqn_name in eqns_source.keys():
        tree = parse(eqns_source[eqn_name], mode='exec')
        code = compile(tree, filename="<ast>", mode='exec')
        Modl.Rewards.eqns.update({eqn_name: code})
        exec(code, {**globals(), **Info}, Modl.Rewards.vals)

        Modl.Rewards.raw_text = eqns_source

    Bilt.__dict__.update({k: v for k, v in Modl.Rewards.vals.items()})

    return soln


def def_value_funcs(soln, CRRA):
    r"""
    Define the value and function and its derivatives for this period.

    See PerfForesightConsumerType.ipynb for a brief explanation
    and the links below for a fuller treatment.

    https://llorracc.github.io/SolvingMicroDSOPs/#vFuncPF

    Parameters
    ----------
    soln : agent_stage_solution

    Returns
    -------
    soln : agent_stage_solution, enhanced with value function

    Notes
    -----
    Call vFuncPF the value function for solution to the perfect foresight CRRA
    consumption problem in t for a consumer with no bequest motive and no
    constraints.  ('Top' because this is an upper bound for the value functions
    that would characterize consumers with constraints or uncertainty).  For
    such a problem, the MPC in period t is constant at :math:`\\kappa_{t}`, and
    calling relative risk aversion :math:`\\rho`, the inverse value function
    vFuncPFNvrs has constant slope :math:`\\kappa_{t}^{-\\rho/(1-\\rho)}` and
    vFuncPFNvrs has value of zero at the lower bound of market resources.
    """
    Bilt, Pars, Modl = soln.Bilt, soln.Pars, soln.Modl

    # Info needed to create the model objects
    Info = Modl.Info = {**Bilt.__dict__, **Pars.__dict__}
    Info['about'] = {'Info available when model creation equations executed'}

    Modl.Value = ValueFunctions()
    Modl.Value.eqns = {}  # Equations
    Modl.Value.vals = {}  # Compiled and executed result at time of exec

    eqns_source = {}  # For storing the equations
    # Pattern below: Equations needed to define value function and derivatives
    # Each eqn is preceded by adding to the scope whatever is needed
    # to make sure the variables in the equation

#    CRRA, MPCmin = Pars.CRRA, Bilt.MPCmin
    eqns_source.update(
        {'vFuncPFNvrsSlopeLim':
         'vFuncPFNvrsSlopeLim = MPCmin ** (-CRRA / (1.0 - CRRA))'})

    # vFuncPFNvrs function
#    mNrmMin = Bilt.mNrmMin
    Info['LinearInterp'] = LinearInterp  # store so it can be retrieved later

    eqns_source.update(
        {'vFuncPFNvrs':
         'vFuncPFNvrs = LinearInterp(' +
            'np.array([mNrmMin, mNrmMin + 1.0]),' +
            'np.array([0.0, vFuncPFNvrsSlopeLim]))'})

    # vFunc and its derivatives
    Info['ValueFuncCRRA'] = ValueFuncCRRA
    Info['MargValueFuncCRRA'] = MargValueFuncCRRA
    Info['MargMargValueFuncCRRA'] = MargMargValueFuncCRRA

    # Derivative 0 (undifferentiated)
    eqns_source.update(
        {'vFunc_D0':
         'vFunc = ValueFuncCRRA(vFuncPFNvrs, CRRA)'})

#    cFunc = Bilt.cFunc

    # Derivative 1
    eqns_source.update(
        {'vFunc_D1':
         'vFunc.dm = MargValueFuncCRRA(cFunc, CRRA)'})

    # Derivative 2
    eqns_source.update(
        {'vFunc_D2':
         'vFunc.dm.dm = MargMargValueFuncCRRA(cFunc, CRRA)'})

    # Store the equations in Modl.Value.eqns so they can be retrieved later
    # then execute them now
    for eqn_name in eqns_source.keys():
        #        print(eqn_name+': '+eqns_source[eqn_name])
        tree = parse(eqns_source[eqn_name], mode='exec')
        code = compile(tree, filename="<ast>", mode='exec')
        Modl.Value.eqns.update({eqn_name: code})
        exec(code, {**globals(), **Modl.Info}, Modl.Value.vals)

    # Add newly created stuff to Bilt namespace
    Bilt.__dict__.update({k: v for k, v in Modl.Value.vals.items()})

    soln.vFunc = Bilt.vFunc  # vFunc needs to be on root as well as Bilt

    Modl.Value.eqns_source = eqns_source  # Save uncompiled source code

    return soln


def_value_CRRA = def_value_funcs


def apply_flat_income_tax(
        IncShkDstn, tax_rate, T_retire, unemployed_indices=None,
        transitory_index=2):
    """
    Apply a flat income tax rate to employed income states.

    Effective only during the working period of life (those before T_retire).

    (Time runs forward in this function.)

    Parameters
    ----------
    IncShkDstn : [distribution.Distribution]
        The discrete approximation to income distribution in each time period.
    tax_rate : float
        A flat income tax rate to be applied to all employed income.
    T_retire : int
        The time index after which the agent retires.
    unemployed_indices : [int]
        Indices of transitory shocks representing unemployment states (no tax).
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
                    IncShkDstn_new[t][i][j] = \
                        IncShkDstn[t][i][j] * (1 - tax_rate)
    return IncShkDstn_new

# =======================================================
# ================ Other useful functions ===============
# =======================================================


def construct_assets_grid(parameters):
    """
    Construct base grid of post-decision states.

    Represents end-of-period assets above the absolute minimum.

    All parameters passed as attributes of the single input parameters.  The
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


def define_reward(soln, reward=def_utility_CRRA):
    """Bellman reward."""
    soln = reward(soln, soln.Pars.CRRA)
    return soln


def define_t_reward(soln, reward):
    """Bellman reward."""
    soln = reward(soln, soln.Pars.CRRA)
    return soln

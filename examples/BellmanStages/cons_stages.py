from dataclasses import dataclass, field

import itertools
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from typing import Any, Callable, Mapping, Sequence
import xarray as xr

from HARK import distribution
from HARK.rewards import CRRAutility, CRRAutilityP, CRRAutility_inv, CRRAutilityP_inv

from HARK.stage import Stage, backwards_induction, simulate_stage


epsilon = 1e-4


### CONSUMPTION STAGE

"""
HARK's `CRRAutility` does not work with an input of 0. We will patch it to fix this.
"""

CRRAutility_hack = lambda c, gam: float('-inf') if c == 0.0 else CRRAutility(c, gam)
CRRAutilityP_hack = lambda c, gam: float('inf') if c == 0.0 else CRRAutilityP(c, gam)


CRRA = 2.0

consumption_stage = Stage(
    ##### Stage Definition -- math!
    transition = lambda x, k, a : {'a' : x['m'] - a['c']},
    transition_der_a = -1,
    transition_der_x = 1,
    transition_inv = lambda y, a : {'m' : y['a'] + a['c']},
    reward = lambda x, k, a : CRRAutility_hack(a['c'], CRRA),
    reward_der = lambda x, k, a: CRRAutilityP_hack(a['c'], CRRA),
    reward_der_inv = lambda uP : (CRRAutilityP_inv(uP, CRRA),),
    inputs = ['m'], 
    actions = ['c'],
    outputs = ['a'],
    action_upper_bound = lambda x, k: (x['m'],) , 
    action_lower_bound = lambda x, k: (0.0,) , 
    discount = .96, # Here, this is just a subjective discount rate.
    
    ##### Inputs to optimizers, interpolators, solvers...
    optimizer_args = {
        'method' : 'Nelder-Mead',
        'options' : {
            'maxiter': 1e3,
            #'disp' : True
        }
    },
    
    v_transform = np.exp, # lambda c: CRRAutility_inv(c, CRRA),
    v_transform_inv = np.log, # lambda c : CRRAutility_inv(c, CRRA),
    v_der_transform = lambda u: u ** -1 , #  CRRAutilityP_inv
    v_der_transform_inv = lambda u : u ** -1, # u ** - rho
    
    # Pre-computed points for the solution to this stage
    solution_points = xr.Dataset(
        data_vars={
            'v_x' : (["m"], [-float('inf')]),
            'v_x_der' : (["m"], [float('inf')]),
            'pi*' : (["m"], [0]),
            #'q' : (["m"], [-float('inf')]),
        },
        coords={
            'm' : (["m"], [0]),
        }
    )
)


### ALLOCATION STAGE

allocation_stage = Stage(
    transition = lambda x, k, a : {'a' : x['a'], 'alpha' : a['alpha']}, 
    inputs = ['a'], 
    actions = ['alpha'],
    outputs = ['a', 'alpha'],
    # Using bounds instead of constraints will result in a different optimizer
    action_upper_bound = lambda x, k: (1,) , 
    action_lower_bound = lambda x, k: (0,) ,

    v_transform = np.exp, # lambda c: CRRAutility_inv(c, CRRA),
    v_transform_inv = np.log, # lambda c : CRRAutility_inv(c, CRRA),
    v_der_transform = lambda u: u ** -1 , #  CRRAutilityP_inv
    v_der_transform_inv = lambda u : u ** -1, # u ** - rho

    optimizer_args = {
        'method' : "trust-constr",
        'options' : {"gtol": 1e-12, "disp": False, "maxiter" : 1e10}
    },
)



### INCOME STAGE

R = 1.03
G = 1.01

sigma_psi = 0.1
sigma_theta = 0.1
sigma_eta = 1.1
p_live = 0.98

def income_transition(x, k, a): 
    return {'m' : ((x['alpha'] * k['eta'] + (1 - x['alpha']) * R) 
                   * x['a'] + k['theta']) 
            / (k['psi'] * G)}

income_stage = Stage(
    transition = income_transition,
    inputs = ['a', 'alpha'],
    discount = lambda x, k, a: p_live * k['psi'] ** (1 - CRRA), 
    shocks = {
        'psi' : distribution.Lognormal(0, sigma_psi),
        'theta' : distribution.Lognormal(0, sigma_theta),
        'eta' : distribution.Lognormal(0, sigma_eta),
        # 'live' : distribution.Bernoulli(p_live) ## Not implemented for now
    },
    outputs = ['m'],

    v_transform = np.exp, # lambda c: CRRAutility_inv(c, CRRA),
    v_transform_inv = np.log, # lambda c : CRRAutility_inv(c, CRRA),
    v_der_transform = lambda u: u ** -1 , #  CRRAutilityP_inv
    v_der_transform_inv = lambda u : u ** -1, # u ** - rho
)


### LABOR AND INVESTING STAGES
### These are the INCOME stage, broken into two, more efficient stages.

def investing_transition(x, k, a): 
    return {'b' : ((x['alpha'] * k['eta'] + (1 - x['alpha']) * R)) * x['a']}

investing_stage = Stage(
    transition = investing_transition,
    inputs = ['a', 'alpha'],
    discount = p_live,
    reward_der = lambda x, k, a : 0,
    shocks = {
        'eta' : distribution.Lognormal(0, sigma_eta),
    },
    outputs = ['b'],

    v_transform = np.exp, # lambda c: CRRAutility_inv(c, CRRA),
    v_transform_inv = np.log, # lambda c : CRRAutility_inv(c, CRRA),
    v_der_transform = lambda u: u ** -1 , #  CRRAutilityP_inv
    v_der_transform_inv = lambda u : u ** -1, # u ** - rho
)

def labor_transition(x, k, a):
    return  {'m' : (x['b'] + k['theta']) / ( k['psi'] * G)}

labor_stage = Stage(
    transition = labor_transition,
    transition_der_x = lambda x, k, a: 1 / (k["psi"] * G),
    inputs = ['b'],
    shocks = {
        'theta' : distribution.Lognormal(0, sigma_theta),
        'psi' : distribution.Lognormal(0, sigma_psi),
        # 'live' : distribution.Bernoulli(p_live) ## Not implemented for now
    },
    outputs = ['m'],
    discount = lambda x, k, a: (G * k['psi']) ** (1 - CRRA), # Check with CDC about G use here. CDC: ^(1 - rho); Seb: ^(rho - 1)
    reward_der = lambda x, k, a : 0,

    v_transform = np.exp, # lambda c: CRRAutility_inv(c, CRRA),
    v_transform_inv = np.log, # lambda c : CRRAutility_inv(c, CRRA),
    v_der_transform = lambda u: u ** (- 1 / CRRA) , # lambda u : u ** -1, #   CRRAutilityP_inv
    v_der_transform_inv = lambda u : u ** (- CRRA), # lambda u: u ** -1 #  u ** - rho
)

def rfree_transition(x, k, a): 
    return {'b' : R * x['a']}

rfree_stage = Stage(
    transition = rfree_transition,
    transition_der_x = lambda x, k, a: R,
    inputs = ['a'],
    discount = p_live,
    reward_der = lambda x, k, a : 0,

    outputs = ['b'],

    v_transform = np.exp, # lambda c: CRRAutility_inv(c, CRRA),
    v_transform_inv = np.log, # lambda c : CRRAutility_inv(c, CRRA),
    v_der_transform = lambda u: u ** -1 , #  CRRAutilityP_inv
    v_der_transform_inv = lambda u : u ** -1, # u ** - rho
)
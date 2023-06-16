from dataclasses import field
import itertools
import numpy as np
from scipy.optimize import minimize, brentq
from typing import Callable, Mapping, Sequence, Tuple
import xarray as xr

def analytic_pi_y_star(
        y,
        v_y_der : Callable[[Mapping, Mapping, Mapping], float],
        dr_da_inv : Callable[[float], float],
        dg_da = -1.0,
        discount = 1.0
        ):
    """
    The optimal action that results in output values y.

    Assumes:
     - dg_da is a constant
     - discount factor is constant

    Params
    ------
    """
    if dg_da is None or not(
        isinstance(dg_da, float) or isinstance(dg_da, int)
        ):
        raise Exception(f"No constant transition derivative found. transition_der_a is {dg_da}")

    if not (isinstance(discount, float) or isinstance(discount, int)):
        raise Exception("Analytic pi_star_y requires constant discount factor (rendering B' = 0).")

    v_y_der_at_y = v_y_der(y)
        
    if isinstance(v_y_der_at_y, xr.DataArray):
        v_y_der_at_y = v_y_der_at_y.values # np.atleast1darray() ?

    if 0 > v_y_der_at_y:
        raise Exception(f"Negative marginal value {v_y_der_at_y} computes at y value of {y}. Reward is {- discount * dg_da * v_y_der_at_y}")

    return dr_da_inv(- discount * dg_da * v_y_der_at_y)

def egm(
        inputs,
        actions,
        g_inv : Callable[[Mapping, Mapping, Mapping], float],
        dr_da_inv, # = lambda uP : (CRRAutilityP_inv(uP, rho),),
        dg_da = -1,
        y_grid : Mapping[str, Sequence] = {}, ## TODO: Better data structure here.
        v_y_der : Callable[[Mapping, Mapping, Mapping], float] = lambda x : 0,
        discount = 1,
    ):
    """
    Given a grid over output
    and marginal output value function,
    compute the optimal action.

    This depends on the stage having an
    *inverse marginal reward function*
    and *inverse transition function*.

    Does not use rootfinding!
    The 'grid' over the input 

    ### ASSUMES: No discounting this phase,
    ###           and... T' = -1 ???
    """

    ## can be functionalized out once we settle
    ## on grid-to-DataArray conversions
    pi_y_data = xr.DataArray(
        np.zeros([len(v) for v in y_grid.values()]),
        dims = y_grid.keys(),
        coords = y_grid
    )

    # Collecting data for the real optimal policy with respect to inputs
    x_val_data = []
    a_val_data = []

    ## duplicated from foc.py; move to a shared helper library?
    def action_zip(a : Tuple):
        """
        Wraps a tuple of values for an action in a dictionary with labels.
        Useful for converting between forms of model equations.

        References 'actions' argument of optimal_policy_foc()
        """
        return {an : av for an,av in zip(actions, a)}

    # can I pass in the grid, rather than iterate?
    for y_point in itertools.product(*y_grid.values()):
        y_vals = {k : v for k, v in zip(y_grid.keys() , y_point)}

        acts = analytic_pi_y_star(
            y_vals,
            v_y_der,
            dr_da_inv,
            dg_da,
            discount
            )

        pi_y_data.sel(**y_vals).variable.data.put(0, acts)

        x_vals = g_inv(y_vals, action_zip(acts))

        x_val_data.append(x_vals)
        a_val_data.append(action_zip(acts))

    ## TODO is this dealing with repeated values?
    x_coords = {
        x : np.array([xv[x] for xv in x_val_data])
        for x
        in inputs
    }

    pi_data = xr.DataArray(
        np.zeros([len(v) for v in x_coords.values()]),
        dims = x_coords.keys(),
        coords = x_coords
    )

    for i, x_vals in enumerate(x_val_data):
        x_vals = x_val_data[i]
        a_vals = a_val_data[i]
        acts = [a_vals[a] for a in a_vals]
  
        pi_data.sel(**x_vals).variable.data.put(0, acts)

    return pi_data, pi_y_data

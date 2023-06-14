from dataclasses import field
import itertools
import numpy as np
from scipy.optimize import minimize, brentq
from typing import Callable, Mapping, Sequence, Tuple
import xarray as xr

"""
Sargent + Stachurski:
x - states
z - shocks
a - actions

New:
y - post-states

French notation:
g - transtion function x,z, a -> y



Question:
    - Do we want to have a canonical 'grid' object that includes variable names and allows for non-cartesian grids?

"""

## TODO: Action handling

def xndindex(ds, dims=None):
    """
    There is currently no integrated way to iterate over an xarray.DataArray with its coordinate labels.

    This method is a workaround from:
    https://github.com/pydata/xarray/issues/2805#issuecomment-1255029201
    """
    if dims is None:
        dims = ds.dims
    elif type(dims) is str:
        dims=[dims]
    else:
        pass
        
    for d in dims:
        if d not in ds.dims:
            raise ValueError("Invalid dimension '{}'. Available dimensions {}".format(d, ds.dims))
            
    iter_dict = {k:v for k,v in ds.sizes.items() if k in dims}
    for d,k in zip(repeat(tuple(iter_dict.keys())),zip(np.ndindex(tuple(iter_dict.values())))):
        yield {k:l for k,l in zip(d,k[0])}


def xz_grids_to_data_array(
        x_grid : Mapping[str, Sequence] = {}, ## TODO: Better data structure here.
        z_grid : Mapping[str, Sequence] = {}
    ):

    coords = {**x_grid, **z_grid}

    da = xr.DataArray(
        np.zeros([len(v) for v in coords.values()]),
        dims = coords.keys(),
        coords = coords
    )

    return da


def optimal_policy_foc(
        g : Callable[[Mapping, Mapping, Mapping], float],  #= lambda x, z, a : {'a' : x['m'] - a['c']},
        actions,
        r, # = lambda x, z, a : u(a['c']),
        dr_da, # = lambda x, z, a: u.prime(a['c']),
        dr_inv, # = lambda uP : (CRRAutilityP_inv(uP, rho),),
        dg_dx = 1,  ## Used in FOC method, step 5
        dg_da = -1,  ## Used in FOC method, step 5
        x_grid : Mapping[str, Sequence] = {}, ## TODO: Better data structure here.
        z_grid : Mapping[str, Sequence] = {},
        v_y_der : Callable[[Mapping, Mapping, Mapping], float] = lambda x : 0,
        discount = 1,
        action_upper_bound : Callable[[Mapping, Mapping], Sequence[float]] = field(default = None), # = lambda x, z: (x['m'] + gamma[0] * theta.X[0] / R,),
        action_lower_bound : Callable[[Mapping, Mapping], Sequence[float]] = field(default = None),
        optimizer_args = None # TODO: For brettq.
    ):
    """
    Given a grid over input and shock state values,
    and marginal output value function,
    compute the optimal action.

    Uses root finding and the first order condition.

    This is written with the expectation that:
        - Actions are scalar
        - the Q function is concave over the action space.

    Functionality is not guaranteed otherwise.


    NOTE: This does not put pre-defined solution points into the solution data.
          That needs to be added in a different step.


    TODO: K -> Z
    TODO: Transition function as an argument

    Parameters
    -----------

    x_grid : Mapping[str, Sequence]
        A mapping of state variable names to a sequence of grid values
    
    z_grid: Mapping[str, Sequence]
        A mapping of shock variable names to grid values


    """
    # Set up data arrays with coordinates based on the grid.
    pi_data = xz_grids_to_data_array(x_grid, z_grid) ## TODO: Default value for pi should be nan.
    q_der_data = xz_grids_to_data_array(x_grid, z_grid)
    ## May need to expand this for multivariate y
    y_data = xz_grids_to_data_array(x_grid, z_grid)

    xz_grid_size = np.prod([len(xv) for xv in x_grid.values()]) * np.prod([len(zv) for zv in z_grid.values()])

    def dq_da(x,z,a, v_y_der):
        print(dr_da, x, z, a, discount, v_y_der)
        return dr_da(x,z,a) + discount * v_y_der(g(x,z,a)) * dg_da # could optionally be dg_da(x,z,a)

    # TODO: replace these with iterators....
    #xz_iterator = xndindex(pi_data)
    #import pdb; pdb.set_trace()

    def action_zip(a : Tuple):
        """
        Wraps a tuple of values for an action in a dictionary with labels.
        Useful for converting between forms of model equations.

        References 'actions' argument of optimal_policy_foc()
        """
        return {an : av for an,av in zip(actions, a)}

    for x_point in itertools.product(*x_grid.values()):
        x_vals = {k : v for k, v in zip(x_grid.keys() , x_point)}

        for z_point in itertools.product(*z_grid.values()):
            z_vals = {k : v for k, v in zip(z_grid.keys() , z_point)}

            # repeated code with the other optimizer -- can be functionalized out?
            """
            if len(self.actions) == 0:
                q_der_xz = self.d_q_d_a(
                    x_vals,
                    z_vals,
                    {}, # no actions
                    v_y_der
                    )

                # this should just be the default value for pi
                pi_data.sel(**x_vals, **z_vals).variable.data.put(0, np.nan)

                # then this can be looped in differently.
                q_der_data.sel(**x_vals, **z_vals).variable.data.put(0, q_der_xz)

                ## TODO: define g
                y = g(x_vals, z_vals, self.action_zip([np.nan]))
                y_n = np.array([y[k] for k in y])
                y_data.sel(**x_vals, **z_vals).variable.data.put(0, y_n)
            """
            
            def foc(a):
                a_vals = action_zip((a,))
                return dq_da(x_vals, z_vals, a_vals, v_y_der)

            # these lower bounds as arugments to the 
            lower_bound = action_lower_bound(x_vals, z_vals)
            upper_bound = action_upper_bound(x_vals, z_vals)

            ##  what if no lower bound?
            q_der_lower = None
            if lower_bound[0] is not None:
                q_der_lower = dq_da(
                    x_vals,
                    z_vals,
                    action_zip(lower_bound),
                    v_y_der
                    )
            else:
                lower_bound[0] = 1e-12 ## a really high number!

            q_der_upper = None
            if upper_bound[0] is not None:
                q_der_upper = dq_da(
                    x_vals,
                    z_vals,
                    action_zip(upper_bound),
                    v_y_der
                    )
            else:
                upper_bound[0] =  1e12 ## a really high number!

            ## TODO: Better handling of case when there is a missing bound?
            if q_der_lower is not None and q_der_upper is not None and q_der_lower < 0 and q_der_upper < 0:
                a0 = lower_bound

                pi_data.sel(**x_vals, **z_vals).variable.data.put(0, a0)
                q_der_data.sel(**x_vals, **z_vals).variable.data.put(0, q_der_lower)
            elif q_der_lower is not None and q_der_upper is not None and q_der_lower > 0 and q_der_upper > 0:
                a0 = upper_bound

                pi_data.sel(**x_vals, **z_vals).variable.data.put(0, a0)
                q_der_data.sel(**x_vals, **z_vals).variable.data.put(0, q_der_upper)
            else:
                ## Better exception handling here
                ## asserting that Q is concave
                if q_der_lower is not None and q_der_upper is not None and not(q_der_lower > 0 and q_der_upper < 0):
                    raise Exception("Cannot solve for optimal policy with FOC if Q is not concave!")

                a0, root_res = brentq(
                    foc,
                    lower_bound[0], # only works with scalar actions
                    upper_bound[0], # only works with scalar actions
                    full_output = True
                )

                if root_res.converged:
                    pi_data.sel(**x_vals, **z_vals).variable.data.put(0, a0)

                    q_der_xz = dq_da(
                        x_vals,
                        z_vals,
                        action_zip((a0,)), # actions are scalar
                        v_y_der
                    )

                    q_der_data.sel(**x_vals, **z_vals).variable.data.put(0, q_der_xz)
                else:
                    print(f"Rootfinding failure at {x_vals}, {z_vals}.")
                    print(root_res)

                    #raise Exception("Failed to optimize.")
                    pi_data.sel(**x_vals, **z_vals).variable.data.put(0,  root_res.root)

                    q_der_xz = dq_da(
                        x_vals,
                        z_vals,
                        action_zip((root_res.root,)), # actions are scalar
                        v_y_der
                    )

                    q_der_data.sel(**x_vals, **z_vals).variable.data.put(0, q_der_xz)

                acts =  np.atleast_1d(pi_data.sel(**x_vals, **z_vals).values)
                y = g(x_vals, z_vals, self.action_zip(acts))
                y_n = np.array([y[k] for k in y])
                y_data.sel(**x_vals, **z_vals).variable.data.put(0, y_n)

    return pi_data, q_der_data, y_data
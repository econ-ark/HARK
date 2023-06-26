"""
This algorithm solves for an agent's policy given:

- a state space :math:`X`
- a shock space :math:`Z`
- an action space :math:`A`
- a post-state space :math:`Y`
- a deterministic transition function :math:`g`

The agent chooses their action after shocks have been realized:

.. math:: g: X \\times Z \\times A \\rightarrow Y

- a reward function :math:`r`
- a post-value function :math:`v_y`
- constraints on the action :math:`\Gamma`
- a scalar discount factor :math:`\\beta`

The problem it solves is of the form:

.. math:: \pi^*(x, z) = \\textrm{argmax}_{a \in \Gamma(x, z)} r(x, z, a) + \\beta v_y(g(x, z, a))]

It solves this on each point on grids over :math:`\\bar{X}` and :math:`\\bar{Z}`,

It does this using the first order condition (FOC).

Given a state-action value function:

.. math::

    q(x, z, a) = r(x, z, a) + \\beta v_y(g(x, z, a))

And the derivative of this with respect to actions:

.. math::

    \\frac{\partial q}{\partial a}(x, z, a) = \\frac{\partial r}{\partial a}(x, z, a) + \\beta \\frac{\partial v_y}{\partial y}(g(x, z, a) \\frac{\partial g}{\partial a}(x, z, a))

The first order condition is that

.. math::

    0 = \\frac{\partial q}{\partial a}(x, z, a)
"""

from dataclasses import field
import itertools
import numpy as np
from scipy.optimize import minimize, brentq
from typing import Callable, Mapping, Sequence, Tuple
import xarray as xr

## TODO: Action handling

def xndindex(ds, dims=None):
    """
    This function returns an iterator over the values in a DataArray or DataSet's coordinates.

    There is currently no integrated way to iterate over an xarray.DataArray with its coordinate labels.

    This method is a workaround from:
    https://github.com/pydata/xarray/issues/2805#issuecomment-1255029201

    Parameters
    -------

    ds: xarray.DataArray or xarray.DataSet
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
    for d,k in zip(itertools.repeat(tuple(iter_dict.keys())),zip(np.ndindex(tuple(iter_dict.values())))):
        yield {k:l for k,l in zip(d,k[0])}


Grid = Mapping[str, Sequence]

def xz_grids_to_data_array(
        x_grid : Grid = {}, ## TODO: Better data structure here.
        z_grid : Grid = {}
    ):
    """
    Construct a zero-valued DataArray with the coordinates
    based on the two Grids passed in.

    Parameters
    ----------
    x_grid: Grid
        A mapping from state variable labels to a sequence of numerical values.
    z_grid: Grid
        A mapping from shock variable labels to a sequence of numerical values.

    Returns
    --------

    da xarray.DataArray
        An xarray.DataArray with coordinates given by both grids.
    """

    coords = {**x_grid, **z_grid}

    da = xr.DataArray(
        np.empty([len(v) for v in coords.values()]),
        dims = coords.keys(),
        coords = coords
    )

    return da


def optimal_policy_foc(
        g : Callable[[Grid, Grid, Grid], float], 
        actions: Sequence[str],
        r : Callable[[Grid, Grid, Grid], float], 
        dr_da, 
        dr_inv,
        dg_dx = 1, 
        dg_da = -1, 
        x_grid : Grid = {},
        z_grid : Grid = {},
        v_y_der : Callable[[Grid], float] = lambda y : 0,
        discount = 1,
        action_upper_bound : Callable[[Grid, Grid], Sequence[float]] = field(default = None),
        action_lower_bound : Callable[[Grid, Grid], Sequence[float]] = field(default = None),
    ):
    """
    Given a grid over input and shock state values,
    and marginal output value function,
    compute the optimal action.

    Uses root finding and the first order condition.

    This is written with the expectation that actions are scalar
    and the q function is concave over the action space.

    The functionality is not guaranteed otherwise.

    Parameters
    -----------
    g:
        Transition functiong g.
    actions:
        Ordered labels of action variables.
    r:
        Reward function r.
    dr_da:
        Derivative of reward function r with respect to actions A.
    dr_inv: 
        Inverse of derivative of reward function
    dg_dx:
        Derivative of transition function g with respect to states X.
    dg_da:
        Derivative of transition function g with respect to actions A.
    x_grid:
        Grid of points x in state space X. A mapping between variable labels
        and sequences of values
    z_grid:
        Grid of points z in state space Z. A mapping between variable labels
        and sequences of values
    v_y_der:
        Derivative of the post-value function v_y with respect to post-states Y.
    discount:
        A scalar discount factor.
    action_upper_bound:
        Function of x and z, giving upper bound on action a.
    action_lower_bound:
        Function of x and z, giving lower bound on action a.

    Returns
    --------

    pi_star_data:
        Data for the optimal policy :math:`\pi^*`.
    q_der_data:
        Data for the derivative of the action value function function :math:`\partial q/\partial a`.
    """
    # Set up data arrays with coordinates based on the grid.
    pi_star_data = xz_grids_to_data_array(x_grid, z_grid) ## TODO: Default value for pi should be nan.
    q_der_data = xz_grids_to_data_array(x_grid, z_grid)

    def dq_da(x,z,a, v_y_der):
        """
        Derivative of the action-value function q with respect to actions a
        """
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
           
            def foc(a):
                a_vals = action_zip((a,))
                return dq_da(x_vals, z_vals, a_vals, v_y_der)

            ##  what if no lower bound?
            q_der_lower = None
            if action_lower_bound is not None:
                lower_bound = action_lower_bound(x_vals, z_vals)
                q_der_lower = dq_da(
                    x_vals,
                    z_vals,
                    action_zip(lower_bound),
                    v_y_der
                    )
            else:
                lower_bound = np.array([-1e-12]) ## a really low number!

            q_der_upper = None
            if action_upper_bound is not None:
                upper_bound = action_upper_bound(x_vals, z_vals)
                q_der_upper = dq_da(
                    x_vals,
                    z_vals,
                    action_zip(upper_bound),
                    v_y_der
                    )
            else:
                upper_bound =  np.array([1e12]) ## a really high number!
     
            if q_der_lower is not None and q_der_upper is not None and not(q_der_lower > 0 and q_der_upper < 0):
                raise Exception("Cannot solve for optimal policy with FOC if Q is not concave!")

            a0, root_res = brentq(
                foc,
                lower_bound[0], # only works with scalar actions
                upper_bound[0], # only works with scalar actions
                full_output = True
            )

            if root_res.converged:
                pi_star_data.sel(**x_vals, **z_vals).variable.data.put(0, a0)

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

                pi_star_data.sel(**x_vals, **z_vals).variable.data.put(0,  root_res.root)

                q_der_xz = dq_da(
                    x_vals,
                    z_vals,
                    action_zip((root_res.root,)), # actions are scalar
                    v_y_der
                )

                q_der_data.sel(**x_vals, **z_vals).variable.data.put(0, q_der_xz)

    return pi_star_data, q_der_data
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

However, this algorithm uses the Endogenous Gridpoints Method (EGM) [1]_,
which solves this problem only indirectly.

It starts with a grid over post-states :math:`\\bar{Y}`.

For each value of :math:`\hat{y}` in the grid, it analytically determines
the action which, when chosen as an optimal solution to the problem,
results in that post-state.

.. math::

    \pi_y(y) = \\frac{\partial r}{\partial a}^{-1}(\\beta \\frac{\partial g}{\partial a} \\frac{\partial v_y}{\partial y}(\hat{y})

It then computes the state that corresponds to that action using
the inverse transition function:

.. math::

    \hat{x} = g^{-1}(\hat{y}, \pi_y(\hat{y}))

The pairs :math:`(\hat{x}, \pi_y(\hat{y}))` then are data points for
the grid for the optimal policy :math:`\pi^*`.
    
.. [1] Carroll, C. D. (2006). The method of endogenous gridpoints for solving dynamic stochastic
   optimization problems. Economics letters, 91(3), 312-320.

"""


import itertools
import numpy as np
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
    The action which, when chosen, results in the post-state value Y.

    Parameters
    -----------
    y:
        The post-state value Y.
        TODO: type signature of input y

    v_y_der:
        Derivative post-value function over post-states Y.

    dr_da_inv:
        Derivative of the reward function r with respect to actions A, inverted.

    dg_da:
        Derivateive of the transition function g with respect to actions A. Must be a constant.

    discount:
        Discount factor. Must be a constant.

    Returns
    -------

    The actions chosen that result in post-state Y.
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
        inputs : Sequence[str],
        actions : Sequence[str],
        g_inv : Callable[[Mapping, Mapping, Mapping], float],
        dr_da_inv, # = lambda uP : (CRRAutilityP_inv(uP, rho),),
        dg_da = -1,
        y_grid : Mapping[str, Sequence] = {}, ## TODO: Better data structure here.
        v_y_der : Callable[[Mapping, Mapping, Mapping], float] = lambda x : 0,
        discount = 1,
    ):
    """
    Given post-states Y and information about an agent's problem,
    compute the actions which, if chosen, result in those post-states,
    and the corresponding states X.
    
    This is a method of computing data about the optimal policy of an
    agent that does not use rootfinding.

    Parameters
    ------------
    inputs:
        Ordered labels of the state variables.

    actions:
        Ordered labels of the action variables.

    g_inv:
        Inverse of the transition function g.

    dr_da_inv:
        Derivative of the reward function r with respect to actions A, inverted.

    dg_da:
        Derivative of the transition function g with respect to action A.

    y_grid:

    v_y_der:
        Derivative of post-value function v_y over post states Y

    discount:
        A discount fact. Scalar.

    Returns
    --------

    pi_data:

    pi_y_data:
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

"""
Use backwards induction to derive the arrival value function
from a continuation value function and stage dynamics.
"""

from HARK.model import DBlock
import itertools
from scipy.optimize import minimize, brentq
from typing import Mapping


def get_action_rule(action):
    """
    Produce a function from any inputs to a given value.
    This is useful for constructing decision rules with fixed actions.
    """

    def ar(*args):
        return action

    return ar


def ar_from_data(da):
    """
    Produce a function from any inputs to a given value.
    This is useful for constructing decision rules with fixed actions.
    """

    def ar(**args):
        return da.interp(**args)

    return ar


Grid = Mapping[str, Sequence]


def grid_to_data_array(
    grid: Grid = {},  ## TODO: Better data structure here.
):
    """
    Construct a zero-valued DataArray with the coordinates
    based on the Grid passed in.

    Parameters
    ----------
    grid: Grid
        A mapping from variable labels to a sequence of numerical values.

    Returns
    --------
    da xarray.DataArray
        An xarray.DataArray with coordinates given by both grids.
    """

    coords = {**grid}

    da = xr.DataArray(
        np.empty([len(v) for v in coords.values()]), dims=coords.keys(), coords=coords
    )

    return da


def vbi_solve(
    block: DBlock, continuation, state_grid: Grid, disc_params, calibration={}
):
    """
    Solve a DBlock using backwards induction on the value function.

    Parameters
    -----------
    block
    continuation

    state_grid: Grid
        This is a grid over all variables that the optimization will range over.
        This should be just the information set of the decision variables.

    disc_params
    calibration
    """

    # state-rule value function
    srv_function = block.get_state_rule_value_function_from_continuation(continuation)

    controls = block.get_controls()

    # pseudo
    policy_data = grid_to_data_array(state_grid)
    value_data = grid_to_data_array(state_grid)

    # loop through every point in the state grid
    for state_point in itertools.product(*state_grid.values()):
        # build a dictionary from these states, as scope for the optimization
        state_vals = {k: v for k, v in zip(state_grid.keys(), state_point)}

        # copy calibration
        # update with state_vals
        # this becomes the before-action states
        pre_states = state_vals + parameters

        # prepare function to optimize
        def negated_value(a):  # old! (should be negative)
            dr = {c: get_action_rule(a[i]) for i, c in enumerate(controls)}

            # negative, for minimization later
            return -srv_function(pre_states, dr)

        ## get lower bound.
        ## not yet implemented
        lower_bound = np.array([-1e-12] * len(controls))  ## a really low number!

        ## get upper bound
        ## not yet implemented
        upper_bound = np.array([1e12] * len(controls))

        # pseudo
        # optimize_action(pre_states, srv_function)

        a_best, root_res = minimize(  # choice of
            negated_value, full_output=True
        )

        dr_best = {c: get_action_rule(a_best[i]) for i, c in enumerate(controls)}

        if root_res.converged:
            policy_data.sel(**state_vals).variable.data.put(0, a_best)
            value_data.sel(**state_vals).variable.data.put(
                0, srv_function(pre_states, dr_best)
            )
        else:
            print(f"Optimization failure at {state_vals}.")
            print(root_res)

            dr_best = {c: get_action_rule(root_res[i]) for i, c in enumerate(controls)}

            policy_data.sel(**state_vals).variable.data.put(0, res.root)  # ?
            value_data.sel(**state_vals).variable.data.put(
                0, srv_function(pre_states, dr_best)
            )

    # use the xarray interpolator to create a decision rule.
    dr_from_data = {
        c: ar_from_data(da)  # maybe needs to be more sensitive to the information set
        for i, c in enumerate(controls)
    }

    dec_vf = block.get_decision_value_function(dr_from_data, continuation)
    arr_vf = block.get_arrival_value_function(disc_params, dr_from_data, continuation)

    return dr_from_data, dec_vf, arr_vf
from dataclasses import dataclass, field
import datetime
from typing import Any, Callable, Mapping, Sequence
import itertools
import numpy as np
from scipy.optimize import minimize
import xarray as xr

from HARK.distribution import Distribution

class SolutionDataset(object):
    def __init__(self, dataset: xr.Dataset, actions = {}):
        self.actions = actions
        self.dataset = dataset

    def __repr__(self):
        return self.dataset.__repr__()
        
    ## TODO: Add in assume sorted to make it faster
    def v_x(self, x : Mapping[str, Any]) -> float:
        return self.dataset['v_x'].interp(**x, kwargs={"fill_value": None}) # Python 3.8 None -> 'extrapolate'

    def pi_star(self, x : Mapping[str, Any], k : Mapping[str, Any]):
        """

        TODO: Option to return a labelled map...
        """
        return self.dataset['pi*'].interp({**x, **k}, kwargs={"fill_value": None}) # Python 3.8 None -> 'extrapolate'
    
    def q(self, x : Mapping[str, Any], k : Mapping[str, Any], a : Mapping[str, Any]) -> float:
        return self.dataset['q'].interp({**x, **k, **a}, kwargs={"fill_value": None}) # Python 3.8 None -> 'extrapolate'


@dataclass
class Stage:
    """A single Bellman problem stage."""
    transition: Callable[[Mapping, Mapping, Mapping], Mapping] # TODO: type signature # TODO: Defaults to identity function
    
    inputs: Sequence[str] = field(default_factory=list)
    shocks: Mapping[str, Distribution] = field(default_factory=dict) # maybe becomes a dictionary, with shocks from a distribution?
    actions: Sequence[str] = field(default_factory=list)
    outputs: Sequence[str] = field(default_factory=list)

    # Type hint is too loose: number or callable supported
    discount: Any = 1.0 
    
    reward: Callable[[Mapping, Mapping, Mapping], Any] = lambda x, k, a : 0 # TODO: type signature # TODO: Defaults to no reward

    # Note the output type of these functions are sequences, to handle multiple actions
    # If not provided, a new default is provided in post_init
    action_upper_bound: Callable[[Mapping, Mapping], Sequence[float]] = field(default = None)
    action_lower_bound: Callable[[Mapping, Mapping], Sequence[float]] = field(default = None)

    # Condition must be continuously valued, with a negative value if it fails
    # Note: I've had problems with the optimizers that use constraints; switching to Bounds -- SB
    constraints: Sequence[Callable[[Mapping, Mapping, Mapping], float]] = field(default_factory=list)

    def __post_init__(self):
        if self.action_upper_bound is None:
            self.action_upper_bound = lambda x, k : [None] * len(self.actions)

        if self.action_lower_bound is None:
            self.action_lower_bound = lambda x, k : [None] * len(self.actions)
        
    def T(self,
          x : Mapping,
          k : Mapping,
          a : Mapping,
          constrain = True) -> Mapping:

        # TODO: if constrain, and action has upper lower bounds, test here.

        if constrain:
            # Do something besides assert in production -- an exception?
            for constraint in self.constraints:
                assert constraint(x, k, a)
        
        return self.transition(x, k, a)
      
    def q(self,
          x : Mapping[str, Any],
          k : Mapping[str, Any],
          a : Mapping[str, Any],
          v_y : Callable[[Mapping, Mapping, Mapping], float]) -> Mapping:
        """
        The 'action-value function' Q.
        Takes state, shock, action states and an end-of-stage value function v_y over domain Y.
        """
        if isinstance(self.discount, float) or isinstance(self.discount, int):
            discount = self.discount
        elif callable(self.discount):
            discount = self.discount(x, k, a)

        return self.reward(x, k, a) + discount * v_y(self.T(x, k, a, constrain = False)) 

    def optimal_policy(self,
                       x_grid : Mapping[str, Sequence] = {},
                       k_grid : Mapping[str, Sequence] = {},
                       v_y : Callable[[Mapping, Mapping, Mapping], float] = lambda x : 0):
        """
        Given a grid over input and shock state values, compute the optimal action.
        Optimizes over values of Q.
        """

        pi_data = xr.DataArray(
            np.zeros([len(v) for v in x_grid.values()] + [len(v) for v in k_grid.values()]),
            dims= {**x_grid, **k_grid}.keys(),
            coords={**x_grid, **k_grid}
        )

        q_data = xr.DataArray(
            np.zeros([len(v) for v in x_grid.values()] + [len(v) for v in k_grid.values()]),
            dims= {**x_grid, **k_grid}.keys(),
            coords={**x_grid, **k_grid}
        )
        
        def q_for_minimizer(action_values, x : Mapping[str, Any] , k : Mapping[str, Any], v_y):
            """Flips negative for the _minimizer_"""
            return -self.q(
                x = x,
                k = k,
                a = {an : av for an,av in zip(self.actions, action_values)},
                v_y = v_y
            )

        ## What is happenign when there are _no_ actions?
        ## Is the optimizer still running?
        xk_grid_size = np.prod([len(xv) for xv in x_grid.values()]) * np.prod([len(kv) for kv in k_grid.values()])
        print(f"Grid size: {xk_grid_size}")
        for x_point in itertools.product(*x_grid.values()):
            x_vals = {k : v for k, v in zip(x_grid.keys() , x_point)}
            
            for k_point in itertools.product(*k_grid.values()):
                k_vals = {k : v for k, v in zip(k_grid.keys() , k_point)}

                if len(self.actions) == 0:
                    q_xk = self.q(
                        x = x_vals,
                        k = k_vals,
                        a = {},
                        v_y = v_y
                    )

                    pi_data.sel(**x_vals, **k_vals).variable.data.put(0, np.nan)
                    q_data.sel(**x_vals, **k_vals).variable.data.put(0, q_xk)
                else:
                    def scipy_constraint(constraint):
                        def scipy_constraint_fun(action_values):
                            """
                            Expects a non-negative number if passing.
                            Will take the minimum of any action condition value tested.
                            """
                            return np.array(constraint(
                                x = x_vals,
                                k = k_vals,
                                a = {an : av for an,av in zip(self.actions, action_values)}
                            ))
                    
                        return {
                            'type' : 'ineq',
                            'fun' : scipy_constraint_fun
                        }

                    bounds = [b for b in zip(
                        self.action_lower_bound(x_vals, k_vals),
                        self.action_upper_bound(x_vals, k_vals)
                    )]
                
                    pi_star_res = minimize(
                        q_for_minimizer,
                        np.ones(len(self.actions)), # JUST TESTING ones as alternative to 0; does this need to be configurable?
                        args = (x_vals, k_vals, v_y),
                        bounds = bounds,
                        constraints = [
                            scipy_constraint(constraint)
                            for constraint
                            in self.constraints
                        ] if len(self.constraints) > 0 else None,
                    )
                
                    if pi_star_res.success:
                        pi_data.sel(**x_vals, **k_vals).variable.data.put(0, pi_star_res.x)
                        q_data.sel(**x_vals, **k_vals).variable.data.put(0, -pi_star_res.fun) # flips it back
                    else:
                        print(pi_star_res)
                        print(x_vals)
                        print(k_vals)

                        #raise Exception("Failed to optimize.")
                        pi_data.sel(**x_vals, **k_vals).variable.data.put(0, np.nan)
                        q_data.sel(**x_vals, **k_vals).variable.data.put(0, np.nan)
                        
                    
        # TODO: Store these values on a grid, so it does not need to be recomputed
        #       when taking expectations
                
        return pi_data, q_data

    def solve(
        self,
        x_grid : Mapping[str, Sequence] = field(default_factory=dict),
        shock_approx_params : Mapping[str, int] = field(default_factory=dict),
        v_y : Callable[[Mapping, Mapping, Mapping], float] = lambda x : 0
        ):
        
        v_x_values = xr.DataArray(
            np.zeros([len(v) for v in x_grid.values()]),
            dims= {**x_grid}.keys(),
            coords={**x_grid}
        )

        discretized_shocks = {}

        for shock in shock_approx_params:
            if shock in self.shocks:
                discretized_shocks[shock] = self.shocks[shock].approx(
                    shock_approx_params[shock]
                    )
            else:
                print(
                    f"Warning: parameter {shock} is not a Distribution found in shocks {self.shocks}"
                )

        k_grid = {
            shock : discretized_shocks[shock].atoms.flatten()
            for shock
            in discretized_shocks
        }

        pi_star_values, q_values = self.optimal_policy(x_grid, k_grid, v_y)

        ## Taking expectations over the generated k-grid, with p_k, of...
            ## Computing optimal policy pi* and q*_value for each x,k

        for x_point in itertools.product(*x_grid.values()):
            x_vals = {k : v for k, v in zip(x_grid.keys() , x_point)}

            ## This is a somewhat hacky way to take expectations
            ## which could be done better with appropriate HARK distribution tools

            # This assumes independent shocks
            # ... but it might work for a properly constructed multivariate?
            # it will need to better use array indexing to work.
            
            # as before but including the index now
            k_grid_i = {
                shock : list(enumerate(discretized_shocks[shock].atoms.flatten()))
                for shock
                in discretized_shocks
            }

            value_x = 0
            
            for k_point in itertools.product(*k_grid_i.values()):
                k_pms = {
                    k : discretized_shocks[k].pmv[v[0]]
                    for k, v
                    in zip(k_grid.keys() , k_point)
                    }
                k_atoms = {
                    k : v[1]
                    for k, v
                    in zip(k_grid.keys(), k_point)
                    }

                if len(k_pms) > 0:
                    total_pm = np.product(list(k_pms.values()))
                else:
                    total_pm = 1

                q_xk = q_values.sel(**x_vals, **k_atoms).values

                if np.isnan(q_xk):
                    print('nan q_xk')

                value_x += q_xk * total_pm

            if np.isnan(value_x):
                print("Oh no a nan!")

            ## Need to take expectation                    
            v_x_values.sel(**x_vals).variable.data.put(
                0, 
                value_x
                )

        return SolutionDataset(xr.Dataset({
            'v_x' : v_x_values,
            'pi*' : pi_star_values,
            'q' : q_values, 
        }), actions = self.actions)

def backwards_induction(stages_data, terminal_v_y):
    """
    Solve each stage starting from the terminal value function.
    """
    
    v_y = terminal_v_y

    sols = []
    
    for t in range(len(stages_data) - 1, -1, -1):
        stage_data = stages_data[t]
        stage = stage_data['stage']

        print(f"{t}: X: {stage.inputs}, K: {list(stage.shocks.keys())}, A: {stage.actions}, Y: {stage.outputs}")
        start_time = datetime.datetime.now()
        
        x_grid = stage_data['x_grid']
        
        if 'shock_approx_params' in stage_data:
            shock_approx_params = stage_data['shock_approx_params']
        else:
            shock_approx_params = {}

        sol = stage.solve(
            x_grid = x_grid,
            shock_approx_params = shock_approx_params,
            v_y = v_y
        )

        sols.insert(0, sol)

        v_y = sol.v_x

        end_time = datetime.datetime.now()

        elapsed_time = end_time - start_time
        print(f"Time to backwards induce v_x: {elapsed_time}")
        
    return sols
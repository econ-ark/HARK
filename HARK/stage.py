from dataclasses import dataclass, field
import datetime
from typing import Any, Callable, Mapping, Sequence
import itertools
import numpy as np
from scipy.optimize import minimize
import xarray as xr

from HARK.distribution import Distribution


epsilon = 1e-4

@dataclass
class SolutionDataset(object):
    dataset : xr.Dataset
    actions: Sequence[str] = field(default_factory=list)

    # Used to tame unruly value functions between interpolations.
    value_transform : Callable[[float], float] = lambda v : v
    value_transform_inv : Callable[[float], float] = lambda v : v

    def __repr__(self):
        return self.dataset.__repr__()
        
    ## TODO: Add in assume sorted to make it faster
    def v_x(self, x : Mapping[str, Any]) -> float:
        # Note: 'fill_value' expects None or 'extrapolate' based on software version?
        return self.dataset.map(self.value_transform)['v_x'] \
                           .interp(**x, kwargs={"fill_value": 'extrapolate'}) \
                           .to_dataset().map(self.value_transform_inv)['v_x']

    def pi_star(self, x : Mapping[str, Any], k : Mapping[str, Any]):
        """

        TODO: Option to return a labelled map...
        """
        # Note: 'fill_value' expects None or 'extrapolate' based on software version?
        return self.dataset['pi*'].interp({**x, **k}, kwargs={"fill_value": 'extrapolate'})
    
    def q(self, x : Mapping[str, Any], k : Mapping[str, Any], a : Mapping[str, Any]) -> float:
        # Note: 'fill_value' expects None or 'extrapolate' based on software version?
        return self.dataset['q'].map(self.value_transform) \
                                .interp({**x, **k, **a}, kwargs={"fill_value": 'extrapolate'}) \
                                .map(self.value_transform_inv)


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
    
    optimizer_args : Mapping[str, Any] = field(default_factory=dict)

    # Used to tame unruly value functions, such as those that go to -inf
    value_transform : Callable[[float], float] = lambda v : v
    value_transform_inv : Callable[[float], float] = lambda v : v

    # used to provide a pi* value for binding states
    pi_star_points : Mapping[tuple[Sequence[float], Sequence[float]], Sequence[float]] = field(default_factory=list)

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

        q = self.reward(x, k, a) + discount * v_y(self.T(x, k, a, constrain = False)) 

        #print(f'q: {q}')
        return q

    def coords_with_pi_star_points(self,
                                   x_grid : Mapping[str, Sequence] = {},
                                   k_grid : Mapping[str, Sequence] = {}):

        new_x_grid = x_grid.copy()
        new_k_grid = k_grid.copy()

        ## Adding given pi* points to coords
        for (x_point, k_point) in self.pi_star_points:
            for xi, x_val in enumerate(x_point):
                ii = np.searchsorted(new_x_grid[self.inputs[xi]], x_point)
                new_x_grid[self.inputs[xi]] = np.insert(new_x_grid[self.inputs[xi]],ii,x_val)

            for ki, k_val in enumerate(k_point):
                ii = np.searchsorted(new_k_grid[self.shocks[ki]], k_point)
                new_k_grid[self.shocks[ki]] = np.insert(new_k_grid[self.shocks[ki]],ii,k_val)

        coords = {**new_x_grid, **new_k_grid}

        return coords, new_x_grid, new_k_grid

    def optimal_policy(self,
                       x_grid : Mapping[str, Sequence] = {},
                       k_grid : Mapping[str, Sequence] = {},
                       v_y : Callable[[Mapping, Mapping, Mapping], float] = lambda x : 0,
                       optimizer_args = None
                       ):
        """
        Given a grid over input and shock state values, compute the optimal action.
        Optimizes over values of Q.
        """

        all_optimizer_args = self.optimizer_args.copy()
        if optimizer_args is not None:
            all_optimizer_args.update(optimizer_args)

        # For now assuming this has the same shape as the x_grid
        a0f = lambda x : 0
        if 'a0f' in all_optimizer_args:
            a0f = all_optimizer_args['a0f']
            del all_optimizer_args['a0f']

        coords, _, _ = self.coords_with_pi_star_points(x_grid, k_grid)

        pi_data = xr.DataArray(
            np.zeros([len(v) for v in coords.values()]),
            dims = {**x_grid, **k_grid}.keys(),
            coords = coords
        )

        q_data = xr.DataArray(
            np.zeros([len(v) for v in coords.values()]),
            dims = {**x_grid, **k_grid}.keys(),
            coords = coords
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

        for (x_point, k_point) in self.pi_star_points:
            x_vals = {k : v for k, v in zip(x_grid.keys() , x_point)}
            k_vals = {k : v for k, v in zip(k_grid.keys() , k_point)}

            acts = self.pi_star_points[(x_point, k_point)]

            q = -q_for_minimizer(acts, x_vals, k_vals, v_y)

            pi_data.sel(**x_vals, **k_vals).variable.data.put(0, acts)
            q_data.sel(**x_vals, **k_vals).variable.data.put(
                0, q
            )

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

                    #print(f'a0: {a0f(x_vals)}')
                
                    pi_star_res = minimize(
                        q_for_minimizer,
                        a0f(x_vals), # compute starting action from states
                        args = (x_vals, k_vals, v_y),
                        bounds = bounds,
                        constraints = [
                            scipy_constraint(constraint)
                            for constraint
                            in self.constraints
                        ] if len(self.constraints) > 0 else None,
                        **all_optimizer_args
                    )
                
                    if pi_star_res.success:
                        pi_data.sel(**x_vals, **k_vals).variable.data.put(0, pi_star_res.x)
                        q_data.sel(**x_vals, **k_vals).variable.data.put(
                            0, -pi_star_res.fun # flips it back
                            ) 
                    else:
                        print(f"Optimization failure at {x_vals}, {k_vals}.")
                        print(pi_star_res)

                        #raise Exception("Failed to optimize.")
                        pi_data.sel(**x_vals, **k_vals).variable.data.put(0,  pi_star_res.x)
                        q_data.sel(**x_vals, **k_vals).variable.data.put(0, pi_star_res.fun)
                        
                    
        # TODO: Store these values on a grid, so it does not need to be recomputed
        #       when taking expectations
                
        return pi_data, q_data

    def solve(
        self,
        x_grid : Mapping[str, Sequence] = field(default_factory=dict),
        shock_approx_params : Mapping[str, int] = field(default_factory=dict),
        v_y : Callable[[Mapping, Mapping, Mapping], float] = lambda x : 0,
        optimizer_args = None
        ):

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

        ## No pre-given pi* values here; maybe shouldn't be here.
        k_grid = {
            shock : discretized_shocks[shock].atoms.flatten()
            for shock
            in discretized_shocks
        }

        ## These grids have the given pi* values added
        _, new_x_grid, new_k_grid = self.coords_with_pi_star_points(x_grid, k_grid)

        v_x_values = xr.DataArray(
            np.zeros([len(v) for v in new_x_grid.values()]),
            dims= {**new_x_grid}.keys(),
            coords={**new_x_grid}
        )

        pi_star_values, q_values = self.optimal_policy(x_grid, k_grid, v_y, optimizer_args = optimizer_args)

        ## Taking expectations over the generated k-grid, with p_k, of...
            ## Computing optimal policy pi* and q*_value for each x,k

        for x_point in itertools.product(*new_x_grid.values()):
            x_vals = {k : v for k, v in zip(x_grid.keys() , x_point)}

            ## This is a somewhat hacky way to take expectations
            ## which could be done better with appropriate HARK distribution tools

            # This assumes independent shocks
            # ... but it might work for a properly constructed multivariate?
            # it will need to better use array indexing to work.
            
            # as before but including the index now

            ## Note: no preset pi* points show up here.
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
                    print(f'nan q_xk at {x_point}, {k_point}')

                value_x += q_xk * total_pm

            if np.isnan(value_x):
                print(f"Computed value v_x at {x_point},{k_point} is nan.")

            ## Need to take expectation                    
            v_x_values.sel(**x_vals).variable.data.put(
                0, 
                value_x
                )

        return SolutionDataset(
            xr.Dataset({
                'v_x' : v_x_values,
                'pi*' : pi_star_values,
                'q' : q_values,
            }), 
            actions = self.actions,
            value_transform = self.value_transform,
            value_transform_inv = self.value_transform_inv
            )

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

        optimizer_args = stage_data['optimizer_args'] if 'optimizer_args' in stage_data else None

        sol = stage.solve(
            x_grid = x_grid,
            shock_approx_params = shock_approx_params,
            v_y = v_y,
            optimizer_args = optimizer_args
        )

        sols.insert(0, sol)

        v_y = sol.v_x

        end_time = datetime.datetime.now()

        elapsed_time = end_time - start_time
        print(f"Time to backwards induce v_x: {elapsed_time}")
        
    return sols
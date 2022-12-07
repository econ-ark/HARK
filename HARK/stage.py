from dataclasses import dataclass, field
import datetime
from typing import Any, Callable, Mapping, Sequence, Tuple
import itertools
import numpy as np
from scipy.optimize import minimize, brentq
import time
import xarray as xr

from HARK.distribution import Distribution


epsilon = 1e-4


def label_index_in_dataset(li, dataset):
    """
    There has got to be a better way to do this.
    """
    try:
        dataset.sel(li)
        return True
    except:
        return False

@dataclass
class SolutionDataset(object):
    dataset : xr.Dataset
    actions: Sequence[str] = field(default_factory=list)
    k_grid : Mapping[str, Sequence[float]] = field(default_factory=dict)

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

    def v_x_der(self, x : Mapping[str, Any]) -> float:
        # Note: 'fill_value' expects None or 'extrapolate' based on software version?
        return self.dataset['v_x_der'].interp(**x, kwargs={"fill_value": 'extrapolate'})

    def pi_star(self, x : Mapping[str, Any], k : Mapping[str, Any], ):
        """

        TODO: Option to return a labelled map...
        """
        # Note: 'fill_value' expects None or 'extrapolate' based on software version?
        ads = self.dataset['pi*'].interp({**x, **k}, kwargs={"fill_value": 'extrapolate'})

        # use of flatten() here for when ads.values is 0 dimensional.
        return {a : v for a,v in zip(self.actions, ads.values.flatten())}
    
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
    reward_der: Callable[[Mapping, Mapping, Mapping], Any] = lambda x, k, a : 0 # TODO: type signature # TODO: Defaults to no reward

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

    # Pre-computed points for the solution to this stage
    solution_points : xr.Dataset = xr.Dataset()

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

    def action_zip(self, a : Tuple):
        """
        Wraps a tuple of values for an action in a dictionary with labels.
        Useful for converting between forms of model equations.
        """
        return {an : av for an,av in zip(self.actions, a)}

    def q_for_minimizer(self, action_values, x : Mapping[str, Any] , k : Mapping[str, Any], v_y):
        """Flips negative for the _minimizer_"""
        return -self.q(
            x = x,
            k = k,
            # Try: a = self.action_zip(action_values)
            a = {an : av for an,av in zip(self.actions, action_values)},
            v_y = v_y
        )

    def d_q_d_a(self, # Trying out this notation instead of q_der_a -- debatable.
          x : Mapping[str, Any],
          k : Mapping[str, Any],
          a : Mapping[str, Any],
          v_y_der : Callable[[Mapping, Mapping, Mapping], float]) -> Mapping:
        """
        The derivative of the action-value function Q with respect to the actions.

        Takes state, shock, action states and an end-of-stage value function v_y over domain Y.
        """
        if isinstance(self.discount, float) or isinstance(self.discount, int):
            discount = self.discount
        elif callable(self.discount):
            ## WARNING: discount_der never defined
            print("discount_der for q_der_a is not implemented. Feature is missing!")
            discount = self.discount_der(x, k, a)

        ## WARNING: reward_der not defined yet
        q_der = self.reward_der(x, k, a) + discount * v_y_der(self.T(x, k, a, constrain = False)) 

        return q_der

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

        ## Add solution points to the x_grid here.
        new_x_grid = x_grid.copy()

        for x_label in self.solution_points.coords:
            for sol_x_val in self.solution_points.coords[x_label]:
                if sol_x_val.values not in x_grid[x_label]:
                    ii = np.searchsorted(new_x_grid[x_label], sol_x_val)
                    new_x_grid[x_label] = np.insert(new_x_grid[x_label], ii, sol_x_val)

        coords = {**new_x_grid, **k_grid}

        pi_data = xr.DataArray(
            np.zeros([len(v) for v in coords.values()]),
            dims = coords.keys(),
            coords = coords
        )

        q_data = xr.DataArray(
            np.zeros([len(v) for v in coords.values()]),
            dims = coords.keys(),
            coords = coords
        )

        ## What is happenign when there are _no_ actions?
        ## Is the optimizer still running?
        xk_grid_size = np.prod([len(xv) for xv in x_grid.values()]) * np.prod([len(kv) for kv in k_grid.values()])

        print(f"Grid size: {xk_grid_size}")
        for x_point in itertools.product(*new_x_grid.values()):
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

                elif 'pi*' in self.solution_points \
                    and label_index_in_dataset(x_vals, self.solution_points['pi*']):
                    acts = np.atleast_1d(self.solution_points['pi*'].sel(x_vals))

                    if not np.any(np.isnan(acts)):
                        ## k_vals is arbitrary here and is define in the previous large loop.
                        q = -self.q_for_minimizer(acts, x_vals, k_vals, v_y)

                        pi_data.sel(**x_vals, **k_vals).variable.data.put(0, acts)
                        q_data.sel(**x_vals, **k_vals).variable.data.put(
                            0, q
                        ) 
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
                        self.q_for_minimizer,
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

    ## WORK IN PROGRESS
    def optimal_policy_foc(self,
                       x_grid : Mapping[str, Sequence] = {},
                       k_grid : Mapping[str, Sequence] = {},
                       v_y_der : Callable[[Mapping, Mapping, Mapping], float] = lambda x : 0,
                       optimizer_args = None # TODO: For brettq.
                       ):
        """
        Given a grid over input and shock state values,
        and marginal output value function,
        compute the optimal action.

        Uses root finding and the first order condition.

        WORK IN PROGRESS

        Note: Only works for scalar actions?
        """

        ## OPTIMIZER SETUP STEPS
        ### Removed. Does brentq take options? Could it be passed optimizer_args?

        ## Add solution points to the x_grid here.
        new_x_grid = x_grid.copy()

        for x_label in self.solution_points.coords:
            for sol_x_val in self.solution_points.coords[x_label]:
                if sol_x_val.values not in x_grid[x_label]:
                    ii = np.searchsorted(new_x_grid[x_label], sol_x_val)
                    new_x_grid[x_label] = np.insert(new_x_grid[x_label], ii, sol_x_val)

        coords = {**new_x_grid, **k_grid}

        pi_data = xr.DataArray(
            np.zeros([len(v) for v in coords.values()]),
            dims = coords.keys(),
            coords = coords
        )

        # We don't get q_data from the rootfinding step.
        # We DO get q_der data and should be tracking it...
        # Legacy code:
        #q_data = xr.DataArray(
        #    np.zeros([len(v) for v in coords.values()]),
        #    dims = coords.keys(),
        #    coords = coords
        #)

        ## What is happenign when there are _no_ actions?
        ## Is the optimizer still running?
        xk_grid_size = np.prod([len(xv) for xv in x_grid.values()]) * np.prod([len(kv) for kv in k_grid.values()])

        print(f"Grid size: {xk_grid_size}")

        def foc(a):
            a_vals = {an : av for an,av in zip(self.actions, (a,))}
            return self.d_q_d_a(x_vals, k_vals, a_vals, v_y_der)

        for x_point in itertools.product(*new_x_grid.values()):
            x_vals = {k : v for k, v in zip(x_grid.keys() , x_point)}

            for k_point in itertools.product(*k_grid.values()):
                k_vals = {k : v for k, v in zip(k_grid.keys() , k_point)}

                # repeated code with the other optimizer -- can be functionalized out?
                if len(self.actions) == 0:
                    q_xk = self.q(
                        x = x_vals,
                        k = k_vals,
                        a = {},
                        v_y = v_y
                    )

                    pi_data.sel(**x_vals, **k_vals).variable.data.put(0, np.nan)

                # This case too is perhaps boilerplate...
                elif 'pi*' in self.solution_points \
                    and label_index_in_dataset(x_vals, self.solution_points['pi*']):
                    acts = np.atleast_1d(self.solution_points['pi*'].sel(x_vals))

                    if not np.any(np.isnan(acts)):
                        ## k_vals is arbitrary here and is define in the previous large loop.
                        # EXCEPT FOR THIS...
                        #q = -q_for_minimizer(acts, x_vals, k_vals, v_y)

                        pi_data.sel(**x_vals, **k_vals).variable.data.put(0, acts)
                        #q_data.sel(**x_vals, **k_vals).variable.data.put(
                        #    0, q
                        #) 
                else:
                    lower_bound = self.action_lower_bound(x_vals, k_vals)
                    upper_bound = self.action_upper_bound(x_vals, k_vals)

                    q_der_lower = self.d_q_d_a(
                        x_vals,
                        k_vals,
                        self.action_zip(lower_bound),
                        v_y_der
                        )
                    q_der_upper = self.d_q_d_a(
                        x_vals,
                        k_vals,
                        self.action_zip(upper_bound),
                        v_y_der
                        )

                    pi_star = None
                    q_der = None

                    ## TODO: Save and return the q_der values -- they will be useful later.
                    if q_der_lower < 0 and q_der_upper < 0:
                        a0 = lower_bound
                        q_der = q_der_lower

                        pi_data.sel(**x_vals, **k_vals).variable.data.put(0, a0)
                    elif q_der_lower > 0 and q_der_upper > 0:
                        a0 = upper_bound
                        q_der = q_der_upper

                        pi_data.sel(**x_vals, **k_vals).variable.data.put(0, a0)
                    else:
                        ## Better exception handling here
                        ## asserting that Q is concave
                        assert q_der_lower > 0 and q_der_upper < 0

                        ## TODO: Replace this the Brentq
                        a0, root_res = brentq(
                            foc,
                            lower_bound,
                            upper_bound
                        )

                        print(root_res)

                        if root_res.converged:
                            pi_data.sel(**x_vals, **k_vals).variable.data.put(0, a0)
                        else:
                            print(f"Rootfinding failure at {x_vals}, {k_vals}.")
                            print(root_res)

                            #raise Exception("Failed to optimize.")
                            pi_data.sel(**x_vals, **k_vals).variable.data.put(0,  root_res.root)

        return pi_data

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

        k_grid = {
            shock : discretized_shocks[shock].atoms.flatten()
            for shock
            in discretized_shocks
        }

        ## Add solution points to the x_grid here.
        new_x_grid = x_grid.copy()

        for x_label in self.solution_points.coords:
            for sol_x_val in np.atleast_1d(self.solution_points.coords[x_label]):
                if sol_x_val not in x_grid[x_label]:
                    ii = np.searchsorted(new_x_grid[x_label], sol_x_val)
                    new_x_grid[x_label] = np.insert(new_x_grid[x_label], ii, sol_x_val)

        v_x_values = xr.DataArray(
            np.zeros([len(v) for v in new_x_grid.values()]),
            dims= {**new_x_grid}.keys(),
            coords={**new_x_grid}
        )
        v_x_der_values = xr.DataArray(
            np.zeros([len(v) for v in new_x_grid.values()]),
            dims= {**new_x_grid}.keys(),
            coords={**new_x_grid}
        )

        pi_star_values, q_values = self.optimal_policy(x_grid, k_grid, v_y, optimizer_args = optimizer_args)

        ## Taking expectations over the generated k-grid, with p_k, of...
            ## Computing optimal policy pi* and q*_value for each x,k

        for x_point in itertools.product(*new_x_grid.values()):

            x_vals = {k : v for k, v in zip(x_grid.keys() , x_point)}

            if 'v_x' in self.solution_points \
                and label_index_in_dataset(x_vals, self.solution_points['v_x']):
                # v_x(x) given by a solution point already
                value_x = np.atleast_1d(self.solution_points['v_x'].sel(x_vals))
                value_der_x = np.atleast_1d(self.solution_points['v_x_der'].sel(x_vals))

            else:
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
                value_der_x = 0

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

                    ### NOTE: This is currently doing lookup rather than computing with the q function.
                    q_xk = q_values.sel(**x_vals, **k_atoms).values

                    ## Computing it is straightforward!
                    action_values_array = np.atleast_1d(pi_star_values.sel(**x_vals, **k_atoms))
                    a_vals_dict = {an : av for an,av in zip(self.actions, action_values_array)}
                    ## works, but redundant for now:
                    computed_q_xk = self.q(x_vals, k_atoms, a_vals_dict, v_y)

                    ### What we want is something like this:
                    q_der_xk = 0 # <-- something meaningful.

                    value_x += q_xk * total_pm
                    value_der_x += q_der_xk


            if np.isnan(value_x):
                print(f"Computed value v_x at {x_point},{k_point} is nan.")

            ## Need to take expectation                    
            v_x_values.sel(**x_vals).variable.data.put(
                0, 
                value_x
                )
            v_x_der_values.sel(**x_vals).variable.data.put(
                0, 
                value_der_x
                )

        return SolutionDataset(
            xr.Dataset({
                'v_x' : v_x_values,
                'v_x_der' : v_x_der_values,
                'pi*' : pi_star_values,
                'q' : q_values,
            }), 
            actions = self.actions,
            value_transform = self.value_transform,
            value_transform_inv = self.value_transform_inv,
            k_grid = k_grid_i
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

def simulate_stage(stage: Stage, x_values: Mapping[str, Any], policy):
    """
    Monte Carlo simulates a stage given input x and policy pi.

    TODO: Multiple inputs at once (i.e. many data points/samples...)
    """
    #The stage can be Monte Carlo simulated forward by:
    # - Sampling $\vec{k} \sim P_\vec{K}$
    k_values = {shock : stage.shocks[shock].draw(1)[0] for shock in stage.shocks}

    # - Determining actions $\vec{a} = \pi(\vec{x}, \vec{k})$
    a_values = policy(x_values, k_values)

    # - Computing reward $F(\vec{x}, \vec{k}, \vec{a})$ and discount factor $\beta(\vec{x}, \vec{k}, \vec{a})$
    reward = stage.reward(x_values, k_values, a_values)
    # TODO: Compute discount factor if it's a function

    # - Transitioning to ouput state $\vec{y} = T(\vec{x}, \vec{k}, \vec{a})$
    y_values = stage.transition(x_values, k_values, a_values)

    return k_values, a_values, y_values, reward

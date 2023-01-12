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

def data_array_to_a_func(data_array, actions):
    def func(x : Mapping[str, Any], k : Mapping[str, Any]):
        # Note: 'fill_value' expects None or 'extrapolate' based on software version?
        ads = data_array.interp({**x, **k}, kwargs={"fill_value": 'extrapolate'})

        # use of flatten() here for when ads.values is 0 dimensional.
        return {a : v for a,v in zip(actions, ads.values.flatten())}

    return func

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

    ## TODO: Should there be a standalone DecisionRule object?
    ## Or is it any function from X x K -> A?
    ## At least, it should be possible to pass this in directly.
    ## Example from 2nd llorracc lecture on consumption and labor supply
    ##         ConsLabor in HARK. has the analytic decision rule.
    ##         https://www.econ2.jhu.edu/people/ccarroll/public/LectureNotes/Consumption/ConsAndLaborSupply/
    def pi_star(self, x : Mapping[str, Any], k : Mapping[str, Any], ):
        """

        TODO: Option to return a labelled map...

        TODO: Use data_array_to_a_func
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

    def q_der(self, x : Mapping[str, Any]) -> float:
        # Note: 'fill_value' expects None or 'extrapolate' based on software version?
        return self.dataset['q_der'].interp(**x, kwargs={"fill_value": 'extrapolate'})


@dataclass# it would be nice to have this, but requires Python 3.10: (kw_only=True)
class Stage:
    """A single Bellman problem stage."""
    transition: Callable[[Mapping, Mapping, Mapping], Mapping] = lambda x, k, a : {}# TODO: type signature # TODO: Defaults to identity function
    transition_der_a: Any = None # Callable[[Mapping, Mapping, Mapping], Mapping] = None
    transition_der_x: Any = None # Callable[[Mapping, Mapping, Mapping], Mapping] = None
    ## Y x A -> X
    transition_inv: Callable[[Mapping, Mapping], Mapping] = None

    inputs: Sequence[str] = field(default_factory=list)
    shocks: Mapping[str, Distribution] = field(default_factory=dict) # maybe becomes a dictionary, with shocks from a distribution?
    actions: Sequence[str] = field(default_factory=list)
    outputs: Sequence[str] = field(default_factory=list)

    # Type hint is too loose: number or callable supported
    discount: Any = 1.0 
    
    reward: Callable[[Mapping, Mapping, Mapping], Any] = lambda x, k, a : 0 # TODO: type signature # TODO: Defaults to no reward
    reward_der: Callable[[Mapping, Mapping, Mapping], Any] = None # TODO: type signature # TODO: Defaults to no reward
    reward_der_inv: Callable[[Mapping, Mapping, Mapping], Any] = None # TODO: type signature # TODO: Defaults to no reward

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

        WARNING: This will be inaccurate if discount is a function of actions.
        """
        if self.transition_der_a is None:
            raise Exception("self.transition_der is None. Cannot compute dQ/da without derivative transition function.")
        elif isinstance(self.transition_der_a, float) or isinstance(self.transition_der_a, int):
            transition_der_a = self.transition_der_a
        elif callable(self.transition_der_a):
            transition_der_a = self.transition_der_a(x, k, a)

        if self.reward_der is None:
            raise Exception("self.reward_der is None. Cannot compute dQ/da without derivative reward function.")

        if isinstance(self.discount, float) or isinstance(self.discount, int):
            discount = self.discount
        elif callable(self.discount):
            ## WARNING: discount_der never defined
            ## ASSUMES that discount is not a function of 'a'!
            discount = self.discount(x, k, a)

        ## WARNING: reward_der not defined yet
        q_der = self.reward_der(x, k, a) + \
            discount * v_y_der(self.T(x, k, a, constrain = False)) * transition_der_a

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

        This is written with the expectation that:
          - Actions are scalar
          - the Q function is concave over the action space.

        Functionality is not guaranteed otherwise.
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

        q_der_data = xr.DataArray(
            np.zeros([len(v) for v in coords.values()]),
            dims = coords.keys(),
            coords = coords
        )

        y_data = xr.DataArray(
            np.zeros([len(v) for v in coords.values()]), ## TODO: bigger for multivariate y?
            dims = coords.keys(),
            coords = coords
        )

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
                    q_der_xk = self.d_q_d_a(
                        x_vals,
                        k_vals,
                        {}, # no actions
                        v_y_der
                        )

                    pi_data.sel(**x_vals, **k_vals).variable.data.put(0, np.nan)
                    q_der_data.sel(**x_vals, **k_vals).variable.data.put(0, q_der_xk)

                    y = self.T(x_vals, k_vals, self.action_zip([np.nan]))
                    y_n = np.array([y[k] for k in y])
                    y_data.sel(**x_vals, **k_vals).variable.data.put(0, y_n)

                # This case too is perhaps boilerplate...
                elif 'pi*' in self.solution_points \
                    and label_index_in_dataset(x_vals, self.solution_points['pi*']):
                    acts = np.atleast_1d(self.solution_points['pi*'].sel(x_vals))

                    if not np.any(np.isnan(acts)):
                        ## k_vals is arbitrary here and is define in the previous large loop.
                        # EXCEPT FOR THIS...
                        q_der_xk = self.d_q_d_a(
                            x_vals,
                            k_vals,
                            self.action_zip(acts),
                            v_y_der
                        )

                        pi_data.sel(**x_vals, **k_vals).variable.data.put(0, acts)
                        q_der_data.sel(**x_vals, **k_vals).variable.data.put(0, q_der_xk)

                        y = self.T(x_vals, k_vals, self.action_zip(acts))
                        y_n = np.array([y[k] for k in y])
                        y_data.sel(**x_vals, **k_vals).variable.data.put(0, y_n)
                else:
                    lower_bound = self.action_lower_bound(x_vals, k_vals)
                    upper_bound = self.action_upper_bound(x_vals, k_vals)

                    ##  what if no lower bound?
                    q_der_lower = None
                    if lower_bound[0] is not None:
                        q_der_lower = self.d_q_d_a(
                        x_vals,
                        k_vals,
                        self.action_zip(lower_bound),
                        v_y_der
                        )
                    else:
                        lower_bound[0] = 1e-12 ## a really high number!

                    q_der_upper = None
                    if upper_bound[0] is not None:
                        q_der_upper = self.d_q_d_a(
                        x_vals,
                        k_vals,
                        self.action_zip(upper_bound),
                        v_y_der
                        )
                    else:
                        upper_bound[0] =  1e12 ## a really high number!

                    ## TODO: Better handling of case when there is a missing bound?
                    if q_der_lower is not None and q_der_upper is not None and q_der_lower < 0 and q_der_upper < 0:
                        a0 = lower_bound

                        pi_data.sel(**x_vals, **k_vals).variable.data.put(0, a0)
                        q_der_data.sel(**x_vals, **k_vals).variable.data.put(0, q_der_lower)
                    elif q_der_lower is not None and q_der_upper is not None and q_der_lower > 0 and q_der_upper > 0:
                        a0 = upper_bound

                        pi_data.sel(**x_vals, **k_vals).variable.data.put(0, a0)
                        q_der_data.sel(**x_vals, **k_vals).variable.data.put(0, q_der_upper)
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
                            pi_data.sel(**x_vals, **k_vals).variable.data.put(0, a0)

                            q_der_xk = self.d_q_d_a(
                                x_vals,
                                k_vals,
                                self.action_zip((a0,)), # actions are scalar
                                v_y_der
                            )

                            q_der_data.sel(**x_vals, **k_vals).variable.data.put(0, q_der_xk)
                        else:
                            print(f"Rootfinding failure at {x_vals}, {k_vals}.")
                            print(root_res)

                            #raise Exception("Failed to optimize.")
                            pi_data.sel(**x_vals, **k_vals).variable.data.put(0,  root_res.root)

                            q_der_xk = self.d_q_d_a(
                                x_vals,
                                k_vals,
                                self.action_zip((root_res.root,)), # actions are scalar
                                v_y_der
                            )

                            q_der_data.sel(**x_vals, **k_vals).variable.data.put(0, q_der_xk)

                    acts =  np.atleast_1d(pi_data.sel(**x_vals, **k_vals).values)
                    y = self.T(x_vals, k_vals, self.action_zip(acts))
                    y_n = np.array([y[k] for k in y])
                    y_data.sel(**x_vals, **k_vals).variable.data.put(0, y_n)

        ## TODO: Generalize this.
        ## Add y values as coordinate to pi.
        ## Assumes invertible transition function.
        ## pi_star = pi_star.assign_coords({'a' : ('m', y_data.values)})

        return pi_data, q_der_data, y_data

    def analytic_pi_star_y(self,
                        y,
                        v_y_der : Callable[[Mapping, Mapping, Mapping], float] = lambda x : 0
                        ):
        """
        The optimal action that results in output values y.

        Available only with reward_der_inv and transition_inv
        are well-defined.
        """
        if self.reward_der_inv is None:
            raise Exception("No inverse marginal reward function found.")

        if self.transition_inv is None:
            raise Exception("No inverse transition function found. ")

        if self.transition_der_a is None or not(
            isinstance(self.transition_der_a, float) or isinstance(self.transition_der_a, int)
            ):
            raise Exception(f"No constant transition derivative found. transition_der_a is {self.transition_der_a}")

        if not isinstance(self.discount, float) or isinstance(self.discount, int):
            raise Exception("Analytic pi_star_y requires constant discount factor (rendering B' = 0).")

        ### TEST: available T_der as constant.

        return self.reward_der_inv(- self.discount * self.transition_der_a * v_y_der(y))

    def optimal_policy_egm(self,
                       y_grid : Mapping[str, Sequence] = {},
                       v_y_der : Callable[[Mapping, Mapping, Mapping], float] = lambda x : 0,
                       ):
        """
        Given a grid over output
        and marginal output value function,
        compute the optimal action.

        ## NO SHOCKS ALLOWED ##

        This depends on the stage having an
        *inverse marginal reward function*
        and *inverse transition function*.

        Does not use rootfinding!
        The 'grid' over the input 

        ### ASSUMES: No discounting this phase,
        ###           and... T' = -1 ???
        """

        if self.reward_der_inv is None:
            raise Exception("No inverse marginal reward function found. EGM requires reward_der_inv defined for this stage.")

        if self.transition_inv is None:
            raise Exception("No inverse transition function found. EGM requires transition_inv defined for this stage.")

        pi_y_data = xr.DataArray(
            np.zeros([len(v) for v in y_grid.values()]),
            dims = y_grid.keys(),
            coords = y_grid
        )

        # Collecting data for the real optimal policy with respect to inputs
        x_val_data = []
        a_val_data = []

        for y_point in itertools.product(*y_grid.values()):
            y_vals = {k : v for k, v in zip(y_grid.keys() , y_point)}

            acts = self.analytic_pi_star_y(y_vals, v_y_der)

            pi_y_data.sel(**y_vals).variable.data.put(0, acts)

            x_vals = self.transition_inv(y_vals, self.action_zip(acts))

            x_val_data.append(x_vals)
            a_val_data.append(self.action_zip(acts))

        ## TODO is this dealing with repeated values?
        x_coords = {
            x : np.array([xv[x] for xv in x_val_data])
            for x
            in self.inputs
        }

        for x_label in self.solution_points.coords:
            for sol_x_val in self.solution_points.coords[x_label]:
                if sol_x_val.values not in x_coords[x_label]:
                    ii = np.searchsorted(x_coords[x_label], sol_x_val)
                    x_coords[x_label] = np.insert(x_coords[x_label], ii, sol_x_val)

        pi_data = xr.DataArray(
            np.zeros([len(v) for v in x_coords.values()]),
            dims = x_coords.keys(),
            coords = x_coords
        )

        if 'pi*' in self.solution_points:
            for x_point in itertools.product(*x_coords.values()):
                x_vals = {k : v for k, v in zip(x_coords.keys() , x_point)}

                if label_index_in_dataset(x_vals, self.solution_points['pi*']):
                    acts = np.atleast_1d(self.solution_points['pi*'].sel(x_vals))

                    pi_data.sel(**x_vals, **{}).variable.data.put(0, acts)

        for i, x_vals in enumerate(x_val_data):
            x_vals = x_val_data[i]
            a_vals = a_val_data[i]
            acts = [a_vals[a] for a in a_vals]
            pi_data.sel(**x_vals).variable.data.put(0, acts)

        return pi_data, pi_y_data


    def discretize_shocks(self, shock_approx_params):
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

        # The grid with indices for each shock value
        k_grid_i = {
                    shock : list(enumerate(discretized_shocks[shock].atoms.flatten()))
                    for shock
                    in discretized_shocks
                }

        return discretized_shocks, k_grid, k_grid_i

    def x_grid_with_solution_points(self, x_grid):
        ## Add solution points to the x_grid here.
        new_x_grid = x_grid.copy()

        for x_label in self.solution_points.coords:
            for sol_x_val in np.atleast_1d(self.solution_points.coords[x_label]):
                if sol_x_val not in x_grid[x_label]:
                    ii = np.searchsorted(new_x_grid[x_label], sol_x_val)
                    new_x_grid[x_label] = np.insert(new_x_grid[x_label], ii, sol_x_val)

        return new_x_grid


    def xarray_expectations(self, func, x_vals, shock_approx_params, pi_star, func_arg):
        """
        Computes the expected value of FUNC over discretized shocks
        assuming policy pi_star, given starting point x_vals.

        Parameters:
        
        func - function over which to take expectations

        x_vals - grid over x values at which to compute the expecation

        shock_approx_params - parameters for the discretization of the shokcs

        pi_star - optimal policy function

        func_arg - additional arguments to func
        """

        ## TODO: Replace with HARK.distribution.calc_expectations of some form.
        ##       Problem: calculating expectation of join distribution over sevearl independent shocks.

        ## This is a somewhat hacky way to take expectations
        ## which could be done better with appropriate HARK distribution tools

        # This assumes independent shocks
        # ... but it might work for a properly constructed multivariate?
        # it will need to better use array indexing to work.
            
        # as before but including the index now

        ## Note: no preset pi* points show up here.
        expected_d = 0
        #value_der_x = 0

        discretized_shocks, k_grid, k_grid_i = self.discretize_shocks(shock_approx_params)

        for k_point in itertools.product(*k_grid_i.values()):
            k_pms = {
                k : discretized_shocks[k].pmv[v[0]]
                for k, v
                in zip(k_grid_i.keys() , k_point)
            }
            k_atoms = {

                k : v[1]
                for k, v
                in zip(k_grid_i.keys(), k_point)
            }

            if len(k_pms) > 0:
                total_pm = np.product(list(k_pms.values()))
            else:
                total_pm = 1

            ### NOTE: This is currently doing lookup rather than computing with the q function.
            d_xk = func(x_vals, k_atoms, pi_star(x_vals, k_atoms), func_arg)
            
            expected_d += d_xk * total_pm

            if np.isnan(expected_d):
                print(f"Computed value at {x_vals},{k_atoms} is nan.")
        
        return expected_d

    def analytic_marginal_value_backup(self, 
        pi_star,
        x_grid : Mapping[str, Sequence] = field(default_factory=dict),
        shock_approx_params : Mapping[str, int] = field(default_factory=dict),
    ):
        """
        Computes the beginning of stage marginal value function v'_x
        analytically.

        Takes an optimal policy pi_star as input.;
        
        Requires:
         - transition_der_x. Marginal transition with respect to starting states x.
         - transition_der_a. Marginal transition with respect to actions a.
         - reward_der. Marginal reward with respect to actions a.

        Assumes dF/dx = 0, and dB/dx = 0.        
        """
        if self.reward_der is None:
            raise Exception("No inverse marginal reward function found.")

        if self.transition_der_a is None or not(
            isinstance(self.transition_der_a, float) or isinstance(self.transition_der_a, int)
            ):
            raise Exception(f"No constant transition derivative found. transition_der_a is {self.transition_der_a}")

        if self.transition_der_x is None or not(
            isinstance(self.transition_der_x, float) or isinstance(self.transition_der_x, int)
            ):
            raise Exception(f"No constant transition derivative found. transition_der_x is {self.transition_der_x}")

        ### What if no marginal value function fiven?
        new_x_grid = self.x_grid_with_solution_points(x_grid)

        v_x_der_values = xr.DataArray(
            np.zeros([len(v) for v in new_x_grid.values()]),
            dims= {**new_x_grid}.keys(),
            coords={**new_x_grid}
        )

        def v_xk_der_x_foc(x, k, a, args):
            if isinstance(self.transition_der_x, float) or isinstance(self.transition_der_x, int):
                transition_der_x = self.transition_der_x
            elif callable(self.transition_der_x):
                transition_der_x = self.transition_der_x(x, k, a)


            if isinstance(self.transition_der_a, float) or isinstance(self.transition_der_a, int):
                transition_der_a = self.transition_der_a
            elif callable(self.transition_der_a):
                transition_der_a = self.transition_der_a(x, k, a)


            return - self.transition_der_x * self.reward_der(x, k, a) / self.transition_der_a

        for x_point in itertools.product(*new_x_grid.values()):

            x_vals = {k : v for k, v in zip(x_grid.keys() , x_point)}

            if "v'_x" in self.solution_points \
                and label_index_in_dataset(x_vals, self.solution_points["v'_x"]):

                value_x_der = np.atleast_1d(self.solution_points["v'_x"].sel(x_vals))
            else:

                value_x_der = self.xarray_expectations(
                    v_xk_der_x_foc,
                    x_vals,
                    shock_approx_params,
                    data_array_to_a_func(pi_star, self.actions),
                    None
                    )
                
            v_x_der_values.sel(**x_vals).variable.data.put(
                0, 
                value_x_der
            )

        return v_x_der_values
        


    def value_backup(
        self,
        pi_star,
        x_grid : Mapping[str, Sequence] = field(default_factory=dict),
        shock_approx_params : Mapping[str, int] = field(default_factory=dict),
        v_y : Callable[[Mapping, Mapping, Mapping], float] = lambda x : 0,
        optimizer_args = None
        ):

        new_x_grid = self.x_grid_with_solution_points(x_grid)

        v_x_values = xr.DataArray(
            np.zeros([len(v) for v in new_x_grid.values()]),
            dims= {**new_x_grid}.keys(),
            coords={**new_x_grid}
        )

        ## Taking expectations over the generated k-grid, with p_k, of...
            ## Computing optimal policy pi* and q*_value for each x,k

        for x_point in itertools.product(*new_x_grid.values()):

            x_vals = {k : v for k, v in zip(x_grid.keys() , x_point)}

            if 'v_x' in self.solution_points \
                and label_index_in_dataset(x_vals, self.solution_points['v_x']):

                value_x = np.atleast_1d(self.solution_points['v_x'].sel(x_vals))
            else:

                value_x = self.xarray_expectations(
                    self.q,
                    x_vals,
                    shock_approx_params,
                    data_array_to_a_func(pi_star, self.actions),
                    v_y
                    )
                
            v_x_values.sel(**x_vals).variable.data.put(
                0, 
                value_x
            )

        ## TODO:
        ## just return the raw data
        ## build the solution object in SOLVE

        return SolutionDataset(
            xr.Dataset({
                'v_x' : v_x_values,
                #'v_x_der' : v_x_der_values, -- 
                'pi*' : pi_star,
                #'q' : q_values,
            }), 
            actions = self.actions,
            value_transform = self.value_transform,
            value_transform_inv = self.value_transform_inv,
            #k_grid = k_grid_i
            )


    def solve(
        self,
        x_grid : Mapping[str, Sequence] = field(default_factory=dict),
        y_grid : Mapping[str, Sequence] = field(default_factory=dict),
        shock_approx_params : Mapping[str, int] = field(default_factory=dict),
        v_y :  Callable[[Mapping, Mapping, Mapping], float] = lambda x : 0,
        v_y_der : Callable[[Mapping, Mapping, Mapping], float] = lambda x : 0,
        next_sol : SolutionDataset = None,
        policy_finder_method = 'opt', # opt, foc, egm
        policy_finder_args = None ## rootfinder_args?
        # in these args? v_y_der : Callable[[Mapping, Mapping, Mapping], float] = None,
        ):

        if next_sol is not None:
            v_y = next_sol.v_x
            v_y_der = next_sol.v_x_der

        ## Pick the policy optimizer and run it with the passed-in values

        if policy_finder_method == 'opt':

            ## Build the grid

            discretized_shocks, k_grid, k_grid_i = self.discretize_shocks(shock_approx_params)

            pi_star, q_data = self.optimal_policy(
                x_grid,
                k_grid,
                v_y
            )

            ## TODO: value backup can take the already discretized shocks,
            ## or the other way around, to m
            sol = self.value_backup(
                pi_star,
                x_grid,
                shock_approx_params,
                v_y
            )

            # TODO: better design Solution object to make this smoother
            sol.dataset['q'] = q_data

            return sol

        elif policy_finder_method == 'foc':
            pass
        elif policy_finder_method == 'egm':
            #if not isinstance(shock_approx_params, Field) and len(shock_approx_params) > 0:
            #    raise Exception("EGM cannot be used in stages with exogenous shocks.")

            return self.solve_egm(y_grid, v_y_der)
            

        else:
            print(f'Did not recognize policy finder method {policy_finder_method}')

        ## do the value_update to get the value data

        ## return the solution object.

        pass

    def solve_egm(
        self,
        y_grid : Mapping[str, Sequence] = field(default_factory=dict),
        v_y_der :  Callable[[Mapping, Mapping, Mapping], float] = lambda x : 0,
        ):
        """
        THIS IS NOT COMPLETE YET
        
        See comments below.
        """

        ### NO SHOCKS
        ### and many other analytic tools in the EGM solver

        pi, pi_y = self.optimal_policy_egm(y_grid, v_y_der)

        v_x_der_values = self.analytic_marginal_value_backup(
            pi,
            dict(pi.coords.items()),
            {} # NO SHOCKS
        )
        
        return SolutionDataset(
            xr.Dataset({
                #'v_x' : v_x_values,
                'v_x_der' : v_x_der_values,
                'pi*' : pi,
                #'q' : q_values,
            }), 
            actions = self.actions,
            #k_grid = k_grid_i
            )

    def marginal_value_backup(
        self,
        pi_star,
        x_grid : Mapping[str, Sequence] = field(default_factory=dict),
        shock_approx_params : Mapping[str, int] = field(default_factory=dict),
        v_y_der : Callable[[Mapping, Mapping, Mapping], float] = lambda x : 0,
        optimizer_args = None ## rootfinder_args?
        ):
        """
        TODO
        """

        ### ASSUME: dF/dx = 0
        ### ASSUME: dB/dx = 0

        if self.transition_der_x is None or not(
            isinstance(self.transition_der_x, float) or isinstance(self.transition_der_x, int)
            ):
            raise Exception(f"No constant transition derivative found. transition_der_x is {self.transition_der_x}")

        ### What if no marginal value function fiven?
        new_x_grid = self.x_grid_with_solution_points(x_grid)

        v_x_der_values = xr.DataArray(
            np.zeros([len(v) for v in new_x_grid.values()]),
            dims= {**new_x_grid}.keys(),
            coords={**new_x_grid}
        )

        def d_q_d_x(x, k, a, args):
            if isinstance(self.transition_der_x, float) or isinstance(self.transition_der_x, int):
                transition_der_x = self.transition_der_x
            elif callable(self.transition_der_x):
                transition_der_x = self.transition_der_x(x, k, a)

            if isinstance(self.discount, float) or isinstance(self.discount, int):
                discount = self.discount
            elif callable(self.discount):
                discount = self.discount(x, k, a)

            return discount * v_y_der(self.transition(x, k, a)) * self.transition_der_x

        for x_point in itertools.product(*new_x_grid.values()):

            x_vals = {k : v for k, v in zip(x_grid.keys() , x_point)}

            if "v'_x" in self.solution_points \
                and label_index_in_dataset(x_vals, self.solution_points["v'_x"]):

                value_x_der = np.atleast_1d(self.solution_points["v'_x"].sel(x_vals))
            else:

                value_x_der = self.xarray_expectations(
                    d_q_d_x,
                    x_vals,
                    shock_approx_params,
                    data_array_to_a_func(pi_star, self.actions),
                    None
                    )
                
            v_x_der_values.sel(**x_vals).variable.data.put(
                0, 
                value_x_der
            )

        return v_x_der_values





def backwards_induction(
    stages_data,
    terminal_v_y: Callable[[Mapping, Mapping, Mapping], float] = lambda x : 0,
    terminal_v_y_der : Callable[[Mapping, Mapping, Mapping], float] = lambda x : 0,):
    """
    #Solve each stage starting from the terminal value function.
    """
    
    last_stage = stages_data[-1]['stage']
    terminal_solution = SolutionDataset(
            xr.Dataset({
                'v_x' : terminal_v_y,
                'v_x_der' : terminal_v_y_der,
            }), 
            actions = last_stage.actions,
            )

    sol = terminal_solution

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

        if 'method' in stage_data:
            method = stage_data['method']
        else:
            method = 'opt'

        optimizer_args = stage_data['optimizer_args'] if 'optimizer_args' in stage_data else None

        sol = stage.solve(
            x_grid = x_grid,
            shock_approx_params = shock_approx_params,
            next_sol = sol,
            policy_finder_method = method
            #
            # optimizer_args = optimizer_args
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

from dataclasses import dataclass, field
from collections.abc import Callable, Mapping, Sequence
import itertools
import numpy as np
from scipy.optimize import minimize
import xarray as xr

from HARK.distribution import Distribution

@dataclass
class Stage:
    """A single Bellman problem stage."""
    transition: Callable[[Mapping, Mapping, Mapping], Mapping] # TODO: type signature # TODO: Defaults to identity function
    
    inputs: Sequence[str] = field(default_factory=list)
    shocks: Mapping[str, Distribution] = field(default_factory=dict) # maybe becomes a dictionary, with shocks from a distribution?
    actions: Sequence[str] = field(default_factory=list)
    outputs: Sequence[str] = field(default_factory=list)

    discount: float = 1.0 # might become more complicated, like a distribution
    
    reward: Callable[[Mapping, Mapping, Mapping], ...] = lambda x, k, a : 0 # TODO: type signature # TODO: Defaults to no reward
    
    # Condition must be continuously valued, with a negative value if it fails
    constraints: Sequence[Callable[[Mapping, Mapping, Mapping], float]] = field(default_factory=list)
        
    def T(self,
          x : Mapping,
          k : Mapping,
          a : Mapping,
          constrain = True) -> Mapping:
        if constrain:
            # Do something besides assert in production -- an exception?
            for constraint in self.constraints:
                assert constraint(x, k, a)
        
        return self.transition(x, k, a)
    
    def v_hat( ## replace with discouunt function...self.discount(x, k, a)
        self,
        x : Mapping[str, ...],
        k : Mapping[str, ...],
        a : Mapping[str, ...],
        v_y : Callable[[Mapping, Mapping, Mapping], float]) -> Mapping:
        """
        Can be overridden.
        """
        return self.discount * v_y(self.T(x, k, a, constrain = False))
    
    def q(self,
          x : Mapping[str, ...],
          k : Mapping[str, ...],
          a : Mapping[str, ...],
          v_y : Callable[[Mapping, Mapping, Mapping], float]) -> Mapping:
        """
        The 'action-value function' Q.
        Takes state, shock, action states and an end-of-stage value function v_y over domain Y."""
        return self.reward(x, k, a) + self.v_hat(x, k, a, v_y) ## maybe substitute in ...
    
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
        
        def q_for_minimizer(action_values, x : Mapping[str, ...] , k : Mapping[str, ...], v_y):
            """Flips negative for the _minimizer_"""
            return -self.q(
                x = x,
                k = k,
                a = {an : av for an,av in zip(self.actions, action_values)},
                v_y = v_y
            )
            
        
        for x_point in itertools.product(*x_grid.values()):
            x_vals = {k : v for k, v in zip(x_grid.keys() , x_point)}
            
            for k_point in itertools.product(*k_grid.values()):
                k_vals = {k : v for k, v in zip(k_grid.keys() , k_point)}
                
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
                
                pi_star_res = minimize(
                    q_for_minimizer,
                    np.zeros(len(self.actions)), # better default than 0?
                    args = (x_vals, k_vals, v_y),
                    constraints = [
                        scipy_constraint(constraint)
                        for constraint
                        in self.constraints
                    ],
                    method="cobyla",
                    options = {
                        #'disp' : True, # for debugging
                        'maxiter' : 200000
                    }
                )
                
                if pi_star_res.success:
                    pi_data.sel(**x_vals, **k_vals).variable.data.put(0, pi_star_res.x)
                    q_data.sel(**x_vals, **k_vals).variable.data.put(0, -pi_star_res.fun) # flips it back
                else:
                    pi_data.sel(**x_vals, **k_vals).variable.data.put(0, np.nan)
                    q_data.sel(**x_vals, **k_vals).variable.data.put(0, np.nan)
                    print(pi_star_res)
                    
        # TODO: Store these values on a grid, so it does not need to be recomputed
        #       when taking expectations
                
        return pi_data, q_data

    def v_x_expectations(
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

        pi_star, q = self.optimal_policy(x_grid, k_grid, v_y)

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

                q_xk = q.sel(**x_vals, **k_atoms).values

                #import pdb; pdb.set_trace()
                value_x += q_xk * total_pm

            if np.isnan(value_x):
                print("Oh no a nan!")
                #import pdb; pdb.set_trace()

            ## Need to take expectation                    
            v_x_values.sel(**x_vals).variable.data.put(
                0, 
                value_x
                )

        return v_x_values

    def get_v_x(
        self,
        x_grid : Mapping[str, Sequence] = field(default_factory=dict),
        shock_approx_params : Mapping[str, int] = field(default_factory=dict),
        v_y : Callable[[Mapping, Mapping, Mapping], float] = lambda x : 0
        ):

        v_x_values = self.v_x_expectations(x_grid, shock_approx_params, v_y)

        def v_x(x : Mapping[str, ...]) -> float:
            return v_x_values.interp(**x)

        return v_x


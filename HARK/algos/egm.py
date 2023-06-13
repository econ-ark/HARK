

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

    v_y_der_at_y = v_y_der(y)
        
    if isinstance(v_y_der_at_y, xr.DataArray):
        v_y_der_at_y = v_y_der_at_y.values # np.atleast1darray() ?


    if 0 > v_y_der_at_y:
        raise Exception(f"Negative marginal value {v_y_der_at_y} computes at y value of {y}. Reward is {- self.discount * self.transition_der_a * v_y_der_at_y}")

    return self.reward_der_inv(- self.discount * self.transition_der_a * v_y_der_at_y)

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

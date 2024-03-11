"""
Functions to support Monte Carlo simulation of models.
"""

from copy import copy
from inspect import signature
from typing import Any, Callable, Mapping, Sequence, Union

import numpy as np

from HARK.distribution import (
    Distribution,
    IndexDistribution,
    TimeVaryingDiscreteDistribution,
)
from HARK.model import Aggregate, Control


def draw_shocks(shocks: Mapping[str, Distribution], conditions: Sequence[int]):
    """
    Draw from each shock distribution values, subject to given conditions.

    Parameters
    ------------
    shocks Mapping[str, Distribution]
        A dictionary-like mapping from shock names to distributions from which to draw

    conditions: Sequence[int]
        An array of conditions, one for each agent.
        Typically these will be agent ages.

    Parameters
    ------------
    draws : Mapping[str, Sequence]
        A mapping from shock names to drawn shock values.
    """
    draws = {}

    for shock_var in shocks:
        shock = shocks[shock_var]

        if isinstance(shock, (int, float)):
            draws[shock_var] = np.ones(len(conditions)) * shock
        elif isinstance(shock, Aggregate):
            draws[shock_var] = shock.dist.draw(1)[0]
        elif isinstance(shock, IndexDistribution) or isinstance(
            shock, TimeVaryingDiscreteDistribution
        ):
            ## TODO  his type test is awkward. They should share a superclass.
            draws[shock_var] = shock.draw(conditions)
        else:
            draws[shock_var] = shock.draw(len(conditions))

    return draws


def simulate_dynamics(
    dynamics: Mapping[str, Union[Callable, Control]],
    pre: Mapping[str, Any],
    dr: Mapping[str, Callable],
):
    """
    From the beginning-of-period state (pre), follow the dynamics,
    including any decision rules, to compute the end-of-period state.

    Parameters
    ------------

    dynamics: Mapping[str, Callable]
        Maps variable names to functions from variables to values.
        Can include Controls
        ## TODO: Make collection of equations into a named type


    pre : Mapping[str, Any]
        Bound values for all variables that must be known before beginning the period's dynamics.


    dr : Mapping[str, Callable]
        Decision rules for all the Control variables in the dynamics.
    """
    vals = pre.copy()

    for varn in dynamics:
        # Using the fact that Python dictionaries are ordered

        feq = dynamics[varn]

        if isinstance(feq, Control):
            # This tests if the decision rule is age varying.
            # If it is, this will be a vector with the decision rule for each agent.
            if isinstance(dr[varn], np.ndarray):
                ## Now we have to loop through each agent, and apply the decision rule.
                ## This is quite slow.
                for i in range(dr[varn].size):
                    vals_i = {
                        var: vals[var][i]
                        if isinstance(vals[var], np.ndarray)
                        else vals[var]
                        for var in vals
                    }
                    vals[varn][i] = dr[varn][i](
                        *[vals_i[var] for var in signature(dr[varn][i]).parameters]
                    )
            else:
                vals[varn] = dr[varn](
                    *[vals[var] for var in signature(dr[varn]).parameters]
                )  # TODO: test for signature match with Control
        else:
            vals[varn] = feq(*[vals[var] for var in signature(feq).parameters])

    return vals


def parameters_by_age(ages, parameters):
    """
    Returns parameters for this model, but with vectorized
    values which map age-varying values to agent ages.

    Parameters
    ----------
    ages: np.array
        An array of agent ages.

    parameters: dict
        A parameters dictionary

    Returns
    --------
    aged_parameters: dict
        A dictionary of parameter values.
        If a parameter is age-varying, the value is a vector
        corresponding to the values for each input age.
    """

    def aged_param(ages, p_value):
        if isinstance(p_value, (float, int)) or callable(p_value):
            return p_value
        elif isinstance(p_value, list) and len(p_value) > 1:
            pv_array = np.array(p_value)

            return np.apply_along_axis(lambda a: pv_array[a], 0, ages)
        else:
            return np.empty(ages.size)

    return {p: aged_param(ages, parameters[p]) for p in parameters}


class Simulator:
    pass


class AgentTypeMonteCarloSimulator(Simulator):
    """
    A Monte Carlo simulation engine based on the HARK.core.AgentType framework.

    Unlike HARK.core.AgentType, this class does not do any model solving,
    and depends on dynamic equations, shocks, and decision rules paased into it.

    The purpose of this class is to provide a way to simulate models without
    relying on inheritance from the AgentType class.

    This simulator makes assumptions about population birth and mortality which
    are not generic. All agents are replaced with newborns when they expire.

    Parameters
    ------------
    parameters: Mapping[str, Any]

    shocks: Mapping[str, Distribution]

    dynamics: Mapping[str, Union[Callable, Control]]

    dr: Mapping[str, Callable]

    initial: dict

    seed : int
        A seed for this instance's random number generator.

    Attributes
    ----------
    agent_count : int
        The number of agents of this type to use in simulation.

    T_sim : int
        The number of periods to simulate.
    """

    state_vars = []

    def __init__(
        self, parameters, shocks, dynamics, dr, initial, seed=0, agent_count=1, T_sim=10
    ):
        super().__init__()

        self.parameters = parameters
        self.shocks = shocks
        self.dynamics = dynamics
        self.dr = dr
        self.initial = initial

        self.seed = seed  # NOQA
        self.agent_count = agent_count
        self.T_sim = T_sim

        # changes here from HARK.core.AgentType
        self.vars = list(shocks.keys()) + list(dynamics.keys())

        self.vars_now = {v: None for v in self.vars}
        self.vars_prev = self.vars_now.copy()

        self.read_shocks = False  # NOQA
        self.shock_history = {}
        self.newborn_init_history = {}
        self.history = {}

        self.reset_rng()  # NOQA

    def reset_rng(self):
        """
        Reset the random number generator for this type.
        """
        self.RNG = np.random.default_rng(self.seed)

    def initialize_sim(self):
        """
        Prepares for a new simulation.  Resets the internal random number generator,
        makes initial states for all agents (using sim_birth), clears histories of tracked variables.
        """
        if self.T_sim <= 0:
            raise Exception(
                "T_sim represents the largest number of observations "
                + "that can be simulated for an agent, and must be a positive number."
            )

        self.reset_rng()
        self.t_sim = 0
        all_agents = np.ones(self.agent_count, dtype=bool)
        blank_array = np.empty(self.agent_count)
        blank_array[:] = np.nan
        for var in self.vars:
            if self.vars_now[var] is None:
                self.vars_now[var] = copy(blank_array)

        self.t_age = np.zeros(
            self.agent_count, dtype=int
        )  # Number of periods since agent entry
        self.t_cycle = np.zeros(
            self.agent_count, dtype=int
        )  # Which cycle period each agent is on

        # Get recorded newborn conditions or initialize blank history.
        if self.read_shocks and bool(self.newborn_init_history):
            for init_var_name in self.initial:
                self.vars_now[init_var_name] = self.newborn_init_history[init_var_name][
                    self.t_sim, :
                ]
        else:
            for var_name in self.initial:
                self.newborn_init_history[var_name] = (
                    np.zeros((self.T_sim, self.agent_count)) + np.nan
                )

        self.sim_birth(all_agents)

        self.clear_history()
        return None

    def sim_one_period(self):
        """
        Simulates one period for this type.  Calls the methods get_mortality(), get_shocks() or
        read_shocks, get_states(), get_controls(), and get_poststates().  These should be defined for
        AgentType subclasses, except get_mortality (define its components sim_death and sim_birth
        instead) and read_shocks.
        """
        # Mortality adjusts the agent population
        self.get_mortality()  # Replace some agents with "newborns"

        # state_{t-1}
        for var in self.vars:
            self.vars_prev[var] = self.vars_now[var]

            if isinstance(self.vars_now[var], np.ndarray):
                self.vars_now[var] = np.empty(self.agent_count)
                self.vars_now[var][:] = np.nan
            else:
                # Probably an aggregate variable. It may be getting set by the Market.
                pass

        shocks_now = {}

        if self.read_shocks:  # If shock histories have been pre-specified, use those
            for var_name in self.shocks:
                shocks_now[var_name] = self.shock_history[var_name][self.t_sim, :]
        else:
            ### BIG CHANGES HERE from HARK.core.AgentType
            shocks_now = draw_shocks(self.shocks, self.t_age)

        pre = parameters_by_age(self.t_age, self.parameters)

        pre.update(self.vars_prev)
        pre.update(shocks_now)
        # Won't work for 3.8: self.parameters | self.vars_prev | shocks_now

        # Age-varying decision rules captured here
        dr = parameters_by_age(self.t_age, self.dr)

        post = simulate_dynamics(self.dynamics, pre, dr)

        self.vars_now = post
        ### BIG CHANGES HERE

        # Advance time for all agents
        self.t_age = self.t_age + 1  # Age all consumers by one period

        # What will we do with cycles?
        # self.t_cycle = self.t_cycle + 1  # Age all consumers within their cycle
        # self.t_cycle[
        #    self.t_cycle == self.T_cycle
        # ] = 0  # Resetting to zero for those who have reached the end

    def make_shock_history(self):
        """
        Makes a pre-specified history of shocks for the simulation.  Shock variables should be named
        in self.shock, a mapping from shock names to distributions.  This method runs a subset
        of the standard simulation loop by simulating only mortality and shocks; each variable named
        in shocks is stored in a T_sim x agent_count array in history dictionary self.history[X].
        Automatically sets self.read_shocks to True so that these pre-specified shocks are used for
        all subsequent calls to simulate().

        Returns
        -------
        shock_history: dict
            The subset of simulation history that are the shocks for each agent and time.
        """
        # Re-initialize the simulation
        self.initialize_sim()
        self.simulate()

        for shock_name in self.shocks:
            self.shock_history[shock_name] = self.history[shock_name]

        # Flag that shocks can be read rather than simulated
        self.read_shocks = True
        self.clear_history()

        return self.shock_history

    def get_mortality(self):
        """
        Simulates mortality or agent turnover.
        Agents die when their states `live` is less than or equal to zero.
        """
        who_dies = self.vars_now["live"] <= 0

        self.sim_birth(who_dies)

        self.who_dies = who_dies
        return None

    def sim_birth(self, which_agents):
        """
        Makes new agents for the simulation.  Takes a boolean array as an input, indicating which
        agent indices are to be "born".  Does nothing by default, must be overwritten by a subclass.

        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.agent_count indicating which agents should be "born".

        Returns
        -------
        None
        """
        if self.read_shocks:
            t = self.t_sim
            initial_vals = {
                init_var: self.newborn_init_history[init_var][t, which_agents]
                for init_var in self.initial
            }

        else:
            initial_vals = draw_shocks(self.initial, np.zeros(which_agents.sum()))

        if np.sum(which_agents) > 0:
            for varn in initial_vals:
                self.vars_now[varn][which_agents] = initial_vals[varn]
                self.newborn_init_history[varn][self.t_sim, which_agents] = (
                    initial_vals[varn]
                )

        self.t_age[which_agents] = 0
        self.t_cycle[which_agents] = 0

    def simulate(self, sim_periods=None):
        """
        Simulates this agent type for a given number of periods. Defaults to
        self.T_sim if no input.
        Records histories of attributes named in self.track_vars in
        self.history[varname].

        Parameters
        ----------
        None

        Returns
        -------
        history : dict
            The history tracked during the simulation.
        """
        if not hasattr(self, "t_sim"):
            raise Exception(
                "It seems that the simulation variables were not initialize before calling "
                + "simulate(). Call initialize_sim() to initialize the variables before calling simulate() again."
            )
        if sim_periods is not None and self.T_sim < sim_periods:
            raise Exception(
                "To simulate, sim_periods has to be larger than the maximum data set size "
                + "T_sim. Either increase the attribute T_sim of this agent type instance "
                + "and call the initialize_sim() method again, or set sim_periods <= T_sim."
            )

        # Ignore floating point "errors". Numpy calls it "errors", but really it's excep-
        # tions with well-defined answers such as 1.0/0.0 that is np.inf, -1.0/0.0 that is
        # -np.inf, np.inf/np.inf is np.nan and so on.
        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            if sim_periods is None:
                sim_periods = self.T_sim

            for t in range(sim_periods):
                self.sim_one_period()

                # track all the vars -- shocks and dynamics
                for var_name in self.vars:
                    self.history[var_name][self.t_sim, :] = self.vars_now[var_name]

                self.t_sim += 1

            return self.history

    def clear_history(self):
        """
        Clears the histories.
        """
        for var_name in self.vars:
            self.history[var_name] = np.empty((self.T_sim, self.agent_count))
            self.history[var_name].fill(np.nan)

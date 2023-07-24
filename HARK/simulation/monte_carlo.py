"""
Functions to support Monte Carlo simulation of models.
"""

from HARK.distribution import Distribution, IndexDistribution, TimeVaryingDiscreteDistribution
from inspect import signature
import numpy as np
from typing import Any, Callable, Mapping, Sequence, Union

class Aggregate:
    """
    Used to designate a shock as an aggregate shock.
    If so designated, draws from the shock will be scalar rather
    than array valued.
    """
    def __init__(self, dist: Distribution):
        self.dist = dist

class Control:
    """
    Should go in HARK.model
    """

    def __init__(self, args):
        pass

def draw_shocks(
        shocks: Mapping[str, Distribution],
        conditions: Sequence[int]
        ):
    """

    Parameters
    ------------
    shocks Mapping[str, Distribution]
        A dictionary-like mapping from shock names to distributions from which to draw

    conditions: Sequence[int]
        An array of conditions, one for each agent.
        Typically these will be agent ages.
    """
    draws = {}

    for shock_var in shocks:
        shock = shocks[shock_var]
        if isinstance(shock, Aggregate):
            draws[shock_var] = shock.dist.draw(1)[0]
        elif isinstance(shock, IndexDistribution) \
            or isinstance(shock, TimeVaryingDiscreteDistribution):
            draws[shock_var] = shock.draw(conditions)
        else:
            draws[shock_var] = shock.draw(len(conditions))

    return draws

def simulate_dynamics(
        dynamics : Mapping[str, Union[Callable, Control]],
        pre : Mapping[str, Any],
        dr : Mapping[str, Callable]
):
    """

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
            vals[varn] = dr[varn](*[
                vals[var]
                for var 
                in signature(dr[varn]).parameters]) # TODO: test for signature match with Control
        else:
            vals[varn] = feq(*[vals[var] for var in signature(feq).parameters])

    return vals

class Simulator():
    pass

class AgentTypeMonteCarloSimulator(Simulator):
    """
    A Monte Carlo simulation engine based on the HARK.core.AgentType framework.
    Unlike HARK.core.AgentType, this class:
      * does not do any model solving
      * depends on dynamic equations, shocks, and decision rules paased into it

    The purpose of this class is to provide a way to simulate models without
    relying on inheritance from the AgentType class.

    This simulator makes assumptions about population birth and mortality which
    are not generic. They are: TODO.
    
    Parameters
    ----------
    seed : int
        A seed for this instance's random number generator.

    Attributes
    ----------
    AgentCount : int
        The number of agents of this type to use in simulation.

    state_vars : list of string
        The string labels for this AgentType's model state variables.
    """

    state_vars = []

    def __init__(
        self,
        parameters,
        shocks,
        dynamics,
        dr,
        seed=0,
        agent_count = 1,
        T_sim = 10
    ):
        super().__init__()

        self.parameters = parameters
        self.shocks = shocks
        self.dynamics = dynamics
        self.dr = dr

        self.seed = seed  # NOQA
        self.agent_count = agent_count
        self.T_sim = T_sim

        # changes here from HARK.core.AgentType
        self.vars = list(shocks.keys()) + list(dynamics.keys())

        self.vars_now = {v: None for v in self.vars}
        self.vars_prev = self.state_now.copy()

        self.read_shocks = False  # NOQA
        self.shock_history = {}
        self.newborn_init_history = {}
        self.history = {}

        self.reset_rng()  # NOQA

    def reset_rng(self):
        """
        Reset the random number generator for this type.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        self.RNG = np.random.default_rng(self.seed)

    def initialize_sim(self):
        """
        Prepares for a new simulation.  Resets the internal random number generator,
        makes initial states for all agents (using sim_birth), clears histories of tracked variables.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if not hasattr(self, "T_sim"):
            raise Exception(
                "To initialize simulation variables it is necessary to first "
                + "set the attribute T_sim to the largest number of observations "
                + "you plan to simulate for each agent including re-births."
            )
        elif self.T_sim <= 0:
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

            # elif self.state_prev[var] is None:
            #    self.state_prev[var] = copy(blank_array)
        self.t_age = np.zeros(
            self.agent_count, dtype=int
        )  # Number of periods since agent entry
        self.t_cycle = np.zeros(
            self.agent_count, dtype=int
        )  # Which cycle period each agent is on
        self.sim_birth(all_agents)

        # If we are asked to use existing shocks and a set of initial conditions
        # exist, use them
        ### TODO what to do with this?
        if self.read_shocks and bool(self.newborn_init_history):
            for var_name in self.state_now:
                # Check that we are actually given a value for the variable
                if var_name in self.newborn_init_history.keys():
                    # Copy only array-like idiosyncratic states. Aggregates should
                    # not be set by newborns
                    idio = (
                        isinstance(self.state_now[var_name], np.ndarray)
                        and len(self.state_now[var_name]) == self.AgentCount
                    )
                    if idio:
                        self.state_now[var_name] = self.newborn_init_history[var_name][
                            0
                        ]

                else:
                    warn(
                        "The option for reading shocks was activated but "
                        + "the model requires state "
                        + var_name
                        + ", not contained in "
                        + "newborn_init_history."
                    )

        self.clear_history()
        return None

    def sim_one_period(self):
        """
        Simulates one period for this type.  Calls the methods get_mortality(), get_shocks() or
        read_shocks, get_states(), get_controls(), and get_poststates().  These should be defined for
        AgentType subclasses, except get_mortality (define its components sim_death and sim_birth
        instead) and read_shocks.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if not hasattr(self, "solution"):
            raise Exception(
                "Model instance does not have a solution stored. To simulate, it is necessary"
                " to run the `solve()` method of the class first."
            )

        # Mortality adjusts the agent population
        self.get_mortality()  # Replace some agents with "newborns"

        # state_{t-1}
        for var in self.vars:
            self.vars_prev[var] = self.vars_now[var]

            if isinstance(self.vars_now[var], np.ndarray):
                self.vars_now[var] = np.empty(self.AgentCount)
            else:
                # Probably an aggregate variable. It may be getting set by the Market.
                pass

        shocks_now = {}

        if self.read_shocks:  # If shock histories have been pre-specified, use those
            for var_name in self.shocks:
                shocks_now[var_name] = self.shock_history[var_name][self.t_sim, :]
        else:  # Otherwise, draw shocks as usual according to subclass-specific method
            ### BIG CHANGES HERE from HARK.core.AgentType
            shocks_now = draw_shocks(self.shocks)

        # maybe need to time index the parameters here somehow?
        pre = self.parameters + shocks_now + self.vars_prev
        
        post = simulate_dynamics(self.dynamics, pre, self.dr)
        
        self.vars_now = post
        ### BIG CHANGES HERE

        # Advance time for all agents
        self.t_age = self.t_age + 1  # Age all consumers by one period
        self.t_cycle = self.t_cycle + 1  # Age all consumers within their cycle
        self.t_cycle[
            self.t_cycle == self.T_cycle
        ] = 0  # Resetting to zero for those who have reached the end

    def make_shock_history(self):
        """
        Makes a pre-specified history of shocks for the simulation.  Shock variables should be named
        in self.shock_vars, a list of strings that is subclass-specific.  This method runs a subset
        of the standard simulation loop by simulating only mortality and shocks; each variable named
        in shock_vars is stored in a T_sim x AgentCount array in history dictionary self.history[X].
        Automatically sets self.read_shocks to True so that these pre-specified shocks are used for
        all subsequent calls to simulate().

        ### TODO: Rethink this for when shocks are passed in.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Re-initialize the simulation
        self.initialize_sim()

        # Make blank history arrays for each shock variable (and mortality)
        for var_name in self.shock_vars:
            self.shock_history[var_name] = (
                np.zeros((self.T_sim, self.agent_count)) + np.nan
            )
        self.shock_history["who_dies"] = np.zeros(
            (self.T_sim, self.agent_count), dtype=bool
        )

        # Also make blank arrays for the draws of newborns' initial conditions
        for var_name in self.state_vars:
            self.newborn_init_history[var_name] = (
                np.zeros((self.T_sim, self.agent_count)) + np.nan
            )

        # Record the initial condition of the newborns created by
        # initialize_sim -> sim_births
        for var_name in self.state_vars:
            # Check whether the state is idiosyncratic or an aggregate
            idio = (
                isinstance(self.state_now[var_name], np.ndarray)
                and len(self.state_now[var_name]) == self.agent_count
            )
            if idio:
                self.newborn_init_history[var_name][self.t_sim] = self.state_now[
                    var_name
                ]
            else:
                # Aggregate state is a scalar. Assign it to every agent.
                self.newborn_init_history[var_name][self.t_sim, :] = self.state_now[
                    var_name
                ]

        # Make and store the history of shocks for each period
        for t in range(self.T_sim):
            # Deaths
            self.get_mortality()
            self.shock_history["who_dies"][t, :] = self.who_dies

            # Initial conditions of newborns
            if np.sum(self.who_dies) > 0:
                for var_name in self.state_vars:
                    # Check whether the state is idiosyncratic or an aggregate
                    idio = (
                        isinstance(self.state_now[var_name], np.ndarray)
                        and len(self.state_now[var_name]) == self.agent_count
                    )
                    if idio:
                        self.newborn_init_history[var_name][
                            t, self.who_dies
                        ] = self.state_now[var_name][self.who_dies]
                    else:
                        self.newborn_init_history[var_name][
                            t, self.who_dies
                        ] = self.state_now[var_name]

            # Other Shocks
            self.get_shocks()
            for var_name in self.shock_vars:
                self.shock_history[var_name][t, :] = self.shocks[var_name]

            self.t_sim += 1
            self.t_age = self.t_age + 1  # Age all consumers by one period
            self.t_cycle = self.t_cycle + 1  # Age all consumers within their cycle
            self.t_cycle[
                self.t_cycle == self.T_cycle
            ] = 0  # Resetting to zero for those who have reached the end

        # Flag that shocks can be read rather than simulated
        self.read_shocks = True

    def get_mortality(self):
        """
        Simulates mortality or agent turnover according to some model-specific rules named sim_death
        and sim_birth (methods of an AgentType subclass).  sim_death takes no arguments and returns
        a Boolean array of size AgentCount, indicating which agents of this type have "died" and
        must be replaced.  sim_birth takes such a Boolean array as an argument and generates initial
        post-decision states for those agent indices.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.read_shocks:
            who_dies = self.shock_history["who_dies"][self.t_sim, :]
            # Instead of simulating births, assign the saved newborn initial conditions
            if np.sum(who_dies) > 0:
                for var_name in self.state_now:
                    if var_name in self.newborn_init_history.keys():
                        # Copy only array-like idiosyncratic states. Aggregates should
                        # not be set by newborns
                        idio = (
                            isinstance(self.state_now[var_name], np.ndarray)
                            and len(self.state_now[var_name]) == self.AgentCount
                        )
                        if idio:
                            self.state_now[var_name][
                                who_dies
                            ] = self.newborn_init_history[var_name][
                                self.t_sim, who_dies
                            ]

                    else:
                        warn(
                            "The option for reading shocks was activated but "
                            + "the model requires state "
                            + var_name
                            + ", not contained in "
                            + "newborn_init_history."
                        )

                # Reset ages of newborns
                self.t_age[who_dies] = 0
                self.t_cycle[who_dies] = 0
        else:
            who_dies = self.sim_death()
            self.sim_birth(who_dies)
        self.who_dies = who_dies
        return None

    def sim_death(self):
        """
        Determines which agents in the current population "die" or should be replaced.  Takes no
        inputs, returns a Boolean array of size self.AgentCount, which has True for agents who die
        and False for those that survive. Returns all False by default, must be overwritten by a
        subclass to have replacement events.

        Parameters
        ----------
        None

        Returns
        -------
        who_dies : np.array
            Boolean array of size self.AgentCount indicating which agents die and are replaced.
        """
        who_dies = np.zeros(self.agent_count, dtype=bool)
        return who_dies

    def sim_birth(self, which_agents):
        """
        Makes new agents for the simulation.  Takes a boolean array as an input, indicating which
        agent indices are to be "born".  Does nothing by default, must be overwritten by a subclass.

        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.AgentCount indicating which agents should be "born".

        Returns
        -------
        None
        """
        print("AgentType subclass must define method sim_birth!")
        return None      

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

        if not hasattr(self, "T_sim"):
            raise Exception(
                "This agent type instance must have the attribute T_sim set to a positive integer."
                + "Set T_sim to match the largest dataset you might simulate, and run this agent's"
                + "initalizeSim() method before running simulate() again."
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

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for var_name in self.vars:
            self.history[var_name] = np.empty((self.T_sim, self.AgentCount))
            self.history[var_name].fill(np.nan)
"""
High-level functions and classes for solving a wide variety of economic models.
The "core" of HARK is a framework for "microeconomic" and "macroeconomic"
models.  A micro model concerns the dynamic optimization problem for some type
of agents, where agents take the inputs to their problem as exogenous.  A macro
model adds an additional layer, endogenizing some of the inputs to the micro
problem by finding a general equilibrium dynamic rule.
"""

# Set logging and define basic functions
# Set logging and define basic functions
import logging
import sys
from collections import namedtuple
from copy import copy, deepcopy
from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, List, Optional, Union
from warnings import warn

import numpy as np
import pandas as pd
from xarray import DataArray

from HARK.distribution import (
    Distribution,
    IndexDistribution,
    TimeVaryingDiscreteDistribution,
    combine_indep_dstns,
)
from HARK.parallel import multi_thread_commands, multi_thread_commands_fake
from HARK.utilities import NullFunc, get_arg_names

logging.basicConfig(format="%(message)s")
_log = logging.getLogger("HARK")
_log.setLevel(logging.ERROR)


def disable_logging():
    _log.disabled = True


def enable_logging():
    _log.disabled = False


def warnings():
    _log.setLevel(logging.WARNING)


def quiet():
    _log.setLevel(logging.ERROR)


def verbose():
    _log.setLevel(logging.INFO)


def set_verbosity_level(level):
    _log.setLevel(level)


class Parameters:
    """
    This class defines an object that stores all of the parameters for a model
    as an internal dictionary. It is designed to also handle the age-varying
    dynamics of parameters.

    Attributes
    ----------

    _length : int
        The terminal age of the agents in the model.
    _invariant_params : list
        A list of the names of the parameters that are invariant over time.
    _varying_params : list
        A list of the names of the parameters that vary over time.
    """

    def __init__(self, **parameters: Any):
        """
        Initializes a Parameters object and parses the age-varying
        dynamics of the parameters.

        Parameters
        ----------

        parameters : keyword arguments
            Any number of keyword arguments of the form key=value.
            To parse a dictionary of parameters, use the ** operator.
        """
        params = parameters.copy()
        self._length = params.pop("T_cycle", None)
        self._invariant_params = set()
        self._varying_params = set()
        self._parameters: Dict[str, Union[int, float, np.ndarray, list, tuple]] = {}

        for key, value in params.items():
            self._parameters[key] = self.__infer_dims__(key, value)

    def __infer_dims__(
        self, key: str, value: Union[int, float, np.ndarray, list, tuple, None]
    ) -> Union[int, float, np.ndarray, list, tuple]:
        """
        Infers the age-varying dimensions of a parameter.

        If the parameter is a scalar, numpy array, or None, it is assumed to be
        invariant over time. If the parameter is a list or tuple, it is assumed
        to be varying over time. If the parameter is a list or tuple of length
        greater than 1, the length of the list or tuple must match the
        `_term_age` attribute of the Parameters object.

        Parameters
        ----------
        key : str
            name of parameter
        value : Any
            value of parameter

        """
        if isinstance(value, (int, float, np.ndarray, type(None))):
            self.__add_to_invariant__(key)
            return value
        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                self.__add_to_invariant__(key)
                return value[0]
            if self._length is None or self._length == 1:
                self._length = len(value)
            if len(value) == self._length:
                self.__add_to_varying__(key)
                return value
            raise ValueError(
                f"Parameter {key} must be of length 1 or {self._length}, not {len(value)}"
            )
        raise ValueError(f"Parameter {key} has unsupported type {type(value)}")

    def __add_to_invariant__(self, key: str):
        """
        Adds parameter name to invariant set and removes from varying set.
        """
        self._varying_params.discard(key)
        self._invariant_params.add(key)

    def __add_to_varying__(self, key: str):
        """
        Adds parameter name to varying set and removes from invariant set.
        """
        self._invariant_params.discard(key)
        self._varying_params.add(key)

    def __getitem__(self, item_or_key: Union[int, str]):
        """
        If item_or_key is an integer, returns a Parameters object with the parameters
        that apply to that age. This includes all invariant parameters and the
        `item_or_key`th element of all age-varying parameters. If item_or_key is a string,
        it returns the value of the parameter with that name.
        """
        if isinstance(item_or_key, int):
            if item_or_key >= self._length:
                raise ValueError(
                    f"Age {item_or_key} is greater than or equal to terminal age {self._length}."
                )

            params = {key: self._parameters[key] for key in self._invariant_params}
            params.update(
                {
                    key: self._parameters[key][item_or_key]
                    for key in self._varying_params
                }
            )
            return Parameters(**params)
        elif isinstance(item_or_key, str):
            return self._parameters[item_or_key]

    def __setitem__(self, key: str, value: Any):
        """
        Sets the value of a parameter.

        Parameters
        ----------
        key : str
            name of parameter
        value : Any
            value of parameter

        """
        if not isinstance(key, str):
            raise ValueError("Parameters must be set with a string key")
        self._parameters[key] = self.__infer_dims__(key, value)

    def keys(self):
        """
        Returns a list of the names of the parameters.
        """
        return self._invariant_params | self._varying_params

    def values(self):
        """
        Returns a list of the values of the parameters.
        """
        return list(self._parameters.values())

    def items(self):
        """
        Returns a list of tuples of the form (name, value) for each parameter.
        """
        return list(self._parameters.items())

    def __iter__(self):
        """
        Allows for iterating over the parameter names.
        """
        return iter(self.keys())

    def __deepcopy__(self, memo):
        """
        Returns a deep copy of the Parameters object.
        """
        return Parameters(**deepcopy(self.to_dict(), memo))

    def to_dict(self):
        """
        Returns a dictionary of the parameters.
        """
        return {key: self._parameters[key] for key in self.keys()}

    def to_namedtuple(self):
        """
        Returns a namedtuple of the parameters.
        """
        return namedtuple("Parameters", self.keys())(**self.to_dict())

    def update(self, other_params):
        """
        Updates the parameters with the values from another
        Parameters object or a dictionary.

        Parameters
        ----------
        other_params : Parameters or dict
            Parameters object or dictionary of parameters to update with.
        """
        if isinstance(other_params, Parameters):
            self._parameters.update(other_params.to_dict())
        elif isinstance(other_params, dict):
            self._parameters.update(other_params)
        else:
            raise ValueError("Parameters must be a dict or a Parameters object")

    def __str__(self):
        """
        Returns a simple string representation of the Parameters object.
        """
        return f"Parameters({str(self.to_dict())})"

    def __repr__(self):
        """
        Returns a detailed string representation of the Parameters object.
        """
        return f"Parameters( _age_inv = {self._invariant_params}, _age_var = {self._varying_params}, | {self.to_dict()})"


class Model:
    """
    A class with special handling of parameters assignment.
    """

    def __init__(self):
        if not hasattr(self, "parameters"):
            self.parameters = {}

    def assign_parameters(self, **kwds):
        """
        Assign an arbitrary number of attributes to this agent.

        Parameters
        ----------
        **kwds : keyword arguments
            Any number of keyword arguments of the form key=value.  Each value
            will be assigned to the attribute named in self.

        Returns
        -------
        none
        """
        self.parameters.update(kwds)
        for key in kwds:
            setattr(self, key, kwds[key])

    def get_parameter(self, name):
        """
        Returns a parameter of this model

        Parameters
        ----------
        name : string
            The name of the parameter to get

        Returns
        -------
        value :
            The value of the parameter
        """
        return self.parameters[name]

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.parameters == other.parameters

        return NotImplemented

    def __str__(self):
        type_ = type(self)
        module = type_.__module__
        qualname = type_.__qualname__

        s = f"<{module}.{qualname} object at {hex(id(self))}.\n"
        s += "Parameters:"

        for p in self.parameters:
            s += f"\n{p}: {self.parameters[p]}"

        s += ">"
        return s

    def describe(self):
        return self.__str__()


class AgentType(Model):
    """
    A superclass for economic agents in the HARK framework. Each model should
    specify its own subclass of AgentType, inheriting its methods and overwriting
    as necessary.  Critically, every subclass of AgentType should define class-
    specific static values of the attributes time_vary and time_inv as lists of
    strings.  Each element of time_vary is the name of a field in AgentSubType
    that varies over time in the model.  Each element of time_inv is the name of
    a field in AgentSubType that is constant over time in the model.

    Parameters
    ----------
    solution_terminal : Solution
        A representation of the solution to the terminal period problem of
        this AgentType instance, or an initial guess of the solution if this
        is an infinite horizon problem.
    cycles : int
        The number of times the sequence of periods is experienced by this
        AgentType in their "lifetime".  cycles=1 corresponds to a lifecycle
        model, with a certain sequence of one period problems experienced
        once before terminating.  cycles=0 corresponds to an infinite horizon
        model, with a sequence of one period problems repeating indefinitely.
    pseudo_terminal : boolean
        Indicates whether solution_terminal isn't actually part of the
        solution to the problem (as a known solution to the terminal period
        problem), but instead represents a "scrap value"-style termination.
        When True, solution_terminal is not included in the solution; when
        false, solution_terminal is the last element of the solution.
    tolerance : float
        Maximum acceptable "distance" between successive solutions to the
        one period problem in an infinite horizon (cycles=0) model in order
        for the solution to be considered as having "converged".  Inoperative
        when cycles>0.
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
        solution_terminal=None,
        pseudo_terminal=True,
        tolerance=0.000001,
        seed=0,
        **kwds,
    ):
        super().__init__()

        if solution_terminal is None:
            solution_terminal = NullFunc()

        self.solution_terminal = solution_terminal  # NOQA
        self.pseudo_terminal = pseudo_terminal  # NOQA
        self.solve_one_period = NullFunc()  # NOQA
        self.tolerance = tolerance  # NOQA
        self.seed = seed  # NOQA
        self.track_vars = []  # NOQA
        self.state_now = {sv: None for sv in self.state_vars}
        self.state_prev = self.state_now.copy()
        self.controls = {}
        self.shocks = {}
        self.read_shocks = False  # NOQA
        self.shock_history = {}
        self.newborn_init_history = {}
        self.history = {}
        self.assign_parameters(**kwds)  # NOQA
        self.reset_rng()  # NOQA

    def add_to_time_vary(self, *params):
        """
        Adds any number of parameters to time_vary for this instance.

        Parameters
        ----------
        params : string
            Any number of strings naming attributes to be added to time_vary

        Returns
        -------
        None
        """
        for param in params:
            if param not in self.time_vary:
                self.time_vary.append(param)

    def add_to_time_inv(self, *params):
        """
        Adds any number of parameters to time_inv for this instance.

        Parameters
        ----------
        params : string
            Any number of strings naming attributes to be added to time_inv

        Returns
        -------
        None
        """
        for param in params:
            if param not in self.time_inv:
                self.time_inv.append(param)

    def del_from_time_vary(self, *params):
        """
        Removes any number of parameters from time_vary for this instance.

        Parameters
        ----------
        params : string
            Any number of strings naming attributes to be removed from time_vary

        Returns
        -------
        None
        """
        for param in params:
            if param in self.time_vary:
                self.time_vary.remove(param)

    def del_from_time_inv(self, *params):
        """
        Removes any number of parameters from time_inv for this instance.

        Parameters
        ----------
        params : string
            Any number of strings naming attributes to be removed from time_inv

        Returns
        -------
        None
        """
        for param in params:
            if param in self.time_inv:
                self.time_inv.remove(param)

    def unpack(self, parameter):
        """
        Unpacks a parameter from a solution object for easier access.
        After the model has been solved, the parameters (like consumption function)
        reside in the attributes of each element of `ConsumerType.solution`
        (e.g. `cFunc`).  This method creates a (time varying) attribute of the given
        parameter name that contains a list of functions accessible by `ConsumerType.parameter`.

        Parameters
        ----------
        parameter: str
            Name of the function to unpack from the solution

        Returns
        -------
        none
        """
        setattr(self, parameter, list())
        for solution_t in self.solution:
            self.__dict__[parameter].append(solution_t.__dict__[parameter])
        self.add_to_time_vary(parameter)

    def solve(self, verbose=False):
        """
        Solve the model for this instance of an agent type by backward induction.
        Loops through the sequence of one period problems, passing the solution
        from period t+1 to the problem for period t.

        Parameters
        ----------
        verbose : boolean
            If True, solution progress is printed to screen.

        Returns
        -------
        none
        """

        # Ignore floating point "errors". Numpy calls it "errors", but really it's excep-
        # tions with well-defined answers such as 1.0/0.0 that is np.inf, -1.0/0.0 that is
        # -np.inf, np.inf/np.inf is np.nan and so on.
        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            self.pre_solve()  # Do pre-solution stuff
            self.solution = solve_agent(
                self, verbose
            )  # Solve the model by backward induction
            self.post_solve()  # Do post-solution stuff

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

    def check_elements_of_time_vary_are_lists(self):
        """
        A method to check that elements of time_vary are lists.
        """
        for param in self.time_vary:
            if not isinstance(
                getattr(self, param),
                (TimeVaryingDiscreteDistribution, IndexDistribution),
            ):
                assert type(getattr(self, param)) == list, (
                    param
                    + " is not a list or time varying distribution,"
                    + " but should be because it is in time_vary"
                )

    def check_restrictions(self):
        """
        A method to check that various restrictions are met for the model class.
        """
        return

    def pre_solve(self):
        """
        A method that is run immediately before the model is solved, to check inputs or to prepare
        the terminal solution, perhaps.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        self.check_restrictions()
        self.check_elements_of_time_vary_are_lists()
        return None

    def post_solve(self):
        """
        A method that is run immediately after the model is solved, to finalize
        the solution in some way.  Does nothing here.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        return None

    def initialize_sim(self):
        """
        Prepares this AgentType for a new simulation.  Resets the internal random number generator,
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
        all_agents = np.ones(self.AgentCount, dtype=bool)
        blank_array = np.empty(self.AgentCount)
        blank_array[:] = np.nan
        for var in self.state_now:
            if self.state_now[var] is None:
                self.state_now[var] = copy(blank_array)

            # elif self.state_prev[var] is None:
            #    self.state_prev[var] = copy(blank_array)
        self.t_age = np.zeros(
            self.AgentCount, dtype=int
        )  # Number of periods since agent entry
        self.t_cycle = np.zeros(
            self.AgentCount, dtype=int
        )  # Which cycle period each agent is on
        self.sim_birth(all_agents)

        # If we are asked to use existing shocks and a set of initial conditions
        # exist, use them
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
        for var in self.state_now:
            self.state_prev[var] = self.state_now[var]

            if isinstance(self.state_now[var], np.ndarray):
                self.state_now[var] = np.empty(self.AgentCount)
            else:
                # Probably an aggregate variable. It may be getting set by the Market.
                pass

        if self.read_shocks:  # If shock histories have been pre-specified, use those
            self.read_shocks_from_history()
        else:  # Otherwise, draw shocks as usual according to subclass-specific method
            self.get_shocks()
        self.get_states()  # Determine each agent's state at decision time
        self.get_controls()  # Determine each agent's choice or control variables based on states
        self.get_poststates()  # Move now state_now to state_prev

        # Advance time for all agents
        self.t_age = self.t_age + 1  # Age all consumers by one period
        self.t_cycle = self.t_cycle + 1  # Age all consumers within their cycle
        self.t_cycle[self.t_cycle == self.T_cycle] = (
            0  # Resetting to zero for those who have reached the end
        )

    def make_shock_history(self):
        """
        Makes a pre-specified history of shocks for the simulation.  Shock variables should be named
        in self.shock_vars, a list of strings that is subclass-specific.  This method runs a subset
        of the standard simulation loop by simulating only mortality and shocks; each variable named
        in shock_vars is stored in a T_sim x AgentCount array in history dictionary self.history[X].
        Automatically sets self.read_shocks to True so that these pre-specified shocks are used for
        all subsequent calls to simulate().

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
                np.zeros((self.T_sim, self.AgentCount)) + np.nan
            )
        self.shock_history["who_dies"] = np.zeros(
            (self.T_sim, self.AgentCount), dtype=bool
        )

        # Also make blank arrays for the draws of newborns' initial conditions
        for var_name in self.state_vars:
            self.newborn_init_history[var_name] = (
                np.zeros((self.T_sim, self.AgentCount)) + np.nan
            )

        # Record the initial condition of the newborns created by
        # initialize_sim -> sim_births
        for var_name in self.state_vars:
            # Check whether the state is idiosyncratic or an aggregate
            idio = (
                isinstance(self.state_now[var_name], np.ndarray)
                and len(self.state_now[var_name]) == self.AgentCount
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
                        and len(self.state_now[var_name]) == self.AgentCount
                    )
                    if idio:
                        self.newborn_init_history[var_name][t, self.who_dies] = (
                            self.state_now[var_name][self.who_dies]
                        )
                    else:
                        self.newborn_init_history[var_name][t, self.who_dies] = (
                            self.state_now[var_name]
                        )

            # Other Shocks
            self.get_shocks()
            for var_name in self.shock_vars:
                self.shock_history[var_name][t, :] = self.shocks[var_name]

            self.t_sim += 1
            self.t_age = self.t_age + 1  # Age all consumers by one period
            self.t_cycle = self.t_cycle + 1  # Age all consumers within their cycle
            self.t_cycle[self.t_cycle == self.T_cycle] = (
                0  # Resetting to zero for those who have reached the end
            )

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
                            self.state_now[var_name][who_dies] = (
                                self.newborn_init_history[
                                    var_name
                                ][self.t_sim, who_dies]
                            )

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
        who_dies = np.zeros(self.AgentCount, dtype=bool)
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

    def get_shocks(self):
        """
        Gets values of shock variables for the current period.  Does nothing by default, but can
        be overwritten by subclasses of AgentType.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        return None

    def read_shocks_from_history(self):
        """
        Reads values of shock variables for the current period from history arrays.
        For each variable X named in self.shock_vars, this attribute of self is
        set to self.history[X][self.t_sim,:].

        This method is only ever called if self.read_shocks is True.  This can
        be achieved by using the method make_shock_history() (or manually after
        storing a "handcrafted" shock history).

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for var_name in self.shock_vars:
            self.shocks[var_name] = self.shock_history[var_name][self.t_sim, :]

    def get_states(self):
        """
        Gets values of state variables for the current period.
        By default, calls transition function and assigns values
        to the state_now dictionary.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        new_states = self.transition()

        for i, var in enumerate(self.state_now):
            # a hack for now to deal with 'post-states'
            if i < len(new_states):
                self.state_now[var] = new_states[i]

        return None

    def transition(self):
        """

        Parameters
        ----------
        None

        [Eventually, to match dolo spec:
        exogenous_prev, endogenous_prev, controls, exogenous, parameters]

        Returns
        -------

        endogenous_state: ()
            Tuple with new values of the endogenous states
        """

        return ()

    def get_controls(self):
        """
        Gets values of control variables for the current period, probably by using current states.
        Does nothing by default, but can be overwritten by subclasses of AgentType.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        return None

    def get_poststates(self):
        """
        Gets values of post-decision state variables for the current period,
        probably by current
        states and controls and maybe market-level events or shock variables.
        Does nothing by
        default, but can be overwritten by subclasses of AgentType.

        DEPRECATED: New models should use the state now/previous rollover
        functionality instead of poststates.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

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

                for var_name in self.track_vars:
                    if var_name in self.state_now:
                        self.history[var_name][self.t_sim, :] = self.state_now[var_name]
                    elif var_name in self.shocks:
                        self.history[var_name][self.t_sim, :] = self.shocks[var_name]
                    elif var_name in self.controls:
                        self.history[var_name][self.t_sim, :] = self.controls[var_name]
                    else:
                        if var_name == "who_dies" and self.t_sim > 1:
                            self.history[var_name][self.t_sim - 1, :] = getattr(
                                self, var_name
                            )
                        else:
                            self.history[var_name][self.t_sim, :] = getattr(
                                self, var_name
                            )
                self.t_sim += 1

            return self.history

    def clear_history(self):
        """
        Clears the histories of the attributes named in self.track_vars.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for var_name in self.track_vars:
            self.history[var_name] = np.empty((self.T_sim, self.AgentCount))
            self.history[var_name].fill(np.nan)


def solve_agent(agent, verbose):
    """
    Solve the dynamic model for one agent type
    using backwards induction.
    This function iterates on "cycles"
    of an agent's model either a given number of times
    or until solution convergence
    if an infinite horizon model is used
    (with agent.cycles = 0).

    Parameters
    ----------
    agent : AgentType
        The microeconomic AgentType whose dynamic problem
        is to be solved.
    verbose : boolean
        If True, solution progress is printed to screen (when cycles != 1).

    Returns
    -------
    solution : [Solution]
        A list of solutions to the one period problems that the agent will
        encounter in his "lifetime".
    """
    # Check to see whether this is an (in)finite horizon problem
    cycles_left = agent.cycles  # NOQA
    infinite_horizon = cycles_left == 0  # NOQA
    # Initialize the solution, which includes the terminal solution if it's not a pseudo-terminal period
    solution = []
    if not agent.pseudo_terminal:
        solution.insert(0, deepcopy(agent.solution_terminal))

    # Initialize the process, then loop over cycles
    solution_last = agent.solution_terminal  # NOQA
    go = True  # NOQA
    completed_cycles = 0  # NOQA
    max_cycles = 5000  # NOQA  - escape clause
    if verbose:
        t_last = time()
    while go:
        # Solve a cycle of the model, recording it if horizon is finite
        solution_cycle = solve_one_cycle(agent, solution_last)
        if not infinite_horizon:
            solution = solution_cycle + solution

        # Check for termination: identical solutions across
        # cycle iterations or run out of cycles
        solution_now = solution_cycle[0]
        if infinite_horizon:
            if completed_cycles > 0:
                solution_distance = solution_now.distance(solution_last)
                agent.solution_distance = (
                    solution_distance  # Add these attributes so users can
                )
                agent.completed_cycles = (
                    completed_cycles  # query them to see if solution is ready
                )
                go = (
                    solution_distance > agent.tolerance
                    and completed_cycles < max_cycles
                )
            else:  # Assume solution does not converge after only one cycle
                solution_distance = 100.0
                go = True
        else:
            cycles_left += -1
            go = cycles_left > 0

        # Update the "last period solution"
        solution_last = solution_now
        completed_cycles += 1

        # Display progress if requested
        if verbose:
            t_now = time()
            if infinite_horizon:
                print(
                    "Finished cycle #"
                    + str(completed_cycles)
                    + " in "
                    + str(t_now - t_last)
                    + " seconds, solution distance = "
                    + str(solution_distance)
                )
            else:
                print(
                    "Finished cycle #"
                    + str(completed_cycles)
                    + " of "
                    + str(agent.cycles)
                    + " in "
                    + str(t_now - t_last)
                    + " seconds."
                )
            t_last = t_now

    # Record the last cycle if horizon is infinite (solution is still empty!)
    if infinite_horizon:
        solution = (
            solution_cycle  # PseudoTerminal=False impossible for infinite horizon
        )

    return solution


def solve_one_cycle(agent, solution_last):
    """
    Solve one "cycle" of the dynamic model for one agent type.  This function
    iterates over the periods within an agent's cycle, updating the time-varying
    parameters and passing them to the single period solver(s).

    Parameters
    ----------
    agent : AgentType
        The microeconomic AgentType whose dynamic problem is to be solved.
    solution_last : Solution
        A representation of the solution of the period that comes after the
        end of the sequence of one period problems.  This might be the term-
        inal period solution, a "pseudo terminal" solution, or simply the
        solution to the earliest period from the succeeding cycle.

    Returns
    -------
    solution_cycle : [Solution]
        A list of one period solutions for one "cycle" of the AgentType's
        microeconomic model.
    """
    # Calculate number of periods per cycle, defaults to 1 if all variables are time invariant
    if len(agent.time_vary) > 0:
        # name = agent.time_vary[0]
        # T = len(eval('agent.' + name))
        T = len(agent.__dict__[agent.time_vary[0]])
    else:
        T = 1

    solve_dict = {parameter: agent.__dict__[parameter] for parameter in agent.time_inv}
    solve_dict.update({parameter: None for parameter in agent.time_vary})

    # Initialize the solution for this cycle, then iterate on periods
    solution_cycle = []
    solution_next = solution_last

    cycles_range = [0] + list(range(T - 1, 0, -1))
    for k in range(T - 1, -1, -1) if agent.cycles == 1 else cycles_range:
        # Update which single period solver to use (if it depends on time)
        if hasattr(agent.solve_one_period, "__getitem__"):
            solve_one_period = agent.solve_one_period[k]
        else:
            solve_one_period = agent.solve_one_period

        if hasattr(solve_one_period, "solver_args"):
            these_args = solve_one_period.solver_args
        else:
            these_args = get_arg_names(solve_one_period)

        # Update time-varying single period inputs
        for name in agent.time_vary:
            if name in these_args:
                solve_dict[name] = agent.__dict__[name][k]
        solve_dict["solution_next"] = solution_next

        # Make a temporary dictionary for this period
        temp_dict = {name: solve_dict[name] for name in these_args}

        # Solve one period, add it to the solution, and move to the next period
        solution_t = solve_one_period(**temp_dict)
        solution_cycle.insert(0, solution_t)
        solution_next = solution_t

    # Return the list of per-period solutions
    return solution_cycle


def make_one_period_oo_solver(solver_class):
    """
    Returns a function that solves a single period consumption-saving
    problem.
    Parameters
    ----------
    solver_class : Solver
        A class of Solver to be used.
    -------
    solver_function : function
        A function for solving one period of a problem.
    """

    def one_period_solver(**kwds):
        solver = solver_class(**kwds)

        # not ideal; better if this is defined in all Solver classes
        if hasattr(solver, "prepare_to_solve"):
            solver.prepare_to_solve()

        solution_now = solver.solve()
        return solution_now

    one_period_solver.solver_class = solver_class
    # This can be revisited once it is possible to export parameters
    one_period_solver.solver_args = get_arg_names(solver_class.__init__)[1:]

    return one_period_solver


# ========================================================================
# ========================================================================


class Market(Model):
    """
    A superclass to represent a central clearinghouse of information.  Used for
    dynamic general equilibrium models to solve the "macroeconomic" model as a
    layer on top of the "microeconomic" models of one or more AgentTypes.

    Parameters
    ----------
    agents : [AgentType]
        A list of all the AgentTypes in this market.
    sow_vars : [string]
        Names of variables generated by the "aggregate market process" that should
        be "sown" to the agents in the market.  Aggregate state, etc.
    reap_vars : [string]
        Names of variables to be collected ("reaped") from agents in the market
        to be used in the "aggregate market process".
    const_vars : [string]
        Names of attributes of the Market instance that are used in the "aggregate
        market process" but do not come from agents-- they are constant or simply
        parameters inherent to the process.
    track_vars : [string]
        Names of variables generated by the "aggregate market process" that should
        be tracked as a "history" so that a new dynamic rule can be calculated.
        This is often a subset of sow_vars.
    dyn_vars : [string]
        Names of variables that constitute a "dynamic rule".
    mill_rule : function
        A function that takes inputs named in reap_vars and returns a tuple the same size and order as sow_vars.  The "aggregate market process" that
        transforms individual agent actions/states/data into aggregate data to
        be sent back to agents.
    calc_dynamics : function
        A function that takes inputs named in track_vars and returns an object
        with attributes named in dyn_vars.  Looks at histories of aggregate
        variables and generates a new "dynamic rule" for agents to believe and
        act on.
    act_T : int
        The number of times that the "aggregate market process" should be run
        in order to generate a history of aggregate variables.
    tolerance: float
        Minimum acceptable distance between "dynamic rules" to consider the
        Market solution process converged.  Distance is a user-defined metric.
    """

    def __init__(
        self,
        agents=None,
        sow_vars=None,
        reap_vars=None,
        const_vars=None,
        track_vars=None,
        dyn_vars=None,
        mill_rule=None,
        calc_dynamics=None,
        act_T=1000,
        tolerance=0.000001,
        **kwds,
    ):
        super().__init__()
        self.agents = agents if agents is not None else list()  # NOQA

        self.reap_vars = reap_vars if reap_vars is not None else list()  # NOQA
        self.reap_state = {var: [] for var in self.reap_vars}

        self.sow_vars = sow_vars if sow_vars is not None else list()  # NOQA
        # dictionaries for tracking initial and current values
        # of the sow variables.
        self.sow_init = {var: None for var in self.sow_vars}
        self.sow_state = {var: None for var in self.sow_vars}

        const_vars = const_vars if const_vars is not None else list()  # NOQA
        self.const_vars = {var: None for var in const_vars}

        self.track_vars = track_vars if track_vars is not None else list()  # NOQA
        self.dyn_vars = dyn_vars if dyn_vars is not None else list()  # NOQA

        if mill_rule is not None:  # To prevent overwriting of method-based mill_rules
            self.mill_rule = mill_rule
        if calc_dynamics is not None:  # Ditto for calc_dynamics
            self.calc_dynamics = calc_dynamics
        self.act_T = act_T  # NOQA
        self.tolerance = tolerance  # NOQA
        self.max_loops = 1000  # NOQA
        self.history = {}
        self.assign_parameters(**kwds)

        self.print_parallel_error_once = True
        # Print the error associated with calling the parallel method
        # "solve_agents" one time. If set to false, the error will never
        # print. See "solve_agents" for why this prints once or never.

    def solve_agents(self):
        """
        Solves the microeconomic problem for all AgentTypes in this market.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        try:
            multi_thread_commands(self.agents, ["solve()"])
        except Exception as err:
            if self.print_parallel_error_once:
                # Set flag to False so this is only printed once.
                self.print_parallel_error_once = False
                print(
                    "**** WARNING: could not execute multi_thread_commands in HARK.core.Market.solve_agents() ",
                    "so using the serial version instead. This will likely be slower. "
                    "The multiTreadCommands() functions failed with the following error:",
                    "\n",
                    sys.exc_info()[0],
                    ":",
                    err,
                )  # sys.exc_info()[0])
            multi_thread_commands_fake(self.agents, ["solve()"])

    def solve(self):
        """
        "Solves" the market by finding a "dynamic rule" that governs the aggregate
        market state such that when agents believe in these dynamics, their actions
        collectively generate the same dynamic rule.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        go = True
        max_loops = self.max_loops  # Failsafe against infinite solution loop
        completed_loops = 0
        old_dynamics = None

        while go:  # Loop until the dynamic process converges or we hit the loop cap
            self.solve_agents()  # Solve each AgentType's micro problem
            self.make_history()  # "Run" the model while tracking aggregate variables
            new_dynamics = self.update_dynamics()  # Find a new aggregate dynamic rule

            # Check to see if the dynamic rule has converged (if this is not the first loop)
            if completed_loops > 0:
                distance = new_dynamics.distance(old_dynamics)
            else:
                distance = 1000000.0

            # Move to the next loop if the terminal conditions are not met
            old_dynamics = new_dynamics
            completed_loops += 1
            go = distance >= self.tolerance and completed_loops < max_loops

        self.dynamics = new_dynamics  # Store the final dynamic rule in self

    def reap(self):
        """
        Collects attributes named in reap_vars from each AgentType in the market,
        storing them in respectively named attributes of self.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        for var in self.reap_state:
            harvest = []

            for agent in self.agents:
                # TODO: generalized variable lookup across namespaces
                if var in agent.state_now:
                    # or state_now ??
                    harvest.append(agent.state_now[var])

            self.reap_state[var] = harvest

    def sow(self):
        """
        Distributes attrributes named in sow_vars from self to each AgentType
        in the market, storing them in respectively named attributes.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        for sow_var in self.sow_state:
            for this_type in self.agents:
                if sow_var in this_type.state_now:
                    this_type.state_now[sow_var] = self.sow_state[sow_var]
                if sow_var in this_type.shocks:
                    this_type.shocks[sow_var] = self.sow_state[sow_var]
                else:
                    setattr(this_type, sow_var, self.sow_state[sow_var])

    def mill(self):
        """
        Processes the variables collected from agents using the function mill_rule,
        storing the results in attributes named in aggr_sow.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        # Make a dictionary of inputs for the mill_rule
        mill_dict = copy(self.reap_state)
        mill_dict.update(self.const_vars)

        # Run the mill_rule and store its output in self
        product = self.mill_rule(**mill_dict)

        for i, sow_var in enumerate(self.sow_state):
            self.sow_state[sow_var] = product[i]

    def cultivate(self):
        """
        Has each AgentType in agents perform their market_action method, using
        variables sown from the market (and maybe also "private" variables).
        The market_action method should store new results in attributes named in
        reap_vars to be reaped later.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        for this_type in self.agents:
            this_type.market_action()

    def reset(self):
        """
        Reset the state of the market (attributes in sow_vars, etc) to some
        user-defined initial state, and erase the histories of tracked variables.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        # Reset the history of tracked variables
        self.history = {var_name: [] for var_name in self.track_vars}

        # Set the sow variables to their initial levels
        for var_name in self.sow_state:
            self.sow_state[var_name] = self.sow_init[var_name]

        # Reset each AgentType in the market
        for this_type in self.agents:
            this_type.reset()

    def store(self):
        """
        Record the current value of each variable X named in track_vars in an
        dictionary field named history[X].

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        for var_name in self.track_vars:
            if var_name in self.sow_state:
                value_now = self.sow_state[var_name]
            elif var_name in self.reap_state:
                value_now = self.reap_state[var_name]
            elif var_name in self.const_vars:
                value_now = self.const_vars[var_name]
            else:
                value_now = getattr(self, var_name)

            self.history[var_name].append(value_now)

    def make_history(self):
        """
        Runs a loop of sow-->cultivate-->reap-->mill act_T times, tracking the
        evolution of variables X named in track_vars in dictionary fields
        history[X].

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        self.reset()  # Initialize the state of the market
        for t in range(self.act_T):
            self.sow()  # Distribute aggregated information/state to agents
            self.cultivate()  # Agents take action
            self.reap()  # Collect individual data from agents
            self.mill()  # Process individual data into aggregate data
            self.store()  # Record variables of interest

    def update_dynamics(self):
        """
        Calculates a new "aggregate dynamic rule" using the history of variables
        named in track_vars, and distributes this rule to AgentTypes in agents.

        Parameters
        ----------
        none

        Returns
        -------
        dynamics : instance
            The new "aggregate dynamic rule" that agents believe in and act on.
            Should have attributes named in dyn_vars.
        """
        # Make a dictionary of inputs for the dynamics calculator
        arg_names = list(get_arg_names(self.calc_dynamics))
        if "self" in arg_names:
            arg_names.remove("self")
        update_dict = {name: self.history[name] for name in arg_names}
        # Calculate a new dynamic rule and distribute it to the agents in agent_list
        dynamics = self.calc_dynamics(**update_dict)  # User-defined dynamics calculator
        for var_name in self.dyn_vars:
            this_obj = getattr(dynamics, var_name)
            for this_type in self.agents:
                setattr(this_type, var_name, this_obj)
        return dynamics


def distribute_params(agent, param_name, param_count, distribution):
    """
    Distributes heterogeneous values of one parameter to the AgentTypes in self.agents.
    Parameters
    ----------
    agent: AgentType
        An agent to clone.
    param_name : string
        Name of the parameter to be assigned.
    param_count : int
        Number of different values the parameter will take on.
    distribution : Distribution
        A 1-D distribution.

    Returns
    -------
    agent_set : [AgentType]
        A list of param_count agents, ex ante heterogeneous with
        respect to param_name. The AgentCount of the original
        will be split between the agents of the returned
        list in proportion to the given distribution.
    """
    param_dist = distribution.discretize(N=param_count)

    agent_set = [deepcopy(agent) for i in range(param_count)]

    for j in range(param_count):
        agent_set[j].assign_parameters(
            **{"AgentCount": int(agent.AgentCount * param_dist.pmv[j])}
        )
        # agent_set[j].__dict__[param_name] = param_dist.atoms[j]

        agent_set[j].assign_parameters(**{param_name: param_dist.atoms[0, j]})

    return agent_set


@dataclass
class AgentPopulation:
    """
    A class for representing a population of ex-ante heterogeneous agents.
    """

    agent_type: AgentType  # type of agent in the population
    parameters: dict  # dictionary of parameters
    seed: int = 0  # random seed
    time_var: List[str] = field(init=False)
    time_inv: List[str] = field(init=False)
    distributed_params: List[str] = field(init=False)
    agent_type_count: Optional[int] = field(init=False)
    term_age: Optional[int] = field(init=False)
    continuous_distributions: Dict[str, Distribution] = field(init=False)
    discrete_distributions: Dict[str, Distribution] = field(init=False)
    population_parameters: List[Dict[str, Union[List[float], float]]] = field(
        init=False
    )
    agents: List[AgentType] = field(init=False)
    agent_database: pd.DataFrame = field(init=False)
    solution: List[Any] = field(init=False)

    def __post_init__(self):
        """
        Initialize the population of agents, determine distributed parameters,
        and infer `agent_type_count` and `term_age`.
        """
        # create a dummy agent and obtain its time-varying
        # and time-invariant attributes
        dummy_agent = self.agent_type()
        self.time_var = dummy_agent.time_vary
        self.time_inv = dummy_agent.time_inv

        # create list of distributed parameters
        # these are parameters that differ across agents
        self.distributed_params = [
            key
            for key, param in self.parameters.items()
            if (isinstance(param, list) and isinstance(param[0], list))
            or isinstance(param, Distribution)
            or (isinstance(param, DataArray) and param.dims[0] == "agent")
        ]

        self.__infer_counts__()

    def __infer_counts__(self):
        """
        Infer `agent_type_count` and `term_age` from the parameters.
        If parameters include a `Distribution` type, a list of lists,
        or a `DataArray` with `agent` as the first dimension, then
        the AgentPopulation contains ex-ante heterogenous agents.
        """

        # infer agent_type_count from distributed parameters
        agent_type_count = 1
        for key in self.distributed_params:
            param = self.parameters[key]
            if isinstance(param, Distribution):
                agent_type_count = None
                warn(
                    "Cannot infer agent_type_count from a Distribution. "
                    "Please provide approximation parameters."
                )
                break
            elif isinstance(param, list):
                agent_type_count = max(agent_type_count, len(param))
            elif isinstance(param, DataArray) and param.dims[0] == "agent":
                agent_type_count = max(agent_type_count, param.shape[0])

        self.agent_type_count = agent_type_count

        # infer term_age from all parameters
        term_age = 1
        for param in self.parameters.values():
            if isinstance(param, Distribution):
                term_age = None
                warn(
                    "Cannot infer term_age from a Distribution. "
                    "Please provide approximation parameters."
                )
                break
            elif isinstance(param, list) and isinstance(param[0], list):
                term_age = max(term_age, len(param[0]))
            elif isinstance(param, DataArray) and param.dims[-1] == "age":
                term_age = max(term_age, param.shape[-1])

        self.term_age = term_age

    def approx_distributions(self, approx_params: dict):
        """
        Approximate continuous distributions with discrete ones. If the initial
        parameters include a `Distribution` type, then the AgentPopulation is
        not ready to solve, and stands for an abstract population. To solve the
        AgentPopulation, we need discretization parameters for each continuous
        distribution. This method approximates the continuous distributions with
        discrete ones, and updates the parameters dictionary.
        """
        self.continuous_distributions = {}
        self.discrete_distributions = {}

        for key, points in approx_params.items():
            param = self.parameters[key]
            if key in self.distributed_params and isinstance(param, Distribution):
                self.continuous_distributions[key] = param
                self.discrete_distributions[key] = param.discretize(points)
            else:
                raise ValueError(
                    f"Warning: parameter {key} is not a Distribution found "
                    f"in agent type {self.agent_type}"
                )

        if len(self.discrete_distributions) > 1:
            joint_dist = combine_indep_dstns(*self.discrete_distributions.values())
        else:
            joint_dist = list(self.discrete_distributions.values())[0]

        for i, key in enumerate(self.discrete_distributions):
            self.parameters[key] = DataArray(joint_dist.atoms[i], dims=("agent"))

        self.__infer_counts__()

    def __parse_parameters__(self) -> None:
        """
        Creates distributed dictionaries of parameters for each ex-ante
        heterogeneous agent in the parameterized population. The parameters
        are stored in a list of dictionaries, where each dictionary contains
        the parameters for one agent. Expands parameters that vary over time
        to a list of length `term_age`.
        """

        population_parameters = []  # container for dictionaries of each agent subgroup
        for agent in range(self.agent_type_count):
            agent_parameters = {}
            for key, param in self.parameters.items():
                if key in self.time_var:
                    # parameters that vary over time have to be repeated
                    if isinstance(param, (int, float)):
                        parameter_per_t = [param] * self.term_age
                    elif isinstance(param, list):
                        if isinstance(param[0], list):
                            parameter_per_t = param[agent]
                        else:
                            parameter_per_t = param
                    elif isinstance(param, DataArray):
                        if param.dims[0] == "agent":
                            if param.dims[-1] == "age":
                                parameter_per_t = param[agent].item()
                            else:
                                parameter_per_t = param.item()
                        elif param.dims[0] == "age":
                            parameter_per_t = param.item()

                    agent_parameters[key] = parameter_per_t

                elif key in self.time_inv:
                    if isinstance(param, (int, float)):
                        agent_parameters[key] = param
                    elif isinstance(param, list):
                        if isinstance(param[0], list):
                            agent_parameters[key] = param[agent]
                        else:
                            agent_parameters[key] = param
                    elif isinstance(param, DataArray) and param.dims[0] == "agent":
                        agent_parameters[key] = param[agent].item()

                else:
                    if isinstance(param, (int, float)):
                        agent_parameters[key] = param  # assume time inv
                    elif isinstance(param, list):
                        if isinstance(param[0], list):
                            agent_parameters[key] = param[agent]  # assume agent vary
                        else:
                            agent_parameters[key] = param  # assume time vary
                    elif isinstance(param, DataArray):
                        if param.dims[0] == "agent":
                            if param.dims[-1] == "age":
                                agent_parameters[key] = param[
                                    agent
                                ].item()  # assume agent vary
                            else:
                                agent_parameters[key] = param.item()  # assume time vary
                        elif param.dims[0] == "age":
                            agent_parameters[key] = param.item()  # assume time vary

            population_parameters.append(agent_parameters)

        self.population_parameters = population_parameters

    def create_distributed_agents(self):
        """
        Parses the parameters dictionary and creates a list of agents with the
        appropriate parameters. Also sets the seed for each agent.
        """

        self.__parse_parameters__()

        rng = np.random.default_rng(self.seed)

        self.agents = [
            self.agent_type(seed=rng.integers(0, 2**31 - 1), **agent_dict)
            for agent_dict in self.population_parameters
        ]

    def create_database(self):
        """
        Optionally creates a pandas DataFrame with the parameters for each agent.
        """
        database = pd.DataFrame(self.population_parameters)
        database["agents"] = self.agents

        self.agent_database = database

    def solve(self):
        """
        Solves each agent of the population serially.
        """

        # see Market class for an example of how to solve distributed agents in parallel

        for agent in self.agents:
            agent.solve()

    def unpack_solutions(self):
        """
        Unpacks the solutions of each agent into an attribute of the population.
        """
        self.solution = [agent.solution for agent in self.agents]

    def initialize_sim(self):
        """
        Initializes the simulation for each agent.
        """
        for agent in self.agents:
            agent.initialize_sim()

    def simulate(self):
        """
        Simulates each agent of the population serially.
        """
        for agent in self.agents:
            agent.simulate()

    def __iter__(self):
        """
        Allows for iteration over the agents in the population.
        """
        return iter(self.agents)

    def __getitem__(self, idx):
        """
        Allows for indexing into the population.
        """
        return self.agents[idx]

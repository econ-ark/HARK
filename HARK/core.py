"""
High-level functions and classes for solving a wide variety of economic models.
The "core" of HARK is a framework for "microeconomic" and "macroeconomic"
models.  A micro model concerns the dynamic optimization problem for some type
of agents, where agents take the inputs to their problem as exogenous.  A macro
model adds an additional layer, endogenizing some of the inputs to the micro
problem by finding a general equilibrium dynamic rule.
"""
import sys
import os
from distutils.dir_util import copy_tree
from .utilities import get_arg_names, NullFunc
from copy import copy, deepcopy
import numpy as np
from time import time
from .parallel import multi_thread_commands, multi_thread_commands_fake
from warnings import warn

# pyflakes kept flagging _log commands in core.py as undefined unless the log stuff
# was here at the beginning (and not just in __init__.py

import logging

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


def core_check_condition(name, test, messages, messaging_level, verbose_messages, fact, stage_solution, quietly=False):
    """
    Checks whether parameter values of a model satisfy a condition

    Parameters
    ----------
    name : string
         Name for the condition.

    test : function(self -> boolean)
         A function (of self) which tests the condition

    verbose : Boolean
         If False, print minimal information about the condition and exit
         If True, print more detailed information and a reference

    messages : dict{boolean : string}
        A dictionary with boolean keys containing values
        for messages to print if the condition is
        true or false.

    messaging_level : int
        Controls verbosity of messages. logging.DEBUG is most verbose,
        logging.INFO is less verbose, logging.WARNING speaks up only if the
        model might have problems, logging.CRITICAL indicates it is degenerate.

    verbose_messages : dict{boolean : string}
        (Optional) A dictionary with boolean keys containing supplemental
        messages to print if the condition is
        true or false if messaging_level is logging.DEBUG

    fact : string
         Name of the fact (for recording the results)

    stage_solution : solution for a stage of the problem containing a condition 
        to be tested.  Must have dict 'conditions' [name]

    quiet : boolean
        If True, construct the conditions but print nothing
    """

    soln = stage_solution

    TF = test(soln)
    soln.Bilt.conditions[name] = TF  # Set whether condition was true or false
    # Messages of severity less than minimum intensity are not displayed
    minimum_intensity = messaging_level  # Do not show messages less important
    _log.setLevel(minimum_intensity)  # Uses python logger setLevel
    # Store the messages for later retrieval if desired
    soln.Bilt.conditions[fact] = (
        messages[soln.Bilt.conditions[name]] +
        verbose_messages[soln.Bilt.conditions[name]]).format(soln.Bilt)
    _log.info(messages[soln.Bilt.conditions[name]].format(soln.Bilt))
    # Debug log gets the verbose info
    _log.debug(
        verbose_messages[soln.Bilt.conditions[name]].format(soln.Bilt))

    return TF


def distance_metric(thing_a, thing_b):
    """
    A "universal distance" metric that can be used as a default in many settings.

    Parameters
    ----------
    thing_a : object
        A generic object.
    thing_b : object
        Another generic object.

    Returns:
    ------------
    distance : float
        The "distance" between thing_a and thing_b.
    """
    # Get the types of the two inputs
    type_a = type(thing_a)
    type_b = type(thing_b)

    if type_a is list and type_b is list:
        len_a = len(thing_a)  # If both inputs are lists, then the distance between
        len_b = len(thing_b)  # them is the maximum distance between corresponding
        if len_a == len_b:  # elements in the lists.  If they differ in length,
            distance_temp = []  # the distance is the difference in lengths.
            for n in range(len_a):
                distance_temp.append(distance_metric(thing_a[n], thing_b[n]))
            distance = max(distance_temp)
        else:
            warn(
                'Objects of different lengths are being compared. ' +
                'Returning difference in lengths.'
                )
            distance = float(abs(len_a - len_b))
    # If both inputs are dictionaries, call distance on the list of its elements
    elif type_a is dict and type_b is dict:

        len_a = len(thing_a)
        len_b = len(thing_b)

        if len_a == len_b:

            # Create versions sorted by key
            sorted_a = dict(sorted(thing_a.items()))
            sorted_b = dict(sorted(thing_b.items()))

            # If keys don't match, print a warning.
            if list(sorted_a.keys()) != list(sorted_b.keys()):
                warn(
                    'Dictionaries with keys that do not match are being ' +
                    'compared.'
                )

            distance = distance_metric(list(sorted_a.values()),
                                       list(sorted_b.values()))

        else:
            # If they have different lengths, log a warning and return the
            # difference in lengths.
            warn(
                'Objects of different lengths are being compared. ' +
                'Returning difference in lengths.'
            )
            distance = float(abs(len_a - len_b))

    # If both inputs are numbers, return their difference
    elif isinstance(thing_a, (int, float)) and isinstance(thing_b, (int, float)):
        distance = float(abs(thing_a - thing_b))
    # If both inputs are array-like, return the maximum absolute difference b/w
    # corresponding elements (if same shape); return largest difference in dimensions
    # if shapes do not align.
    elif hasattr(thing_a, "shape") and hasattr(thing_b, "shape"):
        if thing_a.shape == thing_b.shape:
            distance = np.max(abs(thing_a - thing_b))
        else:
            # Flatten arrays so they have the same dimensions
            distance = np.max(
                abs(thing_a.flatten().shape[0] - thing_b.flatten().shape[0])
            )
    # If none of the above cases, but the objects are of the same class, call
    # the distance method of one on the other
    elif thing_a.__class__.__name__ == thing_b.__class__.__name__:
        if thing_a.__class__.__name__ == "function":
            distance = 0.0
        else:
            distance = thing_a.distance(thing_b)
    else:  # Failsafe: the inputs are very far apart
        distance = 1000.0
    return distance


class MetricObject(object):
    """
    A superclass for object classes in HARK.  Comes with two useful methods:
    a generic/universal distance method and an attribute assignment method.
    """

    distance_criteria = []  # This should be overwritten by subclasses.

    def distance(self, other):
        """
        A generic distance method, which requires the existence of an attribute
        called distance_criteria, giving a list of strings naming the attributes
        to be considered by the distance metric.

        Parameters
        ----------
        other : object
            Another object to compare this instance to.

        Returns
        -------
        (unnamed) : float
            The distance between this object and another, using the "universal
            distance" metric.
        """
        if self.distance_criteria == []:
            _log.critical(
                "distance_criteria not set so distance between "+str(self)+" and " + str(other) + " cannot be computed.")

        distance_list = [0.0]

        for attr_name in self.distance_criteria:
            try:
                obj_a = getattr(self, attr_name)
                obj_b = getattr(other, attr_name)
                distance_list.append(distance_metric(obj_a, obj_b))
            except AttributeError:
                distance_list.append(
                    1000.0
                )  # if either object lacks attribute, they are not the same
        self.distance_last = max(distance_list)  # Store it for later inspection
        return max(distance_list)


class Model(object):
    """
    A class with special handling of parameters assignment.

    Creates a "self.parameters" attribute that contains all
    keywords explicitly passed to the object.
    """

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

    assign_parameters_model = assign_parameters

    def get_parameter(self, name):
        # CDC 20210527: A global search across all HARK, REMARKs,
        # and DemARKs does not find any instance of this being used.
        # We ought either to:
        # 1. use it pervasively
        #    * in which case there should be a companion set_parameter
        # or
        # 2. Eliminate it
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

    get_parameter_model = get_parameter

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.parameters == other.parameters

        return notImplemented

    def __init__(self):
        if not hasattr(self, 'parameters'):
            self.parameters = {}

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

    def __repr__(self):
        return self.__str__()


class AgentType(Model):
    """
    A superclass for economic agents in the HARK framework. Each model should
    specify its own subclass of AgentType, inheriting  methods and overwriting
    as necessary.  Critically, every subclass of AgentType should define class-
    specific static values of the attributes time_vary and time_inv as lists of
    strings.  Each element of time_vary is the name of a field in AgentSubType
    that varies over time in the model. Each element of time_inv is the name of
    a field in AgentSubType that is constant over time in the model.

    Parameters
    ----------
    solution_terminal : Solution
        A representation of the solution to the terminal period/stage of
        this AgentType instance, or an initial guess of the solution if this
        is an infinite horizon problem.
    cycles : int
        The number of times the sequence of periods/stages is experienced by
        this AgentType in their lifetime. cycles=1 corresponds to a lifecycle
        model, with a given sequence of one period/stage problems experienced
        once before terminating.  cycles=0 corresponds to an infinite horizon
        model, with a sequence of one period/stage problems repeating
        indefinitely.
    pseudo_terminal : boolean
        Indicates whether solution_terminal isn't actually part of the
        solution to the problem (as a known solution to the terminal period
        problem), but instead represents a "scrap value"-style termination.
        When True, solution_terminal is not included in the solution; when
        false, solution_terminal is the last element of the solution.
    tolerance : float
        Maximum acceptable "distance" between successive solutions to the one
        period/stage problem in an infinite horizon (cycles=0) model in order
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
        **kwds
    ):
        Model.__init__(self)

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
        self.history = {}

        self.assign_parameters(**kwds)  # NOQA
        self.reset_rng()  # NOQA
        self.prmtv_par = dict()  # 'primitives' define true model
        # 'approximation' pars
        self.aprox_par = {'tolerance': self.tolerance, 'seed': self.seed}
        # if lim exists, sol approaches truth as all aprox_par -> lim
        self.aprox_lim = {'tolerance': 0.0, 'seed': None}

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
            Any number of strings naming attributes to remove from time_vary

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

    # At present, the only solution method is EGM
    # The next method to add is value function iteration
    def solve(self, messaging_level=logging.DEBUG, quietly=False, **kwds):  # AgentType
        """
        Solve the model for this instance of an agent by backward induction.
        Loops through sequence of one period/stage problems, passing solution
        from next period/stage to the problem for current period/stage.

        Parameters
        ----------
        messaging_level : logging level
            From logging.DEBUG to logging.CRITICAL

        quietly : boolean
            If True, suppress output

        Returns
        -------
        none
        """
        self.messaging_level = messaging_level
        self.quietly = quietly
        # Ignore floating point "errors". Numpy calls it "errors", but really it's excep-
        # tions with well-defined answers such as 1.0/0.0 that is np.inf, -1.0/0.0 that is
        # -np.inf, np.inf/np.inf is np.nan and so on.
        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            self.pre_solve()  # Stuff to do before beginning to solve the model
            self.solution = solve_agent(
                self, messaging_level, quietly, **kwds
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
        self.RNG = np.random.RandomState(self.seed)

    reset_rng_mcrlo = mcrlo_reset_rng = reset_rng

    def check_elements_of_time_vary_are_lists(self):
        """
        A method to check that elements of time_vary are lists.
        """
        for param in self.time_vary:
            assert type(getattr(self, param)) == list, (
                param + " is not a list, but should be" + " because it is in time_vary"
            )

    def check_restrictions(self):
        """
        Check that various restrictions are met for the model class.
        """
        return

    def pre_solve(self, quietly):
        """Check agent immediately before beginning embarking on solution.

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

    pre_solve_agent = pre_solve

    def post_solve(self):
        """
        An agent that is run immediately after the model is solved, to finalize
        the solution in some way.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        return None

    post_solve_agent = post_solve

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
        )  # Number of periods/stages since agent entry
        self.t_cycle = np.zeros(
            self.AgentCount, dtype=int
        )  # Which cycle period/stage each agent is on
        self.sim_birth(all_agents)
        self.clear_history()
        return None

    mcrlo_initialize_sim = initialize_sim_mcrlo = initialize_sim

    def sim_one_period(self):  # -> mcrlo_sim_one_prd
        """
        Simulates one period/stage for this type.  Calls the methods get_mortality(), get_shocks() or
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
        self.get_mortality()  # -> mcrlo_get_mortality Replace some with newborns

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
        self.t_cycle[
            self.t_cycle == self.T_cycle
        ] = 0  # Resetting to zero for those who have reached the end

    mcrlo_sim_one_stge = sim_one_period

    def make_shock_history(self):  # -> make_shock_hst
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
        # Make and store the history of shocks for each period
        for t in range(self.T_sim):
            self.get_mortality()
            self.shock_history["who_dies"][t, :] = self.who_dies
            self.get_shocks()
            for var_name in self.shock_vars:
                self.shock_history[var_name][self.t_sim, :] = self.shocks[var_name]

            self.t_sim += 1
            self.t_age = self.t_age + 1  # Age all consumers by one period
            self.t_cycle = self.t_cycle + 1  # Age all consumers within their cycle
            self.t_cycle[
                self.t_cycle == self.T_cycle
            ] = 0  # Resetting to zero for those who have reached the end

        # Flag that shocks can be read rather than simulated
        self.read_shocks = True

    make_shock_history_mcrlo = mcrlo_make_shocks_history = make_shock_history

    def get_mortality(self):  # -> mcrlo_get_mrtlty
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
        else:
            who_dies = self.sim_death()
        self.sim_birth(who_dies)
        self.who_dies = who_dies
        return None

    mcrlo_get_mortality = get_mortality_mcrlo = get_mortality

    def sim_death(self):  # -> mcrlo_sim_deth
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

    def sim_birth(self, which_agents):  # -> mcrlo_sim_brth
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

    def get_shocks(self):  # -> mcrlo_get_shks
        """
        Get values of shock variables for the current stage/period.  

        Does nothing by default, but can be overwritten by subclasses of
        AgentType.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        return None

    mcrlo_get_shocks = get_shocks_mcrlo = get_shocks

    def read_shocks_from_history(self):  # -> mcrlo_hstry_shks_read
        """
        Reads values of shock variables for the current period/stage from history arrays.
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

    mcrlo_read_shocks_from_history = read_shocks_from_history_mcrlo = \
        read_shocks_from_history

    def get_states(self):  # -> mcrlo_ get_stts
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

    mcrlo_get_states = get_states_mcrlo = get_states

    def transition(self):  # -> mcrlo_trnstn = inherit everything
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

    transition_mcrlo = mcrlo_transition = transition

    def get_controls(self):
        """
        Gets values of control variables for the current period/stage, probably by using current states.
        Does nothing by default, but can be overwritten by subclasses of AgentType.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        return None

    get_controls_mcrlo = mcrlo_get_controls = get_controls

    def get_poststates(self):
        """
        Gets values of post-decision state variables for the current period/stage,
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

    get_poststates_mcrlo = mcrlo_get_poststates = get_poststates

    def simulate(self, sim_periods=None):  # -> sim_stage=None
        """
        Simulates this agent type for a given number of periods/stages. Defaults to
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
                        self.history[var_name][self.t_sim, :] = self.state_now[
                            var_name
                        ]
                    elif var_name in self.shocks:
                        self.history[var_name][self.t_sim, :] = self.shocks[var_name]
                    elif var_name in self.controls:
                        self.history[var_name][self.t_sim, :] = self.controls[var_name]
                    else:
                        self.history[var_name][self.t_sim, :] = getattr(self, var_name)
                self.t_sim += 1

            return self.history

    simulate_mcrlo = mcrlo_simulate = simulate

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
            self.history[var_name] = np.empty((self.T_sim, self.AgentCount)) + np.nan

    clear_history_mcrlo = mcrlo_clear_history = clear_history


def solve_agent(agent, messaging_level, quietly=False, **kwds):
    """
    Solve the dynamic model for one agent type using backwards induction.

    Iterates on "cycles" of an agent's model either a given number
    of times or until solution convergence if an infinite horizon model is used
    (with agent.cycles = 0).

    Parameters
    ----------
    agent : AgentType
        The microeconomic AgentType whose dynamic problem
        is to be solved.
    messaging_level : logging level
        Controls amount of output
    quietly : boolean
        If True, all printed output is suppressed.

    Returns
    -------
    solution : [Solution]
        A list of solutions to one period/stage problems that the agent will
        encounter in a "lifetime".
    """
    # Check to see whether this is an (in)finite horizon problem
    cycles_left = agent.cycles  # NOQA
    infinite_horizon = cycles_left == 0  # NOQA
    # If we are resuming a solution process, a solution might already exist
    if not hasattr(agent, 'solution'):  # Not resuming, create anew
        # Initialize the solution, which includes the terminal solution
        solution = []
        pseudo = (agent.pseudo_terminal is True)
        if not pseudo:  # Then it's a real solution; should be part of the list
            solution.insert(0, deepcopy(agent.solution_terminal))
        completed_cycles = 0  # NOQA
        max_cycles = 5000  # NOQA  - escape clause, stop eventually
        solution_next = agent.solution_terminal  # NOQA
        solution_next.cFunc = solution_next.Bilt.cFunc  # Should be generic dr
    else:  # We are resuming solution of a model that has already converged
        solution = agent.solution
        solution_next = agent.solution[0]
        if hasattr(solution_next, 'completed_cycles'):  # keep counting
            completed_cycles = solution_next.completed_cycles
        else:  # start counting
            completed_cycles = solution_next.completed_cycles = 0
        if hasattr(agent, 'max_cycles'):  # keep max
            max_cycles = agent.max_cycles
        else:
            max_cycles = 5000  # stop eventually

    if hasattr(solution_next, 'stge_kind'):
        # User who wants to resume, say to solve to tighter tolerance, needs
        # of course to change the tolerance parameter but must also change the
        # stge_kind from 'finished' to 'iterator'
        if 'iter_status' in solution_next.stge_kind:
            if solution_next.stge_kind['iter_status'] == 'finished':
                _log.info('The model has already been solved.')
#                print('The existing solution solves the problem')
                return agent.solution

    # Initialize the process, then loop over cycles
    go = True  # NOQA

    if messaging_level < logging.INFO:  # INFO or DEBUG or NOTSET
        t_last = time()
    while go:          # Solve a cycle of the model
        solution_cycle = solve_one_cycle(agent, solution_next)
        solution_now = solution_cycle[0]
        if not infinite_horizon:
            # If finite horizon model, add cycle to the growing list
            solution = solution_cycle + solution
            cycles_left += -1
            go = cycles_left > 0
            # Don't count replacement of terminal_partial as a cycle; see below
            if solution_next.Bilt.stge_kind['iter_status'] == 'terminal_partial':
                cycles_left += 1
                completed_cycles += -1
                go = True
        else:  # infinite horizon
            solution = solution_cycle
            solution_now = solution_cycle[0]  # element 0 most recently solved
            solution_now.solution_distance = \
                solution_distance = solution_now.distance(solution_next)
            go = (
                solution_distance > agent.tolerance
                and completed_cycles < max_cycles
            )
            # To permit ".solve()" method on the agent to resume solution after
            # reaching convergence under some initial tolerance, user must
            # * Change something about the problem (e.g., tolerance)
            # * Change solution[0] stge_kind['iter_status'] to 'iterator'
            # * Restart the solution process with [instance].solve()
            # TODO: this has been tested only for PerfForesightConsumerType
            # and IndShockConsumerType.  We should test whether agent is one of
            # those before allowing resume

            if hasattr(agent, 'solve_resume') and (agent.solve_resume is True):  # if resumption requested,
                go = True  # solve one period/stage for sure, then keep going
                agent.solve_resume = False  # go until stop criteria satisfied
            if not go:  # Finished solving
                # All models should incorporate 'stge_kind', but some have not
                # Handle cases where that has not yet been implemented:
                if not hasattr(solution_now.Bilt, 'stge_kind'):
                    solution_now.Bilt.stge_kind = {'iter_status': 'iterator'}
                if solution_next.Bilt.stge_kind['iter_status'] == 'terminal_partial':
                    completed_cycles += -1  # replacement is not a cycle
                else:  # Replacing terminal_partial is not a "real" cycle
                    # This if/else prevents a stage derived from one marked as
                    # 'terminal_partial' from being labeled as
                    # 'finished' even though its distance will be zero
                    # from the 'terminal_partial' stage. Lets us use
                    # our machinery to enrich the terminal_partial stage
                    solution_now.Bilt.stge_kind['iter_status'] = 'finished'
                    # Record the tolerance that was satisfied
                    solution_now.stge_kind['tolerance'] = agent.tolerance
        # Update the "last period/stage solution" for next iteration
        completed_cycles += 1
        solution_now.completed_cycles = deepcopy(completed_cycles)
        solution_next = solution_now

        if not quietly:
            if messaging_level >= logging.DEBUG:
                print('.', end='')  # visually indicate something is happening
            else:  # Display progress
                t_now = time()
                if infinite_horizon:
                    print(
                        "Finished cycle # "
                        + str(completed_cycles).zfill(6)
                        + " in "
                        + str("{:9.6f}".format(t_now - t_last))
                        + " seconds, solution distance = "
                        + str("{:.3e}".format(solution_distance))
                    )
                else:
                    print(
                        "Finished cycle # "
                        + str(completed_cycles).zfill(len(str(agent.cycles)))
                        + " of "
                        + str(agent.cycles)
                        + " in "
                        + str("{:9.6f}".format(t_now - t_last))
                        + " seconds."
                    )
                    t_last = t_now

    return solution


def solve_one_cycle(agent, solution_next):
    """
    Solve one "cycle" of the dynamic model for one agent type. 

    This function iterates over the stages within an agent's cycle, updating
    the stage-varying parameters and passing them to the single-stage
    solver(s).

    Parameters
    ----------
    agent : AgentType
        An instance of an AgentType whose dynamic problem is to be solved.

    solution_next : Solution
        A representation of the solution of the period/stage that comes after the
        end of the sequence of one period/stage problems.  This might be the term-
        inal period/stage solution, a "pseudo terminal" solution, or simply the
        solution to the earliest period/stage from the succeeding cycle.

    Returns
    -------
    full_cycle : [Solution]
        An ordered list of solutions ('stages') for a full "cycle"
        of the AgentType's problem, constructed by backward
        induction.  After construction, each solution is prepended to the
        existing collection of solutions, with the result that in the completed
        `full_cycle` object element 0 is the solution to the problem at the
        beginning of the cycle, and full_cycle[-1] is the solution to the last
        problem completed before moving to the next cycle.
    """
    # Calculate number of stages per cycle;
    # defaults to 1 if all variables are stage invariant
    if len(agent.time_vary) > 0:
        num_stges = len(agent.__dict__[agent.time_vary[0]])
    else:
        num_stges = 1

    # Add to solve_dict all the parameters in time_inv and time_vary
    solve_dict = {parameter: agent.__dict__[parameter]
                  for parameter in agent.time_inv}
    solve_dict.update({parameter: None
                       for parameter in agent.time_vary})

    # Initialize the solution for this cycle, then iterate through stages
    full_cycle = []

    for this_stge in range(num_stges):  # for quarterly model, num_stges = 4 qtrs
        # Update which single period/stage solver to use (if depends on stage)
        if hasattr(agent.solve_one_period, "__getitem__"):  # ->solve_this_stge
            solve_one_period = agent.solve_one_period[num_stges - 1 - this_stge]
        else:
            solve_one_period = agent.solve_one_period

        # Code below has been made standalone in get_solve_one_period_args
        # which returns the solve_dict without the "solution_next" object

        # Construct arguments demanded by solve_one_period
        if hasattr(solve_one_period, "solver_args"):
            these_args = solve_one_period.solver_args
        else:
            these_args = get_arg_names(solve_one_period)

        # Update any stage-varying single period/stage inputs
        # This obtains the value for the current step by indexing
        # into the attribute's list at [num_stges - 1 - this_stge]
        for name in agent.time_vary:
            if name in these_args:
                solve_dict[name] = agent.__dict__[name][num_stges - 1 - this_stge]

        solve_dict['solution_next'] = solution_next  # solution_stage_next
        # solution_next is added here because it is a required argument but is
        # included in neither time_vary nor time_inv. Seems like maybe it could
        # (and should) be included in one of those ...

        # Make a temporary dictionary for this period
        temp_dict = {name: solve_dict[name] for name in these_args}

        # Solve one stage, add it to the collection, and designate
        # the just-solved solution as being in the future for the
        # purposes of any remaining iteration(s)

        solution_current = solve_one_period(**temp_dict)  # -> solve_this_stage
        full_cycle.insert(0, solution_current)
        solution_next = solution_current

    # Return the list of per-period/stage solutions
    return full_cycle


# def get_solve_one_period_args(agent, solve_one_period, stge_which):
#     # Add to dict all the parameters in time_inv and time_vary
#     solve_dict = \
#         {parameter: agent.__dict__[parameter] for parameter in agent.time_inv}
#     solve_dict.update({parameter: None for parameter in agent.time_vary})

#     # The code below allows stage-varying arguments by constructing the
#     # arguments demanded by the given solve_one_period
#     if hasattr(solve_one_period, "solver_args"):
#         these_args = solve_one_period.solver_args
#     else:
#         these_args = get_arg_names(solve_one_period)

#     # Update stage-varying single period/stage inputs
#     # This obtains the value for the current step by indexing
#     # into the attribute's list at [num_stges - 1 - stge]
#     for name in agent.time_vary:
#         if name in these_args:
#             solve_dict[name] = agent.__dict__[name][stge_which]

#     return solve_dict


def make_one_period_oo_solver(solver_class, **kwds):
    # Allow for future methods by making solveMethod an option
    """
    Returns a function that solves a single period/stage problem.
    Parameters
    ----------
    solver_class : Solver
        A class of Solver to be used.

    Returns
    -------
    solver_function : function
        A function for solving one period/stage of a problem.
    """

    def one_period_solver(**kwds):
        solver = solver_class(**kwds)  # defined externally

        if hasattr(solver, "prepare_to_solve"):
            # Steps, if any, to prep for sol of stge
            solver.prepare_to_solve()  # TODO: rename to prepare_to_solve_stge

        solution_current = solver.solve()  # TODO: rename to solve_stge
        return solution_current

    one_period_solver.solver_class = solver_class

    one_period_solver.solver_args = get_arg_names(solver_class.__init__)[1:]

    return one_period_solver  # TODO: rename to one_stge_solver someday


make_one_stage_oo_solver = make_one_period_oo_solver


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
        **kwds
    ):
        super().__init__()
        self.agents = agents if agents is not None else list()  # NOQA

        reap_vars = reap_vars if reap_vars is not None else list()  # NOQA
        self.reap_state = dict([(var, []) for var in reap_vars])

        self.sow_vars = sow_vars if sow_vars is not None else list()  # NOQA
        # dictionaries for tracking initial and current values
        # of the sow variables.
        self.sow_init = dict([(var, None) for var in self.sow_vars])
        self.sow_state = dict([(var, None) for var in self.sow_vars])

        const_vars = const_vars if const_vars is not None else list()  # NOQA
        self.const_vars = dict([(var, None) for var in const_vars])

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
        self.history = {
            var_name: []
            for var_name
            in self.track_vars
        }

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
        history_vars_string = ""
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
        A distribution.

    Returns
    -------
    agent_set : [AgentType]
        A list of param_count agents, ex ante heterogeneous with
        respect to param_name. The AgentCount of the original
        will be split between the agents of the returned
        list in proportion to the given distribution.
    """
    param_dist = distribution.approx(N=param_count)

    agent_set = [deepcopy(agent) for i in range(param_count)]

    for j in range(param_count):
        agent_set[j].AgentCount = int(agent.AgentCount * param_dist.pmf[j])
        # agent_set[j].__dict__[param_name] = param_dist.X[j]

        agent_set[j].assign_parameters(**{param_name: param_dist.X[j]})



    return agent_set

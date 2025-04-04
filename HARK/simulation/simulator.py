"""
A module with classes and functions for automated simulation of HARK.AgentType
models from a human- and machine-readable model specification.
"""

from dataclasses import dataclass, field
from copy import deepcopy
import numpy as np
from sympy.utilities.lambdify import lambdify
from sympy import symbols, IndexedBase
from typing import Callable
from HARK.utilities import NullFunc
from HARK.distributions import Distribution
import yaml

# Prevent pre-commit from removing sympy
x = symbols("x")
del x
y = IndexedBase("y")
del y


@dataclass(kw_only=True)
class ModelEvent:
    """
    Class for representing "events" that happen to agents in the course of their
    model. These might be statements of dynamics, realization of a random shock,
    or the evaluation of a function (potentially a control or other solution-
    based object). This is a superclass for types of events defined below.

    Parameters
    ----------
    description : str
        Text description of this model event.
    statement : str
        The line of the model statement that this event corresponds to.
    parameters : dict
        Dictionary of objects that are static / universal within this event.
    assigns : list[str]
        List of names of variables that this event assigns values for.
    needs : list[str]
        List of names of variables that this event requires to be run.
    data : dict
        Dictionary of current variable values within this event.
    """

    description: str = ""
    statement: str = ""
    parameters: dict = field(default_factory=dict)
    assigns: list[str] = field(default_factory=list)
    needs: list = field(default_factory=list)
    data: dict = field(default_factory=dict)

    def run(self):
        """
        This method should be filled in by each subclass
        """
        pass

    def reset(self):
        self.data = {}

    def assign(self, output):
        if len(self.assigns) > 1:
            assert len(self.assigns) == len(output)
            for j in range(len(self.assigns)):
                var = self.assigns[j]
                self.data[var] = output[j]
        else:
            var = self.assigns[0]
            self.data[var] = output


@dataclass(kw_only=True)
class DynamicEvent(ModelEvent):
    """
    Class for representing model dynamics for an agent, consisting of an expression
    to be evaluated and variables to which the results are assigned.

    Parameters
    ----------
    expr : Callable
        Function or expression to be evaluated for the assigned variables.
    args : list[str]
        Ordered list of argument names for the expression.
    """

    expr: Callable = NullFunc()
    args: list[str] = field(default_factory=list)

    def evaluate(self):
        temp_dict = self.data.copy()
        temp_dict.update(self.parameters)
        args = (temp_dict[arg] for arg in self.args)
        out = self.expr(*args)
        return out

    def run(self):
        self.assign(self.evaluate())


@dataclass(kw_only=True)
class RandomEvent(ModelEvent):
    """
    Class for representing the realization of random variables for an agent,
    consisting of a shock distribution and variables to which the results are assigned.

    Parameters
    ----------
    dstn : Distribution
        Distribution of one or more random variables that are drawn from during
        this event and assigned to the corresponding variables.
    """

    dstn: Distribution
    N: int = 1

    def draw(self):
        out = self.dstn.draw(self.N)
        return out

    def run(self):
        self.assign(self.draw())

    def reset(self):
        self.dstn.reset()
        ModelEvent.reset(self)


@dataclass(kw_only=True)
class RandomIndexedEvent(RandomEvent):
    """
    Class for representing the realization of random variables for an agent,
    consisting of a list of shock distributions, and index for the list, and the
    variables to which the results are assigned.

    Parameters
    ----------
    dstn : [Distribution]
        List of distributions of one or more random variables that are drawn
        from during this event and assigned to the corresponding variables.
    index : str
        Name of the index that is used to choose a distribution for each agent.
    """

    index: str = ""
    dstn: list[Distribution] = field(default_factory=list)

    def draw(self):
        idx = self.data[self.index]
        K = len(self.assigns)
        out = np.empty((K, self.N))
        out.fill(np.nan)
        for k in range(len(self.dstn)):
            these = idx == k
            if not np.any(these):
                continue
            out[:, these] = self.dstn[k].draw(np.sum(these))
        if K == 1:
            out = out.flatten()
        return out

    def reset(self):
        for k in range(len(self.dstn)):
            self.dstn[k].reset()
        ModelEvent.reset(self)


@dataclass(kw_only=True)
class MarkovEvent(ModelEvent):
    """
    Class for representing the realization of a Markov draw for an agent, in which
    a Markov probabilities (array, vector, or a single float) is used to determine
    the realization of some discrete outcome. If the probabilities are a 2D array,
    it represents a Markov matrix (rows sum to 1), and there must be an index; if
    the probabilities are a vector, it should be a stochastic vector; if it's a
    single float, it represents a Bernoulli probability.
    """

    probs: str = ""
    index: str = ""
    N: int = 1
    seed: int = 0  # TODO: There needs to be some way to set this seed

    def __post_init__(self):
        self.reset_rng()

    def reset_rng(self):
        self.RNG = np.random.default_rng(self.seed)

    def draw(self):
        out = -np.ones(self.N, dtype=int)
        probs = self.parameters[self.probs]
        X = self.RNG.random(self.N)
        if self.index:  # it's a Markov matrix
            idx = self.data[self.index]
            J = probs.shape[0]
            for j in range(J):
                these = idx == j
                if not np.any(these):
                    continue
                P = np.cumsum(probs[j, :])
                out[these] = np.searchsorted(P, X[these])
            return out
        if type(probs) is np.array:  # it's a stochastic vector
            P = np.cumsum(probs)
            return np.searchsorted(P, X)
        # Otherwise, this is just a Bernoulli RV
        if type(probs) is not float:
            raise ValueError(
                "Probabilities for an event must be a Markov matrix, stochastic vector, or single probability!"
            )
        P = probs
        return X < P  # basic Bernoulli

    def run(self):
        self.assign(self.draw())

    def reset(self):
        self.reset_rng()
        ModelEvent.reset(self)


@dataclass(kw_only=True)
class EvaluationEvent(ModelEvent):
    """
    Class for representing the evaluation of a model function. This might be from
    the solution of the model (like a policy function or decision rule) or just
    a non-algebraic function used in the model. This looks a lot like DynamicEvent.

    Parameters
    ----------
    func : Callable
        Model function that is evaluated in this event, with the output assigned
        to the appropriate variables.
    """

    func: Callable = NullFunc()
    arguments: list[str] = field(default_factory=list)

    def evaluate(self):
        temp_dict = self.data.copy()
        temp_dict.update(self.parameters)
        args_temp = (temp_dict[arg] for arg in self.arguments)
        out = self.func(*args_temp)
        return out

    def run(self):
        self.assign(self.evaluate())


@dataclass(kw_only=True)
class SimBlock:
    """
    Class for representing a "block" of a simulated model, which might be a whole
    period or a "stage" within a period.

    Parameters
    ----------
    description : str
        Textual description of what happens in this simulated block.
    statement : str
        Verbatim model statement that was used to create this block.
    content : dict
        Dictionary of objects that are constant / universal within the block.
        This includes both traditional numeric parameters as well as functions.
    pre_states : list[str]
        List of inbound states: information available at the *start* of the block.
    events: list[ModelEvent]
        Ordered list of events that happen during the block.
    data: dict
        Dictionary that stores current variable values.
    N : int
        Number of idiosyncratic agents in this block.
    """

    description: str = ""
    statement: str = ""
    content: dict = field(default_factory=dict)
    pre_states: list[str] = field(default_factory=list)
    events: list[ModelEvent] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    N: int = 1

    def run(self):
        """
        Run this simulated block by running each of its events in order.
        """
        for j in range(len(self.events)):
            event = self.events[j]
            for k in range(len(event.assigns)):
                var = event.assigns[k]
                if var in event.data.keys():
                    del event.data[var]
            for k in range(len(event.needs)):
                var = event.needs[k]
                event.data[var] = self.data[var]
            event.N = self.N
            event.run()
            for k in range(len(event.assigns)):
                var = event.assigns[k]
                self.data[var] = event.data[var]

    def reset(self):
        """
        Reset the simulated block by resetting each of its events.
        """
        self.data = {}
        for j in range(len(self.events)):
            self.events[j].reset()

    def distribute_content(self):
        """
        Fill in parameters, functions, and distributions to each event.
        """
        for event in self.events:
            for param in event.parameters.keys():
                try:
                    event.parameters[param] = self.content[param]
                except:
                    raise ValueError(
                        "Could not distribute the parameter called " + param + "!"
                    )
            if (type(event) is RandomEvent) or (type(event) is RandomIndexedEvent):
                try:
                    event.dstn = self.content[event._dstn_name]
                except:
                    raise ValueError(
                        "Could not find a distribution called " + event._dstn_name + "!"
                    )
            if type(event) is EvaluationEvent:
                try:
                    event.func = self.content[event._func_name]
                except:
                    raise ValueError(
                        "Could not find a function called " + event._func_name + "!"
                    )


@dataclass(kw_only=True)
class AgentSimulator:
    """
    A class for representing an entire simulator structure for an AgentType.
    It includes a sequence of SimBlocks representing periods of the model, which
    could be built from the information on an AgentType instance.

    Parameters
    ----------
    description : str
        Textual description of what happens in this simulated block.
    statement : str
        Verbatim model statement that was used to create this simulator.
    comments : dict
        Dictionary of comments or descriptions for various model objects.
    parameters : list[str]
        List of parameter names used in the model.
    distributions : list[str]
        List of distribution names used in the model.
    functions : list[str]
        List of function names used in the model.
    common: list[str]
        Names of variables that are common across idiosyncratic agents.
    types: dict
        Dictionary of data types for all variables in the model.
    N_agents: int
        Number of idiosyncratic agents in this simulation.
    T_total: int
        Total number of periods in these agents' model.
    T_sim: int
        Maximum number of periods that will be simulated, determining the size
        of the history arrays.
    T_age: int
        Period after which to automatically terminate an agent if they would
        survive past this period.
    stop_dead : bool
        Whether simulated agents who draw dead=True should actually cease acting.
        Default is True. Setting to False allows "cohort-style" simulation that
        will generate many agents that survive to old ages. In most cases, T_sim
        should not exceed T_age, unless the user really does want multiple succ-
        essive cohorts to be born and fully simulated.
    replace_dead : bool
        Whether simulated agents who are marked as dead should be replaced with
        newborns (default True) or simply cease acting without replacement (False).
        The latter option is useful for models with state-dependent mortality,
        to allow "cohort-style" simulation with the correct distribution of states
        for survivors at each age. Setting to False has no effect if stop_dead is True.
    periods: list[SimBlock]
        Ordered list of simulation blocks, each representing a period.
    twist : dict
        Dictionary that maps period t-1 variables to period t variables, as a
        relabeling "between" periods.
    initializer : SimBlock
        A special simulated block that should have *no* pre-states, because it
        represents the initialization of "newborn" agents.
    data : dict
        Dictionary that holds *current* values of model variables.
    track_vars : list[str]
        List of names of variables whose history should be tracked in the simulation.
    history : dict
        Dictionary that holds the histories of tracked variables.
    """

    description: str = ""
    statement: str = ""
    comments: dict = field(default_factory=dict)
    parameters: list[str] = field(default_factory=list)
    distributions: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    common: list[str] = field(default_factory=list)
    types: dict = field(default_factory=dict)
    N_agents: int = 1
    T_total: int = 1
    T_sim: int = 1
    T_age: int = 0
    stop_dead: bool = True
    replace_dead: bool = True
    periods: list[SimBlock] = field(default_factory=list)
    twist: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    initializer: SimBlock = SimBlock()
    track_vars: list[str] = field(default_factory=list)
    history: dict = field(default_factory=dict)

    def simulate(self, T=None):
        """
        Simulates the model for T periods, including replacing dead agents as
        warranted and storing tracked variables in the history. If T is not
        specified, the agents are simulated for the entire T_sim periods.
        This is the primary user-facing simulation method.
        """
        if T is None:
            T = self.T_sim - self.t_sim  # All remaining simulated periods
        if (T + self.t_sim) > self.T_sim:
            raise ValueError("Can't simulate more than T_sim periods!")

        # Execute the simulation loop for T periods
        for t in range(T):
            # Do the ordinary work for simulating a period
            self.sim_one_period()

            # Mark agents who have reached maximum allowable age
            if "dead" in self.data.keys() and self.T_age > 0:
                too_old = self.t_age == self.T_age
                self.data["dead"][too_old] = True

            # Record tracked variables and advance age
            self.store_tracked_vars()
            self.advance_age()

            # Handle death and replacement depending on simulation style
            if "dead" in self.data.keys() and self.stop_dead:
                self.mark_dead_agents()
            self.t_sim += 1

    def reset(self):
        """
        Completely reset this simulator back to its original state so that it
        can be run from scratch. This should allow it to generate the same results
        every single time the simulator is run (if nothing changes).
        """
        N = self.N_agents
        T = self.T_sim
        self.t_sim = 0  # Time index for the simulation

        # Reset the variable data and history arrays
        self.clear_data()
        self.history = {}
        for var in self.track_vars:
            self.history[var] = np.empty((T, N), dtype=self.types[var])

        # Reset all of the blocks / periods
        self.initializer.reset()
        for t in range(len(self.periods)):
            self.periods[t].reset()

        # Specify all agents as "newborns" assigned to the initializer block
        self.t_seq_bool_array = np.zeros((self.T_total, N), dtype=bool)
        self.t_age = -np.ones(N, dtype=int)

    def clear_data(self, skip=None):
        """
        Reset all current data arrays back to blank, other than those designated
        to be skipped, if any.

        Parameters
        ----------
        skip : [str] or None
            Names of variables *not* to be cleared from data. Default is None.

        Returns
        -------
        None
        """
        if skip is None:
            skip = []
        N = self.N_agents
        # self.data = {}
        for var in self.types.keys():
            if var in skip:
                continue
            this_type = self.types[var]
            if this_type is float:
                self.data[var] = np.full((N,), np.nan)
            elif this_type is bool:
                self.data[var] = np.zeros((N,), dtype=bool)
            elif this_type is int:
                self.data[var] = np.zeros((N,), dtype=np.int32)
            elif this_type is complex:
                self.data[var] = np.full((N,), np.nan, dtype=complex)
            else:
                raise ValueError(
                    "Type "
                    + str(this_type)
                    + " of variable "
                    + var
                    + " was not recognized!"
                )

    def mark_dead_agents(self):
        """
        Looks at the special data field "dead" and marks those agents for replacement.
        If no variable called "dead" has been defined, this is skipped.
        """
        who_died = self.data["dead"]
        self.t_seq_bool_array[:, who_died] = False
        self.t_age[who_died] = -1

    def create_newborns(self):
        """
        Calls the initializer to generate newborns where needed.
        """
        # Skip this step if there are no newborns
        newborns = self.t_age == -1
        if not np.any(newborns):
            return

        # Generate initial pre-states
        N = np.sum(newborns)
        self.initializer.data = {}  # by definition
        self.initializer.N = N
        self.initializer.run()

        # Set the initial pre-state data for newborns and clear other variables
        init_pre_states = self.periods[0].pre_states
        for var in self.types:
            self.data[var][newborns] = (
                self.initializer.data[var]
                if var in init_pre_states
                else np.empty(N, dtype=self.types[var])
            )

        # Set newborns' period to 0
        self.t_age[newborns] = 0
        self.t_seq_bool_array[0, newborns] = True

    def store_tracked_vars(self):
        """
        Record current values of requested variables in the history dictionary.
        """
        for var in self.track_vars:
            self.history[var][self.t_sim, :] = self.data[var]

    def advance_age(self):
        """
        Increments age for all agents, altering t_age and t_age_bool. Agents in
        the last period of the sequence will be assigned to the initial period.
        In a lifecycle model, those agents should be marked as dead and replaced
        in short order.
        """
        alive = self.t_age >= 0  # Don't age the dead
        self.t_age[alive] += 1
        X = self.t_seq_bool_array  # For shorter typing on next line
        self.t_seq_bool_array[:, alive] = np.concatenate(
            (X[-1:, alive], X[:-1, alive]), axis=0
        )

    def sim_one_period(self):
        """
        Simulates one period of the model by advancing all agents one period.
        This includes creating newborns, but it does NOT include eliminating
        dead agents nor storing tracked results in the history. This method
        should usually not be called by a user, instead using simulate(1) if
        you want to run the model for exactly one period.
        """
        # Use the "twist" information to advance last period's end-of-period
        # information/values to be the pre-states for this period. Then, for
        # any variable other than those brought in with the twist, wipe it clean.
        keepers = []
        for var_tm1 in self.twist:
            var_t = self.twist[var_tm1]
            keepers.append(var_t)
            self.data[var_t] = self.data[var_tm1].copy()
        self.clear_data(skip=keepers)

        # Create newborns first so the pre-states exist. This should be done in
        # the first simulated period (t_sim=0) or if decedents should be replaced.
        if self.replace_dead or self.t_sim == 0:
            self.create_newborns()

        # Loop through ages and run the model on the appropriately aged agents
        for t in range(self.T_total):
            these = self.t_seq_bool_array[t, :]
            if not np.any(these):
                continue  # Skip any "empty ages"
            this_period = self.periods[t]

            data_temp = {var: self.data[var][these] for var in this_period.pre_states}
            this_period.data = data_temp
            this_period.N = np.sum(these)
            this_period.run()

            # Extract all of the variables from this period and write it to data
            for var in this_period.data.keys():
                self.data[var][these] = this_period.data[var]

        # Put time information into the data dictionary
        self.data["t_age"] = self.t_age.copy()
        self.data["t_seq"] = np.argmax(self.t_seq_bool_array, axis=0).astype(int)

    def describe_model(self, display=True):
        """
        Convenience method that prints model information to screen.
        """
        # Make a twist statement
        twist_statement = ""
        for var_tm1 in self.twist.keys():
            var_t = self.twist[var_tm1]
            new_line = var_tm1 + "[t-1] <---> " + var_t + "[t]\n"
            twist_statement += new_line

        # Assemble the overall model statement
        output = ""
        output += "----------------------------------\n"
        output += "%%%%%% INITIALIZATION AT t=0 %%%%%\n"
        output += "----------------------------------\n"
        output += self.initializer.statement
        output += "----------------------------------\n"
        output += "%%%% DYNAMICS WITHIN PERIOD t %%%%\n"
        output += "----------------------------------\n"
        output += self.statement
        output += "----------------------------------\n"
        output += "%%%%%%% RELABELING / TWIST %%%%%%%\n"
        output += "----------------------------------\n"
        output += twist_statement
        output += "-----------------------------------"

        # Return or print the output
        if display:
            print(output)
            return
        else:
            return output

    def describe_symbols(self, display=True):
        """
        Convenience method that prints symbol information to screen.
        """
        # Get names and types
        symbols_lines = []
        comments = []
        for key in self.comments.keys():
            comments.append(self.comments[key])

            # Get type of object
            if key in self.types.keys():
                this_type = str(self.types[key].__name__)
            elif key in self.distributions:
                this_type = "dstn"
            elif key in self.parameters:
                this_type = "param"
            elif key in self.functions:
                this_type = "func"

            # Add tags
            if key in self.common:
                this_type += ", common"
            # if key in self.solution:
            #    this_type += ', solution'
            this_line = key + " (" + this_type + ")"
            symbols_lines.append(this_line)

        # Add comments, aligned
        symbols_text = ""
        longest = np.max([len(this) for this in symbols_lines])
        for j in range(len(symbols_lines)):
            line = symbols_lines[j]
            comment = comments[j]
            L = len(line)
            pad = (longest + 1) - L
            symbols_text += line + pad * " " + ": " + comment + "\n"

        # Return or print the output
        output = symbols_text
        if display:
            print(output)
            return
        else:
            return output

    def describe(self, symbols=True, model=True, display=True):
        """
        Convenience method for showing all information about the model.
        """
        # Asssemble the requested output
        output = self.description + "\n"
        if symbols or model:
            output += "\n"
        if symbols:
            output += "----------------------------------\n"
            output += "%%%%%%%%%%%%% SYMBOLS %%%%%%%%%%%%\n"
            output += "----------------------------------\n"
            output += self.describe_symbols(display=False)
        if model:
            output += self.describe_model(display=False)
        if symbols and not model:
            output += "----------------------------------"

        # Return or print the output
        if display:
            print(output)
            return
        else:
            return output


def make_simulator_from_agent(agent, stop_dead=True, replace_dead=True):
    """
    Build an AgentSimulator instance based on an AgentType instance. The AgentType
    should have its model attribute defined so that it can be parsed and translated
    into the simulator structure. The names of objects in the model statement
    should correspond to attributes of the AgentType

    Parameters
    ----------
    agent : AgentType
        Agents for whom a new simulator is to be constructed.
    stop_dead : bool
        Whether simulated agents who draw dead=True should actually cease acting.
        Default is True. Setting to False allows "cohort-style" simulation that
        will generate many agents that survive to old ages. In most cases, T_sim
        should not exceed T_age, unless the user really does want multiple succ-
        essive cohorts to be born and fully simulated.
    replace_dead : bool
        Whether simulated agents who are marked as dead should be replaced with
        newborns (default True) or simply cease acting without replacement (False).
        The latter option is useful for models with state-dependent mortality,
        to allow "cohort-style" simulation with the correct distribution of states
        for survivors at each age. Setting False has no effect if stop_dead is True.

    Returns
    -------
    new_simulator : AgentSimulator
        A simulator structure based on the agents.
    """
    # Read the model statement into a dictionary, and get names of attributes
    model = yaml.safe_load(agent.model_)
    time_vary = agent.time_vary
    time_inv = agent.time_inv
    cycles = agent.cycles
    T_age = agent.T_age
    comments = {}

    # Make a blank "template" period with structure but no data
    template_period, information, block_comments = make_template_block(model)
    comments.update(block_comments)

    # Make the agent initializer, without parameter values (etc)
    initializer, init_info = make_initializer(model)

    # Extract basic fields from the template period and model
    description = template_period.description
    statement = template_period.statement
    content = template_period.content
    try:
        variables = model["symbols"]["variables"]
    except:
        variables = []
    try:
        common = model["symbols"]["common"]
    except:
        common = []
    try:
        offset = model["symbols"]["offset"]
    except:
        offset = []
    try:
        solution = model["symbols"]["solution"]
    except:
        solution = []
    try:
        twist = model["twist"]
    except:
        twist = {}

    # Get the names of parameters, functions, and distributions
    parameters = []
    functions = []
    distributions = []
    for key in information.keys():
        val = information[key]
        if val is None:
            parameters.append(key)
        elif type(val) is NullFunc:
            functions.append(key)
        elif type(val) is Distribution:
            distributions.append(key)

    # Make a dictionary of data types and add comments
    types = {}
    for var_line in variables:  # Loop through declared variables
        var_name, var_type, desc = parse_declaration_for_parts(var_line)
        if var_type is not None:
            try:
                var_type = eval(var_type)
            except:
                raise ValueError(
                    "Couldn't understand type "
                    + var_type
                    + " for declared variable "
                    + var_name
                    + "!"
                )
        else:
            var_type = float
        types[var_name] = var_type
        comments[var_name] = desc
    for var in information.keys():  # Loop through undeclared variables
        if var in types.keys():
            continue
        this = information[var]
        if (this is None) or (type(this) is Distribution) or (type(this) is NullFunc):
            continue
        types[var] = float
        comments[var] = ""
    if "dead" in types.keys():
        types["dead"] = bool
        comments["dead"] = "whether agent died this period"
    types["t_seq"] = int
    types["t_age"] = int
    comments["t_seq"] = "which period of the sequence the agent is on"
    comments["t_age"] = "how many periods the agent has already lived for"

    # Make a dictionary for the initializer and distribute information
    init_dict = {}
    for name in init_info.keys():
        try:
            init_dict[name] = getattr(agent, name)
        except:
            raise ValueError(
                "Couldn't get a value for initializer object " + name + "!"
            )
    initializer.content = init_dict
    initializer.distribute_content()

    # Make a dictionary of time-invariant parameters
    time_inv_dict = {}
    for name in content:
        if name in time_inv:
            try:
                time_inv_dict[name] = getattr(agent, name)
            except:
                raise ValueError(
                    "Couldn't get a value for time-invariant object " + name + "!"
                )

    # Create a list of periods, pulling appropriate data from the agent for each one
    T_seq = len(agent.solution)  # Number of periods in the solution sequence
    periods = []
    T_cycle = agent.T_cycle
    t_cycle = 0
    for t in range(T_seq):
        new_period = deepcopy(template_period)

        # Make the parameter dictionary for this period
        new_param_dict = deepcopy(time_inv_dict)
        for name in content:
            if name in solution:
                new_param_dict[name] = getattr(agent.solution[t], name)
            elif name in time_vary:
                s = (t_cycle - 1) if name in offset else t_cycle
                new_param_dict[name] = getattr(agent, name)[s]
            elif name in time_inv:
                continue
            else:
                raise ValueError(
                    "Couldn't get a value for time-varying object " + name + "!"
                )

        # Fill in content for this period, then add it to the list
        new_period.content = new_param_dict
        new_period.distribute_content()
        periods.append(new_period)

        # Advance time according to the cycle
        t_cycle += 1
        if t_cycle == T_cycle:
            t_cycle = 0

    # Calculate maximum age
    if T_age is None:
        T_age = 0
    if cycles > 0:
        T_age_max = T_seq - 1
        T_age = np.minimum(T_age_max, T_age)

    # Make and return the new simulator
    new_simulator = AgentSimulator(
        description=description,
        statement=statement,
        comments=comments,
        parameters=parameters,
        functions=functions,
        distributions=distributions,
        common=common,
        types=types,
        N_agents=agent.AgentCount,
        T_total=T_seq,
        T_sim=agent.T_sim,
        T_age=T_age,
        stop_dead=stop_dead,
        replace_dead=replace_dead,
        periods=periods,
        twist=twist,
        initializer=initializer,
        track_vars=agent.track_vars,
    )
    return new_simulator


def make_template_block(model):
    """
    Construct a new SimBlock object as a "template" of the model block. It has
    events and reference information, but no values filled in.

    Parameters
    ----------
    model : dict
        Dictionary with model block information, probably read in as a yaml.

    Returns
    -------
    template_block : SimBlock
        A "template" of this model block, with no parameters (etc) on it.
    info : dict
        Dictionary of model objects that were referenced within the block. Keys
        are object names and entries reveal what kind of object they are:
        - None --> parameter
        - 0 --> outcome variable (including pre-states)
        - NullFunc --> function
        - Distribution --> distribution
    comments : dict
        Dictionary of comments included with declared functions, distributions,
        and parameters.
    """
    # Extract pre-state names
    try:
        pre_states = model["symbols"]["pre_states"]
    except:
        pre_states = []

    # Extract model description
    try:
        description = model["description"]
    except:
        description = ""

    # Extract parameters, functions, and distributions
    comments = {}
    parameters = {}
    if "parameters" in model["symbols"].keys():
        param_lines = model["symbols"]["parameters"]
        for line in param_lines:
            param_name, datatype, desc = parse_declaration_for_parts(line)
            parameters[param_name] = None
            comments[param_name] = desc
            # TODO: what to do with parameter types?

    functions = {}
    if "functions" in model["symbols"].keys():
        func_lines = model["symbols"]["functions"]
        for line in func_lines:
            func_name, datatype, desc = parse_declaration_for_parts(line)
            if (datatype is not None) and (datatype != "func"):
                raise ValueError(
                    func_name
                    + " was declared as a function, but given a different datatype!"
                )
            functions[func_name] = NullFunc()
            comments[func_name] = desc

    distributions = {}
    if "distributions" in model["symbols"].keys():
        dstn_lines = model["symbols"]["distributions"]
        for line in dstn_lines:
            dstn_name, datatype, desc = parse_declaration_for_parts(line)
            if (datatype is not None) and (datatype != "dstn"):
                raise ValueError(
                    dstn_name
                    + " was declared as a distribution, but given a different datatype!"
                )
            distributions[dstn_name] = Distribution()
            comments[dstn_name] = desc

    # Combine those dictionaries into a single "information" dictionary
    content = parameters.copy()
    content.update(functions)
    content.update(distributions)
    info = deepcopy(content)
    for var in pre_states:
        info[var] = 0  # Mark as a state variable

    # Parse the model dynamics
    dynamics = format_block_statement(model["dynamics"])

    # Make the list of ordered events
    events = []
    for line in dynamics:
        # Make the new event and add it to the list
        new_event = make_new_event(line, info)
        events.append(new_event)

        # Add newly assigned variables to the information set
        for var in new_event.assigns:
            if var in info.keys():
                raise ValueError(var + " is assigned, but already exists!")
            info[var] = 0

    # Make a single string model statement
    statement = ""
    longest = np.max([len(event.statement) for event in events])
    for event in events:
        this_statement = event.statement
        L = len(this_statement)
        pad = (longest + 1) - L
        statement += this_statement + pad * " " + ": " + event.description + "\n"

    # Make and return the new SimBlock
    template_block = SimBlock(
        description=description,
        pre_states=pre_states,
        content=content,
        statement=statement,
        events=events,
    )
    return template_block, info, comments


def make_initializer(model):
    """
    Construct a new SimBlock object to be the agent initializer, based on the
    model dictionary. It has structure and events, but no parameters (etc).

    Parameters
    ----------
    model : dict
        Dictionary with model initializer information, probably read in as a yaml.

    Returns
    -------
    initializer : SimBlock
        A "template" of this model block, with no parameters (etc) on it.
    init_requires : dict
        Dictionary of model objects that are needed by the initializer to run.
        Keys are object names and entries reveal what kind of object they are:
        - None --> parameter
        - 0 --> outcome variable (these should include all pre-states)
        - NullFunc --> function
        - Distribution --> distribution
    """
    # Extract pre-state names
    try:
        pre_states = model["symbols"]["pre_states"]
    except:
        pre_states = []

    # Extract parameters, functions, and distributions
    parameters = {}
    if "parameters" in model["symbols"].keys():
        param_lines = model["symbols"]["parameters"]
        for line in param_lines:
            param_name, datatype, desc = parse_declaration_for_parts(line)
            parameters[param_name] = None

    functions = {}
    if "functions" in model["symbols"].keys():
        func_lines = model["symbols"]["functions"]
        for line in func_lines:
            func_name, datatype, desc = parse_declaration_for_parts(line)
            if (datatype is not None) and (datatype != "func"):
                raise ValueError(
                    func_name
                    + " was declared as a function, but given a different datatype!"
                )
            functions[func_name] = NullFunc()

    distributions = {}
    if "distributions" in model["symbols"].keys():
        dstn_lines = model["symbols"]["distributions"]
        for line in dstn_lines:
            dstn_name, datatype, desc = parse_declaration_for_parts(line)
            if (datatype is not None) and (datatype != "dstn"):
                raise ValueError(
                    dstn_name
                    + " was declared as a distribution, but given a different datatype!"
                )
            distributions[dstn_name] = Distribution()

    # Combine those dictionaries into a single "information" dictionary
    content = parameters.copy()
    content.update(functions)
    content.update(distributions)
    info = deepcopy(content)

    # Parse the initialization routine
    initialize = format_block_statement(model["initialize"])

    # Make the list of ordered events
    events = []
    for line in initialize:
        # Make the new event and add it to the list
        new_event = make_new_event(line, info)
        events.append(new_event)

        # Add newly assigned variables to the information set
        for var in new_event.assigns:
            if var in info.keys():
                raise ValueError(var + " is assigned, but already exists!")
            info[var] = 0

    # Verify that all pre-states were created in the initializer
    for var in pre_states:
        if var not in info.keys():
            raise ValueError(
                "The pre-state " + var + " was not set in the initialize block!"
            )

    # Make a blank dictionary with information the initializer needs
    init_requires = {}
    for event in events:
        for var in event.parameters.keys():
            if var not in init_requires.keys():
                try:
                    init_requires[var] = parameters[var]
                except:
                    raise ValueError(
                        var
                        + " was referenced in initialize, but not declared as a parameter!"
                    )
        if type(event) is RandomEvent:
            try:
                dstn_name = event._dstn_name
                init_requires[dstn_name] = distributions[dstn_name]
            except:
                raise ValueError(
                    dstn_name
                    + " was referenced in initialize, but not declared as a distribution!"
                )
        if type(event) is EvaluationEvent:
            try:
                func_name = event._func_name
                init_requires[dstn_name] = functions[func_name]
            except:
                raise ValueError(
                    func_name
                    + " was referenced in initialize, but not declared as a function!"
                )

    # Make a single string initializer statement
    statement = ""
    longest = np.max([len(event.statement) for event in events])
    for event in events:
        this_statement = event.statement
        L = len(this_statement)
        pad = (longest + 1) - L
        statement += this_statement + pad * " " + ": " + event.description + "\n"

    # Make and return the new SimBlock
    initializer = SimBlock(
        description="initialize model agents",
        content=init_requires,
        statement=statement,
        events=events,
    )
    return initializer, init_requires


def make_new_event(statement, info):
    """
    Makes a "blank" version of a model event based on a statement line. Determines
    which objects are needed vs assigned vs parameters / information from context.

    Parameters
    ----------
    statement : str
        One line of a model statement, which will be turned into an event.
    info : dict
        Empty dictionary of model information that already exists. Consists of
        pre-states, already assigned variables, parameters, and functions. Typing
        of each is based on the kind of "empty" object.

    Returns
    -------
    new_event : ModelEvent
        A new model event with values and information missing, but structure set.
    """
    # First determine what kind of event this is
    has_eq = "=" in statement
    has_tld = "~" in statement
    has_amp = "@" in statement
    has_brc = ("{" in statement) and ("}" in statement)
    has_brk = ("[" in statement) and ("]" in statement)
    event_type = None
    if has_eq:
        if has_tld:
            raise ValueError("A statement line can't have both an = and a ~!")
        if has_amp:
            event_type = EvaluationEvent
        else:
            event_type = DynamicEvent
    if has_tld:
        if has_brc:
            event_type = MarkovEvent
        elif has_brk:
            event_type = RandomIndexedEvent
        else:
            event_type = RandomEvent
    if event_type is None:
        raise ValueError("Statement line was not any valid type!")

    # Now make and return an appropriate event for that type
    if event_type is DynamicEvent:
        new_event = make_new_dynamic(statement, info)
    if event_type is RandomEvent:
        new_event = make_new_random(statement, info)
    if event_type is RandomIndexedEvent:
        new_event = make_new_random_indexed(statement, info)
    if event_type is MarkovEvent:
        new_event = make_new_markov(statement, info)
    if event_type is EvaluationEvent:
        new_event = make_new_evaluation(statement, info)
    return new_event


def make_new_dynamic(statement, info):
    """
    Construct a new instance of DynamicEvent based on the given model statement
    line and a blank dictionary of parameters. The statement should already be
    verified to be a valid dynamic statement: it has an = but no ~ or @.

    Parameters
    ----------
    statement : str
        One line dynamics statement, which will be turned into a DynamicEvent.
    info : dict
        Empty dictionary of available information.

    Returns
    -------
    new_dynamic : DynamicEvent
        A new dynamic event with values and information missing, but structure set.
    """
    # Cut the statement up into its LHS, RHS, and description
    lhs, rhs, description = parse_line_for_parts(statement, "=")

    # Parse the LHS (assignment) to get assigned variables
    assigns = parse_assignment(lhs)

    # Parse the RHS (dynamic statement) to extract variable names used
    variables, is_indexed = extract_var_names_from_expr(rhs)

    # Allocate each variable to needed dynamic variables or parameters
    needs = []
    parameters = {}
    for j in range(len(variables)):
        var = variables[j]
        if var not in info.keys():
            raise ValueError(
                var + " is used in a dynamic expression, but does not (yet) exist!"
            )
        val = info[var]
        if type(val) is NullFunc:
            raise ValueError(
                var + " is used in a dynamic expression, but it's a function!"
            )
        if type(val) is Distribution:
            raise ValueError(
                var + " is used in a dynamic expression, but it's a distribution!"
            )
        if val is None:
            parameters[var] = None
        else:
            needs.append(var)

    # Declare a SymPy symbol for each variable used; these are temporary
    _args = []
    for j in range(len(variables)):
        _var = variables[j]
        if is_indexed[j]:
            exec(_var + " = IndexedBase('" + _var + "')")
        else:
            exec(_var + " = symbols('" + _var + "')")
        _args.append(eval(_var))

    # Make a SymPy expression, then lambdify it
    sympy_expr = eval(rhs)
    expr = lambdify(_args, sympy_expr)

    # Make and return the new dynamic event
    new_dynamic = DynamicEvent(
        description=description,
        statement=lhs + " = " + rhs,
        assigns=assigns,
        needs=needs,
        parameters=parameters,
        expr=expr,
        args=variables,
    )
    return new_dynamic


def make_new_random(statement, info):
    """
    Make a new random variable realization event based on the given model statement
    line and a blank dictionary of parameters. The statement should already be
    verified to be a valid random statement: it has a ~ but no = or [].

    Parameters
    ----------
    statement : str
        One line of the model statement, which will be turned into a random event.
    info : dict
        Empty dictionary of available information.

    Returns
    -------
    new_random : RandomEvent
        A new random event with values and information missing, but structure set.
    """
    # Cut the statement up into its LHS, RHS, and description
    lhs, rhs, description = parse_line_for_parts(statement, "~")

    # Parse the LHS (assignment) to get assigned variables
    assigns = parse_assignment(lhs)

    # Verify that the RHS is actually a distribution
    if type(info[rhs]) is not Distribution:
        raise ValueError(
            rhs + " was treated as a distribution, but not declared as one!"
        )

    # Make and return the new random event
    new_random = RandomEvent(
        description=description,
        statement=lhs + " ~ " + rhs,
        assigns=assigns,
        needs=[],
        parameters={},
        dstn=info[rhs],
    )
    new_random._dstn_name = rhs
    return new_random


def make_new_random_indexed(statement, info):
    """
    Make a new indexed random variable realization event based on the given model
    statement line and a blank dictionary of parameters. The statement should
    already be verified to be a valid random statement: it has a ~ and [].

    Parameters
    ----------
    statement : str
        One line of the model statement, which will be turned into a random event.
    info : dict
        Empty dictionary of available information.

    Returns
    -------
    new_random_indexed : RandomEvent
        A new random indexed event with values and information missing, but structure set.
    """
    # Cut the statement up into its LHS, RHS, and description
    lhs, rhs, description = parse_line_for_parts(statement, "~")

    # Parse the LHS (assignment) to get assigned variables
    assigns = parse_assignment(lhs)

    # Split the RHS into the distribution and the index
    dstn, index = parse_random_indexed(rhs)

    # Verify that the RHS is actually a distribution
    if type(info[dstn]) is not Distribution:
        raise ValueError(
            dstn + " was treated as a distribution, but not declared as one!"
        )

    # Make and return the new random indexed event
    new_random_indexed = RandomIndexedEvent(
        description=description,
        statement=lhs + " ~ " + rhs,
        assigns=assigns,
        needs=[index],
        parameters={},
        # dstn=info[dstn],
        index=index,
    )
    new_random_indexed._dstn_name = dstn
    return new_random_indexed


def make_new_markov(statement, info):
    """
    Make a new Markov-type event based on the given model statement line and a
    blank dictionary of parameters. The statement should already be verified to
    be a valid Markov statement: it has a ~ and {} and ().

    Parameters
    ----------
    statement : str
        One line of the model statement, which will be turned into a random event.
    info : dict
        Empty dictionary of available information.

    Returns
    -------
    new_markov : MarkovEvent
        A new Markov draw event with values and information missing, but structure set.
    """
    # Cut the statement up into its LHS, RHS, and description
    lhs, rhs, description = parse_line_for_parts(statement, "~")

    # Parse the LHS (assignment) to get assigned variables
    assigns = parse_assignment(lhs)

    # Parse the RHS (Markov statement) for the array and index
    probs, index = parse_markov(rhs)
    if index is None:
        needs = []
    else:
        needs = [index]

    # Make and return the new Markov event
    new_markov = MarkovEvent(
        description=description,
        statement=lhs + " ~ " + rhs,
        assigns=assigns,
        needs=needs,
        parameters={probs: None},
        probs=probs,
        index=index,
    )
    return new_markov


def make_new_evaluation(statement, info):
    """
    Make a new function evaluation event based the given model statement line
    and a blank dictionary of parameters. The statement should already be verified
    to be a valid evaluation statement: it has an @ and an = but no ~.

    Parameters
    ----------
    statement : str
        One line of the model statement, which will be turned into an eval event.
    info : dict
        Empty dictionary of available information.

    Returns
    -------
    new_evaluation : EvaluationEvent
        A new evaluation event with values and information missing, but structure set.
    """
    # Cut the statement up into its LHS, RHS, and description
    lhs, rhs, description = parse_line_for_parts(statement, "=")

    # Parse the LHS (assignment) to get assigned variables
    assigns = parse_assignment(lhs)

    # Parse the RHS (evaluation) for the function and its arguments
    func, arguments = parse_evaluation(rhs)

    # Allocate each variable to needed dynamic variables or parameters
    needs = []
    parameters = {}
    for j in range(len(arguments)):
        var = arguments[j]
        if var not in info.keys():
            raise ValueError(
                var + " is used in an evaluation statement, but does not (yet) exist!"
            )
        val = info[var]
        if type(val) is NullFunc:
            raise ValueError(
                var
                + " is used as an argument an evaluation statement, but it's a function!"
            )
        if type(val) is Distribution:
            raise ValueError(
                var + " is used in an evaluation statement, but it's a distribution!"
            )
        if val is None:
            parameters[var] = None
        else:
            needs.append(var)

    # Make and return the new evaluation event
    new_evaluation = EvaluationEvent(
        description=description,
        statement=lhs + " = " + rhs,
        assigns=assigns,
        needs=needs,
        parameters=parameters,
        arguments=arguments,
        func=info[func],
    )
    new_evaluation._func_name = func
    return new_evaluation


def parse_declaration_for_parts(line):
    """
    Split a declaration line from a model file into the object's name, its datatype,
    and any provided comment or description.

    Parameters
    ----------
    line : str
        Line of to be parsed into the object name, object type, and a comment or description.

    Returns
    -------
    name : str
        Name of the object.
    datatype : str or None
        Provided datatype string, in parentheses, if any.
    desc : str
        Comment or description, after //, if any.
    """
    # First, separate off the comment or description, if any
    slashes = line.find("\\")
    desc = "" if slashes == -1 else line[(slashes + 2) :].strip()
    rem = line if slashes == -1 else line[:slashes].strip()

    # Now look for bracketing parentheses declaring a datatype
    lp = rem.find("(")
    if lp > -1:
        rp = rem.find(")")
        if rp == -1:
            raise ValueError("Unclosed parentheses on object declaration line!")
        datatype = rem[(lp + 1) : rp].strip()
        leftover = rem[:lp].strip()
    else:
        datatype = None
        leftover = rem

    # What's left over should be the object name
    name = leftover
    # TODO: Check for valid name formatting based on characters.

    return name, datatype, desc


def parse_line_for_parts(statement, symb):
    """
    Split one line of a model statement into its LHS, RHS, and description. The
    description is everything following \\, while the LHS and RHS are determined
    by a special symbol.

    Parameters
    ----------
    statement : str
        One line of a model statement, which will be parsed for its parts.
    symb : char
        The character that represents the divide between LHS and RHS

    Returns
    -------
    lhs : str
        The left-hand (assignment) side of the expression.
    rhs : str
        The right-hand (evaluation) side of the expression.
    desc : str
        The provided description of the expression.
    """
    eq = statement.find(symb)
    lhs = statement[:eq].replace(" ", "")
    not_lhs = statement[(eq + 1) :]
    comment = not_lhs.find("\\")
    desc = "" if comment == -1 else not_lhs[(comment + 2) :].strip()
    rhs = not_lhs if comment == -1 else not_lhs[:comment]
    rhs = rhs.replace(" ", "")
    return lhs, rhs, desc


def parse_assignment(lhs):
    """
    Get ordered list of assigned variables from the LHS of a model line.

    Parameters
    ----------
    lhs : str
        Left-hand side of a model expression

    Returns
    -------
    assigns : List[str]
        List of variable names that are assigned in this model line.
    """
    if lhs[0] == "(":
        if not lhs[-1] == ")":
            raise ValueError("Parentheses on assignment was not closed!")
        assigns = []
        pos = 0
        while pos != -1:
            pos += 1
            end = lhs.find(",", pos)
            var = lhs[pos:end]
            if var != "":
                assigns.append(var)
            pos = end
    else:
        assigns = [lhs]
    return assigns


def extract_var_names_from_expr(expression):
    """
    Parse the RHS of a dynamic model statement to get variable names used in it.

    Parameters
    ----------
    expression : str
        RHS of a model statement to be parsed for variable names.

    Returns
    -------
    var_names : List[str]
        List of variable names used in the expression. These *should* be dynamic
        variables and parameters, but not functions.
    indexed : List[bool]
        Indicators for whether each variable seems to be used with indexing.
    """
    var_names = []
    indexed = []
    math_symbols = "+-/*^%.(),[]{}<>"
    digits = "01234567890"
    cur = ""
    for j in range(len(expression)):
        c = expression[j]
        if (c in math_symbols) or ((c in digits) and cur == ""):
            if cur == "":
                continue
            if cur in var_names:
                cur = ""
                continue
            var_names.append(cur)
            if c == "[":
                indexed.append(True)
            else:
                indexed.append(False)
            cur = ""
        else:
            cur += c
    if cur != "" and cur not in var_names:
        var_names.append(cur)
        indexed.append(False)  # final symbol couldn't possibly be indexed
    return var_names, indexed


def parse_evaluation(expression):
    """
    Separate a function evaluation expression into the function that is called
    and the variable inputs that are passed to it.

    Parameters
    ----------
    expression : str
        RHS of a function evaluation model statement, which will be parsed for
        the function and its inputs.

    Returns
    -------
    func_name : str
        Name of the function that will be called in this event.
    arg_names : List[str]
        List of arguments of the function.
    """
    # Get the name of the function: what's to the left of the @
    amp = expression.find("@")
    func_name = expression[:amp]

    # Check for parentheses formatting
    rem = expression[(amp + 1) :]
    if not rem[0] == "(":
        raise ValueError(
            "The @ in a function evaluation statement must be followed by (!"
        )
    if not rem[-1] == ")":
        raise ValueError("A function evaluation statement must end in )!")
    rem = rem[1:-1]

    # Parse what's inside the parentheses for argument names
    arg_names = []
    pos = 0
    go = True
    while go:
        end = rem.find(",", pos)
        if end > -1:
            arg = rem[pos:end]
        else:
            arg = rem[pos:]
            go = False
        if arg != "":
            arg_names.append(arg)
        pos = end + 1

    return func_name, arg_names


def parse_markov(expression):
    """
    Separate a Markov draw declaration into the array of probabilities and the
    index for idiosyncratic values.

    Parameters
    ----------
    expression : str
        RHS of a function evaluation model statement, which will be parsed for
        the probabilities name and index name.

    Returns
    -------
    probs : str
        Name of the probabilities object in this statement.
    index : str
        Name of the indexing variable in this statement.
    """
    # Get the name of the probabilitie
    lb = expression.find("{")  # this *should* be 0
    rb = expression.find("}")
    if lb == -1 or rb == -1 or rb < (lb + 2):
        raise ValueError("A Markov assignment must have an {array}!")
    probs = expression[(lb + 1) : rb]

    # Get the name of the index, if any
    x = rb + 1
    lp = expression.find("(", x)
    rp = expression.find(")", x)
    if lp == -1 and rp == -1:  # no index present at all
        return probs, None
    if lp == -1 or rp == -1 or rp < (lp + 2):
        raise ValueError("Improper Markov formatting: should be {probs}(index)!")
    index = expression[(lp + 1) : rp]

    return probs, index


def parse_random_indexed(expression):
    """
    Separate an indexed random variable assignment into the distribution and
    the index for it.

    Parameters
    ----------
    expression : str
        RHS of a function evaluation model statement, which will be parsed for
        the distribution name and index name.

    Returns
    -------
    dstn : str
        Name of the distribution in this statement.
    index : str
        Name of the indexing variable in this statement.
    """
    # Get the name of the index
    lb = expression.find("[")
    rb = expression.find("]")
    if lb == -1 or rb == -1 or rb < (lb + 2):
        raise ValueError("An indexed random variable assignment must have an [index]!")
    index = expression[(lb + 1) : rb]

    # Get the name of the distribution
    dstn = expression[:lb]

    return dstn, index


def format_block_statement(statement):
    """
    Ensure that a string stagement of a model block (maybe a period, maybe an
    initializer) is formatted as a list of strings, one statement per entry.

    Parameters
    ----------
    statement : str
        A model statement, which might be for a block or an initializer. The
        statement might be formatted as a list or as a single string.

    Returns
    -------
    block_statements: [str]
        A list of model statements, one per entry.
    """
    if type(statement) is str:
        if statement.find("\n") > -1:
            block_statements = []
            pos = 0
            end = statement.find("\n", pos)
            while end > -1:
                new_line = statement[pos:end]
                block_statements.append(new_line)
                pos = end + 1
                end = statement.find("\n", pos)
        else:
            block_statements = [statement.copy()]
    if type(statement) is list:
        for line in statement:
            if type(line) is not str:
                raise ValueError("The model statement somehow includes a non-string!")
        block_statements = statement.copy()
    return block_statements

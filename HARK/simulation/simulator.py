"""
A module with classes and functions for automated simulation of HARK.AgentType
models from a human- and machine-readable model specification.
"""

from dataclasses import dataclass, field
from copy import deepcopy
import numpy as np
from sympy.utilities.lambdify import lambdify
from sympy import symbols
from typing import Callable
from HARK.utilities import NullFunc
from HARK.distributions import Distribution
import yaml

# Prevent pre-commit from removing sympy
x = symbols("x")
del x


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
    """

    expr: Callable = NullFunc()

    def evaluate(self):
        temp_dict = self.data.copy()
        temp_dict.update(self.parameters)
        out = self.expr(**temp_dict)
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
            if type(event) is RandomEvent:
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
        Verbatim model statement that was used to create this block.
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
    common: list[str] = field(default_factory=list)
    types: dict = field(default_factory=dict)
    N_agents: int = 1
    T_total: int = 1
    T_sim: int = 1
    T_age: int = 0
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
            self.sim_one_period()
            self.store_tracked_vars()
            self.advance_age()
            if "dead" in self.data.keys():
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

        # Reset the variable data
        self.data = {}
        for var in self.types.keys():
            self.data[var] = np.empty(N, dtype=self.types[var])

        # Reset the history arrays
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

    def mark_dead_agents(self):
        """
        Looks at the special data field "dead" and marks those agents for replacement.
        If no variable called "dead" has been defined, this is skipped.
        """
        if self.T_age > 0:
            too_old = self.t_age > self.T_age  # age has already advanced
            self.data["dead"][too_old] = True
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

    def sim_one_period(self):
        """
        Simulates one period of the model by advancing all agents one period.
        This includes creating newborns, but it does NOT include eliminating
        dead agents nor storing tracked results in the history. This method
        should usually not be called by a user, instead using simulate(1) if
        you want to run the model for exactly one period.
        """
        # Use the "twist" information to advance last period's end-of-period
        # information/values to be the pre-states for this period
        for var_tm1 in self.twist:
            var_t = self.twist[var_tm1]
            self.data[var_t] = self.data[var_tm1].copy()

        # Create newborns first so the pre-states exist
        self.create_newborns()

        # Loop through ages and run the model on the appropriately aged agents
        for t in range(self.T_total):
            these = self.t_seq_bool_array[t, :]
            if not np.any(these):
                continue  # Skip any "empty ages"

            data_temp = {
                var: self.data[var][these] for var in self.periods[t].pre_states
            }
            self.periods[t].data = data_temp
            self.periods[t].N = np.sum(these)
            self.periods[t].run()

            # Extract all of the variables from this period and write it to data
            for var in self.periods[t].data.keys():
                self.data[var][these] = self.periods[t].data[var]

        # Put time information into the data dictionary
        self.data["t_age"] = self.t_age.copy()
        self.data["t_seq"] = np.argmax(self.t_seq_bool_array, axis=0).astype(int)

    def advance_age(self):
        """
        Increments age for all agents, altering t_age and t_age_bool. Agents in
        the last period of the sequence will be assigned to the initial period.
        In a lifecycle model, those agents should be marked as dead and replaced
        in short order.
        """
        self.t_age += 1
        X = self.t_seq_bool_array  # For shorter typing on next line
        self.t_seq_bool_array = np.concatenate((X[-1:, :], X[:-1, :]), axis=0)


def make_simulator_from_agent(agent):
    """
    Build an AgentSimulator instance based on an AgentType instance. The AgentType
    should have its model attribute defined so that it can be parsed and translated
    into the simulator structure. The names of objects in the model statement
    should correspond to attributes of the AgentType

    Parameters
    ----------
    agent : AgentType
        Agents for whom a new simulator is to be constructed.

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

    # Make a blank "template" period with structure but no data
    template_period, information = make_template_block(model)

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

    # Make a dictionary of data types
    types = {}
    for var in variables:  # Loop through declared variables
        try:
            var_type = eval(variables[var]["type"])
        except:
            var_type = float
        types[var] = var_type
    for var in information.keys():  # Loop through undeclared variables
        if var in types.keys():
            continue
        this = information[var]
        if (this is None) or (type(this) is Distribution) or (type(this) is NullFunc):
            continue
        types[var] = float
    if "dead" in types.keys():
        types["dead"] = bool
    types["t_cycle"] = int
    types["t_age"] = int

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
    if cycles > 0:
        T_age_max = T_seq - 1
        T_age = np.minimum(T_age_max, T_age)

    # Make and return the new simulator
    new_simulator = AgentSimulator(
        description=description,
        statement=statement,
        common=common,
        types=types,
        N_agents=agent.AgentCount,
        T_total=T_seq,
        T_sim=agent.T_sim,
        T_age=T_age,
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
    try:
        parameters = {param: None for param in model["symbols"]["parameters"]}
    except:
        parameters = {}
    try:
        functions = {func: NullFunc() for func in model["symbols"]["functions"]}
    except:
        functions = {}
    try:
        distributions = {
            dstn: Distribution() for dstn in model["symbols"]["distributions"]
        }
    except:
        distributions = {}

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
    for event in events:
        statement += event.statement + "\n"

    # Make and return the new SimBlock
    template_block = SimBlock(
        description=description,
        pre_states=pre_states,
        content=content,
        statement=statement,
        events=events,
    )
    return template_block, info


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
    try:
        parameters = {param: None for param in model["symbols"]["parameters"]}
    except:
        parameters = {}
    try:
        functions = {func: NullFunc() for func in model["symbols"]["functions"]}
    except:
        functions = {}
    try:
        distributions = {
            dstn: Distribution() for dstn in model["symbols"]["distributions"]
        }
    except:
        distributions = {}

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
    for event in events:
        statement += event.statement + "\n"

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
    # print("Now examining this statement: " + statement)
    has_eq = "=" in statement
    has_tld = "~" in statement
    has_amp = "@" in statement
    event_type = None
    if has_eq:
        if has_tld:
            raise ValueError("A statement line can't have both an = and a ~!")
        if has_amp:
            event_type = EvaluationEvent
        else:
            event_type = DynamicEvent
    if has_tld:
        event_type = RandomEvent
    if event_type is None:
        raise ValueError("Statement line was not any valid type!")

    # print("It's a " + str(event_type.__name__) + "!")

    # Now make and return an appropriate event for that type
    if event_type is DynamicEvent:
        new_event = make_new_dynamic(statement, info)
    if event_type is RandomEvent:
        new_event = make_new_random(statement, info)
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
    lhs, rhs, description = parse_for_parts(statement, "=")

    # Parse the LHS (assignment) to get assigned variables
    assigns = parse_assignment(lhs)

    # Parse the RHS (dynamic statement) to extract variable names used
    variables = extract_var_names_from_expr(rhs)

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
    for _var in variables:
        exec(_var + " = symbols('" + _var + "')")
        _args.append(eval(_var))

    # Make a SymPy expression, then lambdify it
    sympy_expr = eval(rhs)
    expr = lambdify(_args, sympy_expr)

    # Make and return the new dynamic event
    new_dynamic = DynamicEvent(
        description=description,
        statement=lhs + "=" + rhs,
        assigns=assigns,
        needs=needs,
        parameters=parameters,
        expr=expr,
    )
    return new_dynamic


def make_new_random(statement, info):
    """
    Make a new random variable realization event based on a line of the given
    model statement line and a blank dictionary of parameters. The statement
    should already be verified to be a valid random statement: it has a ~ but
    no =.

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
    lhs, rhs, description = parse_for_parts(statement, "~")

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
        statement=lhs + "~" + rhs,
        assigns=assigns,
        needs=[],
        parameters={},
        dstn=info[rhs],
    )
    new_random._dstn_name = rhs
    return new_random


def make_new_evaluation(statement, info):
    """
    Make a new function evaluation event based on a line of the given model
    statement line and a blank dictionary of parameters. The statement should
    already be verified to be a valid evaluation statement: it has an @ and an
    = but no ~.

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
    lhs, rhs, description = parse_for_parts(statement, "=")

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
        statement=lhs + "=" + rhs,
        assigns=assigns,
        needs=needs,
        parameters=parameters,
        arguments=arguments,
        func=info[func],
    )
    new_evaluation._func_name = func
    return new_evaluation


def parse_for_parts(statement, symb):
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
    """
    var_names = []
    math_symbols = "+-/*^%.(),"
    digits = "01234567890"
    cur = ""
    for j in range(len(expression)):
        c = expression[j]
        if (c in math_symbols) or ((c in digits) and cur == ""):
            if cur == "":
                continue
            var_names.append(cur)
            cur = ""
        else:
            cur += c
    if cur != "":
        var_names.append(cur)
    return var_names


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

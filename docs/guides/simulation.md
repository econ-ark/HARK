# New Simulation System and Model Specification Files for HARK

This document explains the formatting and usage of model files in `HARK`, which can be used to greatly simplify simulation code and to quickly and easily convey the mathematical content of a model. It begins with instructions for how to **use** the new simulation system, then proceeds to a specification of the model file format (for adding new models). I then explain how to attach the model file to an existing `HARK` model.

## Basic Simulator Usage

If you want to simulate an `AgentType` instance that already has its model attached, the new system's syntax and usage is nearly identical to the current/legacy `simulate` system.

### Simulation Attributes

First, your `AgentType` should have the following attributes defined:

- `AgentCount`: the number of agents to be simulated, as an integer
- `T_sim`: the total number of periods you want to simulate, as an integer
- `track_vars`: a list of strings naming the idiosyncratic variables that should be tracked/stored in the simulated history
- `T_age`: (optional) the maximum number of periods an agent is allowed to survive, overriding the model's mortality process

In most contexts, you can ignore `T_age` completely. It is only relevant for infinite horizon models where you are concerned that a single simulated agent that (randomly) survives for a very long time could end up with arbitrarily high permanent income and hold a disproportionate share of total wealth (compared to the atomistic size they're supposed to represent).

### Initializing and Running a Simulation

In the current/legacy simulation system, you would do `MyType.initialize_sim()` to get the simulator ready, by clearing any prior history and allocating empty arrays. In the new system, the method call is `MyType.initialize_sym()` (note the vowel change); this is to allow the two simulation systems to co-exist redundantly for testing. After `initialize_sym()` is run, you can do `MyType._simulator.describe()` to get a printed overview of the model.

In most contexts, you want to simply run all `T_sim` simulation periods. In the current/legacy system, this is done by running `MyType.simulate()`; analogously, the command in the new system is `MyType.symulate()`. These commands will populate the `history` and `hystory` attributes of the `AgentType` instance respectively. These are dictionaries whose keys are the variables named in `track_vars` and whose entries are shape `(T_sim, AgentCount)` arrays of simulated values. E.g. `MyType.hystory["cNrm"][5, 263]` represents the normalized consumption of agent `i=263` in period `t=5` of the simulation.

If you instead only want to run *some* simulation periods right now (say, because you are going to exogenously change the simulation state, then run the remaining periods), this can be done by passing an integer argument to the `simulate` or `symulate` methods, e.g. `MyType.symulate(50)`. Omitting the number of periods to simulate will default to running *all remaining* periods (all `T_sim` periods if none have been run yet).

To restart a simulation, the user can call `MyType._simulator.reset()`. This will return the simulator to its initial state, with no history nor simulated periods. All random variables will be automatically reset as well, so that they will generate the same sequence of draws on subsequent runs. If data on the `AgentType` instance changes (e.g., there is a new consumption function, or parameter values have changed), then `initialize_sym()` should be called instead, ensuring that the correct objects are used.

### Examining Simulation Output

In both the current/legacy system and the new system, the `history` and `hystory` attributes are dictionaries with keys named in `track_vars`. The entries are `numpy.array`s of shape `(T_sim, AgentCount)`; in the new simulation system (but not the current/legacy one), the datatype of each array will be appropriate to that variable-- tracked variables default to `float` but could be `bool` or `int`.

Ignoring any weighting scheme you might want to apply, the history arrays can be manipulated like any other arrays. E.g., the time series of the mean of normalized assets is `np.mean(MyType.hystory["aNrm"], axis=1)`. Note that with `replace_dead=False` (see below), some elements of the history arrays will be empty and might have `NaN` or point to uninitialized memory; be careful when aggregating or otherwise processing such data.

### Special Variables

A few variable names have special meaning internal to `HARK`'s simulation system and thus should not be used in model files (except `dead`). These variables can be tracked/recorded as normal, which is sometimes useful for processing other simulated histories.

- `t_age` (int): The number of periods this agent has lived at the *start* of the period; "newborns" have `t_age=0`. In the default simulation style, new newborns replace agents who die. This is a change from the timing of the current/legacy system, which records `t_age` *after* it increments at the very end of the period.
- `dead` (bool): Whether this agent died at the *end* of the period. This variable is read by the simulator to determine who to cease and/or replace. It is appropriate (and expected) to determine `dead` in a model file as part of the model dynamics.
- `t_seq` (int): Which period of the agent's sequence of *solutions* that were on for this period. This will only differ from `HARK`'s internal `t_cycle` when `cycles > 1`, so that knowing which *parameters* in the cycle to look at is not strictly informative of which *solution* (policy function) to use. As `cycles > 1` is rarely used outside of debugging, this is rarely useful to examine.

### Simulation Options

In the current/legacy system, and by default in the new system, the simulation uses a "death and replacement" or "current population" style. At the start of the simulation, all agents are initialized as "newborns", and might age and die according to the model's dynamics. Decedents are replaced with newly initialized newborns at the start of the next period, and these newborns have no relationship to the prior agent other than sharing an index number. This simulation style is appropriate for infinite horizon models (where aggregate outcomes are often of interest) and for finite horizon / lifecycle models in which there are aggregate feedback effects and/or the age distribution of the population is endogenous.

In some contexts, you may want to simulate using a "cohort" style, in which agents are *not* removed or replaced when they die. This is appropriate for lifecycle models with neither dynamic aggregate effects nor endogenous age distribution (that varies within the exercise of interest). To prevent newborns from replacing decedents, simply pass the `replace_dead=False` argument to `initialize_sym`; agents that die during the simulation will leave empty entries in the `hystory` arrays for later periods.

For specific reasons (e.g. you are interested in the model's predictions of outcomes after age 100), you might want a larger population of survivors at later ages than would ordinarily occur. To turn off mortality, pass the `stop_dead=False` argument to `initialize_sym`. Agents that *would* die are simply kept alive and allowed to continue in the model, generating a much larger population of (e.g.) 100 year olds. Note that for some models, this population of survivors is *not* representative of the population that would have arisen naturally with a (much) higher `AgentCount`. If survival or longevity is endogenous to behavior in the model, you must apply an appropriate weighting scheme to make sensible use of the population.

The main advantage of using `stop_dead=False` or `replace_dead=False` arises in lifecycle models with many periods. Preventing death or replacement means that *all* agents are the same age in each simulated period, short-circuiting the need to evaluate the policy function for each age in each simulated period for *some* agents. It also means that processing and interpreting the arrays in `hystory` (see above) is easier, as row `t` represents agent outcomes at model age `t` for all agents.

In most cases, the user wants idiosyncratic shocks to occur across agents, but *sometimes* they want some shock to share a common value across everyone (representing some macroeconomic outcome). As discussed below, the new simulator has functionality for some random variables to work like this by default, but this behavior can be overridden by passing a list of strings in the `common` argument of `initialize_sym()`. The variables named should be those *directly* assigned from a distribution, not variables "downstream" from the random realization. To maximize compatibility, the simulator will assign a value for `common` variables to each agent, but they will all be identical across agents.

> **NB:** This feature currently does not work as intended with lifecycle models. Because each period of the model is a "world unto itself," the random realizations are not shared across them in any way. For truly general "common" shocks, it might be necessary to have a larger structure that feeds in some values from an "aggregate block".


## Model File Specification

Model files for the new simulation system use the YAML format, and they currently live in `/HARK/models/`. Each model file contains information for how *one* `AgentType` subclass works, and provides instructions to the `simulator.py` module on how to simulate the model. To best read this section, it might be helpful to open one or more model files; consider the simplest one, `ConsPerfForesight.yaml`, as well as a more complicated one like `ConsMarkov.yaml`.

### Model File Basics

A model file has up to **six** top-level entries, which will be described in more detail below:

- `name`: A short string that provides a reference label for the model; optional.
- `description`: An optional string that provides an English description or summary of the model. It can be as long or short as you'd like.
- `symbols`: A nested list of the names of objects that appear in the model, and what *kind* of thing each one is.
- `initialize`: A (probably short) set of model statements that specify how a "newborn" agent is created.
- `dynamics`: A set of sequential model statements that specify *what happens* during a period, one statement per line.
- `twist`: A simple list of "intertemporal remappings" that relabel some end-of-period $t$ variables to give them new names for period $t+1$

On any line of a model file, the escape sequence `\\` is used to indicate that everything that follows is a *comment*. These comments are different from YAML comments (using `#`) because the YAML parser *ignores and discards* everything after `#` on each line-- it's not part of the data. In contrast, text after `\\` in a model file is read by `HARK`'s interpreter and *stored* as a comment within `HARK`, for later examination. This is useful for applying descriptive labels to model objects or dynamic statements.

### The `symbols` Section

The `symbols` entry of the model file lists out the variables and objects that appear in the model `dynamics` below, to help `HARK`'s simulator interpret and execute them. Each symbol might appear in more than one sub-entry of `symbols`-- some sections are declarations, while others are "metadata". The sub-entries that delcare objects in `symbols` are:

- `functions`: Names of callable objects used in the model dynamics. Anything that is treated like a function should be named here.
- `distributions`: Names of distributions used in the model dynamics, e.g. `IncShkDstn`.
- `parameters`: Names of all other objects that are common or shared among all agents *within* a period. E.g. in `HARK`'s consumption-saving models, the expected permanent income growth factor `PermGroFac` is a parameter shared by all agents who are in the same period of the model. These can be single numbers or arrays, but cannot be callable functions nor distributions.
- `variables`: Names of *idiosyncratic* variables that are used in the model `dynamics` statement. In most cases, it is not required to declare any `variables` at all; see below.

The sub-entries that provide metadata about objects are:

- `arrival`: Names of model `variables` that must exist at the *start* of a period-- "inbound" information. These variables are `initialize`d for newborns, and drawn through the intertemporal `twist` otherwise.
- `solution`: Names of model objects (of any type) that are part of the solution of the model, rather than part of the problem itself. This tells `HARK` to look for them in the `solution` attribute rather than on the `AgentType` instance itself. For example, `cFunc` should be declared as a `function`, but also listed in the `solution` entry.
- `offset`: Names of model objects that are "off by one" in time due to the funky timing of `HARK`'s solvers; see below.

Objects named as `functions`, `parameters`, or `distributions` literally refer to attributes of the `AgentType` instance (or its `solution` attribute, if also named in the `solution` entry). Objects named in `variables` are outcomes of the simulation and don't exist on the `AgentType` instance prior to simulation.

#### Declaring `variables`

Most idiosyncratic variables that appear in the `dynamics` block do not *need* to be declared in the `variables` sub-entry, but *can* be. Any undeclared idiosyncratic variables will be automatically added, with no comment or explanation and a default datatype of `float`. To demonstrate this, the `ConsPerfForesight_simple.yaml` file has *no* declared `variables` at all.

There are two use cases for making declarations in the `variables` sub-entry. First, some idiosyncratic variables are *not* real-valued, and should not be stored that way. The name of such a variable should be declared, with its datatype (probably `int` or `bool`) following in parentheses. For example, in the `KinkedR` model, there is a dummy variable called `boro` that is `True` if and only if wealth is negative (the agent is borrowing); the `variables` entry for this model includes the line `boro (bool)` so that this data is appropriately recorded in a Boolean array, rather than cast to `float`.

Second, even for real-valued idiosyncratic variables that should be recorded in `float` format, the model file can include a comment or explanation of the variable after the `\\` escape sequence. These comments can be examined within `HARK` by calling `describe()` on the `_simulator` attribute after `initialize_sym()` has been called.

#### Declaring `parameters`, `functions`, and `distributions`

Just like with `variables`, lines in the `parameters`, `functions`, and `distributions` sub-entries can have a comment or description after a `\\`, which will be recorded and displayed with `describe()`. When `MyType.initialize_sym()` is called, the `_simulator` attribute is constructed using the information in the model file. The objects named in `parameters`, `functions`, and `distributions` are pulled from the `AgentType` instance (or its `solution` attribute) and populated into a simulation structure. A `distribution` should be an instance of `HARK`'s `Distribution` class, which has standard methods that will be used by the simulator. A `function` can be any Python object that is callable.

Importantly, the model file *does not care* about whether any of these objects are time-varying or time-invariant. That information is *read from* the `AgentType` when the `_simulator` attribute is built. The current/legacy simulation system has a lot of tedious code for getting the correct parameter values for each period, and sometimes for checking whether that parameter is time-varying or time-invariant *for that agent*. This makes it unnecessarily difficult to *write* simulation code for a new model, when that's supposed to be the *easy* part of structural work! The new simulation system makes it much simpler and easier.

#### Inbound Information: Declaring `arrival` variables

The new simulation structure assumes that the same *kind* of period will happen over and over again to the agents, but with (potentially) different parameter values and/or policy functions (or other solution objects). The events within a period are described in `dynamics` (see below), but some information almost surely exists at the *start* of the period, probably carried over from the prior one-- otherwise, there would be no intertemporal relationships in the model.

Any idiosyncratic variables that exist at the *start* of a period, before anything has happened, should be named in `arrival`. When `HARK` parses the statement of model `dynamics`, it checks that information isn't used before it exists. Objects named in `parameters`, `functions`, and `distributions` are assumed to be static within the period and always exist, but idiosyncratic `variables` must be assigned before they can be used by subsequent events. Naming variables in `arrival` informs `HARK` that these idiosyncratic variables won't be declared within the model `dynamics`, but instead carried over from the past (or were just `initialize`d for newborns).

Alternatively, a `variable` can be designated to be in `arrival` by simply putting an exclamation point `!` after its name (but before its datatype) in the `variables` declaration; as usual, extra spaces are fine. For example, you can specify `kNrm` to be in `arrival` by naming it in `arrival` **or** by including the line `kNrm !` in the `variables` entry (along with whatever comment you'd like). Both notations can be used in the same model file.

As will be discussed more below, each variable named in `arrival` must *both* be assigned in the `initialize` entry (i.e. there is some way to determine its starting value for newborns) *and* be included in the `twist` entry (so that `HARK` knows which variables in $t-1$ correspond to arrival variables for $t$).

#### Special Cases: Declaring the `solution` and `offset` Parameters

Most of the data for the simulator is part of the agents' *problem* for the model, but some parts are the *solution* to that problem; most typically, one or more policy functions that provide instructions for how to choose a control variable conditional on observed information. Such objects should be named in the `solution` sub-entry, which simply tells `HARK` to look for them in the `solution` attribute rather than as attributes of the `AgentType` instance itself. Objects named in `solution` are most likely to be `functions`, but could also be `parameters` or `distributions`.

Rather than explicitly name `solution` objects in the entry, someone who prefers more parsimonious model files can simply add an asterisk `*` after the object name in its original declaration. For example, including `cFunc *` in the `functions` entry designates the consumption function as part of the `solution`.

> **NB:** This sub-entry could be eliminated entirely if we had each `HARK` model copy/reference its policy functions (etc) as part of `post_solve()` using `unpack()`. Then the simulator would look for everything on the `AgentType` itself. This would also enable users to do unconventional things like exogenously specify a non-optimal `cFunc` and simulate the model without ever solving it.

The parameter structure in `HARK` was designed with a "solver first" mentality: the value in `MyType.this_param[t]` is what's needed to solve for the period $t$ policy function, which might *actually* correspond to something that would have the time subscript $t+1$ when written in math. For example, because `HARK` currently computes expectations about $t+1$ when solving the period $t$ problem, the distribution of income shocks `IncShkDstn[t]` actually represents the shocks that could arrive at the start of $t+1$. Many model parameters work this way in `HARK`, in fact. The current/legacy simulation code consequently has a lot of `t-1`'s running around when grabbing parameter values. In the new system, simply name such "time-offset" objects in `offset` and `HARK` will do the rest.

As alternative notation, a plus sign `+` can be included in the declaration of a `parameter`, `function`, or `distribution` to indicate that it should be `offset` when populating periods. Like other notation like this, a model file can include both `+`'s and explicitly named `offset` objects.

> **NB:** If we overhaul `HARK`'s solver timing to put period $t$ expectations in the period $t$ solver, then `offset` can go away.


### The `initialize` Section

Put simply, agents have to start from somewhere. The `initialize` entry contains a statement of model dynamics that are run *only* when a new agent is created, whether at the start of a simulation run or because an agent died and is being replaced. Everything that can be done in the `dynamics` entry (see below) can also be done in the `initialize` entry, except that there is *no* pre-existing idiosyncratic information. The simplest possible `initialize` block will deterministically set each of the `arrival` variables to some specific value, e.g. `pLvlPrev = 1`, but more complex behavior is permitted.

The only hard and fast rule is that *all* `arrival` variables must be assigned within the `initialize` block-- they have to get *some* value. When a newborn agent begins their first simulated period, their `arrival` variables are assumed to exist, and thus they cannot be "blank" or uninitialized. `HARK` will raise an error if you attempt to `initialize_sym()` with a model file that does not assign all `arrival` variables.

The other thing to keep in mind is that any `parameters`, `functions`, or `distributions` referenced in the `initialize` block should *not* be time-varying. There is no sense of time or age *within* the `initialize` block-- it's just "The Before". You can have a special distribution called `kNrmInitDstn` that provides the distribution of initial capital holdings, but you shouldn't refer to `IncShkDstn` within `initialize` because it is (very likely) time-varying and lives in a list.


### The Block Where It Happens: Model `dynamics`

The `dynamics` entry has a sequential list of model *events* that happen each period, one per line. These events are noted in mostly ordinary math, with some special notation for particular situations. There are four main kinds of model events:

- **Dynamic** events are evaluations of ordinary algebraic statements (over `variables` and `parameters`) that are assigned to *exactly* one new `variable`.
- **Random** events draw from a `distribution` and assign the random result to *one or more* new `variables`.
- **Markov** events draw an integer or Boolean random variable based on probabilities in some `parameter`, assigning it to *one* random variable
- **Evaluation** events pass `variables` and/or `parameters` to a `function` and assign the idiosyncratic output to *one or more* new `variables`.

As a general rule, spacing does not matter in the `dynamics` block (nor anywhere else). You can put as many or as few spaces as you want in your math.

#### Ordinary Dynamic Events

A dynamic event is characterized by exactly one *assigned variable* on the left-hand side, an `=`, and an ordinary algebraic statement on the right-hand side. All of the standard math symbols can be used, and `sympy` will automatically cast the carat `^` to Python exponentiation `**`; I have not experimented with non-typical things like pipes `|` for absolute value. For `parameters` that are vector-valued, you can use Python-style indexing with brackets, e.g. `PermGroFac[z]` to get the permanent income growth factor at index `z`. Note that the indexing variable must be an `int` or an error will be raised on execution.

The `symbols` eligible to be referenced on the right-hand side include any `parameters`, any `arrival` variables, and any other `variables` that have already been assigned in this period in *prior* events. Notably, you cannot use any `distributions` nor `functions` in a dynamic event; those are for separate event types. As a simple example, the line `mNrm = bNrm + yNrm` assigns (normalized) market resources as the sum of bank balances and labor income, where the right-hand-side `variables` were assigned in prior model events.

The restriction that only a single outcome variable be assigned is not onerous. Each `variable` should be idiosyncratically single-valued (e.g. one real value per person), so situations where it is "natural" to assign two outcomes from a single line of math are hard to conceive. In any situation where you might want to do so, it should be possible to instead have two consecutive lines, one for each outcome.

#### Random Realization Events

A random event is characterized by one or more assigned variables on the left-hand side, a `~` ("drawn from"), and a `distribution` on the right-hand side. If more than one variable is assigned, their names should be comma-separated and contained within parentheses, e.g. `(PermShk, TranShk) ~ IncShkDstn` means that permanent and transitory shocks are jointly drawn from the income shock distribution. Parentheses are *optional* on the LHS if only one variable is assigned. Differentiating between *distributions* versus the *random variables* that are assigned is both convenient for interacting with existing `HARK` infrastructure and permits natural notation for multivariate distributions.

Just like with `parameters` in dynamic events, the `distribution` in a random realization event can be indexed by an integer. E.g. in the "Markov consumption model", many parameters depend on a discrete state (here denoted $z$) that evolves according to some Markov transition matrix, and the state-dependent income distribution can be represented with `(PermShk, TranShk) ~ IncShkDstn[z]` (as long as `z` has already been assigned).

As noted above, a `variable` called `dead` holds special meaning in `HARK`'s new simulation system. Such a variable should be Boolean valued (e.g. drawn from a `Bernoulli` distribution object), as it will be used by the simulator to determine which agents should be ceased and/or replaced between periods. For example, `HARK`'s consumption-saving models could include the line `dead ~ MortDstn`, where `MortDstn` is an instance of the class `Bernoulli`. A model does not *need* to assign `dead`, but mortality will not occur in the simulator, and unexpected behavior might arise if `T_sim` exceeds the length of the lifecycle.

#### Markov Draw Events

A special kind of random event is handled with special notation: discrete Markov realizations based on numeric probabilities. Such events *could* be handled with ordinary math notation over multiple events, but it would be tedious to write out for such a basic concept. There are three sub-cases of this event.

If there is a `parameter` called `MyProbs` that has a Markov transition matrix as a `numpy.array` (i.e. all values are non-negative and each row sums to 1), then the random transition from state `i` to state `j` is written as the model event `j ~ {MyProbs}(i)`. For example, in the "Markov consumption model", the transition probabilities are stored in a `parameter` called `MrkvArray`, and the discrete state is notated as `z`, so the Markov transition is `z ~ {MrkvArray}(zPrev)`. In this case, `zPrev` is an `arrival` variable-- it's last period's `z`!

In some cases, you don't have a Markov transition *per se*, just a discrete realization of outcomes A, B, or C. If there is a `parameter` called `MyProbs` that has a stochastic vector as a 1D `numpy.array` (all values are non-negative and it sums to 1), then drawing an integer index based on these probabilities is written as the model event `j ~ {MyProbs}`. Note that this is the same as the Markov matrix case, but with no indexing variable.

The simplest possible case is that you have a single probability and just want a weighted coin flip. If the `parameter` called `MyProb` is a single real number, then `j ~ {MyProb}` is treated as a Bernoulli random realization. This is useful for quickly expressing a simple mortality process, e.g. `live ~ {LivPrb}` and then `dead = 1 - live`. The output of the Bernoulli event is a `bool`, and doing one minus a Boolean is effectively the "logical not" operation. `Sympy` does allow the `~` logical not operator, but using it in the new simulator system will confuse the parser because we use it as the "draw from" symbol.

An alternative usage of Bernoulli events is to use an idiosyncratic data `variable` in braces, e.g. `live ~ {LivPrb_i}`, where `LivPrb_i` was assigned as a `variable` in a prior event. This used in `ConsMarkovModel`, where `LivPrb_i = LivPrb[z]` based on discrete state `z` for each agent. It could also be used in models in which survival probability depends on some continuous health state. Note that the object in brackets must be a single name; no expressions of any other kind are allowed. E.g. the parser does not support doing `live ~ {LivPrb[z]}`, nor `dead ~ {1-LivPrb}`.

#### Function Evaluation Events

Sometimes model dynamics include non-standard functions that cannot be expressed algebraically, but have some arbitrary underlying code. Most commonly, the policy function from the model's `solution` is non-parametric and is usually represented by some interpolant. In the new simulation system, evaluating such functions is a special model event.

An evaluation event is characterized by one or more assigned variables on the left-hand side followed by an `=`, and then a `function` and its arguments on the right-hand side, written as `func@(arg1, arg2, ..., argN)`. If more than one variable is assigned, their names should be comma-separated and contained within parentheses, as with random realization events. The inclusion of the `@` symbol for calling an arbitrary `function` object makes the model parser much easier to write-- it can look for an @ and know that the preceeding object should be a `function` and that this is an evaluation event.

The arguments to the function (contained within parentheses and comma-separated) can be any `parameters` or `arrival` variables, as well as any `variables` that have already been assigned in prior events. The system allows for multiple outputs from a single function because sometimes two model outcomes are jointly determined, or the computation to evaluate them separately would be so duplicative as to be wasteful.

Unlike with `parameters` and `distributions`, a `function` cannot be indexed. Instead, a function that you *would* like to index should instead be refactored or wrapped so that the index is an *argument* to the function. For example, in `ConsMarkovModel`, the discrete-state-dependent consumption functions are stored in Python as a list of functions, and the user accesses them as `cFunc[z](m)`. For compatibility with the new simulation system, these functions have been repackaged in a (fairly simple) new object class that has the syntax `cFuncX(z,m)`; the `X` is just to arbitrarily change the name.


### Intertemporal Transitions: Do the `twist`

The final entry in a model file is called `twist`, and it does nothing more than provide a mapping from end-of-period-$t$ variables to beginning-of-period-$t+1$ `arrival` variables. For example, in `HARK`'s consumption-saving models, end-of-period normalized assets (after all actions are accomplished) are denoted $a_t$ and represented in code as `aNrm`, and we want to use this same *value* (but not name) to represent normalized capital holdings at the start of period $t+1$, denoted $k_{t+1}$ and represented in code as `kNrm`. The fact that $a_t = k_{t+1}$ is captured in the `twist` entry `aNrm: kNrm`, an intertemporal remapping. Likewise, permanent income level `pLvl` evolves near the start of each period, but we need to know the *prior* permanent income level to calculate it. This is made accessible with the `twist` entry `pLvl: pLvlPrev`, so that `pLvlPrev` can be used to compute `pLvl`.


## Connecting a Model File to a `HARK` Model

Suppose you have written a YAML file representing your model, and you want to actually use it with your `HARK` model; this section provides instruction for how to do so. All it takes is for the name of the model file to be assigned to the `"model"` entry in the `default_` attribute of your `AgentType` subclass; that's it. As long as the named `parameters`, `functions`, and `distributions` actually exist on your `AgentType` instances (and your model file is validly formatted), it should just work. These files live in `/HARK/models/`.

In unusual cases, you might want to change an `AgentType` instance's simulation model *after* it has been instantiated. This can be done in two ways. First, you can change the instance's `model_file` attribute to name a *different* file in `/HARK/models/`. This attribute is filled in at instantiation by copying from `default_["model"]`, but can be changed afterward. Just be sure to do this before calling `initialize_sym()` to create the simulator object.

Alternatively, you can set the instance's `model_statement` attribute as a string with the *entire contents* of your model, in YAML format. This attribute does not exist by default, but the `initialize_sym()` method looks for a model statement there before defaulting to read the one from the file named in `model_file`. Putting anything in `model_statement` thus overrides the class's default simulation model.

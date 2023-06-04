# ARKitecture of Econ-ARK

This document guides you through the structure of econ-ark and explain the main ingredients. Note that it does not explain _how_ to use it. For this, please follow the example notebooks which you can find on the left.

[Econ-ark](https://github.com/econ-ark) contains the three main repositories [HARK](https://github.com/econ-ark/HARK), [DemARK](https://github.com/econ-ark/DEMARK), and [REMARK](https://github.com/econ-ark/REMARK). On top of that, the [website](https://econ-ark.org/) combines all of them. Hence, if you want to find a notebook search them in [materials](https://econ-ark.org/materials).

- [HARK](https://github.com/econ-ark/HARK): Includes the source code as well as some example notebooks.
- [DemARK](https://github.com/econ-ark/DemARK): Here you can find *Dem*onstrations of tools, AgentTypes, and ModelClasses.
- [REMARK](https://github.com/econ-ark/REMARK): Here you can find _R_[eplications/eproductions] and *E*xplorations *M*ade using _ARK_.

Before describing each repository in detail, some preliminary remarks.

HARK is written in Python, an object-oriented programming (OOP) language that has experienced increasing popularity in the scientific community in the past several years. A significant reason for the adoption of Python is the **_numpy_** and **_scipy_** packages, which offer a wide array of mathematical and statistical functions and tools; HARK makes liberal use of these libraries. Python's object-oriented nature allows models in HARK to be easily extended: more complex models can inherit functions and methods from more fundamental ''parent'' models, eliminating the need to reproduce or repurpose code.

As implied in the previous section, we strongly encourage HARK users to use the Anaconda distribution of Python, which includes all commonly used mathematical and scientific packages, an interactive development environment for iPython (Spyder), and a package manager that allows users to quickly install or update packages not included in the default distribution (conda).

For users unfamiliar with OOP, we strongly encourage you to review the background material on OOP provided by the good people at [QuantEcon](https://python.quantecon.org/intro.html) (for more on them, see below) at this link: [Object Oriented Programming](https://python-programming.quantecon.org/oop_intro.html). Unlike non-OOP languages, OOP bundles together data and functions into _objects_. These can be accessed via: **_object_name.data_** and **_object_name.method_name()_**, respectively. For organizational purposes, definitions of multiple objects are stored in _modules_, which are simply files with a **_.py_** extension. Modules can be accessed in Python via:

```
import module_name as import_name
```

This imports the module and gives it a local name of **_import_name_**. We can access a function within this module by simply typing: **_import_name.function_name()_**. The following example will illustrate the usage of these commands. **_CRRAutility_** is the function object for calculating CRRA utility supplied by **_HARK.utilities_** module. **_CRRAutility_** is called _attributes_ of the module **_HARK.utilities_**. In order to calculate CRRA utility with a consumption of 1 and a coefficient of risk aversion of 2 we run:

```
import HARK.utilities as Hutil

Hutil.CRRAutility(1,2)
```

Python modules in HARK can generally be categorized into three types: tools, models, and applications. **Tool modules** contain functions and classes with general purpose tools that have no inherent ''economic content'', but that can be used in many economic models as building blocks or utilities; they could plausibly be useful in non-economic settings. Tools might include functions for data analysis (e.g. calculating Lorenz shares from data, or constructing a non-parametric kernel regression), functions to create and manipulate discrete approximations to continuous distributions, or classes for constructing interpolated approximations to non-parametric functions. Tool modules generally reside in HARK's root directory and have names like **_HARK.simulation_** and **_HARK.interpolation_**; they do not necessarily do anything when run.

**Model modules** specify particular economic models, including classes to represent agents in the model (and the ''market structure'' in which they interact) and functions for solving the ''one period problem'' of those models. For example, **ConsIndShockModel.py** concerns consumption-saving models in which agents have CRRA utility over consumption and face idiosyncratic shocks to permanent and transitory income. The module includes classes for representing ''types'' of consumers, along with functions for solving (several flavors of) the one period consumption-saving problem. When run, model modules might demonstrate example specifications of their models, filling in the model parameters with arbitrary values. When **_ConsIndShockModel.py_** is run, it specifies an infinite horizon consumer with a particular discount factor, permanent income growth rate, coefficient of relative risk aversion (etc), who faces lognormal shocks to permanent and transitory income each period with a particular standard deviation; it then solves this consumer's problem and graphically displays the results. Model modules generally have **_Model_** in their name.

<!---
**Application modules** use tool and model modules to solve, simulate, and/or estimate economic models *for a particular purpose*.  While tool modules have no particular economic content and model modules describe entire classes of economic models, applications are uses of a model for some research purpose.  For example, ***/Demos/SolvingMicroDSOPs/StructEstimation.py*** uses a consumption-saving model from ***ConsIndShockModel.py***, calibrating it with age-dependent sequences of permanent income growth, survival probabilities, and the standard deviation of income shocks (etc); it then estimates the coefficient of relative risk aversion and shifter for an age-varying sequence of discount factors that best fits simulated wealth profiles to empirical data from the Survey of Consumer Finance.  A particular application might have multiple modules associated with it, all of which generally reside in one directory.
-->

## HARK

After you [installed](https://hark.readthedocs.io/en/latest/quick-start.html) and [cloned the repository of](https://github.com/econ-ark/HARK) HARK, you can explore the content of it. In the subfolder HARK, you can find a range of [general purpose tools](#general-purpose-tools), as well as the next subfolder ConsumptionSaving which has [AgentType Classes](#agenttype-class) and [Market Classes](#market-class).

### General Purpose Tools

HARK's root directory contains six tool modules,<sup id = "a1"> [1](#myfootnote1)</sup> each containing a variety of functions and classes that can be used in many economic models-- or even for mathematical purposes that have nothing to do with economics. Some of the tool modules are very sparely populated at this time, while others are quite large. We expect that all of these modules will grow considerably in the near future, as new tools are ''low hanging fruit'' for contribution to the project.<sup id = "a2"> [2](#myfootnote2)</sup>

<a name="myfootnote1">1</a>: The ''taxonomy'' of these modules is in flux; the functions described here could be combined into fewer modules or further divided by purpose. [↩](#a1)

<a name="myfootnote2">2</a>: That is, as the foundational, building-block elements of HARK, new tools are not difficult to program and do not require extensive integration with many moving parts. [↩](#a2)

#### HARK.core

A key goal of the project is to create modularity and interoperability between models, making them easy to combine, adapt, and extend. To this end, the **_HARK.core_** module specifies a framework for economic models in HARK, creating a common structure for them on two levels that can be called ''microeconomic'' and ''macroeconomic''.

Microeconomic models in HARK use the **_AgentType_** class to represent agents with an intertemporal optimization problem. Each of these models specifies a subclass of **_AgentType_**; an instance of the subclass represents agents who are _ex-ante_ homogeneous-- they have common values for all parameters that describe the problem. For example, **_ConsIndShockModel_** specifies the **_IndShockConsumerType_** class, which has methods specific to consumption-saving models with idiosyncratic shocks to income; an instance of the class might represent all consumers who have a CRRA of 3, discount factor of 0.98, etc. The **_AgentType_** class has a **_solve_** method that acts as a ''universal microeconomic solver'' for any properly formatted model, making it easier to set up a new model and to combine elements from different models; the solver is intended to encompass any model that can be framed as a sequence of one period problems. For a complete description, see section [AgentType Class](#agenttype-class).

Macroeconomic models in HARK use the **_Market_** class to represent a market (or other aggregator) that combines the actions, states, and/or shocks (generally, outcomes) of individual agents in the model into aggregate outcomes that are ''passed back'' to the agents. For example, the market in a consumption-saving model might combine the individual asset holdings of all agents in the market to generate aggregate capital in the economy, yielding the interest rate on assets (as the marginal product of capital); the individual agents then learn the aggregate capital level and interest rate, conditioning their next action on this information. Objects that microeconomic agents treat as exogenous when solving (or simulating) their model are thus endogenous at the macroeconomic level. Like **_AgentType_**, the **_Market_** class also has a **_solve_** method, which seeks out a dynamic general equilibrium: a ''rule'' governing the dynamic evolution of macroeconomic objects such that if agents believe this rule and act accordingly, then their collective actions generate a sequence of macroeconomic outcomes that justify the belief in that rule. For a more complete description, see section [Market Class](#market-class).

Beyond the model frameworks, **_HARK.core_** also defines a ''supersuperclass'' called **_HARKobject_**. When solving a dynamic microeconomic model with an infinite horizon (or searching for a dynamic general equilibrium), it is often required to consider whether two solutions are sufficiently close to each other to warrant stopping the process (i.e. approximate convergence). It is thus necessary to calculate the ''distance'' between two solutions, so HARK specifies that classes should have a **_distance_** method that takes a single input and returns a non-negative value representing the (generally dimensionless) distance between the object in question and the input to the method. As a convenient default, **_HARKobject_** provides a ''universal distance metric'' that should be useful in many contexts.<sup id = "a3"> [3](#myfootnote3)</sup> When defining a new subclass of **_HARKobject_**, the user simply defines the attribute **_distance_criteria_** as a list of strings naming the attributes of the class that should be compared when calculating the distance between two instances of that class. For example, the class **_ConsumerSolution_** has **_distance_criteria = ['cFunc']_**, indicating that only the consumption function attribute of the solution matters when comparing the distance between two instances of **_ConsumerSolution_**. See [here](https://hark.readthedocs.io/en/latest/reference/tools/core.html) for further documentation.

<a name="myfootnote3">3</a>: Roughly speaking, the universal distance metric is a recursive supnorm, returning the largest distance between two instances, among attributes named in **_distance_criteria_**. Those attributes might be complex objects themselves rather than real numbers, generating a recursive call to the universal distance metric. [↩](#a3)

#### HARK.utilities

The **_HARK.utilities_** module carries a double meaning in its name, as it contains both utility functions (and their derivatives, inverses, and combinations thereof) in the economic modeling sense as well as utilities in the sense of general tools. Utility functions included at this time are constant relative risk aversion and constant absolute risk aversion. Other functions in **_HARK.utilities_** include some data manipulation tools (e.g. for calculating an average of data conditional on being within a percentile range of different data), functions for constructing discrete state space grids, convenience functions for retrieving information about functions, and basic plotting tools using **_matplotlib.pyplot_**.

The module also includes functions for constructing discrete approximations to continuous distributions (e.g. **_approxLognormal()_** to approximate a log-normal distribution) as well as manipulating these representations (e.g. appending one outcome to an existing distribution, or combining independent univariate distributions into one multivariate distribution). As a convention in HARK, continuous distributions are approximated as finite discrete distributions when solving models; an _N_-dimensional random variable is formatted as a length _N+1_ list of 1D arrays, with the first element representing event probabilities and all other elements are realizations of the _N_ component RVs. This both simplifies solution methods (reducing numeric integrals to simple dot products) and allows users to easily test whether their chosen degree of discretization yields a sufficient approximation to the full distribution. See [here](https://hark.readthedocs.io/en/latest/reference/tools/utilities.html) for further documentation.

#### HARK.interpolation

The **_HARK.interpolation_** module defines classes for representing interpolated function approximations. Interpolation methods in HARK all inherit from a superclass such as **_HARKinterpolator1D_** or **_HARKinterpolator2D_**, wrapper classes that ensures interoperability across interpolation methods. For example, **_HARKinterpolator1D_** specifies the methods **\_**call**\_** and **_derivative_** to accept an arbitrary array as an input and return an identically shaped array with the interpolated function evaluated at the values in the array or its first derivative, respectively. However, these methods do little on their own, merely reshaping arrays and referring to the **_\_evaluate_** and **_\_der_** methods, which are _not actually defined in_ **_HARKinterpolator1D_**. Each subclass of **_HARKinterpolator1D_** specifies their own implementation of **_\_evaluate_** and **_\_der_** particular to that interpolation method, accepting and returning only 1D arrays. In this way, subclasses of **_HARKinterpolator1D_** are easily interchangeable with each other, as all methods that the user interacts with are identical, varying only by ''internal'' methods.

When evaluating a stopping criterion for an infinite horizon problem, it is often necessary to know the ''distance'' between functions generated by successive iterations of a solution procedure. To this end, each interpolator class in HARK must define a **_distance_** method that takes as an input another instance of the same class and returns a non-negative real number representing the ''distance'' between the two. As each of the **_HARKinterpolatorXD_** classes inherits from **_HARKobject_**, all interpolator classes have the default ''universal'' distance method; the user must simply list the names of the relevant attributes in the attribute **_distance_criteria_** of the class.

Interpolation methods currently implemented in HARK include (multi)linear interpolation up to 4D, 1D cubic spline interpolation, (multi)linear interpolation over 1D interpolations (up to 4D total), (multi)linear interpolation over 2D interpolations (up to 4D total), linear interpolation over 3D interpolations, 2D curvilinear interpolation over irregular grids, and a 1D ''lower envelope'' interpolator. See [here](https://hark.readthedocs.io/en/latest/reference/tools/interpolation.html) for further documentation.

#### HARK.simulation

The **_HARK.simulation_** module provides tools for generating simulated data or shocks for post-solution use of models. Currently implemented distributions include normal, lognormal, Weibull (including exponential), uniform, Bernoulli, and discrete. As an example of their use, these tools are used in the consumption-saving models of **_ConsIndShockModel.py_** to simulate permanent and transitory income shocks as well as unemployment events. See [here](https://hark.readthedocs.io/en/latest/reference/tools/simulation.html) for further documentation.

#### HARK.estimation

Methods for optimizing an objective function for the purposes of estimating a model can be found in **_HARK.estimation_**. As of this writing, the implementation includes only minimization by the Nelder-Mead simplex method, minimization by a derivative-free Powell method variant, and two small tools for resampling data (i.e. for a bootstrap); the minimizers are merely convenience wrappers (with result reporting) for optimizers included in **_scipy.optimize_**. Future functionality will include more robust global search methods, including genetic algorithms, simulated annealing, and differential evolution. See [here](https://hark.readthedocs.io/en/latest/reference/tools/estimation.html) for full documentation.

#### HARK.parallel

By default, processes in Python are single-threaded, using only a single CPU core. The **_HARK.parallel_** module provides basic tools for using multiple CPU cores simultaneously, with minimal effort.<sup id = "a4"> [4](#myfootnote4)</sup> In particular, it provides the function **_multiThreadCommands_**, which takes two arguments: a list of **_AgentType_**s and a list of commands as strings; each command should be a method of the **_AgentType_**s. The function simply distributes the **_AgentType_**s across threads on different cores and executes each command in order, returning no output (the **_AgentType_**s themselves are changed by running the commands). Equivalent results would be achieved by simply looping over each type and running each method in the list. Indeed, **_HARK.parallel_** also has a function called **_multiThreadCommandsFake_** that does just that, with identical syntax to **_multiThreadCommands_**; multithreading in HARK can thus be easily turned on and off.<sup id = "a5"> [5](#myfootnote5)</sup> The module also has functions for a parallel implementation of the Nelder-Mead simplex algorithm, as described in Wiswall and Lee (2011). See [here](https://hark.readthedocs.io/en/latest/reference/tools/parallel.html) for full documentation.

<a name="myfootnote4">4</a>: **_HARK.parallel_** uses two packages that aren't included in the default distribution of Anaconda: **_joblib_** and **_dill_**; see [here](https://hark.readthedocs.io/en/latest/quick-start.html#using-hark-with-anaconda) for instructions on how to install them. [↩](#a4)

<a name="myfootnote5">5</a>: In the future, **_HARK.parallel_** might be absorbed into **_HARK.core_** and **_HARK.estimation_**, particularly if **_joblib_** and **_dill_** become part of the standard Anaconda distribution. [↩](#a5)

### AgentType Class

The core of our microeconomic dynamic optimization framework is a flexible object-oriented representation of economic agents. The **_HARK.core_** module defines a superclass called **_AgentType_**; each model defines a subclass of **_AgentType_**, specifying additional model-specific features and methods while inheriting the methods of the superclass. Most importantly, the method **_solve_** acts as a ''universal solver'' applicable to any (properly formatted) discrete time model. This section describes the format of an instance of **_AgentType_** as it defines a dynamic microeconomic problem. Note that each instance of **_AgentType_** represents an _ex-ante_ heterogeneous ''type'' of agent; _ex-post_ heterogeneity is achieved by simulating many agents of the same type, each of whom receives a unique sequence of shocks.

#### Attributes of an AgentType

A discrete time model in our framework is characterized by a sequence of ''periods'' that the agent will experience. A well-formed instance of **_AgentType_** includes the following attributes:

- **_solveOnePeriod_**: A function, or a list of functions, representing the solution method for a single period of the agent's problem. The inputs passed to a **_solveOnePeriod_** function include all data that characterize the agent's problem in that period, including the solution to the subsequent period's problem (designated as **_solution_next_**). The output of these functions is a single **_Solution_** object, which can be passed to the solver for the previous period.

- **_time_inv_**: A list of strings containing all of the variable names that are passed to at least one function in **_solveOnePeriod_** but do _not_ vary across periods. Each of these variables resides in a correspondingly named attribute of the **_AgentType_** instance.

- **_time_vary_**: A list of strings naming the attributes of this instance that vary across periods. Each of these attributes is a list of period-specific values, which should be of the same length. If the solution method varies across periods, then **_'solveOnePeriod'_** is an element of **_time_vary_**.<sup id = "b1"> [6](#myfootnote6)</sup>

- **_solution_terminal_**: An object representing the solution to the ''terminal'' period of the model. This might represent a known trivial solution that does not require numeric methods, the solution to some previously solved ''next phase'' of the model, a scrap value function, or an initial guess of the solution to an infinite horizon model.

- **_pseudo_terminal_**: A Boolean flag indicating that **_solution_terminal_** is not a proper terminal period solution (rather an initial guess, ''next phase'' solution, or scrap value) and should not be reported as part of the model's solution.

- **_cycles_**: A non-negative integer indicating the number of times the agent will experience the sequence of periods in the problem. For example, **_cycles = 1_** means that the sequence of periods is analogous to a lifecycle model, experienced once from beginning to end; **_cycles = 2_** means that the agent experiences the sequence twice, with the first period in the sequence following the last. An infinite horizon problem in which the sequence of periods repeats indefinitely is indicated with **_cycles = 0_**.

- **_tolerance_**: A positive real number indicating convergence tolerance, representing the maximum acceptable ''distance'' between successive cycle solutions in an infinite horizon model; it is irrelevant when **_cycles > 0_**. As the distance metric on the space of solutions is model-specific, the value of **_tolerance_** is generally dimensionless.

- **_time_flow_**: A Boolean flag indicating the direction that time is ''flowing.'' When **_True_**, the variables listed in **_time_vary_** are listed in ordinary chronological order, with index 0 being the first period; when **_False_**, these lists are in reverse chronological order, with index 0 holding the last period.

An instance of **_AgentType_** also has the attributes named in **_time_vary_** and **_time_inv_**, and may have other attributes that are not included in either (e.g. values not used in the model solution, but instead to construct objects used in the solution).

<a name="myfootnote6">6</a>: **_time_vary_** may include attributes that are never used by a function in **_solveOnePeriod_**. Most saliently, the attribute **_solution_** is time-varying but is not used to solve individual periods. [↩](#b1)

#### A Universal Solver

When an instance of **_AgentType_** invokes its **_solve_** method, the solution to the agent's problem is stored in the attribute **_solution_**. The solution is computed by recursively solving the sequence of periods defined by the variables listed in **_time_vary_** and **_time_inv_** using the functions in **_solveOnePeriod_**. The time-varying inputs are updated each period, including the successive period's solution as **_solution_next_**; the same values of time invariant inputs in **_time_inv_** are passed to the solver in every period. The first call to **_solveOnePeriod_** uses **_solution_terminal_** as **_solution_next_**. In a finite horizon problem, the sequence of periods is solved **_cycles_** times over; in an infinite horizon problem, the sequence of periods is solved until the solutions of successive cycles have a ''distance'' of less than **_tolerance_**.

The output from a function in **_solveOnePeriod_** is an instance of a model-specific solution class. The attributes of a solution to one period of a problem might include behavioral functions, (marginal) value functions, and other variables characterizing the result. Each solution class must have a method called **_distance()_**, which returns the ''distance'' between itself and another instance of the same solution class, so as to define convergence as a stopping criterion; for many models, this will be the ''distance'' between a policy or value function in the solutions. If the solution class is defined as a subclass of **_HARKobject_**, it automatically inherits the default **_distance_** method, so that the user must only list the relevant object attributes in **_distance_criteria_**.

The **_AgentType_** also has methods named **_preSolve_** and **_postSolve_**, both of which take no arguments and do absolutely nothing. A subclass of **_AgentType_** can overwrite these blank methods with its own model specific methods. **_preSolve_** is automatically called near the beginning of the **_solve_** method, before solving the sequence of periods. It is used for specifying tasks that should be done before solving the sequence of periods, such as pre-constructing some objects repeatedly used by the solution method or finding an analytical terminal period solution. For example, the **_IndShockConsumerType_** class in **_ConsIndShockModel_** has a **_preSolve_** method that calls its **_updateSolutionTerminal_** method to ensure that **_solution_terminal_** is consistent with the model parameters. The **_postSolve_** method is called shortly after the sequence of periods is fully solved; it can be used for ''post-processing'' of the solution or performing a step that is only useful after solution convergence. For example, the **_TractableConsumerType_** in **_TractableBufferStockModel_** has a **_postSolve_** method that constructs an interpolated consumption function from the list of stable arm points found during solution.

Our universal solver is written in a very general way that should be applicable to any discrete time optimization problem-- because Python is so flexible in defining objects, the time-varying inputs for each period can take any form. Indeed, the solver does no ''real work'' itself, but merely provides a structure for describing models in the HARK framework, allowing interoperability among current and future modules.

#### The Flow of Time and Other Methods

Because dynamic optimization problems are solved recursively in our framework, it is natural to list time-varying values in reverse chronological order-- the **_solve()_** method loops over the values in each time-varying list in the same direction that a human would read them. When simulating agents after the solution has been obtained, however, it is much more convenient for time-varying parameters to be listed in ordinary chronological order-- the direction in which they will be experienced by simulated agents. To allow the user to set the order in which ''time is flowing'' for an instance of \*\*\*AgentType, the HARK framework includes functionality to easily change ordering of time-varying values.

The attribute **_time_flow_** is **_True_** if variables are listed in ordinary chronological order and **_False_** otherwise. **_AgentType_** has the following methods for manipulating time:

- **_timeReport()_**: Prints to screen a description of the direction that time is flowing, for interactive convenience and as a reminder of the functionality.

- **_timeFlip()_**: Flips the direction of time. Each attribute listed in **_time_vary_** is reversed in place, and the value of **_time_flow_** is toggled.

- **_timeFwd()_**: Sets the direction of time to ordinary chronological order.

- **_timeRev()_**: Sets the direction of time to reverse chronological order.

These methods are invoked to more conveniently access time-varying objects. When a new time-varying attribute is added, its name should be appended to **_time_vary_**, particularly if its values are used in the solution of the model (or is part of the solution itself). For example, the **_solve()_** method automatically adds the string **_'solution'_** to **_time_vary_** if it is not already present. Note that attributes listed in **_time_vary_** _must_ be lists if **_solve()_** or **_timeFlip()_** are used. Some values that could be considered ''time varying'' but are never used to solve the model are more conveniently represented as a **_numpy.array_** object (e.g. the history of a state or control variable from a simulation); because the **_numpy.array_** class does not have a **_reverse()_** method, these attributes should not be listed in **_time_vary_**.

The base **_AgentType_** is sparsely defined, as most ''real'' methods will be application-specific. Two final methods bear mentioning. First, the **\_**call**\_** method points to **_assignParameters()_**, a convenience method for adding or adjusting attributes (inherited from **_HARKobject_**). This method takes any number of keyword arguments, so that code can be parsimoniously written as, for example, **_AgentInstance(attribute1 = value1, attribute2 = value2)_**. Using Python's dictionary capabilities, many attributes can be conveniently set with minimal code. Second, the **_resetRNG_** method simply resets the **_AgentType_**'s random number generator (as the attribute **_RNG_**) using the value in the attribute **_seed_**.<sup id = "b2"> [7](#myfootnote7)</sup> This method is useful for (_inter alia_) ensuring that the same underlying sequence of shocks is used for every simulation run when a model is solved or estimated.

<a name="myfootnote7">7</a>: Every instance of **_AgentType_** is created with a random number generator as an instance of the class **_numpy.random.RandomState_**, with a default **_seed_** of zero. [↩](#b2)

### Market Class

The modeling framework of **_AgentType_** is deemed ''microeconomic'' because it pertains only to the dynamic optimization problem of agents, treating all inputs of the problem as exogenously fixed. In what we label as ''macroeconomic'' models, some of the inputs for the microeconomic models are endogenously determined by the collective states and controls of agents in the model. In a dynamic general equilibrium, there must be consistency between agents' beliefs about these macroeconomic objects, their individual behavior, and the realizations of the macroeconomic objects that result from individual choices.

The **_Market_** class in **_HARK.core_** provides a framework for such macroeconomic models, with a **_solve_** method that searches for a dynamic general equilibrium. An instance of **_Market_** includes a list of **_AgentType_**s that compose the economy, a method for transforming microeconomic outcomes (states, controls, and/or shocks) into macroeconomic outcomes, and a method for interpreting a history or sequence of macroeconomic outcomes into a new ''dynamic rule'' for agents to believe. Agents treat the dynamic rule as an input to their microeconomic problem, conditioning their optimal policy functions on it. A dynamic general equilibrium is a fixed point dynamic rule: when agents act optimally while believing the equilibrium rule, their individual actions generate a macroeconomic history consistent with the equilibrium rule.

#### Down on the Farm

The **_Market_** class uses a farming metaphor to conceptualize the process for generating a history of macroeconomic outcomes in a model. Suppose all **_AgentTypes_** in the economy believe in some dynamic rule (i.e. the rule is stored as attributes of each **_AgentType_**, which directly or indirectly enters their dynamic optimization problem), and that they have each found the solution to their microeconomic model using their **_solve_** method. Further, the macroeconomic and microeconomic states have been reset to some initial orientation.

To generate a history of macroeconomic outcomes, the **_Market_** repeatedly loops over the following steps a set number of times:

- **_sow_**: Distribute the macroeconomic state variables to all **_AgentType_**s in the market.

- **_cultivate_**: Each **_AgentType_** executes their **_marketAction_** method, likely corresponding to simulating one period of the microeconomic model.

- **_reap_**: Microeconomic outcomes are gathered from each **_AgentType_** in the market.

- **_mill_**: Data gathered by **_reap_** is processed into new macroeconomic states according to some ''aggregate market process''.

- **_store_**: Relevant macroeconomic states are added to a running history of outcomes.

This procedure is conducted by the **_makeHistory_** method of **_Market_** as a subroutine of its **_solve_** method. After making histories of the relevant macroeconomic variables, the market then executes its **_calcDynamics_** function with the macroeconomic history as inputs, generating a new dynamic rule to distribute to the **_AgentType_**s in the market. The process then begins again, with the agents solving their updated microeconomic models given the new dynamic rule; the **_solve_** loop continues until the ''distance'' between successive dynamic rules is sufficiently small.

#### Attributes of a Market

To specify a complete instance of **_Market_**, the user should give it the following attributes:<sup id = "c1"> [8](#myfootnote8)</sup>

- **_agents_**: A list of **_AgentType_**s, representing the agents in the market. Each element in **_agents_** represents an _ex-ante_ heterogeneous type; each type could have many _ex-post_ heterogeneous agents.

- **_sow_vars_**: A list of strings naming variables that are output from the aggregate market process, representing the macroeconomic outcomes. These variables will be distributed to the **_agents_** in the **_sow_** step.

- **_reap_vars_**: A list of strings naming variables to be collected from the **_agents_** in the **_reap_** step, to be used as inputs for the aggregate market process.

- **_const_vars_**: A list of strings naming variables used by the aggregate market process that _do not_ come from **_agents_**; they are constant or come from the **_Market_** itself.

- **_track_vars_**: A list of strings naming variables generated by the aggregate market process that should be tracked as a history, to be used when calculating a new dynamic rule. Usually a subset of **_sow_vars_**.

- **_dyn_vars_**: A list of strings naming the variables that constitute a dynamic rule. These will be stored as attributes of the **_agents_** whenever a new rule is calculated.

- **_millRule_**: A function for the ''aggregate market process'', transforming microeconomic outcomes into macroeconomic outcomes. Its inputs are named in **_reap_vars_** and **_const_vars_**, and it returns a single object with attributes named in **_sow_vars_** and/or **_track_vars_**. Can be defined as a method of a subclass of **_Market_**.

- **_calcDynamics_**: A function that generates a new dynamic rule from a history of macroeconomic outcomes. Its inputs are named in **_track_vars_**, and it returns a single object with attributes named in **_dyn_vars_**.

- **_act_T_**: The number of times that the **_makeHistory_** method should execute the ''farming loop'' when generating a new macroeconomic history.

- **_tolerance_**: The minimum acceptable ''distance'' between successive dynamic rules produced by **_calcDynamics_** to constitute a sufficiently converged solution.

Further, each **_AgentType_** in **_agents_** must have two methods not necessary for microeconomic models; neither takes any input (except **_self_**):

- **_marketAction_**: The microeconomic process to be run in the **_cultivate_** step. Likely uses the new macroeconomic outcomes named in **_sow_vars_**; should store new values of relevant microeconomic outcomes in the attributes (of **_self_**) named in **_reap_vars_**.

- **_reset_**: Reset, initialize, or prepare for a new ''farming loop'' to generate a macroeconomic history. Might reset its internal random number generator, set initial state variables, clear personal histories, etc.

When solving macroeconomic models in HARK, the user should also define classes to represent the output from the aggregate market process in **_millRule_** and for the model-specific dynamic rule. The latter should have a **_distance_** method to test for solution convergence; if the class inherits from **_HARKobject_**, the user need only list relevant attributes in **_distance_criteria_**.

<a name="myfootnote8">8</a>: For some purposes, it might be useful to specify a subclass of **_Market_**, defining **_millRule_** and/or **_calcDynamics_** as methods rather than functions. [↩](#c1)

## DemARK

If you want to get a feeling for how the code works and what you can do with it, check out the DemARK [repository](https://github.com/econ-ark/DEMARK) which contains many useful demonstrations of tools, AgentTypes, and ModelClasses.

If you want to run the notebooks on your own machine make sure to install the necessary packages described in the readme file. Afterwards you can dive in the notebook folder. Each example has a markdown (.md) version with explanatory notes. The notebook (.ipynb) describes the method and runs (part of the) code.

## REMARK

HARK can be used to replicate papers as well. For this purpose the _R_[eplications/eproductions] and *E*xplorations *M*ade using _ARK_ (REMARK) [repository](https://github.com/econ-ark/REMARK) was created.

Each replication consists of a _metadata file_ (.md) with an overview, a _notebook_ which replicates the paper, and a _requirement.txt_ file with the necessary packages to run the notebooks on your local mashine.

# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: collapsed,code_folding,name,title,incorrectly_encoded_metadata,pycharm
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3.9 (XPython)
#     language: python
#     name: xpython
# ---

# %% [markdown]
# # PerfForesightConsumerType: Perfect foresight consumption-saving


# %% code_folding=[0]
# Initial imports and notebook setup, click arrow to show

from copy import copy

import matplotlib.pyplot as plt
plt.ion # interactive figures
import numpy as np

from HARK.ConsumptionSaving.ConsIndShockModel import PerfForesightConsumerType
from HARK.utilities import plot_funcs

mystr = lambda number: "{:.4f}".format(number)

# %% [markdown]
# The module `HARK.ConsumptionSaving.ConsIndShockModel` concerns consumption-saving models with idiosyncratic shocks to (non-capital) income.  All of the models assume CRRA utility with geometric discounting, no bequest motive, and income shocks are fully transitory or fully permanent.
#
# `ConsIndShockModel` currently includes three models:
# 1. A very basic "perfect foresight" model with no uncertainty.
# 2. A model with risk over transitory and permanent income shocks.
# 3. The model described in (2), with an interest rate for debt that differs from the interest rate for savings.
#
# This notebook provides documentation for the first of these three models.
# $\newcommand{\CRRA}{\rho}$
# $\newcommand{\DiePrb}{\mathsf{D}}$
# $\newcommand{\LivPrb}{\Pi}$
# $\newcommand{\PermGroFac}{\Gamma}$
# $\newcommand{\Rfree}{\mathsf{R}}$
# $\newcommand{\DiscFac}{\beta}$

# %% [markdown]
# ## Statement of perfect foresight consumption-saving model
#
# The `PerfForesightConsumerType` class the problem of a consumer with Constant Relative Risk Aversion utility
# ${\CRRA}$
# \begin{equation}
# U(C) = \frac{C^{1-\CRRA}}{1-\rho},
# \end{equation}
# has perfect foresight about everything except whether he will survive between the end of period $t$ and the beginning of period $t+1$, which occurs with probability $\LivPrb_{t+1}$.  Permanent labor income $P_t$ grows from period $t$ to period $t+1$ by factor $\PermGroFac_{t+1}$.
#
# At the beginning of period $t$, the consumer has an amount of market resources $M_t$ (which includes both market wealth and currrent income) and must choose how much of those resources to consume $C_t$ and how much to retain in a riskless asset $A_t$, which will earn return factor $\Rfree$.  The consumer cannot necessarily borrow arbitarily; instead, he might be constrained to have a wealth-to-income ratio at least as great as some "artificial borrowing constraint" $\underline{a} \leq 0$.
#
# The agent's flow of future utility $U(C_{t+n})$ from consumption is geometrically discounted by factor $\DiscFac$ per period. If the consumer dies, he receives zero utility flow for the rest of time.
#
# The agent's problem can be written in Bellman form as:
#
# \begin{eqnarray*}
# V_t(M_t,P_t) &=& \max_{C_t}~U(C_t) ~+ \DiscFac \LivPrb_{t+1} V_{t+1}(M_{t+1},P_{t+1}), \\
# & s.t. & \\
# A_t &=& M_t - C_t, \\
# A_t/P_t &\geq& \underline{a}, \\
# M_{t+1} &=& \Rfree A_t + Y_{t+1}, \\
# Y_{t+1} &=& P_{t+1}, \\
# P_{t+1} &=& \PermGroFac_{t+1} P_t.
# \end{eqnarray*}
#
# The consumer's problem is characterized by a coefficient of relative risk aversion $\CRRA$, an intertemporal discount factor $\DiscFac$, an interest factor $\Rfree$, and age-varying sequences of the permanent income growth factor $\PermGroFac_t$ and survival probability $\LivPrb_{t}$.
#
# While it does not reduce the computational complexity of the problem (as permanent income is deterministic, given its initial condition $P_0$), HARK represents this problem with *normalized* variables (represented in lower case), dividing all real variables by permanent income $P_t$ and utility levels by $P_t^{1-\CRRA}$.  The Bellman form of the model thus reduces to:
#
# \begin{eqnarray*}
# v_t(m_t) &=& \max_{c_t}~u(c_t) ~+ \DiscFac \LivPrb_{t+1} \PermGroFac_{t+1}^{1-\CRRA} v_{t+1}(m_{t+1}), \\
# & s.t. & \\
# a_t &=& m_t - c_t, \\
# a_t &\geq& \underline{a}, \\
# m_{t+1} &=& \Rfree/\PermGroFac_{t+1} a_t + 1.
# \end{eqnarray*}
#
# whose first order condition is 
#
# \begin{align}
# u^{\prime}(c_{t}) & = \DiscFac \LivPrb_{t+1} \PermGroFac_{t+1}^{-\CRRA} v_{t+1}^{\prime}(a_{t}(R/\PermGroFac)+1)
# \\ & \equiv \mathfrak{v}_{t}^{\prime}(a_{t}), 
# \end{align}
# where $\mathfrak{v}_{t}(a_{t})$ is the value of ending period $t$ with assets $a_{t}$.

# %% [markdown]
# \begin{eqnarray*}
# v_t(m_t) &=& u(c_t) ~+ \DiscFac \LivPrb_{t+1} \PermGroFac_{t+1}^{1-\CRRA} v_{t+1}(m_{t+1}), \\
# &=& u(c_t)\sum_{t}^{t+h-1}()+(\DiscFac \LivPrb_{t+1} \PermGroFac_{t+1}^{1-\CRRA})^{h} v_{t+h}(1.)\\
# v_t(m_t)-(\DiscFac \LivPrb_{t+1} \PermGroFac_{t+1}^{1-\CRRA})^{h} v_{t+h}(1.)&=& u(c_t)\sum_{t}^{t+h-1}()\\
# \end{eqnarray*}

# %% [markdown]
# ## Solution method for PerfForesightConsumerType
#
# Because of the assumptions of CRRA utility, no risk other than mortality, and no artificial borrowing constraint, the problem has a closed form solution.  In fact, in the absence of a liquidity constraint that could ever bind (equivalently, for $\underline{a} = -\infty$), the consumption function is perfectly linear, and the value function composed with the inverse utility function is also linear.  The mathematical solution of this model is described in detail in the lecture notes [PerfForesightCRRA](https://www.econ2.jhu.edu/people/ccarroll/public/lecturenotes/consumption/PerfForesightCRRA), which also demonstrates that the Euler equation for the problem is
#
# \begin{align}
# u^{\prime}(c_{t}) & = R \beta \LivPrb u^{\prime}(c_{t+1})
# \end{align}
#
# The one period problem for this model is solved by the function `solveConsPerfForesight`, which creates an instance of the class `ConsPerfForesightSolver`. To construct an instance of the class `PerfForesightConsumerType`, several parameters must be passed to its constructor as shown in the table below.

# %% [markdown]
# ## Example parameter values to construct an instance of PerfForesightConsumerType
#
# | Parameter | Description | Code | Example value | Time-varying? |
# | :---: | --- | --- | --- | --- |
# | $\DiscFac$ |Intertemporal discount factor  | $\texttt{DiscFac}$ | $0.96$ |  |
# | $\CRRA $ |Coefficient of relative risk aversion | $\texttt{CRRA}$ | $2.0$ | |
# | $\Rfree$ | Risk free interest factor | $\texttt{Rfree}$ | $1.03$ | |
# | $\LivPrb_{t+1}$ |Survival probability | $\texttt{LivPrb}$ | $[0.98]$ | $\surd$ |
# |$\PermGroFac_{t+1}$|Permanent income growth factor|$\texttt{PermGroFac}$| $[1.01]$ | $\surd$ |
# |$\underline{a}$|Artificial borrowing constraint|$\texttt{BoroCnstArt}$| $None$ |  |
# |$(none)$|Maximum number of gridpoints in consumption function |$\texttt{aXtraCount}$| $200$ |  |
# |$T$| Number of periods in this type's "cycle" |$\texttt{T_cycle}$| $1$ | |
# |(none)| Number of times the "cycle" occurs |$\texttt{cycles}$| $0$ | |
#
# Note that the survival probability and income growth factor have time subscripts; likewise, the example values for these parameters are *lists* rather than simply single floats.  This is because those parameters are *time-varying*: their values can depend on which period of the problem the agent is in.  All time-varying parameters *must* be specified as lists, even if the same value occurs in each period for this type.
#
# The artificial borrowing constraint can be any non-positive `float`, or it can be `None` to indicate no artificial borrowing constraint.  The maximum number of gridpoints in the consumption function is only relevant if the borrowing constraint is not `None`; without an upper bound on the number of gridpoints, kinks in the consumption function will propagate indefinitely in an infinite horizon model if there is a borrowing constraint, eventually resulting in an overflow error.  If there is no artificial borrowing constraint, then the number of gridpoints used to represent the consumption function is always exactly two.
#
# The last two parameters in the table specify the "nature of time" for this type: the number of (non-terminal) periods in this type's "cycle", and the number of times that the "cycle" occurs.  *Every* subclass of `AgentType` uses these two code parameters to define the nature of time.  Here, `T_cycle` has the value $1$, indicating that there is exactly one period in the cycle, while `cycles` is $0$, indicating that the cycle is repeated in *infinite* number of times-- it is an infinite horizon model, with the same "kind" of period repeated over and over.
#
# In contrast, we could instead specify a life-cycle model by setting `T_cycle` to $1$, and specifying age-varying sequences of income growth and survival probability.  In all cases, the number of elements in each time-varying parameter should exactly equal $\texttt{T_cycle}$.
#
# The parameter $\texttt{AgentCount}$ specifies how many consumers there are of this *type*-- how many individuals have these exact parameter values and are *ex ante* homogeneous.  This information is not relevant for solving the model, but is needed in order to simulate a population of agents, introducing *ex post* heterogeneity through idiosyncratic shocks.  Of course, simulating a perfect foresight model is quite boring, as there are *no* idiosyncratic shocks other than death!
#
# The cell below defines a dictionary that can be passed to the constructor method for `PerfForesightConsumerType`, with the values from the table here.

# %% code_folding=[]
PerfForesightDict = {
    # Parameters actually used in the solution method
    "CRRA": 2.0,  # Coefficient of relative risk aversion
    "Rfree": 1.03,  # Interest factor on assets
    "DiscFac": 0.96,  # Default intertemporal discount factor
    "LivPrb": [0.98],  # Survival probability
    "PermGroFac": [1.01],  # Permanent income growth factor
    "BoroCnstArt": None,  # Artificial borrowing constraint
    "aXtraCount": 200,  # Maximum number of gridpoints in consumption function
    # Parameters that characterize the nature of time
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    "cycles": 0,  # Number of times the cycle occurs (0 --> infinitely repeated)
}

# %% [markdown]
# ## Solving and examining the solution of the perfect foresight model
#
# With the dictionary we have just defined, we can create an instance of `PerfForesightConsumerType` by passing the dictionary to the class (as if the class were a function).  This instance can then be solved by invoking its `solve` method.

# %%
PFexample = PerfForesightConsumerType(**PerfForesightDict)
PFexample.cycles = 0
PFexample.solve()

# %% [markdown]
# The $\texttt{solve}$ method fills in the instance's attribute `solution` as a time-varying list of solutions to each period of the consumer's problem.  In this case, `solution` will be a list with exactly one instance of the class `ConsumerSolution`, representing the solution to the infinite horizon model we specified.

# %%
print(PFexample.solution)

# %% [markdown]
# Each element of `solution` has a few attributes. To see all of them, we can use the \texttt{vars} built in function:
#
# the consumption functions reside in the attribute $\texttt{cFunc}$ of each element of `ConsumerType.solution`.  This method creates a (time varying) attribute $\texttt{cFunc}$ that contains a list of consumption functions.

# %%
print(vars(PFexample.solution[0]))

# %% [markdown]
# The two most important attributes of a single period solution of this model are the (normalized) consumption function $\texttt{cFunc}$ and the (normalized) value function $\texttt{vFunc}$.  Let's plot those functions near the lower bound of the permissible state space (the attribute $\texttt{mNrmMin}$ tells us the lower bound of $m_t$ where the consumption function is defined).

# %%
print("Linear perfect foresight consumption function:")
mMin = PFexample.solution[0].mNrmMin
plot_funcs(PFexample.solution[0].cFunc, mMin, mMin + 10.0)

# %%
print("Perfect foresight value function:")
plot_funcs(PFexample.solution[0].vFunc, mMin + 0.1, mMin + 10.1)

# %% [markdown]
# An element of `solution` also includes the (normalized) marginal value function $\texttt{vPfunc}$, and the lower and upper bounds of the marginal propensity to consume (MPC) $\texttt{MPCmin}$ and $\texttt{MPCmax}$.  Note that with a linear consumption function, the MPC is constant, so its lower and upper bound are identical.
#
# ### Liquidity constrained perfect foresight example
#
# Without an artificial borrowing constraint, a perfect foresight consumer is free to borrow against the PDV of the entire future stream of labor income-- "human wealth" $\texttt{hNrm}$-- and will consume a constant proportion of total wealth (market resources plus human wealth).  If we introduce an artificial borrowing constraint, both of these features vanish.  In the cell below, we define a parameter dictionary that prevents the consumer from borrowing *at all*, create and solve a new instance of `PerfForesightConsumerType` with it, and then plot its consumption function.

# %% pycharm={"name": "#%%\n"}
LiqConstrDict = copy(PerfForesightDict)
LiqConstrDict["BoroCnstArt"] = 0.0  # Set the artificial borrowing constraint to zero

LiqConstrExample = PerfForesightConsumerType(**LiqConstrDict)
LiqConstrExample.cycles = 0  # Make this type be infinite horizon
LiqConstrExample.solve()

print("Liquidity constrained perfect foresight consumption function:")
plot_funcs(LiqConstrExample.solution[0].cFunc, 0.0, 10.0)

# %% pycharm= [markdown] {"name": "#%% md\n"}
# At this time, the value function for a perfect foresight consumer with an artificial borrowing constraint is not computed nor included as part of its $\texttt{solution}$.

# %% [markdown]
# ## Simulating the perfect foresight consumer model
#
# Suppose we wanted to simulate many consumers who share the parameter values that we passed to `PerfForesightConsumerType`-- an *ex ante* homogeneous *type* of consumers.  To do this, our instance would have to know *how many* agents there are of this type, as well as their initial levels of assets $a_t$ and permanent income $P_t$.
#
# ### Setting simulation parameters
#
# Let's fill in this information by passing another dictionary to `PFexample` with simulation parameters.  The table below lists the parameters that an instance of `PerfForesightConsumerType` needs in order to successfully simulate its model using the `simulate` method.
#
# | Description | Code | Example value |
# | :---: | --- | --- |
# | Number of consumers of this type | $\texttt{AgentCount}$ | $10000$ |
# | Number of periods to simulate | $\texttt{T_sim}$ | $120$ |
# | Mean of initial log (normalized) assets | $\texttt{aNrmInitMean}$ | $-6.0$ |
# | Stdev of initial log  (normalized) assets | $\texttt{aNrmInitStd}$ | $1.0$ |
# | Mean of initial log permanent income | $\texttt{pLvlInitMean}$ | $0.0$ |
# | Stdev of initial log permanent income | $\texttt{pLvlInitStd}$ | $0.0$ |
# | Aggregrate productivity growth factor | $\texttt{PermGroFacAgg}$ | $1.0$ |
# | Age after which consumers are automatically killed | $\texttt{T_age}$ | $None$ |
#
# We have specified the model so that initial assets and permanent income are both distributed lognormally, with mean and standard deviation of the underlying normal distributions provided by the user.
#
# The parameter $\texttt{PermGroFacAgg}$ exists for compatibility with more advanced models that employ aggregate productivity shocks; it can simply be set to 1.
#
# In infinite horizon models, it might be useful to prevent agents from living extraordinarily long lives through a fortuitous sequence of mortality shocks.  We have thus provided the option of setting $\texttt{T_age}$ to specify the maximum number of periods that a consumer can live before they are automatically killed (and replaced with a new consumer with initial state drawn from the specified distributions).  This can be turned off by setting it to `None`.
#
# The cell below puts these parameters into a dictionary, then gives them to `PFexample`.  Note that all of these parameters *could* have been passed as part of the original dictionary; we omitted them above for simplicity.

# %% pycharm={"name": "#%%\n"}
SimulationParams = {
    "AgentCount": 10000,  # Number of agents of this type
    "T_sim": 120,  # Number of periods to simulate
    "aNrmInitMean": -6.0,  # Mean of log initial assets
    "aNrmInitStd": 1.0,  # Standard deviation of log initial assets
    "pLvlInitMean": 0.0,  # Mean of log initial permanent income
    "pLvlInitStd": 0.0,  # Standard deviation of log initial permanent income
    "PermGroFacAgg": 1.0,  # Aggregate permanent income growth factor
    "T_age": None,  # Age after which simulated agents are automatically killed
}

PFexample.assign_parameters(**SimulationParams)

# %% [markdown] pycharm= [markdown] {"name": "#%% md\n"}
# To generate simulated data, we need to specify which variables we want to track the "history" of for this instance.  To do so, we set the `track_vars` attribute of our `PerfForesightConsumerType` instance to be a list of strings with the simulation variables we want to track.
#
# In this model, valid arguments to `track_vars` include $\texttt{mNrm}$, $\texttt{cNrm}$, $\texttt{aNrm}$, and $\texttt{pLvl}$.  Because this model has no idiosyncratic shocks, our simulated data will be quite boring.
#
# ### Generating simulated data
#
# Before simulating, the `initialize_sim` method must be invoked.  This resets our instance back to its initial state, drawing a set of initial $\texttt{aNrm}$ and $\texttt{pLvl}$ values from the specified distributions and storing them in the attributes $\texttt{aNrmNow_init}$ and $\texttt{pLvlNow_init}$.  It also resets this instance's internal random number generator, so that the same initial states will be set every time `initialize_sim` is called.  In models with non-trivial shocks, this also ensures that the same sequence of shocks will be generated on every simulation run.
#
# Finally, the `simulate` method can be called.

# %% pycharm={"name": "#%%\n"}
PFexample.track_vars = ['mNrm']
PFexample.initialize_sim()
PFexample.simulate()

# %% pycharm= [markdown] {"name": "#%% md\n"}
# Each simulation variable $\texttt{X}$ named in $\texttt{track_vars}$ will have the *history* of that variable for each agent stored in the attribute $\texttt{X_hist}$ as an array of shape $(\texttt{T_sim},\texttt{AgentCount})$.  To see that the simulation worked as intended, we can plot the mean of $m_t$ in each simulated period:

# %% pycharm={"name": "#%%\n"}
plt.plot(np.mean(PFexample.history['mNrm'], axis=1))
plt.xlabel("Time")
plt.ylabel("Mean normalized market resources")
plt.show()

# %% [markdown] pycharm= [markdown] {"name": "#%% md\n"}
# A perfect foresight consumer can borrow against the PDV of his future income-- his human wealth-- and thus as time goes on, our simulated agents approach the (very negative) steady state level of $m_t$ while being steadily replaced with consumers with roughly $m_t=1$.
#
# The slight wiggles in the plotted curve are due to consumers randomly dying and being replaced; their replacement will have an initial state drawn from the distributions specified by the user.  To see the current distribution of ages, we can look at the attribute $\texttt{t_age}$.

# %% pycharm={"name": "#%%\n"}
N = PFexample.AgentCount
F = np.linspace(0.0, 1.0, N)
plt.plot(np.sort(PFexample.t_age), F)
plt.xlabel("Current age of consumers")
plt.ylabel("Cumulative distribution")
plt.show()

# %% [markdown] pycharm= [markdown] {"name": "#%% md\n"}
# The distribution is (discretely) exponential, with a point mass at 120 with consumers who have survived since the beginning of the simulation.
#
# One might wonder why HARK requires users to call `initialize_sim` before calling `simulate`: Why doesn't `simulate` just call `initialize_sim` as its first step?  We have broken up these two steps so that users can simulate some number of periods, change something in the environment, and then resume the simulation.
#
# When called with no argument, `simulate` will simulate the model for $\texttt{T_sim}$ periods.  The user can optionally pass an integer specifying the number of periods to simulate (which should not exceed $\texttt{T_sim}$).
#
# In the cell below, we simulate our perfect foresight consumers for 80 periods, then seize a bunch of their assets (dragging their wealth even more negative), then simulate for the remaining 40 periods.
#
# The `state_prev` attribute of an AgenType stores the values of the model's state variables in the _previous_ period of the simulation.

# %% pycharm={"name": "#%%\n"}
PFexample.initialize_sim()
PFexample.simulate(80)
PFexample.state_prev['aNrm'] += -5.0  # Adjust all simulated consumers' assets downward by 5
PFexample.simulate(40)

plt.plot(np.mean(PFexample.history['mNrm'], axis=1))
plt.xlabel("Time")
plt.ylabel("Mean normalized market resources")
plt.show()

# %% [markdown]
# ### Appendix: Derivation of Analytical Formulae for Liquidity Constraints

# %% [markdown]
#
# In the simple case where there is a constraint that requires the consumer to end the period with nonnegative assets, we can obtain a closed form solution for the liquidity constrained PF consumption function as follows.
#
# Consider the consumer as of the end of the penultimate period $T-1$.  There will be some value $m_{T-1}^{\#,1}$ such that for any $m > m_{T-1}^{\#,1}$ the unconstrained consumer would wish to end the period with positive assets, while for $m < m_{T-1}^{\#,1}$ an unconstrained consumer would borrow.
#
# With a CRRA utility function for which marginal utility is $u^{\prime}(c)=c^{-\rho}$, the Euler equation above says
# \begin{align}
# (c_{T-1}^{\#,1})^{-\rho} & = \mathfrak{v}^{\prime}_{T-1}(\underline{a}_{T-1})
# \\ m^{\#} \equiv m_{T-1}^{\#,1} = c_{T-1}^{\#,1} + \underline{a}_{T-1}& = \left(\mathfrak{v}^{\prime}_{T-1}(\underline{a}_{T-1})\right)^{-1/\CRRA}
# \end{align}
#
# Under the [relevant impatience conditions](https://econ-ark.github.io/BufferStockTheory), we can show that the constraint will bind for $m_{T-n} <m^{\#}$ for all earlier periods (that is, $\forall ~ n>0$) as well.  

# %% [markdown]
# The corresponding value function is 
# \begin{align}
# v_{T-1} &= u(c_{T-1})+ \beta_{T-1} \Gamma_{T-1}^{1-\rho} v_{T}
# \end{align}
# where we assume that the terminal value function has two components:  
# \begin{align}
# v_{T}(m) & = u(c_{T}(m)) + \mathcal{V}_{T}(m-c_{T}(m))
# \end{align}
# where $\mathcal{V}$ is the utility from any bequest made from unconsumed resources.  
#
# For the present, we assume that $\mathcal{V}_{T}(a)=0$:  No utility is gained from unspent assets at death.  (Below see a discussion of an alternative).
#
# In this case, the consumer who chooses to end the penultimate period with a positive amount of assets will consume all remaining assets in period $T$, resulting in a terminal value function of $v_{T}(m_{T})=u(c_{T}(m_{T})=u(c_{T})$.  The Euler equation will not bind for such a consumer, and if minimum income next period is 1, then a consumer on the cusp where the constraint makes a transition from binding to not binding will satisfy
#
# \begin{align}
# \mathfrak{v}^{\prime}_{T-1}(\underline{a}_{T-1}) & = \beta \Gamma_{T}^{-\rho} 1^{-\rho}
# \\ c^{\#,1}_{T-1} & = \left(\beta \Gamma_{T}^{-\rho} 1^{-\rho}\right)^{-1/\CRRA}
# \end{align}
#
# If the consumer satisfies the relevant borrowing constraints, the constraint will also bind in period $T-2$, at the point 
# \begin{align}
# \mathfrak{v}^{\prime}_{T-1}(\underline{a}_{T-1}) & = \beta \Gamma_{T}^{-\rho} 1^{-\rho}
# \\ c^{\#,1}_{T-1} & = \left(\beta \Gamma_{T}^{-\rho} 1^{-\rho}\right)^{-1/\CRRA}
# \end{align}
#
#
# #### Stone Geary Bequests
#
# We now consider a bequest function of a Stone-Geary form like:
# \begin{align}
# \mathcal{V}_{T}(a) & = \left(\frac{(\eta + a)^{1-\rho}}{1-\rho}\right)\Upsilon 
# \end{align}
# where $\eta$ is an intercept that has the effect of causing bequests to be a luxury good: people with an absolute level of market resources $m_{T}$ below a certain level will wish to leave no bequest, $a_{T}=0$.  (As wealth gets arbitrarily large, the ratio of bequest wealth to last period consumption approaches a constant whose size depends on $\Upsilon$).
#
# For a consumer with $m_{T} > m_{T}^{\#,0}$,
# \begin{align}
# v_{T}(m) & = \max_{c} u(c)+\Upsilon u(\eta+(m-c))
# \end{align}
# has FOC
# \begin{align}
# c^{-\rho} & = \Upsilon (\eta+(m-c))^{-\rho}
# \\ c & = \Upsilon^{-1/\rho} (\eta+m-c)
# \\ (1+\Upsilon^{1/\rho})c & = \eta+m
# \\ c & = \left(\frac{\eta+m}{1+\Upsilon^{1/\rho}}\right)
# \end{align}
# and so the point at which the constraint $(m-c) \geq 0$ begins to bind is:
# \begin{align}
# m & = \left(\frac{\eta+m}{(1+\Upsilon^{1/\rho})}\right) 
# \\ (1+\Upsilon^{1/\rho})m & = \left(\eta+m\right) 
# \\ m & = \eta \Upsilon^{1/\rho}
# %
# %\\ \left(\left(\frac{1+\Upsilon^{-1/\rho}}{1+\Upsilon^{-1/\rho}}\right)- \left(\frac{1}{1+\Upsilon^{-1/\rho}}\right)\right) m & = \left(\frac{\eta}{(1+\Upsilon^{-1/\rho})}\right)
# %\\ m & = \frac{\left(\frac{\eta}{(1+\Upsilon^{-1/\rho})}\right)}{1- \left(\frac{1}{(1+\Upsilon^{-1/\rho})}\right)}
# \end{align}
#

# %% [markdown]
# It is easy to show that the ratio of $(\eta + a_{T})$ to $c_{T}$ will be constant at some $\Phi_{T}$, so for this unconstrained consumer value will be 
# \begin{align}
# v^{u}_{T}(m) & = u(c_{T}(m)) + \Upsilon u(\Phi_{T} c_{T}) \\
# (1-\rho)v_{T}(m) & = c_{T}^{1-\rho} + \Upsilon (\Phi_{T} c_{T})^{1-\rho}
# \\ & = c_{T}^{1-\rho}\left(1 + \Upsilon \Phi_{T}\right)
# \\ \Lambda_{T}^{u} \equiv u^{-1}(v_{T}) & = c_{T}\left(1 + \Upsilon \Phi_{T}\right)^{1/(1-\rho)}
# \end{align}
# and the budget constraint requires that
# \begin{align}
# m_{T} & = c_{T} + a_{T} \\
# \eta + a_{T} &= \Phi_{T} c_{T} \\
# a_{T} &= \Phi_{T} c_{T}-\eta \\ 
# \left(\frac{m_{T}+\eta}{1+\Phi_{T}}\right) & = c_{T}
# \end{align}
# or defining $\kappa_{T}=1/(1+\Phi_{T})$ and $\gamma_{T}=\eta/(1+\Phi_{T})$, we have 
# \begin{align}
# \Lambda_{T}^{u} &  = (\kappa_{T}m_{T}+\gamma_{T})\left(1 + \Upsilon \Phi_{T}\right)^{1/(1-\rho)}
# \end{align}
#
# while for the constrained consumer value will be 
# \begin{align}
# v^{c}_{T}(m_{T}) & = u(c_{T}) + \Upsilon u(\eta)
# % \\ (1-\rho)v^{c}_{T}(m) & = m_{T}^{1-\rho} + \Upsilon (\Phi_{T} \eta)^{1-\rho}
#  \\ \left(\left(1-\rho\right)(v^{c}_{T}(m) - \Upsilon u( \eta))\right)^{1/(1-\rho)} & = m_{T} \kappa_{T} \equiv \Lambda^{c}_{T}(m)
# \end{align}

# %% [markdown]
# The value function is therefore piecewise.  Using $\mathbb{1}$ as an indicator of whether the bequest constraint is binding or not, it can be written as the sum of three components:
# \begin{align}
# v_{T}(m) & = (1-\mathbb{1}) v_{T}^{u} + v_{T}^{c}\mathbb{1}
# \\  & = (1-\mathbb{1}) v_{T}^{u} + \left(v_{T}^{c}-\Upsilon (\eta)^{1-\rho}\right)\mathbb{1}+(\Upsilon (\eta)^{1-\rho})\mathbb{1}
# \\  & = (1-\mathbb{1}) u(\Lambda_{T}^{u}) + u(\Lambda_{T}^{c})\mathbb{1}+(\Upsilon (\eta)^{1-\rho})\mathbb{1}
# \end{align}
# where $\Lambda_{T}^{c}$ and $\Lambda_{T}^{u}$ are linear functions and $(\Upsilon (\eta)^{1-\rho})\mathbb{1}$ is a constant.

# %% [markdown]
# We can obtain the $a_{T-2}^{\#,2}$ such that a consumer who was unconstrained between $T-2$ and $T-1$ would arrive in period $T-1$ with $m_{T-1}^{\#,1}$ via the DBC:
# \begin{align}
# m_{T-1}^{\#,1}& = a_{T-2}^{\#,2} (\Rfree/\PermGroFac) + 1
# \\ (\PermGroFac/\Rfree)(m_{T-1}^{\#,1}-1) &= a_{T-2}^{\#,2}
# \end{align}
# and we know that for such an ($T-2$ unconstrained) consumer the growth factor for consumption will be $C_{t+1}/C_{t} = c_{t+1}\PermGroFac/c_{t} = (\Rfree \beta)^{1/\CRRA}$ so the corresponding 
# \begin{align}
# c_{T-2}^{\#,2} & = c_{T-1}^{\#,1} (\PermGroFac/(\Rfree \beta)^{1/\CRRA})
# \end{align}
#
# But if the value of $a_{T-2}^{\#,2}$ obtained from this procedure violates the borrowing constraint, $a_{T-2}^{\#,2} < \underline{a}_{T-2},$ the conclusion must be that the consumer who ended at $m_{T-1}^{\#,1}$ cannot have been unconstrained between $T-2$ and $T-1$.  In this case, the lowest kink point in $T-2$ is the same $m^{\#}$ that obtained in period $T-1$.  
#
# Thus, defining $\vec{\bullet}_{1}$ as the vector of values of $\bullet$ obtained for period $T-1$ above (for example, $\vec{c}_{2}=(c_{T-1}^{\#,1},c_{T-2}^{\#,2})$), we can calculate the locations of the kink points corresponding to horizons at which constraints stop binding iteratively:  Using the $\frown$ operator to append vectors, 
#
# \begin{align}
# \vec{c}_{n+1} & = (c^{\#})^{\frown}\vec{c}_{n}(\PermGroFac/(\Rfree \beta)^{1/\CRRA}) \\
# \vec{a}_{n+1} & = (\underline{a})^{\frown}\left((\vec{m}_{n}-1)(\PermGroFac/\Rfree)\right) \\
# \vec{m}_{n+1} & = (m^{\#})^{\frown}\left(\vec{a}_{n}+\vec{c}_{n}\right)
# \end{align}
#
# Thus for any period $T-n$ the consumption function is defined by the set of line segments connecting the points defined by the corresponding locations in $\vec{m}_{n}$ and $\vec{c}_{n}$ (together with a segment connecting $(0.,0.)$ to $(m^{\#},c^{\#})$, and using the unconstrained perfect foresight consumption function obtaining for points above $(m[n],c[n])$.

# %% [markdown]
# The value function in period $T-1$ can be split into two parts.  Define $n_{T-1}(m)$ as, for any $m$, the number of periods before a constraint (including the `cannot die in debt` constraint for peirod $T$) binds.  That is, for a period $T-1$ consumer with $m < m_{T-1}^{\#,1}$ we will have $n=0$ while for a $T-1$ consumer with $m \geq m_{T-1}^{\#,1}$ we will have $n=1$.
#
# Using this notation, 
#
# \begin{align}
# v_{T-1}(m) & = u(c_{T-1}(m))+ \beta\Gamma_{T-1}^{1-\rho} v_{T} \\ 
# (1-\rho)v_{T-1} & = (c_{T-1}(m))^{1-\rho} + \beta\Gamma_{T-1}^{1-\rho} \left(c_{T}(m_{T})^{1-\rho}+\Upsilon (m_{T}-c_{T})^{1-\rho}\right)
# \end{align}
#
# Dropping the $m_{t}$ arguments to reduce clutter, we can write this as the sum of two components.  For a consumer for whom $n=1$ (the consumer is unconstrained in period $T$, consumption will grow by $\Phi_{T-1}$ between $T-1$ and $T$ (we choose a possibly surprising notational convention to designate the growth factor $\Phi$ that connects $c_{T-1}$ and $c_{T}$ as being associated with period $T-1$.  This is in keeping with our assumption that by the time the consumer is ready to make their decision, the state variable $m_{T}$ must have been determined already by prior events).  

# %% [markdown]
# \begin{align}
# (1-\rho)v_{T-1}^{u} & = c_{T-1}^{1-\rho} + \beta_{T-1}\Gamma_{T-1}^{1-\rho} \left(\Phi_{T-1}c_{T-1})^{1-\rho}+\Upsilon (m_{T}-c_{T})^{1-\rho}\right)
# \\ & = c_{T-1}^{1-\rho}\left(1 + \beta_{T-1}(\Gamma_{T-1}\Phi_{T-1})^{1-\rho}\right)+\Upsilon (m_{T}-\Phi_{T-1}c_{T-1})^{1-\rho}
# \end{align}
#
# while for a consumer for whom $n=2$ (the 'bequest constraint' is also not binding)
# \begin{align}
# (1-\rho)v_{T-2}^{u} & = c_{T-2}^{1-\rho} + \beta_{T-2}(1-\rho)v_{T-1}^{u}
# \end{align}
# while for a consumer for whom $n=2$ (the 'bequest constraint' is also not binding)
# \begin{align}
# (1-\rho)v_{T-1}^{u} & = c_{T-1}^{1-\rho} + \beta_{T-1}\Gamma_{T-1}^{1-\rho} \left(\Phi_{T-1}c_{T-1})^{1-\rho}+\Upsilon (m_{T}-c_{T})^{1-\rho}\right)
# \\ & = c_{T-1}^{1-\rho}\left(1 + \beta_{T-1}(\Gamma_{T-1}\Phi_{T-1})^{1-\rho}+\beta_{T-1}(\Gamma_{T-1}\Phi_{T-1})^{1-\rho}\Upsilon \Phi_{T}^{1-\rho}\right)
# \\ & = c_{T-1}^{1-\rho}\left(1 + \beta_{T-1}(\Gamma_{T-1}\Phi_{T-1})^{1-\rho}(1+\Upsilon \Phi_{T}^{1-\rho})\right)
# \end{align}
# which has the convenient feature that if we define $u^{-1}(v) = \left((1-\rho)v\right)^{1/(1-\rho)}$ we can obtain
# \begin{align}
# (1-\rho)v_{T-1}^{u} & = c_{T-1}^{1-\rho} + \beta_{T-1}\Gamma_{T-1}^{1-\rho} \left(\Phi_{T-1}c_{T-1})^{1-\rho}+\Upsilon (m_{T}-c_{T})^{1-\rho}\right)
# \\ & = c_{T-1}^{1-\rho}\left(1 + \beta_{T-1}(\Gamma_{T-1}\Phi_{T-1})^{1-\rho}+\beta_{T-1}(\Gamma_{T-1}\Phi_{T-1})^{1-\rho}\Upsilon \Phi_{T}^{1-\rho}\right)
# \\ u^{-1}( v_{T-1}^{u} ) & = c_{T-1}\left(1 + \beta_{T-1}(\Gamma_{T-1}\Phi_{T-1})^{1-\rho}(1+\Upsilon \Phi_{T}^{1-\rho})\right)^{1/(1-\rho)}
# \\ & = \kappa_{T-1} (m_{T-1}+h_{T-1})\left(1 + \beta_{T-1}(\Gamma_{T-1}\Phi_{T-1})^{1-\rho}(1+\Upsilon \Phi_{T}^{1-\rho})\right)^{1/(1-\rho)}
# \end{align}

# %% [markdown]
#
# \begin{align}
# v_{t} = & u(c_{t})+ \beta \Gamma^{1-\rho} v_{t+1} \\
# v_{t}/u(c_{t}) = & (1+\beta \Gamma^{1-\rho} (u(c_{t+1})/u(c_{t}) +\beta \Gamma^{1-\rho} v_{t+2}/u(c_{t}) \\
# v_{t}/u(c_{t}) - 1 = & \beta \Gamma^{1-\rho} (1+\beta \Gamma^{1-\rho} v_{t+2}/u(c_{t}))
# \end{align}
#
# Deriving the value function is more complicated, because it needs to be split up into two parts.
#
# \begin{align}
# v_{t} = & u(c_{t})+ \beta \Gamma^{1-\rho} v_{t+1}
# \\ = & \left(1-\rho\right)^{-1}\left(c_{t}^{1-\rho}+\beta \Gamma^{1-\rho} v_{t+1} \right) 
# \\ \left(1-\rho\right) v_{t} = &  c_{t}^{1-\rho}\left(1+...+(\Phi_{\Gamma} \beta \Gamma^{1-\rho})^{n-1}\right)+
# \\ & (\beta \Gamma^{1-\rho})^{n}\left(c_{t+n}^{1-\rho}(\underline{a}_{t+n})+(\beta \Gamma^{1-\rho})c_{t+n+1}^{1-\rho}(\underline{a}_{t+n+1})+...)\right)
# \\ \left(1-\rho\right) v_{t} = &  c_{t}^{1-\rho}\left(1+...+(\Phi_{\Gamma}  \beta \Gamma^{1-\rho})^{n-1}\right)+
# \\ & (\beta \Gamma^{1-\rho})^{n}\left((c_{t+n}(\underline{a}_{t+n})/c_{t})^{1-\rho}+(\beta \Gamma^{1-\rho})(c_{t+n+1}(\underline{a}_{t+n+1})/c_{t})+...)\right)
# \\ (\left(1-\rho\right) v_{t})^{1/(1-\rho)} & = 
# \end{align}

# %% [markdown]
# \begin{align}
# v_{t}(m) & = v_{t}(0)+\kappa_{t} c(m) \\
# v^{-1}_{t}(m) = u^{-1}(v_{t}) & = u^{-1}\left(v_{t}(0)+\kappa_{t} c(m)\right) \\
# \frac{d}{dm} u^{-1}(v_{t}) & = \frac{d}{dv}(u^{-1}(v))\frac{d}{dm}v_{t}(m) \\
# \frac{d}{dm} (v^{-1}_{t}) & = \underbrace{\frac{d}{dv}((1-\rho)v)^{1/(1-\rho)})}_{\equiv ((1-\rho)v)^{-1+1/(1-\rho)}}\frac{d}{dm}v_{t}(m) \\
# (\frac{d}{dm}v^{-1}_{t}(m))\left(v^{-1}_{t}(m)((1-\rho)v)^{-1}\right)^{-1} & = \frac{d}{dm}v_{t}(m) \\ 
# (\frac{d}{dm}v^{-1}_{t}(m))\left(((1-\rho)v)/(v^{-1}_{t}(m)))\right) & = \frac{d}{dm}v_{t}(m) \\ 
# (\frac{d}{dm}v^{-1}_{t}(m))\left((\underbrace{(1-\rho)v}_{\equiv (v^{-1}(m))^{1-\rho}})/(v^{-1}_{t}(m)))\right) & = \frac{d}{dm}v_{t}(m) \\ 
# (\frac{d}{dm}v^{-1}_{t}(m))\left(((1-\rho)v)^{\frac{\rho}{1-\rho}}\right)^{-1} & = \frac{d}{dm}v_{t}(m) \\ 
# (\frac{d}{dm}v^{-1}_{t}(m))\left(((1-\rho)v)^{\frac{1-\rho}{\rho}}\right) & = \frac{d}{dm}v_{t}(m) \\ 
# \frac{d}{dm} u^{-1}(v_{t}) & = \frac{d}{dv}((1-\rho)v)^{1/(1-\rho)})
# \end{align}

# %% [markdown]
# \begin{align}
# v_{t} & = \left(\frac{1}{1-\rho}\right)\left(c_{t}^{1-\rho}\right)(1+\beta\Phi^{1-\rho}+((\beta\Phi)^2)^{1-\rho}+...) \\ 
# v^{\prime}_{t} & = c_{t}^{-\rho}(1+\beta\Phi^{1-\rho}+((\beta\Phi)^2)^{1-\rho}+...) 
# \end{align}

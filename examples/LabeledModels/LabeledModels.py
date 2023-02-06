# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Using xarray to solve Heterogeneous Agent Models
#

# %% [markdown]
# Import required libraries.
#

# %%
from types import SimpleNamespace

import estimagic as em
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from HARK.ConsumptionSaving.ConsIndShockModel import PerfForesightConsumerType
from HARK.rewards import UtilityFuncCRRA
from HARK.utilities import plot_funcs

from xarray import DataArray, Dataset

# %% [markdown]
# Here are some basic parameters that we'll use to construct the model. `CRRA` is the coefficient of constant relative risk aversion, `DiscFac` is the intertemporal discount factor, and `Rfree` is the interest rate on savings. 
#

# %%
epsilon = 1e-6  # lower bound for cash-on-hand
CRRA = 2.0  # coefficient of relative risk aversion
DiscFac = 0.96  # discounting factor
Rfree = 1.03  # risk free interest rate
params = SimpleNamespace(CRRA=CRRA, DiscFac=DiscFac, Rfree=Rfree)


util = UtilityFuncCRRA(CRRA)

# %% [markdown]
# ### The Problem
#

# %% [markdown]
# First, we explore the structure of a perfect-foresight consumption-savings model. The agent's problem is to maximize their present discounted utility of consumption subject to a budget constraint. The recursive problem is given by
#
# $$
# v_t(m_t) = \max_{c_t} u(c_t) + \beta v_{t+1}(m_{t+1}) \\
# s.t. \\
#  a_t = m_t - c_t \\
# m_{t+1} = R  a_t + 1
# $$
#

# %% [markdown]
# This problem can be disected into two stages and two transitions:
#
# First, the agent chooses consumption $c_t$ to maximize their utility given their current cash-on-hand $m_t$ and is left with liquid assets $a_t$. This problem must obey their budget constraint, such that assets is equal to cash-on-hand minus consumption. 
#
# $$
# v_t(m_t) = \max_{c_t} u(c_t) + \beta w_{t}(a_{t}) \\
# s.t. \\
#  a_t = m_t - c_t \\
# $$
#
# Second, the agent receives a constant income and the liquid assets accrue interest, which results in next period's cash-on-hand $m_{t+1}$. 
#
# $$
# w_t(a_t) = v_{t+1}(m_{t+1}) \\
# s.t. \\
#  m_{t+1} = R  a_t + 1
# $$
#
# Although this is very simple, it will be apparent later why this separation is useful.
#

# %% [markdown]
# ### Defining the state space.
#

# %% [markdown]
# We can define the state space two ways: as a numpy grid, or as an xarray.DataArrray.
#

# %%
mVec = np.geomspace(epsilon, 20, 100)  # grid for market resources

# %% [markdown]
# The xr.DataArray will be useful for representing the state space in a more general way. We can define the state space as a 1-dimensional array of cash-on-hand values. For our simple example, we use a 1 variable xr.Dataset to represent the state space.
#

# %%
mNrm = DataArray(
    mVec,
    name="mNrm",
    dims=("mNrm"),
    attrs={"long_name": "Normalized Market Resources"},
)
state = Dataset({"mNrm": mNrm})  # only one state var in this model

# %% [markdown]
# Notice the structure of an xr.Dataset which includes `mNrm` as a dimension.
#

# %%
state

# %% [markdown]
# We can do the same for the liquid assets, which we can refer to as the post-decision state (post-state for short) of the first stage of the problem, or the state of the second stage of the problem.
#

# %%
aNrm = DataArray(
    mVec,
    name="aNrm",
    dims=("aNrm"),
    attrs={"long_name": "Normalized Liquid Assets"},
)
post_state = Dataset({"aNrm": aNrm})

print(post_state)

# %% [markdown]
# We can now define functions over the state space. In this basic model, we need an action/policy/decision to represent consumption. Starting from the last period, we know that the solution is for the agent to consume all of its resources `mNrm`, which induces a linear function. Defining it as a function of the state space is easy, notice in the expression below that the dimension for `cNrm` is `mNrm`. 
#

# %%
# optimal decision is to consume everything in the last period
cNrm = DataArray(
    mVec,
    name="cNrm",
    dims=state.dims,
    coords=state.coords,
    attrs={"long_name": "Consumption"},
)
actions = Dataset({"cNrm": cNrm})
cNrm

# %% [markdown]
# ### The Value and Marginal Value functions.
#

# %% [markdown]
# To define the value and marginal value functions in the last period, we can use the utility and marginal utility functions.
#

# %%
v = util(cNrm)
v.name = "v"
v.attrs = {"long_name": "Value Function"}

v_der = util.der(cNrm)
v_der.name = "v_der"
v_der.attrs = {"long_name": "Marginal Value Function"}

# %% [markdown]
# It will also be useful to define the inverse value and inverse marginal value functions.
#

# %%
v_inv = cNrm.copy()
v_inv.name = "v_inv"
v_inv.attrs = {"long_name": "Inverse Value Function"}

v_der_inv = cNrm.copy()
v_der_inv.name = "v_der_inv"
v_der_inv.attrs = {"long_name": "Inverse Marginal Value Function"}

# %% [markdown]
# We can now create a xr.Dataset to store all of the variables/functions we have created. Datasets are useful containers of variables that are defined over the same dimensions, or in our case states. As we can see, every variable in the dataset shares the same dimension of `mNrm`. 
#

# %%
dataset = Dataset(
    {
        "cNrm": cNrm,
        "v": v,
        "v_der": v_der,
        "v_inv": v_inv,
        "v_der_inv": v_der_inv,
    }
)
dataset

# %% [markdown]
# We can also create separate datasets for the value function variables and the policy function variable.
#

# %%
value_function = Dataset(
    {"v": v, "v_der": v_der, "v_inv": v_inv, "v_der_inv": v_der_inv}
)
policy_function = Dataset({"cNrm": cNrm})

# %% [markdown]
# Up to now I've used the word function for the variables stored as datasets. This is because using the `interp` method we can interpolate the values of the variables at any point in the state space. So, if we have enough points we can approximate the true functions numerically. This is useful for solving the model, as we will see later.
#

# %%
dataset.interp({"mNrm": np.sort(np.random.uniform(epsilon, 20, 10))})


# %% [markdown]
# Because of the curvature of the value and marginal value functions, it'll be useful to use the inverse value and marginal value functions instead and re-curve them. For this, I create a new class `ValueFunctionCRRA` that returns the appropriate value and marginal value functions.
#

# %%
class ValueFunctionCRRA(object):
    def __init__(self, dataset: xr.Dataset, CRRA: float):
        self.dataset = dataset
        self.CRRA = CRRA
        self.u = UtilityFuncCRRA(CRRA)

    def __call__(self, state):
        """
        Interpolate inverse falue function then invert to get value function at given state.
        """

        result = self.u(
            self.dataset["v_inv"].interp(
                state,
                assume_sorted=True,
                kwargs={"fill_value": "extrapolate"},
            )
        )

        result.name = "v"
        result.attrs = self.dataset["v"].attrs

        return result

    def derivative(self, state):
        """
        Interpolate inverse marginal value function then invert to get marginal value function at given state.
        """
        result = self.u.der(
            self.dataset["v_der_inv"].interp(
                state,
                assume_sorted=True,
                kwargs={"fill_value": "extrapolate"},
            )
        )

        result.name = "v_der"
        result.attrs = self.dataset["v"].attrs

        return result

    def evaluate(self, state):
        """
        Interpolate all data variables in the dataset.
        """

        result = self.dataset.interp(state, kwargs={"fill_value": "extrapolate"})
        result.attrs = self.dataset["v"].attrs

        return result


# %% [markdown]
# Now we can create a `ValueFuncCRRA` that will appropriately recurve the value and marginal value functions.

# %%
vfunc = ValueFunctionCRRA(value_function, CRRA)

# %% [markdown]
# For an example of how this is useful, we can create a random grid of states and compare the differences in the 2 approaches.

# %%
rand_states = np.sort(np.random.uniform(mVec[1], mVec[-1], 100))
rand_states

# %% [markdown]
# If we simply linearly interpolate the value and marginal value functions using `xarray` interpolation, we get the following results.

# %%
rand_ds = vfunc.evaluate({"mNrm": rand_states})
rand_ds

# %% [markdown]
# However, if we use the inverse value and marginal value functions to interpolate and then re-curve, the results are slightly different. 

# %%
rand_v = vfunc({"mNrm": rand_states})
np.max(np.abs(rand_v - rand_ds["v"]))

# %% [markdown]
# The correct answer is of course, the re-curving one using `ValueFunctionCRRA`, as evidenced by the following check. As a reminder, the value function at this stage is the utility of consumption, which in the last period is the utility of the cash-on-hand.

# %%
rand_v - util(rand_states)


# %% [markdown]
# ### Transitions
#
# Another useful feature of `xarray` is that we can easily define the state transitions. Using labels, we can define expresive equations that are easy to read and understand.
#

# %%
def state_transition(state=None, action=None, params=None):
    """
    state to post_state transition
    """
    post_state = {}  # pytree
    post_state["aNrm"] = state["mNrm"] - action["cNrm"]
    return post_state


def post_state_transition(post_state=None, params=None):
    """
    post_state to next_state transition
    """
    next_state = {}  # pytree
    next_state["mNrm"] = params.Rfree * post_state["aNrm"] + 1
    return next_state


# %% [markdown]
# This makes it very easy to define simulations of the model given initial states and optimal actions.

# %%
Dataset(state_transition(state, policy_function, params))

# %%
Dataset(post_state_transition(post_state, params))

# %% [markdown]
# These transitions can also be composed. 

# %%
Dataset(post_state_transition(state_transition(state, dataset, params), params))


# %% [markdown]
# We can even define more complex transitions where several variables are created along the way. In the example below, we define the value of an action given some initial state and continuation function, which is the value of having taken that action.
#
# The continuation value function is then the value of some initial post-decision state, which is the value of having taken that action and ending up with next period's state. 

# %%
def value_transition(action=None, state=None, continuation=None, params=None):
    """
    value of action given state and continuation
    """
    variables = {}  # pytree
    post_state = state_transition(state, action, params)
    variables.update(post_state)

    variables["reward"] = util(action["cNrm"])
    variables["v"] = variables["reward"] + params.DiscFac * continuation(post_state)
    variables["v_inv"] = util.inv(variables["v"])

    variables["marginal_reward"] = util.der(action["cNrm"])
    variables["v_der"] = variables["marginal_reward"]  # envelope condition
    variables["v_der_inv"] = util.derinv(variables["v_der"])

    # for estimagic purposes
    variables["contributions"] = variables["v_inv"]
    variables["value"] = np.sum(variables["v_inv"])

    return variables


def continuation_transition(post_state=None, value_next=None, params=None):
    """
    continuation value function of post_states
    """
    variables = {}  # pytree
    next_state = post_state_transition(post_state, params)
    variables.update(next_state)

    variables["v"] = value_next(next_state)
    variables["v_inv"] = util.inv(variables["v"])

    variables["v_der"] = params.Rfree * value_next.derivative(next_state)
    variables["v_der_inv"] = util.derinv(variables["v_der"])

    # for estimagic purposes
    variables["contributions"] = variables["v_inv"]
    variables["value"] = np.sum(variables["v_inv"])

    return variables


# %% [markdown]
# From these transitions, we can easily calculate the continuation value function as follows. 

# %%
v_end = Dataset(continuation_transition(post_state, vfunc, params))
v_end = v_end.drop(["mNrm"])  # next period's mNrm is not needed
v_end

# %%
wfunc = ValueFunctionCRRA(v_end, CRRA)

# %% [markdown]
# For an example, we can calculate the value of taking the same action as in the last period in the second to last period. As a reminder, that action is consuming everything and saving 0.

# %%
Dataset(value_transition(policy_function, state, wfunc, params))

# %% [markdown]
# ## Solving the Model
#
# It should be obvious however, that this is not the optimal action. The optimal action will consist of consuming some of the resources and saving the rest, but how much exactly to save is not straightforward. For this, we can use numerical optimizer `estimagic` to find the optimal action.

# %%
res = em.maximize(
    value_transition,
    params={"cNrm": mVec / 2},
    algorithm="scipy_lbfgsb",
    criterion_kwargs={"state": state, "continuation": wfunc, "params": params},
    lower_bounds={"cNrm": np.zeros_like(mVec)},
    upper_bounds={"cNrm": state["mNrm"].data},
)

c_opt = DataArray(
    res.params["cNrm"],
    name="cNrm",
    dims=state.dims,
    coords=state.coords,
    attrs={"long_name": "consumption"},
)
optimal_actions = Dataset({"cNrm": c_opt})
optimal_value = Dataset(value_transition(optimal_actions, state, wfunc, params))
grid_search = xr.merge([optimal_actions, optimal_value])
grid_search

# %% [markdown]
# As we can see by looking at the `value` variable, the optimization (grid search) method provides a higher value than the naive strategy of consuming everything. We can also easily plot what this maximization looks like.

# %%
grid_search["cNrm"].plot()

# %% [markdown]
# For comparison, we can also check these results against `HARK`'s traditional model solution. 

# %%
hark_agent = PerfForesightConsumerType(
    CRRA=params.CRRA,
    DiscFac=params.DiscFac,
    Rfree=params.Rfree,
    LivPrb=[1.0],
    PermGroFac=[1.0],
    BoroCnstArt=0.0,
)
hark_agent.solve()

np.max(np.abs(hark_agent.solution[0].cFunc(mVec) - grid_search["cNrm"]))


# %% [markdown]
# ## Endogenous Grid Method
#
# As we can see above, the differences are very small. This is because `HARK` uses the endogenous grid method instead of a grid search method to find an optimal solution. To see the endogenous grid method in action, we can instead do the following. 
#
# The endogenous grid method consists of starting from the post-decision state and deriving the optimal action that rationalizes ending up at that state. 
#
# To do this, the endogenous grid method uses the first order condition of the problem, as can be seen in the `egm_transition` function. Having obtained the optimal consumption from a given post-decision state, we can now back out the starting cash-on-hand that would have induced that consumption.
#
#

# %%
def reverse_transition(post_state=None, action=None, params=None):
    states = {}  # pytree
    states["mNrm"] = post_state["aNrm"] + action["cNrm"]

    return states


def egm_transition(post_state=None, continuation=None, params=None):
    """actions from post_states"""

    actions = {}  # pytree
    actions["cNrm"] = util.derinv(params.DiscFac * continuation.derivative(post_state))

    return actions


# %%
acted = egm_transition(post_state, wfunc, params)
states = reverse_transition(post_state, acted, params)

actions = Dataset(acted).swap_dims({"aNrm": "mNrm"})  # egm requires swap dimensions
states = Dataset(states).swap_dims({"aNrm": "mNrm"})

egm_dataset = xr.merge([actions, states])

values = value_transition(actions, states, wfunc, params)
egm_dataset.update(values)

# %% [markdown]
# Because we have imposed an artificial borrowing constraint of 0, we can not optimize our problem at `aNrm` = 0 using the first order condition. Instead, we have to plug in these values. 

# %%
borocnst = Dataset(
    coords={
        "mNrm": 0.0,
        "aNrm": 0.0,
    },
    data_vars={
        "cNrm": 0.0,
        "v": -np.inf,
        "v_inv": 0.0,
        "reward": -np.inf,
        "marginal_reward": np.inf,
        "v_der": np.inf,
        "v_der_inv": 0.0,
    },
)

egm = xr.concat([borocnst, egm_dataset], dim="mNrm", combine_attrs="no_conflicts")
egm

# %% [markdown]
# Now, we can compare the endogenous grid method approach with `HARK`'s solution, and see that the difference is now much smaller and numerically trivial. 

# %%
np.max(np.abs(egm["cNrm"].interp({"mNrm": mVec}) - hark_agent.solution[0].cFunc(mVec)))

# %% [markdown]
# ## `ConsLabeledModels`
#
# The `ConsLabeledModels` module provides a number of models that are defined using the `xarray` framework. Below we show some simple examples of how to use these models.

# %% [markdown]
# ### PerfForesightLabeledType
#
# The `PerfForesightLabeledType` is a perfect foresight model with a constant interest rate and a constant income, so the agent experiences no uncertainty. 

# %%
from HARK.ConsumptionSaving.ConsLabeledModel import (
    PerfForesightLabeledType,
)

agent = PerfForesightLabeledType(cycles=0, BoroCnstArt=-1.0)
agent.solve()

# %%
agent.solution[0].policy["cNrm"].plot()

# %% [markdown]
# The model is equivalent to `PerfForesightConsumerType` presented below. 

# %%
hark_agent = PerfForesightConsumerType(cycles=0, BoroCnstArt=-1.0)
hark_agent.solve()

# %%
plot_funcs(hark_agent.solution[0].cFunc, hark_agent.solution[0].mNrmMin - 1, 25)

# %% [markdown]
# The difference in the two models is small. 

# %%
np.max(
    np.abs(
        hark_agent.solution[0].cFunc(mVec)
        - agent.solution[0].policy["cNrm"].interp({"mNrm": mVec})
    )
)

# %% [markdown]
# ### `IndShockLabeledType`
#
# The `IndShockLabeledType` is a model with idiosyncratic shocks to income. The model is equivalent to `IndShockConsumerType` presented below.

# %%
from HARK.ConsumptionSaving.ConsLabeledModel import IndShockLabeledType

agent = IndShockLabeledType(cycles=0)
agent.solve()

# %%
agent.solution[0].policy["cNrm"].plot()

# %%
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType

hark_agent = IndShockConsumerType(cycles=0, BoroCnstArt=None)
hark_agent.solve()

plot_funcs(hark_agent.solution[0].cFunc, hark_agent.solution[0].mNrmMin - 1, 21)

# %% [markdown]
# The difference in the two models is small.

# %%
mgrid = np.linspace(hark_agent.solution[0].mNrmMin, 20)
np.max(
    np.abs(
        hark_agent.solution[0].cFunc(mgrid)
        - agent.solution[0].policy["cNrm"].interp({"mNrm": mgrid})
    ),
)

# %% [markdown]
# ### RiskyAssetLabeled Type
#
# The `RiskyAssetLabeledType` is a model with idiosyncratic shocks to income and a risky asset. The model is equivalent to `RiskyAssetConsumerType`.

# %%
from HARK.ConsumptionSaving.ConsLabeledModel import RiskyAssetLabeledType

agent = RiskyAssetLabeledType(cycles=0)
agent.solve()

# %%
agent.solution[0].policy["cNrm"].plot()
agent.solution[0].value.dataset["v_inv"].plot()
agent.solution[0].value.dataset["v_der_inv"].plot()

# %% [markdown]
# ### PortfolioLabeledType
#
# The `PortfolioLabeledType` is a model with idiosyncratic shocks to income and a risky asset and a portfolio choice. The model is equivalent to `PortfolioConsumerType`. First we see the consumption function. 

# %%
from HARK.ConsumptionSaving.ConsLabeledModel import PortfolioLabeledType

agent = PortfolioLabeledType(cycles=0)
agent.solve()
agent.solution[0].policy["cNrm"].plot()

# %% [markdown]
# Now we can plot the optimal risky share of portfolio conditional on the initial state of market resources. 

# %%
agent.solution[0].continuation.dataset["stigma"].plot()

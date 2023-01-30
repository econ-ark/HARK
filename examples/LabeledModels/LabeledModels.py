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
# Basic objects that we'll use to construct the model.
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
# First, we explore the structure of a perfect-foresight consumption-savings model. The agent's problem is to maximize their present DiscFaced utility of consumption subject to a budget constraint. The recursive problem is given by
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
# First, the agent chooses consumption $c_t$ to maximize their utility given their current cash-on-hand $m_t$ and is left with liquid assets $a_t$.
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
# The xr.DataArray will be useful for representing the state space in a more general way. We can define the state space as a 1-dimensional array of cash-on-hand values.
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
# Notice the structure of an xr.DataArray.
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
# We can now define functions over the state space. In this basic model, we need action/policy/decision of consumption. Starting from the last period, we know that the solution is for the agent to consume all of its resources. Defining it as a function of the state space is easy.
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
# We can now create a xr.Dataset to store all of the variables/functions we have created. Datasets are useful containers of variables that are defined over the same dimensions, or in our case states.
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
# Up to now I've used the word function for the variables stored as datasets. This is because using the `interp` method we can interpolate the values of the variables at any point in the state space. This is useful for solving the model, as we will see later.
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


# %%
vfunc = ValueFunctionCRRA(value_function, CRRA)

# %%
rand_states = np.sort(np.random.uniform(mVec[1], mVec[-1], 100))
rand_states

# %%
rand_ds = vfunc.evaluate({"mNrm": rand_states})
rand_ds

# %%
rand_v = vfunc({"mNrm": rand_states})
np.max(np.abs(rand_v - rand_ds["v"]))


# %% [markdown]
# ### State transitions
#

# %%
def state_transition(s=None, a=None, params=None):
    """
    state to post_state transition
    """
    ps = {}  # pytree
    ps["aNrm"] = s["mNrm"] - a["cNrm"]
    return ps


def post_state_transition(ps=None, params=None):
    """
    post_state to next_state transition
    """
    ns = {}  # pytree
    ns["mNrm"] = params.Rfree * ps["aNrm"] + 1
    return ns


# %%
ps = Dataset(state_transition(state, policy_function, params))
ps

# %%
ns = Dataset(post_state_transition(ps, params))
ns

# %%
ns = Dataset(post_state_transition(state_transition(state, dataset, params), params))
ns


# %%
def value_transition(a=None, s=None, continuation=None, params=None):
    """
    value of action given state and continuation
    """
    variables = {}  # pytree
    ps = state_transition(s, a, params)
    variables.update(ps)

    variables["reward"] = util(a["cNrm"])
    variables["v"] = variables["reward"] + params.DiscFac * continuation(ps)
    variables["v_inv"] = util.inv(variables["v"])

    variables["marginal_reward"] = util.der(a["cNrm"])
    variables["v_der"] = variables["marginal_reward"]  # envelope condition
    variables["v_der_inv"] = util.derinv(variables["v_der"])

    # for estimagic purposes
    variables["contributions"] = variables["v_inv"]
    variables["value"] = np.sum(variables["v_inv"])

    return variables


def continuation_transition(ps=None, value_next=None, params=None):
    """
    continuation value function of post_states
    """
    variables = {}  # pytree
    ns = post_state_transition(ps, params)
    variables.update(ns)

    variables["v"] = value_next(ns)
    variables["v_inv"] = util.inv(variables["v"])

    variables["v_der"] = params.Rfree * value_next.derivative(ns)
    variables["v_der_inv"] = util.derinv(variables["v_der"])

    # for estimagic purposes
    variables["contributions"] = variables["v_inv"]
    variables["value"] = np.sum(variables["v_inv"])

    return variables


# %%
v_end = Dataset(continuation_transition(post_state, vfunc, params))
v_end = v_end.drop(["mNrm"])

# %%
wfunc = ValueFunctionCRRA(v_end, CRRA)

# %%
Dataset(value_transition(policy_function, state, wfunc, params))

# %%
res = em.maximize(
    value_transition,
    params={"cNrm": mVec / 2},
    algorithm="scipy_lbfgsb",
    criterion_kwargs={"s": state, "continuation": wfunc, "params": params},
    lower_bounds={"cNrm": np.zeros_like(mVec)},
    upper_bounds={"cNrm": state["mNrm"].data},
)

optimal_actions = Dataset(data_vars=res.params, coords={"mNrm": mVec})
optimal_actions

# %%

# %%
c_opt = DataArray(
    res.params["cNrm"],
    name="cNrm",
    dims=state.dims,
    coords=state.coords,
    attrs={"long_name": "consumption"},
)
optimal_actions = Dataset({"cNrm": c_opt})
optimal_actions

# %%
grid_search = Dataset(value_transition(optimal_actions, state, wfunc, params))
grid_search = xr.merge([grid_search, state, optimal_actions])
grid_search

# %%
grid_search["cNrm"].plot()

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

# %%
np.max(np.abs(hark_agent.solution[0].cFunc(mVec) - grid_search["cNrm"]))


# %%
def reverse_transition(ps=None, a=None, params=None):

    states = {}  # pytree
    states["mNrm"] = ps["aNrm"] + a["cNrm"]

    return states


def egm_transition(ps=None, continuation=None, params=None):
    """actions from post_states"""

    actions = {}  # pytree
    actions["cNrm"] = util.derinv(params.DiscFac * continuation.derivative(ps))

    return actions


# %%
acted = egm_transition(post_state, wfunc, params)
states = reverse_transition(post_state, acted, params)

actions = Dataset(acted).swap_dims({"aNrm": "mNrm"})  # egm requires swap dimensions
states = Dataset(states).swap_dims({"aNrm": "mNrm"})

egm_dataset = xr.merge([actions, states])

values = value_transition(actions, states, wfunc, params)
egm_dataset.update(values)

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
borocnst

# %%
egm = xr.concat([borocnst, egm_dataset], dim="mNrm", combine_attrs="no_conflicts")
egm

# %%
np.max(np.abs(egm["cNrm"].interp({"mNrm": mVec}) - hark_agent.solution[0].cFunc(mVec)))

# %%
egm["cNrm"].plot()

# %%
from HARK.ConsumptionSaving.ConsLabeledModel import (
    PerfForesightLabeledType,
)

agent = PerfForesightLabeledType(cycles=0, BoroCnstArt=-1.0)
agent.solve()

# %%
agent.solution[0].policy["cNrm"].plot()

# %%
hark_agent = PerfForesightConsumerType(cycles=0, BoroCnstArt=-1.0)
hark_agent.solve()

# %%
plot_funcs(hark_agent.solution[0].cFunc, hark_agent.solution[0].mNrmMin - 1, 25)

# %%
np.max(
    np.abs(
        hark_agent.solution[0].cFunc(mVec)
        - agent.solution[0].policy["cNrm"].interp({"mNrm": mVec})
    )
)

# %%
agent.completed_cycles

# %%
hark_agent.completed_cycles

# %%
agent.solution[0].value.dataset

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

# %%
plot_funcs(hark_agent.solution[0].cFunc, hark_agent.solution[0].mNrmMin - 1, 21)

# %%
mgrid = np.linspace(hark_agent.solution[0].mNrmMin, 20)
np.max(
    np.abs(
        hark_agent.solution[0].cFunc(mgrid)
        - agent.solution[0].policy["cNrm"].interp({"mNrm": mgrid})
    ),
)

# %%
from HARK.ConsumptionSaving.ConsLabeledModel import RiskyAssetLabeledType

agent = RiskyAssetLabeledType(cycles=0)
agent.solve()

# %%
agent.solution[0].policy["cNrm"].plot()
agent.solution[0].value.dataset["v_inv"].plot()
agent.solution[0].value.dataset["v_der_inv"].plot()

# %%
from HARK.ConsumptionSaving.ConsLabeledModel import PortfolioLabeledType

agent = PortfolioLabeledType(cycles=0)
agent.solve()

# %%
agent.solution[0].policy["cNrm"].plot()

# %%
agent.solution[0].continuation.dataset["stigma"].plot()

# %%

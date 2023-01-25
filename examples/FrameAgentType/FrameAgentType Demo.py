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

# %%
import HARK.ConsumptionSaving.ConsPortfolioFrameModel as cpfm
import HARK.ConsumptionSaving.ConsPortfolioModel as cpm

from HARK.frame import Frame, draw_frame_model
import numpy as np

from HARK.rewards import (
    CRRAutility,
)

# %% [markdown]
# The `FrameAgentType` is an alternative way to specify a model.
#
# The library contains a demonstration of this form of model, `ConsPortfolioFrameModel`, which is a replica of the `ConsPortfolioModel`.
#
# This notebook compares the results of simulations of the two models.

# %%
pct = cpm.PortfolioConsumerType(T_sim=5000, AgentCount=200)
pct.cycles = 0

# Solve the model under the given parameters

pct.solve()
pct.track_vars += [
    "mNrm",
    "cNrm",
    "Share",
    "aNrm",
    "Risky",
    "Adjust",
    "PermShk",
    "TranShk",
    "bNrm",
    "who_dies",
]

pct.make_shock_history()
pct.read_shocks = True

pct.initialize_sim()

pct.simulate()

# %%
pcft = cpfm.PortfolioConsumerFrameType(T_sim=5000, AgentCount=200, read_shocks=True)

pcft.cycles = 0

# Solve the model under the given parameters
pcft.solve()

pcft.track_vars += [
    "mNrm",
    "cNrm",
    "Share",
    "aNrm",
    "Adjust",
    "PermShk",
    "TranShk",
    "bNrm",
    "U",
]

pcft.shock_history = pct.shock_history
pcft.newborn_init_history = pct.newborn_init_history

pcft.initialize_sim()

pcft.simulate()

# %%
import matplotlib.pyplot as plt

plt.plot(range(5000), pct.history["PermShk"].mean(axis=1), label="original")
plt.plot(range(5000), pcft.history["PermShk"].mean(axis=1), label="frames", alpha=0.5)
plt.legend()

# %%
plt.plot(range(5000), pct.history["TranShk"].mean(axis=1), label="original")
plt.plot(range(5000), pcft.history["TranShk"].mean(axis=1), label="frames", alpha=0.5)
plt.legend()

# %%
plt.plot(range(5000), pct.history["bNrm"].mean(axis=1), label="original")
plt.plot(range(5000), pcft.history["bNrm"].mean(axis=1), label="frames", alpha=0.5)
plt.legend()

# %%
# plt.plot(range(5000), pct.history['Risky'].mean(axis=1), label = 'original')
# plt.plot(range(5000), pcft.history['Risky'].mean(axis=1), label = 'frames', alpha = 0.5)
# plt.legend()

# %%
plt.plot(range(5000), pct.history["aNrm"].mean(axis=1), label="original")
plt.plot(range(5000), pcft.history["aNrm"].mean(axis=1), label="frames", alpha=0.5)
plt.legend()

# %%
plt.plot(range(5000), pct.history["mNrm"].mean(axis=1), label="original")
plt.plot(range(5000), pcft.history["mNrm"].mean(axis=1), label="frames", alpha=0.5)
plt.legend()

# %%
plt.plot(range(5000), pct.history["cNrm"].mean(axis=1), label="original")
plt.plot(range(5000), pcft.history["cNrm"].mean(axis=1), label="frames", alpha=0.5)
plt.legend()

# %% [markdown]
# **TODO**: Handly Risky as an aggregate value.

# %%
# pct.history['Risky'][:3, :3]

# %%
# pcft.history['Risky'][:3, :3]

# %%
plt.plot(range(5000), pct.history["Share"].mean(axis=1), label="original")
plt.plot(range(5000), pcft.history["Share"].mean(axis=1), label="frames", alpha=0.5)
plt.legend()

# %%
plt.plot(
    range(5000), pcft.history["cNrm"].mean(axis=1), label="frames - cNrm", alpha=0.5
)
plt.plot(range(5000), pcft.history["U"].mean(axis=1), label="frames - U", alpha=0.5)
plt.legend()

# %%
pcft.history["U"]

# %%
pcft.history["U"].mean(axis=1)

# %%
pcft.history["U"][0, :]

# %%
pcft.history["cNrm"][0, :]

# %%
pcft.parameters["CRRA"]

# %%
CRRAutility(pcft.history["cNrm"][0, :], 5)

# %% [markdown]
# # Visualizing the Transition Equations

# %% [markdown]
# Note that in the HARK `ConsIndShockModel`, from which the `ConsPortfolio` model inherits, the aggregate permanent shocks are considered to be portions of the permanent shocks experienced by the agents, not additions to those idiosyncratic shocks. Hence, they do not show up directly in the problem solved by the agent. This explains why the aggregate income levels are in a separarte component of the graph.

# %%
draw_frame_model(pcft.model, figsize=(14, 12))

# %% [markdown]
# # Building the Solver [INCOMPLETE]

# %% [markdown]
# Preliminery work towards a generic solver for FramedAgentTypes.

# %%
controls = [frame for frame in pcft.frames.values() if frame.control]


# %%
def get_expected_return_function(control: Frame):
    # Input: a control frame
    # Returns: function of the control variable (control frame target)
    #      that returns the expected return, which is
    #          the sum of:
    #              - direct rewards
    #              - expected value of next-frame states (not yet implemented)
    #

    rewards = [child for child in control.children if child.reward]
    expected_values = []  # TODO

    ## note: function signature is what's needed for scipy.optimize
    def expected_return_function(x, *args):
        ##   returns the sum of
        ##     the reward functions evaluated in context of
        ##       - parameters
        ##       - the control variable input

        # x - array of inputs, here the control frame target
        # args - a tuple of other parameters needed to complete the function

        expected_return = 0

        for reward in rewards:
            ## TODO: figuring out the ordering of `x` and `args` needed for multiple downstream scopes

            local_context = {}

            # indexing through the x and args values
            i = 0
            num_control_vars = None

            # assumes that all frame scopes list model variables first, parameters later
            # should enforce or clarify at the frame level.
            for var in reward.scope:
                if var in control.target:
                    local_context[var] = x[i]
                    i = i + 1
                elif var in pcft.parameters:
                    if num_control_vars is None:
                        num_control_vars = i

                    local_context[var] = args[i - num_control_vars]
                    i = i + 1

            # can `self` be implicit here?
            expected_return += reward.transition(reward, **local_context)

        return expected_return

    return expected_return_function


# %%
def optimal_policy_function(control: Frame):

    erf = get_expected_return_function(control)
    constraints = (
        control.constraints
    )  ## these will reference the context of the control transition, including scope

    ## Returns function:
    ##   input: control frame scope
    ##   output: result of scipy.optimize of the erf with respect to constraints
    ##           getting the optimal input (control variable) value
    return func


# %%
def approximate_optimal_policy_function(control, grid):
    ## returns a new function:
    ##   that is an interpolation over optimal_policy_function
    ##   over the grid

    return func

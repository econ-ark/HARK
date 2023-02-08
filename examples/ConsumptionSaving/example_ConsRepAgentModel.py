# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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
from copy import deepcopy
from time import time
import numpy as np
from HARK.utilities import plot_funcs
from HARK.ConsumptionSaving.ConsIndShockModel import init_idiosyncratic_shocks

from HARK.ConsumptionSaving.ConsRepAgentModel import (
    RepAgentConsumerType,
    RepAgentMarkovConsumerType,
)

# %% [markdown]
# This module contains models for solving representative agent (RA) macroeconomic models. This stands in contrast to all other model modules in HARK, which (unsurprisingly) take a heterogeneous agents approach.
# In RA models, all attributes are either time invariant or exist on a short cycle. Also, models must be infinite horizon.

# %% [markdown]
# Each period, the representative agent makes a decision about how much of his resources $m_t$ he should consume $c_t$ and how much should retain as assets $a_t$. He gets a flow of utility from consumption, with CRRA preferences (with coefficient $\rho$). Retained assets are used to finance productive capital $k_{t+1}$ in the next period. Output is produced according to a Cobb-Douglas production function using capital and labor $\ell_{t+1}$, with a capital share of $\alpha$; a fraction $\delta$ of capital depreciates immediately after production.
#
# The agent's labor productivity is subject to permanent and transitory shocks, $\psi_t$ and $\theta_t$ respectively. The representative agent stands in for a continuum of identical households, so markets are assumed competitive: the factor returns to capital and income are the (net) marginal product of these inputs.
#
# In the notation below, all lowercase state and control variables ($m_t$, $c_t$, etc) are normalized by the permanent labor productivity of the agent. The level of these variables at any time $t$ can be recovered by multiplying by permanent labor productivity $p_t$ (itself usually normalized to 1 at model start).

# %% [markdown]
# The agent's problem can be written in Bellman form as:
#
# \begin{eqnarray*}
# v_t(m_t) &=& \max_{c_t} U(c_t) + \beta \mathbb{E} [(\Gamma_{t+1}\psi_{t+1})^{1-\rho} v_{t+1}(m_{t+1})], \\
# a_t &=& m_t - c_t, \\
# \psi_{t+1} &\sim& F_{\psi t+1}, \qquad  \mathbb{E} [F_{\psi t}] = 1,\\
# \theta_{t+1} &\sim& F_{\theta t+1}, \\
# k_{t+1} &=& a_t/(\Gamma_{t+1}\psi_{t+1}), \\
# R_{t+1} &=& 1 - \delta + \alpha (k_{t+1}/\theta_{t+1})^{(\alpha - 1)}, \\
# w_{t+1} &=& (1-\alpha) (k_{t+1}/\theta_{t+1})^\alpha, \\
# m_{t+1} &=& R_{t+1} k_{t+1} + w_{t+1}\theta_{t+1}, \\
# U(c) &=& \frac{c^{1-\rho}}{1-\rho}
# \end{eqnarray*}

# %% [markdown]
# The one period problem for this model is solved by the function $\texttt{solveConsRepAgent}$.

# %%
# Make a quick example dictionary
RA_params = deepcopy(init_idiosyncratic_shocks)
RA_params["DeprFac"] = 0.05
RA_params["CapShare"] = 0.36
RA_params["UnempPrb"] = 0.0
RA_params["LivPrb"] = [1.0]

# %%
# Make and solve a rep agent model
RAexample = RepAgentConsumerType(**RA_params)
t_start = time()
RAexample.solve()
t_end = time()
print(
    "Solving a representative agent problem took " + str(t_end - t_start) + " seconds."
)
plot_funcs(RAexample.solution[0].cFunc, 0, 20)

# %%
# Simulate the representative agent model
RAexample.T_sim = 2000
RAexample.track_vars = ["cNrm", "mNrm", "Rfree", "wRte"]
RAexample.initialize_sim()
t_start = time()
RAexample.simulate()
t_end = time()
print(
    "Simulating a representative agent for "
    + str(RAexample.T_sim)
    + " periods took "
    + str(t_end - t_start)
    + " seconds."
)

# %%
# Make and solve a Markov representative agent
RA_markov_params = deepcopy(RA_params)
RA_markov_params["PermGroFac"] = [[0.97, 1.03]]
RA_markov_params["MrkvArray"] = np.array([[0.99, 0.01], [0.01, 0.99]])
RA_markov_params["Mrkv"] = 0
RAmarkovExample = RepAgentMarkovConsumerType(**RA_markov_params)
RAmarkovExample.IncShkDstn = [2 * [RAmarkovExample.IncShkDstn[0]]]
t_start = time()
RAmarkovExample.solve()
t_end = time()
print(
    "Solving a two state representative agent problem took "
    + str(t_end - t_start)
    + " seconds."
)
plot_funcs(RAmarkovExample.solution[0].cFunc, 0, 10)

# %%
# Simulate the two state representative agent model
RAmarkovExample.T_sim = 2000
RAmarkovExample.track_vars = ["cNrm", "mNrm", "Rfree", "wRte", "Mrkv"]
RAmarkovExample.initialize_sim()
t_start = time()
RAmarkovExample.simulate()
t_end = time()
print(
    "Simulating a two state representative agent for "
    + str(RAexample.T_sim)
    + " periods took "
    + str(t_end - t_start)
    + " seconds."
)

# %%

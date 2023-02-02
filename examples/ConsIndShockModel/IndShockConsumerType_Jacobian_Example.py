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
# # Computing Heterogenous Agent Jacobians in HARK
#
# By William Du
#
# This notebook illustrates how to compute Heterogenous Agent Jacobian matrices in HARK.
#
# These matrices are a fundamental building building block to solving Heterogenous Agent New Keynesian Models with the sequence space jacobian methodology. For more information, see [Auclert, Rognlie, Bardoszy, and Straub (2021)](https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434)
#
# For the IndShockConsumerType, Jacobians of Consumption and Saving can be computed with respect to the following parameters:
# LivPrb, PermShkStd,TranShkStd, DiscFac,UnempPrb, Rfree, IncUnemp.

# %%
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType


import time
import numpy as np
import matplotlib.pyplot as plt


# %% [markdown]
# ## Create Agent

# %%
# Dictionary for Agent

Dict = {
    # Solving Parameters
    "aXtraMax": 1000,
    "aXtraCount": 200,
    # Transition Matrix Simulations Parameters
    "mMax": 10000,
    "mMin": 1e-6,
    "mCount": 300,
    "mFac": 3,
}

# %%

Agent = IndShockConsumerType(**Dict)


# %% [markdown]
# ## Compute Steady State

# %%

start = time.time()
Agent.compute_steady_state()
print("Seconds to compute steady state", time.time() - start)


# %% [markdown]
# ## Compute Jacobians
#
# Shocks possible: LivPrb, PermShkStd,TranShkStd, DiscFac,UnempPrb, Rfree, IncUnemp, DiscFac

# %% [markdown]
# ### Shock to Standard Deviation to Permanent Income Shocks

# %%

start = time.time()

CJAC_Perm, AJAC_Perm = Agent.calc_jacobian("PermShkStd", 300)

print("Seconds to calculate Jacobian", time.time() - start)


# %% [markdown]
# #### Consumption Jacobians

# %%

plt.plot(CJAC_Perm.T[0])
plt.plot(CJAC_Perm.T[10])
plt.plot(CJAC_Perm.T[30])
plt.show()


# %% [markdown]
# #### Asset Jacobians

# %%

plt.plot(AJAC_Perm.T[0])
plt.plot(AJAC_Perm.T[10])
plt.plot(AJAC_Perm.T[30])
plt.plot(AJAC_Perm.T[60])
plt.show()


# %% [markdown]
# ## Shock to Real Interest Rate

# %%
CJAC_Rfree, AJAC_Rfree = Agent.calc_jacobian("Rfree", 300)


# %% [markdown]
# #### Consumption Jacobians

# %%

plt.plot(CJAC_Rfree.T[0])
plt.plot(CJAC_Rfree.T[10])
plt.plot(CJAC_Rfree.T[30])
plt.plot(CJAC_Rfree.T[60])
plt.show()


# %% [markdown]
# #### Asset Jacobians

# %%

plt.plot(AJAC_Rfree.T[0])
plt.plot(AJAC_Rfree.T[10])
plt.plot(AJAC_Rfree.T[30])
plt.plot(AJAC_Rfree.T[60])
plt.show()

# %% [markdown]
# ## Shock to Unemployment Probability

# %%
CJAC_UnempPrb, AJAC_UnempPrb = Agent.calc_jacobian("UnempPrb", 300)


# %%
plt.plot(CJAC_UnempPrb.T[0])
plt.plot(CJAC_UnempPrb.T[10])
plt.plot(CJAC_UnempPrb.T[30])
plt.plot(CJAC_UnempPrb.T[60])
plt.show()

# %%
plt.plot(AJAC_UnempPrb.T[0])
plt.plot(AJAC_UnempPrb.T[10])
plt.plot(AJAC_UnempPrb.T[30])
plt.plot(AJAC_UnempPrb.T[60])
plt.show()

# %% [markdown]
# ## Shock to Discount Factor

# %%
CJAC_DiscFac, AJAC_DiscFac = Agent.calc_jacobian("DiscFac", 300)


# %%
plt.plot(CJAC_DiscFac.T[0])
plt.plot(CJAC_DiscFac.T[10])
plt.plot(CJAC_DiscFac.T[30])
plt.plot(CJAC_DiscFac.T[60])
plt.show()

# %%
plt.plot(AJAC_DiscFac.T[0])
plt.plot(AJAC_DiscFac.T[10])
plt.plot(AJAC_DiscFac.T[30])
plt.plot(AJAC_DiscFac.T[60])
plt.show()

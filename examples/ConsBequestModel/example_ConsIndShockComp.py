# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: hark-dev
#     language: python
#     name: python3
# ---

# %%
from HARK.ConsumptionSaving.ConsBequestModel import BequestWarmGlowConsumerType
from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    init_idiosyncratic_shocks,
)
from HARK.utilities import plot_funcs

# %%
beq_agent = BequestWarmGlowConsumerType(
    **init_idiosyncratic_shocks, TermBeqFac=0.0, BeqFac=0.0
)
beq_agent.cycles = 0
beq_agent.solve()

# %%
ind_agent = IndShockConsumerType(**init_idiosyncratic_shocks)
ind_agent.cycles = 0
ind_agent.solve()

# %%
plot_funcs([beq_agent.solution[0].cFunc, ind_agent.solution[0].cFunc], 0, 10)

# %%

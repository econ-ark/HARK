# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from HARK.ConsumptionSaving.ConsIndShockModel import (
IndShockConsumerType,
init_lifecycle
)

from HARK.Calibration.Calibration import (
    ParseIncomeSpec,
    ParseTimeParams,
    parse_ssa_life_table,
    CGM_income,
    Cagetti_income
)

import matplotlib.pyplot as plt
import pandas as pd
from copy import copy

# %% Alter calibration
birth_age = 21
death_age = 90

income_params = ParseIncomeSpec(age_min = birth_age, age_max = death_age,
                                **CGM_income['NoHS'])

liv_prb = parse_ssa_life_table(filename = 'LifeTables/SSA_LifeTable2017.csv',
                               sep = ',', sex = 'male',
                               min_age = birth_age, max_age = death_age)

time_params = ParseTimeParams(age_birth = birth_age, age_death = death_age)

params = copy(init_lifecycle)
params.update(time_params)
params.update(income_params)
params.update({'LivPrb': liv_prb, 'aNrmInitMean': -100})

# %% Create and solve agent
Agent = IndShockConsumerType(**params)
Agent.solve()

# %% Simulation

# Setup

# Number of agents and periods in the simulation.
Agent.AgentCount = 50
Agent.T_sim      = 200

# Set up the variables we want to keep track of.
Agent.track_vars = ['aNrmNow','cNrmNow', 'pLvlNow', 't_age','mNrmNow']

# Run the simulations
Agent.initializeSim()
Agent.simulate()

# %% Extract and format simulation results

raw_data = {'Age': Agent.history['t_age'].flatten()+birth_age - 1,
            'pIncome': Agent.history['pLvlNow'].flatten(),
            'nrmM': Agent.history['mNrmNow'].flatten(),
            'nrmC': Agent.history['cNrmNow'].flatten()}

Data = pd.DataFrame(raw_data)
Data['Cons'] = Data.nrmC * Data.pIncome
Data['M'] = Data.nrmM * Data.pIncome

# %% Plots

# Find the mean of each variable at every age
AgeMeans = Data.groupby(['Age']).mean().reset_index()

plt.figure()
plt.plot(AgeMeans.Age, AgeMeans.pIncome,
         label = 'Permanent Income')
plt.plot(AgeMeans.Age, AgeMeans.M,
         label = 'Market resources')
plt.plot(AgeMeans.Age, AgeMeans.Cons,
         label = 'Consumption')
plt.legend()
plt.xlabel('Age')
plt.ylabel('Thousands of 1992 USD')
plt.title('Variable Means Conditional on Survival')
plt.grid()

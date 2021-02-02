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
    CGM_income,
    Cagetti_income
)

from HARK.datasets.life_tables.us_ssa.SSATools import parse_ssa_life_table
from HARK.datasets.SCF.WealthIncomeDist.SCFDistTools import income_wealth_dists_from_scf
import matplotlib.pyplot as plt
import pandas as pd
from copy import copy

# %% Alter calibration
birth_age = 25
death_age = 90
education = 'College'

# Income specification
income_params = ParseIncomeSpec(age_min = birth_age, age_max = death_age,
                                **CGM_income[education], SabelhausSong=True)

# Initial distribution of wealth and permanent income
dist_params = income_wealth_dists_from_scf(base_year=1992, age = birth_age,
                                           education = education, wave = 1995)

# We need survival probabilities only up to death_age-1, because survival
# probability at death_age is 1.
liv_prb = parse_ssa_life_table(female = True, cross_sec = True, year = 2004,
                               min_age = birth_age, max_age = death_age - 1)

time_params = ParseTimeParams(age_birth = birth_age, age_death = death_age)

params = copy(init_lifecycle)
params.update(time_params)
params.update(income_params)
params.update(dist_params)
params.update({'LivPrb': liv_prb})

# %% Create and solve agent
Agent = IndShockConsumerType(**params)
Agent.solve()

# %% Simulation

# Setup

# Number of agents and periods in the simulation.
Agent.AgentCount = 500
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

# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 10:50:02 2021

@author: Mateo
"""

from HARK.Calibration.Calibration import (
    ParseIncomeSpec,
    findProfile,
    parse_ssa_life_table,
    Cagetti_income,
    CGM_income
)
import numpy as np
import matplotlib.pyplot as plt

# %% CGM calibration

age_min = 21
age_max = 100

ages = np.arange(age_min, age_max + 1)

plt.figure()
for spec in CGM_income.items():
    
    label = spec[0]
    
    params = ParseIncomeSpec(age_min = age_min, age_max = age_max, **spec[1])
    MeanY = findProfile(params['PermGroFac'], params['P0'])
    
    plt.plot(ages, MeanY, label = label)

plt.title('CGM')
plt.legend()
plt.show()

# %% Cagetti calibration

age_min = 25
age_max = 91

ages = np.arange(age_min, age_max + 1)

plt.figure()
for spec in Cagetti_income.items():
    
    label = spec[0]
    
    params = ParseIncomeSpec(age_min = age_min, age_max = age_max, **spec[1])
    MeanY = findProfile(params['PermGroFac'], params['P0'])
    
    plt.plot(ages, MeanY, label = label)

plt.title('Cagetti')
plt.legend()
plt.show()

# %% Life probabilities

min_age = 21
max_age = 100
ages = np.arange(min_age, max_age)

plt.figure()
for s in ['male', 'female']:
    
    LivPrb = parse_ssa_life_table(filename = 'LifeTables/SSA_LifeTable2017.csv',
                                  sep = ',', sex = s,
                                  min_age = min_age, max_age = max_age)
    
    plt.plot(ages, LivPrb, label = s)
    
plt.legend()
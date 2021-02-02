# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 10:50:02 2021

@author: Mateo
"""

from HARK.Calibration.Calibration import (
    ParseIncomeSpec,
    findProfile,
    Cagetti_income,
    CGM_income,
)

import numpy as np
import matplotlib.pyplot as plt

# %% CGM calibration
adjust_infl_to = 1992

age_min = 21
age_max = 100

ages = np.arange(age_min, age_max + 1)

plt.figure()
for spec in CGM_income.items():

    label = spec[0]

    params = ParseIncomeSpec(
        age_min=age_min, age_max=age_max, adjust_infl_to=adjust_infl_to, **spec[1]
    )
    MeanY = findProfile(params["PermGroFac"], params["P0"])

    plt.plot(ages, MeanY, label=label)

plt.title("CGM")
plt.legend()
plt.show()

# %% Cagetti calibration

age_min = 25
age_max = 91
# Cagetti has a year trend in his specification, so we have to say on what
# year agents enter the model.
start_year = 1980

ages = np.arange(age_min, age_max + 1)

plt.figure()
for spec in Cagetti_income.items():

    label = spec[0]

    params = ParseIncomeSpec(
        age_min=age_min,
        age_max=age_max,
        adjust_infl_to=adjust_infl_to,
        start_year=start_year,
        **spec[1]
    )
    MeanY = findProfile(params["PermGroFac"], params["P0"])

    plt.plot(ages, MeanY, label=label)

plt.title("Cagetti")
plt.legend()
plt.show()

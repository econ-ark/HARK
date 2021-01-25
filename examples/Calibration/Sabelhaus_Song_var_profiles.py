# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 16:44:09 2021

@author: Mateo

This short script demonstrates how to use the module for computing
[1] Sabelhaus & Song (2010) age profiles of income volatility.
It does so by replicating the results from the original paper (Figure 6 in [1])

[1] Sabelhaus, J., & Song, J. (2010). The great moderation in micro labor
    earnings. Journal of Monetary Economics, 57(4), 391-403.

"""

import matplotlib.pyplot as plt
from HARK.Calibration.Income.SabelhausSongProfiles import sabelhaus_song_var_profile

# Set up ages and cohorts at which we will get the variances
age_min = 27
age_max = 54
cohorts = [1940, 1965, None]

# Find volatility profiles using the module
variances = [
    sabelhaus_song_var_profile(age_min=age_min, age_max=age_max, cohort=c)
    for c in cohorts
]

# %% Plots

# Plot transitory shock variances
plt.figure()
for i in range(len(cohorts)):

    coh_label = "aggregate" if cohorts[i] is None else cohorts[i]
    plt.plot(
        variances[i]["Age"],
        variances[i]["TranShkStd"],
        label="Tran. {} cohort".format(coh_label),
    )

plt.legend()

# Plot permanent shock variances
plt.figure()
for i in range(len(cohorts)):

    coh_label = "aggregate" if cohorts[i] is None else cohorts[i]
    plt.plot(
        variances[i]["Age"],
        variances[i]["PermShkStd"],
        label="Perm. {} cohort".format(coh_label),
    )

plt.legend()

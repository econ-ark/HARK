# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 16:44:09 2021

@author: Mateo
"""

import numpy as np
import matplotlib.pyplot as plt
from HARK.Calibration.Income.SabelhausSongProfiles import sabelhaus_song_var_profile

age_min = 27
age_max = 54
ages = np.arange(age_min, age_max + 1)
years = [1940, 1965, None]

variances = [sabelhaus_song_var_profile(age_min = age_min,
                                        age_max = age_max, cohort = y)
             for y in years]

# Plot transitory variances
plt.figure()
for i in range(len(years)):
    plt.plot(ages, variances[i]['TranShkStd'], label = 'Tran. {} cohort'.format(years[i]))
plt.legend()

# Plot permanent variances
plt.figure()
for i in range(len(years)):
    plt.plot(ages, variances[i]['PermShkStd'], label = 'Perm. {} cohort'.format(years[i]))
plt.legend()
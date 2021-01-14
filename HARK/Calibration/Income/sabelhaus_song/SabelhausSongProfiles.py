# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 15:49:43 2021

@author: Mateo
"""

import numpy as np
import pandas as pd
from warnings import warn
import os
from HARK.interpolation import LinearInterp

sabelhaus_song_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'variance_est.csv')

def sabelhaus_song_var_profile(cohort, age_min = 27, age_max = 54):
    
    assert age_max >= age_min, "The maximum age can not be lower than the minimum age."
    
    # Read coefficients
    table = pd.read_csv(sabelhaus_song_file, sep=",")
    
    # Extract coefficients
    beta_eps = table.TrnsCoef[0]
    beta_eta = table.PermCoef[0]
    tran_age_dummies = table.TrnsCoef[1:-1].to_numpy()
    perm_age_dummies = table.PermCoef[3:].to_numpy()
    
    # Make interpolators for transitory and permanent dummies. Alter to use
    # flat extrapolation.
    
    # S&S provide transitory dummies for ages 25-54, so give the interpolator
    # points for 24-55 with flat extremes
    tran_dummy_interp = LinearInterp(np.arange(24, 56),
                                     np.concatenate([[tran_age_dummies[0]],
                                                     tran_age_dummies,
                                                     [tran_age_dummies[-1]]]),
                                     lower_extrap = True)
    
    # S&S provide permanent dummies for ages 27-55, so give the interpolator
    # points for 26-56 with flat extremes
    perm_dummy_interp = LinearInterp(np.arange(26, 57),
                                     np.concatenate([[perm_age_dummies[0]],
                                                     perm_age_dummies,
                                                     [perm_age_dummies[-1]]]),
                                     lower_extrap = True)
    
    if age_min < 27 or age_max > 54:
        warn('Sabelhaus and Song (2010) provide variance profiles for ages '+
             '27 to 54. Extrapolating variances.')
    
    # Construct variances
    # They use 1926 as the base year for cohort effects.
    ages = np.arange(age_min, age_max + 1)
    tran_std = tran_dummy_interp(ages) + (cohort-1926) * beta_eps
    perm_std = perm_dummy_interp(ages) + (cohort-1926) * beta_eta
    
    variances = {'TranShkStd': list(tran_std),
                 'PermShkStd': list(perm_std)}
    
    return(variances)
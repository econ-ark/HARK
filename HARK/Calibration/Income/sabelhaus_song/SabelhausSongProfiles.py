# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 15:49:43 2021

@author: Mateo
"""

import numpy as np
import pandas as pd
from warnings import warn
import os
import matplotlib.pyplot as plt

sabelhaus_song_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'variance_est.csv')

def sabelhaus_song_var_profile(cohort, age_min = 27, age_max = 54):
    
    # Read coefficients
    table = pd.read_csv(sabelhaus_song_file, sep=",")
    
    # Extract coefficients
    beta_eps = table.TrnsCoef[0]
    beta_eta = table.PermCoef[0]
    tran_age_dummies = table.TrnsCoef[1:].to_numpy()
    perm_age_dummies = table.PermCoef[1:].to_numpy()
    ages             = np.arange(25, 56)
    
    positions = np.arange(age_min, age_max + 1) - 25
    variances = {'TranShkStd': tran_age_dummies[positions] + (cohort-1926) * beta_eps,
                 'PermShkStd': perm_age_dummies[positions] + (cohort-1926) * beta_eta}
    
    return(variances)
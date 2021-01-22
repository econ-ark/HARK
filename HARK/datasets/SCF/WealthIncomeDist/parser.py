# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 15:36:14 2021

@author: Mateo
"""

import numpy as np
import pandas as pd
from warnings import warn
from HARK.datasets.cpi.us.CPITools import cpi_deflator
import os

scf_sumstats_dir = os.path.dirname(os.path.abspath(__file__))


def get_scf_distr_stats():
    """
    
    """

    filename = os.path.join(scf_sumstats_dir, "WealthIncomeStats.csv")

    # Read csv
    table = pd.read_csv(filename, sep=",")

    return table


def parse_scf_distr_stats(
    age = None, education = None, wave = None
):
    
    # Pre-process year to make it a five-year bracket as in the table
    if age is not None:
        
        u_bound = int(np.ceil(age/5) * 5)
        l_bound = u_bound - 5
        age_bracket = '(' + str(l_bound) + ',' + str(u_bound) + ']'
    
    else:
        
        # If no age is given, use all age brackets.
        age_bracket = 'All'
    
    # Check whether education is in one of the allowed categories
    if education is not None:
        
        message = ("If an education level is provided, it must be one of " +
                   "'NoHS', 'HS', or 'College'.")
        assert education in ['NoHS','HS','College'], message
        
    else:
        
        education = 'All'
    
    # Parse the wave
    wave_str = 'All' if wave is None else str(int(wave))
    
    # Read table
    filename = os.path.join(scf_sumstats_dir, "WealthIncomeStats.csv")

    # Read csv
    table = pd.read_csv(filename, sep=",",
                        index_col = ['Educ','YEAR','Age_grp'],
                        dtype = {'Educ': str,'YEAR': str,'Age_grp': str,
                                 'BASE_YR': int})
    
    # Try to access the requested combination
    try:
        
        row = table.loc[(education, wave_str, age_bracket)]
        
    except KeyError as e:
        
        message = ("The summary statistics do not contain the "+
                   "Age/Wave/Education combination that was requested.")
        raise Exception(message).with_traceback(e.__traceback__)
    
    # Check for NAs
    if any(row.isna()):
        warn("There were not enough observations in the requested " + 
             "Age/Wave/Education combination to compute all summary" +
             "statistics.")
    
    # to_dict transforms BASE_YR to float from int. Manually fix this
    row_dict = row.to_dict()
    row_dict['BASE_YR'] = int(row_dict['BASE_YR'])
    
    return row_dict

def income_wealth_dists_from_scf(base_year, age = None, education = None, wave = None):

    stats = parse_scf_distr_stats(age, education, wave)
    
    # Find the deflator to adjust nominal quantities. The SCF summary files
    # use the september CPI measurement to deflate, so use that.
    deflator = cpi_deflator(from_year = stats['BASE_YR'], to_year = base_year,
                            base_month='SEP')[0]
    
    # log(X*deflator) = log(x) + deflator.
    # Therefore, the deflator does not apply to:
    # - NrmWealth: it's the ratio of two nominal quantities, so unaltered by base changes.
    # - sd(ln(Permanent income)): the deflator is an additive shift to log-permanent income
    #   so the standard deviation is unchanged.
    
    log_deflator = np.log(deflator)
    param_dict = {
        'aNrmInitMean' : stats['lnNrmWealth.mean'],                 # Mean of log initial assets (only matters for simulation)
        'aNrmInitStd'  : stats['lnNrmWealth.sd'],                   # Standard deviation of log initial assets (only for simulation)
        'pLvlInitMean' : stats['lnPermIncome.mean'] + log_deflator, # Mean of log initial permanent income (only matters for simulation)
        'pLvlInitStd'  : stats['lnPermIncome.sd'],                  # Standard deviation of log initial permanent income (only matters for simulation)
    }
    
    return param_dict
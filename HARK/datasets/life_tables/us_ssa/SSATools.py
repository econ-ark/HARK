# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 15:36:14 2021

@author: Mateo
"""

import numpy as np
import pandas as pd
from warnings import warn
import os

ssa_tables_dir = os.path.dirname(os.path.abspath(__file__))

def get_ssa_life_tables():
    """
    Reads all the SSA life tables and combines them, adding columns indicating
    where each row came from (male or female, historical or projected).

    Returns
    -------
    Pandas DataFrame
        A DataFrame containing the information in SSA life-tables for both
        sexes and all the available years. It returns all the columns in the
        original tables.
        
    """
    # Read the four tables and add columns identifying them
    dsets = []
    for sex in ['M','F']:
        for method in ['Historical', 'Projected']:
            
            # Construct file name
            infix = 'Hist' if method == 'Historical' else 'Alt2'
            filename = os.path.join(ssa_tables_dir,
                                    'PerLifeTables_'+sex+'_'+infix+'_TR2020.csv')
            
            # Read csv
            table = pd.read_csv(filename, sep = ',', skiprows=4)
            
            # Add identifying info
            table['Sex'] = sex
            table['Method'] = method
            
            dsets.append(table)
    
    # Concatenate tables by row and return them
    return pd.concat(dsets)
    
def parse_ssa_life_table(min_age, max_age, female = True,
                         cohort = None, cross_sec = False, year = None):
    """
    Reads (year,age)-specifc death probabilities form SSA life tables and
    transforms them to a list of survival probabilities in the format that
    HARK expects.
    
    Two methods are supported:
        - Cross-sectional: finds the 1-year survival probabilities for
          individuals in the age range for a fixed year.
          In the output,
          SurvPrb(age) = 1 - DeathPrb(age, year)
          
        - Longitudinal: finds the 1-year survival probabilities for individuals
          of a fixed cohort at different ages (and years). 
          In the output,
          SurvPrb(age) = 1 - DeathPrb(age, cohort + age)
    
    Parameters
    ----------
    min_age : int
        Minimum age for survival probabilities.
    max_age : int
        Maximum age for survival probabilities.
    female : bool, optional
        Boolean indicating wether to use female or male survival probabilities.
        The default is True (female).
    cohort : int, optional
        If longitudinal probabilities are requested, this is the birth year of
        the cohort that will be tracked. The default is None.
    cross_sec : bool, optional
        Boolean indicating whether the cross-sectional method should be used.
        The default is False (using the longitudinal method).
    year : int, optional
        If cross-sectional probabilities are requestedm this is the year at
        which they will be taken. The default is None.
        
    Returns
    -------
    LivPrb : [float]
        List of 1-year survival probabilities.
        LivPrb[n] corresponds to the probability that an indivivual of age
        'min_age' + n survives one year, in the year 'year' if the
        cross-sectional method is used or 'cohort' + ('min_age' + n) if the
        longitudinal method is used.
        
    """
    
    
    # Infix for file name depending on sex
    abb = 'F' if female else 'M'
    
    # Find year - age combinations that we need
    assert max_age >= min_age, 'The maximum age can not be lower than the minimum age.'
    ages  = np.arange(min_age, max_age + 1)
    
    if cross_sec:
        
        if year is None:
            raise(TypeError('You must provide a year when using ' + 
                            'cross-sectional survival probabilities.'))
        
        years = np.repeat(year, ages.shape)
        
    else:
        
        if cohort is None:
            raise(TypeError('You must provide a cohort (birth year) when ' + 
                            'using longitudinal survival probabilities.'))
        
        years = cohort + ages
    
    # Create filenames
    
    # Historical and forecasted
    file_hist = os.path.join(ssa_tables_dir, 'PerLifeTables_'+abb+'_Hist_TR2020.csv')
    file_fore = os.path.join(ssa_tables_dir, 'PerLifeTables_'+abb+'_Alt2_TR2020.csv')
    
    # Read them
    hist_tab = pd.read_csv(file_hist, sep = ',', skiprows=4,
                           usecols=['Year','x','q(x)'], index_col = ['Year','x'])
    fore_tab = pd.read_csv(file_fore, sep = ',', skiprows=4,
                           usecols=['Year','x','q(x)'], index_col = ['Year','x'])
    
    # Find the point at which projections start
    max_hist = max(hist_tab.index.get_level_values('Year'))
    
    # Warn the user if projections are used.
    if max(years) > max_hist:
        message = 'Survival probabilities beyond {} are projections.'.format(max_hist)
        warn(message)
        
    # Concatenate them
    tab = pd.concat([hist_tab, fore_tab])
    
    # Subset and sort deathrates.
    
    message = ('Parsed life tables do not contain all the requested ' +
               'age-year combinations.')
    try:
        
        DeathPrb = tab.loc[zip(years,ages)].sort_values(by = 'x')
        
    except KeyError as e:
        
        raise Exception(message).with_traceback(e.__traceback__)
    
    # Transform to numpy survival probabilities
    LivPrb = 1 - DeathPrb['q(x)'].to_numpy()
    
    # Make sure we got all the probabilities
    assert len(LivPrb) == max_age - min_age + 1, message
    
    # Transform from array to list
    LivPrb = list(LivPrb)
    
    return LivPrb
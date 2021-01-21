# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 18:07:41 2021

@author: Mateo
"""
import urllib.request
import pandas as pd
import warnings
import numpy as np

def download_cpi_series():
    
    urllib.request.urlretrieve("https://www.bls.gov/cpi/research-series/r-cpi-u-rs-allitems.xlsx",
                               "r-cpi-u-rs-allitems.xlsx")

def get_cpi_series():
    
    cpi = pd.read_excel("r-cpi-u-rs-allitems.xlsx", skiprows = 5,
                        usecols = "A:N", index_col=0)
    
    return cpi
    
def cpi_deflator(from_year, to_year, base_month = None):
    
    # Check month is conforming
    if base_month is not None:
        
        months = ['JAN','FEB','MAR','APR','MAY','JUNE',
                  'JULY','AUG','SEP','OCT','NOV','DEC']
        
        assert base_month in months, ('If a month is provided, it must be ' +
                                      'one of ' + ','.join(months) + '.')
                                
        column = base_month
            
    else:
        
        warnings.warn('No base month was provided. Using annual CPI averages.')
        column = 'AVG'

    # Get cpi and subset the columns we need.
    cpi = get_cpi_series()
    cpi_series = cpi[[column]].dropna()
    
    try:
        
        deflator = np.divide(cpi_series.loc[from_year].to_numpy(),
                             cpi_series.loc[to_year].to_numpy())
        
    except KeyError as e:
        
        message = ("Could not find a CPI value for the requested " +
                   "year-month combinations.")
        raise Exception(message).with_traceback(e.__traceback__)
    
    return deflator

#cpi_deflator(1989,2007, 'OCT')
#cpi_deflator(1980,2010)
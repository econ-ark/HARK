# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 15:36:14 2021

@author: Mateo
"""

import numpy as np
import pandas as pd
from warnings import warn
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
    age = None, education = None, year = None
):
    
    # Read table
    filename = os.path.join(scf_sumstats_dir, "WealthIncomeStats.csv")

    # Read csv
    table = pd.read_csv(filename, sep=",",
                        index_col = ['Educ','YEAR','Age_grp'],
                        dtype = {'Educ': str,'YEAR': str,'Age_grp': str})
    
    print(table.loc[('College','1995','(20,25]')])

table = get_scf_distr_stats()
hm = parse_scf_distr_stats()
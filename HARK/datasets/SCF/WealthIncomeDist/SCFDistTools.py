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
    A function to read the full table of SCF summary statistics as a Pandas
    DataFrame

    Returns
    -------
    table : pandas DataFrame
        A pandas representation of file WealthIncomeStats.csv. See ./README.md
        for an explanation of the variables in the table and its source.
    """

    # Form the file name
    filename = os.path.join(scf_sumstats_dir, "WealthIncomeStats.csv")

    # Read csv
    table = pd.read_csv(filename, sep=",")

    return table


def parse_scf_distr_stats(age=None, education=None, wave=None):
    """
    A funtion to retreive SCF summary statistics regarding wealth and
    permanent income for a specific SCF wave, age bracket, and education
    level.

    Parameters
    ----------
    age : int, optional
        Age for which to retreive summary statistics. The statistics are
        calculated using 5-year age bins. Therefore, for instance, Age = 23
        will return statistics computed on ages (20,25].
        The default is None. In such case, the function will return statistics
        for the group without any age filtration.
    education : str, optional
        Education level for which to retreive summary statistics. Must be one
        of 'NoHS' (no high-school or GED), 'HS' (high-school or GED), or
        'College'.
        The default is None. In such case, no education filtration is applied
        (all groups are pooled).
    wave : int, optional
        SCF wave to use for summary statistics. Must be one of 1995, 1998,
        2001, 2004, 2007, 2010, 2013, 2016, 2019.
        The default is None. In such case, all waves are used.

    Returns
    -------
    row_dict : dict
        Dictionary with summary statistics for wealth and permanent income
        for the specified group. Its fields correspond to the columns of
        ./WealthIncomeStats.csv, which are described in ./README.md.

    """

    # Pre-process year to make it a five-year bracket as in the table
    if age is not None:

        u_bound = int(np.ceil(age / 5) * 5)
        l_bound = u_bound - 5
        age_bracket = "(" + str(l_bound) + "," + str(u_bound) + "]"

        warn("Returning SCF summary statistics for ages " + age_bracket + ".")

    else:

        # If no age is given, use all age brackets.
        age_bracket = "All"

    # Check whether education is in one of the allowed categories
    if education is not None:

        message = (
            "If an education level is provided, it must be one of "
            + "'NoHS', 'HS', or 'College'."
        )
        assert education in ["NoHS", "HS", "College"], message

    else:

        education = "All"

    # Parse the wave
    wave_str = "All" if wave is None else str(int(wave))

    # Read table
    filename = os.path.join(scf_sumstats_dir, "WealthIncomeStats.csv")

    # Read csv
    table = pd.read_csv(
        filename,
        sep=",",
        index_col=["Educ", "YEAR", "Age_grp"],
        dtype={"Educ": str, "YEAR": str, "Age_grp": str, "BASE_YR": int},
    )

    # Try to access the requested combination
    try:

        row = table.loc[(education, wave_str, age_bracket)]

    except KeyError as e:

        message = (
            "The summary statistics do not contain the "
            + "Age/Wave/Education combination that was requested."
        )
        raise Exception(message).with_traceback(e.__traceback__)

    # Check for NAs
    if any(row.isna()):
        warn(
            "There were not enough observations in the requested "
            + "Age/Wave/Education combination to compute all summary"
            + "statistics."
        )

    # to_dict transforms BASE_YR to float from int. Manually fix this
    row_dict = row.to_dict()
    row_dict["BASE_YR"] = int(row_dict["BASE_YR"])

    return row_dict


def income_wealth_dists_from_scf(base_year, age=None, education=None, wave=None):
    """
    Finds and formats parameters for the initial distributions of permanent
    income and normalized wealth from the SCF's summary statistics.
    
    Many of HARK's models (e.g. PerfForesightConsumerType.simBirth(),
    GenIncProcessConsumerType.simBirth()) assume the initial distribution
    of permanent income (pLvl) and normalized wealth (aNrm) are log-normal.
    They construct these distributions from their means and standard
    deviations, which are parameters to the models. This fuction assigns these
    parameters using summary statistics from the SCF.

    Parameters
    ----------
    base_year : int
        Base year to use for nominal quantities.
    age : int, optional
        Age for which to retreive summary statistics. See
        parse_scf_distr_stats(). The default is None.
    education : str, optional
        Edcuational attainment level for which to retreive summary
        statistics. See parse_scf_distr_stats(). The default is None.
    wave : int, optional
        SCF wave to use for summary statistics. See parse_scf_distr_stats().
        The default is None.

    Returns
    -------
    param_dict : dict
        Dictionary with means and standard deviations of the distributions
        of permanent income and normalized wealth.
    """

    # Extract summary statistics from the SCF table
    stats = parse_scf_distr_stats(age, education, wave)

    # Find the deflator to adjust nominal quantities. The SCF summary files
    # use the september CPI measurement to deflate, so use that.
    deflator = cpi_deflator(
        from_year=stats["BASE_YR"], to_year=base_year, base_month="SEP"
    )[0]

    # log(X*deflator) = log(x) + deflator.
    # Therefore, the deflator does not apply to:
    # - NrmWealth: it's the ratio of two nominal quantities, so unaltered by base changes.
    # - sd(ln(Permanent income)): the deflator is an additive shift to log-permanent income
    #   so the standard deviation is unchanged.

    log_deflator = np.log(deflator)
    param_dict = {
        "aNrmInitMean": stats[
            "lnNrmWealth.mean"
        ],  # Mean of log initial assets (only matters for simulation)
        "aNrmInitStd": stats[
            "lnNrmWealth.sd"
        ],  # Standard deviation of log initial assets (only for simulation)
        "pLvlInitMean": stats["lnPermIncome.mean"]
        + log_deflator,  # Mean of log initial permanent income (only matters for simulation)
        "pLvlInitStd": stats[
            "lnPermIncome.sd"
        ],  # Standard deviation of log initial permanent income (only matters for simulation)
    }

    return param_dict

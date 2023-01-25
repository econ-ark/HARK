"""
Created on Wed Jan 20 18:07:41 2021

@author: Mateo
"""

import os
import urllib.request
import pandas as pd
import numpy as np

from HARK import _log

__all__ = ["get_cpi_series", "cpi_deflator"]

us_cpi_dir = os.path.dirname(os.path.abspath(__file__))


def download_cpi_series():
    """
    A method that downloads the cpi research series file directly from the
    bls site onto the working directory.
    After being converted to a .csv, this is the file that the rest of
    the functions in this script use and must be placed in HARK/datasets/cpi/us.
    This function is not for users but for whenever mantainers want to update
    the cpi series as new data comes out.

    Returns
    -------
    None.

    """
    urllib.request.urlretrieve(
        "https://www.bls.gov/cpi/research-series/r-cpi-u-rs-allitems.xlsx",
        "r-cpi-u-rs-allitems.xlsx",
    )


def get_cpi_series():
    """
    This function reads the cpi series currently in the toolbox and returns it
    as a pandas dataframe.

    Returns
    -------
    cpi : Pandas DataFrame
        DataFrame representation of the CPI research series file from the
        Bureau of Labor Statistics.

    """

    cpi = pd.read_csv(
        os.path.join(us_cpi_dir, "r-cpi-u-rs-allitems.csv"),
        skiprows=5,
        index_col=0,
    )
    return cpi


def cpi_deflator(from_year, to_year, base_month=None):
    """
    Finds cpi deflator to transform quantities measured in "from_year" U.S.
    dollars to "to_year" U.S. dollars.
    The deflators are computed using the "r-cpi-u-rs" series from the BLS.

    Parameters
    ----------
    from_year : int
        Base year in which the nominal quantities are currently expressed.
    to_year : int
        Target year in which you wish to express the quantities.
    base_month : str, optional
        Month at which to take the CPI measurements to calculate the deflator.
        The default is None, and in this case annual averages of the CPI are
        used.

    Returns
    -------
    deflator : numpy array
        A length-1 numpy array with the deflator that, when multiplied by the
        original nominal quantities, rebases them to "to_year" U.S. dollars.

    """

    # Check years are conforming
    assert type(from_year) is int and type(to_year) is int, "Years must be integers."

    # Check month is conforming
    if base_month is not None:

        months = [
            "JAN",
            "FEB",
            "MAR",
            "APR",
            "MAY",
            "JUNE",
            "JULY",
            "AUG",
            "SEP",
            "OCT",
            "NOV",
            "DEC",
        ]

        assert base_month in months, (
            "If a month is provided, it must be " + "one of " + ",".join(months) + "."
        )

        column = base_month

    else:
        _log.debug("No base month was provided. Using annual CPI averages.")
        column = "AVG"

    # Get cpi and subset the columns we need.
    cpi = get_cpi_series()
    cpi_series = cpi[[column]].dropna()

    try:

        deflator = np.divide(
            cpi_series.loc[to_year].to_numpy(), cpi_series.loc[from_year].to_numpy()
        )

    except KeyError as e:

        message = (
            "Could not find a CPI value for the requested " + "year-month combinations."
        )
        raise Exception(message).with_traceback(e.__traceback__)

    return deflator

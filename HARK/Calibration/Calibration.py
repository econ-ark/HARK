# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 15:08:54 2020

@author: Mateo
"""

import numpy as np
from HARK.Calibration.Income.SabelhausSongProfiles import sabelhaus_song_var_profile
from HARK.datasets.cpi.us.CPITools import cpi_deflator

__all__ = ["Cagetti_income", "CGM_income", "ParseIncomeSpec", "findProfile"]


def AgeLogPolyToGrowthRates(coefs, age_min, age_max):
    """
    The deterministic component of permanent income is often expressed as a
    log-polynomial of age. In multiple HARK models, this part of the income
    process is expressed in a sequence of growth factors 'PermGroFac'.
    
    This function computes growth factors from the coefficients of a
    log-polynomial specification
    
    The form of the polynomial is assumed to be
    alpha_0 + age/10 * alpha_1 + age^2/100 * alpha_2 + ... + (age/10)^n * alpha_n
    Be sure to adjust the coefficients accordingly.
    
    Parameters
    ----------
    coefs : numpy array or list of floats
        Coefficients of the income log-polynomial, in ascending degree order
        (starting with the constant).
    age_min : int
        Starting age at which the polynomial applies.
    age_max : int
        Final age at which the polynomial applies.

    Returns
    -------
    GrowthFac : [float] of length age_max - age_min + 1
        List of growth factors that replicate the polynomial.
    
    P0 : float
        Initial level of income implied my the polynomial
    """
    # Figure out the degree of the polynomial
    deg = len(coefs) - 1

    # Create age matrices
    age_10 = np.arange(age_min, age_max + 1).reshape(age_max - age_min + 1, 1) / 10
    age_mat = np.hstack(list(map(lambda n: age_10 ** n, range(deg + 1))))

    # Fing the value of the polynomial
    lnYDet = np.dot(age_mat, np.array(coefs))

    # Find the starting level
    P0 = np.exp(lnYDet[0])

    # Compute growth factors
    GrowthFac = np.exp(np.diff(lnYDet))
    # The last growth factor is nan: we do not know lnYDet(age_max+1)
    GrowthFac = np.append(GrowthFac, np.nan)

    return GrowthFac.tolist(), P0


def findPermGroFacs(age_min, age_max, age_ret, AgePolyCoefs, ReplRate):
    """
    Finds initial income and sequence of growth factors from a polynomial
    specification of log-income, an optional retirement age and a replacement
    rate.
    
    Retirement income will be Income_{age_ret} * ReplRate.

    Parameters
    ----------
    age_min : int
        Initial age at which to compute the income specification.
    age_max : int
        Maximum age up to which the income process must be specified.
    age_ret : int
        Age of retirement. Note that retirement happens after labor income is
        received. For example, age_ret = 65 then the agent will receive labor
        income up to age 65 and retirement benefits starting at age 66.
        If age_ret is None, there will be no retirement.
    AgePolyCoefs : numpy array or list of floats
        Coefficients of the income log-polynomial, in ascending degree order
        (starting with the constant). Income follows the specification:
        ln(P)_age = \sum_{i=1}^{len(AgePolyCoefs)} (age/10)^i * AgePolyCoefs[i]
    ReplRate : float
        Replacement rate for retirement income.

    Returns
    -------
    GroFacs : list
        List of income growth factors.
    Y0 : float
        Level of income at age_min
    """
    
    if age_ret is None:
        
        # If there is no retirement, the age polynomial applies for the whole
        # lifetime
        GroFacs, Y0 = AgeLogPolyToGrowthRates(AgePolyCoefs, age_min, age_max)

    else:

        # First find working age growth rates and starting income
        WrkGroFacs, Y0 = AgeLogPolyToGrowthRates(AgePolyCoefs, age_min, age_ret)

        # Replace the last item, which must be NaN, with the replacement rate
        WrkGroFacs[-1] = ReplRate

        # Now create the retirement phase
        n_ret_years = age_max - age_ret
        RetGroFacs = [1.0] * (n_ret_years - 1) + [np.nan]

        # Concatenate
        GroFacs = WrkGroFacs + RetGroFacs

    return GroFacs, Y0


def ParseIncomeSpec(
    base_monet_year,
    age_min,
    age_max,
    age_ret=None,
    AgePolyCoefs=None,
    ReplRate=None,
    AgePolyRetir=None,
    YearTrend=None,
    start_year=None,
    PermShkStd=None,
    TranShkStd=None,
    SabelhausSong=False,
    adjust_infl_to=None,
    **unused
):

    income_params = {}
    # How many non-terminal periods are there.
    N_periods = age_max - age_min

    if age_ret is not None:
        # How many non terminal periods are spent working
        N_work_periods = age_ret - age_min + 1
        # How many non terminal periods are spent in retirement
        N_ret_periods = age_max - age_ret - 1

    # Growth factors
    if AgePolyCoefs is not None:

        if AgePolyRetir is None:

            PermGroFac, P0 = findPermGroFacs(
                age_min, age_max, age_ret, AgePolyCoefs, ReplRate
            )

        else:

            # Working period
            PermGroWrk, P0 = findPermGroFacs(
                age_min, age_ret, None, AgePolyCoefs, ReplRate
            )
            PLast = findProfile(PermGroWrk[:-1], P0)[-1]

            # Retirement period
            PermGroRet, R0 = findPermGroFacs(
                age_ret + 1, age_max, None, AgePolyRetir, ReplRate
            )

            # Input the replacement rate into the Work grow factors
            PermGroWrk[-1] = R0 / PLast
            PermGroFac = PermGroWrk + PermGroRet

        # In any case, PermGroFac[-1] will be np.nan, signaling that there is
        # no expected growth in the terminal period. Discard it, as HARK expect
        # list of growth rates for non-terminal periods
        PermGroFac = PermGroFac[:-1]

        # Apply the yearly trend if it is given
        if YearTrend is not None:

            # Compute and apply the compounding yearly growth factor
            YearGroFac = np.exp(YearTrend["Coef"])
            PermGroFac = [x * YearGroFac for x in PermGroFac]

            # Aggregate growth
            income_params["PermGroFacAgg"] = YearGroFac

            # Adjust P0 with the year trend
            if start_year is not None:
                P0 = P0 * np.power(YearGroFac, start_year - YearTrend["ZeroYear"])

        income_params["PermGroFac"] = PermGroFac

    else:

        # Placeholder for future ways of storing income calibrations
        raise NotImplementedError()

    # Volatilities
    # In this section, it is important to keep in mind that IncomeDstn[t]
    # is the income distribution from period t to t+1, as perceived in period
    # t.
    # Therefore (assuming an annual model with agents entering at age 0),
    # IncomeDstn[3] would contain the distribution of income shocks that occur
    # at the start of age 4.
    if SabelhausSong:

        if age_ret is None:

            IncShkStds = sabelhaus_song_var_profile(
                cohort=1950, age_min=age_min + 1, age_max=age_max
            )
            PermShkStd = IncShkStds["PermShkStd"]
            TranShkStd = IncShkStds["TranShkStd"]

        else:

            IncShkStds = sabelhaus_song_var_profile(
                cohort=1950, age_min=age_min + 1, age_max=age_ret
            )
            PermShkStd = IncShkStds["PermShkStd"] + [0.0] * (N_ret_periods + 1)
            TranShkStd = IncShkStds["TranShkStd"] + [0.0] * (N_ret_periods + 1)

    else:

        if isinstance(PermShkStd, float) and isinstance(TranShkStd, float):

            if age_ret is None:

                PermShkStd = [PermShkStd] * N_periods
                TranShkStd = [TranShkStd] * N_periods

            else:

                PermShkStd = [PermShkStd] * (N_work_periods - 1) + [0.0] * (
                    N_ret_periods + 1
                )
                TranShkStd = [TranShkStd] * (N_work_periods - 1) + [0.0] * (
                    N_ret_periods + 1
                )

        else:

            raise NotImplementedError()

    income_params["PermShkStd"] = PermShkStd
    income_params["TranShkStd"] = TranShkStd

    # Apply inflation adjustment if requested
    if adjust_infl_to is not None:

        # Deflate using the CPI september measurement, which is what the SCF
        # uses.
        defl = cpi_deflator(
            from_year=base_monet_year, to_year=adjust_infl_to, base_month="SEP"
        )[0]

    else:

        defl = 1

    P0 = P0 * defl
    income_params["P0"] = P0
    income_params["pLvlInitMean"] = np.log(P0)

    return income_params


def findProfile(GroFacs, Y0):

    factors = np.array([Y0] + GroFacs)
    Y = np.cumprod(factors)

    return Y


# Processes from Cocco, Gomes, Maenhout (2005).
CGM_income = {
    "NoHS": {
        "AgePolyCoefs": [-2.1361 + 2.6275, 0.1684 * 10, -0.0353 * 10, 0.0023 * 10],
        "age_ret": 65,
        "ReplRate": 0.8898,
        "PermShkStd": np.sqrt(0.0105),
        "TranShkStd": np.sqrt(0.1056),
        "base_monet_year": 1992,
    },
    "HS": {
        "AgePolyCoefs": [-2.1700 + 2.7004, 0.1682 * 10, -0.0323 * 10, 0.0020 * 10],
        "age_ret": 65,
        "ReplRate": 0.6821,
        "PermShkStd": np.sqrt(0.0106),
        "TranShkStd": np.sqrt(0.0738),
        "base_monet_year": 1992,
    },
    "College": {
        "AgePolyCoefs": [-4.3148 + 2.3831, 0.3194 * 10, -0.0577 * 10, 0.0033 * 10],
        "age_ret": 65,
        "ReplRate": 0.9389,
        "PermShkStd": np.sqrt(0.0169),
        "TranShkStd": np.sqrt(0.0584),
        "base_monet_year": 1992,
    },
}

# Processes from Cagetti (2003).
# - He uses volatilities from Carroll-Samwick (1997)
# - He expresses income in dollars. It is more amicable to express it in
#   thousands of dollars, also making it comparable to CGM. Thus, we substract
#   ln(1e3) = 3*ln(10) from intercepts, which is equivalent to dividing income
#   by a thousand.
Cagetti_income = {
    "NoHS": {
        "AgePolyCoefs": [7.99641616 - 3.0*np.log(10), 1.06559456, -0.14449728, 0.00048128, 0.0004096],
        "AgePolyRetir": [10.84636791- 3.0*np.log(10), -0.24562326],
        "YearTrend": {"Coef": 0.016, "ZeroYear": 1980},
        "age_ret": 65,
        "PermShkStd": np.sqrt(0.0214),  # Take 9-12 from CS
        "TranShkStd": np.sqrt(0.0658),  # Take 9-12 from CS
        "base_monet_year": 1992,
    },
    "HS": {
        "AgePolyCoefs": [
            10.01333075- 3.0*np.log(10),
            -0.563234304,
            0.348710528,
            -0.059442176,
            0.002947072,
        ],
        "AgePolyRetir": [11.21721558- 3.0*np.log(10), -0.26820465],
        "YearTrend": {"Coef": 0.016, "ZeroYear": 1980},
        "age_ret": 65,
        "PermShkStd": np.sqrt(0.0277),  # Take HS diploma from CS
        "TranShkStd": np.sqrt(0.0431),  # Take HS diploma from CS
        "base_monet_year": 1992,
    },
    "College": {
        "AgePolyCoefs": [
            9.916855488- 3.0*np.log(10),
            -0.057984416,
            0.146196992,
            -0.027623424,
            0.001282048,
        ],
        "AgePolyRetir": [10.81011279- 3.0*np.log(10), -0.16610233],
        "YearTrend": {"Coef": 0.016, "ZeroYear": 1980},
        "age_ret": 65,
        "PermShkStd": np.sqrt(0.0146),  # Take College degree from CS
        "TranShkStd": np.sqrt(0.0385),  # Take College degree from CS
        "base_monet_year": 1992,
    },
}

# %% Tools for setting time-related parameters


def ParseTimeParams(age_birth, age_death):
    """
    Converts simple statements of the age at which an agent is born and the
    age at which he dies with certaintiy into the parameters that HARK needs
    for figuring out the timing of the model.

    Parameters
    ----------
    age_birth : int
        Age at which the agent enters the model, e.g., 21.
    age_death : int
        Age at which the agent dies with certainty, e.g., 100.

    Returns
    -------
    dict
        Dictionary with parameters "T_cycle" and "T_age" which HARK expects
        and which map to the birth and death ages specified by the user.

    """
    # T_cycle is the number of non-terminal periods in the agent's problem
    T_cycle = age_death - age_birth
    # T_age is the age at which the agents are killed with certainty in
    # simulations (at the end of the T_age-th period)
    T_age = age_death - age_birth + 1

    return {"T_cycle": T_cycle, "T_age": T_age}

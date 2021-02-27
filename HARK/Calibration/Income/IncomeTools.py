# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 15:08:54 2020

@author: Mateo
"""

# %% Preamble

import numpy as np
from HARK.interpolation import LinearInterp
from HARK.datasets.cpi.us.CPITools import cpi_deflator

from HARK import _log

__all__ = [
    "parse_time_params",
    "Sabelhaus_Song_cohort_trend",
    "Sabelhaus_Song_all_years",
    "sabelhaus_song_var_profile",
    "Cagetti_income",
    "CGM_income",
    "parse_income_spec",
    "find_profile",
]


# %% Tools for setting time-related parameters


def parse_time_params(age_birth, age_death):
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


# %% Tools for finding the mean profiles of permanent income.


def age_log_poly_to_growth_rates(coefs, age_min, age_max):
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


def find_PermGroFacs(age_min, age_max, age_ret, AgePolyCoefs, ReplRate):
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
        GroFacs, Y0 = age_log_poly_to_growth_rates(AgePolyCoefs, age_min, age_max)

    else:

        # First find working age growth rates and starting income
        WrkGroFacs, Y0 = age_log_poly_to_growth_rates(AgePolyCoefs, age_min, age_ret)

        # Replace the last item, which must be NaN, with the replacement rate
        WrkGroFacs[-1] = ReplRate

        # Now create the retirement phase
        n_ret_years = age_max - age_ret
        RetGroFacs = [1.0] * (n_ret_years - 1) + [np.nan]

        # Concatenate
        GroFacs = WrkGroFacs + RetGroFacs

    return GroFacs, Y0


def find_profile(GroFacs, Y0):
    """
    Generates a sequence {Y_{t}}_{t=0}^N from an initial Y_0 and a sequence
    of growth factors GroFac[n] = Y_{n+1}/Y_n

    Parameters
    ----------
    GroFacs : list or numpy array
        Growth factors in chronological order.
    Y0 : float
        initial value of the series.

    Returns
    -------
    Y : numpy array
        Array with the values of the series.

    """
    factors = np.array([Y0] + GroFacs)
    Y = np.cumprod(factors)

    return Y


# %% Tools for life-cycle profiles of income volatility

# The raw results shared by John Sabelhaus contain the following two
# sets of estimates (with and without cohor trends), which we will
# use for constructing the age profiles.

# The first specification contains a cohort trend. The variance of
# (transitory or permanent) shocks to income of a person born in year
# "cohort" and who is now age "age" is
# age_dummy(age) + beta * (cohort - 1926)
# Where we have dummies for ages 27 to 54
Sabelhaus_Song_cohort_trend = {
    "Ages": np.arange(27, 55),
    "AgeDummiesPrm": np.array(
        [
            0.0837941,
            0.0706855,
            0.0638561,
            0.0603879,
            0.0554693,
            0.0532388,
            0.0515262,
            0.0486079,
            0.0455297,
            0.0456573,
            0.0417433,
            0.0420146,
            0.0391508,
            0.0395776,
            0.0369826,
            0.0387158,
            0.0365356,
            0.036701,
            0.0364236,
            0.0358601,
            0.0348528,
            0.0362901,
            0.0373366,
            0.0372724,
            0.0401297,
            0.0415868,
            0.0434772,
            0.046668,
        ]
    ),
    "AgeDummiesTrn": np.array(
        [
            0.1412842,
            0.1477754,
            0.1510265,
            0.1512203,
            0.1516837,
            0.151412,
            0.1489388,
            0.148521,
            0.1470632,
            0.143514,
            0.1411806,
            0.1378733,
            0.135245,
            0.1318365,
            0.1299689,
            0.1255799,
            0.1220823,
            0.1178995,
            0.1148793,
            0.1107577,
            0.1073337,
            0.102347,
            0.0962066,
            0.0918819,
            0.0887777,
            0.0835057,
            0.0766663,
            0.0698848,
        ]
    ),
    "CohortCoefPrm": -0.0005966,
    "CohortCoefTrn": -0.0017764 / 2,
}

# The second specification contains no cohort trend. The variance of
# (transitory or permanent) shocks to income of a person born in year
# "cohort" and who is now age "age" is: age_dummy(age)
# Where we have dummies for ages 27 to 54. We use this "aggregate"
# specification if no cohort is provided.
Sabelhaus_Song_all_years = {
    "Ages": np.arange(27, 55),
    "AgeDummiesPrm": np.array(
        [
            0.0599296,
            0.0474176,
            0.0411848,
            0.0383132,
            0.0339912,
            0.0323573,
            0.0312414,
            0.0289196,
            0.0264381,
            0.0271623,
            0.0238449,
            0.0247128,
            0.0224456,
            0.0234691,
            0.0214706,
            0.0238005,
            0.0222169,
            0.0229789,
            0.0232982,
            0.0233312,
            0.0229205,
            0.0249545,
            0.0265975,
            0.02713,
            0.0305839,
            0.0326376,
            0.0351246,
            0.038912,
        ]
    ),
    "AgeDummiesTrn": np.array(
        [
            0.1061999,
            0.1135794,
            0.1177187,
            0.1188007,
            0.1201523,
            0.1207688,
            0.1191838,
            0.1196542,
            0.1190846,
            0.1164236,
            0.1149784,
            0.1125594,
            0.1108192,
            0.1082989,
            0.1073195,
            0.1038188,
            0.1012094,
            0.0979148,
            0.0957829,
            0.0925495,
            0.0900136,
            0.0859151,
            0.0806629,
            0.0772264,
            0.0750105,
            0.0706267,
            0.0646755,
            0.0587821,
        ]
    ),
    "CohortCoefPrm": 0,
    "CohortCoefTrn": 0,
}


def sabelhaus_song_var_profile(age_min=27, age_max=54, cohort=None, smooth=True):
    """
    This is a function to find the life-cycle profiles of the volatilities
    of transitory and permanent shocks to income using the estimates in
    [1] Sabelhaus and Song (2010).

    Parameters
    ----------
    age_min : int, optional
        Minimum age at which to construct volatilities. The default is 27.
    age_max : int, optional
        Maximum age at which to construct volatilities. The default is 54.
    cohort : int, optional
        Birth year of the hypothetical person for which the volatilities will
        be constructed. The default is None, and in this case the we will
        use the specification that does not have cohort trends.
    smooth: bool, optional
        Boolean indicating whether to smooth the variance profile estimates
        using third degree polynomials for the age dummies estimated by
        Sabelhaus and Song. If False, the original dummies are used.

    Returns
    -------
    profiles : dict
        Dictionary with entries:
            - Ages: array of ages for which we found income volatilities in
                ascending order
            - TranShkStd: array of standard deviations of transitory income
                shocks. Position n corresponds to Ages[n].
            - PermShkStd: array of standard deviations of permanent income
                shocks. Position n corresponds to Ages[n].

        Note that TransShkStd[n] and PermShkStd[n] are the volatilities of
        shocks _experienced_ at age Age[n], (not those expected at Age[n+1]
        from the perspective of Age[n]).

        Note that Sabelhaus and Song work in discrete time and with periods
        that represent one year. Therefore, the outputs must be interpreted
        at the yearly frequency.
    """

    assert age_max >= age_min, (
        "The maximum age can not be lower than the " + "minimum age."
    )

    # Determine which set of estimates to use based on wether a cohort is
    # provided or not.
    if cohort is None:

        spec = Sabelhaus_Song_all_years
        cohort = 0
        _log.debug("No cohort was provided. Using aggregate specification.")

    else:

        spec = Sabelhaus_Song_cohort_trend

    # Extract coefficients
    beta_eps = spec["CohortCoefTrn"]
    beta_eta = spec["CohortCoefPrm"]
    tran_age_dummies = spec["AgeDummiesTrn"]
    perm_age_dummies = spec["AgeDummiesPrm"]

    # Smooth out dummies using a 3rd degree polynomial if requested
    if smooth:

        # Fit polynomials
        tran_poly = np.poly1d(np.polyfit(spec["Ages"], tran_age_dummies, deg=3))
        perm_poly = np.poly1d(np.polyfit(spec["Ages"], perm_age_dummies, deg=3))

        # Replace dummies
        tran_age_dummies = tran_poly(spec["Ages"])
        perm_age_dummies = perm_poly(spec["Ages"])

    # Make interpolators for transitory and permanent dummies. Alter to use
    # flat extrapolation.

    # We use Sabelhaus and Song (2010) dummies for ages 27-54 and extrapolate
    # outside of that just using the endpoints.
    tran_dummy_interp = LinearInterp(
        np.arange(min(spec["Ages"]) - 1, max(spec["Ages"]) + 2),
        np.concatenate(
            [[tran_age_dummies[0]], tran_age_dummies, [tran_age_dummies[-1]]]
        ),
        lower_extrap=True,
    )

    perm_dummy_interp = LinearInterp(
        np.arange(min(spec["Ages"]) - 1, max(spec["Ages"]) + 2),
        np.concatenate(
            [[perm_age_dummies[0]], perm_age_dummies, [perm_age_dummies[-1]]]
        ),
        lower_extrap=True,
    )

    if age_min < 27 or age_max > 54:
        _log.debug(
            "Sabelhaus and Song (2010) provide variance profiles for ages "
            + "27 to 54. Extrapolating variances using the extreme points."
        )

    if cohort < 1926 or cohort > 1980:
        _log.debug(
            "Sabelhaus and Song (2010) use data from birth cohorts "
            + "[1926,1980]. Extrapolating variances."
        )

        cohort = max(min(cohort, 1980), 1926)

    # Construct variances
    # They use 1926 as the base year for cohort effects.
    ages = np.arange(age_min, age_max + 1)
    tran_std = tran_dummy_interp(ages) + (cohort - 1926) * beta_eps
    perm_std = perm_dummy_interp(ages) + (cohort - 1926) * beta_eta

    profiles = {
        "Age": list(ages),
        "TranShkStd": list(tran_std),
        "PermShkStd": list(perm_std),
    }

    return profiles


# %% Encompassing tool to parse full income specifications


def parse_income_spec(
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
):
    """
    A function that produces income growth rates and income shock volatilities

    Parameters
    ----------
    base_monet_year : int
        Base monetary year in which the income process is specified. Answer to
        "In what year's U.S. dollars was income expressed in the process that
        will be parsed?".
    age_min : int
        Age at which agents enter the model.
    age_max : int
        Age at whih agents die with certainty. E.g., if age_max = 100, the
        agent dies at the end of his 100th year of life.
    age_ret : int, optional
        Age of retirement. The default is None.
    AgePolyCoefs : numpy array or list of floats
        Coefficients of the income log-polynomial, in ascending degree order
        (starting with the constant). Permanent income follows the specification:
        ln(P)_age = \sum_{i=1}^{len(AgePolyCoefs)} (age/10)^i * AgePolyCoefs[i].
        The default is None.
    ReplRate : float, optional
        Replacement rate for retirement income. Retirement income will be
        Income_{age_ret} * ReplRate. The default is None.
    AgePolyRetir : numpy array or list of floats
        Specifies a different age polynomial for income after retirement. It
        follows the same convention as AgePolyCoefs. The default is None.
    YearTrend : dict, optional
        Dictionary with entries "Coef" (float) and "ZeroYear" (int). Allows
        a time trend to be added to log-income. If provided, mean log-income at
        age a and year t will be:
        ln P = polynomial(a) + Coef * (t - ZeroYear)
        The default is None.
    start_year : int, optional
        Year at which the agent enters the model. This is important only for
        specifications with a time-trend for income profiles.
        The default is None.
    PermShkStd : float, optional
        Standard deviation of log-permanent-income shocks, if it is constant.
        The default is None.
    TranShkStd : float, optional
        Standard deviation of log-transitory-income shocks, if it is constant.
        The default is None.
    SabelhausSong : bool, optional
        Indicates whether to use transitory and permanent income shock
        volatilities from Sabelhaus & Song (2010) "The Great Moderation in
        Micro Labor Earnings". The default is False.
    adjust_infl_to : int, optional
        Year at which nominal quantities should be expressed. Answers the
        question "In what year's U.S. dollars should income be expressed".
        The default is None. In such case, base_monet_year will be used.

    Returns
    -------
    income_params : dict
        Dictionary with entries:
            - P0: initial level of permanent income.
            - pLvlInitMean: mean of the distribution of log-permanent income.
                np.log(P0) = pLvlInitMean
            - PermGroFac : list of deterministic growth factors for permanent
                income.
            - PermShkStd: list of standard deviations of shocks to
                log-permanent income.
            - TranShkStd: list of standard deviations of transitory shocks
                to income.
            - PermGroFacAgg: if a yearly trend in income is provided, this will
                be the aggregate level of growth in permanent incomes.

        This dictionary has the names and formats that various models in HARK
        expect, so that it can be directly updated into other parameter
        dictionaries.
    """

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

            PermGroFac, P0 = find_PermGroFacs(
                age_min, age_max, age_ret, AgePolyCoefs, ReplRate
            )

        else:

            # Working period
            PermGroWrk, P0 = find_PermGroFacs(
                age_min, age_ret, None, AgePolyCoefs, ReplRate
            )
            PLast = find_profile(PermGroWrk[:-1], P0)[-1]

            # Retirement period
            PermGroRet, R0 = find_PermGroFacs(
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

            # Placeholder for future ways of specifying volatilities
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


# %% Income specifications from various papers

# Processes from Cocco, Gomes, Maenhout (2005):
# Cocco, J. F., Gomes, F. J., & Maenhout, P. J. (2005). Consumption and
# portfolio choice over the life cycle. The Review of Financial Studies,
# 18(2), 491-533.
# - The profiles are provided as presented in the original paper.
# - It seem to us that income peaks at very young ages.
# - We suspect this might be due to the author's treatment of trends in income
#   growth.
# - This can be adressed using the YearTrend and pLvlGroFacAgg options.
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

# Processes from Cagetti (2003)
# Cagetti, M. (2003). Wealth accumulation over the life cycle and precautionary
# savings. Journal of Business & Economic Statistics, 21(3), 339-353.
# - The author generously provided estimates from which the age polynomials
#   and yearly trends were recovered.
# - He uses volatilities from Carroll-Samwick (1997)
# - He expresses income in dollars. It is more amicable to express it in
#   thousands of dollars, also making it comparable to CGM. Thus, we substract
#   ln(1e3) = 3*ln(10) from intercepts, which is equivalent to dividing income
#   by a thousand.
Cagetti_income = {
    "NoHS": {
        "AgePolyCoefs": [
            7.99641616 - 3.0 * np.log(10),
            1.06559456,
            -0.14449728,
            0.00048128,
            0.0004096,
        ],
        "AgePolyRetir": [10.84636791 - 3.0 * np.log(10), -0.24562326],
        "YearTrend": {"Coef": 0.016, "ZeroYear": 1980},
        "age_ret": 65,
        "PermShkStd": np.sqrt(0.0214),  # Take 9-12 from CS
        "TranShkStd": np.sqrt(0.0658),  # Take 9-12 from CS
        "base_monet_year": 1992,
    },
    "HS": {
        "AgePolyCoefs": [
            10.01333075 - 3.0 * np.log(10),
            -0.563234304,
            0.348710528,
            -0.059442176,
            0.002947072,
        ],
        "AgePolyRetir": [11.21721558 - 3.0 * np.log(10), -0.26820465],
        "YearTrend": {"Coef": 0.016, "ZeroYear": 1980},
        "age_ret": 65,
        "PermShkStd": np.sqrt(0.0277),  # Take HS diploma from CS
        "TranShkStd": np.sqrt(0.0431),  # Take HS diploma from CS
        "base_monet_year": 1992,
    },
    "College": {
        "AgePolyCoefs": [
            9.916855488 - 3.0 * np.log(10),
            -0.057984416,
            0.146196992,
            -0.027623424,
            0.001282048,
        ],
        "AgePolyRetir": [10.81011279 - 3.0 * np.log(10), -0.16610233],
        "YearTrend": {"Coef": 0.016, "ZeroYear": 1980},
        "age_ret": 65,
        "PermShkStd": np.sqrt(0.0146),  # Take College degree from CS
        "TranShkStd": np.sqrt(0.0385),  # Take College degree from CS
        "base_monet_year": 1992,
    },
}

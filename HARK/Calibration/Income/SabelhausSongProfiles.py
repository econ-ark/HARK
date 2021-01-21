# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 15:49:43 2021

@author: Mateo

This scripts contains representations of the results in:

[1] Sabelhaus, J., & Song, J. (2010). The great moderation in micro labor
    earnings. Journal of Monetary Economics, 57(4), 391-403.

It provides functions to produce life-cycle profiles of the variances of 
income shocks from the raw results from the paper.

The raw estimates were generously shared by John Sabelhaus.

"""

import numpy as np
from warnings import warn
from HARK.interpolation import LinearInterp


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


def sabelhaus_song_var_profile(age_min=27, age_max=54, cohort=None,
                               smooth = True):
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
    """

    assert age_max >= age_min, (
        "The maximum age can not be lower than the " + "minimum age."
    )

    # Determine which set of estimates to use based on wether a cohort is
    # provided or not.
    if cohort is None:

        spec = Sabelhaus_Song_all_years
        cohort = 0
        warn("No cohort was provided. Using aggregate specification.")

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
        tran_poly = np.poly1d(np.polyfit(spec['Ages'],
                                         tran_age_dummies, deg = 3))
        perm_poly = np.poly1d(np.polyfit(spec['Ages'],
                                         perm_age_dummies, deg = 3))
        
        # Replace dummies
        tran_age_dummies = tran_poly(spec['Ages'])
        perm_age_dummies = perm_poly(spec['Ages'])
        
    # Make interpolators for transitory and permanent dummies. Alter to use
    # flat extrapolation.

    # We use Sabelhaus and Song (2010) dummies for ages 27-54 and extrapolate
    # outside of that just using the endpoints.
    tran_dummy_interp = LinearInterp(
        np.arange(min(spec['Ages'])-1, max(spec['Ages'])+2),
        np.concatenate(
            [[tran_age_dummies[0]], tran_age_dummies, [tran_age_dummies[-1]]]
        ),
        lower_extrap=True,
    )

    perm_dummy_interp = LinearInterp(
        np.arange(min(spec['Ages'])-1, max(spec['Ages'])+2),
        np.concatenate(
            [[perm_age_dummies[0]], perm_age_dummies, [perm_age_dummies[-1]]]
        ),
        lower_extrap=True,
    )

    if age_min < 27 or age_max > 54:
        warn(
            "Sabelhaus and Song (2010) provide variance profiles for ages "
            + "27 to 54. Extrapolating variances using the extreme points."
        )

    if cohort < 1926 or cohort > 1980:
        warn(
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

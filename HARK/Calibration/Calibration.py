# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 15:08:54 2020

@author: Mateo
"""

import numpy as np

def AgeLogPolyToGrowthRates(coefs, age_min, age_max):
    """
    The deterministic component of permanent income is often expressed as a
    log-polynomial of age. In HARK, this part of the income process is
    expressed in a sequence of growth factors 'PermGroFac'.
    
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
    GrowthFac : [float]
        List of growth factors that replicate the polynomial.
    
    Y0 : float
        Initial level of income
    """
    # Figure out the degree of the polynomial
    deg = len(coefs) - 1
    
    # Create age matrices
    age_10 = np.arange(age_min, age_max + 1).reshape(age_max - age_min +1,1)/10
    age_mat = np.hstack(list(map(lambda n: age_10**n, range(deg+1))))
    
    # Fing the value of the polynomial
    lnYDet = np.dot(age_mat, np.array(coefs))
    
    # Find the starting level
    Y0 = np.exp(lnYDet[0])
    
    # Compute growth factors
    GrowthFac = np.exp(np.diff(lnYDet))
    # The last growth factor is nan: we do not know lnYDet(age_max+1)
    GrowthFac = np.append(GrowthFac, np.nan)
    
    return GrowthFac.tolist(), Y0
    
def findProfile(GroFacs, Y0):
    
    factors = np.array([Y0] + GroFacs[:-1])
    Y = np.cumprod(factors)
    
    return Y
    

CGM_income = {
    'NoHS'    : {'Poly': [-2.1361, 0.1684/10, -0.0353/10, 0.0023/10],
                        'ReplRate': 0.8898},
    
    'HS'      : {'Poly': [-2.1700, 0.1682/10, -0.0323/10, 0.0020/10],
                 'ReplRate': 0.6821},

    'College' : {'Poly': [-4.3148, 0.3194/10, -0.0577/10, 0.0033/10],
                 'ReplRate': 0.9389}
}

PermGroFac, Y0 = AgeLogPolyToGrowthRates(CGM_income['NoHS']['Poly'], 21,65)
Y = findProfile(PermGroFac, Y0)

import matplotlib.pyplot as plt

plt.plot(Y)

findProfile(PermGroFac, Y0)
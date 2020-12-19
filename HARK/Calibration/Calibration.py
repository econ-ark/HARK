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
        
    """
    # Figure out the degree of the polynomial
    deg = len(coefs) - 1
    
    # Create age matrices
    age = np.arange(age_min, age_max + 1).reshape(age_max - age_min +1,1)
    age_mat = np.hstack(list(map(lambda n: age**n, range(deg+1))))
    
    # Fing the value of the polynomial
    lnYDet = np.dot(age_mat, np.array(coefs))
    
    # Compute growth factors
    GrowthFac = np.exp(np.diff(lnYDet))
    # The last growth factor is nan: we do not know lnYDet(age_max+1)
    GrowthFac = np.append(GrowthFac, np.nan)
    
    return GrowthFac.tolist()
    

a=-2.170042+2.700381
b1=0.16818
b2=-0.0323371/10
b3=0.0019704/100

CoefsTest = np.array([a, b1, b2, b3])
print(AgeLogPolyToGrowthRates(CoefsTest, 20, 65))
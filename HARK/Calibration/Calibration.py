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

def findPermGroFacs(age_min, age_max, age_ret, PolyCoefs, ReplRate):
    
    if age_ret is None:

        GroFacs, Y0 = AgeLogPolyToGrowthRates(PolyCoefs, age_min, age_max)

    else:

        # First find working age growth rates and starting income
        WrkGroFacs, Y0 = AgeLogPolyToGrowthRates(PolyCoefs, age_min, age_ret)
        
        # Replace the last item, which must be NaN, with the replacement rate
        WrkGroFacs[-1] = ReplRate
        
        # Now create the retirement phase
        n_ret_years = age_max - age_ret
        RetGroFacs = [1.0] * (n_ret_years - 1) + [np.nan]
        
        # Concatenate
        GroFacs = WrkGroFacs + RetGroFacs
    
    return GroFacs, Y0
    

def ParseIncomeSpec(age_min, age_max,
                    age_ret = None, PolyCoefs = None, ReplRate = None,
                    PermShkStd = None, TranShkStd = None,
                    **unused):
    
    N_periods = age_max - age_min + 1
    
    if age_ret is not None:
        N_work_periods = age_ret - age_min + 1
        N_ret_periods  = age_max - age_ret
    
    # Growth factors
    if PolyCoefs is not None:
        
        PermGroFac, P0 = findPermGroFacs(age_min, age_max, age_ret,
                                         PolyCoefs, ReplRate)
        
    else:
        pass
    
    # Volatilities
    if isinstance(PermShkStd, float) and isinstance(TranShkStd, float):
        
        if age_ret is None:
            
            PermShkStd = [PermShkStd] * N_periods
            TranShkStd = [TranShkStd] * N_periods
    
        else:
            
            PermShkStd = [PermShkStd] * N_work_periods + [0.0] * N_ret_periods
            TranShkStd = [TranShkStd] * N_work_periods + [0.0] * N_ret_periods
            
    else:
        pass
    
    return {'PermGroFac': PermGroFac, 'P0': P0,
            'PermShkStd': PermShkStd, 'TranShkStd': TranShkStd}
    
def findProfile(GroFacs, Y0):
    
    factors = np.array([Y0] + GroFacs[:-1])
    Y = np.cumprod(factors)
    
    return Y
    

# Processes from Cocco, Gomes, Maenhout (2005).
CGM_income = {
    'NoHS'    : {'PolyCoefs': [-2.1361 + 2.6275, 0.1684*10, -0.0353*10, 0.0023*10],
                 'age_ret': 65,
                 'ReplRate': 0.8898,
                 'PermShkStd': np.sqrt(0.0105),
                 'TranShkStd': np.sqrt(0.1056),
                 'BaseYear': 1992},
    
    'HS'      : {'PolyCoefs': [-2.1700 + 2.7004, 0.1682*10, -0.0323*10, 0.0020*10],
                 'age_ret': 65,
                 'ReplRate': 0.6821,
                 'PermShkStd': np.sqrt(0.0106),
                 'TranShkStd': np.sqrt(0.0738),
                 'BaseYear': 1992},

    'College' : {'PolyCoefs': [-4.3148 + 2.3831, 0.3194*10, -0.0577*10, 0.0033*10],
                 'age_ret': 65,
                 'ReplRate': 0.9389,
                 'PermShkStd': np.sqrt(0.0169),
                 'TranShkStd': np.sqrt(0.0584),
                 'BaseYear': 1992}
}

# Processes from Cagetti (2003).
# - He uses volatilities from Carroll-Samwick (1997)
Cagetti_income = {
    'NoHS'    : {'PolyCoefs': [1.2430,0.6941,0.0361,-0.0259,0.0018],
                 'age_ret': 65,
                 'ReplRate': 0.8344,
                 'PermShkStd': np.sqrt(0.0214), # Take 9-12 from CS
                 'TranShkStd': np.sqrt(0.0658), # Take 9-12 from CS
                 'BaseYear': 1992},
    
    'HS'      : {'PolyCoefs': [3.0551,-0.6925,0.4339,-0.0703,0.0035],
                 'age_ret': 65,
                 'ReplRate': 0.8033,
                 'PermShkStd': np.sqrt(0.0277), # Take HS diploma from CS
                 'TranShkStd': np.sqrt(0.0431), # Take HS diploma from CS
                 'BaseYear': 1992},

    'College' : {'PolyCoefs': [2.1684,0.5230,-0.0002,-0.0057,0.0001],
                 'age_ret': 65,
                 'ReplRate': 0.7215,
                 'PermShkStd': np.sqrt(0.0146), # Take College degree from CS
                 'TranShkStd': np.sqrt(0.0385), # Take College degree from CS
                 'BaseYear': 1992}
}

import matplotlib.pyplot as plt

# %% CGM calibration

age_min = 21
age_max = 100

ages = np.arange(age_min, age_max + 1)

plt.figure()
for spec in CGM_income.items():
    
    label = spec[0]
    
    params = ParseIncomeSpec(age_min = age_min, age_max = age_max, **spec[1])
    MeanY = findProfile(params['PermGroFac'], params['P0'])
    
    plt.plot(ages, MeanY, label = label)

plt.title('CGM')
plt.legend()
plt.show()

# %% Cagetti calibration

age_min = 25
age_max = 91

ages = np.arange(age_min, age_max + 1)

plt.figure()
for spec in Cagetti_income.items():
    
    label = spec[0]
    
    params = ParseIncomeSpec(age_min = age_min, age_max = age_max, **spec[1])
    MeanY = findProfile(params['PermGroFac'], params['P0'])
    
    plt.plot(ages, MeanY, label = label)

plt.title('Cagetti')
plt.legend()
plt.show()
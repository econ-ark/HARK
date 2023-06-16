"""
This file implements unit tests to check discrete choice functions
"""
# Bring in modules we need
import unittest

from HARK.algos.egm import egm
from HARK.algos.foc import optimal_policy_foc
from HARK.gothic.gothic_class import Gothic
from HARK.gothic.resources import (
    Utility,
    DiscreteApproximation,
    DiscreteApproximationTwoIndependentDistribs,
)

from HARK.rewards import CRRAutilityP, CRRAutilityP_inv
import numpy as np
import os
from scipy import stats
from typing import Any, Mapping

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


# From SolvingMicroDSOPS
# Set up general parameters:

rho = 2.0  ### CRRA coefficient
beta = 0.96  ### discount factor
gamma = Gamma = np.array([1.0])  # permanent income growth factor
# A one-element "time series" array
# (the array structure needed for gothic class below)
R = 1.02  ## Risk free interest factor

# Define utility:
u = Utility(rho)

theta_sigma = 0.5
theta_mu = -0.5 * (theta_sigma**2)
theta_z = stats.lognorm(
    theta_sigma, 0, np.exp(theta_mu)
)  # Create "frozen" distribution instance
theta_grid_N = 7  ### how many grid points to approximate this continuous distribution

theta = DiscreteApproximation(
    N=theta_grid_N, cdf=theta_z.cdf, pdf=theta_z.pdf, invcdf=theta_z.ppf
)

gothic = Gothic(u, beta, rho, gamma, R, theta)

"""
g = lambda x, k, a : {'a' : x['m'] - a['c']},
dg_dx = 1,  ## Used in FOC method, step 5
dg_da = -1,  ## Used in FOC method, step 5
g_inv = lambda y, a : {'m' : y['a'] + a['c']},  ## Used in EGM method, step 8
r = lambda x, k, a : u(a['c']),
dr_da = lambda x, k, a: u.prime(a['c']),
dr_inv = lambda uP : (CRRAutilityP_inv(uP, rho),),

x = ['m'], 
a = ['c'],
y = ['a'],

TODO: Where are the constants from?
action_upper_bound = lambda x, k: (x['m'] + gamma[0] * theta.X[0] / R,),

#discount = beta, <-- Removed because beta is in gothic V!
    
##### Inputs to optimizers, interpolators, solvers...
optimizer_args = {
    'method' : 'Nelder-Mead',
    'options' : {
        'maxiter': 1e3,
        #'disp' : True
    }
},
"""

class foc_test(unittest.TestCase):
    """
    FOC test:

    pi(mVec) == cVec2 (close enough, 10 digits of precision)
    """

    def setUp(self):

        self.mVec = np.load(os.path.join(__location__, "smdsops_mVec.npy")) 
        self.cVec2 = np.load(os.path.join(__location__, "smdsops_cVec2.npy")) 


    def test_optimal_policy_foc(self):

        g = lambda x, k, a : {'a' : x['m'] - a['c']}
        dg_dx = 1  ## Used in FOC method, step 5
        dg_da = -1  ## Used in FOC method, step 5
        g_inv = lambda y, a : {'m' : y['a'] + a['c']}  ## Used in EGM method, step 8
        r = lambda x, k, a : u(a['c'])
        dr_da = lambda x, k, a: (CRRAutilityP(a['c'], rho),) # u.prime(a['c'])
        dr_da_inv = lambda uP : (CRRAutilityP_inv(uP, rho),)

        action_upper_bound = lambda x, k: (x['m'] + gamma[0] * theta.X[0] / R,),

        #discount = beta, <-- Removed because beta is in gothic V!
    
        ##### Inputs to optimizers, interpolators, solvers...
        ## TODO: Is this used?
        optimizer_args = {
            'method' : 'Nelder-Mead',
            'options' : {
                'maxiter': 1e3,
                #'disp' : True
            }
        }

        def consumption_v_y_der(y : Mapping[str,Any]):
            return gothic.VP_Tminus1(y['a'])
        
        pi_star, q_der, y_data = optimal_policy_foc(
            g,
            ['c'],
            r,
            dr_da,
            dr_da_inv,
            dg_dx,
            dg_da,
            {'m' : self.mVec},
            v_y_der = consumption_v_y_der,
            action_upper_bound = lambda x, z: (x['m'] + gamma[0] * theta.X[0] / R,),
            action_lower_bound = lambda x, z: (0,),
        )

        self.assertTrue(np.all(abs(self.cVec2 - pi_star.values) < 1e-12))

class egm_test(unittest.TestCase):
    """
    cVec_egm = egm(aVec) (machine precision)
    pi(mVec_egm) == cVec_egm
    """
    def setUp(self):
        self.aVec = np.load(os.path.join(__location__, "smdsops_aVec.npy")) 
        self.cVec_egm = np.load(os.path.join(__location__, "smdsops_cVec_egm.npy")) 
        self.mVec_egm = np.load(os.path.join(__location__, "smdsops_mVec_egm.npy")) 

    def test_egm(self):

        """

        ### EGM test from SolvingMicroDSOPs with stages
        def consumption_v_y_der(y : Mapping[str,Any]):
            return gothic.VP_Tminus1(y['a'])

        pi, pi_y = stage.optimal_policy_egm(
            y_grid = {'a' : aVec},
            v_y_der = consumption_v_y_der
        )
                                    
        pi.interp({'m' : mVec_egm}) -  cFunc_egm(mVec_egm) == 0
        """

        g = lambda x, k, a : {'a' : x['m'] - a['c']}
        dg_dx = 1  ## Used in FOC method, step 5
        dg_da = -1  ## Used in FOC method, step 5
        g_inv = lambda y, a : {'m' : y['a'] + a['c']}  ## Used in EGM method, step 8
        r = lambda x, k, a : u(a['c'])
        dr_da = lambda x, k, a: (CRRAutilityP(a['c'], rho),) # u.prime(a['c'])
        dr_da_inv = lambda uP : (CRRAutilityP_inv(uP, rho),)

        def consumption_v_y_der(y : Mapping[str,Any]):
            return gothic.VP_Tminus1(y['a'])

        pi_data, pi_y_data = egm(
            ['m'],
            ['c'],
            g_inv,
            dr_da_inv,
            dg_da,
            y_grid = {'a' : self.aVec},
            v_y_der = consumption_v_y_der,
        )

        if not np.all(self.cVec_egm == pi_y_data.values):
            print(self.cVec_egm)
            print(pi_y_data.values)
        
        self.assertTrue(np.all(self.cVec_egm == pi_y_data.values))
        self.assertTrue(np.all(self.mVec_egm == pi_data.coords['m'].values))

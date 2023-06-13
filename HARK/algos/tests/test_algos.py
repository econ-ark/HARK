"""
This file implements unit tests to check discrete choice functions
"""
# Bring in modules we need
import unittest

from HARK.algos.foc import optimal_policy_foc
from HARK.gothic.gothic_class import gothic
from HARK.rewards import CRRAutilityP_inv
import numpy as np
import os
from typing import Any, Mapping

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

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


    def test_x(self):

        g = lambda x, k, a : {'a' : x['m'] - a['c']},
        dg_dx = 1,  ## Used in FOC method, step 5
        dg_da = -1,  ## Used in FOC method, step 5
        g_inv = lambda y, a : {'m' : y['a'] + a['c']},  ## Used in EGM method, step 8
        r = lambda x, k, a : u(a['c']),
        dr_da = lambda x, k, a: u.prime(a['c']),
        dr_inv = lambda uP : (CRRAutilityP_inv(uP, rho),),

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
            {'m' : self.mVec},
            v_y_der = consumption_v_y_der
        )

        self.assertTrue(np.all(self.cVec2 == pi_star.values))

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

        self.assertTrue(np.all(self.aVec + self.cVec_egm == self.mVec_egm))
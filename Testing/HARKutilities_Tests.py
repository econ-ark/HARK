"""
This file implements unit tests to check HARKutilities.py
"""


# First, bring in the file we want
import sys
import os
sys.path.insert(0, os.path.abspath('../'))
import HARKutilities

# Bring in modules we need
import unittest
from copy import deepcopy
import numpy as np

class testsForHARKutilities(unittest.TestCase):
    
    def setUp(self):
        self.c_vals    = np.linspace(.5,10.,20)
        self.CRRA_vals = np.linspace(1.,10.,10)
    
    def first_diff_approx(self,func,x,delta,*args):
        """
        First (centered) difference approximation to a derivative
        """
        return (func(x+delta,*args) - func(x-delta,*args)) / (2. * delta)
        
    def derivative_func_comparison(self,deriv,func):
        for c in self.c_vals:
            for CRRA in self.CRRA_vals:
                diff = abs(deriv(c,CRRA) - self.first_diff_approx(func,c,.000001,CRRA))
                self.assertLess(diff,.01)
    
    def test_CRRAutilityP(self):
        self.derivative_func_comparison(HARKutilities.CRRAutilityP,HARKutilities.CRRAutility)
 
    def test_CRRAutilityPP(self):
        self.derivative_func_comparison(HARKutilities.CRRAutilityPP,HARKutilities.CRRAutilityP)           

    def test_CRRAutilityPPP(self):
        self.derivative_func_comparison(HARKutilities.CRRAutilityPPP,HARKutilities.CRRAutilityPP)   
        
    def test_CRRAutilityPPPP(self):
        self.derivative_func_comparison(HARKutilities.CRRAutilityPPPP,HARKutilities.CRRAutilityPPP)   
            
if __name__ == '__main__':
    print('testing')
    unittest.main()
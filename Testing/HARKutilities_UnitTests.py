"""
This file implements unit tests to check HARKutilities.py
"""
from __future__ import division
from __future__ import print_function

# First, bring in the files we want to test
import sys
import os
sys.path.insert(0, os.path.abspath('../'))
import HARKutilities

# Bring in modules we need
import unittest
import numpy as np

class testsForHARKutilities(unittest.TestCase):
    
    def setUp(self):
        self.c_vals    = np.linspace(.5,10.,20)
        self.CRRA_vals = np.linspace(1.,10.,10)
    
    def first_diff_approx(self,func,x,delta,*args):
        """
        Take the first (centered) difference approximation to the derivative of a function.
        
        """
        return (func(x+delta,*args) - func(x-delta,*args)) / (2. * delta)
        
    def derivative_func_comparison(self,deriv,func):
        """
        This method computes the first difference approximation to the derivative of a function
        "func" and the (supposedly) closed-form derivative of that function ("deriv") over a 
        grid.  It then checks that these two things are "close enough."
        """
        
        # Loop through different values of consumption
        for c in self.c_vals:
            # Loop through different values of risk aversion
            for CRRA in self.CRRA_vals:

                # Calculate the difference between the derivative of the function and the
                # first difference approximation to that derivative.                
                diff = abs(deriv(c,CRRA) - self.first_diff_approx(func,c,.000001,CRRA))

                # Make sure the derivative and its approximation are close                
                self.assertLess(diff,.01)
    
    def test_CRRAutilityP(self):
        # Test the first derivative of the utility function
        self.derivative_func_comparison(HARKutilities.CRRAutilityP,HARKutilities.CRRAutility)
 
    def test_CRRAutilityPP(self):
        # Test the second derivative of the utility function
        self.derivative_func_comparison(HARKutilities.CRRAutilityPP,HARKutilities.CRRAutilityP)           

    def test_CRRAutilityPPP(self):
        # Test the third derivative of the utility function
        self.derivative_func_comparison(HARKutilities.CRRAutilityPPP,HARKutilities.CRRAutilityPP)   
        
    def test_CRRAutilityPPPP(self):
        # Test the fourth derivative of the utility function
        self.derivative_func_comparison(HARKutilities.CRRAutilityPPPP,HARKutilities.CRRAutilityPPP)   
            
if __name__ == '__main__':
    print('testing Harkutilities.py')
    unittest.main()

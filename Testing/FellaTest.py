'''
A testing module for FellaInterp.  Should be deleted before merging into master.
'''

import sys
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('./'))
import numpy as np

from HARKinterpolation import FellaInterp
from HARKutilities import plotFuncs

X = np.array([0.,1.,2.,3.,4])
Y = 2*X
V = X**2

Test = FellaInterp(v0=1.5, control0=0.3, lower_bound=0.0, upper_bound=None)
Test.addNewPoints(X,V,Y,True)
Test.makeValueAndPolicyFuncs()


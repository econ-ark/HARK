'''
A testing module for FellaInterp.  Should be deleted before merging into master.
'''

import sys
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('./'))
import numpy as np
import matplotlib.pyplot as plt

from HARKinterpolation import FellaInterp
from HARKutilities import plotFuncs

#X = np.array([0.,1.,2.,3.,4.,5.,6.,7.])
#Y = 2*X
#V = 0.2*X

#X = np.linspace(0,10,21)
#Y = 2*X
#V = X - 0.1*X**2

#X = np.linspace(0,20,200)
#Y = 2*X
#V = 1. + np.sin(X)
#W = X
#Z = 1. + np.cos(X)

X = np.array([0.,1.,2.,3.,4.,5.,6.,7.,6.2,4.8,5.3,8.0])
Y = 2*np.arange(12,dtype=float)
Y1 = 3*np.arange(12,dtype=float)
V = np.array([0.,1.0,1.8,2.3,2.9,3.3,3.6,3.8,2.0,1.5,4.0,5.0])

plt.plot(X,V)
plt.show()

Test = FellaInterp(v0=1.5, control0=[0.3,0.5], lower_bound=0.0, upper_bound=None)
Test.addNewPoints(X,V,np.vstack((Y,Y1)),True)
#Test.addNewPoints(X,Z,W,True)
Test.makeValueAndPolicyFuncs()

plotFuncs(Test.ValueFunc,0,10)

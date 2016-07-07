# -*- coding: utf-8 -*-
"""
Created on Thu Jul 07 11:18:04 2016

@author: lowd
"""
import numpy as np
import pylab as plt 
from ConsIndShockModel import KinkedRconsumerType
from copy import deepcopy
import ConsumerParameters as Params
from HARKutilities import plotFuncs

# Define the baseline example
BaselineExample = KinkedRconsumerType(**Params.init_kinked_R)
BaselineExample.cycles  = 0 # Make the Example infinite horizon
BaselineExample.BoroCnstArt = -.3
BaselineExample.DiscFac = .4 #chosen so that target debt-to-permanent-income_ratio is about .1
                             # i.e. BaselineExample.cFunc[0](.9) = 1.

# Create the comparison example, a consumer with a borrowing constraint that is looser by .01
XtraCreditExample = deepcopy(BaselineExample)
XtraCreditExample.BoroCnstArt = -.31

# Solve the baseline example and prepare for graphing
BaselineExample.solve()
BaselineExample.unpackcFunc()
BaselineExample.timeFwd()

# Solve the comparison example and prepare for graphing
XtraCreditExample.solve()
XtraCreditExample.unpackcFunc()
XtraCreditExample.timeFwd()

## Plot the consumption functions, if desired.  Not really helpful since they look identical
#print('Consumption functions:')
#plotFuncs([BaselineExample.cFunc[0],XtraCreditExample.cFunc[0]],
#          BaselineExample.solution[0].mNrmMin,5)

# Define some function approximations to the MPC, to approximate the MPC
def ExampleFirstDiff(x):
    return XtraCreditExample.cFunc[0](x) - BaselineExample.cFunc[0](x)

def ExampleFirstDiffMPC(x):
    return ExampleFirstDiff(x) / .01

def UpwardFirstDiffMPC(x):
    return (BaselineExample.cFunc[0](x + .01) - BaselineExample.cFunc[0](x)) / .01

def DownwardFirstDiffMPC(x):
    return (BaselineExample.cFunc[0](x - .01) - BaselineExample.cFunc[0](x)) / -.01

#print('First difference')
#plotFuncs(ExampleFirstDiff,
#          BaselineExample.solution[0].mNrmMin,5)

print('Approx. MPC out of Credit')
plt.ylim([0.,1.2])
plotFuncs(ExampleFirstDiffMPC,
          BaselineExample.solution[0].mNrmMin,5)

print('Upward Approx. MPC out of Income')
plt.ylim([0.,1.2])
plotFuncs(UpwardFirstDiffMPC,
          BaselineExample.solution[0].mNrmMin,5)
          
print('Downward Approx. MPC out of Income')
plt.ylim([0.,1.2])
plotFuncs(DownwardFirstDiffMPC,
          BaselineExample.solution[0].mNrmMin,5)


print('MPC out of Credit v MPC out of Income')
plt.ylim([0.,1.2])
plotFuncs([ExampleFirstDiffMPC,UpwardFirstDiffMPC],
          BaselineExample.solution[0].mNrmMin,5)


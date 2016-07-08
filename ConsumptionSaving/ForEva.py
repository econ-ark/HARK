# -*- coding: utf-8 -*-
"""
Created on Thu Jul 07 11:18:04 2016

@author: lowd
"""
import numpy as np
import pylab as plt 
from ConsIndShockModel import KinkedRconsumerType,IndShockConsumerType
from copy import deepcopy
import ConsumerParameters as Params
from HARKutilities import plotFuncs

# Define the baseline example
BaselineExample = KinkedRconsumerType(**Params.init_kinked_R)
#BaselineExample = IndShockConsumerType(**Params.init_idiosyncratic_shocks)
#BaselineExample.Rfree       = 1.03
BaselineExample.cycles      = 0 # Make the Example infinite horizon
BaselineExample.CRRA        = 2.
BaselineExample.BoroCnstArt = -.3
credit_change               = .001
BaselineExample.DiscFac     = .5 #chosen so that target debt-to-permanent-income_ratio is about .1
                                 # i.e. BaselineExample.cFunc[0](.9) ROUGHLY = 1.

# Create the comparison example, a consumer with a borrowing constraint that is looser by credit_change
XtraCreditExample = deepcopy(BaselineExample)
XtraCreditExample.BoroCnstArt = BaselineExample.BoroCnstArt - credit_change

# Solve the baseline example and prepare for graphing
BaselineExample.solve()
BaselineExample.unpackcFunc()

# Solve the comparison example and prepare for graphing
XtraCreditExample.solve()
XtraCreditExample.unpackcFunc()

## Plot the consumption functions, if desired.  Not really helpful since they look identical
print('Consumption functions:')
plotFuncs([BaselineExample.cFunc[0],XtraCreditExample.cFunc[0]],
          BaselineExample.solution[0].mNrmMin,5)

# Define some function approximations to the MPC, to approximate the MPC
def ExampleFirstDiff(x):
    return XtraCreditExample.cFunc[0](x) - BaselineExample.cFunc[0](x)

def ExampleFirstDiffMPC(x):
    return ExampleFirstDiff(x) / (credit_change / BaselineExample.Rboro)

def UpwardFirstDiffMPC(x):
    return (BaselineExample.cFunc[0](x + credit_change) - BaselineExample.cFunc[0](x)) / credit_change

def DownwardFirstDiffMPC(x):
    return (BaselineExample.cFunc[0](x - credit_change) - BaselineExample.cFunc[0](x)) / -credit_change


x_max = 10.

print('Approx. MPC out of Credit')
plt.ylim([0.,1.2])
plotFuncs(ExampleFirstDiffMPC,
          BaselineExample.solution[0].mNrmMin,x_max)

print('Upward Approx. MPC out of Income')
plt.ylim([0.,1.2])
plotFuncs(UpwardFirstDiffMPC,
          BaselineExample.solution[0].mNrmMin,x_max)
          
print('Downward Approx. MPC out of Income')
plt.ylim([0.,1.2])
plotFuncs(DownwardFirstDiffMPC,
          BaselineExample.solution[0].mNrmMin,x_max)


print('MPC out of Credit v MPC out of Income')
plt.ylim([0.,1.2])
plotFuncs([ExampleFirstDiffMPC,UpwardFirstDiffMPC],
          BaselineExample.solution[0].mNrmMin,x_max)

#class_instance = BaselineExample
#class_instance.ExIncNext = np.dot(class_instance.IncomeDstn[0][0],class_instance.IncomeDstn[0][1]*class_instance.IncomeDstn[0][2])
#mZeroChangeFunc = lambda m : (1.0-class_instance.PermGroFac[0]/class_instance.Rfree)*m + \
#                  (class_instance.PermGroFac[0]/class_instance.Rfree)*class_instance.ExIncNext


#
## Simulate if desired
#BaselineExample.sim_periods = 120
#BaselineExample.makeIncShkHist()
#BaselineExample.initializeSim()
#BaselineExample.simConsHistory()
    

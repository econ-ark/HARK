"""

This is a HARK demo.

It is very heavily commented to that HARK newcomers can use it to figure out HARK works.
(Also: import statements in the code, rather than at the top)

There are many ways to use HARK.  This demo demonstrates one of the most valuable: using HARK
to import and solve a model...

"""
import numpy as np
import pylab as plt 
from copy import deepcopy


####################################################################################################
####################################################################################################
"""
The first step is to create the ConsumerType we want to solve the model for.
"""

## Import the HARK ConsumerType we want 
from ConsIndShockModel import IndShockConsumerType

## Import default parameter values
import ConsumerParameters as Params

## Now, create an instance of the consumer type using the default parameter values
## We create the instance of the consumer type by calling IndShockConsumerType()
## We use the default parameter values by passing **Params.init_idiosyncratic_shocks as an argument
BaselineExample = IndShockConsumerType(**Params.init_idiosyncratic_shocks)


#from ConsIndShockModel import KinkedRconsumerType
#BaselineExample = KinkedRconsumerType(**Params.init_kinked_R)



####################################################################################################
####################################################################################################

"""
The next step is to change the values of parameters as we want.

To see all the parameters used in the model, along with their default values, see
ConsumerParameters.py

Parameter values are stored as attributes of the ConsumerType the values are used for.
For example, the risk-free interest rate Rfree is stored as BaselineExample.Rfree.
Because we created BaselineExample using the default parameters values.
at the moment BaselineExample.Rfree is set to the default value of Rfree (which, at the time
this demo was written, was 1.03).  Therefore, to change the risk-free interest rate used in 
BaselineExample to (say) 1.02, all we need to do is:

BaselineExample.Rfree = 1.02

We will do this, and change a few other parameters while we're at it.
"""


BaselineExample.Rfree       = 1.02 #change the risk-free interest rate
BaselineExample.CRRA        = 2.   # change  the coefficient of relative risk aversion
BaselineExample.BoroCnstArt = -.3  # change the artificial borrowing constraint
BaselineExample.DiscFac     = .5 #chosen so that target debt-to-permanent-income_ratio is about .1
                                 # i.e. BaselineExample.cFunc[0](.9) ROUGHLY = 1.



# The most difficult...
BaselineExample.cycles      = 0 # Make the Example infinite horizon


####################################################################################################
####################################################################################################

"""
Now create a comparison...
"""



# Create the comparison example, a consumer with a borrowing constraint that is looser by credit_change
XtraCreditExample = deepcopy(BaselineExample)



credit_change               = .001
XtraCreditExample.BoroCnstArt = BaselineExample.BoroCnstArt - credit_change





####################################################################################################



# Solve the baseline example and prepare for graphing
BaselineExample.solve()
BaselineExample.unpackcFunc()

# Solve the comparison example and prepare for graphing
XtraCreditExample.solve()
XtraCreditExample.unpackcFunc()

from HARKutilities import plotFuncs

## Plot the consumption functions, if desired.  Not really helpful since they look identical
print('Consumption functions:')
plotFuncs([BaselineExample.cFunc[0],XtraCreditExample.cFunc[0]],
          BaselineExample.solution[0].mNrmMin,5)

# Define some function approximations to the MPC, to approximate the MPC
def ExampleFirstDiff(x):
    return XtraCreditExample.cFunc[0](x) - BaselineExample.cFunc[0](x)

def ExampleFirstDiffMPC(x):
    return ExampleFirstDiff(x) / (credit_change / 1.) # BaselineExample.Rboro)

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
    

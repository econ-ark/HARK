
"""
At the onset of the Great Recession, there was a large drop (X%) in consumer spending on 
non-durables.  Some economists have proffered that this could be attributed to precautionary 
motives-- a perceived increase in household income volatility induces more saving (less consumption)
to protect future consumption against bad income shocks.  How large of an increase in the standard
deviation of (log) permanent income shocks would be necessary to see an X% drop in consumption in
one quarter?  What about transitory income shocks?  How high would the perceived unemployment 
probability have to be?
"""

####################################################################################################
####################################################################################################
"""
The first step is to create the ConsumerType we want to solve the model for.
"""

## Import the HARK ConsumerType we want 
## Here, we bring in an agent making a consumption/savings decision every period, subject
## to transitory and permanent income shocks.
from ConsIndShockModel import IndShockConsumerType

## Import the default parameter values
import ConsumerParameters as BasicParams


import sys 
import os
sys.path.insert(0, os.path.abspath('../cstwMPC'))

import SetupParamsCSTW as cstwParams


BaselineExample = IndShockConsumerType(**cstwParams.init_infinite)
BaselineExample.DiscFac = BasicParams.init_idiosyncratic_shocks['DiscFac']
## Now, create an instance of the consumer type using the default parameter values
## We create the instance of the consumer type by calling IndShockConsumerType()
## We use the default parameter values by passing **Params.init_idiosyncratic_shocks as an argument
#BaselineExample = IndShockConsumerType(**Params.init_idiosyncratic_shocks)




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
"""

## Change some parameter values
"""
TODO: CHANGE PARAMETER VALUES to cstwMPC
"""


## There is one more parameter value we need to change.  This one is more complicated than the rest.
## We could solve the problem for a consumer with an infinite horizon of periods that (ex-ante)
## are all identical.  We could also solve the problem for a consumer with a fininite lifecycle,
## or for a consumer who faces an infinite horizon of periods that cycle (e.g., a ski instructor
## facing an infinite series of winters, with lots of income, and summers, with very little income.)
## The way to differentiate is through the "cycles" attribute, which indicates how often the
## sequence of periods needs to be solved.  The default value is 1, for a consumer with a finite
## lifecycle that is only experienced 1 time.  A consumer who lived that life twice in a row, and
## then died, would have cycles = 2.  But neither is what we want.  Here, we need to set cycles = 0,
## to tell HARK that we are solving the model for an infinite horizon consumer.


## Note that another complication with the cycles attribute is that it does not come from 
## Params.init_idiosyncratic_shocks.  Instead it is a keyword argument to the  __init__() method of 
## IndShockConsumerType.
#BaselineExample.cycles      = 0  
#




####################################################################################################
####################################################################################################
"""
Now, simulate
"""
import numpy as np

### First solve the baseline example.
BaselineExample.solve()

### Now simulate many periods to get to the stationary distribution

BaselineExample.sim_periods = 1000
BaselineExample.makeIncShkHist()
BaselineExample.initializeSim()
BaselineExample.simConsHistory()
    
# Now take the information from the last period, assuming we've reached stationarity
cNrm        = BaselineExample.cHist[-1,:]
pLvl        = BaselineExample.pHist[-1,:]
AgentCount  = cNrm.size
avgC        = np.sum(cNrm*pLvl)/AgentCount




####################################################################################################
####################################################################################################
"""
Now, create functions to change household income volatility in various ways
"""
from copy import deepcopy

def cChangeAfterVolChange(newVals,paramToChange):

    changesInConsumption = []

    for newVal in newVals:

        # Copy everything from the Baseline Example
        NewExample = deepcopy(BaselineExample)
        
        # Change what we want to change
        if paramToChange == "PermShkStd":
            NewExample.PermShkStd = [newVal]
        elif paramToChange == "TranShkStd":
            NewExample.TranShkStd = [newVal]
        elif paramToChange == "UnempPrb":
            NewExample.UnempPrb = newVal #note, unlike the others, not a list
        else:
            raise ValueError,'Invalid parameter to change!'            
        # Solve the new problem
        NewExample.updateIncomeProcess()
        NewExample.solve()
        
        # Advance the simulation one period
        NewExample.advanceIncShks()
        NewExample.advancecFunc()
        NewExample.simOnePrd()
        
        # Get new consumption
        newC    = NewExample.cNow
        newAvgC = np.sum(newC * NewExample.pNow) / AgentCount
        
        # Calculate and return the percent change in consumption
        changeInConsumption = 100. * (newAvgC - avgC) / avgC

        changesInConsumption.append(changeInConsumption)

    return changesInConsumption


def cChangeAfterPrmShkChange(newVals):
    return cChangeAfterVolChange(newVals,"PermShkStd")

def cChangeAfterTranShkChange(newVals):
    return cChangeAfterVolChange(newVals,"TranShkStd")

def cChangeAfterUnempPrbChange(newVals):
    return cChangeAfterVolChange(newVals,"UnempPrb")


## Now, plot the functions we want

# Import a useful plotting function from HARKutilities
from HARKutilities import plotFuncs
import pylab as plt # We need this module to change the y-axis on the graphs

xmin = .01
xmax = .2
targetChangeInC = -10.

plt.ylabel('% Change in Consumption')
plt.hlines(targetChangeInC,xmin,xmax)
plotFuncs([cChangeAfterPrmShkChange],xmin,xmax,N=5,legend_kwds = {'labels': ["PermShk"]})

plt.ylabel('% Change in Consumption')
plt.hlines(targetChangeInC,xmin,xmax)
plotFuncs([cChangeAfterTranShkChange],xmin,xmax,N=5,legend_kwds = {'labels': ["TranShk"]})


plt.ylabel('% Change in Consumption')
plt.hlines(targetChangeInC,xmin,xmax)
plotFuncs([cChangeAfterUnempPrbChange],xmin,xmax,N=5,legend_kwds = {'labels': ["UnempPrb"]})



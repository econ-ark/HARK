"""
At the onset of the Great Recession, there was a large drop (X%) in consumer spending on 
non-durables.  Some economists have proffered that this could be attributed to precautionary 
motives-- a perceived increase in household income uncertainty induces more saving (less consumption)
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

## Import some things from cstwMPC

# First, we need to be able to bring things in from the correct directory
import sys 
import os
sys.path.insert(0, os.path.abspath('../cstwMPC'))

# Now, bring in what we need from cstwMPC
import cstwMPC
import SetupParamsCSTW as cstwParams

# Now, initialize a baseline consumer type, using the default parameters from the infinite horizon cstwMPC
BaselineType = IndShockConsumerType(**cstwParams.init_infinite)

# The cstwMPC parameters do not define a discount factor, since there is ex-ante heterogeneity
# in the discount factor.  To prepare to create this ex-ante heterogeneity, first create
# the desired number of consumer types
from copy import deepcopy
ConsumerTypes = []
num_consumer_types = 3 #7

for nn in range(num_consumer_types):
    newType = deepcopy(BaselineType)    
    ConsumerTypes.append(newType)

## Now, generate the desired ex-ante heterogeneity, by giving the different consumer types
## each with their own discount factor

# First, decide the discount factors to assign
bottomDiscFac = 0.9800
topDiscFac    = 0.9934 
from HARKutilities import approxUniform
DiscFac_list = approxUniform(N=num_consumer_types,bot=bottomDiscFac,top=topDiscFac)[1]

# Now, assign the discount factors we want
cstwMPC.assignBetaDistribution(ConsumerTypes,DiscFac_list)




#####################################################################################################
#####################################################################################################
"""
Now, solve and simulate the model for each consumer type
"""

for ConsumerType in ConsumerTypes:

    ### First solve the baseline example.
    ConsumerType.solve()
    
    ### Now simulate many periods to get to the stationary distribution
    
    ConsumerType.sim_periods = 1000
    ConsumerType.makeIncShkHist()
    ConsumerType.initializeSim()
    ConsumerType.simConsHistory()




#####################################################################################################
#####################################################################################################
"""
Now, create functions to change household income uncertainty in various ways
"""
import numpy as np

def calcAvgC(Types):
    """
    Function to get average consumption in the economy in last simulated period
    """
    numTypes = len(Types)
    AgentCount = Types[0].cHist[-1,:].size * numTypes
    cNrm = np.array([0,])
    pLvl = np.array([0,])
        
        
    for Type in Types:
        ## Now take the information from the last period, assuming we've reached stationarity
        cNrm = np.append(cNrm,Type.cHist[-1,:])     
        pLvl = np.append(pLvl,Type.pHist[-1,:])

    avgC        = np.sum(cNrm*pLvl)/AgentCount
    return avgC
        

def cChangeAfterUncertaintyChange(consumerTypes,newVals,paramToChange):
    """
    Function to calculate the change in average consumption after a change in income uncertainty
    
    Inputs:
        consumerTypes, a list of consumer types
        
        newvals, new values for the income parameters
        
        paramToChange, a string telling the function which part of the income process to change
    """
    changesInConsumption = []
    oldAvgC = calcAvgC(consumerTypes)

    # Loop through the new values to assign, first assigning them, and then
    # solving and simulating another period with those values
    for newVal in newVals:

        # Copy everything we have from the consumerTypes 
        NewConsumerTypes = deepcopy(consumerTypes)
          
        for index,NewConsumerType in enumerate(NewConsumerTypes):
            # Change what we want to change
            if paramToChange == "PermShkStd":
                NewConsumerType.PermShkStd = [newVal]
            elif paramToChange == "TranShkStd":
                NewConsumerType.TranShkStd = [newVal]
            elif paramToChange == "UnempPrb":
                NewConsumerType.UnempPrb = newVal #note, unlike the others, not a list
            else:
                raise ValueError,'Invalid parameter to change!'            
            # Solve the new problem
            NewConsumerType.updateIncomeProcess()
            NewConsumerType.solve()
            
            # Advance the simulation one period
            NewConsumerType.sim_periods = 1
            NewConsumerType.makeIncShkHist()
            NewConsumerType.initializeSim(a_init=ConsumerTypes[index].aHist[-1:,:],
                                          p_init=ConsumerTypes[index].pHist[-1,:])
            NewConsumerType.simConsHistory()

            # Add the new period to the simulation history

            NewConsumerType.cHist = np.append(ConsumerTypes[index].cHist,
                                              NewConsumerType.cNow, #cNow has shape (N,1)
                                              axis=0)

            NewConsumerType.pHist = np.append(ConsumerTypes[index].pHist,
                                              NewConsumerType.pNow[np.newaxis,:], #pNow has shape (N,)
                                              axis=0)
        

                
        # Calculate and return the percent change in consumption
        newAvgC = calcAvgC(NewConsumerTypes)
        changeInConsumption = 100. * (newAvgC - oldAvgC) / oldAvgC

        changesInConsumption.append(changeInConsumption)

    return changesInConsumption

## Define functions that calculate the change in average consumption after income process changes
def cChangeAfterPrmShkChange(newVals):
    return cChangeAfterUncertaintyChange(ConsumerTypes,newVals,"PermShkStd")

def cChangeAfterTranShkChange(newVals):
    return cChangeAfterUncertaintyChange(ConsumerTypes,newVals,"TranShkStd")

def cChangeAfterUnempPrbChange(newVals):
    return cChangeAfterUncertaintyChange(ConsumerTypes,newVals,"UnempPrb")


## Now, plot the functions we want

# Import a useful plotting function from HARKutilities
from HARKutilities import plotFuncs
import pylab as plt # We need this module to change the y-axis on the graphs

ratio_min = 1. # obviously decreasing uncertainty won't do what we want...
ratio_max = 10.
targetChangeInC = -10.
num_points = 10

## First change the variance of the permanent income shock
perm_min = BaselineType.PermShkStd[0] * ratio_min
perm_max = BaselineType.PermShkStd[0] * ratio_max

plt.ylabel('% Change in Consumption')
plt.ylim(-20.,5.)
plt.hlines(targetChangeInC,perm_min,perm_max)
plotFuncs([cChangeAfterPrmShkChange],perm_min,perm_max,N=num_points,legend_kwds = {'labels': ["PermShk"]})


## Now change the variance of the temporary income shock
temp_min = BaselineType.TranShkStd[0] * ratio_min
temp_max = BaselineType.TranShkStd[0] * ratio_max

plt.ylabel('% Change in Consumption')
plt.ylim(-20.,5.)
plt.hlines(targetChangeInC,temp_min,temp_max)
plotFuncs([cChangeAfterTranShkChange],temp_min,temp_max,N=num_points,legend_kwds = {'labels': ["TranShk"]})



## Now change the probability of unemployment
unemp_min = BaselineType.UnempPrb * ratio_min
unemp_max = BaselineType.UnempPrb * ratio_max

plt.ylabel('% Change in Consumption')
plt.ylim(-20.,5.)
plt.hlines(targetChangeInC,unemp_min,unemp_max)
plotFuncs([cChangeAfterUnempPrbChange],unemp_min,unemp_max,N=num_points,legend_kwds = {'labels': ["UnempPrb"]})



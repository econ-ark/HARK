"""
At the onset of the Great Recession, there was a large drop (6.32%, according to FRED) in consumer 
spending on non-durables.  Some economists have proffered that this could be attributed to precautionary 
motives-- a perceived increase in household income uncertainty induces more saving (less consumption)
to protect future consumption against bad income shocks.  How large of an increase in the standard
deviation of (log) permanent income shocks would be necessary to see an 6.32% drop in consumption in
one quarter?  What about transitory income shocks?  How high would the perceived unemployment 
probability have to be?
"""

####################################################################################################
####################################################################################################
"""
The first step is to create the ConsumerType we want to solve the model for.

Model set up:
    * "Standard" infinite horizon consumption/savings model, with mortality and 
      permanent and temporary shocks to income
    * Ex-ante heterogeneity in consumers' discount factors
    
With this basic setup, HARK's IndShockConsumerType is the appropriate ConsumerType.
So we need to prepare the parameters to create that ConsumerType, and then create it.    
"""

## Import some things from cstwMPC

# The first step is to be able to bring things in from different directories
import sys 
import os
sys.path.insert(0, os.path.abspath('../')) #Path to ConsumptionSaving folder
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../cstwMPC')) #Path to cstwMPC folder

# Now, bring in what we need from cstwMPC
import cstwMPC
import SetupParamsCSTW as cstwParams

## Import the HARK ConsumerType we want 
## Here, we bring in an agent making a consumption/savings decision every period, subject
## to transitory and permanent income shocks.
from ConsIndShockModel import IndShockConsumerType

# Now initialize a baseline consumer type, using default parameters from infinite horizon cstwMPC
BaselineType = IndShockConsumerType(**cstwParams.init_infinite)

####################################################################################################
####################################################################################################
"""
Now, add in ex-ante heterogeneity in consumers' discount factors
"""

# The cstwMPC parameters do not define a discount factor, since there is ex-ante heterogeneity
# in the discount factor.  To prepare to create this ex-ante heterogeneity, first create
# the desired number of consumer types
from copy import deepcopy
num_consumer_types   = 7 # declare the number of types we want
ConsumerTypes = [] # initialize an empty list

for nn in range(num_consumer_types):
    # Now create the types, and append them to the list ConsumerTypes
    newType = deepcopy(BaselineType)    
    ConsumerTypes.append(newType)

## Now, generate the desired ex-ante heterogeneity, by giving the different consumer types
## each their own discount factor

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

    ### First solve the problem for this ConsumerType.
    ConsumerType.solve()
    
    ### Now simulate many periods to get to the stationary distribution
    ConsumerType.sim_periods = 1000
    ConsumerType.makeIncShkHist()
    ConsumerType.initializeSim()
    ConsumerType.simConsHistory()


#####################################################################################################
#####################################################################################################
"""
Now, create functions to see how aggregate consumption changes after household income uncertainty 
increases in various ways
"""
import numpy as np

# In order to see how consumption changes, we need to be able to calculate average consumption
# in the last period.  Create a function do to that here.
def calcAvgC(ConsumerTypes):
    """
    This function calculates average consumption in the economy in last simulated period,
    averaging across ConsumerTypes.
    """
    numTypes   = len(ConsumerTypes) # number of agent types in the economy
    AgentCount = ConsumerTypes[0].cHist[-1,:].size * numTypes #total number of agents in the economy
    cNrm = np.array([0,]) #initialize an array to hold consumption (normalized by permanent income)
    pLvl = np.array([0,]) #initialize an array to hold the level of permanent income
        
    # Now loop through all the ConsumerTypes, appending their cNrm and pLvl to the appropriate arrays
    for ConsumerType in ConsumerTypes:
        # Note we take the information from the last period
        cNrm = np.append(cNrm,ConsumerType.cHist[-1,:])     
        pLvl = np.append(pLvl,ConsumerType.pHist[-1,:])

    # Calculate and return average consumption it the economy
    avgC        = np.sum(cNrm*pLvl)/AgentCount 
    return avgC
        
# Now create a function to run the experiment we want -- change income uncertainty, and see
# how consumption changes
def cChangeAfterUncertaintyChange(consumerTypes,newVals,paramToChange):
    """
    Function to calculate the change in average consumption after change(s) in income uncertainty
    Inputs:
        * consumerTypes, a list of consumer types
        * newvals, a list of new values to use for the income parameters
        * paramToChange, a string telling the function which part of the income process to change
    """

    # Initialize an empty list to hold the changes in consumption that happen after parameters change.
    changesInConsumption = []
    
    # Get average consumption before parameters change
    oldAvgC = calcAvgC(consumerTypes)

    # Now loop through the new income parameter values to assign, first assigning them, and then
    # solving and simulating another period with those values
    for newVal in newVals:

        # Copy everything we have from the consumerTypes 
        ConsumerTypesNew = deepcopy(consumerTypes)
          
        for index,ConsumerTypeNew in enumerate(ConsumerTypesNew):
            # Change what we want to change
            if paramToChange == "PermShkStd":
                ConsumerTypeNew.PermShkStd = [newVal]
            elif paramToChange == "TranShkStd":
                ConsumerTypeNew.TranShkStd = [newVal]
            elif paramToChange == "UnempPrb":
                ConsumerTypeNew.UnempPrb = newVal #note, unlike the others, not a list
            else:
                raise ValueError,'Invalid parameter to change!'            

            # Because we changed the income process, and the income process is created
            # during initialization, we need to be sure to update the income process
            ConsumerTypeNew.updateIncomeProcess()

            # Solve the new problem
            ConsumerTypeNew.solve()
            
            # Advance the simulation one period
            ConsumerTypeNew.sim_periods = 1
            ConsumerTypeNew.makeIncShkHist() #make the history of income shocks
            
            ConsumerTypeNew.initializeSim( #prepare to simulate one more period...
              a_init=ConsumerTypes[index].aHist[-1:,:], # using assets from previous period as starting assets...
              p_init=ConsumerTypes[index].pHist[-1,:])  # and permanent income from previous period as starting permanent income
            
            ConsumerTypeNew.simConsHistory() # simulate one more period

            # Add the new period to the simulation history
            ConsumerTypeNew.cHist = np.append(ConsumerTypes[index].cHist,
                                              ConsumerTypeNew.cNow, #cNow has shape (N,1)
                                              axis=0)

            ConsumerTypeNew.pHist = np.append(ConsumerTypes[index].pHist,
                                              ConsumerTypeNew.pNow[np.newaxis,:], #pNow has shape (N,)
                                              axis=0)
        

                
        # Calculate the percent change in consumption, for this value newVal for the given parameter
        newAvgC = calcAvgC(ConsumerTypesNew)
        changeInConsumption = 100. * (newAvgC - oldAvgC) / oldAvgC

        # Append the change in consumption to the list changesInConsumption
        changesInConsumption.append(changeInConsumption)

    # Return the list of changes in consumption
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

ratio_min = 1. # minimum number to multiply income parameter by
targetChangeInC = -6.32 # Source: FRED
num_points = 10 #number of parameter values to plot in graphs

## First change the variance of the permanent income shock
perm_ratio_max = ??? # Put whatever value in you want!  maximum number to multiply std of perm income shock by

perm_min = BaselineType.PermShkStd[0] * ratio_min
perm_max = BaselineType.PermShkStd[0] * perm_ratio_max

plt.ylabel('% Change in Consumption')
plt.xlabel('Std. Dev. of Perm. Income Shock (Baseline = ' + str(round(BaselineType.PermShkStd[0],2)) + ')')
plt.title('Change in Cons. Following Increase in Perm. Income Uncertainty')
plt.ylim(-20.,5.)
plt.hlines(targetChangeInC,perm_min,perm_max)
plotFuncs([cChangeAfterPrmShkChange],perm_min,perm_max,N=num_points)


### Now change the variance of the temporary income shock
#temp_ratio_max = ??? # Put whatever value in you want!  maximum number to multiply std dev of temp income shock by
#
#
#temp_min = BaselineType.TranShkStd[0] * ratio_min
#temp_max = BaselineType.TranShkStd[0] * temp_ratio_max
#
#plt.ylabel('% Change in Consumption')
#plt.xlabel('Std. Dev. of Temp. Income Shock (Baseline = ' + str(round(BaselineType.TranShkStd[0],2)) + ')')
#plt.title('Change in Cons. Following Increase in Temp. Income Uncertainty')
#plt.ylim(-20.,5.)
#plt.hlines(targetChangeInC,temp_min,temp_max)
#plotFuncs([cChangeAfterTranShkChange],temp_min,temp_max,N=num_points)
#
#
#
### Now change the probability of unemployment
#unemp_ratio_max = ??? # Put whatever value in you want!  maximum number to multiply prob of unemployment by
#
#unemp_min = BaselineType.UnempPrb * ratio_min
#unemp_max = BaselineType.UnempPrb * unemp_ratio_max
#
#plt.ylabel('% Change in Consumption')
#plt.xlabel('Unemployment Prob. (Baseline = ' + str(round(BaselineType.UnempPrb,2)) + ')')
#plt.title('Change in Cons. Following Increase in Unemployment Prob.')
#plt.ylim(-20.,5.)
#plt.hlines(targetChangeInC,unemp_min,unemp_max)
#plotFuncs([cChangeAfterUnempPrbChange],unemp_min,unemp_max,N=num_points)



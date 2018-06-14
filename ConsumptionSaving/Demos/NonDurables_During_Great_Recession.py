"""
At the onset of the Great Recession, there was a large drop (6.32%, according to FRED) in consumer 
spending on non-durables.  Some economists have proffered that this could be attributed to precautionary 
motives-- a perceived increase in household income uncertainty induces more saving (less consumption)
to protect future consumption against bad income shocks.  How large of an increase in the standard
deviation of (log) permanent income shocks would be necessary to see an 6.32% drop in consumption in
one quarter?  What about transitory income shocks?  How high would the perceived unemployment 
probability have to be?

####################################################################################################
####################################################################################################

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
from __future__ import division, print_function
from builtins import str
from builtins import range
import sys 
import os
sys.path.insert(0, os.path.abspath('../')) #Path to ConsumptionSaving folder
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../cstwMPC')) #Path to cstwMPC folder
import numpy as np
from copy import deepcopy

# Now, bring in what we need from the cstwMPC parameters
import SetupParamsCSTW as cstwParams
from HARKutilities import approxUniform

## Import the HARK ConsumerType we want 
## Here, we bring in an agent making a consumption/savings decision every period, subject
## to transitory and permanent income shocks.
from ConsIndShockModel import IndShockConsumerType

# Now initialize a baseline consumer type, using default parameters from infinite horizon cstwMPC
BaselineType = IndShockConsumerType(**cstwParams.init_infinite)
BaselineType.AgentCount = 10000 # Assign the baseline consumer type to have many agents in simulation

####################################################################################################
####################################################################################################
"""
Now, add in ex-ante heterogeneity in consumers' discount factors
"""

# The cstwMPC parameters do not define a discount factor, since there is ex-ante heterogeneity
# in the discount factor.  To prepare to create this ex-ante heterogeneity, first create
# the desired number of consumer types
num_consumer_types   = 7 # declare the number of types we want
ConsumerTypes = [] # initialize an empty list

for nn in range(num_consumer_types):
    # Now create the types, and append them to the list ConsumerTypes
    newType = deepcopy(BaselineType)    
    ConsumerTypes.append(newType)
    ConsumerTypes[-1].seed = nn # give each consumer type a different RNG seed

## Now, generate the desired ex-ante heterogeneity, by giving the different consumer types
## each their own discount factor

# First, decide the discount factors to assign
bottomDiscFac  = 0.9800
topDiscFac     = 0.9934 
DiscFac_list   = approxUniform(N=num_consumer_types,bot=bottomDiscFac,top=topDiscFac)[1]

# Now, assign the discount factors we want
for j in range(num_consumer_types):
    ConsumerTypes[j].DiscFac = DiscFac_list[j]

#####################################################################################################
#####################################################################################################
"""
Now, solve and simulate the model for each consumer type
"""

for ConsumerType in ConsumerTypes:
    ### First solve the problem for this ConsumerType.
    ConsumerType.solve()
    
    ### Now simulate many periods to get to the stationary distribution
    ConsumerType.T_sim = 1000
    ConsumerType.initializeSim()
    ConsumerType.simulate()

#####################################################################################################
#####################################################################################################
"""
Now, create functions to see how aggregate consumption changes after household income uncertainty 
increases in various ways
"""

# In order to see how consumption changes, we need to be able to calculate average consumption
# in the last period.  Create a function do to that here.
def calcAvgC(ConsumerTypes):
    """
    This function calculates average consumption in the economy in last simulated period,
    averaging across ConsumerTypes.
    """
    # Make arrays with all types' (normalized) consumption and permanent income level
    cNrm = np.concatenate([ThisType.cNrmNow for ThisType in ConsumerTypes])
    pLvl = np.concatenate([ThisType.pLvlNow for ThisType in ConsumerTypes])
    
    # Calculate and return average consumption level in the economy
    avgC = np.mean(cNrm*pLvl) 
    return avgC
        
# Now create a function to run the experiment we want -- change income uncertainty, and see
# how consumption changes
def cChangeAfterUncertaintyChange(ConsumerTypes,newVals,paramToChange):
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
    oldAvgC = calcAvgC(ConsumerTypes)

    # Now loop through the new income parameter values to assign, first assigning them, and then
    # solving and simulating another period with those values
    for newVal in newVals:
        if paramToChange in ["PermShkStd","TranShkStd"]: # These parameters are time-varying, and thus are contained in a list.
            thisVal = [newVal] # We need to make sure that our updated values are *also* in a (one element) list.
        else:
            thisVal = newVal

        # Copy everything we have from the consumerTypes 
        ConsumerTypesNew = deepcopy(ConsumerTypes)
          
        for index,ConsumerTypeNew in enumerate(ConsumerTypesNew):
            setattr(ConsumerTypeNew,paramToChange,thisVal) # Set the changed value of the parameter        

            # Because we changed the income process, and the income process is created
            # during initialization, we need to be sure to update the income process
            ConsumerTypeNew.updateIncomeProcess()

            # Solve the new problem
            ConsumerTypeNew.solve()
            
            # Initialize the new consumer type to have the same distribution of assets and permanent
            # income as the stationary distribution we simulated above
            ConsumerTypeNew.initializeSim() # Reset the tracked history
            ConsumerTypeNew.aNrmNow = ConsumerTypes[index].aNrmNow # Set assets to stationary distribution
            ConsumerTypeNew.pLvlNow = ConsumerTypes[index].pLvlNow # Set permanent income to stationary dstn
            
            # Simulate one more period, which changes the values in cNrm and pLvl for each agent type
            ConsumerTypeNew.simOnePeriod()

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
import matplotlib.pyplot as plt # We need this module to change the y-axis on the graphs

ratio_min = 1. # minimum number to multiply income parameter by
targetChangeInC = -6.32 # Source: FRED
num_points = 10 #number of parameter values to plot in graphs

## First change the variance of the permanent income shock
perm_ratio_max = 5.0 #??? # Put whatever value in you want!  maximum number to multiply std of perm income shock by

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
#
#

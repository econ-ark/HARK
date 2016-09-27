"""
China's high net saving rate (25%) is a puzzle for economists, particularly in light of a 
consistently high income growth rate.  

If the last exercise made you worry that invoking difficult-to-measure "uncertainty" can explain
anything (e.g. "the stock market fell today because the risk aversion of the representative 
agent increased"), the next exercise may reassure you.  It is designed to show that there are 
limits to the phenomena that can be explained by invoking uncertainty.
 
It asks "what beliefs about uncertainty would Chinese consumers need to hold in order to generate a
saving rate of 25%, given the rapid pace of Chinese growth"?

Theory: huge jump upwards when you increase uncertainty, then decline over time
really: opposite!

"""



####################################################################################################
####################################################################################################
"""
The first step is to create the ConsumerType we want to solve the model for.

In our experiment, consumers will live in a stationary, low-growth environment (intended to 
approximate China before 1978).  Then, unexpectedly, income growth will surge at the same time
that income uncertainty increases (intended to approximate the effect of economic reforms in China
since 1978.)  Consumers believe high-growth, high-uncertainty state is highly persistent, but
temporary.

HARK's Markov ConsumerType will be a very convient way to run this experiment.  So we need to
prepare the parameters to create that ConsumerType, and then create it.
"""

### First bring in default parameter values from cstwPMC.  We will change these as necessary.

# The first step is to be able to bring things in from the correct directory
import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../cstwMPC'))


# Now, bring in what we need from cstwMPC
import cstwMPC
import SetupParamsCSTW as cstwParams
from copy import deepcopy

init_China_parameters = deepcopy(cstwParams.init_infinite)

### Now, change the parameters as necessary

# Declare some other important variables
import numpy as np

StateCount                      = 2 # just a low-growth state, and a high-growth state
ProbGrowthEnds                  = (1./160.)
MrkvArray                       = np.array([[1.,0.],[ProbGrowthEnds,1.-ProbGrowthEnds]])
init_China_parameters['Nagents']   = 10000
init_China_parameters['MrkvArray'] = MrkvArray

### Import the HARK ConsumerType we want 
### Here, we bring in an agent making a consumption/savings decision every period, subject
### to transitory and permanent income shocks, AND a Markov shock

from ConsMarkovModel import MarkovConsumerType
ChinaExample = MarkovConsumerType(**init_China_parameters)

# Currently, Markov states can differ in their interest factor, permanent growth factor, 
# survival probability (???), and income distribution.  Each of these needs to be specifically set.  
# Do that here, except income distribution.  That will be done later.

ChinaExample.assignParameters(PermGroFac = [np.array([1.,1.06 ** (.25)])], #needs to be a list, with 0th element of shape of shape (StateCount,)
                              Rfree      = np.array(StateCount*[init_China_parameters['Rfree']]), #need to be an array, of shape (StateCount,)
                              LivPrb     = [np.array(StateCount*[init_China_parameters['LivPrb']][0])], #needs to be a list, with 0th element of shape of shape (StateCount,)
                              cycles     = 0)





# The cstwMPC parameters do not define a discount factor, since there is ex-ante heterogeneity
# in the discount factor.  To prepare to create this ex-ante heterogeneity, first create
# the desired number of consumer types

ChineseConsumerTypes = []
num_consumer_types = 7

for nn in range(num_consumer_types):
    newType = deepcopy(ChinaExample)    
    ChineseConsumerTypes.append(newType)

## Now, generate the desired ex-ante heterogeneity, by giving the different consumer types
## each with their own discount factor

# First, decide the discount factors to assign
bottomDiscFac = 0.9800
topDiscFac    = 0.9934 

from HARKutilities import approxUniform
DiscFac_list = approxUniform(N=num_consumer_types,bot=bottomDiscFac,top=topDiscFac)[1]

# Now, assign the discount factors we want
cstwMPC.assignBetaDistribution(ChineseConsumerTypes,DiscFac_list)




####################################################################################################
####################################################################################################
"""
Now, write the function to do the experiment

Recall that all parameters have been assigned appropriately, except for the income process.

This is because we want to see how much uncertainty needs to accompany the high-growth state
to generate the high savings rate.
"""



# Decide the income distributions in the low- and high- income states
from ConsIndShockModel import constructLognormalIncomeProcessUnemployment

import ConsumerParameters as IncomeParams

LowGrowthIncomeDstn  = constructLognormalIncomeProcessUnemployment(IncomeParams)[0][0]
LowGrowth_PermShkStd = IncomeParams.PermShkStd



def calcNatlSavingRate(PrmShkVar_multiplier,RNG_seed = 0):

    # First, make a deepcopy of the ChineseConsumerTypes, because we are going to alter them
    NewChineseConsumerTypes = deepcopy(ChineseConsumerTypes)

    # Set the uncertainty in the high-growth state to the desired amount
    PrmShkStd_multiplier    = PrmShkVar_multiplier ** .5
    IncomeParams.PermShkStd = [LowGrowth_PermShkStd[0] * PrmShkStd_multiplier] 

    # Construct the appropriate income distributions
    HighGrowthIncomeDstn = constructLognormalIncomeProcessUnemployment(IncomeParams)[0][0]

    # Initialize national income/consumption
    NatlIncome = 0.
    NatlCons   = 0.



    for NewChineseConsumerType in NewChineseConsumerTypes:
        ### For each consumer type (i.e. each discount factor), calculate total income and consumption

        
        # First give each ConsumerType their own random number seed
        RNG_seed += 19
        NewChineseConsumerType.seed  = RNG_seed
        


        # Start by setting the income distribution appropriately        
        NewChineseConsumerType.IncomeDstn = [[LowGrowthIncomeDstn,HighGrowthIncomeDstn]]


        ####################################################################################################
        """
        Now we are ready to solve the consumer' problem.
        In HARK, this is done by calling the solve() method of the ConsumerType.
        """
        
        NewChineseConsumerType.solve()
        NewChineseConsumerType.solution[0].cFunc[0]
        ####################################################################################################
        """
        Now we are ready to simulate.
        
        This case will be a bit different than most, because agents' *perceptions* of the probability
        of changes in the Chinese economy will differ from the actual probability of changes.  Specifically,
        agents think there is a 0% chance of moving out of the low-growth state, and that there is a 
        (1./160) chance of moving out of the high-growth state.  In reality, we want the Chinese economy
        to to reach the low growth steady state, and then move into the high growth state with probability 
        1.  Then we want it to persist in the high growth state for 40 years. 
        
        
        
        """
        
        ## Now, simulate
        NewChineseConsumerType.sim_periods = 660 # 500 periods to get to steady state, then 40 years of high growth
        
        
        #### Now, CHOOSE the Markov states we want, rather than simulating them according to agents' perceived probabilities
        #DONT INVOKE makeMrkvHist, just set it to whatever we want (1000 periods of no growth, then change)
        #ChinaExample.Mrkv_init = np.zeros(ChinaExample.Nagents,dtype=int) #everyone starts off in low-growth state
        #ChinaExample.makeMrkvHist()
        
        # Declare the history for China that we are interested in
        ChineseHistory          = np.zeros((NewChineseConsumerType.sim_periods,
                                            NewChineseConsumerType.Nagents),dtype=int)
        ChineseHistory[-160:,:] = 1 #high-growth period!
        NewChineseConsumerType.MrkvHist   = ChineseHistory
        
        # Finish the rest of the simulation
        NewChineseConsumerType.makeIncShkHist() #create the history of income shocks, conditional on the Markov state
        NewChineseConsumerType.initializeSim() #get ready to simulate everything else
        NewChineseConsumerType.simConsHistory() #simulate everything else
    
    
    
        ####################################################################################################
        """
        Now the fun part: look at the results!
        """
        
        
        
        NatlIncome     += np.sum((NewChineseConsumerType.aHist * NewChineseConsumerType.pHist*
                                (NewChineseConsumerType.Rfree[0] - 1.)) + NewChineseConsumerType.pHist,axis=1)
        
        
        NatlCons       += np.sum(NewChineseConsumerType.cHist * NewChineseConsumerType.pHist,axis=1)

        

    NatlSavingRate = (NatlIncome - NatlCons)/NatlIncome


    return NatlSavingRate


####################################################################################################
####################################################################################################
"""
Now, run the experiment and plot the results
"""
import pylab as plt

max_increase = 11. #from Spinal Tap

periods_before_start = 5

x = np.arange(-periods_before_start,160,1)

NatlSavingsRates = []
PermShkVarMultipliers = (1.,2.,4.,8.,11.)

index = 0
for PermShkVarMultiplier in PermShkVarMultipliers:
    NatlSavingsRates.append(calcNatlSavingRate(PermShkVarMultiplier,RNG_seed = index)[-160 - periods_before_start:])
    index +=1

plt.ylabel('Natl Savings Rate')
plt.xlabel('Quarters Since Growth Surge')
plt.title('test')
plt.plot(x,NatlSavingsRates[0],label=str(PermShkVarMultipliers[0]))
plt.plot(x,NatlSavingsRates[1],label=str(PermShkVarMultipliers[1]))
plt.plot(x,NatlSavingsRates[2],label=str(PermShkVarMultipliers[2]))
plt.plot(x,NatlSavingsRates[3],label=str(PermShkVarMultipliers[3]))
plt.plot(x,NatlSavingsRates[4],label=str(PermShkVarMultipliers[4]))
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.) #put the legend on top



#put what happens after growth stops in back pocket]

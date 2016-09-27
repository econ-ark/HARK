"""
China's high net saving rate (25%) is a puzzle for economists, particularly in light of a 
consistently high income growth rate.  

If the last exercise made you worry that invoking difficult-to-measure "uncertainty" can explain
anything (e.g. "the stock market fell today because the risk aversion of the representative 
agent increased"), the next exercise may reassure you.  It is designed to show that there are 
limits to the phenomena that can be explained by invoking uncertainty.
 
It asks "what beliefs about uncertainty would Chinese consumers need to hold in order to generate a
saving rate of 25%"?

baseline: china in 1978.  social safety net disappears right when growth rate surges.

theory: huge jump upwards when you increase uncertainty, then decline over time
really: opposite!

"""



####################################################################################################
####################################################################################################
"""
The first step is to create the ConsumerType we want to solve the model for.
"""

## Import the HARK ConsumerType we want 
## Here, we bring in an agent making a consumption/savings decision every period, subject
## to transitory and permanent income shocks, AND a Markov shock
from ConsMarkovModel import MarkovConsumerType
import numpy as np
from copy import deepcopy

# First bring in default parameter values
# First, we need to be able to bring things in from the correct directory
import sys 
import os
sys.path.insert(0, os.path.abspath('../cstwMPC'))

# Now, bring in what we need from cstwMPC
#import cstwMPC
#import SetupParamsCSTW as cstwParams
#import SetupParamsCSTW as HighGrowthParams

import ConsumerParameters as LowGrowthIncomeParams
import ConsumerParameters as HighGrowthIncomeParams
import ConsumerParameters as Params

#import SetupParamsCSTW as LowGrowthIncomeParams
#import SetupParamsCSTW as HighGrowthIncomeParams

cstwParams.init_infinite['Nagents'] = 500
#init_China_example = deepcopy(cstwParams.init_infinite)
init_China_example = deepcopy(Params.init_idiosyncratic_shocks)

# Declare some other important variables
StateCount     = 2 # just a low-growth state, and a high-growth state
ProbGrowthEnds = (1./160.)
MrkvArray      = np.array([[1.,0.],[ProbGrowthEnds,1.-ProbGrowthEnds]])
init_China_example['MrkvArray'] = MrkvArray


ChinaExample = MarkovConsumerType(**init_China_example)

ChinaExample.assignParameters(PermGroFac = [np.array([1.,1.06 ** (.25)])], #needs tobe a list
                              Rfree      = np.array(StateCount*[ChinaExample.Rfree]), #neesd tobe an array
                              LivPrb     = [np.array(StateCount*[ChinaExample.LivPrb])], #needs tobe a list
                              cycles     = 0)




from ConsIndShockModel import constructLognormalIncomeProcessUnemployment

LowGrowthIncomeDstn  = ChinaExample.IncomeDstn[0]


#LowGrowthIncomeDstn  = constructLognormalIncomeProcessUnemployment(LowGrowthIncomeParams)[0][0]
#HighGrowthIncomeDstn = constructLognormalIncomeProcessUnemployment(HighGrowthIncomeParams)[0][0]

ChinaExample.DiscFac = .97
ChinaExample.IncomeDstn = [[LowGrowthIncomeDstn ,LowGrowthIncomeDstn ]]
ChinaExample.solve()








assert False



# The cstwMPC parameters do not define a discount factor, since there is ex-ante heterogeneity
# in the discount factor.  To prepare to create this ex-ante heterogeneity, first create
# the desired number of consumer types
from copy import deepcopy
ChineseConsumerTypes = []
num_consumer_types = 3 #7

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




# Decide the income distributions in the low- and high- income states
from ConsIndShockModel import constructLognormalIncomeProcessUnemployment

#LowGrowthIncomeDstn  = ChinaExample.IncomeDstn[0]

assert False
LowGrowthIncomeDstn  = constructLognormalIncomeProcessUnemployment(LowGrowthIncomeParams)[0][0]
HighGrowthIncomeDstn = constructLognormalIncomeProcessUnemployment(HighGrowthIncomeParams)[0][0]

test = ChineseConsumerTypes[0]

test.IncomeDstn = [[LowGrowthIncomeDstn ,LowGrowthIncomeDstn ]]
test.solve()
















def calcNatlSavingRate(multiplier):
    NewChineseConsumerTypes = deepcopy(ChineseConsumerTypes)

    HighGrowthIncomeParams.PermShkStd = [LowGrowthIncomeParams.PermShkStd[0] * multiplier] 

    LowGrowthIncomeDstn  = constructLognormalIncomeProcessUnemployment(LowGrowthIncomeParams)[0][0]
    HighGrowthIncomeDstn = constructLognormalIncomeProcessUnemployment(HighGrowthIncomeParams)[0][0]

    NatlIncome = 0.
    NatlCons   = 0.

    for NewChineseConsumerType in NewChineseConsumerTypes:
        print('hi')
#        NewChineseConsumerType.IncomeDstn = [[LowGrowthIncomeDstn,HighGrowthIncomeDstn]]
        NewChineseConsumerType.IncomeDstn = [[LowGrowthIncomeDstn ,LowGrowthIncomeDstn ]]
#        test = [np.ones(1),np.ones(1),np.ones(1)]
#        NewChineseConsumerType.IncomeDstn = [[test,test]]

        ####################################################################################################
        """
        Now we are ready to solve the consumer' problem.
        In HARK, this is done by calling the solve() method of the ConsumerType.
        """
        
        NewChineseConsumerType.solve()
        
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
        ChineseHistory          = np.zeros((NewChineseConsumerType.sim_periods,NewChineseConsumerType.Nagents),dtype=int)
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
        
        
        
        NatlIncome     += np.sum(NewChineseConsumerType.aHist * NewChineseConsumerType.pHist*(NewChineseConsumerType.Rfree[0]) + NewChineseConsumerType.pHist,axis=1)
        NatlCons       += np.sum(NewChineseConsumerType.cHist * NewChineseConsumerType.pHist,axis=1)
    NatlSavingRate = (NatlIncome - NatlCons)/NatlIncome


    return NatlSavingRate


#put what happens after growth stops in back pocket]

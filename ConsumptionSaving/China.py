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
import ConsumerParameters as Params
import ConsumerParameters as HighGrowthParams
init_China_example = deepcopy(Params.init_idiosyncratic_shocks)

# Declare some other important variables
StateCount     = 2 # just a low-growth state, and a high-growth state
ProbGrowthEnds = (1./160.)
MrkvArray      = np.array([[1.,0.],[ProbGrowthEnds,1.-ProbGrowthEnds]])
init_China_example['MrkvArray'] = MrkvArray


ChinaExample = MarkovConsumerType(**init_China_example)

#assert False
#ARE PARAMETERS RIGHT FOR A QUARTERLY MODEL??? DiscFac is .96 -- meaning .96**4., or .85 annually!



ChinaExample.assignParameters(PermGroFac = [np.array([1.,1.06 ** (.25)])], #needs tobe a list
                              Rfree      = np.array(StateCount*[1.03]), #neesd tobe an array
                              LivPrb     = [np.array(StateCount*[.98])], #needs tobe a list
                              cycles     = 0)

# Decide the income distributions in the low- and high- income states
from ConsIndShockModel import constructLognormalIncomeProcessUnemployment

LowGrowthIncomeDstn  = ChinaExample.IncomeDstn[0]


def calcNatlSavingRate(multiplier):
    NewExample = deepcopy(ChinaExample)

    HighGrowthParams.PermShkStd = [HighGrowthParams.PermShkStd[0] * multiplier] 
    HighGrowthIncomeDstn = constructLognormalIncomeProcessUnemployment(HighGrowthParams)[0][0]

    
    NewExample.IncomeDstn = [[LowGrowthIncomeDstn,HighGrowthIncomeDstn]]
    
    ####################################################################################################
    """
    Now we are ready to solve the consumer' problem.
    In HARK, this is done by calling the solve() method of the ConsumerType.
    """
    
    NewExample.solve()
    
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
    NewExample.sim_periods = 1160 # 1000 periods to get to steady state, then 40 years of high growth
    
    
    #### Now, CHOOSE the Markov states we want, rather than simulating them according to agents' perceived probabilities
    #DONT INVOKE makeMrkvHist, just set it to whatever we want (1000 periods of no growth, then change)
    #ChinaExample.Mrkv_init = np.zeros(ChinaExample.Nagents,dtype=int) #everyone starts off in low-growth state
    #ChinaExample.makeMrkvHist()
    
    # Declare the history for China that we are interested in
    ChineseHistory          = np.zeros((NewExample.sim_periods,NewExample.Nagents),dtype=int)
    ChineseHistory[-160:,:] = 1 #high-growth period!
    NewExample.MrkvHist   = ChineseHistory
    
    # Finish the rest of the simulation
    NewExample.makeIncShkHist() #create the history of income shocks, conditional on the Markov state
    NewExample.initializeSim() #get ready to simulate everything else
    NewExample.simConsHistory() #simulate everything else
    
    
    
    
    
    
    
    ####################################################################################################
    """
    Now the fun part: look at the results!
    """
    
    
    
    NatlIncome     = np.sum(NewExample.aHist * NewExample.pHist*(NewExample.Rfree[0]) + NewExample.pHist,axis=1)
    NatlCons       = np.sum(NewExample.cHist * NewExample.pHist,axis=1)
    NatlSavingRate = (NatlIncome - NatlCons)/NatlIncome


    return NatlSavingRate


#put what happens after growth stops in back pocket]

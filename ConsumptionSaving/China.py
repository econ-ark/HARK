"""
China's high net saving rate (25%) is a puzzle for economists, particularly in light of a 
consistently high income growth rate.  

If the last exercise made you worry that invoking difficult-to-measure "uncertainty" can explain
 anything (e.g. "the stock market fell today because the risk aversion of the representative 
 agent increased"), the next exercise may reassure you.  It is designed to show that there are 
 limits to the phenomena that can be explained by invoking uncertainty.
 
It asks "what beliefs about uncertainty would Chinese consumers need to hold in order to generate a
saving rate of 25%"?



baseline: china in 1978



social safety net, and growth rate, change at same time

switch from 0% growth to 6% growth.  For 40 years.


theory:
huge jump upwards when you increase uncertainty, then decline over time

really: opposite!

show path of savings rate, vs just one number



put in 1/40 chance returns to 0% growth
[put what happens after growth stops in back pocket]


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


StateCount = 2 # just a low-growth state, and a high-growth state

ProbGrowthEnds = (1./160.)
MrkvArray = np.array([[1.,0.],[ProbGrowthEnds,1.-ProbGrowthEnds]])

import ConsumerParameters as Params

#ARE PARAMETERS RIGHT FOR A QUARTERLY MODEL??? DiscFac is .96 -- meaning .96**4., or .85 annually!

init_China_example = deepcopy(Params.init_idiosyncratic_shocks)
init_China_example['MrkvArray'] = MrkvArray

ChinaExample = MarkovConsumerType(**init_China_example)

ChinaExample.assignParameters(PermGroFac = [np.array([1.,1.06 ** (.25)])], #needs tobe a list
                              Rfree  = np.array(StateCount*[1.03]), #neesd tobe an array
                              LivPrb = [np.array(StateCount*[.98])], #needs tobe a list
                              cycles=0)


LowGrowthIncomeDstn  = ChinaExample.IncomeDstn[0]
HighGrowthIncomeDstn = ChinaExample.IncomeDstn[0]

ChinaExample.IncomeDstn = [[LowGrowthIncomeDstn,HighGrowthIncomeDstn]]

ChinaExample.solve()


## Now, simulate

ChinaExample.sim_periods = 1160 #1000 periods to get to steady state, then 40 years of take-off growth
#ChinaExample.Mrkv_init = np.zeros(ChinaExample.Nagents,dtype=int) #everyone starts off in low-growth state

#### Now, CHOOSE the Markov states we want, rather than simulating them according to agents' perceived probabilities
ChineseHistory = np.zeros((ChinaExample.sim_periods,ChinaExample.Nagents),dtype=int)
ChineseHistory[-160:,:] = 1 #high-growth period!

ChinaExample.MrkvHist = ChineseHistory
ChinaExample.makeIncShkHist()
ChinaExample.initializeSim()
ChinaExample.simConsHistory()

NatlIncome = np.sum(ChinaExample.aHist * ChinaExample.pHist*(ChinaExample.Rfree[0]) + ChinaExample.pHist,axis=1)
NatlCons   = np.sum(ChinaExample.cHist * ChinaExample.pHist,axis=1)
NatlSavingRate = (NatlIncome - NatlCons)/NatlIncome
#NatlSavingRate = 
#Y = np.sum(aNrm*pLvl*(Rfree-1) + pLvl), SavingRate = (Y - np.sum(cNrm*pLvl)/Y

#        SerialUnemploymentExample.sim_periods = 120
#        SerialUnemploymentExample.Mrkv_init = np.zeros(SerialUnemploymentExample.Nagents,dtype=int)
#        SerialUnemploymentExample.makeMrkvHist()
#        SerialUnemploymentExample.makeIncShkHist()
#        SerialUnemploymentExample.initializeSim()
#        SerialUnemploymentExample.simConsHistory()





#DONT INVOKE makeMrkvHist, just set it to whatever we want (1000 periods of no growth, then change)
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



init_China_example = deepcopy(Params.init_idiosyncratic_shocks)
init_China_example['MrkvArray'] = MrkvArray

ChinaExample = MarkovConsumerType(**init_China_example)

ChinaExample.assignParameters(PermGroFac = [np.array([1.,1.06])],
                              Rfree = np.array(StateCount*[1.03]),
                              cycles=0)

UnempPrb = 0.05    # Unemployment probability

IncomeDstnReg = [np.array([1-UnempPrb,UnempPrb]), np.array([1.0,1.0]), np.array([1.0,0.0])]
IncomeDstn = StateCount*[IncomeDstnReg] # Same simple income distribution in each state


ChinaExample.IncomeDstn = [IncomeDstn]  

ChinaExample.solve()
#
## Make a consumer with serially correlated permanent income growth
#UnempPrb = 0.05    # Unemployment probability
#StateCount = 5     # Number of permanent income growth rates
#Persistence = 0.5  # Probability of getting the same permanent income growth rate next period
#
#IncomeDstnReg = [np.array([1-UnempPrb,UnempPrb]), np.array([1.0,1.0]), np.array([1.0,0.0])]
#IncomeDstn = StateCount*[IncomeDstnReg] # Same simple income distribution in each state
#
## Make the state transition array for this type: Persistence probability of remaining in the same state, equiprobable otherwise
#MrkvArray = Persistence*np.eye(StateCount) + (1.0/StateCount)*(1.0-Persistence)*np.ones((StateCount,StateCount))
#    
#    init_serial_growth = copy(Params.init_idiosyncratic_shocks)
#    init_serial_growth['MrkvArray'] = MrkvArray
#    SerialGroExample = MarkovConsumerType(**init_serial_growth)
#    SerialGroExample.assignParameters(Rfree = np.array(np.array(StateCount*[1.03])),    # Same interest factor in each Markov state
#                                   PermGroFac = [np.array([0.97,0.99,1.01,1.03,1.05])], # Different permanent growth factor in each Markov state
#                                   LivPrb = [np.array(StateCount*[0.98])],              # Same survival probability in all states
#                                   cycles = 0)
       
#    
#    # Solve the serially correlated permanent growth shock problem and display the consumption functions
#    start_time = clock()
#    SerialGroExample.solve()
#    end_time = clock()
#    print('Solving a serially correlated growth consumer took ' + mystr(end_time-start_time) + ' seconds.')
#    print('Consumption functions for each discrete state:')
#    plotFuncs(SerialGroExample.solution[0].cFunc,0,10)
#
#
#
#
#DONT INVOKE makeMrkvHist, just set it to whatever we want (1000 periods of no growth, then change)
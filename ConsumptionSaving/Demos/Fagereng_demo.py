'''
This module runs a quick and dirty structural estimation based on Table 9 of 
"MPC Heterogeneity and Household Balance Sheets" by Fagereng, Holm, and Natvik.
Authors use Norweigian administrative data on income, household assets, and lottery
winnings to examine the MPC from transitory income shocks (lottery prizes).  In
Table 9, they report estimated MPC broken down by quartiles of bank deposits and
prize size; this table is reproduced here as MPC_target_base.  In this demo, we
use the Table 9 estimates as targets in a simple structural estimation, seeking
to minimize the sum of squared differences between simulated and estimated MPCs
by changing the (uniform) distribution of discount factors.  Can their results
be rationalized by a simple one-asset consumption-saving model?  This module
includes several options for estimating different specifications:
    
TypeCount : Integer number of discount factors in discrete distribution; can be
            set to 1 to turn off ex ante heterogeneity.
AdjFactor : Scaling factor for the target MPCs; user can try to fit estimated
            MPCs scaled down by (e.g.) 50%.
T_kill    : Maximum number of years the (perpetually young) agents are allowed
            to live.  Because this is quick and dirty, it's also the number of
            periods to simulate.
Splurge   : Amount of lottery prize that an individual will automatically spend
            in a moment of irrational excitement, before coming to his senses
            and behaving according to his consumption function.  The patterns in
            Table 9 can be fit much better when this is set around $700 --> 0.7.
do_secant : Boolean indicator for whether to use "secant MPC", which is average
            MPC over the range of the prize.  MNW believes authors' regressions
            are estimating this rather than point MPC.  When False, structural
            estimation uses point MPC after receiving prize.  NB: This is incom-
            patible with Splurge > 0.
drop_corner : Boolean for whether to include target MPC in the top left corner,
              which is greater than 1.  Authors discuss reasons why the MPC
              from a transitory shock *could* exceed 1.  Option is included here
              because this target tends to push the estimate around a bit.
'''
from __future__ import division, print_function
from builtins import str
from builtins import range
import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../cstwMPC'))

import numpy as np
from copy import deepcopy
from time import clock

from HARKutilities import approxUniform, getPercentiles
from HARKparallel import multiThreadCommands
from HARKestimation import minimizeNelderMead
from ConsIndShockModel import IndShockConsumerType
from SetupParamsCSTW import init_infinite # dictionary with most ConsumerType parameters

TypeCount = 8    # Number of consumer types with heterogeneous discount factors
AdjFactor = 1.0  # Factor by which to scale all of Fagereng's MPCs in Table 9
T_kill = 100     # Don't let agents live past this age
Splurge = 0.0    # Consumers automatically spend this amount of any lottery prize
do_secant = True # If True, calculate MPC by secant, else point MPC
drop_corner = False # If True, ignore upper left corner when calculating distance

# Define the MPC targets from Table 9; element i,j is lottery quartile i, deposit quartile j
MPC_target_base = np.array([[1.047, 0.745, 0.720, 0.490],
                            [0.762, 0.640, 0.559, 0.437],
                            [0.663, 0.546, 0.390, 0.386],
                            [0.354, 0.325, 0.242, 0.216]])
MPC_target = AdjFactor*MPC_target_base

# Define the four lottery sizes, in thousands of USD; these are eyeballed centers/averages
lottery_size = np.array([1.625, 3.3741, 7.129, 40.0])

# Make an initialization dictionary on an annual basis
base_params = deepcopy(init_infinite)
base_params['LivPrb'] = [0.975]
base_params['Rfree'] = 1.04/base_params['LivPrb'][0]
base_params['PermShkStd'] = [0.1]
base_params['TranShkStd'] = [0.1]
base_params['T_age'] = T_kill # Kill off agents if they manage to achieve T_kill working years
base_params['AgentCount'] = 10000
base_params['pLvlInitMean'] = np.log(23.72) # From Table 1, in thousands of USD
base_params['T_sim'] = T_kill  # No point simulating past when agents would be killed off

# Make several consumer types to be used during estimation
BaseType = IndShockConsumerType(**base_params)
EstTypeList = []
for j in range(TypeCount):
    EstTypeList.append(deepcopy(BaseType))
    EstTypeList[-1](seed = j)
    
# Define the objective function
def FagerengObjFunc(center,spread,verbose=False):
    '''
    Objective function for the quick and dirty structural estimation to fit
    Fagereng, Holm, and Natvik's Table 9 results with a basic infinite horizon
    consumption-saving model (with permanent and transitory income shocks).
    
    Parameters
    ----------
    center : float
        Center of the uniform distribution of discount factors.
    spread : float
        Width of the uniform distribution of discount factors.
    verbose : bool
        When True, print to screen MPC table for these parameters.  When False,
        print (center, spread, distance).
        
    Returns
    -------
    distance : float
        Euclidean distance between simulated MPCs and (adjusted) Table 9 MPCs.
    '''
    # Give our consumer types the requested discount factor distribution
    beta_set = approxUniform(N=TypeCount,bot=center-spread,top=center+spread)[1]
    for j in range(TypeCount):
        EstTypeList[j](DiscFac = beta_set[j])
        
    # Solve and simulate all consumer types, then gather their wealth levels
    multiThreadCommands(EstTypeList,['solve()','initializeSim()','simulate()','unpackcFunc()'])
    WealthNow = np.concatenate([ThisType.aLvlNow for ThisType in EstTypeList])
    
    # Get wealth quartile cutoffs and distribute them to each consumer type
    quartile_cuts = getPercentiles(WealthNow,percentiles=[0.25,0.50,0.75])
    for ThisType in EstTypeList:
        WealthQ = np.zeros(ThisType.AgentCount,dtype=int)
        for n in range(3):
            WealthQ[ThisType.aLvlNow > quartile_cuts[n]] += 1
        ThisType(WealthQ = WealthQ)
        
    # Keep track of MPC sets in lists of lists of arrays
    MPC_set_list = [ [[],[],[],[]],
                     [[],[],[],[]],
                     [[],[],[],[]],
                     [[],[],[],[]] ]
    
    # Calculate the MPC for each of the four lottery sizes for all agents
    for ThisType in EstTypeList:
        ThisType.simulate(1)
        c_base = ThisType.cNrmNow
        MPC_this_type = np.zeros((ThisType.AgentCount,4))
        for k in range(4): # Get MPC for all agents of this type
            Llvl = lottery_size[k]
            Lnrm = Llvl/ThisType.pLvlNow
            if do_secant:
                SplurgeNrm = Splurge/ThisType.pLvlNow
                mAdj = ThisType.mNrmNow + Lnrm - SplurgeNrm
                cAdj = ThisType.cFunc[0](mAdj) + SplurgeNrm
                MPC_this_type[:,k] = (cAdj - c_base)/Lnrm
            else:
                mAdj = ThisType.mNrmNow + Lnrm
                MPC_this_type[:,k] = cAdj = ThisType.cFunc[0].derivative(mAdj)
                        
        # Sort the MPCs into the proper MPC sets
        for q in range(4):
            these = ThisType.WealthQ == q
            for k in range(4):
                MPC_set_list[k][q].append(MPC_this_type[these,k])
                
    # Calculate average within each MPC set
    simulated_MPC_means = np.zeros((4,4))
    for k in range(4):
        for q in range(4):
            MPC_array = np.concatenate(MPC_set_list[k][q])
            simulated_MPC_means[k,q] = np.mean(MPC_array)
            
    # Calculate Euclidean distance between simulated MPC averages and Table 9 targets
    diff = simulated_MPC_means - MPC_target
    if drop_corner:
        diff[0,0] = 0.0
    distance = np.sqrt(np.sum((diff)**2))
    if verbose:
        print(simulated_MPC_means)
    else:
        print (center, spread, distance)
    return distance


if __name__ == '__main__':
    
    guess = [0.92,0.03]
    f_temp = lambda x : FagerengObjFunc(x[0],x[1])
    opt_params = minimizeNelderMead(f_temp, guess, verbose=True)
    print('Finished estimating for scaling factor of ' + str(AdjFactor) + ' and "splurge amount" of $' + str(1000*Splurge))
    print('Optimal (beta,nabla) is ' + str(opt_params) + ', simulated MPCs are:')
    dist = FagerengObjFunc(opt_params[0],opt_params[1],True)
    print('Distance from Fagereng et al Table 9 is ' + str(dist))
    
#    t_start = clock()
#    X = FagerengObjFunc(0.814,0.122)
#    t_end = clock()
#    print('That took ' + str(t_end - t_start) + ' seconds.')
#    print(X)
    

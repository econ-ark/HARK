# Import the HARK library.  The assumption is that this code is in a folder
# contained in the HARK folder. 
import sys 
sys.path.insert(0,'../')

import SetupConsumerParameters as Params
import ConsumptionSavingModel as Model
from HARKutilities import plotFunc, plotFuncDer, plotFuncs, calculateMeanOneLognormalDiscreteApprox, createFlatStateSpaceFromIndepDiscreteProbs
from time import clock
from copy import deepcopy
import numpy as np
mystr = lambda number : "{:.4f}".format(number) 

do_hybrid_type = False
do_markov_type = True

# Make and solve a finite consumer type
LifecycleType = Model.ConsumerType(**Params.init_consumer_objects)

start_time = clock()
LifecycleType.solve()
end_time = clock()
print('Solving a lifecycle consumer took ' + mystr(end_time-start_time) + ' seconds.')
LifecycleType.unpack_cFunc()
LifecycleType.timeFwd()

# Plot the consumption functions during working life
print('Consumption functions while working:')
plotFuncs(LifecycleType.cFunc[:40],0,5)
# Plot the consumption functions during retirement
print('Consumption functions while retired:')
plotFuncs(LifecycleType.cFunc[40:],0,5)
LifecycleType.timeRev()



# Make and solve an infinite horizon consumer
InfiniteType = deepcopy(LifecycleType)
InfiniteType.assignParameters(    survival_prob = [0.98],
                                  beta = [0.96],
                                  Gamma = [1.01],
                                  cycles = 0) # This is what makes the type infinite horizon
InfiniteType.income_distrib = [LifecycleType.income_distrib[-1]]
InfiniteType.p_zero_income = [LifecycleType.p_zero_income[-1]]

start_time = clock()
InfiniteType.solve()
end_time = clock()
print('Solving an infinite horizon consumer took ' + mystr(end_time-start_time) + ' seconds.')
InfiniteType.unpack_cFunc()

# Plot the consumption function and MPC for the infinite horizon consumer
print('Consumption function:')
plotFunc(InfiniteType.cFunc[0],0,50)    # plot consumption
print('Marginal consumption function:')
plotFuncDer(InfiniteType.cFunc[0],0,5) # plot MPC
if InfiniteType.calc_vFunc:
    print('Value function:')
    plotFunc(InfiniteType.solution[0].vFunc,0.5,10)


# Make and solve a "cyclical" consumer type who lives the same four quarters repeatedly.
# The consumer has income that greatly fluctuates throughout the year.
CyclicalType = deepcopy(LifecycleType)
CyclicalType.assignParameters(survival_prob = [0.98]*4,
                                  beta = [0.96]*4,
                                  Gamma = [1.1, 0.3, 2.8, 1.1],
                                  cycles = 0) # This is what makes the type (cyclically) infinite horizon)
CyclicalType.income_distrib = [LifecycleType.income_distrib[-1]]*4
CyclicalType.p_zero_income = [LifecycleType.p_zero_income[-1]]*4

start_time = clock()
CyclicalType.solve()
end_time = clock()
print('Solving a cyclical consumer took ' + mystr(end_time-start_time) + ' seconds.')
CyclicalType.unpack_cFunc()
CyclicalType.timeFwd()

# Plot the consumption functions for the cyclical consumer type
print('Quarterly consumption functions:')
plotFuncs(CyclicalType.cFunc,0,5)



# Make and solve a "hybrid" consumer who solves an infinite horizon problem by
# alternating between ENDG and EXOG each period.  Yes, this is weird.
if do_hybrid_type:
    HybridType = deepcopy(InfiniteType)
    HybridType.assignParameters(survival_prob = 2*[0.98],
                                  beta = 2*[0.96],
                                  Gamma = 2*[1.01])
    HybridType.income_distrib = 2*[LifecycleType.income_distrib[-1]]
    HybridType.p_zero_income = 2*[LifecycleType.p_zero_income[-1]]
    HybridType.time_vary.append('solveAPeriod')
    HybridType.solveAPeriod = [Model.consumptionSavingSolverENDG,Model.consumptionSavingSolverEXOG] # alternated between ENDG and EXOG
    
    start_time = clock()
    HybridType.solve()
    end_time = clock()
    print('Solving a "hybrid" consumer took ' + mystr(end_time-start_time) + ' seconds.')
    HybridType.unpack_cFunc()
    
    # Plot the consumption function for the cyclical consumer type
    print('"Hybrid solver" consumption function:')
    plotFunc(HybridType.cFunc[0],0,5)
    

# Make and solve a type that has serially correlated unemployment   
if do_markov_type:
    # Define the Markov transition matrix
    unemp_length = 5
    urate_good = 0.05
    urate_bad = 0.12
    bust_prob = 0.01
    recession_length = 20
    p_reemploy =1.0/unemp_length
    p_unemploy_good = p_reemploy*urate_good/(1-urate_good)
    p_unemploy_bad = p_reemploy*urate_bad/(1-urate_bad)
    boom_prob = 1.0/recession_length
    transition_matrix = np.array([[(1-p_unemploy_good)*(1-bust_prob),p_unemploy_good*(1-bust_prob),(1-p_unemploy_good)*bust_prob,p_unemploy_good*bust_prob],
                                  [p_reemploy*(1-bust_prob),(1-p_reemploy)*(1-bust_prob),p_reemploy*bust_prob,(1-p_reemploy)*bust_prob],
                                  [(1-p_unemploy_bad)*boom_prob,p_unemploy_bad*boom_prob,(1-p_unemploy_bad)*(1-boom_prob),p_unemploy_bad*(1-boom_prob)],
                                  [p_reemploy*boom_prob,(1-p_reemploy)*boom_prob,p_reemploy*(1-boom_prob),(1-p_reemploy)*(1-boom_prob)]])
    
    MarkovType = deepcopy(InfiniteType)
    xi_dist = calculateMeanOneLognormalDiscreteApprox(MarkovType.xi_N, 0.1)
    psi_dist = calculateMeanOneLognormalDiscreteApprox(MarkovType.psi_N, 0.1)
    employed_income_dist = createFlatStateSpaceFromIndepDiscreteProbs(psi_dist, xi_dist)
    employed_income_dist = [np.ones(1),np.ones(1),np.ones(1)]
    unemployed_income_dist = [np.ones(1),np.ones(1),np.zeros(1)]
    p_zero_income = [np.array([0.0,1.0,0.0,1.0])]
    
    MarkovType.solution_terminal.cFunc = 4*[MarkovType.solution_terminal.cFunc]
    MarkovType.solution_terminal.vFunc = 4*[MarkovType.solution_terminal.vFunc]
    MarkovType.solution_terminal.vPfunc = 4*[MarkovType.solution_terminal.vPfunc]
    MarkovType.solution_terminal.vPPfunc = 4*[MarkovType.solution_terminal.vPPfunc]
    MarkovType.solution_terminal.m_underbar = 4*[MarkovType.solution_terminal.m_underbar]
    
    MarkovType.income_distrib = [[employed_income_dist,unemployed_income_dist,employed_income_dist,unemployed_income_dist]]
    MarkovType.p_zero_income = p_zero_income
    MarkovType.transition_matrix = transition_matrix
    MarkovType.time_inv.append('transition_matrix')
    MarkovType.solveAPeriod = Model.consumptionSavingSolverMarkov
    MarkovType.cycles = 0
    
    MarkovType.timeFwd()
    start_time = clock()
    MarkovType.solve()
    end_time = clock()
    print('Solving a Markov consumer took ' + mystr(end_time-start_time) + ' seconds.')
    print('Consumption functions for each discrete state:')
    plotFuncs(MarkovType.solution[0].cFunc,0,50)
    
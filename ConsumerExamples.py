import SetupConsumerParameters as Params
import ConsumptionSavingModel as Model
from HARKutilities import plotFunc, plotFuncDer, plotFuncs
from time import clock
from copy import deepcopy
mystr = lambda number : "{:.4f}".format(number) 

do_hybrid_type = False

# Make and solve a finite consumer type
LifecycleType = Model.ConsumerType(**Params.init_consumer_objects)
#scriptR_shocks, xi_shocks = Model.generateIncomeShockHistoryLognormalUnemployment(LifecycleType)
#LifecycleType.addIncomeShockPaths(scriptR_shocks,xi_shocks)

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
plotFunc(InfiniteType.cFunc[0],0,5)    # plot consumption
print('Marginal consumption function:')
plotFuncDer(InfiniteType.cFunc[0],0,5) # plot MPC
if InfiniteType.calc_vFunc:
    print('Value function:')
    plotFunc(InfiniteType.solution[0].vFunc,0.2,5)


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

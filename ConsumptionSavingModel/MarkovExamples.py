import numpy as np
from ConsumptionSavingModel import ConsumerType, consumptionSavingSolverMarkov
import SetupConsumerParameters as Params
from HARKutilities import plotFunc, plotFuncDer, plotFuncs, approxMeanOneLognormal, addDiscreteOutcomeConstantMean
from time import clock
from copy import deepcopy
mystr = lambda number : "{:.4f}".format(number)

###############################################################################

# Make a consumer who occasionally gets "unemployment immunity" for a fixed period
UnempPrb    = 0.05
ImmunityPrb = 0.01
ImmunityT   = 6

StateCount = ImmunityT+1
IncomeDistReg = [np.array([1-UnempPrb,UnempPrb]), np.array([1.0,1.0]), np.array([1.0/(1.0-UnempPrb),0.0])]
IncomeDistImm = [np.array([1.0]), np.array([1.0]), np.array([1.0])]
IncomeDist = [IncomeDistReg] + ImmunityT*[IncomeDistImm]

transition_array = np.zeros((StateCount,StateCount))
transition_array[0,0] = 1.0 - ImmunityPrb
transition_array[0,ImmunityT] = ImmunityPrb
for j in range(ImmunityT):
    transition_array[j+1,j] = 1.0

ImmunityType = ConsumerType(**Params.init_consumer_objects)
ImmunityType.assignParameters(LivFac = [0.98],
                              DiscFac = [0.96],
                              Rfree = np.array(np.array(StateCount*[1.03])),
                              PermGroFac = [np.array(StateCount*[1.01])],
                              BoroCnst = None,
                              cycles = 0)
ImmunityType.IncomeDist = [IncomeDist]
ImmunityType.transition_array = transition_array
ImmunityType.solveOnePeriod = consumptionSavingSolverMarkov
ImmunityType.time_inv.append('transition_array')

ImmunityType.solution_terminal.cFunc = StateCount*[ImmunityType.solution_terminal.cFunc]
ImmunityType.solution_terminal.vFunc = StateCount*[ImmunityType.solution_terminal.vFunc]
ImmunityType.solution_terminal.vPfunc = StateCount*[ImmunityType.solution_terminal.vPfunc]
ImmunityType.solution_terminal.vPPfunc = StateCount*[ImmunityType.solution_terminal.vPPfunc]
ImmunityType.solution_terminal.mRtoMin = StateCount*[ImmunityType.solution_terminal.mRtoMin]
ImmunityType.solution_terminal.MPCmax = np.array(StateCount*[1.0])
ImmunityType.solution_terminal.MPCmin = np.array(StateCount*[1.0])

start_time = clock()
ImmunityType.solve()
end_time = clock()
print('Solving an "unemployment immunity" consumer took ' + mystr(end_time-start_time) + ' seconds.')
print('Consumption functions for each discrete state:')
plotFuncs(ImmunityType.solution[0].cFunc,0,10)


###############################################################################

# Make a consumer with serially correlated permanent income growth
UnempPrb = 0.05
StateCount = 5
Persistence = 0.5

IncomeDistReg = [np.array([1-UnempPrb,UnempPrb]), np.array([1.0,1.0]), np.array([1.0,0.0])]
IncomeDist = StateCount*[IncomeDistReg]

transition_array = Persistence*np.eye(StateCount) + (1.0/StateCount)*(1.0-Persistence)*np.ones((StateCount,StateCount))

SerialGroType = ConsumerType(**Params.init_consumer_objects)
SerialGroType.assignParameters(LivFac = [0.99375],
                              DiscFac = [0.995],
                              Rfree = np.array(np.array(StateCount*[1.03])),
                              PermGroFac = [np.array([0.97,0.99,1.01,1.03,1.05])],
                              BoroCnst = None,
                              cycles = 0)
SerialGroType.IncomeDist = [IncomeDist]
SerialGroType.transition_array = transition_array
SerialGroType.solveOnePeriod = consumptionSavingSolverMarkov
SerialGroType.time_inv.append('transition_array')

SerialGroType.solution_terminal.cFunc = StateCount*[SerialGroType.solution_terminal.cFunc]
SerialGroType.solution_terminal.vFunc = StateCount*[SerialGroType.solution_terminal.vFunc]
SerialGroType.solution_terminal.vPfunc = StateCount*[SerialGroType.solution_terminal.vPfunc]
SerialGroType.solution_terminal.vPPfunc = StateCount*[SerialGroType.solution_terminal.vPPfunc]
SerialGroType.solution_terminal.mRtoMin = StateCount*[SerialGroType.solution_terminal.mRtoMin]
SerialGroType.solution_terminal.MPCmax = np.array(StateCount*[1.0])
SerialGroType.solution_terminal.MPCmin = np.array(StateCount*[1.0])

start_time = clock()
SerialGroType.solve()
end_time = clock()
print('Solving a serially correlated growth consumer took ' + mystr(end_time-start_time) + ' seconds.')
print('Consumption functions for each discrete state:')
plotFuncs(SerialGroType.solution[0].cFunc,0,10)

###############################################################################

# Make a consumer with serially correlated interest factors
SerialRType = deepcopy(SerialGroType)
SerialRType.assignParameters(PermGroFac = [np.array(StateCount*[1.01])],
                             Rfree = np.array([1.01,1.02,1.03,1.04,1.05]))

start_time = clock()
SerialRType.solve()
end_time = clock()
print('Solving a serially correlated interest consumer took ' + mystr(end_time-start_time) + ' seconds.')
print('Consumption functions for each discrete state:')
plotFuncs(SerialRType.solution[0].cFunc,0,10)

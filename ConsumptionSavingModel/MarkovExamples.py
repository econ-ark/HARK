import numpy as np
from ConsumptionSavingModel import ConsumerType, consumptionSavingSolverMarkov
import SetupConsumerParameters as Params
from HARKutilities import plotFunc, plotFuncDer, plotFuncs
from time import clock
from copy import deepcopy
mystr = lambda number : "{:.4f}".format(number)

###############################################################################

# Make a consumer who occasionally gets "unemployment immunity" for a fixed period
UnempPrb    = 0.05
ImmunityPrb = 0.01
ImmunityT   = 6

StateCount = ImmunityT+1
IncomeDstnReg = [np.array([1-UnempPrb,UnempPrb]), np.array([1.0,1.0]), np.array([1.0/(1.0-UnempPrb),0.0])]
IncomeDstnImm = [np.array([1.0]), np.array([1.0]), np.array([1.0])]
IncomeDstn = [IncomeDstnReg] + ImmunityT*[IncomeDstnImm]

MrkvArray = np.zeros((StateCount,StateCount))
MrkvArray[0,0] = 1.0 - ImmunityPrb
MrkvArray[0,ImmunityT] = ImmunityPrb
for j in range(ImmunityT):
    MrkvArray[j+1,j] = 1.0

ImmunityType = ConsumerType(**Params.init_consumer_objects)
ImmunityType.assignParameters(LivPrb = [0.98],
                              DiscFac = [0.96],
                              Rfree = np.array(np.array(StateCount*[1.03])),
                              PermGroFac = [np.array(StateCount*[1.01])],
                              BoroCnstArt = None,
                              cycles = 0)
ImmunityType.IncomeDstn = [IncomeDstn]
ImmunityType.MrkvArray = MrkvArray
ImmunityType.solveOnePeriod = consumptionSavingSolverMarkov
ImmunityType.time_inv.append('MrkvArray')

ImmunityType.solution_terminal.cFunc = StateCount*[ImmunityType.solution_terminal.cFunc]
ImmunityType.solution_terminal.vFunc = StateCount*[ImmunityType.solution_terminal.vFunc]
ImmunityType.solution_terminal.vPfunc = StateCount*[ImmunityType.solution_terminal.vPfunc]
ImmunityType.solution_terminal.vPPfunc = StateCount*[ImmunityType.solution_terminal.vPPfunc]
ImmunityType.solution_terminal.mNrmMin = StateCount*[ImmunityType.solution_terminal.mNrmMin]
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

IncomeDstnReg = [np.array([1-UnempPrb,UnempPrb]), np.array([1.0,1.0]), np.array([1.0,0.0])]
IncomeDstn = StateCount*[IncomeDstnReg]

MrkvArray = Persistence*np.eye(StateCount) + (1.0/StateCount)*(1.0-Persistence)*np.ones((StateCount,StateCount))

SerialGroType = ConsumerType(**Params.init_consumer_objects)
SerialGroType.assignParameters(LivPrb = [0.99375],
                              DiscFac = [0.995],
                              Rfree = np.array(np.array(StateCount*[1.03])),
                              PermGroFac = [np.array([0.97,0.99,1.01,1.03,1.05])],
                              BoroCnstArt = None,
                              cycles = 0)
SerialGroType.IncomeDstn = [IncomeDstn]
SerialGroType.MrkvArray = MrkvArray
SerialGroType.solveOnePeriod = consumptionSavingSolverMarkov
SerialGroType.time_inv.append('MrkvArray')

SerialGroType.solution_terminal.cFunc = StateCount*[SerialGroType.solution_terminal.cFunc]
SerialGroType.solution_terminal.vFunc = StateCount*[SerialGroType.solution_terminal.vFunc]
SerialGroType.solution_terminal.vPfunc = StateCount*[SerialGroType.solution_terminal.vPfunc]
SerialGroType.solution_terminal.vPPfunc = StateCount*[SerialGroType.solution_terminal.vPPfunc]
SerialGroType.solution_terminal.mNrmMin = StateCount*[SerialGroType.solution_terminal.mNrmMin]
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

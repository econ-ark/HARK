'''
This module demonstrates several examples of the types of models that can be
solved with the Markov extension to the standard consumption-saving model.
'''
import numpy as np   # numeric Python
from ConsumptionSavingModel import ConsumerType, solveConsumptionSavingMarkov # consumer class and one period solver
import SetupConsumerParameters as Params # basic parameters for the problem
from HARKutilities import plotFunc, plotFuncDer, plotFuncs # basic plotting tools
from time import clock    # timing utility
from copy import deepcopy # "deep" copying of complex objects
mystr = lambda number : "{:.4f}".format(number) # formatting numbers as strings

###############################################################################

# Make a consumer who occasionally gets "unemployment immunity" for a fixed period
UnempPrb    = 0.05  # Probability of becoming unemployed each period
ImmunityPrb = 0.01  # Probability of becoming "immune" to unemployment
ImmunityT   = 6     # Number of periods of immunity

StateCount = ImmunityT+1   # Total number of Markov states
IncomeDstnReg = [np.array([1-UnempPrb,UnempPrb]), np.array([1.0,1.0]), np.array([1.0/(1.0-UnempPrb),0.0])] # Ordinary income distribution
IncomeDstnImm = [np.array([1.0]), np.array([1.0]), np.array([1.0])] # Income distribution when unemployed
IncomeDstn = [IncomeDstnReg] + ImmunityT*[IncomeDstnImm] # Income distribution for each Markov state, in a list

# Make the Markov transition array.  MrkvArray[i,j] is the probability of transitioning
# to state j in period t+1 from state i in period t.
MrkvArray = np.zeros((StateCount,StateCount))
MrkvArray[0,0] = 1.0 - ImmunityPrb   # Probability of not becoming immune in ordinary state: stay in ordinary state
MrkvArray[0,ImmunityT] = ImmunityPrb # Probability of becoming immune in ordinary state: begin immunity periods
for j in range(ImmunityT):
    MrkvArray[j+1,j] = 1.0  # When immune, have 100% chance of transition to state with one fewer immunity periods remaining

ImmunityType = ConsumerType(**Params.init_consumer_objects) # Make a basic consumer type
ImmunityType.assignParameters(LivPrb = [0.98],              # Replace with "one period" infinite horizon data
                              DiscFac = [0.96],
                              Rfree = np.array(np.array(StateCount*[1.03])), # Interest factor same in all states
                              PermGroFac = [np.array(StateCount*[1.01])],    # Permanent growth factor same in all states
                              BoroCnstArt = None,                            # No artificial borrowing constraint
                              cycles = 0)                                    # Infinite horizon
ImmunityType.IncomeDstn = [IncomeDstn]
ImmunityType.MrkvArray = MrkvArray
ImmunityType.solveOnePeriod = solveConsumptionSavingMarkov # set appropriate one period solver
ImmunityType.time_inv.append('MrkvArray')                  # add the Markov array to time-invariant solution inputs

# Update the terminal period solution
ImmunityType.solution_terminal.cFunc = StateCount*[ImmunityType.solution_terminal.cFunc]
ImmunityType.solution_terminal.vFunc = StateCount*[ImmunityType.solution_terminal.vFunc]
ImmunityType.solution_terminal.vPfunc = StateCount*[ImmunityType.solution_terminal.vPfunc]
ImmunityType.solution_terminal.vPPfunc = StateCount*[ImmunityType.solution_terminal.vPPfunc]
ImmunityType.solution_terminal.mNrmMin = StateCount*[ImmunityType.solution_terminal.mNrmMin]
ImmunityType.solution_terminal.MPCmax = np.array(StateCount*[1.0])
ImmunityType.solution_terminal.MPCmin = np.array(StateCount*[1.0])

# Solve the unemployment immunity problem and display the consumption functions
start_time = clock()
ImmunityType.solve()
end_time = clock()
print('Solving an "unemployment immunity" consumer took ' + mystr(end_time-start_time) + ' seconds.')
print('Consumption functions for each discrete state:')
mNrmMin = np.min([ImmunityType.solution[0].mNrmMin[j] for j in range(StateCount)])
plotFuncs(ImmunityType.solution[0].cFunc,mNrmMin,10)


###############################################################################

# Make a consumer with serially correlated permanent income growth
UnempPrb = 0.05    # Unemployment probability
StateCount = 5     # Number of permanent income growth rates
Persistence = 0.5  # Probability of getting the same permanent income growth rate next period

IncomeDstnReg = [np.array([1-UnempPrb,UnempPrb]), np.array([1.0,1.0]), np.array([1.0,0.0])]
IncomeDstn = StateCount*[IncomeDstnReg] # Same simple income distribution in each state

# Make the state transition array for this type: Persistence probability of remaining in the same state, equiprobable otherwise
MrkvArray = Persistence*np.eye(StateCount) + (1.0/StateCount)*(1.0-Persistence)*np.ones((StateCount,StateCount))

SerialGroType = ConsumerType(**Params.init_consumer_objects) # Make a basic type
SerialGroType.assignParameters(LivPrb = [0.99375],
                              DiscFac = [0.995],
                              Rfree = np.array(np.array(StateCount*[1.03])),       # Same interest factor in each Markov state
                              PermGroFac = [np.array([0.97,0.99,1.01,1.03,1.05])], # Different permanent growth factor in each Markov state
                              BoroCnstArt = None,
                              cycles = 0)
SerialGroType.IncomeDstn = [IncomeDstn]
SerialGroType.MrkvArray = MrkvArray
SerialGroType.solveOnePeriod = solveConsumptionSavingMarkov # set appropriate one period solver                  
SerialGroType.time_inv.append('MrkvArray')                  # add the Markov array to time-invariant solution inputs

# Update the terminal period solution
SerialGroType.solution_terminal.cFunc = StateCount*[SerialGroType.solution_terminal.cFunc]
SerialGroType.solution_terminal.vFunc = StateCount*[SerialGroType.solution_terminal.vFunc]
SerialGroType.solution_terminal.vPfunc = StateCount*[SerialGroType.solution_terminal.vPfunc]
SerialGroType.solution_terminal.vPPfunc = StateCount*[SerialGroType.solution_terminal.vPPfunc]
SerialGroType.solution_terminal.mNrmMin = StateCount*[SerialGroType.solution_terminal.mNrmMin]
SerialGroType.solution_terminal.MPCmax = np.array(StateCount*[1.0])
SerialGroType.solution_terminal.MPCmin = np.array(StateCount*[1.0])

# Solve the serially correlated permanent growth shock problem and display the consumption functions
start_time = clock()
SerialGroType.solve()
end_time = clock()
print('Solving a serially correlated growth consumer took ' + mystr(end_time-start_time) + ' seconds.')
print('Consumption functions for each discrete state:')
plotFuncs(SerialGroType.solution[0].cFunc,0,10)

###############################################################################

# Make a consumer with serially correlated interest factors
SerialRType = deepcopy(SerialGroType) # Same as the last problem...
SerialRType.assignParameters(PermGroFac = [np.array(StateCount*[1.01])],   # ...but now the permanent growth factor is constant...
                             Rfree = np.array([1.01,1.02,1.03,1.04,1.05])) # ...and the interest factor is what varies across states

# Solve the serially correlated interest rate problem and display the consumption functions
start_time = clock()
SerialRType.solve()
end_time = clock()
print('Solving a serially correlated interest consumer took ' + mystr(end_time-start_time) + ' seconds.')
print('Consumption functions for each discrete state:')
plotFuncs(SerialRType.solution[0].cFunc,0,10)

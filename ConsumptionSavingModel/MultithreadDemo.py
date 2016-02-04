'''
This is a brief demonstration of parallel processing in HARK using HARKparallel.
A benchmark consumption-saving model is solved for individuals whose CRRA varies
between 1 and 8.  The infinite horizon model is solved serially and then in
parallel.  Note that HARKparallel will not work "out of the box", as Anaconda
does not include two packages needed for it.  See HARKparallel.py.
'''

import SetupConsumerParameters as Params
import ConsumptionSavingModel as Model
from HARKutilities import plotFunc, plotFuncDer, plotFuncs
from time import clock
from copy import deepcopy
from HARKparallel import multiThreadCommandsFake, multiThreadCommands
mystr = lambda number : "{:.4f}".format(number)
import numpy as np


if __name__ == '__main__':
    type_count = 32    
    
    # Make the basic type that we'll use as a template.
    # The basic type has an artificially dense assets grid, as the problem to be
    # solved must be sufficiently large for multithreading to be faster than
    # single-threading (looping), due to overhead.
    BasicType = Model.ConsumerType(**Params.init_consumer_objects)
    BasicType.a_max = 100
    BasicType.a_size = 64
    BasicType.updateAssetsGrid()
    BasicType.timeRev()
    BasicType.assignParameters(       survival_prob = [0.98],
                                      beta = [0.96],
                                      Gamma = [1.01],
                                      cycles = 0) # This is what makes the type infinite horizon
    BasicType(calc_vFunc = True)
    BasicType.income_distrib = [BasicType.income_distrib[-1]]
    BasicType.p_zero_income = [BasicType.p_zero_income[-1]]
    
    # Solve it and plot the results, to make sure things are working
    start_time = clock()
    BasicType.solve()
    BasicType.unpack_cFunc()
    end_time = clock()
    print('Solving the basic consumer took ' + mystr(end_time-start_time) + ' seconds.')
    BasicType.unpack_cFunc()
    print('Consumption function:')
    plotFunc(BasicType.cFunc[0],0,5)    # plot consumption
    print('Marginal consumption function:')
    plotFuncDer(BasicType.cFunc[0],0,5) # plot MPC
    if BasicType.calc_vFunc:
        print('Value function:')
        plotFunc(BasicType.solution[0].vFunc,0.2,5)
    
    # Make copies of the basic type, each with a different risk aversion
    BasicType.calc_vFunc = False
    my_agent_list = []
    #rho_list = np.random.permutation(np.linspace(1,8,type_count))
    rho_list = np.linspace(1,8,type_count)
    for i in range(type_count):
        this_agent = deepcopy(BasicType)
        this_agent.assignParameters(rho = rho_list[i])
        my_agent_list.append(this_agent)
    do_this_stuff = ['updateSolutionTerminal()','solve()','unpack_cFunc()']
    
    # Solve the model for each type by looping over the types (not multithreading)
    start_time = clock()
    multiThreadCommandsFake(my_agent_list, do_this_stuff)
    end_time = clock()
    print('Solving ' + str(type_count) +  ' types without multithreading took ' + mystr(end_time-start_time) + ' seconds.')
    
    # Plot the consumption functions for all types on one figure
    plotFuncs([this_type.cFunc[0] for this_type in my_agent_list],0,5)
    
    # Delete the solution for each type to make sure we're not just faking it
    for i in range(type_count):
        my_agent_list[i].solution = None
        my_agent_list[i].cFunc = None
    
    # And here's my shitty, shitty attempt at multithreading:
    start_time = clock()
    multiThreadCommands(my_agent_list, do_this_stuff)
    end_time = clock()
    print('Solving ' + str(type_count) +  ' types with multithreading took ' + mystr(end_time-start_time) + ' seconds.')
    
    # Plot the consumption functions for all types on one figure to see if it worked
    plotFuncs([this_type.cFunc[0] for this_type in my_agent_list],0,5)

'''
A demonstration of parallel processing in HARK using HARKparallel.
A benchmark consumption-saving model is solved for individuals whose CRRA varies
between 1 and 8.  The infinite horizon model is solved serially and then in
parallel.  Note that HARKparallel will not work "out of the box", as Anaconda
does not include two packages needed for it; see HARKparallel.py.  When given a
sufficiently large amount of work for each thread to do, the maximum speedup
factor seems to be around P/2, where P is the number of processors.
'''
import sys 
sys.path.insert(0,'../')

import ConsumerParameters as Params       # Parameters for a consumer type
import ConsIndShockModel as Model         # Consumption-saving model with idiosyncratic shocks
from HARKutilities import plotFuncs, plotFuncsDer # Basic plotting tools
from time import clock                         # Timing utility
from copy import deepcopy                      # "Deep" copying for complex objects
from HARKparallel import multiThreadCommandsFake, multiThreadCommands # Parallel processing
mystr = lambda number : "{:.4f}".format(number)# Format numbers as strings
import numpy as np                             # Numeric Python

if __name__ == '__main__': # Parallel calls *must* be inside a call to __main__
    type_count = 32    # Number of values of CRRA to solve
    
    # Make the basic type that we'll use as a template.
    # The basic type has an artificially dense assets grid, as the problem to be
    # solved must be sufficiently large for multithreading to be faster than
    # single-threading (looping), due to overhead.
    BasicType = Model.IndShockConsumerType(**Params.init_idiosyncratic_shocks)
    BasicType.cycles = 0
    BasicType(aXtraMax  = 100, aXtraCount = 64)
    BasicType(vFuncBool = False, cubicBool = True)
    BasicType.updateAssetsGrid()
    BasicType.timeFwd()    
   
    # Solve the basic type and plot the results, to make sure things are working
    start_time = clock()
    BasicType.solve()
    end_time = clock()
    print('Solving the basic consumer took ' + mystr(end_time-start_time) + ' seconds.')
    BasicType.unpackcFunc()
    print('Consumption function:')
    plotFuncs(BasicType.cFunc[0],0,5)    # plot consumption
    print('Marginal consumption function:')
    plotFuncsDer(BasicType.cFunc[0],0,5) # plot MPC
    if BasicType.vFuncBool:
        print('Value function:')
        plotFuncs(BasicType.solution[0].vFunc,0.2,5)
    
    # Make many copies of the basic type, each with a different risk aversion
    BasicType.vFuncBool = False # just in case it was set to True above
    my_agent_list = []
    CRRA_list = np.linspace(1,8,type_count) # All the values that CRRA will take on
    for i in range(type_count):
        this_agent = deepcopy(BasicType)   # Make a new copy of the basic type
        this_agent.assignParameters(CRRA = CRRA_list[i]) # Give it a unique CRRA value
        my_agent_list.append(this_agent)   # Addd it to the list of agent types
        
    # Make a list of commands to be run in parallel; these should be methods of ConsumerType
    do_this_stuff = ['updateSolutionTerminal()','solve()','unpackcFunc()']
    
    # Solve the model for each type by looping over the types (not multithreading)
    start_time = clock()
    multiThreadCommandsFake(my_agent_list, do_this_stuff) # Fake multithreading, just loops
    end_time = clock()
    print('Solving ' + str(type_count) +  ' types without multithreading took ' + mystr(end_time-start_time) + ' seconds.')
    
    # Plot the consumption functions for all types on one figure
    plotFuncs([this_type.cFunc[0] for this_type in my_agent_list],0,5)
    
    # Delete the solution for each type to make sure we're not just faking it
    for i in range(type_count):
        my_agent_list[i].solution = None
        my_agent_list[i].cFunc = None
        my_agent_list[i].time_vary.remove('solution')
        my_agent_list[i].time_vary.remove('cFunc')
    
    # And here's HARK's initial attempt at multithreading:
    start_time = clock()
    multiThreadCommands(my_agent_list, do_this_stuff) # Actual multithreading
    end_time = clock()
    print('Solving ' + str(type_count) +  ' types with multithreading took ' + mystr(end_time-start_time) + ' seconds.')
    
    # Plot the consumption functions for all types on one figure to see if it worked
    plotFuncs([this_type.cFunc[0] for this_type in my_agent_list],0,5)

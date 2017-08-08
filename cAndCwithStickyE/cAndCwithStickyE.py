'''
Runs the exercises and regressions for the cAndCwithStickyE paper.
'''
import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../ConsumptionSaving'))

import numpy as np
from time import clock
from StickyEmodel import StickyEconsumerType, StickyEmarkovConsumerType, StickyErepAgent, StickyEmarkovRepAgent
import StickyEparams as Params
from ConsAggShockModel import SmallOpenEconomy, SmallOpenMarkovEconomy, CobbDouglasEconomy,CobbDouglasMarkovEconomy
from HARKutilities import plotFuncs
import matplotlib.pyplot as plt
ignore_periods = Params.ignore_periods

# Choose which models to run
do_SOE_simple  = False
do_SOE_markov  = False
do_DSGE_simple = False
do_DSGE_markov = False
do_RA_simple   = False
do_RA_markov   = False


def makeDescriptiveStatistics(Economy):
    '''
    Makes descriptive statistics for a model after it has been solved and simulated,
    like the ones in Table 2 of the draft paper.  Behaves slightly differently for
    heterogeneous agents vs representative agent models.
    
    Parameters
    ----------
    Economy : Market or AgentType
        A representation of the model economy.  For heterogeneous agents specifications,
        this will be an instance of a subclass of Market.  For representative agent
        specifications, this will be an instance of an AgentType subclass.
        
    Returns
    -------
    stat_string : str
        Large string with descriptive statistics.
    '''
    # Extract time series data from the economy
    if hasattr(Economy,'agents'): # If this is a heterogeneous agent specification...
        PlvlAgg_hist = np.cumprod(Economy.PermShkAggHist)
        pLvlAll_hist = np.concatenate([this_type.pLvlTrue_hist for this_type in Economy.agents],axis=1)
        aLvlAll_hist = np.concatenate([this_type.aLvlNow_hist for this_type in Economy.agents],axis=1)
        AlvlAgg_hist = np.mean(aLvlAll_hist,axis=1) # Level of aggregate assets
        AnrmAgg_hist = AlvlAgg_hist/PlvlAgg_hist # Normalized level of aggregate assets
        cLvlAll_hist = np.concatenate([this_type.cLvlNow_hist for this_type in Economy.agents],axis=1)
        ClvlAgg_hist = np.mean(cLvlAll_hist,axis=1) # Level of aggregate consumption
        CnrmAgg_hist = ClvlAgg_hist/PlvlAgg_hist # Normalized level of aggregate consumption
        yLvlAll_hist = np.concatenate([this_type.yLvlNow_hist for this_type in Economy.agents],axis=1)
        YlvlAgg_hist = np.mean(yLvlAll_hist,axis=1) # Level of aggregate consumption
        YnrmAgg_hist = YlvlAgg_hist/PlvlAgg_hist # Normalized level of aggregate consumption
        
        not_newborns = (np.concatenate([this_type.t_age_hist[(ignore_periods+1):,:] for this_type in Economy.agents],axis=1) > 1).flatten()
        Logc = np.log(cLvlAll_hist[ignore_periods:,:])
        DeltaLogc = (Logc[1:] - Logc[0:-1]).flatten()
        DeltaLogc_trimmed = DeltaLogc[not_newborns]
        Loga = np.log(aLvlAll_hist[ignore_periods:,:])
        DeltaLoga = (Loga[1:] - Loga[0:-1]).flatten()
        DeltaLoga_trimmed = DeltaLoga[not_newborns]
        Logp = np.log(pLvlAll_hist[ignore_periods:,:])
        DeltaLogp = (Logp[1:] - Logp[0:-1]).flatten()
        DeltaLogp_trimmed = DeltaLogp[not_newborns]
        Logy = np.log(yLvlAll_hist[ignore_periods:,:])
        Logy_trimmed = Logy
        Logy_trimmed[np.isinf(Logy)] = np.nan
        
    else: # If this is a representative agent specification...
        PlvlAgg_hist = Economy.pLvlTrue_hist
        ClvlAgg_hist = Economy.cLvlNow_hist
        CnrmAgg_hist = ClvlAgg_hist/PlvlAgg_hist
        YnrmAgg_hist = Economy.yNrmTrue_hist
        YlvlAgg_hist = YnrmAgg_hist*PlvlAgg_hist
        AlvlAgg_hist = Economy.aLvlNow_hist
        AnrmAgg_hist = AlvlAgg_hist/PlvlAgg_hist
        
    # Process aggregate data    
    LogC = np.log(ClvlAgg_hist[ignore_periods:])
    LogA = np.log(AlvlAgg_hist[ignore_periods:])
    LogY = np.log(YlvlAgg_hist[ignore_periods:])
    DeltaLogC = LogC[1:] - LogC[0:-1]
    DeltaLogA = LogA[1:] - LogA[0:-1]
    DeltaLogY = LogY[1:] - LogY[0:-1]
    
    # Make and return the output string
    stat_string  = 'Average aggregate asset-to-productivity ratio = ' + str(np.mean(AnrmAgg_hist[ignore_periods:])) + '\n'
    stat_string += 'Average aggregate consumption-to-productivity ratio = ' + str(np.mean(CnrmAgg_hist[ignore_periods:])) + '\n'
    stat_string += 'Stdev of log aggregate asset-to-productivity ratio = ' + str(np.std(np.log(AnrmAgg_hist[ignore_periods:]))) + '\n'
    stat_string += 'Stdev of change in log aggregate consumption level = ' + str(np.std(DeltaLogC)) + '\n'
    stat_string += 'Stdev of change in log aggregate output level = ' + str(np.std(DeltaLogY)) + '\n'
    stat_string += 'Stdev of change in log aggregate assets level = ' + str(np.std(DeltaLogA)) + '\n'
    if hasattr(Economy,'agents'):
        stat_string += 'Cross section stdev of log individual assets = ' + str(np.mean(np.std(Loga,axis=1))) + '\n'
        stat_string += 'Cross section stdev of log individual consumption = ' + str(np.mean(np.std(Logc,axis=1))) + '\n'
        stat_string += 'Cross section stdev of log individual productivity = ' + str(np.mean(np.std(Logp,axis=1))) + '\n'
        stat_string += 'Cross section stdev of log individual non-zero income = ' + str(np.mean(np.std(Logy_trimmed,axis=1))) + '\n'
        stat_string += 'Cross section stdev of change in log individual assets = ' + str(np.std(DeltaLoga_trimmed)) + '\n'
        stat_string += 'Cross section stdev of change in log individual consumption = ' + str(np.std(DeltaLogc_trimmed)) + '\n'
        stat_string += 'Cross section stdev of change in log individual productivity = ' + str(np.std(DeltaLogp_trimmed)) + '\n'
    return stat_string



###############################################################################

if do_SOE_simple:
    # Make a small open economy and the consumers who live in it
    StickySOEconsumers = StickyEconsumerType(**Params.init_SOE_consumer)
    StickySOEconomy = SmallOpenEconomy(agents=[StickySOEconsumers],**Params.init_SOE_market)
    StickySOEconomy.makeAggShkHist()
    StickySOEconsumers.getEconomyData(StickySOEconomy)
    StickySOEconsumers.track_vars = ['aLvlNow','cLvlNow','yLvlNow','pLvlTrue','t_age']
    
    # Solve the model and display some output
    t_start = clock()
    StickySOEconomy.solveAgents()
    StickySOEconomy.makeHistory()
    t_end = clock()
    print('Solving the small open economy took ' + str(t_end-t_start) + ' seconds.')
    
    # Plot the consumption function
    print('Consumption function for the small open economy:')
    cFunc = lambda m : StickySOEconsumers.solution[0].cFunc(m,np.ones_like(m))
    plotFuncs(cFunc,0.0,20.0)
    
    print('Descriptive statistics for the small open economy:')
    print(makeDescriptiveStatistics(StickySOEconomy))


###############################################################################

if do_SOE_markov:
    # Make a consumer type to inhabit the small open Markov economy
    StickySOEmarkovConsumers = StickyEmarkovConsumerType(**Params.init_SOE_mrkv_consumer)
    StickySOEmarkovConsumers.IncomeDstn[0] = Params.StateCount*[StickySOEmarkovConsumers.IncomeDstn[0]]
    StickySOEmarkovConsumers.track_vars = ['aLvlNow','cLvlNow','yLvlNow','pLvlTrue','t_age']
    
    # Make a Cobb-Douglas economy for the agents
    StickySOmarkovEconomy = SmallOpenMarkovEconomy(agents = [StickySOEmarkovConsumers],**Params.init_SOE_mrkv_market)
    StickySOmarkovEconomy.makeAggShkHist() # Simulate a history of aggregate shocks
    StickySOEmarkovConsumers.getEconomyData(StickySOmarkovEconomy) # Have the consumers inherit relevant objects from the economy
    
    # Solve the model
    t_start = clock()
    StickySOmarkovEconomy.solveAgents()
    StickySOmarkovEconomy.makeHistory()
    t_end = clock()
    print('Solving the small open Markov economy took ' + str(t_end-t_start) + ' seconds.')
    
    # Plot the consumption function in each Markov state
    print('Consumption function for the small open Markov economy:')
    m = np.linspace(0,20,500)
    M = np.ones_like(m)
    c = np.zeros((Params.StateCount,m.size))
    for i in range(Params.StateCount):
        c[i,:] = StickySOEmarkovConsumers.solution[0].cFunc[i](m,M)
        plt.plot(m,c[i,:])
    plt.show()
    
    print('Descriptive statistics for the small open Markov economy:')
    print(makeDescriptiveStatistics(StickySOmarkovEconomy))
    

###############################################################################

if do_DSGE_simple:
    # Make a Cobb-Douglas economy and the consumers who live in it
    StickyDSGEconsumers = StickyEconsumerType(**Params.init_DSGE_consumer)
    StickyDSGEeconomy = CobbDouglasEconomy(agents=[StickyDSGEconsumers],**Params.init_DSGE_market)
    StickyDSGEeconomy.makeAggShkHist()
    StickyDSGEconsumers.getEconomyData(StickyDSGEeconomy)
    StickyDSGEconsumers.track_vars = ['aLvlNow','cLvlNow','yLvlNow','pLvlTrue','t_age']
    
    # Solve the model
    t_start = clock()
    StickyDSGEeconomy.solve()
    t_end = clock()
    print('Solving the Cobb-Douglas economy took ' + str(t_end-t_start) + ' seconds.')
    
    # Plot the consumption function
    print('Consumption function for the Cobb-Douglas economy:')
    m = np.linspace(0.,20.,300)
    for M in StickyDSGEconsumers.Mgrid:
        c = StickyDSGEconsumers.solution[0].cFunc(m,M*np.ones_like(m))
        plt.plot(m,c)
    plt.show()
    
    print('Descriptive statistics for the Cobb-Douglas economy:')
    print(makeDescriptiveStatistics(StickyDSGEeconomy))

###############################################################################


if do_DSGE_markov:
    # Make a consumer type to inhabit the small open Markov economy
    StickyDSGEmarkovConsumers = StickyEmarkovConsumerType(**Params.init_DSGE_mrkv_consumer)
    StickyDSGEmarkovConsumers.IncomeDstn[0] = Params.StateCount*[StickyDSGEmarkovConsumers.IncomeDstn[0]]
    StickyDSGEmarkovConsumers.track_vars = ['aLvlNow','cLvlNow','yLvlNow','pLvlTrue','t_age']
    
    # Make a Cobb-Douglas economy for the agents
    StickyDSGEmarkovEconomy = CobbDouglasMarkovEconomy(agents = [StickyDSGEmarkovConsumers],**Params.init_DSGE_mrkv_market)
    StickyDSGEmarkovEconomy.makeAggShkHist() # Simulate a history of aggregate shocks
    StickyDSGEmarkovConsumers.getEconomyData(StickyDSGEmarkovEconomy) # Have the consumers inherit relevant objects from the economy
    
    # Solve the model
    t_start = clock()
    StickyDSGEmarkovEconomy.solve()
    t_end = clock()
    print('Solving the Cobb-Douglas Markov economy took ' + str(t_end-t_start) + ' seconds.')
    
    print('Descriptive statistics for the Cobb-Douglas Markov economy:')
    print(makeDescriptiveStatistics(StickyDSGEmarkovEconomy))
    

###############################################################################

if do_RA_simple:
    # Make a representative agent consumer, then solve and simulate the model
    StickyRAconsumer = StickyErepAgent(**Params.init_RA_consumer)
    StickyRAconsumer.track_vars = ['cLvlNow','yNrmTrue','aLvlNow','pLvlTrue']
    StickyRAconsumer.initializeSim()
    
    t_start = clock()
    StickyRAconsumer.solve()
    StickyRAconsumer.simulate()
    t_end = clock()
    print('Solving the representative agent economy took ' + str(t_end-t_start) + ' seconds.')
    
    print('Consumption function for the representative agent:')
    plotFuncs(StickyRAconsumer.solution[0].cFunc,0,50)
    
    print('Descriptive statistics for the representative agent economy:')
    print(makeDescriptiveStatistics(StickyRAconsumer))
    

###############################################################################

if do_RA_markov:
    # Make a representative agent consumer, then solve and simulate the model
    StickyRAmarkovConsumer = StickyEmarkovRepAgent(**Params.init_RA_mrkv_consumer)
    StickyRAmarkovConsumer.IncomeDstn[0] = Params.StateCount*[StickyRAmarkovConsumer.IncomeDstn[0]]
    StickyRAmarkovConsumer.track_vars = ['cLvlNow','yNrmTrue','aLvlNow','pLvlTrue']
    StickyRAmarkovConsumer.initializeSim()
    
    t_start = clock()
    StickyRAmarkovConsumer.solve()
    StickyRAmarkovConsumer.simulate()
    t_end = clock()
    print('Solving the representative agent Markov economy took ' + str(t_end-t_start) + ' seconds.')
    
    print('Consumption functions for the Markov representative agent:')
    plotFuncs(StickyRAmarkovConsumer.solution[0].cFunc,0,50)
    
    print('Descriptive statistics for the Markov representative agent economy:')
    print(makeDescriptiveStatistics(StickyRAmarkovConsumer))
    
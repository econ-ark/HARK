'''
Runs the exercises and regressions for the cAndCwithStickyE paper.
'''
import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../ConsumptionSaving'))

import numpy as np
from copy import deepcopy
from time import clock
from StickyEmodel import StickyEconsumerType, StickyEmarkovConsumerType, StickyErepAgent, StickyEmarkovRepAgent
import StickyEparams as Params
from ConsAggShockModel import SmallOpenEconomy, SmallOpenMarkovEconomy, CobbDouglasEconomy,CobbDouglasMarkovEconomy
from RepAgentModel import RepAgentMarkovConsumerType
from HARKutilities import plotFuncs
import matplotlib.pyplot as plt
ignore_periods = Params.ignore_periods

# Choose which models to run
do_SOE_simple  = False
do_SOE_markov  = False
do_DSGE_simple = False
do_DSGE_markov = False
do_RA_simple   = False
do_RA_markov   = True

###############################################################################

if do_SOE_simple:
    # Make a small open economy and the consumers who live in it
    StickySOEconsumers = StickyEconsumerType(**Params.init_SOE_consumer)
    StickySOEconomy = SmallOpenEconomy(agents=[StickySOEconsumers],**Params.init_SOE_market)
    StickySOEconomy.makeAggShkHist()
    StickySOEconsumers.getEconomyData(StickySOEconomy)
    StickySOEconsumers.aNrmInitMean = np.log(1.0)  #Don't want newborns to have no assets and also be unemployed
    StickySOEconsumers.track_vars = ['aLvlNow','aNrmNow','mNrmNow','cNrmNow','cLvlNow','mLvlTrueNow','pLvlNow','t_age']
    
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
    PlvlAgg_hist = np.cumprod(StickySOEconomy.PermShkAggHist)
    print('Average aggregate assets = ' + str(np.mean(np.mean(StickySOEconsumers.aLvlNow_hist[ignore_periods:,:],axis=1)/PlvlAgg_hist[ignore_periods:])))
    print('Average aggregate consumption = ' + str(np.mean(np.mean(StickySOEconsumers.cLvlNow_hist[ignore_periods:,:],axis=1)/PlvlAgg_hist[ignore_periods:])))
    print('Standard deviation of log aggregate assets = ' + str(np.std(np.log(np.mean(StickySOEconsumers.aLvlNow_hist[ignore_periods:,:],axis=1)))))
    LogC = np.log(np.mean(StickySOEconsumers.cLvlNow_hist,axis=1))[ignore_periods:]
    DeltaLogC = LogC[1:] - LogC[0:-1]
    LogA = np.log(np.mean(StickySOEconsumers.aLvlNow_hist,axis=1))[ignore_periods:]
    DeltaLogA = LogA[1:] - LogA[0:-1]
    print('Standard deviation of change in log aggregate assets = ' + str(np.std(DeltaLogA)))
    print('Standard deviation of change in log aggregate consumption = ' + str(np.std(DeltaLogC)))
    print('Standard deviation of log individual assets = ' + str(np.mean(np.std(np.log(StickySOEconsumers.aLvlNow_hist[ignore_periods:,:]),axis=1))))
    print('Standard deviation of log individual consumption = ' + str(np.mean(np.std(np.log(StickySOEconsumers.cLvlNow_hist[ignore_periods:,:]),axis=1))))
    print('Standard deviation of log individual productivity = ' + str(np.mean(np.std(np.log(StickySOEconsumers.pLvlNow_hist[ignore_periods:,:]),axis=1))))
    not_newborns = (StickySOEconsumers.t_age_hist[(ignore_periods+1):,:] > 1).flatten()
    Logp = np.log(StickySOEconsumers.pLvlNow_hist)[ignore_periods:,:]
    DeltaLogp = (Logp[1:,:] - Logp[0:-1,:]).flatten()
    print('Standard deviation of change in log individual productivity = ' + str(np.std(DeltaLogp[not_newborns])))
    Logc = np.log(StickySOEconsumers.cLvlNow_hist)[ignore_periods:,:]
    DeltaLogc = (Logc[1:,:] - Logc[0:-1,:]).flatten()
    print('Standard deviation of change in log individual consumption = ' + str(np.std(DeltaLogc[not_newborns])))



###############################################################################


if do_SOE_markov:
    # Make a consumer type to inhabit the small open Markov economy
    StickySOEmarkovConsumers = StickyEmarkovConsumerType(**Params.init_SOE_mrkv_consumer)
    StickySOEmarkovConsumers.IncomeDstn[0] = Params.StateCount*[StickySOEmarkovConsumers.IncomeDstn[0]]
    StickySOEmarkovConsumers.track_vars = ['aLvlNow','aNrmNow','mNrmNow','cNrmNow','cLvlNow','mLvlTrueNow','pLvlNow','t_age']
    
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
    PlvlAgg_hist = np.cumprod(StickySOmarkovEconomy.PermShkAggHist)
    print('Average aggregate assets = ' + str(np.mean(np.mean(StickySOEmarkovConsumers.aLvlNow_hist[ignore_periods:,:],axis=1)/PlvlAgg_hist[ignore_periods:])))
    print('Average aggregate consumption = ' + str(np.mean(np.mean(StickySOEmarkovConsumers.cLvlNow_hist[ignore_periods:,:],axis=1)/PlvlAgg_hist[ignore_periods:])))
    print('Standard deviation of log aggregate assets = ' + str(np.std(np.log(np.mean(StickySOEmarkovConsumers.aLvlNow_hist[ignore_periods:,:],axis=1)))))
    LogC = np.log(np.mean(StickySOEmarkovConsumers.cLvlNow_hist,axis=1))[ignore_periods:]
    DeltaLogC = LogC[1:] - LogC[0:-1]
    LogA = np.log(np.mean(StickySOEmarkovConsumers.aLvlNow_hist,axis=1))[ignore_periods:]
    DeltaLogA = LogA[1:] - LogA[0:-1]
    print('Standard deviation of change in log aggregate assets = ' + str(np.std(DeltaLogA)))
    print('Standard deviation of change in log aggregate consumption = ' + str(np.std(DeltaLogC)))
    print('Standard deviation of log individual assets = ' + str(np.mean(np.std(np.log(StickySOEmarkovConsumers.aLvlNow_hist[ignore_periods:,:]),axis=1))))
    print('Standard deviation of log individual consumption = ' + str(np.mean(np.std(np.log(StickySOEmarkovConsumers.cLvlNow_hist[ignore_periods:,:]),axis=1))))
    print('Standard deviation of log individual productivity = ' + str(np.mean(np.std(np.log(StickySOEmarkovConsumers.pLvlNow_hist[ignore_periods:,:]),axis=1))))
    not_newborns = (StickySOEmarkovConsumers.t_age_hist[(ignore_periods+1):,:] > 1).flatten()
    Logp = np.log(StickySOEmarkovConsumers.pLvlNow_hist)[ignore_periods:,:]
    DeltaLogp = (Logp[1:,:] - Logp[0:-1,:]).flatten()
    print('Standard deviation of change in log individual productivity = ' + str(np.std(DeltaLogp[not_newborns])))
    Logc = np.log(StickySOEmarkovConsumers.cLvlNow_hist)[ignore_periods:,:]
    DeltaLogc = (Logc[1:,:] - Logc[0:-1,:]).flatten()
    print('Standard deviation of change in log individual consumption = ' + str(np.std(DeltaLogc[not_newborns])))



###############################################################################


if do_DSGE_simple:
    # Make a Cobb-Douglas economy and the consumers who live in it
    StickyDSGEconsumers = StickyEconsumerType(**Params.init_DSGE_consumer)
    StickyDSGEeconomy = CobbDouglasEconomy(agents=[StickyDSGEconsumers],**Params.init_DSGE_market)
    StickyDSGEeconomy.makeAggShkHist()
    StickyDSGEconsumers.getEconomyData(StickyDSGEeconomy)
    StickyDSGEconsumers.aNrmInitMean = np.log(1.0)  #Don't want newborns to have no assets and also be unemployed
    StickyDSGEconsumers.track_vars = ['aLvlNow','aNrmNow','mNrmNow','cNrmNow','cLvlNow','mLvlTrueNow','pLvlNow','t_age']
    
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
    PlvlAgg_hist = np.cumprod(StickyDSGEeconomy.PermShkAggHist)
    print('Average aggregate assets = ' + str(np.mean(np.mean(StickyDSGEconsumers.aLvlNow_hist[ignore_periods:,:],axis=1)/PlvlAgg_hist[ignore_periods:])))
    print('Average aggregate consumption = ' + str(np.mean(np.mean(StickyDSGEconsumers.cLvlNow_hist[ignore_periods:,:],axis=1)/PlvlAgg_hist[ignore_periods:])))
    print('Standard deviation of log aggregate assets = ' + str(np.std(np.log(np.mean(StickyDSGEconsumers.aLvlNow_hist[ignore_periods:,:],axis=1)))))
    LogC = np.log(np.mean(StickyDSGEconsumers.cLvlNow_hist,axis=1))[ignore_periods:]
    DeltaLogC = LogC[1:] - LogC[0:-1]
    LogA = np.log(np.mean(StickyDSGEconsumers.aLvlNow_hist,axis=1))[ignore_periods:]
    DeltaLogA = LogA[1:] - LogA[0:-1]
    print('Standard deviation of change in log aggregate assets = ' + str(np.std(DeltaLogA)))
    print('Standard deviation of change in log aggregate consumption = ' + str(np.std(DeltaLogC)))
    print('Standard deviation of log individual assets = ' + str(np.mean(np.std(np.log(StickyDSGEconsumers.aLvlNow_hist[ignore_periods:,:]),axis=1))))
    print('Standard deviation of log individual consumption = ' + str(np.mean(np.std(np.log(StickyDSGEconsumers.cLvlNow_hist[ignore_periods:,:]),axis=1))))
    print('Standard deviation of log individual productivity = ' + str(np.mean(np.std(np.log(StickyDSGEconsumers.pLvlNow_hist[ignore_periods:,:]),axis=1))))
    not_newborns = (StickyDSGEconsumers.t_age_hist[(ignore_periods+1):,:] > 1).flatten()
    Logp = np.log(StickyDSGEconsumers.pLvlNow_hist)[ignore_periods:,:]
    DeltaLogp = (Logp[1:,:] - Logp[0:-1,:]).flatten()
    print('Standard deviation of change in log individual productivity = ' + str(np.std(DeltaLogp[not_newborns])))
    Logc = np.log(StickyDSGEconsumers.cLvlNow_hist)[ignore_periods:,:]
    DeltaLogc = (Logc[1:,:] - Logc[0:-1,:]).flatten()
    print('Standard deviation of change in log individual consumption = ' + str(np.std(DeltaLogc[not_newborns])))




###############################################################################


if do_DSGE_markov:
    # Make a consumer type to inhabit the small open Markov economy
    StickyDSGEmarkovConsumers = StickyEmarkovConsumerType(**Params.init_DSGE_mrkv_consumer)
    StickyDSGEmarkovConsumers.IncomeDstn[0] = Params.StateCount*[StickyDSGEmarkovConsumers.IncomeDstn[0]]
    StickyDSGEmarkovConsumers.track_vars = ['aLvlNow','aNrmNow','mNrmNow','cNrmNow','cLvlNow','mLvlTrueNow','pLvlNow','t_age']
    
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
    PlvlAgg_hist = np.cumprod(StickyDSGEmarkovEconomy.PermShkAggHist)
    print('Average aggregate assets = ' + str(np.mean(np.mean(StickyDSGEmarkovConsumers.aLvlNow_hist[ignore_periods:,:],axis=1)/PlvlAgg_hist[ignore_periods:])))
    print('Average aggregate consumption = ' + str(np.mean(np.mean(StickyDSGEmarkovConsumers.cLvlNow_hist[ignore_periods:,:],axis=1)/PlvlAgg_hist[ignore_periods:])))
    print('Standard deviation of log aggregate assets = ' + str(np.std(np.log(np.mean(StickyDSGEmarkovConsumers.aLvlNow_hist[ignore_periods:,:],axis=1)))))
    LogC = np.log(np.mean(StickyDSGEmarkovConsumers.cLvlNow_hist,axis=1))[ignore_periods:]
    DeltaLogC = LogC[1:] - LogC[0:-1]
    LogA = np.log(np.mean(StickyDSGEmarkovConsumers.aLvlNow_hist,axis=1))[ignore_periods:]
    DeltaLogA = LogA[1:] - LogA[0:-1]
    print('Standard deviation of change in log aggregate assets = ' + str(np.std(DeltaLogA)))
    print('Standard deviation of change in log aggregate consumption = ' + str(np.std(DeltaLogC)))
    print('Standard deviation of log individual assets = ' + str(np.mean(np.std(np.log(StickyDSGEmarkovConsumers.aLvlNow_hist[ignore_periods:,:]),axis=1))))
    print('Standard deviation of log individual consumption = ' + str(np.mean(np.std(np.log(StickyDSGEmarkovConsumers.cLvlNow_hist[ignore_periods:,:]),axis=1))))
    print('Standard deviation of log individual productivity = ' + str(np.mean(np.std(np.log(StickyDSGEmarkovConsumers.pLvlNow_hist[ignore_periods:,:]),axis=1))))
    not_newborns = (StickyDSGEmarkovConsumers.t_age_hist[(ignore_periods+1):,:] > 1).flatten()
    Logp = np.log(StickyDSGEmarkovConsumers.pLvlNow_hist)[ignore_periods:,:]
    DeltaLogp = (Logp[1:,:] - Logp[0:-1,:]).flatten()
    print('Standard deviation of change in log individual productivity = ' + str(np.std(DeltaLogp[not_newborns])))
    Logc = np.log(StickyDSGEmarkovConsumers.cLvlNow_hist)[ignore_periods:,:]
    DeltaLogc = (Logc[1:,:] - Logc[0:-1,:]).flatten()
    print('Standard deviation of change in log individual consumption = ' + str(np.std(DeltaLogc[not_newborns])))

###############################################################################


if do_RA_simple:
    # Make a representative agent consumer, then solve and simulate the model
    StickyRAconsumer = StickyErepAgent(**Params.init_RA_consumer)
    StickyRAconsumer.track_vars = ['cNrmNow','cLvlNow','aNrmNow','pLvlNow','yNrmTrue','aLvlNow','pLvlTrue','yNrmNow']
    StickyRAconsumer.initializeSim()
    
    t_start = clock()
    StickyRAconsumer.solve()
    StickyRAconsumer.simulate()
    t_end = clock()
    print('Solving the representative agent economy took ' + str(t_end-t_start) + ' seconds.')
    
    print('Consumption function for the representative agent:')
    plotFuncs(StickyRAconsumer.solution[0].cFunc,0,50)
    
    print('Descriptive statistics for the representative agent economy:')
    PlvlAgg_hist = StickyRAconsumer.pLvlTrue_hist
    print('Average aggregate assets = ' + str(np.mean(StickyRAconsumer.aLvlNow_hist[ignore_periods:,:]/PlvlAgg_hist[ignore_periods:,:])))
    print('Average aggregate consumption = ' + str(np.mean(StickyRAconsumer.cLvlNow_hist[ignore_periods:,:]/PlvlAgg_hist[ignore_periods:,:])))
    print('Standard deviation of log aggregate assets = ' + str(np.std(np.log(StickyRAconsumer.aLvlNow_hist[ignore_periods:,:]/PlvlAgg_hist[ignore_periods:,:]))))
    LogA = np.log(np.mean(StickyRAconsumer.aLvlNow_hist,axis=1))[ignore_periods:]
    DeltaLogA = LogA[1:] - LogA[0:-1]
    print('Standard deviation of change in log aggregate assets = ' + str(np.std(DeltaLogA)))
    LogC = np.log(np.mean(StickyRAconsumer.cLvlNow_hist,axis=1))[ignore_periods:]
    DeltaLogC = LogC[1:] - LogC[0:-1]
    print('Standard deviation of change in log aggregate consumption = ' + str(np.std(DeltaLogC)))
    LogY = np.log(np.mean(StickyRAconsumer.yNrmTrue_hist*StickyRAconsumer.pLvlTrue_hist,axis=1))[ignore_periods:]
    DeltaLogY = LogY[1:] - LogY[0:-1]
    print('Standard deviation of change in log aggregate output = ' + str(np.std(DeltaLogY)))


###############################################################################


if do_RA_markov:
    # Make a representative agent consumer, then solve and simulate the model
    StickyRAmarkovConsumer = StickyEmarkovRepAgent(**Params.init_RA_mrkv_consumer)
    StickyRAmarkovConsumer.IncomeDstn[0] = Params.StateCount*[StickyRAmarkovConsumer.IncomeDstn[0]]
    StickyRAmarkovConsumer.track_vars = ['cLvlNow','aNrmNow','yNrmTrue','aLvlNow','pLvlTrue','MrkvNow']
    StickyRAmarkovConsumer.initializeSim()
    
    t_start = clock()
    StickyRAmarkovConsumer.solve()
    StickyRAmarkovConsumer.simulate()
    t_end = clock()
    print('Solving the representative agent Markov economy took ' + str(t_end-t_start) + ' seconds.')
    
    print('Consumption functions for the Markov representative agent:')
    plotFuncs(StickyRAmarkovConsumer.solution[0].cFunc,0,50)
    
    print('Descriptive statistics for the representative agent economy:')
    PlvlAgg_hist = StickyRAmarkovConsumer.pLvlTrue_hist
    print('Average aggregate assets = ' + str(np.mean(StickyRAmarkovConsumer.aLvlNow_hist[ignore_periods:,:]/PlvlAgg_hist[ignore_periods:,:])))
    print('Average aggregate consumption = ' + str(np.mean(StickyRAmarkovConsumer.cLvlNow_hist[ignore_periods:,:]/PlvlAgg_hist[ignore_periods:,:])))
    print('Standard deviation of log aggregate assets = ' + str(np.std(np.log(StickyRAmarkovConsumer.aLvlNow_hist[ignore_periods:,:]/PlvlAgg_hist[ignore_periods:,:]))))
    LogA = np.log(np.mean(StickyRAmarkovConsumer.aLvlNow_hist,axis=1))[ignore_periods:]
    DeltaLogA = LogA[1:] - LogA[0:-1]
    print('Standard deviation of change in log aggregate assets = ' + str(np.std(DeltaLogA)))
    LogC = np.log(np.mean(StickyRAmarkovConsumer.cLvlNow_hist,axis=1))[ignore_periods:]
    DeltaLogC = LogC[1:] - LogC[0:-1]
    print('Standard deviation of change in log aggregate consumption = ' + str(np.std(DeltaLogC)))
    LogY = np.log(np.mean(StickyRAmarkovConsumer.yNrmTrue_hist*StickyRAmarkovConsumer.pLvlTrue_hist,axis=1))[ignore_periods:]
    DeltaLogY = LogY[1:] - LogY[0:-1]
    print('Standard deviation of change in log aggregate output = ' + str(np.std(DeltaLogY)))
    
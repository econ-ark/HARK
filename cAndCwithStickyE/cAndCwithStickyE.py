'''
Runs the exercises and regressions for the cAndCwithStickyE paper.
'''
import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../ConsumptionSaving'))

import numpy as np
from StickyEmodel import StickyEconsumerSOEType, StickyEconsumerDSGEType
import StickyEparams as Params
from ConsAggShockModel import SmallOpenEconomy, SmallOpenMarkovEconomy, AggShockMarkovConsumerType
from HARKutilities import plotFuncs
import matplotlib.pyplot as plt

ignore_periods = Params.ignore_periods

# Make a small open economy and the consumers who live in it
StickySOEconsumers     = StickyEconsumerSOEType(**Params.init_SOE_consumer)
StickySOEconomy        = SmallOpenEconomy(**Params.init_SOE_market)
StickySOEconomy.agents = [StickySOEconsumers]
StickySOEconomy.makeAggShkHist()
StickySOEconsumers.getEconomyData(StickySOEconomy)
StickySOEconsumers.aNrmInitMean = np.log(1.0)  #Don't want newborns to have no assets and also be unemployed
StickySOEconsumers.track_vars = ['aLvlNow','aNrmNow','mNrmNow','cNrmNow','cLvlNow','mLvlTrueNow','pLvlNow','pLvlErrNow','TranShkAggNow']

# Solve the model and display some output
StickySOEconomy.solveAgents()
StickySOEconomy.makeHistory()

# Plot some of the results
cFunc = lambda m : StickySOEconsumers.solution[0].cFunc(m,np.ones_like(m))
plotFuncs(cFunc,0.0,20.0)

print('Average aggregate assets = ' + str(np.mean(StickySOEconsumers.aLvlNow_hist[ignore_periods:,:])))
print('Average aggregate consumption = ' + str(np.mean(StickySOEconsumers.cLvlNow_hist[ignore_periods:,:])))
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
Logc = np.log(StickySOEconsumers.cLvlNow_hist)[ignore_periods:,:]
DeltaLogc = Logc[1:,:] - Logc[0:-1,:]
print('Standard deviation of change in log individual consumption = ' + str(np.std(DeltaLogc.flatten())))



###############################################################################


# Make a representative agent consumer, then solve and simulate the model
RAconsumer = StickyEconsumerDSGEType(**Params.init_RA_consumer)
RAconsumer.solve()
RAconsumer.track_vars = ['cNrmNow','cLvlNow','aNrmNow','pLvlNow','yNrmTrue','aLvlNow','pLvlTrue','yNrmNow']
RAconsumer.initializeSim()
RAconsumer.simulate()

plotFuncs(RAconsumer.solution[0].cFunc,0,20)

print('Average aggregate assets = ' + str(np.mean(RAconsumer.aLvlNow_hist[ignore_periods:,:])))
print('Average aggregate consumption = ' + str(np.mean(RAconsumer.cLvlNow_hist[ignore_periods:,:])))
print('Standard deviation of log aggregate assets = ' + str(np.std(np.log(RAconsumer.aLvlNow_hist[ignore_periods:,:]))))
LogA = np.log(np.mean(RAconsumer.aLvlNow_hist,axis=1))[ignore_periods:]
DeltaLogA = LogA[1:] - LogA[0:-1]
print('Standard deviation of change in log aggregate assets = ' + str(np.std(DeltaLogA)))
LogC = np.log(np.mean(RAconsumer.cLvlNow_hist,axis=1))[ignore_periods:]
DeltaLogC = LogC[1:] - LogC[0:-1]
print('Standard deviation of change in log aggregate consumption = ' + str(np.std(DeltaLogC)))
LogY = np.log(np.mean(RAconsumer.yNrmTrue_hist*RAconsumer.pLvlTrue_hist,axis=1))[ignore_periods:]
DeltaLogY = LogY[1:] - LogY[0:-1]
print('Standard deviation of change in log aggregate output = ' + str(np.std(DeltaLogY)))




###############################################################################

# Make a small open markov economy and the consumer who live in it
PermGroFacSet = np.linspace(Params.PermGroFacMin,Params.PermGroFacMax,num=Params.StateCount)

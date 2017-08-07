'''
Runs the exercises and regressions for the cAndCwithStickyE paper.
'''
import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../ConsumptionSaving'))

import numpy as np
from copy import deepcopy
from StickyEmodel import StickyEconsumerType, StickyEmarkovConsumerType, StickyErepAgent
import StickyEparams as Params
from ConsAggShockModel import SmallOpenEconomy, SmallOpenMarkovEconomy, AggShockMarkovConsumerType
from HARKutilities import plotFuncs
import matplotlib.pyplot as plt
ignore_periods = Params.ignore_periods

# Make a small open economy and the consumers who live in it
StickySOEconsumers = StickyEconsumerType(**Params.init_SOE_consumer)
StickySOEconomy = SmallOpenEconomy(agents=[StickySOEconsumers],**Params.init_SOE_market)
StickySOEconomy.makeAggShkHist()
StickySOEconsumers.getEconomyData(StickySOEconomy)
StickySOEconsumers.aNrmInitMean = np.log(1.0)  #Don't want newborns to have no assets and also be unemployed
StickySOEconsumers.track_vars = ['aLvlNow','aNrmNow','mNrmNow','cNrmNow','cLvlNow','mLvlTrueNow','pLvlNow','t_age']

# Solve the model and display some output
StickySOEconomy.solveAgents()
StickySOEconomy.makeHistory()

# Plot some of the results
cFunc = lambda m : StickySOEconsumers.solution[0].cFunc(m,np.ones_like(m))
plotFuncs(cFunc,0.0,20.0)

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


# Make a representative agent consumer, then solve and simulate the model
StickyRAconsumer = StickyErepAgent(**Params.init_RA_consumer)
StickyRAconsumer.solve()
StickyRAconsumer.track_vars = ['cNrmNow','cLvlNow','aNrmNow','pLvlNow','yNrmTrue','aLvlNow','pLvlTrue','yNrmNow']
StickyRAconsumer.initializeSim()
StickyRAconsumer.simulate()

plotFuncs(StickyRAconsumer.solution[0].cFunc,0,20)

print('Average aggregate assets = ' + str(np.mean(StickyRAconsumer.aLvlNow_hist[ignore_periods:,:])))
print('Average aggregate consumption = ' + str(np.mean(StickyRAconsumer.cLvlNow_hist[ignore_periods:,:])))
print('Standard deviation of log aggregate assets = ' + str(np.std(np.log(StickyRAconsumer.aLvlNow_hist[ignore_periods:,:]))))
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


# Make a consumer type to inhabit the Markov economy
StickySOEmarkovConsumers = StickyEmarkovConsumerType(**Params.init_SOE_markov_consumer)
StickySOEmarkovConsumers.IncomeDstn[0] = Params.StateCount*[StickySOEmarkovConsumers.IncomeDstn[0]]
StickySOEmarkovConsumers.track_vars = ['aLvlNow','aNrmNow','mNrmNow','cNrmNow','cLvlNow','mLvlTrueNow','pLvlNow','t_age']

# Make a Cobb-Douglas economy for the agents
StickySOmarkovEconomy = SmallOpenMarkovEconomy(agents = [StickySOEmarkovConsumers],**Params.init_SOE_mrkv_market)
StickySOmarkovEconomy.makeAggShkHist() # Simulate a history of aggregate shocks
StickySOEmarkovConsumers.getEconomyData(StickySOmarkovEconomy) # Have the consumers inherit relevant objects from the economy

# Solve the model and display some output
StickySOmarkovEconomy.solveAgents()
StickySOmarkovEconomy.makeHistory()

# Plot the consumption function in each Markov state
m = np.linspace(0,20,500)
M = np.ones_like(m)
c = np.zeros((Params.StateCount,m.size))
for i in range(Params.StateCount):
    c[i,:] = StickySOEmarkovConsumers.solution[0].cFunc[i](m,M)
    plt.plot(m,c[i,:])
plt.show()

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

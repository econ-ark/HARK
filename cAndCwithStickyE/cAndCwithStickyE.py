'''
Runs the exercises and regressions for the cAndCwithStickyE paper.
'''
import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../ConsumptionSaving'))

import numpy as np
#from copy import copy, deepcopy
from StickyEmodel import StickyEconsumerSOEType, StickyEconsumerDSGEType
from ConsAggShockModel import SmallOpenEconomy
from HARKutilities import plotFuncs
import matplotlib.pyplot as plt

periods_to_sim = 3500
ignore_periods = 1000
UpdatePrb = 1.0

# Define parameters for the small open economy version of the model
init_SOE_consumer = { 'CRRA': 2.0,
                      'DiscFac': 0.969,
                      'LivPrb': [0.995],
                      'PermGroFac': [1.0],
                      'AgentCount': 10000,
                      'aXtraMin': 0.00001,
                      'aXtraMax': 40.0,
                      'aXtraNestFac': 3,
                      'aXtraCount': 48,
                      'aXtraExtra': [None],
                      'PermShkStd': [np.sqrt(0.004)],
                      'PermShkCount': 7,
                      'TranShkStd': [np.sqrt(0.12)],
                      'TranShkCount': 7,
                      'UnempPrb': 0.05,
                      'UnempPrbRet': 0.0,
                      'IncUnemp': 0.0,
                      'IncUnempRet': 0.0,
                      'BoroCnstArt':0.0,
                      'tax_rate':0.0,
                      'T_retire':0,
                      'MgridBase': np.array([0.5,1.5]),
                      'aNrmInitMean' : np.log(0.00001),#gets overidden with much smaller number
                      'aNrmInitStd' : 0.0,
                      'pLvlInitMean' : 0.0,
                      'pLvlInitStd' : 0.0,
                      'PermGroFacAgg' : 1.0,
                      'UpdatePrb' : UpdatePrb,
                      'T_age' : None,
                      'T_cycle' : 1,
                      'cycles' : 0,
                      'T_sim' : periods_to_sim
                    }
                    

init_RA_consumer =  { 'CRRA': 2.0,
                      'DiscFac': 1.0/1.0146501772118186,
                      'LivPrb': [1.0],
                      'PermGroFac': [1.0],
                      'AgentCount': 1,
                      'aXtraMin': 0.00001,
                      'aXtraMax': 80.0,
                      'aXtraNestFac': 3,
                      'aXtraCount': 48,
                      'aXtraExtra': [None],
                      'PermShkStd': [np.sqrt(0.00004)],
                      'PermShkCount': 7,
                      'TranShkStd': [np.sqrt(0.00001)],
                      'TranShkCount': 7,
                      'UnempPrb': 0.0,
                      'UnempPrbRet': 0.0,
                      'IncUnemp': 0.0,
                      'IncUnempRet': 0.0,
                      'BoroCnstArt':0.0,
                      'tax_rate':0.0,
                      'T_retire':0,
                      'aNrmInitMean' : np.log(0.00001),
                      'aNrmInitStd' : 0.0,
                      'pLvlInitMean' : 0.0,
                      'pLvlInitStd' : 0.0,
                      'PermGroFacAgg' : 1.0,
                      'UpdatePrb' : UpdatePrb,
                      'CapShare' : 0.36,
                      'DeprFac' : 1.0 - 0.94**(0.25),
                      'SocPlannerBool' : False,
                      'T_age' : None,
                      'T_cycle' : 1,
                      'T_sim' : periods_to_sim
                    }
                    
init_SOE_market = {  'PermShkAggCount': 3,
                     'TranShkAggCount': 3,
                     'PermShkAggStd': np.sqrt(0.00004),
                     'TranShkAggStd': np.sqrt(0.00001),
                     'PermGroFacAgg': 1.0,
                     'DeprFac': 1.0 - 0.94**(0.25),
                     'CapShare': 0.36,
                     'Rfree': 1.014189682528173,
                     'wRte': 2.5895209258224536,
                     'act_T': periods_to_sim
                     }


# Make a small open economy and the consumers who live in it
StickySOEconsumers     = StickyEconsumerSOEType(**init_SOE_consumer)
StickySOEconomy        = SmallOpenEconomy(**init_SOE_market)
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

#plt.plot(np.mean(StickySOEconsumers.aLvlNow_hist,axis=1))
#plt.ylabel('Aggregate assets')
#plt.show()
#
#plt.plot(np.mean(StickySOEconsumers.mNrmNow_hist*StickySOEconsumers.pLvlNow_hist,axis=1))
#plt.ylabel('Aggregate market resources')
#plt.show()
#
#plt.plot(np.mean(StickySOEconsumers.cLvlNow_hist,axis=1))
#plt.ylabel('Aggregate consumption')
#plt.show()
#
#plt.plot(np.mean(StickySOEconsumers.pLvlNow_hist,axis=1))
#plt.plot(np.mean(StickySOEconsumers.pLvlErrNow_hist,axis=1))
#plt.show()

print('Average aggregate assets = ' + str(np.mean(StickySOEconsumers.aLvlNow_hist[ignore_periods:,:])))
print('Average aggregate consumption = ' + str(np.mean(StickySOEconsumers.cLvlNow_hist[ignore_periods:,:])))
print('Standard deviation of log aggregate assets = ' + str(np.std(np.log(np.mean(StickySOEconsumers.aLvlNow_hist[ignore_periods:,:],axis=1)))))
LogC = np.log(np.mean(StickySOEconsumers.cLvlNow_hist,axis=1))[ignore_periods:]
DeltaLogC = LogC[1:] - LogC[0:-1]
print('Standard deviation of change in log aggregate consumption = ' + str(np.std(DeltaLogC)))
print('Standard deviation of log individual assets = ' + str(np.mean(np.std(np.log(StickySOEconsumers.aLvlNow_hist[ignore_periods:,:]),axis=1))))
print('Standard deviation of log individual consumption = ' + str(np.mean(np.std(np.log(StickySOEconsumers.cLvlNow_hist[ignore_periods:,:]),axis=1))))
print('Standard deviation of log individual productivity = ' + str(np.mean(np.std(np.log(StickySOEconsumers.pLvlNow_hist[ignore_periods:,:]),axis=1))))
Logc = np.log(StickySOEconsumers.cLvlNow_hist)[ignore_periods:,:]
DeltaLogc = Logc[1:,:] - Logc[0:-1,:]
print('Standard deviation of change in log individual consumption = ' + str(np.std(DeltaLogc.flatten())))


# Make a representative agent consumer, then solve and simulate the model
RAconsumer = StickyEconsumerDSGEType(**init_RA_consumer)
RAconsumer.solve()
RAconsumer.track_vars = ['cNrmNow','cLvlNow','aNrmNow','pLvlNow','yNrmNow','aLvlNow','pLvlErrNow']
RAconsumer.initializeSim()
RAconsumer.simulate()

plotFuncs(RAconsumer.solution[0].cFunc,0,20)

pLvlTrue_hist = RAconsumer.pLvlNow_hist*RAconsumer.pLvlErrNow_hist
print('Average aggregate assets = ' + str(np.mean(RAconsumer.aLvlNow_hist[ignore_periods:,:]/pLvlTrue_hist[ignore_periods:,:])))
print('Average aggregate consumption = ' + str(np.mean(RAconsumer.cLvlNow_hist[ignore_periods:,:]/pLvlTrue_hist[ignore_periods:,:])))
print('Standard deviation of log aggregate assets = ' + str(np.std(np.log(RAconsumer.aLvlNow_hist[ignore_periods:,:]/pLvlTrue_hist[ignore_periods:,:]))))
LogC = np.log(np.mean(RAconsumer.cLvlNow_hist,axis=1))[ignore_periods:]
DeltaLogC = LogC[1:] - LogC[0:-1]
print('Standard deviation of change in log aggregate consumption = ' + str(np.std(DeltaLogC)))
LogY = np.log(np.mean(RAconsumer.yNrmNow_hist,axis=1))[ignore_periods:]
DeltaLogY = LogY[1:] - LogY[0:-1]
print('Standard deviation of change in log aggregate output = ' + str(np.std(DeltaLogY)))

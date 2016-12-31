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
from ConsAggShockModel import SmallOpenEconomy, CobbDouglasEconomy
from HARKutilities import plotFuncs
import matplotlib.pyplot as plt

periods_to_sim = 1000
ignore_periods = 100

# Define parameters for the small open economy version of the model
init_SOE_consumer = { 'CRRA': 2.0,
                      'DiscFac': 0.969/0.995,
                      'LivPrb': [0.995],
                      'PermGroFac': [1.0],
                      'AgentCount': 10000,
                      'aXtraMin': 0.00001,
                      'aXtraMax': 40.0,
                      'aXtraNestFac': 3,
                      'aXtraCount': 48,
                      'aXtraExtra': [None],
                      'PermShkStd': [np.sqrt(0.003)],
                      'PermShkCount': 5,
                      'TranShkStd': [np.sqrt(0.12)],
                      'TranShkCount': 5,
                      'UnempPrb': 0.05,
                      'UnempPrbRet': 0.0,
                      'IncUnemp': 0.0,
                      'IncUnempRet': 0.0,
                      'BoroCnstArt':0.0,
                      'tax_rate':0.0,
                      'T_retire':0,
                      'kGridBase': np.array([0.5,1.5]),
                      'aNrmInitMean' : np.log(0.00001),
                      'aNrmInitStd' : 0.0,
                      'pLvlInitMean' : 0.0,
                      'pLvlInitStd' : 0.0,
                      'PermGroFacAgg' : 1.0,
                      'UpdatePrb' : 0.25,
                      'T_age' : None,
                      'T_cycle' : 1,
                      'cycles' : 0,
                      'T_sim' : periods_to_sim
                    }
                    
init_DSGE_consumer = { 'CRRA': 2.0,
                      'DiscFac': 1.0/1.014189682528173,
                      'LivPrb': [1.0],
                      'PermGroFac': [1.0],
                      'AgentCount': 1,
                      'aXtraMin': 0.00001,
                      'aXtraMax': 40.0,
                      'aXtraNestFac': 3,
                      'aXtraCount': 48,
                      'aXtraExtra': [None],
                      'PermShkStd': [0.0],
                      'PermShkCount': 1,
                      'TranShkStd': [0.0],
                      'TranShkCount': 1,
                      'UnempPrb': 0.0,
                      'UnempPrbRet': 0.0,
                      'IncUnemp': 0.0,
                      'IncUnempRet': 0.0,
                      'BoroCnstArt':0.0,
                      'tax_rate':0.0,
                      'T_retire':0,
                      'kGridBase': np.array([0.1,0.3,0.6,0.8,0.9,0.98,1.0,1.02,1.1,1.2,1.6,2.0,3.0]),
                      'aNrmInitMean' : np.log(0.00001),
                      'aNrmInitStd' : 0.0,
                      'pLvlInitMean' : 0.0,
                      'pLvlInitStd' : 0.0,
                      'PermGroFacAgg' : 1.0,
                      'UpdatePrb' : 0.25,
                      'CapShare' : 0.36,
                      'T_age' : None,
                      'T_cycle' : 1,
                      'cycles' : 0,
                      'T_sim' : periods_to_sim
                    }
                    
init_SOE_market = {  'PermShkAggCount': 3,
                     'TranShkAggCount': 3,
                     'PermShkAggStd': np.sqrt(0.00004),
                     'TranShkAggStd': np.sqrt(0.000004),
                     'DeprFac': 1.0 - 0.94**(0.25),
                     'CapShare': 0.36,
                     'Rfree': 1.014189682528173,
                     'wRte': 2.5895209258224536,
                     'act_T': periods_to_sim
                     }
                     
init_DSGE_market = { 'PermShkAggCount': 7,
                     'TranShkAggCount': 7,
                     'PermShkAggStd': np.sqrt(0.00004),
                     'TranShkAggStd': np.sqrt(0.000004),
                     'DeprFac': 1.0 - 0.94**(0.25),
                     'CapShare': 0.36,
                     'CRRA': 2.0,
                     'DiscFac': 1.0/1.014189682528173,
                     'slope_prev': 1.0,
                     'intercept_prev': 0.0,
                     'kSS':12.0**(1.0/(1.0-0.36)),
                     'AggregateL': 1.0,
                     'ignore_periods':ignore_periods,
                     'tolerance':0.0001,
                     'act_T': periods_to_sim
                     }


# Make a small open economy and the consumers who live in it
StickySOEconsumers = StickyEconsumerSOEType(**init_SOE_consumer)
StickySOEconomy = SmallOpenEconomy(**init_SOE_market)
StickySOEconomy.agents = [StickySOEconsumers]
StickySOEconomy.makeAggShkHist()
StickySOEconsumers.getEconomyData(StickySOEconomy)
StickySOEconsumers.track_vars = ['aLvlNow','mNrmNow','cNrmNow','pLvlNow','pLvlErrNow']

# Solve the model and display some output
StickySOEconomy.solveAgents()
StickySOEconomy.makeHistory()

# Plot some of the results
cFunc = lambda m : StickySOEconsumers.solution[0].cFunc(m,np.ones_like(m))
plotFuncs(cFunc,0.0,20.0)

plt.plot(np.mean(StickySOEconsumers.aLvlNow_hist,axis=1))
plt.show()

plt.plot(np.mean(StickySOEconsumers.mNrmNow_hist,axis=1))
plt.show()

plt.plot(np.mean(StickySOEconsumers.cNrmNow_hist*StickySOEconsumers.pLvlNow_hist,axis=1))
plt.show()

plt.plot(np.mean(StickySOEconsumers.pLvlNow_hist,axis=1))
plt.plot(np.mean(StickySOEconsumers.pLvlErrNow_hist,axis=1))
plt.show()

print('Average aggregate assets = ' + str(np.mean(StickySOEconsumers.aLvlNow_hist[ignore_periods:,:])))
print('Average aggregate consumption = ' + str(np.mean(StickySOEconsumers.cNrmNow_hist[ignore_periods:,:]*StickySOEconsumers.pLvlNow_hist[ignore_periods:,:])))
print('Standard deviation of log aggregate assets = ' + str(np.std(np.log(np.mean(StickySOEconsumers.aLvlNow_hist[ignore_periods:,:],axis=1)))))
LogC = np.log(np.mean(StickySOEconsumers.cNrmNow_hist*StickySOEconsumers.pLvlNow_hist,axis=1))[ignore_periods:]
DeltaLogC = LogC[1:] - LogC[0:-1]
print('Standard deviation of change in log aggregate consumption = ' + str(np.std(DeltaLogC)))
print('Standard deviation of log individual assets = ' + str(np.mean(np.std(np.log(StickySOEconsumers.aLvlNow_hist[ignore_periods:,:]),axis=1))))
print('Standard deviation of log individual consumption = ' + str(np.mean(np.std(np.log(StickySOEconsumers.cNrmNow_hist[ignore_periods:,:]*StickySOEconsumers.pLvlNow_hist[ignore_periods:,:]),axis=1))))
print('Standard deviation of log individual productivity = ' + str(np.mean(np.std(np.log(StickySOEconsumers.pLvlNow_hist[ignore_periods:,:]),axis=1))))
Logc = np.log(StickySOEconsumers.cNrmNow_hist*StickySOEconsumers.pLvlNow_hist)[ignore_periods:,:]
DeltaLogc = Logc[1:,:] - Logc[0:-1,:]
print('Standard deviation of change in log individual consumption = ' + str(np.mean(np.std(DeltaLogc,axis=1))))


# Make a Cobb Douglas economy and the representative agent who lives in it
StickyDSGEconsumer = StickyEconsumerDSGEType(**init_DSGE_consumer)
StickyDSGEeconomy = CobbDouglasEconomy(**init_DSGE_market)
StickyDSGEeconomy.agents = [StickyDSGEconsumer]
StickyDSGEeconomy.makeAggShkHist()
StickyDSGEconsumer.getEconomyData(StickyDSGEeconomy)
StickyDSGEconsumer.track_vars = ['aLvlNow','mNrmNow','cNrmNow','pLvlNow','pLvlErrNow']

# Test the solution
StickyDSGEeconomy.solve()

m_grid = np.linspace(0,10,200)
for k in StickyDSGEconsumer.kGrid.tolist():
    c_at_this_k = StickyDSGEconsumer.solution[0].cFunc(m_grid,k*np.ones_like(m_grid))
    plt.plot(m_grid,c_at_this_k)
plt.show()

'''
Runs the exercises and regressions for the cAndCwithStickyE paper.
'''
import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../ConsumptionSaving'))

import numpy as np
#from copy import copy, deepcopy
from StickyEMarkovmodel import StickyEMarkovSOEType
from ConsMarkovModel import MarkovSmallOpenEconomy
from ConsIndShockModel import IndShockConsumerType, constructLognormalIncomeProcessUnemployment
from HARKutilities import plotFuncs, approxMeanOneLognormal
import matplotlib.pyplot as plt

periods_to_sim = 1200
ignore_periods = 500

# Define parameters for the small open economy version of the model
init_MarkovSOE_consumer = { 'CRRA': 2.0,
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
                      'aNrmInitMean' : np.log(0.00001),
                      'aNrmInitStd' : 0.0,
                      'pLvlInitMean' : 0.0,
                      'pLvlInitStd' : 0.0,
                      'PermGroFacAgg' : 1.0,
                      'UpdatePrb' : 0.25,
                      'T_age' : None,
                      'T_cycle' : 1,
                      'cycles' : 0,
                      'T_sim' : periods_to_sim,
                      'vFuncBool' : False,
                      'CubicBool' : False
                    }
                    
#Need to get the income distribution of a standard ConsIndShockModel consumer
StateCount = 5
Persistence = 0.5
TranShkAggStd = 0.0031
TranShkAggCount = 7
DummyForIncomeDstn = IndShockConsumerType(**init_MarkovSOE_consumer)
IncomeDstn, PermShkDstn, TranShkDstn = constructLognormalIncomeProcessUnemployment(DummyForIncomeDstn)
IncomeDstn = StateCount*IncomeDstn
# Make the state transition array for this type: Persistence probability of remaining in the same state, equiprobable otherwise
MrkvArray = Persistence*np.eye(StateCount) + (1.0/StateCount)*(1.0-Persistence)*np.ones((StateCount,StateCount))
init_MarkovSOE_consumer['MrkvArray'] = [MrkvArray]
init_MarkovSOE_consumer['PermGroFac'] = [np.array([0.99,0.995,1.0,1.005,1.01])]
init_MarkovSOE_consumer['LivPrb'] = [np.array(StateCount*[0.995])]
                        
TranShkAggDstn = approxMeanOneLognormal(sigma=TranShkAggStd,N=TranShkAggCount)   
init_MarkovSOE_market = {  
                     'Rfree': np.array(np.array(StateCount*[1.014189682528173])),
                     'act_T': periods_to_sim,
                     'MrkvArray':[MrkvArray],
                     'MrkvPrbsInit':StateCount*[1.0/StateCount],
                     'MktMrkvNow_init':StateCount/2,
                     'aSS':2.0,
                     'TranShkAggDstn':TranShkAggDstn,
                     'TranShkAggNow_init':1.0
                     }
                     
# Make a small open economy and the consumers who live in it
StickyMarkovSOEconsumers     = StickyEMarkovSOEType(**init_MarkovSOE_consumer)
StickyMarkovSOEconsumers.assignParameters(cycles = 0)   # For some reason need to set the explicitly
StickyMarkovSOEconsumers.IncomeDstn = [IncomeDstn] 
StickyMarkovSOEconomy        = MarkovSmallOpenEconomy(**init_MarkovSOE_market)
StickyMarkovSOEconomy.agents = [StickyMarkovSOEconsumers]
StickyMarkovSOEconomy.makeMkvShkHist()
StickyMarkovSOEconsumers.getEconomyData(StickyMarkovSOEconomy)
StickyMarkovSOEconsumers.track_vars = ['aLvlNow','mNrmNow','cNrmNow','pLvlNow','pLvlErrNow','MrkvNow']

# Solve the model and display some output
StickyMarkovSOEconomy.solve()

# Plot some of the results
plotFuncs(StickyMarkovSOEconsumers.solution[0].cFunc,0,10)

plt.plot(np.mean(StickyMarkovSOEconsumers.aLvlNow_hist,axis=1))
plt.show()

plt.plot(np.mean(StickyMarkovSOEconsumers.mNrmNow_hist*StickyMarkovSOEconsumers.pLvlNow_hist,axis=1))
plt.show()

plt.plot(np.mean(StickyMarkovSOEconsumers.cNrmNow_hist*StickyMarkovSOEconsumers.pLvlNow_hist,axis=1))
plt.show()

plt.plot(np.mean(StickyMarkovSOEconsumers.pLvlNow_hist,axis=1))
plt.plot(np.mean(StickyMarkovSOEconsumers.pLvlErrNow_hist,axis=1))
plt.show()

print('Average aggregate assets = ' + str(np.mean(StickyMarkovSOEconsumers.aLvlNow_hist[ignore_periods:,:])))
print('Average aggregate consumption = ' + str(np.mean(StickyMarkovSOEconsumers.cNrmNow_hist[ignore_periods:,:]*StickyMarkovSOEconsumers.pLvlNow_hist[ignore_periods:,:])))
print('Standard deviation of log aggregate assets = ' + str(np.std(np.log(np.mean(StickyMarkovSOEconsumers.aLvlNow_hist[ignore_periods:,:],axis=1)))))
LogC = np.log(np.mean(StickyMarkovSOEconsumers.cNrmNow_hist*StickyMarkovSOEconsumers.pLvlNow_hist,axis=1))[ignore_periods:]
DeltaLogC = LogC[1:] - LogC[0:-1]
print('Standard deviation of change in log aggregate consumption = ' + str(np.std(DeltaLogC)))
print('Standard deviation of log individual assets = ' + str(np.mean(np.std(np.log(StickyMarkovSOEconsumers.aLvlNow_hist[ignore_periods:,:]),axis=1))))
print('Standard deviation of log individual consumption = ' + str(np.mean(np.std(np.log(StickyMarkovSOEconsumers.cNrmNow_hist[ignore_periods:,:]*StickyMarkovSOEconsumers.pLvlNow_hist[ignore_periods:,:]),axis=1))))
print('Standard deviation of log individual productivity = ' + str(np.mean(np.std(np.log(StickyMarkovSOEconsumers.pLvlNow_hist[ignore_periods:,:]),axis=1))))
Logc = np.log(StickyMarkovSOEconsumers.cNrmNow_hist*StickyMarkovSOEconsumers.pLvlNow_hist)[ignore_periods:,:]
DeltaLogc = Logc[1:,:] - Logc[0:-1,:]
print('Standard deviation of change in log individual consumption = ' + str(np.mean(np.std(DeltaLogc,axis=1))))

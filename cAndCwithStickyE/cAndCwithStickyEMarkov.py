'''
Runs the exercises and regressions for the cAndCwithStickyE paper.
'''
import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../ConsumptionSaving'))

import numpy as np
import scipy.linalg as linalg
#from copy import copy, deepcopy
from StickyEMarkovmodel import StickyEMarkovSOEType
from ConsMarkovModel import MarkovSmallOpenEconomy
from ConsIndShockModel import IndShockConsumerType, constructLognormalIncomeProcessUnemployment
from HARKutilities import plotFuncs, approxMeanOneLognormal, TauchenAR1
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.sandbox.regression.gmm as smsrg

periods_to_sim = 3500
ignore_periods = 1000

# Define parameters for the small open economy version of the model
init_MarkovSOE_consumer = { 'CRRA': 2.0,
                      'DiscFac': 0.969,
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
                      'PermGroFacAgg' : 1.0,#+0.015/4.0,
                      'UpdatePrb' : 0.25,
                      'T_age' : None,
                      'T_cycle' : 1,
                      'cycles' : 0,
                      'T_sim' : periods_to_sim,
                      'vFuncBool' : False,
                      'CubicBool' : False
                    }
                    
TranShkAggStd = np.sqrt(0.00001)
TranShkAggCount = 7
LivPrb = 0.995
Rfree = 1.014189682528173

##Markov Parameters follow an AR1
#StateCount = 7
#rho = 0.8
#sigma = np.sqrt(0.00004*(1-rho**2))  #ergodic distribution has same standard deviation as in the rho=0 case
#m=4
#PermGroFac, MrkvArray = TauchenAR1(sigma, rho, StateCount, m)
#PermGroFac = [PermGroFac+1.0]

#Markov Parameters follow a random walk bounded above and below
StateCount = 21
dont_move_prob = 0.0
MrkvArray = linalg.toeplitz([dont_move_prob,(1.0-dont_move_prob)/2.0]+[0.0]*(StateCount-2), [dont_move_prob,(1.0-dont_move_prob)/2.0]+[0.0]*(StateCount-2))
MrkvArray[0,1] = (1.0-dont_move_prob)
MrkvArray[StateCount-1,StateCount-2] = (1.0-dont_move_prob)
PermGroFac = [1.0+np.linspace(-0.015/4.0,0.015/4.0,StateCount)]

###############################################################################

#Need to get the income distribution of a standard ConsIndShockModel consumer
DummyForIncomeDstn = IndShockConsumerType(**init_MarkovSOE_consumer)
IncomeDstn, PermShkDstn, TranShkDstn = constructLognormalIncomeProcessUnemployment(DummyForIncomeDstn)
IncomeDstn = StateCount*IncomeDstn
# Make the state transition array for this type: Persistence probability of remaining in the same state, equiprobable otherwise
init_MarkovSOE_consumer['MrkvArray'] = [MrkvArray]
init_MarkovSOE_consumer['PermGroFac'] = PermGroFac
init_MarkovSOE_consumer['LivPrb'] = [np.array(StateCount*[LivPrb])]
                        
TranShkAggDstn = approxMeanOneLognormal(sigma=TranShkAggStd,N=TranShkAggCount)   
init_MarkovSOE_market = {  
                     'Rfree': np.array(np.array(StateCount*[Rfree])),
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
StickyMarkovSOEconsumers.aNrmInitMean = np.log(1.0)  #Don't want newborns to have no assets and also be unemployed
StickyMarkovSOEconsumers.track_vars = ['aLvlNow','mNrmNow','cNrmNow','pLvlNow','pLvlErrNow','MrkvNow','TranShkAggNow']

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

print('Standard deviation of log aggregate permanent income = ' + str(np.std(np.log(np.mean(StickyMarkovSOEconsumers.pLvlNow_hist[ignore_periods:,:],axis=1)))))
print('Standard deviation of log aggregate  income = ' + str(np.std(np.log(np.mean(StickyMarkovSOEconsumers.TranShkAggNow_hist[ignore_periods:,:]*StickyMarkovSOEconsumers.pLvlNow_hist[ignore_periods:,:],axis=1)))))


# Do aggregate regressions
LogY = np.log(np.mean(StickyMarkovSOEconsumers.TranShkAggNow_hist*StickyMarkovSOEconsumers.pLvlNow_hist/StickyMarkovSOEconsumers.pLvlErrNow_hist,axis=1))[ignore_periods:]
DeltaLogY = LogY[1:] - LogY[0:-1]
A = np.mean(StickyMarkovSOEconsumers.aLvlNow_hist,axis=1)[ignore_periods+1:]

#OLS on log consumption (no measurement error)
mod = sm.OLS(DeltaLogC[1:],sm.add_constant(DeltaLogC[0:-1]))
res1 = mod.fit()
print(res1.summary())

#Add measurement error to LogC
sigma_me = np.std(DeltaLogC)/1.5
np.random.seed(10)
LogC_me = LogC + sigma_me*np.random.normal(0,1,len(LogC))
DeltaLogC_me = LogC_me[1:] - LogC_me[0:-1]

#OLS on log consumption (with measurement error)
mod = sm.OLS(DeltaLogC_me[1:],sm.add_constant(DeltaLogC_me[0:-1]))
res2 = mod.fit()
print(res2.summary())

instruments = sm.add_constant(np.transpose(np.array([DeltaLogC_me[1:-3],DeltaLogC_me[:-4],DeltaLogY[1:-3],DeltaLogY[:-4],A[1:-3],A[:-4]])))
#IV on log consumption (with measurement error)
mod = smsrg.IV2SLS(DeltaLogC_me[4:],sm.add_constant(DeltaLogC_me[3:-1]),instruments)
res3 = mod.fit()
print(res3.summary())

#IV on log income (with measurement error)
mod = smsrg.IV2SLS(DeltaLogC_me[4:],sm.add_constant(DeltaLogY[4:]),instruments)
res4 = mod.fit()
print(res4.summary())

#IV on assets (with measurement error)
mod = smsrg.IV2SLS(DeltaLogC_me[4:],sm.add_constant(A[3:-1]),instruments)
res5 = mod.fit()
print(res5.summary())

#Horserace IV (with measurement error)
regressors = sm.add_constant(np.transpose(np.array([DeltaLogC_me[3:-1],DeltaLogY[4:],A[3:-1]])))
mod = smsrg.IV2SLS(DeltaLogC_me[4:],regressors,instruments)
res6 = mod.fit()
print(res6.summary())

#Also report frictionless results with no measurement error
instruments2 = sm.add_constant(np.transpose(np.array([DeltaLogC[1:-3],DeltaLogC[:-4],DeltaLogY[1:-3],DeltaLogY[:-4],A[1:-3],A[:-4]])))
#IV on log consumption (with measurement error)
mod = smsrg.IV2SLS(DeltaLogC[4:],sm.add_constant(DeltaLogY[4:]),instruments2)
res7 = mod.fit()

mod = smsrg.IV2SLS(DeltaLogC[4:],sm.add_constant(A[3:-1]),instruments2)
res8 = mod.fit()

#Horserace IV (with no measurement error)
regressors = sm.add_constant(np.transpose(np.array([DeltaLogC[3:-1],DeltaLogY[4:],A[3:-1]])))
mod = smsrg.IV2SLS(DeltaLogC[4:],regressors,instruments2)
res9 = mod.fit()

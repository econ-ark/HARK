'''
This module holds calibrated parameter dictionaries for the cAndCwithStickyE paper.
It defines dictionaries for the six types of models in cAndCwithStickyE:
    
1) Small open economy
2) Small open Markov economy
3) Cobb-Douglas closed economy
4) Cobb-Douglas closed Markov economy
5) Representative agent economy
6) Markov representative agent economy
    
For the first four models (heterogeneous agents), it defines dictionaries for
the Market instance as well as the consumers themselves.  All parameters are quarterly.
'''
import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
import numpy as np
from copy import copy
from HARKutilities import approxUniform, approxMeanOneLognormal

# Choose file where the Stata executable can be found.  This should point at the
# exe file itself, but the string does not need to include '.exe'.  Two examples
# are included (for locations on two authors' local computers).  This variable
# is irrelevant when the use_stata boolean in cAndCwithStickyE.py is set to False.
# Using Stata to run the regressions allows the tables to include the KP test
# statistic; Python's statsmodels.api currently does not have this functionality.

# NOTE: To successfully use_stata, you must have Baum, Schaffer, and Stillman's
# ivreg2 Stata module installed, as well as Kleibergen and Schaffer's ranktest
# module.  These modules are archived by RePEc IDEAS at:
# https://ideas.repec.org/c/boc/bocode/s425401.html
# https://ideas.repec.org/c/boc/bocode/s456865.html
# You can also simply type "ssc install ivreg2" and "ssc install ranktest" in Stata.

#stata_exe = "C:\Program Files (x86)\Stata14\stataMP-64"
stata_exe = "C:\Program Files (x86)\Stata15\StataSE-64"

# Choose directory paths relative to the StickyE files
calibration_dir = "./Calibration/" # Relative directory for primitive parameter files
tables_dir = "./Tables/"   # Relative directory for saving tex tables
results_dir = "./Results/" # Relative directory for saving output files
figures_dir = "./Figures/" # Relative directory for saving figures

def importParam(param_name):
    return np.max(np.genfromtxt(calibration_dir + param_name + '.txt'))

# Import primitive parameters from calibrations folder
CRRA = importParam('CRRA')             # Coefficient of relative risk aversion
DeprFacAnn = importParam('DeprFacAnn') # Annual depreciation factor
CapShare = importParam('CapShare')     # Capital's share in production function
KYratioSS = importParam('KYratioSS')   # Steady state capital to output ratio (PF-DSGE)
UpdatePrb = importParam('UpdatePrb')   # Probability that each agent observes the aggregate productivity state each period (in sticky version)
UnempPrb = importParam('UnempPrb')     # Unemployment probability
DiePrb = importParam('DiePrb')         # Quarterly mortality probability
TranShkVarAnn = importParam('TranShkVarAnn')    # Annual variance of idiosyncratic transitory shocks
PermShkVarAnn = importParam('PermShkVarAnn')    # Annual variance of idiosyncratic permanent shocks
TranShkAggVar = importParam('TranShkAggVar') # Variance of aggregate transitory shocks
PermShkAggVar = importParam('PermShkAggVar') # Variance of aggregate permanent shocks

# Calculate parameters based on the primitive parameters
DeprFac = 1. - DeprFacAnn**0.25                  # Quarterly depreciation rate
KSS = KtYratioSS = KYratioSS**(1./(1.-CapShare)) # Steady state Capital to labor productivity
wRteSS = (1.-CapShare)*KSS**CapShare             # Steady state wage rate
rFreeSS = CapShare*KSS**(CapShare-1.)            # Steady state interest rate
RfreeSS = 1. - DeprFac + rFreeSS                 # Steady state return factor
LivPrb = 1. - DiePrb                             # Quarterly survival probability
DiscFacDSGE = RfreeSS**(-1)                      # Discount factor, HA-DSGE and RA models
TranShkVar = TranShkVarAnn*4.                    # Variance of idiosyncratic transitory shocks
PermShkVar = PermShkVarAnn/4.                    # Variance of idiosyncratic permanent shocks
TempDstn = approxMeanOneLognormal(N=7,sigma=np.sqrt(PermShkVar))
DiscFacSOE = 0.99*LivPrb/(RfreeSS*np.dot(TempDstn[0],TempDstn[1]**(-CRRA))) # Discount factor, SOE model

# Choose basic simulation parameters
periods_to_sim = 21010 # Total number of periods to simulate; this might be increased by DSGEmarkov model
ignore_periods = 1000  # Number of simulated periods to ignore (in order to ensure we are near steady state)
interval_size = 200    # Number of periods in each subsample interval
AgentCount = 20000     # Total number of agents to simulate in the economy

# Use smaller sample for micro regression tables to save memory
periods_to_sim_micro = 4000
AgentCount_micro = 5000

# Choose extent of discount factor heterogeneity (inapplicable to representative agent models)
TypeCount = 1           # Number of heterogeneous discount factor types
DiscFacMeanSOE  = DiscFacSOE # Central value of intertemporal discount factor for SOE model
DiscFacMeanDSGE = DiscFacDSGE  # ...for HA-DSGE and RA
DiscFacSpread = 0.0     # Half-width of intertemporal discount factor band, a la cstwMPC

# These parameters are for a rough "beta-dist" specification that fits the wealth distribution in DSGE simple
#TypeCount = 7
#DiscFacMeanSOE = 0.96738
#DiscFacMeanDSGE = 0.96738  
#DiscFacSpread = 0.0227 

# Choose parameters for the Markov models
StateCount = 11         # Number of discrete states in the Markov specifications
PermGroFacMin = 0.9925  # Minimum value of aggregate permanent growth in Markov specifications
PermGroFacMax = 1.0075  # Maximum value of aggregate permanent growth in Markov specifications
Persistence = 0.5       # Base probability that macroeconomic Markov state stays the same; else moves up or down by 1
RegimeChangePrb = 0.00  # Probability of "regime change", randomly jumping to any Markov state

# Make the Markov array with chosen states, persistence, and regime change probability
PolyMrkvArray = np.zeros((StateCount,StateCount))
for i in range(StateCount):
    for j in range(StateCount):
        if i==j:
            PolyMrkvArray[i,j] = Persistence
        elif (i==(j-1)) or (i==(j+1)):
            PolyMrkvArray[i,j] = 0.5*(1.0 - Persistence)
PolyMrkvArray[0,0] += 0.5*(1.0 - Persistence)
PolyMrkvArray[StateCount-1,StateCount-1] += 0.5*(1.0 - Persistence)
PolyMrkvArray *= 1.0 - RegimeChangePrb
PolyMrkvArray += RegimeChangePrb/StateCount

# Define the set of aggregate permanent growth factors that can occur (Markov specifications only)
PermGroFacSet = np.exp(np.linspace(np.log(PermGroFacMin),np.log(PermGroFacMax),num=StateCount))

# Define the set of discount factors that agents have (for SOE and DSGE models)
DiscFacSetSOE  = approxUniform(N=TypeCount,bot=DiscFacMeanSOE-DiscFacSpread,top=DiscFacMeanSOE+DiscFacSpread)[1]
DiscFacSetDSGE = approxUniform(N=TypeCount,bot=DiscFacMeanDSGE-DiscFacSpread,top=DiscFacMeanDSGE+DiscFacSpread)[1]

###############################################################################

# Define parameters for the small open economy version of the model
init_SOE_consumer = { 'CRRA': CRRA,
                      'DiscFac': DiscFacMeanSOE,
                      'LivPrb': [LivPrb],
                      'PermGroFac': [1.0],
                      'AgentCount': AgentCount/TypeCount, # Spread agents evenly among types
                      'aXtraMin': 0.00001,
                      'aXtraMax': 40.0,
                      'aXtraNestFac': 3,
                      'aXtraCount': 48,
                      'aXtraExtra': [None],
                      'PermShkStd': [np.sqrt(PermShkVar)],
                      'PermShkCount': 7,
                      'TranShkStd': [np.sqrt(TranShkVar)],
                      'TranShkCount': 7,
                      'UnempPrb': UnempPrb,
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
                      'UpdatePrb' : UpdatePrb,
                      'T_age' : None,
                      'T_cycle' : 1,
                      'cycles' : 0,
                      'T_sim' : periods_to_sim
                    }

# Define market parameters for the small open economy
init_SOE_market = {  'PermShkAggCount': 5,
                     'TranShkAggCount': 5,
                     'PermShkAggStd': np.sqrt(PermShkAggVar),
                     'TranShkAggStd': np.sqrt(TranShkAggVar),
                     'PermGroFacAgg': 1.0,
                     'DeprFac': DeprFac,
                     'CapShare': CapShare,
                     'Rfree': RfreeSS,
                     'wRte': wRteSS,
                     'act_T': periods_to_sim,
                     }

###############################################################################

# Define parameters for the small open Markov economy version of the model
init_SOE_mrkv_consumer = copy(init_SOE_consumer)
init_SOE_mrkv_consumer['MrkvArray'] = PolyMrkvArray

# Define market parameters for the small open Markov economy
init_SOE_mrkv_market = copy(init_SOE_market)
init_SOE_mrkv_market['MrkvArray'] = PolyMrkvArray
init_SOE_mrkv_market['PermShkAggStd'] = StateCount*[init_SOE_market['PermShkAggStd']]
init_SOE_mrkv_market['TranShkAggStd'] = StateCount*[init_SOE_market['TranShkAggStd']]
init_SOE_mrkv_market['PermGroFacAgg'] = PermGroFacSet
init_SOE_mrkv_market['MrkvNow_init'] = StateCount/2
init_SOE_mrkv_market['loops_max'] = 1

###############################################################################

# Define parameters for the Cobb-Douglas DSGE version of the model
init_DSGE_consumer = copy(init_SOE_consumer)
init_DSGE_consumer['DiscFac'] = DiscFacMeanDSGE
init_DSGE_consumer['aXtraMax'] = 120.0
init_DSGE_consumer['MgridBase'] = np.array([0.1,0.3,0.5,0.6,0.7,0.8,0.9,0.98,1.0,1.02,1.1,1.2,1.3,1.4,1.5,1.6,2.0,3.0,5.0])

# Define market parameters for the Cobb-Douglas economy
init_DSGE_market = copy(init_SOE_market)
init_DSGE_market.pop('Rfree')
init_DSGE_market.pop('wRte')
init_DSGE_market['CRRA'] = CRRA
init_DSGE_market['DiscFac'] = DiscFacMeanDSGE
init_DSGE_market['intercept_prev'] = 0.0
init_DSGE_market['slope_prev'] = 1.0

###############################################################################

# Define parameters for the Cobb-Douglas Markov DSGE version of the model
init_DSGE_mrkv_consumer = copy(init_DSGE_consumer)
init_DSGE_mrkv_consumer['MrkvArray'] = PolyMrkvArray  

# Define market parameters for the Cobb-Douglas Markov economy
init_DSGE_mrkv_market = copy(init_SOE_mrkv_market)
init_DSGE_mrkv_market.pop('Rfree')
init_DSGE_mrkv_market.pop('wRte')
init_DSGE_mrkv_market['CRRA'] = init_DSGE_mrkv_consumer['CRRA']
init_DSGE_mrkv_market['DiscFac'] = init_DSGE_mrkv_consumer['DiscFac']
init_DSGE_mrkv_market['intercept_prev'] = StateCount*[0.0]
init_DSGE_mrkv_market['slope_prev'] = StateCount*[1.0]
init_DSGE_mrkv_market['loops_max'] = 10

###############################################################################

# Define parameters for the representative agent version of the model
init_RA_consumer =  { 'CRRA': CRRA,
                      'DiscFac': DiscFacMeanDSGE,
                      'LivPrb': [1.0],
                      'PermGroFac': [1.0],
                      'AgentCount': 1,
                      'aXtraMin': 0.00001,
                      'aXtraMax': 120.0,
                      'aXtraNestFac': 3,
                      'aXtraCount': 48,
                      'aXtraExtra': [None],
                      'PermShkStd': [np.sqrt(PermShkAggVar)],
                      'PermShkCount': 7,
                      'TranShkStd': [np.sqrt(TranShkAggVar)],
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
                      'CapShare' : CapShare,
                      'DeprFac' : DeprFac,
                      'T_age' : None,
                      'T_cycle' : 1,
                      'T_sim' : periods_to_sim,
                      'tolerance' : 1e-6
                    }

###############################################################################

# Define parameters for the Markov representative agent model
init_RA_mrkv_consumer = copy(init_RA_consumer)
init_RA_mrkv_consumer['MrkvArray'] = PolyMrkvArray
init_RA_mrkv_consumer['MrkvNow'] = [StateCount/2]
init_RA_mrkv_consumer['PermGroFac'] = [PermGroFacSet]

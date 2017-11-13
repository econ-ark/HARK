'''
This module holds calibrated parameter dictionaries for the cAndCwithStickyE paper.
It defines dictionaries for the six types of models in cAndCwithStickyE:
    
1) Small open economy
2) Small open Markov economy
3) Cobb-Douglas economy
4) Cobb-Douglas Markov economy
5) Representative agent economy
6) Markov representative agent economy
    
For the first four models (heterogeneous agents), it defines dictionaries for
the Market instance as well as the consumers themselves.  All parameters are quarterly.
'''
import numpy as np
from copy import copy
from HARKutilities import approxUniform

# Choose basic simulation parameters
UpdatePrb = 0.25       # Probability that each agent observes the aggregate productivity state each period (in sticky version)
periods_to_sim = 21010 # Total number of periods to simulate; this might be increased by DSGEmarkov model
ignore_periods = 1000  # Number of simulated periods to ignore (in order to ensure we are near steady state)
interval_size = 20000  # Number of periods in each subsample interval
AgentCount = 20000     # Total number of agents to simulate in the economy

# Choose extent of discount factor heterogeneity (inapplicable to representative agent models)
TypeCount = 1        # Number of heterogeneous discount factor types
DiscFacMean = 0.969  # Central value of intertemporal discount factor
DiscFacSpread = 0.0  # Half-width of intertemporal discount factor band, a la cstwMPC

# These parameters are for a rough "beta-dist" specification that fits the wealth distribution in DSGE simple
#TypeCount = 7
#DiscFacMean = 0.96738  
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
PermGroFacSet = np.linspace(PermGroFacMin,PermGroFacMax,num=StateCount)

# Define the set of discount factors that agents have
DiscFacSet = approxUniform(N=TypeCount,bot=DiscFacMean-DiscFacSpread,top=DiscFacMean+DiscFacSpread)[1]

###############################################################################

# Define parameters for the small open economy version of the model
init_SOE_consumer = { 'CRRA': 2.0,
                      'DiscFac': DiscFacMean,
                      'LivPrb': [0.995],
                      'PermGroFac': [1.0],
                      'AgentCount': AgentCount/TypeCount, # Spread agents evenly among types
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
                      'UpdatePrb' : UpdatePrb,
                      'T_age' : None,
                      'T_cycle' : 1,
                      'cycles' : 0,
                      'T_sim' : periods_to_sim
                    }

# Define market parameters for the small open economy
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
init_DSGE_consumer['DiscFac'] = DiscFacMean
init_DSGE_consumer['aXtraMax'] = 80.0
init_DSGE_consumer['MgridBase'] = np.array([0.1,0.3,0.6,0.8,0.9,0.98,1.0,1.02,1.1,1.2,1.6,2.0,3.0])

# Define market parameters for the Cobb-Douglas economy
init_DSGE_market = copy(init_SOE_market)
init_DSGE_market.pop('Rfree')
init_DSGE_market.pop('wRte')
init_DSGE_market['CRRA'] = init_DSGE_consumer['CRRA']
init_DSGE_market['DiscFac'] = init_DSGE_consumer['DiscFac']
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
                      'T_sim' : periods_to_sim,
                      'tolerance' : 1e-12
                    }

###############################################################################

# Define parameters for the Markov representative agent model
init_RA_mrkv_consumer = copy(init_RA_consumer)
init_RA_mrkv_consumer['MrkvArray'] = PolyMrkvArray
init_RA_mrkv_consumer['MrkvNow'] = [StateCount/2]
init_RA_mrkv_consumer['PermGroFac'] = [PermGroFacSet]

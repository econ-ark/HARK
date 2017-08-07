'''
This module holds parameter dictionaries for the StickyE paper.
'''
import numpy as np
from copy import copy

periods_to_sim = 3500
ignore_periods = 1000
UpdatePrb = 1.0

# Choose parameters for the Markov models
StateCount = 21
PermGroFacMin = 0.9925
PermGroFacMax = 1.0075
Persistence = 0.9
RegimeChangePrb = 0.01

# Make the Markov array with chosen states and persistence
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

# Define the set of aggregate permanent growth factors that can occur
PermGroFacSet = np.linspace(PermGroFacMin,PermGroFacMax,num=StateCount)

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
                      'UpdatePrb' : UpdatePrb,
                      'T_age' : None,
                      'T_cycle' : 1,
                      'cycles' : 0,
                      'T_sim' : periods_to_sim
                    }

# Define parameters for the Cobb-Douglas DSGE version of the model
init_DSGE_consumer = copy(init_SOE_consumer)
init_DSGE_consumer['DiscFac'] = 1.0/1.0146501772118186
init_DSGE_consumer['aXtraMax'] = 80.0
init_DSGE_consumer['MgridBase'] = np.array([0.1,0.3,0.6,0.8,0.9,0.98,1.0,1.02,1.1,1.2,1.6,2.0,3.0])

# Define parameters for the small open Markov economy version of the model
init_SOE_mrkv_consumer = copy(init_SOE_consumer)
init_SOE_mrkv_consumer['MrkvArray'] = PolyMrkvArray

# Define parameters for the Cobb-Douglas Markov DSGE version of the model
init_DSGE_mrkv_consumer = copy(init_DSGE_consumer)
init_DSGE_mrkv_consumer['MrkvArray'] = PolyMrkvArray  

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

init_DSGE_market = copy(init_SOE_market)
init_DSGE_market.pop('Rfree')
init_DSGE_market.pop('wRte')
init_DSGE_market['CRRA'] = init_DSGE_consumer['CRRA']
init_DSGE_market['DiscFac'] = init_DSGE_consumer['DiscFac']
init_DSGE_market['intercept_prev'] = 0.0
init_DSGE_market['slope_prev'] = 1.0

init_SOE_mrkv_market = {  'PermShkAggCount': 3,
                          'TranShkAggCount': 3,
                          'PermShkAggStd': StateCount*[np.sqrt(0.00004)],
                          'TranShkAggStd': StateCount*[np.sqrt(0.00001)],
                          'MrkvArray' : PolyMrkvArray,
                          'PermGroFacAgg': PermGroFacSet,
                          'DeprFac': 1.0 - 0.94**(0.25),
                          'CapShare': 0.36,
                          'Rfree': 1.014189682528173,
                          'wRte': 2.5895209258224536,
                          'act_T': periods_to_sim,
                          'MrkvNow_init' : StateCount/2,
                          'loops_max' : 1
                          }

init_DSGE_mrkv_market = copy(init_SOE_mrkv_market)
init_DSGE_mrkv_market.pop('Rfree')
init_DSGE_mrkv_market.pop('wRte')
init_DSGE_mrkv_market['CRRA'] = init_DSGE_mrkv_consumer['CRRA']
init_DSGE_mrkv_market['DiscFac'] = init_DSGE_mrkv_consumer['DiscFac']
init_DSGE_mrkv_market['intercept_prev'] = StateCount*[0.0]
init_DSGE_mrkv_market['slope_prev'] = StateCount*[1.0]
init_DSGE_mrkv_market['loops_max'] = 10

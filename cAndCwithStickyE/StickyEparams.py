'''
This module holds parameter dictionaries for the StickyE paper.
'''
import numpy as np
from copy import copy

periods_to_sim = 3500
ignore_periods = 1000
UpdatePrb = 1.0

StateCount = 21
PermGroFacMin = 0.995
PermGroFacMax = 1.005
Persistence = 0.9

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

init_SOE_markov_consumer = copy(init_SOE_consumer)
init_SOE_markov_consumer['MrkvArray'] = np.array([[1.0]]) # dummy value, else breaks on init   

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


init_SOE_mrkv_market = {  'PermShkAggCount': 3,
                     'TranShkAggCount': 3,
                     'PermShkAggStd': [np.sqrt(0.00004)],
                     'TranShkAggStd': [np.sqrt(0.00001)],
                     'MrkvArray' : np.array([1.0]),
                     'PermGroFacAgg': 1.0,
                     'DeprFac': 1.0 - 0.94**(0.25),
                     'CapShare': 0.36,
                     'Rfree': 1.014189682528173,
                     'wRte': 2.5895209258224536,
                     'act_T': periods_to_sim,
                     'MrkvNow_init' : StateCount/2
                     }

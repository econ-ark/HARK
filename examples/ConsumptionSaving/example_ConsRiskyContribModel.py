# %%
'''
Example implementations of HARK.ConsumptionSaving.ConsPortfolioModel
'''
from HARK.ConsumptionSaving.ConsRiskyAssetModel import RiskyContribConsumerType, init_riskyContrib
from HARK.utilities import plotFuncs
from copy import copy
from time import time
import numpy as np
import matplotlib.pyplot as plt

# %%
# Make and solve an example of the risky pension contribution consumer type
init_sticky_share = init_riskyContrib.copy()
init_sticky_share['AdjustPrb'] = 0.15
init_sticky_share['DiscreteShareBool'] = True
init_sticky_share['vFuncBool'] = True


ContribAgent = RiskyContribConsumerType(**init_sticky_share)
# %%
# Make and solve a discrete portfolio choice consumer type
print('Now solving')
t0 = time()
ContribAgent.solve()
t1 = time()
print('Solving took ' + str(t1-t0) + ' seconds.')

# %% Plicy function inspection
ContribAgent.cFuncAdj = [ContribAgent.solution[t].cFuncAdj for t in range(ContribAgent.T_cycle)]
ContribAgent.cFuncFxd = [ContribAgent.solution[t].cFuncFxd for t in range(ContribAgent.T_cycle)]
ContribAgent.DFuncAdj = [ContribAgent.solution[t].cFuncFxd for t in range(ContribAgent.T_cycle)]
ContribAgent.ShareFuncAdj = [ContribAgent.solution[t].ShareFuncAdj for t in range(ContribAgent.T_cycle)]

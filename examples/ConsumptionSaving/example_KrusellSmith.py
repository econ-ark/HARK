from HARK.ConsumptionSaving.ConsAggShockModel import KrusellSmithType, KrusellSmithEconomy
from HARK.utilities import plotFuncs
from time import time
import numpy as np

init_KS_agents = {
    'T_cycle' : 1,
    'DiscFac' : 0.99,
    'CRRA' : 1.0,
    'LbrInd' : 1.,
    'aXtraMin' : 0.001,
    'aXtraMax' : 50.,
    'aXtraCount' : 48,
    'aXtraNestFac' : 2,
    'MgridBase' : np.array([0.1,0.3,0.6,0.8,0.9,0.95,0.98,1.0,1.02,1.05,1.1,1.2,1.6,2.0,3.0]),
    'AgentCount' : 10000
    }

init_KS_economy = {
    'verbose' : True,
    'act_T' : 1100,
    'T_discard' : 100,
    'DampingFac' : 0.5,
    'intercept_prev' : [0., 0.],
    'slope_prev' : [1., 1.],
    'DiscFac' : 0.99,
    'CRRA' : 1.0,
    'ProdB' : 0.99,
    'ProdG' : 1.01,
    'CapShare' : 0.36,
    'DeprFac' : 0.025,
    'DurMeanB' : 8.,
    'DurMeanG' : 8.,
    'SpellMeanB' : 2.5,
    'SpellMeanG' : 1.5,
    'UrateB' : 0.10,
    'UrateG' : 0.04,
    'RelProbBG' : 0.75,
    'RelProbGB' : 1.25,
    'MrkvNow_init' : 0,
    }

TestEconomy = KrusellSmithEconomy(**init_KS_economy)
TestType = KrusellSmithType(**init_KS_agents)
TestType.cycles = 0
TestType.getEconomyData(TestEconomy)
TestEconomy.agents = [TestType]
TestEconomy.makeMrkvHist()

t0 = time()
#TestType.solve()
TestEconomy.solve()
t1 = time()
print('Solving a KS type took ' + str(t1-t0) + ' seconds.')

for j in range(4):
    plotFuncs(TestType.solution[0].cFunc[j].xInterpolators, 0., 50.)

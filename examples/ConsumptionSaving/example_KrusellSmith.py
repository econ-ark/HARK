from HARK.ConsumptionSaving.ConsAggShockModel import KrusellSmithType, CobbDouglasMarkovEconomy
from HARK.utilities import plotFuncs
from time import time

agent_dict = {
    'DiscFac' : 0.99,
    'CRRA' : 2.0,
    'LbrInd' : 1.,
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
    'aXtraMin' : 0.001,
    'aXtraMax' : 20.,
    'aXtraCount' : 48,
    'aXtraNestFac' : 2
    }

TestType = KrusellSmithType(**agent_dict)
TestType.cycles = 0
t0 = time()
TestType.solve()
t1 = time()
print('Solving a KS type took ' + str(t1-t0) + ' seconds.')

for j in range(4):
    plotFuncs(TestType.solution[0].cFunc[j].xInterpolators, 0., 20.)

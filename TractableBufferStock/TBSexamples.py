'''
This module runs an example version of the Tractable Buffer Stock model.
'''
# Import the HARK library.  The assumption is that this code is in a folder
# contained in the HARK folder.  Also import the ConsumptionSavingModel
import sys 
sys.path.insert(0,'../')
sys.path.insert(0,'../ConsumptionSavingModel')

import numpy as np
import TractableBufferStock as Model
from HARKutilities import plotFunc, plotFuncs, plotFuncDer
from ConsumptionSavingModel import ConsumerType, solveConsumptionSavingMarkov
from time import clock

# Define the model primitives
base_primitives = {'UnempPrb' : .00625,
                   'DiscFac' : 0.975,
                   'Rfree' : 1.01,
                   'PermGroFac' : 1.0025,
                   'CRRA' : 1.0}
                   
# Make and solve a tractable consumer type
ExampleType = Model.TractableConsumerType(**base_primitives)
t_start = clock()
ExampleType.solve()
t_end = clock()
print('Solving a tractable consumption-savings model took ' + str(t_end-t_start) + ' seconds.')

# Plot the consumption function and whatnot
m_upper = 1.5*ExampleType.mTarg
conFunc_PF = lambda m: ExampleType.h*ExampleType.PFMPC + ExampleType.PFMPC*m
#plotFuncs([ExampleType.solution[0].cFunc,ExampleType.mSSfunc,ExampleType.cSSfunc],0,m_upper)
plotFuncs([ExampleType.solution[0].cFunc,ExampleType.solution[0].cFunc_U],0,m_upper)

# Now solve the same model using backward induction
init_consumer_objects = {"CRRA":base_primitives['CRRA'],
                        "Rfree":np.array(2*[base_primitives['Rfree']]),
                        "PermGroFac":[np.array(2*[base_primitives['PermGroFac']/(1.0-base_primitives['UnempPrb'])])],
                        "BoroCnstArt":None,
                        "PermShkStd":[0.0],
                        "PermShkCount":1,
                        "TranShkStd":[0.0],
                        "TranShkCount":1,
                        "T_total":1,
                        "UnempPrb":0.0,
                        "UnempPrbRet":0.0,
                        "T_retire":0,
                        "IncUnemp":0.0,
                        "IncUnempRet":0.0,
                        "aXtraMin":0.001,
                        "aXtraMax":ExampleType.mUpperBnd,
                        "aXtraCount":48,
                        "aXtraExtra":[None],
                        "exp_nest":3,
                        "LivPrb":[1.0],
                        "DiscFac":[base_primitives['DiscFac']],
                        'Nagents':1,
                        'psi_seed':0,
                        'xi_seed':0,
                        'unemp_seed':0,
                        'tax_rate':0.0,
                        'vFuncBool':False,
                        'CubicBool':True
                        }
MarkovType = ConsumerType(**init_consumer_objects)
MrkvArray = np.array([[1.0-base_primitives['UnempPrb'],base_primitives['UnempPrb']],[0.0,1.0]])
employed_income_dist = [np.ones(1),np.ones(1),np.ones(1)]
unemployed_income_dist = [np.ones(1),np.ones(1),np.zeros(1)]
MarkovType.solution_terminal.cFunc = 2*[MarkovType.solution_terminal.cFunc]
MarkovType.solution_terminal.vFunc = 2*[MarkovType.solution_terminal.vFunc]
MarkovType.solution_terminal.vPfunc = 2*[MarkovType.solution_terminal.vPfunc]
MarkovType.solution_terminal.vPPfunc = 2*[MarkovType.solution_terminal.vPPfunc]
MarkovType.solution_terminal.mNrmMin = 2*[MarkovType.solution_terminal.mNrmMin]
MarkovType.solution_terminal.MPCmax = np.array(2*[MarkovType.solution_terminal.MPCmax])
MarkovType.IncomeDstn = [[employed_income_dist,unemployed_income_dist]]
MarkovType.MrkvArray = MrkvArray
MarkovType.time_inv.append('MrkvArray')
MarkovType.solveOnePeriod = solveConsumptionSavingMarkov
MarkovType.cycles = 0

t_start = clock()
MarkovType.solve()
t_end = clock()
MarkovType.unpack_cFunc()

print('Solving the same model "the long way" took ' + str(t_end-t_start) + ' seconds.')
#plotFuncs([ExampleType.solution[0].cFunc,ExampleType.solution[0].cFunc_U],0,m_upper)
plotFuncs(MarkovType.cFunc[0],0,m_upper)
diffFunc = lambda m : ExampleType.solution[0].cFunc(m) - MarkovType.cFunc[0][0](m)
print('Difference between the (employed) consumption functions:')
plotFunc(diffFunc,0,m_upper)

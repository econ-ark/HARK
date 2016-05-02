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
from HARKutilities import plotFunc, plotFuncs
from ConsumptionSavingModel import ConsumerType, consumptionSavingSolverMarkov
from time import clock

# Define the model primitives
base_primitives = {'mho' : .00625,
                   'beta' : 0.975,
                   'R' : 1.01,
                   'G' : 1.0025,
                   'rho' : 1.0}
                   
# Make and solve a tractable consumer type
ExampleType = Model.TractableConsumerType(**base_primitives)
t_start = clock()
ExampleType.solve()
t_end = clock()
print('Solving a tractable consumption-savings model took ' + str(t_end-t_start) + ' seconds.')

# Plot the consumption function and whatnot
m_upper = 1.5*ExampleType.m_targ
conFunc_PF = lambda m: ExampleType.h*ExampleType.kappa_PF + ExampleType.kappa_PF*m
#plotFuncs([ExampleType.solution[0].cFunc,ExampleType.mSSfunc,ExampleType.cSSfunc],0,m_upper)
plotFuncs([ExampleType.solution[0].cFunc,ExampleType.solution[0].cFunc_U],0,m_upper)

# Now solve the same model using backward induction
init_consumer_objects = {"CRRA":base_primitives['rho'],
                        "Rfree":base_primitives['R'],
                        "PermGroFac":[base_primitives['G']/(1.0-base_primitives['mho'])],
                        "BoroCnst":None,
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
                        "aDispMin":0.001,
                        "aDispMax":ExampleType.m_max,
                        "aDispCount":16,
                        "aDispExtra":[None],
                        "exp_nest":3,
                        "LivFac":[1.0],
                        "DiscFac":[base_primitives['beta']],
                        'Nagents':1,
                        'psi_seed':0,
                        'xi_seed':0,
                        'unemp_seed':0,
                        'tax_rate':0.0,
                        'vFuncBool':False,
                        'CubicBool':True
                        }
MarkovType = ConsumerType(**init_consumer_objects)
transition_array = np.array([[1.0-base_primitives['mho'],base_primitives['mho']],[0.0,1.0]])
employed_income_dist = [np.ones(1),np.ones(1),np.ones(1)]
unemployed_income_dist = [np.ones(1),np.ones(1),np.zeros(1)]
p_zero_income = [np.array([0.0,1.0])]
MarkovType.solution_terminal.cFunc = 2*[MarkovType.solution_terminal.cFunc]
MarkovType.solution_terminal.vFunc = 2*[MarkovType.solution_terminal.vFunc]
MarkovType.solution_terminal.vPfunc = 2*[MarkovType.solution_terminal.vPfunc]
MarkovType.solution_terminal.vPPfunc = 2*[MarkovType.solution_terminal.vPPfunc]
MarkovType.solution_terminal.mRtoMin = 2*[MarkovType.solution_terminal.mRtoMin]
MarkovType.IncomeDist = [[employed_income_dist,unemployed_income_dist]]
MarkovType.p_zero_income = p_zero_income
MarkovType.transition_array = transition_array
MarkovType.time_inv.append('transition_array')
MarkovType.time_vary.append('p_zero_income')
MarkovType.solveOnePeriod = consumptionSavingSolverMarkov
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

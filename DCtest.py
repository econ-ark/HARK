'''
This is a temporary script for testing the discrete choice module.
'''

from HARKutilities import plotFunc, plotFuncs
import DiscreteChoice as Model
import numpy as np

x_grid = np.linspace(0,10,200)
sigma = 0.2

vFuncA = lambda x : x**0.5
vFuncB = lambda x : (2.0*(x-1.0))**0.5
vFuncC = lambda x : (3.0*(x-2.0))**0.5
vPfuncA = lambda x : 0.5*x**(-0.5)
vPfuncB = lambda x : (2.0*(x-1.0))**(-0.5)
vPfuncC = lambda x : 1.5*(3.0*(x-2.0))**(-0.5)

vFunc = [vFuncA, vFuncB, vFuncC]
vPfunc = [vPfuncA, vPfuncB, vPfuncC]

transFunc = lambda Z : Z
transFuncP = lambda Z : np.ones(Z.shape)
#transFunc = lambda x : x**2
#transFuncP = lambda x : 2*x

solution_tp1 = Model.DiscreteChoiceSolution(vFunc,vPfunc)
solution_tp1.bonus = 'break on through'

solution_t = Model.discreteChoiceContinuousStateSolver(solution_tp1,x_grid,sigma,transFunc,transFuncP)
plotFuncs([solution_t.vFunc,vFuncA,vFuncB,vFuncC],0,10)
plotFuncs([solution_t.vPfunc,vPfuncA,vPfuncB,vPfuncC],0,10)

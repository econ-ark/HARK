'''
This module demonstrates some examples of the discrete choice solver in a
continuous 1D space.
'''

from ConsumptionSavingModel import consumptionSavingSolverENDG, ConsumerType
from DiscreteChoice import discreteChoiceContinuousStateSolver, DiscreteChoiceSolution, makeCRRAtransformations
from HARKutilities import setupGridsExpMult, plotFunc, plotFuncs
import SetupConsumerParameters as Params
import numpy as np
from time import clock
from copy import deepcopy

# Define the solver for the "occupational choice" problem
def occupationalChoiceSolver(solution_tp1,income_distrib,p_zero_income,survival_prob,beta,rho,R,Gamma,constrained,a_grid,calc_vFunc,cubic_splines):
    '''
    Need to write a real description here.
    '''
    # Initialize lists to hold the solution for each occupation
    vFunc = []
    vPfunc = []
    cFunc = []
    m_underbar = []
    gothic_h = []
    kappa_min = []
    kappa_max = []
    v_check = []
    
    # Solve the consumption-saving problem for each possible occupation
    job_count = len(Gamma)
    large_m = 100.0*np.max(a_grid)
    for j in range(job_count):
        solution_temp = consumptionSavingSolverENDG(solution_tp1,income_distrib[j],p_zero_income[j],survival_prob,beta,rho,R,Gamma[j],constrained,a_grid,calc_vFunc,cubic_splines)
        vFunc.append(deepcopy(solution_temp.vFunc))
        vPfunc.append(deepcopy(solution_temp.vPfunc))
        cFunc.append(deepcopy(solution_temp.cFunc))
        m_underbar.append(solution_temp.m_underbar)
        gothic_h.append(solution_temp.gothic_h)
        kappa_min.append(solution_temp.kappa_min)
        kappa_max.append(solution_temp.kappa_max)
        v_check.append(solution_temp.vFunc(large_m))
    
    # Find the best job at "infinity" resources
    v_check = np.asarray(v_check)
    v_best = np.argmax(v_check)
    
    # Construct the solution for this phase
    solution_t = DiscreteChoiceSolution(vFunc,vPfunc)
    solution_t.cFunc = cFunc
    solution_t.m_underbar = np.max(m_underbar)
    solution_t.kappa_min = kappa_min[v_best]
    solution_t.kappa_max = np.min(kappa_max)
    solution_t.gothic_h = gothic_h[v_best]
    solution_t.v_lim_slope = solution_t.kappa_min**(-rho/(1.0-rho))
    solution_t.v_lim_intercept = solution_t.gothic_h*solution_t.v_lim_slope
    
    return solution_t


# Make an infinite horizon agent that has the option of buying out of income shocks
Params.init_consumer_objects['xi_sigma'] = [0.1,0.1,0.1]
Params.init_consumer_objects['psi_sigma'] = [0.1,0.15,0.2]
Params.init_consumer_objects['T_retire'] = 0
Params.init_consumer_objects['T_total'] = 3
Gamma_set = [1.01,1.02,1.03]
TestType = ConsumerType(**Params.init_consumer_objects)
TestType.timeFwd()
TestType.assignParameters(survival_prob = [None,0.98],
                          beta = [None,0.96],
                          Gamma = [None,Gamma_set],
                          rho = 3.0,
                          sigma_epsilon = [0.0001,None],
                          state_grid = setupGridsExpMult(0.001, 50, 64, 3),
                          a_max = 20,
                          a_size = 48,
                          calc_vFunc = True,
                          cubic_splines = False,
                          constrained = True,
                          income_distrib = [None,TestType.income_distrib],
                          p_zero_income = [None,TestType.p_zero_income],
                          solveAPeriod = [discreteChoiceContinuousStateSolver, occupationalChoiceSolver],
                          tolerance = 0.0001,
                          cycles = 0)
TestType.updateAssetsGrid()
TestType.updateSolutionTerminal()
transformations = makeCRRAtransformations(TestType.rho,do_Q=True,do_T=True,do_Z=True)
TestType(transformations = transformations)
#TestType(transformations = None)
TestType.time_inv += ['transformations','state_grid']
TestType.time_vary += ['sigma_epsilon','solveAPeriod']

# Solve the income insurance problem
t_start = clock()
TestType.solve()
t_end = clock()
print('Took ' + str(t_end-t_start) + ' seconds to solve occupational choice problem.')

plotFuncs(TestType.solution[1].vFunc,0.1,3)
Q = transformations.Qfunc
f = lambda x : Q(TestType.solution[1].vFunc[2](x)) - Q(TestType.solution[1].vFunc[0](x))
g = lambda x : Q(TestType.solution[1].vFunc[1](x)) - Q(TestType.solution[1].vFunc[0](x))
plotFuncs([f, g],0.1,50)


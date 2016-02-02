'''
This module demonstrates some examples of the discrete choice solver in a
continuous 1D space.
'''

# Import the HARK library.  The assumption is that this code is in a folder
# contained in the HARK folder. 
import sys 
sys.path.insert(0,'../')

from ConsumptionSavingModel import consumptionSavingSolverENDG, ConsumerType
from DiscreteChoice import discreteChoiceContinuousStateSolver, DiscreteChoiceSolution
from HARKutilities import calculateMeanOneLognormalDiscreteApprox, addDiscreteOutcomeConstantMean, setupGridsExpMult, plotFunc, plotFuncs, CRRAutility_inv, CRRAutility_invP
import SetupConsumerParameters as Params
import numpy as np
from time import clock

premium = 0.01
identityFunc = lambda Z : Z
onesFunc = lambda Z : np.ones(Z.shape)
insured_income_dist = [np.array(1.0),np.array(1.0),np.array(1.0)]

# Define the solver for the "income insurance" problem
def incomeInsuranceSolver(solution_tp1,income_distrib,p_zero_income,survival_prob,beta,rho,R,Gamma,constrained,a_grid,calc_vFunc,cubic_splines):
    '''
    Need to write a real description here.
    '''
    # Solve for optimal consumption in an ordinary period and when income shock insurance is purchased
    solution_regular = consumptionSavingSolverENDG(solution_tp1,income_distrib,p_zero_income,survival_prob,beta,rho,R,Gamma,constrained,a_grid,calc_vFunc,cubic_splines)
    solution_insured = consumptionSavingSolverENDG(solution_tp1,insured_income_dist,0.0,survival_prob,beta,rho,R,Gamma-premium,constrained,a_grid,calc_vFunc,cubic_splines)
    
    # Construct the solution for this phase
    vFunc = [solution_regular.vFunc, solution_insured.vFunc]
    vPfunc = [solution_regular.vPfunc, solution_insured.vPfunc]
    cFunc = [solution_regular.cFunc, solution_insured.cFunc]
    solution_t = DiscreteChoiceSolution(vFunc,vPfunc)
    solution_t.cFunc = cFunc
    
    # Add the "pass through" attributes to the solution and report it
    other_attributes = [key for key in solution_regular.__dict__]
    other_attributes.remove('vFunc')
    other_attributes.remove('vPfunc')
    other_attributes.remove('cFunc')
    for name in other_attributes:
        do_string = 'solution_t.' + name + ' = solution_regular.' + name
        exec(do_string)
    return solution_t

# Make an infinite horizon agent that has the option of buying out of income shocks 
TestType = ConsumerType(**Params.init_consumer_objects)
TestType.timeFwd()
TestType.assignParameters(survival_prob = [None,0.98],
                          beta = [None,0.96],
                          Gamma = [None,1.01],
                          sigma_epsilon = [0.2,None],
                          state_grid = setupGridsExpMult(0.00001, 50, 64, 3),
                          a_size = 48,
                          calc_vFunc = True,
                          cubic_splines = False,
                          constrained = True,
                          #transFunc = lambda x : CRRAutility_inv(x,gam=TestType.rho),
                          #transFuncP = lambda x : CRRAutility_invP(x,gam=TestType.rho),
                          transFunc = identityFunc,
                          transFuncP = onesFunc,
                          income_distrib = [None,TestType.income_distrib[0]],
                          p_zero_income = [None,TestType.p_zero_income[0]],
                          solveAPeriod = [discreteChoiceContinuousStateSolver, incomeInsuranceSolver],
                          cycles = 100)
TestType.updateAssetsGrid()
TestType.time_inv += ['transFunc','transFuncP','state_grid']
TestType.time_vary += ['sigma_epsilon','solveAPeriod']

# Solve the income insurance problem
t_start = clock()
TestType.solve()
t_end = clock()
print('Took ' + str(t_end-t_start) + ' seconds to solve income insurance problem.')


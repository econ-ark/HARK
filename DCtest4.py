'''
This module solves a simple retirement choice model.
'''

from ConsumptionSavingModel import consumptionSavingSolverENDG, ConsumerType
from DiscreteChoice import discreteChoiceContinuousStateSolver, DiscreteChoiceSolution, makeCRRAtransformations
from HARKutilities import setupGridsExpMult, plotFunc, plotFuncs, addDiscreteOutcomeConstantMean, calculateMeanOneLognormalDiscreteApprox, createFlatStateSpaceFromIndepDiscreteProbs, CRRAutility, CRRAutilityP
import SetupConsumerParameters as Params
from HARKcore import AgentType
import numpy as np
from time import time
from copy import deepcopy

# Define the solver for the retirement choice problem
def retirementChoiceSolver(solution_tp1,income_distrib,p_zero_income,survival_prob,beta,rho,R,Gamma,constrained,a_grid,calc_vFunc,cubic_splines,solution_retired):
    '''
    Solves one period of an infinite horizon retirement choice problem.
    '''
    # Solve the period when we're working
    solution_working = consumptionSavingSolverENDG(solution_tp1,income_distrib,p_zero_income,survival_prob,beta,rho,R,Gamma,constrained,a_grid,calc_vFunc,cubic_splines)
    
    # Determine which choice is best at "infinity"
    large_m = 1000.0*np.max(a_grid)
    v_check = np.asarray([solution_working.vFunc(large_m),solution_retired.vFunc(large_m)])
    v_best = np.argmax(v_check)
    
    # Construct and return the solution for this period    
    vFunc = [solution_working.vFunc,solution_retired.vFunc]
    vPfunc = [solution_working.vPfunc,solution_retired.vPfunc]
    solution_t = DiscreteChoiceSolution(vFunc,vPfunc)
    solution_t.cFunc = [solution_working.cFunc,solution_retired.cFunc]
    solution_t.m_underbar = 0
    solution_t.kappa_max = np.min(solution_working.kappa_max,solution_retired.kappa_max)
    if v_best < -1:
        solution_t.kappa_min = solution_working.kappa_min
        solution_t.gothic_h = solution_working.gothic_h
    else:
        solution_t.kappa_min = solution_retired.kappa_min
        solution_t.gothic_h = solution_retired.gothic_h
    solution_t.v_lim_slope = solution_t.kappa_min**(-rho/(1.0-rho))
    solution_t.v_lim_intercept = solution_t.gothic_h*solution_t.v_lim_slope
    return solution_t
    
    
# Make a class to represent the retiring consumer
class RetiringConsumerType(ConsumerType):
    
    def __init__(self,cycles=0,time_flow=True,**kwds):
        '''
        Instantiate a new RetiringConsumerType with given data.
        '''       
        # Initialize a basic AgentType
        AgentType.__init__(self,solution_terminal=deepcopy(ConsumerType.solution_terminal_),cycles=cycles,time_flow=time_flow,pseudo_terminal=True,**kwds)

        # Add consumer-type specific objects, copying to create independent versions
        self.time_vary = deepcopy(ConsumerType.time_vary_)
        self.time_inv = deepcopy(ConsumerType.time_inv_)
        self.time_vary.remove('beta')
        self.time_vary.remove('survival_prob')
        self.time_vary.remove('Gamma')
        self.time_vary += ['sigma_epsilon','solveAPeriod']
        self.time_inv += ['solution_retired','transformations','state_grid','beta','survival_prob','Gamma']
        
        self.solveAPeriod = [discreteChoiceContinuousStateSolver, retirementChoiceSolver]
        self.update()
        
        
    def updateIncomeProcess(self):
        '''
        Updates this agent's income process based on his own attributes.  The
        function that generates the discrete income process can be swapped out
        for a different process.
        '''
        original_time = self.time_flow
        self.timeFwd()
        perm_shocks = calculateMeanOneLognormalDiscreteApprox(self.psi_N, self.psi_sigma)
        temp_shocks = addDiscreteOutcomeConstantMean(calculateMeanOneLognormalDiscreteApprox(self.psi_N, self.xi_sigma),self.income_unemploy,self.p_unemploy)
        self.income_distrib = createFlatStateSpaceFromIndepDiscreteProbs(perm_shocks,temp_shocks)
        self.p_zero_income = np.sum(temp_shocks[0][temp_shocks[1] == 0])
        if not 'income_distrib' in self.time_vary:
            self.time_vary.append('income_distrib')
        if not 'p_zero_income' in self.time_vary:
            self.time_vary.append('p_zero_income')
        if not original_time:
            self.timeRev()
            
    def preSolve(self):
        '''
        Before using time iteration, the retired solution can be found in closed
        form.  It is stored here in scaled form, then unscaled in postSolve.
        '''
        self.value_scale = (1.0-self.labor_supply)**(self.alpha*(1.0-self.rho))
        
        temp_beta = self.survival_prob*self.beta
        kappa_PF = 1.0 - (self.R*temp_beta)**(1.0/self.rho)/self.R
        cFunc_retired = lambda m : kappa_PF*m
        vPfunc_retired = lambda m : CRRAutilityP(m,gam=self.rho)/self.value_scale
        vFunc_retired = lambda m: CRRAutility(kappa_PF**(-self.rho/(1.0-self.rho))*m,gam=self.rho)/self.value_scale
        
        solution_retired = DiscreteChoiceSolution(vFunc_retired,vPfunc_retired)
        solution_retired.cFunc = cFunc_retired
        solution_retired.kappa_min = kappa_PF
        solution_retired.kappa_max = kappa_PF
        solution_retired.gothic_h = 0.0
        
        self.solution_retired = solution_retired
        self.updateSolutionTerminal()
        
    def postSolve(self):
        '''
        Simply un-scales the (marginal) value functions to finish the problem.
        '''
#        time_orig = self.time_flow
#        self.timeFwd()
#        self.solution[0].vFunc = lambda x : self.value_scale*self.solution[0].vFunc(x)
#        self.solution[0].vPfunc = lambda x : self.value_scale*self.solution[0].vPfunc(x)
#        self.solution[1].vFunc[0] = lambda x : self.value_scale*self.solution[1].vFunc[0](x)
#        self.solution[1].vPfunc[0] = lambda x : self.value_scale*self.solution[1].vPfunc[0](x)
#        self.solution[1].vFunc[1] = lambda x : self.value_scale*self.solution[1].vFunc[1](x)
#        self.solution[1].vPfunc[1] = lambda x : self.value_scale*self.solution[1].vPfunc[1](x)
#        if not time_orig:
#            self.timeRev()
            


# Make an example type
Params.init_consumer_objects['xi_sigma'] = 0.1
Params.init_consumer_objects['psi_sigma'] = 0.1
Params.init_consumer_objects['T_retire'] = 0
Params.init_consumer_objects['T_total'] = 1
Params.init_consumer_objects['beta'] = 0.96
TestType = RetiringConsumerType(**Params.init_consumer_objects)
TestType(survival_prob = 0.9,
          beta = 0.96,
          Gamma = 1.00,
          rho = 2.0,
          sigma_epsilon = [0.0001,None],
          state_grid = setupGridsExpMult(0.001, 50, 64, 3),
          a_max = 20,
          a_size = 48,
          calc_vFunc = True,
          cubic_splines = False,
          constrained = True,
          income_distrib = [None,TestType.income_distrib],
          p_zero_income = [None,TestType.p_zero_income],
          labor_supply = 0.5,
          alpha = 1.0,
          tolerance = 0.0001,
          cycles=0)
TestType.updateAssetsGrid()
transformations = makeCRRAtransformations(TestType.rho,do_Q=True,do_T=True,do_Z=True)
TestType(transformations = transformations)

t_start = time()
TestType.solve()
t_end = time()
print('Took ' + str(t_end-t_start) + ' seconds to solve retirement choice problem.')

plotFuncs(TestType.solution[1].vFunc,20,100)
plotFunc(TestType.solution[0].vFunc,20,100)

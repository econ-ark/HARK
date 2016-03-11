'''
This is an extended version of ConsumptionSavingModel that includes preference
shocks to marginal utility as well as an interest rate on borrowing that differs
from the interest rate on saving.  It should be identical to DCL's "DAP model"
and will be used as the "long run model" in the deferred interest project.
'''

import sys 
sys.path.insert(0,'../')

import numpy as np
from HARKcore import AgentType, NullFunc
from HARKutilities import warnings  # Because of "patch" to warnings modules
from HARKutilities import calculateMeanOneLognormalDiscreteApprox, addDiscreteOutcomeConstantMean, createFlatStateSpaceFromIndepDiscreteProbs, setupGridsExpMult, CRRAutility, CRRAutilityP, CRRAutilityPP, CRRAutilityP_inv, CRRAutility_invP, CRRAutility_inv
from HARKinterpolation import ConstrainedComposite, LinearInterp, LinearInterpOnInterp1D
from HARKsimulation import generateMeanOneLognormalDraws, generateBernoulliDraws
from ConsumptionSavingModel import constructAssetsGrid
from copy import deepcopy, copy

utility = CRRAutility
utilityP = CRRAutilityP
utilityPP = CRRAutilityPP
utilityP_inv = CRRAutilityP_inv
utility_invP = CRRAutility_invP
utility_inv = CRRAutility_inv



class ConsumerSolution():
    '''
    A class representing the solution of a single period of a consumption-saving
    problem.  The solution must include a consumption function, but may also include
    the minimum allowable money resources m_underbar, expected human wealth gothic_h,
    and the lower and upper bounds on the MPC kappa_min and kappa_max.  A value
    function can also be included, as well as marginal value and marg marg value.
    '''

    def __init__(self, cFunc=NullFunc, vFunc=NullFunc, vPfunc=NullFunc, vPPfunc=NullFunc, m_underbar=None, gothic_h=None, kappa_min=None, kappa_max=None):
        '''
        The constructor for a new ConsumerSolution object.
        '''
        self.cFunc = cFunc
        self.vFunc = vFunc
        self.vPfunc = vPfunc
        self.vPPfunc = vPPfunc
        self.m_underbar = m_underbar
        self.gothic_h = gothic_h
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max

    def distance(self,solution_other):
        '''
        Returns the distance between single period solutions as the distance
        between their consumption functions.
        '''
        if type(self.cFunc) is list:
            dist_vec = np.zeros(len(self.cFunc)) + np.nan
            for i in range(len(self.cFunc)):
                dist_vec[i] = self.cFunc[i].distance(solution_other.cFunc[i])
            return np.max(dist_vec)
        else:
            return self.cFunc.distance(solution_other.cFunc)
            

class ValueFunc():
    '''
    A class for representing a value function.  The underlying interpolation is
    in the space of (m,u_inv(v)); this class "re-curves" to the value function.
    '''
    def __init__(self,vFuncDecurved,rho):
        self.func = deepcopy(vFuncDecurved)
        self.rho = rho
        
    def __call__(self,m):
        return utility(self.func(m),gam=self.rho)

     
class MargValueFunc():
    '''
    A class for representing a marginal value function in models where the
    standard envelope condition of v'(m) = u'(c(m)) holds (with CRRA utility).
    '''
    def __init__(self,cFunc,rho):
        self.cFunc = deepcopy(cFunc)
        self.rho = rho
        
    def __call__(self,m):
        return utilityP(self.cFunc(m),gam=self.rho)
        
        
class MargMargValueFunc():
    '''
    A class for representing a marginal marginal value function in models where
    the standard envelope condition of v'(m) = u'(c(m)) holds (with CRRA utility).
    '''
    def __init__(self,cFunc,rho):
        self.cFunc = deepcopy(cFunc)
        self.rho = rho
        
    def __call__(self,m):
        c, kappa = self.cFunc.eval_with_derivative(m)
        return kappa*utilityPP(c,gam=self.rho)
        
        
def consPrefShockSolver(solution_tp1,income_distrib,p_zero_income,pref_shock_grid,pref_cdf_grid,survival_prob,beta,rho,R_save,R_borrow,Gamma,constraint,a_grid,calc_vFunc,cubic_splines):
    '''
    Solves a single period of a consumption-saving model with preference shocks
    to marginal utility.  Problem is solved using the method of endogenous gridpoints.

    Parameters:
    -----------
    solution_tp1: ConsumerSolution
        The solution to the following period.
    income_dist: [np.array]
        A list containing three arrays of floats, representing a discrete approx-
        imation to the income process between the period being solved and the one
        immediately following (in solution_tp1).  Order: probs, psi, xi
    p_zero_income: float
        The probability of receiving zero income in the succeeding period.
    pref_shock_grid: np.array
        Array with values of the multiplicative utility shifter.
    pref_cdf_grid: np.array
        Array with CDF values of the values in pref_shock_grid.    
    survival_prob: float
        Probability of surviving to succeeding period.
    beta: float
        Discount factor between this period and the succeeding period.
    rho: float
        The coefficient of relative risk aversion
    R_borrow: float
        Interest factor on assets between this period and the succeeding period
        when assets are negative.
    R_save: float
        Interest factor on assets between this period and the succeeding period
        when assets are positive.
    Gamma: float
        Expected growth factor for permanent income between this period and the
        succeeding period.
    constraint: float
        Borrowing constraint for the minimum allowable assets to end the period
        with.  If it is less than the natural borrowing constraint, then it is
        irrelevant; constraint=None indicates no artificial borrowing constraint.
    a_grid: [float]
        A list of end-of-period asset values (post-decision states) at which to
        solve for optimal consumption.

    Returns:
    -----------
    solution_t: ConsumerSolution
        The solution to this period's problem, obtained using the method of endogenous gridpoints.
    '''

    # Define utility and value functions
    uP = lambda c : utilityP(c,gam=rho)
    uPinv = lambda u : utilityP_inv(u,gam=rho)

    # Set and update values for this period
    effective_beta = beta*survival_prob
    psi_tp1 = income_distrib[1]
    xi_tp1 = income_distrib[2]
    prob_tp1 = income_distrib[0]
    vPfunc_tp1 = solution_tp1.vPfunc
    psi_underbar_tp1 = np.min(psi_tp1)    
    xi_underbar_tp1 = np.min(xi_tp1)
    
    # Calculate the minimum allowable value of money resources in this period
    m_underbar_t = max((solution_tp1.m_underbar - xi_underbar_tp1)*(Gamma*psi_underbar_tp1)/R_borrow, constraint)

    # Define the borrowing constraint (limiting consumption function)
    constraint_t = lambda m: m - m_underbar_t

    # Find data for the unconstrained consumption function in this period
    a = np.sort(np.hstack((a_grid + m_underbar_t,np.array([0.0,0.0]))))
    a_N = a.size
    R_vec = R_save*np.ones(a_N)
    R_vec[0:(np.sum(a<=0)-1)] = R_borrow
    shock_N = xi_tp1.size
    a_temp = np.tile(a,(shock_N,1))
    R_temp = np.tile(R_vec,(shock_N,1))
    psi_temp = (np.tile(psi_tp1,(a_N,1))).transpose()
    xi_temp = (np.tile(xi_tp1,(a_N,1))).transpose()
    prob_temp = (np.tile(prob_tp1,(a_N,1))).transpose()
    m_tp1 = R_temp/(Gamma*psi_temp)*a_temp + xi_temp
    gothicvP = effective_beta*R_vec*Gamma**(-rho)*np.sum(psi_temp**(-rho)*vPfunc_tp1(m_tp1)*prob_temp,axis=0)
    c_base = uPinv(gothicvP)
    pref_N = pref_shock_grid.size
    pref_shock_temp = np.tile(np.reshape(pref_shock_grid**(1.0/rho),(pref_N,1)),(1,a_N))
    c = np.tile(c_base,(pref_N,1))*pref_shock_temp
    m = c + np.tile(a,(pref_N,1))

    # Make the preference-shock specific consumption functions
    cFunc_list = []
    for j in range(pref_N):
        m_temp = m[j,:]
        c_temp = c[j,:]
        cFunc_this_shock = ConstrainedComposite(LinearInterp(m_temp,c_temp),constraint_t)
        cFunc_list.append(cFunc_this_shock)
        
    # Combine the list of consumption functions into a single interpolation
    cFunc_t = LinearInterpOnInterp1D(cFunc_list,pref_shock_grid)
        
    # Make the ex ante marginal value function (before the preference shock)
    m_grid = a_grid + m_underbar_t
    vP_vec = np.zeros_like(m_grid)
    c_bot = cFunc_list[0](m_grid)
    v_bot = (c_bot)**(1.0-rho)
    F_bot = pref_cdf_grid[0]
    for j in range(1,pref_N): # numeric integration over the preference shock
        c_top = cFunc_list[j](m_grid)
        v_top = (c_top)**(1.0-rho)
        F_top = pref_cdf_grid[j]
        scale = (c_top-c_bot)/(F_top-F_bot)
        vP_vec = vP_vec + (v_top - v_bot)/scale
    vP_vec = vP_vec/(1.0-rho)
    c_pseudo = uPinv(vP_vec)
    vPfunc_t = MargValueFunc(LinearInterp(m_grid,c_pseudo),rho)

    # Store the results in a solution object and return it
    solution_t = ConsumerSolution(cFunc=cFunc_t, vPfunc=vPfunc_t, m_underbar=m_underbar_t)
    return solution_t
    
    
    
class PrefShockConsumer(AgentType):
    '''
    An agent in the consumption-saving model with preference shocks.
    '''        
    # Define some universal values for all consumer types
    cFunc_terminal_ = LinearInterp([0.0, 1.0],[0.0,1.0])
    constraint_terminal_ = lambda x: x
    solution_terminal_ = ConsumerSolution(cFunc=ConstrainedComposite(cFunc_terminal_,constraint_terminal_), vFunc = vFunc_terminal_, m_underbar=0.0, gothic_h=0.0, kappa_min=1.0, kappa_max=1.0)
    time_vary_ = ['survival_prob','Gamma']
    time_inv_ = ['beta','rho','R_save','R_borrow','a_grid','constraint']
    
    def __init__(self,cycles=1,time_flow=True,**kwds):
        '''
        Instantiate a new PrefShockConsumer with given data.
        '''       
        # Initialize a basic AgentType
        AgentType.__init__(self,solution_terminal=deepcopy(PrefShockConsumer.solution_terminal_),cycles=cycles,time_flow=time_flow,pseudo_terminal=False,**kwds)

        # Add consumer-type specific objects, copying to create independent versions
        self.time_vary = deepcopy(PrefShockConsumer.time_vary_)
        self.time_inv = deepcopy(PrefShockConsumer.time_inv_)
        self.solveAPeriod = consPrefShockSolver
        self.update()
        
                
    def updateIncomeProcess(self):
        '''
        Updates this agent's income process based on his own attributes.  The
        function that generates the discrete income process can be swapped out
        for a different process.
        '''
        original_time = self.time_flow
        self.timeFwd()
        income_distrib, p_zero_income = constructLognormalIncomeProcessUnemployment(self)
        self.income_distrib = income_distrib
        self.p_zero_income = p_zero_income
        if not 'income_distrib' in self.time_vary:
            self.time_vary.append('income_distrib')
        if not 'p_zero_income' in self.time_vary:
            self.time_vary.append('p_zero_income')
        if not original_time:
            self.timeRev()
            
    def updateAssetsGrid(self):
        '''
       Updates this agent's end-of-period assets grid.
        '''
        a_grid = constructAssetsGrid(self)
        self.a_grid = a_grid
        
    def updateSolutionTerminal(self):
        '''
        Update the terminal period solution
        '''
        self.solution_terminal.vFunc = ValueFunc(self.solution_terminal.cFunc,self.rho)
        self.solution_terminal.vPfunc = MargValueFunc(self.solution_terminal.cFunc,self.rho)
        self.solution_terminal.vPPfunc = MargMargValueFunc(self.solution_terminal.cFunc,self.rho)
        
        
        
        
def constructLognormalIncomeProcessUnemployment(parameters):
    """
    Generates a list of discrete approximations to the income process for each
    life period, from end of life to beginning of life.  Permanent shocks are mean
    one lognormally distributed with standard deviation psi_sigma[t] during the
    working life, and degenerate at 1 in the retirement period.  Transitory shocks
    are mean one lognormally distributed with a point mass at income_unemploy with
    probability p_unemploy while working; they are mean one with a point mass at
    income_unemploy_retire with probability p_unemploy_retire.  Retirement occurs
    after t=final_work_index periods of retirement.

    Note 1: All time in this function runs forward, from t=0 to t=T
    
    Note 2: All parameters are passed as attributes of the input parameters.

    Parameters:
    -----------
    psi_sigma:    [float]
        Array of standard deviations in _permanent_ income uncertainty during
        the agent's life.
    psi_N:      int
        The number of approximation points to be used in the equiprobable
        discrete approximation to the permanent income shock distribution.
    xi_sigma      [float]
        Array of standard deviations in _temporary_ income uncertainty during
        the agent's life.
    xi_N:       int
        The number of approximation points to be used in the equiprobable
        discrete approximation to the permanent income shock distribution.
    p_unemploy:             float
        The probability of becoming unemployed
    p_unemploy_retire:      float
        The probability of not receiving typical retirement income in any retired period
    T_retire:       int
        The index value i equal to the final working period in the agent's life.
        If T_retire <= 0 then there is no retirement.
    income_unemploy:         float
        Income received when unemployed. Often zero.
    income_unemploy_retire:  float
        Income received while "unemployed" when retired. Often zero.
    T_total:       int
        Total number of non-terminal periods in this consumer's life.

    Returns
    =======
    income_distrib:  [income distribution]
        Each element contains the joint distribution of permanent and transitory
        income shocks, as a set of vectors: psi_shock, xi_shock, and pmf. The
        first two are the points in the joint state space, and final vector is
        the joint pmf over those points. For example,
               psi_shock[20], xi_shock[20], and pmf[20]
        refers to the (psi, xi) point indexed by 20, with probability p = pmf[20].
    p_zero_income: [float]
        A list of probabilities of receiving exactly zero income in each period.

    """
    # Unpack the parameters from the input
    psi_sigma = parameters.psi_sigma
    psi_N = parameters.psi_N
    xi_sigma = parameters.xi_sigma
    xi_N = parameters.xi_N
    T_total = parameters.T_total
    T_retire = parameters.T_retire
    p_unemploy = parameters.p_unemploy
    income_unemploy = parameters.income_unemploy
    p_unemploy_retire = parameters.p_unemploy_retire        
    income_unemploy_retire = parameters.income_unemploy_retire
    
    income_distrib = [] # Discrete approximation to income process
    p_zero_income = [] # Probability of zero income in each period of life

    # Fill out a simple discrete RV for retirement, with value 1.0 (mean of shocks)
    # in normal times; value 0.0 in "unemployment" times with small prob.
    if T_retire > 0:
        if p_unemploy_retire > 0:
            retire_perm_income_values = np.array([1.0, 1.0])    # Permanent income is deterministic in retirement (2 states for temp income shocks)
            retire_income_values = np.array([income_unemploy_retire, (1.0-p_unemploy_retire*income_unemploy_retire)/(1.0-p_unemploy_retire)])
            retire_income_probs = np.array([p_unemploy_retire, 1.0-p_unemploy_retire])
        else:
            retire_perm_income_values = np.array([1.0])
            retire_income_values = np.array([1.0])
            retire_income_probs = np.array([1.0])
        income_dist_retire = [retire_income_probs,retire_perm_income_values,retire_income_values]

    # Loop to fill in the list of income_distrib random variables.
    for t in range(T_total): # Iterate over all periods, counting forward

        if T_retire > 0 and t >= T_retire:
            # Then we are in the "retirement period" and add a retirement income object.
            income_distrib.append(deepcopy(income_dist_retire))
            if income_unemploy_retire == 0:
                p_zero_income.append(p_unemploy_retire)
            else:
                p_zero_income.append(0)
        else:
            # We are in the "working life" periods.
            temp_xi_dist = calculateMeanOneLognormalDiscreteApprox(N=xi_N, sigma=xi_sigma[t])
            if p_unemploy > 0:
                temp_xi_dist = addDiscreteOutcomeConstantMean(temp_xi_dist, p=p_unemploy, x=income_unemploy)
            temp_psi_dist = calculateMeanOneLognormalDiscreteApprox(N=psi_N, sigma=psi_sigma[t])
            income_distrib.append(createFlatStateSpaceFromIndepDiscreteProbs(temp_psi_dist, temp_xi_dist))
            if income_unemploy == 0:
                p_zero_income.append(p_unemploy)
            else:
                p_zero_income.append(0)

    return income_distrib, p_zero_income        

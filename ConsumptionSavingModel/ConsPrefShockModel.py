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
from HARKutilities import approxLognormal, CRRAutility, CRRAutilityP, CRRAutilityPP, CRRAutilityP_inv, CRRAutility_invP, CRRAutility_inv
from HARKinterpolation import LowerEnvelope, LinearInterp, LinearInterpOnInterp1D
from HARKsimulation import drawMeanOneLognormal, drawBernoulli
from ConsumptionSavingModel import constructAssetsGrid, constructLognormalIncomeProcessUnemployment
from copy import deepcopy

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
        dist = self.cFunc.distance(solution_other.cFunc)
        #print(dist)
        return dist
        
            

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
        
        
def consPrefShockSolver(solution_next,income_distrib,pref_shock_dist,survival_prob,beta,rho,R_save,R_borrow,Gamma,constraint,a_grid):
    '''
    Solves a single period of a consumption-saving model with preference shocks
    to marginal utility.  Problem is solved using the method of endogenous gridpoints.

    Parameters:
    -----------
    solution_next: ConsumerSolution
        The solution to the following period.
    income_dist: [np.array]
        A list containing three arrays of floats, representing a discrete approx-
        imation to the income process between the period being solved and the one
        immediately following (in solution_next).  Order: probs, psi, xi
    pref_shock_dist: [np.array]
        Discrete distribution of the multiplicative utility shifter.
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
    vPfunc_tp1 = solution_next.vPfunc
    psi_underbar_tp1 = np.min(psi_tp1)    
    xi_underbar_tp1 = np.min(xi_tp1)
    pref_shock_vals = pref_shock_dist[1]
    pref_shock_prob = pref_shock_dist[0]
    
    # Calculate the minimum allowable value of money resources in this period
    m_underbar_t = max((solution_next.m_underbar - xi_underbar_tp1)*(Gamma*psi_underbar_tp1)/R_borrow, constraint)

    # Define the borrowing constraint (limiting consumption function)
    constraint_t = lambda m: m - m_underbar_t

    # Find data for the unconstrained consumption function in this period
    if m_underbar_t < 0.0 and R_save < R_borrow:
        a = np.sort(np.hstack((a_grid + m_underbar_t,np.array([0.0,0.0]))))
    else: # Don't add kink points at zero unless borrowing is possible
        a = deepcopy(a_grid + m_underbar_t)
    a_N = a.size
    R_vec = R_save*np.ones(a_N)
    borrow_count = (np.sum(a<=0)-1)
    if borrow_count > 0:
        R_vec[0:borrow_count] = R_borrow
    shock_N = xi_tp1.size
    a_temp = np.tile(a,(shock_N,1))
    R_temp = np.tile(R_vec,(shock_N,1))
    psi_temp = (np.tile(psi_tp1,(a_N,1))).transpose()
    xi_temp = (np.tile(xi_tp1,(a_N,1))).transpose()
    prob_temp = (np.tile(prob_tp1,(a_N,1))).transpose()
    m_tp1 = R_temp/(Gamma*psi_temp)*a_temp + xi_temp
    gothicvP = effective_beta*R_vec*Gamma**(-rho)*np.sum(psi_temp**(-rho)*vPfunc_tp1(m_tp1)*prob_temp,axis=0)
    c_base = uPinv(gothicvP)
    pref_N = pref_shock_vals.size
    pref_shock_temp = np.tile(np.reshape(pref_shock_vals**(1.0/rho),(pref_N,1)),(1,a_N))
    c = np.tile(c_base,(pref_N,1))*pref_shock_temp
    m = c + np.tile(a,(pref_N,1))

    # Make the preference-shock specific consumption functions
    cFunc_list = []
    for j in range(pref_N):
        m_temp = np.concatenate((np.array([m_underbar_t]),m[j,:]))
        c_temp = np.concatenate((np.array([0.0]),c[j,:]))
        cFunc_this_shock = LowerEnvelope(LinearInterp(m_temp,c_temp),constraint_t)
        cFunc_list.append(cFunc_this_shock)
        
    # Combine the list of consumption functions into a single interpolation
    cFunc_t = LinearInterpOnInterp1D(cFunc_list,pref_shock_vals)
        
    # Make the ex ante marginal value function (before the preference shock)
    m_grid = a_grid + m_underbar_t
    vP_vec = np.zeros_like(m_grid)
    for j in range(1,pref_N): # numeric integration over the preference shock
        vP_vec += uP(cFunc_list[j](m_grid))*pref_shock_prob[j]*pref_shock_vals[j]
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
    solution_terminal_ = ConsumerSolution(cFunc=LowerEnvelope(cFunc_terminal_,constraint_terminal_), vFunc=None, m_underbar=0.0, gothic_h=0.0, kappa_min=1.0, kappa_max=1.0)
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
        self.solveOnePeriod = consPrefShockSolver
        self.update()
        
                
    def updateIncomeProcess(self):
        '''
        Updates this agent's income process based on his own attributes.  The
        function that generates the discrete income process can be swapped out
        for a different process.
        '''
        original_time = self.time_flow
        self.timeFwd()
        income_distrib = constructLognormalIncomeProcessUnemployment(self)
        self.income_distrib = income_distrib
        if not 'income_distrib' in self.time_vary:
            self.time_vary.append('income_distrib')
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
     
    def updatePrefDist(self):
        '''
        Updates this agent's preference shock distribution.
        '''
        pref_shock_dist = approxLognormal(self.pref_shock_N,0.0,self.pref_shock_sigma,tail_N=self.pref_shock_tail_N)
        self.pref_shock_dist = pref_shock_dist
        if not 'pref_shock_dist' in self.time_inv:
            self.time_inv.append('pref_shock_dist')
            
    def update(self):
        '''
        Update income process, assets and preference shock grid, and terminal solution.
        '''
        self.updateIncomeProcess()
        self.updateAssetsGrid()
        self.updateSolutionTerminal()
        self.updatePrefDist()
        
    def addShockPaths(self, perm_shocks,temp_shocks, pref_shocks):
        '''
        Adds paths of simulated shocks to the agent as attributes.
        '''
        original_time = self.time_flow
        self.timeFwd()
        self.perm_shocks = perm_shocks
        self.temp_shocks = temp_shocks
        self.pref_shocks = pref_shocks
        if not 'perm_shocks' in self.time_vary:
            self.time_vary.append('perm_shocks')
        if not 'temp_shocks' in self.time_vary:
            self.time_vary.append('temp_shocks')
        if not 'pref_shocks' in self.time_vary:
            self.time_vary.append('pref_shocks')
        if not original_time:
            self.timeRev()
            
    def simulate(self,w_init,t_first,t_last,which=['w']):
        '''
        Simulate the model forward from initial conditions w_init, beginning in
        t_first and ending in t_last.
        '''
        original_time = self.time_flow
        self.timeFwd()
        if self.cycles > 0:
            cFuncs = self.cFunc[t_first:t_last]
        else:
            cFuncs = t_last*self.cFunc # This needs to be fixed for IH models
        simulated_history = simulateConsumerHistory(cFuncs, w_init, self.perm_shocks[t_first:t_last], self.temp_shocks[t_first:t_last], self.pref_shocks[t_first:t_last],self.R_borrow,self.R_save,which)
        if not original_time:
            self.timeRev()
        return simulated_history
        
    def unpack_cFunc(self):
        '''
        "Unpacks" the consumption functions into their own field for easier access.
        After the model has been solved, the consumption functions reside in the
        attribute cFunc of each element of ConsumerType.solution.  This method
        creates a (time varying) attribute cFunc that contains a list of consumption
        functions.
        '''
        self.cFunc = []
        for solution_t in self.solution:
            self.cFunc.append(solution_t.cFunc)
        if not ('cFunc' in self.time_vary):
            self.time_vary.append('cFunc')
        




def simulateConsumerHistory(cFunc,w0,psi_adj,theta,eta,R_borrow,R_save,which):
    """
    Generates simulated consumer histories.  Agents begin with W/Y ratio of of
    w0 and follow the consumption rules in cFunc each period. Permanent and trans-
    itory shocks are provided in psi_adj and theta.  Note that psi_adj represents
    "adjusted permanent income shock": psi_adj = psi*Gamma.
    
    The histories returned by the simulator are determined by which, a list of
    strings that can include 'w', 'm', 'c', 'a', and 'kappa'.  Other strings will
    cause an error on return.  Outputs are returned in the order listed by the user.
    """
    # Determine the size of potential simulated histories
    periods_to_simulate = len(theta)
    N_agents = len(theta[0])
    
    # Initialize arrays to hold simulated histories as requested
    if 'w' in which:
        w = np.zeros([periods_to_simulate+1,N_agents]) + np.nan
        do_w = True
    else:
        do_w = False
    if 'm' in which:
        m = np.zeros([periods_to_simulate,N_agents]) + np.nan
        do_m = True
    else:
        do_m = False
    if 'c' in which:
        c = np.zeros([periods_to_simulate,N_agents]) + np.nan
        do_c = True
    else:
        do_c = False
    if 'a' in which:
        a = np.zeros([periods_to_simulate,N_agents]) + np.nan
        do_a = True
    else:
        do_a = False
    if 'kappa' in which:
        kappa = np.zeros([periods_to_simulate,N_agents]) + np.nan
        do_k = True
    else:
        do_k = False

    # Initialize the simulation
    w_t = w0
    if do_w:
        w[0,] = w_t
    
    # Run the simulation for all agents:
    for t in range(periods_to_simulate):
        m_t = w_t + theta[t]
        if do_k:
            kappa_t = cFunc[t].derivativeX(m_t,eta[t])        
        c_t = cFunc[t](m_t,eta[t])
        a_t = m_t - c_t
        R_t = R_save*np.ones_like(a_t)
        R_t[a_t < 0.0] = R_borrow
        w_t = R_t*a_t/psi_adj[t]
        
        # Store the requested variables in the history arrays
        if do_w:
            w[t+1,] = w_t
        if do_m:
            m[t,] = m_t
        if do_c:
            c[t,] = c_t
        if do_a:
            a[t,] = a_t
        if do_k:
            kappa[t,] = kappa_t
            
    # Return the simulated histories as requested
    return_list = ''
    for var in which:
        return_list = return_list + var + ', '
    x = len(return_list)
    return_list = return_list[0:(x-2)]
    return eval(return_list)



def generateShockHistoryInfiniteSimple(parameters):
    '''
    Creates arrays of permanent and transitory income shocks for sim_N simulated
    consumers for sim_T identical infinite horizon periods.
    
    Arguments: (as attributes of parameters)
    ----------
    psi_sigma : [float]
        Permanent income standard deviation for the consumer.
    xi_sigma : [float]
        Transitory income standard deviation for the consumer.
    pref_shock_sigma : float
        Preference shock standard deviation for the consumer.
    Gamma : [float]
        Permanent income growth rate for the consumer.
    p_unemploy : float
        The probability of becoming unemployed
    income_unemploy : float
        Income received when unemployed. Often zero.
    sim_N : int
        The number of consumers to generate shocks for.
    RNG : np.random.rand
        This type's random number generator.
    sim_T : int
        Number of periods of shocks to generate.
    
    Returns:
    ----------
    perm_shock_history : [np.array]
        A sim_T-length list of sim_N-length arrays of permanent income shocks.
        The shocks are adjusted to include permanent income growth Gamma.
    trans_shock_history : [np.array]
        A sim_T-length list of sim_N-length arrays of transitory income shocks.
    pref_shock_history : [np.array]
        A sim_T-length list of sim_N-length arrays of preference shocks.
    '''
    # Unpack the parameters
    psi_sigma = parameters.psi_sigma
    xi_sigma = parameters.xi_sigma
    pref_shock_sigma = parameters.pref_shock_sigma
    Gamma = parameters.Gamma
    p_unemploy = parameters.p_unemploy
    income_unemploy = parameters.income_unemploy
    sim_N = parameters.sim_N
    psi_seed = parameters.RNG.randint(2**31-1)
    xi_seed = parameters.RNG.randint(2**31-1)
    unemp_seed = parameters.RNG.randint(2**31-1)
    pref_shock_seed = parameters.RNG.randint(2**31-1)
    sim_T = parameters.sim_T
    
    trans_shock_history = drawMeanOneLognormal(sim_T*xi_sigma, sim_N, xi_seed)
    unemployment_history = drawBernoulli(sim_T*[p_unemploy],sim_N,unemp_seed)
    perm_shock_history = drawMeanOneLognormal(sim_T*psi_sigma, sim_N, psi_seed)
    pref_shock_history = drawMeanOneLognormal(sim_T*[pref_shock_sigma], sim_N, pref_shock_seed)
    for t in range(sim_T):
        perm_shock_history[t] = (perm_shock_history[t]*Gamma[0])
        trans_shock_history[t] = trans_shock_history[t]*(1-p_unemploy*income_unemploy)/(1-p_unemploy)
        trans_shock_history[t][unemployment_history[t]] = income_unemploy
        
    return perm_shock_history, trans_shock_history, pref_shock_history
        
       
if __name__ == '__main__':
    import SetupPrefShockConsParameters as Params
    from HARKutilities import plotFunc, plotFuncDer, plotFuncs
    from time import time
    import matplotlib.pyplot as plt
    
    # Make an example agent type
    ExampleType = PrefShockConsumer(**Params.init_consumer_objects)
    ExampleType(cycles = 0)
#    perm_shocks, temp_shocks, pref_shocks = generateShockHistoryInfiniteSimple(ExampleType)
#    ExampleType.addShockPaths(perm_shocks, temp_shocks, pref_shocks)
#    ExampleType(w_init = np.zeros(ExampleType.sim_N))
    
    # Solve the agent's problem and plot the consumption functions
    t_start = time()
    ExampleType.solve()
    t_end = time()
    print('Solving a preference shock consumer took ' + str(t_end-t_start) + ' seconds.')
    
    m = np.linspace(ExampleType.solution[0].m_underbar,2,200)
    #m = np.linspace(0.0,1.0,200)
    for j in range(ExampleType.pref_shock_dist[1].size):
        pref_shock = ExampleType.pref_shock_dist[1][j]
        c = ExampleType.solution[0].cFunc(m,pref_shock*np.ones_like(m))
        plt.plot(m,c)
    plt.show()
    c = ExampleType.solution[0].cFunc(m,np.ones_like(m))
    k = ExampleType.solution[0].cFunc.derivativeX(m,np.ones_like(m))
    plt.plot(m,c)
    plt.plot(m,k)
    plt.show()
    
    # Simulate some wealth history
#    ExampleType.unpack_cFunc()
#    history = ExampleType.simulate(ExampleType.w_init,0,ExampleType.sim_T,which=['m'])
#    plt.plot(np.mean(history,axis=1))
#    plt.show()
    

'''
This is an extended version of ConsumptionSavingModel that includes preference
shocks to marginal utility as well as an interest rate on borrowing that differs
from the interest rate on saving.  It should be identical to DCL's "DAP model"
and will be used as the "long run model" in the deferred interest project.
'''

import sys 
sys.path.insert(0,'../')

import numpy as np
from HARKcore import AgentType, NullFunc, Solution
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


class ConsumerSolution(Solution):
    '''
    A class representing the solution of a single period of a consumption-saving
    problem.  The solution must include a consumption function, but may also include
    the minimum allowable money resources mNrmMin, expected human wealth hRto,
    and the lower and upper bounds on the MPC MPCmin and MPCmax.  A value
    function can also be included, as well as marginal value and marg marg value.
    '''

    def __init__(self, cFunc=NullFunc, vFunc=NullFunc, vPfunc=NullFunc, vPPfunc=NullFunc, mNrmMin=None, hRto=None, MPCmin=None, MPCmax=None):
        '''
        The constructor for a new ConsumerSolution object.
        '''
        self.cFunc = cFunc
        self.vFunc = vFunc
        self.vPfunc = vPfunc
        self.vPPfunc = vPPfunc
        self.mNrmMin = mNrmMin
        self.hRto = hRto
        self.MPCmin = MPCmin
        self.MPCmax = MPCmax
        self.convergence_criteria = ['cFunc']
       

class ValueFunc():
    '''
    A class for representing a value function.  The underlying interpolation is
    in the space of (m,u_inv(v)); this class "re-curves" to the value function.
    '''
    def __init__(self,vFuncDecurved,CRRA):
        self.func = deepcopy(vFuncDecurved)
        self.CRRA = CRRA
        
    def __call__(self,m):
        return utility(self.func(m),gam=self.CRRA)

     
class MargValueFunc():
    '''
    A class for representing a marginal value function in models where the
    standard envelope condition of v'(m) = u'(c(m)) holds (with CRRA utility).
    '''
    def __init__(self,cFunc,CRRA):
        self.cFunc = deepcopy(cFunc)
        self.CRRA = CRRA
        
    def __call__(self,m):
        return utilityP(self.cFunc(m),gam=self.CRRA)
        
        
class MargMargValueFunc():
    '''
    A class for representing a marginal marginal value function in models where
    the standard envelope condition of v'(m) = u'(c(m)) holds (with CRRA utility).
    '''
    def __init__(self,cFunc,CRRA):
        self.cFunc = deepcopy(cFunc)
        self.CRRA = CRRA
        
    def __call__(self,m):
        c, MPC = self.cFunc.eval_with_derivative(m)
        return MPC*utilityPP(c,gam=self.CRRA)
        
        
def consPrefShockSolver(solution_next,IncomeDstn,PrefShkDstn,LivPrb,DiscFac,CRRA,Rsave,Rboro,PermGroFac,BoroCnstArt,aXtraGrid):
    '''
    Solves a single period of a consumption-saving model with preference shocks
    to marginal utility.  Problem is solved using the method of endogenous gridpoints.

    Parameters:
    -----------
    solution_next: ConsumerSolution
        The solution to the following period.
    IncomeDstn: [np.array]
        A list containing three arrays of floats, representing a discrete approx-
        imation to the income process between the period being solved and the one
        immediately following (in solution_next).  Order: ShkPrbs, PermShkVals, TranShkVals
    PrefShkDstn: [np.array]
        Discrete distribution of the multiplicative utility shifter.
    LivPrb: float
        Probability of surviving to succeeding period.
    DiscFac: float
        Discount factor between this period and the succeeding period.
    CRRA: float
        The coefficient of relative risk aversion
    Rboro: float
        Interest factor on assets between this period and the succeeding period
        when assets are negative.
    Rsave: float
        Interest factor on assets between this period and the succeeding period
        when assets are positive.
    PermGroFac: float
        Expected growth factor for permanent income between this period and the
        succeeding period.
    BoroCnstArt: float
        Borrowing constraint for the minimum allowable assets to end the period
        with.  If it is less than the natural borrowing constraint, then it is
        irrelevant; BoroCnstArt=None indicates no artificial borrowing constraint.
    aXtraGrid: [float]
        A list of end-of-period asset values (post-decision states) at which to
        solve for optimal consumption.

    Returns:
    -----------
    solution_now: ConsumerSolution
        The solution to this period's problem, obtained using the method of endogenous gridpoints.
    '''

    # Define utility and value functions
    uP = lambda c : utilityP(c,gam=CRRA)
    uPinv = lambda u : utilityP_inv(u,gam=CRRA)

    # Set and update values for this period
    DiscFacEff = DiscFac*LivPrb
    PermShkValsNext = IncomeDstn[1]
    TranShkValsNext = IncomeDstn[2]
    IncShkPrbsNext = IncomeDstn[0]
    vPfuncNext = solution_next.vPfunc
    PermShkMinNext = np.min(PermShkValsNext)    
    TranShkMinNext = np.min(TranShkValsNext)
    PrefShkVals = PrefShkDstn[1]
    PrefShkPrbs = PrefShkDstn[0]
    
    # Calculate the minimum allowable value of money resources in this period
    mNrmMinNow = max((solution_next.mNrmMin - TranShkMinNext)*(PermGroFac*PermShkMinNext)/Rboro, BoroCnstArt)

    # Define the borrowing constraint (limiting consumption function)
    cFuncNowCnst = LinearInterp([mNrmMinNow,mNrmMinNow+1.0],[0.0,1.0])

    # Find data for the unconstrained consumption function in this period
    if mNrmMinNow < 0.0 and Rsave < Rboro:
        aGrid = np.sort(np.hstack((aXtraGrid + mNrmMinNow,np.array([0.0,0.0]))))
    else: # Don't add kink points at zero unless borrowing is possible
        aGrid = deepcopy(aXtraGrid + mNrmMinNow)
    a_N = aGrid.size
    R_vec = Rsave*np.ones(a_N)
    borrow_count = (np.sum(aGrid<=0)-1)
    if borrow_count > 0:
        R_vec[0:borrow_count] = Rboro
    shock_N = TranShkValsNext.size
    aGrid_temp = np.tile(aGrid,(shock_N,1))
    R_temp = np.tile(R_vec,(shock_N,1))
    PermShkVals_temp = (np.tile(PermShkValsNext,(a_N,1))).transpose()
    TranShkVals_temp = (np.tile(TranShkValsNext,(a_N,1))).transpose()
    IncShkPrbs_temp = (np.tile(IncShkPrbsNext,(a_N,1))).transpose()
    mNext = R_temp/(PermGroFac*PermShkVals_temp)*aGrid_temp + TranShkVals_temp
    EndOfPrdvP = DiscFacEff*R_vec*PermGroFac**(-CRRA)*np.sum(PermShkVals_temp**(-CRRA)*vPfuncNext(mNext)*IncShkPrbs_temp,axis=0)
    c_base = uPinv(EndOfPrdvP)
    pref_N = PrefShkVals.size
    PrefShk_temp = np.tile(np.reshape(PrefShkVals**(1.0/CRRA),(pref_N,1)),(1,a_N))
    c = np.tile(c_base,(pref_N,1))*PrefShk_temp
    m = c + np.tile(aGrid,(pref_N,1))

    # Make the preference-shock specific consumption functions
    cFunc_list = []
    for j in range(pref_N):
        m_temp = np.concatenate((np.array([mNrmMinNow]),m[j,:]))
        c_temp = np.concatenate((np.array([0.0]),c[j,:]))
        cFunc_this_shock = LowerEnvelope(LinearInterp(m_temp,c_temp),cFuncNowCnst)
        cFunc_list.append(cFunc_this_shock)
        
    # Combine the list of consumption functions into a single interpolation
    cFuncNow = LinearInterpOnInterp1D(cFunc_list,PrefShkVals)
        
    # Make the ex ante marginal value function (before the preference shock)
    m_grid = aXtraGrid + mNrmMinNow
    vP_vec = np.zeros_like(m_grid)
    for j in range(pref_N): # numeric integration over the preference shock
        vP_vec += uP(cFunc_list[j](m_grid))*PrefShkPrbs[j]*PrefShkVals[j]
    c_pseudo = uPinv(vP_vec)
    vPfuncNow = MargValueFunc(LinearInterp(m_grid,c_pseudo),CRRA)

    # Store the results in a solution object and return it
    solution_now = ConsumerSolution(cFunc=cFuncNow, vPfunc=vPfuncNow, mNrmMin=mNrmMinNow)
    return solution_now
    
    
    
class PrefShockConsumer(AgentType):
    '''
    An agent in the consumption-saving model with preference shocks.
    '''        
    # Define some universal values for all consumer types
    cFunc_terminal_ = LinearInterp([0.0, 1.0],[0.0,1.0])
    cFuncCnst_terminal_ = LinearInterp([0.0, 1.0],[0.0,1.0])
    solution_terminal_ = ConsumerSolution(cFunc=LowerEnvelope(cFunc_terminal_,cFuncCnst_terminal_), vFunc=None, mNrmMin=0.0, hRto=0.0, MPCmin=1.0, MPCmax=1.0)
    time_vary_ = ['LivPrb','PermGroFac']
    time_inv_ = ['DiscFac','CRRA','Rsave','Rboro','aXtraGrid','BoroCnstArt']
    
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
        IncomeDstn = constructLognormalIncomeProcessUnemployment(self)
        self.IncomeDstn = IncomeDstn
        if not 'IncomeDstn' in self.time_vary:
            self.time_vary.append('IncomeDstn')
        if not original_time:
            self.timeRev()
            
    def updateAssetsGrid(self):
        '''
       Updates this agent's end-of-period assets grid.
        '''
        aXtraGrid = constructAssetsGrid(self)
        self.aXtraGrid = aXtraGrid
        
    def updateSolutionTerminal(self):
        '''
        Update the terminal period solution
        '''
        self.solution_terminal.vFunc = ValueFunc(self.solution_terminal.cFunc,self.CRRA)
        self.solution_terminal.vPfunc = MargValueFunc(self.solution_terminal.cFunc,self.CRRA)
        self.solution_terminal.vPPfunc = MargMargValueFunc(self.solution_terminal.cFunc,self.CRRA)
     
    def updatePrefDist(self):
        '''
        Updates this agent's preference shock distribution.
        '''
        PrefShkDstn = approxLognormal(self.PrefShk_N,0.0,self.PrefShkStd,tail_N=self.PrefShk_tail_N)
        self.PrefShkDstn = PrefShkDstn
        if not 'PrefShkDstn' in self.time_inv:
            self.time_inv.append('PrefShkDstn')
            
    def update(self):
        '''
        Update income process, assets and preference shock grid, and terminal solution.
        '''
        self.updateIncomeProcess()
        self.updateAssetsGrid()
        self.updateSolutionTerminal()
        self.updatePrefDist()
        
    def addShockPaths(self, perm_shocks,temp_shocks, PrefShks):
        '''
        Adds paths of simulated shocks to the agent as attributes.
        '''
        original_time = self.time_flow
        self.timeFwd()
        self.perm_shocks = perm_shocks
        self.temp_shocks = temp_shocks
        self.PrefShks = PrefShks
        if not 'perm_shocks' in self.time_vary:
            self.time_vary.append('perm_shocks')
        if not 'temp_shocks' in self.time_vary:
            self.time_vary.append('temp_shocks')
        if not 'PrefShks' in self.time_vary:
            self.time_vary.append('PrefShks')
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
        simulated_history = simulateConsumerHistory(cFuncs, w_init, self.perm_shocks[t_first:t_last], self.temp_shocks[t_first:t_last], self.PrefShks[t_first:t_last],self.Rboro,self.Rsave,which)
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
        


def simulateConsumerHistory(cFunc,w0,psi_adj,theta,eta,Rboro,Rsave,which):
    """
    Generates simulated consumer histories.  Agents begin with W/Y ratio of of
    w0 and follow the consumption rules in cFunc each period. Permanent and trans-
    itory shocks are provided in psi_adj and theta.  Note that psi_adj represents
    "adjusted permanent income shock": psi_adj = psi*PermGroFac.
    
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
        R_t = Rsave*np.ones_like(a_t)
        R_t[a_t < 0.0] = Rboro
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
    PermShkStd : [float]
        Permanent income standard deviation for the consumer.
    TranShkStd : [float]
        Transitory income standard deviation for the consumer.
    PrefShkStd : float
        Preference shock standard deviation for the consumer.
    PermGroFac : [float]
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
        The shocks are adjusted to include permanent income growth PermGroFac.
    trans_shock_history : [np.array]
        A sim_T-length list of sim_N-length arrays of transitory income shocks.
    PrefShk_history : [np.array]
        A sim_T-length list of sim_N-length arrays of preference shocks.
    '''
    # Unpack the parameters
    PermShkStd = parameters.PermShkStd
    TranShkStd = parameters.TranShkStd
    PrefShkStd = parameters.PrefShkStd
    PermGroFac = parameters.PermGroFac
    p_unemploy = parameters.p_unemploy
    income_unemploy = parameters.income_unemploy
    sim_N = parameters.sim_N
    psi_seed = parameters.RNG.randint(2**31-1)
    xi_seed = parameters.RNG.randint(2**31-1)
    unemp_seed = parameters.RNG.randint(2**31-1)
    pref_shock_seed = parameters.RNG.randint(2**31-1)
    sim_T = parameters.sim_T
    
    trans_shock_history = drawMeanOneLognormal(sim_T*TranShkStd, sim_N, xi_seed)
    unemployment_history = drawBernoulli(sim_T*[p_unemploy],sim_N,unemp_seed)
    perm_shock_history = drawMeanOneLognormal(sim_T*PermShkStd, sim_N, psi_seed)
    PrefShk_history = drawMeanOneLognormal(sim_T*[PrefShkStd], sim_N, pref_shock_seed)
    for t in range(sim_T):
        perm_shock_history[t] = (perm_shock_history[t]*PermGroFac[0])
        trans_shock_history[t] = trans_shock_history[t]*(1-p_unemploy*income_unemploy)/(1-p_unemploy)
        trans_shock_history[t][unemployment_history[t]] = income_unemploy
        
    return perm_shock_history, trans_shock_history, PrefShk_history
        
       
if __name__ == '__main__':
    import SetupPrefShockConsParameters as Params
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
    
    m = np.linspace(ExampleType.solution[0].mNrmMin,5,200)
    for j in range(ExampleType.PrefShkDstn[1].size):
        PrefShk = ExampleType.PrefShkDstn[1][j]
        c = ExampleType.solution[0].cFunc(m,PrefShk*np.ones_like(m))
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
    

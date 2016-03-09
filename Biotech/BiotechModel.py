'''
This is a first draft of the BIOTECH model solution method.
'''
# Import the HARK library.  The assumption is that this code is in a folder
# contained in the HARK folder. 
import sys 
sys.path.insert(0,'../')

import numpy as np
from scipy import stats
from HARKcore import AgentType
from copy import deepcopy
from HARKcore import solveACycle
from HARKinterpolation import QuadlinearInterp
from HARKutilities import makeMarkovApproxToNormal, calculateLognormalDiscreteApprox, calculateBetaDiscreteApprox, createFlatStateSpaceFromIndepDiscreteProbs, setupGridsExpMult
from HARKestimation import bootstrapSampleFromData
#from time import time

'''
A class representing a biotechnology startup firm.
'''
class BiotechType(AgentType):
    param_list = ['money_min','money_max','money_N','mu_min','mu_max','mu_N','tau_min','tau_max','tau_N','macro_min','macro_max','macro_N','sharesold_N','W_N',
                  'money_init_mean','mu_init_mean','tau_init_mean','money_init_var','mu_init_var','tau_init_var','money_mu_covar','money_tau_covar','mu_tau_covar','value_outside','sigma_found',
                  'low_intensity','high_intensity','low_cost','high_cost','sigma_research',
                  'theta_0','theta_mu','theta_tau','theta_mu_tau','theta_z','lambda_0','lambda_mu','lambda_tau','lambda_mu_tau','lambda_z','V_b','sigma_terminal',
                  'gamma_0','gamma_mu','gamma_tau','gamma_mu_tau','gamma_z','sigma_W','sharesold_a','sharesold_b','sigma_VC','VC_cost',
                  'rho','discount','Qshock_big','Qshock_small','Qprob_big','Qprob_small']                  
    integer_list = [2,5,8,11,12,13]
    param_count = len(param_list)
    input_list = ['mu_weight_matrix_low','mu_weight_matrix_high','m_idx_low_top','m_idx_low_bot','m_alpha_low_top','m_alpha_low_bot','m_idx_high_top','m_idx_high_bot','m_alpha_high_top','m_alpha_high_bot','tau_idx_low_top','tau_idx_low_bot','tau_alpha_low_top','tau_alpha_low_bot','tau_idx_high_top','tau_idx_high_bot','tau_alpha_high_top','tau_alpha_high_bot','mu_idx_P1','z_idx_P1','unaffordable_low','unaffordable_high','sigma_research',
                  'value_private','value_IPO','value_bankrupt','sigma_terminal',
                  'm_idx_bot','m_idx_top','mu_idx_P3','tau_idx_P3','z_idx_P3','m_alpha_bot','m_alpha_top','sharekept','sigma_VC','VC_cost',
                  'discount','mu_weight_matrix_P4','macro_weight_matrix_P4','post_process']
    
    param_list_alt = ['money_min','money_max','money_N','mu_min','mu_max','mu_N','tau_min','tau_max','tau_N','macro_min','macro_max','macro_N','W_N',
                  'money_init_mean','mu_init_mean','tau_init_mean','money_init_var','mu_init_var','tau_init_var','money_mu_covar','money_tau_covar','mu_tau_covar','value_outside','sigma_found',
                  'low_intensity','high_intensity','low_cost','high_cost','sigma_research',
                  'theta_0','theta_mu','theta_tau','theta_mu_tau','theta_z','lambda_0','lambda_mu','lambda_tau','lambda_mu_tau','lambda_z','V_b','sigma_terminal',
                  'gamma_0','gamma_mu','gamma_tau','gamma_mu_tau','gamma_z','sigma_W','sharesold_fixed','pi_0','pi_m','pi_mu','pi_tau','pi_mu_tau','pi_z',
                  'rho','discount','Qshock_big','Qshock_small','Qprob_big','Qprob_small']
    integer_list_alt = [2,5,8,11,12]
    param_count_alt = len(param_list_alt)
    input_list_alt = ['mu_weight_matrix_low','mu_weight_matrix_high','m_idx_low_top','m_idx_low_bot','m_alpha_low_top','m_alpha_low_bot','m_idx_high_top','m_idx_high_bot','m_alpha_high_top','m_alpha_high_bot','tau_idx_low_top','tau_idx_low_bot','tau_alpha_low_top','tau_alpha_low_bot','tau_idx_high_top','tau_idx_high_bot','tau_alpha_high_top','tau_alpha_high_bot','mu_idx_P1','z_idx_P1','unaffordable_low','unaffordable_high','sigma_research',
                  'value_private','value_IPO','value_bankrupt','sigma_terminal',
                  'm_idx_bot','m_idx_top','mu_idx_P3','tau_idx_P3','z_idx_P3','m_alpha_bot','m_alpha_top','VC_yes_prob','VC_no_prob',
                  'discount','mu_weight_matrix_P4','macro_weight_matrix_P4','post_process']
    
    def __init__(self,simple=False,**kwds):        
        '''
        Instantiate a new BiotechType with given data.
        '''       
        # Initialize a basic AgentType
        AgentType.__init__(self,cycles=0,time_flow=True,pseudo_terminal=True,**kwds)
        self.simple = simple
        self.sim_N = 100000 # number of simulated states per firm per month

        # Add biotech-type specific objects, copying to create independent versions
        self.time_vary = ['solveAPeriod']
        if simple:
            self.time_inv = deepcopy(BiotechType.input_list_alt)
            self.solveAPeriod = [solvePhase1, solvePhase2, solvePhase3alt, solvePhase4]
            self.param_count = BiotechType.param_count_alt
            self.integer_list = BiotechType.integer_list_alt
            self.param_list = BiotechType.param_list_alt
        else:
            self.time_inv = deepcopy(BiotechType.input_list)
            self.solveAPeriod = [solvePhase1, solvePhase2, solvePhase3, solvePhase4]
            self.param_count = BiotechType.param_count
            self.integer_list = BiotechType.integer_list
            self.param_list = BiotechType.param_list
        
    def updateParams(self,params_vec):
        for j in range(self.param_count):
            if j in self.integer_list:
                setattr(self,self.param_list[j],int(params_vec[j]))
            else:
                setattr(self,self.param_list[j],params_vec[j])
        self.init_state_mean = np.array([self.money_init_mean,self.mu_init_mean,self.tau_init_mean])
        self.init_state_covar = np.array([[self.money_init_var,self.money_mu_covar,self.money_tau_covar],[self.money_mu_covar,self.mu_init_var,self.mu_tau_covar],[self.money_tau_covar,self.mu_tau_covar,self.tau_init_var]])
                
    def preSolve(self):
        '''
        Pre-solution routine initializes the random number generator for the
        simulation and makes sure not to do post-processing while solving.
        '''
        self.post_process = False
        self.RNG = np.random.RandomState(31382)
                
    def postSolve(self):
        '''
        Post-solution processing for a BiotechType involves creating value function
        interpolations and some other useful objects.  Will eventually convert
        to "double density" versions of this.
        '''
        # Solve one more month of the model
        self.post_process = True
        self.timeRev()
        self.solution = solveACycle(self,self.solution[-1])
        self.post_process = False
        
        # Make overall value functions 
        self.timeFwd()
        self.vFunc1 = QuadlinearInterp(self.solution[0].v_array,self.money_grid,self.mu_grid,self.tau_grid,self.macro_grid)
        self.vFunc2 = QuadlinearInterp(self.solution[1].v_array,self.money_grid,self.mu_grid,self.tau_grid,self.macro_grid)
        self.vFunc3 = QuadlinearInterp(self.solution[2].v_array,self.money_grid,self.mu_grid,self.tau_grid,self.macro_grid)
        self.vFunc4 = QuadlinearInterp(self.solution[3].v_array,self.money_grid,self.mu_grid,self.tau_grid,self.macro_grid)
        
        # Make some additional functions for the simulation
        self.vFuncLow = QuadlinearInterp(self.solution[0].v_low_research,self.money_grid,self.mu_grid,self.tau_grid,self.macro_grid)
        self.vFuncHigh = QuadlinearInterp(self.solution[0].v_high_research,self.money_grid,self.mu_grid,self.tau_grid,self.macro_grid)
        self.vPrivate = lambda mu,tau,macro : np.exp(self.theta_0 + self.theta_mu*mu + self.theta_tau*tau + self.theta_mu_tau*mu*tau + self.theta_z*macro)
        self.vIPO = lambda mu,tau,macro : np.exp(self.lambda_0 + self.lambda_mu*mu + self.lambda_tau*tau + self.lambda_mu_tau*mu*tau + self.lambda_z*macro)
        self.termShockProb = lambda shocks : stats.norm.pdf(np.log(shocks)/self.sigma_terminal + 0.5*self.sigma_terminal)
        if self.simple:
            self.pFunding = lambda money,mu,tau,macro : np.exp(self.pi_0 + self.pi_m*money + self.pi_mu*mu + self.pi_tau*tau + self.pi_mu_tau*mu*tau + self.pi_z*macro)
        else:
            self.pReject = QuadlinearInterp(self.solution[2].reject_prob,self.money_grid,self.mu_grid,self.tau_grid,self.macro_grid)
        self.VCvalFunc = lambda mu,tau,macro : np.exp(self.gamma_0 + self.gamma_mu*mu + self.gamma_tau*tau + self.gamma_mu_tau*mu*tau + self.gamma_z*macro)
        self.VCshockProb = lambda shocks : stats.norm.pdf(np.log(shocks)/self.sigma_VC + 0.5*self.sigma_VC)
   
    def drawInitStates(self,firm_N,macro):
        '''
        Generates arrays of firm_N*sim_N initial states for firms, conditional
        on the macro state and choosing to found.
        '''
        # Draw random initial states
        N = firm_N*self.sim_N
        random_states = self.RNG.multivariate_normal(self.init_state_mean,self.init_state_covar,N)
        money_random = np.exp(random_states[:,0])
        mu_random = random_states[:,1]
        tau_random = np.exp(random_states[:,2])
        macro_all = macro*np.ones_like(money_random)
        
        # Calculate the expected value of founding the firm and not founding
        v_found = self.vFunc1(money_random,mu_random,tau_random,macro_all)
        v_dont = self.value_outside + money_random
        
        # Compute the probability of founding a firm in each initial state
        v_both = np.stack((v_dont,v_found),axis=1)
        v_better = np.tile(np.max(v_both,axis=1),(1,2))
        exp_v_both = np.exp((v_both - v_better)/self.sigma_found)
        prob_found = exp_v_both[:,1]/np.sum(exp_v_both,axis=1)
        
        # Resample the random states based on their relative probabilities
        p_weight = prob_found/np.sum(prob_found)
        init_states = bootstrapSampleFromData(random_states,weights=p_weight,seed=self.RNG.randint(2**31-1))
        money_init = np.exp(init_states[:,0])
        mu_init = init_states[:,1]
        tau_init = np.exp(init_states[:,2])
        
        # Return the distribution of initial states
        return money_init, mu_init, tau_init
        
    def simOneMonth(self,t):
        '''
        Simulates month t of the data, returning the log likelihood of each
        observation in the period.  Takes only period t as an input.
        '''
        # Get the current macro state
        macro_t = self.macro_path[t]
        
        # Add newly founded firms to the list of active firms
        new_firms = np.where(self.start_array[:,t])[0]
        if new_firms.size > 0:
            money_new, mu_new, tau_new = self.drawInitStates(new_firms.size,macro_t)
            money_1 = np.vstack((self.money_t,money_new))
            mu_1 = np.vstack((self.mu_t,mu_new))
            tau_1 = np.vstack((self.tau_t,tau_new))
            self.active_firms = np.hstack((self.active_firms,new_firms))
        else:
            money_1 = self.money_t
            mu_1 = self.mu_t
            tau_1 = self.tau_t
        macro_all = macro_t*np.ones_like(money_1)
        
        # Simulate Phase I: research choice and state updating
        v_zero = self.vFunc2(money_1,mu_1,tau_1,macro_all)
        v_low = self.vFuncLow(money_1,mu_1,tau_1,macro_all)
        v_low[money_1 < self.low_cost] = -np.inf
        v_high = self.vFuncHigh(money_1,mu_1,tau_1,macro_all)
        v_high[money_1 < self.high_cost] = -np.inf
        v_all = np.stack((v_zero,v_low,v_high),2) + self.RNG.gumbel(scale=self.sigma_research,size=(self.active_firms.size,self.sim_N,3))
        r_choice = np.argmax(v_all,axis=2)
        r_shocks = self.RNG.normal(size=(self.active_firms.size,self.sim_N))
        cost_vec = np.array([0,self.low_cost,self.high_cost])
        intensity_vec = np.array([0,self.low_intensity,self.high_intensity])        
        money_2 = money_1 - cost_vec[r_choice]
        mu_2 = (money_1*tau_1 + r_shocks*np.sqrt(intensity_vec[r_choice]))/(tau_1 + intensity_vec[r_choice])
        tau_2 = tau_1 + intensity_vec[r_choice]
        
        # Simulate Phase II: termination decision
        v_continue = self.vFunc3(money_2,mu_2,tau_2,macro_all)
        v_sale_premoney = self.vPrivate(money_2,mu_2,tau_2,macro_all)
        v_sale = v_sale_premoney + money_2
        v_IPO_premoney = self.vIPO(money_2,mu_2,tau_2,macro_all)
        v_IPO = v_IPO_premoney + money_2
        v_all = np.stack((v_continue,v_sale,v_IPO),axis=2)
        v_best = np.max(v_all,axis=2)
        v_all = v_all - np.tile(np.reshape(v_best,(self.active_firms.size,self.sim_N,1)),(1,1,3))
        v_sum = np.sum(np.exp(v_all),axis=2)
        actions = self.action_array[self.active_firms,t]
        v_action = v_all[np.tile(np.reshape(np.arange(self.active_firms.size),(self.active_firms.size,1)),(1,self.sim_N)),np.tile(np.reshape(np.arange(self.sim_N),(1,self.sim_N)),(self.active_firms.size,1)),np.tile(np.reshape(actions,(self.active_firms.size,1)),(1,self.sim_N))]
        p_action = v_action/v_sum
        action_weight = p_action/np.tile(np.reshape(np.sum(p_action,axis=1),(self.active_firms.size,1)),(1,self.sim_N))
        sellers = self.sale_array[self.active_firms,t]
        IPOers = self.IPO_array[self.active_firms,t]
        continuers = np.logical_not(np.logical_or(sellers,IPOers))
        sale_obs_value = self.value_array[self.active_firms[sellers],t]
        IPO_obs_value = self.value_array[self.active_firms[IPOers],t]
        sale_obs_value_premoney = np.tile(np.reshape(sale_obs_value,(np.sum(sellers),1)),(1,self.sim_N)) - money_2[sellers,:]
        sale_obs_value_premoney[sale_obs_value_premoney < 0.000001] = 0.000001 # just in case
        IPO_obs_value_premoney = np.tile(np.reshape(IPO_obs_value,(np.sum(IPOers),1)),(1,self.sim_N)) - money_2[IPOers,:]
        IPO_obs_value_premoney[IPO_obs_value_premoney < 0.000001] = 0.000001 # just in case
        sale_shock = sale_obs_value_premoney/v_sale_premoney[sellers,:]
        IPO_shock = IPO_obs_value_premoney/v_sale_premoney[IPOers,:]
        sale_shock_prob = self.termShockProb(sale_shock)
        IPO_shock_prob = self.termShockProb(IPO_shock)
        self.LL_action[self.active_firms,t] = np.log(np.mean(p_action,axis=1))
        self.LL_value[self.active_firms[sellers],t] = np.log(np.sum(action_weight[sellers,:]*sale_shock_prob,axis=1))
        self.LL_value[self.active_firms[IPOers],t] = np.log(np.sum(action_weight[IPOers,:]*IPO_shock_prob,axis=1))
        
        # Only keep firms that did not terminate        
        action_weight = action_weight[continuers,:]
        self.active_firms = self.active_firms[continuers]
        money_3 = money_2[continuers,:,:,:]
        mu_3 = mu_2[continuers,:,:,:]
        tau_3 = tau_2[continuers,:,:,:]
        macro_all = macro_t*np.ones_like(money_3)
        
        # Simulate Phase III: venture capital decision
        if self.simple:
            1 + 1
        else:
            accepters = self.VC_array[self.active_firms,t]
            rejecters = np.logical_not(accepters)
            obs_valuation = np.tile(np.reshape(self.value_array[self.active_firms[accepters],t],(np.sum(accepters),1)),(1,self.sim_N))
            sharesold = np.tile(np.reshape(self.share_array[self.active_firms[accepters],t],(np.sum(accepters),1)),(1,self.sim_N))
            cash_injection = np.tile(np.reshape(self.cash_array[self.active_firms[accepters],t],(np.sum(accepters),1)),(1,self.sim_N))
            exp_valuation = self.VCvalFunc(money_3[accepters,:],mu_3[accepters,:],tau_3[accepters,:],macro_all)
            VC_shock = obs_valuation/exp_valuation
            VC_shock_prob = self.VCshockProb(VC_shock)
            temp = VC_shock_prob*action_weight[accepters,:]
            VC_weight = temp/np.tile(np.reshape(np.sum(temp,axis=1),(np.sum(accepters),1)),(1,self.sim_N))
            v_accept = (1.0-sharesold)*self.vFunc4(money_3[accepters,:]+cash_injection,mu_3[accepters,:],tau_3[accepters,:],macro_all[accepters,:]) - self.VC_cost
            v_reject = self.vFunc4(money_3[accepters,:],mu_3[accepters,:],tau_3[accepters,:],macro_all[accepters,:])
            exp_v_diff = np.exp(v_reject - v_accept)
            p_accept = 1.0/(1.0 + exp_v_diff)
            p_reject = self.pReject(money_3[rejecters,:],mu_3[rejecters,:],tau_3[rejecters,:],macro_all[rejecters,:])
            self.LL_value[self.active_firms[accepters],t] = np.log(np.sum(temp,axis=1))
            self.LL_choice[self.active_firms[accepters],t] = np.log(np.sum(p_accept*VC_weight,axis=1))
            self.LL_choice[self.active_firms[rejecters],t] = np.log(np.sum(p_reject*action_weight,axis=1))
            
        
    def update(self):
        '''
        Use primitive parameters to construct grids and presolve each phase.
        '''       
        # Construct discrete state grids for each dimension
        self.money_grid = setupGridsExpMult(self.money_min,self.money_max,self.money_N,timestonest=2)
        self.mu_grid = np.linspace(self.mu_min,self.mu_max,self.mu_N)
        self.tau_grid = setupGridsExpMult(self.tau_min,self.tau_max,self.tau_N,timestonest=1)
        self.macro_grid = np.linspace(self.macro_min,self.macro_max,self.macro_N)
        
        # Make the technology shock objects
        self.Qshock_values = np.array([0.0,self.Qshock_small,self.Qshock_big])
        self.Qshock_probs = np.array([1-self.Qprob_small-self.Qprob_big,self.Qprob_small,self.Qprob_big])
        
        # Precalculate objects for each period, generating a *lot* of arrays, etc
        self.mu_weight_matrix_low, self.mu_weight_matrix_high, self.m_idx_low_top, self.m_idx_low_bot, self.m_alpha_low_top, self.m_alpha_low_bot, self.m_idx_high_top, self.m_idx_high_bot, self.m_alpha_high_top, self.m_alpha_high_bot, self.tau_idx_low_top, self.tau_idx_low_bot, self.tau_alpha_low_top, self.tau_alpha_low_bot, self.tau_idx_high_top, self.tau_idx_high_bot, self.tau_alpha_high_top, self.tau_alpha_high_bot, self.mu_idx_P1, self.z_idx_P1, self.unaffordable_low, self.unaffordable_high = precalcPhase1(self.money_grid,self.mu_grid,self.tau_grid,self.macro_grid,self.low_intensity,self.high_intensity,self.low_cost,self.high_cost)
        self.value_private, self.value_IPO, self.value_bankrupt = precalcPhase2(self.money_grid,self.mu_grid,self.tau_grid,self.macro_grid,self.theta_0,self.theta_mu,self.theta_tau,self.theta_mu_tau,self.theta_z,self.lambda_0,self.lambda_mu,self.lambda_tau,self.lambda_mu_tau,self.lambda_z,self.V_b,self.sigma_terminal)
        if self.simple:
            self.m_idx_bot, self.m_idx_top, self.mu_idx_P3, self.tau_idx_P3, self.z_idx_P3, self.m_alpha_bot, self.m_alpha_top, self.VC_yes_prob, self.VC_no_prob = precalcPhase3alt(self.money_grid,self.mu_grid,self.tau_grid,self.macro_grid,self.gamma_0,self.gamma_mu,self.gamma_tau,self.gamma_mu_tau,self.gamma_z,self.sigma_W,self.W_N,self.sharesold_fixed,self.pi_0,self.pi_m,self.pi_mu,self.pi_tau,self.pi_mu_tau,self.pi_z)
        else:
            self.m_idx_bot, self.m_idx_top, self.mu_idx_P3, self.tau_idx_P3, self.z_idx_P3, self.m_alpha_bot, self.m_alpha_top, self.sharekept = precalcPhase3(self.money_grid,self.mu_grid,self.tau_grid,self.macro_grid,self.gamma_0,self.gamma_mu,self.gamma_tau,self.gamma_mu_tau,self.gamma_z,self.sigma_W,self.W_N,self.sharesold_a,self.sharesold_b,self.sharesold_N)
        self.macro_weight_matrix_P4, self.mu_weight_matrix_P4 = precalcPhase4(self.macro_grid,self.mu_grid,self.rho,self.Qshock_values,self.Qshock_probs)
        
        # Make an initial guess of the value array
        v_array_terminal = np.zeros((self.money_N,self.mu_N,self.tau_N,self.macro_N))
        self.solution_terminal = BiotechSolution(v_array = v_array_terminal)


'''
A class representing the solution to a phase in the model.
'''
class BiotechSolution():
    
    def __init__(self,v_array):
        self.v_array = v_array
        
    def distance(self,other):
        v_array_A = self.v_array
        v_array_B = other.v_array
        if v_array_A.shape == v_array_B.shape:
            distance = np.max(np.abs((v_array_A - v_array_B)/v_array_A))
        else:
            distance = 100000
        print(distance)
        return distance
            

def precalcPhase4(macro_grid,mu_grid,rho,Qshock_values,Qshock_probs):
    '''
    Generates weighting matrices to approximate the AR(1) macro process and the
    discrete quality shocks.  The outputs are used to solve the phase 4 model
    with only matrix operations.
    
    Parameters:
    -------------
    macro_grid: numpy.array
        The grid of macro states z_t for the model.
    rho: float
        Correlation of z_t and z_{t+1} for the model.
    mu_grid: numpy.array
        The grid of quality belief mean states mu_t for the model.
    Qshock_values: numpy.array
        The quality shocks that might occur between periods.
    Qshock_probs: numpy.array
        The probability of each quality shock occurring.
        
    Returns:
    -----------
    macro_weight_matrix: numpy.array
        A z_n X z_n transition array.  The i,j-th element represents the weight
        put on future macro state i from current macro state j.
    mu_weight_matrix: numpy.array
        A mu_n X mu_n transition array.  The i,j-th element represents the weight
        put on future mu state i from current mu state j.
    '''
    # Calculate the macro state weighting matrix
    z_n = macro_grid.size
    macro_weight_matrix = np.zeros((z_n,z_n)) + np.nan   
    sigma = 1.0 - rho
    for i in range(z_n):
        macro_mean_next = rho*macro_grid[i]
        macro_weight_matrix[:,i] = makeMarkovApproxToNormal(macro_grid,macro_mean_next,sigma)
        
    # Calculate the belief mean weighting matrix (from technology shocks)
    mu_n = mu_grid.size
    Qshock_n = Qshock_values.size
    mu_weight_matrix = np.zeros((mu_n,mu_n))
    for i in range(mu_n):
        mu_next = mu_grid[i] - Qshock_values
        mu_pos = np.searchsorted(mu_grid,mu_next)
        mu_pos[mu_pos < 1] = 1
        mu_pos[mu_pos > mu_n-1] = mu_n-1
        bot = mu_grid[mu_pos-1]
        top = mu_grid[mu_pos]
        alpha = (mu_next-bot)/(top-bot)
        for q in range(Qshock_n):
            mu_weight_matrix[mu_pos[q]-1,i] = mu_weight_matrix[mu_pos[q]-1,i] + Qshock_probs[q]*(1.0-alpha[q])
            mu_weight_matrix[mu_pos[q],i] = mu_weight_matrix[mu_pos[q],i] + Qshock_probs[q]*alpha[q]
            
    return macro_weight_matrix, mu_weight_matrix
    
    

def precalcPhase3(money_grid,mu_grid,tau_grid,macro_grid,gamma_0,gamma_mu,gamma_tau,gamma_mu_tau,gamma_z,sigma_W,W_N,sharesold_a,sharesold_b,sharesold_N):
    '''
    Generates index and weighting arrays for the phase 3 model.
    
    Parameters:
    -------------
    money_grid: numpy.array
        The grid of money (cash on hand) states m_t for the model.
    mu_grid: numpy.array
        The grid of belief mean states mu_t for the model.
    tau_grid: numpy.array
        The grid of belief inverse variance states tau_t for the model.
    macro_grid: numpy.array
        The grid of macro states z_t for the model.
    gamma_0 : float
        Constant term in expected (log) valuation-at-funding function.
    gamma_mu : float
        Coefficient on belief mean in expected (log) valuation-at-funding function.
    gamma_tau : float
        Coefficient on belief inverse variance in expected (log) valuation-at-
        funding function.
    gamma_mu_tau : float
        Coefficient on interaction b/w belief mean and inverse variance in
        expected (log) valuation-at-funding function.
    gamma_z : float
        Coefficient on macroeconomic state in expected (log) valuation-at-funding
        function.
    sigma_W : float
        Standard deviation of lognormal shocks to valuation-at-funding.
    W_N : int
        Number of points in the discrete distribution of valuation-at-funding
        for each state space point.
    sharesold_a : float
        The a (or alpha) parameter of the beta distribution of sharesold.
    sharesold_b : float
        The b (or beta) parameter of the beta distribution of sharesold.
    sharesold_N : int
        Number of points in the discrete approximation to the lognormal of eta.
       
    Returns:
    -----------
    m_idx_bot : np.array
        Array for the lower indices of money for each gridpoint and each offer.
    m_idx_top : np.array
        Array for the upper indices of money for each gridpoint and each offer.
    mu_idx_P3 : np.array
        Array for the indices of belief mean for each gridpoint and each offer.
    tau_idx_P3 : np.array
        Array for the indices of belief inverse variance for each gridpoint and each offer.
    z_idx_P3 : np.array
        Array for the indices of the macro state for each gridpoint and each offer.
    m_alpha_bot : np.array
        Array for the weights on the lower money gridpoint.
    m_alpha_top : np.array
        Array for the weights on the upper money gridpoint.
    sharekept_array : np.array
        Array for the share of the firm retained by "original" owners (1-eta).
    '''
    # Make arrays of the four state variables
    money_N = money_grid.size
    mu_N = mu_grid.size
    tau_N = tau_grid.size
    macro_N = macro_grid.size
    money_array = np.tile(np.reshape(money_grid,(money_N,1,1,1)),(1,mu_N,tau_N,macro_N))
    mu_array = np.tile(np.reshape(mu_grid,(1,mu_N,1,1)),(money_N,1,tau_N,macro_N))
    tau_array = np.tile(np.reshape(tau_grid,(1,1,tau_N,1)),(money_N,mu_N,1,macro_N))
    macro_array = np.tile(np.reshape(macro_grid,(1,1,1,macro_N)),(money_N,mu_N,tau_N,1))
    
    # Make a joint distribution of lognormal valuation shocks and sharesold offers
    valuation_shock_dist = calculateLognormalDiscreteApprox(W_N,0.0,sigma_W)
    sharesold_dist = calculateBetaDiscreteApprox(sharesold_N,sharesold_a,sharesold_b)
    joint_dist = createFlatStateSpaceFromIndepDiscreteProbs(valuation_shock_dist,sharesold_dist)
    offer_N = W_N*sharesold_N
    val_shock_array = np.tile(np.reshape(joint_dist[1],(1,1,1,1,offer_N)),(money_N,mu_N,tau_N,macro_N,1))
    sharesold_array = np.tile(np.reshape(joint_dist[2],(1,1,1,1,offer_N)),(money_N,mu_N,tau_N,macro_N,1))
    
    # Make an array of expected valuation-at-funding from the project quality
    valuation_avg_array = np.reshape(np.exp(gamma_0 + gamma_mu*mu_array + gamma_tau*tau_array + gamma_mu_tau*mu_array*tau_array + gamma_z*macro_array),(money_N,mu_N,tau_N,macro_N,1))
    
    # Adjust to account for the valuation shock and previous money
    money_array_BIG = np.tile(np.reshape(money_array,((money_N,mu_N,tau_N,macro_N,1))),(1,1,1,1,offer_N))
    valuation_array = val_shock_array*np.tile(valuation_avg_array,(1,1,1,1,offer_N)) + money_array_BIG
    
    # Calculate cash-on-hand after accepting each offer in the distribution
    cashonhand_array = sharesold_array*valuation_array/(1.0-sharesold_array) + money_array_BIG
    
    # Find the position of each cash-on-hand value in the money grid, obeying boundaries
    m_idx_top = np.searchsorted(money_grid,cashonhand_array).astype(np.uint8)
    m_idx_top[m_idx_top < 1] = 1
    m_idx_top[m_idx_top > (money_N-1)] = money_N-1
    m_idx_bot = m_idx_top - 1 # bottom index is one less than top index
    
    # Calculate relative position of cash-on-hand between the bottom and top boundaries
    m_alpha_top = (cashonhand_array - money_grid[m_idx_bot])/(money_grid[m_idx_top] - money_grid[m_idx_bot]).astype(np.float32)
    m_alpha_bot = 1.0 - m_alpha_top # weight on bottom point is 1 minus weight on top point
    
    # Create other precomputed objects (much more simple ones)
    sharekept_array = (1.0 - sharesold_array).astype(np.float32)
    mu_idx_P3 = np.tile(np.reshape(np.array(range(mu_N)),(1,mu_N,1,1,1)),(money_N,1,tau_N,macro_N,offer_N)).astype(np.uint8)
    tau_idx_P3 = np.tile(np.reshape(np.array(range(tau_N)),(1,1,tau_N,1,1)),(money_N,mu_N,1,macro_N,offer_N)).astype(np.uint8)
    z_idx_P3 = np.tile(np.reshape(np.array(range(macro_N)),(1,1,1,macro_N,1)),(money_N,mu_N,tau_N,1,offer_N)).astype(np.uint8)
    
    # Return the precalculated outputs
    return m_idx_bot, m_idx_top, mu_idx_P3, tau_idx_P3, z_idx_P3, m_alpha_bot, m_alpha_top, sharekept_array
    
    
    
def precalcPhase3alt(money_grid,mu_grid,tau_grid,macro_grid,gamma_0,gamma_mu,gamma_tau,gamma_mu_tau,gamma_z,sigma_W,W_N,sharesold_fixed,pi_0,pi_m,pi_mu,pi_tau,pi_mu_tau,pi_z):
    '''
    Generates index and weighting arrays for the alternate phase 3 model, which
    has exogenous venture capital funding events.
    
    Parameters:
    -------------
    money_grid: numpy.array
        The grid of money (cash on hand) states m_t for the model.
    mu_grid: numpy.array
        The grid of belief mean states mu_t for the model.
    tau_grid: numpy.array
        The grid of belief inverse variance states tau_t for the model.
    macro_grid: numpy.array
        The grid of macro states z_t for the model.
    gamma_0 : float
        Constant term in expected (log) valuation-at-funding function.
    gamma_mu : float
        Coefficient on belief mean in expected (log) valuation-at-funding function.
    gamma_tau : float
        Coefficient on belief inverse variance in expected (log) valuation-at-
        funding function.
    gamma_mu_tau : float
        Coefficient on interaction b/w belief mean and inverse variance in
        expected (log) valuation-at-funding function.
    gamma_z : float
        Coefficient on macroeconomic state in expected (log) valuation-at-funding
        function.
    sigma_W : float
        Standard deviation of lognormal shocks to valuation-at-funding.
    W_N : int
        Number of points in the discrete distribution of valuation-at-funding
        for each state space point.
    sharesold_fixed : float
        A fixed share of the firm that is sold to venture capitalists in a round.
    pi_0 : float
        Constant in probability of receiving funding function.
    pi_mu : float
        Coefficient for money in probability of receiving funding function.
    pi_mu : float
        Coefficient for belief mean in probability of receiving funding function.
    pi_tau : float
        Coefficient for belief variance in probability of receiving funding function.
    pi_mu_tau : float
        Coefficient for interaction in probability of receiving funding function.
    pi_z : float
        Coefficient for macro state in probability of receiving funding function.
    
       
    Returns:
    -----------
    m_idx_bot : np.array
        Array for the lower indices of money for each gridpoint when VC occurs.
    m_idx_top : np.array
        Array for the upper indices of money for each gridpoint when VC occurs.
    mu_idx_P3 : np.array
        Array for the indices of belief mean for each gridpoint.
    tau_idx_P3 : np.array
        Array for the indices of belief inverse variance for each gridpoint.
    z_idx_P3 : np.array
        Array for the indices of the macro state for each gridpoint.
    m_alpha_bot : np.array
        Array for the weights on the lower money gridpoint when VC occurs.
    m_alpha_top : np.array
        Array for the weights on the upper money gridpoint when VC occurs.
    VC_yes_prob : np.array
        Probability that funding occurs at each state.
    VC_no_prob : np.array
        Probability that funding does not occur at each state.
    '''
    # Make arrays of the four state variables
    money_N = money_grid.size
    mu_N = mu_grid.size
    tau_N = tau_grid.size
    macro_N = macro_grid.size
    money_array = np.tile(np.reshape(money_grid,(money_N,1,1,1)),(1,mu_N,tau_N,macro_N))
    mu_array = np.tile(np.reshape(mu_grid,(1,mu_N,1,1)),(money_N,1,tau_N,macro_N))
    tau_array = np.tile(np.reshape(tau_grid,(1,1,tau_N,1)),(money_N,mu_N,1,macro_N))
    macro_array = np.tile(np.reshape(macro_grid,(1,1,1,macro_N)),(money_N,mu_N,tau_N,1))
    
    # Make a distribution of lognormal valuation shocks
    valuation_shock_dist = calculateLognormalDiscreteApprox(W_N,0.0,sigma_W)
    val_shock_array = np.tile(np.reshape(valuation_shock_dist[1],(1,1,1,1,W_N)),(money_N,mu_N,tau_N,macro_N,1))
    
    # Make an array of expected valuation-at-funding from the project quality
    valuation_avg_array = np.reshape(np.exp(gamma_0 + gamma_mu*mu_array + gamma_tau*tau_array + gamma_mu_tau*mu_array*tau_array + gamma_z*macro_array),(money_N,mu_N,tau_N,macro_N,1))    
    
    # Apply the lognormal valuation shocks and include money
    money_array_BIG = np.tile(np.reshape(money_array,((money_N,mu_N,tau_N,macro_N,1))),(1,1,1,1,W_N))
    valuation_array = val_shock_array*np.tile(valuation_avg_array,(1,1,1,1,W_N)) + money_array_BIG
    
    # Calculate cash-on-hand after accepting each offer in the distribution
    cashonhand_array = sharesold_fixed*valuation_array/(1.0-sharesold_fixed) + money_array_BIG
    
    m_idx_top = np.searchsorted(money_grid,cashonhand_array).astype(np.uint8)
    m_idx_top[m_idx_top < 1] = 1
    m_idx_top[m_idx_top > (money_N-1)] = money_N-1
    m_idx_bot = m_idx_top - 1 # bottom index is one less than top index
    
    # Calculate relative position of cash-on-hand between the bottom and top boundaries
    m_alpha_top = (cashonhand_array - money_grid[m_idx_bot])/(money_grid[m_idx_top] - money_grid[m_idx_bot]).astype(np.float32)
    m_alpha_bot = 1.0 - m_alpha_top # weight on bottom point is 1 minus weight on top point

    # Calculate the probability of getting funding in each state
    pi_array = np.exp(pi_0 + pi_m*money_array + pi_mu*mu_array + pi_tau*tau_array + pi_mu_tau*mu_array*tau_array + pi_z*macro_array)
    VC_yes_prob = pi_array/(1 + pi_array)
    VC_no_prob = 1.0 - VC_yes_prob
    
    # Create other precomputed objects (much more simple ones)
    mu_idx_P3 = np.tile(np.reshape(np.array(range(mu_N)),(1,mu_N,1,1,1)),(money_N,1,tau_N,macro_N,W_N)).astype(np.uint8)
    tau_idx_P3 = np.tile(np.reshape(np.array(range(tau_N)),(1,1,tau_N,1,1)),(money_N,mu_N,1,macro_N,W_N)).astype(np.uint8)
    z_idx_P3 = np.tile(np.reshape(np.array(range(macro_N)),(1,1,1,macro_N,1)),(money_N,mu_N,tau_N,1,W_N)).astype(np.uint8)
    
    # Return the precalculated objects
    return m_idx_bot, m_idx_top, mu_idx_P3, tau_idx_P3, z_idx_P3, m_alpha_bot, m_alpha_top, VC_yes_prob, VC_no_prob




def precalcPhase2(money_grid,mu_grid,tau_grid,macro_grid,theta_0,theta_mu,theta_tau,theta_mu_tau,theta_z,lambda_0,lambda_mu,lambda_tau,lambda_mu_tau,lambda_z,V_b,sigma_terminal):
    '''
    Generates arrays of (exponentiated) expected utility of private sale and
    going public from each state in the grid, for use in the phase 2 model.
    
    Parameters:
    -------------
    money_grid: numpy.array
        The grid of money (cash on hand) states m_t for the model.
    mu_grid: numpy.array
        The grid of belief mean states mu_t for the model.
    tau_grid: numpy.array
        The grid of belief inverse variance states tau_t for the model.
    macro_grid: numpy.array
        The grid of macro states z_t for the model.
    theta_0 : float
        Constant term in expected (log) valuation-from-private-sale function.
    theta_mu : float
        Coefficient on belief mean in expected (log) valuation-from-private-sale
        function.
    theta_tau : float
        Coefficient on belief inverse variance in expected (log) valuation-from-
        private-sale funding function.
    theta_mu_tau : float
        Coefficient on interaction b/w belief mean and inverse variance in
        expected (log) valuation-from-private-sale function.
    theta_z : float
        Coefficient on macroeconomic state in expected (log) valuation-from-
        private-sale function.
    lambda_0 : float
        Constant term in expected (log) valuation-at-IPO function.
    lambda_mu : float
        Coefficient on belief mean in expected (log) valuation-at-IPO function.
    lambda_tau : float
        Coefficient on belief inverse variance in expected (log) valuation-at-
        IPO function.
    lambda_mu_tau : float
        Coefficient on interaction b/w belief mean and inverse variance in
        expected (log) valuation-at-IPO function.
    lambda_z : float
        Coefficient on macroeconomic state in expected (log) valuation-at-IPO
        function.
    V_b : float
        Constant continuation / scrap value of going bankrupt.
    sigma_terminal : float
        Standard deviation of preference shocks for termination decision.
        
    Returns:
    ----------
    value_private : numpy.array
        An array with the precalculated expected value of
        privately selling the firm.
    value_IPO : numpy.array
        An array with the precalculated expected value of
        selling the firm via an initial public offering (go public).
    value_bankrupt : numpy.array
        An array with the constant value of going bankrupt.
    '''
    # Make arrays of the four state variables
    money_N = money_grid.size
    mu_N = mu_grid.size
    tau_N = tau_grid.size
    macro_N = macro_grid.size
    money_array = np.tile(np.reshape(money_grid,(money_N,1,1,1)),(1,mu_N,tau_N,macro_N))
    mu_array = np.tile(np.reshape(mu_grid,(1,mu_N,1,1)),(money_N,1,tau_N,macro_N))
    tau_array = np.tile(np.reshape(tau_grid,(1,1,tau_N,1)),(money_N,mu_N,1,macro_N))
    macro_array = np.tile(np.reshape(macro_grid,(1,1,1,macro_N)),(money_N,mu_N,tau_N,1))
    
    # Calculate expected value from private sale and IPO
    value_private = np.exp(theta_0 + theta_mu*mu_array + theta_tau*tau_array + theta_mu_tau*mu_array*tau_array + theta_z*macro_array) + money_array
    value_IPO = np.exp(lambda_0 + lambda_mu*mu_array + lambda_tau*tau_array + lambda_mu_tau*mu_array*tau_array + lambda_z*macro_array) + money_array
    
    # Adjust the expected value arrays for output
    value_IPO = value_IPO/sigma_terminal
    value_private = value_private/sigma_terminal
    
    # Make the constant array of bankruptcy / scrap value
    value_bankrupt = V_b/sigma_terminal*np.ones_like(value_IPO)
    
    # Return the precalculated arrays
    return value_private, value_IPO, value_bankrupt



def precalcPhase1(money_grid,mu_grid,tau_grid,macro_grid,low_intensity,high_intensity,low_cost,high_cost):
    '''
    Generates weighting matrices that approximate the normal research shock
    process in Phase 1 of the problem, as well as accompanying weight and index
    arrays for the other dimensions.  All of the (non-Markov) output arrays
    are of size money_N x mu_N x tau_N x macro_N.
    
    Parameters:
    -------------
    money_grid: numpy.array
        The grid of money (cash on hand) states m_t for the model.
    mu_grid: numpy.array
        The grid of belief mean states mu_t for the model.
    tau_grid: numpy.array
        The grid of belief inverse variance states tau_t for the model.
    macro_grid: numpy.array
        The grid of macro states z_t for the model.
    low_intensity : float
        Inverse variance of low intensity research signal.
    high_intensity : float
        Inverse variance of high intensity research signal.
    low_cost : float
        Money cost of low intensity research.
    high_cost : float
        Money cost of high intensity research.
        
    Returns:
    -----------
    mu_weight_matrix_low : numpy.array
        Markov transition matrix for belief mean mu after low intensity research.
        In layer k, the i,j-th element gives the probability of getting future
        mu state i from current mu state j when tau = tau_grid[k].
    mu_weight_matrix_high : numpy.array
        Markov transition matrix for belief mean mu after high intensity research.
        In layer k, the i,j-th element gives the probability of getting future
        mu state i from current mu state j when tau = tau_grid[k].
    m_idx_low_top : numpy.array
        An array of money indices for when low research is chosen (top end).
    m_idx_low_bot : numpy.array
        An array of money indices for when low research is chosen (bottom end).
    m_alpha_low_top : numpy.array
        An array of money weights for when low research is chosen (top end).
    m_alpha_low_bot : numpy.array
        An array of money weights for when low research is chosen (bottom end).
    m_idx_high_top : numpy.array
        An array of money indices for when high research is chosen (top end).
    m_idx_high_bot : numpy.array
        An array of money indices for when high research is chosen (bottom end).
    m_alpha_high_top : numpy.array
        An array of money weights for when high research is chosen (top end).
    m_alpha_high_bot : numpy.array
        An array of money weights for when high research is chosen (top end).
    tau_idx_low_top : numpy.array
        An array of tau indices for when low research is chosen (top end).
    tau_idx_low_bot : numpy.array
        An array of tau indices for when low research is chosen (bottom end).
    tau_alpha_low_top : numpy.array
        An array of tau weights for when low research is chosen (top end).
    tau_alpha_low_bot : numpy.array
        An array of tau weights for when low research is chosen (bottom end).
    tau_idx_high_top : numpy.array
        An array of tau indices for when high research is chosen (top end).
    tau_idx_high_bot : numpy.array
        An array of tau indices for when high research is chosen (bottom end).
    tau_alpha_high_top : numpy.array
        An array of tau weights for when high research is chosen (top end).
    tau_alpha_high_bot : numpy.array
        An array of tau weights for when high research is chosen (bottom end).
    mu_idx_P1 : numpy.array
        An array of belief mean indices.
    z_idx_P1 : numpy.array
        An array of macro state indices.
    unaffordable_low : numpy.array
        A boolean array indicating the states in which low intensity research
        is unaffordable because low_cost > money.
    unaffordable_high : numpy.array
        A boolean array indicating the states in which high intensity research
        is unaffordable because high_cost > money.
    '''
    # Get the size of each state dimension
    money_N = money_grid.size
    mu_N = mu_grid.size
    tau_N = tau_grid.size
    macro_N = macro_grid.size
    
    # Make a Markov array for low research outcomes and high research outcomes
    mu_weight_matrix_low = np.zeros((mu_N,mu_N,tau_N)) + np.nan
    mu_weight_matrix_high = np.zeros((mu_N,mu_N,tau_N)) + np.nan
    for k in range(tau_N):
        tau = tau_grid[k]
        for j in range(mu_N):
            mu = mu_grid[j]
            mu_weight_matrix_low[:,j,k] = makeMarkovApproxToNormal(mu_grid,mu*tau/(tau+low_intensity),np.sqrt(low_intensity)/(tau+low_intensity))
            mu_weight_matrix_high[:,j,k] = makeMarkovApproxToNormal(mu_grid,mu*tau/(tau+high_intensity),np.sqrt(high_intensity)/(tau+high_intensity))
            
    # Make weighting and indexing arrays for money (after paying research costs)
    money_idx_vec_low = np.searchsorted(money_grid,money_grid - low_cost)
    money_idx_vec_low[money_idx_vec_low < 1] = 1
    money_alpha_vec_low = ((money_grid - low_cost) - money_grid[money_idx_vec_low-1])/(money_grid[money_idx_vec_low] - money_grid[money_idx_vec_low-1])
    m_idx_low_top = np.tile(np.reshape(money_idx_vec_low,(money_N,1,1,1)),(1,mu_N,tau_N,macro_N)).astype(np.uint8)
    m_idx_low_bot = np.tile(np.reshape(money_idx_vec_low-1,(money_N,1,1,1)),(1,mu_N,tau_N,macro_N)).astype(np.uint8)
    m_alpha_low_top = np.tile(np.reshape(money_alpha_vec_low,(money_N,1,1,1)),(1,mu_N,tau_N,macro_N)).astype(np.float32)
    m_alpha_low_bot = np.tile(np.reshape(1.0-money_alpha_vec_low,(money_N,1,1,1)),(1,mu_N,tau_N,macro_N)).astype(np.float32)
    money_idx_vec_high = np.searchsorted(money_grid,money_grid - high_cost)
    money_idx_vec_high[money_idx_vec_high < 1] = 1
    money_alpha_vec_high = ((money_grid - high_cost) - money_grid[money_idx_vec_high-1])/(money_grid[money_idx_vec_high] - money_grid[money_idx_vec_high-1])
    m_idx_high_top = np.tile(np.reshape(money_idx_vec_high,(money_N,1,1,1)),(1,mu_N,tau_N,macro_N)).astype(np.uint8)
    m_idx_high_bot = np.tile(np.reshape(money_idx_vec_high-1,(money_N,1,1,1)),(1,mu_N,tau_N,macro_N)).astype(np.uint8)
    m_alpha_high_top = np.tile(np.reshape(money_alpha_vec_high,(money_N,1,1,1)),(1,mu_N,tau_N,macro_N)).astype(np.float32)
    m_alpha_high_bot = np.tile(np.reshape(1.0-money_alpha_vec_high,(money_N,1,1,1)),(1,mu_N,tau_N,macro_N)).astype(np.float32)
    
    # Make weighting and indexing arrays for belief (inverse) variance tau
    tau_idx_vec_low = np.searchsorted(tau_grid,tau_grid + low_intensity)
    tau_idx_vec_low[tau_idx_vec_low > (tau_N-1)] = tau_N-1
    tau_alpha_vec_low = ((tau_grid + low_intensity) - tau_grid[tau_idx_vec_low-1])/(tau_grid[tau_idx_vec_low] - tau_grid[tau_idx_vec_low-1])
    tau_idx_low_top = np.tile(np.reshape(tau_idx_vec_low,(1,1,tau_N,1)),(money_N,mu_N,1,macro_N)).astype(np.uint8)
    tau_idx_low_bot = np.tile(np.reshape(tau_idx_vec_low-1,(1,1,tau_N,1)),(money_N,mu_N,1,macro_N)).astype(np.uint8)
    tau_alpha_low_top = np.tile(np.reshape(tau_alpha_vec_low,(1,1,tau_N,1)),(money_N,mu_N,1,macro_N)).astype(np.float32)
    tau_alpha_low_bot = np.tile(np.reshape(1.0-tau_alpha_vec_low,(1,1,tau_N,1)),(money_N,mu_N,1,macro_N)).astype(np.float32)
    tau_idx_vec_high = np.searchsorted(tau_grid,tau_grid + high_intensity)
    tau_idx_vec_high[tau_idx_vec_high > (tau_N-1)] = tau_N-1
    tau_alpha_vec_high = ((tau_grid + high_intensity) - tau_grid[tau_idx_vec_high-1])/(tau_grid[tau_idx_vec_high] - tau_grid[tau_idx_vec_high-1])
    tau_idx_high_top = np.tile(np.reshape(tau_idx_vec_high,(1,1,tau_N,1)),(money_N,mu_N,1,macro_N)).astype(np.uint8)
    tau_idx_high_bot = np.tile(np.reshape(tau_idx_vec_high-1,(1,1,tau_N,1)),(money_N,mu_N,1,macro_N)).astype(np.uint8)
    tau_alpha_high_top = np.tile(np.reshape(tau_alpha_vec_high,(1,1,tau_N,1)),(money_N,mu_N,1,macro_N)).astype(np.float32)
    tau_alpha_high_bot = np.tile(np.reshape(1.0-tau_alpha_vec_high,(1,1,tau_N,1)),(money_N,mu_N,1,macro_N)).astype(np.float32)
    
    # Make index arrays for the other dimensions
    mu_idx_P1 = np.tile(np.reshape(np.array(range(mu_N)),(1,mu_N,1,1)),(money_N,1,tau_N,macro_N)).astype(np.uint8)
    z_idx_P1 = np.tile(np.reshape(np.array(range(macro_N)),(1,1,1,macro_N)),(money_N,mu_N,tau_N,1)).astype(np.uint8)
    
    # Make boolean arrays to indicate which choices are unaffordable
    unaffordable_low = np.tile(np.reshape(low_cost > money_grid,(money_N,1,1,1)),(1,mu_N,tau_N,macro_N))
    unaffordable_high = np.tile(np.reshape(high_cost > money_grid,(money_N,1,1,1)),(1,mu_N,tau_N,macro_N))
    
    
    # Return the outputs
    return mu_weight_matrix_low, mu_weight_matrix_high, m_idx_low_top, m_idx_low_bot, m_alpha_low_top, m_alpha_low_bot, m_idx_high_top, m_idx_high_bot, m_alpha_high_top, m_alpha_high_bot, tau_idx_low_top, tau_idx_low_bot, tau_alpha_low_top, tau_alpha_low_bot, tau_idx_high_top, tau_idx_high_bot, tau_alpha_high_top, tau_alpha_high_bot, mu_idx_P1, z_idx_P1, unaffordable_low, unaffordable_high
    



def solvePhase4(solution_tp1,discount,mu_weight_matrix_P4,macro_weight_matrix_P4):
    '''
    A function that solves Phase IV of the entrepreneur's problem, given the
    solution to Phase I of the problem.
    
    Parameters:
    -------------
    solution_phase1 : BiotechSolution
        The current guess of the solution to Phase I of the model.
    discount : float
        The (monthly) intertemporal discount factor.
    macro_weight_matrix_P4 : numpy.array
        A z_n X z_n transition array.  The i,j-th element represents the weight
        put on future macro state i from current macro state j.
    mu_weight_matrix_P4 : numpy.array
        A mu_n X mu_n transition array.  The i,j-th element represents the weight
        put on future mu state i from current mu state j.
           
    Returns:    (as attributes of solution_tp1)
    -----------
    v_array : numpy.array
        The value function for Phase IV, represented as a 4D array.
    '''
    # Compute the new value array
    v_array_next = solution_tp1.v_array # order: m, mu, tau, z
    v_array_after_zshocks = np.dot(v_array_next,macro_weight_matrix_P4)
    v_array_after_Qshocks = np.dot(np.transpose(v_array_after_zshocks,(0,2,3,1)),mu_weight_matrix_P4) # order: m, tau, z, mu
    v_array = discount*np.transpose(v_array_after_Qshocks,(0,3,1,2))
    
    # Return the value function as the solution
    solution_t = BiotechSolution(v_array = v_array)
    return solution_t
    

def solvePhase3(solution_tp1,m_idx_bot,m_idx_top,mu_idx_P3,tau_idx_P3,z_idx_P3,m_alpha_bot,m_alpha_top,sharekept,sigma_VC,VC_cost,post_process):
    '''
    A function that solves Phase III of the entrepreneur's problem, given the
    solution to Phase IV of the problem.  All of the input arrays are of size 
    (money_N,mu_N,tau_N,macro_N,W_N*sharesold_N).
    
    Parameters:
    ------------
    solution_tp1 : BiotechSolution
        The current guess of the solution to Phase IV of the model.
    m_idx_bot : np.array
        Array for the lower indices of money for each gridpoint and each offer.
    m_idx_top : np.array
        Array for the upper indices of money for each gridpoint and each offer.
    mu_idx_P3 : np.array
        Array for the indices of belief mean for each gridpoint and each offer.
    tau_idx_P3 : np.array
        Array for the indices of belief inverse variance for each gridpoint and each offer.
    z_idx_P3 : np.array
        Array for the indices of the macro state for each gridpoint and each offer.
    m_alpha_bot : np.array
        Array for the weights on the lower money gridpoint.
    m_alpha_top : np.array
        Array for the weights on the upper money gridpoint.
    sharekept : np.array
        Array for the share of the firm retained by "original" owners (1-eta).
    sigma_VC : float
        Standard deviation of preference shocks for accepting or declining VC offer.
    VC_cost : float
        Fixed cost of an accepted VC offer, borne by the "original" owners.
    post_process : bool
        Indicator for whether the solver should produce interpolated functions.
        False during solution, True during postSolve.
        
    Returns:    (as attributes of solution_tp1)
    -----------
    v_array : numpy.array
        The value function for Phase III, represented as a 4D array.
    '''
    # Extract the size of each state variable's grid    
    v_array_next = solution_tp1.v_array.astype(np.float32)
    N = v_array_next.shape
    money_N = N[0]
    mu_N = N[1]
    tau_N = N[2]
    macro_N = N[3]
    offer_N = sharekept.shape[4]
    
    # Make arrays for value of accepting or rejecting each offer
    v_accept = sharekept*(m_alpha_bot*v_array_next[m_idx_bot,mu_idx_P3,tau_idx_P3,z_idx_P3] + m_alpha_top*v_array_next[m_idx_top,mu_idx_P3,tau_idx_P3,z_idx_P3]) - VC_cost
    v_reject = np.tile(np.reshape(v_array_next,(money_N,mu_N,tau_N,macro_N,1)),(1,1,1,1,offer_N))
    
    # Adjust the value arrays, then exponentiate
    v_both = np.stack((v_reject,v_accept),5)    
    v_better = np.max(v_both,axis=5)
    v_worse = np.min(v_both,axis=5)
    exp_v_diff = np.exp((v_worse - v_better)/sigma_VC)
    
    # Calculate the expectation of the optimal choice value
    v_array = sigma_VC*np.mean(np.log(exp_v_diff + 1.0)+v_better/sigma_VC,axis=4)
    
    # Return the value function as the solution
    solution_t = BiotechSolution(v_array = v_array)
    if post_process: # After solving, calculate the rejection probability
        better_idx = np.argmax(v_both,axis=5)
        prob_better = 1.0/(exp_v_diff + 1.0)
        prob_worse = 1.0 - prob_better
        reject_is_worse = better_idx.astype(bool)
        reject_is_better = np.logical_not(reject_is_worse)
        reject_prob_big = np.zeros_like(prob_better)
        reject_prob_big[reject_is_worse] = prob_worse[reject_is_worse]
        reject_prob_big[reject_is_better] = prob_better[reject_is_better]
        reject_prob = np.mean(reject_prob_big,axis=4)
        solution_t.reject_prob = reject_prob
    return solution_t
    
    
    
def solvePhase3alt(solution_tp1,m_idx_bot,m_idx_top,mu_idx_P3,tau_idx_P3,z_idx_P3,m_alpha_bot,m_alpha_top,VC_yes_prob,VC_no_prob):
    '''
    A function that solves Phase III of the entrepreneur's problem, given the
    solution to Phase IV of the problem.  Most of the input arrays are of size 
    (money_N,mu_N,tau_N,macro_N,W_N).  This is the alternate, simple version
    with an exogenous "helicopter drop" venture capital process.
    
    Parameters:
    ------------
    solution_tp1 : BiotechSolution
        The current guess of the solution to Phase IV of the model.
    m_idx_bot : np.array
        Array for the lower indices of money for each gridpoint and each offer.
    m_idx_top : np.array
        Array for the upper indices of money for each gridpoint and each offer.
    mu_idx_P3 : np.array
        Array for the indices of belief mean for each gridpoint and each offer.
    tau_idx_P3 : np.array
        Array for the indices of belief inverse variance for each gridpoint and each offer.
    z_idx_P3 : np.array
        Array for the indices of the macro state for each gridpoint and each offer.
    m_alpha_bot : np.array
        Array for the weights on the lower money gridpoint when VC occurs.
    m_alpha_top : np.array
        Array for the weights on the upper money gridpoint when VC occurs.
    VC_yes_prob : np.array
        Probability that funding occurs at each state.
    VC_no_prob : np.array
        Probability that funding does not occur at each state.
        
    Returns:    (as attributes of solution_tp1)
    -----------
    v_array : numpy.array
        The value function for Phase III, represented as a 4D array.
    '''
    # Extract the Phase II value function
    v_array_next = solution_tp1.v_array.astype(np.float32)
    
    # Calculate expected value if VC occurs at each state
    v_funding = np.mean(m_alpha_bot*v_array_next[m_idx_bot,mu_idx_P3,tau_idx_P3,z_idx_P3] + m_alpha_top*v_array_next[m_idx_top,mu_idx_P3,tau_idx_P3,z_idx_P3],axis=4)
    
    # Take the average of the possibility that funding does or does not occur
    v_array = VC_yes_prob*v_funding + VC_no_prob*v_array_next
    
    # Return the value function as the solution
    solution_t = BiotechSolution(v_array = v_array)
    return solution_t
    
    
    
    
    
def solvePhase2(solution_tp1,value_private,value_IPO,value_bankrupt,sigma_terminal):
    '''
    A function that solves Phase II of the entrepreneur's problem, given the
    solution to Phase III of the problem.  Both of the input arrays are of size 
    (money_N,mu_N,tau_N,macro_N).
    
    Parameters:
    ------------
    solution_tp1 : BiotechSolution
        The current guess of the solution to Phase III of the model.
    value_private : numpy.array
        An array with the precalculated expected value of privately selling the
        firm.
    value_IPO : numpy.array
        An array with the precalculated expected value of selling the firm via
        an initial public offering (go public).
    value_bankrupt : numpy.array
        An array with the constant value of going bankrupt.
    sigma_terminal : float
        Standard deviation of preference shocks for termination decision.
        
    Returns:    (as attributes of solution_tp1)
    -----------
    v_array : numpy.array
        The value function for Phase II, represented as a 4D array.
    '''
    # Get the expected value of continuing the firm
    v_array_next = solution_tp1.v_array
    N = v_array_next.shape
    money_N = N[0]
    mu_N = N[1]
    tau_N = N[2]
    macro_N = N[3]
    value_continue = v_array_next/sigma_terminal
    
    # Find the best option among the three choices at each state
    value_all = np.stack((value_continue,value_private,value_IPO,value_bankrupt),axis=4)
    value_best = np.max(value_all,axis=4)
    value_best_big = np.tile(np.reshape(value_best,(money_N,mu_N,tau_N,macro_N,1)),(1,1,1,1,4))
    
    # Calculate expected value at the beginning of Phase II
    v_array = sigma_terminal*(np.log(np.sum(np.exp(value_all-value_best_big),axis=4)) + value_best)
    
    # Return the value function as the solution
    solution_t = BiotechSolution(v_array = v_array)
    return solution_t
    
    
      
def solvePhase1(solution_tp1, mu_weight_matrix_low, mu_weight_matrix_high, m_idx_low_top, m_idx_low_bot, m_alpha_low_top, m_alpha_low_bot, m_idx_high_top, m_idx_high_bot, m_alpha_high_top, m_alpha_high_bot,tau_idx_low_top, tau_idx_low_bot, tau_alpha_low_top, tau_alpha_low_bot, tau_idx_high_top, tau_idx_high_bot, tau_alpha_high_top, tau_alpha_high_bot, mu_idx_P1, z_idx_P1, unaffordable_low, unaffordable_high, sigma_research, post_process):
    '''
    A function that solves Phase I of the entrepreneur's problem, given the
    solution to Phase II of the problem.  All of the input arrays are of size 
    (money_N,mu_N,tau_N,macro_N), other than the Markov arrays.
    
    Parameters:
    ------------
    solution_tp1 : BiotechSolution
        The current guess of the solution to Phase II of the model.
    mu_weight_matrix_low : numpy.array
        Markov transition matrix for belief mean mu after low intensity research.
        In layer k, the i,j-th element gives the probability of getting future
        mu state i from current mu state j when tau = tau_grid[k].
    mu_weight_matrix_high : numpy.array
        Markov transition matrix for belief mean mu after high intensity research.
        In layer k, the i,j-th element gives the probability of getting future
        mu state i from current mu state j when tau = tau_grid[k].
    m_idx_low_top : numpy.array
        An array of money indices for when low research is chosen (top end).
    m_idx_low_bot : numpy.array
        An array of money indices for when low research is chosen (bottom end).
    m_alpha_low_top : numpy.array
        An array of money weights for when low research is chosen (top end).
    m_alpha_low_bot : numpy.array
        An array of money weights for when low research is chosen (bottom end).
    m_idx_high_top : numpy.array
        An array of money indices for when high research is chosen (top end).
    m_idx_high_bot : numpy.array
        An array of money indices for when high research is chosen (bottom end).
    m_alpha_high_top : numpy.array
        An array of money weights for when high research is chosen (top end).
    m_alpha_high_bot : numpy.array
        An array of money weights for when high research is chosen (top end).
    tau_idx_low_top : numpy.array
        An array of tau indices for when low research is chosen (top end).
    tau_idx_low_bot : numpy.array
        An array of tau indices for when low research is chosen (bottom end).
    tau_alpha_low_top : numpy.array
        An array of tau weights for when low research is chosen (top end).
    tau_alpha_low_bot : numpy.array
        An array of tau weights for when low research is chosen (bottom end).
    tau_idx_high_top : numpy.array
        An array of tau indices for when high research is chosen (top end).
    tau_idx_high_bot : numpy.array
        An array of tau indices for when high research is chosen (bottom end).
    tau_alpha_high_top : numpy.array
        An array of tau weights for when high research is chosen (top end).
    tau_alpha_high_bot : numpy.array
        An array of tau weights for when high research is chosen (bottom end).
    mu_idx_P1 : numpy.array
        An array of belief mean indices.
    z_idx_P1 : numpy.array
        An array of macro state indices.
    unaffordable_low : numpy.array
        A boolean array indicating the states in which low intensity research
        is unaffordable because low_cost > money.
    unaffordable_high : numpy.array
        A boolean array indicating the states in which high intensity research
        is unaffordable because high_cost > money.
    sigma_research
        Standard deviation of preference shocks on research intensity choice.
    post_process : bool
        Indicator for whether the solver should produce interpolated functions.
        False during solution, True during postSolve.
        
    Returns:    (as attributes of solution_tp1)
    -----------
    v_array : numpy.array
        The value function for Phase I, represented as a 4D array.
    '''
    # Unpack and permute the Phase II value function
    v_array_zero = solution_tp1.v_array # order: m, mu, tau, z
    N = v_array_zero.shape
    money_N = N[0]
    mu_N = N[1]
    tau_N = N[2]
    macro_N = N[3]
    v_array_temp = np.transpose(v_array_zero,(0,2,3,1)) # order: m, tau, z, mu
    
    # Compute expected value after doing low and high intensity research
    v_array_low_0 = np.zeros((money_N,tau_N,macro_N,mu_N))
    v_array_high_0 = np.zeros((money_N,tau_N,macro_N,mu_N))
    for k in range(tau_N):
        v_array_low_0[:,k,:,:] = np.dot(v_array_temp[:,k,:,:],np.reshape(mu_weight_matrix_low[:,:,k],(mu_N,mu_N)))
        v_array_high_0[:,k,:,:] = np.dot(v_array_temp[:,k,:,:],np.reshape(mu_weight_matrix_high[:,:,k],(mu_N,mu_N)))
    v_array_low_0 = np.transpose(v_array_low_0,(0,3,1,2))
    v_array_high_0 = np.transpose(v_array_high_0,(0,3,1,2)) # order: m, mu, tau, z
    
    # Apply the changes to money and belief variance for both levels of research
    v_array_low_1 = (tau_alpha_low_bot*(m_alpha_low_bot*v_array_low_0[m_idx_low_bot,mu_idx_P1,tau_idx_low_bot,z_idx_P1] + m_alpha_low_top*v_array_low_0[m_idx_low_top,mu_idx_P1,tau_idx_low_bot,z_idx_P1])
                + tau_alpha_low_top*(m_alpha_low_bot*v_array_low_0[m_idx_low_bot,mu_idx_P1,tau_idx_low_top,z_idx_P1] + m_alpha_low_top*v_array_low_0[m_idx_low_top,mu_idx_P1,tau_idx_low_top,z_idx_P1]))
    v_array_high_1 = (tau_alpha_high_bot*(m_alpha_high_bot*v_array_high_0[m_idx_high_bot,mu_idx_P1,tau_idx_high_bot,z_idx_P1] + m_alpha_high_top*v_array_high_0[m_idx_high_top,mu_idx_P1,tau_idx_high_bot,z_idx_P1])
                + tau_alpha_high_top*(m_alpha_high_bot*v_array_high_0[m_idx_high_bot,mu_idx_P1,tau_idx_high_top,z_idx_P1] + m_alpha_high_top*v_array_high_0[m_idx_high_top,mu_idx_P1,tau_idx_high_top,z_idx_P1]))
    
    # Adjust the value arrays so they aren't ridiculous (and impose liquidity constraint)
    v_array_low = v_array_low_1 - v_array_zero
    v_array_high = v_array_high_1 - v_array_zero
    v_array_low[unaffordable_low] = -np.inf
    v_array_high[unaffordable_high] = -np.inf
            
    # Calculate expected value across the three research levels
    v_array = sigma_research*np.log(np.ones_like(v_array_zero) + np.exp(v_array_low/sigma_research) + np.exp(v_array_high/sigma_research)) + v_array_zero
    
    # Return the value function as the solution
    solution_t = BiotechSolution(v_array = v_array)
    if post_process:
        #print('Did post-process for Phase I')
        solution_t.v_low_research = v_array_low_1
        solution_t.v_high_research = v_array_high_1
    return solution_t
    
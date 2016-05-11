'''
This package contains the estimations for cstwMPC.
'''

# Import the HARK library.  The assumption is that this code is in a folder
# contained in the HARK folder. Also import ConsumptionSavingModel
import sys 
sys.path.insert(0,'../')
sys.path.insert(0,'../ConsumptionSavingModel')

import numpy as np
from copy import copy, deepcopy
from time import time
from HARKutilities import approxMeanOneLognormal, combineIndepDstns, approxUniform, calcWeightedAvg, getPercentiles, getLorenzShares, calcSubpopAvg
from HARKsimulation import drawDiscrete, drawMeanOneLognormal
from HARKcore import AgentType
from HARKparallel import multiThreadCommandsFake, multiThreadCommands
import SetupParamsCSTW as Params
import ConsumptionSavingModel as Model
from scipy.optimize import golden, brentq
import matplotlib.pyplot as plt
import csv


# =================================================================
# ====== Make an extension of the basic ConsumerType ==============
# =================================================================

class cstwMPCagent(Model.ConsumerType):
    '''
    A consumer type in the cstwMPC model; a slight modification of base ConsumerType.
    '''
    def __init__(self,time_flow=True,**kwds):
        # Initialize a basic AgentType
        AgentType.__init__(self,solution_terminal=deepcopy(Model.ConsumerType.solution_terminal_),time_flow=time_flow,pseudo_terminal=False,**kwds)

        # Add consumer-type specific objects, copying to create independent versions
        self.time_vary = deepcopy(Model.ConsumerType.time_vary_)
        self.time_inv = deepcopy(Model.ConsumerType.time_inv_)
        self.time_vary.remove('DiscFac')
        self.time_inv.append('DiscFac')
        self.solveOnePeriod = Model.consumptionSavingSolverENDG # this can be swapped for consumptionSavingSolverEXOG or another solver
        self.update()
        
    def simulateCSTWa(self):
        self.W_history = self.Y_history*self.simulate(self.w0,0,self.sim_periods,which=['w'])
        
    def simulateCSTWb(self):
        kappa_quarterly = self.simulate(self.w0,0,self.sim_periods,which=['kappa'])
        self.kappa_history = 1.0 - (1.0 - kappa_quarterly)**4
        
    def simulateCSTWc(self):
        self.m_history = self.simulate(self.w0,0,self.sim_periods,which=['m'])
        
    def update(self):
        '''
        Update the income process and the assets grid.
        '''
        orig_flow = self.time_flow        
        if self.cycles == 0: # hacky fix for labor supply l_bar
            self.updateIncomeProcessAlt()
        else:
            self.updateIncomeProcess()
        self.updateAssetsGrid()
        self.updateSolutionTerminal()
        self.timeFwd()
        if self.cycles > 0:
            self.IncomeDstn = Model.applyFlatIncomeTax(self.IncomeDstn,
                                                 tax_rate=self.tax_rate,
                                                 T_retire=self.T_retire,
                                                 unemployed_indices=range(0,(self.xi_N+1)*self.psi_N,self.xi_N+1))
            scriptR_shocks, xi_shocks = Model.generateIncomeShockHistoryLognormalUnemployment(self)    
            self.addIncomeShockPaths(scriptR_shocks,xi_shocks)   
        else:
            scriptR_shocks, xi_shocks = Model.generateIncomeShockHistoryInfiniteSimple(self)
            self.addIncomeShockPaths(scriptR_shocks,xi_shocks)
        if not orig_flow:
            self.timeRev()
            
    def updateIncomeProcessAlt(self):
        tax_rate = (self.IncUnemp*self.UnempPrb)/(self.l_bar*(1.0-self.UnempPrb))
        xi_dist = deepcopy(approxMeanOneLognormal(self.TranShkCount,self.TranShkStd[0]))
        xi_dist[0] = np.insert(xi_dist[0]*(1.0-self.UnempPrb),0,self.UnempPrb)
        xi_dist[1] = np.insert(self.l_bar*xi_dist[1]*(1.0-tax_rate),0,self.IncUnemp)
        psi_dist = approxMeanOneLognormal(self.PermShkCount,self.PermShkStd[0])
        self.IncomeDstn = [combineIndepDstns(psi_dist,xi_dist)]
        if not 'IncomeDstn' in self.time_vary:
            self.time_vary.append('IncomeDstn')
            

    
def assignBetaDistribution(type_list,DiscFac_list):
    '''
    Assigns the discount factors in DiscFac_list to the types in type_list.  If
    there is heterogeneity beyond the discount factor, then the same DiscFac is
    assigned to consecutive types.
    '''
    DiscFac_N = len(DiscFac_list)
    type_N = len(type_list)/DiscFac_N
    j = 0
    b = 0
    while j < len(type_list):
        t = 0
        while t < type_N:
            type_list[j](DiscFac = DiscFac_list[b])
            t += 1
            j += 1
        b += 1           
            
            
# =================================================================
# ====== Make some data analysis and reporting tools ==============
# =================================================================
    
def calculateKYratioDifference(sim_wealth,weights,total_output,target_KY):
    '''
    Calculates the absolute distance between the simulated capital-to-output
    ratio and the true U.S. level.
    
    Parameters:
    -------------
    sim_wealth : numpy.array
        Array with simulated wealth values.
    weights : numpy.array
        List of weights for each row of sim_wealth.
    total_output : float
        Denominator for the simulated K/Y ratio.
    target_KY : float
        Actual U.S. K/Y ratio to match.
        
    Returns:
    ------------
    distance : float
        Absolute distance between simulated and actual K/Y ratios.
    '''
    sim_K = calcWeightedAvg(sim_wealth,weights)
    sim_KY = sim_K/total_output
    distance = (sim_KY - target_KY)**1.0
    return distance
        

def calculateLorenzDifference(sim_wealth,weights,percentiles,target_levels):
    '''
    Calculates the sum of squared differences between the simulatedLorenz curve
    at the specified percentile levels and the target Lorenz levels.
    
    Parameters:
    -------------
    sim_wealth : numpy.array
        Array with simulated wealth values.
    weights : numpy.array
        List of weights for each row of sim_wealth.
    percentiles : [float]
        Points in the distribution of wealth to match.
    target_levels : np.array
        Actual U.S. Lorenz curve levels at the specified percentiles.
        
    Returns:
    -----------
    distance : float
        Sum of squared distances between simulated and target Lorenz curves.
    '''
    sim_lorenz = getLorenzShares(sim_wealth,weights=weights,percentiles=percentiles)
    distance = sum((100*sim_lorenz-100*target_levels)**2)
    return distance


# Define the main simulation process for matching the K/Y ratio
def simulateKYratioDifference(DiscFac,nabla,N,type_list,weights,total_output,target):
    '''
    Assigns a uniform distribution over DiscFac with width 2*nabla and N points, then
    solves and simulates all agent types in type_list and compares the simuated
    K/Y ratio to the target K/Y ratio.
    '''
    if type(DiscFac) in (list,np.ndarray,np.array):
        DiscFac = DiscFac[0]
    DiscFac_list = approxUniform(DiscFac,nabla,N)
    assignBetaDistribution(type_list,DiscFac_list)
    multiThreadCommandsFake(type_list,beta_point_commands)
    my_diff = calculateKYratioDifference(np.vstack((this_type.W_history for this_type in type_list)),np.tile(weights/float(N),N),total_output,target)
    #print('Tried DiscFac=' + str(DiscFac) + ', nabla=' + str(nabla) + ', got diff=' + str(my_diff))
    return my_diff


mystr = lambda number : "{:.3f}".format(number) 
def makeCSTWresults(DiscFac,nabla,save_name=None):
    '''
    Produces a variety of results for the cstwMPC paper (usually after estimating).
    '''
    DiscFac_list = approxUniform(DiscFac,nabla,N=Params.pref_type_count)
    assignBetaDistribution(est_type_list,DiscFac_list)
    multiThreadCommandsFake(est_type_list,results_commands)
    
    lorenz_distance = np.sqrt(betaDistObjective(nabla))
    #lorenz_distance = 0.0
    
    if Params.do_lifecycle: # This can probably be removed
        sim_length = Params.total_T
    else:
        sim_length = Params.sim_periods
    sim_wealth = (np.vstack((this_type.W_history for this_type in est_type_list))).flatten()
    sim_wealth_short = (np.vstack((this_type.W_history[0:sim_length] for this_type in est_type_list))).flatten()
    sim_kappa = (np.vstack((this_type.kappa_history for this_type in est_type_list))).flatten()
    sim_income = (np.vstack((this_type.Y_history[0:sim_length]*np.asarray(this_type.TranShks[0:sim_length]) for this_type in est_type_list))).flatten()
    sim_ratio = (np.vstack((this_type.W_history[0:sim_length]/this_type.Y_history[0:sim_length] for this_type in est_type_list))).flatten()
    if Params.do_lifecycle:
        sim_unemp = (np.vstack((np.vstack((this_type.IncUnemp == np.asarray(this_type.TranShks[0:Params.working_T]),np.zeros((Params.retired_T,Params.sim_pop_size),dtype=bool))) for this_type in est_type_list))).flatten()
        sim_emp = (np.vstack((np.vstack((this_type.IncUnemp != np.asarray(this_type.TranShks[0:Params.working_T]),np.zeros((Params.retired_T,Params.sim_pop_size),dtype=bool))) for this_type in est_type_list))).flatten()
        sim_ret = (np.vstack((np.vstack((np.zeros((Params.working_T,Params.sim_pop_size),dtype=bool),np.ones((Params.retired_T,Params.sim_pop_size),dtype=bool))) for this_type in est_type_list))).flatten()
    else:
        sim_unemp = np.vstack((this_type.IncUnemp == np.asarray(this_type.TranShks[0:sim_length]) for this_type in est_type_list)).flatten()
        sim_emp = np.vstack((this_type.IncUnemp != np.asarray(this_type.TranShks[0:sim_length]) for this_type in est_type_list)).flatten()
        sim_ret = np.zeros(sim_emp.size,dtype=bool)
    sim_weight_all = np.tile(np.repeat(Params.age_weight_all,Params.sim_pop_size),Params.pref_type_count)
    sim_weight_short = np.tile(np.repeat(Params.age_weight_short,Params.sim_pop_size),Params.pref_type_count)
    
    if Params.do_beta_dist and Params.do_lifecycle:
        kappa_mean_by_age_type = (np.mean(np.vstack((this_type.kappa_history for this_type in est_type_list)),axis=1)).reshape((Params.pref_type_count*3,DropoutType.T_total))
        kappa_mean_by_age_pref = np.zeros((Params.pref_type_count,DropoutType.T_total)) + np.nan
        for j in range(Params.pref_type_count):
            kappa_mean_by_age_pref[j,] = Params.d_pct*kappa_mean_by_age_type[3*j+0,] + Params.h_pct*kappa_mean_by_age_type[3*j+1,] + Params.c_pct*kappa_mean_by_age_type[3*j+2,] 
        kappa_mean_by_age = np.mean(kappa_mean_by_age_pref,axis=0)
        kappa_lo_beta_by_age = kappa_mean_by_age_pref[0,]
        kappa_hi_beta_by_age = kappa_mean_by_age_pref[Params.pref_type_count-1,]
    
    lorenz_fig_data = makeLorenzFig(Params.SCF_wealth,Params.SCF_weights,sim_wealth,sim_weight_all)
    mpc_fig_data = makeMPCfig(sim_kappa,sim_weight_short)
    
    kappa_all = calcWeightedAvg(np.vstack((this_type.kappa_history for this_type in est_type_list)),np.tile(Params.age_weight_short/float(Params.pref_type_count),Params.pref_type_count))
    kappa_unemp = np.sum(sim_kappa[sim_unemp]*sim_weight_short[sim_unemp])/np.sum(sim_weight_short[sim_unemp])
    kappa_emp = np.sum(sim_kappa[sim_emp]*sim_weight_short[sim_emp])/np.sum(sim_weight_short[sim_emp])
    kappa_ret = np.sum(sim_kappa[sim_ret]*sim_weight_short[sim_ret])/np.sum(sim_weight_short[sim_ret])
    
    my_cutoffs = [(0.99,1),(0.9,1),(0.8,1),(0.6,0.8),(0.4,0.6),(0.2,0.4),(0.0,0.2)]
    kappa_by_ratio_groups = calcSubpopAvg(sim_kappa,sim_ratio,my_cutoffs,sim_weight_short)
    kappa_by_income_groups = calcSubpopAvg(sim_kappa,sim_income,my_cutoffs,sim_weight_short)
    
    quintile_points = getPercentiles(sim_wealth_short,weights=sim_weight_short,percentiles=[0.2, 0.4, 0.6, 0.8])
    wealth_quintiles = np.ones(sim_wealth_short.size,dtype=int)
    wealth_quintiles[sim_wealth_short > quintile_points[0]] = 2
    wealth_quintiles[sim_wealth_short > quintile_points[1]] = 3
    wealth_quintiles[sim_wealth_short > quintile_points[2]] = 4
    wealth_quintiles[sim_wealth_short > quintile_points[3]] = 5
    MPC_cutoff = getPercentiles(sim_kappa,weights=sim_weight_short,percentiles=[2.0/3.0])
    these_quintiles = wealth_quintiles[sim_kappa > MPC_cutoff]
    these_weights = sim_weight_short[sim_kappa > MPC_cutoff]
    hand_to_mouth_total = np.sum(these_weights)
    hand_to_mouth_pct = []
    for q in range(5):
        hand_to_mouth_pct.append(np.sum(these_weights[these_quintiles == (q+1)])/hand_to_mouth_total)
    
    results_string = 'Estimate is DiscFac=' + str(DiscFac) + ', nabla=' + str(nabla) + '\n'
    results_string += 'Lorenz distance is ' + str(lorenz_distance) + '\n'
    results_string += 'Average MPC for all consumers is ' + mystr(kappa_all) + '\n'
    results_string += 'Average MPC in the top percentile of W/Y is ' + mystr(kappa_by_ratio_groups[0]) + '\n'
    results_string += 'Average MPC in the top decile of W/Y is ' + mystr(kappa_by_ratio_groups[1]) + '\n'
    results_string += 'Average MPC in the top quintile of W/Y is ' + mystr(kappa_by_ratio_groups[2]) + '\n'
    results_string += 'Average MPC in the second quintile of W/Y is ' + mystr(kappa_by_ratio_groups[3]) + '\n'
    results_string += 'Average MPC in the middle quintile of W/Y is ' + mystr(kappa_by_ratio_groups[4]) + '\n'
    results_string += 'Average MPC in the fourth quintile of W/Y is ' + mystr(kappa_by_ratio_groups[5]) + '\n'
    results_string += 'Average MPC in the bottom quintile of W/Y is ' + mystr(kappa_by_ratio_groups[6]) + '\n'
    results_string += 'Average MPC in the top percentile of y is ' + mystr(kappa_by_income_groups[0]) + '\n'
    results_string += 'Average MPC in the top decile of y is ' + mystr(kappa_by_income_groups[1]) + '\n'
    results_string += 'Average MPC in the top quintile of y is ' + mystr(kappa_by_income_groups[2]) + '\n'
    results_string += 'Average MPC in the second quintile of y is ' + mystr(kappa_by_income_groups[3]) + '\n'
    results_string += 'Average MPC in the middle quintile of y is ' + mystr(kappa_by_income_groups[4]) + '\n'
    results_string += 'Average MPC in the fourth quintile of y is ' + mystr(kappa_by_income_groups[5]) + '\n'
    results_string += 'Average MPC in the bottom quintile of y is ' + mystr(kappa_by_income_groups[6]) + '\n'
    results_string += 'Average MPC for the employed is ' + mystr(kappa_emp) + '\n'
    results_string += 'Average MPC for the unemployed is ' + mystr(kappa_unemp) + '\n'
    results_string += 'Average MPC for the retired is ' + mystr(kappa_ret) + '\n'
    results_string += 'Of the population with the 1/3 highest MPCs...' + '\n'
    results_string += mystr(hand_to_mouth_pct[0]*100) + '% are in the bottom wealth quintile,' + '\n'
    results_string += mystr(hand_to_mouth_pct[1]*100) + '% are in the second wealth quintile,' + '\n'
    results_string += mystr(hand_to_mouth_pct[2]*100) + '% are in the third wealth quintile,' + '\n'
    results_string += mystr(hand_to_mouth_pct[3]*100) + '% are in the fourth wealth quintile,' + '\n'
    results_string += 'and ' + mystr(hand_to_mouth_pct[4]*100) + '% are in the top wealth quintile.' + '\n'
    print(results_string)
    
    if save_name is not None:
        with open('./Results/' + save_name + 'LorenzFig.txt','w') as f:
            my_writer = csv.writer(f, delimiter='\t',)
            for j in range(len(lorenz_fig_data[0])):
                my_writer.writerow([lorenz_fig_data[0][j], lorenz_fig_data[1][j], lorenz_fig_data[2][j]])
            f.close()
        with open('./Results/' + save_name + 'MPCfig.txt','w') as f:
            my_writer = csv.writer(f, delimiter='\t')
            for j in range(len(mpc_fig_data[0])):
                my_writer.writerow([lorenz_fig_data[0][j], mpc_fig_data[1][j]])
            f.close()
        if Params.do_beta_dist and Params.do_lifecycle:
            with open('./Results/' + save_name + 'KappaByAge.txt','w') as f:
                my_writer = csv.writer(f, delimiter='\t')
                for j in range(len(kappa_mean_by_age)):
                    my_writer.writerow([kappa_mean_by_age[j], kappa_lo_beta_by_age[j], kappa_hi_beta_by_age[j]])
                f.close()
        with open('./Results/' + save_name + 'Results.txt','w') as f:
            f.write(results_string)
            f.close()


def makeLorenzFig(real_wealth,real_weights,sim_wealth,sim_weights):
    '''
    Produces a Lorenz curve for the distribution of wealth, comparing simulated
    to actual data.  A sub-function of makeCSTWresults().
    '''
    these_percents = np.linspace(0.0001,0.9999,201)
    real_lorenz = getLorenzShares(real_wealth,weights=real_weights,percentiles=these_percents)
    sim_lorenz = getLorenzShares(sim_wealth,weights=sim_weights,percentiles=these_percents)
    plt.plot(100*these_percents,real_lorenz,'-k',linewidth=1.5)
    plt.plot(100*these_percents,sim_lorenz,'--k',linewidth=1.5)
    plt.xlabel('Wealth percentile',fontsize=14)
    plt.ylabel('Cumulative wealth ownership',fontsize=14)
    plt.title('Simulated vs Actual Lorenz Curves',fontsize=16)
    plt.legend(('Actual','Simulated'),loc=2,fontsize=12)
    plt.ylim(-0.01,1)
    plt.show()
    return (these_percents,real_lorenz,sim_lorenz)
    
    
def makeMPCfig(kappa,weights):
    '''
    Plot the CDF of the marginal propensity to consume. A sub-function of makeCSTWresults().
    '''
    these_percents = np.linspace(0.0001,0.9999,201)
    kappa_percentiles = getPercentiles(kappa,weights,percentiles=these_percents)
    plt.plot(kappa_percentiles,these_percents,'-k',linewidth=1.5)
    plt.xlabel('Marginal propensity to consume',fontsize=14)
    plt.ylabel('Cumulative probability',fontsize=14)
    plt.title('CDF of the MPC',fontsize=16)
    plt.show()
    return (these_percents,kappa_percentiles)


def calcKappaMean(DiscFac,nabla):
    '''
    Calculates the average MPC for the given parameters.  This is a very small
    sub-function of makeCSTWresults().
    '''
    DiscFac_list = approxUniform(DiscFac,nabla,N=Params.pref_type_count)
    assignBetaDistribution(est_type_list,DiscFac_list)
    multiThreadCommandsFake(est_type_list,results_commands)
    
    kappa_all = approxUniform(np.vstack((this_type.kappa_history for this_type in est_type_list)),np.tile(Params.age_weight_short/float(Params.pref_type_count),Params.pref_type_count))
    return kappa_all
   
   

# Only run below this line if module is run rather than imported:
if __name__ == "__main__":
    # =================================================================
    # ====== Make the list of consumer types for estimation ===========
    #==================================================================
    
    # Set target Lorenz points and K/Y ratio (MOVE THIS TO SetupParams)
    if Params.do_liquid:
        lorenz_target = np.array([0.0, 0.004, 0.025,0.117])
        KY_target = 6.60
    else: # This is hacky until I can find the liquid wealth data and import it
        lorenz_target = getLorenzShares(Params.SCF_wealth,weights=Params.SCF_weights,percentiles=Params.percentiles_to_match)
        #lorenz_target = np.array([-0.002, 0.01, 0.053,0.171])
        KY_target = 10.26
    
    
    # Make a vector of initial wealth-to-permanent income ratios
    w0_vector = drawDiscrete(P=Params.w0_probs,
                                             X=Params.w0_values,
                                             N=Params.sim_pop_size,
                                             seed=Params.w0_seed)
                                             
    # Make the list of types for this run, whether infinite or lifecycle
    if Params.do_lifecycle:
        # Make base consumer types for each education level
        DropoutType = cstwMPCagent(**Params.init_dropout)
        DropoutType.w0 = w0_vector
        HighschoolType = deepcopy(DropoutType)
        HighschoolType(**Params.adj_highschool)
        CollegeType = deepcopy(DropoutType)
        CollegeType(**Params.adj_college)
        DropoutType.update()
        HighschoolType.update()
        CollegeType.update()
        
        # Make histories of permanent income levels for each education type
        Y0_vector_base = drawMeanOneLognormal(Params.Y0_sigma, Params.sim_pop_size, Params.Y0_seed)
        psi_gamma_history_d = np.zeros((Params.total_T,Params.sim_pop_size)) + np.nan
        psi_gamma_history_h = deepcopy(psi_gamma_history_d)
        psi_gamma_history_c = deepcopy(psi_gamma_history_d)
        for t in range(Params.total_T):
            psi_gamma_history_d[t,] = (Params.econ_growth*DropoutType.PermShks[t]/Params.Rfree)**(-1)
            psi_gamma_history_h[t,] = (Params.econ_growth*HighschoolType.PermShks[t]/Params.Rfree)**(-1)
            psi_gamma_history_c[t,] = (Params.econ_growth*CollegeType.PermShks[t]/Params.Rfree)**(-1)
        Y_history_d = np.cumprod(np.vstack((Params.Y0_d*Y0_vector_base,psi_gamma_history_d)),axis=0)
        Y_history_h = np.cumprod(np.vstack((Params.Y0_h*Y0_vector_base,psi_gamma_history_h)),axis=0)
        Y_history_c = np.cumprod(np.vstack((Params.Y0_c*Y0_vector_base,psi_gamma_history_c)),axis=0)
        DropoutType.Y_history = Y_history_d
        HighschoolType.Y_history = Y_history_h
        CollegeType.Y_history = Y_history_c
        
        # Set the type list for the lifecycle estimation
        short_type_list = [DropoutType, HighschoolType, CollegeType]
        spec_add = 'LC'       
    
    else:
        # Make the base infinite horizon type and assign income shocks
        InfiniteType = cstwMPCagent(**Params.init_infinite)
        InfiniteType.tolerance = 0.0001
        InfiniteType.w0 = w0_vector*0.0
        
        # Make histories of permanent income levels for the infinite horizon type
        Y0_vector_base = np.ones(Params.sim_pop_size,dtype=float)
        psi_gamma_history_i = np.zeros((Params.sim_periods,Params.sim_pop_size)) + np.nan
        for t in range(Params.sim_periods):
            psi_gamma_history_i[t,] = (Params.PermGroFac_i[0]*InfiniteType.PermShks[t]/InfiniteType.Rfree)**(-1)
        Y_history_i = np.cumprod(np.vstack((Y0_vector_base,psi_gamma_history_i)),axis=0)
        InfiniteType.Y_history = Y_history_i
        
        # Use a "tractable consumer" instead if desired
        if Params.do_tractable:
            from TractableBufferStock import TractableConsumerType
            TractableInfType = TractableConsumerType(DiscFac=InfiniteType.DiscFac,
                                                     mho=1-InfiniteType.survival_prob[0],
                                                     R=InfiniteType.Rfree,
                                                     G=InfiniteType.PermGroFac[0],
                                                     rho=InfiniteType.rho,
                                                     sim_periods=InfiniteType.sim_periods,
                                                     IncUnemp=InfiniteType.IncUnemp)
            TractableInfType.timeFwd()
            TractableInfType.Y_history = Y_history_i
            TractableInfType.TranShks = InfiniteType.TranShks
            TractableInfType.PermShks = InfiniteType.PermShks
            TractableInfType.w0 = InfiniteType.w0
               
        # Set the type list for the infinite horizon estimation
        if Params.do_tractable:
            short_type_list = [TractableInfType]
            spec_add = 'TC'
        else:
            short_type_list = [InfiniteType]
            spec_add = 'IH'
    
    # Expand the estimation type list if doing beta-dist
    if Params.do_beta_dist:
        long_type_list = []
        for j in range(Params.pref_type_count):
            long_type_list += deepcopy(short_type_list)
        est_type_list = long_type_list
    else:
        est_type_list = short_type_list
    
    if Params.do_liquid:
        wealth_measure = 'Liquid'
    else:
        wealth_measure = 'NetWorth'
    
    
    # =================================================================
    # ====== Define estimation objectives =============================
    #==================================================================
    
    # Set commands for the beta-point estimation
    beta_point_commands = ['solve()','unpack_cFunc()','timeFwd()','simulateCSTWa()']
    results_commands = ['solve()','unpack_cFunc()','timeFwd()','simulateCSTWa()','simulateCSTWb()']
        
    # Make the objective function for the beta-point estimation
    betaPointObjective = lambda beta : simulateKYratioDifference(beta,
                                                                 nabla=0,
                                                                 N=1,
                                                                 type_list=est_type_list,
                                                                 weights=Params.age_weight_all,
                                                                 total_output=Params.total_output,
                                                                 target=KY_target)
                                                                 
    # Make the objective function for the beta-dist estimation
    def betaDistObjective(nabla):
        # Make the "intermediate objective function" for the beta-dist estimation
        #print('Trying nabla=' + str(nabla))
        intermediateObjective = lambda DiscFac : simulateKYratioDifference(DiscFac,
                                                                 nabla=nabla,
                                                                 N=Params.pref_type_count,
                                                                 type_list=est_type_list,
                                                                 weights=Params.age_weight_all,
                                                                 total_output=Params.total_output,
                                                                 target=KY_target)
        #DiscFac_new = newton(intermediateObjective,Params.DiscFac_guess,maxiter=100)
        DiscFac_new = brentq(intermediateObjective,0.90,0.998,xtol=10**(-8))
        N=Params.pref_type_count
        wealth_sim = (np.vstack((this_type.W_history for this_type in est_type_list))).flatten()
        sim_weights = np.tile(np.repeat(Params.age_weight_all,Params.sim_pop_size),N)
        my_diff = calculateLorenzDifference(wealth_sim,sim_weights,Params.percentiles_to_match,lorenz_target)
        print('DiscFac=' + str(DiscFac_new) + ', nabla=' + str(nabla) + ', diff=' + str(my_diff))
        if my_diff < Params.diff_save:
            Params.DiscFac_save = DiscFac_new
        return my_diff
    
    
    
    # =================================================================
    # ========= Estimating the model ==================================
    #==================================================================
    
    if Params.run_estimation:
        # Estimate the model and time it
        t_start = time()
        if Params.do_beta_dist:
            bracket = (0,0.015) # large nablas break IH version
            nabla = golden(betaDistObjective,brack=bracket,tol=10**(-4))        
            DiscFac = Params.DiscFac_save
            spec_name = spec_add + 'betaDist' + wealth_measure
        else:
            nabla = 0
            if Params.do_tractable:
                top = 0.991
            else:
                top = 1.0
            DiscFac = brentq(betaPointObjective,0.90,top,xtol=10**(-8))
            spec_name = spec_add + 'betaPoint' + wealth_measure
        t_end = time()
        print('Estimate is DiscFac=' + str(DiscFac) + ', nabla=' + str(nabla) + ', took ' + str(t_end-t_start) + ' seconds.')
        #spec_name=None
        makeCSTWresults(DiscFac,nabla,spec_name)
    
    
    
    # =================================================================
    # ========= Relationship between DiscFac and K/Y ratio ===============
    #==================================================================
    
    if Params.find_beta_vs_KY:
        t_start = time()
        DiscFac_list = np.linspace(0.95,1.01,201)
        KY_ratio_list = []
        for DiscFac in DiscFac_list:
            KY_ratio_list.append(betaPointObjective(DiscFac) + KY_target)
        KY_ratio_list = np.array(KY_ratio_list)
        t_end = time()
        plt.plot(DiscFac_list,KY_ratio_list,'-k',linewidth=1.5)
        plt.xlabel(r'Discount factor $\beta$',fontsize=14)
        plt.ylabel('Capital to output ratio',fontsize=14)
        print('That took ' + str(t_end-t_start) + ' seconds.')
        plt.show()
        with open('./Results/' + spec_add + '_KYbyBeta' +  '.txt','w') as f:
                my_writer = csv.writer(f, delimiter='\t',)
                for j in range(len(DiscFac_list)):
                    my_writer.writerow([DiscFac_list[j], KY_ratio_list[j]])
                f.close()
    
    
    
    # =================================================================
    # ========= Sensitivity analysis ==================================
    #==================================================================
    
    # Sensitivity analysis only set up for infinite horizon model!
    if Params.do_lifecycle:
        bracket = (0,0.015)
    else:
        bracket = (0,0.015) # large nablas break IH version
    spec_name = None
    
    if Params.do_sensitivity[0]: # coefficient of relative risk aversion sensitivity analysis
        rho_list = np.linspace(0.5,4.0,15).tolist() #15
        fit_list = []
        DiscFac_list = []
        nabla_list = []
        kappa_list = []
        for rho in rho_list:
            print('Now estimating model with rho = ' + str(rho))
            Params.diff_save = 1000000.0
            for this_type in est_type_list:
                this_type(rho = rho)
                this_type.update()
            output = golden(betaDistObjective,brack=bracket,tol=10**(-4),full_output=True)
            nabla = output[0]
            fit = output[1]
            DiscFac = Params.DiscFac_save
            kappa = calcKappaMean(DiscFac,nabla)
            DiscFac_list.append(DiscFac)
            nabla_list.append(nabla)
            fit_list.append(fit)
            kappa_list.append(kappa)
        with open('./Results/SensitivityRho.txt','w') as f:
            my_writer = csv.writer(f, delimiter='\t',)
            for j in range(len(DiscFac_list)):
                my_writer.writerow([rho_list[j], kappa_list[j], DiscFac_list[j], nabla_list[j], fit_list[j]])
            f.close()
        for this_type in est_type_list:
            this_type(rho = Params.rho)
    
    if Params.do_sensitivity[1]: # transitory income stdev sensitivity analysis
        xi_sigma_list = [0.01] + np.linspace(0.05,0.8,16).tolist() #16
        fit_list = []
        DiscFac_list = []
        nabla_list = []
        kappa_list = []
        for xi_sigma in xi_sigma_list:
            print('Now estimating model with xi_sigma = ' + str(xi_sigma))
            Params.diff_save = 1000000.0
            for this_type in est_type_list:
                this_type(xi_sigma = [xi_sigma])
                this_type.update()
            output = golden(betaDistObjective,brack=bracket,tol=10**(-4),full_output=True)
            nabla = output[0]
            fit = output[1]
            DiscFac = Params.DiscFac_save
            kappa = calcKappaMean(DiscFac,nabla)
            DiscFac_list.append(DiscFac)
            nabla_list.append(nabla)
            fit_list.append(fit)
            kappa_list.append(kappa)
        with open('./Results/SensitivityXiSigma.txt','w') as f:
            my_writer = csv.writer(f, delimiter='\t',)
            for j in range(len(DiscFac_list)):
                my_writer.writerow([xi_sigma_list[j], kappa_list[j], DiscFac_list[j], nabla_list[j], fit_list[j]])
            f.close()
        for this_type in est_type_list:
            this_type(xi_sigma = Params.xi_sigma_i)
            this_type.update()
            
    if Params.do_sensitivity[2]: # permanent income stdev sensitivity analysis
        psi_sigma_list = np.linspace(0.02,0.18,17).tolist() #17
        fit_list = []
        DiscFac_list = []
        nabla_list = []
        kappa_list = []
        for psi_sigma in psi_sigma_list:
            print('Now estimating model with psi_sigma = ' + str(psi_sigma))
            Params.diff_save = 1000000.0
            for this_type in est_type_list:
                this_type(psi_sigma = [psi_sigma])
                this_type.timeRev()
                this_type.update()
            psi_gamma_history_i = np.zeros((Params.sim_periods,Params.sim_pop_size)) + np.nan
            for t in range(Params.sim_periods):
                psi_gamma_history_i[t,] = (Params.PermGroFac_i[0]*est_type_list[0].PermShks[Params.sim_periods-t-1]/InfiniteType.Rfree)**(-1)
            Y_history_i = np.cumprod(np.vstack((Y0_vector_base,psi_gamma_history_i)),axis=0)
            for this_type in est_type_list:
                this_type.Y_history = Y_history_i
            output = golden(betaDistObjective,brack=bracket,tol=10**(-4),full_output=True)
            nabla = output[0]
            fit = output[1]
            DiscFac = Params.DiscFac_save
            kappa = calcKappaMean(DiscFac,nabla)
            DiscFac_list.append(DiscFac)
            nabla_list.append(nabla)
            fit_list.append(fit)
            kappa_list.append(kappa)
        with open('./Results/SensitivityPsiSigma.txt','w') as f:
            my_writer = csv.writer(f, delimiter='\t',)
            for j in range(len(DiscFac_list)):
                my_writer.writerow([psi_sigma_list[j], kappa_list[j], DiscFac_list[j], nabla_list[j], fit_list[j]])
            f.close()
        for this_type in est_type_list:
            this_type(psi_sigma = Params.psi_sigma_i)
            this_type.update()
        psi_gamma_history_i = np.zeros((Params.sim_periods,Params.sim_pop_size)) + np.nan
        for t in range(Params.sim_periods):
            psi_gamma_history_i[t,] = (Params.PermGroFac_i[0]*est_type_list[0].PermShks[Params.sim_periods-t-1]/InfiniteType.Rfree)**(-1)
        Y_history_i = np.cumprod(np.vstack((Y0_vector_base,psi_gamma_history_i)),axis=0)
        for this_type in est_type_list:
            this_type.Y_history = Y_history_i
            
    if Params.do_sensitivity[3]: # unemployment benefits sensitivity analysis
        mu_list = np.linspace(0.0,0.8,17).tolist() #17
        fit_list = []
        DiscFac_list = []
        nabla_list = []
        kappa_list = []
        for mu in mu_list:
            print('Now estimating model with mu = ' + str(mu))
            Params.diff_save = 1000000.0
            for this_type in est_type_list:
                this_type(IncUnemp = mu)
                this_type.timeRev()
                this_type.update()
            output = golden(betaDistObjective,brack=bracket,tol=10**(-4),full_output=True)
            nabla = output[0]
            fit = output[1]
            DiscFac = Params.DiscFac_save
            kappa = calcKappaMean(DiscFac,nabla)
            DiscFac_list.append(DiscFac)
            nabla_list.append(nabla)
            fit_list.append(fit)
            kappa_list.append(kappa)
        with open('./Results/SensitivityMu.txt','w') as f:
            my_writer = csv.writer(f, delimiter='\t',)
            for j in range(len(DiscFac_list)):
                my_writer.writerow([mu_list[j], kappa_list[j], DiscFac_list[j], nabla_list[j], fit_list[j]])
            f.close()
        for this_type in est_type_list:
            this_type(IncUnemp = Params.IncUnemp)
            this_type.update()
    
    if Params.do_sensitivity[4]: # unemployment rate sensitivity analysis
        urate_list = np.linspace(0.02,0.12,16).tolist() #16
        fit_list = []
        DiscFac_list = []
        nabla_list = []
        kappa_list = []
        for urate in urate_list:
            print('Now estimating model with urate = ' + str(urate))
            Params.diff_save = 1000000.0
            for this_type in est_type_list:
                this_type(p_unemploy = urate)
                this_type.update()
            output = golden(betaDistObjective,brack=bracket,tol=10**(-4),full_output=True)
            nabla = output[0]
            fit = output[1]
            DiscFac = Params.DiscFac_save
            kappa = calcKappaMean(DiscFac,nabla)
            DiscFac_list.append(DiscFac)
            nabla_list.append(nabla)
            fit_list.append(fit)
            kappa_list.append(kappa)
        with open('./Results/SensitivityUrate.txt','w') as f:
            my_writer = csv.writer(f, delimiter='\t',)
            for j in range(len(DiscFac_list)):
                my_writer.writerow([urate_list[j], kappa_list[j], DiscFac_list[j], nabla_list[j], fit_list[j]])
            f.close()
        for this_type in est_type_list:
            this_type(p_unemploy = Params.p_unemploy)
            this_type.update()
            
    if Params.do_sensitivity[5]: # mortality rate sensitivity analysis
        death_prob_list = np.linspace(0.003,0.0125,16).tolist() #16
        fit_list = []
        DiscFac_list = []
        nabla_list = []
        kappa_list = []
        for death_prob in death_prob_list:
            print('Now estimating model with death_prob = ' + str(death_prob))
            Params.diff_save = 1000000.0
            for this_type in est_type_list:
                this_type(survival_prob = [1 - death_prob])
            output = golden(betaDistObjective,brack=bracket,tol=10**(-4),full_output=True)
            nabla = output[0]
            fit = output[1]
            DiscFac = Params.DiscFac_save
            kappa = calcKappaMean(DiscFac,nabla)
            DiscFac_list.append(DiscFac)
            nabla_list.append(nabla)
            fit_list.append(fit)
            kappa_list.append(kappa)
        with open('./Results/SensitivityMortality.txt','w') as f:
            my_writer = csv.writer(f, delimiter='\t',)
            for j in range(len(DiscFac_list)):
                my_writer.writerow([death_prob_list[j], kappa_list[j], DiscFac_list[j], nabla_list[j], fit_list[j]])
        for this_type in est_type_list:
            this_type(survival_prob = Params.survival_prob_i)
    
    
    if Params.do_sensitivity[6]: # permanent income growth rate sensitivity analysis
        g_list = np.linspace(0.00,0.04,17).tolist() #17
        fit_list = []
        DiscFac_list = []
        nabla_list = []
        kappa_list = []
        for g in g_list:
            print('Now estimating model with g = ' + str(g))
            Params.diff_save = 1000000.0
            Params.PermGroFac_i = [(1 + g)**0.25]
            for this_type in est_type_list:
                this_type(PermGroFac = Params.PermGroFac_i)
                this_type.timeRev()
                this_type.update()
            psi_gamma_history_i = np.zeros((Params.sim_periods,Params.sim_pop_size)) + np.nan
            for t in range(Params.sim_periods):
                psi_gamma_history_i[t,] = (Params.PermGroFac_i[0]*est_type_list[0].PermShks[Params.sim_periods-t-1]/InfiniteType.Rfree)**(-1)
            Y_history_i = np.cumprod(np.vstack((Y0_vector_base,psi_gamma_history_i)),axis=0)
            for this_type in est_type_list:
                this_type.Y_history = Y_history_i
            output = golden(betaDistObjective,brack=bracket,tol=10**(-4),full_output=True)
            nabla = output[0]
            fit = output[1]
            DiscFac = Params.DiscFac_save
            kappa = calcKappaMean(DiscFac,nabla)
            DiscFac_list.append(DiscFac)
            nabla_list.append(nabla)
            fit_list.append(fit)
            kappa_list.append(kappa)
        with open('./Results/SensitivityG.txt','w') as f:
            my_writer = csv.writer(f, delimiter='\t',)
            for j in range(len(DiscFac_list)):
                my_writer.writerow([g_list[j], kappa_list[j], DiscFac_list[j], nabla_list[j], fit_list[j]])
            f.close()
        Params.PermGroFac_i = [1.01**0.25]
        for this_type in est_type_list:
            this_type(PermGroFac = Params.PermGroFac_i)
            this_type.update()
        psi_gamma_history_i = np.zeros((Params.sim_periods,Params.sim_pop_size)) + np.nan
        for t in range(Params.sim_periods):
            psi_gamma_history_i[t,] = (Params.PermGroFac_i[0]*est_type_list[0].PermShks[Params.sim_periods-t-1]/InfiniteType.Rfree)**(-1)
        Y_history_i = np.cumprod(np.vstack((Y0_vector_base,psi_gamma_history_i)),axis=0)
        for this_type in est_type_list:
            this_type.Y_history = Y_history_i
            
    if Params.do_sensitivity[7]: # interest rate sensitivity analysis
        R_list = np.linspace(1.0,1.04,17).tolist()
        fit_list = []
        DiscFac_list = []
        nabla_list = []
        kappa_list = []
        for R in R_list:
            print('Now estimating model with R = ' + str(R))
            R_adj = R/InfiniteType.survival_prob[0]
            Params.diff_save = 1000000.0
            for this_type in est_type_list:
                this_type(R = R_adj)
                this_type.timeRev()
                this_type.update()
            psi_gamma_history_i = np.zeros((Params.sim_periods,Params.sim_pop_size)) + np.nan
            for t in range(Params.sim_periods):
                psi_gamma_history_i[t,] = (Params.PermGroFac_i[0]*est_type_list[0].PermShks[Params.sim_periods-t-1]/R_adj)**(-1)
            Y_history_i = np.cumprod(np.vstack((Y0_vector_base,psi_gamma_history_i)),axis=0)
            for this_type in est_type_list:
                this_type.Y_history = Y_history_i
            output = golden(betaDistObjective,brack=bracket,tol=10**(-4),full_output=True)
            nabla = output[0]
            fit = output[1]
            DiscFac = Params.DiscFac_save
            kappa = calcKappaMean(DiscFac,nabla)
            DiscFac_list.append(DiscFac)
            nabla_list.append(nabla)
            fit_list.append(fit)
            kappa_list.append(kappa)
        with open('./Results/SensitivityInterestRate.txt','w') as f:
            my_writer = csv.writer(f, delimiter='\t',)
            for j in range(len(DiscFac_list)):
                my_writer.writerow([R_list[j], kappa_list[j], DiscFac_list[j], nabla_list[j], fit_list[j]])
        for this_type in est_type_list:
            this_type(R = 1.01/Params.survival_prob_i[0])
            this_type.update()
        psi_gamma_history_i = np.zeros((Params.sim_periods,Params.sim_pop_size)) + np.nan
        for t in range(Params.sim_periods):
            psi_gamma_history_i[t,] = (Params.PermGroFac_i[0]*est_type_list[0].PermShks[Params.sim_periods-t-1]/InfiniteType.Rfree)**(-1)
        Y_history_i = np.cumprod(np.vstack((Y0_vector_base,psi_gamma_history_i)),axis=0)
        for this_type in est_type_list:
            this_type.Y_history = Y_history_i

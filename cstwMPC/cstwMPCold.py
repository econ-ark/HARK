'''
Nearly all of the estimations for the paper "The Distribution of Wealth and the
Marginal Propensity to Consume", by Chris Carroll, Jiri Slacalek, Kiichi Tokuoka,
and Matthew White.  The micro model is a very slightly altered version of
ConsIndShockModel; the macro model is ConsAggShockModel.  See SetupParamsCSTW
for parameters and execution options.
'''

# Import the HARK library.  The assumption is that this code is in a folder
# contained in the HARK folder. Also import ConsumptionSavingModel
import sys
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../ConsumptionSaving'))

import numpy as np
from copy import deepcopy
from time import time
from HARKutilities import approxMeanOneLognormal, combineIndepDstns, approxUniform, calcWeightedAvg, \
                          getPercentiles, getLorenzShares, calcSubpopAvg
from HARKsimulation import drawDiscrete, drawMeanOneLognormal
from HARKcore import AgentType
from HARKparallel import multiThreadCommandsFake
import SetupParamsCSTW as Params
import ConsIndShockModel as Model
from ConsAggShockModel import CobbDouglasEconomy, AggShockConsumerType
from scipy.optimize import golden, brentq
import matplotlib.pyplot as plt
import csv

# =================================================================
# ====== Make an extension of the basic ConsumerType ==============
# =================================================================

class cstwMPCagent(Model.IndShockConsumerType):
    '''
    A consumer type in the cstwMPC model; a slight modification of base ConsumerType.
    '''
    def __init__(self,time_flow=True,**kwds):
        '''
        Make a new consumer type for the cstwMPC model.

        Parameters
        ----------
        time_flow : boolean
            Indictator for whether time is "flowing" forward for this agent.
        **kwds : keyword arguments
            Any number of keyword arguments of the form key=value.  Each value
            will be assigned to the attribute named in self.

        Returns
        -------
        new instance of cstwMPCagent
        '''
        # Initialize a basic AgentType
        AgentType.__init__(self,solution_terminal=deepcopy(Model.IndShockConsumerType.solution_terminal_),
                           time_flow=time_flow,pseudo_terminal=False,**kwds)

        # Add consumer-type specific objects, copying to create independent versions
        self.time_vary = deepcopy(Model.IndShockConsumerType.time_vary_)
        self.time_inv = deepcopy(Model.IndShockConsumerType.time_inv_)
        self.solveOnePeriod = Model.solveConsIndShock
        self.update()

    def simulateCSTW(self):
        '''
        The simulation method for the no aggregate shocks version of the model.
        Initializes the agent type, simulates a history of state and control
        variables, and stores the wealth history in self.W_history and the
        annualized MPC history in self.kappa_history.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        self.initializeSim()
        self.simConsHistory()
        self.W_history = self.pHist*self.bHist/self.Rfree
        if Params.do_lifecycle:
            self.W_history = self.W_history*self.cohort_scale
        self.kappa_history = 1.0 - (1.0 - self.MPChist)**4

    def update(self):
        '''
        Update the income process, the assets grid, and the terminal solution.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        orig_flow = self.time_flow
        if self.cycles == 0: # hacky fix for labor supply l_bar
            self.updateIncomeProcessAlt()
        else:
            self.updateIncomeProcess()
        self.updateAssetsGrid()
        self.updateSolutionTerminal()
        self.timeFwd()
        self.resetRNG()
        if self.cycles > 0:
            self.IncomeDstn = Model.applyFlatIncomeTax(self.IncomeDstn,
                                                 tax_rate=self.tax_rate,
                                                 T_retire=self.T_retire,
                                                 unemployed_indices=range(0,(self.TranShkCount+1)*
                                                 self.PermShkCount,self.TranShkCount+1))
        self.makeIncShkHist()
        if not orig_flow:
            self.timeRev()

    def updateIncomeProcessAlt(self):
        '''
        An alternative method for constructing the income process in the infinite
        horizon model, where the labor supply l_bar creates a small oddity.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        tax_rate = (self.IncUnemp*self.UnempPrb)/(self.l_bar*(1.0-self.UnempPrb))
        TranShkDstn     = deepcopy(approxMeanOneLognormal(self.TranShkCount,sigma=self.TranShkStd[0],tail_N=0))
        TranShkDstn[0]  = np.insert(TranShkDstn[0]*(1.0-self.UnempPrb),0,self.UnempPrb)
        TranShkDstn[1]  = np.insert(self.l_bar*TranShkDstn[1]*(1.0-tax_rate),0,self.IncUnemp)
        PermShkDstn     = approxMeanOneLognormal(self.PermShkCount,sigma=self.PermShkStd[0],tail_N=0)
        self.IncomeDstn = [combineIndepDstns(PermShkDstn,TranShkDstn)]
        self.TranShkDstn = TranShkDstn
        self.PermShkDstn = PermShkDstn
        self.addToTimeVary('IncomeDstn')


def assignBetaDistribution(type_list,DiscFac_list):
    '''
    Assigns the discount factors in DiscFac_list to the types in type_list.  If
    there is heterogeneity beyond the discount factor, then the same DiscFac is
    assigned to consecutive types.

    Parameters
    ----------
    type_list : [cstwMPCagent]
        The list of types that should be assigned discount factors.
    DiscFac_list : [float] or np.array
        List of discount factors to assign to the types.

    Returns
    -------
    none
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

    Parameters
    ----------
    sim_wealth : numpy.array
        Array with simulated wealth values.
    weights : numpy.array
        List of weights for each row of sim_wealth.
    total_output : float
        Denominator for the simulated K/Y ratio.
    target_KY : float
        Actual U.S. K/Y ratio to match.

    Returns
    -------
    distance : float
        Absolute distance between simulated and actual K/Y ratios.
    '''
    sim_K = calcWeightedAvg(sim_wealth,weights)/(Params.l_bar)
    sim_KY = sim_K/total_output
    distance = (sim_KY - target_KY)**1.0
    return distance


def calculateLorenzDifference(sim_wealth,weights,percentiles,target_levels):
    '''
    Calculates the sum of squared differences between the simulatedLorenz curve
    at the specified percentile levels and the target Lorenz levels.

    Parameters
    ----------
    sim_wealth : numpy.array
        Array with simulated wealth values.
    weights : numpy.array
        List of weights for each row of sim_wealth.
    percentiles : [float]
        Points in the distribution of wealth to match.
    target_levels : np.array
        Actual U.S. Lorenz curve levels at the specified percentiles.

    Returns
    -------
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

    Parameters
    ----------
    DiscFac : float
        Center of the uniform distribution of discount factors.
    nabla : float
        Width of the uniform distribution of discount factors.
    N : int
        Number of discrete consumer types.
    type_list : [cstwMPCagent]
        List of agent types to solve and simulate after assigning discount factors.
    weights : np.array
        Age-conditional array of population weights.
    total_output : float
        Total output of the economy, denominator for the K/Y calculation.
    target : float
        Target level of capital-to-output ratio.

    Returns
    -------
    my_diff : float
        Difference between simulated and target capital-to-output ratios.
    '''
    if type(DiscFac) in (list,np.ndarray,np.array):
        DiscFac = DiscFac[0]
    DiscFac_list = approxUniform(N,DiscFac-nabla,DiscFac+nabla)[1] # only take values, not probs
    assignBetaDistribution(type_list,DiscFac_list)
    multiThreadCommandsFake(type_list,beta_point_commands)
    my_diff = calculateKYratioDifference(np.vstack((this_type.W_history for this_type in type_list)),
                                         np.tile(weights/float(N),N),total_output,target)
    return my_diff


mystr = lambda number : "{:.3f}".format(number)
'''
Truncates a float at exactly three decimal places when displaying as a string.
'''

def makeCSTWresults(DiscFac,nabla,save_name=None):
    '''
    Produces a variety of results for the cstwMPC paper (usually after estimating).

    Parameters
    ----------
    DiscFac : float
        Center of the uniform distribution of discount factors
    nabla : float
        Width of the uniform distribution of discount factors
    save_name : string
        Name to save the calculated results, for later use in producing figures
        and tables, etc.

    Returns
    -------
    none
    '''
    DiscFac_list = approxUniform(N=Params.pref_type_count,bot=DiscFac-nabla,top=DiscFac+nabla)[1]
    assignBetaDistribution(est_type_list,DiscFac_list)
    multiThreadCommandsFake(est_type_list,beta_point_commands)

    lorenz_distance = np.sqrt(betaDistObjective(nabla))

    makeCSTWstats(DiscFac,nabla,est_type_list,Params.age_weight_all,lorenz_distance,save_name)


def makeCSTWstats(DiscFac,nabla,this_type_list,age_weight,lorenz_distance=0.0,save_name=None):
    '''
    Displays (and saves) a bunch of statistics.  Separate from makeCSTWresults()
    for compatibility with the aggregate shock model.

    Parameters
    ----------
    DiscFac : float
        Center of the uniform distribution of discount factors
    nabla : float
        Width of the uniform distribution of discount factors
    this_type_list : [cstwMPCagent]
        List of agent types in the economy.
    age_weight : np.array
        Age-conditional array of weights for the wealth data.
    lorenz_distance : float
        Distance between simulated and actual Lorenz curves, for display.
    save_name : string
        Name to save the calculated results, for later use in producing figures
        and tables, etc.

    Returns
    -------
    none
    '''
    sim_length = this_type_list[0].sim_periods
    sim_wealth = (np.vstack((this_type.W_history for this_type in this_type_list))).flatten()
    sim_wealth_short = (np.vstack((this_type.W_history[0:sim_length,:] for this_type in this_type_list))).flatten()
    sim_kappa = (np.vstack((this_type.kappa_history for this_type in this_type_list))).flatten()
    sim_income = (np.vstack((this_type.pHist[0:sim_length,:]*np.asarray(this_type.TranShkHist[0:sim_length,:]) for this_type in this_type_list))).flatten()
    sim_ratio = (np.vstack((this_type.W_history[0:sim_length,:]/this_type.pHist[0:sim_length,:] for this_type in this_type_list))).flatten()
    if Params.do_lifecycle:
        sim_unemp = (np.vstack((np.vstack((this_type.IncUnemp == this_type.TranShkHist[0:Params.working_T,:],np.zeros((Params.retired_T+1,this_type_list[0].Nagents),dtype=bool))) for this_type in this_type_list))).flatten()
        sim_emp = (np.vstack((np.vstack((this_type.IncUnemp != this_type.TranShkHist[0:Params.working_T,:],np.zeros((Params.retired_T+1,this_type_list[0].Nagents),dtype=bool))) for this_type in this_type_list))).flatten()
        sim_ret = (np.vstack((np.vstack((np.zeros((Params.working_T,this_type_list[0].Nagents),dtype=bool),np.ones((Params.retired_T+1,this_type_list[0].Nagents),dtype=bool))) for this_type in this_type_list))).flatten()
    else:
        sim_unemp = np.vstack((this_type.IncUnemp == this_type.TranShkHist[0:sim_length,:] for this_type in this_type_list)).flatten()
        sim_emp = np.vstack((this_type.IncUnemp != this_type.TranShkHist[0:sim_length,:] for this_type in this_type_list)).flatten()
        sim_ret = np.zeros(sim_emp.size,dtype=bool)
    sim_weight_all = np.tile(np.repeat(age_weight,this_type_list[0].Nagents),Params.pref_type_count)

    if Params.do_beta_dist and Params.do_lifecycle:
        kappa_mean_by_age_type = (np.mean(np.vstack((this_type.kappa_history for this_type in this_type_list)),axis=1)).reshape((Params.pref_type_count*3,DropoutType.T_total+1))
        kappa_mean_by_age_pref = np.zeros((Params.pref_type_count,DropoutType.T_total+1)) + np.nan
        for j in range(Params.pref_type_count):
            kappa_mean_by_age_pref[j,] = Params.d_pct*kappa_mean_by_age_type[3*j+0,] + Params.h_pct*kappa_mean_by_age_type[3*j+1,] + Params.c_pct*kappa_mean_by_age_type[3*j+2,]
        kappa_mean_by_age = np.mean(kappa_mean_by_age_pref,axis=0)
        kappa_lo_beta_by_age = kappa_mean_by_age_pref[0,:]
        kappa_hi_beta_by_age = kappa_mean_by_age_pref[Params.pref_type_count-1,:]

    lorenz_fig_data = makeLorenzFig(Params.SCF_wealth,Params.SCF_weights,sim_wealth,sim_weight_all)
    mpc_fig_data = makeMPCfig(sim_kappa,sim_weight_all)

    kappa_all = calcWeightedAvg(np.vstack((this_type.kappa_history for this_type in this_type_list)),np.tile(age_weight/float(Params.pref_type_count),Params.pref_type_count))
    kappa_unemp = np.sum(sim_kappa[sim_unemp]*sim_weight_all[sim_unemp])/np.sum(sim_weight_all[sim_unemp])
    kappa_emp = np.sum(sim_kappa[sim_emp]*sim_weight_all[sim_emp])/np.sum(sim_weight_all[sim_emp])
    kappa_ret = np.sum(sim_kappa[sim_ret]*sim_weight_all[sim_ret])/np.sum(sim_weight_all[sim_ret])

    my_cutoffs = [(0.99,1),(0.9,1),(0.8,1),(0.6,0.8),(0.4,0.6),(0.2,0.4),(0.0,0.2)]
    kappa_by_ratio_groups = calcSubpopAvg(sim_kappa,sim_ratio,my_cutoffs,sim_weight_all)
    kappa_by_income_groups = calcSubpopAvg(sim_kappa,sim_income,my_cutoffs,sim_weight_all)

    quintile_points = getPercentiles(sim_wealth_short,weights=sim_weight_all,percentiles=[0.2, 0.4, 0.6, 0.8])
    wealth_quintiles = np.ones(sim_wealth_short.size,dtype=int)
    wealth_quintiles[sim_wealth_short > quintile_points[0]] = 2
    wealth_quintiles[sim_wealth_short > quintile_points[1]] = 3
    wealth_quintiles[sim_wealth_short > quintile_points[2]] = 4
    wealth_quintiles[sim_wealth_short > quintile_points[3]] = 5
    MPC_cutoff = getPercentiles(sim_kappa,weights=sim_weight_all,percentiles=[2.0/3.0])
    these_quintiles = wealth_quintiles[sim_kappa > MPC_cutoff]
    these_weights = sim_weight_all[sim_kappa > MPC_cutoff]
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

    Parameters
    ----------
    real_wealth : np.array
        Data on household wealth.
    real_weights : np.array
        Weighting array of the same size as real_wealth.
    sim_wealth : np.array
        Simulated wealth holdings of many households.
    sim_weights :np.array
        Weighting array of the same size as sim_wealth.

    Returns
    -------
    these_percents : np.array
        An array of percentiles of households, by wealth.
    real_lorenz : np.array
        Lorenz shares for real_wealth corresponding to these_percents.
    sim_lorenz : np.array
        Lorenz shares for sim_wealth corresponding to these_percents.
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

    Parameters
    ----------
    kappa : np.array
        Array of (annualized) marginal propensities to consume for the economy.
    weights : np.array
        Age-conditional weight array for the data in kappa.

    Returns
    -------
    these_percents : np.array
        Array of percentiles of the marginal propensity to consume.
    kappa_percentiles : np.array
        Array of MPCs corresponding to the percentiles in these_percents.
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
    sub-function of sensitivityAnalysis.

    Parameters
    ----------
    DiscFac : float
        Center of the uniform distribution of discount factors
    nabla : float
        Width of the uniform distribution of discount factors

    Returns
    -------
    kappa_all : float
        Average marginal propensity to consume in the population.
    '''
    DiscFac_list = approxUniform(N=Params.pref_type_count,bot=DiscFac-nabla,top=DiscFac+nabla)[1]
    assignBetaDistribution(est_type_list,DiscFac_list)
    multiThreadCommandsFake(est_type_list,beta_point_commands)

    kappa_all = calcWeightedAvg(np.vstack((this_type.kappa_history for this_type in est_type_list)),
                                np.tile(Params.age_weight_all/float(Params.pref_type_count),
                                        Params.pref_type_count))
    return kappa_all


def sensitivityAnalysis(parameter,values,is_time_vary):
    '''
    Perform a sensitivity analysis by varying a chosen parameter over given values
    and re-estimating the model at each.  Only works for perpetual youth version.
    Saves numeric results in a file named SensitivityPARAMETER.txt.

    Parameters
    ----------
    parameter : string
        Name of an attribute/parameter of cstwMPCagent on which to perform a
        sensitivity analysis.  The attribute should be a single float.
    values : [np.array]
        Array of values that the parameter should take on in the analysis.
    is_time_vary : boolean
        Indicator for whether the parameter of analysis is time_varying (i.e.
        is an element of cstwMPCagent.time_vary).  While the sensitivity analysis
        should only be used for the perpetual youth model, some parameters are
        still considered "time varying" in the consumption-saving model and
        are encapsulated in a (length=1) list.

    Returns
    -------
    none
    '''
    fit_list = []
    DiscFac_list = []
    nabla_list = []
    kappa_list = []
    for value in values:
        print('Now estimating model with ' + parameter + ' = ' + str(value))
        Params.diff_save = 1000000.0
        old_value_storage = []
        for this_type in est_type_list:
            old_value_storage.append(getattr(this_type,parameter))
            if is_time_vary:
                setattr(this_type,parameter,[value])
            else:
                setattr(this_type,parameter,value)
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
    with open('./Results/Sensitivity' + parameter + '.txt','w') as f:
        my_writer = csv.writer(f, delimiter='\t',)
        for j in range(len(DiscFac_list)):
            my_writer.writerow([values[j], kappa_list[j], DiscFac_list[j], nabla_list[j], fit_list[j]])
        f.close()
    j = 0
    for this_type in est_type_list:
        setattr(this_type,parameter,old_value_storage[j])
        this_type.update()
        j += 1


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
    a_init = drawDiscrete(N=Params.sim_pop_size,P=Params.a0_probs,X=Params.a0_values,seed=Params.a0_seed)

    # Make the list of types for this run, whether infinite or lifecycle
    if Params.do_lifecycle:
        # Make cohort scaling array
        cohort_scale = Params.TFP_growth**(-np.arange(Params.total_T+1))
        cohort_scale_array = np.tile(np.reshape(cohort_scale,(Params.total_T+1,1)),(1,Params.sim_pop_size))

        # Make base consumer types for each education level
        DropoutType = cstwMPCagent(**Params.init_dropout)
        DropoutType.a_init = a_init
        DropoutType.cohort_scale = cohort_scale_array
        HighschoolType = deepcopy(DropoutType)
        HighschoolType(**Params.adj_highschool)
        CollegeType = deepcopy(DropoutType)
        CollegeType(**Params.adj_college)
        DropoutType.update()
        HighschoolType.update()
        CollegeType.update()

        # Make initial distributions of permanent income for each education level
        p_init_base = drawMeanOneLognormal(N=Params.sim_pop_size, sigma=Params.P0_sigma, seed=Params.P0_seed)
        DropoutType.p_init = Params.P0_d*p_init_base
        HighschoolType.p_init = Params.P0_h*p_init_base
        CollegeType.p_init = Params.P0_c*p_init_base

        # Set the type list for the lifecycle estimation
        short_type_list = [DropoutType, HighschoolType, CollegeType]
        spec_add = 'LC'

    else:
        # Make the base infinite horizon type and assign income shocks
        InfiniteType = cstwMPCagent(**Params.init_infinite)
        InfiniteType.tolerance = 0.0001
        InfiniteType.a_init = 0*np.ones_like(a_init)

        # Make histories of permanent income levels for the infinite horizon type
        p_init_base = np.ones(Params.sim_pop_size,dtype=float)
        InfiniteType.p_init = p_init_base

        # Use a "tractable consumer" instead if desired.
        # If you want this to work, you must edit TractableBufferStockModel slightly.
        # See comments around line 34 in that module for instructions.
        if Params.do_tractable:
            from TractableBufferStockModel import TractableConsumerType
            TractableInfType = TractableConsumerType(DiscFac=0.99, # will be overwritten
                                                     UnempPrb=1-InfiniteType.LivPrb[0],
                                                     Rfree=InfiniteType.Rfree,
                                                     PermGroFac=InfiniteType.PermGroFac[0],
                                                     CRRA=InfiniteType.CRRA,
                                                     sim_periods=InfiniteType.sim_periods,
                                                     IncUnemp=InfiniteType.IncUnemp,
                                                     Nagents=InfiniteType.Nagents)
            TractableInfType.p_init = InfiniteType.p_init
            TractableInfType.timeFwd()
            TractableInfType.TranShkHist = InfiniteType.TranShkHist
            TractableInfType.PermShkHist = InfiniteType.PermShkHist
            TractableInfType.a_init = InfiniteType.a_init

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
    beta_point_commands = ['solve()','unpackcFunc()','timeFwd()','simulateCSTW()']

    # Make the objective function for the beta-point estimation
    betaPointObjective = lambda DiscFac : simulateKYratioDifference(DiscFac,
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
        if Params.do_tractable:
            top = 0.98
        else:
            top = 0.998
        DiscFac_new = brentq(intermediateObjective,0.90,top,xtol=10**(-8))
        N=Params.pref_type_count
        sim_wealth = (np.vstack((this_type.W_history for this_type in est_type_list))).flatten()
        sim_weights = np.tile(np.repeat(Params.age_weight_all,Params.sim_pop_size),N)
        my_diff = calculateLorenzDifference(sim_wealth,sim_weights,Params.percentiles_to_match,lorenz_target)
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
                bot = 0.9
                top = 0.98
            else:
                bot = 0.9
                top = 1.0
            DiscFac = brentq(betaPointObjective,bot,top,xtol=10**(-8))
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
        CRRA_list = np.linspace(0.5,4.0,15).tolist() #15
        sensitivityAnalysis('CRRA',CRRA_list,False)

    if Params.do_sensitivity[1]: # transitory income stdev sensitivity analysis
        TranShkStd_list = [0.01] + np.linspace(0.05,0.8,16).tolist() #16
        sensitivityAnalysis('TranShkStd',TranShkStd_list,True)

    if Params.do_sensitivity[2]: # permanent income stdev sensitivity analysis
        PermShkStd_list = np.linspace(0.02,0.18,17).tolist() #17
        sensitivityAnalysis('PermShkStd',PermShkStd_list,True)

    if Params.do_sensitivity[3]: # unemployment benefits sensitivity analysis
        IncUnemp_list = np.linspace(0.0,0.8,17).tolist() #17
        sensitivityAnalysis('IncUnemp',IncUnemp_list,False)

    if Params.do_sensitivity[4]: # unemployment rate sensitivity analysis
        UnempPrb_list = np.linspace(0.02,0.12,16).tolist() #16
        sensitivityAnalysis('UnempPrb',UnempPrb_list,False)

    if Params.do_sensitivity[5]: # mortality rate sensitivity analysis
        LivPrb_list = 1.0 - np.linspace(0.003,0.0125,16).tolist() #16
        sensitivityAnalysis('LivPrb',LivPrb_list,True)

    if Params.do_sensitivity[6]: # permanent income growth rate sensitivity analysis
        PermGroFac_list = np.linspace(0.00,0.04,17).tolist() #17
        sensitivityAnalysis('PermGroFac',PermGroFac_list,True)

    if Params.do_sensitivity[7]: # interest rate sensitivity analysis
        Rfree_list = (np.linspace(1.0,1.04,17)/InfiniteType.survival_prob[0]).tolist()
        sensitivityAnalysis('Rfree',Rfree_list,False)


    # =======================================================================
    # ========= FBS aggregate shocks model ==================================
    #========================================================================
    if Params.do_agg_shocks:
        # These are the perpetual youth estimates in case we want to skip estimation (and we do)
        beta_point_estimate = 0.989142
        beta_dist_estimate  = 0.985773
        nabla_estimate      = 0.0077

        # Make a set of consumer types for the FBS aggregate shocks model
        BaseAggShksType = AggShockConsumerType(**Params.init_agg_shocks)
        agg_shocks_type_list = []
        for j in range(Params.pref_type_count):
            new_type = deepcopy(BaseAggShksType)
            new_type.seed = j
            new_type.resetRNG()
            new_type.makeIncShkHist()
            agg_shocks_type_list.append(new_type)
        if Params.do_beta_dist:
            beta_agg = beta_dist_estimate
            nabla_agg = nabla_estimate
        else:
            beta_agg = beta_point_estimate
            nabla_agg = 0.0
        DiscFac_list_agg = approxUniform(N=Params.pref_type_count,bot=beta_agg-nabla_agg,top=beta_agg+nabla_agg)[1]
        assignBetaDistribution(agg_shocks_type_list,DiscFac_list_agg)

        # Make a market for solving the FBS aggregate shocks model
        agg_shocks_market = CobbDouglasEconomy(agents = agg_shocks_type_list,
                        act_T         = Params.sim_periods_agg_shocks,
                        tolerance     = 0.0001,
                        **Params.aggregate_params)
        agg_shocks_market.makeAggShkHist()

        # Edit the consumer types so they have the right data
        for this_type in agg_shocks_market.agents:
            this_type.p_init = drawMeanOneLognormal(N=this_type.Nagents,sigma=0.9,seed=0)
            this_type.getEconomyData(agg_shocks_market)

        # Solve the aggregate shocks version of the model
        t_start = time()
        agg_shocks_market.solve()
        t_end = time()
        print('Solving the aggregate shocks model took ' + str(t_end - t_start) + ' seconds.')
        for this_type in agg_shocks_type_list:
            this_type.W_history = this_type.pHist*this_type.bHist
            this_type.kappa_history = 1.0 - (1.0 - this_type.MPChist)**4
        agg_shock_weights = np.concatenate((np.zeros(200),np.ones(Params.sim_periods_agg_shocks-200)))
        agg_shock_weights = agg_shock_weights/np.sum(agg_shock_weights)
        makeCSTWstats(beta_agg,nabla_agg,agg_shocks_type_list,agg_shock_weights)

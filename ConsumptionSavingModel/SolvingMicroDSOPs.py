# Import the HARK library.  The assumption is that this code is in a folder
# contained in the HARK folder. 
import sys 
sys.path.insert(0,'../')

import SetupConsumerParameters as Params
import ConsumptionSavingModel as Model
import SetupSCFdata as Data
from HARKsimulation import drawDiscrete
from HARKestimation import minimizeNelderMead, bootstrapSampleFromData
import numpy as np
import pylab
from time import time

# Set booleans to determine which tasks should be done
estimate_model = True
compute_standard_errors = False
make_contour_plot = True

#=====================================================
# Define objects and functions used for the estimation
#=====================================================

# Make a lifecycle consumer to be used for estimation, including simulated shocks
EstimationAgent = Model.ConsumerType(**Params.init_consumer_objects)

# Make histories of permanent and transitory shocks, plus an initial distribution of wealth
scriptR_shocks, xi_shocks = Model.generateIncomeShockHistoryLognormalUnemployment(EstimationAgent)
w0_vector = drawDiscrete(P=Params.initial_wealth_income_ratio_probs,
                                         X=Params.initial_wealth_income_ratio_vals,
                                         N=Params.num_agents,
                                         seed=Params.seed)
EstimationAgent.addIncomeShockPaths(scriptR_shocks,xi_shocks)

# Define the objective function for the estimation
def smmObjectiveFxn(DiscFacAdj, CRRA,
                     agent = EstimationAgent,
                     DiscFacAdj_bound = Params.DiscFacAdj_bound,
                     CRRA_bound = Params.CRRA_bound,
                     empirical_data = Data.w_to_y_data,
                     empirical_weights = Data.empirical_weights,
                     empirical_groups = Data.empirical_groups,
                     initial_wealth = w0_vector,
                     map_simulated_to_empirical_cohorts = Data.simulation_map_cohorts_to_age_indices):
    '''
    The objective function for the SMM estimation.  Given values of discount-factor
    adjuster DiscFacAdj, coeffecient of relative risk aversion CRRA, a base consumer agent
    type, empirical data, and calibrated parameters, this function calculates the
    weighted distance between data and the simulated wealth-to-permanent income ratio.

    Steps:
        a) solve for consumption functions for (DiscFacAdj, CRRA)
        b) simulate wealth holdings for many consumers over time
        c) sum distances between empirical data and simulated medians within
            seven age groupings
    '''
    
    original_time_flow = agent.time_flow
    agent.timeFwd()
    
    # A quick check to make sure that the parameter values are within bounds.
    # Far flung falues of DiscFacAdj or CRRA might cause an error during solution or 
    # simulation, so the objective function doesn't even bother with them.
    if DiscFacAdj < DiscFacAdj_bound[0] or DiscFacAdj > DiscFacAdj_bound[1] or CRRA < CRRA_bound[0] or CRRA > CRRA_bound[1]:
        return 1e30
        
    # Update the agent with a new path of DiscFac based on this DiscFacAdj (and a new CRRA)
    agent(DiscFac = [b*DiscFacAdj for b in Params.DiscFac_timevary], CRRA = CRRA)
    
    # Solve the model for these parameters, then simulate wealth data
    agent.solve()
    agent.unpack_cFunc()
    max_sim_age = max([max(ages) for ages in map_simulated_to_empirical_cohorts])
    sim_w_history = agent.simulate(w_init=initial_wealth,t_first=0,t_last=max_sim_age)
    
    # Find the distance between empirical data and simulated medians for each age group
    group_count = len(map_simulated_to_empirical_cohorts)
    distance_sum = 0
    for g in range(group_count):
        cohort_indices = map_simulated_to_empirical_cohorts[g]
        sim_median = np.median(sim_w_history[cohort_indices,])
        group_indices = empirical_groups == (g+1) # groups are numbered from 1
        distance_sum += np.dot(np.abs(empirical_data[group_indices] - sim_median),empirical_weights[group_indices])
    
    # Restore time to its original direction and report the result
    if not original_time_flow:
        agent.timeRev()  
    return distance_sum
    
# Make a single-input lambda function for use in the optimizer
smmObjectiveFxnReduced = lambda parameters_to_estimate : smmObjectiveFxn(DiscFacAdj=parameters_to_estimate[0],CRRA=parameters_to_estimate[1])


# Define the bootstrap procedure
def calculateStandardErrorsByBootstrap(initial_estimate,N,seed=0,verbose=False):
    '''
    Estimates the SolvingMicroDSOPs model N times, then calculates standard errors.
    '''
    
    t_0 = time()    
    
    # Generate a list of seeds for generating bootstrap samples
    RNG = np.random.RandomState(seed)
    seed_list = RNG.randint(2**31-1,size=N)    
    
    # Estimate the model N times, recording each set of estimated parameters
    estimate_list = []
    for n in range(N):
        t_start = time()
        
        # Bootstrap a new dataset by sampling 
        bootstrap_data = (bootstrapSampleFromData(Data.scf_data_array,seed=seed_list[n])).T
        w_to_y_data_bootstrap = bootstrap_data[0,]
        empirical_groups_bootstrap = bootstrap_data[1,]
        empirical_weights_bootstrap = bootstrap_data[2,]
        
        # Make a temporary function for use in this estimation run
        smmObjectiveFxnBootstrap = lambda parameters_to_estimate : smmObjectiveFxn(DiscFacAdj=parameters_to_estimate[0],
                                                                                   CRRA=parameters_to_estimate[1],
                                                                                   empirical_data = w_to_y_data_bootstrap,
                                                                                   empirical_weights = empirical_weights_bootstrap,
                                                                                   empirical_groups = empirical_groups_bootstrap)
                                                                                   
        # Estimate the model with the bootstrap data and add to list of estimates
        this_estimate = minimizeNelderMead(smmObjectiveFxnBootstrap,initial_estimate)
        estimate_list.append(this_estimate)
        t_now = time()  
        
        # Report progress of the bootstrap
        if verbose:
            print('Finished bootstrap estimation #' + str(n+1) + ' of ' + str(N) + ' in ' + str(t_now-t_start) + ' seconds (' + str(t_now-t_0) + ' cumulative)')
        
    # Calculate the standard errors for each parameter
    estimate_array = (np.array(estimate_list)).T
    DiscFacAdj_std_error = np.std(estimate_array[0])
    CRRA_std_error = np.std(estimate_array[1])
    
    return [DiscFacAdj_std_error, CRRA_std_error]


#=================================================================
# Done defining objects and functions.  Now run them (if desired).
#=================================================================


# Estimate the model using Nelder-Mead
if estimate_model:
    initial_guess = [Params.DiscFacAdj_start,Params.CRRA_start]
    print('Now estimating the model using Nelder-Mead from an initial guess of' + str(initial_guess) + '...')
    model_estimate = minimizeNelderMead(smmObjectiveFxnReduced,initial_guess,verbose=True)
    print('Estimated values: DiscFacAdj=' + str(model_estimate[0]) + ', CRRA=' + str(model_estimate[1]))

# Compute standard errors by bootstrap
if compute_standard_errors:
    std_errors = calculateStandardErrorsByBootstrap(model_estimate,N=Params.bootstrap_size,seed=Params.seed,verbose=True)
    print('Standard errors: DiscFacAdj--> ' + str(std_errors[0]) + ', CRRA--> ' + str(std_errors[1]))

# Make a contour plot of the objective function
if make_contour_plot:
    grid_density = 20
    level_count = 100
    DiscFacAdj_list = np.linspace(0.85,1.05,grid_density)
    CRRA_list = np.linspace(2,8,grid_density)
    CRRA_mesh, DiscFacAdj_mesh = pylab.meshgrid(CRRA_list,DiscFacAdj_list)
    smm_obj_levels = np.empty([grid_density,grid_density])
    for j in range(grid_density):
        DiscFacAdj = DiscFacAdj_list[j]
        for k in range(grid_density):
            CRRA = CRRA_list[k]
            smm_obj_levels[j,k] = smmObjectiveFxn(DiscFacAdj,CRRA)    
    smm_contour = pylab.contourf(CRRA_mesh,DiscFacAdj_mesh,smm_obj_levels,level_count)
    pylab.colorbar(smm_contour)
    pylab.plot(model_estimate[1],model_estimate[0],'*r',ms=15)
    pylab.xlabel(r'coefficient of relative risk aversion $\rho$',fontsize=14)
    pylab.ylabel(r'discount factor adjustment $\beth$',fontsize=14)
    pylab.savefig('SMMcontour.pdf')
    pylab.savefig('SMMcontour.png')
    pylab.show()


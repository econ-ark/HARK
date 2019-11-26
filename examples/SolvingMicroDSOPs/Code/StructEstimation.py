'''
Demonstrates an example estimation of microeconomic dynamic stochastic optimization
problem, as described in Section 9 of Chris Carroll's SolvingMicroDSOPs.pdf notes.
The estimation attempts to match the age-conditional wealth profile of simulated
consumers to the median wealth holdings of seven age groups in the 2004 SCF by
varying only two parameters: the coefficient of relative risk aversion and a scaling
factor for an age-varying sequence of discount factors.  The estimation uses a
consumption-saving model with idiosyncratic shocks to permanent and transitory
income as defined in ConsIndShockModel.
'''
from __future__ import print_function
from __future__ import absolute_import

from builtins import str
from builtins import range

import os
import sys 
import csv
import numpy as np                              # Numeric Python
import pylab                                    # Python reproductions of some Matlab functions
from time import time, clock                    # Timing utility

# Import modules from core HARK libraries:
import HARK.ConsumptionSaving.ConsIndShockModel as Model # The consumption-saving micro model
from HARK.simulation import drawDiscrete         # Method for sampling from a discrete distribution
from HARK.estimation import minimizeNelderMead, bootstrapSampleFromData # Estimation methods

# Find pathname to this file:
my_file_path = os.path.dirname(os.path.abspath(__file__))

# Pathnames to the other files:
calibration_dir = os.path.join(my_file_path, "../Calibration/") # Relative directory for primitive parameter files
tables_dir = os.path.join(my_file_path, "../Tables/") # Relative directory for primitive parameter files
figures_dir = os.path.join(my_file_path, "../Figures/") # Relative directory for primitive parameter files
code_dir = os.path.join(my_file_path, "../Code/") # Relative directory for primitive parameter files


# Import modules from local repository. If local repository is part of HARK, 
# this will import from HARK. Otherwise manual pathname specification is in 
# order.
try: 
    # Import from core HARK code first:
    from HARK.SolvingMicroDSOPs.Calibration import EstimationParameters as Params           # Parameters for the consumer type and the estimation
    from HARK.SolvingMicroDSOPs.Calibration import SetupSCFdata as Data                     # SCF 2004 data on household wealth

except:
    # Need to rely on the manual insertion of pathnames to all files in do_all.py
    # NOTE sys.path.insert(0, os.path.abspath(tables_dir)), etc. may need to be 
    # copied from do_all.py to here
    import EstimationParameters as Params           # Parameters for the consumer type and the estimation
    import SetupSCFdata as Data                     # SCF 2004 data on household wealth



# Set booleans to determine which tasks should be done
local_estimate_model = True             # Whether to estimate the model
local_compute_standard_errors = False   # Whether to get standard errors via bootstrap
local_make_contour_plot = True         # Whether to make a contour map of the objective function

#=====================================================
# Define objects and functions used for the estimation
#=====================================================

class TempConsumerType(Model.IndShockConsumerType):
    '''
    A very lightly edited version of IndShockConsumerType.  Uses an alternate method of making new
    consumers and specifies DiscFac as being age-dependent.  Called "temp" because only used here.
    '''
    def __init__(self,cycles=1,time_flow=True,**kwds):
        '''
        Make a new consumer type.

        Parameters
        ----------
        cycles : int
            Number of times the sequence of periods should be solved.
        time_flow : boolean
            Whether time is currently "flowing" forward for this instance.

        Returns
        -------
        None
        '''
        # Initialize a basic AgentType
        Model.IndShockConsumerType.__init__(self,cycles=cycles,time_flow=time_flow,**kwds)
        self.addToTimeVary('DiscFac') # This estimation uses age-varying discount factors as
        self.delFromTimeInv('DiscFac')# estimated by Cagetti (2003), so switch from time_inv to time_vary

    def simBirth(self,which_agents):
        '''
        Alternate method for simulating initial states for simulated agents, drawing from a finite
        distribution.  Used to overwrite IndShockConsumerType.simBirth, which uses lognormal distributions.

        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.AgentCount indicating which agents should be "born".

        Returns
        -------
        None
        '''
        # Get and store states for newly born agents
        self.aNrmNow[which_agents] = self.aNrmInit[which_agents] # Take directly from pre-specified distribution
        self.pLvlNow[which_agents] = 1.0 # No variation in permanent income needed
        self.t_age[which_agents]   = 0 # How many periods since each agent was born
        self.t_cycle[which_agents] = 0 # Which period of the cycle each agents is currently in
        return None


# Make a lifecycle consumer to be used for estimation, including simulated shocks (plus an initial distribution of wealth)
EstimationAgent = TempConsumerType(**Params.init_consumer_objects)   # Make a TempConsumerType for estimation
EstimationAgent(T_sim = EstimationAgent.T_cycle+1)                   # Set the number of periods to simulate
EstimationAgent.track_vars = ['bNrmNow']                             # Choose to track bank balances as wealth
EstimationAgent.aNrmInit = drawDiscrete(N=Params.num_agents,
                                      P=Params.initial_wealth_income_ratio_probs,
                                      X=Params.initial_wealth_income_ratio_vals,
                                      seed=Params.seed)              # Draw initial assets for each consumer
EstimationAgent.makeShockHistory()

# Define the objective function for the simulated method of moments estimation
def smmObjectiveFxn(DiscFacAdj, CRRA,
                     agent = EstimationAgent,
                     DiscFacAdj_bound = Params.DiscFacAdj_bound,
                     CRRA_bound = Params.CRRA_bound,
                     empirical_data = Data.w_to_y_data,
                     empirical_weights = Data.empirical_weights,
                     empirical_groups = Data.empirical_groups,
                     map_simulated_to_empirical_cohorts = Data.simulation_map_cohorts_to_age_indices):
    '''
    The objective function for the SMM estimation.  Given values of discount factor
    adjuster DiscFacAdj, coeffecient of relative risk aversion CRRA, a base consumer
    agent type, empirical data, and calibrated parameters, this function calculates
    the weighted distance between data and the simulated wealth-to-permanent
    income ratio.

    Steps:
        a) solve for consumption functions for (DiscFacAdj, CRRA)
        b) simulate wealth holdings for many consumers over time
        c) sum distances between empirical data and simulated medians within
            seven age groupings

    Parameters
    ----------
    DiscFacAdj : float
        An adjustment factor to a given age-varying sequence of discount factors.
        I.e. DiscFac[t] = DiscFacAdj*DiscFac_timevary[t].
    CRRA : float
        Coefficient of relative risk aversion.
    agent : ConsumerType
        The consumer type to be used in the estimation, with all necessary para-
        meters defined except the discount factor and CRRA.
    DiscFacAdj_bound : (float,float)
        Lower and upper bounds on DiscFacAdj; if outside these bounds, the function
        simply returns a "penalty value".
    DiscFacAdj_bound : (float,float)
        Lower and upper bounds on CRRA; if outside these bounds, the function
        simply returns a "penalty value".
    empirical_data : np.array
        Array of wealth-to-permanent-income ratios in the data.
    empirical_weights : np.array
        Weights for each observation in empirical_data.
    empirical_groups : np.array
        Array of integers listing the age group for each observation in empirical_data.
    map_simulated_to_empirical_cohorts : [np.array]
        List of arrays of "simulation ages" for each age grouping.  E.g. if the
        0th element is [1,2,3,4,5], then these time indices from the simulation
        correspond to the 0th empirical age group.

    Returns
    -------
    distance_sum : float
        Sum of distances between empirical data observations and the corresponding
        median wealth-to-permanent-income ratio in the simulation.
    '''
    original_time_flow = agent.time_flow
    agent.timeFwd() # Make sure time is flowing forward for the agent

    # A quick check to make sure that the parameter values are within bounds.
    # Far flung falues of DiscFacAdj or CRRA might cause an error during solution or
    # simulation, so the objective function doesn't even bother with them.
    if DiscFacAdj < DiscFacAdj_bound[0] or DiscFacAdj > DiscFacAdj_bound[1] or CRRA < CRRA_bound[0] or CRRA > CRRA_bound[1]:
        return 1e30

    # Update the agent with a new path of DiscFac based on this DiscFacAdj (and a new CRRA)
    agent(DiscFac = [b*DiscFacAdj for b in Params.DiscFac_timevary], CRRA = CRRA)

    # Solve the model for these parameters, then simulate wealth data
    agent.solve()        # Solve the microeconomic model
    agent.unpackcFunc() # "Unpack" the consumption function for convenient access
    max_sim_age = max([max(ages) for ages in map_simulated_to_empirical_cohorts])+1
    agent.initializeSim()                     # Initialize the simulation by clearing histories, resetting initial values
    agent.simulate(max_sim_age)               # Simulate histories of consumption and wealth
    sim_w_history = agent.bNrmNow_hist        # Take "wealth" to mean bank balances before receiving labor income

    # Find the distance between empirical data and simulated medians for each age group
    group_count = len(map_simulated_to_empirical_cohorts)
    distance_sum = 0
    for g in range(group_count):
        cohort_indices = map_simulated_to_empirical_cohorts[g] # The simulated time indices corresponding to this age group
        sim_median = np.median(sim_w_history[cohort_indices,]) # The median of simulated wealth-to-income for this age group
        group_indices = empirical_groups == (g+1) # groups are numbered from 1
        distance_sum += np.dot(np.abs(empirical_data[group_indices] - sim_median),empirical_weights[group_indices]) # Weighted distance from each empirical observation to the simulated median for this age group

    # Restore time to its original direction and report the result
    if not original_time_flow:
        agent.timeRev()
    return distance_sum

# Make a single-input lambda function for use in the optimizer
smmObjectiveFxnReduced = lambda parameters_to_estimate : smmObjectiveFxn(DiscFacAdj=parameters_to_estimate[0],CRRA=parameters_to_estimate[1])
'''
A "reduced form" of the SMM objective function, compatible with the optimizer.
Identical to smmObjectiveFunction, but takes only a single input as a length-2
list representing [DiscFacAdj,CRRA].
'''

# Define the bootstrap procedure
def calculateStandardErrorsByBootstrap(initial_estimate,N,seed=0,verbose=False):
    '''
    Calculates standard errors by repeatedly re-estimating the model with datasets
    resampled from the actual data.

    Parameters
    ----------
    initial_estimate : [float,float]
        The estimated [DiscFacAdj,CRRA], for use as an initial guess for each
        re-estimation in the bootstrap procedure.
    N : int
        Number of times to resample data and re-estimate the model.
    seed : int
        Seed for the random number generator.
    verbose : boolean
        Indicator for whether extra output should be printed for the user.

    Returns
    -------
    standard_errors : [float,float]
        Standard errors calculated by bootstrap: [DiscFacAdj_std_error, CRRA_std_error].
    '''
    t_0 = time()

    # Generate a list of seeds for generating bootstrap samples
    RNG = np.random.RandomState(seed)
    seed_list = RNG.randint(2**31-1,size=N)

    # Estimate the model N times, recording each set of estimated parameters
    estimate_list = []
    for n in range(N):
        t_start = time()

        # Bootstrap a new dataset by resampling from the original data
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

def main(estimate_model=local_estimate_model, compute_standard_errors=local_compute_standard_errors, make_contour_plot=local_make_contour_plot):
    """
    Run the main estimation procedure for SolvingMicroDSOP.
    
    Parameters
    ----------
    estimate_model : bool
        Whether to estimate the model using Nelder-Mead. When True, this is a low-time, low-memory operation.
    
    compute_standard_errors : bool
        Whether to compute standard errors on the estiamtion of the model.
    
    make_contour_plot : bool
        Whether to make the contour plot associate with the estiamte. 
    
    Returns
    -------
    None
    """
    
    
    # Estimate the model using Nelder-Mead
    if estimate_model:
        initial_guess = [Params.DiscFacAdj_start,Params.CRRA_start]
        print('--------------------------------------------------------------------------------')
        print('Now estimating the model using Nelder-Mead from an initial guess of ' + str(initial_guess) + '...')
        print('--------------------------------------------------------------------------------')            
        t_start_estimate = clock()
        model_estimate = minimizeNelderMead(smmObjectiveFxnReduced,initial_guess,verbose=True)
        t_end_estimate = clock()
        time_to_estimate = t_end_estimate-t_start_estimate
        print('Time to execute all:', round(time_to_estimate/60.,2), 'min,', time_to_estimate, 'sec')
        print('Estimated values: DiscFacAdj=' + str(model_estimate[0]) + ', CRRA=' + str(model_estimate[1]))

        # Create the simple estimate table
        estimate_results_file = os.path.join(tables_dir, 'estimate_results.csv')
        with open(estimate_results_file, 'wt') as f:
            writer = csv.writer(f)
            writer.writerow(['DiscFacAdj', 'CRRA'])
            writer.writerow([model_estimate[0], model_estimate[1]])


    if compute_standard_errors and not estimate_model:
        print("To run the bootstrap you must first estimate the model by setting estimate_model = True.")
        
    # Compute standard errors by bootstrap
    if compute_standard_errors and estimate_model:
            

        # Estimate the model:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print("Computing standard errors using",Params.bootstrap_size,"bootstrap replications.")
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        try:
            t_bootstrap_guess = time_to_estimate * Params.bootstrap_size
            print("This will take approximately", round(t_bootstrap_guess/60.,2), "min, ", t_bootstrap_guess, "sec")
        except:
            pass
        t_start_bootstrap = clock()
        std_errors = calculateStandardErrorsByBootstrap(model_estimate,N=Params.bootstrap_size,seed=Params.seed,verbose=True)
        t_end_bootstrap = clock()
        time_to_bootstrap = t_end_bootstrap-t_start_bootstrap
        print('Time to execute all:', round(time_to_bootstrap/60.,2), 'min,', time_to_bootstrap, 'sec')
        print('Standard errors: DiscFacAdj--> ' + str(std_errors[0]) + ', CRRA--> ' + str(std_errors[1]))

        # Create the simple bootstrap table
        bootstrap_results_file = os.path.join(tables_dir, 'bootstrap_results.csv')
        with open(bootstrap_results_file, 'wt') as f:
            writer = csv.writer(f)
            writer.writerow(['DiscFacAdj', 'DiscFacAdj_standard_error', 'CRRA', 'CRRA_standard_error'])
            writer.writerow([model_estimate[0], std_errors[0], model_estimate[1], std_errors[1]])

    # Make a contour plot of the objective function
    if make_contour_plot:
        print('````````````````````````````````````````````````````````````````````````````````')
        print("Creating the contour plot.")
        print('````````````````````````````````````````````````````````````````````````````````')
        t_start_contour = clock()
        grid_density = 20   # Number of parameter values in each dimension
        level_count = 100   # Number of contour levels to plot
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
        t_end_contour = clock()
        time_to_contour = t_end_contour-t_start_contour
        print('Time to execute all:', round(time_to_contour/60.,2), 'min,', time_to_contour, 'sec')
        pylab.colorbar(smm_contour)
        pylab.plot(model_estimate[1],model_estimate[0],'*r',ms=15)
        pylab.xlabel(r'coefficient of relative risk aversion $\rho$',fontsize=14)
        pylab.ylabel(r'discount factor adjustment $\beth$',fontsize=14)
        pylab.savefig(os.path.join(figures_dir, 'SMMcontour.pdf'))
        pylab.savefig(os.path.join(figures_dir, 'SMMcontour.png'))
        pylab.show()



if __name__ == '__main__':
    main()


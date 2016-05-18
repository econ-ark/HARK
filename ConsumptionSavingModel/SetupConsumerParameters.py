'''
The SetupConsumersParameters specifiesthe full set of calibrated values required
to estimate the SolvingMicroDSOPs model.  The empirical data is stored in a
separate csv file and is loaded in SetupSCFdata.  These parameters are also used
as default settings in various examples of ConsumptionSavingModel.
'''

# ---------------------------------------------------------------------------------
# - Define all of the model parameters for SolvingMicroDSOPs and ConsumerExamples -
# ---------------------------------------------------------------------------------

exp_nest = 3                        # Number of times to "exponentially nest" when constructing a_grid
aXtraMin = 0.001                       # Minimum end-of-period assets value in a_grid
aXtraMax = 20                         # Maximum end-of-period assets value in a_grid                  
aXtraHuge = None                       # A very large value of assets to add to the grid, not used
aXtraExtra = None                      # Some other value of assets to add to the grid, not used
aXtraCount = 12                         # Number of points in the grid of assets

BoroCnstArt = 0.0                  # Artificial borrowing constraint
CubicBool = True                # Use cubic spline interpolation when True, linear interpolation when False
vFuncBool = False                 # Whether to calculate the value function during solution

Rfree = 1.03                            # Interest factor on assets
PermShkCount = 7                           # Number of points in discrete approximation to permanent income shocks
TranShkCount = 7                            # Number of points in discrete approximation to transitory income shocks
UnempPrb = 0.05                  # Probability of unemployment while working
UnempPrbRet = 0.005          # Probability of "unemployment" while retired
IncUnemp = 0.3                 # Unemployment benefits replacement rate
IncUnempRet = 0.0        # Ditto when retired

final_age = 90                      # Age at which the problem ends (die with certainty)
retirement_age = 65                 # Age at which the consumer retires
initial_age = 25                    # Age at which the consumer enters the model
TT = final_age - initial_age        # Total number of periods in the model
retirement_t = retirement_age - initial_age - 1

CRRA_start = 4.0                     # Initial guess of rho (CRRA) during estimation
DiscFacAdj_start = 0.99                   # Initial guess of beth during estimation
DiscFacAdj_bound = [0.0001,15.0]          # Bounds for beth; if violated, objective function returns "penalty value"
CRRA_bound = [0.0001,15.0]           # Bounds for rho; if violated, objective function returns "penalty value"

# Expected growth rates of permanent income
PermGroFac = [ 1.025,  1.025,  1.025,  1.025,  1.025,  1.025,  1.025,  1.025,
        1.025,  1.025,  1.025,  1.025,  1.025,  1.025,  1.025,  1.025,
        1.025,  1.025,  1.025,  1.025,  1.025,  1.025,  1.025,  1.025,
        1.025,  1.01 ,  1.01 ,  1.01 ,  1.01 ,  1.01 ,  1.01 ,  1.01 ,
        1.01 ,  1.01 ,  1.01 ,  1.01 ,  1.01 ,  1.01 ,  1.01 ,  0.7 ,
        1.  ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
        1.   ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
        1.   ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ]

# Age-varying discount factors, lifted from Cagetti (2003)
DiscFac_timevary = [1.064914 ,  1.057997 ,  1.051422 ,  1.045179 ,  1.039259 ,
        1.033653 ,  1.028352 ,  1.023348 ,  1.018632 ,  1.014198 ,
        1.010037 ,  1.006143 ,  1.002509 ,  0.9991282,  0.9959943,
        0.9931012,  0.9904431,  0.9880143,  0.9858095,  0.9838233,
        0.9820506,  0.9804866,  0.9791264,  0.9779656,  0.9769995,
        0.9762239,  0.9756346,  0.9752274,  0.9749984,  0.9749437,
        0.9750595,  0.9753422,  0.9757881,  0.9763936,  0.9771553,
        0.9780698,  0.9791338,  0.9803439,  0.981697 ,  0.8287214,
        0.9902111,  0.9902111,  0.9902111,  0.9902111,  0.9902111,
        0.9902111,  0.9902111,  0.9902111,  0.9902111,  0.9902111,
        0.9902111,  0.9902111,  0.9902111,  0.9902111,  0.9902111,
        0.9902111,  0.9902111,  0.9902111,  0.9902111,  0.9902111,
        0.9902111,  0.9902111,  0.9902111,  0.9902111,  0.9902111]

# Survival probabilities
LivPrb = [ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        0.98438596,  0.98438596,  0.98438596,  0.98438596,  0.98438596,
        0.97567062,  0.97567062,  0.97567062,  0.97567062,  0.97567062,
        0.96207901,  0.96207901,  0.96207901,  0.96207901,  0.96207901,
        0.93721595,  0.93721595,  0.93721595,  0.93721595,  0.93721595,
        0.63095734,  0.63095734,  0.63095734,  0.63095734,  0.63095734]

# Standard deviations of permanent income shocks by age
PermShkStd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Standard deviations of transitory income shocks by age
TranShkStd =  [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Age groups for the estimation: calculate average wealth-to-permanent income ratio
# for consumers within each of these age groups, compare actual to simulated data
empirical_cohort_age_groups = [[ 26,27,28,29,30 ],
                     [ 31,32,33,34,35 ],
                     [ 36,37,38,39,40 ],
                     [ 41,42,43,44,45 ],
                     [ 46,47,48,49,50 ],
                     [ 51,52,53,54,55 ],
                     [ 56,57,58,59,60 ]]

initial_wealth_income_ratio_vals = [0.17, 0.5, 0.83]            # Three point discrete distribution of initial w
initial_wealth_income_ratio_probs = [0.33333, 0.33333, 0.33334] # Equiprobable discrete distribution of initial w
num_agents = 10000                                              # Number of agents to simulate
bootstrap_size = 50                                             # Number of re-estimations to do during bootstrap
seed = 31382

# ------------------------------------------------------------------------------
# -- Set up the dictionary "container" for making a base lifecycle type --------
# ------------------------------------------------------------------------------

# Dictionary that can be passed to ConsumerType to instantiate
init_consumer_objects = {"CRRA":CRRA_start,
                        "Rfree":Rfree,
                        "PermGroFac":PermGroFac,
                        "BoroCnstArt":BoroCnstArt,
                        "PermShkStd":PermShkStd,
                        "PermShkCount":PermShkCount,
                        "TranShkStd":TranShkStd,
                        "TranShkCount":TranShkCount,
                        "T_total":TT,
                        "UnempPrb":UnempPrb,
                        "UnempPrbRet":UnempPrbRet,
                        "T_retire":retirement_t,
                        "IncUnemp":IncUnemp,
                        "IncUnempRet":IncUnempRet,
                        "aXtraMin":aXtraMin,
                        "aXtraMax":aXtraMax,
                        "aXtraCount":aXtraCount,
                        "aXtraExtra":[aXtraExtra,aXtraHuge],
                        "exp_nest":exp_nest,
                        "LivPrb":LivPrb,
                        "DiscFac":DiscFac_timevary,
                        'Nagents':num_agents,
                        'seed':seed,
                        'tax_rate':0.0,
                        'vFuncBool':vFuncBool,
                        'CubicBool':CubicBool
                        }

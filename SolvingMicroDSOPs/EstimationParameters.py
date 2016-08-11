'''
SetupConsumersParameters specifies the full set of calibrated values required
to estimate the SolvingMicroDSOPs model.  The empirical data is stored in a
separate csv file and is loaded in SetupSCFdata.  These parameters are also used
as default settings in various examples of ConsumptionSavingModel.
'''
from copy import copy
import numpy as np
# ---------------------------------------------------------------------------------
# - Define all of the model parameters for SolvingMicroDSOPs and ConsumerExamples -
# ---------------------------------------------------------------------------------

exp_nest = 3                        # Number of times to "exponentially nest" when constructing a_grid
aXtraMin = 0.001                    # Minimum end-of-period "assets above minimum" value
aXtraMax = 20                       # Minimum end-of-period "assets above minimum" value               
aXtraHuge = None                    # A very large value of assets to add to the grid, not used
aXtraExtra = None                   # Some other value of assets to add to the grid, not used
aXtraCount = 8                      # Number of points in the grid of "assets above minimum"

#PNG Commented this out 2016-07-29
#BoroCnstArt = 0.0                   # Artificial borrowing constraint; imposed minimum level of end-of period assets
CubicBool = True                    # Use cubic spline interpolation when True, linear interpolation when False
vFuncBool = False                   # Whether to calculate the value function during solution

Rfree = 1.03                        # Interest factor on assets
PermShkCount = 7                    # Number of points in discrete approximation to permanent income shocks
TranShkCount = 7                    # Number of points in discrete approximation to transitory income shocks
UnempPrb = 0.005                     # Probability of unemployment while working
UnempPrbRet = 0.000                 # Probability of "unemployment" while retired
IncUnemp = 0.0                      # Unemployment benefits replacement rate
IncUnempRet = 0.0                   # "Unemployment" benefits when retired

final_age = 90                      # Age at which the problem ends (die with certainty)
retirement_age = 65                 # Age at which the consumer retires
initial_age = 25                    # Age at which the consumer enters the model
TT = final_age - initial_age        # Total number of periods in the model
retirement_t = retirement_age - initial_age - 1

CRRA_start = 4.0                    # Initial guess of the coefficient of relative risk aversion during estimation (rho)
DiscFacAdj_start = 0.99             # Initial guess of the adjustment to the discount factor during estimation (beth)
DiscFacAdj_bound = [0.0001,15.0]    # Bounds for beth; if violated, objective function returns "penalty value"
CRRA_bound = [0.0001,15.0]          # Bounds for rho; if violated, objective function returns "penalty value"

# Expected growth rates of permanent income over the lifecycle, starting from age 25
PermGroFac = [ 1.025,  1.025,  1.025,  1.025,  1.025,  1.025,  1.025,  1.025,
        1.025,  1.025,  1.025,  1.025,  1.025,  1.025,  1.025,  1.025,
        1.025,  1.025,  1.025,  1.025,  1.025,  1.025,  1.025,  1.025,
        1.025,  1.01 ,  1.01 ,  1.01 ,  1.01 ,  1.01 ,  1.01 ,  1.01 ,
        1.01 ,  1.01 ,  1.01 ,  1.01 ,  1.01 ,  1.01 ,  1.01 ,  0.7 , # <-- This represents retirement
        1.  ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
        1.   ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
        1.   ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ]

# Age-varying discount factors over the lifecycle, lifted from Cagetti (2003)
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

# Survival probabilities over the lifecycle, starting from age 25
LivPrb = [ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
           1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
           1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
           1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
           1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
           1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
           1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
           1.        ,  1.        ,  1.        ,  1.        ,  1.        , # <-- automatic survival to age 65
           0.98438596,  0.98438596,  0.98438596,  0.98438596,  0.98438596,
           0.97567062,  0.97567062,  0.97567062,  0.97567062,  0.97567062,
           0.96207901,  0.96207901,  0.96207901,  0.96207901,  0.96207901,
           0.93721595,  0.93721595,  0.93721595,  0.93721595,  0.93721595,
           0.63095734,  0.63095734,  0.63095734,  0.63095734,  0.63095734]

#Borrowing constraint over lifecycle, starting from age 25 
age_relaxed = 45       
BoroCnstArt_timevary =  []  
for i in range(age_relaxed - initial_age):     
    BoroCnstArt_timevary.append(0.0)
for i in range(final_age - age_relaxed):     
    BoroCnstArt_timevary.append(0.0)
    
rebate_age_65 = 0.0

# Standard deviations of permanent income shocks by age, starting from age 25
PermShkStd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, # <-- no permanent income shocks after retirement
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

#PermShkStd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # <-- no permanent income shocks after retirement
#0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Standard deviations of transitory income shocks by age, starting from age 25
TranShkStd =  [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, # <-- no transitory income shocs after retirement
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
seed = 31382                                                    # Just an integer to seed the estimation

# -----------------------------------------------------------------------------
# -- Set up the dictionary "container" for making a basic lifecycle type ------
# -----------------------------------------------------------------------------

# Dictionary that can be passed to ConsumerType to instantiate
init_consumer_objects = {"CRRA":CRRA_start,
                        "Rfree":Rfree,
                        "PermGroFac":PermGroFac,
                        "BoroCnstArt": BoroCnstArt_timevary, #"BoroCnstArt":BoroCnstArt,
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
                        'CubicBool':CubicBool,
                        'two_state':False,
                        'rebate_age_65':rebate_age_65
                        }


# -----------------------------------------------------------------------------
# ----- Define additional parameters for the aggregate shocks model -----------
# -----------------------------------------------------------------------------
kGridBase = np.array([0.3,0.6,0.8,0.9,0.98,1.0,1.02,1.1,1.2,1.6])  # Grid of capital-to-labor-ratios (factors)

# Parameters for a Cobb-Douglas economy
PermShkAggCount = 3           # Number of points in discrete approximation to aggregate permanent shock dist
TranShkAggCount = 3           # Number of points in discrete approximation to aggregate transitory shock dist
PermShkAggStd = 0.01          # Standard deviation of log aggregate permanent shocks
TranShkAggStd = 0.01          # Standard deviation of log aggregate transitory shocks
DeprFac = 0.1                 # Capital depreciation rate
CapShare = 0.3                # Capital's share of income
CRRAPF = 1.0                  # CRRA of perfect foresight calibration
DiscFacPF = 0.96              # Discount factor of perfect foresight calibration
intercept_prev = 0.1          # Intercept of log-capital-ratio function
slope_prev = 0.9              # Slope of log-capital-ratio function

# Make a dictionary to specify an aggregate shocks consumer
init_agg_shocks = copy(init_consumer_objects)
del init_agg_shocks['Rfree']       # Interest factor is endogenous in agg shocks model
del init_agg_shocks['BoroCnstArt'] # Not supported yet for agg shocks model
del init_agg_shocks['CubicBool']   # Not supported yet for agg shocks model
del init_agg_shocks['vFuncBool']   # Not supported yet for agg shocks model
init_agg_shocks['kGridBase'] = kGridBase
init_agg_shocks['aXtraCount'] = 20
init_agg_shocks['two_state'] = True

# Make a dictionary to specify a Cobb-Douglas economy
init_cobb_douglas = {'PermShkAggCount': PermShkAggCount,
                     'TranShkAggCount': TranShkAggCount,
                     'PermShkAggStd': PermShkAggStd,
                     'TranShkAggStd': TranShkAggStd,
                     'DeprFac': DeprFac,
                     'CapShare': CapShare,
                     'CRRA': CRRAPF,
                     'DiscFac': DiscFacPF,
                     'slope_prev': slope_prev,
                     'intercept_prev': intercept_prev
                     }

if __name__ == '__main__':
    print("Sorry, EstimationParameters doesn't actually do anything on its own.")
    print("This module is imported by StructEstimation, providing calibrated ")
    print("parameters for the example estimation.  Please see that module if you ")
    print("want more interesting output.")


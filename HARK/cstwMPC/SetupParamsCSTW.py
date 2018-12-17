'''
Loads parameters used in the cstwMPC estimations.
'''

from __future__ import division, print_function

from builtins import range
import numpy as np
import csv
from copy import  deepcopy
import os

# Choose percentiles of the data to match and which estimation to run
spec_name = 'BetaDistPY'
param_name = 'DiscFac'        # Which parameter to introduce heterogeneity in
dist_type = 'uniform'         # Which type of distribution to use
do_lifecycle = False          # Use lifecycle model if True, perpetual youth if False
do_param_dist = False         # Do param-dist version if True, param-point if False
run_estimation = True         # Runs the estimation if True
find_beta_vs_KY = False       # Computes K/Y ratio for a wide range of beta; should have do_beta_dist = False
do_sensitivity = [False, False, False, False, False, False, False, False] # Choose which sensitivity analyses to run: rho, xi_sigma, psi_sigma, mu, urate, mortality, g, R
do_liquid = False             # Matches liquid assets data when True, net worth data when False
do_tractable = False          # Uses a "tractable consumer" rather than solving full model when True
do_agg_shocks = False         # Solve the FBS aggregate shocks version of the model
SCF_data_file = 'SCFwealthDataReduced.txt'
percentiles_to_match = [0.2, 0.4, 0.6, 0.8]    # Which points of the Lorenz curve to match in beta-dist (must be in (0,1))
#percentiles_to_match = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # Can use this line if you want to match more percentiles
if do_param_dist:
    pref_type_count = 7       # Number of discrete beta types in beta-dist
else:
    pref_type_count = 1       # Just one beta type in beta-point

# Set basic parameters for the lifecycle micro model
init_age = 24                 # Starting age for agents
Rfree = 1.04**(0.25)          # Quarterly interest factor
working_T = 41*4              # Number of working periods
retired_T = 55*4              # Number of retired periods
T_cycle = working_T+retired_T # Total number of periods
CRRA = 1.0                    # Coefficient of relative risk aversion
DiscFac_guess = 0.99          # Initial starting point for discount factor
UnempPrb = 0.07               # Probability of unemployment while working
UnempPrbRet = 0.0005          # Probabulity of "unemployment" while retired
IncUnemp = 0.15               # Unemployment benefit replacement rate
IncUnempRet = 0.0             # Ditto when retired
BoroCnstArt = 0.0             # Artificial borrowing constraint

# Set grid sizes
PermShkCount = 5              # Number of points in permanent income shock grid
TranShkCount = 5              # Number of points in transitory income shock grid
aXtraMin = 0.00001            # Minimum end-of-period assets in grid
aXtraMax = 20                 # Maximum end-of-period assets in grid
aXtraCount = 20               # Number of points in assets grid
aXtraNestFac = 3              # Number of times to 'exponentially nest' when constructing assets grid
CubicBool = False             # Whether to use cubic spline interpolation
vFuncBool = False             # Whether to calculate the value function during solution

# Set simulation parameters
if do_param_dist:
    if do_agg_shocks:
        Population = 16800
    else:
        Population = 14000
else:
    if do_agg_shocks:
        Population = 9600
    else:
        Population = 10000    # Total number of simulated agents in the population
T_sim_PY = 1200               # Number of periods to simulate (idiosyncratic shocks model, perpetual youth)
T_sim_LC = 1200               # Number of periods to simulate (idiosyncratic shocks model, lifecycle)
T_sim_agg_shocks = 1200       # Number of periods to simulate (aggregate shocks model)
ignore_periods_PY = 400       # Number of periods to throw out when looking at history (perpetual youth)
ignore_periods_LC = 400       # Number of periods to throw out when looking at history (lifecycle)
T_age = T_cycle + 1           # Don't let simulated agents survive beyond this age
pLvlInitMean_d = np.log(5)    # average initial permanent income, dropouts
pLvlInitMean_h = np.log(7.5)  # average initial permanent income, HS grads
pLvlInitMean_c = np.log(12)   # average initial permanent income, college grads
pLvlInitStd = 0.4             # Standard deviation of initial permanent income
aNrmInitMean = np.log(0.5)    # log initial wealth/income mean
aNrmInitStd  = 0.5            # log initial wealth/income standard deviation

# Set population macro parameters
PopGroFac = 1.01**(0.25)      # Population growth rate
PermGroFacAgg = 1.015**(0.25) # TFP growth rate
d_pct = 0.11                  # proportion of HS dropouts
h_pct = 0.55                  # proportion of HS graduates
c_pct = 0.34                  # proportion of college graduates
TypeWeight_lifecycle = [d_pct,h_pct,c_pct]

# Set indiividual parameters for the infinite horizon model
IndL = 10.0/9.0               # Labor supply per individual (constant)
PermGroFac_i = [1.000**0.25]  # Permanent income growth factor (no perm growth)
DiscFac_i = 0.97              # Default intertemporal discount factor
LivPrb_i = [1.0 - 1.0/160.0]  # Survival probability
PermShkStd_i = [(0.01*4/11)**0.5] # Standard deviation of permanent shocks to income
TranShkStd_i = [(0.01*4)**0.5]    # Standard deviation of transitory shocks to income

# Define the paths of permanent and transitory shocks (from Sabelhaus and Song)
TranShkStd = (np.concatenate((np.linspace(0.1,0.12,17), 0.12*np.ones(17), np.linspace(0.12,0.075,61), np.linspace(0.074,0.007,68), np.zeros(retired_T+1)))*4)**0.5
TranShkStd = np.ndarray.tolist(TranShkStd)
PermShkStd = np.concatenate((((0.00011342*(np.linspace(24,64.75,working_T-1)-47)**2 + 0.01)/(11.0/4.0))**0.5,np.zeros(retired_T+1)))
PermShkStd = np.ndarray.tolist(PermShkStd)

# Set aggregate parameters for the infinite horizon model
PermShkAggCount = 3                # Number of discrete permanent aggregate shocks
TranShkAggCount = 3                # Number of discrete transitory aggregate shocks
PermShkAggStd = np.sqrt(0.00004)   # Standard deviation of permanent aggregate shocks
TranShkAggStd = np.sqrt(0.00001)   # Standard deviation of transitory aggregate shocks
CapShare = 0.36                    # Capital's share of output
DeprFac = 0.025                    # Capital depreciation factor
CRRAPF = 1.0                       # CRRA in perfect foresight calibration
DiscFacPF = 0.99                   # Discount factor in perfect foresight calibration
slope_prev = 1.0                   # Initial slope of kNextFunc (aggregate shocks model)
intercept_prev = 0.0               # Initial intercept of kNextFunc (aggregate shocks model)

from pathlib import Path

# Import survival probabilities from SSA data
# The default is US SSA data from year 2010 in USactuarial.txt, but should the users wish to use data from other years
# or other countries, they could do so by naming the alternative data file "alternativedata.txt" and save this in the "/home/users/Downloads/" folder


README = Path("/home/users/Downloads/alternativedata.txt")
if README.is_file():
    print ("File does exist")
    data_location = os.path.dirname(os.path.abspath(__file__))
    f = open(data_location + '/' + 'alternativedata.txt','r')

else:
    print ("File does not exist")
    data_location = os.path.dirname(os.path.abspath(__file__))
    f = open(data_location + '/' + 'USactuarial.txt','r')

actuarial_reader = csv.reader(f,delimiter='\t')
raw_actuarial = list(actuarial_reader)
base_death_probs = []
for j in range(len(raw_actuarial)):
    base_death_probs += [float(raw_actuarial[j][4])] # This effectively assumes that everyone is a white woman
f.close

# Import adjustments for education and apply them to the base mortality rates
f = open(data_location + '/' + 'EducMortAdj.txt','r')
adjustment_reader = csv.reader(f,delimiter=' ')
raw_adjustments = list(adjustment_reader)
d_death_probs = []
h_death_probs = []
c_death_probs = []
for j in range(76):
    d_death_probs += [base_death_probs[j + init_age]*float(raw_adjustments[j][1])]
    h_death_probs += [base_death_probs[j + init_age]*float(raw_adjustments[j][2])]
    c_death_probs += [base_death_probs[j + init_age]*float(raw_adjustments[j][3])]
for j in range(76,96):
    d_death_probs += [base_death_probs[j + init_age]*float(raw_adjustments[75][1])]
    h_death_probs += [base_death_probs[j + init_age]*float(raw_adjustments[75][2])]
    c_death_probs += [base_death_probs[j + init_age]*float(raw_adjustments[75][3])]
LivPrb_d = []
LivPrb_h = []
LivPrb_c = []
for j in range(len(d_death_probs)): # Convert annual mortality rates to quarterly survival rates
    LivPrb_d += 4*[(1 - d_death_probs[j])**0.25]
    LivPrb_h += 4*[(1 - h_death_probs[j])**0.25]
    LivPrb_c += 4*[(1 - c_death_probs[j])**0.25]

# Define permanent income growth rates for each education level (from Cagetti 2003)
PermGroFac_d_base = [5.2522391e-002,  5.0039782e-002,  4.7586132e-002,  4.5162424e-002,  4.2769638e-002,  4.0408757e-002,  3.8080763e-002,  3.5786635e-002,  3.3527358e-002,  3.1303911e-002,  2.9117277e-002,  2.6968437e-002,  2.4858374e-002, 2.2788068e-002,  2.0758501e-002,  1.8770655e-002,  1.6825511e-002,  1.4924052e-002,  1.3067258e-002,  1.1256112e-002, 9.4915947e-003,  7.7746883e-003,  6.1063742e-003,  4.4876340e-003,  2.9194495e-003,  1.4028022e-003, -6.1326258e-005, -1.4719542e-003, -2.8280999e-003, -4.1287819e-003, -5.3730185e-003, -6.5598280e-003, -7.6882288e-003, -8.7572392e-003, -9.7658777e-003, -1.0713163e-002, -1.1598112e-002, -1.2419745e-002, -1.3177079e-002, -1.3869133e-002, -4.3985368e-001, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003]
PermGroFac_h_base = [4.1102173e-002,  4.1194381e-002,  4.1117402e-002,  4.0878307e-002,  4.0484168e-002,  3.9942056e-002,  3.9259042e-002,  3.8442198e-002,  3.7498596e-002,  3.6435308e-002,  3.5259403e-002,  3.3977955e-002,  3.2598035e-002,  3.1126713e-002,  2.9571062e-002,  2.7938153e-002,  2.6235058e-002,  2.4468848e-002,  2.2646594e-002,  2.0775369e-002,  1.8862243e-002,  1.6914288e-002,  1.4938576e-002,  1.2942178e-002,  1.0932165e-002,  8.9156095e-003,  6.8995825e-003,  4.8911556e-003,  2.8974003e-003,  9.2538802e-004, -1.0178097e-003, -2.9251214e-003, -4.7894755e-003, -6.6038005e-003, -8.3610250e-003, -1.0054077e-002, -1.1675886e-002, -1.3219380e-002, -1.4677487e-002, -1.6043137e-002, -5.5864350e-001, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002]
PermGroFac_c_base = [3.9375106e-002,  3.9030288e-002,  3.8601230e-002,  3.8091011e-002,  3.7502710e-002,  3.6839406e-002,  3.6104179e-002,  3.5300107e-002,  3.4430270e-002,  3.3497746e-002,  3.2505614e-002,  3.1456953e-002,  3.0354843e-002,  2.9202363e-002,  2.8002591e-002,  2.6758606e-002,  2.5473489e-002,  2.4150316e-002,  2.2792168e-002,  2.1402124e-002,  1.9983263e-002,  1.8538663e-002,  1.7071404e-002,  1.5584565e-002,  1.4081224e-002,  1.2564462e-002,  1.1037356e-002,  9.5029859e-003,  7.9644308e-003,  6.4247695e-003,  4.8870812e-003,  3.3544449e-003,  1.8299396e-003,  3.1664424e-004, -1.1823620e-003, -2.6640003e-003, -4.1251914e-003, -5.5628564e-003, -6.9739162e-003, -8.3552918e-003, -6.8938447e-001, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004]
PermGroFac_d_base += 31*[PermGroFac_d_base[-1]] # Add 31 years of the same permanent income growth rate to the end of the sequence
PermGroFac_h_base += 31*[PermGroFac_h_base[-1]]
PermGroFac_c_base += 31*[PermGroFac_c_base[-1]]
PermGroFac_d_retire = PermGroFac_d_base[40]     # Store the big shock to permanent income at retirement
PermGroFac_h_retire = PermGroFac_h_base[40]
PermGroFac_c_retire = PermGroFac_c_base[40]
PermGroFac_d_base[40] = PermGroFac_d_base[39]   # Overwrite the "retirement drop" with the adjacent growth rate
PermGroFac_h_base[40] = PermGroFac_h_base[39]
PermGroFac_c_base[40] = PermGroFac_c_base[39]
PermGroFac_d = []
PermGroFac_h = []
PermGroFac_c = []
for j in range(len(PermGroFac_d_base)):         # Make sequences of quarterly permanent income growth factors from annual permanent income growth rates
    PermGroFac_d += 4*[(1 + PermGroFac_d_base[j])**0.25]
    PermGroFac_h += 4*[(1 + PermGroFac_h_base[j])**0.25]
    PermGroFac_c += 4*[(1 + PermGroFac_c_base[j])**0.25]
PermGroFac_d[working_T-1] = 1 + PermGroFac_d_retire  # Put the big shock at retirement back into the sequence
PermGroFac_h[working_T-1] = 1 + PermGroFac_h_retire
PermGroFac_c[working_T-1] = 1 + PermGroFac_c_retire

# Import the SCF wealth data
f = open(data_location + '/' + SCF_data_file,'r')
SCF_reader = csv.reader(f,delimiter='\t')
SCF_raw = list(SCF_reader)
SCF_wealth = np.zeros(len(SCF_raw)) + np.nan
SCF_weights = deepcopy(SCF_wealth)
for j in range(len(SCF_raw)):
    SCF_wealth[j] = float(SCF_raw[j][0])
    SCF_weights[j] = float(SCF_raw[j][1])


# Make dictionaries for lifecycle consumer types
init_dropout = {"CRRA":CRRA,
                "Rfree":Rfree,
                "PermGroFac":PermGroFac_d,
                "PermGroFacAgg":PermGroFacAgg,
                "BoroCnstArt":BoroCnstArt,
                "CubicBool":CubicBool,
                "vFuncBool":vFuncBool,
                "PermShkStd":PermShkStd,
                "PermShkCount":PermShkCount,
                "TranShkStd":TranShkStd,
                "TranShkCount":TranShkCount,
                "T_cycle":T_cycle,
                "UnempPrb":UnempPrb,
                "UnempPrbRet":UnempPrbRet,
                "T_retire":working_T-1,
                "IncUnemp":IncUnemp,
                "IncUnempRet":IncUnempRet,
                "aXtraMin":aXtraMin,
                "aXtraMax":aXtraMax,
                "aXtraCount":aXtraCount,
                "aXtraExtra":[],
                "aXtraNestFac":aXtraNestFac,
                "LivPrb":LivPrb_d,
                "DiscFac":DiscFac_guess, # dummy value, will be overwritten
                'AgentCount': 0, # this is overwritten by parameter distributor
                'T_sim':T_sim_LC,
                'T_age':T_age,
                'aNrmInitMean':aNrmInitMean,
                'aNrmInitStd':aNrmInitStd,
                'pLvlInitMean':pLvlInitMean_d,
                'pLvlInitStd':pLvlInitStd
                }
adj_highschool = {"PermGroFac":PermGroFac_h,"LivPrb":LivPrb_h,'pLvlInitMean':pLvlInitMean_h}
adj_college = {"PermGroFac":PermGroFac_c,"LivPrb":LivPrb_c,'pLvlInitMean':pLvlInitMean_c}

# Make a dictionary for the infinite horizon type
init_infinite = {"CRRA":CRRA,
                "Rfree":1.01/LivPrb_i[0],
                "PermGroFac":PermGroFac_i,
                "PermGroFacAgg":1.0,
                "BoroCnstArt":BoroCnstArt,
                "CubicBool":CubicBool,
                "vFuncBool":vFuncBool,
                "PermShkStd":PermShkStd_i,
                "PermShkCount":PermShkCount,
                "TranShkStd":TranShkStd_i,
                "TranShkCount":TranShkCount,
                "UnempPrb":UnempPrb,
                "IncUnemp":IncUnemp,
                "UnempPrbRet":None,
                "IncUnempRet":None,
                "aXtraMin":aXtraMin,
                "aXtraMax":aXtraMax,
                "aXtraCount":aXtraCount,
                "aXtraExtra":[None],
                "aXtraNestFac":aXtraNestFac,
                "LivPrb":LivPrb_i,
                "DiscFac":DiscFac_i, # dummy value, will be overwritten
                "cycles":0,
                "T_cycle":1,
                "T_retire":0,
                'T_sim':T_sim_PY,
                'T_age': 400,
                'IndL': IndL,
                'aNrmInitMean':np.log(0.00001),
                'aNrmInitStd':0.0,
                'pLvlInitMean':0.0,
                'pLvlInitStd':0.0,
                'AgentCount':0, # will be overwritten by parameter distributor
                }

# Make a base dictionary for the cstwMPCmarket
init_market = {'LorenzBool': False,
               'ManyStatsBool': False,
               'ignore_periods':0,    # Will get overwritten
               'PopGroFac':0.0,       # Will get overwritten
               'T_retire':0,          # Will get overwritten
               'TypeWeights':[],      # Will get overwritten
               'Population':Population,
               'act_T':0,             # Will get overwritten
               'IncUnemp':IncUnemp,
               'cutoffs':[(0.99,1),(0.9,1),(0.8,1),(0.6,0.8),(0.4,0.6),(0.2,0.4),(0.0,0.2)],
               'LorenzPercentiles':percentiles_to_match,
               'AggShockBool':do_agg_shocks
               }


# Make a dictionary for the aggregate shocks type
init_agg_shocks = deepcopy(init_infinite)
init_agg_shocks['T_sim'] = T_sim_agg_shocks
init_agg_shocks['tolerance'] = 0.0001
init_agg_shocks['MgridBase'] = np.array([0.1,0.3,0.6,0.8,0.9,0.98,1.0,1.02,1.1,1.2,1.6,2.0,3.0])

# Make a dictionary for the aggrege shocks market
aggregate_params = {'PermShkAggCount': PermShkAggCount,
                    'TranShkAggCount': TranShkAggCount,
                    'PermShkAggStd': PermShkAggStd,
                    'TranShkAggStd': TranShkAggStd,
                    'DeprFac': DeprFac,
                    'CapShare': CapShare,
                    'AggregateL':(1.0-UnempPrb)*IndL,
                    'CRRA': CRRAPF,
                    'DiscFac': DiscFacPF,
                    'LivPrb': LivPrb_i[0],
                    'slope_prev': slope_prev,
                    'intercept_prev': intercept_prev,
                    'act_T':T_sim_agg_shocks,
                    'ignore_periods':200,
                    'tolerance':0.0001
                    }

def main():
    print("Sorry, SetupParamsCSTWnew doesn't actually do anything on its own.")
    print("This module is imported by cstwMPCnew, providing data and calibrated")
    print("parameters for the various estimations.  Please see that module if")
    print("you want more interesting output.")

if __name__ == '__main__':
    main()


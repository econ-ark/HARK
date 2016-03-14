'''
This package load parameters used in the cstwMPC estimations.
'''
import numpy as np
import csv
from copy import copy, deepcopy

# Choose percentiles of the data to match and which estimation to run
do_lifecycle = False          # Use lifecycle model if True, perpetual youth if False
do_beta_dist = False           # Do beta-dist version if True, beta-point if False
run_estimation = True         # Runs the estimation if True
find_beta_vs_KY = False       # Computes K/Y ratio for a wide range of beta; should have do_beta_dist = False
do_sensitivity = [False, False, False, False, False, False, False, False] # Choose which sensitivity analyses to run: rho, xi_sigma, psi_sigma, mu, urate, mortality, g, R
do_liquid = False             # Matches liquid assets data when True, net worth data when False
do_tractable = False          # Uses a "tractable consumer" rather than solving full model when True
SCF_data_file = 'SCFwealthDataReduced.txt'
percentiles_to_match = [0.2, 0.4, 0.6, 0.8]    # Which points of the Lorenz curve to match in beta-dist (must be in (0,1))
#percentiles_to_match = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
if do_beta_dist:
    pref_type_count = 7       # Number of discrete beta types in beta-dist
else:
    pref_type_count = 1       # Just one beta type in beta-point

# Set basic parameters for the lifecycle micro model
init_age = 24                 # Starting age for agents
R = 1.04**(0.25)              # Quarterly interest factor
working_T = 41*4              # Number of working periods
retired_T = 55*4              # Number of retired periods
total_T = working_T+retired_T # Total number of periods
rho = 1.0                     # Coefficient of relative risk aversion   
beta_guess = 0.99             # Initial starting point for discount factor
p_unemploy = 0.07             # Probability of unemployment while working
p_unemploy_retire = 0.0005    # Probabulity of "unemployment" while retired
income_unemploy = 0.15        # Unemployment benefit replacement rate
income_unemploy_retire = 0.0  # Ditto when retired
Y0_sigma = 0.4                # Standard deviation of initial permanent income
borrowing_constraint = 0.0

# Set grid sizes
psi_N = 5                     # Number of points in permanent income shock grid
xi_N = 5                      # Number of points in transitory income shock grid
a_min = 0.00001                 # Minimum end-of-period assets in grid
a_max = 20                    # Maximum end-of-period assets in grid
a_size = 20                    # Number of points in assets grid
exp_nest = 3                  # Number of times to 'exponentially nest' when constructing assets grid
sim_pop_size = 2000           # Number of simulated agents per preference type
cubic_splines = False          # Whether to use cubic spline interpolation
calc_vFunc = False            # Whether to calculate the value function during solution

# Set random seeds
perm_seed = 31382             # Seed for permanent income shocks
temp_seed = 112883            # Seed for transitory income shocks
unemp_seed = 92615            # Seed for unemployment shocks
w0_seed = 138                 # Seed for initial wealth draws
Y0_seed = 666                 # Seed for initial permanent income draws

# Define the paths of permanent and transitory shocks (from Sabelhaus and Song)
xi_sigma = (np.concatenate((np.linspace(0.1,0.12,17), 0.12*np.ones(17), np.linspace(0.12,0.075,61), np.linspace(0.074,0.007,68), np.zeros(retired_T+1)))*4)**0.5
xi_sigma = np.ndarray.tolist(xi_sigma)
psi_sigma = np.concatenate((((0.00011342*(np.linspace(24,64.75,working_T-1)-47)**2 + 0.01)/(11.0/4.0))**0.5,np.zeros(retired_T+1)))
psi_sigma = np.ndarray.tolist(psi_sigma)

# Import survival probabilities from SSA data
f = open('USactuarial.txt','r')
actuarial_reader = csv.reader(f,delimiter='\t')
raw_actuarial = list(actuarial_reader)
base_death_probs = []
for j in range(len(raw_actuarial)):
    base_death_probs += [float(raw_actuarial[j][4])]
f.close
f = open('EducMortAdj.txt','r')
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
survival_probs_d = []
survival_probs_h = []
survival_probs_c = []
for j in range(len(d_death_probs)):
    survival_probs_d += 4*[(1 - d_death_probs[j])**0.25]
    survival_probs_h += 4*[(1 - h_death_probs[j])**0.25]
    survival_probs_c += 4*[(1 - c_death_probs[j])**0.25]

# Define permanent income growth rates for each education level
gamma_d_base = [5.2522391e-002,  5.0039782e-002,  4.7586132e-002,  4.5162424e-002,  4.2769638e-002,  4.0408757e-002,  3.8080763e-002,  3.5786635e-002,  3.3527358e-002,  3.1303911e-002,  2.9117277e-002,  2.6968437e-002,  2.4858374e-002, 2.2788068e-002,  2.0758501e-002,  1.8770655e-002,  1.6825511e-002,  1.4924052e-002,  1.3067258e-002,  1.1256112e-002, 9.4915947e-003,  7.7746883e-003,  6.1063742e-003,  4.4876340e-003,  2.9194495e-003,  1.4028022e-003, -6.1326258e-005, -1.4719542e-003, -2.8280999e-003, -4.1287819e-003, -5.3730185e-003, -6.5598280e-003, -7.6882288e-003, -8.7572392e-003, -9.7658777e-003, -1.0713163e-002, -1.1598112e-002, -1.2419745e-002, -1.3177079e-002, -1.3869133e-002, -4.3985368e-001, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003]
gamma_h_base = [4.1102173e-002,  4.1194381e-002,  4.1117402e-002,  4.0878307e-002,  4.0484168e-002,  3.9942056e-002,  3.9259042e-002,  3.8442198e-002,  3.7498596e-002,  3.6435308e-002,  3.5259403e-002,  3.3977955e-002,  3.2598035e-002,  3.1126713e-002,  2.9571062e-002,  2.7938153e-002,  2.6235058e-002,  2.4468848e-002,  2.2646594e-002,  2.0775369e-002,  1.8862243e-002,  1.6914288e-002,  1.4938576e-002,  1.2942178e-002,  1.0932165e-002,  8.9156095e-003,  6.8995825e-003,  4.8911556e-003,  2.8974003e-003,  9.2538802e-004, -1.0178097e-003, -2.9251214e-003, -4.7894755e-003, -6.6038005e-003, -8.3610250e-003, -1.0054077e-002, -1.1675886e-002, -1.3219380e-002, -1.4677487e-002, -1.6043137e-002, -5.5864350e-001, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002]
gamma_c_base = [3.9375106e-002,  3.9030288e-002,  3.8601230e-002,  3.8091011e-002,  3.7502710e-002,  3.6839406e-002,  3.6104179e-002,  3.5300107e-002,  3.4430270e-002,  3.3497746e-002,  3.2505614e-002,  3.1456953e-002,  3.0354843e-002,  2.9202363e-002,  2.8002591e-002,  2.6758606e-002,  2.5473489e-002,  2.4150316e-002,  2.2792168e-002,  2.1402124e-002,  1.9983263e-002,  1.8538663e-002,  1.7071404e-002,  1.5584565e-002,  1.4081224e-002,  1.2564462e-002,  1.1037356e-002,  9.5029859e-003,  7.9644308e-003,  6.4247695e-003,  4.8870812e-003,  3.3544449e-003,  1.8299396e-003,  3.1664424e-004, -1.1823620e-003, -2.6640003e-003, -4.1251914e-003, -5.5628564e-003, -6.9739162e-003, -8.3552918e-003, -6.8938447e-001, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004]
gamma_d_base += 31*[gamma_d_base[-1]]
gamma_h_base += 31*[gamma_h_base[-1]]
gamma_c_base += 31*[gamma_c_base[-1]]
gamma_d_retire = gamma_d_base[40]
gamma_h_retire = gamma_h_base[40]
gamma_c_retire = gamma_c_base[40]
gamma_d_base[40] = gamma_d_base[39]
gamma_h_base[40] = gamma_h_base[39]
gamma_c_base[40] = gamma_c_base[39]
Gamma_d = []
Gamma_h = []
Gamma_c = []
for j in range(len(gamma_d_base)):
    Gamma_d += 4*[(1 + gamma_d_base[j])**0.25]
    Gamma_h += 4*[(1 + gamma_h_base[j])**0.25]
    Gamma_c += 4*[(1 + gamma_c_base[j])**0.25]
Gamma_d[working_T-1] = 1 + gamma_d_retire
Gamma_h[working_T-1] = 1 + gamma_h_retire
Gamma_c[working_T-1] = 1 + gamma_c_retire

# Set population macro parameters
pop_growth = 1.01**(0.25)      # population growth rate
econ_growth = 1.015**(0.25)    # TFP growth rate
d_pct = 0.11                   # proportion of HS dropouts
h_pct = 0.55                   # proportion of HS graduates
c_pct = 0.34                   # proportion of college graduates
Y0_d = 5                       # average initial permanent income, dropouts
Y0_h = 7.5                     # average initial permanent income, HS grads
Y0_c = 12                      # average initial permanent income, college grads
w0_values = [0.17, 0.5, 0.83]  # initial wealth/income ratio values
w0_probs = [1.0/3.0, 1.0/3.0, 1.0/3.0] # ...and probabilities 

# Calculate the social security tax rate for the economy
d_income = np.concatenate((np.array([1]),np.cumprod(Gamma_d)))*Y0_d
h_income = np.concatenate((np.array([1]),np.cumprod(Gamma_h)))*Y0_h
c_income = np.concatenate((np.array([1]),np.cumprod(Gamma_c)))*Y0_c
cohort_weight = pop_growth**np.array(np.arange(0,-(total_T+1),-1))
econ_weight = econ_growth**np.array(np.arange(0,-(total_T+1),-1))
d_survival_cum = np.concatenate((np.array([1]),np.cumprod(survival_probs_d)))
h_survival_cum = np.concatenate((np.array([1]),np.cumprod(survival_probs_h)))
c_survival_cum = np.concatenate((np.array([1]),np.cumprod(survival_probs_c)))
total_income_working = (d_pct*d_income[0:working_T]*d_survival_cum[0:working_T] + h_pct*h_income[0:working_T]*h_survival_cum[0:working_T] + c_pct*c_income[0:working_T]*c_survival_cum[0:working_T])*cohort_weight[0:working_T]*econ_weight[0:working_T]
total_income_retired = (d_pct*d_income[working_T:total_T]*d_survival_cum[working_T:total_T] + h_pct*h_income[working_T:total_T]*h_survival_cum[working_T:total_T] + c_pct*c_income[working_T:total_T]*c_survival_cum[working_T:total_T])*cohort_weight[working_T:total_T]*econ_weight[working_T:total_T]
tax_rate_SS = np.sum(total_income_retired)/np.sum(total_income_working)
tax_rate_U = p_unemploy*income_unemploy
tax_rate = tax_rate_SS + tax_rate_U

# Generate normalized weighting vectors for each age and education level
age_size_d = d_pct*cohort_weight*d_survival_cum
age_size_h = h_pct*cohort_weight*h_survival_cum
age_size_c = c_pct*cohort_weight*c_survival_cum
total_pop_size = sum(age_size_d) + sum(age_size_h) + sum(age_size_c)
age_weight_d = age_size_d/total_pop_size
age_weight_h = age_size_h/total_pop_size
age_weight_c = age_size_c/total_pop_size
age_weight_all = np.concatenate((age_weight_d,age_weight_h,age_weight_c))
age_weight_short = np.concatenate((age_weight_d[0:total_T],age_weight_h[0:total_T],age_weight_c[0:total_T]))
total_output = np.sum(total_income_working)/total_pop_size

# Set indiividual parameters for the infinite horizon model
l_bar = 10.0/9.0
Gamma_i = [1.000**0.25]
beta_i = 0.99
survival_prob_i = [1.0 - 1.0/160.0]
if do_liquid:
    psi_sigma_i = [(0.01*4/16)**0.5]
else:
    psi_sigma_i = [(0.01*4/11)**0.5]
xi_sigma_i = [(0.01*4)**0.5]
sim_periods = 1000
age_weight_i = survival_prob_i**np.arange(0,sim_periods+1,dtype=float)
total_pop_size_i = np.sum(age_weight_i)
age_weight_i = age_weight_i/total_pop_size_i
if not do_lifecycle:
    age_weight_all = age_weight_i
    age_weight_short = age_weight_i[0:sim_periods]
    total_output = l_bar

# Import the SCF wealth data
f = open(SCF_data_file,'r')
SCF_reader = csv.reader(f,delimiter='\t')
SCF_raw = list(SCF_reader)
SCF_wealth = np.zeros(len(SCF_raw)) + np.nan
SCF_weights = deepcopy(SCF_wealth)
for j in range(len(SCF_raw)):
    SCF_wealth[j] = float(SCF_raw[j][0])
    SCF_weights[j] = float(SCF_raw[j][1])


# Make dictionaries for lifecycle consumer types
init_dropout = {"rho":rho,
                "R":R,
                "Gamma":Gamma_d,
                "constraint":borrowing_constraint,
                "cubic_splines":cubic_splines,
                "calc_vFunc":calc_vFunc,
                "psi_sigma":psi_sigma,
                "psi_N":psi_N,
                "xi_sigma":xi_sigma,
                "xi_N":xi_N,
                "T_total":total_T,
                "p_unemploy":p_unemploy,
                "p_unemploy_retire":p_unemploy_retire,
                "T_retire":working_T-1,
                "income_unemploy":income_unemploy,
                "income_unemploy_retire":income_unemploy_retire,
                "a_min":a_min,
                "a_max":a_max,
                "a_size":a_size,
                "a_extra":[],
                "exp_nest":exp_nest,
                "survival_prob":survival_probs_d,
                "beta":beta_guess, # dummy value, will be overwritten
                "tax_rate":tax_rate_SS, # for math reasons, only SS tax goes here
                'Nagents':sim_pop_size,
                'psi_seed':perm_seed,
                'xi_seed':temp_seed,
                'unemp_seed':unemp_seed,
                'sim_periods':total_T,
                }
init_highschool = copy(init_dropout)
init_highschool["Gamma"] = Gamma_h
init_highschool["survival_prob"] = survival_probs_h
adj_highschool = {"Gamma":Gamma_h,"survival_prob":survival_probs_h}
init_college = copy(init_dropout)
init_college["Gamma"] = Gamma_c
init_college["survival_prob"] = survival_probs_c
adj_college = {"Gamma":Gamma_c,"survival_prob":survival_probs_c}

# Make a dictionary for the infinite horizon type
init_infinite = {"rho":rho,
                "R":1.01/survival_prob_i[0],
                "Gamma":Gamma_i,
                "constraint":borrowing_constraint,
                "cubic_splines":cubic_splines,
                "calc_vFunc":calc_vFunc,
                "psi_sigma":psi_sigma_i,
                "psi_N":psi_N,
                "xi_sigma":xi_sigma_i,
                "xi_N":xi_N,
                "p_unemploy":p_unemploy,
                "income_unemploy":income_unemploy,
                "p_unemploy_retire":None,
                "income_unemploy_retire":None,
                "a_min":a_min,
                "a_max":a_max,
                "a_size":a_size,
                "a_extra":[9000.0],
                "exp_nest":exp_nest,
                "survival_prob":survival_prob_i,
                "beta":beta_i, # dummy value, will be overwritten
                "cycles":0,
                "T_total":1,
                "T_retire":0,
                "tax_rate":0.0,
                'sim_periods':sim_periods,
                'Nagents':sim_pop_size,
                'psi_seed':perm_seed,
                'xi_seed':temp_seed,
                'unemp_seed':unemp_seed,
                'l_bar':l_bar,
                }

# Make dictionaries for constructing income shocks
make_shocks_dropout = {'psi_sigma':psi_sigma,
                      'xi_sigma':xi_sigma,
                      'Gamma':Gamma_d,
                      'R':R,
                      'p_unemploy':p_unemploy,
                      'p_unemploy_retire':p_unemploy_retire,
                      'income_unemploy':income_unemploy,
                      'income_unemploy_retire':income_unemploy_retire,
                      'T_retire':retired_T+1,
                      'Nagents':sim_pop_size,
                      'psi_seed':perm_seed,
                      'xi_seed':temp_seed,
                      'unemp_seed':unemp_seed,
                      'tax_rate':tax_rate_SS
                      }
make_shocks_highschool = copy(make_shocks_dropout)
make_shocks_highschool['Gamma'] = Gamma_h
make_shocks_college = copy(make_shocks_dropout)
make_shocks_college['Gamma'] = Gamma_c
make_shocks_infinite = {'psi_sigma':psi_sigma_i[0],
                        'xi_sigma':xi_sigma_i[0],
                        'Gamma':Gamma_i[0],
                        'R':1.01/survival_prob_i[0],
                        'p_unemploy':p_unemploy,
                        'income_unemploy':income_unemploy,
                        'Nagents':sim_pop_size,
                        'psi_seed':perm_seed,
                        'xi_seed':temp_seed,
                        'unemp_seed':unemp_seed,
                        'sim_periods':sim_periods
                        }

beta_save = beta_guess
diff_save = 1000000.0
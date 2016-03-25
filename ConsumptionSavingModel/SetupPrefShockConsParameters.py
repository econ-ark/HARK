'''
Provides default values for ConsPrefShockModel.
'''
from numpy.random import RandomState

exp_nest = 3                        # Number of times to "exponentially nest" when constructing a_grid
a_min = 0.001                       # Minimum end-of-period assets value in a_grid
a_max = 100                         # Maximum end-of-period assets value in a_grid                  
a_huge = None                       # A very large value of assets to add to the grid, not used
a_extra = None                      # Some other value of assets to add to the grid, not used
a_size = 64                        # Number of points in the grid of assets

constraint = None                   # Artificial borrowing constraint
R_borrow = 1.10                     # Interest factor on assets when negative
R_save = 1.03                       # Interest factor on assets when positive

psi_N = 15                           # Number of points in discrete approximation to permanent income shocks
xi_N = 15                           # Number of points in discrete approximation to transitory income shocks
psi_sigma = [0.10]                   # Standard deviation of permanent income shocks
xi_sigma = [0.10]                    # Standard deviation of transitory income shocks

p_unemploy = 0.05                  # Probability of unemployment while working
p_unemploy_retire = 0.0005          # Probability of "unemployment" while retired
income_unemploy = 0.4               # Unemployment benefits replacement rate
income_unemploy_retire = 0.0        # Ditto when retired

pref_shock_sigma = 0.2              # Standard deviation of marginal utility shocks
pref_shock_N = 24                   # Number of points in discrete approximation to preference shock dist

TT = 1                              # Total number of periods in the model
T_retire = 0                        # Turn off retirement

sim_T = 200                         # Number of periods to simulate
sim_N = 1000                        # Number of agents to simulate
seed = 31382                        # Any old integer
RNG = RandomState(seed)             # A random number generator

rho = 4.0                          # Coefficient of relative risk aversion
beta = 0.96                         # Time preference discount factor
Gamma = [1.01]                      # Timepath of expected permanent income growth
survival_prob = [0.98]              # Timepath of survival probabilities

# Dictionary that can be passed to PrefShockConsumer to instantiate
init_consumer_objects = {"rho":rho,
                        "R_borrow":R_borrow,
                        "R_save":R_save,
                        "Gamma":Gamma,
                        "constraint":constraint,
                        "psi_sigma":psi_sigma,
                        "psi_N":psi_N,
                        "xi_sigma":xi_sigma,
                        "xi_N":xi_N,
                        "pref_shock_sigma":pref_shock_sigma,
                        "pref_shock_N":pref_shock_N,
                        "T_total":TT,
                        "p_unemploy":p_unemploy,
                        "p_unemploy_retire":p_unemploy_retire,
                        "T_retire":T_retire,
                        "income_unemploy":income_unemploy,
                        "income_unemploy_retire":income_unemploy_retire,
                        "a_min":a_min,
                        "a_max":a_max,
                        "a_size":a_size,
                        "a_extra":[a_extra,a_huge],
                        "exp_nest":exp_nest,
                        "survival_prob":survival_prob,
                        "beta":beta,
                        "sim_T":sim_T,
                        "sim_N":sim_N,
                        "RNG":RNG
                        }
'''
Provides default values for ConsPrefShockModel.
'''
from numpy.random import RandomState

exp_nest = 3                        # Number of times to "exponentially nest" when constructing a_grid
aDispMin = 0.001                       # Minimum end-of-period assets value in a_grid
aDispMax = 100                         # Maximum end-of-period assets value in a_grid                  
aDispHuge = None                       # A very large value of assets to add to the grid, not used
aDispExtra = None                      # Some other value of assets to add to the grid, not used
aDispCount = 64                        # Number of points in the grid of assets

constraint = None                   # Artificial borrowing constraint
R_borrow = 1.20                     # Interest factor on assets when negative
R_save = 1.03                       # Interest factor on assets when positive

PermShkCount = 15                           # Number of points in discrete approximation to permanent income shocks
TranShkCount = 15                           # Number of points in discrete approximation to transitory income shocks
PermShkStd = [0.1]                   # Standard deviation of permanent income shocks
TranShkStd = [0.1]                    # Standard deviation of transitory income shocks

UnempPrb = 0.05                  # Probability of unemployment while working
UnempPrbRet = 0.0005          # Probability of "unemployment" while retired
IncUnemp = 0.4               # Unemployment benefits replacement rate
IncUnempRet = 0.0        # Ditto when retired

pref_shock_sigma = 0.30              # Standard deviation of marginal utility shocks
pref_shock_N = 12                    # Number of points in discrete approximation to preference shock dist
pref_shock_tail_N = 4                # Number of points in each tail of the preference shock distribution

TT = 1                              # Total number of periods in the model
T_retire = 0                        # Turn off retirement

sim_T = 500                         # Number of periods to simulate
sim_N = 10000                       # Number of agents to simulate
seed = 31382                        # Any old integer
RNG = RandomState(seed)             # A random number generator

rho = 3.0                          # Coefficient of relative risk aversion
beta = 0.96                         # Time preference discount factor
Gamma = [1.02]                      # Timepath of expected permanent income growth
survival_prob = [0.98]              # Timepath of survival probabilities

# Dictionary that can be passed to PrefShockConsumer to instantiate
init_consumer_objects = {"rho":rho,
                        "R_borrow":R_borrow,
                        "R_save":R_save,
                        "Gamma":Gamma,
                        "constraint":constraint,
                        "PermShkStd":PermShkStd,
                        "PermShkCount":PermShkCount,
                        "TranShkStd":TranShkStd,
                        "TranShkCount":TranShkCount,
                        "pref_shock_sigma":pref_shock_sigma,
                        "pref_shock_N":pref_shock_N,
                        "pref_shock_tail_N":pref_shock_tail_N,
                        "T_total":TT,
                        "UnempPrb":UnempPrb,
                        "UnempPrbRet":UnempPrb,
                        "T_retire":T_retire,
                        "IncUnemp":IncUnemp,
                        "IncUnempRet":IncUnempRet,
                        "aDispMin":aDispMin,
                        "aDispMax":aDispMax,
                        "aDispCount":aDispCount,
                        "aDispExtra":[aDispExtra,aDispHuge],
                        "exp_nest":exp_nest,
                        "survival_prob":survival_prob,
                        "beta":beta,
                        "sim_T":sim_T,
                        "sim_N":sim_N,
                        "RNG":RNG
                        }
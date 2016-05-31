'''
Provides default values for ConsPrefShockModel.
'''
from numpy.random import RandomState

exp_nest = 3                        # Number of times to "exponentially nest" when constructing a_grid
aXtraMin = 0.001                       # Minimum end-of-period assets value in a_grid
aXtraMax = 100                         # Maximum end-of-period assets value in a_grid                  
aXtraHuge = None                       # A very large value of assets to add to the grid, not used
aXtraExtra = None                      # Some other value of assets to add to the grid, not used
aXtraCount = 64                        # Number of points in the grid of assets

BoroCnstArt = None                   # Artificial borrowing constraint
Rboro = 1.20                     # Interest factor on assets when negative
Rsave = 1.03                       # Interest factor on assets when positive

PermShkCount = 7                           # Number of points in discrete approximation to permanent income shocks
TranShkCount = 7                           # Number of points in discrete approximation to transitory income shocks
PermShkStd = [0.1]                   # Standard deviation of permanent income shocks
TranShkStd = [0.1]                    # Standard deviation of transitory income shocks

UnempPrb = 0.05                  # Probability of unemployment while working
UnempPrbRet = 0.0005          # Probability of "unemployment" while retired
IncUnemp = 0.3               # Unemployment benefits replacement rate
IncUnempRet = 0.0        # Ditto when retired

PrefShkStd = 0.30              # Standard deviation of marginal utility shocks
PrefShk_N = 12                    # Number of points in discrete approximation to preference shock dist
PrefShk_tail_N = 4                # Number of points in each tail of the preference shock distribution

TT = 1                              # Total number of periods in the model
T_retire = 0                        # Turn off retirement

sim_T = 500                         # Number of periods to simulate
sim_N = 10000                       # Number of agents to simulate
seed = 31382                        # Any old integer
RNG = RandomState(seed)             # A random number generator

CRRA = 4.0                          # Coefficient of relative risk aversion
DiscFac = 0.96                         # Time preference discount factor
PermGroFac = [1.02]                      # Timepath of expected permanent income growth
LivPrb = [0.98]                     # Timepath of survival probabilities

# Dictionary that can be passed to PrefShockConsumer to instantiate
init_consumer_objects = {"CRRA":CRRA,
                        "Rboro":Rboro,
                        "Rsave":Rsave,
                        "PermGroFac":PermGroFac,
                        "BoroCnstArt":BoroCnstArt,
                        "PermShkStd":PermShkStd,
                        "PermShkCount":PermShkCount,
                        "TranShkStd":TranShkStd,
                        "TranShkCount":TranShkCount,
                        "PrefShkStd":PrefShkStd,
                        "PrefShk_N":PrefShk_N,
                        "PrefShk_tail_N":PrefShk_tail_N,
                        "T_total":TT,
                        "UnempPrb":UnempPrb,
                        "UnempPrbRet":UnempPrb,
                        "T_retire":T_retire,
                        "IncUnemp":IncUnemp,
                        "IncUnempRet":IncUnempRet,
                        "aXtraMin":aXtraMin,
                        "aXtraMax":aXtraMax,
                        "aXtraCount":aXtraCount,
                        "aXtraExtra":[aXtraExtra,aXtraHuge],
                        "exp_nest":exp_nest,
                        "LivPrb":LivPrb,
                        "DiscFac":DiscFac,
                        "sim_T":sim_T,
                        "sim_N":sim_N,
                        "RNG":RNG
                        }
'''
Set if parameters for the first journey
'''
from copy import copy
import numpy as np

# -----------------------------------------------------------------------------
# --- Define all of the parameters for the perfect foresight model ------------
# -----------------------------------------------------------------------------

CRRA = 2.0                          # Coefficient of relative risk aversion
Rfree = 1.03                        # Interest factor on assets
DiscFac = 0.96                      # Intertemporal discount factor
LivPrb = [1.0]                     # Survival probability
PermGroFac = [1.0]                 # Permanent income growth factor
# Number of agents of this type (only matters for simulation)
AgentCount = 10000
# Mean of log initial assets (only matters for simulation)
aNrmInitMean = 0.0
# Standard deviation of log initial assets (only for simulation)
aNrmInitStd = 1.0
# Mean of log initial permanent income (only matters for simulation)
pLvlInitMean = 0.0
# Standard deviation of log initial permanent income (only matters for simulation)
pLvlInitStd = 0.0
# Aggregate permanent income growth factor (only matters for simulation)
PermGroFacAgg = 1.0
# Age after which simulated agents are automatically killed
T_age = None
T_cycle = 1                         # Number of periods in the cycle for this agent type

# Make a dictionary to specify a perfect foresight consumer type
init_perfect_foresight = {'CRRA': CRRA,
                          'Rfree': Rfree,
                          'DiscFac': DiscFac,
                          'LivPrb': LivPrb,
                          'PermGroFac': PermGroFac,
                          'AgentCount': AgentCount,
                          'aNrmInitMean': aNrmInitMean,
                          'aNrmInitStd': aNrmInitStd,
                          'pLvlInitMean': pLvlInitMean,
                          'pLvlInitStd': pLvlInitStd,
                          'PermGroFacAgg': PermGroFacAgg,
                          'T_age': T_age,
                          'T_cycle': T_cycle
                          }

# -----------------------------------------------------------------------------
# --- Define additional parameters for the idiosyncratic shocks model ---------
# -----------------------------------------------------------------------------

# Parameters for constructing the "assets above minimum" grid
aXtraMin = 0.001                    # Minimum end-of-period "assets above minimum" value
aXtraMax = 20                       # Maximum end-of-period "assets above minimum" value
# aXtraExtra = [None]                   # Some other value of "assets above minimum" to add to the grid, not used
# Exponential nesting factor when constructing "assets above minimum" grid
aXtraNestFac = 3
# Number of points in the grid of "assets above minimum"
aXtraCount = 48

# Parameters describing the income process
# Number of points in discrete approximation to permanent income shocks
PermShkCount = 7
# Number of points in discrete approximation to transitory income shocks
TranShkCount = 7
# Standard deviation of log permanent income shocks
PermShkStd = [0.1]
# Standard deviation of log transitory income shocks
TranShkStd = [0.2]
UnempPrb = 0.005                     # Probability of unemployment while working
UnempPrbRet = 0.005                 # Probability of "unemployment" while retired
IncUnemp = 0.3                      # Unemployment benefits replacement rate
IncUnempRet = 0.0                   # "Unemployment" benefits when retired
tax_rate = 0.0                      # Flat income tax rate
# Period of retirement (0 --> no retirement)
T_retire = 0

# A few other parameters
# Artificial borrowing constraint; imposed minimum level of end-of period assets
BoroCnstArt = 0.0
# Use cubic spline interpolation when True, linear interpolation when False
CubicBool = True
# Whether to calculate the value function during solution
vFuncBool = False

# Make a dictionary to specify an idiosyncratic income shocks consumer
init_idiosyncratic_shocks = {'CRRA': CRRA,
                             'Rfree': Rfree,
                             'DiscFac': DiscFac,
                             'LivPrb': LivPrb,
                             'PermGroFac': PermGroFac,
                             'AgentCount': AgentCount,
                             'aXtraMin': aXtraMin,
                             'aXtraMax': aXtraMax,
                             'aXtraNestFac': aXtraNestFac,
                             'aXtraCount': aXtraCount,
                             # 'aXtraExtra': [aXtraExtra],
                             'PermShkStd': PermShkStd,
                             'PermShkCount': PermShkCount,
                             'TranShkStd': TranShkStd,
                             'TranShkCount': TranShkCount,
                             'UnempPrb': UnempPrb,
                             'UnempPrbRet': UnempPrbRet,
                             'IncUnemp': IncUnemp,
                             'IncUnempRet': IncUnempRet,
                             'BoroCnstArt': BoroCnstArt,
                             'tax_rate': 0.0,
                             'vFuncBool': vFuncBool,
                             'CubicBool': CubicBool,
                             'T_retire': T_retire,
                             'aNrmInitMean': aNrmInitMean,
                             'aNrmInitStd': aNrmInitStd,
                             'pLvlInitMean': pLvlInitMean,
                             'pLvlInitStd': pLvlInitStd,
                             'PermGroFacAgg': PermGroFacAgg,
                             'T_age': T_age,
                             'T_cycle': T_cycle
                             }

# Make a dictionary to specify a lifecycle consumer with a finite horizon

# -----------------------------------------------------------------------------
# ----- Define additional parameters for the aggregate shocks model -----------
# -----------------------------------------------------------------------------
# Grid of capital-to-labor-ratios (factors)
MgridBase = np.array([0.1, 0.3, 0.6, 0.8, 0.9, 0.98,
                     1.0, 1.02, 1.1, 1.2, 1.6, 2.0, 3.0])

# Parameters for a Cobb-Douglas economy
PermGroFacAgg = 1.00          # Aggregate permanent income growth factor
# Number of points in discrete approximation to aggregate permanent shock dist
PermShkAggCount = 1
# Number of points in discrete approximation to aggregate transitory shock dist
TranShkAggCount = 1
PermShkAggStd = 0.00        # Standard deviation of log aggregate permanent shocks
TranShkAggStd = 0.00        # Standard deviation of log aggregate transitory shocks
DeprFac = 0.025               # Capital depreciation rate
CapShare = 0.36               # Capital's share of income
DiscFacPF = DiscFac           # Discount factor of perfect foresight calibration
# Coefficient of relative risk aversion of perfect foresight calibration
CRRAPF = CRRA
intercept_prev = 0.0          # Intercept of aggregate savings function
slope_prev = 1.0              # Slope of aggregate savings function
# Whether to print solution progress to screen while solving
verbose_cobb_douglas = True
# Number of simulated "burn in" periods to discard when updating AFunc
T_discard = 200
# Damping factor when updating AFunc; puts DampingFac weight on old params, rest on new
DampingFac = 0.5
max_loops = 20                # Maximum number of AFunc updating loops to allow

# Make a dictionary to specify an aggregate shocks consumer
init_agg_shocks = copy(init_idiosyncratic_shocks)
# Interest factor is endogenous in agg shocks model
del init_agg_shocks['Rfree']
del init_agg_shocks['CubicBool']    # Not supported yet for agg shocks model
del init_agg_shocks['vFuncBool']    # Not supported yet for agg shocks model
init_agg_shocks['PermGroFac'] = [1.0]
init_agg_shocks['MgridBase'] = MgridBase
init_agg_shocks['aXtraCount'] = 24
init_agg_shocks['aNrmInitStd'] = 0.0
init_agg_shocks['LivPrb'] = LivPrb


# Make a dictionary to specify a Cobb-Douglas economy
init_cobb_douglas = {'PermShkAggCount': PermShkAggCount,
                     'TranShkAggCount': TranShkAggCount,
                     'PermShkAggStd': PermShkAggStd,
                     'TranShkAggStd': TranShkAggStd,
                     'DeprFac': DeprFac,
                     'CapShare': CapShare,
                     'DiscFac': DiscFacPF,
                     'CRRA': CRRAPF,
                     'PermGroFacAgg': PermGroFacAgg,
                     'AggregateL': 1.0,
                     'act_T': 1200,
                     'intercept_prev': intercept_prev,
                     'slope_prev': slope_prev,
                     'verbose': verbose_cobb_douglas,
                     'T_discard': T_discard,
                     'DampingFac': DampingFac,
                     'max_loops': max_loops
                     }

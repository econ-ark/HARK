'''
Specifies examples of the full set of parameters required to solve various
consumption-saving models.  These models can be found in ConsIndShockModel,
ConsAggShockModel, ConsPrefShockModel, and ConsMarkovModel.
'''
from __future__ import division, print_function
from copy import copy
import numpy as np

# -----------------------------------------------------------------------------
# --- Define all of the parameters for the perfect foresight model ------------
# -----------------------------------------------------------------------------

CRRA = 2.0
Rfree = 1.03        # Interest factor on assets
DiscFac = 0.96      # Intertemporal discount factor
LivPrb = [0.98]     # Survival probability
PermGroFac = [1.01] # Permanent income growth factor
BoroCnstArt = None  # Artificial borrowing constraint
MaxKinks = 400      # Maximum number of grid points to allow in cFunc (should be large)
AgentCount = 10000  # Number of agents of this type (only matters for simulation)
aNrmInitMean = 0.0  # Mean of log initial assets (only matters for simulation)
aNrmInitStd = 1.0   # Standard deviation of log initial assets (only for simulation)
pLvlInitMean = 0.0  # Mean of log initial permanent income (only matters for simulation)
pLvlInitStd = 0.0   # Standard deviation of log initial permanent income (only matters for simulation)
PermGroFacAgg = 1.0 # Aggregate permanent income growth factor (only matters for simulation)
T_age = None        # Age after which simulated agents are automatically killed
T_cycle = 1         # Number of periods in the cycle for this agent type

# -----------------------------------------------------------------------------
# --- Define additional parameters for the idiosyncratic shocks model ---------
# -----------------------------------------------------------------------------

# dummy stuff to get things passing before full refactor is complete
init_idiosyncratic_shocks = {
    'Rfree' : 1.03,
    'CubicBool' : False,
    'vFuncBool' : False,
    'PermShkStd': [0.1],    # Standard deviation of log permanent income shocks
    'PermShkCount': 7,      # Number of points in discrete approximation to permanent income shocks
    'TranShkStd': [0.1],    # Standard deviation of log transitory income shocks
    'TranShkCount': 7,      # Number of points in discrete approximation to transitory income shocks
    'UnempPrb': 0.05,       # Probability of unemployment while working
    'UnempPrbRet': 0.005,   # Probability of "unemployment" while retired
    'IncUnemp': 0.3,        # Unemployment benefits replacement rate
    'IncUnempRet': 0.0,     # "Unemployment" benefits when retired
    'BoroCnstArt': 0.0,     # Artificial borrowing constraint; imposed minimum level of end-of period assets
    'tax_rate': 0.0,        # Flat income tax rate
    'T_retire': 0, # Period of retirement (0 --> no retirement)
    # assets above grid parameters
    'aXtraMin': 0.001,      # Minimum end-of-period "assets above minimum" value
    'aXtraMax': 20,         # Maximum end-of-period "assets above minimum" value
    'aXtraNestFac': 3,      # Exponential nesting factor when constructing "assets above minimum" grid
    'aXtraCount': 48,       # Number of points in the grid of "assets above minimum"
    'aXtraExtra': [None],   # Some other value of "assets above minimum" to add to the grid, not used
    # Income process variables
}




# -----------------------------------------------------------------------------
# ----- Define additional parameters for the aggregate shocks model -----------
# -----------------------------------------------------------------------------
MgridBase = np.array([0.1,0.3,0.6,0.8,0.9,0.98,1.0,1.02,1.1,1.2,1.6,2.0,3.0])  # Grid of capital-to-labor-ratios (factors)

# Parameters for a Cobb-Douglas economy
PermGroFacAgg = 1.00          # Aggregate permanent income growth factor
PermShkAggCount = 3           # Number of points in discrete approximation to aggregate permanent shock dist
TranShkAggCount = 3           # Number of points in discrete approximation to aggregate transitory shock dist
PermShkAggStd = 0.0063        # Standard deviation of log aggregate permanent shocks
TranShkAggStd = 0.0031        # Standard deviation of log aggregate transitory shocks
DeprFac = 0.025               # Capital depreciation rate
CapShare = 0.36               # Capital's share of income
DiscFacPF = DiscFac           # Discount factor of perfect foresight calibration
CRRAPF = CRRA                 # Coefficient of relative risk aversion of perfect foresight calibration
intercept_prev = 0.0          # Intercept of aggregate savings function
slope_prev = 1.0              # Slope of aggregate savings function
verbose_cobb_douglas = True   # Whether to print solution progress to screen while solving
T_discard = 200               # Number of simulated "burn in" periods to discard when updating AFunc
DampingFac = 0.5              # Damping factor when updating AFunc; puts DampingFac weight on old params, rest on new
max_loops = 20                # Maximum number of AFunc updating loops to allow

# Make a dictionary to specify an aggregate shocks consumer
init_agg_shocks = copy(init_idiosyncratic_shocks)
del init_agg_shocks['Rfree']        # Interest factor is endogenous in agg shocks model
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
                     'AggregateL':1.0,
                     'intercept_prev': intercept_prev,
                     'slope_prev': slope_prev,
                     'verbose': verbose_cobb_douglas,
                     'T_discard': T_discard,
                     'DampingFac': DampingFac,
                     'max_loops': max_loops
                     }

# -----------------------------------------------------------------------------
# ----- Define additional parameters for the Markov agg shocks model ----------
# -----------------------------------------------------------------------------
# This example makes a high risk, low growth state and a low risk, high growth state
MrkvArray = np.array([[0.90,0.10],[0.04,0.96]])
PermShkAggStd = [0.012,0.006]     # Standard deviation of log aggregate permanent shocks by state
TranShkAggStd = [0.006,0.003]     # Standard deviation of log aggregate transitory shocks by state
PermGroFacAgg = [0.98,1.02]       # Aggregate permanent income growth factor

# Make a dictionary to specify a Markov aggregate shocks consumer
init_agg_mrkv_shocks = copy(init_agg_shocks)
init_agg_mrkv_shocks['MrkvArray'] = MrkvArray

# Make a dictionary to specify a Markov Cobb-Douglas economy
init_mrkv_cobb_douglas = copy(init_cobb_douglas)
init_mrkv_cobb_douglas['PermShkAggStd'] = PermShkAggStd
init_mrkv_cobb_douglas['TranShkAggStd'] = TranShkAggStd
init_mrkv_cobb_douglas['PermGroFacAgg'] = PermGroFacAgg
init_mrkv_cobb_douglas['MrkvArray'] = MrkvArray
init_mrkv_cobb_douglas['MrkvNow_init'] = 0
init_mrkv_cobb_douglas['slope_prev'] = 2*[slope_prev]
init_mrkv_cobb_douglas['intercept_prev'] = 2*[intercept_prev]

'''
Specifies examples of the full set of parameters required to solve various
consumption-saving models.  These models can be found in ConsIndShockModel,
ConsAggShockModel, ConsPrefShockModel, and ConsMarkovModel.
'''
from copy import copy
import numpy as np

# -----------------------------------------------------------------------------
# --- Define all of the parameters for the perfect foresight model ------------
# -----------------------------------------------------------------------------

CRRA = 4.0                          # Coefficient of relative risk aversion
Rfree = 1.03                        # Interest factor on assets
DiscFac = 0.96                      # Intertemporal discount factor
LivPrb = [0.98]                     # Survival probability
PermGroFac = [1.01]                 # Permanent income growth factor
Nagents = 10000                     # Number of agents of this type (only matters for simulation)

# Make a dictionary to specify a perfect foresight consumer type
init_perfect_foresight = { 'CRRA': CRRA,
                           'Rfree': Rfree,
                           'DiscFac': DiscFac,
                           'LivPrb': LivPrb,
                           'PermGroFac': PermGroFac,
                           'Nagents': Nagents
                          }
                                                   
# -----------------------------------------------------------------------------
# --- Define additional parameters for the idiosyncratic shocks model ---------
# -----------------------------------------------------------------------------

# Parameters for constructing the "assets above minimum" grid
exp_nest = 3                        # Number of times to "exponentially nest" when constructing a_grid
aXtraMin = 0.001                    # Minimum end-of-period "assets above minimum" value
aXtraMax = 20                       # Minimum end-of-period "assets above minimum" value               
aXtraHuge = None                    # A very large value of assets to add to the grid, not used
aXtraExtra = None                   # Some other value of assets to add to the grid, not used
aXtraCount = 12                     # Number of points in the grid of "assets above minimum"

# Parameters describing the income process
PermShkCount = 7                    # Number of points in discrete approximation to permanent income shocks
TranShkCount = 7                    # Number of points in discrete approximation to transitory income shocks
PermShkStd = [0.1]                  # Standard deviation of log permanent income shocks
TranShkStd = [0.1]                  # Standard deviation of log transitory income shocks
UnempPrb = 0.05                     # Probability of unemployment while working
UnempPrbRet = 0.005                 # Probability of "unemployment" while retired
IncUnemp = 0.3                      # Unemployment benefits replacement rate
IncUnempRet = 0.0                   # "Unemployment" benefits when retired

# A few other parameters
BoroCnstArt = 0.0                   # Artificial borrowing constraint; imposed minimum level of end-of period assets
tax_rate = 0.0                      # Flat income tax rate
CubicBool = True                    # Use cubic spline interpolation when True, linear interpolation when False
vFuncBool = False                   # Whether to calculate the value function during solution
T_total = 1                         # Total number of periods in cycle for this agent
T_retire = 0                        # Period of retirement (0 --> no retirement)

# Make a dictionary to specify an idiosyncratic income shocks consumer
init_idiosyncratic_shocks = { 'CRRA': CRRA,
                              'Rfree': Rfree,
                              'DiscFac': DiscFac,
                              'LivPrb': LivPrb,
                              'PermGroFac': PermGroFac,
                              'Nagents': Nagents,
                              'exp_nest': exp_nest,
                              'aXtraMin': aXtraMin,
                              'aXtraMax': aXtraMax,
                              'aXtraCount': aXtraCount,
                              'aXtraExtra': [aXtraExtra,aXtraHuge],
                              'PermShkStd': PermShkStd,
                              'PermShkCount': PermShkCount,
                              'TranShkStd': TranShkStd,
                              'TranShkCount': TranShkCount,
                              'UnempPrb': UnempPrb,
                              'UnempPrbRet': UnempPrbRet,
                              'IncUnemp': IncUnemp,
                              'IncUnempRet': IncUnempRet,
                              'BoroCnstArt': BoroCnstArt,
                              'tax_rate':0.0,
                              'vFuncBool':vFuncBool,
                              'CubicBool':CubicBool,
                              'T_retire':T_retire,
                              'T_total':T_total
                             }
                             
# Make a dictionary to specify a lifecycle consumer with a finite horizon
init_lifecycle = copy(init_idiosyncratic_shocks)
init_lifecycle['PermGroFac'] = [1.01,1.01,1.01,1.01,1.01,1.02,1.02,1.02,1.02,1.02]
init_lifecycle['PermShkStd'] = [0.1,0.2,0.1,0.2,0.1,0.2,0.1,0,0,0]
init_lifecycle['TranShkStd'] = [0.3,0.2,0.1,0.3,0.2,0.1,0.3,0,0,0]
init_lifecycle['LivPrb']     = [0.99,0.98,0.97,0.96,0.95,0.94,0.93,0.92,0.91,0.90]
init_lifecycle['T_total']    = 10
init_lifecycle['T_retire']   = 7

# Make a dictionary to specify an infinite consumer with a four period cycle
init_cyclical = copy(init_idiosyncratic_shocks)
init_cyclical['PermGroFac'] = [1.082251, 2.8, 0.3, 1.1]
init_cyclical['PermShkStd'] = [0.1,0.1,0.1,0.1]
init_cyclical['TranShkStd'] = [0.1,0.1,0.1,0.1]
init_cyclical['LivPrb']     = 4*[0.98]
init_cyclical['T_total']    = 4


# -----------------------------------------------------------------------------
# -------- Define additional parameters for the "kinked R" model --------------
# -----------------------------------------------------------------------------

Rboro = 1.20           # Interest factor on assets when borrowing, a < 0
Rsave = 1.02           # Interest factor on assets when saving, a > 0

# Make a dictionary to specify a "kinked R" idiosyncratic shock consumer
init_kinked_R = copy(init_idiosyncratic_shocks)
del init_kinked_R['Rfree'] # get rid of constant interest factor
init_kinked_R['Rboro'] = Rboro
init_kinked_R['Rsave'] = Rsave
init_kinked_R['BoroCnstArt'] = None # kinked R is a bit silly if borrowing not allowed
init_kinked_R['aXtraCount'] = 48
init_kinked_R['CubicBool'] = False # kinked R currently only compatible with linear cFunc


# -----------------------------------------------------------------------------
# ----- Define additional parameters for the preference shock model -----------
# -----------------------------------------------------------------------------

PrefShkCount = 12        # Number of points in discrete approximation to preference shock dist
PrefShk_tail_N = 4       # Number of "tail points" on each end of pref shock dist
PrefShkStd = [0.30]      # Standard deviation of utility shocks

# Make a dictionary to specify a preference shock consumer
init_preference_shocks = copy(init_kinked_R)
init_preference_shocks['PrefShkCount'] = PrefShkCount
init_preference_shocks['PrefShk_tail_N'] = PrefShk_tail_N
init_preference_shocks['PrefShkStd'] = PrefShkStd


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
init_agg_shocks = copy(init_idiosyncratic_shocks)
del init_agg_shocks['Rfree']       # Interest factor is endogenous in agg shocks model
del init_agg_shocks['BoroCnstArt'] # Not supported yet for agg shocks model
del init_agg_shocks['CubicBool']   # Not supported yet for agg shocks model
del init_agg_shocks['vFuncBool']   # Not supported yet for agg shocks model
init_agg_shocks['kGridBase'] = kGridBase
init_agg_shocks['aXtraCount'] = 20

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

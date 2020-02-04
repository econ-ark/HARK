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

CRRA = 2.0                          # Coefficient of relative risk aversion
Rfree = 1.03                        # Interest factor on assets
DiscFac = 0.96                      # Intertemporal discount factor
LivPrb = [0.98]                     # Survival probability
PermGroFac = [1.01]                 # Permanent income growth factor
BoroCnstArt = None                  # Artificial borrowing constraint
MaxKinks = 400                      # Maximum number of grid points to allow in cFunc (should be large)
AgentCount = 10000                  # Number of agents of this type (only matters for simulation)
aNrmInitMean = 0.0                  # Mean of log initial assets (only matters for simulation)
aNrmInitStd  = 1.0                  # Standard deviation of log initial assets (only for simulation)
pLvlInitMean = 0.0                  # Mean of log initial permanent income (only matters for simulation)
pLvlInitStd  = 0.0                  # Standard deviation of log initial permanent income (only matters for simulation)
PermGroFacAgg = 1.0                 # Aggregate permanent income growth factor (only matters for simulation)
T_age = None                        # Age after which simulated agents are automatically killed
T_cycle = 1                         # Number of periods in the cycle for this agent type

# Make a dictionary to specify a perfect foresight consumer type
init_perfect_foresight = { 'CRRA': CRRA,
                           'Rfree': Rfree,
                           'DiscFac': DiscFac,
                           'LivPrb': LivPrb,
                           'PermGroFac': PermGroFac,
                           'BoroCnstArt': BoroCnstArt,
                           'MaxKinks': MaxKinks,
                           'AgentCount': AgentCount,
                           'aNrmInitMean' : aNrmInitMean,
                           'aNrmInitStd' : aNrmInitStd,
                           'pLvlInitMean' : pLvlInitMean,
                           'pLvlInitStd' : pLvlInitStd,
                           'PermGroFacAgg' : PermGroFacAgg,
                           'T_age' : T_age,
                           'T_cycle' : T_cycle
                          }

# -----------------------------------------------------------------------------
# --- Define additional parameters for the idiosyncratic shocks model ---------
# -----------------------------------------------------------------------------

# Parameters for constructing the "assets above minimum" grid
aXtraMin = 0.001                    # Minimum end-of-period "assets above minimum" value
aXtraMax = 20                       # Maximum end-of-period "assets above minimum" value
aXtraExtra = None                   # Some other value of "assets above minimum" to add to the grid, not used
aXtraNestFac = 3                    # Exponential nesting factor when constructing "assets above minimum" grid
aXtraCount = 48                     # Number of points in the grid of "assets above minimum"

# Parameters describing the income process
PermShkCount = 7                    # Number of points in discrete approximation to permanent income shocks
TranShkCount = 7                    # Number of points in discrete approximation to transitory income shocks
PermShkStd = [0.1]                  # Standard deviation of log permanent income shocks
TranShkStd = [0.1]                  # Standard deviation of log transitory income shocks
UnempPrb = 0.05                     # Probability of unemployment while working
UnempPrbRet = 0.005                 # Probability of "unemployment" while retired
IncUnemp = 0.3                      # Unemployment benefits replacement rate
IncUnempRet = 0.0                   # "Unemployment" benefits when retired
tax_rate = 0.0                      # Flat income tax rate
T_retire = 0                        # Period of retirement (0 --> no retirement)

# A few other parameters
BoroCnstArt = 0.0                  # Artificial borrowing constraint; imposed minimum level of end-of period assets
CubicBool = False                  # Use cubic spline interpolation when True, linear interpolation when False
vFuncBool = False                  # Whether to calculate the value function during solution

# Make a dictionary to specify an idiosyncratic income shocks consumer
init_idiosyncratic_shocks = { 'CRRA': CRRA,
                              'Rfree': Rfree,
                              'DiscFac': DiscFac,
                              'LivPrb': LivPrb,
                              'PermGroFac': PermGroFac,
                              'AgentCount': AgentCount,
                              'aXtraMin': aXtraMin,
                              'aXtraMax': aXtraMax,
                              'aXtraNestFac':aXtraNestFac,
                              'aXtraCount': aXtraCount,
                              'aXtraExtra': [aXtraExtra],
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
                              'aNrmInitMean' : aNrmInitMean,
                              'aNrmInitStd' : aNrmInitStd,
                              'pLvlInitMean' : pLvlInitMean,
                              'pLvlInitStd' : pLvlInitStd,
                              'PermGroFacAgg' : PermGroFacAgg,
                              'T_age' : T_age,
                              'T_cycle' : T_cycle
                             }

# Make a dictionary to specify a lifecycle consumer with a finite horizon
init_lifecycle = copy(init_idiosyncratic_shocks)
init_lifecycle['PermGroFac'] = [1.01,1.01,1.01,1.01,1.01,1.02,1.02,1.02,1.02,1.02]
init_lifecycle['PermShkStd'] = [0.1,0.2,0.1,0.2,0.1,0.2,0.1,0,0,0]
init_lifecycle['TranShkStd'] = [0.3,0.2,0.1,0.3,0.2,0.1,0.3,0,0,0]
init_lifecycle['LivPrb']     = [0.99,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
init_lifecycle['T_cycle']    = 10
init_lifecycle['T_retire']   = 7
init_lifecycle['T_age']      = 11 # Make sure that old people die at terminal age and don't turn into newborns!

# Make a dictionary to specify an infinite consumer with a four period cycle
init_cyclical = copy(init_idiosyncratic_shocks)
init_cyclical['PermGroFac'] = [1.082251, 2.8, 0.3, 1.1]
init_cyclical['PermShkStd'] = [0.1,0.1,0.1,0.1]
init_cyclical['TranShkStd'] = [0.1,0.1,0.1,0.1]
init_cyclical['LivPrb']     = 4*[0.98]
init_cyclical['T_cycle']    = 4


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
init_kinked_R['CubicBool'] = True # kinked R is now compatible with linear cFunc and cubic cFunc
init_kinked_R['aXtraCount'] = 48   # ...so need lots of extra gridpoints to make up for it


# -----------------------------------------------------------------------------
# ----- Define additional parameters for the preference shock model -----------
# -----------------------------------------------------------------------------

PrefShkCount = 12        # Number of points in discrete approximation to preference shock dist
PrefShk_tail_N = 4       # Number of "tail points" on each end of pref shock dist
PrefShkStd = [0.30]      # Standard deviation of utility shocks

# Make a dictionary to specify a preference shock consumer
init_preference_shocks = copy(init_idiosyncratic_shocks)
init_preference_shocks['PrefShkCount'] = PrefShkCount
init_preference_shocks['PrefShk_tail_N'] = PrefShk_tail_N
init_preference_shocks['PrefShkStd'] = PrefShkStd
init_preference_shocks['aXtraCount'] = 48
init_preference_shocks['CubicBool'] = False # pref shocks currently only compatible with linear cFunc

# Make a dictionary to specify a "kinky preference" consumer, who has both shocks
# to utility and a different interest rate on borrowing vs saving
init_kinky_pref = copy(init_kinked_R)
init_kinky_pref['PrefShkCount'] = PrefShkCount
init_kinky_pref['PrefShk_tail_N'] = PrefShk_tail_N
init_kinky_pref['PrefShkStd'] = PrefShkStd

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


# -----------------------------------------------------------------------------
# ----- Define additional parameters for the persistent shocks model ----------
# -----------------------------------------------------------------------------

pLvlPctiles = np.concatenate(([0.001, 0.005, 0.01, 0.03], np.linspace(0.05, 0.95, num=19),[0.97, 0.99, 0.995, 0.999]))
PrstIncCorr = 0.98       # Serial correlation coefficient for permanent income

# Make a dictionary for the "explicit permanent income" idiosyncratic shocks model
init_explicit_perm_inc = copy(init_idiosyncratic_shocks)
init_explicit_perm_inc['pLvlPctiles'] = pLvlPctiles
init_explicit_perm_inc['PermGroFac'] = [1.0] # long run permanent income growth doesn't work yet
init_explicit_perm_inc['aXtraMax'] = 30
init_explicit_perm_inc['aXtraExtra'] = [0.005,0.01]

# Make a dictionary for the "persistent idiosyncratic shocks" model
init_persistent_shocks = copy(init_explicit_perm_inc)
init_persistent_shocks['PrstIncCorr'] = PrstIncCorr

# -----------------------------------------------------------------------------
# ----- Define additional parameters for the medical shocks model -------------
# -----------------------------------------------------------------------------

CRRAmed = 1.5*CRRA     # Coefficient of relative risk aversion for medical care
MedShkAvg = [0.001]    # Average of medical need shocks
MedShkStd = [5.0]      # Standard deviation of (log) medical need shocks
MedShkCount = 5        # Number of medical shock points in "body"
MedShkCountTail = 15   # Number of medical shock points in "tail" (upper only)
MedPrice = [1.5]       # Relative price of a unit of medical care

# Make a dictionary for the "medical shocks" model
init_medical_shocks = copy(init_persistent_shocks)
init_medical_shocks['CRRAmed'] = CRRAmed
init_medical_shocks['MedShkAvg'] = MedShkAvg
init_medical_shocks['MedShkStd'] = MedShkStd
init_medical_shocks['MedShkCount'] = MedShkCount
init_medical_shocks['MedShkCountTail'] = MedShkCountTail
init_medical_shocks['MedPrice'] = MedPrice
init_medical_shocks['aXtraCount'] = 32

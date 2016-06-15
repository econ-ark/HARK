'''
Specifies examples of the full set of parameters required to solve various
consumption-saving models.
'''
# -----------------------------------------------------------------------------
# --- Define all of the parameters for the perfect foresight model ------------
# -----------------------------------------------------------------------------

CRRA = 4.0                          # Coefficient of relative risk aversion
Rfree = 1.03                        # Interest factor on assets
DiscFac = [0.96]                    # Intertemporal discount factor
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


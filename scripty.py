from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
from copy import copy

CRRA = 2.0                          # Coefficient of relative risk aversion
Rfree = 1.01                        # Interest factor on assets
DiscFac = 0.96                      # Intertemporal discount factor
LivPrb = [0.98]                     # Survival probability
PermGroFac = [1.01]                 # Permanent income growth factor
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
                           'AgentCount': AgentCount,
                           'aNrmInitMean' : aNrmInitMean,
                           'aNrmInitStd' : aNrmInitStd,
                           'pLvlInitMean' : pLvlInitMean,
                           'pLvlInitStd' : pLvlInitStd,
                           'PermGroFacAgg' : PermGroFacAgg,
                           'T_age' : T_age,
                           'T_cycle' : T_cycle
                          }


RiskPremium = 0.1
# Parameters for constructing the "assets above minimum" grid
aXtraMin = 0.001                    # Minimum end-of-period "assets above minimum" value
aXtraMax = 20                       # Maximum end-of-period "assets above minimum" value
aXtraExtra = None                   # Some other value of "assets above minimum" to add to the grid, not used
aXtraNestFac = 3                    # Exponential nesting factor when constructing "assets above minimum" grid
aXtraCount = 48                     # Number of points in the grid of "assets above minimum"

# Parameters describing the income process
PermShkCount = 1                    # Number of points in discrete approximation to permanent income shocks
TranShkCount = 1                    # Number of points in discrete approximation to transitory income shocks
PermShkStd = [0.1]                  # Standard deviation of log permanent income shocks
TranShkStd = [0.1]                  # Standard deviation of log transitory income shocks
UnempPrb = 0.00                     # Probability of unemployment while working
UnempPrbRet = 0.000                 # Probability of "unemployment" while retired
IncUnemp = 1.0                     # Unemployment benefits replacement rate
IncUnempRet = 1.0                   # "Unemployment" benefits when retired
tax_rate = 0.0                      # Flat income tax rate
T_retire = 0                        # Period of retirement (0 --> no retirement)

# A few other parameters
BoroCnstArt = 0.0                  # Artificial borrowing constraint; imposed minimum level of end-of period assets
CubicBool = False                  # Use cubic spline interpolation when True, linear interpolation when False
vFuncBool = True                  # Whether to calculate the value function during solution

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
                              'T_cycle' : T_cycle,
                              'RiskPremium' : RiskPremium,
                              'TradesStocks' : True,
}

init_lifecycle = copy(init_idiosyncratic_shocks)
init_lifecycle['CRRA'] = 6.0
init_lifecycle['PermGroFac'] = [1.01,1.01,1.01,1.01,1.01,1.02,1.02,1.02,1.02,1.02]
init_lifecycle['PermShkStd'] = [0,0,0,0,0,0,0,0,0,0]
init_lifecycle['TranShkStd'] = [0.3,0.2,0.1,0.3,0.2,0.1,0.3,0,0,0]
init_lifecycle['LivPrb']     = [0.99,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
init_lifecycle['T_cycle']    = 10
init_lifecycle['T_retire']   = 7
init_lifecycle['T_age']      = 11 # Make sure that old people die at terminal age and don't turn into newborns!

LifecycleExample = IndShockConsumerType(**init_lifecycle)

LifecycleExample.cycles = 1 # Make this consumer live a sequence of periods -- a lifetime -- exactly once
LifecycleExample.solve()

import numpy as np
import matplotlib.pyplot as plt
grid = np.linspace(0,10,140)
cons = LifecycleExample.solution[0].cFunc(grid)
plt.plot(grid, cons)
cons = LifecycleExample.solution[1].cFunc(grid)
plt.plot(grid, cons)
cons = LifecycleExample.solution[2].cFunc(grid)
plt.plot(grid, cons)
cons = LifecycleExample.solution[3].cFunc(grid)
plt.plot(grid, cons)
plt.show()



import numpy as np
import matplotlib.pyplot as plt
grid = np.linspace(0,10,140)
cons = LifecycleExample.solution[0].vFunc(grid)
plt.plot(grid, cons)
cons = LifecycleExample.solution[1].vFunc(grid)
plt.plot(grid, cons)
cons = LifecycleExample.solution[2].vFunc(grid)
plt.plot(grid, cons)
cons = LifecycleExample.solution[3].vFunc(grid)
plt.plot(grid, cons)
plt.show()

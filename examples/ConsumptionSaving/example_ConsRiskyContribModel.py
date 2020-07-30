# %%
'''
Example implementations of HARK.ConsumptionSaving.ConsPortfolioModel
'''
from HARK.ConsumptionSaving.ConsRiskyContribModel import PortfolioConsumerType, init_portfolio
from HARK.ConsumptionSaving.ConsIndShockModel import init_lifecycle
from HARK.utilities import plotFuncs
from copy import copy
from time import time
import numpy as np
import matplotlib.pyplot as plt

# %%
# Make and solve an example of the risky pension contribution consumer type
init_sticky_share = init_portfolio.copy()
init_sticky_share['AdjustPrb'] = 0.15

# %%
# Make and solve a discrete portfolio choice consumer type
print('Now solving a portfolio choice problem with "sticky" pension contribution shares; this might take a moment...')
StickyType = PortfolioConsumerType(**init_sticky_share)
StickyType.cycles = 0
t0 = time()
StickyType.solve()
t1 = time()
StickyType.cFuncAdj = [StickyType.solution[t].cFuncAdj for t in range(StickyType.T_cycle)]
StickyType.cFuncFxd = [StickyType.solution[t].cFuncFxd for t in range(StickyType.T_cycle)]
StickyType.ShareFunc = [StickyType.solution[t].ShareFuncAdj for t in range(StickyType.T_cycle)]
print('Solving an infinite horizon sticky portfolio choice problem took ' + str(t1-t0) + ' seconds.')

# %%
# Plot the consumption and risky-share functions
print('Consumption function over market resources when the agent can adjust his portfolio:')
plotFuncs(StickyType.cFuncAdj[0], 0., 50.)

# %%
print("Consumption function over market resources when the agent CAN'T adjust, by current share:")
M = np.linspace(0., 50., 200)
for s in np.linspace(0.,1.,21):
    C = StickyType.cFuncFxd[0](M, s*np.ones_like(M))
    plt.plot(M,C)
plt.xlim(0.,50.)
plt.ylim(0.,None)
plt.show()

# %%
print('Risky asset share function over market resources (when possible to adjust):')
print('Optimal (blue) versus Theoretical Limit (orange)')
plt.xlabel('Normalized Market Resources')
plt.ylabel('Portfolio Share')
plt.ylim(0.0,1.0)
plotFuncs([StickyType.ShareFunc[0]
           ,lambda m: StickyType.ShareLimit*np.ones_like(m)
          ] , 0., 200.)
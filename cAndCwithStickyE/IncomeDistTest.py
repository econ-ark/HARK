# Import the HARK library.  The assumption is that this code is in a folder
# contained in the HARK folder. Also import ConsumptionSavingModel
import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../ConsumptionSaving'))

import numpy as np
import matplotlib.pyplot as plt
from HARKsimulation import drawMeanOneLognormal

N = 10000
T = 10000
PermShkStd = np.sqrt(0.00436)
LivPrb = 0.995
DiePrb = 1 - LivPrb

pLvl_t = np.ones(N)
pLvl_hist = np.ones((T,N))
for t in range(T):
    draws = np.random.rand(N)
    who_dies = draws < DiePrb
    pLvl_t[who_dies] = 1.0
    Shks = drawMeanOneLognormal(N,sigma=PermShkStd,seed=np.random.randint(0,2**31-1))
    pLvl_t = pLvl_t*Shks
    pLvl_hist[t,:] = pLvl_t
    
plt.plot(np.std(pLvl_hist,axis=1))
plt.show()

plt.plot(np.std(np.log(pLvl_hist),axis=1))
plt.show()
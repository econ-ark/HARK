'''
This is a small module to test the ConsAggMarkovModel.
'''

import sys 
sys.path.insert(0,'../')

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from time import clock

from ConsAggShockModel import AggShockMarkovConsumerType, CobbDouglasMarkovEconomy
import ConsumerParameters as Params
mystr = lambda number : "{:.4f}".format(number)

# Make an aggregate shocks consumer type
AggShockMrkvExample = AggShockMarkovConsumerType(**Params.init_agg_mrkv_shocks)
AggShockMrkvExample.IncomeDstn[0] = 2*[AggShockMrkvExample.IncomeDstn[0]]
AggShockMrkvExample.cycles = 0

# Make a Cobb-Douglas economy for the agents
EconomyExample = CobbDouglasMarkovEconomy(agents = [AggShockMrkvExample],**Params.init_mrkv_cobb_douglas)
EconomyExample.makeAggShkHist() # Simulate a history of aggregate shocks

# Have the consumers inherit relevant objects from the economy
AggShockMrkvExample.getEconomyData(EconomyExample)

# Solve the microeconomic model for the aggregate shocks example type (and display results)
#t_start = clock()
#AggShockMrkvExample.solve()
#t_end = clock()
#print('Solving an aggregate shocks Markov consumer took ' + mystr(t_end-t_start) + ' seconds.')
#print('Consumption function at each aggregate market resources-to-labor ratio gridpoint (for each macro state):')
#m_grid = np.linspace(0,10,200)
#AggShockMrkvExample.unpackcFunc()
#for i in range(2):
#    for M in AggShockMrkvExample.Mgrid.tolist():
#        mMin = AggShockMrkvExample.solution[0].mNrmMin[i](M)
#        c_at_this_M = AggShockMrkvExample.cFunc[0][i](m_grid+mMin,M*np.ones_like(m_grid))
#        plt.plot(m_grid+mMin,c_at_this_M)
#    plt.show()

# Solve the "macroeconomic" model by searching for a "fixed point dynamic rule"
t_start = clock()
EconomyExample.solve()
t_end = clock()
print('Solving the "macroeconomic" aggregate shocks model took ' + str(t_end - t_start) + ' seconds.')
print('Aggregate savings as a function of aggregate market resources (for each macro state):')
m_grid = np.linspace(0,10,200)
AggShockMrkvExample.unpackcFunc()
for i in range(2):
    for M in AggShockMrkvExample.Mgrid.tolist():
        mMin = AggShockMrkvExample.solution[0].mNrmMin[i](M)
        c_at_this_M = AggShockMrkvExample.cFunc[0][i](m_grid+mMin,M*np.ones_like(m_grid))
        plt.plot(m_grid+mMin,c_at_this_M)
    plt.show()

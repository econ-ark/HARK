from HARK.ConsumptionSaving.ConsAggShockModel import KrusellSmithType, KrusellSmithEconomy
from HARK.utilities import plotFuncs
from time import time
import matplotlib.pyplot as plt

# Make default KS agent type and economy
TestEconomy = KrusellSmithEconomy()
TestType = KrusellSmithType()
TestType.cycles = 0
TestType.getEconomyData(TestEconomy)
TestEconomy.agents = [TestType]
TestEconomy.makeMrkvHist()

# Solve the Krusell-Smith economy
t0 = time()
TestEconomy.solve()
t1 = time()
print('Solving a KS type took ' + str(t1-t0) + ' seconds.')

state_names = ['bad economy, unemployed', 'bad economy, employed',
               'good economy, unemployed', 'good economy, employed']

# Plot the consumption function for each discrete state
for j in range(4):
    plt.xlabel(r'Idiosyncratic market resources $m$')
    plt.ylabel(r'Consumption $c$')
    plt.title('Consumption function by aggregate market resources: ' + state_names[j])
    plotFuncs(TestType.solution[0].cFunc[j].xInterpolators, 0., 50.)

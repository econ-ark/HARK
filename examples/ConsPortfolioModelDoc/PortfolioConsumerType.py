'''
Example implementations of HARK.ConsumptionSaving.ConsPortfolioModel
'''
from HARK.ConsumptionSaving.ConsPortfolioModel import PortfolioConsumerType
from HARK.ConsumptionSaving.ConsumerParameters import init_lifecycle, init_portfolio
from HARK.utilities import plotFuncs
from copy import copy
from time import time

# Make and solve an example portfolio choice consumer type
print('Now solving an example portfolio choice problem; this might take a minute...')
MyType = PortfolioConsumerType()
MyType.cycles = 10
t0 = time()
MyType.solve()
t1 = time()
MyType.cFunc = [MyType.solution[t].cFunc[0][0] for t in range(MyType.T_cycle)]
MyType.RiskyShareFunc = [MyType.solution[t].RiskyShareFunc[0][0] for t in range(MyType.T_cycle)]
print('Solving an infinite horizon portfolio choice problem took ' + str(t1-t0) + ' seconds.')

# Plot the consumption and risky-share functions
print('Consumption function over market resources:')
plotFuncs(MyType.cFunc[0], 0., 20.)
print('Risky asset share function over market resources:')
plotFuncs(MyType.RiskyShareFunc[0], 0., 20.)

# Now simulate this consumer type
MyType.track_vars = ['cNrmNow', 'RiskyShareNow', 'aNrmNow', 't_age']
MyType.T_sim = 100
MyType.initializeSim()
MyType.simulate()

print('\n\n\n')

###############################################################################

# Make a second example type, but this one has *age-varying* perceptions of risky asset returns.
# Begin by making a lifecycle dictionary, but adjusted for the portfolio choice model.
init_age_varying_risk_perceptions = copy(init_lifecycle)
init_age_varying_risk_perceptions['RiskyCount']      = init_portfolio['RiskyCount']
init_age_varying_risk_perceptions['RiskyShareCount'] = init_portfolio['RiskyShareCount']
init_age_varying_risk_perceptions['aXtraMax']        = init_portfolio['aXtraMax']
init_age_varying_risk_perceptions['aXtraCount']      = init_portfolio['aXtraCount']
init_age_varying_risk_perceptions['aXtraNestFac']    = init_portfolio['aXtraNestFac']
init_age_varying_risk_perceptions['BoroCnstArt']     = init_portfolio['BoroCnstArt']
init_age_varying_risk_perceptions['CRRA']            = init_portfolio['CRRA']
init_age_varying_risk_perceptions['DiscFac']         = init_portfolio['DiscFac']

init_age_varying_risk_perceptions['RiskyAvg']        = 10*[1.08]
init_age_varying_risk_perceptions['RiskyStd']        = [0.20,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29]
init_age_varying_risk_perceptions['RiskyAvgTrue']    = 1.08
init_age_varying_risk_perceptions['RiskyStdTrue']    = 0.20
AgeVaryingRiskPercType = PortfolioConsumerType(**init_age_varying_risk_perceptions)
AgeVaryingRiskPercType.cycles = 1

# Solve the agent type with age-varying risk perceptions
print('Now solving a portfolio choice problem with age-varying risk perceptions...')
t0 = time()
AgeVaryingRiskPercType.solve()
AgeVaryingRiskPercType.cFunc = [AgeVaryingRiskPercType.solution[t].cFunc[0][0] for t in range(AgeVaryingRiskPercType.T_cycle)]
AgeVaryingRiskPercType.RiskyShareFunc = [AgeVaryingRiskPercType.solution[t].RiskyShareFunc[0][0] for t in range(AgeVaryingRiskPercType.T_cycle)]
t1 = time()
print('Solving a ' + str(AgeVaryingRiskPercType.T_cycle) + ' period portfolio choice problem with age-varying risk perceptions took ' + str(t1-t0) + ' seconds.')

# Plot the consumption and risky-share functions
print('Consumption function over market resources in each lifecycle period:')
plotFuncs(AgeVaryingRiskPercType.cFunc, 0., 20.)
print('Risky asset share function over market resources in each lifecycle period:')
plotFuncs(AgeVaryingRiskPercType.RiskyShareFunc, 0., 20.)

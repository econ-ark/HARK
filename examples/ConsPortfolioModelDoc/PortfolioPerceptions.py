'''
A (mostly) live-programming exercise with Mridul on March 7, 2020, to make a
variation of the portfolio model in which *beliefs* about return risk vary by
age, but the true risk level does not.
'''
from HARK.ConsumptionSaving.ConsPortfolioModel import PortfolioConsumerType
from HARK.ConsumptionSaving.ConsumerParameters import init_lifecycle
from copy import copy

init_portfolio = copy(init_lifecycle) # Default parameter values for inf horiz model
init_portfolio['RiskyAvg']        = 1.08
init_portfolio['RiskyStd']        = 0.20
init_portfolio['RiskyCount']      = 11  # Number of points in the approximation; 2 points is minimum
init_portfolio['RiskyShareCount'] = 25  # How many discrete points to allow in the share approximation
init_portfolio['Rfree']           = 1.0 # Riskfree return factor is 1 (interest rate is zero)
init_portfolio['CRRA']            = 6.0 # Relative risk aversion

# Uninteresting technical parameters:
init_portfolio['aXtraMax']        = 100 
init_portfolio['aXtraCount']      = 50
init_portfolio['aXtraNestFac']    = 1
init_portfolio['BoroCnstArt']     = 0.0 # important for theoretical reasons
init_portfolio['DiscFac'] = 0.90

# Create portfolio choice consumer type
MyType = PortfolioConsumerType(**init_portfolio)
MyType.cycles = 1
MyType.solve()
MyType.cFunc = [MyType.solution[t].cFunc[0][0] for t in range(MyType.T_cycle)]
MyType.RiskyShareFunc = [MyType.solution[t].RiskyShareFunc[0][0] for t in range(MyType.T_cycle)]

import copy
import HARK.ConsumptionSaving.ConsPortfolioModel as cpm
import HARK.ConsumptionSaving.ConsumerParameters as param
import unittest

class testPortfolioConsumerType(unittest.TestCase):

    def setUp(self):

        # Parameters from Mehra and Prescott (1985):
        Avg = 1.08 # equity premium 
        Std = 0.20 # standard deviation of rate-of-return shocks 

        RiskyDstnFunc = cpm.RiskyDstnFactory(RiskyAvg=Avg, RiskyStd=Std)       # Generates nodes for integration
        RiskyDrawFunc = cpm.LogNormalRiskyDstnDraw(RiskyAvg=Avg, RiskyStd=Std) # Function to generate draws from a lognormal distribution

        init_portfolio = copy.copy(param.init_idiosyncratic_shocks) # Default parameter values for inf horiz model - including labor income with transitory and permanent shocks
        init_portfolio['approxRiskyDstn'] = RiskyDstnFunc
        init_portfolio['drawRiskyFunc']   = RiskyDrawFunc
        init_portfolio['RiskyCount']      = 2   # Number of points in the approximation; 2 points is minimum
        init_portfolio['RiskyShareCount'] = 25  # How many discrete points to allow for portfolio share
        init_portfolio['Rfree']           = 1.0 # Riskfree return factor (interest rate is zero)
        init_portfolio['CRRA']            = 6.0 # Relative risk aversion

        # Uninteresting technical parameters:
        init_portfolio['aXtraMax']        = 100 
        init_portfolio['aXtraCount']      = 50
        init_portfolio['BoroCnstArt']     = 0.0 # important for theoretical reasons
        init_portfolio['DiscFac'] = 0.92 # Make them impatient even wrt a riskfree return of 1.08 

        # Create portfolio choice consumer type
        self.pcct = cpm.PortfolioConsumerType(**init_portfolio)

        # %% {"code_folding": []}
        # Solve the model under the given parameters

        self.pcct.solve()

    def test_RiskyShareFunc(self):

        self.assertAlmostEqual(
            self.pcct.solution[0].RiskyShareFunc[0][0](2).tolist(),
            0.8796982720076255)

        self.assertAlmostEqual(
            self.pcct.solution[0].RiskyShareFunc[0][0](8).tolist(),
            0.69738175)

from HARK.ConsumptionSaving.TractableBufferStockModel import TractableConsumerType
import unittest


class testPortfolioConsumerType(unittest.TestCase):
    def setUp(self):
        # Create portfolio choice consumer type
        self.tct = TractableConsumerType()
        # Solve the model under the given parameters
        self.tct.solve()

    def test_simulation(self):

        self.tct.T_sim = 30
        self.tct.AgentCount = 10
        self.tct.track_vars += ["mLvlNow"]
        self.tct.initializeSim()
        self.tct.simulate()

        self.assertAlmostEqual(self.tct.history["mLvlNow"][15][0], 5.304746367434934)

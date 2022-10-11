from HARK.ConsumptionSaving.TractableBufferStockModel import TractableConsumerType
import unittest


class testTractableBufferStock(unittest.TestCase):
    def setUp(self):
        # Create portfolio choice consumer type
        self.tct = TractableConsumerType()
        # Solve the model under the given parameters
        self.tct.solve()

    def test_simulation(self):

        self.tct.T_sim = 30 # Number of periods to simulate
        self.tct.AgentCount = 10 # Number of agents to simulate
        self.tct.aLvlInitMean = 0.0  # Mean of log initial assets for new agents
        self.tct.aLvlInitStd = 1.0 # stdev of log initial assets for new agents
        self.tct.T_cycle = 1
        self.tct.track_vars += ["mLvl", "cLvlNow", "aLvl"]
        self.tct.initialize_sim()
        self.tct.simulate()

        self.assertAlmostEqual(self.tct.history["mLvl"][15][0] - self.tct.history["cLvlNow"][15][0], self.tct.history["aLvl"][15][0] )

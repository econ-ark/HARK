import HARK.ConsumptionSaving.ConsPortfolioModel as cpm
import unittest


class testPortfolioConsumerType(unittest.TestCase):
    def setUp(self):
        # Create portfolio choice consumer type
        self.pcct = cpm.PortfolioConsumerType()
        self.pcct.cycles = 0

        # Solve the model under the given parameters

        self.pcct.solve()

    def test_RiskyShareFunc(self):

        self.assertAlmostEqual(
            self.pcct.solution[0].ShareFuncAdj(8).tolist(), 0.9507419932531964
        )

        self.assertAlmostEqual(
            self.pcct.solution[0].ShareFuncAdj(16).tolist(), 0.6815883614201397
        )

    def test_simulation(self):

        self.pcct.T_sim = 30
        self.pcct.AgentCount = 10
        self.pcct.track_vars += ['aNrmNow']
        self.pcct.initializeSim()
        self.pcct.simulate()

        self.assertAlmostEqual(
            self.pcct.history['aNrmNow'][15][0], 5.304746367434934
        )

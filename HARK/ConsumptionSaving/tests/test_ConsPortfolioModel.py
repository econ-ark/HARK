import HARK.ConsumptionSaving.ConsPortfolioModel as cpm
import unittest

class testPortfolioConsumerType(unittest.TestCase):

    def setUp(self):
        # Create portfolio choice consumer type
        self.pcct = cpm.PortfolioConsumerType()

        # %% {"code_folding": []}
        # Solve the model under the given parameters

        self.pcct.solve()

    def test_RiskyShareFunc(self):

        self.assertAlmostEqual(
            self.pcct.solution[0].RiskyShareFunc[0][0](2).tolist(),
            0.44093501839091315)

        self.assertAlmostEqual(
            self.pcct.solution[0].RiskyShareFunc[0][0](8).tolist(),
            0.34742262624144954)

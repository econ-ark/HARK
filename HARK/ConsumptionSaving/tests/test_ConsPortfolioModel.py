import HARK.ConsumptionSaving.ConsPortfolioModel as cpm
import unittest

class testPortfolioConsumerType(unittest.TestCase):

    def setUp(self):
        # Create portfolio choice consumer type
        self.pcct = cpm.PortfolioConsumerType()
        self.pcct.cycles = 0
        
        # %% {"code_folding": []}
        # Solve the model under the given parameters

        self.pcct.solve()

    def test_RiskyShareFunc(self):

        self.assertAlmostEqual(
            self.pcct.solution[0].ShareFuncAdj(8).tolist(),
            0.9507419932531964)

        self.assertAlmostEqual(
            self.pcct.solution[0].ShareFuncAdj(16).tolist(),
            0.6815883614201397)

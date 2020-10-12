import HARK.ConsumptionSaving.ConsPortfolioModel as cpm
import numpy as np
import unittest


class PortfolioConsumerTypeTestCase(unittest.TestCase):
    def setUp(self):
        # Create portfolio choice consumer type
        self.pcct = cpm.PortfolioConsumerType()
        self.pcct.cycles = 0

        # Solve the model under the given parameters

        self.pcct.solve()

class UnitsPortfolioConsumerTypeTestCase(PortfolioConsumerTypeTestCase):

    def test_RiskyShareFunc(self):

        self.assertAlmostEqual(
            self.pcct.solution[0].ShareFuncAdj(8).tolist(), 0.9507419932531964
        )

        self.assertAlmostEqual(
            self.pcct.solution[0].ShareFuncAdj(16).tolist(), 0.6815883614201397
        )

    def test_solution(self):

        self.assertAlmostEqual(
            self.pcct.solution[0].cFuncAdj(10).tolist(),
            1.6996557721625785
        )

        self.assertAlmostEqual(
            self.pcct.solution[0].ShareFuncAdj(10).tolist(),
            0.8498496999408691
        )

    def test_simOnePeriod(self):

        self.pcct.T_sim = 30
        self.pcct.AgentCount = 10
        self.pcct.track_vars += ['aNrmNow']
        self.pcct.initializeSim()

        self.assertFalse(
            np.any(self.pcct.AdjustNow)
        )

        self.pcct.simOnePeriod()

        self.assertAlmostEqual(
            self.pcct.ShareNow[0],
            0.8627164488246847
        )
        self.assertAlmostEqual(
            self.pcct.cNrmNow[0],
            1.67874799
        )

class SimulatePortfolioConsumerTypeTestCase(PortfolioConsumerTypeTestCase):

    def test_simulation(self):

        self.pcct.T_sim = 30
        self.pcct.AgentCount = 10
        self.pcct.track_vars += [
            'mNrmNow','cNrmNow','aNrmNow'
        ]
        self.pcct.initializeSim()

        self.pcct.simulate()

        self.assertAlmostEqual(
            self.pcct.history['mNrmNow'][0][0], 9.70233892
        )

        self.assertAlmostEqual(
            self.pcct.history['cNrmNow'][0][0], 1.6787479894848298
        )

        self.assertAlmostEqual(
            self.pcct.history['aNrmNow'][0][0], 8.023590930905383
        )

        self.assertAlmostEqual(
            self.pcct.history['mNrmNow'][1][0], 8.287859049575047
        )

        self.assertAlmostEqual(
            self.pcct.history['cNrmNow'][1][0], 1.5773607434989751
        )

        self.assertAlmostEqual(
            self.pcct.history['aNrmNow'][1][0], 6.710498306076072
        )

        self.assertAlmostEqual(
            self.pcct.history['aNrmNow'][15][0], 5.304746367434934
        )

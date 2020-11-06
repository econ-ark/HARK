import HARK.ConsumptionSaving.ConsPortfolioFrameModel as cpfm
import numpy as np
import unittest


class PortfolioConsumerTypeTestCase(unittest.TestCase):
    def setUp(self):
        # Create portfolio choice consumer type
        self.pcct = cpfm.PortfolioConsumerFrameType()
        self.pcct.cycles = 0

        # Solve the model under the given parameters

        self.pcct.solve()

class UnitsPortfolioConsumerTypeTestCase(PortfolioConsumerTypeTestCase):
    def test_simOnePeriod(self):

        self.pcct.T_sim = 30
        self.pcct.AgentCount = 10
        self.pcct.track_vars += ['aNrmNow']
        self.pcct.initializeSim()

        self.assertFalse(
            np.any(self.pcct.shocks['AdjustNow'])
        )

        self.pcct.simOnePeriod()

        self.assertAlmostEqual(
            self.pcct.controls["ShareNow"][0],
            0.8627164488246847
        )
        self.assertAlmostEqual(
            self.pcct.controls["cNrmNow"][0],
            1.67874799
        )

class SimulatePortfolioConsumerTypeTestCase(PortfolioConsumerTypeTestCase):

    def test_simulation(self):

        self.pcct.T_sim = 30
        self.pcct.AgentCount = 10
        self.pcct.track_vars += [
            'mNrmNow',
            'cNrmNow',
            'ShareNow',
            'aNrmNow',
            'RiskyNow',
            'RportNow',
            'AdjustNow',
            'PermShkNow',
            'bNrmNow'
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
            self.pcct.history['ShareNow'][0][0], 0.8627164488246847
        )

        self.assertAlmostEqual(
            self.pcct.history['aNrmNow'][0][0], 8.023590930905383
        )

        self.assertAlmostEqual(
            self.pcct.history['AdjustNow'][0][0], 1.0
        )


        # the next period
        self.assertAlmostEqual(
            self.pcct.history['RiskyNow'][1][0], 0.8950304697526602
        )

        self.assertAlmostEqual(
            self.pcct.history['RportNow'][1][0], 0.9135595661654792
        )

        self.assertAlmostEqual(
            self.pcct.history['AdjustNow'][1][0], 1.0
        )

        self.assertAlmostEqual(
            self.pcct.history['PermShkNow'][1][0], 1.0050166461586711
        )

        self.assertAlmostEqual(
            self.pcct.history['bNrmNow'][1][0], 7.293439643953855
        )

        self.assertAlmostEqual(
            self.pcct.history['mNrmNow'][1][0], 8.287859049575047
        )

        self.assertAlmostEqual(
            self.pcct.history['cNrmNow'][1][0], 1.5773607434989751
        )

        self.assertAlmostEqual(
            self.pcct.history['ShareNow'][1][0], 0.9337608822146805
        )

        self.assertAlmostEqual(
            self.pcct.history['aNrmNow'][1][0], 6.710498306076072
        )

        self.assertAlmostEqual(
            self.pcct.history['aNrmNow'][15][0], 5.304746367434934
        )

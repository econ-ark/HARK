import unittest

import numpy as np

import HARK.ConsumptionSaving.ConsPortfolioModel as cpm


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
            self.pcct.solution[0].cFuncAdj(10).tolist(), 1.6996557721625785
        )

        self.assertAlmostEqual(
            self.pcct.solution[0].ShareFuncAdj(10).tolist(), 0.8498496999408691
        )

    def test_sim_one_period(self):
        self.pcct.T_sim = 30
        self.pcct.AgentCount = 10
        self.pcct.track_vars += ["aNrm"]
        self.pcct.initialize_sim()

        self.assertFalse(np.any(self.pcct.shocks["Adjust"]))

        self.pcct.sim_one_period()

        self.assertAlmostEqual(self.pcct.controls["Share"][0], 0.8627164488246847)
        self.assertAlmostEqual(self.pcct.controls["cNrm"][0], 1.67874799)


class SimulatePortfolioConsumerTypeTestCase(PortfolioConsumerTypeTestCase):
    def test_simulation(self):
        self.pcct.T_sim = 30
        self.pcct.AgentCount = 10
        self.pcct.track_vars += [
            "mNrm",
            "cNrm",
            "Share",
            "aNrm",
            "Risky",
            "Rport",
            "Adjust",
            "PermShk",
            "bNrm",
        ]
        self.pcct.initialize_sim()

        self.pcct.simulate()

        self.assertAlmostEqual(self.pcct.history["mNrm"][0][0], 9.70233892)

        self.assertAlmostEqual(self.pcct.history["cNrm"][0][0], 1.6787479894848298)

        self.assertAlmostEqual(self.pcct.history["Share"][0][0], 0.8627164488246847)

        self.assertAlmostEqual(self.pcct.history["aNrm"][0][0], 8.023590930905383)

        self.assertAlmostEqual(self.pcct.history["Adjust"][0][0], 1.0)

        # the next period
        self.assertAlmostEqual(self.pcct.history["Risky"][1][0], 0.8950304697526602)

        self.assertAlmostEqual(self.pcct.history["Rport"][1][0], 0.9135595661654792)

        self.assertAlmostEqual(self.pcct.history["Adjust"][1][0], 1.0)

        self.assertAlmostEqual(self.pcct.history["PermShk"][1][0], 1.0050166461586711)

        self.assertAlmostEqual(self.pcct.history["bNrm"][1][0], 7.293439643953855)

        self.assertAlmostEqual(self.pcct.history["mNrm"][1][0], 8.287859049575047)

        self.assertAlmostEqual(self.pcct.history["cNrm"][1][0], 1.5773607434989751)

        self.assertAlmostEqual(self.pcct.history["Share"][1][0], 0.9337608822146805)

        self.assertAlmostEqual(self.pcct.history["aNrm"][1][0], 6.710498306076072)

        self.assertAlmostEqual(self.pcct.history["aNrm"][15][0], 5.304746367434934)


class testPortfolioConsumerTypeSticky(unittest.TestCase):
    def test_sticky(self):
        # Make another example type, but this one can only update their risky portfolio
        # share in any particular period with 15% probability.
        init_sticky_share = cpm.init_portfolio.copy()
        init_sticky_share["AdjustPrb"] = 0.15

        # Create portfolio choice consumer type
        self.sticky = cpm.PortfolioConsumerType(**init_sticky_share)
        self.sticky.cycles = 0

        # Solve the model under the given parameters

        self.sticky.solve()

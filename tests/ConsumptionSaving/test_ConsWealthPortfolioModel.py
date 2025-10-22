import unittest
from tests import HARK_PRECISION

from HARK.ConsumptionSaving.ConsWealthPortfolioModel import WealthPortfolioConsumerType


class testWealthPortfolioConsumerType(unittest.TestCase):
    def setUp(self):
        self.agent = WealthPortfolioConsumerType(vFuncBool=True)
        self.agent.solve()

    def test_cFunc(self):
        cFunc = self.agent.solution[0].cFuncAdj
        mNrm = 10.0
        cNrm = cFunc(mNrm)
        self.assertAlmostEqual(cNrm, 4.21636, places=HARK_PRECISION)

    def test_vFunc(self):
        vFunc = self.agent.solution[0].vFuncAdj
        mNrm = 2.0
        v = vFunc(mNrm)
        self.assertAlmostEqual(v, -0.25811, places=HARK_PRECISION)

    def test_simulation(self):
        self.agent.T_sim = 10
        self.agent.track_vars = ["cNrm", "aNrm", "Share"]
        self.agent.initialize_sim()
        self.agent.simulate()

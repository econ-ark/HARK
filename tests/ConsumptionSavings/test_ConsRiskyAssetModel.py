import unittest
from HARK.tests import HARK_PRECISION
from HARK.ConsumptionSaving.ConsRiskyAssetModel import IndShockRiskyAssetConsumerType


class testRiskyAssetConsumerType(unittest.TestCase):
    def setUp(self):
        self.agent = IndShockRiskyAssetConsumerType()
        self.agent.vFuncBool = False
        self.agent.solve()

    def test_solution(self):
        cFunc = self.agent.solution[0].cFunc
        mNrm = 10.0
        self.assertAlmostEqual(cFunc(mNrm).tolist(), 5.637216, places=HARK_PRECISION)

    # TODO: Turn this on after solver overhaul branch is merged
    # def test_value(self):
    #     vFunc = self.agent.solution[0].vFunc
    #     mNrm = 10.0
    #     self.assertAlmostEqual(vFunc(mNrm), -0.0000, places=HARK_PRECISION)

    def test_simulation(self):
        self.agent.T_sim = 10
        self.agent.track_vars = ["mNrm", "cNrm", "aNrm"]
        self.agent.make_shock_history()
        self.agent.initialize_sim()
        self.agent.simulate()

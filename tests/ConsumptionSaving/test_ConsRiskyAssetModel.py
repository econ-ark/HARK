import unittest
from tests import HARK_PRECISION
from HARK.ConsumptionSaving.ConsRiskyAssetModel import IndShockRiskyAssetConsumerType


class testBasicRiskyAssetConsumerType(unittest.TestCase):
    def setUp(self):
        self.agent = IndShockRiskyAssetConsumerType(vFuncBool=True)
        self.agent.solve()

    def test_solution(self):
        cFunc = self.agent.solution[0].cFunc
        mNrm = 10.0
        self.assertAlmostEqual(cFunc(mNrm).tolist(), 5.637216, places=HARK_PRECISION)

    def test_value(self):
        vFunc = self.agent.solution[0].vFunc
        mNrm = 10.0
        self.assertAlmostEqual(vFunc(mNrm), -0.3447, places=HARK_PRECISION)

    def test_simulation(self):
        self.agent.T_sim = 10
        self.agent.track_vars = ["mNrm", "cNrm", "aNrm"]
        self.agent.make_shock_history()
        self.agent.initialize_sim()
        self.agent.simulate()

    def test_zero_inc_unemp(self):
        AltType = IndShockRiskyAssetConsumerType(IncUnemp=0.0)
        AltType.solve()


class testCubicRiskyAssetConsumerType(unittest.TestCase):
    def setUp(self):
        self.agent = IndShockRiskyAssetConsumerType(CubicBool=True)
        self.agent.solve()

    def test_solution(self):
        cFunc = self.agent.solution[0].cFunc
        mNrm = 10.0
        self.assertAlmostEqual(cFunc(mNrm).tolist(), 5.637216, places=HARK_PRECISION)

    def test_simulation(self):
        self.agent.T_sim = 10
        self.agent.track_vars = ["mNrm", "cNrm", "aNrm"]
        self.agent.make_shock_history()
        self.agent.initialize_sim()
        self.agent.simulate()


class testNonIndeptRiskyAssetConsumerType(unittest.TestCase):
    def setUp(self):
        self.agent = IndShockRiskyAssetConsumerType(
            IndepDstnBool=False, CubicBool=True, vFuncBool=True
        )
        self.agent.solve()

    def test_solution(self):
        cFunc = self.agent.solution[0].cFunc
        mNrm = 10.0
        self.assertAlmostEqual(cFunc(mNrm).tolist(), 5.637216, places=HARK_PRECISION)

    def test_simulation(self):
        self.agent.T_sim = 10
        self.agent.track_vars = ["mNrm", "cNrm", "aNrm"]
        self.agent.make_shock_history()
        self.agent.initialize_sim()
        self.agent.simulate()


class testPortChoiceConsumerType(unittest.TestCase):
    def setUp(self):
        self.agent = IndShockRiskyAssetConsumerType(vFuncBool=True, PortfolioBool=True)
        self.agent.solve()

    def test_solution(self):
        cFunc = self.agent.solution[0].cFunc
        mNrm = 10.0
        self.assertAlmostEqual(cFunc(mNrm).tolist(), 5.637216, places=HARK_PRECISION)

    def test_value(self):
        vFunc = self.agent.solution[0].vFunc
        mNrm = 10.0
        self.assertAlmostEqual(vFunc(mNrm), -0.3447, places=HARK_PRECISION)

    def test_simulation(self):
        self.agent.T_sim = 10
        self.agent.track_vars = ["mNrm", "cNrm", "aNrm"]
        self.agent.make_shock_history()
        self.agent.initialize_sim()
        self.agent.simulate()


class testNonIndepPortChoiceConsumerType(unittest.TestCase):
    def setUp(self):
        self.agent = IndShockRiskyAssetConsumerType(
            IndepDstnBool=False,
            PortfolioBool=True,
            vFuncBool=True,
        )
        self.agent.solve()

    def test_solution(self):
        cFunc = self.agent.solution[0].cFunc
        mNrm = 10.0
        self.assertAlmostEqual(cFunc(mNrm).tolist(), 5.637216, places=HARK_PRECISION)

    def test_value(self):
        vFunc = self.agent.solution[0].vFunc
        mNrm = 10.0
        self.assertAlmostEqual(vFunc(mNrm), -0.3447, places=HARK_PRECISION)

    def test_simulation(self):
        self.agent.T_sim = 10
        self.agent.track_vars = ["mNrm", "cNrm", "aNrm"]
        self.agent.make_shock_history()
        self.agent.initialize_sim()
        self.agent.simulate()

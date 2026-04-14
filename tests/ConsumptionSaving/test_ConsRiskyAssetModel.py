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
        self.agent = IndShockRiskyAssetConsumerType(
            vFuncBool=True,
            RiskyShareFixed=None,
            ShareAugFac=2,
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


class testNonIndepPortChoiceConsumerType(unittest.TestCase):
    def setUp(self):
        self.agent = IndShockRiskyAssetConsumerType(
            IndepDstnBool=False,
            RiskyShareFixed=None,
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


class testZeroIncShkPortChoiceConsumerType(unittest.TestCase):
    def setUp(self):
        self.agent = IndShockRiskyAssetConsumerType(
            RiskyShareFixed=None,
            CubicBool=True,
            IncUnemp=0.0,
        )
        self.agent.solve()

        self.agent_alt = IndShockRiskyAssetConsumerType(
            RiskyShareFixed=None,
            CubicBool=True,
            IncUnemp=0.0,
            IndepDstnBool=False,
        )
        self.agent.solve()
        self.agent_alt.solve()

    def test_solution(self):
        cFunc = self.agent.solution[0].cFunc
        mNrm = 2.0
        self.assertAlmostEqual(cFunc(mNrm).tolist(), 1.43258, places=HARK_PRECISION)

    def test_solution_alt(self):
        cFunc = self.agent_alt.solution[0].cFunc
        mNrm = 2.0
        self.assertAlmostEqual(cFunc(mNrm).tolist(), 1.43258, places=HARK_PRECISION)


class testZeroIncShkRiskyAssetConsumerType(unittest.TestCase):
    def setUp(self):
        self.agent = IndShockRiskyAssetConsumerType(
            CubicBool=True,
            IncUnemp=0.0,
        )
        self.agent.solve()

        self.agent_alt = IndShockRiskyAssetConsumerType(
            CubicBool=True,
            IncUnemp=0.0,
            IndepDstnBool=False,
        )
        self.agent.solve()
        self.agent_alt.solve()

    def test_solution(self):
        cFunc = self.agent.solution[0].cFunc
        mNrm = 2.0
        self.assertAlmostEqual(cFunc(mNrm).tolist(), 1.43258, places=HARK_PRECISION)

    def test_solution_alt(self):
        cFunc = self.agent_alt.solution[0].cFunc
        mNrm = 2.0
        self.assertAlmostEqual(cFunc(mNrm).tolist(), 1.43258, places=HARK_PRECISION)


class testInvalidRiskyAssetType(unittest.TestCase):
    def test_BoroCnstArt(self):
        agent = IndShockRiskyAssetConsumerType(BoroCnstArt=-1.0)
        self.assertRaises(ValueError, agent.solve)

        agent = IndShockRiskyAssetConsumerType(BoroCnstArt=-1.0, RiskyShareFixed=None)
        self.assertRaises(ValueError, agent.solve)

    def test_constructors(self):
        self.assertRaises(
            AttributeError, IndShockRiskyAssetConsumerType, AdjustPrb=[0.8, 0.9]
        )

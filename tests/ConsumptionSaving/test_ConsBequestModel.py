import unittest
from tests import HARK_PRECISION
from HARK.ConsumptionSaving.ConsBequestModel import (
    BequestWarmGlowConsumerType,
    BequestWarmGlowPortfolioType,
)


class testWarmGlowConsumerType(unittest.TestCase):
    def setUp(self):
        self.agent = BequestWarmGlowConsumerType(BeqFac=1.0)
        self.agent.vFuncBool = True
        self.agent.solve()

    def test_solution(self):
        cFunc = self.agent.solution[0].cFunc
        mNrm = 10.0
        self.assertAlmostEqual(cFunc(mNrm).tolist(), 5.56409, places=HARK_PRECISION)

    def test_value(self):
        vFunc = self.agent.solution[0].vFunc
        mNrm = 10.0
        self.assertAlmostEqual(vFunc(mNrm), -0.35313, places=HARK_PRECISION)

    def test_simulation(self):
        self.agent.T_sim = 10
        self.agent.track_vars = ["mNrm", "cNrm", "aNrm"]
        self.agent.make_shock_history()
        self.agent.initialize_sim()
        self.agent.simulate()

    def test_cubic(self):
        CubicType = BequestWarmGlowConsumerType(BeqFac=1.0, CubicBool=True)
        CubicType.solve()
        cFunc = CubicType.solution[0].cFunc
        mNrm = 10.0
        self.assertAlmostEqual(cFunc(mNrm).tolist(), 5.56409, places=HARK_PRECISION)


class testBequestWarmGlowPortfolioType(unittest.TestCase):
    def setUp(self):
        self.agent = BequestWarmGlowPortfolioType(
            BeqFac=1.0, BeqFacTerm=1.0, vFuncBool=True
        )
        self.agent.solve()

    def test_consumption(self):
        cFunc = self.agent.solution[0].cFuncAdj
        mNrm = 10.0
        self.assertAlmostEqual(cFunc(mNrm).tolist(), 2.19432, places=HARK_PRECISION)

    def test_share(self):
        ShareFunc = self.agent.solution[0].ShareFuncAdj
        mNrm = 10.0
        self.assertAlmostEqual(ShareFunc(mNrm).tolist(), 0.75504, places=HARK_PRECISION)

    def test_value(self):
        vFunc = self.agent.solution[0].vFuncAdj
        mNrm = 10.0
        self.assertAlmostEqual(vFunc(mNrm), -0.15244, places=HARK_PRECISION)

    def test_simulation(self):
        self.agent.T_sim = 10
        self.agent.track_vars = ["mNrm", "cNrm", "aNrm", "Share"]
        self.agent.make_shock_history()
        self.agent.initialize_sim()
        self.agent.simulate()

    def test_no_value(self):
        basic_type = BequestWarmGlowPortfolioType(BeqFac=1.0, BeqFacTerm=1.0)
        basic_type.solve()  # this just covers a trivial case

    def test_advanced(self):
        OtherType = BequestWarmGlowPortfolioType(
            BeqFac=1.0,
            BeqFacTerm=1.0,
            AdjustPrb=0.6,
            vFuncBool=True,
            DiscreteShareBool=True,
        )
        OtherType.solve()
        mNrm = 10.0
        cFunc = OtherType.solution[0].cFuncAdj
        self.assertAlmostEqual(cFunc(mNrm), 2.18674, places=HARK_PRECISION)

    def test_invalid(self):
        BadType = BequestWarmGlowPortfolioType(BeqFac=1.0, BoroCnstArt=-1.0)
        self.assertRaises(ValueError, BadType.solve)

        BadType = BequestWarmGlowPortfolioType(DiscreteShareBool=True, vFuncBool=False)
        self.assertRaises(ValueError, BadType.solve)

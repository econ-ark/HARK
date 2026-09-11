import unittest

import numpy as np

from tests import HARK_PRECISION
from HARK.ConsumptionSaving.ConsBequestModel import (
    BequestWarmGlowConsumerType,
    BequestWarmGlowPortfolioType,
)


class testWarmGlowConsumerType(unittest.TestCase):
    def setUp(self):
        self.agent = BequestWarmGlowConsumerType()
        self.agent.vFuncBool = True
        self.agent.solve()

    def test_solution(self):
        cFunc = self.agent.solution[0].cFunc
        mNrm = 10.0
        self.assertAlmostEqual(cFunc(mNrm).tolist(), 1.68867, places=HARK_PRECISION)

    def test_value(self):
        vFunc = self.agent.solution[0].vFunc
        mNrm = 10.0
        self.assertAlmostEqual(vFunc(mNrm), -4.01233, places=HARK_PRECISION)

    def test_simulation(self):
        self.agent.T_sim = 10
        self.agent.track_vars = ["mNrm", "cNrm", "aNrm"]
        self.agent.make_shock_history()
        self.agent.initialize_sim()
        self.agent.simulate()

    def test_cubic(self):
        CubicType = BequestWarmGlowConsumerType(CubicBool=True)
        CubicType.solve()
        cFunc = CubicType.solution[0].cFunc
        mNrm = 10.0
        self.assertAlmostEqual(cFunc(mNrm).tolist(), 1.688675, places=HARK_PRECISION)

    def test_borrowing(self):
        ThisType = BequestWarmGlowConsumerType(BeqFac=1.0, BoroCnstArt=None)
        ThisType.solve()


class testBequestWarmGlowPortfolioType(unittest.TestCase):
    def setUp(self):
        self.agent = BequestWarmGlowPortfolioType(vFuncBool=True)
        self.agent.solve()

    def test_consumption(self):
        cFunc = self.agent.solution[0].cFuncAdj
        mNrm = 10.0
        self.assertAlmostEqual(cFunc(mNrm).tolist(), 1.70232, places=HARK_PRECISION)

    def test_share(self):
        ShareFunc = self.agent.solution[0].ShareFuncAdj
        mNrm = 10.0
        self.assertAlmostEqual(ShareFunc(mNrm).tolist(), 0.96250, places=HARK_PRECISION)

    def test_value(self):
        vFunc = self.agent.solution[0].vFuncAdj
        mNrm = 10.0
        self.assertAlmostEqual(vFunc(mNrm), -3.94804, places=HARK_PRECISION)

    def test_zero_inc_shk(self):
        ZeroShkType = BequestWarmGlowPortfolioType(BeqInt=0.0, IncUnemp=0.0)
        ZeroShkType.solve()
        ZeroShkType.unpack("cFuncAdj")
        mNrm = 2.0
        self.assertAlmostEqual(
            ZeroShkType.cFuncAdj[0](mNrm), 0.42861, places=HARK_PRECISION
        )

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
            AdjustPrb=0.6,
            vFuncBool=True,
            DiscreteShareBool=True,
        )
        OtherType.solve()
        mNrm = 10.0
        cFunc = OtherType.solution[0].cFuncAdj
        self.assertAlmostEqual(cFunc(mNrm), 1.70249, places=HARK_PRECISION)

    def test_invalid(self):
        BadType = BequestWarmGlowPortfolioType(BeqFac=1.0, BoroCnstArt=-1.0)
        self.assertRaises(ValueError, BadType.solve)

        BadType = BequestWarmGlowPortfolioType(DiscreteShareBool=True, vFuncBool=False)
        self.assertRaises(ValueError, BadType.solve)


class testLogUtilityValue(unittest.TestCase):
    def test_value_one_period_before_terminal(self):
        """
        With log utility, value one period before the end matches the Bellman equation.

        The terminal agent splits m between consumption c and a bequest a with
        1 / c = BeqFac / (a + BeqShift), so terminal value is
        log(c) + BeqFac * log(a + BeqShift), which carries (1 + BeqFac) * log(P).
        A period earlier, value adds the discounted terminal value plus
        (1 + BeqFac) * log(PermGroFac * psi), and the warm glow
        (1 - LivPrb) * BeqFac * log(a + BeqShift) of the bequest left then (issue #75).
        """
        m = np.array([0.5, 2.0, 10.0, 19.0])
        for AgentType in (BequestWarmGlowConsumerType, BequestWarmGlowPortfolioType):
            agent = AgentType(cycles=1, CRRA=1.0, vFuncBool=True)
            agent.solve()
            solution = agent.solution[0]
            Rfree = np.atleast_1d(agent.Rfree)[0]
            if AgentType is BequestWarmGlowPortfolioType:
                ShkDstn = agent.ShockDstn[0]
                PermShk, TranShk, Risky = ShkDstn.atoms
                c = solution.cFuncAdj(m)
                Share = solution.ShareFuncAdj(m)[:, None]
                Rport = Rfree + Share * (Risky - Rfree)
                vSolved = solution.vFuncAdj(m)
            else:
                ShkDstn = agent.IncShkDstn[0]
                PermShk, TranShk = ShkDstn.atoms
                c = solution.cFunc(m)
                Rport = Rfree
                vSolved = solution.vFunc(m)
            BeqFac, BeqShift = agent.BeqFac, agent.BeqShift
            a = m - c
            growth = agent.PermGroFac[0] * PermShk
            mNext = Rport * a[:, None] / growth + TranShk
            cNext = np.minimum((mNext + BeqShift) / (1.0 + BeqFac), mNext)
            vNext = np.log(cNext) + BeqFac * np.log(mNext - cNext + BeqShift)
            vNext += (1.0 + BeqFac) * np.log(growth)
            bequest = (1.0 - agent.LivPrb[0]) * BeqFac * np.log(a + BeqShift)
            beta = agent.DiscFac * agent.LivPrb[0]
            v = np.log(c) + beta * vNext @ ShkDstn.pmv + bequest
            np.testing.assert_allclose(vSolved, v, rtol=0, atol=1e-4)

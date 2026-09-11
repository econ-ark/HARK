import unittest

import numpy as np

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

    def test_invalid(self):
        ThisType = WealthPortfolioConsumerType(BoroCnstArt=-1.0)
        self.assertRaises(ValueError, ThisType.solve)

    def test_ZeroIncShk(self):
        ThisType = WealthPortfolioConsumerType(IncUnemp=0.0)
        ThisType.solve()
        ThisType.unpack("cFuncAdj")
        self.assertAlmostEqual(
            ThisType.cFuncAdj[0](2.0), 0.97694, places=HARK_PRECISION
        )


class testLogUtilityValue(unittest.TestCase):
    def test_value_one_period_before_terminal(self):
        """
        With log utility, value one period before the end matches the Bellman equation.

        With WealthShare = 0 utility is log(c). At the chosen share,
        m' = Rport * a / (PermGroFac * psi) + theta, and continuation value is
        log(m') + log(PermGroFac * psi), where permanent income growth enters
        additively (issue #75).
        """
        agent = WealthPortfolioConsumerType(
            CRRA=1.0, vFuncBool=True, WealthShare=0.0, WealthShift=0.0
        )
        agent.solve()
        solution = agent.solution[0]
        PermShk, TranShk, Risky = agent.ShockDstn[0].atoms
        m = np.array([0.5, 2.0, 10.0, 19.0])
        c = solution.cFuncAdj(m)
        Share = solution.ShareFuncAdj(m)[:, None]
        Rport = agent.Rfree[0] + Share * (Risky - agent.Rfree[0])
        growth = agent.PermGroFac[0] * PermShk
        mNext = Rport * (m - c)[:, None] / growth + TranShk
        vNext = (np.log(mNext) + np.log(growth)) @ agent.ShockDstn[0].pmv
        v = np.log(c) + agent.DiscFac * agent.LivPrb[0] * vNext
        np.testing.assert_allclose(solution.vFuncAdj(m), v, rtol=0, atol=1e-4)

import unittest

import numpy as np

from tests import HARK_PRECISION

from HARK.ConsumptionSaving.ConsWealthUtilityModel import (
    WealthUtilityConsumerType,
    CapitalistSpiritConsumerType,
)


class testWealthUtilityConsumerType(unittest.TestCase):
    def setUp(self):
        self.agent = WealthUtilityConsumerType(cycles=0, vFuncBool=True)
        self.agent.solve()

    def test_cFunc(self):
        cFunc = self.agent.solution[0].cFunc
        mNrm = 5.0
        cNrm = cFunc(mNrm)
        self.assertAlmostEqual(cNrm, 0.94275, places=HARK_PRECISION)

    def test_vFunc(self):
        vFunc = self.agent.solution[0].vFunc
        mNrm = 5.0
        v = vFunc(mNrm)
        self.assertAlmostEqual(v, -11.2017, places=HARK_PRECISION)

    def test_sim(self):
        self.agent.T_sim = 100
        self.agent.AgentCount = 1000
        self.agent.initialize_sim()
        self.agent.simulate()


class testWealthUtilityLogValue(unittest.TestCase):
    def test_value_one_period_before_terminal(self):
        """
        With log utility, value one period before the end matches the Bellman equation.

        Utility is the log of x = c**(1 - WealthShare) * (a + WealthShift)**WealthShare,
        which carries log(P) with weight one, so continuation value adds
        log(PermGroFac * psi) to the terminal value log(c_T(m')) (issue #75).
        """
        agent = WealthUtilityConsumerType(cycles=1, CRRA=1.0, vFuncBool=True)
        agent.solve()
        solution = agent.solution[0]
        PermShk, TranShk = agent.IncShkDstn[0].atoms
        growth = agent.PermGroFac[0] * PermShk
        m = solution.mNrmMin + np.array([2.0, 10.0, 19.0])
        c = solution.cFunc(m)
        a = m - c
        mNext = agent.Rfree[0] * a[:, None] / growth + TranShk
        vNext = np.log(agent.solution_terminal.cFunc(mNext)) + np.log(growth)
        x = (
            c ** (1.0 - agent.WealthShare)
            * (a + agent.WealthShift) ** agent.WealthShare
        )
        beta = agent.DiscFac * agent.LivPrb[0]
        v = np.log(x) + beta * vNext @ agent.IncShkDstn[0].pmv
        np.testing.assert_allclose(solution.vFunc(m), v, rtol=0, atol=2e-4)


class testWealthUtilityOddParams(unittest.TestCase):
    def setUp(self):
        self.agent1 = WealthUtilityConsumerType(cycles=0, WealthShift=2.0)
        self.agent1.solve()

        self.agent2 = WealthUtilityConsumerType(cycles=0, WealthShare=0.0)
        self.agent2.solve()

    def test_cFunc(self):
        cFunc = self.agent1.solution[0].cFunc
        mNrm = 5.0
        cNrm = cFunc(mNrm)
        self.assertAlmostEqual(cNrm, 0.98474, places=HARK_PRECISION)

    def test_trivial(self):
        cFunc = self.agent2.solution[0].cFunc
        mNrm = 5.0
        cNrm = cFunc(mNrm)
        self.assertAlmostEqual(cNrm, 1.37170, places=HARK_PRECISION)


class testCapitalistSpiritConsumerType(unittest.TestCase):
    def setUp(self):
        self.agent = CapitalistSpiritConsumerType(vFuncBool=True)
        self.agent.solve()

    def test_cFunc(self):
        cFunc = self.agent.solution[0].cFunc
        mLvl = 5.0
        pLvl = 1.7
        cNrm = cFunc(mLvl, pLvl)
        self.assertAlmostEqual(cNrm, 2.24766, places=HARK_PRECISION)

    def test_vFunc(self):
        vFunc = self.agent.solution[0].vFunc
        mLvl = 5.0
        pLvl = 1.7
        v = vFunc(mLvl, pLvl)
        self.assertAlmostEqual(v, -1.35278, places=HARK_PRECISION)

    def test_no_vFunc(self):
        self.agent.assign_parameters(vFuncBool=False)
        self.agent.solve()

    def test_sim(self):
        self.agent.T_sim = 100
        self.agent.AgentCount = 1000
        self.agent.initialize_sim()
        self.agent.simulate()


class testInvalidParams(unittest.TestCase):
    def test_invalid(self):
        MyType = WealthUtilityConsumerType(cycles=0, CubicBool=True)
        self.assertRaises(NotImplementedError, MyType.solve)

        MyType = CapitalistSpiritConsumerType(WealthCurve=1.3)
        self.assertRaises(ValueError, MyType.solve)

        MyType = CapitalistSpiritConsumerType(WealthCurve=-0.5)
        self.assertRaises(ValueError, MyType.solve)

        MyType = CapitalistSpiritConsumerType(WealthFac=-1.5)
        self.assertRaises(ValueError, MyType.solve)

        MyType = CapitalistSpiritConsumerType(WealthShift=-2.0)
        self.assertRaises(ValueError, MyType.solve)

        MyType = CapitalistSpiritConsumerType(vFuncBool=True, WealthCurve=0.25)
        self.assertRaises(ValueError, MyType.solve)

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


def make_log_capitalist(periods):
    """Solve a CapitalistSpiritConsumerType with log utility over `periods` periods."""
    time_varying = {
        "PermGroFac": [1.0],
        "PermShkStd": [0.1],
        "TranShkStd": [0.1],
        "LivPrb": [0.98],
        "Rfree": [1.03],
    }
    agent = CapitalistSpiritConsumerType(
        CRRA=1.0,
        vFuncBool=True,
        cycles=1,
        T_cycle=periods,
        **{key: val * periods for key, val in time_varying.items()},
    )
    agent.solve()
    return agent


class testCapitalistSpiritLogValue(unittest.TestCase):
    def test_value_one_period_before_last(self):
        """
        With log utility, value one period before the last matches the Bellman equation.

        In the last period 1 / c = WealthFac * (a + WealthShift)**(-nu), where
        nu = WealthCurve, found here by bisection. Value there is log(c) plus the
        wealth utility of a. A period earlier, value adds the discounted expectation
        of that value at mLvl' = Rfree * a + pLvl' * theta (issue #75).
        """
        agent = make_log_capitalist(2)
        nu = agent.WealthCurve
        fac, shift = agent.WealthFac, agent.WealthShift

        def warm(a):
            return fac * (a + shift) ** (1.0 - nu) / (1.0 - nu)

        def v_last(m):
            lo, hi = np.zeros_like(m), m.copy()
            for _ in range(100):
                c = 0.5 * (lo + hi)
                high = 1.0 / c > fac * (m - c + shift) ** (-nu)
                lo, hi = np.where(high, c, lo), np.where(high, hi, c)
            c = 0.5 * (lo + hi)
            return np.log(c) + warm(m - c)

        solution = agent.solution[0]
        pLvl = agent.pLvlGrid[0][agent.pLvlGrid[0].size // 2]
        mLvl = solution.mLvlMin(pLvl) + pLvl * np.array([0.5, 2.0, 5.0, 20.0])
        pLvls = np.full_like(mLvl, pLvl)
        np.testing.assert_allclose(
            agent.solution[1].vFunc(mLvl, pLvls), v_last(mLvl), rtol=0, atol=1e-4
        )
        PermShk, TranShk = agent.IncShkDstn[0].atoms
        c = solution.cFunc(mLvl, pLvls)
        a = mLvl - c
        pLvlNext = agent.pLvlNextFunc[0](pLvl) * PermShk
        mLvlNext = agent.Rfree[0] * a[:, None] + pLvlNext * TranShk
        beta = agent.DiscFac * agent.LivPrb[0]
        v = np.log(c) + warm(a) + beta * v_last(mLvlNext) @ agent.IncShkDstn[0].pmv
        np.testing.assert_allclose(solution.vFunc(mLvl, pLvls), v, rtol=0, atol=1e-4)

    def test_long_horizon_envelope(self):
        """
        With log utility, value over 20 periods stays finite and v'(m) = u'(c(m)).

        Wealth utility makes value grow each period, so without a value scale the
        pseudo-inverse exp(v) overflows and value turns into NaN.
        """
        agent = make_log_capitalist(20)
        solution = agent.solution[0]
        pLvl = agent.pLvlGrid[0][agent.pLvlGrid[0].size // 2]
        mLvl = solution.mLvlMin(pLvl) + pLvl * np.array([0.5, 2.0, 5.0, 20.0, 40.0])
        pLvls = np.full_like(mLvl, pLvl)
        step = 1e-5
        vP = (
            solution.vFunc(mLvl + step, pLvls) - solution.vFunc(mLvl - step, pLvls)
        ) / (2 * step)
        np.testing.assert_allclose(vP * solution.cFunc(mLvl, pLvls), 1.0, atol=1e-2)

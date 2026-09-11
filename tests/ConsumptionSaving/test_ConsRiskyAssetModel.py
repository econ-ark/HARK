import unittest

import numpy as np

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

    def test_log_utility_value_one_period_before_terminal(self):
        """
        With log utility, value one period before the end matches the Bellman equation.

        With a fixed risky share, m' = Rport * a / (PermGroFac * psi) + theta, and
        continuation value is log(m') + log(PermGroFac * psi), where permanent
        income growth enters additively (issue #75). The independent and the joint
        shock distributions take different solver paths, so both are checked.
        """
        for IndepDstnBool in (True, False):
            agent = IndShockRiskyAssetConsumerType(
                cycles=1, CRRA=1.0, vFuncBool=True, IndepDstnBool=IndepDstnBool
            )
            agent.solve()
            solution = agent.solution[0]
            PermShk, TranShk, Risky = agent.ShockDstn[0].atoms
            Share = np.atleast_1d(agent.RiskyShareFixed)[0]
            Rport = Share * Risky + (1.0 - Share) * agent.Rfree[0]
            growth = agent.PermGroFac[0] * PermShk
            m = solution.mNrmMin + np.array([0.5, 2.0, 10.0, 19.0])
            c = solution.cFunc(m)
            mNext = Rport * (m - c)[:, None] / growth + TranShk
            vNext = (np.log(mNext) + np.log(growth)) @ agent.ShockDstn[0].pmv
            v = np.log(c) + agent.DiscFac * agent.LivPrb[0] * vNext
            np.testing.assert_allclose(solution.vFunc(m), v, rtol=0, atol=1e-6)


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

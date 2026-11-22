"""
A few small tests for the tractable buffer stock model.
"""

import unittest
import warnings
import numpy as np
import HARK.ConsumptionSaving.TractableBufferStockModel as Model


class FuncTest(unittest.TestCase):
    def setUp(self):
        base_primitives = {
            "UnempPrb": 0.015,
            "DiscFac": 0.9,
            "Rfree": 1.1,
            "PermGroFac": 1.05,
            "CRRA": 0.95,
        }
        test_model = Model.TractableConsumerType(**base_primitives)
        test_model.solve()
        cNrm_list = np.array(
            [
                0.0,
                0.61704,
                0.75129,
                0.82421,
                0.87326,
                0.90904,
                0.93626,
                0.95749,
                0.97432,
                0.98783,
                0.99877,
                1.04998,
                1.09884,
                1.10791,
                1.11855,
                1.13100,
                1.14550,
                1.16234,
                1.18180,
                1.20421,
                1.22989,
                1.25918,
                1.29242,
                1.32998,
                1.37222,
                1.41952,
                1.47224,
                1.53077,
            ]
        )
        self.cNrm_targ = cNrm_list
        self.agent = test_model

    def test_equalityOfSolutions(self):
        cNrm_model = np.array(self.agent.solution[0].cNrm_list)
        self.assertTrue(np.allclose(cNrm_model, self.cNrm_targ, atol=1e-08))

    def test_simulation(self):
        agent = self.agent
        agent.AgentCount = 1000
        agent.T_sim = 100
        agent.track_vars = ["cNrm", "aNrm"]
        agent.initialize_sim()
        agent.simulate()

    def test_mpc_at_target_in_unit_interval(self):
        # At this interest rate, a search started at 0 found the negative root of the
        # quadratic for the MPC at target, and the consumption function missed the
        # target.
        agent = Model.TractableConsumerType(
            UnempPrb=0.005, DiscFac=1 / 1.10, Rfree=1.07, PermGroFac=1.0, CRRA=2.0
        )
        agent.solve()
        MPCtarg = agent.MPCtarg
        self.assertTrue(0.0 < MPCtarg < 1.0)
        self.assertAlmostEqual(
            float(agent.solution[0].cFunc(agent.mTarg)), agent.cTarg, places=8
        )
        # The MPC satisfies the differentiated Euler equation at the target
        natural = (
            agent.Beth
            * agent.Rnrm
            * (
                (1.0 - agent.UnempPrb) * MPCtarg
                + agent.UnempPrb
                * (agent.cTargU / agent.cTarg) ** (-agent.CRRA - 1.0)
                * agent.PFMPC
            )
        )
        self.assertAlmostEqual(MPCtarg, natural / (1.0 + natural), places=10)

    def test_mpc_at_zero_above_old_bracket(self):
        # Here the MPC as market resources go to zero is 0.995. The search for it on
        # [PFMPC, 0.99] raised a ValueError.
        above = Model.TractableConsumerType(
            UnempPrb=0.0005, DiscFac=1 / 1.10, Rfree=1.03, PermGroFac=1.0, CRRA=1.0
        )
        above.solve()
        self.assertTrue(0.99 < above.MPCmax < 1.0)
        # At both calibrations it satisfies the equation that defines it
        for agent in (above, self.agent):
            MPCmax = agent.MPCmax
            natural = (
                agent.Beth
                * agent.Rnrm
                * agent.UnempPrb
                * agent.PFMPC
                * (agent.PFMPC * agent.Rnrm * (1.0 - MPCmax) / MPCmax)
                ** (-agent.CRRA - 1.0)
            )
            self.assertAlmostEqual(MPCmax, natural / (1.0 + natural), places=10)
            self.assertAlmostEqual(
                float(agent.solution[0].cFunc.derivative(0.0)), MPCmax, places=10
            )

    def test_mpc_slope_and_curvature_at_target(self):
        # The MMPC and MMMPC at the target solve equations that are linear in them, so
        # they are computed exactly rather than searched for: each equation's residual
        # vanishes there, to rounding relative to the size of its terms
        calibrations = (
            {},
            {"CRRA": 2.0},
            {"CRRA": 4.0, "UnempPrb": 0.03, "PermGroFac": 1.0},
            {"UnempPrb": 0.005, "DiscFac": 1 / 1.10, "Rfree": 1.07, "PermGroFac": 1.0},
        )
        for kwds in calibrations:
            agent = Model.TractableConsumerType(**kwds)
            agent.solve()
            k, k1, k2 = agent.MPCtarg, agent.MMPCtarg, agent.MMMPCtarg
            p, q, R, B, kU = (
                1.0 - agent.UnempPrb,
                agent.UnempPrb,
                agent.Rnrm,
                agent.Beth,
                agent.PFMPC,
            )
            rho, c, cU = agent.CRRA, agent.cTarg, agent.cTargU
            u2 = lambda x: -rho * x ** (-rho - 1.0)
            u3 = lambda x: rho * (rho + 1.0) * x ** (-rho - 2.0)
            u4 = lambda x: -rho * (rho + 1.0) * (rho + 2.0) * x ** (-rho - 3.0)
            terms1 = [
                k1 * u2(c),
                k**2 * u3(c),
                B * p * k * k1 * R * u2(c),
                -B * p * (1 - k) ** 2 * k1 * R**2 * u2(c),
                B * kU * q * k1 * R * u2(cU),
                -B * p * (1 - k) ** 2 * k**2 * R**2 * u3(c),
                -B * kU**2 * q * (1 - k) ** 2 * R**2 * u3(cU),
            ]
            terms2 = [
                k2 * u2(c),
                3 * k * k1 * u3(c),
                k**3 * u4(c),
                B * p * k * k2 * R * u2(c),
                3 * B * p * (1 - k) * k1**2 * R**2 * u2(c),
                -B * p * (1 - k) ** 3 * k2 * R**3 * u2(c),
                B * kU * q * k2 * R * u2(cU),
                3 * B * p * (1 - k) * k**2 * k1 * R**2 * u3(c),
                -3 * B * p * (1 - k) ** 3 * k * k1 * R**3 * u3(c),
                3 * B * kU**2 * q * (1 - k) * k1 * R**2 * u3(cU),
                -B * p * (1 - k) ** 3 * k**3 * R**3 * u4(c),
                -B * kU**3 * q * (1 - k) ** 3 * R**3 * u4(cU),
            ]
            for terms in (terms1, terms2):
                scale = sum(abs(x) for x in terms)
                self.assertLess(abs(sum(terms)) / scale, 1e-12, msg=str(kwds))

    def test_warns_when_the_stable_arm_stops_short(self):
        # With the growth patience factor this close to one, the stable arm needs more
        # points than the solver's cycles allow, and it stops short of its bounds
        PermGroFacCmp = 1.0025 / (1.0 - 0.00625)
        agent = Model.TractableConsumerType(DiscFac=0.9997 * PermGroFacCmp / 1.01)
        with self.assertWarnsRegex(UserWarning, "stable arm stopped short"):
            agent.solve()
        arm = agent.solution[0].mNrm_list[1:]  # the first point is m = 0
        self.assertTrue(arm[0] > agent.mLowerBnd and arm[-1] < agent.mUpperBnd)
        # Where the arm reaches its bounds, nothing is said
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Model.TractableConsumerType().solve()
        self.assertFalse(any("stable arm" in str(w.message) for w in caught))

    def test_invalid(self):
        # Fail impatience conditions
        ThisType = Model.TractableConsumerType(PermGroFac=0.6)
        self.assertRaises(ValueError, ThisType.solve)

        ThisType = Model.TractableConsumerType(DiscFac=1.2)
        self.assertRaises(ValueError, ThisType.solve)

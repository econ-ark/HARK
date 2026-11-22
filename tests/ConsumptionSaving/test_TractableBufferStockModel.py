"""
A few small tests for the tractable buffer stock model.
"""

import unittest
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

    def test_invalid(self):
        # Fail impatience conditions
        ThisType = Model.TractableConsumerType(PermGroFac=0.6)
        self.assertRaises(ValueError, ThisType.solve)

        ThisType = Model.TractableConsumerType(DiscFac=1.2)
        self.assertRaises(ValueError, ThisType.solve)

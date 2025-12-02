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

"""
Created on Thu Mar 24 11:01:50 2016

@author: kaufmana
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
        return np.array(test_model.solution[0].cNrm_list), cNrm_list

    def test_equalityOfSolutions(self):
        results = self.setUp()
        self.assertTrue(np.allclose(results[0], results[1], atol=1e-08))


if __name__ == "__main__":
    unittest.main()

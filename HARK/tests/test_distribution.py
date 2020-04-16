import numpy as np
import unittest

import HARK.distribution as distribution

class DistributionTests(unittest.TestCase):
    '''
    Tests for simulation.py sampling distributions
    with default seed.
    '''

    def test_drawDiscrete(self):
        self.assertEqual(
            distribution.DiscreteDistribution(
                np.ones(1),
                np.zeros(1)
            ).drawDiscrete(1)[0],
            0)


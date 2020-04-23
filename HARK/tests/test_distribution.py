import numpy as np
import unittest

import HARK.distribution as distribution

class DiscreteDistributionTests(unittest.TestCase):
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


class DistributionClassTests(unittest.TestCase):
    '''
    Tests for simulation.py sampling distributions
    with default seed.
    '''

    def test_drawMeanOneLognormal(self):
        self.assertEqual(
            distribution.drawMeanOneLognormal(1)[0],
            3.5397367004222002)

    def test_Lognormal(self):
        self.assertEqual(
            distribution.Lognormal().draw(1)[0],
            5.836039190663969)

    def test_Normal(self):
        self.assertEqual(
            distribution.Normal().draw(1)[0],
            1.764052345967664)

    def test_Weibull(self):
        self.assertEqual(
            distribution.Weibull().draw(1)[0],
            0.79587450816311)

    def test_Uniform(self):
        self.assertEqual(
            distribution.Uniform().draw(1)[0],
            0.5488135039273248)

    def test_Bernoulli(self):
        self.assertEqual(
            distribution.Bernoulli().draw(1)[0],
            False)

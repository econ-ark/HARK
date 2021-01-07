import numpy as np
import unittest

import HARK.distribution as distribution


class DiscreteDistributionTests(unittest.TestCase):
    """
    Tests for distribution.py sampling distributions
    with default seed.
    """

    def test_drawDiscrete(self):
        self.assertEqual(
            distribution.DiscreteDistribution(np.ones(1), np.zeros(1)).drawDiscrete(1)[
                0
            ],
            0,
        )


class DistributionClassTests(unittest.TestCase):
    """
    Tests for distribution.py sampling distributions
    with default seed.
    """

    def test_drawMeanOneLognormal(self):
        self.assertEqual(distribution.MeanOneLogNormal().draw(1)[0], 3.5397367004222002)

    def test_Lognormal(self):

        dist = distribution.Lognormal()

        self.assertEqual(dist.draw(1)[0], 5.836039190663969)

        dist.draw(100)
        dist.reset()

        self.assertEqual(dist.draw(1)[0], 5.836039190663969)

    def test_Normal(self):
        dist = distribution.Normal()

        self.assertEqual(dist.draw(1)[0], 1.764052345967664)

        dist.draw(100)
        dist.reset()

        self.assertEqual(dist.draw(1)[0], 1.764052345967664)

    def test_Weibull(self):
        self.assertEqual(distribution.Weibull().draw(1)[0], 0.79587450816311)

    def test_Uniform(self):
        uni = distribution.Uniform()

        self.assertEqual(distribution.Uniform().draw(1)[0], 0.5488135039273248)

        self.assertEqual(
            distribution.calcExpectation(uni.approx(10))[0],
            0.5
        )

    def test_Bernoulli(self):
        self.assertEqual(distribution.Bernoulli().draw(1)[0], False)


class MarkovProcessTests(unittest.TestCase):
    """
    Tests for MarkovProcess class.
    """

    def test_draw(self):

        mrkv_array = np.array(
            [[.75, .25],[0.1, 0.9]]
        )

        mp = distribution.MarkovProcess(mrkv_array)

        new_state = mp.draw(np.zeros(100).astype(int))

        self.assertEqual(new_state.sum(), 20)

        new_state = mp.draw(new_state)

        self.assertEqual(new_state.sum(), 39)

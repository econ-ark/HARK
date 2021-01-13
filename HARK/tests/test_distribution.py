import numpy as np
import unittest

import HARK.distribution as distribution

from HARK.distribution import (
    Bernoulli,
    DiscreteDistribution,
    Lognormal,
    MeanOneLogNormal,
    Normal,
    Uniform,
    Weibull,
    calcExpectation,
    combineIndepDstns
)


class DiscreteDistributionTests(unittest.TestCase):
    """
    Tests for distribution.py sampling distributions
    with default seed.
    """

    def test_drawDiscrete(self):
        self.assertEqual(
            DiscreteDistribution(
                np.ones(1),
                np.zeros(1)).drawDiscrete(1)[
                0
            ],
            0,
        )

    def test_calcExpectation(self):
        dd_0_1_20 = Normal().approx(20)
        dd_1_1_40 = Normal(mu = 1).approx(40)
        dd_10_10_100 = Normal(mu = 10, sigma = 10).approx(100)

        ce1 = calcExpectation(dd_0_1_20)
        ce2 = calcExpectation(dd_1_1_40)
        ce3 = calcExpectation(dd_10_10_100)

        self.assertAlmostEqual(ce1, 0.0)
        self.assertAlmostEqual(ce2, 1.0)
        self.assertAlmostEqual(ce3, 10.0)

        ce4= calcExpectation(
            dd_0_1_20,
            lambda x: 2 ** x
        )

        self.assertAlmostEqual(ce4, 1.27153712)

        ce5 = calcExpectation(
            dd_1_1_40,
            lambda x: 2 * x
        )

        self.assertAlmostEqual(ce5, 2.0)

        ce6 = calcExpectation(
            dd_10_10_100,
            lambda x, y: 2 * x + y,
            20
        )

        self.assertAlmostEqual(ce6, 40.0)

        ce7 = calcExpectation(
            dd_0_1_20,
            lambda x, y: x + y,
            np.hstack(np.array([0,1,2,3,4,5]))
        )

        self.assertAlmostEqual(ce7.flat[3], 3.0)

        PermShkDstn = MeanOneLogNormal().approx(200)
        TranShkDstn = MeanOneLogNormal().approx(200)
        IncomeDstn = combineIndepDstns(PermShkDstn, TranShkDstn)

        ce8 = calcExpectation(
            IncomeDstn,
            lambda X: X[0] + X[1]
        )

        self.assertAlmostEqual(ce8, 2.0)

        ce9 = calcExpectation(
            IncomeDstn,
            lambda X, a, r: r / X[0] * a + X[1],
            np.array([0,1,2,3,4,5]), # an aNrmNow grid?
            1.05 # an interest rate?
        )

        self.assertAlmostEqual(
            ce9[3][0],
            9.518015322143837
        )

class DistributionClassTests(unittest.TestCase):
    """
    Tests for distribution.py sampling distributions
    with default seed.
    """

    def test_drawMeanOneLognormal(self):
        self.assertEqual(MeanOneLogNormal().draw(1)[0], 3.5397367004222002)

    def test_Lognormal(self):

        dist = Lognormal()

        self.assertEqual(dist.draw(1)[0], 5.836039190663969)

        dist.draw(100)
        dist.reset()

        self.assertEqual(dist.draw(1)[0], 5.836039190663969)

    def test_Normal(self):
        dist = Normal()

        self.assertEqual(dist.draw(1)[0], 1.764052345967664)

        dist.draw(100)
        dist.reset()

        self.assertEqual(dist.draw(1)[0], 1.764052345967664)

    def test_Weibull(self):
        self.assertEqual(
            Weibull().draw(1)[0],
            0.79587450816311)

    def test_Uniform(self):
        uni = Uniform()

        self.assertEqual(
            Uniform().draw(1)[0],
            0.5488135039273248)

        self.assertEqual(
            calcExpectation(uni.approx(10)),
            0.5
        )

    def test_Bernoulli(self):
        self.assertEqual(
            Bernoulli().draw(1)[0], False
        )


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

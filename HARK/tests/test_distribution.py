import numpy as np
import unittest

import HARK.distribution as distribution

from HARK.distribution import (
    Bernoulli,
    IndexDistribution,
    DiscreteDistribution,
    Lognormal,
    MeanOneLogNormal,
    Normal,
    MVNormal,
    Uniform,
    Weibull,
    calc_expectation,
    combine_indep_dstns
)


class DiscreteDistributionTests(unittest.TestCase):
    """
    Tests for distribution.py sampling distributions
    with default seed.
    """

    def test_draw(self):
        self.assertEqual(
            DiscreteDistribution(np.ones(1), np.zeros(1)).draw(1)[0], 0,
        )

    def test_calc_expectation(self):
        dd_0_1_20 = Normal().approx(20)
        dd_1_1_40 = Normal(mu=1).approx(40)
        dd_10_10_100 = Normal(mu=10, sigma=10).approx(100)

        ce1 = calc_expectation(dd_0_1_20)
        ce2 = calc_expectation(dd_1_1_40)
        ce3 = calc_expectation(dd_10_10_100)

        self.assertAlmostEqual(ce1, 0.0)
        self.assertAlmostEqual(ce2, 1.0)
        self.assertAlmostEqual(ce3, 10.0)

        ce4= calc_expectation(
            dd_0_1_20,
            lambda x: 2 ** x
        )

        self.assertAlmostEqual(ce4, 1.27153712)

        ce5 = calc_expectation(
            dd_1_1_40,
            lambda x: 2 * x
        )

        self.assertAlmostEqual(ce5, 2.0)

        ce6 = calc_expectation(
            dd_10_10_100,
            lambda x, y: 2 * x + y,
            20
        )

        self.assertAlmostEqual(ce6, 40.0)

        ce7 = calc_expectation(
            dd_0_1_20,
            lambda x, y: x + y,
            np.hstack(np.array([0,1,2,3,4,5]))
        )

        self.assertAlmostEqual(ce7.flat[3], 3.0)

        PermShkDstn = MeanOneLogNormal().approx(200)
        TranShkDstn = MeanOneLogNormal().approx(200)
        IncShkDstn = combine_indep_dstns(PermShkDstn, TranShkDstn)

        ce8 = calc_expectation(
            IncShkDstn,
            lambda X: X[0] + X[1]
        )

        self.assertAlmostEqual(ce8, 2.0)

        ce9 = calc_expectation(
            IncShkDstn,
            lambda X, a, r: r / X[0] * a + X[1],
            np.array([0, 1, 2, 3, 4, 5]),  # an aNrmNow grid?
            1.05,  # an interest rate?
        )

        self.assertAlmostEqual(ce9[3], 9.518015322143837)


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

    def test_MVNormal(self):
        dist = MVNormal()

        self.assertTrue(
            np.allclose(dist.draw(1)[0], np.array([2.76405235, 1.40015721]))
        )

        dist.draw(100)
        dist.reset()

        self.assertTrue(
            np.allclose(dist.draw(1)[0], np.array([2.76405235, 1.40015721]))
        )

    def test_Weibull(self):
        self.assertEqual(Weibull().draw(1)[0], 0.79587450816311)

    def test_Uniform(self):
        uni = Uniform()

        self.assertEqual(Uniform().draw(1)[0], 0.5488135039273248)

        self.assertEqual(
            calc_expectation(uni.approx(10)),
            0.5
        )

    def test_Bernoulli(self):
        self.assertEqual(Bernoulli().draw(1)[0], False)

class IndexDistributionClassTests(unittest.TestCase):
    """
    Tests for distribution.py sampling distributions
    with default seed.
    """

    def test_IndexDistribution(self):
        cd = IndexDistribution(Bernoulli, {'p' : [.01, .5, .99]})

        conditions = np.array([0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2])

        draws = cd.draw(conditions)

        self.assertEqual(draws[:4].sum(), 0)
        self.assertEqual(draws[-4:].sum(),4)
        self.assertEqual(cd[2].p.tolist(), .99)

    def test_IndexDistribution_approx(self):
        cd = IndexDistribution(
            Lognormal,
            {
                'mu' : [.01, .5, .99],
                'sigma' : [.05, .05, .05]
            }
        )

        approx = cd.approx(10)

        draw = approx[2].draw(5)

        self.assertAlmostEqual(draw[1], 2.93868620)

    def test_IndexDistribution_seeds(self):
        cd = IndexDistribution(
            Lognormal,
            {
                'mu' : [1, 1],
                'sigma' : [1, 1]
            }
        )

        draw_0 = cd[0].draw(1).tolist()
        draw_1 = cd[1].draw(1).tolist()

        self.assertNotEqual(draw_0, draw_1)

class MarkovProcessTests(unittest.TestCase):
    """
    Tests for MarkovProcess class.
    """

    def test_draw(self):

        mrkv_array = np.array([[0.75, 0.25], [0.1, 0.9]])

        mp = distribution.MarkovProcess(mrkv_array)

        new_state = mp.draw(np.zeros(100).astype(int))

        self.assertEqual(new_state.sum(), 20)

        new_state = mp.draw(new_state)

        self.assertEqual(new_state.sum(), 39)

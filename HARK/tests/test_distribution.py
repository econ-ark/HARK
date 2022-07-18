import unittest

import numpy as np

from HARK.distribution import (
    Bernoulli,
    calc_expectation,
    distr_of_function,
    calc_lognormal_style_pars_from_normal_pars,
    calc_normal_style_pars_from_lognormal_pars,
    combine_indep_dstns,
    DiscreteDistribution,
    IndexDistribution,
    Lognormal,
    MarkovProcess,
    MeanOneLogNormal,
    MVNormal,
    Normal,
    Uniform,
    Weibull,
)


class DiscreteDistributionTests(unittest.TestCase):
    """
    Tests for distribution.py sampling distributions
    with default seed.
    """

    def test_draw(self):
        self.assertEqual(
            DiscreteDistribution(np.ones(1), np.zeros(1)).draw(1)[0],
            0,
        )

    def test_distr_of_function(self):

        # Function 1 -> 1
        # Approximate the lognormal expectation
        sig = 0.05
        norm = Normal(mu=-(sig ** 2) / 2, sigma=sig).approx(131)
        my_logn = distr_of_function(norm, func=lambda x: np.exp(x))
        exp = calc_expectation(my_logn)
        self.assertAlmostEqual(exp, 1.0)

        # Function 1 -> n
        # Mean and variance of the normal
        norm = Normal(mu=0.0, sigma=1.0).approx(5)
        moments = distr_of_function(norm, lambda x: np.array([x, x ** 2]))
        exp = calc_expectation(moments).flatten()
        self.assertAlmostEqual(exp[0], 0.0)
        self.assertAlmostEqual(exp[1], 1.0)

        # Function n -> 1
        # Expectation of the sum of two independent normals
        mu_a, mu_b = 1.0, 2.0
        si_a, si_b = 3.0, 4.0
        norm_a = Normal(mu=mu_a, sigma=si_a).approx(5)
        norm_b = Normal(mu=mu_b, sigma=si_b).approx(5)
        binorm = combine_indep_dstns(norm_a, norm_b)
        mysum = distr_of_function(binorm, lambda x: np.sum(x))
        exp = calc_expectation(mysum)
        self.assertAlmostEqual(exp[0], mu_a + mu_b)

        # Function n -> m
        # Mean and variance of two normals
        moments = distr_of_function(
            binorm,
            lambda x: np.array([x[0], (x[0] - mu_a) ** 2, x[1], (x[1] - mu_b) ** 2]),
        )
        exp = calc_expectation(moments)
        self.assertAlmostEqual(exp[0], mu_a)
        self.assertAlmostEqual(exp[1], si_a ** 2)
        self.assertAlmostEqual(exp[2], mu_b)
        self.assertAlmostEqual(exp[3], si_b ** 2)

    def test_calc_expectation(self):
        dd_0_1_20 = Normal().approx(20)
        dd_1_1_40 = Normal(mu=1).approx(40)
        dd_10_10_100 = Normal(mu=10, sigma=10).approx(100)

        ce1 = calc_expectation(dd_0_1_20)
        ce2 = calc_expectation(dd_1_1_40)
        ce3 = calc_expectation(dd_10_10_100)

        self.assertAlmostEqual(ce1[0], 0.0)
        self.assertAlmostEqual(ce2[0], 1.0)
        self.assertAlmostEqual(ce3[0], 10.0)

        ce4 = calc_expectation(dd_0_1_20, lambda x: 2 ** x)

        self.assertAlmostEqual(ce4[0], 1.27153712)

        ce5 = calc_expectation(dd_1_1_40, lambda x: 2 * x)

        self.assertAlmostEqual(ce5[0], 2.0)

        ce6 = calc_expectation(dd_10_10_100, lambda x, y: 2 * x + y, 20)

        self.assertAlmostEqual(ce6[0], 40.0)

        ce7 = calc_expectation(
            dd_0_1_20, lambda x, y: x + y, np.hstack(np.array([0, 1, 2, 3, 4, 5]))
        )

        self.assertAlmostEqual(ce7.flat[3], 3.0)

        PermShkDstn = MeanOneLogNormal().approx(200)
        TranShkDstn = MeanOneLogNormal().approx(200)
        IncShkDstn = combine_indep_dstns(PermShkDstn, TranShkDstn)

        ce8 = calc_expectation(IncShkDstn, lambda X: X[0] + X[1])

        self.assertAlmostEqual(ce8, 2.0)

        ce9 = calc_expectation(
            IncShkDstn,
            lambda X, a, r: r / X[0] * a + X[1],
            np.array([0, 1, 2, 3, 4, 5]),  # an aNrmNow grid?
            1.05,  # an interest rate?
        )

        self.assertAlmostEqual(ce9[3], 9.518015322143837)

    def test_self_calc_expectation(self):
        dd_0_1_20 = Normal().approx(20)
        dd_1_1_40 = Normal(mu=1).approx(40)
        dd_10_10_100 = Normal(mu=10, sigma=10).approx(100)

        ce1 = dd_0_1_20.calc_expectation()
        ce2 = dd_1_1_40.calc_expectation()
        ce3 = dd_10_10_100.calc_expectation()

        self.assertAlmostEqual(ce1[0], 0.0)
        self.assertAlmostEqual(ce2[0], 1.0)
        self.assertAlmostEqual(ce3[0], 10.0)

        ce4 = dd_0_1_20.calc_expectation(lambda x: 2**x)

        self.assertAlmostEqual(ce4[0], 1.27153712)

        ce5 = dd_1_1_40.calc_expectation(lambda x: 2 * x)

        self.assertAlmostEqual(ce5[0], 2.0)

        ce6 = dd_10_10_100.calc_expectation(lambda x, y: 2 * x + y, 20)

        self.assertAlmostEqual(ce6[0], 40.0)

        ce7 = dd_0_1_20.calc_expectation(
            lambda x, y: x + y, np.hstack(np.array([0, 1, 2, 3, 4, 5]))
        )

        self.assertAlmostEqual(ce7.flat[3], 3.0)

        PermShkDstn = MeanOneLogNormal().approx(200)
        TranShkDstn = MeanOneLogNormal().approx(200)
        IncShkDstn = combine_indep_dstns(PermShkDstn, TranShkDstn)

        ce8 = IncShkDstn.calc_expectation(lambda X: X[0] + X[1])

        self.assertAlmostEqual(ce8, 2.0)

        ce9 = IncShkDstn.calc_expectation(
            lambda X, a, r: r / X[0] * a + X[1],
            np.array([0, 1, 2, 3, 4, 5]),  # an aNrmNow grid?
            1.05,  # an interest rate?
        )

        self.assertAlmostEqual(ce9[3], 9.518015322143837)

class MatrixDiscreteDistributionTests(unittest.TestCase):
    """
    Tests matrix-valued discrete distribution.
    """

    def setUp(self):

        self.draw_1 = np.array(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],]
        )

        self.draw_2 = -1 * self.draw_1

        X = np.stack([self.draw_1, self.draw_2], axis=-1)
        pmf = np.array([0.5, 0.5])

        self.mat_distr = DiscreteDistribution(pmf, X, seed=0)

    def test_draw(self):
        """
        Check that the draws are the matrices we
        want them to be
        """

        draw = self.mat_distr.draw(1)
        self.assertTrue(np.allclose(draw[..., 0], self.draw_2))

    def test_expected(self):

        # Expectation without transformation
        exp = calc_expectation(self.mat_distr)

        # Check the expectation is of the shape we want
        self.assertTrue(exp.shape[0] == self.draw_1.shape[0])
        self.assertTrue(exp.shape[1] == self.draw_1.shape[1])

        # Check that its value is what we expect
        self.assertTrue(np.allclose(exp, 0.0))

        # Expectation of the sum
        exp = calc_expectation(self.mat_distr, func=np.sum)
        self.assertTrue(float(exp) == 0.0)

    def test_distr_of_fun(self):

        # A function that receives a (2,n,m) matrix
        # and sums across n, getting a (2,1,m) matrix
        def myfunc(mat):

            return np.sum(mat, axis=1, keepdims=True)

        mydistr = distr_of_function(self.mat_distr, myfunc)

        # Check the dimensions
        self.assertTrue(mydistr.dim() == (2, 1, 3))


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
        self.assertAlmostEqual(Weibull().draw(1)[0], 0.79587450816311)

    def test_Uniform(self):
        uni = Uniform()

        self.assertEqual(Uniform().draw(1)[0], 0.5488135039273248)

        self.assertEqual(calc_expectation(uni.approx(10)), 0.5)

    def test_Bernoulli(self):
        self.assertEqual(Bernoulli().draw(1)[0], False)


class IndexDistributionClassTests(unittest.TestCase):
    """
    Tests for distribution.py sampling distributions
    with default seed.
    """

    def test_IndexDistribution(self):
        cd = IndexDistribution(Bernoulli, {"p": [0.01, 0.5, 0.99]})

        conditions = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2])

        draws = cd.draw(conditions)

        self.assertEqual(draws[:4].sum(), 0)
        self.assertEqual(draws[-4:].sum(), 4)
        self.assertEqual(cd[2].p.tolist(), 0.99)

    def test_IndexDistribution_approx(self):
        cd = IndexDistribution(
            Lognormal, {"mu": [0.01, 0.5, 0.99], "sigma": [0.05, 0.05, 0.05]}
        )

        approx = cd.approx(10)

        draw = approx[2].draw(5)

        self.assertAlmostEqual(draw[1], 2.93868620)

    def test_IndexDistribution_seeds(self):
        cd = IndexDistribution(Lognormal, {"mu": [1, 1], "sigma": [1, 1]})

        draw_0 = cd[0].draw(1).tolist()
        draw_1 = cd[1].draw(1).tolist()

        self.assertNotEqual(draw_0, draw_1)


class MarkovProcessTests(unittest.TestCase):
    """
    Tests for MarkovProcess class.
    """

    def test_draw(self):
        mrkv_array = np.array([[0.75, 0.25], [0.1, 0.9]])

        mp = MarkovProcess(mrkv_array)

        new_state = mp.draw(np.zeros(100).astype(int))

        self.assertEqual(new_state.sum(), 20)

        new_state = mp.draw(new_state)

        self.assertEqual(new_state.sum(), 39)


class LogNormalToNormalTests(unittest.TestCase):
    """
    Tests methods to convert between lognormal and normal parameters.
    """

    def test_lognorm_to_norm(self):
        avg_ln, std_ln = 1.0, 0.2
        avg_n, std_n = calc_normal_style_pars_from_lognormal_pars(avg_ln, std_ln)
        avg_hat, std_hat = calc_lognormal_style_pars_from_normal_pars(avg_n, std_n)

        self.assertAlmostEqual(avg_ln, avg_hat)
        self.assertAlmostEqual(std_ln, std_hat)

    def test_norm_to_lognorm(self):
        avg_n, std_n = 1.0, 0.2
        avg_ln, std_ln = calc_lognormal_style_pars_from_normal_pars(avg_n, std_n)
        avg_hat, std_hat = calc_normal_style_pars_from_lognormal_pars(avg_ln, std_ln)

        self.assertAlmostEqual(avg_n, avg_hat)
        self.assertAlmostEqual(std_n, std_hat)


class NormalDistTest(unittest.TestCase):
    def test_approx_equiprobable(self):

        mu, sigma = 5.0, 27.0

        points = Normal(mu, sigma).approx_equiprobable(701).X

        self.assertAlmostEqual(np.mean(points), mu, places=7)
        self.assertAlmostEqual(np.std(points), sigma, places=2)

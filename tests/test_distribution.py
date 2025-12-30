import unittest

import numpy as np
import xarray as xr

from HARK.distributions import (
    Bernoulli,
    DiscreteDistribution,
    DiscreteDistributionLabeled,
    IndexDistribution,
    Lognormal,
    MarkovProcess,
    MeanOneLogNormal,
    Normal,
    Uniform,
    Weibull,
    calc_expectation,
    calc_lognormal_style_pars_from_normal_pars,
    calc_normal_style_pars_from_lognormal_pars,
    combine_indep_dstns,
    distr_of_function,
    expected,
    approx_beta,
    make_markov_approx_to_normal,
    make_markov_approx_to_normal_by_monte_carlo,
    make_tauchen_ar1,
    MultivariateNormal,
    MultivariateLogNormal,
    approx_lognormal_gauss_hermite,
)
from tests import HARK_PRECISION


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
        norm = Normal(mu=-(sig**2) / 2, sigma=sig).discretize(131, method="hermite")
        my_logn = distr_of_function(norm, func=lambda x: np.exp(x))
        exp = calc_expectation(my_logn)
        self.assertAlmostEqual(float(exp), 1.0)

        # Function 1 -> n
        # Mean and variance of the normal
        norm = Normal(mu=0.0, sigma=1.0).discretize(5, method="hermite")
        moments = distr_of_function(norm, lambda x: np.array([x, x**2]))
        exp = calc_expectation(moments).flatten()
        self.assertAlmostEqual(exp[0], 0.0)
        self.assertAlmostEqual(exp[1], 1.0)

        # Function n -> 1
        # Expectation of the sum of two independent normals
        mu_a, mu_b = 1.0, 2.0
        si_a, si_b = 3.0, 4.0
        norm_a = Normal(mu=mu_a, sigma=si_a).discretize(5, method="hermite")
        norm_b = Normal(mu=mu_b, sigma=si_b).discretize(5, method="hermite")
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
        self.assertAlmostEqual(exp[1], si_a**2)
        self.assertAlmostEqual(exp[2], mu_b)
        self.assertAlmostEqual(exp[3], si_b**2)

    def test_calc_expectation(self):
        dd_0_1_20 = Normal().discretize(20, method="hermite")
        dd_1_1_40 = Normal(mu=1).discretize(40, method="hermite")
        dd_10_10_100 = Normal(mu=10, sigma=10).discretize(100, method="hermite")

        ce1 = calc_expectation(dd_0_1_20)
        ce2 = calc_expectation(dd_1_1_40)
        ce3 = calc_expectation(dd_10_10_100)

        self.assertAlmostEqual(ce1[0], 0.0)
        self.assertAlmostEqual(ce2[0], 1.0)
        self.assertAlmostEqual(ce3[0], 10.0)

        ce4 = calc_expectation(dd_0_1_20, lambda x: 2**x)

        self.assertAlmostEqual(ce4[0], 1.27154, places=HARK_PRECISION)

        ce5 = calc_expectation(dd_1_1_40, lambda x: 2 * x)

        self.assertAlmostEqual(ce5[0], 2.0)

        ce6 = calc_expectation(dd_10_10_100, lambda x, y: 2 * x + y, 20)

        self.assertAlmostEqual(ce6[0], 40.0)

        ce7 = calc_expectation(
            dd_0_1_20, lambda x, y: x + y, np.hstack(np.array([0, 1, 2, 3, 4, 5]))
        )

        self.assertAlmostEqual(ce7.flat[3], 3.0)

        PermShkDstn = MeanOneLogNormal().discretize(200, method="equiprobable")
        TranShkDstn = MeanOneLogNormal().discretize(200, method="equiprobable")
        IncShkDstn = combine_indep_dstns(PermShkDstn, TranShkDstn)

        ce8 = calc_expectation(IncShkDstn, lambda atoms: atoms[0] + atoms[1])

        self.assertAlmostEqual(ce8, 2.0)

        ce9 = calc_expectation(
            IncShkDstn,
            lambda atoms, a, r: r / atoms[0] * a + atoms[1],
            np.array([0, 1, 2, 3, 4, 5]),  # an aNrmNow grid?
            1.05,  # an interest rate?
        )

        self.assertAlmostEqual(ce9[3], 9.51802, places=HARK_PRECISION)

    def test_self_expected_value(self):
        dd_0_1_20 = Normal().discretize(20, method="hermite")
        dd_1_1_40 = Normal(mu=1).discretize(40, method="hermite")
        dd_10_10_100 = Normal(mu=10, sigma=10).discretize(100, method="hermite")

        ce1 = expected(dist=dd_0_1_20)
        ce2 = expected(dist=dd_1_1_40)
        ce3 = expected(dist=dd_10_10_100)

        self.assertAlmostEqual(ce1[0], 0.0)
        self.assertAlmostEqual(ce2[0], 1.0)
        self.assertAlmostEqual(ce3[0], 10.0)

        ce4 = expected(lambda x: 2**x, dd_0_1_20)

        self.assertAlmostEqual(ce4[0], 1.27154, places=HARK_PRECISION)

        ce5 = expected(func=lambda x: 2 * x, dist=dd_1_1_40)

        self.assertAlmostEqual(ce5[0], 2.0)

        ce6 = expected(lambda x, y: 2 * x + y, dd_10_10_100, args=(20))

        self.assertAlmostEqual(ce6[0], 40.0)

        ce7 = expected(
            func=lambda x, y: x + y,
            dist=dd_0_1_20,
            args=(np.hstack([0, 1, 2, 3, 4, 5])),
        )

        self.assertAlmostEqual(ce7.flat[3], 3.0)

        PermShkDstn = MeanOneLogNormal().discretize(200, method="equiprobable")
        TranShkDstn = MeanOneLogNormal().discretize(200, method="equiprobable")
        IncShkDstn = combine_indep_dstns(PermShkDstn, TranShkDstn)

        ce8 = expected(lambda atoms: atoms[0] + atoms[1], dist=IncShkDstn)

        self.assertAlmostEqual(ce8, 2.0)

        ce9 = expected(
            func=lambda atoms, a, r: r / atoms[0] * a + atoms[1],
            dist=IncShkDstn,
            args=(
                np.array([0, 1, 2, 3, 4, 5]),  # an aNrmNow grid?
                1.05,  # an interest rate?
            ),
        )

        self.assertAlmostEqual(ce9[3], 9.51802, places=HARK_PRECISION)

    def test_self_dist_of_func(self):
        # Function 1 -> 1
        # Approximate the lognormal expectation
        sig = 0.05
        norm = Normal(mu=-(sig**2) / 2, sigma=sig).discretize(131, method="hermite")
        my_logn = norm.dist_of_func(lambda x: np.exp(x))
        exp = my_logn.expected()
        self.assertAlmostEqual(float(exp), 1.0)

        # Function 1 -> n
        # Mean and variance of the normal
        norm = Normal(mu=0.0, sigma=1.0).discretize(5, method="hermite")
        moments = norm.dist_of_func(lambda x: np.array([x, x**2]))
        exp = moments.expected().flatten()
        self.assertAlmostEqual(exp[0], 0.0)
        self.assertAlmostEqual(exp[1], 1.0)

        # Function n -> 1
        # Expectation of the sum of two independent normals
        mu_a, mu_b = 1.0, 2.0
        si_a, si_b = 3.0, 4.0
        norm_a = Normal(mu=mu_a, sigma=si_a).discretize(5, method="hermite")
        norm_b = Normal(mu=mu_b, sigma=si_b).discretize(5, method="hermite")
        binorm = combine_indep_dstns(norm_a, norm_b)
        mysum = binorm.dist_of_func(func=lambda x: np.sum(x, axis=0))
        exp = mysum.expected()
        self.assertAlmostEqual(exp[0], mu_a + mu_b)

        # Function n -> m
        # Mean and variance of two normals
        moments = binorm.dist_of_func(
            func=lambda x: np.array(
                [x[0], (x[0] - mu_a) ** 2, x[1], (x[1] - mu_b) ** 2]
            ),
        )
        exp = moments.expected()
        self.assertAlmostEqual(exp[0], mu_a)
        self.assertAlmostEqual(exp[1], si_a**2)
        self.assertAlmostEqual(exp[2], mu_b)
        self.assertAlmostEqual(exp[3], si_b**2)


class MatrixDiscreteDistributionTests(unittest.TestCase):
    """
    Tests matrix-valued discrete distribution.
    """

    def setUp(self):
        self.draw_1 = np.array(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ]
        )

        self.draw_2 = -1 * self.draw_1

        atoms = np.stack([self.draw_1, self.draw_2], axis=-1)
        pmv = np.array([0.5, 0.5])

        self.mat_distr = DiscreteDistribution(pmv, atoms, seed=0)

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
        MeanOneLogNormal().draw(1)[0]

    def test_Lognormal(self):
        dist = Lognormal()

        dist.draw(1)[0]

        dist.draw(100)
        dist.reset()

        dist.draw(1)[0]

    def test_Lognormal_hermite_discretizer(self):
        dstn = Lognormal(mu=0.5, sigma=0.2)
        discrete_dstn = dstn.discretize(9, method="hermite")
        exp_discrete = np.dot(discrete_dstn.pmv, discrete_dstn.atoms.T)
        self.assertAlmostEqual(dstn.mean(), exp_discrete)
        self.assertAlmostEqual(np.sum(discrete_dstn.pmv), 1.0)

    def test_Lognormal_from_mean_std(self):
        dstn = Lognormal(mean=1.3, std=0.4)
        self.assertAlmostEqual(1.3, dstn.mean())

    def test_Normal(self):
        dist = Normal()

        dist.draw(1)[0]

        dist.draw(100)
        dist.reset()

        dist.draw(1)[0]

    def test_MultivariateNormal(self):
        dist = MultivariateNormal()
        dist.draw(100)
        dist.reset()

    def test_MultivariateLogNormal(self):
        dstn = MultivariateLogNormal(mu=[-0.2, 0.3], Sigma=[[1.0, 0.3], [0.3, 1.0]])
        X = np.random.rand(100, 2)

        dstn.draw(100)
        dstn.reset()

        cdf_vals = dstn._cdf(X)
        self.assertTrue(np.all(cdf_vals <= 1.0))
        self.assertTrue(np.all(cdf_vals >= 0.0))
        self.assertRaises(ValueError, dstn._cdf, X.T)

        pdf_vals = dstn._pdf(X)
        self.assertTrue(np.all(pdf_vals >= 0.0))

        marg_pdf = dstn._marginal(X, dim=0)
        self.assertTrue(np.all(marg_pdf >= 0.0))

        marg_cdf = dstn._marginal_cdf(X, dim=1)
        self.assertTrue(np.all(marg_cdf <= 1.0))
        self.assertTrue(np.all(marg_cdf >= 0.0))

        discrete_dstn = dstn.discretize(9)
        self.assertAlmostEqual(discrete_dstn.atoms.shape[1], 81)
        self.assertAlmostEqual(np.sum(discrete_dstn.pmv), 1.0)

        discrete_dstn = dstn.discretize(9, endpoints=True)

        discrete_dstn = dstn.discretize(9, decomp="sqrt")
        self.assertAlmostEqual(discrete_dstn.atoms.shape[1], 81)
        self.assertAlmostEqual(np.sum(discrete_dstn.pmv), 1.0)

        discrete_dstn = dstn.discretize(9, decomp="eig")
        self.assertAlmostEqual(discrete_dstn.atoms.shape[1], 81)
        self.assertAlmostEqual(np.sum(discrete_dstn.pmv), 1.0)

        discrete_dstn = MultivariateLogNormal().discretize(9)
        self.assertAlmostEqual(discrete_dstn.atoms.shape[1], 81)
        self.assertAlmostEqual(np.sum(discrete_dstn.pmv), 1.0)

        self.assertRaises(
            NotImplementedError, dstn.discretize, 7, decomp="well hello there"
        )

    def test_Weibull(self):
        Weibull().draw(1)[0]

    def test_Uniform(self):
        uni = Uniform()

        Uniform().draw(1)[0]

        self.assertAlmostEqual(
            float(calc_expectation(uni.discretize(10, method="equiprobable"))),
            0.5,
        )

        uni_discrete = uni.discretize(10, method="equiprobable", endpoints=True)

        self.assertEqual(uni_discrete.atoms[0][0], 0.0)
        self.assertEqual(uni_discrete.atoms[0][-1], 1.0)
        self.assertAlmostEqual(
            float(calc_expectation(uni.discretize(10, method="equiprobable"))),
            0.5,
        )

    def test_Bernoulli(self):
        Bernoulli().draw(1)[0]

    def test_Bernoulli_combine_indep_dstns(self):
        """Test that combine_indep_dstns works with Bernoulli distributions"""
        # Test 1: Single Bernoulli distribution
        b = Bernoulli(p=0.3)
        result = combine_indep_dstns(b)

        # Result should be essentially the same as the input
        self.assertEqual(len(result.pmv), 2)
        self.assertAlmostEqual(result.pmv[0], 0.7)  # P(0)
        self.assertAlmostEqual(result.pmv[1], 0.3)  # P(1)

        # Test 2: Two independent Bernoulli distributions
        b1 = Bernoulli(p=0.3)
        b2 = Bernoulli(p=0.4)
        result = combine_indep_dstns(b1, b2)

        # Should have 4 outcomes: (0,0), (0,1), (1,0), (1,1)
        self.assertEqual(len(result.pmv), 4)
        self.assertEqual(result.atoms.shape, (2, 4))  # 2 variables, 4 outcomes

        # Check probabilities
        expected_probs = [
            0.7 * 0.6,
            0.7 * 0.4,
            0.3 * 0.6,
            0.3 * 0.4,
        ]  # P(0,0), P(0,1), P(1,0), P(1,1)
        for i, expected_prob in enumerate(expected_probs):
            self.assertAlmostEqual(result.pmv[i], expected_prob, places=5)

    def test_Bernoulli_labeled_combine_indep_dstns(self):
        """Test that combine_indep_dstns works with labeled Bernoulli distributions"""
        b = Bernoulli(p=0.5)
        bl = DiscreteDistributionLabeled.from_unlabeled(b, var_names="b")
        result = combine_indep_dstns(bl)

        # Result should be essentially the same as the input
        self.assertEqual(len(result.pmv), 2)
        self.assertAlmostEqual(result.pmv[0], 0.5)  # P(0)
        self.assertAlmostEqual(result.pmv[1], 0.5)  # P(1)


class IndexDistributionClassTests(unittest.TestCase):
    """
    Tests for distribution.py sampling distributions
    with default seed.
    """

    def test_IndexDistribution(self):
        cd = IndexDistribution(Bernoulli, {"p": [0.01, 0.5, 0.99]}, seed=0)

        conditions = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2])

        draws = cd.draw(conditions)

        self.assertEqual(draws[:4].sum(), 0)
        self.assertEqual(draws[-4:].sum(), 4)
        self.assertEqual(cd[2].p.tolist(), 0.99)

    def test_IndexDistribution_approx(self):
        cd = IndexDistribution(
            Lognormal, {"mu": [0.01, 0.5, 0.99], "sigma": [0.05, 0.05, 0.05]}, seed=0
        )

        approx = cd.discretize(10)

        draw = approx[2].draw(5)

        self.assertAlmostEqual(draw[1], 2.70826, places=HARK_PRECISION)

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

        mp = MarkovProcess(mrkv_array, seed=0)

        new_state = mp.draw(np.zeros(100).astype(int))

        self.assertEqual(new_state.sum(), 31)

        new_state = mp.draw(new_state)

        self.assertEqual(new_state.sum(), 45)


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

        points = Normal(mu, sigma).discretize(701, method="equiprobable").atoms

        self.assertAlmostEqual(np.mean(points), mu, places=7)
        self.assertAlmostEqual(np.std(points), sigma, places=2)


class DiscreteDistributionLabeledTests(unittest.TestCase):
    """
    Tests for distribution.py sampling distributions
    with default seed.
    """

    def test_draw(self):
        self.assertEqual(
            DiscreteDistributionLabeled(np.ones(2) / 2, np.zeros(2)).draw(1)[0],
            0,
        )

    def test_self_dist_of_func(self):
        # Function 1 -> 1
        # Approximate the lognormal expectation
        sig = 0.05
        mu = -(sig**2) / 2
        norm = Normal(mu=mu, sigma=sig).discretize(131, method="hermite")
        my_logn = DiscreteDistributionLabeled.from_unlabeled(
            norm.dist_of_func(func=lambda x: np.exp(x)),
            name="Lognormal Approximation",  # name of the distribution
            # assign limit properties
            attrs={"limit": {"mu": mu, "sigma": sig}},
        )
        exp = my_logn.expected()
        self.assertAlmostEqual(exp[0], 1.0)

        # Function 1 -> n
        # Mean and variance of the normal
        norm = Normal(mu=0.0, sigma=1.0).discretize(5, method="hermite")
        moments = DiscreteDistributionLabeled.from_unlabeled(
            norm.dist_of_func(lambda x: np.vstack([x, x**2])),
            name="Moments of Normal Distribution",
            var_names=["mean", "variance"],
            attrs={"limit": {"name": "Normal", "mu": 0.0, "sigma": 1.0}},
        )
        exp = moments.expected().flatten()
        self.assertAlmostEqual(exp[0], 0.0)
        self.assertAlmostEqual(exp[1], 1.0)

        # Function n -> 1
        # Expectation of the sum of two independent normals
        mu_a, mu_b = 1.0, 2.0
        si_a, si_b = 3.0, 4.0
        norm_a = Normal(mu=mu_a, sigma=si_a).discretize(5, method="hermite")
        norm_b = Normal(mu=mu_b, sigma=si_b).discretize(5, method="hermite")
        binorm = combine_indep_dstns(norm_a, norm_b)
        mysum = DiscreteDistributionLabeled.from_unlabeled(
            binorm.dist_of_func(lambda x: np.sum(x, axis=0)),  # vectorized sum
            name="Sum of two independent normals",
        )
        exp = mysum.expected()
        self.assertAlmostEqual(exp[0], mu_a + mu_b)

        # Function n -> m
        # Mean and variance of two normals
        moments = DiscreteDistributionLabeled.from_unlabeled(
            binorm.dist_of_func(
                lambda x: np.array([x[0], (x[0] - mu_a) ** 2, x[1], (x[1] - mu_b) ** 2])
            ),
            name="Moments of two independent normals",
            var_names=["mean_1", "variance_1", "mean_2", "variance_2"],
        )
        exp = moments.expected()
        self.assertAlmostEqual(exp[0], mu_a)
        self.assertAlmostEqual(exp[1], si_a**2)
        self.assertAlmostEqual(exp[2], mu_b)
        self.assertAlmostEqual(exp[3], si_b**2)

    def test_self_expected_value(self):
        PermShkDstn = MeanOneLogNormal().discretize(200, method="equiprobable")
        TranShkDstn = MeanOneLogNormal().discretize(200, method="equiprobable")
        IncShkDstn = combine_indep_dstns(
            PermShkDstn,
            TranShkDstn,
        )

        IncShkDstn = DiscreteDistributionLabeled.from_unlabeled(
            IncShkDstn,
            name="Distribution of shocks to Income",
            var_names=["perm_shk", "tran_shk"],
        )

        ce1 = expected(
            func=lambda dist: 1 / dist["perm_shk"] + dist["tran_shk"],
            dist=IncShkDstn,
        )

        self.assertAlmostEqual(ce1, 3.70413, places=HARK_PRECISION)

        ce2 = expected(
            func=lambda dist, a, r: r / dist["perm_shk"] * a + dist["tran_shk"],
            dist=IncShkDstn,
            args=(
                np.array([0, 1, 2, 3, 4, 5]),  # an aNrmNow grid?
                1.05,  # an interest rate?
            ),
        )

        self.assertAlmostEqual(ce2[3], 9.51802, places=HARK_PRECISION)

    def test_getters_setters(self):
        # Create some dummy dsnt
        dist = DiscreteDistributionLabeled(
            pmv=np.array([0.5, 0.5]), atoms=np.array([-1.0, 1.0]), var_names=["my_var"]
        )

        # Seed
        my_seed = 3
        dist.seed = my_seed
        self.assertTrue(my_seed == dist.seed)

        # RNG
        my_rng = np.random.default_rng(5)
        dist.RNG = my_rng
        self.assertTrue(my_rng == dist.RNG)

    def test_combine_labeled_dist(self):
        # Create some dstns
        a = DiscreteDistributionLabeled(
            pmv=np.array([0.1, 0.9]), atoms=np.array([-1.0, 1.0]), var_names="a"
        )
        b = DiscreteDistributionLabeled(
            pmv=np.array([0.5, 0.5]), atoms=np.array([0.0, 1.0]), var_names="b"
        )
        c = DiscreteDistributionLabeled(
            pmv=np.array([0.3, 0.7]), atoms=np.array([0.5, 1.0]), var_names="c"
        )

        # Test some combinations
        abc = combine_indep_dstns(a, b, c)
        # Check the order
        self.assertTrue(
            np.all(
                np.isclose(
                    abc.expected(),
                    np.concatenate([a.expected(), b.expected(), c.expected()]),
                )
            )
        )
        # Check by label
        self.assertEqual(abc.expected(lambda x: x["b"]), b.expected()[0])
        self.assertAlmostEqual(
            abc.expected(lambda x: x["a"] * x["c"]), a.expected()[0] * c.expected()[0]
        )

        # Combine labeled and non labeled distribution
        x = DiscreteDistribution(pmv=np.array([0.5, 0.5]), atoms=np.array([1.0, 2.0]))

        xa = combine_indep_dstns(x, a)
        self.assertFalse(isinstance(xa, DiscreteDistributionLabeled))
        self.assertTrue(
            np.all(xa.expected() == np.concatenate([x.expected(), a.expected()]))
        )

        # Combine multidimensional labeled
        d = DiscreteDistributionLabeled(
            pmv=np.array([0.3, 0.7]), atoms=np.array([-0.5, -1.0]), var_names="d"
        )
        e = DiscreteDistributionLabeled(
            pmv=np.array([0.3, 0.7]), atoms=np.array([0.0, -1.0]), var_names="e"
        )
        de = combine_indep_dstns(d, e)

        abcde = combine_indep_dstns(abc, de)
        self.assertTrue(
            np.allclose(
                abcde.expected(
                    lambda x: np.array([x["d"], x["e"], x["a"], x["b"], x["c"]])
                ),
                np.concatenate([de.expected(), abc.expected()]),
            )
        )

    def test_Bernoulli_to_labeled(self):
        p = 0.4
        foo = Bernoulli(p)
        bern = DiscreteDistributionLabeled.from_unlabeled(foo, var_names=["foo"])
        self.assertTrue(np.allclose(bern.expected(), p))


class labeled_transition_tests(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_expectation_transformation(self):
        # Create a basic labeled distribution
        base_dist = DiscreteDistributionLabeled(
            pmv=np.array([0.5, 0.5]),
            atoms=np.array([[1.0, 2.0], [3.0, 4.0]]),
            var_names=["a", "b"],
        )

        # Define a transition function
        def transition(shocks, state):
            state_new = {}
            state_new["m"] = state["m"] * shocks["a"]
            state_new["n"] = state["n"] * shocks["b"]
            return state_new

        m = xr.DataArray(np.linspace(0, 10, 11), name="m", dims=("grid",))
        n = xr.DataArray(np.linspace(0, -10, 11), name="n", dims=("grid",))
        state_grid = xr.Dataset({"m": m, "n": n})

        # Evaluate labeled transformation

        # Direct expectation
        exp1 = base_dist.expected(transition, state=state_grid)
        # Expectation after transformation
        new_state_dstn = base_dist.dist_of_func(transition, state=state_grid)
        # TODO: needs a cluncky identity function with an extra argument because
        # DDL.expected() behavior is very different with and without kwargs.
        # Fix!
        exp2 = new_state_dstn.expected(lambda x, unused: x, unused=0)

        assert np.all(exp1["m"] == exp2["m"]).item()
        assert np.all(exp1["n"] == exp2["n"]).item()


class TestTauchenAR1(unittest.TestCase):
    def test_tauchen(self):
        # Test with a simple AR(1) process
        N = 5
        sigma = 1.0
        ar_1 = 0.9
        bound = 3.0

        # By default, inflendpoint = True
        standard = make_tauchen_ar1(N, sigma, ar_1, bound)
        alternative = make_tauchen_ar1(N, sigma, ar_1, bound, inflendpoint=False)

        # Check that the grid points of the two methods are identical
        self.assertTrue(np.all(np.equal(standard[0], alternative[0])))

        # Check the shape of the transition matrix
        self.assertEqual(standard[1].shape, (N, N))
        self.assertEqual(alternative[1].shape, (N, N))

        # Check that the sum of each row in the transition matrix is 1
        self.assertTrue(np.allclose(np.sum(standard[1], axis=1), np.ones(N)))
        self.assertTrue(np.allclose(np.sum(alternative[1], axis=1), np.ones(N)))

        # Check that [k]-th column ./ [k-1]-th column are identical (k = 3, ..., N-1)
        # Note: the first and the last column of the 'standard' transition matrix are inflated
        if N > 3:
            for i in range(2, N - 1):
                self.assertTrue(
                    np.allclose(
                        standard[1][:, i] * alternative[1][:, i - 1],
                        alternative[1][:, i] * standard[1][:, i - 1],
                    )
                )


class test_assorted_functions(unittest.TestCase):
    def test_approx_beta(self):
        dstn = approx_beta(15, 0.5, 2.0)
        self.assertTrue(isinstance(dstn, DiscreteDistribution))
        self.assertAlmostEqual(np.sum(dstn.pmv), 1.0)

    def test_make_markov_approx_to_normal(self):
        X = np.linspace(-4.0, 6.0, 50)
        vec = make_markov_approx_to_normal(X, 0.9, 1.3)
        self.assertAlmostEqual(np.sum(vec), 1.0)
        self.assertAlmostEqual(np.dot(X, vec), 0.9)

    def test_make_markov_approx_to_normal_by_MC(self):
        X = np.linspace(-4.0, 6.0, 25)
        vec = make_markov_approx_to_normal_by_monte_carlo(X, 0.9, 1.3)
        self.assertAlmostEqual(np.sum(vec), 1.0)
        self.assertAlmostEqual(vec.size, 25)


class testsForDCEGM(unittest.TestCase):
    def setUp(self):
        # setup the parameters to loop over
        self.mu_normals = np.linspace(-3.0, 2.0, 50)
        self.std_normals = np.linspace(0.01, 2.0, 50)

    def test_mu_normal(self):
        for mu_normal in self.mu_normals:
            for std_normal in self.std_normals:
                d = Normal(mu_normal).discretize(40, method="hermite")
                self.assertTrue(sum(d.pmv * d.atoms[0, :]) - mu_normal < 1e-12)

    def test_mu_lognormal_from_normal(self):
        for mu_normal in self.mu_normals:
            for std_normal in self.std_normals:
                d = approx_lognormal_gauss_hermite(40, mu_normal, std_normal)
                self.assertTrue(
                    abs(
                        sum(d.pmv * d.atoms[0, :])
                        - calc_lognormal_style_pars_from_normal_pars(
                            mu_normal, std_normal
                        )[0]
                    )
                    < 1e-12
                )


class test_MVNormalApprox(unittest.TestCase):
    def setUp(self):
        N = 5

        # 2-D distribution
        self.mu2 = np.array([5, -10])
        self.Sigma2 = np.array([[2, -0.6], [-0.6, 1]])
        self.dist2D = MultivariateNormal(self.mu2, self.Sigma2)
        self.dist2D_approx = self.dist2D.discretize(N, method="hermite")

        # 3-D Distribution
        self.mu3 = np.array([5, -10, 0])
        self.Sigma3 = np.array([[2, -0.6, 0.1], [-0.6, 1, 0.2], [0.1, 0.2, 3]])
        self.dist3D = MultivariateNormal(self.mu3, self.Sigma3)
        self.dist3D_approx = self.dist3D.discretize(N, method="hermite")

    def test_means(self):
        mu_2D = calc_expectation(self.dist2D_approx)
        self.assertTrue(np.allclose(mu_2D, self.mu2, rtol=1e-5))

        mu_3D = calc_expectation(self.dist3D_approx)
        self.assertTrue(np.allclose(mu_3D, self.mu3, rtol=1e-5))

    def test_VCOV(self):
        def vcov_fun(X, mu):
            return np.outer(X - mu, X - mu)

        Sig_2D = calc_expectation(self.dist2D_approx, vcov_fun, self.mu2)
        self.assertTrue(np.allclose(Sig_2D, self.Sigma2, rtol=1e-5))

        Sig_3D = calc_expectation(self.dist3D_approx, vcov_fun, self.mu3)
        self.assertTrue(np.allclose(Sig_3D, self.Sigma3, rtol=1e-5))

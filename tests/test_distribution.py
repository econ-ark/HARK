import unittest
import warnings

import numpy as np
import xarray as xr

from HARK.distributions import (
    Bernoulli,
    calc_expectation,
    DiscreteDistribution,
    DiscreteDistributionLabeled,
    IndexDistribution,
    Lognormal,
    MarkovProcess,
    MeanOneLogNormal,
    Normal,
    Uniform,
    Weibull,
    calc_lognormal_style_pars_from_normal_pars,
    calc_normal_style_pars_from_lognormal_pars,
    combine_indep_dstns,
    distr_of_function,
    expected,
    expected_with_loop,
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

    def test_draw_events_shuffle_matches_shuffle_draw_indices(self):
        """draw_events(..., shuffle=True) returns the same multiset of indices as draw(..., shuffle=True)."""
        pmv = np.array([0.95, 0.05])
        atoms = np.array([0.0, 1.0])
        d = DiscreteDistribution(pmv, atoms, seed=12345)
        N = 10_000
        idx_shuffle = d.draw_events(N, shuffle=True)
        draws = d.draw(N, shuffle=True, atoms=np.arange(2, dtype=int))
        self.assertTrue(np.array_equal(np.sort(idx_shuffle), np.sort(draws)))
        self.assertEqual(np.bincount(idx_shuffle, minlength=2)[0], 9500)
        self.assertEqual(np.bincount(idx_shuffle, minlength=2)[1], 500)

    def test_distr_of_function(self):
        # Function 1 -> 1
        # Approximate the lognormal expectation
        sig = 0.05
        norm = Normal(mu=-(sig**2) / 2, sigma=sig).discretize(131, method="hermite")
        my_logn = distr_of_function(norm, func=lambda x: np.exp(x))
        exp = expected(None, my_logn)
        self.assertAlmostEqual(exp[0], 1.0)

        # Function 1 -> n
        # Mean and variance of the normal
        norm = Normal(mu=0.0, sigma=1.0).discretize(5, method="hermite")
        moments = distr_of_function(norm, lambda x: np.array([x, x**2]))
        exp = expected(None, moments).flatten()
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
        exp = expected(None, mysum)
        self.assertAlmostEqual(exp[0], mu_a + mu_b)

        # Function n -> m
        # Mean and variance of two normals
        moments = distr_of_function(
            binorm,
            lambda x: np.array([x[0], (x[0] - mu_a) ** 2, x[1], (x[1] - mu_b) ** 2]),
        )
        exp = expected(None, moments)
        self.assertAlmostEqual(exp[0], mu_a)
        self.assertAlmostEqual(exp[1], si_a**2)
        self.assertAlmostEqual(exp[2], mu_b)
        self.assertAlmostEqual(exp[3], si_b**2)

    def test_expected_with_loop(self):
        dd_0_1_20 = Normal().discretize(20, method="hermite")
        dd_1_1_40 = Normal(mu=1).discretize(40, method="hermite")
        dd_10_10_100 = Normal(mu=10, sigma=10).discretize(100, method="hermite")

        ce1 = expected(None, dd_0_1_20, vectorized=False)
        ce2 = expected(None, dd_1_1_40, vectorized=False)
        ce3 = expected(None, dd_10_10_100, vectorized=False)

        self.assertAlmostEqual(ce1[0], 0.0)
        self.assertAlmostEqual(ce2[0], 1.0)
        self.assertAlmostEqual(ce3[0], 10.0)

        ce4 = expected(lambda x: 2**x, dd_0_1_20, vectorized=False)

        self.assertAlmostEqual(ce4[0], 1.27154, places=HARK_PRECISION)

        ce5 = expected(lambda x: 2 * x, dd_1_1_40, vectorized=False)

        self.assertAlmostEqual(ce5[0], 2.0)

        ce6 = expected(lambda x, y: 2 * x + y, dd_10_10_100, 20, vectorized=False)

        self.assertAlmostEqual(ce6[0], 40.0)

        ce7 = expected(
            lambda x, y: x + y,
            dd_0_1_20,
            np.hstack(np.array([0, 1, 2, 3, 4, 5])),
            vectorized=False,
        )

        self.assertAlmostEqual(ce7.flat[3], 3.0)

        PermShkDstn = MeanOneLogNormal().discretize(200, method="equiprobable")
        TranShkDstn = MeanOneLogNormal().discretize(200, method="equiprobable")
        IncShkDstn = combine_indep_dstns(PermShkDstn, TranShkDstn)

        ce8 = expected(lambda atoms: atoms[0] + atoms[1], IncShkDstn, vectorized=False)

        self.assertAlmostEqual(ce8, 2.0)

        ce9 = expected(
            lambda atoms, a, r: r / atoms[0] * a + atoms[1],
            IncShkDstn,
            (
                np.array([0, 1, 2, 3, 4, 5]),  # an aNrmNow grid?
                1.05,
            ),  # an interest rate?
            vectorized=False,
        )

        self.assertAlmostEqual(ce9[3], 9.51802, places=HARK_PRECISION)

    def test_self_expected_value(self):
        dd_0_1_20 = Normal().discretize(20, method="hermite")
        dd_1_1_40 = Normal(mu=1).discretize(40, method="hermite")
        dd_10_10_100 = Normal(mu=10, sigma=10).discretize(100, method="hermite")

        ce1 = expected(dstn=dd_0_1_20)
        ce2 = expected(dstn=dd_1_1_40)
        ce3 = expected(dstn=dd_10_10_100)

        self.assertAlmostEqual(ce1[0], 0.0)
        self.assertAlmostEqual(ce2[0], 1.0)
        self.assertAlmostEqual(ce3[0], 10.0)

        ce4 = expected(lambda x: 2**x, dd_0_1_20)

        self.assertAlmostEqual(ce4[0], 1.27154, places=HARK_PRECISION)

        ce5 = expected(func=lambda x: 2 * x, dstn=dd_1_1_40)

        self.assertAlmostEqual(ce5[0], 2.0)

        ce6 = expected(lambda x, y: 2 * x + y, dd_10_10_100, args=(20))

        self.assertAlmostEqual(ce6[0], 40.0)

        ce7 = expected(
            func=lambda x, y: x + y,
            dstn=dd_0_1_20,
            args=(np.hstack([0, 1, 2, 3, 4, 5])),
        )

        self.assertAlmostEqual(ce7.flat[3], 3.0)

        PermShkDstn = MeanOneLogNormal().discretize(200, method="equiprobable")
        TranShkDstn = MeanOneLogNormal().discretize(200, method="equiprobable")
        IncShkDstn = combine_indep_dstns(PermShkDstn, TranShkDstn)

        ce8 = expected(lambda atoms: atoms[0] + atoms[1], dstn=IncShkDstn)

        self.assertAlmostEqual(ce8, 2.0)

        ce9 = expected(
            func=lambda atoms, a, r: r / atoms[0] * a + atoms[1],
            dstn=IncShkDstn,
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
        self.assertAlmostEqual(exp[0], 1.0)

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

    def test_shuffle(self):
        X = np.arange(5)
        P = np.array([0.1, 0.2, 7 / 30, 7 / 30, 7 / 30])
        F = DiscreteDistribution(P, X, seed=0)
        data = F.draw(1000, shuffle=True)

        counts = np.zeros(5)
        for j in range(5):
            counts[j] = np.sum(data == j)

        self.assertTrue(counts[0] == 100)
        self.assertTrue(counts[1] == 200)
        self.assertTrue(counts[2] >= 233)
        self.assertTrue(counts[3] >= 233)
        self.assertTrue(counts[4] >= 233)
        self.assertTrue(counts[2] <= 234)
        self.assertTrue(counts[3] <= 234)
        self.assertTrue(counts[4] <= 234)

    def test_shuffle_unbiased_at_small_N(self):
        """Shuffled draws must satisfy E[count_j] == N * pmv[j] at every N.

        The test above draws N=1000, where floor(N*P) already allocates
        almost every slot and any error in distributing the remainder is
        invisible. The error is O(M/N), so it only shows up when N is small
        relative to the number of atoms, which is the regime sim_birth hits
        when it redraws a cohort of newly born agents.
        """
        P = np.array([0.55, 0.25, 0.15, 0.05])
        X = np.arange(4)
        N, reps = 6, 6000

        F = DiscreteDistribution(P, X, seed=0)
        counts = np.zeros(4)
        for _ in range(reps):
            data = F.draw(N, shuffle=True)
            for j in range(4):
                counts[j] += np.sum(data == j)

        realized = counts / counts.sum()
        # 0.01 is roughly 5 standard errors here, and the failure this guards
        # against is about 0.023, so the gap is not a matter of tuning.
        for j in range(4):
            self.assertAlmostEqual(realized[j], P[j], delta=0.01)

    def test_shuffle_draws_nothing_when_N_is_zero(self):
        """N=0 must return an empty array rather than raising.

        sim_birth runs every period and asks for zero draws in any period
        where nobody died, so this is reachable in ordinary simulation.
        """
        F = DiscreteDistribution(np.array([0.7, 0.2, 0.1]), np.arange(3), seed=0)
        for zero in (0, np.int64(0)):
            data = F.draw(zero, shuffle=True)
            self.assertEqual(data.shape, (0,))

    def test_repr(self):
        X = np.arange(5)
        P = np.array([0.1, 0.2, 7 / 30, 7 / 30, 7 / 30])
        F = DiscreteDistribution(P, X, seed=0)

        desc = str(F)
        self.assertTrue(
            desc == "DiscreteDistribution with 5 atoms, inf=0, sup=4, seed=0"
        )

    def test_calc_exp_labeled(self):
        F = DiscreteDistributionLabeled(
            atoms=np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [0.3, 0.8, -0.2, 0.0, 9.0]]),
            pmv=np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
            name="test distribution",
            var_names=["x", "y"],
        )

        def my_func(S, z):
            return S["x"] + 2 * S["y"] + 3 * z

        self.assertAlmostEqual(
            expected(my_func, F, z=3.0),
            expected(my_func, F, z=3.0, vectorized=False),
            places=HARK_PRECISION,
        )


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
        exp = expected(None, self.mat_distr)

        # Check the expectation is of the shape we want
        self.assertTrue(exp.shape[0] == self.draw_1.shape[0])
        self.assertTrue(exp.shape[1] == self.draw_1.shape[1])

        # Check that its value is what we expect
        self.assertTrue(np.allclose(exp, 0.0))

        # Expectation of the sum
        exp = expected(np.sum, self.mat_distr, vectorized=False)
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
            expected(None, uni.discretize(10, method="equiprobable"))[0],
            0.5,
        )

        uni_discrete = uni.discretize(10, method="equiprobable", endpoints=True)

        self.assertEqual(uni_discrete.atoms[0][0], 0.0)
        self.assertEqual(uni_discrete.atoms[0][-1], 1.0)
        self.assertAlmostEqual(
            expected(None, uni.discretize(10, method="equiprobable"))[0],
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

    def test_shuffle_crn_across_matrices(self):
        """Source states with identical transition rows should produce
        identical draws across two calls with the same seed, even when
        other source states have different rows that consume different
        amounts of RNG state.

        This is the common-random-numbers guarantee needed for scenario
        comparison experiments (e.g., counterfactual policy evaluation):
        agents whose transition probabilities are unchanged between two
        matrices must see identical state assignments so that the
        treatment-effect variance is driven only by the rows that
        actually differ.

        The test uses row 0 probabilities that produce different
        leftover-slot counts and different permutation consumption
        between the two matrices, then checks that row 1's output (with
        identical probabilities) is identical across multiple seeds.
        Without per-source-state sub-RNG isolation, row 0's RNG drift
        contaminates row 1's permutation and this assertion fails for
        most seeds.
        """
        # 7 agents in each source state - small enough to exercise
        # leftover-slot assignment, large enough that the permutation
        # drift is clearly visible across seeds.
        state = np.array([0] * 7 + [1] * 7, dtype=int)

        # Row 0 differs between matrices (TM_a: K=[3,3] M=1;
        # TM_b: K=[5,2] M=0).  Row 1 is identical in both.
        TM_a = np.array([[0.50, 0.50], [0.30, 0.70]])
        TM_b = np.array([[5.0 / 7.0, 2.0 / 7.0], [0.30, 0.70]])

        # Check across multiple seeds - the naive implementation fails
        # for the majority of seeds, while the sub-RNG-isolated
        # implementation passes for all of them.
        for seed in range(20):
            mp_a = MarkovProcess(TM_a, seed=seed)
            mp_b = MarkovProcess(TM_b, seed=seed)
            new_a = mp_a.draw(state, shuffle=True)
            new_b = mp_b.draw(state, shuffle=True)
            source_1 = state == 1
            np.testing.assert_array_equal(
                new_a[source_1],
                new_b[source_1],
                err_msg=(
                    f"seed={seed}: source-1 draws differ between two "
                    f"calls with identical row-1 probabilities. This "
                    f"indicates that RNG state from row 0's processing "
                    f"contaminated row 1's permutation (CRN violation)."
                ),
            )

    def test_shuffle_with_draws_converges_to_iid(self):
        """The new draws= argument should produce per-agent assignments
        that converge to per-agent iid as N -> infinity (Glivenko-
        Cantelli).  At N=10000 with a 2-state chain, per-agent match
        rate between iid and draws=-shuffle should be > 99%.

        The default shuffle (no draws=) uses a random permutation
        independent of any per-agent input, so agent-by-agent it
        does NOT match iid even at large N - typically agreeing only
        on the trivial "stay-in-same-target" mass.
        """
        P = np.array([[0.95, 0.05], [0.30, 0.70]])
        N = 10_000
        rng = np.random.default_rng(seed=42)
        # Half emp, half unemp - exercises both rows of P.
        state = np.repeat([0, 1], N // 2)
        u = rng.uniform(size=N)

        # Per-agent iid via searchsorted on cumulative cond_mrkv.
        cdf = np.cumsum(P, axis=1)
        new_iid = np.empty(N, dtype=int)
        for j in range(2):
            mask = state == j
            new_iid[mask] = np.searchsorted(cdf[j, :], u[mask])

        # New mode: rank-based stratified inverse-CDF with shared draws.
        mp_strat = MarkovProcess(P, seed=42)
        new_strat = mp_strat.draw(state, shuffle=True, draws=u)

        match_strat = (new_iid == new_strat).mean()
        self.assertGreater(
            match_strat,
            0.99,
            f"draws=-shuffle should match iid agent-by-agent at >99% "
            f"for N={N}; got {match_strat:.4f}",
        )

    def test_shuffle_with_draws_preserves_quotas(self):
        """The new draws= argument must preserve the quota-exact
        target counts that are the original purpose of shuffle.  Each
        per-source-state target count K[j,k] should equal
        floor(N_j * P[j,k]) plus at most +/-1 from leftover allocation.
        """
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        N = 1000
        rng = np.random.default_rng(seed=1)
        state = (rng.uniform(size=N) > 0.5).astype(int)
        u = rng.uniform(size=N)

        mp = MarkovProcess(P, seed=0)
        new_state = mp.draw(state, shuffle=True, draws=u)

        for j in range(2):
            N_j = (state == j).sum()
            for k in range(2):
                count = int(((state == j) & (new_state == k)).sum())
                expected = N_j * P[j, k]
                self.assertLessEqual(
                    abs(count - expected),
                    1,
                    f"source {j} -> target {k}: count {count} not within "
                    f"+/-1 of expected quota {expected:.1f}",
                )

    def test_shuffle_draws_and_sort_key_mutually_exclusive(self):
        """draws= and sort_key= specify mutually exclusive assignment
        modes; passing both should raise ValueError.
        """
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        N = 100
        rng = np.random.default_rng(seed=0)
        state = (rng.uniform(size=N) > 0.5).astype(int)
        u = rng.uniform(size=N)

        mp = MarkovProcess(P, seed=0)
        with self.assertRaises(ValueError):
            mp.draw(state, shuffle=True, draws=u, sort_key=u)

    def test_shuffle_unbiased_when_quotas_are_fractional(self):
        """Shuffled transitions must satisfy E[count_k] == N_j * P[j,k].

        The quota tests above all use populations where N_j * P[j,k] is
        already an integer, so ``floor`` allocates every slot, M is zero and
        the leftover-allocation path never runs.  That is exactly the regime
        where the two implementations of the floor-plus-leftover algorithm
        agreed, and it is why a biased allocator lived in
        ``_draw_shuffled`` while the copy in ``DiscreteDistribution.draw``
        was being fixed.  Allocating the leftover proportional to P instead
        of to the fractional remainders overweights the modal target and
        starves the rare one.

        N=13 against [0.7, 0.2, 0.1] is analytic rather than noisy: K =
        [9, 2, 1] leaves exactly M=1 slot, whose correct destination
        probabilities are the remainders [0.1, 0.6, 0.3].  Allocating it
        proportional to P instead sends it to state 0 with probability
        0.696, giving E[count_0] = 9.696 against a target of 9.1.  The
        tolerance below is far tighter than that 0.6 error and far looser
        than the Monte Carlo standard error at this many repetitions.
        """
        row = np.array([0.7, 0.2, 0.1])
        N, reps = 13, 6000
        TM = np.tile(row, (len(row), 1))

        mp = MarkovProcess(TM, seed=0)
        state = np.zeros(N, dtype=int)
        total = np.zeros(len(row))
        for _ in range(reps):
            total += np.bincount(mp.draw(state, shuffle=True), minlength=len(row))

        realized = total / reps
        np.testing.assert_allclose(realized, N * row, atol=0.06)

    def test_shuffle_ignored_arguments_warn(self):
        """sort_key= and draws= do nothing without shuffle=True.

        Silently ignoring them lets a caller believe a variance reduction is
        in effect when the draw is plain iid, which is the failure mode that
        is hardest to notice: the run completes and the numbers look fine.
        """
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        N = 50
        rng = np.random.default_rng(seed=0)
        state = (rng.uniform(size=N) > 0.5).astype(int)
        u = rng.uniform(size=N)

        for kwargs in ({"sort_key": u}, {"draws": u}, {"sort_key": u, "draws": u}):
            mp = MarkovProcess(P, seed=0)
            with self.assertWarns(UserWarning):
                mp.draw(state, **kwargs)

        # The default path must stay silent.
        mp = MarkovProcess(P, seed=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mp.draw(state)
        self.assertEqual([str(w.message) for w in caught], [])


class ReplicatesTests(unittest.TestCase):
    """Tests for DiscreteDistribution.draw(replicates=...)."""

    def test_replicates_gives_exact_counts(self):
        P = np.array([1 / 3, 1 / 6, 1 / 2])
        dstn = DiscreteDistribution(P, np.arange(3), seed=0)
        drawn = dstn.draw(replicates=2)
        self.assertEqual(len(drawn), 12)
        counts = np.bincount(drawn.astype(int), minlength=3)
        np.testing.assert_array_equal(counts, [4, 2, 6])

    def test_replicates_allows_zero_probability_atoms(self):
        """A zero-mass atom is legitimate and must not be refused.

        It simply gets zero slots.  Requiring every atom count to be
        strictly positive rejected the whole request instead.
        """
        P = np.array([0.5, 0.0, 0.5])
        dstn = DiscreteDistribution(P, np.arange(3), seed=0)
        drawn = dstn.draw(replicates=3)
        counts = np.bincount(drawn.astype(int), minlength=3)
        np.testing.assert_array_equal(counts, [3, 0, 3])

    def test_replicates_must_be_a_positive_integer(self):
        """Zero or negative replicates used to return an empty array."""
        dstn = DiscreteDistribution(np.array([0.25, 0.75]), np.arange(2), seed=0)
        for bad in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                dstn.draw(replicates=bad)

    def test_replicates_rejects_N_and_replicates_together(self):
        dstn = DiscreteDistribution(np.array([0.25, 0.75]), np.arange(2), seed=0)
        with self.assertRaises(ValueError):
            dstn.draw(4, replicates=1)

    def test_replicates_warns_only_when_J_min_exceeds_the_rarest_atom(self):
        """The warning must key on something that means what it says.

        Covering an atom of probability p_min needs ceil(1/p_min) draws no
        matter what, so a two-point [0.05, 0.95] needing J_min=20 is
        arithmetic, not a surprise.  An earlier version warned whenever some
        1/p_j was non-integral, which is true of nearly every non-uniform
        pmv, so it fired on essentially every realistic income grid.  The
        case worth flagging is the LCM blowing up past that floor.
        """
        quiet = DiscreteDistribution(np.array([0.05, 0.95]), np.arange(2), seed=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            quiet.draw(replicates=1)
        self.assertEqual([str(w.message) for w in caught], [])

        # 0.07 unemployment crossed with 7 equiprobable employed states:
        # the rarest atom needs 100 draws, but the LCM needs 700.
        P = np.concatenate([[0.07], np.full(7, 0.93 / 7)])
        loud = DiscreteDistribution(P, np.arange(8), seed=0)
        with self.assertWarns(UserWarning):
            loud.draw(replicates=1)


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
            dstn=IncShkDstn,
        )

        self.assertAlmostEqual(ce1, 3.70413, places=HARK_PRECISION)

        ce2 = expected(
            func=lambda dist, a, r: r / dist["perm_shk"] * a + dist["tran_shk"],
            dstn=IncShkDstn,
            args=(
                np.array([0, 1, 2, 3, 4, 5]),  # an aNrmNow grid?
                1.05,  # an interest rate?
            ),
        )

        self.assertAlmostEqual(ce2[3], 9.51802, places=HARK_PRECISION)

    def test_labels_parameter(self):
        """Test that labels=True uses dict indexing and labels=False uses integer indexing."""
        PermShkDstn = MeanOneLogNormal().discretize(200, method="equiprobable")
        TranShkDstn = MeanOneLogNormal().discretize(200, method="equiprobable")
        IncShkDstn = combine_indep_dstns(PermShkDstn, TranShkDstn)
        IncShkDstn = DiscreteDistributionLabeled.from_unlabeled(
            IncShkDstn,
            name="Income shocks",
            var_names=["perm_shk", "tran_shk"],
        )

        # labels=True (default): func receives dict with named keys
        ce_labeled = IncShkDstn.expected(lambda d: d["perm_shk"] + d["tran_shk"])

        # labels=True explicit: same result
        ce_labeled_explicit = IncShkDstn.expected(
            lambda d: d["perm_shk"] + d["tran_shk"], labels=True
        )

        # labels=False: func receives raw numpy atoms (integer indexing)
        ce_unlabeled = IncShkDstn.expected(lambda x: x[0] + x[1], labels=False)

        self.assertAlmostEqual(ce_labeled, ce_labeled_explicit, places=HARK_PRECISION)
        self.assertAlmostEqual(ce_labeled, ce_unlabeled, places=HARK_PRECISION)

        # labels=False with args
        ce_with_args = IncShkDstn.expected(
            lambda x, k: x[0] * k + x[1], 2.0, labels=False
        )
        ce_with_args_labeled = IncShkDstn.expected(
            lambda d, k: d["perm_shk"] * k + d["tran_shk"], 2.0, labels=True
        )
        self.assertAlmostEqual(
            ce_with_args, ce_with_args_labeled, places=HARK_PRECISION
        )

        # labels parameter should not leak to func (issue #1487)
        ce_no_leak = IncShkDstn.expected(
            lambda d: d["perm_shk"] + d["tran_shk"], labels=True
        )
        self.assertAlmostEqual(ce_no_leak, ce_labeled, places=HARK_PRECISION)

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
        mu_2D = expected(None, self.dist2D_approx)
        self.assertTrue(np.allclose(mu_2D, self.mu2, rtol=1e-5))

        mu_3D = expected(None, self.dist3D_approx)
        self.assertTrue(np.allclose(mu_3D, self.mu3, rtol=1e-5))

    def test_VCOV(self):
        def vcov_fun(X, mu):
            return np.outer(X - mu, X - mu)

        Sig_2D = expected(vcov_fun, self.dist2D_approx, self.mu2, vectorized=False)
        self.assertTrue(np.allclose(Sig_2D, self.Sigma2, rtol=1e-5))

        Sig_3D = expected(vcov_fun, self.dist3D_approx, self.mu3, vectorized=False)
        self.assertTrue(np.allclose(Sig_3D, self.Sigma3, rtol=1e-5))


class CalcExpectationDeprecatedAlias(unittest.TestCase):
    """calc_expectation (renamed in 0.17.2) survives as a warning-bearing
    alias that delegates exactly to expected_with_loop."""

    def test_alias_delegates_and_warns(self):
        dd = DiscreteDistribution(np.array([0.25, 0.75]), np.array([2.0, 4.0]), seed=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = calc_expectation(dd, lambda x: x * x)
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))
        expected = expected_with_loop(dd, lambda x: x * x)
        np.testing.assert_allclose(got, expected)
class StreamInvarianceGoldens(unittest.TestCase):
    """Default-path RNG-stream pins captured on main at a25d3ae0, before
    the shuffle/draws/replicates additions.

    These sequences must never change while the new parameters keep their
    defaults: downstream reproducibility depends on the exact draw stream,
    not just its distribution (cf. the #1719 release note).
    """

    def test_markov_process_default_stream_unchanged(self):
        mp = MarkovProcess(np.array([[0.7, 0.3], [0.4, 0.6]]), seed=12345)
        states = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1])
        seq1 = mp.draw(states.copy())
        seq2 = mp.draw(seq1.copy())
        self.assertEqual(list(seq1), [0, 0, 1, 1, 0, 0, 1, 0, 0, 1])
        self.assertEqual(list(seq2), [0, 1, 1, 0, 0, 1, 1, 0, 1, 0])

    def test_discrete_distribution_default_streams_unchanged(self):
        dd = DiscreteDistribution(
            np.array([0.2, 0.3, 0.5]), np.array([1.0, 2.0, 3.0]), seed=98765
        )
        ev = dd.draw_events(12)
        dr = dd.draw(12)
        self.assertEqual(list(ev), [0, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 1])
        self.assertEqual(
            [float(x) for x in dr],
            [2.0, 3.0, 3.0, 1.0, 2.0, 1.0, 3.0, 2.0, 2.0, 1.0, 1.0, 3.0],
        )

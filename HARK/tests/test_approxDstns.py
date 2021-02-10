"""
This file implements unit tests approximate distributions.
"""

# Bring in modules we need
import HARK.distribution as distribution
import unittest
import numpy as np


class testsForDCEGM(unittest.TestCase):
    def setUp(self):
        # setup the parameters to loop over
        self.muNormals = np.linspace(-3.0, 2.0, 50)
        self.stdNormals = np.linspace(0.01, 2.0, 50)

    def test_mu_normal(self):
        for muNormal in self.muNormals:
            for stdNormal in self.stdNormals:
                d = distribution.Normal(muNormal).approx(40)
                self.assertTrue(sum(d.pmf * d.X) - muNormal < 1e-12)

    def test_mu_lognormal_from_normal(self):
        for muNormal in self.muNormals:
            for stdNormal in self.stdNormals:
                d = distribution.approxLognormalGaussHermite(40, muNormal, stdNormal)
                self.assertTrue(
                    abs(
                        sum(d.pmf * d.X)
                        - distribution.calcLognormalStyleParsFromNormalPars(
                            muNormal, stdNormal
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
        self.dist2D = distribution.MVNormal(self.mu2, self.Sigma2)
        self.dist2D_approx = self.dist2D.approx(N)

        # 3-D Distribution
        self.mu3 = np.array([5, -10, 0])
        self.Sigma3 = np.array([[2, -0.6, 0.1], [-0.6, 1, 0.2], [0.1, 0.2, 3]])
        self.dist3D = distribution.MVNormal(self.mu3, self.Sigma3)
        self.dist3D_approx = self.dist3D.approx(N)

    def test_means(self):

        mu_2D = distribution.calcExpectation(self.dist2D_approx)
        self.assertTrue(np.allclose(mu_2D, self.mu2, rtol=1e-5))

        mu_3D = distribution.calcExpectation(self.dist3D_approx)
        self.assertTrue(np.allclose(mu_3D, self.mu3, rtol=1e-5))

    def test_VCOV(self):

        vcov_fun = lambda X, mu: np.outer(X - mu, X - mu)

        Sig_2D = distribution.calcExpectation(self.dist2D_approx, vcov_fun, self.mu2)[
            :, :, 0
        ]
        self.assertTrue(np.allclose(Sig_2D, self.Sigma2, rtol=1e-5))

        Sig_3D = distribution.calcExpectation(self.dist3D_approx, vcov_fun, self.mu3)[
            :, :, 0
        ]
        self.assertTrue(np.allclose(Sig_3D, self.Sigma3, rtol=1e-5))

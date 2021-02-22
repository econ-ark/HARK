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
        self.mu_normals = np.linspace(-3.0, 2.0, 50)
        self.std_normals = np.linspace(0.01, 2.0, 50)

    def test_mu_normal(self):
        for mu_normal in self.mu_normals:
            for std_normal in self.std_normals:
                d = distribution.Normal(mu_normal).approx(40)
                self.assertTrue(sum(d.pmf * d.X) - mu_normal < 1e-12)

    def test_mu_lognormal_from_normal(self):
        for mu_normal in self.mu_normals:
            for std_normal in self.std_normals:
                d = distribution.approx_lognormal_gauss_hermite(40, mu_normal, std_normal)
                self.assertTrue(
                    abs(
                        sum(d.pmf * d.X)
                        - distribution.calc_lognormal_style_pars_from_normal_pars(
                            mu_normal, std_normal
                        )[0]
                    )
                    < 1e-12
                )

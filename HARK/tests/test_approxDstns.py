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

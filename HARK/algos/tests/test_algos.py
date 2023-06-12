"""
This file implements unit tests to check discrete choice functions
"""
# Bring in modules we need
import unittest

import numpy as np

class foc_test(unittest.TestCase):
    """
    FOC test:

    pi(mVec) == cVec2 (close enough, 10 digits of precision)
    """

    def setUp(self):

        self.mVec = np.load("smdsops_mVec.npy")
        self.cVec2 = np.load("smdsops_cVec2.npy")


    def test_x(self):

        self.assertTrue(np.all(self.cVec2 > 0))

class egm_test(unittest.TestCase):
    """
    cVec_egm = egm(aVec) (machine precision)
    pi(mVec_egm) == cVec_egm
    """
    def setUp(self):
        self.aVec = np.load("smdsops_aVec.npy")
        self.cVec_egm = np.load("smdsops_cVec_egm.npy")
        self.mVec_egm = np.load("smdsops_mVec_egm.npy")

    def test_egm(self):

        self.assertTrue(np.all(self.aVec + self.cVec_egm == self.mVec_egm))
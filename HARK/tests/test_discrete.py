"""
This file implements unit tests to check discrete choice functions
"""
from __future__ import print_function, division
from __future__ import absolute_import

from HARK import interpolation

# Bring in modules we need
import unittest
import numpy as np


class testsForDiscreteChoice(unittest.TestCase):
    def setUp(self):
        self.Vs2D = np.stack((np.zeros(3), np.ones(3)))
        self.Vref2D = np.array([1.0, 1.0, 1.0])
        self.Pref2D = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

        self.Vs3D = np.array([[0.0, 1.0, 4.0], [1.0, 2.0, 0.0], [3.0, 0.0, 2.0]])
        self.Vref3D = np.array([[3.0, 2.0, 4.0]])
        self.Pref3D = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        # maxV = self.Vs3D.max()
        # self.Vref3D = maxV + np.log(np.sum(np.exp(self.Vs3D-maxV),axis=0))
        # self.Pref3D = np.log(np.sum(np.exp(self.Vs3D-maxV),axis=0))

    def test_noShock2DBothEqualValue(self):
        # Test the value functions and policies of the 2D case
        sigma = 0.0
        V, P = interpolation.calc_log_sum_choice_probs(self.Vs2D, sigma)
        self.assertTrue((V == self.Vref2D).all)
        self.assertTrue((P == self.Pref2D).all)

    def test_noShock2DBoth(self):
        # Test the value functions and policies of the 2D case
        sigma = 0.0
        V, P = interpolation.calc_log_sum_choice_probs(self.Vs2D, sigma)
        self.assertTrue((V == self.Vref2D).all)
        self.assertTrue((P == self.Pref2D).all)

    def test_noShock2DIndividual(self):
        # Test the value functions and policies of the 2D case
        sigma = 0.0
        V = interpolation.calc_log_sum(self.Vs2D, sigma)
        P = interpolation.calc_choice_probs(self.Vs2D, sigma)
        self.assertTrue((V == self.Vref2D).all())
        self.assertTrue((P == self.Pref2D).all())

    def test_noShock3DBothEqualValue(self):
        # Test the value functions and policies of the 3D case
        sigma = 0.0
        V, P = interpolation.calc_log_sum_choice_probs(self.Vs3D, sigma)
        self.assertTrue((V == self.Vref3D).all)
        self.assertTrue((P == self.Pref3D).all)

    def test_noShock3DBoth(self):
        # Test the value functions and policies of the 3D case
        sigma = 0.0
        V, P = interpolation.calc_log_sum_choice_probs(self.Vs3D, sigma)
        self.assertTrue((V == self.Vref3D).all)
        self.assertTrue((P == self.Pref3D).all)

    def test_noShock3DIndividual(self):
        # Test the value functions and policies of the 3D case
        sigma = 0.0
        V = interpolation.calc_log_sum(self.Vs3D, sigma)
        P = interpolation.calc_choice_probs(self.Vs3D, sigma)
        self.assertTrue((V == self.Vref3D).all())
        self.assertTrue((P == self.Pref3D).all())

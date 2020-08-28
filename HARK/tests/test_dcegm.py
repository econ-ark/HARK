"""
This file implements unit tests to check discrete choice functions
"""
from HARK import dcegm

# Bring in modules we need
import unittest
import numpy as np


class testsForDCEGM(unittest.TestCase):
    def setUp(self):
        self.commonM = np.linspace(0, 10.0, 30)
        self.m_in = np.array([1.0, 2.0, 3.0, 2.5, 2.0, 4.0, 5.0, 6.0])
        self.c_in = np.array([1.0, 2.0, 3.0, 2.5, 2.0, 4.0, 5.0, 6.0])
        self.v_in = np.array([0.5, 1.0, 1.5, 0.75, 0.5, 3.5, 5.0, 7.0])

    def test_crossing(self):
        # Test that the upper envelope has the approximate correct value
        # where the two increasing segments with m_1 = [2, 3] and m_2 = [2.0, 4.0]
        # is the correct value.
        #
        # Calculate the crossing by hand
        slope_1 = (1.5 - 1.0) / (3.0 - 2.0)
        slope_2 = (3.5 - 0.5) / (4.0 - 2.0)
        m_cross = 2.0 + (0.5 - 1.0) / (slope_1 - slope_2)

        m_out, c_out, v_out = dcegm.calcMultilineEnvelope(
            self.m_in, self.c_in, self.v_in, self.commonM
        )

        m_idx = 0
        for m in m_out:
            if m > m_cross:
                break
            m_idx += 1

        # Just right of the cross, the second segment is optimal
        true_v = 0.5 + (m_out[m_idx] - 2.0) * slope_2
        self.assertTrue(abs(v_out[m_idx] - true_v) < 1e-12)

    # also test that first elements are 0 etc

    # def test_crossing_in_grid(self):
    #     # include crossing m in common grid
    #     commonM_augmented = np.append(self.commonM, m_cross).sort()
    #
    #     m_out, c_out, v_out = calcMultilineEnvelope(self.m_in, self.c_in, self.v_in, self.commonM)
    #
    #     self.assertTrue(

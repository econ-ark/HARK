"""
This file implements unit tests to check discrete choice functions
"""
# Bring in modules we need
import unittest

import numpy as np

from HARK import dcegm


class envelope_test(unittest.TestCase):
    def setUp(self):
        self.segments = [
            [np.array([0.0, 1.0]), np.array([0.0, 1.0])],
            [np.array([0.0, 1.0]), np.array([1.0, 0.0])],
        ]

    def test_upper_envelope(self):
        # Compute
        x_env, y_env, inds = dcegm.upper_envelope(self.segments)

        # true envelope
        true_x = np.array([0.0, 0.5, np.nextafter(0.5, 1), 1.0])
        true_y = np.array([1.0, 0.5, 0.5, 1.0])
        true_inds = np.array([1, 1, 0, 0])

        self.assertTrue(np.allclose(x_env, true_x))
        self.assertTrue(np.allclose(y_env, true_y))
        self.assertTrue(np.allclose(inds, true_inds))


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

        # Now with EGM tools.
        # 1. Find and form non-decreasing segments
        starts, ends = dcegm.calc_nondecreasing_segments(self.m_in, self.v_in)
        segments = [
            [
                self.m_in[range(starts[j], ends[j] + 1)],
                self.v_in[range(starts[j], ends[j] + 1)],
            ]
            for j in range(len(starts))
        ]
        # 2. Find the upper envelope
        m_env, v_env, inds = dcegm.upper_envelope(segments)

        # Compare the crossings
        m_idx = 0
        for m in m_env:
            if m > m_cross:
                break
            m_idx += 1

        # Just right of the cross, the second segment is optimal
        true_v = 0.5 + (m_env[m_idx] - 2.0) * slope_2
        self.assertTrue(abs(v_env[m_idx] - true_v) < 1e-12)

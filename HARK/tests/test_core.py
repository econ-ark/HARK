"""
This file implements unit tests for interpolation methods
"""
from HARK.core import HARKobject

import numpy as np
import unittest

class testHARKobject(unittest.TestCase):
    def setUp(self):
        self.obj_a = HARKobject()
        self.obj_b = HARKobject()

    def test_distance(self):
        self.assertRaises(AttributeError, self.obj_a.distance(self.obj_b))

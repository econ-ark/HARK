"""
This file implements unit tests for interpolation methods
"""
from HARK.core import HARKobject, distanceMetric, AgentType

import numpy as np
import unittest


class testdistanceMetric(unittest.TestCase):
    def setUp(self):
        self.list_a = [1.0, 2.1, 3]
        self.list_b = [3.1, 4, -1.4]
        self.list_c = [8.6, 9]
        self.obj_a = HARKobject()
        self.obj_b = HARKobject()
        self.obj_c = HARKobject()

    def test_list(self):
        # same length
        self.assertEqual(distanceMetric(self.list_a, self.list_b), 4.4)
        # different length
        self.assertEqual(distanceMetric(self.list_b, self.list_c), 1.0)
        # sanity check, same objects
        self.assertEqual(distanceMetric(self.list_b, self.list_b), 0.0)

    def test_array(self):
        # same length
        self.assertEqual(
            distanceMetric(np.array(self.list_a), np.array(self.list_b)), 4.4
        )
        # different length
        self.assertEqual(
            distanceMetric(np.array(self.list_b).reshape(1, 3), np.array(self.list_c)),
            1.0,
        )
        # sanity check, same objects
        self.assertEqual(
            distanceMetric(np.array(self.list_b), np.array(self.list_b)), 0.0
        )

    def test_hark_object_distance(self):
        self.obj_a.distance_criteria = ["var_1", "var_2", "var_3"]
        self.obj_b.distance_criteria = ["var_1", "var_2", "var_3"]
        self.obj_c.distance_criteria = ["var_5"]
        # if attributes don't exist or don't match
        self.assertEqual(distanceMetric(self.obj_a, self.obj_b), 1000.0)
        self.assertEqual(distanceMetric(self.obj_a, self.obj_c), 1000.0)
        # add single numbers to attributes
        self.obj_a.var_1, self.obj_a.var_2, self.obj_a.var_3 = 0.1, 1, 2.1
        self.obj_b.var_1, self.obj_b.var_2, self.obj_b.var_3 = 1.8, -1, 0.1
        self.assertEqual(distanceMetric(self.obj_a, self.obj_b), 2.0)

        # sanity check - same objects
        self.assertEqual(distanceMetric(self.obj_a, self.obj_a), 0.0)


class testHARKobject(unittest.TestCase):
    def setUp(self):
        # similar test to distanceMetric
        self.obj_a = HARKobject()
        self.obj_b = HARKobject()
        self.obj_c = HARKobject()

    def test_distance(self):
        self.obj_a.distance_criteria = ["var_1", "var_2", "var_3"]
        self.obj_b.distance_criteria = ["var_1", "var_2", "var_3"]
        self.obj_c.distance_criteria = ["var_5"]
        self.obj_a.var_1, self.obj_a.var_2, self.obj_a.var_3 = [0.1], [1, 2], [2.1]
        self.obj_b.var_1, self.obj_b.var_2, self.obj_b.var_3 = [1.8], [0, 0.1], [1.1]
        self.assertEqual(self.obj_a.distance(self.obj_b), 1.9)
        # change the length of a attribute list
        self.obj_b.var_1, self.obj_b.var_2, self.obj_b.var_3 = [1.8], [0, 0, 0.1], [1.1]
        self.assertEqual(self.obj_a.distance(self.obj_b), 1.7)
        # sanity check
        self.assertEqual(self.obj_b.distance(self.obj_b), 0.0)


class testAgentType(unittest.TestCase):
    def setUp(self):
        self.agent = AgentType()

    def test_solve(self):
        self.agent.time_vary = ["vary_1"]
        self.agent.time_inv = ["inv_1"]
        self.agent.vary_1 = [1.1, 1.2, 1.3, 1.4]
        self.agent.inv_1 = 1.05
        # to test the superclass we create a dummy solveOnePeriod function
        # for our agent, which doesn't do anything, instead of using a NullFunc
        self.agent.solveOnePeriod = lambda vary_1: HARKobject()
        self.agent.solve()
        self.assertEqual(len(self.agent.solution), 4)
        self.assertTrue(isinstance(self.agent.solution[0], HARKobject))

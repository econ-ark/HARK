"""
This file implements unit tests for core HARK functionality.
"""
from HARK.core import Model, MetricObject, distance_metric, AgentType, distribute_params
from HARK.distribution import Uniform

import numpy as np
import unittest


class test_distance_metric(unittest.TestCase):
    def setUp(self):
        self.list_a = [1.0, 2.1, 3]
        self.list_b = [3.1, 4, -1.4]
        self.list_c = [8.6, 9]
        self.obj_a = MetricObject()
        self.obj_b = MetricObject()
        self.obj_c = MetricObject()

    def test_list(self):
        # same length
        self.assertEqual(distance_metric(self.list_a, self.list_b), 4.4)
        # different length
        self.assertEqual(distance_metric(self.list_b, self.list_c), 1.0)
        # sanity check, same objects
        self.assertEqual(distance_metric(self.list_b, self.list_b), 0.0)

    def test_array(self):
        # same length
        self.assertEqual(
            distance_metric(np.array(self.list_a), np.array(self.list_b)), 4.4
        )
        # different length
        self.assertEqual(
            distance_metric(np.array(self.list_b).reshape(1, 3), np.array(self.list_c)),
            1.0,
        )
        # sanity check, same objects
        self.assertEqual(
            distance_metric(np.array(self.list_b), np.array(self.list_b)), 0.0
        )

    def test_hark_object_distance(self):
        self.obj_a.distance_criteria = ["var_1", "var_2", "var_3"]
        self.obj_b.distance_criteria = ["var_1", "var_2", "var_3"]
        self.obj_c.distance_criteria = ["var_5"]
        # if attributes don't exist or don't match
        self.assertEqual(distance_metric(self.obj_a, self.obj_b), 1000.0)
        self.assertEqual(distance_metric(self.obj_a, self.obj_c), 1000.0)
        # add single numbers to attributes
        self.obj_a.var_1, self.obj_a.var_2, self.obj_a.var_3 = 0.1, 1, 2.1
        self.obj_b.var_1, self.obj_b.var_2, self.obj_b.var_3 = 1.8, -1, 0.1
        self.assertEqual(distance_metric(self.obj_a, self.obj_b), 2.0)

        # sanity check - same objects
        self.assertEqual(distance_metric(self.obj_a, self.obj_a), 0.0)


class test_MetricObject(unittest.TestCase):
    def setUp(self):
        # similar test to distance_metric
        self.obj_a = MetricObject()
        self.obj_b = MetricObject()
        self.obj_c = MetricObject()

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


class test_AgentType(unittest.TestCase):
    def setUp(self):
        self.agent = AgentType(cycles = 1)

    def test_solve(self):
        self.agent.time_vary = ["vary_1"]
        self.agent.time_inv = ["inv_1"]
        self.agent.vary_1 = [1.1, 1.2, 1.3, 1.4]
        self.agent.inv_1 = 1.05
        # to test the superclass we create a dummy solve_one_period function
        # for our agent, which doesn't do anything, instead of using a NullFunc
        self.agent.solve_one_period = lambda vary_1: MetricObject()
        self.agent.solve()
        self.assertEqual(len(self.agent.solution), 4)
        self.assertTrue(isinstance(self.agent.solution[0], MetricObject))

    def test___repr__(self):
        self.assertTrue('Parameters' in self.agent.__repr__())

    def test___eq__(self):
        agent2 = AgentType(cycles = 1)
        agent3 = AgentType(cycels = 2)

        self.assertEqual(self.agent, agent2)
        self.assertNotEqual(self.agent, agent3)

class test_distribute_params(unittest.TestCase):
    def setUp(self):
        self.agent = AgentType(cycles=1, AgentCount=3)


    def test_distribute_params(self):
        dist = Uniform(bot=0.9, top=0.94)

        self.agents = distribute_params(self.agent, 'DiscFac', 3, dist)

        self.assertTrue(all(['DiscFac' in agent.parameters for agent in self.agents]))
        self.assertTrue(all([self.agents[i].parameters['DiscFac'] == dist.approx(3).X[0,i] for i in range(3)]))
        self.assertEqual(self.agents[0].parameters['AgentCount'], 1)



"""
Unit tests for HARK.metric. Very little is tested here; the tests for describing
distance metric are mostly to ensure test coverage.
"""

# Bring in modules we need
import unittest

from HARK.metric import distance_metric, describe_metric, MetricObject
from HARK.models import IndShockConsumerType


class testsForDictionaryMetric(unittest.TestCase):
    def setUp(self):
        dict_A = {
            "height": 5.0,
            "width": 3.0,
            "depth": 2.0,
        }
        dict_B = dict_A.copy()
        dict_B["distance_criteria"] = ["height"]
        dict_C = dict_B.copy()
        dict_C["width"] += 1.0
        self.A = dict_A
        self.B = dict_B
        self.C = dict_C

    def test_same(self):
        dist = distance_metric(self.A, self.A)
        self.assertAlmostEqual(dist, 0.0)
        dist = distance_metric(self.B, self.B)
        self.assertAlmostEqual(dist, 0.0)
        dist = distance_metric(self.C, self.C)
        self.assertAlmostEqual(dist, 0.0)

    def test_diff(self):
        dist = distance_metric(self.A, self.B)
        self.assertAlmostEqual(dist, 1000.0)
        dist = distance_metric(self.B, self.C)
        self.assertAlmostEqual(dist, 0.0)
        dist = distance_metric(self.A, self.C)
        self.assertAlmostEqual(dist, 1000.0)
        del self.C["distance_criteria"]
        dist = distance_metric(self.A, self.C)
        self.assertAlmostEqual(dist, 1.0)


class testsForDescribeDistance(unittest.TestCase):
    def setUp(self):
        agent = IndShockConsumerType()
        agent.solve()
        self.agent = agent

        dict_A = {
            "height": 5.0,
            "width": 3.0,
            "depth": 2.0,
        }
        dict_B = dict_A.copy()
        dict_B["distance_criteria"] = ["height"]
        self.A = dict_A
        self.B = dict_B

    def test_solution(self):
        self.agent.solution[0].describe_distance()
        out = self.agent.solution[0].describe_distance(display=False)
        self.agent.solution[0].describe_distance(max_depth=0)
        self.agent.solution[0].describe_distance(max_depth=1)
        self.agent.solution[0].describe_distance(max_depth=2)
        self.agent.solution[0].describe_distance(max_depth=3)
        self.agent.solution[0].describe_distance(max_depth=4)

    def test_dictionary(self):
        describe_metric(self.A)
        describe_metric(self.B)

    def test_null(self):
        X = MetricObject()
        X.describe_distance()

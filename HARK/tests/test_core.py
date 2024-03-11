"""
This file implements unit tests for core HARK functionality.
"""

import unittest

import numpy as np
import pytest

from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    init_idiosyncratic_shocks,
)
from HARK.core import AgentPopulation, AgentType, Parameters, distribute_params
from HARK.distribution import Uniform
from HARK.metric import MetricObject, distance_metric


class test_distance_metric(unittest.TestCase):
    def setUp(self):
        self.list_a = [1.0, 2.1, 3]
        self.list_b = [3.1, 4, -1.4]
        self.list_c = [8.6, 9]
        self.obj_a = MetricObject()
        self.obj_b = MetricObject()
        self.obj_c = MetricObject()
        self.dict_a = {"a": 1, "b": 2}
        self.dict_b = {"a": 3, "b": 4}
        self.dict_c = {"a": 5, "f": 6}

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

    def test_dict(self):
        # Same keys (max of diffs across keys)
        self.assertEqual(distance_metric(self.dict_a, self.dict_b), 2.0)
        # Different keys
        self.assertEqual(distance_metric(self.dict_a, self.dict_c), 1000.0)

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
        self.agent = AgentType(cycles=1)

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

    def test_describe(self):
        self.assertTrue("Parameters" in self.agent.describe())

    def test___eq__(self):
        agent2 = AgentType(cycles=1)
        agent3 = AgentType(cycels=2)

        self.assertEqual(self.agent, agent2)
        self.assertNotEqual(self.agent, agent3)


class test_distribute_params(unittest.TestCase):
    def setUp(self):
        self.agent = AgentType(cycles=1, AgentCount=3)

    def test_distribute_params(self):
        dist = Uniform(bot=0.9, top=0.94)

        self.agents = distribute_params(self.agent, "DiscFac", 3, dist)

        self.assertTrue(all(["DiscFac" in agent.parameters for agent in self.agents]))
        self.assertTrue(
            all(
                [
                    self.agents[i].parameters["DiscFac"]
                    == dist.discretize(3, method="equiprobable").atoms[0, i]
                    for i in range(3)
                ]
            )
        )
        self.assertEqual(self.agents[0].parameters["AgentCount"], 1)


class test_agent_population(unittest.TestCase):
    def setUp(self):
        params = init_idiosyncratic_shocks.copy()
        params["CRRA"] = Uniform(2.0, 10)
        params["DiscFac"] = Uniform(0.9, 0.99)

        self.agent_pop = AgentPopulation(IndShockConsumerType, params)

    def test_distributed_params(self):
        self.assertTrue("CRRA" in self.agent_pop.distributed_params)
        self.assertTrue("DiscFac" in self.agent_pop.distributed_params)

    def test_approx_agents(self):
        self.agent_pop.approx_distributions({"CRRA": 3, "DiscFac": 4})

        self.assertTrue("CRRA" in self.agent_pop.continuous_distributions)
        self.assertTrue("DiscFac" in self.agent_pop.continuous_distributions)
        self.assertTrue("CRRA" in self.agent_pop.discrete_distributions)
        self.assertTrue("DiscFac" in self.agent_pop.discrete_distributions)

        self.assertEqual(self.agent_pop.agent_type_count, 12)

    def test_create_agents(self):
        self.agent_pop.approx_distributions({"CRRA": 3, "DiscFac": 4})
        self.agent_pop.create_distributed_agents()

        self.assertEqual(len(self.agent_pop.agents), 12)


class test_parameters(unittest.TestCase):
    def setUp(self):
        self.params = Parameters(T_cycle=3, a=1, b=[2, 3, 4], c=np.array([5, 6, 7]))

    def test_init(self):
        self.assertEqual(self.params._length, 3)
        self.assertEqual(self.params._invariant_params, {"a", "c"})
        self.assertEqual(self.params._varying_params, {"b"})

    def test_getitem(self):
        self.assertEqual(self.params["a"], 1)
        self.assertEqual(self.params[0]["b"], 2)
        self.assertEqual(self.params["c"][1], 6)

    def test_setitem(self):
        self.params["d"] = 8
        self.assertEqual(self.params["d"], 8)

    def test_update(self):
        self.params.update({"a": 9, "b": [10, 11, 12]})
        self.assertEqual(self.params["a"], 9)
        self.assertEqual(self.params[0]["b"], 10)

    def test_initialization(self):
        params = Parameters(a=1, b=[1, 2], T_cycle=2)
        assert params._length == 2
        assert params._invariant_params == {"a"}
        assert params._varying_params == {"b"}

    def test_infer_dims_scalar(self):
        params = Parameters(a=1)
        assert params["a"] == 1

    def test_infer_dims_array(self):
        params = Parameters(b=np.array([1, 2]))
        assert all(params["b"] == np.array([1, 2]))

    def test_infer_dims_list_varying(self):
        params = Parameters(b=[1, 2], T_cycle=2)
        assert params["b"] == [1, 2]

    def test_infer_dims_list_invariant(self):
        params = Parameters(b=[1])
        assert params["b"] == 1

    def test_setitem(self):
        params = Parameters(a=1)
        params["b"] = 2
        assert params["b"] == 2

    def test_keys_values_items(self):
        params = Parameters(a=1, b=2)
        assert set(params.keys()) == {"a", "b"}
        assert set(params.values()) == {1, 2}
        assert set(params.items()) == {("a", 1), ("b", 2)}

    def test_to_dict(self):
        params = Parameters(a=1, b=2)
        assert params.to_dict() == {"a": 1, "b": 2}

    def test_to_namedtuple(self):
        params = Parameters(a=1, b=2)
        named_tuple = params.to_namedtuple()
        assert named_tuple.a == 1
        assert named_tuple.b == 2

    def test_update_params(self):
        params1 = Parameters(a=1, b=2)
        params2 = Parameters(a=3, c=4)
        params1.update(params2)
        assert params1["a"] == 3
        assert params1["c"] == 4

    def test_unsupported_type_error(self):
        with pytest.raises(ValueError):
            Parameters(b={1, 2})

    def test_get_item_dimension_error(self):
        params = Parameters(b=[1, 2], T_cycle=2)
        with pytest.raises(ValueError):
            params[2]

    def test_getitem_with_key(self):
        params = Parameters(a=1, b=[2, 3], T_cycle=2)
        assert params["a"] == 1
        assert params["b"] == [2, 3]

    def test_getitem_with_item(self):
        params = Parameters(a=1, b=[2, 3], T_cycle=2)
        age_params = params[1]
        assert age_params["a"] == 1
        assert age_params["b"] == 3

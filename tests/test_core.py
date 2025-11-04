"""
This file implements unit tests for core HARK functionality.
"""

import unittest

import numpy as np
import pytest
from copy import deepcopy

from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    init_idiosyncratic_shocks,
)
from HARK.core import AgentPopulation, AgentType, Parameters, distribute_params
from HARK.distributions import Uniform
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
        self.agent.T_cycle = 4
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
        self.agent_pop.approx_distributions(
            {
                "CRRA": {"N": 3, "method": "equiprobable"},
                "DiscFac": {"N": 4, "method": "equiprobable"},
            }
        )

        self.assertTrue("CRRA" in self.agent_pop.continuous_distributions)
        self.assertTrue("DiscFac" in self.agent_pop.continuous_distributions)
        self.assertTrue("CRRA" in self.agent_pop.discrete_distributions)
        self.assertTrue("DiscFac" in self.agent_pop.discrete_distributions)

        self.assertEqual(self.agent_pop.agent_type_count, 12)

    def test_create_agents(self):
        self.agent_pop.approx_distributions(
            {
                "CRRA": {"N": 3, "method": "equiprobable"},
                "DiscFac": {"N": 4, "method": "equiprobable"},
            }
        )
        self.agent_pop.create_distributed_agents()

        self.assertEqual(len(self.agent_pop.agents), 12)


@pytest.fixture
def sample_params():
    return Parameters(a=1, b=[2, 3, 4], c=5.0, d=[6.0, 7.0, 8.0], T_cycle=3)


class TestParameters:
    def test_initialization(self, sample_params):
        assert sample_params._length == 3
        assert sample_params._invariant_params == {"a", "c"}
        assert sample_params._varying_params == {"b", "d"}
        assert sample_params._parameters["T_cycle"] == 3

    def test_getitem(self, sample_params):
        assert sample_params["a"] == 1
        assert sample_params["b"] == [2, 3, 4]
        assert sample_params[0]["b"] == 2
        assert sample_params[1]["d"] == 7.0

    def test_setitem(self, sample_params):
        sample_params["e"] = 9
        assert sample_params["e"] == 9
        assert "e" in sample_params._invariant_params

        sample_params["f"] = [10, 11, 12]
        assert sample_params["f"] == [10, 11, 12]
        assert "f" in sample_params._varying_params

    def test_get(self, sample_params):
        assert sample_params.get("a") == 1
        assert sample_params.get("z", 100) == 100

    def test_set_many(self, sample_params):
        sample_params.set_many(g=13, h=[14, 15, 16])
        assert sample_params["g"] == 13
        assert sample_params["h"] == [14, 15, 16]

    def test_is_time_varying(self, sample_params):
        assert sample_params.is_time_varying("b") is True
        assert sample_params.is_time_varying("a") is False

    def test_to_dict(self, sample_params):
        params_dict = sample_params.to_dict()
        assert isinstance(params_dict, dict)
        assert params_dict["a"] == 1
        assert params_dict["b"] == [2, 3, 4]

    def test_update(self, sample_params):
        new_params = Parameters(a=100, e=200)
        sample_params.update(new_params)
        assert sample_params["a"] == 100
        assert sample_params["e"] == 200

    @pytest.mark.parametrize("invalid_key", [1, 2.0, None, []])
    def test_setitem_invalid_key(self, sample_params, invalid_key):
        with pytest.raises(ValueError):
            sample_params[invalid_key] = 42

    def test_setitem_invalid_value_length(self, sample_params):
        with pytest.raises(ValueError):
            sample_params["invalid"] = [1, 2]  # Should be length 1 or 3

    def test_iter(self, sample_params):
        """Test iteration over parameter names."""
        param_names = list(sample_params)
        assert "a" in param_names
        assert "b" in param_names
        assert "c" in param_names
        assert "d" in param_names
        assert "T_cycle" in param_names

    def test_len(self, sample_params):
        """Test length returns number of parameters."""
        assert len(sample_params) == 5  # a, b, c, d, T_cycle

    def test_keys(self, sample_params):
        """Test keys() method returns parameter names."""
        keys = list(sample_params.keys())
        assert "a" in keys
        assert "b" in keys
        assert "c" in keys
        assert "d" in keys
        assert "T_cycle" in keys

    def test_values(self, sample_params):
        """Test values() method returns parameter values."""
        values = list(sample_params.values())
        assert 1 in values
        assert [2, 3, 4] in values
        assert 5.0 in values
        assert [6.0, 7.0, 8.0] in values
        assert 3 in values

    def test_items(self, sample_params):
        """Test items() method returns (name, value) pairs."""
        items = dict(sample_params.items())
        assert items["a"] == 1
        assert items["b"] == [2, 3, 4]
        assert items["c"] == 5.0
        assert items["d"] == [6.0, 7.0, 8.0]
        assert items["T_cycle"] == 3

    def test_repr(self, sample_params):
        """Test __repr__ returns detailed string representation."""
        repr_str = repr(sample_params)
        assert "Parameters" in repr_str
        assert "_length=3" in repr_str
        assert "_invariant_params" in repr_str
        assert "_varying_params" in repr_str

    def test_str(self, sample_params):
        """Test __str__ returns simple string representation."""
        str_repr = str(sample_params)
        assert "Parameters" in str_repr

    def test_getattr(self, sample_params):
        """Test attribute-style access to parameters."""
        assert sample_params.a == 1
        assert sample_params.b == [2, 3, 4]
        assert sample_params.c == 5.0
        assert sample_params.d == [6.0, 7.0, 8.0]

    def test_getattr_nonexistent(self, sample_params):
        """Test attribute-style access raises AttributeError for non-existent parameters."""
        with pytest.raises(AttributeError):
            _ = sample_params.nonexistent_param

    def test_setattr(self, sample_params):
        """Test attribute-style setting of parameters."""
        sample_params.new_param = 42
        assert sample_params["new_param"] == 42
        assert "new_param" in sample_params._invariant_params

        sample_params.new_varying = [1, 2, 3]
        assert sample_params["new_varying"] == [1, 2, 3]
        assert "new_varying" in sample_params._varying_params

    def test_contains(self, sample_params):
        """Test membership testing with 'in' operator."""
        assert "a" in sample_params
        assert "b" in sample_params
        assert "nonexistent" not in sample_params

    def test_copy(self, sample_params):
        """Test deep copy functionality."""
        copied_params = sample_params.copy()

        # Check that it's a different object
        assert copied_params is not sample_params
        assert copied_params._parameters is not sample_params._parameters

        # Check that values are equal
        assert copied_params["a"] == sample_params["a"]
        assert copied_params["b"] == sample_params["b"]

        # Check that modifying copy doesn't affect original
        copied_params["a"] = 999
        assert sample_params["a"] == 1
        assert copied_params["a"] == 999

    def test_add_to_time_vary(self, sample_params):
        """Test adding parameters to time-varying set."""
        # Initially 'a' is time-invariant
        assert not sample_params.is_time_varying("a")

        # Add to time-varying
        sample_params.add_to_time_vary("a")
        assert sample_params.is_time_varying("a")
        assert "a" not in sample_params._invariant_params

    def test_add_to_time_vary_nonexistent(self, sample_params):
        """Test adding non-existent parameter to time-varying set issues warning."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sample_params.add_to_time_vary("nonexistent")
            assert len(w) == 1
            assert "does not exist" in str(w[0].message)

    def test_add_to_time_inv(self, sample_params):
        """Test adding parameters to time-invariant set."""
        # Initially 'b' is time-varying
        assert sample_params.is_time_varying("b")

        # Add to time-invariant
        sample_params.add_to_time_inv("b")
        assert not sample_params.is_time_varying("b")
        assert "b" not in sample_params._varying_params

    def test_add_to_time_inv_nonexistent(self, sample_params):
        """Test adding non-existent parameter to time-invariant set issues warning."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sample_params.add_to_time_inv("nonexistent")
            assert len(w) == 1
            assert "does not exist" in str(w[0].message)

    def test_del_from_time_vary(self, sample_params):
        """Test removing parameters from time-varying set."""
        assert "b" in sample_params._varying_params
        sample_params.del_from_time_vary("b")
        assert "b" not in sample_params._varying_params

    def test_del_from_time_inv(self, sample_params):
        """Test removing parameters from time-invariant set."""
        assert "a" in sample_params._invariant_params
        sample_params.del_from_time_inv("a")
        assert "a" not in sample_params._invariant_params

    def test_to_namedtuple(self, sample_params):
        """Test conversion to namedtuple."""
        nt = sample_params.to_namedtuple()
        assert nt.a == 1
        assert nt.b == [2, 3, 4]
        assert nt.c == 5.0
        assert nt.d == [6.0, 7.0, 8.0]
        assert nt.T_cycle == 3

    def test_getitem_out_of_bounds(self, sample_params):
        """Test accessing age index out of bounds raises ValueError."""
        with pytest.raises(ValueError, match="out of bounds"):
            _ = sample_params[3]  # Max index is 2 (T_cycle=3)

        with pytest.raises(ValueError, match="out of bounds"):
            _ = sample_params[10]

    def test_getitem_nonexistent_key(self, sample_params):
        """Test accessing non-existent parameter raises KeyError."""
        with pytest.raises(KeyError):
            _ = sample_params["nonexistent"]

    def test_getitem_wrong_type(self, sample_params):
        """Test accessing with wrong type raises TypeError."""
        with pytest.raises(TypeError, match="must be an integer"):
            _ = sample_params[3.14]

        with pytest.raises(TypeError, match="must be an integer"):
            _ = sample_params[[1, 2]]

    def test_update_from_dict(self, sample_params):
        """Test updating from a dictionary."""
        update_dict = {"a": 999, "new_param": 123}
        sample_params.update(update_dict)
        assert sample_params["a"] == 999
        assert sample_params["new_param"] == 123

    def test_update_invalid_type(self, sample_params):
        """Test updating with invalid type raises TypeError."""
        with pytest.raises(
            TypeError, match="must be a Parameters object or a dictionary"
        ):
            sample_params.update([1, 2, 3])

        with pytest.raises(
            TypeError, match="must be a Parameters object or a dictionary"
        ):
            sample_params.update("invalid")

    def test_setitem_numpy_array(self):
        """Test setting parameters with numpy arrays."""
        params = Parameters(T_cycle=3)
        arr = np.array([1.0, 2.0, 3.0])
        params["arr_param"] = arr
        assert np.array_equal(params["arr_param"], arr)
        assert "arr_param" in params._invariant_params

    def test_setitem_boolean(self):
        """Test setting parameters with boolean values."""
        params = Parameters(T_cycle=3)
        params["bool_param"] = True
        assert params["bool_param"] is True
        assert "bool_param" in params._invariant_params

    def test_setitem_callable(self):
        """Test setting parameters with callable values."""
        params = Parameters(T_cycle=3)

        def test_func():
            return 42

        params["func_param"] = test_func
        assert params["func_param"] == test_func
        assert "func_param" in params._invariant_params

    def test_setitem_none(self):
        """Test setting parameters with None value."""
        params = Parameters(T_cycle=3)
        params["none_param"] = None
        assert params["none_param"] is None
        assert "none_param" in params._invariant_params

    def test_setitem_distribution(self):
        """Test setting parameters with Distribution objects."""
        params = Parameters(T_cycle=3)
        dist = Uniform(0, 1)
        params["dist_param"] = dist
        assert params["dist_param"] == dist
        assert "dist_param" in params._invariant_params

    def test_setitem_tuple(self):
        """Test setting parameters with tuple values."""
        params = Parameters(T_cycle=3)
        params["tuple_param"] = (1, 2, 3)
        assert params["tuple_param"] == (1, 2, 3)
        assert "tuple_param" in params._varying_params

    def test_setitem_single_element_list(self):
        """Test setting parameters with single-element list."""
        params = Parameters(T_cycle=3)
        params["single_list"] = [42]
        assert params["single_list"] == 42  # Should be unwrapped
        assert "single_list" in params._invariant_params

    def test_initialization_no_t_cycle(self):
        """Test initialization without T_cycle defaults to 1."""
        params = Parameters(a=1, b=2)
        assert params._length == 1
        assert params["T_cycle"] == 1

    def test_initialization_t_cycle_inferred(self):
        """Test T_cycle is inferred from list length."""
        params = Parameters(a=[1, 2, 3, 4])
        assert params._length == 4
        # Note: T_cycle parameter is set during init, _length is inferred from lists
        assert (
            params["T_cycle"] == 1
        )  # Initial value since T_cycle not explicitly provided

    def test_getitem_age_with_numpy_array(self):
        """Test accessing by age when parameters include numpy arrays."""
        params = Parameters(a=np.array([1, 2, 3]), b=[4, 5, 6], c=7, T_cycle=3)
        age_params = params[1]
        # Numpy arrays are time-invariant, so the entire array is returned
        assert np.array_equal(age_params["a"], np.array([1, 2, 3]))
        assert age_params["b"] == 5
        assert age_params["c"] == 7

    def test_getitem_age_with_tuple(self):
        """Test accessing by age when parameters are tuples."""
        params = Parameters(a=(1, 2, 3), b=4, T_cycle=3)
        age_params = params[2]
        assert age_params["a"] == 3
        assert age_params["b"] == 4

    def test_setitem_unsupported_type(self):
        """Test setting parameter with unsupported type raises ValueError."""
        params = Parameters(T_cycle=3)
        with pytest.raises(ValueError, match="Unsupported type"):
            params["invalid"] = {"nested": "dict"}

    def test_empty_initialization(self):
        """Test initialization with no parameters except T_cycle."""
        params = Parameters(T_cycle=5)
        assert params._length == 5
        assert len(params) == 1  # Only T_cycle
        assert params["T_cycle"] == 5


class TestSolveFrom(unittest.TestCase):
    def shorten_params(self, params, length):
        par = deepcopy(params)
        for key in params.keys():
            if isinstance(params[key], list):
                par[key] = params[key][:length]
        par["T_cycle"] = length
        return par

    def setUp(self):
        # Create a 3-period parametrization of the IndShockConsumerType model
        self.params = init_idiosyncratic_shocks.copy()
        self.params.update(
            {
                "T_cycle": 3,
                "PermGroFac": [1.05, 1.10, 1.3],
                "LivPrb": [0.95, 0.9, 0.85],
                "TranShkStd": [0.1] * 3,
                "PermShkStd": [0.1] * 3,
                "Rfree": [1.02] * 3,
            }
        )

    def test_solve_from(self):
        # Create an IndShockConsumerType agent
        agent = IndShockConsumerType(**self.params)
        # Solve the model
        agent.solve()
        # Solution must have length 4 (includes terminal)
        assert len(agent.solution) == 4
        # Now create an agent with only the first 2 periods
        agent_2 = IndShockConsumerType(**self.shorten_params(self.params, 2))
        # Solve from the third solution of the previous agent
        agent_2.solve(from_solution=agent.solution[2])
        # The solutions (up to 2) must be the same
        for t, s2 in enumerate(agent_2.solution):
            self.assertEqual(s2.distance(agent.solution[t]), 0.0)


class ExtraConstructorTests(unittest.TestCase):
    def setUp(self):
        self.agent = IndShockConsumerType(cycles=0)

    def test_describe_constructors(self):
        self.agent.describe_constructors()

    def test_missing_input(self):
        self.agent.del_param("PermShkCount")
        self.assertRaises(Exception, self.agent.construct)

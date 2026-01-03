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
from HARK.ConsumptionSaving.TractableBufferStockModel import TractableConsumerType
from HARK.core import (
    AgentPopulation,
    AgentType,
    Parameters,
    distribute_params,
)
from HARK import (
    disable_logging,
    enable_logging,
    warnings,
    quiet,
    verbose,
)
from HARK.distributions import Uniform
from HARK.metric import MetricObject, distance_metric
from xarray import DataArray


class test_logging(unittest.TestCase):
    def test_funcs(self):
        disable_logging()
        enable_logging()
        warnings()
        quiet()
        verbose()


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
            distance_metric(np.array(self.list_a), np.array(self.list_c)),
            1.0,
        )
        # different dimensions
        self.assertEqual(
            distance_metric(np.array(self.list_a).reshape(1, 3), np.array(self.list_b)),
            10000.0,
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

    def test_incompatible(self):
        f = lambda x: x + 5.0
        self.assertAlmostEqual(distance_metric(self.obj_a, f), 1000.0)
        A = AgentType()
        self.assertAlmostEqual(distance_metric(self.obj_a, A), 1000.0)


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

    def test_del_from_X(self):
        MyType = IndShockConsumerType()
        MyType.del_from_time_inv("DiscFac")
        MyType.del_from_time_vary("Rfree")
        out = MyType.get_parameter("CRRA")
        self.assertAlmostEqual(out, MyType.CRRA)

    def test_sim_failures(self):
        MyType = IndShockConsumerType()
        MyType.solve()
        self.assertRaises(Exception, MyType.initialize_sim)
        MyType.T_sim = -10
        self.assertRaises(Exception, MyType.initialize_sim)
        self.assertRaises(Exception, MyType.simulate)

        MyType.assign_parameters(T_sim=10)
        MyType.initialize_sim()
        MyType.del_param("T_sim")
        self.assertRaises(Exception, MyType.simulate)

        MyType.assign_parameters(T_sim=10)
        MyType.initialize_sim()
        self.assertRaises(Exception, MyType.simulate, 20)


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


class test_agent_population_comprehensive(unittest.TestCase):
    """Comprehensive tests for AgentPopulation class."""

    def setUp(self):
        """Set up test fixtures with various parameter configurations."""
        # Basic parameters with distributions
        self.params_with_dist = init_idiosyncratic_shocks.copy()
        self.params_with_dist["CRRA"] = Uniform(2.0, 10)
        self.params_with_dist["DiscFac"] = Uniform(0.9, 0.99)

        # Parameters with DataArray (ex-ante heterogeneous agents)
        # Use xarray.DataArray with 'agent' dimension for distributed parameters
        from xarray import DataArray

        self.params_with_lists = init_idiosyncratic_shocks.copy()
        self.params_with_lists["CRRA"] = DataArray([2.0, 4.0, 6.0], dims=("agent",))
        self.params_with_lists["DiscFac"] = DataArray(
            [0.95, 0.96, 0.97], dims=("agent",)
        )

        # Parameters with time-varying lists
        self.params_time_varying = init_idiosyncratic_shocks.copy()
        self.params_time_varying["LivPrb"] = [[0.95, 0.96, 0.97], [0.96, 0.97, 0.98]]

        # Homogeneous parameters (scalars only)
        self.params_homogeneous = init_idiosyncratic_shocks.copy()

    def test_initialization_with_distributions(self):
        """Test AgentPopulation initialization with Distribution parameters."""
        agent_pop = AgentPopulation(IndShockConsumerType, self.params_with_dist)
        self.assertEqual(agent_pop.agent_type, IndShockConsumerType)
        self.assertEqual(agent_pop.seed, 0)
        self.assertIsNotNone(agent_pop.time_var)
        self.assertIsNotNone(agent_pop.time_inv)
        self.assertEqual(len(agent_pop.distributed_params), 2)

    def test_initialization_with_lists(self):
        """Test AgentPopulation initialization with list parameters."""
        agent_pop = AgentPopulation(IndShockConsumerType, self.params_with_lists)
        self.assertEqual(agent_pop.agent_type_count, 3)
        self.assertIn("CRRA", agent_pop.distributed_params)
        self.assertIn("DiscFac", agent_pop.distributed_params)

    def test_initialization_homogeneous(self):
        """Test AgentPopulation with homogeneous (scalar) parameters."""
        agent_pop = AgentPopulation(IndShockConsumerType, self.params_homogeneous)
        self.assertEqual(agent_pop.agent_type_count, 1)
        self.assertEqual(len(agent_pop.distributed_params), 0)

    def test_infer_counts_with_lists(self):
        """Test inference of agent_type_count from list parameters."""
        agent_pop = AgentPopulation(IndShockConsumerType, self.params_with_lists)
        self.assertEqual(agent_pop.agent_type_count, 3)

    def test_infer_counts_time_varying(self):
        """Test inference of term_age from time-varying parameters."""
        agent_pop = AgentPopulation(IndShockConsumerType, self.params_time_varying)
        self.assertEqual(agent_pop.agent_type_count, 2)
        self.assertEqual(agent_pop.term_age, 3)

    def test_approx_distributions_creates_discrete(self):
        """Test that approx_distributions creates discrete distributions."""
        agent_pop = AgentPopulation(IndShockConsumerType, self.params_with_dist)
        agent_pop.approx_distributions(
            {
                "CRRA": {"N": 3, "method": "equiprobable"},
                "DiscFac": {"N": 4, "method": "equiprobable"},
            }
        )
        self.assertEqual(len(agent_pop.continuous_distributions), 2)
        self.assertEqual(len(agent_pop.discrete_distributions), 2)
        self.assertEqual(agent_pop.agent_type_count, 12)

    def test_approx_distributions_updates_parameters(self):
        """Test that approx_distributions updates the parameters dict."""
        agent_pop = AgentPopulation(IndShockConsumerType, self.params_with_dist)
        agent_pop.approx_distributions(
            {
                "CRRA": {"N": 3, "method": "equiprobable"},
                "DiscFac": {"N": 4, "method": "equiprobable"},
            }
        )
        from xarray import DataArray

        self.assertIsInstance(agent_pop.parameters["CRRA"], DataArray)
        self.assertIsInstance(agent_pop.parameters["DiscFac"], DataArray)

    def test_parse_parameters(self):
        """Test parameter parsing for distributed agents."""
        agent_pop = AgentPopulation(IndShockConsumerType, self.params_with_lists)
        agent_pop.create_distributed_agents()
        # After creating agents, check that they received correct CRRA values
        self.assertEqual(len(agent_pop.agents), 3)
        # Check that CRRA values are correctly assigned to each agent
        self.assertEqual(agent_pop.agents[0].CRRA, 2.0)
        self.assertEqual(agent_pop.agents[1].CRRA, 4.0)
        self.assertEqual(agent_pop.agents[2].CRRA, 6.0)

    def test_create_distributed_agents_count(self):
        """Test that create_distributed_agents creates correct number of agents."""
        agent_pop = AgentPopulation(IndShockConsumerType, self.params_with_lists)
        agent_pop.create_distributed_agents()
        self.assertEqual(len(agent_pop.agents), 3)

    def test_create_distributed_agents_types(self):
        """Test that created agents are of correct type."""
        agent_pop = AgentPopulation(IndShockConsumerType, self.params_with_lists)
        agent_pop.create_distributed_agents()
        for agent in agent_pop.agents:
            self.assertIsInstance(agent, IndShockConsumerType)

    def test_create_distributed_agents_parameters(self):
        """Test that agents receive correct parameters."""
        agent_pop = AgentPopulation(IndShockConsumerType, self.params_with_lists)
        agent_pop.create_distributed_agents()
        self.assertEqual(agent_pop.agents[0].CRRA, 2.0)
        self.assertEqual(agent_pop.agents[1].CRRA, 4.0)
        self.assertEqual(agent_pop.agents[2].CRRA, 6.0)

    def test_create_distributed_agents_seeds(self):
        """Test that agents receive different random seeds."""
        agent_pop = AgentPopulation(
            IndShockConsumerType, self.params_with_lists, seed=42
        )
        agent_pop.create_distributed_agents()
        seeds = [agent.seed for agent in agent_pop.agents]
        # All seeds should be different
        self.assertEqual(len(seeds), len(set(seeds)))

    def test_create_database(self):
        """Test creation of agent database."""
        agent_pop = AgentPopulation(IndShockConsumerType, self.params_with_lists)
        agent_pop.create_distributed_agents()
        agent_pop.create_database()
        import pandas as pd

        self.assertIsInstance(agent_pop.agent_database, pd.DataFrame)
        self.assertEqual(len(agent_pop.agent_database), 3)
        self.assertIn("agents", agent_pop.agent_database.columns)
        self.assertIn("CRRA", agent_pop.agent_database.columns)

    def test_solve(self):
        """Test that solve method works on all agents."""
        agent_pop = AgentPopulation(IndShockConsumerType, self.params_with_lists)
        agent_pop.create_distributed_agents()
        agent_pop.solve()
        # Check that all agents have solutions
        for agent in agent_pop.agents:
            self.assertTrue(hasattr(agent, "solution"))
            self.assertIsNotNone(agent.solution)

    def test_unpack_solutions(self):
        """Test unpacking solutions from agents."""
        agent_pop = AgentPopulation(IndShockConsumerType, self.params_with_lists)
        agent_pop.create_distributed_agents()
        agent_pop.solve()
        agent_pop.unpack_solutions()
        self.assertEqual(len(agent_pop.solution), 3)
        for solution in agent_pop.solution:
            self.assertIsNotNone(solution)

    def test_initialize_sim(self):
        """Test initialization of simulation for all agents."""
        # Add simulation parameters
        params = self.params_with_lists.copy()
        params["T_sim"] = 100
        params["AgentCount"] = 10
        agent_pop = AgentPopulation(IndShockConsumerType, params)
        agent_pop.create_distributed_agents()
        agent_pop.solve()
        agent_pop.initialize_sim()
        # Check that all agents have t_sim initialized
        for agent in agent_pop.agents:
            self.assertTrue(hasattr(agent, "t_sim"))
            self.assertEqual(agent.t_sim, 0)

    def test_simulate(self):
        """Test simulation of agent population."""
        # Add simulation parameters
        params = self.params_with_lists.copy()
        params["T_sim"] = 100
        params["AgentCount"] = 10
        agent_pop = AgentPopulation(IndShockConsumerType, params)
        agent_pop.create_distributed_agents()
        agent_pop.solve()
        agent_pop.initialize_sim()
        agent_pop.simulate()
        # Check that simulation advanced time
        for agent in agent_pop.agents:
            self.assertTrue(agent.t_sim > 0)

    def test_iteration(self):
        """Test iteration over agents in population."""
        agent_pop = AgentPopulation(IndShockConsumerType, self.params_with_lists)
        agent_pop.create_distributed_agents()
        count = 0
        for agent in agent_pop:
            self.assertIsInstance(agent, IndShockConsumerType)
            count += 1
        self.assertEqual(count, 3)

    def test_indexing(self):
        """Test indexing into agent population."""
        agent_pop = AgentPopulation(IndShockConsumerType, self.params_with_lists)
        agent_pop.create_distributed_agents()
        first_agent = agent_pop[0]
        self.assertIsInstance(first_agent, IndShockConsumerType)
        self.assertEqual(first_agent.CRRA, 2.0)

    def test_negative_indexing(self):
        """Test negative indexing into agent population."""
        agent_pop = AgentPopulation(IndShockConsumerType, self.params_with_lists)
        agent_pop.create_distributed_agents()
        last_agent = agent_pop[-1]
        self.assertIsInstance(last_agent, IndShockConsumerType)
        self.assertEqual(last_agent.CRRA, 6.0)

    def test_seed_reproducibility(self):
        """Test that same seed produces reproducible agent populations."""
        agent_pop1 = AgentPopulation(
            IndShockConsumerType, self.params_with_lists, seed=42
        )
        agent_pop1.create_distributed_agents()

        agent_pop2 = AgentPopulation(
            IndShockConsumerType, self.params_with_lists, seed=42
        )
        agent_pop2.create_distributed_agents()

        # Seeds should be the same for corresponding agents
        for i in range(3):
            self.assertEqual(agent_pop1.agents[i].seed, agent_pop2.agents[i].seed)

    def test_different_seeds(self):
        """Test that different seeds produce different agent populations."""
        agent_pop1 = AgentPopulation(
            IndShockConsumerType, self.params_with_lists, seed=42
        )
        agent_pop1.create_distributed_agents()

        agent_pop2 = AgentPopulation(
            IndShockConsumerType, self.params_with_lists, seed=24
        )
        agent_pop2.create_distributed_agents()

        # Seeds should be different for corresponding agents
        seeds_match = sum(
            agent_pop1.agents[i].seed == agent_pop2.agents[i].seed for i in range(3)
        )
        self.assertLess(seeds_match, 3)  # At most some might match by chance

    def test_time_vary_and_time_inv_preserved(self):
        """Test that time_vary and time_inv attributes are preserved."""
        agent_pop = AgentPopulation(IndShockConsumerType, self.params_with_lists)
        dummy_agent = IndShockConsumerType()
        self.assertEqual(agent_pop.time_var, dummy_agent.time_vary)
        self.assertEqual(agent_pop.time_inv, dummy_agent.time_inv)


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

    def test_t_cycle_validation(self):
        """Test that T_cycle must be >= 1."""
        with pytest.raises(ValueError, match="T_cycle must be >= 1"):
            Parameters(T_cycle=0)

        with pytest.raises(ValueError, match="T_cycle must be >= 1"):
            Parameters(T_cycle=-1)

        # T_cycle=1 should work
        params = Parameters(T_cycle=1)
        assert params._length == 1

    def test_explicit_time_inv_override(self):
        """Test explicitly marking parameters as time-invariant."""
        # Normally, a list would be time-varying
        params = Parameters(
            T_cycle=3, asset_list=["stocks", "bonds", "cash"], _time_inv=["asset_list"]
        )
        assert "asset_list" in params._invariant_params
        assert "asset_list" not in params._varying_params
        assert params["asset_list"] == ["stocks", "bonds", "cash"]

    def test_explicit_time_vary_override(self):
        """Test explicitly marking parameters as time-varying."""
        # Normally, a numpy array would be time-invariant
        params = Parameters(
            T_cycle=3, values=np.array([1, 2, 3]), _time_vary=["values"]
        )
        assert "values" in params._varying_params
        assert "values" not in params._invariant_params

    def test_frozen_mode(self):
        """Test that frozen Parameters cannot be modified."""
        params = Parameters(T_cycle=3, a=1, b=[2, 3, 4], frozen=True)

        # Attempt to modify should raise RuntimeError
        with pytest.raises(RuntimeError, match="Cannot modify frozen"):
            params["c"] = 5

        with pytest.raises(RuntimeError, match="Cannot modify frozen"):
            params["a"] = 10

        with pytest.raises(RuntimeError, match="Cannot modify frozen"):
            params.new_attr = 20

    def test_frozen_allows_read(self):
        """Test that frozen Parameters can still be read."""
        params = Parameters(T_cycle=3, a=1, b=[2, 3, 4], frozen=True)
        assert params["a"] == 1
        assert params.a == 1
        assert params["b"] == [2, 3, 4]

    def test_at_age_method(self):
        """Test the at_age() method."""
        params = Parameters(T_cycle=3, beta=[0.95, 0.96, 0.97], sigma=2.0)

        # Get parameters for age 0
        age_0 = params.at_age(0)
        assert age_0.beta == 0.95
        assert age_0.sigma == 2.0

        # Get parameters for age 1
        age_1 = params.at_age(1)
        assert age_1.beta == 0.96
        assert age_1.sigma == 2.0

        # Get parameters for age 2
        age_2 = params.at_age(2)
        assert age_2.beta == 0.97
        assert age_2.sigma == 2.0

    def test_at_age_out_of_bounds(self):
        """Test that at_age() raises ValueError for invalid age."""
        params = Parameters(T_cycle=3, beta=[0.95, 0.96, 0.97])

        with pytest.raises(ValueError, match="out of bounds"):
            params.at_age(3)

        with pytest.raises(ValueError, match="out of bounds"):
            params.at_age(-1)

    def test_validate_success(self):
        """Test validate() passes for valid parameters."""
        params = Parameters(T_cycle=3, beta=[0.95, 0.96, 0.97], sigma=2.0)
        params.validate()  # Should not raise

    def test_validate_failure_wrong_length(self):
        """Test validate() fails for mismatched lengths."""
        params = Parameters(T_cycle=3, beta=[0.95, 0.96, 0.97])
        # Manually add a time-varying parameter with wrong length
        params._parameters["gamma"] = [1.0, 2.0]  # Wrong length
        params._varying_params.add("gamma")

        with pytest.raises(ValueError, match="validation failed"):
            params.validate()

    def test_2d_array_time_varying(self):
        """Test that 2D numpy arrays with first dim = T_cycle are time-varying."""
        # 2D array with shape (3, 2) where first dim matches T_cycle
        arr_2d = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        params = Parameters(T_cycle=3, matrix=arr_2d)

        assert "matrix" in params._varying_params
        assert "matrix" not in params._invariant_params
        assert np.array_equal(params["matrix"], arr_2d)

    def test_2d_array_time_invariant(self):
        """Test that 2D numpy arrays with first dim != T_cycle are time-invariant."""
        # 2D array with shape (2, 3) where first dim doesn't match T_cycle
        arr_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        params = Parameters(T_cycle=3, matrix=arr_2d)

        assert "matrix" in params._invariant_params
        assert "matrix" not in params._varying_params

    def test_namedtuple_caching(self):
        """Test that to_namedtuple() caches the namedtuple class."""
        params = Parameters(T_cycle=3, a=1, b=2)

        # First call should create cache
        nt1 = params.to_namedtuple()
        assert params._namedtuple_cache is not None

        # Second call should use cached class
        nt2 = params.to_namedtuple()
        assert type(nt1) is type(nt2)  # Same class

    def test_combined_explicit_overrides(self):
        """Test using both _time_inv and _time_vary together."""
        params = Parameters(
            T_cycle=3,
            list_param=[1, 2, 3],  # Would be time-varying
            array_param=np.array([4, 5, 6]),  # Would be time-invariant
            _time_inv=["list_param"],
            _time_vary=["array_param"],
        )

        assert "list_param" in params._invariant_params
        assert "array_param" in params._varying_params

    def test_2d_array_at_t_cycle_1(self):
        """Test that 2D array with shape (1, n) at T_cycle=1 is time-varying."""
        # Edge case: T_cycle=1 with 2D array shape (1, 5)
        arr_2d = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        params = Parameters(T_cycle=1, matrix=arr_2d)

        # Should be time-varying since first dim matches T_cycle
        assert "matrix" in params._varying_params
        assert "matrix" not in params._invariant_params

    def test_validate_0d_array(self):
        """Test that validate() catches 0-dimensional arrays marked as time-varying."""
        params = Parameters(T_cycle=3, beta=0.95)

        # Manually add a 0-dimensional array as time-varying (invalid state)
        params._parameters["scalar_array"] = np.array(5.0)  # 0-dimensional
        params._varying_params.add("scalar_array")

        # Should fail validation
        with pytest.raises(ValueError, match="0-dimensional array"):
            params.validate()


class TestSolveWithParameters(unittest.TestCase):
    """Test solving an agent with Parameters object for params."""

    def test_solve_agent_with_parameters(self):
        """Test that an agent can be solved when its params are a Parameters object."""
        BaseType = TractableConsumerType()
        BaseType.solve()
        BaseType.unpack("cFunc")

        AltType = TractableConsumerType()
        temp = {
            key: AltType.parameters[key]
            if type(AltType.parameters[key]) is not dict
            else None
            for key in AltType.parameters
        }
        new_params = Parameters(**temp)
        AltType.parameters = new_params
        AltType.solve()
        AltType.unpack("cFunc")

        mNrm = 5.0
        cNrm_targ = BaseType.cFunc[0](mNrm)
        self.assertAlmostEqual(cNrm_targ, AltType.cFunc[0](mNrm))


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


# Define a constructor that will definitely throw an error
def broken_constructor(IncShkDstn):
    X = np.linspace(0.0, 1.0, 21)
    raise ValueError("This is an intentional error from a broken constructor")


class ExtraConstructorTests(unittest.TestCase):
    def setUp(self):
        self.agent = IndShockConsumerType(cycles=0)

    def test_describe_constructors(self):
        self.agent.describe_constructors()
        self.agent.describe_constructors("blorppity")

    def test_missing_input(self):
        self.agent.del_param("PermShkCount")
        self.agent.describe_constructors("IncShkDstn")
        self.assertRaises(Exception, self.agent.construct)

    def test_missing_constructor(self):
        self.assertRaises(KeyError, self.agent.construct, "blorppity")

    def test_broken_constructor(self):
        self.agent.constructors["TranShkDstn"] = broken_constructor
        self.assertRaises(ValueError, self.agent.construct, "TranShkDstn")


class TestAgentPopulationParseParameters(unittest.TestCase):
    """Tests for AgentPopulation.__parse_parameters__ covering all code paths."""

    def test_time_var_scalar_parameter(self):
        """Test time-varying parameter with scalar value gets repeated."""
        params = init_idiosyncratic_shocks.copy()
        # CRRA is in time_inv, use a param that's in time_vary
        # Rfree is time-varying in IndShockConsumerType
        params["CRRA"] = DataArray([2.0, 3.0], dims=("agent",))
        params["Rfree"] = 1.03  # scalar for time-varying param

        agent_pop = AgentPopulation(IndShockConsumerType, params)
        agent_pop.create_distributed_agents()

        # Rfree should be repeated for each time period
        for agent in agent_pop.agents:
            self.assertIsNotNone(agent.Rfree)

    def test_time_var_list_of_lists_parameter(self):
        """Test time-varying parameter with list of lists (agent x time)."""
        params = init_idiosyncratic_shocks.copy()
        params["CRRA"] = DataArray([2.0, 3.0], dims=("agent",))
        # Rfree varies by agent and time: 2 agents, each with their own time series
        params["Rfree"] = [[1.02], [1.03]]

        agent_pop = AgentPopulation(IndShockConsumerType, params)
        agent_pop.create_distributed_agents()

        self.assertEqual(len(agent_pop.agents), 2)
        self.assertEqual(agent_pop.agents[0].Rfree, [1.02])
        self.assertEqual(agent_pop.agents[1].Rfree, [1.03])

    def test_time_var_simple_list_parameter(self):
        """Test time-varying parameter with simple list (same for all agents)."""
        params = init_idiosyncratic_shocks.copy()
        params["CRRA"] = DataArray([2.0, 3.0], dims=("agent",))
        params["Rfree"] = [1.03]  # simple list, same for all agents

        agent_pop = AgentPopulation(IndShockConsumerType, params)
        agent_pop.create_distributed_agents()

        self.assertEqual(len(agent_pop.agents), 2)
        for agent in agent_pop.agents:
            self.assertEqual(agent.Rfree, [1.03])

    def test_time_var_dataarray_agent_age_dims(self):
        """Test time-varying parameter with DataArray having (agent, age) dims."""
        params = init_idiosyncratic_shocks.copy()
        params["CRRA"] = DataArray([2.0, 3.0], dims=("agent",))
        # Rfree with both agent and age dimensions (2 agents x 3 ages)
        params["Rfree"] = DataArray(
            [[1.01, 1.02, 1.03], [1.04, 1.05, 1.06]], dims=("agent", "age")
        )

        agent_pop = AgentPopulation(IndShockConsumerType, params)
        agent_pop.create_distributed_agents()

        self.assertEqual(len(agent_pop.agents), 2)
        self.assertEqual(agent_pop.agents[0].CRRA, 2.0)
        self.assertEqual(agent_pop.agents[1].CRRA, 3.0)
        # Each agent gets their own list of values across ages
        self.assertEqual(agent_pop.agents[0].Rfree, [1.01, 1.02, 1.03])
        self.assertEqual(agent_pop.agents[1].Rfree, [1.04, 1.05, 1.06])

    def test_time_var_dataarray_age_only_dim(self):
        """Test time-varying parameter with DataArray having only age dim."""
        params = init_idiosyncratic_shocks.copy()
        params["CRRA"] = DataArray([2.0, 3.0], dims=("agent",))
        # Rfree with only age dimension (same across agents)
        params["Rfree"] = DataArray([1.03], dims=("age",))

        agent_pop = AgentPopulation(IndShockConsumerType, params)
        agent_pop.create_distributed_agents()

        self.assertEqual(len(agent_pop.agents), 2)
        self.assertEqual(agent_pop.agents[0].CRRA, 2.0)
        self.assertEqual(agent_pop.agents[1].CRRA, 3.0)
        # Time-varying params return lists (one value per age)
        self.assertEqual(agent_pop.agents[0].Rfree, [1.03])
        self.assertEqual(agent_pop.agents[1].Rfree, [1.03])

    def test_time_inv_scalar_parameter(self):
        """Test time-invariant parameter with scalar value."""
        params = init_idiosyncratic_shocks.copy()
        params["CRRA"] = DataArray([2.0, 3.0], dims=("agent",))
        params["DiscFac"] = 0.96  # scalar time-invariant

        agent_pop = AgentPopulation(IndShockConsumerType, params)
        agent_pop.create_distributed_agents()

        for agent in agent_pop.agents:
            self.assertEqual(agent.DiscFac, 0.96)

    def test_time_inv_list_of_lists_parameter(self):
        """Test time-invariant parameter with list of lists (agent-varying)."""
        params = init_idiosyncratic_shocks.copy()
        params["CRRA"] = DataArray([2.0, 3.0], dims=("agent",))
        # DiscFac varies by agent only (list of lists but each inner has 1 element)
        params["DiscFac"] = [[0.95], [0.97]]

        agent_pop = AgentPopulation(IndShockConsumerType, params)
        agent_pop.create_distributed_agents()

        self.assertEqual(len(agent_pop.agents), 2)
        self.assertEqual(agent_pop.agents[0].DiscFac, [0.95])
        self.assertEqual(agent_pop.agents[1].DiscFac, [0.97])

    def test_time_inv_simple_list_parameter(self):
        """Test time-invariant parameter with simple list."""
        params = init_idiosyncratic_shocks.copy()
        params["CRRA"] = DataArray([2.0, 3.0], dims=("agent",))
        params["DiscFac"] = [0.96]  # simple list

        agent_pop = AgentPopulation(IndShockConsumerType, params)
        agent_pop.create_distributed_agents()

        for agent in agent_pop.agents:
            self.assertEqual(agent.DiscFac, [0.96])

    def test_time_inv_dataarray_agent_dim(self):
        """Test time-invariant parameter with DataArray having agent dim."""
        params = init_idiosyncratic_shocks.copy()
        params["CRRA"] = DataArray([2.0, 3.0], dims=("agent",))
        params["DiscFac"] = DataArray([0.95, 0.97], dims=("agent",))

        agent_pop = AgentPopulation(IndShockConsumerType, params)
        agent_pop.create_distributed_agents()

        self.assertEqual(len(agent_pop.agents), 2)
        self.assertEqual(agent_pop.agents[0].DiscFac, 0.95)
        self.assertEqual(agent_pop.agents[1].DiscFac, 0.97)

    def test_unknown_param_scalar(self):
        """Test parameter not in time_var or time_inv with scalar value."""
        params = init_idiosyncratic_shocks.copy()
        params["CRRA"] = DataArray([2.0, 3.0], dims=("agent",))
        params["CustomParam"] = 42  # not in time_var or time_inv

        agent_pop = AgentPopulation(IndShockConsumerType, params)
        agent_pop.create_distributed_agents()

        for agent in agent_pop.agents:
            self.assertEqual(agent.CustomParam, 42)

    def test_unknown_param_list_of_lists(self):
        """Test parameter not in time_var or time_inv with list of lists."""
        params = init_idiosyncratic_shocks.copy()
        params["CRRA"] = DataArray([2.0, 3.0], dims=("agent",))
        params["CustomParam"] = [[1, 2], [3, 4]]  # agent-varying

        agent_pop = AgentPopulation(IndShockConsumerType, params)
        agent_pop.create_distributed_agents()

        self.assertEqual(agent_pop.agents[0].CustomParam, [1, 2])
        self.assertEqual(agent_pop.agents[1].CustomParam, [3, 4])

    def test_unknown_param_simple_list(self):
        """Test parameter not in time_var or time_inv with simple list."""
        params = init_idiosyncratic_shocks.copy()
        params["CRRA"] = DataArray([2.0, 3.0], dims=("agent",))
        params["CustomParam"] = [1, 2, 3]  # assumed time-varying

        agent_pop = AgentPopulation(IndShockConsumerType, params)
        agent_pop.create_distributed_agents()

        for agent in agent_pop.agents:
            self.assertEqual(agent.CustomParam, [1, 2, 3])

    def test_time_var_dataarray_agent_only_dim(self):
        """Test time-varying parameter with DataArray having only agent dim."""
        params = init_idiosyncratic_shocks.copy()
        params["CRRA"] = DataArray([2.0, 3.0], dims=("agent",))
        # Rfree with only agent dimension (no age)
        params["Rfree"] = DataArray([1.02, 1.03], dims=("agent",))

        agent_pop = AgentPopulation(IndShockConsumerType, params)
        agent_pop.create_distributed_agents()

        self.assertEqual(len(agent_pop.agents), 2)
        self.assertEqual(agent_pop.agents[0].Rfree, 1.02)
        self.assertEqual(agent_pop.agents[1].Rfree, 1.03)

    def test_unknown_param_dataarray_agent_age_dims(self):
        """Test unknown parameter with DataArray having (agent, age) dims."""
        params = init_idiosyncratic_shocks.copy()
        params["CRRA"] = DataArray([2.0, 3.0], dims=("agent",))
        params["CustomParam"] = DataArray([[1, 2], [3, 4]], dims=("agent", "age"))

        agent_pop = AgentPopulation(IndShockConsumerType, params)
        agent_pop.create_distributed_agents()

        self.assertEqual(len(agent_pop.agents), 2)
        self.assertEqual(agent_pop.agents[0].CustomParam, [1, 2])
        self.assertEqual(agent_pop.agents[1].CustomParam, [3, 4])

    def test_unknown_param_dataarray_agent_only_dim(self):
        """Test unknown parameter with DataArray having only agent dim."""
        params = init_idiosyncratic_shocks.copy()
        params["CRRA"] = DataArray([2.0, 3.0], dims=("agent",))
        params["CustomParam"] = DataArray([10, 20], dims=("agent",))

        agent_pop = AgentPopulation(IndShockConsumerType, params)
        agent_pop.create_distributed_agents()

        self.assertEqual(len(agent_pop.agents), 2)
        self.assertEqual(agent_pop.agents[0].CustomParam, 10)
        self.assertEqual(agent_pop.agents[1].CustomParam, 20)

    def test_unknown_param_dataarray_age_only_dim(self):
        """Test unknown parameter with DataArray having only age dim."""
        params = init_idiosyncratic_shocks.copy()
        params["CRRA"] = DataArray([2.0, 3.0], dims=("agent",))
        params["CustomParam"] = DataArray([1, 2, 3], dims=("age",))

        agent_pop = AgentPopulation(IndShockConsumerType, params)
        agent_pop.create_distributed_agents()

        self.assertEqual(len(agent_pop.agents), 2)
        # Both agents get the same age-varying list
        self.assertEqual(agent_pop.agents[0].CustomParam, [1, 2, 3])
        self.assertEqual(agent_pop.agents[1].CustomParam, [1, 2, 3])

    def test_approx_distributions_invalid_param(self):
        """Test approx_distributions raises error for non-distribution param."""
        params = init_idiosyncratic_shocks.copy()
        params["CRRA"] = Uniform(2.0, 10)

        agent_pop = AgentPopulation(IndShockConsumerType, params)

        with self.assertRaises(ValueError):
            agent_pop.approx_distributions(
                {
                    "DiscFac": {"N": 3, "method": "equiprobable"}
                }  # DiscFac is not a Distribution
            )

    def test_approx_distributions_single_distribution(self):
        """Test approx_distributions with a single distribution."""
        params = init_idiosyncratic_shocks.copy()
        params["CRRA"] = Uniform(2.0, 10)

        agent_pop = AgentPopulation(IndShockConsumerType, params)
        agent_pop.approx_distributions({"CRRA": {"N": 5, "method": "equiprobable"}})

        self.assertEqual(agent_pop.agent_type_count, 5)
        self.assertIn("CRRA", agent_pop.discrete_distributions)

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


class test_agent_population_comprehensive(unittest.TestCase):
    """Comprehensive tests for AgentPopulation class."""

    def setUp(self):
        """Set up test fixtures with various parameter configurations."""
        # Basic parameters with distributions
        self.params_with_dist = init_idiosyncratic_shocks.copy()
        self.params_with_dist["CRRA"] = Uniform(2.0, 10)
        self.params_with_dist["DiscFac"] = Uniform(0.9, 0.99)

        # Parameters with list of lists (ex-ante heterogeneous agents)
        self.params_with_lists = init_idiosyncratic_shocks.copy()
        self.params_with_lists["CRRA"] = [2.0, 4.0, 6.0]
        self.params_with_lists["DiscFac"] = [0.95, 0.96, 0.97]

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
        agent_pop._AgentPopulation__parse_parameters__()
        self.assertEqual(len(agent_pop.population_parameters), 3)
        self.assertIsInstance(agent_pop.population_parameters[0], dict)
        # Check that CRRA values are correctly assigned
        self.assertEqual(agent_pop.population_parameters[0]["CRRA"], 2.0)
        self.assertEqual(agent_pop.population_parameters[1]["CRRA"], 4.0)
        self.assertEqual(agent_pop.population_parameters[2]["CRRA"], 6.0)

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
        agent_pop = AgentPopulation(IndShockConsumerType, self.params_with_lists)
        agent_pop.create_distributed_agents()
        agent_pop.solve()
        agent_pop.initialize_sim()
        # Check that all agents have t_sim initialized
        for agent in agent_pop.agents:
            self.assertTrue(hasattr(agent, "t_sim"))
            self.assertEqual(agent.t_sim, 0)

    def test_simulate(self):
        """Test simulation of agent population."""
        agent_pop = AgentPopulation(IndShockConsumerType, self.params_with_lists)
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

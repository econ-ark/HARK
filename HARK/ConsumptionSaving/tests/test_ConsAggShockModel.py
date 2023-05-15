import unittest

import numpy as np

from HARK import distribute_params
from HARK.ConsumptionSaving.ConsAggShockModel import (
    AggShockConsumerType,
    AggShockMarkovConsumerType,
    CobbDouglasEconomy,
    CobbDouglasMarkovEconomy,
    KrusellSmithEconomy,
    KrusellSmithType,
)
from HARK.distribution import Uniform
from HARK.tests import HARK_PRECISION


class testAggShockConsumerType(unittest.TestCase):
    def setUp(self):
        agent = AggShockConsumerType()
        agent.AgentCount = 900  # Very low number of agents for the sake of speed
        agent.cycles = 0

        # Make agents heterogeneous in their discount factor
        self.agents = distribute_params(
            agent, "DiscFac", 3, Uniform(bot=0.90, top=0.94)  # Impatient agents
        )

        # Make an economy with those agents living in it
        self.economy = CobbDouglasEconomy(agents=self.agents)

    def test_distribute_params(self):
        self.assertEqual(self.agents[1].AgentCount, 300)

    def test_agent(self):
        # Have one consumer type inherit relevant objects from the economy,
        # then solve their microeconomic problem
        self.agents[0].get_economy_data(self.economy)
        self.agents[0].solve()
        self.assertAlmostEqual(
            self.agents[0].solution[0].cFunc(10.0, self.economy.MSS),
            3.22908,
            places=HARK_PRECISION,
        )

    def test_macro(self):
        self.economy.act_T = 400  # Short simulation history
        self.economy.max_loops = 3  # Give up quickly for the sake of time
        self.economy.make_AggShkHist()  # Simulate a history of aggregate shocks
        self.economy.verbose = False  # Turn off printed messages

        # Give data about the economy to all the agents in it
        for this_type in self.economy.agents:
            this_type.get_economy_data(self.economy)
        self.economy.solve()  # Solve for the general equilibrium of the economy

        self.economy.AFunc = self.economy.dynamics.AFunc
        self.assertAlmostEqual(self.economy.AFunc.slope, 1.11810, places=HARK_PRECISION)

        # simulation test -- seed/generator specific
        # self.assertAlmostEqual(self.economy.history["MaggNow"][10], 7.45632, place = HARK_PRECISION)


class testAggShockMarkovConsumerType(unittest.TestCase):
    def setUp(self):
        # Make one agent type and an economy for it to live in
        self.agent = AggShockMarkovConsumerType()
        self.agent.cycles = 0
        self.agent.AgentCount = 1000  # Low number of simulated agents
        self.agent.IncShkDstn = [2 * [self.agent.IncShkDstn[0]]]  ## see #557
        self.economy = CobbDouglasMarkovEconomy(agents=[self.agent])

    def test_agent(self):
        # Have one consumer type inherit relevant objects from the economy,
        # then solve their microeconomic problem
        self.agent.get_economy_data(self.economy)
        self.agent.solve()
        self.assertAlmostEqual(
            self.agent.solution[0].cFunc[0](10.0, self.economy.MSS),
            2.56359,
            places=HARK_PRECISION,
        )

    def test_economy(self):
        # Adjust the economy so that it (fake) solves quickly
        self.economy.act_T = 500  # Short simulation history
        self.economy.max_loops = 3  # Just quiet solving early
        self.economy.verbose = False  # Turn off printed messages

        self.agent.get_economy_data(self.economy)
        self.economy.make_AggShkHist()  # Make a simulated history of aggregate shocks
        self.economy.solve()  # Solve for the general equilibrium of the economy

        self.economy.AFunc = self.economy.dynamics.AFunc
        self.assertAlmostEqual(
            self.economy.AFunc[0].slope, 1.08987, places=HARK_PRECISION
        )

        # simulation test -- seed/generator specific
        # self.assertAlmostEqual(self.economy.history["AaggNow"][5], 9.46776, place = HARK_PRECISION)


class KrusellSmithTestCase(unittest.TestCase):
    def setUp(self):
        # Make one agent type and an economy for it to live in
        self.agent = KrusellSmithType()
        self.agent.cycles = 0
        self.agent.AgentCount = 1000  # Low number of simulated agents
        self.economy = KrusellSmithEconomy(agents=[self.agent])
        self.economy.max_loops = 2  # Quit early
        self.economy.act_T = 1100  # Shorter simulated period
        self.economy.discard_periods = 100
        self.economy.verbose = False  # Turn off printed messages

    def teardown(self):
        self.agent = None
        self.economy = None


class KrusellSmithAgentTestCase(KrusellSmithTestCase):
    def test_agent(self):
        self.agent.get_economy_data(self.economy)
        self.agent.solve()
        self.assertAlmostEqual(
            self.agent.solution[0].cFunc[0](10.0, self.economy.MSS),
            1.23868,
            places=HARK_PRECISION,
        )


class KrusellSmithMethodsTestCase(KrusellSmithTestCase):
    def test_methods(self):
        self.agent.get_economy_data(self.economy)

        self.assertAlmostEqual(self.agent.AFunc[0].slope, 1.0)

        self.assertAlmostEqual(self.agent.AFunc[1].slope, 1.0)

        self.agent.reset()
        self.economy.reset()

        self.agent.track_vars += ["EmpNow"]
        # self.economy.track_vars += ['EmpNow']

        self.assertEqual(
            np.sum(self.agent.state_now["EmpNow"] & self.agent.state_now["EmpNow"][-1]),
            900,
        )

        self.assertEqual(
            np.sum(
                self.agent.state_now["EmpNow"][:-1] & self.agent.state_now["EmpNow"][1:]
            ),
            810,
        )

        self.economy.make_Mrkv_history()  # Make a simulated history of aggregate shocks

        # simulation test -- seed/generator specific
        # self.assertAlmostEqual(
        #    np.mean(self.economy.MrkvNow_hist),
        #    0.48182
        # )

        # object attributes that are conditions
        # for precompute_arrays
        self.assertEqual(self.agent.aGrid.size, 32)
        self.assertAlmostEqual(self.agent.aGrid[5], 0.34260, places=HARK_PRECISION)

        self.economy.solve_agents()

        # testing precompute_arrays()
        self.assertAlmostEqual(
            self.agent.mNextArray[5, 2, 3, 0], 0.34880, places=HARK_PRECISION
        )

        # testing make_grid()
        self.assertAlmostEqual(self.agent.aGrid[1], 0.05532, places=HARK_PRECISION)

        self.assertAlmostEqual(self.economy.MSS, 13.32723, places=HARK_PRECISION)

        # testing update_solution_terminal()
        self.assertEqual(
            self.agent.solution_terminal.cFunc[0](10, self.economy.MSS), 10
        )

        self.assertAlmostEqual(
            self.economy.agents[0].solution[0].cFunc[0](10, self.economy.MSS).tolist(),
            1.2386774,
            places=4,
        )

        self.assertAlmostEqual(self.agent.AFunc[1].slope, 1.0)

        self.economy.make_history()

        emp_totals = np.sum(self.agent.history["EmpNow"], axis=0)

        # simulation test -- seed/generator specific
        # self.assertEqual(emp_totals[0], 1011)
        # self.assertEqual(emp_totals[2], 1009)
        # self.assertEqual(emp_totals[9], 1042)

        # simulation test -- seed/generator specific
        # self.assertAlmostEqual(
        #    self.economy.history['Aprev'][0],
        #    11.83133
        # )

        # simulation test -- seed/generator specific
        # self.assertAlmostEqual(
        #    self.economy.history['Aprev'][1],
        #    11.26076
        # )

        # simulation test -- seed/generator specific
        # self.assertAlmostEqual(
        #    self.economy.history['Aprev'][2],
        #    10.72309
        # )

        # simulation test -- seed/generator specific
        # self.assertAlmostEqual(
        #    self.economy.history['Mnow'][10],
        #    self.economy.history['Mnow'][10]
        # )

        new_dynamics = self.economy.update_dynamics()

        self.assertAlmostEqual(
            new_dynamics.AFunc[0].slope, 1.01779, places=HARK_PRECISION
        )

        self.assertAlmostEqual(
            new_dynamics.AFunc[1].slope, 1.02020, places=HARK_PRECISION
        )


class KrusellSmithEconomyTestCase(KrusellSmithTestCase):
    def test_economy(self):
        self.assertAlmostEqual(self.economy.AFunc[0].slope, 1.0)

        self.assertAlmostEqual(self.economy.AFunc[1].slope, 1.0)

        self.agent.get_economy_data(self.economy)
        self.economy.make_Mrkv_history()  # Make a simulated history of aggregate shocks
        self.economy.solve()  # Solve for the general equilibrium of the economy

        self.economy.AFunc = self.economy.dynamics.AFunc
        self.assertAlmostEqual(
            self.economy.AFunc[0].slope, 1.02108, places=HARK_PRECISION
        )

        # simulation test -- seed/generator specific
        # self.assertAlmostEqual(self.economy.history["Aprev"][4], 11.00911, place = HARK_PRECISION)

        # simulation test -- seed/generator specific
        # self.assertAlmostEqual(self.economy.history['Mrkv'][40], 1)

        # simulation test -- seed/generator specific
        # self.assertAlmostEqual(self.economy.history["Urate"][12], 0.04000, place = HARK_PRECISION)

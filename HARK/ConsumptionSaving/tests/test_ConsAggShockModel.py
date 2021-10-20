from HARK import distribute_params
from HARK.ConsumptionSaving.ConsAggShockModel import (
    AggShockConsumerType,
    CobbDouglasEconomy,
    AggShockMarkovConsumerType,
    CobbDouglasMarkovEconomy,
    KrusellSmithType,
    KrusellSmithEconomy,
)
from HARK.distribution import Uniform
import numpy as np
import unittest


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
            self.agents[0].solution[0].cFunc(10.0, self.economy.MSS), 3.229078148576943
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
        self.assertAlmostEqual(
            self.economy.AFunc.slope, 1.116259456228145
        )

        self.assertAlmostEqual(self.economy.history["MaggNow"][10], 7.456324335623432)


class testAggShockMarkovConsumerType(unittest.TestCase):
    def setUp(self):
        # Make one agent type and an economy for it to live in
        self.agent = AggShockMarkovConsumerType()
        self.agent.cycles = 0
        self.agent.AgentCount = 1000  # Low number of simulated agents
        self.agent.IncShkDstn[0] = 2 * [self.agent.IncShkDstn[0]]  ## see #557
        self.economy = CobbDouglasMarkovEconomy(agents=[self.agent])

    def test_agent(self):
        # Have one consumer type inherit relevant objects from the economy,
        # then solve their microeconomic problem
        self.agent.get_economy_data(self.economy)
        self.agent.solve()
        self.assertAlmostEqual(
            self.agent.solution[0].cFunc[0](10.0, self.economy.MSS), 2.5635896520991377
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
            self.economy.AFunc[0].slope, 1.0904698841958917
        )

        self.assertAlmostEqual(self.economy.history["AaggNow"][5], 9.467758924955897)


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
            self.agent.solution[0].cFunc[0](10.0, self.economy.MSS), 1.23867751
        )

class KrusellSmithMethodsTestCase(KrusellSmithTestCase):

    def test_methods(self):
        self.agent.get_economy_data(self.economy)

        self.assertAlmostEqual(
            self.agent.AFunc[0].slope,
            1.0
        )

        self.assertAlmostEqual(
            self.agent.AFunc[1].slope,
            1.0
        )

        self.agent.reset()
        self.economy.reset()

        self.agent.track_vars += ['EmpNow']
        # self.economy.track_vars += ['EmpNow']

        self.assertEqual(
            np.sum(self.agent.state_now['EmpNow'] & self.agent.state_now['EmpNow'][-1]),
            900
        )

        self.assertEqual(
            np.sum(self.agent.state_now['EmpNow'][:-1] & self.agent.state_now['EmpNow'][1:]),
            816
        )

        self.economy.make_Mrkv_history()  # Make a simulated history of aggregate shocks
        self.assertAlmostEqual(
            np.mean(self.economy.MrkvNow_hist),
            0.4818181818181818
        )

        # object attributes that are conditions
        # for precompute_arrays
        self.assertEqual(
            self.agent.aGrid.size,
            32
        )
        self.assertAlmostEqual(
            self.agent.aGrid[5],
            0.3426040963137289
        )

        self.economy.solve_agents()

        # testing precompute_arrays()
        self.assertAlmostEqual(
            self.agent.mNextArray[5,2,3,0],
            0.34879574548563563
        )

        # testing make_grid()
        self.assertAlmostEqual(
            self.agent.aGrid[1], 0.05531643953496124
        )

        self.assertEqual(
            self.economy.MSS, 13.327225348792547
        )

        # testing update_solution_terminal()
        self.assertEqual(
            self.agent.solution_terminal.cFunc[0](10,self.economy.MSS),
            10
        )

        self.assertAlmostEqual(
            self.economy.agents[0].solution[0].cFunc[0](
                10,self.economy.MSS
            ).tolist(),
            1.2386775112633517
        )

        self.assertAlmostEqual(
            self.agent.AFunc[1].slope,
            1.0
        )

        self.economy.make_history()

        emp_totals = np.sum(self.agent.history['EmpNow'], axis = 0)

        self.assertEqual(emp_totals[0], 1011)
        self.assertEqual(emp_totals[2], 1009)
        self.assertEqual(emp_totals[9], 1042)

        self.assertAlmostEqual(
            self.economy.history['Aprev'][0],
            11.83133152125537
        )

        self.assertAlmostEqual(
            self.economy.history['Aprev'][1],
            11.260758949042032
        )

        self.assertAlmostEqual(
            self.economy.history['Aprev'][2],
            10.72309150585389
        )

        self.assertAlmostEqual(
            self.economy.history['Mnow'][10],
            self.economy.history['Mnow'][10]
        )

        new_dynamics = self.economy.update_dynamics()

        self.assertAlmostEqual(
            new_dynamics.AFunc[0].slope,
            1.0030279334157743
        )

        self.assertAlmostEqual(
            new_dynamics.AFunc[1].slope,
            1.0151043538766693
        )

class KrusellSmithEconomyTestCase(KrusellSmithTestCase):

    def test_economy(self):
        self.assertAlmostEqual(
            self.economy.AFunc[0].slope,
            1.0
        )

        self.assertAlmostEqual(
            self.economy.AFunc[1].slope,
            1.0
        )

        self.agent.get_economy_data(self.economy)
        self.economy.make_Mrkv_history()  # Make a simulated history of aggregate shocks
        self.economy.solve()  # Solve for the general equilibrium of the economy

        self.economy.AFunc = self.economy.dynamics.AFunc
        self.assertAlmostEqual(self.economy.AFunc[0].slope, 1.0014463644834282)

        self.assertAlmostEqual(self.economy.history["Aprev"][4], 11.009107526443584)

        self.assertAlmostEqual(self.economy.history['Mrkv'][40], 1)

        self.assertAlmostEqual(self.economy.history["Urate"][12], 0.040000000000000036)

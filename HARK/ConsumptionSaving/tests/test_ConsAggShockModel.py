from HARK import distributeParams
from HARK.ConsumptionSaving.ConsAggShockModel import AggShockConsumerType, CobbDouglasEconomy, AggShockMarkovConsumerType, CobbDouglasMarkovEconomy
from HARK.distribution import Uniform
import numpy as np
import unittest

class testAggShockConsumerType(unittest.TestCase):

    def setUp(self):
        agent = AggShockConsumerType()
        agent.AgentCount = 900 # Very low number of agents for the sake of speed
        agent.cycles = 0
        
        # Make agents heterogeneous in their discount factor
        self.agents = distributeParams(agent,
                                       'DiscFac',
                                       3,
                                       Uniform(bot=.90, top=.94) # Impatient agents
                                       )
        
        # Make an economy with those agents living in it
        self.economy = CobbDouglasEconomy(agents=self.agents)

    def test_distributeParams(self):
        self.assertEqual(self.agents[1].AgentCount, 300)

    def test_agent(self):
        # Have one consumer type inherit relevant objects from the economy,
        # then solve their microeconomic problem
        self.agents[0].getEconomyData(self.economy)
        self.agents[0].solve()
        self.assertAlmostEqual(self.agents[0].solution[0].cFunc(10., self.economy.MSS),
                         3.229078148576943)
        
    def test_macro(self):
        self.economy.act_T = 400 # Short simulation history
        self.economy.max_loops = 3 # Give up quickly for the sake of time
        self.economy.makeAggShkHist() # Simulate a history of aggregate shocks
        self.economy.verbose = False # Turn off printed messages
        
        # Give data about the economy to all the agents in it
        for this_type in self.economy.agents:
            this_type.getEconomyData(self.economy)
        self.economy.solve() # Solve for the general equilibrium of the economy
        
        self.economy.AFunc = self.economy.dynamics.AFunc
        self.assertAlmostEqual(self.economy.AFunc.slope,
                         1.124330884813638)


class testAggShockMarkovConsumerType(unittest.TestCase):

    def setUp(self):
        # Make one agent type and an economy for it to live in
        self.agent = AggShockMarkovConsumerType()
        self.agent.cycles = 0
        self.agent.AgentCount = 1000 # Low number of simulated agents
        self.agent.IncomeDstn[0] = 2*[self.agent.IncomeDstn[0]] ## see #557
        self.economy = CobbDouglasMarkovEconomy(
            agents = [self.agent])
        
    def test_agent(self):
        # Have one consumer type inherit relevant objects from the economy,
        # then solve their microeconomic problem
        self.agent.getEconomyData(self.economy)
        self.agent.solve()
        self.assertAlmostEqual(self.agent.solution[0].cFunc[0](10., self.economy.MSS),
                         2.5635896520991377)

    def test_economy(self):
        # Adjust the economy so that it (fake) solves quickly
        self.economy.act_T = 500 # Short simulation history
        self.economy.max_loops = 3 # Just quiet solving early
        self.economy.verbose = False # Turn off printed messages
        
        self.agent.getEconomyData(self.economy)
        self.economy.makeAggShkHist() # Make a simulated history of aggregate shocks
        self.economy.solve() # Solve for the general equilibrium of the economy
        
        self.economy.AFunc = self.economy.dynamics.AFunc
        self.assertAlmostEqual(self.economy.AFunc[0].slope,
                         1.0921217053006234)
                         

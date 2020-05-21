from HARK import distributeParams
from HARK.ConsumptionSaving.ConsAggShockModel import AggShockConsumerType, CobbDouglasEconomy, AggShockMarkovConsumerType, CobbDouglasMarkovEconomy
from HARK.distribution import Uniform
import numpy as np
import unittest

class testAggShockConsumerType(unittest.TestCase):

    def setUp(self):
        agent = AggShockConsumerType()
        agent.AgentCount = 60
        agent.cycles = 0
        
        self.agents = distributeParams(agent,
                                       'DiscFac',
                                       3,
                                       Uniform(bot=.96,
                                              top=.98)
                                       )

        self.economy = EconomyExample = CobbDouglasEconomy(
            agents=self.agents)

    def test_distributeParams(self):
        self.assertEqual(self.agents[1].AgentCount, 20)

    def test_economy(self):
        # Make a Cobb-Douglas economy for the agents
        self.economy.makeAggShkHist()  # Simulate a history of aggregate shocks

        # Have the consumers inherit relevant objects from the economy
        self.agents[0].getEconomyData(self.economy)

        self.agents[0].solve()

class testAggShockMarkovConsumerType(unittest.TestCase):

    def setUp(self):

        self.agent = AggShockMarkovConsumerType()
        self.agent.IncomeDstn[0] = 2*[self.agent.IncomeDstn[0]] ## see #557
        self.economy = CobbDouglasMarkovEconomy(
            agents = [self.agent])

    def test_economy(self):

        self.agent.getEconomyData(self.economy) # Makes attributes of the economy, attributes of the agent
        self.economy.makeAggShkHist() # Make a simulated history of the economy

        # Set tolerance level. 
        self.economy.tolerance = 0.5

        # Solve macro problem by finding a fixed point for beliefs
        # This takes too long!
        #self.economy.solve()

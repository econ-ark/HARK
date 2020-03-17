from HARK.ConsumptionSaving.ConsAggShockModel import AggShockMarkovConsumerType, CobbDouglasMarkovEconomy
import numpy as np
import unittest


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

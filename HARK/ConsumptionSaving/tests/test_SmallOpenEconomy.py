import copy
from HARK import distributeParams
from HARK.ConsumptionSaving.ConsAggShockModel import AggShockConsumerType, SmallOpenEconomy, init_cobb_douglas
from HARK.distribution import Uniform
import numpy as np
import unittest

class testSmallOpenEconomy(unittest.TestCase):

    def test_small_open(self):
        agent = AggShockConsumerType()
        agent.AgentCount = 100 # Very low number of agents for the sake of speed
        agent.cycles = 0
        
        # Make agents heterogeneous in their discount factor
        agents = distributeParams(agent,
                                  'DiscFac',
                                  3,
                                   Uniform(bot=.90, top=.94) # Impatient agents
                                 )
        
        # Make an economy with those agents living in it
        small_economy = SmallOpenEconomy(
            agents=agents,
            Rfree = 0.2,
            wRte = 0.2,
            KtoLnow = 1,
            **copy.copy(init_cobb_douglas)
        )

        small_economy.act_T = 400 # Short simulation history
        small_economy.max_loops = 3 # Give up quickly for the sake of time
        small_economy.makeAggShkHist() # Simulate a history of aggregate shocks
        small_economy.verbose = False # Turn off printed messages
        
        # Give data about the economy to all the agents in it
        for this_type in small_economy.agents:
            this_type.getEconomyData(small_economy)

        small_economy.solve() 

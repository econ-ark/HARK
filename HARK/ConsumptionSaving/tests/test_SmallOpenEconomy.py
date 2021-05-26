import copy
from HARK import distribute_params
from HARK.ConsumptionSaving.ConsAggShockModel import (
    AggShockConsumerType,
    SmallOpenEconomy,
    init_cobb_douglas,
)
from HARK.distribution import Uniform
import numpy as np
import unittest


class testSmallOpenEconomy(unittest.TestCase):
    def test_small_open(self):
        agent = AggShockConsumerType()
        agent.AgentCount = 100  # Very low number of agents for the sake of speed
        agent.cycles = 0

        # Make agents heterogeneous in their discount factor
        agents = distribute_params(
            agent, "DiscFac", 3, Uniform(bot=0.90, top=0.94)  # Impatient agents
        )

        # Make an economy with those agents living in it
        small_economy = SmallOpenEconomy(
            agents=agents,
            Rfree=1.03,
            wRte=1.0,
            KtoLnow=1.0,
            **copy.copy(init_cobb_douglas)
        )

        small_economy.act_T = 400  # Short simulation history
        small_economy.max_loops = 3  # Give up quickly for the sake of time
        small_economy.make_AggShkHist()  # Simulate a history of aggregate shocks
        small_economy.verbose = False  # Turn off printed messages

        # Give data about the economy to all the agents in it
        for this_type in small_economy.agents:
            this_type.get_economy_data(small_economy)

        small_economy.solve()

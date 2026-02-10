import copy
import unittest


from HARK import distribute_params
from HARK.ConsumptionSaving.ConsAggShockModel import (
    AggShockConsumerType,
    SmallOpenEconomy,
    init_cobb_douglas,
)
from HARK.distributions import Uniform


class testSmallOpenEconomy(unittest.TestCase):
    def test_small_open(self):
        agent = AggShockConsumerType()
        agent.AgentCount = 100  # Very low number of agents for the sake of speed
        agent.cycles = 0

        # Make agents heterogeneous in their discount factor
        agents = distribute_params(
            agent,
            "DiscFac",
            3,
            Uniform(bot=0.90, top=0.94),  # Impatient agents
        )

        # Make an economy with those agents living in it
        small_economy = SmallOpenEconomy(
            agents=agents,
            Rfree=1.03,
            wRte=1.0,
            KtoLnow=1.0,
            act_T=400,  # Short simulation history
            max_loops=3,  # Give up quickly for the sake of time
            **copy.copy(init_cobb_douglas),
        )
        small_economy.verbose = False
        small_economy.make_AggShkHist()  # Simulate a history of aggregate shocks
        small_economy.give_agent_params()

        small_economy.solve()

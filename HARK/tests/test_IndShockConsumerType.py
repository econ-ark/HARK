from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType

import numpy as np
import unittest

class testIndShockConsumerType(unittest.TestCase):

    def test_getShocks(self):
        agent = IndShockConsumerType(
            AgentCount = 2,
            T_sim = 10
        )

        agent.solve()

        agent.initializeSim()
        agent.simBirth(np.array([True,False]))
        agent.simOnePeriod()
        agent.simBirth(np.array([False,True]))

        agent.getShocks()

        self.assertEqual(agent.PermShkNow[0],
                         1.0050166461586711)
        self.assertEqual(agent.PermShkNow[1],
                         1.0050166461586711)
        self.assertEqual(agent.TranShkNow[0],
                         1.1176912196531754)
        

    

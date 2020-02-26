from HARK.ConsumptionSaving.ConsIndShockModel import PerfForesightConsumerType

import unittest

class testPerfForesightConsumerType(unittest.TestCase):

    def setUp(self):
        self.agent = PerfForesightConsumerType()
        self.agent_infinite = PerfForesightConsumerType(cycles=0)
        
    def test_default_solution(self):
        self.agent.solve()
        c = self.agent.solution[0].cFunc

        self.assertEqual(c.x_list[0], -0.9805825242718447)
        self.assertEqual(c.x_list[1], 0.01941747572815533)
        self.assertEqual(c.y_list[0], 0)
        self.assertEqual(c.y_list[1], 0.511321002804608)
        self.assertEqual(c.decay_extrap, False)
    
    def test_checkConditions(self):
        self.agent_infinite.checkConditions()

        self.assertTrue(self.agent_infinite.AIC)
        self.assertTrue(self.agent_infinite.GICPF)
        self.assertTrue(self.agent_infinite.RIC)
        self.assertTrue(self.agent_infinite.FHWC)

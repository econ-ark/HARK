from HARK.ConsumptionSaving.ConsIndShockModel import PerfForesightConsumerType

import unittest

class testPerfForesightConsumerType(unittest.TestCase):

    def test_default_solution(self):
        agent = PerfForesightConsumerType()
        agent.solve()
        c = agent.solution[0].cFunc

        self.assertEqual(c.x_list[0], -0.9805825242718447)
        self.assertEqual(c.x_list[1], 0.01941747572815533)
        self.assertEqual(c.y_list[0], 0)
        self.assertEqual(c.y_list[1], 0.511321002804608)
        self.assertEqual(c.decay_extrap, False)
    
    

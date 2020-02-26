from HARK.ConsumptionSaving.ConsIndShockModel import PerfForesightConsumerType
import numpy as np
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

    def test_simulation(self):

        self.agent_infinite.solve()

        # Create parameter values necessary for simulation
        SimulationParams = {
            "AgentCount" : 10000,  # Number of agents of this type
            "T_sim" : 120,         # Number of periods to simulate
            "aNrmInitMean" : -6.0, # Mean of log initial assets
            "aNrmInitStd"  : 1.0,  # Standard deviation of log initial assets
            "pLvlInitMean" : 0.0,  # Mean of log initial permanent income
            "pLvlInitStd"  : 0.0,  # Standard deviation of log initial permanent income
            "PermGroFacAgg" : 1.0, # Aggregate permanent income growth factor
            "T_age" : None,        # Age after which simulated agents are automatically killed
        }

        self.agent_infinite(**SimulationParams) # This implicitly uses the assignParameters method of AgentType

        # Create PFexample object
        self.agent_infinite.track_vars = ['mNrmNow']
        self.agent_infinite.initializeSim()
        self.agent_infinite.simulate()

        self.assertAlmostEqual(
            np.mean(self.agent_infinite.mNrmNow_hist,axis=1)[40],
            -23.008063500363942
        )

        self.assertAlmostEqual(
            np.mean(self.agent_infinite.mNrmNow_hist,axis=1)[100],
            -27.164608851546927
        )

        ## Try now with the manipulation at time step 80

        self.agent_infinite.initializeSim()
        self.agent_infinite.simulate(80)
        self.agent_infinite.aNrmNow += -5. # Adjust all simulated consumers' assets downward by 5
        self.agent_infinite.simulate(40)

        self.assertAlmostEqual(
            np.mean(self.agent_infinite.mNrmNow_hist,axis=1)[40],
            -23.008063500363942
        )

        self.assertAlmostEqual(
            np.mean(self.agent_infinite.mNrmNow_hist,axis=1)[100],
            -29.140261331951606
        )

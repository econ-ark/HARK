from HARK.ConsumptionSaving.ConsIndShockModel import PerfForesightConsumerType
import numpy as np
import unittest


class testPerfForesightConsumerType(unittest.TestCase):
    def setUp(self):
        self.agent = PerfForesightConsumerType()
        self.agent_infinite = PerfForesightConsumerType(cycles=0)
        
        PF_dictionary = {
            "CRRA": 2.5,
            "DiscFac": 0.96,
            "Rfree": 1.03,
            "LivPrb": [0.98],
            "PermGroFac": [1.01],
            "T_cycle": 1,
            "cycles": 0,
            "AgentCount": 10000,
        }
        self.agent_alt = PerfForesightConsumerType(**PF_dictionary)

    def test_default_solution(self):
        self.agent.solve()
        c = self.agent.solution[0].cFunc

        self.assertEqual(c.x_list[0], -0.9805825242718447)
        self.assertEqual(c.x_list[1], 0.01941747572815533)
        self.assertEqual(c.y_list[0], 0)
        self.assertEqual(c.y_list[1], 0.511321002804608)
        self.assertEqual(c.decay_extrap, False)

    def test_another_solution(self):
        self.agent_alt.DiscFac = 0.90
        self.agent_alt.solve()
        self.assertAlmostEqual(
            self.agent_alt.solution[0].cFunc(10).tolist(), 3.9750093524820787
        )

    def test_check_conditions(self):
        self.agent_infinite.check_conditions()
        self.assertTrue(self.agent_infinite.conditions["AIC"])
        self.assertTrue(self.agent_infinite.conditions["GICRaw"])
        self.assertTrue(self.agent_infinite.conditions["RIC"])
        self.assertTrue(self.agent_infinite.conditions["FHWC"])

    def test_simulation(self):

        self.agent_infinite.solve()

        # Create parameter values necessary for simulation
        SimulationParams = {
            "AgentCount": 10000,  # Number of agents of this type
            "T_sim": 120,  # Number of periods to simulate
            "aNrmInitMean": -6.0,  # Mean of log initial assets
            "aNrmInitStd": 1.0,  # Standard deviation of log initial assets
            "pLvlInitMean": 0.0,  # Mean of log initial permanent income
            "pLvlInitStd": 0.0,  # Standard deviation of log initial permanent income
            "PermGroFacAgg": 1.0,  # Aggregate permanent income growth factor
            "T_age": None,  # Age after which simulated agents are automatically killed
        }

        self.agent_infinite.assign_parameters(
            **SimulationParams
        )  # This implicitly uses the assign_parameters method of AgentType

        # Create PFexample object
        self.agent_infinite.track_vars = ["mNrm"]
        self.agent_infinite.initialize_sim()
        self.agent_infinite.simulate()

        self.assertAlmostEqual(
            np.mean(self.agent_infinite.history["mNrm"], axis=1)[40],
            -23.008063500363942,
        )

        self.assertAlmostEqual(
            np.mean(self.agent_infinite.history["mNrm"], axis=1)[100],
            -27.164608851546927,
        )

        ## Try now with the manipulation at time step 80

        self.agent_infinite.initialize_sim()
        self.agent_infinite.simulate(80)

        # This actually does nothing because aNrmNow is
        # epiphenomenal. Probably should change mNrmNow instead
        self.agent_infinite.state_now['aNrm'] += (
            -5.0
        )
        self.agent_infinite.simulate(40)

        self.assertAlmostEqual(
            np.mean(self.agent_infinite.history["mNrm"], axis=1)[40],
            -23.008063500363942,
        )

        self.assertAlmostEqual(
            np.mean(self.agent_infinite.history["mNrm"], axis=1)[100],
            -29.140261331951606,
        )
        
    def test_stable_points(self):
        
        # Solve the constrained agent. Stable points exists only with a
        # borrowing constraint.
        constrained_agent = PerfForesightConsumerType(cycles = 0, BoroCnstArt = 0.0)
        
        constrained_agent.solve()
        
        # Check against pre-computed values.
        self.assertEqual(constrained_agent.solution[0].mBalLvl , 1.0)
        # Check that they are both the same, since the problem is deterministic
        self.assertEqual(constrained_agent.solution[0].mBalLvl,
                         constrained_agent.solution[0].mNrmTrg)

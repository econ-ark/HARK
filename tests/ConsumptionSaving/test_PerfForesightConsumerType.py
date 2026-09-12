import unittest

import numpy as np

from HARK.ConsumptionSaving.ConsIndShockModel import PerfForesightConsumerType
from tests import HARK_PRECISION


class testPerfForesightConsumerType(unittest.TestCase):
    def setUp(self):
        self.agent = PerfForesightConsumerType()
        self.agent_infinite = PerfForesightConsumerType(cycles=0)

        PF_dictionary = {
            "CRRA": 2.5,
            "DiscFac": 0.96,
            "Rfree": [1.03],
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

        self.assertAlmostEqual(c.x_list[0], -0.98058, places=HARK_PRECISION)
        self.assertAlmostEqual(c.x_list[1], 0.01942, places=HARK_PRECISION)
        self.assertEqual(c.y_list[0], 0)
        self.assertAlmostEqual(c.y_list[1], 0.51132, places=HARK_PRECISION)
        self.assertEqual(c.decay_extrap, False)

    def test_another_solution(self):
        self.agent_alt.DiscFac = 0.90
        self.agent_alt.solve()
        self.assertAlmostEqual(
            self.agent_alt.solution[0].cFunc(10).tolist(),
            3.97501,
            places=HARK_PRECISION,
        )

    def test_check_conditions(self):
        self.agent_infinite.check_conditions()
        self.assertTrue(self.agent_infinite.conditions["AIC"])
        self.assertTrue(self.agent_infinite.conditions["GICRaw"])
        self.assertTrue(self.agent_infinite.conditions["RIC"])
        self.assertTrue(self.agent_infinite.conditions["FHWC"])

    def test_failed_conditions(self):
        TestType = PerfForesightConsumerType(cycles=0, quiet=False, verbose=False)
        TestType.check_conditions()

        # make DiscFac way too big
        TestType = PerfForesightConsumerType(cycles=0, DiscFac=1.06)
        TestType.check_conditions()

        # make PermGroFac big
        TestType = PerfForesightConsumerType(cycles=0, DiscFac=0.96, PermGroFac=[1.1])
        TestType.check_conditions()

        # make Rfree too big
        TestType = PerfForesightConsumerType(cycles=0, Rfree=[1.1])
        TestType.check_conditions()

        # test constrained outcomes
        TestType = PerfForesightConsumerType(cycles=0, Rfree=[0.9], BoroCnstArt=0.0)
        TestType.check_conditions()
        TestType = PerfForesightConsumerType(
            cycles=0, PermGroFac=[0.9], BoroCnstArt=0.0
        )
        TestType.check_conditions()
        TestType = PerfForesightConsumerType(
            cycles=0, Rfree=[0.9], PermGroFac=[0.9], BoroCnstArt=0.0
        )
        TestType.check_conditions()

    def test_simulation(self):
        self.agent_infinite.solve()

        # Create parameter values necessary for simulation
        SimulationParams = {
            "AgentCount": 10000,  # Number of agents of this type
            "T_sim": 120,  # Number of periods to simulate
            "kLogInitMean": -6.0,  # Mean of log initial assets
            "kLogInitStd": 1.0,  # Standard deviation of log initial assets
            "pLogInitMean": 0.0,  # Mean of log initial permanent income
            "pLogInitStd": 0.0,  # Standard deviation of log initial permanent income
            "PermGroFacAgg": 1.0,  # Aggregate permanent income growth factor
            "T_age": None,  # Age after which simulated agents are automatically killed
        }

        self.agent_infinite.assign_parameters(
            **SimulationParams
        )  # This implicitly uses the assign_parameters method of AgentType

        # Create PFexample object
        self.agent_infinite.track_vars = ["bNrm", "mNrm", "TranShk"]
        self.agent_infinite.initialize_sim()
        self.agent_infinite.simulate()

        self.assertAlmostEqual(
            np.mean(self.agent_infinite.history["mNrm"], axis=1)[40],
            np.mean(self.agent_infinite.history["bNrm"], axis=1)[40]
            + np.mean(self.agent_infinite.history["TranShk"], axis=1)[40],
        )

        # simulation test -- seed/generator specific
        # self.assertAlmostEqual(
        #    np.mean(self.agent_infinite.history["mNrm"], axis=1)[100],
        #    -27.16461,
        # )

        ## Try now with the manipulation at time step 80

        self.agent_infinite.initialize_sim()
        self.agent_infinite.simulate(80)

        # This actually does nothing because aNrmNow is
        # epiphenomenal. Probably should change mNrmNow instead
        self.agent_infinite.state_now["aNrm"] += -5.0
        self.agent_infinite.simulate(40)

        # simulation test -- seed/generator specific
        # self.assertAlmostEqual(
        #    np.mean(self.agent_infinite.history["mNrm"], axis=1)[40],
        #    -23.00806,
        # )

        # simulation test -- seed/generator specific
        # self.assertAlmostEqual(
        #    np.mean(self.agent_infinite.history["mNrm"], axis=1)[100],
        #    -29.14026,
        # )

    def test_stable_points(self):
        # Solve the constrained agent. Stable points exists only with a
        # borrowing constraint.
        constrained_agent = PerfForesightConsumerType(cycles=0, BoroCnstArt=0.0)

        constrained_agent.solve()

        # Check against pre-computed values.
        self.assertEqual(constrained_agent.solution[0].mNrmStE, 1.0)
        # Check that they are both the same, since the problem is deterministic
        self.assertEqual(
            constrained_agent.solution[0].mNrmStE, constrained_agent.solution[0].mNrmTrg
        )

    def test_value_function_matches_closed_form(self):
        """
        Unconstrained infinite-horizon value matches its closed form, log utility included.

        Consumption is c = MPC * (m + hNrm) with MPC = 1 - (beta * R)**(1 / CRRA) / R,
        where beta includes LivPrb. Value is u(c) / MPC when CRRA != 1. With log
        utility it is log(c) / MPC + beta * log(beta * R) / MPC**2 (issue #75).
        """
        m = np.array([0.0, 1.0, 5.0, 50.0])
        for CRRA in (1.0, 2.0):
            agent = PerfForesightConsumerType(cycles=0, CRRA=CRRA, BoroCnstArt=None)
            agent.solve()
            solution = agent.solution[0]
            beta = agent.DiscFac * agent.LivPrb[0]
            R = agent.Rfree[0]
            MPC = 1.0 - (beta * R) ** (1.0 / CRRA) / R
            c = MPC * (m + solution.hNrm)
            if CRRA == 1.0:
                v = np.log(c) / MPC + beta * np.log(beta * R) / MPC**2
            else:
                v = c ** (1.0 - CRRA) / (1.0 - CRRA) / MPC
            np.testing.assert_allclose(solution.cFunc(m), c, rtol=1e-10)
            np.testing.assert_allclose(solution.vFunc(m), v, rtol=1e-10)

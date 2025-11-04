import unittest

import numpy as np

import HARK.ConsumptionSaving.ConsPortfolioModel as cpm
from tests import HARK_PRECISION


class PortfolioConsumerTypeTestCase(unittest.TestCase):
    def setUp(self):
        # Create portfolio choice consumer type
        self.pcct = cpm.PortfolioConsumerType()
        self.pcct.cycles = 0

        # Solve the model under the given parameters

        self.pcct.solve()


class StickyPortfolioConsumerType(unittest.TestCase):
    def setUp(self):
        # Create sticky portfolio choice consumer type
        self.pcct = cpm.PortfolioConsumerType(AdjProb=0.5)
        self.pcct.cycles = 0

        # Solve the model under the given parameters

    def test_solver(self):
        self.pcct.solve()


class UnitsPortfolioConsumerTypeTestCase(PortfolioConsumerTypeTestCase):
    def test_RiskyShareFunc(self):
        self.assertAlmostEqual(
            self.pcct.solution[0].ShareFuncAdj(8).tolist(),
            0.95074,
            places=HARK_PRECISION,
        )

        self.assertAlmostEqual(
            self.pcct.solution[0].ShareFuncAdj(16).tolist(),
            0.68159,
            places=HARK_PRECISION,
        )

    def test_solution(self):
        self.assertAlmostEqual(
            self.pcct.solution[0].cFuncAdj(10).tolist(), 1.69966, places=HARK_PRECISION
        )

        self.assertAlmostEqual(
            self.pcct.solution[0].ShareFuncAdj(10).tolist(),
            0.84985,
            places=HARK_PRECISION,
        )

    def test_null_solution(self):
        soln = cpm.PortfolioSolution()

    def test_sim_one_period(self):
        self.pcct.T_sim = 30
        self.pcct.AgentCount = 10
        self.pcct.track_vars += ["aNrm", "mNrm", "bNrm", "TranShk", "cNrm"]
        self.pcct.initialize_sim()

        self.assertFalse(np.any(self.pcct.shocks["Adjust"]))

        # simulation test -- seed/generator specific
        # self.assertAlmostEqual(self.pcct.state_now["pLvl"][0], 1.0)

        # simulation test -- seed/generator specific
        # self.assertAlmostEqual(self.pcct.state_now["aNrm"][0], 7.25703, place = HARK_PRECISION)

        # simulation test -- seed/generator specific
        # self.assertAlmostEqual(self.pcct.Rfree[0], 1.03)

        # simulation test -- seed/generator specific
        # self.assertAlmostEqual(self.pcct.state_now["PlvlAgg"], 1.0)

        self.pcct.sim_one_period()

        self.assertAlmostEqual(
            self.pcct.state_now["mNrm"][0],
            self.pcct.state_now["bNrm"][0] + self.pcct.shocks["TranShk"][0],
        )

        self.assertAlmostEqual(
            self.pcct.controls["cNrm"][0],
            self.pcct.solution[0].cFuncAdj(self.pcct.state_now["mNrm"][0]),
        )

        self.assertAlmostEqual(
            self.pcct.controls["Share"][0],
            self.pcct.solution[0].ShareFuncAdj(self.pcct.state_now["mNrm"][0]),
        )

        self.assertAlmostEqual(
            self.pcct.state_now["aNrm"][0],
            self.pcct.state_now["mNrm"][0] - self.pcct.controls["cNrm"][0],
        )

        # a drawn shock ; may not be robust to RNG/disitrubition implementations
        # self.assertAlmostEqual(self.pcct.shocks["Adjust"][0], 1.0)


class SimulatePortfolioConsumerTypeTestCase(PortfolioConsumerTypeTestCase):
    def test_simulation(self):
        self.pcct.T_sim = 30
        self.pcct.AgentCount = 10
        self.pcct.track_vars += [
            "mNrm",
            "cNrm",
            "Share",
            "aNrm",
            "Risky",
            "Rport",
            "Adjust",
            "PermShk",
            "bNrm",
            "TranShk",
        ]
        self.pcct.initialize_sim()

        self.pcct.simulate()

        self.assertAlmostEqual(
            self.pcct.history["mNrm"][0][0],
            self.pcct.history["bNrm"][0][0] + self.pcct.history["TranShk"][0][0],
        )

        self.assertAlmostEqual(
            self.pcct.history["cNrm"][0][0],
            self.pcct.solution[0].cFuncAdj(self.pcct.history["mNrm"][0][0]),
        )

        self.assertAlmostEqual(
            self.pcct.history["Share"][0][0],
            self.pcct.solution[0].ShareFuncAdj(self.pcct.history["mNrm"][0][0]),
        )

        self.assertAlmostEqual(
            self.pcct.history["aNrm"][0][0],
            self.pcct.history["mNrm"][0][0] - self.pcct.history["cNrm"][0][0],
        )

        self.assertAlmostEqual(self.pcct.history["Adjust"][0][0], 1.0)

        # the next period

        self.assertAlmostEqual(
            self.pcct.history["mNrm"][1][0],
            self.pcct.history["bNrm"][1][0] + self.pcct.history["TranShk"][1][0],
        )

        self.assertAlmostEqual(
            self.pcct.history["cNrm"][1][0],
            self.pcct.solution[0].cFuncAdj(self.pcct.history["mNrm"][1][0]),
        )

        self.assertAlmostEqual(
            self.pcct.history["Share"][1][0],
            self.pcct.solution[0].ShareFuncAdj(self.pcct.history["mNrm"][1][0]),
        )

        self.assertAlmostEqual(
            self.pcct.history["aNrm"][1][0],
            self.pcct.history["mNrm"][1][0] - self.pcct.history["cNrm"][1][0],
        )

        self.assertAlmostEqual(
            self.pcct.history["aNrm"][15][0],
            self.pcct.history["mNrm"][15][0] - self.pcct.history["cNrm"][15][0],
        )


class testPortfolioConsumerTypeSticky(unittest.TestCase):
    def test_sticky(self):
        # Make another example type, but this one can only update their risky portfolio
        # share in any particular period with 15% probability.
        init_sticky_share = cpm.init_portfolio.copy()
        init_sticky_share["AdjustPrb"] = 0.15

        # Create portfolio choice consumer type
        self.sticky = cpm.PortfolioConsumerType(**init_sticky_share)

        # Solve the model under the given parameters
        self.sticky.solve()


class testPortfolioConsumerTypeDiscrete(unittest.TestCase):
    def test_discrete(self):
        # Make example type of agent who can only choose risky share along discrete grid
        init_discrete_share = cpm.init_portfolio.copy()
        # PortfolioConsumerType requires vFuncBool to be True when DiscreteShareBool is True
        init_discrete_share["DiscreteShareBool"] = True
        init_discrete_share["vFuncBool"] = True

        # Create portfolio choice consumer type
        self.discrete = cpm.PortfolioConsumerType(**init_discrete_share)

        # Solve model under given parameters
        self.discrete.solve()


class testPortfolioConsumerTypeJointDist(unittest.TestCase):
    def test_joint_dist(self):
        # Create portfolio choice consumer type
        self.joint_dist = cpm.PortfolioConsumerType()
        self.joint_dist.IndepDstnBool = False

        # Solve model under given parameters
        self.joint_dist.solve()


class testPortfolioConsumerTypeDiscreteAndJoint(unittest.TestCase):
    def test_discrete_and_joint(self):
        # Make example type of agent who can only choose risky share along
        # discrete grid and income dist is correlated with risky dist
        init_discrete_and_joint_share = cpm.init_portfolio.copy()
        # PortfolioConsumerType requires vFuncBool to be True when DiscreteShareBool is True
        init_discrete_and_joint_share["DiscreteShareBool"] = True
        init_discrete_and_joint_share["vFuncBool"] = True

        # Create portfolio choice consumer type
        self.discrete_and_joint = cpm.PortfolioConsumerType(
            **init_discrete_and_joint_share
        )
        self.discrete_and_joint.IndepDstnBool = False

        # Solve model under given parameters
        self.discrete_and_joint.solve()


class testPortfolioConsumerTypeDiscreteJointSticky(unittest.TestCase):
    def test_unusual(self):
        # Make example of an agent who choosese share on grid, can only change
        # portfolio sometimes, and treats income dstn as correlated with returns
        WeirdType = cpm.PortfolioConsumerType(
            DiscreteShareBool=True,
            vFuncBool=True,
            IndepDstnBool=False,
            AdjustProb=0.3,
        )
        WeirdType.solve()


class testRiskyReturnDim(PortfolioConsumerTypeTestCase):
    def test_simulation(self):
        # Setup
        self.pcct.T_sim = 30
        self.pcct.AgentCount = 10
        self.pcct.track_vars += [
            "mNrm",
            "cNrm",
            "Risky",
        ]
        # Common (default) simulation
        self.pcct.initialize_sim()
        self.pcct.simulate()
        # Assety that all columns of Risky are the same
        self.assertTrue(
            np.all(
                self.pcct.history["Risky"]
                == self.pcct.history["Risky"][:, 0][:, np.newaxis]
            )
        )
        # Agent specific simulation
        self.pcct.sim_common_Rrisky = False
        self.pcct.initialize_sim()
        self.pcct.simulate()
        # Assety that all columns of Risky are not the same
        self.assertFalse(
            np.all(
                self.pcct.history["Risky"]
                == self.pcct.history["Risky"][:, 0][:, np.newaxis]
            )
        )


class test_time_varying_Risky_Rfree_and_Adj(unittest.TestCase):
    def setUp(self):
        # Create a parameter dictionary for a three period problem
        self.params = cpm.init_portfolio.copy()
        # Update time varying parameters
        self.params.update(
            {
                "cycles": 1,
                "T_cycle": 3,
                "T_age": 3,
                "Rfree": [1.0, 0.99, 0.98],
                "RiskyAvg": [1.01, 1.02, 1.03],
                "RiskyStd": [0.0, 0.0, 0.0],
                "RiskyCount": 1,
                "AdjustPrb": [0.0, 1.0, 0.0],
                "PermGroFac": [1.0, 1.0, 1.0],
                "LivPrb": [0.5, 0.5, 0.5],
                "PermShkStd": [0.0, 0.0, 0.0],
                "TranShkStd": [0.0, 0.0, 0.0],
                "T_sim": 50,
                "sim_common_Rrisky": False,
                "AgentCount": 10,
            }
        )

        # Create and solve agent
        self.agent = cpm.PortfolioConsumerType(**self.params)
        self.agent.solve()

    def test_draws(self):
        # Simulate the agent
        self.agent.track_vars = ["t_age", "t_cycle", "Adjust", "Risky"]
        self.agent.initialize_sim()
        self.agent.simulate()

        # Check that returns and adjustment draws are correct
        Rrisky_draws = self.agent.history["Risky"]
        Adjust_draws = self.agent.history["Adjust"]
        # t_age is increased before being recorded
        t_age = self.agent.history["t_age"] - 1

        # Check that the draws are correct
        self.assertTrue(np.all(Rrisky_draws[t_age == 1] == 1.01))
        self.assertTrue(np.all(Rrisky_draws[t_age == 2] == 1.02))
        # Adjust
        self.assertTrue(np.all(Adjust_draws[t_age == 1] == 0))
        self.assertTrue(np.all(Adjust_draws[t_age == 2] == 1))

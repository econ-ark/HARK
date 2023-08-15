import unittest

import numpy as np

import HARK.ConsumptionSaving.ConsPortfolioModel as cpm
from HARK import make_one_period_oo_solver
from HARK.tests import HARK_PRECISION
from copy import copy
from HARK.distribution import DiscreteDistributionLabeled


class PortfolioConsumerTypeTestCase(unittest.TestCase):
    def setUp(self):
        # Create portfolio choice consumer type
        self.pcct = cpm.PortfolioConsumerType()
        self.pcct.cycles = 0

        # Solve the model under the given parameters

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
        self.joint_dist.solve_one_period = make_one_period_oo_solver(
            cpm.ConsPortfolioJointDistSolver
        )

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
        self.discrete_and_joint.solve_one_period = make_one_period_oo_solver(
            cpm.ConsPortfolioJointDistSolver
        )

        # Solve model under given parameters
        self.discrete_and_joint.solve()


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


class test_time_varying_Risky_and_Adj(unittest.TestCase):
    def setUp(self):
        # Create a parameter dictionary for a three period problem
        self.params = cpm.init_portfolio.copy()
        # Update time varying parameters
        self.params.update(
            {
                "cycles": 1,
                "T_cycle": 3,
                "T_age": 3,
                "Rfree": 1.0,
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


from HARK.ConsumptionSaving.ConsIndShockModel import init_lifecycle
from HARK.ConsumptionSaving.ConsRiskyAssetModel import risky_asset_parms


class test_transition_mat(unittest.TestCase):
    def setUp(self):
        # Define some default newborn distribution over all states
        self.newborn_dstn = DiscreteDistributionLabeled(
            pmv=np.array([1.0]),
            atoms=np.array([[1.0], [1.0], [0.5], [1.0]]),
            var_names=["PLvl", "mNrm", "Share", "Adjust"],
        )

    def test_LC(self):
        # Low number of points, else RAM reqs are high
        npoints = 50

        # Create an lc agent
        lc_pars = copy(init_lifecycle)
        lc_pars.update(risky_asset_parms)
        lc_pars["DiscFac"] = 0.9
        agent = cpm.PortfolioConsumerType(**lc_pars)
        agent.solve()

        # Make shock distribution and grid
        agent.make_shock_distributions()
        agent.make_state_grid(
            PLvlGrid=None,
            mNrmGrid=np.linspace(0, 20, npoints),
            ShareGrid=None,
            AdjustGrid=None,
        )
        # Solve
        agent.solve()
        # Check that it is indeed an LC model
        assert len(agent.solution) > 10

        # Get transition matrices
        agent.find_transition_matrices(newborn_dstn=self.newborn_dstn)
        assert len(agent.solution) - 1 == len(agent.trans_mat.living_transitions)

        # Check the bruteforce representation that treats age as a state.
        full_mat = agent.trans_mat.get_full_tmat()
        # Rows of valid transition matrix sum to 1.0
        self.assertTrue(np.allclose(np.sum(full_mat, 1), 1.0))

        # Check iterating distributions forward

        # Set an initial distribution where everyone starts at the youngest age,
        # in the first gridpoint.
        dstn = np.zeros((npoints, len(agent.trans_mat.living_transitions)))
        dstn[0, 0] = 1.0
        # Find steady_state
        ss_dstn = agent.trans_mat.find_steady_state_dstn(dstn_init=dstn, max_iter=1e4)

    def test_adjust(self):
        # Create agent
        npoints = 5
        agent = cpm.PortfolioConsumerType(**cpm.init_portfolio)
        agent.make_shock_distributions()
        agent.make_state_grid(
            PLvlGrid=None,
            mNrmGrid=np.linspace(0, 10, npoints),
            ShareGrid=None,
            AdjustGrid=None,
        )
        agent.solve()
        agent.find_transition_matrices(newborn_dstn=self.newborn_dstn)
        self.assertTrue(
            agent.trans_mat.living_transitions[0].size == np.power(npoints, 2)
        )

    def test_calvo(self):
        # Create agent that has some chance of not being able to
        # adjust
        params = copy(cpm.init_portfolio)
        params["AdjustPrb"] = 0.5

        agent = cpm.PortfolioConsumerType(**params)
        agent.make_shock_distributions()
        # Share and adjust become states, so we need grids for them
        agent.make_state_grid(
            PLvlGrid=None,
            mNrmGrid=np.linspace(0, 30, 50),
            ShareGrid=np.linspace(0, 1, 10),
            AdjustGrid=np.array([False, True]),
        )
        agent.solve()
        agent.find_transition_matrices(newborn_dstn=self.newborn_dstn)
        self.assertTrue(
            agent.trans_mat.living_transitions[0].size == np.power(50 * 10 * 2, 2)
        )

        # Check that we can simulate it
        dstn = np.zeros((len(agent.state_grid.coords["mesh"]), 1))
        dstn[0, 0] = 1.0

        for _ in range(1000):
            dstn = agent.trans_mat.iterate_dstn_forward(dstn)

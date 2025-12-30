"""
This file implements unit tests for the Monte Carlo simulation module
"""

import unittest

import numpy as np

from HARK.distributions import Bernoulli, IndexDistribution, MeanOneLogNormal
from HARK.model import Aggregate, Control, DBlock
from HARK.simulation.monte_carlo import *

cons_shocks = {
    "agg_gro": Aggregate(MeanOneLogNormal(1)),
    "psi": IndexDistribution(MeanOneLogNormal, {"sigma": [1.0, 1.1]}),
    "theta": MeanOneLogNormal(1),
    "live": Bernoulli(p=1.0),
}

cons_pre = {
    "R": 1.05,
    "aNrm": 1,
    "gamma": 1.1,
    "psi": 1.1,  # TODO: draw this from a shock,
    "theta": 1.1,  # TODO: draw this from a shock
}

cons_dynamics = {
    "G": lambda gamma, psi: gamma * psi,
    "Rnrm": lambda R, G: R / G,
    "bNrm": lambda Rnrm, aNrm: Rnrm * aNrm,
    "mNrm": lambda bNrm, theta: bNrm + theta,
    "cNrm": Control(["mNrm"]),
    "aNrm": lambda mNrm, cNrm: mNrm - cNrm,
}

cons_dr = {"cNrm": lambda mNrm: mNrm / 2}


class test_draw_shocks(unittest.TestCase):
    def test_draw_shocks(self):
        drawn = draw_shocks(cons_shocks, np.array([0, 1]))

        self.assertEqual(len(drawn["theta"]), 2)
        self.assertEqual(len(drawn["psi"]), 2)
        self.assertTrue(isinstance(drawn["agg_gro"], float))


class test_simulate_dynamics(unittest.TestCase):
    def test_simulate_dynamics(self):
        post = simulate_dynamics(cons_dynamics, cons_pre, cons_dr)

        self.assertAlmostEqual(post["cNrm"], 0.98388429)


class test_AgentTypeMonteCarloSimulator(unittest.TestCase):
    def setUp(self):
        self.calibration = {  # TODO
            "G": 1.05,
        }
        self.block = DBlock(
            **{
                "shocks": {
                    "theta": MeanOneLogNormal(1, seed=303030),
                    "agg_R": Aggregate(MeanOneLogNormal(1, seed=202020)),
                    "live": Bernoulli(p=1.0, seed=101010),
                },
                "dynamics": {
                    "b": lambda agg_R, G, a: agg_R * G * a,
                    "m": lambda b, theta: b + theta,
                    "c": Control(["m"]),
                    "a": lambda m, c: m - c,
                },
            }
        )

        self.initial = {"a": MeanOneLogNormal(1), "live": 1}

        self.dr = {"c": lambda m: m / 2}

    def test_simulate(self):
        self.simulator = AgentTypeMonteCarloSimulator(
            self.calibration,
            self.block,
            self.dr,
            self.initial,
            agent_count=30,
        )

        self.simulator.initialize_sim()
        history = self.simulator.simulate()

        a1 = history["a"][5]
        b1 = (
            history["a"][4] * history["agg_R"][5] * self.calibration["G"]
            + history["theta"][5]
            - history["c"][5]
        )

        np.testing.assert_allclose(a1, b1)

    def test_make_shock_history(self):
        self.simulator = AgentTypeMonteCarloSimulator(
            self.calibration,
            self.block,
            self.dr,
            self.initial,
            agent_count=30,
        )

        self.simulator.make_shock_history()

        newborn_init_1 = self.simulator.newborn_init_history.copy()
        shocks_1 = self.simulator.shock_history.copy()

        self.simulator.initialize_sim()
        self.simulator.simulate()

        self.assertEqual(newborn_init_1, self.simulator.newborn_init_history)
        self.assertTrue(np.all(self.simulator.history["theta"] == shocks_1["theta"]))


class test_AgentTypeMonteCarloSimulatorAgeVariance(unittest.TestCase):
    def setUp(self):
        self.calibration = {  # TODO
            "G": 1.05,
        }
        self.block = DBlock(
            **{
                "shocks": {
                    "theta": MeanOneLogNormal(1),
                    "agg_R": Aggregate(MeanOneLogNormal(1)),
                    "live": Bernoulli(p=1.0),
                    "psi": IndexDistribution(MeanOneLogNormal, {"sigma": [1.0, 1.1]}),
                },
                "dynamics": {
                    "b": lambda agg_R, G, a: agg_R * G * a,
                    "m": lambda b, theta: b + theta,
                    "c": Control(["m"]),
                    "a": lambda m, c: m - c,
                },
            }
        )

        self.initial = {"a": MeanOneLogNormal(1), "live": 1}
        self.dr = {"c": [lambda m: m * 0.5, lambda m: m * 0.9]}

    def test_simulate(self):
        self.simulator = AgentTypeMonteCarloSimulator(
            self.calibration,
            self.block,
            self.dr,
            self.initial,
            agent_count=30,
        )

        self.simulator.initialize_sim()
        history = self.simulator.simulate(sim_periods=2)

        a1 = history["a"][1]
        b1 = history["m"][1] - self.dr["c"][1](history["m"][1])

        np.testing.assert_allclose(a1, b1)


class test_MonteCarloSimulator(unittest.TestCase):
    def setUp(self):
        self.calibration = {  # TODO
            "G": 1.05,
        }
        self.block = DBlock(
            **{
                "shocks": {
                    "theta": MeanOneLogNormal(1),
                    "agg_R": Aggregate(MeanOneLogNormal(1)),
                },
                "dynamics": {
                    "b": lambda agg_R, G, a: agg_R * G * a,
                    "m": lambda b, theta: b + theta,
                    "c": Control(["m"]),
                    "a": lambda m, c: m - c,
                },
            }
        )

        self.initial = {"a": MeanOneLogNormal(1)}

        self.dr = {"c": lambda m: m / 2}

    def test_simulate(self):
        self.simulator = MonteCarloSimulator(
            self.calibration,
            self.block,
            self.dr,
            self.initial,
            agent_count=30,
        )

        self.simulator.initialize_sim()
        history = self.simulator.simulate()

        a1 = history["a"][5]
        b1 = (
            history["a"][4] * history["agg_R"][5] * self.calibration["G"]
            + history["theta"][5]
            - history["c"][5]
        )
        np.testing.assert_allclose(a1, b1)

    def test_calibration_unmodified(self):
        self.simulator = MonteCarloSimulator(
            self.calibration,
            self.block,
            self.dr,
            self.initial,
            agent_count=1,
        )

        self.simulator.initialize_sim()
        self.simulator.sim_one_period()

        self.assertEqual(self.calibration, {"G": 1.05})

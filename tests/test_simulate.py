"""
This file implements unit tests for simulation methods, including a basic failure
test for simulating with no solution present and more extensive tests for the new
simulator structure. Simulation tests for various HARK models are in the model tests.
"""

# Bring in modules we need
import unittest
import numpy as np
from HARK.Calibration.Income.IncomeTools import (
    Cagetti_income,
    parse_income_spec,
    parse_time_params,
)
from HARK.Calibration.life_tables.us_ssa.SSATools import parse_ssa_life_table
from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    init_lifecycle,
)
from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType


class testsForIndShk(unittest.TestCase):
    def setUp(self):
        self.model = IndShockConsumerType(**init_lifecycle)

    def test_no_solve(self):
        model = self.model
        # Correctly assign time variables
        model.T_sim = 4
        model.t_cycle = 0
        # But forget to solve, and go straight to simulate
        with self.assertRaises(Exception):
            model.initialize_sim()
            model.simulate()


class testReproducibleSim(unittest.TestCase):
    def setUp(self):
        temp_dict = {
            "T_sim": 100,
            "AgentCount": 1000,
            "track_vars": ["cNrm"],
        }
        self.agentA = IndShockConsumerType(seed=12345, **temp_dict)
        self.agentB = IndShockConsumerType(seed=12345, **temp_dict)
        self.agentC = IndShockConsumerType(seed=54321, **temp_dict)
        self.agentA.solve()
        self.agentB.solve()
        self.agentC.solve()

    def test_sim_match(self):
        self.agentA.initialize_sim()
        self.agentA.simulate()
        self.agentB.initialize_sim()
        self.agentB.simulate()
        c_A = self.agentA.history["cNrm"][93, 138]
        c_B = self.agentB.history["cNrm"][93, 138]
        self.assertEqual(c_A, c_B)

    def test_sim_mismatch(self):
        self.agentA.initialize_sim()
        self.agentA.simulate()
        self.agentC.initialize_sim()
        self.agentC.simulate()
        c_A = self.agentA.history["cNrm"][97, 420]
        c_C = self.agentC.history["cNrm"][97, 420]
        self.assertNotAlmostEqual(c_A, c_C)


class testSimulatorClass(unittest.TestCase):
    def setUp(self):
        self.agent = IndShockConsumerType(
            cycles=0,
            T_sim=100,
            AgentCount=10000,
            tolerance=1e-12,
        )
        self.agent.NewbornTranShk = True
        self.agent.solve()
        self.agent.track_vars = ["cNrm", "aNrm", "pLvl"]

        kNrm_grid = {
            "min": 0.0,
            "max": 60.0,
            "N": 401,
        }
        cNrm_grid = {
            "min": 0.0,
            "max": 5.0,
            "N": 151,
        }
        self.grid_specs = {"kNrm": kNrm_grid, "cNrm": cNrm_grid}

    def test_sim_match(self):
        self.agent.initialize_sim()
        self.agent.simulate()
        self.agent.initialize_sym()
        self.agent.symulate()

        C0 = np.mean(self.agent.history["cNrm"], axis=1)
        C1 = np.mean(self.agent.hystory["cNrm"], axis=1)
        A0 = np.mean(self.agent.history["aNrm"], axis=1)
        A1 = np.mean(self.agent.hystory["aNrm"], axis=1)
        P0 = np.mean(self.agent.history["pLvl"], axis=1)
        P1 = np.mean(self.agent.hystory["pLvl"], axis=1)

        C_pct_diff = (C1 - C0) / C0
        A_pct_diff = (A1 - A0) / A0
        P_pct_diff = (P1 - P0) / P0

        # Check whether means are similar after 100 periods
        self.assertTrue(np.abs(C_pct_diff[-1]) < 0.01)
        self.assertTrue(np.abs(A_pct_diff[-1]) < 0.01)
        self.assertTrue(np.abs(P_pct_diff[-1]) < 0.01)

    def test_make_transition_matrices(self):
        self.agent.initialize_sym()
        self.agent._simulator.make_transition_matrices(self.grid_specs, norm="PermShk")
        trans_array = self.agent._simulator.trans_arrays[0]

        # Verify that it's a Markov matrix
        self.assertTrue(np.all(np.isclose(np.sum(trans_array, axis=1), 1.0)))

        # Find the steady state distribution and long run average consumption
        self.agent._simulator.find_steady_state()
        cNrm_avg = self.agent._simulator.get_long_run_average("cNrm")
        self.assertTrue(np.isreal(cNrm_avg))

    def test_make_basic_SSJ(self):
        self.agent.initialize_sym()  # trigger storing backup

        # Make SSJs w.r.t. interest factor
        dC_dR, dA_dR = self.agent.make_basic_SSJ(
            "Rfree",
            ["cNrm", "aNrm"],
            self.grid_specs,
            norm="PermShk",
            offset=True,
            verbose=True,
        )

        # Verify that all of the SSJs return near zero (for shocks < 100 periods ahead)
        self.assertTrue(np.all(np.isclose(dC_dR[-1, :100], 0.0)))
        self.assertTrue(np.all(np.isclose(dA_dR[-1, :100], 0.0)))

        # Calculate the shock responses "manually" and make sure they match
        resp_C, resp_A = self.agent.calc_impulse_response_manually(
            "Rfree",
            ["cNrm", "aNrm"],
            self.grid_specs,
            s=50,
            norm="PermShk",
            offset=True,
            verbose=True,
        )
        self.assertTrue(np.all(np.isclose(resp_C, dC_dR[:, 50], atol=1e-5)))
        self.assertTrue(np.all(np.isclose(resp_A, dA_dR[:, 50], atol=5e-4)))

    def test_SSJ_no_list(self):
        dC_dR = self.agent.make_basic_SSJ(
            "BoroCnstArt",
            "cNrm",
            self.grid_specs,
            norm="PermShk",
            offset=True,
            eps=-0.001,
        )

        dC_dsigma_psi = self.agent.make_basic_SSJ(
            "PermShkStd",
            "cNrm",
            self.grid_specs,
            norm="PermShk",
            offset=True,
            eps=-0.001,
            construct=["IncShkDstn", "PermShkDstn", "TranShkDstn"],
        )

    def test_describe_model(self):
        self.agent.describe_model()  # check that it doesn't crash

    def test_SSJ_errors(self):
        # Not infinite horizon
        MyType = IndShockConsumerType()
        self.assertRaises(ValueError, MyType.make_basic_SSJ, "Rfree", "cNrm", None)

        # No grid provided
        MyType = IndShockConsumerType(cycles=0)
        self.assertRaises(
            ValueError,
            MyType.make_basic_SSJ,
            "Rfree",
            "cNrm",
            {"kNrm": self.grid_specs["kNrm"]},
        )

        # Non-existent shock
        self.assertRaises(
            ValueError, MyType.make_basic_SSJ, "Rboro", "cNrm", self.grid_specs
        )

        # Invalid shock
        self.assertRaises(
            TypeError, MyType.make_basic_SSJ, "IncShkDstn", "cNrm", self.grid_specs
        )

    def test_MSR_errors(self):
        # Not infinite horizon
        MyType = IndShockConsumerType()
        self.assertRaises(
            ValueError, MyType.calc_impulse_response_manually, "Rfree", "cNrm", None
        )

        # No grid provided
        MyType = IndShockConsumerType(cycles=0)
        self.assertRaises(
            ValueError,
            MyType.calc_impulse_response_manually,
            "Rfree",
            "cNrm",
            {"kNrm": self.grid_specs["kNrm"]},
        )

        # Non-existent shock
        self.assertRaises(
            ValueError,
            MyType.calc_impulse_response_manually,
            "Rboro",
            "cNrm",
            self.grid_specs,
        )

        # Invalid shock
        self.assertRaises(
            TypeError,
            MyType.calc_impulse_response_manually,
            "IncShkDstn",
            "cNrm",
            self.grid_specs,
        )


class testGridSimulation(unittest.TestCase):
    def setUp(self):
        # Specify some choices
        birth_age = 25
        death_age = 110
        adjust_infl_to = 1992
        income_calib = Cagetti_income
        education = "College"

        # Income specification
        income_params = parse_income_spec(
            age_min=birth_age,
            age_max=death_age,
            adjust_infl_to=adjust_infl_to,
            **income_calib[education],
            SabelhausSong=True,
        )

        LivPrb = parse_ssa_life_table(
            female=True,
            cross_sec=True,
            year=2004,
            age_min=birth_age,
            age_max=death_age,
        )

        # Parameters related to the number of periods implied by the calibration
        time_params = parse_time_params(age_birth=birth_age, age_death=death_age)

        # Make a dictionary for our lifecycle agent
        lifecycle_dict = {}
        lifecycle_dict.update(income_params)
        lifecycle_dict["LivPrb"] = LivPrb
        lifecycle_dict.update(time_params)
        lifecycle_dict["Rfree"] = lifecycle_dict["T_cycle"] * [1.02]

        kNrm_grid = {
            "min": 0.0,
            "max": 200.0,
            "N": 801,
        }
        cNrm_grid = {
            "min": 0.0,
            "max": 15.0,
            "N": 251,
        }
        self.grid_specs = {"kNrm": kNrm_grid, "cNrm": cNrm_grid}

        # Make and solve a lifecycle agent type
        self.agent = IndShockConsumerType(**lifecycle_dict, T_sim=85)
        self.agent.solve()
        self.agent.track_vars = ["aNrm", "cNrm", "pLvl"]

    def test_LC_grid_simulation(self):
        self.agent.initialize_sym(stop_dead=False)

        # Make mean trajectories of normalized assets and consumption by grid method
        self.agent._simulator.make_transition_matrices(self.grid_specs)
        self.agent._simulator.simulate_cohort_by_grids(["cNrm", "aNrm"])
        grid_sim_C = self.agent._simulator.history_avg["cNrm"]
        grid_sim_A = self.agent._simulator.history_avg["aNrm"]

        # Make mean trajectories of normalized assets and consumption by Monte Carlo
        self.agent.symulate()
        MC_sim_C = np.mean(self.agent.hystory["cNrm"], axis=1)
        MC_sim_A = np.mean(self.agent.hystory["aNrm"], axis=1)

        # Verify that they're close together at all ages
        self.assertTrue(np.all(np.abs(grid_sim_A[:-1] - MC_sim_A) < 0.03))
        self.assertTrue(np.all(np.abs(grid_sim_C[:-1] - MC_sim_C) < 0.005))

    def test_LC_grid_dstn_generation(self):
        self.agent.initialize_sym(stop_dead=False)

        # Make mean trajectories of normalized assets and consumption by grid method
        self.agent._simulator.make_transition_matrices(self.grid_specs)
        self.agent._simulator.simulate_cohort_by_grids(["cNrm", "aNrm"], calc_dstn=True)


class testMarkovEvents(unittest.TestCase):
    # This runs some simple tests to cover parts of the simulator that are only
    # reached by the "Markov"-type models

    def setUp(self):
        self.agent = MarkovConsumerType(
            cycles=0,
            T_sim=100,
            PermShkCount=3,
            TranShkCount=3,
        )
        self.agent.solve()
        kNrm_grid = {
            "min": 0.0,
            "max": 100.0,
            "N": 301,
        }
        cNrm_grid = {
            "min": 0.0,
            "max": 15.0,
            "N": 201,
        }
        Mrkv_grid = {"N": 2}
        self.grid_specs = {"kNrm": kNrm_grid, "cNrm": cNrm_grid, "zPrev": Mrkv_grid}

    def test_simulation(self):
        self.agent.track_vars = ["aNrm", "cNrm", "Mrkv"]
        self.agent.initialize_sym()
        self.agent.symulate()

    def test_markov_SSJ(self):
        dC_dp1, dA_dp1 = self.agent.make_basic_SSJ(
            "Mrkv_p11",
            ["cNrm", "aNrm"],
            self.grid_specs,
            norm="PermShk",
            offset=True,
            T_max=100,
        )

    def test_common(self):
        self.agent.track_vars = ["aNrm", "cNrm", "Mrkv"]
        self.agent.initialize_sym(common="Mrkv")
        self.agent.symulate()

        Mrkv_hist = self.agent.hystory["Mrkv"]
        for t in range(self.agent.T_sim):
            self.assertTrue(np.all(Mrkv_hist[t, :] == Mrkv_hist[t, 0]))

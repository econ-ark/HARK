"""
This file implements unit tests for simulation methods, including a basic failure
test for simulating with no solution present and more extensive tests for the new
simulator structure. Simulation tests for various HARK models are in the model tests.
"""

# Bring in modules we need
import unittest
import numpy as np
from copy import deepcopy
from HARK.utilities import make_grid_exp_mult, plot_SSJ
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
from HARK.ConsumptionSaving.ConsRiskyAssetModel import RiskyAssetConsumerType
from HARK.ConsumptionSaving.ConsGenIncProcessModel import PersistentShockConsumerType
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
            "order": 2.5,
        }
        kNrm_grid_alt = {
            "min": 0.0,
            "max": 60.0,
            "N": 401,
            "nest": 2,
        }
        kNrm_grid_ult = {"custom": make_grid_exp_mult(0.0, 60.0, 401, timestonest=2)}
        cNrm_grid = {
            "min": 0.0,
            "max": 5.0,
            "N": 151,
        }
        self.grid_specs = {"kNrm": kNrm_grid, "cNrm": cNrm_grid}
        self.grid_specs_alt = {"kNrm": kNrm_grid_alt, "cNrm": cNrm_grid}
        self.grid_specs_ult = {"kNrm": kNrm_grid_ult, "cNrm": cNrm_grid}

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

        # Test the plotting tool
        plot_SSJ(dC_dR, [0, 10, 30, 50, 80], "consumption", "interest rate")

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

    def test_make_LC_SSJ(self):
        agent = IndShockConsumerType(**init_lifecycle)

        # Define grid specifications
        wealth_grid = {"min": 0.0, "max": 30.0, "N": 150, "order": 2.0}
        con_grid = {"min": 0.0, "max": 10.0, "N": 201}
        my_grid_specs = {"kNrm": wealth_grid, "cNrm": con_grid}

        # Make a life-cycle SSJ
        J_A_R, J_C_R = agent.make_basic_SSJ(
            "Rfree",
            ["aNrm", "cNrm"],
            my_grid_specs,
            offset=True,
            norm="PermShk",
            trend="PermGroFac",
            verbose=True,
        )

        self.assertTrue(np.all(np.isreal(J_A_R)))
        self.assertTrue(np.all(np.isreal(J_C_R)))

    def test_grid_formats(self):
        # Check that we get the same results for similar grids
        SSJ_base = self.agent.make_basic_SSJ(
            "Rfree",
            "cNrm",
            self.grid_specs,
            norm="PermShk",
            offset=True,
        )

        SSJ_alt = self.agent.make_basic_SSJ(
            "Rfree",
            "cNrm",
            self.grid_specs_alt,
            norm="PermShk",
            offset=True,
        )

        SSJ_ult = self.agent.make_basic_SSJ(
            "Rfree",
            "cNrm",
            self.grid_specs_ult,
            norm="PermShk",
            offset=True,
        )

        self.assertTrue(np.all(np.isclose(SSJ_base, SSJ_alt, atol=1e-4)))
        self.assertTrue(np.all(np.isclose(SSJ_base, SSJ_ult, atol=1e-4)))

    def test_SSJ_no_list(self):
        dC_daBar = self.agent.make_basic_SSJ(
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

    def test_simulate_shock_by_grids(self):
        MyType = IndShockConsumerType(cycles=0)
        MyType.solve()
        MyType.initialize_sym()
        T = 200
        my_grids = {
            "kNrm": {"min": 0.0, "max": 30.0, "N": 501, "order": 2.5},
            "cNrm": {"min": 0.0, "max": 3.0, "N": 301},
        }

        MyType._simulator.make_transition_matrices(my_grids, norm="PermShk")
        MyType._simulator.simulate_shock_by_grids(
            "aNrm", T, "aNrm * 1.1", calc_dstn=True
        )
        A_avg = MyType._simulator.history_avg["aNrm"]
        A_dstn = MyType._simulator.history_dstn["aNrm"]
        A_LR = MyType._simulator.get_long_run_average("aNrm")
        self.assertEqual(A_avg.size, T)
        self.assertEqual(A_dstn.shape[0], 501)
        self.assertEqual(A_dstn.shape[1], T)
        self.assertTrue(np.all(np.isreal(A_avg)))
        self.assertTrue(np.all(np.isreal(A_dstn)))
        self.assertTrue(np.all(np.isclose(np.sum(A_dstn, axis=0), 1.0)))
        self.assertAlmostEqual(A_avg[-1], A_LR)

    def test_simulate_shock_custom_dstn(self):
        MyType = IndShockConsumerType(cycles=0)
        MyType.solve()
        MyType.initialize_sym()
        T = 200
        my_grids = {
            "kNrm": {"min": 0.0, "max": 30.0, "N": 501, "order": 2.5},
            "cNrm": {"min": 0.0, "max": 3.0, "N": 301},
        }

        # Make a rather extreme custom distribution
        my_dstn = np.zeros(501)
        my_dstn[0] = 0.5
        my_dstn[-1] = 0.5

        MyType._simulator.make_transition_matrices(my_grids, norm="PermShk")
        MyType._simulator.find_steady_state()
        MyType._simulator.simulate_shock_by_grids("aNrm", T, from_dstn=my_dstn)
        A_avg = MyType._simulator.history_avg["aNrm"]
        A_LR = MyType._simulator.get_long_run_average("aNrm")
        self.assertEqual(A_avg.size, T)
        self.assertTrue(np.all(np.isreal(A_avg)))
        self.assertAlmostEqual(A_avg[-1], A_LR)

    def test_SSbyG_errors(self):
        MyType = IndShockConsumerType(cycles=0)
        MyType.solve()
        MyType.initialize_sym()
        T = 200
        my_grids = {
            "kNrm": {"min": 0.0, "max": 30.0, "N": 501, "order": 2.5},
            "cNrm": {"min": 0.0, "max": 3.0, "N": 301},
        }

        self.assertRaises(  # run before transition matrices exist
            KeyError,
            MyType._simulator.simulate_shock_by_grids,
            ["aNrm"],
            T,
            "aNrm * 1.1",
        )
        MyType._simulator.make_transition_matrices(my_grids, norm="PermShk")
        self.assertRaises(  # run without any shock
            ValueError, MyType._simulator.simulate_shock_by_grids, ["aNrm"], T
        )
        self.assertRaises(  # run with no output
            ValueError,
            MyType._simulator.simulate_shock_by_grids,
            ["aNrm"],
            T,
            "aNrm * 1.1",
            calc_avg=False,
        )
        self.assertRaises(  # run with invalid operator
            ValueError,
            MyType._simulator.simulate_shock_by_grids,
            ["aNrm"],
            T,
            "aNrm / 1.1",
        )
        self.assertRaises(  # try to use non-continuation state
            KeyError,
            MyType._simulator.simulate_shock_by_grids,
            ["aNrm"],
            T,
            "mNrm * 1.1",
        )
        self.assertRaises(  # try to use invalid number
            ValueError,
            MyType._simulator.simulate_shock_by_grids,
            ["aNrm"],
            T,
            "aNrm * 1.ae1",
        )
        bad_grid = np.zeros(501)
        bad_grid[0] = 0.5
        self.assertRaises(  # grid doesn't sum to 1
            ValueError,
            MyType._simulator.simulate_shock_by_grids,
            ["aNrm"],
            T,
            from_dstn=bad_grid,
        )
        bad_grid = np.zeros(502)
        bad_grid[0] = 0.5
        bad_grid[-1] = 0.5
        self.assertRaises(  # grid has wrong size
            ValueError,
            MyType._simulator.simulate_shock_by_grids,
            ["aNrm"],
            T,
            from_dstn=bad_grid,
        )


class testFindTarget(unittest.TestCase):
    def setUp(self):
        ThisType = IndShockConsumerType(cycles=0)
        ThisType.solve()
        ThisType.unpack("cFunc")
        self.agent = ThisType

    def test_match_handcrafted(self):
        ThisType = self.agent
        m_targ_old = ThisType.bilt["mNrmTrg"]
        m_targ_new = ThisType.find_target("mNrm")
        self.assertAlmostEqual(m_targ_old, m_targ_new, places=6)

    def test_risky_asset_type(self):
        RiskyType = RiskyAssetConsumerType(
            cycles=0, RiskyShareFixed=False, CRRA=5.0, Rfree=[1.02], RiskyAvg=1.04
        )
        RiskyType.solve()
        m_targ = RiskyType.find_target("mNrm")
        self.assertFalse(np.isnan(m_targ))  # did it succeed?

    def test_persistent_type(self):
        PersistentType = PersistentShockConsumerType(cycles=0)
        PersistentType.solve()
        m_targ = PersistentType.find_target("mLvl", pLvl=1.0)
        self.assertFalse(np.isnan(m_targ))  # did it succeed?

    def test_invalid(self):
        # Must be infinite horizon
        FiniteType = IndShockConsumerType(cycles=10)
        FiniteType.solve()
        self.assertRaises(ValueError, FiniteType.find_target, "mNrm")

        # Non-existent target variable
        MyType = self.agent
        self.assertRaises(ValueError, MyType.find_target, "mLvl")

        # Non-existent fixed variable
        self.assertRaises(ValueError, MyType.find_target, "mLvl", blorpity=5.0)

        # Unsolved type
        ThisType = IndShockConsumerType(cycles=0, DiscFac=1.0)
        self.assertRaises(AttributeError, ThisType.find_target, "mNrm")

        # Non-existent target
        ThisType.solve()
        m_targ = ThisType.find_target("mNrm")
        self.assertTrue(np.isnan(m_targ))  # too patient for target to exist


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


class testsForIncomeWeightedMeasure(unittest.TestCase):
    """
    The Harmenberg (income-weighted) measure under mortality with permanent income
    growth: weighting by the growth of the normalized level (norm='G') with newborns
    entering at the complement of the surviving mass. Closed-form, identity, option,
    Monte Carlo and error-path tests.
    """

    def setUp(self):
        self.agent = IndShockConsumerType(cycles=0, tolerance=1e-12)
        self.agent.solve()
        self.grid_specs = {
            "kNrm": {"min": 0.0, "max": 40.0, "N": 301, "order": 2.5},
            "cNrm": {"min": 0.0, "max": 3.0, "N": 201},
        }

    def _matrices(self, agent, **kwargs):
        agent.initialize_sym()
        X = agent._simulator
        X.make_transition_matrices(self.grid_specs, **kwargs)
        X.find_steady_state()
        return X

    def test_newborn_share_matches_closed_form(self):
        # With survival L and growth G, the newborns' share of the stationary
        # income-weighted mass is 1 - L*G, whatever the grid.
        X = self._matrices(self.agent, norm="G")
        share = np.dot(X.steady_state_dstn, X.newborn_shares[0])
        L = float(self.agent.LivPrb[0])
        G = float(self.agent.PermGroFac[0])
        self.assertAlmostEqual(share, 1.0 - L * G, places=10)
        self.assertTrue(np.all(np.isclose(np.sum(X.trans_arrays[0], axis=1), 1.0)))

    def test_long_run_dstn_is_stochastic_under_norm(self):
        X = self._matrices(self.agent, norm="G")
        self.assertAlmostEqual(np.sum(X.get_long_run_dstn("cNrm")), 1.0, places=12)

    def test_norm_G_equals_PermShk_without_growth(self):
        agent = IndShockConsumerType(cycles=0, tolerance=1e-12, PermGroFac=[1.0])
        agent.solve()
        X0 = self._matrices(deepcopy(agent), norm="PermShk")
        X1 = self._matrices(deepcopy(agent), norm="G")
        self.assertTrue(np.allclose(X0.trans_arrays[0], X1.trans_arrays[0], atol=1e-14))
        self.assertAlmostEqual(
            X0.get_long_run_average("cNrm"), X1.get_long_run_average("cNrm"), places=12
        )

    def test_trend_option_reproduces_shock_only_weighting(self):
        # When newborns inherit all of the growth, weighting by the shock alone is
        # exact and newborn_growth=PermGroFac must reproduce it.
        G = float(self.agent.PermGroFac[0])
        X0 = self._matrices(deepcopy(self.agent), norm="PermShk")
        X1 = self._matrices(deepcopy(self.agent), norm="G", newborn_growth=G)
        self.assertTrue(np.allclose(X0.trans_arrays[0], X1.trans_arrays[0], atol=1e-14))
        self.assertAlmostEqual(
            X0.get_long_run_average("aNrm"), X1.get_long_run_average("aNrm"), places=10
        )

    def test_income_weighted_mean_matches_monte_carlo(self):
        # Ground truth: simulate the population, weight each agent by its permanent
        # income. The fixed measure matches; weighting by the shock alone is biased.
        X_fix = self._matrices(deepcopy(self.agent), norm="G")
        X_old = self._matrices(deepcopy(self.agent), norm="PermShk")
        a_fix = X_fix.get_long_run_average("aNrm")
        a_old = X_old.get_long_run_average("aNrm")
        sim = deepcopy(self.agent)
        sim.AgentCount = 10000
        sim.T_sim = 300
        sim.seed = 31415
        sim.track_vars = ["pLvl", "aNrm"]
        sim.initialize_sim()
        sim.simulate()
        P = sim.history["pLvl"][-100:]
        A = sim.history["aNrm"][-100:]
        a_mc = np.sum(P * A) / np.sum(P)
        self.assertLess(abs(a_fix - a_mc) / a_mc, 0.015)
        self.assertLess(abs(a_fix - a_mc), abs(a_old - a_mc))

    def test_no_mortality_with_growth_uses_the_trend(self):
        # without deaths there are no newborns; the cross-section is stationary
        # relative to the growth trend, so norm='G' must reproduce the shock-only
        # weighting (exact in that case) rather than raise
        agent = IndShockConsumerType(
            cycles=0, tolerance=1e-12, LivPrb=[1.0], PermGroFac=[1.01]
        )
        agent.solve()
        X0 = self._matrices(deepcopy(agent), norm="PermShk")
        X1 = self._matrices(deepcopy(agent), norm="G")
        self.assertTrue(np.allclose(X0.trans_arrays[0], X1.trans_arrays[0], atol=1e-13))
        self.assertAlmostEqual(
            X0.get_long_run_average("cNrm"), X1.get_long_run_average("cNrm"), places=10
        )
        self.assertEqual(float(np.sum(X1.newborn_shares[0])), 0.0)

    def test_raises_when_growth_outpaces_mortality(self):
        agent = IndShockConsumerType(
            cycles=0, tolerance=1e-8, LivPrb=[0.995], PermGroFac=[1.01]
        )
        agent.solve()
        agent.initialize_sym()
        with self.assertRaises(ValueError):
            agent._simulator.make_transition_matrices(self.grid_specs, norm="G")

    def test_basic_SSJ_under_norm_G(self):
        agent = IndShockConsumerType(cycles=0, tolerance=1e-12, PermGroFac=[1.0])
        agent.solve()
        J0 = deepcopy(agent).make_basic_SSJ(
            "Rfree",
            "cNrm",
            self.grid_specs,
            T_max=40,
            norm="PermShk",
            offset=True,
            solved=True,
        )
        J1 = deepcopy(agent).make_basic_SSJ(
            "Rfree",
            "cNrm",
            self.grid_specs,
            T_max=40,
            norm="G",
            offset=True,
            solved=True,
        )
        self.assertTrue(np.allclose(J0, J1, atol=1e-10))
        J2 = deepcopy(self.agent).make_basic_SSJ(
            "Rfree",
            "cNrm",
            self.grid_specs,
            T_max=40,
            norm="G",
            offset=True,
            solved=True,
        )
        self.assertEqual(J2.shape, (40, 40))
        self.assertTrue(np.all(np.isfinite(J2)))

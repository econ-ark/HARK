import unittest

import numpy as np

from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    init_lifecycle,
)
from HARK.ConsumptionSaving.ConsIndShockModelFast import IndShockConsumerTypeFast
from tests.ConsumptionSaving.test_IndShockConsumerType import (
    CyclicalDict,
    IdiosyncDict,
    LifecycleDict,
)
from tests import HARK_PRECISION


class testIndShockConsumerTypeFast(unittest.TestCase):
    def setUp(self):
        self.agent = IndShockConsumerTypeFast(AgentCount=2, T_sim=10)
        self.agent.solve()

    def test_get_shocks(self):
        self.agent.initialize_sim()
        self.agent.sim_birth(np.array([True, False]))
        self.agent.sim_one_period()
        self.agent.sim_birth(np.array([False, True]))

        self.agent.get_shocks()

        # Verify shocks are generated with correct shape
        self.assertEqual(len(self.agent.shocks["PermShk"]), 2)
        self.assertEqual(len(self.agent.shocks["TranShk"]), 2)

    def test_ConsIndShockSolverBasic(self):
        LifecycleExample = IndShockConsumerTypeFast(**init_lifecycle)
        LifecycleExample.cycles = 1
        LifecycleExample.solve()

        # test the solution_terminal
        self.assertAlmostEqual(LifecycleExample.solution[-1].cFunc(2).tolist(), 2)

        self.assertAlmostEqual(
            LifecycleExample.solution[9].cFunc(1), 0.79430, places=HARK_PRECISION
        )
        self.assertAlmostEqual(
            LifecycleExample.solution[8].cFunc(1), 0.79392, places=HARK_PRECISION
        )
        self.assertAlmostEqual(
            LifecycleExample.solution[7].cFunc(1), 0.79253, places=HARK_PRECISION
        )

        self.assertAlmostEqual(
            LifecycleExample.solution[0].cFunc(1).tolist(),
            0.75062,
            places=HARK_PRECISION,
        )
        self.assertAlmostEqual(
            LifecycleExample.solution[1].cFunc(1).tolist(),
            0.75864,
            places=HARK_PRECISION,
        )
        self.assertAlmostEqual(
            LifecycleExample.solution[2].cFunc(1).tolist(),
            0.76812,
            places=HARK_PRECISION,
        )

    def test_simulated_values(self):
        self.agent.initialize_sim()
        self.agent.simulate()

        # Verify simulation produces valid results
        self.assertEqual(len(self.agent.state_now["aLvl"]), 2)
        self.assertTrue(np.all(np.isfinite(self.agent.state_now["aLvl"])))

    def test_income_dist_random_seeds(self):
        """Test that different seeds produce different income distributions."""
        a1 = IndShockConsumerTypeFast(seed=1000)
        a2 = IndShockConsumerTypeFast(seed=200)

        self.assertFalse(a1.PermShkDstn.seed == a2.PermShkDstn.seed)

    def test_check_conditions(self):
        """Test check_conditions method with various parameter configurations."""
        TestType = IndShockConsumerTypeFast(cycles=0, quiet=False, verbose=False)
        TestType.check_conditions()

        # make DiscFac way too big
        TestType = IndShockConsumerTypeFast(cycles=0, DiscFac=1.06)
        TestType.check_conditions()

        # make PermGroFac big
        TestType = IndShockConsumerTypeFast(cycles=0, DiscFac=0.96, PermGroFac=[1.1])
        TestType.check_conditions()

        # make Rfree too big
        TestType = IndShockConsumerTypeFast(cycles=0, Rfree=[1.1])
        TestType.check_conditions()

        # Make unemployment very likely
        TestType = IndShockConsumerTypeFast(
            cycles=0, Rfree=[0.93], IncUnemp=0.0, UnempPrb=0.99
        )
        TestType.check_conditions()

    def test_invalid_beta(self):
        """Test that negative discount factor raises ValueError."""
        TestType = IndShockConsumerTypeFast(DiscFac=-0.1, cycles=0)
        self.assertRaises(ValueError, TestType.solve)

    def test_crra_one_not_supported(self):
        """Test that CRRA=1 (log utility) raises a clear error message."""
        TestType = IndShockConsumerTypeFast(CRRA=1.0, cycles=0)
        with self.assertRaises(ValueError) as context:
            TestType.solve()
        self.assertIn("CRRA=1", str(context.exception))
        self.assertIn("log utility", str(context.exception))

    def test_replicate_sim(self):
        """Test that simulation results are reproducible with same seed."""
        TestType = IndShockConsumerTypeFast(cycles=0, seed=12022025, T_sim=100)
        TestType.solve()
        TestType.initialize_sim()
        TestType.simulate()
        A0 = np.mean(TestType.state_now["aLvl"])

        # Make sure a simulation result is replicated when re-run
        TestType.initialize_sim()
        TestType.simulate()
        A1 = np.mean(TestType.state_now["aLvl"])
        self.assertAlmostEqual(A0, A1)


class testBufferStock(unittest.TestCase):
    """Tests of the results of the BufferStock REMARK."""

    def setUp(self):
        # Set the parameters for the baseline results in the paper
        # using the variable values defined in the cell above
        self.base_params = {
            "PermGroFac": [1.03],
            "Rfree": [1.04],
            "DiscFac": 0.96,
            "CRRA": 2.0,
            "UnempPrb": 0.005,
            "IncUnemp": 0.0,
            "PermShkStd": [0.1],
            "TranShkStd": [0.1],
            "LivPrb": [1.0],
            "CubicBool": True,
            "T_cycle": 1,
            "BoroCnstArt": None,
        }

    def test_baseEx(self):
        baseEx = IndShockConsumerTypeFast(**self.base_params)
        baseEx.cycles = 100  # Make this type have a finite horizon (Set T = 100)

        baseEx.solve()
        baseEx.unpack("cFunc")

        m = np.linspace(0, 9.5, 1000)

        c_m = baseEx.cFunc[0](m)
        c_t1 = baseEx.cFunc[-2](m)
        c_t5 = baseEx.cFunc[-6](m)
        c_t10 = baseEx.cFunc[-11](m)

        self.assertAlmostEqual(c_m[500], 1.40081, places=HARK_PRECISION)
        self.assertAlmostEqual(c_t1[500], 2.92274, places=HARK_PRECISION)
        self.assertAlmostEqual(c_t5[500], 1.73506, places=HARK_PRECISION)
        self.assertAlmostEqual(c_t10[500], 1.49914, places=HARK_PRECISION)
        self.assertAlmostEqual(c_t10[600], 1.61015, places=HARK_PRECISION)
        self.assertAlmostEqual(c_t10[700], 1.71965, places=HARK_PRECISION)

    def test_GICRawFails(self):
        GICRaw_fail_dictionary = dict(self.base_params)
        GICRaw_fail_dictionary["Rfree"] = [1.08]
        GICRaw_fail_dictionary["PermGroFac"] = [1.00]
        GICRaw_fail_dictionary["cycles"] = (
            0  # cycles=0 makes this an infinite horizon consumer
        )

        GICRawFailExample = IndShockConsumerTypeFast(**GICRaw_fail_dictionary)

        GICRawFailExample.solve()
        GICRawFailExample.unpack("cFunc")
        m = np.linspace(0, 5, 1000)
        c_m = GICRawFailExample.cFunc[0](m)

        self.assertAlmostEqual(c_m[500], 0.77726, places=HARK_PRECISION)
        self.assertAlmostEqual(c_m[700], 0.83926, places=HARK_PRECISION)

        self.assertFalse(GICRawFailExample.conditions["GICRaw"])

    def test_infinite_horizon(self):
        baseEx_inf = IndShockConsumerTypeFast(**self.base_params)
        baseEx_inf.cycles = 0
        baseEx_inf.solve()
        baseEx_inf.unpack("cFunc")

        m1 = np.linspace(
            1, baseEx_inf.solution[0].mNrmStE, 50
        )  # m1 defines the plot range on the left of target m value (e.g. m <= target m)
        c_m1 = baseEx_inf.cFunc[0](m1)

        self.assertAlmostEqual(c_m1[0], 0.85279, places=HARK_PRECISION)
        self.assertAlmostEqual(c_m1[-1], 1.00363, places=HARK_PRECISION)

        x1 = np.linspace(0, 25, 1000)
        cfunc_m = baseEx_inf.cFunc[0](x1)

        self.assertAlmostEqual(cfunc_m[500], 1.89021, places=HARK_PRECISION)
        self.assertAlmostEqual(cfunc_m[700], 2.15915, places=HARK_PRECISION)

        m = np.linspace(0.001, 8, 1000)

        # Use the HARK method derivative to get the derivative of cFunc, and the values are just the MPC
        MPC = baseEx_inf.cFunc[0].derivative(m)

        self.assertAlmostEqual(MPC[500], 0.08415, places=HARK_PRECISION)
        self.assertAlmostEqual(MPC[700], 0.07173, places=HARK_PRECISION)


class testIndShockConsumerTypeFastExample(unittest.TestCase):
    def test_infinite_horizon(self):
        IndShockExample = IndShockConsumerTypeFast(**IdiosyncDict)
        IndShockExample.cycles = 0
        IndShockExample.solve()

        self.assertAlmostEqual(
            IndShockExample.solution[0].mNrmStE, 1.54882, places=HARK_PRECISION
        )
        self.assertAlmostEqual(
            IndShockExample.solution[0].cFunc.functions[0].x_list[0],
            -0.25018,
            places=HARK_PRECISION,
        )

        IndShockExample.track_vars = ["aNrm", "mNrm", "cNrm", "pLvl"]
        IndShockExample.initialize_sim()
        IndShockExample.simulate()

        # Verify simulation produces valid arrays
        self.assertEqual(IndShockExample.history["mNrm"].shape[1], 10000)


class testIndShockConsumerTypeFastLifecycle(unittest.TestCase):
    def test_lifecyle(self):
        LifecycleExample = IndShockConsumerTypeFast(**LifecycleDict)
        LifecycleExample.cycles = 1
        LifecycleExample.solve()

        self.assertEqual(len(LifecycleExample.solution), 11)

        mMin = np.min(
            [
                LifecycleExample.solution[t].mNrmMin
                for t in range(LifecycleExample.T_cycle)
            ]
        )

        self.assertAlmostEqual(
            LifecycleExample.solution[5].cFunc(3).tolist(),
            2.12998,
            places=HARK_PRECISION,
        )


class testIndShockConsumerTypeFastCyclical(unittest.TestCase):
    def test_cyclical(self):
        CyclicalExample = IndShockConsumerTypeFast(**CyclicalDict)
        CyclicalExample.cycles = 0  # Make this consumer type have an infinite horizon
        CyclicalExample.solve()

        self.assertAlmostEqual(
            CyclicalExample.solution[3].cFunc(3).tolist(),
            1.59584,
            places=HARK_PRECISION,
        )


class testValueFunction(unittest.TestCase):
    """Tests for the value function implementation in the fast solver."""

    def setUp(self):
        self.params = {
            "PermGroFac": [1.03],
            "Rfree": [1.04],
            "DiscFac": 0.96,
            "CRRA": 2.0,
            "UnempPrb": 0.005,
            "IncUnemp": 0.0,
            "PermShkStd": [0.1],
            "TranShkStd": [0.1],
            "LivPrb": [1.0],
            "CubicBool": False,
            "vFuncBool": True,
            "T_cycle": 1,
            "BoroCnstArt": 0.0,
        }

    def test_vFunc_matches_standard(self):
        """Test that Fast vFunc matches standard implementation."""
        # Standard version
        std = IndShockConsumerType(**self.params)
        std.cycles = 0
        std.solve()

        # Fast version
        fast = IndShockConsumerTypeFast(**self.params)
        fast.cycles = 0
        fast.solve()

        # Compare at various points
        # Value functions may have slightly more numerical error due to
        # additional transformations, so we use HARK_PRECISION - 1
        test_points = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
        for m in test_points:
            std_v = std.solution[0].vFunc(m)
            fast_v = fast.solution[0].vFunc(m)
            self.assertAlmostEqual(std_v, fast_v, places=HARK_PRECISION - 1)

    def test_vFunc_finite_values(self):
        """Test that vFunc returns finite values."""
        fast = IndShockConsumerTypeFast(**self.params)
        fast.cycles = 0
        fast.solve()

        test_points = [0.5, 1.0, 2.0, 5.0, 10.0]
        for m in test_points:
            v = fast.solution[0].vFunc(m)
            self.assertTrue(np.isfinite(v), f"vFunc({m}) = {v} is not finite")

    def test_vFunc_lifecycle(self):
        """Test vFunc in lifecycle model."""
        lifecycle_params = init_lifecycle.copy()
        lifecycle_params["vFuncBool"] = True

        fast = IndShockConsumerTypeFast(**lifecycle_params)
        fast.cycles = 1
        fast.solve()

        # Check vFunc at various periods and values
        for t in [0, 5, 9]:
            v = fast.solution[t].vFunc(2.0)
            self.assertTrue(np.isfinite(v), f"vFunc at t={t} returned {v}")

    def test_vFunc_with_cubic(self):
        """Test vFunc with cubic interpolation enabled."""
        params = self.params.copy()
        params["CubicBool"] = True
        params["BoroCnstArt"] = None

        fast = IndShockConsumerTypeFast(**params)
        fast.cycles = 0
        fast.solve()

        test_points = [1.0, 2.0, 5.0]
        for m in test_points:
            v = fast.solution[0].vFunc(m)
            self.assertTrue(np.isfinite(v), f"vFunc({m}) = {v} is not finite")


class testConsistencyWithStandard(unittest.TestCase):
    """Tests that Fast implementation produces identical results to standard."""

    def test_cFunc_matches_exactly(self):
        """Test that consumption functions match between implementations."""
        params = IdiosyncDict.copy()

        std = IndShockConsumerType(**params)
        std.cycles = 0
        std.solve()

        fast = IndShockConsumerTypeFast(**params)
        fast.cycles = 0
        fast.solve()

        test_points = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
        for m in test_points:
            std_c = std.solution[0].cFunc(m)
            fast_c = fast.solution[0].cFunc(m)
            self.assertAlmostEqual(std_c, fast_c, places=HARK_PRECISION)

    def test_solution_attributes_match(self):
        """Test that solution attributes match between implementations."""
        params = IdiosyncDict.copy()

        std = IndShockConsumerType(**params)
        std.cycles = 0
        std.solve()

        fast = IndShockConsumerTypeFast(**params)
        fast.cycles = 0
        fast.solve()

        # Check key solution attributes
        self.assertAlmostEqual(
            std.solution[0].mNrmStE,
            fast.solution[0].mNrmStE,
            places=HARK_PRECISION,
        )
        self.assertAlmostEqual(
            std.solution[0].MPCmin,
            fast.solution[0].MPCmin,
            places=HARK_PRECISION,
        )
        self.assertAlmostEqual(
            std.solution[0].MPCmax,
            fast.solution[0].MPCmax,
            places=HARK_PRECISION,
        )
        self.assertAlmostEqual(
            std.solution[0].hNrm,
            fast.solution[0].hNrm,
            places=HARK_PRECISION,
        )

    def test_lifecycle_matches(self):
        """Test lifecycle model matches standard implementation."""
        std = IndShockConsumerType(**LifecycleDict)
        std.cycles = 1
        std.solve()

        fast = IndShockConsumerTypeFast(**LifecycleDict)
        fast.cycles = 1
        fast.solve()

        # Check consumption at each period
        for t in range(10):
            for m in [1.0, 2.0, 3.0]:
                std_c = std.solution[t].cFunc(m)
                fast_c = fast.solution[t].cFunc(m)
                self.assertAlmostEqual(std_c, fast_c, places=HARK_PRECISION)


class testTerminalSolutionBoundaries(unittest.TestCase):
    """Tests for edge cases at terminal solution grid boundaries."""

    def test_near_zero_m_values(self):
        """Test behavior for very small m values near grid minimum."""
        fast = IndShockConsumerTypeFast(cycles=0)
        fast.solve()

        # Test values near the grid minimum (1e-6)
        small_m_values = [1e-5, 1e-4, 0.001, 0.01]
        for m in small_m_values:
            c = fast.solution[0].cFunc(m)
            self.assertTrue(np.isfinite(c), f"cFunc({m}) = {c} is not finite")
            self.assertGreaterEqual(c, 0, f"cFunc({m}) = {c} is negative")

    def test_large_m_values(self):
        """Test behavior for large m values at and beyond grid maximum."""
        fast = IndShockConsumerTypeFast(cycles=0)
        fast.solve()

        # Test values at and beyond the grid maximum (100.0)
        large_m_values = [50.0, 100.0, 150.0, 200.0]
        for m in large_m_values:
            c = fast.solution[0].cFunc(m)
            self.assertTrue(np.isfinite(c), f"cFunc({m}) = {c} is not finite")
            self.assertGreater(c, 0, f"cFunc({m}) = {c} is not positive")
            # Consumption should be less than resources
            self.assertLess(c, m, f"cFunc({m}) = {c} exceeds resources")

    def test_vFunc_boundary_values(self):
        """Test vFunc at terminal solution grid boundaries."""
        params = {
            "CRRA": 2.0,
            "vFuncBool": True,
            "cycles": 0,
        }
        fast = IndShockConsumerTypeFast(**params)
        fast.solve()

        # Test various m values including boundaries
        test_points = [0.001, 0.1, 1.0, 10.0, 50.0, 100.0]
        for m in test_points:
            v = fast.solution[0].vFunc(m)
            self.assertTrue(np.isfinite(v), f"vFunc({m}) = {v} is not finite")

    def test_crra_near_one_rejected(self):
        """Test that CRRA values very close to 1 are also rejected."""
        # Values very close to 1.0 should also be rejected due to np.isclose
        near_one_values = [1.0, 0.9999999999, 1.0000000001]
        for crra in near_one_values:
            agent = IndShockConsumerTypeFast(CRRA=crra, cycles=0)
            with self.assertRaises(ValueError) as context:
                agent.solve()
            self.assertIn("CRRA=1", str(context.exception))


class testTerminalSolutionDirect(unittest.TestCase):
    """Direct unit tests for the make_solution_terminal_fast function."""

    def test_terminal_solution_values(self):
        """Test that terminal solution has correct mathematical properties."""
        from HARK.ConsumptionSaving.ConsIndShockModelFast import (
            make_solution_terminal_fast,
            IndShockSolution,
            TERMINAL_GRID_MIN,
            TERMINAL_GRID_MAX,
            TERMINAL_GRID_SIZE,
        )

        CRRA = 2.0
        terminal = make_solution_terminal_fast(IndShockSolution, CRRA)

        # Grid should use module constants
        self.assertEqual(len(terminal.mNrmGrid), TERMINAL_GRID_SIZE)
        self.assertAlmostEqual(terminal.mNrmGrid[0], TERMINAL_GRID_MIN)
        self.assertAlmostEqual(terminal.mNrmGrid[-1], TERMINAL_GRID_MAX)

        # At terminal: vNvrs(m) = m (consume everything, so v(m) = u(m), vNvrs = u_inv(u(m)) = m)
        np.testing.assert_array_almost_equal(terminal.vNvrs, terminal.mNrmGrid)

        # vNvrsP = d(vNvrs)/dm = 1 everywhere
        np.testing.assert_array_almost_equal(
            terminal.vNvrsP, np.ones_like(terminal.mNrmGrid)
        )

        # MPCminNvrs = 1 (not 0 as was incorrectly set before)
        self.assertEqual(terminal.MPCminNvrs, 1.0)

        # hNrm = 0 at terminal (no future income)
        self.assertEqual(terminal.hNrm, 0.0)

        # MPC = 1 everywhere at terminal
        np.testing.assert_array_almost_equal(terminal.MPC, np.array([1.0, 1.0]))

    def test_terminal_solution_with_different_crra(self):
        """Test terminal solution with various CRRA values."""
        from HARK.ConsumptionSaving.ConsIndShockModelFast import (
            make_solution_terminal_fast,
            IndShockSolution,
        )

        # Test with various valid CRRA values
        for crra in [0.5, 1.5, 2.0, 3.0, 5.0]:
            with self.subTest(crra=crra):
                terminal = make_solution_terminal_fast(IndShockSolution, crra)
                # The key mathematical property: vNvrs(m) = m holds for all CRRA != 1
                np.testing.assert_array_almost_equal(terminal.vNvrs, terminal.mNrmGrid)
                self.assertEqual(terminal.MPCminNvrs, 1.0)
                self.assertEqual(terminal.hNrm, 0.0)


class testCRRARange(unittest.TestCase):
    """Test solver with various CRRA values."""

    def test_various_crra_values(self):
        """Test solver works correctly across CRRA range."""
        crra_values = [0.5, 1.5, 2.0, 3.0, 5.0]
        for crra in crra_values:
            with self.subTest(crra=crra):
                fast = IndShockConsumerTypeFast(CRRA=crra, cycles=0)
                fast.solve()
                # Verify solution is valid
                c = fast.solution[0].cFunc(2.0)
                self.assertTrue(
                    np.isfinite(c), f"cFunc(2.0) not finite for CRRA={crra}"
                )
                self.assertGreater(c, 0, f"cFunc(2.0) not positive for CRRA={crra}")

    def test_crra_consistency_across_values(self):
        """Test that fast and standard solvers match across CRRA values."""
        crra_values = [0.5, 1.5, 2.0, 3.0]
        for crra in crra_values:
            with self.subTest(crra=crra):
                std = IndShockConsumerType(CRRA=crra, cycles=0)
                std.solve()

                fast = IndShockConsumerTypeFast(CRRA=crra, cycles=0)
                fast.solve()

                # Check consumption function matches
                for m in [1.0, 2.0, 5.0]:
                    std_c = std.solution[0].cFunc(m)
                    fast_c = fast.solution[0].cFunc(m)
                    self.assertAlmostEqual(std_c, fast_c, places=HARK_PRECISION)


class testVPfuncConsistency(unittest.TestCase):
    """Tests for marginal value function consistency."""

    def test_vPfunc_matches_standard(self):
        """Test that marginal value functions match between implementations."""
        params = IdiosyncDict.copy()
        params["vFuncBool"] = True

        std = IndShockConsumerType(**params)
        std.cycles = 0
        std.solve()

        fast = IndShockConsumerTypeFast(**params)
        fast.cycles = 0
        fast.solve()

        # Compare vPfunc at various points
        # vPfunc may have slightly more numerical error, so use HARK_PRECISION - 1
        test_points = [1.0, 2.0, 3.0, 5.0]
        for m in test_points:
            with self.subTest(m=m):
                std_vp = std.solution[0].vPfunc(m)
                fast_vp = fast.solution[0].vPfunc(m)
                self.assertAlmostEqual(std_vp, fast_vp, places=HARK_PRECISION - 1)


class testPerfForesightFast(unittest.TestCase):
    """Tests for PerfForesightConsumerTypeFast."""

    def test_perf_foresight_crra_one_not_supported(self):
        """Test that CRRA=1 raises error for PerfForesightConsumerTypeFast."""
        from HARK.ConsumptionSaving.ConsIndShockModelFast import (
            PerfForesightConsumerTypeFast,
        )

        agent = PerfForesightConsumerTypeFast(CRRA=1.0, cycles=0)
        with self.assertRaises(ValueError) as context:
            agent.solve()
        self.assertIn("CRRA=1", str(context.exception))
        self.assertIn("log utility", str(context.exception))

    def test_perf_foresight_basic_solve(self):
        """Test that PerfForesightConsumerTypeFast solves correctly."""
        from HARK.ConsumptionSaving.ConsIndShockModelFast import (
            PerfForesightConsumerTypeFast,
        )

        agent = PerfForesightConsumerTypeFast(CRRA=2.0, cycles=0)
        agent.solve()

        # Verify solution is valid
        c = agent.solution[0].cFunc(2.0)
        self.assertTrue(np.isfinite(c))
        self.assertGreater(c, 0)


class testCRRANearOneWarning(unittest.TestCase):
    """Tests for warning when CRRA is near 1."""

    def test_crra_near_one_warning(self):
        """Test that CRRA values in (0.99, 1.01) but not exactly 1 issue a warning."""
        import warnings

        # These values are in (0.99, 1.01) but not caught by np.isclose
        near_one_values = [0.995, 1.005, 0.999, 1.001]
        for crra in near_one_values:
            with self.subTest(crra=crra):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    agent = IndShockConsumerTypeFast(CRRA=crra, cycles=0)
                    agent.solve()
                    # Check that a warning was issued
                    self.assertEqual(len(w), 1)
                    self.assertIn("close to 1", str(w[0].message))
                    self.assertIn("numerical instability", str(w[0].message))

    def test_crra_outside_warning_range_no_warning(self):
        """Test that CRRA values outside (0.99, 1.01) do not issue warnings."""
        import warnings

        safe_values = [0.5, 0.98, 1.02, 2.0, 5.0]
        for crra in safe_values:
            with self.subTest(crra=crra):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    agent = IndShockConsumerTypeFast(CRRA=crra, cycles=0)
                    agent.solve()
                    # Check that no warning was issued
                    self.assertEqual(len(w), 0)

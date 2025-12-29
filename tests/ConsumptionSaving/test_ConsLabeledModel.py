"""
Tests for labeled consumption-saving models.

Tests cover:
- PerfForesightLabeledType
- IndShockLabeledType
- RiskyAssetLabeledType
- PortfolioLabeledType
- Input validation
"""

import unittest

import numpy as np

from HARK.ConsumptionSaving.ConsLabeledModel import (
    IndShockLabeledType,
    PerfForesightLabeledType,
    PortfolioLabeledType,
    RiskyAssetLabeledType,
    make_solution_terminal_labeled,
)
from HARK.Labeled.solution import ValueFuncCRRALabeled
from tests import HARK_PRECISION


class test_PerfForesightLabeledType(unittest.TestCase):
    def setUp(self):
        self.agent = PerfForesightLabeledType()

    def test_default_solution(self):
        self.agent.solve()
        c = self.agent.solution[0].policy["cNrm"].to_numpy()
        m = self.agent.solution[0].policy["mNrm"].to_numpy()

        # Values updated after fixing borrowing constraint formula
        # to include PermGroFac (was missing in original implementation)
        self.assertAlmostEqual(m[0], -0.98058, places=HARK_PRECISION)
        self.assertAlmostEqual(m[1], -0.97854, places=HARK_PRECISION)
        self.assertEqual(c[0], 0)
        self.assertAlmostEqual(c[1], 0.00105, places=HARK_PRECISION)

    def test_solution_has_value_function(self):
        self.agent.solve()
        self.assertIsNotNone(self.agent.solution[0].value)
        self.assertIsInstance(self.agent.solution[0].value, ValueFuncCRRALabeled)

    def test_solution_has_policy(self):
        self.agent.solve()
        policy = self.agent.solution[0].policy
        self.assertIn("cNrm", policy)
        self.assertIn("mNrm", policy)


class test_IndShockConsumerType(unittest.TestCase):
    def setUp(self):
        self.agent = IndShockLabeledType(cycles=10)

    def test_IndShockLabeledType(self):
        self.agent.solve()
        c = self.agent.solution[0].policy["cNrm"].to_numpy()
        m = self.agent.solution[0].policy["mNrm"].to_numpy()

        self.assertAlmostEqual(c[4], 0.47038, places=HARK_PRECISION)
        self.assertAlmostEqual(m[4], -0.72898, places=HARK_PRECISION)

    def test_consumption_increasing_in_wealth(self):
        """Consumption should be monotonically increasing in market resources."""
        self.agent.solve()
        c = self.agent.solution[0].policy["cNrm"].to_numpy()
        # Check monotonicity (all differences should be non-negative)
        self.assertTrue(np.all(np.diff(c) >= 0))


class test_RiskyAssetLabeledType(unittest.TestCase):
    def setUp(self):
        self.agent = RiskyAssetLabeledType()
        self.agent.cycles = 10

    def test_solve(self):
        self.agent.solve()
        self.assertIsNotNone(self.agent.solution)
        self.assertEqual(len(self.agent.solution), 11)  # 10 cycles + terminal

    def test_policy_shape(self):
        self.agent.solve()
        c = self.agent.solution[0].policy["cNrm"]
        m = self.agent.solution[0].policy["mNrm"]
        self.assertTrue(len(c) > 0)
        self.assertTrue(len(m) > 0)

    def test_consumption_positive(self):
        """Consumption should be non-negative."""
        self.agent.solve()
        c = self.agent.solution[0].policy["cNrm"].to_numpy()
        self.assertTrue(np.all(c >= 0))

    def test_solution_values(self):
        """Check that solution values are reasonable."""
        self.agent.solve()
        c = self.agent.solution[0].policy["cNrm"].to_numpy()
        m = self.agent.solution[0].policy["mNrm"].to_numpy()
        # Consumption at moderate wealth should be positive and reasonable
        self.assertAlmostEqual(c[5], 0.78381, places=HARK_PRECISION)
        self.assertAlmostEqual(m[5], 0.84578, places=HARK_PRECISION)


class test_PortfolioLabeledType(unittest.TestCase):
    def setUp(self):
        self.agent = PortfolioLabeledType()
        self.agent.cycles = 10

    def test_solve(self):
        self.agent.solve()
        self.assertIsNotNone(self.agent.solution)
        self.assertEqual(len(self.agent.solution), 10)

    def test_policy_shape(self):
        self.agent.solve()
        c = self.agent.solution[0].policy["cNrm"]
        self.assertTrue(len(c) > 0)

    def test_consumption_positive(self):
        """Consumption should be non-negative."""
        self.agent.solve()
        c = self.agent.solution[0].policy["cNrm"].to_numpy()
        self.assertTrue(np.all(c >= 0))

    def test_solution_values(self):
        """Check that solution values are reasonable."""
        self.agent.solve()
        c = self.agent.solution[0].policy["cNrm"].to_numpy()
        m = self.agent.solution[0].policy["mNrm"].to_numpy()
        self.assertAlmostEqual(c[5], 0.65750, places=HARK_PRECISION)
        self.assertAlmostEqual(m[5], 0.73061, places=HARK_PRECISION)

    def test_continuation_exists(self):
        """Portfolio model should have continuation function."""
        self.agent.solve()
        self.assertIsNotNone(self.agent.solution[0].continuation)


class test_InputValidation(unittest.TestCase):
    """Test input validation for labeled models."""

    def test_terminal_solution_crra_positive(self):
        """CRRA must be positive."""
        grid = np.array([0.0, 1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            make_solution_terminal_labeled(-1.0, grid)

    def test_terminal_solution_crra_finite(self):
        """CRRA must be finite."""
        grid = np.array([0.0, 1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            make_solution_terminal_labeled(np.inf, grid)

    def test_terminal_solution_grid_nonempty(self):
        """Grid cannot be empty."""
        with self.assertRaises(ValueError):
            make_solution_terminal_labeled(2.0, np.array([]))

    def test_terminal_solution_grid_nonnegative(self):
        """Grid values must be non-negative."""
        with self.assertRaises(ValueError):
            make_solution_terminal_labeled(2.0, np.array([-1.0, 0.0, 1.0]))


class test_EconomicParameterValidation(unittest.TestCase):
    """Test validation of economic parameters in solvers."""

    def test_livprb_must_be_in_range(self):
        """LivPrb must be in (0, 1]."""
        agent = PerfForesightLabeledType()
        agent.LivPrb = [0.0]  # Invalid - must be > 0
        with self.assertRaises(ValueError):
            agent.solve()

    def test_discfac_must_be_positive(self):
        """DiscFac must be positive."""
        agent = PerfForesightLabeledType()
        agent.DiscFac = 0.0  # Invalid - must be > 0
        with self.assertRaises(ValueError):
            agent.solve()

    def test_rfree_must_be_positive(self):
        """Rfree must be positive."""
        agent = PerfForesightLabeledType()
        agent.Rfree = [0.0]  # Invalid - must be > 0
        with self.assertRaises(ValueError):
            agent.solve()

    def test_permgrofac_must_be_positive(self):
        """PermGroFac must be positive."""
        agent = PerfForesightLabeledType()
        agent.PermGroFac = [0.0]  # Invalid - must be > 0
        with self.assertRaises(ValueError):
            agent.solve()


class test_BackwardsCompatibility(unittest.TestCase):
    """Test backwards compatibility of imports."""

    def test_import_from_ConsLabeledModel(self):
        """All classes should be importable from ConsLabeledModel."""
        from HARK.ConsumptionSaving.ConsLabeledModel import (
            ConsFixedPortfolioLabeledSolver,
            ConsIndShockLabeledSolver,
            ConsPerfForesightLabeledSolver,
            ConsPortfolioLabeledSolver,
            ConsRiskyAssetLabeledSolver,
            ConsumerSolutionLabeled,
            IndShockLabeledType,
            PerfForesightLabeledType,
            PortfolioLabeledType,
            RiskyAssetLabeledType,
            ValueFuncCRRALabeled,
            init_ind_shock_labeled,
            init_perf_foresight_labeled,
            init_portfolio_labeled,
            init_risky_asset_labeled,
        )

        # Just check they're importable and not None
        self.assertIsNotNone(PerfForesightLabeledType)
        self.assertIsNotNone(IndShockLabeledType)
        self.assertIsNotNone(RiskyAssetLabeledType)
        self.assertIsNotNone(PortfolioLabeledType)
        self.assertIsNotNone(ConsPerfForesightLabeledSolver)
        self.assertIsNotNone(ConsIndShockLabeledSolver)
        self.assertIsNotNone(ConsRiskyAssetLabeledSolver)
        self.assertIsNotNone(ConsFixedPortfolioLabeledSolver)
        self.assertIsNotNone(ConsPortfolioLabeledSolver)
        self.assertIsNotNone(ValueFuncCRRALabeled)
        self.assertIsNotNone(ConsumerSolutionLabeled)
        self.assertIsNotNone(init_perf_foresight_labeled)
        self.assertIsNotNone(init_ind_shock_labeled)
        self.assertIsNotNone(init_risky_asset_labeled)
        self.assertIsNotNone(init_portfolio_labeled)

    def test_config_dicts_are_dicts(self):
        """Config dicts should be proper dictionaries."""
        from HARK.ConsumptionSaving.ConsLabeledModel import (
            init_ind_shock_labeled,
            init_perf_foresight_labeled,
            init_portfolio_labeled,
            init_risky_asset_labeled,
        )

        self.assertIsInstance(init_perf_foresight_labeled, dict)
        self.assertIsInstance(init_ind_shock_labeled, dict)
        self.assertIsInstance(init_risky_asset_labeled, dict)
        self.assertIsInstance(init_portfolio_labeled, dict)

    def test_config_dicts_have_constructors(self):
        """Config dicts should have constructors key."""
        from HARK.ConsumptionSaving.ConsLabeledModel import (
            init_ind_shock_labeled,
            init_perf_foresight_labeled,
            init_portfolio_labeled,
            init_risky_asset_labeled,
        )

        self.assertIn("constructors", init_perf_foresight_labeled)
        self.assertIn("constructors", init_ind_shock_labeled)
        self.assertIn("constructors", init_risky_asset_labeled)
        self.assertIn("constructors", init_portfolio_labeled)


if __name__ == "__main__":
    unittest.main()

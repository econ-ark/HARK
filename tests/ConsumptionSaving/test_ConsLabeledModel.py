"""
Tests for labeled consumption-saving models.

Tests cover:
- PerfForesightLabeledType
- IndShockLabeledType
- RiskyAssetLabeledType
- PortfolioLabeledType
- Input validation
- Transitions classes
- Solution distance calculation
- Factory functions
"""

import unittest
from types import SimpleNamespace

import numpy as np
import xarray as xr

from HARK.ConsumptionSaving.ConsLabeledModel import (
    IndShockLabeledType,
    PerfForesightLabeledType,
    PortfolioLabeledType,
    RiskyAssetLabeledType,
    make_solution_terminal_labeled,
)
from HARK.Labeled.solution import ConsumerSolutionLabeled, ValueFuncCRRALabeled
from HARK.Labeled.transitions import (
    FixedPortfolioTransitions,
    IndShockTransitions,
    PerfectForesightTransitions,
    PortfolioTransitions,
    RiskyAssetTransitions,
)
from HARK.rewards import UtilityFuncCRRA
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
        self.assertGreater(len(c), 0)
        self.assertGreater(len(m), 0)

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
        self.assertGreater(len(c), 0)

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


class test_InfiniteHorizon(unittest.TestCase):
    """Test infinite horizon convergence (cycles=0)."""

    def test_indshock_converges(self):
        """IndShockLabeledType should converge in infinite horizon."""
        agent = IndShockLabeledType(cycles=0)
        agent.solve()
        # Should have exactly one solution (converged)
        self.assertEqual(len(agent.solution), 1)
        # Solution should have reasonable values
        c = agent.solution[0].policy["cNrm"].to_numpy()
        self.assertTrue(np.all(c >= 0))
        self.assertTrue(np.all(np.diff(c) >= 0))  # Monotonic


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


class test_Transitions(unittest.TestCase):
    """Test transition classes for labeled models."""

    def setUp(self):
        self.params = SimpleNamespace(
            Rfree=1.03,
            PermGroFac=1.01,
            CRRA=2.0,
            RiskyShareFixed=0.5,
        )
        self.utility = UtilityFuncCRRA(2.0)

    def test_perfect_foresight_post_state(self):
        """Test perfect foresight state transition."""
        trans = PerfectForesightTransitions()
        post_state = {"aNrm": 10.0}
        result = trans.post_state(post_state, None, self.params)

        expected_mNrm = 10.0 * 1.03 / 1.01 + 1
        self.assertAlmostEqual(result["mNrm"], expected_mNrm, places=5)

    def test_perfect_foresight_no_shocks_required(self):
        """Perfect foresight should not require shocks."""
        trans = PerfectForesightTransitions()
        self.assertFalse(trans.requires_shocks)

    def test_ind_shock_post_state(self):
        """Test income shock state transition."""
        trans = IndShockTransitions()
        post_state = {"aNrm": 10.0}
        shocks = {"perm": 1.05, "tran": 0.9}
        result = trans.post_state(post_state, shocks, self.params)

        expected_mNrm = 10.0 * 1.03 / (1.01 * 1.05) + 0.9
        self.assertAlmostEqual(result["mNrm"], expected_mNrm, places=5)

    def test_ind_shock_requires_shocks(self):
        """IndShock should require shocks."""
        trans = IndShockTransitions()
        self.assertTrue(trans.requires_shocks)

    def test_ind_shock_missing_keys(self):
        """IndShock should raise KeyError for missing shock keys."""
        trans = IndShockTransitions()
        post_state = {"aNrm": 10.0}
        shocks = {"perm": 1.05}  # Missing 'tran'
        with self.assertRaises(KeyError):
            trans.post_state(post_state, shocks, self.params)

    def test_risky_asset_post_state(self):
        """Test risky asset state transition."""
        trans = RiskyAssetTransitions()
        post_state = {"aNrm": 10.0}
        shocks = {"perm": 1.05, "tran": 0.9, "risky": 1.08}
        result = trans.post_state(post_state, shocks, self.params)

        expected_mNrm = 10.0 * 1.08 / (1.01 * 1.05) + 0.9
        self.assertAlmostEqual(result["mNrm"], expected_mNrm, places=5)

    def test_risky_asset_missing_keys(self):
        """RiskyAsset should raise KeyError for missing shock keys."""
        trans = RiskyAssetTransitions()
        post_state = {"aNrm": 10.0}
        shocks = {"perm": 1.05, "tran": 0.9}  # Missing 'risky'
        with self.assertRaises(KeyError):
            trans.post_state(post_state, shocks, self.params)

    def test_fixed_portfolio_post_state(self):
        """Test fixed portfolio state transition."""
        trans = FixedPortfolioTransitions()
        post_state = {"aNrm": 10.0}
        shocks = {"perm": 1.05, "tran": 0.9, "risky": 1.08}
        result = trans.post_state(post_state, shocks, self.params)

        # rPort = Rfree + (risky - Rfree) * RiskyShareFixed
        r_port = 1.03 + (1.08 - 1.03) * 0.5
        expected_mNrm = 10.0 * r_port / (1.01 * 1.05) + 0.9
        self.assertAlmostEqual(result["mNrm"], expected_mNrm, places=5)
        self.assertIn("rPort", result)
        self.assertIn("rDiff", result)

    def test_portfolio_post_state(self):
        """Test portfolio state transition with stigma."""
        trans = PortfolioTransitions()
        post_state = {"aNrm": 10.0, "stigma": 0.3}
        shocks = {"perm": 1.05, "tran": 0.9, "risky": 1.08}
        result = trans.post_state(post_state, shocks, self.params)

        # rPort = Rfree + (risky - Rfree) * stigma
        r_port = 1.03 + (1.08 - 1.03) * 0.3
        expected_mNrm = 10.0 * r_port / (1.01 * 1.05) + 0.9
        self.assertAlmostEqual(result["mNrm"], expected_mNrm, places=5)


class test_SolutionDistance(unittest.TestCase):
    """Test solution distance calculation."""

    def test_distance_same_solution(self):
        """Distance between identical solutions should be zero."""
        agent = IndShockLabeledType(cycles=10)
        agent.solve()
        sol = agent.solution[0]
        dist = sol.distance(sol)
        self.assertEqual(dist, 0.0)

    def test_distance_different_solutions(self):
        """Distance between different solutions should be positive."""
        agent = IndShockLabeledType(cycles=10)
        agent.solve()
        sol0 = agent.solution[0]
        sol1 = agent.solution[1]
        dist = sol0.distance(sol1)
        self.assertGreater(dist, 0)
        self.assertTrue(np.isfinite(dist))

    def test_distance_invalid_type(self):
        """Distance with non-solution should raise TypeError."""
        agent = IndShockLabeledType(cycles=10)
        agent.solve()
        sol = agent.solution[0]
        with self.assertRaises(TypeError):
            sol.distance("not a solution")


class test_ValueFuncValidation(unittest.TestCase):
    """Test ValueFuncCRRALabeled validation."""

    def test_invalid_dataset_type(self):
        """Should raise TypeError for non-Dataset input."""
        with self.assertRaises(TypeError):
            ValueFuncCRRALabeled("not a dataset", 2.0)

    def test_missing_required_variables(self):
        """Should raise ValueError for missing required variables."""
        ds = xr.Dataset({"v": xr.DataArray([1.0, 2.0])})
        with self.assertRaises(ValueError) as ctx:
            ValueFuncCRRALabeled(ds, 2.0)
        self.assertIn("missing required variables", str(ctx.exception).lower())


class test_ConsumerSolutionValidation(unittest.TestCase):
    """Test ConsumerSolutionLabeled validation."""

    def test_invalid_value_type(self):
        """Should raise TypeError for invalid value type."""
        policy = xr.Dataset({"cNrm": xr.DataArray([1.0, 2.0])})
        with self.assertRaises(TypeError):
            ConsumerSolutionLabeled(
                value="not a vfunc", policy=policy, continuation=None
            )

    def test_invalid_policy_type(self):
        """Should raise TypeError for invalid policy type."""
        # Create a valid value function first
        agent = PerfForesightLabeledType()
        agent.solve()
        vfunc = agent.solution[0].value

        with self.assertRaises(TypeError):
            ConsumerSolutionLabeled(
                value=vfunc, policy="not a dataset", continuation=None
            )


class test_FixedPortfolioSolver(unittest.TestCase):
    """Test ConsFixedPortfolioLabeledSolver functionality."""

    def test_fixed_portfolio_solves(self):
        """Fixed portfolio agent should solve without error."""

        # Create a risky asset type and use fixed portfolio
        agent = RiskyAssetLabeledType()
        agent.cycles = 5
        agent.RiskyShareFixed = [0.5]  # Fixed 50% allocation
        agent.solve()
        self.assertIsNotNone(agent.solution)

    def test_fixed_portfolio_share_validation(self):
        """RiskyShareFixed must be in [0, 1]."""
        from HARK.ConsumptionSaving.ConsLabeledModel import (
            ConsFixedPortfolioLabeledSolver,
        )

        # We test via the solver directly
        agent = PerfForesightLabeledType()
        agent.solve()
        terminal = agent.solution[0]

        with self.assertRaises(ValueError):
            ConsFixedPortfolioLabeledSolver(
                solution_next=terminal,
                ShockDstn=None,  # Would normally need a distribution
                LivPrb=0.99,
                DiscFac=0.96,
                CRRA=2.0,
                Rfree=1.03,
                PermGroFac=1.01,
                BoroCnstArt=0.0,
                aXtraGrid=np.linspace(0, 10, 20),
                RiskyShareFixed=1.5,  # Invalid - must be <= 1
            )


class test_TerminalSolutionValidation(unittest.TestCase):
    """Test terminal solution factory validation."""

    def test_grid_not_increasing(self):
        """Grid must be strictly increasing."""
        with self.assertRaises(ValueError):
            make_solution_terminal_labeled(2.0, np.array([1.0, 0.5, 2.0]))

    def test_crra_nan(self):
        """CRRA cannot be NaN."""
        with self.assertRaises(ValueError):
            make_solution_terminal_labeled(np.nan, np.array([0.0, 1.0, 2.0]))


if __name__ == "__main__":
    unittest.main()

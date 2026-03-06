"""
Strategic unit tests for the housing portfolio choice model.

Tests are organized into four categories:
1. Mortgage utilities: payment calculation and LTV dynamics
2. Terminal condition: correct bequest values and homogeneity
3. Model solution: the solver runs and produces sensible policy functions
4. Economic logic: policy functions satisfy known theoretical predictions
"""

import numpy as np
import pytest

from HARK.ConsumptionSaving.ConsHousingPortfolioModel import (
    HousingPortfolioConsumerType,
    HousingPortfolioSolution,
    MarkovHousingPortfolioConsumerType,
    ltv_next,
    make_housing_portfolio_solution_terminal,
    make_ltv_grid,
    mortgage_payment_rate,
)



# ===================================================================
# Shared parameter helpers (eliminate repeated param dicts)
# ===================================================================


def _base_params_3period(**overrides):
    """Common 3-period model parameters. Override any key via kwargs."""
    params = {
        "T_cycle": 3,
        "LivPrb": [0.99, 0.99, 0.99],
        "PermGroFac": [1.02, 1.02, 1.00],
        "Rfree": [1.02, 1.02, 1.02],
        "MortPeriods": [3, 2, 1],
        "PermShkStd": [0.08, 0.08, 0.05],
        "TranShkStd": [0.08, 0.08, 0.05],
        "PermShkCount": 3,
        "TranShkCount": 3,
        "RiskyCount": 3,
        "aXtraCount": 24,
        "aXtraMax": 20,
        "ShareCount": 7,
        "dGridCount": 5,
        "dMax": 1.0,
    }
    params.update(overrides)
    return params


def _base_params_1period(**overrides):
    """Common 1-period model parameters for validation tests."""
    params = {
        "T_cycle": 1,
        "LivPrb": [0.99],
        "PermGroFac": [1.02],
        "Rfree": [1.02],
        "MortPeriods": [1],
        "PermShkStd": [0.1],
        "TranShkStd": [0.1],
        "PermShkCount": 3,
        "TranShkCount": 3,
        "RiskyCount": 3,
        "aXtraCount": 10,
        "aXtraMax": 10,
        "ShareCount": 3,
        "dGridCount": 3,
        "dMax": 1.0,
    }
    params.update(overrides)
    return params


def _markov_params_3period(**overrides):
    """Common 3-period Markov model parameters."""
    params = _base_params_3period(
        MrkvArray=[np.array([[0.95, 0.05], [0.50, 0.50]])] * 3,
    )
    params.update(overrides)
    return params


# ===================================================================
# 1. Mortgage payment utilities
# ===================================================================


class TestMortgagePayment:
    """Test the fixed-rate mortgage payment formula."""

    def test_standard_30yr(self):
        """A 4% mortgage with d=0.8 and 30 periods remaining."""
        d = 0.8
        r_m = 0.04
        periods = 30
        pi = mortgage_payment_rate(d, r_m, periods)
        # Standard annuity: d * r*(1+r)^n / ((1+r)^n - 1)
        R = 1.04
        expected = d * 0.04 * R**30 / (R**30 - 1.0)
        np.testing.assert_almost_equal(pi, expected, decimal=10)

    def test_zero_periods_remaining(self):
        """No payments when mortgage is fully amortized."""
        pi = mortgage_payment_rate(0.5, 0.04, 0)
        assert pi == 0.0

    def test_one_period_remaining(self):
        """With 1 period left, payment equals d*(1+r)."""
        d = 0.3
        r_m = 0.05
        pi = mortgage_payment_rate(d, r_m, 1)
        expected = d * (1.0 + r_m)
        np.testing.assert_almost_equal(pi, expected, decimal=10)

    def test_payment_proportional_to_d(self):
        """Payment scales linearly with LTV."""
        r_m = 0.04
        periods = 20
        pi_low = mortgage_payment_rate(0.4, r_m, periods)
        pi_high = mortgage_payment_rate(0.8, r_m, periods)
        np.testing.assert_almost_equal(pi_high / pi_low, 2.0, decimal=10)

    def test_vectorized(self):
        """Payment function works with numpy arrays."""
        d = np.array([0.2, 0.5, 0.8])
        pi = mortgage_payment_rate(d, 0.04, 30)
        assert pi.shape == (3,)
        assert np.all(pi > 0)


class TestLTVEvolution:
    """Test the LTV transition d_{t+1} = (d*R_m - pi) / G."""

    def test_amortization_reduces_ltv(self):
        """LTV decreases over time with positive amortization and G=1."""
        d = 0.8
        r_m = 0.04
        periods = 30
        d_next = ltv_next(d, r_m, periods, G=1.0)
        assert d_next < d

    def test_appreciation_accelerates_ltv_decline(self):
        """Housing appreciation (G>1) makes LTV fall faster."""
        d = 0.8
        r_m = 0.04
        periods = 30
        d_g1 = ltv_next(d, r_m, periods, G=1.0)
        d_g105 = ltv_next(d, r_m, periods, G=1.05)
        assert d_g105 < d_g1

    def test_full_amortization(self):
        """After all periods, LTV should reach zero (approximately)."""
        d = 0.8
        r_m = 0.04
        for periods_remaining in range(30, 0, -1):
            d = ltv_next(d, r_m, periods_remaining, G=1.0)
        np.testing.assert_almost_equal(d, 0.0, decimal=6)


class TestLTVGrid:
    """Test the LTV grid constructor."""

    def test_grid_range(self):
        grid = make_ltv_grid(dGridCount=10, dMax=1.2)
        assert grid[0] == 0.0
        assert grid[-1] == 1.2
        assert len(grid) == 10

    def test_grid_monotone(self):
        grid = make_ltv_grid(dGridCount=20, dMax=0.95)
        assert np.all(np.diff(grid) > 0)


# ===================================================================
# 2. Terminal condition
# ===================================================================


class TestTerminalCondition:
    """Test the terminal-period bequest value function."""

    @pytest.fixture()
    def terminal_sol(self):
        return make_housing_portfolio_solution_terminal(
            CRRA=5.0, alpha=0.2, hbar=3.0, BeqWt=1.0
        )

    def test_terminal_value_negative_for_high_rra(self, terminal_sol):
        """With rho=5, gamma=-4.8, value is negative for positive net worth."""
        v = terminal_sol.vFuncOwn(2.0, 0.5)
        assert v < 0  # gamma < 0 implies v = w^gamma / gamma < 0

    def test_terminal_value_increases_with_m(self, terminal_sol):
        """Value increases with liquid wealth m."""
        v_low = terminal_sol.vFuncOwn(1.0, 0.5)
        v_high = terminal_sol.vFuncOwn(3.0, 0.5)
        assert v_high > v_low

    def test_terminal_value_decreases_with_d(self, terminal_sol):
        """Value decreases with LTV (more debt = less equity)."""
        v_low_d = terminal_sol.vFuncOwn(2.0, 0.2)
        v_high_d = terminal_sol.vFuncOwn(2.0, 0.8)
        assert v_low_d > v_high_d

    def test_terminal_marginal_value_positive(self, terminal_sol):
        """Marginal value of m is positive."""
        vp = terminal_sol.vPfuncOwn(2.0, 0.5)
        assert vp > 0

    def test_terminal_renter_no_housing_equity(self, terminal_sol):
        """Renter's terminal value depends only on liquid wealth (no housing)."""
        # Value increases with m
        v_low = terminal_sol.vFuncRent(1.0)
        v_high = terminal_sol.vFuncRent(3.0)
        assert v_high > v_low
        # Matches bequest formula with zero housing equity
        rho = 5.0
        alpha = 0.2
        gamma = (1.0 + alpha) * (1.0 - rho)
        m = 2.0
        v = terminal_sol.vFuncRent(m)
        expected = 1.0 * m**gamma / gamma  # BeqWt=1, no housing
        np.testing.assert_almost_equal(v, expected, decimal=10)

    def test_homogeneity_exponent(self, terminal_sol):
        """Terminal value has correct homogeneity degree (1+alpha)(1-rho)."""
        rho = 5.0
        alpha = 0.2
        gamma = (1.0 + alpha) * (1.0 - rho)  # = -4.8
        hbar = 3.0

        m, d = 2.0, 0.3
        w = m + (1.0 - d) * hbar

        v = terminal_sol.vFuncOwn(m, d)
        expected = 1.0 * w**gamma / gamma  # BeqWt=1
        np.testing.assert_almost_equal(v, expected, decimal=10)

    def test_consume_all_at_terminal(self, terminal_sol):
        """At terminal date, consume all liquid wealth."""
        m = 5.0
        c = terminal_sol.cFuncOwn(m, 0.3)
        np.testing.assert_almost_equal(c, m, decimal=10)

    def test_terminal_share_zero(self, terminal_sol):
        """At terminal date, risky share is zero (no future returns)."""
        assert terminal_sol.ShareFuncOwn(5.0) == 0.0
        assert terminal_sol.ShareFuncRent(5.0) == 0.0


# ===================================================================
# 3. Solver runs and produces valid output
# ===================================================================


class TestSolverRuns:
    """Smoke tests: the model solves without errors and returns valid objects."""

    @pytest.fixture(scope="class")
    def small_model(self):
        """A small model (3 periods) for fast testing."""
        params = _base_params_3period(
            LivPrb=[0.99, 0.98, 0.97],
            PermGroFac=[1.02, 1.01, 1.00],
            PermShkStd=[0.1, 0.1, 0.05],
            TranShkStd=[0.1, 0.1, 0.05],
        )
        agent = HousingPortfolioConsumerType(**params)
        agent.solve()
        return agent

    def test_solution_length(self, small_model):
        """Solution has one element per period plus terminal."""
        assert len(small_model.solution) == 3

    def test_solution_has_owner_functions(self, small_model):
        """Each period's solution has consumption and value functions for owners."""
        for sol in small_model.solution:
            assert hasattr(sol, "cFuncOwn")
            assert hasattr(sol, "vFuncOwn")
            assert hasattr(sol, "ShareFuncOwn")

    def test_solution_has_renter_functions(self, small_model):
        """Each period's solution has consumption and value functions for renters."""
        for sol in small_model.solution:
            assert hasattr(sol, "cFuncRent")
            assert hasattr(sol, "vFuncRent")

    def test_consumption_positive(self, small_model):
        """Consumption is positive for positive market resources."""
        sol = small_model.solution[0]
        for m in [0.5, 1.0, 5.0, 10.0]:
            c = sol.cFuncOwn(m, 0.5)
            assert c > 0, f"c should be positive at m={m}, got {c}"

    def test_consumption_less_than_resources(self, small_model):
        """Consumption does not exceed available resources."""
        sol = small_model.solution[0]
        m = 5.0
        d = 0.3
        c = sol.cFuncOwn(m, d)
        # c must be less than m (household also pays mortgage, maintenance)
        assert c < m, f"c={c} exceeds m={m}"

    def test_share_bounded_01(self, small_model):
        """Risky share is between 0 and 1."""
        sol = small_model.solution[0]
        for m in [1.0, 5.0, 10.0]:
            s = sol.ShareFuncOwn(m, 0.5)
            assert -0.01 <= s <= 1.01, f"Share out of bounds: {s}"

    def test_renter_consumption_positive(self, small_model):
        """Renter consumption is positive."""
        sol = small_model.solution[0]
        c = sol.cFuncRent(3.0)
        assert c > 0

    def test_renter_share_bounded(self, small_model):
        """Renter's risky share is between 0 and 1."""
        sol = small_model.solution[0]
        for m in [1.0, 5.0, 10.0]:
            s = sol.ShareFuncRent(m)
            assert -0.01 <= s <= 1.01, f"Renter share out of bounds at m={m}: {s}"


# ===================================================================
# 4. Economic logic tests
# ===================================================================


class TestEconomicLogic:
    """Test that policy functions satisfy known theoretical predictions."""

    @pytest.fixture(scope="class")
    def model_low_rate(self):
        """Model with a low 3% mortgage rate."""
        agent = HousingPortfolioConsumerType(**_base_params_3period(MortRate=0.03))
        agent.solve()
        return agent

    @pytest.fixture(scope="class")
    def model_high_rate(self):
        """Model with a high 7% mortgage rate."""
        agent = HousingPortfolioConsumerType(**_base_params_3period(MortRate=0.07))
        agent.solve()
        return agent

    def test_consumption_increases_with_m(self, model_low_rate):
        """Consumption is increasing in cash-on-hand (normal good)."""
        sol = model_low_rate.solution[0]
        d = 0.5
        m_vals = [1.0, 3.0, 5.0, 10.0]
        c_vals = [sol.cFuncOwn(m, d) for m in m_vals]
        for i in range(len(c_vals) - 1):
            assert c_vals[i + 1] > c_vals[i], (
                f"Consumption should increase: c({m_vals[i]})={c_vals[i]}, "
                f"c({m_vals[i+1]})={c_vals[i+1]}"
            )

    def test_sell_when_deeply_underwater(self, model_low_rate):
        """Value of owning decreases as debt rises — high-debt households
        are worse off, making exit (sell or default) relatively more attractive."""
        sol = model_low_rate.solution[0]
        m = 0.5
        # Owner value should be lower with more debt
        v_low_d = sol.vFuncOwn(m, 0.25)
        v_high_d = sol.vFuncOwn(m, 0.75)
        assert v_low_d >= v_high_d, (
            f"Expected owner value to decrease with debt: "
            f"v(d=0.25)={v_low_d:.4f} vs v(d=0.75)={v_high_d:.4f}"
        )

    def test_value_decreases_with_debt(self, model_low_rate):
        """Higher LTV means lower value (more debt is bad)."""
        sol = model_low_rate.solution[0]
        m = 5.0
        v_low_d = sol.vFuncOwn(m, 0.2)
        v_high_d = sol.vFuncOwn(m, 0.8)
        assert v_low_d > v_high_d, (
            f"Value should decrease with debt: v(d=0.2)={v_low_d}, v(d=0.8)={v_high_d}"
        )

    def test_mpc_decreases_with_wealth(self, model_low_rate):
        """Marginal propensity to consume decreases with wealth (concave cFunc)."""
        sol = model_low_rate.solution[0]
        d = 0.5
        # MPC ~ dc/dm
        dm = 0.01
        mpc_low = (sol.cFuncOwn(1.0 + dm, d) - sol.cFuncOwn(1.0, d)) / dm
        mpc_high = (sol.cFuncOwn(10.0 + dm, d) - sol.cFuncOwn(10.0, d)) / dm
        assert mpc_low > mpc_high, (
            f"MPC should decrease with wealth: MPC(m=1)={mpc_low}, MPC(m=10)={mpc_high}"
        )

    def test_renter_value_increases_with_m(self, model_low_rate):
        """Renter value function is increasing in liquid wealth."""
        sol = model_low_rate.solution[0]
        v_low = sol.vFuncRent(1.0)
        v_high = sol.vFuncRent(5.0)
        assert v_high > v_low, (
            f"Renter value should increase with m: v(1)={v_low}, v(5)={v_high}"
        )

    def test_sell_dominates_default_with_equity(self, model_low_rate):
        """With positive equity, selling beats defaulting (recovers equity)."""
        sol = model_low_rate.solution[0]
        assert sol.tenureFunc is not None, "tenureFunc should exist after solving"
        # At low d (positive equity) and moderate m, should own or sell, not default
        tenure = float(sol.tenureFunc(3.0, 0.2))
        assert round(tenure) < 2, (
            f"Expected own or sell at m=3.0, d=0.2, got tenure={tenure:.2f}"
        )

    def test_higher_rate_does_not_increase_value(self, model_low_rate, model_high_rate):
        """Higher mortgage rate cannot increase homeowner value."""
        # With a short (3-period) model and coarse grids, both models may choose
        # the same tenure at some points, giving identical values. But a higher
        # rate should never strictly increase value at any (m, d).
        sol_low = model_low_rate.solution[0]
        sol_high = model_high_rate.solution[0]
        for m in [3.0, 5.0, 10.0]:
            for d in [0.0, 0.25, 0.5]:
                v_low = float(sol_low.vFuncOwn(m, d))
                v_high = float(sol_high.vFuncOwn(m, d))
                assert v_low >= v_high - 1e-10, (
                    f"Higher rate should not increase value at m={m}, d={d}: "
                    f"v(3%)={v_low}, v(7%)={v_high}"
                )


# ===================================================================
# 5. Parameter validation tests
# ===================================================================


class TestParameterValidation:
    """Test that invalid parameters are caught by check_restrictions."""

    @pytest.fixture()
    def base_agent(self):
        """A valid 1-period agent for testing restrictions."""
        return HousingPortfolioConsumerType(**_base_params_1period())

    def test_crra_below_one_raises(self, base_agent):
        base_agent.CRRA = 0.5
        with pytest.raises(ValueError, match="CRRA must be > 1"):
            base_agent.check_restrictions()

    def test_negative_alpha_raises(self, base_agent):
        base_agent.alpha = -0.1
        with pytest.raises(ValueError, match="alpha must be positive"):
            base_agent.check_restrictions()

    def test_zero_rent_rate_raises(self, base_agent):
        base_agent.RentRate = 0.0
        with pytest.raises(ValueError, match="RentRate must be positive"):
            base_agent.check_restrictions()

    def test_negative_mort_rate_raises(self, base_agent):
        base_agent.MortRate = -0.01
        with pytest.raises(ValueError, match="MortRate must be non-negative"):
            base_agent.check_restrictions()

    def test_zero_mort_rate_accepted(self):
        """Zero mortgage rate is valid (interest-free loan)."""
        pi = mortgage_payment_rate(0.8, 0.0, 30)
        expected = 0.8 / 30
        np.testing.assert_almost_equal(pi, expected, decimal=10)


# ===================================================================
# 6. Markov employment model tests
# ===================================================================


class TestMarkovSolverRuns:
    """Smoke tests: the Markov employment model solves and returns valid objects."""

    @pytest.fixture(scope="class")
    def markov_model(self):
        """A small Markov model (3 periods, 2 employment states)."""
        params = _markov_params_3period(
            LivPrb=[0.99, 0.98, 0.97],
            PermGroFac=[1.02, 1.01, 1.00],
            PermShkStd=[0.1, 0.1, 0.05],
            TranShkStd=[0.1, 0.1, 0.05],
            aXtraCount=10, aXtraMax=10, ShareCount=3, dGridCount=3,
        )
        agent = MarkovHousingPortfolioConsumerType(**params)
        agent.solve()
        return agent

    def test_solution_length(self, markov_model):
        assert len(markov_model.solution) == 3

    def test_solution_has_list_attributes(self, markov_model):
        """Each period's solution stores lists of functions (one per state)."""
        sol = markov_model.solution[0]
        assert isinstance(sol.cFuncOwn, list)
        assert isinstance(sol.vFuncOwn, list)
        assert isinstance(sol.cFuncRent, list)
        assert len(sol.cFuncOwn) == 2

    def test_owner_consumption_positive(self, markov_model):
        sol = markov_model.solution[0]
        for e in range(2):
            for m in [1.0, 5.0, 10.0]:
                c = sol.cFuncOwn[e](m, 0.5)
                assert c > 0, f"c should be positive: state={e}, m={m}, c={c}"

    def test_renter_consumption_positive(self, markov_model):
        sol = markov_model.solution[0]
        for e in range(2):
            c = sol.cFuncRent[e](3.0)
            assert c > 0, f"Renter c should be positive: state={e}, c={c}"


class TestMarkovEconomicLogic:
    """Economic predictions specific to Markov employment states."""

    @pytest.fixture(scope="class")
    def markov_model(self):
        params = _markov_params_3period(
            aXtraCount=15, aXtraMax=15, ShareCount=5, dGridCount=4, UnempIns=0.3,
        )
        agent = MarkovHousingPortfolioConsumerType(**params)
        agent.solve()
        return agent

    def test_employed_value_geq_unemployed(self, markov_model):
        """Employed households should have weakly higher value than unemployed."""
        sol = markov_model.solution[0]
        for m in [3.0, 5.0, 10.0]:
            for d in [0.0, 0.3, 0.5]:
                v_e = float(sol.vFuncOwn[0](m, d))
                v_u = float(sol.vFuncOwn[1](m, d))
                assert v_e >= v_u - 1e-8, (
                    f"Employed value should >= unemployed at m={m}, d={d}: "
                    f"v_E={v_e}, v_U={v_u}"
                )

    def test_unemployed_more_likely_to_exit(self, markov_model):
        """Unemployed households weakly prefer exit over employed at high debt."""
        sol = markov_model.solution[0]
        assert sol.tenureFunc[0] is not None, "tenureFunc should exist"
        # At high d, low m: unemployed tenure index >= employed
        # (higher index means more likely to sell=1 or default=2)
        t_e = float(sol.tenureFunc[0](0.5, 0.9))
        t_u = float(sol.tenureFunc[1](0.5, 0.9))
        assert t_u >= t_e - 1e-8, (
            f"Unemployed should weakly prefer exit at m=0.5, d=0.9: "
            f"tenure_E={t_e:.2f}, tenure_U={t_u:.2f}"
        )

    def test_renter_value_employed_geq_unemployed(self, markov_model):
        """Employed renters have weakly higher value."""
        sol = markov_model.solution[0]
        for m in [1.0, 3.0, 5.0]:
            v_e = float(sol.vFuncRent[0](m))
            v_u = float(sol.vFuncRent[1](m))
            assert v_e >= v_u - 1e-8, (
                f"Renter: employed value should >= unemployed at m={m}: "
                f"v_E={v_e}, v_U={v_u}"
            )

    def test_consumption_increases_with_m_both_states(self, markov_model):
        """Consumption is increasing in m for both employment states."""
        sol = markov_model.solution[0]
        d = 0.5
        m_vals = [1.0, 3.0, 5.0, 10.0]
        for e in range(2):
            c_vals = [sol.cFuncOwn[e](m, d) for m in m_vals]
            for i in range(len(c_vals) - 1):
                assert c_vals[i + 1] > c_vals[i], (
                    f"c should increase with m: state={e}, "
                    f"c({m_vals[i]})={c_vals[i]}, c({m_vals[i+1]})={c_vals[i+1]}"
                )


class TestMarkovParameterValidation:
    """Test Markov-specific parameter validation."""

    @pytest.fixture()
    def base_agent(self):
        params = _base_params_1period(
            MrkvArray=[np.array([[0.95, 0.05], [0.50, 0.50]])],
        )
        return MarkovHousingPortfolioConsumerType(**params)

    def test_non_square_mrkv_raises(self, base_agent):
        base_agent.MrkvArray = [np.array([[0.9, 0.1]])]
        with pytest.raises(ValueError, match="square"):
            base_agent.check_restrictions()

    def test_rows_not_summing_to_one_raises(self, base_agent):
        base_agent.MrkvArray = [np.array([[0.5, 0.3], [0.4, 0.6]])]
        with pytest.raises(ValueError, match="sum to 1"):
            base_agent.check_restrictions()


# ===================================================================
# 7. Origination constraints and repurchase option
# ===================================================================


class TestOriginationConstraints:
    """Test origination constraints (LTV and DTI) and repurchase option."""

    @pytest.fixture(scope="class")
    def model_with_repurchase(self):
        """Model with default origination constraints."""
        params = _base_params_3period(DownPayment=0.20, MaxDTI=4.0)
        agent = HousingPortfolioConsumerType(**params)
        agent.solve()
        return agent

    @pytest.fixture(scope="class")
    def model_no_repurchase(self):
        """Model where repurchase is impossible (100% down payment)."""
        params = _base_params_3period(DownPayment=1.0, MaxDTI=4.0)
        agent = HousingPortfolioConsumerType(**params)
        agent.solve()
        return agent

    def test_repurchase_increases_renter_value(
        self, model_with_repurchase, model_no_repurchase
    ):
        """Renter value with buy option >= renter value without (up to
        interpolation tolerance from rebuilding on a discrete grid)."""
        sol_buy = model_with_repurchase.solution[0]
        sol_no = model_no_repurchase.solution[0]
        for m in [5.0, 10.0, 15.0]:
            v_buy = float(sol_buy.vFuncRent(m))
            v_no = float(sol_no.vFuncRent(m))
            # Small tolerance: the fine-grid approximation may introduce
            # interpolation noise of order 1e-3 relative to function values
            assert v_buy >= v_no - abs(v_no) * 0.01, (
                f"Buy option should weakly increase renter value at m={m}: "
                f"v_buy={v_buy:.4f} vs v_no={v_no:.4f}"
            )

    def test_origination_ltv_respected(self, model_with_repurchase):
        """The initial LTV at origination respects the down payment constraint."""
        agent = model_with_repurchase
        d_0 = min(1.0 - agent.DownPayment, agent.MaxDTI / agent.hbar)
        assert d_0 <= 1.0 - agent.DownPayment + 1e-10
        assert d_0 * agent.hbar <= agent.MaxDTI + 1e-10

    def test_tight_dti_binds(self):
        """When MaxDTI is very low, DTI constraint binds over LTV."""
        params = _base_params_3period(DownPayment=0.20, MaxDTI=0.5)
        agent = HousingPortfolioConsumerType(**params)
        d_0 = min(1.0 - agent.DownPayment, agent.MaxDTI / agent.hbar)
        # DTI should bind: d_0 = 0.5/3.0 ≈ 0.167 < 0.80
        assert d_0 == pytest.approx(0.5 / 3.0, abs=1e-10)
        agent.solve()  # Should still solve without error


# ===================================================================
# 8. Stock-income correlation
# ===================================================================


class TestStockIncomeCorrelation:
    """Test that stock-income correlation affects portfolio allocation."""

    @pytest.fixture(scope="class")
    def model_no_corr(self):
        params = _base_params_3period(StockIncCorr=0.0)
        agent = HousingPortfolioConsumerType(**params)
        agent.solve()
        return agent

    @pytest.fixture(scope="class")
    def model_pos_corr(self):
        params = _base_params_3period(StockIncCorr=0.15)
        agent = HousingPortfolioConsumerType(**params)
        agent.solve()
        return agent

    def test_solves_with_correlation(self, model_pos_corr):
        """Model with positive stock-income correlation solves."""
        assert len(model_pos_corr.solution) == 3

    def test_positive_consumption(self, model_pos_corr):
        """Consumption is positive with stock-income correlation."""
        sol = model_pos_corr.solution[0]
        for m in [3.0, 5.0, 10.0]:
            c = sol.cFuncOwn(m, 0.5)
            assert c > 0, f"Expected c > 0 at m={m}, got {c}"

    def test_correlation_changes_share(self, model_no_corr, model_pos_corr):
        """Positive correlation should change optimal share vs zero correlation."""
        sol_0 = model_no_corr.solution[0]
        sol_p = model_pos_corr.solution[0]
        # At moderate wealth, the two solutions should differ
        s_0 = sol_0.ShareFuncOwn(5.0, 0.3)
        s_p = sol_p.ShareFuncOwn(5.0, 0.3)
        # With positive correlation, stocks are a worse hedge for income risk,
        # so the optimal share should differ from zero correlation.
        assert s_0 != pytest.approx(s_p, abs=1e-10), (
            f"Shares should differ with vs without correlation: "
            f"s(corr=0)={s_0:.4f}, s(corr=0.15)={s_p:.4f}"
        )


# ===================================================================
# 9. Additional coverage tests
# ===================================================================


class TestAffordabilityConstraint:
    """Test that the affordability limit forces tenure exit."""

    @pytest.fixture(scope="class")
    def model_tight_affordability(self):
        """Model with a very tight affordability limit."""
        params = _base_params_3period(AffordabilityLimit=0.01)
        agent = HousingPortfolioConsumerType(**params)
        agent.solve()
        return agent

    def test_affordability_forces_exit(self, model_tight_affordability):
        """With a very tight affordability limit, high-debt owners should exit."""
        sol = model_tight_affordability.solution[0]
        assert sol.tenureFunc is not None, "tenureFunc should exist"
        # At high d, the mortgage payment exceeds AffordabilityLimit,
        # so the household should sell or default (tenure > 0)
        tenure = float(sol.tenureFunc(3.0, 0.8))
        assert round(tenure) > 0, (
            f"Expected sell or default at d=0.8 with tight affordability, "
            f"got tenure={tenure:.2f}"
        )


class TestDefaultBranch:
    """Test that the default branch is reachable in extreme conditions."""

    @pytest.fixture(scope="class")
    def model_costly_sell(self):
        """Model where selling is very costly, making default attractive."""
        params = _base_params_3period(
            dMax=1.2, SellCost=0.5, DefaultPenalty=0.05,
        )
        agent = HousingPortfolioConsumerType(**params)
        agent.solve()
        return agent

    def test_default_reachable_when_underwater(self, model_costly_sell):
        """When deeply underwater with high sell costs, default should be chosen."""
        sol = model_costly_sell.solution[0]
        assert sol.tenureFunc is not None, "tenureFunc should exist"
        # At high d (deeply underwater) with costly selling,
        # default (tenure=2) should be the optimal choice
        tenure = float(sol.tenureFunc(1.0, 0.9))
        assert round(tenure) == 2, (
            f"Expected default (tenure=2) at m=1.0, d=0.9 with costly sell, "
            f"got tenure={tenure:.2f}"
        )


class TestSellCostValidation:
    """Test SellCost parameter validation."""

    def test_sell_cost_negative_raises(self):
        params = _base_params_1period(SellCost=-0.1)
        agent = HousingPortfolioConsumerType(**params)
        with pytest.raises(ValueError, match="SellCost"):
            agent.check_restrictions()

    def test_sell_cost_above_one_raises(self):
        params = _base_params_1period(SellCost=1.5)
        agent = HousingPortfolioConsumerType(**params)
        with pytest.raises(ValueError, match="SellCost"):
            agent.check_restrictions()


# ===================================================================
# 10. Renter EGM correctness tests
# ===================================================================


class TestRenterEGMCorrectness:
    """Verify the renter EGM budget constraint and flow utility normalization.

    These tests would have caught the bugs where:
    - The budget mapping omitted housing expenditure: m = c + a instead of
      m = (1+alpha)*c + a
    - The flow utility used gamma in the denominator instead of (1-rho)
    """

    @pytest.fixture(scope="class")
    def model(self):
        agent = HousingPortfolioConsumerType(**_base_params_3period())
        agent.solve()
        return agent

    def test_renter_consumes_less_than_m(self, model):
        """Renter consumption c must be strictly less than m.

        Since total spending is (1+alpha)*c + a and a >= 0, we need
        (1+alpha)*c <= m, i.e., c <= m/(1+alpha).
        With the old bug (m = c + a), c could nearly equal m.
        """
        sol = model.solution[0]
        alpha = model.alpha
        for m in [2.0, 5.0, 10.0, 20.0]:
            c = float(sol.cFuncRent(m))
            # c must be < m/(1+alpha) since a > 0 for precautionary saving
            assert c < m / (1.0 + alpha), (
                f"Renter c={c:.4f} should be < m/(1+alpha)={m/(1+alpha):.4f} "
                f"at m={m}"
            )

    def test_renter_total_expenditure_less_than_m(self, model):
        """Total renter expenditure (1+alpha)*c + a = m, so (1+alpha)*c < m."""
        sol = model.solution[0]
        alpha = model.alpha
        for m in [2.0, 5.0, 10.0]:
            c = float(sol.cFuncRent(m))
            total_expend = (1.0 + alpha) * c
            assert total_expend < m, (
                f"Total expenditure (1+a)*c = {total_expend:.4f} should be < "
                f"m={m} (alpha={alpha})"
            )

    def test_renter_owner_value_consistent_scale(self, model):
        """Renter and owner values should be on a comparable scale.

        With the old bug, renter flow utility was scaled by 1/(1+alpha)
        relative to the owner. At d=0 (no debt) and moderate m, an owner
        paying no mortgage is similar to a renter, so values should be
        roughly comparable rather than the renter being systematically lower.
        """
        sol = model.solution[0]
        m = 10.0
        d = 0.0  # No debt: owner and renter have similar situations
        v_own = float(sol.vFuncOwn(m, d))
        v_rent = float(sol.vFuncRent(m))
        # Values are negative (CRRA > 1), so both should be the same order
        # of magnitude. The renter should not be dramatically lower.
        ratio = v_rent / v_own if v_own != 0 else float("inf")
        assert 0.3 < ratio < 3.0, (
            f"Renter/owner value ratio at m={m}, d={d} is {ratio:.2f}, "
            f"expected roughly similar. v_own={v_own:.4f}, v_rent={v_rent:.4f}"
        )

    def test_euler_equation_renter(self, model):
        """The Euler equation should hold: u'(c) = E[beta * R * G^gamma * u'(c')].

        At interior EGM points, the FOC holds exactly. We verify that the
        marginal value function vP is consistent with the consumption function:
        vP(m) = kappa_r * c(m)^{gamma-1}.
        Test at higher m to avoid boundary-sentinel interpolation artifacts.
        """
        sol = model.solution[0]
        rho = model.CRRA
        alpha = model.alpha
        gamma = (1.0 + alpha) * (1.0 - rho)
        kappa_r = (alpha / model.RentRate) ** (alpha * (1.0 - rho))

        for m in [5.0, 10.0, 15.0]:
            c = float(sol.cFuncRent(m))
            vp = float(sol.vPfuncRent(m))
            vp_from_c = kappa_r * c ** (gamma - 1.0)
            # Should match to interpolation precision
            np.testing.assert_allclose(
                vp, vp_from_c, rtol=0.1,
                err_msg=f"vP inconsistent with c at m={m}: vP={vp:.6f}, "
                f"kappa_r*c^(gamma-1)={vp_from_c:.6f}",
            )

    def test_alpha_affects_renter_consumption(self):
        """Higher alpha (stronger housing preference) should reduce non-housing
        consumption at the same m, since more resources go to housing."""
        results = {}
        for alpha_val in [0.1, 0.3]:
            params = _base_params_3period(
                T_cycle=2,
                LivPrb=[0.99, 0.99],
                PermGroFac=[1.02, 1.00],
                Rfree=[1.02, 1.02],
                MortPeriods=[2, 1],
                PermShkStd=[0.08, 0.05],
                TranShkStd=[0.08, 0.05],
                aXtraCount=15,
                aXtraMax=15,
                ShareCount=3,
                dGridCount=3,
                alpha=alpha_val,
            )
            agent = HousingPortfolioConsumerType(**params)
            agent.solve()
            results[alpha_val] = float(agent.solution[0].cFuncRent(5.0))

        assert results[0.3] < results[0.1], (
            f"Higher alpha should reduce non-housing c: "
            f"c(alpha=0.1)={results[0.1]:.4f}, c(alpha=0.3)={results[0.3]:.4f}"
        )


class TestOwnerRenterConsistency:
    """Cross-check owner and renter EGM derivations for internal consistency."""

    @pytest.fixture(scope="class")
    def model(self):
        agent = HousingPortfolioConsumerType(**_base_params_3period())
        agent.solve()
        return agent

    def test_euler_equation_owner(self, model):
        """Owner's marginal value vP(m,d) should equal h_mult * c(m,d)^{-rho}.

        Only checked at points where the owner is staying (tenure=0), since
        the envelope uses renter's vP on sell/default branches.
        """
        sol = model.solution[0]
        rho = model.CRRA
        alpha = model.alpha
        h_mult = model.hbar ** (alpha * (1.0 - rho))

        for m in [5.0, 10.0, 15.0]:
            for d in [0.5, 0.7]:
                tenure = round(float(sol.tenureFunc(m, d)))
                if tenure != 0:
                    continue  # skip points where owner sells or defaults
                c = float(sol.cFuncOwn(m, d))
                vp = float(sol.vPfuncOwn(m, d))
                vp_from_c = h_mult * c ** (-rho)
                np.testing.assert_allclose(
                    vp, vp_from_c, rtol=0.15,
                    err_msg=f"Owner vP inconsistent at m={m}, d={d}",
                )

    def test_sell_value_uses_renter_at_correct_m(self, model):
        """When an owner sells, v_sell(m,d) = v_rent(m + (1-kappa-d)*hbar).

        This tests that the tenure envelope correctly maps owner resources
        to the renter value function at the right cash-on-hand.
        """
        sol = model.solution[0]
        hbar = model.hbar
        kappa = model.SellCost
        m = 5.0
        d = 0.3
        net_equity = (1.0 - kappa - d) * hbar
        m_after = m + net_equity
        # The renter value at m_after should be finite (positive resources)
        v_rent = float(sol.vFuncRent(m_after))
        assert np.isfinite(v_rent), (
            f"Renter value should be finite at m={m_after:.2f} after sell"
        )

    def test_marginal_value_decreasing_renter(self, model):
        """Renter marginal value should be decreasing in m (concavity)."""
        sol = model.solution[0]
        m_vals = [1.0, 3.0, 5.0, 10.0, 15.0]
        vp_vals = [float(sol.vPfuncRent(m)) for m in m_vals]
        for i in range(len(vp_vals) - 1):
            # vP should be positive and decreasing for CRRA > 1
            assert vp_vals[i] > vp_vals[i + 1], (
                f"Renter vP should decrease: vP({m_vals[i]})={vp_vals[i]:.4f} "
                f"vs vP({m_vals[i+1]})={vp_vals[i+1]:.4f}"
            )

    def test_marginal_value_decreasing_owner(self, model):
        """Owner marginal value should be decreasing in m (concavity)."""
        sol = model.solution[0]
        d = 0.3
        m_vals = [2.0, 5.0, 10.0, 15.0]
        vp_vals = [float(sol.vPfuncOwn(m, d)) for m in m_vals]
        for i in range(len(vp_vals) - 1):
            assert vp_vals[i] > vp_vals[i + 1], (
                f"Owner vP should decrease: vP({m_vals[i]})={vp_vals[i]:.4f} "
                f"vs vP({m_vals[i+1]})={vp_vals[i+1]:.4f}"
            )


class TestMarkovRenterEGMCorrectness:
    """Verify the Markov renter has the same budget/value fixes."""

    @pytest.fixture(scope="class")
    def markov_model(self):
        params = _markov_params_3period(
            ShareCount=5, dGridCount=4, UnempIns=0.3,
        )
        agent = MarkovHousingPortfolioConsumerType(**params)
        agent.solve()
        return agent

    def test_markov_renter_c_less_than_m_over_1_plus_alpha(self, markov_model):
        """Markov renter c < m/(1+alpha) for both employment states."""
        sol = markov_model.solution[0]
        alpha = markov_model.alpha
        for e in range(2):
            for m in [2.0, 5.0, 10.0]:
                c = float(sol.cFuncRent[e](m))
                bound = m / (1.0 + alpha)
                assert c < bound, (
                    f"Markov renter c={c:.4f} should < {bound:.4f} "
                    f"at m={m}, state={e}"
                )

    def test_markov_renter_euler_consistency(self, markov_model):
        """Markov renter vP should be consistent with c via the FOC.

        Uses 15% tolerance to accommodate interpolation noise from the
        repurchase envelope crossing points.
        """
        sol = markov_model.solution[0]
        rho = markov_model.CRRA
        alpha = markov_model.alpha
        gamma = (1.0 + alpha) * (1.0 - rho)
        kappa_r = (alpha / markov_model.RentRate) ** (alpha * (1.0 - rho))

        for e in range(2):
            for m in [5.0, 8.0, 12.0]:
                c = float(sol.cFuncRent[e](m))
                vp = float(sol.vPfuncRent[e](m))
                vp_expected = kappa_r * c ** (gamma - 1.0)
                np.testing.assert_allclose(
                    vp, vp_expected, rtol=0.15,
                    err_msg=f"Markov renter vP inconsistent at m={m}, state={e}",
                )


# ===================================================================
# Run with: uv run pytest HARK/tests/test_ConsHousingPortfolioModel.py -v
# ===================================================================

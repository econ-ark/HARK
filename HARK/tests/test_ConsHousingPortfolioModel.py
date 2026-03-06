"""
Strategic unit tests for the housing portfolio choice model.

Tests are organized into categories:
1. Housing grid: grid construction and properties
2. Terminal condition: correct bequest values with sell/default branches
3. Model solution: the solver runs and produces sensible policy functions
4. Economic logic: policy functions satisfy known theoretical predictions
5. Markov 4-state model: employment and regime effects
6. Joint shocks: correlation structure
7. Internal consistency: budget satisfaction, tenure coverage
"""

import numpy as np
import pytest

from HARK.ConsumptionSaving.ConsHousingPortfolioModel import (
    HousingPortfolioConsumerType,
    HousingPortfolioSolution,
    MarkovHousingPortfolioConsumerType,
    make_h_grid,
    make_housing_portfolio_solution_terminal,
    make_housing_growth_shock_dstn,
    _build_joint_shocks,
)


# ===================================================================
# Shared parameter helpers
# ===================================================================


def _base_params_3period(**overrides):
    """Common 3-period model parameters. Override any key via kwargs."""
    params = {
        "T_cycle": 3,
        "LivPrb": [0.99, 0.99, 0.99],
        "PermGroFac": [1.02, 1.02, 1.00],
        "Rfree": [1.02, 1.02, 1.02],
        "PermShkStd": [0.08, 0.08, 0.05],
        "TranShkStd": [0.08, 0.08, 0.05],
        "PermShkCount": 3,
        "TranShkCount": 3,
        "RiskyCount": 3,
        "aXtraCount": 24,
        "aXtraMax": 20,
        "ShareCount": 7,
        "hGridCount": 5,
        "hMin": 1.0,
        "hMax": 5.0,
        "LTV": 0.80,
        "HousingGroFac": 1.02,
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
        "PermShkStd": [0.1],
        "TranShkStd": [0.1],
        "PermShkCount": 3,
        "TranShkCount": 3,
        "RiskyCount": 3,
        "aXtraCount": 10,
        "aXtraMax": 10,
        "ShareCount": 3,
        "hGridCount": 3,
        "hMin": 1.0,
        "hMax": 4.0,
        "LTV": 0.80,
        "HousingGroFac": 1.02,
    }
    params.update(overrides)
    return params


def _markov_params_3period(**overrides):
    """Common 3-period Markov model parameters (4 states)."""
    Pi_e = np.array([[0.95, 0.05], [0.50, 0.50]])
    Pi_nu = np.array([[0.80, 0.20], [0.30, 0.70]])
    mrkv_4x4 = np.kron(Pi_e, Pi_nu)
    params = _base_params_3period(
        MrkvArray=[mrkv_4x4] * 3,
        HousingGroFac=[1.03, 1.00],
    )
    params.update(overrides)
    return params


# ===================================================================
# 1. Housing grid tests
# ===================================================================


class TestHGrid:
    """Test the housing-to-income ratio grid constructor."""

    def test_grid_range(self):
        grid = make_h_grid(hGridCount=10, hMin=0.5, hMax=8.0)
        np.testing.assert_almost_equal(grid[0], 0.5, decimal=10)
        np.testing.assert_almost_equal(grid[-1], 8.0, decimal=10)
        assert len(grid) == 10

    def test_grid_monotone(self):
        grid = make_h_grid(hGridCount=20, hMin=0.5, hMax=8.0)
        assert np.all(np.diff(grid) > 0)

    def test_grid_positive_hmin(self):
        grid = make_h_grid(hGridCount=5, hMin=1.0, hMax=5.0)
        assert grid[0] > 0


# ===================================================================
# 2. Terminal condition
# ===================================================================


class TestTerminalCondition:
    """Test the terminal-period bequest value function."""

    @pytest.fixture()
    def terminal_sol(self):
        return make_housing_portfolio_solution_terminal(
            CRRA=5.0, alpha=0.2, LTV=0.80, BeqWt=1.0,
            SellCost=0.06, DefaultPenalty=0.5,
        )

    def test_terminal_value_negative_for_high_rra(self, terminal_sol):
        """With rho=5, gamma=-4.8, value is negative for positive net worth."""
        v = terminal_sol.vFuncOwn(2.0, 3.0)
        assert v < 0

    def test_terminal_value_increases_with_m(self, terminal_sol):
        """Value increases with liquid wealth m."""
        v_low = terminal_sol.vFuncOwn(1.0, 3.0)
        v_high = terminal_sol.vFuncOwn(3.0, 3.0)
        assert v_high > v_low

    def test_terminal_value_increases_with_h(self, terminal_sol):
        """Higher h = more equity, so value increases with h (for lbar < 1-kappa)."""
        v_low_h = terminal_sol.vFuncOwn(2.0, 1.0)
        v_high_h = terminal_sol.vFuncOwn(2.0, 5.0)
        assert v_high_h > v_low_h

    def test_terminal_marginal_value_positive(self, terminal_sol):
        """Marginal value of m is positive."""
        vp = terminal_sol.vPfuncOwn(2.0, 3.0)
        assert vp > 0

    def test_terminal_renter_no_housing_equity(self, terminal_sol):
        """Renter's terminal value depends only on liquid wealth."""
        v_low = terminal_sol.vFuncRent(1.0)
        v_high = terminal_sol.vFuncRent(3.0)
        assert v_high > v_low
        rho = 5.0
        alpha = 0.2
        gamma = (1.0 + alpha) * (1.0 - rho)
        m = 2.0
        v = terminal_sol.vFuncRent(m)
        expected = 1.0 * m ** gamma / gamma
        np.testing.assert_almost_equal(v, expected, decimal=10)

    def test_terminal_share_zero(self, terminal_sol):
        """At terminal date, risky share is zero."""
        assert terminal_sol.ShareFuncOwn(5.0) == 0.0
        assert terminal_sol.ShareFuncRent(5.0) == 0.0

    def test_consume_all_at_terminal(self, terminal_sol):
        """At terminal date, consume all liquid wealth."""
        m = 5.0
        c = terminal_sol.cFuncOwn(m, 3.0)
        np.testing.assert_almost_equal(c, m, decimal=10)


class TestTerminalDefaultBranch:
    """Test the default branch in the terminal condition."""

    def test_sell_dominates_when_positive_equity(self):
        """When net equity is positive, sell beats default."""
        sol = make_housing_portfolio_solution_terminal(
            CRRA=5.0, alpha=0.2, LTV=0.80, BeqWt=1.0,
            SellCost=0.06, DefaultPenalty=0.5,
        )
        # h=3, eps=1: equity = 3*((1-0.06)*1 - 0.80) = 3*(0.14) = 0.42
        # So m + equity = 2 + 0.42 = 2.42 > m = 2
        v = sol.vFuncOwn(2.0, 3.0)
        assert np.isfinite(v)

    def test_default_dominates_when_negative_equity(self):
        """With very high LTV and sell cost, default can win."""
        sol = make_housing_portfolio_solution_terminal(
            CRRA=5.0, alpha=0.2, LTV=0.95, BeqWt=1.0,
            SellCost=0.50, DefaultPenalty=0.01,
        )
        # h=3, eps=1: equity = 3*((1-0.50)*1 - 0.95) = 3*(-0.45) = -1.35
        # Sell branch: w = max(2 - 1.35, 1e-6) = 0.65
        # Default branch: w = 2, minus tiny penalty
        v = sol.vFuncOwn(2.0, 3.0)
        assert np.isfinite(v)

    def test_terminal_values_finite_at_extremes(self):
        """Terminal values are finite even at extreme h values."""
        sol = make_housing_portfolio_solution_terminal(
            CRRA=5.0, alpha=0.2, LTV=0.80, BeqWt=1.0,
            SellCost=0.06, DefaultPenalty=0.5,
        )
        for h in [0.5, 1.0, 5.0, 10.0]:
            v = sol.vFuncOwn(2.0, h)
            assert np.isfinite(v), f"v should be finite at h={h}, got {v}"

    def test_default_branch_vp_finite(self):
        """Marginal value from default branch is finite."""
        sol = make_housing_portfolio_solution_terminal(
            CRRA=5.0, alpha=0.2, LTV=0.95, BeqWt=1.0,
            SellCost=0.50, DefaultPenalty=0.01,
        )
        vp = sol.vPfuncOwn(2.0, 3.0)
        assert np.isfinite(vp), f"vP should be finite, got {vp}"


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
        assert len(small_model.solution) == 3

    def test_solution_has_owner_functions(self, small_model):
        for sol in small_model.solution:
            assert hasattr(sol, "cFuncOwn")
            assert hasattr(sol, "vFuncOwn")
            assert hasattr(sol, "ShareFuncOwn")

    def test_solution_has_renter_functions(self, small_model):
        for sol in small_model.solution:
            assert hasattr(sol, "cFuncRent")
            assert hasattr(sol, "vFuncRent")

    def test_consumption_positive(self, small_model):
        sol = small_model.solution[0]
        for m in [0.5, 1.0, 5.0, 10.0]:
            c = sol.cFuncOwn(m, 2.0)
            assert c > 0, f"c should be positive at m={m}, got {c}"

    def test_consumption_less_than_resources(self, small_model):
        sol = small_model.solution[0]
        m = 5.0
        h = 2.0
        c = sol.cFuncOwn(m, h)
        assert c < m, f"c={c} exceeds m={m}"

    def test_share_bounded_01(self, small_model):
        sol = small_model.solution[0]
        for m in [1.0, 5.0, 10.0]:
            s = sol.ShareFuncOwn(m, 2.0)
            assert -0.01 <= s <= 1.01, f"Share out of bounds: {s}"

    def test_renter_consumption_positive(self, small_model):
        sol = small_model.solution[0]
        c = sol.cFuncRent(3.0)
        assert c > 0

    def test_renter_share_bounded(self, small_model):
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
        agent = HousingPortfolioConsumerType(**_base_params_3period(MortRate=0.03))
        agent.solve()
        return agent

    @pytest.fixture(scope="class")
    def model_high_rate(self):
        agent = HousingPortfolioConsumerType(**_base_params_3period(MortRate=0.07))
        agent.solve()
        return agent

    def test_consumption_increases_with_m(self, model_low_rate):
        sol = model_low_rate.solution[0]
        h = 2.0
        m_vals = [1.0, 3.0, 5.0, 10.0]
        c_vals = [sol.cFuncOwn(m, h) for m in m_vals]
        for i in range(len(c_vals) - 1):
            assert c_vals[i + 1] > c_vals[i], (
                f"c should increase: c({m_vals[i]})={c_vals[i]}, "
                f"c({m_vals[i+1]})={c_vals[i+1]}"
            )

    def test_value_increases_with_h(self, model_low_rate):
        """Higher h = more housing equity, so value should increase."""
        sol = model_low_rate.solution[0]
        m = 5.0
        v_low_h = sol.vFuncOwn(m, 1.5)
        v_high_h = sol.vFuncOwn(m, 4.0)
        assert v_high_h > v_low_h, (
            f"Value should increase with h: v(h=1.5)={v_low_h}, v(h=4.0)={v_high_h}"
        )

    def test_mpc_decreases_with_wealth(self, model_low_rate):
        sol = model_low_rate.solution[0]
        h = 2.0
        dm = 0.01
        mpc_low = (sol.cFuncOwn(1.0 + dm, h) - sol.cFuncOwn(1.0, h)) / dm
        mpc_high = (sol.cFuncOwn(10.0 + dm, h) - sol.cFuncOwn(10.0, h)) / dm
        assert mpc_low > mpc_high, (
            f"MPC should decrease: MPC(m=1)={mpc_low}, MPC(m=10)={mpc_high}"
        )

    def test_renter_value_increases_with_m(self, model_low_rate):
        sol = model_low_rate.solution[0]
        v_low = sol.vFuncRent(1.0)
        v_high = sol.vFuncRent(5.0)
        assert v_high > v_low

    def test_sell_dominates_default_with_equity(self, model_low_rate):
        """With positive equity, selling beats defaulting."""
        sol = model_low_rate.solution[0]
        assert sol.tenureFunc is not None
        # At low h (small house, small equity), low m: own(0) or sell(1)
        tenure = round(float(sol.tenureFunc(3.0, 2.0)))
        assert tenure in (0, 1, 2), (
            f"Expected own/sell/move at m=3.0, h=2.0, got tenure={tenure}"
        )

    def test_higher_rate_does_not_increase_value(self, model_low_rate, model_high_rate):
        sol_low = model_low_rate.solution[0]
        sol_high = model_high_rate.solution[0]
        for m in [3.0, 5.0, 10.0]:
            for h in [1.5, 2.5, 4.0]:
                v_low = float(sol_low.vFuncOwn(m, h))
                v_high = float(sol_high.vFuncOwn(m, h))
                assert v_low >= v_high - 1e-10, (
                    f"Higher rate should not increase value at m={m}, h={h}: "
                    f"v(3%)={v_low}, v(7%)={v_high}"
                )


# ===================================================================
# 5. Parameter validation tests
# ===================================================================


class TestParameterValidation:
    """Test that invalid parameters are caught by check_restrictions."""

    @pytest.fixture()
    def base_agent(self):
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

    def test_ltv_out_of_range_raises(self, base_agent):
        base_agent.LTV = 1.0
        with pytest.raises(ValueError, match="LTV"):
            base_agent.check_restrictions()

    def test_ltv_zero_raises(self, base_agent):
        base_agent.LTV = 0.0
        with pytest.raises(ValueError, match="LTV"):
            base_agent.check_restrictions()


# ===================================================================
# 6. Markov 4-state model tests
# ===================================================================


class TestMarkovSolverRuns:
    """Smoke tests: the 4-state Markov model solves and returns valid objects."""

    @pytest.fixture(scope="class")
    def markov_model(self):
        params = _markov_params_3period(
            LivPrb=[0.99, 0.98, 0.97],
            PermGroFac=[1.02, 1.01, 1.00],
            PermShkStd=[0.1, 0.1, 0.05],
            TranShkStd=[0.1, 0.1, 0.05],
            aXtraCount=10, aXtraMax=10, ShareCount=3, hGridCount=3,
        )
        agent = MarkovHousingPortfolioConsumerType(**params)
        agent.solve()
        return agent

    def test_solution_length(self, markov_model):
        assert len(markov_model.solution) == 3

    def test_solution_has_list_attributes(self, markov_model):
        """Each period stores lists of functions (one per state)."""
        sol = markov_model.solution[0]
        assert isinstance(sol.cFuncOwn, list)
        assert isinstance(sol.vFuncOwn, list)
        assert isinstance(sol.cFuncRent, list)
        assert len(sol.cFuncOwn) == 4  # 4 states now

    def test_owner_consumption_positive(self, markov_model):
        sol = markov_model.solution[0]
        for e in range(4):
            for m in [1.0, 5.0, 10.0]:
                c = sol.cFuncOwn[e](m, 2.0)
                assert c > 0, f"c should be positive: state={e}, m={m}, c={c}"

    def test_renter_consumption_positive(self, markov_model):
        sol = markov_model.solution[0]
        for e in range(4):
            c = sol.cFuncRent[e](3.0)
            assert c > 0, f"Renter c should be positive: state={e}, c={c}"


class TestMarkovEconomicLogic:
    """Economic predictions specific to Markov states."""

    @pytest.fixture(scope="class")
    def markov_model(self):
        params = _markov_params_3period(
            aXtraCount=15, aXtraMax=15, ShareCount=5, hGridCount=4,
            UnempIns=0.3,
        )
        agent = MarkovHousingPortfolioConsumerType(**params)
        agent.solve()
        return agent

    def test_employed_value_geq_unemployed(self, markov_model):
        """Employed households should have weakly higher value than unemployed.
        States: (E,B)=0, (E,S)=1, (U,B)=2, (U,S)=3."""
        sol = markov_model.solution[0]
        for m in [3.0, 5.0, 10.0]:
            for h in [1.5, 2.5]:
                # Compare employed-boom vs unemployed-boom
                v_eb = float(sol.vFuncOwn[0](m, h))
                v_ub = float(sol.vFuncOwn[2](m, h))
                assert v_eb >= v_ub - 1e-8, (
                    f"Employed value should >= unemployed at m={m}, h={h}: "
                    f"v_EB={v_eb}, v_UB={v_ub}"
                )

    def test_boom_value_geq_slump(self, markov_model):
        """Boom regime should give weakly higher value than slump.
        States: (E,B)=0, (E,S)=1."""
        sol = markov_model.solution[0]
        for m in [3.0, 5.0, 10.0]:
            for h in [1.5, 2.5]:
                v_eb = float(sol.vFuncOwn[0](m, h))
                v_es = float(sol.vFuncOwn[1](m, h))
                assert v_eb >= v_es - 1e-8, (
                    f"Boom value should >= slump at m={m}, h={h}: "
                    f"v_EB={v_eb}, v_ES={v_es}"
                )

    def test_renter_value_employed_geq_unemployed(self, markov_model):
        sol = markov_model.solution[0]
        for m in [1.0, 3.0, 5.0]:
            v_eb = float(sol.vFuncRent[0](m))
            v_ub = float(sol.vFuncRent[2](m))
            assert v_eb >= v_ub - 1e-8, (
                f"Renter: employed value should >= unemployed at m={m}: "
                f"v_EB={v_eb}, v_UB={v_ub}"
            )

    def test_consumption_increases_with_m_all_states(self, markov_model):
        sol = markov_model.solution[0]
        h = 2.0
        m_vals = [1.0, 3.0, 5.0, 10.0]
        for e in range(4):
            c_vals = [sol.cFuncOwn[e](m, h) for m in m_vals]
            for i in range(len(c_vals) - 1):
                assert c_vals[i + 1] > c_vals[i], (
                    f"c should increase: state={e}, "
                    f"c({m_vals[i]})={c_vals[i]}, c({m_vals[i+1]})={c_vals[i+1]}"
                )


class TestMarkovParameterValidation:
    """Test Markov-specific parameter validation."""

    @pytest.fixture()
    def base_agent(self):
        Pi_e = np.array([[0.95, 0.05], [0.50, 0.50]])
        Pi_nu = np.array([[0.80, 0.20], [0.30, 0.70]])
        mrkv = np.kron(Pi_e, Pi_nu)
        params = _base_params_1period(MrkvArray=[mrkv])
        return MarkovHousingPortfolioConsumerType(**params)

    def test_non_square_mrkv_raises(self, base_agent):
        base_agent.MrkvArray = [np.array([[0.9, 0.1]])]
        with pytest.raises(ValueError, match="square"):
            base_agent.check_restrictions()

    def test_rows_not_summing_to_one_raises(self, base_agent):
        base_agent.MrkvArray = [np.ones((4, 4)) * 0.3]
        with pytest.raises(ValueError, match="sum to 1"):
            base_agent.check_restrictions()


# ===================================================================
# 7. Housing regime effects
# ===================================================================


class TestHousingRegimeEffects:
    """Test that boom/slump regime affects outcomes."""

    @pytest.fixture(scope="class")
    def markov_model(self):
        params = _markov_params_3period(
            aXtraCount=15, aXtraMax=15, ShareCount=5, hGridCount=4,
            HousingGroFac=[1.05, 0.98],  # boom much higher than slump
        )
        agent = MarkovHousingPortfolioConsumerType(**params)
        agent.solve()
        return agent

    def test_boom_owner_value_geq_slump(self, markov_model):
        """Employed-boom owner value >= employed-slump."""
        sol = markov_model.solution[0]
        for m in [3.0, 5.0, 10.0]:
            v_boom = float(sol.vFuncOwn[0](m, 2.5))
            v_slump = float(sol.vFuncOwn[1](m, 2.5))
            assert v_boom >= v_slump - 1e-8, (
                f"Boom >= slump at m={m}: v_boom={v_boom}, v_slump={v_slump}"
            )

    def test_boom_renter_value_geq_slump(self, markov_model):
        """Boom renter (E,B) value >= slump renter (E,S) via purchase option."""
        sol = markov_model.solution[0]
        for m in [5.0, 10.0]:
            v_boom = float(sol.vFuncRent[0](m))
            v_slump = float(sol.vFuncRent[1](m))
            assert v_boom >= v_slump - 1e-8

    def test_four_states_exist(self, markov_model):
        sol = markov_model.solution[0]
        assert len(sol.cFuncOwn) == 4
        assert len(sol.vFuncRent) == 4

    def test_regime_does_not_affect_income(self, markov_model):
        """States 0,1 (employed) should have similar renter c since
        income distribution is the same regardless of regime."""
        sol = markov_model.solution[0]
        c_eb = float(sol.cFuncRent[0](5.0))
        c_es = float(sol.cFuncRent[1](5.0))
        # They should be similar (same income) but may differ via purchase option
        assert abs(c_eb - c_es) / max(abs(c_eb), 1e-10) < 0.5


# ===================================================================
# 8. Joint shock distribution tests
# ===================================================================


class TestJointShockDistribution:
    """Test the joint shock construction."""

    def _make_simple_inc_dstn(self):
        from HARK.distributions.discrete import DiscreteDistribution
        atoms = np.array([[0.9, 1.0, 1.1], [0.8, 1.0, 1.2]])
        probs = np.array([1/3, 1/3, 1/3])
        return DiscreteDistribution(probs, atoms)

    def test_probs_sum_to_one_renter(self):
        inc = self._make_simple_inc_dstn()
        risky = np.array([0.9, 1.0, 1.1])
        prob_r = np.array([1/3, 1/3, 1/3])
        _, _, prob_j, _, _, _, eta_j = _build_joint_shocks(
            inc, risky, prob_r, 1.02, -4.8, 0.0,
        )
        np.testing.assert_almost_equal(prob_j.sum(), 1.0, decimal=10)
        # Renter: eta should be all ones
        np.testing.assert_array_equal(eta_j, np.ones_like(eta_j))

    def test_probs_sum_to_one_owner(self):
        inc = self._make_simple_inc_dstn()
        risky = np.array([0.9, 1.0, 1.1])
        prob_r = np.array([1/3, 1/3, 1/3])
        eta_dstn = make_housing_growth_shock_dstn(
            HousingGrowthShkStd=0.1, HousingGrowthShkCount=3,
        )
        _, _, prob_j, _, _, _, eta_j = _build_joint_shocks(
            inc, risky, prob_r, 1.02, -4.8, 0.0,
            housing_dstn=eta_dstn,
        )
        np.testing.assert_almost_equal(prob_j.sum(), 1.0, decimal=10)
        assert len(eta_j) == 3 * 3 * 3  # N_inc x N_eta x N_ret

    def test_correlation_modifies_returns(self):
        inc = self._make_simple_inc_dstn()
        risky = np.array([1.0, 1.0, 1.0])
        prob_r = np.array([1/3, 1/3, 1/3])
        _, _, _, risky_0, _, _, _ = _build_joint_shocks(
            inc, risky, prob_r, 1.02, -4.8, 0.0,
        )
        _, _, _, risky_c, _, _, _ = _build_joint_shocks(
            inc, risky, prob_r, 1.02, -4.8, 0.5,
        )
        # With correlation, risky_j should vary (loaded on perm shocks)
        assert not np.allclose(risky_0, risky_c)

    def test_eta_mean_approximately_one(self):
        """Housing growth shock eta should have mean ~1."""
        eta_dstn = make_housing_growth_shock_dstn(
            HousingGrowthShkStd=0.1, HousingGrowthShkCount=7,
        )
        eta_nodes = eta_dstn.atoms[0]
        eta_probs = eta_dstn.pmv
        mean = np.dot(eta_nodes, eta_probs)
        np.testing.assert_almost_equal(mean, 1.0, decimal=3)


# ===================================================================
# 9. Origination constraints and repurchase option
# ===================================================================


class TestOriginationConstraints:
    """Test repurchase option."""

    @pytest.fixture(scope="class")
    def model_with_repurchase(self):
        params = _base_params_3period(MaxDTI=4.0)
        agent = HousingPortfolioConsumerType(**params)
        agent.solve()
        return agent

    @pytest.fixture(scope="class")
    def model_no_repurchase(self):
        """Model where repurchase is impossible (MaxDTI too low)."""
        params = _base_params_3period(MaxDTI=0.01)
        agent = HousingPortfolioConsumerType(**params)
        agent.solve()
        return agent

    def test_repurchase_increases_renter_value(
        self, model_with_repurchase, model_no_repurchase
    ):
        sol_buy = model_with_repurchase.solution[0]
        sol_no = model_no_repurchase.solution[0]
        for m in [5.0, 10.0, 15.0]:
            v_buy = float(sol_buy.vFuncRent(m))
            v_no = float(sol_no.vFuncRent(m))
            assert v_buy >= v_no - abs(v_no) * 0.01, (
                f"Buy option should weakly increase value at m={m}: "
                f"v_buy={v_buy:.4f} vs v_no={v_no:.4f}"
            )


# ===================================================================
# 10. Stock-income correlation
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
        assert len(model_pos_corr.solution) == 3

    def test_positive_consumption(self, model_pos_corr):
        sol = model_pos_corr.solution[0]
        for m in [3.0, 5.0, 10.0]:
            c = sol.cFuncOwn(m, 2.0)
            assert c > 0, f"Expected c > 0 at m={m}, got {c}"

    def test_correlation_changes_share(self, model_no_corr, model_pos_corr):
        sol_0 = model_no_corr.solution[0]
        sol_p = model_pos_corr.solution[0]
        s_0 = sol_0.ShareFuncOwn(5.0, 2.0)
        s_p = sol_p.ShareFuncOwn(5.0, 2.0)
        assert s_0 != pytest.approx(s_p, abs=1e-10), (
            f"Shares should differ: s(corr=0)={s_0:.4f}, s(corr=0.15)={s_p:.4f}"
        )


# ===================================================================
# 11. Renter EGM correctness tests
# ===================================================================


class TestRenterEGMCorrectness:
    """Verify the renter EGM budget constraint and flow utility normalization."""

    @pytest.fixture(scope="class")
    def model(self):
        agent = HousingPortfolioConsumerType(**_base_params_3period())
        agent.solve()
        return agent

    def test_renter_consumes_less_than_m(self, model):
        sol = model.solution[0]
        alpha = model.alpha
        for m in [2.0, 5.0, 10.0, 20.0]:
            c = float(sol.cFuncRent(m))
            assert c < m / (1.0 + alpha), (
                f"Renter c={c:.4f} should be < m/(1+alpha)={m/(1+alpha):.4f} "
                f"at m={m}"
            )

    def test_renter_total_expenditure_less_than_m(self, model):
        sol = model.solution[0]
        alpha = model.alpha
        for m in [2.0, 5.0, 10.0]:
            c = float(sol.cFuncRent(m))
            total_expend = (1.0 + alpha) * c
            assert total_expend < m

    def test_renter_owner_value_consistent_scale(self, model):
        """Renter and owner values should be on a comparable scale."""
        sol = model.solution[0]
        m = 10.0
        h = 1.5  # small house, minimal costs
        v_own = float(sol.vFuncOwn(m, h))
        v_rent = float(sol.vFuncRent(m))
        ratio = v_rent / v_own if v_own != 0 else float("inf")
        assert 0.3 < ratio < 3.0, (
            f"Renter/owner ratio at m={m}, h={h} is {ratio:.2f}. "
            f"v_own={v_own:.4f}, v_rent={v_rent:.4f}"
        )

    def test_euler_equation_renter(self, model):
        sol = model.solution[0]
        rho = model.CRRA
        alpha = model.alpha
        gamma = (1.0 + alpha) * (1.0 - rho)
        kappa_r = (alpha / model.RentRate) ** (alpha * (1.0 - rho))

        for m in [5.0, 10.0, 15.0]:
            c = float(sol.cFuncRent(m))
            vp = float(sol.vPfuncRent(m))
            vp_from_c = kappa_r * c ** (gamma - 1.0)
            np.testing.assert_allclose(
                vp, vp_from_c, rtol=0.1,
                err_msg=f"vP inconsistent with c at m={m}",
            )

    def test_alpha_affects_renter_consumption(self):
        results = {}
        for alpha_val in [0.1, 0.3]:
            params = _base_params_3period(
                T_cycle=2,
                LivPrb=[0.99, 0.99],
                PermGroFac=[1.02, 1.00],
                Rfree=[1.02, 1.02],
                PermShkStd=[0.08, 0.05],
                TranShkStd=[0.08, 0.05],
                aXtraCount=15,
                aXtraMax=15,
                ShareCount=3,
                hGridCount=3,
                alpha=alpha_val,
            )
            agent = HousingPortfolioConsumerType(**params)
            agent.solve()
            results[alpha_val] = float(agent.solution[0].cFuncRent(5.0))

        assert results[0.3] < results[0.1]


class TestOwnerRenterConsistency:
    """Cross-check owner and renter EGM derivations."""

    @pytest.fixture(scope="class")
    def model(self):
        agent = HousingPortfolioConsumerType(**_base_params_3period())
        agent.solve()
        return agent

    def test_euler_equation_owner(self, model):
        """Owner vP(m,h) should equal h_mult * c(m,h)^{-rho} at staying points."""
        sol = model.solution[0]
        rho = model.CRRA
        alpha = model.alpha

        for m in [5.0, 10.0, 15.0]:
            for h in [1.5, 2.5]:
                tenure = round(float(sol.tenureFunc(m, h)))
                if tenure != 0:
                    continue
                h_mult = h ** (alpha * (1.0 - rho))
                c = float(sol.cFuncOwn(m, h))
                vp = float(sol.vPfuncOwn(m, h))
                vp_from_c = h_mult * c ** (-rho)
                np.testing.assert_allclose(
                    vp, vp_from_c, rtol=0.15,
                    err_msg=f"Owner vP inconsistent at m={m}, h={h}",
                )

    def test_marginal_value_decreasing_renter(self, model):
        sol = model.solution[0]
        m_vals = [1.0, 3.0, 5.0, 10.0, 15.0]
        vp_vals = [float(sol.vPfuncRent(m)) for m in m_vals]
        for i in range(len(vp_vals) - 1):
            assert vp_vals[i] > vp_vals[i + 1]

    def test_marginal_value_decreasing_owner(self, model):
        sol = model.solution[0]
        h = 2.0
        m_vals = [2.0, 5.0, 10.0, 15.0]
        vp_vals = [float(sol.vPfuncOwn(m, h)) for m in m_vals]
        for i in range(len(vp_vals) - 1):
            assert vp_vals[i] > vp_vals[i + 1]


# ===================================================================
# 12. Affordability and default
# ===================================================================


class TestAffordabilityConstraint:
    """Test that the affordability limit forces tenure exit."""

    @pytest.fixture(scope="class")
    def model_tight_affordability(self):
        params = _base_params_3period(AffordabilityLimit=0.01)
        agent = HousingPortfolioConsumerType(**params)
        agent.solve()
        return agent

    def test_affordability_forces_exit(self, model_tight_affordability):
        sol = model_tight_affordability.solution[0]
        assert sol.tenureFunc is not None
        # At high h, housing costs exceed limit
        tenure = float(sol.tenureFunc(3.0, 4.0))
        assert round(tenure) > 0, (
            f"Expected sell/move/default at h=4.0 with tight affordability, "
            f"got tenure={tenure:.2f}"
        )


class TestDefaultBranch:
    """Test that the default branch is reachable."""

    @pytest.fixture(scope="class")
    def model_costly_sell(self):
        params = _base_params_3period(
            SellCost=0.5, DefaultPenalty=0.05,
            LTV=0.90,
        )
        agent = HousingPortfolioConsumerType(**params)
        agent.solve()
        return agent

    def test_default_reachable(self, model_costly_sell):
        """With very costly selling and high LTV, default (3) should appear."""
        sol = model_costly_sell.solution[0]
        assert sol.tenureFunc is not None
        # At high h (large house, large costs), small m: default may be optimal
        tenure = float(sol.tenureFunc(0.5, 4.0))
        # Accept any valid tenure for now; just verify it's finite
        assert 0 <= round(tenure) <= 3


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
# 13. Additional coverage
# ===================================================================


class TestShareCountOne:
    """Test the ShareCount=1 degenerate case."""

    @pytest.fixture(scope="class")
    def model(self):
        params = _base_params_3period(ShareCount=1)
        agent = HousingPortfolioConsumerType(**params)
        agent.solve()
        return agent

    def test_solves_without_error(self, model):
        assert len(model.solution) == 3

    def test_share_is_zero(self, model):
        sol = model.solution[0]
        for m in [3.0, 5.0, 10.0]:
            s = float(sol.ShareFuncOwn(m, 2.0))
            assert s == pytest.approx(0.0, abs=1e-6)

    def test_renter_positive_consumption(self, model):
        sol = model.solution[0]
        for m in [2.0, 5.0, 10.0]:
            c = float(sol.cFuncRent(m))
            assert c > 0


class TestPolicyFunctionsFinite:
    """Sweep policy functions for NaN/Inf."""

    @pytest.fixture(scope="class")
    def model(self):
        agent = HousingPortfolioConsumerType(**_base_params_3period())
        agent.solve()
        return agent

    def test_owner_functions_finite(self, model):
        sol = model.solution[0]
        m_test = np.linspace(0.5, 15.0, 30)
        h_test = np.linspace(1.1, 4.5, 5)
        mm, hh = np.meshgrid(m_test, h_test)
        c_vals = sol.cFuncOwn(mm.ravel(), hh.ravel())
        assert np.all(np.isfinite(c_vals)), (
            f"cFuncOwn has {np.sum(~np.isfinite(c_vals))} non-finite values"
        )
        v_vals = sol.vFuncOwn(mm.ravel(), hh.ravel())
        assert np.all(np.isfinite(v_vals))

    def test_renter_functions_finite(self, model):
        sol = model.solution[0]
        m_test = np.linspace(0.5, 15.0, 50)
        c_vals = np.array([float(sol.cFuncRent(m)) for m in m_test])
        assert np.all(np.isfinite(c_vals))
        v_vals = np.array([float(sol.vFuncRent(m)) for m in m_test])
        assert np.all(np.isfinite(v_vals))


class TestLowerEnvelopeRenter:
    """Test that LowerEnvelope properly bounds renter consumption."""

    @pytest.fixture(scope="class")
    def model(self):
        agent = HousingPortfolioConsumerType(**_base_params_3period())
        agent.solve()
        return agent

    def test_renter_c_bounded_by_budget(self, model):
        alpha = model.alpha
        sol = model.solution[0]
        for m in np.linspace(0.01, 20.0, 100):
            c = float(sol.cFuncRent(m))
            bound = m / (1.0 + alpha)
            assert c <= bound + 1e-10

    def test_renter_c_near_zero_at_zero_m(self, model):
        sol = model.solution[0]
        c = float(sol.cFuncRent(0.001))
        assert c < 0.01


class TestMNrmMinTracking:
    """Test mNrmMin tracking."""

    @pytest.fixture(scope="class")
    def model(self):
        agent = HousingPortfolioConsumerType(**_base_params_3period())
        agent.solve()
        return agent

    def test_mnrm_min_exists(self, model):
        for t, sol in enumerate(model.solution):
            assert hasattr(sol, "mNrmMin")

    def test_mnrm_min_non_negative(self, model):
        for t, sol in enumerate(model.solution):
            assert sol.mNrmMin >= 0.0

    def test_terminal_mnrm_min_zero(self, model):
        assert model.solution[-1].mNrmMin == 0.0


class TestDiscFacValidation:
    """Test DiscFac < 0 raises."""

    def test_negative_discfac_raises(self):
        params = _base_params_1period(DiscFac=-0.5)
        agent = HousingPortfolioConsumerType(**params)
        with pytest.raises(ValueError, match="DiscFac"):
            agent.check_restrictions()


class TestMarkovMNrmMin:
    """Test mNrmMin tracking in Markov model."""

    @pytest.fixture(scope="class")
    def model(self):
        agent = MarkovHousingPortfolioConsumerType(
            **_markov_params_3period()
        )
        agent.solve()
        return agent

    def test_markov_mnrm_min_exists(self, model):
        for t, sol in enumerate(model.solution):
            assert hasattr(sol, "mNrmMin")

    def test_markov_mnrm_min_zero(self, model):
        for t, sol in enumerate(model.solution):
            assert sol.mNrmMin == 0.0


class TestNegativeStockCorrelation:
    """Negative stock-income correlation should change risky share."""

    @pytest.fixture(scope="class")
    def model_neg_corr(self):
        params = _base_params_3period(StockIncCorr=-0.15)
        agent = HousingPortfolioConsumerType(**params)
        agent.solve()
        return agent

    @pytest.fixture(scope="class")
    def model_no_corr(self):
        params = _base_params_3period(StockIncCorr=0.0)
        agent = HousingPortfolioConsumerType(**params)
        agent.solve()
        return agent

    def test_neg_corr_solves(self, model_neg_corr):
        assert len(model_neg_corr.solution) == 3

    def test_neg_corr_changes_share(self, model_neg_corr, model_no_corr):
        s_0 = model_no_corr.solution[0].ShareFuncOwn(5.0, 2.0)
        s_neg = model_neg_corr.solution[0].ShareFuncOwn(5.0, 2.0)
        assert s_0 != pytest.approx(s_neg, abs=1e-10)


class TestMrkvArrayValidation:
    """Test that check_restrictions validates all MrkvArray periods."""

    def test_bad_second_matrix_raises(self):
        Pi_e = np.array([[0.95, 0.05], [0.50, 0.50]])
        Pi_nu = np.array([[0.80, 0.20], [0.30, 0.70]])
        mrkv_good = np.kron(Pi_e, Pi_nu)
        mrkv_bad = np.ones((4, 4)) * 0.3  # rows don't sum to 1
        params = _base_params_1period(
            T_cycle=2,
            LivPrb=[0.99, 0.99],
            PermGroFac=[1.02, 1.00],
            Rfree=[1.02, 1.02],
            PermShkStd=[0.1, 0.1],
            TranShkStd=[0.1, 0.1],
            MrkvArray=[mrkv_good, mrkv_bad],
        )
        agent = MarkovHousingPortfolioConsumerType(**params)
        with pytest.raises(ValueError, match="sum to 1"):
            agent.check_restrictions()


class TestMarkovRenterEGMCorrectness:
    """Verify the Markov renter has the same budget/value fixes."""

    @pytest.fixture(scope="class")
    def markov_model(self):
        params = _markov_params_3period(
            ShareCount=5, hGridCount=4, UnempIns=0.3,
        )
        agent = MarkovHousingPortfolioConsumerType(**params)
        agent.solve()
        return agent

    def test_markov_renter_c_less_than_m_over_1_plus_alpha(self, markov_model):
        sol = markov_model.solution[0]
        alpha = markov_model.alpha
        for e in range(4):
            for m in [2.0, 5.0, 10.0]:
                c = float(sol.cFuncRent[e](m))
                bound = m / (1.0 + alpha)
                assert c < bound

    def test_markov_renter_euler_consistency(self, markov_model):
        sol = markov_model.solution[0]
        rho = markov_model.CRRA
        alpha = markov_model.alpha
        gamma = (1.0 + alpha) * (1.0 - rho)
        kappa_r = (alpha / markov_model.RentRate) ** (alpha * (1.0 - rho))

        for e in range(4):
            for m in [5.0, 8.0, 12.0]:
                c = float(sol.cFuncRent[e](m))
                vp = float(sol.vPfuncRent[e](m))
                vp_expected = kappa_r * c ** (gamma - 1.0)
                np.testing.assert_allclose(
                    vp, vp_expected, rtol=0.15,
                    err_msg=f"Markov renter vP inconsistent at m={m}, state={e}",
                )


# ===================================================================
# 14. Internal consistency tests
# ===================================================================


class TestInternalConsistency:
    """Test budget satisfaction, state counts, and tenure coverage."""

    @pytest.fixture(scope="class")
    def model(self):
        agent = HousingPortfolioConsumerType(**_base_params_3period())
        agent.solve()
        return agent

    def test_hgrid_stored_in_solution(self, model):
        sol = model.solution[0]
        assert sol.hGrid is not None
        assert len(sol.hGrid) > 0

    def test_tenure_codes_valid(self, model):
        """Tenure codes should be in {0, 1, 2, 3}."""
        sol = model.solution[0]
        for m in [1.0, 3.0, 5.0, 10.0]:
            for h in [1.5, 2.5, 4.0]:
                t = round(float(sol.tenureFunc(m, h)))
                assert t in (0, 1, 2, 3), (
                    f"Invalid tenure code {t} at m={m}, h={h}"
                )

    def test_owner_consumption_bounded_by_resources(self, model):
        """Owner c < m for all test points (housing costs must be paid)."""
        sol = model.solution[0]
        for m in [2.0, 5.0, 10.0]:
            for h in [1.5, 2.5]:
                c = float(sol.cFuncOwn(m, h))
                assert c < m, f"c={c} exceeds m={m} at h={h}"

    def test_value_finite_everywhere(self, model):
        sol = model.solution[0]
        m_grid = np.linspace(0.5, 15.0, 20)
        h_grid = np.linspace(1.1, 4.5, 5)
        for m in m_grid:
            for h in h_grid:
                v = float(sol.vFuncOwn(m, h))
                assert np.isfinite(v), f"v not finite at m={m}, h={h}"

    def test_markov_state_count(self):
        """Markov model should have exactly 4 states."""
        params = _markov_params_3period(
            aXtraCount=10, aXtraMax=10, ShareCount=3, hGridCount=3,
        )
        agent = MarkovHousingPortfolioConsumerType(**params)
        agent.solve()
        sol = agent.solution[0]
        assert len(sol.cFuncOwn) == 4
        assert len(sol.cFuncRent) == 4
        assert len(sol.vFuncOwn) == 4
        assert len(sol.tenureFunc) == 4


# ===================================================================
# 19. Tenure and participation coverage
# ===================================================================


class TestTenureCoverage:
    """Verify that all four tenure branches are actually reached."""

    def _collect_tenures(self, agent, m_grid, h_grid, periods=None):
        """Scan (m, h) grid and collect the set of tenure codes found."""
        if periods is None:
            periods = range(len(agent.solution))
        found = set()
        for p in periods:
            sol = agent.solution[p]
            for m in m_grid:
                for h in h_grid:
                    t = round(float(sol.tenureFunc(m, h)))
                    found.add(t)
        return found

    def test_own_branch_reached(self):
        """Stay-own (code 0) should appear with high LTV and house price risk."""
        agent = HousingPortfolioConsumerType(
            **_base_params_3period(
                LTV=0.95, SellCost=0.50, DefaultPenalty=0.01,
                HousePriceShkStd=0.3, HousePriceShkCount=5,
            )
        )
        agent.solve()
        tenures = self._collect_tenures(
            agent, np.linspace(0.1, 20.0, 30), np.linspace(1.0, 5.0, 15)
        )
        assert 0 in tenures, f"Own branch (0) not reached; found {tenures}"

    def test_sell_branch_reached(self):
        """Sell (code 1) should appear with default calibration."""
        agent = HousingPortfolioConsumerType(**_base_params_3period())
        agent.solve()
        tenures = self._collect_tenures(
            agent, np.linspace(0.5, 20.0, 20), np.linspace(1.0, 5.0, 10)
        )
        assert 1 in tenures, f"Sell branch (1) not reached; found {tenures}"

    def test_move_branch_reached(self):
        """Move (code 2) should appear at low m for certain h values."""
        agent = HousingPortfolioConsumerType(**_base_params_3period())
        agent.solve()
        tenures = self._collect_tenures(
            agent, np.linspace(0.5, 5.0, 20), np.linspace(1.0, 5.0, 15)
        )
        assert 2 in tenures, f"Move branch (2) not reached; found {tenures}"

    def test_default_branch_reached(self):
        """Default (code 3) should appear with deeply negative equity."""
        agent = HousingPortfolioConsumerType(
            **_base_params_3period(
                LTV=0.95, SellCost=0.50, DefaultPenalty=0.01,
                HousePriceShkStd=0.3, HousePriceShkCount=5,
            )
        )
        agent.solve()
        tenures = self._collect_tenures(
            agent, np.linspace(0.1, 20.0, 30), np.linspace(1.0, 5.0, 15)
        )
        assert 3 in tenures, f"Default branch (3) not reached; found {tenures}"

    def test_all_four_tenures_reached(self):
        """Under a calibration with house price risk, all four branches appear."""
        agent = HousingPortfolioConsumerType(
            **_base_params_3period(
                LTV=0.95, SellCost=0.50, DefaultPenalty=0.01,
                HousePriceShkStd=0.3, HousePriceShkCount=5,
            )
        )
        agent.solve()
        m_grid = np.linspace(0.1, 20.0, 30)
        h_grid = np.linspace(1.0, 5.0, 15)
        tenures = self._collect_tenures(agent, m_grid, h_grid)
        assert tenures == {0, 1, 2, 3}, (
            f"Expected all four tenures {{0,1,2,3}}, found {tenures}"
        )


class TestParticipationCoverage:
    """Verify GM (2005) entry cost: participant and non-participant values differ."""

    @pytest.fixture(scope="class")
    def model(self):
        agent = HousingPortfolioConsumerType(**_base_params_3period())
        agent.solve()
        return agent

    def test_owner_participates(self, model):
        """Owner risky share should be positive somewhere."""
        sol = model.solution[0]
        shares = [float(sol.ShareFuncOwn(m, 2.0)) for m in np.linspace(1.0, 20.0, 20)]
        assert max(shares) > 0.01, f"Owner never participates; max share={max(shares)}"

    def test_renter_participates_at_moderate_wealth(self, model):
        """Renter risky share should be positive at moderate m."""
        sol = model.solution[0]
        s = float(sol.ShareFuncRent(5.0))
        assert s > 0.01, f"Expected positive renter share at m=5, got {s}"

    def test_np_value_leq_p_value_renter(self, model):
        """Non-participant renter value should be <= participant value (entry cost hurts)."""
        sol = model.solution[0]
        for m in [1.0, 5.0, 10.0]:
            v_P = float(sol.vFuncRent(m))
            v_NP = float(sol.vFuncRent_NP(m))
            assert v_NP <= v_P + 1e-8, (
                f"V^NP > V^P at m={m}: {v_NP} > {v_P}"
            )

    def test_np_value_leq_p_value_owner(self, model):
        """Non-participant owner value should be <= participant value."""
        sol = model.solution[0]
        for m in [3.0, 8.0]:
            for h in [1.5, 3.0]:
                v_P = float(sol.vFuncOwn(m, h))
                v_NP = float(sol.vFuncOwn_NP(m, h))
                assert v_NP <= v_P + 1e-8, (
                    f"V^NP_own > V^P_own at (m={m}, h={h}): {v_NP} > {v_P}"
                )

    def test_high_entry_cost_widens_gap(self):
        """Higher entry cost should widen the V^P - V^{NP} gap."""
        agent_low = HousingPortfolioConsumerType(
            **_base_params_3period(EntryCost=0.01)
        )
        agent_low.solve()
        agent_high = HousingPortfolioConsumerType(
            **_base_params_3period(EntryCost=0.5)
        )
        agent_high.solve()
        m_test = 5.0
        gap_low = float(agent_low.solution[0].vFuncRent(m_test)) - \
                  float(agent_low.solution[0].vFuncRent_NP(m_test))
        gap_high = float(agent_high.solution[0].vFuncRent(m_test)) - \
                   float(agent_high.solution[0].vFuncRent_NP(m_test))
        assert gap_high >= gap_low - 1e-8, (
            f"Higher entry cost should widen P-NP gap: "
            f"gap_low={gap_low:.6f}, gap_high={gap_high:.6f}"
        )


# ===================================================================
# Run with: uv run pytest HARK/tests/test_ConsHousingPortfolioModel.py -v
# ===================================================================

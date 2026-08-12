"""Tests for HARK.dual_measure (Harmenberg neutral-measure parallel tracking).

Covers:
1. ``make_Q_measure_dstn``: correct psi/E[psi] reweighting, atoms unchanged.
2. ``_cdf_invert``: deterministic CDF inversion.
3. **Default-path invariance**: composing ``DualMeasureMixin`` without
   enabling it leaves a simulation bit-identical to the plain agent
   (the non-disruption guarantee for this pure-addition module).
4. Dual-mode smoke: Q-side states/history exist, are finite, and the
   Q-measure aggregate of cNrm tracks the P-measure aggregate of
   cLvl/E[pLvl] (the Harmenberg identity) within sampling tolerance.
5. ``compute_mean_pLvl``: closed-form degenerate case.
"""

import numpy as np
import pytest

from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
from HARK.distributions.discrete import DiscreteDistribution
from HARK.dual_measure import (
    DualMeasureMixin,
    _cdf_invert,
    compute_mean_pLvl,
    make_Q_measure_dstn,
)


class DualIndShock(DualMeasureMixin, IndShockConsumerType):
    pass


def _small_agent(cls, seed=31382, agent_count=400, t_sim=15):
    agent = cls(AgentCount=agent_count, T_sim=t_sim, seed=seed)
    agent.track_vars = ["cNrm", "pLvl"]
    agent.solve()
    return agent


def test_make_q_measure_dstn_reweights_by_psi():
    pmv = np.array([0.25, 0.5, 0.25])
    perm = np.array([0.9, 1.0, 1.1])
    tran = np.array([0.8, 1.0, 1.2])
    dstn = DiscreteDistribution(pmv, [perm, tran], seed=0)

    q = make_Q_measure_dstn(dstn)

    expected = pmv * perm / np.dot(pmv, perm)
    expected /= expected.sum()
    assert np.allclose(q.pmv, expected)
    assert np.array_equal(q.atoms[0], perm)
    assert np.array_equal(q.atoms[1], tran)
    assert np.isclose(q.pmv.sum(), 1.0)


def test_make_q_measure_dstn_degenerate_psi_passthrough():
    pmv = np.array([0.5, 0.5])
    perm = np.array([1.0, 1.0])  # zero-variance permanent shock
    tran = np.array([0.7, 1.3])
    dstn = DiscreteDistribution(pmv, [perm, tran], seed=0)
    assert make_Q_measure_dstn(dstn) is dstn


def test_cdf_invert_known_case():
    pmv = np.array([0.2, 0.3, 0.5])
    draws = np.array([0.0, 0.19, 0.21, 0.49, 0.51, 0.99])
    idx = _cdf_invert(draws, pmv)
    assert np.array_equal(idx, np.array([0, 0, 1, 1, 2, 2]))


def test_default_off_is_bit_identical_to_plain_agent():
    """With dual_measure never enabled, the mixin must not change anything."""
    plain = _small_agent(IndShockConsumerType)
    mixed = _small_agent(DualIndShock)
    assert mixed.dual_measure is False

    plain.initialize_sim()
    plain.simulate()
    mixed.initialize_sim()
    mixed.simulate()

    for var in ("cNrm", "pLvl"):
        assert np.array_equal(plain.history[var], mixed.history[var]), var


def test_dual_mode_smoke_and_harmenberg_identity():
    agent = _small_agent(DualIndShock, agent_count=2000, t_sim=40)
    agent.setup_Q_measure()
    assert agent.dual_measure is True

    agent.initialize_sim()
    agent.simulate()

    # Q-side structures exist and are finite after burn-in
    assert set(agent.history_Q) == set(agent.track_vars)
    q_cnrm = agent.history_Q["cNrm"]
    assert np.isfinite(q_cnrm[5:]).all()

    # Harmenberg identity: N * E[pLvl] * mean_Q(cNrm) ~= sum_P(cNrm * pLvl).
    burn = 10
    agg_q = agent.aggregate_Q("cNrm", burn=burn)
    p_cnrm = agent.history["cNrm"][burn:]
    p_plvl = agent.history["pLvl"][burn:]
    agg_p = np.nansum(p_cnrm * p_plvl, axis=1)
    ratio = np.nanmean(agg_q) / np.nanmean(agg_p)
    assert 0.9 < ratio < 1.1, ratio


def test_compute_mean_plvl_degenerate_case():
    """With g=1 the turnover and growth terms cancel exactly:
    E[pLvl] = E[pLvl_init] * g * (1-L)/(1-L^T) * (1-(Lg)^T)/(1-Lg) = 1
    for pLogInit=(0,0), any LivPrb<1."""

    class Stub:
        LivPrb = [0.95]
        PermGroFac = [1.0]
        T_age = 400
        pLogInitMean = 0.0
        pLogInitStd = 0.0

    assert np.isclose(compute_mean_pLvl(Stub()), 1.0)


def test_cache_flag_leaves_p_stream_bit_identical():
    """The _cache_base_shock_draws flag records draws without touching
    the P-stream: identical histories, plus the recorded dict."""
    plain = _small_agent(IndShockConsumerType)
    cached = _small_agent(IndShockConsumerType)
    cached._cache_base_shock_draws = True
    plain.initialize_sim()
    plain.simulate()
    cached.initialize_sim()
    cached.simulate()
    for var in ("cNrm", "pLvl"):
        assert np.array_equal(plain.history[var], cached.history[var]), var
    assert cached._base_shock_draws
    # initialize_sim clears the cache, so the unflagged agent ends with an
    # empty dict rather than no attribute at all.
    assert plain._base_shock_draws == {}


def test_dual_mode_consumes_recorded_draws_exactly():
    """With a DEGENERATE permanent shock, Q == P distribution, so when the
    Q side inverts the recorded P uniforms it must reproduce the P shocks
    exactly, an end-to-end equality proof of the wiring."""
    agent = DualIndShock(AgentCount=300, T_sim=10, seed=31382)
    agent.PermShkStd = [0.0]
    agent.update_income_process()
    agent.track_vars = ["cNrm", "pLvl"]
    agent.solve()
    agent.setup_Q_measure()
    agent._cache_base_shock_draws = True
    agent.initialize_sim()
    agent.simulate()
    np.testing.assert_array_equal(agent.shocks_Q["TranShk"], agent.shocks["TranShk"])
    np.testing.assert_array_equal(agent.shocks_Q["PermShk"], agent.shocks["PermShk"])


def test_setup_q_measure_arms_the_base_draw_cache():
    """setup_Q_measure must switch the cache on and register IncShkDstn_Q.

    Without the flag the Q side draws independently; without the
    registration reset_rng leaves the Q generators where the previous run
    left them.  Either omission decouples P from Q, so both are asserted
    here and the correlation is checked across repeated initialize_sim()
    calls rather than only on the first run.
    """
    agent = _small_agent(DualIndShock, t_sim=6)
    agent.setup_Q_measure()
    assert agent._cache_base_shock_draws is True
    assert "IncShkDstn_Q" in agent.distributions

    corrs = []
    for _ in range(4):
        agent.initialize_sim()
        agent.simulate()
        p = np.asarray(agent.state_now["pLvl"], dtype=float)
        q = np.asarray(agent.state_now_Q["pLvl"], dtype=float)
        corrs.append(np.corrcoef(p, q)[0, 1])
    assert min(corrs) > 0.9, f"P/Q coupling decayed across runs: {corrs}"
    assert max(corrs) - min(corrs) < 1e-12, f"runs not reproducible: {corrs}"


def test_income_shuffle_with_cache_warns_and_empties_the_cache():
    """The two flags are mutually exclusive, and must say so out loud."""
    agent = _small_agent(DualIndShock, t_sim=3)
    agent.income_shuffle = True
    agent.setup_Q_measure()
    agent.initialize_sim()
    with pytest.warns(RuntimeWarning, match="income_shuffle"):
        agent.simulate()
    assert agent._base_shock_draws == {}


def test_initialize_sim_clears_stale_base_draws():
    """A cached run followed by an uncached one must not reuse the old
    uniforms; before this was cleared the reader silently paired the new
    period's agents with the previous run's draws."""
    agent = _small_agent(IndShockConsumerType, t_sim=3)
    agent._cache_base_shock_draws = True
    agent.initialize_sim()
    agent.simulate()
    assert agent._base_shock_draws

    agent._cache_base_shock_draws = False
    agent.initialize_sim()
    assert agent._base_shock_draws == {}
    agent.simulate()
    assert agent._base_shock_draws == {}

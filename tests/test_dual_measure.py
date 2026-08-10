"""Tests for HARK.dual_measure (Harmenberg neutral-measure parallel tracking).

Covers:
1. ``make_Q_measure_dstn`` — correct psi/E[psi] reweighting, atoms unchanged.
2. ``_cdf_invert`` — deterministic CDF inversion.
3. **Default-path invariance** — composing ``DualMeasureMixin`` without
   enabling it leaves a simulation bit-identical to the plain agent
   (the non-disruption guarantee for this pure-addition module).
4. Dual-mode smoke — Q-side states/history exist, are finite, and the
   Q-measure aggregate of cNrm tracks the P-measure aggregate of
   cLvl/E[pLvl] (the Harmenberg identity) within sampling tolerance.
5. ``compute_mean_pLvl`` — closed-form degenerate case.
"""

import numpy as np

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

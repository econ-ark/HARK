"""Tests for HARK.simulation.normalization.

Covers:
1. Shock-mean normalization is exact when enabled.
2. **Default-path invariance** — composing the mixins with switches at
   their defaults leaves simulations bit-identical to the plain agent
   (the non-disruption guarantee for this pure-addition module).
3. Scalar-growth pLvl normalization pins per-cohort log-moments to their
   analytical values.
4. Markov (vector-growth) machinery: stationary-weighted drift, the
   automatic mean-only moments mode, and the scalar-ness of the targets
   (the shape regression that motivated the Markov generalization).
5. Composition with DualMeasureMixin (skipped until HARK.dual_measure
   is merged): hook chaining and the MRO-order guard.
"""

import numpy as np
import pytest

from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
from HARK.simulation.normalization import (
    PermanentIncomeNormalizationMixin,
    ShockNormalizationMixin,
    _stationary_distribution,
)


class NormalizedIndShock(
    ShockNormalizationMixin, PermanentIncomeNormalizationMixin, IndShockConsumerType
):
    pass


def _agent(cls, seed=20260810, agent_count=800, t_sim=12, **attrs):
    agent = cls(AgentCount=agent_count, T_sim=t_sim, seed=seed)
    agent.track_vars = ["pLvl", "cNrm"]
    for name, value in attrs.items():
        setattr(agent, name, value)
    agent.solve()
    agent.initialize_sim()
    agent.simulate()
    return agent


def test_stationary_distribution_two_state():
    transition = np.array([[0.9, 0.1], [0.5, 0.5]])
    pi = _stationary_distribution(transition)
    assert np.allclose(pi @ transition, pi)
    assert np.isclose(pi.sum(), 1.0)
    assert np.allclose(pi, [5.0 / 6.0, 1.0 / 6.0])


def test_shock_means_exact_when_enabled():
    agent = _agent(NormalizedIndShock, normalize_shocks=True)
    # The last drawn shock arrays have exactly unit cross-sectional mean.
    assert np.isclose(np.mean(agent.shocks["PermShk"]), 1.0, atol=1e-12)
    assert np.isclose(np.mean(agent.shocks["TranShk"]), 1.0, atol=1e-12)


def test_defaults_are_bit_identical_to_plain_agent():
    plain = _agent(IndShockConsumerType)
    mixed = _agent(NormalizedIndShock)
    assert mixed.normalize_shocks is False and mixed.normalize_pLvl is False
    for var in ("pLvl", "cNrm"):
        assert np.array_equal(plain.history[var], mixed.history[var]), var


def test_scalar_growth_pins_cohort_log_moments():
    agent = _agent(NormalizedIndShock, agent_count=3000, normalize_pLvl=True)
    log_p = np.log(agent.state_now["pLvl"])
    checked = 0
    for k in np.unique(agent.t_age):
        mask = agent.t_age == k
        if mask.sum() < 30 or k < 1:
            continue
        # state_now was normalized at pre-increment age k - 1
        mu_k, sigma_k = agent._analytical_log_pLvl_moments(k - 1)
        assert np.isclose(np.mean(log_p[mask]), mu_k, atol=1e-8), k
        assert np.isclose(np.std(log_p[mask]), sigma_k, atol=1e-8), k
        checked += 1
    assert checked >= 3


def test_mean_only_mode_shifts_but_does_not_rescale():
    agent = _agent(
        NormalizedIndShock,
        agent_count=3000,
        normalize_pLvl=True,
        pLvl_norm_moments="mean",
    )
    log_p = np.log(agent.state_now["pLvl"])
    for k in np.unique(agent.t_age):
        mask = agent.t_age == k
        if mask.sum() < 30 or k < 2:
            continue
        mu_k, sigma_k = agent._analytical_log_pLvl_moments(k - 1)
        assert np.isclose(np.mean(log_p[mask]), mu_k, atol=1e-8)
        # spread is whatever the sample produced, not the analytic target
        assert not np.isclose(np.std(log_p[mask]), sigma_k, atol=1e-12)
        break


class _MarkovStub(PermanentIncomeNormalizationMixin):
    """Bare stub carrying the attributes the analytic machinery reads."""

    def __init__(self):
        self.PermGroFac = [np.array([1.02, 0.99])]
        self.MrkvArray = [np.array([[0.9, 0.1], [0.5, 0.5]])]

        class _D:
            def __init__(self, pmv, atoms):
                self.pmv = np.asarray(pmv)
                self.atoms = np.asarray(atoms)

        self.PermShkDstn = [
            [_D([0.5, 0.5], [[0.95, 1.05]]), _D([0.5, 0.5], [[0.9, 1.1]])]
        ]
        self.pLogInitMean = 0.0
        self.pLogInitStd = 0.1


def test_markov_effective_drift_is_stationary_weighted_scalar():
    stub = _MarkovStub()
    pi = _stationary_distribution(stub.MrkvArray[0])
    expected = float(pi @ np.log(stub.PermGroFac[0]))
    got = stub._effective_log_PermGroFac()
    assert np.isclose(got, expected)
    assert np.isscalar(got) or np.ndim(got) == 0


def test_markov_moments_are_scalars_and_auto_mode_is_mean_only():
    stub = _MarkovStub()
    mu, sigma = stub._analytical_log_pLvl_moments(7)
    # the shape regression: the pre-generalization formula produced a
    # VECTOR mu here and crashed the cohort assignment downstream
    assert np.ndim(mu) == 0 and np.ndim(sigma) == 0
    assert stub._resolved_moments_mode() == "mean"


def test_composition_with_dual_measure():
    dual = pytest.importorskip("HARK.dual_measure")

    class DualNormalized(
        dual.DualMeasureMixin,
        PermanentIncomeNormalizationMixin,
        IndShockConsumerType,
    ):
        pass

    plain = _agent(IndShockConsumerType)
    composed = _agent(DualNormalized)
    for var in ("pLvl", "cNrm"):
        assert np.array_equal(plain.history[var], composed.history[var]), var

    class WrongOrder(
        PermanentIncomeNormalizationMixin,
        dual.DualMeasureMixin,
        IndShockConsumerType,
    ):
        pass

    bad = WrongOrder(AgentCount=10, T_sim=2, seed=0)
    bad.solve()
    bad.setup_Q_measure()
    bad.initialize_sim()
    with pytest.raises(RuntimeError, match="DualMeasureMixin BEFORE"):
        bad.sim_one_period()

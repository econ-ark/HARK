"""Tests for HARK.simulation.normalization.

Covers:
1. Shock-mean normalization hits the right target: ``PermGroFac`` for
   ``PermShk`` (HARK folds growth into that array) and 1.0 for ``TranShk``,
   checked against an un-normalized agent's realized moments so the test
   cannot pass by agreeing with the module's own formula.
2. **Default-path invariance**: composing the mixins with switches at
   their defaults leaves simulations bit-identical to the plain agent
   (the non-disruption guarantee for this pure-addition module).
3. Scalar-growth pLvl normalization pins per-cohort log-moments, with the
   shock-period count checked against a target recomputed in the test from
   the model's own ``PermShkDstn`` atoms rather than read back from the
   module.
4. Life-cycle awareness: age-varying ``PermGroFac`` and ``PermShkStd`` give
   an age-varying target, not period 0's parameters extrapolated.
5. Markov (vector-growth) machinery: stationary-weighted drift, the
   automatic mean-only moments mode, and the scalar-ness of the targets
   (the shape regression that motivated the Markov generalization).
6. Composition with DualMeasureMixin (skipped until HARK.dual_measure
   is merged).
"""

import numpy as np
import pytest

from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    init_lifecycle,
)
from HARK.simulation.normalization import (
    PermanentIncomeNormalizationMixin,
    ShockNormalizationMixin,
    _stationary_distribution,
)


class NormalizedIndShock(
    ShockNormalizationMixin, PermanentIncomeNormalizationMixin, IndShockConsumerType
):
    pass


def _agent(cls, seed=20260810, agent_count=800, t_sim=12, params=None, **attrs):
    agent = cls(AgentCount=agent_count, T_sim=t_sim, seed=seed, **(params or {}))
    agent.track_vars = ["pLvl", "cNrm", "PermShk", "TranShk"]
    for name, value in attrs.items():
        setattr(agent, name, value)
    agent.solve()
    agent.initialize_sim()
    agent.simulate()
    return agent


def _log_shk_moments(dstn):
    """(E[log psi], Var[log psi]) recomputed here, independently of the module."""
    log_atoms = np.log(np.asarray(dstn.atoms).flatten())
    pmv = np.asarray(dstn.pmv)
    e1 = float(pmv @ log_atoms)
    return e1, float(pmv @ log_atoms**2 - e1**2)


def test_stationary_distribution_two_state():
    transition = np.array([[0.9, 0.1], [0.5, 0.5]])
    pi = _stationary_distribution(transition)
    assert np.allclose(pi @ transition, pi)
    assert np.isclose(pi.sum(), 1.0)
    assert np.allclose(pi, [5.0 / 6.0, 1.0 / 6.0])


# ----------------------------------------------------------------------
# Shock normalization
# ----------------------------------------------------------------------


def test_shock_means_hit_growth_adjusted_targets():
    """PermShk is mean-``PermGroFac``, not mean-one.

    ``get_shocks`` stores ``psi * PermGroFac`` in ``shocks["PermShk"]``, and
    ``transition()`` applies it directly to ``pLvl``. Normalizing that array
    to 1.0 removes permanent income growth rather than sampling noise.
    """
    agent = _agent(NormalizedIndShock, agent_count=20000, normalize_shocks=True)
    G = float(np.asarray(agent.PermGroFac[0]).flatten()[0])
    assert G != 1.0, "the defaults must actually have growth for this to bite"
    assert np.isclose(np.mean(agent.shocks["PermShk"]), G, atol=1e-12)
    assert np.isclose(np.mean(agent.shocks["TranShk"]), 1.0, atol=1e-12)


def test_shock_normalization_preserves_permanent_income_growth():
    """Normalized shock means match an UN-normalized agent's realized means.

    The reference is a plain agent's Monte Carlo moment, not the module's own
    analytical target, so a wrong target cannot make this pass.
    """
    plain = _agent(IndShockConsumerType, agent_count=20000, t_sim=8)
    normed = _agent(
        NormalizedIndShock, agent_count=20000, t_sim=8, normalize_shocks=True
    )

    mean_plain = float(np.mean(plain.history["PermShk"]))
    mean_normed = float(np.mean(normed.history["PermShk"]))
    # Sampling noise at this population is ~1e-4; deleting growth would move
    # the mean by PermGroFac - 1 = 1e-2.
    assert abs(mean_normed - mean_plain) < 5e-3, (mean_plain, mean_normed)

    growth_plain = plain.history["pLvl"][-1].mean() / plain.history["pLvl"][0].mean()
    growth_normed = normed.history["pLvl"][-1].mean() / normed.history["pLvl"][0].mean()
    assert abs(growth_normed - growth_plain) < 0.01, (growth_plain, growth_normed)
    assert growth_normed > 1.0


def test_shock_normalization_no_ops_under_read_shocks_with_warning():
    agent = NormalizedIndShock(AgentCount=200, T_sim=3, seed=7, normalize_shocks=True)
    agent.solve()
    agent.make_shock_history()
    agent.read_shocks = True
    agent.initialize_sim()
    with pytest.warns(UserWarning, match="read_shocks"):
        agent.simulate()


# ----------------------------------------------------------------------
# Default-path invariance
# ----------------------------------------------------------------------


def test_defaults_are_bit_identical_to_plain_agent():
    plain = _agent(IndShockConsumerType)
    mixed = _agent(NormalizedIndShock)
    assert mixed.normalize_shocks is False and mixed.normalize_pLvl is False
    for var in ("pLvl", "cNrm", "PermShk", "TranShk"):
        assert np.array_equal(plain.history[var], mixed.history[var]), var
    assert plain.RNG.bit_generator.state == mixed.RNG.bit_generator.state


# ----------------------------------------------------------------------
# pLvl normalization
# ----------------------------------------------------------------------


def test_scalar_growth_pins_cohort_log_moments():
    """Cohort ``k`` has been through ``k`` permanent shocks, not ``k - 2``.

    ``post_state_hook`` runs before ``t_age`` is advanced, so a cohort reading
    ``t_age == k`` there has already taken ``k + 1`` shocks; after the advance
    the same agents read ``t_age == k + 1``. Newborns are not exempt:
    ``get_shocks`` redraws a random ``PermShk`` for them and pins only
    ``TranShk``. The targets below are rebuilt here from ``PermShkDstn`` so
    that the count is genuinely under test.
    """
    agent = _agent(NormalizedIndShock, agent_count=20000, t_sim=20, normalize_pLvl=True)
    log_p = np.log(agent.state_now["pLvl"])
    e_log_psi, var_log_psi = _log_shk_moments(agent.PermShkDstn[0])
    g_log = float(np.log(np.asarray(agent.PermGroFac[0]).flatten()[0]))

    checked = 0
    for k in np.unique(agent.t_age):
        mask = agent.t_age == k
        if mask.sum() < 50:
            continue
        mu_k = agent.pLogInitMean + k * (g_log + e_log_psi)
        sigma_k = np.sqrt(agent.pLogInitStd**2 + k * var_log_psi)
        assert np.isclose(np.mean(log_p[mask]), mu_k, atol=1e-8), k
        assert np.isclose(np.std(log_p[mask]), sigma_k, atol=1e-8), k
        checked += 1
    assert checked >= 3


def test_normalized_cohort_spread_tracks_unnormalized_cohorts():
    """Normalization removes sampling noise, not the shock-accumulation profile.

    Compared against a plain agent's realized cohort moments, so a target with
    the wrong number of shock periods (which collapses young cohorts to zero
    spread and lags every older one) cannot pass.
    """
    plain = _agent(IndShockConsumerType, agent_count=20000, t_sim=20)
    normed = _agent(
        NormalizedIndShock, agent_count=20000, t_sim=20, normalize_pLvl=True
    )

    checked = []
    for k in np.unique(plain.t_age):
        m_plain = plain.t_age == k
        m_normed = normed.t_age == k
        if m_plain.sum() < 100 or m_normed.sum() < 100 or k < 1:
            continue
        sd_plain = np.std(np.log(plain.state_now["pLvl"][m_plain]))
        sd_normed = np.std(np.log(normed.state_now["pLvl"][m_normed]))
        # A 30% band is ~8 Monte Carlo standard errors at these cohort sizes.
        # Undercounting the shock periods by two collapses cohorts 1 and 2 to
        # zero spread and shrinks cohort 3 by 42%.
        assert 0.7 * sd_plain < sd_normed < 1.3 * sd_plain, (k, sd_plain, sd_normed)
        checked.append(int(k))
    assert {1, 2, 3}.issubset(checked), checked
    assert len(checked) >= 8, checked


def test_lifecycle_targets_vary_with_age():
    """Age-varying ``PermGroFac``/``PermShkStd`` produce an age-varying target.

    ``init_lifecycle`` has a hump-shaped growth profile; reading period 0's
    parameters and multiplying by age fabricates a straight line through it.
    """
    agent = NormalizedIndShock(**init_lifecycle)
    assert len(np.unique(np.asarray(agent.PermGroFac, dtype=float))) > 1

    increments = [
        agent._analytical_log_pLvl_moments(k + 1)[0]
        - agent._analytical_log_pLvl_moments(k)[0]
        for k in range(agent.T_cycle - 1)
    ]
    # Each increment is log G_t + E[log psi_t] for a DIFFERENT t.
    assert len(np.unique(np.round(increments, 12))) > 1

    # Going from age k to age k + 1 adds the period whose t_cycle is k + 1,
    # which draws from income-process index k.
    for k in range(1, agent.T_cycle - 1):
        g_log = float(np.log(np.asarray(agent.PermGroFac[k]).flatten()[0]))
        e_log_psi, _ = _log_shk_moments(agent.PermShkDstn[k])
        assert np.isclose(increments[k], g_log + e_log_psi, atol=1e-12), k


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


def test_warns_when_sim_one_period_bypasses_the_hook():
    class BespokePipeline(NormalizedIndShock):
        def sim_one_period(self):
            self._sim_period_prologue()
            self.get_states()
            self.get_controls()
            self.get_poststates()
            self._sim_period_epilogue()

    agent = BespokePipeline(AgentCount=100, T_sim=2, seed=3, normalize_pLvl=True)
    agent.solve()
    with pytest.warns(UserWarning, match="post_state_hook"):
        agent.initialize_sim()


# ----------------------------------------------------------------------
# Markov machinery
# ----------------------------------------------------------------------


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

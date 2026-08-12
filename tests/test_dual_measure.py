"""Tests for HARK.dual_measure (Harmenberg neutral-measure parallel tracking).

Covers:
1. ``make_Q_measure_dstn``: correct psi/E[psi] reweighting, atoms unchanged,
   and a warning rather than a silent pass-through when there is nothing to
   reweight.
2. ``_cdf_invert``: deterministic CDF inversion.
3. **Default-path invariance**: composing ``DualMeasureMixin`` without
   enabling it leaves a simulation bit-identical to the plain agent, RNG
   stream included (the non-disruption guarantee for this pure-addition
   module); and enabling it does not change what the P pipeline records.
4. Dual-mode correctness: the realized Q sample carries the Q measure's own
   mean permanent shock, checked against the analytical E[psi^2]/E[psi]^2
   with a Q-equals-P control that must fail the same assertion.  The
   Harmenberg aggregation identity is checked separately, and is documented
   there as a check on ``aggregate_Q``'s bookkeeping rather than on the
   reweighting, because in this calibration it holds either way.
5. Q-state integrity: the states in ``history_Q`` satisfy the transition
   arithmetic that defines them, and an unwritten Q state reads as NaN.
6. ``compute_mean_pLvl``: closed-form degenerate case.
"""

from copy import deepcopy

import numpy as np
import pytest

from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
from HARK.ConsumptionSaving.ConsMarkovModel import (
    MarkovConsumerType,
    init_indshk_markov,
)
from HARK.distributions.discrete import DiscreteDistribution
from HARK.dual_measure import (
    DualMeasureMixin,
    _cdf_invert,
    compute_mean_pLvl,
    make_Q_measure_dstn,
)


class DualIndShock(DualMeasureMixin, IndShockConsumerType):
    pass


class DualMarkov(DualMeasureMixin, MarkovConsumerType):
    pass


class _NoReweight(DualIndShock):
    """Control: dual mode wired up, but with the Q measure set back to P.

    Used to show that a test can tell the reweighting apart from its absence.
    A statistic this class also satisfies is not testing the neutral measure.
    """

    def setup_Q_measure(self):
        super().setup_Q_measure()
        self.IncShkDstn_Q = list(self.IncShkDstn)


def _small_agent(cls, seed=31382, agent_count=400, t_sim=15):
    agent = cls(AgentCount=agent_count, T_sim=t_sim, seed=seed)
    agent.track_vars = ["cNrm", "pLvl"]
    agent.solve()
    return agent


def _markov_agent(cls, seed=9001, agent_count=3000, t_sim=12):
    params = deepcopy(init_indshk_markov)
    params["MrkvArray"] = [np.array([[0.9, 0.1], [0.2, 0.8]])]
    params["constructors"] = dict(params["constructors"])
    params["constructors"]["MrkvArray"] = None
    params["LivPrb"] = [np.array([0.9, 0.9])]
    params.update(AgentCount=agent_count, T_sim=t_sim, seed=seed, T_age=None)
    agent = cls(**params)
    agent.cycles = 0
    agent.IncShkDstn = [[agent.IncShkDstn[0][0]] * 2]
    agent.track_vars = ["cNrm", "pLvl"]
    agent.solve()
    return agent


def _psi_moment_ratio(dstn):
    """E[psi^2] / E[psi]^2, the factor by which the Q measure lifts E[psi]."""
    psi = dstn.atoms[0]
    E1 = float(np.dot(dstn.pmv, psi))
    E2 = float(np.dot(dstn.pmv, psi**2))
    return E2 / E1**2


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


def test_degenerate_perm_shock_warns_instead_of_silently_passing_through():
    """A distribution with no permanent dispersion cannot be reweighted.

    Returning it unchanged is the right answer, but doing so quietly leaves a
    caller believing they enabled a variance reduction that is not running.
    """
    pmv = np.array([0.5, 0.5])
    tran = np.array([0.7, 1.3])
    dstn = DiscreteDistribution(pmv, [np.array([1.0, 1.0]), tran], seed=0)
    with pytest.warns(RuntimeWarning, match="no dispersion"):
        assert make_Q_measure_dstn(dstn) is dstn

    neg = DiscreteDistribution(pmv, [np.array([-1.0, 1.0]), tran], seed=0)
    with pytest.warns(RuntimeWarning, match="non-positive mean"):
        assert make_Q_measure_dstn(neg) is neg


def test_setup_q_measure_warns_when_nothing_can_be_reweighted():
    """Dual mode over an entirely degenerate income process is a no-op.

    It still runs the whole Q pipeline, so it costs roughly double for
    results identical to P.  Say so rather than let it look like it worked.
    """
    agent = DualIndShock(AgentCount=100, T_sim=3, seed=31382)
    agent.PermShkStd = [0.0]
    agent.update_income_process()
    agent.track_vars = ["cNrm"]
    agent.solve()
    with pytest.warns(RuntimeWarning, match="no income distribution"):
        agent.setup_Q_measure()
    assert agent.dual_measure is True


def test_setup_q_measure_is_quiet_when_reweighting_is_possible(recwarn):
    """The degenerate warnings must not fire on an ordinary calibration."""
    agent = _small_agent(DualIndShock, t_sim=3)
    agent.setup_Q_measure()
    messages = [str(w.message) for w in recwarn]
    assert not [m for m in messages if "neutral measure" in m or "dispersion" in m], (
        messages
    )


def test_cdf_invert_known_case():
    pmv = np.array([0.2, 0.3, 0.5])
    draws = np.array([0.0, 0.19, 0.21, 0.49, 0.51, 0.99])
    idx = _cdf_invert(draws, pmv)
    assert np.array_equal(idx, np.array([0, 0, 1, 1, 2, 2]))


def test_default_off_is_bit_identical_to_plain_agent():
    """With dual_measure never enabled, the mixin must not change anything.

    Tracks every state variable plus ``MPCnow`` (recorded through the base
    class's ``getattr`` fall-through) and checks that the RNG ends in the
    same place, so a stream consumed but discarded would show up too.
    """
    track = ["cNrm", "pLvl", "mNrm", "aNrm", "bNrm", "kNrm", "aLvl", "MPCnow"]
    plain = _small_agent(IndShockConsumerType)
    mixed = _small_agent(DualIndShock)
    plain.track_vars = list(track)
    mixed.track_vars = list(track)
    assert mixed.dual_measure is False

    plain.initialize_sim()
    plain.simulate()
    mixed.initialize_sim()
    mixed.simulate()

    for var in track:
        assert np.array_equal(plain.history[var], mixed.history[var]), var
    assert plain.RNG.bit_generator.state == mixed.RNG.bit_generator.state


def test_dual_mode_smoke():
    """Q-side structures exist, are the right shape, and are finite."""
    agent = _small_agent(DualIndShock, agent_count=2000, t_sim=40)
    agent.setup_Q_measure()
    assert agent.dual_measure is True

    agent.initialize_sim()
    agent.simulate()

    assert set(agent.history_Q) == set(agent.track_vars)
    for var in agent.track_vars:
        assert agent.history_Q[var].shape == agent.history[var].shape
        assert np.isfinite(agent.history_Q[var]).all(), var


def test_q_shocks_follow_the_neutral_measure():
    """The realized Q sample must carry the Q measure's own mean, not P's.

    Reweighting by psi/E[psi] lifts the mean permanent shock by exactly
    E[psi^2]/E[psi]^2, so that ratio is an analytical target the simulation
    either hits or misses.  ``_NoReweight`` is run alongside as a control:
    it reproduces the P mean exactly (ratio 1), which is what any test of
    this feature has to be able to reject.  The previously shipped assertion
    here (0.9 < aggregate ratio < 1.1) could not: that ratio came out 1.0012
    with the reweighting deleted, comfortably inside the band.
    """
    target = None
    ratios = {}
    for cls in (DualIndShock, _NoReweight):
        agent = _small_agent(cls, agent_count=1000, t_sim=20)
        agent.track_vars = ["PermShk"]
        agent.setup_Q_measure()
        agent.initialize_sim()
        agent.simulate()
        target = _psi_moment_ratio(agent.IncShkDstn[0])
        ratios[cls.__name__] = float(
            np.nanmean(agent.history_Q["PermShk"])
            / np.nanmean(agent.history["PermShk"])
        )

    assert target > 1.005, f"calibration has too little dispersion to test: {target}"
    # Sampling error in this statistic measured at most 2.8e-4 over 10 seeds.
    assert abs(ratios["DualIndShock"] - target) < 1e-3, ratios
    # The control sits 9.4e-3 away, an order of magnitude outside that band.
    assert abs(ratios["_NoReweight"] - target) > 5e-3, ratios


def test_aggregate_q_matches_the_pLvl_weighted_p_mean():
    """``aggregate_Q`` reproduces sum_P(cNrm * pLvl), the Harmenberg identity.

    This checks the bookkeeping in ``aggregate_Q`` (burn-in, E[pLvl] scaling,
    agent count), not the reweighting: in this calibration cNrm is almost
    independent of pLvl, so the identity holds to ~1e-3 whether or not the Q
    measure is applied.  ``test_q_shocks_follow_the_neutral_measure`` is what
    tests the reweighting.  The tolerance is set from the measured spread of
    this ratio across seeds (worst case 1.6e-3 over six seeds).
    """
    agent = _small_agent(DualIndShock, agent_count=2000, t_sim=40)
    agent.setup_Q_measure()
    agent.initialize_sim()
    agent.simulate()

    burn = 10
    agg_q = agent.aggregate_Q("cNrm", burn=burn)
    p_cnrm = agent.history["cNrm"][burn:]
    p_plvl = agent.history["pLvl"][burn:]
    agg_p = np.nansum(p_cnrm * p_plvl, axis=1)
    ratio = np.nanmean(agg_q) / np.nanmean(agg_p)
    assert abs(ratio - 1.0) < 0.01, ratio


def test_dual_mode_still_records_plain_attribute_variables():
    """Turning dual mode on must not change what the P pipeline records.

    ``AgentType.simulate`` falls through to ``getattr(self, var_name)`` for a
    tracked variable that is not a key of ``state_now``/``shocks``/
    ``controls``.  ``MPCnow`` is the one every IndShock model exposes that
    way.  An earlier version of this mixin copied the base recording loop and
    dropped that branch, so ``history['MPCnow']`` came back all-NaN whenever
    ``dual_measure`` was on, silently, while every other tracked variable
    looked right.
    """
    track = ["cNrm", "MPCnow", "who_dies"]

    off = _small_agent(DualIndShock)
    off.track_vars = list(track)
    off.initialize_sim()
    off.simulate()

    on = _small_agent(DualIndShock)
    on.track_vars = list(track)
    on.setup_Q_measure()
    on.initialize_sim()
    on.simulate()

    assert not np.isnan(on.history["MPCnow"]).all()
    for var in track:
        np.testing.assert_array_equal(off.history[var], on.history[var], err_msg=var)


def test_q_states_satisfy_the_transition_identities():
    """Every Q state in ``history_Q`` must be the quantity its name claims.

    ``_transition_Q`` used to compute ``bNrm`` and discard it, and never
    touched ``kNrm``, while ``_lag_Q_states`` blanked ``state_now_Q`` with
    ``np.empty``.  Both were recorded verbatim, so ``history_Q['bNrm']`` and
    ``history_Q['kNrm']`` held whatever numpy last left in those buffers:
    finite, in the right range, and wrong.  Checking the arithmetic that
    defines them catches that where a range or finiteness check does not.
    """
    agent = _small_agent(DualIndShock, agent_count=400, t_sim=12)
    agent.track_vars = ["kNrm", "bNrm", "mNrm", "aNrm", "cNrm", "PermShk", "TranShk"]
    agent.setup_Q_measure()
    agent.initialize_sim()
    agent.simulate()

    h = agent.history_Q
    for var in agent.track_vars:
        assert np.isfinite(h[var]).all(), f"{var} has non-finite entries"

    # mNrm = bNrm + TranShk, within the period, for every agent.
    np.testing.assert_allclose(h["mNrm"], h["bNrm"] + h["TranShk"], rtol=0, atol=1e-12)
    # aNrm = mNrm - cNrm.
    np.testing.assert_allclose(h["aNrm"], h["mNrm"] - h["cNrm"], rtol=0, atol=1e-12)
    # bNrm = (Rfree / PermShk) * kNrm, so bNrm * PermShk = Rfree * kNrm.
    Rfree = float(np.asarray(agent.Rfree).ravel()[0])
    np.testing.assert_allclose(
        h["bNrm"] * h["PermShk"], Rfree * h["kNrm"], rtol=1e-12, atol=1e-12
    )


def test_lagging_q_states_blanks_them_with_nan():
    """A Q state nobody writes must read back as NaN, not as recycled memory.

    ``np.empty`` hands back a buffer numpy has already used, so a forgotten
    write shows up as plausible numbers rather than as a failure.  NaN is the
    loud version.  Same class of bug as issue #1809.
    """
    agent = _small_agent(DualIndShock, agent_count=200, t_sim=4)
    agent.setup_Q_measure()
    agent.initialize_sim()
    agent.simulate()

    agent._lag_Q_states()
    blanked = [
        var for var, val in agent.state_now_Q.items() if isinstance(val, np.ndarray)
    ]
    assert blanked, "no array-valued Q states to check"
    for var in blanked:
        assert np.isnan(agent.state_now_Q[var]).all(), var


def test_markov_newborn_q_shocks_are_redrawn_not_pinned_to_permgrofac():
    """Markov newborns get a random permanent shock on the Q side too.

    ``MarkovConsumerType.get_shocks`` redraws newborn permanent shocks from
    ``IncShkDstn[0][j]``; it does not hand them a deterministic
    ``PermGroFac``.  The Q side used to hardcode the deterministic version,
    which stripped newborn permanent-shock dispersion out of the Q sample and
    biased ``mNrm_Q`` through ``ReffNow = Rport / PermShk``.
    """
    agent = _markov_agent(DualMarkov, agent_count=2000)
    agent.setup_Q_measure()
    agent.initialize_sim()
    agent.simulate()

    # Force a cohort of newborns and redraw, so the branch is exercised
    # while t_age is still 0 (simulate() advances it before returning).
    agent.t_age[:] = 0
    agent._base_shock_draws = {}
    agent._draw_Q_shocks_markov()

    perm_q = agent.shocks_Q["PermShk"]
    mrkv = agent.shocks["Mrkv"]
    for j in range(agent.MrkvArray[0].shape[0]):
        these = mrkv == j
        if these.sum() < 50:
            continue
        atoms = agent.IncShkDstn_Q[0][j].atoms[0] * agent.PermGroFac[0][j]
        assert np.std(perm_q[these]) > 0.0, (
            f"state {j}: newborn Q permanent shocks have no dispersion, "
            "so they were pinned rather than drawn"
        )
        assert np.isin(np.round(perm_q[these], 12), np.round(atoms, 12)).all(), (
            f"state {j}: newborn Q shocks are not atoms of IncShkDstn_Q[0][{j}]"
        )
        # NewbornTransShk is off in this calibration, so TranShk is pinned.
        np.testing.assert_array_equal(
            agent.shocks_Q["TranShk"][these], np.ones(int(these.sum()))
        )


def test_markov_dual_mode_q_states_are_finite_and_consistent():
    """The Markov branch fills the same Q states as the IndShock branch."""
    agent = _markov_agent(DualMarkov, agent_count=1000)
    agent.track_vars = ["cNrm", "pLvl", "mNrm", "bNrm", "kNrm", "aNrm", "TranShk"]
    agent.setup_Q_measure()
    agent.initialize_sim()
    agent.simulate()

    h = agent.history_Q
    for var in agent.track_vars:
        assert np.isfinite(h[var]).all(), var
    np.testing.assert_allclose(h["mNrm"], h["bNrm"] + h["TranShk"], rtol=0, atol=1e-12)


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
    # A degenerate permanent shock is exactly the case setup_Q_measure warns
    # about; here it is the point of the test rather than a mistake.
    with pytest.warns(RuntimeWarning):
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

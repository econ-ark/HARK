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

import warnings
from copy import deepcopy

import numpy as np
import pytest

from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    init_idiosyncratic_shocks,
    init_lifecycle,
)
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


def test_markov_newborn_draws_reach_the_base_draw_cache():
    """The newborn cell records its uniforms, like every other cell does.

    ``MarkovConsumerType.get_shocks`` builds ``_base_shock_draws`` keyed
    ``(t, j)`` for the main loop, and ``_draw_Q_shocks_markov`` looks up
    ``("newborn", j)`` for the newborn cell.  Nothing wrote that key for a
    while: the newborn block had only the ``income_shuffle`` and original-RNG
    branches, so the lookup could never hit and Markov newborns silently drew
    an independent Q shock instead of sharing the P uniforms.

    Asserting on the keys rather than on the shocks is deliberate.  A test
    that clears ``_base_shock_draws`` by hand and then checks the Q shocks
    cannot tell a designed fallback from a cache that is never populated,
    because both take the same branch.
    """
    agent = _markov_agent(DualMarkov, agent_count=600, t_sim=10)
    agent.setup_Q_measure()
    assert agent._cache_base_shock_draws, "setup_Q_measure should arm the cache"
    agent.initialize_sim()
    agent.simulate()

    keys = agent._base_shock_draws.keys()
    newborn_keys = sorted(k for k in keys if isinstance(k, tuple) and k[0] == "newborn")
    assert newborn_keys, (
        "no ('newborn', j) key was recorded, so _draw_Q_shocks_markov's "
        f"lookup can never hit; keys present were {sorted(map(str, keys))}"
    )
    for _, draws in ((k, agent._base_shock_draws[k]) for k in newborn_keys):
        assert draws.size > 0
        assert np.all((draws >= 0.0) & (draws <= 1.0)), "not uniforms"


def test_markov_base_draw_cache_leaves_the_p_stream_alone():
    """Arming the cache must not perturb the P simulation.

    The cache branch draws its uniforms and inverts them exactly as
    ``draw_events`` does, so turning it on changes what is recorded but not
    what is simulated.  Without this, the newborn branch added alongside the
    cache could shift the RNG stream and every P value with it.
    """
    tracked = ["cNrm", "pLvl"]

    def run(cache):
        agent = _markov_agent(DualMarkov, agent_count=600, t_sim=10)
        agent._cache_base_shock_draws = cache
        agent.track_vars = list(tracked)
        agent.initialize_sim()
        agent.simulate()
        return agent

    off, on = run(False), run(True)
    for var in tracked:
        assert np.array_equal(off.history[var], on.history[var]), (
            f"{var} differs with the base-draw cache on, so the cache branch "
            "is not consuming the same randomness as the default path"
        )
    assert off.RNG.bit_generator.state == on.RNG.bit_generator.state


def _cyclical_agent(cls, seed=4242, agent_count=4000, t_sim=8):
    """An infinite-horizon agent with a genuinely 2-period cycle.

    Every other fixture in this file has ``T_cycle == 1``, where
    ``IncShkDstn`` is a one-element list and indices 0 and -1 name the same
    object.  That makes any period-indexing error in the Q pipeline
    unobservable.  The two periods here carry deliberately different
    ``PermShkStd`` and ``PermGroFac``.

    The ``PermGroFac`` asymmetry is the load-bearing one and must not be
    simplified away. Both quantities differ, but the assertion in
    ``test_q_draws_use_the_same_period_as_p_in_a_cyclical_model`` reads only
    the cross-sectional MEAN, and only the growth channel moves it far
    enough to leave the pass band. Measured: with the bug restored and
    ``PermGroFac=[1.00, 1.50]`` the ratios are [0.6728, 1.5595] and the test
    rejects; with ``PermGroFac=[1.0, 1.0]`` they are [1.0020, 1.0397] and it
    does not. That test asserts the two differ, so this cannot rot silently.
    """
    params = deepcopy(init_idiosyncratic_shocks)
    params.update(
        T_cycle=2,
        PermShkStd=[0.05, 0.20],
        TranShkStd=[0.10, 0.10],
        LivPrb=[0.98, 0.98],
        PermGroFac=[1.00, 1.50],
        Rfree=[1.03, 1.03],
        AgentCount=agent_count,
        T_sim=t_sim,
        seed=seed,
    )
    agent = cls(**params)
    agent.cycles = 0
    agent.track_vars = ["cNrm", "pLvl"]
    agent.solve()
    return agent


def test_q_draws_use_the_same_period_as_p_in_a_cyclical_model():
    """P and Q must index the income process with the same offset.

    `_draw_Q_shocks_indshock` used `t - 1 if cycles == 1 else t`, while the P
    side (`IndShockConsumerType.get_shocks`) uses `t - 1` unconditionally and
    so does the Markov Q path.  At `cycles=0, T_cycle=2` that put Q a period
    ahead of P in both `IncShkDstn_Q` and `PermGroFac`, so the two measures
    were drawing from different periods' income processes within the same
    simulated period -- which also breaks the shared-base-draw coupling the
    module exists for, since Q inverted period t-1's uniforms through period
    t's CDF.

    The two periods differ in PermGroFac (1.00 vs 1.50), so getting the offset
    wrong shifts the Q permanent-shock mean by about 50%.
    """
    agent = _cyclical_agent(DualIndShock)
    agent.setup_Q_measure()
    agent.initialize_sim()

    # Compared as a Q/P ratio rather than against PermGroFac[t - 1] directly.
    # `t_cycle` is advanced by `_sim_period_epilogue` after the draw, so
    # reconstructing the draw-time index from the post-step value is off by
    # one -- and the ratio does not need it. Both sides fold in the same
    # growth factor, so it cancels, leaving only the Q reweighting factor
    # E[psi^2]/E[psi]^2. That is at most 1 + max(PermShkStd)^2 = 1.04 here.
    # The bug made Q use the other period's factor, sending the ratio to
    # 1.50 or 0.67.
    # The two periods must differ in PermGroFac, not merely in PermShkStd.
    # All of this test's power comes from the growth channel: with uniform
    # growth the buggy ratios are [1.0020, 1.0397] (measured), entirely
    # inside the band below, and the test goes silent. The PermShkStd
    # asymmetry contributes nothing to the assertion, which reads only the
    # mean.
    assert agent.PermGroFac[0] != agent.PermGroFac[1], (
        "this test detects the bug through the growth channel; with uniform "
        "PermGroFac the buggy ratios sit inside the band"
    )

    worst = 0.0
    for _ in range(6):
        agent.sim_one_period()
        for t in np.unique(agent.t_cycle):
            cell = agent.t_cycle == t
            if cell.sum() < 100:
                continue
            mean_p = np.mean(agent.shocks["PermShk"][cell])
            mean_q = np.mean(agent.shocks_Q["PermShk"][cell])
            ratio = mean_q / mean_p
            worst = max(worst, abs(ratio - 1.0))
            assert 0.95 < ratio < 1.15, (
                f"t_cycle={t}: mean Q PermShk / mean P PermShk = {ratio:.4f}. "
                "Both measures fold in the same PermGroFac, so this ratio can "
                "only be the Q reweighting factor (<= 1.04 in this "
                "calibration). A value near 1.5 or 0.67 means Q indexed a "
                "different period's income process than P did."
            )
    # Guard against the assertion above passing vacuously on a calibration
    # where the two periods happen to agree: the reweighting must be visible.
    assert worst > 1e-3, (
        "the Q reweighting is not measurable in this fixture, so the bound "
        "above would pass even if Q and P shared no periods at all"
    )


def test_cohort_mass_normalizer_has_the_right_limit_at_no_mortality():
    """(1 - L)/(1 - L**T) is 0/0 at L == 1, and the limit is 1/T, not 0.

    Both callers got this wrong in opposite directions before it was shared:
    `compute_mean_pLvl`'s C_norm had no guard and divided zero by zero, and
    `compute_pLvl_factor`'s delta_eff guarded but returned `1 - LivPrb`,
    which is exactly 0 where the answer is 1/T_age. LivPrb == 1 is a
    reachable calibration -- it is what you set for no mortality.
    """
    from HARK.dual_measure import _cohort_mass_normalizer as norm

    for T_age in (10, 50, 400):
        got = norm(1.0, T_age)
        assert np.isfinite(got), "LivPrb == 1 must not produce nan or inf"
        assert got == pytest.approx(1.0 / T_age), (
            "with no mortality every cohort is the same size, so newborns "
            f"are 1/{T_age} of the population"
        )

    # Approaching the degenerate point must converge to the limit, which is
    # what makes the guard a limit rather than a special case bolted on.
    T_age = 50
    for LivPrb in (0.99, 0.999, 0.9999, 0.99999):
        assert norm(LivPrb, T_age) == pytest.approx(1.0 / T_age, abs=6e-3)
    assert norm(0.99999, T_age) == pytest.approx(1.0 / T_age, abs=1e-4)


def test_cohort_mass_normalizer_matches_the_formulas_it_replaced():
    """Away from the degenerate point, nothing may move."""
    from HARK.dual_measure import _cohort_mass_normalizer as norm

    for T_age in (10, 50, 65, 400):
        for LivPrb in (0.90, 0.95, 0.98, 0.99, 0.999):
            old_C_norm = (1.0 - LivPrb) / (1.0 - LivPrb**T_age)
            L_T = LivPrb**T_age
            old_delta_eff = 1.0 - (LivPrb - L_T) / (1.0 - L_T)
            got = norm(LivPrb, T_age)
            assert got == pytest.approx(old_C_norm, rel=1e-15)
            assert got == pytest.approx(old_delta_eff, rel=1e-15)


def test_lifecycle_setup_does_not_warn_once_per_retirement_period():
    """Degenerate periods are normal in a lifecycle, so they must not shout.

    Retirement periods are built with ``n_approx_Perm = 1``, so the permanent
    shock is a point mass and P equals Q there by construction, not by
    accident. A stock ``init_lifecycle`` has 25 such periods out of 65, and
    warning per period emitted 25 identical messages from a single
    ``setup_Q_measure()`` call. The cost is not the noise itself: it buries
    the aggregate warning, which is the one that means the caller enabled
    dual mode and will get nothing for it.
    """
    agent = DualIndShock(**deepcopy(init_lifecycle))
    agent.track_vars = ["cNrm"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent.setup_Q_measure()
    degenerate_warnings = [
        str(w.message) for w in caught if "no dispersion" in str(w.message)
    ]
    assert not degenerate_warnings, (
        f"{len(degenerate_warnings)} per-period warnings on a stock lifecycle"
    )

    # The information is still available, just not shouted.
    assert len(agent.Q_degenerate_periods) == 25
    assert agent.Q_degenerate_periods == sorted(agent.Q_degenerate_periods)
    # ...and 40 periods really did reweight, which is why the aggregate
    # warning below must stay silent here.
    assert agent._any_reweighting_happened()


def test_make_q_measure_dstn_still_warns_by_default():
    """Suppression is opt-in, so a direct caller keeps the diagnostic."""
    dstn = DiscreteDistribution(
        np.array([0.5, 0.5]), [np.array([1.0, 1.0]), np.array([0.7, 1.3])], seed=0
    )
    with pytest.warns(RuntimeWarning, match="no dispersion"):
        make_Q_measure_dstn(dstn)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        make_Q_measure_dstn(dstn, warn=False)
    assert not caught


def test_setup_q_measure_refuses_models_whose_get_Rport_reads_p_side_state():
    """A Q agent must not be priced at the P agent's portfolio return.

    `_transition_Q` calls `self.get_Rport()`. That is sound when the return
    depends only on model constants or on state both measures share, and
    unsound when `get_Rport` reads state the Q pipeline never mirrors:

      KinkedRconsumerType   picks Rboro vs Rsave from state_prev["aNrm"]
      KinkyPrefConsumerType delegates to KinkedRconsumerType.get_Rport
      ConsRiskyAssetModel   builds the return from controls["Share"], the
                            P agent's realized portfolio choice, for which
                            no Q-side counterpart exists at all

    Composing any of those produces a wrong number rather than a degraded
    one, so this refuses instead of warning. KinkyPref is in the list on
    purpose: inspecting only its own one-line body reports it clean, which
    is why the check follows one level of delegation.
    """
    from HARK.ConsumptionSaving.ConsIndShockModel import KinkedRconsumerType
    from HARK.ConsumptionSaving.ConsPrefShockModel import KinkyPrefConsumerType
    from HARK.ConsumptionSaving.ConsRiskyAssetModel import RiskyAssetConsumerType

    for base in (KinkedRconsumerType, KinkyPrefConsumerType, RiskyAssetConsumerType):
        cls = type("Dual" + base.__name__, (DualMeasureMixin, base), {})
        agent = cls(AgentCount=50, T_sim=3, seed=1)
        agent.track_vars = ["cNrm"]
        agent.solve()
        with pytest.raises(NotImplementedError, match="get_Rport"):
            agent.setup_Q_measure()


def test_setup_q_measure_admits_models_whose_get_Rport_is_measure_neutral():
    """The guard must not shut out the models the mixin is built for."""
    for agent in (
        _small_agent(DualIndShock, t_sim=3),
        _markov_agent(DualMarkov, 200, 3),
    ):
        agent.setup_Q_measure()
        assert agent.dual_measure is True


def test_dual_mode_on_leaves_the_p_history_and_rng_untouched():
    """Enabling dual mode must not perturb the P measure at all.

    `test_default_off_is_bit_identical_to_plain_agent` covers the mixin being
    PRESENT but off. This covers it being ON, which is the configuration where
    `_step_Q_measure` runs and where `sim_one_period` shares
    `_sim_period_prologue` with the base class. The Q pipeline reads
    `state_prev`, and the prologue is what populates it, so the two interact
    precisely here.

    That the P stream survives is not incidental: `setup_Q_measure` arms the
    base-draw cache, and the Q side inverts the SAME recorded uniforms rather
    than drawing its own. If it ever drew independently, the RNG would
    advance and every P value after the first period would move. So this
    pins the shared-uniform design, not just the delegation.
    """
    tracked = ["cNrm", "pLvl", "mNrm", "aNrm", "bNrm", "kNrm", "aLvl"]

    def run(dual):
        agent = DualIndShock(AgentCount=500, T_sim=20, seed=31382)
        agent.track_vars = list(tracked)
        agent.solve()
        if dual:
            agent.setup_Q_measure()
        agent.initialize_sim()
        agent.simulate()
        return agent

    off, on = run(False), run(True)
    assert on.dual_measure is True and off.dual_measure is False

    for var in tracked:
        assert np.array_equal(off.history[var], on.history[var]), (
            f"{var} moved when dual mode was enabled, so the Q pipeline is "
            "consuming randomness the P pipeline used to get"
        )
    assert off.RNG.bit_generator.state == on.RNG.bit_generator.state, (
        "the RNG ended in a different place with dual mode on, so a stream "
        "was consumed or skipped even though the histories happen to match"
    )
    # Not vacuous: dual mode must actually have produced a Q history.
    assert set(on.history_Q) == set(tracked)
    assert np.isfinite(on.history_Q["cNrm"]).any()

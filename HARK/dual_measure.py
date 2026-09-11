"""
Dual-measure (P/Q) Monte Carlo simulation mixin for HARK.

The Harmenberg (2021) neutral measure Q reweights permanent income shock
probabilities by psi/E[psi], eliminating permanent-income sampling noise
from aggregate consumption estimates (3--17x variance reduction).

This module provides a mixin class ``DualMeasureMixin`` that, when composed
with any ``IndShockConsumerType`` subclass, runs the standard P-measure
simulation pipeline alongside a parallel Q-measure state update in a
single pass.  Markov transitions, mortality draws, and base uniform random
numbers are shared between the two measures; only the shock magnitudes (and
consequently pLvl, mNrm, cNrm, aNrm) diverge.

Usage::

    from HARK.dual_measure import DualMeasureMixin
    from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType

    class DualIndShock(DualMeasureMixin, IndShockConsumerType):
        pass

    agent = DualIndShock(**params)
    agent.solve()
    agent.setup_Q_measure()  # auto-computes IncShkDstn_Q
    agent.initialize_sim()
    agent.simulate()

    P_cNrm = agent.history['cNrm']
    Q_cNrm = agent.history_Q['cNrm']

Reference: Harmenberg (2021), "Aggregation with a permanent income shock",
           *Journal of Economic Dynamics and Control*.
"""

import warnings

import numpy as np

from HARK.distributions.discrete import DiscreteDistribution, cdf_invert

#: Kept as a module-level alias: this module defined `_cdf_invert` before
#: it moved next to the distribution it inverts, and tests import it here.
_cdf_invert = cdf_invert

__all__ = [
    "make_Q_measure_dstn",
    "DualMeasureMixin",
    "compute_mean_pLvl",
    "compute_pLvl_factor",
]


def make_Q_measure_dstn(dstn, warn=True):
    """Reweight a DiscreteDistribution by psi/E[psi] (Harmenberg neutral measure).

    Parameters
    ----------
    dstn : DiscreteDistribution
        Joint (PermShk, TranShk) distribution under the physical measure.
        ``dstn.atoms[0]`` must be the permanent shock values.
    warn : bool
        Whether to warn when no reweighting is possible.  Callers reweighting
        a single distribution want the warning and get it by default.
        ``setup_Q_measure`` passes False because it maps this over every
        period of a lifecycle, where degenerate periods are the normal case
        rather than a symptom: retirement periods are built with
        ``n_approx_Perm = 1``, so P equals Q there by construction.  Warning
        once per period made a stock ``init_lifecycle`` emit 25 identical
        warnings from one call, which buries the aggregate warning that does
        mean something.  It reports the count instead.

    Returns
    -------
    DiscreteDistribution
        New distribution with Q-measure probabilities and the same atoms.
        If the permanent shock has zero variance, or a non-positive mean,
        there is no neutral measure to construct and the original
        distribution is returned unchanged: the caller gets a Q measure
        identical to P, which is a no-op rather than the variance reduction
        the reweighting is asked for.
    """
    perm_atoms = dstn.atoms[0]
    E_perm = np.dot(dstn.pmv, perm_atoms)
    if E_perm <= 0 or np.std(perm_atoms) < 1e-12:
        if warn:
            warnings.warn(
                "make_Q_measure_dstn: the permanent shock has "
                + (
                    f"non-positive mean ({E_perm:.6g})"
                    if E_perm <= 0
                    else "no dispersion"
                )
                + ", so no neutral-measure reweighting is possible; returning "
                "the P-measure distribution unchanged. Q-measure results will "
                "equal P-measure results for this distribution.",
                RuntimeWarning,
                stacklevel=2,
            )
        return dstn
    Q_pmv = dstn.pmv * perm_atoms / E_perm
    Q_pmv /= Q_pmv.sum()
    return DiscreteDistribution(Q_pmv, dstn.atoms, seed=dstn.seed)


#: Normalization flags that adjust the P side without a Q counterpart, with
#: the mechanism each one uses. Both are refused by :func:`_refuse_normalization`.
_NORMALIZATION_FLAGS = (
    (
        "normalize_shocks",
        "rescales shocks['PermShk'] inside get_shocks, after the base uniforms "
        "the Q pipeline inverts were recorded",
    ),
    (
        "normalize_pLvl",
        "adjusts state_now['pLvl'] in post_state_hook, which the Q pipeline "
        "does not mirror",
    ),
)


def _refuse_normalization(agent):
    """Refuse to set up dual mode on an agent that normalizes its P side.

    Both mixins in :mod:`HARK.simulation.normalization` adjust P-side
    quantities that the Q pipeline never sees, so P's sampling noise is
    removed and Q's is left intact.

    That is not a coupling nit. Dual mode exists to show that the neutral
    measure carries *less* noise than P, and either composition reverses the
    comparison. Measured cross-seed standard deviations:

    ``normalize_shocks``, period-mean shock deviation, 12 seeds at 2000 agents
        off: 2.03e-3 (P) against 2.03e-3 (Q).  on: 6.1e-17 (P) against
        2.03e-3 (Q).
    ``normalize_pLvl``, final-period mean pLvl, 10 seeds at 2000 agents
        off: 9.71e-3 (P) against 1.03e-2 (Q).  on: 5.91e-4 (P) against
        1.03e-2 (Q), with the Q history bit-identical either way.

    A user stacking a variance-reduction feature onto dual mode -- the natural
    thing to try, and what this module's own docstring example does -- would
    conclude the neutral measure increases variance.

    The damage is not confined to the variance comparison. ``aggregate_Q``
    multiplies a moment taken from the *P* history by a mean taken from the
    *Q* history, so under either composition it combines a pinned half with an
    unpinned one and the level estimate itself is biased. That closes the last
    reading under which the combination is still useful: it is not merely the
    comparison that breaks.

    Composing them properly means normalizing the Q side to the Q measure's
    own targets -- for shocks, ``PermGroFac * E[psi**2] / E[psi]**2`` rather
    than ``PermGroFac``. That is a design decision about what normalization
    means under a change of measure, so this refuses rather than guessing.
    """
    for flag, mechanism in _NORMALIZATION_FLAGS:
        if getattr(agent, flag, False):
            raise NotImplementedError(
                f"{type(agent).__name__} sets {flag}=True, which {mechanism}. "
                "The P side would be pinned while Q kept its sampling noise, "
                "reversing the variance comparison dual mode exists to "
                "demonstrate and biasing aggregate_Q, which mixes a P-side "
                f"moment with a Q-side mean. Set {flag}=False to use the "
                "neutral measure, or drop DualMeasureMixin to use "
                "normalization; normalizing the Q side to its own targets is "
                "not implemented. This applies to both normalize_shocks and "
                "normalize_pLvl, so switching to the other one is not a "
                "workaround."
            )


class DualMeasureMixin:
    """Mixin that adds Harmenberg neutral-measure (Q) parallel tracking.

    Compose with any ``IndShockConsumerType`` subclass via MRO::

        class DualAgent(DualMeasureMixin, IndShockConsumerType):
            pass

    When ``dual_measure=True`` (set by :meth:`setup_Q_measure`),
    :meth:`sim_one_period` runs the standard P-measure pipeline and then a
    parallel Q-measure state update that reuses the same mortality draws,
    Markov transitions, and base uniform random numbers.

    **Zero impact on base classes**: ``AgentType``, ``IndShockConsumerType``,
    and ``MarkovConsumerType`` are not modified.  The mixin overrides
    ``sim_one_period`` and ``simulate`` via MRO.  Neither reimplements the
    P-pipeline: ``simulate`` delegates each period to ``super().simulate(1)``,
    and ``sim_one_period`` runs the base class's own
    :meth:`~HARK.core.AgentType._sim_period_prologue` and
    :meth:`~HARK.core.AgentType._sim_period_epilogue` around the Q-step.
    """

    dual_measure = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_Q_measure(self):
        """Auto-compute ``IncShkDstn_Q`` from ``IncShkDstn`` and enable dual mode.

        For each period's income shock distribution, the Q-measure reweights
        the probability mass by psi/E[psi].  The atoms (shock values) are
        unchanged; only their sampling probabilities differ.

        Also turns on ``_cache_base_shock_draws`` so that ``get_shocks()``
        records the uniforms the P-draw consumed, and registers
        ``IncShkDstn_Q`` with ``self.distributions`` so ``reset_rng`` rewinds
        it alongside the P-side.  Both are needed for the shared base draws
        this class documents: without the flag the Q-side draws
        independently, and without the registration the Q generators keep
        advancing across ``initialize_sim()`` calls, so the coupling holds on
        the first run and decays afterwards.  Set
        ``_cache_base_shock_draws = False`` after this call to get
        independent Q draws instead.

        Warns when no distribution admits a neutral measure (every permanent
        shock degenerate), because dual mode then runs the whole Q pipeline
        to reproduce the P answer at twice the cost and none of the variance
        reduction.  ``dual_measure`` is still set: the Q pipeline is well
        defined in that case, just not useful.
        """
        # _transition_Q calls self.get_Rport() to price the Q agent's assets.
        # That is fine when get_Rport is a model constant or depends only on
        # state the two measures share, which covers IndShock (Rfree by
        # t_cycle), Markov (indexed by the shared shocks["Mrkv"]) and AggShock
        # (a scalar RfreeNow). It is unsound when get_Rport reads state the Q
        # pipeline does not mirror: KinkedRconsumerType picks Rboro vs Rsave
        # from state_prev["aNrm"], KinkyPrefConsumerType delegates to it, and
        # ConsRiskyAssetModel builds the return from controls["Share"], the P
        # agent's realized portfolio choice, for which there is no Q-side
        # counterpart at all. Composing those gives the Q agent the P agent's
        # return: a wrong number, not a degraded one, so this refuses rather
        # than warns.
        p_side = _get_Rport_reads_p_side(type(self).get_Rport)
        if p_side:
            raise NotImplementedError(
                f"{type(self).__name__}.get_Rport reads "
                f"{sorted(p_side)}, which the Q pipeline does not mirror, so "
                "the Q measure would be priced at the P agent's portfolio "
                "return. DualMeasureMixin requires a get_Rport that depends "
                "only on model constants or on state shared by both measures "
                "(as in IndShockConsumerType, MarkovConsumerType and "
                "AggShockConsumerType). Kinked-R and portfolio-choice models "
                "need a Q-aware get_Rport before they can be composed."
            )

        _refuse_normalization(self)

        # warn=False here, and the degenerate periods are counted instead.
        # Mapping over a lifecycle hits legitimately degenerate periods as a
        # matter of course: retirement periods are constructed with
        # n_approx_Perm = 1, so P equals Q there by design. A stock
        # init_lifecycle emits 25 of those from a single call, which trains
        # the reader to filter the module and so hides the aggregate warning
        # below, which is the one that means something.
        self.IncShkDstn_Q = []
        for period_dstn in self.IncShkDstn:
            if isinstance(period_dstn, (list, tuple)):
                self.IncShkDstn_Q.append(
                    [make_Q_measure_dstn(d, warn=False) for d in period_dstn]
                )
            else:
                self.IncShkDstn_Q.append(make_Q_measure_dstn(period_dstn, warn=False))

        # Recorded rather than warned: inspectable after the fact without
        # costing anything on the healthy path.
        self.Q_degenerate_periods = [
            t
            for t, (p, q) in enumerate(zip(self.IncShkDstn, self.IncShkDstn_Q))
            if p is q
        ]

        if not self._any_reweighting_happened():
            warnings.warn(
                "setup_Q_measure: no income distribution admitted a neutral "
                "measure, so every Q distribution is the P distribution and "
                "dual mode will reproduce the P results exactly, at roughly "
                "twice the cost. Leave dual_measure off unless the permanent "
                "shock has dispersion.",
                RuntimeWarning,
                stacklevel=2,
            )

        if "IncShkDstn_Q" not in self.distributions:
            self.distributions = list(self.distributions) + ["IncShkDstn_Q"]
        self._cache_base_shock_draws = True
        self.dual_measure = True

    def _any_reweighting_happened(self):
        """True when at least one Q distribution differs from its P source.

        ``make_Q_measure_dstn`` returns its argument unchanged when there is
        nothing to reweight, so identity of the objects is an exact test.
        """
        for p_dstn, q_dstn in zip(self.IncShkDstn, self.IncShkDstn_Q):
            if isinstance(p_dstn, (list, tuple)):
                if any(q is not p for p, q in zip(p_dstn, q_dstn)):
                    return True
            elif q_dstn is not p_dstn:
                return True
        return False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize_sim(self):
        """Extend: allocate Q-state arrays after the P-pipeline initializes."""
        super().initialize_sim()
        if self.dual_measure:
            self._initialize_sim_Q()

    def _initialize_sim_Q(self):
        """Allocate Q-side dictionaries that mirror the P-side."""
        self.state_now_Q = {}
        self.state_prev_Q = {}
        self.shocks_Q = {}
        self.controls_Q = {}

        for var in self.state_now:
            val = self.state_now[var]
            if isinstance(val, np.ndarray):
                self.state_now_Q[var] = val.copy()
            else:
                self.state_now_Q[var] = val

        self.clear_history_Q()

    def clear_history_Q(self):
        """Allocate NaN-filled Q-history arrays for every tracked variable."""
        self.history_Q = {}
        for var_name in self.track_vars:
            self.history_Q[var_name] = np.empty((self.T_sim, self.AgentCount))
            self.history_Q[var_name].fill(np.nan)

    # ------------------------------------------------------------------
    # post-state hook (self-contained default)
    # ------------------------------------------------------------------

    def post_state_hook(self):
        """Extension point invoked between ``get_states()`` and
        ``get_controls()`` inside this mixin's ``sim_one_period``.

        The default does nothing (beyond deferring to a base-class hook of
        the same name, if one ever exists), so composing this mixin changes
        no behavior.  Cooperating mixins (e.g. a pLvl-normalization mixin)
        can override it to adjust states before controls are computed.
        """
        sup = getattr(super(), "post_state_hook", None)
        if sup is not None:
            sup()

    # ------------------------------------------------------------------
    # sim_one_period override
    # ------------------------------------------------------------------

    def sim_one_period(self):
        """Run the P-pipeline, then the Q-pipeline before time advancement.

        Calling ``super().sim_one_period()`` and appending the Q-step does not
        work: the base class advances ``t_age`` and ``t_cycle`` at the end,
        and the Q-pipeline needs them at the pre-increment values the
        P-pipeline used.  That is why the base class exposes the two halves
        separately, so the Q-step can sit between them.  Everything before
        and after it is the base class's own code, reached through the MRO,
        not a copy of it living here.
        """
        self._sim_period_prologue()
        self.get_states()
        self.post_state_hook()
        self.get_controls()
        self.get_poststates()

        # --- Q-pipeline (while t_age / t_cycle still match P's view) ---
        if self.dual_measure:
            self._step_Q_measure()

        self._sim_period_epilogue()

    # ------------------------------------------------------------------
    # Q-measure one-period pipeline
    # ------------------------------------------------------------------

    def _step_Q_measure(self):
        """Q-measure state evolution for one period.

        The ordering mirrors the P-pipeline in ``AgentType.sim_one_period()``:
        lag states -> shocks -> transition -> controls -> poststates.
        Mortality/rebirth is already handled in the P-pipeline; we synchronize
        the Q-states with the same ``who_dies`` mask.
        """
        self._sync_mortality_Q()
        self._lag_Q_states()
        self._draw_Q_shocks()
        self._transition_Q()
        self._get_controls_Q()
        self._get_poststates_Q()

    def _sync_mortality_Q(self):
        """Apply the same death/rebirth events to Q-states.

        ``self.who_dies`` was set by the P-pipeline's ``get_mortality()``.
        By the time this runs, the P-pipeline has already completed its full
        cycle: the newborn initial states (from ``sim_birth``) were moved into
        ``state_prev`` during the lag step.  We copy those to ``state_now_Q``
        for the dead agents so that ``_lag_Q_states`` will propagate them into
        ``state_prev_Q``, matching the P-pipeline's treatment of newborns.
        """
        who_dies = getattr(self, "who_dies", None)
        if who_dies is None or not np.any(who_dies):
            return
        for var in self.state_now_Q:
            val_prev = self.state_prev.get(var)
            if val_prev is not None and isinstance(val_prev, np.ndarray):
                self.state_now_Q[var][who_dies] = val_prev[who_dies]

    def _lag_Q_states(self):
        """Copy ``state_now_Q`` -> ``state_prev_Q``, blank ``state_now_Q``.

        The blank is NaN rather than ``np.empty``: a state the Q pipeline
        forgets to write then shows up as NaN in ``history_Q`` instead of as
        whatever numbers numpy last left in that buffer.  Uninitialized memory
        reads back as plausible-looking values in exactly the right range for
        the variable it is masquerading as, which is the failure mode of
        issue #1809.
        """
        for var in self.state_now_Q:
            self.state_prev_Q[var] = self.state_now_Q[var]
            if isinstance(self.state_now_Q[var], np.ndarray):
                self.state_now_Q[var] = np.full(self.AgentCount, np.nan)
            # scalar (aggregate) vars are kept as-is

    @property
    def _is_markov(self):
        """True when the agent has Markov-indexed income distributions."""
        return "Mrkv" in getattr(self, "shocks", {})

    def _draw_Q_shocks(self):
        """Draw Q-measure shocks using the base uniforms saved by get_shocks().

        The P-pipeline's ``get_shocks()`` stores ``self._base_shock_draws``,
        a dict whose keys are either scalar ``t_cycle`` values (IndShock)
        or ``(t_cycle, mrkv_state)`` tuples (Markov).  We invert through the
        Q-CDF to get Q-shock indices.
        """
        if self._is_markov:
            self._draw_Q_shocks_markov()
        else:
            self._draw_Q_shocks_indshock()

    def _draw_Q_shocks_indshock(self):
        """Q-shock draw for IndShockConsumerType (non-Markov)."""
        base_draws_dict = getattr(self, "_base_shock_draws", {})
        newborn = self.t_age == 0

        PermShkQ = np.zeros(self.AgentCount)
        TranShkQ = np.zeros(self.AgentCount)

        for t in np.unique(self.t_cycle):
            idx = self.t_cycle == t
            # t - 1 unconditionally, matching IndShockConsumerType.get_shocks
            # (`t = s - 1`) and _draw_Q_shocks_markov (`IncShkDstn_Q[t - 1]`).
            # This was `t - 1 if self.cycles == 1 else t`, which put Q one
            # period ahead of P in both the shock distribution and the growth
            # factor whenever cycles != 1 and T_cycle > 1 -- including
            # cycles=0, this module's own documented usage. It was invisible
            # because at T_cycle == 1 the list has one element and indices 0
            # and -1 name it.
            t_key = t - 1
            N = np.sum(idx)
            if N > 0:
                IncShkDstnQ = self.IncShkDstn_Q[t_key]
                PermGroFacNow = self.PermGroFac[t_key]

                base_draws = base_draws_dict.get(t)
                if base_draws is not None:
                    indices_Q = _cdf_invert(base_draws, IncShkDstnQ.pmv)
                else:
                    indices_Q = IncShkDstnQ.draw_events(N)

                PermShkQ[idx] = IncShkDstnQ.atoms[0][indices_Q] * PermGroFacNow
                TranShkQ[idx] = IncShkDstnQ.atoms[1][indices_Q]

        N_new = np.sum(newborn)
        if N_new > 0:
            IncShkDstnQ_0 = self.IncShkDstn_Q[0]
            PermGroFacNow = self.PermGroFac[0]
            base_new = base_draws_dict.get("newborn")
            if base_new is not None:
                indices_Q = _cdf_invert(base_new, IncShkDstnQ_0.pmv)
            else:
                indices_Q = IncShkDstnQ_0.draw_events(N_new)
            PermShkQ[newborn] = IncShkDstnQ_0.atoms[0][indices_Q] * PermGroFacNow
            TranShkQ[newborn] = IncShkDstnQ_0.atoms[1][indices_Q]

            if not getattr(self, "NewbornTransShk", False):
                TranShkQ[newborn] = 1.0

        self.shocks_Q["PermShk"] = PermShkQ
        self.shocks_Q["TranShk"] = TranShkQ

    def _draw_Q_shocks_markov(self):
        """Q-shock draw for MarkovConsumerType (Mrkv-indexed distributions).

        Base draws are keyed by ``(t_cycle, mrkv_state)`` tuples.
        Markov states are shared between P and Q; only the income shock
        magnitudes differ due to Q-reweighting.

        Newborns mirror ``MarkovConsumerType.get_shocks``: their permanent
        shock is redrawn from ``IncShkDstn[0][j]`` rather than being the
        deterministic ``PermGroFac[0][j]``, and ``TranShk`` is pinned to 1
        only when ``NewbornTransShk`` is off.  Under
        ``_cache_base_shock_draws`` the newborn redraw records its uniforms
        under ``("newborn", j)``, so newborn P and Q permanent shocks share
        them like every other cell.  The independent draw below is the
        fallback for when that key is absent, which is any run with the cache
        off.
        """
        base_draws_dict = getattr(self, "_base_shock_draws", {})
        MrkvNow = self.shocks["Mrkv"]
        newborn = self.t_age == 0

        PermShkQ = np.zeros(self.AgentCount)
        TranShkQ = np.zeros(self.AgentCount)

        for t in range(self.T_cycle):
            J = self.MrkvArray[t].shape[0]
            for j in range(J):
                these = np.logical_and(t == self.t_cycle, j == MrkvNow)
                N = np.sum(these)
                if N > 0:
                    IncShkDstnQ = self.IncShkDstn_Q[t - 1][j]
                    PermGroFacNow = self.PermGroFac[t - 1][j]

                    base_draws = base_draws_dict.get((t, j))
                    if base_draws is not None:
                        indices_Q = _cdf_invert(base_draws, IncShkDstnQ.pmv)
                    else:
                        indices_Q = IncShkDstnQ.draw_events(N)

                    PermShkQ[these] = IncShkDstnQ.atoms[0][indices_Q] * PermGroFacNow
                    TranShkQ[these] = IncShkDstnQ.atoms[1][indices_Q]

        # Newborns: redraw from period 0's distribution, as the P side does.
        if np.any(newborn):
            for j in range(self.MrkvArray[0].shape[0]):
                these_nb = np.logical_and(newborn, j == MrkvNow)
                N_new = np.sum(these_nb)
                if N_new == 0:
                    continue
                IncShkDstnQ_0 = self.IncShkDstn_Q[0][j]
                PermGroFacNow = self.PermGroFac[0][j]

                base_new = base_draws_dict.get(("newborn", j))
                if base_new is not None:
                    indices_Q = _cdf_invert(base_new, IncShkDstnQ_0.pmv)
                else:
                    indices_Q = IncShkDstnQ_0.draw_events(N_new)

                PermShkQ[these_nb] = IncShkDstnQ_0.atoms[0][indices_Q] * PermGroFacNow
                TranShkQ[these_nb] = IncShkDstnQ_0.atoms[1][indices_Q]

            if not getattr(self, "NewbornTransShk", False):
                TranShkQ[newborn] = 1.0

        self.shocks_Q["PermShk"] = PermShkQ
        self.shocks_Q["TranShk"] = TranShkQ

    def _transition_Q(self):
        """Compute Q-measure states from Q-shocks.

        Writes every state this transition defines, in the same order and by
        the same formulas as ``IndShockConsumerType.transition``: ``kNrm``
        and ``bNrm`` are stored rather than computed and thrown away, so a
        caller tracking them gets the Q quantity that carries that name
        instead of a stale buffer.
        """
        pLvlPrev = self.state_prev_Q["pLvl"]
        kNrm = self.state_prev_Q["aNrm"]
        RportNow = self.get_Rport()

        pLvlNow = pLvlPrev * self.shocks_Q["PermShk"]
        ReffNow = RportNow / self.shocks_Q["PermShk"]
        bNrmNow = ReffNow * kNrm
        mNrmNow = bNrmNow + self.shocks_Q["TranShk"]

        if "kNrm" in self.state_now_Q:
            self.state_now_Q["kNrm"] = kNrm
        if "bNrm" in self.state_now_Q:
            self.state_now_Q["bNrm"] = bNrmNow
        self.state_now_Q["pLvl"] = pLvlNow
        self.state_now_Q["mNrm"] = mNrmNow

    def _get_controls_Q(self):
        """Evaluate the same cFunc at Q-measure mNrm.

        For Markov models, ``solution[t].cFunc`` is a list indexed by the
        (shared) Markov state.  For IndShock, it's a single function.
        """
        cNrmQ = np.full(self.AgentCount, np.nan)

        if self._is_markov:
            MrkvNow = self.shocks["Mrkv"]
            for t in range(self.T_cycle):
                J = self.MrkvArray[t].shape[0]
                for j in range(J):
                    these = np.logical_and(t == self.t_cycle, j == MrkvNow)
                    if np.any(these):
                        cNrmQ[these] = self.solution[t].cFunc[j](
                            self.state_now_Q["mNrm"][these]
                        )
        else:
            for t in np.unique(self.t_cycle):
                idx = self.t_cycle == t
                if np.any(idx):
                    cNrmQ[idx] = self.solution[t].cFunc(self.state_now_Q["mNrm"][idx])

        self.controls_Q["cNrm"] = cNrmQ

    def _get_poststates_Q(self):
        """Compute Q-measure end-of-period assets and propagate shared states."""
        self.state_now_Q["aNrm"] = self.state_now_Q["mNrm"] - self.controls_Q["cNrm"]
        if "aLvl" in self.state_now:
            self.state_now_Q["aLvl"] = (
                self.state_now_Q["aNrm"] * self.state_now_Q["pLvl"]
            )
        if "PlvlAgg" in self.state_prev_Q:
            self.state_now_Q["PlvlAgg"] = self.state_now.get("PlvlAgg", 1.0)
        # Markov state is shared between P and Q
        if "Mrkv" in self.state_now:
            self.state_now_Q["Mrkv"] = self.state_now["Mrkv"].copy()

    # ------------------------------------------------------------------
    # simulate override: record Q-history
    # ------------------------------------------------------------------

    def simulate(self, sim_periods=None):
        """Extend: record Q-history alongside the base class's P-history.

        The P side is delegated to ``super().simulate()`` one period at a
        time rather than reimplemented here.  An earlier version copied the
        base recording loop and dropped its final ``else`` branch, so any
        tracked variable that is a plain attribute instead of a key in
        ``state_now``/``shocks``/``controls`` (``MPCnow`` is the common one)
        was silently left as NaN whenever dual mode was on.  Delegating keeps
        that class of drift from recurring: turning dual mode on cannot
        change what the P pipeline records, by construction.
        """
        if not self.dual_measure:
            return super().simulate(sim_periods)

        if not hasattr(self, "t_sim"):
            raise Exception(
                "It seems that the simulation variables were not initialize before "
                + "calling simulate(). Call initialize_sim() to initialize the "
                + "variables before calling simulate() again."
            )
        if not hasattr(self, "T_sim"):
            raise Exception(
                "This agent type instance must have the attribute T_sim set to a "
                + "positive integer."
            )
        if sim_periods is not None and self.T_sim < sim_periods:
            raise Exception(
                "To simulate, sim_periods has to be larger than the maximum data "
                + "set size T_sim."
            )

        if sim_periods is None:
            sim_periods = self.T_sim - self.t_sim

        for _ in range(sim_periods):
            # One period of the unmodified P pipeline, including its own
            # history recording and its own t_sim increment.
            super().simulate(1)
            self._record_Q_history(self.t_sim - 1)

        return self.history

    def _record_Q_history(self, t_rec):
        """Record the Q-side counterparts of ``track_vars`` at row ``t_rec``.

        Mirrors the base class's P recording, minus its ``who_dies`` special
        case: mortality is drawn once in the P pipeline and shared, so there
        is no separate Q death mask to record.  A tracked variable with no
        Q counterpart is left at NaN rather than being filled from the P side,
        so ``history_Q`` never reports a P quantity under a Q name.
        """
        for var_name in self.track_vars:
            if var_name in self.state_now_Q:
                value = self.state_now_Q[var_name]
            elif var_name in self.shocks_Q:
                value = self.shocks_Q[var_name]
            elif var_name in self.controls_Q:
                value = self.controls_Q[var_name]
            else:
                continue
            self.history_Q[var_name][t_rec, :] = value

    # ------------------------------------------------------------------
    # Aggregation utilities
    # ------------------------------------------------------------------

    def aggregate_Q(self, var="cNrm", burn=0, N=None, E_pLvl=None, pLvl_factor=None):
        """Compute level-aggregate consumption from Q-measure history.

        The Harmenberg identity gives:

            C_P(t) = N * E_P[p] * F(t) * mean_Q(cNrm_Q(t))

        where ``F(t) = pLvl_factor(t)`` tracks how ``E[p_t]/E[p_ss]``
        evolves (equals 1 in a stationary economy).

        Parameters
        ----------
        var : str
            Variable name in ``history_Q`` to aggregate.  Typically
            ``'cNrm'``.  The variable must be in normalized (per-unit-
            permanent-income) space.
        burn : int
            Burn-in periods to skip.
        N : int or None
            Agent count for level scaling.  Defaults to ``self.AgentCount``.
        E_pLvl : float or None
            Steady-state ``E[p]``.  If None, estimated empirically from
            the P-measure history.
        pLvl_factor : np.ndarray or None
            Scaling ``E[p_t]/E[p_ss]``, one entry per simulated period.
            Pass the full, un-burned series of length ``self.T_sim``: this
            method applies ``[burn:]`` itself, so a pre-burned array of
            length ``T_sim - burn`` would be trimmed a second time.  If
            None, assumed to be 1 for all periods (stationary economy).

        Returns
        -------
        np.ndarray of shape ``(T - burn,)``
            Level-aggregate time series.
        """
        if not self.dual_measure:
            raise ValueError("aggregate_Q requires dual_measure=True")

        Q_nrm = self.history_Q[var][burn:]
        mean_Q = np.nanmean(Q_nrm, axis=1)
        T = len(mean_Q)

        if N is None:
            N = self.AgentCount
        if E_pLvl is None:
            E_pLvl = np.nanmean(self.history["pLvl"][burn:])
        if pLvl_factor is None:
            pLvl_factor = np.ones(T)
        else:
            pLvl_factor = np.asarray(pLvl_factor)[burn:]

        return N * E_pLvl * pLvl_factor * mean_Q


# ======================================================================
# Standalone aggregation helpers
# ======================================================================


#: Names that mean "P-side" when read inside ``get_Rport``.  The Q pipeline
#: mirrors ``shocks`` into ``shocks_Q`` and states into ``state_now_Q``, but
#: nothing mirrors these, so a ``get_Rport`` that reads them returns the P
#: agent's answer no matter which measure is asking.
_P_SIDE_NAMES = frozenset({"state_prev", "state_now", "controls"})


def _get_Rport_reads_p_side(fn, depth=2, seen=None):
    """Names from ``_P_SIDE_NAMES`` that ``fn`` reads, following delegation.

    Static inspection of ``co_names``, the same technique
    ``HARK.simulation.normalization._warn_if_hook_unreachable`` uses to detect
    a ``sim_one_period`` that never reaches ``post_state_hook``.  One level of
    ``return OtherClass.get_Rport(self)`` is followed, because
    ``KinkyPrefConsumerType`` is exactly that and inspecting only its own body
    reports it clean.

    This is a heuristic and is honest about being one: it sees names, not
    dataflow, so a subclass that reaches P-side state through a helper several
    frames down slips past.  It is a backstop against the compositions that
    exist today, not a proof of safety for compositions that do not.
    """
    if seen is None:
        seen = set()
    if fn is None or depth < 0 or fn in seen:
        return set()
    seen.add(fn)
    code = getattr(fn, "__code__", None)
    if code is None:
        return set()
    names = set(code.co_names)
    hits = names & _P_SIDE_NAMES
    for name in names:
        delegate = getattr(fn.__globals__.get(name), "get_Rport", None)
        if delegate is not None:
            hits |= _get_Rport_reads_p_side(delegate, depth - 1, seen)
    return hits


def _cohort_mass_normalizer(LivPrb, T_age):
    """``(1 - LivPrb) / (1 - LivPrb**T_age)``, with the right limit at 1.

    The share of a stationary population that is newborn, when survival is
    ``LivPrb`` each period and everyone dies at ``T_age``.  Two callers need
    it: ``compute_mean_pLvl`` as ``C_norm``, and ``compute_pLvl_factor`` as
    ``delta_eff``, whose ``1 - (L - L**T)/(1 - L**T)`` is this rearranged.

    Both ends are 0 at ``LivPrb == 1``, and the limit is ``1 / T_age``, not
    0: with no mortality the cohorts are equally sized, so newborns are one
    of ``T_age`` of them.  L'Hopital on ``(1 - L) / (1 - L**T)`` gives
    ``1 / (T * L**(T-1))``, which is ``1 / T`` at ``L = 1``.

    This is worth a shared function because the two call sites disagreed
    about the degenerate case in opposite directions: one had no guard and
    divided 0 by 0, the other guarded and returned ``1 - LivPrb``, which is
    0 exactly where the answer is ``1 / T_age``.
    """
    L_T = LivPrb**T_age
    if abs(1.0 - L_T) < 1e-12:
        return 1.0 / T_age
    return (1.0 - LivPrb) / (1.0 - L_T)


def compute_mean_pLvl(agent, g=None):
    """Analytical steady-state E[pLvl] for an infinite-horizon HARK agent.

    Computes the ergodic cross-sectional mean of permanent income
    accounting for mortality-driven turnover and permanent income growth.

    Parameters
    ----------
    agent : AgentType
        Must have attributes ``LivPrb``, ``PermGroFac``, ``T_age``
        (or defaults to 400), and ``pLogInitMean``/``pLogInitStd``.
    g : float or None
        Effective permanent income growth factor per period.
        Defaults to ``agent.PermGroFac[0]`` (scalar or ``[0]`` element).
        In models with unemployment, set ``g = (1-u)*G + u`` where
        ``u`` is the ergodic unemployment rate and ``G = PermGroFac``.

    Returns
    -------
    float
        E[pLvl] in the stationary cross-section.
    """
    LivPrb = agent.LivPrb[0]
    if isinstance(LivPrb, (list, np.ndarray)):
        LivPrb = LivPrb[0]

    if g is None:
        PGF = agent.PermGroFac[0]
        if isinstance(PGF, (list, np.ndarray)):
            PGF = PGF[0]
        g = PGF

    pLogInitMean = getattr(agent, "pLogInitMean", getattr(agent, "pLvlInitMean", 0.0))
    pLogInitStd = getattr(agent, "pLogInitStd", getattr(agent, "pLvlInitStd", 0.0))
    E_pLvl_init = np.exp(pLogInitMean + 0.5 * pLogInitStd**2)

    T_age = getattr(agent, "T_age", 400) or 400

    # Aggregate pLvl over the stationary age distribution. A cohort of age a
    # has survived a periods (mass LivPrb**a) and grown a times (factor g**a),
    # so its contribution scales as (LivPrb * g)**a and the sum over ages
    # 0..T_age-1 is geometric in Lg = LivPrb * g.
    Lg = LivPrb * g
    if abs(Lg - 1.0) < 1e-12:
        # Removable singularity: at Lg == 1 every cohort contributes equally,
        # so the sum is just the number of cohorts. The closed form below is
        # 0/0 here and numerically unstable nearby.
        geo_sum = float(T_age)
    else:
        geo_sum = (1.0 - Lg**T_age) / (1.0 - Lg)
    # Divides out the total cohort mass, leaving a per-capita mean.
    C_norm = _cohort_mass_normalizer(LivPrb, T_age)

    return E_pLvl_init * g * C_norm * geo_sum


def compute_pLvl_factor(agent, unemployment_path, g_base=None):
    """Compute pLvl_factor(t) = E[p_t] / E[p_ss] along a shock path.

    In periods with elevated unemployment, average permanent income
    growth slows because unemployed agents get PermShk = 1 (no growth)
    while employed agents grow at PermGroFac.  This AR(1) recurrence
    tracks the deviation from steady state:

        F(t+1) = (1 - delta) * g_rec(t) * F(t)
                 + [1 - (1 - delta) * g_base]

    where delta = effective death rate, g_rec(t) = (1-u_t)*G + u_t,
    and g_base = (1-u_ss)*G + u_ss.

    Parameters
    ----------
    agent : AgentType
        Must have ``PermGroFac``, ``LivPrb``, ``T_age``.
    unemployment_path : array-like of shape (T,)
        Unemployment rate ``u_t`` at each period.  For the baseline
        (no shock), pass a constant array at the ergodic rate.
    g_base : float or None
        Steady-state growth factor.  If None, uses the first entry
        of ``unemployment_path`` to compute it.

    Returns
    -------
    np.ndarray of shape (T,)
        pLvl_factor time series, starting at 1.0.
    """
    u_path = np.asarray(unemployment_path, dtype=float)
    T = len(u_path)

    G = agent.PermGroFac[0]
    if isinstance(G, (list, np.ndarray)):
        G = G[0]

    LivPrb = agent.LivPrb[0]
    if isinstance(LivPrb, (list, np.ndarray)):
        LivPrb = LivPrb[0]

    T_age = getattr(agent, "T_age", 400) or 400
    # Effective death rate accounting for forced death at T_age.
    # 1 - (L - L**T)/(1 - L**T) is (1 - L)/(1 - L**T) rearranged, so this is
    # the same quantity compute_mean_pLvl calls C_norm.
    delta_eff = _cohort_mass_normalizer(LivPrb, T_age)

    if g_base is None:
        g_base = (1.0 - u_path[0]) * G + u_path[0]

    F = np.ones(T)
    for t in range(1, T):
        g_rec = (1.0 - u_path[t]) * G + u_path[t]
        F[t] = (1.0 - delta_eff) * g_rec * F[t - 1] + (1.0 - (1.0 - delta_eff) * g_base)

    return F

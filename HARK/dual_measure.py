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

import numpy as np
from copy import copy

from HARK.distributions.discrete import DiscreteDistribution


def make_Q_measure_dstn(dstn):
    """Reweight a DiscreteDistribution by psi/E[psi] (Harmenberg neutral measure).

    Parameters
    ----------
    dstn : DiscreteDistribution
        Joint (PermShk, TranShk) distribution under the physical measure.
        ``dstn.atoms[0]`` must be the permanent shock values.

    Returns
    -------
    DiscreteDistribution
        New distribution with Q-measure probabilities and the same atoms.
        If the permanent shock has zero variance the original distribution
        is returned unchanged.
    """
    perm_atoms = dstn.atoms[0]
    E_perm = np.dot(dstn.pmv, perm_atoms)
    if E_perm <= 0 or np.std(perm_atoms) < 1e-12:
        return dstn
    Q_pmv = dstn.pmv * perm_atoms / E_perm
    Q_pmv /= Q_pmv.sum()
    return DiscreteDistribution(Q_pmv, dstn.atoms, seed=dstn.seed)


def _cdf_invert(base_draws, pmv):
    """Map uniform draws to atom indices via CDF inversion.

    Parameters
    ----------
    base_draws : np.ndarray of shape (N,)
        Uniform [0, 1) random numbers.
    pmv : np.ndarray
        Probability mass vector.

    Returns
    -------
    np.ndarray of int
        Indices into the atom arrays.
    """
    return np.searchsorted(np.cumsum(pmv), base_draws)


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
    ``sim_one_period`` and ``simulate`` via MRO, calling ``super()`` for the
    P-pipeline.
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
        """
        self.IncShkDstn_Q = []
        for period_dstn in self.IncShkDstn:
            if isinstance(period_dstn, (list, tuple)):
                self.IncShkDstn_Q.append(
                    [make_Q_measure_dstn(d) for d in period_dstn]
                )
            else:
                self.IncShkDstn_Q.append(make_Q_measure_dstn(period_dstn))
        self.dual_measure = True

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
    # sim_one_period override
    # ------------------------------------------------------------------

    def sim_one_period(self):
        """Run the P-pipeline, then the Q-pipeline before time advancement.

        We cannot simply call ``super().sim_one_period()`` and append the
        Q-step, because the base class increments ``t_age`` and ``t_cycle``
        at the end.  The Q-pipeline needs these at their pre-increment values
        (same as the P-pipeline used).  So we replicate the base-class logic
        with the Q-step inserted before the time advancement.
        """
        if not hasattr(self, "solution"):
            raise Exception(
                "Model instance does not have a solution stored. "
                "To simulate, run the `solve()` method first."
            )

        # --- P-pipeline (mirrors AgentType.sim_one_period) ---
        self.get_mortality()

        for var in self.state_now:
            self.state_prev[var] = self.state_now[var]
            if isinstance(self.state_now[var], np.ndarray):
                self.state_now[var] = np.empty(self.AgentCount)
            else:
                pass

        if self.read_shocks:
            self.read_shocks_from_history()
        else:
            self.get_shocks()
        self.get_states()
        self.post_state_hook()
        self.get_controls()
        self.get_poststates()

        # --- Q-pipeline (while t_age / t_cycle still match P's view) ---
        if self.dual_measure:
            self._step_Q_measure()

        # --- Advance time (same as AgentType.sim_one_period) ---
        self.t_age = self.t_age + 1
        self.t_cycle = self.t_cycle + 1
        self.t_cycle[self.t_cycle == self.T_cycle] = 0

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
        """Copy ``state_now_Q`` -> ``state_prev_Q``, blank ``state_now_Q``."""
        for var in self.state_now_Q:
            self.state_prev_Q[var] = self.state_now_Q[var]
            if isinstance(self.state_now_Q[var], np.ndarray):
                self.state_now_Q[var] = np.empty(self.AgentCount)
            # scalar (aggregate) vars are kept as-is

    @property
    def _is_markov(self):
        """True when the agent has Markov-indexed income distributions."""
        return "Mrkv" in getattr(self, "shocks", {})

    def _draw_Q_shocks(self):
        """Draw Q-measure shocks using the base uniforms saved by get_shocks().

        The P-pipeline's ``get_shocks()`` stores ``self._base_shock_draws``
        — a dict whose keys are either scalar ``t_cycle`` values (IndShock)
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
            t_key = t - 1 if self.cycles == 1 else t
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
            PermShkQ[newborn] = (
                IncShkDstnQ_0.atoms[0][indices_Q] * PermGroFacNow
            )
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

                    PermShkQ[these] = (
                        IncShkDstnQ.atoms[0][indices_Q] * PermGroFacNow
                    )
                    TranShkQ[these] = IncShkDstnQ.atoms[1][indices_Q]

        # Newborns: deterministic PermShk = PermGroFac, TranShk = 1 (same as P)
        for j in range(self.MrkvArray[0].shape[0]):
            these_nb = np.logical_and(newborn, j == MrkvNow)
            if np.any(these_nb):
                PermShkQ[these_nb] = self.PermGroFac[0][j]
        TranShkQ[newborn] = 1.0

        self.shocks_Q["PermShk"] = PermShkQ
        self.shocks_Q["TranShk"] = TranShkQ

    def _transition_Q(self):
        """Compute Q-measure states: pLvl_Q, mNrm_Q from Q-shocks."""
        pLvlPrev = self.state_prev_Q["pLvl"]
        kNrm = self.state_prev_Q["aNrm"]
        RportNow = self.get_Rport()

        pLvlNow = pLvlPrev * self.shocks_Q["PermShk"]
        ReffNow = RportNow / self.shocks_Q["PermShk"]
        bNrmNow = ReffNow * kNrm
        mNrmNow = bNrmNow + self.shocks_Q["TranShk"]

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
                    cNrmQ[idx] = self.solution[t].cFunc(
                        self.state_now_Q["mNrm"][idx]
                    )

        self.controls_Q["cNrm"] = cNrmQ

    def _get_poststates_Q(self):
        """Compute Q-measure end-of-period assets and propagate shared states."""
        self.state_now_Q["aNrm"] = (
            self.state_now_Q["mNrm"] - self.controls_Q["cNrm"]
        )
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
        """Extend: record Q-history alongside P-history each period.

        The base ``simulate()`` calls ``sim_one_period()`` (which now includes
        ``_step_Q_measure()``) and records P-history.  We override to also
        record Q-history in the same loop.
        """
        if not self.dual_measure:
            return super().simulate(sim_periods)

        if not hasattr(self, "t_sim"):
            raise Exception(
                "Call initialize_sim() before simulate()."
            )
        if sim_periods is None:
            sim_periods = self.T_sim - self.t_sim

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            for t in range(sim_periods):
                self.sim_one_period()

                # Record P-history (same as AgentType.simulate)
                for var_name in self.track_vars:
                    if var_name in self.state_now:
                        self.history[var_name][self.t_sim, :] = self.state_now[
                            var_name
                        ]
                    elif var_name in self.shocks:
                        self.history[var_name][self.t_sim, :] = self.shocks[
                            var_name
                        ]
                    elif var_name in self.controls:
                        self.history[var_name][self.t_sim, :] = self.controls[
                            var_name
                        ]
                    else:
                        if var_name == "who_dies" and self.t_sim > 1:
                            self.history[var_name][self.t_sim - 1, :] = (
                                getattr(self, var_name)
                            )

                # Record Q-history
                for var_name in self.track_vars:
                    if var_name in self.state_now_Q:
                        self.history_Q[var_name][self.t_sim, :] = (
                            self.state_now_Q[var_name]
                        )
                    elif var_name in self.shocks_Q:
                        self.history_Q[var_name][self.t_sim, :] = (
                            self.shocks_Q[var_name]
                        )
                    elif var_name in self.controls_Q:
                        self.history_Q[var_name][self.t_sim, :] = (
                            self.controls_Q[var_name]
                        )

                self.t_sim += 1

        return self.history

    # ------------------------------------------------------------------
    # Aggregation utilities
    # ------------------------------------------------------------------

    def aggregate_Q(self, var="cNrm", burn=0, N=None, E_pLvl=None,
                    pLvl_factor=None):
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
            Shape ``(T,)`` scaling ``E[p_t]/E[p_ss]``.  If None, assumed
            to be 1 for all periods (stationary economy).

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

    pLogInitMean = getattr(agent, "pLogInitMean",
                           getattr(agent, "pLvlInitMean", 0.0))
    pLogInitStd = getattr(agent, "pLogInitStd",
                          getattr(agent, "pLvlInitStd", 0.0))
    E_pLvl_init = np.exp(pLogInitMean + 0.5 * pLogInitStd ** 2)

    T_age = getattr(agent, "T_age", 400) or 400

    Lg = LivPrb * g
    if abs(Lg - 1.0) < 1e-12:
        geo_sum = float(T_age)
    else:
        geo_sum = (1.0 - Lg ** T_age) / (1.0 - Lg)
    C_norm = (1.0 - LivPrb) / (1.0 - LivPrb ** T_age)

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
    # Effective death rate accounting for forced death at T_age
    L_T = LivPrb ** T_age
    delta_eff = 1.0 - (LivPrb - L_T) / (1.0 - L_T) if abs(1 - L_T) > 1e-12 else 1.0 - LivPrb

    if g_base is None:
        g_base = (1.0 - u_path[0]) * G + u_path[0]

    F = np.ones(T)
    for t in range(1, T):
        g_rec = (1.0 - u_path[t]) * G + u_path[t]
        F[t] = ((1.0 - delta_eff) * g_rec * F[t - 1]
                + (1.0 - (1.0 - delta_eff) * g_base))

    return F

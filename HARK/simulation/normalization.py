"""Mixins for simulation variance reduction via moment normalization.

These mixins adjust cross-sectional distributions during simulation so that
empirical moments match their analytical values, eliminating sampling noise
in aggregates without requiring special population sizes.

Both mixins are strictly opt-in: composing them into a class and leaving
their switches at the defaults (``normalize_shocks=False``,
``normalize_pLvl=False``) changes nothing about a simulation.

Usage::

    from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
    from HARK.simulation.normalization import (
        PermanentIncomeNormalizationMixin,
        ShockNormalizationMixin,
    )

    class NormalizedIndShock(
        ShockNormalizationMixin,
        PermanentIncomeNormalizationMixin,
        IndShockConsumerType,
    ):
        pass

    agent = NormalizedIndShock(normalize_pLvl=True, normalize_shocks=True, ...)

Both mixins hook into HARK's own extension points: ``get_shocks`` (via
``super()``) and ``AgentType.post_state_hook``, the no-op invoked by
``sim_one_period`` between ``get_states()`` and ``get_controls()``. Neither
overrides ``sim_one_period`` itself, so a model with its own
``sim_one_period`` (e.g. ``ConsRiskyContribConsumerType``, whose staged
structure forces one) is not shadowed. Such a model may not call
``post_state_hook`` at all, and ``initialize_sim`` warns when that is the
case rather than letting normalization silently never run.

Growth is inside the permanent shock
------------------------------------
HARK folds the expected growth factor into the permanent shock array:
``PermShkNow = psi * PermGroFac``, with psi the mean-one innovation (see
``IndShockConsumerType.get_shocks``, and ``ConsMarkovModel`` per discrete
state). The cross-sectional mean of ``shocks["PermShk"]`` is therefore
``PermGroFac``, not 1.0, and ``ShockNormalizationMixin`` targets that.
Rescaling ``PermShk`` to 1.0 would delete permanent income growth rather
than sampling noise, because ``transition()`` applies the array directly as
``pLvl = pLvlPrev * PermShk``. Transitory shocks genuinely are mean-one, so
1.0 remains the target there.

Age awareness
-------------
Targets are computed period by period from the income process actually used
in each simulated period, so life-cycle calibrations with age-varying
``PermGroFac`` and ``PermShkDstn`` (HARK ships several) get the right
profile rather than period 0's parameters extrapolated. The indexing
mirrors ``get_shocks``: an agent at ``t_cycle == j`` draws from index
``(j - 1) % T_cycle``, except in its birth period, which is special-cased to
index 0.

Design notes
------------
* **Markov-capable.** Models whose per-period growth factor is a *vector*
  over discrete (Markov) states are supported: the analytical drift uses
  the stationary-distribution-weighted mean log growth, and the moment
  targeting automatically degrades to MEAN-ONLY (see below).
* **Why mean-only under state-dependent growth.** With heterogeneous
  growth histories (e.g. employed vs unemployed spells), the cross-
  sectional variance of ``log pLvl`` at a given age exceeds the pure
  shock-accumulation term by the variance contributed by heterogeneous
  Markov paths. The simple analytical variance formula used here counts
  only the shock term, so rescaling the cross-sectional spread to it
  would *shrink genuine heterogeneity*. Shifting the mean is exact and
  safe; therefore ``pLvl_norm_moments="auto"`` applies mean-and-std for
  scalar-growth models and mean-only for vector-growth models. Models
  with a correct model-specific variance target can override
  ``_analytical_log_pLvl_moments`` and set
  ``pLvl_norm_moments="mean_and_std"``.
* **Stationarity caveat.** The stationary-weighted drift is the exact
  population drift only once the discrete state has reached its stationary
  distribution. In a life-cycle Markov model (``T_cycle > 1`` with vector
  growth) young cohorts have not, so the mean target is approximate and
  ``initialize_sim`` warns.
"""

import warnings

import numpy as np


__all__ = ["ShockNormalizationMixin", "PermanentIncomeNormalizationMixin"]


def _stationary_distribution(transition_matrix):
    """Stationary distribution of a row-stochastic transition matrix.

    Parameters
    ----------
    transition_matrix : np.ndarray of shape (S, S)
        Row-stochastic Markov transition matrix.

    Returns
    -------
    np.ndarray of shape (S,)
        The stationary probability vector (left eigenvector for
        eigenvalue 1, normalized to sum to one).
    """
    eigenvalues, eigenvectors = np.linalg.eig(np.asarray(transition_matrix).T)
    idx = int(np.argmin(np.abs(eigenvalues - 1.0)))
    pi = np.real(eigenvectors[:, idx])
    pi = np.abs(pi)
    return pi / pi.sum()


def _warn_once(agent, key, message):
    """Emit ``message`` the first time ``key`` is raised for ``agent``.

    Normalization runs every simulated period, so an unguarded warning
    about a persistent condition would fire hundreds of times per run and
    train users to filter the whole module out.
    """
    seen = getattr(agent, "_normalization_warned", None)
    if seen is None:
        seen = set()
        agent._normalization_warned = seen
    if key in seen:
        return
    seen.add(key)
    warnings.warn(message, stacklevel=3)


class _NormalizationIndexMixin:
    """Shared per-agent indexing into the income process.

    Both mixins need to know which period's income process each agent is
    drawing from, and both are commonly composed onto the same class, so
    the logic lives here rather than being duplicated.
    """

    def _income_dstn_index(self):
        """Per-agent index into ``IncShkDstn`` / ``PermGroFac`` for this period.

        Mirrors ``IndShockConsumerType.get_shocks``: agents at ``t_cycle == j``
        use index ``j - 1`` (so ``t_cycle == 0`` wraps to the last period),
        while newborns are special-cased to index 0. Valid only before
        ``t_age`` and ``t_cycle`` are advanced, i.e. anywhere inside
        ``sim_one_period``.
        """
        T_cycle = int(getattr(self, "T_cycle", 1)) or 1
        t_cycle = np.asarray(self.t_cycle, dtype=int)
        idx = (t_cycle - 1) % T_cycle
        newborn = np.asarray(self.t_age, dtype=int) == 0
        return np.where(newborn, 0, idx)


class ShockNormalizationMixin(_NormalizationIndexMixin):
    """Opt-in normalization of income shocks to pin cross-sectional means.

    After ``get_shocks()`` draws ``PermShk`` and ``TranShk`` for the
    population, rescales each so that the cross-sectional mean equals its
    theoretical value: ``PermGroFac`` for ``PermShk`` (HARK folds growth
    into that array; see the module docstring) and 1.0 for ``TranShk``.
    This makes the aggregate effect of sampling noise exact in every
    period, regardless of population size, without touching the
    deterministic growth trend.

    Agents are grouped by the income process they actually drew from --
    the period index from :meth:`_income_dstn_index` and, when the model
    carries one, the discrete (Markov) state -- and normalized within
    group, since only within a group is the target common.

    Attributes
    ----------
    normalize_shocks : bool
        When True, shock normalization is applied. Default False (no
        behavior change).
    """

    normalize_shocks = False

    def get_shocks(self):
        """Draw shocks, then normalize cross-sectional means if enabled."""
        super().get_shocks()
        if not getattr(self, "normalize_shocks", False):
            return
        self._normalize_shock_means()

    def read_shocks_from_history(self):
        """Replay stored shocks, warning that normalization is bypassed."""
        if getattr(self, "normalize_shocks", False):
            _warn_once(
                self,
                "read_shocks",
                "ShockNormalizationMixin: normalize_shocks=True has no effect "
                "while read_shocks is True. Shocks are replayed from "
                "self.shock_history, which bypasses get_shocks() entirely. "
                "Normalize when the history is generated (make_shock_history) "
                "if exact means are wanted in the replay.",
            )
        super().read_shocks_from_history()

    def _shock_group_labels(self):
        """Integer group labels: agents sharing an income process and target."""
        columns = [self._income_dstn_index()]
        if "Mrkv" in getattr(self, "shocks", {}):
            columns.append(np.asarray(self.shocks["Mrkv"]).astype(int))
        keys = np.column_stack(columns)
        return np.unique(keys, axis=0, return_inverse=True)[1].reshape(-1)

    def _perm_shk_mean_target(self):
        """Per-agent cross-sectional mean of ``shocks["PermShk"]``.

        This is ``PermGroFac`` for the period each agent drew from, because
        HARK multiplies the mean-one innovation by the growth factor before
        storing it (see the module docstring). Returns None when
        ``PermGroFac`` is missing or has a shape this mixin cannot resolve,
        in which case ``PermShk`` is left untouched rather than normalized
        to a guess.
        """
        PermGroFac = getattr(self, "PermGroFac", None)
        if PermGroFac is None:
            return None
        idx = self._income_dstn_index()
        mrkv = None
        if "Mrkv" in getattr(self, "shocks", {}):
            mrkv = np.asarray(self.shocks["Mrkv"]).astype(int)

        target = np.empty(len(idx), dtype=float)
        for i in np.unique(idx):
            selected = idx == i
            try:
                entry = np.asarray(PermGroFac[int(i)], dtype=float).flatten()
            except (IndexError, KeyError, TypeError, ValueError):
                return None
            if entry.size == 1:
                target[selected] = entry[0]
            elif mrkv is not None and mrkv[selected].max() < entry.size:
                target[selected] = entry[mrkv[selected]]
            else:
                return None
        return target

    def _normalize_shock_means(self):
        """Rescale ``PermShk`` and ``TranShk`` so group means hit their targets.

        Within each group from :meth:`_shock_group_labels`, multiplies the
        shock array by ``target / empirical_mean``. Groups of fewer than two
        agents, and groups whose empirical mean is numerically zero, are
        skipped with a warning: the class advertises exactness "regardless of
        population size", and a silent skip would quietly break that promise.
        """
        shocks = getattr(self, "shocks", {})
        labels = self._shock_group_labels()

        targets = {"TranShk": np.ones(len(labels))}
        perm_target = self._perm_shk_mean_target()
        if perm_target is not None:
            targets["PermShk"] = perm_target
        elif "PermShk" in shocks:
            _warn_once(
                self,
                "no_perm_target",
                "ShockNormalizationMixin: could not resolve PermGroFac into a "
                "per-agent mean for PermShk, so PermShk is left un-normalized. "
                "Override _perm_shk_mean_target() for this model.",
            )

        # Models that pin newborn TranShk to exactly 1.0 (NewbornTransShk
        # False) are making a deliberate choice; rescaling would undo it.
        held_out = {}
        if not getattr(self, "NewbornTransShk", True):
            held_out["TranShk"] = np.asarray(self.t_age, dtype=int) == 0

        skipped = 0
        for name, target in targets.items():
            if name not in shocks:
                continue
            shock_arr = shocks[name]
            exclude = held_out.get(name)

            for g in np.unique(labels):
                mask = labels == g
                if exclude is not None:
                    mask = np.logical_and(mask, ~exclude)
                if mask.sum() < 2:
                    skipped += int(mask.sum())
                    continue
                empirical_mean = float(np.mean(shock_arr[mask]))
                if abs(empirical_mean) < 1e-16:
                    skipped += int(mask.sum())
                    continue
                shock_arr[mask] *= float(np.mean(target[mask])) / empirical_mean

        if skipped:
            _warn_once(
                self,
                "small_shock_group",
                "ShockNormalizationMixin: skipped normalization for groups "
                "with fewer than 2 agents or a numerically zero mean "
                f"({skipped} agent-slots in the first such period). Cross-"
                "sectional shock means are NOT exact for those agents; raise "
                "AgentCount or coarsen the grouping if exactness is needed.",
            )


class PermanentIncomeNormalizationMixin(_NormalizationIndexMixin):
    """Opt-in per-cohort ``pLvl`` normalization for variance reduction.

    Within each age cohort ``k``, adjusts ``log(pLvl)`` so that its
    cross-sectional moments match analytical values. The affine transform
    in log space preserves the rank ordering of agents' permanent incomes,
    so wealth-income correlations are unaffected. Normalized state
    variables (``mNrm``, ``bNrm``) are rescaled inversely so that level
    quantities (``mLvl = mNrm * pLvl``) are preserved.

    Attributes
    ----------
    normalize_pLvl : bool
        When True, normalization runs each simulated period. Default
        False (no behavior change).
    pLvl_norm_moments : str
        ``"auto"`` (default): mean-and-std for scalar-growth models,
        mean-only for vector-(Markov-)growth models. ``"mean_and_std"``
        or ``"mean"`` force the respective behavior. See the module
        docstring for why mean-only is the safe default under
        state-dependent growth.
    """

    normalize_pLvl = False
    pLvl_norm_moments = "auto"
    _pLvl_norm_adjust_vars = ("mNrm", "bNrm")
    _pLvl_norm_min_cohort = 5

    # ------------------------------------------------------------------
    # Analytical targets
    # ------------------------------------------------------------------

    def _growth_is_vector(self):
        """True when any period's growth factor is state-dependent."""
        for pgf in self.PermGroFac:
            if isinstance(pgf, (list, tuple, np.ndarray)) and np.size(pgf) > 1:
                return True
        return False

    def _permshk_log_moments(self, t=0):
        """(E[log psi], Var[log psi]) of period ``t``'s permanent innovation.

        Computed directly from the discrete approximation's atoms, which
        is exact for the distribution actually simulated. These are the
        moments of the mean-one innovation only; the growth factor enters
        separately through :meth:`_effective_log_PermGroFac`. For
        state-dependent shock distributions (a list per Markov state),
        moments are weighted by the stationary distribution of
        ``MrkvArray[t]``.
        """
        dstn = self.PermShkDstn[t]
        if isinstance(dstn, (list, tuple)):
            weights = self._stationary_weights(len(dstn), t)
            first = np.zeros(len(dstn))
            second = np.zeros(len(dstn))
            for s, d in enumerate(dstn):
                log_atoms = np.log(d.atoms.flatten())
                first[s] = np.dot(d.pmv, log_atoms)
                second[s] = np.dot(d.pmv, log_atoms**2)
            e1 = float(np.dot(weights, first))
            e2 = float(np.dot(weights, second))
        else:
            log_atoms = np.log(dstn.atoms.flatten())
            e1 = float(np.dot(dstn.pmv, log_atoms))
            e2 = float(np.dot(dstn.pmv, log_atoms**2))
        return e1, max(e2 - e1**2, 0.0)

    def _stationary_weights(self, n_states, t=0):
        """Stationary weights over Markov states, or uniform if absent."""
        mrkv = getattr(self, "MrkvArray", None)
        if mrkv is not None:
            transition = mrkv[t] if isinstance(mrkv, (list, tuple)) else mrkv
            transition = np.asarray(transition)
            if transition.shape == (n_states, n_states):
                return _stationary_distribution(transition)
        _warn_once(
            self,
            "no_mrkv_array",
            "PermanentIncomeNormalizationMixin: state-dependent growth "
            "without a matching MrkvArray; using uniform state weights "
            "for the analytical pLvl drift.",
        )
        return np.full(n_states, 1.0 / n_states)

    def _effective_log_PermGroFac(self, t=0):
        """Per-period drift of ``E[log pLvl]`` from period ``t``'s growth factor.

        Scalar growth: ``log(PermGroFac[t])``. Vector (Markov) growth:
        the stationary-distribution-weighted mean of ``log(G_s)``, the
        exact drift of the population mean of ``log pLvl`` for agents
        moving on the chain at stationarity. Override for model-specific
        targets (e.g. level-targeting instead of log-mean-targeting).
        """
        pgf = np.asarray(self.PermGroFac[t], dtype=float).flatten()
        if pgf.size == 1:
            return float(np.log(pgf[0]))
        weights = self._stationary_weights(pgf.size, t)
        return float(np.dot(weights, np.log(pgf)))

    def _log_pLvl_step_moments(self, t):
        """(mean, variance) of one period's increment to ``log pLvl``.

        The simulated permanent shock is ``psi * PermGroFac``, so the log
        increment has mean ``E[log psi] + log G`` and variance
        ``Var[log psi]``. Cached per period index; the cache is cleared by
        :meth:`initialize_sim`.
        """
        cache = getattr(self, "_pLvl_norm_step_cache", None)
        if cache is None:
            cache = {}
            self._pLvl_norm_step_cache = cache
        if t not in cache:
            e_log_psi, var_log_psi = self._permshk_log_moments(t)
            cache[t] = (e_log_psi + self._effective_log_PermGroFac(t), var_log_psi)
        return cache[t]

    def _income_dstn_index_history(self, age_k):
        """Income-process indices used in every period from birth to ``age_k``.

        Normalization runs inside ``post_state_hook``, before
        ``sim_one_period`` advances ``t_age``, so an agent reading
        ``t_age == k`` has already been through ``k + 1`` transitions, each
        one multiplying ``pLvl`` by ``psi * PermGroFac``. The sequence
        therefore has ``k + 1`` entries. Period ``j`` uses index
        ``(j - 1) % T_cycle``, except the birth period, which
        ``get_shocks`` special-cases to index 0. Newborns are NOT exempt
        from the permanent shock: ``get_shocks`` redraws a random ``PermShk``
        for them and pins only ``TranShk``.
        """
        T_cycle = int(getattr(self, "T_cycle", 1)) or 1
        return [0] + [(j - 1) % T_cycle for j in range(1, int(age_k) + 1)]

    def _analytical_log_pLvl_moments(self, age_k):
        """Analytical ``(mu, sigma)`` of ``log(pLvl)`` at age ``age_k``.

        Accumulates the per-period log increments over the agent's realized
        income-process history (:meth:`_income_dstn_index_history`)::

            mu_k    = pLogInitMean + sum_j (log G_j + E[log psi_j])
            sigma_k = sqrt(pLogInitStd**2 + sum_j Var[log psi_j])

        Summing period by period rather than multiplying period 0's moments
        by ``k`` is what makes life-cycle calibrations with age-varying
        ``PermGroFac`` or ``PermShkStd`` come out right. Under vector growth
        ``sigma_k`` omits the Markov-path variance contribution, which is
        exactly why the ``"auto"`` moments mode is mean-only there.

        Parameters
        ----------
        age_k : int
            Periods since birth for the cohort, as read from ``t_age``
            inside ``post_state_hook``.

        Returns
        -------
        mu_k : float
        sigma_k : float
        """
        mu_k = float(getattr(self, "pLogInitMean", getattr(self, "pLvlInitMean", 0.0)))
        p_init_std = float(
            getattr(self, "pLogInitStd", getattr(self, "pLvlInitStd", 0.0))
        )
        var_k = p_init_std**2
        for t in self._income_dstn_index_history(age_k):
            step_mean, step_var = self._log_pLvl_step_moments(t)
            mu_k += step_mean
            var_k += step_var
        return float(mu_k), float(np.sqrt(var_k))

    def _resolved_moments_mode(self):
        mode = getattr(self, "pLvl_norm_moments", "auto")
        if mode == "auto":
            return "mean" if self._growth_is_vector() else "mean_and_std"
        if mode not in ("mean", "mean_and_std"):
            raise ValueError(f"Unknown pLvl_norm_moments: {mode!r}")
        return mode

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def post_sim_normalize_pLvl(self):
        """Normalize ``pLvl`` to the analytical per-cohort log-moments.

        Applies the resolved moments mode per age cohort, then rescales
        the variables in ``_pLvl_norm_adjust_vars`` by the inverse pLvl
        ratio so level quantities are preserved. Cohorts smaller than
        ``_pLvl_norm_min_cohort``, and cohorts whose members disagree about
        ``t_cycle`` (so their income history is not the one
        :meth:`_income_dstn_index_history` reconstructs), are left alone
        with a warning rather than normalized to a target that does not
        apply to them.
        """
        if not getattr(self, "normalize_pLvl", False):
            return

        mode = self._resolved_moments_mode()
        log_p = np.log(np.maximum(self.state_now["pLvl"], 1e-16))
        T_cycle = int(getattr(self, "T_cycle", 1)) or 1
        t_cycle = np.asarray(self.t_cycle, dtype=int)
        skipped_small = 0
        skipped_mixed = 0

        for k in np.unique(self.t_age):
            mask = self.t_age == k
            if mask.sum() < self._pLvl_norm_min_cohort:
                skipped_small += int(mask.sum())
                continue
            if T_cycle > 1 and np.any(t_cycle[mask] != int(k) % T_cycle):
                skipped_mixed += int(mask.sum())
                continue

            mu_k, sigma_k = self._analytical_log_pLvl_moments(k)
            mu_hat = np.mean(log_p[mask])

            if mode == "mean":
                log_p[mask] += mu_k - mu_hat
            else:
                sigma_hat = np.std(log_p[mask])
                if sigma_hat > 1e-10:
                    log_p[mask] = mu_k + (sigma_k / sigma_hat) * (log_p[mask] - mu_hat)
                else:
                    log_p[mask] = mu_k
                    _warn_once(
                        self,
                        "degenerate_cohort",
                        "PermanentIncomeNormalizationMixin: a cohort had "
                        "numerically zero cross-sectional spread in log pLvl, "
                        "so it was collapsed to the analytical mean and its "
                        "std target was NOT imposed. Cross-sectional "
                        "dispersion is not exact for that cohort.",
                    )

        if skipped_small:
            _warn_once(
                self,
                "small_cohort",
                "PermanentIncomeNormalizationMixin: skipped cohorts smaller "
                f"than _pLvl_norm_min_cohort={self._pLvl_norm_min_cohort} "
                f"({skipped_small} agents in the first such period). Their "
                "pLvl moments are NOT exact; raise AgentCount or lower "
                "_pLvl_norm_min_cohort if exactness is needed.",
            )
        if skipped_mixed:
            _warn_once(
                self,
                "mixed_t_cycle_cohort",
                "PermanentIncomeNormalizationMixin: skipped cohorts whose "
                "agents do not all share t_cycle == t_age % T_cycle "
                f"({skipped_mixed} agents in the first such period). The "
                "analytical target assumes agents are born at t_cycle == 0 "
                "and age in lockstep with the cycle; override "
                "_income_dstn_index_history for models that stagger entry.",
            )

        pLvl_old = self.state_now["pLvl"].copy()
        self.state_now["pLvl"][:] = np.exp(log_p)
        ratio = pLvl_old / np.maximum(self.state_now["pLvl"], 1e-16)
        for var in self._pLvl_norm_adjust_vars:
            if var in self.state_now and isinstance(self.state_now[var], np.ndarray):
                self.state_now[var] *= ratio

    # ------------------------------------------------------------------
    # Wiring (via AgentType.post_state_hook; no HARK.core changes required)
    # ------------------------------------------------------------------

    def initialize_sim(self):
        """Reset the moment cache and check that the hook will actually fire."""
        super().initialize_sim()
        self._pLvl_norm_step_cache = {}
        if not getattr(self, "normalize_pLvl", False):
            return
        self._warn_if_hook_unreachable()
        if int(getattr(self, "T_cycle", 1)) > 1 and self._growth_is_vector():
            _warn_once(
                self,
                "lifecycle_markov",
                "PermanentIncomeNormalizationMixin: state-dependent growth in "
                "a life-cycle model (T_cycle > 1). The analytical mean target "
                "weights growth by the chain's STATIONARY distribution, which "
                "young cohorts have not reached, so the target is approximate "
                "and normalizing to it introduces bias. Override "
                "_effective_log_PermGroFac with a cohort-correct drift, or "
                "leave normalize_pLvl off.",
            )

    def _warn_if_hook_unreachable(self):
        """Warn when this class's ``sim_one_period`` never calls the hook.

        ``AgentType.sim_one_period`` invokes ``post_state_hook`` between
        ``get_states`` and ``get_controls``, but a model with a bespoke
        pipeline (``ConsRiskyContribConsumerType`` and the Monte Carlo
        simulators) overrides ``sim_one_period`` without it, and
        normalization would then never run at all. Inspecting the resolved
        function's code object catches that at setup instead of leaving a
        silently inert switch.
        """
        func = getattr(type(self), "sim_one_period", None)
        code = getattr(func, "__code__", None)
        if code is None or "post_state_hook" in code.co_names:
            return
        _warn_once(
            self,
            "hook_unreachable",
            "PermanentIncomeNormalizationMixin: normalize_pLvl=True but "
            f"{type(self).__name__}.sim_one_period does not call "
            "post_state_hook(), so normalization will never run. Wire "
            "post_sim_normalize_pLvl() into that model's own pipeline.",
        )

    def post_state_hook(self):
        """Run the normalization; chain to any base-class hook first."""
        sup = getattr(super(), "post_state_hook", None)
        if sup is not None:
            sup()
        self.post_sim_normalize_pLvl()

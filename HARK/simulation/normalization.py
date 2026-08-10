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

Composing with :class:`HARK.dual_measure.DualMeasureMixin`: put
``DualMeasureMixin`` FIRST in the MRO. Its ``sim_one_period`` invokes
``post_state_hook()`` at the right point, which chains to this module's
normalization::

    class DualNormalized(
        DualMeasureMixin, PermanentIncomeNormalizationMixin, IndShockConsumerType
    ):
        pass

Design notes
------------
* **Self-contained.** ``PermanentIncomeNormalizationMixin`` carries its own
  ``sim_one_period`` (replicating ``AgentType.sim_one_period`` with the
  normalization inserted between ``get_states`` and ``get_controls``, so
  that controls are computed from normalized states). It requires no
  changes to ``HARK.core``.
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
"""

import warnings

import numpy as np


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


class ShockNormalizationMixin:
    """Opt-in normalization of income shocks to pin cross-sectional means.

    After ``get_shocks()`` draws ``PermShk`` and ``TranShk`` for the
    population, rescales each so that the cross-sectional mean equals the
    theoretical value (1.0 for mean-one distributions). This makes the
    aggregate effect of shocks exact in every period, regardless of
    population size.

    For agents in different groups (e.g., Markov states with different
    shock distributions), normalization is applied per group.

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

    def _normalize_shock_means(self):
        """Rescale PermShk and TranShk so cross-sectional means are exact.

        For each group of agents sharing the same shock distribution
        (Markov state if present, otherwise the whole population), divides
        each shock by the group's empirical mean so the new mean is
        exactly 1.0. Skips groups with fewer than 2 agents or an
        empirical mean too close to zero.
        """
        if "Mrkv" in getattr(self, "shocks", {}):
            groups = self.shocks["Mrkv"].astype(int)
            unique_groups = np.unique(groups)
        else:
            groups = np.zeros(self.AgentCount, dtype=int)
            unique_groups = [0]

        for shock_name in ["PermShk", "TranShk"]:
            if shock_name not in self.shocks:
                continue
            shock_arr = self.shocks[shock_name]

            for g in unique_groups:
                mask = groups == g
                if mask.sum() < 2:
                    continue
                empirical_mean = np.mean(shock_arr[mask])
                if abs(empirical_mean) < 1e-16:
                    continue
                shock_arr[mask] *= 1.0 / empirical_mean


class PermanentIncomeNormalizationMixin:
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
        """True when the period-0 growth factor is state-dependent."""
        pgf = self.PermGroFac[0]
        return isinstance(pgf, (list, tuple, np.ndarray)) and np.size(pgf) > 1

    def _permshk_log_moments(self):
        """(E[log psi], Var[log psi]) of the period-0 permanent shock.

        Computed directly from the discrete approximation's atoms, which
        is exact for the distribution actually simulated. For
        state-dependent shock distributions (a list per Markov state),
        moments are weighted by the stationary distribution of
        ``MrkvArray[0]``.
        """
        dstn = self.PermShkDstn[0]
        if isinstance(dstn, (list, tuple)):
            weights = self._stationary_weights(len(dstn))
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

    def _stationary_weights(self, n_states):
        """Stationary weights over Markov states, or uniform if absent."""
        mrkv = getattr(self, "MrkvArray", None)
        if mrkv is not None:
            transition = mrkv[0] if isinstance(mrkv, (list, tuple)) else mrkv
            transition = np.asarray(transition)
            if transition.shape == (n_states, n_states):
                return _stationary_distribution(transition)
        warnings.warn(
            "PermanentIncomeNormalizationMixin: state-dependent growth "
            "without a matching MrkvArray; using uniform state weights "
            "for the analytical pLvl drift.",
            stacklevel=2,
        )
        return np.full(n_states, 1.0 / n_states)

    def _effective_log_PermGroFac(self):
        """Per-period drift of ``E[log pLvl]`` from growth factors.

        Scalar growth: ``log(PermGroFac[0])``. Vector (Markov) growth:
        the stationary-distribution-weighted mean of ``log(G_s)`` — the
        exact drift of the population mean of ``log pLvl`` for agents
        moving on the chain at stationarity. Override for model-specific
        targets (e.g. level-targeting instead of log-mean-targeting).
        """
        pgf = np.asarray(self.PermGroFac[0], dtype=float).flatten()
        if pgf.size == 1:
            return float(np.log(pgf[0]))
        weights = self._stationary_weights(pgf.size)
        return float(np.dot(weights, np.log(pgf)))

    def _analytical_log_pLvl_moments(self, age_k):
        """Analytical ``(mu, sigma)`` of ``log(pLvl)`` at age ``age_k``.

        ``mu_k = pLogInitMean + k * g_log + max(k - 1, 0) * E[log psi]``
        ``sigma_k = sqrt(pLogInitStd**2 + max(k - 1, 0) * Var[log psi])``

        where ``g_log`` is :meth:`_effective_log_PermGroFac` and the
        shock moments come from the simulated discrete distribution
        (:meth:`_permshk_log_moments`). The first growth step after birth
        is deterministic (newborns receive ``PermShk = 1``), hence the
        ``max(k - 1, 0)`` effective shock periods. Under vector growth
        ``sigma_k`` omits the Markov-path variance contribution — which
        is exactly why the ``"auto"`` moments mode is mean-only there.

        Parameters
        ----------
        age_k : int
            Periods since birth for the cohort.

        Returns
        -------
        mu_k : float
        sigma_k : float
        """
        e_log_psi, var_log_psi = self._permshk_log_moments()
        g_log = self._effective_log_PermGroFac()
        eff_periods = max(int(age_k) - 1, 0)
        p_init_mean = getattr(self, "pLogInitMean", getattr(self, "pLvlInitMean", 0.0))
        p_init_std = getattr(self, "pLogInitStd", getattr(self, "pLvlInitStd", 0.0))
        mu_k = p_init_mean + int(age_k) * g_log + eff_periods * e_log_psi
        sigma_k = np.sqrt(p_init_std**2 + eff_periods * var_log_psi)
        return float(mu_k), float(sigma_k)

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

        Applies the resolved moments mode per age cohort (skipping
        cohorts smaller than ``_pLvl_norm_min_cohort``), then rescales
        the variables in ``_pLvl_norm_adjust_vars`` by the inverse pLvl
        ratio so level quantities are preserved.
        """
        if not getattr(self, "normalize_pLvl", False):
            return

        mode = self._resolved_moments_mode()
        log_p = np.log(np.maximum(self.state_now["pLvl"], 1e-16))

        for k in np.unique(self.t_age):
            mask = self.t_age == k
            if mask.sum() < self._pLvl_norm_min_cohort:
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

        pLvl_old = self.state_now["pLvl"].copy()
        self.state_now["pLvl"][:] = np.exp(log_p)
        ratio = pLvl_old / np.maximum(self.state_now["pLvl"], 1e-16)
        for var in self._pLvl_norm_adjust_vars:
            if var in self.state_now and isinstance(self.state_now[var], np.ndarray):
                self.state_now[var] *= ratio

    # ------------------------------------------------------------------
    # Wiring (self-contained; no HARK.core changes required)
    # ------------------------------------------------------------------

    def post_state_hook(self):
        """Run the normalization; chain to any base-class hook first."""
        sup = getattr(super(), "post_state_hook", None)
        if sup is not None:
            sup()
        self.post_sim_normalize_pLvl()

    def sim_one_period(self):
        """Replicate ``AgentType.sim_one_period`` with normalization
        inserted between ``get_states()`` and ``get_controls()``, so that
        controls are computed from the normalized states.

        When composed together with ``DualMeasureMixin``, put
        ``DualMeasureMixin`` FIRST in the MRO: its own ``sim_one_period``
        calls ``post_state_hook()`` (which chains to this mixin's), and
        this replicate — which knows nothing of the Q-measure — must not
        shadow it.
        """
        if getattr(self, "dual_measure", False):
            raise RuntimeError(
                "Compose DualMeasureMixin BEFORE "
                "PermanentIncomeNormalizationMixin in the MRO; this "
                "mixin's sim_one_period does not run the Q-measure step."
            )
        if not hasattr(self, "solution"):
            raise Exception(
                "Model instance does not have a solution stored. To "
                "simulate, it is necessary to run the `solve()` method "
                "first."
            )

        self.get_mortality()

        for var in self.state_now:
            self.state_prev[var] = self.state_now[var]
            if isinstance(self.state_now[var], np.ndarray):
                self.state_now[var] = np.empty(self.AgentCount)

        if self.read_shocks:
            self.read_shocks_from_history()
        else:
            self.get_shocks()
        self.get_states()
        self.post_state_hook()
        self.get_controls()
        self.get_poststates()

        self.t_age = self.t_age + 1
        self.t_cycle = self.t_cycle + 1
        self.t_cycle[self.t_cycle == self.T_cycle] = 0

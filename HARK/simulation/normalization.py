"""Mixins for simulation variance reduction via moment normalization.

These mixins adjust cross-sectional distributions during simulation so that
empirical moments match their analytical values, eliminating sampling noise
without requiring N to be a multiple of J_min.

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

    agent = NormalizedIndShock(
        normalize_pLvl=True,
        normalize_shocks=True,
        ...
    )
"""

import numpy as np


class ShockNormalizationMixin:
    """Opt-in normalization of income shocks to pin cross-sectional means.

    After get_shocks() draws PermShk and TranShk for the population,
    rescales each so that the cross-sectional mean equals the theoretical
    value (1.0 for mean-one distributions).  This ensures that the aggregate
    effect of shocks is exact in every period, regardless of population size.

    For agents in different groups (e.g., Markov states with different shock
    distributions), normalization is applied per group.

    Attributes
    ----------
    normalize_shocks : bool
        When True, shock normalization is applied.  Default False.
    """

    normalize_shocks = False

    def get_shocks(self):
        """Draw shocks, then normalize means if enabled."""
        super().get_shocks()
        if not getattr(self, "normalize_shocks", False):
            return
        self._normalize_shock_means()

    def _normalize_shock_means(self):
        """Rescale PermShk and TranShk so cross-sectional means are exact.

        For each group of agents sharing the same shock distribution,
        divides each shock by the group's empirical mean (so the new
        mean is exactly 1.0).  Skips groups with fewer than 2 agents
        or where the empirical mean is too close to zero.
        """
        # Determine grouping: if Markov states exist, normalize per state;
        # otherwise normalize the whole population at once.
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
                n_g = mask.sum()
                if n_g < 2:
                    continue

                empirical_mean = np.mean(shock_arr[mask])
                if abs(empirical_mean) < 1e-16:
                    continue

                # Rescale so cross-sectional mean is exactly 1.0
                shock_arr[mask] *= 1.0 / empirical_mean


class PermanentIncomeNormalizationMixin:
    """Opt-in per-cohort pLvl normalization for simulation variance reduction.

    After each sim_one_period, adjusts pLvl so that within each age cohort k,
    E[log p | age=k] and Var[log p | age=k] match analytical values.  For a
    lognormal distribution, this pins all power moments E[p^k] simultaneously.

    The affine transform in log-space (shift + scale) preserves rank ordering
    of agents' permanent incomes, so wealth-income correlations are unaffected.

    Attributes
    ----------
    normalize_pLvl : bool
        When True, normalization is applied after each period.  Default False.
    """

    normalize_pLvl = False

    def _analytical_log_pLvl_moments(self, age_k):
        """Compute analytical (mu, sigma) of log(pLvl) for agents at age k.

        Default implementation for models with state-independent permanent
        shocks (e.g., IndShockConsumerType):

            mu_k = pLogInitMean + k * log(PermGroFac) - eff_periods * sigma_psi^2 / 2
            sigma_k = sqrt(pLogInitStd^2 + eff_periods * sigma_psi^2)

        where eff_periods = max(k - 1, 0) because the first growth step after
        birth is deterministic (newborns get PermShk=1).

        Subclasses with state-dependent growth (e.g., Markov models with
        different PermGroFac per state) should override this method.

        Parameters
        ----------
        age_k : int
            Age of the cohort (periods since birth).

        Returns
        -------
        mu_k : float
            Analytical mean of log(pLvl) for this cohort.
        sigma_k : float
            Analytical std dev of log(pLvl) for this cohort.
        """
        # Permanent shock variance from the PermShkDstn
        perm_dstn = self.PermShkDstn[0]
        log_perm = np.log(perm_dstn.atoms.flatten())
        sigma_psi_sq = (
            np.dot(perm_dstn.pmv, log_perm**2) - np.dot(perm_dstn.pmv, log_perm) ** 2
        )

        log_G = np.log(self.PermGroFac[0])

        # First growth step after birth is deterministic (newborns get PermShk=1)
        eff_periods = max(age_k - 1, 0)

        mu_k = self.pLogInitMean + age_k * log_G - eff_periods * sigma_psi_sq / 2
        sigma_k = np.sqrt(self.pLogInitStd**2 + eff_periods * sigma_psi_sq)

        return mu_k, sigma_k

    def post_sim_normalize_pLvl(self):
        """Normalize pLvl to match analytical per-cohort log-moments.

        Adjusts log(pLvl) within each age cohort so that the cross-sectional
        mean and std match the analytical values.  Skips cohorts with fewer
        than 5 agents.
        """
        if not getattr(self, "normalize_pLvl", False):
            return

        log_p = np.log(np.maximum(self.state_now["pLvl"], 1e-16))
        unique_ages = np.unique(self.t_age)

        for k in unique_ages:
            mask = self.t_age == k
            n_k = mask.sum()
            if n_k < 5:
                continue

            mu_k, sigma_k = self._analytical_log_pLvl_moments(k)
            mu_hat = np.mean(log_p[mask])
            sigma_hat = np.std(log_p[mask])

            if sigma_hat > 1e-10:
                log_p[mask] = mu_k + (sigma_k / sigma_hat) * (log_p[mask] - mu_hat)
            else:
                log_p[mask] = mu_k

        # Adjust normalized variables to keep level values unchanged.
        # pLvl is about to change; any var defined as varLvl / pLvl must
        # scale by the inverse ratio so that varLvl = varNrm * pLvl is preserved.
        pLvl_old = self.state_now["pLvl"].copy()
        self.state_now["pLvl"][:] = np.exp(log_p)
        ratio = pLvl_old / np.maximum(self.state_now["pLvl"], 1e-16)
        for var in ("mNrm", "bNrm"):
            if var in self.state_now and isinstance(self.state_now[var], np.ndarray):
                self.state_now[var] *= ratio

    def post_state_hook(self):
        """Normalize pLvl before controls are computed.

        Uses the ``post_state_hook`` introduced in ``AgentType.sim_one_period``
        so that ``cLvl = cNrm * pLvl`` and ``cLvl_splurge`` see the normalized
        ``pLvl``, not just the next period's dynamics.
        """
        super().post_state_hook()
        self.post_sim_normalize_pLvl()

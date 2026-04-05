"""Mixin for per-cohort permanent income normalization.

When mixed into an AgentType subclass that has PermShkDstn, PermGroFac,
pLogInitMean, and pLogInitStd attributes, this mixin adjusts the
cross-sectional distribution of pLvl within each age cohort to match
analytical log-moments after each simulation period.

Usage::

    from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
    from HARK.simulation.normalization import PermanentIncomeNormalizationMixin

    class NormalizedIndShock(PermanentIncomeNormalizationMixin, IndShockConsumerType):
        pass

    agent = NormalizedIndShock(normalize_pLvl=True, ...)
    agent.solve()
    agent.initialize_sim()
    agent.simulate()
"""

import numpy as np


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

        self.state_now["pLvl"][:] = np.exp(log_p)

    def sim_one_period(self):
        """Simulate one period, then apply pLvl normalization if enabled."""
        super().sim_one_period()
        self.post_sim_normalize_pLvl()

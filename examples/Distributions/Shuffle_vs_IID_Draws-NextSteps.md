# Next steps: Markov shuffle + pLvl normalization

**Date:** April 5, 2026  
**From:** HAFiscal validation AI  
**Status:** Income shuffle is working end-to-end. Two items remain.

---

## Current state

- **Income shuffle:** DONE. `income_shuffle=True` on both `IndShockConsumerType` and `MarkovConsumerType`. Committed as `dbb569db` on `harmenberg-dual-measure`. HAFiscal's `pyproject.toml` now points to this branch and tests pass.

- **Markov transition shuffle:** Not yet implemented.

- **pLvl normalization:** Not yet implemented.

---

## Item 1: Markov transition shuffle

### What to implement

Add `shuffle` parameter to `MarkovProcess.draw(state, shuffle=False)` in `HARK/distributions/base.py`.

The current code (`_rng.choice` per agent) should become the `shuffle=False` path. The `shuffle=True` path:

1. For each source state $j$, count $N_j$ agents currently in state $j$
2. Compute target counts: use the same floor-plus-leftover-slots algorithm that `DiscreteDistribution.draw(shuffle=True)` uses internally — don't reinvent the rounding logic
3. Randomly permute which agents get assigned to which target state
4. Fall back to iid when `N_j * min(transition_matrix[j, :]) < 1` (too few agents for meaningful deterministic counts)

### Where to wire it

In `MarkovConsumerType` (file `HARK/ConsumptionSaving/ConsMarkovModel.py`), add a class attribute:

```python
# In simulation defaults or class body:
markov_shuffle = False
```

In `get_markov_states()` (line ~954), pass through:

```python
markov_process = MarkovProcess(
    self.MrkvArray[t], seed=self.RNG.integers(0, 2**31 - 1)
)
MrkvNow[right_age] = markov_process.draw(
    MrkvPrev[right_age], shuffle=self.markov_shuffle
)
```

### Dual-measure interaction

Markov transitions are SHARED between P and Q tracks (both see the same employment state). So the shuffle applies to the shared transitions — no special Q handling needed.

### Test

With 10,000 agents, `markov_shuffle=True`, and a 2-state Markov with P = [[0.95, 0.05], [0.5, 0.5]]:
- After one step from 9500 employed + 500 unemployed: exactly 475 lose jobs, exactly 250 get jobs (up to ±1 from rounding)
- Verify state counts match `round(N_j * P[j, j'])` every period for 100 periods

---

## Item 2: pLvl normalization mixin

### What to implement

A mixin class that, after each `sim_one_period`, adjusts the cross-sectional pLvl distribution within each age cohort to match analytical moments.

```python
class PermanentIncomeNormalizationMixin:
    normalize_pLvl = False
    
    def _analytical_log_pLvl_moments(self, age_k):
        """
        Returns (mu_k, sigma_k) for cohort at age k.
        
        Default implementation (suitable for IndShock and Markov with
        state-independent permanent shocks):
        
            mu_k = pLogInitMean + (k+1) * log(g_eff)
                   - eff_shock_periods * sigma_psi_sq / 2
            sigma_k = sqrt(pLogInitStd^2 + eff_shock_periods * sigma_psi_sq)
        
        where:
            g_eff = (1-u)*G_employed + u*G_unemployed
            eff_shock_periods = (1-u) * max(k-1, 0)
                (BUG-003: first growth step is deterministic)
                (BUG-019: unemployed periods have no permanent shock)
            sigma_psi_sq = Var[log(PermShk)] from employed IncShkDstn
        
        Downstream models (e.g., HAFiscal's AggFiscalType) can override
        this to use their own SST helpers.
        """
        raise NotImplementedError("Subclass must implement or use default")
    
    def post_sim_normalize_pLvl(self):
        if not self.normalize_pLvl:
            return
        log_p = np.log(np.maximum(self.state_now['pLvl'], 1e-16))
        for k in range(self.T_age):
            mask = (self.t_age == k)
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
        self.state_now['pLvl'][:] = np.exp(log_p)
```

### Clarifications from the earlier exchange

**The affine transform preserves rank ordering.** `mu_k + (sigma_k / sigma_hat) * (log_p - mu_hat)` is strictly increasing in `log_p` (it's a linear rescaling in log space). Agent i with higher pLvl always keeps higher pLvl after adjustment. All rank-based correlations (wealth-income, etc.) are exactly preserved.

**The within-cohort lognormal approximation is already deployed and validated.** `compute_pLvl_distribution` in `tm_methods.py` uses the same `(1-u)*k` formula and achieves +0.14% accuracy for E[p^{-ρ}] against MC. The normalization mixin reuses the same approximation — it's not introducing a new one.

**The formulas are not a research task — they're existing code.** HAFiscal's `income_process_sst.py` has:
- `effective_pLvl_growth(agent, u)` → the g_eff
- `effective_perm_shock_variance_periods(k, agent, u)` → the (1-u)*k
- `_get_perm_shock_var(agent)` → sigma_psi_sq from the employed IncShkDstn

The default implementation should use `PermGroFac[0]` for G_employed and compute u from the Markov ergodic (or accept it as a parameter).

### Where to call it

In `sim_one_period`, after the P-pipeline completes and before time advancement. For `DualMeasureMixin` compatibility, there should also be a `post_sim_normalize_pLvl_Q` that uses Q-measure analytical moments (higher growth: `g_Q = (1-u)*G*E_Q[psi] + u`).

### Test

Run `IndShockConsumerType` with `normalize_pLvl=True` for 200 periods. For each age cohort k with >10 agents:
- `|mean(log_p[age==k]) - mu_k_analytical| < 1e-10`
- `|std(log_p[age==k]) - sigma_k_analytical| < 1e-10`

Run with `normalize_pLvl=False` and verify these are NOT tight (they'll have O(1/sqrt(N_k)) noise).

---

## Implementation order

1. **Markov shuffle** first — it's orthogonal to everything else and high value
2. **pLvl normalization** second — needs the mixin pattern and the `_analytical_log_pLvl_moments` hook
3. Push both to `harmenberg-dual-measure`
4. Tell us — we'll reinstall, enable all three on `AggFiscalType`, and run the Gatekeeper convergence sweep

---

## How we'll validate from HAFiscal

Once both items are on `harmenberg-dual-measure`:

```python
# In HAFiscal's AggFiscalType or test setup:
agent.income_shuffle = True
agent.markov_shuffle = True
agent.normalize_pLvl = True
```

Then run Gatekeeper at N ∈ {5k, 10k, 20k, 40k} and verify:
- AggCons SD drops ~70%+ (from pLvl norm + income shuffle)
- E[u'] SD drops ~25%+ (from pLvl norm)
- Employment fraction noise drops (from Markov shuffle)
- TM-MC gaps shrink faster with N than without variance reduction

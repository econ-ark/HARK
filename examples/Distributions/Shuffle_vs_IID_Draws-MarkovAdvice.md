# Implementing variance-reduced MC for Markov models in HARK

**Date:** April 5, 2026  
**Context:** HAFiscal TM-vs-MC validation; `docs/plan-reshuffle-pr1244-revival.md`  
**Branch:** `harmenberg-dual-measure`

This document describes the specific changes needed in HARK's Markov simulation infrastructure to support three variance-reduction techniques: income shock shuffling, Markov transition shuffling, and permanent-income normalization. These are complementary layers that can be enabled independently.

---

## Current HARK architecture (what exists)

### `DiscreteDistribution.draw(N, shuffle=True)` — DONE

`HARK/distributions/discrete.py` line 294. When `shuffle=True`, outcomes are assigned to match theoretical probabilities as closely as possible (floor + leftover slots + random permutation), then shuffled. Merged via PR #1691.

**Not yet used** in any `get_shocks` method — all simulation code still calls `draw(N)` (iid) or manual CDF inversion.

### `MarkovProcess.draw(state)` — per-agent iid

`HARK/distributions/base.py` line 172. For each agent, independently draws a new state from `transition_matrix[current_state, :]` via `np.random.choice`. No shuffling option.

### `MarkovConsumerType.get_markov_states()` — calls `MarkovProcess.draw`

`HARK/ConsumptionSaving/ConsMarkovModel.py` line 954. For each `t_cycle`, creates a fresh `MarkovProcess` and calls `draw(MrkvPrev[right_age])`. Transitions are fully independent across agents.

### `MarkovConsumerType.get_shocks()` — manual CDF inversion

Line 988. For each `(t_cycle, mrkv_state)` slice of agents:
```python
base_draws = IncShkDstnNow._rng.uniform(size=N)
EventDraws = np.searchsorted(np.cumsum(IncShkDstnNow.pmv), base_draws)
PermShkNow[these] = IncShkDstnNow.atoms[0][EventDraws] * PermGroFacNow
TranShkNow[these] = IncShkDstnNow.atoms[1][EventDraws]
```

This is iid sampling via CDF inversion. The `base_draws` are stored in `_base_shock_draws` for dual-measure Q-track replay.

### `DualMeasureMixin` — Q-track via CDF reinversion

`HARK/dual_measure.py`. The Q-track uses the same `base_draws` (uniform) but inverts through the Q-reweighted CDF (`IncShkDstn_Q`). This ensures P and Q paths share the same "luck ordering" while drawing from different distributions.

---

## Change 1: Income shock shuffling in `get_shocks`

### Goal

Replace iid CDF inversion with `IncShkDstnNow.draw(N, shuffle=True)` per `(t_cycle, mrkv_state)` slice, controlled by an opt-in flag.

### What to change in `MarkovConsumerType.get_shocks()`

Add a parameter `income_shuffle` (default `False`) to the class. In `get_shocks`, replace:

```python
# CURRENT (iid):
base_draws = IncShkDstnNow._rng.uniform(size=N)
EventDraws = np.searchsorted(np.cumsum(IncShkDstnNow.pmv), base_draws)
PermShkNow[these] = IncShkDstnNow.atoms[0][EventDraws] * PermGroFacNow
TranShkNow[these] = IncShkDstnNow.atoms[1][EventDraws]
```

with:

```python
# NEW (shuffled when enabled):
if self.income_shuffle:
    draws = IncShkDstnNow.draw(N, shuffle=True)
    # draws is (N,) array of atom indices or (N, n_vars) array of values
    # depending on DiscreteDistribution.draw return convention
    PermShkNow[these] = draws[:, 0] * PermGroFacNow  # or draws['PermShk']
    TranShkNow[these] = draws[:, 1]                   # or draws['TranShk']
    base_draws_dict[(t, j)] = None  # no base_draws to store
else:
    # existing iid path (unchanged)
    base_draws = IncShkDstnNow._rng.uniform(size=N)
    ...
```

### Interaction with `DualMeasureMixin`

The dual-measure Q-track replays the P-track's `base_draws` through the Q CDF. With shuffling, there are no `base_draws` (the shuffled assignment IS the draw). Two options:

**Option A (recommended):** When `income_shuffle=True`, the Q-track also uses `IncShkDstn_Q.draw(N, shuffle=True)` independently. The P and Q tracks get different shock realizations but both have exact marginal frequencies. Since P-Q comparison for $p$-linear aggregates holds period-by-period regardless of individual shock pairing, this is fine for Class A. For per-agent P-Q pairing (used in some diagnostic comparisons), this option breaks the pairing.

**Option B:** Share the shuffle permutation. `draw(N, shuffle=True)` internally computes an assignment (atom indices), then permutes. If we expose the pre-permutation assignment, the Q-track could use the same permutation on its own frequency-matched assignment. This preserves per-agent P-Q pairing but requires exposing internals of `draw`.

**Recommendation:** Start with Option A. Per-agent P-Q pairing is not needed for the paper's results.

### Also apply to `IndShockConsumerType.get_shocks()`

Same pattern but simpler (no Markov state slicing — all agents draw from the same `IncShkDstn[t]`).

---

## Change 2: Markov transition shuffling in `get_markov_states`

### Goal

For each source state $j$ with $N_j$ agents, deterministically compute the target counts $N_{j \to j'} = \text{round}(N_j \cdot P(j \to j'))$ and randomly select WHICH agents make each transition. Falls back to iid draws when $N_j$ is too small (say $< 20$).

### What to change

**`MarkovProcess` (or a new subclass/option):**

Add a `shuffle` parameter to `MarkovProcess.draw`:

```python
def draw(self, state, shuffle=False):
    if not shuffle:
        return self._draw_iid(state)  # existing behavior
    return self._draw_shuffled(state)

def _draw_shuffled(self, state):
    """Deterministic state counts, random agent assignment."""
    new_state = np.empty_like(state)
    J = self.transition_matrix.shape[1]
    for j in range(J):
        agents_in_j = np.where(state == j)[0]
        N_j = len(agents_in_j)
        if N_j < 20:
            # Fall back to iid for small populations
            for idx in agents_in_j:
                new_state[idx] = self._rng.choice(J, p=self.transition_matrix[j])
            continue
        # Deterministic target counts
        probs = self.transition_matrix[j]
        targets = np.round(N_j * probs).astype(int)
        # Adjust for rounding (ensure sum = N_j)
        diff = N_j - targets.sum()
        if diff != 0:
            # Add/subtract from the state with largest fractional part
            fracs = N_j * probs - np.floor(N_j * probs)
            adjust_idx = np.argsort(fracs)
            for d in range(abs(diff)):
                targets[adjust_idx[-(d+1) if diff > 0 else d]] += np.sign(diff)
        # Randomly assign agents to target states
        perm = self._rng.permutation(agents_in_j)
        offset = 0
        for jp in range(J):
            new_state[perm[offset:offset+targets[jp]]] = jp
            offset += targets[jp]
    return new_state
```

**`MarkovConsumerType.get_markov_states()`:**

Add a `markov_shuffle` parameter (default `False`). When True, pass `shuffle=True` to `MarkovProcess.draw`:

```python
markov_process = MarkovProcess(self.MrkvArray[t], seed=...)
MrkvNow[right_age] = markov_process.draw(
    MrkvPrev[right_age], shuffle=self.markov_shuffle
)
```

### Interaction with `DualMeasureMixin`

The P and Q tracks must share the same Markov transitions (both see the same employment state). Markov shuffling applies to the SHARED transitions, so both tracks automatically get the same deterministic state counts.

---

## Change 3: Permanent-income normalization

### Goal

Each period, after `sim_one_period` updates pLvl, adjust the cross-sectional distribution of pLvl within each age cohort to match the analytical per-cohort lognormal moments.

### What to add

A new mixin `PermanentIncomeNormalizationMixin` (or integrate into `AgentType`):

```python
class PermanentIncomeNormalizationMixin:
    """
    Opt-in per-cohort pLvl normalization.
    
    After each sim_one_period, adjusts pLvl so that within each age
    cohort k, E[log p | age=k] and Var[log p | age=k] match the
    analytical values. For a lognormal distribution, this pins all
    power moments E[p^k] simultaneously.
    """
    normalize_pLvl = False
    
    def _analytical_log_pLvl_moments(self, age_k):
        """
        Returns (mu_k, sigma_k) for the analytical distribution of
        log(pLvl) at age k.
        
        mu_k = pLogInitMean + (k+1) * log(g_eff)
               - effective_shock_periods * sigma_psi_sq / 2
        sigma_k = sqrt(pLogInitStd^2 + effective_shock_periods * sigma_psi_sq)
        
        where g_eff = (1-u)*G_emp + u*G_unemp and
        effective_shock_periods = (1-u) * max(k-1, 0) (BUG-003: first
        growth step is deterministic; BUG-019: unemployed periods have
        no permanent shock).
        """
        # Implementation uses the same formulas as
        # tm_methods.compute_pLvl_distribution
        ...
    
    def post_sim_normalize_pLvl(self):
        """Called after sim_one_period to normalize pLvl."""
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

### Where to call it

In `sim_one_period`, after the P-pipeline (get_mortality → get_shocks → get_states → get_controls → get_poststates) and before time advancement:

```python
def sim_one_period(self):
    # ... existing P-pipeline ...
    self.post_sim_normalize_pLvl()  # NEW
    # ... Q-pipeline (if dual measure) ...
    # ... time advancement ...
```

For the dual-measure Q-track: a separate `post_sim_normalize_pLvl_Q()` using Q-measure analytical moments (higher growth rate: $g_Q = (1-u) G \mathbb{E}_Q[\psi] + u$ where $\mathbb{E}_Q[\psi] = \mathbb{E}_P[\psi^2]$).

### Why per-cohort is exact

Within age cohort $k$, pLvl is the product of $k$ iid lognormal shocks times a lognormal initial draw. This IS a single lognormal. For a lognormal, pinning $\mathbb{E}[\log p]$ and $\text{Var}[\log p]$ pins ALL power moments $\mathbb{E}[p^k] = \exp(k\mu + k^2\sigma^2/2)$. So two parameters per cohort → all moments exact.

The cross-sectional distribution is a mixture of lognormals over cohorts. The per-cohort normalization makes each component exact, which makes the mixture exact.

### Tested variance reduction (HAFiscal, N=20k, ρ=2)

| Aggregate | Raw MC SD | Normalized SD | Reduction |
|-----------|-----------|--------------|:---------:|
| AggCons ($p$-linear, $k=1$) | 2013 | 519 | **74%** |
| $\mathbb{E}[u']$ ($p$-nonlinear, $k=-2$) | 1.41e-4 | 0.996e-4 | **29%** |
| $\mathbb{E}[p^{-\rho}]$ | 1.06e-4 | 0.52e-4 | **51%** |

---

## Interaction matrix: which techniques work together

| Technique | Affects | Works with dual measure? | Works with shuffled income? | Works with Markov shuffle? |
|-----------|---------|:------------------------:|:---------------------------:|:--------------------------:|
| **Income shuffle** | Per-period $(\psi, \theta)$ | Option A: independent Q draws. Option B: shared permutation. | — | Yes (orthogonal) |
| **Markov shuffle** | Employment state counts | Yes (shared transitions) | Yes (orthogonal) | — |
| **pLvl normalization** | Cross-sectional pLvl distribution | Yes (separate P and Q moments) | Yes (orthogonal) | Yes (orthogonal) |

All three are orthogonal — they pin different aspects of the simulation and can be enabled in any combination. The "fully reduced" MC has all three active simultaneously.

---

## Parameters to add to `MarkovConsumerType`

| Parameter | Type | Default | Effect |
|-----------|------|---------|--------|
| `income_shuffle` | bool | `False` | Use `draw(N, shuffle=True)` for income shocks |
| `markov_shuffle` | bool | `False` | Deterministic Markov state counts |
| `normalize_pLvl` | bool | `False` | Per-cohort pLvl log-moment pinning |

All default to `False` for backward compatibility. HAFiscal (or any downstream model) enables them as needed.

---

## Testing strategy

1. **Unit test (income shuffle):** For a `MarkovConsumerType` with known `IncShkDstn`, verify that with `income_shuffle=True`, the cross-sectional shock frequencies per `(t_cycle, Mrkv)` slice exactly match `pmv` (up to integer rounding).

2. **Unit test (Markov shuffle):** For a known `MrkvArray` with 1000 agents, verify that state counts after transition match `round(N_j * P(j→j'))` exactly when `markov_shuffle=True`.

3. **Unit test (pLvl normalization):** For a `MarkovConsumerType` run for 100 periods with `normalize_pLvl=True`, verify that `E[log(pLvl) | age=k]` and `Var[log(pLvl) | age=k]` match analytical values within machine precision for each age cohort.

4. **Integration test:** Run the same economy with all three off (baseline) vs all three on. Verify:
   - AggCons time-series SD decreases substantially (~70%+)
   - $\mathbb{E}[u']$ time-series SD does not increase (and ideally decreases)
   - Long-run means are statistically identical (the techniques reduce variance, not bias)

5. **Convergence test:** Run at N ∈ {5k, 10k, 20k, 40k}. With all three active, the gap between MC and TM should shrink faster with N than without — confirming the variance reduction accelerates convergence.

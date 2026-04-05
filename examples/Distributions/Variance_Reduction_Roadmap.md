# From moment-pinning to the covariance kernel: a roadmap for variance reduction in HARK

**Date:** April 5, 2026  
**Context:** Insights from HAFiscal TM-vs-MC validation work  
**Target audience:** An AI (or developer) working on HARK's simulation infrastructure to implement these improvements

---

## The big idea

HARK's Monte Carlo simulations carry unnecessary sampling noise in their aggregates because the simulator treats the permanent-income distribution as fully random, even though much of that distribution is **analytically known**. A graduated series of improvements — from simple moment-pinning through income shuffling to covariance kernels — can eliminate most of this noise while preserving the per-agent detail that MC provides and TM cannot.

The progression forms a spectrum:

| Technique | What's analytically pinned | Noise eliminated | Complexity |
|-----------|--------------------------|-----------------|-----------|
| Raw MC | Nothing | — | Baseline |
| **pLvl log-moment normalization** | $\mathbb{E}[\log p]$ and $\text{Var}[\log p]$ per age cohort | All $\mathbb{E}[p^k]$ drift | Trivial (2 lines/period) |
| **Income shock shuffling** (`shuffle=True`) | Per-period $(\psi, \theta)$ frequencies | Shock distribution ≠ theoretical | PR #1691 done; PR #1244 revival |
| **Markov transition shuffling** | Employment state counts | Employment fraction noise | New (PR 3 in reshuffle plan) |
| **Harmenberg neutral measure** (MC-Q) | $p$-weighted distribution $\pi_Q(m,j)$ | All noise for $p$-linear aggregates | Implemented (`DualMeasureMixin`) |
| **Transition matrix (TM)** | Entire $\pi(m,j)$ distribution | All sampling noise | Implemented (`tm_methods.py`) |
| **TM + covariance kernel** | $\pi(m,j)$ + within-period shock integration | Everything except $p$-$a$ correlation | Implemented (`compute_kernels`) |

Each layer eliminates a different noise source. They can be combined independently. The first three are MC-side improvements (make MC smoother); the last two are TM-side (make TM handle nonlinear functionals). Together they make MC and TM converge faster to the same answer.

---

## 1. pLvl log-moment normalization

### The problem

In a buffer-stock model with permanent income shocks, each agent's $p_{i,t+1} = p_{i,t} \cdot G \cdot \psi_{i,t+1}$. With finite $N$ agents, the cross-sectional mean $\bar{p}_t = (1/N)\sum p_{i,t}$ drifts randomly — even though the analytical $\mathbb{E}[p]$ is known exactly from the age-cohort lognormal mixture. This drift adds O(1/√N) noise to every aggregate.

### The fix

Each period, after computing pLvl for all agents, apply a **location-scale adjustment in log space** within each age cohort $k$:

$$\log p_i^{\text{adj}} = \mu_k + \frac{\sigma_k}{\hat\sigma_k}(\log p_i - \hat\mu_k)$$

where $\mu_k$ and $\sigma_k$ are the **analytical** mean and standard deviation of $\log p$ for cohort $k$ (known in closed form from the model parameters), and $\hat\mu_k$, $\hat\sigma_k$ are the **sample** statistics for the agents currently at age $k$.

### Why per-cohort matters

The cross-sectional pLvl distribution is a **mixture of lognormals** over age cohorts. For a single lognormal, pinning $\mathbb{E}[\log p]$ and $\text{Var}[\log p]$ pins **all** power moments $\mathbb{E}[p^k] = \exp(k\mu + k^2\sigma^2/2)$. Within each cohort, the distribution IS a single lognormal (product of $k$ iid lognormal shocks), so pinning two log-moments per cohort pins all power moments exactly.

The variance of $\log p$ for cohort $k$ is:

$$\sigma^2_k = \sigma^2_{\text{init}} + k_{\text{eff}} \cdot \sigma^2_\psi$$

where $k_{\text{eff}} = (1-u) \cdot k$ accounts for unemployed periods receiving no permanent shock (`perm_shocks_during_unemployment=False`). This formula is already implemented in HARK/HAFiscal as `effective_perm_shock_variance_periods` in `income_process_sst.py`.

### What it buys

Tested on HAFiscal with N=20,000, T=300, ρ=2:

| Statistic | Raw MC | Pin mean only | **Pin log-moments** |
|-----------|--------|:------------:|:-------------------:|
| AggCons per-period SD ($p$-linear) | 2013 | 496 (-75%) | **519 (-74%)** |
| $\mathbb{E}[p^{-\rho}]$ per-period SD | 1.06e-4 | 1.52e-4 (+43%) | **5.21e-5 (-51%)** |
| $\mathbb{E}[u'(c)]$ per-period SD | 1.41e-4 | 1.91e-4 (+35%) | **9.96e-5 (-29%)** |

**Key finding:** pinning only the mean ($\mathbb{E}[p]$) helps $p$-linear aggregates but **hurts** $p$-nonlinear ones. Pinning both log-moments helps **everything** — because controlling $\text{Var}[\log p]$ controls $\mathbb{E}[p^k]$ for all $k$ simultaneously.

### Implementation in HARK

This should be an opt-in feature on `AgentType` or a mixin:

```python
class PermanentIncomeNormalizationMixin:
    normalize_pLvl = False  # default off for backward compatibility
    
    def post_transition_normalize_pLvl(self):
        """After sim_one_period, adjust pLvl to match analytical per-cohort distribution."""
        if not self.normalize_pLvl:
            return
        log_p = np.log(np.maximum(self.state_now['pLvl'], 1e-16))
        for k in range(self.T_age):
            mask = (self.t_age == k)
            if mask.sum() < 5:  # too few agents for meaningful statistics
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

The `_analytical_log_pLvl_moments(k)` method returns $(\mu_k, \sigma_k)$ from the closed-form formula.

---

## 2. Income shock shuffling (`draw(N, shuffle=True)`)

### The problem

Each period, agents draw $(\psi, \theta)$ from `IncShkDstn`. With iid draws, the empirical frequency of each shock atom fluctuates around the theoretical probability. For example, if a particular $(\psi_3, \theta_7)$ pair has probability 2%, in a population of 1000 agents you'd expect 20 agents to get that pair, but the actual count follows Binomial(1000, 0.02) with SD ≈ 4.4 — a 22% relative fluctuation.

### The fix

`DiscreteDistribution.draw(N, shuffle=True)` (PR #1691, already merged on `main`) assigns outcomes so that the empirical frequencies **exactly match** the theoretical probabilities (up to integer rounding), then randomly permutes the assignment across agents. This eliminates the per-period shock-frequency noise entirely.

### Current status in HARK

- `draw(..., shuffle=True)` is implemented on `DiscreteDistribution` (PR #1691, merged).
- **Not yet wired** into `IndShockConsumerType.get_shocks` or `MarkovConsumerType.get_shocks` — the simulation still uses `draw(N)` (iid). PR #1244 revival plan (see `docs/plan-reshuffle-pr1244-revival.md`) outlines how to wire it.
- For `MarkovConsumerType`: each `(t_cycle, Mrkv)` slice draws shocks for a subset of agents; `shuffle=True` should be used per slice.

### Interaction with pLvl normalization

These are complementary, not competing:
- Shuffling pins the **per-period shock frequencies** (the input to the transition)
- pLvl normalization pins the **accumulated level distribution** (the output after many transitions)

With both active, the MC has exact shock inputs AND exact level outputs. The only remaining noise is in the **normalized wealth** $m_i$ cross-section.

---

## 3. Markov transition shuffling

### The problem

Employment transitions (employed ↔ unemployed) are drawn independently per agent: each agent flips a biased coin. With 1000 unemployed agents and P(stay unemployed) = 0.5, the actual count staying unemployed is Binomial(1000, 0.5), SD ≈ 16. This adds noise to the employment fraction.

### The fix

Deterministic transitions for large groups, random selection of WHICH agents transition:

```python
N_j = count of agents in state j
for each target state j':
    target_count = round(N_j * P(j → j'))
    randomly select target_count agents from state j to move to j'
```

This preserves individual agent identity (needed for per-agent tracking) while making aggregate state counts deterministic. Falls back to random draws when $N_j$ is too small for rounding to work (say $N_j < 20$).

### Current status in HARK

- Not implemented. PR 3 in the reshuffle revival plan (`docs/plan-reshuffle-pr1244-revival.md`) addresses this.
- Design constraint for Harmenberg dual simulation: the permutation must be **shared** across P and Q tracks so physical shocks stay aligned.
- The `AggIndMrkvConsumerType.get_micro_markov_states` method is the target for this change.

---

## 4. The method-parity framework

### Four classes of paper outputs

Every aggregate statistic in a heterogeneous-agent model falls into one of four classes, depending on how it relates to permanent income $p$:

| Class | Form | Which methods agree (asymptotically) |
|-------|------|-------------------------------------|
| **A** ($p$-linear) | $\mathbb{E}[p \cdot f(m,j)]$ | MC-P = MC-Q = TM-P = TM-Q |
| **B** ($p$-nonlinear) | $\mathbb{E}[p^k \cdot f(m,j,\theta)]$, $k \neq 1$ | MC-P = TM-P+kernel; MC-Q = TM-Q+kernel (P ≠ Q) |
| **C** (distributional) | Lorenz, Gini, quantiles of $p \cdot a$ | MC-P only |
| **D** (non-separable) | Policy depends on $p$ (e.g., check phase-out) | MC; TM with $p$-buckets |

**Critical rule:** P and Q measures give **different answers** for Class B. Never compare MC-Q against MC-P for welfare or marginal utility. The Harmenberg neutral measure ($Q$) is designed for Class A only.

### Why this matters for HARK

Any validation or testing infrastructure should be organized by class. Comparing MC-Q to MC-P on a Class B object looks like a 20% "error" but is actually expected behavior. HARK's testing framework should know the class of each comparison and apply appropriate tolerances.

---

## 5. The covariance kernel (for TM evaluation of Class B)

### The problem

The TM tracks $\pi(m, j)$ but not $p$. For Class A ($p$-linear), the Harmenberg factorization gives exact results: $\mathbb{E}[p \cdot f(m)] = \mathbb{E}[p] \cdot \mathbb{E}_Q[f(m)]$. For Class B ($p$-nonlinear, e.g., welfare $u(c) = (pX)^{1-\rho}/(1-\rho)$), the TM must approximate the joint.

The naive product-measure approximation ($\mathbb{E}[p^k] \times \mathbb{E}[X^{-\rho}]$) has ~5% error from two sources:
1. **$\theta$-$m$ coupling** (~4-6%): the transitory shock $\theta$ that enters the HAFiscal splurge $S\theta$ is the same $\theta$ embedded in $m = b + \theta$. The TM plugs in $\mathbb{E}[\theta]$, missing both Jensen's inequality and the conditional structure.
2. **$p$-$m$ covariance** (~0.4%): the permanent shock $\psi$ enters both $p$ and $m$ with opposite signs (the "$\psi$-channel" from BST `ApndxBalancedGrowthcNrmAndCov`).

### The kernel

At each TM grid point with savings $a = m - c(m,j)$, integrate **forward** through the budget constraint, keeping $\theta$ explicit:

$$\kappa(a, j) = \sum_{j'} P(j \to j') \sum_{\psi,\theta} \Pr(\psi,\theta|j') \; (G_{j'}\psi)^k \; f\!\left(\frac{Ra}{G_{j'}\psi}+\theta,\; j',\; \theta\right)$$

Then: $\mathbb{E}[p^k \cdot f] \approx \mathbb{E}[p^k] \cdot \sum_{m,j} \pi(m,j) \; \kappa(m - c(m,j), j)$

This resolves the $\theta$-$m$ coupling (because $\theta$ appears explicitly in both $m' = Ra/(G\psi) + \theta$ and $f$) and the $\psi$-channel (through the $(G\psi)^k$ factor). Result: ~0.15-0.5% error vs MC.

### The kernel generalizes to any $k$

The SAME kernel structure works for:
- $k = -\rho$: marginal utility $u'(c) = (pX)^{-\rho}$
- $k = 1-\rho$: CRRA welfare $u(c) = (pX)^{1-\rho}/(1-\rho)$
- $k = 1$ (log utility): additive factorization $\log(pX) = \log p + \log X$ (simplest case)

Multiple kernels can share a single loop — just accumulate different $(G\psi)^k X^k$ values at each shock atom.

### Measure consistency (BUG-020)

The kernel must use **measure-consistent** inputs:
- For TM-P: P-measure shocks and P-measure $\mathbb{E}_P[p^k]$
- For TM-Q: Q-reweighted shocks and Q-measure $\mathbb{E}_Q[p_Q^k]$

Mixing P shocks with a Q ergodic (or vice versa) produces a hybrid that matches neither MC-P nor MC-Q. This was BUG-020 in HAFiscal, fixed by adding `IncShkDstn_override` and `E_pk_override` parameters to `compute_kernels`.

### Current status in HARK

- `compute_kernels` is implemented in HAFiscal's `tm_methods.py`. It is **not** part of core HARK yet.
- The kernel is specific to models with the splurge structure ($X = (1-S)c(m) + S\theta$). For standard HARK models without a splurge ($S = 0$), the $\theta$-$m$ coupling vanishes and the kernel simplifies to direct evaluation of $c(m)^k$ at each grid point — no shock integration needed.
- For HARK models where TranShk enters consumption directly (not just through $m$), the full kernel would be needed.

---

## 6. MC-Q pLvl initialization

### The problem

When running Harmenberg dual-measure MC (`DualAggFiscalType`), the Q-path pLvl evolves under Q-reweighted permanent shocks: $p_Q$ grows faster than $p_P$ because Q upweights large $\psi$ realizations. If the Q-path starts with the P-measure pLvl distribution, it takes hundreds of periods for the Q cross-section to reach its own stationary distribution. During this transient, all Q-path statistics for $p$-nonlinear functionals are biased.

### The fix

Initialize `pLvl_Q` from the Q-stationary distribution using Q-measure growth and variance:

```python
g_Q = (1-u) * G * E_Q[psi] + u       # Q-measure level growth (> g_P)
sigma_psi_sq_Q = Var_Q[log psi]        # Q-measure shock variance
# Per cohort at age k:
mu_k_Q = pLogInitMean + k * log(g_Q)
sigma_k_Q = sqrt(pLogInitStd^2 + (1-u)*k * sigma_psi_sq_Q)
```

This is the same lognormal-mixture formula as for P, but with Q parameters. The Q path then starts near its stationary distribution, and the `t_start` cutoff (discarding early periods) handles the remaining transient.

### Implication for HARK

If HARK's `DualMeasureMixin` is to be used for validating $p$-nonlinear functionals, it should:
1. Compute the Q-stationary pLvl distribution (from Q-measure growth/variance)
2. Initialize `state_now_Q['pLvl']` from that distribution (not copy from P)
3. Provide a `t_start` parameter for tail-averaging of nonlinear statistics

---

## 7. How these pieces fit together for HARK

### What HARK core should provide

1. **`draw(N, shuffle=True)`** — already on `main` (PR #1691). ✓
2. **Income shuffling in `get_shocks`** — opt-in `income_shuffle=True` on `IndShockConsumerType` and `MarkovConsumerType`. Wire `draw(N, shuffle=True)` per `(t_cycle, Mrkv)` slice. (PR 2 in reshuffle plan.)
3. **Markov transition shuffling** — opt-in deterministic state-count evolution in `get_micro_markov_states` or equivalent. (PR 3.)
4. **pLvl normalization mixin** — opt-in per-cohort log-moment pinning after each `sim_one_period`. New.
5. **Q-path pLvl initialization** — in `DualMeasureMixin._initialize_sim_Q`, draw `pLvl_Q` from Q-stationary distribution. New.

### What downstream models (HAFiscal) provide

6. **Covariance kernel** — `compute_kernels` for TM evaluation of $p$-nonlinear functionals. Currently in `tm_methods.py`; could be generalized to core HARK if other models need it.
7. **Method-parity testing** — `compare_four_methods` organized by class (A/B/C/D) with within-measure comparisons for Class B.

### The "fully shuffled + normalized" MC

With items 1-5 all active, the MC would have:
- **Exact employment fractions** (Markov shuffle)
- **Exact shock frequencies** per state (income shuffle)
- **Exact pLvl moments** per age cohort (normalization)
- **Only normalized wealth $(m_i)$ retains sampling noise**

This is as close to TM as MC can get while still tracking individual agents. The remaining MC advantage: it computes Class C objects (Lorenz curves, per-agent welfare by wealth percentile) that TM cannot.

---

## 8. Key references

| Reference | What it provides |
|-----------|-----------------|
| `HAFiscal-Latest/plans/method-parity-map.md` | Which methods agree on which paper outputs |
| `HAFiscal-Latest/plans/kernel-integration-spec.md` | How to integrate kernels into the production pipeline |
| `HAFiscal-Latest/history/20260404-TM-nonlinear-functionals-summary.md` | Kernel error hierarchy and findings |
| `HAFiscal-Latest/Code/HA-Models/TM_MC_Marginal_Utility_Convergence_revised.ipynb` | Kernel derivation and numerical validation |
| `HAFiscal-Latest/Code/HA-Models/Covariance_Kernel_General_Theory.ipynb` | General theory: which functionals admit kernels |
| `HARK/docs/plan-reshuffle-pr1244-revival.md` | Three-PR strategy for income/Markov shuffling |
| Harmenberg (2021, *JEDynCon*) | Neutral measure: the theoretical foundation for MC-Q |
| Young (2010) | Histogram/TM method as alternative to MC |
| Carroll (2022, BST) `ApndxHarKmenberg` | Covariance kernel $\gamma(a)$, when joint distribution is required |

# Mapping Du's Transition_Matrix_Example to the Unified Framework

This document records how every major element of Will Du's
`examples/SequenceSpaceJacobians/Transition_Matrix_Example.ipynb` maps onto
the unified mathematical framework in `mathematical-framework-unified.ipynb`.

## 1. The Model's Perch Structure

Du uses `NewKeynesianConsumerType` — a single-stage-per-period model.  The
three perches are:

| Perch | Framework symbol | HARK variable | Description |
|-------|-----------------|---------------|-------------|
| **Arrival** | $x_a = b$ | `bNext` | Bank balance = $R \cdot a_{-1}$ |
| **Decision** | $x_v = m$ | `mNrm`, `dist_mGrid` | Market resources after income |
| **Continuation** | $x_e = a$ | `aNrm`, `aPol_Grid` | End-of-period assets |

### Transition functions

| Framework | Formula | HARK code location |
|-----------|---------|-------------------|
| $g_{av}(b, \zeta)$ | $m = b / \psi + \theta$ | `utilities.py` line 786: `mNext_ij = bNext[i] / perm_shks + tran_shks` |
| $g_{ve}(m, c)$ | $a = m - c$ | `ConsNewKeynesianModel.py` line 330: `aNext = dist_mGrid - self.solution[0].cFunc(dist_mGrid)` |
| $g_{ea+}(a)$ | $b_+ = R \cdot a$ | `ConsNewKeynesianModel.py` line 338: `bNext = self.Rfree[0] * aNext` |

### Shock configuration

Pre-decision shocks only ($\mathcal{Z}_{av}$ non-trivial, $\mathcal{Z}_{ve}$
trivial).  Shocks $\zeta_{av} = (\psi, \theta)$ are discretized to
`PermShkCount × TranShkCount = 5 × 5 = 25` points.

### Branching (mortality)

The code implements branching via `LivPrb`:

```
TranMatrix[:, i] += LivPrb * jump_to_grid(...) + (1 - LivPrb) * NewBornDist
```

With probability `LivPrb = 0.99375` the agent survives (standard transition);
with probability `1 - LivPrb` the agent dies and is replaced by a newborn
drawn from `NewBornDist`.  This is the branching stage from unified framework
Section 10.2.

## 2. Du's Notebook Cell-by-Cell Framework Mapping

### Part A: Steady-State MC vs TM (Cells 8–20)

#### MC simulation (Cell 11)

```python
example1.initialize_sim()
example1.simulate()
```

Implements the three-step MC loop (unified framework Section 6.1):
1. **$\Gamma_{av}$**: Draw $(\psi^{(i)}, \theta^{(i)}) \sim Q$, compute
   $m^{(i)} = R \cdot a_{-1}^{(i)} / \psi^{(i)} + \theta^{(i)}$
2. **$\Gamma_{ve}$**: Evaluate $c^{(i)} = c^*(m^{(i)})$, compute
   $a^{(i)} = m^{(i)} - c^{(i)}$
3. **$\Gamma_{ea+}$**: Set $b_+^{(i)} = R \cdot a^{(i)}$

Parameters: $N = 200{,}000$ agents, $T = 1{,}100$ periods (first 400 discarded
as burn-in).

Aggregates are sample means at the decision perch:
```python
Monte_Carlo_Assets = np.mean(example1.state_now["aNrm"] * example1.state_now["pLvl"])
```

#### TM construction (Cells 13–14)

**Step 1 — Grid** (`define_distribution_grid`):
- Discretizes $\mathcal{X}_v$ with `mCount = 90` points (decision-perch grid)
- Discretizes $p$ with `num_pointsP = 110` points
- Total grid: $M = 90 \times 110 = 9{,}900$ points

**Step 2 — Build $\boldsymbol{\Pi}$** (`calc_transition_matrix`):
- For each of the $M$ grid points, traces through all perch transitions:
  $g_{ve} \to g_{ea+} \to g_{av}$ (computing `bNext`, then `mNext_ij`)
- Applies `jump_to_grid_2D` (the lottery method) at each step
- Adds mortality branching
- Result: `self.tran_matrix`, a $9{,}900 \times 9{,}900$ matrix

**Step 3 — Ergodic distribution** (`calc_ergodic_dist`):
- Finds eigenvector of $\boldsymbol{\Pi}$ with eigenvalue 1
- Uses `scipy.sparse.linalg.eigs`
- Stores as `vec_erg_dstn` (vector) and `erg_dstn` (reshaped $90 \times 110$)

**Step 4 — Aggregates** (Cell 14):
```python
AggC = np.dot(gridc.flatten(), vecDstn)
AggA = np.dot(grida.flatten(), vecDstn)
```
Deterministic dot product $\tilde{h} = \mathbf{h}^\top \mathbf{p}^*$ (unified
framework Section 7.5).

#### Key plot (Cell 19)

MC aggregate assets fluctuate (sampling noise from $\Gamma_{av}$); TM is a
flat horizontal line (deterministic).  The visual gap between the MC mean and
the TM line is the discretization bias $b_M$ (unified framework Section 8.1).

#### Precision vs Accuracy (Cell 20)

Du's narrative matches the framework's bias-variance tradeoff exactly:
- MC: unbiased but noisy (variance $\sigma^2/N$)
- TM: deterministic but biased (grid discretization error)

### Part B: Harmenberg's Neutral Measure (Cells 31–42)

#### Dimension reduction (Cells 35–36)

```python
ss.neutral_measure = True
ss.mCount = 1000
ss.mMax = 3000
```

Maps to unified framework Section 8.4:
- `dist_pGrid` collapses to `[1]` (arrival state $\mathcal{X}_a$ becomes 1D)
- `gen_tran_matrix_1D` called instead of `gen_tran_matrix_2D`
- Grid is $1{,}000 \times 1 = 1{,}000$ points (vs $90 \times 110 = 9{,}900$)
- Computation: ~7s vs ~47s

#### Grid convergence (Cells 40–41)

```python
mpoints = [100, 150, 200, 500, 3000]
```

Sweeps grid resolution $M$ and plots TM aggregates converging toward the MC
mean.  This is the empirical demonstration of
$\text{Bias}_{\text{TM}}(M) \to 0$ as $M \to \infty$ (unified framework
Section 9.2).

### Part C: MIT Shock / Finite Horizon (Cells 43–60)

#### Setup (Cells 46–50)

- `FinHorizonAgent` with `T_cycle = 20`, `cycles = 1`
- Interest rate perturbation: `dx = -0.05` at period $t = 10$
- Terminal solution set to steady-state consumption function
- `solve()` produces period-dependent policies $\pi_t^*(x_v)$

Maps to unified framework Section 11 (finite horizon extension).

#### TM forward evolution (Cell 56)

```python
for i in range(20):
    dstn = np.dot(FinHorizonAgent.tran_matrix[i], dstn)
    C = np.dot(c_[i], dstn)
    A = np.dot(a_[i], dstn)
```

This is exactly $\mathbf{p}_{a,t+1} = \boldsymbol{\Pi}_t \mathbf{p}_{a,t}$
with per-period aggregates $\tilde{h}_t = \mathbf{h}_t^\top \mathbf{p}_{a,t}$.

#### Result (Cells 58–60)

MC and TM impulse response paths for aggregate consumption and assets nearly
overlay, confirming that both methods converge for transition dynamics when the
TM grid is sufficiently fine.

## 3. HARK API ↔ Framework Mapping

| HARK method | Framework operation | Section |
|-------------|-------------------|---------|
| `agent.solve()` | Solve Bellman: $\mathcal{V}(x_v) = \max_\pi [r + \beta \mathcal{E}(g_{ve})]$ | §3.4 |
| `agent.simulate()` | MC: per-agent traversal of $\Gamma_{av} \to \Gamma_{ve} \to \Gamma_{ea+}$ | §6.1 |
| `define_distribution_grid()` | Discretize $\mathcal{X}_v$ (and $\mathcal{X}_a$ for $p$) | §7.1 |
| `calc_transition_matrix()` | Build $\boldsymbol{\Pi} \approx \Gamma_{ea+} \circ \Gamma_{ve} \circ \Gamma_{av}$ | §7.2–7.3 |
| `calc_ergodic_dist()` | Find $\mathbf{p}^* = \boldsymbol{\Pi} \mathbf{p}^*$ | §7.4 |
| `np.dot(h, vecDstn)` | Aggregate $\tilde{h} = \mathbf{h}^\top \mathbf{p}$ | §7.5 |
| `neutral_measure = True` | Harmenberg: collapse $\mathcal{X}_a$ from 2D to 1D | §8.4 |
| `jump_to_grid_1D/2D` | Lottery method (mean-preserving grid projection) | §7.2 |
| `gen_tran_matrix_1D/2D` | Assemble $\boldsymbol{\Pi}$ column by column | §7.3 |

## 4. Key Source Files

| File | Contents |
|------|----------|
| `HARK/ConsumptionSaving/ConsNewKeynesianModel.py` | `NewKeynesianConsumerType` class with `define_distribution_grid`, `calc_transition_matrix`, `calc_ergodic_dist` |
| `HARK/utilities.py` lines 570–850 | `jump_to_grid_1D`, `jump_to_grid_2D`, `gen_tran_matrix_1D`, `gen_tran_matrix_2D` (numba-compiled) |
| `examples/SequenceSpaceJacobians/Transition_Matrix_Example.ipynb` | Du's demonstration notebook |

## 5. What Du Demonstrates vs What the Framework Calls For

### Demonstrated

| Framework element | Du's demonstration |
|---|---|
| MC is unbiased but noisy | Fluctuating aggregate asset time series |
| TM is deterministic but biased | Flat line offset from MC mean |
| Bias $\to 0$ as $M \to \infty$ | Grid convergence with $M \in \{100, 150, 200, 500, 3000\}$ |
| Harmenberg reduces dimension | 47s → 7s; 2D → 1D grid; improved accuracy |
| Finite horizon: $\boldsymbol{\Pi}_t$ sequence | MIT shock with 20 per-period transition matrices |
| MC and TM agree on impulse responses | Consumption and asset IRFs nearly overlay |
| Mortality as branching | `LivPrb` splits population in transition matrix |

### Not yet demonstrated (opportunities for extension)

| Framework element | Gap in Du's notebook |
|---|---|
| Explicit MSE decomposition ($\text{Bias}^2 + \text{Var}$) | Gap not computed numerically |
| MC confidence bands ($\sigma_h / \sqrt{N}$) | Sampling error not quantified |
| Perch-level narrative | Distributions not labeled by perch ($\mu_v$ vs $\mu_e$) |
| Timing breakdown by operation | Full TM time reported, not split by build vs eigensolve |
| Lottery error analysis | Conditional variance underestimation not examined |
| Multi-stage periods (cons + portfolio) | Single-stage model only |
| Alternative calibrations | One parameter set only |

## 6. Du's Parameters (for reproduction)

```python
{
    "CRRA": 2,
    "Rfree": [1.04**0.25],
    "DiscFac": 0.975,
    "LivPrb": [0.99375],
    "PermGroFac": [1.00],
    "AgentCount": 200000,
    "T_sim": 1100,
    "PermShkStd": [0.06],
    "PermShkCount": 5,
    "TranShkStd": [0.2],
    "TranShkCount": 5,
    "UnempPrb": 0.00,
    "IncUnemp": 0.0,
    "mCount": 90,
    "mFac": 3,
    "mMax": 10000,
}
```

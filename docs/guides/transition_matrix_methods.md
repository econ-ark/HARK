# Transition Matrix Methods: Converting Monte Carlo Simulations

This guide explains how to adapt HARK code that uses Monte Carlo (MC)
simulation to use transition matrix (TM) methods instead, and when each
approach is appropriate. It covers single-state models
(`IndShockConsumerType` / `NewKeynesianConsumerType`), discrete Markov
models (`MarkovConsumerType`), and general-equilibrium aggregate-shock
models (`CobbDouglasMarkovEconomy` / Krusell-Smith).

## When to Use Which Method

| Criterion | Monte Carlo | Transition Matrix |
|-----------|-------------|-------------------|
| **Bias** | 0 (unbiased) | \(O(\Delta g)\) (grid discretization) |
| **Variance** | \(O(1/N)\) (sampling noise) | 0 (deterministic) |
| **Speed** | Slow for large N | Fast (matrix ops); ~100× for KS forward propagation |
| **Memory** | \(O(N)\) per variable | \(O(M^2)\) for TM of M grid points |
| **Best for** | Individual paths, percentiles, Gini, MSM estimation | Steady-state aggregates, SSJ Jacobians, impulse responses |
| **Not suited for** | Very precise aggregate moments (need huge N) | Path-level statistics requiring individual histories |

**Rule of thumb:** If you need aggregate steady-state moments, ergodic
distributions, or linearized impulse responses (Jacobians), TM is faster
and more precise. If you need agent-level panel data, distributional
statistics beyond means, or path-level randomness, MC is the right tool.

## Core Idea

Both methods solve the same model — the difference is how they
propagate the *distribution* of agents forward in time.

- **MC:** Track N agents through stochastic shocks. The distribution is an
  empirical measure: \(\hat{\mu}_t^N = \frac{1}{N}\sum_{i=1}^N \delta_{x_t^{(i)}}\).
- **TM:** Discretize the state space onto a grid of M points. The
  distribution is a probability vector \(\mathbf{p}_t \in \mathbb{R}^M\)
  that evolves deterministically: \(\mathbf{p}_{t+1} = \boldsymbol{\Pi}\, \mathbf{p}_t\).

The transition matrix \(\boldsymbol{\Pi}\) is built from the same policy
functions and shock distributions that MC uses for individual agents.
Each column of \(\boldsymbol{\Pi}\) describes, for an agent currently at
grid point \(i\), the probability of landing at each grid point \(j\)
next period — integrating over all possible shocks.


## The Lottery Method (How Grid Projection Works)

When an agent at grid point \(g_i\) receives shocks and transitions to
next-period state \(x'\), the value \(x'\) will generally fall *between*
grid points. TM assigns probability to the two nearest grid points in a
mean-preserving way:

If \(g_j \leq x' \leq g_{j+1}\):

```
ω = (x' - g_j) / (g_{j+1} - g_j)
Prob(land at g_{j+1}) = ω
Prob(land at g_j)     = 1 - ω
```

This preserves the conditional mean (\(\mathbb{E}[g] = x'\)) while
introducing a small conditional variance reduction — the source of TM's
discretization bias. The bias shrinks as the grid becomes finer.

In HARK, the lottery projection is implemented by:
- `jump_to_grid_1D(m_vals, probs, dist_mGrid)` — for 1D state spaces
- `jump_to_grid_2D(m_vals, perm_vals, probs, dist_mGrid, dist_pGrid)` — for 2D (m, p)

Both are Numba-compiled and live in `HARK.utilities`.


## Conversion Workflow: Single-State Models

This is the simplest case. Classes: `IndShockConsumerType` (via
`NewKeynesianConsumerType`).

### Step 1: Solve the model (same as MC)

```python
from HARK.ConsumptionSaving.ConsNewKeynesianModel import NewKeynesianConsumerType

agent = NewKeynesianConsumerType()
agent.cycles = 0
agent.solve()
```

### Step 2: Enable the neutral measure (Harmenberg trick)

When `PermGroFac ≠ 1`, the full state space is 2D: (normalized market
resources *m*, permanent income level *p*). A 2D TM is expensive and
suffers from p-grid truncation errors (~30% on level aggregates).

Harmenberg (2021) defines a *neutral measure* that reweights permanent
shock probabilities as \(q^*(\psi_k) = \psi_k \cdot q(\psi_k)\). Under
this measure, the chain on normalized *m* alone is Markov, collapsing
the grid back to 1D. Level aggregates recover via:

\[
\bar{C}_{\text{level}} = \mathbb{E}^*[c(m)] \times \overline{p}
\]

where \(\overline{p}\) is computed analytically.

```python
agent.neutral_measure = True
agent.construct("IncShkDstn", "TranShkDstn", "PermShkDstn")
```

**Critical:** The neutral measure must only be used for the TM
construction step, *never* for the Bellman equation / solver. Solve with
the true shock distribution first; then switch to the neutral measure
for distribution propagation.

**Subtlety:** Under the neutral measure, \(\mathbb{E}^*[c(m)] \neq
\mathbb{E}[c(m)]\). The neutral-measure expectation computes the
level-weighted aggregate \(\mathbb{E}[c(m) \cdot p] / \mathbb{E}[p]\),
not the plain cross-sectional mean. The difference arises because
\(\text{cov}(c(m), p) < 0\).

### Step 3: Define the distribution grid

```python
agent.define_distribution_grid()
```

This builds a multi-exponentially-spaced grid over *m* (and *p* if not
using the neutral measure). Key parameters:

| Parameter | Default source | Effect |
|-----------|---------------|--------|
| `mMin` | `agent.mMin` | Lower bound of m-grid |
| `mMax` | `agent.mMax` | Upper bound of m-grid |
| `mCount` | `agent.mCount` | Number of grid points |
| `mFac` | `agent.mFac` | Exponential nesting depth (0=exponential, -1=linear) |
| `m_density` | 0 | Midpoint-insertion passes for grid refinement |

Or pass a custom grid directly:

```python
import numpy as np
my_grid = np.linspace(0.001, 50.0, 500)
agent.define_distribution_grid(dist_mGrid=my_grid)
```

### Step 4: Build the transition matrix

```python
agent.calc_transition_matrix()
```

This constructs `agent.tran_matrix` (an M×M or M\*P × M\*P matrix),
plus `agent.cPol_Grid` and `agent.aPol_Grid` (policy functions evaluated
on the grid).

Under the hood, this calls `gen_tran_matrix_1D` (neutral measure / 1D)
or `gen_tran_matrix_2D` (full 2D), both Numba-parallelized.

### Step 5: Find the ergodic distribution

```python
agent.calc_ergodic_dist()
```

Finds the eigenvector of the TM with eigenvalue 1 using
`scipy.sparse.linalg.eigs`. Stored as `agent.vec_erg_dstn` (flat
vector) and `agent.erg_dstn` (reshaped array).

### Step 6: Compute aggregates

```python
C_ss = np.dot(agent.cPol_Grid, agent.vec_erg_dstn.flatten())
A_ss = np.dot(agent.aPol_Grid, agent.vec_erg_dstn.flatten())
```

Or use the all-in-one pipeline:

```python
agent.compute_pe_steady_state()
print(agent.A_ss, agent.C_ss)
```

### Complete example: MC vs TM comparison

```python
import numpy as np
from HARK.ConsumptionSaving.ConsNewKeynesianModel import NewKeynesianConsumerType

agent = NewKeynesianConsumerType()
agent.cycles = 0
agent.solve()

# --- MC path ---
agent.T_sim = 1200
agent.AgentCount = 50000
agent.track_vars = ["aNrm", "cNrm", "pLvl"]
agent.initialize_sim()
agent.simulate()

A_mc = np.mean(agent.history["aNrm"][400:] * agent.history["pLvl"][400:])

# --- TM path ---
agent.neutral_measure = True
agent.construct("IncShkDstn", "TranShkDstn", "PermShkDstn")
agent.define_distribution_grid()
agent.calc_transition_matrix()
agent.calc_ergodic_dist()

A_tm = np.dot(agent.aPol_Grid, agent.vec_erg_dstn.flatten())

print(f"MC aggregate assets:  {A_mc:.6f}")
print(f"TM aggregate assets:  {A_tm:.6f}")
```


## Conversion Workflow: Markov Models

Classes: `MarkovConsumerType`. The state space is now (m, j) where j
indexes J discrete Markov states (e.g., employed/unemployed).

### Key difference: block-structured TM

The TM is (M\*J) × (M\*J), organized in J² blocks. Block (j→j')
encodes the probability that an agent in Markov state j at grid point
\(g_i\) ends up in Markov state j' at grid point \(g_k\), weighted by
the Markov transition probability `MrkvArray[j, j']`.

Index mapping: state (m-index=i, Markov-index=j) maps to flat index
`j * M + i`.

### The constructor override gotcha

HARK's constructor system automatically rebuilds `MrkvArray` from
`Mrkv_p11` and `Mrkv_p22` via `make_simple_binary_markov()`, silently
overriding any `MrkvArray` you pass directly. Two workarounds:

1. Pass the constructor's input parameters instead:
   ```python
   params["Mrkv_p11"] = 0.95
   params["Mrkv_p22"] = 0.95
   ```

2. Disable the constructor and pass `MrkvArray` directly:
   ```python
   params["constructors"]["MrkvArray"] = None
   params["MrkvArray"] = [my_custom_matrix]
   ```

**Diagnostic:** If MC simulation shows Markov state fractions that don't
match your intended matrix (e.g., 85.7% in state 0 instead of 50%),
the constructor has overridden your matrix.

### Markov TM workflow

```python
from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType

agent = MarkovConsumerType()
agent.cycles = 0
agent.solve()

# Enable neutral measure
agent.neutral_measure = True
agent.construct("IncShkDstn", "TranShkDstn", "PermShkDstn")

# Build grid and TM
agent.define_distribution_grid()
agent.calc_transition_matrix()
agent.calc_ergodic_dist()

# Aggregates by Markov state
M = len(agent.dist_mGrid)
J = agent.MrkvArray[0].shape[0]
dstn = agent.vec_erg_dstn.flatten()

for j in range(J):
    dstn_j = dstn[j * M : (j + 1) * M]
    C_j = np.dot(agent.cPol_Grid[j], dstn_j)
    print(f"State {j}: C = {C_j:.6f}, mass = {dstn_j.sum():.4f}")
```

Or all at once:

```python
A_ss, C_ss = agent.compute_pe_steady_state()
```

### Income parameters for Markov models

Several income-related parameters must be arrays (one value per Markov
state) rather than scalars:

- `Rfree`: list of J interest factors
- `PermGroFac`: list of J permanent income growth factors
- `LivPrb`: list of J survival probabilities
- `PermShkStd`, `TranShkStd`: arrays of shape (T_cycle, J)

When building temporary finite-horizon agents (e.g., for Jacobians),
replicate these with `np.tile`.


## Conversion Workflow: General Equilibrium (Krusell-Smith)

In Krusell-Smith and similar GE models, prices and aggregate variables
are *endogenous* — they depend on the distribution, which depends on
prices. The consumption function is 2D: `cFunc[j](m, M)` where M is
aggregate market resources.

### TM-in-KS: Forward propagation with `make_history_tm()`

The `CobbDouglasMarkovEconomy` class has a `make_history_tm()` method
that replaces MC forward propagation in the KS loop:

```python
from HARK.ConsumptionSaving.ConsAggShockModel import (
    KrusellSmithType, KrusellSmithEconomy
)

agent = KrusellSmithType()
agent.cycles = 0
agent.AgentCount = 5000

economy = KrusellSmithEconomy(agents=[agent])
economy.max_loops = 10
economy.verbose = True

# Standard MC-based KS solve
economy.solve()

# After solving, generate TM-based aggregate history for comparison
M_tm, A_tm = economy.make_history_tm(num_pointsM=200, mMax=50.0)
```

Each period, `make_history_tm()`:
1. Evaluates the 2D policy `cFunc[j](m_grid, M_current)` at the current
   aggregate state
2. Builds a 1D TM for that period
3. Propagates the distribution vector forward by one step
4. Computes aggregate M and A from the new distribution

This is ~100× faster than MC for forward propagation (e.g., 2.5s vs
243s over 11,000 periods with 5,000 MC agents).

### Fixing M to build a steady-state TM

For diagnostic purposes, you can build a TM at a fixed aggregate state:

```python
M_fixed = 10.5  # fixed aggregate market resources
for j in range(J):
    c_j = cFunc[j](dist_mGrid, M_fixed * np.ones_like(dist_mGrid))
    a_j = dist_mGrid - c_j
```

A steady-state TM (fixed M) gives ~0.89 correlation with the MC
trajectory. The full `make_history_tm()` achieves ~0.997 correlation by
updating M each period.


## Sequence-Space Jacobians (SSJ)

TM methods are a prerequisite for computing SSJ Jacobians (Auclert et al.
2021), which linearize heterogeneous-agent models around steady state.

### For single-state models

```python
agent = NewKeynesianConsumerType()
agent.compute_pe_steady_state()

T = 300
CJAC_Rfree, AJAC_Rfree = agent.calc_jacobian("Rfree", T)
```

### For Markov models

```python
agent = MarkovConsumerType()
agent.compute_pe_steady_state()

T = 300
CJAC, AJAC = agent.calc_jacobian("Rfree", T)
```

The Jacobian computation:
1. Creates a temporary T-period finite-horizon agent
2. Perturbs the parameter at period T-1 by dx=0.0001
3. Solves the perturbed agent
4. Builds per-period TMs and policies
5. Applies the Fake News decomposition:
   \(J = F_0 + \sum_{s \geq 1} F_s \, \mathcal{D} \, \mathcal{P}^{s-1}\)


## Diagnostic Checklist

Use these checks to validate TM results before trusting them.

### 1. Column sums

The TM should be column-stochastic (HARK convention for the TM itself,
distinct from the row-stochastic MrkvArray). Every column should sum to 1:

```python
col_sums = agent.tran_matrix.sum(axis=0)
print(f"Column sums: min={col_sums.min():.10f}, max={col_sums.max():.10f}")
assert np.allclose(col_sums, 1.0, atol=1e-10)
```

**Diagnostic patterns:**
- Sums ≈ 0.5 and 1.5 → Markov indexing is transposed (row vs column
  stochastic confusion)
- Sums < 1 everywhere → Missing death/rebirth contribution
- Sums slightly off → Numerical edge effects at grid boundaries

### 2. Ergodic Markov fractions

The marginal distribution over Markov states should match the analytical
stationary distribution:

```python
M = len(agent.dist_mGrid)
J = agent.MrkvArray[0].shape[0]
dstn = agent.vec_erg_dstn.flatten()

for j in range(J):
    print(f"State {j} mass: {dstn[j*M:(j+1)*M].sum():.6f}")

# Compare with analytical stationary distribution
pi = MarkovConsumerType._calc_markov_stationary(agent.MrkvArray[0])
print(f"Analytical: {pi}")
```

### 3. MC vs TM aggregate comparison

Run both methods and compare steady-state aggregates:

```python
# TM aggregates
agent.compute_pe_steady_state()
A_tm, C_tm = agent.A_ss, agent.C_ss

# MC aggregates (after long burn-in)
agent.neutral_measure = False
agent.construct("IncShkDstn", "TranShkDstn", "PermShkDstn")
agent.T_sim = 1200
agent.AgentCount = 50000
agent.initialize_sim()
agent.simulate()

A_mc = np.mean(agent.history["aNrm"][400:] * agent.history["pLvl"][400:])
print(f"TM: A={A_tm:.6f}, MC: A={A_mc:.6f}")
```

### 4. Boundary mass

Check that negligible probability mass sits at the grid boundaries:

```python
dstn = agent.vec_erg_dstn.flatten()
M = len(agent.dist_mGrid)
for j in range(J):
    block = dstn[j*M:(j+1)*M]
    print(f"State {j}: left edge = {block[0]:.2e}, right edge = {block[-1]:.2e}")
```

If significant mass is at the boundaries, extend the grid (`mMax`) or
increase resolution (`mCount`).


## Common Pitfalls

### 1. PermShk includes PermGroFac

In HARK's `get_shocks()`, the permanent shock stored in
`self.shocks["PermShk"]` is the *composite* of the raw idiosyncratic
shock and `PermGroFac`:

```python
PermShkNow = IncShkDstn.atoms[0] * PermGroFacNow  # composite
```

When building a TM by hand, you must replicate this:

```python
mNext = Rfree[jp] * a / (raw_perm_shk * PermGroFac[jp]) + tran_shk
```

The `jp` index is the *target* (destination) Markov state, not the source
state — matching HARK's simulation timing where next-period prices
depend on where you end up.

### 2. Row-stochastic MrkvArray vs column-stochastic TM

HARK's `MrkvArray` is **row-stochastic**: `MrkvArray[i, j]` = P(go to
state j | in state i). Rows sum to 1.

HARK's TM from `gen_tran_matrix_*` is **column-stochastic**: column `i`
gives the distribution of next-period states for an agent currently at
state `i`. Columns sum to 1.

Mixing these conventions leads to incorrect TMs. When in doubt, check the
`MarkovProcess.draw()` method which reads `transition_matrix[s, :]`
(confirming row-stochastic for MrkvArray).

### 3. Off-by-one in forward iteration

When propagating distributions forward (e.g., for impulse responses),
compute aggregates *before* transitioning:

```python
# CORRECT
for t in range(T):
    C[t] = np.dot(c_policy[t], dstn)       # aggregate FIRST
    A[t] = np.dot(a_policy[t], dstn)
    dstn = TM[t] @ dstn                     # THEN transition

# WRONG (shifted by one period)
for t in range(T):
    dstn = TM[t] @ dstn                     # transition FIRST
    C[t] = np.dot(c_policy[t], dstn)        # aggregate is one period late
```

### 4. Newborn transitory shock suppression

HARK forces `TranShk = 1.0` for agents with `t_age = 0` when
`NewbornTransShk = False` (the default). This biases the first-period MC
distribution. Workaround after initialization:

```python
agent.initialize_sim()
agent.t_age = np.ones(agent.AgentCount, dtype=int)
```

For TM construction, newborns are projected onto the grid via
`jump_to_grid_1D` starting at `m = 1.0` (normalized), which implicitly
handles this correctly.

### 5. Grid convergence

TM results depend on grid resolution. Always run a grid sweep:

```python
for n_pts in [100, 200, 500, 1000, 2000]:
    agent.define_distribution_grid(num_pointsM=n_pts)
    agent.calc_transition_matrix()
    agent.calc_ergodic_dist()
    A = np.dot(agent.aPol_Grid, agent.vec_erg_dstn.flatten())
    print(f"M={n_pts}: A_ss = {A:.8f}")
```

Typically 200–500 grid points suffice for 4–6 digit accuracy.


## API Reference Summary

### `NewKeynesianConsumerType` (single-state)

| Method | Purpose |
|--------|---------|
| `define_distribution_grid(dist_mGrid, dist_pGrid, m_density, num_pointsM, num_pointsP, max_p_fac)` | Build m-grid (and p-grid if not neutral measure) |
| `calc_transition_matrix(shk_dstn)` | Build M×M (or M\*P × M\*P) TM |
| `calc_ergodic_dist(transition_matrix)` | Find stationary distribution via eigendecomposition |
| `compute_pe_steady_state()` | All-in-one: solve → neutral measure → grid → TM → ergodic → A_ss, C_ss |
| `calc_jacobian(shk_param, T)` | T×T SSJ Jacobians via Fake News Algorithm |

### `MarkovConsumerType` (discrete Markov states)

| Method | Purpose |
|--------|---------|
| `define_distribution_grid(dist_mGrid, num_pointsM, timestonest, m_density)` | Build 1D m-grid; state space is (m, j) with M\*J points |
| `calc_transition_matrix(shk_dstn)` | Build (M\*J) × (M\*J) block TM |
| `calc_ergodic_dist(transition_matrix)` | Ergodic distribution over full (m, j) space |
| `compute_pe_steady_state()` | Full pipeline for Markov models |
| `calc_jacobian(shk_param, T)` | SSJ Jacobians for Markov models |

### `CobbDouglasMarkovEconomy` (KS general equilibrium)

| Method | Purpose |
|--------|---------|
| `make_history_tm(num_pointsM, mMax)` | TM-based forward propagation in KS loop |

### Low-level utilities (`HARK.utilities`)

| Function | Purpose |
|----------|---------|
| `jump_to_grid_1D(m_vals, probs, dist_mGrid)` | Mean-preserving lottery onto 1D grid |
| `jump_to_grid_2D(m_vals, perm_vals, probs, dist_mGrid, dist_pGrid)` | Mean-preserving lottery onto 2D grid |
| `gen_tran_matrix_1D(dist_mGrid, bNext, shk_prbs, perm_shks, tran_shks, LivPrb, NewBornDist)` | 1D TM (single-state or neutral measure) |
| `gen_tran_matrix_2D(...)` | 2D TM for (m, p) state space |
| `gen_tran_matrix_1D_markov(dist_mGrid, aPol_Grid, MrkvArray, Rfree_arr, PermGroFac_arr, LivPrb_arr, shk_prbs, perm_shks, tran_shks, NewBornDist)` | Block TM for Markov models |
| `make_grid_exp_mult(ming, maxg, ng, timestonest)` | Multi-exponentially spaced grid |


## Learning Path

The `examples/MonteCarlovsTransitionMatrix/` directory contains three
notebooks that build intuition step by step:

| Notebook | What it teaches |
|----------|-----------------|
| `PE_MarkovConsumerType.ipynb` | Part 1: 4-state unemployment (1D grid). Part 2: PermGroFac ≠ 1 on 2D grid. Part 3: Harmenberg neutral measure collapses back to 1D. |
| `GE_KrusellSmith.ipynb` | Part 1: TM with endogenous aggregate state (KS). Part 2: `make_history_tm()` forward propagation in KS loop. |
| `Validation_and_SSJ.ipynb` | Part 1: Validates production `MarkovConsumerType` TM methods. Part 2: Sequence-space Jacobians via Fake News Algorithm. |

Also see `examples/SequenceSpaceJacobians/Transition_Matrix_Example.ipynb`
for MC vs TM head-to-head comparisons with MSE decomposition and
Harmenberg demonstrations.


## Minor prerequisite changes

### `NewKeynesianConsumerType`: model YAML pointer

`NewKeynesianConsumerType` was the only `IndShockConsumerType` subclass
missing a `"model"` key in its `default_` dict.  This PR adds
`"model": "ConsIndShock.yaml"` (one line) so that `initialize_sym()` can
build an `AgentSimulator` — the same YAML-driven simulation backend that
every other HARK consumer type already supports.  Without this,
`Transition_Matrix_Example.ipynb` (which uses `NewKeynesianConsumerType`
for its MC-vs-TM comparisons) would fail when calling the new
AgentSimulator API.

The notebook uses `NewKeynesianConsumerType` rather than plain
`IndShockConsumerType` for consistency with the existing SSJ example
notebooks in the same directory, which frame the agent problem in
HANK/New Keynesian terms.  The two classes share identical dynamics;
`NewKeynesianConsumerType` simply passes additional aggregate labor
income variables into the income process.

## References

- Harmenberg, K. (2021). "Aggregation with a permanent income neutral
  measure." *Journal of Economic Dynamics and Control*.
- Auclert, A., Bardóczy, B., Rognlie, M., & Straub, L. (2021). "Using
  the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent
  Models." *Econometrica*.
- Young, E. R. (2010). "Solving the Incomplete Markets Model with
  Aggregate Uncertainty using the Krusell-Smith Algorithm and Non-Stochastic
  Simulations." *Journal of Economic Dynamics and Control*.
- Den Haan, W. J. (2010). "Comparison of Solutions to the Incomplete
  Markets Model with Aggregate Uncertainty." *Journal of Economic Dynamics
  and Control*.

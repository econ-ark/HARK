# Lessons Learned: Transition Matrix Methods for Markov Models

*Accumulated across notebooks 01–09 and the production code in Phases A–D.*

---

## Part I — Bugs and near-misses (from notebook 1: `markov-tm-prototype`)

### Bug 1: Markov matrix indexing convention (row- vs column-stochastic)

**Symptom:** Transition matrix column sums ranged from 0.5 to 1.5 instead of 1.0.

**Root cause:** HARK uses **row-stochastic** Markov matrices — `MrkvArray[i, j]` =
P(go to state j | in state i) — but the prototype code assumed **column-stochastic**
(`MrkvArray[j', j]` = P(j → j')).  This is a common trap because much of the
mathematical literature on Markov chains writes the transition kernel as a
column-stochastic operator acting on distribution vectors from the left.

**Evidence chain that would have caught it:**
1. `MarkovProcess.draw()` samples via `transition_matrix[s, :]` — reading row s
2. `make_simple_binary_markov` builds `[[p11, 1-p11], [1-p22, p22]]` — rows sum to 1
3. With the default `Mrkv_p11=0.9, Mrkv_p22=0.4`, the matrix is `[[0.9, 0.1], [0.6, 0.4]]` — column 0 sums to 1.5

**Proposed source code improvements:**

1. **`MarkovProcess` class docstring** (`HARK/distributions/base.py:172`):
   The current docstring says only "An array of floats representing a probability mass
   for each state transition."  It should explicitly state:

   > `transition_matrix` is row-stochastic: `transition_matrix[i, j]` =
   > P(transition to state j | currently in state i).  Rows must sum to 1.

2. **`make_simple_binary_markov` docstring** (`HARK/ConsumptionSaving/ConsMarkovModel.py:61`):
   The Returns section says "List of 2x2 Markov transition arrays" but does not
   state the convention.  Add:

   > Each array is row-stochastic: `MrkvArray[i, j]` = P(go to state j | in state i).
   > Row 0 = `[p11, 1-p11]`, Row 1 = `[1-p22, p22]`.

3. **`markov_constructor_dict`** (`ConsMarkovModel.py:687`):
   Add a one-line comment: `# Builds row-stochastic MrkvArray from Mrkv_p11, Mrkv_p22`

---

### Bug 2: Constructor silently overrides explicit parameter

**Symptom:** Passing `MrkvArray=[my_matrix]` in the params dict had no effect.
The agent used a completely different Markov matrix built from default `Mrkv_p11=0.9,
Mrkv_p22=0.4`.

**Root cause:** HARK's constructor system maps attribute names to builder functions.
`markov_constructor_dict["MrkvArray"] = make_simple_binary_markov` means the
constructor always runs `make_simple_binary_markov(T_cycle, Mrkv_p11, Mrkv_p22)` and
assigns the result to `self.MrkvArray`, regardless of whether `MrkvArray` was
explicitly passed.

**How it was detected:** MC simulation showed 85.7% of agents in state 0, which
matches the stationary distribution of the *default* Markov matrix `[[0.9, 0.1],
[0.6, 0.4]]` (π₀ = 6/7 ≈ 0.857), not the intended symmetric `[[0.9, 0.1],
[0.1, 0.9]]` (π₀ = 0.5).

**Proposed source code improvements:**

1. **`init_indshk_markov` dict** (`ConsMarkovModel.py:748`):
   Add a comment block above it explaining the constructor pattern:

   > Parameters in this dict are either (a) used directly as agent attributes,
   > or (b) consumed by constructor functions listed in `constructors`.
   > Constructor-built attributes (like `MrkvArray`) cannot be overridden by
   > passing them directly — you must pass the constructor's *input* params
   > instead (e.g., `Mrkv_p11`, `Mrkv_p22` for the Markov matrix).

2. **Defensive check in the Model base class** (`HARK/model.py`):
   When a constructor builds an attribute that was also explicitly passed in params,
   emit a warning:
   ```python
   import warnings
   if attr_name in user_params and attr_name in self.constructors:
       warnings.warn(
           f"Parameter '{attr_name}' was passed explicitly but will be "
           f"overridden by constructor '{self.constructors[attr_name].__name__}'. "
           f"Pass the constructor's input params instead.",
           stacklevel=2,
       )
   ```

---

### Near-miss: PermShk includes PermGroFac

**Not a bug in this case** (because we used PermGroFac = 1.0), but would have been
a bug if PermGroFac differed by state.

In `get_shocks()` (line 1021):
```python
PermShkNow[these] = IncShkDstnNow.atoms[0][EventDraws] * PermGroFacNow
```

The raw permanent shock from the distribution is multiplied by `PermGroFac` before
being stored.  The transition equation then uses this composite:
```python
bNrm = Rfree * state_prev["aNrm"] / self.shocks["PermShk"]
```

This means that when building a transition matrix, you must replicate this:
```python
mNext = Rfree[jp] * a / (raw_perm_shk * PermGroFac[jp]) + tran_shk
```

**Proposed improvement:** Add a comment in `get_shocks()` at the PermShk line:
```python
# PermShk combines the raw idiosyncratic shock with PermGroFac.
# TM construction must replicate: mNext = R*a / (raw_psi * PermGroFac) + theta
PermShkNow[these] = IncShkDstnNow.atoms[0][EventDraws] * PermGroFacNow
```

---

---

## Part II — Lessons from scaling up (notebooks 3–6)

### Lesson 4: PermGroFac ≠ 1 forces a 2D grid or the Harmenberg neutral measure

**Discovered in:** `serial-growth-tm-2d` (notebook 3)

When `PermGroFac` varies across Markov states, the distribution of permanent
income `p` is non-degenerate.  The state space becomes `(m, p, j)` and the
transition matrix grows as `(M × P × J)²`.  With M=80, P=25, J=5 the TM
has 10,000 states and ~800 MB of memory.

Worse, **p-grid truncation** causes large errors in level aggregates (~30%)
because the ergodic distribution of `p` has a long right tail that extends
beyond any practical grid.

**Resolution:** Harmenberg's permanent-income-neutral measure (notebook 4)
collapses the grid back to `(m, j)` by reweighting permanent shock
probabilities: `P*(ψ_k) = ψ_k · P(ψ_k)`.  The PermGroFac factor remains in
the transition formula as a known constant:
```python
mNext = R[jp] * a / (perm_shks * PermGroFac[jp]) + tran_shks
```
Level aggregates recover via `C_level = E*[c(m)] × MeanPLvl`, where `MeanPLvl`
can be computed analytically.

### Lesson 5: The neutral measure computes E[c·p]/E[p], not E[c]

**Discovered in:** `tm-consolidation` (notebook 5)

Under the neutral measure, the ergodic distribution π*(m) is the *p-weighted*
cross-sectional distribution, not the plain cross-sectional distribution.
Therefore:

- `C_ss = E*[c(m)] = E[c(m)·p] / E[p]` — level-weighted normalized aggregate
- This is **not** the same as `E[c(m)]` (plain cross-sectional mean)

The difference arises because `cov(c(m), p) < 0`: agents who received large
permanent shocks have high `p` but low normalized `m` (and hence low `c(m)`).

MC estimates of `E[c·p]/E[p]` are extremely noisy because rare agents with
very high `p` dominate the numerator.  This is precisely the problem the
neutral measure solves for the TM — it computes the level aggregate exactly
without ever discretizing `p`.

### Lesson 6: AggShock cFunc is 2D — fix M to build a TM

**Discovered in:** `agg-shock-markov-tm` (notebook 6)

In `AggShockMarkovConsumerType`, the consumption function depends on both
individual market resources and aggregate market resources: `cFunc[j](m, M)`.
To build a 1D transition matrix, you must evaluate the policy at a fixed M:
```python
c_j = cFunc[j](dist_mGrid, M_fixed * np.ones(M_grid))
```
Prices R and W are also M-dependent (via Cobb-Douglas production):
```python
R = Rfunc(K);  W = wFunc(K);  where K = AFunc[j](M)
```
A steady-state TM (fixed M) gives a correlation of ~0.89 with the MC
trajectory.  A full TM-in-KS implementation would build TMs at a grid of
M values and interpolate, achieving near-perfect tracking.

---

## Part III — Process lessons

### 1. Always verify the agent's actual attributes after construction

Before building any derived computation (transition matrix, custom simulation),
print the agent's actual constructed attributes:
```python
print(agent.MrkvArray[0])      # Verify Markov matrix
print(agent.Rfree[0])          # Verify interest rates
print(agent.PermGroFac[0])     # Verify growth factors
```
Don't trust that params you passed made it through the constructor system unchanged.

### 2. Use MC simulation as a sanity check for TM results

The MC simulation uses HARK's tested code paths. Before debugging TM code, verify
that MC aggregates and state fractions match expectations (e.g., Markov stationary
distribution). If MC fractions don't match your intended Markov matrix, the problem
is in model setup, not TM construction.

### 3. Column sum validation is the first diagnostic

If transition matrix column sums ≠ 1.0, the matrix is wrong. The magnitude and
pattern of the deviation often diagnoses the bug:
- Sums ≈ 0.5 and 1.5 → Markov indexing is transposed (row vs column stochastic)
- Sums < 1.0 everywhere → missing death/rebirth contribution
- Sums slightly off → numerical edge effects in the lottery method

### 4. Read the `draw()` / sampling code, not just the docstring

The `MarkovProcess.draw()` method unambiguously shows the convention via
`transition_matrix[s, :]`.  The docstring was vague.  When in doubt about
conventions, read the code that *consumes* the data structure.

### 5. Validate against the built-in pipeline before extending

When building hand-rolled TM code, first verify that it reproduces HARK's
existing `NewKeynesianConsumerType.calc_transition_matrix()` output exactly
(max element-wise diff = 0.0).  Only then extend to Markov states — and verify
that the Markov code with J=1 still matches the single-state built-in.

### 6. Start with PermGroFac=1.0, add complexity incrementally

The first three prototypes deliberately set `PermGroFac=1.0` across all
Markov states to keep the grid 1D.  This isolates TM construction bugs from
permanent-income dynamics bugs.  Only after the 1D case is validated should
you tackle PermGroFac ≠ 1 (either via 2D grid or Harmenberg).

---

## Part IV — Lessons from production code (Phases B–D)

### 7. Match the NK model's TM construction exactly before extending

When adding TM methods to `MarkovConsumerType` (Phase B), the first test was
an exact match: call `gen_tran_matrix_1D_markov` with J=1 using the
`NewKeynesianConsumerType`'s own policy function and verify element-wise
identity with its `tran_matrix`.  This isolated TM construction logic from
solver differences and caught a newborn-distribution convention mismatch
(newborns start at `m=1.0`, not `m=theta`).

### 8. Replicate income parameters correctly for finite-horizon agents

`calc_jacobian` (Phase D) creates temporary finite-horizon agents with
`T_cycle=T`.  Income process parameters like `PermShkStd` and `TranShkStd`
must be 2D arrays of shape `(T_cycle, K_states)` for Markov models.  Use
`np.tile` to replicate the infinite-horizon values to the correct shape.
Similarly, `UnempPrb` and `IncUnemp` must be proper arrays, not scalars.

### 9. Disable constructors when building temporary agents

When creating temporary agents for Jacobian computation, the `MrkvArray`
constructor (`make_simple_binary_markov`) expects `Mrkv_p11` and `Mrkv_p22`
of length `T_cycle`, which is impractical for large T.  Instead, set
`params["constructors"]["MrkvArray"] = None` and pass `MrkvArray` directly
as `T * [MrkvArr]`.

### 10. TM-in-KS is ~100× faster than MC for forward propagation

In the Krusell-Smith loop (Phase C), `make_history_tm()` replaced MC
simulation for forward-propagating the distribution.  With a 200-point
m-grid, TM propagation over 11,000 periods took ~2.5 seconds vs. ~243
seconds for MC with 5,000 agents.  Correlation between MC and TM aggregate
trajectories: 0.997 (M) and 0.995 (A).

---

## Part V — Proposed HARK source improvements

The following improvements to HARK's source code were identified during this
project but have **not yet been implemented**.  They are collected here so
they are not lost when the working documents are archived.

### Docstring fixes

1. **`MarkovProcess` class** (`HARK/distributions/base.py`):
   State that `transition_matrix` is row-stochastic: `transition_matrix[i, j]`
   = P(transition to state j | currently in state i).  Rows must sum to 1.

2. **`make_simple_binary_markov`** (`HARK/ConsumptionSaving/ConsMarkovModel.py`):
   State that each returned array is row-stochastic.  Document the layout:
   Row 0 = `[p11, 1-p11]`, Row 1 = `[1-p22, p22]`.

3. **`markov_constructor_dict`** (`ConsMarkovModel.py`):
   Add comment: `# Builds row-stochastic MrkvArray from Mrkv_p11, Mrkv_p22`.

4. **`get_shocks()` in `ConsMarkovModel.py`**:
   At the `PermShkNow` assignment, add a comment explaining that PermShk
   includes PermGroFac and that TM construction must replicate this.

### Defensive behavior

5. **Constructor override warning** (`HARK/model.py`):
   When a constructor builds an attribute that was also explicitly passed in
   the params dict, emit a warning so users know their value was silently
   replaced.

### Documentation

6. **Model-to-simulation-method inventory**:
   HARK has no single place listing which agent types support which simulation
   methods (MC, TM, SSJ).  A table in the docs or a top-level README would
   help users find the right class for their needs.  A draft table was
   created during this project (see `_archive/CONTEXT-FOR-AI.md` lines 89–111).

7. **HARK API-to-math mapping**:
   A table mapping HARK methods (`solve()`, `simulate()`,
   `define_distribution_grid()`, `calc_transition_matrix()`, etc.) to their
   mathematical operations would help users connect code to theory.  A draft
   was created during this project (see
   `_archive/REFERENCE-du-notebook-framework-mapping.md` Section 3).

---

## Part VI — Systematic Audit Fixes (applied across NB01–NB09 + mathematical-framework)

The following issues were identified during a systematic audit of all
`sims-about/` notebooks, modeled on the 17 categories of fixes applied to
`Transition_Matrix_Example.ipynb`.  This section documents which fixes were
applied to which notebooks.

### Fix A: `t_age` newborn transitory shock suppression

**Applied to:** NB01, NB02, NB03, NB04, NB05

**Problem:** HARK's `get_shocks()` forces `TranShk = 1.0` for agents with
`t_age = 0` when `NewbornTransShk = False`.  This biases the first period
of MC simulation for all newborn agents.

**Fix:** After `agent.initialize_sim()`, add:
```python
agent.t_age = np.ones(agent.AgentCount, dtype=int)
```

**Not applicable to:** NB06 (MC runs internally via `econ.solve()`), NB07
(no MC simulation), NB08 (MC runs internally), NB09 (no MC simulation).

### Fix B: Probability density vs probability mass in distribution plots

**Applied to:** NB01, NB02, NB03, NB04

**Problem:** Distribution comparison plots computed MC histograms as
`h / h.sum()` (probability mass per bin), then overlaid TM probability mass
at grid points.  On non-uniform grids (exponentially spaced), this produces
misleading distribution shapes — bins with larger widths appear to have
more probability.

**Fix:** Convert both TM and MC distributions to density:
- TM: divide probability mass by midpoint bin widths
- MC: use `plt.hist(..., density=True)` with uniform bins

### Fix C: MC vs TM timing instrumentation

**Applied to:** NB01, NB02, NB03, NB04, NB05, NB06

**Problem:** No timing comparisons between MC and TM methods, so users
could not assess the practical speedup of TM over MC.

**Fix:** Wrap MC simulation and TM build/ergodic computation in `time.time()`
calls.  Print a summary:
```
--- Timing Summary ---
MC simulation:    X.XXs (N agents)
TM build + ergo:  X.XXs (M m-pts × J states)
Speedup:          X.X×
```

### Fix D: Figure naming and `source_hidden` metadata

**Applied to:** NB01, NB02, NB03, NB04, NB05, NB06, NB08

**Problem:** Plotting cells had no canonical names and were not configured
to hide source code in JupyterLab.

**Fix:** Added `# [descriptive_snake_case_name]` as the first line of each
plotting cell.  Set `"jupyter": {"source_hidden": true}` in cell metadata.

### Fix E: Standardized MC/TM plot colors and legends

**Applied to:** NB01, NB02, NB03, NB04, NB05, NB06, NB08

**Problem:** MC and TM lines used inconsistent colors across notebooks.
Legends did not indicate grid resolution or agent count.

**Fix:** Defined module-level constants `COLOR_MC = "tab:blue"` and
`COLOR_TM = "tab:orange"`.  All MC vs TM comparison plots use these colors.
Legend labels include `(N agents)` for MC and `(M m-pts)` for TM.

### Fix F: Calibration documentation

**Applied to:** All notebooks (NB01–NB09) and `mathematical-framework.ipynb`

**Problem:** Notebooks used various parameter sets without documenting their
origin.

**Fix:** Added a paragraph to each notebook's model setup section identifying
the calibration source (e.g., "`init_indshk_markov` defaults," "Krusell &
Smith 1998," "pedagogical illustration") and noting any custom overrides.

### Fix G: `burn_in` replaced with named `BURNIN` constant

**Applied to:** NB01, NB02, NB03, NB04

**Problem:** Hard-coded `burn_in = 400` variable defined late in the notebook.

**Fix:** Defined `BURNIN = 400` in the imports cell alongside other constants.
Removed separate `burn_in` assignment.

### Fix H: Code vectorization

**Applied to:** NB03 (triple-nested loop for aggregates replaced with
vectorized `block @ dist_pGrid` computation)

### Fix I: R/Gamma indexing verification

**Verified in:** NB01, NB02, NB03, NB04 (all use target state `jp` for
`Rfree` and `PermGroFac`—correct per HARK convention)

**Also fixed in:** `mathematical-framework.ipynb` Section 10, where the
formula was corrected from `R_j / Γ_j` to `R_{j'} / Γ_{j'}`.

---

### Open Issues (documented but not resolved)

**NB08 — MC/TM level mismatch (~22%):**  The Krusell-Smith notebook shows
mean(M) = 13.0 for MC vs mean(M) = 10.1 for TM.  Hypothesized causes:
(1) Different distribution initialization, (2) Neutral-measure aggregation
needing MeanPLvl correction, (3) Insufficient MC convergence.  Documented
as a known limitation in the notebook.

**NB09 — Jacobian vs finite-difference disagreement (~28%):**  The SSJ
notebook's `calc_jacobian()` differs substantially from finite-difference
TM propagation.  Hypothesized causes: (1) Off-by-one in FD loop
(transition-then-compute vs compute-then-transition), (2) Perturbed agent
using steady-state policy grids, (3) Incorrect order of operations.
Documented as a known issue in the notebook.

---

### Mathematical Framework Corrections

**Applied to:** `mathematical-framework.ipynb`

1. **Neutral-measure pitfall warning** (Section 11): Added blockquote warning
   that neutral-measure income must only be used for TM construction, never
   for solving the Bellman equation.

2. **t_age newborn note** (Section 6): Added description of HARK's transitory
   shock suppression for `t_age=0` agents and the workaround.

3. **MIT shock terminology** (Section 8): Replaced "MIT shocks" with
   "anticipated deviations (perfect-foresight transition paths)" and added
   terminology note.

4. **R_{j'}/Γ_{j'} correction** (Sections 4, 10): Fixed indexing from source
   state to target state, matching HARK's simulation timing.

5. **Newborn distribution discussion** (Section 14): Added full discussion of
   `d_newborn` choices and `correct_newborn_dist`.

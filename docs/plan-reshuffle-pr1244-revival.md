# Plan: Revive and extend PR #1244-style reshuffling on current HARK

**Canonical file:** `docs/plan-reshuffle-pr1244-revival.md`
**Integration branch:** `harmenberg-dual-measure`
**Status:** Partially implemented (v3 — updated after code review of integration branch).
**Audience:** HARK maintainers and HAFiscal / downstream users.
**Related:** [PR #1244](https://github.com/econ-ark/HARK/pull/1244) (unmerged), [PR #1691](https://github.com/econ-ark/HARK/pull/1691) (merged: `exact_match` → `shuffle` on `DiscreteDistribution.draw`), [Issue #1690](https://github.com/econ-ark/HARK/issues/1690).

---

## What's already done (on `harmenberg-dual-measure`)

Before planning remaining work, here is what the integration branch already contains beyond `main`:

### 1. `AggIndMrkvConsumerType` — hierarchical Markov base class (DONE)

**File:** `HARK/ConsumptionSaving/ConsAggIndMarkovModel.py` (new, 301 lines)

- `MarkovConsumerType` subclass with two-step Markov draw:
  1. `get_macro_markov_states()` — reads aggregate state
  2. `get_micro_markov_states()` — draws idiosyncratic states (default: stochastic via `RNG.choice`)
  3. Combines: `shocks["Mrkv"] = N * MacroMrkv + MicroMrkv`
- Utility functions: `make_hierarchical_mrkv_array()`, `extract_cond_mrkv_arrays()`, `construct_MrkvIndArray()`
- Supports both "simple" (HAFiscal-style: micro depends on destination macro only) and "general" (KS-style: micro depends on both source and destination macro) conditional transition formats.

### 2. `KrusellSmithType` refactored onto `AggIndMrkvConsumerType` (DONE)

**File:** `HARK/ConsumptionSaving/ConsAggShockModel.py` (modified)

- `KrusellSmithType` now inherits from `AggIndMrkvConsumerType` instead of `AgentType`.
- Uses `shocks["MrkvAgg"]` instead of `shocks["Mrkv"]` for the macro state.
- Overrides `get_micro_markov_states()` with **exact-match permutation** for employment transitions (the KS boolean-deck pattern from PR #1244's intent).
- `KrusellSmithTypeHM` + `KrusellSmithEconomyHM` — reference implementations for verification against the original.
- `KrusellSmithEconomy` now stores `MacroMrkvArray` and `CondMrkvArrays` alongside `MrkvAggArray`/`MrkvIndArray`.

### 3. Newborn PermShk bug fix in `MarkovConsumerType` (DONE)

**File:** `HARK/ConsumptionSaving/ConsMarkovModel.py:1025-1034` (modified)

Previously `PermShkNow[newborn] = 1.0` suppressed **both** the idiosyncratic ψ shock **and** deterministic `PermGroFac` growth. Now applies per-state `PermGroFac[0][j]` to newborns, preserving calibrated cross-sectional dispersion while still allowing permanent income growth.

### 4. Tests (DONE)

- `tests/ConsumptionSaving/test_ConsMarkovModel.py`: new `test_NewbornPermShkIncludesGrowth` class (two tests).
- `examples/ConsAggShockModel/test_ks_hierarchical_markov.py`: comparison script verifying original KS matches refactored KS.

---

## What remains

### Current state of shuffle/draw in HARK

| Feature | Status | Location |
|---------|--------|----------|
| `DiscreteDistribution.draw(N, shuffle=True)` | Merged to `main` via #1691 | `distributions/discrete.py:205-277` |
| `draw_events()` shuffle support | None — CDF inversion only | `distributions/discrete.py:191-203` |
| `AggShockConsumerType.get_shocks()` | Uses `shuffle=True` unconditionally | `ConsAggShockModel.py:1124,1140` |
| `IndShockConsumerType.get_shocks()` | No shuffle. Non-newborns: `draw(N)`; newborns: `draw_events(N)` + atom indexing | `ConsIndShockModel.py:2134-2197` |
| `MarkovConsumerType.get_shocks()` | No shuffle. Uses `draw_events(N)` + atom indexing per `(t,j)` slice | `ConsMarkovModel.py:988-1029` |
| Agent-level `income_shuffle` parameter | Not implemented | — |

**Key asymmetry to watch:** `IndShock` non-newborns use `draw(N)` (returns values `[PermShk, TranShk]`), but IndShock newborns and all of `MarkovConsumerType` use `draw_events(N)` (returns **indices**, then manually indexes `atoms[0]`, `atoms[1]`). Switching to `draw(N, shuffle=...)` changes return semantics — the atom-indexing lines must be **replaced**, not just the draw call.

---

## Remaining work: two tasks

### Task A — Educational notebook (no simulation core changes)

**Location:** `examples/Distributions/Shuffle_vs_IID_Draws.ipynb`

**Content:**

- Builds small `DiscreteDistribution` instances (univariate and bivariate) and compares **histogram / total variation distance** for:
  - `draw(N, shuffle=False)` — standard i.i.d. MC
  - `draw(N, shuffle=True)` — exact-marginal shuffled
  - `draw_events(N)` — CDF inversion (reference)
- Shows convergence: shuffle achieves near-zero TV at any N; i.i.d. MC converges as O(1/sqrt(N)).
- States clearly: shuffle per draw batch ≈ exact marginal match to `pmv`; it does **not** fix joint state issues (e.g. TM-init `(p,m,j)` in HAFiscal).
- Pins HARK version / commit in a markdown cell.

**Acceptance:** `nbconvert --execute` succeeds; no API change; no new parameters.

---

### Task B — Opt-in income shuffling for `IndShockConsumerType` + `MarkovConsumerType`

**Design decision — opt-in vs unconditional:**
`AggShockConsumerType` already uses `shuffle=True` unconditionally. For `IndShock` and `Markov`, we add `income_shuffle` (default `False`) to preserve backward compatibility. If experience shows shuffle is always beneficial, a future change can flip the default.

**Content:**

1. **`IndShockConsumerType`** (`ConsIndShockModel.py:2134-2197`)
   - Add boolean parameter `income_shuffle` (default `False`).
   - **Non-newborn path** (line 2168): trivial — `draw(N)` → `draw(N, shuffle=self.income_shuffle)`. Return shape is already `[PermShk, TranShk]`; no other changes needed.
   - **Newborn path** (lines 2178-2189): currently uses `draw_events(N)` returning indices, then manually indexes `atoms[0]` and `atoms[1]`. Two options:
     - **(a) Recommended:** Switch to `draw(N, shuffle=self.income_shuffle)` returning values directly (matches the `AggShockConsumerType` pattern). Delete the manual atom-indexing. Preserve `NewbornTransShk` override afterward.
     - **(b) Conservative:** Leave newborn path as-is (i.i.d. for small-N newborn cohorts where shuffle has minimal benefit).

2. **`MarkovConsumerType`** (`ConsMarkovModel.py:1004-1024`)
   - Inherits `income_shuffle` from `IndShockConsumerType`.
   - **Per-slice draw** (line 1020): Replace:
     ```python
     EventDraws = IncShkDstnNow.draw_events(N)
     PermShkNow[these] = IncShkDstnNow.atoms[0][EventDraws] * PermGroFacNow
     TranShkNow[these] = IncShkDstnNow.atoms[1][EventDraws]
     ```
     with:
     ```python
     ShockDraws = IncShkDstnNow.draw(N, shuffle=self.income_shuffle)
     PermShkNow[these] = ShockDraws[0] * PermGroFacNow
     TranShkNow[these] = ShockDraws[1]
     ```
     This mirrors the `AggShockConsumerType` pattern exactly.
   - **Newborn handling** (lines 1025-1034 on this branch): Now applies per-state `PermGroFac[0][j]`; no draw to change. Leave as-is.
   - **Caution:** Per-slice N can be small (few agents in a rare Markov state). Shuffle quality degrades for small N but is still at least as good as i.i.d.

3. **Tests**
   - Integration test: `IndShockConsumerType` with `income_shuffle=True`, fixed seed → `solve()` + `simulate()`, verify PermShk empirical frequencies match `pmv` within tolerance.
   - Integration test: `MarkovConsumerType` with `income_shuffle=True`, fixed seed → same, per Markov state slice.
   - Verify backward compat: existing tests pass unchanged with default `income_shuffle=False`.
   - Check interaction with `from_dstn` option (recently added, commit `4266659b`).

**Acceptance:** Full `pytest` green; backward compatible default `False`.

---

## Future directions (not scoped for current work)

These are **not** concrete tasks yet — included for planning context only.

1. **Harmenberg dual-measure alignment:** If shuffle or exact-match permutation is used with dual (P and Q) simulation, any permutation must be shared across both tracks so physical shocks stay aligned. This is a design constraint for any future HAFiscal integration, not a HARK-core change.

2. **Documentation:** Section in `docs/guides/simulation.md` on when shuffle helps TM-MC comparisons vs when it does not; interaction with Harmenberg permanent-income-neutral measure.

3. **HAFiscal downstream:** Parameter wiring for `verify_four_methods` / Gatekeeper — likely lives in HAFiscal-Latest repo, not in HARK.

4. **Scenario-comparison simulator utility.** The current shuffle machinery provides the *primitive* for common random numbers across counterfactual experiments: `MarkovProcess._draw_shuffled` gives bit-identical output on unchanged source-state rows, `DiscreteDistribution.draw(shuffle=True)` gives exact-marginal matching, and `make_shock_history(shuffle=True)` wires these into HARK's standard simulation loop.

   What is *not* yet provided is a convenient top-level API for the typical counterfactual workflow:

   > *"Run this agent through parameter set `A`, then through parameter set `B`, with guaranteed CRN; report the difference in aggregate `f`."*

   Downstream projects (HAFiscal being the motivating example) currently roll their own pattern — pre-compute a shared shock history via `make_idiosyncratic_shock_histories`, manually override Markov states at the experiment onset, apply policy-specific income manipulation as a post-processing step, and replay via `read_shocks=True`. This pattern exists because HARK's standard `simulate()` loop doesn't natively support:

   - **Injecting a "jump" in Markov state at a specific simulation time** (for recession-onset shocks or policy start dates).
   - **Parameter swaps on a live agent** without re-initialising its random state (needed to keep base and policy runs seed-aligned after the swap).
   - **Policy-specific post-processing hooks** on shock histories (needed to apply unemployment-benefit schedules, tax-cut multipliers, stimulus checks on top of a shared shock history).

   A HARK-core "scenario runner" utility — perhaps a context manager or a thin wrapper class — could formalise this pattern: given an agent, a baseline parameter dict, a policy parameter dict, and a seed, run both simulations with guaranteed CRN and return aligned shock and state histories. This would let HAFiscal delete its custom `_hit_with_recession_shock*` machinery and use the standard HARK path with two flag flips, and would give any other counterfactual researcher a principled way to do the same.

   **Scope estimate:** medium-to-large PR. Involves introducing a new simulation-time API (probably in `HARK/simulation/`), factoring out the "pre-draw-then-replay" pattern so it doesn't require agent-level hooks, and documenting the CRN guarantees users can rely on. The primitives are all in place (items 1–5 of the main reshuffle plan are done) — this is an ergonomics / composability layer on top.

---

## What not to port from #1244 (unless re-justified)

- **`draw_events(..., exact_match)`** duplicate algorithm — use **`draw(..., shuffle=True)`** only.
- **`perf_reshuffle`** and global **LCM `AgentCount` raises** — defer; #1691 reduces need for "nice" N.
- **Death-shock reshuffling** — only if a concrete use case appears; skip in v1.

---

## Integration branch strategy

All work (Tasks A and B, plus the already-completed infrastructure) lives on the **`harmenberg-dual-measure`** branch, which includes `AggIndMrkvConsumerType` from the `ConsAggIndMarkovModel` line plus dual-measure / Harmenberg tooling not yet on `main`.

When ready for upstream:
- Open PRs from branches rebased onto current `main` per maintainer preference.
- The integration branch remains the daily driver until those PRs land.
- Merge `origin/main` into the integration branch regularly to stay current.

---

## Retrieving the old example notebook (PR #1244)

```bash
git fetch origin pull/1244/head:pr-1244
git show pr-1244:examples/ConsIndShockModel/IndShockConsumerType_Reshuffling_Example.ipynb > /tmp/IndShockConsumerType_Reshuffling_Example.ipynb
```

Use as **narrative inspiration** only; rewrite against `shuffle=True` and current class APIs.

---

## Document history

| Date | Note |
|------|------|
| 2026-04-04 | Initial plan added to HARK repo for multi-machine work. |
| 2026-04-04 | Duplicated at canonical path; `docs/guides/` holds pointer only. |
| 2026-04-04 | Revised for Harmenberg/ConsAggIndMarkovModel-first integration branch workflow. |
| 2026-04-04 | **v3:** Revised after code review of `harmenberg-dual-measure` branch. Documented completed work (AggIndMrkvConsumerType, KS refactor, newborn fix). Reduced scope to two remaining tasks. Added concrete code snippets, draw-return-type asymmetry notes, `from_dstn` checklist item. Demoted PR 3 to future directions (already implemented as `get_micro_markov_states` override in KS). |

# Shuffling / Variance-Reduction Machinery — PR Change Summary

This document enumerates every change made to the HARK base code to support
the optional "shuffling" variance-reduction machinery, suitable for inclusion
in a standalone PR. All new behavior is **opt-in** via flags that default to
`False` (or via a separately-imported mixin), so existing simulations are
bit-for-bit unchanged.

## Motivation

Monte Carlo simulation in HARK suffers from cross-sectional sampling noise:
with finite `AgentCount`, empirical shock means, death counts, and Markov
transition counts deviate from their analytical targets, injecting noise into
aggregate statistics. The shuffling machinery offers several opt-in techniques
that pin these moments exactly while preserving (or improving) the realism of
the cross-section. They are individually toggleable so users can compose them.

## Files changed

| File | Status | Purpose |
|------|--------|---------|
| `HARK/distributions/discrete.py` | modified | Add `replicates=` and `shuffle=` to `DiscreteDistribution.draw()` |
| `HARK/distributions/base.py` | modified | Add `shuffle=` and `sort_key=` to `MarkovProcess.draw()` |
| `HARK/ConsumptionSaving/ConsIndShockModel.py` | modified | Wire `income_shuffle` and `death_shuffle` flags into `get_shocks()` / `sim_death()` |
| `HARK/ConsumptionSaving/ConsMarkovModel.py` | modified | Wire `income_shuffle`, `markov_shuffle`, `balanced_transitions`, `death_shuffle` |
| `HARK/simulation/normalization.py` | **new** | `ShockNormalizationMixin`, `PermanentIncomeNormalizationMixin` |
| `examples/Distributions/Shuffle_vs_IID_Draws.ipynb` | **new** | Educational notebook with quantitative comparisons |
| `tests/...` (income_shuffle, Markov shuffle, pLvl normalization) | **new** | Unit tests for each technique |

No existing public APIs change signature in a breaking way: every new
parameter is keyword-only with a default that reproduces the old behavior.

## 1. `DiscreteDistribution.draw()` — `shuffle` and `replicates`

`HARK/distributions/discrete.py`

The `draw()` method gains two new keyword arguments:

- **`shuffle: bool = False`** — When `True`, draws are produced by the
  floor-plus-leftover algorithm: each atom `j` is drawn exactly
  `floor(N * p_j)` times, then the remaining slots are filled by sampling
  without replacement from the residual mass. Result: empirical atom
  frequencies match `p_j` to within rounding, eliminating multinomial noise.
- **`replicates: Optional[int] = None`** — Convenience: instead of choosing
  `N`, the user requests `k` copies of the *minimal full-coverage sample*.
  The minimal sample size `J_min` is the LCM of the denominators of the
  probability vector (expressed as exact rationals via `fractions.Fraction`).
  For a hierarchically-independent joint shock (e.g., 0.05 unemployment ×
  3×3 equiprobable employment shocks), `J_min` is the product of the
  inverse conditional probabilities. `replicates` implies `shuffle=True`.

A new private helper `_resolve_replicates()` performs the rational
decomposition, raises `ValueError` if `J_min` exceeds `max_J_min` (default
10,000) — i.e., if some conditional probability is finer than `1/max_J_min`,
making shuffled draws impractical — and emits a `warnings.warn` for joint
distributions where not every `1/p_j` is an integer.

`N` becomes `Optional[int]`; the method raises if neither `N` nor
`replicates` is supplied. Existing call sites that pass `N` positionally are
unaffected.

## 2. `MarkovProcess.draw()` — `shuffle` and `sort_key`

`HARK/distributions/base.py`

The original behavior is preserved in a private `_draw_iid()` method.
`draw()` gains:

- **`shuffle: bool = False`** — When `True`, dispatches to `_draw_shuffled()`,
  which for each source state `j` computes deterministic target counts via
  the same floor-plus-leftover algorithm and assigns agents to target
  states. Falls back to iid for source states where
  `N_j * min(probs) < 1` (population too small for meaningful determinism).
- **`sort_key: Optional[np.ndarray] = None`** — When provided alongside
  `shuffle=True`, agents within each source state are ordered by `sort_key`
  and target states are assigned by *systematic sampling* (uniform spacing
  with a random offset, smallest target groups first). This makes the
  subgroup transitioning to each target state representative of the source
  population with respect to the sort variable. The intended use is to pass
  `pLvl` so that, e.g., the agents transitioning into unemployment are
  systematically spread across the income distribution.

  An important cautionary comment in the docstring and call sites notes
  that `aNrm`/wealth must **not** be used as a sort key — doing so creates
  a feedback loop in which low-wealth agents are repeatedly selected for
  adverse transitions and become trapped in poverty.

## 3. `IndShockConsumerType` — `income_shuffle`, `death_shuffle`

`HARK/ConsumptionSaving/ConsIndShockModel.py`

### New simulation defaults

```python
"income_shuffle": False,  # Use shuffled draws for income shocks
"death_shuffle":  False,  # Use deterministic death counts
```

### `PerfForesightConsumerType.sim_death()`

Refactored: when `death_shuffle=True`, dispatches to the new
`_sim_death_shuffled()` helper, which groups agents by their death
probability and selects exactly `round(N_group * p)` agents to die via
`RNG.choice(..., replace=False)`. Eliminates binomial noise in mortality.
Default path is unchanged.

### `IndShockConsumerType.get_shocks()`

Two changes:

1. **`income_shuffle` branch.** When the flag is set, income shocks are
   drawn with `IncShkDstnNow.draw(N, shuffle=True)`, so empirical permanent
   and transitory shock frequencies match the discretization exactly.
2. **Optional base-uniform draw caching.** When the instance attribute
   `_cache_base_shock_draws` is `True` (set by dual-measure mixins),
   shocks are drawn via explicit `_rng.uniform(size=N)` followed by
   `np.searchsorted(np.cumsum(pmv), ...)` and the base uniform draws are
   stashed in `self._base_shock_draws` (a dict keyed by `t_cycle` value plus
   `"newborn"`); `self._newborn_mask` is also exposed. When the flag is
   `False` (the default), the original `draw_events(N)` code path is used
   verbatim, so seeded reproductions of pre-shuffle simulations are
   bit-for-bit preserved.

The newborn redraw block receives the same treatment for both branches.

## 4. `MarkovConsumerType` — `markov_shuffle`, `balanced_transitions`, `income_shuffle`, `death_shuffle`

`HARK/ConsumptionSaving/ConsMarkovModel.py`

### New defaults in `init_indshk_markov`

```python
"markov_shuffle":       False,
"balanced_transitions": False,
```

(`income_shuffle` and `death_shuffle` are inherited from the IndShock
defaults.)

### `sim_death()`

Same refactor as the base class: routes to `_sim_death_shuffled()` when
`death_shuffle=True`.

### `sim_birth()` / Markov state update

The per-period Markov transition draw now passes `shuffle=self.markov_shuffle`
and, when `balanced_transitions=True`, `sort_key=self.state_now["pLvl"][right_age]`.
The sort-key block carries an explicit comment warning against using `aNrm`.

### `get_shocks()`

Two changes paralleling the IndShock version:

1. **`income_shuffle` branch** for each `(t_cycle, mrkv_state)` cell.
2. **Optional base-uniform caching** in `self._base_shock_draws` keyed by
   `(t, j)` tuples, for dual-measure replay. Gated on
   `self._cache_base_shock_draws`; default path is the original
   `draw_events(N)` code, bit-for-bit preserved.

### Newborn permanent-shock fix (incidental but bundled)

Previously `PermShkNow[newborn] = 1.0` zeroed out *both* the idiosyncratic
permanent shock ψ *and* the deterministic growth factor `PermGroFac`,
causing newborns to silently lose one period of permanent income growth
every lifetime. The new code sets, per Markov state,
`PermShkNow[these_nb] = self.PermGroFac[0][j]`, restoring the deterministic
growth while still suppressing the idiosyncratic shock (whose dispersion is
already represented in `pLvlInitDstn`). A long comment in the source
explains the reasoning. This fix is logically independent of shuffling but
was discovered while wiring it in; reviewers may wish to split it out.

## 5. New module: `HARK/simulation/normalization.py`

A small standalone module containing two mixins. They are not used by any
existing class — users opt in by subclassing.

### `ShockNormalizationMixin`

Wraps `get_shocks()`. After the parent draws `PermShk` and `TranShk`,
rescales each within-group (per Markov state when present, otherwise
whole-population) so that the empirical mean is exactly 1.0. Activated by
the instance attribute `normalize_shocks = True`. Per-group, skips groups
with `< 2` agents or empirical means within `1e-16` of zero.

### `PermanentIncomeNormalizationMixin`

Wraps `sim_one_period()`. After each period, applies an affine transform
in log-space to `state_now["pLvl"]` within each age cohort `k`, so that
`E[log p | age=k]` and `Var[log p | age=k]` match analytical values:

```
mu_k    = pLogInitMean + k * log(PermGroFac) - eff_periods * sigma_psi^2 / 2
sigma_k = sqrt(pLogInitStd^2 + eff_periods * sigma_psi^2)
```

with `eff_periods = max(k - 1, 0)` because the first growth step after birth
is deterministic. For lognormal pLvl, pinning the first two log-moments
pins all power moments `E[p^k]` simultaneously. The affine log-transform
preserves rank ordering, so wealth–income correlations are unchanged.

The default `_analytical_log_pLvl_moments()` is appropriate for
state-independent permanent shocks (e.g., `IndShockConsumerType`).
Subclasses with state-dependent growth (e.g., Markov models) should
override it. Cohorts with fewer than 5 agents are skipped.

## 6. Tests

New tests cover (file paths follow the existing `tests/` layout):

- `DiscreteDistribution.draw(shuffle=True)` and `replicates=` exact-count
  guarantees and `J_min` overflow.
- `MarkovProcess.draw(shuffle=True)` exact transition counts; iid fallback
  for tiny populations; `sort_key` representativeness.
- `IndShockConsumerType` and `MarkovConsumerType` simulate cleanly with
  `income_shuffle=True`, `death_shuffle=True`, `markov_shuffle=True`,
  `balanced_transitions=True`, both individually and in combination.
- `PermanentIncomeNormalizationMixin` matches analytical per-cohort
  log-moments to machine precision.

## 7. Notebook

`examples/Distributions/Shuffle_vs_IID_Draws.ipynb` is a self-contained
educational notebook that:

1. Motivates the noise problem with an iid baseline.
2. Demonstrates `DiscreteDistribution.draw(shuffle=True)` and `replicates=`.
3. Layers each technique (`income_shuffle`, `death_shuffle`,
   `markov_shuffle`, `balanced_transitions`, the two normalization mixins)
   onto an `IndShock` and a `Markov` employment model.
4. Reports variance reduction, marginal-utility metrics, and wall-clock
   timing for each combination.

## Backward compatibility

- Every new parameter defaults to the old behavior.
- No existing class gains a new base class. The two mixins are opt-in via
  user subclassing.
- The base-uniform-draws caching path in `IndShockConsumerType.get_shocks`
  and `MarkovConsumerType.get_shocks` is gated on the opt-in
  `_cache_base_shock_draws` attribute. With the default `False`, the
  original `IncShkDstnNow.draw(N)` / `draw_events(N)` calls are preserved
  verbatim, so seeded reproductions of pre-shuffle simulations are
  bit-for-bit identical.
- The `MarkovConsumerType` newborn `PermShk` fix is a bug fix that *will*
  change simulated paths for any user of `MarkovConsumerType`. It can be
  split into its own commit/PR if reviewers want to keep this PR purely
  additive.

## Suggested commit ordering for the standalone PR

1. `feat(distributions): add shuffle= and replicates= to DiscreteDistribution.draw`
2. `feat(distributions): add shuffle= and sort_key= to MarkovProcess.draw`
3. `feat(IndShock): add income_shuffle and death_shuffle flags`
4. `feat(Markov): add markov_shuffle, balanced_transitions, death_shuffle`
5. `fix(Markov): preserve PermGroFac for newborns` *(optional split)*
6. `feat(simulation): add ShockNormalizationMixin and PermanentIncomeNormalizationMixin`
7. `test: cover all shuffle/normalization paths`
8. `docs: add Shuffle_vs_IID_Draws educational notebook`

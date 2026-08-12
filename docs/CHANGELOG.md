# Release Notes

## Introduction

This document contains the release notes of HARK. HARK aims to produce an open source repository of highly modular, easily interoperable code for solving, simulating, and estimating dynamic economic models with heterogeneous agents.

For more information on HARK, see [our Github organization](https://github.com/econ-ark).

## Changes

### 0.17.3 (dev)

Release Date: TBD

#### Release Notes

(None yet)

#### Major Changes

- Raises the minimum supported Python to 3.12 and adds 3.14, following [SPEC 0](https://scientific-python.org/specs/spec-0000/), which drops a Python version three years after release. Python 3.11 left that window in October 2025, and numpy and scipy both already require 3.12 or newer, so installing HARK on 3.10 or 3.11 resolved a dependency stack from two years ago rather than the one HARK is developed against. The CI matrix now sweeps 3.12, 3.13 and 3.14 on Linux and covers 3.13 on macOS and Windows.
- **Breaking:** `ConsAggIndMarkovModel` is rewritten: `AggIndMrkvConsumerType(MarkovConsumerType)` replaces the former `AggIndMarkovConsumerType(AgentType)` (hierarchical macro+micro Markov states via `shocks["Mrkv"]`, overridable `get_macro_markov_states`/`get_micro_markov_states`, pure-`MarkovConsumerType` fallback when the hierarchical counts are unset). The old name is **removed, not aliased** - the old class's contract (`MrkvCombined`, `AgentType` base) differed enough that a silent alias would mislead; imports fail loudly instead. `KrusellSmithType` reparents onto the new class (with a no-op `sim_death` preserving its no-mortality RNG stream); its default simulation path is unchanged, evidenced by the untouched seeded KS test suite. [#1798](https://github.com/econ-ark/HARK/pull/1798)
- **Breaking (Krusell-Smith only):** the aggregate Markov state key on `KrusellSmithType` and `KrusellSmithEconomy` is renamed `"Mrkv"` → `"MrkvAgg"` (`shock_vars_`, `shocks`, `sow_vars`, `track_vars`, `sow_init`, and `economy.history`). This frees the `"Mrkv"` name for the *idiosyncratic* Markov state in the hierarchical-Markov refactor and removes the collision between agent-level and aggregate-level state keys. `AggShockMarkovConsumerType` and the Cobb-Douglas Markov economies are unchanged. Migration: replace `history["Mrkv"]`/`shocks["Mrkv"]` with `"MrkvAgg"` in KS-based code. [#1797](https://github.com/econ-ark/HARK/pull/1797)
- future item
- future item
- future item

#### Minor Changes

- Speeds up multi-dimensional interpolation by replacing `np.unique(..., axis=0)` in `_iter_unique_pairs` with a packed integer key. The old form built a void-dtype view of the rows and lexsorted it, which dominated solver time: profiling the slowest test in the suite, `argsort` accounted for 71.3s of 143.7s. Most callers pass a single position array, where the row-lexsort machinery was being paid to sort one column. Measured on the same machine: that test 147.62s to 70.11s (2.11x), and the full test suite 413.02s to 242.46s (1.70x). Results are unchanged; the helper is shared by the 2D/3D/4D interpolators, so any model doing multi-dimensional interpolation benefits. [#1817](https://github.com/econ-ark/HARK/pull/1817)
- Fixes `HARK.dual_measure` indexing the Q income process one period ahead of P whenever `cycles != 1`: `_draw_Q_shocks_indshock` chose `IncShkDstn_Q[t]` where `get_shocks` uses `IncShkDstn[t - 1]`, so an infinite-horizon agent with `T_cycle > 1` drew the Q sample from the wrong period's distribution and scaled it by the wrong `PermGroFac`. Invisible until now because every fixture used `T_cycle == 1`, where indices 0 and -1 name the same element.
- Fixes `MarkovProcess.draw(shuffle=True)` returning uninitialized memory for an agent whose source state has no row in the transition matrix. The output buffer is now sentinel-filled and verified, so those agents raise `IndexError` (as the unshuffled path already did) instead of silently inheriting the previous period's `Mrkv` values.
- Fixes a division by zero at `LivPrb == 1` in `compute_mean_pLvl`, and a wrong limit in the corresponding guard in `compute_pLvl_factor`. Both compute the newborn share of a stationary population; it is now one shared helper returning `1 / T_age` at the no-mortality limit rather than `nan` or `0`.
- `MarkovConsumerType.get_shocks` now records newborn base draws under `("newborn", j)`, the key `HARK.dual_measure` already looked up and nothing wrote, so Markov newborns share their uniforms with the Q measure like every other cell. The P stream is unchanged.
- `setup_Q_measure` refuses, rather than silently mispricing, a composition whose `get_Rport` reads P-side state (`KinkedRconsumerType`, `KinkyPrefConsumerType`, and the `ConsRiskyAssetModel` branch, which would give the Q agent the P agent's realized portfolio share).
- `setup_Q_measure` no longer emits one degenerate-distribution warning per period; a stock `init_lifecycle` produced 25, which buried the aggregate warning that matters. The periods are recorded in `Q_degenerate_periods` instead.
- `MarkovConsumerType.get_markov_states` warns and falls back when `balanced_transitions` is set without a `pLvl` to sort on, matching `AggIndMrkvConsumerType` instead of raising a bare `KeyError`; both now share one helper.
- `DualMeasureMixin.sim_one_period` calls `AgentType._sim_period_prologue`/`_sim_period_epilogue` instead of hand-copying them, so a future change to the prologue cannot silently skip dual mode. Verified bit-identical.
- Declares `init_shuffle` in `PerfForesightConsumerType_simulation_defaults` and `init_indshk_markov`, where its readers live; it was declared only on `IndShockConsumerType` and worked through a `getattr` fallback.
- Restores ruff's default file discovery (`extend-include` rather than `include`), so `ruff check <dir>` no longer reports "All checks passed" after inspecting zero Python files.
- `AgentType.get_states` now raises when `transition()` returns more values than there are states. States are assigned by position, so the loop silently dropped the tail. A return *shorter* than the state list stays legal, because it is deliberate: `GenIncProcessConsumerType` declares five states and returns three, writing the rest by name in `get_poststates`. The comment there now also records that reordering `state_vars` silently reassigns every value, which is how three models came to declare an `aNrm` that nothing wrote.
- `MarkovProcess.draw(shuffle=True)` now warns when a source state has too few agents for deterministic counts and falls back to iid. The exact transition counts the shuffled path advertises were silently withdrawn for those agents while the rest of the population kept them; both normalization mixins already warn on their analogous skips. Aggregated into one warning naming the affected source states, rather than one per state per period.
- `AgentType._sim_period_prologue` now blanks each period's ndarray states with `nan` instead of `np.empty`. A state that no later step writes previously held whatever was in the freed buffer, which in practice is usually the previous period's values, so the gap read as plausible data rather than as a defect; it now surfaces as `nan` rather than as plausible numbers. The same change in `AgentSimulator`'s newborn path (`HARK.simulator`) replaces an `np.empty` under a comment promising to "clear" the variable; both now route through one type-dispatching blank helper, which was already `nan`-filling elsewhere in that file. [#1809](https://github.com/econ-ark/HARK/issues/1809)
- Fixes three models reporting an uninitialized `aNrm`: `GenIncProcessConsumerType`, `MedShockConsumerType`, and `MedExtMargConsumerType`. The variable is declared in `state_vars` but these models work in levels, so nothing in `transition` writes it and only `sim_birth` ever touched it; every continuing agent carried whatever the per-period blanking left behind. On a 100-agent, 10-period `GenIncProcess` run, 998 of 1000 tracked cells disagreed with `aLvl / pLvl`, taking values like `3.96e-319` -- subnormals, i.e. freed memory. `MedShockConsumerType` was worse: all 1600 cells of a 200-agent, 8-period run. The definition now lives in one place, `GenIncProcessConsumerType.set_aNrm_from_levels`, which the two `get_poststates` overrides call, so a further override cannot silently reopen it. Found because the `nan` blanking above made the second and third instances visible.
- Removes a duplicate `"mLvl"` from `MedShockConsumerType.state_vars`, which appended a name the inherited list already contained.
- `setup_Q_measure` now also refuses an agent with `normalize_pLvl=True`, and `PermanentIncomeNormalizationMixin` refuses an agent already in dual mode. Same class of defect as `normalize_shocks` below by a different mechanism: the per-cohort `pLvl` adjustment runs in `post_state_hook` and the Q pipeline does not mirror it at all. Measured at 1000 agents over 10 periods, enabling it moves the P history by 1.45e-2 and leaves the Q history *bit-identical*; across 10 seeds at 2000 agents the standard deviation of the final-period mean `pLvl` falls from 9.71e-3 to 5.91e-4 for P while Q stays at 1.03e-2. Both refusal messages now say the other flag is not a workaround, since the previous one-sided guard told users to turn off `normalize_shocks` and the module's own docstring example enables both.
- `setup_Q_measure` now refuses an agent with `normalize_shocks=True`, and `ShockNormalizationMixin` refuses an agent already in dual mode. Composing the two variance-reduction features reversed the result dual mode exists to demonstrate: normalization rescales `shocks["PermShk"]` in place after the base uniforms were recorded and before the Q pipeline inverts them, so P's cross-sectional shock mean was pinned exactly while Q kept all of its sampling noise. Measured over 12 seeds at 2000 agents, the cross-seed standard deviation of the period-mean deviation is 2.03e-3 for both measures with normalization off, and 6.1e-17 for P against 2.03e-3 for Q with it on -- so a user enabling both, which nothing warned against, would have concluded the neutral measure increases variance. Making them genuinely composable requires normalizing the Q draws to the Q measure's own mean, `PermGroFac * E[psi^2] / E[psi]^2`, which is a design decision rather than a fix, so this refuses instead of guessing.
- `ShockNormalizationMixin`'s zero-mean guard is now relative to the shock scale rather than an absolute `1e-16`. The guard protects a division by the empirical mean, so what matters is the mean's size next to the values it came from: the absolute form skipped a group whose shocks were legitimately all near `1e-18`, and accepted a mean of `1e-10` among values of order `1e6`, where the rescale factor is order `1e16`. No shipped calibration reaches either end; the simulation fingerprint is unchanged.
- Removes the unreachable first branch of `AggIndMrkvConsumerType.get_macro_markov_states`. Its docstring advertised `self.EconomyMrkvNow` as the primary lookup, but nothing in HARK assigns that attribute, so the `hasattr` guard always fell through to `shocks["MrkvAgg"]`.
- Renames the `examples` workflow from "Test examples as a cron job" to "Test examples", which is what it does: besides the nightly run it also fires on every push to `main` and every PR against `main`. Those pre-merge runs are the ones that catch a change breaking an example notebook, so the name was corrected rather than the triggers.
- Declares the public API of the recently added modules: `__all__` for `HARK.simulation.normalization` and `HARK.ConsumptionSaving.ConsAggIndMarkovModel`, and an API-reference section for `HARK.simulation.normalization` on the Simulation tools page. [#1811](https://github.com/econ-ark/HARK/pull/1811)
- Excludes scipy 1.18.0, whose `PPoly`-family objects (e.g. `CubicHermiteSpline`) cannot be `deepcopy`-ed (`TypeError: cannot pickle 'module' object`), breaking `ValueFuncCRRA` construction and the existing test suite wherever that scipy version is resolved. [#1788](https://github.com/econ-ark/HARK/pull/1788)
- Adds opt-in `markov_shuffle` and `balanced_transitions` parameters to `MarkovConsumerType.get_markov_states`: quota-exact Markov transitions via `MarkovProcess.draw(shuffle=True)`, optionally with systematic sampling by pLvl. Default False; the default call is unchanged. [#1793](https://github.com/econ-ark/HARK/pull/1793)
- Adds opt-in low-variance draw modes to the distributions layer: `MarkovProcess.draw(shuffle=, sort_key=, draws=)` (quota-exact state transitions with optional stratified rank assignment), `DiscreteDistribution.draw_events(shuffle=)`, and `DiscreteDistribution.draw(replicates=)` (exact full-coverage samples). The leftover-slot allocation that all of these share now lives in one function, `HARK.distributions.base.allocate_remainder_slots`, rather than being written out once per call site: the two earlier copies diverged, with the `MarkovProcess` one still allocating leftovers proportional to the transition row instead of to the fractional remainders, which overweighted the modal target by up to 6.5% and starved the rarest by 15% whenever `N_j * P[j,k]` was not an integer. `replicates` now rejects non-positive values instead of silently returning an empty sample, accepts zero-probability atoms, and warns only when the minimal sample is larger than the rarest atom alone requires. `sort_key=`/`draws=` warn when passed with `shuffle=False` rather than being silently ignored. Every default path is textually identical to the previous code and pinned by RNG-stream golden tests. [#1786](https://github.com/econ-ark/HARK/pull/1786)
- Adds `HARK.simulation.normalization`: opt-in mixins that pin simulated cross-sectional moments to their analytical values, removing sampling noise from aggregates without requiring special population sizes. `ShockNormalizationMixin` rescales each period's drawn shocks so their cross-sectional means are exact; because HARK stores `psi * PermGroFac` in `shocks["PermShk"]` (growth is folded into the permanent "shock"), the target for `PermShk` is `PermGroFac`, not 1.0 - normalizing it to 1.0 would delete permanent income growth rather than sampling noise. `PermanentIncomeNormalizationMixin` pins per-cohort log-`pLvl` moments; a cohort reading `t_age == k` inside `post_state_hook` has already taken `k + 1` permanent shocks (newborns are not exempt: `get_shocks` redraws a random `PermShk` for them and pins only `TranShk`), and the targets accumulate period by period over each cohort's realized income-process history, so life-cycle calibrations with age-varying `PermGroFac` or `PermShkStd` get the right age profile instead of period 0's parameters extrapolated. Both are Markov-capable, with per-state targets and an automatic mean-only mode under state-dependent growth, and both warn rather than degrade silently when they cannot deliver exactness (small groups or cohorts, `read_shocks` replay, staggered entry, a model whose `sim_one_period` never reaches `post_state_hook`). Wiring is through `AgentType.post_state_hook` rather than an overridden `sim_one_period`, so models with their own simulation pipeline are not shadowed. Purely additive: defaults change no behavior and no existing file is modified. [#1784](https://github.com/econ-ark/HARK/pull/1784)
- `make_hierarchical_mrkv_array` auto-detects a general nested `[i][j]` conditional-matrix format (source-and-destination conditioning, Krusell-Smith style) alongside the existing flat destination-conditioned format (unchanged for existing callers); adds its inverse `extract_cond_mrkv_arrays`; `KrusellSmithEconomy.make_MrkvArray` now also stores `MacroMrkvArray`/`CondMrkvArrays` and the KS agent's `market_vars` distributes them (additive plumbing for the hierarchical refactor). `extract_cond_mrkv_arrays` validates its input: it raises `ValueError` if `MrkvIndArray` is not `(M*N) x (M*N)`, or if any block is not the macro probability times a row-stochastic matrix, which is the necessary and sufficient condition for the extracted arrays to be transition matrices. [#1796](https://github.com/econ-ark/HARK/pull/1796)
- Adds an opt-in `init_shuffle` parameter (`PerfForesightConsumerType.sim_birth`, mirrored for `MarkovConsumerType`'s initial Markov states): exact-marginal initial-state draws via `DiscreteDistribution.draw(shuffle=True)`, removing sampling noise in the initial cross-section. Default False; the kwarg is passed only when enabled, so duck-typed continuous init distributions keep working; pinned by a behavior-golden test. [#1791](https://github.com/econ-ark/HARK/pull/1791)
- Makes `CubicHermiteInterp` `deepcopy`-able and picklable independent of scipy internals: the wrapped scipy spline is excluded from serialized state and deterministically rebuilt on restore, so attribute caching like scipy 1.18.0's unpicklable module objects (scipy issue #25489) can no longer break serialization of HARK solutions. [#1802](https://github.com/econ-ark/HARK/pull/1802)
- Removes the `!=1.18.0` scipy exclusion added in [#1788](https://github.com/econ-ark/HARK/pull/1788), which was a stopgap for the deepcopy failure the entry above fixes at its root. Note that pickles written by this version cannot be loaded by earlier HARK, since `_chs` is no longer stored in serialized state. [#1802](https://github.com/econ-ark/HARK/pull/1802)
- Restores `calc_expectation` as a `DeprecationWarning`-bearing alias of `expected_with_loop` (renamed in 0.17.2), preserving import compatibility for downstream code pinned to earlier versions - including frozen reproduction archives that cannot be edited. Slated for removal in a future release. [#1800](https://github.com/econ-ark/HARK/pull/1800)
- Exports `KrusellSmithType`, `KrusellSmithEconomy`, `init_KS_agents`, `init_KS_economy` in `HARK.ConsumptionSaving.ConsAggShockModel.__all__` (they were defined but unlisted); fixes a stale sentence in the KrusellSmithType example notebook. [#1795](https://github.com/econ-ark/HARK/pull/1795)
- Adds `AgentType.post_state_hook()`: a no-op extension point invoked by `sim_one_period` between `get_states()` and `get_controls()`, for mixins that adjust states before controls are computed (e.g. variance-reduction normalization). Default behavior is bit-identical (pinned by a behavior-golden test). [#1787](https://github.com/econ-ark/HARK/pull/1787)
- Adds an opt-in `death_shuffle` parameter (`PerfForesightConsumerType.sim_death`, mirrored in `MarkovConsumerType`): for each distinct death probability, the number of deaths is set by floor-plus-remainder and the agents who die are drawn uniformly at random from that group. Each agent's marginal death probability is still `DiePrb` and the expected number of deaths is unchanged; deaths within a group become negatively correlated (`-1/(N-1)`), which is what removes the binomial noise. The reduction scales with `N_group * DiePrb`, so it is exact only for a single large group with `T_age=None` (death count constant at every period) and partial otherwise: on a 65-age lifecycle calibration the variance falls 34% at `AgentCount=1000` and 96% at `AgentCount=20000`, and with `T_age` set, old-age deaths are added afterward and are not de-noised. Default False; the default RNG path is preserved verbatim and pinned by a behavior-golden test. [#1790](https://github.com/econ-ark/HARK/pull/1790)
- Fixes `DiscreteDistribution.draw(shuffle=True)`, which distributed the leftover slots after `floor(N*pmv)` in proportion to `pmv` itself rather than to the fractional remainders, so `E[count_j]` was not `N*pmv[j]`. The remainder is now allocated by systematic sampling, whose inclusion probability is exactly the remainder. The error was `O(M/N)`: invisible at population-sized draws, and large for the small cohorts `sim_birth` redraws, where the rarest atom came out about half as often as it should. `draw(0, shuffle=True)` also raised rather than returning an empty array, which `sim_birth` hits in any period with no deaths. Because `ConsAggShockModel` draws shuffled income shocks, the simulated Krusell-Smith economy changes slightly and `AFunc[0].slope` in `testAggShockMarkovConsumerType` moves from 1.05654 to 1.06030.
- Adds `HARK.dual_measure`: opt-in Harmenberg neutral-measure (Q) parallel tracking via `DualMeasureMixin`, plus standalone aggregation helpers (`compute_mean_pLvl`, `compute_pLvl_factor`). Purely additive: no existing class or default behavior changes, and the default-off path is bit-identical to the plain agent including RNG stream position. `simulate()` delegates each period's P-measure recording to `AgentType.simulate` instead of reimplementing its loop, so enabling dual mode cannot change what the P pipeline records; an earlier draft dropped the base loop's `getattr` fall-through and left `history["MPCnow"]` all-NaN whenever `dual_measure` was on. `_transition_Q` stores `kNrm` and `bNrm` rather than computing `bNrm` and discarding it, and `_lag_Q_states` blanks with `np.nan` rather than `np.empty`, so a Q state nobody writes reads back as NaN instead of as recycled buffer contents that are finite and in range for the variable they are standing in for. Markov newborns redraw their Q permanent shock from `IncShkDstn_Q[0][j]` and gate `TranShk` on `NewbornTransShk`, matching `MarkovConsumerType.get_shocks` rather than pinning psi to 1. `make_Q_measure_dstn` and `setup_Q_measure` warn when the permanent shock is degenerate, instead of quietly handing back a Q measure equal to P. [#1783](https://github.com/econ-ark/HARK/pull/1783)
- `AgentType.make_shock_history` gains an opt-in `shuffle=` keyword (default False): the pre-drawn shock history can be generated with the low-variance draw modes temporarily enabled. The original body survives verbatim as `_make_shock_history`; the default path delegates to it unchanged and is pinned by a stream-golden test. [#1794](https://github.com/econ-ark/HARK/pull/1794)
- `AggIndMrkvConsumerType.get_micro_markov_states` gains the opt-in `markov_shuffle` branch: quota-exact micro-state transitions per (macro, source-micro) cell via `MarkovProcess.draw(shuffle=True)`, supporting both conditional-matrix formats, with `balanced_transitions` (systematic sampling by pLvl) available. Default remains iid `RNG.choice`, unchanged. A macro transition that carries agents but has zero probability under `MacroMrkvArray` now raises `ValueError` naming the `(macro_prev, macro_next)` cell, instead of leaving those agents with uninitialised micro states; `balanced_transitions=True` without a `pLvl` in `state_now` now warns rather than silently falling back to unbalanced shuffling. [#1801](https://github.com/econ-ark/HARK/pull/1801)
- Extends `income_shuffle` to `MarkovConsumerType.get_shocks` (per-state and newborn draws). Default False; original RNG paths preserved verbatim. [#1792](https://github.com/econ-ark/HARK/pull/1792)
- Adds an opt-in `income_shuffle` parameter to `IndShockConsumerType.get_shocks`: exact floor-plus-leftover shock frequencies per period (via `DiscreteDistribution.draw(shuffle=True)`) instead of iid sampling. Default False; the default RNG path is preserved verbatim and pinned by a stream-golden test. [#1789](https://github.com/econ-ark/HARK/pull/1789)
- Wires `HARK.dual_measure`'s base-draw cache through `IndShockConsumerType.get_shocks` and `MarkovConsumerType.get_shocks`: under the `_cache_base_shock_draws` flag, income-shock uniforms are recorded for Q-CDF inversion using the same draws and the same inversion as the default path; the P-stream is bit-identical with the flag on or off (tested). `setup_Q_measure` now turns the flag on and registers `IncShkDstn_Q` with `self.distributions`, so the shared base draws the module documents actually happen and survive repeated `initialize_sim()` calls; pass `_cache_base_shock_draws = False` afterwards for independent Q draws. Setting `income_shuffle` and the cache together records no uniforms, and now warns instead of silently decoupling P from Q (this covers the `income_shuffle` collision specifically, not every way P and Q can decouple; see the `normalize_shocks` entry below for the other one). `initialize_sim` clears `_base_shock_draws` so a later run cannot consume a previous run's uniforms. [#1799](https://github.com/econ-ark/HARK/pull/1799)
- future item
- future item
- future item


### 0.17.2

Release Date: May 1, 2026

#### Release Notes

This is a moderately sized release with several exciting new features, as well as many small improvements and fixes.
Most of the breaking changes (see below) are very small adjustments to parameter names or formats; two functions also had their name change.
The only significant breaking change is a reworking of the interaction between `AgentType` instances and their associated `Market` with respect to aggregate-level parameters.

The new features are headlined by the addition of two models with consumption habits in the new `ConsHabitModel` module.
Additionally, HARK's automatic HA-SSJ construction method has been extended to life-cycle models, rather than only infinite horizon models.

There are some breaking changes:

- `AgentType` subclasses that had a `get_economy_data` method now use the general `AgentType.get_market_params` method, which exactly replicates their prior operation. See #1719
- As a consequence of the above, random seeds on the distributions of some `AgentType` subclasses will change because the order in which they are created during instantiation has changed.
- Parameter `PortfolioBool` has been deprecated. To allow portfolio choice for `RiskyAssetConsumerType`, just set `RiskyShareFixed=None`. #1740
- The parameter `BeqCRRA` has been deprecated; agents with a warm glow bequest motive must use the same CRRA as their ordinary utility function. #1758
- "Terminal bequest parameters" have been deprecated; agents have the same bequest motive in period T as they do in all other periods. #1758
- `calc_expectation` has been renamed to `expected_with_loop`; use `expected` and pass `vectorized=False` for this functionality. #1763
- The argument `dist` in `expected` has been renamed to `dstn`. #1763
- The function `make_exponential_grid` has been renamed to `make_polynomial_grid` to reduce confusion with `make_grid_exp_mult`. #1762

#### Major Changes

- The new way to set up `AgentType` instances with an associated `Market` is to create them (with the agents in the `Market`'s `agents` attribute), then invoke the `Market`'s new `give_agent_params()` method. #1719
- The above method calls each `agent`'s `get_market_params()` method, which references the `market_vars` class attribute for the names of objects to take from the associated `Market`.
- All interpolator classes now have default derivative methods using finite differences. These are fallback methods, and are already overridden by most subclasses. #1723
- New consumption-saving model with habit formation has been added; extends IndShockConsumerType model. #1739
- Added habit-formation model with portfolio allocation, along with example notebooks. #1748
- Simulator class has new method `simulate_shock_by_grids` to perturb the steady state distribution and then simulate by matrix transition methods. #1754
- Simplify parameters in `ConsBequestModel.py` to eliminate "terminal" bequest parameters and different CRRA for bequests than consumption. #1758
- The `make_basic_SSJ` method can now handle life-cycle models (`cycles=1`) as well as standard infinite horizon models. #1718

#### Minor Changes

- The special constructor `get_it_from` can now interpret the referenced attribute being a single value (any numeric or string) and will simply copy it to the new name. #1719
- A `Market`'s `calc_dynamics` function/method can now use arguments other than those named in `track_vars`; HARK will look for those names as attributes of the `Market`. #1719
- The _derY method for `LowerEnvelope2D` and `LowerEnvelope3D` were previously bugged and returned nonsense, now fixed. #1723
- Updated syntax in a few places that tried to convert singleton array to a float, to ensure compatibility with NumPy 2.4+ #1725
- Add new income shock constructor that incorporates Velasquez-Giraldo's representation of medical expenses as negative transitory income shocks. #1724
- Add parameter dictionary with Fulford and Low's estimates for *all* expenses (not just medical) for use by MedShockConsumerType. #1724
- Refactoring of representative agent model solver and the "labeled" submodule. #1727
- Example notebooks for all models with portfolio choice have been significantly expanded and improved. #1740
- Example notebooks for models in `ConsAggShockModel.py` have been improved and expanded from their prior form. #1738
- The `labels` argument now works as intended with `distribution.expected`. #1742
- Example notebook `Transition_Matrix_Example.ipynb` has been cleaned up and expanded. #1744
- `AggIndMarkovConsumerType` added for models with both aggregate (shared) and idiosyncratic discrete states; `KrusellSmithType` refactored to extend it. #1747
- Light safety fixes to the new `HabitConsumerType`. #1753
- Example notebooks for models in `ConsBequestModel.py` have been improved and expanded from their prior form. #1754
- Example notebooks for KinkedRconsumerType, MarkovConsumerType, LaborIntMargConsumerType, and TractableBufferStockConsumerType have been improved and expanded. #1743
- Handling of income shocks for model "newborns" has been made consistent across models, with transitory shocks optional. #1760
- Tests added to handle a variety of unusual corner cases. #1761
- Computing expectations now always uses `expected`; if the function cannot accept vector arguments, pass `vectorized=False`. #1763
- Matrix transition methods (including HA-SSJ) now support multi-exponential grids, as well as fully custom grids. #1762
- `HARK.interpolation` refactored to reduce repetition and code clutter. #1765
- Small documentation notebook for life-cycle HA-SSJ construction has been added. #1718
- `HARK.simulator` and experimental Monte Carlo submodule refactored to reduce repetition. #1766
- `HARK.distributions` refactored to reduce repetition and improve structures. #1767
- Additional refactoring in `Labeled`, `SSJutils`, `utilities`, and `metric` to reduce code repetition. #1768


### 0.17.1

Release Date: February 2, 2026

#### Release Notes

This is a relatively small release that includes various adjustments and improvements (see Minor Changes), as well as several new features and an algebraic revision to some models (Major Changes).

There are some breaking changes:

- The `exact_match` option for `DiscreteDistribution.draw` has been renamed to `shuffle`, and its behavior has changed slightly. See #1691.
- Both `AgentType` subclasses in ConsPrefShockModel have had their utility function adjusted, moving the preference shock inside the CRRA term. See #1708.
- If `calc_expectation` is used with a `DiscreteDistributionLabeled`, the function must reference indices of the distribution by name, not position number. See #1713.
- Method `NewKeynesianConsumerType.compute_steady_state` has been renamed to `compute_pe_steady_state`. See #1711.

#### Major Changes

- Added `find_target` method to `AgentType`, automating search for target value of state variables. [#1698](https://github.com/econ-ark/HARK/pull/1698)
- Utility function for `PrefShockConsumerType` and `KinkyPrefConsumerType` was algebraically rearranged. There is no functional difference, but the scale of preference shocks that yields a given level of consumption variation will be different. [#1708](https://github.com/econ-ark/HARK/pull/1708/)
- The format of the utility function for `MedShockConsumerType` has been revised; prior distributions of MedShk will need to be adjusted. See #1706.
- The policy function representation for `MedShockConsumerType` has been revised, and old classes have been moved to LegacyOOsolvers.
- The utility function for `MedShockConsumerType` has been algebraically rearranged, moving MedShk inside of the second CRRA term and adding a new parameter MedShift (default near zero). [#1706](https://github.com/econ-ark/HARK/pull/1706)
- Function `plot_func_slices` has been added to `HARK.utilities` for convenient in-line plotting of multivariate functions [#1695](https://github.com/econ-ark/HARK/pull/1695)
- New method `AgentType.export_to_df` added to flexibly export simulated `history` to a `pandas.DataFrame`. [#1712](https://github.com/econ-ark/HARK/pull/1712)

#### Minor Changes

- Revised `exact_match` option for `DiscreteDistribution.draw` to `shuffle` to be more robust to population draw size. [#1691](https://github.com/econ-ark/HARK/pull/1691)
- multi_thread_commands[_fake] no longer requires empty parentheses to be included with each method name (now optional). [#1692](https://github.com/econ-ark/HARK/pull/1692)
- Added __repr__ method for DiscreteDistribution (and subclasses) to display basic information about itself.
- All AgentTypes now have sensible defaults for track_vars if none is provided. [#1693](https://github.com/econ-ark/HARK/pull/1693)
- `AgentType.unpack` and the new simulation structure appropriately handle solutions represented as dictionaries. [#1709](https://github.com/econ-ark/HARK/pull/1709)
- `calc_expectation` now works with `DiscreteDistributionLabeled` instances when `func` references RVs by name, but *not* by position numbers. [#1713](https://github.com/econ-ark/HARK/pull/1713)
- Repository now includes AI prompts to aid users when updating their project code from one version of HARK to another. [#1696](https://github.com/econ-ark/HARK/pull/1696)
- 2D, 3D, and 4D interpolator classes no longer require that their arguments have the same size/shape; now they must only be jointly broadcastable. [#1701](https://github.com/econ-ark/HARK/pull/1701)
- Method name change for `NewKeynesianConsumerType`: `compute_steady_state` is now `compute_pe_steady_state`. [#1711](https://github.com/econ-ark/HARK/pull/1711)
- Life-cycle parameter calibrations from Carroll 1997 (QJE) have been added to `ConsIndShockModel`. [#1715](https://github.com/econ-ark/HARK/pull/1715)


### 0.17.0

Release Date: January 4, 2026

#### Release Notes

This release has many small improvements and fixes to existing HARK capabilities, listed below under Minor Changes. It also includes expanded and improved documentation/learning materials in examples/Gentle-Intro. To copy those example notebooks into a local working directory for easy use, simply execute these two commands in a Python environment and then follow the prompts:

`from HARK import install_examples`
`install_examples()`

Four new consumption-saving models have been added, listed below under Major Changes.

There are some breaking changes:

- TimeVaryingDiscreteDistribution has been removed; use IndexDistribution instead, and see #1592.
- FixedPortfolioShareRiskyAssetConsumerType is removed, but now incorporated as RiskyAssetConsumerType with PortfolioBool=False. Default behavior of latter class is unchanged; see #1607.
- The content of HARK.parallel has been moved to HARK.core, and the former is deprecated. Import from HARK.core and see #1614.
- parse_ssa_life_table now returns one fewer survival probability by default, to match output length of parse_income_spec; pass terminal=True to restore old behavior. Argument min_age has been renamed to age_min for consistency. See #1629.
- The parameter tau in RiskyContribModel has been renamed to WithdrawTax to match HARK notation style; see #1639.
- Simulation method get_Rfree() has been renamed to get_Rport(), but no functional changes; see #1646.
- The parameter DeprFac has been renamed to DeprRte to reflect its actual usage.
- All distributions now default to using a random seed if none is provided. If your code relied on HARK defaulting to a specific seed, it will not reproduce exactly. See #1641.
- The function apply_flat_income_tax has been removed, but it has not been used at all since 2016.
- Content from ConsLabeledModel has been split up into files in the Labeled submodule. See #1684.

#### Major Changes

- Basic health investment model added in new module ConsHealthModel. [#1567](https://github.com/econ-ark/HARK/pull/1567)
- Extensive margin medical care choice model added to ConsMedModel. [#1595](https://github.com/econ-ark/HARK/pull/1595)
- TRP-style wealth-in-utility model *without* portfolio choice added in new module ConsWealthUtilityModel. [#1634](https://github.com/econ-ark/HARK/pull/1634)
- "Capitalist spirit" style wealth-in-utility model added in new module ConsWealthUtilityModel. [#1634](https://github.com/econ-ark/HARK/pull/1634)

#### Minor Changes

- Fixed terminal solution initialization in IndShockConsumerTypeFast for proper numba compatibility, added CRRA=1 validation with clear error message, and expanded test coverage. [#1649](https://github.com/econ-ark/HARK/pull/1649)
- Turns off use_infimum feature in ConsIndShock solver because it did not work properly when vFunc=True [#1589](https://github.com/econ-ark/HARK/pull/1589)
- Consolidates `TimeVaryingDiscreteDistribution` into `IndexDistribution`. For time-varying discrete behavior, use `IndexDistribution(distributions=[...])`. [#1592](https://github.com/econ-ark/HARK/pull/1592)
- Krusell-Smith model guide added to documentation. [#1594](https://github.com/econ-ark/HARK/pull/1594)
- Added additional options and simplified syntax for non-default constructors when instantiating agents. [#1591](https://github.com/econ-ark/HARK/pull/1591)
- Added options for custom indexer and pre-computation of coefficients to LinearInterp. [#1593](https://github.com/econ-ark/HARK/pull/1593)
- Fixed bug that prevented combine_indep_dstn from working with Bernoulli distributions. [#1581](https://github.com/econ-ark/HARK/pull/1581)
- Introductory / instructional notebooks significantly expanded. [#1597](https://github.com/econ-ark/HARK/pull/1597)
- Lognormal discrete approximation math has been simplified. [#1598](https://github.com/econ-ark/HARK/pull/1598)
- Directory structure for consumption-saving examples regularized. [#1596](https://github.com/econ-ark/HARK/pull/1596)
- Fixed share model has been combined with RiskyAssetConsumerType's PortfolioBool=False option. [#1607](https://github.com/econ-ark/HARK/pull/1607)
- Deprecate HARK.parallel, moving the three functions there to HARK.core. [#1614](https://github.com/econ-ark/HARK/pull/1614)
- Test coverage expanded to cover almost all content #1606 #1610 #1617 #1619 #1623 #1624 #1625 #1626 #1628 #1684
- Consumption-saving models now aliased at HARK.models and HARK.ConsumptionSaving; some calibration tools also aliased at HARK.Calibration [#1629](https://github.com/econ-ark/HARK/pull/1629)
- AgentType.solve() can be passed postsolve=False to skip post-processing call to post_solve(). [#1631](https://github.com/econ-ark/HARK/pull/1631)
- The /examples directory can be copied to a directory of user's choice with HARK.install_examples() [#1630](https://github.com/econ-ark/HARK/pull/1630)
- Improved and expanded features for Parameters class in HARK.core [#1627](https://github.com/econ-ark/HARK/pull/1627)
- Fixed the representation of the terminal period solution in ConsPrefShock [#1638](https://github.com/econ-ark/HARK/pull/1638)
- Renamed tau to WithdrawTax in RiskyContribModel [#1639](https://github.com/econ-ark/HARK/pull/1639)
- Valid bounds checking on make_grid_exp_mult [#1640](https://github.com/econ-ark/HARK/pull/1640)
- Ensure utility functions return NaN for negative consumption [#1640](https://github.com/econ-ark/HARK/pull/1640)
- Fixed a bug with resetting the RNG of IndexDistributions, restoring replicability of simulations [#1643](https://github.com/econ-ark/HARK/pull/1643)
- Legacy simulation methods now use get_Rport() instead of get_Rfree() [#1646](https://github.com/econ-ark/HARK/pull/1646)
- Fixed a bug that occured when changing an AgentType's AgentCount attribute after simulating [#1647](https://github.com/econ-ark/HARK/pull/1647)
- Add describe_distance() method to MetricObject, generating text description of how "distance" is calculated for an object [#1648](https://github.com/econ-ark/HARK/pull/1648)
- Default behavior of seeds for distribution classes has been revised. [#1641](https://github.com/econ-ark/HARK/pull/1641)
- Terminal solution representation for the "fast" solvers (using numba) has been cleaned up. [#1649](https://github.com/econ-ark/HARK/pull/1649)
- Refactored ConsLabeledModel to use new HARK.Labeled subpackage with modular architecture (config, factories, transitions, solvers, solution, agents). Added comprehensive input validation, runtime warnings for numerical issues, and expanded test coverage. [#1650](https://github.com/econ-ark/HARK/pull/1650)


### 0.16.1

Release Date: July 24, 2025

This release includes various small changes and improvements, as well as one significant new feature: (almost) all AgentType subclasses can now construct HA-SSJs (for use with the sequence_jacobian toolkit) in standard infinite horizon problems for arbitrary shock variables and arbitrary model outputs. This capability is powered by a new simulation structure that uses YAML-based model files to define dynamics, which in turn can be used to automatically transform HARK's model solution representations (policy functions over continuous spaces) into the grid-based representation needed for efficient computation of the fake news algorithm. Don't worry, that all happens under the hood.

See documentation notebooks in /examples/SequenceSpaceJacobians/ . The capabilities of our SSJ calculator will be expanded in the near future to include lifecycle models.

#### Major Changes

- Adds a new simulator structure based on YAML model files, replicating legacy simulation results [#1545](https://github.com/econ-ark/HARK/pull/1545)
- Can convert solved HARK models into transition matrix-based discretized grid representations [#1545](https://github.com/econ-ark/HARK/pull/1545)
- Can produce sequence-space Jacobians for infinite horizon problems for all HARK AgentType subclasses for which this is appropriate [#1545](https://github.com/econ-ark/HARK/pull/1545)

#### Minor Changes

- Allows lifecycle models to be solved backward starting from non-terminal period (with custom solution) [#1545](https://github.com/econ-ark/HARK/pull/1545)
- Adds new interpolator class IndexedInterp with alternative notation for functions with mixed discrete-continuous domain [#1545](https://github.com/econ-ark/HARK/pull/1545)
- New notebook with tutorial for (old and new) simulation methods [#1545](https://github.com/econ-ark/HARK/pull/1545)
- Constructor make_grid_exp_mult allows linearly spaced grid with timestonest=-1 [#1545](https://github.com/econ-ark/HARK/pull/1545)
- Adds documentation for new simulator structure and basic SSJ calculator [#1545](https://github.com/econ-ark/HARK/pull/1545)
- Fixed a rare bug that could occur with unusual constructor dependencies resulting in incomplete updates. [#1575](https://github.com/econ-ark/HARK/pull/1575/)
- Added a reference to a trivial constructor that was missing from the WealthPortfolio model. [#1583](https://github.com/econ-ark/HARK/pull/1583)
- Documentation files have been moved from /Documentation/ to /docs/ [#1579](https://github.com/econ-ark/HARK/pull/1579)
- All tests have been consolidated into a single directory, rather than being scattered about. [#1578](https://github.com/econ-ark/HARK/pull/1578)
- Add a special README so that the robots know we're on their side when the singularity arrives. [#1577](https://github.com/econ-ark/HARK/pull/1577)


### 0.16.0

Release Date: June 9, 2025

The items listed as "Developmental Features" are an independent system that is not connected to HARK's existing model structure.

The most likely code-breaking change in this release is the reorganization of `HARK.distribution`. If your project code tells you that it can't find the module `HARK.distribution`, just change the import name to `HARK.distributions` (note the plural s).

Additionally, several parameters have been lightly renamed:

aNrmInitMean --> kLogInitMean
aNrmInitStd --> kLogInitStd
pLvlInitMean --> pLogInitMean
pLvlInitStd --> pLogInitStd

Finally, the legacy option for Rfree to be time-invariant has been removed from most models to allow the code to be simplified. If you used time-invariant Rfree, you will need to change your parameterization from Rfree = Rfree_value to T_cycle*[Rfree_value].

#### Major Changes

- Reorganizes the `HARK.distribution` file into `HARK.distributions` submodule with various files for readability and extensibility [#1496](https://github.com/econ-ark/HARK/pull/1496)
- Regularizes `AgentType` initialization methods and moves all constructed model objects to `constructors` [#1529](https://github.com/econ-ark/HARK/pull/1529) and [#1530](https://github.com/econ-ark/HARK/pull/1529)

#### Developmental Features

- Adds a discretize method to DBlocks and RBlocks [#1460](https://github.com/econ-ark/HARK/pull/1460)
- Allows structural equations in model files to be provided in string form [#1427](https://github.com/econ-ark/HARK/pull/1427)
- Introduces `HARK.parser' module for parsing configuration files into models [#1427](https://github.com/econ-ark/HARK/pull/1427)
- Allows construction of shocks with arguments based on mathematical expressions [#1464](https://github.com/econ-ark/HARK/pull/1464)
- YAML configuration file for the normalized consumption and portolio choice [#1465](https://github.com/econ-ark/HARK/pull/1465)

#### Minor Changes

- Fixes bug in `AgentPopulation` that caused discretization of distributions to not work. [#1275](https://github.com/econ-ark/HARK/pull/1275)
- Adds support for distributions, booleans, and callables as parameters in the `Parameters` class. [#1387](https://github.com/econ-ark/HARK/pull/1387)
- Removes a specific way of accounting for ``employment'' in the idiosyncratic-shocks income process. [#1473](https://github.com/econ-ark/HARK/pull/1473)
- Adds income process constructor for the discrete Markov state consumption-saving model. [#1484](https://github.com/econ-ark/HARK/pull/1484)
- Changes the behavior of make_lognormal_RiskyDstn so that the standard deviation represents the standard deviation of log(returns)
- Adds detailed parameter and LaTeX documentation to most models.
- Add PermGroFac constructor that explicitly combines idiosyncratic and aggregate sources of growth. [1489](https://github.com/econ-ark/HARK/pull/1489)
- Suppress warning from calc_stable_points when it would be raised by inapplicable AgentType subclasses. [1493](https://github.com/econ-ark/HARK/pull/1493)
- Fixes notation errors in IndShockConsumerType.make_euler_error_func from prior changes. [1495](https://github.com/econ-ark/HARK/pull/1495)
- Fixes typos in IdentityFunction interpolator class. [1492](https://github.com/econ-ark/HARK/pull/1492)
- Expands functionality of Cobb-Douglas aggregator for CRRA utility. [1363](https://github.com/econ-ark/HARK/pull/1363)
- Improved tracking of the bounds of support for distributions, and (some) solvers now respect those bounds when computing the "worst outcome". [1524](https://github.com/econ-ark/HARK/pull/1524)
- Adds a new function for using Tauchen's method to approximate an AR1 process. [#1521](https://github.com/econ-ark/HARK/pull/1521)
- Adds additional functionality to the CubicHermiteInterp class, imported from scipy.interpolate. [#1020](https://github.com/econ-ark/HARK/pull/1020/)
- Allows users to pass a generic solution object to agent solvers to be used as the initial condition of backward induction. [#1543](https://github.com/econ-ark/HARK/pull/1543)
- Adds support for Python 3.13 and related package updates. [#1549](https://github.com/econ-ark/HARK/pull/1549)
- Move sim_birth methods to constructed distributions, lightly rename parameters. [#1553](https://github.com/econ-ark/HARK/pull/1553)
- Cleans up warnings for the distance metric and prevents simulation history from being returned as output. [#1563](https://github.com/econ-ark/HARK/pull/1563)
- Assorted small code cleanup tasks proposed by codex. [#1555] [#1556] [#1557] [#1558] [#1559] [#1560] [#1561] [#1562]
- Updated and expanded documentation of sequence space Jacobian examples. [#1564] [#1568] [#1501] [#1490] [#1481] [#1475]
- Improved documentation of cycles and timing of microeconomic models. [#1571](https://github.com/econ-ark/HARK/pull/1571)

### 0.15.1

Release Date: June 15, 2024

This minor release was produced prior to CEF 2024 to enable public usage of HARK with the SSJ toolkit.

#### Major Changes

none

#### Minor Changes

- Adds example of integration of HARK with SSJ toolkit. [#1447](https://github.com/econ-ark/HARK/pull/1447)
- Maintains compatibility between EconForge interpolation and numba [#1457](https://github.com/econ-ark/HARK/pull/1457)
- Renanmes 'SSJ_example' to 'HANKFiscal_example' so that it is more informative. [#1509](https://github.com/econ-ark/HARK/pull/1509)

### 0.15.0

Release Date: June 4, 2024

Note: Due to major changes on this release, you may need to adjust how AgentTypes are instantiated in your projects using HARK. If you are manually constructing "complicated" objects like MrkvArray, they should be assigned to your instances *after* initialization, not passed as part of the parameter dictionary. See also the new constructor methodology for how to pass parameters for such constructed inputs.

This release drops support for Python 3.8 and 3.9, consistent with SPEC 0, and adds support for Python 3.11 and 3.12. We expect that all HARK features still work with the older versions, but they are no longer part of our testing regimen.

#### Major Changes

- Drop official support for Python 3.8 and 3.9, add support for 3.11 and 3.12. [#1415](https://github.com/econ-ark/HARK/pull/1415)
- Replace object-oriented solvers with single function versions. [#1394](https://github.com/econ-ark/HARK/pull/1394)
- Object-oriented solver code has been moved to /HARK/ConsumptionSaving/LegacyOOsolvers.py, for legacy support of downstream projects.
- AgentTypeMonteCarloSimulator now requires model shock, parameter, and dynamics information to be organized into 'blocks'. The DBlock object is introduced. [#1411](https://github.com/econ-ark/HARK/pull/1411)
- RBlock object allows for recursive composition of DBlocks in models, as demonstrated by the AgentTypeMonteCarloSimulator [#1417](https://github.com/econ-ark/HARK/pull/1417/)
- Transtion, reward, state-rule value function, decision value function, and arrival value function added to DBlock [#1417](https://github.com/econ-ark/HARK/pull/1417/)
- All methods that construct inputs for solvers are now functions that are specified in the dictionary attribute `constructors`. [#1410](https://github.com/econ-ark/HARK/pull/1410)
- Such constructed inputs can use alternate parameterizations / formats by changing the `constructor` function and providing its arguments in `parameters`.
- Move `HARK.datasets` to `HARK.Calibration` for better organization of data and calibration tools. [#1430](https://github.com/econ-ark/HARK/pull/1430)

#### Minor Changes

- Add option to pass pre-built grid to `LinearFast`. [1388](https://github.com/econ-ark/HARK/pull/1388)
- Moves calculation of stable points out of ConsIndShock solver, into method called by post_solve [#1349](https://github.com/econ-ark/HARK/pull/1349)
- Adds cubic spline interpolation and value function construction to "warm glow bequest" models.
- Fixes cubic spline interpolation for ConsMedShockModel.
- Moves computation of "stable points" from inside of ConsIndShock solver to a post-solution method. [#1349](https://github.com/econ-ark/HARK/pull/1349)
- Corrects calculation of "human wealth" under risky returns, providing correct limiting linear consumption function. [#1403](https://github.com/econ-ark/HARK/pull/1403)
- Removed 'parameters' from new block definitions; these are now 'calibrations' provided separately.
- Create functions for well-known and repeated calculations in single-function solvers. [1395](https://github.com/econ-ark/HARK/pull/1395)
- Re-work WealthPortfolioSolver to use approximate EGM method [#1404](https://github.com/econ-ark/HARK/pull/1404)
- Default parameter dictionaries for AgentType subclasses have been "flattened": all parameters appear in one place for each model, rather than inheriting from parent models' dictionaries. The only exception is submodels *within* a file when only 1 or 2 parameters are added or changed. [#1425](https://github.com/econ-ark/HARK/pull/1425)
- Fix minor bug in `HARK.distributions.Bernoulli` to allow conversion into `DiscreteDistributionLabeled`. [#1432](https://github.com/econ-ark/HARK/pull/1432)

### 0.14.1

Release date: February 28, 2024

#### Major Changes

none

#### Minor Changes

- Fixes a bug in make_figs arising from the metadata argument being incompatible with jpg. [#1386](https://github.com/econ-ark/HARK/pull/1386)
- Reverts behavior of the repr method of the Model class, so that long strings aren't generated. Full description is available with describe(). [#1390](https://github.com/econ-ark/HARK/pull/1390)

### 0.14.0

Release Date: February 12, 2024

#### Major Changes

- Adds `HARK.core.AgentPopulation` class to represent a population of agents with ex-ante heterogeneous parametrizations as distributions. [#1237](https://github.com/econ-ark/HARK/pull/1237)
- Adds `HARK.core.Parameters` class to represent a collection of time varying and time invariant parameters in a model. [#1240](https://github.com/econ-ark/HARK/pull/1240)
- Adds `HARK.simulation.monte_carlo` module for generic Monte Carlo simulation functions using Python model configurations. [1296](https://github.com/econ-ark/HARK/pull/1296)

#### Minor Changes

- Adds option `sim_common_Rrisky` to control whether risky-asset models draw common or idiosyncratic returns in simulation. [#1250](https://github.com/econ-ark/HARK/pull/1250),[#1253](https://github.com/econ-ark/HARK/pull/1253)
- Addresses [#1255](https://github.com/econ-ark/HARK/issues/1255). Makes age-varying stochastic returns possible and draws from their discretized version. [#1262](https://github.com/econ-ark/HARK/pull/1262)
- Fixes bug in the metric that compares dictionaries with the same keys. [#1260](https://github.com/econ-ark/HARK/pull/1260)

### 0.13.0

Release Date: February 16, 2023

#### Major Changes

- Updates the DCEGM tools to address the flaws identified in [issue #1062](https://github.com/econ-ark/HARK/issues/1062). PR: [1100](https://github.com/econ-ark/HARK/pull/1100).
- Updates `IndexDstn`, introducing the option to use an existing RNG instead of creating a new one, and creating and storing all the conditional distributions at initialization. [1104](https://github.com/econ-ark/HARK/pull/1104)
- `make_shock_history` and `read_shocks == True` now store and use the random draws that determine newborn's initial states [#1101](https://github.com/econ-ark/HARK/pull/1101).
- `FrameModel` and `FrameSet` classes introduced for more modular construction of framed models. `FrameAgentType` dedicated to simulation. [#1117](https://github.com/econ-ark/HARK/pull/1117)
- General control transitions based on decision rules in `FrameAgentType`. [#1117](https://github.com/econ-ark/HARK/pull/1117)
- Adds `distr_of_function` tool to calculate the distribution of a function of a discrete random variable. [#1144](https://github.com/econ-ark/HARK/pull/1144)
- Changes the `DiscreteDistribution` class to allow for arbitrary array-valued random variables. [#1146](https://github.com/econ-ark/HARK/pull/1146)
- Adds `IndShockRiskyAssetConsumerType` as agent which can invest savings all in safe asset, all in risky asset, a fixed share in risky asset, or optimize its portfolio. [#1107](https://github.com/econ-ark/HARK/issues/1107)
- Updates all HARK models to allow for age-varying interest rates. [#1150](https://github.com/econ-ark/HARK/pull/1150)
- Adds `DiscreteDistribution.expected` method which expects vectorized functions and is faster than `HARK.distributions.calc_expectation`. [#1156](https://github.com/econ-ark/HARK/pull/1156)
- Adds `DiscreteDistributionXRA` class which extends `DiscreteDistribution` to allow for underlying data to be stored in a `xarray.DataArray` object. [#1156](https://github.com/econ-ark/HARK/pull/1156)
- Adds keyword argument `labels` to `expected()` when using `DiscreteDistributionXRA` to allow for expressive functions that use labeled xarrays. [#1156](https://github.com/econ-ark/HARK/pull/1156)
- Adds a wrapper for [`interpolation.py`](https://github.com/EconForge/interpolation.py) for fast multilinear interpolation. [#1151](https://github.com/econ-ark/HARK/pull/1151)
- Adds support for the calculation of dreivatives in the `interpolation.py` wrappers. [#1157](https://github.com/econ-ark/HARK/pull/1157)
- Adds class `DecayInterp` to `econforgeinterp.py`. It implements interpolators that "decay" to some limiting function when extrapolating. [#1165](https://github.com/econ-ark/HARK/pull/1165)
- Add methods to non stochastically simulate an economy by computing transition matrices. Functions to compute transition matrices and ergodic distribution have been added [#1155](https://github.com/econ-ark/HARK/pull/1155).
- Fixes a bug that causes `t_age` and `t_cycle` to get out of sync when reading pre-computed mortality. [#1181](https://github.com/econ-ark/HARK/pull/1181)
- Adds Methods to calculate Heterogenous Agent Jacobian matrices. [#1185](https://github.com/econ-ark/HARK/pull/1185)
- Enhances `combine_indep_dstns` to work with labeled distributions (`DiscreteDistributionLabeled`). [#1191](https://github.com/econ-ark/HARK/pull/1191)
- Updates the `numpy` random generator from `RandomState` to `Generator`. [#1193](https://github.com/econ-ark/HARK/pull/1193)
- Turns the income and income+return distributions into `DiscreteDistributionLabeled` objects. [#1189](https://github.com/econ-ark/HARK/pull/1189)
- Creates `UtilityFuncCRRA` which is an object oriented utility function with a coefficient of constant relative risk aversion and includes derivatives and inverses. Also creates `UtilityFuncCobbDouglas`, `UtilityFuncCobbDouglasCRRA`, and `UtilityFuncConstElastSubs`. [#1168](https://github.com/econ-ark/HARK/pull/1168)
- Reorganizes `HARK.distributions`. All distributions now inherit all features from `scipy.stats`. New `ContinuousFrozenDistribution` and `DiscreteFrozenDistribution` to use `scipy.stats` distributions not yet implemented in HARK. New `Distribution.discretize(N, method = "***")` replaces `Distribution.approx(N)`. New `DiscreteDistribution.limit` attribute describes continuous origin and discretization method. [#1197](https://github.com/econ-ark/HARK/pull/1197).
- Creates new class of _labeled_ models under `ConsLabeledModel` that use xarray for more expressive modeling of underlying mathematical and economics variables. [#1177](https://github.com/econ-ark/HARK/pull/1177)

#### Minor Changes

- Updates the lognormal-income-process constructor from `ConsIndShockModel.py` to use `IndexDistribution`. [#1024](https://github.com/econ-ark/HARK/pull/1024), [#1115](https://github.com/econ-ark/HARK/pull/1115)
- Allows for age-varying unemployment probabilities and replacement incomes with the lognormal income process constructor. [#1112](https://github.com/econ-ark/HARK/pull/1112)
- Option to have newborn IndShockConsumerType agents with a transitory income shock in the first period. Default is false, meaning they only have a permanent income shock in period 1 and permanent AND transitory in the following ones. [#1126](https://github.com/econ-ark/HARK/pull/1126)
- Adds `benchmark` utility to profile the performance of `HARK` solvers. [#1131](https://github.com/econ-ark/HARK/pull/1131)
- Fixes scaling bug in Normal equiprobable approximation method. [1139](https://github.com/econ-ark/HARK/pull/1139)
- Removes the extra-dimension that was returned by `calc_expectations` in some instances. [#1149](https://github.com/econ-ark/HARK/pull/1149)
- Adds `HARK.distributions.expected` alias for `DiscreteDistribution.expected`. [#1156](https://github.com/econ-ark/HARK/pull/1156)
- Renames attributes in `DiscreteDistribution`: `X` to `atoms` and `pmf` to `pmv`. [#1164](https://github.com/econ-ark/HARK/pull/1164), [#1051](https://github.com/econ-ark/HARK/pull/1151), [#1159](https://github.com/econ-ark/HARK/pull/1159).
- Remove or replace automated tests that depend on brittle simulation results. [#1148](https://github.com/econ-ark/HARK/pull/1148)
- Updates asset grid constructor from `ConsIndShockModel.py` to allow for linearly-spaced grids when `aXtraNestFac == -1`. [#1172](https://github.com/econ-ark/HARK/pull/1172)
- Renames `DiscreteDistributionXRA` to `DiscreteDistributionLabeled` and updates methods [#1170](https://github.com/econ-ark/HARK/pull/1170)
- Renames `HARK.numba` to `HARK.numba_tools` [#1183](https://github.com/econ-ark/HARK/pull/1183)
- Adds the RNG seed as a property of `DiscreteDistributionLabeled` [#1184](https://github.com/econ-ark/HARK/pull/1184)
- Updates the `approx` method of `HARK.distributions.Uniform` to include the endpoints of the distribution with infinitesimally small (zero) probability mass. [#1180](https://github.com/econ-ark/HARK/pull/1180)
- Refactors tests to incorporate custom precision `HARK_PRECISION = 4`. [#1193](https://github.com/econ-ark/HARK/pull/1193)
- Cast `DiscreteDistribution.pmv` attribute as a `np.ndarray`. [#1199](https://github.com/econ-ark/HARK/pull/1199)
- Update structure of dynamic interest rate. [#1221](https://github.com/econ-ark/HARK/pull/1221)

### 0.12.0

Release Date: December 14, 2021

#### Major Changes

- FrameAgentType for modular definitions of agents [#865](https://github.com/econ-ark/HARK/pull/865) [#1064](https://github.com/econ-ark/HARK/pull/1064)
- Frame relationships with backward and forward references, with plotting example [#1071](https://github.com/econ-ark/HARK/pull/1071)
- PortfolioConsumerFrameType, a port of PortfolioConsumerType to use Frames [#865](https://github.com/econ-ark/HARK/pull/865)
- Input parameters for cyclical models now indexed by t [#1039](https://github.com/econ-ark/HARK/pull/1039)
- A IndexDistribution class for representing time-indexed probability distributions [#1018](https://github.com/econ-ark/HARK/pull/1018/).
- Adds new consumption-savings-portfolio model `RiskyContrib`, which represents an agent who can save in risky and risk-free assets but faces
  frictions to moving funds between them. To circumvent these frictions, he has access to an income-deduction scheme to accumulate risky assets.
  PR: [#832](https://github.com/econ-ark/HARK/pull/832). See [this forthcoming REMARK](https://github.com/Mv77/RiskyContrib) for the model's details.
- 'cycles' agent property moved from constructor argument to parameter [#1031](https://github.com/econ-ark/HARK/pull/1031)
- Uses iterated expectations to speed-up the solution of `RiskyContrib` when income and returns are independent [#1058](https://github.com/econ-ark/HARK/pull/1058).
- `ConsPortfolioSolver` class for solving portfolio choice model replaces `solveConsPortfolio` method [#1047](https://github.com/econ-ark/HARK/pull/1047)
- `ConsPortfolioDiscreteSolver` class for solving portfolio choice model when allowed share is on a discrete grid [#1047](https://github.com/econ-ark/HARK/pull/1047)
- `ConsPortfolioJointDistSolver` class for solving portfolio chioce model when the income and risky return shocks are not independent [#1047](https://github.com/econ-ark/HARK/pull/1047)

#### Minor Changes

- Using Lognormal.from_mean_std in the forward simulation of the RiskyAsset model [#1019](https://github.com/econ-ark/HARK/pull/1019)
- Fix bug in DCEGM's primary kink finder due to numpy no longer accepting NaN in integer arrays [#990](https://github.com/econ-ark/HARK/pull/990).
- Add a general class for consumers who can save using a risky asset [#1012](https://github.com/econ-ark/HARK/pull/1012/).
- Add Boolean attribute 'PerfMITShk' to consumption models. When true, allows perfect foresight MIT shocks to be simulated. [#1013](https://github.com/econ-ark/HARK/pull/1013).
- Track and update start-of-period (pre-income) risky and risk-free assets as states in the `RiskyContrib` model [1046](https://github.com/econ-ark/HARK/pull/1046).
- distribute_params now uses assign_params to create consistent output [#1044](https://github.com/econ-ark/HARK/pull/1044)
- The function that computes end-of-period derivatives of the value function was moved to the inside of `ConsRiskyContrib`'s solver [#1057](https://github.com/econ-ark/HARK/pull/1057)
- Use `np.fill(np.nan)` to clear or initialize the arrays that store simulations. [#1068](https://github.com/econ-ark/HARK/pull/1068)
- Add Boolean attribute 'neutral_measure' to consumption models. When true, simulations are more precise by allowing permanent shocks to be drawn from a neutral measure (see Harmenberg 2021). [#1069](https://github.com/econ-ark/HARK/pull/1069)
- Fix mathematical limits of model example in `example_ConsPortfolioModel.ipynb` [#1047](https://github.com/econ-ark/HARK/pull/1047)
- Update `ConsGenIncProcessModel.py` to use `calc_expectation` method [#1072](https://github.com/econ-ark/HARK/pull/1072)
- Fix bug in `calc_normal_style_pars_from_lognormal_pars` due to math error. [#1076](https://github.com/econ-ark/HARK/pull/1076)
- Fix bug in `distribute_params` so that `AgentCount` parameter is updated. [#1089](https://github.com/econ-ark/HARK/pull/1089)
- Fix bug in 'vFuncBool' option for 'MarkovConsumerType' so that the value function may now be calculated. [#1095](https://github.com/econ-ark/HARK/pull/1095)

### 0.11.0

Release Date: March 4, 2021

#### Major Changes

- Converts non-mathematical code to PEP8 compliant form [#953](https://github.com/econ-ark/HARK/pull/953)
- Adds a constructor for LogNormal distributions from mean and standard deviation [#891](https://github.com/econ-ark/HARK/pull/891/)
- Uses new LogNormal constructor in ConsPortfolioModel [#891](https://github.com/econ-ark/HARK/pull/891/)
- calcExpectations method for taking the expectation of a distribution over a function [#884](<https://github.com/econ-ark/HARK/pull/884/>] (#897)[https://github.com/econ-ark/HARK/pull/897/)
- Implements the multivariate normal as a supported distribution, with a discretization method. See [#948](https://github.com/econ-ark/HARK/pull/948).
- Centralizes the definition of value, marginal value, and marginal marginal value functions that use inverse-space
  interpolation for problems with CRRA utility. See [#888](https://github.com/econ-ark/HARK/pull/888).
- MarkovProcess class used in ConsMarkovModel, ConsRepAgentModel, ConsAggShockModel [#902](https://github.com/econ-ark/HARK/pull/902) [#929](https://github.com/econ-ark/HARK/pull/929)
- replace HARKobject base class with MetricObject and Model classes [#903](https://github.com/econ-ark/HARK/pull/903/)
- Add **repr** and **eq** methods to Model class [#903](https://github.com/econ-ark/HARK/pull/903/)
- Adds SSA life tables and methods to extract survival probabilities from them [#986](https://github.com/econ-ark/HARK/pull/906).
- Adds the U.S. CPI research series and tools to extract inflation adjustments from it [#930](https://github.com/econ-ark/HARK/pull/930).
- Adds a module for extracting initial distributions of permanent income (`pLvl`) and normalized assets (`aNrm`) from the SCF [#932](https://github.com/econ-ark/HARK/pull/932).
- Fix the return fields of `dcegm/calcCrossPoints`[#909](https://github.com/econ-ark/HARK/pull/909).
- Corrects location of constructor documentation to class string for Sphinx rendering [#908](https://github.com/econ-ark/HARK/pull/908)
- Adds a module with tools for parsing and using various income calibrations from the literature. It includes the option of using life-cycle profiles of income shock variances from [Sabelhaus and Song (2010)](https://www.sciencedirect.com/science/article/abs/pii/S0304393210000358). See [#921](https://github.com/econ-ark/HARK/pull/921), [#941](https://github.com/econ-ark/HARK/pull/941), [#980](https://github.com/econ-ark/HARK/pull/980).
- remove "Now" from model variable names [#936](https://github.com/econ-ark/HARK/pull/936)
- remove Model.**call**; use Model init in Market and AgentType init to standardize on parameters dictionary [#947](https://github.com/econ-ark/HARK/pull/947)
- Moves state MrkvNow to shocks['Mrkv'] in AggShockMarkov and KrusellSmith models [#935](https://github.com/econ-ark/HARK/pull/935)
- Replaces `ConsIndShock`'s `init_lifecycle` with an actual life-cycle calibration [#951](https://github.com/econ-ark/HARK/pull/951).

#### Minor Changes

- Move AgentType constructor parameters docs to class docstring so it is rendered by Sphinx.
- Remove uses of deprecated time.clock [#887](https://github.com/econ-ark/HARK/pull/887)
- Change internal representation of parameters to Distributions to ndarray type
- Rename IncomeDstn to IncShkDstn
- AgentType simulate() method now returns history. [#916](https://github.com/econ-ark/HARK/pull/916)
- Rename DiscreteDistribution.drawDiscrete() to draw()
- Update documentation and warnings around IncShkDstn [#955](https://github.com/econ-ark/HARK/pull/955)
- Adds csv files to `MANIFEST.in`. [957](https://github.com/econ-ark/HARK/pull/957)

### 0.10.8

Release Date: Nov. 05 2020

#### Major Changes

- Namespace variables for the Market class [#765](https://github.com/econ-ark/HARK/pull/765)
- We now have a Numba based implementation of PerfForesightConsumerType model available as PerfForesightConsumerTypeFast [#774](https://github.com/econ-ark/HARK/pull/774)
- Namespace for exogenous shocks [#803](https://github.com/econ-ark/HARK/pull/803)
- Namespace for controls [#855](https://github.com/econ-ark/HARK/pull/855)
- State and poststate attributes replaced with state_now and state_prev namespaces [#836](https://github.com/econ-ark/HARK/pull/836)

#### Minor Changes

- Use shock_history namespace for pre-evaluated shock history [#812](https://github.com/econ-ark/HARK/pull/812)
- Fixes seed of PrefShkDstn on initialization and add tests for simulation output
- Reformat code style using black

### 0.10.7

Release Date: 08-08-2020

#### Major Changes

- Add a custom KrusellSmith Model [#762](https://github.com/econ-ark/HARK/pull/762)
- Simulations now uses a dictionary `history` to store state history instead of `_hist` attributes [#674](https://github.com/econ-ark/HARK/pull/674)
- Removed time flipping and time flow state, "forward/backward time" through data access [#570](https://github.com/econ-ark/HARK/pull/570)
- Simulation draw methods are now individual distributions like `Uniform`, `Lognormal`, `Weibull` [#624](https://github.com/econ-ark/HARK/pull/624)

#### Minor Changes

- unpackcFunc is deprecated, use unpack(parameter) to unpack a parameter after solving the model [#784](https://github.com/econ-ark/HARK/pull/784)
- Remove deprecated Solution Class, use HARKObject across the codebase [#772](https://github.com/econ-ark/HARK/pull/772)
- Add option to find crossing points in the envelope step of DCEGM algorithm [#758](https://github.com/econ-ark/HARK/pull/758)
- Fix reset bug in the behaviour of AgentType.resetRNG(), implemented individual resetRNG methods for AgentTypes [#757](https://github.com/econ-ark/HARK/pull/757)
- Seeds are set at initialisation of a distribution object rather than draw method [#691](https://github.com/econ-ark/HARK/pull/691) [#750](https://github.com/econ-ark/HARK/pull/750), [#729](https://github.com/econ-ark/HARK/pull/729)
- Deal with portfolio share of 'bad' assets [#749](https://github.com/econ-ark/HARK/pull/749)
- Fix bug in make_figs utilities function [#755](https://github.com/econ-ark/HARK/pull/755)
- Fix typo bug in Perfect Foresight Model solver [#743](https://github.com/econ-ark/HARK/pull/743)
- Add initial support for logging in ConsIndShockModel [#714](https://github.com/econ-ark/HARK/pull/714)
- Speed up simulation in AggShockMarkovConsumerType [#702](https://github.com/econ-ark/HARK/pull/702)
- Fix logic bug in DiscreteDistribution draw method [#715](https://github.com/econ-ark/HARK/pull/715)
- Implemented distributeParams to distributes heterogeneous values of one parameter to a set of agents [#692](https://github.com/econ-ark/HARK/pull/692)
- NelderMead is now part of estimation [#693](https://github.com/econ-ark/HARK/pull/693)
- Fix typo bug in parallel [#682](https://github.com/econ-ark/HARK/pull/682)
- Fix DiscreteDstn to make it work with multivariate distributions [#646](https://github.com/econ-ark/HARK/pull/646)
- BayerLuetticke removed from HARK, is now a REMARK [#603](https://github.com/econ-ark/HARK/pull/603)
- cstwMPC removed from HARK, is now a REMARK [#666](https://github.com/econ-ark/HARK/pull/666)
- SolvingMicroDSOPs removed from HARK, is now a REMARK [#651](https://github.com/econ-ark/HARK/pull/651)
- constructLogNormalIncomeProcess is now a method of IndShockConsumerType [#661](https://github.com/econ-ark/HARK/pull/661)
- Discretize continuous distributions [#657](https://github.com/econ-ark/HARK/pull/657)
- Data used in cstwMPC is now in HARK.datasets [#622](https://github.com/econ-ark/HARK/pull/622)
- Refactor checkConditions by adding a checkCondition method instead of writing custom checks for each condition [#568](https://github.com/econ-ark/HARK/pull/568)
- Examples update [#768](https://github.com/econ-ark/HARK/pull/768), [#759](https://github.com/econ-ark/HARK/pull/759), [#756](https://github.com/econ-ark/HARK/pull/756), [#727](https://github.com/econ-ark/HARK/pull/727), [#698](https://github.com/econ-ark/HARK/pull/698), [#697](https://github.com/econ-ark/HARK/pull/697), [#561](https://github.com/econ-ark/HARK/pull/561), [#654](https://github.com/econ-ark/HARK/pull/654), [#633](https://github.com/econ-ark/HARK/pull/633), [#775](https://github.com/econ-ark/HARK/pull/775)

### 0.10.6

Release Date: 17-04-2020

#### Major Changes

- Add Bellman equations for cyclical model example [#600](https://github.com/econ-ark/HARK/pull/600)

- read_shocks now reads mortality as well [#613](https://github.com/econ-ark/HARK/pull/613)

- Discrete probability distributions are now classes [#610](https://github.com/econ-ark/HARK/pull/610)

#### Minor Changes

### 0.10.5

Release Date: 24-03-2020

#### Major Changes

- Default parameters dictionaries for ConsumptionSaving models have been moved from ConsumerParameters to nearby the classes that use them. [#527](https://github.com/econ-ark/HARK/pull/527)

- Improvements and cleanup of ConsPortfolioModel, and adding the ability to specify an age-varying list of RiskyAvg and RiskyStd. [#577](https://github.com/econ-ark/HARK/pull/527)

- Rewrite and simplification of ConsPortfolioModel solver. [#594](https://github.com/econ-ark/HARK/pull/594)

#### Minor Changes

### 0.10.4

Release Date: 05-03-2020

#### Major Changes

- Last release to support Python 2.7, future releases of econ-ark will support Python 3.6+ [#478](https://github.com/econ-ark/HARK/pull/478)
- Move non-reusable model code to examples directory, BayerLuetticke, FashionVictim now in examples instead of in HARK code [#442](https://github.com/econ-ark/HARK/pull/442)
- Load default parameters for ConsumptionSaving models [#466](https://github.com/econ-ark/HARK/pull/466)
- Improved implementaion of parallelNelderMead [#300](https://github.com/econ-ark/HARK/pull/300)

#### Minor Changes

- Notebook utility functions for determining platform, GUI, latex (installation) are available in HARK.utilities [#512](https://github.com/econ-ark/HARK/pull/512)
- Few DemARKs moved to examples [#472](https://github.com/econ-ark/HARK/pull/472)
- MaxKinks available in ConsumerParameters again [#486](https://github.com/econ-ark/HARK/pull/486)

### 0.10.3

Release Date: 12-12-2019

#### Major Changes

- Added constrained perfect foresight model solution. ([#299](https://github.com/econ-ark/HARK/pull/299)

#### Minor Changes

- Fixed slicing error in minimizeNelderMead. ([#460](https://github.com/econ-ark/HARK/pull/460))
- Fixed matplotlib GUI error. ([#444](https://github.com/econ-ark/HARK/pull/444))
- Pinned sphinx dependency. ([#436](https://github.com/econ-ark/HARK/pull/436))
- Fixed bug in ConsPortfolioModel in which the same risky rate of return would be drawn over and over. ([#433](https://github.com/econ-ark/HARK/pull/433))
- Fixed sphinx dependency errors. ([#411](https://github.com/econ-ark/HARK/pull/411))
- Refactored simultation.py. ([#408](https://github.com/econ-ark/HARK/pull/408))
- AgentType.simulate() now throws informative errors if
  attributes required for simulation do not exist, or initializeSim() has
  never been called. ([#320](https://github.com/econ-ark/HARK/pull/320))

### 0.10.2

Release Date: 10-03-2019

#### Minor Changes

- Add some bugfixes and unit tests to HARK.core. ([#401](https://github.com/econ-ark/HARK/pull/401))
- Fix error in discrete portfolio choice's AdjustPrb. ([#391](https://github.com/econ-ark/HARK/pull/391))

### 0.10.1.dev5

Release Date: 09-25-2019

#### Minor Changes

- Added portfolio choice between risky and safe assets (ConsPortfolioModel). ([#241](https://github.com/econ-ark/HARK/pull/241))

### 0.10.1.dev4

Release Date: 09-19-2019

#### Minor Changes

- Fixes cubic interpolation in KinkedRSolver. ([#386](https://github.com/econ-ark/HARK/pull/386))
- Documentes the procedure for constructing value function inverses and fixes bug in which survival rate was not included in absolute patience factor. ([#383](https://github.com/econ-ark/HARK/pull/383))
- Fixes problems that sometimes prevented multiprocessing from working. ([#377](https://github.com/econ-ark/HARK/pull/377))

### 0.10.1.dev3

Release Date: 07-23-2019

#### Minor Changes

- Missed pre-solve fix (see [#363](https://github.com/econ-ark/HARK/pull/363) for more context). ([#367](https://github.com/econ-ark/HARK/pull/367))

### 0.10.1.dev2

Release Date: 07-22-2019

#### Minor Changes

- Revert pre-solve commit due to bug. ([#363](https://github.com/econ-ark/HARK/pull/363))

### 0.10.1.dev1

Release Date: 07-20-2019

#### Breaking Changes

- See #302 under minor changes.

#### Major Changes

- Adds BayerLuetticke notebooks and functionality. ([#328](https://github.com/econ-ark/HARK/pull/328))

#### Minor Changes

- Fixes one-asset HANK models for endowment economy (had MP wired in as the shock). ([#355](https://github.com/econ-ark/HARK/pull/355))
- Removes jupytext \*.py files. ([#354](https://github.com/econ-ark/HARK/pull/354))
- Reorganizes documentation and configures it to work with Read the Docs. ([#353](https://github.com/econ-ark/HARK/pull/353))
- Adds notebook illustrating dimensionality reduction in Bayer and Luetticke. ([#345](https://github.com/econ-ark/HARK/pull/345))
- Adds notebook illustrating how the Bayer & Luetticke invoke the discrete cosine transformation(DCT) and fixed copula to reduce dimensions of the problem.([#344](https://github.com/econ-ark/HARK/pull/344))
- Makes BayerLuetticke HANK tools importable as a module. ([#342](https://github.com/econ-ark/HARK/pull/342))
- Restores functionality of SGU_solver. ([#341](https://github.com/econ-ark/HARK/pull/341))
- Fixes datafile packaging issue. ([#332](https://github.com/econ-ark/HARK/pull/332))
- Deletes .py file from Bayer-Luetticke folder. ([#329](https://github.com/econ-ark/HARK/pull/329))
- Add an empty method for preSolve called checkRestrictions that can be overwritten in classes inheriting from AgentType to check for illegal parameter values. ([#324](https://github.com/econ-ark/HARK/pull/324))
- Adds a call to updateIncomeProcess() in preSolve() to avoid solutions being based on wrong income process specifications if some parameters change between two solve() calls. ([#323](https://github.com/econ-ark/HARK/pull/323))
- Makes checkConditions() less verbose when the checks are not actually performed by converting a print statement to an inline comment. ([#321](https://github.com/econ-ark/HARK/pull/321))
- Raises more readable exception when simultate() is called without solving first. ([#315](https://github.com/econ-ark/HARK/pull/315))
- Removes testing folder (part of ongoing test restructuring). ([#304](https://github.com/econ-ark/HARK/pull/304))
- Fixes unintended behavior in default simDeath(). Previously, all agents would die off in the first period, but they were meant to always survive. ([#302](https://github.com/econ-ark/HARK/pull/302)) **Warning**: Potentially breaking change.

### 0.10.1

Release Date: 05-30-2019

No changes from 0.10.0.dev3.

### 0.10.0.dev3

Release Date: 05-18-2019

#### Major Changes

- Fixes multithreading problems by using Parallels(backend='multiprocessing'). ([287](https://github.com/econ-ark/HARK/pull/287))
- Fixes bug caused by misapplication of check_conditions. ([284](https://github.com/econ-ark/HARK/pull/284))
- Adds functions to calculate quadrature nodes and weights for numerically evaluating expectations in the presence of (log-)normally distributed random variables. ([258](https://github.com/econ-ark/HARK/pull/258))

#### Minor Changes

- Adds method decorator which validates that arguments passed in are not empty. ([282](https://github.com/econ-ark/HARK/pull/282)
- Lints a variety of files. These PRs include some additional/related minor changes, like replacing an exec function, removing some lambdas, adding some files to .gitignore, etc. ([274](https://github.com/econ-ark/HARK/pull/274), [276](https://github.com/econ-ark/HARK/pull/276), [277](https://github.com/econ-ark/HARK/pull/277), [278](https://github.com/econ-ark/HARK/pull/278), [281](https://github.com/econ-ark/HARK/pull/281))
- Adds vim swp files to gitignore. ([269](https://github.com/econ-ark/HARK/pull/269))
- Adds version dunder in init. ([265](https://github.com/econ-ark/HARK/pull/265))
- Adds flake8 to requirements.txt and config. ([261](https://github.com/econ-ark/HARK/pull/261))
- Adds some unit tests for IndShockConsumerType. ([256](https://github.com/econ-ark/HARK/pull/256))

### 0.10.0.dev2

Release Date: 04-18-2019

#### Major Changes

None

#### Minor Changes

- Fix verbosity check in ConsIndShockModel. ([250](https://github.com/econ-ark/HARK/pull/250))

#### Other Changes

None

### 0.10.0.dev1

Release Date: 04-12-2019

#### Major Changes

- Adds [tools](https://github.com/econ-ark/HARK/blob/master/HARK/dcegm.py) to solve problems that arise from the interaction of discrete and continuous variables, using the [DCEGM](https://github.com/econ-ark/DemARK/blob/master/notebooks/DCEGM-Upper-Envelope.ipynb) method of [Iskhakov et al.](https://onlinelibrary.wiley.com/doi/abs/10.3982/QE643), who apply the their discrete-continuous solution algorithm to the problem of optimal endogenous retirement; their results are replicated using our new tool [here](https://github.com/econ-ark/EndogenousRetirement/blob/master/Endogenous-Retirement.ipynb). ([226](https://github.com/econ-ark/HARK/pull/226))
- Parameters of ConsAggShockModel.CobbDouglasEconomy.updateAFunc and ConsAggShockModel.CobbDouglasMarkovEconomy.updateAFunc that govern damping and the number of discarded 'burn-in' periods were previously hardcoded, now proper instance-level parameters. ([244](https://github.com/econ-ark/HARK/pull/244))
- Improve accuracy and performance of functions for evaluating the integrated value function and conditional choice probabilities for models with extreme value type I taste shocks. ([242](https://github.com/econ-ark/HARK/pull/242))
- Add calcLogSum, calcChoiceProbs, calcLogSumChoiceProbs to HARK.interpolation. ([209](https://github.com/econ-ark/HARK/pull/209), [217](https://github.com/econ-ark/HARK/pull/217))
- Create tool to produce an example "template" of a REMARK based on SolvingMicroDSOPs. ([176](https://github.com/econ-ark/HARK/pull/176))

#### Minor Changes

- Moved old utilities tests. ([245](https://github.com/econ-ark/HARK/pull/245))
- Deleted old files related to "cstwMPCold". ([239](https://github.com/econ-ark/HARK/pull/239))
- Set numpy floating point error level to ignore. ([238](https://github.com/econ-ark/HARK/pull/238))
- Fixed miscellaneous imports. ([212](https://github.com/econ-ark/HARK/pull/212), [224](https://github.com/econ-ark/HARK/pull/224), [225](https://github.com/econ-ark/HARK/pull/225))
- Improve the tests of buffer stock model impatience conditions in IndShockConsumerType. ([219](https://github.com/econ-ark/HARK/pull/219))
- Add basic support for Travis continuous integration testing. ([208](https://github.com/econ-ark/HARK/pull/208))
- Add SciPy to requirements.txt. ([207](https://github.com/econ-ark/HARK/pull/207))
- Fix indexing bug in bilinear interpolation. ([194](https://github.com/econ-ark/HARK/pull/194))
- Update the build process to handle Python 2 and 3 compatibility. ([172](https://github.com/econ-ark/HARK/pull/172))
- Add MPCnow attribute to ConsGenIncProcessModel. ([170](https://github.com/econ-ark/HARK/pull/170))
- All standalone demo files have been removed. The content that was in these files can now be found in similarly named Jupyter notebooks in the DEMARK repository. Some of these notebooks are also linked from econ-ark.org. ([229](https://github.com/econ-ark/HARK/pull/229), [243](https://github.com/econ-ark/HARK/pull/243))

#### Other Notes

- Not all changes from 0.9.1 may be listed in these release notes. If you are having trouble addressing a breaking change, please reach out to us.

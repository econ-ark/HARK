# Shuffle PR — Proposed Improvements (deferred)

Captured 2026-04-08. Revisit before finalizing the standalone shuffling PR(s).
Each item is tagged with the file it touches and whether that file is already
modified on `harmenberg-dual-measure` relative to `main` (M = modified,
N = new on branch, NEW = newly proposed file).

## `HARK/distributions/discrete.py` (M)

1. Drop or downgrade the "Joint distribution detected" `warnings.warn` in
   `_resolve_replicates` — fires on every realistic income distribution.
2. Improve the `J_min > max_J_min` ValueError message to suggest the fix
   ("increase `max_J_min=` or coarsen the discretization").
3. Add a worked numerical example to the `replicates=` docstring
   (e.g. 0.05 unemp × 7 × 7 ⇒ J_min = 980).
4. Move `from fractions import Fraction` to module top with `gcd`, `reduce`.
5. Validate `replicates >= 1` with a clear error.

## `HARK/distributions/base.py` (M)

6. Hoist the "do NOT use aNrm/wealth as sort_key" warning from the
   ConsMarkovModel call site into the `sort_key` docstring in base.py
   as a `Warning:` admonition.
7. Rephrase the iid-fallback condition in plain language.
8. Add a one-line comment in the leftover loop explaining why `Q_adj` is
   recomputed each iteration.
9. Add a module-level note that `shuffle=True` is deterministic in count
   but stochastic in agent-to-target assignment.

## `HARK/ConsumptionSaving/ConsIndShockModel.py` (M)

10. Comment/harden `np.unique(DiePrb)` float-equality assumption in
    `_sim_death_shuffled`.
11. Document that `_sim_death_shuffled` uses `self.RNG.choice` rather than
    constructing a fresh `Uniform` (reproducibility implications).
12. Factor the triple-branch (shuffle / cache / default) in `get_shocks`
    into a private helper `_draw_income_shocks(...)` to deduplicate main
    loop and newborn block.
13. Add a CHANGELOG entry listing every new flag.

## `HARK/ConsumptionSaving/ConsMarkovModel.py` (M)

14. Split the newborn `PermGroFac` bug fix into its own commit + dedicated
    CHANGELOG line flagged as "changes simulated paths".
15. Use `getattr(self, "markov_shuffle", False)` (with default) for
    consistency with the other shuffle flags.

## `HARK/simulation/normalization.py` (N)

16. **Potential bug:** `ShockNormalizationMixin` divides `PermShk` by its
    empirical mean, but by the time it runs, `PermShk` has already been
    multiplied by `PermGroFac` — so the true target mean is `PermGroFac`,
    not 1.0. Verify and fix (normalize before the `PermGroFac`
    multiplication, or divide by `mean / PermGroFac`).
17. `_analytical_log_pLvl_moments` uses `self.PermGroFac[0]` — only correct
    for stationary models. Either assert stationarity or compute via the
    cumulative product up to age `k`.
18. Replace the hard-coded `5`-agent cohort threshold with a class attribute
    `min_cohort_size = 5`.
19. Document the required MRO in the mixin docstring and add an MRO check
    in an example / test.
20. Add `__all__`.

## Tests (N on branch)

21. Audit `tests/ConsumptionSaving/test_IndShockConsumerType.py` (+337)
    for assertions that pin RNG-derived numerical values — prefer
    statistical tolerances.
22. Add a bit-for-bit regression test vs `main` for the default
    (non-shuffle, non-cache) code path.

## `examples/Distributions/` (N on branch)

23. Remove the five working-notes `.md` files (`-ImplementationReport`,
    `-MarkovAdvice`, `-MarkovAdvice_reply`, `-MarkovAdvice_response`,
    `-NextSteps`) from the PR — keep only the notebook.
24. Notebook hygiene: strip per-machine timings and absolute paths; keep
    qualitative comparisons.

## `docs/shuffle-pr-changes.md` (NEW)

25. Add a one-table summary of all new flags (name, default, scope, effect,
    RNG-sequence impact) up front.
26. Cite Harmenberg (2021) and explain how shuffling relates to /
    differs from `neutral_measure` (the branch name implies a connection).
27. Link to PR #1244 (original reshuffle PR being revived).

## Cross-cutting

28. `docs/CHANGELOG.md` (M) — add one entry per logical PR, not per commit.
29. Document the `_cache_base_shock_draws` subclass hook in the
    `IndShockConsumerType` class docstring
    (`HARK/ConsumptionSaving/ConsIndShockModel.py`, M) under a
    "Subclass hooks" section.
30. Naming nit: consider `stratified_transitions` or
    `representative_transitions` instead of `balanced_transitions`
    (`HARK/ConsumptionSaving/ConsMarkovModel.py`, M). Rename now if at all,
    since post-release renames are painful.

## Must-fix-before-PR (flagged)

- **#16** — verify/fix `ShockNormalizationMixin` PermShk rescaling.
- **#23** — remove working-notes `.md` files from `examples/Distributions/`.

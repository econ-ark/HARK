# Response to Markov variance-reduction advice

**Date:** April 5, 2026
**Context:** Assessment of `Shuffle_vs_IID_Draws-MarkovAdvice.md` after implementing income shuffling and building the Markov notebook section
**Branch:** `harmenberg-dual-measure`

---

## Change 1: Income shock shuffling — DONE, but the advice is outdated

The advice document describes the `get_shocks()` code as using manual CDF inversion (`np.searchsorted` on uniform base draws). That's **not what the current code does** — it uses `draw_events()` + atom indexing. The advice was written against a different version (possibly the `DualMeasureMixin` branch or an older state).

We've already implemented this change. The actual implementation was simpler than the advice suggests: just pass `shuffle=self.income_shuffle` to the existing `.draw(N)` call. No need for the `if/else` branching the advice proposes — `.draw(N, shuffle=False)` is equivalent to the old behavior.

The DualMeasure interaction (Options A/B) is reasonable analysis but out of scope for now.

## Change 2: Markov transition shuffling — Feasible and valuable

This is the most impactful suggestion. The notebook results confirm why: the **variance reduction ratio for income shuffling alone was ~4–5x** in the Markov model, but the Markov transition noise is a separate, large noise source that income shuffling doesn't touch.

The proposed `_draw_shuffled` implementation is **mostly sound** but has issues:

- **The rounding adjustment is fragile.** Sorting by fractional parts and adjusting is fine, but the code has a sign error: `adjust_idx[-(d+1) if diff > 0 else d]` is confusing and likely buggy for `diff < 0`. A cleaner approach: use the same floor-plus-leftover algorithm that `DiscreteDistribution.draw(shuffle=True)` already uses internally. Reuse that pattern rather than reinventing it.

- **The `N_j < 20` threshold is arbitrary.** In practice, the issue is that deterministic rounding with very few agents creates artifacts (e.g., 3 agents with 5% unemployment = 0 unemployed every period). A better criterion: fall back to iid when `N_j * min(probs) < 1`, i.e., when the smallest expected count rounds to zero.

- **The right place to implement this** is on `MarkovProcess.draw()` in `HARK/distributions/base.py`, exactly as the advice suggests. It's orthogonal to income shuffling and should be a separate parameter (`markov_shuffle`).

**Verdict:** Worth implementing as a follow-up. The notebook's Discussion section already flags this as a natural extension.

## Change 3: Permanent-income normalization — Conceptually sound, but tricky in practice

The idea of pinning per-cohort `E[log p]` and `Var[log p]` to analytical values is elegant, and the lognormal argument (two moments pin all power moments) is correct.

However, several practical concerns:

- **The `_analytical_log_pLvl_moments` function is hard to write correctly for Markov models.** The advice itself flags this: you need `g_eff`, effective shock periods accounting for unemployment (no permanent shock when unemployed), and the interaction between Markov state history and growth. This is model-specific — the formulas change if PermGroFac differs across states (as in our notebook, where employed agents grow at 1.00453 and unemployed at 1.0). Getting these formulas right for a general Markov model is a significant research task, not a simple code change.

- **It assumes pLvl is lognormal within each cohort.** This is true for IndShockConsumerType (product of iid lognormals), but for Markov models it's only approximately true — an agent who has spent 3 of 5 periods unemployed has a different pLvl distribution than one who was employed throughout, even within the same age cohort. The mixture over employment histories within a cohort is NOT a single lognormal.

- **The affine rescaling `mu_k + (sigma_k/sigma_hat) * (log_p - mu_hat)` changes the rank ordering** of agents' permanent incomes, which could create subtle issues with wealth-income correlations in the simulation.

- **The claimed 74% SD reduction** is impressive but would need careful validation that it doesn't introduce bias in nonlinear aggregates. The advice claims all power moments are pinned, but this only holds within each cohort if the cohort truly IS lognormal.

**Verdict:** The IndShockConsumerType case (no Markov) is clean and implementable. The Markov case needs more theoretical work to handle state-dependent growth correctly. Defer this until the simpler cases are thoroughly validated.

## Overall assessment

The interaction matrix (all three orthogonal) is correct. The priority ordering should be:

1. **Income shuffle** — Done
2. **Markov transition shuffle** — High value, moderate effort, clean implementation
3. **pLvl normalization** — High potential value, but complex to get right for Markov models; start with the non-Markov case

# Plan: Revive and extend PR #1244-style reshuffling on current HARK

**Canonical file (clone this path):** `docs/plan-reshuffle-pr1244-revival.md`  
**Git branch (plan + changelog only):** `docs/plan-reshuffle-pr1244-revival`  
**Integration work (code):** should live on a **single long-lived integration branch** (see [Integration branch strategy](#integration-branch-strategy-harmenberg--hafiscal-tooling-first)) — not scattered across many local names.

**Status:** Planning document (not yet implemented).  
**Audience:** HARK maintainers and HAFiscal / downstream users.  
**Related:** [PR #1244](https://github.com/econ-ark/HARK/pull/1244) (unmerged), [PR #1691](https://github.com/econ-ark/HARK/pull/1691) (merged: `exact_match` → `shuffle` on `DiscreteDistribution.draw`), [Issue #1690](https://github.com/econ-ark/HARK/issues/1690).

---

## Integration branch strategy (Harmenberg / HAFiscal tooling first)

HAFiscal-related work in this ecosystem has **not** been only on `main`. In particular:

- **`AggIndMrkvConsumerType`** and hierarchical macro+micro Markov live on the upstream branch **`ConsAggIndMarkovModel`** (and merges thereof).
- Local development has used a branch named **`harmenberg-dual-measure`**, which in at least one clone **tracks `origin/ConsAggIndMarkovModel`** and carries **additional** work (e.g. dual-measure / Harmenberg tooling) that is **not** necessarily on `main` yet.

**Implication for this plan:** reshuffle implementation and notebooks should be **developed and tested on top of the same line of commits you use for HAFiscal**, not on bare `main` alone, unless you enjoy repeated merge pain.

### Recommended: one integration branch on `origin`

To avoid a **proliferation of local-only branches** that nobody can find “two nanoseconds from now”:

1. **Publish** a single upstream integration branch with a **stable, memorable name**, for example:
   - **`harmenberg-dual-measure`**, or  
   - **`integration/hafiscal-harmenberg-reshuffle`** (if you prefer a namespace).

2. **Base** that branch on whichever of these is already your “source of truth” for HAFiscal:
   - `origin/ConsAggIndMarkovModel`, **plus** your dual-measure commits (if they are only local, **push** them to this integration branch first).

3. **Merge `origin/main` into it regularly** (or rebase if your team prefers), so you keep **#1691 `shuffle`** and other upstream fixes.

4. **Merge `origin/docs/plan-reshuffle-pr1244-revival`** into the same integration branch (or `git merge origin/docs/plan-reshuffle-pr1244-revival`). That branch is **documentation-only** today and should merge **cleanly**.

5. **Do not merge** the stale **`pull/1244/head`** tree as a merge commit into that line. Use **`git fetch origin pull/1244/head:pr-1244`** only to **read** the old notebook and intent; re-implement simulation changes using **`draw(..., shuffle=True)`** on top of the integration branch.

6. **Upstream PRs to econ-ark/HARK `main`**: when code is ready, open PRs **from a fork/branch** that is either based on current `main` or clearly rebased onto `main` per maintainer preference. The integration branch remains your **daily driver** until those PRs land.

---

## What to do on another local HARK clone

Use this **once** to get a tree that matches “Harmenberg line + plan doc + optional #1244 notebook ref”:

```bash
cd /path/to/HARK
git fetch origin
git fetch origin pull/1244/head:pr-1244   # optional: keeps old example notebook as ref pr-1244

# Choose ONE of the following as your starting tip:

# A) If you have published integration branch harmenberg-dual-measure:
git switch -c harmenberg-dual-measure origin/harmenberg-dual-measure

# B) If integration work is not pushed yet — use upstream Markov branch (HAFiscal-relevant):
git switch -c harmenberg-dual-measure origin/ConsAggIndMarkovModel

# Bring plan document into this branch (if not already included in your integration tip):
git merge origin/docs/plan-reshuffle-pr1244-revival -m "Merge plan branch: reshuffle PR1244 revival doc"

# Stay current with upstream main (resolve conflicts as needed):
git merge origin/main -m "Merge main into harmenberg-dual-measure"
```

**Verify the plan file exists:**

```bash
test -f docs/plan-reshuffle-pr1244-revival.md && echo OK
```

**Optional — extract old reshuffling example notebook** (for reference only):

```bash
git show pr-1244:examples/ConsIndShockModel/IndShockConsumerType_Reshuffling_Example.ipynb > /tmp/IndShockConsumerType_Reshuffling_Example.ipynb
```

**Housekeeping:** After you are happy with the integration tip, **push** it so the other machine can `git fetch` the same name:

```bash
git push -u origin harmenberg-dual-measure
```

(Adjust remote name if you use a fork.)

---

## Goals

1. **Compatibility:** Make reshuffling / exact-match income draws work with **current HARK** (post-#1691 `shuffle` API), implemented and tested on the **same branch line as Harmenberg / dual-measure / ConsAggIndMarkovModel work**, not only on isolated `main` checkouts.
2. **Markov:** Extend beyond `IndShockConsumerType` to **`MarkovConsumerType`** (and thus **`AggIndMrkvConsumerType`** / HAFiscal-style models).
3. **Usability:** Provide a **notebook** and **tests** that compare standard MC vs shuffled draws for the metrics downstream users care about (aggregates, histograms, TM–MC context).
4. **Branch hygiene:** **One** published integration branch + **`main`** + optional **`docs/plan-reshuffle-pr1244-revival`** for the written plan; avoid many undocumented local-only branches.

---

## What PR #1244 actually changed (historical)

On an older `main`, #1244 touched:

| File | Change |
|------|--------|
| `HARK/distribution.py` | `DiscreteDistribution.draw_events(..., exact_match=)` with round-cumsum-permute. |
| `HARK/ConsumptionSaving/ConsIndShockModel.py` | `reshuffle` / `perf_reshuffle`; `get_shocks` used `draw(..., exact_match=...)`; strict `AgentCount` LCM checks; death draws via `Uniform._approx_equiprobable` + exact match. |
| `examples/.../IndShockConsumerType_Reshuffling_Example.ipynb` | Side-by-side baseline vs reshuffle. |
| Tests + `Documentation/CHANGELOG.md` | IndShock tests. |

That branch was **never merged**. Maintainers noted the **example notebook** should be preserved or reproduced.

---

## Conflicts with current HARK (`main` after #1691)

1. **Module layout:** `HARK/distribution.py` → **`HARK/distributions/discrete.py`** (and related packages).
2. **API:** `exact_match` → **`shuffle`** on **`DiscreteDistribution.draw(..., shuffle=True)`** only; algorithm is the generalized floor + leftover slots + permutation ([#1691](https://github.com/econ-ark/HARK/pull/1691)).
3. **`draw_events`:** Still **no** `shuffle` / `exact_match` on `main`. Prefer **`draw(N, shuffle=True)`** for any discrete multivariate income draw instead of duplicating logic on `draw_events`.
4. **Fragile bits in #1244:** e.g. `(float).is_integer()`, and **hard LCM `AgentCount` exceptions** that are **weaker** after #1691 (nice `N` no longer required for a good match).
5. **`MarkovConsumerType.get_shocks` on current `main`:** Uses **uniform + `searchsorted(cumsum(pmv))`** per `(t_cycle, Mrkv)` slice (verify against **your** integration branch at implementation time). Replacement should use **`IncShkDstnNow.draw(N, shuffle=True)`** with the same indexing as `IndShockConsumerType` uses for `draw(N)`.

**Do not** merge #1244 as-is; **rebuild** the intent on **`draw(..., shuffle=True)`** and a small, explicit agent-level API.

---

## Recommended three-PR strategy (upstream `main` targets)

These are **logical** PRs toward **econ-ark/HARK `main`**. Implementation and QA should still be done on your **integration branch** first, then cherry-pick or rebase into PR branches as needed.

### PR 1 — Education only (no simulation core changes)

**Branch name suggestion:** `docs/reshuffle-example-notebook` (or `examples/discrete-shuffle-vs-iid`).

**Content:**

- New or revived notebook under `examples/` (e.g. `examples/Distributions/` or `examples/ConsIndShockModel/`) that:
  - Builds a small **`DiscreteDistribution`** and compares **histogram / TV** of outcomes for `draw(N, shuffle=False)` vs `draw(N, shuffle=True)` vs `draw_events(N)`.
  - States clearly: shuffle **per draw batch** ≈ exact **marginal** match to `pmv`; it does **not** fix joint state issues (e.g. TM-init `(p,m,j)` in HAFiscal).
  - Pins **HARK version / commit** in a markdown cell.
- Optional one-line pointer in `docs/CHANGELOG.md` under Documentation.

**Acceptance:** `nbconvert --execute` on the notebook in CI or documented manual run; no API change.

---

### PR 2 — Opt-in income shuffling: `IndShockConsumerType` + `MarkovConsumerType`

**Branch name suggestion:** `feature/income-shuffle-simulation`.

**Content:**

1. **`IndShockConsumerType`**
   - Add a boolean parameter (name TBD: `income_shuffle`, `shuffle_income_draws`, or align with old `reshuffle`) default **`False`**.
   - In **`get_shocks`**, when `True`, use `IncShkDstnNow.draw(N, shuffle=True)` instead of `draw(N)` / `draw_events(N)` for the relevant blocks; preserve **newborn** semantics from **current** `main` (PermShk / TranShk rules, `NewbornTransShk`, etc.).
   - **Avoid** #1244’s `perf_reshuffle` and **hard LCM exceptions** in v1; optional **warning** if `N` is tiny.
2. **`MarkovConsumerType.get_shocks`**
   - For each mask `these` with count `N`, use `IncShkDstnNow.draw(N, shuffle=self.income_shuffle)` (or inherited flag) instead of manual CDF inversion, assigning `PermShkNow[these]` and `TranShkNow[these]` like today.
3. **Tests**
   - Extend patterns in `tests/test_distribution.py` (`test_shuffle`) to **multivariate** `IncShkDstn` draws where needed.
   - Small integration test: `MarkovConsumerType`, fixed seed, `shuffle=True` → empirical frequencies close to `pmv` per state slice.

**Acceptance:** Full `pytest` green; backward compatible default `False`.

---

### PR 3 — Micro Markov exact-mass transitions + downstream (HAFiscal) notes

**Branch name suggestion:** `feature/micro-markov-exact-mass-optional` (may be split further).

**Content:**

1. **Optional** exact-count / permutation for **micro** employment (or general micro) transitions in **`AggIndMrkvConsumerType.get_micro_markov_states`**, analogous to Krusell–Smith **`RNG.permutation` on precomputed boolean decks** — **not** the same API as `DiscreteDistribution.draw(..., shuffle=True)`.
2. **Design constraint:** If used with **Harmenberg / dual** simulation, any permutation must be **shared** across P and Q tracks so physical shocks stay aligned.
3. **Documentation:** Short section in this plan’s successor or in `docs/guides/simulation.md` on when shuffle helps TM–MC comparisons vs when it does not.
4. **HAFiscal (downstream):** Parameter wiring, `verify_four_methods` / Gatekeeper notes — may live in **HAFiscal-Latest** repo, not necessarily in HARK PR #3.

**Acceptance:** Feature behind a clear default-off flag; tests on employment fractions conditional on `(macro_prev → macro_now)` for a minimal economy.

---

## What not to port from #1244 (unless re-justified)

- **`draw_events(..., exact_match)`** duplicate algorithm — use **`draw(..., shuffle=True)`** only.
- **`perf_reshuffle`** and global **LCM `AgentCount` raises** — defer; #1691 reduces need for “nice” `N`.
- **Death-shock reshuffling** — only if a concrete use case appears; skip in v1.

---

## Notebook / product notes for downstream (e.g. HAFiscal)

- **TM vs MC:** Shuffled MC **income** draws do not by themselves align **TM** (which does not use finite-`N` shuffle) with MC; document when comparing means.
- **Uncertainty:** `shuffle=True` breaks naive **i.i.d.** interpretation of within-period cross-sectional noise; use **across-seed** or **time** variation for bands.
- **RNG:** Clarify interaction of **`DiscreteDistribution` seed** vs **`AgentType.RNG`** for reproducibility.
- **Cross-links:** HAFiscal notebooks such as `pLvl_TM_init_ergodic_gap.ipynb` / Gatekeeper — optional callout that shuffle does **not** replace **joint** `(p,m,j)` burn-in.

---

## Implementation checklist (for PR 2)

- [ ] Re-read **`MarkovConsumerType.get_shocks`** on **your integration branch** (and compare to `origin/main` if they diverge).
- [ ] Use **`IncShkDstnNow.draw(N, shuffle=True)`** return shape consistent with existing **`draw(N)`** path.
- [ ] Newborns: match **current** `PermShk` / `TranShk` policy on the branch you target for merge.
- [ ] Run **`pytest`** for `tests/test_distribution.py` and ConsumptionSaving tests on the **integration** branch.

---

## Retrieving the old example notebook (PR #1244)

```bash
git fetch origin pull/1244/head:pr-1244
git show pr-1244:examples/ConsIndShockModel/IndShockConsumerType_Reshuffling_Example.ipynb > IndShockConsumerType_Reshuffling_Example.ipynb
```

Use it as **narrative inspiration** only; rewrite against **`shuffle=True`** and current class APIs.

---

## Document history

| Date | Note |
|------|------|
| 2026-04-04 | Initial plan added to HARK repo for multi-machine work. |
| 2026-04-04 | Duplicated at `docs/plan-reshuffle-pr1244-revival.md` (expected path); `docs/guides/` holds a pointer only. |
| 2026-04-04 | Revised: **Harmenberg / ConsAggIndMarkovModel-first** integration branch, branch-combination recipe, other-machine steps, de-emphasize `main`-only workflow. |

# PROMPT (HARK repo): the powerlaw-extrapolation illustrative notebook + the binding-constraint gate refinement

**Status:** PENDING owner release. **Vehicle (owner ruling, 2026-07-15): fold into the open
PR [econ-ark/HARK#1782](https://github.com/econ-ark/HARK/pull/1782)** — commits go on its
head branch `fix-aggshock-pf-decay-extrap`; push only on the owner's word.
**Audience:** an AI with NO prior knowledge of this program. Everything you need is stated
here or linked. Work in this checkout (`/home/shared/github/econ-ark/HARK-1782-powerlaw-decay`)
if you are on the owner's machine; otherwise clone econ-ark/HARK, check out
`fix-aggshock-pf-decay-extrap`, and use the PUBLIC links below wherever a local path is
marked (local).

---

## 0. Background in one page (read this even if you read nothing else)

Buffer-stock consumption theory proves two laws about the consumption function `c(m)`
(normalized market resources `m`) beyond any finite solution grid:

- **Above** (high wealth): the "gap" below the perfect-foresight rule,
  `g = c̄(m) − c(m)` with `c̄(m) = κ̲·(m + hNrm)`, decays as a power law
  `g ∝ w̄^{−min(1, q↑)}` in total wealth `w̄ = m + hNrm`, where `q↑` is an eigenvalue
  computed from primitives. Read:
  <https://llorracc.github.io/BufferStockTheory-Latest/powerlaw-decay-theory/> (the
  synthesis paper) — esp. §0–§2.
- **Below** (the borrowing-constraint end): consumption approaches the maximal-MPC line at
  rate `q↓ = ρ` (the CRRA): `c = κ̄·m^e − K·(m^e)^{1+ρ}`, `m^e = m − mNrmMin`,
  `κ̄ = 1 − ℘_eff^{1/ρ}Þ_R` (Theorem CE; extended to permanent shocks as Theorem CE-ψ with
  a regime criterion). Read: the statement page §5 and §5b —
  <https://llorracc.github.io/BufferStockTheory-Latest/powerlaw-decay-theory/statement>
  (anchors: `#st-thm-ce`, `#st-thm-ce-psi`, `#st-prop-c1-psi`, `#st-rem-ce-regime`,
  `#st-cor-c4`; fragments must be lowercase).
- The practical payoff — a small grid finished with these tails reproduces a vastly larger
  grid's solution, and the tails matter INSIDE the solver (the Euler expectations), not
  just on the returned policy — is demonstrated with figures at
  <https://llorracc.github.io/BufferStockTheory-Latest/powerlaw-decay-theory/extrapolators-in-practice>.
  **Your notebook reproduces the spirit of that page's experiment inside HARK.**

**The h-convention trap (memorize):** HARK's `hNrm` EXCLUDES the current period's income;
the theory pages' `h` INCLUDES it (`h = hNrm + 1`). The PF rule is
`c̄(m) = MPCmin·(m + hNrm)`. When this prompt says "human wealth" for grid sizing, it means
**`hNrm`** unless it says otherwise; state the convention once in the notebook.

## 1. What ALREADY EXISTS on this branch (do not re-implement)

PR #1782's head (`1e3ae928` at authoring time) already contains, tested (full
`tests/ConsumptionSaving/` suite: 404 passed):

- The **option surface** on `IndShockConsumerType`: `decay_extrap_form ∈ {None,'powerlaw'}`
  (top tail; exponent auto-computed as `min(1, q_star)` via
  `pf_decay.powerlaw_decay_params_from_agent`, overridable via `decay_extrap_Q`) and
  `decay_extrap_form_lower ∈ {None,'kappabar'}` (bottom tail). Defaults `None` =
  byte-identical stock behavior. Wiring: `pre_solve` swaps in
  `solve_one_period_ConsIndShock_with_tails`; the tails act in BOTH roles (in-solve
  expectations + returned policy).
- `HARK.interpolation.KappaBarTailInterp` (the Theorem-CE bottom tail; guards, `try_make`).
- `pf_decay.ce_psi_regime` (the Theorem CE-ψ regime gate; `ConstraintEndRegimeWarning`),
  `pf_decay.aXtraMin_from_tail_tol`, `pf_decay.aXtraMax_from_tail_tol` (grid-design rules),
  `pf_decay.powerlaw_tail_diagnostic` / `rel_gap_at` (anchor-quality diagnostics).
- The test port: `tests/ConsumptionSaving/test_powerlaw_extrap.py` (40 tests: nested-grid
  fidelity, two-roles, regime gate, byte-zero pins platform-gated) — REUSE its patterns.
- `docs/CHANGELOG.md` bullets under 0.17.3(dev) describing the above.

Your work is a **delta**: Tasks A–C below.

## 2. TASK A — the binding-constraint gate (code change + tests)

**Owner requirement (verbatim intent):** the bottom tail must be built whenever the
NATURAL borrowing constraint is the binding one — including when an artificial constraint
exists but is slack — and must NOT be built when the artificial constraint strictly binds.

**Current behavior to change:** the landed code activates the bottom tail only when
`BoroCnstArt is None` and refuses whenever an artificial constraint is present (see the
refusal path in `solve_one_period_ConsIndShock_with_tails` and its test). That is more
conservative than the theory requires.

**The precise gate:** per period,

    bottom tail active  ⟺  BoroCnstArt is None  OR  BoroCnstNat ≥ BoroCnstArt

- Equality counts as natural (the binding object is then the natural constraint and
  Theorem CE applies).
- When `BoroCnstArt > BoroCnstNat` (artificial strictly binds): refuse, with a clear
  warning — and the docstring must state the THEORY reason: the constraint end is then a
  kink with MPC = 1 on the constrained segment, so Theorem CE's `MPC → κ̄` mechanism does
  not operate there.
- **Per-period for lifecycle agents**: `BoroCnstNat` varies with age; the gate is evaluated
  each backward step, so the tail may be active at some ages and not others.
- The tail's `m^e = m − mNrmMin` coordinate uses the natural constraint's value in the
  allowed cases (it equals `mNrmMin` there by the max).

**Tests (extend `test_powerlaw_extrap.py`):** three cases — no artificial constraint
(tail built; existing), artificial-but-slack `BoroCnstArt < BoroCnstNat` (tail BUILT —
the new behavior; assert fidelity on a small nested check), artificial-binding (clean
refusal + warning; solution equals the tails-off solution on that period). Plus one
lifecycle case where the gate flips across ages (assert per-period wrap types). Keep the
byte-zero default regression untouched and passing.

## 3. TASK B — the analytic validity threshold `w̄₀` (small code + it feeds the notebook)

The theorems are asymptotic: the one-step gap machinery is PROVEN for `w̄ ≥ w̄₀`, and `w̄₀`
has a fully explicit primitive formula in the proofs (written `x₀` there). From
`theory/powerlaw-decay/stage_A_proof.md` in the BufferStockTheory-Latest repo
(local: /home/shared/github/llorracc/BufferStockTheory-Latest; public:
<https://github.com/llorracc/BufferStockTheory-Latest/blob/master/theory/powerlaw-decay/stage_A_proof.md>),
display (5.0a)/(5.0) around line 251:

    x₀⁰ := max{ h + m̄,  (8(ρ+1)ḡ/κ̲ + C₀)/Þ_Γ,  8(ρ+1)C₀/Þ_Γ,  (h+1+C₀)/Þ_Γ,  2ζ }
    x₀  := max{ x₀⁰,  2K̂ }

Take EVERY constant's definition (`ḡ, C₀, ζ, m̄, K_R, K̂`) verbatim from that document's §5
(the `K_R` display sits near line 308) — do not improvise. Note the document's own honesty
(statement.md Remark 7): these constants are **deliberately crude** — `x₀` certifies
validity with explicit constants; it is NOT where extrapolation first becomes accurate in
practice (empirically that happens much earlier).

**Implement** `pf_decay.powerlaw_validity_threshold(...)` returning at least the K̂-free
`x₀⁰` (all five terms are trivial primitives) and, if you implement `K̂` too, the full
`x₀`; if you omit `K̂`, return `x₀⁰` with a documented note that the full threshold adds
`2K̂` (formula cited). THEOREM-REF pin to the stage_A display. Unit-test against a
hand-computed case. **Do NOT make it a refusal gate** — it is a diagnostic: the layered
criteria are (i) `w̄₀` = the guaranteed-validity floor (crude), (ii)
`aXtraMax_from_tail_tol` = the operative quality rule, (iii) "top knot above human wealth"
(`m ≳ h`) = the pedagogical intuition. The notebook presents all three in that order.

## 4. TASK C — the illustrative Jupyter notebook

**Location:** `examples/ConsumptionSaving/PowerlawExtrapolation.ipynb` (committed executed,
per the sibling examples' convention; keep total runtime ≤ ~3 minutes — HARK CI executes
example notebooks; every cell deterministic, no RNG without a seed).

**Agent configuration (PINNED — the tolerance table below was computed on exactly this):**
preferences matching HAFiscal's **College-TOP** type with a **zero-income unemployment
atom** (owner ruling: preferences-only match, clean Theorem-CE bottom): `CRRA = 2.0`,
`DiscFac = 0.995714`, `Rfree = [1.01]`, `PermGroFac = [1 + 0.01958/4]` (quarterly),
`LivPrb = [1 − 1/160]`, **`PermShkStd = [0.003**0.5]`, `TranShkStd = [0.12**0.5]`**
(HAFiscal's quarterly volatilities — do NOT use HARK's 0.1/0.1 defaults, which drop
`q_star` to 0.43 and change every number below), `PermShkCount = TranShkCount = 7`,
`UnempPrb = 0.027`, `IncUnemp = 0.0` (⇒ `mNrmMin = 0`, `℘_eff = UnempPrb`), infinite
horizon (`cycles = 0`). Print the derived theory quantities up front via `pf_decay`:
`q_star` (measured on this config: **0.6727** < 1, so the realized top exponent is
`q_star` itself — say so), `MPCmin` (0.01021), `MPCmax = κ̄`, `hNrm` (**196.8**; BST's
h = hNrm + 1 = 197.8 — state the convention), `ce_psi_regime` (expect regime I), and
`powerlaw_validity_threshold`. EXPECT and explain a `NearResonanceWarning`: this
calibration's `λ_B = E[ψ²]/(ℛÞ_Γ)` sits within 1% of the `q* = 1` knife-edge, so the
`B_ψ` closed-form amplitude route is degraded — knot-matched amplitudes (the default)
are the right choice here; one sentence in the notebook.

**The measured tolerance ↔ grid-top table (computed 2026-07-15 on the pinned config;
`tail_tol` = the RELATIVE consumption gap `(c̄−c)/c` at the knot = the certified error of
handing off to the PF rule there — i.e. what the powerlaw tail SAVES):**

| top knot | aXtraMax ≈ | tail_tol (rel gap) |
|---|---|---|
| m = 1.5·h | 292 | 1.26e-1 |
| m = 2·h   | 390 | **9.4e-2** |
| m = 3·h   | 586 | 6.0e-2 |
| m = 4·h   | 782 | **4.2e-2** |
| m = 24·h (≈ x₀⁰ scale) | 4,698 | 3.7e-3 |

(h = h_BST = 197.8; inverse check: `aXtraMax_from_tail_tol(tol=9.4e-2)` returns the 2·h
knot to ~5%.) The notebook should present this table and its reading: at the owner's
4·h grid the PF rail's handoff error is ~4% of consumption — exactly the error the
powerlaw tail eliminates (the tail is value-matched at the knot, so ITS handoff error is
zero at the knot and grows only through amplitude mis-anchoring, which the fidelity
panels measure directly).

**Narrative sections (with the MyST site links inline — lowercase fragments):**

1. *What the theory says* — 3 short paragraphs + the links (§0's three bullets above),
   including the owner's point stated correctly: *the theorem is asymptotic; the proofs'
   explicit validity threshold `w̄₀` is computed below; in practice the extrapolation
   anchor (the top gridpoint) should sit where market wealth exceeds human wealth — the
   plots draw the `m = hNrm + 1` line. The hard-wired ex-ante recipe is
   `aXtraMax = 2·h_BST` (primitive-computable); the quantitative check is EX POST:
   report `tail_tol` after solving, and `aXtraMax_from_tail_tol` supports a
   solve-measure-re-grid refinement when a target tolerance is demanded.*
2. *Three grid configurations, one truth:*
   - **Truth**: one big-grid solve (e.g. `aXtraMax = 1e6`, `aXtraCount` a few thousand,
     `aXtraNestFac` per the test port's pattern) — solved once, reused.
   - **(G1) Failure case** (owner-specified): `aXtraMax = 4`, small `aXtraCount` (e.g. 16).
     Show: the top knot sits at `m ≈ 5 ≪ hNrm ≈ 197` (pre-asymptotic anchor), the
     extrapolated tail visibly departs the truth above the knot, AND — the teaching moment —
     **HARK's own tooling predicts this**: `powerlaw_tail_diagnostic` flags the anchor,
     `aXtraMax_from_tail_tol` reports the `aXtraMax` actually needed for a target
     tolerance, and on the bottom side too few points violate the `#st-cor-c4` knot rule
     with `aXtraMin_from_tail_tol` as the fix.
   - **(G2) Owner's prescription**: grid extending to **4× human wealth**
     (`aXtraMax ≈ 782 ≈ 4·h_BST` — state the h-convention), sensible `aXtraCount`
     (e.g. 48). Per the table above the PF-rail handoff error there is ~4.2%; expect the
     powerlaw tail to beat it by orders of magnitude — report the measured sup relative
     errors above and below vs truth, and the improvement factor vs rails.
   - **(G3) Guaranteed regime**: a grid whose top knot exceeds the computed `w̄₀ − hNrm`
     (report the number; for these parameters expect `x₀⁰` of order tens of `h` —
     dominated by the `8(ρ+1)ḡ/κ̲` term). Contrast: G2 already performs excellently
     *empirically* though it sits below the *guaranteed* threshold — guarantee vs
     practice, honestly displayed.
3. *The fidelity experiment* (the analog of the last experiment on the
   extrapolators-in-practice page): for G2, solve with tails ON and OFF
   (`decay_extrap_form(:_lower)` set vs `None`), evaluate both against the truth at truth
   gridpoints **above the top knot and below the bottom knot**, and plot: (a) the gap
   `g(w̄)` log–log above (truth line, small-grid knots, tail dashed, the `m = hNrm+1` and
   `w̄₀` verticals); (b) `c/m^e → κ̄` and the `γ = κ̄m^e − c` slope-`(1+ρ)` law below;
   (c) relative-error panels tails-vs-rails; (d) **MPC panels**: MPC → `MPCmin` from above
   at the top, MPC → `κ̄` at the bottom (the most economically legible display). Reuse the
   plotting/measurement patterns of `tests/ConsumptionSaving/test_powerlaw_extrap.py`
   and, for style, the figure code in BufferStockTheory-Latest
   `theory/powerlaw-decay/make_extrap_fidelity_figures.py` (local) — adapt, don't import.
4. *The binding-constraint gate demo* (Task A, user-facing): three cells —
   no artificial constraint (tail on), `BoroCnstArt = −0.5` with a looser natural
   constraint (tail STILL on — the new behavior), `BoroCnstArt = 0.2 > BoroCnstNat`
   (warning + clean refusal; solution matches tails-off). One sentence each on why.
5. *Two-roles coda*: one small cell demonstrating in-solve vs evaluation-only attachment
   (reuse the test port's helper) with the measured factor, and the takeaway sentence:
   the tail must live inside the solver loop.

## 5. The links package (curated; public first, local in parentheses)

| What | Link |
|---|---|
| The synthesis paper (findings, intuition, translation tables) | <https://llorracc.github.io/BufferStockTheory-Latest/powerlaw-decay-theory/> |
| Theorem statements incl. CE / CE-ψ, anchors `#st-thm-ce`, `#st-thm-ce-psi`, `#st-cor-c4` | <https://llorracc.github.io/BufferStockTheory-Latest/powerlaw-decay-theory/statement> |
| The fidelity experiment page (fig7/fig8, two-roles) | <https://llorracc.github.io/BufferStockTheory-Latest/powerlaw-decay-theory/extrapolators-in-practice> |
| Mathematical derivations: top ladder / constants incl. `x₀` (5.0a) | BufferStockTheory-Latest repo `theory/powerlaw-decay/stage_A_proof.md` (local: /home/shared/github/llorracc/BufferStockTheory-Latest/...) |
| Constraint-end proofs (ψ≡1 and ψ-general) | `theory/powerlaw-decay/constraint_end_proof.md`, `constraint_end_proof_psi.md` (same repo) |
| Figure-generating code (styles + measurement patterns) | `theory/powerlaw-decay/make_extrap_fidelity_figures.py`, `make_ce_psi_fig9.py` (same repo) |
| Reference implementation + parameter map + HARK appendix | `theory/powerlaw-decay/powerlaw_extrap_lib.py`, `extrap_fidelity_notes.md` (same repo) |
| Pre-registered batteries + committed outputs | `theory/powerlaw-decay/verify_extrap_fidelity_checks.py`, `verify_ce_psi_checks.py` (+ `_out.txt`) |
| The PR this folds into | <https://github.com/econ-ark/HARK/pull/1782> |

## 6. Process, gates, and the report

- Branch: commits on `fix-aggshock-pf-decay-extrap`, HARK-conventional messages;
  CHANGELOG bullet(s) linking `#1782`; `ruff check` clean; new/changed files
  ruff-formatted (existing dirty files keep their in-file style).
- Gates before you report: full `tests/ConsumptionSaving/` green (the 404 baseline + your
  additions); the notebook executes top-to-bottom in a fresh kernel deterministically
  within the runtime budget; the byte-zero default regression untouched and passing;
  every link in the notebook resolves (fetch each once).
- **Push only on the owner's explicit word** — report first with: commits, test tallies,
  the notebook's measured fidelity numbers (G1 vs G2 vs G3, above and below), the computed
  `w̄₀` value, and any deviations with reasons.

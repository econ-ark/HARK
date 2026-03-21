# Bug Report: MarkovConsumerType Newborn PermShk Suppresses PermGroFac

> **This file and the accompanying MWE script should be deleted before
> this PR is merged into `main`.**

## The Bug

`MarkovConsumerType.get_shocks()` in `ConsMarkovModel.py` contains the
following block that fires every period for newborn agents:

```python
newborn = self.t_age == 0
PermShkNow[newborn] = 1.0
TranShkNow[newborn] = 1.0
```

The composite `PermShk` used by HARK's simulation encodes **both** the
idiosyncratic permanent shock `ψ` and the deterministic growth factor
`PermGroFac` (denoted `G`):

```
PermShk = ψ × G
```

Setting `PermShk = 1.0` for newborns suppresses **both** components.
The `transition()` method (in `ConsIndShockModel.py`) applies:

```
pLvl_new = pLvl_old × PermShk
```

So a newborn with `pLvl_0` drawn from the birth distribution ends up
with `pLvl = pLvl_0 × 1.0 = pLvl_0` after their first period — missing
one period of deterministic permanent income growth.

## The Parent Class Does Not Have This Bug

`IndShockConsumerType.get_shocks()` in `ConsIndShockModel.py` redraws
newborn shocks from the first period's income distribution (lines
2175–2189), giving them `PermShk = ψ × G` — the same composite as
every other agent.  Only the transitory shock is optionally suppressed
via the `NewbornTransShk` flag.

## Consequence

After `A` periods of life (`t_age = A`):

| Version | pLvl formula | E[pLvl \| age A] |
|---|---|---|
| IndShockConsumerType | `pLvl_0 × G^A × ∏ψ` | `E[pLvl_0] × G^A` |
| MarkovConsumerType (buggy) | `pLvl_0 × G^(A-1) × ∏ψ` | `E[pLvl_0] × G^(A-1)` |

Every agent loses one factor of `G` over their lifetime.  Cross-sectional
`E[pLvl]` is biased downward by a factor of `G`.  For typical calibrations
this is 0.35–0.5% — small enough to have gone unnoticed.

## The Fix

Replace the blanket `PermShkNow[newborn] = 1.0` with a per-Markov-state
assignment of `PermShk = PermGroFac[j]`:

```python
newborn = self.t_age == 0
for j in range(self.MrkvArray[0].shape[0]):
    these_nb = np.logical_and(newborn, j == MrkvNow)
    if np.any(these_nb):
        PermShkNow[these_nb] = self.PermGroFac[0][j]
TranShkNow[newborn] = 1.0
```

This restores deterministic growth while deliberately suppressing the
random `ψ` for newborns.  Suppressing `ψ` preserves the calibrated
cross-sectional dispersion from `pLvlInitStd` — adding `ψ` on top
would inflate birth dispersion beyond the calibrated value.

## Design Note: Remaining Behavioral Difference from IndShockConsumerType

After this fix, `MarkovConsumerType` gives newborns `PermShk = G` (no
random `ψ`), while `IndShockConsumerType` gives newborns `PermShk = ψ × G`
(including random `ψ`).  This is a deliberate choice: the Markov version
preserves the calibrated birth dispersion.  Whether `IndShockConsumerType`
should also suppress `ψ` for newborns is a separate design question not
addressed here.

## Minimum Working Example

The script `_DELETE_BEFORE_MERGE_mwe_newborn_permshk.py` demonstrates the
bug by creating two equivalent agents — one `IndShockConsumerType` and one
`MarkovConsumerType` with identical parameters — and comparing newborn `pLvl`
after one period.

### Output on `main` (buggy code):

```
=================================================================
MWE: Newborn pLvl after 1 period (deterministic ψ=1, G=1.05)
=================================================================

  IndShockConsumerType:  mean(pLvl) = 1.0500000000  std = 0.00e+00
  MarkovConsumerType:    mean(pLvl) = 1.0000000000  std = 0.00e+00

  Expected (correct):    pLvl = pLvl_0 × G = 1.0 × 1.05 = 1.05

  *** BUG PRESENT: MarkovConsumerType newborn pLvl = 1.0 (no growth)
      Missing PermGroFac factor. Discrepancy = 4.7619%

-----------------------------------------------------------------
Stochastic test: ψ ∈ {0.8, 1.2} with equal probability
-----------------------------------------------------------------

  IndShockConsumerType:  mean=1.050017  std=0.210000
  MarkovConsumerType:    mean=1.000000  std=0.000000

  Expected mean  = E[ψ] × G = 1.0 × 1.05 = 1.05
  Expected std   = std(ψ) × G = 0.2000 × 1.05 = 0.2100

  *** BUG PRESENT: MarkovConsumerType newborns have ZERO ψ-dispersion
      AND mean pLvl = 1.0 (missing both ψ and PermGroFac)
```

### Output on `fix/markov-newborn-permshk-missing-growth` (fixed code):

```
=================================================================
MWE: Newborn pLvl after 1 period (deterministic ψ=1, G=1.05)
=================================================================

  IndShockConsumerType:  mean(pLvl) = 1.0500000000  std = 0.00e+00
  MarkovConsumerType:    mean(pLvl) = 1.0500000000  std = 0.00e+00

  Expected (correct):    pLvl = pLvl_0 × G = 1.0 × 1.05 = 1.05

  ✓ FIX VERIFIED: MarkovConsumerType newborn pLvl = G = 1.05 (correct)

-----------------------------------------------------------------
Stochastic test: ψ ∈ {0.8, 1.2} with equal probability
-----------------------------------------------------------------

  IndShockConsumerType:  mean=1.050638  std=0.209999
  MarkovConsumerType:    mean=1.050000  std=0.000000

  Expected mean  = E[ψ] × G = 1.0 × 1.05 = 1.05
  Expected std   = std(ψ) × G = 0.2000 × 1.05 = 0.2100

  ✓ FIX VERIFIED: newborns get PermGroFac only (no ψ),
    preserving calibrated pLvlInitStd dispersion.
    mean = 1.050000 = G ✓,  std = 0 (ψ suppressed) ✓

  Note: IndShockConsumerType gives newborns ψ×G (std > 0).
  This is a known behavioral difference, not a bug.
```

## Affected Code

The bug is in HARK's `ConsMarkovModel.py` and affects all subclasses:

- `MarkovConsumerType` (direct)
- `AggIndMrkvConsumerType` (inherits `get_shocks` from `MarkovConsumerType`)
- `AggFiscalType` in HAFiscal (chains to `MarkovConsumerType.get_shocks`)
- Any other subclass that does not override `get_shocks`

The bug has been present since at least HARK 0.14.1.

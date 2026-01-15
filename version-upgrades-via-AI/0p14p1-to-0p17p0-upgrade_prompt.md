# HARK Migration Prompt: 0.14.1 → 0.17.0

**Regenerated**: 2026-01-15
**Meta-prompt version**: Adds target dependency extraction + symbol-based search + cleanup hard-gates
**Source**: HARK 0.14.1
**Target**: HARK 0.17.0

---

## Stage 0: Pre-flight (MANDATORY)

In the **target codebase repo**:

```bash
git remote -v                    # Correct repo?
git branch --show-current        # Correct base branch?
git status                       # Must be clean

git rev-parse HEAD               # Record base SHA: _______________
```

---

## Stage 1 (Discovery): Target-Dependency-First (MANDATORY)

### 1.0 Acquire both HARK versions for comparison

Use PyPI extraction:

```bash
pip download econ-ark==0.14.1 econ-ark==0.17.0 --no-deps -d /tmp/hark_diff
cd /tmp/hark_diff
unzip -q econ_ark-0.14.1-*.whl -d src
unzip -q econ_ark-0.17.0-*.whl -d tgt

diff -rq src/HARK tgt/HARK | sort > /tmp/hark_structural_diff.txt
```

### 1.0b Target Dependency Extraction (MANDATORY - DO THIS FIRST)

#### 1.0b.1 Extract all HARK imports from target

```bash
grep -rn "from HARK\\|import HARK" $TARGET_DIR --include="*.py" | sort -u > target_hark_imports.txt
```

#### 1.0b.2 Extract HARK inheritance in target

```bash
grep -rn "class.*\\(.*Type\\)\\|class.*\\(.*Solver\\)\\|class.*\\(.*Consumer\\)" $TARGET_DIR --include="*.py"
```

#### 1.0b.3 Import smoke test against HARK 0.17.0 (BLOCKING)

For each import pattern in `target_hark_imports.txt`, verify it imports with the **target** HARK.

**If any fail, stop and fix before proceeding.**

Common failures you must explicitly check:
- `ConsIndShockSolver` is **NOT** in `HARK.ConsumptionSaving.ConsIndShockModel` in 0.17.0 (moved to `HARK.ConsumptionSaving.LegacyOOsolvers`).
- `MargValueFunc2D` is **NOT** in `HARK.ConsumptionSaving.ConsAggShockModel` in 0.17.0 (use `HARK.interpolation.MargValueFuncCRRA`).

### 1.2b Export removal check (MANDATORY)

For each module imported by the target, compare `dir(module)` between versions and record removed exports.

### 1.2c Relocation detection (MANDATORY)

For each removed export that the target uses, locate it in 0.17.0:

```bash
# Example:
grep -rn "class ConsIndShockSolver" tgt/HARK --include="*.py"
```

---

## Stage 2 (Transformation): Apply fixes to target codebase

### Step 1: Update import paths (HIGH CONFIDENCE)

#### 1.1 `HARK.distribution` → `HARK.distributions`

```bash
grep -rn "from HARK\\.distribution import" $TARGET_DIR --include="*.py"
```

Replace:
- `from HARK.distribution import ...` → `from HARK.distributions import ...`

#### 1.2 `HARK.parallel` → `HARK.core`

```bash
grep -rn "from HARK\\.parallel import" $TARGET_DIR --include="*.py"
```

Replace:
- `from HARK.parallel import multi_thread_commands` → `from HARK.core import multi_thread_commands`
- `from HARK.parallel import multi_thread_commands_fake` → `from HARK.core import multi_thread_commands_fake`

#### 1.3 `HARK.datasets` → `HARK.Calibration`

```bash
grep -rn "from HARK\\.datasets import" $TARGET_DIR --include="*.py" || true
```

Replace:
- `from HARK.datasets import ...` → `from HARK.Calibration import ...`

---

### Step 1b: Symbol-based search (CRITICAL)

**Do not rely on import-path greps.** Symbols can be imported from multiple locations.

```bash
# Must find/replace ALL occurrences, regardless of how they were imported
grep -rn "multiThreadCommands\\|multiThreadCommandsFake" $TARGET_DIR --include="*.py"
grep -rn "drawDiscrete" $TARGET_DIR --include="*.py"
```

Migration expectations:
- `multiThreadCommands` → `multi_thread_commands`
- `multiThreadCommandsFake` → `multi_thread_commands_fake`

---

### Step 2: Fix relocated/removed symbols (CRITICAL - BLOCKING)

#### 2.1 OOP solver classes moved to `LegacyOOsolvers`

If your code imports any solver classes from `ConsIndShockModel`, update imports:

- **OLD**: `from HARK.ConsumptionSaving.ConsIndShockModel import ConsIndShockSolver`
- **NEW**: `from HARK.ConsumptionSaving.LegacyOOsolvers import ConsIndShockSolver`

Search:

```bash
grep -rn "ConsIndShockSolver" $TARGET_DIR --include="*.py"
```

#### 2.2 `MargValueFunc2D` removed from `ConsAggShockModel`

If your code imports/uses `MargValueFunc2D`:

- **OLD**: `from HARK.ConsumptionSaving.ConsAggShockModel import MargValueFunc2D`
- **NEW**: `from HARK.interpolation import MargValueFuncCRRA as MargValueFunc2D`

Search:

```bash
grep -rn "MargValueFunc2D" $TARGET_DIR --include="*.py"
```

---

### Step 3: Fix removed method `update_solution_terminal` (CRITICAL - BLOCKING)

`update_solution_terminal()` was removed from:
- `AggShockConsumerType`
- `AggShockMarkovConsumerType`
- `MarkovConsumerType`
- `KrusellSmithType`

Search:

```bash
grep -rn "update_solution_terminal\\|updateSolutionTerminal" $TARGET_DIR --include="*.py"
```

Replacement functions in 0.17.0:

| Old | New | Import From |
|-----|-----|-------------|
| `AggShockConsumerType.update_solution_terminal(self)` | `make_aggshock_solution_terminal(CRRA)` | `HARK.ConsumptionSaving.ConsAggShockModel` |
| `AggShockMarkovConsumerType.update_solution_terminal(self)` | `make_aggmrkv_solution_terminal(CRRA, MrkvArray)` | `HARK.ConsumptionSaving.ConsAggShockModel` |
| `MarkovConsumerType.update_solution_terminal(self)` | `make_markov_solution_terminal(CRRA, MrkvArray)` | `HARK.ConsumptionSaving.ConsMarkovModel` |
| `IndShockConsumerType.update_solution_terminal(self)` | `make_basic_CRRA_solution_terminal(CRRA)` | `HARK.ConsumptionSaving.ConsIndShockModel` |

---

### Step 4: Parameter renames (MEDIUM CONFIDENCE)

Rename keys when they flow into `IndShockConsumerType`/`MarkovConsumerType` families:

| Old | New |
|-----|-----|
| `aNrmInitMean` | `kLogInitMean` |
| `aNrmInitStd` | `kLogInitStd` |
| `pLvlInitMean` | `pLogInitMean` |
| `pLvlInitStd` | `pLogInitStd` |

#### 4.1 Update definitions AND accesses (MANDATORY)

```bash
# Definitions (dict literals)
grep -rn "'aNrmInitMean'\\|'pLvlInitMean'" $TARGET_DIR --include="*.py"

# Accesses (dict indexing) - MUST be empty after migration
grep -rn "\\['aNrmInitMean'\\]\\|\\['pLvlInitMean'\\]" $TARGET_DIR --include="*.py"
```

---

### Step 5: camelCase → snake_case compliance (CRITICAL - BLOCKING)

HARK calls lifecycle methods in snake_case. Any camelCase override will **never be invoked**.

#### 5.1 Comprehensive camelCase definitions scan (MANDATORY)

```bash
# Must return empty (or all exceptions annotated with "# OK:")
grep -rn "def [a-z][a-zA-Z]*[A-Z]" $TARGET_DIR --include="*.py"
```

#### 5.2 CamelCase call-site scan (MANDATORY)

```bash
# Must return empty
grep -rn "\\.preSolve(\\|\\.initializeSim(\\|\\.simBirth(\\|\\.simDeath(\\|\\.getShocks(\\|\\.getStates(\\|\\.getMortality(\\|\\.getControls(\\|\\.getMarkovStates(\\|\\.calcAgeDistribution(\\|\\.initializeAges(" $TARGET_DIR --include="*.py"
```

---

### Step 6: Update config/dependency pins (HIGH CONFIDENCE)

- HARK 0.17.0 requires **Python ≥ 3.10**.

Search:

```bash
find $TARGET_DIR -type f \( -name "requirements*.txt" -o -name "pyproject.toml" -o -name "environment*.yml" -o -name "environment*.yaml" -o -path "*/binder/*" \) | \
  xargs grep -l "econ-ark" 2>/dev/null
```

Update:
- `econ-ark==0.14.1` → `econ-ark>=0.17.0`
- `python=3.9` → `python=3.10`
- `requires-python = ">=3.9,<3.10"` → `requires-python = ">=3.10"`

---

## Final Cleanup Verification (MANDATORY)

All of these must be empty before committing:

```bash
# Old import paths
grep -rn "from HARK\\.distribution import\\|from HARK\\.parallel import\\|from HARK\\.datasets import" $TARGET_DIR --include="*.py"

# Old symbol names (search by NAME, not path)
grep -rn "multiThreadCommands\\|multiThreadCommandsFake\\|drawDiscrete" $TARGET_DIR --include="*.py"

# Any camelCase defs
grep -rn "def [a-z][a-zA-Z]*[A-Z]" $TARGET_DIR --include="*.py"

# Old parameter key accesses
grep -rn "\\['aNrmInitMean'\\]\\|\\['pLvlInitMean'\\]" $TARGET_DIR --include="*.py"

# camelCase lifecycle calls
grep -rn "\\.preSolve(\\|\\.initializeSim(\\|\\.getShocks(\\|\\.getStates(" $TARGET_DIR --include="*.py"

# Removed method calls
grep -rn "AggShockConsumerType\\.update_solution_terminal\\|IndShockConsumerType\\.update_solution_terminal\\|MarkovConsumerType\\.update_solution_terminal" $TARGET_DIR --include="*.py"
```

---

## Semantic Verification (USER-DRIVEN)

Run a baseline command **before** migration and rerun **after** migration.

```bash
<USER_COMMAND> > baseline_results.txt 2>&1
<USER_COMMAND> > post_migration_results.txt 2>&1

diff baseline_results.txt post_migration_results.txt
```

---

## What’s *not* automatically required by 0.14.1→0.17.0 (but still scan target)

Some patterns were already updated in HARK 0.14.1, but target code might be older:
- `.pmf` → `.pmv`
- `.X` → `.atoms`
- `RNG.randint` → `RNG.integers`

If the target codebase contains these, treat them as **legacy pre-0.14.1 patterns** and fix them.

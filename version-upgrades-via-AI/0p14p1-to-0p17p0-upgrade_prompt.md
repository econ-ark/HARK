# HARK Migration Prompt: 0.14.1 → 0.17.0

**Generated**: 2026-01-15
**Meta-prompt version**: With Stage 0, method removals, target compliance, and user-driven verification
**Source**: HARK 0.14.1
**Target**: HARK 0.17.0

---

## Stage 0: Pre-flight

Before beginning, confirm:

```bash
# In target codebase repo
git remote -v                    # Correct repo?
git branch --show-current        # Correct base branch?
git status                       # Clean working tree?
git rev-parse HEAD               # Record base SHA: _______________
```

**Filesystem note**: macOS/Windows are typically case-insensitive. No case-only renames are required for this migration.

---

## Prerequisites

1. **Backup your codebase**: `git checkout -b hark-migration-0141-to-0170`
2. **Install target HARK**: `pip install econ-ark==0.17.0` in a test environment
3. **Semantic Baseline**: Before making ANY changes, run your verification command (see Section 7)

---

## Step 1: Update Import Paths (HIGH CONFIDENCE)

### 1.1 Distribution Module

**Search:**
```bash
grep -rn "from HARK\.distribution import\|from HARK import distribution" --include="*.py"
```

**Replace:**
```
from HARK.distribution import  →  from HARK.distributions import
```

### 1.2 Parallel Module

**Search:**
```bash
grep -rn "from HARK\.parallel import\|from HARK import parallel" --include="*.py"
```

**Replace:**
```
from HARK.parallel import multi_thread_commands  →  from HARK.core import multi_thread_commands
from HARK.parallel import multi_thread_commands_fake  →  from HARK.core import multi_thread_commands_fake
```

### 1.3 Datasets Module

**Search:**
```bash
grep -rn "from HARK\.datasets import\|from HARK import datasets" --include="*.py"
```

**Replace:**
```
from HARK.datasets import  →  from HARK.Calibration import
```

---

## Step 2: Update Parameter Names (MEDIUM CONFIDENCE)

### 2.1 Search for Old Parameter Names

```bash
grep -rn "'aNrmInitMean'\|'aNrmInitStd'\|'pLvlInitMean'\|'pLvlInitStd'" --include="*.py"
```

### 2.2 Rename Rules

Only rename if the parameter is passed to `IndShockConsumerType`, `MarkovConsumerType`, or subclasses:

| Old | New |
|-----|-----|
| `'aNrmInitMean'` | `'kLogInitMean'` |
| `'aNrmInitStd'` | `'kLogInitStd'` |
| `'pLvlInitMean'` | `'pLogInitMean'` |
| `'pLvlInitStd'` | `'pLogInitStd'` |

Also rename variable names that feed into these parameters:
```python
# OLD
pLvlInitMean_d = np.log(5)
# NEW
pLogInitMean_d = np.log(5)
```

---

## Step 3: Fix Method Removals (CRITICAL - BLOCKING)

### 3.1 `update_solution_terminal` Method REMOVED

**⚠️ BREAKING CHANGE**: The `update_solution_terminal()` method was **REMOVED** from:
- `AggShockConsumerType` (line 182 in 0.14.1)
- `AggShockMarkovConsumerType` (line 535 in 0.14.1)
- `MarkovConsumerType` (line 931 in 0.14.1)
- `KrusellSmithType` (line 787 in 0.14.1)

**Search:**
```bash
grep -rn "update_solution_terminal\|updateSolutionTerminal" --include="*.py"
```

### 3.2 Replacement Functions in HARK 0.17.0

| Old Method | New Function | Import From |
|------------|--------------|-------------|
| `AggShockConsumerType.update_solution_terminal(self)` | `make_aggshock_solution_terminal(CRRA)` | `HARK.ConsumptionSaving.ConsAggShockModel` |
| `AggShockMarkovConsumerType.update_solution_terminal(self)` | `make_aggmrkv_solution_terminal(CRRA, MrkvArray)` | `HARK.ConsumptionSaving.ConsAggShockModel` |
| `MarkovConsumerType.update_solution_terminal(self)` | `make_markov_solution_terminal(CRRA, MrkvArray)` | `HARK.ConsumptionSaving.ConsMarkovModel` |

### 3.3 Migration Pattern

**OLD code (0.14.1):**
```python
def updateSolutionTerminal(self):
    AggShockConsumerType.update_solution_terminal(self)
    # Custom logic...
    StateCount = self.MrkvArray[-1].shape[0]
    self.solution_terminal.cFunc = StateCount * [self.solution_terminal.cFunc]
```

**NEW code (0.17.0):**
```python
from HARK.ConsumptionSaving.ConsAggShockModel import make_aggshock_solution_terminal

def update_solution_terminal(self):  # Note: snake_case
    self.solution_terminal = make_aggshock_solution_terminal(self.CRRA)
    # Custom logic...
    StateCount = self.MrkvArray[-1].shape[0]
    self.solution_terminal.cFunc = StateCount * [self.solution_terminal.cFunc]
```

---

## Step 4: Fix camelCase Method Definitions (CRITICAL)

### 4.1 Why This Matters

HARK calls lifecycle methods using **snake_case** names. If your code defines methods in **camelCase**, they will **never be invoked**.

**Search for camelCase method definitions:**
```bash
grep -rn "def [a-z][a-zA-Z]*[A-Z]" --include="*.py"
```

### 4.2 HARK Lifecycle Methods (Must Be snake_case)

| camelCase (WRONG) | snake_case (CORRECT) |
|-------------------|---------------------|
| `def preSolve(self)` | `def pre_solve(self)` |
| `def initializeSim(self)` | `def initialize_sim(self)` |
| `def simBirth(self, ...)` | `def sim_birth(self, ...)` |
| `def simDeath(self)` | `def sim_death(self)` |
| `def getShocks(self)` | `def get_shocks(self)` |
| `def getStates(self)` | `def get_states(self)` |
| `def getControls(self)` | `def get_controls(self)` |
| `def getMortality(self)` | `def get_mortality(self)` |
| `def getEconomyData(self, ...)` | `def get_economy_data(self, ...)` |
| `def marketAction(self)` | `def market_action(self)` |
| `def saveState(self)` | `def save_state(self)` |
| `def restoreState(self)` | `def restore_state(self)` |
| `def makeShockHistory(self)` | `def make_shock_history(self)` |
| `def makeIdiosyncraticShockHistories(self)` | `def make_idiosyncratic_shock_histories(self)` |
| `def calcAgeDistribution(self)` | `def calc_age_distribution(self)` |
| `def initializeAges(self)` | `def initialize_ages(self)` |
| `def updateSolutionTerminal(self)` | `def update_solution_terminal(self)` |

### 4.3 Also Rename Method CALLS

```bash
grep -rn "\.preSolve(\|\.initializeSim(\|\.simBirth(\|\.simDeath(\|\.getShocks(\|\.getStates(\|\.getMortality(\|\.getEconomyData(\|\.saveState(\|\.restoreState(" --include="*.py"
```

### 4.4 Important: Apply to ALL Code

**Do NOT skip any files as "dead code".** All code must be made compliant.

---

## Step 5: Update Config Files (HIGH CONFIDENCE)

### 5.1 Find Config Files

```bash
find . -type f \( -name "requirements*.txt" -o -name "pyproject.toml" -o -name "environment*.yml" \) | xargs grep -l "econ-ark"
```

### 5.2 Update Version Pins

| Old | New |
|-----|-----|
| `econ-ark==0.14.1` | `econ-ark>=0.17.0` |
| `econ-ark=0.14.1` | `econ-ark>=0.17.0` |

### 5.3 Update Python Version

HARK 0.17.0 requires **Python ≥3.10**. Check:
- `pyproject.toml`: `requires-python`
- `environment.yml`: `python=` version

---

## Step 6: Verify No Old Patterns Remain

```bash
# Imports (should return nothing)
grep -rn "from HARK\.distribution import" --include="*.py"
grep -rn "from HARK\.parallel import" --include="*.py"
grep -rn "from HARK\.datasets import" --include="*.py"

# Removed methods (should return nothing)
grep -rn "\.update_solution_terminal(" --include="*.py"
grep -rn "AggShockConsumerType\.update_solution_terminal" --include="*.py"

# camelCase lifecycle methods (should return nothing)
grep -rn "def preSolve\|def initializeSim\|def simBirth\|def simDeath\|def getShocks\|def getStates\|def getMortality" --include="*.py"
```

---

## Step 7: Semantic Verification (USER-DRIVEN)

### 7.1 Before Migration

**Question for User**: What command should I run to generate a baseline of computational results?

Example: `./reproduce.sh --comp min` (takes ~1 hour)

**Run and record:**
```bash
# User-provided command
<USER_COMMAND> > baseline_results.txt 2>&1
```

### 7.2 After Migration

Rerun the same command with the NEW HARK version:
```bash
<USER_COMMAND> > post_migration_results.txt 2>&1
```

**Compare:**
```bash
diff baseline_results.txt post_migration_results.txt
```

**Requirement**: Results must match or differences must be explained and accepted.

---

## Summary of Breaking Changes

| Category | Count | Confidence | Blocking? |
|----------|-------|------------|-----------|
| Import: distribution→distributions | Variable | HIGH | No |
| Import: parallel→core | Variable | HIGH | No |
| Import: datasets→Calibration | Variable | HIGH | No |
| Param: aNrmInitMean→kLogInitMean | Variable | MEDIUM | No |
| **Method removal: update_solution_terminal** | **4 classes** | **HIGH** | **YES** |
| **camelCase→snake_case method defs** | **Variable** | **HIGH** | **YES** |
| Config: econ-ark version pin | Variable | HIGH | No |
| Config: Python ≥3.10 | 1 | HIGH | No |

---

## What's NOT Needed (Unchanged Between 0.14.1 and 0.17.0)

- ❌ `.pmf` → `.pmv` (already `.pmv` in 0.14.1)
- ❌ `.X` → `.atoms` (already `.atoms` in 0.14.1)
- ❌ `RNG.randint()` → `RNG.integers()` (already `.integers()` in 0.14.1)
- ❌ `DiscreteDistribution` constructor signature (unchanged)
- ❌ HARK method renames (already snake_case in 0.14.1)

**However**, your TARGET CODEBASE may use old patterns if written for an even older HARK version.

---

## Quick Reference: Find ALL Patterns

```bash
echo "=== Import paths ==="
grep -rn "from HARK\.distribution import\|from HARK\.parallel import\|from HARK\.datasets import" --include="*.py"

echo "=== Parameter names ==="
grep -rn "'aNrmInitMean'\|'aNrmInitStd'\|'pLvlInitMean'\|'pLvlInitStd'" --include="*.py"

echo "=== Removed methods ==="
grep -rn "update_solution_terminal\|updateSolutionTerminal" --include="*.py"

echo "=== camelCase method definitions ==="
grep -rn "def preSolve\|def initializeSim\|def simBirth\|def simDeath\|def getShocks\|def getStates\|def getControls\|def getMortality\|def getEconomyData\|def saveState\|def restoreState\|def makeShockHistory\|def calcAgeDistribution\|def initializeAges\|def updateSolutionTerminal" --include="*.py"

echo "=== camelCase method calls ==="
grep -rn "\.preSolve(\|\.initializeSim(\|\.simBirth(\|\.simDeath(\|\.getShocks(\|\.getStates(\|\.getMortality(\|\.getEconomyData(\|\.saveState(\|\.restoreState(" --include="*.py"

echo "=== Config files ==="
grep -rn "econ-ark.*0\.14" --include="*.txt" --include="*.toml" --include="*.yml" --include="*.yaml"
```

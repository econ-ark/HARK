# HARK Migration Prompt: 0.14.1 → 0.17.0

**Generated**: 2026-01-15 (Regenerated with method removal and compliance checks)
**Based on**: Change Inventory from Stage 1 Discovery (with sections 1.4.4 and 1.4.5)
**Source**: HARK 0.14.1
**Target**: HARK 0.17.0

---

## Prerequisites

Before running this migration:

1. **Backup your codebase**: `git checkout -b hark-migration-0141-to-0170`
2. **Install target HARK**: `pip install econ-ark==0.17.0` in a test environment
3. **Review the Change Inventory**: `inventory/0.14.1-to-0.17.0-change-inventory.md`

---

## Step 1: Update Import Paths (HIGH CONFIDENCE)

### 1.1 Distribution Module

**Search for:**
```bash
grep -rn "from HARK\.distribution import\|from HARK import distribution" --include="*.py"
```

**Replace:**
```
from HARK.distribution import → from HARK.distributions import
```

### 1.2 Parallel Module

**Search for:**
```bash
grep -rn "from HARK\.parallel import\|from HARK import parallel" --include="*.py"
```

**Replace:**
```
from HARK.parallel import multi_thread_commands → from HARK.core import multi_thread_commands
from HARK.parallel import multi_thread_commands_fake → from HARK.core import multi_thread_commands_fake
```

### 1.3 Datasets Module

**Search for:**
```bash
grep -rn "from HARK\.datasets import\|from HARK import datasets" --include="*.py"
```

**Replace:**
```
from HARK.datasets import → from HARK.Calibration import
from HARK.datasets.load_data import → from HARK.Calibration.load_data import
```

---

## Step 2: Update Parameter Names (MEDIUM CONFIDENCE)

### 2.1 Search for Old Parameter Names

```bash
grep -rn "'aNrmInitMean'\|'aNrmInitStd'\|'pLvlInitMean'\|'pLvlInitStd'" --include="*.py"
```

### 2.2 Rename Rules

For each occurrence, check if the parameter dict is passed to:
- `IndShockConsumerType`
- `KinkedRconsumerType`
- `MarkovConsumerType`
- `PortfolioConsumerType`
- Or any class inheriting from these

**If YES, rename:**

| Old | New |
|-----|-----|
| `'aNrmInitMean'` | `'kLogInitMean'` |
| `'aNrmInitStd'` | `'kLogInitStd'` |
| `'pLvlInitMean'` | `'pLogInitMean'` |
| `'pLvlInitStd'` | `'pLogInitStd'` |

### 2.3 Also Rename Variable Names

If the file defines variables with these names that are later used in parameter dicts:
```python
# OLD
pLvlInitMean_d = np.log(5)
# NEW
pLogInitMean_d = np.log(5)
```

---

## Step 3: Fix Method Removals (CRITICAL - BLOCKING)

### 3.1 `update_solution_terminal` Method Removed

**⚠️ BREAKING CHANGE**: The `update_solution_terminal()` method was **REMOVED** from:
- `AggShockConsumerType`
- `AggShockMarkovConsumerType`
- `MarkovConsumerType`
- `KrusellSmithType`

**Search for usage:**
```bash
grep -rn "update_solution_terminal\|updateSolutionTerminal" --include="*.py"
```

**Replacement functions in HARK 0.17.0:**

| Old Method | New Function | Import From |
|------------|--------------|-------------|
| `AggShockConsumerType.update_solution_terminal(self)` | `make_aggshock_solution_terminal(CRRA)` | `HARK.ConsumptionSaving.ConsAggShockModel` |
| `AggShockMarkovConsumerType.update_solution_terminal(self)` | `make_aggmrkv_solution_terminal(CRRA, MrkvArray)` | `HARK.ConsumptionSaving.ConsAggShockModel` |
| `MarkovConsumerType.update_solution_terminal(self)` | `make_markov_solution_terminal(CRRA, MrkvArray)` | `HARK.ConsumptionSaving.ConsMarkovModel` |

### 3.2 Migration Pattern

**OLD code (0.14.1):**
```python
def updateSolutionTerminal(self):
    AggShockConsumerType.update_solution_terminal(self)
    # Custom logic after parent call...
    StateCount = self.MrkvArray[-1].shape[0]
    self.solution_terminal.cFunc = StateCount * [self.solution_terminal.cFunc]
```

**NEW code (0.17.0):**
```python
from HARK.ConsumptionSaving.ConsAggShockModel import make_aggshock_solution_terminal

def update_solution_terminal(self):  # Note: snake_case now
    self.solution_terminal = make_aggshock_solution_terminal(self.CRRA)
    # Custom logic after...
    StateCount = self.MrkvArray[-1].shape[0]
    self.solution_terminal.cFunc = StateCount * [self.solution_terminal.cFunc]
```

**Key changes:**
1. Import the standalone function
2. Call function and assign result to `self.solution_terminal`
3. Rename method to snake_case if it was camelCase

---

## Step 4: Fix camelCase Method Definitions (CRITICAL)

### 4.1 Why This Matters

HARK calls lifecycle methods using **snake_case** names. If your code defines methods in **camelCase**, they will **never be invoked** by HARK.

**Search for camelCase method definitions:**
```bash
grep -rn "def [a-z][a-zA-Z]*[A-Z]" --include="*.py"
```

### 4.2 HARK Lifecycle Methods That MUST Be snake_case

| camelCase (WRONG) | snake_case (CORRECT) |
|-------------------|---------------------|
| `def preSolve(self)` | `def pre_solve(self)` |
| `def postSolve(self)` | `def post_solve(self)` |
| `def initializeSim(self)` | `def initialize_sim(self)` |
| `def simBirth(self, ...)` | `def sim_birth(self, ...)` |
| `def simDeath(self)` | `def sim_death(self)` |
| `def getShocks(self)` | `def get_shocks(self)` |
| `def getStates(self)` | `def get_states(self)` |
| `def getControls(self)` | `def get_controls(self)` |
| `def getPostStates(self)` | `def get_poststates(self)` |
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

When renaming definitions, also rename all calls:
- `self.preSolve()` → `self.pre_solve()`
- `ParentClass.initializeSim(self)` → `ParentClass.initialize_sim(self)`
- `agent.getShocks()` → `agent.get_shocks()`

**Search for camelCase method calls:**
```bash
grep -rn "\.preSolve(\|\.initializeSim(\|\.simBirth(\|\.simDeath(\|\.getShocks(\|\.getStates(\|\.getMortality(\|\.getEconomyData(\|\.saveState(\|\.restoreState(" --include="*.py"
```

### 4.4 Important: Apply to ALL Code

**Do NOT skip any files as "dead code" or "unused".** All code in the repository must be made compliant. Code that appears unused today may be resurrected later.

---

## Step 5: Update Config Files (HIGH CONFIDENCE)

### 5.1 Find Config Files

```bash
find . -type f \( -name "requirements*.txt" -o -name "pyproject.toml" -o -name "environment*.yml" -o -name "setup.py" \) | xargs grep -l "econ-ark"
```

### 5.2 Update Version Pins

| Old Pattern | New Pattern |
|-------------|-------------|
| `econ-ark==0.14.1` | `econ-ark>=0.17.0` |
| `econ-ark=0.14.1` | `econ-ark>=0.17.0` |
| `econ-ark>=0.14` | `econ-ark>=0.17.0` |

### 5.3 Update Python Version (If Needed)

HARK 0.17.0 requires Python ≥3.10. Check:
- `pyproject.toml`: `requires-python`
- `environment.yml`: `python=` version

---

## Step 6: Verify No Old Patterns Remain

### 6.1 Import Verification

```bash
# Should return nothing
grep -rn "from HARK\.distribution import" --include="*.py"
grep -rn "from HARK\.parallel import" --include="*.py"
grep -rn "from HARK\.datasets import" --include="*.py"
```

### 6.2 Method Removal Verification

```bash
# Should return nothing (no calls to removed methods)
grep -rn "\.update_solution_terminal(" --include="*.py"
grep -rn "AggShockConsumerType\.update_solution_terminal" --include="*.py"
grep -rn "MarkovConsumerType\.update_solution_terminal" --include="*.py"
```

### 6.3 camelCase Method Verification

```bash
# Should return nothing (no camelCase HARK lifecycle methods)
grep -rn "def preSolve\|def initializeSim\|def simBirth\|def simDeath\|def getShocks\|def getStates\|def getMortality" --include="*.py"
```

### 6.4 Syntax Check

```bash
find . -name "*.py" -exec python -m py_compile {} \;
```

---

## Step 7: Run Tests

```bash
# Import check
python -c "from HARK.distributions import DiscreteDistribution; print('OK')"

# Run your test suite
pytest
```

---

## Summary of Changes

| Category | Count | Confidence | Blocking? |
|----------|-------|------------|-----------|
| Import: distribution→distributions | Variable | HIGH | No |
| Import: parallel→core | Variable | HIGH | No |
| Import: datasets→Calibration | Variable | HIGH | No |
| Param: aNrmInitMean→kLogInitMean | Variable | MEDIUM | No |
| **Method removal: update_solution_terminal** | **Variable** | **HIGH** | **YES** |
| **camelCase→snake_case method defs** | **Variable** | **HIGH** | **YES** |
| Config: econ-ark version | Variable | HIGH | No |

---

## What's NOT Needed (Unchanged Between 0.14.1 and 0.17.0)

The following were already correct in HARK 0.14.1:

- ❌ `.pmf` → `.pmv` - already `.pmv` in 0.14.1
- ❌ `.X` → `.atoms` - already `.atoms` in 0.14.1
- ❌ `RNG.randint()` → `RNG.integers()` - already `.integers()` in 0.14.1
- ❌ `DiscreteDistribution` positional argument order - unchanged

**However**, your TARGET CODEBASE may still use old patterns if it was written for an even older HARK version. Always scan your codebase for these patterns regardless.

---

## Manual Review Items

1. **Rfree as list**: If your code assigns `Rfree` as a scalar, check if it needs to be a list
2. **SCF calibration outputs**: `aNrmInitMean` is still used in SCF output dicts; may need conversion layer
3. **Custom `update_solution_terminal` logic**: Review what your customizations did and adapt to new function-based pattern

---

## Quick Reference: All Breaking Changes

```bash
# Run this to find ALL patterns that need changing:

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

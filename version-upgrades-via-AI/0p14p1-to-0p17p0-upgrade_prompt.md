# HARK Migration Prompt: 0.14.1 → 0.17.0

**Generated**: 2026-01-15
**Based on**: Change Inventory from Stage 1 Discovery
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

## Step 3: Update Config Files (HIGH CONFIDENCE)

### 3.1 Find Config Files

```bash
find . -type f \( -name "requirements*.txt" -o -name "pyproject.toml" -o -name "environment*.yml" -o -name "setup.py" \) | xargs grep -l "econ-ark"
```

### 3.2 Update Version Pins

| Old Pattern | New Pattern |
|-------------|-------------|
| `econ-ark==0.14.1` | `econ-ark>=0.17.0` |
| `econ-ark=0.14.1` | `econ-ark>=0.17.0` |
| `econ-ark>=0.14` | `econ-ark>=0.17.0` |

### 3.3 Update Python Version (If Needed)

HARK 0.17.0 requires Python ≥3.10. Check:
- `pyproject.toml`: `requires-python`
- `environment.yml`: `python=` version

---

## Step 4: Verify No Old Patterns Remain

### 4.1 Import Verification

```bash
# Should return nothing
grep -rn "from HARK\.distribution import" --include="*.py"
grep -rn "from HARK\.parallel import" --include="*.py"
grep -rn "from HARK\.datasets import" --include="*.py"
```

### 4.2 Syntax Check

```bash
find . -name "*.py" -exec python -m py_compile {} \;
```

---

## Step 5: Run Tests

```bash
# Import check
python -c "from HARK.distributions import DiscreteDistribution; print('OK')"

# Run your test suite
pytest
```

---

## Summary of Changes

| Category | Pattern Count | Confidence |
|----------|---------------|------------|
| Import: distribution→distributions | Variable | HIGH |
| Import: parallel→core | Variable | HIGH |
| Import: datasets→Calibration | Variable | HIGH |
| Param: aNrmInitMean→kLogInitMean | Variable | MEDIUM |
| Config: econ-ark version | Variable | HIGH |

---

## What's NOT Needed (Already Current in 0.14.1)

The following changes are **NOT required** because 0.14.1 already uses the new patterns:

- ❌ Method renames (camelCase→snake_case) - already snake_case
- ❌ `.pmf` → `.pmv` - already `.pmv`
- ❌ `.X` → `.atoms` - already `.atoms`
- ❌ `RNG.randint()` → `RNG.integers()` - already `.integers()`
- ❌ `DiscreteDistribution` constructor signature - unchanged

---

## Manual Review Items

1. **Rfree as list**: If your code assigns `Rfree` as a scalar, check if it needs to be a list
2. **SCF calibration outputs**: `aNrmInitMean` is still used in SCF output dicts; may need conversion layer

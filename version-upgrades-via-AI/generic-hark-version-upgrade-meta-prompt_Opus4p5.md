# Meta-Prompt (Opus 4.5): Two-Stage HARK Version Upgrade System

## Overview

This meta-prompt implements a **two-stage workflow** for upgrading codebases between HARK versions:

```
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: Discovery                                             │
│  Produces: Change Inventory (verified, reviewable artifact)     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                   [Human review checkpoint]
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: Transformation (user chooses one)                     │
│    • Option A: Human Migration Checklist                        │
│    • Option B: Automated Migration Script                       │
└─────────────────────────────────────────────────────────────────┘
```

**Why two stages?** Discovery and transformation are different tasks with different failure modes. Separating them allows:
- Verification of what was found before any code is changed
- Choice of transformation approach based on codebase size and risk tolerance
- Debugging: if results are wrong, you know whether discovery or transformation failed
- Reusability: the Change Inventory is documentation even without automation


---

## Inputs (Fill These In)

```yaml
source_version: "{SOURCE_VERSION}"      # e.g., "0.14.1"
target_version: "{TARGET_VERSION}"      # e.g., "0.16.1"
target_codebase: "{TARGET_DIR}"         # optional for Stage 1, required for Stage 2
file_types:
  code: [".py", ".ipynb"]               # optionally add ".sh"
  config: ["requirements*.txt", "pyproject.toml", "setup.py", "setup.cfg",
           "environment*.yml", "environment*.yaml", "Pipfile", "binder/*"]
constraints: []                         # e.g., ["no network", "no git tags"]
```

---

# ═══════════════════════════════════════════════════════════════════
# STAGE 0: PRE-FLIGHT (MANDATORY)
# ═══════════════════════════════════════════════════════════════════

## Purpose

Ensure you are upgrading the *intended* codebase from the correct baseline, with a clean and reproducible starting point.

## 0.1 Branch and Working Tree Hygiene (MANDATORY)

In the TARGET CODEBASE repo, verify:

```bash
# Confirm you are in the correct repo and on the intended base branch
git remote -v
git branch --show-current

# Confirm clean working tree
git status

# Ensure you are up-to-date with upstream
git fetch --all --prune
# If using main/master, choose ONE:
#   git pull --ff-only
# or
#   git rebase origin/<base-branch>
```

**Hard requirements**:
- If `git status` is not clean, STOP and commit/stash/clean before proceeding.
- Record the base commit SHA (for reproducibility):

```bash
git rev-parse HEAD
```

## 0.2 Environment & Filesystem Notes (IMPORTANT)

Record:
- OS + filesystem case-sensitivity (macOS/Windows often case-insensitive)
- Python version
- How HARK versions are obtained (PyPI wheels vs git tags)

This matters for case-only renames and reproducibility.

## 0.3 Target Directory / Scope Confirmation (MANDATORY)

Write down:
- `TARGET_DIR` being upgraded
- Whether you are upgrading *library code*, *application code*, or *research code*
- Whether you must upgrade **all files** (default: YES; never skip “dead code”)

---

# ═══════════════════════════════════════════════════════════════════
# STAGE 1: DISCOVERY
# ═══════════════════════════════════════════════════════════════════

## Purpose

Produce a **Change Inventory**: a structured, evidence-backed document listing every breaking change between `{SOURCE_VERSION}` and `{TARGET_VERSION}`.

This document is the **contract** for Stage 2. Nothing should be transformed that isn't in the inventory.

---

## 1.0 Acquire Both Versions (Prerequisite)

You MUST have both versions accessible for comparison.

### Option A: PyPI Extraction (No Git Required)

```bash
pip download econ-ark=={SOURCE_VERSION} econ-ark=={TARGET_VERSION} --no-deps -d /tmp/hark_diff
cd /tmp/hark_diff
unzip -q econ_ark-{SOURCE_VERSION}-*.whl -d src
unzip -q econ_ark-{TARGET_VERSION}-*.whl -d tgt
```

### Option B: Git Clone with Tags

```bash
git clone https://github.com/econ-ark/HARK.git
cd HARK
# Verify tags exist
git tag | grep -E "^{SOURCE_VERSION}$|^{TARGET_VERSION}$"
```

### Verification Checkpoint ☐

Before proceeding, confirm ALL of these:
- ☐ I can `ls src/HARK/` and `ls tgt/HARK/` (or equivalent git paths)
- ☐ I can run `diff -rq src/HARK tgt/HARK`
- ☐ I can import modules from each version (for API extraction)

**If any fail, STOP and explain what's blocking you.**

---

## 1.1 Structural Changes (Files/Directories)

**Goal**: Identify modules renamed, moved, split, merged, added, or removed.

```bash
# List structural differences
diff -rq src/HARK tgt/HARK 2>/dev/null | sort

# Files only in source (removed or renamed)
diff -rq src/HARK tgt/HARK | grep "Only in src"

# Files only in target (added or renamed-to)
diff -rq src/HARK tgt/HARK | grep "Only in tgt"
```

**Record in Change Inventory**:

| Change Type | Old Path | New Path | Evidence |
|-------------|----------|----------|----------|
| Module renamed | `distribution.py` | `distributions/` | `diff -rq` output |
| Module removed | `frame.py` | — | Only in src |
| Module added | — | `simulator.py` | Only in tgt |

---

### 1.1b Case-only Rename Hazards (IMPORTANT)

On case-insensitive filesystems (common on macOS/Windows), **renames that only change letter-case** (e.g., `Foo.py` → `foo.py`) can be missed or behave strangely in git checkouts and merges.

#### Required Checks

```bash
# Detect whether git thinks the working tree is case-insensitive
# (true often implies case-insensitive FS behavior)
git config --get core.ignorecase || true

# If you suspect case-only changes, check using a case-sensitive environment (Linux CI) and/or:
# - ensure no files differ only by case
# - ensure renames are done as two-step renames (Foo.py -> Foo_tmp.py -> foo.py)
```

#### Migration Rule

- Avoid introducing case-only renames during an API upgrade unless necessary.
- If required, document them explicitly in the Change Inventory and handle with a two-step rename.

---
## 1.2 API Surface Extraction (MANDATORY)

This is the **most important** discovery step. It catches constructor changes, signature changes, and subtle renames that file diffs miss.

### 1.2.1 Generate API Reports

For each version, produce a structured report:

```python
# api_extract.py - run once per version
import importlib
import inspect
import json
import pkgutil
import sys

def extract_api(package_path, package_name="HARK"):
    """Extract public API: modules, classes, functions, signatures."""
    sys.path.insert(0, package_path)
    api = {"modules": {}}

    try:
        pkg = importlib.import_module(package_name)
    except ImportError as e:
        return {"error": str(e)}

    for importer, modname, ispkg in pkgutil.walk_packages(
        pkg.__path__, prefix=f"{package_name}."
    ):
        try:
            mod = importlib.import_module(modname)
            mod_info = {"classes": {}, "functions": {}}

            for name, obj in inspect.getmembers(mod):
                if name.startswith("_"):
                    continue
                if inspect.isclass(obj) and obj.__module__ == modname:
                    mod_info["classes"][name] = {
                        "bases": [b.__name__ for b in obj.__bases__],
                        "init_sig": str(inspect.signature(obj.__init__))
                                   if hasattr(obj, "__init__") else None,
                        "methods": [m for m in dir(obj) if not m.startswith("_")]
                    }
                elif inspect.isfunction(obj) and obj.__module__ == modname:
                    mod_info["functions"][name] = {
                        "signature": str(inspect.signature(obj))
                    }

            if mod_info["classes"] or mod_info["functions"]:
                api["modules"][modname] = mod_info
        except Exception as e:
            api["modules"][modname] = {"error": str(e)}

    return api

if __name__ == "__main__":
    import sys
    result = extract_api(sys.argv[1])
    print(json.dumps(result, indent=2))
```

Run for both versions:
```bash
python api_extract.py src > api_src.json
python api_extract.py tgt > api_tgt.json
```

### 1.2.2 Diff the API Reports

Compare the JSON files to find:
- **Removed symbols**: in src but not tgt
- **Added symbols**: in tgt but not src
- **Signature changes**: same name, different signature
- **Moved symbols**: same name+signature, different module

**Record in Change Inventory** (example):

| Symbol | Change | Old | New | Evidence |
|--------|--------|-----|-----|----------|
| `DiscreteDistribution.__init__` | Signature | `(pmf, X, seed=0)` | `(pmv, atoms, seed=0, limit=None)` | API diff |
| `initializeSim` | Renamed | `initializeSim` | `initialize_sim` | API diff |
| `MargValueFunc2D` | Moved/Aliased | `ConsAggShockModel` | Requires `as MargValueFunc2D` | API diff |

---

## 1.3 Deprecation Warnings as Discovery Source

**NEW**: Run old code against new HARK and capture warnings.

```python
import warnings
warnings.filterwarnings("record")

# Import the new HARK
import HARK

# Try to use old patterns
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", DeprecationWarning)
    # Import modules that might emit warnings
    from HARK.ConsumptionSaving import ConsIndShockModel
    # ... more imports ...

    for warning in w:
        print(f"DEPRECATED: {warning.message}")
```

**Record any deprecation warnings in the Change Inventory.**

---

## 1.4 Systematic Rename Waves

### 1.4.1 Method Renames (camelCase → snake_case)

```bash
# Extract all method definitions from both versions
grep -rh "def [a-z]" src/HARK --include="*.py" | sed 's/.*def //' | sed 's/(.*/:src/' | sort -u > methods_src.txt
grep -rh "def [a-z]" tgt/HARK --include="*.py" | sed 's/.*def //' | sed 's/(.*/:tgt/' | sort -u > methods_tgt.txt

# Find methods only in source (potentially renamed)
comm -23 <(cut -d: -f1 methods_src.txt | sort -u) <(cut -d: -f1 methods_tgt.txt | sort -u)
```

**Build a complete mapping table**:

| Old Name | New Name | Applies To | Contexts |
|----------|----------|------------|----------|
| `initializeSim` | `initialize_sim` | AgentType subclasses | def, self., var., Parent.() |
| `preSolve` | `pre_solve` | AgentType subclasses | def, self., var., Parent.() |
| `addToTimeVary` | `add_to_time_vary` | AgentType subclasses | def, self., var. |
| `drawDiscrete` | `draw_events` | DiscreteDistribution | var. |
| `multiThreadCommands` | `multi_thread_commands` | HARK module | import, call |

### 1.4.2 Contexts That Must Be Rewritten

For EACH renamed symbol, identify ALL contexts:

1. **Definitions**: `def oldName(self):`
2. **Self calls**: `self.oldName()`
3. **Variable calls**: `agent.oldName()`
4. **Indexed calls**: `agents[i].oldName()`
5. **Parent calls**: `ParentClass.oldName(self)`
6. **String commands**: `'oldName()'` (used in multi_thread_commands)
7. **Attribute assignment**: `self.oldName = ...`
8. **Dictionary keys**: `{'oldName': value}`
9. **Import aliases**: `from X import oldName as alias`

### 1.4.3 Attribute/Property Renames (DISTINCT FROM METHOD RENAMES)

**⚠️ THIS IS COMMONLY MISSED!** Attribute renames are syntactically different from method renames:
- Method: `obj.oldMethod()` (has parentheses)
- Attribute: `obj.oldAttr` (no parentheses, reading a value)

These require SEPARATE discovery because grep patterns for methods won't catch them.

**Discovery commands**:

```bash
# Find attribute/property definitions in HARK classes
# Look for self.xyz = patterns that changed names
grep -rh "self\.[a-z].*=" src/HARK --include="*.py" | sed 's/.*self\.//' | sed 's/ .*//' | sort -u > attrs_src.txt
grep -rh "self\.[a-z].*=" tgt/HARK --include="*.py" | sed 's/.*self\.//' | sed 's/ .*//' | sort -u > attrs_tgt.txt
comm -23 attrs_src.txt attrs_tgt.txt  # Attributes only in source

# Find @property decorators that changed
diff <(grep -B1 -rh "@property" src/HARK --include="*.py" | grep "def ") \
     <(grep -B1 -rh "@property" tgt/HARK --include="*.py" | grep "def ")

# Specifically for distribution objects - compare class attributes
git diff {SOURCE_VERSION}..{TARGET_VERSION} -- HARK/distributions/*.py | grep -E "^\+.*self\.|^\-.*self\."
```

**Critical attributes to check**:

| Class | Old Attribute | New Attribute | Context |
|-------|--------------|---------------|---------|
| `DiscreteDistribution` | `.pmf` | `.pmv` | `dist.pmf` access |
| `DiscreteDistribution` | `.X` | `.atoms` | `dist.X` access |
| Agent types | `.IncomeDstn` | `.IncShkDstn` | `agent.IncomeDstn[0]` |

**Search patterns for attribute access** (different from method calls!):

```bash
# Attribute read access (no parentheses after)
grep -rE "\.pmf\b[^(]|\.pmf$" $TARGET_DIR --include="*.py"
grep -rE "\.X\b[^(]|\.X$" $TARGET_DIR --include="*.py"
grep -rE "\.IncomeDstn\b" $TARGET_DIR --include="*.py"

# Getattr patterns
grep -rE "getattr\(.*['\"]pmf['\"]|getattr\(.*['\"]X['\"]" $TARGET_DIR --include="*.py"
```

**Record in Change Inventory**:

| Old Attribute | New Attribute | On Class(es) | Access Patterns to Search |
|--------------|---------------|--------------|---------------------------|
| `.pmf` | `.pmv` | DiscreteDistribution, subclasses | `\.pmf\b`, `['pmf']`, `getattr(...,'pmf')` |
| `.X` | `.atoms` | DiscreteDistribution, subclasses | `\.X\b` (careful: X is common!) |
| `.IncomeDstn` | `.IncShkDstn` | IndShockConsumerType, subclasses | `\.IncomeDstn` |

---


### 1.4.4 Method Removals and Architectural Changes (CRITICAL - COMMONLY MISSED)

⚠️ **This is distinct from method RENAMES.** A method rename keeps the same functionality under a new name. A method REMOVAL means the method no longer exists and may be replaced with a completely different pattern (e.g., standalone function, constructor-dict).

#### Discovery Commands

```bash
# Extract class method names from both HARK versions
grep -rh "def [a-z]" src/HARK/ConsumptionSaving --include="*.py" | \
    sed 's/.*def //' | sed 's/(.*$//' | sort -u > methods_src.txt
grep -rh "def [a-z]" tgt/HARK/ConsumptionSaving --include="*.py" | \
    sed 's/.*def //' | sed 's/(.*$//' | sort -u > methods_tgt.txt

# Find methods in source but NOT in target (potential removals)
comm -23 methods_src.txt methods_tgt.txt > methods_removed.txt

# Check if "removed" methods became standalone functions
for method in $(cat methods_removed.txt); do
    echo "=== $method ==="
    grep -rn "^def $method\|^def ${method}_" tgt/HARK --include="*.py" | head -3
done
```

#### High-Priority Classes to Check for Removals

| Class | Location | Why Important |
|-------|----------|---------------|
| `AggShockConsumerType` | ConsAggShockModel | Aggregate shock modeling |
| `MarkovConsumerType` | ConsMarkovModel | Markov state transitions |
| `IndShockConsumerType` | ConsIndShockModel | Base agent type |
| `AgentType` | core.py | Base class for all agents |

#### For Each Removed Method, Determine:

1. **Was it renamed?** Check if a similar snake_case version exists in target
2. **Was it converted to a standalone function?** Search for `def method_name(` at module level
3. **Was it replaced with a constructor-dict pattern?** Search for patterns like:
   ```python
   constructor_dict = {
       "solution_terminal": make_xxx_solution_terminal,
   }
   ```
4. **Was it truly removed with no replacement?** Document as blocking breaking change

#### Record in Change Inventory:

| Class | Removed Method | Replacement | Migration Pattern |
|-------|----------------|-------------|-------------------|
| `AggShockConsumerType` | `update_solution_terminal` | `make_aggshock_solution_terminal(CRRA)` | Method→Function |
| `MarkovConsumerType` | `update_solution_terminal` | `make_markov_solution_terminal(CRRA, MrkvArray)` | Method→Function |

#### Migration Rules for Method→Function Conversions:

```python
# OLD pattern (0.14.1) - calling parent class method:
def updateSolutionTerminal(self):
    AggShockConsumerType.update_solution_terminal(self)
    # ... additional customization ...

# NEW pattern (0.17.0) - calling standalone function:
from HARK.ConsumptionSaving.ConsAggShockModel import make_aggshock_solution_terminal

def update_solution_terminal(self):
    self.solution_terminal = make_aggshock_solution_terminal(self.CRRA)
    # ... additional customization ...
```

### 1.4.5 Target Codebase Method Compliance Check (CRITICAL - COMMONLY MISSED)

⚠️ **The target codebase may have NEVER been compliant** with even the SOURCE HARK version. This step catches legacy code that has camelCase method definitions when HARK expects snake_case.

**Important principle**: Do NOT skip any code as "dead" or "unused". Code that appears unused today may be resurrected later. ALL code in the target codebase must be made compliant.

#### Why This Matters

If user code defines:
```python
def preSolve(self):  # camelCase
    ...
```

But HARK calls:
```python
agent.pre_solve()  # snake_case
```

The user's method will NEVER be invoked! This is a silent failure.

#### Discovery Commands (Run Against TARGET CODEBASE)

```bash
# Find ALL camelCase method definitions in target codebase that might be HARK overrides
grep -rn "def [a-z][a-zA-Z]*[A-Z]" $TARGET_DIR --include="*.py" > camelcase_methods.txt

# Compare against HARK's expected lifecycle method names
HARK_LIFECYCLE_METHODS=(
    "pre_solve" "post_solve" "initialize_sim" "sim_birth" "sim_death"
    "get_shocks" "get_states" "get_controls" "get_poststates" "get_mortality"
    "update_solution_terminal" "make_shock_history" "read_shocks_from_history"
    "get_economy_data" "market_action" "calc_age_distribution" "initialize_ages"
    "save_state" "restore_state" "reset"
)

# For each HARK method, check if target has camelCase version
for method in "${HARK_LIFECYCLE_METHODS[@]}"; do
    camel=$(echo "$method" | sed -r 's/_([a-z])/\U\1/g')  # Convert to camelCase
    grep -rn "def $camel(" $TARGET_DIR --include="*.py"
done
```

#### Build Compliance Table

| Target File | Has Definition | HARK Expects | Status |
|-------------|----------------|--------------|--------|
| `AggFiscalModel.py` | `def pre_solve(self)` | `pre_solve` | ✅ OK |
| `EstimAggFiscalModel.py` | `def preSolve(self)` | `pre_solve` | ❌ RENAME |
| `EstimAggFiscalModel.py` | `def initializeSim(self)` | `initialize_sim` | ❌ RENAME |

#### Deliverable

A complete list of camelCase method definitions in the target codebase that must be renamed to snake_case to match HARK's expected names.

---

## 1.5 Constructor and Signature Changes (CRITICAL)

This category has historically caused the most missed upgrades.

### 1.5.1 High-Priority Classes to Check

| Class | Location | Why Important |
|-------|----------|---------------|
| `DiscreteDistribution` | distributions/ | Widely used, params renamed |
| `DiscreteDistributionLabeled` | distributions/ | Same |
| `Uniform`, `Normal`, `Lognormal` | distributions/ | Common constructors |
| `IndShockConsumerType` | ConsIndShockModel | Base agent class |
| `PerfForesightConsumerType` | ConsIndShockModel | Base agent class |
| `MarkovConsumerType` | ConsMarkovModel | Common extension |
| `Parameters` | core.py | Config patterns |

### 1.5.2 For Each Class with Changed Constructor

Document:

```markdown
### DiscreteDistribution

**Old signature** (≤{SOURCE_VERSION}):
```python
def __init__(self, pmf, X, seed=0)
```

**New signature** ({TARGET_VERSION}):
```python
def __init__(self, pmv, atoms, seed=0, limit=None)
```

**Changes**:
- `pmf` → `pmv` (keyword rename)
- `X` → `atoms` (keyword rename)
- Added: `limit` (optional, default None)

**Migration rules**:
- Keyword args: rename `pmf=` to `pmv=`, `X=` to `atoms=`
- Positional args: FLAG FOR REVIEW (order unchanged but verify semantics)
- If first arg is scalar: WARN (pmv expects array)
```

---

## 1.6 Parameter Usage Tracing (MODEL-SPECIFIC RENAMES)

### ⚠️ THIS IS COMMONLY MISSED!

Some parameter renames are "model-specific" - they only apply when parameters are used with certain model classes. Simply grepping for the parameter names is NOT sufficient. You must **trace where parameters flow**.

### 1.6.1 The Problem

Consider this scenario:
- File A defines: `params = {"aNrmInitMean": 0.5}`
- File B imports params from A and passes them to `KinkedRconsumerType(**params)`
- `KinkedRconsumerType` inherits from `IndShockConsumerType` which expects `kLogInitMean`
- File A needs the rename even though it doesn't import any model class!

### 1.6.2 Required Analysis

For EACH model-specific parameter rename:

1. **Identify which model classes require the new names**
   ```bash
   # Find which models use kLogInitMean (new name)
   grep -r "kLogInitMean" tgt/HARK --include="*.py" | grep -v "test"
   ```

2. **Find all files that define these parameters**
   ```bash
   # Find parameter definitions in target codebase
   grep -r "aNrmInitMean\|aNrmInitStd\|pLvlInitMean\|pLvlInitStd" $TARGET_DIR --include="*.py"
   ```

3. **Trace parameter flow to model instantiation**
   For each file that defines old-named parameters:
   - What does it export? (module-level dicts, function returns)
   - What imports from it?
   - Does the chain eventually reach a model constructor that needs new names?

   ```bash
   # Example: Find what imports from a params file
   grep -r "from SetupParams\|import.*SetupParams" $TARGET_DIR --include="*.py"

   # Then check what model type those files instantiate
   grep -E "Type\(|ConsumerType\(" <importing_file>
   ```

### 1.6.3 Model Inheritance Lookup

Create a lookup of which model types require which parameter names:

| Model Type | Inherits From | Parameter Style |
|------------|---------------|-----------------|
| `IndShockConsumerType` | `PerfForesightConsumerType` | NEW (`kLogInitMean`) |
| `KinkedRconsumerType` | `IndShockConsumerType` | NEW (`kLogInitMean`) |
| `PerfForesightConsumerType` | `AgentType` | NEW (`kLogInitMean`) |
| `MarkovConsumerType` | `IndShockConsumerType` | NEW (`kLogInitMean`) |
| `PortfolioConsumerType` | `IndShockConsumerType` | NEW (`kLogInitMean`) |
| `ConsWealthPortfolioType` | different base | OLD (`aNrmInitMean`) |
| `AggShockConsumerType` | different inheritance | CHECK EACH |

### 1.6.4 Deliverable

For each file with model-specific parameters, document:

| File | Parameters Found | Consumed By | Model Type | Rename Needed? |
|------|------------------|-------------|------------|----------------|
| `SetupParamsCSTW.py` | `aNrmInitMean` | `Estimation_*.py` | `KinkedRconsumerType` | YES |
| `Parameters.py` | `aNrmInitMean` | `EstimAggFiscalModel.py` | `AggFiscalType` | CHECK |

### 1.6.5 Migration Rule

```
IF file defines old parameter names (aNrmInitMean, etc.)
AND those parameters flow to a model that inherits from IndShockConsumerType
THEN rename the parameters
ELSE leave unchanged OR flag for manual review
```

**Confidence**: MEDIUM (requires tracing, may have false positives/negatives)

### 1.6.6 REQUIRED DELIVERABLE: Parameter Flow Map (to avoid “SetupParamsCSTW” misses)

The Change Inventory MUST include a **Parameter Flow Map** that explicitly connects:
- where old keys are defined
- where they are imported/copied
- the exact constructor call site that consumes them
- the consuming model class’ inheritance (whether it requires the new keys)

Minimum table (add rows for every file with `aNrmInit*`/`pLvlInit*` keys):

| Parameter-Defining File | Symbol (dict name) | Old Keys Found | Consuming File(s) | Consuming Call Site | Consuming Class | Needs Rename? | Evidence |
|-------------------------|--------------------|----------------|-------------------|---------------------|-----------------|--------------|----------|
| `SetupParamsCSTW.py` | `init_infinite` | `aNrmInitMean`, `pLvlInitStd` | `Estimation_BetaNablaSplurge.py` | `KinkedRconsumerType(**base_params)` | `KinkedRconsumerType` | YES | grep + AST trace |

**Hard rule**: it is NOT acceptable to mark these renames as “model-specific → skip everywhere”. You must either (a) prove a file needs renames via this map, or (b) explicitly justify why it does not.

---

## 1.7 Type and Semantic Changes

### 1.7.1 Scalar → List/Array Changes

| Parameter | Old Type | New Type | Context | Migration |
|-----------|----------|----------|---------|-----------|
| `Rfree` | `float` | `List[float]` | Agent params | `1.03` → `[1.03]` or `T_cycle * [1.03]` |

**⚠️ BREAKING CHANGE in HARK 0.16.1**: `Rfree` was added to `time_vary_` by default:
- **0.14.1**: `time_vary_ = ["LivPrb", "PermGroFac"]` - Rfree NOT included
- **0.16.1**: `time_vary_ = ["LivPrb", "PermGroFac", "Rfree"]` - Rfree IS included

Code like `self.Rfree[0]` is now called, which will crash if `Rfree` is a scalar:
```python
TypeError: 'float' object is not subscriptable
```

**Detection**:
```bash
grep -rE "'Rfree'\s*:\s*[0-9]|base_params\['Rfree'\]\s*=\s*[0-9]" --include="*.py" | grep -v "\["
```

**Hazard**: If `Rfree` is assigned from a variable, we can't know its type → FLAG FOR REVIEW.

### 1.7.2 NumPy RNG API Changes

| Old | New | Context |
|-----|-----|---------|
| `RNG.randint(0, N)` | `RNG.integers(N)` | RandomState → Generator |
| `RNG.random_integers(a, b)` | `RNG.integers(a, b+1)` | Deprecated |

### 1.7.3 Property ↔ Method Transitions

Check if any `obj.thing()` became `obj.thing` or vice versa.

```bash
# Look for @property additions/removals
diff <(grep -r "@property" src/HARK) <(grep -r "@property" tgt/HARK)
```

### 1.7.4 NOTE: Model-specific parameter renames are handled in Section 1.6

Do not “re-discover” them here; instead ensure the **Parameter Flow Map** exists and is complete.

## 1.7 Exception and Error Changes

```bash
# Find exception class definitions
grep -rh "class.*Exception\|class.*Error" src/HARK --include="*.py"
grep -rh "class.*Exception\|class.*Error" tgt/HARK --include="*.py"
```

Record any renamed or removed exceptions.

---


## 1.7b Default Value Changes (SILENT BREAKING CHANGES)

⚠️ **These are the hardest to detect** because the API appears unchanged but behavior differs.

### Why This Matters

If a parameter's default value changes:
```python
# OLD (0.14.1)
def solve(self, verbose=False):  # Default: quiet

# NEW (0.17.0)
def solve(self, verbose=True):   # Default: noisy
```

Code calling `agent.solve()` without explicit `verbose=` will behave differently.

### Discovery Commands

```bash
# Extract function signatures with defaults from both versions
grep -rh "def.*=.*:" src/HARK --include="*.py" | sed 's/.*def /def /' > sigs_src.txt
grep -rh "def.*=.*:" tgt/HARK --include="*.py" | sed 's/.*def /def /' > sigs_tgt.txt

# Diff to find changed defaults
diff sigs_src.txt sigs_tgt.txt | grep -E "^<|^>" | head -50
```

### High-Priority Parameters to Check

| Class/Function | Parameter | Why Important |
|----------------|-----------|---------------|
| `AgentType.solve()` | `verbose` | Changes logging behavior |
| `DiscreteDistribution()` | `seed` | Changes reproducibility |
| Simulation methods | `T_sim` | Changes simulation length |

### Record in Change Inventory:

| Function | Parameter | Old Default | New Default | Impact |
|----------|-----------|-------------|-------------|--------|
| (fill in) | (fill in) | (fill in) | (fill in) | (fill in) |

---

## 1.7c Return Type and Behavioral Changes (SILENT BUGS)

### Return Type Changes

Methods may return different types even with same signature:
```python
# OLD: Returns numpy array
def get_distribution(self):
    return np.array([...])

# NEW: Returns DiscreteDistribution object
def get_distribution(self):
    return DiscreteDistribution([...])
```

### Discovery Commands

```bash
# Look for return statements in key functions
for func in solve get_shocks get_controls; do
    echo "=== $func ==="
    grep -A 20 "def $func" src/HARK/core.py | grep "return" | head -3
    echo "---"
    grep -A 20 "def $func" tgt/HARK/core.py | grep "return" | head -3
done
```

### Behavioral Changes (Same Signature, Different Logic)

These are nearly impossible to detect automatically. **Mitigations**:

1. Read CHANGELOG/release notes between versions
2. Run test suite and look for different outputs (not just pass/fail)
3. Compare example notebook outputs between versions

---

## 1.7d Serialization and State Compatibility

### Why This Matters

If users have:
- Pickled agent objects from the old HARK version
- Saved simulation states
- JSON/YAML config files with HARK object references

These may not load correctly with the new version.

### Discovery Commands

```bash
# Check for __getstate__/__setstate__ changes (pickle compatibility)
diff <(grep -rn "__getstate__\|__setstate__\|__reduce__" src/HARK) \
     <(grep -rn "__getstate__\|__setstate__\|__reduce__" tgt/HARK)

# Check for class attribute renames that affect serialization
grep -rn "self\.[a-z].*=" src/HARK/core.py | head -30
grep -rn "self\.[a-z].*=" tgt/HARK/core.py | head -30
```

### Document in Change Inventory:

- Classes with changed `__init__` signatures (pickles will fail to load)
- Renamed instance attributes (old pickles have wrong attribute names)
- Removed classes (pickles reference non-existent classes)

---

## 1.7e Dynamic/Reflection-Based Access (HARD TO DETECT)

### Why This Matters

Code using `getattr()`, `exec()`, `eval()`, or string-based method dispatch will fail silently:

```python
# This pattern breaks if 'initializeSim' was renamed to 'initialize_sim'
method_name = 'initializeSim'
getattr(agent, method_name)()  # AttributeError at runtime!

# Or in HARK's multi_thread_commands:
multi_thread_commands(agents, ['solve', 'initializeSim'])  # Fails!
```

### Discovery Commands (TARGET CODEBASE)

```bash
# Find getattr/setattr/hasattr calls
grep -rn "getattr\|setattr\|hasattr" $TARGET_DIR --include="*.py"

# Find string lists that might be method names
grep -rn "\[.*'[a-z_]*'.*\]" $TARGET_DIR --include="*.py" | grep -i "command\|method\|func"

# Find multi_thread_commands usage
grep -rn "multi_thread_commands\|multiThreadCommands" $TARGET_DIR --include="*.py"
```

### Migration Rule

For each string-based method reference found:
1. Check if the method name was renamed
2. Update the string to the new name
3. If the string comes from config/user input, add validation

---

## 1.7f Test-Driven Verification (REQUIRED VALIDATION)

### Why This Matters

**The best way to validate a migration is to run the test suite.** Tests that pass on the old version but fail on the new version indicate breaking changes.

### Required Steps

1. **Run target codebase tests with OLD HARK**:
   ```bash
   pip install econ-ark=={SOURCE_VERSION}
   pytest --tb=short > tests_old.log 2>&1
   ```

2. **Run target codebase tests with NEW HARK** (after migration):
   ```bash
   pip install econ-ark=={TARGET_VERSION}
   pytest --tb=short > tests_new.log 2>&1
   ```

3. **Compare results**:
   ```bash
   diff tests_old.log tests_new.log
   ```

4. **Investigate any tests that**:
   - Pass with old HARK but fail with new → Breaking change not handled
   - Fail with both → Pre-existing issue (not migration-related)
   - Pass with new but failed with old → Migration fixed an issue

### If Tests Don't Exist

Create minimal smoke tests:
```python
# test_migration_smoke.py
def test_import_succeeds():
    from HARK.distributions import DiscreteDistribution
    from HARK.core import AgentType

def test_basic_agent_creation():
    from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
    agent = IndShockConsumerType()
    agent.solve()

def test_simulation_runs():
    agent = IndShockConsumerType()
    agent.solve()
    agent.initialize_sim()
    agent.simulate()
```

## 1.8 Legacy Pattern Archaeology

User code may use patterns from versions OLDER than `{SOURCE_VERSION}` due to backward-compatibility shims.

### 1.8.1 Check for Removed Aliases

```bash
# Look for alias assignments in source that don't exist in target
grep -rh "^[A-Za-z_]* = " src/HARK --include="*.py" | grep -v "def \|class " > aliases_src.txt
grep -rh "^[A-Za-z_]* = " tgt/HARK --include="*.py" | grep -v "def \|class " > aliases_tgt.txt
diff aliases_src.txt aliases_tgt.txt
```

### 1.8.2 Known Legacy Patterns (Pre-0.11)

If upgrading from a version that may have had backward-compat shims, also scan for:

| Legacy Pattern | Removed In | Current Form |
|----------------|------------|--------------|
| `from HARK.distribution import` | 0.16 | `from HARK.distributions import` |
| `IncomeDstn` | 0.15? | `IncShkDstn` |
| Various camelCase methods | 0.15+ | snake_case |

---

## 1.9 Evidence from Examples/Tests/DemARKs

The official HARK repo's own examples are **ground truth** for what changed.

```bash
# Diff examples between versions
git diff {SOURCE_VERSION}..{TARGET_VERSION} -- examples/
git diff {SOURCE_VERSION}..{TARGET_VERSION} -- HARK/tests/

# If DemARK repo is available
git diff {SOURCE_VERSION}..{TARGET_VERSION} -- notebooks/
```

**Extract concrete before→after snippets** that the migration tool must reproduce.

---

## 1.10 Project Configuration Files (Dependency Version Pins)

**This is commonly missed!** Any file in the target codebase that pins the HARK version must be updated.

**Required scan of TARGET CODEBASE**:

```bash
# Find all files that might contain HARK version pins
find $TARGET_DIR -type f \( \
    -name "requirements*.txt" -o \
    -name "pyproject.toml" -o \
    -name "setup.py" -o \
    -name "setup.cfg" -o \
    -name "environment*.yml" -o \
    -name "environment*.yaml" -o \
    -name "Pipfile" -o \
    -name "*.toml" \
\) 2>/dev/null | xargs grep -l -i "econ-ark\|hark" 2>/dev/null

# Also check binder/ directory specifically (common for reproducible environments)
find $TARGET_DIR -path "*/binder/*" -type f 2>/dev/null | xargs grep -l -i "econ-ark\|hark" 2>/dev/null
```

**Record in Change Inventory**:

| File | Field/Line | Current Version | Target Version |
|------|------------|-----------------|----------------|
| `binder/requirements.txt` | `econ-ark==X.Y.Z` | `{SOURCE_VERSION}` | `{TARGET_VERSION}` |
| `pyproject.toml` | `dependencies` | ... | ... |

**Migration Rule**:
- Pattern: `econ-ark[=<>~!]*{SOURCE_VERSION_PATTERN}`
- Replace with: `econ-ark=={TARGET_VERSION}` (or appropriate specifier for ranges)
- Confidence: HIGH (simple string replacement in known file types)

**Why this matters**: If config files still specify the old HARK version, the codebase will install the wrong version and the migration will appear to have failed even though the code transformations are correct.

---

## 1.12 Change Impact Analysis (Blast Radius)

**Goal**: Identify "Hot Modules" and high-risk areas where breaking changes will have the largest impact.

```bash
# Find modules with high incoming dependency counts
grep -rh "^from HARK" $TARGET_DIR --include="*.py" | cut -d' ' -f2 | cut -d'.' -f1-3 | sort | uniq -c | sort -nr | head -10
```

**Record in Change Inventory**:
- List top 5 most-imported modules in the target codebase.
- Mark any module in the "Renamed" or "Removed" list that is also a "Hot Module" as **HIGH RISK**.

## 1.13 Semantic Baseline Generation (SCALABLE / FAST)

**Goal**: Record a mathematical "Fingerprint" of the logic before migration. This is NOT a full 5-day reproduction; it is a **fast sanity check** (target: < 5 mins).

### Performance Triage (Avoid the 5-Day Run)

If the main reproduction script is slow, **DO NOT run it fully**. Instead, create a "Mini-Reproduction":
1. **Scale down Agents**: Set `AgentCount = 100` (or 1% of baseline).
2. **Shorten Time**: Set `T_sim = 5` (or a small number of periods).
3. **Fixed Iterations**: Disable convergence criteria; run for 3-5 iterations.
4. **Deterministic Seed**: Use `seed=42` or similar.

### Procedure

1. **Record output of the Mini-Reproduction**:
   ```bash
   # Example: run a scaled-down simulation
   python reproduce_lite.py --seed 42 > baseline_lite.txt
   ```

**Record in Change Inventory**:
- The scaling factors used (e.g., "Scaled to 100 agents, 5 periods").
- The baseline hash/value from the lite run.

---

## 1.11 Compile the Change Inventory Document

After completing all discovery steps, produce a single **Change Inventory** document:

```markdown
# Change Inventory: HARK {SOURCE_VERSION} → {TARGET_VERSION}

## Discovery Metadata
- Generated: {date}
- Source version: {SOURCE_VERSION}
- Target version: {TARGET_VERSION}
- Discovery method: [PyPI extraction / Git diff]
- Modules skipped (import errors): [list]

## Summary Statistics
- Module renames: N
- Class/function moves: N
- Method renames: N
- Constructor signature changes: N
- Type changes: N
- Total breaking changes: N

## Detailed Changes

### 1. Module/Import Changes
[Table with evidence]

### 2. Method Renames
[Complete mapping table with contexts]

### 3. Attribute/Property Renames (DISTINCT FROM METHODS)
[Table of attribute access patterns that changed - NO parentheses]

| Old Attribute | New Attribute | On Class(es) | Search Pattern |
|--------------|---------------|--------------|----------------|
| `.pmf` | `.pmv` | DiscreteDistribution | `\.pmf\b` |
| `.X` | `.atoms` | DiscreteDistribution | `\.X\b` |
| `.IncomeDstn` | `.IncShkDstn` | Agent types | `\.IncomeDstn` |

### 4. Constructor Signature Changes
[Per-class documentation with old/new signatures]

### 5. Type/Semantic Changes
[Table with migration rules]

### 6. Dependency API Changes
[NumPy, etc.]

### 7. Project Config Files (Version Pins)
[Table of files that pin HARK version and need updating]

| File | Current | Target |
|------|---------|--------|
| `binder/requirements.txt` | `econ-ark==0.14.1` | `econ-ark==0.16.1` |

### 8. Legacy Patterns to Scan For
[If applicable]

## Confidence Levels
- HIGH: Can auto-transform safely
- MEDIUM: Can transform but verify
- LOW: Flag for manual review

## Known Ambiguities
[Patterns that look like they might need changes but we're not sure]
```

---

## STAGE 1 OUTPUT

**Deliverable**: The Change Inventory document (Markdown + optional JSON).

**Next step**: Human reviews the Change Inventory, then chooses Stage 2A or 2B.


---

# ═══════════════════════════════════════════════════════════════════
# ⛔ STAGE 1 COMPLETION GATE (MANDATORY)
# ═══════════════════════════════════════════════════════════════════

## YOU MUST NOT PROCEED TO STAGE 2 UNTIL ALL OF THE FOLLOWING ARE TRUE:

### 1.A Required Discovery Commands Executed

You must have ACTUALLY RUN (not just read about) these commands and recorded their output:

| Command | Purpose | Output Location |
|---------|---------|-----------------|
| `diff -rq src/HARK tgt/HARK` | Structural changes | Section 1.1 of Inventory |
| `comm -23 methods_src.txt methods_tgt.txt` | Method renames | Section 1.4.1 of Inventory |
| `grep -rh "def [a-z]" src/HARK \| ...` | Full method list | methods_src.txt |
| `grep -rh "def [a-z]" tgt/HARK \| ...` | Full method list | methods_tgt.txt |
| API extraction script on BOTH versions | Signature changes | api_src.json, api_tgt.json |

**Checkpoint**: ☐ I have run ALL commands above and have the output files.

### 1.B Change Inventory Completeness Check

The Change Inventory document MUST have:

- ☐ **NO "TBD" entries** - Every field must have an actual value or "N/A"
- ☐ **Actual counts** in Summary Statistics (not placeholders)
- ☐ **Evidence column filled** for every change (grep output, diff line, API diff)
- ☐ **Complete method rename table** with ALL camelCase→snake_case mappings found
- ☐ **Module removal/move table** listing ALL structural changes from `diff -rq`

### 1.C Specific Items That MUST Be Documented (Common Misses)

Before proceeding, verify you have explicitly addressed:

| Item | Check | Where Documented |
|------|-------|------------------|
| `HARK.parallel` module status | ☐ Does it exist in target? Where did its functions move? | Section 1.1 |
| `RNG.randint()` vs `RNG.integers()` | ☐ Which does target HARK use? | Section 1.7.2 |
| `.drawDiscrete()` method | ☐ What is it renamed to? | Section 1.4.1 |
| ALL camelCase methods in src | ☐ Listed and mapped to snake_case equivalents | Section 1.4.1 |
| `DiscreteDistribution` constructor | ☐ Full signature diff documented | Section 1.5.2 |
| Method REMOVALS (not renames) | ☐ Methods that exist in source but NOT in target | Section 1.4.4 |
| Method→Function conversions | ☐ Instance methods converted to standalone functions | Section 1.4.4 |
| Target codebase camelCase methods | ☐ ALL camelCase method defs that should be snake_case | Section 1.4.5 |
| Default value changes | ☐ Parameters with changed defaults documented | Section 1.7b |
| Serialization compatibility | ☐ Pickle/state file impacts identified | Section 1.7d |
| Dynamic method references | ☐ getattr/string-based method calls checked | Section 1.7e |
| Test-driven verification | ☐ Tests run with both HARK versions | Section 1.7f |
| Change Impact Analysis | ☐ Blast Radius Map of high-risk modules | Section 1.12 |
| Semantic Baseline | ☐ Lite/Fast sanity check recorded before migration | Section 1.13 |
| Stage 0 pre-flight | ☐ Clean working tree + correct base branch + recorded SHA | Stage 0 |
| Case-only rename hazards | ☐ Checked for case-only renames + documented handling | Section 1.1b |
| Docs/changelog impacts | ☐ Docs/README/CHANGELOG updates identified if needed | Stage 2 Safety (2.S5) |
| Dry-run + idempotence plan | ☐ Transformation approach supports dry-run + idempotence | Stage 2 Safety (2.S1-2.S2) |
| CI/acceptance gates | ☐ Acceptance criteria defined (CI green, grep clean, tests) | Stage 2 Safety (2.S4) |

### 1.D Verification Test

Pick 3 Python files from the TARGET CODEBASE that you believe need changes. For each:

1. List what changes you expect based on the Inventory
2. Run `grep` for old patterns
3. Confirm the Inventory would catch them all

If ANY expected change is NOT in the Inventory, **STOP AND FIX THE INVENTORY**.

### 1.E Hard Gate Statement

**I certify that:**
- ☐ All discovery commands in Section 1.0-1.10 have been executed
- ☐ The Change Inventory has NO TBD/placeholder entries
- ☐ Method rename table is COMPLETE (not a sample)
- ☐ Method REMOVAL table is COMPLETE (methods removed from HARK, not renamed)
- ☐ Target codebase camelCase methods identified (even in "dead" code)
- ☐ Default value changes checked
- ☐ Dynamic method references (getattr, string commands) checked
- ☐ Test suite run with both HARK versions (or smoke tests created)
- ☐ Stage 0 pre-flight completed (clean tree, correct base branch, recorded SHA)
- ☐ Case-only rename hazards checked (and plan documented if relevant)
- ☐ Documentation/CHANGELOG impact assessed and plan recorded
- ☐ Stage 2 plan supports dry-run + idempotence (or manual-only declared)
- ☐ Stage 2 acceptance criteria defined (CI green, greps clean, tests)
- ☐ Semantic Baseline (Lite/Fast) recorded and hash verified
- ☐ High-risk "Hot Modules" identified for manual review
- ☐ Semantic Signature Comparison (2.V1) passed (post-migration results match baseline)
- ☐ I have verified with 3 test files that the Inventory is comprehensive

**⛔ IF ANY BOX ABOVE IS UNCHECKED, DO NOT PROCEED TO STAGE 2.**


---

# ═══════════════════════════════════════════════════════════════════
# STAGE 2: TRANSFORMATION
# ═══════════════════════════════════════════════════════════════════

After the Change Inventory is reviewed and approved, choose ONE of:

- **Stage 2A**: Human Migration Checklist (manual approach)
- **Stage 2B**: Automated Migration Script (programmatic approach)

Both take the Change Inventory as input.

## Stage 2 Safety and Reproducibility Requirements (MANDATORY)

These requirements are standard best practice for migrations/codemods.

### 2.S1 Dry-run first

Before editing files, run a **dry-run** that only reports what *would* change:
- files that would be touched
- counts per change category
- a preview diff (or patch output)

If you cannot produce a dry-run, you MUST clearly label the upgrade as **manual-only** and increase review requirements.

### 2.S2 Idempotence

The transformation MUST be **idempotent**:
- Running it once applies changes
- Running it a second time produces **no further diffs**

Verification:
```bash
# Run migration twice; second run should yield no changes
# (exact command depends on Stage 2B implementation)
git diff --stat
```

### 2.S3 Atomic commits and rollback plan

- Prefer **atomic commits** by category (imports, API renames, config pins, etc.)
- Provide a **rollback plan**:
  - Either: one revertable commit
  - Or: documented sequence of `git revert <sha>` for each commit

### 2.S4 CI and acceptance gates

A migration PR must meet these acceptance criteria:
- CI is green
- Test suite passes on supported platforms (at minimum Linux case-sensitive)
- Old-pattern greps return empty (imports, removed symbols, camelCase lifecycle methods)
- If automation is used: dry-run report is attached and idempotence verified

### 2.S5 Documentation requirements

If any public API/config/CLI behavior changes:
- Update migration notes / README / docs
- Add a CHANGELOG or UPGRADE note listing breaking changes and before/after snippets

### 2.S6 Backward compatibility and deprecation strategy

Best practice: decide whether to provide temporary shims for renamed/removed APIs.

Choose ONE:
- **Hard break**: remove old names immediately; document in migration guide; bump major version if applicable.
- **Shim + deprecate**: keep old names as wrappers that call the new implementation and emit `DeprecationWarning`, with a documented removal timeline.

If the TARGET CODEBASE is a library used by downstream code, prefer **Shim + deprecate** unless there is a strong reason not to.

Example shim pattern:
```python
import warnings

def oldName(*args, **kwargs):
    warnings.warn("oldName is deprecated; use new_name", DeprecationWarning, stacklevel=2)
    return new_name(*args, **kwargs)
```

### 2.S7 Change Attribution (Audit Trail)

**Best practice**: Every automated change must be attributable to a specific rule in the Change Inventory.

- Migration scripts MUST log every change with a reference ID.
- Example log format: `[CI-1.2] Renamed drawDiscrete to draw_events in file_a.py:45`

### 2.S8 AST-Based Precision (Optional but Recommended)

For high-noise symbols (e.g., `X`, `pmf`), simple regex might be too dangerous.

**Recommended approach**:
- Use `libcst` or `bowler` to find only those occurrences that are attributes of a HARK class.
- If using regex, use "Strict Context" (e.g., `grep -B 5 "DiscreteDistribution" | grep ".pmf"`) to confirm the object type.

---

---

# STAGE 2A: Human Migration Checklist

## Purpose

Generate a step-by-step checklist that a human can follow to manually upgrade their codebase. Best for:
- Small codebases (<20 files)
- Learning how HARK changed
- High-risk codebases where automation feels dangerous
- One-time upgrades not worth scripting

## Output Format

```markdown
# Migration Checklist: HARK {SOURCE_VERSION} → {TARGET_VERSION}

## Prerequisites
- [ ] Back up your codebase
- [ ] Create a new branch: `git checkout -b hark-upgrade`
- [ ] Install target HARK version in a test environment

## Step 1: Update Imports

### 1.1 Module renames
Search for: `from HARK.distribution import`
Replace with: `from HARK.distributions import`
Files to check: [use `grep -r` to find]

### 1.2 Import aliases needed
Search for: `from HARK.ConsumptionSaving.ConsAggShockModel import MargValueFunc2D`
Replace with: `from HARK.ConsumptionSaving.ConsAggShockModel import MargValueFuncCRRA as MargValueFunc2D`

## Step 2: Rename Methods

### 2.1 initializeSim → initialize_sim
Search patterns:
- `def initializeSim`
- `self.initializeSim`
- `.initializeSim(`
- `'initializeSim'`

[Repeat for each method rename]

## Step 3: Update Constructor Calls

### 3.1 DiscreteDistribution
Search for: `DiscreteDistribution(`
For each occurrence:
- [ ] If using `pmf=`, change to `pmv=`
- [ ] If using `X=`, change to `atoms=`
- [ ] If using positional args, verify first arg is array (not scalar)

## Step 4: Fix Type Changes

### 4.1 Rfree scalar → list
Search for: `Rfree` assignments
For each:
- [ ] If `Rfree = 1.03`, change to `Rfree = [1.03]`
- [ ] If `Rfree = variable`, verify variable is a list

## Step 5: Validate

- [ ] Run `python -m py_compile yourfile.py` on each modified file
- [ ] Run your test suite
- [ ] Import your main modules and verify no errors

## Step 6: Review Warnings

These patterns need manual judgment:
- [ ] [List from Change Inventory ambiguities]
```

---

# STAGE 2B: Automated Migration Script

## Purpose

Generate a Python tool that programmatically transforms codebases. Best for:
- Large codebases (>20 files)
- Repeated use across multiple projects
- CI/CD integration
- Generating detailed reports

## Tool Requirements

### 2B.1 Interface

```bash
# Dry run (default) - show what would change
python -m hark_migrate --source-version 0.14.1 --target-version 0.16.1 \
    --target-dir /path/to/codebase --dry-run

# Apply changes
python -m hark_migrate ... --apply

# Generate report only
python -m hark_migrate ... --report-only --output migration_report.md

# Parameter rename modes (MODEL-SPECIFIC; must not miss flow-based cases like SetupParamsCSTW.py)
# - off: never rename model-specific params; only warn
# - auto: rename ONLY when flow-tracing shows dict kwargs feed an IndShock-family constructor (recommended default)
# - on: force rename everywhere (dangerous; generally not recommended)
python -m hark_migrate ... --param-renames auto

# Include config files (requirements/environment/pyproject/binder) that pin econ-ark/HARK versions
python -m hark_migrate ... --include-config-files
```

### 2B.2 Architecture

```
hark_migrate/
├── __init__.py
├── __main__.py          # Entry point
├── cli.py               # Argument parsing
├── inventory.py         # Load Change Inventory
├── scanner.py           # Find code + config files (py/ipynb + requirements/environment/pyproject/binder)
├── transformers/
│   ├── __init__.py
│   ├── base.py          # Transformer interface
│   ├── imports.py       # Import rewrites
│   ├── methods.py       # Method renames (with parentheses)
│   ├── attributes.py    # Attribute/property renames (WITHOUT parentheses: .pmf→.pmv, .IncomeDstn→.IncShkDstn)
│   ├── constructors.py  # Constructor migrations
│   ├── types.py         # Type changes (Rfree, etc.)
│   ├── params_flow.py   # Context-sensitive param renames via flow tracing (aNrmInit*↔kLogInit*)
│   ├── config_files.py  # Update version pins in requirements/environment/pyproject/binder
│   └── api.py           # Other API changes
├── validators.py        # AST/JSON validation
├── reporters.py         # Report generation
└── notebook.py          # Notebook-safe editing
```

### 2B.3 Transformer Contract

Each transformer must:

```python
class Transformer(ABC):
    name: str                    # e.g., "method_rename"
    confidence: str              # "high", "medium", "low"

    @abstractmethod
    def should_transform(self, line: str, context: Context) -> bool:
        """Return True if this transformer applies."""

    @abstractmethod
    def transform(self, line: str, context: Context) -> TransformResult:
        """Return transformed line + metadata."""

    @abstractmethod
    def get_warning(self, line: str, context: Context) -> Optional[str]:
        """Return warning message if transform is risky."""
```

### 2B.4 Editing Strategy

**For `.py` files**:
- **Preferred**: CST-based editing (LibCST) to preserve formatting
- **Acceptable**: Line-based regex for trivial changes (imports only)
- **Required**: Validate with `ast.parse()` after every file edit

**For `.ipynb` files**:
- Parse with `json.load()` or `nbformat.read()`
- Edit ONLY `cell["source"]` for code cells
- Preserve all metadata, outputs, cell IDs
- Validate JSON structure after editing
- Optionally validate with `nbformat.validate()`

### 2B.5 Safety Requirements

1. **Idempotence**: Running twice produces same result (no double-transforms)
2. **Atomicity**: Either all changes to a file succeed, or none do
3. **Validation**: Every edited file must pass syntax check before writing
4. **Backup**: Optionally create `.bak` files before modifying

### 2B.5.0 CRITICAL: Blocking vs Non-Blocking Issues

The migration tool MUST distinguish between:

| Category | Behavior | Examples |
|----------|----------|----------|
| **BLOCKING** | Abort/require confirmation | `DiscreteDistribution` with scalar first arg (pmv expects array) |
| **HIGH-PRIORITY WARNING** | Proceed but require manual confirmation | Positional args in changed constructors |
| **INFORMATIONAL** | Log only | Already-correct patterns |

**Hard rule**: The following MUST be BLOCKING (not mere warnings):

1. **Constructor type mismatches**: `DiscreteDistribution(scalar, ...)` where first arg should be array
2. **Old keyword parameter names**: `DiscreteDistribution(pmf=..., X=...)`
3. **Deprecated API calls with no direct replacement**

The report MUST have a section titled **"⛔ BLOCKING ISSUES (must fix manually)"** that:
- Is shown BEFORE the PR can be considered complete
- Lists ALL blocking issues (not truncated)
- Requires explicit `--force` flag to proceed if any blocking issues exist

**Why this matters**: A warning that scrolls off-screen or is truncated to "10 of 47" effectively doesn't exist. Critical issues must BLOCK, not warn.

### 2B.5.1 REQUIRED: Flow-based Parameter Renames (AUTO mode)

The tool MUST implement a safe default behavior for model-specific parameter renames:

- **Goal**: apply `aNrmInit*→kLogInit*` and `pLvlInit*→pLogInit*` ONLY when the parameter dict is proven (by heuristics) to be passed as `**kwargs` into a constructor/class that requires the new keys (e.g., `KinkedRconsumerType`, `IndShockConsumerType` family).
- **Rationale**: this avoids both failure modes:
  - “skip everywhere” (misses `SetupParamsCSTW.py`-style cases)
  - “rename everywhere” (breaks models that still expect old keys, e.g. wealth-portfolio calibration files)

**Minimum heuristics required** (implement at least these):
1. Detect dict literals / assignments containing old keys (e.g., `{'aNrmInitMean': ...}`).
2. Detect `**dict_var` expansion into calls like `KinkedRconsumerType(**dict_var)` or `SomeType(**params)`.
3. Detect simple import flow:
   - `from SetupParamsCSTW import init_infinite`
   - then later `base_params = init_infinite.copy()` (or update/merge)
   - then `KinkedRconsumerType(**base_params)`
4. If (1)+(2) are satisfied but the consuming class is unknown, WARN (do not rename).

**Reporting requirement**: the report must include the Parameter Flow Map rows it used to justify each rename.

### 2B.6 Reporting

Generate both JSON (machine-readable) and Markdown (human-readable) reports:

```markdown
# Migration Report: {TARGET_DIR}

## Summary
- Files scanned: 45
- Files modified: 12
- Changes applied: 87
- Warnings: 6
- Errors: 0

## Changes by Category
| Category | Count | Confidence |
|----------|-------|------------|
| import_rename | 23 | high |
| method_rename | 45 | high |
| constructor_fix | 8 | medium |
| type_change | 11 | medium |

## Warnings (Manual Review Needed)
- `models/agent.py:142`: DiscreteDistribution uses positional args - verify semantics
- `utils/helpers.py:89`: Rfree assigned from variable - verify it's a list

## Per-File Details
[...]
```

### 2B.7 Coverage Verification

The tool MUST include a final verification step:

```python
def verify_coverage(target_dir: str, inventory: ChangeInventory) -> CoverageReport:
    """Verify no old patterns remain."""
    remaining = {}

    for old_pattern in inventory.get_all_old_patterns():
        matches = grep_for_pattern(target_dir, old_pattern)
        if matches:
            remaining[old_pattern] = matches

    return CoverageReport(
        old_patterns_checked=len(inventory.get_all_old_patterns()),
        remaining_occurrences=remaining,
        is_complete=(len(remaining) == 0)
    )
```

Output:
```markdown
## Coverage Verification

Patterns checked: 47
Remaining occurrences: 2

### Remaining (require justification or manual fix)
- `initializeSim`: 1 occurrence in `legacy/old_code.py:23` (in comment, OK to ignore)
- `aNrmInitMean`: 1 occurrence in `config/defaults.py:45` (intentional for old HARK compat)

### Config File Version Pins Checked
- `binder/requirements.txt`: ✅ Updated to `econ-ark=={TARGET_VERSION}`
- `pyproject.toml`: Not present
- `environment.yml`: Not present
```

---

## Common Failure Modes (Guard Against These)

The generated tool must explicitly handle:

1. **Import aliases**: `from X import Y as Z` - if Y is renamed, grep for Y won't find uses of Z
2. **String commands**: `['initializeSim()']` - method names in strings
3. **`**kwargs` pass-through**: Wrapper functions forwarding renamed params
4. **Conditional imports**: `try: from X import Y except: from Z import Y`
5. **Dynamic attribute access**: `getattr(obj, 'oldMethod')`
6. **Docstrings/comments**: May contain old names (transform or leave?)
7. **Notebook outputs**: Don't edit outputs, only source
8. **Model-specific renames**: Some renames only apply to certain model subclasses
9. **Project config files forgotten**: `requirements.txt`, `pyproject.toml`, `binder/`, `environment.yml` still pin old HARK version — code migration succeeds but wrong HARK version gets installed
10. **Attribute vs method confusion**: `obj.pmf` (attribute access) is NOT caught by method-rename patterns that look for `obj.pmf()` (with parentheses). Attributes like `.pmf→.pmv`, `.X→.atoms`, `.IncomeDstn→.IncShkDstn` require separate detection.
11. **Warnings that don't block**: A detector that generates "warnings" but doesn't BLOCK migration is useless if the warnings scroll off-screen, get truncated ("10 of 47..."), or are buried in verbose output. **Critical issues like type mismatches in constructor args MUST be BLOCKING, not informational warnings.**
12. **Constructor positional arg type mismatches**: `DiscreteDistribution(scalar, ...)` where `pmv` expects an array. The detector might flag this but if it's just a warning, it gets ignored. Must be BLOCKING.

---

## Output of Stage 2B

**Deliverable**: A complete migration tool (Python package) that:
- Takes the Change Inventory as configuration
- Transforms `.py` and `.ipynb` files
- Produces detailed reports
- Validates all changes
- Verifies coverage

---

# APPENDIX: Quick Reference

## When to Use Stage 2A (Checklist) vs 2B (Script)

| Factor | 2A: Checklist | 2B: Script |
|--------|---------------|------------|
| Codebase size | <20 files | >20 files |
| Upgrade frequency | One-time | Repeated |
| Risk tolerance | Low (want control) | Medium-high |
| Time available | More | Less |
| Technical comfort | Any | Comfortable with automation |
| Need for reports | No | Yes |

## Minimal Discovery for Small Upgrades

If upgrading between minor versions (e.g., 0.16.0 → 0.16.1), you may skip some discovery steps. But ALWAYS do:
- Structural diff (1.1)
- API surface extraction for changed modules (1.2)
- Check release notes

## Testing the Change Inventory

Before using the inventory for transformation:
1. Pick 2-3 files you KNOW need changes
2. Verify the inventory identifies those changes

# STAGE 2.V: VERIFICATION

## 2.V1 Semantic Signature Comparison (FINAL VALIDATION)

**Goal**: Prove that the migration did not break the logic using the **Mini-Reproduction** from Section 1.13.

### Procedure

1. **Rerun the LITE baseline** established in Section 1.13 using the NEW HARK version:
   ```bash
   python reproduce_lite.py --seed 42 > post_migration_lite.txt
   ```
2. **Compare results**:
   ```bash
   diff baseline_lite.txt post_migration_lite.txt
   ```

**Requirement**: The outputs must be identical (bit-for-bit) or within mathematical tolerance. This proves the **machinery** still produces the same results, even if we haven't run the full 5-day simulation.

**If results differ**:
- Investigate behavioral changes (1.7c) or default value changes (1.7b).
- Do NOT merge until the difference is explained.

---

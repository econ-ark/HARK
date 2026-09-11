# Meta-Prompt: Generate HARK Version Upgrade Script

## Purpose

This is a **meta-prompt** that generates a version-specific upgrade prompt. When you provide a source version and target version, this prompt will guide you to:

1. **Discover** all breaking changes between the versions by analyzing actual code
2. **Generate** a detailed, version-specific prompt documenting those changes
3. That generated prompt can then be used to **create** the actual migration script

---

## CRITICAL: Start with CHANGELOG, Verify with Diffs

**FIRST**, read `docs/CHANGELOG.md` in the HARK repository. This file lists all breaking changes for each version in a structured format. It is the authoritative source for intentional API changes.

**THEN**, verify and supplement with actual source diffs. The CHANGELOG may not capture every behavioral change, and your target code may use undocumented patterns.

**Do NOT rely solely on CHANGELOG or release notes** - always verify with actual code diffs to discover changes that may not have been documented.

**Use one of these accessible diff methods:**

### Option A: PyPI Package Diff (RECOMMENDED - No Git Required)

```bash
# Download both versions directly from PyPI
pip download econ-ark=={SOURCE_VERSION} econ-ark=={TARGET_VERSION} --no-deps -d /tmp/hark_diff

# Extract packages
cd /tmp/hark_diff
unzip -q econ_ark-{SOURCE_VERSION}-py2.py3-none-any.whl -d source_version
unzip -q econ_ark-{TARGET_VERSION}-py2.py3-none-any.whl -d target_version

# STRUCTURAL DIFF: What files changed?
diff -rq source_version/HARK target_version/HARK

# FILES ONLY IN SOURCE (removed in target)
diff -rq source_version/HARK target_version/HARK | grep "Only in source_version"

# FILES ONLY IN TARGET (added)
diff -rq source_version/HARK target_version/HARK | grep "Only in target_version"

# DETAILED FILE DIFFS
diff source_version/HARK/core.py target_version/HARK/core.py
diff source_version/HARK/utilities.py target_version/HARK/utilities.py
# ... etc for each changed file
```

### Option B: Git Diff (Requires Cloned Repo with Tags)

```bash
git clone https://github.com/econ-ark/HARK.git
cd HARK
git diff {SOURCE_VERSION}..{TARGET_VERSION} -- 'HARK/*.py'
```

### VERIFICATION CHECKPOINT ⚠️

**Before proceeding to Phase 2, you MUST confirm:**

1. ☐ I have successfully downloaded/accessed both HARK versions
2. ☐ I have run the structural diff (`diff -rq`) and recorded which files were added/removed/modified
3. ☐ I have run detailed diffs on at least these key files:
   - `core.py`
   - `utilities.py`
   - `distribution.py` or `distributions/__init__.py`
   - `ConsumptionSaving/ConsIndShockModel.py`
   - `ConsumptionSaving/ConsMarkovModel.py`
   - `ConsumptionSaving/ConsAggShockModel.py`

**If you cannot confirm all three checkpoints, STOP and explain what blocked you.**

---

## Phase 1: Systematic Discovery

Discovery MUST proceed in this order. Complete each category before moving to the next.

### 1.1 STRUCTURAL CHANGES (Do First!)

**Question**: What files/directories were added, removed, or reorganized?

```bash
# List ALL structural differences
diff -rq source_version/HARK target_version/HARK 2>/dev/null | sort
```

**Record in your notes:**
- Files removed (exist only in source)
- Files added (exist only in target)
- Directories that became files (or vice versa)
- Modules split into packages (e.g., `file.py` → `file/` directory)

**Example output to look for:**
```
Only in source_version/HARK: distribution.py
Only in target_version/HARK: distributions
Only in source_version/HARK: frame.py
Only in target_version/HARK: simulator.py
```

**This tells you**: `distribution.py` was replaced by `distributions/` package, `frame.py` was removed, `simulator.py` was added.

### 1.2 CLASS-LEVEL CHANGES

**Question**: What classes were added, removed, or renamed?

```bash
# Compare class definitions in each file
for file in core.py utilities.py ConsumptionSaving/*.py; do
    echo "=== $file ==="
    echo "SOURCE:"
    grep -n "^class " source_version/HARK/$file 2>/dev/null | head -20
    echo "TARGET:"
    grep -n "^class " target_version/HARK/$file 2>/dev/null | head -20
done
```

**Record in your notes:**
- Classes removed (exist only in source)
- Classes added (exist only in target)
- Classes renamed
- Classes that became functions (or vice versa)

### 1.3 FUNCTION-LEVEL CHANGES (Not Just Methods!)

**Question**: What standalone functions were added, removed, or renamed?

**CRITICAL**: This is NOT about class methods. This is about module-level functions like `make_assets_grid()` in `utilities.py`.

```bash
# Compare function definitions in utilities.py
echo "=== utilities.py functions ==="
echo "SOURCE:"
grep -n "^def " source_version/HARK/utilities.py
echo "TARGET:"
grep -n "^def " target_version/HARK/utilities.py

# Compare function definitions in distribution modules
echo "=== distribution module functions ==="
echo "SOURCE:"
grep -n "^def " source_version/HARK/distribution.py 2>/dev/null
echo "TARGET:"
grep -rn "^def " target_version/HARK/distributions/*.py 2>/dev/null
```

**Record in your notes:**
- Functions removed
- Functions added
- Functions renamed (e.g., `construct_assets_grid` → `make_assets_grid`)

### 1.4 METHOD-LEVEL CHANGES

**Question**: What class methods were renamed (the camelCase → snake_case changes)?

```bash
# Find all method definitions in core classes
for file in core.py ConsumptionSaving/ConsIndShockModel.py ConsumptionSaving/ConsMarkovModel.py; do
    echo "=== $file methods ==="
    diff <(grep "def " source_version/HARK/$file) \
         <(grep "def " target_version/HARK/$file) || true
done
```

**Common patterns to find:**
- `initializeSim` → `initialize_sim`
- `preSolve` → `pre_solve`
- `addToTimeVary` → `add_to_time_vary`
- etc.

### 1.5 CONSTRUCTOR SIGNATURE CHANGES

**Question**: Did any commonly-used class constructors change their parameter names or signatures?

```bash
# Compare __init__ methods of key classes
echo "=== DiscreteDistribution constructor ==="
grep -A 30 "class DiscreteDistribution" source_version/HARK/distribution.py | grep -A 20 "def __init__"
grep -A 30 "class DiscreteDistribution" target_version/HARK/distributions/discrete.py | grep -A 20 "def __init__"

echo "=== Parameters class constructor ==="
grep -A 30 "class Parameters" source_version/HARK/core.py | grep -A 20 "def __init__"
grep -A 30 "class Parameters" target_version/HARK/core.py | grep -A 20 "def __init__"
```

**Record:**
- Parameter renames (e.g., `pmf` → `pmv`, `X` → `atoms`)
- New required parameters
- Changed default values
- Changed parameter order

### 1.6 PARAMETER/ATTRIBUTE NAME CHANGES

**Question**: What parameter dictionary keys or attribute names changed?

```bash
# Search for known patterns
grep -n "InitMean\|InitStd" source_version/HARK/ConsumptionSaving/ConsIndShockModel.py
grep -n "InitMean\|InitStd" target_version/HARK/ConsumptionSaving/ConsIndShockModel.py
```

**Common patterns:**
- `aNrmInitMean` → `kLogInitMean`
- `pLvlInitMean` → `pLogInitMean`
- `IncomeDstn` → `IncShkDstn`

### 1.7 IMPORT PATH CHANGES

**Question**: What imports need to change?

This is derived from structural changes (1.1) plus class/function moves:

```bash
# What was in distribution.py that moved?
grep "^from HARK.distribution import" source_version/HARK/examples/*.py 2>/dev/null || true
# How should it be imported now?
grep "^from HARK.distributions import" target_version/HARK/examples/*.py 2>/dev/null || true
```

### 1.8 BEHAVIORAL/API CHANGES

**Question**: Did any methods change their behavior, return types, or expected inputs?

This requires reading the actual diffs more carefully:

```bash
# Look for significant method body changes
diff source_version/HARK/core.py target_version/HARK/core.py | head -200
```

---

## Phase 2: Organize Discoveries

After completing ALL of Phase 1, organize your discoveries into this structure:

```markdown
# Discovered Changes: HARK {SOURCE_VERSION} → {TARGET_VERSION}

## Structural Changes
| Change Type | Old | New | Notes |
|-------------|-----|-----|-------|
| Module split | distribution.py | distributions/ (package) | Now has submodules |
| File removed | frame.py | (deleted) | Functionality moved |
| File added | (new) | simulator.py | New simulation framework |

## Class Changes
| Change Type | Class | Old Location | New Location | Notes |
|-------------|-------|--------------|--------------|-------|
| Removed | ConsMarkovSolver | ConsMarkovModel.py | (deleted) | Use constructor functions |
| Removed | MargValueFunc2D | ConsAggShockModel.py | (deleted) | Use MargValueFuncCRRA |

## Function Changes
| Change Type | Old Name | New Name | Module | Notes |
|-------------|----------|----------|--------|-------|
| Renamed | construct_assets_grid | make_assets_grid | utilities.py | |

## Method Renames
| Old Name | New Name | Classes Affected |
|----------|----------|------------------|
| initializeSim | initialize_sim | AgentType, subclasses |
| preSolve | pre_solve | AgentType |
[... complete list ...]

## Constructor Signature Changes
| Class | Parameter Changes | Notes |
|-------|-------------------|-------|
| DiscreteDistribution | pmf→pmv, X→atoms | Keyword args break |
| Parameters | (document changes) | |

## Parameter/Attribute Renames
| Old Name | New Name | Context |
|----------|----------|---------|
| aNrmInitMean | kLogInitMean | Agent init params |
[... complete list ...]

## Import Path Changes
| Old Import | New Import |
|------------|------------|
| from HARK.distribution import X | from HARK.distributions import X |
[... complete list ...]
```

---

## Phase 3: Generate Version-Specific Prompt

Only after completing Phases 1 and 2, generate the detailed migration prompt.

The generated prompt MUST include:

### Required Sections

1. **All Discovered Structural Changes** - From section 1.1
2. **All Class Additions/Removals** - From section 1.2
3. **All Function Renames** - From section 1.3 (NOT JUST METHODS!)
4. **All Method Renames** - From section 1.4
5. **All Constructor Signature Changes** - From section 1.5
6. **All Parameter/Attribute Renames** - From section 1.6
7. **All Import Path Changes** - From section 1.7
8. **Transformation Rules** for each category
9. **Edge Cases** discovered
10. **Manual Review Items** - things that can't be auto-transformed
11. **Validation Test Cases**

### Template for Generated Prompt

```markdown
# Prompt: HARK {SOURCE_VERSION} → {TARGET_VERSION} Migration Script

## Critical Requirement

This script MUST handle the following categories of changes. DO NOT assume
this list is complete - it was systematically discovered by diffing the actual
HARK source packages.

## 1. Structural Changes

[From Phase 1.1 - what files/directories changed]

## 2. Import Statement Changes

[From Phase 1.7 - complete mapping of old → new imports]

## 3. Class Changes

[From Phase 1.2 - classes added, removed, renamed]

## 4. Function Renames (Module-Level)

[From Phase 1.3 - NON-METHOD functions that were renamed]

| Old | New | Module |
|-----|-----|--------|
[complete table]

## 5. Method Renames (Class Methods)

[From Phase 1.4 - complete mapping]

| Old | New | Classes |
|-----|-----|---------|
[complete table]

## 6. Constructor Signature Changes

[From Phase 1.5 - with before/after signatures]

## 7. Parameter/Attribute Renames

[From Phase 1.6 - complete mapping]

## 8. Transformation Contexts

All renames must be applied in these contexts:
- Method definitions: `def old_name(self):` → `def new_name(self):`
- Self calls: `self.old_name()` → `self.new_name()`
- Variable calls: `obj.old_name()` → `obj.new_name()`
- Parent calls: `Parent.old_name(self)` → `Parent.new_name(self)`
- String literals: `'old_name()'` → `'new_name()'`
- Attributes: `self.oldAttr =` → `self.new_attr =`
- Import statements
- Function calls (not just methods!)

## 9. Manual Review Items

These patterns cannot be safely auto-transformed:
[List from Phase 1 discoveries]

## 10. Validation

After transformation, verify:
- [ ] All imports resolve
- [ ] No syntax errors
- [ ] Quick smoke test runs

Test cases:
[Before/after examples]
```

---

## Validation Checklist

Before finalizing the generated prompt, verify:

1. ☐ **Structural changes**: Did you document files added/removed/reorganized?
2. ☐ **Class changes**: Did you check for class additions AND removals?
3. ☐ **Function changes**: Did you check module-level functions (not just methods)?
4. ☐ **Constructor changes**: Did you compare `__init__` signatures of key classes?
5. ☐ **Evidence**: Do you have actual diff output supporting each discovery?
6. ☐ **Completeness**: Would another AI be able to write a complete migration script from your generated prompt alone?

---

## Common Discovery Pitfalls to Avoid

| Pitfall | What Gets Missed | Prevention |
|---------|------------------|------------|
| Not running actual diffs | Most structural changes | Use PyPI download method if no git access |
| Only checking class methods | Module-level function renames | Check `grep "^def "` in ALL modules |
| Skipping structural diff | Module splits, file removals | Always run `diff -rq` first |
| Only looking for modifications | Class/function removals | Explicitly look for "Only in source" |
| Treating example lists as complete | Renames not in examples | Treat any example list as <50% complete |
| Assuming git access | Unable to do discovery at all | Provide alternative (PyPI) method |

---

## Key Files to Always Diff

These files frequently contain breaking changes:

| File | Check For |
|------|-----------|
| `core.py` | AgentType methods, Parameters class, Market class |
| `utilities.py` | Grid construction functions, helper functions |
| `distribution.py` / `distributions/` | Distribution constructors, draw methods |
| `interpolation.py` | Value function classes |
| `ConsIndShockModel.py` | Consumer type constructors, solver methods |
| `ConsMarkovModel.py` | Markov consumer types, solver changes |
| `ConsAggShockModel.py` | Aggregate shock types, MargValueFunc |

---

## Important Context

User code may be written against an older HARK version than the "source version" - for example, code written for HARK 0.10.x that nominally "works" with 0.14.1 due to backward compatibility aliases. When upgrading such code to a newer version, you must discover:

1. Changes between the nominal source and target versions
2. Legacy patterns from OLDER versions that may still exist in user code

This is why discovery must check the FULL history of naming conventions and API changes.

---

*End of Meta-Prompt*

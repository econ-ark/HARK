# Meta-Prompt (GPT5.5): Generate a Reliable HARK Version Upgrade Script

## Purpose

This is a **meta-prompt** that generates a **version-specific upgrade prompt**. When given a HARK source version and target version, it must:

1. **Discover** breaking changes by inspecting *actual code + actual runtime signatures*, not just notes.
2. **Document** those changes as a complete, actionable spec (the “upgrade prompt”).
3. Enable generating a **migration tool** that edits real codebases safely (`.py` + `.ipynb`), produces reports, and minimizes false edits.

This meta-prompt is designed to **reduce missed upgrades**, especially the historically common failure mode: *constructor/signature changes and subtle type/semantic shifts that don’t show up in superficial greps*.

---

## Inputs (Fill These In)

- **HARK source version**: `{SOURCE_VERSION}` (e.g., `0.14.1`)
- **HARK target version**: `{TARGET_VERSION}` (e.g., `0.16.1`)
- **Target codebase directory to migrate**: `{TARGET_DIR}` (optional at discovery time, required for tool validation)
- **What file types to transform**:
  - Code files: `.py`, `.ipynb` (optionally `.sh` if it contains embedded python)
  - **Config files**: `requirements*.txt`, `pyproject.toml`, `setup.py`, `setup.cfg`, `environment*.yml`, `Pipfile`, `binder/*`
- **Execution constraints** (if any): no network, no git tags, etc.

---

## Non-Negotiable Principles (Hard Requirements)

1. **Evidence-first**: every “breaking change” listed must include at least one piece of evidence:
   - a diff excerpt OR
   - an extracted signature before/after OR
   - an example/test change OR
   - a runtime import+inspect comparison.
2. **API surface extraction is mandatory**: do not rely only on `git diff` or `diff -rq`.
3. **No regex-only migrations for Python** (except trivial import-only rewrites). Prefer **CST/AST-based transforms** for `.py`.
4. **Notebook-safe editing**: only edit **code cell sources**, preserve notebook structure/metadata.
5. **Coverage proof**: the final output must include a “how we know we didn’t miss obvious things” section:
   - remaining old identifiers count = 0 (or an explicit allowlist),
   - plus a triaged manual-review list.

---

## Phase 0 — Acquire Comparable Source/Target Code (Two Independent Paths)

You MUST be able to access both versions in a comparable form.

### Option A (No Git): PyPI wheel/source extraction

```bash
pip download econ-ark=={SOURCE_VERSION} econ-ark=={TARGET_VERSION} --no-deps -d /tmp/hark_diff
cd /tmp/hark_diff
python -c "import glob; print(glob.glob('econ_ark-*'))"
unzip -q econ_ark-{SOURCE_VERSION}-*.whl -d src
unzip -q econ_ark-{TARGET_VERSION}-*.whl -d tgt
```

### Option B (Git tags/commits available): git diff

```bash
git clone https://github.com/econ-ark/HARK.git
cd HARK
git diff {SOURCE_VERSION}..{TARGET_VERSION} -- HARK
```

### Verification checkpoint (must explicitly confirm)

- ☐ I can list source tree + target tree on disk
- ☐ I can run a structural diff of the HARK package directory
- ☐ I can run the API-surface extraction script(s) below on both versions

If you cannot confirm these, STOP and explain the blocker.

---

## Phase 1 — Systematic Discovery (Order Matters)

You MUST complete discovery in this order, and you MUST produce a structured “Change Inventory”.

### 1.1 Structural changes (files/dirs moved, split, removed)

Do a tree diff and record:
- modules renamed (file → file, file → package dir)
- packages created/removed
- new “entrypoint” modules (e.g., new `simulator.py`, etc.)

Example commands:

```bash
diff -rq src/HARK tgt/HARK | sort
```

### 1.2 Extract API surface (MANDATORY)

This is the biggest improvement vs typical prompts: build a before/after “API signature map”.

**Goal**: detect renames, signature changes, default changes, and callability changes *even if a file diff is noisy or a change is subtle*.

Produce (at minimum) a JSON for each version containing:
- module list
- for each module: classes (and base classes), functions
- for each callable: `inspect.signature()` if possible
- for dataclasses / attrs-like: field names if possible

Suggested extraction strategy:
- Install each version into *separate environments* (or run from extracted paths with `PYTHONPATH`).
- Use `pkgutil.walk_packages` + `importlib` to import modules.
- Gracefully skip heavy modules that require optional deps; record what was skipped.

**Deliverable**: a small “api_report_{version}.json” per version AND a “api_diff.md” that highlights:
- removed symbols
- added symbols
- signature changes (incl. constructors)
- moved symbols (heuristic: same name+signature in different module)

### 1.3 Identify systematic rename waves (naming conventions)

Find bulk renames like:
- camelCase → snake_case
- `IncomeDstn` → `IncShkDstn`
- module renames (`distribution` → `distributions`)

**Requirement**: provide a mapping table and explicitly list all contexts that must be rewritten:
- definitions
- calls on variables (`x.method(...)`)
- calls on indexed expressions (`xs[i].method(...)`)
- parent calls (`Parent.old(self)`)
- strings used as command lists (`"initializeSim()"`)

### 1.4 Constructor / signature changes (CRITICAL)

This category must be treated as first-class, not an afterthought.

**Algorithm**:
- From API diff: list classes whose `__init__` signature changed.
- Also explicitly check “high fan-out” classes used widely (distributions, interpolation, core Agent classes).
- For each: record old signature and new signature, and categorize changes:
  - keyword rename(s) (e.g., `pmf→pmv`, `X→atoms`)
  - positional order change
  - new required args
  - semantics/default changes

**Manual review policy**:
- If positional args are used and order changed OR meaning/type changed → WARN, don’t auto-edit blindly.
- If keyword args use old names → can usually auto-rename with high confidence.

### 1.5 Type/shape/semantic changes (scalar→list, RNG, etc.)

These often require context-sensitive changes, not mechanical renames:
- scalar→sequence expectations (e.g., time-varying parameters)
- numpy RNG API shifts (`randint` vs `integers`)
- return shape changes that break downstream code

**Requirement**: for each type change, include at least one *call site pattern* and a safe rewrite rule.

### 1.6 Backward-compatibility aliases and “hidden old code”

User code may still use older-than-source-era APIs due to compatibility shims.

**Required check**:
- look for alias assignments / re-exports in target version that kept old names temporarily
- scan further back in history (or older wheels) if your source→target diff shows “removal of alias”

Deliverable: a “Legacy Patterns to Still Scan For” section (e.g., pre-0.11 names).

### 1.7 Examples/tests/demos as ground truth migration evidence

Diff:
- HARK `examples/`
- HARK `tests/`
- any “DemARKs”/downstream examples if available

These diffs are used to:
- confirm what *actually changed in real code*
- reveal corner cases (strings, notebooks, odd constructors)

Deliverable: list of concrete "before→after" snippets that the migration tool must reproduce.

### 1.8 Project Configuration Files (Dependency Version Pins)

**This is commonly missed!** Any file in the target codebase that pins the HARK version must be updated.

**Required scan**:

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

# Also check binder/ directory specifically
find $TARGET_DIR -path "*/binder/*" -type f 2>/dev/null | xargs grep -l -i "econ-ark\|hark" 2>/dev/null
```

**Deliverable**: A table of files requiring version pin updates:

| File | Field/Line | Current Version | Target Version |
|------|------------|-----------------|----------------|
| `binder/requirements.txt` | `econ-ark==X.Y.Z` | `{SOURCE_VERSION}` | `{TARGET_VERSION}` |
| `pyproject.toml` | `dependencies` | ... | ... |

**Migration Rule**:
- Pattern: `econ-ark[=<>~!]*{SOURCE_VERSION_PATTERN}`
- Replace with: `econ-ark=={TARGET_VERSION}` (or appropriate specifier)
- Confidence: HIGH (simple string replacement)

**Why this matters**: If config files still specify the old HARK version, the codebase will install the wrong version and the migration will appear to have failed even though the code is correct.

---

## Phase 2 — Generate the Version-Specific Upgrade Prompt (The Output of This Meta-Prompt)

Generate a single markdown prompt with this structure:

1. **Scope and guarantees**
2. **How discovery was performed** (commands run, what was skipped)
3. **Change Inventory** (tables + evidence)
4. **Rewrite Rules**:
   - rule name
   - what it changes (old→new)
   - contexts it applies to
   - confidence (high/medium/low)
   - when to WARN instead of auto-edit
5. **Notebook policy**: code cells only; preserve metadata; don’t rewrite outputs unless requested
6. **Validation plan** (below)
7. **Known hazards** (model-specific renames, ambiguous patterns)
8. **Manual review checklist** (what humans must confirm)

### Rewrite Rules: Required coverage categories

Your upgrade prompt MUST explicitly cover:
- module/import moves
- systematic name changes (methods, functions, attributes)
- constructor signature changes
- type changes (scalar/list etc.)
- dependency API changes (numpy, pandas, etc.)
- string-based command patterns
- notebooks (`.ipynb`) code cell updates
- **project config files** (`requirements.txt`, `pyproject.toml`, `environment.yml`, `binder/`, etc.) with HARK version pins

---

## Phase 3 — Requirements for the Migration Tool That the Upgrade Prompt Will Produce

The version-specific prompt must instruct the implementing agent to build a tool with:

- **Modes**: `--dry-run` (default), `--apply`
- **Reports**: JSON + Markdown
  - counts by rule
  - per-file diffs (or snippets)
  - warnings/errors
- **Safety checks**:
  - parse Python AST after edits (`ast.parse`)
  - parse notebooks via `nbformat` or JSON validation
  - never corrupt notebook structure
- **Editing strategy**:
  - `.py`: CST-based editing strongly preferred (LibCST-style) to preserve formatting and avoid regex breakage
  - allow regex only for extremely trivial token-level edits (e.g., import module rename), and still validate with `ast.parse`
- **Idempotence**:
  - second run should produce zero additional edits (or only reformatting if enabled)

---

## Phase 4 — Validation and “Did We Miss Anything?” (Mandatory)

The version-specific prompt must require these validations:

1. **Static validation**
   - `ast.parse` on all modified `.py`
   - notebook JSON parse (and optionally `nbformat.validate`)
2. **Coverage checks**
   - grep for old identifiers (the rename mapping keys) → must be zero OR explicitly justified
   - grep for old import paths → must be zero OR explicitly justified
   - grep for old HARK version pins in config files (`econ-ark=={SOURCE_VERSION}`) → must be zero
3. **Golden testbeds**
   - run tool on known code that was upgraded historically (examples/DemARKs)
   - compare against “expected” diffs where possible
4. **Runtime smoke (when feasible)**
   - import key modules
   - instantiate a few core classes in target version with migrated params

Deliverable: a short “Coverage Proof” section listing:
- which identifiers were checked
- remaining occurrences (if any) and why

---

## Common Failure Modes (Explicitly Guard Against These)

Your generated version-specific prompt must explicitly warn against:

1. Only diffing a few "key files" and missing signature changes elsewhere
2. Only grepping method definitions (`def ...`) and missing:
   - constructor kwargs renames
   - default changes
   - renamed dataclass fields / dictionary keys
3. Regex-only edits that corrupt code or miss complex contexts
4. Forgetting string-based commands (command lists in strings)
5. Breaking notebooks by editing JSON incorrectly or rewriting non-code cells
6. Applying model-specific renames globally without gating/warnings
7. **Forgetting to update project config files** (`requirements.txt`, `pyproject.toml`, `binder/`, `environment.yml`) that pin the old HARK version — the code migration succeeds but the project still installs the wrong HARK version

---

## Output of THIS Meta-Prompt

Your output (as the meta-prompt executor) must be the complete **version-specific upgrade prompt** for `{SOURCE_VERSION} → {TARGET_VERSION}` that satisfies all requirements above.

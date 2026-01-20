# Case Study: HAFiscal Migration from HARK 0.14.1 to 0.17.0

## What is HAFiscal?

[HAFiscal](https://github.com/llorracc/HAFiscal) is a heterogeneous-agent fiscal policy
model that uses HARK for its consumption-saving agents. It was originally built on
HARK 0.14.1 and has been successfully migrated to HARK 0.17.0.

## Purpose of This Document

This case study documents lessons learned during HAFiscal's migration, which may help:
1. Other downstream projects upgrading from older HARK versions
2. HARK developers understand how users interact with the library
3. Inform potential documentation improvements

## Key Finding

**HAFiscal achieved numerical identity between HARK 0.14.1 and 0.17.0** after fixing
two issues in HAFiscal's code. These issues arose because HAFiscal relied on
undocumented internal HARK behavior rather than the prescribed API usage.

## Issue 1: Incomplete State Preservation

### The Pattern HAFiscal Used

HAFiscal implemented custom `save_state()` and `restore_state()` methods to checkpoint
simulation state for counterfactual experiments. However, it only saved a subset of
state variables:

```python
# HAFiscal's original code - incomplete
def save_state(self):
    self.aNrm_base = self.state_now['aNrm'].copy()
    self.pLvl_base = self.state_now['pLvl'].copy()
    # mNrm, bNrm were NOT saved
```

### Why This Worked in HARK 0.14.1

In 0.14.1, `initialize_sim()` had this implementation:
```python
if self.state_now[var] is None:
    self.state_now[var] = copy(blank_array)
```

Since `mNrm` and `bNrm` were already populated (not `None`), they weren't overwritten.
HAFiscal's incomplete save/restore happened to work.

### Why It Stopped Working in HARK 0.17.0

In 0.17.0, `initialize_sim()` unconditionally initializes all state variables:
```python
self.state_now[var] = copy(blank_array)
```

This is more correct behavior for an initialization function, but it exposed HAFiscal's
incomplete state management.

### The Fix

HAFiscal now saves and restores ALL state variables it needs:
```python
def save_state(self):
    self.aNrm_base = self.state_now['aNrm'].copy()
    self.pLvl_base = self.state_now['pLvl'].copy()
    self.bNrm_base = self.state_now['bNrm'].copy()  # Added
    self.mNrm_base = self.state_now['mNrm'].copy()  # Added
```

## Issue 2: Implicit RNG Seed Dependency

### The Pattern HAFiscal Used

HAFiscal relied on income shock distributions (`IncShkDstn`) having consistent default
seeds across HARK versions, without explicitly setting them.

### What Changed

The internal seeding happens to differ between versions (0.14.1: `763607780`,
0.17.0: `263618650`). Neither value is documented or guaranteed.

### The Fix

HAFiscal now explicitly sets seeds for reproducibility:
```python
for agent in agents:
    agent.IncShkDstn[0].seed = 763607780
    agent.IncShkDstn[0].reset()
```

## Verification

With these fixes, HAFiscal produces **numerically identical results** on both HARK versions.

## Suggestions for HARK

Based on this experience, HARK could help future users by:

1. **Documenting `initialize_sim()` behavior** - Clarify that it resets ALL state
   variables, so users know they must save everything they need.

2. **Adding a state checkpoint utility** - A built-in `save_all_state()`/`restore_all_state()`
   would prevent users from accidentally omitting variables.

3. **Documenting RNG seeding practices** - Guidance on explicit seed management for
   reproducibility across versions.

4. **Adding deprecation warnings** - When common misuse patterns are detected, a warning
   could guide users toward proper usage.

## References

- HAFiscal repository: https://github.com/llorracc/HAFiscal
- Verification performed: January 2026

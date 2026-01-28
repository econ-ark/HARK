# Case Study: HAFiscal Migration from HARK 0.14.1 to 0.17.0

This document describes the experience of migrating HAFiscal, a large-scale
heterogeneous agent fiscal policy model, from HARK 0.14.1 to HARK 0.17.0.

## Project Overview

**HAFiscal** is a codebase for the paper "Welfare and Spending Effects of
Consumption Stimulus Policies" (Carroll, Crawley, Du, Frankovic, Tretvoll).

- ~15,000 lines of Python code
- Uses HARK's MarkovConsumerType with aggregate shocks
- Complex interpolation with 2D value functions
- Multi-day full simulation runtime

## Migration Outcome

**Success:** After applying fixes, HARK 0.14.1 and HARK 0.17.0 produce
**numerically identical results** (all differences < 10^-9).

## Challenges Encountered

### Challenge 1: Interpolation Broadcasting (HARK Bug - Fixed in PR #1701)

**Symptom:** IndexError when calling value functions with mixed scalar/array inputs.

**Root Cause:** HAFiscal evaluates consumption functions like:

    `cNrm = cFunc(mNrm_array, Cagg_scalar)`

Due to a simplification in HARK.interpolation, HARK 0.17.0's interpolation classes
failed when inputs have different shapes. This coding pattern was not expected
and not intentionally supported, but happened to work for some interpolators in
0.16.1 and prior.

**Resolution:** PR #1701 added np.broadcast_arrays() to HARKinterpolator2D/3D/4D.
This change takes effect in HARK 0.17.1, unreleased at the time of this writing.
In the HAFiscal project code, interpolant evaluations like the one above are now:

    `cNrm = cFunc(mNrm_array, np.full(mNrm_array.shape, Cagg_scalar))`

This change allows the project code to run with the currently released HARK.

### Challenge 2: RNG Behavior Changes

**Symptom:** Simulations produced different results even with the same seed.

**Root Cause:** HARK 0.17.0 changed reset_rng() behavior - it now resets ALL
distributions in self.distributions, not just IncShkDstn.

**Resolution:** HAFiscal added override methods to replicate 0.14.1 behavior.

### Challenge 3: Pre-existing Bug Discovery

Migration testing uncovered a pre-existing bug in HAFiscal's borrowing constraint
calculation. This demonstrates the value of numerical verification during migration.

## Verification Results

After fixes, 21 result files compared - all identical within floating-point precision:

- base_results.csv: 3.49e-10
- recession_results.csv: 3.71e-10
- All other files: < 3.8e-10

## Recommendations for Other Migrators

1. Start with numerical verification tests before attempting migration
2. Use incremental testing to isolate where divergence occurs
3. Check RNG behavior if your code depends on reproducible random sequences
4. Pin to specific commits/tags during migration

## References

- HAFiscal: https://github.com/llorracc/HAFiscal-Latest
- PR #1701: https://github.com/econ-ark/HARK/pull/1701
- Full docs: https://github.com/llorracc/HAFiscal-Latest/blob/master-with-borocnstnat-fix-using-0p17p0/docs/HARK_Migration_Guide.md

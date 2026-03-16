# Plan: Monte Carlo vs Transition Matrix Guide

## Goal

Create a Jupyter notebook guide that teaches Monte Carlo users how to compare
their simulation results with transition matrix methods, starting from Will Du's
working example and adapting to a new model parameterization.

## Context

Transition matrix methods in HARK live exclusively in `NewKeynesianConsumerType`
(a subclass of `IndShockConsumerType`). The key methods are:

- `define_distribution_grid()` — grids over normalized market resources and permanent income
- `calc_transition_matrix()` — builds sparse Markov transition matrix on the joint state space
- `calc_ergodic_dist()` — finds the unit eigenvector (steady-state distribution)
- `compute_pe_steady_state()` — convenience wrapper: solve + transition matrix + ergodic dist

Any model that can be expressed as a `NewKeynesianConsumerType` can use both
Monte Carlo and transition matrix simulation. Will Du's
`examples/SequenceSpaceJacobians/Transition_Matrix_Example.ipynb` is the only
existing head-to-head comparison.

## Choice of "similar but different" model

Since transition matrix methods are only available in `NewKeynesianConsumerType`,
the new example must use that class but with a different calibration.

- **Option A (recommended start): Higher risk aversion + lower discount factor.**
  CRRA=5 (vs Du's CRRA=2), DiscFac=0.96 (vs 0.975). Produces a more
  precautionary agent with a fatter-tailed wealth distribution. MC noise is
  larger in the tails, making the comparison more interesting.

- **Option B: Different income process.**
  Higher permanent shock variance (PermShkStd=0.12 vs ~0.06) or unemployment
  (UnempPrb=0.07, IncUnemp=0.3). More dispersed distribution where grid
  resolution matters more.

- **Option C (stretch): Finite lifecycle.**
  `cycles=1` with age-varying PermGroFac, LivPrb, income shocks. Exercises the
  finite-horizon branch of `calc_transition_matrix()` (list of per-period
  matrices).

## Notebook structure

### Section 1: Introduction and Motivation

- Audience: users who run MC simulations and want to validate or complement
  with transition matrices.
- What transition matrices give you that MC doesn't: deterministic aggregates,
  exact steady-state distributions, Jacobians for linearized dynamics.
- What MC gives you that transition matrices don't: no grid discretization
  error, works for any model with `sim_one_period`, path-level statistics.

### Section 2: The Baseline Model (Du's parameterization)

- Import `NewKeynesianConsumerType`, set up Du's dictionary (CRRA=2,
  DiscFac=0.975, Rfree=1.04^0.25), solve.
- MC: `simulate()`, compute aggregate C and A.
- Transition matrix: `calc_transition_matrix()` + `calc_ergodic_dist()`,
  compute aggregate C and A.
- Compare: aggregates, time-series path, distribution of mNrm.
- Cleaned-up, annotated version of Du's Cells 8-24.

### Section 3: A New Calibration (the adapted example)

- Change parameters (CRRA=5, DiscFac=0.96, and/or unemployment).
- Solve. Run MC. Run transition matrix. Compare.
- Highlight: fatter tails, more MC noise, grid points needed to capture tail.
- Show Harmenberg neutral measure improvement.
- Show convergence: vary `mCount` and plot transition matrix aggregates
  converging to MC.

### Section 4: Practical Guidance

- When to use which method.
- How to set grid parameters (`mCount`, `mMax`, `mFac`, `num_pointsP`).
- Harmenberg trick: when and why.
- Diagnostics: if MC and transition matrix disagree, what to check.

### Section 5 (stretch): Lifecycle Extension

- Finite-horizon with age-varying parameters.
- Per-period transition matrices.
- Evolve distribution forward and compare to MC lifecycle paths.

## Implementation approach

1. Start from Du's code: copy working cells into new notebook, clean up.
2. Verify the baseline runs.
3. Add new calibration, iterate on grid settings.
4. Write practical guidance.
5. File location: `sims-about/` in the HARK repo.

## Key constraint

Transition matrix methods exist only in `NewKeynesianConsumerType`.
However, it is a drop-in replacement for `IndShockConsumerType` (inherits from
it), so any IndShock calibration can be run through `NewKeynesianConsumerType`.

## Iterative workflow

- Iteration 1: Baseline (Du's params) working in new notebook with clean exposition.
- Iteration 2: New calibration added, comparison plots working.
- Iteration 3: Harmenberg trick, grid convergence, practical guidance.
- Iteration 4 (stretch): Lifecycle extension.

# Project Goal: Simulation Methods Comparison Guide

## What

Create a pedagogical Jupyter notebook that serves as a practical guide for HARK
users who currently use Monte Carlo simulation and want to understand how to
compare their results with transition matrix methods.

## Why

HARK supports two fundamentally different simulation approaches:

1. **Monte Carlo (MC)**: Draw shocks for N agents, step forward period by
   period, record histories. Used by every model in HARK via
   `AgentType.simulate()`.

2. **Transition matrices (TM)**: Discretize states onto grids, build Markov
   transition matrices, evolve distributions deterministically. Currently
   implemented only in `NewKeynesianConsumerType`.

These methods have complementary strengths:

- MC is accurate (continuous state space) but imprecise (stochastic noise in
  aggregates).
- TM is precise (deterministic, no sampling noise) but less accurate
  (discretization error from finite grids).

Despite this, there is only one existing example that compares them head-to-head
(Will Du's `Transition_Matrix_Example.ipynb`). Users who want to cross-validate
their MC results, or who need the precision of TM for applications like
Sequence Space Jacobians, have no gentle on-ramp.

## For whom

- Researchers using HARK who run MC simulations and want to validate aggregates
  or distributions against a non-stochastic benchmark.
- Users building HANK models who need transition matrices for SSJ computation
  and want to understand the relationship to MC simulation.
- Students learning heterogeneous-agent methods who want to see both approaches
  applied to the same model.

## Approach

1. Start from Will Du's working comparison code.
2. Clean it up into a well-documented guide with clear exposition.
3. Adapt the comparison to at least one additional calibration (different from
   Du's) so users see how the tradeoffs change with model characteristics.
4. Provide practical guidance on grid tuning, Harmenberg's neutral measure,
   and diagnostics.

## Success criteria

- A self-contained notebook that a HARK user can run end-to-end.
- Clear side-by-side comparisons of MC and TM for at least two calibrations.
- Practical advice that helps users decide when and how to use each method.

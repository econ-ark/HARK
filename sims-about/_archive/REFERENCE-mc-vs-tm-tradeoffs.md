# Reference: Monte Carlo vs Transition Matrix Tradeoffs

## The fundamental tradeoff

| Dimension | Monte Carlo | Transition Matrix |
|-----------|------------|-------------------|
| **Noise** | Stochastic — aggregates fluctuate across runs | Deterministic — aggregates are exact given grid |
| **Grid error** | None (continuous state space) | Discretization error (improves with more grid points) |
| **Speed (steady state)** | Fast to simulate, slow to converge (need many agents/periods) | Slow to build matrix, instant steady state via eigenvector |
| **Speed (dynamics)** | Re-simulate entire population each period | Matrix-vector multiply per period |
| **Harmenberg trick** | Applies (reweights agents for better aggregation) | Applies (collapses pLvl dimension, dramatic speedup) |
| **Model generality** | Works for any model with `sim_one_period` | Only `NewKeynesianConsumerType` currently |
| **Path-level stats** | Yes (individual histories, percentiles, panel regressions) | No (only distributional aggregates) |
| **SSJ Jacobians** | Cannot compute directly | Required input for fake-news algorithm |

## When to use Monte Carlo

- You need individual-level histories (panel data, lifecycle paths)
- Your model doesn't fit `NewKeynesianConsumerType` (portfolio choice, health,
  habit formation, etc.)
- You want path-level statistics (percentiles, Gini, mobility)
- You're doing method of simulated moments (MSM) estimation
- Quick prototyping where noise is acceptable

## When to use transition matrices

- You need precise steady-state aggregates (no sampling noise)
- You're computing SSJ Jacobians for HANK models
- You need impulse response functions to MIT shocks
- You want the exact steady-state wealth distribution
- Speed matters and you can use Harmenberg's trick

## When to use both

- **Cross-validation:** Run MC to check that TM aggregates are close (if they
  disagree, the grid is probably too coarse).
- **Development workflow:** Use TM for quick steady-state checks, MC for final
  distributional analysis.
- **Publication:** Report TM aggregates for precision, MC distributions for
  individual-level moments.

## Common pitfalls

1. **Grid too coarse:** TM with 90 mGrid points can have significant
   discretization error. Use Harmenberg + 1000+ points.
2. **Tail truncation:** If `mMax` is too small, the TM misses the upper tail.
   Check that the ergodic distribution has negligible mass near the boundary.
3. **MC not converged:** With too few agents or periods, MC aggregates are
   noisy. Use 50,000+ agents and 1,000+ periods (after burn-in).
4. **Forgetting Harmenberg:** Without the neutral measure, the 2D (mNrm, pLvl)
   grid is expensive and less accurate. Always use Harmenberg for infinite-
   horizon problems unless you specifically need the pLvl distribution.

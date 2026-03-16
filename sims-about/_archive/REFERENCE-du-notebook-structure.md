# Reference: Structure of Will Du's Transition_Matrix_Example.ipynb

Location: `examples/SequenceSpaceJacobians/Transition_Matrix_Example.ipynb`
Author: William Du (wdu9@jhu.edu)
Size: ~967KB (large due to embedded plot output)

## Cell-by-cell summary

### Setup (Cells 0-7)

| Cell | Type | Content |
|------|------|---------|
| 0 | markdown | Title: "Using Transition Matrix Methods under IndShockConsumerType" |
| 1 | markdown | Overview of three key functions: `define_distribution_grid`, `calc_transition_matrix`, `calc_ergodic_dist` |
| 2 | markdown | "Set up Computational Environment" |
| 3 | code | Imports: time, deepcopy, matplotlib, numpy, NewKeynesianConsumerType |
| 4 | markdown | "Set up the Dictionary" |
| 5 | code | Parameter dictionary: CRRA=2, DiscFac=0.975, Rfree=1.04^0.25, AgentCount=50000, T_sim=1000, mCount=90, mFac=3, mMax=10000 |
| 6 | markdown | "Create an Instance and Solve" |
| 7 | code | `example1 = NewKeynesianConsumerType(**Dict); example1.solve()` |

### Part A: Steady-state MC vs TM comparison (Cells 8-20)

| Cell | Type | Content |
|------|------|---------|
| 8 | markdown | "Simulation: Transition Matrix vs Monte Carlo" |
| 9 | markdown | Section description |
| 10 | markdown | "Method 1: Monte Carlo" |
| 11 | code | `initialize_sim()`, `simulate()`, compute `Monte_Carlo_Consumption`, `Monte_Carlo_Assets` |
| 12 | markdown | "Method 2: Transition Matrices" |
| 13 | code | `define_distribution_grid(num_pointsP=110)`, `calc_transition_matrix()` (~47s), `calc_ergodic_dist()` |
| 14 | code | Compute TM aggregate C and A from ergodic distribution |
| 15 | markdown | "Comparing Steady State Outputs" |
| 16 | code | Print both sets of aggregates |
| 17 | markdown | "Comparing Simulated Path of Aggregate Assets" |
| 18 | code | Extract MC aggregate asset time series from history |
| 19 | code | Plot MC path (fluctuating) vs TM path (flat horizontal line) |
| 20 | markdown | **"Precision vs Accuracy"** — key conceptual discussion |

### Distribution comparisons (Cells 21-30)

| Cell | Type | Content |
|------|------|---------|
| 21-22 | md+code | Distribution of normalized market resources (mNrm) |
| 23-24 | md+code | Distribution of permanent income (pLvl) |
| 25-28 | md+code | Distribution of wealth in levels (mLvl) — includes `jump_to_grid_fast()` |
| 29-30 | md+code | Distribution of liquid assets (aLvl) |

### Part B: Harmenberg neutral measure (Cells 31-42)

| Cell | Type | Content |
|------|------|---------|
| 31-32 | markdown | Setup for MIT shock experiment; compute steady state |
| 33 | code | Create steady-state agent, solve |
| 34 | markdown | "Simulating With Harmenberg (2021) Method" |
| 35 | code | `ss.neutral_measure = True`, `mCount=1000`, `mMax=3000` |
| 36 | code | TM with Harmenberg (~7s vs 47s without) |
| 37 | markdown | Speedup discussion |
| 38 | code | Three-way plot: MC vs TM vs TM-Harmenberg |
| 39 | markdown | "Increasing gridpoints increases accuracy" |
| 40 | code | Grid convergence: mpoints = [100, 150, 200, 500, 3000] |
| 41 | code | Convergence plot |
| 42 | markdown | Harmenberg improves both speed and accuracy |

### Part C: MIT shock experiment (Cells 43-60)

| Cell | Type | Content |
|------|------|---------|
| 43-44 | md+code | MC simulation with Harmenberg trick |
| 45-46 | md+code | Solve finite-horizon agent anticipating R shock at t=10 |
| 47-48 | md+code | Implement perturbation: dx=-0.05 at period i=10 |
| 49-50 | md+code | Solve the perturbed agent |
| 51-52 | md+code | MC simulation with Harmenberg for perturbed agent |
| 53-54 | md+code | Calculate TM with Harmenberg for perturbed agent (~1s) |
| 55-56 | md+code | Evolve distribution forward using per-period TMs |
| 57-58 | md+code | Plot: path of aggregate consumption (TM vs MC) |
| 59-60 | md+code | Plot: path of aggregate assets (TM vs MC) |
| 61 | code | Empty |

## Key parameters in Du's example

```python
{
    "CRRA": 2,
    "Rfree": [1.04**0.25],
    "DiscFac": 0.975,
    "LivPrb": [0.99375],
    "PermGroFac": [1.00],
    "AgentCount": 50000,
    "T_sim": 1000,
    "PermShkCount": 5,
    "TranShkCount": 5,
    "PermShkStd": [0.06],
    "TranShkStd": [0.3],
    "UnempPrb": 0.07,
    "IncUnemp": 0.3,
    "mCount": 90,
    "mFac": 3,
    "mMax": 10000,
}
```

## Key outputs and findings

1. **Aggregate assets:** MC fluctuates around ~1.53; TM gives exactly ~1.48
   (grid discretization error). With Harmenberg + 3000 grid points, TM
   converges much closer to MC.

2. **Time series:** MC path oscillates (sampling noise); TM is a flat line
   (deterministic).

3. **Distributions:** Generally similar shape, with some discrepancy in tails
   where grid resolution matters.

4. **MIT shock:** Both methods produce nearly identical impulse response paths
   for aggregate C and A after an anticipated interest rate shock.

5. **Harmenberg speedup:** 47s -> 7s for transition matrix computation, with
   1000 grid points instead of 90.

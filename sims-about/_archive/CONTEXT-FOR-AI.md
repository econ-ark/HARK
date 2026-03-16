# Context Prompt for AI Assistants Working on This Project

## Project

Build a Jupyter notebook guide comparing Monte Carlo and transition matrix
simulation methods in HARK. The notebook lives in `sims-about/` in the HARK
repo root.

## Key facts about HARK's simulation infrastructure

### Three simulation approaches

1. **Monte Carlo** — `AgentType.simulate()` in `HARK/core.py`. Draws shocks
   for N agents, steps forward via `sim_one_period()` (mortality -> shocks ->
   states -> controls -> post-states), records `track_vars` in `history`.

2. **Transition matrices** — `NewKeynesianConsumerType` in
   `HARK/ConsumptionSaving/ConsNewKeynesianModel.py`. Methods:
   `define_distribution_grid()`, `calc_transition_matrix()`,
   `calc_ergodic_dist()`, `compute_pe_steady_state()`. Builds sparse Markov
   transition matrix over discretized (mNrm, pLvl) state space using the
   "lottery" / "jump to grid" method.

3. **Sequence Space Jacobians (SSJ)** — `HARK/SSJutils.py`. Uses transition
   matrices internally. Computes linearized impulse responses via the
   fake-news algorithm (Auclert et al. 2021).

### Where transition matrix methods live

ONLY in `NewKeynesianConsumerType`. This class inherits from
`IndShockConsumerType` and adds:
- HANK-style income process parameters (wage, labor, tax_rate)
- `define_distribution_grid()` — grids over mNrm and pLvl
- `calc_transition_matrix()` — builds transition matrix (infinite or finite horizon)
- `calc_ergodic_dist()` — eigenvector of transition matrix
- `compute_pe_steady_state()` — solve + TM + ergodic dist + aggregate C,A
- `calc_jacobian()` — SSJ Jacobians via fake-news algorithm

Since it inherits from `IndShockConsumerType`, any IndShock calibration can
be run through `NewKeynesianConsumerType` to get both MC and TM.

### How `calc_transition_matrix()` works

1. Evaluate consumption function on mNrm grid: `aNext = mGrid - cFunc(mGrid)`
2. Bank balances: `bNext = Rfree * aNext`
3. For each shock realization (discretized income distribution):
   - Compute next-period mNrm
   - Use `jump_to_grid_1D` or `jump_to_grid_2D` to assign probability mass to
     nearest grid points (preserving conditional mean — the "lottery" method)
4. Weight by survival probability; dead agents replaced by newborn distribution
5. Result: sparse transition matrix T where T[i,j] = Prob(state j -> state i)

For infinite horizon: single transition matrix. For finite horizon: list of
per-period matrices.

### Harmenberg (2021) neutral measure

Reformulates the problem to eliminate the permanent income grid (2D -> 1D),
dramatically reducing computation time and improving accuracy. Activated by
`agent.neutral_measure = True` before `update_income_process()`.

### The existing comparison: Will Du's notebook

File: `examples/SequenceSpaceJacobians/Transition_Matrix_Example.ipynb`
Author: William Du (wdu9@jhu.edu)

Structure (61 cells):
- **Part A (Cells 8-24):** Steady-state comparison. Same solved model
  simulated two ways. MC: `simulate()` with 50,000 agents, 1000 periods.
  TM: `calc_transition_matrix()` + `calc_ergodic_dist()`. Compares:
  aggregate C and A, time-series paths, distributions of mNrm, pLvl,
  wealth, liquid assets.

- **Part B (Cells 34-42):** Harmenberg neutral measure. Shows speedup
  (47s -> 7s) and accuracy improvement. Grid convergence experiment.

- **Part C (Cells 45-60):** MIT shock experiment. Anticipated interest rate
  change at t=10. Both MC and TM produce nearly identical impulse responses.

Key parameters: CRRA=2, DiscFac=0.975, Rfree=1.04^0.25, infinite horizon.

Key finding: TM is perfectly precise (flat aggregate path) but has grid
discretization error. MC is accurate (no grid error) but noisy.

## Complete inventory of models and simulation methods

### ConsumptionSaving models (all solve-then-simulate)

| Model | Agent type(s) | Simulation method |
|-------|--------------|-------------------|
| ConsIndShockModel | PerfForesightConsumerType, IndShockConsumerType, KinkedRconsumerType | Monte Carlo |
| ConsIndShockModelFast | PerfForesightConsumerTypeFast, IndShockConsumerTypeFast | Monte Carlo (Numba solver) |
| ConsAggShockModel | AggShockConsumerType, KrusellSmithType, CobbDouglasEconomy | MC + Markov transitions for aggregate state |
| ConsMarkovModel | MarkovConsumerType | MC + Markov state transitions |
| ConsNewKeynesianModel | NewKeynesianConsumerType | MC + transition matrices + SSJ |
| ConsPrefShockModel | PrefShockConsumerType, KinkyPrefConsumerType | Monte Carlo |
| ConsBequestModel | BequestWarmGlowConsumerType, BequestWarmGlowPortfolioType | Monte Carlo |
| ConsGenIncProcessModel | GenIncProcessConsumerType, PersistentShockConsumerType | Monte Carlo |
| ConsMedModel | MedShockConsumerType, MedExtMargConsumerType | Monte Carlo |
| ConsPortfolioModel | PortfolioConsumerType | Monte Carlo |
| ConsRiskyAssetModel | RiskyAssetConsumerType | Monte Carlo |
| ConsRepAgentModel | RepAgentConsumerType, RepAgentMarkovConsumerType | MC + Markov |
| TractableBufferStockModel | TractableConsumerType | Monte Carlo |
| ConsWealthUtilityModel | WealthUtilityConsumerType, CapitalistSpiritConsumerType | Monte Carlo |
| ConsWealthPortfolioModel | WealthPortfolioConsumerType | Monte Carlo |
| ConsLaborModel | LaborIntMargConsumerType | Monte Carlo |
| ConsHealthModel | BasicHealthConsumerType | Monte Carlo |
| ConsRiskyContribModel | RiskyContribConsumerType | Custom multi-stage MC |
| ConsHabitModel | HabitConsumerType | Monte Carlo |
| ConsLabeledModel | IndShockLabeledType, PortfolioLabeledType, etc. | Same as underlying |
| ConsSequentialPortfolioModel | SequentialPortfolioConsumerType | Monte Carlo |

### Core simulation modules

| Module | What it provides | Method |
|--------|-----------------|--------|
| `HARK/core.py` (AgentType) | `simulate()`, `sim_one_period()`, `get_shocks/states/controls/poststates` | Monte Carlo |
| `HARK/simulator.py` (AgentSimulator) | `simulate(T)`, `make_transition_matrices()`, `simulate_cohort_by_grids()`, `find_steady_state()` | MC + Transition matrix |
| `HARK/simulation/monte_carlo.py` | `Simulator`, `AgentTypeMonteCarloSimulator`, `MonteCarloSimulator` | Monte Carlo (DBlock-based) |
| `HARK/model.py` (DBlock) | `simulate_dynamics()`, `transition()` | Used by both |
| `HARK/SSJutils.py` | `make_basic_SSJ_matrices()`, fake-news algorithm | SSJ (transition matrix) |
| `HARK/mat_methods.py` | `mass_to_grid()` — lottery method | Transition matrix support |

### Examples that compare MC and TM

Only two notebooks directly compare the methods:

1. **`examples/SequenceSpaceJacobians/Transition_Matrix_Example.ipynb`**
   — Head-to-head comparison of MC vs TM for same IndShock model.
   Steady-state aggregates, distributions, MIT shock impulse responses.
   Shows Harmenberg trick and grid convergence.

2. **`examples/SequenceSpaceJacobians/KS-HARK-presentation.ipynb`**
   — Krusell-Smith GE model. Uses TM (via `compute_pe_steady_state()`) for
   steady state, then SSJ Jacobians for dynamics. Not a direct MC vs TM
   comparison but uses both internally.

### Summary by simulation method across all examples

| Method | Where used |
|--------|-----------|
| Monte Carlo only | 18 of 21 ConsumptionSaving models; most examples |
| MC + Markov transitions | ConsAggShockModel, ConsMarkovModel, ConsRepAgentModel |
| MC + transition matrix | ConsNewKeynesianModel, Transition_Matrix_Example, KS-HARK-presentation |
| Transition matrix only | AgentSimulator.simulate_cohort_by_grids(), find_steady_state() |
| SSJ (Jacobians) | SSJutils.py, 4 SSJ example notebooks |

## Important files for this project

- `HARK/ConsumptionSaving/ConsNewKeynesianModel.py` — The only model with TM
- `HARK/utilities.py` — `gen_tran_matrix_1D`, `gen_tran_matrix_2D`, `jump_to_grid_1D`, `jump_to_grid_2D`
- `examples/SequenceSpaceJacobians/Transition_Matrix_Example.ipynb` — Du's comparison
- `examples/ConsIndShockModel/IndShockConsumerType.ipynb` — Standard MC example
- `HARK/core.py` — AgentType.simulate() infrastructure

## Rules

- Do NOT modify files in `project/repos/` (submodules) in the HARK_ask-your-project repo.
- The HARK repo at `/Volumes/Sync/GitHub/econ-ark/HARK` is the working copy.
- New files for this project go in `sims-about/` at the HARK repo root.

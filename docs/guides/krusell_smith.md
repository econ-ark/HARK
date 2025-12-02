# Krusell-Smith Model in HARK

## Introduction

This guide provides a comprehensive overview of the Krusell-Smith (1998) model implementation in HARK, based on the seminal paper "Income and Wealth Heterogeneity in the Macroeconomy" published in the Journal of Political Economy.

The Krusell-Smith model is a heterogeneous agent macroeconomic model that examines how individual income and wealth heterogeneity affects aggregate economic outcomes. In this model, households face idiosyncratic employment shocks in an economy with aggregate productivity shocks that follow a Markov process.

HARK's implementation provides tools for both solving the individual household problem and finding the general equilibrium aggregate saving rule through simulation and regression methods.

## Quick Start

### Minimal Working Example

```python
from HARK.ConsumptionSaving.ConsAggShockModel import (
    KrusellSmithType,
    KrusellSmithEconomy
)

# Create agent
agent = KrusellSmithType()
agent.cycles = 0
agent.AgentCount = 5000

# Create economy
economy = KrusellSmithEconomy(agents=[agent])
economy.max_loops = 10
economy.verbose = True

# Link agent to economy
agent.get_economy_data(economy)

# Solve for equilibrium
economy.make_AggShkHist()
economy.solve()

# Access results
print(f"AFunc (bad): intercept={economy.AFunc[0].intercept}, slope={economy.AFunc[0].slope}")
print(f"AFunc (good): intercept={economy.AFunc[1].intercept}, slope={economy.AFunc[1].slope}")
```

## Model Components

### `KrusellSmithType` Class

The `KrusellSmithType` class represents individual agents in the Krusell-Smith economy. This class is found in `HARK.ConsumptionSaving.ConsAggShockModel`.

**Key Features:**

- Agents face idiosyncratic employment shocks
- Aggregate state follows a two-state Markov process (bad=0, good=1)
- Agents form expectations about aggregate capital based on perceived aggregate market resources
- Uses specialized solution methods optimized for the KS structure

**Important Attributes:**

```python
time_inv_ = ["DiscFac", "CRRA", "aGrid", "ProbArray", "mNextArray", "MnextArray", "RnextArray"]
state_vars = ["aNow", "mNow", "EmpNow"]  # Current state: assets, market resources, employment
shock_vars_ = ["Mrkv"]  # Markov state shock
```

**Initialization:**

```python
from HARK.ConsumptionSaving.ConsAggShockModel import KrusellSmithType

agent = KrusellSmithType()
agent.cycles = 0  # Infinite horizon
agent.AgentCount = 5000  # Number of agents to simulate
```

Note: The `KrusellSmithType` must be used in conjunction with a `KrusellSmithEconomy` instance. Use the `get_economy_data()` method to import economy-determined objects into the agent.

### `KrusellSmithEconomy` Class

The `KrusellSmithEconomy` class represents the macroeconomic environment in which `KrusellSmithType` agents live. This is a subclass of `Market` that implements the aggregate dynamics and equilibrium computation.

**Key Features:**

- Two-state Markov process for aggregate productivity (good/bad)
- State-dependent unemployment rates
- Computes equilibrium aggregate saving rules
- Simulates aggregate economic history

**Tracked Variables:**

```python
sow_vars = ["Mnow", "Aprev", "Mrkv", "Rnow", "Wnow"]  # Variables distributed to agents
reap_vars = ["aNow", "EmpNow"]  # Variables collected from agents
track_vars = ["Mrkv", "Aprev", "Mnow", "Urate"]  # Variables tracked in history
dyn_vars = ["AFunc"]  # Dynamic rules that evolve during solution
```

**Initialization:**

```python
from HARK.ConsumptionSaving.ConsAggShockModel import KrusellSmithEconomy

economy = KrusellSmithEconomy(agents=[agent])
economy.max_loops = 10  # Maximum iterations for equilibrium
economy.act_T = 11000  # Simulation periods
economy.T_discard = 1000  # Initial periods to discard
```

## Model Parameters

### Agent Parameters (`init_KS_agents`)

The default agent parameters are stored in the `init_KS_agents` dictionary (line ~1596 in `ConsAggShockModel.py`):

| Parameter     | Default Value | Description                                    |
| ------------- | ------------- | ---------------------------------------------- |
| `DiscFac`     | 0.99          | Discount factor                                |
| `CRRA`        | 1.0           | Coefficient of relative risk aversion          |
| `LbrInd`      | 1.0           | Labor supply when employed                     |
| `aMin`        | 0.001         | Minimum asset value                            |
| `aMax`        | 50.0          | Maximum asset value                            |
| `aCount`      | 32            | Number of points in asset grid                 |
| `aNestFac`    | 2             | Nesting factor for exponentially-spaced grids  |
| `MaggCount`   | 25            | Number of aggregate market resource gridpoints |
| `MaggPerturb` | 0.01          | Perturbation around steady state               |
| `MaggExpFac`  | 0.12          | Expansion factor for aggregate grid            |
| `AgentCount`  | 5000          | Number of agents to simulate                   |

### Economy Parameters (`init_KS_economy`)

The default economy parameters are stored in the `init_KS_economy` dictionary (line ~2935 in `ConsAggShockModel.py`):

| Parameter    | Default Value | Description                                    |
| ------------ | ------------- | ---------------------------------------------- |
| `DiscFac`    | 0.99          | Discount factor                                |
| `CRRA`       | 1.0           | Coefficient of relative risk aversion          |
| `LbrInd`     | 0.3271        | Aggregate labor supply (from Krusell-Smith)    |
| `ProdB`      | 0.99          | Aggregate productivity in bad state            |
| `ProdG`      | 1.01          | Aggregate productivity in good state           |
| `CapShare`   | 0.36          | Capital share in production                    |
| `DeprRte`    | 0.025         | Depreciation rate                              |
| `DurMeanB`   | 8.0           | Mean duration of bad aggregate state           |
| `DurMeanG`   | 8.0           | Mean duration of good aggregate state          |
| `SpellMeanB` | 2.5           | Mean unemployment spell in bad state           |
| `SpellMeanG` | 1.5           | Mean unemployment spell in good state          |
| `UrateB`     | 0.10          | Unemployment rate in bad state                 |
| `UrateG`     | 0.04          | Unemployment rate in good state                |
| `RelProbBG`  | 0.75          | Relative probability of bad to good transition |
| `RelProbGB`  | 1.25          | Relative probability of good to bad transition |
| `DampingFac` | 0.5           | Damping factor for updating beliefs            |
| `act_T`      | 11000         | Total periods to simulate                      |
| `T_discard`  | 1000          | Periods to discard before computing moments    |

## Solution Algorithm

### Individual Problem

The `solve_KrusellSmith` function (line ~816) solves the individual agent's problem given perceived aggregate dynamics. The solution method exploits the special structure of the KS model:

1. **Pre-computation**: Many transition probabilities and next-period states can be pre-computed because of the discrete nature of aggregate and idiosyncratic shocks
2. **Bellman Iteration**: Solves the dynamic programming problem using backward induction
3. **Bilinear Interpolation**: Policy functions are represented as bilinear interpolants over (individual assets, aggregate resources)

### Equilibrium Computation

The `KrusellSmithEconomy.solve()` method finds the general equilibrium through the following algorithm:

1. **Initialize**: Start with initial guess for aggregate saving rule `AFunc` in each state
2. **Solve Micro Problem**: Each agent solves their problem given current `AFunc`
3. **Simulate**: Simulate the economy forward using agents' policy functions
4. **Regress**: Estimate new aggregate saving rule from simulated data
   - Form: `log(A'/M) = intercept + slope * log(M)`
   - Separate regression for each aggregate state (good/bad)
5. **Update**: Update `AFunc` with damping: `AFunc_new = lambda*AFunc_old + (1-lambda)*AFunc_estimated`
6. **Check Convergence**: Repeat until `AFunc` stabilizes

For a detailed implementation example, see the [KrusellSmith.ipynb](https://github.com/econ-ark/KrusellSmith/blob/master/Code/Python/KrusellSmith.ipynb) notebook which walks through each step with visualizations.

## Usage Examples

### Solving Individual Problem Only

```python
# First link to economy to get aggregate dynamics
agent.get_economy_data(economy)

# Construct solution inputs
agent.construct()

# Solve the individual problem
agent.solve()

# Access consumption function
# cFunc[state](individual_resources, aggregate_resources)
c = agent.solution[0].cFunc[0](10.0, economy.MSS)
print(f"Consumption at m=10, M=MSS in bad state: {c}")
```

### Simulating Individual Histories

```python
# Initialize simulation
agent.T_sim = 500
agent.track_vars = ['aNow', 'mNow', 'cNow', 'EmpNow']
agent.initialize_sim()

# Simulate
agent.simulate()

# Access simulated data
import numpy as np
mean_wealth = np.mean(agent.history['aNow'])
employment_rate = np.mean(agent.history['EmpNow'])
```

### Checking Convergence

```python
if economy.dynamics.distance < economy.tolerance:
    print("Converged!")
else:
    print(f"Not converged. Distance: {economy.dynamics.distance}")
```

### Visualizing Results

```python
import matplotlib.pyplot as plt
import numpy as np

agent.T_sim = 500
agent.track_vars = ['aNow']
agent.initialize_sim()
agent.simulate()

plt.plot(np.mean(agent.history['aNow'], axis=1))
plt.title('Mean Assets Over Time')
plt.xlabel('Time Period')
plt.ylabel('Mean Assets')
plt.show()
```

## Code Organization and Resources

### Main Components

| Component             | Location                                      | Line  | Purpose                    |
| --------------------- | --------------------------------------------- | ----- | -------------------------- |
| `KrusellSmithType`    | `HARK/ConsumptionSaving/ConsAggShockModel.py` | ~1615 | Agent class                |
| `KrusellSmithEconomy` | `HARK/ConsumptionSaving/ConsAggShockModel.py` | ~2962 | Economy class              |
| `solve_KrusellSmith`  | `HARK/ConsumptionSaving/ConsAggShockModel.py` | ~816  | Solution function          |
| `init_KS_agents`      | `HARK/ConsumptionSaving/ConsAggShockModel.py` | ~1596 | Default agent parameters   |
| `init_KS_economy`     | `HARK/ConsumptionSaving/ConsAggShockModel.py` | ~2935 | Default economy parameters |

### Helper Functions

- `construct()`: Builds solution infrastructure from parameters
- `get_economy_data()`: Imports aggregate dynamics from economy into agent
- `make_AggShkHist()`: Generates simulated history of aggregate shocks

### Essential Methods

**Agent Methods:**
```python
agent.get_economy_data(economy)  # Import aggregate dynamics
agent.construct()                # Build solution infrastructure
agent.solve()                    # Solve individual problem
agent.initialize_sim()           # Setup simulation
agent.simulate()                 # Simulate individual histories
```

**Economy Methods:**
```python
economy.make_AggShkHist()        # Generate aggregate shock history
economy.solve()                  # Find equilibrium
economy.reset()                  # Reset to initial state
```

## Tests

The test suite in `tests/ConsumptionSaving/test_ConsAggShockModel.py` includes comprehensive tests:

- **`KrusellSmithTestCase`** (line ~106): Base test case with setup/teardown
- **`KrusellSmithAgentTestCase`** (line ~123): Tests agent solution accuracy and consumption function values
- **`KrusellSmithMethodsTestCase`** (line ~135): Tests precomputation methods, construction and reset
- **`KrusellSmithEconomyTestCase`** (line ~245): Tests economy initialization and equilibrium dynamics

Run tests with:
```bash
# All KS tests
uv run pytest tests/ConsumptionSaving/test_ConsAggShockModel.py -k Krusell -v

# Specific test class
uv run pytest tests/ConsumptionSaving/test_ConsAggShockModel.py::KrusellSmithAgentTestCase -v

# Single test
uv run pytest tests/ConsumptionSaving/test_ConsAggShockModel.py::KrusellSmithAgentTestCase::test_agent -v
```

## Examples and Notebooks

### 1. Main Example Notebook

**Location:** `examples/ConsumptionSaving/example_ConsAggShockModel.ipynb`

- Set `solve_krusell_smith = True` in Cell 2 to run KS model
- Demonstrates both micro and macro solution
- Compares to other aggregate shock models
- Shows simulation and visualization

**For a complete replication:** See the [KrusellSmith.ipynb notebook](https://github.com/econ-ark/KrusellSmith/blob/master/Code/Python/KrusellSmith.ipynb) in the external REMARK repository, which provides a full implementation matching the original 1998 paper.

### 2. KS with Sequence Space Jacobian

**Location:** `examples/ConsNewKeynesianModel/KS-HARK-presentation.ipynb`

- Title: "Solving Krusell Smith Model with HARK and SSJ"
- Author: William Du
- Shows integration with SSJ toolkit
- Computes Jacobians for general equilibrium
- Modern approach to heterogeneous agent models

### 3. SSJ Explanation

**Location:** `examples/ConsNewKeynesianModel/SSJ_explanation.ipynb`

- Explains HARK-SSJ integration
- Advanced general equilibrium techniques
- References Krusell-Smith as example

### 4. Journey Notebooks

**Journey-PhD.ipynb:** `examples/Journeys/Journey-PhD.ipynb`

- Section 5.3: Tutorial reference
- Points to KS REMARK implementation
- Explains `CobbDouglasMarkovEconomy` usage

**Journey-Policymaker.ipynb:** `examples/Journeys/Journey-Policymaker.ipynb`

- References KS REMARK
- Discusses HARK-SSJ linkage
- Cites Krusell and Smith (1998) paper

## Related Models in HARK

HARK includes related aggregate shock models that generalize or simplify the Krusell-Smith framework:

### `AggShockConsumerType`

A more general aggregate shock consumer that allows for:

- Continuous distributions of income shocks
- Flexible aggregate production
- General equilibrium with `CobbDouglasEconomy`

### `AggShockMarkovConsumerType`

Similar to `AggShockConsumerType` but with multiple Markov states, solved in conjunction with `CobbDouglasMarkovEconomy`.

### `CobbDouglasEconomy` and `CobbDouglasMarkovEconomy`

More flexible economy classes that allow for:

- Multiple agent types
- Heterogeneous discount factors
- State-dependent or time-varying parameters

## Tips and Best Practices

### Computational Efficiency

**Start Small:** Use fewer agents and shorter simulations for initial exploration
```python
agent.AgentCount = 1000  # Instead of 5000
economy.act_T = 1100  # Instead of 11000
economy.max_loops = 2  # Quick convergence check
```

**Damping Factor:** Use appropriate damping to ensure convergence
```python
economy.DampingFac = 0.5  # Default is good starting point
```

**Grid Sizes:** Balance accuracy and speed

- `aCount`: More points = more accuracy but slower
- `MaggCount`: Critical for capturing aggregate dynamics

### Debugging

**Verbose Output:** Enable detailed logging
```python
economy.verbose = True
```

**Check Equilibrium:** After solving, verify convergence
```python
print(f"Converged: {economy.dynamics.distance < economy.tolerance}")
print(f"Final distance: {economy.dynamics.distance}")
```

**Examine History:** Plot simulated paths
```python
import matplotlib.pyplot as plt
plt.plot(economy.history['Mnow'])
plt.title('Aggregate Market Resources')
plt.show()
```

### Common Issues and Solutions

| Issue            | Solution                                                 |
| ---------------- | -------------------------------------------------------- |
| Non-convergence  | Increase `max_loops`, adjust `DampingFac` (0.3-0.7)      |
| Slow computation | Reduce `AgentCount`, `aCount`, `MaggCount`               |
| Memory error     | Reduce `AgentCount`, limit `track_vars`                  |
| Explosive paths  | Check parameters (esp. `DiscFac`, `Rfree`), verify grids |

## Advanced Topics

### Integration with Sequence Space Jacobian

HARK's Krusell-Smith implementation can be integrated with the Sequence Space Jacobian toolkit for advanced general equilibrium analysis. See `examples/ConsNewKeynesianModel/KS-HARK-presentation.ipynb` for a complete example.

Key steps:

1. Solve KS model in HARK for steady state
2. Compute heterogeneous agent Jacobians
3. Combine with aggregate block (firm, market clearing)
4. Analyze impulse responses to shocks

### Extending the Model

The modular structure allows for extensions:

- **Different production functions**: Modify `CobbDouglasEconomy` methods
- **Alternative aggregate dynamics**: Subclass `KrusellSmithEconomy`
- **Richer idiosyncratic shocks**: Adjust income process in agent's constructor
- **Multiple agent types**: Pass list of different `KrusellSmithType` agents to economy
- **Deep learning approaches**: See [Main_KS.ipynb](https://github.com/marcmaliar/deep-learning-euler-method-krusell-smith/blob/main/code/python/Main_KS.ipynb) for neural network-based solution methods

### Custom Parameters

```python
# Modify specific parameters
agent = KrusellSmithType(
    DiscFac=0.98,
    CRRA=2.0,
    AgentCount=10000
)

economy = KrusellSmithEconomy(
    agents=[agent],
    max_loops=20,
    DampingFac=0.3
)
```

## References

### Primary Reference

Krusell, P., & Smith, Jr, A. A. (1998). "Income and wealth heterogeneity in the macroeconomy." Journal of Political Economy, 106(5), 867-896. DOI: 10.1086/250034
https://www.journals.uchicago.edu/doi/abs/10.1086/250034

### Related Literature

Auclert, A., Bardoczy, B., Rognlie, M., & Straub, L. (2021). "Using the sequence-space jacobian to solve and estimate heterogeneous-agent models." Econometrica, 89(5), 2375-2408. DOI: 10.3982/ECTA17434
https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434

### HARK Documentation

- A Gentle Introduction to HARK: https://docs.econ-ark.org/examples/Gentle-Intro/Gentle-Intro-To-HARK.html
- ConsAggShockModel API Reference: https://docs.econ-ark.org/reference/ConsumptionSaving/ConsAggShockModel.html
- Market Class Documentation: https://docs.econ-ark.org/reference/core.html#market

### External Resources

**KrusellSmith REMARK Repository:** https://github.com/econ-ark/KrusellSmith

This is a complete replication of Krusell and Smith (1998) using HARK. The repository contains:

- Python implementation in `Code/Python/KrusellSmith.py`
- Interactive Jupyter notebook with full replication (`Code/Python/KrusellSmith.ipynb`)
- Reproducible results using `nbreproduce` (Docker-based) or conda environment
- Figures and output matching the original paper
- Complete bibliography and citations

The notebook demonstrates the complete workflow including:
- Setting up the economy with two aggregate states (good and bad)
- Configuring heterogeneous agents with different employment shock processes
- Solving for the equilibrium aggregate saving rules via simulation
- Comparing simulated moments to the original Krusell-Smith results
- Visualizing policy functions and aggregate dynamics

This REMARK (Replications/Reproductions and Explorations Made using ARK) provides a working example of how to implement and solve the full Krusell-Smith model with HARK, making it an excellent resource for learning the complete workflow from model specification to result generation.

**Deep Learning Euler Method for Krusell-Smith:** https://github.com/marcmaliar/deep-learning-euler-method-krusell-smith/

An alternative solution approach by Marc Maliar that solves the Krusell-Smith model using deep learning combined with the Euler equation method. This repository demonstrates:

- Modern machine learning techniques applied to heterogeneous agent models
- Neural network approximation of policy functions
- Euler equation accuracy as the solution criterion
- Interactive Jupyter notebook (`code/python/Main_KS.ipynb`) with complete implementation
- Comparison of deep learning results to traditional solution methods
- Reproducible via Binder or local execution

The [Main_KS.ipynb notebook](https://github.com/marcmaliar/deep-learning-euler-method-krusell-smith/blob/main/code/python/Main_KS.ipynb) walks through:
- Setting up the Krusell-Smith economy parameters
- Training neural networks to approximate decision rules
- Evaluating solution accuracy using Euler equation errors
- Benchmarking computational efficiency

This approach represents an innovative alternative to traditional value function iteration or policy function iteration methods, showing how deep learning can be applied to solve complex macroeconomic models efficiently.

**Additional Resources:**

- REMARK Index: https://github.com/econ-ark/REMARK/blob/master/REMARKs/KrusellSmith.md
- KS-HARK-presentation (online): https://docs.econ-ark.org/examples/ConsNewKeynesianModel/KS-HARK-presentation.html
- Econ-ARK Main Site: https://econ-ark.org
- HARK GitHub Repository: https://github.com/econ-ark/HARK

## Support and Community

- GitHub Issues: https://github.com/econ-ark/HARK/issues
- GitHub Discussions: https://github.com/econ-ark/HARK/discussions
- Econ-ARK Forum: https://econ-ark.org
- Documentation: https://docs.econ-ark.org

## Summary

The Krusell-Smith model in HARK provides:

- Faithful implementation of the seminal KS (1998) model
- Efficient solution methods exploiting model structure
- Flexible equilibrium computation with damping and convergence checks
- Integration capabilities with modern tools (SSJ)
- Extensive documentation and examples
- Tested codebase with comprehensive test suite

For questions or contributions, visit the Econ-ARK GitHub repository or join the discussion on the Econ-ARK forum.

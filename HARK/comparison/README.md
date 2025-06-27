# HARK Model Comparison Infrastructure

This module provides a unified framework for comparing different solution methods for heterogeneous agent models, enabling systematic evaluation of computational approaches.

## Overview

The comparison infrastructure addresses a key need in computational economics: comparing different solution methods (traditional dynamic programming, deep learning, SSJ, etc.) for the same economic model. It provides:

- **Unified parameter specification** across different solution methods
- **Automatic parameter translation** between method-specific formats
- **Standardized interfaces** through adapter pattern
- **Common metrics** for solution quality assessment
- **Integrated simulation capabilities**
- **Automated comparison reports**

## Key Components

### 1. ModelComparison (`base.py`)

The main orchestrator class that manages the comparison workflow:

```python
from HARK.comparison import ModelComparison

# Define economic primitives
primitives = {
    'CRRA': 2.0,
    'DiscFac': 0.96,
    'Rfree': 1.03,
    # ... other parameters
}

# Create comparison object
comp = ModelComparison(
    model_description_md="# My Model Description",
    primitives=primitives
)

# Add solution methods
comp.add_solution_method('method_name', method_config)

# Solve and compare
comp.solve('method_name')
results = comp.compare_solutions()

# NEW: Baseline solution functionality
# Save a benchmark solution
comp.save_baseline_solution('benchmark_method', 'benchmark.pkl')

# Load benchmark in new session
comp.load_baseline_solution('benchmark.pkl', 'loaded_benchmark')

# Compare methods against baseline
baseline_comparison = comp.compare_against_baseline('loaded_benchmark')
```

### 2. EconomicMetrics (`metrics.py`)

Implements standard metrics for evaluating solution quality:

- **Euler equation errors** - Following Maliar, Maliar & Winant (2021)
- **Bellman equation errors** - Value function accuracy
- **Den Haan-Marcet statistics** - Forecast accuracy
- **Wealth distribution metrics** - Gini coefficient, percentiles, shares
- **Convergence metrics** - For iterative methods

### 3. ParameterTranslator (`parameter_translation.py`)

Handles automatic translation between different parameter naming conventions:

- HARK: `DiscFac`, `CRRA`, `Rfree`
- SSJ: `beta`, `eis`, `r_ss`
- Maliar: `beta`, `gamma`, `R`

### 4. Solution Adapters (`adapters/`)

Standardized interfaces for different solution methods:

- **HARKAdapter**: Native HARK methods (ConsIndShockModel, ConsAggShockModel)
- **SSJAdapter**: Sequence Space Jacobian methods
- **MaliarAdapter**: Deep learning methods (Euler, Bellman, reward)
- **ExternalAdapter**: Load solutions from files/URLs
- **AiyagariAdapter**: Specialized for Aiyagari equilibrium models

## Baseline Solutions

The comparison framework supports saving and loading baseline solutions for efficient benchmarking. This is particularly useful when you want to test multiple methods (K methods) against a single benchmark without re-solving the benchmark each time.

### Key Benefits

1. **Efficiency**: Solve expensive benchmark once, reuse multiple times
2. **Consistency**: All comparisons use exactly the same baseline
3. **Reproducibility**: Baselines include all necessary information
4. **Flexibility**: Load different baselines for different comparisons

### Baseline Workflow

```python
# 1. Solve and save benchmark
comp.add_solution_method('benchmark', high_accuracy_config)
comp.solve('benchmark')
comp.save_baseline_solution('benchmark', 'my_benchmark.pkl')

# 2. In new session - load benchmark and test methods
comp = ModelComparison(description, primitives)
comp.load_baseline_solution('my_benchmark.pkl', 'baseline')

# Add and solve test methods
comp.add_solution_method('fast_method_1', config1)
comp.add_solution_method('fast_method_2', config2)
comp.solve('fast_method_1')
comp.solve('fast_method_2')

# 3. Compare against baseline
comparison = comp.compare_against_baseline('baseline')
print(comparison[['method', 'computation_time', 'time_ratio', 'gini_rel_diff']])
```

### Managing Baselines

```python
# List loaded baselines
baselines_df = comp.list_baselines()

# Clear baselines from memory
comp.clear_baselines(['baseline_name'])

# Load multiple baselines
comp.load_baseline_solution('benchmark1.pkl', 'bench1')
comp.load_baseline_solution('benchmark2.pkl', 'bench2')
```

## Usage Examples

### Comparing Krusell-Smith Solutions

```python
from HARK.comparison import ModelComparison

# Define Krusell-Smith primitives
ks_primitives = {
    'DiscFac': 0.99,
    'CRRA': 1.0,
    'CapShare': 0.36,
    'DeprFac': 0.025,
    # ... aggregate states and transitions
}

# Create comparison
ks_comp = ModelComparison("Krusell-Smith Model", ks_primitives)

# Add methods
ks_comp.add_solution_method('krusell_smith/HARK', {
    'AgentCount': 10000,
    'tolerance': 0.0001
})

ks_comp.add_solution_method('maliar_winant_euler', {
    'nn_layers': [64, 64, 32],
    'epochs': 100
})

# Solve all methods
for method in ['krusell_smith/HARK', 'maliar_winant_euler']:
    ks_comp.solve(method)

# Compare results
comparison_df = ks_comp.compare_solutions(
    save_report=True,
    report_path="ks_comparison.md"
)
```

### Loading External Solutions

```python
# From file
comp.add_solution_method('external', {
    'solution_path': 'path/to/solution.pkl'
})

# From URL
comp.add_solution_method('external', {
    'github_url': 'https://github.com/.../solution.json'
})

# Direct data
comp.add_solution_method('external', {
    'solution_data': {
        'cFunc': consumption_function,
        'vFunc': value_function,
        'AFunc': aggregate_law
    }
})
```

## Metrics

### Euler Equation Errors

Measures how well the consumption policy satisfies the Euler equation:

```
E[β * R * (c_t+1/c_t)^(-γ)] = 1
```

### Den Haan-Marcet Statistic

Tests forecast accuracy of aggregate laws of motion using R² of:

```
K_{t+1}^actual vs K_{t+1}^forecast
```

### Wealth Distribution

- Gini coefficient
- Percentiles (10th, 25th, 50th, 75th, 90th, 95th, 99th)
- Top wealth shares (top 10%, top 1%)
- Moments (mean, std, skewness, kurtosis)

## Requirements

- HARK >= 0.13.0
- NumPy
- Pandas
- SciPy

Optional:
- PyTorch (for deep learning methods)
- sequence-jacobian (for SSJ methods)
- tabulate (for markdown reports)

## Installation

The comparison module is included with HARK. To use all features:

```bash
pip install econ-ark[comparison]
# or
conda install -c conda-forge econ-ark pytorch sequence-jacobian
```

## Testing

Run tests with coverage:

```bash
pytest HARK/comparison --cov=HARK/comparison
```

## Future Extensions

1. **Additional solution methods**:
   - Endogenous grid method variants
   - Projection methods
   - Machine learning approaches

2. **Enhanced metrics**:
   - Policy function smoothness
   - Computational efficiency benchmarks
   - Memory usage profiling

3. **Visualization tools**:
   - Policy function plots
   - Convergence diagnostics
   - Interactive comparison dashboards

4. **Model library**:
   - Pre-configured standard models
   - Reference implementations
   - Benchmark solutions

## Contributing

To add a new solution method:

1. Create adapter in `adapters/`
2. Implement required interface methods
3. Add to adapter factory in `base.py`
4. Add parameter mappings to `parameter_translation.py`
5. Write tests in `tests/`

## References

- Maliar, L., Maliar, S., & Winant, P. (2021). "Deep learning for solving dynamic economic models." Journal of Monetary Economics.
- Auclert, A., Bardóczy, B., Rognlie, M., & Straub, L. (2021). "Using the sequence-space Jacobian to solve and estimate heterogeneous-agent models."
- Krusell, P., & Smith Jr, A. A. (1998). "Income and wealth heterogeneity in the macroeconomy." Journal of Political Economy.
- Aiyagari, S. R. (1994). "Uninsured idiosyncratic risk and aggregate saving." Quarterly Journal of Economics.

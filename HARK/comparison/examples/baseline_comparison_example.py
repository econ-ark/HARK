"""
Example: Using Baseline Solutions for Model Comparison

This script demonstrates how to save and load baseline solutions for efficient
comparison testing. This is particularly useful when you want to test K methods
against a benchmark without re-solving the benchmark each time.
"""

from HARK.comparison import ModelComparison

# Define model primitives (using a simple consumption-saving model example)
model_primitives = {
    # Preferences
    "DiscFac": 0.96,  # Discount factor
    "CRRA": 2.0,  # Risk aversion
    "LivPrb": 1.0,  # Survival probability
    # Income process
    "PermShkStd": 0.1,  # Permanent shock std dev
    "TranShkStd": 0.2,  # Transitory shock std dev
    "UnempPrb": 0.05,  # Unemployment probability
    "IncUnemp": 0.3,  # Unemployment benefit
    # Borrowing constraint
    "BoroCnstArt": 0.0,  # Borrowing constraint
    # Interest rate
    "Rfree": 1.03,
}

# Create model comparison object
comparison = ModelComparison(
    model_description_md="""
    # Consumption-Saving Model Comparison

    This example demonstrates baseline solution functionality for comparing
    different solution methods for a consumption-saving model with:
    - Permanent and transitory income shocks
    - Unemployment risk
    - Borrowing constraints
    """,
    primitives=model_primitives,
)

# Add different solution method configurations
print("Setting up solution methods...")

# Benchmark method (high accuracy, slower)
comparison.add_solution_method(
    "benchmark_hark",
    {
        "AgentCount": 10000,
        "T_cycle": 1,
        "cycles": 0,  # Infinite horizon
        "TranShkCount": 7,
        "PermShkCount": 7,
        "aXtraMin": 0.001,
        "aXtraMax": 50.0,
        "aXtraCount": 48,
        "aXtraNestFac": 3,
        "vFuncBool": True,
        "CubicBool": True,  # High accuracy
    },
)

# Fast method 1 (fewer grid points)
comparison.add_solution_method(
    "fast_hark_1",
    {
        "AgentCount": 5000,
        "T_cycle": 1,
        "cycles": 0,
        "TranShkCount": 5,
        "PermShkCount": 5,
        "aXtraMin": 0.001,
        "aXtraMax": 30.0,
        "aXtraCount": 24,  # Fewer grid points
        "aXtraNestFac": 2,
        "vFuncBool": False,  # Skip value function
        "CubicBool": False,  # Linear interpolation
    },
)

# Fast method 2 (even fewer grid points)
comparison.add_solution_method(
    "fast_hark_2",
    {
        "AgentCount": 2000,
        "T_cycle": 1,
        "cycles": 0,
        "TranShkCount": 3,
        "PermShkCount": 3,
        "aXtraMin": 0.001,
        "aXtraMax": 20.0,
        "aXtraCount": 16,  # Even fewer grid points
        "aXtraNestFac": 2,
        "vFuncBool": False,
        "CubicBool": False,
    },
)

# ============================================================================
# STEP 1: Solve and save the benchmark (baseline) solution
# ============================================================================

print("\n" + "=" * 60)
print("STEP 1: Solving and saving benchmark solution")
print("=" * 60)

# Solve the benchmark method
print("Solving benchmark method...")
try:
    benchmark_solution = comparison.solve("benchmark_hark", verbose=True)
    print("✓ Benchmark solution completed successfully!")

    # Compute metrics for the benchmark
    print("Computing metrics for benchmark...")
    comparison.compute_metrics("benchmark_hark")

    # Run simulation for benchmark
    print("Running simulation for benchmark...")
    comparison.simulate_policy("benchmark_hark", periods=500, num_agents=5000, seed=42)

    # Save the benchmark solution
    print("Saving benchmark solution...")
    baseline_file = comparison.save_baseline_solution(
        "benchmark_hark", filepath="baselines/benchmark_solution.pkl"
    )
    print(f"✓ Benchmark saved to: {baseline_file}")

except Exception as e:
    print(f"✗ Benchmark solution failed: {e}")
    exit(1)

# ============================================================================
# STEP 2: Clear solutions and test loading baseline
# ============================================================================

print("\n" + "=" * 60)
print("STEP 2: Testing baseline loading")
print("=" * 60)

# Clear current solutions to simulate fresh start
print("Clearing current solutions to simulate fresh session...")
comparison.solutions.clear()
comparison.metrics.clear()
comparison.adapters.clear()
comparison.simulation_results.clear()
comparison.computation_times.clear()

# Load the baseline solution
print("Loading baseline solution...")
baseline_data = comparison.load_baseline_solution(
    "baselines/benchmark_solution.pkl", baseline_name="loaded_benchmark"
)
print("✓ Baseline loaded successfully!")

# List loaded baselines
print("\nLoaded baselines:")
baselines_df = comparison.list_baselines()
if not baselines_df.empty:
    print(baselines_df.to_string(index=False))

# ============================================================================
# STEP 3: Solve other methods and compare against baseline
# ============================================================================

print("\n" + "=" * 60)
print("STEP 3: Solving other methods and comparing")
print("=" * 60)

# Solve the fast methods
methods_to_test = ["fast_hark_1", "fast_hark_2"]

for method in methods_to_test:
    print(f"\nSolving {method}...")
    try:
        comparison.solve(method, verbose=True)
        print(f"✓ {method} completed!")
    except Exception as e:
        print(f"✗ {method} failed: {e}")
        continue

# Run simulations for all methods
print("\nRunning simulations for comparison methods...")
comparison.simulate_policy(
    periods=500,
    num_agents=5000,
    seed=42,  # Same seed for fair comparison
)

# Compute metrics for all methods
print("Computing metrics for all methods...")
comparison.compute_metrics()

# Compare against baseline
print("\nComparing methods against baseline...")
comparison_df = comparison.compare_against_baseline(
    "loaded_benchmark",
    methods=methods_to_test,
    metrics_to_compare=["gini", "mean_wealth", "welfare_equivalent"],
)

if not comparison_df.empty:
    print("\nComparison Results:")
    print("=" * 50)

    # Display key comparison metrics
    display_cols = [
        "method",
        "computation_time",
        "time_ratio",
        "gini",
        "gini_rel_diff",
        "mean_wealth",
        "mean_wealth_rel_diff",
    ]

    # Filter to available columns
    available_cols = [col for col in display_cols if col in comparison_df.columns]
    if available_cols:
        print(comparison_df[available_cols].round(4).to_string(index=False))
    else:
        print(comparison_df.to_string(index=False))

    # Summary statistics
    print("\nSummary:")
    print("-" * 30)
    for _, row in comparison_df.iterrows():
        method = row["method"]
        time_ratio = row.get("time_ratio", "N/A")
        if time_ratio != "N/A":
            print(f"{method}: {time_ratio:.2f}x faster than baseline")
        else:
            print(f"{method}: computation time comparison not available")

# ============================================================================
# STEP 4: Demonstrate saving multiple baselines
# ============================================================================

print("\n" + "=" * 60)
print("STEP 4: Demonstrating multiple baselines")
print("=" * 60)

# Save one of the fast methods as another baseline
if "fast_hark_1" in comparison.solutions:
    print("Saving fast_hark_1 as an alternative baseline...")
    comparison.save_baseline_solution(
        "fast_hark_1", filepath="baselines/fast_baseline.pkl"
    )

    # Load it back with a different name
    comparison.load_baseline_solution(
        "baselines/fast_baseline.pkl", baseline_name="fast_baseline"
    )

    print("\nAll loaded baselines:")
    print(comparison.list_baselines().to_string(index=False))

# ============================================================================
# STEP 5: Demonstrate workflow for testing K methods against benchmark
# ============================================================================

print("\n" + "=" * 60)
print("STEP 5: Simulating workflow for testing K methods")
print("=" * 60)

print("Simulating a fresh session where we test multiple methods...")

# Create a new comparison instance to simulate fresh start
fresh_comparison = ModelComparison(
    model_description_md="Fresh comparison session", primitives=model_primitives
)

# Load the benchmark
print("Loading benchmark in fresh session...")
fresh_comparison.load_baseline_solution(
    "baselines/benchmark_solution.pkl", baseline_name="benchmark"
)

# Add and solve new methods to test
test_methods = {
    "test_method_A": {
        "AgentCount": 3000,
        "T_cycle": 1,
        "cycles": 0,
        "TranShkCount": 4,
        "PermShkCount": 4,
        "aXtraMin": 0.001,
        "aXtraMax": 25.0,
        "aXtraCount": 20,
        "aXtraNestFac": 2,
        "vFuncBool": False,
        "CubicBool": False,
    },
    "test_method_B": {
        "AgentCount": 1000,
        "T_cycle": 1,
        "cycles": 0,
        "TranShkCount": 3,
        "PermShkCount": 3,
        "aXtraMin": 0.001,
        "aXtraMax": 15.0,
        "aXtraCount": 12,
        "aXtraNestFac": 1,
        "vFuncBool": False,
        "CubicBool": False,
    },
}

# Add and solve test methods
for method_name, config in test_methods.items():
    print(f"Testing {method_name}...")
    fresh_comparison.add_solution_method(method_name, config)
    try:
        fresh_comparison.solve(method_name, verbose=False)
        print(f"✓ {method_name} solved successfully")
    except Exception as e:
        print(f"✗ {method_name} failed: {e}")

# Compare all test methods against the loaded benchmark
if fresh_comparison.solutions:
    print("\nComparing test methods against loaded benchmark...")
    test_comparison = fresh_comparison.compare_against_baseline("benchmark")

    if not test_comparison.empty:
        print(
            test_comparison[["method", "computation_time", "time_ratio"]].to_string(
                index=False
            )
        )

# ============================================================================
# CLEANUP
# ============================================================================

print("\n" + "=" * 60)
print("CLEANUP")
print("=" * 60)

# Demonstrate clearing baselines
print("Clearing loaded baselines...")
comparison.clear_baselines()

print("\nBaselines after clearing:")
print(comparison.list_baselines())

print("\nExample completed successfully!")
print("=" * 60)

# Summary of key benefits
print("""
KEY BENEFITS OF BASELINE FUNCTIONALITY:

1. EFFICIENCY: Solve expensive benchmark once, reuse multiple times
2. CONSISTENCY: All comparisons use exactly the same baseline
3. REPRODUCIBILITY: Baselines include all necessary information
4. FLEXIBILITY: Load different baselines for different comparisons
5. WORKFLOW: Supports testing K methods against benchmark efficiently

TYPICAL WORKFLOW:
1. Solve and save your benchmark: comparison.save_baseline_solution()
2. Load benchmark in new session: comparison.load_baseline_solution()
3. Solve your K test methods: comparison.solve()
4. Compare against baseline: comparison.compare_against_baseline()
""")

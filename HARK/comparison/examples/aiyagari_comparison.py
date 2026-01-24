"""
Example: Comparing Aiyagari Model Solutions

This script demonstrates how to use the model comparison infrastructure
to compare different solution methods for the Aiyagari (1994) model.
"""

import numpy as np
from HARK.comparison import ModelComparison

# Define Aiyagari model primitives
aiyagari_primitives = {
    # Preferences
    "DiscFac": 0.96,  # Discount factor
    "CRRA": 2.0,  # Risk aversion
    "LivPrb": 1.0,  # Survival probability (1 = no death)
    # Production
    "CapShare": 0.36,  # Capital share in production
    "DeprFac": 0.08,  # Depreciation rate
    "LbrInd": 1.0,  # Labor supply (normalized)
    # Income process
    "PermShkStd": 0.0,  # No permanent shocks in basic Aiyagari
    "TranShkStd": 0.2,  # Transitory income shock std dev
    "UnempPrb": 0.0,  # No unemployment in basic model
    "IncUnemp": 0.0,  # Unemployment benefit
    # Borrowing constraint
    "BoroCnstArt": 0.0,  # Borrowing constraint (0 = no borrowing)
    # Initial interest rate guess
    "Rfree": 1.03,  # Will be determined in equilibrium
}

# Create model comparison object
aiyagari_comparison = ModelComparison(
    model_description_md="""
    # Aiyagari (1994) Model

    The Aiyagari model is the canonical incomplete markets model with:
    - Idiosyncratic income risk (no aggregate risk)
    - Borrowing constraints
    - General equilibrium with endogenous interest rate

    This comparison evaluates different solution approaches:
    1. HARK native implementation
    2. External implementations (e.g., from QuantEcon)
    3. Alternative numerical methods
    """,
    primitives=aiyagari_primitives,
)

# Add method-specific configurations
print("Adding solution method configurations...")

# 1. HARK native method - find equilibrium
aiyagari_comparison.add_solution_method(
    "aiyagari/HARK",
    {
        "find_equilibrium": True,  # Find equilibrium interest rate
        "r_min": 0.001,  # Min interest rate to search
        "r_max": 0.04,  # Max interest rate to search
        "equilibrium_tol": 1e-4,  # Tolerance for equilibrium
        # Agent configuration
        "AgentCount": 10000,  # Number of agents for distribution
        "T_cycle": 1,  # Infinite horizon
        "cycles": 0,  # Infinite horizon
        # Income process discretization
        "TranShkCount": 7,  # Number of transitory shock points
        "PermShkCount": 1,  # No permanent shocks
        # Asset grid
        "aXtraMin": 0.001,  # Minimum assets
        "aXtraMax": 50.0,  # Maximum assets
        "aXtraCount": 48,  # Number of grid points
        "aXtraNestFac": 3,  # Nesting factor for grid
        # Computational options
        "vFuncBool": True,  # Compute value function
        "CubicBool": False,  # Use linear interpolation
    },
)

# 2. HARK at fixed interest rate (partial equilibrium)
aiyagari_comparison.add_solution_method(
    "aiyagari/HARK_partial",
    {
        "find_equilibrium": False,  # Use fixed interest rate
        "Rfree": 1.02,  # Fixed interest rate
        "compute_distribution": True,
        # Same agent configuration
        "AgentCount": 10000,
        "T_cycle": 1,
        "cycles": 0,
        "TranShkCount": 7,
        "PermShkCount": 1,
        "aXtraMin": 0.001,
        "aXtraMax": 50.0,
        "aXtraCount": 48,
        "aXtraNestFac": 3,
        "vFuncBool": True,
        "CubicBool": False,
    },
)

# 3. External solution (e.g., from QuantEcon or other source)
aiyagari_comparison.add_solution_method(
    "external",
    {
        # For demonstration, create mock solution data
        "solution_data": {
            "description": "Mock Aiyagari solution from external source",
            "r_equilibrium": 0.02,
            "K_equilibrium": 3.5,
            "method": "endogenous_grid_method",
            "source": "QuantEcon implementation",
            # Mock policy function data
            "cFunc": {
                "grid": np.linspace(0.1, 20.0, 100),
                "values": 0.1 * np.linspace(0.1, 20.0, 100),  # Mock linear policy
            },
            # Mock distribution data
            "asset_distribution": np.random.lognormal(1.0, 0.5, 10000),
        }
    },
)

# Solve using available methods
print("\nSolving model with different methods...")

# Try HARK equilibrium method
try:
    print("\n1. Solving for equilibrium with HARK...")
    hark_eq_solution = aiyagari_comparison.solve("aiyagari/HARK", verbose=True)
    print("   HARK equilibrium solution completed!")
    if "r_equilibrium" in hark_eq_solution:
        print(f"   Equilibrium interest rate: {hark_eq_solution['r_equilibrium']:.4f}")
        print(f"   Equilibrium capital: {hark_eq_solution['K_equilibrium']:.2f}")
except Exception as e:
    print(f"   HARK equilibrium solution failed: {e}")

# Try HARK partial equilibrium
try:
    print("\n2. Solving at fixed interest rate with HARK...")
    hark_partial_solution = aiyagari_comparison.solve(
        "aiyagari/HARK_partial", verbose=True
    )
    print("   HARK partial equilibrium solution completed!")
except Exception as e:
    print(f"   HARK partial solution failed: {e}")

# Load external solution
try:
    print("\n3. Loading external solution...")
    ext_solution = aiyagari_comparison.solve("external", verbose=True)
    print("   External solution loaded!")
except Exception as e:
    print(f"   External solution failed: {e}")

# Run simulations
print("\n\nRunning simulations...")
simulation_results = aiyagari_comparison.simulate_policy(
    periods=500,  # Shorter for stationary model
    num_agents=10000,
    burn_in=100,
    seed=42,
)

# Compute metrics
print("\nComputing solution metrics...")
metrics = aiyagari_comparison.compute_metrics()

# Focus on wealth distribution metrics for Aiyagari
print("\nWealth Distribution Metrics:")
print("-" * 40)
for method, method_metrics in metrics.items():
    print(f"\n{method}:")
    if "gini" in method_metrics:
        print(f"  Gini coefficient: {method_metrics['gini']:.3f}")
    if "mean_wealth" in method_metrics:
        print(f"  Mean wealth: {method_metrics['mean_wealth']:.2f}")
    if "wealth_share_top10" in method_metrics:
        print(f"  Top 10% wealth share: {method_metrics['wealth_share_top10']:.1%}")

# Generate comparison report
print("\nGenerating comparison report...")
comparison_df = aiyagari_comparison.compare_solutions(
    save_report=True, report_path="aiyagari_comparison_report.md"
)

# Display results
print("\nComparison Results:")
print("=" * 50)

if len(comparison_df) > 0:
    # Select key columns for display
    display_cols = ["method", "gini", "mean_wealth", "computation_time"]
    display_cols = [col for col in display_cols if col in comparison_df.columns]
    if display_cols:
        print(comparison_df[display_cols].to_string())
else:
    print("No solutions available for comparison.")

# Demonstrate accessing individual solutions
print("\n\nAccessing individual solution components:")
for method_name, solution in aiyagari_comparison.solutions.items():
    print(f"\n{method_name}:")
    if "r_equilibrium" in solution:
        print(f"  Equilibrium r: {solution['r_equilibrium']:.4f}")
    if "K_equilibrium" in solution:
        print(f"  Equilibrium K: {solution['K_equilibrium']:.2f}")

    # Get policy function
    adapter = aiyagari_comparison.adapters[method_name]
    c_func = adapter.get_consumption_policy()
    if c_func is not None:
        # Evaluate at a test point
        test_m = 5.0
        test_c = c_func(test_m)
        print(f"  c({test_m:.1f}) = {test_c:.3f}")

print("\n\nKey insights from Aiyagari model comparison:")
print("1. Equilibrium interest rate balances supply and demand for capital")
print("2. Wealth distribution depends on income risk and borrowing constraints")
print("3. Different solution methods should yield similar equilibrium outcomes")
print("4. Computational efficiency varies across methods")
print("5. Accuracy can be assessed via Euler equation errors")

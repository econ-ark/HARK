"""
Example: Comparing Krusell-Smith Model Solutions

This script demonstrates how to use the model comparison infrastructure
to compare different solution methods for the Krusell-Smith model.
"""

from HARK.comparison import ModelComparison

# Define Krusell-Smith model primitives
ks_primitives = {
    # Preferences
    "DiscFac": 0.99,  # Discount factor
    "CRRA": 1.0,  # Risk aversion
    # Production
    "CapShare": 0.36,  # Capital share in production
    "DeprFac": 0.025,  # Depreciation rate
    "LbrInd": 1.0,  # Labor supply (normalized)
    # Aggregate states
    "ProdB": 0.99,  # Productivity in bad state
    "ProdG": 1.01,  # Productivity in good state
    "UrateB": 0.10,  # Unemployment rate in bad state
    "UrateG": 0.04,  # Unemployment rate in good state
    # Transition probabilities
    "DurMeanB": 8.0,  # Mean duration of bad state
    "DurMeanG": 8.0,  # Mean duration of good state
    "SpellMeanB": 2.5,  # Mean unemployment spell in bad state
    "SpellMeanG": 1.5,  # Mean unemployment spell in good state
}

# Create model comparison object
ks_comparison = ModelComparison(
    model_description_md="""
    # Krusell-Smith (1998) Model

    The Krusell-Smith model is a heterogeneous agent model with:
    - Idiosyncratic employment risk
    - Aggregate productivity shocks
    - Incomplete markets (only risk-free bond)
    - Bounded rationality (agents use simple forecasting rule)

    This comparison evaluates different solution methods:
    1. HARK native implementation (fixed-point iteration)
    2. Maliar, Maliar & Winant (2021) deep learning methods
    3. Sequence Space Jacobian (if available)
    """,
    primitives=ks_primitives,
)

# Add method-specific configurations
print("Adding solution method configurations...")

# 1. HARK native method
ks_comparison.add_solution_method(
    "krusell_smith/HARK",
    {
        "AgentCount": 10000,  # Number of agents to simulate
        "act_T": 11000,  # Total simulation periods
        "T_discard": 1000,  # Periods to discard (burn-in)
        "tolerance": 0.0001,  # Convergence tolerance
        "DampingFac": 0.5,  # Damping for fixed-point iteration
        # Asset grid
        "aXtraMin": 0.001,
        "aXtraMax": 50.0,
        "aXtraCount": 48,
        "aXtraNestFac": 3,
        # Transition matrix adjustment
        "RelProbBG": 0.75,  # Relative prob of B->G transition
        "RelProbGB": 1.25,  # Relative prob of G->B transition
    },
)

# 2. Deep learning methods (placeholders - need actual implementations)
# Note: These would require the actual Maliar et al. code to be available
ks_comparison.add_solution_method(
    "maliar_winant_euler",
    {
        "n_agents": 10000,
        "n_periods": 1000,
        "nn_layers": [64, 64, 32],
        "activation": "relu",
        "learning_rate": 0.001,
        "batch_size": 256,
        "epochs": 100,
    },
)

ks_comparison.add_solution_method(
    "maliar_winant_bellman",
    {
        "n_agents": 10000,
        "n_periods": 1000,
        "nn_layers": [64, 64, 32],
        "activation": "relu",
        "learning_rate": 0.001,
        "batch_size": 256,
        "epochs": 100,
    },
)

# 3. External solution (if available from file or URL)
# This demonstrates loading a pre-computed solution
ks_comparison.add_solution_method(
    "external",
    {
        # Could specify path to solution file
        # 'solution_path': 'path/to/ks_solution.pkl',
        # Or GitHub URL
        # 'github_url': 'https://github.com/.../ks_solution.json',
        # For demo, we'll create mock data
        "solution_data": {
            "description": "Mock external KS solution for demonstration",
            "AFunc": [0.995, -0.05],  # Simple linear forecast rule coefficients
            "method": "value_function_iteration",
            "source": "External implementation",
        }
    },
)

# Solve using available methods
print("\nSolving model with different methods...")

# Try HARK method (this would work with full HARK installation)
try:
    print("\n1. Solving with HARK native method...")
    hark_solution = ks_comparison.solve("krusell_smith/HARK", verbose=True)
    print("   HARK solution completed!")
except Exception as e:
    print(f"   HARK solution failed: {e}")
    print("   (This is expected if ConsAggShockModel is not fully set up)")

# Try deep learning methods (would need actual implementations)
for method in ["maliar_winant_euler", "maliar_winant_bellman"]:
    try:
        print(f"\n2. Solving with {method}...")
        dl_solution = ks_comparison.solve(method, verbose=True)
        print(f"   {method} solution completed!")
    except ImportError as e:
        print(f"   {method} requires PyTorch: {e}")
    except Exception as e:
        print(f"   {method} solution failed: {e}")

# Load external solution (this should work)
try:
    print("\n3. Loading external solution...")
    ext_solution = ks_comparison.solve("external", verbose=True)
    print("   External solution loaded!")
except Exception as e:
    print(f"   External solution failed: {e}")

# Run simulations for solved methods
print("\n\nRunning simulations...")
simulation_results = ks_comparison.simulate_policy(
    periods=1000, num_agents=10000, burn_in=100, seed=42
)

# Compute metrics for all solutions
print("\nComputing solution metrics...")
metrics = ks_comparison.compute_metrics()

# Generate comparison report
print("\nGenerating comparison report...")
comparison_df = ks_comparison.compare_solutions(
    save_report=True, report_path="krusell_smith_comparison_report.md"
)

# Display results
print("\nComparison Results:")
print("=" * 50)

if len(comparison_df) > 0:
    print(comparison_df.to_string())
else:
    print("No solutions available for comparison.")
    print("(This is expected without full implementations)")

# Show what the comparison framework provides
print("\n\nKey features demonstrated:")
print("1. Unified parameter specification")
print("2. Automatic parameter translation for different methods")
print("3. Standardized solution interface through adapters")
print("4. Common metrics computation (Euler errors, wealth distribution, etc.)")
print("5. Integrated simulation capabilities")
print("6. Automated comparison reports")

print("\n\nTo fully utilize this framework:")
print("- Install required dependencies (PyTorch for deep learning)")
print("- Implement or link actual solution methods")
print("- Provide real solution data for external adapter")
print("- Run on a machine with sufficient computational resources")

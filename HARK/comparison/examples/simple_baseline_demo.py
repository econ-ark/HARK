"""
Simple demonstration of baseline functionality for model comparison.

This script shows how to save and load baseline solutions without relying on
complex solution adapters - it uses mock solutions to demonstrate the workflow.
"""

import numpy as np
from HARK.comparison import ModelComparison
import tempfile
import os


def create_mock_solution(accuracy_level="high"):
    """Create a mock solution for demonstration purposes."""
    if accuracy_level == "high":
        return {
            "policy_function": f"high_accuracy_policy_{np.random.randint(1000)}",
            "value_function": f"high_accuracy_value_{np.random.randint(1000)}",
            "solve_time": 120.5,
            "grid_points": 48,
            "accuracy": "high",
        }
    else:
        return {
            "policy_function": f"fast_policy_{np.random.randint(1000)}",
            "value_function": f"fast_value_{np.random.randint(1000)}",
            "solve_time": 45.2,
            "grid_points": 24,
            "accuracy": "medium",
        }


def mock_metrics(solution_type="benchmark"):
    """Generate mock metrics."""
    base_gini = 0.4 if solution_type == "benchmark" else 0.4 + np.random.normal(0, 0.02)
    base_wealth = (
        3.5 if solution_type == "benchmark" else 3.5 + np.random.normal(0, 0.1)
    )

    return {
        "gini": max(0, min(1, base_gini)),
        "mean_wealth": max(0, base_wealth),
        "welfare_equivalent": 0.85 + np.random.normal(0, 0.01),
    }


def main():
    print("=" * 60)
    print("BASELINE FUNCTIONALITY DEMONSTRATION")
    print("=" * 60)

    # Define model primitives
    primitives = {
        "DiscFac": 0.96,
        "CRRA": 2.0,
        "Rfree": 1.03,
        "TranShkStd": 0.2,
        "PermShkStd": 0.1,
    }

    # Create comparison object
    comparison = ModelComparison(
        model_description_md="Demonstration model for baseline functionality",
        primitives=primitives,
    )

    print("\nStep 1: Creating mock benchmark solution...")

    # Mock a high-accuracy benchmark solution
    benchmark_solution = create_mock_solution("high")
    benchmark_metrics = mock_metrics("benchmark")

    method_name = "benchmark_method"
    comparison.solutions[method_name] = benchmark_solution
    comparison.metrics[method_name] = benchmark_metrics
    comparison.computation_times[method_name] = benchmark_solution["solve_time"]
    comparison.method_configs[method_name] = {
        "grid_points": 48,
        "accuracy": "high",
        "description": "High accuracy benchmark",
    }

    print(
        f"✓ Benchmark solution created (solve time: {benchmark_solution['solve_time']:.1f}s)"
    )

    # Save the baseline
    print("\nStep 2: Saving baseline solution...")

    with tempfile.TemporaryDirectory() as temp_dir:
        baseline_path = os.path.join(temp_dir, "demo_baseline.pkl")

        saved_path = comparison.save_baseline_solution(method_name, baseline_path)
        print(f"✓ Baseline saved to: {os.path.basename(saved_path)}")

        # Clear current solutions to simulate fresh session
        print("\nStep 3: Simulating fresh session...")
        comparison.solutions.clear()
        comparison.metrics.clear()
        comparison.computation_times.clear()

        # Load the baseline
        print("Loading baseline solution...")
        loaded_data = comparison.load_baseline_solution(
            baseline_path, "loaded_benchmark"
        )
        print("✓ Baseline loaded successfully!")

        # Show baseline info
        print("\nStep 4: Baseline information:")
        baselines_df = comparison.list_baselines()
        print(baselines_df.to_string(index=False))

        # Create and test some fast methods
        print("\nStep 5: Testing fast methods against baseline...")

        fast_methods = {
            "fast_method_1": {"grid_points": 24, "accuracy": "medium"},
            "fast_method_2": {"grid_points": 16, "accuracy": "low"},
        }

        for method_name, config in fast_methods.items():
            # Create mock solution
            solution = create_mock_solution("fast")
            metrics = mock_metrics("fast")

            # Add to comparison
            comparison.solutions[method_name] = solution
            comparison.metrics[method_name] = metrics
            comparison.computation_times[method_name] = solution["solve_time"]
            comparison.method_configs[method_name] = config

            print(f"✓ {method_name} solved (solve time: {solution['solve_time']:.1f}s)")

        # Compare against baseline
        print("\nStep 6: Comparing methods against baseline...")
        comparison_df = comparison.compare_against_baseline("loaded_benchmark")

        if not comparison_df.empty:
            print("\nComparison Results:")
            print("-" * 50)

            # Display key metrics
            display_cols = [
                "method",
                "computation_time",
                "time_ratio",
                "gini",
                "gini_rel_diff",
                "mean_wealth",
                "mean_wealth_rel_diff",
            ]

            available_cols = [
                col for col in display_cols if col in comparison_df.columns
            ]
            if available_cols:
                print(comparison_df[available_cols].round(4).to_string(index=False))

            print("\nSummary:")
            print("-" * 20)
            for _, row in comparison_df.iterrows():
                method = row["method"]
                time_ratio = row.get("time_ratio", "N/A")
                gini_diff = row.get("gini_rel_diff", "N/A")

                if time_ratio != "N/A":
                    print(f"{method}: {time_ratio:.2f}x faster than baseline")
                    if (
                        gini_diff != "N/A"
                        and gini_diff is not None
                        and isinstance(gini_diff, (int, float))
                    ):
                        print(
                            f"  -> Gini coefficient differs by {gini_diff:.3f} ({gini_diff * 100:.1f}%)"
                        )

        # Demonstrate multiple baselines
        print("\nStep 7: Testing multiple baselines...")

        # Save one of the fast methods as another baseline
        comparison.save_baseline_solution(
            "fast_method_1", os.path.join(temp_dir, "fast_baseline.pkl")
        )

        comparison.load_baseline_solution(
            os.path.join(temp_dir, "fast_baseline.pkl"), "fast_baseline"
        )

        print("All loaded baselines:")
        print(
            comparison.list_baselines()[
                ["baseline_name", "original_method", "computation_time"]
            ].to_string(index=False)
        )

        # Clean up
        print("\nStep 8: Cleanup...")
        comparison.clear_baselines()
        print("✓ Baselines cleared from memory")

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 60)

    print("""
    KEY BENEFITS DEMONSTRATED:

    1. EFFICIENCY: Solved expensive benchmark once, reused multiple times
    2. CONSISTENCY: All comparisons used exactly the same baseline
    3. PERSISTENCE: Baselines saved to disk for later use
    4. FLEXIBILITY: Multiple baselines can be loaded and managed
    5. WORKFLOW: Clean separation between benchmark solving and testing

    TYPICAL USAGE PATTERN:

    # Session 1: Create and save benchmark
    comparison.solve('expensive_benchmark_method')
    comparison.save_baseline_solution('expensive_benchmark_method', 'benchmark.pkl')

    # Session 2: Load benchmark and test methods
    comparison.load_baseline_solution('benchmark.pkl', 'benchmark')
    comparison.solve('fast_method_1')
    comparison.solve('fast_method_2')
    results = comparison.compare_against_baseline('benchmark')
    """)


if __name__ == "__main__":
    main()

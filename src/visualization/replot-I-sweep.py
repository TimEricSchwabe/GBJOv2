import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.visualization.plot_optimization_results import plot_optimization_steps_sweep

sweep_run_dir = "...optimization_results/steps_sweep/run_20251230_114235"

output_dir = os.path.join(sweep_run_dir, "plots_custom")

metrics = ["mean", "median"]


methods_subset = [
    "GBJO", 
    "Genetic Search", 
    "Iterative Improvement",
    "Neural Sort",
    "CMA"
]


exclude_sizes = [] 

# ---------------------

def main():
    if not os.path.exists(sweep_run_dir):
        print(f"Error: Run directory not found: {sweep_run_dir}")
        return

    print(f"Plotting from: {sweep_run_dir}")
    print(f"Metrics: {metrics}")
    print(f"Methods: {methods_subset if methods_subset else 'All'}")
    print(f"Excluded sizes: {exclude_sizes if exclude_sizes else 'None'}")

    for metric in metrics:
        print(f"\n--- Generating plots for metric: {metric} ---")
        plot_optimization_steps_sweep(
            root_sweep_dir=sweep_run_dir,
            output_dir=output_dir,
            methods_to_plot=methods_subset,
            exclude_query_sizes=exclude_sizes,
            metric=metric
        )

if __name__ == "__main__":
    main()

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

# Configuration
FILE1 = "optimization_results/wikidata-path/detailed_results.json"  # k=1 data
FILE2 = "optimization_results/wikidata-path-k-5/detailed_results.json"  # k=5 data

FILE1 = "optimization_results/lubm-path/detailed_results.json"  # k=1 data
FILE2 = "optimization_results/lubm-path-k-5/detailed_results.json"  # k=5 data


def load_and_extract_costs(file_path):
    """Load results and extract costs by query size."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    gradient_costs = {}
    greedy_costs = {}
    
    for query_result in data:
        if 'plans' not in query_result or 'ntriplepattern' not in query_result:
            continue
            
        size = query_result['ntriplepattern']
        
        # Extract gradient costs
        if 'gradient' in query_result['plans']:
            pred_cost = query_result['plans']['gradient'].get('predicted_cost')
            if pred_cost is not None and pred_cost != float('inf'):
                if size not in gradient_costs:
                    gradient_costs[size] = []
                gradient_costs[size].append(pred_cost)
        
        # Extract greedy costs
        if 'greedy' in query_result['plans']:
            pred_cost = query_result['plans']['greedy'].get('predicted_cost')
            if pred_cost is not None and pred_cost != float('inf'):
                if size not in greedy_costs:
                    greedy_costs[size] = []
                greedy_costs[size].append(pred_cost)
    
    return gradient_costs, greedy_costs

def calculate_medians(costs_by_size):
    """Calculate median costs for each query size."""
    sizes = sorted(costs_by_size.keys())
    medians = [np.median(costs_by_size[size]) for size in sizes]
    return sizes, medians

# Load data from both files
gradient_k1, greedy = load_and_extract_costs(FILE1)
gradient_k5, _ = load_and_extract_costs(FILE2)

# Calculate medians
greedy_sizes, greedy_medians = calculate_medians(greedy)
k1_sizes, k1_medians = calculate_medians(gradient_k1)
k5_sizes, k5_medians = calculate_medians(gradient_k5)

# Create lineplot
plt.figure()
plt.plot(greedy_sizes, greedy_medians, 's-', label='Greedy', markersize=3)
plt.plot(k1_sizes, k1_medians, 'o-', label='Gradient ($k$=1)', linestyle='--', markersize=3)
plt.plot(k5_sizes, k5_medians, 'o-', label='Gradient ($k$=5)', linestyle='--', markersize=3)

plt.xlabel('Query Size')
plt.ylabel('Median Predicted Cost')
plt.yscale('log')
plt.legend(frameon=True, loc='upper left', framealpha=0.)
plt.margins(x=0.05, y=0.05)
plt.tight_layout()

# Save plots
plt.savefig('lineplot_k_comparison.pdf', bbox_inches='tight')
plt.show()

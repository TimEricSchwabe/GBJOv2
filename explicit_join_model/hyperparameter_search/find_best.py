#!/usr/bin/env python3
"""
Script to find the best hyperparameter configurations from CSV results.
Finds top 10 configs with lowest mean cost where failure rate is below threshold.
"""

import pandas as pd

# Load CSV file
csv_file = "/home/tim/query_optimization/hyperparam_search_results/stars-2-to-14-tp/results.csv"
df = pd.read_csv(csv_file)

# Filter where failure rate <= 0.2 and find top 10 with lowest cost
filtered = df[df['failure_rate'] <= 0.2]
best = filtered.nsmallest(10, 'mean_cost')

# Display results
print(f"Top {len(best)} configs (lowest mean_cost, failure_rate <= 0.2):")
for i, (_, row) in enumerate(best.iterrows(), 1):
    print(f"\n{i}. Cost: {row['mean_cost']:.4f}, Failure: {row['failure_rate']:.4f}")
    config_cols = [col for col in row.index if col.startswith('config/')]
    for col in config_cols:
        print(f"   {col.replace('config/', '')}: {row[col]}")

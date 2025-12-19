#!/usr/bin/env python3
"""
Plot optimization results from saved JSON data using Pandas.
Refactored for conciseness and clarity.
"""

import json
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import Dict, Any

# Try to use scienceplots if available
try:
    import scienceplots
    plt.style.use('science')
except ImportError:
    # Fallback to a clean style
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        plt.style.use('seaborn-v0_8-whitegrid')

# Map JSON keys to Display Names
METHODS_MAP = {
    'exhaustive': 'Exhaustive',
    'greedy': 'Greedy',
    'gradient': 'Gradient',
    'dp': 'DP',
    'random': 'Random',
    'II': 'Iterative Improvement',
    'GEQO': 'Genetic Search',
    'NeuralSort': 'Neural Sort',
    'CMA': 'CMA'
}

# Define the list of methods to plot (keys from METHODS_MAP) determines Order and Selection
METHODS_TO_PLOT = [
    'exhaustive',
    'dp',
    'gradient',
    'II',
    'greedy',
    'GEQO',
    'random',
    'NeuralSort',
    'CMA'
]

# Define consistent styles to match original plots exactly
METHOD_STYLES = {
    'Gradient': {'color': 'blue', 'marker': 'o', 'markeredgecolor': 'white', 'markersize': 5},
    'Greedy': {'color': 'green', 'marker': 's', 'markeredgecolor': 'white', 'markersize': 5},
    'DP': {'color': 'purple', 'marker': 'd', 'markeredgecolor': 'white', 'markersize': 5},
    'Random': {'color': 'red', 'marker': '^', 'markeredgecolor': 'white', 'markersize': 5},
    'Exhaustive': {'color': 'orange', 'marker': 'x', 'markeredgecolor': 'white', 'markersize': 5},
    'Iterative Improvement': {'color': 'brown', 'marker': 'P', 'markeredgecolor': 'white', 'markersize': 6},
    'Genetic Search': {'color': 'cyan', 'marker': '*', 'markeredgecolor': 'black', 'markersize': 8},
    'Neural Sort': {'color': 'yellow', 'marker': 'v', 'markeredgecolor': 'white', 'markersize': 5},
    'CMA': {'color': 'pink', 'marker': 'h', 'markeredgecolor': 'white', 'markersize': 5}
}

def load_data(results_dir: str) -> pd.DataFrame:
    """Load and restructure JSON results into a Pandas DataFrame."""
    results_file = os.path.join(results_dir, "detailed_results.json")
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file, 'r') as f:
        raw_data = json.load(f)

    print(f"Loaded {len(raw_data)} queries from {results_file}")

    rows = []
    for q in raw_data:
        if 'plans' not in q:
            continue
        
        row = {'query_size': q.get('ntriplepattern', 0)}
        plans = q['plans']
        
        # Extract costs for each method
        for method_key, method_name in METHODS_MAP.items():
            if method_key in plans:
                p = plans[method_key]
                # Real cost
                r_cost = p.get('real_cost')
                if r_cost is None or r_cost == float('inf'): 
                    r_cost = np.nan
                row[f'{method_name}_real'] = r_cost
                
                # Predicted cost
                p_cost = p.get('predicted_cost')
                if p_cost is None or p_cost == float('inf'): 
                    p_cost = np.nan
                row[f'{method_name}_pred'] = p_cost
            else:
                row[f'{method_name}_real'] = np.nan
                row[f'{method_name}_pred'] = np.nan
        
        rows.append(row)

    return pd.DataFrame(rows)

def plot_overall_boxplot(df: pd.DataFrame, output_dir: str):
    """Overall Box plot of costs (log scale)"""
    # Filter for columns with data
    cols = [c for c in df.columns if ('_real' in c or '_pred' in c) and df[c].notna().any()]
    
    # Create a rank map for sorting based on METHODS_TO_PLOT
    method_ranks = {METHODS_MAP[k]: i for i, k in enumerate(METHODS_TO_PLOT) if k in METHODS_MAP}

    # Sort columns: Method order, then Real vs Pred
    def sort_key(col):
        method = col.split('_')[0]
        type_ = col.split('_')[1]
        # Default to 99 if method not in rank map
        order = method_ranks
        type_order = {'real': 0, 'pred': 1}
        return (order.get(method, 99), type_order.get(type_, 99))
    
    cols = sorted(cols, key=sort_key)
    
    if not cols:
        print("No data for boxplot.")
        return

    plt.figure(figsize=(14, 8))
    # Using Pandas boxplot wrapper
    df[cols].boxplot(rot=45)
    plt.yscale('log')
    plt.title('Overall Cost Distribution (Log Scale)')
    plt.ylabel('Cost')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_boxplot.png'))
    plt.close()

def plot_mean_costs_bar(df: pd.DataFrame, output_dir: str):
    """Mean costs bar plot (true and predicted)"""
    means = df[[c for c in df.columns if '_real' in c or '_pred' in c]].mean()
    if means.empty:
        print("No data for mean barplot.")
        return
        
    methods = sorted(list(set([c.split('_')[0] for c in means.index])))
    
    # Sort methods by METHODS_TO_PLOT order
    method_ranks = {METHODS_MAP[k]: i for i, k in enumerate(METHODS_TO_PLOT) if k in METHODS_MAP}
    methods.sort(key=lambda m: method_ranks.get(m, 99))

    real_means = [means.get(f'{m}_real', np.nan) for m in methods]
    pred_means = [means.get(f'{m}_pred', np.nan) for m in methods]
    
    # Filter out methods where both are NaN to clean up plot
    valid_indices = [i for i, (r, p) in enumerate(zip(real_means, pred_means)) if not (np.isnan(r) and np.isnan(p))]
    if not valid_indices:
        return
        
    methods = [methods[i] for i in valid_indices]
    real_means = [real_means[i] for i in valid_indices]
    pred_means = [pred_means[i] for i in valid_indices]
    
    x = np.arange(len(methods))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    
    # Use consistent colors if possible
    real_colors = [METHOD_STYLES.get(m, {}).get('color', None) for m in methods]
    pred_colors = [METHOD_STYLES.get(m, {}).get('color', None) for m in methods] # Use same color for pred but maybe lighter? 
    # For now, stick to standard bar plot logic (grouping Real vs Pred), but maybe matching the method colors is better?
    # Original plot: Single bar per method? No, original had comparison. 
    # Let's stick to simple comparison: Blue for Real, Orange for Pred? Or stick to user request "mean costs bar plot".
    # I'll stick to a simple grouped bar plot.
    
    plt.bar(x - width/2, real_means, width, label='True Cost', alpha=0.8)
    plt.bar(x + width/2, pred_means, width, label='Predicted Cost', alpha=0.8)
    
    plt.ylabel('Mean Cost')
    plt.title('Mean Costs by Method')
    plt.xticks(x, methods)
    plt.legend()
    plt.yscale('log') 
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_costs_barplot.png'))
    plt.close()

def plot_lineplots_by_size(df: pd.DataFrame, output_dir: str):
    """True and predicted lineplots per query size (mean cost)"""
    if 'query_size' not in df.columns:
        return

    grouped = df.groupby('query_size').median()
    
    # 1. True Costs
    plt.figure(figsize=(10, 6))
    real_cols = [c for c in grouped.columns if '_real' in c]
    has_data = False
    
    # Sort columns to ensure consistent legend order
    method_ranks = {METHODS_MAP[k]: i for i, k in enumerate(METHODS_TO_PLOT) if k in METHODS_MAP}
    real_cols.sort(key=lambda c: method_ranks.get(c.replace('_real', ''), 99))
    
    for col in real_cols:
        if grouped[col].notna().any():
            method = col.replace('_real', '')
            style = METHOD_STYLES.get(method, {})
            
            plt.plot(grouped.index, grouped[col], 
                     label=method,
                     color=style.get('color'),
                     marker=style.get('marker'),
                     markeredgecolor=style.get('markeredgecolor'),
                     markersize=style.get('markersize'),
                     linestyle='-') # Solid line for true costs
            has_data = True
            
    if has_data:
        plt.xlabel('Query Size')
        plt.ylabel('Median True Cost')
        plt.yscale('log')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.margins(x=0.05, y=0.05)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lineplot_true_costs.pdf'))
        plt.savefig(os.path.join(output_dir, 'lineplot_true_costs.png'))
    plt.close()

    # 2. Predicted Costs
    plt.figure(figsize=(10, 6))
    pred_cols = [c for c in grouped.columns if '_pred' in c]
    has_data = False
    
    # Sort columns
    method_ranks = {METHODS_MAP[k]: i for i, k in enumerate(METHODS_TO_PLOT) if k in METHODS_MAP}
    pred_cols.sort(key=lambda c: method_ranks.get(c.replace('_pred', ''), 99))
    
    for col in pred_cols:
        if grouped[col].notna().any():
            method = col.replace('_pred', '')
            style = METHOD_STYLES.get(method, {})
            
            plt.plot(grouped.index, grouped[col], 
                     label=method,
                     color=style.get('color'),
                     marker=style.get('marker'),
                     markeredgecolor=style.get('markeredgecolor'),
                     markersize=style.get('markersize'),
                     linestyle='--') # Dashed line for predicted costs
            has_data = True
            
    if has_data:
        plt.xlabel('Query Size')
        plt.ylabel('Median Predicted Cost')
        plt.yscale('log')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.margins(x=0.05, y=0.05)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lineplot_predicted_costs.png'))
    plt.close()

def plot_boxplot_per_size(df: pd.DataFrame, output_dir: str):
    """Boxplot per query size (true cost)"""
    real_cols = [c for c in df.columns if '_real' in c]
    if not real_cols:
        return
        
    melted = df.melt(id_vars=['query_size'], value_vars=real_cols, var_name='Method', value_name='Cost')
    melted['Method'] = melted['Method'].str.replace('_real', '')
    melted = melted.dropna(subset=['Cost'])
    
    if melted.empty:
        return

    plt.figure(figsize=(12, 8))
    
    try:
        import seaborn as sns
        # Create a palette dictionary
        palette = {m: METHOD_STYLES.get(m, {}).get('color') for m in melted['Method'].unique()}
        # Remove None values if any method is missing from style
        palette = {k: v for k, v in palette.items() if v}
        
        sns.boxplot(data=melted, x='query_size', y='Cost', hue='Method', palette=palette)
    except ImportError:
        # Fallback
        melted.boxplot(column='Cost', by=['query_size', 'Method'], rot=45)
        
    plt.yscale('log')
    plt.title('Real Cost Distribution by Query Size')
    plt.ylabel('Real Cost')
    plt.xlabel('Query Size')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplot_per_size_true.png'))
    plt.close()

def plot_scatter_correlations(df: pd.DataFrame, output_dir: str):
    """Scatter plots: Gradient-Greedy, Gradient-DP, DP-Greedy (True and Predicted)"""
    pairs = [
        ('Gradient', 'Greedy'),
        ('Gradient', 'DP'),
        ('DP', 'Greedy')
    ]
    types = {'True': '_real', 'Predicted': '_pred'}
    
    for label, suffix in types.items():
        for m1, m2 in pairs:
            col1 = f'{m1}{suffix}'
            col2 = f'{m2}{suffix}'
            
            if col1 not in df.columns or col2 not in df.columns:
                continue
                
            data = df[[col1, col2]].dropna()
            if data.empty:
                continue
                
            plt.figure(figsize=(8, 8))
            
            # Use specific colors for scatters if desired, but pairs have two methods.
            # Original used blue for Grad-Greedy, Orange for Grad-DP.
            color = 'blue'
            if 'DP' in (m1, m2):
                color = 'orange'
            
            plt.scatter(data[col1], data[col2], alpha=0.6, edgecolors='k', s=50, c=color)
            
            # Diagonal line (x=y)
            if data[col1].min() > 0 and data[col2].min() > 0:
                min_val = min(data[col1].min(), data[col2].min()) * 0.9
                max_val = max(data[col1].max(), data[col2].max()) * 1.1
                plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='x=y')
                plt.xlim(min_val, max_val)
                plt.ylim(min_val, max_val)
            
            plt.xlabel(f'{m1} Cost')
            plt.ylabel(f'{m2} Cost')
            plt.title(f'{label}: {m1} vs {m2}')
            plt.yscale('log')
            plt.xscale('log')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'scatter_{label.lower()}_{m1}_{m2}.png'))
            plt.close()

def plot_win_loss_heatmap(df: pd.DataFrame, output_dir: str):
    """Win/Loss Heatmap comparing algorithms pairwise."""
    methods = [METHODS_MAP[m] for m in METHODS_TO_PLOT if m in METHODS_MAP]
    # Filter methods that have data in the dataframe
    methods = [m for m in methods if f'{m}_real' in df.columns and df[f'{m}_real'].notna().any()]
    
    if not methods:
        print("No data for win/loss heatmap.")
        return

    # Initialize win matrix with NaN (diagonal will stay NaN, off-diagonal will be filled)
    win_matrix = pd.DataFrame(np.nan, index=methods, columns=methods)
    
    for m1 in methods:
        for m2 in methods:
            if m1 == m2:
                continue
            
            c1 = df[f'{m1}_real']
            c2 = df[f'{m2}_real']
            
            # Mask for where both have valid costs
            mask = c1.notna() & c2.notna()
            v1 = c1[mask]
            v2 = c2[mask]
            
            if v1.empty:
                continue
                
            # Algorithm A is better than B if cost(A) < cost(B)
            # and they are NOT within 1% range: abs(cost(A) - cost(B)) / max(cost(A), cost(B)) >= 0.01
            rel_diff = (v2 - v1).abs() / np.maximum(v1, v2)
            m1_wins = ((v1 < v2) & (rel_diff >= 0.01)).sum()
            m2_wins = ((v2 < v1) & (rel_diff >= 0.01)).sum()
            
            if m1_wins + m2_wins > 0:
                win_matrix.loc[m1, m2] = m1_wins / (m1_wins + m2_wins)
            else:
                win_matrix.loc[m1, m2] = 0.5 # Equal if all queries are ties

    plt.figure(figsize=(12, 10))
    
    try:
        import seaborn as sns
        sns.heatmap(win_matrix, annot=True, fmt='.2f', cmap='YlGnBu', cbar_kws={'label': 'Win Rate (Fraction of non-tie queries)'})
    except ImportError:
        print("Seaborn not found, using matplotlib fallback")
        # Fallback to matplotlib imshow
        im = plt.imshow(win_matrix.values, cmap='YlGnBu')
        plt.colorbar(im, label='Win Rate (Fraction of non-tie queries)')
        # Add labels
        for i in range(len(methods)):
            for j in range(len(methods)):
                plt.text(j, i, f"{win_matrix.iloc[i, j]:.2f}", ha="center", va="center", color="black")
        plt.xticks(np.arange(len(methods)), methods, rotation=45)
        plt.yticks(np.arange(len(methods)), methods)

    plt.xlabel('Algorithm B (Loser)')
    plt.ylabel('Algorithm A (Winner)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'win_loss_heatmap.png'))
    plt.close()

def plot_optimality_gap(df: pd.DataFrame, output_dir: str):
    """Optimality gap plot: mean ratio to best cost per query, grouped by query size."""
    if 'query_size' not in df.columns:
        print("No query_size column for optimality gap plot.")
        return
    
    methods = [METHODS_MAP[m] for m in METHODS_TO_PLOT if m in METHODS_MAP]
    # Filter methods that have data
    real_cols = [f'{m}_real' for m in methods if f'{m}_real' in df.columns and df[f'{m}_real'].notna().any()]
    methods = [col.replace('_real', '') for col in real_cols]
    
    if not methods:
        print("No data for optimality gap plot.")
        return
    
    # Compute the best (minimum) cost per query across all methods
    df_costs = df[real_cols].copy()
    best_cost = df_costs.min(axis=1)
    
    # Compute ratio to best for each method (optimality gap)
    gap_cols = []
    for m in methods:
        col = f'{m}_real'
        gap_col = f'{m}_gap'
        df[gap_col] = df[col] / best_cost
        gap_cols.append(gap_col)
    
    # Group by query size and compute mean optimality gap
    grouped = df.groupby('query_size')[gap_cols].mean()
    
    # Sort methods by METHODS_TO_PLOT order for consistent legend
    method_ranks = {METHODS_MAP[k]: i for i, k in enumerate(METHODS_TO_PLOT) if k in METHODS_MAP}
    methods_sorted = sorted(methods, key=lambda m: method_ranks.get(m, 99))
    
    plt.figure(figsize=(10, 6))
    has_data = False
    
    for m in methods_sorted:
        gap_col = f'{m}_gap'
        if gap_col in grouped.columns and grouped[gap_col].notna().any():
            style = METHOD_STYLES.get(m, {})
            plt.plot(grouped.index, grouped[gap_col],
                     label=m,
                     color=style.get('color'),
                     marker=style.get('marker'),
                     markeredgecolor=style.get('markeredgecolor'),
                     markersize=style.get('markersize'),
                     linestyle='-')
            has_data = True
    
    if has_data:
        plt.xlabel('Number of Triple Patterns')
        plt.ylabel('Mean Optimality Gap (Ratio to Best Cost)')
        plt.yscale('log')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Optimal (1.0)')
        plt.legend()
        plt.margins(x=0.05, y=0.05)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'optimality_gap.pdf'))
        plt.savefig(os.path.join(output_dir, 'optimality_gap.png'))
    plt.close()
    
    # Clean up temporary columns
    for gap_col in gap_cols:
        if gap_col in df.columns:
            df.drop(columns=[gap_col], inplace=True)

def plot_performance_profile(df: pd.DataFrame, output_dir: str):
    """Dolan-Moré performance profile plot.
    
    x-axis: performance ratio τ (cost / best_cost)
    y-axis: fraction of queries where method achieves ratio ≤ τ
    """
    methods = [METHODS_MAP[m] for m in METHODS_TO_PLOT if m in METHODS_MAP]
    # Filter methods that have data
    real_cols = [f'{m}_real' for m in methods if f'{m}_real' in df.columns and df[f'{m}_real'].notna().any()]
    methods = [col.replace('_real', '') for col in real_cols]
    
    if not methods:
        print("No data for performance profile.")
        return
    
    # Compute the best (minimum) cost per query across all methods
    df_costs = df[real_cols].copy()
    best_cost = df_costs.min(axis=1)
    
    # Compute performance ratio for each method (cost / best_cost)
    ratios = {}
    for m in methods:
        col = f'{m}_real'
        ratio = df[col] / best_cost
        # Drop NaN values for this method
        ratios[m] = ratio.dropna().values
    
    # Sort methods by METHODS_TO_PLOT order for consistent legend
    method_ranks = {METHODS_MAP[k]: i for i, k in enumerate(METHODS_TO_PLOT) if k in METHODS_MAP}
    methods_sorted = sorted(methods, key=lambda m: method_ranks.get(m, 99))
    
    plt.figure(figsize=(10, 6))
    
    # Determine x-axis range (from 1.0 to max ratio across all methods)
    all_ratios = np.concatenate([ratios[m] for m in methods if len(ratios[m]) > 0])
    max_ratio = min(np.percentile(all_ratios, 99), 100)  # Cap at 99th percentile or 100
    
    # Create x values for plotting (performance ratio τ)
    x_vals = np.linspace(1.0, max_ratio, 500)
    
    for m in methods_sorted:
        if len(ratios[m]) == 0:
            continue
        
        style = METHOD_STYLES.get(m, {})
        n_queries = len(ratios[m])
        
        # Compute CDF: fraction of queries where ratio <= τ
        y_vals = np.array([np.sum(ratios[m] <= tau) / n_queries for tau in x_vals])
        
        plt.plot(x_vals, y_vals,
                 label=m,
                 color=style.get('color'),
                 linestyle='-',
                 linewidth=2)
    
    plt.xlabel('Performance Ratio τ (Cost / Best Cost)')
    plt.ylabel('Fraction of Queries with Ratio ≤ τ')
    plt.title('Dolan-Moré Performance Profile')
    plt.xlim(1.0, max_ratio)
    plt.ylim(0, 1.05)
    plt.xscale('log')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_profile.pdf'))
    plt.savefig(os.path.join(output_dir, 'performance_profile.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot optimization results.")
    parser.add_argument("results_dir", nargs='?', default="optimization_results/run_20251219_110756", 
                        help="Directory containing detailed_results.json")
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = os.path.join(results_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading data from {results_dir}...")
    try:
        df = load_data(results_dir)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if df.empty:
        print("No data found.")
        return

    print("Generating plots...")
    plot_overall_boxplot(df, output_dir)
    plot_mean_costs_bar(df, output_dir)
    plot_lineplots_by_size(df, output_dir)
    plot_boxplot_per_size(df, output_dir)
    plot_scatter_correlations(df, output_dir)
    plot_win_loss_heatmap(df, output_dir)
    plot_optimality_gap(df, output_dir)
    plot_performance_profile(df, output_dir)
    
    print(f"Done. Plots saved to {output_dir}")

if __name__ == "__main__":
    main()

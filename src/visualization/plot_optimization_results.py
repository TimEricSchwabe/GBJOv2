#!/usr/bin/env python3

import json
import os
import argparse
from tkinter import FALSE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from typing import Dict, Any
from matplotlib.ticker import LogLocator, ScalarFormatter, FuncFormatter, NullLocator

SINGLE_COL_WIDTH = 3.25  # inches (IJCAI single column)
DOUBLE_COL_WIDTH = 6.75  # inches (IJCAI double column)
FIGSIZE_SINGLE = (SINGLE_COL_WIDTH, 2.4)
FIGSIZE_DOUBLE = (DOUBLE_COL_WIDTH, 2.4)
FIGSIZE_SQUARE = (SINGLE_COL_WIDTH, SINGLE_COL_WIDTH)

# Save kwargs for high-quality PDF output
SAVE_KWARGS = {'dpi': 300, 'bbox_inches': 'tight', 'pad_inches': 0.02}

def setup_paper_style():
    plt.rcParams.update({
        # Text rendering - try LaTeX first, fallback to mathtext
        'text.usetex': False,  # Set True if LaTeX is installed
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        
        # Font sizes appropriate for print at column width
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'legend.fontsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        
        # Line and marker sizes for visibility in print
        'lines.linewidth': 1.2,
        'lines.markersize': 4,
        'axes.linewidth': 0.6,
        
        # Grid styling (subtle)
        'grid.linewidth': 0.3,
        'grid.alpha': 0.4,
        
        # Legend styling - compact for paper figures
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        'legend.borderpad': 0.2,
        'legend.handlelength': 1.0,
        'legend.labelspacing': 0.2,
        'legend.columnspacing': 0.5,
        'legend.handletextpad': 0.3,
        
        # Figure settings
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        
        # Axes
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.axisbelow': True,
    })

# Apply paper style on import
setup_paper_style()

# Map JSON keys to Display Names
METHODS_MAP = {
    'exhaustive': 'Exhaustive',
    'greedy': 'Greedy',
    'gradient': 'GBJO',
    'dp': 'DP',
    'random': 'Random',
    'II': 'Iterative Improvement',
    'GEQO': 'Genetic Search',
    'NeuralSort': 'Neural Sort',
    'CMA': 'CMA',
    'GBJO_CG': 'GBJO_CG',
    'GBJO-Meta': 'GBJO-Meta'
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
    'CMA',
    'GBJO_CG',
    'GBJO-Meta'
]


METHOD_STYLES = {
    'Exhaustive': {'color': '#E69F00', 'marker': 'o', 'linestyle': '-'},       # Orange
    'DP': {'color': '#56B4E9', 'marker': 's', 'linestyle': '--'},              # Sky Blue  
    'GBJO': {'color': '#0072B2', 'marker': '^', 'linestyle': '-'},         # Bluish Green
    'Iterative Improvement': {'color': '#F0E442', 'marker': 'D', 'linestyle': ':'},  # Yellow
    'Greedy': {'color': '#009E73', 'marker': 'v', 'linestyle': '-.'},          # Blue
    'Genetic Search': {'color': '#D55E00', 'marker': 'P', 'linestyle': '-'},   # Vermillion
    'Random': {'color': '#CC79A7', 'marker': 'X', 'linestyle': '--'},          # Reddish Purple
    'Neural Sort': {'color': '#666666', 'marker': 'h', 'linestyle': ':'},      # Dark Gray
    'CMA': {'color': '#000000', 'marker': '*', 'linestyle': '-.'},             # Black
    'GBJO-Meta': {'color': '#FF69B4', 'marker': '*', 'linestyle': '-.'},       # Pink
    'GBJO_CG': {'color': '#9E69FF', 'marker': 'p', 'linestyle': '--'},          # Purple
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

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    # Using Pandas boxplot wrapper
    # Remove outliers (showfliers=False) and return dict to access artists
    bp = df[cols].boxplot(rot=45, ax=ax, showfliers=False, return_type='dict')
    ax.set_yscale('log')
    ax.set_ylabel('Cost')

    # Add method names vertically above each box
    for i, col in enumerate(cols):
        # Determine method name
        parts = col.split('_')
        method = parts[0]
        
        # Upper whisker for box i is at index 2*i + 1
        if 2 * i + 1 < len(bp['whiskers']):
            top_whisker = bp['whiskers'][2 * i + 1]
            y_data = top_whisker.get_ydata()
            
            if len(y_data) > 0:
                y_max = np.max(y_data)
                
                # Use method color
                style = METHOD_STYLES.get(method, {})
                color = style.get('color', 'black')
                
                # Place text above the whisker
                # Multiply by 1.15 for a small gap in log scale
                ax.text(i + 1, y_max * 1.15, method, 
                        rotation=90, ha='center', va='bottom', 
                        fontsize=6, color=color, clip_on=False)

    fig.savefig(os.path.join(output_dir, 'overall_boxplot.pdf'), **SAVE_KWARGS)
    plt.close(fig)

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
    
    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    
    # Use colorblind-friendly colors for True vs Predicted
    ax.bar(x - width/2, real_means, width, label='True Cost', color='#0072B2', alpha=0.85)
    ax.bar(x + width/2, pred_means, width, label='Predicted Cost', color='#E69F00', alpha=0.85)
    
    ax.set_ylabel('Mean Cost')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend(loc='best', fontsize=6, frameon=True, handlelength=0.8)
    ax.set_yscale('log')
    
    fig.savefig(os.path.join(output_dir, 'mean_costs_barplot.pdf'), **SAVE_KWARGS)
    plt.close(fig)

def plot_lineplots_by_size(df: pd.DataFrame, output_dir: str, metric: str = "median"):
    """True and predicted lineplots per query size (mean cost)"""
    if 'query_size' not in df.columns:
        return

    grouped = df.groupby('query_size').agg(metric)
    
    # Sort columns to ensure consistent legend order
    method_ranks = {METHODS_MAP[k]: i for i, k in enumerate(METHODS_TO_PLOT) if k in METHODS_MAP}
    
    # 1. True Costs
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    real_cols = [c for c in grouped.columns if '_real' in c]
    has_data = False
    
    real_cols.sort(key=lambda c: method_ranks.get(c.replace('_real', ''), 99))
    
    for col in real_cols:
        if grouped[col].notna().any():
            method = col.replace('_real', '')
            style = METHOD_STYLES.get(method, {})
            
            ax.plot(grouped.index, grouped[col], 
                    label=method,
                    color=style.get('color'),
                    marker=style.get('marker'),
                    linestyle=style.get('linestyle', '-'),
                    markeredgecolor='white',
                    markeredgewidth=0.3)
            has_data = True
            
    if has_data:
        ax.set_xlabel('Query Size (Triple Patterns)')
        #ax.set_ylabel('Median True Cost')
        ax.set_yscale('log')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # Compact legend outside plot or in best location
        ax.legend(loc='best', ncol=3, fontsize=5, frameon=True, 
                  handlelength=0.8, columnspacing=0.4)
        ax.margins(x=0.05, y=0.1)
        ax.grid(False)
        fig.savefig(os.path.join(output_dir, f'lineplot_true_costs_{metric}.pdf'), **SAVE_KWARGS)
    plt.close(fig)

    # 2. Predicted Costs
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    pred_cols = [c for c in grouped.columns if '_pred' in c]
    has_data = False
    
    pred_cols.sort(key=lambda c: method_ranks.get(c.replace('_pred', ''), 99))
    
    for col in pred_cols:
        if grouped[col].notna().any():
            method = col.replace('_pred', '')
            style = METHOD_STYLES.get(method, {})
            
            ax.plot(grouped.index, grouped[col], 
                    label=method,
                    color=style.get('color'),
                    marker=style.get('marker'),
                    linestyle=style.get('linestyle', '-'),
                    markeredgecolor='white',
                    markeredgewidth=0.3)
            has_data = True
            
    if has_data:
        ax.set_xlabel('Query Size (Triple Patterns)')
        metric_label = metric.capitalize()
        ax.set_ylabel(f'{metric_label} Predicted Cost')
        ax.set_yscale('log')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc='best', ncol=3, fontsize=5, frameon=True,
                  handlelength=0.8, columnspacing=0.4)
        ax.margins(x=0.05, y=0.1)
        fig.savefig(os.path.join(output_dir, f'lineplot_predicted_costs_{metric}.pdf'), **SAVE_KWARGS)
    plt.close(fig)

def plot_lineplots_by_size_geomean(df: pd.DataFrame, output_dir: str):
    """True and predicted lineplots per query size (geometric mean cost)."""
    if 'query_size' not in df.columns:
        return

    def geometric_mean(s: pd.Series) -> float:
        s = s.dropna()
        s = s[s > 0]
        if s.empty:
            return np.nan
        return float(np.exp(np.mean(np.log(s.values))))

    # Aggregate all cost columns by query_size using geometric mean
    cost_cols = [c for c in df.columns if c.endswith('_real') or c.endswith('_pred')]
    if not cost_cols:
        return

    grouped = df.groupby('query_size')[cost_cols].agg(geometric_mean)

    # Sort columns to ensure consistent legend order
    method_ranks = {METHODS_MAP[k]: i for i, k in enumerate(METHODS_TO_PLOT) if k in METHODS_MAP}

    # 1. True Costs
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    real_cols = [c for c in grouped.columns if '_real' in c]
    has_data = False

    real_cols.sort(key=lambda c: method_ranks.get(c.replace('_real', ''), 99))

    for col in real_cols:
        if grouped[col].notna().any():
            method = col.replace('_real', '')
            style = METHOD_STYLES.get(method, {})

            ax.plot(grouped.index, grouped[col],
                    label=method,
                    color=style.get('color'),
                    marker=style.get('marker'),
                    linestyle=style.get('linestyle', '-'),
                    markeredgecolor='white',
                    markeredgewidth=0.3)
            has_data = True

    if has_data:
        ax.set_xlabel('Query Size (Triple Patterns)')
        ax.set_ylabel('Geometric Mean True Cost')
        ax.set_yscale('log')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc='best', ncol=3, fontsize=5, frameon=True,
                  handlelength=0.8, columnspacing=0.4)
        ax.margins(x=0.05, y=0.1)
        fig.savefig(os.path.join(output_dir, 'lineplot_true_costs_geomean.pdf'), **SAVE_KWARGS)
    plt.close(fig)

    # 2. Predicted Costs
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    pred_cols = [c for c in grouped.columns if '_pred' in c]
    has_data = False

    pred_cols.sort(key=lambda c: method_ranks.get(c.replace('_pred', ''), 99))

    for col in pred_cols:
        if grouped[col].notna().any():
            method = col.replace('_pred', '')
            style = METHOD_STYLES.get(method, {})

            ax.plot(grouped.index, grouped[col],
                    label=method,
                    color=style.get('color'),
                    marker=style.get('marker'),
                    linestyle=style.get('linestyle', '-'),
                    markeredgecolor='white',
                    markeredgewidth=0.3)
            has_data = True

    if has_data:
        ax.set_xlabel('Query Size (Triple Patterns)')
        ax.set_ylabel('Geometric Mean Predicted Cost')
        ax.set_yscale('log')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc='best', ncol=3, fontsize=5, frameon=True,
                  handlelength=0.8, columnspacing=0.4)
        ax.margins(x=0.05, y=0.1)
        fig.savefig(os.path.join(output_dir, 'lineplot_predicted_costs_geomean.pdf'), **SAVE_KWARGS)
    plt.close(fig)

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

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    
    try:
        import seaborn as sns

        method_ranks = {METHODS_MAP[k]: i for i, k in enumerate(METHODS_TO_PLOT) if k in METHODS_MAP}
        unique_methods = melted['Method'].unique()
        hue_order = sorted(unique_methods, key=lambda m: method_ranks.get(m, 99))
        
        palette = {m: METHOD_STYLES.get(m, {}).get('color') for m in hue_order}
        missing = [m for m, c in palette.items() if c is None]
        if missing:
            fallback_colors = sns.color_palette("colorblind", n_colors=len(missing))
            for m, c in zip(missing, fallback_colors):
                palette[m] = c
        
        sns.boxplot(data=melted, x='query_size', y='Cost', hue='Method', 
                    palette=palette, hue_order=hue_order, ax=ax,
                    linewidth=0.6, showfliers=False)
        
        # Add labels vertically above whiskers
        unique_sizes = sorted(melted['query_size'].unique())
        num_hues = len(hue_order)
        width = 0.8 
        hue_width = width / num_hues
        
        for i, size in enumerate(unique_sizes):
            for j, method in enumerate(hue_order):
                subset = melted[(melted['query_size'] == size) & (melted['Method'] == method)]
                if subset.empty:
                    continue
                
                costs = subset['Cost']
                if costs.empty:
                    continue
                    
                q1 = costs.quantile(0.25)
                q3 = costs.quantile(0.75)
                iqr = q3 - q1
                upper_lim = q3 + 1.5 * iqr
                whisker_val = costs[costs <= upper_lim].max()
                if pd.isna(whisker_val):
                     whisker_val = costs.max()
                
                # Calculate x position
                # x-tick is at i
                x_pos = i + (j - num_hues/2 + 0.5) * hue_width
                
                style = METHOD_STYLES.get(method, {})
                color = style.get('color', 'black')
                
                ax.text(x_pos, whisker_val * 1.25, method,
                        rotation=90, ha='center', va='bottom',
                        fontsize=5, color=color)

        ax.legend(loc='best', ncol=3, fontsize=5, frameon=True,
                  handlelength=0.8, columnspacing=0.4)
    except ImportError:
        melted.boxplot(column='Cost', by=['query_size', 'Method'], rot=45, ax=ax, showfliers=False)
        
    ax.set_yscale('log')
    ax.set_ylabel('True Cost')
    ax.set_xlabel('Query Size (Triple Patterns)')
    
    fig.savefig(os.path.join(output_dir, 'boxplot_per_size_true.pdf'), **SAVE_KWARGS)
    plt.close(fig)

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
                
            fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
            
            # Use colorblind-friendly colors
            color = '#0072B2'  # Blue
            if 'DP' in (m1, m2):
                color = '#D55E00'  # Vermillion
            
            ax.scatter(data[col1], data[col2], alpha=0.6, edgecolors='k', 
                       s=15, c=color, linewidths=0.3)
            
            # Diagonal line (x=y)
            if data[col1].min() > 0 and data[col2].min() > 0:
                min_val = min(data[col1].min(), data[col2].min()) * 0.9
                max_val = max(data[col1].max(), data[col2].max()) * 1.1
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, 
                        linewidth=0.8, label='$x=y$')
                ax.set_xlim(min_val, max_val)
                ax.set_ylim(min_val, max_val)
            
            ax.set_xlabel(f'{m1} Cost')
            ax.set_ylabel(f'{m2} Cost')
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.legend(loc='lower right', fontsize=5, frameon=True, handlelength=0.8)
            
            fig.savefig(os.path.join(output_dir, f'scatter_{label.lower()}_{m1}_{m2}.pdf'), **SAVE_KWARGS)
            plt.close(fig)

def plot_win_loss_heatmap(df: pd.DataFrame, output_dir: str):
    """Win/Loss Heatmap comparing algorithms pairwise."""
    methods = [METHODS_MAP[m] for m in METHODS_TO_PLOT if m in METHODS_MAP]
    # Filter methods that have data in the dataframe
    methods = [m for m in methods if f'{m}_real' in df.columns and df[f'{m}_real'].notna().any()]
    
    if not methods:
        print("No data for win/loss heatmap.")
        return

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
                win_matrix.loc[m1, m2] = 0.5  # Equal if all queries are ties

    # Determine figure size based on number of methods
    n_methods = len(methods)
    fig_width = min(DOUBLE_COL_WIDTH, SINGLE_COL_WIDTH + n_methods * 0.3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_width * 0.85))
    
    try:
        import seaborn as sns
        sns.heatmap(win_matrix, annot=True, fmt='.2f', cmap='YlGnBu', 
                    cbar_kws={'label': 'Win Rate', 'shrink': 0.8},
                    ax=ax, annot_kws={'size': 6}, linewidths=0.5)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    except ImportError:
        print("Seaborn not found, using matplotlib fallback")
        im = ax.imshow(win_matrix.values, cmap='YlGnBu')
        plt.colorbar(im, ax=ax, label='Win Rate', shrink=0.8)
        for i in range(len(methods)):
            for j in range(len(methods)):
                ax.text(j, i, f"{win_matrix.iloc[i, j]:.2f}", ha="center", va="center", 
                        color="black", fontsize=6)
        ax.set_xticks(np.arange(len(methods)))
        ax.set_yticks(np.arange(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_yticklabels(methods)

    ax.set_xlabel('Algorithm B')
    ax.set_ylabel('Algorithm A')
    
    fig.savefig(os.path.join(output_dir, 'win_loss_heatmap.pdf'), **SAVE_KWARGS)
    plt.close(fig)

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
    
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    has_data = False
    
    for m in methods_sorted:
        gap_col = f'{m}_gap'
        if gap_col in grouped.columns and grouped[gap_col].notna().any():
            style = METHOD_STYLES.get(m, {})
            ax.plot(grouped.index, grouped[gap_col],
                    label=m,
                    color=style.get('color'),
                    marker=style.get('marker'),
                    linestyle=style.get('linestyle', '-'),
                    markeredgecolor='white',
                    markeredgewidth=0.3)
            has_data = True
    
    if has_data:
        ax.set_xlabel('Number of Triple Patterns')
        ax.set_ylabel('Mean Optimality Gap')
        ax.set_yscale('log')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=0.8, label='Optimal')
        ax.legend(loc='best', ncol=3, fontsize=5, frameon=True,
                  handlelength=0.8, columnspacing=0.4)
        ax.margins(x=0.05, y=0.1)
        fig.savefig(os.path.join(output_dir, 'optimality_gap.pdf'), **SAVE_KWARGS)
    plt.close(fig)
    
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
    
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    
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
        auc = np.trapz(y_vals, x_vals)
        print(f"AUC for {m}: {auc:.4f} (max ratio capped at {max_ratio:.2f})")
        
        ax.plot(x_vals, y_vals,
                label=m,
                color=style.get('color'),
                linestyle=style.get('linestyle', '-'),
                linewidth=1.2)
    
    ax.set_xlabel(r'Performance Ratio $\tau$')
    #ax.set_ylabel(r'$P(\mathrm{ratio} \leq \tau)$')
    ax.set_xlim(1.0, max_ratio)
    ax.set_ylim(0, 1.02)
    ax.set_xscale('log')
    custom_ticks = [1, 2, 5, 10, 20, 50, 100] 
    # Filter them to only show what fits in your current plot range
    ticks_to_show = [t for t in custom_ticks if t <= max_ratio]
    
    ax.set_xticks(ticks_to_show)
    
    # 2. Use a lambda to force integer display (this removes the .0)
    from matplotlib.ticker import FuncFormatter, NullFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
    
    # 3. This is the key to stopping the crowding: hide all minor ticks/labels
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis='x', rotation=0) 
    # Place legend outside the plot area to avoid overlap
    ax.legend(loc='lower right', ncol=3, fontsize=5, frameon=True,
              handlelength=0.8, columnspacing=0.4, bbox_to_anchor=(0.98, 0.02))
    ax.grid(False)
    fig.savefig(os.path.join(output_dir, 'performance_profile.pdf'), **SAVE_KWARGS)
    plt.close(fig)


def plot_optimization_steps_sweep(root_sweep_dir: str, output_dir: str, 
                                  methods_to_plot: list = None,
                                  exclude_query_sizes: list = None,
                                  metric: str = "median"):
    """
    Plot metric (median/mean/geomean) true and predicted cost vs optimization_steps 
    by walking through subdirectories (steps_*) in root_sweep_dir.
    
    Args:
        root_sweep_dir: Directory containing steps_X/detailed_results.json folders
        output_dir: Where to save plots
        methods_to_plot: List of method names to include (None = all)
        exclude_query_sizes: List of query sizes (integers) to exclude
        metric: "median", "mean", or "geomean"
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Map friendly names to internal keys
    PLANKEY_TO_DISPLAY = {
        "gradient": "GBJO",
        "GEQO": "Genetic Search",
        "II": "Iterative Improvement",
        "NeuralSort": "Neural Sort",
        "CMA": "CMA",
        "random": "Random",
    }
    
    # 1. Discover step directories
    step_dirs = {}
    try:
        for entry in os.listdir(root_sweep_dir):
            if entry.startswith("steps_") and os.path.isdir(os.path.join(root_sweep_dir, entry)):
                try:
                    step_val = int(entry.split("_")[1])
                    step_dirs[step_val] = os.path.join(root_sweep_dir, entry)
                except ValueError:
                    continue
    except FileNotFoundError:
        print(f"Sweep directory not found: {root_sweep_dir}")
        return

    if not step_dirs:
        print(f"No steps_* directories found in {root_sweep_dir}")
        return
        
    sorted_steps = sorted(step_dirs.keys())
    
    # 2. Collect data for each step
    # structure: {cost_type: {method_display_name: {step_val: aggregated_cost}}}
    aggregated_data = {"real": {}, "pred": {}} 
    
    for step in sorted_steps:
        res_file = os.path.join(step_dirs[step], "detailed_results.json")
        if not os.path.exists(res_file):
            continue
            
        with open(res_file, 'r') as f:
            queries = json.load(f)
            
        # Extract costs per method
        method_costs_real = {} # {display_name: [list of costs]}
        method_costs_pred = {} # {display_name: [list of costs]}
        
        for q in queries:
            q_size = q.get('ntriplepattern', 0)
            if exclude_query_sizes and q_size in exclude_query_sizes:
                continue
                
            plans = q.get('plans', {})
            for key, plan_data in plans.items():
                display_name = PLANKEY_TO_DISPLAY.get(key, key)
                
                # Real cost
                r_cost = plan_data.get('real_cost')
                if r_cost is not None and r_cost != float('inf') and np.isfinite(r_cost):
                    if display_name not in method_costs_real:
                        method_costs_real[display_name] = []
                    method_costs_real[display_name].append(float(r_cost))
                
                # Predicted cost
                p_cost = plan_data.get('predicted_cost')
                if p_cost is not None and p_cost != float('inf') and np.isfinite(p_cost):
                    if display_name not in method_costs_pred:
                        method_costs_pred[display_name] = []
                    method_costs_pred[display_name].append(float(p_cost))
        
        # Helper to aggregate
        def get_agg(costs, metric_name):
            if not costs: return np.nan
            if metric_name == "median":
                return np.median(costs)
            elif metric_name == "mean":
                return np.mean(costs)
            elif metric_name == "geomean":
                pos_costs = [c for c in costs if c > 0]
                return np.exp(np.mean(np.log(pos_costs))) if pos_costs else np.nan
            return np.nan

        for m_name, costs in method_costs_real.items():
            val = get_agg(costs, metric)
            if not np.isnan(val):
                if m_name not in aggregated_data["real"]:
                    aggregated_data["real"][m_name] = {}
                aggregated_data["real"][m_name][step] = val

        for m_name, costs in method_costs_pred.items():
            val = get_agg(costs, metric)
            if not np.isnan(val):
                if m_name not in aggregated_data["pred"]:
                    aggregated_data["pred"][m_name] = {}
                aggregated_data["pred"][m_name][step] = val

    # 3. Plotting
    for cost_type in ["real", "pred"]:
        type_data = aggregated_data[cost_type]
        if not type_data:
            continue
            
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        
        # Filter methods if requested
        available_methods = list(type_data.keys())
        if methods_to_plot:
            plot_methods = [m for m in available_methods if m in methods_to_plot]
        else:
            plot_methods = available_methods

        # Sort by preferred order
        preferred_order = ["GBJO", "Iterative Improvement", "Genetic Search", "Neural Sort", "CMA"]
        plot_methods.sort(key=lambda m: preferred_order.index(m) if m in preferred_order else 999)
        
        has_any = False
        for method in plot_methods:
            data_points = type_data[method]
            xs = []
            ys = []
            for s in sorted_steps:
                if s in data_points:
                    xs.append(s)
                    ys.append(data_points[s])
            
            if not xs:
                continue
                
            style = METHOD_STYLES.get(method, {})
            ax.plot(xs, ys, 
                    label=method,
                    color=style.get("color"),
                    marker=style.get("marker", "o"),
                    linestyle=style.get("linestyle", "-"),
                    markeredgecolor="white",
                    markeredgewidth=0.3)
            has_any = True
            
        if not has_any:
            plt.close(fig)
            continue

        ax.set_xlabel("Optimization Steps")
        metric_label = metric.capitalize()
        if metric == "geomean": metric_label = "Geometric Mean"
        
        type_label = "True" if cost_type == "real" else "Predicted"
        ax.set_ylabel(f"{metric_label} {type_label} Cost")
        ax.set_yscale("log")
        
        # X-axis formatting
        try:
            ax.set_xscale("log")
            ax.set_xticks(sorted_steps)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
            ax.xaxis.set_minor_locator(NullLocator())
        except Exception:
            ax.set_xticks(sorted_steps)

        ax.legend(loc="best", ncol=2, fontsize=6, frameon=True, handlelength=0.9, columnspacing=0.6)
        ax.grid(False)

        suffix = "true" if cost_type == "real" else "predicted"
        out_filename = f"optimization_steps_sweep_{metric}_{suffix}.pdf"
        out_path = os.path.join(output_dir, out_filename)
        fig.savefig(out_path, **SAVE_KWARGS)
        plt.close(fig)
        print(f"Saved sweep plot to: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot optimization results.")
    parser.add_argument("results_dir", nargs='?', default="optimization_results/FINAL-LUBM-STAR", 
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
    plot_lineplots_by_size(df, output_dir, metric="mean")
    plot_lineplots_by_size_geomean(df, output_dir)
    plot_boxplot_per_size(df, output_dir)
    plot_scatter_correlations(df, output_dir)
    plot_win_loss_heatmap(df, output_dir)
    plot_optimality_gap(df, output_dir)
    plot_performance_profile(df, output_dir)
    
    print(f"Done. Plots saved to {output_dir}")

if __name__ == "__main__":
    main()

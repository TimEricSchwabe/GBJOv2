#!/usr/bin/env python3
"""
Query Analysis Visualization Script

This script analyzes query data from a JSON file and creates:
1. A bar plot showing the number of queries per star_size
2. Individual histograms for each query size showing the distribution of cardinalities

Usage:
    python query_analysis_plots.py /path/to/your/query_data.json
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter
import argparse

def load_query_data(file_path):
    """
    Load query data from JSON file.
    
    Args:
        file_path: Path to the JSON file containing query data
        
    Returns:
        List of query dictionaries
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} queries from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format: {e}")
        sys.exit(1)

def analyze_query_data(data):
    """
    Analyze the query data to extract star_size and cardinality information.
    
    Args:
        data: List of query dictionaries
        
    Returns:
        Tuple of (size_counts, cardinalities_by_size)
    """
    size_counts = Counter()
    cardinalities_by_size = defaultdict(list)
    
    for query in data:
        if 'star_size' not in query or 'y' not in query:
            print(f"Warning: Query missing 'star_size' or 'y' field: {query.get('query_hash', 'unknown')}")
            continue
            
        star_size = query['star_size']
        cardinality = query['y']
        
        size_counts[star_size] += 1
        cardinalities_by_size[star_size].append(cardinality)
    
    print(f"Found queries with star_sizes: {sorted(size_counts.keys())}")
    print(f"Query counts by size: {dict(sorted(size_counts.items()))}")
    
    return size_counts, cardinalities_by_size

def plot_queries_per_size(size_counts, output_dir="."):
    """
    Create a bar plot showing the number of queries per star_size.
    
    Args:
        size_counts: Counter object with star_size counts
        output_dir: Directory to save the plot
    """
    sizes = sorted(size_counts.keys())
    counts = [size_counts[size] for size in sizes]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(sizes, counts, color='steelblue', alpha=0.7, edgecolor='black')
    
    plt.xlabel('Query Size (Number of Triple Patterns)')
    plt.ylabel('Number of Queries')
    plt.title('Distribution of Queries by Star Size')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01, 
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'queries_per_size.png', dpi=300, bbox_inches='tight')
    print(f"Saved bar plot to: {Path(output_dir) / 'queries_per_size.png'}")
    plt.show()

def plot_cardinality_histograms(cardinalities_by_size, output_dir="."):
    """
    Create individual histograms for each query size showing cardinality distributions.
    
    Args:
        cardinalities_by_size: Dictionary mapping star_size to list of cardinalities
        output_dir: Directory to save the plots
    """
    sizes = sorted(cardinalities_by_size.keys())
    
    # Calculate grid layout
    n_sizes = len(sizes)
    n_cols = min(3, n_sizes)  # Max 3 columns
    n_rows = (n_sizes + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # Handle case where we have only one subplot
    if n_sizes == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, size in enumerate(sizes):
        cardinalities = cardinalities_by_size[size]
        
        ax = axes[i] if n_sizes > 1 else axes[0]
        
        # Create histogram
        n_bins = min(50, max(10, len(set(cardinalities))))  # Adaptive bin count
        ax.hist(cardinalities, bins=n_bins, color='lightcoral', alpha=0.7, 
                edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Cardinality')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Cardinality Distribution\n(Star Size = {size}, n = {len(cardinalities)})')
        ax.grid(alpha=0.3)
        
        # Use log scale for y-axis if we have a wide range of cardinalities
        if max(cardinalities) / min(cardinalities) > 1000:
            ax.set_xscale('log')
            ax.set_xlabel('Cardinality (log scale)')
        
        # Add statistics text
        mean_card = np.mean(cardinalities)
        median_card = np.median(cardinalities)
        stats_text = f'Mean: {mean_card:.1e}\nMedian: {median_card:.1e}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide empty subplots
    for i in range(n_sizes, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'cardinality_histograms.png', dpi=300, bbox_inches='tight')
    print(f"Saved histograms to: {Path(output_dir) / 'cardinality_histograms.png'}")
    plt.show()

def print_summary_statistics(cardinalities_by_size):
    """
    Print summary statistics for each query size.
    
    Args:
        cardinalities_by_size: Dictionary mapping star_size to list of cardinalities
    """
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for size in sorted(cardinalities_by_size.keys()):
        cardinalities = cardinalities_by_size[size]
        cardinalities_array = np.array(cardinalities)
        
        print(f"\nStar Size {size} ({len(cardinalities)} queries):")
        print(f"  Mean cardinality:   {np.mean(cardinalities_array):.2e}")
        print(f"  Median cardinality: {np.median(cardinalities_array):.2e}")
        print(f"  Min cardinality:    {np.min(cardinalities_array):,}")
        print(f"  Max cardinality:    {np.max(cardinalities_array):,}")
        print(f"  Std deviation:      {np.std(cardinalities_array):.2e}")
        
        # Percentiles
        p25, p75 = np.percentile(cardinalities_array, [25, 75])
        print(f"  25th percentile:    {p25:.2e}")
        print(f"  75th percentile:    {p75:.2e}")

def main():
    # Define input parameters
    json_file = "/home/tim/CQOS-dataset/lubm/star/Queries_9_to_14.json"  # Path to query data
    output_dir = "."  # Directory to save plots (default: current directory)
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and analyze data
    print("Loading query data...")
    data = load_query_data(json_file)
    print("\nAnalyzing data...")
    size_counts, cardinalities_by_size = analyze_query_data(data)
    
    if not size_counts:
        print("Error: No valid query data found!")
        sys.exit(1)
    
    # Create visualizations
    print("\nCreating bar plot of queries per size...")
    plot_queries_per_size(size_counts, output_dir)
    
    print("\nCreating cardinality histograms...")
    plot_cardinality_histograms(cardinalities_by_size, output_dir)
    
    # Print summary statistics
    print_summary_statistics(cardinalities_by_size)
    
    print(f"\nAll plots saved to: {output_dir}")

if __name__ == "__main__":
    main() 